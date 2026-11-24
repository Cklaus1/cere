// K12 device sampler (m27 design note I10, D2-D5, D7); contracts in k12_sampler.h. Each kernel follows the
// arithmetic of its `host/sampler_ref.cpp` counterpart: fp32 penalties without contraction, fp32 z = x / T, fp64
// log-sum-exp and masses, the D3 rank rules, the D4 counter and fp64 unnormalized scan.
//
// math_reference: sglang
#include "kernels/k12_sampler.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cere::k12 {
namespace {

constexpr int kThreads = 1024;  // single-block kernels: one block over the row
constexpr int kWarps = kThreads / 32;
constexpr int kBlock = 256;  // threads of a block of the vocab-wide kernels
constexpr int kGrid = 512;   // their blocks: a fixed count, so every reduction over the row has one fixed order
constexpr int kElementwise = 256;
constexpr int kCandidates = kThreads;  // ranks keep_mask resolves in one block (one per thread)
constexpr int kRadixBins = 2048;       // radix-select digits: bits 31-21, 20-10, 9-0 of the fp32 pattern
constexpr int kPasses = 3;
constexpr uint32_t kNone = 0xffffffffu;
constexpr uint32_t kInfBits = 0x7f800000u;  // +inf: every bisection predicate holds there
constexpr unsigned kFull = 0xffffffffu;
constexpr uint32_t kResolved = 0;  // Work::mode: resolve_kernel wrote the mask
constexpr uint32_t kMasked = 1;    // Work::mode: the cut can lie beyond kCandidates ranks; masked_kernel decides

static_assert(kChunks == kThreads, "the draw's chunk prefix runs one chunk per thread of a kThreads block");
static_assert(kRadixBins % kBlock == 0, "find_bin gives each thread whole digits");

__host__ __device__ constexpr int radix_shift(int pass) { return pass == 0 ? 21 : pass == 1 ? 10 : 0; }
__host__ __device__ constexpr int radix_width(int pass) { return pass == 2 ? 10 : 11; }
static_assert(radix_shift(0) == radix_shift(1) + radix_width(1) && radix_shift(1) == radix_width(2) &&
                  radix_shift(0) + radix_width(0) == 32,
              "the passes' digits tile the 32-bit pattern");

// ---- D4 counter: murmur_hash32 (sampler_ref.cpp, SGLang murmur_hash.py:20-48,92-101) ----

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

__device__ __forceinline__ uint32_t mix(uint32_t h, uint32_t k) {
    k *= 0xCC9E2D51u;
    k = rotl32(k, 15);
    k *= 0x1B873593u;
    h ^= k;
    h = rotl32(h, 13);
    return h * 5u + 0xE6546B64u;
}

__device__ uint32_t murmur_hash32(uint64_t seed, uint32_t key, uint32_t c) {
    uint32_t h = mix(0, uint32_t(seed));
    h = mix(h, uint32_t(seed >> 32));
    h = mix(h, key);
    h = mix(h, c);
    h ^= 16u;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}

// u = (x >> 11) * 2^-53 with x = h_ca * 2^32 + h_cb (exact in fp64).
__device__ double uniform(uint64_t seed, uint32_t key, uint32_t ca, uint32_t cb) {
    const uint64_t x = (uint64_t(murmur_hash32(seed, key, ca)) << 32) | murmur_hash32(seed, key, cb);
    return double(x >> 11) * 0x1.0p-53;
}

__device__ __forceinline__ bool kept(const uint32_t *mask, int64_t i) { return (mask[i >> 5] >> (i & 31)) & 1u; }

// ---- block reductions (one block of kN threads; every thread receives the result) ----

struct Add {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a + b; }
};
struct Max {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct Min {
    __device__ float operator()(float a, float b) const { return fminf(a, b); }
};
struct MinId {
    __device__ uint32_t operator()(uint32_t a, uint32_t b) const { return a < b ? a : b; }
};

// A fixed combination order (xor butterfly in each warp, then the warps in order, by thread 0), so a repeat is
// bit-identical, and every block of a grid reducing the same values gets the same result.
template <int kN, typename T, typename Op>
__device__ T block_reduce(T v, Op op) {
    constexpr int kW = kN / 32;
    __shared__ T part[kW];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(kFull, v, o));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        T acc = part[0];
        for (int w = 1; w < kW; ++w) acc = op(acc, part[w]);
        part[0] = acc;
    }
    __syncthreads();
    v = part[0];
    __syncthreads();  // part is reused by the next call
    return v;
}

// Ordered candidates: larger value first, then the lower id; kNone is worse than every candidate. A total order
// whatever the values: NaN ranks above +inf (`k1_primitives.cu` top2's order: a NaN logit is a model fault that
// greedy surfaces, never hides), NaNs among themselves by the lower id.
struct Cand {
    float v;
    uint32_t i;
};

__device__ __forceinline__ bool better(const Cand &a, const Cand &b) {
    if (a.i == kNone) return false;
    if (b.i == kNone) return true;
    const bool a_nan = a.v != a.v;
    const bool b_nan = b.v != b.v;
    if (a_nan || b_nan) return a_nan != b_nan ? a_nan : a.i < b.i;
    if (a.v != b.v) return a.v > b.v;
    return a.i < b.i;
}

__device__ Cand block_best(Cand c) {
    __shared__ Cand part[kWarps];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const Cand other{__shfl_xor_sync(kFull, c.v, o), __shfl_xor_sync(kFull, c.i, o)};
        if (better(other, c)) c = other;
    }
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = c;
    __syncthreads();
    c = part[0];
    for (int w = 1; w < kWarps; ++w)
        if (better(part[w], c)) c = part[w];
    __syncthreads();
    return c;
}

// ---- scratch (scratch_bytes(): probabilities and keep_mask use Work, a draw DrawScratch; nothing persists) ----

// Zeroed by keep_mask before its radix passes.
struct Counters {
    uint32_t hist[kPasses][kRadixBins];  // pass q: ids with p > 0 whose higher digits are the selected ones, by digit q
    uint32_t above;                      // ids collected into Work::cand
};

// The rank keep_mask resolves (select_digits).
struct Select {
    uint32_t positive;  // ids with p > 0
    uint32_t sel;       // the ranks resolve_kernel orders: min(top_k, positive, kCandidates)
    uint32_t bits;      // the selected digits; after kPasses passes the fp32 pattern t of rank sel - 1's probability
    uint32_t rank;      // rank sel - 1 among the ids whose patterns start with `bits`
};

struct Work {
    float max_part[kGrid];   // probabilities: per-block max of the mass_z of z
    double sum_part[kGrid];  // probabilities: per-block fp64 sum of exp(mass_z(z) - max)
    uint32_t nan_part[kGrid];  // probabilities: per-block lowest id whose z is NaN (kNone: none)
    Counters counters;
    Select select;               // written by collect_kernel's block 0
    uint32_t tie_count[kGrid];   // ids with p == t in each collect block's word span
    uint32_t cand[kCandidates];  // the ids with p > t (fewer than sel), in no particular order
    uint32_t mode;               // kResolved or kMasked
};

struct DrawScratch {
    double prefix[kChunks + 1];  // the fp64 kept mass before chunk c; prefix[kChunks] = S
    uint32_t last_kept;
};

bool scratch_ok(const void *scratch) {
    return scratch != nullptr && (reinterpret_cast<uintptr_t>(scratch) & 7u) == 0;
}

// ---- process (D2) ----

__global__ void process_kernel(const bf16 *__restrict__ logits, int64_t vocab, const uint32_t *__restrict__ counts,
                               float f, float p, float r, const uint32_t *__restrict__ mask, float *__restrict__ out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    float x = __bfloat162float(logits[i]);
    if (counts != nullptr) {
        const uint32_t c = counts[i];
        if (c > 0) {
            x = __fsub_rn(x, __fadd_rn(__fmul_rn(f, __uint2float_rn(c)), p));
            x = x < 0.0f ? __fmul_rn(x, r) : __fdiv_rn(x, r);
        }
    }
    if (mask != nullptr && !kept(mask, i)) x = -INFINITY;
    out[i] = x;
}

__global__ void bias_kernel(const uint32_t *__restrict__ ids, const float *__restrict__ values, int64_t stride,
                            int64_t count, float *__restrict__ out) {
    const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= count) return;
    const uint32_t id = ids[j * stride];
    out[id] = __fadd_rn(out[id], values[j * stride]);
}

// ---- top2_f32 ----

__device__ __forceinline__ void push(Cand &best, Cand &second, const Cand &c) {
    if (better(c, best)) {
        second = best;
        best = c;
    } else if (better(c, second)) {
        second = c;
    }
}

__global__ void top2_kernel(const float *__restrict__ values, int64_t vocab, uint32_t *ids, float *scores) {
    __shared__ Cand sb[kWarps], ss[kWarps];
    Cand best{-INFINITY, kNone}, second{-INFINITY, kNone};
    for (int64_t j = threadIdx.x; j < vocab; j += kThreads) push(best, second, Cand{values[j], uint32_t(j)});
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const Cand ob{__shfl_xor_sync(kFull, best.v, o), __shfl_xor_sync(kFull, best.i, o)};
        const Cand os{__shfl_xor_sync(kFull, second.v, o), __shfl_xor_sync(kFull, second.i, o)};
        push(best, second, ob);
        push(best, second, os);
    }
    if ((threadIdx.x & 31) == 0) {
        sb[threadIdx.x >> 5] = best;
        ss[threadIdx.x >> 5] = second;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int w = 1; w < kWarps; ++w) {
        push(best, second, sb[w]);
        push(best, second, ss[w]);
    }
    ids[0] = best.i;
    ids[1] = second.i;
    scores[0] = best.v;
    scores[1] = second.v;
}

// ---- probabilities and log-sum-exp ----

__device__ __forceinline__ float scaled(const float *x, int64_t i, float temperature) {
    return temperature > 0.0f ? __fdiv_rn(x[i], temperature) : x[i];
}

// What an entry of z contributes to the softmax: +inf saturates to the largest finite float (so the +inf entries
// split the mass and the finite ones lose it), NaN and -inf carry none (-inf). The identity for every finite z.
__device__ __forceinline__ float mass_z(float z) { return z != z ? -INFINITY : fminf(z, FLT_MAX); }

// The fp64 log-sum-exp of z (sampler_ref: m = max z, lse = m + log(sum exp(z - m))), one kThreads block; entries
// by mass_z; -inf when no entry carries mass.
__device__ double log_sum_exp(const float *x, int64_t vocab, float temperature) {
    float m = -INFINITY;
    for (int64_t j = threadIdx.x; j < vocab; j += kThreads) m = fmaxf(m, mass_z(scaled(x, j, temperature)));
    m = block_reduce<kThreads>(m, Max{});
    double sum = 0.0;
    for (int64_t j = threadIdx.x; j < vocab; j += kThreads) {
        const float z = mass_z(scaled(x, j, temperature));
        if (z > -INFINITY) sum += exp(double(z) - double(m));
    }
    sum = block_reduce<kThreads>(sum, Add{});
    return double(m) + log(sum);
}

constexpr int64_t kGridStride = int64_t(kGrid) * kBlock;

__device__ __forceinline__ int64_t grid_first() { return int64_t(blockIdx.x) * kBlock + threadIdx.x; }

// The kGrid per-block partials combined in one fixed order (thread t folds partials t, t + kBlock, ... in order,
// then block_reduce), so every block of the grid gets the same bits.
template <typename T, typename Op>
__device__ T reduce_parts(const T *part, T init, Op op) {
    T v = init;
    for (int b = threadIdx.x; b < kGrid; b += kBlock) v = op(v, part[b]);
    return block_reduce<kBlock>(v, op);
}

// probabilities, three kGrid x kBlock launches: the per-block max of mass_z(z), the per-block fp64 sum of
// exp(mass_z(z) - m) (m the combined max), then every block combines the sums (the same lse in every block) and
// writes its ids' probabilities. A row with no mass (every z NaN or -inf: m = -inf, lse = -inf) puts all of it on
// its lowest NaN id (id 0 when none), the greedy token of the row, so the mask and the draw always have an id.
__global__ void __launch_bounds__(kBlock) max_kernel(const float *__restrict__ x, int64_t vocab, float temperature,
                                                     Work *__restrict__ w) {
    float m = -INFINITY;
    uint32_t nan_id = kNone;
    for (int64_t j = grid_first(); j < vocab; j += kGridStride) {
        const float z = __fdiv_rn(x[j], temperature);
        m = fmaxf(m, mass_z(z));
        if (z != z && nan_id == kNone) nan_id = uint32_t(j);
    }
    m = block_reduce<kBlock>(m, Max{});
    nan_id = block_reduce<kBlock>(nan_id, MinId{});
    if (threadIdx.x == 0) {
        w->max_part[blockIdx.x] = m;
        w->nan_part[blockIdx.x] = nan_id;
    }
}

__global__ void __launch_bounds__(kBlock) sum_kernel(const float *__restrict__ x, int64_t vocab, float temperature,
                                                     Work *__restrict__ w) {
    const double m = reduce_parts(w->max_part, -INFINITY, Max{});
    double sum = 0.0;
    for (int64_t j = grid_first(); j < vocab; j += kGridStride) {
        const float z = mass_z(__fdiv_rn(x[j], temperature));
        if (z > -INFINITY) sum += exp(double(z) - m);
    }
    sum = block_reduce<kBlock>(sum, Add{});
    if (threadIdx.x == 0) w->sum_part[blockIdx.x] = sum;
}

__global__ void __launch_bounds__(kBlock) probs_kernel(const float *__restrict__ x, int64_t vocab, float temperature,
                                                       const Work *__restrict__ w, float *__restrict__ probs,
                                                       double *lse_out) {
    const float m = reduce_parts(w->max_part, -INFINITY, Max{});
    const double sum = reduce_parts(w->sum_part, 0.0, Add{});
    const double log_sum = log(sum);
    const double lse = double(m) + log_sum;
    // m + log(sum) rounds log(sum) away once m passes 2^53 (reachable only through a saturated +inf, where every tie
    // would then read p = 1): the shift is applied to z first there. Every row below that keeps exp(z - lse).
    const bool huge = m > 0x1p53f;
    if (m > -INFINITY) {
        for (int64_t j = grid_first(); j < vocab; j += kGridStride) {
            const float z = mass_z(__fdiv_rn(x[j], temperature));
            probs[j] = z > -INFINITY ? float(exp(huge ? (double(z) - double(m)) - log_sum : double(z) - lse)) : 0.0f;
        }
    } else {  // no mass anywhere (m is the same in every block)
        const uint32_t nan_id = reduce_parts(w->nan_part, kNone, MinId{});
        const uint32_t only = nan_id != kNone ? nan_id : 0u;
        for (int64_t j = grid_first(); j < vocab; j += kGridStride) probs[j] = j == only ? 1.0f : 0.0f;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *lse_out = lse;
}

// ---- keep mask (D3), candidate path ----
//
// The kept ids are a prefix of the ranking (p > 0 descending, ties to the lower id). With sel = min(top_k,
// #positive, kCandidates), the pattern t of rank sel - 1's probability comes from an exact radix select on the fp32
// patterns (positive floats order as their bit patterns): three passes over the row count the positive ids under
// each digit of the pattern, restricted to the digits selected so far. One pass then collects the ids with p > t and
// counts the ids with p == t per block; one block orders those sel ranks (the lowest-id ties fill the ranks after
// the ids above t) and runs sampler_ref::keep_mask's scan on them, the fp64 masses summed in rank order. The cut can
// lie beyond the sel ranks only when top_k > kCandidates (min_p == 0: the first kCandidates ranks hold at most top_p
// of the mass; min_p > 0: the renormalization needs all top_k ranks): the row is then kMasked and masked_kernel
// decides it with the whole-row bisection.

// The digit holding rank `rank` (0 = the largest) in the descending digit order of hist[0, kRadixBins): x = the
// digit, y = the ids in higher digits. One block of kBlock threads; rank < the sum of hist.
__device__ uint2 find_bin(const uint32_t *hist, uint32_t rank) {
    constexpr int kPer = kRadixBins / kBlock;
    __shared__ uint32_t scan[kBlock];
    __shared__ uint2 found;
    const int t = threadIdx.x;
    uint32_t own = 0;
    for (int b = t * kPer; b < (t + 1) * kPer; ++b) own += hist[b];
    scan[t] = own;
    __syncthreads();
    for (int off = 1; off < kBlock; off <<= 1) {  // inclusive suffix sums: the ids in thread t's digits and above
        const uint32_t add = t + off < kBlock ? scan[t + off] : 0u;
        __syncthreads();
        scan[t] += add;
        __syncthreads();
    }
    const uint32_t higher = scan[t] - own;
    if (higher <= rank && rank < scan[t]) {  // exactly one thread
        uint32_t acc = higher;
        for (int b = (t + 1) * kPer - 1; b >= t * kPer; --b) {
            const uint32_t n = hist[b];
            if (rank < acc + n) {
                found = make_uint2(uint32_t(b), acc);
                break;
            }
            acc += n;
        }
    }
    __syncthreads();
    const uint2 out = found;
    __syncthreads();  // scan and found are reused by the next call
    return out;
}

// The rank to resolve and the digits of its pattern the first `done` passes selected (one block of kBlock threads).
__device__ Select select_digits(const Counters &c, int64_t top_k, int done) {
    uint32_t own = 0;
    for (int b = threadIdx.x; b < kRadixBins; b += kBlock) own += c.hist[0][b];
    Select s{block_reduce<kBlock>(own, Add{}), 0, 0, 0};
    s.sel = uint32_t(min(min(top_k, int64_t(s.positive)), int64_t(kCandidates)));
    if (s.sel == 0) return s;
    s.rank = s.sel - 1;
    for (int q = 0; q < done; ++q) {
        const uint2 f = find_bin(c.hist[q], s.rank);
        s.rank -= f.y;
        s.bits = (s.bits << radix_width(q)) | f.x;
    }
    return s;
}

// Radix pass kPass: counts the ids with p > 0 whose pattern starts with the digits selected so far, by digit kPass.
template <int kPass>
__global__ void __launch_bounds__(kBlock) hist_kernel(const float *__restrict__ p, int64_t vocab, int64_t top_k,
                                                      Work *__restrict__ w) {
    constexpr int kShift = radix_shift(kPass);
    constexpr int kHigh = kShift + radix_width(kPass);
    constexpr uint32_t kDigit = (1u << radix_width(kPass)) - 1u;
    __shared__ uint32_t bins[kRadixBins];
    for (int b = threadIdx.x; b < kRadixBins; b += kBlock) bins[b] = 0;
    uint32_t prefix = 0;
    if constexpr (kPass > 0) {
        const Select s = select_digits(w->counters, top_k, kPass);
        if (s.sel == 0) return;  // no positive probability (the same in every thread)
        prefix = s.bits;
    }
    __syncthreads();
    for (int64_t j = grid_first(); j < vocab; j += kGridStride) {
        const float x = p[j];
        if (!(x > 0.0f)) continue;
        const uint32_t v = __float_as_uint(x);
        if constexpr (kPass > 0) {
            if ((v >> kHigh) != prefix) continue;
        }
        atomicAdd(&bins[(v >> kShift) & kDigit], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < kRadixBins; b += kBlock)
        if (bins[b] != 0) atomicAdd(&w->counters.hist[kPass][b], bins[b]);
}

// Block b owns the mask words [b * span, (b + 1) * span): it clears them, appends its ids with p > t to Work::cand
// and counts its ids with p == t.
__global__ void __launch_bounds__(kBlock) collect_kernel(const float *__restrict__ p, int64_t vocab, int64_t top_k,
                                                         int64_t span, Work *__restrict__ w,
                                                         uint32_t *__restrict__ mask) {
    const Select s = select_digits(w->counters, top_k, kPasses);
    if (blockIdx.x == 0 && threadIdx.x == 0) w->select = s;
    const int64_t words = (vocab + 31) / 32;
    const int64_t w0 = int64_t(blockIdx.x) * span;
    const int64_t w1 = min(words, w0 + span);
    for (int64_t i = w0 + threadIdx.x; i < w1; i += kBlock) mask[i] = 0;
    uint32_t ties = 0;
    if (s.sel > 0) {
        const int64_t i1 = min(vocab, w1 * 32);
        for (int64_t i = w0 * 32 + threadIdx.x; i < i1; i += kBlock) {
            const float x = p[i];
            if (!(x > 0.0f)) continue;
            const uint32_t v = __float_as_uint(x);
            if (v > s.bits) {
                w->cand[atomicAdd(&w->counters.above, 1u)] = uint32_t(i);  // above < sel <= kCandidates
            } else if (v == s.bits) {
                ++ties;
            }
        }
    }
    ties = block_reduce<kBlock>(ties, Add{});
    if (threadIdx.x == 0) w->tie_count[blockIdx.x] = ties;
}

// One block (kThreads): orders the sel ranks, runs sampler_ref::keep_mask's scan on them and sets the kept ids' mask
// bits, or marks the row kMasked.
__global__ void __launch_bounds__(kThreads) resolve_kernel(const float *__restrict__ p, int64_t vocab, int64_t top_k,
                                                           float top_p, float min_p, int64_t span, int collect_blocks,
                                                           Work *__restrict__ w, uint32_t *__restrict__ mask) {
    __shared__ float av[kCandidates];      // the collected ids' p (unordered)
    __shared__ uint32_t ai[kCandidates];   // the collected ids
    __shared__ float rv[kCandidates];      // rank r's p
    __shared__ uint32_t ri[kCandidates];   // rank r's id
    __shared__ double qd[kCandidates];     // min_p > 0: the renormalized masses q[r]
    __shared__ uint32_t base[kGrid];       // ids with p == t in the collect blocks before b
    __shared__ uint32_t warp_n[kWarps];
    __shared__ uint32_t s_n, s_mode;
    __shared__ double s_total, s_threshold;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const Select s = w->select;
    if (s.sel == 0) {  // no positive probability: the mask stays clear (sampler_ref keeps nothing)
        if (t == 0) w->mode = kResolved;
        return;
    }
    const uint32_t sel = s.sel;
    const uint32_t above = w->counters.above;  // < sel
    const uint32_t ties = sel - above;         // the `ties` lowest ids with p == t take ranks above .. sel - 1
    float v = 0.0f;
    uint32_t id = kNone;
    if (t < int(above)) {
        id = w->cand[t];
        v = p[id];
        av[t] = v;
        ai[t] = id;
    }
    // base: exclusive prefix of tie_count over the collect blocks (warp 0, lane l owns a run of blocks).
    if (t < 32) {
        constexpr int kRun = (kGrid + 31) / 32;
        const int b0 = lane * kRun, b1 = min(collect_blocks, b0 + kRun);
        uint32_t own = 0;
        for (int b = b0; b < b1; ++b) own += w->tie_count[b];
        uint32_t incl = own;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t up = __shfl_up_sync(kFull, incl, o);
            if (lane >= o) incl += up;
        }
        uint32_t acc = incl - own;
        for (int b = b0; b < b1; ++b) {
            base[b] = acc;
            acc += w->tie_count[b];
        }
    }
    __syncthreads();
    if (t < int(above)) {
        uint32_t r = 0;
        for (uint32_t j = 0; j < above; ++j) r += av[j] > v || (av[j] == v && ai[j] < id);
        rv[r] = v;
        ri[r] = id;
    }
    // The ties in id order: the blocks' spans in order, each scanned in kThreads chunks with a block-wide ballot.
    const float tv = __uint_as_float(s.bits);
    for (int b = 0; b < collect_blocks && base[b] < ties; ++b) {
        if (w->tie_count[b] == 0) continue;
        const int64_t i1 = min(vocab, (int64_t(b) + 1) * span * 32);
        uint32_t done = base[b];
        for (int64_t i = int64_t(b) * span * 32; i < i1 && done < ties; i += kThreads) {
            const int64_t j = i + t;
            const bool tie = j < i1 && __float_as_uint(p[j]) == s.bits;
            const unsigned ballot = __ballot_sync(kFull, tie);
            if (lane == 0) warp_n[t >> 5] = uint32_t(__popc(ballot));
            __syncthreads();
            uint32_t pos = done + uint32_t(__popc(ballot & ((1u << lane) - 1u)));
            uint32_t total = 0;
            for (int x = 0; x < kWarps; ++x) {
                if (x < (t >> 5)) pos += warp_n[x];
                total += warp_n[x];
            }
            if (tie && pos < ties) {
                rv[above + pos] = tv;
                ri[above + pos] = uint32_t(j);
            }
            __syncthreads();  // warp_n is rewritten by the next chunk
            done += total;
        }
    }
    __syncthreads();
    const uint32_t k = uint32_t(min(top_k, int64_t(s.positive)));  // sampler_ref's k
    const double cut = top_p;
    bool keep = false;
    if (min_p > 0.0f) {
        // top_k, renormalize, top_p, renormalize, q >= q[0] * min_p (renormalized): needs all k ranks.
        if (k > sel) {
            if (t == 0) w->mode = kMasked;
            return;
        }
        if (t == 0) {
            double total = 0.0;
            for (uint32_t r = 0; r < sel; ++r) total += double(rv[r]);
            s_total = total;
        }
        __syncthreads();
        if (t < int(sel)) qd[t] = double(rv[t]) / s_total;
        __syncthreads();
        if (t == 0) {
            double before = 0.0, kept_total = 0.0;
            uint32_t n = 0;
            for (uint32_t r = 0; r < sel; ++r) {
                if (before <= cut) {
                    kept_total += qd[r];
                    n = r + 1;
                }
                before += qd[r];
            }
            s_n = n;
            s_threshold = (qd[0] / kept_total) * double(min_p);
            s_total = kept_total;
        }
        __syncthreads();
        keep = t < int(s_n) && qd[t] / s_total >= s_threshold;
    } else {
        // The joint mask: rank < k and the fp64 mass ranked before <= top_p (monotone in the rank).
        if (t == 0) {
            double before = 0.0;
            uint32_t n = 0;
            bool resolved = k <= sel;
            for (uint32_t r = 0; r < sel; ++r) {
                if (!(before <= cut)) {
                    resolved = true;
                    break;
                }
                n = r + 1;
                before += double(rv[r]);
            }
            s_n = n;
            s_mode = resolved ? kResolved : kMasked;
        }
        __syncthreads();
        if (s_mode == kMasked) {
            if (t == 0) w->mode = kMasked;
            return;
        }
        keep = t < int(s_n);
    }
    if (keep) atomicOr(mask + (ri[t] >> 5), 1u << (ri[t] & 31));  // collect_kernel cleared every word
    if (t == 0) w->mode = kResolved;
}

// ---- keep mask (D3), whole-row path (a kMasked row) ----

struct Above {
    uint32_t count;  // ids with p > v
    double mass;     // their fp64 mass
};

__device__ Above above(const float *p, int64_t vocab, float v) {
    uint32_t count = 0;
    double mass = 0.0;
    for (int64_t j = threadIdx.x; j < vocab; j += kThreads) {
        const float x = p[j];
        if (x > v) {
            ++count;
            mass += double(x);
        }
    }
    return Above{block_reduce<kThreads>(count, Add{}), block_reduce<kThreads>(mass, Add{})};
}

// The least probability >= v (v > 0).
__device__ float least_from(const float *p, int64_t vocab, float v) {
    float t = INFINITY;
    for (int64_t j = threadIdx.x; j < vocab; j += kThreads) {
        const float x = p[j];
        if (x >= v) t = fminf(t, x);
    }
    return block_reduce<kThreads>(t, Min{});
}

// The lowest fp32 bit pattern in [0, +inf] where `holds` (monotone: false below, true from there on) is true.
template <typename Pred>
__device__ uint32_t lowest_holding(Pred holds) {
    uint32_t lo = 0, hi = kInfBits;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (holds(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

// Every id with p > t and the m lowest ids with p == t. For a probability value v, the first id with p == v has rank
// C(v) = #{p > v} and the mass ranked before it is S(v) (fp64), so "that id is kept" (rank < top_k, mass / denom <=
// top_p) is monotone in v and bisection finds the least such value t. denom is 1 (min_p 0: the joint mask on the
// unfiltered distribution) or the top-k mass (min_p > 0: top_k, renormalize, then top_p). min_p > 0 then keeps the
// prefix ids whose renormalized probability is >= p_max * min_p (renormalized). A thread owns whole mask words and
// writes each of them. One kThreads block, launched only for a row resolve_kernel marked kMasked.
__global__ void __launch_bounds__(kThreads) masked_kernel(const float *__restrict__ p, int64_t vocab, int64_t top_k,
                                                          float top_p, float min_p, const Work *__restrict__ w,
                                                          uint32_t *__restrict__ mask) {
    __shared__ uint32_t tie_base[kThreads];
    if (w->mode != kMasked) return;
    const int64_t words = (vocab + 31) / 32;
    const int64_t per = (words + kThreads - 1) / kThreads;
    const int64_t w0 = min(words, int64_t(threadIdx.x) * per);
    const int64_t w1 = min(words, w0 + per);
    const Above positive = above(p, vocab, 0.0f);
    const uint32_t k = uint32_t(min(top_k, int64_t(positive.count)));
    const double cut = top_p;
    double denom = 1.0;
    if (min_p > 0.0f) {
        // The k-th ranked value t_k (the least value with fewer than k ids above it) and the top-k mass.
        const uint32_t vk = lowest_holding([&](uint32_t b) { return above(p, vocab, __uint_as_float(b)).count < k; });
        const float tk = least_from(p, vocab, __uint_as_float(vk));
        const Above a = above(p, vocab, tk);
        denom = a.mass + double(k - a.count) * double(tk);
    }
    const uint32_t vp = lowest_holding([&](uint32_t b) {
        const Above a = above(p, vocab, __uint_as_float(b));
        return a.count < k && a.mass / denom <= cut;
    });
    const float t = least_from(p, vocab, __uint_as_float(vp));
    const Above a = above(p, vocab, t);
    uint32_t ties = 0;
    for (int64_t j = threadIdx.x; j < vocab; j += kThreads) ties += p[j] == t;
    ties = block_reduce<kThreads>(ties, Add{});
    // Tie j (in id order) has the mass a.mass + j * t ranked before it; j = 0 is kept (t was found so).
    const double td = t;
    const auto within = [&](uint32_t j) { return (a.mass + double(j) * td) / denom <= cut; };
    double guess = floor((cut * denom - a.mass) / td) + 1.0;
    guess = fmin(fmax(guess, 1.0), double(ties));
    uint32_t fit = uint32_t(guess);
    while (fit > 1 && !within(fit - 1)) --fit;
    while (fit < ties && within(fit)) ++fit;
    const uint32_t m = min(min(ties, k - a.count), fit);
    double kept_total = 1.0, threshold = 0.0;
    if (min_p > 0.0f) {
        float pmax = 0.0f;
        for (int64_t j = threadIdx.x; j < vocab; j += kThreads) pmax = fmaxf(pmax, p[j]);
        pmax = block_reduce<kThreads>(pmax, Max{});
        kept_total = (a.mass + double(m) * td) / denom;
        threshold = (double(pmax) / denom / kept_total) * double(min_p);
    }
    // The tie index of this thread's first id: ties in the words before it.
    uint32_t mine = 0;
    for (int64_t i = w0 * 32; i < min(vocab, w1 * 32); ++i) mine += p[i] == t;
    tie_base[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t acc = 0;
        for (int w = 0; w < kThreads; ++w) {
            const uint32_t n = tie_base[w];
            tie_base[w] = acc;
            acc += n;
        }
    }
    __syncthreads();
    uint32_t tie = tie_base[threadIdx.x];
    for (int64_t w = w0; w < w1; ++w) {
        uint32_t word = 0;
        for (int b = 0; b < 32; ++b) {
            const int64_t i = w * 32 + b;
            if (i >= vocab) break;
            const float x = p[i];
            bool keep = x > t;
            if (x == t) keep = tie++ < m;
            if (keep && min_p > 0.0f) keep = double(x) / denom / kept_total >= threshold;
            word |= uint32_t(keep) << b;
        }
        mask[w] = word;
    }
}

// ---- draw (D4) ----

__device__ __forceinline__ int64_t chunk_len(int64_t vocab) { return (vocab + kChunks - 1) / kChunks; }

// The fp64 kept mass of each chunk in id order, then prefix[c] = the chunks before c summed in chunk order (the
// sequential sum of the old serial loop: adding an empty chunk's +0.0 leaves it unchanged, so only the non-empty
// chunks are added, by thread 0, in order).
__global__ void __launch_bounds__(kChunks) draw_prefix_kernel(const float *__restrict__ p,
                                                              const uint32_t *__restrict__ mask, int64_t vocab,
                                                              DrawScratch *scratch) {
    __shared__ double sums[kChunks];
    __shared__ double cum[kChunks + 1];  // cum[j]: the first j non-empty chunks summed in order
    __shared__ uint32_t order[kChunks];  // the non-empty chunks in order
    __shared__ uint32_t warp_n[kChunks / 32];
    __shared__ uint32_t last_kept;
    const int c = threadIdx.x;
    const int lane = c & 31;
    if (c == 0) last_kept = 0;
    const int64_t len = chunk_len(vocab);
    const int64_t i0 = min(vocab, int64_t(c) * len);
    const int64_t i1 = min(vocab, i0 + len);
    double s = 0.0;
    uint32_t l = kNone;
    for (int64_t i = i0; i < i1;) {  // id order; an empty mask word is skipped whole
        const uint32_t word = mask[i >> 5];
        if (word == 0) {
            i = (i | 31) + 1;
            continue;
        }
        if ((word >> (i & 31)) & 1u) {
            s += double(p[i]);
            l = uint32_t(i);
        }
        ++i;
    }
    sums[c] = s;
    const bool nonempty = s != 0.0;
    const unsigned ballot = __ballot_sync(kFull, nonempty);
    if (lane == 0) warp_n[c >> 5] = uint32_t(__popc(ballot));
    __syncthreads();
    if (l != kNone) atomicMax(&last_kept, l);
    uint32_t before = uint32_t(__popc(ballot & ((1u << lane) - 1u)));
    uint32_t count = 0;
    for (int x = 0; x < kChunks / 32; ++x) {
        if (x < (c >> 5)) before += warp_n[x];
        count += warp_n[x];
    }
    if (nonempty) order[before] = uint32_t(c);
    __syncthreads();
    if (c == 0) {
        double acc = 0.0;
        cum[0] = acc;
        for (uint32_t j = 0; j < count; ++j) {
            acc += sums[order[j]];
            cum[j + 1] = acc;
        }
    }
    __syncthreads();
    scratch->prefix[c] = cum[before];
    if (c == 0) {
        scratch->prefix[kChunks] = cum[count];
        scratch->last_kept = last_kept;
    }
}

template <bool kSeeded>
__global__ void draw_kernel(const float *__restrict__ p, const uint32_t *__restrict__ mask, int64_t vocab,
                            const DrawScratch *__restrict__ scratch, const double *__restrict__ us,
                            uint64_t first_seed, uint32_t key, int64_t n, uint32_t *__restrict__ ids) {
    const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const double u = kSeeded ? uniform(first_seed + uint64_t(j), key, 0, 1) : us[j];
    const double target = u * scratch->prefix[kChunks];
    // The last chunk whose kept mass before it is <= target: no earlier id can exceed target.
    int lo = 0, hi = kChunks - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (scratch->prefix[mid] <= target) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int64_t len = chunk_len(vocab);
    for (int c = lo; c < kChunks; ++c) {
        double acc = scratch->prefix[c];
        const int64_t i1 = min(vocab, int64_t(c + 1) * len);
        for (int64_t i = int64_t(c) * len; i < i1;) {  // id order; an empty mask word is skipped whole
            const uint32_t word = mask[i >> 5];
            if (word == 0) {
                i = (i | 31) + 1;
                continue;
            }
            if ((word >> (i & 31)) & 1u) {
                acc += double(p[i]);
                if (acc > target) {
                    ids[j] = uint32_t(i);
                    return;
                }
            }
            ++i;
        }
    }
    ids[j] = scratch->last_kept;  // u * S at or above the scanned total
}

// ---- finalize, logprobs (D7), count_commit (D5) ----

__global__ void finalize_kernel(const float *__restrict__ x, const uint32_t *__restrict__ top2,
                                const uint32_t *__restrict__ drawn, cere_token_score *out) {
    const uint32_t id = drawn != nullptr ? *drawn : top2[0];
    const uint32_t runner_up = id == top2[0] ? top2[1] : top2[0];
    *out = cere_token_score{id, runner_up, x[id], x[runner_up]};
}

__global__ void top_logprobs_kernel(const float *__restrict__ x, int64_t vocab, float temperature,
                                    const uint32_t *committed_dev, uint32_t committed_host, int64_t n,
                                    float *committed_logprob, uint32_t *top_ids, float *top_lps, int64_t stride) {
    const double lse = log_sum_exp(x, vocab, temperature);
    if (threadIdx.x == 0) {
        const uint32_t c = committed_dev != nullptr ? *committed_dev : committed_host;
        *committed_logprob = float(double(scaled(x, c, temperature)) - lse);
    }
    // Selection: round r takes the best candidate ranked after round r - 1's.
    Cand prev{INFINITY, kNone};
    for (int64_t r = 0; r < n; ++r) {
        Cand best{-INFINITY, kNone};
        for (int64_t j = threadIdx.x; j < vocab; j += kThreads) {
            const Cand c{scaled(x, j, temperature), uint32_t(j)};
            const bool after = prev.i == kNone || better(prev, c);
            if (after && better(c, best)) best = c;
        }
        best = block_best(best);
        if (threadIdx.x == 0) {
            top_ids[r * stride] = best.i;
            top_lps[r * stride] = float(double(best.v) - lse);
        }
        prev = best;
    }
}

__global__ void count_commit_kernel(uint32_t *counts, const uint32_t *__restrict__ ids, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + ids[i], 1u);
}

bool vocab_ok(int64_t vocab) { return vocab >= 2 && vocab <= std::numeric_limits<int32_t>::max(); }

unsigned blocks(int64_t n) { return unsigned((n + kElementwise - 1) / kElementwise); }

cudaError_t draw(const float *probs, const uint32_t *mask, int64_t vocab, const double *us, uint64_t first_seed,
                 int64_t n, uint32_t key, uint32_t *ids, void *scratch, cudaStream_t stream) {
    if (!probs || !mask || !vocab_ok(vocab) || n < 0 || n > std::numeric_limits<int32_t>::max() || (n > 0 && !ids) ||
        !scratch_ok(scratch)) {
        return cudaErrorInvalidValue;
    }
    if (n == 0) return cudaSuccess;
    auto *s = static_cast<DrawScratch *>(scratch);
    draw_prefix_kernel<<<1, kChunks, 0, stream>>>(probs, mask, vocab, s);
    if (us != nullptr) {
        draw_kernel<false><<<blocks(n), kElementwise, 0, stream>>>(probs, mask, vocab, s, us, 0, 0, n, ids);
    } else {
        draw_kernel<true><<<blocks(n), kElementwise, 0, stream>>>(probs, mask, vocab, s, nullptr, first_seed, key, n,
                                                                  ids);
    }
    return cudaGetLastError();
}

}  // namespace

cudaError_t process(const bf16 *logits, int64_t vocab, const uint32_t *counts, float frequency_penalty,
                    float presence_penalty, float repetition_penalty, const uint32_t *mask_dev, const uint32_t *bias_ids,
                    const float *bias_values, int64_t bias_stride, int64_t bias_count, float *out,
                    cudaStream_t stream) {
    if (!logits || !out || !vocab_ok(vocab) || bias_count < 0 || bias_count > vocab ||
        (bias_count > 0 && (!bias_ids || !bias_values || bias_stride < 1))) {
        return cudaErrorInvalidValue;
    }
    process_kernel<<<blocks(vocab), kElementwise, 0, stream>>>(logits, vocab, counts, frequency_penalty,
                                                               presence_penalty, repetition_penalty, mask_dev, out);
    if (bias_count > 0) {
        bias_kernel<<<blocks(bias_count), kElementwise, 0, stream>>>(bias_ids, bias_values, bias_stride, bias_count,
                                                                     out);
    }
    return cudaGetLastError();
}

cudaError_t top2_f32(const float *values, int64_t vocab, uint32_t *ids, float *scores, cudaStream_t stream) {
    if (!values || !ids || !scores || !vocab_ok(vocab)) return cudaErrorInvalidValue;
    top2_kernel<<<1, kThreads, 0, stream>>>(values, vocab, ids, scores);
    return cudaGetLastError();
}

size_t scratch_bytes() { return sizeof(Work) > sizeof(DrawScratch) ? sizeof(Work) : sizeof(DrawScratch); }

cudaError_t probabilities(const float *processed, int64_t vocab, float temperature, float *probs, double *lse,
                          void *scratch, cudaStream_t stream) {
    if (!processed || !probs || !lse || !vocab_ok(vocab) || !(temperature > 0.0f) || !scratch_ok(scratch)) {
        return cudaErrorInvalidValue;
    }
    auto *w = static_cast<Work *>(scratch);
    max_kernel<<<kGrid, kBlock, 0, stream>>>(processed, vocab, temperature, w);
    sum_kernel<<<kGrid, kBlock, 0, stream>>>(processed, vocab, temperature, w);
    probs_kernel<<<kGrid, kBlock, 0, stream>>>(processed, vocab, temperature, w, probs, lse);
    return cudaGetLastError();
}

cudaError_t keep_mask(const float *probs, int64_t vocab, int64_t top_k, float top_p, float min_p, uint32_t *mask,
                      void *scratch, cudaStream_t stream) {
    if (!probs || !mask || !vocab_ok(vocab) || top_k < 1 || top_k > vocab || !(top_p > 0.0f && top_p <= 1.0f) ||
        !(min_p >= 0.0f && min_p <= 1.0f) || !scratch_ok(scratch)) {
        return cudaErrorInvalidValue;
    }
    auto *w = static_cast<Work *>(scratch);
    const cudaError_t cleared =
        cudaMemsetAsync(static_cast<unsigned char *>(scratch) + offsetof(Work, counters), 0, sizeof(Counters), stream);
    if (cleared != cudaSuccess) return cleared;
    hist_kernel<0><<<kGrid, kBlock, 0, stream>>>(probs, vocab, top_k, w);
    hist_kernel<1><<<kGrid, kBlock, 0, stream>>>(probs, vocab, top_k, w);
    hist_kernel<2><<<kGrid, kBlock, 0, stream>>>(probs, vocab, top_k, w);
    const int64_t words = (vocab + 31) / 32;
    const int64_t span = (words + kGrid - 1) / kGrid;
    const int collect_blocks = int((words + span - 1) / span);  // <= kGrid
    collect_kernel<<<collect_blocks, kBlock, 0, stream>>>(probs, vocab, top_k, span, w, mask);
    resolve_kernel<<<1, kThreads, 0, stream>>>(probs, vocab, top_k, top_p, min_p, span, collect_blocks, w, mask);
    // Only top_k > kCandidates can leave the cut beyond the resolved ranks (resolve_kernel's kMasked).
    if (top_k > kCandidates) masked_kernel<<<1, kThreads, 0, stream>>>(probs, vocab, top_k, top_p, min_p, w, mask);
    return cudaGetLastError();
}

cudaError_t draw_u(const float *probs, const uint32_t *mask, int64_t vocab, const double *us, int64_t n, uint32_t *ids,
                   void *scratch, cudaStream_t stream) {
    if (n > 0 && !us) return cudaErrorInvalidValue;
    return draw(probs, mask, vocab, us, 0, n, 0, ids, scratch, stream);
}

cudaError_t draw_seeded(const float *probs, const uint32_t *mask, int64_t vocab, uint64_t first_seed, int64_t n,
                        uint32_t key, uint32_t *ids, void *scratch, cudaStream_t stream) {
    return draw(probs, mask, vocab, nullptr, first_seed, n, key, ids, scratch, stream);
}

cudaError_t finalize(const float *processed, const uint32_t *top2_ids, const uint32_t *drawn, cere_token_score *out,
                     cudaStream_t stream) {
    if (!processed || !top2_ids || !out) return cudaErrorInvalidValue;
    finalize_kernel<<<1, 1, 0, stream>>>(processed, top2_ids, drawn, out);
    return cudaGetLastError();
}

cudaError_t top_logprobs(const float *processed, int64_t vocab, float temperature, const uint32_t *committed_dev,
                         uint32_t committed_host, int64_t n, float *committed_logprob, uint32_t *top_ids,
                         float *top_logprobs, int64_t stride, cudaStream_t stream) {
    if (!processed || !committed_logprob || !vocab_ok(vocab) || !(temperature >= 0.0f) || n < 0 ||
        n > int64_t(CERE_MAX_TOP_LOGPROBS) || n > vocab || (n > 0 && (!top_ids || !top_logprobs || stride < 1)) ||
        (committed_dev == nullptr && committed_host >= uint64_t(vocab))) {
        return cudaErrorInvalidValue;
    }
    top_logprobs_kernel<<<1, kThreads, 0, stream>>>(processed, vocab, temperature, committed_dev, committed_host, n,
                                                    committed_logprob, top_ids, top_logprobs, stride);
    return cudaGetLastError();
}

cudaError_t count_commit(uint32_t *counts, const uint32_t *ids, int64_t n, cudaStream_t stream) {
    if (!counts || n < 0 || n > std::numeric_limits<int32_t>::max() || (n > 0 && !ids)) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    count_commit_kernel<<<blocks(n), kElementwise, 0, stream>>>(counts, ids, n);
    return cudaGetLastError();
}

}  // namespace cere::k12
