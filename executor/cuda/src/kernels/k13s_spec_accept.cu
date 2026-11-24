// K13S phase-S accept/resample (m29 design note D6; ledger M8); contracts in k13s_spec_accept.h. The counter and
// the inverse CDF follow kernels/k12_sampler.cu (murmur_hash32, uniform, draw_prefix_kernel/draw_kernel) with the
// draft's id removed from the resample's kept set.
//
// math_reference: sglang
#include "kernels/k13s_spec_accept.h"

#include <cstddef>
#include <limits>

#include "kernels/k12_sampler.h"

namespace cere::k13s {
namespace {

using k12::kChunks;

constexpr unsigned kFull = 0xffffffffu;
constexpr int kElementwise = 256;
constexpr size_t kAlign = 256;
// ledger M8 "Counter RNG channels"
constexpr uint32_t kAcceptA = 4, kAcceptB = 5, kResampleA = 6, kResampleB = 7;

// ---- counter (k12_sampler.cu's D4 counter, bit for bit) ----

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

__host__ __device__ __forceinline__ int64_t chunk_len(int64_t vocab) { return (vocab + kChunks - 1) / kChunks; }

// One draft row's fp64 sums: prefix[c] = the kept mass without the draft before chunk c (prefix[kChunks] = S'),
// total = the kept mass with it (S), last_kept = the last kept id other than the draft (0 when there is none).
// The rows' probabilities come from k12 `probabilities` and keep_mask, which never hold NaN and always keep an id
// (a row with non-finite logits included), so total > 0 and the accept ratio is finite.
struct RowPrefix {
    double prefix[kChunks + 1];
    double total;
    uint32_t last_kept;
};

// Block r: row r's RowPrefix (draft drafts[r], or `draft` when drafts is null). Each thread sums one chunk in id
// order twice (with and without the draft); thread 0 then adds the non-empty chunks' sums in chunk order (an empty
// chunk adds +0.0, which changes nothing), as k12's draw_prefix_kernel.
__global__ void __launch_bounds__(kChunks) prefix_kernel(const float *__restrict__ probs,
                                                         const uint32_t *__restrict__ masks, int64_t vocab,
                                                         const uint32_t *__restrict__ drafts, uint32_t draft,
                                                         RowPrefix *__restrict__ out) {
    __shared__ double sums[kChunks];       // with the draft
    __shared__ double rest[kChunks];       // without it
    __shared__ double cum[kChunks + 1];    // cum[j]: the first j non-empty chunks' `rest` summed in order
    __shared__ uint32_t order[kChunks];    // the non-empty chunks in order
    __shared__ uint32_t warp_n[kChunks / 32];
    __shared__ uint32_t last_kept;
    const int64_t r = blockIdx.x;
    const int64_t words = (vocab + 31) / 32;
    const float *p = probs + r * vocab;
    const uint32_t *mask = masks + r * words;
    const uint32_t d = drafts != nullptr ? drafts[r] : draft;
    RowPrefix *row = out + r;
    const int c = threadIdx.x;
    const int lane = c & 31;
    if (c == 0) last_kept = 0;
    const int64_t len = chunk_len(vocab);
    const int64_t i0 = min(vocab, int64_t(c) * len);
    const int64_t i1 = min(vocab, i0 + len);
    double s = 0.0, s_rest = 0.0;
    uint32_t l = 0xffffffffu;
    for (int64_t i = i0; i < i1;) {  // id order; an empty mask word is skipped whole
        const uint32_t word = mask[i >> 5];
        if (word == 0) {
            i = (i | 31) + 1;
            continue;
        }
        if ((word >> (i & 31)) & 1u) {
            const double x = double(p[i]);
            s += x;
            if (uint32_t(i) != d) {
                s_rest += x;
                l = uint32_t(i);
            }
        }
        ++i;
    }
    sums[c] = s;
    rest[c] = s_rest;
    const bool nonempty = s != 0.0;
    const unsigned ballot = __ballot_sync(kFull, nonempty);
    if (lane == 0) warp_n[c >> 5] = uint32_t(__popc(ballot));
    __syncthreads();
    if (l != 0xffffffffu) atomicMax(&last_kept, l);
    uint32_t before = uint32_t(__popc(ballot & ((1u << lane) - 1u)));
    uint32_t count = 0;
    for (int x = 0; x < kChunks / 32; ++x) {
        if (x < (c >> 5)) before += warp_n[x];
        count += warp_n[x];
    }
    if (nonempty) order[before] = uint32_t(c);
    __syncthreads();
    if (c == 0) {
        double acc = 0.0, acc_rest = 0.0;
        cum[0] = acc_rest;
        for (uint32_t j = 0; j < count; ++j) {
            acc += sums[order[j]];
            acc_rest += rest[order[j]];
            cum[j + 1] = acc_rest;
        }
        row->total = acc;
    }
    __syncthreads();
    row->prefix[c] = cum[before];
    if (c == 0) {
        row->prefix[kChunks] = cum[count];
        row->last_kept = last_kept;
    }
}

// Trial t on row r = per_row ? t : 0 (draft drafts[r], or `draft`): the accept coin, then on reject the inverse CDF
// draw over the kept ids other than the draft (k12 draw_kernel's search and scan).
__global__ void decide_kernel(const float *__restrict__ probs, const uint32_t *__restrict__ masks, int64_t vocab,
                              const uint32_t *__restrict__ drafts, uint32_t draft, const RowPrefix *__restrict__ rows,
                              bool per_row, uint64_t first_seed, uint64_t seed_step, uint32_t key0, uint32_t key_step,
                              int64_t trials, uint32_t *__restrict__ ids, uint8_t *__restrict__ accepted) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= trials) return;
    const int64_t r = per_row ? t : 0;
    const int64_t words = (vocab + 31) / 32;
    const float *p = probs + r * vocab;
    const uint32_t *mask = masks + r * words;
    const uint32_t d = drafts != nullptr ? drafts[r] : draft;
    const RowPrefix &row = rows[r];
    const uint64_t seed = first_seed + uint64_t(t) * seed_step;
    const uint32_t key = key0 + uint32_t(t) * key_step;
    const double u = uniform(seed, key, kAcceptA, kAcceptB);
    if (int64_t(d) < vocab && kept(mask, d) && u < double(p[d]) / row.total) {
        ids[t] = d;
        accepted[t] = 1;
        return;
    }
    accepted[t] = 0;
    const double target = uniform(seed, key, kResampleA, kResampleB) * row.prefix[kChunks];
    // The last chunk whose mass before it is <= target: no earlier id can exceed target.
    int lo = 0, hi = kChunks - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (row.prefix[mid] <= target) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int64_t len = chunk_len(vocab);
    for (int c = lo; c < kChunks; ++c) {
        double acc = row.prefix[c];
        const int64_t i1 = min(vocab, int64_t(c + 1) * len);
        for (int64_t i = int64_t(c) * len; i < i1;) {  // id order; an empty mask word is skipped whole
            const uint32_t word = mask[i >> 5];
            if (word == 0) {
                i = (i | 31) + 1;
                continue;
            }
            if (((word >> (i & 31)) & 1u) && uint32_t(i) != d) {
                acc += double(p[i]);
                if (acc > target) {
                    ids[t] = uint32_t(i);
                    return;
                }
            }
            ++i;
        }
    }
    ids[t] = row.last_kept;  // u' * S' at or above the scanned total
}

// The chain: the accepted drafts in order, then the first reject's resampled id, or the bonus id after n accepts.
__global__ void commit_kernel(const uint32_t *__restrict__ row_ids, const uint8_t *__restrict__ row_accepted,
                              int64_t n, const uint32_t *__restrict__ bonus, uint32_t *__restrict__ committed,
                              uint32_t *__restrict__ count) {
    for (int64_t j = 0; j < n; ++j) {
        committed[j] = row_ids[j];
        if (!row_accepted[j]) {
            *count = uint32_t(j + 1);
            return;
        }
    }
    committed[n] = *bonus;
    *count = uint32_t(n + 1);
}

size_t aligned(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

bool vocab_ok(int64_t vocab) { return vocab >= 2 && vocab <= std::numeric_limits<int32_t>::max(); }

bool scratch_ok(const void *scratch) {
    return scratch != nullptr && (reinterpret_cast<uintptr_t>(scratch) & 7u) == 0;
}

unsigned blocks(int64_t n) { return unsigned((n + kElementwise - 1) / kElementwise); }

// spec_accept's scratch for n drafts: k12's draw scratch, the bonus id, n RowPrefix, n row ids, n row flags.
struct ChainScratch {
    void *k12;
    uint32_t *bonus;
    RowPrefix *rows;
    uint32_t *row_ids;
    uint8_t *row_accepted;
};

size_t carve_chain(int64_t n, unsigned char *base, ChainScratch *out) {
    size_t at = 0;
    const auto take = [&](size_t bytes) {
        void *p = base != nullptr ? base + at : nullptr;
        at += aligned(bytes);
        return p;
    };
    ChainScratch s;
    s.k12 = take(k12::scratch_bytes());
    s.bonus = static_cast<uint32_t *>(take(sizeof(uint32_t)));
    s.rows = static_cast<RowPrefix *>(take(size_t(n) * sizeof(RowPrefix)));
    s.row_ids = static_cast<uint32_t *>(take(size_t(n) * sizeof(uint32_t)));
    s.row_accepted = static_cast<uint8_t *>(take(size_t(n)));
    if (out != nullptr) *out = s;
    return at;
}

}  // namespace

size_t scratch_bytes(int64_t max_drafts) {
    if (max_drafts < 0) max_drafts = 0;
    if (max_drafts > kMaxDrafts) max_drafts = kMaxDrafts;
    return carve_chain(max_drafts, nullptr, nullptr);
}

cudaError_t spec_accept(const float *probs, const uint32_t *masks, int64_t vocab, const uint32_t *drafts, int64_t n,
                        uint64_t seed, uint32_t key0, uint32_t *committed, uint32_t *count, void *scratch,
                        cudaStream_t stream) {
    if (!probs || !masks || !vocab_ok(vocab) || n < 0 || n > kMaxDrafts || (n > 0 && !drafts) || !committed ||
        !count || !scratch_ok(scratch)) {
        return cudaErrorInvalidValue;
    }
    ChainScratch s;
    carve_chain(n, static_cast<unsigned char *>(scratch), &s);
    const int64_t words = (vocab + 31) / 32;
    if (n > 0) {
        prefix_kernel<<<unsigned(n), kChunks, 0, stream>>>(probs, masks, vocab, drafts, 0, s.rows);
        decide_kernel<<<blocks(n), kElementwise, 0, stream>>>(probs, masks, vocab, drafts, 0, s.rows, true, seed, 0,
                                                              key0, 1, n, s.row_ids, s.row_accepted);
        const cudaError_t launched = cudaGetLastError();
        if (launched != cudaSuccess) return launched;
    }
    const cudaError_t drawn = k12::draw_seeded(probs + n * vocab, masks + n * words, vocab, seed, 1,
                                               key0 + uint32_t(n), s.bonus, s.k12, stream);
    if (drawn != cudaSuccess) return drawn;
    commit_kernel<<<1, 1, 0, stream>>>(s.row_ids, s.row_accepted, n, s.bonus, committed, count);
    return cudaGetLastError();
}

size_t trial_scratch_bytes() { return sizeof(RowPrefix); }

cudaError_t row_trials(const float *probs, const uint32_t *mask, int64_t vocab, uint32_t draft, uint64_t first_seed,
                       uint64_t seed_step, uint32_t key0, uint32_t key_step, int64_t trials, uint32_t *ids,
                       uint8_t *accepted, void *scratch, cudaStream_t stream) {
    if (!probs || !mask || !vocab_ok(vocab) || trials < 0 || trials > std::numeric_limits<int32_t>::max() ||
        (trials > 0 && (!ids || !accepted)) || !scratch_ok(scratch)) {
        return cudaErrorInvalidValue;
    }
    if (trials == 0) return cudaSuccess;
    auto *row = static_cast<RowPrefix *>(scratch);
    prefix_kernel<<<1, kChunks, 0, stream>>>(probs, mask, vocab, nullptr, draft, row);
    decide_kernel<<<blocks(trials), kElementwise, 0, stream>>>(probs, mask, vocab, nullptr, draft, row, false,
                                                               first_seed, seed_step, key0, key_step, trials, ids,
                                                               accepted);
    return cudaGetLastError();
}

}  // namespace cere::k13s
