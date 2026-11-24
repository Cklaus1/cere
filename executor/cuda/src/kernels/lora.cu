// Track LoRA: the adapter table and its shrink / expand kernels. Contract in lora.h.
#include "kernels/lora.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cere::kernels::lora {
namespace {

constexpr uint32_t kRowsPerBlock = 8;
constexpr uint32_t kShrinkWarps = 8;  // one rank index j per warp
constexpr uint32_t kExpandThreads = 128;
constexpr uint32_t kMaxRank = 256;
constexpr uint64_t kAlign = 256;

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

struct SegArgs {
    uint64_t a_offset;
    uint64_t b_offset;
    uint32_t col;
    uint32_t n;
};

// Row r's adapter: index ri = per_row ? r : 0 into slot / rank / scale.
struct Args {
    const __nv_bfloat16 *x;
    __nv_bfloat16 *y;
    const __nv_bfloat16 *pool;
    float2 *t;  // (hi, lo) double-float
    uint64_t slot_elems;
    uint32_t m;
    uint32_t k;
    uint32_t ld_y;
    uint32_t rank;  // R: the scratch's per-segment stride and B's row stride
    uint32_t per_row;
    SegArgs seg[kMaxSegments];
    int8_t slot[kMaxRows];
    uint16_t seg_rank[kMaxRows][kMaxSegments];
    float scale[kMaxRows];
};

// Knuth's TwoSum: s + e == a + b exactly (round-to-nearest intrinsics: never contracted or reassociated).
__device__ __forceinline__ void two_sum(const float a, const float b, float &s, float &e) {
    s = __fadd_rn(a, b);
    const float bp = __fsub_rn(s, a);
    e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bp)), __fsub_rn(b, bp));
}

// (hi, lo) += p (Ogita-Rump-Oishi Sum2: hi carries the rounded sum, lo the sum of the exact rounding errors).
__device__ __forceinline__ void add2(float &hi, float &lo, const float p) {
    float e;
    two_sum(hi, p, hi, e);
    lo = __fadd_rn(lo, e);
}

// (hi, lo) += (h2, l2).
__device__ __forceinline__ void add22(float &hi, float &lo, const float h2, const float l2) {
    float e;
    two_sum(hi, h2, hi, e);
    lo = __fadd_rn(__fadd_rn(lo, l2), e);
}

// bf16 of the double d rounded once to nearest even (normal range): rounded to odd in float first (24 >= 8 + 2 bits).
__device__ __forceinline__ __nv_bfloat16 bf16_of(const double d) {
    float f = __double2float_rz(d);
    if (static_cast<double>(f) != d) f = __int_as_float(__float_as_int(f) | 1);
    return __float2bfloat16_rn(f);
}

// x · a over 8 BF16 pairs into (hi, lo): every product of two BF16 values is exact in FP32.
__device__ __forceinline__ void dot8(const uint4 &xa, const uint4 &aa, float &hi, float &lo) {
    const __nv_bfloat162 *xv = reinterpret_cast<const __nv_bfloat162 *>(&xa);
    const __nv_bfloat162 *av = reinterpret_cast<const __nv_bfloat162 *>(&aa);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 xf = __bfloat1622float2(xv[e]);
        const float2 af = __bfloat1622float2(av[e]);
        add2(hi, lo, __fmul_rn(xf.x, af.x));
        add2(hi, lo, __fmul_rn(xf.y, af.y));
    }
}

// t[r, s·R + j] = Σ_i x[r, i] · A[j, i] as a double-float: lane l sums i in [8l + 256c, 8l + 256c + 8) for c
// ascending, then a fixed xor-shuffle tree. k is a multiple of 8 (Table::init).
__global__ void shrink_kernel(const Args p) {
    const uint32_t s = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t j = blockIdx.z * kShrinkWarps + (threadIdx.x >> 5);
    const SegArgs seg = p.seg[s];
    for (uint32_t i = 0; i < kRowsPerBlock; ++i) {
        const uint32_t r = blockIdx.x * kRowsPerBlock + i;
        if (r >= p.m) break;
        const uint32_t ri = p.per_row ? r : 0;
        const int slot = p.slot[ri];
        if (slot < 0 || j >= p.seg_rank[ri][s]) continue;
        const __nv_bfloat16 *a = p.pool + uint64_t(slot) * p.slot_elems + seg.a_offset + uint64_t(j) * p.k;
        const __nv_bfloat16 *x = p.x + uint64_t(r) * p.k;
        float hi = 0.0f, lo = 0.0f;
        for (uint32_t c = lane * 8; c < p.k; c += 256) {
            dot8(*reinterpret_cast<const uint4 *>(x + c), *reinterpret_cast<const uint4 *>(a + c), hi, lo);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float h2 = __shfl_xor_sync(0xffffffffu, hi, o);
            const float l2 = __shfl_xor_sync(0xffffffffu, lo, o);
            add22(hi, lo, h2, l2);
        }
        if (lane == 0) p.t[uint64_t(r) * kMaxSegments * p.rank + s * p.rank + j] = make_float2(hi, lo);
    }
}

// y[r, col + c] = bf16(float(y) + scale · Σ_j t[r, s·R + j] · B[c, j]), the sum a double-float with j ascending (TwoProd
// by fmaf for t.hi · B; BF16 B is exact in FP32) and the result rounded once; a zero product skips the write (y keeps
// its bits, -0.0 included).
__global__ void expand_kernel(const Args p) {
    __shared__ float2 ts[kRowsPerBlock][kMaxRank];
    const uint32_t s = blockIdx.y;
    const SegArgs seg = p.seg[s];
    const uint32_t r0 = blockIdx.x * kRowsPerBlock;
    for (uint32_t idx = threadIdx.x; idx < kRowsPerBlock * p.rank; idx += blockDim.x) {
        const uint32_t i = idx / p.rank, j = idx % p.rank;
        const uint32_t r = r0 + i;
        ts[i][j] = r < p.m ? p.t[uint64_t(r) * kMaxSegments * p.rank + s * p.rank + j] : make_float2(0.0f, 0.0f);
    }
    __syncthreads();
    const uint32_t c = blockIdx.z * kExpandThreads + threadIdx.x;
    if (c >= seg.n) return;
    for (uint32_t i = 0; i < kRowsPerBlock; ++i) {
        const uint32_t r = r0 + i;
        if (r >= p.m) break;
        const uint32_t ri = p.per_row ? r : 0;
        const int slot = p.slot[ri];
        const uint32_t rank = p.seg_rank[ri][s];
        if (slot < 0 || rank == 0) continue;
        const __nv_bfloat16 *b = p.pool + uint64_t(slot) * p.slot_elems + seg.b_offset + uint64_t(c) * p.rank;
        float hi = 0.0f, lo = 0.0f;
        for (uint32_t j = 0; j < rank; j += 8) {
            const uint4 bb = *reinterpret_cast<const uint4 *>(b + j);
            const __nv_bfloat162 *bv = reinterpret_cast<const __nv_bfloat162 *>(&bb);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float bf = e % 2 == 0 ? __low2float(bv[e / 2]) : __high2float(bv[e / 2]);
                const float2 t = ts[i][j + e];
                const float ph = __fmul_rn(t.x, bf);
                const float pe = __fmaf_rn(t.x, bf, -ph);  // t.hi · b - ph, exact
                add2(hi, lo, ph);
                lo = __fadd_rn(lo, __fmaf_rn(t.y, bf, pe));
            }
        }
        const float scale = p.scale[ri];
        if (scale == 0.0f || (hi == 0.0f && lo == 0.0f)) continue;
        __nv_bfloat16 *y = p.y + uint64_t(r) * p.ld_y + seg.col + c;
        const double delta = static_cast<double>(scale) * (static_cast<double>(hi) + static_cast<double>(lo));
        *y = bf16_of(static_cast<double>(__bfloat162float(*y)) + delta);
    }
}

}  // namespace

Table::~Table() {
    if (memory_ != nullptr) cudaFree(memory_);
    if (copy_ != nullptr) cudaStreamDestroy(copy_);
}

uint64_t Table::bytes(uint32_t adapters, uint32_t rank, uint64_t elements_per_rank, uint32_t max_tokens) {
    return align_up(uint64_t{adapters} * rank * elements_per_rank * sizeof(__nv_bfloat16)) +
           align_up(uint64_t{max_tokens} * kMaxSegments * rank * sizeof(float2));
}

cere_status Table::init(uint32_t adapters, uint32_t rank, uint32_t max_tokens, std::vector<TargetLayout> targets,
                        std::vector<Entry> entries) {
    if (memory_ != nullptr || adapters == 0 || adapters > 127 || rank == 0 || rank % 8 != 0 || rank > kMaxRank ||
        max_tokens == 0 || targets.empty()) {
        return CERE_ERR_ARGUMENT;
    }
    uint64_t elements_per_rank = 0;
    for (const TargetLayout &t : targets) {
        if (t.k == 0 || t.n == 0 || t.k % 8 != 0 || t.a_offset % 8 != 0 || t.b_offset % 8 != 0) {
            return CERE_ERR_ARGUMENT;
        }
        elements_per_rank += uint64_t{t.k} + t.n;
    }
    for (const Entry &e : entries) {
        if (e.w == nullptr || e.segments == 0 || e.segments > kMaxSegments) return CERE_ERR_ARGUMENT;
        for (uint32_t s = 0; s < e.segments; ++s) {
            if (e.seg[s].target >= targets.size() || targets[e.seg[s].target].k != e.k ||
                targets[e.seg[s].target].n != e.seg[s].n) {
                return CERE_ERR_ARGUMENT;
            }
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
        return reinterpret_cast<uintptr_t>(a.w) < reinterpret_cast<uintptr_t>(b.w);
    });
    for (size_t i = 1; i < entries.size(); ++i) {
        if (entries[i].w == entries[i - 1].w) return CERE_ERR_ARGUMENT;
    }
    const uint64_t pool_bytes = align_up(uint64_t{adapters} * rank * elements_per_rank * sizeof(__nv_bfloat16));
    const uint64_t total = bytes(adapters, rank, elements_per_rank, max_tokens);
    if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
        copy_ = nullptr;
        cudaGetLastError();
        return CERE_ERR_INTERNAL;
    }
    void *memory = nullptr;
    if (cudaMalloc(&memory, total) != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_CAPACITY;
    }
    if (cudaMemsetAsync(memory, 0, total, copy_) != cudaSuccess || cudaStreamSynchronize(copy_) != cudaSuccess) {
        cudaFree(memory);
        cudaGetLastError();
        return CERE_ERR_INTERNAL;
    }
    memory_ = memory;
    pool_ = static_cast<__nv_bfloat16 *>(memory);
    scratch_ = reinterpret_cast<float2 *>(static_cast<char *>(memory) + pool_bytes);
    adapters_ = adapters;
    rank_ = rank;
    max_tokens_ = max_tokens;
    slot_elems_ = uint64_t{rank} * elements_per_rank;
    targets_ = std::move(targets);
    entries_ = std::move(entries);
    ranks_.assign(uint64_t{adapters} * targets_.size(), 0);
    scales_.assign(adapters, 0.0f);
    return CERE_OK;
}

cere_status Table::load(uint32_t slot, uint32_t rank, float scale, uint64_t count, const uint32_t *target_index,
                        const void *const *a, const void *const *b) {
    if (memory_ == nullptr || slot >= adapters_ || rank == 0 || rank > rank_ || !std::isfinite(scale) ||
        target_index == nullptr || a == nullptr || b == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (target_index[i] >= targets_.size() || a[i] == nullptr || b[i] == nullptr) return CERE_ERR_ARGUMENT;
    }
    unload(slot);
    __nv_bfloat16 *base = pool_ + uint64_t{slot} * slot_elems_;
    bool ok = cudaMemsetAsync(base, 0, slot_elems_ * sizeof(__nv_bfloat16), copy_) == cudaSuccess;
    for (uint64_t i = 0; ok && i < count; ++i) {
        const TargetLayout &t = targets_[target_index[i]];
        const size_t row = size_t{rank} * sizeof(__nv_bfloat16);
        ok = cudaMemcpyAsync(base + t.a_offset, a[i], row * t.k, cudaMemcpyHostToDevice, copy_) == cudaSuccess &&
             cudaMemcpy2DAsync(base + t.b_offset, size_t{rank_} * sizeof(__nv_bfloat16), b[i], row, row, t.n,
                               cudaMemcpyHostToDevice, copy_) == cudaSuccess;
    }
    const cudaError_t sync = cudaStreamSynchronize(copy_);
    if (!ok || sync != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_INTERNAL;
    }
    const uint16_t padded = uint16_t((rank + 7) / 8 * 8);
    for (uint64_t i = 0; i < count; ++i) {
        ranks_[uint64_t{slot} * targets_.size() + target_index[i]] = padded;
    }
    scales_[slot] = scale;
    return CERE_OK;
}

void Table::unload(uint32_t slot) {
    if (slot >= adapters_) return;
    std::fill_n(ranks_.begin() + int64_t(uint64_t{slot} * targets_.size()), targets_.size(), uint16_t{0});
    scales_[slot] = 0.0f;
}

const Entry *Table::find(const void *w) const {
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), w, [](const Entry &e, const void *p) {
        return reinterpret_cast<uintptr_t>(e.w) < reinterpret_cast<uintptr_t>(p);
    });
    return it != entries_.end() && it->w == w ? &*it : nullptr;
}

void Table::activate(int32_t slot) {
    per_row_ = false;
    uniform_ = slot >= 0 && uint32_t(slot) < adapters_ ? slot : -1;
    active_ = uniform_ >= 0;
}

void Table::activate_rows(const int32_t *slots, uint32_t rows) {
    per_row_ = true;
    uniform_ = -1;
    active_ = false;
    for (uint32_t r = 0; r < kMaxRows; ++r) {
        const int32_t s = r < rows && slots != nullptr ? slots[r] : -1;
        slots_[r] = int8_t(s >= 0 && uint32_t(s) < adapters_ ? s : -1);
        active_ = active_ || slots_[r] >= 0;
    }
}

void Table::deactivate() {
    active_ = false;
    per_row_ = false;
    uniform_ = -1;
}

cere_status Table::apply(const Entry &e, const __nv_bfloat16 *x, uint32_t m, __nv_bfloat16 *y, uint32_t ld_y,
                         cudaStream_t stream) const {
    if (!active_ || m == 0) return CERE_OK;
    if (m > max_tokens_ || x == nullptr || y == nullptr || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        (per_row_ && m > kMaxRows)) {
        return CERE_ERR_ARGUMENT;
    }
    Args p{};
    p.x = x;
    p.y = y;
    p.pool = pool_;
    p.t = scratch_;
    p.slot_elems = slot_elems_;
    p.m = m;
    p.k = e.k;
    p.ld_y = ld_y;
    p.rank = rank_;
    p.per_row = per_row_ ? 1 : 0;
    const uint32_t rows = per_row_ ? m : 1;
    uint32_t max_rank = 0, max_n = 0;
    for (uint32_t s = 0; s < e.segments; ++s) {
        const TargetLayout &t = targets_[e.seg[s].target];
        p.seg[s] = SegArgs{t.a_offset, t.b_offset, e.seg[s].col, e.seg[s].n};
        max_n = std::max(max_n, e.seg[s].n);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        const int32_t slot = per_row_ ? slots_[r] : uniform_;
        p.slot[r] = int8_t(slot);
        p.scale[r] = slot >= 0 ? scales_[slot] : 0.0f;
        for (uint32_t s = 0; s < e.segments; ++s) {
            const uint16_t rank = slot >= 0 ? ranks_[uint64_t(slot) * targets_.size() + e.seg[s].target] : 0;
            p.seg_rank[r][s] = rank;
            max_rank = std::max<uint32_t>(max_rank, rank);
        }
    }
    if (max_rank == 0) return CERE_OK;
    const unsigned row_blocks = (m + kRowsPerBlock - 1) / kRowsPerBlock;
    shrink_kernel<<<dim3(row_blocks, e.segments, max_rank / kShrinkWarps), kShrinkWarps * 32, 0, stream>>>(p);
    expand_kernel<<<dim3(row_blocks, e.segments, (max_n + kExpandThreads - 1) / kExpandThreads), kExpandThreads, 0,
                    stream>>>(p);
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::kernels::lora
