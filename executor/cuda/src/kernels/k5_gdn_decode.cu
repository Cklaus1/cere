// K5 decode (m22n design note D5): single-token gated delta rule, FlashInfer BF16-state MTP-ILP4 arithmetic.
//
// math_reference: flashinfer
// math_reference: sglang
//
// Written from the math of flashinfer/gdn_kernels/gdn_decode_bf16_state.py (the T=1 path of
// gdn_decode_bf16state_mtp_ilp4_kernel) and the PTX the image's CuTe DSL (fastmath=True) emits for it; see
// k5_gdn_decode.h. Nothing is vendored. Every rounding step is spelled out (no contraction left to the compiler):
// a warp owns 4 value rows of one value head, lane l owns k in [4l, 4l+4).
#include "kernels/k5_gdn_decode.h"

#include "cere_model_constants.h"

#include <cstdint>

namespace cere::k5 {
namespace {

constexpr int kKeyHeads = gen::kLinearNumKeyHeads;
constexpr int kValueHeads = gen::kLinearNumValueHeads;
static_assert(gen::kLinearKeyHeadDim == gen::kLinearValueHeadDim, "one GDN head dim");
constexpr int kHeadDim = gen::kLinearKeyHeadDim;  // key and value head dim
constexpr int kLaneElems = kHeadDim / 32;
constexpr int kRowsPerWarp = 4;  // ILP4
constexpr int kWarpsPerBlock = 4;
constexpr int kRowsPerBlock = kRowsPerWarp * kWarpsPerBlock;
constexpr float kL2Eps = 1e-6f;
constexpr float kSoftplusThreshold = 20.0f;
constexpr float kLog2e = 0x1.715476p+0f;  // float(log2(e)), the constant fast exp multiplies by
constexpr float kLn2 = 0x1.62e430p-1f;    // float(ln(2)), the constant fast log multiplies by

static_assert(kLaneElems == 4, "lane owns 4 consecutive k");

__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float lg2_approx(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rsqrt_approx(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// exp(x) as the fast path lowers it: ex2.approx.ftz(x * log2(e)).
__device__ __forceinline__ float exp_fast(float x) { return ex2_approx(__fmul_rn(x, kLog2e)); }

__device__ __forceinline__ float butterfly_sum(float s) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, offset));
    }
    return s;
}

__device__ __forceinline__ void load4(const __nv_bfloat16 *p, float out[4]) {
    const uint2 raw = *reinterpret_cast<const uint2 *>(p);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.y));
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = hi.x;
    out[3] = hi.y;
}

__device__ __forceinline__ void store4(__nv_bfloat16 *p, const float in[4]) {
    uint2 raw;
    *reinterpret_cast<__nv_bfloat162 *>(&raw.x) = __floats2bfloat162_rn(in[0], in[1]);
    *reinterpret_cast<__nv_bfloat162 *>(&raw.y) = __floats2bfloat162_rn(in[2], in[3]);
    *reinterpret_cast<uint2 *>(p) = raw;
}

// rsqrt(sum x^2 + eps) over the warp: per lane sum += x*x as FMAs in k order, then the xor butterfly 16..1.
__device__ __forceinline__ float inv_l2(const float x[4]) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        s = __fmaf_rn(x[i], x[i], s);
    }
    return rsqrt_approx(__fadd_rn(butterfly_sum(s), kL2Eps));
}

// sum_i h[i] * x[i] over the row: two FMA chains (even, odd k) per lane, their sum, then the butterfly.
__device__ __forceinline__ float row_dot(const float h[4], const float x[4]) {
    float even = __fmaf_rn(h[0], x[0], 0.0f);
    float odd = __fmaf_rn(h[1], x[1], 0.0f);
    even = __fmaf_rn(h[2], x[2], even);
    odd = __fmaf_rn(h[3], x[3], odd);
    return butterfly_sum(__fadd_rn(even, odd));
}

// One token's step on a warp's 4 state rows h (FP32 copies of the BF16 state, updated in place) into their 4 outputs
// o: the lane's q and k slices (key head, k in [k0, k0 + 4)), the warp's 4 v entries and the value head's a, b,
// A_log and dt_bias. Both kernels run exactly this arithmetic per token.
__device__ __forceinline__ void decode_step(float (&h)[kRowsPerWarp][kLaneElems], const __nv_bfloat16 *q_lane,
                                            const __nv_bfloat16 *k_lane, const __nv_bfloat16 *v_rows,
                                            __nv_bfloat16 a, __nv_bfloat16 b, float A_log, __nv_bfloat16 dt_bias,
                                            float scale, float (&o)[kRowsPerWarp]) {
    float q[kLaneElems];
    float k[kLaneElems];
    load4(q_lane, q);
    load4(k_lane, k);
    const float inv_q = __fmul_rn(inv_l2(q), scale);
    const float inv_k = inv_l2(k);
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        q[i] = __fmul_rn(inv_q, q[i]);
        k[i] = __fmul_rn(inv_k, k[i]);
    }

    // Gating, FP32: x = a + dt_bias; softplus = (x <= 20) ? log(1+exp(x)) : x (a select, not a blend: log(1+exp(x))
    // is +inf for x > ~88, and inf * 0 would make the blend NaN); decay = exp(-exp(A_log) * softplus);
    // beta = 1 / (1 + exp(-b)) with a correctly rounded reciprocal.
    const float x = __fadd_rn(__bfloat162float(a), __bfloat162float(dt_bias));
    const float log1p_exp = __fmul_rn(lg2_approx(__fadd_rn(exp_fast(x), 1.0f)), kLn2);
    const float softplus = x <= kSoftplusThreshold ? log1p_exp : x;
    const float decay = exp_fast(-__fmul_rn(softplus, exp_fast(A_log)));
    const float beta = __frcp_rn(__fadd_rn(exp_fast(-__bfloat162float(b)), 1.0f));

    float v[kRowsPerWarp];
    load4(v_rows, v);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int i = 0; i < kLaneElems; ++i) {
            h[r][i] = __fmul_rn(h[r][i], decay);
        }
        const float vn = __fmul_rn(beta, __fsub_rn(v[r], row_dot(h[r], k)));
#pragma unroll
        for (int i = 0; i < kLaneElems; ++i) {
            h[r][i] = __fmaf_rn(k[i], vn, h[r][i]);
        }
        o[r] = row_dot(h[r], q);
    }
}

__global__ void __launch_bounds__(kWarpsPerBlock * 32) delta_rule_decode_kernel(DecodeArgs p) {
    const int hv = blockIdx.x;
    const int hk = hv / (kValueHeads / kKeyHeads);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int v0 = blockIdx.y * kRowsPerBlock + warp * kRowsPerWarp;
    const int k0 = lane * kLaneElems;

    __nv_bfloat16 *state = p.state + static_cast<int64_t>(hv) * kHeadDim * kHeadDim;
    float h[kRowsPerWarp][kLaneElems];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        load4(state + static_cast<int64_t>(v0 + r) * kHeadDim + k0, h[r]);
    }
    float o[kRowsPerWarp];
    decode_step(h, p.q + hk * kHeadDim + k0, p.k + hk * kHeadDim + k0, p.v + hv * kHeadDim + v0, p.a[hv], p.b[hv],
                p.A_log[hv], p.dt_bias[hv], p.scale, o);
    if (lane == 0) {
        store4(p.o + hv * kHeadDim + v0, o);
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        store4(state + static_cast<int64_t>(v0 + r) * kHeadDim + k0, h[r]);
    }
}

// A chain's rows in order on registers (grid.z = chain): after each row but the chain's last the rows are stored to
// that row's save and rounded to BF16 in place, the values the next one-token launch would load; after the last,
// stored to the chain's state.
__global__ void __launch_bounds__(kWarpsPerBlock * 32) delta_rule_decode_rows_kernel(DecodeRowsArgs p) {
    const int hv = blockIdx.x;
    const int hk = hv / (kValueHeads / kKeyHeads);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int v0 = blockIdx.y * kRowsPerBlock + warp * kRowsPerWarp;
    const int k0 = lane * kLaneElems;
    const int64_t head = static_cast<int64_t>(hv) * kHeadDim * kHeadDim;
    __nv_bfloat16 *const state = p.states[blockIdx.z];
    const int first = p.first[blockIdx.z];
    const int end = first + p.rows[blockIdx.z];

    float h[kRowsPerWarp][kLaneElems];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        load4(state + head + static_cast<int64_t>(v0 + r) * kHeadDim + k0, h[r]);
    }
    for (int t = first; t < end; ++t) {
        const int64_t qkv = t * p.qkv_stride;
        const int64_t ab = t * p.ab_stride + hv;
        float o[kRowsPerWarp];
        decode_step(h, p.step.q + qkv + hk * kHeadDim + k0, p.step.k + qkv + hk * kHeadDim + k0,
                    p.step.v + qkv + hv * kHeadDim + v0, p.step.a[ab], p.step.b[ab], p.step.A_log[hv],
                    p.step.dt_bias[hv], p.step.scale, o);
        if (lane == 0) {
            store4(p.step.o + t * p.o_stride + hv * kHeadDim + v0, o);
        }
        const bool last = t + 1 == end;
        __nv_bfloat16 *dst = (last ? state : p.saves[t]) + head;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            store4(dst + static_cast<int64_t>(v0 + r) * kHeadDim + k0, h[r]);
#pragma unroll
            for (int i = 0; i < kLaneElems; ++i) {
                h[r][i] = __bfloat162float(__float2bfloat16_rn(h[r][i]));
            }
        }
    }
}

bool aligned8(const void *ptr) { return ptr != nullptr && reinterpret_cast<uintptr_t>(ptr) % 8 == 0; }

// Every pointer of `args` but the state.
bool inputs_ok(const DecodeArgs &args) {
    return aligned8(args.q) && aligned8(args.k) && aligned8(args.v) && aligned8(args.o) && args.a != nullptr &&
           args.b != nullptr && args.A_log != nullptr && args.dt_bias != nullptr;
}

bool args_ok(const DecodeArgs &args) { return inputs_ok(args) && aligned8(args.state); }

}  // namespace

cudaError_t delta_rule_decode(const DecodeArgs &args, cudaStream_t stream) {
    if (!args_ok(args)) {
        return cudaErrorInvalidValue;
    }
    delta_rule_decode_kernel<<<dim3(kValueHeads, kHeadDim / kRowsPerBlock), kWarpsPerBlock * 32, 0, stream>>>(args);
    return cudaGetLastError();
}

cudaError_t delta_rule_decode_rows(const DecodeRowsArgs &args, cudaStream_t stream) {
    // Row r's q and k (kKeyHeads x kHeadDim each), v and o (kValueHeads x kHeadDim) never overlap row r + 1's.
    constexpr int64_t kRow = kValueHeads * kHeadDim;
    if (!inputs_ok(args.step) || args.chains < 1 || args.chains > kDecodeMaxRows) {
        return cudaErrorInvalidValue;
    }
    int next = 0;
    for (int c = 0; c < args.chains; ++c) {
        if (!aligned8(args.states[c]) || args.first[c] != next || args.rows[c] < 1 ||
            args.rows[c] > kDecodeMaxRows - next) {
            return cudaErrorInvalidValue;
        }
        next += args.rows[c];
        for (int r = args.first[c]; r + 1 < next; ++r) {
            if (!aligned8(args.saves[r])) {
                return cudaErrorInvalidValue;
            }
        }
    }
    if (next > 1 && (args.qkv_stride < kRow || args.qkv_stride % 4 != 0 || args.ab_stride < kValueHeads ||
                     args.o_stride < kRow || args.o_stride % 4 != 0)) {
        return cudaErrorInvalidValue;
    }
    delta_rule_decode_rows_kernel<<<dim3(kValueHeads, kHeadDim / kRowsPerBlock, static_cast<unsigned>(args.chains)),
                                    kWarpsPerBlock * 32, 0, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace cere::k5
