#pragma once
// Inkling 2 Mini (inkling design note, G1): the shared numeric contract of every inkling CPU reference
// (executor/ref/inkling/<op>_ref.h) and every inkling kernel, CUDA and Vulkan. Header-only C++17, no CUDA/Vulkan
// dependency; the same source compiles as host C++ and as CUDA device code (CERE_HD), so a kernel that calls these
// routines and follows the reduction order below is bit-equal to its CPU reference (G1 = bit-equal, not tolerance).
//
// The contract (cross-backend bit equality):
// 1. Rounding. Every float operation is one IEEE-754 binary32 operation rounded to nearest even. A fused
//    multiply-add is written explicitly (ref::fma); every other multiply and add is unfused (ref::mul, ref::add,
//    or a plain `*`/`+` in host code, which this header's users compile without contraction: GCC/Clang in ISO
//    mode, -std=c++17, keep -ffp-contract=off; the inkling CUDA TUs are compiled with -fmad=false, and the
//    Vulkan shaders mark results NoContraction). Division and square root are correctly rounded (host native;
//    CUDA __fdiv_rn/__fsqrt_rn). rsqrt(x) := div(1, sqrt(x)) (two roundings); never rsqrtf or a hardware
//    approximation.
// 2. Transcendentals. exp, log, tanh and sigmoid are the fixed fma-only routines below (range reduction plus a
//    polynomial with the documented constants, at most 2 ulp from the exact value; executor/tests/host/
//    test_inkling_common_ref.cpp sweeps them). Inkling kernels call these, never expf/__expf/tanhf/logf.
// 3. Reductions. Every reduction (sum, dot, max) is ref::reduce_*<L>: lane t of L accumulates elements t, t+L,
//    t+2L, ... sequentially (sum: acc = add(acc, x); dot: acc = fma(a, b, acc); max: acc = max(acc, x)), starting
//    from 0.0f (sum, dot) or -INFINITY (max); then a halving tree over the lanes, offsets L/2, L/4, ..., 1, lane
//    i (i < off) = op(lane i, lane i + off); the result is lane 0. That equals lane 0 of a shfl_xor butterfly
//    (IEEE add and max are commutative), so a warp or a subgroup reduces with xor offsets L/2 .. 1 and a
//    block-wide L reduces through shared memory with the same offsets. Fixed widths: L = 256 for a 2304-wide
//    row (hidden size), 32 for a 128-wide row (head dim), 16 for d_rel 16, 64 for a rank-64 row.
// 4. Formats. bf16, FP4 E2M1, FP8 E4M3 (fn: no infinity, 0x7f/0xff NaN) and UE8M0 decode exactly (decode_*);
//    float -> bf16 rounds to nearest even (encode_bf16).
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define CERE_HD __host__ __device__
#else
#define CERE_HD
#endif

namespace ref {

CERE_HD inline float bits_float(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
#endif
}

CERE_HD inline uint32_t float_bits(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
#endif
}

// ---- 1. Rounded primitives -------------------------------------------------------------------------------------
CERE_HD inline float fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
CERE_HD inline float mul(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
CERE_HD inline float add(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
CERE_HD inline float sub(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
CERE_HD inline float div(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
CERE_HD inline float sqrt(float x) {
#if defined(__CUDA_ARCH__)
    return __fsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}
CERE_HD inline float rsqrt(float x) { return div(1.0f, ref::sqrt(x)); }
// IEEE maxNum: a NaN operand loses to a number (fmaxf on both backends).
CERE_HD inline float max(float a, float b) { return fmaxf(a, b); }

// ---- 2. Transcendentals ----------------------------------------------------------------------------------------
// 2^n for an integer n in [-149, 127], exact (subnormal below -126).
CERE_HD inline float exp2i(int n) {
    if (n >= -126) return bits_float(uint32_t(n + 127) << 23);
    return bits_float(uint32_t(1) << (n + 149));
}

// exp(x): k = round(x * log2(e)) by the 1.5 * 2^23 shifter, r = x - k ln2 (Cody-Waite, ln2 = 0.693359375 -
// 2.12194440e-4, both steps fused), exp(r) = 1 + r + r^2 * P(r) with the Cephes expf coefficients
// P = 5.0000001201e-1, 1.6666665459e-1, 4.1665795894e-2, 8.3334519073e-3, 1.3981999507e-3, 1.9875691500e-4
// (Horner, fused), then the result times 2^k (two exact-power factors when k leaves the normal range).
// x > 88.72283905 -> +inf, x < -103.972084 -> +0, NaN -> NaN.
CERE_HD inline float exp(float x) {
    if (x != x) return x;
    if (x > 88.72283905f) return INFINITY;
    if (x < -103.972084f) return 0.0f;
    const float shifter = 12582912.0f;  // 1.5 * 2^23
    const float k = sub(fma(x, 1.44269504088896341f, shifter), shifter);
    float r = fma(k, -0.693359375f, x);
    r = fma(k, 2.12194440e-4f, r);
    float p = 1.9875691500e-4f;
    p = fma(p, r, 1.3981999507e-3f);
    p = fma(p, r, 8.3334519073e-3f);
    p = fma(p, r, 4.1665795894e-2f);
    p = fma(p, r, 1.6666665459e-1f);
    p = fma(p, r, 5.0000001201e-1f);
    const float y = add(fma(mul(r, r), p, r), 1.0f);
    const int n = int(k);
    if (n >= -126 && n <= 127) return mul(y, exp2i(n));
    const int h = n / 2;
    return mul(mul(y, exp2i(h)), exp2i(n - h));
}

// log(x): x = m * 2^e with m in [sqrt(1/2), sqrt(2)) (subnormals scaled by 2^23 first), f = m - 1, z = f^2,
// log(x) = f - z/2 + f z Q(f) + e ln2 with the Cephes logf coefficients
// Q = 3.3333331174e-1, -2.4999993993e-1, 2.0000714765e-1, -1.6668057665e-1, 1.4249322787e-1, -1.2420140846e-1,
// 1.1676998740e-1, -1.1514610310e-1, 7.0376836292e-2 (Horner from the last, fused) and ln2 = 0.693359375 -
// 2.12194440e-4. The order: y = fma(e, -2.12194440e-4, mul(mul(f, z), Q)), y = fma(z, -0.5, y), s = add(f, y),
// result = fma(e, 0.693359375, s). x < 0 or NaN -> NaN, 0 -> -inf, +inf -> +inf.
CERE_HD inline float log(float x) {
    if (x != x || x < 0.0f) return NAN;
    if (x == 0.0f) return -INFINITY;
    if (x == INFINITY) return x;
    int e = 0;
    uint32_t u = float_bits(x);
    if ((u >> 23) == 0) {  // subnormal
        u = float_bits(mul(x, 8388608.0f));
        e = -23;
    }
    e += int(u >> 23) - 126;
    float m = bits_float((u & 0x007fffffu) | 0x3f000000u);  // [0.5, 1)
    if (m < 0.707106781186547524f) {
        e -= 1;
        m = add(m, m);
    }
    const float f = sub(m, 1.0f);
    const float z = mul(f, f);
    float q = 7.0376836292e-2f;
    q = fma(q, f, -1.1514610310e-1f);
    q = fma(q, f, 1.1676998740e-1f);
    q = fma(q, f, -1.2420140846e-1f);
    q = fma(q, f, 1.4249322787e-1f);
    q = fma(q, f, -1.6668057665e-1f);
    q = fma(q, f, 2.0000714765e-1f);
    q = fma(q, f, -2.4999993993e-1f);
    q = fma(q, f, 3.3333331174e-1f);
    const float ef = float(e);
    float y = fma(ef, -2.12194440e-4f, mul(mul(f, z), q));
    y = fma(z, -0.5f, y);
    const float s = add(f, y);
    return fma(ef, 0.693359375f, s);
}

// tanh(x): |x| < 0.625: x + x z P(z), z = x^2, Cephes tanhf P = -3.33332819422e-1, 1.33314422036e-1,
// -5.37397155531e-2, 2.06390887954e-2, -5.70498872745e-3 (Horner from the last, fused; result fma(mul(x, z), P, x));
// 0.625 <= |x| < 9.0: sign(x) * (1 - 2 / (exp(2|x|) + 1)); |x| >= 9.0: sign(x) * 1. NaN -> NaN.
CERE_HD inline float tanh(float x) {
    if (x != x) return x;
    const float a = fabsf(x);
    float y;
    if (a < 0.625f) {
        const float z = mul(x, x);
        float p = -5.70498872745e-3f;
        p = fma(p, z, 2.06390887954e-2f);
        p = fma(p, z, -5.37397155531e-2f);
        p = fma(p, z, 1.33314422036e-1f);
        p = fma(p, z, -3.33332819422e-1f);
        return fma(mul(x, z), p, x);
    }
    if (a >= 9.0f) {
        y = 1.0f;
    } else {
        y = sub(1.0f, div(2.0f, add(ref::exp(add(a, a)), 1.0f)));
    }
    return x < 0.0f ? -y : y;
}

// sigmoid(x) = n / (1 + e): x < 0: e = exp(x), n = e; else e = exp(-x), n = 1 (so e <= 1). The rounding of 1 + e
// is carried: d = add(1, e), c = add(sub(1, d), e) (exact, Fast2Sum), q = div(n, d), r = fma(-q, d, n),
// r = fma(-q, c, r), result = add(q, div(r, d)). exp(+-inf) and NaN propagate (sigmoid(-inf) = 0, (+inf) = 1).
CERE_HD inline float sigmoid(float x) {
    const bool neg = x < 0.0f;
    const float e = ref::exp(neg ? x : -x);
    const float n = neg ? e : 1.0f;
    const float d = add(1.0f, e);
    const float c = add(sub(1.0f, d), e);
    const float q = div(n, d);
    float r = fma(-q, d, n);
    r = fma(-q, c, r);
    return add(q, div(r, d));
}

// ---- 3. Reductions (the order every inkling kernel reproduces) -------------------------------------------------
template <int L> CERE_HD inline float reduce_tree_(float *lane) {
    static_assert(L > 0 && (L & (L - 1)) == 0, "reduction width is a power of two");
    for (int off = L / 2; off >= 1; off /= 2)
        for (int i = 0; i < off; ++i) lane[i] = add(lane[i], lane[i + off]);
    return lane[0];
}

template <int L> CERE_HD inline float reduce_sum(const float *x, int n) {
    float lane[L];
    for (int t = 0; t < L; ++t) {
        float acc = 0.0f;
        for (int i = t; i < n; i += L) acc = add(acc, x[i]);
        lane[t] = acc;
    }
    return reduce_tree_<L>(lane);
}

template <int L> CERE_HD inline float reduce_dot(const float *a, const float *b, int n) {
    float lane[L];
    for (int t = 0; t < L; ++t) {
        float acc = 0.0f;
        for (int i = t; i < n; i += L) acc = fma(a[i], b[i], acc);
        lane[t] = acc;
    }
    return reduce_tree_<L>(lane);
}

template <int L> CERE_HD inline float reduce_max(const float *x, int n) {
    static_assert(L > 0 && (L & (L - 1)) == 0, "reduction width is a power of two");
    float lane[L];
    for (int t = 0; t < L; ++t) {
        float acc = -INFINITY;
        for (int i = t; i < n; i += L) acc = max(acc, x[i]);
        lane[t] = acc;
    }
    for (int off = L / 2; off >= 1; off /= 2)
        for (int i = 0; i < off; ++i) lane[i] = max(lane[i], lane[i + off]);
    return lane[0];
}

// Fixed reduction widths (contract item 3).
constexpr int kReduceHidden = 256;  // 2304-wide rows
constexpr int kReduceHead = 32;     // 128-wide rows
constexpr int kReduceRel = 16;      // d_rel 16
constexpr int kReduceRank = 64;     // rank-64 rows

// ---- 4. Formats ------------------------------------------------------------------------------------------------
CERE_HD inline float decode_bf16(uint16_t v) { return bits_float(uint32_t(v) << 16); }

// Round to nearest even; a NaN stays a (quiet) NaN.
CERE_HD inline uint16_t encode_bf16(float f) {
    const uint32_t u = float_bits(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x0040u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// FP4 E2M1 (low 4 bits): sign bit 3, magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
CERE_HD inline float decode_e2m1(uint8_t v) {
    const uint32_t m = v & 7u;
    const float mag = m < 2 ? 0.5f * float(m) : exp2i(int(m >> 1) - 1) * (1.0f + 0.5f * float(m & 1u));
    return (v & 8u) ? -mag : mag;
}

// FP8 E4M3 fn: bias 7, subnormals m * 2^-9, 0x7f/0xff NaN, no infinity (max 448).
CERE_HD inline float decode_e4m3(uint8_t v) {
    const uint32_t e = (v >> 3) & 15u, m = v & 7u;
    float mag;
    if (e == 15u && m == 7u) return NAN;
    if (e == 0u) mag = float(m) * exp2i(-9);
    else mag = bits_float(((e + 120u) << 23) | (m << 20));
    return (v & 0x80u) ? -mag : mag;
}

// UE8M0 (MXFP8 block scale): 2^(v - 127); 0xff NaN; v = 0 is 2^-127 (a float subnormal).
CERE_HD inline float decode_ue8m0(uint8_t v) {
    if (v == 0xffu) return NAN;
    if (v == 0u) return bits_float(0x00400000u);
    return bits_float(uint32_t(v) << 23);
}

}  // namespace ref
