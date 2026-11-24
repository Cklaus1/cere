#pragma once
// Inkling 2 Mini (inkling design note 3.4, 5, G1): the CPU reference of the routed-expert activation quantizers,
// SGLang's routed_quant.cuh semantics operation by operation (the checkpoint's declared activation quantization, arm
// Q, decision D-I1). Header-only C++17 on common_ref.h (contract items 1, 4): every multiply, add and divide below is
// one IEEE binary32 operation rounded to nearest even, no fused operations, no flush to zero.
//
// Hardware conversions, emulated exactly (the CUDA kernels use the PTX instructions; the G1 ktest sweeps both):
//   encode_e4m3_satfinite(v) = cvt.rn.satfinite.e4m3x2.f32: round to nearest even on the E4M3 fn grid (subnormal step
//     2^-9), |v| >= 448 (incl. inf) -> +-448 (0x7e), the sign of v kept (-0 -> 0x80); NaN -> 0x7f (out of contract).
//   encode_e2m1_satfinite(v) = cvt.rn.satfinite.e2m1x2.f32: round to nearest even on {0, .5, 1, 1.5, 2, 3, 4, 6}
//     (ties 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4), |v| > 5 (incl. inf) -> 6, sign bit
//     3 = sign of v (-0 -> code 8); NaN is out of contract.
//
// NVFP4 row quantizer (per-token amax, block 16, E4M3 block scales, fouroversix_mae; routed_quant.cuh:43-82, 193-247):
//   amax    = max |x_i| over the row (exact; order-free)
//   encoder = fminf(div(1536, amax), FLT_MAX); decoder = div(1, encoder); outer = mul(amax, float(1.0 / 1536.0))
//   per block b of 16 (block_max = max |x_i| in the block):
//     s6 = e4m3(fminf(mul(block_max, mul(encoder, float(1.0 / 6.0))), 448))
//     s4 = e4m3(fminf(mul(block_max, mul(encoder, 0.25)), 448))
//     inv6 = fminf(div(1, mul(s6, decoder)), FLT_MAX); inv4 likewise
//     r6_i = e2m1(mul(x_i, inv6)); err6_i = |sub(mul(mul(r6_i, s6), outer), x_i)|; r4 / err4 likewise
//     err = pairwise sum over the 16 in index order: level 1 adds (0,1), (2,3), ..., then (01,23), ... (4 levels)
//     choose 4 iff err4 < err6 (strict; NaN compares false -> 6); the block's codes and scale byte from the choice.
//   Storage (CERE's, not SGLang's swizzled SFA): codes [k/2] (element 2j in the low nibble of byte j, the checkpoint's
//   order), scales [k/16] (row-major), amax FP32. The GEMM's outer factor uses amax (moe_ref.h).
//
// MXFP8 row quantizer (block 32, UE8M0; routed_quant.cuh:329-354, ops/moe/inkling2_moe.py _mxfp8_block):
//   amax = max |x_i| over the block; raw = mul(amax, float(1.0 / 448.0))
//   exponent = ((bits(raw) + 0x7fffff) & 0x7f800000) >> 23 (raw rounded up to a power of two);
//   exponent = raw <= 2^-127 ? 0 : min(exponent, 254); raw NaN -> 255
//   scale = fmaxf(2^(exponent - 127) as bits exponent << 23, 2^-127)
//   inverse = exponent == 0 ? 2^127 : div(1, scale); exponent == 255 -> 0
//   value_i = e4m3(fminf(fmaxf(mul(x_i, inverse), -448), 448)); the exponent byte is the UE8M0 scale.
//   Storage: values [k] (E4M3 bytes), exponents [k/32].
#include <cstdint>

#include "common_ref.h"

namespace ref {

constexpr float kFltMax = 3.4028234663852886e38f;
constexpr int kNvfp4Block = 16;
constexpr int kMxfp8Block = 32;

CERE_HD inline uint8_t encode_e4m3_satfinite(float v) {
    if (v != v) return 0x7fu;
    const uint8_t sign = uint8_t((float_bits(v) >> 24) & 0x80u);
    const float a = fabsf(v);
    if (a >= 448.0f) return uint8_t(sign | 0x7eu);
    if (a < 0.015625f) {  // below 2^-6: the subnormal grid m * 2^-9, m = 0..8 (8 is 2^-6, code 0x08)
        const float q = sub(add(mul(a, 512.0f), 12582912.0f), 12582912.0f);  // RNE to an integer (1.5 * 2^23 shifter)
        return uint8_t(sign | uint8_t(q));
    }
    uint32_t u = float_bits(a);
    u = u + 0x7ffffu + ((u >> 20) & 1u);  // RNE to 3 mantissa bits (may carry into the exponent)
    const int e = int(u >> 23) - 127;     // -6 .. 8
    return uint8_t(sign | uint8_t((uint32_t(e + 7) << 3) | ((u >> 20) & 7u)));
}

CERE_HD inline uint8_t encode_e2m1_satfinite(float v) {
    const uint8_t sign = uint8_t((float_bits(v) >> 28) & 8u);
    const float a = fabsf(v);
    uint8_t m;
    if (a <= 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a <= 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a <= 2.5f) m = 4;
    else if (a < 3.5f) m = 5;
    else if (a <= 5.0f) m = 6;
    else m = 7;
    return uint8_t(sign | m);
}

// One NVFP4 block of 16 values with the row's amax and the block's max |x|: codes [8] (element 2j low nibble of
// byte j), the E4M3 scale byte; returns true when the /4 scale was chosen. `fouroversix` false is the always-/6 rule
// (not the checkpoint's; kept for the ktest's comparison only, never used by the model path).
CERE_HD inline bool nvfp4_block(const float *x, float amax, float block_max, uint8_t *codes, uint8_t *scale,
                                bool fouroversix = true) {
    const float encoder = fminf(div(1536.0f, amax), kFltMax);
    const float decoder = div(1.0f, encoder);
    const float outer = mul(amax, float(1.0 / 1536.0));
    const uint8_t b6 = encode_e4m3_satfinite(fminf(mul(block_max, mul(encoder, float(1.0 / 6.0))), 448.0f));
    const uint8_t b4 = encode_e4m3_satfinite(fminf(mul(block_max, mul(encoder, 0.25f)), 448.0f));
    const float s6 = decode_e4m3(b6), s4 = decode_e4m3(b4);
    const float inv6 = fminf(div(1.0f, mul(s6, decoder)), kFltMax);
    const float inv4 = fminf(div(1.0f, mul(s4, decoder)), kFltMax);
    uint8_t c6[16], c4[16];
    float e6[16], e4[16];
    for (int i = 0; i < 16; ++i) {
        c6[i] = encode_e2m1_satfinite(mul(x[i], inv6));
        e6[i] = fabsf(sub(mul(mul(decode_e2m1(c6[i]), s6), outer), x[i]));
        c4[i] = encode_e2m1_satfinite(mul(x[i], inv4));
        e4[i] = fabsf(sub(mul(mul(decode_e2m1(c4[i]), s4), outer), x[i]));
    }
    for (int w = 8; w >= 1; w /= 2) {
        for (int i = 0; i < w; ++i) {
            e6[i] = add(e6[2 * i], e6[2 * i + 1]);
            e4[i] = add(e4[2 * i], e4[2 * i + 1]);
        }
    }
    const bool choose4 = fouroversix && e4[0] < e6[0];
    const uint8_t *c = choose4 ? c4 : c6;
    for (int j = 0; j < 8; ++j) codes[j] = uint8_t(c[2 * j] | (c[2 * j + 1] << 4));
    *scale = choose4 ? b4 : b6;
    return choose4;
}

// One row of k values (k a multiple of 16): codes [k/2], scales [k/16]; returns the row's amax. *chose4 (nullable)
// counts the blocks that took the /4 scale.
inline float nvfp4_quantize_row(const float *x, int k, uint8_t *codes, uint8_t *scales, int *chose4 = nullptr) {
    float amax = 0.0f;
    for (int i = 0; i < k; ++i) amax = fmaxf(amax, fabsf(x[i]));
    int n4 = 0;
    for (int b = 0; b < k / 16; ++b) {
        float block_max = 0.0f;
        for (int i = 0; i < 16; ++i) block_max = fmaxf(block_max, fabsf(x[16 * b + i]));
        n4 += nvfp4_block(x + 16 * b, amax, block_max, codes + 8 * b, scales + b) ? 1 : 0;
    }
    if (chose4 != nullptr) *chose4 = n4;
    return amax;
}

// One MXFP8 block of 32 values: values [32] E4M3 bytes; returns the UE8M0 exponent byte.
CERE_HD inline uint8_t mxfp8_block(const float *x, uint8_t *values) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float min_scale = bits_float(0x00400000u);
    const float raw = mul(amax, float(1.0 / 448.0));
    int exponent = int(((float_bits(raw) + 0x007fffffu) & 0x7f800000u) >> 23);
    exponent = raw <= min_scale ? 0 : (exponent < 254 ? exponent : 254);
    if (raw != raw) exponent = 255;
    const float scale = fmaxf(bits_float(uint32_t(exponent) << 23), min_scale);
    float inverse = exponent == 0 ? bits_float(0x7f000000u) : div(1.0f, scale);
    if (exponent == 255) inverse = 0.0f;
    for (int i = 0; i < 32; ++i) values[i] = encode_e4m3_satfinite(fminf(fmaxf(mul(x[i], inverse), -448.0f), 448.0f));
    return uint8_t(exponent);
}

// One row of k values (k a multiple of 32): values [k], exponents [k/32].
inline void mxfp8_quantize_row(const float *x, int k, uint8_t *values, uint8_t *exponents) {
    for (int b = 0; b < k / 32; ++b) exponents[b] = mxfp8_block(x + 32 * b, values + 32 * b);
}

}  // namespace ref
