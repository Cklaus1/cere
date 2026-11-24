// K0 (m22 design note K0, Q14): NVFP4 device helpers shared by the K0 kernels and the K7 MoE (CUDA sources only).
//
// math_reference: flashinfer
// FlashInfer 0.6.17 csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh (`cvt_warp_fp16_to_fp4`, its
// default fast-math E4M3 branch, `fp32_vec_to_e2m1`, `reciprocal_approximate_ftz`) and
// csrc/nv_internal/tensorrt_llm/thop/fp4Op.cpp (`E2M1AndUFP8SFScaleToFloatV2`, `e2M1ToFloat`): the arithmetic is
// restated here operation by operation so the results are bit-identical; no code is copied.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace cere::kernels::nvfp4 {

// E2M1 code -> float: bit 3 sign, bits 0-2 the magnitude index. -0.0 for code 8, as the reference's `-result`.
__device__ __forceinline__ float e2m1_to_float(uint32_t code) {
    constexpr float kMagnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kMagnitude[code & 7u];
    return (code & 8u) ? -v : v;
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 v;
    v.__x = bits;
    return static_cast<float>(v);
}

// rcp.approx.ftz.f32, the reference's `reciprocal_approximate_ftz`.
__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// Eight floats -> eight E2M1 codes (cvt.rn.satfinite.e2m1x2.f32, element 2j in the low nibble of byte j), the
// reference's `fp32_vec_to_e2m1` operand order: the instruction puts its second source in the low nibble.
__device__ __forceinline__ uint32_t e2m1x8(const float (&a)[8]) {
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "f"(a[4]), "f"(a[5]), "f"(a[6]), "f"(a[7]));
    return val;
}

// Quantizes one block of 16 BF16 values (given as float, which is exact) with the per-tensor scale `global_scale`.
// Returns the 8 code bytes as two words (element 2j in the low nibble of byte j) and the FP8 E4M3 block scale.
// Mirrors cvt_warp_fp16_to_fp4<bf16, 16, *, UE8M0_SF=false, DISABLE_FP4_QUANT_FAST_MATH=false> without 4over6:
//   vecMax       = max |x|                      (exact; the max of BF16 values)
//   SFValue      = global_scale * (vecMax * rcp(6))
//   sf           = e4m3(SFValue) (satfinite, round to nearest even)
//   outputScale  = vecMax != 0 ? rcp(float(sf) * rcp(global_scale)) : 0
//   code_i       = e2m1(x_i * outputScale)      (cvt.rn.satfinite)
__device__ __forceinline__ void quantize_block16(const float (&x)[16], float global_scale, uint32_t (&codes)[2],
                                                 uint8_t *scale) {
    float vec_max = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        vec_max = fmaxf(vec_max, fabsf(x[i]));
    }
    float sf_value = global_scale * (vec_max * rcp_approx_ftz(6.0f));
    const __nv_fp8_e4m3 sf(sf_value);
    *scale = sf.__x;
    sf_value = static_cast<float>(sf);
    const float output_scale = vec_max != 0.0f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(global_scale)) : 0.0f;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        float scaled[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled[i] = x[w * 8 + i] * output_scale;
        }
        codes[w] = e2m1x8(scaled);
    }
}

// Four E2M1 codes (the low 16 bits of `v`, code i in bits 4i..4i+3) -> four int8 values 2 * e2m1 (byte i), exact
// integers in {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}.
__device__ __forceinline__ uint32_t e2m1x4_to_s8x4(uint32_t v) {
    const uint32_t index = v & 0x7777u;                                    // magnitude index of each code
    const uint32_t positive = __byte_perm(0x03020100u, 0x0C080604u, index);  // {0, 1, 2, 3, 4, 6, 8, 12}
    const uint32_t negative = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, index);  // the same, negated (code 8 -> 0)
    const uint32_t spread = __byte_perm(v & 0x0F0Fu, (v >> 4) & 0x0F0Fu, 0x5140u);  // code i in byte i
    const uint32_t sign = ((spread >> 3) & 0x01010101u) * 0xFFu;                   // 0xFF in negative bytes
    return (positive & ~sign) | (negative & sign);
}

}  // namespace cere::kernels::nvfp4
