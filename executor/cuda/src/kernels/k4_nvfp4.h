// K4 NVFP4 PLE row decode (fp4local design note D8, D14, FL-13): one 160-wide PLE table row of the FP4-Local
// checkpoint, stored as 80 bytes of packed E2M1 codes (element 2j in the low nibble of byte j, 2j + 1 in the high
// nibble), 10 E4M3 block scales (one per 16 consecutive elements) and one F32 global scale for the whole table
// (`ngram_embedding.weight_scale_2`). Element i of a row is
//   bf16_rne(f32(e2m1(code_i) * e4m3(scale_{i / 16})) * global)
// with the first product exact in F32 (at most 5 significant bits, magnitudes 2^-10 .. 2,688) and one F32 rounding
// for the global, then round-to-nearest-even to BF16, as K4's FP8 path rounds its scaled rows. An E4M3 NaN byte (0x7F,
// 0xFF) gives a NaN element. Header-only and CUDA-free outside nvcc, so the host test checks the same functions the
// kernel runs (executor/tests/host/test_k4_nvfp4.cpp) against an independent CPU reference; the GPU suite checks the
// kernel against a second one (executor/tests/gpu/test_kernels_k4n_nvfp4.py).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define CERE_K4N_FN __host__ __device__ __forceinline__
#else
#define CERE_K4N_FN inline
#endif

namespace cere::k4::nvfp4 {

constexpr int kRowElements = 160;                  // k4::kHeadDim: one n-gram head's slice of ple_embed_dim
constexpr int kBlock = 16;                         // elements per E4M3 block scale
constexpr int kRowBytes = kRowElements / 2;        // 80 packed bytes per row
constexpr int kRowScales = kRowElements / kBlock;  // 10 block scales per row
constexpr int kBlockBytes = kBlock / 2;            // 8 packed bytes per block

CERE_K4N_FN float from_bits(uint32_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

CERE_K4N_FN uint32_t to_bits(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    return bits;
#endif
}

// E2M1 code (low 4 bits of `code`): sign bit 3, exponent bits 2-1 (bias 1), mantissa bit 0; magnitudes
// {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Code 8 is -0.
CERE_K4N_FN float e2m1(uint32_t code) {
    const uint32_t sign = (code & 0x8u) << 28;
    const uint32_t exp = (code >> 1) & 0x3u;
    const uint32_t man = code & 0x1u;
    // exp 0: subnormal man * 0.5 (0 or 2^-1); else (1 + man / 2) * 2^(exp - 1).
    const uint32_t mag = exp == 0 ? (man == 0 ? 0u : 0x3F000000u) : ((exp - 1u + 127u) << 23) | (man << 22);
    return from_bits(sign | mag);
}

// E4M3 (OCP `e4m3fn`) byte: sign bit 7, exponent bits 6-3 (bias 7), mantissa bits 2-0; no infinities, 0x7F/0xFF NaN,
// subnormals man * 2^-9, largest 448. Exact in F32.
CERE_K4N_FN float e4m3(uint32_t byte) {
    const uint32_t sign = (byte & 0x80u) << 24;
    const uint32_t exp = (byte >> 3) & 0xFu;
    const uint32_t man = byte & 0x7u;
    if (exp == 0xFu && man == 0x7u) return from_bits(sign | 0x7FC00000u);
    if (exp == 0) return from_bits(sign | to_bits(float(man) * 0.001953125f));  // man * 2^-9, exact
    return from_bits(sign | ((exp - 7u + 127u) << 23) | (man << 20));
}

// The F32 value of one element before the BF16 rounding, given its block's decoded scale: (E2M1 x E4M3, exact) x
// global (one F32 rounding). Two products, no sum, so no FMA contraction can change it.
CERE_K4N_FN float scaled(uint32_t code, float scale, float global) {
    const float block = e2m1(code) * scale;
    return block * global;
}

// The same from the block's E4M3 byte.
CERE_K4N_FN float element(uint32_t code, uint32_t scale_byte, float global) {
    return scaled(code, e4m3(scale_byte), global);
}

// The E2M1 code of element `i` (0 <= i < 160) of a packed row.
CERE_K4N_FN uint32_t code_of(const uint8_t *packed_row, int i) {
    const uint32_t byte = packed_row[i / 2];
    return (i % 2 == 0) ? (byte & 0xFu) : (byte >> 4);
}

// Round-to-nearest-even F32 -> BF16 bits; any NaN gives the canonical 0x7FFF (the bits `cvt.rn.bf16.f32` writes).
// Host side of the kernel's `__float2bfloat16_rn`.
CERE_K4N_FN uint16_t bf16_rne_bits(float f) {
    const uint32_t bits = to_bits(f);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FFFu;
    return uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// One row decoded on the host: out[i] = BF16 bits of element i. The CPU dequant reference of the K4 NVFP4 gather.
inline void decode_row(const uint8_t *packed_row, const uint8_t *scales_row, float global, uint16_t *out) {
    for (int i = 0; i < kRowElements; ++i) {
        out[i] = bf16_rne_bits(element(code_of(packed_row, i), scales_row[i / kBlock], global));
    }
}

}  // namespace cere::k4::nvfp4

#undef CERE_K4N_FN
