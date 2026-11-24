// K2 MXFP8 -> BF16 operand conversion (device only; contract in k2_mxfp8.h): shared by k2m::dequantize and the small-M
// kernel's MXFP8 operand (k2_small_m.cu), so both produce the same bits, the exact upcast fp4local design note D4.
#pragma once

#include <cstdint>

namespace cere::kernels::k2m {

// The exponent add of scale byte s (k2_mxfp8.h kMinScale..kMaxScale) in BF16 exponent units: (s - 15) << 7, the F16
// bias (15) out and the scale's 2^(s - 127) in (BF16 and UE8M0 share the bias 127). Two's complement, so negative for
// s < 15; bf16x2 adds it per half without a carry between the halves going wrong (below).
__device__ __forceinline__ uint32_t exponent_add(uint32_t s) {
    return static_cast<uint32_t>((static_cast<int32_t>(s) - 15) * 128);
}

// Four E4M3 bytes (byte i = element i) -> two F16x2 words {elements 0, 1} and {2, 3} (element 2j in the low half).
// Exact: every E4M3 value, subnormals included, is a normal F16 or +-0.
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t v, uint32_t &lo, uint32_t &hi) {
    asm("{\n"
        ".reg .b16 a, b;\n"
        "mov.b32 {a, b}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 %0, a;\n"
        "cvt.rn.f16x2.e4m3x2 %1, b;\n"
        "}"
        : "=r"(lo), "=r"(hi)
        : "r"(v));
}

// F16x2 word h of E4M3 values -> the BF16x2 word of those values times 2^(s - 127), add = exponent_add(s). Per half:
// an E4M3 value's F16 has at most 3 mantissa bits below the leading one (the top 3 of 10), so (bits & 0x7FFF) >> 3 is
// its BF16 magnitude with the F16 exponent field; adding (s - 15) << 7 to a nonzero half re-biases it to a BF16
// exponent in [6 + 10 - 15, 23 + 246 - 15] = [1, 254] (F16 fields 6..23 for 2^-9..448): a normal BF16, the exact
// product. A zero half (+-0) gets no add and keeps its sign. The 32-bit add of nz * add, nz in {0, 1} per half, is the
// per-half sum modulo 2^32, and each half's sum lies in [0, 0x7FFF], so the packed result is exact whatever the
// borrows of a negative add.
__device__ __forceinline__ uint32_t bf16x2(uint32_t h, uint32_t add) {
    const uint32_t mag = h & 0x7FFF7FFFu;
    const uint32_t nz = ((mag + 0x7FFF7FFFu) >> 15) & 0x00010001u;  // bit 0 / 16: that half is nonzero
    return (h & 0x80008000u) | (((mag >> 3) & 0x0FFF0FFFu) + nz * add);
}

// Eight E4M3 bytes (v.x bytes 0..3 = elements 0..3, v.y = 4..7) with one scale -> eight BF16 values as a uint4
// (element 2j in the low half of word j), the operand layout of a BF16 16-byte load.
__device__ __forceinline__ uint4 bf16x8(uint2 v, uint32_t add) {
    uint32_t h0, h1, h2, h3;
    e4m3x4_to_f16x4(v.x, h0, h1);
    e4m3x4_to_f16x4(v.y, h2, h3);
    return make_uint4(bf16x2(h0, add), bf16x2(h1, add), bf16x2(h2, add), bf16x2(h3, add));
}

}  // namespace cere::kernels::k2m
