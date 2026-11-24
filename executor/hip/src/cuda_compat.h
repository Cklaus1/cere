// AmdAmd PAR-6: CUDA intrinsics hipify-perl leaves untranslated, force-included (-include) into every generated HIP
// kernel source. The CUDA sources stay untouched (they are lock_native build inputs).
#pragma once
#include <hip/hip_bf16.h>

// CUDA's float -> BF16 conversions (cvt.rn.bf16.f32): round to nearest even, as HIP's __float2bfloat16, and every
// NaN the canonical 0x7FFF (HIP keeps the sign and gives 0x7FC0 / 0xFFC0; k4n_test found it on NaN-scale PLE rows).
// gen_hip.py maps the generated sources' __float2bfloat16 here too.
__host__ __device__ inline __hip_bfloat16 __cere_float2bfloat16(float f) {
    if (f != f) {
        __hip_bfloat16_raw r;
        r.x = 0x7FFF;
        return __hip_bfloat16(r);
    }
    return __float2bfloat16(f);
}
__host__ __device__ inline __hip_bfloat16 __float2bfloat16_rn(float f) { return __cere_float2bfloat16(f); }
__host__ __device__ inline __hip_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return __halves2bfloat162(__cere_float2bfloat16(a), __cere_float2bfloat16(b));
}

// __ldg (read-only cache load) for every type: a plain load (gfx11 has no separate read-only path to request).
template <class T> __device__ inline T __cere_ldg(const T *p) { return *p; }
// The bf16 bit pattern held in a 16-bit register as float (what PTX fma.rn.f32.bf16 widens; exact).
__device__ inline float __cere_bf16f(unsigned short h) { return __uint_as_float(static_cast<unsigned>(h) << 16); }

// CUDA's __fadd_rn / __fsub_rn / __fmul_rn are never contracted into an FMA (the CUDA docs' guarantee, which the
// double-float sums of lora.cu and the "one fp32 multiply" of k12/k5 rely on). HIP's versions are plain operators that
// -ffp-contract=fast may fuse, so gen_hip.py maps them here: contraction off inside, so the operation carries no
// contract flag and cannot be fused with its neighbours after inlining.
__device__ inline float __cere_fadd_rn(float a, float b) {
#pragma clang fp contract(off)
    return a + b;
}
__device__ inline float __cere_fsub_rn(float a, float b) {
#pragma clang fp contract(off)
    return a - b;
}
__device__ inline float __cere_fmul_rn(float a, float b) {
#pragma clang fp contract(off)
    return a * b;
}
__device__ inline double __cere_dadd_rn(double a, double b) {
#pragma clang fp contract(off)
    return a + b;
}
__device__ inline double __cere_dmul_rn(double a, double b) {
#pragma clang fp contract(off)
    return a * b;
}

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
// CUDA's __dp4a (signed): the sum of the four int8 x int8 products plus c, exact (v_dot4_i32_iu8 on gfx11, both
// operands signed, no clamp).
__device__ __forceinline__ int __dp4a(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
#endif

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
// CUDA's __byte_perm(x, y, s) (default mode: output byte i = byte (s >> 4 i) & 7 of the 8 bytes {y:x}, x low) as one
// v_perm_b32: HIP's own __byte_perm indexes a byte array (scratch memory on gfx11; the hot E2M1 unpack of K7 / K13
// ran from scratch). v_perm takes bytes 0-3 from its second operand, so (y, x) and the nibbles spread to bytes.
__device__ __forceinline__ unsigned int __cere_byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    const unsigned int sel = (s & 7u) | ((s & 0x70u) << 4) | ((s & 0x700u) << 8) | ((s & 0x7000u) << 12);
    return __builtin_amdgcn_perm(y, x, sel);
}
#endif

