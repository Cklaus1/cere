// K0 (m22 design note K0, Q14): NVFP4 primitives of the checkpoint layout, host launch API.
//
// Layout (proven by X2 on the real checkpoint): a weight [n, k] is `weight` uint8 [n, k/2], row-major, element 2j in
// the low nibble and 2j+1 in the high nibble of byte j (E2M1 codes: bit 3 sign, bits 0-2 index into
// {0, 0.5, 1, 1.5, 2, 3, 4, 6}); `weight_scale` FP8 E4M3 [n, k/16], row-major (the linear layout): one scale per 16
// consecutive k elements; `weight_scale_2` one F32 per tensor. Value = e2m1 * (float(fp8 scale) * weight_scale_2).
//
// The activation quantizer reproduces FlashInfer `fp4_quantize(x, global_scale, 16, sf_use_ue8m0=False,
// is_sf_swizzled_layout=False)` bit for bit (codes and scales); `global_scale` is the FP32 value that call receives,
// i.e. 1 / max(input_scale) of the layer and GEMM (the reference's one scalar per layer and GEMM).
#pragma once

#include "cere_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cere::kernels {

// out[r][j] = e2m1(code(r, j)) * (float(scale[r][j / 16]) * global_scale), F32 [n, k] row-major.
// k must be a positive multiple of 16; CERE_ERR_ARGUMENT otherwise, CERE_ERR_INTERNAL on a launch failure.
cere_status nvfp4_dequant(const uint8_t *weight, const uint8_t *scale, float global_scale, float *out, int64_t n,
                          int64_t k, cudaStream_t stream);

// codes uint8 [m, k/2] (element 2j in the low nibble), scales FP8 E4M3 [m, k/16] row-major, from BF16 x [m, k].
// k must be a positive multiple of 16; CERE_ERR_ARGUMENT otherwise, CERE_ERR_INTERNAL on a launch failure.
cere_status fp4_quantize(const __nv_bfloat16 *x, float global_scale, uint8_t *codes, uint8_t *scales, int64_t m,
                         int64_t k, cudaStream_t stream);

}  // namespace cere::kernels
