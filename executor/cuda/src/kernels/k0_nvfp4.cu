// K0 (m22 design note K0, Q14): NVFP4 raw-byte dequant and the FP4 activation quantizer. Contract in k0_nvfp4.h.
//
// math_reference: flashinfer
// FlashInfer 0.6.17 csrc/nv_internal/tensorrt_llm/thop/fp4Op.cpp (`E2M1AndUFP8SFScaleToFloatV2`, linear layout) and
// csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh (`quantize_with_block_size`, linear SF layout); the
// per-element arithmetic is in k0_nvfp4.cuh.
#include "kernels/k0_nvfp4.h"

#include "kernels/k0_nvfp4.cuh"

namespace cere::kernels {
namespace {

constexpr int kThreads = 256;

// One thread per 16-element block of one row: 8 code bytes and one scale.
__global__ void __launch_bounds__(kThreads)
    dequant_kernel(const uint8_t *__restrict__ weight, const uint8_t *__restrict__ scale, float global_scale,
                   float *__restrict__ out, int64_t blocks) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (b >= blocks) {
        return;
    }
    // The reference's product order: (float(fp8 scale) * global_scale), then e2m1 * that.
    const float block_scale = nvfp4::e4m3_to_float(scale[b]) * global_scale;
    const uint8_t *codes = weight + b * 8;
    float *dst = out + b * 16;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t byte = codes[j];
        dst[2 * j] = nvfp4::e2m1_to_float(byte & 0xFu) * block_scale;
        dst[2 * j + 1] = nvfp4::e2m1_to_float(byte >> 4) * block_scale;
    }
}

// One thread per 16-element block of one row.
__global__ void __launch_bounds__(kThreads)
    quantize_kernel(const __nv_bfloat16 *__restrict__ x, float global_scale, uint8_t *__restrict__ codes,
                    uint8_t *__restrict__ scales, int64_t blocks) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (b >= blocks) {
        return;
    }
    float v[16];
    const __nv_bfloat16 *src = x + b * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = __bfloat162float(src[i]);
    }
    uint32_t words[2];
    nvfp4::quantize_block16(v, global_scale, words, scales + b);
    uint8_t *dst = codes + b * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        dst[j] = static_cast<uint8_t>(words[j / 4] >> (8 * (j % 4)));
    }
}

bool shape_ok(int64_t rows, int64_t k) { return rows > 0 && k > 0 && k % 16 == 0 && rows <= (INT64_MAX / k); }

unsigned grid_for(int64_t blocks) { return static_cast<unsigned>((blocks + kThreads - 1) / kThreads); }

}  // namespace

cere_status nvfp4_dequant(const uint8_t *weight, const uint8_t *scale, float global_scale, float *out, int64_t n,
                          int64_t k, cudaStream_t stream) {
    if (weight == nullptr || scale == nullptr || out == nullptr || !shape_ok(n, k)) {
        return CERE_ERR_ARGUMENT;
    }
    const int64_t blocks = n * (k / 16);
    dequant_kernel<<<grid_for(blocks), kThreads, 0, stream>>>(weight, scale, global_scale, out, blocks);
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status fp4_quantize(const __nv_bfloat16 *x, float global_scale, uint8_t *codes, uint8_t *scales, int64_t m,
                         int64_t k, cudaStream_t stream) {
    if (x == nullptr || codes == nullptr || scales == nullptr || !shape_ok(m, k)) {
        return CERE_ERR_ARGUMENT;
    }
    const int64_t blocks = m * (k / 16);
    quantize_kernel<<<grid_for(blocks), kThreads, 0, stream>>>(x, global_scale, codes, scales, blocks);
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::kernels
