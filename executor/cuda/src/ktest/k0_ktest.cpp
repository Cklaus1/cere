// K0 kernel-test entry points (executor/tests/gpu/test_layout_k0_nvfp4.py, suite `layout`, run by X2): NVFP4
// raw-byte dequant and the FP4 activation quantizer. Device pointers, int64 sizes, the stream last; 0 = success,
// else a cere_status.
#include "kernels/k0_nvfp4.h"

extern "C" __attribute__((visibility("default"))) int cere_ktest_nvfp4_dequant(const void *weight, const void *scale,
                                                                               float global_scale, void *out,
                                                                               int64_t n, int64_t k,
                                                                               cudaStream_t stream) {
    return cere::kernels::nvfp4_dequant(static_cast<const uint8_t *>(weight), static_cast<const uint8_t *>(scale),
                                        global_scale, static_cast<float *>(out), n, k, stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_fp4_quantize(const void *x, float global_scale,
                                                                              void *codes, void *scales, int64_t m,
                                                                              int64_t k, cudaStream_t stream) {
    return cere::kernels::fp4_quantize(static_cast<const __nv_bfloat16 *>(x), global_scale,
                                       static_cast<uint8_t *>(codes), static_cast<uint8_t *>(scales), m, k, stream);
}
