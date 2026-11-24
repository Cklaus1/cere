// K5 decode kernel-test entry point (libcere_ktest.so), the contract of executor/tests/gpu/test_kernels_k5_decode.py:
// device pointers to contiguous row-major buffers, a float scalar, the stream last; 0 or a cere_status.
#include "cere_executor.h"
#include "kernels/k5_gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

using bf16 = __nv_bfloat16;

cere_status status_of(cudaError_t err) {
    if (err == cudaSuccess) {
        return CERE_OK;
    }
    return err == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
}

}  // namespace

#pragma GCC visibility push(default)
extern "C" {

int cere_ktest_gdn_decode_step(const void *q, const void *k, const void *v, const void *a, const void *b,
                               const void *A_log, const void *dt_bias, void *state, void *o, float scale,
                               cudaStream_t stream) {
    cere::k5::DecodeArgs args{};
    args.q = static_cast<const bf16 *>(q);
    args.k = static_cast<const bf16 *>(k);
    args.v = static_cast<const bf16 *>(v);
    args.a = static_cast<const bf16 *>(a);
    args.b = static_cast<const bf16 *>(b);
    args.A_log = static_cast<const float *>(A_log);
    args.dt_bias = static_cast<const bf16 *>(dt_bias);
    args.state = static_cast<bf16 *>(state);
    args.o = static_cast<bf16 *>(o);
    args.scale = scale;
    return status_of(cere::k5::delta_rule_decode(args, stream));
}

}  // extern "C"
#pragma GCC visibility pop
