// K5 chunked-prefill kernel-test entry point (libcere_ktest.so), the contract of
// executor/tests/gpu/test_kernels_k5_chunk.py: device pointers to contiguous row-major buffers, int64_t counts, float
// scalars, the stream last; 0 or a cere_status. The entry point allocates the workspace per call and synchronizes the
// stream before freeing it (the executor carves it from its layer workspace instead).
#include "cere_executor.h"
#include "kernels/k5_gdn_chunk.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;

cere_status status_of(cudaError_t err) {
    if (err == cudaSuccess) {
        return CERE_OK;
    }
    return err == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
}

// One delta_rule_chunk call on a workspace of its own, synchronized before the workspace is freed.
int run_chunk(const cere::k5::ChunkArgs &args, cudaStream_t stream) {
    const size_t bytes = cere::k5::chunk_workspace_bytes(args.tokens);
    if (bytes == 0) {
        return CERE_ERR_ARGUMENT;
    }
    void *workspace = nullptr;
    cudaError_t err = cudaMalloc(&workspace, bytes);
    if (err != cudaSuccess) {
        return status_of(err);
    }
    err = cere::k5::delta_rule_chunk(args, workspace, stream);
    const cudaError_t sync = cudaStreamSynchronize(stream);
    const cudaError_t freed = cudaFree(workspace);
    if (err == cudaSuccess) {
        err = sync != cudaSuccess ? sync : freed;
    }
    return status_of(err);
}

cere::k5::ChunkArgs chunk_args(const void *q, const void *k, const void *v, const void *g, const void *beta,
                               void *state, void *o, int64_t tokens, float scale) {
    cere::k5::ChunkArgs args{};
    args.q = static_cast<const bf16 *>(q);
    args.k = static_cast<const bf16 *>(k);
    args.v = static_cast<const bf16 *>(v);
    args.g = static_cast<const float *>(g);
    args.beta = static_cast<const float *>(beta);
    args.state = static_cast<bf16 *>(state);
    args.o = static_cast<bf16 *>(o);
    args.tokens = tokens;
    args.scale = scale;
    return args;
}

}  // namespace

#pragma GCC visibility push(default)
extern "C" {

int cere_ktest_gdn_chunk_prefill(const void *q, const void *k, const void *v, const void *g, const void *beta,
                                 void *state, void *o, int64_t tokens, float scale, cudaStream_t stream) {
    return run_chunk(chunk_args(q, k, v, g, beta, state, o, tokens, scale), stream);
}

// m210 D15⁗′: the same call with the FP32 state arguments (ChunkArgs::state_in32, capture32, capture_chunk; null
// pointers and -1 leave them off).
int cere_ktest_gdn_chunk_prefill_fp32(const void *q, const void *k, const void *v, const void *g, const void *beta,
                                      void *state, void *o, int64_t tokens, float scale, const void *state_in32,
                                      void *capture32, int64_t capture_chunk, cudaStream_t stream) {
    cere::k5::ChunkArgs args = chunk_args(q, k, v, g, beta, state, o, tokens, scale);
    args.state_in32 = static_cast<const float *>(state_in32);
    args.capture32 = static_cast<float *>(capture32);
    args.capture_chunk = capture_chunk;
    return run_chunk(args, stream);
}

}  // extern "C"
#pragma GCC visibility pop
