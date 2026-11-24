// Inkling 2 Mini S8: kernel entry point of libcere_ktest.so over the greedy scores (model/inkling/logits.h), for
// executor/tests/gpu/test_inkling_kernels_logits.py. logits FP32 [rows, ld] (device); out_scores (device
// cere_token_score [rows], nullable) gets top2_rows, ids (device int32 [rows], nullable) argmax_rows. Enqueues on
// `stream` and returns the first launch status that is not 0 (cudaSuccess).
#include <cstdint>

#include "model/inkling/logits.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace ink = cere::inkling;

CERE_KTEST int cere_ktest_inkling_top2(const float *logits, int64_t rows, int64_t vocab, int64_t ld, void *out_scores,
                                       int32_t *ids, cudaStream_t stream) {
    if (out_scores != nullptr) {
        const cudaError_t e =
            ink::top2_rows(logits, rows, vocab, ld, static_cast<cere_token_score *>(out_scores), stream);
        if (e != cudaSuccess) return int(e);
    }
    if (ids != nullptr) return int(ink::argmax_rows(logits, rows, vocab, ld, ids, stream));
    return 0;
}
