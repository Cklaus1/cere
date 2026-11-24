// SelfTune ST4 S3c: the whole-model entry point of family sdar (executor/tests/gpu/test_sdar_kernels_model.py), the
// recorded-step replay of condition 1. It runs the model's forward (model/sdar/sdar_model.h CudaForward, the
// device's) teacher-forced along a recorded generation: the prompt's full blocks are prefilled, then every decode
// block is denoised once per recorded step, and after each step the recorded step's position receives the recorded
// token (not CERE's pick), so every step sees the reference's block; the block's recorded tokens are then committed.
// Each step's per-row top token, BF16 tie count and unmask index are written out for the caller to compare with the
// recording. Compiled only into a CERE_MODEL=<sdar descriptor> library.
#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

#include "cere_executor.h"
#include "ktest/sdar/sdar_ktest_model.h"
#include "model/sdar/sdar_model.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

using cere::sdar::kScheduleBlock;

// One step's record: the unmask index, then per block row its top id and tie count.
constexpr int64_t kRecordInts = 1 + 2 * int64_t(kScheduleBlock);

}  // namespace

// tensors, names, dims, count: ktest::load's (ktest/sdar/sdar_ktest_model.h).
// tokens: uint32 host [total], the recorded sequence (prompt, then generation); prompt_len < total; total minus the
// prompt's full blocks is a multiple of 4. order: int32 host [blocks, 4], per decode block the position (0..3) each
// step unmasked, -1 after the block's last step. out: int32 host [blocks, 4, 9], per step its record (index, then
// id and ties of rows 0..3); the steps after a block's last are left as they were. mask_id: the mask token.
CERE_KTEST int cere_ktest_sdar_replay(const void *tensors, const void *names, const void *dims, int64_t count,
                                      const void *tokens, int64_t prompt_len, int64_t total, const void *order,
                                      void *out, int64_t mask_id, cudaStream_t stream) {
    using namespace cere::sdar;
    if (tensors == nullptr || names == nullptr || dims == nullptr || tokens == nullptr || order == nullptr ||
        out == nullptr || count <= 0 || prompt_len <= 0 || prompt_len >= total || mask_id < 0) {
        return CERE_ERR_ARGUMENT;
    }
    const uint64_t prefill = uint64_t(prompt_len) / kScheduleBlock * kScheduleBlock;
    if ((uint64_t(total) - prefill) % kScheduleBlock != 0) return CERE_ERR_ARGUMENT;
    const uint64_t blocks = (uint64_t(total) - prefill) / kScheduleBlock;
    const auto *seq = static_cast<const uint32_t *>(tokens);
    const auto *steps = static_cast<const int32_t *>(order);
    auto *rec = static_cast<int32_t *>(out);

    cere::sdar::ktest::Model m;
    if (const cere_status st = cere::sdar::ktest::load(tensors, names, dims, count, uint64_t(total), &m);
        st != CERE_OK) {
        return st;
    }
    CudaForward forward(m.weights, m.gemms, m.ws, m.k_cache, m.v_cache, m.rows, stream);

    if (prefill > 0) {
        if (const cere_status st = forward.prefill(seq, prefill, 0); st != CERE_OK) return st;
    }
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint64_t at = prefill + b * kScheduleBlock;
        uint32_t block[kScheduleBlock];
        bool masked[kScheduleBlock];
        for (uint64_t r = 0; r < kScheduleBlock; ++r) {
            masked[r] = at + r >= uint64_t(prompt_len);
            block[r] = masked[r] ? uint32_t(mask_id) : seq[at + r];
        }
        for (uint64_t s = 0; s < kScheduleBlock; ++s) {
            const int32_t p = steps[b * kScheduleBlock + s];
            if (p < 0) break;
            if (p >= int32_t(kScheduleBlock) || !masked[p]) return CERE_ERR_ARGUMENT;
            RowPick picks[kScheduleBlock];
            int32_t index = -1;
            if (const cere_status st = forward.denoise(block, masked, at, picks, &index); st != CERE_OK) return st;
            int32_t *r = rec + (b * kScheduleBlock + s) * kRecordInts;
            r[0] = index;
            for (uint64_t q = 0; q < kScheduleBlock; ++q) {
                r[1 + 2 * q] = int32_t(picks[q].id);
                r[2 + 2 * q] = int32_t(picks[q].ties);
            }
            block[p] = seq[at + uint64_t(p)];
            masked[p] = false;
        }
        if (std::any_of(masked, masked + kScheduleBlock, [](bool m) { return m; })) return CERE_ERR_ARGUMENT;
        if (const cere_status st = forward.commit(block, at); st != CERE_OK) return st;
    }
    return CERE_OK;
}
