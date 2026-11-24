// SelfTune ST4 S4a (selftune design note ST4, D-S4-1, D-S4-2, D-S4-4): the e2e free run of family sdar
// (executor/tests/gpu/sdar_generate.py, scripts/live/dice_judge.py). It runs the device's schedule (model/sdar/
// schedule.h `step`) over the device's forward (model/sdar/sdar_model.h CudaForward) exactly as the host drives a
// session: the prompt in one call, then every scored token fed back, free to the schedule's full length (no stop
// token ends it, so every block runs whole). A recording Forward between the two keeps, per denoise, the
// device-written unmask index and the per-row top id and BF16 tie count, and per forward kind its count (the host
// tally the judge cross-checks against the nsys kernel records). The whole run is one NVTX range named by `label`
// and each decode block b a nested range `block <b>` (its first denoise to its commit), so the judge can attribute
// the trace's kernels to the prompt and the block. Compiled only into a CERE_MODEL=<sdar descriptor> library.
#include <cstdint>
#include <cstring>
#include <string>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "cere_executor.h"
#include "ktest/sdar/sdar_ktest_model.h"
#include "model/sdar/schedule.h"
#include "model/sdar/sdar_model.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

using cere::sdar::Forward;
using cere::sdar::kScheduleBlock;
using cere::sdar::RowPick;

// One denoise's record: the unmask index, then per block row its top id and tie count.
constexpr int64_t kRecordInts = 1 + 2 * int64_t(kScheduleBlock);

// The device's forward with every pass recorded. Decode block b starts at base = prefill + 4b.
class Recording final : public Forward {
public:
    Recording(Forward &inner, uint64_t prefill, uint64_t blocks, int32_t *rec, int32_t *tally)
        : inner_(inner), prefill_(prefill), blocks_(blocks), rec_(rec), tally_(tally) {}

    cere_status prefill(const uint32_t *tokens, uint64_t n, uint64_t pos) override {
        if (pos + n > prefill_) return CERE_ERR_INTERNAL;
        tally_[0] += 1;
        tally_[1] += int32_t(n);
        return inner_.prefill(tokens, n, pos);
    }

    cere_status denoise(const uint32_t *block, const bool *masked, uint64_t base, RowPick *picks,
                        int32_t *index) override {
        uint64_t b = 0;
        if (!block_of(base, &b)) return CERE_ERR_INTERNAL;
        int32_t &denoised = tally_[2 + 2 * b];
        if (denoised >= int32_t(kScheduleBlock)) return CERE_ERR_INTERNAL;  // the record buffer's bound
        if (!open_) nvtxRangePushA(("block " + std::to_string(b)).c_str());
        open_ = true;
        if (const cere_status st = inner_.denoise(block, masked, base, picks, index); st != CERE_OK) return st;
        int32_t *r = rec_ + (b * kScheduleBlock + uint64_t(denoised)) * kRecordInts;
        r[0] = *index;
        for (uint64_t q = 0; q < kScheduleBlock; ++q) {
            r[1 + 2 * q] = int32_t(picks[q].id);
            r[2 + 2 * q] = int32_t(picks[q].ties);
        }
        denoised += 1;
        return CERE_OK;
    }

    cere_status commit(const uint32_t *block, uint64_t base) override {
        uint64_t b = 0;
        if (!block_of(base, &b)) return CERE_ERR_INTERNAL;
        tally_[3 + 2 * b] += 1;
        const cere_status st = inner_.commit(block, base);
        if (open_) nvtxRangePop();
        open_ = false;
        return st;
    }

private:
    bool block_of(uint64_t base, uint64_t *b) const {
        if (base < prefill_ || (base - prefill_) % kScheduleBlock != 0) return false;
        *b = (base - prefill_) / kScheduleBlock;
        return *b < blocks_;
    }

    Forward &inner_;
    uint64_t prefill_;
    uint64_t blocks_;
    int32_t *rec_;
    int32_t *tally_;
    bool open_ = false;  // a block's NVTX range is pushed (its first denoise) and not yet popped (its commit)
};

}  // namespace

// tensors, names, dims, count: ktest::load's (ktest/sdar/sdar_ktest_model.h). prompt: uint32 host [prompt_len];
// gen_len: the tokens generated, so that prompt_len + gen_len is a multiple of 4 (the schedule's whole blocks); the
// session's context is prompt_len + gen_len. mask_id: the mask token. sdar_gemm: the CERE_LEVER_SDAR_GEMM_* the
// Gemms run (SdarOptIn, O320: dice_judge's profile's lever). label: NUL-terminated, the NVTX range's name.
// Outputs (host; blocks = (prompt_len + gen_len) / 4 - prompt_len / 4):
// - gen: uint32 [gen_len], the scored tokens in order;
// - rec: int32 [blocks, 4, 9], per denoise of each block its record (index, then id and ties of rows 0..3), -1
//   after a block's last denoise;
// - tally: int32 [2 + 2 * blocks]: prefill forwards, prefill tokens, then per block its denoise and commit forwards.
CERE_KTEST int cere_ktest_sdar_generate(const void *tensors, const void *names, const void *dims, int64_t count,
                                        const void *prompt, int64_t prompt_len, int64_t gen_len, int64_t mask_id,
                                        int64_t sdar_gemm, const char *label, void *gen, void *rec, void *tally,
                                        cudaStream_t stream) {
    using namespace cere::sdar;
    if (tensors == nullptr || names == nullptr || dims == nullptr || prompt == nullptr || label == nullptr ||
        gen == nullptr || rec == nullptr || tally == nullptr || count <= 0 || prompt_len <= 0 || gen_len <= 0 ||
        mask_id < 0 || sdar_gemm < 0 || sdar_gemm > int64_t(CERE_LEVER_SDAR_GEMM_CHANGING) ||
        (prompt_len + gen_len) % int64_t(kScheduleBlock) != 0) {
        return CERE_ERR_ARGUMENT;
    }
    const uint64_t total = uint64_t(prompt_len + gen_len);
    const uint64_t prefill = uint64_t(prompt_len) / kScheduleBlock * kScheduleBlock;
    const uint64_t blocks = (total - prefill) / kScheduleBlock;
    auto *out = static_cast<uint32_t *>(gen);
    auto *records = static_cast<int32_t *>(rec);
    auto *counts = static_cast<int32_t *>(tally);
    std::memset(counts, 0, (2 + 2 * blocks) * sizeof(int32_t));
    for (uint64_t i = 0; i < blocks * kScheduleBlock * uint64_t(kRecordInts); ++i) records[i] = -1;

    cere::sdar::ktest::Model m;
    if (const cere_status st = cere::sdar::ktest::load(tensors, names, dims, count, total, &m, uint32_t(sdar_gemm));
        st != CERE_OK) {
        return st;
    }
    CudaForward device(m.weights, m.gemms, m.ws, m.k_cache, m.v_cache, m.rows, stream);
    Recording forward(device, prefill, blocks, records, counts);
    Session s;
    s.capacity = total;

    nvtxRangePushA(label);
    uint64_t consumed = 0;
    cere_token_score score{};
    cere_status st = step(forward, s, static_cast<const uint32_t *>(prompt), uint64_t(prompt_len), 0,
                          uint32_t(mask_id), &consumed, &score);
    if (st == CERE_OK && consumed != uint64_t(prompt_len)) st = CERE_ERR_INTERNAL;
    for (int64_t i = 0; st == CERE_OK && i < gen_len; ++i) {
        out[i] = score.id;
        if (i + 1 == gen_len) break;
        const uint32_t fed = score.id;
        st = step(forward, s, &fed, 1, 0, uint32_t(mask_id), &consumed, &score);
    }
    if (st == CERE_OK && cudaStreamSynchronize(stream) != cudaSuccess) st = CERE_ERR_INTERNAL;
    nvtxRangePop();
    return st;
}
