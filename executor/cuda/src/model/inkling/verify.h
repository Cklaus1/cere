// Inkling 2 Mini S9 phase B (inkling design note D-I-S9B): the kernels of a speculative verify and its commit.
// A verify call runs k rows per session (the unfed token, then the drafts) through the decode path; its rows must
// produce the logits of k sequential M = 1 decodes bit for bit, and only the accepted rows may change the session:
//
// - Conv (kernels/inkling/conv.h): row j of a session's run reads its window from the session's pre-verify FP32
//   state and the inputs of the run's earlier rows, exactly the window conv_*_prefill gives that row (the same
//   ref::conv_*_point over the same floats), and writes no state. The inputs of every visit's four convs stay in a
//   stage (the forward writes them there); conv_commit then advances each state over the kept rows only
//   (conv_ref.h advance_state of those rows).
// - Sliding KV: the verify's sliding K/V rows stay in a stage (the attention reads the call's own keys from k/v, the
//   older keys from the ring); ring_commit stores the kept rows (slot = position % 1024, attn.h kv_store). Storing a
//   rejected row would overwrite a slot still inside the window of the next rows. Full visits store in place: a
//   rejected row's pool row is past the session's position, never read before the next call rewrites it.
//
// Every function enqueues on `stream`, never synchronizes and returns the launch status (cudaErrorInvalidValue for a
// malformed argument, nothing launched). Compiled with -fmad=false; the conv points are conv_ref.h's.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "model/inkling/forward.h"

namespace cere::inkling {

constexpr int64_t kVerifyRows = kMaxDecodeRows;  // rows of one verify call, every session's run together
constexpr int64_t kMaxRunRows = 16;              // rows of one session's run (the token + up to 15 drafts)
// Conv input stage, BF16: region (visit v, conv c) at stage + (v * 4 + c) * kConvStageRegion holds the call's rows
// [kVerifyRows][C] (row stride C; C = 1024 for K / V, 2304 for the attention and MLP convs).
constexpr int64_t kConvStageRegion = kVerifyRows * kHidden;
constexpr int64_t kConvStageElems = int64_t(plan::kVisits) * 4 * kConvStageRegion;
// Sliding K/V stage, BF16 [kSlidingVisits][K, V][kVerifyRows][1024].
constexpr int64_t kKvStageElems = int64_t(plan::kSlidingVisits) * 2 * kVerifyRows * kKvRow;

// One session's run of a verify call and what the commit keeps of it (device memory, an array of S).
struct CommitSeq {
    int64_t first = 0;  // the run's first row in the call
    int64_t keep = 0;   // rows kept: [first, first + keep), 1 <= keep <= the run's rows
    int64_t pos0 = 0;   // the position of the run's first row
    int64_t out = 0;    // gather_rows: the first destination row of the kept rows
    float *conv = nullptr;  // the session's conv states (Session::conv)
    bf16 *ring = nullptr;   // the session's sliding rings (Session::ring)
};

// Conv of the verify rows: M rows, row r of a session's run at index idx[r] (device int32 [M]; rows r - idx[r] ..
// r - 1 are the run's earlier rows) with its session's state states[r] (device [M]; equal for the rows of one run).
// Each row's input is also written to stage (BF16 [M][C], row stride C) for conv_commit. state_only (the fail-first
// mutant): every row's window from the state alone, as a lone decode row.
cudaError_t conv_static_verify(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *w, float *const *states,
                               const int32_t *idx, bf16 *y, int64_t ldy, bf16 *stage, bool state_only,
                               cudaStream_t stream);
cudaError_t conv_dynamic_verify(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *static_w, const bf16 *dyn,
                                int64_t ld_dyn, float *const *states, const int32_t *idx, bf16 *y, int64_t ldy,
                                bf16 *stage, bool state_only, cudaStream_t stream);

// Advances every conv state of the S sessions over their kept rows of the conv input stage.
cudaError_t conv_commit(const bf16 *stage, const CommitSeq *seqs, int64_t S, cudaStream_t stream);
// Stores the kept rows of the sliding K/V stage into the sessions' rings. max_keep >= every keep.
cudaError_t ring_commit(const bf16 *kv_stage, const CommitSeq *seqs, int64_t S, int64_t max_keep, cudaStream_t stream);
// dst row out + i = src row first + i (i < keep) of each session; rows of `width` BF16 (a multiple of 8) at strides
// ld_src / ld_dst (multiples of 8).
cudaError_t gather_rows(const bf16 *src, int64_t ld_src, int64_t width, const CommitSeq *seqs, int64_t S,
                        int64_t max_keep, bf16 *dst, int64_t ld_dst, cudaStream_t stream);

}  // namespace cere::inkling
