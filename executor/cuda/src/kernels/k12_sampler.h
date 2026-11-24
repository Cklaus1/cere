// K12 device sampler (m27 design note I10, D2-D5, D7; ledger table R K12): the one per-item logits stage of the
// scored row (ledger M6). `host/sampler_ref` is the semantic reference of every function here; the kernel suite
// `executor/tests/gpu/test_kernels_k12_sampler.py` (C11) compares the two.
//
// Per sampled step the model forward (model/forward.cpp) runs, on the BF16 logits of the scored row:
//   process (BF16 -> fp32, penalties from the session's counts, mask_dev, logit_bias) into proc_logits;
//   top2_f32 (greedy with processors, and the runner-up of a sampled token);
//   for temperature > 0: probabilities (x / T, fp64 log-sum-exp), keep_mask (D3), draw_seeded (D4);
//   finalize (the committed token's cere_token_score); top_logprobs (D7, when requested); count_commit (D5).
//
// Every function enqueues on `stream`, allocates nothing, synchronizes nothing and returns the launch status
// (cudaErrorInvalidValue for arguments outside its contract). `vocab` is in [2, 2^31). Every reduction has one fixed
// order (one block over the row, or a fixed grid whose per-block partials are combined in block order), so a repeat
// is bit-identical. `scratch` is device memory of scratch_bytes(), 8-byte aligned, shared by probabilities,
// keep_mask and the draws: nothing in it outlives the call that wrote it.
//
// math_reference: sglang
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_executor.h"

namespace cere::k12 {

using bf16 = __nv_bfloat16;

// The draw's fp64 scan runs over kChunks contiguous id ranges (one per thread of the scan block).
constexpr int kChunks = 1024;

// D2 without the temperature: out[i] = fp32(logits[i]); for counts[i] > 0, x - (f * c + p) (f * c one fp32 multiply,
// no contraction) then x < 0 ? x * r : x / r; then ids whose bit is clear in mask_dev (ceil(vocab / 32) words, bit
// i % 32 of word i / 32; null = no mask) become -inf; then out[bias_ids[j * stride]] += bias_values[j * stride] for
// j < bias_count (ids unique and < vocab, the caller validated them; stride in 4-byte words: 1 for separate arrays,
// 2 for cere_logit_bias). counts null = no output tokens yet.
cudaError_t process(const bf16 *logits, int64_t vocab, const uint32_t *counts, float frequency_penalty,
                    float presence_penalty, float repetition_penalty, const uint32_t *mask_dev, const uint32_t *bias_ids,
                    const float *bias_values, int64_t bias_stride, int64_t bias_count, float *out, cudaStream_t stream);

// The two highest values, ties to the lower id (`k1_primitives.cu` top2's order): ids[0] the first maximum, ids[1]
// the first maximum without it; scores their values. NaN ranks above +inf (NaNs by the lower id): a NaN logit is
// the greedy token, never skipped.
cudaError_t top2_f32(const float *values, int64_t vocab, uint32_t *ids, float *scores, cudaStream_t stream);

// z = x / T (fp32, temperature > 0), lse = max + log(sum exp(z - max)) in fp64, probs[i] = fp32(exp(z - lse)); -inf
// entries get 0. *lse (device, fp64) receives the log-sum-exp. A fixed grid over the row: the fp64 sum adds each
// block's ids in a fixed order, then the blocks' partials in block order. Non-finite z (a fault upstream): +inf
// counts as the largest finite float (the +inf ids split the mass, the finite ids get 0), NaN carries no mass (0,
// like -inf), so probs never holds NaN; a row with no mass at all (every z NaN or -inf; lse = -inf) puts p = 1 on
// its lowest NaN id (id 0 without one), the token greedy picks, so keep_mask and the draws always have an id.
cudaError_t probabilities(const float *processed, int64_t vocab, float temperature, float *probs, double *lse,
                          void *scratch, cudaStream_t stream);

// D3 keep mask of `probs` (sampler_ref::keep_mask's semantics): ids with p > 0 ranked by p, ties to the lower id;
// min_p == 0: rank < top_k and the fp64 mass ranked before <= top_p; min_p > 0: top_k, renormalize, top_p,
// renormalize, p >= p_max * min_p. An exact radix select on the fp32 bit patterns finds the probability of rank
// min(top_k, #positive, 1024) - 1; one block orders those ranks and runs sampler_ref's scan on them (fp64 masses
// summed in rank order). A row whose cut can lie beyond those ranks (top_k > 1024 and top_p not reached within
// them, or min_p > 0 with top_k > 1024 positive ids) takes the whole-row path: 31-step bisections on fp32 bit
// patterns (counts and fp64 masses of p > v). mask: ceil(vocab / 32) words. top_k in [1, vocab], top_p in (0, 1],
// min_p in [0, 1].
cudaError_t keep_mask(const float *probs, int64_t vocab, int64_t top_k, float top_p, float min_p, uint32_t *mask,
                      void *scratch, cudaStream_t stream);

// Bytes of the scratch of probabilities, keep_mask and the draws (the draws' part: kChunks + 1 fp64 chunk prefixes
// and the last kept id).
size_t scratch_bytes();

// D4 inverse CDF over the kept ids in id order for each u: the fp64 running sum of the kept probabilities against
// u * S (S the fp64 kept total), the first id whose running sum exceeds it, clamped to the last kept id. The mask
// keeps at least one id. draw_u reads us[n] (fp64, [0, 1)); draw_seeded uses u = uniform(first_seed + j, key, 0, 1)
// (murmur_hash32 counter, ledger M8) for draw j. ids[n].
cudaError_t draw_u(const float *probs, const uint32_t *mask, int64_t vocab, const double *us, int64_t n, uint32_t *ids,
                   void *scratch, cudaStream_t stream);
cudaError_t draw_seeded(const float *probs, const uint32_t *mask, int64_t vocab, uint64_t first_seed, int64_t n,
                        uint32_t key, uint32_t *ids, void *scratch, cudaStream_t stream);

// The committed token's score (cere_executor.h cere_token_score): id = *drawn (null: top2_ids[0]), runner_up the most
// probable other token (top2_ids[1] when id is top2_ids[0], else top2_ids[0]), logits from `processed`.
cudaError_t finalize(const float *processed, const uint32_t *top2_ids, const uint32_t *drawn, cere_token_score *out,
                     cudaStream_t stream);

// D7: z = temperature > 0 ? processed / T : processed; log_softmax(z) with an fp64 log-sum-exp, unfiltered. The
// log-sum-exp counts z like `probabilities` (NaN carries no mass, +inf is the largest finite float; lse = -inf when
// no z carries mass); the top-n order ranks NaN above +inf, ties to the lower id; a NaN entry's logprob is NaN.
// *committed_logprob = the committed token's (committed_dev, device, or committed_host when it is null); the top n
// (n <= CERE_MAX_TOP_LOGPROBS, descending, ties to the lower id) go to top_ids[j * stride] / top_logprobs[j * stride]
// (stride in 4-byte words: 1 for separate arrays, 2 for cere_logprob_entry). n = 0 writes no entry.
cudaError_t top_logprobs(const float *processed, int64_t vocab, float temperature, const uint32_t *committed_dev,
                         uint32_t committed_host, int64_t n, float *committed_logprob, uint32_t *top_ids,
                         float *top_logprobs, int64_t stride, cudaStream_t stream);

// D5: counts[ids[i]] += 1 for i < n.
cudaError_t count_commit(uint32_t *counts, const uint32_t *ids, int64_t n, cudaStream_t stream);

}  // namespace cere::k12
