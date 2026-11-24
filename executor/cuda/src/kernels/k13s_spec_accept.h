// K13S phase-S accept/resample (m29 design note D6, M29S2-13; ledger M8 "Counter RNG channels"; ledger table R
// K13S): SGLang's target-only rule with thresholds 1.0 on one speculative chain, on device buffers, with no host
// round trip per row. Kernel suite: `executor/tests/gpu/test_kernels_spec_accept.py` (K6).
//
// A chain of one session has n drafts d_1..d_n (n in [0, kMaxDrafts]) and n + 1 verify rows at consecutive
// positions; row j (0-based) chooses the token at absolute position key0 + j. Row j's inputs are K12's sampler stage
// outputs for that row (kernels/k12_sampler.h): `probabilities` (fp32 [vocab], the softmax of processed / T) and its
// `keep_mask` (ceil(vocab / 32) words, bit i % 32 of word i / 32 = kept; at least one id kept). The row's
// distribution is p_j(i) = kept(i) p[i] / S_j with S_j the fp64 kept total (K12's draw total: fp64 sums of each of
// k12::kChunks contiguous id ranges in id order, then those sums in chunk order).
//
// Rule, for j = 0, 1, ... while every earlier draft was accepted (ledger M8 channels, murmur counter of k12):
//   j < n:  u = uniform(seed, key0 + j, 4, 5); accept d = d_{j+1} iff kept(d) and u < fp64(p[d]) / S_j (fp64), so
//           p(d) = 0 is never accepted and p(d) = 1 always; on reject the committed id is the inverse CDF draw of
//           u' = uniform(seed, key0 + j, 6, 7) over the kept ids other than d (k12 draw_seeded's method: the fp64
//           running sum in id order, restarted at each chunk's fp64 prefix, against u' * S'_j where S'_j is the kept
//           total without d; the first id exceeding it, else the last kept id other than d) and the chain stops;
//   j = n:  (every draft accepted) the bonus id is k12::draw_seeded(row n, seed, key0 + n): plain sampling with
//           channels (0, 1), the existing K12 path.
// Every row's decision and resample draw are evaluated in parallel; the chain is then resolved in one thread. A
// repeat is bit-identical (fixed summation orders, counter RNG).
//
// Every function enqueues on `stream`, allocates nothing, synchronizes nothing and returns the launch status
// (cudaErrorInvalidValue for arguments outside its contract). `vocab` is in [2, 2^31).
//
// math_reference: sglang
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cere::k13s {

// Drafts per chain (the verify rows are n + 1 <= kMaxDrafts + 1).
constexpr int64_t kMaxDrafts = 15;

// Bytes of the device scratch of spec_accept for chains of up to `max_drafts` drafts (8-byte aligned memory; nothing
// in it outlives the call). Holds k12::scratch_bytes() for the bonus draw.
size_t scratch_bytes(int64_t max_drafts);

// One chain. probs: fp32 [n + 1, vocab] (row j at probs + j * vocab); masks: uint32 [n + 1, words]
// (words = ceil(vocab / 32), row j at masks + j * words); drafts: uint32 [n] (device; an id >= vocab is never
// accepted); seed: the request's sampling seed; key0: the absolute position row 0 chooses. Outputs (device):
// committed uint32 [n + 1] (committed[0..count) are the committed ids: the accepted drafts, then the resampled id
// or the bonus id; entries from count on are unspecified), count uint32 [1] in [1, n + 1] (accepted drafts =
// count - 1). scratch: scratch_bytes(n) bytes.
cudaError_t spec_accept(const float *probs, const uint32_t *masks, int64_t vocab, const uint32_t *drafts, int64_t n,
                        uint64_t seed, uint32_t key0, uint32_t *committed, uint32_t *count, void *scratch,
                        cudaStream_t stream);

// Bytes of the device scratch of row_trials (8-byte aligned).
size_t trial_scratch_bytes();

// spec_accept's row decision (the same kernels) for `trials` independent coins on one row with draft `draft` (host
// value; >= vocab = never accepted): trial t uses seed first_seed + t * seed_step and key key0 + t * key_step
// (uint32 wrap). ids uint32 [trials]: the committed id (draft on accept, else the resampled id); accepted uint8
// [trials]: 1 = accepted. For K6's distribution and bit-exact host-reference checks.
cudaError_t row_trials(const float *probs, const uint32_t *mask, int64_t vocab, uint32_t draft, uint64_t first_seed,
                       uint64_t seed_step, uint32_t key0, uint32_t key_step, int64_t trials, uint32_t *ids,
                       uint8_t *accepted, void *scratch, cudaStream_t stream);

}  // namespace cere::k13s
