// K9 QSA sparse attention (m22c design note D3, F4/F5): the indexer (index q norm + MRoPE), the compressed-key
// cache (mean of 4 raw index keys -> Gemma norm (1 + w) -> MRoPE at the group start -> store, with a pending ring of
// the last incomplete group), the block selection (F32 relu(q.k) summed over the 4 index heads / sqrt(128), visible
// blocks floor((p + 1) / 4), top-512, expansion + tail + stable compaction into a row of 2,051 logical positions) and
// the sparse GQA over the selected positions. A port of SGLang's QSAIndexer (`$S/layers/attention/qsa/qsa_indexer.py`)
// with its fused prep kernels (`sglang/kernels/jit/csrc/attention/qsa_indexer.cuh`), MQA scores (`qsa/mqa.py`),
// `fast_topk` (`sglang/kernels/jit/csrc/elementwise/fast_topk.cuh`) and expansion (`qsa/kernel.py:120-149`).
//
// Per full-attention call above the dense limit (spec D2), the caller (model forward) runs:
//   K2 GEMM hidden -> qk [tokens, 640] (index_qk_proj); indexer_qk (index q); qsa_compress_store (raw index keys of
//   this call -> ring + compressed keys); k6::prologue (attention q/k norm, mRoPE, gate, FP8 KV append);
//   qsa_select (indices); qsa_attention (sparse GQA); k1::sigmoid_mul (output gate); K2 o_proj.
// Every call from position 0 runs indexer_qk + qsa_compress_store, so later queries see every complete block.
//
// Addressing: `slots` (int64, device) maps a logical position to its KV cache row exactly as k6 (`k6_attention.h`),
// null = identity. The compressed key of block b (logical positions 4b .. 4b + 3) lives at row slots[4b] / 4 of
// `keys` (null slots: row b); a caller's allocator keeps each group of 4 inside one aligned 4-row page, so slot / 4
// is unique per group. There is no dump slot: a call writes only the compressed rows of groups completed by it and
// the session's ring, and reads only slots[0, seq_len) and the compressed rows of blocks [0, floor(seq_len / 4)).
//
// Positions: the text entry points take 1-D positions as K6. `positions[t]` is the rotary position of new
// token t (null: pos0 + t); the logical position of new token t is always pos0 + t. A group whose first member came
// from an earlier call (it sits in the ring) is rotated at positions[0] - (pos0 - 4g) (text positions are
// consecutive). M2.13-L migrates K9 to mRoPE for image sessions' prompt rows (contract C5): indexer_qk_mrope and
// qsa_compress_store_mrope take positions3, int64 device [3][p3_stride] axis-major (t, h, w), whose column c holds
// the 3-axis rotary position of logical position pos0 - p3_lead + c, p3_lead = pos0 % 4; new token t is column
// p3_lead + t. Rotary frequency i (0..31, NeoX half split) reads its cos/sin from the rope_table row of axis
// a(i) = 1 if i % 3 == 1 and i < 33, 2 if i % 3 == 2 and i < 30, else 0 (interleaved mrope_section [11, 11, 10]:
// K1 mrope_axis, k1_primitives.cu:153-162; SGLang QSAIndexer._rope_axis_map, qsa_indexer.py:147-153, applied by
// qsa_mrope_apply, qsa_indexer.cuh:88-125). The compressed key of group g is rotated at the stored 3-axis position
// of its FIRST member (logical 4g, column 4g - (pos0 - p3_lead)), as SGLang's compress reads
// rope_position_buffer[locs[0]] (qsa_indexer.cuh:326,378-381). With three equal axes both are bit-identical to
// indexer_qk / qsa_compress_store at those positions (same BF16 rounding sequence).
//
// Every entry point takes a cudaStream_t, allocates nothing, synchronizes nothing and returns CERE_OK, or
// CERE_ERR_ARGUMENT for any shape outside the compiled support table / a null required pointer / short scratch,
// or CERE_ERR_INTERNAL for a launch failure.
//
// math_reference: sglang
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_executor.h"
#include "cere_model_constants.h"

namespace cere::k9 {

using bf16 = __nv_bfloat16;

// The compiled support table: the A4 model's QSA indexer and full-attention shape (support.cpp:28-32).
constexpr int kIndexHeads = gen::kIndexerNHeads;        // indexer_n_heads
constexpr int kIndexHeadDim = gen::kIndexerHeadDim;     // indexer_head_dim (one index KV head)
static_assert(gen::kIndexerKvHeads == 1, "one index KV head");
constexpr int kQkWidth = (kIndexHeads + 1) * kIndexHeadDim;  // index_qk_proj rows: 4 q heads then 1 k head (640)
constexpr int kRotaryDim = gen::kRotaryDim;  // the attention RoPE the indexer reuses (partial_rotary_factor 0.25 of 256)
constexpr int kRatio = gen::kIndexerCompressRatio;      // indexer_compress_ratio
constexpr int kTokenTopk = gen::kIndexerBudget;         // indexer_budget
constexpr int kBlockTopk = kTokenTopk / kRatio;   // 512
constexpr int kRowWidth = kTokenTopk + kRatio - 1;  // 2,051 selected logical positions per query row
constexpr int kQHeads = gen::kNumAttentionHeads;  // attention: as k6
constexpr int kKvHeads = gen::kNumKeyValueHeads;
constexpr int kHeadDim = gen::kHeadDim;
constexpr float kSoftmaxScale = 0.0625f;          // head_dim^-0.5
static_assert(kSoftmaxScale * kSoftmaxScale * kHeadDim == 1.0f, "kSoftmaxScale is head_dim^-0.5");
constexpr float kScoreDivisor = 11.313708498984761f;  // float(math.sqrt(128)), the MQA score divisor (mqa.py)
constexpr size_t kLogitsTileBytes = size_t{128} << 20;  // _QSA_PREFILL_LOGITS_BUDGET_BYTES (qsa_indexer.py:30)
// qsa_attention calls of at most this many rows (without P4) run the split-K kernel, so row t of such a call is
// bit-equal to the one-row call on row t; above it the kernel runs unsplit (other numerics).
constexpr int kSplitMaxRows = 16;

// Query rows scored per logits tile for keys = floor(ctx / 4) compressed keys, as _qsa_prefill_row_chunk_size
// (qsa_indexer.py:31-41): multiples of 128 / kIndexHeads rows, at most kLogitsTileBytes of F32 logits (min rows).
int64_t logits_tile_rows(int64_t rows, int64_t ctx);

// Bytes of qsa_select's scratch (the F32 logits tile) for `rows` query rows at sequence length <= ctx.
size_t select_scratch_bytes(int64_t rows, int64_t ctx);
// Bytes of qsa_attention's scratch (split-K partials for calls of <= 16 rows; 0 otherwise).
size_t attention_scratch_bytes(int64_t rows);

// The whole K9 scratch of one call of `rows` tokens at context ctx (spec D5): index q [rows, 4, 128] BF16, indices
// [rows, 2051] int32 and max(select, attention) work bytes, each 256-byte aligned.
size_t scratch_bytes(int64_t rows, int64_t ctx);

struct Scratch {
    bf16* q_idx;        // [rows, kIndexHeads, kIndexHeadDim]
    int32_t* indices;   // [rows, kRowWidth]
    void* work;         // select / attention scratch
    size_t work_bytes;
};
// Carves a scratch_bytes(rows, ctx) region (256-byte aligned base) into the pieces above.
cere_status carve_scratch(void* base, size_t bytes, int64_t rows, int64_t ctx, Scratch* out);

// Device-resident index bounds (audit KernelsAttention F3). A rotary position r reads row r of cos_sin, valid for
// rope_first <= r < rope_end (the table as the caller passes it: model.h rope_table(w, s) covers rows -rope_delta ..
// table rows - rope_delta); a logical position p indexes `slots` (and the compressed rows through it), valid for
// 0 <= p < context (the slot map's length). An entry point checks the positions it gets as host scalars before any
// launch (CERE_ERR_ARGUMENT). A kernel that reads one from device memory (positions, positions3, a device position)
// outside its range uses the nearest valid value instead (its outputs are those of that value), so no access leaves
// the table or the slot map, and stores kBoundFault to *fault (a word the device writes: model forward's mapped
// staging, Workspace::bound_fault; null: clamp only). The caller fails such a call (forward reads the word after its
// drain: CERE_ERR_INTERNAL, the session broken). Valid inputs are unaffected bit for bit. Required: rope_first <
// rope_end where a call rotates, context >= 1 where it addresses positions.
struct Bounds {
    int64_t rope_first;
    int64_t rope_end;
    int64_t context;
    uint32_t* fault;
};
constexpr uint32_t kBoundFault = 1;

// Index q prep, as SGLang's qsa_index_q_norm_rope_store (bit-for-bit arithmetic): qk [tokens, 640] BF16 (row stride
// qk_stride elements, the index_qk_proj output); for each head h < 4,
//   q_idx[t, h, :] = mrope(bf16(gemma_rmsnorm(qk[t, 128h : 128h + 128], q_norm_weight, eps)))
// with F32 sum of squares in flashinfer's partial order, rsqrtf, and cos/sin rounded to BF16 and every rotary product
// rounded to BF16. cos_sin is the K1 rope_table [n_pos, 64] (row p = [cos | sin]); positions as above, rotary rows
// inside bounds.rope_first .. rope_end (host pos0 .. pos0 + tokens; device positions clamped, see Bounds).
// q_idx is contiguous [tokens, 4, 128]. The raw index key of token t is qk[t, 512 : 640].
cere_status indexer_qk(const bf16* qk, int64_t qk_stride, const bf16* q_norm_weight, float eps, const float* cos_sin,
                       const Bounds& bounds, const int64_t* positions, int64_t pos0, int64_t tokens, bf16* q_idx,
                       cudaStream_t stream);

// Compressed-key store for new tokens pos0 .. pos0 + tokens - 1 of one session, as SGLang's
// qsa_index_k_compress_store: every group g with pos0 <= 4g + 3 < pos0 + tokens (complete after this call) gets
//   keys[slot(g)] = mrope(bf16(gemma_rmsnorm(bf16(F32 sequential mean of its 4 raw keys)), k_norm_weight, eps)),
//   rotated at its first member's rotary position;
// members before pos0 are read from `ring` (BF16 [4, 128], member p at ring row p % 4), the rest from k_idx
// ([tokens, 128], row stride k_stride elements; e.g. qk + 512 with stride 640). Afterwards the raw keys of the
// incomplete last group (positions >= floor((pos0 + tokens) / 4) * 4) are stored to ring rows p % 4.
// slot(g) = slots ? slots[4g] / 4 : g. ring and keys are explicit buffers (staging-capable, spec D3 M29S-01).
// Bounds: pos0 + tokens <= bounds.context; with positions null, rotary rows floor(pos0 / 4) * 4 .. pos0 + tokens - 1
// inside bounds.rope_first .. rope_end (CERE_ERR_ARGUMENT); device positions are clamped (see Bounds).
cere_status qsa_compress_store(const bf16* k_idx, int64_t k_stride, const bf16* k_norm_weight, float eps,
                               const float* cos_sin, const Bounds& bounds, const int64_t* positions,
                               const int64_t* slots, int64_t pos0, int64_t tokens, bf16* ring, bf16* keys,
                               cudaStream_t stream);

// indexer_qk with token t rotated at the mRoPE position positions3[:, p3_lead + t] (see Positions). 0 <= p3_lead < 4,
// p3_stride >= p3_lead + tokens. Every positions3 value is a rotary row (clamped into bounds, see Bounds).
cere_status indexer_qk_mrope(const bf16* qk, int64_t qk_stride, const bf16* q_norm_weight, float eps,
                             const float* cos_sin, const Bounds& bounds, const int64_t* positions3, int64_t p3_stride,
                             int64_t p3_lead, int64_t tokens, bf16* q_idx, cudaStream_t stream);

// qsa_compress_store with group g rotated at the mRoPE position of its first member, positions3[:, 4g - (pos0 -
// p3_lead)] (see Positions). p3_lead must equal pos0 % 4; p3_stride >= p3_lead + tokens. Ring handling as
// qsa_compress_store; pos0 + tokens <= bounds.context, positions3 values clamped into bounds (see Bounds).
cere_status qsa_compress_store_mrope(const bf16* k_idx, int64_t k_stride, const bf16* k_norm_weight, float eps,
                                     const float* cos_sin, const Bounds& bounds, const int64_t* positions3,
                                     int64_t p3_stride, int64_t p3_lead, const int64_t* slots, int64_t pos0,
                                     int64_t tokens, bf16* ring, bf16* keys, cudaStream_t stream);

// qsa_compress_store of one decode token (tokens 1, k_idx row stride unused) at the device position *position
// (m23 design note P1, M23P1-03: the decode graph's step): the compress or the ring store is chosen on the device
// from *position % 4, with the arithmetic of qsa_compress_store at pos0 = *position. *position is clamped to
// 0 .. bounds.context - 1 and the group's rotary row into bounds (see Bounds).
cere_status qsa_compress_store_step(const bf16* k_idx, const bf16* k_norm_weight, float eps, const float* cos_sin,
                                    const Bounds& bounds, const int64_t* slots, const int64_t* position, bf16* ring,
                                    bf16* keys, cudaStream_t stream);

// qsa_compress_store(k_idx, k_stride, ..., positions null, slots, pos0 = *position, tokens, ...) with pos0 read on the
// device (the spec_graph draft graph's MTP extend, m29 design note M29S2-18b): the same compressed keys and ring
// rows, bit for bit, from two launches (the compress, its grid at the most groups `tokens` tokens complete with the
// extra warps exiting, then the ring store), so a captured call replays at any position. 1 <= tokens <=
// bounds.context; *position is clamped to 0 .. bounds.context - tokens, the rotary rows into bounds (see Bounds).
cere_status qsa_compress_store_at(const bf16* k_idx, int64_t k_stride, const bf16* k_norm_weight, float eps,
                                  const float* cos_sin, const Bounds& bounds, const int64_t* slots,
                                  const int64_t* position, int64_t tokens, bf16* ring, bf16* keys,
                                  cudaStream_t stream);

// The verify graph's undo of one compressed key (decode_step = spec_graph, m29 design note M29S2-18b, M29S2-20):
// forward_verify_rows' host key_row rule on the device. Verify row j (1 .. n) of a chain of n <= 16 drafts sits at the
// device position *position = p + j; let q be the first position >= p + j with q % 4 == 3. When q <= p + n, the
// chain may complete group (q - 3) / 4, and its pool row slots[q - 3] / 4 of keys ([128] BF16 each) is copied to
// `saved` ([128]); otherwise nothing is written. One launch, no host scalar of the position. *position is clamped to
// 0 .. bounds.context - 1 (bounds' rope fields unused; see Bounds).
cere_status qsa_key_save_step(const bf16* keys, const int64_t* slots, const Bounds& bounds, const int64_t* position,
                              int64_t j, int64_t n, bf16* saved, cudaStream_t stream);

// One full-attention layer's QSA verify work over the verify graph's rows in one launch (fuse-glue design note
// FG-6), byte-identical to the per-row launches it replaces (forward.cpp attention_chains): block c runs chain c's
// rows (chain_first[c] .. + chain_rows[c], the chains contiguous from row 0) in row order; per row r, when
// undo_ring[r] is set (a draft row), the undo copy of ring[r] ([4, 128] BF16) to undo_ring[r] and, when undo_key[r]
// is set, qsa_key_save_step(keys[r], slots[r], positions + r, j[r], n[r], undo_key[r]); then, when k_idx[r] is set,
// qsa_compress_store_step(k_idx[r], k_norm_weight, eps, cos_sin, slots[r], positions + r, ring[r], keys[r]). The
// chains' rings, keys and undo buffers are disjoint (one sequence per chain). Row r's position is clamped to
// 0 .. context[r] - 1 (its slot map's length, >= 1), its rotary row into bounds (bounds.context unused; see Bounds).
constexpr int kChainsStepRows = 16;
struct ChainsStep {
    const bf16* k_idx[kChainsStepRows];
    const int64_t* slots[kChainsStepRows];
    bf16* ring[kChainsStepRows];
    bf16* keys[kChainsStepRows];
    bf16* undo_ring[kChainsStepRows];
    bf16* undo_key[kChainsStepRows];
    int64_t context[kChainsStepRows];
    int32_t j[kChainsStepRows];
    int32_t n[kChainsStepRows];
    int32_t chain_first[kChainsStepRows];
    int32_t chain_rows[kChainsStepRows];
};
cere_status qsa_chains_step(const ChainsStep& rows, int chains, const bf16* k_norm_weight, float eps,
                            const float* cos_sin, const Bounds& bounds, const int64_t* positions,
                            cudaStream_t stream);

// Block logits, as SGLang's MQA (mqa.py torch reference / TileLang kernels): for query row t at logical position
// p_t (positions_logical[t]; null: seq_len - rows + t) and block b < floor((p_t + 1) / 4),
//   logits[t, b] = (sum_h relu(F32 dot(q_idx[t, h], keys[slot(b)]))) / kScoreDivisor,
// -inf for floor((p_t + 1) / 4) <= b < logits_ld. Requires logits_ld >= floor(seq_len / 4); a device p_t outside
// 0 .. seq_len - 1 is clamped into it (Bounds' rule, no fault word), so no read passes the row. impl 0 =
// qsa_select's choice for a call of `rows` rows (calls of up to kSplitMaxRows rows: logits_small_kernel, track LongK9
// long-k9 design note LK9-02; larger calls: logits_prefill_kernel), 1 = logits_kernel, 2 = logits_small_kernel;
// every choice writes the same bytes.
cere_status qsa_logits(const bf16* q_idx, const bf16* keys, const int64_t* slots, const int64_t* positions_logical,
                       int64_t rows, int64_t seq_len, float* logits, int64_t logits_ld, int impl,
                       cudaStream_t stream);

// Token selection, as SGLang's select_prefill_tokens / select_decode_tokens: per query row (logical position p_t as
// in qsa_logits, every p_t < seq_len), the top-512 visible blocks by their F32 logit (fast_topk's order: the
// float's order-preserving uint32 key, so -0 < +0; equal keys are taken lowest block first; every visible block
// when floor((p_t + 1) / 4) <= 512), each expanded to its 4 positions, blocks in ascending order; then the tail
// floor((p_t + 1) / 4) * 4 .. p_t; then -1 padding to 2,051 entries (stable compaction: valid entries first).
// Logits are scored in tiles of logits_tile_rows rows inside `scratch` (select_scratch_bytes(rows, seq_len)).
// out_indices int32 [rows, 2051] (logical positions). The top-512 of calls of up to kSplitMaxRows rows over at most
// 65,536 blocks (seq_len <= 262,147) runs on 8-block clusters per row (track LongK9, long-k9 design note LK9-01),
// larger calls over at most 32,768 blocks (seq_len <= 131,075) on one register-resident block per row
// (topk_prefill_kernel), the rest on one block per row (topk_kernel); all write the same bytes (qsa_topk). Their
// logits run on logits_small_kernel (LK9-02) or logits_prefill_kernel (qsa_logits impl 0).
cere_status qsa_select(const bf16* q_idx, const bf16* keys, const int64_t* slots, const int64_t* positions_logical,
                       int64_t rows, int64_t seq_len, void* scratch, size_t scratch_bytes, int32_t* out_indices,
                       cudaStream_t stream);

// qsa_select of one decode row at the device logical position *position (P1, M23P1-03): the logits grid spans the
// context's floor(ctx / 4) blocks with logits_ld = floor(ctx / 4); a block of keys at or past the row's visible
// blocks exits before any store (those logits stay unwritten and are never read), the rest score exactly as
// qsa_select, and the top-512 / expansion read the row's visible blocks only, so out_indices [1, 2051] equals
// qsa_select's at seq_len = *position + 1. scratch: >= floor(ctx / 4) F32 (select_scratch_bytes(1, ctx)).
// 1 <= bounds.context <= ctx; *position is clamped to 0 .. bounds.context - 1 (see Bounds; rope fields unused).
cere_status qsa_select_step(const bf16* q_idx, const bf16* keys, const int64_t* slots, const Bounds& bounds,
                            const int64_t* position, int64_t ctx, void* scratch, size_t scratch_bytes,
                            int32_t* out_indices, cudaStream_t stream);

// The top-512 + expansion step of qsa_select alone, on a logits tile [rows, ld] F32 (row t at logits + t * ld;
// positions and seq_len as qsa_select, ld >= floor(seq_len / 4); only row t's visible blocks are read): impl 0 =
// qsa_select's choice for a call of `rows` rows, 1 = one block per row (topk_kernel, the reference), 2 = the cluster
// kernel (CERE_ERR_ARGUMENT when ld > 65,536). For the kernel tests (cere_ktest_qsa_topk); rows <= 65,535.
cere_status qsa_topk(const float* logits, int64_t ld, const int64_t* positions_logical, int64_t rows,
                     int64_t seq_len, int impl, int32_t* out_indices, cudaStream_t stream);

// Sparse GQA, as qsa_sparse_attention_reference (kernel.py:322-345) with F32 scores, softmax and accumulation:
//   out[t, h] = softmax_j(q[t, h] . K[indices[t, j]] * scale) V[indices[t, j]] over the valid (>= 0) entries,
// zeros for a row without valid entries; query head h uses KV head h / 12. q [rows, 24 * 256] BF16 (row stride
// q_stride, normed and rotated by k6::prologue). K/V source as k6 (spec D3): either BF16 rows k/v (row strides in
// elements; logical position j at row j: chunk 0, no prefix) or the FP8 E4M3 cache k_cache/v_cache [rows, 2 * 256]
// bytes dequantized with scale 1.0 (position j at row slots[j], or j when slots is null); exactly one pair non-null.
// Output BF16 before the gate: out [rows, 24 * 256] (row stride out_stride). scratch: attention_scratch_bytes(rows).
// use_p4 (the model passes Workspace::prefill_p4, its prefill_attention lever): calls of >= p4::kMinRows rows (by
// `cls` below) with 16-byte aligned q/out and strides that are multiples of 8 run the P4 tensor-core kernel instead
// (p4_attention.h); calls of fewer rows run this one. `cls` (m210 D15⁗′; 0 = rows, else rows <= cls): the rows the P4
// and the split choice are made for, so a pinned call of fewer than p4::kMinRows rows whose class chose P4 runs P4
// (its rows equal the cold call's bit for bit; BeatSglang S7: an 18K hot tail of 16 rows was UNSUPPORTED before).
cere_status qsa_attention(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t rows, const bf16* k,
                          int64_t k_stride, const bf16* v, int64_t v_stride, const uint8_t* k_cache,
                          const uint8_t* v_cache, const int64_t* slots, float scale, void* scratch,
                          size_t scratch_bytes, bf16* out, int64_t out_stride, cudaStream_t stream, bool use_p4,
                          int64_t cls = 0);

// Query heads per warp of the rows kernel (rows_attention_kernel: the splits == 1 calls, i.e. more than kSplitMaxRows
// rows, with 16-byte aligned q/out and strides that are multiples of 8; track PrefillTtft, PT-02), chosen by the
// cold-L2 ktest timing (executor/tests/gpu/test_kernels_k9r_rows.py, suite kernels_k9r).
constexpr int kRowsHeads = 12;

// qsa_attention with the rows kernel's heads per warp given: rows_heads 0 runs attention_kernel for every call (the
// pre-PT-02 and pre-LK9-02 path, the ktests' bit-identity reference), 3, 6 or 12 the rows kernel for the splits == 1
// calls and, on the FP8 cache with a 16-byte aligned q and a q_stride that is a multiple of 8, split_attention_kernel
// for the splits > 1 calls (track LongK9, long-k9 design note LK9-02); anything else CERE_ERR_ARGUMENT. Every
// choice returns the same bytes (cere_ktest_qsa_attention_heads).
cere_status qsa_attention_heads(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t rows, const bf16* k,
                                int64_t k_stride, const bf16* v, int64_t v_stride, const uint8_t* k_cache,
                                const uint8_t* v_cache, const int64_t* slots, float scale, void* scratch,
                                size_t scratch_bytes, bf16* out, int64_t out_stride, cudaStream_t stream, bool use_p4,
                                int64_t cls, int rows_heads);

}  // namespace cere::k9
