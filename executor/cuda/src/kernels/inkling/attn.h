// Inkling 2 Mini attention (inkling design note 3.3, slice S2): every op between the BF16 projections and o_proj,
// for sliding (window 1,024 over a per-visit KV ring) and full (paged KV pool) layers. The math, op by op and in
// order, is the CPU reference executor/ref/inkling/attn_ref.h; each launcher's output equals it bit for bit (G1).
//
// The forward (S5) per attention visit, n rows at positions given per row:
//   GEMMs (S5): q_raw [n, 3072], r_raw [n, 384], sink [n, 24], gate [n, 3072] from the q input; k/v GEMM + S3's conv
//   -> k_conv / v_conv [n, 1024] (BF16, row-major, any row stride)
//   1. post(...)         q_raw -> q, r_raw -> r, k_conv -> k, v_conv -> v (may run in place)
//   2. prefill(...) or decode(...)   -> out [n, 3072] BF16 = gated attention, the input of o_proj
//   3. kv_store(...)     the call's k/v rows into the ring (sliding) or the pool (full). AFTER step 2: the attention
//                        reads keys before pos0 from the cache and the call's own keys from k/v; storing first would
//                        overwrite ring slots that still hold keys inside the window of the call's first rows.
// Rows of one call: a table of sequences (Seq, device memory), each a run of consecutive rows [row, row + rows) at
// positions pos0 .. pos0 + rows - 1 of one session, with that session's cache of this visit. Prefill: usually one
// sequence (a chunk, rows <= kMaxPrefillRows); decode: up to kMaxDecodeRows rows in total over up to that many
// sequences (one row per session, or several draft rows of one session for a verify).
//
// Determinism (G4, CERE D9): a row's output depends only on its own q/r/sink/gate row, its position and the K/V
// bits at its key positions. Keys are visited in fixed absolute tiles (64), spans (256) and groups (4,096); prefill
// and decode implement the same order, so a row's output is bit-identical across chunking (512 vs 4,096), batch
// composition (1 vs 4 sessions), row order, prefill vs decode and run to run (no atomics, no split depending on M).
//
// Layouts (all BF16 unless stated, row-major, strides in elements):
//   q  [rows, 24 * 128]  query head h at columns h*128 ..; r [rows, 24 * 16]; k, v [rows, 8 * 128] (KV head g at
//   g*128); sink [rows, 24]; gate [rows, 24 * 128]; out [rows, 24 * 128]. Query head h uses KV head h / 3.
//   q_scale [24, 128], rel_scale [24, 16], rel_proj [16, 1024] (the checkpoint's q_l2_clip.scale,
//   rel_l2_clip.scale, rel_logits_proj.proj).
//   Sliding cache: per (session, visit) a ring K and a ring V, each [kRingSlots = 1024, 1024]; key at position j in
//   slot j % 1024 (D-I3). Full cache: per full visit a K pool and a V pool, each [pool_tokens, 1024]; key at position
//   j in pool row slots[j] (int64, device: the session's slot map of host/kv_pool.h, 64-token pages, slots[p] =
//   page * 64 + p % 64); slots null = row j (a contiguous per-session buffer).
//   positions: int64 device [rows], the absolute position of each row (read by post on full layers).
// Every launcher enqueues on `stream`, never synchronizes, and returns cudaErrorInvalidValue (nothing launched) on
// a malformed argument. Host-side arguments are bounds only (n_seqs, max_rows, max_pos, total_rows), so a captured
// graph replays at any positions within its bounds; the positions, rows and pointers are read on the device.
//
// Math: executor/ref/inkling/attn_ref.h (REF transformers modeling_inkling2_loop.py @046bdb8c).
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inkling/attn_ref.h"

namespace cere::inkling::attn {

using bf16 = __nv_bfloat16;
using ref::attn::kFull;
using ref::attn::kSliding;
using ref::attn::Mutant;

constexpr int kHeads = ref::attn::kHeads;        // 24
constexpr int kKvHeads = ref::attn::kKvHeads;    // 8
constexpr int kHeadDim = ref::attn::kHeadDim;    // 128
constexpr int kQRow = ref::attn::kQRow;          // 3072
constexpr int kKvRow = ref::attn::kKvRow;        // 1024
constexpr int kRelRow = ref::attn::kRelRow;      // 384
constexpr int kRingSlots = ref::attn::kRingSlots;
constexpr int64_t kMaxPrefillRows = 4096;  // rows of one sequence in a prefill call
constexpr int64_t kMaxDecodeRows = 64;     // rows of a decode call (all sequences)

// 1. The post-projection ops of one call of `rows` rows (one launch; each part runs when its input is non-null):
//   q_out = q post (scale, L2 clip 22.627, then on full layers the log scale tau(position)), q_in -> q_out;
//   r_out = rel post (scale, L2 clip 8.0, tau on full layers);
//   k_out = k post (L2 clip 22.627); v_out = v post (unweighted RMSNorm, eps 1e-6).
// kind: kSliding or kFull; positions: required for full layers when q_in or r_in is non-null. Outputs may alias
// their inputs (same stride). mutant: ref::attn::kNone in the forward; kNoLogScale (ktest only) skips tau.
struct PostArgs {
    int kind = kSliding;
    int64_t rows = 0;
    const int64_t *positions = nullptr;
    const bf16 *q_in = nullptr;
    int64_t q_in_stride = kQRow;
    const bf16 *q_scale = nullptr;
    bf16 *q_out = nullptr;
    int64_t q_out_stride = kQRow;
    const bf16 *r_in = nullptr;
    int64_t r_in_stride = kRelRow;
    const bf16 *rel_scale = nullptr;
    bf16 *r_out = nullptr;
    int64_t r_out_stride = kRelRow;
    const bf16 *k_in = nullptr;
    int64_t k_in_stride = kKvRow;
    bf16 *k_out = nullptr;
    int64_t k_out_stride = kKvRow;
    const bf16 *v_in = nullptr;
    int64_t v_in_stride = kKvRow;
    bf16 *v_out = nullptr;
    int64_t v_out_stride = kKvRow;
    int mutant = 0;
};
cudaError_t post(const PostArgs &a, cudaStream_t stream);

// One sequence of a call (device memory, an array of n_seqs). k_cache/v_cache: this session's ring (sliding) or
// this visit's pool (full); slots: the session's slot map (full; null = identity; ignored for sliding).
struct Seq {
    int64_t row;   // first row of the sequence in q/r/k/v/sink/gate/out
    int64_t rows;  // >= 1
    int64_t pos0;  // position of its first row (keys < pos0 come from the cache)
    bf16 *k_cache;
    bf16 *v_cache;
    const int64_t *slots;
};

// 2. Gated attention over the sequences' rows. q, r: post outputs; k, v: post outputs of the same rows (the keys at
// positions >= pos0); sink, gate: the projections. out row = for each query head: softmax over
// [scores with rel bias over the head's keys] ++ [sink] (sink with zero value), times V, BF16, times
// sigmoid(gate), BF16. Rows of different sequences are independent; out may not alias any input.
// Bounds: n_seqs >= 1; max_rows >= every seq.rows; max_pos >= every seq.pos0 + seq.rows (decode on full layers
// sizes its grid with it; prefill ignores it).
struct AttnArgs {
    int kind = kSliding;
    const Seq *seqs = nullptr;
    int n_seqs = 0;
    int64_t max_rows = 0;
    int64_t max_pos = 0;
    const bf16 *q = nullptr;
    int64_t q_stride = kQRow;
    const bf16 *r = nullptr;
    int64_t r_stride = kRelRow;
    const bf16 *rel_proj = nullptr;
    const bf16 *k = nullptr;
    int64_t k_stride = kKvRow;
    const bf16 *v = nullptr;
    int64_t v_stride = kKvRow;
    const bf16 *sink = nullptr;
    int64_t sink_stride = kHeads;
    const bf16 *gate = nullptr;
    int64_t gate_stride = kQRow;
    bf16 *out = nullptr;
    int64_t out_stride = kQRow;
    // ktest only (fail-first): kNoWindow (sliding sees every past key), kNoSink, kBiasOffByOne (bias of d + 1);
    // prefill only: kMutantLaneSum (q.k's 32 lane chains summed in lane order instead of reduce_dot<32>'s tree).
    int mutant = 0;
};
constexpr int kMutantLaneSum = 5;

// Prefill kernel (PA): one block per 16 rows of a sequence and KV head walks every key of those rows in 64-key tiles
// (no workspace; ~99 KB dynamic shared memory, opted in at launch). max_rows <= kMaxPrefillRows. Any number of
// sequences.
cudaError_t prefill(const AttnArgs &a, cudaStream_t stream);

// Decode kernels: per row, span of 256 keys and KV head a cluster of 8 blocks writes the span partials (each block
// streams 1/8 of the span's K and V), then one block per row folds them and runs the epilogue; full layers repeat
// this in passes of kDecodeSpansPerPass spans (16,384 keys) so the workspace is bounded.
// total_rows: >= every seq.row + seq.rows, <= kMaxDecodeRows; max_rows <= kMaxDecodeRows. Bit-identical to prefill.
constexpr int kDecodeSpansPerPass = 64;
size_t decode_workspace_bytes(int64_t total_rows);  // 256-byte aligned base required
cudaError_t decode(const AttnArgs &a, int64_t total_rows, void *workspace, size_t workspace_bytes,
                   cudaStream_t stream);

// 3. KV store of the sequences' rows (k, v: the post outputs, [rows, 1024] at k_stride / v_stride): sliding: rows
// i >= seq.rows - 1024 to ring slot (pos0 + i) % 1024; full: row i to pool row slots[pos0 + i] (or pos0 + i).
// max_rows >= every seq.rows. Bit copies.
cudaError_t kv_store(int kind, const Seq *seqs, int n_seqs, int64_t max_rows, const bf16 *k, int64_t k_stride,
                     const bf16 *v, int64_t v_stride, cudaStream_t stream);

}  // namespace cere::inkling::attn
