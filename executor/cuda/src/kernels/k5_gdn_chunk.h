// K5 chunked prefill (m22n design note D2, D6): the gated delta rule of one sequence in chunks of 64 tokens, the WY
// form of FLA's chunk_gated_delta_rule as SGLang's Triton GDN backend dispatches it for an extend, with FLA's rounding
// points (BF16 q, k, A, w, u, v_new, per-chunk h, the intra-chunk attention matrix, o and the final state).
//
// math_reference: flash-linear-attention
// math_reference: sglang
//
// Written from the math of sglang/kernels/ops/attention/fla/ chunk.py, cumsum.py, chunk_fwd.py (fused kkt + solve),
// wy_fast.py, chunk_delta_h.py and chunk_o.py; nothing is vendored.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::k5 {

constexpr int kChunkTokens = 64;          // FLA CHUNK_SIZE (BT)
constexpr int64_t kChunkMaxTokens = 4096;  // one call: at most workspace_rows = min(max_context, 4096) rows

// Operand conversion of the TF32 block-merge MMAs (chunk_fwd.py:270-304, input_precision "tf32"), chosen by the C6
// rows: 0 = raw FP32 bits (the tensor core truncates the low 13 mantissa bits; what Triton's PTX for sm_120 passes;
// on FLA-normalised q, k bit-equal to FLA on every C6 case but 1-2 elements of o), 1 = cvt.rna.tf32.f32 first
// (rel-L2 up to 1.8e-3 vs FLA on the same inputs).
constexpr int kChunkTf32Variant = 0;

// One sequence of `tokens` (1..4096) rows, all buffers contiguous:
//   q, k  bf16 [tokens, 16, 128] raw; L2-normalised inside (k5::l2norm, eps 1e-6) to BF16 first;
//   v     bf16 [tokens, 48, 128];
//   g     f32 [tokens, 48] log-space decay, beta f32 [tokens, 48] (fused_gdn_gating's outputs);
//   state bf16 [48, 128 (v), 128 (k)] in/out (SGLang's K-last layout; zeros = fresh sequence);
//   o     bf16 [tokens, 48, 128].
// Value head h uses key head h / 3. Per chunk of 64 rows (G = per-chunk inclusive cumsum of g, FP32):
//   A = (I + tril(diag(beta) (k k^T * exp(G_i - G_j)), -1))^-1, stored BF16;
//   u = BF16(A BF16(v * beta)), w = BF16(A BF16(k * beta * exp(G)));
//   h_c = BF16(h); v_new = BF16(u - w h_c^T);
//   o = BF16((q h_c^T) * exp(G) * scale + (BF16(tril(q k^T * exp(G_i - G_j))) v_new) * scale);
//   h = h * exp(G_last) + k^T BF16((u - w h_c^T) * exp(G_last - G));
// final state = BF16(h). Deterministic: no atomics, no split-K; identical calls give identical bytes.
// The FP32 state (m210 D15⁗′, an end state at a 64-aligned row inside a call): `state_in32` (nullable) is an FP32
// state [48, 128 (v), 128 (k)] the recurrence starts from instead of `state` (which is then only written);
// `capture32` (nullable) receives, in that layout, the FP32 h at the start of chunk `capture_chunk` (the state as of
// row 64 * capture_chunk; capture_chunk == number of chunks is the final h before its BF16 rounding). Neither changes
// a computed value: a call from row 64 c with state_in32 = the capture at chunk c continues the longer call exactly.
// `state_in32_hi` / `capture32_hi` (nullable, each with its base pointer): value heads 24..47 of that state live there
// ([24, 128, 128]) and the base holds heads 0..23 only (the prefix cache's two-slot end state).
struct ChunkArgs {
    const __nv_bfloat16 *q;
    const __nv_bfloat16 *k;
    const __nv_bfloat16 *v;
    const float *g;
    const float *beta;
    __nv_bfloat16 *state;
    __nv_bfloat16 *o;
    int64_t tokens;
    float scale;
    const float *state_in32 = nullptr;
    float *capture32 = nullptr;
    int64_t capture_chunk = -1;
    const float *state_in32_hi = nullptr;
    float *capture32_hi = nullptr;
};

// Device scratch bytes `delta_rule_chunk` needs for `tokens` rows (normalised q and k, G, u/v_new, w and the BF16 state
// at every chunk boundary); 0 when tokens is outside 1..4096.
size_t chunk_workspace_bytes(int64_t tokens);

// `workspace` must hold chunk_workspace_bytes(args.tokens) bytes, 256-byte aligned. Launches on `stream` only.
cudaError_t delta_rule_chunk(const ChunkArgs &args, void *workspace, cudaStream_t stream);

}  // namespace cere::k5
