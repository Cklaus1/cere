// Inkling 2 Mini block drafter (inkling design note 3.5, S9 phase B): the drafter's own kernels, everything between
// its BF16 projections (kernels/inkling/linear.h, kernels/k2_gemm.h) that the target's kernels do not already cover.
// The math, op by op and in order, is the CPU reference executor/ref/inkling/drafter_ref.h (+ norm_ref.h); each
// launcher's output equals it bit for bit (G1, executor/tests/gpu/test_inkling_kernels_drafter.py). The model code
// that orders them is model/inkling/drafter.cpp (DRAFTER.md sections 3-5).
//
// Layouts (BF16 unless stated, row-major, strides in elements): a hidden row 2304; q [rows, 24 * 128] (query head h
// at h*128); k, v [rows, 8 * 128] (KV head g = h / 3); a block is 8 consecutive rows (positions L .. L + 7).
// RoPE rows `cs`: FP32 [rows, 128] = cos[64] then sin[64] of the row's position, staged by the host
// (drafter_ref.h rope_cos_sin).
// Every launcher enqueues on `stream`, never synchronizes and returns cudaErrorInvalidValue (nothing launched) on a
// malformed argument. Host-side arguments are counts and bounds only: per-row positions, pointers and block starts
// are read on the device, so a captured graph replays for any values within its bounds.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling::draft {

using bf16 = __nv_bfloat16;

constexpr int kBlock = 8;          // rows of a block (the anchor + 7 masks)
constexpr int kHeads = 24;         // query heads
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kQRow = kHeads * kHeadDim;      // 3072
constexpr int kKvRow = kKvHeads * kHeadDim;   // 1024
constexpr int kRopeRow = kHeadDim;            // cos[64] + sin[64]
constexpr int kWindow = 1024;                 // local layers' window = ring slots
constexpr int kRank = 256;
constexpr int kProj = 3 * kRank;              // 768: gate | cand | out
constexpr int kStepVocab = 200058;            // ids >= this never win a Markov step

// ---- mix_rmsnorm (drafter_ref.h mix_rmsnorm) ----------------------------------------------------------------------
// For i < rows and p < planes: plane p's row i = mix_rmsnorm(taps row src[i] (src null: row i), alpha row a0 + p,
// w[p]). taps: row r at taps + r * tap_ld, tap t at + t * 2304 (tap order 5, 23, 41, 48). alpha FP32 [4][4] (device,
// BF16 values). Output: plane p row i at y + p * plane_stride + i * ldy, or (dst non-null, planes == 1) at dst[i].
struct MixArgs {
    const bf16 *taps = nullptr;
    int64_t tap_ld = 0;
    const int32_t *src = nullptr;  // device [rows] or null
    int64_t rows = 0;
    const float *alpha = nullptr;
    int a0 = 0;
    int planes = 1;  // 1..3
    const bf16 *w[3] = {nullptr, nullptr, nullptr};
    bf16 *y = nullptr;
    int64_t ldy = 0;
    int64_t plane_stride = 0;
    bf16 *const *dst = nullptr;  // device [rows] or null
};
cudaError_t mix_rmsnorm(const MixArgs &a, cudaStream_t stream);

// ---- residual add + RMSNorm (norm_ref.h order), the drafter's pre-norm step ----------------------------------------
// Row i < rows maps to input row r = (i / take) * group + skip + i % take. res_out[r] = add_in ? bf16(add_in[r] +
// res_in[r]) : res_in[r] (copied); y[i] = bf16(rmsnorm(res_out[r]) * w). All rows 2304 wide at stride 2304.
// res_out may alias res_in (not add_in); y aliases nothing.
struct AddNormArgs {
    const bf16 *add_in = nullptr;
    const bf16 *res_in = nullptr;
    bf16 *res_out = nullptr;
    const bf16 *w = nullptr;
    bf16 *y = nullptr;
    int64_t rows = 0;
    int group = 1, skip = 0, take = 1;
};
cudaError_t add_rmsnorm(const AddNormArgs &a, cudaStream_t stream);

// act[r, i] = bf16(silu(g[r, i]) * u[r, i]), silu(x) = div(x, add(1, exp(-x))) (SGLang's SiluAndMul: FP32, one
// rounding). g, u, act [rows, n] at stride n (n a multiple of 8).
cudaError_t silu_mul(const bf16 *g, const bf16 *u, bf16 *act, int64_t rows, int64_t n, cudaStream_t stream);

// ---- head norm + RoPE ----------------------------------------------------------------------------------------------
// The block's fused q/k form (drafter_ref.h head_norm_rope, ctx = false) in place: q [rows, 3072] (24 heads, q_norm)
// and k [rows, 1024] (8 heads, k_norm) at row strides ldq / ldk; row r rotated by cs[r].
cudaError_t qk_norm_rope(bf16 *q, int64_t ldq, const bf16 *q_norm, bf16 *k, int64_t ldk, const bf16 *k_norm,
                         const float *cs, int64_t rows, cudaStream_t stream);

// One context row's destination: the session's caches (DraftSession of model/inkling/drafter.h) and the position.
struct CtxRow {
    bf16 *ring;       // [2 local layers][K, V][1024 slots][1024]; position j at slot j % 1024
    bf16 *full;       // [K, V][max_ctx][1024]; position j at row j
    int64_t pos;
    int64_t max_ctx;
    int64_t ring_write;  // 0: a later position of the same call takes this row's ring slot (the row skips the ring)
};
// Context store (DRAFTER.md section 3): for row i < rows and layer l < 3, the k_proj_ctx output k[l][i] (BF16 [8 x
// 128], at k + l * layer_stride + i * 1024) becomes bf16(rope(float(bf16(rmsnorm_128(k) * k_norm_ctx[l])))) (the
// ctx form of head_norm_rope) at the row's position in layer l's cache; v[l][i] is copied unchanged. Layers 0 and 2:
// ring plane 0 and 1 (rows with ring_write = 0 skip them: the caller clears it on every row whose slot a larger
// position of the same call also writes, so a slot ends holding its largest position, as in-order stores would);
// layer 1: the full cache. rows device [rows] (CtxRow), cs [rows, 128].
cudaError_t ctx_store(const bf16 *k, const bf16 *v, int64_t layer_stride, const bf16 *const k_norm[3],
                      const float *cs, const CtxRow *rows_tab, int64_t rows, cudaStream_t stream);

// ---- block attention (drafter_ref.h attend) ------------------------------------------------------------------------
// One block's keys on one layer: positions [lo, L + 8), lo = max(0, L - 1024) on local layers (window anchored at the
// block start), 0 on the full layer; j < L from ctx_k/ctx_v (row j % 1024 when ring, else row j), j >= L the block's
// own k/v rows.
struct BlockSeq {
    const bf16 *ctx_k;
    const bf16 *ctx_v;
    int64_t L;
    int64_t ring;  // 0 or 1
};
// Fail-first mutants (ktest only): kCausal, a causal block mask (row r sees keys <= L + r); kRowWindow, a per-row
// sliding window on local layers (row r sees keys >= max(0, L + r - 1023), the target's window, not the block's).
enum class AttnMutant : int { kNone = 0, kCausal = 1, kRowWindow = 2 };
struct BlockAttnArgs {
    const BlockSeq *seqs = nullptr;  // device [blocks]
    int64_t blocks = 0;              // 1..64
    bool local = false;              // layers 0 and 2
    const bf16 *q = nullptr;         // [blocks * 8, 3072] (after qk_norm_rope)
    const bf16 *k = nullptr;         // [blocks * 8, 1024]
    const bf16 *v = nullptr;
    bf16 *o = nullptr;               // [blocks * 8, 3072], no alias
    int64_t bound = 0;               // host: every block's L + 8 <= bound
    void *ws = nullptr;              // block_attention_workspace_bytes(blocks), 256-aligned
    size_t ws_bytes = 0;
    AttnMutant mutant = AttnMutant::kNone;
};
size_t block_attention_workspace_bytes(int64_t blocks);
cudaError_t block_attention(const BlockAttnArgs &a, cudaStream_t stream);

// ---- Markov head ---------------------------------------------------------------------------------------------------
// The rnn input of step j for block b < blocks (row b of inp, BF16 [blocks, 5120]): [state[b], markov_w1[prev_b],
// hid[b * hid_rows + j], *f_rnn[b]]; prev_b = anchors[b] at j = 0, else the previous step's pick (keys[b]), which is
// also written to ids[b * 7 + j - 1] and keys[b] reset to 0. assemble = false: the pick only (after the last step).
cudaError_t markov_input(const bf16 *state, const bf16 *w1, const bf16 *hid, int hid_rows, int j,
                         const bf16 *const *f_rnn, const uint32_t *anchors, unsigned long long *keys, uint32_t *ids,
                         bf16 *inp, int64_t blocks, bool assemble, cudaStream_t stream);
// drafter_ref.h markov_gates on proj = bf16(acc + bias) (acc FP32 [blocks, 768], the rnn_proj accumulator; bias BF16
// [768] or null: proj = bf16(acc)). state BF16 [blocks, 256] updated in place, latent BF16 [blocks, 256].
cudaError_t markov_gates(const float *acc, const bf16 *bias, bf16 *state, bf16 *latent, int64_t blocks,
                         cudaStream_t stream);
// drafter_ref.h markov_step for blocks b < blocks: latent [blocks, 256], fold_w [vocab][256], fold_b [vocab], base
// row b at base + b * ld_base ([200058]). keys[b] (zero on entry) gets the step's pick: key = (ordered logit << 32) |
// ~id over the non-NaN logits (atomicMax: the max logit, then the first id); 0 = no candidate (id 0, as the ref).
// logits (nullable): BF16 [blocks, ld_logits] the step logits.
cudaError_t markov_vocab(const bf16 *latent, const bf16 *fold_w, const bf16 *fold_b, const bf16 *base,
                         int64_t ld_base, int64_t blocks, unsigned long long *keys, bf16 *logits, int64_t ld_logits,
                         cudaStream_t stream);
__host__ __device__ inline uint32_t key_id(unsigned long long key) {
    return key == 0 ? 0u : ~uint32_t(key & 0xffffffffull);
}

// ---- fold (drafter_ref.h fold_row) on the GPU: FP64, k ascending, exact products, __dadd_rn ----------------------
// lm BF16 [rows][2304] (the unembed's first rows), hw BF16 [2304][256], hb BF16 [2304] -> fold_w [rows][256],
// fold_b [rows].
cudaError_t fold(const bf16 *lm, int64_t rows, const bf16 *hw, const bf16 *hb, bf16 *fold_w, bf16 *fold_b,
                 cudaStream_t stream);

}  // namespace cere::inkling::draft
