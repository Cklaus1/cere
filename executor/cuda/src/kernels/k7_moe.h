// K7 (m22 design note K7, the NVFP4 MoE row of "Model-specific kernel decisions", Q14): the MoE block of one layer,
// host launch API.
//
//   out = sum_k w_k * down_e(q2(silu(gate_e(q1(x))) * up_e(q1(x)))) + sigmoid(x . shared_expert_gate) * shared(x)
//
// Every step mirrors the reference (SGLang `Qwen2MoeSparseMoeBlock` with `--moe-runner-backend flashinfer_cutlass`
// and the ModelOpt NVFP4 MoE method):
//   * router: BF16 logits x . router^T (K2 GEMM), FP32 softmax over all experts, slot k = the k-th largest logit (ties:
//     lowest expert id), weights = the selected probabilities divided by their sum (renormalize);
//   * q1/q2: the K0 activation quantizer with ONE global scale per layer and GEMM, a1 = 1 / max(w13 input_scale) over
//     all experts and the gate/up shards, a2 = 1 / max(w2 input_scale) over all experts (FP32);
//   * W4A4 grouped GEMM on the checkpoint layout (see k0_nvfp4.h): per 16-element block the E2M1 x E2M1 products
//     are summed exactly (int8 dot products of 2 * e2m1), scaled by the two FP8 block scales (an exact product) and
//     accumulated in FP32; the result is multiplied by the expert's alpha and rounded to BF16, as the reference's
//     GEMM epilogue: g1_alphas[e] = max(w13 input_scale) * gate.weight_scale_2[e] for BOTH the gate and the up
//     output (the reference's single-alpha semantics), g2_alphas[e] = max(w2 input_scale) * down.weight_scale_2[e];
//   * SwiGLU on the BF16 gate/up results in FP32, rounded to BF16, then q2;
//   * combine: the BF16 expert outputs weighted in FP32 in slot order and rounded to BF16, then the shared expert
//     (BF16 K2 GEMMs gate/up, silu(gate) * up rounded to BF16, down) added as f + sigmoid(x . g) * shared in FP32.
// Rows are grouped by expert with a deterministic (stable) counting sort; every output element has a fixed reduction
// order, so the block is bit-reproducible run to run and independent of the batch composition.
#pragma once

#include "cere_executor.h"
#include "kernels/k2_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::kernels {

struct MoeDims {
    uint32_t hidden = 0;        // H, a multiple of 128
    uint32_t num_experts = 0;   // E, 1..1024
    uint32_t top_k = 0;         // K, 1..min(E, 32)
    uint32_t inter = 0;         // I (moe_intermediate_size), a multiple of 128
    uint32_t shared_inter = 0;  // S (shared_expert_intermediate_size)
};

// One layer's weights, device pointers in the checkpoint layout (16-byte aligned).
struct MoeLayer {
    const __nv_bfloat16 *router = nullptr;      // mlp.gate.weight [E, H]
    const uint8_t *gate_weight = nullptr;       // [E, I, H/2] E2M1 codes, element 2j in the low nibble
    const uint8_t *gate_scale = nullptr;        // [E, I, H/16] FP8 E4M3, linear layout
    const uint8_t *up_weight = nullptr;         // [E, I, H/2]
    const uint8_t *up_scale = nullptr;          // [E, I, H/16]
    const uint8_t *down_weight = nullptr;       // [E, H, I/2]
    const uint8_t *down_scale = nullptr;        // [E, H, I/16]
    const float *g1_alphas = nullptr;           // [E] max(w13 input_scale) * gate.weight_scale_2
    const float *g2_alphas = nullptr;           // [E] max(w2 input_scale) * down.weight_scale_2
    float a1_global_scale = 0.0f;               // FP32 1 / max(w13 input_scale)
    float a2_global_scale = 0.0f;               // FP32 1 / max(w2 input_scale)
    const __nv_bfloat16 *shared_gate = nullptr;         // [S, H]
    const __nv_bfloat16 *shared_up = nullptr;           // [S, H]
    const __nv_bfloat16 *shared_down = nullptr;         // [H, S]
    const __nv_bfloat16 *shared_expert_gate = nullptr;  // [H] (mlp.shared_expert_gate.weight)
#if defined(CERE_MOE_P3B)
    const float *act_scales = nullptr;  // P3b only (p3b/p3b_moe.h): device FP32 {a1_global_scale, a2_global_scale}
#endif
};

// The K2 plans the block runs (BF16 outputs, max_m >= the block's m): router (n = E, k = H), shared gate/up
// (n = S, k = H; run once with each weight) and shared down (n = H, k = S). `cls` (m210 D15⁗′, 0 = the block's m):
// the rows every row-count choice of the block (its GEMMs, P3b's tactic bucket) is made for, as GemmPlan::run's.
// The shared-expert fork (m23 design note P11R-23; all three null: none): `side` a non-blocking stream of the
// block's device, `fork` and `join` events (cudaEventDisableTiming) owned with it. A block that forks
// (moe_shared_forks) launches its shared expert on `side` after the work `stream` holds at the block's start, the
// router and the routed experts on `stream` meanwhile, and the combine on `stream` after the shared expert (one
// branch each in a captured graph). The launches and their operands are the serial block's, so every output bit is.
struct MoePlans {
    const GemmPlan *router = nullptr;
    const GemmPlan *shared_gate_up = nullptr;
    const GemmPlan *shared_down = nullptr;
    uint32_t cls = 0;
    cudaStream_t side = nullptr;
    cudaEvent_t fork = nullptr;
    cudaEvent_t join = nullptr;
};

// True when a block of m tokens without hook points that runs its own shared expert forks it: `plans` has the fork
// and its row class (plans.cls, else m) <= k2s::kMaxM (decode and verify rows: no K2 launch of the block uses the
// pinned split-K workspace or the FP8 staging buffer, which both branches would share; a larger class runs serially).
bool moe_shared_forks(const MoePlans &plans, uint32_t m);

// The forked shared expert of a block that forks (moe_shared_forks): records plans.fork on `stream`, launches the
// shared expert of x [m, H] on plans.side after it into `workspace` (moe_block's own shared-expert regions; at least
// moe_workspace_bytes(dims, m) bytes, 256-byte aligned) and records plans.join there; *shared = its down-projection
// output BF16 [m, H] in `workspace`. Nothing on `stream` may read *shared or reuse the workspace before
// moe_shared_join. CERE_ERR_ARGUMENT for invalid dims/pointers, a short workspace or a block that does not fork
// (nothing launched), CERE_ERR_INTERNAL on a launch, event or GEMM failure: what the side stream got is then
// joined into `stream` (best effort) before the error returns, so no launched work is left unjoined.
cere_status moe_shared_fork(const MoeDims &dims, const MoeLayer &layer, const MoePlans &plans,
                            const __nv_bfloat16 *x, uint32_t m, void *workspace, size_t workspace_bytes,
                            cudaStream_t stream, const __nv_bfloat16 **shared);

// Orders `stream`'s later work after the shared expert moe_shared_fork launched (cudaStreamWaitEvent on plans.join).
cere_status moe_shared_join(const MoePlans &plans, cudaStream_t stream);

// A caller's hold on a successful moe_shared_fork: moe_shared_join on the normal path (join()), else, on every other
// way out of the scope (an error return after the fork), the destructor joins best effort, so the side stream's
// shared expert is never left running against the block's workspace nor an unjoined branch in a stream capture.
class SharedForkScope {
public:
    SharedForkScope(const MoePlans &plans, cudaStream_t stream) : plans_(plans), stream_(stream) {}
    SharedForkScope(const SharedForkScope &) = delete;
    SharedForkScope &operator=(const SharedForkScope &) = delete;
    ~SharedForkScope() {
        if (armed_) {
            (void)moe_shared_join(plans_, stream_);
        }
    }
    void arm() { armed_ = true; }  // moe_shared_fork succeeded
    cere_status join() {
        armed_ = false;
        return moe_shared_join(plans_, stream_);
    }

private:
    const MoePlans &plans_;
    cudaStream_t stream_;
    bool armed_ = false;
};

// Bytes of the workspace moe_block needs for up to max_m tokens (0 for invalid dims).
size_t moe_workspace_bytes(const MoeDims &dims, uint32_t max_m);

// Router on BF16 logits [m, E]: ids int32 [m, K], weights float32 [m, K]. Every token's K ids are K distinct experts
// in [0, E) whatever the logits hold (NaN logits rank below every number; ties, lowest id first). A row of finite
// logits gets the softmax weights, renormalized over its K picks; a row whose softmax denominator is not a finite
// positive number (a NaN or +Inf logit) gets weights 0, never NaN.
cere_status moe_route(const __nv_bfloat16 *logits, int32_t *topk_ids, float *topk_weights, uint32_t m,
                      uint32_t num_experts, uint32_t top_k, cudaStream_t stream);

// Optional per-block taps (m22p design note Interfaces, D3/D5). Outputs (each nullable, BF16 [m, H], device):
// `routed` = the BF16 weighted sum of the routed experts before the shared expert is added (the value SGLang's
// FusedMoE `mlp.experts` module returns, before `fused_gate_sigmoid_mul_add` adds the gated shared expert in place;
// qwen2_moe.py:609-631), `shared` = the shared expert's down-projection output before its sigmoid gate
// (`mlp.shared_expert.down_proj`); each receives the row the combine adds.
// Inputs (each nullable, BF16 [m, H], device, read in stream order): `routed_in` replaces the routed sum (router,
// grouping and the expert GEMMs do not run), `shared_in` replaces the shared down-projection output (the shared
// expert's GEMMs do not run); the combine is unchanged (BF16 routed row + sigmoid(x . g) * shared row, FP32).
// Point hook (nullable `point`): called on the host in block order with `values` device BF16 [m, cols] row-major;
// `name` is layer-relative:
//   "gemm.moe_router.in" (x, cols H), "gemm.moe_router.out" (router logits, cols E), [detail] "mlp.gate" (the same
//   logits after the hook), then per shared GEMM call (gate, then up): "gemm.moe_shared_gate_up.in" (x),
//   "gemm.moe_shared_gate_up.out" (cols S), [detail] "mlp.shared_expert.gate_up_proj" (gate | up, cols 2 S),
//   "gemm.moe_shared_down.in" (SwiGLU output, cols S), "gemm.moe_shared_down.out" (cols H).
// The hook may overwrite a "*.out" point's values in stream order (a replacement the rest of the block consumes); it
// must not write any other point. [detail] points are called only when `detail` is true (gate | up is staged in the
// workspace's SwiGLU region, which is dead by then). Points of a part replaced by `routed_in` / `shared_in` /
// `routed_rows` are not called. A non-OK status aborts the block with it. A null `taps` (or every field null) costs
// nothing.
// Post-route hook (nullable `route`, weightlab design note WL-08): called on the host, with `ctx`, after moe_route
// and before anything reads the routing, with the device ids int32 / weights FP32 [m, top_k] (row-major). It may
// overwrite them in stream order (each token's ids distinct and in [0, E), weights finite): the grouping, the expert
// GEMMs, the combine and P3b's topk copy-out read what it leaves. Not called when the router does not run
// (`routed_in`, `routed_rows`). Independent of `point`: setting only `route` keeps the shared-expert fork.
// `routed_rows` (nullable; with `routed_pos` and `routed_scales`, exclusive with `routed_in`): the per-(token, slot)
// expert outputs BF16 [m * top_k, H] in any row order (P3b's permuted GEMM2 rows, m23 design note P13), row of
// (token t, slot k) = `routed_pos`[k * m + t] (slot-major, FlashInfer's unpermuted_row_to_permuted_row), weight
// `routed_scales`[t * top_k + k] (FP32 [m, top_k]). The router, grouping and expert GEMMs do not run; the combine forms
// the routed row as FlashInfer's finalizeMoeRoutingKernel does (FP32 acc = 0, then acc = fma.rn.ftz(w_k, y_k, acc) in
// slot order, BF16 round-to-nearest-even), so it is bit-equal to FlashInfer's finalized row fed through `routed_in`.
struct MoeTaps {
    __nv_bfloat16 *routed = nullptr;
    __nv_bfloat16 *shared = nullptr;
    const __nv_bfloat16 *routed_in = nullptr;
    const __nv_bfloat16 *shared_in = nullptr;
    const __nv_bfloat16 *routed_rows = nullptr;
    const int32_t *routed_pos = nullptr;
    const float *routed_scales = nullptr;
    cere_status (*point)(void *ctx, const char *name, __nv_bfloat16 *values, uint32_t rows, uint32_t cols,
                         cudaStream_t stream) = nullptr;
    void *ctx = nullptr;
    cere_status (*route)(void *ctx, int32_t *topk_ids, float *topk_weights, uint32_t rows, uint32_t top_k,
                         cudaStream_t stream) = nullptr;
    bool detail = false;
};

// out BF16 [m, H] from x BF16 [m, H]. `workspace` (256-byte aligned) holds at least moe_workspace_bytes(dims, m).
// No allocation and no synchronization (a hook may synchronize). CERE_ERR_ARGUMENT for invalid dims/pointers or a
// short workspace, CERE_ERR_INTERNAL on a launch or GEMM failure. `taps` (nullable): see MoeTaps.
cere_status moe_block(const MoeDims &dims, const MoeLayer &layer, const MoePlans &plans, const __nv_bfloat16 *x,
                      uint32_t m, __nv_bfloat16 *out, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                      const MoeTaps *taps = nullptr);

}  // namespace cere::kernels
