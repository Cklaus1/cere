// K13 (m29 design note D3, K2): the BF16 MoE block of the MTP draft layer (`mtp.layers.0.mlp`), host launch API.
//
//   out = sum_k w_k * down_e(silu(gate_e(x)) * up_e(x)) + sigmoid(x . shared_expert_gate) * shared(x)
//
// The same block as K7 (kernels/k7_moe.h) on BF16 expert weights instead of NVFP4: router and routing are K7's
// (moe_route on the K2 BF16 logits); the expert GEMMs read BF16 weights and accumulate in FP32 with a fixed k order
// per output element, round gate/up to BF16, SwiGLU in FP32 rounded to BF16, down accumulated in FP32 rounded to
// BF16; the combine (slot order, FP32, rounded to BF16, then + sigmoid(x . g) * shared in FP32) is K7's. Rows are
// grouped by expert with a stable counting sort, and every output element's reduction order is independent of the
// other rows, so row r of a call with m rows is bit-equal to the single-row call on that row for every m (the
// shared/router GEMMs keep this when `plans` are rows plans (GemmPlan::init_rows) for m <= 16; prefill extends at
// m up to 4,096 use the max_m plans and are judged against the oracle only). A routed tail of at most 16 rows on
// NVFP4 experts (the MTP draft's calls) runs a one-block grouping and small-tail GEMVs (all of a warp's weight loads
// issued up front, programmatic dependent launches) with the same per-element arithmetic, bit-equal to the general
// path's grouped GEMVs (draft-span design note DS-2).
#pragma once

#include "cere_executor.h"
#include "kernels/k2_gemm.h"
#include "kernels/k7_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::kernels {

// One layer's weights, device pointers in the checkpoint layout (16-byte aligned).
struct MoeBf16Layer {
    const __nv_bfloat16 *router = nullptr;              // mlp.gate.weight [E, H]
    const __nv_bfloat16 *gate_up = nullptr;             // mlp.experts.gate_up_proj [E, 2 I, H]: gate rows 0..I-1
    const __nv_bfloat16 *down = nullptr;                // mlp.experts.down_proj [E, H, I]
    const __nv_bfloat16 *shared_gate = nullptr;         // [S, H]
    const __nv_bfloat16 *shared_up = nullptr;           // [S, H]
    const __nv_bfloat16 *shared_down = nullptr;         // [H, S]
    const __nv_bfloat16 *shared_expert_gate = nullptr;  // [H] (mlp.shared_expert_gate.weight)
    // NVFP4 W4A16 routed experts (track Fp4S3, fp4local design note "S3 perf plan"; the S3 FP4-Local artifact's
    // fused MTP experts). Null gate_up_q: the BF16 gate_up/down above. Else gate_up and down are null and the experts
    // are packed E2M1 (two per byte, the even column in the low nibble) with E4M3 block scales (16 k each) and F32
    // globals: gate_up_q [E, 2 I, H / 2] (gate rows 0..I-1), gate_up_scale [E, 2 I, H / 16], gate_up_global [E, 2]
    // (gate, up); down_q [E, H, I / 2], down_scale [E, H, I / 16], down_global [E]. The expert GEMMs read each weight
    // as BF16(RN(RN(e2m1 * scale) * global)) in F32, the value cerec convert's S2 path stores, in the BF16 kernel's
    // reduction order, so out is bit-equal to the BF16 call on the S2 tensors.
    const uint8_t *gate_up_q = nullptr;
    const uint8_t *gate_up_scale = nullptr;
    const float *gate_up_global = nullptr;
    const uint8_t *down_q = nullptr;
    const uint8_t *down_scale = nullptr;
    const float *down_global = nullptr;
};

// Bytes of the workspace moe_bf16_block needs for up to max_m tokens (0 for invalid dims).
size_t moe_bf16_workspace_bytes(const MoeDims &dims, uint32_t max_m);

// out BF16 [m, H] from x BF16 [m, H]; dims as K7 (H, E, K, I, S). `plans` as K7's MoePlans (BF16 outputs, max_m >= m).
// `first` in [0, m): the router and the shared expert run on all m rows (their GEMM plans see the call's m), the
// routed experts and the combine on rows [first, m) alone, so rows [first, m) of out are bit-equal to the first = 0
// call's and rows [0, first) of out are not written (the MTP extend's last_only call reads row m - 1 only).
// `workspace` (256-byte aligned) holds at least moe_bf16_workspace_bytes(dims, m). No allocation, no synchronization.
// CERE_ERR_ARGUMENT for invalid dims/pointers, first >= m or a short workspace, CERE_ERR_INTERNAL on a launch or GEMM
// failure.
cere_status moe_bf16_block(const MoeDims &dims, const MoeBf16Layer &layer, const MoePlans &plans,
                           const __nv_bfloat16 *x, uint32_t m, uint32_t first, __nv_bfloat16 *out, void *workspace,
                           size_t workspace_bytes, cudaStream_t stream);

}  // namespace cere::kernels
