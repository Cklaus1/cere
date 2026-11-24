// K5R: the STP stage-a region probe (selftune design note D11, D12, phase STP; libcere_ktest.so only, never a
// build input of libcere_executor.so's sources). One GDN layer's decode mixer (k5::layer_forward DECODE at 1 row;
// gemm_segments + k5::chain_core + out_proj, the verify composition of model forward_verify_rows, at 2..4 rows) as one
// persistent "region kernel": a fixed instruction list over device-function copies of today's kernels' arithmetic,
// with the out_proj weights (and the core's state) prefetched into L2 while the input projections stream.
//
// Program (one launch, every CTA resident, a grid barrier between waves):
//   wave 1  GEMM  hidden [rows, 2560] x (in_proj_qkv | in_proj_z | in_proj_b | in_proj_a)^T -> mixed, z, b, a
//                 (16-column tiles; each tile also prefetches its share of out_proj, the SSM and conv state and the
//                 conv weights into L2; the input weights load with an L2 evict-first policy)
//   wave 2  CORE  per value head and 64-row half: the conv (q, k recomputed per warp; v owned), the delta-rule decode
//                 of every row, o; the head's second arriving CTA runs the gated RMSNorm of the head -> y
//   wave 3  GEMM  y [rows, 6144] x out_proj^T -> out [rows, 2560] (8-column tiles, L2-resident weights)
//           CONV  the q and k channels' conv state (and saves), whose old values wave 2 read
//
// Bits: every output (out, the conv and SSM state, the saves) is bit-equal to today's ops on the same inputs:
//   GEMMs   k2_small_m's contract (k2_small_m.h): slice p of split_warps(k) = 16 warps takes steps p, p + 16, ...
//           with the same m16n8k16 MMAs, the slices summed in slice order (one CTA, no cluster at k 2,560 and 6,144);
//           the tile width and the load cache policy change no bit. The host refuses (CERE_ERR_UNSUPPORTED) when
//           split_warps no longer gives 16 at both k.
//   conv    k5_gdn.cu's conv_silu_token (BF16 tap products, FP32 sum, Triton's SiLU lowering), state as conv1d_silu_rows
//   decode  k5_gdn_decode.cu's decode_step per token, the state stored / rounded as delta_rule_decode_rows
//   norm    k1_primitives.cu's rmsnorm_kernel<true> at 128 columns (one warp per row)
#pragma once

#include "cere_executor.h"
#include "kernels/k5_gdn_layer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::ktest::k5r {

constexpr int kMaxRows = 4;  // a verify chain (k5::kChainMaxRows)

// Buffers of one call, all device pointers, contiguous row-major, 16-byte aligned. hidden [rows, 2560] bf16;
// conv_state [10240, 3] and ssm_state [48, 128, 128] bf16 in/out; conv_saves[t] / ssm_saves[t] (t < rows - 1): the
// state after row t (as chain_core's saves); out [rows, 2560] bf16.
struct Args {
    const __nv_bfloat16 *hidden;
    k5::LayerWeights w;
    __nv_bfloat16 *conv_state;
    __nv_bfloat16 *ssm_state;
    __nv_bfloat16 *conv_saves[kMaxRows - 1];
    __nv_bfloat16 *ssm_saves[kMaxRows - 1];
    __nv_bfloat16 *out;
    int rows;
};

// Workspace of the region at `rows` rows (its counters and the mixed, z, b, a, o, y intermediates); 0 outside
// 1..kMaxRows. The caller zeroes it once (workspace_init); every launch leaves the counters zero again.
size_t workspace_bytes(int rows);
cudaError_t workspace_init(void *workspace, size_t bytes, cudaStream_t stream);

// The kernel's arithmetic variant: kExact, or kMutantSliceSum (fail-first only: every GEMM's K slices, in-projections
// and out_proj, summed in reverse slice order, which must break the bit-exact check).
enum class Variant : int { kExact = 0, kMutantSliceSum = 1 };

// Enqueues the region on `stream` (one cooperative launch, graph-capturable). CERE_ERR_ARGUMENT for buffers outside
// the contract, CERE_ERR_UNSUPPORTED when the small-M slice split differs from the region's, else the launch status.
cere_status forward(const Args &args, void *workspace, size_t workspace_bytes, Variant variant, cudaStream_t stream);

}  // namespace cere::ktest::k5r
