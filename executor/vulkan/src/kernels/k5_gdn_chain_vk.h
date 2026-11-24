// VulkanAmd P3 (T3): k5::chain_core on Vulkan, the GDN mixer core over the rows of a verify step's chains
// (executor/cuda/src/kernels/k5_gdn_layer.h:110-138, definition k5_gdn_layer.cu:221, 251-316).
//
// Contract (the CUDA header's): `rows` = the sum of chains.rows, chains back to back from row 0; inputs as layer_core's
// (row r of mixed [rows, 10240], z [rows, 6144], b / a [rows, 48], all contiguous BF16); y bf16 [rows, 6144] (may alias
// mixed but no other input). Bit-equal, per chain, to its rows' layer_core(DECODE) calls in order on states[c]
// (k5_gdn_layer_vk.h): y, the final conv and ssm state, and saves[r] holding the state row r's call leaves (the undo
// snapshot right before row r + 1) for every row that is not its chain's last, written in-kernel.
//
// Composition, as the CUDA definition (three steps): vkk5::conv1d_silu_rows (one grid slice per chain; conv state and
// conv saves), vkk5::delta_rule_decode_rows (one grid slice per chain; ssm state and ssm saves) and one
// vkk1::rmsnorm_gated_sigmoid over rows * 48 head rows (eps kNormEps). Recorded into the context's open command buffer
// with a barrier between dependent dispatches and none after the last: the caller records a barrier before reading y,
// the states or the saves. No new shader: the per-row arithmetic of each step is the one layer_core(DECODE) runs
// (conv1d_silu / delta_rule_decode / the same gated norm), and the *_rows shaders are those shaders' chain forms
// (k5_gdn_vk.h, asserted equal to sequential one-token calls by k5_pieces_test / k5_decode_test).
//
// Differences from the CUDA header: buffer device addresses; the two *_rows steps read their chain tables from a
// caller-provided host-visible buffer (`table`, kChainCoreTableBytes from the 16-byte aligned `table_off`, written at
// record time through the mapping; not to be rewritten until the submit completes), because the 384-byte tables do not
// fit the 128-byte push-constant block. Errors as the CUDA header (CERE_ERR_ARGUMENT, nothing recorded), plus the
// Vulkan alignment rules (mixed / b / a / conv states 2-byte, ssm states and saves 8-byte, z / y / norm 16-byte, A_log
// 4-byte, workspace 256-byte) and a table that is not host-visible or too small.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "kernels/k5_gdn_vk.h"
#include "launch/vk_launch.h"

namespace cere::vkk5 {

// The most rows of one chain_core call, all its chains together: a verify step's rows.
constexpr int64_t kChainMaxRows = 16;
static_assert(kChainMaxRows == kMaxRows, "one chain bound (the *_rows pieces')");
// Table bytes one chain_core call takes at table_off: the conv chain table, then the decode chain table.
constexpr uint64_t kChainCoreTableBytes = 2 * uint64_t(kChainTableBytes);

// The verify chains of one call (the CUDA header's Chains): chain c has rows[c] >= 1 consecutive rows of the sequence
// states[c] holds (distinct sequences: no conv or ssm address shared between chains), and saves[r] is the undo
// snapshot of row r (conv bf16 [10240, 3] and ssm bf16 [48, 128, 128], aliasing no state) for every row that is not its
// chain's last.
struct Chains {
    int64_t count = 0;  // 1..kChainMaxRows
    int64_t rows[kChainMaxRows] = {};
    LayerState states[kChainMaxRows] = {};
    LayerState saves[kChainMaxRows] = {};
};

// Every pipeline chain_core requests (no shape parameter: the shapes are the model's compile-time constants): the *_rows
// pieces of K5 and K1's gated rmsnorm of width kHeadDim.
std::vector<vk::PipelineSpec> pipeline_specs();

// Workspace of chain_core for `rows` rows: conv_out bf16 [rows, 10240] and o bf16 [rows, 6144], each 256-byte aligned.
// 0 when rows is outside 1..kChainMaxRows.
size_t chain_core_workspace_bytes(int64_t rows);

// See above. w: conv1d, A_log, dt_bias and norm are read (the projection weights are not).
cere_status chain_core(vk::Context &ctx, const LayerWeights &w, const Chains &chains, uint64_t mixed, uint64_t z, uint64_t b,
                       uint64_t a, int64_t rows, uint64_t y, uint64_t workspace, size_t workspace_bytes, const vk::Buffer &table,
                       uint64_t table_off);

}  // namespace cere::vkk5
