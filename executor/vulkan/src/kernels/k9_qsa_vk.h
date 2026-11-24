// VulkanAmd P2: K9 QSA sparse attention on Vulkan (shaders/k9_prep.comp, k9_logits.comp, k9_topk.comp, k9_attn.comp,
// k9_combine.comp; helpers in k9_common.comp).
//
// Contract: executor/cuda/src/kernels/k9_qsa.h (cere::k9), re-proven per device (vulkan-amd design note
// "Contracts"). The A4 shape: index q 4 heads x 128 dims, one index KV head, compress ratio 4, top-512 blocks, rows of
// 2,051 logical positions; sparse GQA of 24 query heads x 256 dims over 2 KV heads, scale 1/16. Device tensors are
// buffer device addresses (16-byte aligned where noted; every tensor inside one <= 4 GiB buffer); `positions`,
// `slots` and device positions are int64; the fault word (Bounds::fault, 0 = none) is a uint32 the device sets to 1 when
// it clamps a device position (k9_qsa.h Bounds).
//
// Arithmetic (see the shaders' headers):
//  - prep (indexer_qk, qsa_compress_store*): the CUDA kernels' rounding sequence with a correctly rounded sqrt and
//    division in the norm (CUDA: rsqrtf), so the result is the host mirror's IEEE value;
//  - logits: FP32 FMA chains (exact BF16 products), relu, head sum in order, correctly rounded division by
//    sqrt(128); one arithmetic for every row tiling (impl 1 = 8 rows per workgroup, impl 2 = 1 row; impl 0 picks);
//  - selection: integer radix select + ordered compaction (exact; one kernel: impl 0, 1 run it, impl 2 (CUDA's
//    cluster kernel) is CERE_ERR_UNSUPPORTED as on HIP);
//  - sparse GQA: fixed 64-entry tiles of the row's entry list, FP32; calls of <= 16 rows (by cls) write per-tile
//    partials + a fixed-order combine (decode / verify / extend), longer calls one workgroup per (row, KV head) with an
//    online softmax over the tiles (prefill). Both are per-row by construction: a row's bits never depend on the call's
//    other rows, so a verify chain's row equals its one-row call and any partition of a call gives the same bytes. The
//    two paths are different arithmetic (as CUDA/HIP's split and P4 paths).
// Contract differences from CUDA/HIP (per-device contract; bit-identity is a non-goal): logits and scores are FP32 FMA
// chains, not BF16 WMMA; the prefill path keeps p and V in FP32 (HIP's P4 rounds p to BF16); use_p4 does not change
// the path (HIP: every > 16-row call runs P4 too). As HIP (amd-amd step 16), the FP8 prefill with scratch decodes the
// call's positions 0 .. max index once into BF16 K / V copies in the scratch (k9_kvcopy.comp) and reads positions
// below the copy's capacity from there: the same exact values, so the same bits with or without scratch.
//
// Every function records into the context's open command buffer (Context::begin .. submit_wait), with barriers
// between its own dependent dispatches; the caller orders its later reads (Context::barrier). Arguments outside the
// contract: CERE_ERR_ARGUMENT before anything is recorded.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk9 {

constexpr int kIndexHeads = 4;
constexpr int kIndexHeadDim = 128;
constexpr int kQkWidth = (kIndexHeads + 1) * kIndexHeadDim;  // 640
constexpr int kRotaryDim = 64;
constexpr int kRatio = 4;
constexpr int kTokenTopk = 2048;
constexpr int kBlockTopk = kTokenTopk / kRatio;  // 512
constexpr int kRowWidth = kTokenTopk + kRatio - 1;  // 2,051
constexpr int kQHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kGroup = kQHeads / kKvHeads;
constexpr int kQWidth = kQHeads * kHeadDim;
constexpr int kKvWidth = kKvHeads * kHeadDim;
constexpr float kSoftmaxScale = 0.0625f;
constexpr float kScoreDivisor = 11.313708498984761f;
constexpr size_t kLogitsTileBytes = size_t{128} << 20;
constexpr int kSplitMaxRows = 16;
// rows_heads (qsa_attention_heads) picks the P.V thread layout of the attention kernels: 3, 6 or 12 query heads per
// thread (4 / 2 / 1 dims), 0 = 12; every layout writes the same bytes. The default is the measured fastest (3).
constexpr int kRowsHeads = 3;
constexpr int kTiles = (kRowWidth + 63) / 64;  // 33 entry tiles of 64
constexpr int64_t kMinP4Rows = 17;
constexpr uint32_t kBoundFault = 1;

struct Bounds {
    int64_t rope_first;
    int64_t rope_end;
    int64_t context;
    uint64_t fault;  // device address of a uint32 fault word; 0 = clamp only
};

// Options: the required subgroup size of the attention / selection pipelines (the prep pipelines always require 32), and
// (P4-A) the path of qsa_attention's > 16-row calls: kFp32 the P2 rows kernel (unchanged), kCoopmat
// shaders/p4att_sparse.comp: one workgroup per (row, KV head), a 16-row FP16 x FP16 -> FP32 coopmat tile (the 12 heads)
// over fixed 32-entry tiles of the row's entry list, operands carried as shaders/p4att_common.comp states (per-row /
// per-entry power-of-two scaling, p rounded to FP16), then the rows kernel built with FALLBACK 1 recomputes exactly the
// rows it flagged (a non-finite q / K / V value, an FP8 NaN byte, a V row maximum >= 2^16 among the row's valid
// entries): their bits are the FP32 path's. Per-row by construction (any partition / cls gives the same bytes),
// deterministic; numerics change against the FP32 path. Needs VK_KHR_cooperative_matrix (CERE_ERR_UNSUPPORTED
// without) and scratch for the flags and the block routes (coopmat_scratch_bytes(rows), at the start of `scratch`); rows_heads and the FP8
// decoded copy do not apply (the coopmat kernel reads the E4M3 cache directly). Calls of <= 16 rows (by cls) are the
// split path whatever the option, bit for bit.
// kCoopmat also routes 16-row blocks whose index rows are exactly the positions 0 .. p + u (the rows of a prompt below
// 2,048 tokens: every block selected) through the dense kernel (shaders/p4att_dense.comp, K/V shared by 16 rows): the
// sparse kernel runs the same per-row arithmetic, so the bytes do not depend on the routing (dense_blocks false: every
// row on the sparse kernel; the ktest compares the two).
enum class Prefill : uint32_t { kFp32 = 0, kCoopmat = 1 };
struct Options {
    uint32_t subgroup = 64;
    Prefill prefill = Prefill::kFp32;
    bool dense_blocks = true;
};
size_t coopmat_scratch_bytes(int64_t rows);

// Every pipeline the functions below can request with `opt`: the prep kernels (modes 0..5, and 1 / 2 "at"), the indexer
// logits (R 1 and 8), top-k, the split attention's combine, and per KV format asked (bf16_kv: k / v tensors, fp8_kv: the FP8
// cache; fp8 also the kvcopy decode kernels) the split (class <= 16) and rows (> 16) attention kernels for each
// `rows_heads` value (0 = 12, 3, 6, 12: the PV head tile; the default is kRowsHeads = 3). Model deployment: Options{64},
// both formats, rows_heads {kRowsHeads}. Refused options give an empty list.
std::vector<vk::PipelineSpec> pipeline_specs(const Options &opt, bool bf16_kv, bool fp8_kv, const std::vector<int> &rows_heads);

int64_t logits_tile_rows(int64_t rows, int64_t ctx);
size_t select_scratch_bytes(int64_t rows, int64_t ctx);
size_t attention_scratch_bytes(int64_t rows);
size_t scratch_bytes(int64_t rows, int64_t ctx);
struct Scratch {
    uint64_t q_idx, indices, work;
    size_t work_bytes;
};
cere_status carve_scratch(uint64_t base, size_t bytes, int64_t rows, int64_t ctx, Scratch *out);

cere_status indexer_qk(vk::Context &ctx, uint64_t qk, int64_t qk_stride, uint64_t q_norm_weight, float eps,
                       uint64_t cos_sin, const Bounds &bounds, uint64_t positions, int64_t pos0, int64_t tokens,
                       uint64_t q_idx);
cere_status qsa_compress_store(vk::Context &ctx, uint64_t k_idx, int64_t k_stride, uint64_t k_norm_weight, float eps,
                               uint64_t cos_sin, const Bounds &bounds, uint64_t positions, uint64_t slots,
                               int64_t pos0, int64_t tokens, uint64_t ring, uint64_t keys);
cere_status qsa_compress_store_step(vk::Context &ctx, uint64_t k_idx, uint64_t k_norm_weight, float eps,
                                    uint64_t cos_sin, const Bounds &bounds, uint64_t slots, uint64_t position,
                                    uint64_t ring, uint64_t keys);
cere_status qsa_compress_store_at(vk::Context &ctx, uint64_t k_idx, int64_t k_stride, uint64_t k_norm_weight,
                                  float eps, uint64_t cos_sin, const Bounds &bounds, uint64_t slots,
                                  uint64_t position, int64_t tokens, uint64_t ring, uint64_t keys);
cere_status qsa_key_save_step(vk::Context &ctx, uint64_t keys, uint64_t slots, const Bounds &bounds,
                              uint64_t position, int64_t j, int64_t n, uint64_t saved);
// One full-attention layer's QSA verify work over the verify graph's rows in one dispatch (k9_qsa.h qsa_chains_step):
// workgroup c runs chain c's rows (chain_first[c] .. + chain_rows[c], chains contiguous from row 0) in row order; per
// row r, when undo_ring[r] is set, the copy of ring[r] ([4, 128] BF16) to undo_ring[r] and, when undo_key[r] is set,
// qsa_key_save_step(keys[r], slots[r], positions + r, j[r], n[r], undo_key[r]); then, when k_idx[r] is set,
// qsa_compress_store_step(k_idx[r], ..., slots[r], positions + r, ring[r], keys[r]). Row r's position is clamped to
// 0 .. context[r] - 1, its rotary row into bounds (bounds.context unused). The arrays are device addresses (0 = null);
// they are written at record time into `table` (a host-visible buffer of >= kChainsTableBytes that the shader reads by
// address): one table per call in flight, not reused before the submission completes.
constexpr int kChainsStepRows = 16;
constexpr size_t kChainsTableBytes = 1152;
struct ChainsStep {
    uint64_t k_idx[kChainsStepRows];
    uint64_t slots[kChainsStepRows];
    uint64_t ring[kChainsStepRows];
    uint64_t keys[kChainsStepRows];
    uint64_t undo_ring[kChainsStepRows];
    uint64_t undo_key[kChainsStepRows];
    int64_t context[kChainsStepRows];
    int32_t j[kChainsStepRows];
    int32_t n[kChainsStepRows];
    int32_t chain_first[kChainsStepRows];
    int32_t chain_rows[kChainsStepRows];
};
static_assert(sizeof(ChainsStep) == kChainsTableBytes, "the table layout of k9_prep.comp MODE 5");
cere_status qsa_chains_step(vk::Context &ctx, const ChainsStep &rows, int chains, uint64_t k_norm_weight, float eps,
                            uint64_t cos_sin, const Bounds &bounds, uint64_t positions, vk::Buffer &table);

// REFUSED (CERE_ERR_UNSUPPORTED): the mRoPE variants serve image sessions (vision is a non-goal until after P3).
cere_status indexer_qk_mrope(vk::Context &ctx, uint64_t qk, int64_t qk_stride, uint64_t q_norm_weight, float eps,
                             uint64_t cos_sin, const Bounds &bounds, uint64_t positions3, int64_t p3_stride,
                             int64_t p3_lead, int64_t tokens, uint64_t q_idx);
cere_status qsa_compress_store_mrope(vk::Context &ctx, uint64_t k_idx, int64_t k_stride, uint64_t k_norm_weight,
                                     float eps, uint64_t cos_sin, const Bounds &bounds, uint64_t positions3,
                                     int64_t p3_stride, int64_t p3_lead, uint64_t slots, int64_t pos0, int64_t tokens,
                                     uint64_t ring, uint64_t keys);

cere_status qsa_logits(vk::Context &ctx, uint64_t q_idx, uint64_t keys, uint64_t slots, uint64_t positions_logical,
                       int64_t rows, int64_t seq_len, uint64_t logits, int64_t logits_ld, int impl,
                       const Options &opt = {});
cere_status qsa_select(vk::Context &ctx, uint64_t q_idx, uint64_t keys, uint64_t slots, uint64_t positions_logical,
                       int64_t rows, int64_t seq_len, uint64_t scratch, size_t scratch_bytes, uint64_t out_indices,
                       const Options &opt = {});
cere_status qsa_select_step(vk::Context &ctx, uint64_t q_idx, uint64_t keys, uint64_t slots, const Bounds &bounds,
                            uint64_t position, int64_t ctx_len, uint64_t scratch, size_t scratch_bytes,
                            uint64_t out_indices, const Options &opt = {});
cere_status qsa_topk(vk::Context &ctx, uint64_t logits, int64_t ld, uint64_t positions_logical, int64_t rows,
                     int64_t seq_len, int impl, uint64_t out_indices, const Options &opt = {});

cere_status qsa_attention(vk::Context &ctx, uint64_t q, int64_t q_stride, uint64_t indices, int64_t rows, uint64_t k,
                          int64_t k_stride, uint64_t v, int64_t v_stride, uint64_t k_cache, uint64_t v_cache,
                          uint64_t slots, float scale, uint64_t scratch, size_t scratch_bytes, uint64_t out,
                          int64_t out_stride, bool use_p4, int64_t cls = 0, const Options &opt = {});
cere_status qsa_attention_heads(vk::Context &ctx, uint64_t q, int64_t q_stride, uint64_t indices, int64_t rows,
                                uint64_t k, int64_t k_stride, uint64_t v, int64_t v_stride, uint64_t k_cache,
                                uint64_t v_cache, uint64_t slots, float scale, uint64_t scratch,
                                size_t scratch_bytes, uint64_t out, int64_t out_stride, bool use_p4, int64_t cls,
                                int rows_heads, const Options &opt = {});

}  // namespace cere::vkk9
