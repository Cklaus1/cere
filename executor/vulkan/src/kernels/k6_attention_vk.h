// VulkanAmd P2: K6 dense causal GQA on Vulkan (shaders/k6_decode.comp, k6_combine.comp, k6_prefill.comp, k6_gate.comp).
//
// Contract: executor/cuda/src/kernels/k6_attention.h (cere::k6::attention / dense_attention / dense_attention_rows),
// re-proven per device (vulkan-amd design note "Contracts"): out[t, h] = softmax_j(q[t, h] . K[j] * scale) V[j] over
// keys j = 0 .. pos0 + t, KV head h / 12, 24 query heads x 256 dims, 2 KV heads; FP32 scores / softmax / accumulation,
// BF16 output (RNE, NaN 0x7FFF). K/V: BF16 rows (k/v with row strides in elements, key j at row j) or the FP8 E4M3 cache
// (k_cache/v_cache [rows, 512] bytes, scale 1.0, key j at row slots[j] (int64), row j when slots is 0); exactly one pair
// non-zero. Addresses are buffer device addresses (16-byte aligned); every tensor lies in one <= 4 GiB buffer.
//
// Paths (chosen by sel = cls ? cls : tokens, as k6::attention's class pin): sel <= kSplitMaxTokens runs the split-key
// kernel (KS keys per split, a function of the row's own key count) + the split combine; sel > kSplitMaxTokens runs
// the FP32 prefill tile kernel. Both are row-partition invariant by construction (a row's bits depend on its position,
// its q row and the keys only; see the shaders), so the class pin holds for any cls and a verify chain's row t equals
// the one-token call at pos0 + t. The two paths are different arithmetic: a row's bits depend on the path its call's
// class selects (as on CUDA/HIP).
//
// P4-A (cooperative-matrix prefill, Options::prefill = Prefill::kCoopmat): sel > kSplitMaxTokens runs
// shaders/p4att_dense.comp (FP16 x FP16 -> FP32 16x16x16 coopmat, per-row / per-key power-of-two operand scaling, p
// rounded to FP16; the carrying rules and their bounds are in shaders/p4att_common.comp), then the FP32 prefill kernel
// built with FALLBACK 1, which recomputes exactly the rows the coopmat kernel flagged (a non-finite q / K / V value, an
// FP8 NaN byte, a V row maximum >= 2^16 at or before the row's position): a flagged row's bits are the FP32 path's.
// Row-partition invariant as the FP32 path (a row's bits depend on its position, q row and keys only), deterministic.
// Numerics change against the FP32 path (different summation order, p carried in 11 bits); the decode / split side
// (sel <= 16) is untouched by the option, bit for bit. Needs VK_KHR_cooperative_matrix (CERE_ERR_UNSUPPORTED without) and
// workspace for the per-token flags (workspace_bytes with the same Options), also for attention().
//
// The functions record into the context's open command buffer (Context::begin .. submit_wait), with a barrier between
// their own dispatches; the caller orders its later reads of `out` (Context::barrier). CERE_ERR_ARGUMENT outside the
// contract, before anything is recorded.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk6 {

constexpr int kQHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kGroup = kQHeads / kKvHeads;
constexpr int kQWidth = kQHeads * kHeadDim;
constexpr int kKvWidth = kKvHeads * kHeadDim;
constexpr int64_t kMaxContext = 2048;
constexpr int64_t kSplitMaxTokens = 16;
constexpr float kSoftmaxScale = 0.0625f;  // head_dim^-0.5

// Build-time variant of the split kernel (keys per split; build.d/k6_attention.sh compiles 64 and 128). The default
// is the one the timing chose; a different KS is a different summation order (a different, equally valid, device
// contract instance), so one process uses one KS.
enum class Prefill : uint32_t { kFp32 = 0, kCoopmat = 1 };
struct Options {
    uint32_t keys_per_split = 64;
    uint32_t subgroup = 64;  // required subgroup size of every K6 SIMT pipeline (32 or 64); the coopmat kernel uses 32
    Prefill prefill = Prefill::kFp32;  // the > 16-row path (P4-A); kFp32 is the P2 kernel, unchanged
};

// Workspace bytes for calls of up to max_tokens rows (split partials for min(max_tokens, 16) rows, plus the BF16
// pre-gate output of dense_attention's prefill path).
// Every pipeline attention / dense_attention / dense_attention_rows can request with `opt` (keys_per_split, subgroup):
// the split-decode kernel and the prefill kernel of each KV format asked (BF16 KV: attention with k / v; FP8 KV: the
// cache), the combine kernel and the dense gate kernel. Model deployment: Options{64 (or the tuned 128), 64}, both KV
// formats. Refused options (ks_ok false) give an empty list.
std::vector<vk::PipelineSpec> pipeline_specs(const Options &opt, bool bf16_kv, bool fp8_kv);

size_t workspace_bytes(int64_t max_tokens, const Options &opt = {});

cere_status attention(vk::Context &ctx, uint64_t q, int64_t tokens, int64_t pos0, uint64_t k, int64_t k_stride,
                      uint64_t v, int64_t v_stride, uint64_t k_cache, uint64_t v_cache, uint64_t slots, float scale,
                      uint64_t workspace, size_t workspace_bytes, uint64_t out, int64_t cls = 0,
                      const Options &opt = {});

// attention with the scale kSoftmaxScale, then the output gate (K1 sigmoid_mul's expression):
// out [tokens, 6144] (row stride out_stride) = bf16(attn * sigmoid(gate)), gate [tokens, 6144] contiguous. K/V source:
// kv_from_cache false reads the BF16 rows (k stride 512, v stride v_stride), true the FP8 cache through slots.
cere_status dense_attention(vk::Context &ctx, uint64_t q, uint64_t k, uint64_t v, int64_t v_stride, uint64_t gate,
                            int64_t tokens, int64_t prefix_len, bool kv_from_cache, uint64_t k_cache, uint64_t v_cache,
                            uint64_t slots, uint64_t workspace, size_t workspace_bytes, uint64_t out,
                            int64_t out_stride, int64_t cls = 0, const Options &opt = {});

// A verify chain: rows 1 .. 16 at positions pos0 .. pos0 + rows - 1 over the FP8 cache; row t is bit-equal to
// dense_attention(q + t row, ..., gate + t row, tokens 1, prefix_len pos0 + t, kv_from_cache true). One split dispatch
// and one combine for all rows (the split kernel is per-row by construction).
cere_status dense_attention_rows(vk::Context &ctx, uint64_t q, uint64_t gate, int64_t rows, int64_t pos0,
                                 uint64_t k_cache, uint64_t v_cache, uint64_t slots, uint64_t workspace,
                                 size_t workspace_bytes, uint64_t out, int64_t out_stride, const Options &opt = {});

// P4-A device facts the coopmat path's error bounds depend on (shaders/p4att_probe.comp, one submit): out[0] the FP16
// bits of f16(2^-20) converted in a shader (0x0010: subnormals kept), out[1] / out[2] the float bits of a coopmat
// product of 16 FP16 2^-20 (subnormal) / 2^-14 (normal) operands with 1 (2^-16 / 2^-10 when kept), out[3] the
// accumulator length. CERE_ERR_UNSUPPORTED without cooperative matrices.
cere_status coopmat_probe(vk::Context &ctx, uint32_t out[4]);

// SPIR-V names (build.d/k6_attention.sh compiles the same set).
std::string decode_spv(bool fp8, uint32_t ks);
std::string combine_spv(uint32_t ks);

}  // namespace cere::vkk6
