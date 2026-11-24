// VulkanAmd P3 (T3): see k6_prologue_vk.h. Order of operations as executor/cuda/src/kernels/k6_attention.cu:1203-1216.
#include "kernels/k6_prologue_vk.h"

#include "kernels/k1_primitives_vk.h"

namespace cere::vkk6 {
namespace {

using namespace prologue_shape;

uint64_t align256(uint64_t n) { return (n + 255) & ~uint64_t(255); }
bool al(uint64_t a, uint64_t n) { return a % n == 0; }
bool stride_ok(int64_t s, int64_t width) { return s >= width && s % 8 == 0 && s < (int64_t(1) << 31); }

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs() {
    // prologue = vkk1::qk_norm_rope_gate (head_dim kHeadDim) + vkk1::kv_store_fp8: those two entries of K1's list.
    std::vector<vk::PipelineSpec> s;
    for (const vk::PipelineSpec &p : vkk1::pipeline_specs({}, {kHeadDim}))
        if (p.spv == "k1_qk_norm_rope_gate.spv" || p.spv == "k1_kv_store.spv") s.push_back(p);
    return s;
}

size_t prologue_bytes(int64_t max_tokens) {
    if (max_tokens <= 0) return 0;
    const uint64_t t = uint64_t(max_tokens);
    return size_t(align256(t * kQWidth * 2) + align256(t * kKvWidth * 2) + align256(t * kQWidth * 2));
}

cere_status prologue(vk::Context &ctx, uint64_t q_gate, int64_t q_gate_stride, uint64_t k, int64_t k_stride, uint64_t v,
                     int64_t v_stride, uint64_t q_norm_weight, uint64_t k_norm_weight, float eps, uint64_t cos_sin,
                     int64_t context, uint64_t positions, uint64_t positions3, uint64_t slots, int64_t tokens,
                     int64_t prefix_len, uint64_t k_cache, uint64_t v_cache, uint64_t q_out, uint64_t k_out,
                     uint64_t gate_out) {
    // The CUDA prologue's argument checks (k6_attention.cu:1207-1209), then the two K1 calls' own rules checked here
    // as well, so a refused call records nothing (the second call cannot fail after the first was recorded).
    if (!q_gate || !k || !v || !q_norm_weight || !k_norm_weight || !cos_sin || !k_cache || !v_cache || !q_out || !k_out ||
        !gate_out || tokens < 0 || prefix_len < 0 || context <= 0 || prefix_len > context - tokens)
        return CERE_ERR_ARGUMENT;
    if (positions3) return CERE_ERR_UNSUPPORTED;  // mRoPE: not on Vulkan (k1_primitives_vk.h)
    if (tokens == 0) return CERE_OK;
    if (tokens > kMaxTokens || !stride_ok(q_gate_stride, int64_t(kQHeads) * 2 * kHeadDim) || !stride_ok(k_stride, kKvWidth) ||
        !stride_ok(v_stride, kKvWidth) || !al(q_gate, 16) || !al(k, 16) || !al(v, 16) || !al(q_norm_weight, 16) ||
        !al(k_norm_weight, 16) || !al(cos_sin, 4) || !al(q_out, 16) || !al(k_out, 16) || !al(gate_out, 16) || !al(k_cache, 8) ||
        !al(v_cache, 8) || (positions && !al(positions, 8)) || (slots && !al(slots, 8)))
        return CERE_ERR_ARGUMENT;
    cere_status st = vkk1::qk_norm_rope_gate(ctx, q_gate, q_gate_stride, k, k_stride, q_norm_weight, k_norm_weight, cos_sin,
                                             positions, 0, prefix_len, tokens, kQHeads, kKvHeads, kHeadDim, kRotaryDim, eps,
                                             q_out, k_out, gate_out);
    if (st != CERE_OK) return st;
    ctx.barrier();  // the store reads k_out
    return vkk1::kv_store_fp8(ctx, k_out, kKvWidth, v, v_stride, tokens, kKvWidth, k_cache, v_cache,
                              slots ? slots + uint64_t(prefix_len) * 8u : 0, prefix_len);
}

}  // namespace cere::vkk6
