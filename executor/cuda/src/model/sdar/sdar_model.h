// SelfTune ST4 S3c (selftune design note ST4): weights, plan and forward of family sdar (DICE-8B: SDAR block
// diffusion over Qwen3-8B dense layers). The shapes are the library's compiled constants (cere_model_constants.h,
// generated from executor/models/dice-8b.json), which api.cpp checked the descriptor's config against.
//
// Layer forward (DICE modeling_sdar.py at the S2 pin, BF16 tensors between ops):
//   h = rmsnorm(x, input_layernorm); q, k, v = h · W^T; q, k = rmsnorm per head (q_norm, k_norm); NeoX RoPE at
//   absolute positions; K/V stored to the session's cache rows; block-causal GQA attention; x = x + attn · W_o^T;
//   h = rmsnorm(x, post_attention_layernorm); x = x + (silu(h · W_gate^T) * (h · W_up^T)) · W_down^T.
// Then (denoise only) rmsnorm(x, model.norm), lm_head in FP32, the per-row BF16 top token and tie count, and the
// unmask choice (kernels/sdar/sdar_kernels.h).
#pragma once

#include "cere_executor.h"
#include "kernels/sdar/sdar_gemm.h"
#include "kernels/sdar/sdar_kernels.h"
#include "model/sdar/schedule.h"

#include <cstdint>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace cere::sdar {

// Rows of one forward pass; a longer prefill runs in passes of this many rows (a multiple of kBlock, so every pass
// starts on a block boundary).
constexpr uint64_t kMaxRows = 2048;

// One checkpoint tensor of the plan (host/device.h TensorPlan): its name, shape and bytes, and the byte offset the
// device placed it at in the model's weight arena.
struct TensorSlot {
    std::string name;
    uint32_t dtype;
    uint32_t ndim;
    uint64_t shape[CERE_TENSOR_MAX_DIMS];
    uint64_t bytes;
    uint64_t offset;
};

// Weight-arena layout of a plan's device tensors (256-byte aligned, in plan order) except that each layer's
// mlp.gate_proj and mlp.up_proj are placed back to back (gate first), so the two form one [2 * intermediate, hidden]
// matrix and one GEMM computes the gate_up rows silu_mul reads. Returns the arena size.
uint64_t layout_tensors(std::vector<TensorSlot> *slots);

struct LayerWeights {
    const bf16 *input_norm = nullptr;
    const bf16 *post_norm = nullptr;
    const bf16 *q_norm = nullptr;
    const bf16 *k_norm = nullptr;
    const bf16 *q = nullptr;
    const bf16 *k = nullptr;
    const bf16 *v = nullptr;
    const bf16 *o = nullptr;
    const bf16 *gate_up = nullptr;
    const bf16 *down = nullptr;
};

struct Weights {
    const bf16 *embed = nullptr;
    const bf16 *norm = nullptr;
    const bf16 *lm_head = nullptr;
    std::vector<LayerWeights> layers;
};

// Binds every tensor the forward reads by name, checking its dtype (BF16) and shape against the compiled constants;
// CERE_ERR_ARGUMENT names nothing, so a missing, misshapen or extra tensor is reported on stderr by name.
cere_status bind_weights(const std::vector<TensorSlot> &slots, const uint8_t *arena, Weights *out);

// Device workspace of one model: the activations of kMaxRows rows and the 4-row logits of a denoise.
struct Workspace {
    bf16 *x = nullptr;       // [kMaxRows, hidden] residual stream
    bf16 *h = nullptr;       // [kMaxRows, hidden] normed
    bf16 *q = nullptr;       // [kMaxRows, heads * head_dim]
    bf16 *k = nullptr;       // [kMaxRows, kv_heads * head_dim]
    bf16 *v = nullptr;       // [kMaxRows, kv_heads * head_dim]
    bf16 *attn = nullptr;    // [kMaxRows, heads * head_dim]
    bf16 *tmp = nullptr;     // [kMaxRows, hidden]
    bf16 *gate_up = nullptr; // [kMaxRows, 2 * intermediate]
    bf16 *act = nullptr;     // [kMaxRows, intermediate]
    float *logits = nullptr; // [kBlock, vocab]
    uint32_t *ids = nullptr; // [kMaxRows]
    TopTie *top = nullptr;   // [kBlock]
    uint8_t *masked = nullptr; // [kBlock]
    int32_t *index = nullptr;  // [1]
};

// Device bytes of the workspace (one allocation) and of one session's KV cache of max_context_tokens.
uint64_t workspace_bytes();
uint64_t kv_bytes(uint64_t max_context_tokens);
// Carves the workspace out of `base` (workspace_bytes(), 256-byte aligned).
Workspace carve_workspace(uint8_t *base);

// The device buffer the changing GEMM entries reduce through (kChangingWorkspaceBytes), null under identical.
struct GemmWorkspace {
    void *ptr = nullptr;
    GemmWorkspace() = default;
    GemmWorkspace(const GemmWorkspace &) = delete;
    GemmWorkspace &operator=(const GemmWorkspace &) = delete;
    GemmWorkspace(GemmWorkspace &&other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    GemmWorkspace &operator=(GemmWorkspace &&other) noexcept;
    ~GemmWorkspace();
};

// The GEMM plans of the model's six shapes (host only: init on the device's cuBLASLt handle). `impl` is the
// CERE_SLOT_SDAR_GEMM choice (SdarOptIn, O320): identical runs sdar_gemm.h's pinned bit-identical picks; changing
// also allocates the workspace and gives every GEMM changing_pins() (Gemm::set_changing), so a create on another
// cuBLASLt version or device, or with a config cuBLASLt refuses, fails UNSUPPORTED naming sdar_gemm.changing. Any
// other impl is CERE_ERR_ARGUMENT.
struct Gemms {
    Gemm qo;       // n = k = hidden (q_proj, o_proj)
    Gemm kv;       // n = kv_heads * head_dim, k = hidden
    Gemm gate_up;  // n = 2 * intermediate, k = hidden
    Gemm down;     // n = hidden, k = intermediate
    Gemm lm_head;  // n = vocab, k = hidden, FP32 out, m <= kBlock
    GemmWorkspace workspace;
    cere_status init(cublasLtHandle_t handle, uint32_t impl = CERE_LEVER_SDAR_GEMM_IDENTICAL);
};

// Device bytes Gemms::init allocates for `impl` (beyond workspace_bytes()).
uint64_t gemm_workspace_bytes(uint32_t impl);

// The forward passes of one session (schedule.h Forward) over its KV cache: k_cache / v_cache hold one
// [kv_rows, kv_heads * head_dim] block per layer, layer after layer.
class CudaForward final : public Forward {
public:
    CudaForward(const Weights &weights, Gemms &gemms, const Workspace &ws, bf16 *k_cache, bf16 *v_cache,
                uint64_t kv_rows, cudaStream_t stream)
        : w_(weights), gemms_(gemms), ws_(ws), k_cache_(k_cache), v_cache_(v_cache), kv_rows_(kv_rows),
          stream_(stream) {}

    cere_status prefill(const uint32_t *tokens, uint64_t n, uint64_t pos) override;
    cere_status denoise(const uint32_t *block, const bool *masked, uint64_t base, RowPick *picks,
                        int32_t *index) override;
    cere_status commit(const uint32_t *block, uint64_t base) override;

private:
    // The layer stack over n <= kMaxRows rows at positions [pos, pos + n) (pos % kBlock == 0), storing their K/V.
    cere_status layers(const uint32_t *tokens, uint64_t n, uint64_t pos);

    const Weights &w_;
    Gemms &gemms_;
    const Workspace &ws_;
    bf16 *k_cache_;
    bf16 *v_cache_;
    uint64_t kv_rows_;
    cudaStream_t stream_;
};

}  // namespace cere::sdar
