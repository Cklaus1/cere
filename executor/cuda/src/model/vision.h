// The A4 model's vision tower on the device (m213 design note D10): one encoder pass per image, SGLang
// `Qwen3VLMoeVisionModel.forward` (sglang/srt/models/qwen3_vl.py:882-992) composed from the KV kernels
// (kernels/kv_vision.h) and K2 GEMMs with F32 output, so every linear rounds to BF16 once as an addmm:
//   patch_embed (Conv3d with stride = kernel as a [P, 1,536] x [1,536, 1,152] GEMM) + bilinear pos_embed;
//   27 x { x += proj(attn(rope2d(qkv(norm1(x))))); x += fc2(gelu_tanh(fc1(norm2(x)))) };
//   merger: fc2(gelu_erf(fc1(norm(x).view(-1, 4,608)))) -> [P / 4, 2,560].
// No CUDA include beyond the runtime types; no allocation (the caller passes the scratch).
//
// math_reference: sglang
#pragma once

#include "cere_executor.h"
#include "cere_model_constants.h"
#include "kernels/k2_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace cere::model::vision {

using bf16 = __nv_bfloat16;

constexpr uint32_t kDepth = gen::kVisionDepth;
constexpr uint32_t kHidden = gen::kVisionHiddenSize;
constexpr uint32_t kQkvWidth = 3 * kHidden;
constexpr uint32_t kInter = gen::kVisionIntermediateSize;
// in_channels 3 x temporal_patch 2 x 16 x 16
constexpr uint32_t kPatchRow =
    gen::kVisionInChannels * gen::kVisionTemporalPatchSize * gen::kVisionPatchSize * gen::kVisionPatchSize;
constexpr uint32_t kMergeUnit = gen::kVisionSpatialMergeSize * gen::kVisionSpatialMergeSize;  // spatial_merge_size^2
constexpr uint32_t kMergeRow = kMergeUnit * kHidden;  // 4,608
constexpr uint32_t kOutHidden = gen::kVisionOutHiddenSize;  // out_hidden_size (= the text hidden size)
constexpr uint32_t kPosSide = 48;      // sqrt(num_position_embeddings 2,304)
static_assert(kPosSide * kPosSide == gen::kVisionNumPositionEmbeddings, "a square position-embedding grid");
constexpr float kEps = 1e-6f;          // nn.LayerNorm eps (qwen3_vl.py:196)
// Patches of one image at most: the image buffers hold 4,096 merged rows (model/layout.h kMaxImageTokens).
constexpr uint32_t kMaxPatches = 4096 * kMergeUnit;

// The 333 `model.visual.*` tensors in a fixed order: patch_embed.proj.{weight, bias}, pos_embed.weight, then per block
// norm1.{weight, bias}, attn.qkv.{weight, bias}, attn.proj.{weight, bias}, norm2.{weight, bias},
// mlp.linear_fc1.{weight, bias}, mlp.linear_fc2.{weight, bias}, then merger.norm.{weight, bias},
// merger.linear_fc1.{weight, bias}, merger.linear_fc2.{weight, bias}. All BF16, DEVICE.
constexpr uint32_t kBlockTensors = 12;
constexpr uint32_t kTensorCount = 3 + kDepth * kBlockTensors + 6;
enum Block : uint32_t { kNorm1W, kNorm1B, kQkvW, kQkvB, kProjW, kProjB, kNorm2W, kNorm2B, kFc1W, kFc1B, kFc2W, kFc2B };
enum Merger : uint32_t { kMNormW, kMNormB, kMFc1W, kMFc1B, kMFc2W, kMFc2B };

// Checkpoint name and shape of tensor i (i < kTensorCount); `shape` receives up to 5 dims, returns ndim.
std::string tensor_name(uint32_t i);
uint32_t tensor_shape(uint32_t i, uint64_t shape[5]);

struct Weights {
    const bf16 *tensor[kTensorCount] = {};  // in the order above; null when the model has no vision tower
    bool bound() const { return tensor[0] != nullptr; }
    const bf16 *patch_w() const { return tensor[0]; }  // [1,152, 3, 2, 16, 16] = [1,152, 1,536]
    const bf16 *patch_b() const { return tensor[1]; }
    const bf16 *pos_embed() const { return tensor[2]; }  // [2,304, 1,152]
    const bf16 *block(uint32_t b, Block which) const { return tensor[3 + b * kBlockTensors + which]; }
    const bf16 *merger(Merger which) const { return tensor[3 + kDepth * kBlockTensors + which]; }
};

// The encoder's seven K2 plans (F32 output), max_m = patches (5 tower plans) or patches / 4 (2 merger plans).
struct Plans {
    kernels::GemmPlan patch;   // n 1,152, k 1,536
    kernels::GemmPlan qkv;     // n 3,456, k 1,152
    kernels::GemmPlan proj;    // n 1,152, k 1,152
    kernels::GemmPlan fc1;     // n 4,304, k 1,152
    kernels::GemmPlan fc2;     // n 1,152, k 4,304
    kernels::GemmPlan merge1;  // n 4,608, k 4,608
    kernels::GemmPlan merge2;  // n 2,560, k 4,608
};

// Creates every plan of `plans` for images of up to max_patches patches (a multiple of 4).
cere_status init_plans(Plans *plans, cublasLtHandle_t handle, uint32_t max_patches);

// Device scratch bytes of one encoder pass over up to `patches` patches (256-byte aligned regions).
uint64_t scratch_bytes(uint64_t patches);

// Taps (all nullable): sink receives device BF16 [rows, cols] readable in `stream` order during the call. Names:
// "patch_embed", "pos_embed" (patch_embed + pos_embed, the block 0 input); per block b (block = b):
// "blocks.norm1", "blocks.attn.qkv", "blocks.attn.core" (attention output before proj), "blocks.attn.residual"
// (x + proj), "blocks.norm2", "blocks.mlp.act" (gelu(fc1)), "blocks.out"; then (block = kDepth) "merger.norm"
// ([P / 4, 4,608] view), "merger.act", "merger.out" ([P / 4, 2,560]).
struct Taps {
    void *ctx = nullptr;
    cere_status (*sink)(void *ctx, const char *name, uint32_t block, const bf16 *values, uint64_t rows, uint64_t cols,
                        cudaStream_t stream) = nullptr;
};

// One image of grid (t, h, w) patches (h, w even, t * h * w <= the plans' max_patches): pixels BF16 [t * h * w,
// 1,536] in processor (merge-window) order -> out BF16 [t * h * w / 4, 2,560] (row stride 2,560). scratch:
// scratch_bytes(t * h * w), 256-byte aligned. CERE_ERR_ARGUMENT before any launch on a bad argument;
// CERE_ERR_INTERNAL on a failed launch. Enqueued on `stream`, never synchronizes.
cere_status encode(const Weights &w, const Plans &plans, const bf16 *pixels, uint32_t grid_t, uint32_t grid_h,
                   uint32_t grid_w, void *scratch, size_t scratch_size, bf16 *out, const Taps *taps,
                   cudaStream_t stream);

}  // namespace cere::model::vision
