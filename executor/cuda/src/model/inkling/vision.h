// Inkling 2 Mini's vision tower on the device (inkling design note 3.6, slice S7): transformers 5.24.0
// `Inkling2VisionModel.forward` (modeling_inkling2.py:1563-1603) over the images of one request in one pass, composed
// from the kernels of kernels/inkling/vision.h and K2 GEMMs with F32 output (every linear rounds to BF16 once):
//   patch embedding (Linear 1,200 -> 1,024, no bias) + bilinear 32 x 32 position table, one class token per image;
//   pre LayerNorm; 23 x { x += lambda_1 * out(attn(rope2d(q, k), v)(ln_1(x))); x += lambda_2 * fc2(gelu(fc1(ln_2(x)))) }
//   with attention bidirectional within each image (cu_seqlens); merger on the patch rows (class rows dropped):
//   proj_out(gelu(proj_in(x.view(-1, 4,096)))) -> [tokens, 2,304], one row per `<|unused_200054|>` placeholder.
// The caller (the text forward, S5/S6) splices the rows over the placeholder rows of the embedded prompt (the text
// model has no RoPE, so nothing else depends on the image). No allocation (the caller passes the scratch).
//
// math_reference: transformers
#pragma once

#include "cere_executor.h"
#include "kernels/inkling/vision.h"
#include "kernels/k2_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace cere::model::inkling_vision {

using bf16 = __nv_bfloat16;
namespace kv = cere::inkling::vision;

// Patches of one call at most (all images): 4,096 image tokens of 2 x 2 patches.
constexpr uint32_t kMaxPatches = 16384;
constexpr uint32_t kMaxImages = kv::kMaxImages;

// The 422 `model.visual.*` tensors (all BF16, DEVICE) in a fixed order: the 8 globals (Global), then per layer the 18
// of Layer. Checkpoint names: tensor_name(i).
enum Global : uint32_t {
    kPatchW,      // backbone.embeddings.patch_embedding.weight [1,024, 1,200]
    kPosTable,    // backbone.embeddings.position_embedding [1,024 positions, 1,024]
    kClass,       // backbone.embeddings.class_embedding [1,024]
    kClassPos,    // backbone.embeddings.cls_position_embedding [1,024]
    kPreW,        // backbone.ln_pre.weight
    kPreB,        // backbone.ln_pre.bias
    kProjIn,      // proj_in.weight [4,096, 4,096] (merger linear_1, no bias)
    kProjOut,     // proj_out.weight [2,304, 4,096] (merger linear_2, no bias)
    kGlobalCount
};
enum Layer : uint32_t {
    kLn1W, kLn1B, kQW, kQB, kKW, kKB, kVW, kVB, kOutW, kOutB, kLs1, kLn2W, kLn2B, kFc1W, kFc1B, kFc2W, kFc2B, kLs2,
    kLayerCount
};
constexpr uint32_t kTensorCount = kGlobalCount + kv::kLayers * kLayerCount;
static_assert(kTensorCount == 422, "the checkpoint's model.visual tensor count");

// Checkpoint name and shape of tensor i (i < kTensorCount); `shape` receives up to 2 dims, returns ndim.
std::string tensor_name(uint32_t i);
uint32_t tensor_shape(uint32_t i, uint64_t shape[2]);

struct Weights {
    const bf16 *tensor[kTensorCount] = {};  // in the order above
    bool bound() const;                     // every tensor set
    const bf16 *global(Global which) const { return tensor[which]; }
    const bf16 *layer(uint32_t l, Layer which) const { return tensor[kGlobalCount + l * kLayerCount + which]; }
};

// The encoder's K2 plans (F32 output) for up to max_patches patches (+ one class row per image).
struct Plans {
    kernels::GemmPlan patch;  // n 1,024, k 1,200
    kernels::GemmPlan attn;   // n 1,024, k 1,024 (q, k, v, out)
    kernels::GemmPlan fc1;    // n 4,096, k 1,024
    kernels::GemmPlan fc2;    // n 1,024, k 4,096
    kernels::GemmPlan proj_in;   // n 4,096, k 4,096 (m = tokens of one image)
    kernels::GemmPlan proj_out;  // n 2,304, k 4,096
};

// Creates every plan for calls of up to max_patches patches (a multiple of 4, <= kMaxPatches). Host only.
cere_status init_plans(Plans *plans, cublasLtHandle_t handle, uint32_t max_patches);

// Device scratch bytes of one call over `patches` patches of `images` images (256-byte aligned regions).
uint64_t scratch_bytes(uint64_t patches, uint32_t images);

// One image's grid in patches (t = 1): h = 2 * cell rows, w = 2 * cell columns (the tokenizer sidecar's grid_thw).
struct Grid {
    uint32_t h;
    uint32_t w;
};

// Fail-first mutants (kernel tests only; null in production): no_class runs the tower without the class tokens,
// swap_rope_axes rotates by (w, h) instead of (h, w).
struct Mutant {
    bool no_class = false;
    bool swap_rope_axes = false;
};

// Encodes `count` images (1..kMaxImages) in one pass: pixels BF16 [sum h * w, 1,200] (each image's rows in the
// processor's merge-block order, images concatenated) -> out BF16 [sum h * w / 4, 2,304] (row stride 2,304, images in
// order: image i's rows are its placeholders' embeddings). h, w even and >= 2; sum h * w <= plans' max_patches.
// scratch: scratch_bytes(sum h * w, count), 256-byte aligned. CERE_ERR_ARGUMENT before any launch on a bad argument;
// CERE_ERR_INTERNAL on a failed launch. Enqueued on `stream`, never synchronizes.
cere_status encode(const Weights &w, const Plans &plans, const bf16 *pixels, const Grid *grids, uint32_t count,
                   void *scratch, size_t scratch_size, bf16 *out, cudaStream_t stream, const Mutant *mutant = nullptr);

}  // namespace cere::model::inkling_vision
