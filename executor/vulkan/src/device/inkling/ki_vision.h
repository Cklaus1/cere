// Inkling S12 (inkling design note 3.6): Inkling 2 Mini's vision tower on Vulkan, the port of the CUDA tower
// executor/cuda/src/model/inkling/vision.h (S7; transformers 5.24.0 `Inkling2VisionModel.forward`,
// modeling_inkling2.py:1563-1603) over the images of one request in one pass:
//   patch embedding (Linear 1,200 -> 1,024, no bias) + bilinear 32 x 32 position table, one class token per image;
//   pre LayerNorm; 23 x { x += ls_1 * out(attn(rope2d(q, k), v)(ln_1(x))); x += ls_2 * fc2(gelu(fc1(ln_2(x)))) } with
//   attention bidirectional within each image; merger per image on its patch rows (class row dropped):
//   proj_out(gelu(proj_in(x.view(-1, 4,096)))) -> [tokens, 2,304], one row per `<|unused_200054|>` placeholder.
// Every linear has an FP32 output rounded to BF16 once (shaders/kiv_vision.comp: bias / activation / residual), as the
// CUDA tower. Linears run on vki::dense or, with Options::k2, on K2 (the patch embedding, k = 1,200, always on dense).
// Gate (statistical, as S7's): ktest/ki_vision.cpp + tools/ki_vision_compare.py against the recorded reference tower.
// Memory: host-visible buffers only (UMA: AMD's RADV, lavapipe). One submit per encoder layer (AMD's ring timeout).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cere_executor.h"
#include "device/inkling/ki_model.h"
#include "launch/vk_launch.h"

namespace cere::vkink {

constexpr uint32_t kVisHidden = 1024, kVisInter = 4096, kVisLayers = 23, kPatchRow = 1200, kMergeRow = 4096;
constexpr uint32_t kProjHidden = 4096, kVisOut = kHidden;
constexpr uint32_t kVisGlobals = 8, kVisPerLayer = 18, kVisTensors = kVisGlobals + kVisLayers * kVisPerLayer;  // 422
constexpr uint32_t kVisMaxImages = 32;    // per call (the tokenizer sidecar's MAX_IMAGES)
constexpr uint32_t kVisMaxPatches = 16384;  // per call: 4,096 image tokens of 2 x 2 patches

// Checkpoint name of vision tensor i (< kVisTensors, CUDA tower order: 8 globals then 18 per layer) and its shape
// (returns ndim, shape[0..1]).
std::string vision_tensor_name(uint32_t i);
uint32_t vision_tensor_shape(uint32_t i, uint64_t shape[2]);

// One image: its grid in 20-px patches (h, w even, >= 2). Its pixel rows are h * w rows of 1,200 BF16 in the processor's
// merge-block order (InklingProcessor.patch_rows); it yields h * w / 4 output rows.
struct VisionGrid {
    uint32_t h = 0, w = 0;
};

class Vision {
public:
    struct Options {
        uint32_t max_patches = 4096;  // patches of one call (a multiple of 4, <= kVisMaxPatches)
        bool k2 = false;              // linears on K2 (cooperative matrices, AMD)
        // Fail-first mutants (ktest only): the tower without class tokens; RoPE h and w positions swapped.
        bool no_class = false, swap_rope = false;
    };

    explicit Vision(vk::Context &ctx) : ctx_(ctx) {}
    ~Vision();
    Vision(const Vision &) = delete;
    Vision &operator=(const Vision &) = delete;

    // Binds the 422 `model.visual.*` tensors (exact sizes), allocates the scratch for opt.max_patches, warms the
    // pipelines. ARGUMENT (why: the tensor) on a missing / wrongly sized tensor; UNSUPPORTED without host-visible memory
    // or, with k2, without the cooperative-matrix GEMM.
    cere_status init(const Find &find, const Options &opt, std::string *why);
    // Host-visible input / output: pixels BF16 [opt.max_patches][1,200]; out BF16 [opt.max_patches / 4][2,304].
    uint16_t *pixels_host() const { return static_cast<uint16_t *>(host(pixels_)); }
    const uint16_t *out_host() const { return static_cast<const uint16_t *>(host(out_)); }
    uint64_t out_addr() const { return out_; }
    // Encodes `count` images whose pixel rows are in pixels_host() (images concatenated) into out_host() (image i's
    // rows follow image i - 1's). ARGUMENT on a bad grid or more patches than opt.max_patches.
    cere_status encode(const VisionGrid *grids, uint32_t count);
    uint64_t weight_bytes() const { return weight_bytes_; }

private:
    cere_status alloc(uint64_t bytes, uint64_t *addr);
    void *host(uint64_t addr) const;
    cere_status linear(uint64_t x, uint32_t wi, uint64_t y, uint32_t m, uint32_t n, uint32_t k);
    cere_status op(uint32_t mode, const void *push, uint32_t gx, uint32_t gy, uint32_t gz = 1);
    uint64_t W(uint32_t i) const { return w_[i]; }
    uint64_t WL(uint32_t l, uint32_t j) const { return w_[kVisGlobals + l * kVisPerLayer + j]; }

    vk::Context &ctx_;
    Options opt_;
    bool ready_ = false;
    uint64_t w_[kVisTensors] = {};
    uint64_t winfo_[kVisTensors] = {};
    uint64_t weight_bytes_ = 0;
    uint64_t pixels_ = 0, out_ = 0, x_ = 0, normed_ = 0, qkv_ = 0, attn_ = 0, wide_ = 0, acc_ = 0, info_ = 0, imgs_ = 0,
             rope_ = 0, k2_ws_ = 0;
    uint32_t rope_positions_ = 0;
    std::vector<vk::Buffer> owned_;
};

}  // namespace cere::vkink
