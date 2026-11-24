// The A4 vision tower forward (model/vision.h, m213 design note D10).
//
// math_reference: sglang
#include "model/vision.h"

#include "kernels/kv_vision.h"

#include <algorithm>
#include <initializer_list>

namespace cere::model::vision {

namespace {

static_assert(kv::kHidden == int(kHidden) && kv::kQkvWidth == int(kQkvWidth) && kv::kMerge * kv::kMerge == int(kMergeUnit),
              "vision.h and kv_vision.h describe the same tower");

constexpr const char *kBlockNames[kBlockTensors] = {
    "norm1.weight",          "norm1.bias",          "attn.qkv.weight",       "attn.qkv.bias",
    "attn.proj.weight",      "attn.proj.bias",      "norm2.weight",          "norm2.bias",
    "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight", "mlp.linear_fc2.bias"};
constexpr const char *kMergerNames[6] = {"norm.weight",       "norm.bias",       "linear_fc1.weight",
                                         "linear_fc1.bias",   "linear_fc2.weight", "linear_fc2.bias"};

uint64_t align256(uint64_t n) { return (n + 255) & ~uint64_t(255); }

cere_status status(cudaError_t e) {
    if (e == cudaSuccess) return CERE_OK;
    cudaGetLastError();
    return e == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
}

#define VISION_TRY(expr)                              \
    do {                                              \
        const cere_status s_ = (expr);                \
        if (s_ != CERE_OK) return s_;                 \
    } while (0)
#define VISION_TRY_CUDA(expr) VISION_TRY(status(expr))

// Scratch regions of one pass over P patches, in carving order.
struct Scratch {
    bf16 *x, *normed, *wide, *q, *k, *attn;
    float *acc, *cos_sin;
};

uint64_t carve(uint8_t *base, uint64_t P, Scratch *s) {
    uint64_t at = 0;
    const auto take = [&](uint64_t bytes) {
        uint8_t *p = base == nullptr ? nullptr : base + at;
        at += align256(bytes);
        return p;
    };
    const uint64_t b2 = sizeof(bf16);
    Scratch r{};
    r.x = reinterpret_cast<bf16 *>(take(P * kHidden * b2));
    r.normed = reinterpret_cast<bf16 *>(take(P * kHidden * b2));
    r.wide = reinterpret_cast<bf16 *>(take(P * kInter * b2));  // qkv [P, 3,456], gelu(fc1) [P, 4,304], merger act
    r.q = reinterpret_cast<bf16 *>(take(P * kHidden * b2));
    r.k = reinterpret_cast<bf16 *>(take(P * kHidden * b2));
    r.attn = reinterpret_cast<bf16 *>(take(P * kHidden * b2));
    r.acc = reinterpret_cast<float *>(take(P * kInter * sizeof(float)));  // every linear's F32 output
    r.cos_sin = reinterpret_cast<float *>(take(P * 2 * kv::kRotaryHalf * sizeof(float)));  // max(h, w) <= P rows
    if (s != nullptr) *s = r;
    return at;
}

}  // namespace

std::string tensor_name(uint32_t i) {
    const std::string root = "model.visual.";
    if (i == 0) return root + "patch_embed.proj.weight";
    if (i == 1) return root + "patch_embed.proj.bias";
    if (i == 2) return root + "pos_embed.weight";
    i -= 3;
    if (i < kDepth * kBlockTensors) {
        return root + "blocks." + std::to_string(i / kBlockTensors) + "." + kBlockNames[i % kBlockTensors];
    }
    return root + "merger." + kMergerNames[i - kDepth * kBlockTensors];
}

uint32_t tensor_shape(uint32_t i, uint64_t shape[5]) {
    const auto set = [shape](std::initializer_list<uint64_t> dims) {
        uint32_t n = 0;
        for (const uint64_t d : dims) shape[n++] = d;
        return n;
    };
    if (i == 0) return set({kHidden, 3, 2, 16, 16});
    if (i == 1) return set({kHidden});
    if (i == 2) return set({uint64_t(kPosSide) * kPosSide, kHidden});
    i -= 3;
    if (i < kDepth * kBlockTensors) {
        switch (Block(i % kBlockTensors)) {
            case kQkvW: return set({kQkvWidth, kHidden});
            case kQkvB: return set({kQkvWidth});
            case kProjW: return set({kHidden, kHidden});
            case kFc1W: return set({kInter, kHidden});
            case kFc1B: return set({kInter});
            case kFc2W: return set({kHidden, kInter});
            default: return set({kHidden});  // norm1/norm2 weight and bias, proj and fc2 bias
        }
    }
    switch (Merger(i - kDepth * kBlockTensors)) {
        case kMFc1W: return set({kMergeRow, kMergeRow});
        case kMFc1B: return set({kMergeRow});
        case kMFc2W: return set({kOutHidden, kMergeRow});
        case kMFc2B: return set({kOutHidden});
        default: return set({kHidden});  // merger norm (pre-shuffle: over 1,152)
    }
}

cere_status init_plans(Plans *p, cublasLtHandle_t handle, uint32_t max_patches) {
    if (p == nullptr || handle == nullptr || max_patches < kMergeUnit || max_patches % kMergeUnit != 0) {
        return CERE_ERR_ARGUMENT;
    }
    using kernels::GemmOut;
    const uint32_t merged = max_patches / kMergeUnit;
    VISION_TRY(p->patch.init(handle, kHidden, kPatchRow, max_patches, GemmOut::kF32));
    VISION_TRY(p->qkv.init(handle, kQkvWidth, kHidden, max_patches, GemmOut::kF32));
    VISION_TRY(p->proj.init(handle, kHidden, kHidden, max_patches, GemmOut::kF32));
    VISION_TRY(p->fc1.init(handle, kInter, kHidden, max_patches, GemmOut::kF32));
    VISION_TRY(p->fc2.init(handle, kHidden, kInter, max_patches, GemmOut::kF32));
    VISION_TRY(p->merge1.init(handle, kMergeRow, kMergeRow, merged, GemmOut::kF32));
    VISION_TRY(p->merge2.init(handle, kOutHidden, kMergeRow, merged, GemmOut::kF32));
    return CERE_OK;
}

uint64_t scratch_bytes(uint64_t patches) { return carve(nullptr, patches, nullptr); }

cere_status encode(const Weights &w, const Plans &plans, const bf16 *pixels, uint32_t grid_t, uint32_t grid_h,
                   uint32_t grid_w, void *scratch, size_t scratch_size, bf16 *out, const Taps *taps,
                   cudaStream_t stream) {
    const uint64_t P = uint64_t(grid_t) * grid_h * grid_w;
    if (!w.bound() || pixels == nullptr || out == nullptr || scratch == nullptr || grid_t == 0 || grid_h == 0 ||
        grid_w == 0 || grid_h % kv::kMerge != 0 || grid_w % kv::kMerge != 0 || P > plans.patch.max_m() ||
        P / kMergeUnit > plans.merge1.max_m() || scratch_size < scratch_bytes(P) ||
        reinterpret_cast<uintptr_t>(scratch) % 256 != 0) {
        return CERE_ERR_ARGUMENT;
    }
    Scratch s{};
    carve(static_cast<uint8_t *>(scratch), P, &s);
    const uint32_t m = uint32_t(P);
    const uint32_t merged = m / kMergeUnit;
    const auto tap = [&](const char *name, uint32_t block, const bf16 *v, uint64_t rows, uint64_t cols) {
        return taps == nullptr || taps->sink == nullptr ? CERE_OK : taps->sink(taps->ctx, name, block, v, rows, cols,
                                                                                stream);
    };
    // y = bias_act(x · W^T) through the F32 accumulator (one BF16 rounding, then the residual add when given).
    const auto linear = [&](const kernels::GemmPlan &plan, const bf16 *x, const bf16 *weight, const bf16 *bias,
                            uint32_t rows, kv::Act act, const bf16 *residual, bf16 *y) {
        VISION_TRY(plan.run(x, weight, rows, s.acc, stream));
        return status(kv::bias_act(s.acc, bias, residual, rows, int(plan.n()), act, y, plan.n(), stream));
    };

    // Patch embedding (Conv3d, stride = kernel) + the interpolated position embedding (x += pos_embeds).
    VISION_TRY(linear(plans.patch, pixels, w.patch_w(), w.patch_b(), m, kv::Act::kBias, nullptr, s.x));
    VISION_TRY(tap("patch_embed", 0, s.x, P, kHidden));
    VISION_TRY_CUDA(kv::pos_embed(w.pos_embed(), int(kPosSide), grid_t, grid_h, grid_w, true, s.x, s.x, stream));
    VISION_TRY(tap("pos_embed", 0, s.x, P, kHidden));
    VISION_TRY_CUDA(kv::rope_table(s.cos_sin, std::max(grid_h, grid_w), stream));

    const int64_t frame = int64_t(grid_h) * grid_w;  // cu_seqlens: one segment per frame
    for (uint32_t b = 0; b < kDepth; ++b) {
        VISION_TRY_CUDA(kv::layernorm(s.x, kHidden, w.block(b, kNorm1W), w.block(b, kNorm1B), kEps, m, kHidden,
                                      s.normed, kHidden, stream));
        VISION_TRY(tap("blocks.norm1", b, s.normed, P, kHidden));
        VISION_TRY(linear(plans.qkv, s.normed, w.block(b, kQkvW), w.block(b, kQkvB), m, kv::Act::kBias, nullptr,
                          s.wide));
        VISION_TRY(tap("blocks.attn.qkv", b, s.wide, P, kQkvWidth));
        VISION_TRY_CUDA(kv::rope2d(s.wide, kQkvWidth, s.cos_sin, std::max(grid_h, grid_w), grid_t, grid_h, grid_w, s.q,
                                   s.k, stream));
        VISION_TRY_CUDA(kv::attention(s.q, s.k, kHidden, s.wide + 2 * kHidden, kQkvWidth, m, frame, s.attn, stream));
        VISION_TRY(tap("blocks.attn.core", b, s.attn, P, kHidden));
        VISION_TRY(linear(plans.proj, s.attn, w.block(b, kProjW), w.block(b, kProjB), m, kv::Act::kBias, s.x, s.x));
        VISION_TRY(tap("blocks.attn.residual", b, s.x, P, kHidden));
        VISION_TRY_CUDA(kv::layernorm(s.x, kHidden, w.block(b, kNorm2W), w.block(b, kNorm2B), kEps, m, kHidden,
                                      s.normed, kHidden, stream));
        VISION_TRY(tap("blocks.norm2", b, s.normed, P, kHidden));
        VISION_TRY(linear(plans.fc1, s.normed, w.block(b, kFc1W), w.block(b, kFc1B), m, kv::Act::kGeluTanh, nullptr,
                          s.wide));
        VISION_TRY(tap("blocks.mlp.act", b, s.wide, P, kInter));
        VISION_TRY(linear(plans.fc2, s.wide, w.block(b, kFc2W), w.block(b, kFc2B), m, kv::Act::kBias, s.x, s.x));
        VISION_TRY(tap("blocks.out", b, s.x, P, kHidden));
    }

    // Merger: LayerNorm over 1,152 per patch, the view to [P / 4, 4,608], fc1, GELU (erf), fc2.
    VISION_TRY_CUDA(kv::layernorm(s.x, kHidden, w.merger(kMNormW), w.merger(kMNormB), kEps, m, kHidden, s.normed,
                                  kHidden, stream));
    VISION_TRY(tap("merger.norm", kDepth, s.normed, merged, kMergeRow));
    VISION_TRY(linear(plans.merge1, s.normed, w.merger(kMFc1W), w.merger(kMFc1B), merged, kv::Act::kGeluErf, nullptr,
                      s.wide));
    VISION_TRY(tap("merger.act", kDepth, s.wide, merged, kMergeRow));
    VISION_TRY(linear(plans.merge2, s.wide, w.merger(kMFc2W), w.merger(kMFc2B), merged, kv::Act::kBias, nullptr, out));
    return tap("merger.out", kDepth, out, merged, kOutHidden);
}

}  // namespace cere::model::vision
