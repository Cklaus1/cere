// Inkling 2 Mini's vision tower forward (model/inkling/vision.h, inkling design note 3.6, S7).
//
// math_reference: transformers
#include "model/inkling/vision.h"

#include <initializer_list>

namespace cere::model::inkling_vision {

namespace {

constexpr uint32_t kHidden = kv::kHidden;
constexpr uint32_t kInter = kv::kInter;

constexpr const char *kGlobalNames[kGlobalCount] = {
    "backbone.embeddings.patch_embedding.weight", "backbone.embeddings.position_embedding",
    "backbone.embeddings.class_embedding",        "backbone.embeddings.cls_position_embedding",
    "backbone.ln_pre.weight",                     "backbone.ln_pre.bias",
    "proj_in.weight",                             "proj_out.weight"};
constexpr const char *kLayerNames[kLayerCount] = {
    "ln_1.weight",           "ln_1.bias",           "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.out_proj.weight", "self_attn.out_proj.bias", "ls_1.gamma",  "ln_2.weight",
    "ln_2.bias",             "mlp.c_fc.weight",     "mlp.c_fc.bias",           "mlp.c_proj.weight",
    "mlp.c_proj.bias",       "ls_2.gamma"};

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

// Scratch regions of one pass over S sequence rows (patches + class rows), in carving order.
struct Scratch {
    bf16 *x, *normed, *q, *k, *v, *attn, *wide;
    float *acc;
};

uint64_t carve(uint8_t *base, uint64_t S, Scratch *s) {
    uint64_t at = 0;
    const auto take = [&](uint64_t bytes) {
        uint8_t *p = base == nullptr ? nullptr : base + at;
        at += align256(bytes);
        return p;
    };
    const uint64_t b2 = sizeof(bf16);
    Scratch r{};
    r.x = reinterpret_cast<bf16 *>(take(S * kHidden * b2));
    r.normed = reinterpret_cast<bf16 *>(take(S * kHidden * b2));
    r.q = reinterpret_cast<bf16 *>(take(S * kHidden * b2));
    r.k = reinterpret_cast<bf16 *>(take(S * kHidden * b2));
    r.v = reinterpret_cast<bf16 *>(take(S * kHidden * b2));
    r.attn = reinterpret_cast<bf16 *>(take(S * kHidden * b2));
    r.wide = reinterpret_cast<bf16 *>(take(S * kInter * b2));  // gelu(fc1) [S, 4,096], merger gelu [T, 4,096]
    // every linear's F32 output: q | k | v [S, 1,024] each, fc1 [S, 4,096], proj_in / proj_out [T, <= 4,096]
    r.acc = reinterpret_cast<float *>(take(S * kInter * sizeof(float)));
    if (s != nullptr) *s = r;
    return at;
}

}  // namespace

std::string tensor_name(uint32_t i) {
    const std::string root = "model.visual.";
    if (i < kGlobalCount) return root + kGlobalNames[i];
    i -= kGlobalCount;
    return root + "backbone.layers." + std::to_string(i / kLayerCount) + "." + kLayerNames[i % kLayerCount];
}

uint32_t tensor_shape(uint32_t i, uint64_t shape[2]) {
    const auto set = [shape](std::initializer_list<uint64_t> dims) {
        uint32_t n = 0;
        for (const uint64_t d : dims) shape[n++] = d;
        return n;
    };
    if (i < kGlobalCount) {
        switch (Global(i)) {
            case kPatchW: return set({kHidden, kv::kPatchRow});
            case kPosTable: return set({uint64_t(kv::kPosSide) * kv::kPosSide, kHidden});
            case kProjIn: return set({kv::kProjHidden, kv::kMergeRow});
            case kProjOut: return set({kv::kOutHidden, kv::kProjHidden});
            default: return set({kHidden});
        }
    }
    switch (Layer((i - kGlobalCount) % kLayerCount)) {
        case kQW: case kKW: case kVW: case kOutW: return set({kHidden, kHidden});
        case kFc1W: return set({kInter, kHidden});
        case kFc1B: return set({kInter});
        case kFc2W: return set({kHidden, kInter});
        default: return set({kHidden});
    }
}

bool Weights::bound() const {
    for (const bf16 *t : tensor)
        if (t == nullptr) return false;
    return true;
}

cere_status init_plans(Plans *p, cublasLtHandle_t handle, uint32_t max_patches) {
    if (p == nullptr || handle == nullptr || max_patches < 4 || max_patches % 4 != 0 || max_patches > kMaxPatches) {
        return CERE_ERR_ARGUMENT;
    }
    using kernels::GemmOut;
    const uint32_t rows = max_patches + kMaxImages;  // + the class rows
    const uint32_t tokens = max_patches / 4;
    VISION_TRY(p->patch.init(handle, kHidden, kv::kPatchRow, max_patches, GemmOut::kF32));
    VISION_TRY(p->attn.init(handle, kHidden, kHidden, rows, GemmOut::kF32));
    VISION_TRY(p->fc1.init(handle, kInter, kHidden, rows, GemmOut::kF32));
    VISION_TRY(p->fc2.init(handle, kHidden, kInter, rows, GemmOut::kF32));
    VISION_TRY(p->proj_in.init(handle, kv::kProjHidden, kv::kMergeRow, tokens, GemmOut::kF32));
    VISION_TRY(p->proj_out.init(handle, kv::kOutHidden, kv::kProjHidden, tokens, GemmOut::kF32));
    return CERE_OK;
}

uint64_t scratch_bytes(uint64_t patches, uint32_t images) { return carve(nullptr, patches + images, nullptr); }

cere_status encode(const Weights &w, const Plans &plans, const bf16 *pixels, const Grid *grids, uint32_t count,
                   void *scratch, size_t scratch_size, bf16 *out, cudaStream_t stream, const Mutant *mutant) {
    if (!w.bound() || pixels == nullptr || grids == nullptr || out == nullptr || scratch == nullptr || count == 0 ||
        count > kMaxImages || reinterpret_cast<uintptr_t>(scratch) % 256 != 0) {
        return CERE_ERR_ARGUMENT;
    }
    const Mutant plain{};
    const Mutant &m = mutant == nullptr ? plain : *mutant;
    const int32_t has_class = m.no_class ? 0 : 1;
    kv::Images images{};
    images.count = static_cast<int32_t>(count);
    uint64_t P = 0, S = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const Grid g = grids[i];
        if (g.h < 2 || g.w < 2 || g.h % 2 != 0 || g.w % 2 != 0 || uint64_t(g.h) * g.w > kMaxPatches) {
            return CERE_ERR_ARGUMENT;
        }
        const uint64_t n = uint64_t(g.h) * g.w;
        images.image[i] = kv::Image{static_cast<int32_t>(S), static_cast<int32_t>(P), static_cast<int32_t>(g.h),
                                    static_cast<int32_t>(g.w), has_class};
        if (static_cast<int64_t>(n) + has_class > images.max_seq) images.max_seq = static_cast<int32_t>(n + has_class);
        P += n;
        S += n + has_class;
        if (P > plans.patch.max_m()) return CERE_ERR_ARGUMENT;
    }
    if (S > plans.attn.max_m() || P / 4 > plans.proj_in.max_m() || scratch_size < scratch_bytes(P, count)) {
        return CERE_ERR_ARGUMENT;
    }
    Scratch s{};
    carve(static_cast<uint8_t *>(scratch), S, &s);
    const uint32_t rows = static_cast<uint32_t>(S);
    float *acc_q = s.acc;
    float *acc_k = s.acc + S * kHidden;
    float *acc_v = s.acc + 2 * S * kHidden;

    // Embeddings + pre LayerNorm.
    VISION_TRY(plans.patch.run(pixels, w.global(kPatchW), static_cast<uint32_t>(P), s.acc, stream));
    VISION_TRY_CUDA(kv::embed(s.acc, w.global(kPosTable), w.global(kClass), w.global(kClassPos), images, s.normed,
                              stream));
    VISION_TRY_CUDA(kv::layernorm(s.normed, kHidden, w.global(kPreW), w.global(kPreB), kv::kEps, rows, kHidden, s.x,
                                  kHidden, stream));

    for (uint32_t l = 0; l < kv::kLayers; ++l) {
        VISION_TRY_CUDA(kv::layernorm(s.x, kHidden, w.layer(l, kLn1W), w.layer(l, kLn1B), kv::kEps, rows, kHidden,
                                      s.normed, kHidden, stream));
        VISION_TRY(plans.attn.run(s.normed, w.layer(l, kQW), rows, acc_q, stream));
        VISION_TRY(plans.attn.run(s.normed, w.layer(l, kKW), rows, acc_k, stream));
        VISION_TRY(plans.attn.run(s.normed, w.layer(l, kVW), rows, acc_v, stream));
        VISION_TRY_CUDA(kv::qkv_rope(acc_q, acc_k, acc_v, w.layer(l, kQB), w.layer(l, kKB), w.layer(l, kVB), images,
                                     m.swap_rope_axes, s.q, s.k, s.v, stream));
        VISION_TRY_CUDA(kv::attention(s.q, s.k, s.v, images, s.attn, stream));
        VISION_TRY(plans.attn.run(s.attn, w.layer(l, kOutW), rows, s.acc, stream));
        VISION_TRY_CUDA(kv::scaled_residual(s.acc, w.layer(l, kOutB), w.layer(l, kLs1), rows, s.x, stream));

        VISION_TRY_CUDA(kv::layernorm(s.x, kHidden, w.layer(l, kLn2W), w.layer(l, kLn2B), kv::kEps, rows, kHidden,
                                      s.normed, kHidden, stream));
        VISION_TRY(plans.fc1.run(s.normed, w.layer(l, kFc1W), rows, s.acc, stream));
        VISION_TRY_CUDA(kv::bias_act(s.acc, w.layer(l, kFc1B), rows, kInter, kv::Act::kGeluErf, s.wide, kInter,
                                     stream));
        VISION_TRY(plans.fc2.run(s.wide, w.layer(l, kFc2W), rows, s.acc, stream));
        VISION_TRY_CUDA(kv::scaled_residual(s.acc, w.layer(l, kFc2B), w.layer(l, kLs2), rows, s.x, stream));
    }

    // Merger per image on its patch rows (the class row dropped): [n / 4, 4,096] views of the contiguous rows.
    uint64_t token = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const kv::Image &im = images.image[i];
        const uint32_t tokens = static_cast<uint32_t>(im.h) * static_cast<uint32_t>(im.w) / 4;
        const bf16 *patches = s.x + (uint64_t(im.seq_start) + im.has_class) * kHidden;
        bf16 *dst = out + token * kv::kOutHidden;
        VISION_TRY(plans.proj_in.run(patches, w.global(kProjIn), tokens, s.acc, stream));
        VISION_TRY_CUDA(kv::bias_act(s.acc, nullptr, tokens, kv::kProjHidden, kv::Act::kGeluErf, s.wide,
                                     kv::kProjHidden, stream));
        VISION_TRY(plans.proj_out.run(s.wide, w.global(kProjOut), tokens, s.acc, stream));
        VISION_TRY_CUDA(kv::bias_act(s.acc, nullptr, tokens, kv::kOutHidden, kv::Act::kNone, dst, kv::kOutHidden,
                                     stream));
        token += tokens;
    }
    return CERE_OK;
}

}  // namespace cere::model::inkling_vision
