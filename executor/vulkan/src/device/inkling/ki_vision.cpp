// Inkling S12: ki_vision.h (the order of operations is executor/cuda/src/model/inkling/vision.cpp's encode).
#include "device/inkling/ki_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

#include "kernels/k2_gemm_vk.h"
#include "kernels/ki_ops_vk.h"

namespace cere::vkink {
namespace {

constexpr uint64_t kTail = 256;  // slack after every buffer (vector loads may read past a row end)
constexpr float kLnEps = 1e-5f;  // vision_config layer_norm_eps
constexpr uint32_t kRopeFreqs = 16, kVisHeads = 16, kVisHeadDim = 64;

enum Global : uint32_t { kPatchW, kPosTable, kClass, kClassPos, kPreW, kPreB, kProjIn, kProjOut };
enum LayerT : uint32_t {
    kLn1W, kLn1B, kQW, kQB, kKW, kKB, kVW, kVB, kOutW, kOutB, kLs1, kLn2W, kLn2B, kFc1W, kFc1B, kFc2W, kFc2B, kLs2
};
constexpr const char *kGlobalNames[kVisGlobals] = {
    "backbone.embeddings.patch_embedding.weight", "backbone.embeddings.position_embedding",
    "backbone.embeddings.class_embedding",        "backbone.embeddings.cls_position_embedding",
    "backbone.ln_pre.weight",                     "backbone.ln_pre.bias",
    "proj_in.weight",                             "proj_out.weight"};
constexpr const char *kLayerNames[kVisPerLayer] = {
    "ln_1.weight",           "ln_1.bias",           "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.out_proj.weight", "self_attn.out_proj.bias", "ls_1.gamma",  "ln_2.weight",
    "ln_2.bias",             "mlp.c_fc.weight",     "mlp.c_fc.bias",           "mlp.c_proj.weight",
    "mlp.c_proj.bias",       "ls_2.gamma"};

struct VisPush {  // shaders/kiv_vision.comp
    uint64_t a, b, c, d, e, f, g, h;
    uint32_t n0, n1, n2, n3;
    float f0, f1;
};
static_assert(sizeof(VisPush) == 88, "kiv_vision.comp push layout");

#define KV_TRY(expr)                       \
    do {                                   \
        const cere_status s_ = (expr);     \
        if (s_ != CERE_OK) return s_;      \
    } while (0)

}  // namespace

std::string vision_tensor_name(uint32_t i) {
    const std::string root = "model.visual.";
    if (i < kVisGlobals) return root + kGlobalNames[i];
    i -= kVisGlobals;
    return root + "backbone.layers." + std::to_string(i / kVisPerLayer) + "." + kLayerNames[i % kVisPerLayer];
}

uint32_t vision_tensor_shape(uint32_t i, uint64_t shape[2]) {
    const auto set = [shape](std::initializer_list<uint64_t> dims) {
        uint32_t n = 0;
        for (const uint64_t d : dims) shape[n++] = d;
        return n;
    };
    if (i < kVisGlobals) {
        switch (Global(i)) {
            case kPatchW: return set({kVisHidden, kPatchRow});
            case kPosTable: return set({32 * 32, kVisHidden});
            case kProjIn: return set({kProjHidden, kMergeRow});
            case kProjOut: return set({kVisOut, kProjHidden});
            default: return set({kVisHidden});
        }
    }
    switch (LayerT((i - kVisGlobals) % kVisPerLayer)) {
        case kQW: case kKW: case kVW: case kOutW: return set({kVisHidden, kVisHidden});
        case kFc1W: return set({kVisInter, kVisHidden});
        case kFc1B: return set({kVisInter});
        case kFc2W: return set({kVisHidden, kVisInter});
        default: return set({kVisHidden});
    }
}

Vision::~Vision() {
    for (vk::Buffer &b : owned_) ctx_.free(b);
}

cere_status Vision::alloc(uint64_t bytes, uint64_t *addr) {
    vk::Buffer b;
    if (const cere_status st = ctx_.allocate(bytes + kTail, b, false); st != CERE_OK) return st;
    if (b.map == nullptr) {
        ctx_.free(b);
        return CERE_ERR_UNSUPPORTED;  // pixels in and rows out go through the mapping (UMA only)
    }
    std::memset(b.map, 0, bytes + kTail);
    owned_.push_back(b);
    *addr = b.addr;
    return CERE_OK;
}

void *Vision::host(uint64_t addr) const {
    for (const vk::Buffer &b : owned_)
        if (b.addr == addr) return b.map;
    return nullptr;
}

cere_status Vision::init(const Find &find, const Options &opt, std::string *why) {
    if (ready_ || opt.max_patches < 4 || opt.max_patches % 4 != 0 || opt.max_patches > kVisMaxPatches)
        return CERE_ERR_ARGUMENT;
    opt_ = opt;
    for (uint32_t i = 0; i < kVisTensors; ++i) {
        uint64_t shape[2];
        const uint32_t nd = vision_tensor_shape(i, shape);
        const uint64_t bytes = 2 * (nd == 1 ? shape[0] : shape[0] * shape[1]);
        TensorView t;
        if (!find(vision_tensor_name(i), &t) || t.bytes != bytes || t.addr % 16 != 0) {
            if (why != nullptr) *why = vision_tensor_name(i);
            return CERE_ERR_ARGUMENT;
        }
        w_[i] = t.addr;
        weight_bytes_ += bytes;
    }
    const uint64_t P = opt.max_patches, S = P + kVisMaxImages;
    KV_TRY(alloc(P * kPatchRow * 2, &pixels_));
    KV_TRY(alloc(P / 4 * kVisOut * 2, &out_));
    KV_TRY(alloc(S * kVisHidden * 2, &x_));
    KV_TRY(alloc(S * kVisHidden * 2, &normed_));
    KV_TRY(alloc(3 * S * kVisHidden * 2, &qkv_));
    KV_TRY(alloc(S * kVisHidden * 2, &attn_));
    KV_TRY(alloc(S * kVisInter * 2, &wide_));  // gelu(fc1) [S, 4,096], merger gelu [T, 4,096]
    KV_TRY(alloc(S * kVisInter * 4, &acc_));   // every linear's F32 output (q | k | v: 3 x [S, 1,024])
    KV_TRY(alloc(S * 8 * 4, &info_));
    KV_TRY(alloc(kVisMaxImages * 2 * 4, &imgs_));
    // The axial RoPE's cos / sin per position (0 = class row, grid index + 1) and frequency, as the CUDA tower:
    // inv_freq = 1 / theta^(2f / 32) in FP32, angle = float(pos) * inv_freq, cosf / sinf.
    rope_positions_ = kVisMaxPatches / 2 + 2;
    KV_TRY(alloc(uint64_t(rope_positions_) * kRopeFreqs * 2 * 4, &rope_));
    {
        float inv[kRopeFreqs];
        for (uint32_t f = 0; f < kRopeFreqs; ++f)
            inv[f] = 1.0f / std::pow(10000.0f, float(2 * f) / float(kVisHeadDim / 2));
        float *t = static_cast<float *>(host(rope_));
        for (uint32_t p = 0; p < rope_positions_; ++p)
            for (uint32_t f = 0; f < kRopeFreqs; ++f) {
                const float angle = float(p) * inv[f];
                t[(p * kRopeFreqs + f) * 2] = std::cos(angle);
                t[(p * kRopeFreqs + f) * 2 + 1] = std::sin(angle);
            }
    }
    std::vector<vk::PipelineSpec> specs = vki::pipeline_specs(false);
    for (int m = 0; m < 6; ++m) specs.push_back({"kiv_vision_m" + std::to_string(m) + ".spv", sizeof(VisPush), 0, {}});
    if (opt.k2) {
        KV_TRY(alloc(vkk2::gemm_workspace_bytes(uint32_t(S), kVisInter), &k2_ws_));
        const std::vector<vk::PipelineSpec> k2 = vkk2::pipeline_specs(true, false, {kVisHidden, kVisInter});
        specs.insert(specs.end(), k2.begin(), k2.end());
    }
    std::string w;
    if (const cere_status st = ctx_.warm(specs, &w); st != CERE_OK) {
        if (why != nullptr) *why = "pipelines: " + w;
        return st;
    }
    if (opt.k2) {
        if (!vkk2::gemm_ready(ctx_) && vkk2::gemm_probe(ctx_) != CERE_OK) {
            if (why != nullptr) *why = "k2: no cooperative-matrix GEMM on this device";
            return CERE_ERR_UNSUPPORTED;
        }
        // Prepared row info of every K2 weight (k % 64 == 0: all but the patch embedding), one submit.
        KV_TRY(ctx_.begin());
        for (uint32_t i = 0; i < kVisTensors; ++i) {
            uint64_t shape[2];
            if (vision_tensor_shape(i, shape) != 2 || i == kPatchW) continue;
            if (const cere_status st = alloc((shape[0] * 4 + 15) / 16 * 16, &winfo_[i]); st != CERE_OK) {
                ctx_.submit_wait();
                return st;
            }
            if (const cere_status st =
                    vkk2::gemm_prepare_weight(ctx_, w_[i], 0, uint32_t(shape[0]), uint32_t(shape[1]), winfo_[i]);
                st != CERE_OK) {
                ctx_.submit_wait();
                return st;
            }
        }
        KV_TRY(ctx_.submit_wait());
    }
    ready_ = true;
    return CERE_OK;
}

cere_status Vision::op(uint32_t mode, const void *push, uint32_t gx, uint32_t gy, uint32_t gz) {
    return ctx_.dispatch(ctx_.pipeline("kiv_vision_m" + std::to_string(mode) + ".spv", sizeof(VisPush)), push, gx, gy,
                         gz);
}

// y F32 [m][n] = x BF16 [m][k] . W[wi] BF16 [n][k]^T, then a barrier.
cere_status Vision::linear(uint64_t x, uint32_t wi, uint64_t y, uint32_t m, uint32_t n, uint32_t k) {
    if (!opt_.k2 || wi == kPatchW) {
        KV_TRY(vki::dense(ctx_, x, w_[wi], y, m, n, k, true));
    } else {
        // GEMMs share one workspace: each waits for the previous one (the caller's barrier after each linear).
        KV_TRY(vkk2::run(ctx_, x, w_[wi], 0, y, m, n, k, true, 0, k2_ws_, winfo_[wi]));
    }
    return ctx_.barrier();
}

cere_status Vision::encode(const VisionGrid *grids, uint32_t count) {
    if (!ready_ || grids == nullptr || count == 0 || count > kVisMaxImages) return CERE_ERR_ARGUMENT;
    const uint32_t has_class = opt_.no_class ? 0 : 1;
    uint32_t *info = static_cast<uint32_t *>(host(info_));
    uint32_t *imgs = static_cast<uint32_t *>(host(imgs_));
    uint64_t P = 0, S = 0;
    uint32_t max_seq = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const VisionGrid g = grids[i];
        if (g.h < 2 || g.w < 2 || g.h % 2 != 0 || g.w % 2 != 0 || uint64_t(g.h) * g.w > kVisMaxPatches ||
            P + uint64_t(g.h) * g.w > opt_.max_patches) {
            return CERE_ERR_ARGUMENT;
        }
        const uint32_t n = g.h * g.w;
        imgs[2 * i] = uint32_t(S);
        imgs[2 * i + 1] = n + has_class;
        max_seq = std::max(max_seq, n + has_class);
        uint32_t *r = info + S * 8;
        if (has_class) {
            std::memset(r, 0, 32);
            r[0] = ~0u;
            r += 8;
        }
        for (uint32_t p = 0; p < n; ++p, r += 8) {  // merge-block order (CUDA block_order)
            const uint32_t bw = g.w / 2, in_col = p % 2, in_row = (p / 2) % 2, bc = (p / 4) % bw, br = p / (4 * bw);
            r[0] = uint32_t(P) + p;
            r[1] = br * 2 + in_row;
            r[2] = bc * 2 + in_col;
            r[3] = g.h;
            r[4] = g.w;
            r[5] = r[6] = r[7] = 0;
        }
        P += n;
        S += n + has_class;
    }
    const uint32_t rows = uint32_t(S);
    const uint64_t accq = acc_;  // q | k | v: 3 x [S, 1,024] F32
    const auto layernorm = [&](uint64_t x, uint64_t w, uint64_t b, uint64_t y) {
        const VisPush p{x, w, b, y, 0, 0, 0, 0, rows, kVisHidden, 0, 0, kLnEps, 0};
        KV_TRY(op(2, &p, rows, 1));
        return ctx_.barrier();
    };
    const auto bias_act = [&](uint64_t acc, uint64_t bias, uint32_t m, uint32_t n, bool gelu, uint64_t y) {
        const VisPush p{acc, bias, y, 0, 0, 0, 0, 0, m, n, n, gelu ? 1u : 0u, 0, 0};
        KV_TRY(op(0, &p, (n + 255) / 256, m));
        return ctx_.barrier();
    };
    const auto residual = [&](uint64_t bias, uint64_t gamma) {
        const VisPush p{acc_, bias, gamma, x_, 0, 0, 0, 0, rows, 0, 0, 0, 0, 0};
        KV_TRY(op(1, &p, kVisHidden / 256, rows));
        return ctx_.barrier();
    };
    const auto fail = [&](cere_status st) {
        ctx_.submit_wait();  // returns the latched error without running a half-recorded command buffer
        return st;
    };
#define KV_REC(expr)                                  \
    do {                                              \
        const cere_status s_ = (expr);                \
        if (s_ != CERE_OK) return fail(s_);           \
    } while (0)

    // Embeddings + pre LayerNorm.
    KV_TRY(ctx_.begin());
    KV_REC(linear(pixels_, kPatchW, acc_, uint32_t(P), kVisHidden, kPatchRow));
    {
        const VisPush p{acc_, W(kPosTable), W(kClass), W(kClassPos), info_, normed_, 0, 0, rows, 0, 0, 0, 0, 0};
        KV_REC(op(3, &p, kVisHidden / 256, rows));
        KV_REC(ctx_.barrier());
    }
    KV_REC(layernorm(normed_, W(kPreW), W(kPreB), x_));
    KV_TRY(ctx_.submit_wait());

    const float scale_log2 = float(1.0 / std::sqrt(double(kVisHeadDim)) * 1.4426950408889634);
    for (uint32_t l = 0; l < kVisLayers; ++l) {  // one submit per layer
        KV_TRY(ctx_.begin());
        KV_REC(layernorm(x_, WL(l, kLn1W), WL(l, kLn1B), normed_));
        const uint32_t qkv[3] = {kQW, kKW, kVW};
        for (uint32_t j = 0; j < 3; ++j)
            KV_REC(linear(normed_, kVisGlobals + l * kVisPerLayer + qkv[j], accq + j * S * kVisHidden * 4, rows,
                          kVisHidden, kVisHidden));
        {
            const VisPush p{accq,  WL(l, kQB), WL(l, kKB), WL(l, kVB), qkv_, info_, rope_, 0,
                            rows,  opt_.swap_rope ? 1u : 0u, 0, 0, 0, 0};
            KV_REC(op(4, &p, kVisHidden / 2 / 256, rows));
            KV_REC(ctx_.barrier());
        }
        {
            const VisPush p{qkv_, attn_, imgs_, 0, 0, 0, 0, 0, rows, 0, 0, 0, scale_log2, 0};
            KV_REC(op(5, &p, (max_seq + 31) / 32, kVisHeads, count));
            KV_REC(ctx_.barrier());
        }
        KV_REC(linear(attn_, kVisGlobals + l * kVisPerLayer + kOutW, acc_, rows, kVisHidden, kVisHidden));
        KV_REC(residual(WL(l, kOutB), WL(l, kLs1)));
        KV_REC(layernorm(x_, WL(l, kLn2W), WL(l, kLn2B), normed_));
        KV_REC(linear(normed_, kVisGlobals + l * kVisPerLayer + kFc1W, acc_, rows, kVisInter, kVisHidden));
        KV_REC(bias_act(acc_, WL(l, kFc1B), rows, kVisInter, true, wide_));
        KV_REC(linear(wide_, kVisGlobals + l * kVisPerLayer + kFc2W, acc_, rows, kVisHidden, kVisInter));
        KV_REC(residual(WL(l, kFc2B), WL(l, kLs2)));
        KV_TRY(ctx_.submit_wait());
    }

    // Merger per image on its patch rows (class row dropped): [n / 4, 4,096] views of the contiguous rows.
    KV_TRY(ctx_.begin());
    uint64_t token = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t tokens = grids[i].h * grids[i].w / 4;
        const uint64_t patches = x_ + (uint64_t(imgs[2 * i]) + has_class) * kVisHidden * 2;
        KV_REC(linear(patches, kProjIn, acc_, tokens, kProjHidden, kMergeRow));
        KV_REC(bias_act(acc_, 0, tokens, kProjHidden, true, wide_));
        KV_REC(linear(wide_, kProjOut, acc_, tokens, kVisOut, kProjHidden));
        KV_REC(bias_act(acc_, 0, tokens, kVisOut, false, out_ + token * kVisOut * 2));
        token += tokens;
    }
#undef KV_REC
    return ctx_.submit_wait();
}

}  // namespace cere::vkink
