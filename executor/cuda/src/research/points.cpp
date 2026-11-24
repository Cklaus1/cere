#include "research/points.h"

#include <cstring>

namespace cere::research {
namespace {

constexpr uint32_t kBf16 = CERE_RESEARCH_DTYPE_BF16;
constexpr uint32_t kI32 = CERE_RESEARCH_DTYPE_I32;
constexpr uint32_t kF32 = CERE_RESEARCH_DTYPE_F32;

// In PointId order.
constexpr PointInfo kPoints[] = {
    {"native.layers.L.attn_mix.in", PointId::kAttnMixIn, Where::kAll, 1, {{"", kBf16, kHcHidden}, {}}, 0, false,
     false, false, false},
    {"native.layers.L.attn_mix.out", PointId::kAttnMixOut, Where::kAll, 1, {{"", kBf16, kHidden}, {}}, kHidden, true,
     false, false, false},
    {"native.layers.L.mlp_mix.in", PointId::kMlpMixIn, Where::kAll, 1, {{"", kBf16, kHcHidden}, {}}, 0, false, false,
     false, false},
    {"native.layers.L.mlp_mix.out", PointId::kMlpMixOut, Where::kAll, 1, {{"", kBf16, kHidden}, {}}, kHidden, true,
     false, false, false},
    {"native.layers.L.gemm.attn_qkv.out", PointId::kAttnQkv, Where::kFullAttention, 1, {{"", kBf16, kAttnQkvCols}, {}},
     kAttnQkvCols, true, false, false, false},
    {"native.layers.L.gemm.gdn_qkv.out", PointId::kGdnQkv, Where::kGdn, 1, {{"", kBf16, kGdnQkvCols}, {}},
     kGdnQkvCols, true, false, false, false},
    {"native.layers.L.gemm.gdn_z.out", PointId::kGdnZ, Where::kGdn, 1, {{"", kBf16, kGdnZCols}, {}}, kGdnZCols, true,
     false, false, false},
    {"model.layers.L.self_attn.o_proj", PointId::kOProj, Where::kFullAttention, 1, {{"", kBf16, kHidden}, {}},
     kHidden, true, false, false, false},
    {"model.layers.L.linear_attn.out_proj", PointId::kOutProj, Where::kGdn, 1, {{"", kBf16, kHidden}, {}}, kHidden,
     true, false, false, false},
    {"native.layers.L.gemm.moe_router.out", PointId::kRouterOut, Where::kAll, 1, {{"", kBf16, kExperts}, {}},
     kExperts, true, true, false, false},
    {"native.layers.L.moe_route", PointId::kRoute, Where::kAll, 2, {{".ids", kI32, kTopK}, {".weights", kF32, kTopK}},
     kExperts, false, false, true, false},
    {"model.layers.L.mlp.experts", PointId::kExperts, Where::kAll, 1, {{"", kBf16, kHidden}, {}}, kHidden, true, false,
     false, false},
    {"model.layers.L.mlp.shared_expert.down_proj", PointId::kSharedExpert, Where::kAll, 1, {{"", kBf16, kHidden}, {}},
     kHidden, true, false, false, false},
    // The shared expert's down_proj input (the SwiGLU output, K7's "gemm.moe_shared_down.in"): capture only, as K7's
    // ".in" hook points are read only.
    {"native.layers.L.gemm.moe_shared_down.in", PointId::kSharedDownIn, Where::kAll, 1,
     {{"", kBf16, kSharedInterCols}, {}}, 0, false, false, false, false},
    {"native.final_mix.out", PointId::kFinalMixOut, Where::kFinal, 1, {{"", kBf16, kHidden}, {}}, kHidden, true, false,
     false, false},
    {"native.logits.topk", PointId::kLogitsTopk, Where::kFinal, 2, {{".ids", kI32, 0}, {".logprobs", kF32, 0}}, 0,
     false, false, false, true},
    {"native.logits", PointId::kLogits, Where::kFinal, 1, {{"", kBf16, kVocab}, {}}, 0, false, false, false, true},
};
static_assert(sizeof(kPoints) / sizeof(kPoints[0]) == size_t(PointId::kCount), "one entry per PointId");

constexpr bool in_order() {
    for (size_t i = 0; i < size_t(PointId::kCount); ++i) {
        if (size_t(kPoints[i].id) != i) return false;
    }
    return true;
}
static_assert(in_order(), "kPoints is in PointId order");

}  // namespace

const PointInfo *find_point(const char *name) {
    if (name == nullptr) return nullptr;
    for (const PointInfo &p : kPoints) {
        if (std::strcmp(p.name, name) == 0) return &p;
    }
    return nullptr;
}

const PointInfo &point_info(PointId id) { return kPoints[size_t(id)]; }

uint64_t point_layers(const PointInfo &point) {
    if (point.where == Where::kFinal) return uint64_t(1) << kLayers;
    uint64_t mask = 0;
    for (uint32_t L = 0; L < kLayers; ++L) {
        const bool full = full_attention_layer(L);
        if (point.where == Where::kAll || (point.where == Where::kFullAttention) == full) mask |= uint64_t(1) << L;
    }
    return mask;
}

uint32_t dtype_bytes(uint32_t dtype) { return dtype == CERE_RESEARCH_DTYPE_BF16 ? 2 : 4; }

uint32_t part_cols(const PointInfo &point, uint32_t part, uint32_t top_k) {
    return point.part[part].cols != 0 ? point.part[part].cols : top_k;
}

uint64_t part_bytes(const PointInfo &point, uint32_t part, uint32_t top_k) {
    return uint64_t(part_cols(point, part, top_k)) * dtype_bytes(point.part[part].dtype);
}

uint64_t row_bytes(const PointInfo &point, uint32_t top_k) {
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < point.parts; ++i) bytes += part_bytes(point, i, top_k);
    return bytes;
}

}  // namespace cere::research
