// WeightLab research points (weightlab design note WL-02, cere_executor.h "WeightLab research"): the one host-only
// table of the points a research plan names, shared by the CUDA library and the CPU toy (the toy validates and writes
// rows exactly as the CUDA build would). The shapes are the CUDA model's (Qwen3.8-Flash-Next, model/model.h): 48
// layers (full attention at L % 4 == 3, Gated DeltaNet elsewhere), hidden 2,560 in 4 HC streams, 512 experts with
// top-10 routing, vocab 248,320.
#pragma once

#include "cere_executor.h"
#include "cere_model_constants.h"

#include <cstdint>

namespace cere::research {

#if CERE_GEN_FAMILY == CERE_FAMILY_QWEN4_EXP
inline constexpr uint32_t kLayers = gen::kNumHiddenLayers;      // CERE_RESEARCH_FINAL_LAYER
inline constexpr uint32_t kHidden = gen::kHiddenSize;
inline constexpr uint32_t kHcHidden = gen::kHcCount * kHidden;  // 10,240: the HC residual state of one row
// q|gate 12,288 + k 512 + v 512
inline constexpr uint32_t kAttnQkvCols =
    2 * gen::kNumAttentionHeads * gen::kHeadDim + 2 * gen::kNumKeyValueHeads * gen::kHeadDim;
inline constexpr uint32_t kGdnQkvCols =
    2 * gen::kLinearNumKeyHeads * gen::kLinearKeyHeadDim + gen::kLinearNumValueHeads * gen::kLinearValueHeadDim;
inline constexpr uint32_t kGdnZCols = gen::kLinearNumValueHeads * gen::kLinearValueHeadDim;
inline constexpr uint32_t kExperts = gen::kNumExperts;
inline constexpr uint32_t kTopK = gen::kNumExpertsPerTok;
inline constexpr uint32_t kSharedInterCols = gen::kSharedExpertIntermediateSize;  // the shared down_proj input
inline constexpr uint32_t kVocab = gen::kVocabSize;
#else
// Research points exist for the Qwen family only (SelfTune ST4): on any other family every research entry point is
// CERE_ERR_UNSUPPORTED before it reads this table, so its points have no columns; the final points keep bit 48.
inline constexpr uint32_t kLayers = CERE_RESEARCH_FINAL_LAYER;
inline constexpr uint32_t kHidden = 0;
inline constexpr uint32_t kHcHidden = 0;
inline constexpr uint32_t kAttnQkvCols = 0;
inline constexpr uint32_t kGdnQkvCols = 0;
inline constexpr uint32_t kGdnZCols = 0;
inline constexpr uint32_t kExperts = 0;
inline constexpr uint32_t kTopK = 0;
inline constexpr uint32_t kVocab = 0;
inline constexpr uint32_t kSharedInterCols = 0;
#endif
static_assert(kLayers == CERE_RESEARCH_FINAL_LAYER, "the final points take layer bit 48");

// Full attention at layers 3, 7, ..., 47 (full_attention_interval 4); Gated DeltaNet at the other 36.
constexpr bool full_attention_layer(uint32_t layer) { return layer % 4 == 3; }

enum class PointId : uint8_t {
    kAttnMixIn,
    kAttnMixOut,
    kMlpMixIn,
    kMlpMixOut,
    kAttnQkv,
    kGdnQkv,
    kGdnZ,
    kOProj,
    kOutProj,
    kRouterOut,
    kRoute,
    kExperts,
    kSharedExpert,
    kSharedDownIn,
    kFinalMixOut,
    kLogitsTopk,
    kLogits,
    kCount,
};

// Which layers a point exists at.
enum class Where : uint8_t { kAll, kFullAttention, kGdn, kFinal };

// One tensor of a capture row: `suffix` ("" for a one-part point, else ".ids" / ".weights" / ".logprobs"), dtype
// (CERE_RESEARCH_DTYPE_*), cols (0: the op's top_k, native.logits.topk only).
struct Part {
    const char *suffix;
    uint32_t dtype;
    uint32_t cols;
};

struct PointInfo {
    const char *name;  // with the literal "L" for the layer
    PointId id;
    Where where;
    uint32_t parts;
    Part part[2];
    // The columns an intervention edits (BF16 values; the router point's are the expert count) and which kinds it
    // takes: `dense` add / ablate_* / replace, `router` expert_bias / expert_mask, `route` expert_force (check_edit's
    // point_cols is then the expert count).
    uint32_t edit_cols;
    bool dense;
    bool router;
    bool route;
    // Logits points: scored rows only, so generated positions only.
    bool logits;
};

// The table entry of `name` (exact match), null for an unknown point.
const PointInfo *find_point(const char *name);
const PointInfo &point_info(PointId id);

// The layer mask the point exists at (bit L, L < 48; bit 48 for the final points).
uint64_t point_layers(const PointInfo &point);

// Bytes of one row of part `part` (top_k: the op's, native.logits.topk only), and of all its parts.
uint64_t part_bytes(const PointInfo &point, uint32_t part, uint32_t top_k);
uint64_t row_bytes(const PointInfo &point, uint32_t top_k);
uint32_t part_cols(const PointInfo &point, uint32_t part, uint32_t top_k);
uint32_t dtype_bytes(uint32_t dtype);

}  // namespace cere::research
