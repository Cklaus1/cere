// Prefill selection signature: the CUDA-side collector (contract in prefill_signature.h).
#include "model/prefill_signature.h"

#include "kernels/k2_gemm.h"
#include "kernels/k3_hc.h"
#include "kernels/k5_gdn_chunk.h"
#include "kernels/k6_attention.h"
#include "model/model.h"

#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cere::model::prefill_sig {
namespace {

const char *out_name(kernels::GemmOut out) { return out == kernels::GemmOut::kF32 ? "f32" : "bf16"; }

template <typename T>
bool read_attr(const cublasLtMatmulAlgo_t &algo, cublasLtMatmulAlgoConfigAttributes_t attr, T *value) {
    size_t written = 0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, value, sizeof(T), &written) == CUBLAS_STATUS_SUCCESS;
}

PinTable pin_table(const kernels::GemmPins &pins) {
    PinTable t;
    t.n = pins.n;
    t.k = pins.k;
    t.out = out_name(pins.out);
    t.cublaslt_version = pins.cublaslt_version;
    for (uint32_t i = 0; i < pins.count; ++i) {
        const kernels::GemmPinRange &r = pins.ranges[i];
        PinRange range;
        range.m_lo = r.m_lo;
        range.m_hi = r.m_hi;
        range.config.algo_id = r.config.algo_id;
        range.config.tile = uint32_t(r.config.tile);
        range.config.stages = uint32_t(r.config.stages);
        range.config.splitk = r.config.splitk;
        range.config.reduction = uint32_t(r.config.reduction);
        range.config.partials = r.config.partials;
        range.config.custom = r.config.custom_option;
        t.ranges.push_back(range);
    }
    return t;
}

cere_status plan_selection(const char *name, const kernels::GemmPlan &plan, PlanSelection *out) {
    out->name = name;
    out->n = plan.n();
    out->k = plan.k();
    out->out = out_name(plan.out());
    out->max_m = plan.max_m();
    const cublasLtMatmulAlgo_t *algos = nullptr;
    const uint32_t count = plan.bucket_algos(&algos);
    for (uint32_t b = 0; b < count; ++b) {
        BucketAlgo a;
        a.m_hi = uint32_t{1} << b;
        uint16_t inner = 0;
        uint16_t cluster = 0;
        if (!read_attr(algos[b], CUBLASLT_ALGO_CONFIG_ID, &a.algo_id) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_TILE_ID, &a.tile) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_STAGES_ID, &a.stages) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &a.splitk) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &a.reduction) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &a.custom) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &a.swizzle) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner) ||
            !read_attr(algos[b], CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster)) {
            return CERE_ERR_INTERNAL;
        }
        a.inner = inner;
        a.cluster = cluster;
        out->buckets.push_back(a);
    }
    return CERE_OK;
}

}  // namespace

cere_status collect(const Workspace &ws, PrefillSelection *out) {
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    PrefillSelection s;
    s.cublaslt_version = cublasLtGetVersion();
    // gemm_pins_for's tables, in its order (k2_gemm.h kGemmPinsTables: torch's pins, then the bucket re-pins).
    for (uint32_t i = 0; i < kernels::kGemmPinsTableCount; ++i) {
        s.pins.push_back(pin_table(*kernels::kGemmPinsTables[i]));
    }
    // The plans instantiate_model builds at max_m = ws.max_tokens (forward.cpp): not lm_head (m = 1), not the rows
    // plans of the batched decode step. attn_index_qk exists only above k6::kMaxContext (max_m 0: left out).
    const struct {
        const char *name;
        const kernels::GemmPlan *plan;
    } plans[] = {
        {"hc_down", &ws.hc.down},
        {"hc_up", &ws.hc.up},
        {"attn_qkv", &ws.attn_qkv},
        {"attn_o", &ws.attn_o},
        {"attn_index_qk", &ws.attn_index_qk},
        {"gdn_qkv", &ws.gdn_qkv},
        {"gdn_z", &ws.gdn_z},
        {"gdn_ba", &ws.gdn_ba},
        {"gdn_out", &ws.gdn_out},
        {"ple_key", &ws.ple_key},
        {"ple_value", &ws.ple_value},
        {"moe_router", &ws.moe_router},
        {"moe_shared_gate_up", &ws.moe_shared_gate_up},
        {"moe_shared_down", &ws.moe_shared_down},
    };
    for (const auto &p : plans) {
        if (p.plan->max_m() == 0) continue;
        PlanSelection plan;
        if (const cere_status st = plan_selection(p.name, *p.plan, &plan); st != CERE_OK) return st;
        s.plans.push_back(std::move(plan));
    }
#if defined(CERE_MOE_P3B)
    s.p3b = true;
    s.p3b_buckets.assign(p3b::kPinnedBuckets, p3b::kPinnedBuckets + p3b::kPinnedBucketCount);
#endif
    s.k5_chunk_tokens = uint32_t(k5::kChunkTokens);
    s.k6_max_context = uint32_t(k6::kMaxContext);
    *out = std::move(s);
    return CERE_OK;
}

}  // namespace cere::model::prefill_sig
