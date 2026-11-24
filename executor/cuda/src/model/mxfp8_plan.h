// The CUDA-free part of the MXFP8 dense weights (track Fp4S3, fp4local design note "S3 perf plan", FL-30;
// kernels/k2_mxfp8.h): which target GEMM weights of a plan are MXFP8, whether their groups are uniform, and the bytes
// of the dequantize staging the model's k2m::Table needs. Pure arithmetic over the host plan's tensor names, dtypes
// and shapes, so the memory plan sizes the staging before ALLOCATE binds anything (Device::estimate == allocate), the
// binder refuses with the same text, and the host tests check it without CUDA
// (executor/tests/host/test_mxfp8_plan.cpp).
//
// A GEMM weight `P.weight` [n, k] (P one of the 384 dense projections of the 48 target layers below; the MTP layer's
// stay BF16) is BF16, or MXFP8: F8_E4M3 [n, k] with `P.weight_scale` U8 [n, k / 32] (UE8M0), both DEVICE, k a multiple
// of 32. A group the forward runs as one unit has one format: attention q|k|v (one packed GEMM: the weights one
// arena_order run and the scales another), the four GDN in-projections (one gemm_segments launch) and the shared
// expert's gate/up/down (one gemm_segments launch for gate/up, the same plans' down).
#pragma once

#include "cere_executor.h"
#include "host/device.h"
#include "model/layout.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cere::model::mxfp8 {

constexpr uint64_t kGroupK = 32;  // k per scale byte (kernels/k2_mxfp8.h k2m::kGroupK)

// What plan_of found.
struct Plan {
    std::string error;           // empty: every group is well formed (the binder refuses with this text otherwise)
    uint64_t tensors = 0;        // MXFP8 weight tensors (384 on the S3 artifact)
    uint64_t gemms = 0;          // MXFP8 GEMMs, q|k|v counted once: the k2m::Table entries bind_weights adds (360)
    uint64_t staging_bytes = 0;  // the largest MXFP8 GEMM's BF16 upcast 2 n k (k2m::Table::staging_needed) rounded
                                 // up to layout::kAlign; 0 without MXFP8 weights (the plan is then unchanged)
};

namespace detail {

enum class Format { kAbsent, kBf16, kMxfp8, kInvalid };

struct Index {
    explicit Index(const host::ModelPlan &plan) : plan_(plan) {
        map_.reserve(plan.tensors.size());
        for (size_t i = 0; i < plan.tensors.size(); ++i) map_.emplace(plan.tensors[i].name, i);
    }
    const host::TensorPlan *find(const std::string &name) const {
        const auto it = map_.find(name);
        return it == map_.end() ? nullptr : &plan_.tensors[it->second];
    }

private:
    const host::ModelPlan &plan_;
    std::unordered_map<std::string_view, size_t> map_;
};

inline std::string shape_text(const host::TensorPlan &t) {
    std::string s = "[";
    for (uint32_t d = 0; d < t.ndim; ++d) s += (d ? ", " : "") + std::to_string(t.shape[d]);
    return s + "]";
}

// The format of `prefix`.weight; with kMxfp8 its [n, k] in *n / *k; with kInvalid the reason in *error.
inline Format format_of(const Index &index, const std::string &prefix, uint64_t *n, uint64_t *k, std::string *error) {
    const std::string wname = prefix + ".weight";
    const std::string sname = prefix + ".weight_scale";
    const host::TensorPlan *w = index.find(wname);
    const host::TensorPlan *s = index.find(sname);
    if (w == nullptr) return Format::kAbsent;  // the binder names the missing tensor
    if (w->dtype != CERE_DTYPE_F8_E4M3) {
        if (s == nullptr) return Format::kBf16;  // any other dtype: the binder's BF16 check names it
        *error = "tensor " + sname + ": an MXFP8 scale next to a weight of dtype " + std::to_string(w->dtype) +
                 " (MXFP8 needs " + wname + " F8_E4M3)";
        return Format::kInvalid;
    }
    if (w->ndim != 2 || w->shape[1] == 0 || w->shape[1] % kGroupK != 0 || w->placement != CERE_PLACEMENT_DEVICE) {
        *error = "tensor " + wname + ": MXFP8 weight " + shape_text(*w) + " placement " +
                 std::to_string(w->placement) + ", expected DEVICE [n, k] with k a multiple of 32";
        return Format::kInvalid;
    }
    if (s == nullptr) {
        *error = "missing tensor " + sname + " (the UE8M0 scales of the MXFP8 weight " + wname + ")";
        return Format::kInvalid;
    }
    if (s->dtype != CERE_DTYPE_U8 || s->ndim != 2 || s->shape[0] != w->shape[0] ||
        s->shape[1] != w->shape[1] / kGroupK || s->placement != CERE_PLACEMENT_DEVICE) {
        *error = "tensor " + sname + ": dtype " + std::to_string(s->dtype) + " shape " + shape_text(*s) +
                 " placement " + std::to_string(s->placement) + ", expected U8 (UE8M0) DEVICE [" +
                 std::to_string(w->shape[0]) + ", " + std::to_string(w->shape[1] / kGroupK) + "]";
        return Format::kInvalid;
    }
    *n = w->shape[0];
    *k = w->shape[1];
    return Format::kMxfp8;
}

}  // namespace detail

// The GEMM weight prefixes of target layer `layer` ("model.language_model.layers.L.").
inline std::string layer_prefix(uint32_t layer) { return "model.language_model.layers." + std::to_string(layer) + "."; }

// Checks every MXFP8 candidate of `plan` (the groups of the header comment, per the layer kind of
// cfg.full_attention_layers) and sizes the staging. Never throws except std::bad_alloc.
inline Plan plan_of(const host::ModelPlan &plan) {
    // A group: up to 4 GEMMs of one format; a GEMM: up to 3 packed parts (q|k|v).
    struct Group {
        const char *gemm[4][3];
    };
    static const Group kAttention[] = {
        {{{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}}},
        {{{"self_attn.o_proj"}}},
        {{{"self_attn.indexer.index_qk_proj"}}},
    };
    static const Group kGdn[] = {
        {{{"linear_attn.in_proj_qkv"},
          {"linear_attn.in_proj_z"},
          {"linear_attn.in_proj_b"},
          {"linear_attn.in_proj_a"}}},
        {{{"linear_attn.out_proj"}}},
    };
    static const Group kShared = {
        {{"mlp.shared_expert.gate_proj"}, {"mlp.shared_expert.up_proj"}, {"mlp.shared_expert.down_proj"}}};
    const detail::Index index(plan);
    Plan out;
    uint64_t largest = 0;
    const auto check = [&](const std::string &prefix, const Group &g) {
        bool any_bf16 = false;
        bool any_mxfp8 = false;
        std::string names;
        for (const auto &gemm : g.gemm) {
            uint64_t rows = 0;
            uint64_t k = 0;
            bool mxfp8 = false;
            for (const char *part : gemm) {
                if (part == nullptr) break;
                names += (names.empty() ? "" : ", ") + prefix + part + ".weight";
                uint64_t n = 0;
                uint64_t kk = 0;
                std::string error;
                switch (detail::format_of(index, prefix + part, &n, &kk, &error)) {
                    case detail::Format::kAbsent: break;
                    case detail::Format::kBf16: any_bf16 = true; break;
                    case detail::Format::kInvalid:
                        if (out.error.empty()) out.error = error;
                        break;
                    case detail::Format::kMxfp8:
                        any_mxfp8 = mxfp8 = true;
                        if (k != 0 && kk != k && out.error.empty()) {
                            out.error = "tensor " + prefix + part + ".weight: k " + std::to_string(kk) +
                                        " differs from its packed group's " + std::to_string(k);
                        }
                        k = kk;
                        rows += n;
                        ++out.tensors;
                        break;
                }
            }
            if (mxfp8) {
                ++out.gemms;
                largest = std::max(largest, 2 * rows * k);
            }
        }
        if (any_bf16 && any_mxfp8 && out.error.empty()) {
            out.error = "tensors " + names + ": BF16 and MXFP8 mixed in one group (one packed or segmented GEMM "
                        "runs one format)";
        }
    };
    const cere_qwen4_exp_config &cfg = plan.config;
    for (uint32_t l = 0; l < cfg.num_hidden_layers && l < 64; ++l) {
        const std::string prefix = layer_prefix(l);
        if (layout::full_attention(cfg, l)) {
            for (const Group &g : kAttention) check(prefix, g);
        } else {
            for (const Group &g : kGdn) check(prefix, g);
        }
        check(prefix, kShared);
    }
    out.staging_bytes = (largest + layout::kAlign - 1) / layout::kAlign * layout::kAlign;
    return out;
}

}  // namespace cere::model::mxfp8
