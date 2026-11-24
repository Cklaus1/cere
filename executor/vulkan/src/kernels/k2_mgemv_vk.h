// VulkanAmd P1: K2's small-M GEMV and the MXFP8 upcast on Vulkan (shaders/k2_mgemv.comp, k2_mxfp8_dequant.comp).
//
// Same contract as executor/hip/src/k2_mgemv.h: y[r, j] = sum_i x[r, i] w[j, i] for 1 <= m <= kGemvMaxM; w BF16 [n, k]
// (scales == 0) or MXFP8 (q E4M3 [n, k], scales UE8M0 [n, k / 32]); y FP32 or BF16 (RNE, NaN 0x7FFF). Addresses are
// buffer device addresses, 16-byte aligned. k % 16 == 0 (BF16) or k % 32 == 0 (MXFP8); the weight's bytes < 2^32 (one <= 4 GiB
// buffer per tensor: a larger tensor is split by rows by the caller). CERE_ERR_ARGUMENT outside the contract.
//
// The summation order is fixed by the mapping, and the order part of the mapping (G, E) is chosen from k alone (mapping_for): every row's
// bits are independent of m and of the call's first row, and an MXFP8 weight gives its BF16 upcast's bits.
// The functions record into the context's open command buffer (Context::begin .. submit_wait); no allocation.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk2 {

constexpr uint32_t kGemvMaxM = 16;

// Appends the specs of `src` to `dst` that `dst` does not hold yet (a union of pipeline lists keeps each pipeline once).
inline void append_unique(std::vector<vk::PipelineSpec> &dst, const std::vector<vk::PipelineSpec> &src) {
    for (const vk::PipelineSpec &s : src) {
        bool have = false;
        for (const vk::PipelineSpec &o : dst)
            have = have || (o.spv == s.spv && o.push_bytes == s.push_bytes && o.required_subgroup == s.required_subgroup && o.spec == s.spec);
        if (!have) dst.push_back(s);
    }
}


// G lanes per output column, E elements per lane per step, WG threads per workgroup, U steps loaded ahead, J columns
// per lane group (sharing each decoded activation). G and E fix the summation order; WG, U and J are speed-only.
struct Mapping {
    uint32_t G = 32, E = 16, WG = 256, U = 1, J = 1;
};
// The default mapping: G and E from k alone (the order), WG / U / J from (k, m) by speed.
Mapping mapping_for(uint32_t k, uint32_t m);

// The SPIR-V file name of a mapping for a format and row capacity (build.sh compiles the same set).
std::string spv_name(bool mx, uint32_t mr, const Mapping &mp);

// The pipeline gemv requests for a format, a row count m (1..kGemvMaxM) and a mapping: the single selection shared by
// gemv() and pipeline_specs().
vk::PipelineSpec gemv_spec(bool mx, uint32_t m, const Mapping &mp);

// Every pipeline gemv (default mapping, any m in 1..16) can request for the given k values (BF16 weights: bf16_ks; MXFP8
// weights: mx_ks), plus the MXFP8 upcast's when `dequant`. Model values (qwen38-flash-next): BF16 k in {320 (K3 up
// projection), 640 (shared-expert down, S), 2560 (hidden: K3's hc*hidden is 10240, K4, K5 in-projections, K7/K13 router
// and shared gate/up), 6144 (K5 out_proj), 10240 (K3 down)}; MXFP8 k in {2560, 6144} (the family's own tests). Forced
// mappings (`force`, tuning only) are not enumerated.
std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<uint32_t> &bf16_ks, const std::vector<uint32_t> &mx_ks, bool dequant);

// Records one GEMV dispatch with the default mapping, or `force` (tuning/timing only: a different G or E is a
// different order, so a forced mapping is never used for the invariance rows).
cere_status gemv(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t scales, uint64_t y, uint32_t m, uint32_t n,
                 uint32_t k, bool out_f32, const Mapping *force = nullptr);

// Records the exact BF16 upcast of an MXFP8 weight into w.
cere_status mxfp8_dequantize(vk::Context &ctx, uint64_t q, uint64_t scales, uint32_t n, uint32_t k, uint64_t w);

}  // namespace cere::vkk2
