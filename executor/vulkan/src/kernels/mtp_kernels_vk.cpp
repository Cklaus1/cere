// VulkanAmd P3 (T3): see mtp_kernels_vk.h.
#include "kernels/mtp_kernels_vk.h"

namespace cere::vkmtp {
namespace {

bool al(uint64_t a, uint64_t n) { return a != 0 && a % n == 0; }
constexpr uint64_t k4G = uint64_t(1) << 32;

// k1_rmsnorm.comp's push block (k1_primitives_vk.cpp NormPush), byte for byte.
struct NormPush { uint64_t x, z, w, y; uint32_t xs, zs, ys, groups, dim, gemma; float eps; uint32_t pad; };
static_assert(sizeof(NormPush) == 64, "push layout matches k1_rmsnorm.comp");

struct AddPush { uint64_t hyper, e; uint32_t vecs; uint32_t pad; };  // vecs: 8-element groups of hyper
static_assert(sizeof(AddPush) == 24, "push layout matches mtp_add_embedding.comp");

struct ArgmaxPush { uint64_t logits, map, draft; uint32_t count, vec; };
static_assert(sizeof(ArgmaxPush) == 32, "push layout matches mtp_argmax_map.comp");

constexpr uint32_t kAddThreads = 256;
constexpr uint32_t kMaxGroups = 65535;

}  // namespace

// k1_rmsnorm.comp's workgroup size (a specialization constant), as k1_primitives_vk.cpp norm_wg: one selection here for
// gemma_rows and pipeline_specs.
static uint32_t norm_wg(int dim) {
    const uint32_t wg = uint32_t(((dim / 8 + 31) / 32) * 32);
    return wg > 256 ? 256 : wg;
}

std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<int> &extra_norm_dims) {
    std::vector<vk::PipelineSpec> s = {{"mtp_add_embedding.spv", sizeof(AddPush), 0, {}}, {"mtp_argmax_map.spv", sizeof(ArgmaxPush), 32, {}}};
    std::vector<int> dims = {kHcHidden};  // norm_hidden
    dims.insert(dims.end(), extra_norm_dims.begin(), extra_norm_dims.end());
    for (int d : dims) {
        const vk::PipelineSpec n{"k1_rmsnorm_g0.spv", sizeof(NormPush), 32, {norm_wg(d)}};
        bool have = false;
        for (const vk::PipelineSpec &o : s) have = have || (o.spv == n.spv && o.spec == n.spec);
        if (!have) s.push_back(n);
    }
    return s;
}

namespace detail {
cere_status gemma_rows(vk::Context &ctx, uint64_t x, uint64_t w, float eps, int64_t rows, int dim, uint64_t y) {
    if (!al(x, 16) || !al(w, 16) || !al(y, 16) || rows < 1 || rows > kMaxRows || dim < 8 || dim % 8 || dim > 16384 ||
        uint64_t(rows) * uint64_t(dim) * 2 > k4G)
        return CERE_ERR_ARGUMENT;
    const uint32_t wg = norm_wg(dim);
    const NormPush pc{x, 0, w, y, uint32_t(dim), 0, uint32_t(dim), 1, uint32_t(dim), 1, eps, 0};
    ctx.dispatch(ctx.pipeline("k1_rmsnorm_g0.spv", sizeof pc, 32, {wg}), &pc, uint32_t(rows));
    return CERE_OK;
}
}  // namespace detail

cere_status norm_hidden(vk::Context &ctx, uint64_t x, uint64_t w, float eps, int64_t rows, uint64_t y) {
    return detail::gemma_rows(ctx, x, w, eps, rows, kHcHidden, y);
}

cere_status add_embedding(vk::Context &ctx, uint64_t hyper, uint64_t e, int64_t rows) {
    if (!al(hyper, 16) || !al(e, 16) || rows < 1 || uint64_t(rows) * kHcHidden * 2 >= k4G) return CERE_ERR_ARGUMENT;
    const uint64_t vecs = uint64_t(rows) * (kHcHidden / 8);
    const uint64_t groups = (vecs + kAddThreads - 1) / kAddThreads;
    const uint32_t gx = uint32_t(groups < kMaxGroups ? groups : kMaxGroups);
    const uint32_t gy = uint32_t((groups + gx - 1) / gx);
    if (gy > kMaxGroups) return CERE_ERR_ARGUMENT;
    const AddPush pc{hyper, e, uint32_t(vecs), 0};
    ctx.dispatch(ctx.pipeline("mtp_add_embedding.spv", sizeof pc), &pc, gx, gy);
    return CERE_OK;
}

cere_status argmax_map(vk::Context &ctx, uint64_t logits, uint32_t count, uint64_t map, int64_t rows, uint64_t draft) {
    if (!al(logits, 2) || !al(map, 4) || !al(draft, 4) || count < 1 || count >= (1u << 31) || uint64_t(count) * 2 >= k4G ||
        rows < 1 || rows > kMaxRows)
        return CERE_ERR_ARGUMENT;
    const bool vec = logits % 16 == 0 && count % 8 == 0;  // every row 16-byte aligned
    const ArgmaxPush pc{logits, map, draft, count, vec ? 1u : 0u};
    ctx.dispatch(ctx.pipeline("mtp_argmax_map.spv", sizeof pc, 32), &pc, uint32_t(rows));
    return CERE_OK;
}

}  // namespace cere::vkmtp
