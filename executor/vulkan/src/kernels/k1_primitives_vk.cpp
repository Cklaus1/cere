// VulkanAmd P2: see k1_primitives_vk.h.
#include "kernels/k1_primitives_vk.h"

#include <algorithm>
#include <cstring>

namespace cere::vkk1 {
namespace {

bool al(uint64_t a, uint64_t n) { return a % n == 0; }
constexpr uint32_t kWg = 64;

struct EmbPush { uint64_t table, ids, out; uint32_t hidden, n; };
struct NormPush { uint64_t x, z, w, y; uint32_t xs, zs, ys, groups, dim, gemma; float eps; uint32_t pad; };
struct SigPush { uint64_t x, gate, out; uint32_t xs, gs, os, width; };
struct TablePush { uint64_t out, n_pos; uint32_t rd; float theta; };
struct RopePush { uint64_t x, y, cos_sin, positions, pos0; uint32_t xs, ys, head_dim, rd; };
struct Top2Push { uint64_t logits, ids, scores, stride, vocab; uint32_t aligned16, out_stride; };
struct KvPush { uint64_t k, v, kc, vc, slots, slot0, at; uint32_t ks, vs, width, pad; };
struct KvRowsPush { uint64_t k, v, table, at; uint32_t ks, vs, width, pad; };
struct QkPush { uint64_t q_gate, k, qw, kw, cos_sin, positions, q_out, k_out, gate_out, pos0; uint32_t qgs, ks, qh, kvh, hd, rd; float eps, pad; };
struct CopyPush { uint64_t table; };
static_assert(sizeof(NormPush) == 64 && sizeof(RopePush) == 56 && sizeof(KvPush) == 72 && sizeof(QkPush) == 112 && sizeof(Top2Push) == 48, "push layouts");

bool small(int64_t v) { return v >= 0 && v < (int64_t(1) << 31); }

// Workgroup size (specialization constant 0) of the row-norm shaders: one 32-lane group per 256 columns, <= 256.
uint32_t norm_wg(int dim) { return std::min<uint32_t>(uint32_t(((dim / 8 + 31) / 32) * 32), 256); }
uint32_t qk_wg(int hd) { return uint32_t(((hd / 8 + 31) / 32) * 32); }

}  // namespace

cere_status embedding_gather(vk::Context &ctx, uint64_t table, int64_t vocab, int hidden, uint64_t ids, int64_t n, uint64_t out) {
    if (n == 0) return CERE_OK;
    if (!table || !ids || !out || vocab < 1 || hidden < 8 || hidden % 8 || n < 0 || n > 65535 || !al(table, 16) || !al(out, 16) ||
        !al(ids, 4) || uint64_t(vocab) * uint64_t(hidden) * 2 >= (uint64_t(1) << 32))
        return CERE_ERR_ARGUMENT;
    const EmbPush pc{table, ids, out, uint32_t(hidden), uint32_t(n)};
    ctx.dispatch(ctx.pipeline("k1_embedding.spv", sizeof pc), &pc, (uint32_t(hidden) / 8 + kWg - 1) / kWg, uint32_t(n));
    return CERE_OK;
}

static cere_status norm(vk::Context &ctx, bool gated, uint64_t x, int64_t xs, uint64_t z, int64_t zs, uint64_t w, int groups, bool gemma,
                        float eps, int64_t rows, int dim, uint64_t y, int64_t ys) {
    if (rows == 0) return CERE_OK;
    if (!x || !w || !y || (gated && !z) || rows < 0 || rows > 0x7fffffff || groups < 1 || dim < 8 || dim % 8 || dim > 8192 || xs % 8 ||
        ys % 8 || (gated && zs % 8) || xs < dim || ys < dim || !small(xs) || !small(ys) || !small(zs) || !al(x, 16) || !al(y, 16) ||
        !al(w, 16) || (gated && !al(z, 16)))
        return CERE_ERR_ARGUMENT;
    const NormPush pc{x, z, w, y, uint32_t(xs), uint32_t(zs), uint32_t(ys), uint32_t(groups), uint32_t(dim), gemma ? 1u : 0u, eps, 0};
    const vk::PipelineSpec ps = rmsnorm_spec(gated, dim);  // the one selection, shared with pipeline_specs and the other families' lists
    ctx.dispatch(ctx.pipeline(ps.spv, ps.push_bytes, ps.required_subgroup, ps.spec), &pc, uint32_t(rows));
    return CERE_OK;
}

cere_status rmsnorm(vk::Context &ctx, uint64_t x, int64_t xs, uint64_t w, int groups, bool gemma, float eps, int64_t rows, int dim,
                    uint64_t y, int64_t ys) {
    return norm(ctx, false, x, xs, 0, 0, w, groups, gemma, eps, rows, dim, y, ys);
}
cere_status rmsnorm_gated_sigmoid(vk::Context &ctx, uint64_t x, int64_t xs, uint64_t z, int64_t zs, uint64_t w, float eps, int64_t rows,
                                  int dim, uint64_t y, int64_t ys) {
    return norm(ctx, true, x, xs, z, zs, w, 1, false, eps, rows, dim, y, ys);
}

cere_status rope_table(vk::Context &ctx, uint64_t cos_sin, int64_t n_pos, int rd, float theta) {
    if (n_pos == 0) return CERE_OK;
    if (!cos_sin || n_pos < 0 || rd < 2 || rd > 1024 || rd % 2 || !al(cos_sin, 4) || uint64_t(n_pos) * uint64_t(rd) * 4 >= (uint64_t(1) << 32))
        return CERE_ERR_ARGUMENT;
    const TablePush pc{cos_sin, uint64_t(n_pos), uint32_t(rd), theta};
    const uint64_t items = uint64_t(n_pos) * (rd / 2);
    ctx.dispatch(ctx.pipeline("k1_rope_table.spv", sizeof pc), &pc, uint32_t((items + 63) / 64));
    return CERE_OK;
}

cere_status rope(vk::Context &ctx, uint64_t x, int64_t xs, uint64_t y, int64_t ys, uint64_t cos_sin, uint64_t positions, uint64_t positions3,
                 int64_t pos0, int64_t tokens, int heads, int hd, int rd) {
    if (positions3) return CERE_ERR_UNSUPPORTED;
    if (tokens == 0) return CERE_OK;
    if (!x || !y || !cos_sin || tokens < 0 || tokens > 65535 || heads < 1 || heads > 65535 || hd < 2 || hd > 1024 || rd < 2 || rd % 2 || rd > hd ||
        hd % 2 || xs % 2 || ys % 2 || !small(xs) || !small(ys) || !al(x, 4) || !al(y, 4) || !al(cos_sin, 4) || (positions && !al(positions, 8)) ||
        (!positions && pos0 < 0))
        return CERE_ERR_ARGUMENT;
    const RopePush pc{x, y, cos_sin, positions, uint64_t(pos0), uint32_t(xs), uint32_t(ys), uint32_t(hd), uint32_t(rd)};
    ctx.dispatch(ctx.pipeline("k1_rope.spv", sizeof pc), &pc, uint32_t(tokens), uint32_t(heads));
    return CERE_OK;
}

cere_status sigmoid_mul(vk::Context &ctx, uint64_t x, int64_t xs, uint64_t gate, int64_t gs, int64_t rows, int width, uint64_t out, int64_t os) {
    if (rows == 0) return CERE_OK;
    if (!x || !gate || !out || rows < 0 || rows > 65535 || width < 8 || width % 8 || xs % 8 || gs % 8 || os % 8 || !small(xs) || !small(gs) ||
        !small(os) || !al(x, 16) || !al(gate, 16) || !al(out, 16))
        return CERE_ERR_ARGUMENT;
    const SigPush pc{x, gate, out, uint32_t(xs), uint32_t(gs), uint32_t(os), uint32_t(width)};
    ctx.dispatch(ctx.pipeline("k1_sigmoid_mul.spv", sizeof pc), &pc, (uint32_t(width) / 8 + kWg - 1) / kWg, uint32_t(rows));
    return CERE_OK;
}

cere_status top2_split(vk::Context &ctx, uint64_t logits, int64_t stride, int64_t rows, int64_t vocab, uint64_t ids, uint64_t scores) {
    if (rows == 0) return CERE_OK;
    if (!logits || !ids || !scores || rows < 0 || rows > 65535 || vocab < 1 || vocab >= (int64_t(1) << 32) || stride < vocab || !al(logits, 2) ||
        !al(ids, 4) || !al(scores, 4))
        return CERE_ERR_ARGUMENT;
    const bool aligned = al(logits, 16) && stride % 8 == 0;  // every row 16-byte aligned
    const Top2Push pc{logits, ids, scores, uint64_t(stride), uint64_t(vocab), aligned ? 1u : 0u, 2};
    ctx.dispatch(ctx.pipeline("k1_top2.spv", sizeof pc, 32), &pc, uint32_t(rows));
    return CERE_OK;
}

cere_status kv_store_fp8(vk::Context &ctx, uint64_t k, int64_t ks, uint64_t v, int64_t vs, int64_t tokens, int width, uint64_t kc, uint64_t vc,
                         uint64_t slots, int64_t slot0) {
    if (tokens == 0) return CERE_OK;
    if (!k || !v || !kc || !vc || tokens < 0 || tokens > 65535 || width < 8 || width % 8 || ks % 8 || vs % 8 || !small(ks) || !small(vs) ||
        !al(k, 16) || !al(v, 16) || !al(kc, 8) || !al(vc, 8) || (slots && !al(slots, 8)) || (!slots && slot0 < 0))
        return CERE_ERR_ARGUMENT;
    const KvPush pc{k, v, kc, vc, slots, uint64_t(slot0), 0, uint32_t(ks), uint32_t(vs), uint32_t(width), 0};
    ctx.dispatch(ctx.pipeline("k1_kv_store.spv", sizeof pc), &pc, (uint32_t(width) / 8 + kWg - 1) / kWg, uint32_t(tokens));
    return CERE_OK;
}

cere_status qk_norm_rope_gate(vk::Context &ctx, uint64_t q_gate, int64_t qgs, uint64_t k, int64_t ks, uint64_t qw, uint64_t kw, uint64_t cos_sin,
                              uint64_t positions, uint64_t positions3, int64_t pos0, int64_t tokens, int qh, int kvh, int hd, int rd, float eps,
                              uint64_t q_out, uint64_t k_out, uint64_t gate_out) {
    if (positions3) return CERE_ERR_UNSUPPORTED;
    if (tokens == 0) return CERE_OK;
    if (!q_gate || !k || !qw || !kw || !cos_sin || !q_out || !k_out || !gate_out || tokens < 0 || tokens > 65535 || qh < 1 || kvh < 0 ||
        qh + kvh > 65535 || hd < 8 || hd % 8 || hd > 1024 || rd < 2 || rd % 2 || rd > hd || qgs % 8 || ks % 8 || !small(qgs) || !small(ks) ||
        qgs < int64_t(qh) * 2 * hd || (kvh > 0 && ks < int64_t(kvh) * hd) || !al(q_gate, 16) || !al(k, 16) || !al(qw, 16) || !al(kw, 16) ||
        !al(cos_sin, 4) || !al(q_out, 16) || !al(k_out, 16) || !al(gate_out, 16) || (positions && !al(positions, 8)) || (!positions && pos0 < 0))
        return CERE_ERR_ARGUMENT;
    const uint32_t wg = qk_wg(hd);
    const QkPush pc{q_gate, k, qw, kw, cos_sin, positions, q_out, k_out, gate_out, uint64_t(pos0), uint32_t(qgs), uint32_t(ks), uint32_t(qh),
                    uint32_t(kvh), uint32_t(hd), uint32_t(rd), eps, 0};
    ctx.dispatch(ctx.pipeline("k1_qk_norm_rope_gate.spv", sizeof pc, 32, {wg}), &pc, uint32_t(tokens), uint32_t(qh + kvh));
    return CERE_OK;
}

cere_status kv_store_fp8_at(vk::Context &ctx, uint64_t k, int64_t ks, uint64_t v, int64_t vs, int64_t tokens, int width, uint64_t kc, uint64_t vc,
                            uint64_t slots, uint64_t at) {
    if (tokens == 0) return CERE_OK;
    if (!k || !v || !kc || !vc || !slots || !at || tokens < 0 || tokens > 65535 || width < 8 || width % 8 || ks % 8 || vs % 8 || !small(ks) ||
        !small(vs) || !al(k, 16) || !al(v, 16) || !al(kc, 8) || !al(vc, 8) || !al(slots, 8) || !al(at, 8))
        return CERE_ERR_ARGUMENT;
    const KvPush pc{k, v, kc, vc, slots, 0, at, uint32_t(ks), uint32_t(vs), uint32_t(width), 0};
    ctx.dispatch(ctx.pipeline("k1_kv_store.spv", sizeof pc), &pc, (uint32_t(width) / 8 + kWg - 1) / kWg, uint32_t(tokens));
    return CERE_OK;
}

cere_status kv_store_fp8_rows(vk::Context &ctx, uint64_t k, int64_t ks, uint64_t v, int64_t vs, int64_t tokens, int width, const KvRows &rows,
                              uint64_t at, vk::Buffer &table) {
    if (!k || !v || !at || tokens < 0 || tokens > kKvRowsMax || width < 8 || width % 8 || ks % 8 || vs % 8 || !small(ks) || !small(vs) || !al(k, 16) ||
        !al(v, 16) || !al(at, 8) || !table.map || table.size < 24 * kKvRowsMax)
        return CERE_ERR_ARGUMENT;
    for (int64_t t = 0; t < tokens; ++t)
        if (!rows.k_cache[t] || !rows.v_cache[t] || !rows.slots[t] || !al(rows.k_cache[t], 8) || !al(rows.v_cache[t], 8) || !al(rows.slots[t], 8))
            return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    uint64_t *e = static_cast<uint64_t *>(table.map);
    for (int64_t t = 0; t < tokens; ++t) { e[3 * t] = rows.k_cache[t]; e[3 * t + 1] = rows.v_cache[t]; e[3 * t + 2] = rows.slots[t]; }
    const KvRowsPush pc{k, v, table.addr, at, uint32_t(ks), uint32_t(vs), uint32_t(width), 0};
    ctx.dispatch(ctx.pipeline("k1_kv_store_rows.spv", sizeof pc), &pc, (uint32_t(width) / 8 + kWg - 1) / kWg, uint32_t(tokens));
    return CERE_OK;
}

cere_status copy_segments(vk::Context &ctx, const CopySegment *segs, int count, vk::Buffer &table) {
    if (!segs || count < 1 || count > kMaxCopySegments || !table.map || table.size < 48ull * uint64_t(count)) return CERE_ERR_ARGUMENT;
    uint64_t max_units = 0;
    for (int i = 0; i < count; ++i) {
        const CopySegment &s = segs[i];
        if (!s.dst || !s.src || s.rows > 65535) return CERE_ERR_ARGUMENT;  // as the CUDA launch: null is refused even for 0 bytes
        const bool vec = ((s.dst | s.src | s.dst_pitch | s.src_pitch | s.width) & 15) == 0;
        max_units = std::max(max_units, (vec ? s.width / 16 : s.width) * s.rows);
    }
    if (max_units == 0) return CERE_OK;
    std::memcpy(table.map, segs, sizeof(CopySegment) * size_t(count));
    // A segment with 0 bytes has width 0 or rows 0 in its table entry: the shader's loop runs zero times.
    const uint32_t gx = uint32_t(std::min<uint64_t>(std::max<uint64_t>((max_units + 255) / 256, 1), 256));
    const CopyPush pc{table.addr};
    ctx.dispatch(ctx.pipeline("k1_copy_segments.spv", sizeof pc), &pc, gx, uint32_t(count));
    return CERE_OK;
}

cere_status top2(vk::Context &ctx, uint64_t logits, int64_t stride, int64_t rows, int64_t vocab, uint64_t out) {
    if (rows == 0) return CERE_OK;
    if (!logits || !out || rows < 0 || rows > 65535 || vocab < 1 || vocab >= (int64_t(1) << 32) || stride < vocab || !al(logits, 2) || !al(out, 16))
        return CERE_ERR_ARGUMENT;
    const bool aligned = al(logits, 16) && stride % 8 == 0;
    const Top2Push pc{logits, out, out + 8, uint64_t(stride), uint64_t(vocab), aligned ? 1u : 0u, 4};
    ctx.dispatch(ctx.pipeline("k1_top2.spv", sizeof pc, 32), &pc, uint32_t(rows));
    return CERE_OK;
}

vk::PipelineSpec rmsnorm_spec(bool gated, int dim) {
    return {gated ? "k1_rmsnorm_g1.spv" : "k1_rmsnorm_g0.spv", sizeof(NormPush), 32, {norm_wg(dim)}};
}

std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<int> &norm_dims, const std::vector<int> &qk_head_dims) {
    std::vector<vk::PipelineSpec> s = {
        {"k1_embedding.spv", sizeof(EmbPush), 0, {}},     {"k1_sigmoid_mul.spv", sizeof(SigPush), 0, {}},
        {"k1_rope_table.spv", sizeof(TablePush), 0, {}},  {"k1_rope.spv", sizeof(RopePush), 0, {}},
        {"k1_top2.spv", sizeof(Top2Push), 32, {}},        {"k1_kv_store.spv", sizeof(KvPush), 0, {}},
        {"k1_kv_store_rows.spv", sizeof(KvRowsPush), 0, {}}, {"k1_copy_segments.spv", sizeof(CopyPush), 0, {}},
    };
    for (int d : norm_dims) {
        s.push_back(rmsnorm_spec(false, d));
        s.push_back(rmsnorm_spec(true, d));
    }
    for (int hd : qk_head_dims) s.push_back({"k1_qk_norm_rope_gate.spv", sizeof(QkPush), 32, {qk_wg(hd)}});
    return s;
}

}  // namespace cere::vkk1
