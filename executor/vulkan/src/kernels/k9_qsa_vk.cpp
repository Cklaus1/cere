// VulkanAmd P2: see k9_qsa_vk.h.
#include "kernels/k9_qsa_vk.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

namespace cere::vkk9 {
namespace {

struct PrepPush {
    uint64_t src, weight, cs, pos, slots, ring, dst, fault;
    int32_t pos0;
    uint32_t tokens, stride;
    float eps;
    int32_t rope_first, rope_end, context, g_lo, groups, j, n, pad;
};
static_assert(sizeof(PrepPush) == 112, "push layout matches k9_prep.comp");

struct LogitsPush {
    uint64_t q, keys, slots, pos, logits, fault;
    uint32_t row0, tile_rows, rows_total, seq_len, ld, skip;
};
static_assert(sizeof(LogitsPush) == 72, "push layout matches k9_logits.comp");

struct TopkPush {
    uint64_t logits, pos, out, fault;
    uint32_t ld, row0, rows_total, seq_len;
};
static_assert(sizeof(TopkPush) == 48, "push layout matches k9_topk.comp");

struct AttnPush {
    uint64_t q, idx, k, v, slots, out_or_po, pml, kb, vb;
    uint32_t q_stride, out_stride, kstride, vstride;
    float sl2;
    uint32_t cap;
};
static_assert(sizeof(AttnPush) == 96, "push layout matches k9_attn.comp");

struct CoopPush {
    uint64_t q, idx, k, v, slots, out, flags, route;
    uint32_t q_stride, out_stride, kstride, vstride;
    float sl2;
    uint32_t rows;
};
static_assert(sizeof(CoopPush) == 88, "push layout matches p4att_sparse.comp");

struct DensePush {  // p4att_dense.comp's (vkk6's CoopPush) with a route
    uint64_t q, k, v, slots, out, flags, route;
    uint32_t tokens, pos0, kstride, vstride;
    float sl2;
    uint32_t q_stride, out_stride, pad;
};
static_assert(sizeof(DensePush) == 88, "push layout matches p4att_dense.comp");

struct RoutePush {
    uint64_t idx, route;
    uint32_t rows, pad;
};
static_assert(sizeof(RoutePush) == 24, "push layout matches p4att_route.comp");

struct FallbackPush {
    AttnPush base;
    uint64_t flags;
};
static_assert(sizeof(FallbackPush) == 104, "push layout matches k9_attn.comp FALLBACK 1");

struct KvCopyPush {
    uint64_t idx, kc, vc, slots, kb, vb, nword;
    uint32_t count, cap;
};
static_assert(sizeof(KvCopyPush) == 64, "push layout matches k9_kvcopy.comp");

struct CombinePush {
    uint64_t po, pml, dst;
    uint32_t out_stride, pad;
};
static_assert(sizeof(CombinePush) == 32, "push layout matches k9_combine.comp");

constexpr float kLog2e = 1.4426950408889634f;
constexpr int64_t kI32 = INT32_MAX;

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }
bool a16(uint64_t a) { return a % 16 == 0; }
bool a256(uint64_t a) { return a % 256 == 0; }
bool sg_ok(const Options &o) {
    return (o.subgroup == 32 || o.subgroup == 64) && (o.prefill == Prefill::kFp32 || o.prefill == Prefill::kCoopmat);
}
bool coop(const Options &o) { return o.prefill == Prefill::kCoopmat; }
std::string coop_spv(bool fp8) { return fp8 ? "p4att_sparse_fp8.spv" : "p4att_sparse_bf16.spv"; }
std::string fallback_spv(bool fp8) { return fp8 ? "k9_attn_rows_fb_fp8.spv" : "k9_attn_rows_fb_bf16.spv"; }
std::string dense_spv(bool fp8) { return fp8 ? "p4att_dense_fp8.spv" : "p4att_dense_bf16.spv"; }
size_t flag_bytes(int64_t rows) { return (size_t(rows) * kKvHeads * 3 * sizeof(uint32_t) + 255) & ~size_t(255); }
size_t route_bytes(int64_t rows) { return (size_t((rows + 15) / 16) * sizeof(uint32_t) + 255) & ~size_t(255); }
bool i32(int64_t x) { return x >= -kI32 && x <= kI32; }

// k9_qsa.cu's entry-point Bounds checks.
bool rope_ok(const Bounds &b) { return b.rope_first < b.rope_end && i32(b.rope_first) && i32(b.rope_end); }
bool rope_rows_ok(const Bounds &b, int64_t lo, int64_t pos0, int64_t tokens) {
    return rope_ok(b) && lo >= b.rope_first && pos0 <= b.rope_end && tokens <= b.rope_end - pos0;
}
bool context_ok(const Bounds &b, int64_t pos0, int64_t tokens) {
    return b.context >= 1 && i32(b.context) && pos0 <= b.context && tokens <= b.context - pos0;
}

PrepPush prep_push(uint64_t src, uint64_t weight, float eps, uint64_t cs, const Bounds &b, uint64_t pos,
                   uint64_t slots, uint64_t ring, uint64_t dst) {
    PrepPush p{};
    p.src = src; p.weight = weight; p.cs = cs; p.pos = pos; p.slots = slots; p.ring = ring; p.dst = dst;
    p.fault = b.fault; p.eps = eps;
    p.rope_first = int32_t(b.rope_first); p.rope_end = int32_t(b.rope_end);
    p.context = int32_t(b.context > kI32 ? kI32 : b.context);
    return p;
}

std::string prep_spv(int mode, bool at) { return "k9_prep_m" + std::to_string(mode) + (at ? "_at" : "") + ".spv"; }
const vk::Pipeline &prep_pipe(vk::Context &ctx, int mode, bool at) {
    return ctx.pipeline(prep_spv(mode, at), sizeof(PrepPush), 32);
}

std::string logits_spv(int r) { return "k9_logits_r" + std::to_string(r) + ".spv"; }
std::string attn_spv(bool split, bool fp8, int pvh) {
    return std::string("k9_attn_") + (split ? "split" : "rows") + (fp8 ? "_fp8" : "_bf16") + "_pvh" + std::to_string(pvh) + ".spv";
}

int logits_r(int64_t rows, int impl) { return impl == 1 ? 8 : impl == 2 ? 1 : rows <= 1 ? 1 : 8; }
int logits_kb(int r) { return r == 1 ? 256 : 64; }

void record_logits(vk::Context &ctx, uint64_t q, uint64_t keys, uint64_t slots, uint64_t pos, int64_t row0,
                   int64_t tile_rows, int64_t rows_total, int64_t seq_len, uint64_t fault, uint64_t logits, int64_t ld,
                   bool skip, int r, const Options &o) {
    if (ld <= 0 || tile_rows <= 0) return;
    const vk::Pipeline &p = ctx.pipeline(logits_spv(r), sizeof(LogitsPush), o.subgroup);
    const LogitsPush pc{q + uint64_t(row0) * kIndexHeads * kIndexHeadDim * 2, keys, slots, pos, logits, fault,
                        uint32_t(row0), uint32_t(tile_rows), uint32_t(rows_total), uint32_t(seq_len), uint32_t(ld),
                        skip ? 1u : 0u};
    ctx.dispatch(p, &pc, uint32_t((ld + logits_kb(r) - 1) / logits_kb(r)), uint32_t((tile_rows + r - 1) / r));
}

void record_topk(vk::Context &ctx, uint64_t logits, int64_t ld, uint64_t pos, int64_t row0, int64_t tile_rows,
                 int64_t rows_total, int64_t seq_len, uint64_t fault, uint64_t out, const Options &o) {
    const vk::Pipeline &p = ctx.pipeline("k9_topk.spv", sizeof(TopkPush), o.subgroup);
    const TopkPush pc{logits, pos, out, fault, uint32_t(ld), uint32_t(row0), uint32_t(rows_total), uint32_t(seq_len)};
    ctx.dispatch(p, &pc, uint32_t(tile_rows));
}

size_t partial_bytes_o(int64_t rows) { return align256(size_t(rows) * kQHeads * kTiles * kHeadDim * sizeof(float)); }
size_t partial_bytes_ml(int64_t rows) { return align256(size_t(rows) * kQHeads * kTiles * 2 * sizeof(float)); }

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(const Options &opt, bool bf16_kv, bool fp8_kv, const std::vector<int> &rows_heads) {
    std::vector<vk::PipelineSpec> s;
    if (!sg_ok(opt)) return s;
    // The K9 prep modes (0 indexer_qk / compress_store, 1 / 2 compress, 3 step, 4 key save, 5 chains step; 1 / 2 also in their
    // "at" form), on 32-lane subgroups whatever Options says (prep_pipe).
    for (int mode = 0; mode <= 5; ++mode) s.push_back({prep_spv(mode, false), sizeof(PrepPush), 32, {}});
    for (int mode = 1; mode <= 2; ++mode) s.push_back({prep_spv(mode, true), sizeof(PrepPush), 32, {}});
    // Indexer logits (R = 1: one row; R = 8: eight rows; impl 0 / 1 / 2 and the step path choose between them) and top-k.
    for (int r : {1, 8}) s.push_back({logits_spv(r), sizeof(LogitsPush), uint32_t(opt.subgroup), {}});
    s.push_back({"k9_topk.spv", sizeof(TopkPush), opt.subgroup, {}});
    s.push_back({"k9_combine.spv", sizeof(CombinePush), opt.subgroup, {}});  // the split attention's combine
    for (int fp8 = 0; fp8 < 2; ++fp8) {
        if (!(fp8 ? fp8_kv : bf16_kv)) continue;
        for (int rh : rows_heads) {
            if (rh != 0 && rh != 3 && rh != 6 && rh != 12) continue;  // qsa_attention_heads refuses it
            const int pvh = rh == 0 ? 12 : rh;
            for (int split = 0; split < 2; ++split) {
                if (!split && coop(opt)) continue;  // the > 16-row side is the coopmat kernel + its fallback
                s.push_back({attn_spv(split != 0, fp8 != 0, pvh), sizeof(AttnPush), opt.subgroup, {}});
            }
        }
        if (coop(opt)) {
            if (opt.dense_blocks) {
                s.push_back({"p4att_route.spv", sizeof(RoutePush), 32, {}});
                s.push_back({dense_spv(fp8 != 0), sizeof(DensePush), 32, {}});
            }
            s.push_back({coop_spv(fp8 != 0), sizeof(CoopPush), 32, {}});
            s.push_back({fallback_spv(fp8 != 0), sizeof(FallbackPush), opt.subgroup, {}});
        }
    }
    // FP8 KV rows path with scratch: the positions decoded once into BF16 copies.
    if (fp8_kv && !coop(opt))
        for (int mode = 0; mode < 3; ++mode)
            s.push_back({"k9_kvcopy_m" + std::to_string(mode) + ".spv", sizeof(KvCopyPush), opt.subgroup, {}});
    // Deduplicate (rows_heads 0 and 12 select the same pipeline).
    std::vector<vk::PipelineSpec> u;
    for (const vk::PipelineSpec &p : s) {
        bool have = false;
        for (const vk::PipelineSpec &o : u) have = have || (o.spv == p.spv && o.push_bytes == p.push_bytes && o.required_subgroup == p.required_subgroup && o.spec == p.spec);
        if (!have) u.push_back(p);
    }
    return u;
}

int64_t logits_tile_rows(int64_t rows, int64_t ctx) {
    const int64_t keys = ctx / kRatio;
    if (rows <= 0 || keys <= 0) return rows > 1 ? rows : 1;
    const int64_t block_q = 128 / kIndexHeads;
    const int64_t bytes_per_row = keys * int64_t(sizeof(float));
    int64_t max_padded = int64_t(kLogitsTileBytes) / bytes_per_row;
    if (max_padded < block_q) max_padded = block_q;
    max_padded = max_padded / block_q * block_q;
    if (max_padded < block_q) max_padded = block_q;
    return rows < max_padded ? rows : max_padded;
}

size_t select_scratch_bytes(int64_t rows, int64_t ctx) {
    if (rows <= 0 || ctx < kRatio) return 0;
    return align256(size_t(logits_tile_rows(rows, ctx)) * size_t(ctx / kRatio) * sizeof(float));
}

size_t coopmat_scratch_bytes(int64_t rows) { return rows <= 0 ? 0 : flag_bytes(rows) + route_bytes(rows); }

size_t attention_scratch_bytes(int64_t rows) {
    if (rows <= 0 || rows > kSplitMaxRows) return 0;
    return partial_bytes_o(rows) + partial_bytes_ml(rows);
}

size_t scratch_bytes(int64_t rows, int64_t ctx) {
    if (rows <= 0) return 0;
    const size_t r = size_t(rows);
    const size_t sel = select_scratch_bytes(rows, ctx), att = attention_scratch_bytes(rows);
    return align256(r * kIndexHeads * kIndexHeadDim * 2) + align256(r * kRowWidth * 4) + (sel > att ? sel : att);
}

cere_status carve_scratch(uint64_t base, size_t bytes, int64_t rows, int64_t ctx, Scratch *out) {
    if (!out || !base || rows <= 0 || ctx <= 0 || !a256(base) || bytes < scratch_bytes(rows, ctx))
        return CERE_ERR_ARGUMENT;
    const size_t r = size_t(rows);
    uint64_t p = base;
    out->q_idx = p;
    p += align256(r * kIndexHeads * kIndexHeadDim * 2);
    out->indices = p;
    p += align256(r * kRowWidth * 4);
    out->work = p;
    out->work_bytes = bytes - size_t(p - base);
    return CERE_OK;
}

cere_status indexer_qk(vk::Context &ctx, uint64_t qk, int64_t qk_stride, uint64_t q_norm_weight, float eps,
                       uint64_t cos_sin, const Bounds &bounds, uint64_t positions, int64_t pos0, int64_t tokens,
                       uint64_t q_idx) {
    if (!qk || !q_norm_weight || !cos_sin || !q_idx || tokens < 0 || pos0 < 0 || qk_stride < kQkWidth ||
        qk_stride > kI32 || !(eps > 0.0f) || tokens > kI32 || pos0 > kI32 || !rope_ok(bounds) || q_idx % 8)
        return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    if (!positions && !rope_rows_ok(bounds, pos0, pos0, tokens)) return CERE_ERR_ARGUMENT;
    PrepPush p = prep_push(qk, q_norm_weight, eps, cos_sin, bounds, positions, 0, 0, q_idx);
    p.pos0 = int32_t(pos0); p.tokens = uint32_t(tokens); p.stride = uint32_t(qk_stride);
    ctx.dispatch(prep_pipe(ctx, 0, false), &p, uint32_t(tokens));
    return CERE_OK;
}

cere_status qsa_compress_store(vk::Context &ctx, uint64_t k_idx, int64_t k_stride, uint64_t k_norm_weight, float eps,
                               uint64_t cos_sin, const Bounds &bounds, uint64_t positions, uint64_t slots,
                               int64_t pos0, int64_t tokens, uint64_t ring, uint64_t keys) {
    if (!k_idx || !k_norm_weight || !cos_sin || !ring || !keys || tokens < 0 || pos0 < 0 ||
        k_stride < kIndexHeadDim || k_stride > kI32 || !(eps > 0.0f) || !rope_ok(bounds) || keys % 8)
        return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    if (!context_ok(bounds, pos0, tokens) ||
        (!positions && !rope_rows_ok(bounds, pos0 / kRatio * kRatio, pos0, tokens)))
        return CERE_ERR_ARGUMENT;
    const int64_t end = pos0 + tokens, g_lo = pos0 / kRatio, g_hi = end / kRatio;
    PrepPush p = prep_push(k_idx, k_norm_weight, eps, cos_sin, bounds, positions, slots, ring, keys);
    p.pos0 = int32_t(pos0); p.tokens = uint32_t(tokens); p.stride = uint32_t(k_stride);
    p.g_lo = int32_t(g_lo); p.groups = int32_t(g_hi - g_lo);
    if (g_hi > g_lo) {
        ctx.dispatch(prep_pipe(ctx, 1, false), &p, uint32_t((g_hi - g_lo + 3) / 4));
        ctx.barrier();  // the compress may read ring rows the ring store overwrites
    }
    const int64_t pending = g_hi * kRatio > pos0 ? g_hi * kRatio : pos0;
    if (pending < end) ctx.dispatch(prep_pipe(ctx, 2, false), &p, 1);
    return CERE_OK;
}

cere_status qsa_compress_store_step(vk::Context &ctx, uint64_t k_idx, uint64_t k_norm_weight, float eps,
                                    uint64_t cos_sin, const Bounds &bounds, uint64_t slots, uint64_t position,
                                    uint64_t ring, uint64_t keys) {
    if (!k_idx || !k_norm_weight || !cos_sin || !position || !ring || !keys || !(eps > 0.0f) || !rope_ok(bounds) ||
        bounds.context < 1 || !i32(bounds.context) || keys % 8)
        return CERE_ERR_ARGUMENT;
    PrepPush p = prep_push(k_idx, k_norm_weight, eps, cos_sin, bounds, position, slots, ring, keys);
    ctx.dispatch(prep_pipe(ctx, 3, false), &p, 1);
    return CERE_OK;
}

cere_status qsa_compress_store_at(vk::Context &ctx, uint64_t k_idx, int64_t k_stride, uint64_t k_norm_weight,
                                  float eps, uint64_t cos_sin, const Bounds &bounds, uint64_t slots,
                                  uint64_t position, int64_t tokens, uint64_t ring, uint64_t keys) {
    if (!k_idx || !k_norm_weight || !cos_sin || !position || !ring || !keys || tokens < 1 || tokens > kI32 ||
        k_stride < kIndexHeadDim || k_stride > kI32 || !(eps > 0.0f) || !rope_ok(bounds) ||
        bounds.context < tokens || !i32(bounds.context) || keys % 8)
        return CERE_ERR_ARGUMENT;
    PrepPush p = prep_push(k_idx, k_norm_weight, eps, cos_sin, bounds, position, slots, ring, keys);
    p.tokens = uint32_t(tokens); p.stride = uint32_t(k_stride);
    const int64_t groups = (kRatio - 1 + tokens) / kRatio;  // the most groups `tokens` tokens complete
    ctx.dispatch(prep_pipe(ctx, 1, true), &p, uint32_t((groups + 3) / 4));
    ctx.barrier();
    ctx.dispatch(prep_pipe(ctx, 2, true), &p, 1);
    return CERE_OK;
}

cere_status qsa_key_save_step(vk::Context &ctx, uint64_t keys, uint64_t slots, const Bounds &bounds,
                              uint64_t position, int64_t j, int64_t n, uint64_t saved) {
    if (!keys || !slots || !position || !saved || j < 1 || n < j || n > 4 * kRatio || bounds.context < 1 ||
        !i32(bounds.context))
        return CERE_ERR_ARGUMENT;
    PrepPush p{};
    p.src = keys; p.slots = slots; p.pos = position; p.dst = saved; p.fault = bounds.fault;
    p.context = int32_t(bounds.context); p.j = int32_t(j); p.n = int32_t(n);
    ctx.dispatch(prep_pipe(ctx, 4, false), &p, 1);
    return CERE_OK;
}

cere_status qsa_chains_step(vk::Context &ctx, const ChainsStep &rows, int chains, uint64_t k_norm_weight, float eps,
                            uint64_t cos_sin, const Bounds &bounds, uint64_t positions, vk::Buffer &table) {
    if (!positions || chains < 1 || chains > kChainsStepRows || !table.map || table.size < kChainsTableBytes ||
        table.addr % 8)
        return CERE_ERR_ARGUMENT;
    int next = 0;
    for (int c = 0; c < chains; ++c) {
        const int first = rows.chain_first[c], count = rows.chain_rows[c];
        if (first != next || count < 1 || count > kChainsStepRows - first) return CERE_ERR_ARGUMENT;
        next = first + count;
        for (int r = first; r < next; ++r) {
            if (rows.context[r] < 1 || !i32(rows.context[r])) return CERE_ERR_ARGUMENT;
            if ((rows.undo_ring[r] || rows.k_idx[r]) && !rows.ring[r]) return CERE_ERR_ARGUMENT;
            if (rows.undo_key[r] && (!rows.undo_ring[r] || !rows.keys[r] || !rows.slots[r] || rows.j[r] < 1 ||
                                     rows.n[r] < rows.j[r] || rows.n[r] > 4 * kRatio))
                return CERE_ERR_ARGUMENT;
            if (rows.k_idx[r] && (!k_norm_weight || !cos_sin || !rows.keys[r] || rows.keys[r] % 8 || !(eps > 0.0f) ||
                                  !rope_ok(bounds)))
                return CERE_ERR_ARGUMENT;
        }
    }
    std::memcpy(table.map, &rows, sizeof rows);
    PrepPush p = prep_push(table.addr, k_norm_weight, eps, cos_sin, bounds, positions, 0, 0, 0);
    ctx.dispatch(prep_pipe(ctx, 5, false), &p, uint32_t(chains));
    return CERE_OK;
}

cere_status indexer_qk_mrope(vk::Context &, uint64_t, int64_t, uint64_t, float, uint64_t, const Bounds &, uint64_t,
                             int64_t, int64_t, int64_t, uint64_t) {
    return CERE_ERR_UNSUPPORTED;
}
cere_status qsa_compress_store_mrope(vk::Context &, uint64_t, int64_t, uint64_t, float, uint64_t, const Bounds &,
                                     uint64_t, int64_t, int64_t, uint64_t, int64_t, int64_t, uint64_t, uint64_t) {
    return CERE_ERR_UNSUPPORTED;
}

cere_status qsa_logits(vk::Context &ctx, uint64_t q_idx, uint64_t keys, uint64_t slots, uint64_t positions_logical,
                       int64_t rows, int64_t seq_len, uint64_t logits, int64_t logits_ld, int impl,
                       const Options &opt) {
    if (!sg_ok(opt) || !q_idx || !keys || !logits || rows < 0 || seq_len <= 0 || seq_len > kI32 ||
        logits_ld < seq_len / kRatio || logits_ld > kI32 || (!positions_logical && rows > seq_len) || !a16(q_idx) ||
        !a16(keys) || logits % 4 || rows > 65535 * 8 || impl < 0 || impl > 2 ||
        uint64_t(rows) * uint64_t(logits_ld) * 4 > (uint64_t(1) << 32) - 4)
        return CERE_ERR_ARGUMENT;
    if (rows == 0) return CERE_OK;
    const int r = logits_r(rows, impl);
    if ((rows + r - 1) / r > 65535) return CERE_ERR_ARGUMENT;
    record_logits(ctx, q_idx, keys, slots, positions_logical, 0, rows, rows, seq_len, 0, logits, logits_ld, false, r,
                  opt);
    return CERE_OK;
}

cere_status qsa_topk(vk::Context &ctx, uint64_t logits, int64_t ld, uint64_t positions_logical, int64_t rows,
                     int64_t seq_len, int impl, uint64_t out_indices, const Options &opt) {
    if (!sg_ok(opt) || !logits || !out_indices || rows <= 0 || rows > 65535 || seq_len <= 0 || seq_len > kI32 ||
        ld < seq_len / kRatio || ld > kI32 || (!positions_logical && rows > seq_len) || impl < 0 || impl > 2 ||
        logits % 4 || out_indices % 4 || uint64_t(rows) * uint64_t(ld) * 4 > (uint64_t(1) << 32) - 4)
        return CERE_ERR_ARGUMENT;
    if (impl == 2) return CERE_ERR_UNSUPPORTED;  // CUDA's thread-block-cluster kernel (as HIP: not on this device)
    record_topk(ctx, logits, ld, positions_logical, 0, rows, rows, seq_len, 0, out_indices, opt);
    return CERE_OK;
}

cere_status qsa_select(vk::Context &ctx, uint64_t q_idx, uint64_t keys, uint64_t slots, uint64_t positions_logical,
                       int64_t rows, int64_t seq_len, uint64_t scratch, size_t scratch_bytes, uint64_t out_indices,
                       const Options &opt) {
    if (!sg_ok(opt) || !q_idx || !keys || !out_indices || rows < 0 || seq_len <= 0 || seq_len > kI32 ||
        (!positions_logical && rows > seq_len) || !a16(q_idx) || !a16(keys) || rows > 65535 || out_indices % 4)
        return CERE_ERR_ARGUMENT;
    if (rows == 0) return CERE_OK;
    const int64_t ld = seq_len / kRatio;
    int64_t tile = rows;
    if (ld > 0) {
        if (!scratch || !a16(scratch)) return CERE_ERR_ARGUMENT;
        tile = logits_tile_rows(rows, seq_len);
        const int64_t fit = int64_t(scratch_bytes / (size_t(ld) * sizeof(float)));
        if (fit < tile) tile = fit >= 32 ? fit / 32 * 32 : fit;
        if (tile <= 0) return CERE_ERR_ARGUMENT;
    }
    const int r = logits_r(rows, 0);
    for (int64_t row0 = 0; row0 < rows; row0 += tile) {
        const int64_t n = rows - row0 < tile ? rows - row0 : tile;
        if (ld > 0) {
            record_logits(ctx, q_idx, keys, slots, positions_logical, row0, n, rows, seq_len, 0, scratch, ld, false,
                          r, opt);
            ctx.barrier();
        }
        record_topk(ctx, scratch, ld, positions_logical, row0, n, rows, seq_len, 0, out_indices, opt);
        if (row0 + n < rows) ctx.barrier();  // the next tile's logits overwrite the scratch
    }
    return CERE_OK;
}

cere_status qsa_select_step(vk::Context &ctx, uint64_t q_idx, uint64_t keys, uint64_t slots, const Bounds &bounds,
                            uint64_t position, int64_t ctx_len, uint64_t scratch, size_t scratch_bytes,
                            uint64_t out_indices, const Options &opt) {
    const int64_t ld = ctx_len / kRatio;
    if (!sg_ok(opt) || !q_idx || !keys || !position || !out_indices || ld <= 0 || ctx_len > kI32 || !scratch ||
        !a16(scratch) || scratch_bytes < size_t(ld) * sizeof(float) || !a16(q_idx) || !a16(keys) ||
        bounds.context < 1 || bounds.context > ctx_len || out_indices % 4)
        return CERE_ERR_ARGUMENT;
    record_logits(ctx, q_idx, keys, slots, position, 0, 1, 1, bounds.context, bounds.fault, scratch, ld, true, 1, opt);
    ctx.barrier();
    record_topk(ctx, scratch, ld, position, 0, 1, 1, bounds.context, bounds.fault, out_indices, opt);
    return CERE_OK;
}

cere_status qsa_attention(vk::Context &ctx, uint64_t q, int64_t q_stride, uint64_t indices, int64_t rows, uint64_t k,
                          int64_t k_stride, uint64_t v, int64_t v_stride, uint64_t k_cache, uint64_t v_cache,
                          uint64_t slots, float scale, uint64_t scratch, size_t scratch_bytes, uint64_t out,
                          int64_t out_stride, bool use_p4, int64_t cls, const Options &opt) {
    return qsa_attention_heads(ctx, q, q_stride, indices, rows, k, k_stride, v, v_stride, k_cache, v_cache, slots,
                               scale, scratch, scratch_bytes, out, out_stride, use_p4, cls, kRowsHeads, opt);
}

cere_status qsa_attention_heads(vk::Context &ctx, uint64_t q, int64_t q_stride, uint64_t indices, int64_t rows,
                                uint64_t k, int64_t k_stride, uint64_t v, int64_t v_stride, uint64_t k_cache,
                                uint64_t v_cache, uint64_t slots, float scale, uint64_t scratch,
                                size_t scratch_bytes, uint64_t out, int64_t out_stride, bool use_p4, int64_t cls,
                                int rows_heads, const Options &opt) {
    (void)use_p4;  // the path is chosen by the class alone (k9_qsa_vk.h)
    if (rows_heads != 0 && rows_heads != 3 && rows_heads != 6 && rows_heads != 12) return CERE_ERR_ARGUMENT;
    const bool bf16_kv = k && v && !k_cache && !v_cache;
    const bool fp8_kv = !k && !v && k_cache && v_cache;
    if (!sg_ok(opt) || !q || !indices || !out || rows < 0 || rows > 65535 || !(bf16_kv || fp8_kv) ||
        q_stride < kQWidth || out_stride < kQWidth || q_stride % 8 || out_stride % 8 || !a16(q) || !a16(out) ||
        indices % 4 || !std::isfinite(scale) || uint64_t(rows) * uint64_t(q_stride > out_stride ? q_stride : out_stride) * 2 > (uint64_t(1) << 32))
        return CERE_ERR_ARGUMENT;
    if (bf16_kv && (k_stride < kKvWidth || v_stride < kKvWidth || k_stride % 8 || v_stride % 8 || !a16(k) || !a16(v) ||
                    k_stride > (1 << 20) || v_stride > (1 << 20)))
        return CERE_ERR_ARGUMENT;
    if (fp8_kv && (!a16(k_cache) || !a16(v_cache))) return CERE_ERR_ARGUMENT;
    if (rows == 0) return CERE_OK;
    if (cls != 0 && (cls < rows || cls > kI32)) return CERE_ERR_ARGUMENT;
    const int64_t sel = cls != 0 ? cls : rows;
    const bool split = sel <= kSplitMaxRows;
    if (split && (!scratch || !a256(scratch) || scratch_bytes < attention_scratch_bytes(rows))) return CERE_ERR_ARGUMENT;
    const int pvh = rows_heads == 0 ? 12 : rows_heads;
    AttnPush pc{};
    pc.q = q; pc.idx = indices; pc.slots = fp8_kv ? slots : 0;
    pc.k = fp8_kv ? k_cache : k; pc.v = fp8_kv ? v_cache : v;
    pc.q_stride = uint32_t(q_stride); pc.out_stride = uint32_t(out_stride);
    pc.kstride = fp8_kv ? uint32_t(kKvWidth) : uint32_t(k_stride * 2);
    pc.vstride = fp8_kv ? uint32_t(kKvWidth) : uint32_t(v_stride * 2);
    pc.sl2 = scale * kLog2e;
    if (!split && coop(opt)) {
        if (!ctx.info().coopmat) return CERE_ERR_UNSUPPORTED;
        if (!scratch || !a256(scratch) || scratch_bytes < coopmat_scratch_bytes(rows)) return CERE_ERR_ARGUMENT;
        // [flags: (row, KV head) x 3 words][route: one word per 16-row block]
        const uint64_t flags = scratch, route = opt.dense_blocks ? scratch + flag_bytes(rows) : 0;
        const uint32_t blocks = uint32_t((rows + 15) / 16);
        if (route) {
            const RoutePush rp{indices, route, uint32_t(rows), 0};
            ctx.dispatch(ctx.pipeline("p4att_route.spv", sizeof(RoutePush), 32), &rp, blocks);
            ctx.barrier();
            const DensePush dp{q, pc.k, pc.v, pc.slots, out, flags, route, uint32_t(rows), 0, pc.kstride, pc.vstride,
                               pc.sl2, uint32_t(q_stride), uint32_t(out_stride), 0};
            ctx.dispatch(ctx.pipeline(dense_spv(fp8_kv), sizeof(DensePush), 32), &dp, kKvHeads * 3, blocks);
        }
        const CoopPush cp{q, indices, pc.k, pc.v, pc.slots, out, flags, route, uint32_t(q_stride), uint32_t(out_stride),
                          pc.kstride, pc.vstride, pc.sl2, uint32_t(rows)};
        ctx.dispatch(ctx.pipeline(coop_spv(fp8_kv), sizeof(CoopPush), 32), &cp, uint32_t(rows), kKvHeads);
        ctx.barrier();
        pc.out_or_po = out;
        pc.cap = 0;  // the fallback reads the E4M3 cache directly (the copy does not change bits)
        const FallbackPush fp{pc, scratch};
        ctx.dispatch(ctx.pipeline(fallback_spv(fp8_kv), sizeof(FallbackPush), opt.subgroup), &fp, uint32_t(rows), kKvHeads);
        return CERE_OK;
    }
    const std::string name = attn_spv(split, fp8_kv, pvh);
    const vk::Pipeline &p = ctx.pipeline(name, sizeof(AttnPush), opt.subgroup);
    if (!split) {
        pc.out_or_po = out;
        // FP8 with scratch: the call's positions decoded once into BF16 copies (k9_kvcopy.comp); same bits either way
        const size_t per = size_t(kKvWidth) * 2;
        const uint64_t cap64 = fp8_kv && scratch && a256(scratch) && scratch_bytes > 256 + 2 * per
                                   ? uint64_t((scratch_bytes - 256) / (2 * per)) : 0;
        const uint32_t cap = uint32_t(cap64 > 524280 ? 524280 : cap64);  // the decode grid stays <= 65,535 groups
        if (cap > 0) {
            const KvCopyPush kp{indices, k_cache, v_cache, slots, scratch + 256, scratch + 256 + uint64_t(cap) * per,
                                scratch, uint32_t(rows * kRowWidth), cap};
            ctx.dispatch(ctx.pipeline("k9_kvcopy_m0.spv", sizeof(KvCopyPush), opt.subgroup), &kp, 1);
            ctx.barrier();
            const uint32_t groups = uint32_t(std::min<int64_t>(512, (rows * kRowWidth + 255) / 256));
            ctx.dispatch(ctx.pipeline("k9_kvcopy_m1.spv", sizeof(KvCopyPush), opt.subgroup), &kp, groups);
            ctx.barrier();
            ctx.dispatch(ctx.pipeline("k9_kvcopy_m2.spv", sizeof(KvCopyPush), opt.subgroup), &kp, (cap * 32 + 255) / 256);
            ctx.barrier();
            pc.kb = kp.kb; pc.vb = kp.vb; pc.cap = cap;
        }
        ctx.dispatch(p, &pc, uint32_t(rows), kKvHeads);
        return CERE_OK;
    }
    pc.out_or_po = scratch;
    pc.pml = scratch + partial_bytes_o(rows);
    ctx.dispatch(p, &pc, kTiles, kKvHeads, uint32_t(rows));
    ctx.barrier();
    const vk::Pipeline &cp = ctx.pipeline("k9_combine.spv", sizeof(CombinePush), opt.subgroup);
    const CombinePush cpc{pc.out_or_po, pc.pml, out, uint32_t(out_stride), 0};
    ctx.dispatch(cp, &cpc, uint32_t(rows), kQHeads);
    return CERE_OK;
}

}  // namespace cere::vkk9
