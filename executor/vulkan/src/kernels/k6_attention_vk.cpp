// VulkanAmd P2: see k6_attention_vk.h.
#include "kernels/k6_attention_vk.h"

#include <cstring>

namespace cere::vkk6 {
namespace {

struct DecodePush {
    uint64_t q, k, v, slots, po, pml;
    uint32_t tokens, pos0, kstride, vstride;
    float sl2;
    uint32_t pad;
};
static_assert(sizeof(DecodePush) == 72, "push layout matches k6_decode.comp");

struct CombinePush {
    uint64_t po, pml, out, gate;
    uint32_t tokens, pos0, out_stride, gate_stride;
};
static_assert(sizeof(CombinePush) == 48, "push layout matches k6_combine.comp");

struct PrefillPush {
    uint64_t q, k, v, slots, out;
    uint32_t tokens, pos0, kstride, vstride;
    float sl2;
    uint32_t pad;
};
static_assert(sizeof(PrefillPush) == 64, "push layout matches k6_prefill.comp");

struct CoopPush {
    uint64_t q, k, v, slots, out, flags, route;
    uint32_t tokens, pos0, kstride, vstride;
    float sl2;
    uint32_t q_stride, out_stride, pad;
};
static_assert(sizeof(CoopPush) == 88, "push layout matches p4att_dense.comp");

struct FallbackPush {
    PrefillPush base;
    uint64_t flags;
};
static_assert(sizeof(FallbackPush) == 72, "push layout matches k6_prefill.comp FALLBACK 1");

struct GatePush {
    uint64_t a, gate, out;
    uint32_t tokens, out_stride, gate_stride, pad;
};
static_assert(sizeof(GatePush) == 40, "push layout matches k6_gate.comp");

constexpr float kLog2e = 1.4426950408889634f;

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }
bool a16(uint64_t a) { return a % 16 == 0; }
bool ks_ok(const Options &o) {
    return (o.keys_per_split == 64 || o.keys_per_split == 128) && (o.subgroup == 32 || o.subgroup == 64) &&
           (o.prefill == Prefill::kFp32 || o.prefill == Prefill::kCoopmat);
}
bool coop(const Options &o) { return o.prefill == Prefill::kCoopmat; }
size_t flags_bytes(int64_t tokens) { return align256(size_t(tokens) * kKvHeads * 3 * sizeof(uint32_t)); }
uint32_t max_splits(const Options &o) { return uint32_t(kMaxContext) / o.keys_per_split; }

// The four pipelines' selection, one place for the dispatches and pipeline_specs.
vk::PipelineSpec decode_spec(bool fp8, const Options &o) { return {decode_spv(fp8, o.keys_per_split), sizeof(DecodePush), o.subgroup, {}}; }
vk::PipelineSpec combine_spec(const Options &o) { return {combine_spv(o.keys_per_split), sizeof(CombinePush), o.subgroup, {}}; }
vk::PipelineSpec prefill_spec(bool fp8, const Options &o) {
    return {fp8 ? "k6_prefill_fp8.spv" : "k6_prefill_bf16.spv", sizeof(PrefillPush), o.subgroup, {}};
}
vk::PipelineSpec gate_spec(const Options &o) { return {"k6_gate.spv", sizeof(GatePush), o.subgroup, {}}; }
vk::PipelineSpec coop_spec(bool fp8) { return {fp8 ? "p4att_dense_fp8.spv" : "p4att_dense_bf16.spv", sizeof(CoopPush), 32, {}}; }
vk::PipelineSpec fallback_spec(bool fp8, const Options &o) {
    return {fp8 ? "k6_prefill_fb_fp8.spv" : "k6_prefill_fb_bf16.spv", sizeof(FallbackPush), o.subgroup, {}};
}

struct Partials { uint64_t po, pml; size_t bytes; };
Partials partials(uint64_t ws, int64_t rows, const Options &o) {
    const size_t r = size_t(rows) * kQHeads * max_splits(o);
    const size_t po = align256(r * kHeadDim * sizeof(float)), pml = align256(r * 2 * sizeof(float));
    return {ws, ws + po, po + pml};
}

// The split dispatch + the combine (gate 0: none), rows t at positions pos0 + t.
void record_split(vk::Context &ctx, bool fp8, uint64_t q, int64_t tokens, int64_t pos0, uint64_t k, uint32_t kstride_b,
                  uint64_t v, uint32_t vstride_b, uint64_t slots, float scale, const Partials &p, uint64_t out,
                  int64_t out_stride, uint64_t gate, int64_t gate_stride, const Options &o) {
    const vk::PipelineSpec ds = decode_spec(fp8, o), cs = combine_spec(o);
    const vk::Pipeline &dp = ctx.pipeline(ds.spv, ds.push_bytes, ds.required_subgroup, ds.spec);
    const DecodePush dpc{q, k, v, slots, p.po, p.pml, uint32_t(tokens), uint32_t(pos0), kstride_b, vstride_b,
                         scale * kLog2e, 0};
    const uint32_t splits = uint32_t((pos0 + tokens + o.keys_per_split - 1) / o.keys_per_split);  // the last row's (most)
    ctx.dispatch(dp, &dpc, splits, kKvHeads, uint32_t(tokens));
    ctx.barrier();
    const vk::Pipeline &cp = ctx.pipeline(cs.spv, cs.push_bytes, cs.required_subgroup, cs.spec);
    const CombinePush cpc{p.po, p.pml, out, gate, uint32_t(tokens), uint32_t(pos0), uint32_t(out_stride), uint32_t(gate_stride)};
    ctx.dispatch(cp, &cpc, uint32_t(tokens), kQHeads);
}

// The > 16-row path: the FP32 tile kernel, or (P4-A) the coopmat kernel + the FP32 fallback of its flagged rows
// (`flags`: flags_bytes(tokens) of workspace).
void record_prefill(vk::Context &ctx, bool fp8, uint64_t q, int64_t tokens, int64_t pos0, uint64_t k, uint32_t kstride_b,
                    uint64_t v, uint32_t vstride_b, uint64_t slots, float scale, uint64_t out, const Options &o,
                    uint64_t flags = 0) {
    const PrefillPush pc{q, k, v, slots, out, uint32_t(tokens), uint32_t(pos0), kstride_b, vstride_b, scale * kLog2e, 0};
    if (!coop(o)) {
        const vk::PipelineSpec ps = prefill_spec(fp8, o);
        const vk::Pipeline &pp = ctx.pipeline(ps.spv, ps.push_bytes, ps.required_subgroup, ps.spec);
        ctx.dispatch(pp, &pc, kKvHeads, uint32_t((tokens + 3) / 4));
        return;
    }
    const vk::PipelineSpec cs = coop_spec(fp8), fs = fallback_spec(fp8, o);
    const CoopPush cp{q, k, v, slots, out, flags, 0, uint32_t(tokens), uint32_t(pos0), kstride_b, vstride_b, scale * kLog2e,
                      uint32_t(kQWidth), uint32_t(kQWidth), 0};
    ctx.dispatch(ctx.pipeline(cs.spv, cs.push_bytes, cs.required_subgroup, cs.spec), &cp, kKvHeads * 3,
                 uint32_t((tokens + 15) / 16));
    ctx.barrier();
    const FallbackPush fp{pc, flags};
    ctx.dispatch(ctx.pipeline(fs.spv, fs.push_bytes, fs.required_subgroup, fs.spec), &fp, kKvHeads,
                 uint32_t((tokens + 3) / 4));
}

struct Source { bool fp8; uint64_t k, v; uint32_t ks, vs; };

// Validates attention's arguments; fills the K/V source.
bool check(uint64_t q, int64_t tokens, int64_t pos0, uint64_t k, int64_t k_stride, uint64_t v, int64_t v_stride,
           uint64_t k_cache, uint64_t v_cache, uint64_t out, int64_t cls, const Options &o, Source &src) {
    const bool bf16_kv = k && v && !k_cache && !v_cache;
    const bool fp8_kv = !k && !v && k_cache && v_cache;
    if (!ks_ok(o) || !q || !out || !a16(q) || tokens < 0 || pos0 < 0 || pos0 + tokens > kMaxContext ||
        !(bf16_kv || fp8_kv) || cls < 0 || (cls != 0 && cls < tokens) ||
        (bf16_kv && (!a16(k) || !a16(v) || k_stride % 8 || v_stride % 8 || k_stride < kKvWidth || v_stride < kKvWidth ||
                     k_stride * 2 * kMaxContext >= (int64_t(1) << 32) || v_stride * 2 * kMaxContext >= (int64_t(1) << 32))) ||
        (fp8_kv && (!a16(k_cache) || !a16(v_cache))))
        return false;
    src = bf16_kv ? Source{false, k, v, uint32_t(k_stride * 2), uint32_t(v_stride * 2)}
                  : Source{true, k_cache, v_cache, uint32_t(kKvWidth), uint32_t(kKvWidth)};
    return true;
}

}  // namespace

std::string decode_spv(bool fp8, uint32_t ks) {
    return std::string("k6_decode_") + (fp8 ? "fp8" : "bf16") + "_ks" + std::to_string(ks) + ".spv";
}
std::string combine_spv(uint32_t ks) { return "k6_combine_ks" + std::to_string(ks) + ".spv"; }

std::vector<vk::PipelineSpec> pipeline_specs(const Options &opt, bool bf16_kv, bool fp8_kv) {
    std::vector<vk::PipelineSpec> s;
    if (!ks_ok(opt)) return s;
    s.push_back(combine_spec(opt));
    s.push_back(gate_spec(opt));  // dense_attention's prefill path
    for (int fp8 = 0; fp8 < 2; ++fp8) {
        if (!(fp8 ? fp8_kv : bf16_kv)) continue;
        s.push_back(decode_spec(fp8 != 0, opt));   // the split path (tokens / class <= 16)
        if (coop(opt)) {
            s.push_back(coop_spec(fp8 != 0));              // the prefill path (> 16), P4-A
            s.push_back(fallback_spec(fp8 != 0, opt));     // its exact fallback
        } else {
            s.push_back(prefill_spec(fp8 != 0, opt));      // the prefill path (> 16)
        }
    }
    return s;
}

size_t workspace_bytes(int64_t max_tokens, const Options &opt) {
    if (max_tokens <= 0 || !ks_ok(opt)) return 0;
    const int64_t split_rows = max_tokens < kSplitMaxTokens ? max_tokens : kSplitMaxTokens;
    return partials(0, split_rows, opt).bytes + align256(size_t(max_tokens) * kQWidth * 2) +
           (coop(opt) && max_tokens > kSplitMaxTokens ? flags_bytes(max_tokens) : 0);
}

// dense_attention's workspace: [split partials][BF16 pre-gate output][flags]
static uint64_t ws_attn(uint64_t ws, int64_t tokens, const Options &o) {
    return ws + partials(0, tokens < kSplitMaxTokens ? tokens : kSplitMaxTokens, o).bytes;
}
static uint64_t ws_flags(uint64_t ws, int64_t tokens, const Options &o) {
    return ws_attn(ws, tokens, o) + align256(size_t(tokens) * kQWidth * 2);
}

cere_status attention(vk::Context &ctx, uint64_t q, int64_t tokens, int64_t pos0, uint64_t k, int64_t k_stride,
                      uint64_t v, int64_t v_stride, uint64_t k_cache, uint64_t v_cache, uint64_t slots, float scale,
                      uint64_t workspace, size_t ws_bytes, uint64_t out, int64_t cls, const Options &opt) {
    Source s;
    if (!check(q, tokens, pos0, k, k_stride, v, v_stride, k_cache, v_cache, out, cls, opt, s)) return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    const uint64_t sl = s.fp8 ? slots : 0;
    const int64_t sel = cls != 0 ? cls : tokens;
    if (sel > kSplitMaxTokens) {
        if (coop(opt)) {
            if (!ctx.info().coopmat) return CERE_ERR_UNSUPPORTED;
            if (!workspace || !a16(workspace) || ws_bytes < flags_bytes(tokens)) return CERE_ERR_ARGUMENT;
        }
        record_prefill(ctx, s.fp8, q, tokens, pos0, s.k, s.ks, s.v, s.vs, sl, scale, out, opt, workspace);
        return CERE_OK;
    }
    const Partials p = partials(workspace, tokens, opt);
    if (!workspace || !a16(workspace) || ws_bytes < p.bytes) return CERE_ERR_ARGUMENT;
    record_split(ctx, s.fp8, q, tokens, pos0, s.k, s.ks, s.v, s.vs, sl, scale, p, out, kQWidth, 0, 0, opt);
    return CERE_OK;
}

cere_status dense_attention(vk::Context &ctx, uint64_t q, uint64_t k, uint64_t v, int64_t v_stride, uint64_t gate,
                            int64_t tokens, int64_t prefix_len, bool kv_from_cache, uint64_t k_cache, uint64_t v_cache,
                            uint64_t slots, uint64_t workspace, size_t ws_bytes, uint64_t out, int64_t out_stride,
                            int64_t cls, const Options &opt) {
    Source s;
    if (!gate || !a16(gate) || out_stride < kQWidth || out_stride % 8 || !workspace || !a16(workspace) ||
        ws_bytes < workspace_bytes(tokens, opt) ||
        !check(q, tokens, prefix_len, kv_from_cache ? 0 : k, kKvWidth, kv_from_cache ? 0 : v, v_stride,
               kv_from_cache ? k_cache : 0, kv_from_cache ? v_cache : 0, out, cls, opt, s))
        return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    const uint64_t sl = s.fp8 ? slots : 0;
    const int64_t sel = cls != 0 ? cls : tokens;
    if (sel > kSplitMaxTokens) {
        if (coop(opt) && !ctx.info().coopmat) return CERE_ERR_UNSUPPORTED;
        const uint64_t attn = ws_attn(workspace, tokens, opt);
        record_prefill(ctx, s.fp8, q, tokens, prefix_len, s.k, s.ks, s.v, s.vs, sl, kSoftmaxScale, attn, opt,
                       ws_flags(workspace, tokens, opt));
        ctx.barrier();
        const vk::PipelineSpec gs = gate_spec(opt);
        const vk::Pipeline &gp = ctx.pipeline(gs.spv, gs.push_bytes, gs.required_subgroup, gs.spec);
        const GatePush gpc{attn, gate, out, uint32_t(tokens), uint32_t(out_stride), uint32_t(kQWidth), 0};
        ctx.dispatch(gp, &gpc, kQWidth / 256, uint32_t(tokens));
        return CERE_OK;
    }
    record_split(ctx, s.fp8, q, tokens, prefix_len, s.k, s.ks, s.v, s.vs, sl, kSoftmaxScale,
                 partials(workspace, tokens, opt), out, out_stride, gate, kQWidth, opt);
    return CERE_OK;
}

cere_status dense_attention_rows(vk::Context &ctx, uint64_t q, uint64_t gate, int64_t rows, int64_t pos0,
                                 uint64_t k_cache, uint64_t v_cache, uint64_t slots, uint64_t workspace,
                                 size_t ws_bytes, uint64_t out, int64_t out_stride, const Options &opt) {
    Source s;
    if (rows < 1 || rows > kSplitMaxTokens || !gate || !a16(gate) || out_stride < kQWidth || out_stride % 8 ||
        !workspace || !a16(workspace) || ws_bytes < workspace_bytes(rows, opt) ||
        !check(q, rows, pos0, 0, 0, 0, 0, k_cache, v_cache, out, 0, opt, s))
        return CERE_ERR_ARGUMENT;
    record_split(ctx, true, q, rows, pos0, s.k, s.ks, s.v, s.vs, slots, kSoftmaxScale, partials(workspace, rows, opt),
                 out, out_stride, gate, kQWidth, opt);
    return CERE_OK;
}

cere_status coopmat_probe(vk::Context &ctx, uint32_t out[4]) {
    if (!ctx.info().coopmat) return CERE_ERR_UNSUPPORTED;
    vk::Buffer b;
    cere_status st = ctx.allocate(256, b);
    if (st != CERE_OK) return st;
    std::memset(b.map, 0xFF, 256);
    const uint64_t pc = b.addr;
    const vk::Pipeline *p = nullptr;
    st = ctx.pipeline(vk::PipelineSpec{"p4att_probe.spv", sizeof(pc), 32, {}}, &p);
    if (st == CERE_OK) st = ctx.begin();
    if (st == CERE_OK) {
        ctx.dispatch(*p, &pc, 1);
        ctx.barrier();
        st = ctx.submit_wait();
    }
    if (st == CERE_OK) std::memcpy(out, b.map, 16);
    ctx.free(b);
    return st;
}

}  // namespace cere::vkk6
