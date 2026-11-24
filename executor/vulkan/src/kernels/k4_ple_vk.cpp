// VulkanAmd P2: see k4_ple_vk.h.
#include "kernels/k4_ple_vk.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"

namespace cere::vkk4 {
namespace {

struct HashPush { uint64_t windows, mult, vocab, offs, ids; int64_t eos; uint64_t outputs; };
struct GatherPush { uint64_t desc, ids, out, scale, rows, n; uint32_t rpb, pad; };
struct EmbedPush {
    uint64_t desc, segments, tokens, emb, mult, vocab, offs, scale;
    int64_t eos;
    uint64_t rows;
    uint32_t rpb, nseg, total, pad;
    uint64_t history;  // nseg == 1: the only segment's history
};
struct GatePush { uint64_t kn, qn, value, gated; uint32_t rows; float inv_sqrt_hidden; };
struct ConvPush { uint64_t gn, gated, conv_w, query, out, segments; uint32_t nseg, add_query; };
struct StatePush { uint64_t jobs, tokens, gn; };
static_assert(sizeof(HashPush) == 56 && sizeof(GatherPush) == 56 && sizeof(EmbedPush) == 104 && sizeof(GatePush) == 40 &&
                  sizeof(ConvPush) == 56 && sizeof(StatePush) == 24,
              "push layouts match the k4_*.comp blocks");

// The call table (k4_embed.comp / k4_conv.comp Segments, k4_state.comp Jobs).
struct Segment { uint64_t history, conv_state; uint32_t first, count, pad0, pad1; };
struct Job { uint64_t hist_src, hist_dst, conv_src, conv_dst; uint32_t first, r, pad0, pad1; };
static_assert(sizeof(Segment) == 32 && sizeof(Job) == 48, "call table entries");
constexpr uint32_t kSegOff = 0;
constexpr uint32_t kJobOff = kMaxRows * sizeof(Segment);  // 512
constexpr uint32_t kMaxJobs = 2 * kMaxRows + 1;           // saves + commits + a capture
static_assert(kJobOff + kMaxJobs * sizeof(Job) <= kCallTableBytes, "call table size");

constexpr uint32_t kChannelBlocks = kChannels / 128;  // conv / state workgroups per job or segment
constexpr int64_t kMaxTokens = 16383;                 // rows * 4 workgroups of the norms / gate stay <= 65,535

bool al(uint64_t a, uint64_t n) { return a % n == 0; }

// The format-dependent SPIR-V names: one selection for the dispatches and for pipeline_specs.
const char *embed_spv(bool nvfp4) { return nvfp4 ? "k4_embed_nvfp4.spv" : "k4_embed_fp8.spv"; }
const char *gather_spv(bool nvfp4) { return nvfp4 ? "k4_gather_nvfp4.spv" : "k4_gather_fp8.spv"; }
size_t align256(size_t b) { return (b + 255) & ~size_t(255); }

// 2-D grid for `n` workgroups (x <= 65,535; the shaders use wg_linear and drop the spill).
void grid2(uint64_t n, uint32_t &gx, uint32_t &gy) {
    gx = uint32_t(std::min<uint64_t>(n, 65535));
    gy = uint32_t((n + gx - 1) / gx);
}

bool view_ok(const TableView &v, TableFormat f) {
    if (v.desc == 0 || !al(v.desc, 8) || v.rows < 0 || v.rows >= (int64_t(1) << 32) || v.rows_per_block == 0 || v.blocks == 0 ||
        v.blocks > kMaxBlocks || uint64_t(v.blocks) * v.rows_per_block < uint64_t(v.rows))
        return false;
    // In-block byte offsets are 32-bit.
    return uint64_t(v.rows_per_block) * row_bytes(f) < (uint64_t(1) << 32) &&
           uint64_t(v.rows_per_block) * std::max<uint32_t>(scale_bytes(f), 1u) < (uint64_t(1) << 32);
}

// emb | key | value | qn | gated | the key and value K2 GEMM workspaces (k2_gemm_vk.h: one each, no barrier between).
struct Layout { uint64_t emb, key, value, qn, gated, gk, gv; size_t bytes; };
Layout layout(uint64_t base, int64_t t) {
    const size_t n = size_t(t);
    const size_t emb = align256(n * kEmbedDim * 2), wide = align256(n * kChannels * 2), narrow = align256(n * kHidden * 2);
    const size_t gk = align256(vkk2::gemm_workspace_bytes(uint32_t(t), kChannels)), gv = align256(vkk2::gemm_workspace_bytes(uint32_t(t), kHidden));
    const uint64_t g0 = base + emb + 3 * wide + narrow;
    return {base, base + emb, base + emb + wide, base + emb + wide + narrow, base + emb + 2 * wide + narrow, g0, g0 + gk,
            emb + 3 * wide + narrow + gk + gv};
}
constexpr uint64_t kValueInfoOff = (uint64_t(kChannels) * 4 + 15) / 16 * 16;  // proj_info: key's words, then value's

bool table_ok(const vk::Buffer &b, uint32_t off) {
    return b.map != nullptr && b.addr != 0 && al(off, 16) && uint64_t(off) + kCallTableBytes <= b.size;
}

void record_embed(vk::Context &ctx, const Weights &w, uint64_t segments, uint64_t history0, uint32_t nseg, uint32_t total, uint64_t tokens,
                  uint64_t emb) {
    const bool nv = w.format == TableFormat::kNvfp4;
    const EmbedPush pc{w.table.desc, segments, tokens, emb, w.multipliers, w.head_vocab_sizes, w.head_offsets,
                       nv ? w.global_scale : w.table_scale, w.eos_token_id, uint64_t(w.table.rows), w.table.rows_per_block,
                       nseg, total, 0, history0};
    uint32_t gx, gy;
    grid2(total, gx, gy);
    ctx.dispatch(ctx.pipeline(embed_spv(nv), sizeof pc), &pc, gx, gy);
}

// The stages after the embedding (k4_ple.cu gate_rows + the conv + the state jobs) over `total` rows; the call table at
// `tab` holds nseg segments and ns save jobs followed by nc commit jobs.
cere_status record_step(vk::Context &ctx, const Weights &w, uint64_t tab, uint64_t history0, uint32_t nseg, uint32_t max_count, uint32_t total,
                        uint64_t tokens, uint64_t query, bool add_query, uint64_t out, const Layout &ws, uint32_t ns, uint32_t nc, uint32_t cls) {
    record_embed(ctx, w, tab + kSegOff, history0, nseg, total, tokens, ws.emb);
    ctx.barrier();
    // key_proj / value_proj: K2 at row class cls (HIP k4_ple.cu gate_rows: plans.key / plans.value ->run(..., plans.cls)).
    cere_status s = vkk2::run(ctx, ws.emb, w.key_proj, 0, ws.key, total, kChannels, kEmbedDim, false, cls, ws.gk, w.proj_info);
    if (s != CERE_OK) return s;
    s = vkk2::run(ctx, ws.emb, w.value_proj, 0, ws.value, total, kHidden, kEmbedDim, false, cls, ws.gv,
                  w.proj_info ? w.proj_info + kValueInfoOff : 0);
    if (s != CERE_OK) return s;
    ctx.barrier();
    const int64_t rows = int64_t(total) * kHcCount;
    s = vkk1::rmsnorm(ctx, ws.key, kHidden, w.norm_key, kHcCount, true, w.eps, rows, kHidden, ws.key, kHidden);
    if (s != CERE_OK) return s;
    if ((s = vkk1::rmsnorm(ctx, query, kHidden, w.norm_query, kHcCount, true, w.eps, rows, kHidden, ws.qn, kHidden)) != CERE_OK) return s;
    ctx.barrier();
    // `gate / math.sqrt(hidden_size)` is a multiply by the FP32 reciprocal (PyTorch div_true_kernel_cuda), as k4_ple.cu.
    const GatePush gp{ws.key, ws.qn, ws.value, ws.gated, uint32_t(rows), 1.0f / float(std::sqrt(double(kHidden)))};
    uint32_t gx, gy;
    grid2(uint64_t(rows), gx, gy);
    ctx.dispatch(ctx.pipeline("k4_gate.spv", sizeof gp, 32), &gp, gx, gy);
    ctx.barrier();
    if ((s = vkk1::rmsnorm(ctx, ws.gated, kHidden, w.norm_conv, kHcCount, true, w.eps, rows, kHidden, ws.key, kHidden)) != CERE_OK) return s;
    ctx.barrier();
    const ConvPush cp{ws.key, ws.gated, w.conv_weight, query, out, tab + kSegOff, nseg, add_query ? 1u : 0u};
    ctx.dispatch(ctx.pipeline("k4_conv.spv", sizeof cp), &cp, kChannelBlocks, (max_count + 15) / 16, nseg);
    ctx.barrier();
    const vk::Pipeline &sp = ctx.pipeline("k4_state.spv", sizeof(StatePush));
    if (ns > 0) {
        const StatePush p{tab + kJobOff, tokens, ws.key};
        ctx.dispatch(sp, &p, kChannelBlocks, ns);
        ctx.barrier();
    }
    const StatePush p{tab + kJobOff + ns * uint32_t(sizeof(Job)), tokens, ws.key};
    ctx.dispatch(sp, &p, kChannelBlocks, nc);
    return CERE_OK;
}

bool weights_ok(const Weights &w) {
    const auto a16 = [](uint64_t p) { return p != 0 && p % 16 == 0; };
    return check_table(w) == CERE_OK && w.multipliers && al(w.multipliers, 8) && w.head_vocab_sizes && al(w.head_vocab_sizes, 8) &&
           w.head_offsets && al(w.head_offsets, 8) && a16(w.key_proj) && a16(w.value_proj) && a16(w.norm_key) && a16(w.norm_query) &&
           a16(w.norm_conv) && w.conv_weight && al(w.conv_weight, 8) && al(w.proj_info, 16);
}

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(bool fp8_table, bool nvfp4_table, bool gemm) {
    std::vector<vk::PipelineSpec> s = {{"k4_hash.spv", sizeof(HashPush), 0, {}}, {"k4_gate.spv", sizeof(GatePush), 32, {}},
                                       {"k4_conv.spv", sizeof(ConvPush), 0, {}}, {"k4_state.spv", sizeof(StatePush), 0, {}}};
    for (int nv = 0; nv < 2; ++nv) {
        if (!(nv ? nvfp4_table : fp8_table)) continue;
        s.push_back({embed_spv(nv != 0), sizeof(EmbedPush), 0, {}});
        s.push_back({gather_spv(nv != 0), sizeof(GatherPush), 0, {}});
    }
    // record_step: K1's gemma rmsnorm over kHidden (key, query and conv norms), K2's key / value projections (k = kEmbedDim).
    vkk2::append_unique(s, {vkk1::rmsnorm_spec(false, kHidden)});
    vkk2::append_unique(s, vkk2::pipeline_specs({uint32_t(kEmbedDim)}, {}, false));
    if (gemm) vkk2::append_unique(s, vkk2::pipeline_specs(true, false));  // row classes >= 17 (forward cls / t > 16)
    return s;
}

bool alloc_host_heap(vk::Context &ctx, uint64_t bytes, vk::Buffer &out, uint32_t *memory_type, std::string *why) {
    // The launch layer's host-heap allocation (host-visible + coherent, not device-local; exact flags first).
    if (ctx.allocate(bytes, out, true, why) != CERE_OK) return false;
    if (memory_type) *memory_type = out.memory_type;
    return true;
}

bool SplitTable::create(vk::Context &ctx, TableFormat format, int64_t rows, uint64_t max_block_bytes, Memory memory, std::string *why) {
    destroy(ctx);
    const auto fail = [&](const std::string &m) { if (why) *why = m; destroy(ctx); return false; };
    format_ = format;
    memory_ = memory;
    const uint64_t rb = row_bytes(format), sb = scale_bytes(format);
    const uint64_t limit = std::min(ctx.info().max_buffer, ctx.info().max_alloc);
    max_block_bytes = std::min(max_block_bytes, limit);
    if (rows <= 0 || rows >= (int64_t(1) << 32)) return fail("table rows outside [1, 2^32)");
    if (max_block_bytes < 512 + rb + sb) return fail("block too small for one row");
    // Rows per block: packed rows, a 256-byte aligned gap, then (NVFP4) the rows' block scales.
    uint64_t rpb = (max_block_bytes - 256) / (rb + sb);
    rpb = std::min<uint64_t>(rpb, uint64_t(rows));
    rpb = std::min<uint64_t>(rpb, ((uint64_t(1) << 32) - 1) / rb);
    const uint64_t blocks = (uint64_t(rows) + rpb - 1) / rpb;
    if (blocks > kMaxBlocks) return fail("table needs " + std::to_string(blocks) + " blocks > kMaxBlocks");
    scale_off_ = sb ? align256(rpb * rb) : 0;
    view_.rows = rows;
    view_.rows_per_block = uint32_t(rpb);
    view_.blocks = uint32_t(blocks);
    if (!ctx.alloc(kBlockTableBytes, desc_)) return fail("block table allocation refused");
    std::memset(desc_.map, 0, kBlockTableBytes);
    uint64_t *bases = static_cast<uint64_t *>(desc_.map);
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint64_t n = std::min<uint64_t>(rpb, uint64_t(rows) - b * rpb);
        const uint64_t bytes = sb ? scale_off_ + n * sb : n * rb;
        vk::Buffer buf;
        std::string w;
        const bool ok = memory == Memory::kHostHeap ? alloc_host_heap(ctx, bytes, buf, &memory_type_, &w) : ctx.allocate(bytes, buf, false, &w) == CERE_OK;
        if (!ok) return fail("block " + std::to_string(b) + " (" + std::to_string(bytes) + " B): " + (w.empty() ? "Context::alloc refused" : w));
        blocks_.push_back(buf);
        bases[b] = buf.addr;
        bases[kMaxBlocks + b] = sb ? buf.addr + scale_off_ : 0;
    }
    view_.desc = desc_.addr;
    return true;
}

void SplitTable::destroy(vk::Context &ctx) {
    for (auto &b : blocks_) ctx.free(b);
    blocks_.clear();
    ctx.free(desc_);
    view_ = TableView();
}

uint8_t *SplitTable::row(int64_t r) const {
    const uint64_t b = uint64_t(r) / view_.rows_per_block;
    return static_cast<uint8_t *>(blocks_[b].map) + (uint64_t(r) - b * view_.rows_per_block) * row_bytes(format_);
}

uint8_t *SplitTable::scales(int64_t r) const {
    const uint64_t b = uint64_t(r) / view_.rows_per_block;
    return static_cast<uint8_t *>(blocks_[b].map) + scale_off_ + (uint64_t(r) - b * view_.rows_per_block) * kNvfp4RowScales;
}

cere_status ngram_hash_windows(vk::Context &ctx, uint64_t windows, int64_t n, uint64_t multipliers, uint64_t head_vocab_sizes,
                               uint64_t head_offsets, int64_t eos_token_id, uint64_t ids) {
    if (n < 0 || n > int64_t(INT32_MAX)) return CERE_ERR_ARGUMENT;
    if (n == 0) return CERE_OK;
    if (!windows || !multipliers || !head_vocab_sizes || !head_offsets || !ids || !al(windows, 8) || !al(multipliers, 8) ||
        !al(head_vocab_sizes, 8) || !al(head_offsets, 8) || !al(ids, 8))
        return CERE_ERR_ARGUMENT;
    const HashPush pc{windows, multipliers, head_vocab_sizes, head_offsets, ids, eos_token_id, uint64_t(n) * kNgramHeads};
    uint32_t gx, gy;
    grid2((pc.outputs + 255) / 256, gx, gy);
    ctx.dispatch(ctx.pipeline("k4_hash.spv", sizeof pc), &pc, gx, gy);
    return CERE_OK;
}

static cere_status gather_any(vk::Context &ctx, TableFormat f, const TableView &t, uint64_t scale, uint64_t ids, int64_t n, uint64_t out) {
    if (n < 0 || n > int64_t(INT32_MAX) || !view_ok(t, f)) return CERE_ERR_ARGUMENT;
    if (f == TableFormat::kNvfp4 && (scale == 0 || !al(scale, 4))) return CERE_ERR_ARGUMENT;
    if (scale && !al(scale, 2)) return CERE_ERR_ARGUMENT;
    if (n == 0) return CERE_OK;
    if (!ids || !out || !al(ids, 8) || !al(out, 16)) return CERE_ERR_ARGUMENT;
    const GatherPush pc{t.desc, ids, out, scale, uint64_t(t.rows), uint64_t(n), t.rows_per_block, 0};
    uint32_t gx, gy;
    grid2((uint64_t(n) + 15) / 16, gx, gy);
    ctx.dispatch(ctx.pipeline(gather_spv(f == TableFormat::kNvfp4), sizeof pc), &pc, gx, gy);
    return CERE_OK;
}

cere_status gather(vk::Context &ctx, const TableView &table, uint64_t ids, int64_t n, uint64_t scale, uint64_t out) {
    return gather_any(ctx, TableFormat::kFp8, table, scale, ids, n, out);
}

cere_status gather_nvfp4(vk::Context &ctx, const TableView &table, uint64_t global_scale, uint64_t ids, int64_t n, uint64_t out) {
    return gather_any(ctx, TableFormat::kNvfp4, table, global_scale, ids, n, out);
}

cere_status check_table(const Weights &w) {
    if (!view_ok(w.table, w.format)) return CERE_ERR_ARGUMENT;
    switch (w.format) {
        case TableFormat::kFp8:
            return w.global_scale == 0 && (w.table_scale == 0 || al(w.table_scale, 2)) ? CERE_OK : CERE_ERR_ARGUMENT;
        case TableFormat::kNvfp4:
            return w.table_scale == 0 && w.global_scale != 0 && al(w.global_scale, 4) ? CERE_OK : CERE_ERR_ARGUMENT;
    }
    return CERE_ERR_ARGUMENT;
}

cere_status embed(vk::Context &ctx, const Weights &w, uint64_t history, uint64_t tokens, int64_t t, uint64_t emb, vk::Buffer &call_table,
                  uint32_t table_off) {
    if (t <= 0 || t > kMaxTokens || check_table(w) != CERE_OK || !history || !tokens || !emb || !al(history, 8) || !al(tokens, 8) ||
        !al(emb, 16) || !w.multipliers || !w.head_vocab_sizes || !w.head_offsets || !table_ok(call_table, table_off))
        return CERE_ERR_ARGUMENT;
    Segment seg{history, 0, 0, uint32_t(t), 0, 0};
    std::memcpy(static_cast<uint8_t *>(call_table.map) + table_off + kSegOff, &seg, sizeof seg);
    record_embed(ctx, w, call_table.addr + table_off + kSegOff, history, 1, uint32_t(t), tokens, emb);
    return CERE_OK;
}

size_t workspace_bytes(int64_t max_tokens) { return max_tokens <= 0 ? 0 : layout(0, max_tokens).bytes; }

size_t proj_info_bytes() { return size_t(kValueInfoOff + (uint64_t(kHidden) * 4 + 15) / 16 * 16); }

cere_status prepare_projections(vk::Context &ctx, const Weights &w) {
    if (!w.proj_info || !al(w.proj_info, 16) || !w.key_proj || !al(w.key_proj, 16) || !w.value_proj || !al(w.value_proj, 16))
        return CERE_ERR_ARGUMENT;
    const cere_status s = vkk2::gemm_prepare_weight(ctx, w.key_proj, 0, kChannels, kEmbedDim, w.proj_info);
    if (s != CERE_OK) return s;
    return vkk2::gemm_prepare_weight(ctx, w.value_proj, 0, kHidden, kEmbedDim, w.proj_info + kValueInfoOff);
}

cere_status forward(vk::Context &ctx, const Weights &w, const State &state, uint64_t tokens, int64_t t, uint64_t query, bool add_query,
                    uint64_t out, uint64_t workspace, size_t workspace_size, vk::Buffer &call_table, uint32_t table_off,
                    const Capture *capture, uint32_t cls) {
    // Every check precedes the first recorded command (k4::forward's rule).
    if (t <= 0 || t > kMaxTokens || !weights_ok(w) || !tokens || !al(tokens, 8) || !query || !al(query, 16) || !out || !al(out, 2) ||
        !workspace || !al(workspace, 256) || workspace_size < workspace_bytes(t) || !state.history || !al(state.history, 8) ||
        !state.conv_state || !al(state.conv_state, 2) || !table_ok(call_table, table_off) || (cls != 0 && int64_t(cls) < t))
        return CERE_ERR_ARGUMENT;
    // A class >= 17 runs the K2 GEMM, which needs cooperative matrices (and vkk2::gemm_probe: not visible here; an
    // unprobed context fails at the projection, after the embedding was recorded).
    if ((cls != 0 ? int64_t(cls) : t) >= int64_t(vkk2::kGemmMinM) && !vkk2::gemm_ready(ctx)) return CERE_ERR_UNSUPPORTED;
    if (capture && (capture->row < kConvStateLen || capture->row >= t || !capture->history || !al(capture->history, 8) ||
                    !capture->conv_state || !al(capture->conv_state, 2)))
        return CERE_ERR_ARGUMENT;
    uint8_t *tab = static_cast<uint8_t *>(call_table.map) + table_off;
    const Segment seg{state.history, state.conv_state, 0, uint32_t(t), 0, 0};
    std::memcpy(tab + kSegOff, &seg, sizeof seg);
    uint32_t ns = 0;
    if (capture) {
        const Job j{state.history, capture->history, state.conv_state, capture->conv_state, 0, uint32_t(capture->row), 0, 0};
        std::memcpy(tab + kJobOff, &j, sizeof j);
        ns = 1;
    }
    const Job commit{state.history, state.history, state.conv_state, state.conv_state, 0, uint32_t(t), 0, 0};
    std::memcpy(tab + kJobOff + ns * sizeof(Job), &commit, sizeof commit);
    return record_step(ctx, w, call_table.addr + table_off, state.history, 1, uint32_t(t), uint32_t(t), tokens, query, add_query, out,
                       layout(workspace, t), ns, 1, cls);
}

cere_status forward_rows(vk::Context &ctx, const Weights &w, const RowsCall &call, uint64_t tokens, int64_t m, uint64_t query, bool add_query,
                         uint64_t out, uint64_t workspace, size_t workspace_size, vk::Buffer &call_table, uint32_t table_off) {
    if (m <= 0 || m > int64_t(kMaxRows) || !weights_ok(w) || !tokens || !al(tokens, 8) || !query || !al(query, 16) || !out || !al(out, 2) ||
        !workspace || !al(workspace, 256) || workspace_size < workspace_bytes(m) || !table_ok(call_table, table_off))
        return CERE_ERR_ARGUMENT;
    if (call.sequences == 0 || call.sequences > uint32_t(m)) return CERE_ERR_ARGUMENT;
    uint32_t seq_of[kMaxRows], j_of[kMaxRows], next = 0, max_count = 0;
    for (uint32_t q = 0; q < call.sequences; ++q) {
        const State &st = call.states[q];
        if (call.first[q] != next || call.count[q] == 0 || call.count[q] > uint32_t(m) - next || !st.history || !al(st.history, 8) ||
            !st.conv_state || !al(st.conv_state, 2))
            return CERE_ERR_ARGUMENT;
        for (uint32_t j = 0; j < call.count[q]; ++j) { seq_of[next + j] = q; j_of[next + j] = j; }
        next += call.count[q];
        max_count = std::max(max_count, call.count[q]);
    }
    if (next != uint32_t(m)) return CERE_ERR_ARGUMENT;
    for (uint32_t r = 0; r < uint32_t(m); ++r) {
        const State &sv = call.saves[r];
        if ((sv.history == 0) != (sv.conv_state == 0) || (sv.history && (!al(sv.history, 8) || !al(sv.conv_state, 2)))) return CERE_ERR_ARGUMENT;
    }
    uint8_t *tab = static_cast<uint8_t *>(call_table.map) + table_off;
    for (uint32_t q = 0; q < call.sequences; ++q) {
        const Segment seg{call.states[q].history, call.states[q].conv_state, call.first[q], call.count[q], 0, 0};
        std::memcpy(tab + kSegOff + q * sizeof(Segment), &seg, sizeof seg);
    }
    uint32_t ns = 0;
    for (uint32_t r = 0; r < uint32_t(m); ++r) {
        if (!call.saves[r].history) continue;
        const State &st = call.states[seq_of[r]];
        const Job j{st.history, call.saves[r].history, st.conv_state, call.saves[r].conv_state, call.first[seq_of[r]], j_of[r] + 1, 0, 0};
        std::memcpy(tab + kJobOff + ns * sizeof(Job), &j, sizeof j);
        ++ns;
    }
    for (uint32_t q = 0; q < call.sequences; ++q) {
        const State &st = call.states[q];
        const Job j{st.history, st.history, st.conv_state, st.conv_state, call.first[q], call.count[q], 0, 0};
        std::memcpy(tab + kJobOff + (ns + q) * sizeof(Job), &j, sizeof j);
    }
    return record_step(ctx, w, call_table.addr + table_off, call.states[0].history, call.sequences, max_count, uint32_t(m), tokens, query, add_query, out,
                       layout(workspace, m), ns, call.sequences, 0);  // HIP: rows plans, class <= 16 (the GEMV)
}

}  // namespace cere::vkk4
