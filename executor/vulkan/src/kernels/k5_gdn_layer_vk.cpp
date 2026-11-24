// VulkanAmd P2: see k5_gdn_layer_vk.h. Order of operations as executor/cuda/src/kernels/k5_gdn_layer.cu.
#include "kernels/k5_gdn_layer_vk.h"

#include <algorithm>

#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"
#include "kernels/k5_gdn_chunk_vk.h"
#include "kernels/k5_gdn_vk.h"

namespace cere::vkk5 {
namespace {

constexpr uint64_t kAlign = 256;
constexpr float kScale = 0.08838834764831845f;  // 128^-0.5
constexpr int64_t kNormRowsPerCall = 65535;      // gated-norm rows per dispatch (the portable group-count minimum)

uint64_t round_up(uint64_t b) { return (b + kAlign - 1) / kAlign * kAlign; }

class Carver {
public:
    explicit Carver(uint64_t base) : base_(base) {}
    uint64_t take(uint64_t bytes) { const uint64_t r = base_ + off_; off_ += round_up(bytes); return r; }
    uint64_t bytes() const { return off_; }
private:
    uint64_t base_;
    uint64_t off_ = 0;
};

struct CoreWorkspace {
    uint64_t conv_out;  // [T, 10240]: PREFILL q [T, 2048] | k [T, 2048] | v [T, 6144] planes; DECODE one q | k | v row
    uint64_t o;         // [T, 6144]
    uint64_t g, beta;   // f32 [T, 48]
    uint64_t chunk;     // chunk_workspace_bytes(T)
    uint64_t bytes;
};

CoreWorkspace carve_core(Carver &c, int64_t tokens) {
    const uint64_t t = uint64_t(tokens);
    CoreWorkspace ws{};
    const uint64_t start = c.bytes();
    ws.conv_out = c.take(t * kConvDim * 2);
    ws.o = c.take(t * kValueDim * 2);
    ws.g = c.take(t * kValueHeads * 4);
    ws.beta = c.take(t * kValueHeads * 4);
    ws.chunk = c.take(chunk_workspace_bytes(tokens));
    ws.bytes = c.bytes() - start;
    return ws;
}

constexpr uint32_t kMaxHidden = 65536;  // layer_forward's hidden_size limit (the out_proj GEMM workspace is sized for it)

struct LayerWorkspace {
    uint64_t mixed;  // [T, 10240], then y [T, 6144]
    uint64_t z;      // [T, 6144]
    uint64_t b, a;   // [T, 48]
    uint64_t g_qkv, g_z, g_b, g_a;  // K2 GEMM workspaces (g_qkv sized for n = kMaxHidden: reused by out_proj)
    uint64_t core;
    uint64_t core_bytes;
    uint64_t bytes;
};

LayerWorkspace carve_layer(uint64_t base, int64_t tokens) {
    const uint64_t t = uint64_t(tokens);
    Carver c(base);
    LayerWorkspace ws{};
    ws.mixed = c.take(t * kConvDim * 2);
    ws.z = c.take(t * kValueDim * 2);
    ws.b = c.take(t * kValueHeads * 2);
    ws.a = c.take(t * kValueHeads * 2);
    ws.g_qkv = c.take(vkk2::gemm_workspace_bytes(uint32_t(t), std::max<uint32_t>(kConvDim, kMaxHidden)));
    ws.g_z = c.take(vkk2::gemm_workspace_bytes(uint32_t(t), kValueDim));
    ws.g_b = c.take(vkk2::gemm_workspace_bytes(uint32_t(t), kValueHeads));
    ws.g_a = c.take(vkk2::gemm_workspace_bytes(uint32_t(t), kValueHeads));
    ws.core = c.take(0);
    ws.core_bytes = carve_core(c, tokens).bytes;
    ws.bytes = c.bytes();
    return ws;
}

bool tokens_valid(int64_t tokens) { return tokens >= 1 && tokens <= kChunkMaxTokens; }

bool end_valid(const EndState &end, uint32_t operation, int64_t tokens) {
    if (operation != CERE_OP_PREFILL) return end.ssm_in32 == 0 && end.row < 0;
    if (end.row < 0) return end.ssm_out32 == 0 && end.conv_out == 0;
    return end.ssm_out32 != 0 && end.conv_out != 0 && end.row >= kChunkTokens && end.row < tokens && end.row % kChunkTokens == 0;
}

cere_status core_prefill(vk::Context &ctx, const LayerWeights &w, LayerState state, uint64_t mixed, uint64_t b, uint64_t a,
                         int64_t tokens, const CoreWorkspace &ws, const EndState *end) {
    if (end != nullptr && end->row >= 0) {  // the conv state as of `row`, read from the raw projections
        if (cere_status st = conv1d_state_at(ctx, mixed, kConvDim, end->conv_out, end->row, kConvDim); st != CERE_OK) return st;
    }
    const int64_t offsets[] = {0, kKeyDim, 2 * kKeyDim};
    const int64_t widths[] = {kKeyDim, kKeyDim, kValueDim};
    uint64_t planes[3];
    uint64_t at = ws.conv_out;
    for (int i = 0; i < 3; ++i) {
        planes[i] = at;
        const cere_status st = conv1d_silu(ctx, mixed + uint64_t(offsets[i]) * 2, kConvDim, w.conv1d + uint64_t(offsets[i]) * kConvWidth * 2,
                                           state.conv + uint64_t(offsets[i]) * kConvState * 2, at, widths[i], tokens, widths[i]);
        if (st != CERE_OK) return st;
        at += uint64_t(tokens) * uint64_t(widths[i]) * 2;
    }
    if (cere_status st = gating(ctx, a, kValueHeads, b, kValueHeads, w.A_log, w.dt_bias, ws.g, ws.beta, tokens, kValueHeads); st != CERE_OK)
        return st;
    ctx.barrier();
    ChunkArgs ca{};
    ca.q = planes[0];
    ca.k = planes[1];
    ca.v = planes[2];
    ca.g = ws.g;
    ca.beta = ws.beta;
    ca.state = state.ssm;
    ca.o = ws.o;
    ca.tokens = tokens;
    ca.scale = kScale;
    if (end != nullptr) {
        ca.state_in32 = end->ssm_in32;
        ca.capture32 = end->ssm_out32;
        ca.capture_chunk = end->row >= 0 ? end->row / kChunkTokens : -1;
        ca.state_in32_hi = end->ssm_in32_hi;
        ca.capture32_hi = end->ssm_out32_hi;
    }
    return delta_rule_chunk(ctx, ca, ws.chunk);
}

cere_status core_decode(vk::Context &ctx, const LayerWeights &w, LayerState state, uint64_t mixed, uint64_t b, uint64_t a,
                        const CoreWorkspace &ws) {
    if (cere_status st = conv1d_silu(ctx, mixed, kConvDim, w.conv1d, state.conv, ws.conv_out, kConvDim, 1, kConvDim); st != CERE_OK)
        return st;
    ctx.barrier();
    DecodeArgs da{};
    da.q = ws.conv_out;
    da.k = ws.conv_out + uint64_t(kKeyDim) * 2;
    da.v = ws.conv_out + uint64_t(2 * kKeyDim) * 2;
    da.a = a;
    da.b = b;
    da.A_log = w.A_log;
    da.dt_bias = w.dt_bias;
    da.state = state.ssm;
    da.o = ws.o;
    da.scale = kScale;
    return delta_rule_decode(ctx, da);
}

// LayerWeights::info parts (16-byte aligned): qkv, z, b, a, out.
struct Info { uint64_t qkv, z, b, a, out, bytes; };
uint64_t round16(uint64_t b) { return (b + 15) / 16 * 16; }
Info info_parts(uint64_t base, int64_t hidden_size) {
    Info i{};
    i.qkv = base;
    i.z = i.qkv + round16(uint64_t(kConvDim) * 4);
    i.b = i.z + round16(uint64_t(kValueDim) * 4);
    i.a = i.b + round16(uint64_t(kValueHeads) * 4);
    i.out = i.a + round16(uint64_t(kValueHeads) * 4);
    i.bytes = i.out + round16(uint64_t(hidden_size) * 4) - base;
    return i;
}

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<uint32_t> &hidden_sizes, bool gemm, bool mx) {
    // layer_core: prefill = conv1d_silu (+ conv_state), gating, delta_rule_chunk (its kernels + l2norm); decode = conv1d_silu,
    // delta_rule_decode; then the gated norm (K1, width kHeadDim). Not the *_rows pieces (the chain's).
    std::vector<vk::PipelineSpec> s = piece_pipeline_specs(kPieceDecode | kPieceConvGating);
    vkk2::append_unique(s, chunk_pipeline_specs());
    vkk2::append_unique(s, {vkk1::rmsnorm_spec(true, kHeadDim)});
    // layer_forward's five K2 products: k = hidden_size (qkv, z, b, a) and k = kValueDim (out_proj), BF16.
    std::vector<uint32_t> ks = {uint32_t(kValueDim)};
    ks.insert(ks.end(), hidden_sizes.begin(), hidden_sizes.end());
    vkk2::append_unique(s, vkk2::pipeline_specs(ks, mx ? ks : std::vector<uint32_t>{}, false));  // MXFP8 projections: the same ks
    if (gemm) vkk2::append_unique(s, vkk2::pipeline_specs(true, mx));  // row classes >= 17 (prefill) and prepare_weights
    return s;
}

size_t core_workspace_bytes(int64_t max_tokens) {
    if (!tokens_valid(max_tokens)) return 0;
    Carver c(0);
    return size_t(carve_core(c, max_tokens).bytes);
}

size_t layer_workspace_bytes(int64_t max_tokens) { return tokens_valid(max_tokens) ? size_t(carve_layer(0, max_tokens).bytes) : 0; }

size_t weight_info_bytes(int64_t hidden_size) {
    return hidden_size >= 16 && hidden_size % 16 == 0 && hidden_size <= kMaxHidden ? size_t(info_parts(0, hidden_size).bytes) : 0;
}

cere_status prepare_weights(vk::Context &ctx, const LayerWeights &w, int64_t hidden_size) {
    const auto a16 = [](uint64_t p) { return p != 0 && p % 16 == 0; };
    if (!a16(w.info) || !a16(w.in_proj_qkv) || !a16(w.in_proj_z) || !a16(w.in_proj_b) || !a16(w.in_proj_a) || !a16(w.out_proj) ||
        hidden_size < 64 || hidden_size % 64 != 0 || hidden_size > kMaxHidden)
        return CERE_ERR_ARGUMENT;
    const Info i = info_parts(w.info, hidden_size);
    const uint32_t h = uint32_t(hidden_size);
    cere_status s;
    if ((s = vkk2::gemm_prepare_weight(ctx, w.in_proj_qkv, w.in_proj_qkv_scales, kConvDim, h, i.qkv)) != CERE_OK) return s;
    if ((s = vkk2::gemm_prepare_weight(ctx, w.in_proj_z, w.in_proj_z_scales, kValueDim, h, i.z)) != CERE_OK) return s;
    if ((s = vkk2::gemm_prepare_weight(ctx, w.in_proj_b, w.in_proj_b_scales, kValueHeads, h, i.b)) != CERE_OK) return s;
    if ((s = vkk2::gemm_prepare_weight(ctx, w.in_proj_a, w.in_proj_a_scales, kValueHeads, h, i.a)) != CERE_OK) return s;
    return vkk2::gemm_prepare_weight(ctx, w.out_proj, w.out_proj_scales, h, kValueDim, i.out);
}

cere_status layer_core(vk::Context &ctx, const LayerWeights &w, LayerState state, uint32_t operation, uint64_t mixed,
                       uint64_t z, uint64_t b, uint64_t a, int64_t tokens, uint64_t y, uint64_t workspace,
                       size_t workspace_bytes, const EndState *end) {
    if (operation == CERE_OP_VERIFY) return CERE_ERR_UNSUPPORTED;
    if ((operation != CERE_OP_PREFILL && operation != CERE_OP_DECODE) || mixed == 0 || z == 0 || b == 0 || a == 0 || y == 0 ||
        workspace == 0 || state.conv == 0 || state.ssm == 0 || w.conv1d == 0 || w.A_log == 0 || w.dt_bias == 0 || w.norm == 0 ||
        !tokens_valid(tokens) || (operation == CERE_OP_DECODE && tokens != 1) || workspace % kAlign != 0 ||
        (end != nullptr && !end_valid(*end, operation, tokens)))
        return CERE_ERR_ARGUMENT;
    Carver c(workspace);
    const CoreWorkspace ws = carve_core(c, tokens);
    if (ws.bytes > workspace_bytes) return CERE_ERR_ARGUMENT;
    const cere_status st = operation == CERE_OP_PREFILL ? core_prefill(ctx, w, state, mixed, b, a, tokens, ws, end)
                                                        : core_decode(ctx, w, state, mixed, b, a, ws);
    if (st != CERE_OK) return st;
    ctx.barrier();
    const int64_t rows = tokens * kValueHeads;
    for (int64_t r0 = 0; r0 < rows; r0 += kNormRowsPerCall) {
        const int64_t n = std::min<int64_t>(kNormRowsPerCall, rows - r0);
        const uint64_t off = uint64_t(r0) * kHeadDim * 2;
        if (cere_status s = vkk1::rmsnorm_gated_sigmoid(ctx, ws.o + off, kHeadDim, z + off, kHeadDim, w.norm, kNormEps, n, kHeadDim,
                                                        y + off, kHeadDim);
            s != CERE_OK)
            return s;
    }
    return CERE_OK;
}

cere_status layer_forward(vk::Context &ctx, const LayerWeights &w, LayerState state, uint32_t operation, uint64_t hidden,
                          int64_t tokens, int64_t hidden_size, uint64_t out, uint64_t workspace, size_t workspace_bytes,
                          uint32_t cls, const EndState *end) {
    if (operation == CERE_OP_VERIFY) return CERE_ERR_UNSUPPORTED;
    if (hidden == 0 || out == 0 || workspace == 0 || w.in_proj_qkv == 0 || w.in_proj_z == 0 || w.in_proj_b == 0 ||
        w.in_proj_a == 0 || w.out_proj == 0 || (operation != CERE_OP_PREFILL && operation != CERE_OP_DECODE) ||
        (operation == CERE_OP_DECODE && tokens != 1) || !tokens_valid(tokens) || hidden_size < 16 || hidden_size % 16 != 0 ||
        hidden_size > int64_t(kMaxHidden) || workspace % kAlign != 0 || (cls != 0 && int64_t(cls) < tokens) || w.info % 16 != 0)
        return CERE_ERR_ARGUMENT;
    // An MXFP8 in-projection needs k = hidden_size % 32 (K2's MXFP8 GEMV rule): refused before anything is recorded.
    if ((w.in_proj_qkv_scales || w.in_proj_z_scales || w.in_proj_b_scales || w.in_proj_a_scales) && hidden_size % 32 != 0)
        return CERE_ERR_ARGUMENT;
    const LayerWorkspace ws = carve_layer(workspace, tokens);
    if (ws.bytes > workspace_bytes) return CERE_ERR_ARGUMENT;
    // The K2 GEMM (row class >= 17) needs k % 64 (hidden_size for the in-projections) and cooperative matrices: refused
    // before anything is recorded (an unprobed context included: vkk2::gemm_ready).
    if ((cls != 0 ? int64_t(cls) : tokens) >= int64_t(vkk2::kGemmMinM) && (hidden_size % 64 != 0 || !vkk2::gemm_ready(ctx)))
        return CERE_ERR_UNSUPPORTED;
    const Info inf = info_parts(w.info, hidden_size);
    const uint32_t m = uint32_t(tokens), h = uint32_t(hidden_size);
    auto k2 = [&](uint64_t x, uint64_t wt, uint64_t sc, uint64_t y, uint32_t n, uint32_t k, uint64_t gws, uint64_t winfo) {
        return vkk2::run(ctx, x, wt, sc, y, m, n, k, false, cls, gws, w.info ? winfo : 0);
    };
    cere_status st = k2(hidden, w.in_proj_qkv, w.in_proj_qkv_scales, ws.mixed, kConvDim, h, ws.g_qkv, inf.qkv);
    if (st == CERE_OK) st = k2(hidden, w.in_proj_z, w.in_proj_z_scales, ws.z, kValueDim, h, ws.g_z, inf.z);
    if (st == CERE_OK) st = k2(hidden, w.in_proj_b, w.in_proj_b_scales, ws.b, kValueHeads, h, ws.g_b, inf.b);
    if (st == CERE_OK) st = k2(hidden, w.in_proj_a, w.in_proj_a_scales, ws.a, kValueHeads, h, ws.g_a, inf.a);
    if (st != CERE_OK) return st;
    ctx.barrier();
    st = layer_core(ctx, w, state, operation, ws.mixed, ws.z, ws.b, ws.a, tokens, ws.mixed, ws.core, size_t(ws.core_bytes), end);
    if (st != CERE_OK) return st;
    ctx.barrier();
    return k2(ws.mixed, w.out_proj, w.out_proj_scales, out, h, kValueDim, ws.g_qkv, inf.out);  // g_qkv: free after the barriers above
}

}  // namespace cere::vkk5
