// VulkanAmd P3 (T3): see k5_gdn_chain_vk.h. Order of operations and argument checks as
// executor/cuda/src/kernels/k5_gdn_layer.cu:251-316.
#include "kernels/k5_gdn_chain_vk.h"

#include "kernels/k1_primitives_vk.h"

namespace cere::vkk5 {
namespace {

constexpr uint64_t kAlign = 256;
constexpr float kScale = 0.08838834764831845f;  // 128^-0.5 (k5_gdn_layer_vk.cpp, k5_gdn_layer.cu kScale)

uint64_t round_up(uint64_t b) { return (b + kAlign - 1) / kAlign * kAlign; }
bool al(uint64_t a, uint64_t n) { return a != 0 && a % n == 0; }

struct ChainWorkspace {
    uint64_t conv_out;  // [rows, 10240]: row t's q | k | v
    uint64_t o;         // [rows, 6144]
    uint64_t bytes;
};

ChainWorkspace carve_chain(uint64_t base, int64_t rows) {
    const uint64_t t = uint64_t(rows);
    ChainWorkspace ws{};
    ws.conv_out = base;
    ws.o = base + round_up(t * kConvDim * 2);
    ws.bytes = round_up(t * kConvDim * 2) + round_up(t * kValueDim * 2);
    return ws;
}

bool chain_rows_valid(int64_t rows) { return rows >= 1 && rows <= kChainMaxRows; }

bool shares(const LayerState &s, const LayerState &t) {
    return s.conv == t.conv || s.ssm == t.ssm || s.conv == t.ssm || s.ssm == t.conv;
}

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs() {
    // chain_core: conv1d_silu_rows, delta_rule_decode_rows, then K1's gated rmsnorm of width kHeadDim.
    std::vector<vk::PipelineSpec> s = piece_pipeline_specs(kPieceRows);
    s.push_back(vkk1::rmsnorm_spec(true, kHeadDim));
    return s;
}

size_t chain_core_workspace_bytes(int64_t rows) { return chain_rows_valid(rows) ? size_t(carve_chain(0, rows).bytes) : 0; }

cere_status chain_core(vk::Context &ctx, const LayerWeights &w, const Chains &chains, uint64_t mixed, uint64_t z, uint64_t b,
                       uint64_t a, int64_t rows, uint64_t y, uint64_t workspace, size_t workspace_bytes, const vk::Buffer &table,
                       uint64_t table_off) {
    if (mixed == 0 || z == 0 || b == 0 || a == 0 || y == 0 || workspace == 0 || w.conv1d == 0 || w.A_log == 0 || w.dt_bias == 0 ||
        w.norm == 0 || !chain_rows_valid(rows) || chains.count < 1 || chains.count > kChainMaxRows || workspace % kAlign != 0)
        return CERE_ERR_ARGUMENT;
    // Vulkan alignment rules of the three steps (k5_gdn_vk.cpp inputs_ok / conv1d_silu_rows, k1 rmsnorm_gated_sigmoid).
    if (!al(mixed, 2) || !al(b, 2) || !al(a, 2) || !al(z, 16) || !al(y, 16) || !al(w.conv1d, 2) || !al(w.A_log, 4) ||
        !al(w.dt_bias, 2) || !al(w.norm, 16))
        return CERE_ERR_ARGUMENT;
    const ChainWorkspace ws = carve_chain(workspace, rows);
    if (ws.bytes > workspace_bytes) return CERE_ERR_ARGUMENT;
    if (table.map == nullptr || table_off % 16 != 0 || table_off > table.size || table.size - table_off < kChainCoreTableBytes)
        return CERE_ERR_ARGUMENT;
    ConvChains conv{};
    DecodeRowsArgs da{};
    conv.chains = da.chains = int(chains.count);
    int64_t next = 0;
    for (int64_t c = 0; c < chains.count; ++c) {
        const LayerState &s = chains.states[c];
        if (s.conv == 0 || s.ssm == 0 || !al(s.conv, 2) || !al(s.ssm, 8) || chains.rows[c] < 1 || chains.rows[c] > rows - next)
            return CERE_ERR_ARGUMENT;
        for (int64_t o = 0; o < c; ++o)
            if (shares(s, chains.states[o])) return CERE_ERR_ARGUMENT;
        conv.first[c] = da.first[c] = int(next);
        conv.rows[c] = da.rows[c] = int(chains.rows[c]);
        conv.states[c] = s.conv;
        da.states[c] = s.ssm;
        next += chains.rows[c];
        for (int64_t r = conv.first[c]; r + 1 < next; ++r) {
            const LayerState &save = chains.saves[r];
            if (save.conv == 0 || save.ssm == 0 || !al(save.conv, 2) || !al(save.ssm, 8)) return CERE_ERR_ARGUMENT;
            for (int64_t o = 0; o < chains.count; ++o)
                if (shares(save, chains.states[o])) return CERE_ERR_ARGUMENT;
            conv.saves[r] = save.conv;
            da.saves[r] = save.ssm;
        }
    }
    if (next != rows) return CERE_ERR_ARGUMENT;
    // conv1d + SiLU over all 10,240 channels of each row, the chain's conv state carried row to row (saves in-kernel).
    cere_status st = conv1d_silu_rows(ctx, mixed, kConvDim, w.conv1d, conv, ws.conv_out, kConvDim, kConvDim, table, table_off);
    if (st != CERE_OK) return st;
    ctx.barrier();
    da.step.q = ws.conv_out;
    da.step.k = ws.conv_out + uint64_t(kKeyDim) * 2;
    da.step.v = ws.conv_out + uint64_t(2 * kKeyDim) * 2;
    da.step.a = a;
    da.step.b = b;
    da.step.A_log = w.A_log;
    da.step.dt_bias = w.dt_bias;
    da.step.state = 0;  // not read by the rows form
    da.step.o = ws.o;
    da.step.scale = kScale;
    da.qkv_stride = kConvDim;
    da.ab_stride = kValueHeads;
    da.o_stride = kValueDim;
    st = delta_rule_decode_rows(ctx, da, table, table_off + kChainTableBytes);
    if (st != CERE_OK) return st;
    ctx.barrier();
    return vkk1::rmsnorm_gated_sigmoid(ctx, ws.o, kHeadDim, z, kHeadDim, w.norm, kNormEps, rows * kValueHeads, kHeadDim, y, kHeadDim);
}

}  // namespace cere::vkk5
