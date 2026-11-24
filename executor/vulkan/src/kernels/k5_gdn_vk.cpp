// VulkanAmd P2: see k5_gdn_vk.h.
#include "kernels/k5_gdn_vk.h"

#include <cstring>

namespace cere::vkk5 {
namespace {

struct DecodePush {
    uint64_t q, k, v, a, b, A_log, dt_bias, state, o;
    float scale;
    uint32_t pad;
};
static_assert(sizeof(DecodePush) == 80, "push layout matches k5_gdn_decode.comp ROWS 0");

struct DecodeRowsPush {
    uint64_t q, k, v, a, b, A_log, dt_bias, o, table;
    float scale;
    uint32_t qkv_stride, ab_stride, o_stride;
};
static_assert(sizeof(DecodeRowsPush) == 88, "push layout matches k5_gdn_decode.comp ROWS 1");

struct ConvPush {
    uint64_t x, w, state, y;
    uint32_t x_stride, y_stride, tokens, dim;
};
static_assert(sizeof(ConvPush) == 48, "push layout matches k5_gdn_conv.comp MODE 0");

struct ConvStatePush {
    uint64_t x, state;
    uint32_t x_stride, tokens, dim, pad;
};
static_assert(sizeof(ConvStatePush) == 32, "push layout matches k5_gdn_conv.comp MODE 1");

struct ConvRowsPush {
    uint64_t x, w, y, table;
    uint32_t x_stride, y_stride, dim, pad;
};
static_assert(sizeof(ConvRowsPush) == 48, "push layout matches k5_gdn_conv.comp MODE 2");

struct GatingPush {
    uint64_t a, b, A_log, dt_bias, g, beta;
    uint32_t a_stride, b_stride, tokens, heads;
};
static_assert(sizeof(GatingPush) == 64, "push layout matches k5_gdn_gating.comp");

struct L2Push {
    uint64_t x, y;
    uint32_t rows;
    float eps;
};
static_assert(sizeof(L2Push) == 24, "push layout matches k5_gdn_l2norm.comp");

// The chain table the *_rows shaders read (std430: 16 u64 states, 16 u64 saves, 16 u32 first, 16 u32 rows).
struct Table {
    uint64_t states[kMaxRows];
    uint64_t saves[kMaxRows];
    uint32_t first[kMaxRows];
    uint32_t rows[kMaxRows];
};
static_assert(sizeof(Table) == kChainTableBytes, "chain table layout");

constexpr uint64_t k4G = uint64_t(1) << 32;
constexpr uint32_t kMaxGroups = 65535;  // the Vulkan-guaranteed minimum of maxComputeWorkGroupCount
constexpr uint32_t kConvThreads = 256;
constexpr uint32_t kConvTokensPerInvocation = 16;

bool aligned(uint64_t a, uint64_t n) { return a != 0 && a % n == 0; }

bool inputs_ok(const DecodeArgs &p) {
    return aligned(p.q, 8) && aligned(p.k, 8) && aligned(p.v, 8) && aligned(p.o, 8) && aligned(p.a, 2) &&
           aligned(p.b, 2) && aligned(p.A_log, 4) && aligned(p.dt_bias, 2);
}

bool write_table(const vk::Buffer &table, uint64_t off, const Table &t) {
    if (table.map == nullptr || off % 16 != 0 || off + kChainTableBytes > table.size) return false;
    std::memcpy(static_cast<uint8_t *>(table.map) + off, &t, sizeof t);
    return true;
}

// Chains back to back from row 0, each >= 1 row, <= kMaxRows rows in all; states set; every row but a chain's last has
// a save. Returns the total row count, 0 when invalid.
template <class A> int chains_ok(const A &c, uint64_t save_align) {
    if (c.chains < 1 || c.chains > kMaxRows) return 0;
    int next = 0;
    for (int i = 0; i < c.chains; ++i) {
        if (!aligned(c.states[i], save_align) || c.first[i] != next || c.rows[i] < 1 || c.rows[i] > kMaxRows - next) return 0;
        next += c.rows[i];
        for (int r = c.first[i]; r + 1 < next; ++r)
            if (!aligned(c.saves[r], save_align)) return 0;
    }
    return next;
}

template <class A> Table table_of(const A &c) {
    Table t{};
    for (int i = 0; i < c.chains; ++i) {
        t.states[i] = c.states[i];
        t.first[i] = uint32_t(c.first[i]);
        t.rows[i] = uint32_t(c.rows[i]);
    }
    for (int r = 0; r < kMaxRows; ++r) t.saves[r] = c.saves[r];
    return t;
}

}  // namespace

std::vector<vk::PipelineSpec> piece_pipeline_specs(uint32_t pieces) {
    std::vector<vk::PipelineSpec> s;
    if (pieces & kPieceDecode) s.push_back({"k5_gdn_decode.spv", sizeof(DecodePush), 32, {}});
    if (pieces & kPieceRows) {
        s.push_back({"k5_gdn_decode_rows.spv", sizeof(DecodeRowsPush), 32, {}});
        s.push_back({"k5_gdn_conv_rows.spv", sizeof(ConvRowsPush), 0, {}});
    }
    if (pieces & kPieceConvGating) {
        s.push_back({"k5_gdn_conv.spv", sizeof(ConvPush), 0, {}});
        s.push_back({"k5_gdn_conv_state.spv", sizeof(ConvStatePush), 0, {}});
        s.push_back({"k5_gdn_gating.spv", sizeof(GatingPush), 0, {}});
    }
    if (pieces & kPieceL2Norm) s.push_back({"k5_gdn_l2norm.spv", sizeof(L2Push), 32, {}});
    return s;
}

cere_status delta_rule_decode(vk::Context &ctx, const DecodeArgs &args) {
    if (!inputs_ok(args) || !aligned(args.state, 8)) return CERE_ERR_ARGUMENT;
    const vk::Pipeline &p = ctx.pipeline("k5_gdn_decode.spv", sizeof(DecodePush), 32);
    const DecodePush pc{args.q, args.k, args.v, args.a, args.b, args.A_log, args.dt_bias, args.state, args.o, args.scale, 0};
    ctx.dispatch(p, &pc, kValueHeads, kHeadDim / 16);
    return CERE_OK;
}

cere_status delta_rule_decode_rows(vk::Context &ctx, const DecodeRowsArgs &args, const vk::Buffer &table,
                                   uint64_t table_off) {
    constexpr int64_t kRow = int64_t(kValueHeads) * kHeadDim;  // q|k (2 x 16 x 128) and v, o (48 x 128) per row
    if (!inputs_ok(args.step)) return CERE_ERR_ARGUMENT;
    const int total = chains_ok(args, 8);
    if (total == 0) return CERE_ERR_ARGUMENT;
    if (total > 1 && (args.qkv_stride < kRow || args.qkv_stride % 4 != 0 || args.ab_stride < kValueHeads ||
                      args.o_stride < kRow || args.o_stride % 4 != 0))
        return CERE_ERR_ARGUMENT;
    // 32-bit byte offsets: the last row's extent inside 4 GiB.
    const uint64_t last = uint64_t(total - 1);
    if (args.qkv_stride < 0 || args.ab_stride < 0 || args.o_stride < 0 ||
        last * uint64_t(args.qkv_stride) * 2 + uint64_t(kRow) * 2 > k4G ||
        last * uint64_t(args.ab_stride) * 2 + kValueHeads * 2 > k4G || last * uint64_t(args.o_stride) * 2 + uint64_t(kRow) * 2 > k4G)
        return CERE_ERR_ARGUMENT;
    if (!write_table(table, table_off, table_of(args))) return CERE_ERR_ARGUMENT;
    const vk::Pipeline &p = ctx.pipeline("k5_gdn_decode_rows.spv", sizeof(DecodeRowsPush), 32);
    const DecodeRowsPush pc{args.step.q, args.step.k, args.step.v, args.step.a, args.step.b, args.step.A_log,
                            args.step.dt_bias, args.step.o, table.addr + table_off, args.step.scale,
                            uint32_t(args.qkv_stride), uint32_t(args.ab_stride), uint32_t(args.o_stride)};
    ctx.dispatch(p, &pc, kValueHeads, kHeadDim / 16, uint32_t(args.chains));
    return CERE_OK;
}

cere_status conv1d_silu(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t weight, uint64_t state, uint64_t y,
                        int64_t y_stride, int64_t tokens, int64_t dim) {
    if (!aligned(x, 2) || !aligned(weight, 2) || !aligned(state, 2) || !aligned(y, 2) || tokens < 1 || dim < 1 ||
        x_stride < dim || y_stride < dim)
        return CERE_ERR_ARGUMENT;
    const uint64_t chunks = (uint64_t(tokens) + kConvTokensPerInvocation - 1) / kConvTokensPerInvocation;
    const uint64_t blocks = (uint64_t(dim) + kConvThreads - 1) / kConvThreads;
    if (chunks > kMaxGroups || blocks > kMaxGroups || (uint64_t(tokens - 1) * x_stride + dim) * 2 > k4G ||
        (uint64_t(tokens - 1) * y_stride + dim) * 2 > k4G)
        return CERE_ERR_ARGUMENT;
    const ConvPush pc{x, weight, state, y, uint32_t(x_stride), uint32_t(y_stride), uint32_t(tokens), uint32_t(dim)};
    ctx.dispatch(ctx.pipeline("k5_gdn_conv.spv", sizeof(ConvPush)), &pc, uint32_t(blocks), uint32_t(chunks));
    ctx.barrier();  // the state dispatch overwrites what the conv dispatch reads
    const ConvStatePush ps{x, state, uint32_t(x_stride), uint32_t(tokens), uint32_t(dim), 0};
    ctx.dispatch(ctx.pipeline("k5_gdn_conv_state.spv", sizeof(ConvStatePush)), &ps, uint32_t(blocks));
    return CERE_OK;
}

cere_status conv1d_state_at(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t state, int64_t tokens, int64_t dim) {
    if (!aligned(x, 2) || !aligned(state, 2) || tokens < kConvState || dim < 1 || x_stride < dim) return CERE_ERR_ARGUMENT;
    const uint64_t blocks = (uint64_t(dim) + kConvThreads - 1) / kConvThreads;
    if (blocks > kMaxGroups || (uint64_t(tokens - 1) * x_stride + dim) * 2 > k4G) return CERE_ERR_ARGUMENT;
    const ConvStatePush ps{x, state, uint32_t(x_stride), uint32_t(tokens), uint32_t(dim), 0};
    ctx.dispatch(ctx.pipeline("k5_gdn_conv_state.spv", sizeof(ConvStatePush)), &ps, uint32_t(blocks));
    return CERE_OK;
}

cere_status conv1d_silu_rows(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t weight, const ConvChains &chains,
                             uint64_t y, int64_t y_stride, int64_t dim, const vk::Buffer &table, uint64_t table_off) {
    if (!aligned(x, 2) || !aligned(weight, 2) || !aligned(y, 2) || dim < 1 || x_stride < dim || y_stride < dim)
        return CERE_ERR_ARGUMENT;
    const int total = chains_ok(chains, 2);
    if (total == 0) return CERE_ERR_ARGUMENT;
    for (int r = 0; r + 1 < total; ++r)  // a save never aliases a state (rows that are chains' last have no save)
        for (int s = 0; s < chains.chains; ++s)
            if (chains.saves[r] != 0 && chains.saves[r] == chains.states[s]) return CERE_ERR_ARGUMENT;
    const uint64_t blocks = (uint64_t(dim) + kConvThreads - 1) / kConvThreads;
    if (blocks > kMaxGroups || (uint64_t(total - 1) * x_stride + dim) * 2 > k4G ||
        (uint64_t(total - 1) * y_stride + dim) * 2 > k4G)
        return CERE_ERR_ARGUMENT;
    if (!write_table(table, table_off, table_of(chains))) return CERE_ERR_ARGUMENT;
    const ConvRowsPush pc{x, weight, y, table.addr + table_off, uint32_t(x_stride), uint32_t(y_stride), uint32_t(dim), 0};
    ctx.dispatch(ctx.pipeline("k5_gdn_conv_rows.spv", sizeof(ConvRowsPush)), &pc, uint32_t(blocks), uint32_t(chains.chains));
    return CERE_OK;
}

cere_status gating(vk::Context &ctx, uint64_t a, int64_t a_stride, uint64_t b, int64_t b_stride, uint64_t A_log,
                   uint64_t dt_bias, uint64_t g, uint64_t beta, int64_t tokens, int64_t heads) {
    if (!aligned(a, 2) || !aligned(b, 2) || !aligned(A_log, 4) || !aligned(dt_bias, 2) || !aligned(g, 4) ||
        !aligned(beta, 4) || tokens < 1 || heads < 1 || a_stride < heads || b_stride < heads)
        return CERE_ERR_ARGUMENT;
    const uint64_t n = uint64_t(tokens) * uint64_t(heads);
    const uint64_t blocks = (n + 255) / 256;
    if (blocks > kMaxGroups || n * 4 > k4G || (uint64_t(tokens - 1) * a_stride + heads) * 2 > k4G ||
        (uint64_t(tokens - 1) * b_stride + heads) * 2 > k4G)
        return CERE_ERR_ARGUMENT;
    const GatingPush pc{a, b, A_log, dt_bias, g, beta, uint32_t(a_stride), uint32_t(b_stride), uint32_t(tokens), uint32_t(heads)};
    ctx.dispatch(ctx.pipeline("k5_gdn_gating.spv", sizeof(GatingPush)), &pc, uint32_t(blocks));
    return CERE_OK;
}

cere_status l2norm(vk::Context &ctx, uint64_t x, uint64_t y, int64_t rows, int64_t dim, float eps) {
    if (!aligned(x, 16) || !aligned(y, 16) || rows < 1 || dim != kHeadDim) return CERE_ERR_ARGUMENT;
    const uint64_t blocks = (uint64_t(rows) + 15) / 16;
    if (blocks > kMaxGroups || uint64_t(rows) * kHeadDim * 2 > k4G) return CERE_ERR_ARGUMENT;
    const L2Push pc{x, y, uint32_t(rows), eps};
    ctx.dispatch(ctx.pipeline("k5_gdn_l2norm.spv", sizeof(L2Push), 32), &pc, uint32_t(blocks));
    return CERE_OK;
}

}  // namespace cere::vkk5
