// VulkanAmd P2: see k5_gdn_chunk_vk.h.
#include "kernels/k5_gdn_chunk_vk.h"

#include "kernels/k5_gdn_vk.h"

namespace cere::vkk5 {
namespace {

constexpr int KH = kKeyHeads, VH = kValueHeads, D = kHeadDim, BT = kChunkTokens;

struct PermutePush {
    uint64_t src, dst;
    uint32_t T, H;
};
static_assert(sizeof(PermutePush) == 24, "k5_gdn_chunk.comp STAGE 0 / 5");
struct CumsumPush {
    uint64_t g, beta, G, B;
    uint32_t T, chunks;
};
static_assert(sizeof(CumsumPush) == 40, "k5_gdn_chunk.comp STAGE 1");
struct PrepPush {
    uint64_t qn, kn, vt, G, B, P, u, w;
    uint32_t T, pad;
};
static_assert(sizeof(PrepPush) == 72, "k5_gdn_chunk.comp STAGE 2");
struct RecurPush {
    uint64_t u, w, kn, G, hc, vn, state, in32, in32_hi, cap, cap_hi;
    uint32_t T, chunks, cap_chunk, has_cap;
};
static_assert(sizeof(RecurPush) == 104, "k5_gdn_chunk.comp STAGE 3");
struct OutPush {
    uint64_t qn, hc, P, vn, G, ot;
    uint32_t T, chunks;
    float scale;
    uint32_t pad;
};
static_assert(sizeof(OutPush) == 64, "k5_gdn_chunk.comp STAGE 4");

struct Workspace {
    uint64_t qn, kn, vt, G, B, u, w, vn, P, hc, ot;
};

uint64_t align256(uint64_t b) { return (b + 255) & ~uint64_t(255); }

uint64_t carve(uint64_t base, int64_t T, Workspace *ws) {
    uint64_t off = 0;
    const uint64_t t = uint64_t(T), chunks = (t + BT - 1) / BT;
    auto take = [&](uint64_t bytes) { const uint64_t r = base + off; off += align256(bytes); return r; };
    Workspace w{};
    w.qn = take(t * KH * D * 2);
    w.kn = take(t * KH * D * 2);
    w.vt = take(t * VH * D * 2);
    w.G = take(t * VH * 4);
    w.B = take(t * VH * 4);
    w.u = take(t * VH * D * 2);
    w.w = take(t * VH * D * 2);
    w.vn = take(t * VH * D * 2);
    w.P = take(t * VH * BT * 2);
    w.hc = take(chunks * VH * D * D * 2);
    w.ot = take(t * VH * D * 2);  // >= 2 x [T, 16, 128]: the token-major normalised q, k before the output
    if (ws) *ws = w;
    return off;
}

bool aligned(uint64_t a, uint64_t n) { return a != 0 && a % n == 0; }

}  // namespace

std::vector<vk::PipelineSpec> chunk_pipeline_specs() {
    std::vector<vk::PipelineSpec> s = {{"k5_gdn_chunk_permute.spv", sizeof(PermutePush), 0, {}},
                                       {"k5_gdn_chunk_cumsum.spv", sizeof(CumsumPush), 0, {}},
                                       {"k5_gdn_chunk_prep.spv", sizeof(PrepPush), 32, {}},
                                       {"k5_gdn_chunk_recur.spv", sizeof(RecurPush), 32, {}},
                                       {"k5_gdn_chunk_out.spv", sizeof(OutPush), 0, {}},
                                       {"k5_gdn_chunk_unpermute.spv", sizeof(PermutePush), 0, {}}};
    for (const vk::PipelineSpec &p : piece_pipeline_specs(kPieceL2Norm)) s.push_back(p);  // the q / k l2norms
    return s;
}

size_t chunk_workspace_bytes(int64_t tokens) {
    if (tokens < 1 || tokens > kChunkMaxTokens) return 0;
    return size_t(carve(0, tokens, nullptr));
}

cere_status delta_rule_chunk(vk::Context &ctx, const ChunkArgs &a, uint64_t workspace) {
    if (a.tokens < 1 || a.tokens > kChunkMaxTokens || !aligned(a.q, 16) || !aligned(a.k, 16) || !aligned(a.v, 16) ||
        !aligned(a.o, 16) || !aligned(a.g, 4) || !aligned(a.beta, 4) || !aligned(a.state, 4) || !aligned(workspace, 256) ||
        (a.state_in32 != 0 && a.state_in32 % 4 != 0) || (a.capture32 != 0 && a.capture32 % 4 != 0) ||
        (a.state_in32_hi != 0 && (a.state_in32 == 0 || a.state_in32_hi % 4 != 0)) ||
        (a.capture32_hi != 0 && (a.capture32 == 0 || a.capture32_hi % 4 != 0)))
        return CERE_ERR_ARGUMENT;
    const int64_t T = a.tokens, chunks = (T + BT - 1) / BT;
    if (a.capture32 != 0 && (a.capture_chunk < 0 || a.capture_chunk > chunks)) return CERE_ERR_ARGUMENT;
    Workspace ws{};
    carve(workspace, T, &ws);
    const uint32_t t32 = uint32_t(T), c32 = uint32_t(chunks);
    const uint64_t qtok = ws.ot, ktok = ws.ot + uint64_t(T) * KH * D * 2;
    if (cere_status s = l2norm(ctx, a.q, qtok, T * KH, D, 1e-6f); s != CERE_OK) return s;
    if (cere_status s = l2norm(ctx, a.k, ktok, T * KH, D, 1e-6f); s != CERE_OK) return s;
    const vk::Pipeline &perm = ctx.pipeline("k5_gdn_chunk_permute.spv", sizeof(PermutePush));
    {   // v and the cumsum do not depend on the norms
        const PermutePush pv{a.v, ws.vt, t32, VH};
        ctx.dispatch(perm, &pv, uint32_t((uint64_t(T) * VH * 16 + 255) / 256));
        const CumsumPush pc{a.g, a.beta, ws.G, ws.B, t32, c32};
        ctx.dispatch(ctx.pipeline("k5_gdn_chunk_cumsum.spv", sizeof(CumsumPush)), &pc, uint32_t((chunks * VH + 255) / 256));
    }
    ctx.barrier();
    const PermutePush pq{qtok, ws.qn, t32, KH}, pk{ktok, ws.kn, t32, KH};
    ctx.dispatch(perm, &pq, uint32_t((uint64_t(T) * KH * 16 + 255) / 256));
    ctx.dispatch(perm, &pk, uint32_t((uint64_t(T) * KH * 16 + 255) / 256));
    ctx.barrier();
    const PrepPush pp{ws.qn, ws.kn, ws.vt, ws.G, ws.B, ws.P, ws.u, ws.w, t32, 0};
    ctx.dispatch(ctx.pipeline("k5_gdn_chunk_prep.spv", sizeof(PrepPush), 32), &pp, c32, VH);
    ctx.barrier();
    const RecurPush pr{ws.u, ws.w, ws.kn, ws.G, ws.hc, ws.vn, a.state, a.state_in32, a.state_in32_hi, a.capture32,
                       a.capture32_hi, t32, c32, a.capture32 != 0 ? uint32_t(a.capture_chunk) : 0u, a.capture32 != 0 ? 1u : 0u};
    ctx.dispatch(ctx.pipeline("k5_gdn_chunk_recur.spv", sizeof(RecurPush), 32), &pr, VH, D / 32);
    ctx.barrier();
    const OutPush po{ws.qn, ws.hc, ws.P, ws.vn, ws.G, ws.ot, t32, c32, a.scale, 0};
    ctx.dispatch(ctx.pipeline("k5_gdn_chunk_out.spv", sizeof(OutPush)), &po, c32, VH, D / 64);
    ctx.barrier();
    const PermutePush pb{ws.ot, a.o, t32, VH};
    ctx.dispatch(ctx.pipeline("k5_gdn_chunk_unpermute.spv", sizeof(PermutePush)), &pb, uint32_t((T + 15) / 16));
    return CERE_OK;
}

}  // namespace cere::vkk5
