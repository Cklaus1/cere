// VulkanAmd P2: see k13s_spec_accept_vk.h.
#include "kernels/k13s_spec_accept_vk.h"

#include "kernels/k12_sampler_vk.h"

namespace cere::vkk13s {
namespace {

constexpr int64_t kMaxVocab = int64_t(1) << 30;
constexpr uint64_t kAlign = 256, kRowPrefix = 8216;
constexpr uint32_t kSubgroup = 32;

struct AcceptPush {
    uint64_t probs, masks, drafts, rows, ids, accepted, first_seed, seed_step, bonus, committed, count;
    uint32_t vocab, draft, key0, key_step, trials, per_row, n;
};
static_assert(sizeof(AcceptPush) == 120, "push layout");

bool al(uint64_t a, uint64_t n) { return a % n == 0; }
bool vocab_ok(int64_t v) { return v >= 2 && v <= kMaxVocab; }
uint64_t aligned(uint64_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// spec_accept's scratch for n drafts: K12's draw scratch, the bonus id, n RowPrefix, n row ids, n row flags.
struct Carve { uint64_t k12, bonus, rows, row_ids, row_acc, total; };
Carve carve(int64_t n, uint64_t base) {
    Carve c{};
    uint64_t at = 0;
    const auto take = [&](uint64_t bytes) { const uint64_t p = base + at; at += aligned(bytes); return p; };
    c.k12 = take(vkk12::scratch_bytes());
    c.bonus = take(4);
    c.rows = take(uint64_t(n) * kRowPrefix);
    c.row_ids = take(uint64_t(n) * 4);
    c.row_acc = take(uint64_t(n));
    c.total = at;
    return c;
}

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs() {
    // spec_accept: accept s0 / s1 (also row_trials), s2, and the bonus id's seeded draw (K12's prefix + seeded pipelines).
    std::vector<vk::PipelineSpec> s = {{"k13s_accept_s0.spv", sizeof(AcceptPush), kSubgroup, {}},
                                       {"k13s_accept_s1.spv", sizeof(AcceptPush), kSubgroup, {}},
                                       {"k13s_accept_s2.spv", sizeof(AcceptPush), 0, {}}};
    for (const vk::PipelineSpec &d : vkk12::draw_pipeline_specs(true, false)) s.push_back(d);
    return s;
}

size_t scratch_bytes(int64_t max_drafts) {
    if (max_drafts < 0) max_drafts = 0;
    if (max_drafts > kMaxDrafts) max_drafts = kMaxDrafts;
    return carve(max_drafts, 0).total;
}

cere_status spec_accept(vk::Context &ctx, uint64_t probs, uint64_t masks, int64_t vocab, uint64_t drafts, int64_t n, uint64_t seed, uint32_t key0,
                        uint64_t committed, uint64_t count, uint64_t scratch) {
    if (!probs || !masks || !vocab_ok(vocab) || n < 0 || n > kMaxDrafts || (n > 0 && !drafts) || !committed || !count || !scratch || !al(scratch, 8) ||
        !al(probs, 4) || !al(masks, 4) || (n > 0 && !al(drafts, 4)) || !al(committed, 4) || !al(count, 4))
        return CERE_ERR_ARGUMENT;
    const Carve s = carve(n, scratch);
    const uint64_t words = uint64_t((vocab + 31) / 32);
    AcceptPush pc{probs, masks, drafts, s.rows, s.row_ids, s.row_acc, seed, 0, s.bonus, committed, count,
                  uint32_t(vocab), 0, key0, 1, uint32_t(n), 1, uint32_t(n)};
    if (n > 0) {
        ctx.dispatch(ctx.pipeline("k13s_accept_s0.spv", sizeof pc, kSubgroup), &pc, uint32_t(n));
        ctx.barrier();
        ctx.dispatch(ctx.pipeline("k13s_accept_s1.spv", sizeof pc, kSubgroup), &pc, uint32_t((n + 127) / 128));
        ctx.barrier();
    }
    // The bonus id: k12 draw_seeded of row n (plain sampling, channels (0, 1)) at key0 + n.
    const cere_status st = vkk12::draw_seeded(ctx, probs + uint64_t(n) * uint64_t(vocab) * 4, masks + uint64_t(n) * words * 4, vocab, seed, 1,
                                              key0 + uint32_t(n), s.bonus, s.k12);
    if (st != CERE_OK) return st;
    ctx.barrier();
    ctx.dispatch(ctx.pipeline("k13s_accept_s2.spv", sizeof pc, 0), &pc, 1);
    return CERE_OK;
}

size_t trial_scratch_bytes() { return kRowPrefix; }

cere_status row_trials(vk::Context &ctx, uint64_t probs, uint64_t mask, int64_t vocab, uint32_t draft, uint64_t first_seed, uint64_t seed_step,
                       uint32_t key0, uint32_t key_step, int64_t trials, uint64_t ids, uint64_t accepted, uint64_t scratch) {
    if (!probs || !mask || !vocab_ok(vocab) || trials < 0 || trials > 0x7fffffff || (trials > 0 && (!ids || !accepted)) || !scratch || !al(scratch, 8) ||
        !al(probs, 4) || !al(mask, 4) || (trials > 0 && !al(ids, 4)))
        return CERE_ERR_ARGUMENT;
    if (trials == 0) return CERE_OK;
    AcceptPush pc{probs, mask, 0, scratch, ids, accepted, first_seed, seed_step, 0, 0, 0, uint32_t(vocab), draft, key0, key_step, uint32_t(trials), 0, 0};
    ctx.dispatch(ctx.pipeline("k13s_accept_s0.spv", sizeof pc, kSubgroup), &pc, 1);
    ctx.barrier();
    ctx.dispatch(ctx.pipeline("k13s_accept_s1.spv", sizeof pc, kSubgroup), &pc, uint32_t((trials + 127) / 128));
    return CERE_OK;
}

}  // namespace cere::vkk13s
