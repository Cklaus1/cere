// VulkanAmd P2: see k12_sampler_vk.h.
#include "kernels/k12_sampler_vk.h"

#include <cmath>

namespace cere::vkk12 {
namespace {

constexpr int64_t kMaxVocab = int64_t(1) << 30;
constexpr uint32_t kGrid = 512, kScratch = 38952;
constexpr uint32_t kSubgroup = 32;

bool al(uint64_t a, uint64_t n) { return a % n == 0; }
bool vocab_ok(int64_t v) { return v >= 2 && v <= kMaxVocab; }
bool scratch_ok(uint64_t s) { return s != 0 && al(s, 8); }

struct ProcPush { uint64_t logits, counts, mask, out, bias_ids, bias_values; uint32_t vocab, bias_stride, bias_count; float f, p, r; };
struct Top2Push { uint64_t values, ids, scores; uint32_t vocab, pad; };
struct ProbsPush { uint64_t x, probs, work, lse; uint32_t vocab; float temperature; };
struct KeepPush { uint64_t probs, work, mask; uint32_t vocab, top_k; float top_p, min_p; uint32_t span, collect_blocks; };
struct DrawPush { uint64_t probs, mask, scratch, us, ids, first_seed; uint32_t vocab, key, n, pad; };
struct SmallPush { uint64_t a, b, c, d; uint32_t n, pad; };
struct LogprobsPush { uint64_t x, committed_dev, committed_logprob, top_ids, top_lps; uint32_t vocab, committed_host, n, stride; float temperature; uint32_t pad; };
static_assert(sizeof(ProcPush) == 72 && sizeof(Top2Push) == 32 && sizeof(ProbsPush) == 40 && sizeof(KeepPush) == 48 &&
              sizeof(DrawPush) == 64 && sizeof(SmallPush) == 40 && sizeof(LogprobsPush) == 64, "push layouts");

// keep_mask's three histogram passes, one list for the dispatch loop and pipeline_specs.
constexpr const char *kKeepHist[3] = {"k12_keep_hist0.spv", "k12_keep_hist1.spv", "k12_keep_hist2.spv"};
constexpr int64_t kKeepMaskedAbove = 1024;  // keep_mask runs the masked pass only for top_k above this

uint32_t groups(uint64_t n, uint32_t per) { return uint32_t((n + per - 1) / per); }

cere_status draw(vk::Context &ctx, uint64_t probs, uint64_t mask, int64_t vocab, uint64_t us, uint64_t first_seed, int64_t n, uint32_t key,
                 uint64_t ids, uint64_t scratch, bool seeded) {
    if (!probs || !mask || !vocab_ok(vocab) || n < 0 || n > 0x7fffffff || (n > 0 && !ids) || !scratch_ok(scratch) || !al(probs, 4) ||
        !al(mask, 4) || !al(ids, 4) || (!seeded && n > 0 && (!us || !al(us, 8))))
        return CERE_ERR_ARGUMENT;
    if (n == 0) return CERE_OK;
    const DrawPush pc{probs, mask, scratch, us, ids, first_seed, uint32_t(vocab), key, uint32_t(n), 0};
    ctx.dispatch(ctx.pipeline("k12_draw_prefix.spv", sizeof pc, kSubgroup), &pc, 1);
    ctx.barrier();
    ctx.dispatch(ctx.pipeline(seeded ? "k12_draw_seeded.spv" : "k12_draw_u.spv", sizeof pc, kSubgroup), &pc, groups(uint64_t(n), 256));
    return CERE_OK;
}

}  // namespace

std::vector<vk::PipelineSpec> draw_pipeline_specs(bool seeded, bool u) {
    std::vector<vk::PipelineSpec> s = {{"k12_draw_prefix.spv", sizeof(DrawPush), kSubgroup, {}}};
    if (seeded) s.push_back({"k12_draw_seeded.spv", sizeof(DrawPush), kSubgroup, {}});
    if (u) s.push_back({"k12_draw_u.spv", sizeof(DrawPush), kSubgroup, {}});
    return s;
}

std::vector<vk::PipelineSpec> pipeline_specs(int64_t max_top_k) {
    std::vector<vk::PipelineSpec> s = {{"k12_process.spv", sizeof(ProcPush), kSubgroup, {}},
                                       {"k12_bias.spv", sizeof(ProcPush), kSubgroup, {}},
                                       {"k12_top2.spv", sizeof(Top2Push), kSubgroup, {}},
                                       {"k12_probs_s0.spv", sizeof(ProbsPush), kSubgroup, {}},
                                       {"k12_probs_s1.spv", sizeof(ProbsPush), kSubgroup, {}},
                                       {"k12_probs_s2.spv", sizeof(ProbsPush), kSubgroup, {}},
                                       {"k12_keep_clear.spv", sizeof(KeepPush), kSubgroup, {}}};
    for (const char *h : kKeepHist) s.push_back({h, sizeof(KeepPush), kSubgroup, {}});
    s.push_back({"k12_keep_collect.spv", sizeof(KeepPush), kSubgroup, {}});
    s.push_back({"k12_keep_resolve.spv", sizeof(KeepPush), kSubgroup, {}});
    if (max_top_k > kKeepMaskedAbove) s.push_back({"k12_keep_masked.spv", sizeof(KeepPush), kSubgroup, {}});
    for (const vk::PipelineSpec &d : draw_pipeline_specs(true, true)) s.push_back(d);
    s.push_back({"k12_finalize.spv", sizeof(SmallPush), 0, {}});
    s.push_back({"k12_logprobs.spv", sizeof(LogprobsPush), kSubgroup, {}});
    s.push_back({"k12_count_commit.spv", sizeof(SmallPush), 0, {}});
    return s;
}

cere_status process(vk::Context &ctx, uint64_t logits, int64_t vocab, uint64_t counts, float f, float p, float r, uint64_t mask_dev,
                    uint64_t bias_ids, uint64_t bias_values, int64_t bias_stride, int64_t bias_count, uint64_t out) {
    if (!logits || !out || !vocab_ok(vocab) || bias_count < 0 || bias_count > vocab ||
        (bias_count > 0 && (!bias_ids || !bias_values || bias_stride < 1 || bias_stride > (int64_t(1) << 20))) || !al(logits, 2) ||
        !al(out, 4) || (counts && !al(counts, 4)) || (mask_dev && !al(mask_dev, 4)) || (bias_count > 0 && (!al(bias_ids, 4) || !al(bias_values, 4))))
        return CERE_ERR_ARGUMENT;
    ProcPush pc{logits, counts, mask_dev, out, bias_ids, bias_values, uint32_t(vocab), uint32_t(bias_stride), uint32_t(bias_count), f, p, r};
    ctx.dispatch(ctx.pipeline("k12_process.spv", sizeof pc, kSubgroup), &pc, groups(uint64_t(vocab), 256));
    if (bias_count > 0) {
        ctx.barrier();
        ctx.dispatch(ctx.pipeline("k12_bias.spv", sizeof pc, kSubgroup), &pc, groups(uint64_t(bias_count), 256));
    }
    return CERE_OK;
}

cere_status top2_f32(vk::Context &ctx, uint64_t values, int64_t vocab, uint64_t ids, uint64_t scores) {
    if (!values || !ids || !scores || !vocab_ok(vocab) || !al(values, 4) || !al(ids, 4) || !al(scores, 4)) return CERE_ERR_ARGUMENT;
    const Top2Push pc{values, ids, scores, uint32_t(vocab), 0};
    ctx.dispatch(ctx.pipeline("k12_top2.spv", sizeof pc, kSubgroup), &pc, 1);
    return CERE_OK;
}

size_t scratch_bytes() { return kScratch; }

cere_status probabilities(vk::Context &ctx, uint64_t processed, int64_t vocab, float temperature, uint64_t probs, uint64_t lse, uint64_t scratch) {
    if (!processed || !probs || !lse || !vocab_ok(vocab) || !(temperature > 0.0f) || !std::isfinite(temperature) || !scratch_ok(scratch) ||
        !al(processed, 4) || !al(probs, 4) || !al(lse, 8))
        return CERE_ERR_ARGUMENT;
    const ProbsPush pc{processed, probs, scratch, lse, uint32_t(vocab), temperature};
    ctx.dispatch(ctx.pipeline("k12_probs_s0.spv", sizeof pc, kSubgroup), &pc, kGrid);
    ctx.barrier();
    ctx.dispatch(ctx.pipeline("k12_probs_s1.spv", sizeof pc, kSubgroup), &pc, kGrid);
    ctx.barrier();
    ctx.dispatch(ctx.pipeline("k12_probs_s2.spv", sizeof pc, kSubgroup), &pc, kGrid);
    return CERE_OK;
}

cere_status keep_mask(vk::Context &ctx, uint64_t probs, int64_t vocab, int64_t top_k, float top_p, float min_p, uint64_t mask, uint64_t scratch) {
    if (!probs || !mask || !vocab_ok(vocab) || top_k < 1 || top_k > vocab || !(top_p > 0.0f && top_p <= 1.0f) || !(min_p >= 0.0f && min_p <= 1.0f) ||
        !scratch_ok(scratch) || !al(probs, 4) || !al(mask, 4))
        return CERE_ERR_ARGUMENT;
    const uint32_t words = uint32_t((vocab + 31) / 32);
    const uint32_t span = (words + kGrid - 1) / kGrid;
    const uint32_t collect_blocks = (words + span - 1) / span;  // <= kGrid
    const KeepPush pc{probs, scratch, mask, uint32_t(vocab), uint32_t(top_k), top_p, min_p, span, collect_blocks};
    ctx.dispatch(ctx.pipeline("k12_keep_clear.spv", sizeof pc, kSubgroup), &pc, 25);
    ctx.barrier();
    for (const char *h : kKeepHist) {
        ctx.dispatch(ctx.pipeline(h, sizeof pc, kSubgroup), &pc, kGrid);
        ctx.barrier();
    }
    ctx.dispatch(ctx.pipeline("k12_keep_collect.spv", sizeof pc, kSubgroup), &pc, collect_blocks);
    ctx.barrier();
    ctx.dispatch(ctx.pipeline("k12_keep_resolve.spv", sizeof pc, kSubgroup), &pc, 1);
    // Only top_k > 1024 can leave the cut beyond the resolved ranks (resolve marks the row kMasked).
    if (top_k > kKeepMaskedAbove) {
        ctx.barrier();
        ctx.dispatch(ctx.pipeline("k12_keep_masked.spv", sizeof pc, kSubgroup), &pc, 1);
    }
    return CERE_OK;
}

cere_status draw_u(vk::Context &ctx, uint64_t probs, uint64_t mask, int64_t vocab, uint64_t us, int64_t n, uint64_t ids, uint64_t scratch) {
    if (n > 0 && !us) return CERE_ERR_ARGUMENT;
    return draw(ctx, probs, mask, vocab, us, 0, n, 0, ids, scratch, false);
}
cere_status draw_seeded(vk::Context &ctx, uint64_t probs, uint64_t mask, int64_t vocab, uint64_t first_seed, int64_t n, uint32_t key,
                        uint64_t ids, uint64_t scratch) {
    return draw(ctx, probs, mask, vocab, 0, first_seed, n, key, ids, scratch, true);
}

cere_status finalize(vk::Context &ctx, uint64_t processed, uint64_t top2_ids, uint64_t drawn, uint64_t out) {
    if (!processed || !top2_ids || !out || !al(processed, 4) || !al(top2_ids, 4) || (drawn && !al(drawn, 4)) || !al(out, 4)) return CERE_ERR_ARGUMENT;
    const SmallPush pc{processed, top2_ids, drawn, out, 0, 0};
    ctx.dispatch(ctx.pipeline("k12_finalize.spv", sizeof pc, 0), &pc, 1);
    return CERE_OK;
}

cere_status top_logprobs(vk::Context &ctx, uint64_t processed, int64_t vocab, float temperature, uint64_t committed_dev, uint32_t committed_host,
                         int64_t n, uint64_t committed_logprob, uint64_t top_ids, uint64_t top_lps, int64_t stride) {
    if (!processed || !committed_logprob || !vocab_ok(vocab) || !(temperature >= 0.0f) || !std::isfinite(temperature) || n < 0 ||
        n > int64_t(CERE_MAX_TOP_LOGPROBS) || n > vocab || (n > 0 && (!top_ids || !top_lps || stride < 1 || stride > (int64_t(1) << 20))) ||
        (committed_dev == 0 && int64_t(committed_host) >= vocab) || !al(processed, 4) || !al(committed_logprob, 4) || (committed_dev && !al(committed_dev, 4)) ||
        (n > 0 && (!al(top_ids, 4) || !al(top_lps, 4))))
        return CERE_ERR_ARGUMENT;
    const LogprobsPush pc{processed, committed_dev, committed_logprob, top_ids, top_lps, uint32_t(vocab), committed_host, uint32_t(n), uint32_t(stride), temperature, 0};
    ctx.dispatch(ctx.pipeline("k12_logprobs.spv", sizeof pc, kSubgroup), &pc, 1);
    return CERE_OK;
}

cere_status count_commit(vk::Context &ctx, uint64_t counts, uint64_t ids, int64_t n) {
    if (!counts || n < 0 || n > 0x7fffffff || (n > 0 && !ids) || !al(counts, 4) || (n > 0 && !al(ids, 4))) return CERE_ERR_ARGUMENT;
    if (n == 0) return CERE_OK;
    const SmallPush pc{counts, ids, 0, 0, uint32_t(n), 0};
    ctx.dispatch(ctx.pipeline("k12_count_commit.spv", sizeof pc, 0), &pc, groups(uint64_t(n), 64));
    return CERE_OK;
}

}  // namespace cere::vkk12
