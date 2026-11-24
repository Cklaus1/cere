// VulkanAmd P3 T6: the speculation methods of VulkanDevice (vulkan-amd-p3-design design note section 1 rows
// draft_supported .. times_spec_rounds, task T6). Restated from executor/cuda/src/device/cuda_device.cpp:
// draft_supported :1084-1086, set_session_draft :2104-2112, speculate_rows :2342-2389 + speculate_group :2404-2554,
// max_chain_drafts :2391-2394 (CudaModel::max_chain_drafts :406-410), pending_drafts :2396-2402. The recording (verify
// rows, commit + MTP draft, the prompt extend) is vulkan_forward.cpp's (vk_verify, vk_commit_draft, vk_extend_draft).
//
// Round structure (one participant: max_sessions 1, so one call group): (1) verify, one command buffer: the chain's
// n + 1 rows (n <= 3 drafts) through the target model, undo snapshots taken in-kernel / by copies, the per-row top-2
// (and for a sampled chain phase S: K12 stages per row and k13s::spec_accept); (2) the accept on the host from the
// read-back scores (greedy D5: k leading drafts equal the previous row's id; sampled D6: spec_accept's committed ids),
// truncated after the first stop token and at max_commit (M29S2-21); (3) commit + draft, one command buffer: the undo
// restore of rows >= a, the sampled commit counts, then the MTP extend over the a committed rows and two MTP decode
// steps (the next proposals). Statuses, never exits: a failure before the commit consumed nothing (the session is
// broken when anything ran, MF-2); one in the commit + draft submit returns the committed tokens with the error.
//
// Scope refusals (CERE_ERR_UNSUPPORTED, nothing run): everything the levers refuse at allocate (spec_graph decode,
// batched verify) never reaches here. A constrained (vocab_mask) row with drafts (grammar drafts, GrammarSpec GS-02)
// runs since P4-F: one mask per verify row (the host's n + 1 consecutive masks), the caller's drafts only. Caller drafts (VERIFY, lookup drafts the host picked) are verified like the MTP
// proposals, at most max_chain_drafts (3).
#include "device/vulkan_device.h"
#include "model/layout.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace cere::vkdev {

namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

bool VulkanDevice::draft_supported(const host::DeviceModel &model) const {
    const auto &m = static_cast<const VulkanModel &>(model);
    return kDraftForwardImplemented && m.plan().speculative_draft_tokens > 0 && m.weights().has_draft;
}

cere_status VulkanDevice::set_session_draft(host::DeviceModel &model, host::DeviceSession &session, bool on) {
    auto &m = static_cast<VulkanModel &>(model);
    auto &s = static_cast<VulkanSession &>(session);
    if (&s.model() != &m) return CERE_ERR_ARGUMENT;
    if (on && !draft_supported(m)) return CERE_ERR_UNSUPPORTED;
    if (s.position() != 0 || s.prompt_open()) return CERE_ERR_ARGUMENT;
    s.drafting_ = on;
    return CERE_OK;
}

uint32_t VulkanDevice::max_chain_drafts(const host::DeviceModel &model) const {
    const auto &m = static_cast<const VulkanModel &>(model);
    if (!draft_supported(m)) return 3;  // the interface default for a model without drafts
    const uint64_t verify = m.layout().verify_rows;
    return uint32_t(std::min<uint64_t>({uint64_t(kVkMaxChainDrafts), verify == 0 ? 0 : verify - 1,
                                        m.plan().max_sessions * model::layout::kUndoSnapshots}));
}

uint32_t VulkanDevice::pending_drafts(const host::DeviceSession &session, uint32_t *ids, uint32_t max) const {
    const auto &s = static_cast<const VulkanSession &>(session);
    if (!s.drafting() || ids == nullptr) return 0;
    const uint32_t n = std::min(s.proposals()[0], max);
    std::copy(s.proposals() + 1, s.proposals() + 1 + n, ids);
    return n;
}

cere_status VulkanDevice::speculate_rows(host::DeviceModel &model, const host::SpecRowArgs *rows, uint32_t row_count,
                                         host::SpecRowResult *out, cere_status *row_status) {
    auto &m = static_cast<VulkanModel &>(model);
    // Malformed call: nothing runs, row_status untouched (cuda_device.cpp:2346-2357).
    if (rows == nullptr || out == nullptr || row_status == nullptr || row_count == 0 ||
        row_count > m.plan().max_sessions) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t r = 0; r < row_count; ++r) {
        const host::DeviceSession *s = rows[r].session;
        if (s == nullptr || &static_cast<const VulkanSession *>(s)->model() != &m) return CERE_ERR_ARGUMENT;
        if (rows[r].sampling == nullptr && rows[r].logprobs != nullptr) return CERE_ERR_ARGUMENT;
        for (uint32_t q = 0; q < r; ++q) {
            if (rows[q].session == s) return CERE_ERR_ARGUMENT;
        }
    }
    if (!m.instantiated()) return CERE_ERR_ARGUMENT;
    if (!draft_supported(m)) return CERE_ERR_UNSUPPORTED;
    const uint32_t max_chain = max_chain_drafts(m);
    const uint32_t vocab = m.plan().config.vocab_size;
    // Per-row preconditions before any launch (cuda_device.cpp:2416-2450); the rows still OK are the participants.
    uint32_t participants = 0, index = 0;
    for (uint32_t r = 0; r < row_count; ++r) {
        const host::SpecRowArgs &a = rows[r];
        auto &s = static_cast<VulkanSession &>(*a.session);
        const uint32_t *drafts = a.drafts != nullptr ? a.drafts : s.proposals() + 1;
        const uint32_t n = a.drafts != nullptr ? a.draft_count : std::min(a.draft_count, s.proposals()[0]);
        const uint64_t position = s.position();
        bool drafts_ok = true;
        for (uint32_t j = 0; j < n && j < max_chain; ++j) drafts_ok &= drafts[j] < vocab;
        cere_status st = CERE_OK;
        if (s.broken()) {
            st = CERE_ERR_INTERNAL;
        } else if (!forward_supported()) {
            st = CERE_ERR_UNSUPPORTED;
        } else if (position >= s.max_context_tokens() || n >= s.max_context_tokens() - position) {
            st = CERE_ERR_CAPACITY;
        } else if (!s.drafting() || position == 0 || s.prompt_open() || a.token >= vocab || a.max_commit == 0 ||
                   a.draft_count > max_chain || a.draft_count >= a.max_commit ||
                   (a.drafts != nullptr && a.draft_count == 0) || !drafts_ok) {
            st = CERE_ERR_ARGUMENT;
        } else if (a.sampling != nullptr) {
            const bool masked = a.sampling->vocab_mask != nullptr;
            // A constrained row with drafts (GrammarSpec GS-02, cuda_device.cpp:2438-2444; P4-F) verifies the
            // caller's grammar-checked drafts, at most CERE_MAX_CONSTRAINED_DRAFTS, one mask per verify row; never
            // its own MTP proposals.
            if (masked && a.draft_count != 0 &&
                (a.drafts == nullptr || a.draft_count > CERE_MAX_CONSTRAINED_DRAFTS)) {
                st = CERE_ERR_ARGUMENT;
            } else {
                st = vk_check_sampler(*a.sampling, vocab);
            }
        }
        row_status[r] = st;
        if (st != CERE_OK) continue;
        ++participants;
        index = r;
    }
    if (participants == 0) return CERE_OK;
    if (participants > 1) return CERE_ERR_INTERNAL;  // unreachable: row_count <= max_sessions == 1

    const host::SpecRowArgs &a = rows[index];
    auto &s = static_cast<VulkanSession &>(*a.session);
    const uint32_t n = a.drafts != nullptr ? a.draft_count : std::min(a.draft_count, s.proposals()[0]);
    const uint32_t *drafts = a.drafts != nullptr ? a.drafts : s.proposals() + 1;
    uint32_t tokens[kVkMaxChainDrafts + 1];
    tokens[0] = a.token;
    for (uint32_t j = 0; j < n; ++j) tokens[j + 1] = drafts[j];
    cere_token_score scores[kVkMaxChainDrafts + 1] = {};
    cere_token_logprobs logprobs[kVkMaxChainDrafts + 1] = {};
    VkVerifyCall vc;
    vc.tokens = tokens;
    vc.n = n;
    vc.sampler = a.sampling;
    vc.logprobs = a.logprobs != nullptr;
    vc.out = scores;
    vc.out_logprobs = logprobs;
    bool launched = false;
    cere_status st = vk_verify(*ctx_, m, s, vc, &launched);
    if (st != CERE_OK) {
        if (launched) s.set_broken();  // MF-2
        faulted();
        row_status[index] = st;
        return st;
    }
    // The accept (cuda_device.cpp:2473-2496): k leading drafts equal the previous row's committed id (greedy: the raw
    // top-1; sampled: spec_accept's committed id), a = k + 1 tokens, truncated after the first stop token and at
    // max_commit.
    const auto ta = Clock::now();
    uint32_t k = 0;
    while (k < n && scores[k].id == tokens[k + 1]) ++k;
    uint32_t c = std::min(k + 1, a.max_commit);
    for (uint32_t i = 0; a.is_stop != nullptr && i < c; ++i) {
        if (a.is_stop(a.stop_ctx, scores[i].id)) c = i + 1;
    }
    host::SpecRowResult &o = out[index];
    o = host::SpecRowResult{};
    o.committed = c;
    o.accepted = std::min(c, k);
    std::copy(scores, scores + c, o.scores);
    if (a.logprobs != nullptr) std::copy(logprobs, logprobs + c, a.logprobs);
    uint32_t committed[kVkMaxChainDrafts + 1];
    for (uint32_t i = 0; i < c; ++i) committed[i] = scores[i].id;
    VkForwardStats::Spec &sp = m.forward_stats().spec;
    sp.host_accept_ms += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
    // Commit + draft-extend (cuda_device.cpp:2497-2552), one command buffer.
    VkCommitDraftCall cd;
    cd.n = n;
    cd.a = c;
    cd.committed = committed;
    cd.counted = a.sampling != nullptr && a.sampling->sampled;
    cd.key_rows = vc.key_rows;
    launched = false;
    st = vk_commit_draft(*ctx_, m, s, cd, &launched);
    if (st != CERE_OK) {
        // After the verify the session is undefined unless the commit ran (MF-2); its committed tokens stay reported.
        s.set_broken();
        faulted();
        row_status[index] = st;
        return st;
    }
    ++sp.rounds;
    sp.drafted += n;
    sp.accepted += o.accepted;
    sp.committed += c;
    ++sp.accepted_hist[std::min<uint32_t>(o.accepted, 15)];
    return CERE_OK;
}

}  // namespace cere::vkdev
