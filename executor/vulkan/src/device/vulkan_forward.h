// VulkanAmd P3 T5: the Vulkan text forward (vulkan-amd-p3-design design note sections 0, 2, 4, task T5).
//
// A NEW Vulkan copy of executor/cuda/src/model/forward.cpp's eager text path (OD-1: the forward is forked, not
// abstracted): it walks the same op list (model/plan.h plan::kDecode) in the same order with the same row-class (`cls`)
// arguments the CUDA forward passes on this path (cls 0 without a VkEndCall), and records the Vulkan kernels
// (executor/vulkan/src/kernels/*_vk.h) into ONE command buffer per decode step (and per short prefill), submitted once;
// a long prefill chunk (<= workspace_rows = 4,096 rows) is split into command buffers of kPrefillLayersPerSubmit layers
// (the amdgpu job timeout, below). Each command buffer ends with a completion canary the host checks. The host then
// reads the scored token's cere_token_score (and the logprobs entry) from the host-visible IO block
// (VkRegionKind::kIoTop2 / kIoLogprobs) and the K9 bound-fault word.
//
// Ported (cite: C = executor/cuda/src/model/forward.cpp): run_layers C:783-913 (plan walk), attention_rows C:310-445
// (the > 2,048 plan: index keys + compressed keys every call, K6 prologue, dense K6 while prefix + n <= 2,048, else K9),
// full_attention C:567-583 (eager), gdn C:591-615 (the k5::layer_forward fast path), moe C:678-688 (no taps),
// sample_row C:722-770 (K12 stage), forward C:2114-2331 (preconditions, ids staging, bound fault, position).
// NOT ported (refused by the device before anything runs, design 2.2): CUDA graphs and every device-position
// *_step / *_at variant, batched verify (attention_chain(s)), images / mRoPE, LoRA, an MTP layer under an EndCall
// (a drafting session's end states), MixedPD segments, taps / research hooks, multi-row decode batching
// (forward_decode_rows). D15 end states (VkEndCall) are ported: C:2123-2140 checks, the Tap cls / EndKv plumbing.
// T6 (below): verify rows (C:1575-1721), commit (C:2611-2664) and the MTP draft layer (mtp.cu) reuse the same Rec
// pieces (attention_rows, mix, combine, moe, the segment / canary submit).
//
// MXFP8 (S3: attention q|k|v / o_proj / indexer, all five GDN projections, the shared expert): the forward's own K2
// calls pass each VkDense's scales; the K5 layer and K7 block get them through vkk5::LayerWeights::*_scales and
// vkk7::MoeLayer::shared_*_scales (an internal benchmark report; a build without those fields
// fails a static_assert). lm_head, embed_tokens, router, K3 and K4 weights are BF16 (the binder checks each dtype).
#pragma once

#include "cere_executor.h"
#include "host/device.h"
#include "launch/vk_launch.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cere::vkdev {

class VulkanModel;
class VulkanSession;
struct VkSessionState;

// Policy (AGENTS.md 13).
// Count the K2 GEMM's exact-fallback rows (k2_gemm_vk.h: an x row FP16 x 2^e cannot carry exactly is recomputed on the
// FP32 route) per consumer on every prefill call of row class >= 17: one vkk2 row scan (the GEMM's own scan shader,
// via gemm_prepare_weight) per consumer input into the unused tap_routed workspace region, counted on the host after the
// submit. Evidence the spec asks for (spec P2 "facts that change the plan", design section 7). Decode (class <= 16, the
// GEMV) has no fallback route.
constexpr bool kCountExactFallback = true;
// A PREFILL call of at least kSegmentMinRows rows is submitted as several command buffers of kPrefillLayersPerSubmit
// layers each: one submission must finish within the amdgpu job timeout (measured: a 3,000-row prefill in ONE command
// buffer was aborted by the kernel driver's ring timeout and a GPU reset, an internal benchmark report).
// Short prompts stay one command buffer, decode steps one or two (kDecodeFirstSubmitLayers). Same kernels, same
// order: a split changes no output bit.
constexpr uint32_t kPrefillLayersPerSubmit = 4;
constexpr uint32_t kSegmentMinRows = 256;
// P4-G: segments of one segmented PREFILL call in flight at once (submitted, not yet retired): the host records and
// submits segment k + 1 while the GPU runs segment k, so the GPU does not idle between segments. Each command buffer
// keeps its own completion marker and canary; the amdgpu job timeout is per job, so kPrefillLayersPerSubmit sizes each
// command buffer as before (an internal benchmark report). The device creates its
// context with this many command-buffer slots. 1 = the P4-E serial order.
constexpr uint32_t kPrefillSubmitsInFlight = 2;
static_assert(kPrefillSubmitsInFlight >= 1 && kPrefillSubmitsInFlight <= vk::kMaxInFlight, "launch-layer slots");
// VulkanDecode (B1): a DECODE step is submitted as two command buffers, the first after its first
// kDecodeFirstSubmitLayers layers, both in flight: the GPU runs the first while the host records the rest (~0.6-1 ms
// of host recording per step no longer in front of the GPU). The first segment's GPU time (~3 ms) covers the rest's
// recording and submit. A recording refusal after the first submit breaks the session (*launched), as a segmented
// prefill's. Needs kPrefillSubmitsInFlight >= 2 (else one command buffer, as before).
constexpr uint32_t kDecodeFirstSubmitLayers = 4;
// VulkanDecode (B1): a DECODE call's last retire blocks until the previous decode step's GPU time (from the first
// submit's return) minus this margin, then polls the timeline (vk::Context::wait spin_from_ns): the host sees the GPU's
// end without the sleeping-thread wake-up. The margin covers the GPU starting before the submit call returns (~0.4
// ms, P4-G) and the step-to-step GPU time jitter; a later end just polls longer (at most vk::Context::kMaxSpinNs).
constexpr double kDecodeSpinMarginMs = 0.75;

// P4-G: host / GPU timing of the segments of segmented PREFILL calls (vk::SubmitTiming summed). gap: the GPU's idle
// time between two consecutive segments of one call (begin stamp - previous end stamp, GPU clock). The calibrated sums
// (start_after_submit, retire_after_end) exist only in the test build with CERE_VK_SUBMIT_TIMING=1.
struct VkSubmitBreakdown {
    uint64_t submits = 0, gaps = 0, calibrated = 0;
    double frame_gpu_ms = 0, max_frame_gpu_ms = 0, submit_call_ms = 0, wait_call_ms = 0, gap_ms = 0, max_gap_ms = 0;
    double start_after_submit_ms = 0, retire_after_end_ms = 0;
    void add(const vk::SubmitTiming &t, bool first_of_call) {
        ++submits;
        frame_gpu_ms += t.gpu_ms > 0 ? t.gpu_ms : 0;
        max_frame_gpu_ms = t.gpu_ms > max_frame_gpu_ms ? t.gpu_ms : max_frame_gpu_ms;
        submit_call_ms += t.submit_call_ms > 0 ? t.submit_call_ms : 0;
        wait_call_ms += t.wait_call_ms > 0 ? t.wait_call_ms : 0;
        if (!first_of_call && t.gpu_gap_before_ms >= 0) {
            ++gaps;
            gap_ms += t.gpu_gap_before_ms;
            max_gap_ms = t.gpu_gap_before_ms > max_gap_ms ? t.gpu_gap_before_ms : max_gap_ms;
        }
        if (t.start_after_submit_ms == t.start_after_submit_ms) {  // not NaN
            ++calibrated;
            start_after_submit_ms += t.start_after_submit_ms;
            retire_after_end_ms += t.retire_after_end_ms;
        }
    }
    void merge(const VkSubmitBreakdown &o) {
        submits += o.submits;
        gaps += o.gaps;
        calibrated += o.calibrated;
        frame_gpu_ms += o.frame_gpu_ms;
        max_frame_gpu_ms = o.max_frame_gpu_ms > max_frame_gpu_ms ? o.max_frame_gpu_ms : max_frame_gpu_ms;
        submit_call_ms += o.submit_call_ms;
        wait_call_ms += o.wait_call_ms;
        gap_ms += o.gap_ms;
        max_gap_ms = o.max_gap_ms > max_gap_ms ? o.max_gap_ms : max_gap_ms;
        start_after_submit_ms += o.start_after_submit_ms;
        retire_after_end_ms += o.retire_after_end_ms;
    }
};

// The K2 GEMM consumers whose x rows the counters scan (the GEMM's output route depends on its x row and w row; S3's
// weight rows are never flagged, spec P2).
enum class VkGemmConsumer : uint32_t {
    kHcDown = 0,      // K3 hc_mix down projection, x = normed [m, 4 H]
    kHcUp,            // K3 hc_mix up projection, x = act(t) [m, lowrank]
    kAttnQkvIndex,    // attention q|k|v and indexer index_qk, x = mixed [m, H]
    kAttnO,           // attention o_proj, x = attn_out [m, 6,144]
    kGdnIn,           // K5 in_proj_qkv / z / b / a, x = mixed [m, H]
    kGdnOut,          // K5 out_proj, x = y [m, 6,144]
    kMoeIn,           // K7 router, shared gate / up, x = mixed [m, H]
    kMoeSharedDown,   // K7 shared down, x = SwiGLU [m, S]
    kPleProj,         // K4 key_proj / value_proj, x = emb [m, 2,560]
    kCount
};
const char *vk_gemm_consumer_name(VkGemmConsumer c);

struct VkFallbackCounter {
    uint64_t rows = 0;     // x rows scanned (calls of class >= 17)
    uint64_t flagged = 0;  // rows the GEMM recomputes on the exact FP32 route (info bit 31: FB)
    uint64_t split = 0;    // P4-B: rows on the split route (info bits 31..30 = 01: matrix unit + exact tail chain)
    uint64_t scans = 0;
};

// P4-E: the coopmat prefill attention's exact-fallback counters (vkk6 / vkk9 Prefill::kCoopmat flag words: one per
// (token, KV head, head third); a flagged (token, KV head) is recomputed by the FP32 kernel), per kernel, counted on
// every counted PREFILL call (kCountExactFallback, class >= 17) that ran the coopmat path: the flags region is copied
// into the stats area after the attention and read on the host after the submit.
enum class VkAttnPath : uint32_t { kK6Dense = 0, kK9Sparse, kCount };
struct VkAttnFlagCounter {
    uint64_t calls = 0, rows = 0;     // attention calls / token rows (x 2 KV heads x 3 words each)
    uint64_t flagged_words = 0;       // non-zero flag words
    uint64_t flagged_pairs = 0;       // (token, KV head) pairs with a non-zero word: recomputed on the FP32 kernel
    uint64_t flagged_rows = 0;        // tokens with any flagged pair
};

// Test build only (CERE_VK_FORWARD_TAPS, env CERE_VK_FORWARD_PROFILE=1): GPU time of a segmented PREFILL call by
// forward op family (timestamps at op boundaries; every op ends with the forward's global barrier, so the stamps
// split serialized work). K2 GEMMs run inside the families that call them; "k2_scan" is the fallback counters' scans.
enum class VkProf : uint32_t {
    kOther = 0, kEmbed, kPle, kHcMix, kHcCombine, kAttnQkv, kAttnIndexStore, kAttnPrologue, kK6Dense, kK9IndexerQ,
    kK9Select, kK9Attn, kK9Gate, kAttnO, kGdn, kMoe, kFinal, kK2Scan, kFlagCopy, kCount
};
const char *vk_prof_name(VkProf p);

// Per-model forward statistics (the device reports them; forward_s3 prints them).
struct VkForwardStats {
    uint64_t calls = 0, prefill_calls = 0, decode_calls = 0, tokens = 0;
    // Last call: wall-clock split (host recording, submit + wait) and the GPU time between its first and last
    // timestamp; launch-layer commands recorded (dispatches + barriers + stamps, Context::device_commands delta) and the
    // forward's own barrier count; table-ring bytes used.
    double last_record_ms = 0, last_wait_ms = 0, last_gpu_ms = 0, last_read_ms = 0;
    uint64_t last_commands = 0, last_barriers = 0, last_ring_bytes = 0, last_rows = 0, last_submits = 0;
    uint32_t last_operation = 0;
    // Sums over all calls of each operation.
    double prefill_record_ms = 0, prefill_gpu_ms = 0, prefill_wall_ms = 0;
    double decode_record_ms = 0, decode_gpu_ms = 0, decode_wall_ms = 0;
    VkFallbackCounter fallback[uint32_t(VkGemmConsumer::kCount)];
    uint64_t fallback_dropped = 0;  // scans skipped because the stats area was full
    // P4-E
    VkAttnFlagCounter attn_flags[uint32_t(VkAttnPath::kCount)];
    uint64_t prefill_coopmat_calls = 0, prefill_fp32_calls = 0;  // PREFILL calls by the attention path they ran
    double prefill_max_submit_gpu_ms = 0, last_max_submit_gpu_ms = 0;  // the longest command buffer (stamps 0..1)
    VkSubmitBreakdown prefill_submits;  // P4-G: segmented PREFILL calls' segments
    uint32_t last_depth = 1;            // P4-G: segments in flight of the last call (1: serial)
    uint32_t prefill_last_depth = 0;    // P4-G: the same, of the last segmented PREFILL call
    // VulkanDecode: DECODE calls' segments (the GPU idle between a split step's two command buffers) and, calibrated
    // (test build, CERE_VK_SUBMIT_TIMING=1), the host return of each step's LAST wait after its GPU end stamp.
    VkSubmitBreakdown decode_submits;
    uint64_t decode_calibrated = 0;
    double decode_retire_after_end_ms = 0;
    // P4-G: host time after the last retire (bound fault, scored row, counters added; inside prefill_wall_ms) and the
    // counters' host summing at segment retires (inside the wait / open_segment time, overlapped with the GPU).
    double prefill_read_ms = 0, prefill_count_ms = 0;
    double prof_ms[uint32_t(VkProf::kCount)] = {};  // test build only (above)
    uint64_t prof_segments = 0, prof_dropped = 0;
    // T6: speculation rounds and prompt extends.
    struct Spec {
        uint64_t rounds = 0, drafted = 0, accepted = 0, committed = 0;
        uint64_t accepted_hist[16] = {};  // rounds by accepted drafts
        uint64_t extends = 0, extend_rows = 0;
        // GPU (stamps) / host recording / wall (record + submit-wait + read) per submit kind; host_accept_ms: the
        // accept decision between the verify and the commit + draft submit.
        double verify_record_ms = 0, verify_gpu_ms = 0, verify_wall_ms = 0;
        double restore_gpu_ms = 0, draft_gpu_ms = 0, cd_record_ms = 0, cd_gpu_ms = 0, cd_wall_ms = 0;
        double host_accept_ms = 0;
        double extend_gpu_ms = 0, extend_wall_ms = 0;
        double extend_prof_ms[uint32_t(VkProf::kCount)] = {};  // test build only: the extends by op family (VkProf)
        uint64_t last_verify_commands = 0, last_verify_barriers = 0, last_cd_commands = 0, last_cd_barriers = 0;
        double max_verify_gpu_ms = 0, max_cd_gpu_ms = 0, max_extend_gpu_ms = 0;
        // Undo restores (a <= n), those that also restored a completed QSA key row, verify rows on the K9 sparse path.
        uint64_t restores = 0, key_restores = 0, sparse_rows = 0;
    } spec;
};

// One PREFILL call's end-state work (D15; model.h EndCall :706-717, restated for addresses): through views of state
// slot pairs (vk_bind_fixed_state). `cls` (0 = n) is the row count of the cold call these rows belong to: every
// row-count choice of the call is made for it. Capture (`row` > 0: a multiple of 64 strictly inside the call): the
// state as of `row` goes to the pair `a`, `b` (GDN conv, FP32 SSM heads 0..23 in a's SSM regions and 24..47 in b's,
// PLE history and conv state; the caller copied the rest at the call start). Resume (`in_a`, `in_b` non-null): the GDN
// chunk rule starts from that pair's FP32 SSM. A chunk-0 end state's BF16 K/V run (`kv`, `in_kv`: per full-attention
// layer ascending, K then V [rows, 512]): a capture writes the rows [0, position + row) there, a resume attends over
// the saved rows [0, position) then its own (the dense K6 path, reading BF16 as the cold call did).
struct VkEndCall {
    uint32_t cls = 0;
    int64_t row = -1;
    const VkSessionState *a = nullptr, *b = nullptr;
    const VkSessionState *in_a = nullptr, *in_b = nullptr;
    uint64_t kv = 0;  // device address, 0 = none
    int64_t kv_rows = 0;
    uint64_t in_kv = 0;
    int64_t in_kv_rows = 0;
};

// One forward call (forward.cpp:2114 forward's arguments on the first-light path, plus the EndCall).
struct VkForwardCall {
    uint32_t operation = 0;               // CERE_OP_PREFILL (any n <= workspace_rows) or CERE_OP_DECODE (n = 1)
    const uint32_t *tokens = nullptr;
    uint32_t n = 0;
    const host::ItemSampling *sampler = nullptr;  // null: greedy top-2 on the raw logits (C:887-896)
    cere_token_score *out = nullptr;      // the scored (last) row's top-2 / committed score
    cere_token_logprobs *out_logprobs = nullptr;  // non-null only with sampler
    const VkEndCall *end = nullptr;       // PREFILL only; null = cls 0, no end state
};
// T6: on a model with drafts every forward call (as CUDA, C:876-878: ws.hc_rows exists exactly with drafts) also
// copies the HC rows before the final mixer (SGLang's hc_hidden_states, the MTP extend input) into
// VkWorkspace::mtp_hc_rows; the copy changes no output of the call.

// ---- T6: speculation (vulkan-amd-p3-design design note task T6; CUDA forward.cpp run_verify C:1575-1721,
// forward_verify_rows C:2447-2609, verify_commit C:2611-2664, mtp.cu run :271-462, cuda_device.cpp draft :1433-1499)

// The most drafts of one chain on Vulkan: layout::verify_rows(max_sessions 1) - 1 = 3 (= the undo snapshots of one
// session, layout::kUndoSnapshots).
constexpr uint32_t kVkMaxChainDrafts = 3;

// One verify call of one session (one chain): row 0 consumes tokens[0] (the last committed, unfed token) at the
// session position p, rows 1..n the drafts tokens[1..n] at p+1..p+n, all rows in ONE command buffer. Each row's state
// mutations are saved in-kernel / copied into the session's undo snapshots (undo[j - 1] = the state after rows
// 0..j-1, taken before row j mutates it). The session position stays p. out[0..n]: each row's score (greedy: the raw
// top-2, bit-equal to plain decode's at p + j when rows 0..j-1 are what plain decode committed; sampled: phase S,
// k13s::spec_accept's committed id and its score). out_logprobs[0..n] when `logprobs` (needs sampler). key_rows[j-1]
// is written for the commit.
struct VkVerifyCall {
    const uint32_t *tokens = nullptr;  // [n + 1]
    uint32_t n = 0;                    // drafts, 0..kVkMaxChainDrafts
    const host::ItemSampling *sampler = nullptr;
    bool logprobs = false;
    cere_token_score *out = nullptr;              // [n + 1]
    cere_token_logprobs *out_logprobs = nullptr;  // [n + 1] when logprobs
    int64_t key_rows[kVkMaxChainDrafts] = {-1, -1, -1};
};
cere_status vk_verify(vk::Context &ctx, VulkanModel &m, VulkanSession &s, VkVerifyCall &call, bool *launched);

// The commit of a verified chain plus the next draft, ONE command buffer (C:2611-2664 verify_commit, then
// cuda_device.cpp speculate_group's draft-extend :2510-2546 and draft :1433-1499): when a <= n the state after rows
// 0..a-1 is restored from undo[a - 1] (GDN conv / SSM, PLE history / conv, QSA ring and the completed key row),
// a sampled chain's committed ids enter out_counts, the position advances by a; then the MTP extend over the a
// committed rows (the verify's hc rows 0..a-1, next ids = the committed tokens) seeded at row a - 1 and up to
// layout::kDraftSteps one-row MTP decode steps: the session's next proposals (VulkanSession::proposals).
struct VkCommitDraftCall {
    uint32_t n = 0;                       // the chain's drafts (the verify call's n)
    uint32_t a = 0;                       // committed tokens, 1..n + 1
    const uint32_t *committed = nullptr;  // [a] the committed ids
    bool counted = false;                 // a sampled chain: the committed ids enter out_counts
    const int64_t *key_rows = nullptr;    // the verify call's key_rows
};
cere_status vk_commit_draft(vk::Context &ctx, VulkanModel &m, VulkanSession &s, const VkCommitDraftCall &call,
                            bool *launched);

// The MTP extend of a drafting session's PREFILL step (cuda_device.cpp extend_draft :1410-1431 + draft): n rows at
// position - n .. position - 1 (the forward just consumed them; their hc rows are in VkWorkspace::mtp_hc_rows), next
// ids = tokens[1..n-1] then `next` (the scored token, or the prompt's next token while it stays open); then the MTP
// decode steps unless the prompt stays open (layout::draft_decode_steps). Its own command buffer.
cere_status vk_extend_draft(vk::Context &ctx, VulkanModel &m, VulkanSession &s, const uint32_t *tokens, uint32_t n,
                            uint32_t next, bool prompt_open, bool *launched);

// What the MXFP8 consumer wiring of this build is: "k5:wired k7:wired" or with "unwired" (compile-time detection of
// the scale fields in vkk5::LayerWeights / vkk7::MoeLayer).
const char *vk_forward_mxfp8_wiring();
// True when every MXFP8 dense weight of `m` can reach its kernel with its scales (false: some K5 / K7 MXFP8 weight
// would be read as BF16; *why names it).
bool vk_forward_mxfp8_ok(const VulkanModel &m, std::string *why);

// The forward call: preconditions first (CERE_ERR_ARGUMENT / CAPACITY / UNSUPPORTED, `*launched` false, nothing
// recorded, the session untouched); then one command buffer, one submit, the IO read-back. On success the session's
// position advances by n and *out (and *out_logprobs) hold the scored row's result. A failure with `*launched` true
// leaves the session's state undefined (the caller breaks it, MF-2); a device loss faults the Context (sticky).
cere_status vk_forward(vk::Context &ctx, VulkanModel &m, VulkanSession &s, const VkForwardCall &call, bool *launched);

// The sampler preconditions (forward.cpp:1952-1964 check_sampler, restated: model.h is CUDA-typed).
cere_status vk_check_sampler(const host::ItemSampling &s, uint32_t vocab);

}  // namespace cere::vkdev
