// VulkanAmd P3 T5: the Vulkan text forward. See vulkan_forward.h. Every ported block cites its CUDA counterpart
// (C = executor/cuda/src/model/forward.cpp, M = executor/cuda/src/model/model.cpp) so a reviewer can diff intent.
#include "device/vulkan_forward.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "device/vulkan_device.h"
#include "kernels/k12_sampler_vk.h"
#include "kernels/k13_moe_bf16_vk.h"
#include "kernels/k13s_spec_accept_vk.h"
#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"
#include "kernels/k3_hc_vk.h"
#include "kernels/k4_ple_vk.h"
#include "kernels/k5_gdn_chunk_vk.h"
#include "kernels/k5_gdn_chain_vk.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "kernels/k6_attention_vk.h"
#include "kernels/k6_prologue_vk.h"
#include "kernels/k7_moe_vk.h"
#include "kernels/k9_qsa_vk.h"
#include "kernels/mtp_kernels_vk.h"
#include "model/layout.h"
#include "model/plan.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace cere::vkdev {

namespace {

namespace plan = cere::model::plan;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
constexpr uint64_t a256(uint64_t b) { return (b + 255) / 256 * 256; }
constexpr uint32_t kGemmClass = vkk2::kGemmMinM;  // row classes >= this run the K2 GEMM (the fallback counters' scope)

#define VK_TRY(expr)                    \
    do {                                \
        const cere_status s_ = (expr);  \
        if (s_ != CERE_OK) return s_;   \
    } while (0)

// ---- MXFP8 consumer wiring (compile-time detection of the kernels' scale fields) ------------------------------------
// The S3 artifact's GDN projections and shared-expert projections are MXFP8 (vulkan_bind.h VkDense::scales). The K5
// layer / K7 block take their scales through vkk5::LayerWeights::*_scales / vkk7::MoeLayer::shared_*_scales
// (an internal benchmark report).
template <class W>
constexpr auto gdn_wired(int) -> decltype(std::declval<W &>().in_proj_qkv_scales = 0,
                                          std::declval<W &>().in_proj_z_scales = 0,
                                          std::declval<W &>().in_proj_b_scales = 0,
                                          std::declval<W &>().in_proj_a_scales = 0,
                                          std::declval<W &>().out_proj_scales = 0, bool()) {
    return true;
}
template <class W>
constexpr bool gdn_wired(long) {
    return false;
}
template <class W>
constexpr auto moe_wired(int) -> decltype(std::declval<W &>().shared_gate_scales = 0,
                                          std::declval<W &>().shared_up_scales = 0,
                                          std::declval<W &>().shared_down_scales = 0, bool()) {
    return true;
}
template <class W>
constexpr bool moe_wired(long) {
    return false;
}
constexpr bool kGdnWired = gdn_wired<vkk5::LayerWeights>(0);
constexpr bool kMoeWired = moe_wired<vkk7::MoeLayer>(0);
// The MXFP8 consumer work landed (an internal benchmark report): a kernel header without the
// scale fields would make the forward read E4M3 bytes as BF16, so it no longer builds.
static_assert(kGdnWired && kMoeWired, "vkk5::LayerWeights / vkk7::MoeLayer must carry the MXFP8 scale fields");

template <class W>
void set_gdn_scales(W &g, const VkDense *d) {
    if constexpr (gdn_wired<W>(0)) {
        g.in_proj_qkv_scales = d[0].scales;
        g.in_proj_z_scales = d[1].scales;
        g.in_proj_b_scales = d[2].scales;
        g.in_proj_a_scales = d[3].scales;
        g.out_proj_scales = d[4].scales;
    } else {
        (void)g;
        (void)d;  // unreachable: static_assert above
    }
}
template <class W>
void set_moe_scales(W &l, const VkDense *d) {
    if constexpr (moe_wired<W>(0)) {
        l.shared_gate_scales = d[0].scales;
        l.shared_up_scales = d[1].scales;
        l.shared_down_scales = d[2].scales;
    } else {
        (void)l;
        (void)d;  // unreachable: static_assert above
    }
}

// K7's workspace carve (k7_moe_vk.cpp `carve`, restated: the shared-expert SwiGLU product s_h is the shared down
// projection's x, read back by the fallback counter after moe_block). Offsets only; nothing is written through it.
uint64_t k7_s_h_offset(const vkk7::MoeDims &d, uint64_t m) {
    const uint64_t r = m * d.top_k;
    return a256(m * d.num_experts * 2) + a256(r * 4) + a256(r * 4) + a256(m * d.hidden / 2) + a256(m * d.hidden / 16) +
           a256(m * 4) + a256((2ull * d.num_experts + 2) * 4) + a256(r * 4) + a256(r * 4) + a256(r * d.inter * 2) +
           a256(r * d.inter / 2) + a256(r * d.inter / 16) + a256(r * d.hidden * 2) + a256(m * d.shared_inter * 2) +
           a256(m * d.shared_inter * 2);
}

// The K12 stage's buffers (model.cpp:345-358 carve_sampler, from the start of the shared scratch: the sampler runs
// after every layer, when no kernel holds the scratch). bias / mask / logprobs live in the IO block instead (host
// written / read through its mapping), so only the device-only buffers are carved; the order and alignment are
// model.cpp's (the memory plan sized the region with the full carve).
struct SamplerScratch {
    uint64_t probs, keep, lse, scratch, top2_ids, top2_scores, drawn;
};
SamplerScratch carve_sampler(uint64_t base, uint64_t vocab) {
    SamplerScratch s{};
    uint64_t at = 0;
    const auto take = [&](uint64_t bytes) {
        at = a256(at);
        const uint64_t p = base + at;
        at += bytes;
        return p;
    };
    s.probs = take(vocab * 4);
    s.keep = take((vocab + 31) / 32 * 4);
    s.lse = take(8);
    s.scratch = take(vkk12::scratch_bytes());
    s.top2_ids = take(8);
    s.top2_scores = take(8);
    s.drawn = take(4);
    return s;
}

// T6: the batched lm_head rows of a verify call (model.cpp:362-370 rows_logits): BF16 [rows <= 16, vocab] after the
// full sampler carve (carve_sampler above plus bias, logprobs and mask, model.cpp:345-358: the memory plan sized the
// scratch entry "K12 sampler stage + rows lm_head" this way).
uint64_t rows_logits_addr(uint64_t base, uint64_t vocab) {
    const uint64_t words = (vocab + 31) / 32;
    uint64_t at = 0;
    for (const uint64_t b : {vocab * 4, words * 4, uint64_t(8), uint64_t(vkk12::scratch_bytes()), uint64_t(8),
                             uint64_t(8), uint64_t(4), vocab * uint64_t(sizeof(cere_logit_bias)),
                             uint64_t(sizeof(cere_token_logprobs)), words * 4}) {
        at = a256(at) + b;
    }
    return base + a256(at);
}

// T6: model.h SpecSampler (model.cpp:536-559 carve_spec_sampler) in the kSpecSampler region, verify_rows rows.
struct SpecRegions {
    uint64_t proc, probs, keep, top2_ids, top2_scores, committed, count, counts, accept;
};
SpecRegions carve_spec(uint64_t base, uint64_t rows, uint64_t vocab) {
    const uint64_t words = (vocab + 31) / 32;
    SpecRegions r{};
    uint64_t at = 0;
    const auto take = [&](uint64_t bytes) {
        at = a256(at);
        const uint64_t p = base + at;
        at += bytes;
        return p;
    };
    r.proc = take(rows * vocab * 4);
    r.probs = take(rows * vocab * 4);
    r.keep = take(rows * words * 4);
    r.top2_ids = take(rows * 2 * 4);
    r.top2_scores = take(rows * 2 * 4);
    r.committed = take(rows * 4);
    r.count = take(rows * 4);
    r.counts = take(vocab * 4);
    r.accept = take(vkk13s::scratch_bytes(kVkMaxChainDrafts));
    return r;
}

// P4-G: copy from host-visible device memory, which the host maps write-combined (uncached) on this UMA device: plain
// loads read it at ~86 MB/s (measured, report P4-G), SSE4.1 streaming loads (movntdqa) fetch whole lines. x86-64 only;
// elsewhere, and for an unaligned head / tail, memcpy.
#if defined(__x86_64__)
__attribute__((target("sse4.1"))) void copy_from_device(void *dst, const void *src, size_t bytes) {
    const uintptr_t s0 = reinterpret_cast<uintptr_t>(src);
    const size_t head = std::min(bytes, size_t((16 - s0 % 16) % 16));
    std::memcpy(dst, src, head);
    const size_t body = (bytes - head) / 64 * 64;
    const uint8_t *s = static_cast<const uint8_t *>(src) + head;
    uint8_t *d = static_cast<uint8_t *>(dst) + head;
    for (size_t i = 0; i < body; i += 64) {
        __m128i *p = reinterpret_cast<__m128i *>(const_cast<uint8_t *>(s + i));
        const __m128i a = _mm_stream_load_si128(p), b = _mm_stream_load_si128(p + 1);
        const __m128i c = _mm_stream_load_si128(p + 2), e = _mm_stream_load_si128(p + 3);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d + i), a);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d + i + 16), b);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d + i + 32), c);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d + i + 48), e);
    }
    std::memcpy(d + body, s + body, bytes - head - body);
}
#else
void copy_from_device(void *dst, const void *src, size_t bytes) { std::memcpy(dst, src, bytes); }
#endif

// One forward call's recording state.
struct Rec {
    vk::Context &ctx;
    VulkanModel &m;
    VulkanSession &s;
    const VkModelWeights &w;
    const VkWorkspace &ws;
    const cere_qwen4_exp_config &c;
    uint32_t n;
    uint32_t op;
    uint64_t prefix;
    uint64_t barriers = 0;
    // D15: the call's row class (0 = n; every row-count choice is made for it) and its EndCall (null = none)
    uint32_t cls = 0;
    const VkEndCall *end = nullptr;
    // IO block
    VulkanModel::IoRegion io_ids32, io_ids64, io_top2, io_logprobs, io_fault, io_bias, io_mask;
    // fallback counters
    bool counting = false;
    uint64_t stats_addr = 0, stats_words = 0, stats_used = 0;
    struct Scan {
        VkGemmConsumer who;
        uint64_t word;
        uint32_t rows;
    };
    std::vector<Scan> scans;
    uint64_t dropped = 0;
    // P4-E: the coopmat prefill attention (VulkanModel::prefill_coopmat) and its flag copies in the stats area
    bool coop = false;
    struct FlagCopy {
        VkAttnPath path;
        uint64_t word;
        uint32_t rows;
    };
    std::vector<FlagCopy> flag_copies;
    double max_submit_ms = 0;
    // test-build profile (VkProf): stamp k >= 2 opens an interval of family prof_fam[k], closed by stamp k + 1 or 1
    bool prof = false;
    uint32_t prof_n = 2;
    VkProf prof_fam[vk::Context::kStamps] = {};
    double prof_ms[uint32_t(VkProf::kCount)] = {};
    uint64_t prof_segments = 0, prof_dropped = 0;
    vkk3::HcPlans hc_plans{};
    vkk3::HcShape hc_shape{};

    cere_status bar() {
        ++barriers;
        return ctx.barrier();
    }
    // Submission segments (vulkan_forward.h kPrefillLayersPerSubmit, kDecodeFirstSubmitLayers): every segment is one
    // command buffer, closed by a completion canary the host checks after the wait. P4-G: up to `depth` segments are in
    // flight (submitted, not yet retired; kPrefillSubmitsInFlight for a segmented call, 2 for a split DECODE call, else
    // 1); open_segment retires the oldest when the queue is full, the call's end retires all (drain). Each in-flight
    // segment has its own canary words (canary_at).
    bool *launched = nullptr;
    bool segmented = false;
    uint32_t split_at = 0;  // a split DECODE call: layers of its first segment (kDecodeFirstSubmitLayers), else 0
    uint32_t depth = 1;
    uint32_t layers_in_segment = 0, submits = 0, issued = 0;
    uint64_t canary_seq = 0, ring_max = 0;
    double record_ms = 0, wait_ms = 0, gpu_ms = 0;
    // A DECODE call: the previous decode step's GPU time (0: none), and when its first submit call returned; the
    // call's last retire polls from the expected GPU end minus kDecodeSpinMarginMs (vk::Context::wait spin_from_ns).
    double expect_gpu_ms = 0;
    Clock::time_point first_submit{};
    double last_retire_after_end_ms = vk::kNaN;  // the last retired segment's (calibrated timing; else NaN)
    Clock::time_point seg_start{};
    VulkanModel::IoRegion io_canary;
    struct Pending {
        uint64_t ticket = 0, canary = 0;
        uint32_t canary_index = 0, prof_n = 2;
        size_t scans_end = 0, flags_end = 0;  // this segment's counter scans / flag copies: [.._done, .._end)
        VkProf prof_fam[vk::Context::kStamps];
    };
    // P4-G: the counters of retired segments, summed on the host when each segment retires (while the next one runs)
    // from ONE bulk copy of the segment's stats words: per-word reads of the uncached device memory cost ~70 us per
    // prompt row (an internal benchmark report). vk_forward adds them to the stats.
    size_t scans_done = 0, flags_done = 0;
    VkFallbackCounter fb_acc[uint32_t(VkGemmConsumer::kCount)];
    VkAttnFlagCounter attn_acc[uint32_t(VkAttnPath::kCount)];
    std::vector<uint32_t> stats_host;
    double count_ms = 0;
    Pending pending[vk::kMaxInFlight];
    uint32_t pending_head = 0, pending_count = 0;
    VkSubmitBreakdown sub{};  // this call's segments (vulkan_forward.h)

    uint8_t *canary_at(uint32_t index) const { return io_canary.host + 32ull * index; }
    cere_status open_segment() {
        if (pending_count >= depth) VK_TRY(retire_oldest());
        VK_TRY(ctx.begin());
        seg_start = Clock::now();
        prof_n = 2;
        return ctx.stamp(0);
    }
    // Retires the oldest in-flight segment: the launch layer's wait (its completion marker, in submission order), then
    // this segment's canary, GPU time and profile stamps (the stamps of the last retired command buffer).
    cere_status retire_oldest(int64_t spin_from_ns = -1) {
        Pending &q = pending[pending_head];
        const auto t1 = Clock::now();
        const cere_status st = ctx.wait(q.ticket, spin_from_ns);
        wait_ms += ms_since(t1);
        if (st != CERE_OK) return ctx.faulted() ? CERE_ERR_INTERNAL : st;
        pending_head = (pending_head + 1) % vk::kMaxInFlight;
        --pending_count;
        uint64_t seen = 0;
        std::memcpy(&seen, canary_at(q.canary_index) + 16, 8);
        if (seen != q.canary) {
            std::fprintf(stderr, "cere: vk_forward submission %u did not complete (canary %llu, expected %llu): the "
                                 "kernel driver aborted it (job timeout / GPU reset)\n",
                         submits, (unsigned long long)seen, (unsigned long long)q.canary);
            return CERE_ERR_INTERNAL;
        }
        const double seg_ms = ctx.stamp_ms(0, 1);
        gpu_ms += seg_ms;
        max_submit_ms = std::max(max_submit_ms, seg_ms);
        if (prof) {
            prof_ms[uint32_t(VkProf::kOther)] += ctx.stamp_ms(0, q.prof_n > 2 ? 2 : 1);
            for (uint32_t k = 2; k < q.prof_n; ++k)
                prof_ms[uint32_t(q.prof_fam[k])] += ctx.stamp_ms(k, k + 1 < q.prof_n ? k + 1 : 1);
            ++prof_segments;
        }
        const vk::SubmitTiming &t = ctx.last_submit_timing();
        sub.add(t, submits == 0);
        last_retire_after_end_ms = t.retire_after_end_ms;
        ++submits;
        count_through(q.scans_end, q.flags_end);
        return CERE_OK;
    }
    // Sums the scans / flag copies up to (scans_end, flags_end), recorded in segments that have retired: their words
    // in the stats area are final. Same arithmetic as the P4-E end-of-call loop.
    void count_through(size_t scans_end, size_t flags_end) {
        if (!counting || (scans_done >= scans_end && flags_done >= flags_end)) return;
        const auto t0 = Clock::now();
        uint64_t lo = UINT64_MAX, hi = 0;
        for (size_t i = scans_done; i < scans_end; ++i) {
            lo = std::min<uint64_t>(lo, scans[i].word);
            hi = std::max<uint64_t>(hi, scans[i].word + scans[i].rows);
        }
        for (size_t i = flags_done; i < flags_end; ++i) {
            lo = std::min<uint64_t>(lo, flag_copies[i].word);
            hi = std::max<uint64_t>(hi, flag_copies[i].word + uint64_t(flag_copies[i].rows) * 6);
        }
        const uint8_t *base = m.host_of(stats_addr);
        if (base != nullptr && hi > lo) {
            stats_host.resize(hi - lo);
            copy_from_device(stats_host.data(), base + lo * 4, (hi - lo) * 4);  // one streaming read of the range
        }
        const auto word = [&](uint64_t w) { return base != nullptr ? stats_host[w - lo] : 0u; };
        for (; scans_done < scans_end; ++scans_done) {
            const Scan &sc = scans[scans_done];
            VkFallbackCounter &k = fb_acc[uint32_t(sc.who)];
            uint64_t flagged = 0, split = 0;
            for (uint32_t i = 0; i < sc.rows && base != nullptr; ++i) {
                const uint32_t v = word(sc.word + i);
                flagged += (v >> 31) & 1;
                split += (v >> 30) == 1u;  // P4-B: SPLIT (bit 30 without bit 31)
            }
            k.rows += sc.rows;
            k.flagged += flagged;
            k.split += split;
            ++k.scans;
        }
        for (; flags_done < flags_end; ++flags_done) {
            const FlagCopy &fc = flag_copies[flags_done];
            VkAttnFlagCounter &a = attn_acc[uint32_t(fc.path)];
            ++a.calls;
            a.rows += fc.rows;
            for (uint32_t t = 0; t < fc.rows && base != nullptr; ++t) {
                bool any_row = false;
                for (uint32_t g = 0; g < 2; ++g) {
                    bool any_pair = false;
                    for (uint32_t h = 0; h < 3; ++h) {
                        const uint32_t v = word(fc.word + (uint64_t(t) * 2 + g) * 3 + h);
                        a.flagged_words += v != 0;
                        any_pair = any_pair || v != 0;
                    }
                    a.flagged_pairs += any_pair;
                    any_row = any_row || any_pair;
                }
                a.flagged_rows += any_row;
            }
        }
        count_ms += ms_since(t0);
    }
    // Retires every in-flight segment (the call's end, and every error path after a submit). The first error is
    // returned; after a fault the rest cannot be waited (the launch layer refuses) and stay unverified.
    cere_status drain() {
        cere_status first = CERE_OK;
        while (pending_count > 0) {
            const uint32_t before = pending_count;
            int64_t spin_from_ns = -1;
            if (pending_count == 1 && expect_gpu_ms > 0) {
                const auto at = first_submit + std::chrono::duration_cast<Clock::duration>(
                                                   std::chrono::duration<double, std::milli>(expect_gpu_ms - kDecodeSpinMarginMs));
                spin_from_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count();
            }
            const cere_status st = retire_oldest(spin_from_ns);
            if (st == CERE_OK) continue;
            if (first == CERE_OK) first = st;
            // a canary mismatch (the entry was retired) keeps going; a fault or a wait refusal stops
            if (ctx.faulted() || pending_count == before) break;
        }
        return first;
    }
    // Test-build profile mark: the work recorded from here to the next mark (or the segment's end) is family f.
    void mark(VkProf f) {
#if defined(CERE_VK_FORWARD_TAPS)
        if (!prof) return;
        if (prof_n + 1 >= vk::Context::kStamps) {
            ++prof_dropped;
            return;
        }
        prof_fam[prof_n] = f;
        if (ctx.stamp(prof_n) == CERE_OK) ++prof_n;
#else
        (void)f;
#endif
    }
    // Ends the open command buffer with the completion canary (16 bytes copied from io_canary[0, 16) to [16, 32): the
    // last command) and submits it. A submission the kernel driver aborted (its job timeout, a GPU reset) can still
    // signal the timeline semaphore without an error (RADV reports the lost context only at the next submit), so a
    // missing canary (checked at retire_oldest, after the launch layer's own marker) is a device failure: INTERNAL.
    // P4-G: submits without waiting; the segment joins the in-flight queue (open_segment / drain retire it). With depth
    // 1 every segment is retired before the next begin: the previous serial record -> submit + wait -> record order.
    cere_status submit_segment() {
        VK_TRY(ctx.stamp(1));
        VK_TRY(bar());
        ++canary_seq;
        const uint32_t ci = issued % vk::kMaxInFlight;  // distinct from every other in-flight segment's (depth <= max)
        uint8_t *cw = canary_at(ci);
        std::memcpy(cw, &canary_seq, 8);
        std::memset(cw + 16, 0, 16);
        const uint64_t ca = io_canary.addr + 32ull * ci;
        vkk1::CopySegment seg{ca + 16, 16, ca, 16, 16, 1};
        vk::TableRegion t{};
        VK_TRY(ctx.table_alloc(sizeof seg, 16, &t));
        vk::Buffer tb = t.view();
        VK_TRY(vkk1::copy_segments(ctx, &seg, 1, tb));
        VK_TRY(bar());
        ring_max = std::max<uint64_t>(ring_max, ctx.table_ring_used());
        record_ms += ms_since(seg_start);
        const auto t1 = Clock::now();
        *launched = true;
        uint64_t ticket = 0;
        const cere_status st = ctx.submit(&ticket);
        wait_ms += ms_since(t1);
        if (st != CERE_OK) {
            if (ctx.faulted()) return CERE_ERR_INTERNAL;
            if (issued == 0) *launched = false;  // a latched recording error of the first segment: nothing ran
            return st;
        }
        Pending &q = pending[(pending_head + pending_count) % vk::kMaxInFlight];
        q.ticket = ticket;
        q.canary = canary_seq;
        q.canary_index = ci;
        q.prof_n = prof_n;
        q.scans_end = scans.size();
        q.flags_end = flag_copies.size();
        if (prof) std::copy(prof_fam, prof_fam + prof_n, q.prof_fam);
        ++pending_count;
        if (issued == 0) first_submit = Clock::now();
        ++issued;
        return CERE_OK;
    }
    // After each layer: a segmented (long PREFILL) call closes the segment every kPrefillLayersPerSubmit layers; a
    // split DECODE call closes its first segment after split_at layers, once.
    cere_status layer_done() {
        if (segmented) {
            if (++layers_in_segment < kPrefillLayersPerSubmit) return CERE_OK;
            layers_in_segment = 0;
        } else if (split_at == 0 || issued > 0 || ++layers_in_segment < split_at) {
            return CERE_OK;
        }
        VK_TRY(submit_segment());
        return open_segment();
    }
    // The consumers' x-row info words (vkk2::gemm's own scan of its x, SPLIT rows with k2_gemm tail_split; a row's
    // info word has bit 31 set when the GEMM recomputes the row's outputs on the FP32 route) into the stats area.
    // arm(), before a consumer: reserves the consumer's words and arms a K2 x-info tap (vk::Context::arm_k2_info), so
    // the consumer's first GEMM on exactly that x writes its info words there (no second read of x). tapped(), after
    // the consumer and its barrier: a tap no GEMM took (no GEMM on that x, m, k) gets the same scan recorded now
    // (vkk2::gemm_scan_rows), then a barrier before the next kernel may overwrite the scanned input (WAR).
    struct Armed {
        VkGemmConsumer who;
        uint64_t x, word;
        uint32_t rows, k;
    };
    Armed armed[vk::Context::kK2InfoTaps];
    uint32_t armed_n = 0;
    void arm(VkGemmConsumer who, uint64_t x, uint32_t rows, uint32_t k) {
        if (!counting || rows < kGemmClass) return;
        if (stats_used + rows > stats_words || armed_n == vk::Context::kK2InfoTaps) {
            ++dropped;
            return;
        }
        armed[armed_n] = Armed{who, x, stats_used, rows, k};
        ctx.arm_k2_info(armed_n++, x, rows, k, stats_addr + stats_used * 4);
        stats_used += (rows + 3) / 4 * 4;  // keep each consumer's words 16-byte aligned
    }
    cere_status tapped() {
        bool rescanned = false;
        for (uint32_t i = 0; i < armed_n; ++i) {
            const Armed &a = armed[i];
            if (!ctx.k2_info_hit(i)) {
                mark(VkProf::kK2Scan);
                if (vkk2::gemm_scan_rows(ctx, a.x, a.rows, a.k, stats_addr + a.word * 4) != CERE_OK) {
                    ++dropped;
                    continue;
                }
                rescanned = true;
            }
            scans.push_back(Scan{a.who, a.word, a.rows});
        }
        armed_n = 0;
        ctx.clear_k2_info();
        return rescanned ? bar() : CERE_OK;
    }
    // P4-E: copy a coopmat attention call's flag words ([rows, 2 KV heads, 3] u32 at `flags`, written by the call
    // after the caller's barrier) into the stats area, then a barrier (WAR: the next kernels reuse the scratch).
    cere_status copy_flags(VkAttnPath path, uint64_t flags, uint32_t rows) {
        if (!counting || rows < kGemmClass) return CERE_OK;
        const uint64_t words = uint64_t(rows) * 6;
        if (stats_used + words + 4 > stats_words) {
            ++dropped;
            return CERE_OK;
        }
        mark(VkProf::kFlagCopy);
        const uint64_t bytes = (words * 4 + 15) / 16 * 16;  // the flags region is 256-aligned: <= 12 bytes past the end
        const vkk1::CopySegment seg{stats_addr + stats_used * 4, bytes, flags, bytes, bytes, 1};
        VK_TRY(copy(&seg, 1));
        flag_copies.push_back(FlagCopy{path, stats_used, rows});
        stats_used += (words + 3) / 4 * 4;
        return bar();
    }

    const VkHcWeights &hc_of(uint32_t layer, plan::Mix mix) const {  // C:208-211 plan_hc
        if (mix == plan::Mix::kFinal) return w.final_mixer;
        return mix == plan::Mix::kAttn ? w.layers[layer].attn_hc : w.layers[layer].mlp_hc;
    }

    // ---- embed.tokens (C:813-834): gather into ws.mixed, then the HC expansion hyper[t] = cat(x[t] x hc) ----------
    cere_status embed() {
        VK_TRY(vkk1::embedding_gather(ctx, w.embed, int64_t(c.vocab_size), int(c.hidden_size), io_ids32.addr, n, ws.mixed));
        VK_TRY(bar());
        // C:830-833: kHc cudaMemcpy2DAsync; on Vulkan one copy_segments dispatch of hc strided segments.
        const uint64_t row = uint64_t(c.hidden_size) * 2;
        vkk1::CopySegment seg[8];
        for (uint32_t g = 0; g < c.hc_count; ++g) {
            seg[g] = vkk1::CopySegment{ws.hyper + g * row, row * c.hc_count, ws.mixed, row, row, n};
        }
        vk::TableRegion t{};
        VK_TRY(ctx.table_alloc(sizeof(vkk1::CopySegment) * c.hc_count, 16, &t));
        vk::Buffer tb = t.view();
        VK_TRY(vkk1::copy_segments(ctx, seg, int(c.hc_count), tb));
        return bar();
    }

    // ---- embed.ple (C:835-838, C:796-800): k4::forward, add_query into ws.hyper at the call's class; an end
    // capture's PLE state as of its row into slot a's PLE region (k4::Capture) ------------------------------------
    cere_status ple() {
        vk::TableRegion t{};
        VK_TRY(ctx.table_alloc(vkk4::kCallTableBytes, 16, &t));
        vk::Buffer tb = t.view();
        vkk4::Capture cap{};
        const bool captures = end != nullptr && end->row > 0;
        if (captures) cap = vkk4::Capture{end->row, end->a->ple.history, end->a->ple.conv_state};
        arm(VkGemmConsumer::kPleProj, ws.scratch, n, c.ple_embed_dim);  // k4 layout(): emb at the workspace base
        VK_TRY(vkk4::forward(ctx, w.ple, s.state().ple, io_ids64.addr, n, ws.hyper, true, ws.hyper, ws.scratch,
                             size_t(ws.scratch_bytes), tb, 0, captures ? &cap : nullptr, cls));
        VK_TRY(bar());
        return tapped();
    }

    // ---- mix.hc (C:142-150 run_mix, C:839-848): hc_mix, or hc_mix_normed when the combine before fused the norm ----
    cere_status mix(const VkHcWeights &hc, bool normed) {
        // k3_hc_vk.cpp hc_mix: down x = normed [m, hc * hidden]; up x = t at workspace + a256(max_m * lowrank * 4).
        arm(VkGemmConsumer::kHcDown, ws.normed, n, c.hc_count * c.hidden_size);
        arm(VkGemmConsumer::kHcUp, ws.scratch + a256(uint64_t(hc_plans.max_m) * c.hc_lowrank * 4), n, c.hc_lowrank);
        if (normed) {
            VK_TRY(vkk3::hc_mix_normed(ctx, hc_plans, ws.normed, hc.down, hc.up, n, ws.mixed, ws.scratch,
                                       vkk3::HcMixPath::kAuto, cls));
        } else {
            VK_TRY(vkk3::hc_mix(ctx, hc_plans, ws.hyper, hc.norm, hc.down, hc.up, n, ws.normed, ws.mixed, ws.scratch,
                                vkk3::HcMixPath::kAuto, cls));
        }
        VK_TRY(bar());
        return tapped();
    }

    // ---- combine.hc (C:182-188, C:869-872) ----------------------------------------------------------------------
    cere_status combine(const VkHcWeights &hc, uint64_t next_norm) {
        if (next_norm == 0) {
            VK_TRY(vkk3::hc_combine(ctx, hc_shape, ws.block_out, ws.hyper, ws.normed, hc.inject, n, ws.hyper));
        } else {
            VK_TRY(vkk3::hc_combine_norm(ctx, hc_shape, ws.block_out, ws.hyper, ws.normed, hc.inject, next_norm, n,
                                         ws.hyper));
        }
        return bar();
    }

    // K2 projection y = x w^T (C:219-228 gemm without taps: plan.run(x, w, n, y, stream, tap.cls)).
    cere_status gemm(const VkDense &d, uint64_t x, uint64_t y) {
        return vkk2::run(ctx, x, d.w, d.scales, y, n, d.n, d.k, false, cls, ws.gemm_ws, d.info);
    }

    // K2 projection of `rows` rows (y = x w^T at the call's class, C:219-228; rows <= cls when cls != 0).
    cere_status gemm_m(const VkDense &d, uint64_t x, uint64_t y, uint32_t rows) {
        return vkk2::run(ctx, x, d.w, d.scales, y, rows, d.n, d.k, false, cls, ws.gemm_ws, d.info);
    }

    // The KV state one attention_rows call reads and writes (forward.cpp AttnKv): a target layer's or the MTP layer's
    // pool entry and pending ring; the session's slot map and context always.
    struct KvRefs {
        uint64_t k_cache, v_cache, qsa_ring, qsa_keys;
    };
    KvRefs target_kv(uint32_t layer) const {
        const VkSessionState &st = s.state();
        return KvRefs{st.k_cache[layer], st.v_cache[layer], st.qsa_ring[layer], st.qsa_keys[layer]};
    }

    // A chunk-0 end state's BF16 K/V rows of one attention layer (model.h EndKv :1005-1015): a capture's rows [0, rows)
    // into k / v, a resume's saved rows [0, in_rows) from in_k / in_v ([rows, 512] each, device addresses).
    struct EndKvRefs {
        uint64_t k = 0, v = 0;
        int64_t rows = 0;
        uint64_t in_k = 0, in_v = 0;
        int64_t in_rows = 0;
    };
    // model.h end_kv(call, index) :1018-1030: attention layer `index`'s rows of the call's runs.
    EndKvRefs end_kv_of(uint64_t index) const {
        const uint64_t row_bytes = uint64_t(c.num_key_value_heads) * c.head_dim * 2;
        EndKvRefs out;
        if (end->kv != 0) {
            out.k = end->kv + 2 * index * uint64_t(end->kv_rows) * row_bytes;
            out.v = out.k + uint64_t(end->kv_rows) * row_bytes;
            out.rows = end->kv_rows;
        }
        if (end->in_kv != 0) {
            out.in_k = end->in_kv + 2 * index * uint64_t(end->in_kv_rows) * row_bytes;
            out.in_v = out.in_k + uint64_t(end->in_kv_rows) * row_bytes;
            out.in_rows = end->in_kv_rows;
        }
        return out;
    }

    // ---- attention_rows (C:310-445), the > 2,048 plan only (VkLimits refuses smaller contexts, OD-5) ------------
    // `t_rows` rows at positions pre_u .. pre_u + t_rows - 1: qkv [t, 13,312], mixed [t, H] (the indexer input),
    // attn_out [t, 6,144], on the KV state `kv`; at the Rec's class `cls`, with a chunk-0 end state's K/V (end_kv).
    cere_status attention_rows(const VkAttentionWeights &aw, const KvRefs &kv, uint64_t pre_u, uint32_t t_rows,
                               uint64_t qkv, uint64_t mixed, uint64_t attn_out, const EndKvRefs *end_kv = nullptr) {
        const VkSessionState &st = s.state();
        const uint64_t qg_width = uint64_t(c.num_attention_heads) * 2 * c.head_dim;  // 12,288 (q | gate)
        const uint64_t kv_width = uint64_t(c.num_key_value_heads) * c.head_dim;      // 512
        const uint64_t qkv_width = qg_width + 2 * kv_width;                          // 13,312
        const uint64_t attn_width = uint64_t(c.num_attention_heads) * c.head_dim;   // 6,144
        const uint64_t qk_width = uint64_t(c.indexer_n_heads + 1) * c.indexer_head_dim;  // 640
        const float eps = float(c.rms_norm_eps);
        const uint64_t k = qkv + qg_width * 2, v = k + kv_width * 2;
        const uint64_t slots = st.slot_map;
        const int64_t tokens = int64_t(t_rows), pre = int64_t(pre_u);
        const uint64_t t = t_rows;
        // Scratch carving (C:331-341): q, k_rot, gate, qk, then the dense workspace or the K9 path.
        uint64_t p = ws.scratch;
        const uint64_t q = p;
        p += a256(t * attn_width * 2);
        const uint64_t k_rot = p;
        p += a256(t * kv_width * 2);
        const uint64_t gate = p;
        p += a256(t * attn_width * 2);
        const uint64_t qk = p;
        p += a256(t * qk_width * 2);
        if (p - ws.scratch > ws.scratch_bytes) return CERE_ERR_INTERNAL;
        const uint64_t rest = ws.scratch_bytes - (p - ws.scratch);
        // A class pin (m210 D15) covers the K9 path and the dense K6 rows of a chunk-0 end state's resume; any other
        // dense K6 call chooses by its own rows (C:319-321).
        const bool pinned = cls != 0 && cls != t_rows;
        // P4-E: the > 16-row path (the forward's ws.prefill_p4, C:393 / C:414-418): the cooperative-matrix kernels
        // when the model's prefill_attention is p4 (probe passed), else the FP32 kernels. <= 16 rows: the split path.
        const vkk6::Options k6o{64, 64, coop ? vkk6::Prefill::kCoopmat : vkk6::Prefill::kFp32};
        const vkk9::Options k9o{64, coop ? vkk9::Prefill::kCoopmat : vkk9::Prefill::kFp32, true};
        const int64_t sel = cls != 0 ? int64_t(cls) : tokens;  // the kernels' path choice (k6 / k9 `sel`)
        const bool coop_call = coop && sel > int64_t(vkk6::kSplitMaxTokens);
        const bool dense = pre + tokens <= int64_t(vkk6::kMaxContext);  // C:376
        // EndKv, checked before any launch (C:341-357): dense rows only; a resume right after its saved rows; a
        // capture strictly inside its call, at prefix 0 or after a resume; room in the scratch for a resume's row sets
        // ahead of the dense workspace.
        const uint64_t row_bytes = kv_width * 2;
        uint64_t end_bytes = 0;
        if (end_kv != nullptr) {
            const bool capture = end_kv->k != 0;
            const bool resume = end_kv->in_k != 0;
            const bool capture_ok = !capture || (end_kv->rows > pre && end_kv->rows < pre + tokens &&
                                                 (pre == 0 || resume) && end_kv->v != 0);
            const bool resume_ok = !resume || (end_kv->in_rows > 0 && pre == end_kv->in_rows && end_kv->in_v != 0);
            if (!dense || !(capture || resume) || !capture_ok || !resume_ok) return CERE_ERR_ARGUMENT;
            if (resume) end_bytes = 2 * a256((uint64_t(pre) + t) * row_bytes);
            if (rest < end_bytes + vkk6::workspace_bytes(tokens, k6o)) return CERE_ERR_ARGUMENT;
        }
        if (dense && pinned && (end_kv == nullptr || end_kv->in_k == 0)) return CERE_ERR_UNSUPPORTED;  // C:380
        // Index keys of this call -> the ring and the compressed keys (every call, from position 0; C:359-371).
        mark(VkProf::kAttnIndexStore);
        VK_TRY(gemm_m(aw.indexer_qk, mixed, qk, t_rows));  // C:361 ws.attn_index_qk.run(mixed, indexer_qk, n, qk, cls)
        VK_TRY(bar());
        const uint64_t k_idx = qk + uint64_t(c.indexer_n_heads) * c.indexer_head_dim * 2;
        // C:363 k9_bounds(w, ws, rope.table, kv.context): rope_delta 0 (text) -> rows [0, max_context_tokens).
        const vkk9::Bounds bounds{0, int64_t(w.max_context_tokens), int64_t(st.max_context_tokens), io_fault.addr};
        const uint64_t rope = w.rope_cos_sin;  // C:792 rope_table(w, session) at rope_delta 0
        VK_TRY(vkk9::qsa_compress_store(ctx, k_idx, int64_t(qk_width), aw.indexer_k_norm, eps, rope, bounds, 0, slots,
                                        pre, tokens, kv.qsa_ring, kv.qsa_keys));  // C:369-370
        // C:372-375 k6::prologue (independent of the compress-store above: one barrier for both).
        mark(VkProf::kAttnPrologue);
        VK_TRY(vkk6::prologue(ctx, qkv, int64_t(qkv_width), k, int64_t(qkv_width), v, int64_t(qkv_width), aw.q_norm,
                              aw.k_norm, eps, rope, int64_t(w.max_context_tokens), 0, 0, slots, tokens, pre,
                              kv.k_cache, kv.v_cache, q, k_rot, gate));
        VK_TRY(bar());
        if (dense) {
            // The dense workspace's coopmat flags (k6_attention_vk.cpp ws_flags: after the partials and the pre-gate
            // output of `tokens` rows).
            const auto flags_at = [&](uint64_t wsp) {
                const uint64_t fb = a256(uint64_t(tokens) * vkk6::kKvHeads * 3 * 4);
                const uint64_t wb = vkk6::workspace_bytes(tokens, k6o);
                return wsp + (tokens > int64_t(vkk6::kSplitMaxTokens) ? wb - fb : wb);
            };
            if (end_kv == nullptr || end_kv->in_k == 0) {
                if (end_kv != nullptr) {
                    // Capture (C:381-387): the cold call's own rows [0, rows), as its dense attention reads them.
                    const uint64_t saved = uint64_t(end_kv->rows);
                    const vkk1::CopySegment seg[2] = {{end_kv->k, row_bytes, k_rot, row_bytes, row_bytes, saved},
                                                      {end_kv->v, row_bytes, v, qkv_width * 2, row_bytes, saved}};
                    VK_TRY(copy(seg, 2));  // reads only: no barrier before the attention below
                }
                // C:388-391: BF16 K/V of the call at prefix 0, else the FP8 cache; gate applied inside.
                mark(VkProf::kK6Dense);
                VK_TRY(vkk6::dense_attention(ctx, q, k_rot, v, int64_t(qkv_width), gate, tokens, pre, pre > 0,
                                             kv.k_cache, kv.v_cache, slots, p, size_t(rest), attn_out,
                                             int64_t(attn_width), 0, k6o));
                VK_TRY(bar());
                if (coop && tokens > int64_t(vkk6::kSplitMaxTokens)) {
                    VK_TRY(copy_flags(VkAttnPath::kK6Dense, flags_at(p), t_rows));
                }
                return CERE_OK;
            }
            // Resume (C:393-413): the saved rows [0, prefix) and this call's rows after them, BF16 as the cold call
            // had them, at the cold call's class.
            const uint64_t saved = uint64_t(end_kv->in_rows);
            const uint64_t all = saved + t;
            const uint64_t k_all = p;
            const uint64_t v_all = p + a256(all * row_bytes);
            const uint64_t dws = p + end_bytes;
            const vkk1::CopySegment in[4] = {
                {k_all, row_bytes, end_kv->in_k, row_bytes, row_bytes, saved},
                {k_all + saved * row_bytes, row_bytes, k_rot, row_bytes, row_bytes, t},
                {v_all, row_bytes, end_kv->in_v, row_bytes, row_bytes, saved},
                {v_all + saved * row_bytes, row_bytes, v, qkv_width * 2, row_bytes, t}};
            VK_TRY(copy(in, 4));
            VK_TRY(bar());
            if (end_kv->k != 0) {
                // D15: a capture past the resume: the rows [0, rows) of the same run, saved ones then its own.
                const uint64_t rows = uint64_t(end_kv->rows);
                const vkk1::CopySegment out[2] = {{end_kv->k, row_bytes, k_all, row_bytes, row_bytes, rows},
                                                  {end_kv->v, row_bytes, v_all, row_bytes, row_bytes, rows}};
                VK_TRY(copy(out, 2));  // reads only
            }
            mark(VkProf::kK6Dense);
            VK_TRY(vkk6::dense_attention(ctx, q, k_all, v_all, int64_t(kv_width), gate, tokens, pre, false,
                                         kv.k_cache, kv.v_cache, slots, dws, size_t(rest - end_bytes), attn_out,
                                         int64_t(attn_width), cls, k6o));
            VK_TRY(bar());
            if (coop_call) VK_TRY(copy_flags(VkAttnPath::kK6Dense, flags_at(dws), t_rows));
            return CERE_OK;
        }
        // K9 path (C:418-443).
        const uint64_t attn = p;
        p += a256(t * attn_width * 2);
        if (p - ws.scratch > ws.scratch_bytes) return CERE_ERR_INTERNAL;
        const uint64_t k9_bytes = ws.scratch_bytes - (p - ws.scratch);
        vkk9::Scratch k9s{};
        VK_TRY(vkk9::carve_scratch(p, size_t(k9_bytes), tokens, int64_t(w.max_context_tokens), &k9s));  // C:422
        mark(VkProf::kK9IndexerQ);
        VK_TRY(vkk9::indexer_qk(ctx, qk, int64_t(qk_width), aw.indexer_q_norm, eps, rope, bounds, 0, pre, tokens,
                                k9s.q_idx));  // C:427-428
        VK_TRY(bar());
        mark(VkProf::kK9Select);
        VK_TRY(vkk9::qsa_select(ctx, k9s.q_idx, kv.qsa_keys, slots, 0, tokens, pre + tokens, k9s.work,
                                k9s.work_bytes, k9s.indices));  // C:430-431
        VK_TRY(bar());
        mark(VkProf::kK9Attn);
        if (pre == 0) {  // C:433-436: the call's own BF16 rows
            VK_TRY(vkk9::qsa_attention(ctx, q, int64_t(attn_width), k9s.indices, tokens, k_rot, int64_t(kv_width), v,
                                       int64_t(qkv_width), 0, 0, 0, vkk9::kSoftmaxScale, k9s.work, k9s.work_bytes,
                                       attn, int64_t(attn_width), coop, cls, k9o));
        } else {  // C:437-441: the FP8 cache
            VK_TRY(vkk9::qsa_attention(ctx, q, int64_t(attn_width), k9s.indices, tokens, 0, 0, 0, 0, kv.k_cache,
                                       kv.v_cache, slots, vkk9::kSoftmaxScale, k9s.work, k9s.work_bytes, attn,
                                       int64_t(attn_width), coop, cls, k9o));
        }
        VK_TRY(bar());
        if (coop_call) VK_TRY(copy_flags(VkAttnPath::kK9Sparse, k9s.work, t_rows));  // the flags at the scratch start
        mark(VkProf::kK9Gate);
        VK_TRY(vkk1::sigmoid_mul(ctx, attn, int64_t(attn_width), gate, int64_t(attn_width), tokens, int(attn_width),
                                 attn_out, int64_t(attn_width)));  // C:442-443
        return bar();
    }

    // ---- full_attention (C:567-583, eager: position == null) ----------------------------------------------------
    cere_status full_attention(uint32_t layer) {
        const VkAttentionWeights &aw = w.layers[layer].attn;
        mark(VkProf::kAttnQkv);
        arm(VkGemmConsumer::kAttnQkvIndex, ws.mixed, n, c.hidden_size);  // also index_qk's x (same rows)
        VK_TRY(gemm(aw.qkv, ws.mixed, ws.qkv));                           // C:569 attn_qkv
        VK_TRY(bar());
        VK_TRY(tapped());
        // A chunk-0 end state's attention rows (EndCall kv, in_kv; C:575-579): its capture and / or resume of this
        // layer's BF16 K/V, at index end_kv_index(cfg, layer) (model.h :1034: the full-attention layers below it).
        const bool has_end_kv = end != nullptr && (end->kv != 0 || end->in_kv != 0);
        EndKvRefs ekv{};
        if (has_end_kv) {
            uint64_t index = 0;
            for (uint32_t j = 0; j < layer; ++j) index += (c.full_attention_layers >> j) & 1;
            ekv = end_kv_of(index);
        }
        VK_TRY(attention_rows(aw, target_kv(layer), prefix, n, ws.qkv, ws.mixed, ws.attn_out,
                              has_end_kv ? &ekv : nullptr));  // C:579-580
        mark(VkProf::kAttnO);
        arm(VkGemmConsumer::kAttnO, ws.attn_out, n, uint32_t(c.num_attention_heads) * c.head_dim);
        VK_TRY(gemm(aw.o_proj, ws.attn_out, ws.block_out));  // C:582 attn_o
        VK_TRY(bar());
        return tapped();
    }

    // ---- gdn (C:591-615: the k5::layer_forward fast path at the call's class; the EndCall becomes the layer's
    // vkk5::EndState over the slot views: a resume reads the FP32 SSM of the `in` pair, a capture writes the conv
    // state and FP32 SSM to the `a`/`b` pair) -------------------------------------------------------------------
    cere_status gdn(uint32_t layer) {
        const VkLayerWeights &lw = w.layers[layer];
        vkk5::LayerWeights g = lw.gdn;
        set_gdn_scales(g, lw.gdn_dense);
        vkk5::EndState end_state{};
        const vkk5::EndState *es = nullptr;
        if (end != nullptr && (end->in_a != nullptr || end->row > 0)) {
            if (end->in_a != nullptr) {
                end_state.ssm_in32 = end->in_a->gdn[layer].ssm;
                end_state.ssm_in32_hi = end->in_b->gdn[layer].ssm;
            }
            if (end->row > 0) {
                end_state.ssm_out32 = end->a->gdn[layer].ssm;
                end_state.ssm_out32_hi = end->b->gdn[layer].ssm;
                end_state.conv_out = end->a->gdn[layer].conv;
                end_state.row = end->row;
            }
            es = &end_state;
        }
        // k5_gdn_layer_vk.cpp carve_layer: in-projections x = hidden (ws.mixed); out_proj x = y [n, 6,144] at the
        // workspace base (layer ws.mixed, y written in place).
        arm(VkGemmConsumer::kGdnIn, ws.mixed, n, c.hidden_size);
        arm(VkGemmConsumer::kGdnOut, ws.scratch, n, c.linear_num_value_heads * c.linear_value_head_dim);
        VK_TRY(vkk5::layer_forward(ctx, g, s.state().gdn[layer], op, ws.mixed, n, c.hidden_size, ws.block_out,
                                   ws.scratch, size_t(ws.scratch_bytes), cls, es));
        VK_TRY(bar());
        return tapped();
    }

    // ---- moe (C:678-688: CERE_MOE_BLOCK = the K7 build, plans at the call's class, no taps) --------------------
    cere_status moe(uint32_t layer) {
        const VkLayerWeights &lw = w.layers[layer];
        vkk7::MoeLayer ml = lw.moe;
        set_moe_scales(ml, lw.shared_dense);
        const vkk7::MoeDims dims{c.hidden_size, c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size,
                                 c.shared_expert_intermediate_size};
        vkk7::MoeOptions o{};
        o.cls = cls;
        arm(VkGemmConsumer::kMoeIn, ws.mixed, n, c.hidden_size);
        arm(VkGemmConsumer::kMoeSharedDown, ws.scratch + k7_s_h_offset(dims, n), n, c.shared_expert_intermediate_size);
        VK_TRY(vkk7::moe_block(ctx, dims, ml, ws.mixed, n, ws.block_out, ws.scratch, size_t(ws.scratch_bytes), &o));
        VK_TRY(bar());
        return tapped();
    }

    // ---- sample_row (C:722-770): the K12 stage on the scored row's BF16 logits ------------------------------------
    cere_status sample_row(const host::ItemSampling &smp, bool logprobs, uint32_t key) {
        const uint64_t V = c.vocab_size;
        const cere_sampling_params &p = smp.params;
        const SamplerScratch sc = carve_sampler(ws.scratch, V);
        const uint64_t bias_count = smp.sampled ? smp.bias.size() : 0;
        if (bias_count > 0) std::memcpy(io_bias.host, smp.bias.data(), bias_count * sizeof(cere_logit_bias));  // C:727-731
        uint64_t mask = 0;
        if (smp.vocab_mask != nullptr) {  // C:733-738
            std::memcpy(io_mask.host, smp.vocab_mask, (V + 31) / 32 * 4);
            mask = io_mask.addr;
        }
        if (smp.sampled) {  // C:740-747
            VK_TRY(vkk12::process(ctx, ws.logits, int64_t(V), s.state().out_counts, p.frequency_penalty,
                                  p.presence_penalty, p.repetition_penalty, mask, bias_count ? io_bias.addr : 0,
                                  bias_count ? io_bias.addr + 4 : 0, 2, int64_t(bias_count), ws.proc_logits));
        } else {
            VK_TRY(vkk12::process(ctx, ws.logits, int64_t(V), 0, 0.0f, 0.0f, 1.0f, mask, 0, 0, 1, 0, ws.proc_logits));
        }
        VK_TRY(bar());
        const float temperature = smp.sampled ? p.temperature : 0.0f;
        const uint64_t score = io_top2.addr;  // the scored row's cere_token_score (rows = 1)
        if (smp.sampled) {
            VK_TRY(vkk12::top2_f32(ctx, ws.proc_logits, int64_t(V), sc.top2_ids, sc.top2_scores));  // C:750
            VK_TRY(bar());
            uint64_t drawn = 0;
            if (temperature != 0.0f) {  // C:752-759
                VK_TRY(vkk12::probabilities(ctx, ws.proc_logits, int64_t(V), temperature, sc.probs, sc.lse, sc.scratch));
                VK_TRY(bar());
                VK_TRY(vkk12::keep_mask(ctx, sc.probs, int64_t(V), int64_t(p.top_k), p.top_p, p.min_p, sc.keep,
                                        sc.scratch));
                VK_TRY(bar());
                VK_TRY(vkk12::draw_seeded(ctx, sc.probs, sc.keep, int64_t(V), p.seed, 1, key, sc.drawn, sc.scratch));
                VK_TRY(bar());
                drawn = sc.drawn;
            }
            VK_TRY(vkk12::finalize(ctx, ws.proc_logits, sc.top2_ids, drawn, score));  // C:760
            VK_TRY(bar());
        }
        if (logprobs) {  // C:762-766: logprob at &lp->logprob, top entries at &lp->top[0] (stride 2 words)
            VK_TRY(vkk12::top_logprobs(ctx, ws.proc_logits, int64_t(V), temperature, score, 0,
                                       int64_t(smp.top_logprobs), io_logprobs.addr + offsetof(cere_token_logprobs, logprob),
                                       io_logprobs.addr + offsetof(cere_token_logprobs, top),
                                       io_logprobs.addr + offsetof(cere_token_logprobs, top) + 4, 2));
            VK_TRY(bar());
        }
        if (smp.sampled) {  // C:768 D5: a sampled item counts its committed token
            VK_TRY(vkk12::count_commit(ctx, s.state().out_counts, score, 1));
            VK_TRY(bar());
        }
        return CERE_OK;
    }

    // ---- T6: shared pieces --------------------------------------------------------------------------------------
    uint64_t hc_rows_dst = 0;  // VkWorkspace::mtp_hc_rows on a model with drafts (C:876 ws.hc_rows), else 0
    uint64_t hc_row_bytes() const { return uint64_t(c.hc_count) * c.hidden_size * 2; }

    // One k1::copy_segments dispatch of `count` segments, its table from the ring; no barrier.
    cere_status copy(const vkk1::CopySegment *seg, int count) {
        vk::TableRegion t{};
        VK_TRY(ctx.table_alloc(sizeof(vkk1::CopySegment) * uint64_t(count), 16, &t));
        vk::Buffer tb = t.view();
        return vkk1::copy_segments(ctx, seg, count, tb);
    }
    // C:876-878 / C:1689-1691: the call's HC rows before the final mixer into ws.hc_rows.
    cere_status copy_hc_rows() {
        if (hc_rows_dst == 0) return CERE_OK;
        const uint64_t bytes = uint64_t(n) * hc_row_bytes();
        const vkk1::CopySegment seg{hc_rows_dst, bytes, ws.hyper, bytes, bytes, 1};
        VK_TRY(copy(&seg, 1));
        return bar();
    }

    // ---- T6: verify rows (C:1575-1721 run_verify for one chain; per_row attention, C:1632-1648) ------------------
    VkUndoState *undo = nullptr;         // the session's snapshots
    uint32_t chain_n = 0;                // the chain's drafts (n = chain_n + 1 rows)
    const int64_t *key_rows = nullptr;   // [chain_n]

    // undo_qsa (C:1236-1251, eager): before row j >= 1 mutates layer L's QSA ring (and, when the chain completes a
    // group, its compressed-key row), both are copied into undo[j - 1]: the state after rows 0..j-1.
    cere_status undo_qsa(uint32_t j, uint32_t L) {
        if (j == 0) return CERE_OK;
        const VkUndoState &u = undo[j - 1];
        const VkSessionState &st = s.state();
        const uint64_t ring = model::layout::qsa_ring_bytes(c), key = model::layout::qsa_key_row_bytes(c);
        vkk1::CopySegment seg[2];
        int count = 0;
        seg[count++] = vkk1::CopySegment{u.qsa_ring[L], ring, st.qsa_ring[L], ring, ring, 1};
        if (key_rows[j - 1] >= 0) {
            seg[count++] = vkk1::CopySegment{u.qsa_key[L], key, st.qsa_keys[L] + uint64_t(key_rows[j - 1]) * key, key,
                                             key, 1};
        }
        VK_TRY(copy(seg, count));
        return bar();  // WAR: row j's compress-store rewrites the ring / key row
    }

    // Full attention of the verify rows (C:1617-1651, per_row): the q|k|v projection of all rows, then attention_rows
    // once per row at its own position p + j (row j's K/V, ring and key are written before row j + 1 attends), then
    // o_proj of all rows. The projections are K2 GEMVs (rows <= 16: each row bit-equal to its m = 1 call).
    cere_status verify_attention(uint32_t L) {
        const VkAttentionWeights &aw = w.layers[L].attn;
        const uint64_t qkv_width = uint64_t(c.num_attention_heads) * 2 * c.head_dim +
                                   2 * uint64_t(c.num_key_value_heads) * c.head_dim;
        const uint64_t attn_width = uint64_t(c.num_attention_heads) * c.head_dim;
        VK_TRY(gemm(aw.qkv, ws.mixed, ws.qkv));
        VK_TRY(bar());
        for (uint32_t j = 0; j < n; ++j) {
            VK_TRY(undo_qsa(j, L));
            VK_TRY(attention_rows(aw, target_kv(L), prefix + j, 1, ws.qkv + j * qkv_width * 2,
                                  ws.mixed + uint64_t(j) * c.hidden_size * 2, ws.attn_out + j * attn_width * 2));
        }
        VK_TRY(gemm(aw.o_proj, ws.attn_out, ws.block_out));
        return bar();
    }

    // GDN of the verify rows (C:1653-1673): the four in-projections of all rows (C's gemm_segments: count separate K2
    // calls, bit-equal), one vkk5::chain_core over the chain (per row bit-equal to layer_core(DECODE), writing the undo
    // saves in-kernel), out_proj of all rows. Scratch as C's carve_gdn_rows.
    cere_status verify_gdn(uint32_t L) {
        const VkLayerWeights &lw = w.layers[L];
        const uint64_t r = n;
        const uint64_t conv_dim = uint64_t(vkk5::kConvDim), value_dim = uint64_t(vkk5::kValueDim);
        const uint64_t heads = c.linear_num_value_heads;
        uint64_t p = ws.scratch;
        const uint64_t g_mixed = p;
        p += a256(r * conv_dim * 2);
        const uint64_t g_z = p;
        p += a256(r * value_dim * 2);
        const uint64_t g_b = p;
        p += a256(r * heads * 2);
        const uint64_t g_a = p;
        p += a256(r * heads * 2);
        const uint64_t g_y = p;
        p += a256(r * value_dim * 2);
        const uint64_t core = p;
        const uint64_t core_bytes = vkk5::chain_core_workspace_bytes(int64_t(r));
        if (core_bytes == 0 || p + core_bytes - ws.scratch > ws.scratch_bytes) return CERE_ERR_INTERNAL;
        VK_TRY(gemm(lw.gdn_dense[0], ws.mixed, g_mixed));
        VK_TRY(gemm(lw.gdn_dense[1], ws.mixed, g_z));
        VK_TRY(gemm(lw.gdn_dense[2], ws.mixed, g_b));
        VK_TRY(gemm(lw.gdn_dense[3], ws.mixed, g_a));
        VK_TRY(bar());
        vkk5::Chains chains{};
        chains.count = 1;
        chains.rows[0] = int64_t(r);
        chains.states[0] = s.state().gdn[L];
        for (uint32_t j = 0; j < chain_n; ++j) chains.saves[j] = undo[j].gdn[L];
        vk::TableRegion t{};
        VK_TRY(ctx.table_alloc(vkk5::kChainCoreTableBytes, 16, &t));
        vk::Buffer tb = t.view();
        vkk5::LayerWeights g = lw.gdn;
        set_gdn_scales(g, lw.gdn_dense);
        VK_TRY(vkk5::chain_core(ctx, g, chains, g_mixed, g_z, g_b, g_a, int64_t(r), g_y, core, size_t(core_bytes), tb,
                                0));
        VK_TRY(bar());
        VK_TRY(gemm(lw.gdn_dense[4], g_y, ws.block_out));
        return bar();
    }

    // The PLE layer of the verify rows (C:1597-1601): one k4::forward_rows, the chain one sequence, row j < n saving
    // the state before row j + 1 into undo[j] in-kernel.
    cere_status verify_ple() {
        vkk4::RowsCall call{};
        call.sequences = 1;
        call.states[0] = s.state().ple;
        call.first[0] = 0;
        call.count[0] = n;
        for (uint32_t j = 0; j < chain_n; ++j) call.saves[j] = undo[j].ple;
        vk::TableRegion t{};
        VK_TRY(ctx.table_alloc(vkk4::kCallTableBytes, 16, &t));
        vk::Buffer tb = t.view();
        VK_TRY(vkk4::forward_rows(ctx, w.ple, call, io_ids64.addr, n, ws.hyper, true, ws.hyper, ws.scratch,
                                  size_t(ws.scratch_bytes), tb, 0));
        return bar();
    }

    // Phase S stage of verify row r (C:1465-1507) on its lm_head row `logits_r`.
    cere_status verify_sample_stage(const host::ItemSampling &smp, uint32_t r, uint64_t logits_r,
                                    const SpecRegions &sp, const SamplerScratch &sc) {
        const uint64_t V = c.vocab_size, words = (V + 31) / 32;
        const uint64_t proc = sp.proc + uint64_t(r) * V * 4;
        if (!smp.sampled) {
            VK_TRY(vkk12::process(ctx, logits_r, int64_t(V), 0, 0.0f, 0.0f, 1.0f, 0, 0, 0, 1, 0, proc));
            return bar();
        }
        const cere_sampling_params &p = smp.params;
        const uint64_t bias_count = smp.bias.size();
        uint64_t counts = s.state().out_counts;
        if (r > 0) {
            // Row j's counts: out_counts plus the drafts d_1..d_j (plain decode's at p + j once they are committed).
            if (r == 1) {
                const uint64_t bytes = V * 4;
                const vkk1::CopySegment seg{sp.counts, bytes, s.state().out_counts, bytes, bytes, 1};
                VK_TRY(copy(&seg, 1));
                VK_TRY(bar());
            }
            VK_TRY(vkk12::count_commit(ctx, sp.counts, io_ids32.addr + 4ull * r, 1));
            VK_TRY(bar());
            counts = sp.counts;
        }
        // P4-F (C:1488-1494, GrammarSpec GS-02): a constrained chain holds n + 1 consecutive masks (row j's = the
        // grammar's mask after d_1..d_j), uploaded together to the IO block (verify_rows masks); row r reads its own.
        const uint64_t mask = smp.vocab_mask != nullptr ? io_mask.addr + uint64_t(r) * words * 4 : 0;
        VK_TRY(vkk12::process(ctx, logits_r, int64_t(V), counts, p.frequency_penalty, p.presence_penalty,
                              p.repetition_penalty, mask, bias_count ? io_bias.addr : 0,
                              bias_count ? io_bias.addr + 4 : 0, 2, int64_t(bias_count), proc));
        VK_TRY(bar());
        VK_TRY(vkk12::top2_f32(ctx, proc, int64_t(V), sp.top2_ids + 8ull * r, sp.top2_scores + 8ull * r));
        if (p.temperature != 0.0f) {
            const uint64_t probs = sp.probs + uint64_t(r) * V * 4;
            VK_TRY(vkk12::probabilities(ctx, proc, int64_t(V), p.temperature, probs, sc.lse, sc.scratch));
            VK_TRY(bar());
            VK_TRY(vkk12::keep_mask(ctx, probs, int64_t(V), int64_t(p.top_k), p.top_p, p.min_p,
                                    sp.keep + uint64_t(r) * words * 4, sc.scratch));
        }
        return bar();
    }

    // Phase S decision (C:1514-1546) of the chain (its first row 0, index 0).
    cere_status verify_sample_commit(const host::ItemSampling &smp, bool logprobs, const SpecRegions &sp) {
        const uint64_t V = c.vocab_size;
        const float temperature = smp.sampled ? smp.params.temperature : 0.0f;
        if (smp.sampled && temperature != 0.0f) {
            uint8_t *committed = m.host_of(sp.committed);
            if (committed == nullptr) return CERE_ERR_INTERNAL;
            std::memset(committed, 0, (chain_n + 1) * 4ull);  // host-coherent, before the submit (C: cudaMemsetAsync)
            VK_TRY(vkk13s::spec_accept(ctx, sp.probs, sp.keep, int64_t(V), io_ids32.addr + 4, int64_t(chain_n),
                                       smp.params.seed, uint32_t(prefix), sp.committed, sp.count, sp.accept));
            VK_TRY(bar());
        }
        for (uint32_t r = 0; r < n; ++r) {
            const uint64_t proc = sp.proc + uint64_t(r) * V * 4;
            const uint64_t score = io_top2.addr + 16ull * r;
            if (smp.sampled) {
                VK_TRY(vkk12::finalize(ctx, proc, sp.top2_ids + 8ull * r,
                                       temperature != 0.0f ? sp.committed + 4ull * r : 0, score));
                VK_TRY(bar());
            }
            if (!logprobs) continue;
            const uint64_t lp = io_logprobs.addr + uint64_t(r) * sizeof(cere_token_logprobs);
            VK_TRY(vkk12::top_logprobs(ctx, proc, int64_t(V), temperature, score, 0, int64_t(smp.top_logprobs),
                                       lp + offsetof(cere_token_logprobs, logprob),
                                       lp + offsetof(cere_token_logprobs, top),
                                       lp + offsetof(cere_token_logprobs, top) + 4, 2));
            VK_TRY(bar());
        }
        return CERE_OK;
    }

    cere_status run_verify(const host::ItemSampling *sampler, bool logprobs) {
        VK_TRY(embed());
        bool normed = false;
        for (const plan::Op &op : plan::kDecode) {
            if (!plan::per_layer(op)) continue;
            const uint32_t L = op.layer;
            switch (op.slot) {
                case plan::Slot::kEmbedPle:
                    VK_TRY(verify_ple());
                    break;
                case plan::Slot::kMixHc:
                    VK_TRY(mix(hc_of(L, op.mix), normed));
                    break;
                case plan::Slot::kAttentionFull:
                    VK_TRY(verify_attention(L));
                    break;
                case plan::Slot::kAttentionLinearGdn:
                    VK_TRY(verify_gdn(L));
                    break;
                case plan::Slot::kFfnMoe:
                    VK_TRY(moe(L));
                    break;
                case plan::Slot::kCombineHc: {
                    const uint64_t next = op.fused ? hc_of(op.target_layer, op.target_mix).norm : 0;
                    VK_TRY(combine(hc_of(L, op.mix), next));
                    normed = op.fused;
                    break;
                }
                default:
                    return CERE_ERR_INTERNAL;
            }
        }
        VK_TRY(copy_hc_rows());                  // C:1689-1691
        VK_TRY(mix(w.final_mixer, normed));      // C:1693
        // lm_head of all rows at once into rows_logits (C:1699-1703: the rows plan, row r bit-equal to the m = 1
        // call; on Vulkan the K2 GEMV is row-invariant by construction), top-2 per row into the IO block.
        const uint64_t V = c.vocab_size;
        const uint64_t rows_logits = rows_logits_addr(ws.scratch, V);
        VK_TRY(vkk2::gemv(ctx, ws.mixed, w.lm_head, 0, rows_logits, n, c.vocab_size, c.hidden_size, false));
        VK_TRY(bar());
        VK_TRY(vkk1::top2(ctx, rows_logits, int64_t(V), n, int64_t(V), io_top2.addr));
        VK_TRY(bar());
        if (sampler == nullptr) return CERE_OK;
        const SpecRegions sp = carve_spec(m.region_addr(VkRegionKind::kSpecSampler), m.layout().verify_rows, V);
        const SamplerScratch sc = carve_sampler(ws.scratch, V);
        if (sampler->sampled && !sampler->bias.empty()) {
            std::memcpy(io_bias.host, sampler->bias.data(), sampler->bias.size() * sizeof(cere_logit_bias));
        }
        // n rows = the chain's n masks (one for an unconstrained-draft row); vk_verify checked they fit io_mask.
        if (sampler->vocab_mask != nullptr) {
            std::memcpy(io_mask.host, sampler->vocab_mask, uint64_t(n) * ((V + 31) / 32) * 4);
        }
        for (uint32_t r = 0; r < n; ++r) {
            VK_TRY(verify_sample_stage(*sampler, r, rows_logits + uint64_t(r) * V * 2, sp, sc));
        }
        return verify_sample_commit(*sampler, logprobs, sp);
    }

    // ---- T6: the MTP draft layer (mtp.cu run :271-462: one segment, cls 0, no stage sink, no tails) ---------------
    struct MtpArgs {
        uint32_t m;            // rows at positions p0 .. p0 + m - 1
        uint64_t p0;
        uint64_t next_ids;     // int32 [m] device: the next token of each row
        uint64_t hc_in, hc_out;  // BF16 [m, 10,240] (may alias)
        uint64_t draft_ids;    // u32 [m - first]
        bool last_only;        // the draft of row m - 1 alone (mtp.cu:299)
    };
    const VkHcWeights &mtp_hc(plan::Mix mix) const {
        return mix == plan::Mix::kAttn ? w.mtp.attn_hc : mix == plan::Mix::kMlp ? w.mtp.mlp_hc : w.mtp.mixer;
    }
    cere_status mtp_forward(const MtpArgs &a) {
        const VkMtpWeights &mw = w.mtp;
        const VkSessionState &st = s.state();
        const uint32_t rows = a.m;
        n = rows;  // the shared helpers' row count
        const uint64_t H = c.hidden_size, HCW = uint64_t(c.hc_count) * H;
        const float eps = float(c.rms_norm_eps);
        const bool rows_plans = rows <= 16;  // mtp.cu:283 (cls 0): the rows plans, i.e. the K2 GEMV classes
        const uint32_t first = a.last_only ? rows - 1 : 0;
        const KvRefs kv{st.mtp_k_cache, st.mtp_v_cache, st.mtp_qsa_ring, st.mtp_qsa_keys};
        if (kv.k_cache == 0 || kv.v_cache == 0 || kv.qsa_ring == 0 || kv.qsa_keys == 0 || mw.lm_head == 0 ||
            mw.map == 0 || mw.count == 0) {
            return CERE_ERR_INTERNAL;
        }
        for (const plan::Op &op : plan::kDraft) {
            switch (op.slot) {
                case plan::Slot::kEmbedTokens: {
                    mark(VkProf::kEmbed);
                    // Fusion (mtp.cu:304-337): e = fc_embedding(norm_e(embed(next))) in ws.mixed; ws.hyper =
                    // fc_hidden(norm_h(hc_in) per stream) + e.
                    VK_TRY(vkk1::embedding_gather(ctx, w.embed, int64_t(c.vocab_size), int(H), a.next_ids, rows,
                                                  ws.block_out));
                    VK_TRY(bar());
                    VK_TRY(vkk1::rmsnorm(ctx, ws.block_out, int64_t(H), mw.norm_embedding, 1, true, eps, rows, int(H),
                                         ws.block_out, int64_t(H)));
                    VK_TRY(bar());
                    VK_TRY(gemm_m(mw.fc_embedding, ws.block_out, ws.mixed, rows));
                    VK_TRY(vkmtp::norm_hidden(ctx, a.hc_in, mw.norm_hidden, eps, rows, ws.normed));
                    VK_TRY(bar());
                    const uint32_t streams = uint32_t(c.hc_count) * rows;
                    if (rows_plans) {  // mtp.cu:324-328: 16-stream calls on the rows plan
                        for (uint32_t i = 0; i < streams; i += 16) {
                            VK_TRY(gemm_m(mw.fc_hidden, ws.normed + uint64_t(i) * H * 2, ws.hyper + uint64_t(i) * H * 2,
                                          std::min<uint32_t>(16, streams - i)));
                        }
                    } else {  // mtp.cu:330: one call of `streams` rows (row-count invariant GEMM: chunks of ws.rows)
                        const VkDense &d = mw.fc_hidden;
                        for (uint32_t i = 0; i < streams; i += uint32_t(ws.rows)) {
                            const uint32_t k = std::min<uint32_t>(uint32_t(ws.rows), streams - i);
                            VK_TRY(vkk2::run(ctx, ws.normed + uint64_t(i) * H * 2, d.w, d.scales,
                                             ws.hyper + uint64_t(i) * H * 2, k, d.n, d.k, false, streams, ws.gemm_ws,
                                             d.info));
                        }
                    }
                    VK_TRY(bar());
                    VK_TRY(vkmtp::add_embedding(ctx, ws.hyper, ws.mixed, rows));
                    VK_TRY(bar());
                    break;
                }
                case plan::Slot::kMixHc: {
                    mark(VkProf::kHcMix);
                    const VkHcWeights &g = mtp_hc(op.mix);
                    if (op.fused) {
                        VK_TRY(vkk3::hc_mix_normed(ctx, hc_plans, ws.normed, g.down, g.up, rows, ws.mixed, ws.scratch,
                                                   vkk3::HcMixPath::kAuto, 0));
                    } else {
                        VK_TRY(vkk3::hc_mix(ctx, hc_plans, ws.hyper, g.norm, g.down, g.up, rows, ws.normed, ws.mixed,
                                            ws.scratch, vkk3::HcMixPath::kAuto, 0));
                    }
                    VK_TRY(bar());
                    break;
                }
                case plan::Slot::kAttentionFull: {
                    mark(VkProf::kAttnQkv);
                    VK_TRY(gemm_m(mw.attn.qkv, ws.mixed, ws.qkv, rows));
                    VK_TRY(bar());
                    VK_TRY(attention_rows(mw.attn, kv, a.p0, rows, ws.qkv, ws.mixed, ws.attn_out));
                    mark(VkProf::kAttnO);
                    VK_TRY(gemm_m(mw.attn.o_proj, ws.attn_out, ws.block_out, rows));
                    VK_TRY(bar());
                    break;
                }
                case plan::Slot::kCombineHc: {
                    mark(VkProf::kHcCombine);
                    const VkHcWeights &g = mtp_hc(op.mix);
                    if (op.fused) {
                        VK_TRY(vkk3::hc_combine_norm(ctx, hc_shape, ws.block_out, ws.hyper, ws.normed, g.inject,
                                                     mtp_hc(op.target_mix).norm, rows, ws.hyper));
                    } else {
                        VK_TRY(vkk3::hc_combine(ctx, hc_shape, ws.block_out, ws.hyper, ws.normed, g.inject, rows,
                                                ws.hyper));
                    }
                    VK_TRY(bar());
                    break;
                }
                case plan::Slot::kFfnMoe: {
                    mark(VkProf::kMoe);
                    // mtp.cu:439-447: a last_only call routes the experts of row m - 1 alone (bit-equal, K13).
                    const vkk13::MoeDims dims{c.hidden_size, c.num_experts, c.num_experts_per_tok,
                                              c.moe_intermediate_size, c.shared_expert_intermediate_size};
                    vkk13::MoeBf16Options o{};
                    o.cls = 0;
                    VK_TRY(vkk13::moe_bf16_block(ctx, dims, mw.moe, ws.mixed, rows, a.last_only ? rows - 1 : 0,
                                                 ws.block_out, ws.scratch, size_t(ws.scratch_bytes), &o));
                    VK_TRY(bar());
                    break;
                }
                case plan::Slot::kNorm: {
                    mark(VkProf::kFinal);
                    // mtp.cu:449-454: the layer's output rows to hc_out, then the final mixer on rows [first, m).
                    const uint64_t bytes = uint64_t(rows) * HCW * 2;
                    const vkk1::CopySegment seg{a.hc_out, bytes, ws.hyper, bytes, bytes, 1};
                    VK_TRY(copy(&seg, 1));
                    VK_TRY(vkk3::hc_mix(ctx, hc_plans, ws.hyper + uint64_t(first) * HCW * 2, mw.mixer.norm,
                                        mw.mixer.down, mw.mixer.up, rows - first, ws.normed, ws.mixed, ws.scratch,
                                        vkk3::HcMixPath::kAuto, 0));
                    VK_TRY(bar());
                    break;
                }
                case plan::Slot::kLmHead: {
                    // hot_lm_head (mtp.cu:237-254): the hot lm_head in chunks of 16 rows into the scratch, argmax + map.
                    const uint32_t count = rows - first;
                    for (uint32_t i = 0; i < count; i += 16) {
                        const uint32_t k = std::min<uint32_t>(16, count - i);
                        VK_TRY(vkk2::gemv(ctx, ws.mixed + uint64_t(i) * H * 2, mw.lm_head, 0, ws.scratch, k, mw.count,
                                          uint32_t(H), false));
                        VK_TRY(bar());
                        VK_TRY(vkmtp::argmax_map(ctx, ws.scratch, mw.count, mw.map, k, a.draft_ids + 4ull * i));
                        VK_TRY(bar());
                    }
                    break;
                }
                default:
                    return CERE_ERR_INTERNAL;  // embed.ple / GDN: not in a draft plan (mtp.cu static_assert)
            }
        }
        return CERE_OK;
    }

    // cuda_device.cpp draft :1433-1499 (eager): the extend of `rows` rows at p0 (hc rows `hc`, next ids `next`) seeded
    // at its last row, then layout::draft_decode_steps one-row MTP decode steps from that row's hc row and the
    // previous step's draft. Returns the proposal count (1 + steps) in *count.
    cere_status draft(uint64_t p0, uint32_t rows, uint64_t hc, uint64_t next, bool prompt_open, uint32_t *count) {
        const VkSessionState &st = s.state();
        const uint64_t proposals = st.proposals;
        if (proposals == 0) return CERE_ERR_INTERNAL;
        const uint64_t steps = model::layout::draft_decode_steps(p0, rows, st.max_context_tokens, prompt_open);
        VK_TRY(mtp_forward(MtpArgs{rows, p0, next, hc, hc, proposals + 4, true}));
        const uint64_t row = hc + uint64_t(rows - 1) * hc_row_bytes();
        uint32_t k = 1;
        for (uint64_t position = p0 + rows; k < 1 + steps; ++position, ++k) {
            VK_TRY(mtp_forward(MtpArgs{1, position, proposals + 4ull * k, row, row, proposals + 4ull * (k + 1), true}));
        }
        *count = k;
        return CERE_OK;
    }

#if defined(CERE_VK_FORWARD_TAPS)
    // TEST BUILD ONLY (forward_s3's static variant; never in the library: AGENTS.md 14): the last row of ws.hyper after
    // each layer (and ws.mixed after the final mixer) copied into the unused tap_shared region, row L at L * 20,480 B.
    cere_status tap(uint64_t src, uint64_t bytes, uint32_t slot) {
        const uint64_t row = uint64_t(c.hc_count) * c.hidden_size * 2;
        vkk1::CopySegment seg{ws.tap_shared + uint64_t(slot) * row, bytes, src, bytes, bytes, 1};
        vk::TableRegion t{};
        VK_TRY(ctx.table_alloc(sizeof seg, 16, &t));
        vk::Buffer tb = t.view();
        VK_TRY(vkk1::copy_segments(ctx, &seg, 1, tb));
        return bar();
    }
#endif

    // ---- run_layers (C:783-913), the whole decode plan of one call ----------------------------------------------
    cere_status run_layers(const host::ItemSampling *sampler, bool logprobs) {
        bool normed = false;  // C:801: ws.normed holds the next mix's norm (combine)
        for (const plan::Op &op : plan::kDecode) {
            const uint32_t L = op.layer;
            switch (op.slot) {
                case plan::Slot::kEmbedTokens:
                    mark(VkProf::kEmbed);
                    VK_TRY(embed());
                    break;
                case plan::Slot::kEmbedPle:
                    mark(VkProf::kPle);
                    VK_TRY(ple());
                    break;
                case plan::Slot::kMixHc:
                    mark(VkProf::kHcMix);
                    VK_TRY(mix(hc_of(L, op.mix), normed));
                    break;
                case plan::Slot::kAttentionFull:
                    VK_TRY(full_attention(L));
                    break;
                case plan::Slot::kAttentionLinearGdn:
                    mark(VkProf::kGdn);
                    VK_TRY(gdn(L));
                    break;
                case plan::Slot::kFfnMoe:
                    mark(VkProf::kMoe);
                    VK_TRY(moe(L));
                    break;
                case plan::Slot::kCombineHc: {  // C:869-872 (fused_norm C:214-216)
                    mark(VkProf::kHcCombine);
                    const uint64_t next = op.fused ? hc_of(op.target_layer, op.target_mix).norm : 0;
                    VK_TRY(combine(hc_of(L, op.mix), next));
                    normed = op.fused;
                    if (op.mix == plan::Mix::kMlp) VK_TRY(layer_done());
#if defined(CERE_VK_FORWARD_TAPS)
                    if (op.mix == plan::Mix::kMlp) {
                        const uint64_t row = uint64_t(c.hc_count) * c.hidden_size * 2;
                        VK_TRY(tap(ws.hyper + uint64_t(n - 1) * row, row, L));
                    }
#endif
                    break;
                }
                case plan::Slot::kNorm:
                    mark(VkProf::kFinal);
                    // C:876-878: the HC rows before the final mixer (the MTP extend input), on a model with drafts.
                    VK_TRY(copy_hc_rows());
                    VK_TRY(mix(w.final_mixer, normed));  // C:880
#if defined(CERE_VK_FORWARD_TAPS)
                    VK_TRY(tap(ws.mixed + uint64_t(n - 1) * c.hidden_size * 2, uint64_t(c.hidden_size) * 2,
                               c.num_hidden_layers));
#endif
                    break;
                case plan::Slot::kLmHead: {
                    // C:884-896 with rows = 1: lm_head of the last row (lm_head_last_row = an m = 1 K2 GEMV), then
                    // top-2 unless a sampled item's finalize writes the score.
                    const uint64_t last = ws.mixed + uint64_t(n - 1) * c.hidden_size * 2;
                    VK_TRY(vkk2::gemv(ctx, last, w.lm_head, 0, ws.logits, 1, c.vocab_size, c.hidden_size, false));
                    VK_TRY(bar());
                    if (sampler == nullptr || !sampler->sampled) {
                        VK_TRY(vkk1::top2(ctx, ws.logits, c.vocab_size, 1, c.vocab_size, io_top2.addr));
                        VK_TRY(bar());
                    }
                    if (sampler != nullptr) {  // C:898-901: key = the session position of the last fed token
                        VK_TRY(sample_row(*sampler, logprobs, uint32_t(prefix + n - 1)));
                    }
                    break;
                }
            }
        }
        return CERE_OK;
    }
};

}  // namespace

const char *vk_prof_name(VkProf p) {
    static const char *const names[] = {"other", "embed", "ple", "hc_mix", "hc_combine", "attn_qkv_gemm",
                                        "attn_index_qk+store", "attn_prologue", "k6_dense", "k9_indexer_q",
                                        "k9_select", "k9_attention", "k9_gate", "attn_o_gemm", "gdn_k5", "moe_k7",
                                        "final+lm_head", "k2_scan", "attn_flag_copy"};
    static_assert(sizeof(names) / sizeof(names[0]) == uint32_t(VkProf::kCount), "VkProf names");
    return uint32_t(p) < uint32_t(VkProf::kCount) ? names[uint32_t(p)] : "?";
}

const char *vk_gemm_consumer_name(VkGemmConsumer c) {
    switch (c) {
        case VkGemmConsumer::kHcDown: return "hc_mix.down (x=normed)";
        case VkGemmConsumer::kHcUp: return "hc_mix.up (x=act(t))";
        case VkGemmConsumer::kAttnQkvIndex: return "attn qkv+index_qk (x=mixed)";
        case VkGemmConsumer::kAttnO: return "attn o_proj (x=attn_out)";
        case VkGemmConsumer::kGdnIn: return "gdn in_proj qkv/z/b/a (x=mixed)";
        case VkGemmConsumer::kGdnOut: return "gdn out_proj (x=y)";
        case VkGemmConsumer::kMoeIn: return "moe router+shared gate/up (x=mixed)";
        case VkGemmConsumer::kMoeSharedDown: return "moe shared down (x=SwiGLU)";
        case VkGemmConsumer::kPleProj: return "ple key/value_proj (x=emb)";
        default: return "?";
    }
}

const char *vk_forward_mxfp8_wiring() {
    if (kGdnWired && kMoeWired) return "k5:wired k7:wired";
    if (kGdnWired) return "k5:wired k7:UNWIRED";
    if (kMoeWired) return "k5:UNWIRED k7:wired";
    return "k5:UNWIRED k7:UNWIRED";
}

bool vk_forward_mxfp8_ok(const VulkanModel &m, std::string *why) {
    const VkModelWeights &w = m.weights();
    for (uint32_t L = 0; L < w.config.num_hidden_layers; ++L) {
        const VkLayerWeights &lw = w.layers[L];
        if (!kGdnWired && !lw.full_attention) {
            for (const VkDense &d : lw.gdn_dense) {
                if (d.mxfp8()) {
                    if (why) *why = "layer " + std::to_string(L) + ": an MXFP8 GDN projection, and vkk5::LayerWeights "
                                    "has no scale fields in this build (TODO(MXFP8))";
                    return false;
                }
            }
        }
        if (!kMoeWired) {
            for (const VkDense &d : lw.shared_dense) {
                if (d.mxfp8()) {
                    if (why) *why = "layer " + std::to_string(L) + ": an MXFP8 shared-expert projection, and vkk7::"
                                    "MoeLayer has no scale fields in this build (TODO(MXFP8))";
                    return false;
                }
            }
        }
    }
    return true;
}

// forward.cpp:1952-1964 check_sampler.
cere_status vk_check_sampler(const host::ItemSampling &s, uint32_t vocab) {
    if (s.top_logprobs > CERE_MAX_TOP_LOGPROBS) return CERE_ERR_ARGUMENT;
    if (!s.sampled) return CERE_OK;
    const cere_sampling_params &p = s.params;
    if (!(p.temperature >= 0.0f) || p.top_k < 1 || p.top_k > vocab || !(p.top_p > 0.0f && p.top_p <= 1.0f) ||
        !(p.min_p >= 0.0f && p.min_p <= 1.0f) || s.bias.size() > vocab) {
        return CERE_ERR_ARGUMENT;
    }
    for (const cere_logit_bias &b : s.bias) {
        if (b.id >= vocab) return CERE_ERR_ARGUMENT;
    }
    return CERE_OK;
}

namespace {

uint64_t g_canary = 0;  // submissions of this process (one thread records at a time: the host serializes calls)

// The IO regions, HC plans, canary and hc-rows destination of one recording (every entry point).
cere_status init_rec(Rec &r, uint32_t rows, bool *launched) {
    VulkanModel &m = r.m;
    r.io_ids32 = m.io(VkRegionKind::kIoIds32);
    r.io_ids64 = m.io(VkRegionKind::kIoIds64);
    r.io_top2 = m.io(VkRegionKind::kIoTop2);
    r.io_logprobs = m.io(VkRegionKind::kIoLogprobs);
    r.io_fault = m.io(VkRegionKind::kIoBoundFault);
    r.io_bias = m.io(VkRegionKind::kIoBias);
    r.io_mask = m.io(VkRegionKind::kIoMask);
    if (r.io_ids32.host == nullptr || r.io_ids64.host == nullptr || r.io_top2.host == nullptr ||
        r.io_logprobs.host == nullptr || r.io_fault.host == nullptr || r.io_bias.host == nullptr ||
        r.io_mask.host == nullptr || r.io_ids32.bytes < rows * 4ull || r.io_ids64.bytes < rows * 8ull ||
        r.io_top2.bytes < rows * uint64_t(sizeof(cere_token_score))) {
        return CERE_ERR_INTERNAL;
    }
    r.coop = m.prefill_coopmat();  // P4-E: the model's prefill attention path (every Rec: prefill, extend, verify)
    // P4-B / D7: the model's K2 GEMM x-row routing (k2_gemm tail_split: SPLIT rows; else FB as P2), read by every
    // vkk2::gemm this Rec records.
    r.ctx.set_k2_tail_split(m.levers().k2_gemm == CERE_LEVER_K2_TAIL_SPLIT);
    r.hc_shape = vkk3::HcShape{r.c.hc_count, r.c.hidden_size, r.c.hc_lowrank, float(r.c.rms_norm_eps)};
    VK_TRY(r.hc_plans.init(r.hc_shape, uint32_t(r.ws.rows)));
    r.launched = launched;
    r.io_canary = m.io(VkRegionKind::kIoTableRing);  // the plan's IO ring region: unused by the device (T2b report)
    if (r.io_canary.host == nullptr || r.io_canary.bytes < 32ull * vk::kMaxInFlight) return CERE_ERR_INTERNAL;
    r.canary_seq = (++g_canary) * 1000;
    // C:876 / C:1689: ws.hc_rows exists exactly on a model with drafts.
    r.hc_rows_dst = r.w.has_draft ? r.ws.mtp_hc_rows : 0;
    if (r.w.has_draft && r.hc_rows_dst == 0) return CERE_ERR_INTERNAL;
    return CERE_OK;
}

// After the recording of the last segment: submit it, or on a recording refusal close the open command buffer
// without running it. Returns the call's status.
cere_status close_rec(Rec &r, cere_status st, const char *what) {
    const bool recording_failed = st != CERE_OK && r.ctx.recording();
    r.ctx.clear_k2_info();  // a consumer that failed between arm() and tapped() leaves no tap armed past its Rec
    if (st == CERE_OK) st = r.submit_segment();
    if (st != CERE_OK && recording_failed) {
        // A refusal while recording: the open command buffer must not run. Latch it (an empty pipeline's dispatch
        // latches ARGUMENT, vk_launch.h) so submit submits nothing, then close it. Earlier segments of a segmented
        // call did run (*launched true then: the session is broken by the caller).
        r.ctx.dispatch(vk::Pipeline{}, nullptr, 1);
        r.ctx.submit();
        std::fprintf(stderr, "cere: %s recording refused status=%d op=%u n=%u position=%llu segment=%u\n", what,
                     int(st), r.op, r.n, (unsigned long long)r.s.position(), r.issued);
    }
    // P4-G: nothing of this call stays in flight, whatever happened (the host reads the IO block next, and the next
    // call's host writes reuse it).
    const cere_status d = r.drain();
    return st != CERE_OK ? st : d;
}

// The K9 bound-fault word (C:1968-1976 take_bound_fault): read and cleared.
cere_status take_bound_fault(const Rec &r) {
    volatile uint32_t *word = reinterpret_cast<volatile uint32_t *>(r.io_fault.host);
    const uint32_t code = *word;
    if (code == 0) return CERE_OK;
    *word = 0;
    std::fprintf(stderr, "cere: device_bound_fault code=%u\n", unsigned(code));
    return CERE_ERR_INTERNAL;
}

}  // namespace

#if defined(CERE_VK_DISPATCH_PROFILE)
}  // namespace cere::vkdev
extern "C" void cere_vk_dprof_phase(uint32_t operation);  // vk_launch.cpp, the forward_s3_dprof test build only
namespace cere::vkdev {
#endif

cere_status vk_forward(vk::Context &ctx, VulkanModel &m, VulkanSession &s, const VkForwardCall &call, bool *launched) {
    *launched = false;
#if defined(CERE_VK_DISPATCH_PROFILE)
    cere_vk_dprof_phase(call.operation);
#endif
    const VkModelWeights &w = m.weights();
    const cere_qwen4_exp_config &c = w.config;
    const VkWorkspace &ws = m.workspace();
    const uint32_t n = call.n;
    // Preconditions (C:2120-2156, CudaDevice::forward cuda_device.cpp:1176-1181), before anything is recorded.
    if (!m.instantiated() || &s.model() != &m || call.tokens == nullptr || call.out == nullptr) return CERE_ERR_ARGUMENT;
    if (s.broken()) return CERE_ERR_INTERNAL;
    if (ctx.faulted()) return CERE_ERR_INTERNAL;
    if (n == 0 || n > ws.rows) return CERE_ERR_ARGUMENT;
    if (call.operation != CERE_OP_PREFILL && (call.operation != CERE_OP_DECODE || n != 1)) return CERE_ERR_ARGUMENT;
    if (s.position() + n > s.max_context_tokens()) return CERE_ERR_CAPACITY;
    if (call.sampler != nullptr) {
        VK_TRY(vk_check_sampler(*call.sampler, c.vocab_size));
    } else if (call.out_logprobs != nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t t = 0; t < n; ++t) {
        if (call.tokens[t] >= c.vocab_size) return CERE_ERR_ARGUMENT;
    }
    // The op list is the generated descriptor's; the bound model must have the same layer kinds.
    if (c.num_hidden_layers != cere::gen::kNumHiddenLayers) return CERE_ERR_UNSUPPORTED;
    for (const plan::Op &op : plan::kDecode) {
        if (op.slot == plan::Slot::kAttentionFull && !w.layers[op.layer].full_attention) return CERE_ERR_UNSUPPORTED;
        if (op.slot == plan::Slot::kAttentionLinearGdn && w.layers[op.layer].full_attention) return CERE_ERR_UNSUPPORTED;
    }
    {
        std::string why;
        if (!vk_forward_mxfp8_ok(m, &why)) {
            static bool said = false;
            if (!said) std::fprintf(stderr, "cere: vk_forward refused: %s\n", why.c_str());
            said = true;
            return CERE_ERR_UNSUPPORTED;
        }
    }
    // D15 EndCall (C:2123-2140): PREFILL only; the class covers the call; a capture at a multiple of 64 strictly inside
    // it with a slot pair, a resume with one, the two pairs disjoint; K/V runs exactly the capture's / resume's rows.
    const VkEndCall *end = call.end;
    if (end != nullptr) {
        const bool capture = end->row >= 0;
        const bool resume = end->in_a != nullptr;
        const auto same = [](const VkSessionState *x, const VkSessionState *y) { return x->slot == y->slot; };
        const bool pairs_ok =
            (!capture || (end->a != nullptr && end->b != nullptr)) && (!resume || end->in_b != nullptr) &&
            (!capture || !resume ||
             (!same(end->a, end->in_a) && !same(end->a, end->in_b) && !same(end->b, end->in_a) &&
              !same(end->b, end->in_b)));
        const bool kv_ok = end->kv == 0 || (capture && end->kv_rows > 0 &&
                                            uint64_t(end->kv_rows) == s.position() + uint64_t(end->row) &&
                                            (s.position() == 0 || end->in_kv != 0));
        const bool in_kv_ok =
            end->in_kv == 0 || (resume && end->in_kv_rows > 0 && s.position() == uint64_t(end->in_kv_rows));
        if (call.operation != CERE_OP_PREFILL || (end->cls != 0 && (end->cls < n || end->cls > ws.rows)) ||
            (capture && (end->row == 0 || end->row >= int64_t(n) || end->row % vkk5::kChunkTokens != 0)) ||
            !pairs_ok || !kv_ok || !in_kv_ok) {
            return CERE_ERR_ARGUMENT;
        }
    }

    Rec r{ctx, m, s, w, ws, c, n, call.operation, s.position()};
    r.cls = end != nullptr ? end->cls : 0;
    r.end = end;
    VK_TRY(init_rec(r, n, launched));
    r.counting = kCountExactFallback && (r.cls != 0 ? r.cls : n) >= kGemmClass && ws.tap_routed != 0;
    if (r.counting) {
        r.stats_addr = ws.tap_routed;
        r.stats_words = ws.rows * c.hidden_size * 2 / 4;  // carve_workspace_rows: tap_routed [rows, hidden] BF16
    }

    // Token ids through the mapped IO block (C:2230-2233, C:2275-2277: no H2D copy on UMA).
    for (uint32_t t = 0; t < n; ++t) {
        const int32_t i32 = int32_t(call.tokens[t]);
        const int64_t i64 = int64_t(call.tokens[t]);
        std::memcpy(r.io_ids32.host + 4ull * t, &i32, 4);
        std::memcpy(r.io_ids64.host + 8ull * t, &i64, 8);
    }

    VkForwardStats &fs = m.forward_stats();
    const uint64_t cmd0 = ctx.device_commands();
    r.segmented = call.operation == CERE_OP_PREFILL && n >= kSegmentMinRows;
    r.depth = r.segmented ? std::min(kPrefillSubmitsInFlight, ctx.max_in_flight()) : 1u;
    if (call.operation == CERE_OP_DECODE && std::min(kPrefillSubmitsInFlight, ctx.max_in_flight()) >= 2) {
        r.split_at = kDecodeFirstSubmitLayers;  // vulkan_forward.h: two command buffers, both in flight
        r.depth = 2;
    }
    if (call.operation == CERE_OP_DECODE && fs.last_operation == CERE_OP_DECODE) r.expect_gpu_ms = fs.last_gpu_ms;
#if defined(CERE_VK_FORWARD_TAPS)
    r.prof = r.segmented && std::getenv("CERE_VK_FORWARD_PROFILE") != nullptr;  // test build only
    if (std::getenv("CERE_VK_FORWARD_SERIAL") != nullptr) {                     // test build only: the P4-E order
        r.depth = 1;
        r.split_at = 0;
    }
    // Test build only, A/B of the two decode levers: one command buffer per step; no polling at the end.
    if (std::getenv("CERE_VK_DECODE_ONE_SUBMIT") != nullptr) {
        r.depth = r.segmented ? r.depth : 1;
        r.split_at = 0;
    }
    if (std::getenv("CERE_VK_DECODE_NO_SPIN") != nullptr) r.expect_gpu_ms = 0;
    ctx.set_calibrated_timing(std::getenv("CERE_VK_SUBMIT_TIMING") != nullptr);  // test build only
#endif
    cere_status st = r.open_segment();
    if (st == CERE_OK) st = r.run_layers(call.sampler, call.out_logprobs != nullptr);
    st = close_rec(r, st, "vk_forward");
    if (st != CERE_OK) return st;
    const uint64_t ring = r.ring_max;
    const double record_ms = r.record_ms, wait_ms = r.wait_ms;
    const auto t2 = Clock::now();
    VK_TRY(take_bound_fault(r));  // C:1968-1976, read and cleared whatever happens next
    // The scored row (C:2323-2328).
    std::memcpy(call.out, r.io_top2.host, sizeof(cere_token_score));
    if (call.out_logprobs != nullptr) {
        cere_token_logprobs staged{};
        std::memcpy(&staged, r.io_logprobs.host, sizeof staged);
        cere_token_logprobs lp{};
        lp.id = call.out->id;
        lp.logprob = staged.logprob;
        lp.top_count = call.sampler->top_logprobs;
        for (uint32_t i = 0; i < lp.top_count; ++i) lp.top[i] = staged.top[i];
        *call.out_logprobs = lp;
    }
    // Exact-fallback counters: summed per segment at its retire (Rec::count_through), added here.
    if (r.counting) {
        for (uint32_t i = 0; i < uint32_t(VkGemmConsumer::kCount); ++i) {
            VkFallbackCounter &k = fs.fallback[i];
            const VkFallbackCounter &a = r.fb_acc[i];
            k.rows += a.rows;
            k.flagged += a.flagged;
            k.split += a.split;
            k.scans += a.scans;
        }
        for (uint32_t i = 0; i < uint32_t(VkAttnPath::kCount); ++i) {
            VkAttnFlagCounter &k = fs.attn_flags[i];
            const VkAttnFlagCounter &a = r.attn_acc[i];
            k.calls += a.calls;
            k.rows += a.rows;
            k.flagged_words += a.flagged_words;
            k.flagged_pairs += a.flagged_pairs;
            k.flagged_rows += a.flagged_rows;
        }
        fs.fallback_dropped += r.dropped;
    }
    s.set_position(s.position() + n);  // C:2329
    // Statistics.
    const double gpu_ms = r.gpu_ms;
    const double read_ms = ms_since(t2);
    ++fs.calls;
    fs.tokens += n;
    fs.last_record_ms = record_ms;
    fs.last_wait_ms = wait_ms;
    fs.last_gpu_ms = gpu_ms;
    fs.last_read_ms = read_ms;
    fs.last_commands = ctx.device_commands() - cmd0;
    fs.last_barriers = r.barriers;
    fs.last_ring_bytes = ring;
    fs.last_submits = r.submits;
    fs.last_rows = n;
    fs.last_operation = call.operation;
    fs.last_max_submit_gpu_ms = r.max_submit_ms;
    fs.last_depth = r.depth;
    if (r.segmented) {
        fs.prefill_submits.merge(r.sub);
        fs.prefill_last_depth = r.depth;
    } else if (call.operation == CERE_OP_DECODE) {
        fs.decode_submits.merge(r.sub);
        if (r.last_retire_after_end_ms == r.last_retire_after_end_ms) {  // not NaN: calibrated
            ++fs.decode_calibrated;
            fs.decode_retire_after_end_ms += r.last_retire_after_end_ms;
        }
    }
    for (uint32_t i = 0; i < uint32_t(VkProf::kCount); ++i) fs.prof_ms[i] += r.prof_ms[i];
    fs.prof_segments += r.prof_segments;
    fs.prof_dropped += r.prof_dropped;
    if (call.operation == CERE_OP_PREFILL) {
        ++fs.prefill_calls;
        fs.prefill_max_submit_gpu_ms = std::max(fs.prefill_max_submit_gpu_ms, r.max_submit_ms);
        ++(r.coop ? fs.prefill_coopmat_calls : fs.prefill_fp32_calls);
        fs.prefill_record_ms += record_ms;
        fs.prefill_gpu_ms += gpu_ms;
        fs.prefill_wall_ms += record_ms + wait_ms + read_ms;
        fs.prefill_read_ms += read_ms;
        fs.prefill_count_ms += r.count_ms;
    } else {
        ++fs.decode_calls;
        fs.decode_record_ms += record_ms;
        fs.decode_gpu_ms += gpu_ms;
        fs.decode_wall_ms += record_ms + wait_ms + read_ms;
    }
    return CERE_OK;
}

// ---- T6: speculation entry points ---------------------------------------------------------------------------------

cere_status vk_verify(vk::Context &ctx, VulkanModel &m, VulkanSession &s, VkVerifyCall &call, bool *launched) {
    *launched = false;
#if defined(CERE_VK_DISPATCH_PROFILE)
    cere_vk_dprof_phase(CERE_OP_VERIFY);
#endif
    const VkModelWeights &w = m.weights();
    const cere_qwen4_exp_config &c = w.config;
    const VkWorkspace &ws = m.workspace();
    // Preconditions (C:2453-2490 forward_verify_rows, restated for one chain), before anything is recorded.
    if (!m.instantiated() || &s.model() != &m || call.tokens == nullptr || call.out == nullptr) return CERE_ERR_ARGUMENT;
    if (!w.has_draft || s.undo_count() < kVkMaxChainDrafts) return CERE_ERR_UNSUPPORTED;
    if (s.broken() || ctx.faulted()) return CERE_ERR_INTERNAL;
    const uint32_t n = call.n, rows = n + 1;
    if (n > kVkMaxChainDrafts || rows > m.layout().verify_rows || rows > ws.rows) return CERE_ERR_ARGUMENT;
    const uint64_t p = s.position();
    if (p >= s.max_context_tokens() || n >= s.max_context_tokens() - p) return CERE_ERR_ARGUMENT;
    for (uint32_t j = 0; j <= n; ++j) {
        if (call.tokens[j] >= c.vocab_size) return CERE_ERR_ARGUMENT;
    }
    if (call.sampler == nullptr) {
        if (call.logprobs) return CERE_ERR_ARGUMENT;
    } else {
        if (call.logprobs && call.out_logprobs == nullptr) return CERE_ERR_ARGUMENT;
        VK_TRY(vk_check_sampler(*call.sampler, c.vocab_size));
        // P4-F grammar drafts (C:2453-2490 GS-02): a constrained chain's rows + 1 masks must fit the IO mask region
        // (the memory plan carves verify_rows masks).
        if (call.sampler->vocab_mask != nullptr &&
            uint64_t(rows) * ((c.vocab_size + 31) / 32) * 4 > m.io(VkRegionKind::kIoMask).bytes) {
            return CERE_ERR_ARGUMENT;
        }
    }
    if (c.num_hidden_layers != cere::gen::kNumHiddenLayers) return CERE_ERR_UNSUPPORTED;
    // The K9 path's completed compressed-key rows (C:2475-2490): row j of the chain saves into undo[j - 1] the pool
    // row of the first group it or a later row completes (q >= p + j, q % ratio == ratio - 1, q <= p + n), else -1.
    const uint64_t ratio = c.indexer_compress_ratio;
    const bool qsa = w.max_context_tokens > uint64_t(vkk6::kMaxContext);
    if (ratio == 0) return CERE_ERR_UNSUPPORTED;
    for (uint32_t j = 1; j <= n; ++j) {
        call.key_rows[j - 1] = -1;
        const uint64_t q = p + j + (ratio - 1 - (p + j) % ratio) % ratio;
        if (!qsa || q > p + n) continue;
        const uint64_t first = q + 1 - ratio;
        const uint64_t page = first / host::kKvPageTokens;
        if (page >= s.pages().size()) return CERE_ERR_ARGUMENT;
        call.key_rows[j - 1] =
            int64_t((uint64_t(s.pages()[page]) * host::kKvPageTokens + first % host::kKvPageTokens) / ratio);
    }

    Rec r{ctx, m, s, w, ws, c, rows, CERE_OP_VERIFY, p};
    VK_TRY(init_rec(r, rows, launched));
    r.undo = &s.undo(0);
    r.chain_n = n;
    r.key_rows = call.key_rows;
    for (uint32_t t = 0; t < rows; ++t) {
        const int32_t i32 = int32_t(call.tokens[t]);
        const int64_t i64 = int64_t(call.tokens[t]);
        std::memcpy(r.io_ids32.host + 4ull * t, &i32, 4);
        std::memcpy(r.io_ids64.host + 8ull * t, &i64, 8);
    }
    const uint64_t cmd0 = ctx.device_commands();
    cere_status st = r.open_segment();
    if (st == CERE_OK) st = r.run_verify(call.sampler, call.logprobs);
    st = close_rec(r, st, "vk_verify");
    if (st != CERE_OK) return st;
    const auto t2 = Clock::now();
    VK_TRY(take_bound_fault(r));
    std::memcpy(call.out, r.io_top2.host, rows * sizeof(cere_token_score));
    if (call.logprobs) {
        for (uint32_t q = 0; q < rows; ++q) {  // C:2600-2606 staged_logprobs
            cere_token_logprobs staged{};
            std::memcpy(&staged, r.io_logprobs.host + q * sizeof(cere_token_logprobs), sizeof staged);
            cere_token_logprobs lp{};
            lp.id = call.out[q].id;
            lp.logprob = staged.logprob;
            lp.top_count = call.sampler->top_logprobs;
            for (uint32_t i = 0; i < lp.top_count; ++i) lp.top[i] = staged.top[i];
            call.out_logprobs[q] = lp;
        }
    }
    VkForwardStats::Spec &sp = m.forward_stats().spec;
    sp.verify_record_ms += r.record_ms;
    sp.verify_gpu_ms += r.gpu_ms;
    sp.verify_wall_ms += r.record_ms + r.wait_ms + ms_since(t2);
    sp.max_verify_gpu_ms = std::max(sp.max_verify_gpu_ms, r.gpu_ms);
    sp.last_verify_commands = ctx.device_commands() - cmd0;
    sp.last_verify_barriers = r.barriers;
    for (uint32_t j = 0; j < rows; ++j) sp.sparse_rows += p + j + 1 > uint64_t(vkk6::kMaxContext) ? 1 : 0;
    return CERE_OK;
}

namespace {

// The session's proposals after a draft submit: count from the host, ids [1..count] from the mapping (written by
// argmax_map); the host copy and the device word 0 (cuda_device.cpp:1485-1488).
cere_status read_proposals(VulkanModel &m, VulkanSession &s, uint32_t count) {
    uint8_t *p = m.host_of(s.state().proposals);
    if (p == nullptr) return CERE_ERR_INTERNAL;
    uint32_t words[4] = {count, 0, 0, 0};
    std::memcpy(words + 1, p + 4, 12);
    for (uint32_t i = count + 1; i < 4; ++i) words[i] = 0;
    std::memcpy(p, &count, 4);
    s.set_proposals(words);
    return CERE_OK;
}

}  // namespace

cere_status vk_commit_draft(vk::Context &ctx, VulkanModel &m, VulkanSession &s, const VkCommitDraftCall &call,
                            bool *launched) {
    *launched = false;
    const VkModelWeights &w = m.weights();
    const cere_qwen4_exp_config &c = w.config;
    const VkWorkspace &ws = m.workspace();
    const uint32_t n = call.n, a = call.a;
    // C:2611-2619 verify_commit's preconditions.
    if (!m.instantiated() || &s.model() != &m || call.committed == nullptr || a == 0 || a > n + 1 ||
        n > kVkMaxChainDrafts || (a <= n && call.key_rows == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    if (!w.has_draft || s.undo_count() < kVkMaxChainDrafts) return CERE_ERR_UNSUPPORTED;
    if (s.broken() || ctx.faulted()) return CERE_ERR_INTERNAL;
    const uint64_t p = s.position();
    if (p >= s.max_context_tokens() || a > s.max_context_tokens() - p) return CERE_ERR_ARGUMENT;
    for (uint32_t i = 0; i < a; ++i) {
        if (call.committed[i] >= c.vocab_size) return CERE_ERR_ARGUMENT;
    }
    Rec r{ctx, m, s, w, ws, c, a, CERE_OP_DECODE, p};
    VK_TRY(init_rec(r, a, launched));
    for (uint32_t i = 0; i < a; ++i) {
        const int32_t i32 = int32_t(call.committed[i]);
        std::memcpy(r.io_ids32.host + 4ull * i, &i32, 4);
    }
    const VkSessionState &st = s.state();
    const uint64_t cmd0 = ctx.device_commands();
    uint32_t count = 0;
    cere_status rs = r.open_segment();
    const auto body = [&]() -> cere_status {
        if (a <= n) {
            // The state after rows 0..a-1: undo[a - 1], every piece in one k1::copy_segments (C:2627-2646, DG-03).
            const VkUndoState &u = s.undo(a - 1);
            const int64_t key_row = call.key_rows[a - 1];
            const uint64_t key = model::layout::qsa_key_row_bytes(c);
            vkk1::CopySegment pieces[vkk1::kMaxCopySegments];
            int k = 0;
            const auto flat = [&](uint64_t dst, uint64_t src, uint64_t bytes) {
                pieces[k++] = vkk1::CopySegment{dst, bytes, src, bytes, bytes, 1};
            };
            for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
                if (k + 2 > vkk1::kMaxCopySegments - 2) return CERE_ERR_INTERNAL;
                if (model::layout::full_attention(c, L)) {
                    flat(st.qsa_ring[L], u.qsa_ring[L], model::layout::qsa_ring_bytes(c));
                    if (key_row >= 0) flat(st.qsa_keys[L] + uint64_t(key_row) * key, u.qsa_key[L], key);
                } else {
                    flat(st.gdn[L].conv, u.gdn[L].conv, model::layout::gdn_conv_bytes(c));
                    flat(st.gdn[L].ssm, u.gdn[L].ssm, model::layout::gdn_ssm_bytes(c));
                }
            }
            flat(st.ple.history, u.ple.history, model::layout::ple_history_bytes(c));
            flat(st.ple.conv_state, u.ple.conv_state, model::layout::ple_conv_bytes(c));
            VK_TRY(r.copy(pieces, k));
            VK_TRY(r.bar());
            // C:2647-2662: further groups a longer chain completed (never with n <= 3; ported for the rule).
            if (key_row >= 0) {
                const uint64_t ratio = c.indexer_compress_ratio;
                const uint64_t q1 = p + a + (ratio - 1 - (p + a) % ratio) % ratio;
                for (uint64_t next = q1 - p; next < n && call.key_rows[next] >= 0; next += ratio) {
                    const VkUndoState &v = s.undo(uint32_t(next));
                    k = 0;
                    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
                        if (!model::layout::full_attention(c, L)) continue;
                        flat(st.qsa_keys[L] + uint64_t(call.key_rows[next]) * key, v.qsa_key[L], key);
                    }
                    VK_TRY(r.copy(pieces, k));
                    VK_TRY(r.bar());
                }
            }
        }
        // D5 (M29S2-14): only the committed tokens of a sampled item count (C:2665).
        if (call.counted) {
            VK_TRY(vkk12::count_commit(ctx, st.out_counts, r.io_ids32.addr, int64_t(a)));
            VK_TRY(r.bar());
        }
        VK_TRY(ctx.stamp(2));
        // Draft-extend (cuda_device.cpp:2510-2546, one chain: base 0): the MTP over the a committed rows at p .. p + a
        // - 1 (the verify's hc rows), next ids = the committed tokens, then the decode steps.
        return r.draft(p, a, ws.mtp_hc_rows, r.io_ids32.addr, false, &count);
    };
    if (rs == CERE_OK) rs = body();
    rs = close_rec(r, rs, "vk_commit_draft");
    if (rs != CERE_OK) return rs;
    const auto t2 = Clock::now();
    VK_TRY(take_bound_fault(r));
    s.set_position(p + a);  // C:2666
    VK_TRY(read_proposals(m, s, count));
    VkForwardStats::Spec &sp = m.forward_stats().spec;
    const double restore = ctx.stamp_ms(0, 2), draft = ctx.stamp_ms(2, 1);
    sp.cd_record_ms += r.record_ms;
    sp.cd_gpu_ms += r.gpu_ms;
    sp.restore_gpu_ms += restore;
    sp.draft_gpu_ms += draft;
    sp.cd_wall_ms += r.record_ms + r.wait_ms + ms_since(t2);
    sp.max_cd_gpu_ms = std::max(sp.max_cd_gpu_ms, r.gpu_ms);
    sp.last_cd_commands = ctx.device_commands() - cmd0;
    sp.last_cd_barriers = r.barriers;
    sp.restores += a <= n ? 1 : 0;
    sp.key_restores += a <= n && call.key_rows[a - 1] >= 0 ? 1 : 0;
    return CERE_OK;
}

cere_status vk_extend_draft(vk::Context &ctx, VulkanModel &m, VulkanSession &s, const uint32_t *tokens, uint32_t n,
                            uint32_t next, bool prompt_open, bool *launched) {
    *launched = false;
    const VkModelWeights &w = m.weights();
    const cere_qwen4_exp_config &c = w.config;
    const VkWorkspace &ws = m.workspace();
    // cuda_device.cpp:1413-1418.
    if (!s.drafting()) return CERE_OK;
    if (&s.model() != &m || tokens == nullptr || n == 0 || n > s.position() || n > ws.rows || !w.has_draft) {
        return CERE_ERR_ARGUMENT;
    }
    if (s.broken() || ctx.faulted()) return CERE_ERR_INTERNAL;
    if (next >= c.vocab_size) return CERE_ERR_ARGUMENT;
    const uint64_t p0 = s.position() - n;
    Rec r{ctx, m, s, w, ws, c, n, CERE_OP_PREFILL, p0};
    VK_TRY(init_rec(r, n, launched));
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t i32 = int32_t(i + 1 < n ? tokens[i + 1] : next);
        std::memcpy(r.io_ids32.host + 4ull * i, &i32, 4);
    }
    uint32_t count = 0;
#if defined(CERE_VK_FORWARD_TAPS)
    r.prof = std::getenv("CERE_VK_FORWARD_PROFILE") != nullptr;  // test build only
#endif
    cere_status st = r.open_segment();
    if (st == CERE_OK) st = r.draft(p0, n, ws.mtp_hc_rows, r.io_ids32.addr, prompt_open, &count);
    st = close_rec(r, st, "vk_extend_draft");
    if (st != CERE_OK) return st;
    const auto t2 = Clock::now();
    VK_TRY(take_bound_fault(r));
    VK_TRY(read_proposals(m, s, count));
    VkForwardStats::Spec &sp = m.forward_stats().spec;
    ++sp.extends;
    sp.extend_rows += n;
    sp.extend_gpu_ms += r.gpu_ms;
    sp.extend_wall_ms += r.record_ms + r.wait_ms + ms_since(t2);
    sp.max_extend_gpu_ms = std::max(sp.max_extend_gpu_ms, r.gpu_ms);
    for (uint32_t i = 0; i < uint32_t(VkProf::kCount); ++i) sp.extend_prof_ms[i] += r.prof_ms[i];
    return CERE_OK;
}

}  // namespace cere::vkdev
