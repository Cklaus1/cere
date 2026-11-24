// VulkanAmd P3 T1: the launch layer's library contract (vk_launch.h), positive and negative rows. Links the library
// object (vk_launch_core.o: no exit, no printing) and the generated embedded-SPIR-V table, NOT the ktests'
// exit-on-error convenience: every refusal below must come back as a status with the process alive.
//
//   vk_launch_test <spv_dir>    (spv_dir only to compare the embedded bytes with the compiled files; no context here
//                                ever has a shader directory set)
//
// Never faults or resets the GPU: the sticky-fault rows mark the context faulted through the test hook
// (Context::inject_fault_for_test) and check that nothing reaches the device afterwards; the kill-ordering rows (P4-G)
// drop or skew a completion marker on the host side while the GPU runs normally.
//
//   vk_launch_test <spv_dir> --queue-probe MS COUNT   (opt-in, P4-G: COUNT command buffers of ~MS ms GPU work, two in
//                                                      flight, back to back; see queue_probe)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k3_hc_vk.h"
#include "launch/vk_launch.h"

using namespace cere;
using vk::Context;

static int g_fail = 0, g_pass = 0;
static void report(const char *name, bool ok, const std::string &detail) {
    std::printf("%-34s %s  %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
    std::fflush(stdout);
    ++(ok ? g_pass : g_fail);
}

static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static std::vector<uint16_t> randbf(size_t n, std::mt19937 &g, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = f2bf(d(g));
    return v;
}
static uint8_t fp8_ref(float x) {  // nearest E4M3 (ties to even mantissa), saturating as K1's kv_store (k1_test's reference)
    uint32_t u; std::memcpy(&u, &x, 4); const uint8_t s = uint8_t((u >> 24) & 0x80);
    if (std::isnan(x)) return s | 0x7f;
    const float a = std::fabs(x);
    if (a > 464.f) return s | 0x7e;
    uint8_t best = 0; double bd = INFINITY;
    for (uint8_t c = 0; c <= 0x7e; ++c) {
        const int e = c >> 3, m = c & 7;
        const double v = e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1 + m / 8.0) * std::ldexp(1.0, e - 7);
        const double d = std::fabs(v - a);
        if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = c; }
    }
    return s | best;
}

// Allocation helper for the positive rows (a refusal here is a FAIL row and the caller returns).
static bool make(Context &ctx, uint64_t bytes, vk::Buffer &b, const char *what) {
    std::string why;
    if (ctx.allocate(bytes, b, false, &why) != CERE_OK) {
        report(what, false, "allocation refused: " + why);
        return false;
    }
    std::memset(b.map, 0, bytes);
    return true;
}

static std::unique_ptr<Context> open0(const char *row) {
    std::unique_ptr<Context> c;
    std::string why;
    const cere_status s = Context::create(0, &c, &why);
    if (s != CERE_OK) report(row, false, "create(0) = " + std::to_string(s) + ": " + why);
    return c;
}

static std::string st(cere_status s) { return std::to_string(int(s)); }

// --- P4-G: pipelined submits (several command buffers in flight) -----------------------------------------------------
static std::unique_ptr<Context> open_slots(const char *row, uint32_t slots) {
    std::unique_ptr<Context> c;
    std::string why;
    vk::CreateOptions o;
    o.max_in_flight = slots;
    const cere_status s = Context::create(0, &c, &why, o);
    if (s != CERE_OK) report(row, false, "create(0, max_in_flight " + std::to_string(slots) + ") = " + st(s) + ": " + why);
    return c;
}

// `calls` embedding gathers table[ids] -> out (each the same rows: the last one's result is the output).
static cere_status gathers(Context &c, uint64_t table, int64_t vocab, int hidden, uint64_t ids, int64_t n, uint64_t out, int calls) {
    cere_status r = CERE_OK;
    for (int k = 0; k < calls && r == CERE_OK; ++k) {
        r = vkk1::embedding_gather(c, table, vocab, hidden, ids, n, out);
        if (r == CERE_OK) r = c.barrier();
    }
    return r;
}

static void pipelined_rows(std::mt19937 &g) {
    const int hidden = 64;
    const int64_t vocab = 16, n = 8;
    // misuse at create
    {
        std::unique_ptr<Context> bad;
        std::string w0, w5;
        vk::CreateOptions o0, o5;
        o0.max_in_flight = 0;
        o5.max_in_flight = vk::kMaxInFlight + 1;
        const cere_status s0 = Context::create(0, &bad, &w0, o0);
        const cere_status s5 = Context::create(0, &bad, &w5, o5);
        report("pipelined: bad max_in_flight", s0 == CERE_ERR_ARGUMENT && s5 == CERE_ERR_ARGUMENT && !bad,
               "0 -> " + st(s0) + " \"" + w0 + "\"; " + std::to_string(vk::kMaxInFlight + 1) + " -> " + st(s5));
    }
    std::unique_ptr<Context> cp = open_slots("pipelined", 2);
    if (!cp) return;
    Context &c = *cp;
    auto table = randbf(size_t(vocab * hidden), g, 1.f);
    std::vector<int32_t> ids1(n), ids2(n);
    for (auto &i : ids1) i = int32_t(g() % vocab);
    for (auto &i : ids2) i = int32_t(g() % n);  // B gathers rows of A's output
    vk::Buffer dt, di1, di2, o1, o2, o3;
    if (!(make(c, table.size() * 2, dt, "pipelined") && make(c, n * 4, di1, "pipelined") && make(c, n * 4, di2, "pipelined") &&
          make(c, uint64_t(n) * hidden * 2, o1, "pipelined") && make(c, uint64_t(n) * hidden * 2, o2, "pipelined") &&
          make(c, uint64_t(n) * hidden * 2, o3, "pipelined")))
        return;
    std::memcpy(dt.map, table.data(), table.size() * 2);
    std::memcpy(di1.map, ids1.data(), n * 4);
    std::memcpy(di2.map, ids2.data(), n * 4);
    const auto expect_b = [&](const void *out) {  // B's output: rows ids2 of A's output = table rows ids1[ids2[t]]
        const uint16_t *o = static_cast<const uint16_t *>(out);
        for (int64_t t = 0; t < n; ++t)
            if (std::memcmp(o + t * hidden, &table[size_t(ids1[size_t(ids2[t])]) * hidden], hidden * 2) != 0) return false;
        return true;
    };

    // (1) two in flight, B reads A's output: the leading barrier orders them; slots, ring parts, tickets, stamps
    {
        std::memset(o1.map, 0, n * hidden * 2);
        std::memset(o2.map, 0, n * hidden * 2);
        uint64_t t1 = 0, t2 = 0;
        vk::TableRegion ra, rb;
        const uint64_t c0 = c.device_commands();
        cere_status r = c.begin();
        const uint64_t begin_idle = c.device_commands() - c0;
        if (r == CERE_OK) r = c.table_alloc(64, 16, &ra);
        if (r == CERE_OK) r = c.stamp(0);
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o1.addr, 3000);  // long: B must wait for it
        if (r == CERE_OK) r = c.stamp(1);
        if (r == CERE_OK) r = c.submit(&t1);
        const uint32_t f1 = c.in_flight();
        const uint64_t c1 = c.device_commands();
        if (r == CERE_OK) r = c.begin();
        const uint64_t begin_busy = c.device_commands() - c1;
        if (r == CERE_OK) r = c.table_alloc(64, 16, &rb);
        if (r == CERE_OK) r = c.stamp(0);
        if (r == CERE_OK) r = gathers(c, o1.addr, n, hidden, di2.addr, n, o2.addr, 1);
        if (r == CERE_OK) r = c.stamp(1);
        if (r == CERE_OK) r = c.submit(&t2);
        const uint32_t f2 = c.in_flight();
        // every slot busy: begin refuses and records nothing
        const uint64_t c2 = c.device_commands();
        const cere_status b3 = c.begin();
        const bool refused_clean = b3 == CERE_ERR_ARGUMENT && !c.recording() && c.device_commands() == c2 && !c.faulted();
        const cere_status w1 = c.wait(t1);
        const double s1 = c.stamp_ms(0, 1);
        const uint64_t n1 = c.last_completed_nonce(), lr1 = c.last_retired();
        const uint32_t f3 = c.in_flight();
        const bool a_done = std::memcmp(o1.map, &table[size_t(ids1[0]) * hidden], hidden * 2) == 0;
        const cere_status w2 = c.wait(t2);
        const double s2 = c.stamp_ms(0, 1);
        const uint64_t n2 = c.last_completed_nonce();
        const vk::SubmitTiming tm = c.last_submit_timing();
        const bool b_ok = expect_b(o2.map);
        const bool parts = ra.addr != rb.addr && (rb.addr >= ra.addr + c.table_ring_bytes() || ra.addr >= rb.addr + c.table_ring_bytes());
        char m[512];
        std::snprintf(m, sizeof m,
                      "tickets %llu,%llu; in flight 1 -> 2 -> (begin refused, nothing recorded) -> wait(t1): %u -> wait(t2): 0; begin "
                      "commands idle %llu / with one in flight %llu (leading barrier); ring parts %llu B apart; stamps A %.3f ms, "
                      "B %.4f ms; nonces +1; B (reads A's output) bit-exact %s; B's GPU gap after A %.4f ms",
                      (unsigned long long)t1, (unsigned long long)t2, f3, (unsigned long long)begin_idle,
                      (unsigned long long)begin_busy, (unsigned long long)(rb.addr > ra.addr ? rb.addr - ra.addr : ra.addr - rb.addr),
                      s1, s2, b_ok ? "yes" : "NO", tm.gpu_gap_before_ms);
        report("pipelined: two in flight", r == CERE_OK && t2 == t1 + 1 && f1 == 1 && f2 == 2 && refused_clean && w1 == CERE_OK &&
                                               lr1 == t1 && f3 == 1 && a_done && w2 == CERE_OK && c.in_flight() == 0 && n2 == n1 + 1 &&
                                               b_ok && parts && begin_idle == 1 && begin_busy == 2 && s1 > s2 && s2 > 0 &&
                                               tm.ticket == t2 && tm.gpu_gap_before_ms >= 0,
               m);
        // waiting again for a retired ticket is OK; 0 and a never-submitted ticket are ARGUMENT; wait_all with nothing
        // in flight is OK
        const cere_status wa = c.wait(t1), wz = c.wait(0), wf = c.wait(t2 + 5), wl = c.wait_all();
        report("pipelined: wait misuse", wa == CERE_OK && wz == CERE_ERR_ARGUMENT && wf == CERE_ERR_ARGUMENT && wl == CERE_OK && !c.faulted(),
               "retired again -> " + st(wa) + ", wait(0) -> " + st(wz) + ", unsubmitted -> " + st(wf) + ", wait_all idle -> " + st(wl));
    }
    // (2) wait(later) retires the earlier one first; wait_all retires everything
    {
        uint64_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
        cere_status r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o1.addr, 500);
        if (r == CERE_OK) r = c.submit(&t1);
        if (r == CERE_OK) r = c.begin();
        if (r == CERE_OK) r = gathers(c, o1.addr, n, hidden, di2.addr, n, o2.addr, 1);
        if (r == CERE_OK) r = c.submit(&t2);
        const cere_status w = c.wait(t2);
        const bool both = c.last_retired() == t2 && c.in_flight() == 0;
        if (r == CERE_OK) r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o3.addr, 1);
        if (r == CERE_OK) r = c.submit(&t3);
        if (r == CERE_OK) r = c.begin();
        if (r == CERE_OK) r = c.submit(&t4);  // empty command buffer
        const cere_status wl = c.wait_all();
        report("pipelined: ordering", r == CERE_OK && w == CERE_OK && both && wl == CERE_OK && c.last_retired() == t4 && t4 == t1 + 3 &&
                                          c.in_flight() == 0 && expect_b(o2.map),
               "wait(t2) retired t1 and t2 (last_retired " + std::to_string(t2) + "); wait_all retired t3, t4 (empty cb); B bit-exact");
    }
    // (3) a latched recording error with another submit in flight: nothing submitted, no ticket used, the in-flight one
    // unaffected; submit without begin is ARGUMENT
    {
        uint64_t t1 = 0, tl = 7;
        cere_status r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o1.addr, 1);
        if (r == CERE_OK) r = c.submit(&t1);
        const uint64_t c0 = c.device_commands();
        if (r == CERE_OK) r = c.begin();
        const uint64_t pc[4] = {};
        c.dispatch(c.pipeline("no_such_kernel.spv", 16), pc, 1);
        const cere_status sl = c.submit(&tl);
        const uint64_t used = c.device_commands() - c0;
        const cere_status sn = c.submit();
        const cere_status w = c.wait(t1);
        report("pipelined: latched error in flight", r == CERE_OK && sl == CERE_ERR_UNSUPPORTED && tl == 0 && c.last_submitted() == t1 &&
                                                         sn == CERE_ERR_ARGUMENT && w == CERE_OK && c.last_retired() == t1 && !c.faulted(),
               "latched submit -> " + st(sl) + " ticket 0, last_submitted unchanged, " + std::to_string(used) +
                   " command(s) (the begin, its barrier) and no queue submit; submit without begin -> " + st(sn) + "; t1 retired OK");
    }
    // (4) one slot: submit without waiting, then begin refuses until wait
    {
        std::unique_ptr<Context> c1 = open_slots("pipelined: one slot", 1);
        if (c1) {
            uint64_t t = 0;
            cere_status r = c1->begin();
            if (r == CERE_OK) r = c1->submit(&t);
            const cere_status b = c1->begin();
            const cere_status w = c1->wait(t);
            const cere_status b2 = c1->begin();
            const cere_status s2 = c1->submit_wait();
            report("pipelined: one slot", r == CERE_OK && b == CERE_ERR_ARGUMENT && w == CERE_OK && b2 == CERE_OK && s2 == CERE_OK &&
                                              c1->max_in_flight() == 1,
                   "submit, begin -> " + st(b) + " (wait first), wait " + st(w) + ", then begin / submit_wait OK");
        }
    }
    // (5) GPU gap: back to back (in flight) vs serial with host work between
    {
        uint64_t t1 = 0, t2 = 0;
        cere_status r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o1.addr, 4000);
        if (r == CERE_OK) r = c.submit(&t1);
        if (r == CERE_OK) r = c.begin();  // recorded and submitted while t1 runs
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o3.addr, 1);
        if (r == CERE_OK) r = c.submit(&t2);
        if (r == CERE_OK) r = c.wait(t1);
        if (r == CERE_OK) r = c.wait(t2);
        const double piped = c.last_submit_timing().gpu_gap_before_ms;
        if (r == CERE_OK) r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o1.addr, 2000);
        if (r == CERE_OK) r = c.submit_wait();
        const auto h0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - h0 < std::chrono::milliseconds(5)) {}
        if (r == CERE_OK) r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o3.addr, 2000);
        if (r == CERE_OK) r = c.submit_wait();
        const double serial = c.last_submit_timing().gpu_gap_before_ms;
        char m[200];
        std::snprintf(m, sizeof m, "GPU idle between two command buffers: in flight %.4f ms, serial with 5 ms host work %.3f ms", piped, serial);
        report("pipelined: GPU gap", r == CERE_OK && piped >= 0 && piped < 1.0 && serial >= 5.0, m);
    }
    // (5b) VulkanDecode: wait(t, spin_from_ns) blocks until spin_from_ns, then polls the timeline (at most
    // Context::kMaxSpinNs) before the bounded blocking wait; the same retire (marker, stamps) as wait(t). spin_from in
    // the past polls at once; far in the future still returns at the GPU end (the blocking part ends on the signal).
    {
        const auto now = [] {
            return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch()).count());
        };
        std::memset(o1.map, 0, n * hidden * 2);
        uint64_t t1 = 0, t2 = 0, t3 = 0;
        cere_status r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o1.addr, 3000);
        if (r == CERE_OK) r = c.submit(&t1);
        const int64_t s1 = now();
        const cere_status w1 = r == CERE_OK ? c.wait(t1, s1 + 1000000) : r;  // block 1 ms, then poll
        const double ms1 = double(now() - s1) * 1e-6, g1 = c.last_submit_gpu_ms();
        bool a_ok = true;
        for (int64_t t = 0; t < n; ++t)
            a_ok = a_ok && std::memcmp(static_cast<const uint16_t *>(o1.map) + t * hidden,
                                       &table[size_t(ids1[size_t(t)]) * hidden], hidden * 2) == 0;
        if (r == CERE_OK) r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o3.addr, 1000);
        if (r == CERE_OK) r = c.submit(&t2);
        const cere_status w2 = r == CERE_OK ? c.wait(t2, 0) : r;  // in the past: poll at once
        if (r == CERE_OK) r = c.begin();
        if (r == CERE_OK) r = gathers(c, dt.addr, vocab, hidden, di1.addr, n, o3.addr, 1);
        if (r == CERE_OK) r = c.submit(&t3);
        const int64_t s3 = now();
        const cere_status w3 = r == CERE_OK ? c.wait(t3, s3 + 10000000000ll) : r;  // 10 s ahead
        const double ms3 = double(now() - s3) * 1e-6;
        char m[240];
        std::snprintf(m, sizeof m,
                      "spin from +1 ms: %s in %.3f ms (GPU %.3f ms), output bit-exact %s; from the past: %s; from +10 s: %s "
                      "in %.3f ms",
                      st(w1).c_str(), ms1, g1, a_ok ? "yes" : "NO", st(w2).c_str(), st(w3).c_str(), ms3);
        report("pipelined: spin wait", r == CERE_OK && w1 == CERE_OK && w2 == CERE_OK && w3 == CERE_OK && a_ok && g1 > 0 &&
                                           c.last_retired() == t3 && c.in_flight() == 0 && ms3 < 1000.0 && !c.faulted(),
               m);
    }
    c.free(dt); c.free(di1); c.free(di2); c.free(o1); c.free(o2); c.free(o3);

    // (6) kill ordering, simulated (the GPU runs normally): the EARLIER of two in-flight submits loses its marker (its
    // command buffer does not record the write: the word stays as the host cleared it); the later one's marker is
    // fine. wait(later) faults on the earlier, neither is reported complete, every later call refuses untouched.
    {
        std::unique_ptr<Context> ck = open_slots("kill ordering: earlier", 2);
        if (ck) {
            vk::Buffer b;
            ck->allocate(8192, b);
            std::memset(b.map, 0, 8192);
            uint64_t t0 = 0, t1 = 0, t2 = 0;
            cere_status r = ck->begin();
            if (r == CERE_OK) r = ck->submit_wait();  // one good submit first
            t0 = ck->last_retired();
            const uint64_t good = ck->last_completed_nonce();
            ck->omit_completion_marker_for_test();
            if (r == CERE_OK) r = ck->begin();
            if (r == CERE_OK) r = vkk1::embedding_gather(*ck, b.addr, 4, 64, b.addr + 4096, 1, b.addr + 1024);
            if (r == CERE_OK) r = ck->submit(&t1);
            if (r == CERE_OK) r = ck->begin();
            if (r == CERE_OK) r = vkk1::embedding_gather(*ck, b.addr, 4, 64, b.addr + 4096, 1, b.addr + 2048);
            if (r == CERE_OK) r = ck->submit(&t2);
            const cere_status w = ck->wait(t2);
            const uint64_t c0 = ck->device_commands();
            const cere_status w2 = ck->wait(t2), wa = ck->wait_all(), bb = ck->begin(), ss = ck->submit();
            const std::string &why = ck->fault_reason();
            const bool reason = why.find("completion marker") != std::string::npos &&
                                why.find("ticket " + std::to_string(t1)) != std::string::npos &&
                                why.find("1 later in-flight submit(s) discarded") != std::string::npos;
            report("kill ordering: earlier killed", r == CERE_OK && w == CERE_ERR_INTERNAL && ck->faulted() && reason &&
                                                        ck->last_retired() == t0 && ck->last_completed_nonce() == good &&
                                                        w2 == CERE_ERR_INTERNAL && wa == CERE_ERR_INTERNAL && bb == CERE_ERR_INTERNAL &&
                                                        ss == CERE_ERR_INTERNAL && ck->device_commands() == c0 &&
                                                        ck->last_submit_gpu_ms() < 0,
                       "wait(t" + std::to_string(t2) + ") -> " + st(w) + ": t" + std::to_string(t1) + " (no marker) and t" +
                           std::to_string(t2) + " (marker written) both unretired, last_retired " + std::to_string(t0) +
                           "; later wait/wait_all/begin/submit " + st(CERE_ERR_INTERNAL) + " without device commands; reason \"" +
                           why + "\"");
            ck->free(b);
        }
    }
    // (7) the LATER of two in-flight submits is killed (skewed nonce): the earlier retires fine, the later faults.
    {
        std::unique_ptr<Context> ck = open_slots("kill ordering: later", 2);
        if (ck) {
            vk::Buffer b;
            ck->allocate(8192, b);
            std::memset(b.map, 0, 8192);
            uint64_t t1 = 0, t2 = 0;
            cere_status r = ck->begin();
            if (r == CERE_OK) r = vkk1::embedding_gather(*ck, b.addr, 4, 64, b.addr + 4096, 1, b.addr + 1024);
            if (r == CERE_OK) r = ck->submit(&t1);
            ck->skew_completion_nonce_for_test();
            if (r == CERE_OK) r = ck->begin();
            if (r == CERE_OK) r = vkk1::embedding_gather(*ck, b.addr, 4, 64, b.addr + 4096, 1, b.addr + 2048);
            if (r == CERE_OK) r = ck->submit(&t2);
            const cere_status w1 = ck->wait(t1);
            const uint64_t lr = ck->last_retired();
            const cere_status w2 = ck->wait(t2);
            const bool reason = ck->fault_reason().find("ticket " + std::to_string(t2)) != std::string::npos &&
                                ck->fault_reason().find("discarded") == std::string::npos;
            report("kill ordering: later killed", r == CERE_OK && w1 == CERE_OK && lr == t1 && w2 == CERE_ERR_INTERNAL && ck->faulted() &&
                                                      ck->last_retired() == t1 && reason,
                   "wait(t1) " + st(w1) + " (retired, verified), wait(t2) " + st(w2) + ", faulted; last_retired stays " +
                       std::to_string(t1) + "; reason \"" + ck->fault_reason() + "\"");
            ck->free(b);
        }
    }
    // (8) the serial path's own marker check still catches a dropped marker write (one slot, submit_wait)
    {
        std::unique_ptr<Context> ck = open_slots("kill: serial omitted marker", 1);
        if (ck) {
            ck->omit_completion_marker_for_test();
            cere_status r = ck->begin();
            const cere_status s = r == CERE_OK ? ck->submit_wait() : r;
            report("kill: serial omitted marker", s == CERE_ERR_INTERNAL && ck->faulted() && ck->last_completed_nonce() == 0 &&
                                                      ck->fault_reason().find("found 0x0000000000000000") != std::string::npos,
                   "submit_wait -> " + st(s) + "; reason \"" + ck->fault_reason() + "\"");
        }
    }
    // (8b) the spin wait checks the marker the same way (a dropped marker write faults after the poll saw the signal)
    {
        std::unique_ptr<Context> ck = open_slots("kill: spin wait omitted marker", 1);
        if (ck) {
            ck->omit_completion_marker_for_test();
            uint64_t t = 0;
            cere_status r = ck->begin();
            if (r == CERE_OK) r = ck->submit(&t);
            const cere_status w = r == CERE_OK ? ck->wait(t, 0) : r;
            report("kill: spin wait omitted marker", w == CERE_ERR_INTERNAL && ck->faulted() && ck->last_retired() == 0 &&
                                                         ck->fault_reason().find("found 0x0000000000000000") != std::string::npos,
                   "wait(t, spin from the past) -> " + st(w) + "; reason \"" + ck->fault_reason() + "\"");
        }
    }
}

// Opt-in (--queue-probe MS COUNT): COUNT command buffers of about MS ms of GPU work each (K1 copies of 256 MiB), at most
// two in flight, the GPU busy back to back for about COUNT x MS ms. Checks every marker, reports each command buffer's
// GPU time and the gaps. Purpose: verify that the amdgpu job timeout (~2 s here) applies per command buffer, not to a
// queue that never drains. Run only under the GPU lock; dmesg before / after.
static void queue_probe(double ms, int count) {
    std::unique_ptr<Context> cp = open_slots("queue probe", 2);
    if (!cp) return;
    Context &c = *cp;
    const uint64_t bytes = 256ull << 20;
    vk::Buffer a, b;
    if (!(make(c, bytes, a, "queue probe") && make(c, bytes, b, "queue probe"))) return;
    const vkk1::CopySegment seg{b.addr, bytes, a.addr, bytes, bytes, 1};
    const auto record = [&](int copies) {
        cere_status r = c.begin();
        for (int k = 0; k < copies && r == CERE_OK; ++k) {
            vk::TableRegion t;
            r = c.table_alloc(sizeof seg, 16, &t);
            vk::Buffer tb = t.view();
            if (r == CERE_OK) r = vkk1::copy_segments(c, &seg, 1, tb);
            if (r == CERE_OK) r = c.barrier();
        }
        return r;
    };
    // calibrate: 4 copies, serial
    cere_status r = record(4);
    if (r == CERE_OK) r = c.submit_wait();
    const double per = c.last_submit_gpu_ms() / 4;
    const int copies = per > 0 ? std::max(1, int(ms / per)) : 1;
    std::printf("queue probe: %.3f ms per 256 MiB copy, %d copies per command buffer (target %.0f ms), %d command buffers\n", per,
                copies, ms, count);
    std::vector<uint64_t> tickets;
    std::vector<double> gpu, gap;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < count && r == CERE_OK; ++i) {
        if (c.in_flight() == 2) {
            r = c.wait(tickets[gpu.size()]);
            if (r != CERE_OK) break;
            gpu.push_back(c.last_submit_gpu_ms());
            gap.push_back(c.last_submit_timing().gpu_gap_before_ms);
        }
        uint64_t t = 0;
        if ((r = record(copies)) == CERE_OK) r = c.submit(&t);
        tickets.push_back(t);
    }
    while (r == CERE_OK && gpu.size() < tickets.size()) {
        r = c.wait(tickets[gpu.size()]);
        if (r != CERE_OK) break;
        gpu.push_back(c.last_submit_gpu_ms());
        gap.push_back(c.last_submit_timing().gpu_gap_before_ms);
    }
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    double sum = 0, mx = 0, gmax = 0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        sum += gpu[i];
        mx = std::max(mx, gpu[i]);
        if (i) gmax = std::max(gmax, gap[i]);
        std::printf("  cb %zu: gpu %.1f ms, gap before %.3f ms\n", i, gpu[i], gap[i]);
    }
    char m[300];
    std::snprintf(m, sizeof m, "%zu of %d command buffers retired, markers OK; GPU busy %.1f ms of %.1f ms wall, longest cb %.1f ms, "
                  "max gap %.3f ms; status %d%s", gpu.size(), count, sum, wall, mx, gmax, int(r),
                  c.faulted() ? (" FAULT: " + c.fault_reason()).c_str() : "");
    report("queue probe (opt-in)", r == CERE_OK && int(gpu.size()) == count && !c.faulted(), m);
    c.free(a);
    c.free(b);
}

int main(int argc, char **argv) {
    const std::string spv_dir = argc > 1 ? argv[1] : "build/vulkan/spv";
    double queue_probe_ms = 0;
    int queue_probe_count = 0;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--queue-probe" && i + 2 < argc) {
            queue_probe_ms = std::atof(argv[i + 1]);
            queue_probe_count = std::atoi(argv[i + 2]);
        }

    // --- create / ordinals ----------------------------------------------------------------------------------------
    uint32_t n_gpu = 0, n_all = 0;
    std::string why;
    const cere_status cs = Context::count_devices(&n_gpu, false, &why);
    Context::count_devices(&n_all, true, &why);
    report("count_devices", cs == CERE_OK && n_gpu >= 1 && n_all >= n_gpu,
           std::to_string(n_gpu) + " GPU(s), " + std::to_string(n_all) + " with software devices");
    std::unique_ptr<Context> ctxp;
    why.clear();
    cere_status s = Context::create(0, &ctxp, &why);
    if (s != CERE_OK || !ctxp) {
        report("create ordinal 0", false, "status " + st(s) + ": " + why);
        std::printf("vk_launch_test: %d PASS, %d FAIL\n", g_pass, g_fail + 1);
        return 1;
    }
    Context &ctx = *ctxp;
    const vk::DeviceInfo &inf = ctx.info();
    char d[512];
    std::snprintf(d, sizeof d, "%s (vendor 0x%04x device 0x%04x, type %d), driver %s %s, ordinal %u, coopmat %d fp64 %d int-dot %d",
                  inf.name.c_str(), inf.vendor_id, inf.device_id, int(inf.device_type), inf.driver_name.c_str(), inf.driver_info.c_str(),
                  inf.ordinal, int(inf.coopmat), int(inf.float64), int(inf.integer_dot));
    report("create ordinal 0", inf.ordinal == 0 && !ctx.faulted() && inf.float64 && inf.integer_dot &&
                                   inf.device_type != VK_PHYSICAL_DEVICE_TYPE_CPU && inf.vendor_id != 0,
           d);
    {
        std::unique_ptr<Context> bad;
        std::string w1, w2;
        const cere_status s1 = Context::create(n_gpu, &bad, &w1);
        const bool null1 = !bad;
        const cere_status s2 = Context::create(n_gpu + 100, &bad, &w2);
        report("create out-of-range ordinal", s1 == CERE_ERR_UNSUPPORTED && s2 == CERE_ERR_UNSUPPORTED && null1 && !bad && !w1.empty(),
               "ordinal " + std::to_string(n_gpu) + " -> " + st(s1) + " \"" + w1 + "\"; +100 -> " + st(s2) + "; process alive");
        vk::CreateOptions o;
        o.wait_timeout_ns = 0;
        std::string w3;
        const cere_status s3 = Context::create(0, &bad, &w3, o);
        report("create bad options", s3 == CERE_ERR_ARGUMENT && !bad, "wait_timeout_ns 0 -> " + st(s3) + " \"" + w3 + "\"");
    }

    // --- allocation limits ----------------------------------------------------------------------------------------
    {
        vk::Buffer b;
        std::string w1, w2;
        const cere_status s1 = ctx.allocate(inf.max_buffer + 1, b, false, &w1);
        const bool empty1 = b.buf == VK_NULL_HANDLE && b.map == nullptr;
        const cere_status s2 = ctx.allocate(0, b, false, &w2);
        const bool ok_bool = !ctx.alloc(inf.max_alloc + 1, b) && !ctx.last_error().empty();
        report("over-limit buffer", s1 == CERE_ERR_CAPACITY && empty1 && s2 == CERE_ERR_ARGUMENT && ok_bool && !ctx.faulted(),
               "maxBufferSize+1 -> " + st(s1) + " \"" + w1 + "\"; 0 B -> " + st(s2) + "; alloc() false with last_error");
        vk::Buffer h;
        const cere_status s3 = ctx.allocate(4096, h, true, &w1);
        const VkMemoryPropertyFlags f = ctx.memory_flags(h.memory_type);
        report("host-heap allocation", s3 == CERE_OK && h.map && h.addr && !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                                           (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
               "status " + st(s3) + ", memory type " + std::to_string(h.memory_type) + " flags 0x" + std::to_string(f));
        ctx.free(h);
    }

    // --- embedded SPIR-V ------------------------------------------------------------------------------------------
    {
        size_t n = 0;
        const vk::EmbeddedShader *t = vk::embedded_shaders(&n);
        size_t same = 0, missing = 0;
        bool sorted = true;
        for (size_t i = 0; i < n; ++i) {
            if (i && std::strcmp(t[i - 1].name, t[i].name) >= 0) sorted = false;
            std::ifstream f(spv_dir + "/" + t[i].name, std::ios::binary);
            if (!f) { ++missing; continue; }
            std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            same += bytes.size() == t[i].bytes && std::memcmp(bytes.data(), t[i].words, t[i].bytes) == 0 && std::strlen(t[i].sha256) == 64;
        }
        report("embedded table", n > 0 && same == n && sorted,
               std::to_string(n) + " modules, " + std::to_string(same) + " byte-equal to " + spv_dir + " (" + std::to_string(missing) +
                   " missing there), sorted " + (sorted ? "yes" : "NO"));
    }
    std::mt19937 g(7);
    {   // a kernel through the embedded table only: no shader directory is set on this context
        const int64_t vocab = 300; const int hidden = 64; const int64_t n = 9;
        auto table = randbf(size_t(vocab * hidden), g, 1.f);
        std::vector<int32_t> ids(n);
        for (auto &i : ids) i = int32_t(g() % vocab);
        vk::Buffer dt, di, dout;
        if (make(ctx, table.size() * 2, dt, "embedded path") && make(ctx, ids.size() * 4, di, "embedded path") &&
            make(ctx, uint64_t(n) * hidden * 2, dout, "embedded path")) {
            std::memcpy(dt.map, table.data(), table.size() * 2);
            std::memcpy(di.map, ids.data(), ids.size() * 4);
            cere_status r = ctx.begin();
            if (r == CERE_OK) r = vkk1::embedding_gather(ctx, dt.addr, vocab, hidden, di.addr, n, dout.addr);
            if (r == CERE_OK) r = ctx.barrier();
            if (r == CERE_OK) r = ctx.submit_wait();
            bool ok = r == CERE_OK;
            const uint16_t *o = static_cast<const uint16_t *>(dout.map);
            for (int64_t t = 0; t < n && ok; ++t) ok = std::memcmp(o + t * hidden, &table[size_t(ids[t]) * hidden], hidden * 2) == 0;
            report("embedded-shader path", ok, "embedding_gather (k1_embedding.spv from the table, no shader dir): status " + st(r) +
                                                   ", bit-exact rows");
        }
        ctx.free(dt); ctx.free(di); ctx.free(dout);
    }

    // --- missing shader -------------------------------------------------------------------------------------------
    {
        const vk::Pipeline *p = nullptr;
        std::string w;
        const cere_status s1 = ctx.pipeline(vk::PipelineSpec{"no_such_kernel.spv", 16, 0, {}}, &p, &w);
        // the kernels' recording idiom: the failure latches, nothing is submitted, the context is not faulted
        const uint64_t c0 = ctx.device_commands();
        cere_status b = ctx.begin();
        const uint64_t pc[4] = {};
        const cere_status s2 = ctx.dispatch(ctx.pipeline("no_such_kernel.spv", 16), pc, 1);
        const cere_status s3 = ctx.barrier();
        const cere_status s4 = ctx.submit_wait();
        const uint64_t used = ctx.device_commands() - c0;
        report("missing shader", s1 == CERE_ERR_UNSUPPORTED && !p && !w.empty() && b == CERE_OK && s2 == CERE_ERR_UNSUPPORTED &&
                                     s3 == CERE_ERR_UNSUPPORTED && s4 == CERE_ERR_UNSUPPORTED && used == 1 && !ctx.faulted() && !ctx.recording(),
               "pipeline() " + st(s1) + " \"" + w + "\"; dispatch " + st(s2) + ", later barrier " + st(s3) + ", submit_wait " + st(s4) +
                   " (commands after begin: " + std::to_string(used - 1) + ", no submit); not faulted");
        // the next command buffer is clean
        b = ctx.begin();
        const cere_status s5 = ctx.submit_wait();
        report("after a latched error", b == CERE_OK && s5 == CERE_OK, "begin " + st(b) + ", empty submit_wait " + st(s5));
        std::string w2, w3;
        const cere_status s6 = ctx.pipeline(vk::PipelineSpec{"k1_embedding.spv", 40, 7, {}}, &p, &w2);
        const cere_status s7 = ctx.pipeline(vk::PipelineSpec{"k1_embedding.spv", 132, 0, {}}, &p, &w3);
        report("bad pipeline keys", s6 == CERE_ERR_UNSUPPORTED && s7 == CERE_ERR_ARGUMENT,
               "subgroup 7 -> " + st(s6) + " \"" + w2 + "\"; push 132 B -> " + st(s7));
    }

    // --- warm-up ----------------------------------------------------------------------------------------------------
    {
        std::unique_ptr<Context> c2 = open0("warm-up");
        if (c2) {
            const auto specs = vkk1::pipeline_specs({64, 2560}, {256});
            bool none = true;
            for (auto &sp : specs) none = none && !c2->has_pipeline(sp);
            std::string w;
            const cere_status sw = c2->warm(specs, &w);
            bool all = true;
            for (auto &sp : specs) all = all && c2->has_pipeline(sp);
            const size_t n0 = c2->pipeline_count();
            // dispatch after the warm-up: rmsnorm at dim 64 creates nothing new
            const int rows = 3, dim = 64;
            auto x = randbf(size_t(rows * dim), g, 2.f), wt = randbf(dim, g, 0.5f);
            vk::Buffer dx, dw, dy;
            bool ran = false, close = true;
            if (make(*c2, x.size() * 2, dx, "warm-up") && make(*c2, wt.size() * 2, dw, "warm-up") && make(*c2, x.size() * 2, dy, "warm-up")) {
                std::memcpy(dx.map, x.data(), x.size() * 2);
                std::memcpy(dw.map, wt.data(), wt.size() * 2);
                cere_status r = c2->begin();
                if (r == CERE_OK) r = vkk1::rmsnorm(*c2, dx.addr, dim, dw.addr, 1, false, 1e-6f, rows, dim, dy.addr, dim);
                if (r == CERE_OK) r = c2->barrier();
                if (r == CERE_OK) r = c2->submit_wait();
                ran = r == CERE_OK;
                const uint16_t *y = static_cast<const uint16_t *>(dy.map);
                for (int rr = 0; rr < rows; ++rr) {
                    double ss = 0;
                    for (int i = 0; i < dim; ++i) ss += double(bf2f(x[rr * dim + i])) * bf2f(x[rr * dim + i]);
                    const double inv = 1.0 / std::sqrt(ss / dim + 1e-6);
                    for (int i = 0; i < dim; ++i) {
                        const double ref = bf2f(x[rr * dim + i]) * inv * bf2f(wt[i]);
                        close = close && std::fabs(bf2f(y[rr * dim + i]) - ref) <= 1e-2 * std::fabs(ref) + 1e-3;
                    }
                }
            }
            c2->free(dx); c2->free(dw); c2->free(dy);
            std::string wb;
            const cere_status sb = c2->warm({vk::PipelineSpec{"k1_rope.spv", 56, 0, {}}, vk::PipelineSpec{"missing_one.spv", 8, 0, {}}}, &wb);
            report("warm-up then dispatch", none && sw == CERE_OK && all && n0 == specs.size() && ran && close &&
                                                c2->pipeline_count() == n0,  // k1_rope of the negative list was warmed already
                   std::to_string(specs.size()) + " K1 specs: none existed, warm " + st(sw) + ", all exist; rmsnorm dim 64 ran " +
                       (ran ? "ok" : "NO") + (close ? ", matches host" : ", DIFFERS") + ", no pipeline created at dispatch");
            report("warm-up negative", sb == CERE_ERR_UNSUPPORTED && wb.find("missing_one.spv") != std::string::npos,
                   "list with a missing module -> " + st(sb) + " \"" + wb + "\"");
        }
    }

    // --- begin / submit misuse --------------------------------------------------------------------------------------
    {
        const cere_status a = ctx.barrier(), b = ctx.stamp(0), c = ctx.submit_wait();
        vk::TableRegion tr;
        const cere_status t0 = ctx.table_alloc(16, 16, &tr);
        report("recording outside begin", a == CERE_ERR_ARGUMENT && b == CERE_ERR_ARGUMENT && c == CERE_ERR_ARGUMENT &&
                                              t0 == CERE_ERR_ARGUMENT && !ctx.faulted(),
               "barrier " + st(a) + ", stamp " + st(b) + ", submit_wait " + st(c) + ", table_alloc " + st(t0));
        // second begin without submit: refused, the open command buffer survives and runs
        vk::Buffer dt, di, dout;
        const int hidden = 64;
        auto table = randbf(size_t(4 * hidden), g, 1.f);
        std::vector<int32_t> ids = {3, 1};
        if (make(ctx, table.size() * 2, dt, "second begin") && make(ctx, 8, di, "second begin") && make(ctx, 2 * hidden * 2, dout, "second begin")) {
            std::memcpy(dt.map, table.data(), table.size() * 2);
            std::memcpy(di.map, ids.data(), 8);
            const cere_status b1 = ctx.begin();
            const cere_status k = vkk1::embedding_gather(ctx, dt.addr, 4, hidden, di.addr, 2, dout.addr);
            const cere_status b2 = ctx.begin();
            const bool still = ctx.recording() && ctx.record_status() == CERE_OK;
            ctx.barrier();
            const cere_status sw = ctx.submit_wait();
            const uint16_t *o = static_cast<const uint16_t *>(dout.map);
            const bool out_ok = std::memcmp(o, &table[3 * hidden], hidden * 2) == 0 && std::memcmp(o + hidden, &table[1 * hidden], hidden * 2) == 0;
            report("second begin without submit", b1 == CERE_OK && k == CERE_OK && b2 == CERE_ERR_ARGUMENT && still && sw == CERE_OK && out_ok,
                   "second begin -> " + st(b2) + "; the open command buffer kept its work and ran (output bit-exact)");
        }
        ctx.free(dt); ctx.free(di); ctx.free(dout);
        const cere_status b3 = ctx.begin();
        const cere_status s1 = ctx.stamp(vk::Context::kStamps);
        const cere_status s2 = ctx.submit_wait();
        report("bad timestamp index", b3 == CERE_OK && s1 == CERE_ERR_ARGUMENT && s2 == CERE_ERR_ARGUMENT && !ctx.faulted(),
               "stamp(kStamps) -> " + st(s1) + ", latched: submit_wait " + st(s2) + " (nothing submitted)");
    }

    // --- table ring ---------------------------------------------------------------------------------------------------
    {
        const uint64_t ring = ctx.table_ring_bytes();
        vk::TableRegion r1, r2, r3, r4, bad;
        cere_status s0 = ctx.begin();
        const cere_status a1 = ctx.table_alloc(24, 16, &r1);
        const cere_status a2 = ctx.table_alloc(1, 1, &r2);
        const cere_status a3 = ctx.table_alloc(8, 256, &r3);
        const cere_status a4 = ctx.table_alloc(8, 3, &bad);
        const uint64_t used = ctx.table_ring_used();
        const cere_status a5 = ctx.table_alloc(ring, 16, &bad);
        const bool unchanged = ctx.table_ring_used() == used && bad.host == nullptr;
        ctx.submit_wait();
        const bool aligned = r1.addr % 16 == 0 && r3.addr % 256 == 0 && r2.addr == r1.addr + 24 && r3.addr >= r2.addr + 1 && r1.host && r3.host &&
                             static_cast<uint8_t *>(r3.host) - static_cast<uint8_t *>(r1.host) == int64_t(r3.addr - r1.addr);
        report("table ring allocation", s0 == CERE_OK && a1 == CERE_OK && a2 == CERE_OK && a3 == CERE_OK && aligned,
               "ring " + std::to_string(ring) + " B: 24 B @16, 1 B @1, 8 B @256: addresses aligned, host/device offsets agree");
        report("table ring refusals", a4 == CERE_ERR_ARGUMENT && a5 == CERE_ERR_CAPACITY && unchanged && !ctx.faulted(),
               "align 3 -> " + st(a4) + "; " + std::to_string(ring) + " B more (overflow) -> " + st(a5) + ", ring state unchanged");
        s0 = ctx.begin();
        const uint64_t after_begin = ctx.table_ring_used();
        const cere_status a6 = ctx.table_alloc(24, 16, &r4);
        ctx.submit_wait();
        report("table ring reset at begin", s0 == CERE_OK && after_begin == 0 && a6 == CERE_OK && r4.addr == r1.addr,
               "used " + std::to_string(after_begin) + " after begin; first region reused");
        // fill to the end, then exactly overflow by one byte
        s0 = ctx.begin();
        vk::TableRegion full, one;
        const cere_status f1 = ctx.table_alloc(ring, 1, &full);
        const cere_status f2 = ctx.table_alloc(1, 1, &one);
        ctx.submit_wait();
        report("table ring wrap / full", f1 == CERE_OK && f2 == CERE_ERR_CAPACITY, "whole ring -> " + st(f1) + ", one more byte -> " + st(f2));
    }
    {   // a ring region as kv_store_fp8_rows' caller-provided table (the kernel's signature unchanged)
        const int width = 64, T = 4;
        const int64_t slots_n = 8;
        auto k = randbf(size_t(T * width), g, 60.f), v = randbf(size_t(T * width), g, 0.5f);
        vk::Buffer dk, dv, kc[2], vc[2], sm[T], at, ref_table;
        bool ok = make(ctx, k.size() * 2, dk, "ring + kv_store_fp8_rows") && make(ctx, v.size() * 2, dv, "ring + kv_store_fp8_rows") &&
                  make(ctx, T * 8, at, "ring + kv_store_fp8_rows");
        for (int c = 0; c < 2 && ok; ++c)
            ok = make(ctx, slots_n * width, kc[c], "ring + kv_store_fp8_rows") && make(ctx, slots_n * width, vc[c], "ring + kv_store_fp8_rows");
        std::vector<std::vector<int64_t>> maps(T);
        for (int t = 0; t < T && ok; ++t) {
            maps[t] = {int64_t((t * 3 + 1) % slots_n), int64_t((t * 5 + 2) % slots_n), int64_t((t + 7) % slots_n)};
            ok = make(ctx, 3 * 8, sm[t], "ring + kv_store_fp8_rows");
            if (ok) std::memcpy(sm[t].map, maps[t].data(), 3 * 8);
        }
        if (ok) {
            std::memcpy(dk.map, k.data(), k.size() * 2);
            std::memcpy(dv.map, v.data(), v.size() * 2);
            std::vector<int64_t> atv = {0, 1, 0, 1};  // slots: row 0 -> 1, row 2 -> 7 (cache 0); row 1 -> 7, row 3 -> 1 (cache 1)
            std::memcpy(at.map, atv.data(), T * 8);
            vkk1::KvRows rows{};
            for (int t = 0; t < T; ++t) { rows.k_cache[t] = kc[t % 2].addr; rows.v_cache[t] = vc[t % 2].addr; rows.slots[t] = sm[t].addr; }
            for (int c = 0; c < 2; ++c) { std::memset(kc[c].map, 0xAB, slots_n * width); std::memset(vc[c].map, 0xAB, slots_n * width); }
            vk::TableRegion pad, tr;
            cere_status r = ctx.begin();
            if (r == CERE_OK) r = ctx.table_alloc(40, 16, &pad);  // the table does not start at the ring's offset 0
            if (r == CERE_OK) r = ctx.table_alloc(24 * vkk1::kKvRowsMax, 16, &tr);
            vk::Buffer view = tr.view();
            if (r == CERE_OK) r = vkk1::kv_store_fp8_rows(ctx, dk.addr, width, dv.addr, width, T, width, rows, at.addr, view);
            if (r == CERE_OK) r = ctx.barrier();
            if (r == CERE_OK) r = ctx.submit_wait();
            // host reference: row t -> cache t % 2 at slot maps[t][at[t]]
            std::vector<uint8_t> ek[2], ev[2];
            for (int c = 0; c < 2; ++c) { ek[c].assign(slots_n * width, 0xAB); ev[c].assign(slots_n * width, 0xAB); }
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < width; ++i) {
                    const int64_t sl = maps[t][atv[t]];
                    ek[t % 2][sl * width + i] = fp8_ref(bf2f(k[t * width + i]));
                    ev[t % 2][sl * width + i] = fp8_ref(bf2f(v[t * width + i]));
                }
            int64_t diff = 0;
            for (int c = 0; c < 2; ++c)
                for (int64_t i = 0; i < slots_n * width; ++i)
                    diff += (static_cast<uint8_t *>(kc[c].map)[i] != ek[c][i]) + (static_cast<uint8_t *>(vc[c].map)[i] != ev[c][i]);
            ctx.free(view);  // a view: no-op
            report("ring region as a kernel table", r == CERE_OK && diff == 0 && tr.addr % 16 == 0 && view.buf == VK_NULL_HANDLE,
                   "kv_store_fp8_rows, 4 rows into 2 caches, table in the ring at offset " + std::to_string(tr.addr - pad.addr) +
                       " past another region: status " + st(r) + ", " + std::to_string(diff) + " bytes differ from the host reference");
        }
        for (auto *b : {&dk, &dv, &kc[0], &kc[1], &vc[0], &vc[1], &at, &ref_table}) ctx.free(*b);
        for (auto &b : sm) ctx.free(b);
    }

    // --- per-context probe state; unprobed context refused before recording -----------------------------------------
    {
        std::unique_ptr<Context> c3 = open0("probe state");
        if (c3) {
            const bool unprobed = c3->probe(vk::Probe::kK2GemmLayout) == vk::ProbeState::kUnprobed &&
                                  c3->probe(vk::Probe::kK7MoeLayout) == vk::ProbeState::kUnprobed && !vkk2::gemm_ready(*c3);
            // K3 hc_mix at row class 20 on the unprobed context: refused before its norm is recorded. Fake 256-aligned
            // addresses: nothing may be recorded, so nothing is ever submitted with them.
            vkk3::HcPlans plans;
            const cere_status pi = plans.init(vkk3::HcShape{4, 2560, 64, 1e-6f}, 32);
            const uint64_t fake = uint64_t(1) << 40;
            cere_status b = c3->begin();
            const uint64_t c0 = c3->device_commands();
            const cere_status h = vkk3::hc_mix(*c3, plans, fake, fake + 0x10000, fake + 0x20000, fake + 0x30000, 20, fake + 0x40000,
                                               fake + 0x50000, fake + 0x100000, vkk3::HcMixPath::kAuto, 0);
            const uint64_t recorded = c3->device_commands() - c0;
            const cere_status sw = c3->submit_wait();
            report("unprobed context refused early", unprobed && pi == CERE_OK && b == CERE_OK && h == CERE_ERR_UNSUPPORTED && recorded == 0 &&
                                                         sw == CERE_OK,
                   "hc_mix m 20 (K2 GEMM class) on an unprobed context -> " + st(h) + ", " + std::to_string(recorded) +
                       " commands recorded; the empty command buffer submits " + st(sw));
            const cere_status gp = vkk2::gemm_probe(*c3);
            const bool pass = c3->probe(vk::Probe::kK2GemmLayout) == vk::ProbeState::kPass && vkk2::gemm_ready(*c3);
            const bool other = ctx.probe(vk::Probe::kK2GemmLayout) == vk::ProbeState::kUnprobed && !vkk2::gemm_ready(ctx);
            report("probe state is per context", gp == CERE_OK && pass && other,
                   "gemm_probe " + st(gp) + " on context B: B kPass / gemm_ready; context A still kUnprobed");
        }
    }

    // --- completion marker and submit GPU time (P3 T1b) -----------------------------------------------------------------
    {
        const int hidden = 64, n = 4;
        auto table = randbf(size_t(8 * hidden), g, 1.f);
        std::vector<int32_t> ids = {1, 7, 3, 0};
        vk::Buffer dt, di, dout;
        if (make(ctx, table.size() * 2, dt, "completion marker") && make(ctx, 16, di, "completion marker") &&
            make(ctx, uint64_t(n) * hidden * 2, dout, "completion marker")) {
            std::memcpy(dt.map, table.data(), table.size() * 2);
            std::memcpy(di.map, ids.data(), 16);
            auto one = [&](int calls) {
                cere_status r = ctx.begin();
                if (r == CERE_OK) r = ctx.stamp(0);
                for (int c = 0; c < calls && r == CERE_OK; ++c) {
                    r = vkk1::embedding_gather(ctx, dt.addr, 8, hidden, di.addr, n, dout.addr);
                    if (r == CERE_OK) r = ctx.barrier();
                }
                if (r == CERE_OK) r = ctx.stamp(1);
                if (r == CERE_OK) r = ctx.submit_wait();
                return r;
            };
            const cere_status r1 = one(1);
            const uint64_t n1 = ctx.last_completed_nonce();
            const double g1 = ctx.last_submit_gpu_ms();
            const cere_status r2 = one(1);
            const uint64_t n2 = ctx.last_completed_nonce();
            const cere_status r3 = ctx.begin() == CERE_OK ? ctx.submit_wait() : CERE_ERR_INTERNAL;  // an empty command buffer
            const uint64_t n3 = ctx.last_completed_nonce();
            const bool out_ok = std::memcmp(dout.map, &table[1 * hidden], hidden * 2) == 0;
            char m[256];
            std::snprintf(m, sizeof m, "3 submits OK, nonces 0x%016llx -> +1 -> +1 (empty cb too), output bit-exact %s",
                          (unsigned long long)n1, out_ok ? "yes" : "NO");
            report("completion marker positive", r1 == CERE_OK && r2 == CERE_OK && r3 == CERE_OK && n1 != 0 && n2 == n1 + 1 && n3 == n2 + 1 &&
                                                     out_ok && !ctx.faulted() && g1 > 0,
                   m);
            // GPU time: the frame (begin .. end stamps) covers the caller's stamps 0 .. 1 around the work
            const cere_status r4 = one(400);
            const double frame = ctx.last_submit_gpu_ms(), inner = ctx.stamp_ms(0, 1);
            std::snprintf(m, sizeof m, "1 call: %.4f ms; 400 calls: frame %.4f ms >= inner stamps %.4f ms; budget %.0f ms, over %d", g1, frame,
                          inner, vk::kMaxSubmitGpuMs, int(ctx.last_submit_over_budget()));
            report("last submit GPU time", r4 == CERE_OK && g1 > 0 && g1 < 50 && frame >= inner && inner > 0 && frame < 1000 &&
                                               !ctx.last_submit_over_budget() && frame > g1,
                   m);
        }
        ctx.free(dt); ctx.free(di); ctx.free(dout);
        std::printf("%-34s INFO  VK_EXT_device_fault enabled %d (diagnostic only: queried after a fault)\n", "driver fault ext",
                    int(ctx.info().device_fault));
    }
    {   // simulated aborted job: the context expects a different nonce than the command buffer writes (the GPU runs
        // normally); submit_wait must fault the context and every later call must refuse without touching the device.
        std::unique_ptr<Context> c5 = open0("completion marker negative");
        if (c5) {
            vk::Buffer b;
            c5->allocate(4096, b);
            std::memset(b.map, 0, 4096);
            const std::vector<int32_t> ids = {0};
            std::memcpy(static_cast<uint8_t *>(b.map) + 2048, ids.data(), 4);
            c5->skew_completion_nonce_for_test();
            cere_status r = c5->begin();
            if (r == CERE_OK) r = vkk1::embedding_gather(*c5, b.addr, 4, 64, b.addr + 2048, 1, b.addr + 1024);
            const cere_status sw = r == CERE_OK ? c5->submit_wait() : r;
            const uint64_t c0 = c5->device_commands();
            const cere_status b2 = c5->begin(), s2 = c5->submit_wait();
            vk::TableRegion tr;
            const cere_status t2 = c5->table_alloc(16, 16, &tr);
            const bool reason = c5->fault_reason().find("completion marker") != std::string::npos &&
                                c5->last_error().find("completion marker") != std::string::npos;
            report("completion marker negative", r == CERE_OK && sw == CERE_ERR_INTERNAL && c5->faulted() && reason && b2 == CERE_ERR_INTERNAL &&
                                                     s2 == CERE_ERR_INTERNAL && t2 == CERE_ERR_INTERNAL && c5->device_commands() == c0 &&
                                                     c5->last_completed_nonce() == 0 && c5->last_submit_gpu_ms() < 0,
                   "skewed nonce -> submit_wait " + st(sw) + ", faulted, later begin/submit/table_alloc " + st(b2) + "/" + st(s2) + "/" +
                       st(t2) + " without device commands; reason \"" + c5->fault_reason() + "\"");
            c5->free(b);
        }
    }

    pipelined_rows(g);
    if (queue_probe_ms > 0) queue_probe(queue_probe_ms, queue_probe_count);

    // --- sticky fault (simulated: the GPU is never faulted) -------------------------------------------------------------
    {
        std::unique_ptr<Context> c4 = open0("sticky fault");
        if (c4) {
            const vk::Pipeline *p = nullptr;
            c4->pipeline(vk::PipelineSpec{"k1_embedding.spv", 40, 0, {}}, &p);
            vk::Buffer keep;
            c4->allocate(4096, keep);
            // faulted while recording: the open command buffer is never submitted
            cere_status b0 = c4->begin();
            c4->inject_fault_for_test("simulated device loss");
            const uint64_t c0 = c4->device_commands();
            const uint64_t push[5] = {};
            const cere_status d1 = c4->dispatch(*p, push, 1);
            const cere_status s1 = c4->submit_wait();
            const cere_status b1 = c4->begin();
            const cere_status d2 = c4->dispatch(*p, push, 1);
            const cere_status br = c4->barrier();
            const cere_status sp = c4->stamp(0);
            const cere_status s2 = c4->submit_wait();
            vk::TableRegion tr;
            const cere_status ta = c4->table_alloc(16, 16, &tr);
            vk::Buffer nb;
            const cere_status al = c4->allocate(4096, nb);
            const vk::Pipeline *p2 = nullptr;
            const cere_status pl = c4->pipeline(vk::PipelineSpec{"k1_rope.spv", 56, 0, {}}, &p2);
            const cere_status kk = vkk1::embedding_gather(*c4, keep.addr, 4, 64, keep.addr, 1, keep.addr);
            const uint64_t touched = c4->device_commands() - c0;
            const bool all_internal = d1 == CERE_ERR_INTERNAL && s1 == CERE_ERR_INTERNAL && b1 == CERE_ERR_INTERNAL && d2 == CERE_ERR_INTERNAL &&
                                      br == CERE_ERR_INTERNAL && sp == CERE_ERR_INTERNAL && s2 == CERE_ERR_INTERNAL && ta == CERE_ERR_INTERNAL &&
                                      al == CERE_ERR_INTERNAL && pl == CERE_ERR_INTERNAL;
            report("sticky fault", b0 == CERE_OK && c4->faulted() && all_internal && touched == 0 && !nb.buf && !p2 &&
                                       c4->fault_reason().find("simulated") != std::string::npos,
                   "after inject: dispatch/submit/begin/barrier/stamp/table_alloc/allocate/pipeline all " + st(CERE_ERR_INTERNAL) +
                       ", kernel call returned " + st(kk) + "; device commands after the fault: " + std::to_string(touched) +
                       "; reason \"" + c4->fault_reason() + "\"");
            c4->free(keep);  // host-side release still works on a faulted context
            report("faulted context teardown", keep.buf == VK_NULL_HANDLE, "free() released; destructor runs next");
        }
    }

    std::printf("vk_launch_test: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
