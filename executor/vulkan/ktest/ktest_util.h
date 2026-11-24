// VulkanAmd P1 ktest helpers: BF16 conversions as the HIP ktests define them, buffer upload/readback on the UMA
// mapping, and the GEMV timing method shared by k2v_test / k2mx_test.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "kernels/k2_mgemv_vk.h"
#include "launch/vk_launch.h"

namespace kt {

// VulkanAutotune ktest hooks (vulkan-autotune design note "Kernel-side hooks"; scripts/tune/vulkan_autotune.py run
// drives them). Flags of the ktest binaries only: the library and the server never read them.
//   --tune KEY=VALUE  H4: a host-side knob the ktest passes into its kernel object's Options (each ktest lists the keys
//                     it takes; an unknown key exits 2, so no candidate is ever timed with a knob silently ignored);
//   --identity DIR    H5: every output a contract row or timing shape produces is also written to DIR/<tag>.bin, so
//                     the runner compares a candidate's bytes with the default point's (`identity_default`);
//   --samples         H6: every timed shape prints `TIMING <shape> <mode> n <N> ms <s_1> .. <s_N>` (issue order), and
//                     the ktest times every shape with >= 15 samples.
struct Hooks {
    std::map<std::string, std::string> tune;
    std::string identity_dir;
    bool samples = false;
};
inline Hooks &hooks() { static Hooks h; return h; }
// Consumes argv[i] (and its value) when it is a hook flag; true if consumed.
inline bool hook_arg(int argc, char **argv, int &i) {
    const std::string a = argv[i];
    if (a == "--samples") { hooks().samples = true; return true; }
    if ((a == "--tune" || a == "--identity") && i + 1 < argc) {
        const std::string v = argv[++i];
        if (a == "--identity") { hooks().identity_dir = v; return true; }
        const size_t eq = v.find('=');
        if (eq == std::string::npos || eq == 0) { std::printf("FATAL --tune %s: want KEY=VALUE\n", v.c_str()); std::exit(2); }
        hooks().tune[v.substr(0, eq)] = v.substr(eq + 1);
        return true;
    }
    return false;
}
// Exits 2 when a --tune key is not one this ktest takes.
inline void tune_keys(std::initializer_list<const char *> known) {
    for (const auto &kv : hooks().tune) {
        bool ok = false;
        for (const char *k : known) ok |= kv.first == k;
        if (!ok) { std::printf("FATAL --tune %s: this ktest takes no such knob\n", kv.first.c_str()); std::exit(2); }
    }
}
inline uint32_t tune_u32(const char *key, uint32_t def) {
    const auto it = hooks().tune.find(key);
    return it == hooks().tune.end() ? def : uint32_t(std::strtoul(it->second.c_str(), nullptr, 10));
}
// H5: the bytes of one output under `tag` (file name characters only: [A-Za-z0-9_.-]).
inline void identity(const std::string &tag, const void *p, size_t bytes) {
    if (hooks().identity_dir.empty()) return;
    const std::string path = hooks().identity_dir + "/" + tag + ".bin";
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr || std::fwrite(p, 1, bytes, f) != bytes || std::fclose(f) != 0) {
        std::printf("FATAL identity write %s\n", path.c_str());
        std::exit(2);
    }
}
template <class T> void identity(const std::string &tag, const std::vector<T> &v) { identity(tag, v.data(), v.size() * sizeof(T)); }
// H6: the samples of one timed shape, in issue order (mode "warm" | "cold").
inline void samples(const std::string &shape, const char *mode, const std::vector<double> &ms) {
    if (!hooks().samples) return;
    std::printf("TIMING %s %s n %zu ms", shape.c_str(), mode, ms.size());
    for (double v : ms) std::printf(" %.5f", v);
    std::printf("\n");
    std::fflush(stdout);
}
// The repetitions of a timed shape: the ktest's own count, at least 15 under --samples.
inline int reps(int own) { return hooks().samples ? std::max(own, 15) : own; }

inline float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

inline cere::vk::Buffer make(cere::vk::Context &ctx, uint64_t bytes) {
    cere::vk::Buffer b;
    if (!ctx.alloc(bytes + 256, b)) {  // +256: tail slack as the HIP tests allocate
        std::printf("FATAL allocation of %llu bytes: %s\n", (unsigned long long)(bytes + 256), ctx.last_error().c_str());
        std::exit(2);
    }
    std::memset(b.map, 0, bytes + 256);
    return b;
}
template <class T> cere::vk::Buffer upload(cere::vk::Context &ctx, const std::vector<T> &h) {
    cere::vk::Buffer b = make(ctx, h.size() * sizeof(T));
    std::memcpy(b.map, h.data(), h.size() * sizeof(T));
    return b;
}
template <class T> std::vector<T> host(const cere::vk::Buffer &b, size_t n, size_t off_bytes = 0) {
    std::vector<T> h(n);
    std::memcpy(h.data(), static_cast<const uint8_t *>(b.map) + off_bytes, n * sizeof(T));
    return h;
}

inline void must(cere_status s, const char *what) {
    if (s != CERE_OK) { std::printf("FATAL %s returned %d\n", what, int(s)); std::exit(2); }
}

// One call, recorded, submitted and waited for.
inline void gemv_sync(cere::vk::Context &ctx, uint64_t x, uint64_t w, uint64_t s, uint64_t y, uint32_t m, uint32_t n,
                      uint32_t k, bool f32, const cere::vkk2::Mapping *force = nullptr) {
    ctx.begin();
    must(cere::vkk2::gemv(ctx, x, w, s, y, m, n, k, f32, force), "gemv");
    ctx.barrier();
    ctx.submit_wait();
}

struct Timing { double ms_median, ms_min, ms_max; std::vector<double> ms; };  // ms: every rep, issue order

// The settle pre-roll, ms (env KT_SETTLE_MS, default 20; 0 = the P1 method without it). On gfx1151 with
// power_dpm_force_performance_level auto, a stream of small serialized calls (short-K m 16: ~0.05 ms each, a barrier
// between them) can be timed with the memory clocks in a low DPM state for whole windows (0.05-0.18 ms run to run;
// steady at level high, and inside the model, where large streams run between them). Each window is therefore
// preceded, in the same command buffer, by DRAM-bound sweeps: BF16 m 1 GEMVs at k 2,560 over the weight copies' whole
// span (copies x copy_bytes, as rows), repeated until ~settle ms at 200 GB/s (<= 64 sweeps), into a scratch output.
inline double settle_ms() {
    const char *e = std::getenv("KT_SETTLE_MS");
    return e != nullptr ? std::atof(e) : 20.0;
}

// GPU-timestamp time of one GEMV call: max(3, copies) untimed warm-up calls (every weight copy touched by the GPU once,
// so first-touch / page-table effects of the CPU-filled buffers are not timed), the settle sweeps (above), then `reps`
// calls each between its own pair of timestamps with a barrier after it (calls serialized, as a HIP stream
// serializes them); median / min / max over the reps. Call i reads weight copy i % copies (copies laid out
// `copy_bytes` apart; copies > 1 defeats the GPU caches for weights smaller than the MALL; scales likewise
// `scale_bytes` apart). `per` > 1 times `per` consecutive calls (a barrier between them) between one pair of
// timestamps and reports the per-call mean of that window. `x` holds >= 2,560 BF16 values. per 1 with
// KT_SETTLE_MS=0 is the original P1 method.
inline Timing time_gemv(cere::vk::Context &ctx, uint64_t x, uint64_t w, uint64_t copy_bytes, uint64_t s,
                        uint64_t scale_bytes, uint32_t copies, uint64_t y, uint32_t m, uint32_t n, uint32_t k,
                        const cere::vkk2::Mapping *force = nullptr, int reps = 15, uint32_t per = 1) {
    if (2 * reps > int(cere::vk::Context::kStamps) || per == 0) std::exit(2);
    uint32_t c = 0;
    auto wa = [&](uint32_t i) { return w + uint64_t(i % copies) * copy_bytes; };
    auto sa = [&](uint32_t i) { return s ? s + uint64_t(i % copies) * scale_bytes : 0; };
    const double settle = settle_ms();
    const uint64_t span = uint64_t(copies) * copy_bytes;
    const uint32_t sweep_n = uint32_t(span / (2560 * 2));
    const uint32_t sweeps = settle <= 0 || sweep_n == 0 ? 0 : uint32_t(std::min(64.0, std::ceil(settle * 2e8 / double(span))));
    cere::vk::Buffer scratch;
    if (sweeps > 0 && !ctx.alloc(uint64_t(sweep_n) * 8, scratch)) std::exit(2);
    ctx.begin();
    for (uint32_t i = 0; i < std::max<uint32_t>(3, copies); ++i, ++c) {
        must(cere::vkk2::gemv(ctx, x, wa(c), sa(c), y, m, n, k, false, force), "gemv");
        ctx.barrier();
    }
    for (uint32_t i = 0; i < sweeps; ++i) {
        must(cere::vkk2::gemv(ctx, x, w, 0, scratch.addr, 1, sweep_n, 2560, false), "settle gemv");
        ctx.barrier();
    }
    for (int i = 0; i < reps; ++i) {
        ctx.stamp(2 * i);
        for (uint32_t j = 0; j < per; ++j, ++c) {
            if (j > 0) ctx.barrier();
            must(cere::vkk2::gemv(ctx, x, wa(c), sa(c), y, m, n, k, false, force), "gemv");
        }
        ctx.stamp(2 * i + 1);
        ctx.barrier();
    }
    ctx.submit_wait();
    if (sweeps > 0) ctx.free(scratch);
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1) / per);
    if (std::getenv("KT_SAMPLES") != nullptr) {  // diagnostic: every rep in issue order, before the sort
        std::printf("samples ms:");
        for (double v : t) std::printf(" %.3f", v);
        std::printf("\n");
    }
    const std::vector<double> issued = t;
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t.front(), t.back(), issued};
}

}  // namespace kt
