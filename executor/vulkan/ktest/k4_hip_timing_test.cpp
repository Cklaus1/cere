// VulkanAmd P2: same-session HIP reference timing of the hipified K4 PLE (gen_hip.py's kernels/k4_ple.hip, K2 through the
// HIP adapter + hipBLASLt) at the shapes k4_ple_test's "time" / "large" sections time on Vulkan, with the same method:
// warm-up calls, then "call" = 15 calls each between its own pair of hipEvents (median, min, max) and "stream" = samples
// of back-to-back calls / burst (median); every call reads distinct random ids / tokens where the Vulkan side does. The
// tables are mapped pinned host memory (hipHostMalloc, as the HIP executor places the PLE table). HIP is the reference
// here, not part of the Vulkan executor; built by build.d/k4_ple.sh only when hipcc and a hipified tree exist. Not a
// contract test. Usage: k4_hip_timing [time] [large <GB>].
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "kernels/k4_nvfp4.h"
#include "kernels/k4_ple.h"

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using bf16 = __hip_bfloat16;
namespace k4 = cere::k4;
namespace n4 = cere::k4::nvfp4;

static const int64_t kMult[3] = {23703573157769LL, 20109073645365LL, 8052911324071LL};
static const int64_t kVocab[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
static const int64_t kOffset[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                                    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
constexpr int64_t kEos = 248044, H = k4::kHidden, C = k4::kChannels, E = k4::kEmbedDim;

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static T *pinned(size_t n) { void *p; HCK(hipHostMalloc(&p, n * sizeof(T) + 256, hipHostMallocMapped)); return static_cast<T *>(p); }
template <class T> static T *alias(T *h) { void *d; HCK(hipHostGetDevicePointer(&d, h, 0)); return static_cast<T *>(d); }

static int calls_of(int warm = 20, int reps = 15, int samples = 5, int burst = 20) { return warm + reps + samples * burst; }
static void time_it(const std::string &name, const std::function<void(int)> &call, const std::string &extra_fmt = "", double rows = 0, double bytes_per_row = 0,
                    int warm = 20, int reps = 15, int samples = 5, int burst = 20) {
    int c = 0;
    for (int i = 0; i < warm; ++i) call(c++);
    std::vector<hipEvent_t> ev(2 * std::max(reps, samples));
    for (auto &e : ev) HCK(hipEventCreate(&e));
    for (int i = 0; i < reps; ++i) { HCK(hipEventRecord(ev[2 * i], nullptr)); call(c++); HCK(hipEventRecord(ev[2 * i + 1], nullptr)); }
    HCK(hipDeviceSynchronize());
    std::vector<double> t, u;
    for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
    for (int s = 0; s < samples; ++s) {
        HCK(hipEventRecord(ev[2 * s], nullptr));
        for (int i = 0; i < burst; ++i) call(c++);
        HCK(hipEventRecord(ev[2 * s + 1], nullptr));
    }
    HCK(hipDeviceSynchronize());
    for (int s = 0; s < samples; ++s) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * s], ev[2 * s + 1])); u.push_back(ms / burst); }
    for (auto &e : ev) HCK(hipEventDestroy(e));
    std::sort(t.begin(), t.end()); std::sort(u.begin(), u.end());
    std::printf("    %-40s call %9.2f us (min %.2f, max %.2f)  stream %9.2f us/call", name.c_str(), t[t.size() / 2] * 1e3, t.front() * 1e3,
                t.back() * 1e3, u[u.size() / 2] * 1e3);
    if (rows > 0) {
        const double s = t[t.size() / 2] * 1e-3;
        std::printf("  %.1f Mrows/s", rows / s / 1e6);
        if (bytes_per_row > 0) std::printf(", %.1f GB/s of row+scale bytes", rows * bytes_per_row / s / 1e9);
    }
    std::printf("%s\n", extra_fmt.c_str());
}

static void time_section() {
    std::printf("timing (hipEvents; call = one call between its own events, median of 15 after 20 warm-up; stream = 20 back-to-back / 20, median of 5):\n");
    std::mt19937_64 g(77);
    const int ncalls = calls_of();
    {
        std::vector<int64_t> win(size_t(512) * 3 * ncalls);
        for (auto &x : win) x = int64_t(g() % 248320);
        auto *dw = dev(win); auto *dm = dev(std::vector<int64_t>(kMult, kMult + 3)); auto *dv = dev(std::vector<int64_t>(kVocab, kVocab + 16));
        auto *doff = dev(std::vector<int64_t>(kOffset, kOffset + 16));
        int64_t *dids; HCK(hipMalloc(&dids, size_t(512) * 16 * 8));
        for (int n : {1, 512})
            time_it("hash " + std::to_string(n) + " windows", [&](int i) {
                if (k4::ngram_hash_windows(dw + size_t(i) * n * 3, n, dm, dv, doff, kEos, dids, nullptr) != CERE_OK) std::exit(2);
            });
        HCK(hipFree(dw)); HCK(hipFree(dm)); HCK(hipFree(dv)); HCK(hipFree(doff)); HCK(hipFree(dids));
    }
    const int64_t big = 2000000;
    for (int nv = 0; nv < 2; ++nv) {
        const size_t rb = nv ? n4::kRowBytes : 160;
        uint8_t *ht = pinned<uint8_t>(size_t(big) * rb), *hs = nv ? pinned<uint8_t>(size_t(big) * n4::kRowScales) : nullptr;
        for (int64_t r = 0; r < big; ++r) {
            uint64_t v[20];
            for (auto &x : v) x = g();
            std::memcpy(ht + size_t(r) * rb, v, rb);
            if (nv) std::memcpy(hs + size_t(r) * n4::kRowScales, v + 10, n4::kRowScales);
        }
        float *hg = pinned<float>(1); *hg = 0.0123f;
        auto *dsc = dev(std::vector<uint16_t>{f2bf(0.5f)});
        std::vector<int64_t> ids(size_t(8192) * ncalls);
        for (auto &x : ids) x = int64_t(g() % uint64_t(big));
        auto *di = dev(ids);
        uint16_t *dout; HCK(hipMalloc(&dout, size_t(8192) * 320));
        for (int n : {16, 8192})
            time_it(std::string("gather ") + (nv ? "nvfp4 " : "fp8 ") + std::to_string(n) + " ids (" + (n == 16 ? "1 token" : "512 tokens") + ")", [&](int i) {
                const int64_t *ia = di + size_t(i) * n;
                const cere_status s = nv ? k4::gather_nvfp4(alias(ht), alias(hs), alias(hg), big, ia, n, reinterpret_cast<bf16 *>(dout), nullptr)
                                         : k4::gather(alias(ht), big, ia, n, reinterpret_cast<const bf16 *>(dsc), reinterpret_cast<bf16 *>(dout), nullptr);
                if (s != CERE_OK) std::exit(2);
            }, "", n);
        if (nv) {
            std::vector<int64_t> vocab(16, 125000), offs(16), mult(kMult, kMult + 3);
            for (int h = 0; h < 16; ++h) offs[h] = h * 125000;
            std::vector<int64_t> tokens(size_t(512) * ncalls);
            for (auto &x : tokens) x = int64_t(g() % 248320);
            auto *dm = dev(mult); auto *dv = dev(vocab); auto *doff = dev(offs); auto *dh = dev(std::vector<int64_t>{kEos, kEos}); auto *dtok = dev(tokens);
            k4::Weights w{};
            w.table = alias(ht); w.table_rows = big; w.format = k4::TableFormat::kNvfp4; w.block_scales = alias(hs); w.global_scale = alias(hg);
            w.multipliers = dm; w.head_vocab_sizes = dv; w.head_offsets = doff; w.eos_token_id = kEos;
            uint16_t *demb; HCK(hipMalloc(&demb, size_t(512) * E * 2));
            for (int t : {1, 512})
                time_it("embed nvfp4 t " + std::to_string(t), [&](int i) {
                    if (k4::embed(w, dh, dtok + size_t(i) * t, t, reinterpret_cast<bf16 *>(demb), nullptr) != CERE_OK) std::exit(2);
                });
            HCK(hipFree(dm)); HCK(hipFree(dv)); HCK(hipFree(doff)); HCK(hipFree(dh)); HCK(hipFree(dtok)); HCK(hipFree(demb));
        }
        HCK(hipHostFree(ht)); if (hs) HCK(hipHostFree(hs)); HCK(hipHostFree(hg)); HCK(hipFree(dsc)); HCK(hipFree(di)); HCK(hipFree(dout));
    }
    {   // forward: the k4_forward_test weights (mt19937_64 g(41)), FP8 table on the device as that test places it
        std::mt19937_64 gf(41); std::normal_distribution<float> nd(0.f, 1.f);
        auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(gf) * s); return v; };
        const int64_t rows = 16 * 1000; std::vector<uint8_t> table(rows * k4::kHeadDim);
        for (auto &b : table) { b = uint8_t(gf() % 0x70); if (gf() & 1) b |= 0x80; }
        std::vector<int64_t> vocab(16, 1000), offs(16); for (int h = 0; h < 16; ++h) offs[h] = h * 1000;
        std::vector<int64_t> mult(kMult, kMult + 3);
        auto key_proj = rnd(size_t(C) * E, 0.02f), value_proj = rnd(size_t(H) * E, 0.02f);
        auto norm_key = rnd(C, 0.1f), norm_query = rnd(C, 0.1f), norm_conv = rnd(C, 0.1f), conv_w = rnd(size_t(C) * 4, 0.3f);
        k4::Weights w{}; w.table = dev(table); w.table_rows = rows; w.table_scale = reinterpret_cast<const bf16 *>(dev(std::vector<uint16_t>{f2bf(1.0f)}));
        w.multipliers = dev(mult); w.head_vocab_sizes = dev(vocab); w.head_offsets = dev(offs);
        w.key_proj = reinterpret_cast<const bf16 *>(dev(key_proj)); w.value_proj = reinterpret_cast<const bf16 *>(dev(value_proj));
        w.norm_key = reinterpret_cast<const bf16 *>(dev(norm_key)); w.norm_query = reinterpret_cast<const bf16 *>(dev(norm_query));
        w.norm_conv = reinterpret_cast<const bf16 *>(dev(norm_conv)); w.conv_weight = reinterpret_cast<const bf16 *>(dev(conv_w));
        w.eos_token_id = kEos; w.eps = 1e-6f; w.format = k4::TableFormat::kFp8;
        const int64_t TM = 512;
        hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) std::exit(2);
        cere::kernels::GemmPlan pk, pv;
        if (pk.init(lt, C, E, TM, cere::kernels::GemmOut::kBF16) != CERE_OK || pv.init(lt, H, E, TM, cere::kernels::GemmOut::kBF16) != CERE_OK) { std::printf("plan init failed\n"); std::exit(2); }
        k4::Plans plans{&pk, &pv, 0};
        std::vector<int64_t> tokens(TM); for (auto &x : tokens) x = int64_t(gf() % 248320);
        std::vector<uint16_t> q(size_t(TM) * C); for (auto &x : q) x = f2bf(nd(gf));
        auto *dtok = dev(tokens); auto *dq = dev(q); auto *dh = dev(std::vector<int64_t>{kEos, kEos}); auto *dc = dev(std::vector<uint16_t>(size_t(C) * 9, 0));
        uint16_t *dout; HCK(hipMalloc(&dout, size_t(TM) * C * 2));
        const size_t ws = k4::workspace_bytes(TM); void *dws; HCK(hipMalloc(&dws, ws + 256));
        for (int t : {1, 16, 512}) {
            const bool heavy = t == 512;
            k4::State st{dh, reinterpret_cast<bf16 *>(dc)};
            time_it("forward t " + std::to_string(t) + " (add_query)", [&](int) {
                if (k4::forward(w, plans, st, dtok, t, reinterpret_cast<const bf16 *>(dq), true, reinterpret_cast<bf16 *>(dout), dws, ws, nullptr) != CERE_OK) std::exit(2);
            }, "", 0, 0, heavy ? 3 : 20, 15, heavy ? 3 : 5, heavy ? 5 : 20);
        }
        {   // forward_rows: one 4-row verify chain with 3 saves, on the rows plans (init_rows: per-row m = 1 calls on HIP)
            cere::kernels::GemmPlan rk, rv;
            if (rk.init_rows(lt, pk, k4::kMaxRows) != CERE_OK || rv.init_rows(lt, pv, k4::kMaxRows) != CERE_OK) { std::printf("rows plan init failed\n"); std::exit(2); }
            k4::Plans rplans{&rk, &rv, 0};
            k4::RowsCall call;
            call.sequences = 1; call.states[0] = {dh, reinterpret_cast<bf16 *>(dc)}; call.first[0] = 0; call.count[0] = 4;
            for (int r = 0; r < 3; ++r) {
                int64_t *h; uint16_t *c; HCK(hipMalloc(&h, 16)); HCK(hipMalloc(&c, size_t(C) * 9 * 2));
                call.saves[r] = {h, reinterpret_cast<bf16 *>(c)};
            }
            time_it("forward_rows 1 chain x 4 (3 saves)", [&](int) {
                if (k4::forward_rows(w, rplans, call, dtok, 4, reinterpret_cast<const bf16 *>(dq), true, reinterpret_cast<bf16 *>(dout), dws, ws, nullptr) != CERE_OK) std::exit(2);
            });
        }
    }
}

static uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static void large_section(double gb) {
    const int64_t rows = int64_t(gb * 1e9 / 90);
    std::printf("large table: NVFP4, %lld rows (%.2f GB of rows + scales), mapped pinned host (hipHostMalloc)\n", (long long)rows, rows * 90.0 / 1e9);
    const auto t0 = std::chrono::steady_clock::now();
    uint8_t *ht = pinned<uint8_t>(size_t(rows) * n4::kRowBytes), *hs = pinned<uint8_t>(size_t(rows) * n4::kRowScales);
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int k = 0; k < 16; ++k)
        th.emplace_back([&, k] {
            for (int64_t r = rows * k / 16; r < rows * (k + 1) / 16; ++r) {
                uint64_t w[12];
                for (int j = 0; j < 12; ++j) w[j] = mix64(uint64_t(r) * 16 + j);
                std::memcpy(ht + size_t(r) * n4::kRowBytes, w, n4::kRowBytes);
                std::memcpy(hs + size_t(r) * n4::kRowScales, w + 10, n4::kRowScales);
            }
        });
    for (auto &x : th) x.join();
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("    allocate %.2f s, pattern fill %.2f s (16 threads)\n", std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
    float *hg = pinned<float>(1); *hg = 0.0123f;
    std::mt19937_64 g(23);
    const int ncalls = calls_of();
    for (int nrows : {16, 8192, 1 << 20}) {
        const bool huge = nrows == (1 << 20);
        std::vector<int64_t> tid(size_t(nrows) * (huge ? 1 : ncalls));
        for (auto &x : tid) x = int64_t(g() % uint64_t(rows));
        auto *di = dev(tid);
        uint16_t *dout; HCK(hipMalloc(&dout, size_t(nrows) * 320));
        time_it("large gather nvfp4 " + std::to_string(nrows) + " ids", [&](int i) {
            if (k4::gather_nvfp4(alias(ht), alias(hs), alias(hg), rows, di + (huge ? 0 : size_t(i) * nrows), nrows, reinterpret_cast<bf16 *>(dout), nullptr) != CERE_OK) std::exit(2);
        }, "", nrows, 90, huge ? 2 : 20, 15, huge ? 3 : 5, huge ? 5 : 20);
        HCK(hipFree(di)); HCK(hipFree(dout));
    }
    HCK(hipHostFree(ht)); HCK(hipHostFree(hs)); HCK(hipHostFree(hg));
}

int main(int argc, char **argv) {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0));
    std::printf("HIP device: %s (%s)\n", prop.name, prop.gcnArchName);
    std::vector<std::string> a(argv + 1, argv + argc);
    if (a.empty()) a = {"time"};
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == "time") time_section();
        else if (a[i] == "large" && i + 1 < a.size()) large_section(std::atof(a[++i].c_str()));
        else { std::printf("unknown section %s\n", a[i].c_str()); return 2; }
    }
    return 0;
}
