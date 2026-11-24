// AmdAmd PAR-6 step 2k: K4 PLE gather and embed (executor/cuda/src/kernels/k4_ple.cu, hipified) on gfx1151.
//  gather: out[i] = bf16(fp8 E4M3 row ids[i]) times the BF16 scale; ids outside [0, rows) give zero rows. Compared
//          bit for bit with a host decode (E4M3 is exact in BF16; the product rounded once to BF16).
//  embed:  for t tokens after history [2], token i's window (ctx[i], ctx[i+1], ctx[i+2]) of ctx = history ++ tokens,
//          its 16 n-gram ids (SGLang's hash, the real multipliers) gathered and concatenated into emb [t, 2560];
//          history left unchanged. Heads use small runtime vocab sizes (1,000 ids each) so the table fits in a test;
//          compared bit for bit with the host hash (as k4_hash_test) + host gather, incl. EOS in the history and tokens.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k4_ple.h"

namespace k4 = cere::k4;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 16)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float e4m3(uint8_t c) {  // OCP E4M3 (finite: 0x7f/0xff are NaN)
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}
static const int64_t kMult[3] = {23703573157769LL, 20109073645365LL, 8052911324071LL};
constexpr int64_t kEos = 248044;
static int64_t py_mod(int64_t x, int64_t v) { int64_t r = x % v; return r < 0 ? r + v : r; }
static void ref_hash(const int64_t w[3], const int64_t *vocab, const int64_t *offs, int64_t out[16]) {
    int64_t prev_eos = -1; for (int i = 0; i < 2; ++i) if (w[i] == kEos) prev_eos = i;
    const int64_t pis = 2 - (prev_eos + 1), sh[3] = {w[2], pis >= 1 ? w[1] : kEos, pis >= 2 ? w[0] : kEos};
    for (int ng = 2; ng <= 3; ++ng) {
        int64_t mix = int64_t(uint64_t(sh[0]) * uint64_t(kMult[0]));
        for (int p = 1; p < ng; ++p) mix ^= int64_t(uint64_t(sh[p]) * uint64_t(kMult[p]));
        for (int h = 0; h < 8; ++h) { const int hd = (ng - 2) * 8 + h; out[hd] = py_mod(mix, vocab[hd]) + offs[hd]; }
    }
}

int main() {
    std::mt19937_64 g(31);
    const int64_t rows = 16 * 1000, D = k4::kHeadDim;  // 160 FP8 bytes per row
    std::vector<uint8_t> table(rows * D); for (auto &b : table) { b = uint8_t(g()); if ((b & 0x7f) == 0x7f) b ^= 1; }  // no NaN codes
    const uint16_t scale = f2bf(0.0625f * 3.f);
    auto *dt = dev(table); auto *ds = dev(std::vector<uint16_t>{scale});
    auto row_ref = [&](int64_t id, uint16_t *dst) {
        for (int64_t j = 0; j < D; ++j) dst[j] = (id < 0 || id >= rows) ? 0 : f2bf(bf2f(f2bf(e4m3(table[id * D + j]))) * bf2f(scale));
    };
    {   // gather, incl. out-of-range ids
        std::vector<int64_t> ids = {0, 1, 15999, 7, -1, 16000, 123456789, 8000};
        auto *di = dev(ids); uint16_t *dout; HCK(hipMalloc(&dout, ids.size() * D * 2));
        if (k4::gather(dt, rows, di, int64_t(ids.size()), reinterpret_cast<const __hip_bfloat16 *>(ds), reinterpret_cast<__hip_bfloat16 *>(dout), nullptr) != CERE_OK) { std::printf("gather failed\n"); return 2; }
        auto out = host(dout, ids.size() * D); std::vector<uint16_t> want(ids.size() * D);
        for (size_t i = 0; i < ids.size(); ++i) row_ref(ids[i], &want[i * D]);
        int64_t diff = 0; for (size_t i = 0; i < want.size(); ++i) diff += out[i] != want[i];
        std::printf("%-8s %s  8 ids (3 out of range -> zero rows): %ld of %ld values differ (bit-exact)\n", "gather", diff ? "FAIL" : "PASS", (long)diff, (long)want.size());
        if (diff) ++g_fail;
    }
    {   // embed
        std::vector<int64_t> vocab(16, 1000), offs(16); for (int h = 0; h < 16; ++h) offs[h] = h * 1000;
        std::vector<int64_t> mult(kMult, kMult + 3);
        const int64_t t = 9; std::vector<int64_t> tokens(t); for (auto &x : tokens) x = int64_t(g() % 248320);
        tokens[4] = kEos;  // an EOS inside the step
        std::vector<int64_t> history = {kEos, int64_t(g() % 248320)};  // an EOS in the history
        auto *dm = dev(mult); auto *dv = dev(vocab); auto *doff = dev(offs); auto *dh = dev(history); auto *dtok = dev(tokens);
        k4::Weights w{}; w.table = dt; w.table_rows = rows; w.table_scale = reinterpret_cast<const __hip_bfloat16 *>(ds);
        w.multipliers = dm; w.head_vocab_sizes = dv; w.head_offsets = doff; w.eos_token_id = kEos; w.format = k4::TableFormat::kFp8;
        uint16_t *demb; HCK(hipMalloc(&demb, t * k4::kEmbedDim * 2));
        const cere_status st = k4::embed(w, dh, dtok, t, reinterpret_cast<__hip_bfloat16 *>(demb), nullptr);
        if (st != CERE_OK) { std::printf("embed failed: %d\n", st); return 2; }
        auto emb = host(demb, t * k4::kEmbedDim); auto hist_after = host(dh, 2);
        std::vector<int64_t> ctx = history; ctx.insert(ctx.end(), tokens.begin(), tokens.end());
        int64_t diff = 0; std::vector<uint16_t> row(D);
        for (int64_t i = 0; i < t; ++i) {
            int64_t ids[16]; ref_hash(&ctx[i], vocab.data(), offs.data(), ids);
            for (int h = 0; h < 16; ++h) { row_ref(ids[h], row.data()); for (int64_t j = 0; j < D; ++j) diff += emb[i * k4::kEmbedDim + h * D + j] != row[j]; }
        }
        const bool hist_ok = hist_after == history;
        std::printf("%-8s %s  9 tokens after history (EOS in history and step): %ld of %ld values differ; history unchanged %s\n", "embed",
                    diff == 0 && hist_ok ? "PASS" : "FAIL", (long)diff, (long)(t * k4::kEmbedDim), hist_ok ? "yes" : "NO");
        if (diff || !hist_ok) ++g_fail;
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
