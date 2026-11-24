// AmdAmd PAR-6 step 2j: K4 PLE n-gram hash (executor/cuda/src/kernels/k4_ple.cu, hipified) on gfx1151, bit-exact
// against an independent host port of SGLang's `_hash_contexts` + `_shift_right_ignore_eos` (lovedheart/sglang
// d4689328e python/sglang/srt/models/qwen4_exp.py:603-679) on [n, 3] windows, with Flash-Next's real constants (read
// from its GGUF: 45-bit layer multipliers and the 16 head vocab sizes/offsets). Windows: random ids up to the vocab,
// ids at the top of the vocab (the largest products: 248,319 x 2.37e13 = 5.9e18, still below 2^63, so with valid ids
// the int64 products never wrap; the reference wraps like torch anyway), and EOS in every window position.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k4_ple.h"

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

static const int64_t kMult[3] = {23703573157769LL, 20109073645365LL, 8052911324071LL};
static const int64_t kVocab[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
static const int64_t kOffset[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                                    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
constexpr int64_t kEos = 248044;

static int64_t wrap_mul(int64_t a, int64_t b) { return int64_t(uint64_t(a) * uint64_t(b)); }  // torch long * wraps
static int64_t py_mod(int64_t x, int64_t v) { int64_t r = x % v; return r < 0 ? r + v : r; }   // torch.remainder

// SGLang's hash of one [3]-token context row, at its last position.
static void ref_hash(const int64_t w[3], int64_t out[16]) {
    // _shift_right_ignore_eos for the last index (2): prev_eos = the last EOS among positions 0..1 (else -1);
    // shift n is valid iff 2 - (prev_eos + 1) >= n, else EOS.
    int64_t prev_eos = -1; for (int i = 0; i < 2; ++i) if (w[i] == kEos) prev_eos = i;
    const int64_t pos_in_segment = 2 - (prev_eos + 1);
    const int64_t s0 = w[2], s1 = pos_in_segment >= 1 ? w[1] : kEos, s2 = pos_in_segment >= 2 ? w[0] : kEos;
    const int64_t shifted[3] = {s0, s1, s2};
    for (int ngram = 2; ngram <= 3; ++ngram) {
        int64_t mix = wrap_mul(shifted[0], kMult[0]);
        for (int pos = 1; pos < ngram; ++pos) mix ^= wrap_mul(shifted[pos], kMult[pos]);
        for (int h = 0; h < 8; ++h) { const int head = (ngram - 2) * 8 + h; out[head] = py_mod(mix, kVocab[head]) + kOffset[head]; }
    }
}

int main() {
    std::mt19937_64 g(99); std::vector<int64_t> win;
    auto add = [&](int64_t a, int64_t b, int64_t c) { win.insert(win.end(), {a, b, c}); };
    for (int i = 0; i < 3000; ++i) add(int64_t(g() % 248320), int64_t(g() % 248320), int64_t(g() % 248320));
    for (int i = 0; i < 500; ++i) add(248320 - 1 - int64_t(g() % 64), 248320 - 1 - int64_t(g() % 64), 248320 - 1 - int64_t(g() % 64));
    for (int e = 0; e < 8; ++e) {  // EOS patterns: every subset of the three positions
        for (int i = 0; i < 50; ++i) add(e & 1 ? kEos : int64_t(g() % 248320), e & 2 ? kEos : int64_t(g() % 248320), e & 4 ? kEos : int64_t(g() % 248320));
    }
    const int64_t n = int64_t(win.size()) / 3;
    int64_t *dw, *dm, *dv, *doff, *dids;
    HCK(hipMalloc(&dw, win.size() * 8)); HCK(hipMalloc(&dm, 24)); HCK(hipMalloc(&dv, 128)); HCK(hipMalloc(&doff, 128)); HCK(hipMalloc(&dids, n * 16 * 8));
    HCK(hipMemcpy(dw, win.data(), win.size() * 8, hipMemcpyHostToDevice)); HCK(hipMemcpy(dm, kMult, 24, hipMemcpyHostToDevice));
    HCK(hipMemcpy(dv, kVocab, 128, hipMemcpyHostToDevice)); HCK(hipMemcpy(doff, kOffset, 128, hipMemcpyHostToDevice));
    if (cere::k4::ngram_hash_windows(dw, n, dm, dv, doff, kEos, dids, nullptr) != CERE_OK) { std::printf("launch failed\n"); return 2; }
    HCK(hipDeviceSynchronize()); std::vector<int64_t> ids(n * 16); HCK(hipMemcpy(ids.data(), dids, ids.size() * 8, hipMemcpyDeviceToHost));
    int64_t bad = 0, overflowed = 0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t want[16]; ref_hash(&win[i * 3], want);
        for (int h = 0; h < 16; ++h) bad += ids[i * 16 + h] != want[h];
        overflowed += (__int128)win[i * 3 + 2] * kMult[0] > INT64_MAX;
    }
    std::printf("ngram_hash_windows  %s  %ld windows (500 at the top of the vocab, 400 with EOS patterns): %ld of %ld ids differ\n",
                bad ? "FAIL" : "PASS", (long)n, (long)bad, (long)(n * 16));
    (void)overflowed;
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
