// Inkling 2 Mini S2 (inkling design note 3.3, G1): host test of the attention CPU reference
// (executor/ref/inkling/attn_ref.h). Checks the reference itself, before any kernel is compared with it: the post ops
// (tau, the L2 clips hit and not hit, the value RMSNorm), the relative bias range, the attention of one row against a
// naive double-precision softmax (window, bias, sink, gate), the window and bias edges, and that the output of a row
// depends only on the key bits at its positions: keys from the ring vs from fresh rows, a paged slot map vs the
// identity (bit-identical).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../ref/inkling/attn_ref.h"
#include "check.h"

namespace {

namespace R = ref::attn;

uint32_t rng_state = 0x2468ace1u;
uint32_t next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
float uniform(float lo, float hi) { return lo + (hi - lo) * float(next_u32() >> 8) * (1.0f / 16777216.0f); }
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi) {
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = ref::encode_bf16(uniform(lo, hi));
    return v;
}
float f(uint16_t b) { return ref::decode_bf16(b); }

void test_tau() {
    CHECK(R::tau(0) == 1.0f);
    CHECK(R::tau(8190) == 1.0f);
    CHECK(R::tau(8191) == 1.0f);  // (8191 + 1) / 8192 = 1, log 1 = 0
    const double want = 1.0 + 0.1 * std::log(16384.0 / 8192.0);
    CHECK(std::fabs(double(R::tau(16383)) - want) < 1e-6);
    const double want_1m = 1.0 + 0.1 * std::log(1048576.0 / 8192.0);
    CHECK(std::fabs(double(R::tau(1048575)) - want_1m) < 1e-6);
}

double norm(const uint16_t *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += double(f(x[i])) * f(x[i]);
    return std::sqrt(s);
}

void test_post() {
    // k post: a short vector (norm < 22.6) is unchanged; a long one is clipped to norm 22.627 (BF16 rounding).
    std::vector<uint16_t> small = random_bf16(R::kKvRow, -0.5f, 0.5f), big = random_bf16(R::kKvRow, -40.0f, 40.0f);
    std::vector<uint16_t> out(R::kKvRow);
    R::k_post_row(small.data(), out.data());
    CHECK(out == small);
    R::k_post_row(big.data(), out.data());
    for (int g = 0; g < R::kKvHeads; ++g) CHECK(std::fabs(norm(&out[g * 128], 128) - 22.627416997969522) < 0.1);
    // q post: scale applied before the clip; full layers multiply by tau after the BF16 rounding.
    std::vector<uint16_t> q = random_bf16(R::kQRow, -3.0f, 3.0f), scale = random_bf16(R::kQRow, 0.5f, 2.0f);
    std::vector<uint16_t> qs(R::kQRow), qf(R::kQRow);
    R::q_post_row(R::kSliding, q.data(), scale.data(), 20000, qs.data());
    R::q_post_row(R::kFull, q.data(), scale.data(), 20000, qf.data());
    const float t = R::tau(20000);
    for (int i = 0; i < R::kQRow; ++i) CHECK(qf[i] == ref::encode_bf16(ref::mul(f(qs[i]), t)));
    for (int h = 0; h < R::kHeads; ++h) CHECK(norm(&qs[h * 128], 128) <= 22.627416997969522 * 1.01);
    // rel post: threshold 8.
    std::vector<uint16_t> r = random_bf16(R::kRelRow, -6.0f, 6.0f), rs = random_bf16(R::kRelRow, 0.5f, 1.5f);
    std::vector<uint16_t> ro(R::kRelRow);
    R::rel_post_row(R::kSliding, r.data(), rs.data(), 5, ro.data());
    for (int h = 0; h < R::kHeads; ++h) CHECK(norm(&ro[h * 16], 16) <= 8.0 * 1.01);
    // v post: per-head RMS 1.
    std::vector<uint16_t> v = random_bf16(R::kKvRow, -7.0f, 7.0f), vo(R::kKvRow);
    R::v_post_row(v.data(), vo.data());
    for (int g = 0; g < R::kKvHeads; ++g) CHECK(std::fabs(norm(&vo[g * 128], 128) / std::sqrt(128.0) - 1.0) < 0.01);
    // All zero: clip leaves zeros, the RMSNorm gives zeros (no NaN).
    std::vector<uint16_t> zero(R::kKvRow, 0);
    R::k_post_row(zero.data(), out.data());
    CHECK(out == zero);
    R::v_post_row(zero.data(), vo.data());
    CHECK(vo == zero);
}

void test_rel_bias() {
    std::vector<uint16_t> r = random_bf16(16, -2.0f, 2.0f), proj = random_bf16(16 * 1024, -1.0f, 1.0f);
    CHECK(R::rel_bias(r.data(), proj.data(), -1) == 0.0f);
    CHECK(R::rel_bias(r.data(), proj.data(), 1024) == 0.0f);
    for (int d : {0, 1, 511, 1023}) {
        double s = 0.0;
        for (int i = 0; i < 16; ++i) s += double(f(r[i])) * f(proj[i * 1024 + d]);
        CHECK(std::fabs(R::rel_bias(r.data(), proj.data(), d) - s) <= std::fabs(s) * 0.01 + 1e-3);
    }
}

// One attention problem: a session with `total` positions of K/V (all fresh rows at pos0 0), queries at rows.
struct Problem {
    int kind;
    int64_t total;
    std::vector<uint16_t> q, r, k, v, sink, gate, proj;
    explicit Problem(int kind_, int64_t total_) : kind(kind_), total(total_) {
        q = random_bf16(size_t(total) * R::kQRow, -1.0f, 1.0f);
        r = random_bf16(size_t(total) * R::kRelRow, -2.0f, 2.0f);
        k = random_bf16(size_t(total) * R::kKvRow, -1.0f, 1.0f);
        v = random_bf16(size_t(total) * R::kKvRow, -1.0f, 1.0f);
        sink = random_bf16(size_t(total) * R::kHeads, -2.0f, 4.0f);
        gate = random_bf16(size_t(total) * R::kQRow, -3.0f, 3.0f);
        proj = random_bf16(16 * 1024, -1.0f, 1.0f);
    }
    R::Keys fresh() const {
        return R::Keys{kind, 0, 0, k.data(), v.data(), R::kKvRow, R::kKvRow, nullptr, nullptr, nullptr};
    }
    std::vector<uint16_t> row(const R::Keys &keys, int64_t p, int64_t q_row) const {
        std::vector<uint16_t> out(R::kQRow);
        R::attention_row(keys, &q[q_row * R::kQRow], &r[q_row * R::kRelRow], proj.data(), &sink[q_row * R::kHeads],
                         &gate[q_row * R::kQRow], p, out.data());
        return out;
    }
};

// Naive double attention of row p (REF semantics, no intermediate rounding).
std::vector<double> naive(const Problem &pb, int64_t p) {
    std::vector<double> out(R::kQRow);
    const int64_t lo = pb.kind == R::kSliding ? std::max<int64_t>(0, p - 1023) : 0;
    for (int h = 0; h < R::kHeads; ++h) {
        const int g = h / 3;
        std::vector<double> s;
        for (int64_t j = lo; j <= p; ++j) {
            double dot = 0.0;
            for (int i = 0; i < 128; ++i) dot += double(f(pb.q[p * 3072 + h * 128 + i])) * f(pb.k[j * 1024 + g * 128 + i]);
            double bias = 0.0;  // REF: the rel_logits_proj output is a BF16 tensor (one rounding of the dot)
            if (p - j < 1024) {
                for (int i = 0; i < 16; ++i) bias += double(f(pb.r[p * 384 + h * 16 + i])) * f(pb.proj[i * 1024 + (p - j)]);
                bias = f(ref::encode_bf16(float(bias)));
            }
            s.push_back(dot / std::sqrt(128.0) + bias);
        }
        const double z = f(pb.sink[p * 24 + h]);
        double m = z;
        for (double x : s) m = std::max(m, x);
        double l = std::exp(z - m);
        for (double x : s) l += std::exp(x - m);
        for (int d = 0; d < 128; ++d) {
            double o = 0.0;
            for (size_t x = 0; x < s.size(); ++x) o += std::exp(s[x] - m) / l * f(pb.v[(lo + x) * 1024 + g * 128 + d]);
            const double gt = f(pb.gate[p * 3072 + h * 128 + d]);
            out[h * 128 + d] = o / (1.0 + std::exp(-gt));
        }
    }
    return out;
}

double rel_l2(const std::vector<uint16_t> &a, const std::vector<double> &b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        num += (f(a[i]) - b[i]) * (f(a[i]) - b[i]);
        den += b[i] * b[i];
    }
    return std::sqrt(num / den);
}

void test_vs_naive() {
    for (int kind : {int(R::kSliding), int(R::kFull)}) {
        Problem pb(kind, 1400);
        for (int64_t p : {0, 63, 64, 300, 1023, 1024, 1399}) {
            const double e = rel_l2(pb.row(pb.fresh(), p, p), naive(pb, p));
            if (!(e < 5e-3)) std::fprintf(stderr, "kind %d p %lld rel-L2 %g\n", kind, (long long)p, e);
            CHECK(e < 5e-3);
        }
    }
}

// The row at p reads exactly the keys [lo, p]: a key at p - 1024 (sliding) or > p is never read; p - 1023 is.
void test_edges() {
    Problem pb(R::kSliding, 1400);
    const int64_t p = 1300;
    const std::vector<uint16_t> base = pb.row(pb.fresh(), p, p);
    Problem changed = pb;
    for (int i = 0; i < 1024; ++i) {
        changed.k[(p - 1024) * 1024 + i] = ref::encode_bf16(50.0f);
        changed.k[(p + 1) * 1024 + i] = ref::encode_bf16(50.0f);
    }
    CHECK(changed.row(changed.fresh(), p, p) == base);
    changed.k[(p - 1023) * 1024] = ref::encode_bf16(50.0f);
    CHECK(changed.row(changed.fresh(), p, p) != base);
    // Full layers: the key at distance 1024 has no bias but is attended.
    Problem full(R::kFull, 1400);
    const std::vector<uint16_t> fb = full.row(full.fresh(), p, p);
    Problem fc = full;
    for (int i = 0; i < 1024; ++i) fc.k[(p - 1100) * 1024 + i] = ref::encode_bf16(3.0f);
    CHECK(fc.row(fc.fresh(), p, p) != fb);
    // The proj column at distance 1024 does not exist: changing column 1023 changes the row, a (fictional) shift
    // of every key by one distance changes it too.
    Problem pc = full;
    for (int i = 0; i < 16; ++i) pc.proj[i * 1024 + 1023] = ref::encode_bf16(7.0f);
    CHECK(pc.row(pc.fresh(), p, p) != fb);
}

// The same rows computed with keys < pos0 from a ring (sliding) or a paged pool (full) equal the all-fresh result.
void test_key_sources() {
    {
        Problem pb(R::kSliding, 1400);
        const int64_t pos0 = 1100;
        std::vector<uint16_t> kr(1024 * 1024), vr(1024 * 1024);
        for (int64_t j = 0; j < pos0; ++j)
            for (int i = 0; i < 1024; ++i) {
                kr[(j % 1024) * 1024 + i] = pb.k[j * 1024 + i];
                vr[(j % 1024) * 1024 + i] = pb.v[j * 1024 + i];
            }
        const R::Keys ring{R::kSliding, 0, pos0, &pb.k[pos0 * 1024], &pb.v[pos0 * 1024], 1024, 1024,
                           kr.data(), vr.data(), nullptr};
        for (int64_t p : {int64_t(1100), int64_t(1101), int64_t(1399)})
            CHECK(pb.row(ring, p, p) == pb.row(pb.fresh(), p, p));
    }
    {
        Problem pb(R::kFull, 1400);
        const int64_t pos0 = 1100, pages = (pos0 + 63) / 64;
        std::vector<int64_t> slots(pos0);
        for (int64_t j = 0; j < pos0; ++j) slots[j] = ((pages - 1 - j / 64) * 64) + j % 64;  // reversed pages
        std::vector<uint16_t> kp(pages * 64 * 1024), vp(pages * 64 * 1024);
        for (int64_t j = 0; j < pos0; ++j)
            for (int i = 0; i < 1024; ++i) {
                kp[slots[j] * 1024 + i] = pb.k[j * 1024 + i];
                vp[slots[j] * 1024 + i] = pb.v[j * 1024 + i];
            }
        const R::Keys pool{R::kFull, 0, pos0, &pb.k[pos0 * 1024], &pb.v[pos0 * 1024], 1024, 1024,
                           kp.data(), vp.data(), slots.data()};
        for (int64_t p : {int64_t(1100), int64_t(1399)}) CHECK(pb.row(pool, p, p) == pb.row(pb.fresh(), p, p));
    }
}

}  // namespace

int main() {
    test_tau();
    test_post();
    test_rel_bias();
    test_vs_naive();
    test_edges();
    test_key_sources();
    return cere::test::finish("test_inkling_attn_ref");
}
