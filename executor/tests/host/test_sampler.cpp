// m27 design note C2: the CPU sampler reference (host/sampler_ref) against hand-computed answers and, when the C3
// oracle fixture tests/fixtures/native/sampler_cases.json exists, against SGLang's sampler pieces. Once the fixture
// is committed (listed in REPO_MANIFEST.json), its absence or a schema mismatch fails.
#include "check.h"
#include "host/sampler_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace sr = cere::sampler_ref;

constexpr double kAlpha = 1e-4;
constexpr uint32_t kDraws = 20000;

#define CHECK_NEAR(actual, expected, tol)                                                                             \
    do {                                                                                                               \
        const double cere_a_ = (actual), cere_e_ = (expected);                                                         \
        if (!(std::fabs(cere_a_ - cere_e_) <= (tol))) {                                                                \
            std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s = %.17g, expected %.17g (tol %g)\n", __FILE__, __LINE__, \
                         #actual, cere_a_, cere_e_, double(tol));                                                      \
            ++cere::test::failures();                                                                                  \
        }                                                                                                              \
    } while (0)

// ---- minimal JSON (the fixture is machine-written by gen_sampler_fixtures.py) ----

struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    double number = 0.0;
    std::string text;
    std::vector<Json> items;
    std::map<std::string, Json> fields;

    const Json &operator[](const std::string &key) const {
        static const Json missing;
        const auto it = fields.find(key);
        return it == fields.end() ? missing : it->second;
    }
    const Json &operator[](size_t i) const { return items[i]; }
};

struct Parser {
    const std::string &s;
    size_t i = 0;
    bool ok = true;

    void ws() {
        while (i < s.size() && std::strchr(" \t\r\n", s[i])) ++i;
    }
    bool take(const char *word) {
        const size_t n = std::strlen(word);
        if (s.compare(i, n, word) != 0) return false;
        i += n;
        return true;
    }
    Json value() {
        ws();
        Json out;
        if (i >= s.size()) {
            ok = false;
            return out;
        }
        const char c = s[i];
        if (c == '{') {
            out.kind = Json::Object;
            ++i;
            ws();
            if (s[i] == '}') return ++i, out;
            while (ok) {
                ws();
                const std::string key = string();
                ws();
                if (s[i++] != ':') ok = false;
                out.fields[key] = value();
                ws();
                if (s[i] == ',') {
                    ++i;
                } else {
                    if (s[i++] != '}') ok = false;
                    break;
                }
            }
        } else if (c == '[') {
            out.kind = Json::Array;
            ++i;
            ws();
            if (s[i] == ']') return ++i, out;
            while (ok) {
                out.items.push_back(value());
                ws();
                if (s[i] == ',') {
                    ++i;
                } else {
                    if (s[i++] != ']') ok = false;
                    break;
                }
            }
        } else if (c == '"') {
            out.kind = Json::String;
            out.text = string();
        } else if (take("true")) {
            out.kind = Json::Bool;
            out.number = 1.0;
        } else if (take("false")) {
            out.kind = Json::Bool;
        } else if (take("null")) {
        } else {
            out.kind = Json::Number;
            if (take("NaN")) {
                out.number = std::nan("");
            } else if (take("Infinity")) {
                out.number = INFINITY;
            } else if (take("-Infinity")) {
                out.number = -INFINITY;
            } else {
                char *end = nullptr;
                out.number = std::strtod(s.c_str() + i, &end);
                if (end == s.c_str() + i) ok = false;
                i = size_t(end - s.c_str());
            }
        }
        return out;
    }
    std::string string() {
        std::string out;
        if (s[i++] != '"') ok = false;
        while (ok && i < s.size() && s[i] != '"') {
            if (s[i] == '\\') ++i;  // the fixture escapes nothing beyond `\"`/`\\`
            out += s[i++];
        }
        ++i;
        return out;
    }
};

std::vector<uint8_t> base64(const std::string &text) {
    std::vector<uint8_t> out;
    uint32_t acc = 0;
    int bits = 0;
    for (char c : text) {
        int v;
        if (c >= 'A' && c <= 'Z') v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+') v = 62;
        else if (c == '/') v = 63;
        else continue;
        acc = (acc << 6) | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t(acc >> bits));
        }
    }
    return out;
}

template <class T> std::vector<T> words(const std::string &text) {
    const std::vector<uint8_t> raw = base64(text);
    std::vector<T> out(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
    return out;
}

std::string slurp(const std::string &path, bool *exists) {
    std::ifstream in(path, std::ios::binary);
    *exists = bool(in);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string fixtures_dir() {
    std::string layout = CERE_LAYOUT_FIXTURE;
    return layout.substr(0, layout.rfind('/'));
}

// ---- statistics ----

// Regularized upper incomplete gamma Q(a, x) (series below a + 1, Lentz continued fraction above).
double gamma_q(double a, double x) {
    if (x <= 0.0) return 1.0;
    const double log_front = a * std::log(x) - x - std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a, sum = term;
        for (int n = 1; n < 1000; ++n) {
            term *= x / (a + n);
            sum += term;
            if (term < sum * 1e-16) break;
        }
        return 1.0 - sum * std::exp(log_front);
    }
    double b = x + 1.0 - a, c = 1e300, d = 1.0 / b, h = d;
    for (int n = 1; n < 1000; ++n) {
        const double an = -n * (n - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c;
        if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) < 1e-16) break;
    }
    return std::exp(log_front) * h;
}

double chi2_sf(double x, int df) { return gamma_q(df / 2.0, x / 2.0); }

struct Gof {
    int bins = 0;
    double p = 1.0;
};

// Chi-square GOF of `observed` counts against expected counts `expected` (one entry per bin).
Gof gof(const std::vector<double> &expected, const std::vector<double> &observed) {
    Gof out;
    out.bins = int(expected.size());
    if (out.bins < 2) return out;
    double x = 0.0;
    for (size_t b = 0; b < expected.size(); ++b) x += (observed[b] - expected[b]) * (observed[b] - expected[b]) / expected[b];
    out.p = chi2_sf(x, out.bins - 1);
    return out;
}

// Bin assignment per id (-1 = not binned) and the expected count per bin.
struct Bins {
    std::vector<int> of;
    std::vector<double> expected;
};

// T1: kept ids in id order, B = min(K, 20) equal-mass bins (an id goes to the bin holding its mass midpoint), then
// adjacent bins merge until each expects >= 5.
Bins t1_bins(const std::vector<uint32_t> &kept, const std::vector<double> &mass, uint64_t vocab, double n) {
    const size_t big = std::min<size_t>(kept.size(), 20);
    std::vector<int> raw(kept.size());
    std::vector<double> raw_mass(big, 0.0);
    double before = 0.0;
    for (size_t j = 0; j < kept.size(); ++j) {
        const double mid = before + mass[j] / 2.0;
        raw[j] = int(std::min<double>(double(big) - 1.0, std::floor(mid * double(big))));
        raw_mass[size_t(raw[j])] += mass[j];
        before += mass[j];
    }
    std::vector<int> merged(big);
    std::vector<double> expected;
    double acc = 0.0;
    for (size_t b = 0; b < big; ++b) {
        acc += raw_mass[b] * n;
        merged[b] = int(expected.size());
        if (acc >= 5.0) {
            expected.push_back(acc);
            acc = 0.0;
        }
    }
    if (acc > 0.0) {
        if (expected.empty()) expected.push_back(0.0);
        expected.back() += acc;
        for (size_t b = 0; b < big; ++b) merged[b] = std::min<int>(merged[b], int(expected.size()) - 1);
    }
    Bins out{std::vector<int>(vocab, -1), expected};
    for (size_t j = 0; j < kept.size(); ++j) out.of[kept[j]] = merged[size_t(raw[j])];
    return out;
}

// T2: the min(K, 19) most probable kept ids that expect >= 5 as their own bins, one pooled bin for the rest.
Bins t2_bins(const std::vector<uint32_t> &kept, const std::vector<double> &mass, uint64_t vocab, double n) {
    std::vector<size_t> order(kept.size());
    for (size_t j = 0; j < order.size(); ++j) order[j] = j;
    std::stable_sort(order.begin(), order.end(), [&mass](size_t a, size_t b) { return mass[a] > mass[b]; });
    Bins out{std::vector<int>(vocab, -1), {}};
    const size_t own = std::min<size_t>(kept.size(), 19);
    double pooled = 0.0;
    for (size_t r = 0; r < order.size(); ++r) {
        const size_t j = order[r];
        if (r < own && mass[j] * n >= 5.0) {
            out.of[kept[j]] = int(out.expected.size());
            out.expected.push_back(mass[j] * n);
        } else {
            out.of[kept[j]] = -2;
            pooled += mass[j] * n;
        }
    }
    if (pooled > 0.0) {
        for (int &b : out.of)
            if (b == -2) b = int(out.expected.size());
        out.expected.push_back(pooled);
    }
    return out;
}

Gof run_gof(const Bins &bins, const std::vector<uint32_t> &drawn) {
    std::vector<double> observed(bins.expected.size(), 0.0);
    for (uint32_t id : drawn)
        if (bins.of[id] >= 0) observed[size_t(bins.of[id])] += 1.0;
    return gof(bins.expected, observed);
}

// The fp64 running sums of the kept probabilities in id order (the D4 scan, for fast repeated draws).
struct Cdf {
    std::vector<uint32_t> ids;
    std::vector<double> sums;
    uint32_t pick(double u, size_t *index = nullptr) const {
        const double target = u * sums.back();
        const size_t j = size_t(std::upper_bound(sums.begin(), sums.end(), target) - sums.begin());
        const size_t at = std::min(j, ids.size() - 1);
        if (index) *index = at;
        return ids[at];
    }
};

Cdf cdf(const std::vector<float> &probs, const std::vector<uint32_t> &mask) {
    Cdf out;
    double acc = 0.0;
    for (uint64_t i = 0; i < probs.size(); ++i)
        if (sr::kept(mask, i)) {
            acc += probs[i];
            out.ids.push_back(uint32_t(i));
            out.sums.push_back(acc);
        }
    return out;
}

std::vector<uint32_t> ids_of(const std::vector<uint32_t> &mask, uint64_t vocab) {
    std::vector<uint32_t> out;
    for (uint64_t i = 0; i < vocab; ++i)
        if (sr::kept(mask, i)) out.push_back(uint32_t(i));
    return out;
}

uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// The draws of the distributional tests: seeds 0..N-1 at one key, channels (0, 1).
std::vector<uint32_t> seeded_draws(const Cdf &c, uint32_t key) {
    std::vector<uint32_t> out(kDraws);
    for (uint32_t s = 0; s < kDraws; ++s) out[s] = c.pick(sr::uniform(s, key, 0, 1));
    return out;
}

uint16_t to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

// ---- fixture-free checks ----

void test_murmur_and_uniform() {
    CHECK_EQ(sr::murmur_hash32(0, 0, 0), 0x8134CDF8u);
    CHECK_EQ(sr::murmur_hash32(1, 2, 3), 0x23B53F91u);
    CHECK_EQ(sr::murmur_hash32(~0ull, 0xFFFFFFFFu, 7), 0x8ED99669u);
    CHECK_EQ(sr::murmur_hash32(0x0123456789ABCDEFull, 4242, 1), 0x13DDB8D7u);
    CHECK_EQ(sr::murmur_hash32(20261010, 159, 2), 0xC73F3F6Cu);
    CHECK(sr::u_from_bits(0) == 0.0);
    CHECK(sr::u_from_bits(~0ull) == 1.0 - 0x1.0p-53);
    CHECK(sr::u_from_bits(1ull << 63) == 0.5);
    const uint64_t x = (uint64_t(sr::murmur_hash32(99, 7, 0)) << 32) | sr::murmur_hash32(99, 7, 1);
    CHECK(sr::uniform(99, 7, 0, 1) == sr::u_from_bits(x));
    CHECK(sr::uniform(99, 7, 0, 1) != sr::uniform(99, 8, 0, 1));
    CHECK_EQ(sr::choice_seed(99, 1), (uint64_t(sr::murmur_hash32(99, 1, 2)) << 32) | sr::murmur_hash32(99, 1, 3));
    CHECK(sr::choice_seed(99, 1) != sr::choice_seed(99, 2));
}

void test_process() {
    CHECK(sr::bf16_to_f32(0x3F80) == 1.0f);
    CHECK(std::isinf(sr::bf16_to_f32(0xFF80)) && sr::bf16_to_f32(0xFF80) < 0.0f);
    const uint16_t row[5] = {to_bf16(2.0f), to_bf16(-1.0f), to_bf16(0.5f), 0xFF80, to_bf16(0.75f)};
    const uint32_t counts[5] = {3, 2, 0, 1, 0};
    const sr::Bias bias[2] = {{2, 4.0f}, {0, -1.0f}};
    float out[5];
    sr::process(row, 5, counts, 0.5f, 0.25f, 1.5f, nullptr, bias, 2, out);
    CHECK(out[0] == (2.0f - (0.5f * 3.0f + 0.25f)) / 1.5f - 1.0f);
    CHECK(out[1] == (-1.0f - (0.5f * 2.0f + 0.25f)) * 1.5f);
    CHECK(out[2] == 4.5f);
    CHECK(std::isinf(out[3]) && out[3] < 0.0f);
    CHECK(out[4] == 0.75f);
    sr::process(row, 5, nullptr, 0.5f, 0.25f, 1.5f, nullptr, nullptr, 0, out);  // no output tokens: the input
    CHECK(out[0] == 2.0f && out[1] == -1.0f && out[4] == 0.75f);
}

void test_probabilities_and_logprobs() {
    const float x[4] = {1.0f, 2.0f, 2.0f, -INFINITY};
    float p[4];
    const double lse = sr::probabilities(x, 4, 0.5f, p);
    CHECK_NEAR(lse, std::log(std::exp(2.0) + 2.0 * std::exp(4.0)), 1e-12);
    CHECK(p[3] == 0.0f);
    CHECK(p[1] == p[2] && p[1] > p[0]);
    CHECK_NEAR(double(p[0]) + p[1] + p[2], 1.0, 1e-6);
    const sr::Logprobs greedy = sr::logprobs(x, 4, 0.0f, 0, 2);
    const double lse1 = std::log(std::exp(1.0) + 2.0 * std::exp(2.0));
    CHECK_NEAR(greedy.committed, 1.0 - lse1, 1e-6);
    CHECK_EQ(greedy.top.size(), 2);
    CHECK_EQ(greedy.top[0].id, 1);  // tie between ids 1 and 2 goes to the lower id
    CHECK_EQ(greedy.top[1].id, 2);
    CHECK_NEAR(greedy.top[0].logprob, 2.0 - lse1, 1e-6);
    const sr::Logprobs sampled = sr::logprobs(x, 4, 0.5f, 3, 20);
    CHECK_EQ(sampled.top.size(), 4);
    CHECK(std::isinf(sampled.committed));
    CHECK_NEAR(sampled.top[0].logprob, 4.0 - lse, 1e-6);
    CHECK_EQ(sampled.top[2].id, 0);
    CHECK(sr::logprobs(x, 4, 0.0f, 1, 0).top.empty());
}

void test_keep_mask() {
    const float a[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    CHECK(ids_of(sr::keep_mask(a, 4, 4, 0.6f, 0.0f), 4) == std::vector<uint32_t>({0, 1}));
    CHECK(ids_of(sr::keep_mask(a, 4, 1, 1.0f, 0.0f), 4) == std::vector<uint32_t>({0}));
    CHECK(ids_of(sr::keep_mask(a, 4, 4, 1.0f, 0.0f), 4) == std::vector<uint32_t>({0, 1, 2, 3}));
    const float ties[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    CHECK(ids_of(sr::keep_mask(ties, 4, 2, 1.0f, 0.0f), 4) == std::vector<uint32_t>({0, 1}));
    std::vector<uint32_t> boundary;
    // Mass ranked before id 2 is exactly 0.5 = top_p: kept (SGLang keeps `sum - p <= top_p`), and in the band.
    CHECK(ids_of(sr::keep_mask(ties, 4, 4, 0.5f, 0.0f, &boundary), 4) == std::vector<uint32_t>({0, 1, 2}));
    CHECK(boundary == std::vector<uint32_t>({2}));
    const float reversed[4] = {0.1f, 0.2f, 0.3f, 0.4f};  // ranks follow probability, not id
    CHECK(ids_of(sr::keep_mask(reversed, 4, 2, 1.0f, 0.0f), 4) == std::vector<uint32_t>({2, 3}));
    const float zero[3] = {0.5f, 0.0f, 0.5f};  // p = 0 (a -inf logit) is never kept
    CHECK(ids_of(sr::keep_mask(zero, 3, 3, 1.0f, 0.0f), 3) == std::vector<uint32_t>({0, 2}));
    CHECK(ids_of(sr::keep_mask(zero, 3, 3, 1.0f, 0.5f), 3) == std::vector<uint32_t>({0, 2}));
    // Sequential (min_p > 0): top-k 3 renorm {.556,.222,.222}; top-p .6 keeps ranks 0-1 (id 1 before id 2 by tie);
    // renorm {.714,.286}; min_p .5 drops id 1, min_p .3 keeps it.
    const float seq[4] = {0.5f, 0.2f, 0.2f, 0.1f};
    CHECK(ids_of(sr::keep_mask(seq, 4, 3, 0.6f, 0.5f), 4) == std::vector<uint32_t>({0}));
    CHECK(ids_of(sr::keep_mask(seq, 4, 3, 0.6f, 0.3f), 4) == std::vector<uint32_t>({0, 1}));
    // The same request under the joint rule (min_p = 0) keeps by the unrenormalized mass: .5, then .7 > .6 stops.
    CHECK(ids_of(sr::keep_mask(seq, 4, 3, 0.6f, 0.0f), 4) == std::vector<uint32_t>({0, 1}));
    CHECK(ids_of(sr::keep_mask(seq, 4, 3, 0.45f, 0.0f), 4) == std::vector<uint32_t>({0}));
    // min_p band: the threshold equals id 1's renormalized probability.
    const float band[2] = {0.5f, 0.25f};
    boundary.clear();
    CHECK(ids_of(sr::keep_mask(band, 2, 2, 1.0f, 0.5f, &boundary), 2) == std::vector<uint32_t>({0, 1}));
    CHECK(boundary == std::vector<uint32_t>({1}));
}

void test_draw_and_top2() {
    const float probs[5] = {0.5f, 0.0f, 0.25f, 0.125f, 0.125f};
    std::vector<uint32_t> mask(1, 0u);
    for (uint32_t id : {0u, 2u, 3u}) mask[0] |= 1u << id;  // S = 0.875
    CHECK_EQ(sr::draw(probs, 5, mask, 0.0), 0);
    CHECK_EQ(sr::draw(probs, 5, mask, 0.5), 0);  // u*S = 0.4375 < 0.5
    CHECK_EQ(sr::draw(probs, 5, mask, 0.6), 2);  // 0.525
    CHECK_EQ(sr::draw(probs, 5, mask, 0.86), 3);
    CHECK_EQ(sr::draw(probs, 5, mask, sr::u_from_bits(~0ull)), 3);
    CHECK_EQ(sr::draw(probs, 5, mask, 1.0), 3);  // u*S >= S clamps to the last kept id
    const float v[4] = {1.0f, 3.0f, 3.0f, 2.0f};
    CHECK_EQ(sr::top2(v, 4).id, 1);
    CHECK_EQ(sr::top2(v, 4).runner_up, 2);
    const float w[3] = {-INFINITY, -INFINITY, 1.0f};
    CHECK_EQ(sr::top2(w, 3).id, 2);
    CHECK_EQ(sr::top2(w, 3).runner_up, 0);
    const float flat[2] = {5.0f, 5.0f};
    CHECK_EQ(sr::top2(flat, 2).id, 0);
    CHECK_EQ(sr::top2(flat, 2).runner_up, 1);
    uint32_t counts[3] = {0, 0, 0};
    sr::count_commit(counts, 1);
    sr::count_commit(counts, 1);
    CHECK_EQ(counts[1], 2);
}

// The seeded draw path end to end on a synthetic row: the D4 counter, the u map and the inverse CDF give the
// reference's own distribution (T1 and T2 at alpha 1e-4; the seeds are fixed, so the result is reproducible).
void test_seeded_draws_follow_the_distribution() {
    const uint64_t vocab = 4096;
    std::vector<uint16_t> row(vocab);
    uint64_t state = 7;
    for (uint64_t i = 0; i < vocab; ++i) {
        const uint64_t rank = 1 + splitmix64(&state) % vocab;
        row[i] = to_bf16(float(-1.1 * std::log(double(rank))));
    }
    std::vector<float> processed(vocab), probs(vocab);
    sr::process(row.data(), vocab, nullptr, 0, 0, 1, nullptr, nullptr, 0, processed.data());
    sr::probabilities(processed.data(), vocab, 0.8f, probs.data());
    for (const auto &params : {std::make_pair(uint64_t(vocab), 0.9f), std::make_pair(uint64_t(20), 1.0f)}) {
        const std::vector<uint32_t> mask = sr::keep_mask(probs.data(), vocab, params.first, params.second, 0.0f);
        const Cdf c = cdf(probs, mask);
        for (size_t j = 0; j < 64; ++j) {
            const double u = sr::uniform(j, 31, 0, 1);
            CHECK_EQ(c.pick(u), sr::draw(probs.data(), vocab, mask, u));
        }
        std::vector<double> mass(c.ids.size());
        for (size_t j = 0; j < c.ids.size(); ++j) mass[j] = probs[c.ids[j]] / c.sums.back();
        const std::vector<uint32_t> drawn = seeded_draws(c, 31);
        for (uint32_t id : drawn) CHECK(sr::kept(mask, id));
        const Gof t1 = run_gof(t1_bins(c.ids, mass, vocab, kDraws), drawn);
        const Gof t2 = run_gof(t2_bins(c.ids, mass, vocab, kDraws), drawn);
        CHECK(t1.bins >= 2 && t2.bins >= 2);
        if (!(t1.p >= kAlpha && t2.p >= kAlpha))
            std::fprintf(stderr, "self draws top_k %llu: T1 p %g (%d bins), T2 p %g (%d bins)\n",
                         (unsigned long long)params.first, t1.p, t1.bins, t2.p, t2.bins);
        CHECK(t1.p >= kAlpha);
        CHECK(t2.p >= kAlpha);
    }
}

void test_chi2_known_answers() {
    CHECK_NEAR(chi2_sf(3.841458820694124, 1), 0.05, 1e-9);
    CHECK_NEAR(chi2_sf(30.143527205646159, 19), 0.05, 1e-9);
    CHECK_NEAR(chi2_sf(2.0, 2), std::exp(-1.0), 1e-12);
    CHECK_NEAR(chi2_sf(43.820, 19), 1e-3, 2e-6);  // table value chi2_{0.001}(19) = 43.820
}

// xgrammar 0.2.1 `testing.bitmask_to_bool_mask`, vendored semantics (the A5 bit order; the pinned image's function
// reads `bool_mask[:, i] = (bit_mask[:, i // 32] & (1 << (i % 32))) != 0` for i < vocab).
std::vector<bool> bitmask_to_bool_mask(const std::vector<uint32_t> &bits, uint64_t vocab) {
    std::vector<bool> out(vocab);
    for (uint64_t i = 0; i < vocab; ++i) out[i] = (bits[i / 32] & (1u << (i % 32))) != 0;
    return out;
}

// m28 design note C5 on the CPU reference (D9, ledger M6): `process` with a vocab mask (random, single bit, partial
// last word with every bit past the vocab set) equals the unmasked processed row with the cleared ids at -inf and
// `logit_bias` added after the mask (a bias on a cleared id leaves it -inf); top2 is the masked argmax; a null mask
// is the unmasked row.
void test_masked_process() {
    const uint64_t vocab = 1000;  // 31 full words, then 8 ids in a partial last word
    const uint64_t n_words = (vocab + 31) / 32;
    std::vector<uint16_t> row(vocab);
    std::vector<uint32_t> counts(vocab, 0u);
    uint64_t state = 2028;
    for (uint64_t i = 0; i < vocab; ++i) {
        row[i] = to_bf16(float(int64_t(splitmix64(&state) % 2001) - 1000) / 125.0f);
        if (i % 7 == 0) counts[i] = 1 + uint32_t(i % 3);
    }
    row[999] = to_bf16(20.0f);  // the unmasked maximum lies in the partial word
    std::vector<uint32_t> random(n_words), single(n_words, 0u), last(n_words, 0u);
    for (uint32_t &w : random) w = uint32_t(splitmix64(&state));
    single[500 / 32] = 1u << (500 % 32);
    last.back() = 0xFFFFFFFFu ^ (1u << (999 % 32));  // ids 992-998 allowed, 999 cleared, bits 8-31 past the vocab set
    const std::pair<const char *, std::vector<uint32_t>> cases[] = {
        {"random", random}, {"single-bit", single}, {"partial-last-word", last}};
    const sr::Bias bias[3] = {{999, 50.0f}, {500, -0.5f}, {3, 1.0f}};
    std::vector<float> plain(vocab), got(vocab);
    sr::process(row.data(), vocab, counts.data(), 0.4f, 0.2f, 1.3f, nullptr, nullptr, 0, plain.data());
    sr::process(row.data(), vocab, counts.data(), 0.4f, 0.2f, 1.3f, nullptr, nullptr, 0, got.data());
    CHECK(std::memcmp(plain.data(), got.data(), vocab * sizeof(float)) == 0);
    for (const auto &c : cases) {
        const std::vector<bool> ok = bitmask_to_bool_mask(c.second, vocab);
        sr::process(row.data(), vocab, counts.data(), 0.4f, 0.2f, 1.3f, c.second.data(), bias, 3, got.data());
        int bad = 0;
        uint64_t best = vocab;
        for (uint64_t i = 0; i < vocab; ++i) {
            float want = plain[i];
            for (const sr::Bias &b : bias)
                if (b.id == i) want += b.bias;
            if (!ok[i]) want = -INFINITY;
            bad += !(got[i] == want);
            if (ok[i] && (best == vocab || got[i] > got[best])) best = i;
        }
        if (bad) std::fprintf(stderr, "masked process %s: %d ids differ\n", c.first, bad);
        CHECK_EQ(bad, 0);
        CHECK(best < vocab);
        CHECK_EQ(sr::top2(got.data(), vocab).id, uint32_t(best));
    }
    sr::process(row.data(), vocab, counts.data(), 0.4f, 0.2f, 1.3f, single.data(), nullptr, 0, got.data());
    CHECK_EQ(sr::top2(got.data(), vocab).id, 500);
    CHECK_EQ(sr::top2(got.data(), vocab).runner_up, 0);  // every other id is -inf: ties to the lowest id
    sr::process(row.data(), vocab, counts.data(), 0.4f, 0.2f, 1.3f, last.data(), nullptr, 0, got.data());
    CHECK(std::isinf(got[999]) && got[999] < 0.0f && got[998] == plain[998]);
}

// ---- the C3 oracle ----

bool fixture_committed() {
    bool exists = false;
    const std::string manifest = slurp(fixtures_dir() + "/../../REPO_MANIFEST.json", &exists);
    return exists && manifest.find("\"tests/fixtures/native/sampler_cases.json\"") != std::string::npos;
}

void check_case(const Json &c, const std::vector<uint16_t> &row, size_t index, int *testable) {
    const std::string name = c["name"].text;
    const uint64_t vocab = row.size();
    std::vector<uint32_t> counts(vocab, 0u);
    for (const Json &e : c["counts"].items) counts[size_t(e[0].number)] = uint32_t(e[1].number);
    std::vector<sr::Bias> bias;
    for (const Json &e : c["logit_bias"].items) bias.push_back({uint32_t(e[0].number), float(e[1].number)});
    const float f = float(c["frequency_penalty"].number), pr = float(c["presence_penalty"].number);
    // M2.8 (m28 design note C9): a masked case's vocab_mask (absent = none) applies between the penalties and the
    // bias; the ids it clears are -inf in the oracle (the fixture omits them from `processed`).
    const std::vector<uint32_t> vmask = words<uint32_t>(c["vocab_mask"].text);
    const bool masked = !vmask.empty();
    if (masked) CHECK_EQ(vmask.size(), (vocab + 31) / 32);
    if (masked && vmask.size() != (vocab + 31) / 32) return;
    const auto allowed = [&](uint64_t i) { return !masked || ((vmask[i >> 5] >> (i & 31)) & 1u) != 0; };
    std::vector<float> processed(vocab);
    sr::process(row.data(), vocab, counts.data(), f, pr, float(c["repetition_penalty"].number),
                masked ? vmask.data() : nullptr, bias.data(), bias.size(), processed.data());
    // Penalties: per logit |native - SGLang| <= 2^-22 max(1, |x|) + count |f| 2^-23 (D2 arithmetic).
    std::vector<double> oracle(vocab);
    for (uint64_t i = 0; i < vocab; ++i) oracle[i] = sr::bf16_to_f32(row[i]);
    for (const Json &e : c["processed"].items) oracle[size_t(e[0].number)] = e[1].number;
    for (uint64_t i = 0; i < vocab; ++i)
        if (!allowed(i)) oracle[i] = -INFINITY;
    int bad = 0;
    for (uint64_t i = 0; i < vocab; ++i) {
        const double x = sr::bf16_to_f32(row[i]);
        if (std::isinf(oracle[i]) || std::isinf(processed[i])) {
            bad += oracle[i] != double(processed[i]);
            continue;
        }
        const double tol = 0x1.0p-22 * std::max(1.0, std::fabs(x)) + counts[i] * std::fabs(f) * 0x1.0p-23;
        bad += !(std::fabs(double(processed[i]) - oracle[i]) <= tol);
    }
    if (bad) std::fprintf(stderr, "%s: %d processed logits outside the D2 tolerance\n", name.c_str(), bad);
    CHECK_EQ(bad, 0);
    const double t = c["temperature"].number;
    const double lse = c["lse"].number;
    const sr::Logprobs lp = sr::logprobs(processed.data(), vocab, float(t), 0, 20);
    // Logprobs within 1e-5 of the fp64 oracle; the reported top-20 is the oracle's up to that tolerance.
    double outside_best = -INFINITY;
    std::vector<bool> listed(vocab, false);
    for (const auto &e : lp.top) listed[e.id] = true;
    for (uint64_t i = 0; i < vocab; ++i)
        if (!listed[i]) outside_best = std::max(outside_best, (t == 0.0 ? oracle[i] : oracle[i] / t) - lse);
    for (const auto &e : lp.top) {
        const double want = (t == 0.0 ? oracle[e.id] : oracle[e.id] / t) - lse;
        if (std::isinf(want)) CHECK(std::isinf(e.logprob) && e.logprob < 0.0f);  // a -inf input stays -inf (D7)
        else CHECK_NEAR(e.logprob, want, 1e-5);
        CHECK(want >= outside_best - 2e-5);
    }
    if (masked) {
        // C9: logprobs are post-mask. A listed id the mask clears has a -inf logprob, a masked committed id too, and
        // an allowed committed id's logprob is the oracle's over the masked row (the fixture's lse is SGLang's after
        // the mask).
        for (const auto &e : lp.top) CHECK(allowed(e.id) || (std::isinf(e.logprob) && e.logprob < 0.0f));
        uint64_t first_allowed = 0, first_masked = 0;
        while (!allowed(first_allowed)) ++first_allowed;
        while (first_masked < vocab && allowed(first_masked)) ++first_masked;
        const sr::Logprobs one = sr::logprobs(processed.data(), vocab, float(t), uint32_t(first_allowed), 0);
        CHECK_NEAR(one.committed, (t == 0.0 ? oracle[first_allowed] : oracle[first_allowed] / t) - lse, 1e-5);
        if (first_masked < vocab) {
            const float lp = sr::logprobs(processed.data(), vocab, float(t), uint32_t(first_masked), 0).committed;
            CHECK(std::isinf(lp) && lp < 0.0f);
        }
    }
    if (t == 0.0) {
        const sr::Top2 top = sr::top2(processed.data(), vocab);
        CHECK_EQ(top.id, uint32_t(c["greedy"]["id"].number));
        CHECK_EQ(top.runner_up, uint32_t(c["greedy"]["runner_up"].number));
        CHECK(allowed(top.id) && allowed(top.runner_up));
        return;
    }
    std::vector<float> probs(vocab);
    sr::probabilities(processed.data(), vocab, float(t), probs.data());
    std::vector<uint32_t> boundary;
    const std::vector<uint32_t> mask = sr::keep_mask(probs.data(), vocab, uint64_t(c["top_k"].number),
                                                     float(c["top_p"].number), float(c["min_p"].number), &boundary);
    std::vector<uint32_t> want = words<uint32_t>(c["mask"].text);
    CHECK_EQ(want.size(), mask.size());
    if (want.size() != mask.size()) return;
    std::vector<uint32_t> differ;
    for (uint64_t i = 0; i < vocab; ++i)
        if (sr::kept(mask, i) != sr::kept(want, i)) differ.push_back(uint32_t(i));
    for (uint32_t id : differ) {
        const bool listed_boundary = std::binary_search(boundary.begin(), boundary.end(), id);
        std::fprintf(stderr, "%s: mask differs at id %u (%s)\n", name.c_str(), id,
                     listed_boundary ? "within the 1e-6 band, listed" : "outside the band");
        CHECK(listed_boundary);
    }
    // Draw (primary, exact): the oracle's mask, the reference's probabilities and scan, 4,096 u values.
    const Cdf oc = cdf(probs, want);
    uint64_t state = std::strtoull(c["draw_seed"].text.c_str(), nullptr, 10);
    const Json &ids = c["draw_ids"];
    CHECK_EQ(ids.items.size(), 4096);
    int moved = 0, escaped = 0;
    for (size_t j = 0; j < ids.items.size(); ++j) {
        const uint64_t x = j == 0 ? 0 : j == 1 ? ~0ull : splitmix64(&state);
        const double u = sr::u_from_bits(x);
        size_t at = 0;
        const uint32_t got = oc.pick(u, &at);
        escaped += !allowed(got) + !allowed(uint32_t(ids[j].number));  // C9: no draw outside the vocab mask
        if (j % 256 < 2) CHECK_EQ(sr::draw(probs.data(), vocab, want, u), got);
        const uint32_t expect = uint32_t(ids[j].number);
        if (got == expect) continue;
        const auto it = std::lower_bound(oc.ids.begin(), oc.ids.end(), expect);
        const size_t other = size_t(it - oc.ids.begin());
        bool near = it != oc.ids.end() && *it == expect;
        const double target = u * oc.sums.back();
        for (size_t k = std::min(at, other); near && k < std::max(at, other); ++k)
            near = std::fabs(oc.sums[k] - target) <= 1e-6;
        if (!near) std::fprintf(stderr, "%s: draw %zu id %u, oracle %u\n", name.c_str(), j, got, expect);
        CHECK(near);
        ++moved;
    }
    CHECK_EQ(escaped, 0);
    if (moved) std::printf("%s: %d draws within 1e-6 of a CDF boundary differ (listed)\n", name.c_str(), moved);
    if (!c["distributional"].number) return;
    // Distributional (secondary): N draws, seeds 0..N-1, the reference's own mask; expected from the oracle's
    // probabilities exp(processed/T - lse) over the oracle's mask.
    const Cdf rc = cdf(probs, mask);
    const std::vector<uint32_t> drawn = seeded_draws(rc, uint32_t(index));
    int outside = 0;
    for (uint32_t id : drawn)
        if (!sr::kept(want, id) && !std::binary_search(boundary.begin(), boundary.end(), id)) ++outside;
    CHECK_EQ(outside, 0);
    int escaped_seeded = 0;
    for (uint32_t id : drawn) escaped_seeded += !allowed(id);
    CHECK_EQ(escaped_seeded, 0);  // C9: none of the N seeded draws lies outside the vocab mask (no band excuse)
    std::vector<double> mass(oc.ids.size());
    double total = 0.0;
    for (size_t j = 0; j < oc.ids.size(); ++j) total += (mass[j] = std::exp(oracle[oc.ids[j]] / t - lse));
    for (double &m : mass) m /= total;
    for (int test = 1; test <= 2; ++test) {
        const Bins bins = test == 1 ? t1_bins(oc.ids, mass, vocab, kDraws) : t2_bins(oc.ids, mass, vocab, kDraws);
        const Gof g = run_gof(bins, drawn);
        if (g.bins < 2) {
            std::printf("%s T%d: fewer than 2 bins, covered by the exact draw check\n", name.c_str(), test);
            continue;
        }
        ++*testable;
        std::printf("%s T%d: %d bins, p %.3g\n", name.c_str(), test, g.bins, g.p);
        CHECK(g.p >= kAlpha);
    }
}

void test_oracle_fixture() {
    bool exists = false;
    const std::string text = slurp(fixtures_dir() + "/native/sampler_cases.json", &exists);
    if (!exists) {
        CHECK(!fixture_committed());
        std::printf("test_sampler: oracle fixture absent (C3 not yet run in the pinned image): oracle checks skipped\n");
        return;
    }
    Parser parser{text};
    const Json doc = parser.value();
    CHECK(parser.ok);
    CHECK(doc["schema"].text == "cere-sampler-cases-v1");
    if (!parser.ok || doc["schema"].text != "cere-sampler-cases-v1") return;
    CHECK_EQ(doc["murmur"].items.size(), 10000);
    int murmur_bad = 0;
    for (const Json &m : doc["murmur"].items)
        murmur_bad += sr::murmur_hash32(std::strtoull(m[0].text.c_str(), nullptr, 10), uint32_t(m[1].number),
                                        uint32_t(m[2].number)) != uint32_t(m[3].number);
    CHECK_EQ(murmur_bad, 0);
    std::map<std::string, std::vector<uint16_t>> rows;
    for (const auto &r : doc["rows"].fields) rows[r.first] = words<uint16_t>(r.second.text);
    int testable = 0;
    for (size_t i = 0; i < doc["cases"].items.size(); ++i) {
        const Json &c = doc["cases"][i];
        check_case(c, rows[c["row"].text], i, &testable);
    }
    std::printf("test_sampler: %zu oracle cases, %d testable T1/T2 tests\n", doc["cases"].items.size(), testable);
}

}  // namespace

int main() {
    test_murmur_and_uniform();
    test_process();
    test_masked_process();
    test_probabilities_and_logprobs();
    test_keep_mask();
    test_draw_and_top2();
    test_chi2_known_answers();
    test_seeded_draws_follow_the_distribution();
    test_oracle_fixture();
    return cere::test::finish("test_sampler");
}
