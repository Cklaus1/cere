// CPU reference of the M2.7 sampler; see sampler_ref.h.
#include "host/sampler_ref.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace cere::sampler_ref {
namespace {

uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

uint32_t mix(uint32_t h, uint32_t k) {
    k *= 0xCC9E2D51u;
    k = rotl32(k, 15);
    k *= 0x1B873593u;
    h ^= k;
    h = rotl32(h, 13);
    return h * 5u + 0xE6546B64u;
}

// The kept ids (p > 0), most probable first, ties to the lower id.
std::vector<uint32_t> ranked(const float *probs, uint64_t vocab) {
    std::vector<uint32_t> order;
    for (uint64_t i = 0; i < vocab; ++i)
        if (probs[i] > 0.0f) order.push_back(uint32_t(i));
    std::sort(order.begin(), order.end(), [probs](uint32_t a, uint32_t b) {
        return probs[a] != probs[b] ? probs[a] > probs[b] : a < b;
    });
    return order;
}

constexpr double kBand = 1e-6;

}  // namespace

uint32_t murmur_hash32(uint64_t seed, uint32_t key, uint32_t c) {
    uint32_t h = mix(0, uint32_t(seed));
    h = mix(h, uint32_t(seed >> 32));
    h = mix(h, key);
    h = mix(h, c);
    h ^= 16u;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}

double u_from_bits(uint64_t x) { return double(x >> 11) * 0x1.0p-53; }

double uniform(uint64_t seed, uint32_t key, uint32_t ca, uint32_t cb) {
    const uint64_t x = (uint64_t(murmur_hash32(seed, key, ca)) << 32) | murmur_hash32(seed, key, cb);
    return u_from_bits(x);
}

uint64_t choice_seed(uint64_t seed, uint32_t i) {
    return (uint64_t(murmur_hash32(seed, i, 2)) << 32) | murmur_hash32(seed, i, 3);
}

float bf16_to_f32(uint16_t bits) {
    const uint32_t wide = uint32_t(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

void process(const uint16_t *logits_bf16, uint64_t vocab, const uint32_t *counts, float frequency_penalty,
             float presence_penalty, float repetition_penalty, const uint32_t *mask, const Bias *bias,
             size_t bias_count, float *out) {
    for (uint64_t i = 0; i < vocab; ++i) {
        float x = bf16_to_f32(logits_bf16[i]);
        if (counts && counts[i] > 0) {
            x -= frequency_penalty * float(counts[i]) + presence_penalty;
            x = x < 0.0f ? x * repetition_penalty : x / repetition_penalty;
        }
        if (mask && !((mask[i >> 5] >> (i & 31)) & 1u)) x = -std::numeric_limits<float>::infinity();
        out[i] = x;
    }
    for (size_t j = 0; j < bias_count; ++j) out[bias[j].id] += bias[j].bias;
}

double probabilities(const float *processed, uint64_t vocab, float temperature, float *probs) {
    double m = -std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < vocab; ++i) {
        probs[i] = processed[i] / temperature;
        m = std::max(m, double(probs[i]));
    }
    double sum = 0.0;
    for (uint64_t i = 0; i < vocab; ++i) sum += std::exp(double(probs[i]) - m);
    const double lse = m + std::log(sum);
    for (uint64_t i = 0; i < vocab; ++i) probs[i] = float(std::exp(double(probs[i]) - lse));
    return lse;
}

std::vector<uint32_t> keep_mask(const float *probs, uint64_t vocab, uint64_t top_k, float top_p, float min_p,
                                std::vector<uint32_t> *boundary) {
    std::vector<uint32_t> mask((vocab + 31) / 32, 0u);
    const std::vector<uint32_t> order = ranked(probs, vocab);
    const double p_cut = top_p;
    const size_t k = size_t(std::min<uint64_t>(top_k, order.size()));
    auto keep = [&mask](uint32_t id) { mask[id / 32] |= 1u << (id % 32); };
    auto near = [boundary](uint32_t id, double value, double threshold, double band) {
        if (boundary && std::fabs(value - threshold) <= band) boundary->push_back(id);
    };
    if (min_p == 0.0f) {
        double before = 0.0;
        for (size_t r = 0; r < order.size(); ++r) {
            const uint32_t id = order[r];
            if (r < k) {
                near(id, before, p_cut, kBand);
                if (before <= p_cut) keep(id);
            }
            before += probs[id];
        }
    } else {
        std::vector<double> q(k);
        double total = 0.0;
        for (size_t r = 0; r < k; ++r) total += (q[r] = probs[order[r]]);
        double before = 0.0;
        double kept_total = 0.0;
        size_t n = 0;
        for (size_t r = 0; r < k; ++r) {
            q[r] /= total;
            near(order[r], before, p_cut, kBand);
            if (before <= p_cut) {
                kept_total += q[r];
                n = r + 1;
            }
            before += q[r];
        }
        const double threshold = (q[0] / kept_total) * double(min_p);
        for (size_t r = 0; r < n; ++r) {
            const double value = q[r] / kept_total;
            near(order[r], value, threshold, kBand * threshold);
            if (value >= threshold) keep(order[r]);
        }
    }
    if (boundary) std::sort(boundary->begin(), boundary->end());
    return mask;
}

uint32_t draw(const float *probs, uint64_t vocab, const std::vector<uint32_t> &mask, double u) {
    double total = 0.0;
    uint32_t last = 0;
    for (uint64_t i = 0; i < vocab; ++i)
        if (kept(mask, i)) {
            total += probs[i];
            last = uint32_t(i);
        }
    const double target = u * total;
    double acc = 0.0;
    for (uint64_t i = 0; i < vocab; ++i)
        if (kept(mask, i)) {
            acc += probs[i];
            if (acc > target) return uint32_t(i);
        }
    return last;
}

Top2 top2(const float *values, uint64_t vocab) {
    Top2 out{0, 1};
    if (values[1] > values[0]) out = {1, 0};
    for (uint64_t i = 2; i < vocab; ++i) {
        if (values[i] > values[out.id]) {
            out.runner_up = out.id;
            out.id = uint32_t(i);
        } else if (values[i] > values[out.runner_up]) {
            out.runner_up = uint32_t(i);
        }
    }
    return out;
}

Logprobs logprobs(const float *processed, uint64_t vocab, float temperature, uint32_t committed, uint32_t n) {
    std::vector<float> z(processed, processed + vocab);
    if (temperature != 0.0f)
        for (float &v : z) v /= temperature;
    double m = -std::numeric_limits<double>::infinity();
    for (float v : z) m = std::max(m, double(v));
    double sum = 0.0;
    for (float v : z) sum += std::exp(double(v) - m);
    const double lse = m + std::log(sum);
    std::vector<uint32_t> ids(vocab);
    std::iota(ids.begin(), ids.end(), 0u);
    const size_t top = std::min<size_t>(n, vocab);
    std::partial_sort(ids.begin(), ids.begin() + top, ids.end(),
                      [&z](uint32_t a, uint32_t b) { return z[a] != z[b] ? z[a] > z[b] : a < b; });
    Logprobs out{float(double(z[committed]) - lse), {}};
    for (size_t j = 0; j < top; ++j) out.top.push_back({ids[j], float(double(z[ids[j]]) - lse)});
    return out;
}

}  // namespace cere::sampler_ref
