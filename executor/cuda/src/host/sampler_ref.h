// CPU reference of the M2.7 sampler (m27 design note D2-D5, D7, C2): the toy device, the host tests and the K12
// kernel suite all compare against it. Pure C++17, no CUDA and no ABI struct bodies. One item, one scored row.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cere::sampler_ref {

// `murmur_hash32_kernel` (SGLang `kernels/ops/sampling/murmur_hash.py:20-48,92-101`) for one (seed, key, column):
// MurmurHash3_x86_32 with seed 0 over the 16 bytes seed_lo, seed_hi, key, c.
uint32_t murmur_hash32(uint64_t seed, uint32_t key, uint32_t c);

// D4 / ledger M8: u = (x >> 11) * 2^-53 in [0, 1).
double u_from_bits(uint64_t x);

// x = h_ca * 2^32 + h_cb with h_c = murmur_hash32(seed, key, c); returns u_from_bits(x). A plain draw uses channels
// (0, 1) and key = the session position of the last fed token.
double uniform(uint64_t seed, uint32_t key, uint32_t ca, uint32_t cb);

// The seed of sampled choice i >= 1 (M8 channels 2, 3, key = i); choice 0 uses `seed` itself.
uint64_t choice_seed(uint64_t seed, uint32_t i);

float bf16_to_f32(uint16_t bits);

struct Bias {
    uint32_t id;
    float bias;
};

// D2 plus M2.8's mask (m28 design note D9, ledger M6): BF16 -> fp32, then for an id with count c > 0
// `x - (f*c + p)` (f*c one fp32 multiply) and the repetition scaling (x < 0: x*r, else x/r, as SGLang), then ids
// whose bit is clear in `mask` (ceil(vocab/32) words, id j allowed iff (mask[j >> 5] >> (j & 31)) & 1, the xgrammar
// layout; null = no mask) become -inf, then `logit_bias` added in list order. `counts` holds `vocab` entries or is
// null (no output tokens yet).
void process(const uint16_t *logits_bf16, uint64_t vocab, const uint32_t *counts, float frequency_penalty,
             float presence_penalty, float repetition_penalty, const uint32_t *mask, const Bias *bias,
             size_t bias_count, float *out);

// Sampled path: z = x / T in fp32, the fp64 log-sum-exp of z, p = float(exp(z - lse)); -inf entries get 0.
// Returns the log-sum-exp.
double probabilities(const float *processed, uint64_t vocab, float temperature, float *probs);

// D3 keep mask over p > 0 entries, ranked by probability with ties to the lower id, masses in fp64 from the fp32
// probabilities. min_p == 0: the joint mask (rank < top_k and the mass ranked before the id <= top_p). min_p > 0:
// top_k then renormalize, top_p (same rule) then renormalize, then p >= p_max * min_p. Returns ceil(vocab/32)
// words, bit i % 32 of word i / 32 set = kept. `boundary`, when given, receives the ids whose ranked-before mass
// lies within 1e-6 (abs) of top_p or whose probability lies within 1e-6 (rel) of p_max * min_p (C2's exclusion band).
std::vector<uint32_t> keep_mask(const float *probs, uint64_t vocab, uint64_t top_k, float top_p, float min_p,
                                std::vector<uint32_t> *boundary = nullptr);

inline bool kept(const std::vector<uint32_t> &mask, uint64_t id) { return (mask[id / 32] >> (id % 32)) & 1u; }

// D4 inverse CDF over the kept ids in id order: the fp64 running sum of the unnormalized probabilities against u*S
// (S the fp64 kept total); the first id whose running sum exceeds u*S, clamped to the last kept id. The mask must
// keep at least one id.
uint32_t draw(const float *probs, uint64_t vocab, const std::vector<uint32_t> &mask, double u);

struct Top2 {
    uint32_t id;
    uint32_t runner_up;
};

// The two highest values, ties to the lower id (`k1_primitives.cu:228-235`). vocab >= 2.
Top2 top2(const float *values, uint64_t vocab);

struct LogprobEntry {
    uint32_t id;
    float logprob;
};

struct Logprobs {
    float committed;                 // the committed token's logprob
    std::vector<LogprobEntry> top;   // the top-N, descending, ties to the lower id
};

// D7: log_softmax(processed) for temperature 0 (greedy), log_softmax(processed / T) otherwise, unfiltered, fp64
// log-sum-exp; the committed token plus the top `n` (n <= 20).
Logprobs logprobs(const float *processed, uint64_t vocab, float temperature, uint32_t committed, uint32_t n);

// D5: the committed token's count (`out_counts`, reset at every PREFILL work item by the device).
inline void count_commit(uint32_t *counts, uint32_t id) { ++counts[id]; }

}  // namespace cere::sampler_ref
