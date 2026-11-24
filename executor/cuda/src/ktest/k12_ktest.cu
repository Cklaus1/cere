// K12 kernel-test entry points (executor/tests/gpu/test_kernels_k12_sampler.py; prototypes in its docstring;
// m27 design note C11, ledger table R K12). Device pointers, int64 sizes, `float` scalars, the stream last; 0 =
// success (CERE_OK), CERE_ERR_ARGUMENT for arguments outside a kernel's contract, CERE_ERR_CAPACITY when the
// stream-ordered scratch cannot be allocated, CERE_ERR_INTERNAL for a failed launch or copy. Contracts in
// kernels/k12_sampler.h; `cere_ktest_sample_ref_keep_mask` runs `host/sampler_ref` (C2) on a host copy, and
// `cere_ktest_process_masked` (M2.8, m28 design note C5/C9, ledger table R K12) runs `process` with a mask and,
// when asked, `sampler_ref::process` with the same mask on host copies.
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "host/sampler_ref.h"
#include "kernels/k12_sampler.h"
#include "ktest/k1_ktest.h"

using cere::k12::bf16;
using cere::ktest::status_of;

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

constexpr size_t kAlign = 256;

size_t aligned(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

bool vocab_ok(int64_t vocab) { return vocab >= 2 && vocab <= std::numeric_limits<int32_t>::max(); }

// Stream-ordered device scratch freed on scope exit (cudaFreeAsync on the same stream).
class StreamScratch {
  public:
    StreamScratch(size_t bytes, cudaStream_t stream) : stream_(stream) {
        if (cudaMallocAsync(&ptr_, bytes, stream) != cudaSuccess) ptr_ = nullptr;
    }
    StreamScratch(const StreamScratch &) = delete;
    StreamScratch &operator=(const StreamScratch &) = delete;
    // CERE_ERR_INTERNAL when the free fails, else `status`.
    int release(int status) {
        const cudaError_t freed = ptr_ != nullptr ? cudaFreeAsync(ptr_, stream_) : cudaSuccess;
        ptr_ = nullptr;
        return status == CERE_OK && freed != cudaSuccess ? CERE_ERR_INTERNAL : status;
    }
    ~StreamScratch() { release(CERE_OK); }
    unsigned char *get() const { return static_cast<unsigned char *>(ptr_); }

  private:
    void *ptr_ = nullptr;
    cudaStream_t stream_;
};

// The kernels' own probabilities and keep mask of processed / T, then `draw` (a draw_u or draw_seeded call) on them,
// in stream-ordered scratch.
template <typename Draw>
int sampled(const void *processed, int64_t vocab, float temperature, int64_t top_k, float top_p, float min_p,
            cudaStream_t stream, Draw draw) {
    if (!processed || !vocab_ok(vocab)) return CERE_ERR_ARGUMENT;
    const size_t probs_bytes = aligned(size_t(vocab) * sizeof(float));
    const size_t mask_bytes = aligned(size_t((vocab + 31) / 32) * sizeof(uint32_t));
    const size_t lse_bytes = aligned(sizeof(double));
    StreamScratch scratch(probs_bytes + mask_bytes + lse_bytes + cere::k12::scratch_bytes(), stream);
    if (scratch.get() == nullptr) return CERE_ERR_CAPACITY;
    unsigned char *base = scratch.get();
    auto *probs = reinterpret_cast<float *>(base);
    auto *mask = reinterpret_cast<uint32_t *>(base + probs_bytes);
    auto *lse = reinterpret_cast<double *>(base + probs_bytes + mask_bytes);
    void *work = base + probs_bytes + mask_bytes + lse_bytes;
    cudaError_t err = cere::k12::probabilities(static_cast<const float *>(processed), vocab, temperature, probs, lse,
                                               work, stream);
    if (err == cudaSuccess) err = cere::k12::keep_mask(probs, vocab, top_k, top_p, min_p, mask, work, stream);
    if (err == cudaSuccess) err = draw(probs, mask, work);
    return scratch.release(status_of(err));
}

// The device buffers of cere_ktest_sample_stage_time, carved like model.cpp carve_sampler.
struct StageBuffers {
    float *proc = nullptr;
    float *probs = nullptr;
    uint32_t *keep = nullptr;
    double *lse = nullptr;
    void *scratch = nullptr;
    uint32_t *top2_ids = nullptr;
    float *top2_scores = nullptr;
    uint32_t *drawn = nullptr;
};

// Carves StageBuffers from `base` (null: sizes only); returns the bytes used.
size_t carve_stage(int64_t vocab, unsigned char *base, StageBuffers *out) {
    size_t at = 0;
    const auto take = [&](size_t bytes) {
        void *p = base != nullptr ? base + at : nullptr;
        at += aligned(bytes);
        return p;
    };
    StageBuffers b;
    b.proc = static_cast<float *>(take(size_t(vocab) * sizeof(float)));
    b.probs = static_cast<float *>(take(size_t(vocab) * sizeof(float)));
    b.keep = static_cast<uint32_t *>(take(size_t((vocab + 31) / 32) * sizeof(uint32_t)));
    b.lse = static_cast<double *>(take(sizeof(double)));
    b.scratch = take(cere::k12::scratch_bytes());
    b.top2_ids = static_cast<uint32_t *>(take(2 * sizeof(uint32_t)));
    b.top2_scores = static_cast<float *>(take(2 * sizeof(float)));
    b.drawn = static_cast<uint32_t *>(take(sizeof(uint32_t)));
    if (out != nullptr) *out = b;
    return at;
}

// forward.cpp sample_row's sampled path (no logit_bias, no logprobs) for each row r: process with counts row r, then
// top2_f32, probabilities, keep_mask, draw_seeded (seed, key key0 + r), finalize into scores[r], count_commit.
cudaError_t sample_stage(const bf16 *logits, int64_t vocab, int64_t rows, uint32_t *counts, float temperature,
                         int64_t top_k, float top_p, float min_p, uint64_t seed, uint32_t key0, const StageBuffers &b,
                         cere_token_score *scores, cudaStream_t stream) {
    for (int64_t r = 0; r < rows; ++r) {
        uint32_t *row_counts = counts + r * vocab;
        cudaError_t err = cere::k12::process(logits + r * vocab, vocab, row_counts, 0.0f, 0.0f, 1.0f, nullptr, nullptr,
                                             nullptr, 1, 0, b.proc, stream);
        if (err == cudaSuccess) err = cere::k12::top2_f32(b.proc, vocab, b.top2_ids, b.top2_scores, stream);
        if (err == cudaSuccess) {
            err = cere::k12::probabilities(b.proc, vocab, temperature, b.probs, b.lse, b.scratch, stream);
        }
        if (err == cudaSuccess) {
            err = cere::k12::keep_mask(b.probs, vocab, top_k, top_p, min_p, b.keep, b.scratch, stream);
        }
        if (err == cudaSuccess) {
            err = cere::k12::draw_seeded(b.probs, b.keep, vocab, seed, 1, key0 + uint32_t(r), b.drawn, b.scratch,
                                         stream);
        }
        if (err == cudaSuccess) err = cere::k12::finalize(b.proc, b.top2_ids, b.drawn, scores + r, stream);
        if (err == cudaSuccess) err = cere::k12::count_commit(row_counts, &scores[r].id, 1, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

}  // namespace

CERE_KTEST int cere_ktest_sample_process(const void *logits, int64_t vocab, const void *counts,
                                         float frequency_penalty, float presence_penalty, float repetition_penalty,
                                         const void *bias_ids, const void *bias_values, int64_t bias_count,
                                         void *processed, cudaStream_t stream) {
    return status_of(cere::k12::process(static_cast<const bf16 *>(logits), vocab, static_cast<const uint32_t *>(counts),
                                        frequency_penalty, presence_penalty, repetition_penalty, nullptr,
                                        static_cast<const uint32_t *>(bias_ids), static_cast<const float *>(bias_values),
                                        1, bias_count, static_cast<float *>(processed), stream));
}

// M2.8 (m28 design note A5, C5): k12 `process` with mask_dev = `mask` (ceil(vocab / 32) words, non-null), then,
// when `ref_processed` is non-null, `sampler_ref::process` on host copies of the same inputs written there.
CERE_KTEST int cere_ktest_process_masked(const void *logits, int64_t vocab, const void *counts,
                                         float frequency_penalty, float presence_penalty, float repetition_penalty,
                                         const void *mask, const void *bias_ids, const void *bias_values,
                                         int64_t bias_count, void *processed, void *ref_processed,
                                         cudaStream_t stream) {
    if (!logits || !mask || !processed || !vocab_ok(vocab) || bias_count < 0 || bias_count > vocab ||
        (bias_count > 0 && (!bias_ids || !bias_values))) {
        return CERE_ERR_ARGUMENT;
    }
    const int status = status_of(cere::k12::process(
        static_cast<const bf16 *>(logits), vocab, static_cast<const uint32_t *>(counts), frequency_penalty,
        presence_penalty, repetition_penalty, static_cast<const uint32_t *>(mask),
        static_cast<const uint32_t *>(bias_ids), static_cast<const float *>(bias_values), 1, bias_count,
        static_cast<float *>(processed), stream));
    if (status != CERE_OK || ref_processed == nullptr) return status;
    try {
        const size_t n = size_t(vocab), words = size_t((vocab + 31) / 32), nb = size_t(bias_count);
        std::vector<uint16_t> x(n);
        std::vector<uint32_t> c(counts != nullptr ? n : 0), m(words), ids(nb);
        std::vector<float> values(nb), out(n);
        const auto down = [&](void *dst, const void *src, size_t bytes) {
            return bytes == 0 || cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
        };
        if (!down(x.data(), logits, n * sizeof(uint16_t)) || !down(c.data(), counts, c.size() * sizeof(uint32_t)) ||
            !down(m.data(), mask, words * sizeof(uint32_t)) || !down(ids.data(), bias_ids, nb * sizeof(uint32_t)) ||
            !down(values.data(), bias_values, nb * sizeof(float)) || cudaStreamSynchronize(stream) != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
        std::vector<cere::sampler_ref::Bias> bias(nb);
        for (size_t j = 0; j < nb; ++j) bias[j] = {ids[j], values[j]};
        cere::sampler_ref::process(x.data(), uint64_t(vocab), counts != nullptr ? c.data() : nullptr,
                                   frequency_penalty, presence_penalty, repetition_penalty, m.data(), bias.data(),
                                   nb, out.data());
        if (cudaMemcpyAsync(ref_processed, out.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}

CERE_KTEST int cere_ktest_sample_top2_f32(const void *values, int64_t vocab, void *ids, void *scores,
                                          cudaStream_t stream) {
    return status_of(cere::k12::top2_f32(static_cast<const float *>(values), vocab, static_cast<uint32_t *>(ids),
                                         static_cast<float *>(scores), stream));
}

CERE_KTEST int cere_ktest_sample_probs(const void *processed, int64_t vocab, float temperature, int64_t top_k,
                                       float top_p, float min_p, void *probs, void *mask, void *lse,
                                       cudaStream_t stream) {
    StreamScratch work(cere::k12::scratch_bytes(), stream);
    if (work.get() == nullptr) return CERE_ERR_CAPACITY;
    cudaError_t err = cere::k12::probabilities(static_cast<const float *>(processed), vocab, temperature,
                                               static_cast<float *>(probs), static_cast<double *>(lse), work.get(),
                                               stream);
    if (err == cudaSuccess) {
        err = cere::k12::keep_mask(static_cast<const float *>(probs), vocab, top_k, top_p, min_p,
                                   static_cast<uint32_t *>(mask), work.get(), stream);
    }
    return work.release(status_of(err));
}

CERE_KTEST int cere_ktest_sample_draw(const void *processed, int64_t vocab, float temperature, int64_t top_k,
                                      float top_p, float min_p, const void *us, int64_t n, void *ids,
                                      cudaStream_t stream) {
    return sampled(processed, vocab, temperature, top_k, top_p, min_p, stream,
                   [&](const float *probs, const uint32_t *mask, void *scratch) {
                       return cere::k12::draw_u(probs, mask, vocab, static_cast<const double *>(us), n,
                                                static_cast<uint32_t *>(ids), scratch, stream);
                   });
}

CERE_KTEST int cere_ktest_sample_seeded(const void *processed, int64_t vocab, float temperature, int64_t top_k,
                                        float top_p, float min_p, int64_t first_seed, int64_t n_seeds, int64_t key,
                                        void *ids, cudaStream_t stream) {
    if (first_seed < 0 || key < 0 || key > int64_t(std::numeric_limits<uint32_t>::max())) return CERE_ERR_ARGUMENT;
    return sampled(processed, vocab, temperature, top_k, top_p, min_p, stream,
                   [&](const float *probs, const uint32_t *mask, void *scratch) {
                       return cere::k12::draw_seeded(probs, mask, vocab, uint64_t(first_seed), n_seeds,
                                                     uint32_t(key), static_cast<uint32_t *>(ids), scratch, stream);
                   });
}

CERE_KTEST int cere_ktest_sample_top_logprobs(const void *processed, int64_t vocab, float temperature,
                                              int64_t committed, int64_t n, void *committed_logprob, void *top_ids,
                                              void *top_logprobs, cudaStream_t stream) {
    if (committed < 0 || committed >= vocab) return CERE_ERR_ARGUMENT;
    return status_of(cere::k12::top_logprobs(static_cast<const float *>(processed), vocab, temperature, nullptr,
                                             uint32_t(committed), n, static_cast<float *>(committed_logprob),
                                             static_cast<uint32_t *>(top_ids), static_cast<float *>(top_logprobs), 1,
                                             stream));
}

CERE_KTEST int cere_ktest_sample_count_commit(void *counts, const void *ids, int64_t n, cudaStream_t stream) {
    return status_of(
        cere::k12::count_commit(static_cast<uint32_t *>(counts), static_cast<const uint32_t *>(ids), n, stream));
}

CERE_KTEST int cere_ktest_sample_stage_time(const void *logits, int64_t vocab, int64_t rows, void *counts,
                                            float temperature, int64_t top_k, float top_p, float min_p, int64_t seed,
                                            int64_t steps, void *step_us, void *ids, cudaStream_t stream) {
    constexpr int64_t kWarmup = 3;
    if (!logits || !counts || !step_us || !ids || !vocab_ok(vocab) || rows < 1 || rows > 64 || seed < 0 ||
        steps < 1 || steps > 1000 || !(temperature > 0.0f)) {
        return CERE_ERR_ARGUMENT;
    }
    const size_t stage_bytes = carve_stage(vocab, nullptr, nullptr);
    StreamScratch scratch(stage_bytes + aligned(size_t(rows) * sizeof(cere_token_score)), stream);
    if (scratch.get() == nullptr) return CERE_ERR_CAPACITY;
    StageBuffers b;
    carve_stage(vocab, scratch.get(), &b);
    auto *scores = reinterpret_cast<cere_token_score *>(scratch.get() + stage_bytes);
    const auto *x = static_cast<const bf16 *>(logits);
    auto *c = static_cast<uint32_t *>(counts);
    auto *out_ids = static_cast<uint32_t *>(ids);
    std::vector<cudaEvent_t> events;
    int status = CERE_OK;
    try {
        events.reserve(size_t(2 * steps));
        for (int64_t e = 0; e < 2 * steps && status == CERE_OK; ++e) {
            cudaEvent_t ev = nullptr;
            if (cudaEventCreate(&ev) != cudaSuccess) {
                status = CERE_ERR_INTERNAL;
            } else {
                events.push_back(ev);
            }
        }
    } catch (const std::bad_alloc &) {
        status = CERE_ERR_CAPACITY;
    }
    cudaError_t err = cudaSuccess;
    for (int64_t s = 0; s < kWarmup && status == CERE_OK && err == cudaSuccess; ++s) {
        err = sample_stage(x, vocab, rows, c, temperature, top_k, top_p, min_p, uint64_t(seed), 0, b, scores, stream);
    }
    for (int64_t s = 0; s < steps && status == CERE_OK && err == cudaSuccess; ++s) {
        err = cudaEventRecord(events[size_t(2 * s)], stream);
        if (err == cudaSuccess) {
            err = sample_stage(x, vocab, rows, c, temperature, top_k, top_p, min_p, uint64_t(seed),
                               uint32_t(s * rows), b, scores, stream);
        }
        if (err == cudaSuccess) err = cudaEventRecord(events[size_t(2 * s + 1)], stream);
        if (err == cudaSuccess) {
            err = cudaMemcpy2DAsync(out_ids + s * rows, sizeof(uint32_t), &scores[0].id, sizeof(cere_token_score),
                                    sizeof(uint32_t), size_t(rows), cudaMemcpyDeviceToDevice, stream);
        }
    }
    if (status == CERE_OK && err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (status == CERE_OK) status = status_of(err);
    for (int64_t s = 0; s < steps && status == CERE_OK; ++s) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, events[size_t(2 * s)], events[size_t(2 * s + 1)]) != cudaSuccess) {
            status = CERE_ERR_INTERNAL;
        } else {
            static_cast<float *>(step_us)[s] = ms * 1000.0f;
        }
    }
    for (cudaEvent_t ev : events) {
        if (cudaEventDestroy(ev) != cudaSuccess && status == CERE_OK) status = CERE_ERR_INTERNAL;
    }
    return scratch.release(status);
}

CERE_KTEST int cere_ktest_sample_ref_keep_mask(const void *probs, int64_t vocab, int64_t top_k, float top_p,
                                               float min_p, void *mask, cudaStream_t stream) {
    if (!probs || !mask || !vocab_ok(vocab) || top_k < 1 || top_k > vocab) return CERE_ERR_ARGUMENT;
    try {
        std::vector<float> host(static_cast<size_t>(vocab));
        if (cudaMemcpyAsync(host.data(), probs, host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
        const std::vector<uint32_t> words =
            cere::sampler_ref::keep_mask(host.data(), uint64_t(vocab), uint64_t(top_k), top_p, min_p);
        if (cudaMemcpyAsync(mask, words.data(), words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}
