// K13S kernel-test entry points (executor/tests/gpu/test_kernels_spec_accept.py, prototypes in its docstring;
// m29 design note K6, I8 `cere_ktest_spec_accept`). Device pointers, int64 sizes and scalars, the stream last;
// 0 = success (CERE_OK), CERE_ERR_ARGUMENT for arguments outside a kernel's contract, CERE_ERR_CAPACITY when the
// stream-ordered scratch cannot be allocated, CERE_ERR_INTERNAL for a failed launch or free. Contracts in
// kernels/k13s_spec_accept.h.
#include <cstdint>
#include <limits>

#include "kernels/k13s_spec_accept.h"
#include "ktest/k1_ktest.h"

using cere::ktest::status_of;

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

// Stream-ordered device scratch; release() frees it on the same stream and folds a failed free into the status.
class StreamScratch {
  public:
    StreamScratch(size_t bytes, cudaStream_t stream) : stream_(stream) {
        if (cudaMallocAsync(&ptr_, bytes, stream) != cudaSuccess) ptr_ = nullptr;
    }
    StreamScratch(const StreamScratch &) = delete;
    StreamScratch &operator=(const StreamScratch &) = delete;
    int release(int status) {
        const cudaError_t freed = ptr_ != nullptr ? cudaFreeAsync(ptr_, stream_) : cudaSuccess;
        ptr_ = nullptr;
        return status == CERE_OK && freed != cudaSuccess ? CERE_ERR_INTERNAL : status;
    }
    ~StreamScratch() { release(CERE_OK); }
    void *get() const { return ptr_; }

  private:
    void *ptr_ = nullptr;
    cudaStream_t stream_;
};

}  // namespace

// `chains` independent chains on the same rows: chain c uses seed seed + c and key0; committed uint32 [chains, n + 1],
// counts uint32 [chains].
CERE_KTEST int cere_ktest_spec_accept(const void *probs, const void *masks, int64_t vocab, const void *drafts,
                                      int64_t n, int64_t seed, int64_t key0, int64_t chains, void *committed,
                                      void *counts, cudaStream_t stream) {
    if (chains < 1 || chains > std::numeric_limits<int32_t>::max() || key0 < 0 ||
        key0 > int64_t(std::numeric_limits<uint32_t>::max()) || !committed || !counts) {
        return CERE_ERR_ARGUMENT;
    }
    if (n < 0 || n > cere::k13s::kMaxDrafts) return CERE_ERR_ARGUMENT;
    StreamScratch scratch(cere::k13s::scratch_bytes(n), stream);
    if (scratch.get() == nullptr) return CERE_ERR_CAPACITY;
    auto *out = static_cast<uint32_t *>(committed);
    auto *cnt = static_cast<uint32_t *>(counts);
    cudaError_t err = cudaSuccess;
    for (int64_t c = 0; c < chains && err == cudaSuccess; ++c) {
        err = cere::k13s::spec_accept(static_cast<const float *>(probs), static_cast<const uint32_t *>(masks), vocab,
                                      static_cast<const uint32_t *>(drafts), n, uint64_t(seed) + uint64_t(c),
                                      uint32_t(key0), out + c * (n + 1), cnt + c, scratch.get(), stream);
    }
    return scratch.release(status_of(err));
}

// `trials` coins on one row (k13s::row_trials); seeds and keys as int64 bit patterns (uint64 seed, uint32 key).
// ids uint32 [trials], accepted uint8 [trials].
CERE_KTEST int cere_ktest_spec_row_trials(const void *probs, const void *mask, int64_t vocab, int64_t draft,
                                          int64_t first_seed, int64_t seed_step, int64_t key0, int64_t key_step,
                                          int64_t trials, void *ids, void *accepted, cudaStream_t stream) {
    if (draft < 0 || draft > int64_t(std::numeric_limits<uint32_t>::max()) || key0 < 0 ||
        key0 > int64_t(std::numeric_limits<uint32_t>::max()) || key_step < 0 ||
        key_step > int64_t(std::numeric_limits<uint32_t>::max())) {
        return CERE_ERR_ARGUMENT;
    }
    StreamScratch scratch(cere::k13s::trial_scratch_bytes(), stream);
    if (scratch.get() == nullptr) return CERE_ERR_CAPACITY;
    const cudaError_t err = cere::k13s::row_trials(
        static_cast<const float *>(probs), static_cast<const uint32_t *>(mask), vocab, uint32_t(draft),
        uint64_t(first_seed), uint64_t(seed_step), uint32_t(key0), uint32_t(key_step), trials,
        static_cast<uint32_t *>(ids), static_cast<uint8_t *>(accepted), scratch.get(), stream);
    return scratch.release(status_of(err));
}
