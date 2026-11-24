// SHA-256 with the x86 SHA extensions (SHA-NI) when the CPU has them, else cere::host::Sha256, the scalar path. Both
// give the FIPS 180-4 digest. Users: the ktest state digests (cere_ktest_state_digest, cere_ktest_draft_digest, about
// 75 MB of session state per call after every speculation round of test_model_k8_forward.py, while the GPU idles) and
// the O47 segment's per-chunk content digests (host_share.cpp region_digest and the creator's hashing during UPLOAD,
// beat-sglang design note L9). The scalar path runs at ~0.4 GB/s on the dev host and SHA-NI at ~2 GB/s.
#pragma once

#include "host/sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace cere::host {

// The CPU has SHA-NI, SSSE3 and SSE4.1 (CPUID).
bool cpu_has_sha_ni();
// The path a default-constructed Sha256Ni takes, decided once per process: SHA-NI when the CPU has it, unless the
// environment sets CERE_SHA256_FORCE_SCALAR=1 (test hook).
bool sha_ni_selected();
// The selection rule of sha_ni_selected for an explicit CERE_SHA256_FORCE_SCALAR value (nullptr: unset).
bool sha_ni_wanted(bool cpu, const char *force_scalar);

class Sha256Ni {
public:
    Sha256Ni() : Sha256Ni(sha_ni_selected()) {}
    // `ni` true needs cpu_has_sha_ni().
    explicit Sha256Ni(bool ni);
    void update(const void *data, size_t len);
    // The digest of every byte passed to update; the object is spent afterwards.
    std::array<uint8_t, 32> finish();
    bool ni() const { return ni_; }

private:
    bool ni_;
    cere::host::Sha256 scalar_;
    uint32_t h_[8];
    uint8_t buffer_[64];
    size_t buffered_ = 0;
    uint64_t total_ = 0;  // bytes
};

}  // namespace cere::host
