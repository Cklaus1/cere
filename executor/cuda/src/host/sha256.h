// SHA-256 (FIPS 180-4) for the snapshot `layout_id` and the state digest (m211 design note D3, ledger M4): in-tree,
// so the library gains no NEEDED entry (no CUDA includes).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace cere::host {

class Sha256 {
public:
    Sha256();
    void update(const void *data, size_t len);
    // The digest of every byte passed to update; the object is spent afterwards.
    std::array<uint8_t, 32> finish();

private:
    void block(const uint8_t *p);

    uint32_t h_[8];
    uint8_t buffer_[64];
    size_t buffered_ = 0;
    uint64_t total_ = 0;  // bytes
};

std::array<uint8_t, 32> sha256(const void *data, size_t len);

}  // namespace cere::host
