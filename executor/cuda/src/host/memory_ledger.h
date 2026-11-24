// The executor's own memory ledger (m211 design note R1 [REVISED 2026-10-11]): the exact bytes a device holds from
// its own allocations, by kind, so R1 can compare a build's memory byte-exact instead of through process-external
// counters (cudaMemGetInfo, MemAvailable), which move by up to ±0.85 MiB with no change in the executor. A device
// records an allocation right after it succeeds and drops it when the allocation is freed (LedgerEntry); nothing
// else writes the ledger. Thread-safe; no CUDA include.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace cere::host {

enum class Held : size_t {
    kDevice = 0,  // device memory the executor allocates (CUDA: the ALLOCATE cudaMalloc, its only one)
    kRing = 1,    // pinned host bounce ring (CUDA: cudaHostAlloc at executor creation)
    kMapped = 2,  // pinned, device-mapped host block of a model (CUDA: cudaHostAlloc(cudaHostAllocMapped))
    kLocked = 3,  // mlock'd host memory (host::LockedMemory, the M2.11 snapshot arena; whole pages)
    kStaging = 4,  // caller memory registered as put staging (CUDA: cudaHostRegister; m216 M216S-11)
};
constexpr size_t kHeldKinds = 5;

class MemoryLedger {
public:
    void add(Held kind, uint64_t bytes) { held_[size_t(kind)].fetch_add(bytes, std::memory_order_relaxed); }
    void sub(Held kind, uint64_t bytes) { held_[size_t(kind)].fetch_sub(bytes, std::memory_order_relaxed); }
    uint64_t held(Held kind) const { return held_[size_t(kind)].load(std::memory_order_relaxed); }

private:
    std::atomic<uint64_t> held_[kHeldKinds] = {};
};

// One recorded allocation: `hold` adds its bytes to the ledger, `drop` (or the destructor) subtracts them once.
class LedgerEntry {
public:
    LedgerEntry() = default;
    ~LedgerEntry() { drop(); }
    LedgerEntry(const LedgerEntry &) = delete;
    LedgerEntry &operator=(const LedgerEntry &) = delete;

    // Records `bytes` of `kind` in `ledger`; an entry that already holds drops its bytes first.
    void hold(MemoryLedger &ledger, Held kind, uint64_t bytes) {
        drop();
        ledger.add(kind, bytes);
        ledger_ = &ledger;
        kind_ = kind;
        bytes_ = bytes;
    }
    void drop() {
        if (ledger_ != nullptr) ledger_->sub(kind_, bytes_);
        ledger_ = nullptr;
        bytes_ = 0;
    }
    uint64_t bytes() const { return bytes_; }

private:
    MemoryLedger *ledger_ = nullptr;
    Held kind_ = Held::kDevice;
    uint64_t bytes_ = 0;
};

}  // namespace cere::host
