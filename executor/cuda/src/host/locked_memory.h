// Page-aligned host memory locked in RAM for its whole life (the M2.11 snapshot arena, m211 design note D5): one
// anonymous private mapping, zero-filled, excluded from core dumps (MADV_DONTDUMP) and from fork children
// (MADV_DONTFORK), mlock'd (never pageable, so session state cannot reach swap), and zeroed before it is unlocked and
// unmapped. It is plain process memory, never device-mapped and never registered with CUDA, so it costs no GPU
// memory; the CUDA device moves bytes between it and the GPU through its pinned bounce ring. mlock counts against
// RLIMIT_MEMLOCK unless the process holds CAP_IPC_LOCK. No CUDA include.
#pragma once

#include "cere_executor.h"

#include <cstdint>
#include <string>

namespace cere::host {

class LockedMemory {
public:
    LockedMemory() = default;
    ~LockedMemory();
    LockedMemory(const LockedMemory &) = delete;
    LockedMemory &operator=(const LockedMemory &) = delete;

    // Takes `bytes` zero bytes (the mapping rounded up to whole pages) into an empty object; 0 takes nothing (data()
    // stays null). CERE_ERR_CAPACITY when the mapping, either madvise or the mlock fails, with *error (if non-null)
    // naming the failed call and, for mlock, the RLIMIT_MEMLOCK soft limit and the bytes it must allow; there is no
    // pageable fallback and nothing is kept. CERE_ERR_INTERNAL when the object already holds memory.
    cere_status allocate(uint64_t bytes, std::string *error);

    uint8_t *data() const { return data_; }
    uint64_t bytes() const { return bytes_; }
    // The bytes mlock'd (bytes() rounded up to whole pages): what VmLck grows by while the object holds memory.
    uint64_t locked_bytes() const { return mapped_; }

private:
    uint8_t *data_ = nullptr;
    uint64_t bytes_ = 0;   // as requested
    uint64_t mapped_ = 0;  // bytes_ rounded up to whole pages
};

}  // namespace cere::host
