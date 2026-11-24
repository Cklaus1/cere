#include "host/locked_memory.h"

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace cere::host {

namespace {

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

std::string os_error(int code) { return std::strerror(code); }

}  // namespace

LockedMemory::~LockedMemory() {
    if (data_ == nullptr) return;
    // Zeroed while still locked, so the bytes never reach swap; munmap is opaque to the compiler, which keeps the
    // stores.
    std::memset(data_, 0, mapped_);
    munlock(data_, mapped_);
    munmap(data_, mapped_);
}

cere_status LockedMemory::allocate(uint64_t bytes, std::string *error) {
    if (data_ != nullptr) return CERE_ERR_INTERNAL;
    if (bytes == 0) return CERE_OK;
    const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
    if (bytes > UINT64_MAX - (page - 1)) {
        set_error(error, "mmap of " + std::to_string(bytes) + " B: size overflows a page multiple");
        return CERE_ERR_CAPACITY;
    }
    const uint64_t mapped = (bytes + page - 1) / page * page;
    // Anonymous private memory is zero-filled by the kernel and page-aligned.
    void *p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        set_error(error, "mmap of " + std::to_string(mapped) + " B failed: " + os_error(errno));
        return CERE_ERR_CAPACITY;
    }
    if (madvise(p, mapped, MADV_DONTDUMP) != 0 || madvise(p, mapped, MADV_DONTFORK) != 0) {
        const int code = errno;
        munmap(p, mapped);
        set_error(error, "madvise(MADV_DONTDUMP/MADV_DONTFORK) of " + std::to_string(mapped) + " B failed: " +
                             os_error(code));
        return CERE_ERR_CAPACITY;
    }
    if (mlock(p, mapped) != 0) {
        const int code = errno;
        munmap(p, mapped);  // nothing was written: the pages are still the kernel's zeros
        rlimit limit{};
        std::string soft = "unknown";
        if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
            soft = limit.rlim_cur == RLIM_INFINITY ? std::string("unlimited") : std::to_string(limit.rlim_cur) + " B";
        }
        set_error(error, "mlock of " + std::to_string(mapped) + " B failed: " + os_error(code) +
                             " (RLIMIT_MEMLOCK soft limit " + soft + "; it must allow at least " +
                             std::to_string(mapped) + " B locked, or the process needs CAP_IPC_LOCK)");
        return CERE_ERR_CAPACITY;
    }
    data_ = static_cast<uint8_t *>(p);
    bytes_ = bytes;
    mapped_ = mapped;
    return CERE_OK;
}

}  // namespace cere::host
