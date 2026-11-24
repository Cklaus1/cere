#include "host/host_share.h"

#include "host/sha256.h"
#include "host/sha256_ni.h"

#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <new>
#include <set>
#include <system_error>
#include <thread>

namespace cere::host {

namespace {

constexpr char kMagic[8] = {'C', 'E', 'R', 'E', 'H', 'S', 'H', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kMaxThreads = 32;
constexpr char kPrefix[] = "cere-host-";
// L9: populate calls of 2 MiB (each holds the process's mmap lock for reading: a longer call stalls every mmap,
// munmap and mprotect of the upload, load-crit design note LC-05).
constexpr uint64_t kPopulateStep = uint64_t(2) << 20;
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23  // Linux 5.14
#endif

// Header fields at fixed offsets (little-endian, the host's order on x86_64).
struct Header {
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t region_bytes;
    uint64_t chunk_bytes;
    uint8_t name_digest[32];
    uint8_t content_digest[32];
};
static_assert(sizeof(Header) <= SharedSegment::kHeaderBytes, "the header fits its page");

// Names this process is creating (an open of one of them from the same process would wait on its own lock).
std::mutex g_creating_mutex;
std::set<std::string> g_creating;

std::string os_error(int code) { return std::strerror(code); }

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

bool lower_hex(const std::string &s) {
    return std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

std::string hex(const std::array<uint8_t, 32> &d) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t b : d) {
        out.push_back(digits[b >> 4]);
        out.push_back(digits[b & 15]);
    }
    return out;
}

std::array<uint8_t, 32> unhex(const std::string &s) {
    std::array<uint8_t, 32> out{};
    auto nibble = [](char c) { return uint8_t(c <= '9' ? c - '0' : c - 'a' + 10); };
    for (size_t i = 0; i < 32 && 2 * i + 1 < s.size(); ++i) out[i] = uint8_t(nibble(s[2 * i]) << 4 | nibble(s[2 * i + 1]));
    return out;
}

int lock(int fd, int op) {
    int r = 0;
    do {
        r = flock(fd, op);
    } while (r != 0 && errno == EINTR);
    return r;
}

bool full_pread(int fd, void *buf, size_t len, off_t at) {
    auto *p = static_cast<uint8_t *>(buf);
    while (len > 0) {
        const ssize_t n = pread(fd, p, len, at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        len -= size_t(n);
        at += n;
    }
    return true;
}

bool full_pwrite(int fd, const void *buf, size_t len, off_t at) {
    const auto *p = static_cast<const uint8_t *>(buf);
    while (len > 0) {
        const ssize_t n = pwrite(fd, p, len, at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        len -= size_t(n);
        at += n;
    }
    return true;
}

size_t hash_threads(uint64_t chunks) {
    size_t cpus = 1;
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) cpus = size_t(std::max(1, CPU_COUNT(&set)));
    return size_t(std::min<uint64_t>({uint64_t(cpus), uint64_t(kMaxThreads), std::max<uint64_t>(chunks, 1)}));
}

// Runs `fn(i)` for every i in [0, items) on `n` threads (i strided by n); when a thread cannot start, the calling
// thread takes its items.
template <typename F>
void on_threads(size_t n, uint64_t items, F &&fn) {
    auto run = [&](uint64_t first) {
        for (uint64_t i = first; i < items; i += n) fn(i);
    };
    std::vector<std::thread> threads;
    size_t started = 0;
    try {
        threads.reserve(n);
        for (size_t t = 1; t < n; ++t) {
            threads.emplace_back(run, t);
            ++started;
        }
    } catch (const std::exception &) {
        // Fewer threads than planned: the calling thread takes the items of every thread that did not start.
    }
    run(0);
    for (size_t t = started + 1; t < n; ++t) run(t);
    for (std::thread &t : threads) t.join();
}

// sha256 of chunk `c` of a region of `bytes` at `data` (SHA-NI when the CPU has it).
std::array<uint8_t, 32> chunk_digest(const uint8_t *data, uint64_t bytes, uint64_t c) {
    const uint64_t at = c * SharedSegment::kChunkBytes;
    Sha256Ni h;
    h.update(data + at, size_t(std::min(SharedSegment::kChunkBytes, bytes - at)));
    return h.finish();
}

std::array<uint8_t, 32> digest_of_sums(const std::vector<std::array<uint8_t, 32>> &sums) {
    Sha256 h;
    for (const auto &s : sums) h.update(s.data(), s.size());
    return h.finish();
}

}  // namespace

int parse_share_option(const std::string &key, const std::string &value, ShareConfig *cfg) {
    if (key == "host_share") {
        if (!cfg->dir.empty() || value.size() < 2 || value[0] != '/' || value.find(',') != std::string::npos) return -1;
        cfg->dir = value;
        return 1;
    }
    if (key == "artifact") {
        if (!cfg->artifact.empty() || value.size() != 64 || !lower_hex(value)) return -1;
        cfg->artifact = value;
        return 1;
    }
    if (key == "host_share_keep") {
        if (cfg->keep || value != "1") return -1;
        cfg->keep = true;
        return 1;
    }
    return 0;
}

bool share_config_complete(const ShareConfig &cfg) {
    return cfg.dir.empty() == cfg.artifact.empty() && (!cfg.keep || !cfg.dir.empty());
}

std::string region_description(const ModelPlan &plan, const std::vector<uint64_t> &offsets, uint64_t region_bytes) {
    std::string d;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        const TensorPlan &t = plan.tensors[i];
        if (t.placement != CERE_PLACEMENT_HOST_PINNED) continue;
        d += t.name;
        d.push_back('\0');
        d += std::to_string(t.dtype);
        for (uint32_t k = 0; k < t.ndim; ++k) d += "," + std::to_string(t.shape[k]);
        d += "@" + std::to_string(offsets[i]) + "\n";
    }
    d += "region " + std::to_string(region_bytes);
    return d;
}

std::string segment_name(const ShareConfig &cfg, const std::string &layout_tag, const std::string &description) {
    Sha256 h;
    const std::string key = std::string("cere-host-share-v1") + '\0' + layout_tag + '\0' + cfg.artifact + '\0';
    h.update(key.data(), key.size());
    h.update(description.data(), description.size());
    return kPrefix + hex(h.finish());
}

bool segment_published(const ShareConfig &cfg, const std::string &name) {
    struct stat st{};
    return stat((cfg.dir + "/" + name).c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::array<uint8_t, 32> region_digest(const uint8_t *data, uint64_t bytes) {
    const uint64_t chunks = (bytes + SharedSegment::kChunkBytes - 1) / SharedSegment::kChunkBytes;
    std::vector<std::array<uint8_t, 32>> sums(chunks);  // std::bad_alloc: never a made-up digest
    on_threads(hash_threads(chunks), chunks, [&](uint64_t c) { sums[c] = chunk_digest(data, bytes, c); });
    return digest_of_sums(sums);
}

cere_status SharedSegment::open(const ShareConfig &cfg, const std::string &name, uint64_t region_bytes,
                                bool writable_attach, const RegionRanges &fill, std::unique_ptr<SharedSegment> *out,
                                std::string *error) {
    try {
        if (region_bytes == 0 || name.size() != sizeof(kPrefix) - 1 + 64) {
            set_error(error, "shared host segment: empty region or malformed name");
            return CERE_ERR_INTERNAL;
        }
        const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
        if (region_bytes > UINT64_MAX - kHeaderBytes - page) {
            set_error(error, "shared host segment: region size overflows");
            return CERE_ERR_CAPACITY;
        }
        std::unique_ptr<SharedSegment> s(new SharedSegment());
        s->path_ = cfg.dir + "/" + name;
        s->lock_path_ = s->path_ + ".lock";
        s->tmp_path_ = s->path_ + ".tmp";
        s->keep_ = cfg.keep;
        s->name_digest_ = unhex(name.substr(sizeof(kPrefix) - 1));
        s->bytes_ = region_bytes;
        s->mapped_ = (region_bytes + page - 1) / page * page;
        // L9: the fill bytes of each chunk (a creator's write that completes them hashes the chunk).
        RegionRanges sorted = fill;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size(); ++i) {
            const auto [at, len] = sorted[i];
            if (at > region_bytes || len > region_bytes - at ||
                (i > 0 && sorted[i - 1].first + sorted[i - 1].second > at)) {
                set_error(error, "shared host segment: fill ranges overlap or leave the region");
                return CERE_ERR_INTERNAL;
            }
        }
        const uint64_t chunks = (region_bytes + kChunkBytes - 1) / kChunkBytes;
        s->expect_.assign(chunks, 0);
        for (const auto &[at, len] : sorted) {
            for (uint64_t from = at, end = at + len; from < end;) {
                const uint64_t c = from / kChunkBytes;
                const uint64_t to = std::min(end, (c + 1) * kChunkBytes);
                s->expect_[c] += to - from;
                from = to;
            }
        }
        {
            std::lock_guard<std::mutex> guard(g_creating_mutex);
            if (g_creating.count(s->path_) != 0) {
                set_error(error, s->path_ + ": this process is still creating it (the model that creates it is not "
                                            "sealed)");
                return CERE_ERR_BUSY;
            }
        }
        s->lock_fd_ = ::open(s->lock_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (s->lock_fd_ < 0) {
            set_error(error, s->lock_path_ + ": open failed: " + os_error(errno));
            return CERE_ERR_INTERNAL;
        }
        if (lock(s->lock_fd_, LOCK_EX) != 0) {
            set_error(error, s->lock_path_ + ": flock failed: " + os_error(errno));
            return CERE_ERR_INTERNAL;
        }
        const off_t file_bytes = off_t(kHeaderBytes + s->mapped_);
        s->fd_ = ::open(s->path_.c_str(), (writable_attach ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW);
        if (s->fd_ >= 0) {
            // Attach: header, size, mapping, content digest.
            struct stat st{};
            Header h{};
            if (fstat(s->fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size != file_bytes ||
                !full_pread(s->fd_, &h, sizeof h, 0)) {
                set_error(error, s->path_ + ": not a segment of " + std::to_string(file_bytes) + " B");
                return CERE_ERR_INTERNAL;
            }
            if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.version != kVersion ||
                h.region_bytes != region_bytes || h.chunk_bytes != kChunkBytes ||
                std::memcmp(h.name_digest, s->name_digest_.data(), 32) != 0) {
                set_error(error, s->path_ + ": header differs from this model's segment");
                return CERE_ERR_INTERNAL;
            }
            const int prot = writable_attach ? PROT_READ | PROT_WRITE : PROT_READ;
            void *p = mmap(nullptr, s->mapped_, prot, MAP_SHARED, s->fd_, off_t(kHeaderBytes));
            if (p == MAP_FAILED) {
                set_error(error, s->path_ + ": mmap failed: " + os_error(errno));
                return CERE_ERR_INTERNAL;
            }
            s->data_ = static_cast<uint8_t *>(p);
            s->writable_ = writable_attach;
            madvise(p, s->mapped_, MADV_DONTDUMP);
            // m216 design note L4c: the content digest is checked by `verify` (before INSTANTIATE publishes or
            // seals anything), so the hashing overlaps UPLOAD. Until then this process holds LOCK_SH (a concurrent
            // last detach never unlinks the file under it) but is not a mapper: a failed verify never unlinks.
            std::memcpy(s->content_digest_.data(), h.content_digest, 32);
            if (lock(s->fd_, LOCK_SH) != 0) {
                set_error(error, s->path_ + ": flock failed: " + os_error(errno));
                return CERE_ERR_INTERNAL;
            }
            close(s->lock_fd_);
            s->lock_fd_ = -1;
            *out = std::move(s);
            return CERE_OK;
        }
        if (errno != ENOENT) {
            set_error(error, s->path_ + ": open failed: " + os_error(errno));
            return CERE_ERR_INTERNAL;
        }
        // Create: a .tmp left by a dead creator is truncated (its lock is free, so nobody fills it).
        const std::string &tmp = s->tmp_path_;
        s->fd_ = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (s->fd_ < 0) {
            set_error(error, tmp + ": open failed: " + os_error(errno));
            return CERE_ERR_INTERNAL;
        }
        s->creating_ = true;
        {
            std::lock_guard<std::mutex> guard(g_creating_mutex);
            g_creating.insert(s->path_);
        }
        // L9: no posix_fallocate (it zeroed the whole file on one thread under the inode lock, ~4 s for 28.8 GB):
        // the free space is checked here, the pages are allocated by `populate` and the writes. f_blocks 0 is a
        // filesystem without a size limit.
        struct statvfs vfs{};
        if (fstatvfs(s->fd_, &vfs) != 0) {
            set_error(error, tmp + ": fstatvfs failed: " + os_error(errno));
            return CERE_ERR_INTERNAL;
        }
        const uint64_t free_bytes = uint64_t(vfs.f_bavail) * uint64_t(vfs.f_frsize);
        if (vfs.f_blocks != 0 && free_bytes < uint64_t(file_bytes)) {
            set_error(error, tmp + ": " + std::to_string(free_bytes) + " B free, the segment needs " +
                                 std::to_string(file_bytes) + " B");
            return CERE_ERR_CAPACITY;
        }
        if (ftruncate(s->fd_, file_bytes) != 0) {
            const int code = errno;
            set_error(error, tmp + ": ftruncate failed: " + os_error(code));
            return code == EFBIG || code == ENOSPC ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
        }
        void *p = mmap(nullptr, s->mapped_, PROT_READ | PROT_WRITE, MAP_SHARED, s->fd_, off_t(kHeaderBytes));
        if (p == MAP_FAILED) {
            set_error(error, tmp + ": mmap failed: " + os_error(errno));
            return CERE_ERR_INTERNAL;
        }
        s->data_ = static_cast<uint8_t *>(p);
        madvise(p, s->mapped_, MADV_DONTDUMP);
        s->filled_ = std::make_unique<std::atomic<uint64_t>[]>(chunks);
        s->summed_ = std::make_unique<std::atomic<bool>[]>(chunks);
        s->sums_.resize(chunks);
        if (lock(s->fd_, LOCK_SH) != 0) {
            set_error(error, tmp + ": flock failed: " + os_error(errno));
            return CERE_ERR_INTERNAL;
        }
        *out = std::move(s);
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        set_error(error, "shared host segment: out of memory");
        return CERE_ERR_CAPACITY;
    }
}

SharedSegment::~SharedSegment() {
    if (data_ != nullptr) munmap(data_, mapped_);
    if (creating_) {
        // Never published: the lock is still held, nobody else maps the .tmp.
        if (fd_ >= 0) unlink(tmp_path_.c_str());
        std::lock_guard<std::mutex> guard(g_creating_mutex);
        g_creating.erase(path_);
    } else if (mapper_ && !keep_) {
        // m216 design note L4: with host_share_keep the published segment outlives its last mapper (removed by the
        // operator or the last detach of a lane without the option).
        if (lock_fd_ < 0) lock_fd_ = ::open(lock_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (lock_fd_ >= 0 && lock(lock_fd_, LOCK_EX) == 0 && lock(fd_, LOCK_EX | LOCK_NB) == 0) {
            // No other mapper holds LOCK_SH: the last detach unlinks, if the name still refers to this file.
            struct stat mine{}, named{};
            if (fstat(fd_, &mine) == 0 && stat(path_.c_str(), &named) == 0 && mine.st_dev == named.st_dev &&
                mine.st_ino == named.st_ino) {
                unlink(path_.c_str());
            }
        }
    }
    if (fd_ >= 0) close(fd_);
    if (lock_fd_ >= 0) close(lock_fd_);
}

int SharedSegment::populate_range(uint64_t offset, uint64_t len) const {
    const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
    const uint64_t from = offset / page * page;
    const uint64_t to = std::min(mapped_, (offset + len + page - 1) / page * page);
    if (to <= from) return 0;
    int r = 0;
    do {
        r = madvise(data_ + from, to - from, MADV_POPULATE_WRITE);
    } while (r != 0 && errno == EINTR);
    return r == 0 ? 0 : errno;
}

cere_status SharedSegment::populate_failure(int code, uint64_t offset, uint64_t len, std::string *error) const {
    // EFAULT: the fault would have raised SIGBUS (the filesystem is full); ENOMEM: no memory for the pages.
    const bool full = code == EFAULT || code == ENOMEM || code == ENOSPC;
    set_error(error, tmp_path_ + ": MADV_POPULATE_WRITE of region bytes [" + std::to_string(offset) + ", " +
                         std::to_string(offset + len) + ") failed: " + os_error(code) +
                         (full ? " (the filesystem is full)"
                               : code == EINVAL ? " (the kernel lacks MADV_POPULATE_WRITE, Linux 5.14)" : ""));
    return full ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
}

void SharedSegment::hash_chunk(uint64_t c) {
    const uint64_t at = c * kChunkBytes;
    // The chunk's unfilled bytes (alignment gaps) are read too: their pages are allocated first, so a full
    // filesystem never SIGBUSes the read (the chunk is left to publish, which reports it).
    if (!populated_.load(std::memory_order_acquire) && populate_range(at, std::min(kChunkBytes, bytes_ - at)) != 0) {
        return;
    }
    sums_[c] = chunk_digest(data_, bytes_, c);
    summed_[c].store(true, std::memory_order_release);
    hashed_.fetch_add(1);
}

cere_status SharedSegment::write(uint64_t offset, const void *src, uint64_t len, std::string *error) {
    if (offset > bytes_ || len > bytes_ - offset) {
        set_error(error, path_ + ": write outside the region");
        return CERE_ERR_ARGUMENT;
    }
    if (creating_) {
        if (!populated_.load(std::memory_order_acquire)) {
            if (const int code = populate_range(offset, len); code != 0) {
                return populate_failure(code, offset, len, error);
            }
        }
        std::memcpy(data_ + offset, src, len);
        // L9: the write that completes a chunk's fill bytes hashes it (acq_rel: every other write's bytes of the
        // chunk happen before the hash).
        for (uint64_t at = offset, end = offset + len; at < end;) {
            const uint64_t c = at / kChunkBytes;
            const uint64_t to = std::min(end, (c + 1) * kChunkBytes);
            if (filled_[c].fetch_add(to - at, std::memory_order_acq_rel) + (to - at) == expect_[c]) hash_chunk(c);
            at = to;
        }
        return CERE_OK;
    }
    if (std::memcmp(data_ + offset, src, len) != 0) {
        set_error(error, path_ + ": the put bytes at region offset " + std::to_string(offset) + " (" +
                             std::to_string(len) + " B) differ from the shared segment");
        return CERE_ERR_INTERNAL;
    }
    return CERE_OK;
}

cere_status SharedSegment::populate(size_t threads, std::string *error) {
    if (!creating_ || populated_.load()) return CERE_OK;
    const uint64_t steps = (mapped_ + kPopulateStep - 1) / kPopulateStep;
    std::atomic<int> failed{0};
    std::atomic<uint64_t> failed_at{0};
    on_threads(std::min(std::max<size_t>(threads, 1), hash_threads(steps)), steps, [&](uint64_t i) {
        if (failed.load(std::memory_order_relaxed) != 0) return;
        const uint64_t at = i * kPopulateStep;
        if (const int code = populate_range(at, std::min(kPopulateStep, mapped_ - at)); code != 0) {
            int none = 0;
            if (failed.compare_exchange_strong(none, code)) failed_at.store(at);
        }
    });
    if (const int code = failed.load(); code != 0) {
        const uint64_t at = failed_at.load();
        return populate_failure(code, at, std::min(kPopulateStep, mapped_ - at), error);
    }
    populated_.store(true, std::memory_order_release);
    return CERE_OK;
}

cere_status SharedSegment::publish(std::string *error) {
    if (!creating_) return CERE_OK;
    const std::string &tmp = tmp_path_;
    Header h{};
    std::memcpy(h.magic, kMagic, sizeof kMagic);
    h.version = kVersion;
    h.region_bytes = bytes_;
    h.chunk_bytes = kChunkBytes;
    std::memcpy(h.name_digest, name_digest_.data(), 32);
    // Every page is allocated before the unfilled bytes are read (a no-op after `populate`).
    if (!populated_.load(std::memory_order_acquire)) {
        if (const int code = populate_range(0, mapped_); code != 0) return populate_failure(code, 0, mapped_, error);
        populated_.store(true, std::memory_order_release);
    }
    std::array<uint8_t, 32> digest;
    try {
        // L9: the chunks the writes did not hash (no fill bytes, or filled other than `fill` says) are hashed now.
        std::vector<uint64_t> missing;
        for (uint64_t c = 0; c < expect_.size(); ++c) {
            if (!summed_[c].load(std::memory_order_acquire) || filled_[c].load() != expect_[c]) missing.push_back(c);
        }
        on_threads(hash_threads(missing.size()), missing.size(),
                   [&](uint64_t i) { sums_[missing[i]] = chunk_digest(data_, bytes_, missing[i]); });
        digest = digest_of_sums(sums_);
    } catch (const std::bad_alloc &) {
        // DM-6: no digest is published (a zero one would make every attach fail its verification).
        set_error(error, tmp + ": no memory for the content digest of " + std::to_string(bytes_) + " B");
        return CERE_ERR_CAPACITY;
    }
    std::memcpy(h.content_digest, digest.data(), 32);
    if (!full_pwrite(fd_, &h, sizeof h, 0)) {
        set_error(error, tmp + ": header write failed: " + os_error(errno));
        return CERE_ERR_INTERNAL;
    }
    if (mprotect(data_, mapped_, PROT_READ) != 0 || fchmod(fd_, 0400) != 0) {
        set_error(error, tmp + ": mprotect/fchmod failed: " + os_error(errno));
        return CERE_ERR_INTERNAL;
    }
    if (rename(tmp.c_str(), path_.c_str()) != 0) {
        set_error(error, tmp + ": rename to " + path_ + " failed: " + os_error(errno));
        return CERE_ERR_INTERNAL;
    }
    creating_ = false;
    mapper_ = true;
    {
        std::lock_guard<std::mutex> guard(g_creating_mutex);
        g_creating.erase(path_);
    }
    close(lock_fd_);
    lock_fd_ = -1;
    return CERE_OK;
}

cere_status SharedSegment::protect(std::string *error) {
    if (!writable_) return CERE_OK;
    if (mprotect(data_, mapped_, PROT_READ) != 0) {
        set_error(error, path_ + ": mprotect failed: " + os_error(errno));
        return CERE_ERR_INTERNAL;
    }
    writable_ = false;
    return CERE_OK;
}

cere_status SharedSegment::verify(std::string *error) {
    if (creating_ || mapper_) return CERE_OK;
    std::array<uint8_t, 32> digest;
    try {
        digest = region_digest(data_, bytes_);
    } catch (const std::bad_alloc &) {
        set_error(error, path_ + ": no memory to verify the content digest of " + std::to_string(bytes_) + " B");
        return CERE_ERR_CAPACITY;  // not a mismatch: the segment is not judged tampered
    }
    if (std::memcmp(digest.data(), content_digest_.data(), 32) != 0) {
        set_error(error, path_ + ": content digest differs from the recorded one (tampered or corrupt; the segment is "
                                 "left in place)");
        return CERE_ERR_INTERNAL;
    }
    mapper_ = true;
    return CERE_OK;
}

}  // namespace cere::host
