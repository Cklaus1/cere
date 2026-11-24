// Shared HOST_PINNED segment (build-loop-ple design note, opportunity O47): the read-only host tensors of a model (the PLE
// table, scale and index) live in one file per artifact and layout under a host directory (the lanes mount the same
// /dev/shm path), mapped MAP_SHARED by every native process that loads that artifact, so two lanes hold one copy. The
// device pins the mapping itself (CUDA: cudaHostRegister); this file has no CUDA include. The toy device uses it too.
//
// device_id options (CUDA "<ordinal>,...", toy "toy,..."): "host_share=<absolute dir without commas>" and
// "artifact=<the artifact manifest sha256, 64 lowercase hex>", both or neither; without them the device keeps its
// private copy, with them every share failure fails closed (no private fallback). "host_share_keep=1" (only with
// both; m216 design note L4): this process never unlinks the published segment at detach, so a restart attaches.
//
// Segment `<dir>/cere-host-<hex>`: <hex> = sha256 of the device's layout tag, the artifact and the region description
// (every HOST_PINNED tensor's name, dtype, shape and offset, then the region bytes). File = a 4,096-byte header (magic
// "CEREHSH1", version 1, region bytes, chunk bytes, the name digest, the content digest), then the region rounded up
// to whole pages. Content digest = sha256 of the concatenated sha256 of each 64 MiB chunk of the region (SHA-NI when
// the CPU has it, host/sha256_ni.h; the chunks are hashed in parallel). `<dir>/cere-host-<hex>.lock` (never
// unlinked) serializes create, attach and detach.
//
// open: under flock(.lock, LOCK_EX) either attaches (the published file exists: header checked, region mapped
// PROT_READ, or with `writable_attach` O_RDWR + PROT_READ|PROT_WRITE until `protect`, flock(<name>, LOCK_SH) taken)
// or creates `<name>.tmp` (O_TRUNC, mode 0600, ftruncate, mapped PROT_READ|PROT_WRITE; beat-sglang design note L9:
// no zeroing at ALLOCATE, the filesystem's free space (fstatvfs) short of the file is CERE_ERR_CAPACITY) and keeps
// the lock until publish, so a concurrent open of the same name waits. The creator's pages are allocated by
// `populate` (the device's background pinner) and by each `write`, which populates its own range first
// (MADV_POPULATE_WRITE: a filesystem that filled up since open is CERE_ERR_CAPACITY, never a SIGBUS). `verify` (an
// attach, m216 design note L4c: off the lock, overlapping UPLOAD, before the model is sealed) recomputes the
// content digest and compares it; any difference is CERE_ERR_INTERNAL and the file stays for the operator; a match
// makes the process a mapper. `write` copies (creator before publish) or compares (otherwise; a difference is
// CERE_ERR_INTERNAL); the creator's write that completes a chunk (every byte of the `fill` ranges inside it put)
// hashes that chunk (L9). `publish` (the creator at INSTANTIATE): the content digest from the chunk digests (a chunk
// not yet hashed is hashed then), header, mprotect PROT_READ, fchmod 0400, rename to <name>, unlock; the creator is
// then a mapper. The destructor (detach; the device unpins first) unmaps and, under the lock, unlinks an unpublished
// .tmp, or (a mapper without keep) the published file when flock(LOCK_EX|LOCK_NB) shows no other process holding
// LOCK_SH. A killed process leaves a published file with no mapper; the next attach reuses it only when its digest
// verifies.
// Errors carry a message naming the path and the failed call. Never throws.
#pragma once

#include "cere_executor.h"
#include "host/device.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cere::host {

struct ShareConfig {
    std::string dir;       // empty: no sharing
    std::string artifact;  // 64 lowercase hex
    bool keep = false;     // host_share_keep=1
    bool enabled() const { return !dir.empty(); }
};

// One device_id option: 1 when `key` is a share option and `value` is valid and not a repeat (stored in *cfg), -1
// when it is a share option that is malformed or repeated, 0 when `key` is not a share option.
int parse_share_option(const std::string &key, const std::string &value, ShareConfig *cfg);
// Both share options or neither; host_share_keep only with them.
bool share_config_complete(const ShareConfig &cfg);

// The region description of the segment name: every HOST_PINNED tensor of `plan` in desc order (name, dtype, shape,
// offsets[i]), then `region_bytes`.
std::string region_description(const ModelPlan &plan, const std::vector<uint64_t> &offsets, uint64_t region_bytes);
// "cere-host-" + hex sha256 of the key (layout tag, artifact, region description).
std::string segment_name(const ShareConfig &cfg, const std::string &layout_tag, const std::string &description);
// True when the published segment `name` exists in cfg.dir (the host estimate leaves its region out).
bool segment_published(const ShareConfig &cfg, const std::string &name);

// Content digest of `bytes` (see above), hashed on up to 32 threads. Throws std::bad_alloc when the per-chunk digests
// cannot be held (never a made-up digest: SharedSegment::publish and verify turn it into CERE_ERR_CAPACITY).
std::array<uint8_t, 32> region_digest(const uint8_t *data, uint64_t bytes);

// [offset, offset + bytes) of the region: the HOST_PINNED tensors a creator's puts fill, each byte once.
using RegionRanges = std::vector<std::pair<uint64_t, uint64_t>>;

class SharedSegment {
public:
    static constexpr uint64_t kHeaderBytes = 4096;
    static constexpr uint64_t kChunkBytes = uint64_t(64) << 20;

    // Attaches to or creates the segment `name` of `region_bytes` (> 0) bytes (see above). *error is set on failure.
    // An attach checks the header and maps the region; its content digest is checked by `verify` (m216 L4c), which
    // must pass before a model built on the region is sealed. `writable_attach`: an attach maps
    // the region writable so a device without read-only host registration can pin it (CUDA rw_mprotect path); the
    // caller pins it, then calls `protect`. Creating maps writable either way. `fill`: the ranges a creator's writes
    // fill (inside the region, disjoint; CERE_ERR_INTERNAL otherwise); a chunk is hashed by the write that completes
    // its bytes of them. An attach ignores it.
    static cere_status open(const ShareConfig &cfg, const std::string &name, uint64_t region_bytes,
                            bool writable_attach, const RegionRanges &fill, std::unique_ptr<SharedSegment> *out,
                            std::string *error);
    ~SharedSegment();
    SharedSegment(const SharedSegment &) = delete;
    SharedSegment &operator=(const SharedSegment &) = delete;

    uint8_t *data() const { return data_; }
    uint64_t bytes() const { return bytes_; }
    uint64_t mapped_bytes() const { return mapped_; }
    // True for the creator until publish: the mapping is writable. Otherwise it is PROT_READ.
    bool creating() const { return creating_; }
    const std::string &path() const { return path_; }

    // The creator's chunk count and how many chunks are hashed so far (tests, the seal's log line).
    uint64_t chunks() const { return expect_.size(); }
    uint64_t chunks_hashed() const { return hashed_.load(); }

    // UPLOAD of [offset, offset + len) of the region (in range): copy while creating (its pages populated first:
    // CERE_ERR_CAPACITY when the filesystem is full; the write that completes a chunk hashes it), else compare.
    // Concurrent with other writes and with `populate`.
    cere_status write(uint64_t offset, const void *src, uint64_t len, std::string *error);
    // The creator (no-op otherwise): allocates every page of the mapping (MADV_POPULATE_WRITE in 2 MiB calls, which
    // never change a written byte) on `threads` threads. CERE_ERR_CAPACITY when the filesystem is full.
    cere_status populate(size_t threads, std::string *error);
    // INSTANTIATE: publishes a created segment (no-op otherwise).
    cere_status publish(std::string *error);
    // After pinning a writable attach: mprotect PROT_READ (no-op for a creator, whose publish does it, or a PROT_READ
    // attach).
    cere_status protect(std::string *error);
    // An attach: recomputes the content digest (region_digest) and compares it with the header's; on a match this
    // process becomes a mapper (its last detach may unlink). No-op for a creator and an already verified attach.
    // CERE_ERR_INTERNAL on a mismatch (the segment is left in place). Not concurrent with itself.
    cere_status verify(std::string *error);

private:
    SharedSegment() = default;

    std::string path_;       // <dir>/<name>, the published file
    std::string lock_path_;  // <path>.lock
    std::string tmp_path_;   // <path>.tmp (the creator's file until publish)
    std::array<uint8_t, 32> name_digest_{};
    int fd_ = -1;       // the segment (.tmp while creating), holding LOCK_SH
    int lock_fd_ = -1;  // the lock file, held LOCK_EX while creating
    uint8_t *data_ = nullptr;
    uint64_t bytes_ = 0;
    uint64_t mapped_ = 0;
    bool creating_ = false;
    bool mapper_ = false;    // holds the mapper LOCK_SH of a published, verified segment (a failed attach never unlinks)
    bool writable_ = false;  // a writable attach not yet protected
    bool keep_ = false;      // ShareConfig::keep: the detach never unlinks the published segment
    std::array<uint8_t, 32> content_digest_{};  // an attach: the header's content digest, checked by verify
    // The creator's chunks (L9): the fill bytes each holds, the bytes written so far, its digest once hashed.
    std::vector<uint64_t> expect_;
    std::unique_ptr<std::atomic<uint64_t>[]> filled_;
    std::unique_ptr<std::atomic<bool>[]> summed_;
    std::vector<std::array<uint8_t, 32>> sums_;
    std::atomic<uint64_t> hashed_{0};
    std::atomic<bool> populated_{false};  // `populate` succeeded: every page is allocated

    // MADV_POPULATE_WRITE of [offset, offset + len) of the mapping (rounded out to pages): 0 or the errno.
    int populate_range(uint64_t offset, uint64_t len) const;
    // The status and message of a populate that failed with errno `code`.
    cere_status populate_failure(int code, uint64_t offset, uint64_t len, std::string *error) const;
    // Hashes chunk `c` into sums_[c] (once).
    void hash_chunk(uint64_t c);
};

}  // namespace cere::host
