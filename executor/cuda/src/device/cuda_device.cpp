#include "device/cuda_device.h"

#include "device/context_generation.h"
#include "fault/fault_inject.h"
#include "host/lora.h"

#include "kernels/k1_primitives.h"
#include "kernels/k6_attention.h"
#include "model/layout.h"
#include "model/mtp.h"
#include "model/prefill_signature.h"
#include "model/segment.h"
#include "research/hook.h"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <sys/mman.h>

namespace cere::device {

namespace {

constexpr uint64_t kHalfBytes = kSlotBytes / 2;
static_assert(model::segment::kLayers == model::kLayers, "model/segment.h counts the model's layers");
constexpr size_t kMaxOrdinalDigits = 9;

// The live CudaDevices of this process per tracked ordinal (cuda_device.h "Device release").
std::atomic<int> g_live_devices[kTrackedDevices]{};

// DM-2: the ordinals a sticky CUDA error poisoned (CudaDevice::probe_sticky); cleared when the device is released.
std::atomic<bool> g_poisoned[kTrackedDevices]{};

// The CUDA errors that leave the context unusable for good (every later call fails with them until the device is
// reset): the device-side faults, as opposed to the launch / allocation failures fail() clears.
bool is_sticky(cudaError_t e) {
    switch (e) {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorECCUncorrectable:
    case cudaErrorAssert:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorUnknown:
        return true;
    default:
        return false;
    }
}

// Serializes a tracked ordinal's device opens with its device destructions, so a release never resets the context of an
// executor being created on the same card; an empty lock for an untracked ordinal.
std::unique_lock<std::mutex> ordinal_lock(int ordinal) {
    static std::mutex mutexes[kTrackedDevices];
    return ordinal >= 0 && ordinal < kTrackedDevices ? std::unique_lock<std::mutex>(mutexes[ordinal])
                                                     : std::unique_lock<std::mutex>();
}

// CudaDevice::draft_rows's regions past a verify's rows (at most kMaxRows) in ws.ids32 / ws.host_ids32 and ws.hc_rows
// (m23 design note P11R-21): the extends' rows contiguous at kExtendAt (ids and hc), then at kDraftAt the proposals
// [kProposalWords, sessions] (ids) and every session's decode-step hc row (hc).
constexpr uint32_t kProposalWords = 1 + 1 + uint32_t(model::layout::kDraftSteps);  // count, d1, one per decode step
constexpr uint32_t kExtendAt = kernels::GemmPlan::kMaxRows;
constexpr uint32_t kDraftAt = kExtendAt + kernels::GemmPlan::kMaxRows;

// A failed allocation is CERE_ERR_CAPACITY, any other CUDA error CERE_ERR_INTERNAL. The (non-sticky) error is
// cleared so a later call does not report it again.
cere_status fail(cudaError_t error) {
    cudaGetLastError();
    return error == cudaErrorMemoryAllocation ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
}

// The device-id format of cuda_device.h; *ordinal, *share and *release are set on CERE_OK.
cere_status parse_device_id(const char *id, int *ordinal, host::ShareConfig *share, bool *release) {
    const std::string full(id);
    const size_t comma = full.find(',');
    const std::string s = full.substr(0, comma);
    if (s.empty()) return CERE_ERR_ARGUMENT;
    const size_t sign = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    const std::string digits = s.substr(sign);
    const bool decimal = !digits.empty() && std::all_of(digits.begin(), digits.end(),
                                                         [](char c) { return c >= '0' && c <= '9'; });
    if (!decimal) return sign != 0 ? CERE_ERR_ARGUMENT : CERE_ERR_UNSUPPORTED;
    if (sign != 0 || (digits.size() > 1 && digits[0] == '0') || digits.size() > kMaxOrdinalDigits) {
        return CERE_ERR_ARGUMENT;
    }
    host::ShareConfig cfg;
    bool release_device = false;
    for (size_t pos = comma; pos != std::string::npos;) {
        const size_t next = full.find(',', pos + 1);
        const std::string option = full.substr(pos + 1, next == std::string::npos ? std::string::npos : next - pos - 1);
        const size_t eq = option.find('=');
        if (eq == std::string::npos) return CERE_ERR_ARGUMENT;
        const std::string key = option.substr(0, eq), value = option.substr(eq + 1);
        if (key == "release_device") {
            if (release_device || value != "1") return CERE_ERR_ARGUMENT;
            release_device = true;
        } else if (host::parse_share_option(key, value, &cfg) != 1) {
            return CERE_ERR_ARGUMENT;
        }
        pos = next;
    }
    if (!host::share_config_complete(cfg)) return CERE_ERR_ARGUMENT;
    const int n = std::stoi(digits);
    if (release_device && n >= kTrackedDevices) return CERE_ERR_ARGUMENT;
    *ordinal = n;
    *share = std::move(cfg);
    *release = release_device;
    return CERE_OK;
}

// MemAvailable of /proc/meminfo in bytes; 0 when it cannot be read.
uint64_t mem_available_bytes() {
    std::FILE *f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return 0;
    char line[256];
    unsigned long long kib = 0;
    bool found = false;
    while (!found && std::fgets(line, sizeof line, f) != nullptr) {
        found = std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1;
    }
    std::fclose(f);
    return found ? uint64_t(kib) * 1024 : 0;
}

// CERE_OK when the process loads CUDA modules eagerly (CUDA_MODULE_LOADING=EAGER), CERE_ERR_UNSUPPORTED when lazily.
cere_status require_eager_loading() {
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
    const cudaError_t e =
        cudaGetDriverEntryPointByVersion("cuModuleGetLoadingMode", &fn, 11070, cudaEnableDefault, &query);
    if (e != cudaSuccess) return fail(e);
    if (query != cudaDriverEntryPointSuccess || fn == nullptr) return CERE_ERR_INTERNAL;
    CUmoduleLoadingMode mode = CU_MODULE_LAZY_LOADING;
    if (reinterpret_cast<PFN_cuModuleGetLoadingMode_v11070>(fn)(&mode) != CUDA_SUCCESS) return CERE_ERR_INTERNAL;
    return mode == CU_MODULE_EAGER_LOADING ? CERE_OK : CERE_ERR_UNSUPPORTED;
}

// Calls f(copy, offset, n) for the runs of `bytes` bytes of the copies' concatenated stream, from `copy` at `offset`.
template <typename F>
void for_runs(const std::vector<model::StateCopy> &copies, size_t copy, uint64_t offset, uint64_t bytes, F &&f) {
    for (; bytes > 0; ++copy, offset = 0) {
        const model::StateCopy &c = copies[copy];
        const uint64_t n = std::min(c.bytes - offset, bytes);
        if (n != 0) f(c, offset, n);
        bytes -= n;
    }
}

}  // namespace

// ---- CudaModel ----

CudaModel::CudaModel(int ordinal, ModelLayout layout, std::vector<uint32_t> placements)
    : ordinal_(ordinal), layout_(std::move(layout)), placements_(std::move(placements)),
      slot_used_(layout_.max_sessions, false), pool_(layout_.pool_tokens / host::kKvPageTokens) {}

CudaModel::~CudaModel() {
    release_pin();
    if (pinner_.joinable()) pinner_.join();  // before the block it pins is unregistered and unmapped
    if (device_ == nullptr && host_block_ == nullptr && registered_.empty()) return;
    cudaSetDevice(ordinal_);
    model::release_decode_graphs(&workspace_);  // the graphs reference device_ (forward.cpp graph_step)
    workspace_.lora = nullptr;
    lora_.reset();
    workspace_.research = nullptr;
    research_.reset();
    cudaFree(device_);
    for (uint8_t *piece : registered_) cudaHostUnregister(piece);  // before segment_ (or the private mmap) unmaps
    if (host_map_bytes_ != 0) {
        munmap(host_block_, host_map_bytes_);
    } else {
        cudaFreeHost(host_block_);
    }
    cudaGetLastError();
}

// m216 design note L3. Each page of the block is faulted in (zero-filled when absent) by an atomic `fetch_or(0)` of
// its first byte: a locked read-modify-write that stores the byte it read, so the staging threads' concurrent writes
// into the same block keep their bytes; the populate runs on parallel ranges because faulting 50 GB on one thread is
// the slow part of pinning. A write fault takes only the faulting VMA's lock. MADV_POPULATE_WRITE (in 8 MiB calls,
// load-speed design note LS-03) held the process's mmap lock for each call, so every mmap/munmap/mprotect of the
// process (the upload's allocations, thread stacks, malloc arenas) waited ~16 ms until the populate ended and the read
// pass made no progress beside it (load-crit design note LC-05). cudaHostRegister then pins the populated pages
// without writing them, once every DEVICE byte is put (load-soft design note LSF-01: a registration beside the
// device copies stalls them; with no DEVICE byte it starts at once). A failure is reported by join_pin (INSTANTIATE
// fails; the model is never sealed).
void CudaModel::start_pin() {
    constexpr uint64_t kPage = 4096;
    constexpr uint64_t kParts = 16;
    if (device_put_target_ == 0) release_pin();
    try {
        pinner_ = std::thread([this] {
            const uint64_t n = host_map_bytes_;
            const uint64_t part = ((n + kParts - 1) / kParts + kPage - 1) / kPage * kPage;
            std::vector<std::thread> parts;
            bool populated = true;
            try {
                for (uint64_t off = 0; off < n; off += part) {
                    parts.emplace_back([this, off, end = std::min(off + part, n)] {
                        for (uint64_t at = off; at < end; at += kPage) {
                            __atomic_fetch_or(host_block_ + at, uint8_t{0}, __ATOMIC_RELAXED);
                        }
                    });
                }
            } catch (const std::system_error &) {
                populated = false;
            }
            for (std::thread &t : parts) t.join();
            if (!populated) {
                pin_status_ = CERE_ERR_CAPACITY;
                return;
            }
            {
                // LSF-01: the registration waits for the last DEVICE put (or the join).
                std::unique_lock<std::mutex> lock(pin_mutex_);
                pin_cv_.wait(lock, [this] { return pin_go_; });
            }
            if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) {
                pin_status_ = fail(e);
                return;
            }
            if (const cudaError_t e = register_region(host_block_, n, cudaHostRegisterMapped | cudaHostRegisterPortable);
                e != cudaSuccess) {
                std::fprintf(stderr, "cere: cudaHostRegister of %llu B failed: %s\n",
                             static_cast<unsigned long long>(n), cudaGetErrorString(e));
                pin_status_ = fail(e);
                return;
            }
            // The weights were bound to host_device_ == host_block_ at ALLOCATE (checked at device open that the
            // device uses the host pointer for registered memory); the alias is confirmed here.
            void *alias = nullptr;
            if (const cudaError_t e = cudaHostGetDevicePointer(&alias, host_block_, 0); e != cudaSuccess) {
                pin_status_ = fail(e);
                return;
            }
            if (alias != host_device_) pin_status_ = CERE_ERR_INTERNAL;
        });
    } catch (const std::system_error &) {
        pin_status_ = CERE_ERR_INTERNAL;
    }
}

// m216 design note L4c: the shared region's content digest (an attach; a creator has nothing to verify) and its
// registration run in the background, in parallel with each other and with UPLOAD; ALLOCATE keeps the header check.
// beat-sglang design note L9: a creator first populates the mapping (SharedSegment::populate on 4 threads, the
// fastest count on the dev host's tmpfs: 8 GB in 0.65 s against 0.95 s on 16; open no longer zeroes it with
// posix_fallocate), so its puts land in allocated pages and the registration pins
// pages that exist; the registration of either waits for the last DEVICE put (LSF-01, as the private pin: beside the
// device copies it stalled them, the attach's first ~2 s read nothing). The weights were bound to host_device_ == the
// segment mapping (the device uses the host pointer for registered memory, checked at ALLOCATE); the alias is
// confirmed after the registration. A failure is reported by join_pin.
void CudaModel::start_shared_pin(bool rw) {
    if (device_put_target_ == 0) release_pin();
    try {
        pinner_ = std::thread([this, rw] {
            host::SharedSegment &s = *segment_;
            std::string verify_error;
            cere_status verified = CERE_OK;
            std::thread verifier;
            try {
                verifier = std::thread([&] {
                    verified = s.creating() ? s.populate(4, &verify_error) : s.verify(&verify_error);
                });
            } catch (const std::system_error &) {
                // No second thread: populate or verify before registering.
                verified = s.creating() ? s.populate(4, &verify_error) : s.verify(&verify_error);
            }
            const cere_status pinned = [&]() -> cere_status {
                if (s.creating() && verifier.joinable()) {
                    // The creator registers populated pages only (a registration faults absent ones on one thread).
                    verifier.join();
                    if (verified != CERE_OK) return verified;
                }
                {
                    std::unique_lock<std::mutex> lock(pin_mutex_);
                    pin_cv_.wait(lock, [this] { return pin_go_; });
                }
                if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
                const unsigned flags =
                    cudaHostRegisterMapped | cudaHostRegisterPortable | (rw ? 0u : cudaHostRegisterReadOnly);
                if (const cudaError_t e = register_region(s.data(), s.mapped_bytes(), flags); e != cudaSuccess) {
                    std::fprintf(stderr, "cere: cudaHostRegister of %llu B of %s failed: %s\n",
                                 static_cast<unsigned long long>(s.mapped_bytes()), s.path().c_str(),
                                 cudaGetErrorString(e));
                    return fail(e);
                }
                std::string error;
                if (const cere_status st = s.protect(&error); st != CERE_OK) {
                    std::fprintf(stderr, "cere: %s\n", error.c_str());
                    return st;
                }
                void *alias = nullptr;
                if (const cudaError_t e = cudaHostGetDevicePointer(&alias, s.data(), 0); e != cudaSuccess) {
                    return fail(e);
                }
                return alias == host_device_ ? CERE_OK : CERE_ERR_INTERNAL;
            }();
            if (verifier.joinable()) verifier.join();
            if (verified != CERE_OK) std::fprintf(stderr, "cere: %s\n", verify_error.c_str());
            pin_status_ = verified != CERE_OK ? verified : pinned;
        });
    } catch (const std::system_error &) {
        pin_status_ = CERE_ERR_INTERNAL;
    }
}

cere_status CudaModel::join_pin() {
    release_pin();
    if (pinner_.joinable()) pinner_.join();
    return pin_status_;
}

void CudaModel::release_pin() {
    {
        const std::lock_guard<std::mutex> lock(pin_mutex_);
        if (pin_go_) return;
        pin_go_ = true;
    }
    pin_cv_.notify_all();
}

void CudaModel::device_put(uint64_t bytes) {
    if (device_put_bytes_.fetch_add(bytes) + bytes >= device_put_target_) release_pin();
}

cudaError_t CudaModel::register_region(uint8_t *base, uint64_t bytes, unsigned flags) {
    try {
        registered_.reserve((bytes + kRegisterChunkBytes - 1) / kRegisterChunkBytes);
    } catch (const std::bad_alloc &) {
        return cudaErrorMemoryAllocation;
    }
    for (uint64_t off = 0; off < bytes; off += kRegisterChunkBytes) {
        const uint64_t n = std::min(kRegisterChunkBytes, bytes - off);
        if (const cudaError_t e = cudaHostRegister(base + off, n, flags); e != cudaSuccess) return e;
        registered_.push_back(base + off);
    }
    return cudaSuccess;
}

void *CudaModel::device_ptr(uint64_t tensor_index) const {
    const uint64_t at = layout_.offsets[tensor_index];
    if (at == kNoStorage) return nullptr;
    return (placements_[tensor_index] == CERE_PLACEMENT_HOST_PINNED ? host_device_ : device_) + at;
}

uint8_t *CudaModel::host_ptr(uint64_t tensor_index) const {
    if (placements_[tensor_index] != CERE_PLACEMENT_HOST_PINNED) return nullptr;
    return host_ + layout_.offsets[tensor_index];
}

uint8_t *CudaModel::session_slot(uint64_t slot) const {
    return device_ + layout_.sessions_offset + slot * layout_.session_bytes;
}

uint8_t *CudaModel::state_slot(uint64_t slot) const {
    return device_ + layout_.state_pool_offset + slot * layout_.state_bytes;
}

uint8_t *CudaModel::undo_slot(uint64_t slot, uint64_t j) const {
    if (layout_.undo_bytes == 0) return nullptr;
    return device_ + layout_.undo_offset + (slot * model::layout::kUndoSnapshots + j) * layout_.undo_bytes;
}

model::UndoState *CudaModel::undo_pool() {
    if (layout_.undo_bytes == 0) return nullptr;
    if (undo_pool_.empty()) {
        const uint64_t count = layout_.max_sessions * model::layout::kUndoSnapshots;
        undo_pool_.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            model::UndoState &u = undo_pool_[i];
            model::layout::Carver c{undo_slot(i / model::layout::kUndoSnapshots, i % model::layout::kUndoSnapshots)};
            model::layout::carve_undo(weights_.config, c, [&u](model::layout::Region r, uint32_t l, uint8_t *p) {
                using model::layout::Region;
                switch (r) {
                    case Region::kGdnConv: u.gdn[l].conv = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kGdnSsm: u.gdn[l].ssm = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kPleHistory: u.ple.history = reinterpret_cast<int64_t *>(p); break;
                    case Region::kPleConv: u.ple.conv_state = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kQsaRing: u.qsa_ring[l] = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kUndoKeyRow: u.qsa_key[l] = reinterpret_cast<model::bf16 *>(p); break;
                    default: break;
                }
            });
        }
    }
    return undo_pool_.data();
}

uint32_t CudaModel::max_chain_drafts() const {
    if (layout_.undo_bytes == 0 || layout_.verify_rows == 0) return 0;
    return uint32_t(std::min<uint64_t>({model::layout::kMaxChainDrafts, layout_.verify_rows - 1,
                                        layout_.max_sessions * model::layout::kUndoSnapshots}));
}

cere_status CudaModel::arm_page_order(const uint32_t *order, uint64_t n) {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    return page_order_.arm(order, n);
}

uint64_t CudaModel::free_pages() const {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    return pool_.free_pages();
}

uint32_t CudaModel::page_refs(uint32_t id) const {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    return id < pool_.pages() ? pool_.refs(id) : 0;
}

// ---- CudaSession ----

CudaSession::CudaSession(CudaModel &model, uint64_t slot, uint64_t max_context_tokens, std::vector<uint32_t> pages)
    : model_(model), slot_(slot), max_context_tokens_(max_context_tokens), pages_(std::move(pages)) {}

CudaSession::~CudaSession() {
    std::lock_guard<std::mutex> lock(model_.slots_mutex_);
    if (model_.parked_ == this) {  // MixedPD: a paused call of a released session never resumes
        model_.parked_ = nullptr;
        model_.park_.next_layer = 0;
    }
    model_.slot_used_[slot_] = false;
    host::release_pages_logged(model_.pool_, pages_.data(), pages_.size(), "cuda_session_destroy");
}

// ---- CudaDevice ----

// One ring slot for the lifetime of one `write` or state copy; waits while every slot is leased.
class CudaDevice::Lease {
public:
    explicit Lease(CudaDevice &device) : device_(device) {
        std::unique_lock<std::mutex> lock(device_.ring_mutex_);
        device_.ring_free_.wait(lock, [&] { return !device_.free_slots_.empty(); });
        index_ = device_.free_slots_.back();
        device_.free_slots_.pop_back();
    }
    ~Lease() {
        {
            std::lock_guard<std::mutex> lock(device_.ring_mutex_);
            device_.free_slots_.push_back(index_);
        }
        device_.ring_free_.notify_one();
    }
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;

    Slot &slot() const { return device_.slots_[index_]; }

private:
    CudaDevice &device_;
    size_t index_ = 0;
};

// One staged stream for the lifetime of one write_staged; waits while every stream is leased.
class CudaDevice::StagedLease {
public:
    explicit StagedLease(CudaDevice &device) : device_(device) {
        std::unique_lock<std::mutex> lock(device_.staged_mutex_);
        device_.staged_free_.wait(lock, [&] { return !device_.free_staged_.empty(); });
        index_ = device_.free_staged_.back();
        device_.free_staged_.pop_back();
    }
    ~StagedLease() {
        {
            std::lock_guard<std::mutex> lock(device_.staged_mutex_);
            device_.free_staged_.push_back(index_);
        }
        device_.staged_free_.notify_one();
    }
    StagedLease(const StagedLease &) = delete;
    StagedLease &operator=(const StagedLease &) = delete;

    cudaStream_t stream() const { return device_.staged_streams_[index_]; }

private:
    CudaDevice &device_;
    size_t index_ = 0;
};

CudaDevice::CudaDevice(int ordinal, host::ShareConfig share, bool release)
    : ordinal_(ordinal), share_(std::move(share)), release_(release) {
    if (ordinal_ < kTrackedDevices) g_live_devices[ordinal_].fetch_add(1);
}

CudaDevice::~CudaDevice() {
    std::unique_lock<std::mutex> lock = ordinal_lock(ordinal_);
    cudaSetDevice(ordinal_);
    if (gemm_ != nullptr) kernels::gemm_handle_destroy(gemm_);
    for (Slot &s : slots_) {
        for (cudaEvent_t e : s.done) {
            if (e != nullptr) cudaEventDestroy(e);
        }
        if (s.stream != nullptr) cudaStreamDestroy(s.stream);
    }
    for (cudaStream_t s : staged_streams_) {
        if (s != nullptr) cudaStreamDestroy(s);
    }
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    cudaFreeHost(ring_);
    cudaGetLastError();
    // Device release (cuda_device.h): the models, sessions and snapshots of the executor are gone by now (cere_executor
    // destroys them before its device), so after this the context holds nothing of the executor's.
    if (ordinal_ < kTrackedDevices && g_live_devices[ordinal_].fetch_sub(1) == 1 && release_) {
        context_generation(ordinal_).fetch_add(1);
        cudaDeviceReset();
        cudaGetLastError();
        g_poisoned[ordinal_].store(false);  // a fresh context has no sticky error
    }
}

bool CudaDevice::faulted() const { return ordinal_ < kTrackedDevices && g_poisoned[ordinal_].load(); }

void CudaDevice::probe_sticky() const {
    if (ordinal_ >= kTrackedDevices) return;
    // cudaPeekAtLastError reports a sticky error again on every call (a non-sticky one was cleared by fail()).
    const cudaError_t e = cudaPeekAtLastError();
    if (!is_sticky(e) || g_poisoned[ordinal_].exchange(true)) return;
    std::fprintf(stderr, "cere: device_poisoned ordinal=%d error=%s\n", ordinal_, cudaGetErrorName(e));
}

cere_status CudaDevice::open(const char *device_id, std::unique_ptr<host::Device> *out) {
    int ordinal = 0;
    host::ShareConfig share;
    bool release = false;
    if (const cere_status st = parse_device_id(device_id, &ordinal, &share, &release); st != CERE_OK) return st;
    int count = 0;
    if (const cudaError_t e = cudaGetDeviceCount(&count); e != cudaSuccess) {
        cudaGetLastError();
        return (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) ? CERE_ERR_UNSUPPORTED
                                                                             : CERE_ERR_INTERNAL;
    }
    if (ordinal >= count) return CERE_ERR_UNSUPPORTED;
    cudaDeviceProp prop{};
    if (const cudaError_t e = cudaGetDeviceProperties(&prop, ordinal); e != cudaSuccess) return fail(e);
    if (prop.major != 12 || prop.minor != 0 || !prop.canMapHostMemory || !prop.unifiedAddressing) {
        return CERE_ERR_UNSUPPORTED;
    }
    // A device that fails init is destroyed after the ordinal's lock is released (its destructor takes it).
    std::unique_ptr<CudaDevice> device;
    {
        std::unique_lock<std::mutex> lock = ordinal_lock(ordinal);
        device.reset(new (std::nothrow) CudaDevice(ordinal, std::move(share), release));
        if (device == nullptr) return CERE_ERR_CAPACITY;
        if (const cere_status st = device->init(); st != CERE_OK) return st;
    }
    *out = std::move(device);
    return CERE_OK;
}

cere_status CudaDevice::init() {
    // cudaSetDevice initializes the primary context; with eager loading every module is loaded by then.
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    if (const cudaError_t e = cudaFree(nullptr); e != cudaSuccess) return fail(e);
    if (const cere_status st = require_eager_loading(); st != CERE_OK) return st;
    if (const cere_status st = kernels::gemm_handle_create(&gemm_); st != CERE_OK) return st;
    void *ring = nullptr;
    if (const cudaError_t e = cudaHostAlloc(&ring, kRingSlots * kSlotBytes, cudaHostAllocDefault); e != cudaSuccess) {
        return fail(e);
    }
    ring_ = static_cast<uint8_t *>(ring);
    ring_held_.hold(ledger_, host::Held::kRing, kRingSlots * kSlotBytes);
    if (const cudaError_t e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking); e != cudaSuccess) {
        return fail(e);
    }
    try {
        free_slots_.reserve(kRingSlots);
        free_staged_.reserve(kStagedStreams);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    for (size_t i = 0; i < kRingSlots; ++i) {
        Slot &s = slots_[i];
        s.host = ring_ + i * kSlotBytes;
        if (const cudaError_t e = cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking); e != cudaSuccess) {
            return fail(e);
        }
        for (cudaEvent_t &event : s.done) {
            if (const cudaError_t e = cudaEventCreateWithFlags(&event, cudaEventDisableTiming); e != cudaSuccess) {
                return fail(e);
            }
        }
        free_slots_.push_back(kRingSlots - 1 - i);
    }
    for (size_t i = 0; i < kStagedStreams; ++i) {
        if (const cudaError_t e = cudaStreamCreateWithFlags(&staged_streams_[i], cudaStreamNonBlocking);
            e != cudaSuccess) {
            return fail(e);
        }
        free_staged_.push_back(kStagedStreams - 1 - i);
    }
    const uint64_t available = mem_available_bytes();
    host_budget_ = available > kHostReserveBytes ? available - kHostReserveBytes : 0;
    return CERE_OK;
}

// The text-only A4 entry and the A4 with its vision tower (m213 design note D9, D10, ledger D 112): a plan matches
// the vision entry only when the desc carries the compiled vision_config; its image buffers, encoder scratch and plans
// follow plan.support->vision (memory_plan, bind_weights, instantiate_model), and cere_execution_capabilities reports
// native_image_input (capability 61) from the same entry. A text-only plan is unchanged.
std::vector<const host::Support *> CudaDevice::supported() const {
    return {&host::a4_support(), &host::a4_vision_support()};
}

uint64_t CudaDevice::host_budget_bytes() const { return host_budget_; }

std::string CudaDevice::segment_name(const host::ModelPlan &plan, const ModelLayout &l) const {
    return host::segment_name(share_, "cuda-sm120-v1", host::region_description(plan, l.offsets, l.region_bytes));
}

host::MemoryEstimate CudaDevice::estimate(const host::ModelPlan &plan) const {
    try {
        const ModelLayout layout = plan_layout(plan);
        if (!share_.enabled() || layout.region_bytes == 0 || layout.overflow) {
            return {layout.device_bytes, layout.pinned_bytes};
        }
        // Shared: the staging block and the arena, plus the region only while this process would create it.
        const HostLayout h = shared_host_layout(layout);
        uint64_t host = h.pinned_bytes;
        if (h.overflow) return {UINT64_MAX, UINT64_MAX};
        if (!host::segment_published(share_, segment_name(plan, layout)) &&
            __builtin_add_overflow(host, layout.region_bytes, &host)) {
            return {UINT64_MAX, UINT64_MAX};
        }
        return {layout.device_bytes, host};
    } catch (const std::bad_alloc &) {
        return {UINT64_MAX, UINT64_MAX};  // never fits a budget
    }
}

cere_status CudaDevice::allocate_shared(const host::ModelPlan &plan, CudaModel &m) {
    const ModelLayout &l = m.layout();
    int read_only = 0;
    if (const cudaError_t e = cudaDeviceGetAttribute(&read_only, cudaDevAttrHostRegisterReadOnlySupported, ordinal_);
        e != cudaSuccess) {
        return fail(e);
    }
    int same = 0;
    if (const cudaError_t e = cudaDeviceGetAttribute(&same, cudaDevAttrCanUseHostPointerForRegisteredMem, ordinal_);
        e != cudaSuccess) {
        return fail(e);
    }
    if (same == 0) return CERE_ERR_UNSUPPORTED;
    // The registration path comes from the attribute only (maintainer delegate decision 2026-10-12): read_only where
    // supported, else rw_mprotect (writable map, pin without the ReadOnly flag, then mprotect PROT_READ). No private
    // fallback: a failed registration on either path fails the load (at INSTANTIATE, m216 L4c).
    const bool rw = read_only == 0;
    std::fprintf(stderr, "cere: host_share_registration=%s\n", rw ? "rw_mprotect" : "read_only");
    std::string error;
    // L9: the creator's puts fill every HOST_PINNED tensor once; the put that completes a 64 MiB chunk hashes it.
    host::RegionRanges fill;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        if (plan.tensors[i].placement == CERE_PLACEMENT_HOST_PINNED && l.offsets[i] != kNoStorage &&
            plan.tensors[i].bytes != 0) {
            fill.emplace_back(l.offsets[i], plan.tensors[i].bytes);
        }
    }
    if (const cere_status st = host::SharedSegment::open(share_, segment_name(plan, l), l.region_bytes, rw, fill,
                                                         &m.segment_, &error);
        st != CERE_OK) {
        std::fprintf(stderr, "cere: %s\n", error.c_str());
        return st;
    }
    host::SharedSegment &s = *m.segment_;
    m.host_share_rw_ = rw;
    m.host_ = s.data();
    m.host_device_ = s.data();
    m.region_held_.hold(ledger_, host::Held::kMapped, s.mapped_bytes());
    const HostLayout h = shared_host_layout(l);
    void *block = nullptr;
    if (const cudaError_t e = cudaHostAlloc(&block, h.mapped_bytes, cudaHostAllocMapped); e != cudaSuccess) {
        return fail(e);
    }
    m.host_block_ = static_cast<uint8_t *>(block);
    m.host_held_.hold(ledger_, host::Held::kMapped, h.mapped_bytes);
    m.staging_ = m.host_block_;
    return CERE_OK;
}

cere_status CudaDevice::allocate(const host::ModelPlan &plan, std::unique_ptr<host::DeviceModel> *out) {
    try {
        // LoadCrit LC-03 (O159): ALLOCATE's split (one stderr line per allocated model, as the seal's
        // spec_round_costs line); `estimate` runs plan_layout once more before it, under the executor's lock.
        using Clock = std::chrono::steady_clock;
        using Ms = std::chrono::duration<double, std::milli>;
        const auto started = Clock::now();
        const std::vector<const host::Support *> accepted = supported();
        if (std::find(accepted.begin(), accepted.end(), plan.support) == accepted.end()) return CERE_ERR_UNSUPPORTED;
        ModelLayout layout = plan_layout(plan);
        if (layout.overflow) return CERE_ERR_CAPACITY;
        const auto laid_out = Clock::now();
        // D6′: capture and load are one copy each only while the fixed state is one run of the slot.
        if (plan.prefix_state_slots != 0 && layout.state_bytes == 0) return CERE_ERR_UNSUPPORTED;
        std::vector<uint32_t> placements(plan.tensors.size());
        for (size_t i = 0; i < plan.tensors.size(); ++i) placements[i] = plan.tensors[i].placement;
        auto model = std::make_unique<CudaModel>(ordinal_, std::move(layout), std::move(placements));
        model->drafts_ = plan.speculative_draft_tokens;
        model->levers_ = plan.levers;
        const ModelLayout &l = model->layout();
        if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
        void *device = nullptr;
        if (const cudaError_t e = cudaMalloc(&device, l.device_bytes); e != cudaSuccess) return fail(e);
        model->device_ = static_cast<uint8_t *>(device);
        model->device_held_.hold(ledger_, host::Held::kDevice, l.device_bytes);
        const auto malloced = Clock::now();
        if (share_.enabled() && l.region_bytes != 0) {
            if (const cere_status st = allocate_shared(plan, *model); st != CERE_OK) return st;
        } else if (l.host_bytes != 0) {
            // m216 design note L3: the block is mapped now and pinned in the background while UPLOAD writes it
            // (started at the end of ALLOCATE, M216S-12); the weights bind to its device alias, which is the host
            // pointer on this device (checked here, and confirmed after the pin).
            int same = 0;
            if (const cudaError_t e =
                    cudaDeviceGetAttribute(&same, cudaDevAttrCanUseHostPointerForRegisteredMem, ordinal_);
                e != cudaSuccess) {
                return fail(e);
            }
            if (same == 0) return CERE_ERR_UNSUPPORTED;
            const uint64_t map_bytes = (l.host_bytes + 4095) / 4096 * 4096;
            void *host = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (host == MAP_FAILED) return CERE_ERR_CAPACITY;  // the partial model's destructor frees the arena
            model->host_block_ = static_cast<uint8_t *>(host);
            model->host_map_bytes_ = map_bytes;
            model->host_ = model->host_block_;
            model->staging_ = model->host_ + l.staging_offset;
            model->host_held_.hold(ledger_, host::Held::kMapped, l.host_bytes);
            model->host_device_ = model->host_block_;
        }
        const auto mapped = Clock::now();
        std::string error;
        if (l.snapshot_bytes != 0) {
            if (const cere_status st = model->snapshot_.allocate(l.snapshot_bytes, &error); st != CERE_OK) {
                // The partial model's destructor frees the device arena and the mapped block.
                std::fprintf(stderr, "cere: snapshot arena of %llu B: %s\n",
                             static_cast<unsigned long long>(l.snapshot_bytes), error.c_str());
                return st;
            }
            model->snapshot_held_.hold(ledger_, host::Held::kLocked, model->snapshot_.locked_bytes());
        }
        const auto arena_done = Clock::now();
        CudaModel &m = *model;
        m.arena = m.snapshot_.data();
        const model::ArenaView arena{&plan, l.offsets.data(), m.device_, m.host_device_, m.device_ + l.derived_offset};
        if (const cere_status st = model::bind_weights(arena, &m.weights_, &error); st != CERE_OK) {
            std::fprintf(stderr, "cere: %s\n", error.c_str());
            return st;
        }
        if (const cere_status st = model::bind_workspace(plan.config, l.workspace_rows, plan.max_context_tokens,
                                                         m.device_ + l.workspace_offset, m.staging_,
                                                         &m.workspace_, l.draft_count);
            st != CERE_OK) {
            return st;
        }
        // Track Fp4S3 (fp4local design note FL-30): the MXFP8 weights' dequantize staging (memory_plan.h region 9);
        // a BF16 artifact has none and an empty table.
        if (const cere_status st = model::bind_mxfp8(
                &m.weights_, reinterpret_cast<model::bf16 *>(m.device_ + l.mxfp8_staging_offset),
                l.mxfp8_staging_bytes, &m.workspace_);
            st != CERE_OK) {
            std::fprintf(stderr, "cere: MXFP8 staging of %llu B short of the table's %zu B\n",
                         static_cast<unsigned long long>(l.mxfp8_staging_bytes), m.weights_.mxfp8.staging_needed());
            return st;
        }
        // m213 design note D10-D12: the image buffers and the encoder scratch (the whole workspace region).
        if (plan.support->vision != nullptr) {
            // Host staging offsets are in the mapped block; shared, its private tail starts at staging_offset.
            const auto host_at = [&](uint64_t off) {
                return m.segment_ ? m.host_block_ + (off - l.staging_offset) : m.host_ + off;
            };
            model::Workspace &ws = m.workspace_;
            ws.region = m.device_ + l.workspace_offset;
            ws.region_bytes = l.workspace_bytes;
            ws.image_embeds = reinterpret_cast<model::bf16 *>(m.device_ + l.image_embed_offset);
            ws.image_out = reinterpret_cast<model::bf16 *>(m.device_ + l.image_out_offset);
            ws.image_patches = reinterpret_cast<model::bf16 *>(m.device_ + l.image_patch_offset);
            ws.host_image_patches = host_at(l.image_staging_offset);
            ws.image_patch_bytes = l.image_patch_bytes;
            ws.positions3 = reinterpret_cast<int64_t *>(m.device_ + l.image_positions_offset);
            ws.host_positions3 = reinterpret_cast<int64_t *>(host_at(l.image_positions_staging_offset));
            ws.max_image_patches =
                uint32_t(model::layout::max_image_patches(*plan.support->vision, plan.max_context_tokens));
        }
        // R of the batched decode step (m26 design note D3/D6): the plan's max_sessions; rows plans at INSTANTIATE.
        // With drafts the verify call's rows (m29 design note M29S2-05): min(16, 4 x max_sessions), whose sampler
        // stages (phase S) live in the draft-only region.
        m.workspace_.max_rows = uint32_t(plan.max_sessions);
        if (m.drafts_ > 0) {
            m.workspace_.max_rows = uint32_t(l.verify_rows);
            model::bind_spec_sampler(plan.config, l.verify_rows, m.device_ + l.spec_sampler_offset, &m.workspace_);
        }
        // MixedPD (mixed-pd design note): the park region of a layer-sliced plan (memory_plan.h), hyper then normed.
        if (l.park_bytes != 0) {
            m.park_.hyper = reinterpret_cast<model::bf16 *>(m.device_ + l.park_offset);
            m.park_.normed = reinterpret_cast<model::bf16 *>(m.device_ + l.park_offset + l.park_bytes / 2);
        }
        if (const cere_status st = model::bind_pool(plan.config, m.device_ + l.pool_offset, l.pool_tokens, &m.kv_,
                                                    l.draft_count != 0);
            st != CERE_OK) {
            return st;
        }
        // M2.9 (m29 design note D3): the device copy of the draft token map (INSTANTIATE gathers the hot lm_head).
        if (m.weights_.has_draft) {
            if (const cere_status st = model::mtp::upload_map(&m.weights_, plan.draft_token_map, stream_);
                st != CERE_OK) {
                return st;
            }
        }
        // m216 design note M216S-12: the background pin (private: populate, then register; shared: a creator's
        // populate or an attach's digest, then register) starts after ALLOCATE's last CUDA call, which would otherwise
        // wait for a registration (the draft map's copy waited 7.7 s, M216S-10); its pieces let UPLOAD's copies run
        // in between. Either registration waits for the last DEVICE put (LSF-01; L9 for the shared one).
        if (m.segment_ != nullptr || m.host_map_bytes_ != 0) {
            for (size_t i = 0; i < plan.tensors.size(); ++i) {
                if (plan.tensors[i].placement == CERE_PLACEMENT_DEVICE) m.device_put_target_ += plan.tensors[i].bytes;
            }
        }
        if (m.segment_ != nullptr) {
            m.start_shared_pin(m.host_share_rw_);
        } else if (m.host_map_bytes_ != 0) {
            m.start_pin();
        }
        const auto bound = Clock::now();
        std::fprintf(stderr,
                     "cere: allocate layout_ms=%.1f malloc_ms=%.1f host_ms=%.1f snapshot_ms=%.1f bind_ms=%.1f\n",
                     Ms(laid_out - started).count(), Ms(malloced - laid_out).count(), Ms(mapped - malloced).count(),
                     Ms(arena_done - mapped).count(), Ms(bound - arena_done).count());
        *out = std::move(model);
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
}

cere_status CudaDevice::copy_to_device(uint8_t *dst, const uint8_t *src, uint64_t len) {
    Lease lease(*this);
    Slot &s = lease.slot();
    // Two halves alternate: the CPU fills one while the DMA engine drains the other.
    bool pending[2] = {false, false};
    cudaError_t error = cudaSuccess;
    for (int half = 0; len > 0; half ^= 1) {
        const uint64_t n = std::min(len, kHalfBytes);
        uint8_t *stage = s.host + uint64_t(half) * kHalfBytes;
        if (pending[half] && (error = cudaEventSynchronize(s.done[half])) != cudaSuccess) break;
        std::memcpy(stage, src, n);
        if ((error = cudaMemcpyAsync(dst, stage, n, cudaMemcpyHostToDevice, s.stream)) != cudaSuccess) break;
        if ((error = cudaEventRecord(s.done[half], s.stream)) != cudaSuccess) break;
        pending[half] = true;
        dst += n;
        src += n;
        len -= n;
    }
    // Drained even after an error, so the slot is never reused while a copy still reads it.
    const cudaError_t sync = cudaStreamSynchronize(s.stream);
    if (error == cudaSuccess) error = sync;
    return error == cudaSuccess ? CERE_OK : fail(error);
}

cere_status CudaDevice::write(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                              uint64_t len) {
    auto &m = static_cast<CudaModel &>(model);
    const uint64_t at = m.layout_.offsets[tensor_index];
    if (at == kNoStorage) return CERE_ERR_ARGUMENT;
    const auto *src = static_cast<const uint8_t *>(bytes);
    if (m.placements_[tensor_index] == CERE_PLACEMENT_HOST_PINNED) {
        if (m.segment_ == nullptr) {
            std::memcpy(m.host_ + at + offset, src, len);
            return CERE_OK;
        }
        std::string error;
        const cere_status st = m.segment_->write(at + offset, src, len, &error);
        if (st != CERE_OK) std::fprintf(stderr, "cere: %s\n", error.c_str());
        return st;
    }
    // `write` runs on the caller's (staging) thread, whose current device is not necessarily ours.
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    const cere_status st = copy_to_device(m.device_ + at + offset, src, len);
    if (st == CERE_OK) m.device_put(len);
    return st;
}

namespace {

// m216 design note L7 (one producer, N consumers): one staging block may be registered with the executors of
// several devices in one process. CUDA pins a host range once per process, so the pin is shared: the first
// registration pins it portable (page-locked for every device's context), later ones of the same range count a
// reference, and the last unregistration unpins it.
struct StagingPin {
    uint64_t bytes = 0;
    uint32_t refs = 0;
};
std::mutex g_staging_pins_mutex;
std::map<const uint8_t *, StagingPin> g_staging_pins;

cere_status pin_staging(const uint8_t *base, uint64_t bytes) {
    const std::lock_guard<std::mutex> lock(g_staging_pins_mutex);
    auto [it, added] = g_staging_pins.try_emplace(base);
    if (!added) {
        if (it->second.bytes != bytes) return CERE_ERR_ARGUMENT;  // another range at this base is still pinned
        ++it->second.refs;
        return CERE_OK;
    }
    if (const cudaError_t e = cudaHostRegister(const_cast<uint8_t *>(base), bytes, cudaHostRegisterPortable);
        e != cudaSuccess) {
        g_staging_pins.erase(it);
        std::fprintf(stderr, "cere: cudaHostRegister of %llu B of put staging failed: %s\n",
                     static_cast<unsigned long long>(bytes), cudaGetErrorString(e));
        cudaGetLastError();
        return CERE_ERR_CAPACITY;
    }
    it->second = StagingPin{bytes, 1};
    return CERE_OK;
}

void unpin_staging(const uint8_t *base) {
    const std::lock_guard<std::mutex> lock(g_staging_pins_mutex);
    const auto it = g_staging_pins.find(base);
    if (it == g_staging_pins.end() || --it->second.refs != 0) return;
    cudaHostUnregister(const_cast<uint8_t *>(base));
    cudaGetLastError();
    g_staging_pins.erase(it);
}

}  // namespace

cere_status CudaDevice::register_staging(const uint8_t *base, uint64_t bytes) {
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    try {
        auto [it, added] = staging_held_.try_emplace(base);
        if (!added) return CERE_ERR_ARGUMENT;  // the host keeps ranges disjoint; never reached
        if (const cere_status st = pin_staging(base, bytes); st != CERE_OK) {
            staging_held_.erase(it);
            return st;
        }
        it->second.hold(ledger_, host::Held::kStaging, bytes);
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
}

void CudaDevice::unregister_staging(const uint8_t *base, uint64_t) {
    if (staging_held_.erase(base) == 0) return;
    cudaSetDevice(ordinal_);
    unpin_staging(base);
}

// The source is page-locked (the host found it inside a registered range), so the DMA engine reads it in place: no
// bounce copy, and the leased stream is synchronized before return, so the caller may reuse its buffer at once.
cere_status CudaDevice::write_staged(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset,
                                     const void *bytes, uint64_t len) {
    auto &m = static_cast<CudaModel &>(model);
    if (m.placements_[tensor_index] != CERE_PLACEMENT_DEVICE) return write(model, tensor_index, offset, bytes, len);
    const uint64_t at = m.layout_.offsets[tensor_index];
    if (at == kNoStorage) return CERE_ERR_ARGUMENT;
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    StagedLease lease(*this);
    cudaError_t error = cudaMemcpyAsync(m.device_ + at + offset, bytes, len, cudaMemcpyHostToDevice, lease.stream());
    const cudaError_t sync = cudaStreamSynchronize(lease.stream());
    if (error == cudaSuccess) error = sync;
    if (error != cudaSuccess) return fail(error);
    m.device_put(len);
    return CERE_OK;
}

// m216 design note L4: an attached segment holds every HOST_PINNED byte already (its header was checked at ALLOCATE;
// L4c: its content digest is verified before INSTANTIATE seals); a private block or a segment this process creates
// has to be put.
cere_status CudaDevice::host_resident(const host::DeviceModel &model, uint64_t tensor_index,
                                      const uint8_t **out) const {
    const auto &m = static_cast<const CudaModel &>(model);
    if (m.segment_ == nullptr || m.segment_->creating()) return CERE_ERR_UNSUPPORTED;
    const uint64_t at = m.layout_.offsets[tensor_index];
    if (at == kNoStorage || m.placements_[tensor_index] != CERE_PLACEMENT_HOST_PINNED) return CERE_ERR_ARGUMENT;
    *out = m.segment_->data() + at;
    return CERE_OK;
}

cere_status CudaDevice::instantiate(host::DeviceModel &model) {
    // Every put was stream-synchronized before it returned; the device is drained once more so an asynchronous error
    // of any earlier work surfaces here, before the model is reported sealed. Modules were loaded eagerly at executor
    // creation and the arenas hold the workspace, the derived region and every session slot, so readiness starts no
    // lazy device work; the plans are the last host objects the model builds.
    auto &m = static_cast<CudaModel &>(model);
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    // L3/L4c: every HOST_PINNED byte is put or resident; the region is pinned (its alias confirmed) and an attached
    // segment's content digest verified before the model is sealed and any kernel reads it.
    if (const cere_status st = m.join_pin(); st != CERE_OK) return st;
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return fail(e);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return fail(e);
    if (m.segment_ != nullptr && m.segment_->creating()) {
        // Every HOST_PINNED byte is written: the creator publishes the region (an attacher has nothing to do). L9:
        // the puts hashed the chunks they completed, so the publish hashes only the rest.
        const auto started = std::chrono::steady_clock::now();
        const uint64_t hashed = m.segment_->chunks_hashed();
        std::string error;
        if (const cere_status st = m.segment_->publish(&error); st != CERE_OK) {
            std::fprintf(stderr, "cere: %s\n", error.c_str());
            return st;
        }
        std::fprintf(stderr, "cere: host_share publish_ms=%.1f chunks=%llu hashed_in_pass=%llu\n",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
                     static_cast<unsigned long long>(m.segment_->chunks()), static_cast<unsigned long long>(hashed));
    }
    if (!m.instantiated_) {
        // MF-1: instantiate_model rewrites the MoE block scales in place, so a failed attempt is final.
        if (m.instantiate_attempted_) return CERE_ERR_INTERNAL;
        m.instantiate_attempted_ = true;
        if (const cere_status st = model::instantiate_model(&m.weights_, &m.workspace_, gemm_, stream_); st != CERE_OK) {
            std::fprintf(stderr, "cere: instantiate_model failed (status %d)\n", int(st));
            probe_sticky();
            return st;
        }
        // OH-08: a failure after instantiate_model rewrote the scales, which the retry must not repeat (MF-1).
        CERE_FAULT_POINT(seal, return CERE_ERR_INTERNAL);
        if (m.weights_.has_draft) {
            if (const cere_status st = model::mtp::instantiate(&m.weights_, &m.workspace_, gemm_, stream_);
                st != CERE_OK) {
                return st;
            }
        }
        // k2_weights (restart-only, fixed at create): FP8 converts the dense projections in place, after the MTP
        // draft lm_head was gathered from the BF16 lm_head.
        if (m.levers_.k2_weights == CERE_LEVER_K2_WEIGHTS_FP8) {
            if (const cere_status st = model::convert_fp8_weights(&m.weights_, &m.workspace_, stream_);
                st != CERE_OK) {
                std::fprintf(stderr, "cere: convert_fp8_weights failed (status %d)\n", int(st));
                return st;
            }
        }
        m.instantiated_ = true;
    }
    // The plans start on the default selections (GemmPlan::init: cuBLASLt; Workspace::prefill_p4 false; decode graphs
    // off); the model's levers switch them here, as a hot swap does (set_levers).
    model::apply_levers(m.levers_, &m.workspace_);
    forward_ready_.store(true);
    return CERE_OK;
}

uint32_t CudaDevice::moe_lever() const {
#if defined(CERE_MOE_P3B)
    return CERE_LEVER_MOE_P3B;
#else
    return CERE_LEVER_MOE_K7;
#endif
}

cere_status CudaDevice::set_levers(host::DeviceModel &model, const host::Levers &levers) {
    // Hot-safe (cere_model_set_selection runs with no ticket in flight, so no forward call reads the flags meanwhile):
    // every selection runs over the buffers ALLOCATE carved whatever the levers, so no workspace byte is allocated or
    // freed. k2_gemm: the small-M kernel keeps its split-K partials in shared memory / cluster DSMEM (k2_small_m.h, no
    // workspace) and the batched lm_head's rows buffer (Workspace::rows_logits) is carved inside `scratch` for every
    // model; the rows plans hold both fused masks (GemmPlan::init_rows). prefill_attention: P4 takes no workspace and
    // K6/K9's scratch is carved for every call. decode_step: the graphs replay the eager step's launches over the same
    // workspace (ids64's tail stages the row positions, layout::decode_graph_fits; P1 M23P1-03 adds no byte), and
    // apply_levers destroys every captured executable, so no graph keeps a kernel the swap deselected; the next
    // eligible step captures anew. draft_tokens is the host's (verify window inside the allocated draft buffers);
    // moe_gemm is validated only (no selection in this library).
    auto &m = static_cast<CudaModel &>(model);
    if (m.instantiated_) {
        if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
        model::apply_levers(levers, &m.workspace_);
    }
    m.levers_ = levers;
    return CERE_OK;
}

bool CudaDevice::forward_supported() const { return forward_ready_.load(); }

bool CudaDevice::draft_supported(const host::DeviceModel &model) const {
    return static_cast<const CudaModel &>(model).drafts_ > 0;
}

bool CudaDevice::sampler_supported() const { return forward_supported(); }

cere_status CudaDevice::create_session(host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                                       std::unique_ptr<host::DeviceSession> *out) {
    auto &m = static_cast<CudaModel &>(model);
    if (lazy && m.layout_.state_slots == 0) return CERE_ERR_ARGUMENT;
    const uint64_t n_pages = lazy ? 0 : host::kv_pages(max_context_tokens);
    uint64_t slot = 0;
    std::vector<uint32_t> pages;
    {
        // One critical section (m25 F7): the slot, the session's pages (lowest free first, D3; none when lazy, M2.10
        // D3′) and K11's armed order, which this create consumes whatever its outcome. CAPACITY keeps nothing.
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        host::PagePermutation order = std::move(m.page_order_);
        m.page_order_.clear();
        const auto free = std::find(m.slot_used_.begin(), m.slot_used_.end(), false);
        if (free == m.slot_used_.end()) return CERE_ERR_CAPACITY;
        if (!lazy) {
            try {
                pages.reserve(n_pages);
            } catch (const std::bad_alloc &) {
                return CERE_ERR_CAPACITY;
            }
            if (const cere_status st = m.pool_.take_pages(n_pages, &pages); st != CERE_OK) return st;
            if (const cere_status st = order.apply(&pages); st != CERE_OK) {
                host::release_pages_logged(m.pool_, pages.data(), pages.size(), "cuda_session_create");
                return st;
            }
        }
        slot = uint64_t(free - m.slot_used_.begin());
        *free = true;
    }
    // The allocation is sequenced before the constructor's arguments, so a failed new leaves `pages` untouched.
    std::unique_ptr<CudaSession> session(new (std::nothrow) CudaSession(m, slot, max_context_tokens, std::move(pages)));
    if (session == nullptr) {
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        m.slot_used_[slot] = false;
        host::release_pages_logged(m.pool_, pages.data(), pages.size(), "cuda_session_create");
        return CERE_ERR_CAPACITY;
    }
    // From here the session's destructor frees the slot and the pages on every error.
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    if (const cere_status st = model::bind_session(m.weights_.config, m.session_slot(slot), max_context_tokens, m.kv_,
                                                   &session->state_);
        st != CERE_OK) {
        return st;
    }
    // M2.9 (M29S2-20): the slot's undo snapshots (pointers only; filled by each verify before use).
    if (m.drafts_ > 0) {
        for (uint64_t j = 0; j < model::layout::kUndoSnapshots; ++j) {
            model::UndoState &u = session->undo_[j];
            model::layout::Carver c{m.undo_slot(slot, j)};
            model::layout::carve_undo(m.weights_.config, c, [&u](model::layout::Region r, uint32_t l, uint8_t *p) {
                using model::layout::Region;
                switch (r) {
                    case Region::kGdnConv: u.gdn[l].conv = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kGdnSsm: u.gdn[l].ssm = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kPleHistory: u.ple.history = reinterpret_cast<int64_t *>(p); break;
                    case Region::kPleConv: u.ple.conv_state = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kQsaRing: u.qsa_ring[l] = reinterpret_cast<model::bf16 *>(p); break;
                    case Region::kUndoKeyRow: u.qsa_key[l] = reinterpret_cast<model::bf16 *>(p); break;
                    default: break;
                }
            });
            u.key_row = -1;
        }
    }
    if (const cere_status st = model::session_reset(m.weights_.config, &session->state_, stream_); st != CERE_OK) {
        return st;
    }
    // A lazy session's slot map stays zeroed until bind_pages writes it.
    if (!lazy) {
        if (const cere_status st = model::session_set_pages(&session->state_, session->pages_.data(),
                                                            session->pages_.size(), m.layout_.pool_tokens, stream_);
            st != CERE_OK) {
            return st;
        }
    }
    session->lazy_ = lazy;
    if (const cudaError_t e = cudaStreamSynchronize(stream_); e != cudaSuccess) return fail(e);
    *out = std::move(session);
    return CERE_OK;
}

cere_status CudaDevice::forward(CudaModel &model, CudaSession &session, uint32_t operation, const uint32_t *tokens,
                                uint32_t n, model::ForwardTaps *taps, cere_token_score *out,
                                const model::SamplerArgs *sampler, cere_token_logprobs *logprobs,
                                const host::StepImages *images, const model::EndCall *end, model::Segment *segment) {
    if (!model.instantiated_ || &session.model_ != &model || tokens == nullptr || out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (session.broken_) return CERE_ERR_INTERNAL;
    if (session.pages_.empty()) return CERE_ERR_ARGUMENT;  // lazy and unbound: its slot map points nowhere (D3′)
    if (n == 0 || n > session.state_.max_context_tokens - session.state_.position) return CERE_ERR_CAPACITY;
    // `step` runs on the polling thread, whose current device is not necessarily ours.
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    bool launched = false;
    const cere_status st = model::forward(model.weights_, model.workspace_, session.state_, operation, tokens, n, taps,
                                          out, sampler, logprobs, stream_, launched, images, end, segment);
    // Without `launched` the call returned from a precondition (the session untouched); after it any failure,
    // whatever its status, leaves the session undefined (MF-2).
    if (st != CERE_OK && launched) {
        probe_sticky();
        session.broken_ = true;
    }
    return st;
}

cere_status CudaDevice::mtp_forward(CudaModel &model, CudaSession &session, const model::mtp::MtpCall &call,
                                    model::bf16 *logits) {
    if (!model.instantiated_ || !model.weights_.has_draft || &session.model_ != &model) return CERE_ERR_ARGUMENT;
    if (session.broken_) return CERE_ERR_INTERNAL;
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    const cere_status st =
        model::mtp::forward_logits(model.weights_, model.workspace_, session.state_, call, logits, stream_);
    if (st != CERE_OK) return st;
    if (const cudaError_t e = cudaStreamSynchronize(stream_); e != cudaSuccess) return fail(e);
    // Audit KernelsAttention F3: a K9 launch clamped a device position (model::take_bound_fault).
    if (const cere_status bound = model::take_bound_fault(model.workspace_); bound != CERE_OK) {
        session.broken_ = true;
        return bound;
    }
    return CERE_OK;
}

cere_status CudaDevice::forward_rows(CudaModel &model, CudaSession *const *sessions, uint32_t rows,
                                     const uint32_t *tokens, model::ForwardTaps *taps, cere_token_score *out,
                                     const model::SamplerArgs *const *samplers,
                                     cere_token_logprobs *const *logprobs) {
    if (!model.instantiated_ || sessions == nullptr || rows == 0 || rows > kernels::GemmPlan::kMaxRows) {
        return CERE_ERR_ARGUMENT;
    }
    model::SessionState *states[kernels::GemmPlan::kMaxRows];
    for (uint32_t r = 0; r < rows; ++r) {
        if (sessions[r] == nullptr || &sessions[r]->model_ != &model) return CERE_ERR_ARGUMENT;
        if (sessions[r]->broken_) return CERE_ERR_INTERNAL;
        if (sessions[r]->pages_.empty()) return CERE_ERR_ARGUMENT;
        states[r] = &sessions[r]->state_;
    }
    // Called on the polling thread, whose current device is not necessarily ours.
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    bool launched = false;
    const cere_status st = model::forward_decode_rows(model.weights_, model.workspace_, states, rows, tokens, taps,
                                                      out, samplers, logprobs, stream_, launched);
    // Without `launched` the call returned from a precondition (nothing ran); after it any failure leaves every
    // row's session undefined (MF-2).
    if (st != CERE_OK && launched) {
        probe_sticky();
        for (uint32_t r = 0; r < rows; ++r) sessions[r]->broken_ = true;
    }
    return st;
}

cere_status CudaDevice::step(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *tokens,
                             uint64_t count, uint64_t limit, const host::StepImages *images, uint64_t *consumed,
                             const host::ItemSampling *sampling, cere_token_score *out,
                             cere_token_logprobs *logprobs, const host::EndStep *end, host::StepSlice *slice) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (slice != nullptr) slice->paused = false;
    if (consumed == nullptr || count == 0) return CERE_ERR_ARGUMENT;
    if (sampling == nullptr && logprobs != nullptr) return CERE_ERR_ARGUMENT;
    *consumed = 0;
    if (count > s.state_.max_context_tokens - s.state_.position) return CERE_ERR_CAPACITY;
    // The host feeds a session's prompt from its first step (a PREFILL work item appends to an empty session), one
    // chunk per step (M2.6 D2a: the rest of an open prompt continues as PREFILL), and one generated token in every
    // later step (DECODE continues from the last committed token, cere_executor.h). A prefix hit (load_slot) leaves
    // the prompt open, so the tail after it is PREFILL too (M2.10 D3′).
    const uint32_t operation = (s.state_.position == 0 || s.prompt_open_) ? CERE_OP_PREFILL : CERE_OP_DECODE;
    // MixedPD (StepSlice): a paused call resumes before any other PREFILL of the model; a resume without a slice
    // runs to its end. Its EndStep capture took the start state at the first segment.
    const bool resuming = m.parked_ == &s;
    if (m.parked_ != nullptr && !resuming && operation == CERE_OP_PREFILL) return CERE_ERR_ARGUMENT;
    // A resumed call that fails anywhere leaves the session's earlier segments applied: it breaks the session and
    // frees the park for the model's other prompts.
    struct Abandon {
        CudaModel &m;
        CudaSession &s;
        bool armed;
        ~Abandon() {
            if (!armed) return;
            m.park_.next_layer = 0;
            m.parked_ = nullptr;
            s.broken_ = true;
        }
    } abandon{m, s, resuming};
    const bool sliced = slice != nullptr && slice->budget_ms > 0 && m.park_.hyper != nullptr &&
                        operation == CERE_OP_PREFILL;
    const uint64_t chunk = prefill_chunk(m);
    // M2.10 D4′: `limit` ends the call at a capture point (0 = none).
    const auto n = static_cast<uint32_t>(std::min(std::min(chunk, count), limit == 0 ? count : limit));
    // D15⁗′ (host/device.h EndStep): the pairs' views and the call's EndCall, checked before anything runs.
    model::EndCall call;
    model::SessionState pair[2];
    model::SessionState in_pair[2];
    if (end != nullptr) {
        const bool capture = end->row > 0;
        const bool resume = end->resume;
        // A chunk-0 end state's K/V runs (EndStep::kv_slots, in_kv_slots): the capture's rows [0, position + row),
        // the resume's rows [0, position) after load_slot. The cold call's rows lie inside its class.
        const uint64_t position = s.state_.position;
        const uint64_t kv_cls = end->cls != 0 ? end->cls : n;
        const uint64_t slots = m.layout_.state_slots;
        const uint64_t out_end = uint64_t(end->slot) + 2 + end->kv_slots;
        const uint64_t in_end = uint64_t(end->in_slot) + 2 + end->in_kv_slots;
        const bool capture_ok =
            !capture || (out_end <= slots && end->row < n &&
                         (end->kv_slots == 0 ||
                          ((position == 0 || (resume && end->in_kv_slots != 0)) && position + n <= kv_cls &&
                           end_kv_slots(model, position + end->row, kv_cls) == end->kv_slots)));
        const bool resume_ok =
            !resume || (in_end <= slots && (end->in_kv_slots == 0 ||
                                            (position > 0 && position + n <= kv_cls &&
                                             end_kv_slots(model, position, kv_cls) == end->in_kv_slots)));
        const bool disjoint = !capture || !resume || out_end <= end->in_slot || in_end <= end->slot;
        if (operation != CERE_OP_PREFILL || &s.model_ != &m || s.pages_.empty() || (!capture && !resume) ||
            (end->cls != 0 && end->cls < n) || end->cls > m.workspace_.max_tokens || !capture_ok || !resume_ok ||
            !disjoint) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint32_t i = 0; i < 2; ++i) {
            if (capture) {
                if (const cere_status st = model::bind_fixed_state(m.weights_.config, m.state_slot(end->slot + i),
                                                                   s.pages_.size(), m.kv_, &pair[i]);
                    st != CERE_OK) {
                    return st;
                }
            }
            if (resume) {
                if (const cere_status st = model::bind_fixed_state(m.weights_.config, m.state_slot(end->in_slot + i),
                                                                   s.pages_.size(), m.kv_, &in_pair[i]);
                    st != CERE_OK) {
                    return st;
                }
            }
        }
        call.cls = uint32_t(end->cls);
        if (capture) {
            call.row = int64_t(end->row);
            call.a = &pair[0];
            call.b = &pair[1];
            if (end->kv_slots != 0) {
                call.kv = reinterpret_cast<model::bf16 *>(m.state_slot(end->slot + 2));
                call.kv_rows = int64_t(position + end->row);
            }
        }
        if (resume) {
            call.in_a = &in_pair[0];
            call.in_b = &in_pair[1];
            if (end->in_kv_slots != 0) {
                call.in_kv = reinterpret_cast<const model::bf16 *>(m.state_slot(end->in_slot + 2));
                call.in_kv_rows = int64_t(position);
            }
        }
        // A capture's slot first takes the whole fixed state as of the call start (rings and the rest); the forward
        // then overwrites its GDN, SSM and PLE parts as of `row` (stream order).
        if (capture && !resuming) {
            if (const cere_status st = capture_state(model, session, end->slot); st != CERE_OK) return st;
        }
    }
    // Only the call that ends the fed tokens scores the next token, so only it runs the sampler stage.
    const bool last = n == count;
    cere_token_score score{};
    cere_token_logprobs lp{};
    model::Segment segment;
    const uint32_t first = m.park_.next_layer;
    if (sliced || resuming) {
        segment.park = &m.park_;
        segment.layers =
            sliced ? model::segment::segment_layers(
                         first, m.park_ms_per_layer_,
                         model::segment::fit_ms_per_layer(m.park_ms_per_layer_, n, m.park_tokens_), slice->budget_ms)
                   : model::kLayers;
    }
    const auto started = std::chrono::steady_clock::now();
    if (const cere_status st = forward(m, s, operation, tokens, n, nullptr, &score, last ? sampling : nullptr,
                                       last && logprobs != nullptr ? &lp : nullptr, images,
                                       end != nullptr ? &call : nullptr, segment.park != nullptr ? &segment : nullptr);
        st != CERE_OK) {
        return st;  // a first segment's failure after its first launch broke the session and cleared the park
    }
    abandon.armed = false;
    const std::chrono::duration<double, std::milli> ms = std::chrono::steady_clock::now() - started;
    if (segment.paused) {
        m.park_ms_per_layer_ = ms.count() / double(m.park_.next_layer - first);
        m.park_tokens_ = n;
        m.parked_ = &s;
        slice->paused = true;
        return CERE_OK;
    }
    // BcSegment (O283): a sliced call from layer 0 that ran whole is measured too, its tail counted as
    // kTailLayers layers (the budget's own model of a whole call), since before BcSegment every such call paused and
    // was measured; without this a stale measurement from another prompt length would decide the next fit.
    if (sliced && first == 0) {
        m.park_ms_per_layer_ = ms.count() / double(model::kLayers + model::segment::kTailLayers);
        m.park_tokens_ = n;
    }
    m.parked_ = nullptr;
    s.prompt_open_ = !last;
    // M2.9 D2: a drafting session's MTP extend over the chunk (next ids = the chunk shifted by one, then the next
    // prompt token or the scored token), at the chunk's rotary positions (an image prompt's: the call's prompt_rope,
    // whose columns the forward left in ws.positions3); its decode steps only after the prompt's last chunk
    // (layout::draft_decode_steps). A failure breaks the session (its target state already advanced).
    const model::Rope prompt =
        images != nullptr ? model::prompt_rope(m.weights_, m.workspace_, s.state_.position - n, n) : model::Rope{};
    // A chunk-0 end state's MTP layer K/V: the runs' last layer.
    const bool mtp_end = call.kv != nullptr || call.in_kv != nullptr;
    const model::EndKv mtp_kv =
        mtp_end ? model::end_kv(call, model::end_kv_index(m.weights_.config, model::kLayers)) : model::EndKv{};
    if (const cere_status st = extend_draft(m, s, tokens, n, last ? score.id : tokens[n], !last, call.cls,
                                            images != nullptr ? &prompt : nullptr, mtp_end ? &mtp_kv : nullptr);
        st != CERE_OK) {
        return st;
    }
    *consumed = n;
    if (last) {
        *out = score;
        if (logprobs != nullptr) *logprobs = lp;
    }
    return CERE_OK;
}

cere_status CudaDevice::extend_draft(CudaModel &model, CudaSession &session, const uint32_t *tokens, uint32_t n,
                                     uint32_t next, bool prompt_open, uint32_t cls, const model::Rope *rope,
                                     const model::EndKv *end_kv) {
    if (!session.drafting_) return CERE_OK;
    if (&session.model_ != &model || tokens == nullptr || n == 0 || n > session.state_.position) {
        return CERE_ERR_ARGUMENT;
    }
    if (session.broken_) return CERE_ERR_INTERNAL;
    // The forward drained the stream, so the mapped staging is free.
    model::Workspace &ws = model.workspace_;
    if (n > ws.max_tokens) return CERE_ERR_ARGUMENT;
    for (uint32_t r = 0; r + 1 < n; ++r) ws.host_ids32[r] = int32_t(tokens[r + 1]);
    ws.host_ids32[n - 1] = int32_t(next);
    if (const cudaError_t e =
            cudaMemcpyAsync(ws.ids32, ws.host_ids32, n * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        e != cudaSuccess) {
        session.broken_ = true;
        return fail(e);
    }
    return draft(model, session, session.state_.position - n, n, ws.hc_rows,
                 reinterpret_cast<const uint32_t *>(ws.ids32), cls, rope, end_kv, prompt_open);
}

cere_status CudaDevice::draft(CudaModel &model, CudaSession &session, uint64_t p0, uint32_t rows, model::bf16 *hc,
                              const uint32_t *next, uint32_t cls, const model::Rope *rope,
                              const model::EndKv *end_kv, bool prompt_open) {
    model::SessionState &state = session.state_;
    uint32_t *proposals = state.proposals;
    model::Workspace &ws = model.workspace_;
    cere_status st = CERE_OK;
    uint32_t count = 1;
    const uint64_t steps = model::layout::draft_decode_steps(p0, rows, state.max_context_tokens, prompt_open);
    if (cls == 0 && rope == nullptr && end_kv == nullptr && state.rope_delta == 0 && ws.graphs.spec &&
        steps == model::layout::kDraftSteps &&
        model::layout::spec_draft_graph_fits(p0, rows, state.max_context_tokens, ws.max_tokens)) {
        // decode_step = spec_graph (M29S2-18b): the same launches replayed from the draft graph of `rows`, every
        // decode step of the eager loop below included (the fit rule leaves both inside the context).
        st = model::draft_graph(model.weights_, ws, state, p0, rows, hc, next, proposals, stream_);
        count = 1 + uint32_t(model::layout::kDraftSteps);
    } else {
        if (mtp_positions_.size() < rows) mtp_positions_.resize(rows);
        for (uint32_t r = 0; r < rows; ++r) mtp_positions_[r] = p0 + r;
        model::mtp::MtpCall call;
        call.m = rows;
        call.positions = mtp_positions_.data();
        call.next_ids = next;
        call.hc_in = hc;
        call.hc_out = hc;
        call.draft_ids = proposals + 1;
        call.last_only = true;
        call.cls = cls;
        if (rope != nullptr) call.rope = *rope;
        call.end_kv = end_kv;
        st = model::mtp::forward(model.weights_, ws, state, call, stream_);
        // The decode steps (layout::draft_decode_steps) continue from the extend's last row with its own hc row and
        // the previous step's draft, at position + rope_delta.
        call.rope = model::Rope{};
        call.end_kv = nullptr;
        model::bf16 *row = hc + uint64_t(rows - 1) * model::kHcHidden;
        for (uint64_t position = p0 + rows; st == CERE_OK && count < 1 + steps; ++position, ++count) {
            call.m = 1;
            call.cls = 0;
            call.positions = &position;
            call.next_ids = proposals + count;
            call.hc_in = row;
            call.hc_out = row;
            call.draft_ids = proposals + count + 1;
            st = model::mtp::forward(model.weights_, ws, state, call, stream_);
        }
    }
    session.proposals_[0] = count;
    cudaError_t e = cudaSuccess;
    if (st == CERE_OK) e = cudaMemcpyAsync(proposals, &session.proposals_[0], sizeof(uint32_t),
                                           cudaMemcpyHostToDevice, stream_);
    if (st == CERE_OK && e == cudaSuccess) {
        e = cudaMemcpyAsync(session.proposals_, proposals, sizeof(session.proposals_), cudaMemcpyDeviceToHost,
                            stream_);
    }
    // Drained even after an error, so no launch still reads the staging or the hc rows.
    const cudaError_t sync = cudaStreamSynchronize(stream_);
    if (e == cudaSuccess) e = sync;
    if (st == CERE_OK && e != cudaSuccess) st = fail(e);
    // Taken (and cleared) whatever the status: a K9 launch clamped a device position (model::take_bound_fault).
    if (const cere_status bound = model::take_bound_fault(ws); st == CERE_OK) st = bound;
    if (st != CERE_OK) {
        session.proposals_[0] = 0;
        session.broken_ = true;
    }
    return st;
}

bool CudaDevice::draft_rows_fit(const CudaModel &model, CudaSession *const *sessions, const uint32_t *commit,
                                uint32_t count) const {
    const model::Workspace &ws = model.workspace_;
    if (count < 2 || count > ws.mtp.rows || count > kernels::GemmPlan::kMaxRows || ws.hc_rows == nullptr ||
        uint64_t(kDraftAt) + uint64_t(kProposalWords) * count > ws.max_tokens) {
        return false;
    }
    uint32_t rows = 0;
    for (uint32_t p = 0; p < count; ++p) {
        const model::SessionState &s = sessions[p]->state_;
        // Both decode steps run (draft's eager loop stops at the end of the context).
        if (s.position + model::layout::kDraftSteps > s.max_context_tokens) return false;
        rows += commit[p];
    }
    return rows <= ws.mtp.rows && rows <= kExtendAt;
}

cere_status CudaDevice::draft_rows(CudaModel &model, CudaSession *const *sessions, const uint32_t *commit,
                                   const uint32_t *base, uint32_t count) {
    model::Workspace &ws = model.workspace_;
    constexpr uint64_t kHcRowBytes = model::kHcHidden * sizeof(model::bf16);
    uint32_t *ids = reinterpret_cast<uint32_t *>(ws.ids32);
    uint32_t *drafts = ids + kDraftAt;
    const uint32_t *host = reinterpret_cast<const uint32_t *>(ws.host_ids32) + kDraftAt;
    model::bf16 *extend = ws.hc_rows + uint64_t(kExtendAt) * model::kHcHidden;
    model::SessionState *states[kernels::GemmPlan::kMaxRows];
    uint64_t p0s[kernels::GemmPlan::kMaxRows];
    for (uint32_t p = 0; p < count; ++p) {
        states[p] = &sessions[p]->state_;
        p0s[p] = states[p]->position - commit[p];
    }
    cere_status st = CERE_OK;
    cudaError_t e = cudaSuccess;
    // Each session's verify hc rows (at base[p]) -> contiguous at kExtendAt, one kernel per k1::kMaxCopySegments.
    for (uint32_t p0 = 0, at = 0; p0 < count && e == cudaSuccess; p0 += uint32_t(k1::kMaxCopySegments)) {
        k1::CopySegment copies[k1::kMaxCopySegments];
        const uint32_t n = std::min(count - p0, uint32_t(k1::kMaxCopySegments));
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t p = p0 + i;
            copies[i] = k1::CopySegment{extend + uint64_t(at) * model::kHcHidden, kHcRowBytes,
                                        ws.hc_rows + uint64_t(base[p]) * model::kHcHidden, kHcRowBytes, kHcRowBytes,
                                        commit[p]};
            at += commit[p];
        }
        e = k1::copy_segments(copies, int(n), stream_);
    }
    if (e == cudaSuccess) {
        st = model::mtp::draft_rows(model.weights_, ws, states, p0s, commit, count, ids + kExtendAt, extend,
                                    ws.hc_rows + uint64_t(kDraftAt) * model::kHcHidden, drafts, stream_);
    }
    // Column p of the proposals -> session p's (one kernel per k1::kMaxCopySegments sessions), then every column to
    // the staging: one drain for all the sessions.
    for (uint32_t p0 = 0; p0 < count && st == CERE_OK && e == cudaSuccess; p0 += uint32_t(k1::kMaxCopySegments)) {
        k1::CopySegment copies[k1::kMaxCopySegments];
        const uint32_t n = std::min(count - p0, uint32_t(k1::kMaxCopySegments));
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t p = p0 + i;
            copies[i] = k1::CopySegment{states[p]->proposals, sizeof(uint32_t), drafts + p, count * sizeof(uint32_t),
                                        sizeof(uint32_t), kProposalWords};
        }
        e = k1::copy_segments(copies, int(n), stream_);
    }
    if (st == CERE_OK && e == cudaSuccess) {
        e = cudaMemcpyAsync(ws.host_ids32 + kDraftAt, drafts, kProposalWords * count * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, stream_);
    }
    // Drained even after an error, so no launch still reads the staging or the hc rows.
    const cudaError_t sync = cudaStreamSynchronize(stream_);
    if (e == cudaSuccess) e = sync;
    if (st == CERE_OK && e != cudaSuccess) st = fail(e);
    if (const cere_status bound = model::take_bound_fault(ws); st == CERE_OK) st = bound;  // as draft
    for (uint32_t p = 0; p < count; ++p) {
        CudaSession &s = *sessions[p];
        for (uint32_t j = 0; j < kProposalWords; ++j) s.proposals_[j] = st == CERE_OK ? host[j * count + p] : 0;
        if (st != CERE_OK) s.broken_ = true;
    }
    return st;
}

cere_status CudaDevice::stage_to_host(const std::vector<model::StateCopy> &copies, uint8_t *dst) {
    Lease lease(*this);
    Slot &s = lease.slot();
    // Two halves alternate: the DMA engine gathers runs into one while the CPU copies the other, landed, into `dst`.
    size_t begin_copy[2] = {0, 0};      // where each half's bytes start in the copies' stream
    uint64_t begin_offset[2] = {0, 0};
    uint64_t filled[2] = {0, 0};        // bytes gathered into each half and not yet copied out
    cudaError_t error = cudaSuccess;
    const auto land = [&](int h) {
        if (filled[h] == 0 || error != cudaSuccess) return;
        if ((error = cudaEventSynchronize(s.done[h])) != cudaSuccess) return;
        const uint8_t *stage = s.host + uint64_t(h) * kHalfBytes;
        for_runs(copies, begin_copy[h], begin_offset[h], filled[h], [&](const model::StateCopy &c, uint64_t at,
                                                                        uint64_t n) {
            std::memcpy(dst + c.image_offset + at, stage, n);
            stage += n;
        });
        filled[h] = 0;
    };
    int half = 0;
    uint64_t fill = 0;
    for (size_t i = 0; i < copies.size() && error == cudaSuccess; ++i) {
        const model::StateCopy &c = copies[i];
        for (uint64_t off = 0; off < c.bytes;) {
            if (fill == kHalfBytes) {
                if ((error = cudaEventRecord(s.done[half], stream_)) != cudaSuccess) break;
                filled[half] = fill;
                half ^= 1;
                fill = 0;
                land(half);  // while the half just recorded is still gathering
                if (error != cudaSuccess) break;
            }
            if (fill == 0) {
                begin_copy[half] = i;
                begin_offset[half] = off;
            }
            const uint64_t n = std::min(c.bytes - off, kHalfBytes - fill);
            uint8_t *stage = s.host + uint64_t(half) * kHalfBytes + fill;
            if ((error = cudaMemcpyAsync(stage, c.device + off, n, cudaMemcpyDeviceToHost, stream_)) != cudaSuccess) {
                break;
            }
            fill += n;
            off += n;
        }
    }
    if (error == cudaSuccess && fill != 0 && (error = cudaEventRecord(s.done[half], stream_)) == cudaSuccess) {
        filled[half] = fill;
    }
    land(half ^ 1);
    land(half);
    // Drained even after an error, so the slot is never reused while a copy still writes it.
    const cudaError_t sync = cudaStreamSynchronize(stream_);
    if (error == cudaSuccess) error = sync;
    return error == cudaSuccess ? CERE_OK : fail(error);
}

cere_status CudaDevice::stage_to_device(const std::vector<model::StateCopy> &copies, const uint8_t *src) {
    Lease lease(*this);
    Slot &s = lease.slot();
    // Two halves alternate: the CPU fills one while the DMA engine scatters the other.
    bool pending[2] = {false, false};
    int half = 0;
    uint64_t fill = 0;
    cudaError_t error = cudaSuccess;
    for (const model::StateCopy &c : copies) {
        for (uint64_t off = 0; off < c.bytes;) {
            if (fill == kHalfBytes) {
                if ((error = cudaEventRecord(s.done[half], stream_)) != cudaSuccess) break;
                pending[half] = true;
                half ^= 1;
                fill = 0;
            }
            if (fill == 0 && pending[half]) {
                if ((error = cudaEventSynchronize(s.done[half])) != cudaSuccess) break;
                pending[half] = false;
            }
            const uint64_t n = std::min(c.bytes - off, kHalfBytes - fill);
            uint8_t *stage = s.host + uint64_t(half) * kHalfBytes + fill;
            std::memcpy(stage, src + c.image_offset + off, n);
            if ((error = cudaMemcpyAsync(c.device + off, stage, n, cudaMemcpyHostToDevice, stream_)) != cudaSuccess) {
                break;
            }
            fill += n;
            off += n;
        }
        if (error == cudaSuccess && c.zero_bytes != 0) {
            error = cudaMemsetAsync(c.device + c.bytes, 0, c.zero_bytes, stream_);
        }
        if (error != cudaSuccess) break;
    }
    // Drained even after an error, so the slot is never reused while a copy still reads it.
    const cudaError_t sync = cudaStreamSynchronize(stream_);
    if (error == cudaSuccess) error = sync;
    return error == cudaSuccess ? CERE_OK : fail(error);
}

cere_status CudaDevice::save_state(host::DeviceModel &model, host::DeviceSession &session,
                                   const host::snapshot::Plan &plan, uint64_t position, uint8_t *dst) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (s.broken_ || s.state_.position != position) return CERE_ERR_INTERNAL;
    if (s.prompt_open_) return CERE_ERR_ARGUMENT;  // mid-prompt (M2.6 chunked prefill): no decode-ready state
    return save_call_end_state(m, s, plan, position, dst);
}

cere_status CudaDevice::save_call_end_state(CudaModel &m, CudaSession &s, const host::snapshot::Plan &plan,
                                            uint64_t position, uint8_t *dst) {
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (s.broken_ || s.state_.position != position) return CERE_ERR_INTERNAL;
    if (dst == nullptr || s.pages_.empty()) return CERE_ERR_ARGUMENT;
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    std::vector<model::StateCopy> copies;
    if (const cere_status st =
            model::save_copies(m.weights_.config, s.state_, s.pages_.data(), s.pages_.size(), plan, 0, &copies);
        st != CERE_OK) {
        return st;
    }
    // Every byte of `dst` is written by the CPU before return, so none lands once the host frees its arena range.
    return stage_to_host(copies, dst);
}

cere_status CudaDevice::load_state(host::DeviceModel &model, host::DeviceSession &session,
                                   const host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (s.broken_) return CERE_ERR_INTERNAL;
    if (s.state_.position != 0 || s.pages_.empty()) return CERE_ERR_ARGUMENT;
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    std::vector<model::StateCopy> copies;
    cere_status st = src == nullptr ? CERE_ERR_ARGUMENT
                                    : model::load_copies(m.weights_.config, s.state_, s.pages_.data(), s.pages_.size(),
                                                         plan, 0, position, &copies);
    if (st == CERE_OK) st = stage_to_device(copies, src);
    // A drafting plan's kind 10 carries the pending proposals: the host copy follows the restored slot.
    if (st == CERE_OK && s.state_.proposals != nullptr) {
        if (const cudaError_t e = cudaMemcpyAsync(s.proposals_, s.state_.proposals, sizeof(s.proposals_),
                                                  cudaMemcpyDeviceToHost, stream_);
            e != cudaSuccess) {
            st = fail(e);
        } else if (const cudaError_t sync = cudaStreamSynchronize(stream_); sync != cudaSuccess) {
            st = fail(sync);
        } else if (s.proposals_[0] > 3) {
            st = CERE_ERR_ARGUMENT;  // not a count this executor writes
        }
    }
    if (st == CERE_OK) {
        s.state_.position = position;
        return CERE_OK;
    }
    s.proposals_[0] = 0;
    // The session stays fresh: a partial load is undone by a reset, which zeroes the slot map, so the map is
    // rewritten from the session's pages as create_session does; a session that cannot be reset is unusable.
    s.state_.position = 0;
    if (model::session_reset(m.weights_.config, &s.state_, stream_) != CERE_OK ||
        model::session_set_pages(&s.state_, s.pages_.data(), s.pages_.size(), m.layout_.pool_tokens, stream_) !=
            CERE_OK ||
        cudaStreamSynchronize(stream_) != cudaSuccess) {
        cudaGetLastError();
        s.broken_ = true;
    }
    return st;
}

// ---- M2.10 prefix cache ----

namespace {

// Every id of ids[0, n) is an allocated page of the pool (one with a reference).
bool pages_valid(const host::PagePool &pool, const uint32_t *ids, uint64_t n) {
    if (ids == nullptr && n != 0) return false;
    for (uint64_t i = 0; i < n; ++i) {
        if (ids[i] >= pool.pages() || pool.refs(ids[i]) == 0) return false;
    }
    return true;
}

// The first_row of a prefix plan (D9: KV rows [first_row, position) in kinds 1-3): from kind 1's size, attention
// layers x rows x kv_heads * head_dim FP8 bytes. plan_copies then checks every section's size against it.
bool prefix_first_row(const cere_qwen4_exp_config &cfg, const host::snapshot::Plan &plan, uint64_t position,
                      uint64_t *out) {
    uint64_t attention = 0;
    for (uint32_t l = 0; l < cfg.num_hidden_layers && l < 64; ++l) attention += model::layout::full_attention(cfg, l);
    const uint64_t row = attention * cfg.num_key_value_heads * cfg.head_dim;
    for (const host::snapshot::Section &s : plan.sections) {
        if (s.kind != host::snapshot::kAttnK) continue;
        if (row == 0 || s.bytes % row != 0 || s.bytes / row > position) return false;
        *out = position - s.bytes / row;
        return true;
    }
    return false;
}

// The copies of a prefix node's sections (model::bind_fixed_state over `slot` and `pages`), split into the runs the
// ring moves and the kind 8 output counts (device null: a node has none; zeros in the image). The pages holding rows
// [first_row, position) must be allocated (else CERE_ERR_ARGUMENT); entries below first_row / 64 are not read.
cere_status prefix_copies(CudaModel &m, const model::KvPool &kv, const cere_qwen4_exp_config &cfg,
                          const uint32_t *pages, uint64_t n_pages, uint32_t slot, const host::snapshot::Plan &plan,
                          uint64_t position, bool load, std::vector<model::StateCopy> *runs,
                          std::vector<model::StateCopy> *zeros) {
    uint64_t first_row = 0;
    if (pages == nullptr || slot >= m.layout().state_slots || n_pages == 0 ||
        n_pages > UINT64_MAX / host::kKvPageTokens || position > n_pages * host::kKvPageTokens ||
        !prefix_first_row(cfg, plan, position, &first_row)) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint64_t i = first_row / host::kKvPageTokens; i < host::kv_pages(position); ++i) {
        if (m.page_refs(pages[i]) == 0) return CERE_ERR_ARGUMENT;
    }
    model::SessionState view;
    if (const cere_status st = model::bind_fixed_state(cfg, m.state_slot(slot), n_pages, kv, &view); st != CERE_OK) {
        return st;
    }
    std::vector<model::StateCopy> copies;
    if (load) {
        if (const cere_status st = model::load_copies(cfg, view, pages, n_pages, plan, first_row, position, &copies);
            st != CERE_OK) {
            return st;
        }
    } else {
        view.position = position;
        if (const cere_status st = model::save_copies(cfg, view, pages, n_pages, plan, first_row, &copies);
            st != CERE_OK) {
            return st;
        }
    }
    for (const model::StateCopy &c : copies) (c.device == nullptr ? zeros : runs)->push_back(c);
    return CERE_OK;
}

}  // namespace

uint64_t CudaDevice::free_pages(const host::DeviceModel &model) const {
    return static_cast<const CudaModel &>(model).free_pages();
}

uint32_t CudaDevice::page_refs(const host::DeviceModel &model, uint32_t id) const {
    return static_cast<const CudaModel &>(model).page_refs(id);
}

cere_status CudaDevice::retain_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) {
    if (ids == nullptr && n != 0) return CERE_ERR_ARGUMENT;
    auto &m = static_cast<CudaModel &>(model);
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    return m.pool_.retain(ids, n);
}

cere_status CudaDevice::release_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) {
    if (ids == nullptr && n != 0) return CERE_ERR_ARGUMENT;
    auto &m = static_cast<CudaModel &>(model);
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    return m.pool_.release_pages(ids, n);
}

cere_status CudaDevice::take_pages(host::DeviceModel &model, uint64_t n, std::vector<uint32_t> *out) {
    auto &m = static_cast<CudaModel &>(model);
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    if (n > m.pool_.free_pages()) return CERE_ERR_CAPACITY;
    try {
        out->reserve(out->size() + n);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return m.pool_.take_pages(n, out);
}

cere_status CudaDevice::bind_pages(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *prefix,
                                   uint64_t n_prefix) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (s.broken_) return CERE_ERR_INTERNAL;
    const uint64_t n_pages = host::kv_pages(s.max_context_tokens_);
    if (!s.lazy_ || !s.pages_.empty() || s.state_.position != 0 || n_prefix > n_pages) return CERE_ERR_ARGUMENT;
    std::vector<uint32_t> pages;
    try {
        pages.reserve(n_pages);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    {
        // One critical section: the shared pages gain the session's reference and its own pages are taken (lowest
        // free first) together, so CAPACITY keeps nothing.
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        if (!pages_valid(m.pool_, prefix, n_prefix)) return CERE_ERR_ARGUMENT;
        pages.assign(prefix, prefix + n_prefix);
        if (const cere_status st = m.pool_.take_pages(n_pages - n_prefix, &pages); st != CERE_OK) return st;
        if (const cere_status st = m.pool_.retain(prefix, n_prefix); st != CERE_OK) {
            host::release_pages_logged(m.pool_, pages.data() + n_prefix, pages.size() - n_prefix, "cuda_bind_pages");
            return st;
        }
    }
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) {
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        host::release_pages_logged(m.pool_, pages.data(), pages.size(), "cuda_bind_pages");
        return fail(e);
    }
    // Stream order puts the map before the session's first step or load_slot.
    if (const cere_status st =
            model::session_set_pages(&s.state_, pages.data(), pages.size(), m.layout_.pool_tokens, stream_);
        st != CERE_OK) {
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        host::release_pages_logged(m.pool_, pages.data(), pages.size(), "cuda_bind_pages");
        return st;
    }
    s.pages_ = std::move(pages);
    return CERE_OK;
}

std::vector<uint32_t> CudaDevice::session_page_list(const host::DeviceSession &session) const {
    return static_cast<const CudaSession &>(session).pages_;
}

uint64_t CudaDevice::session_position(const host::DeviceSession &session) const {
    return static_cast<const CudaSession &>(session).state_.position;
}

uint64_t CudaDevice::prefill_chunk(const host::DeviceModel &model) const {
    const auto &m = static_cast<const CudaModel &>(model);
    const uint64_t rows = m.workspace_.max_tokens;
    return (m.prefill_chunk_ == 0 || m.prefill_chunk_ > rows) ? rows : m.prefill_chunk_;
}

bool CudaDevice::prefill_selection(const host::DeviceModel &model, std::string &out) const {
    const auto &m = static_cast<const CudaModel &>(model);
    if (!m.instantiated_) return false;
    model::prefill_sig::PrefillSelection selection;
    if (model::prefill_sig::collect(m.workspace_, &selection) != CERE_OK) {
        throw std::runtime_error("cere: prefill selection: cuBLASLt algorithm attribute read failed");
    }
    out += model::prefill_sig::selection_text(selection);
    return true;
}

uint32_t CudaDevice::end_kv_slots(const host::DeviceModel &model, uint64_t rows, uint64_t cls) const {
    const auto &m = static_cast<const CudaModel &>(model);
    if (rows == 0 || cls <= rows || cls > uint64_t(k6::kMaxContext) ||
        m.workspace_.max_context_tokens <= uint64_t(k6::kMaxContext) || m.layout_.state_bytes == 0) {
        return 0;
    }
    const uint64_t layers = model::end_kv_index(m.weights_.config, model::kLayers) + (m.weights_.has_draft ? 1 : 0);
    const uint64_t bytes = model::end_kv_bytes(layers, rows);
    return uint32_t((bytes + m.layout_.state_bytes - 1) / m.layout_.state_bytes);
}

uint64_t CudaDevice::row_sig(const host::DeviceModel &model, uint64_t s, uint64_t cls) const {
    const auto &m = static_cast<const CudaModel &>(model);
    const host::Levers &l = m.levers_;
    if (cls < kRowSigMinRows || (l.k2_gemm != CERE_LEVER_K2_CUBLASLT && l.k2_gemm != CERE_LEVER_K2_SMALL_M) ||
        l.prefill_attention != CERE_LEVER_PREFILL_CUDA_CORE || l.k2_weights != CERE_LEVER_K2_WEIGHTS_BF16) {
        return cls;
    }
    // Bit 63 keeps a signature apart from every class; bit 62 the dense chunk-0 call; bits 0-7 and 8-15 the split
    // range (index + 1, GemmPlan::kMaxPins <= 255) of each pinned plan.
    uint64_t sig = uint64_t{1} << 63;
    if (s == 0 && cls <= uint64_t(k6::kMaxContext)) sig |= uint64_t{1} << 62;
    uint32_t shift = 0;
    for (const uint32_t n : {model::kSharedInter, model::kQkvWidth}) {
        const kernels::GemmPins *pins = kernels::gemm_pins_for(n, model::kHidden, kernels::GemmOut::kBF16);
        for (uint32_t i = 0; pins != nullptr && i < pins->count; ++i) {
            const kernels::GemmPinRange &r = pins->ranges[i];
            if (cls >= r.m_lo && cls <= r.m_hi && r.config.splitk > 1) sig |= uint64_t(i + 1) << shift;
        }
        shift += 8;
    }
    return sig;
}

cere_status CudaDevice::capture_state(host::DeviceModel &model, const host::DeviceSession &session, uint32_t slot) {
    auto &m = static_cast<CudaModel &>(model);
    const auto &s = static_cast<const CudaSession &>(session);
    if (&s.model_ != &m || slot >= m.layout_.state_slots) return CERE_ERR_ARGUMENT;
    if (s.broken_) return CERE_ERR_INTERNAL;
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    const uint8_t *run = s.state_.slot + model::fixed_state_offset(m.weights_.config, s.state_.max_context_tokens);
    if (const cudaError_t e =
            cudaMemcpyAsync(m.state_slot(slot), run, m.layout_.state_bytes, cudaMemcpyDeviceToDevice, stream_);
        e != cudaSuccess) {
        return fail(e);
    }
    return CERE_OK;
}

cere_status CudaDevice::load_slot(host::DeviceModel &model, host::DeviceSession &session, uint32_t slot,
                                  uint64_t position) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m || slot >= m.layout_.state_slots) return CERE_ERR_ARGUMENT;
    if (s.broken_) return CERE_ERR_INTERNAL;
    if (s.pages_.empty() || s.state_.position != 0 || s.prompt_open_ || position == 0 ||
        position % host::kKvPageTokens != 0 || position >= s.state_.max_context_tokens) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    uint8_t *run = s.state_.slot + model::fixed_state_offset(m.weights_.config, s.state_.max_context_tokens);
    if (const cudaError_t e =
            cudaMemcpyAsync(run, m.state_slot(slot), m.layout_.state_bytes, cudaMemcpyDeviceToDevice, stream_);
        e != cudaSuccess) {
        // Nothing was enqueued: the session is still fresh.
        return fail(e);
    }
    // The state of a session that consumed `position` prompt tokens in calls ending at a capture point: the slot map
    // was written at bind_pages and out_counts are still zero (no sampled step), so only the position and the open
    // prompt remain (D3′).
    s.state_.position = position;
    s.prompt_open_ = true;
    return CERE_OK;
}

cere_status CudaDevice::save_prefix(host::DeviceModel &model, const uint32_t *pages, uint64_t n_pages, uint32_t slot,
                                    const host::snapshot::Plan &plan, uint64_t position, uint8_t *dst) {
    auto &m = static_cast<CudaModel &>(model);
    if (dst == nullptr) return CERE_ERR_ARGUMENT;
    std::vector<model::StateCopy> runs;
    std::vector<model::StateCopy> zeros;
    try {
        if (const cere_status st = prefix_copies(m, m.kv_, m.weights_.config, pages, n_pages, slot, plan, position,
                                                 false, &runs, &zeros);
            st != CERE_OK) {
            return st;
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    for (const model::StateCopy &c : zeros) std::memset(dst + c.image_offset, 0, c.bytes);
    // Stream order puts the slot's last capture before the gather; drained before return like save_state.
    return stage_to_host(runs, dst);
}

cere_status CudaDevice::load_prefix(host::DeviceModel &model, const uint32_t *pages, uint64_t n_pages, uint32_t slot,
                                    const host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) {
    auto &m = static_cast<CudaModel &>(model);
    if (src == nullptr) return CERE_ERR_ARGUMENT;
    std::vector<model::StateCopy> runs;
    std::vector<model::StateCopy> zeros;  // kind 8: a node has no output counts, nothing to load
    try {
        if (const cere_status st = prefix_copies(m, m.kv_, m.weights_.config, pages, n_pages, slot, plan, position,
                                                 true, &runs, &zeros);
            st != CERE_OK) {
            return st;
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    return stage_to_device(runs, src);
}

cere_status CudaDevice::step_rows(host::DeviceModel &model, const host::RowArgs *rows, uint32_t row_count,
                                  cere_token_score *out, cere_status *row_status) {
    auto &m = static_cast<CudaModel &>(model);
    // Malformed call: nothing runs, row_status untouched.
    if (rows == nullptr || out == nullptr || row_status == nullptr || row_count == 0 ||
        row_count > m.layout_.max_sessions || row_count > kernels::GemmPlan::kMaxRows) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t r = 0; r < row_count; ++r) {
        const host::DeviceSession *s = rows[r].session;
        if (s == nullptr || &static_cast<const CudaSession *>(s)->model_ != &m) return CERE_ERR_ARGUMENT;
        if (rows[r].sampling == nullptr && rows[r].logprobs != nullptr) return CERE_ERR_ARGUMENT;
        for (uint32_t q = 0; q < r; ++q) {
            if (rows[q].session == s) return CERE_ERR_ARGUMENT;
        }
    }
    if (!m.instantiated_) return CERE_ERR_ARGUMENT;
    // Per-row preconditions (D7), before any launch; the rows still OK are the participants.
    const uint32_t vocab = m.weights_.config.vocab_size;
    CudaSession *sessions[kernels::GemmPlan::kMaxRows];
    uint32_t tokens[kernels::GemmPlan::kMaxRows];
    uint32_t index[kernels::GemmPlan::kMaxRows];  // participant p is row index[p]
    const model::SamplerArgs *samplers[kernels::GemmPlan::kMaxRows];
    cere_token_logprobs *logprobs[kernels::GemmPlan::kMaxRows];
    uint32_t participants = 0;
    for (uint32_t r = 0; r < row_count; ++r) {
        auto &s = static_cast<CudaSession &>(*rows[r].session);
        cere_status st = CERE_OK;
        if (s.broken_) {
            st = CERE_ERR_INTERNAL;
        } else if (s.state_.position >= s.state_.max_context_tokens) {
            st = CERE_ERR_CAPACITY;
        } else if (s.state_.position == 0 || s.prompt_open_ || rows[r].token >= vocab) {
            st = CERE_ERR_ARGUMENT;
        } else if (rows[r].sampling != nullptr) {
            st = model::check_sampler(*rows[r].sampling, vocab);  // ledger CR4: the row's own sampler preconditions
        }
        row_status[r] = st;
        if (st != CERE_OK) continue;
        sessions[participants] = &s;
        tokens[participants] = rows[r].token;
        index[participants] = r;
        samplers[participants] = rows[r].sampling;
        logprobs[participants] = rows[r].logprobs;
        ++participants;
    }
    if (participants == 0) return CERE_OK;
    if (participants == 1) {
        // One row is today's single-row DECODE forward (A1), with the row's sampling.
        const uint32_t r = index[0];
        auto &s = static_cast<CudaSession &>(*rows[r].session);
        const cere_status st =
            forward(m, s, CERE_OP_DECODE, &tokens[0], 1, nullptr, &out[r], samplers[0], logprobs[0]);
        if (st != CERE_OK) {
            s.broken_ = true;
            row_status[r] = st;
        }
        return st;
    }
    cere_token_score scores[kernels::GemmPlan::kMaxRows];
    const cere_status st = forward_rows(m, sessions, participants, tokens, nullptr, scores, samplers, logprobs);
    for (uint32_t p = 0; p < participants; ++p) {
        const uint32_t r = index[p];
        if (st == CERE_OK) {
            out[r] = scores[p];
        } else {
            static_cast<CudaSession &>(*rows[r].session).broken_ = true;
            row_status[r] = st;
        }
    }
    return st;
}

cere_status CudaDevice::set_session_draft(host::DeviceModel &model, host::DeviceSession &session, bool on) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (on && !draft_supported(m)) return CERE_ERR_UNSUPPORTED;
    if (s.state_.position != 0 || s.prompt_open_) return CERE_ERR_ARGUMENT;
    s.drafting_ = on;
    return CERE_OK;
}

uint64_t CudaDevice::lora_bytes(const host::DeviceModel &model, uint32_t adapters, uint32_t rank) const {
    const auto &m = static_cast<const CudaModel &>(model);
    const uint64_t per_rank = host::lora::elements_per_rank(m.weights_.config);
    if (per_rank == 0 || m.workspace_.max_tokens == 0) return 0;
    return kernels::lora::Table::bytes(adapters, host::lora::padded_rank(rank), per_rank,
                                       uint32_t(m.workspace_.max_tokens));
}

cere_status CudaDevice::lora_reserve(host::DeviceModel &model, uint32_t adapters, uint32_t rank) {
    auto &m = static_cast<CudaModel &>(model);
    if (m.lora_ != nullptr || m.instantiated_) return CERE_ERR_ARGUMENT;
    const cere_qwen4_exp_config &cfg = m.weights_.config;
    const uint32_t padded = host::lora::padded_rank(rank);
    // Each target's A [R, in] then B [out, R], in ascending target id (host::lora::targets).
    std::vector<kernels::lora::TargetLayout> layouts;
    std::vector<int32_t> index(size_t(cfg.num_hidden_layers) * CERE_LORA_MODULES, -1);
    uint64_t offset = 0;
    for (const uint32_t t : host::lora::targets(cfg)) {
        host::lora::Dims d;
        host::lora::dims(cfg, t, &d);
        index[t] = int32_t(layouts.size());
        layouts.push_back(kernels::lora::TargetLayout{offset, offset + uint64_t{padded} * d.in, d.in, d.out});
        offset += uint64_t{padded} * (uint64_t{d.in} + d.out);
    }
    // The weights the plans run: the packed q|k|v (three targets side by side), o_proj, the shared gate / up / down.
    std::vector<kernels::lora::Entry> entries;
    const auto single = [&](const void *w, uint32_t target) -> bool {
        if (w == nullptr || index[target] < 0) return false;
        const kernels::lora::TargetLayout &l = layouts[size_t(index[target])];
        kernels::lora::Entry e;
        e.w = w;
        e.k = l.k;
        e.segments = 1;
        e.seg[0] = kernels::lora::Segment{uint32_t(index[target]), 0, l.n};
        entries.push_back(e);
        return true;
    };
    for (uint32_t layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        const model::LayerWeights &lw = m.weights_.layers[layer];
        const uint32_t base = layer * CERE_LORA_MODULES;
        if (lw.full_attention) {
            const int32_t q = index[base + CERE_LORA_Q_PROJ], k = index[base + CERE_LORA_K_PROJ],
                          v = index[base + CERE_LORA_V_PROJ];
            if (lw.attn.qkv == nullptr || q < 0 || k < 0 || v < 0) return CERE_ERR_INTERNAL;
            kernels::lora::Entry e;
            e.w = lw.attn.qkv;
            e.k = layouts[size_t(q)].k;
            e.segments = 3;
            uint32_t col = 0;
            for (uint32_t s = 0; s < 3; ++s) {
                const int32_t i = s == 0 ? q : s == 1 ? k : v;
                e.seg[s] = kernels::lora::Segment{uint32_t(i), col, layouts[size_t(i)].n};
                col += layouts[size_t(i)].n;
            }
            entries.push_back(e);
            if (!single(lw.attn.o_proj, base + CERE_LORA_O_PROJ)) return CERE_ERR_INTERNAL;
        }
        if (!single(lw.moe.shared_gate, base + CERE_LORA_SHARED_GATE) ||
            !single(lw.moe.shared_up, base + CERE_LORA_SHARED_UP) ||
            !single(lw.moe.shared_down, base + CERE_LORA_SHARED_DOWN)) {
            return CERE_ERR_INTERNAL;
        }
    }
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    auto table = std::make_unique<kernels::lora::Table>();
    if (const cere_status st = table->init(adapters, padded, uint32_t(m.workspace_.max_tokens), std::move(layouts),
                                           std::move(entries));
        st != CERE_OK) {
        return st;
    }
    m.lora_held_.hold(ledger_, host::Held::kDevice, lora_bytes(m, adapters, rank));
    m.lora_ = std::move(table);
    m.lora_index_ = std::move(index);
    m.workspace_.lora = m.lora_.get();
    return CERE_OK;
}

cere_status CudaDevice::lora_load(host::DeviceModel &model, uint32_t slot, uint32_t rank, float scale, uint64_t count,
                                  const uint32_t *targets, const void *const *a, const void *const *b) {
    auto &m = static_cast<CudaModel &>(model);
    if (m.lora_ == nullptr) return CERE_ERR_UNSUPPORTED;
    std::vector<uint32_t> index(count);
    for (uint64_t i = 0; i < count; ++i) {
        if (targets[i] >= m.lora_index_.size() || m.lora_index_[targets[i]] < 0) return CERE_ERR_ARGUMENT;
        index[i] = uint32_t(m.lora_index_[targets[i]]);
    }
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    return m.lora_->load(slot, rank, scale, count, index.data(), a, b);
}

cere_status CudaDevice::lora_unload(host::DeviceModel &model, uint32_t slot) {
    auto &m = static_cast<CudaModel &>(model);
    if (m.lora_ == nullptr) return CERE_ERR_UNSUPPORTED;
    m.lora_->unload(slot);
    return CERE_OK;
}

cere_status CudaDevice::set_session_lora(host::DeviceModel &model, host::DeviceSession &session, int32_t slot) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (m.lora_ == nullptr) return CERE_ERR_UNSUPPORTED;
    if (s.state_.position != 0 || s.prompt_open_) return CERE_ERR_ARGUMENT;
    s.state_.lora_slot = slot;
    return CERE_OK;
}

cere_status CudaDevice::research_reserve(host::DeviceModel &model, uint64_t staging_bytes, uint64_t plan_bytes,
                                         uint64_t host_bytes, uint8_t **ring) {
    auto &m = static_cast<CudaModel &>(model);
    if (m.research_ != nullptr || m.instantiated_ || ring == nullptr) return CERE_ERR_ARGUMENT;
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    auto staging = std::make_unique<research::Staging>();
    if (const cere_status st = staging->allocate(staging_bytes, plan_bytes, host_bytes); st != CERE_OK) return st;
    m.research_device_held_.hold(ledger_, host::Held::kDevice, staging_bytes + plan_bytes);
    m.research_ring_held_.hold(ledger_, host::Held::kRing, host_bytes);
    *ring = staging->ring();
    m.research_ = std::move(staging);
    m.workspace_.research = m.research_.get();
    return CERE_OK;
}

cere_status CudaDevice::research_plan_load(host::DeviceModel &model, uint32_t slot, const research::Plan &plan) {
    auto &m = static_cast<CudaModel &>(model);
    if (m.research_ == nullptr) return CERE_ERR_UNSUPPORTED;
    if (uint64_t(slot + 1) * CERE_RESEARCH_MAX_PLAN_BYTES > m.research_->plan_bytes() ||
        plan.operand_bytes > CERE_RESEARCH_MAX_PLAN_BYTES) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    uint8_t *base = m.research_->plan(slot);
    const auto put = [&](uint64_t offset, const void *src, size_t bytes) -> cudaError_t {
        if (bytes == 0) return cudaSuccess;
        return cudaMemcpyAsync(base + offset, src, bytes, cudaMemcpyHostToDevice, stream_);
    };
    cudaError_t e = cudaSuccess;
    for (const research::Op &op : plan.ops) {
        if (e == cudaSuccess) e = put(op.vector_offset, op.vector.data(), op.vector.size() * sizeof(uint16_t));
        if (e == cudaSuccess) e = put(op.experts_offset, op.experts.data(), op.experts.size() * sizeof(int32_t));
        if (e == cudaSuccess) e = put(op.weights_offset, op.weights.data(), op.weights.size() * sizeof(float));
    }
    // Pageable sources: the copies are staged before they return, the sync orders them before any later step.
    const cudaError_t sync = cudaStreamSynchronize(stream_);
    if (e != cudaSuccess) return fail(e);
    return sync == cudaSuccess ? CERE_OK : fail(sync);
}

cere_status CudaDevice::set_session_research(host::DeviceModel &model, host::DeviceSession &session,
                                             const research::Binding *binding) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (m.research_ == nullptr) return CERE_ERR_UNSUPPORTED;
    if (s.state_.position != 0 || s.prompt_open_) return CERE_ERR_ARGUMENT;
    s.state_.research = binding;
    return CERE_OK;
}

cere_status CudaDevice::research_sync(host::DeviceModel &model) {
    auto &m = static_cast<CudaModel &>(model);
    if (m.research_ == nullptr) return CERE_OK;
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    const cudaError_t e = cudaStreamSynchronize(stream_);
    return e == cudaSuccess ? CERE_OK : fail(e);
}

cere_status CudaDevice::set_rope_delta(host::DeviceModel &model, host::DeviceSession &session, int64_t rope_delta) {
    auto &m = static_cast<CudaModel &>(model);
    auto &s = static_cast<CudaSession &>(session);
    if (&s.model_ != &m || rope_delta > 0) return CERE_ERR_ARGUMENT;
    s.state_.rope_delta = rope_delta;
    return CERE_OK;
}

cere_status CudaDevice::verify_chains(CudaModel &model, CudaSession *const *sessions, const model::VerifyRow *rows,
                                      uint32_t chains, model::ForwardTaps *taps, cere_token_score *out,
                                      cere_token_logprobs *out_logprobs) {
    // Called on the polling thread, whose current device is not necessarily ours.
    if (const cudaError_t e = cudaSetDevice(ordinal_); e != cudaSuccess) return fail(e);
    bool launched = false;
    const cere_status st = model::forward_verify_rows(model.weights_, model.workspace_, rows, chains, taps, out,
                                                      stream_, launched, out_logprobs);
    // Without `launched` the call returned from a precondition (nothing ran); after it any failure leaves every
    // chain's session undefined (MF-2).
    if (st != CERE_OK && launched) {
        probe_sticky();
        for (uint32_t c = 0; c < chains; ++c) sessions[c]->broken_ = true;
    }
    return st;
}

cere_status CudaDevice::commit_chains(CudaModel &model, CudaSession *const *sessions, const model::VerifyRow *rows,
                                      const uint32_t *commit, uint32_t chains) {
    cere_status st = CERE_OK;
    for (uint32_t c = 0, first = 0; c < chains && st == CERE_OK; first += rows[c].n + 1, ++c) {
        // A sampled chain's committed ids: ws.spec.committed from its first verify row (forward_verify_rows).
        const uint32_t *committed =
            model.workspace_.spec.committed == nullptr ? nullptr : model.workspace_.spec.committed + first;
        st = model::verify_commit(model.weights_.config, rows[c], commit[c], stream_, committed);
    }
    const cudaError_t sync = cudaStreamSynchronize(stream_);
    if (st == CERE_OK && sync != cudaSuccess) st = fail(sync);
    if (st != CERE_OK) {
        for (uint32_t c = 0; c < chains; ++c) sessions[c]->broken_ = true;
    }
    return st;
}

cere_status CudaDevice::verify_rows(CudaModel &model, CudaSession *const *sessions, const uint32_t *const *tokens,
                                    const uint32_t *n, uint32_t chains, model::ForwardTaps *taps,
                                    cere_token_score *out, const uint32_t *commit) {
    constexpr uint32_t kMaxRows = kernels::GemmPlan::kMaxRows;
    if (!model.instantiated_ || sessions == nullptr || tokens == nullptr || n == nullptr || out == nullptr ||
        commit == nullptr || chains == 0 || chains > kMaxRows) {
        return CERE_ERR_ARGUMENT;
    }
    if (model.drafts_ == 0) return CERE_ERR_UNSUPPORTED;
    model::VerifyRow rows[kMaxRows];
    for (uint32_t c = 0; c < chains; ++c) {
        CudaSession *s = sessions[c];
        if (s == nullptr || &s->model_ != &model || commit[c] == 0 || commit[c] > n[c] + 1) return CERE_ERR_ARGUMENT;
        if (s->broken_) return CERE_ERR_INTERNAL;
        rows[c] = model::VerifyRow{&s->state_, s->undo_, tokens[c], n[c], s->pages_.data(), s->pages_.size()};
    }
    if (const cere_status st = verify_chains(model, sessions, rows, chains, taps, out); st != CERE_OK) return st;
    return commit_chains(model, sessions, rows, commit, chains);
}

cere_status CudaDevice::speculate_rows(host::DeviceModel &model, const host::SpecRowArgs *rows, uint32_t row_count,
                                       host::SpecRowResult *out, cere_status *row_status) {
    constexpr uint32_t kMaxRows = kernels::GemmPlan::kMaxRows;
    auto &m = static_cast<CudaModel &>(model);
    // Malformed call: nothing runs, row_status untouched.
    if (rows == nullptr || out == nullptr || row_status == nullptr || row_count == 0 ||
        row_count > m.layout_.max_sessions || row_count > kMaxRows) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t r = 0; r < row_count; ++r) {
        const host::DeviceSession *s = rows[r].session;
        if (s == nullptr || &static_cast<const CudaSession *>(s)->model_ != &m) return CERE_ERR_ARGUMENT;
        if (rows[r].sampling == nullptr && rows[r].logprobs != nullptr) return CERE_ERR_ARGUMENT;
        for (uint32_t q = 0; q < r; ++q) {
            if (rows[q].session == s) return CERE_ERR_ARGUMENT;
        }
    }
    if (!m.instantiated_) return CERE_ERR_ARGUMENT;
    if (m.drafts_ == 0) return CERE_ERR_UNSUPPORTED;
    const uint32_t max_chain = m.max_chain_drafts();
    // P8 (m23 design note): one verify call holds the plan's verify_rows rows (min(16, 4 x max_sessions), the
    // n <= 16 rule), so a round of more rows runs as consecutive groups in row order, each the largest prefix whose
    // chains (draft_count + 1 rows each) fit; up to 4 rows (every plan before P8) it is one group, the call as before.
    // DraftAccept: a group holding a chain longer than kUndoSnapshots takes its chains' undo snapshots from the
    // model's pool (CudaModel::undo_pool), so it also ends before its drafts exceed the pool. Rows are independent
    // (per-row bit-identity at every m <= 16), so the grouping changes no row's tokens. A group that fails ends the
    // round: its status on its rows (speculate_group) and on every later row.
    const uint32_t capacity = uint32_t(m.layout_.verify_rows);
    const uint64_t pool = m.layout_.max_sessions * model::layout::kUndoSnapshots;
    for (uint32_t first = 0; first < row_count;) {
        uint32_t count = 0;
        bool deep = false;
        for (uint32_t used = 0, snaps = 0; first + count < row_count; ++count) {
            const uint32_t n = std::min(rows[first + count].draft_count, max_chain);
            used += n + 1;
            snaps += n;
            deep |= n > model::layout::kUndoSnapshots;
            if (count > 0 && (used > capacity || (deep && snaps > pool))) break;
        }
        if (const cere_status st = speculate_group(m, rows + first, count, out + first, row_status + first);
            st != CERE_OK) {
            for (uint32_t r = first + count; r < row_count; ++r) row_status[r] = st;
            return st;
        }
        first += count;
    }
    return CERE_OK;
}

uint32_t CudaDevice::max_chain_drafts(const host::DeviceModel &model) const {
    const auto &m = static_cast<const CudaModel &>(model);
    return m.drafts_ > 0 ? m.max_chain_drafts() : 3;
}

uint32_t CudaDevice::pending_drafts(const host::DeviceSession &session, uint32_t *ids, uint32_t max) const {
    const auto &s = static_cast<const CudaSession &>(session);
    if (!s.drafting_ || ids == nullptr) return 0;
    const uint32_t n = std::min(s.proposals_[0], max);
    std::copy(s.proposals_ + 1, s.proposals_ + 1 + n, ids);
    return n;
}

cere_status CudaDevice::speculate_group(CudaModel &m, const host::SpecRowArgs *rows, uint32_t row_count,
                                        host::SpecRowResult *out, cere_status *row_status) {
    constexpr uint32_t kMaxRows = kernels::GemmPlan::kMaxRows;
    constexpr uint32_t kMaxChain = uint32_t(model::layout::kMaxChainDrafts);
    const uint32_t max_chain = m.max_chain_drafts();
    // Per-row preconditions, before any launch; the rows still OK are the participants.
    const uint32_t vocab = m.weights_.config.vocab_size;
    CudaSession *sessions[kMaxRows];
    model::VerifyRow chains[kMaxRows];
    uint32_t tokens[kMaxRows][kMaxChain + 1];
    uint32_t index[kMaxRows];  // participant p is row index[p]
    uint32_t participants = 0;
    for (uint32_t r = 0; r < row_count; ++r) {
        const host::SpecRowArgs &a = rows[r];
        auto &s = static_cast<CudaSession &>(*a.session);
        // Caller drafts (VERIFY) or the pending MTP proposals (as many as the round asks for and the last draft left).
        const uint32_t *drafts = a.drafts != nullptr ? a.drafts : s.proposals_ + 1;
        const uint32_t n = a.drafts != nullptr ? a.draft_count : std::min(a.draft_count, s.proposals_[0]);
        const uint64_t position = s.state_.position;
        bool drafts_ok = true;
        for (uint32_t j = 0; j < n && j < max_chain; ++j) drafts_ok &= drafts[j] < vocab;
        cere_status st = CERE_OK;
        if (s.broken_) {
            st = CERE_ERR_INTERNAL;
        } else if (position >= s.state_.max_context_tokens || n >= s.state_.max_context_tokens - position) {
            st = CERE_ERR_CAPACITY;
        } else if (!s.drafting_ || position == 0 || s.prompt_open_ || a.token >= vocab || a.max_commit == 0 ||
                   a.draft_count > max_chain || a.draft_count >= a.max_commit ||
                   (a.drafts != nullptr && a.draft_count == 0) || !drafts_ok) {
            st = CERE_ERR_ARGUMENT;
        } else if (a.sampling != nullptr) {
            // The row's own sampler preconditions (as step_rows). A constrained row with drafts (GrammarSpec GS-02)
            // verifies the caller's grammar-checked drafts, at most CERE_MAX_CONSTRAINED_DRAFTS, one mask per row;
            // never its own MTP proposals.
            const bool masked = a.sampling->vocab_mask != nullptr;
            st = masked && a.draft_count != 0 && (a.drafts == nullptr || a.draft_count > CERE_MAX_CONSTRAINED_DRAFTS)
                     ? CERE_ERR_ARGUMENT
                     : model::check_sampler(*a.sampling, vocab);
        }
        row_status[r] = st;
        if (st != CERE_OK) continue;
        tokens[participants][0] = a.token;
        for (uint32_t j = 0; j < n; ++j) tokens[participants][j + 1] = drafts[j];
        chains[participants] = model::VerifyRow{&s.state_,       s.undo_,         tokens[participants],
                                                n,               s.pages_.data(), s.pages_.size(),
                                                a.sampling,      a.logprobs != nullptr};
        sessions[participants] = &s;
        index[participants] = r;
        ++participants;
    }
    if (participants == 0) return CERE_OK;
    // DraftAccept: a group with a chain longer than kUndoSnapshots verifies every chain into a contiguous range of
    // the model's undo pool (speculate_rows keeps its drafts within the pool); the snapshots are scratch of this call.
    bool deep = false;
    for (uint32_t p = 0; p < participants; ++p) deep |= chains[p].n > model::layout::kUndoSnapshots;
    if (deep) {
        model::UndoState *pool = nullptr;
        try {
            pool = m.undo_pool();
        } catch (const std::bad_alloc &) {
            for (uint32_t p = 0; p < participants; ++p) row_status[index[p]] = CERE_ERR_CAPACITY;
            return CERE_ERR_CAPACITY;
        }
        for (uint32_t p = 0, at = 0; p < participants; at += chains[p].n, ++p) chains[p].undo = pool + at;
    }

    cere_token_score scores[kMaxRows];
    cere_token_logprobs logprobs[kMaxRows];
    cere_status st = verify_chains(m, sessions, chains, participants, nullptr, scores, logprobs);
    if (st != CERE_OK) {
        for (uint32_t p = 0; p < participants; ++p) row_status[index[p]] = st;
        return st;
    }
    // The accept: k leading drafts equal the previous row's committed id (D5 greedy: the raw top-1; D6 sampled:
    // k13s::spec_accept's decision, which never resamples a rejected draft; temperature 0: the processed top-1), a =
    // k + 1 tokens (the accepted drafts and row k's id), truncated after the first stop token and at max_commit
    // (M29S2-21).
    uint32_t commit[kMaxRows];
    for (uint32_t p = 0, base = 0; p < participants; base += chains[p].n + 1, ++p) {
        const host::SpecRowArgs &a = rows[index[p]];
        const cere_token_score *row = scores + base;
        uint32_t k = 0;
        while (k < chains[p].n && row[k].id == chains[p].tokens[k + 1]) ++k;
        uint32_t c = std::min(k + 1, a.max_commit);
        for (uint32_t i = 0; a.is_stop != nullptr && i < c; ++i) {
            if (a.is_stop(a.stop_ctx, row[i].id)) c = i + 1;
        }
        commit[p] = c;
        host::SpecRowResult &o = out[index[p]];
        o = host::SpecRowResult{};
        o.committed = c;
        o.accepted = std::min(c, k);
        std::copy(row, row + c, o.scores);
        if (a.logprobs != nullptr) std::copy(logprobs + base, logprobs + base + c, a.logprobs);
    }
    st = commit_chains(m, sessions, chains, commit, participants);
    if (st != CERE_OK) {
        for (uint32_t p = 0; p < participants; ++p) row_status[index[p]] = st;
        return st;
    }
    // Draft-extend (D2, F3): per chain, the MTP over its a committed rows at p .. p + a - 1 (the verify rows' target
    // hc rows, next ids = the committed tokens), seeded at row a - 1, then the decode steps: the next proposals. Two
    // or more chains draft together when they fit (draft_rows, P11R-21: one MTP forward per pass over every chain's
    // rows, the same bits), else chain by chain (draft). The commit drained the stream, so the mapped staging is free.
    model::Workspace &ws = m.workspace_;
    uint32_t bases[kMaxRows];
    for (uint32_t p = 0, base = 0; p < participants; base += chains[p].n + 1, ++p) bases[p] = base;
    const bool together = draft_rows_fit(m, sessions, commit, participants);
    uint32_t staged = 0;
    uint32_t staged_from = 0;
    if (together) {
        // The committed ids contiguous at kExtendAt, the proposal counts at kDraftAt.
        staged_from = kExtendAt;
        for (uint32_t p = 0, at = kExtendAt; p < participants; ++p) {
            const host::SpecRowResult &o = out[index[p]];
            for (uint32_t i = 0; i < o.committed; ++i) ws.host_ids32[at++] = int32_t(o.scores[i].id);
            ws.host_ids32[kDraftAt + p] = int32_t(kProposalWords - 1);
        }
        staged = kDraftAt + participants;
    } else {
        for (uint32_t p = 0; p < participants; ++p) {
            const host::SpecRowResult &o = out[index[p]];
            for (uint32_t i = 0; i < o.committed; ++i) ws.host_ids32[bases[p] + i] = int32_t(o.scores[i].id);
            staged = bases[p] + o.committed;
        }
    }
    if (const cudaError_t e = cudaMemcpyAsync(ws.ids32 + staged_from, ws.host_ids32 + staged_from,
                                              (staged - staged_from) * sizeof(int32_t), cudaMemcpyHostToDevice,
                                              stream_);
        e != cudaSuccess) {
        st = fail(e);
    }
    if (together && st == CERE_OK) {
        st = draft_rows(m, sessions, commit, bases, participants);
    } else {
        for (uint32_t p = 0; p < participants && st == CERE_OK; ++p) {
            CudaSession &s = *sessions[p];
            st = draft(m, s, s.state_.position - commit[p], commit[p],
                       ws.hc_rows + uint64_t(bases[p]) * model::kHcHidden,
                       reinterpret_cast<const uint32_t *>(ws.ids32) + bases[p]);
        }
    }
    // A draft failure (after the commit) breaks every participant; their committed tokens stay reported.
    if (st != CERE_OK) {
        for (uint32_t p = 0; p < participants; ++p) {
            sessions[p]->broken_ = true;
            row_status[index[p]] = st;
        }
    }
    return st;
}

}  // namespace cere::device

namespace cere::host {

cere_status create_device(const char *device_id, std::unique_ptr<Device> *out) {
    return cere::device::CudaDevice::open(device_id, out);
}

}  // namespace cere::host
