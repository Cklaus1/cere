// VulkanAmd launch layer (P1, made library-safe in P3 T1): a minimal Vulkan compute launch layer (plain C++17, raw
// Vulkan 1.3, no frameworks) that the ktests and the executor library (libcere_executor_vulkan.so) share.
//
// What it does: selects Vulkan physical device `ordinal` among the GPUs in enumeration order (Context::create), one
// compute queue with timestamps, enables the features the kernels rely on (buffer device address,
// shaderInt64/Int16/Float16/Float64, 8/16-bit storage, subgroup size control, timeline semaphores, synchronization2,
// integer dot product; cooperative matrices when present), allocates buffers whose size is refused above the device's
// maxBufferSize / maxMemoryAllocationSize (4 GiB - 4 on RADV here: tensors larger than that must be split by the
// caller, spec G0 consequence 1), builds compute pipelines from SPIR-V embedded in the binary (or, when a directory
// was set explicitly, from files) cached by (name, push size, required subgroup size, specialization constants) in an
// in-process VkPipelineCache, records dispatches with push constants (kernels read their tensors through buffer
// device addresses: no descriptor sets), submits with a timeline semaphore and waits for it with a bounded timeout,
// brackets dispatches with GPU timestamps, and hands out per-command-buffer descriptor-table space (table_alloc).
//
// Errors (library path): nothing here terminates the process. Every fallible call returns a cere_status:
//   CERE_ERR_ARGUMENT     misuse (bad sizes, second begin() without submit, recording calls outside begin/submit),
//   CERE_ERR_UNSUPPORTED  missing device / feature / extension / shader name / subgroup size,
//   CERE_ERR_CAPACITY     VK_ERROR_OUT_OF_*_MEMORY, a buffer above the device limits, table ring overflow,
//   CERE_ERR_INTERNAL     anything else, and every call on a faulted context.
// Recording calls (dispatch / barrier / stamp) latch the first error of the open command buffer: later recording calls
// of that command buffer record nothing, and submit_wait() returns the latched error without submitting (nothing of a
// half-recorded command buffer ever runs). That is not a fault: the next begin() starts clean.
// Fault model: a failed vkQueueSubmit2, a failed or timed-out vkWaitSemaphores, VK_ERROR_DEVICE_LOST from any call, and
// a missing completion marker after a successful wait (below) make the context FAULTED (sticky): faulted() is true from then on, and every later begin / record / submit / alloc /
// pipeline / table call returns CERE_ERR_INTERNAL without touching the device (the device reports it through the host
// layer's refuse_faulted). Only destruction remains (free() still releases host-side handles).
//
// Completion marker (P3 T1b): amdgpu kills a job that outruns its ring timeout (about 2 s on AMD's kernel) and resets the
// ring, yet RADV still signals the timeline (vkWaitSemaphores returns VK_SUCCESS) and reports DEVICE_LOST only at the
// NEXT submit (an internal benchmark report). So submit_wait() appends to every command buffer a
// barrier and an 8-byte vkCmdUpdateBuffer of a per-submit nonce into a Context-owned host-visible word (the host clears
// it before the submit); after the wait the host compares it. A mismatch = the job did not run to its end: the context
// faults (fault_reason() / last_error() say "completion marker"), submit_wait() returns CERE_ERR_INTERNAL. No pipeline
// is involved (RADV writes an update this small with a CP WRITE_DATA packet in the command stream), so pipeline_count()
// and warm-up lists are unaffected.
// Submit GPU time: every command buffer is bracketed by two timestamps in a Context-owned query pool (independent of
// stamp()'s 256); last_submit_gpu_ms() reports it. kMaxSubmitGpuMs is the budget a caller should chunk its submits
// under; the launch layer only reports (last_submit_over_budget()), it never enforces or splits.
//
// Pipelined submits (P4-G): a context owns CreateOptions::max_in_flight command-buffer SLOTS (1..kMaxInFlight). Each
// slot has its own command buffer, stamp pools, completion-marker word and table-ring part. submit() ends and submits
// the open command buffer WITHOUT waiting and returns its ticket (the timeline value it signals); begin() may then open
// the next one on a free slot while the GPU runs the first. wait(t) RETIRES, strictly in submission order, every
// in-flight submit with ticket <= t: a bounded wait on its timeline value, then its completion marker. submit_wait() =
// submit() + wait(its ticket). A command buffer begun while another submit is in flight starts with a full memory
// barrier (all earlier commands in submission order, all memory writes -> all later commands), so each command buffer
// runs after the previous one finished: the GPU executes them back to back, the host never sits between them.
// Kill ordering: the first retire that fails (wait error / timeout / DEVICE_LOST, or a missing marker = a job the driver
// killed) faults the context. That ticket and EVERY later in-flight ticket are never reported complete (last_retired()
// stays below the failed ticket, last_completed_nonce() is the last good one), whatever their own markers say: their
// inputs came from the killed job. wait() returns CERE_ERR_INTERNAL whichever ticket was asked, and so does every later
// call. The amdgpu job timeout is per job (drm_sched re-arms its timer when the previous job completes), so the budget
// kMaxSubmitGpuMs stays a per-command-buffer budget with several in flight (bench/reports/
// vulkan-amd-p4-g-pipelined-submits-20261116.md measures it).
//
// Threading: a Context is used by ONE thread at a time (the caller serializes every call, const ones included);
// nothing here locks.
//
// What it does not do: memory pooling, multi-queue overlap, or persistence of the pipeline cache (live handles stay in
// process; AGENTS.md 7).
//
// Test convenience (exit on error): launch/vk_launch_testing.h, a separate translation unit that the library never
// links.
#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "cere_executor.h"

namespace cere::vk {

// Policy defaults (typed; CreateOptions overrides them per context).
constexpr uint64_t kDefaultWaitTimeoutNs = 120ull * 1000 * 1000 * 1000;  // submit_wait bound; exceeded = fault
constexpr uint64_t kDefaultTableRingBytes = 1ull << 20;                   // descriptor-table ring (design 4.2)
constexpr uint64_t kMaxTableAlign = 4096;
// GPU-time budget of one submit (a hint, never enforced here): half of the ~2 s amdgpu ring (job) timeout measured on
// AMD (T5 report), the margin covering clock / load variation. A submit above it risks being killed by the kernel
// driver (which the completion marker then reports as a fault). The launch layer never reads or changes the timeout.
constexpr double kMaxSubmitGpuMs = 1000.0;
// Command-buffer slots a context can have in flight (CreateOptions::max_in_flight is 1..kMaxInFlight).
constexpr uint32_t kMaxInFlight = 4;
// Threads Context::warm compiles pipelines on (RADV/ACO compiles on the calling thread; ~130 pipelines at instantiate).
constexpr uint32_t kWarmThreads = 8;

// VkResult -> cere_status (see the header comment); result_name for messages.
cere_status status_of(VkResult r);
const char *result_name(VkResult r);

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    uint64_t addr = 0;     // buffer device address
    uint64_t size = 0;
    void *map = nullptr;   // persistent mapping when host-visible (UMA: device-local memory is host-visible)
    uint32_t memory_type = UINT32_MAX;  // the Vulkan memory type index the buffer lives in (UINT32_MAX: a view)
};

struct Pipeline {
    VkPipeline pipe = VK_NULL_HANDLE;  // VK_NULL_HANDLE: the pipeline could not be built (see Context::pipeline)
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t push_bytes = 0;
};

// One pipeline key: SPIR-V name (as compiled into $OUT/spv, e.g. "k1_embedding.spv"), push-constant bytes, required
// subgroup size (0 = driver default, else 32 / 64) and 32-bit specialization constants with ids 0, 1, ...
struct PipelineSpec {
    std::string spv;
    uint32_t push_bytes = 0;
    uint32_t required_subgroup = 0;
    std::vector<uint32_t> spec;
};

// One SPIR-V module embedded in the binary by executor/vulkan/tools/embed_spv.py (generated table, sorted by name).
struct EmbeddedShader {
    const char *name;      // "k1_embedding.spv"
    const uint32_t *words;
    uint32_t bytes;
    const char *sha256;    // hex of the module's bytes
};
// The embedded table of this binary (count 0 when the generated table object is not linked, e.g. a ktest).
const EmbeddedShader *embedded_shaders(size_t *count);
const EmbeddedShader *find_embedded_shader(const std::string &name);

struct DeviceInfo {
    std::string name, driver_name, driver_info;
    uint32_t ordinal = 0;            // index among the selectable devices (Context::create)
    uint32_t vendor_id = 0, device_id = 0;
    VkPhysicalDeviceType device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    uint32_t api_version = 0, driver_version = 0, subgroup_default = 0, subgroup_min = 0, subgroup_max = 0;
    uint64_t max_buffer = 0, max_alloc = 0;
    double timestamp_period_ns = 0;
    bool coopmat = false;      // VK_KHR_cooperative_matrix enabled on the device
    bool float64 = false;      // shaderFloat64 enabled (required: true on every created context)
    bool integer_dot = false;  // shaderIntegerDotProduct enabled (required: true on every created context)
    bool device_fault = false; // VK_EXT_device_fault enabled (diagnostics only: queried after a fault, see the .cpp)
    bool calibrated = false;   // VK_EXT_calibrated_timestamps (DEVICE + CLOCK_MONOTONIC) enabled: SubmitTiming only
};

struct CreateOptions {
    bool allow_software = false;                         // also offer CPU devices (llvmpipe): tests only
    uint64_t wait_timeout_ns = kDefaultWaitTimeoutNs;    // submit_wait bound (0 is refused)
    uint64_t table_ring_bytes = kDefaultTableRingBytes;  // per command buffer; 0: no ring (table_alloc refuses)
    uint32_t max_in_flight = 1;                          // command-buffer slots (1..kMaxInFlight; else ARGUMENT)
};

// Host / GPU timing of one retired submit (diagnostics; P4-G). Times in ms; NaN = unknown. The calibrated fields need
// VK_EXT_calibrated_timestamps with the DEVICE and CLOCK_MONOTONIC domains (DeviceInfo::calibrated).
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
struct SubmitTiming {
    uint64_t ticket = 0;
    double gpu_ms = -1;                     // GPU: the command buffer's begin .. end stamps (= last_submit_gpu_ms)
    double submit_call_ms = -1;             // host: vkEndCommandBuffer + vkQueueSubmit2
    double wait_call_ms = -1;               // host: blocked in vkWaitSemaphores for this ticket at retire
    double gpu_gap_before_ms = -1;          // GPU: this begin stamp - the previous retired submit's end stamp (same
                                            // clock; < 0 for the first submit or after a gap in tickets)
    double start_after_submit_ms = kNaN;  // calibrated: GPU begin stamp - host return of vkQueueSubmit2
    double retire_after_end_ms = kNaN;    // calibrated: host return of the wait - GPU end stamp
    double calibration_dev_ms = kNaN;     // the calibration's max deviation
};

// Per-context results of one-time device probes that kernels need (they used to be process globals).
enum class Probe : uint32_t { kK2GemmLayout = 0, kK7MoeLayout = 1, kCount = 2 };
enum class ProbeState : uint8_t { kUnprobed = 0, kPass = 1, kFail = 2 };

// A region of the descriptor-table ring: valid from table_alloc until the submit of the command buffer it was
// allocated in completes (submit_wait returned); the next begin() reuses the space.
struct TableRegion {
    uint64_t addr = 0;     // device address (aligned as asked)
    void *host = nullptr;  // host pointer (coherent mapping) to write the table through
    uint64_t bytes = 0;
    // A non-owning Buffer view, for kernels whose signature takes `vk::Buffer &table` (they read map / addr / size).
    // free() on a view is a no-op (buf is VK_NULL_HANDLE).
    Buffer view() const {
        Buffer b;
        b.addr = addr;
        b.size = bytes;
        b.map = host;
        return b;
    }
};

class Context {
public:
    // Selects physical device `ordinal` among the selectable devices (integrated and discrete GPUs, plus CPU devices
    // when opts.allow_software) in vkEnumeratePhysicalDevices order, checks the required features, and creates the
    // device, queue, command pool, timeline semaphore, timestamp pool, pipeline cache and table ring.
    // Out-of-range ordinal / no device / missing feature or extension: CERE_ERR_UNSUPPORTED; out of memory:
    // CERE_ERR_CAPACITY; *why says which. *out is set only on CERE_OK. Never exits.
    static cere_status create(uint32_t ordinal, std::unique_ptr<Context> *out, std::string *why,
                              const CreateOptions &opts = CreateOptions());
    // Number of selectable devices (as create counts them).
    static cere_status count_devices(uint32_t *count, bool allow_software, std::string *why);
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    const DeviceInfo &info() const { return info_; }
    VkInstance instance() const { return inst_; }
    VkPhysicalDevice physical_device() const { return pd_; }
    VkDevice device() const { return dev_; }
    VkMemoryPropertyFlags memory_flags(uint32_t memory_type) const;

    // --- fault state -------------------------------------------------------------------------------------------
    bool faulted() const { return faulted_; }
    const std::string &fault_reason() const { return fault_reason_; }
    VkResult fault_result() const { return fault_result_; }
    // TEST ONLY: marks the context faulted without touching the device (exercises the refusal paths). The library
    // never calls it.
    void inject_fault_for_test(const char *why);
    // Commands this context has recorded or submitted (vkCmd*, vkQueueSubmit2) since creation: lets a test prove that
    // refused calls did not touch the device.
    uint64_t device_commands() const { return device_commands_; }
    // TEST ONLY: the next submit_wait() expects a nonce different from the one its command buffer writes, exactly what
    // an aborted job looks like (the marker is not the expected value), without touching the GPU's execution.
    void skew_completion_nonce_for_test() { nonce_skew_ = 1; }
    // TEST ONLY: the next submitted command buffer does not record its marker write, so the GPU really leaves the
    // host-cleared word as it was: what the host sees of a job the driver killed before its end.
    void omit_completion_marker_for_test() { omit_marker_ = true; }
    // TEST ONLY: called with every non-OK status a begin / dispatch / barrier / stamp / submit_wait call returns. The
    // ktests install one that exits (launch/vk_launch_testing.h) so a launch error stays fatal there; the library
    // never installs one.
    using ErrorHook = void (*)(const Context &ctx, cere_status s);
    void set_error_hook_for_test(ErrorHook h) { error_hook_ = h; }

    // --- memory ------------------------------------------------------------------------------------------------
    // A host-visible buffer with a device address and a persistent mapping. host_heap false: device-local +
    // host-visible preferred (UMA), else any host-visible. host_heap true: a host-visible + coherent, NOT device-local
    // type (the host heap, spec G0 consequence 2; exactly HOST_VISIBLE|HOST_COHERENT preferred = uncached), refused
    // (UNSUPPORTED) when the device has none. bytes 0: ARGUMENT; above maxBufferSize / maxMemoryAllocationSize:
    // CAPACITY. On any refusal `out` is empty and nothing is kept.
    cere_status allocate(uint64_t bytes, Buffer &out, bool host_heap = false, std::string *why = nullptr);
    // allocate() as a bool; the reason of a refusal is kept in last_error() (nothing is printed).
    bool alloc(uint64_t bytes, Buffer &out, bool host_heap = false);
    void free(Buffer &b);
    const std::string &last_error() const { return last_error_; }

    // --- pipelines -----------------------------------------------------------------------------------------------
    // Fallible: builds (or finds cached) the pipeline of `s`. The SPIR-V comes from the embedded table first, then from
    // the shader directory when set_shader_dir was called. Unknown name / bad module / unsupported subgroup size:
    // UNSUPPORTED.
    cere_status pipeline(const PipelineSpec &s, const Pipeline **out, std::string *why = nullptr);
    // Kernel-recording form (the kernels' `ctx.dispatch(ctx.pipeline(...), ...)` idiom): on failure returns an empty
    // Pipeline (pipe == VK_NULL_HANDLE) and keeps the status in last_error(); dispatch() of an empty pipeline latches
    // that status into the open command buffer, so submit_wait() returns it and nothing is submitted. warm() at
    // instantiate makes sure this cannot fail during a forward.
    const Pipeline &pipeline(const std::string &spv, uint32_t push_bytes, uint32_t required_subgroup = 0,
                             const std::vector<uint32_t> &spec = {});
    // Creates every pipeline of `specs` now; the first failure in `specs` order is returned (*why names the spec).
    // The compiles run on up to kWarmThreads threads (RADV compiles on the calling thread); the result is the same
    // pipeline set as one pipeline() call per spec.
    cere_status warm(const std::vector<PipelineSpec> &specs, std::string *why = nullptr);
    bool has_pipeline(const PipelineSpec &s) const;
    size_t pipeline_count() const { return pipes_.size(); }
    // Enables file loading as a fallback after the embedded table (the ktests pass argv[1] here).
    void set_shader_dir(const std::string &d) { shader_dir_ = d; shader_dir_set_ = true; }

    // --- kernel selections -----------------------------------------------------------------------------------------
    // CERE_LEVER_K2_TAIL_SPLIT (vulkan-amd design note P4-B, D7): when on, the K2 GEMM's x-row scan may mark a row
    // SPLIT (matrix unit without its tail values + an exact FP32 fma chain over them); off (the default) every x row
    // that is not exact as FP16 * 2^shift is FB (its outputs on the exact FP32 route). Read by every vkk2::gemm call
    // recorded after it is set; the device sets it from the model's levers before recording a forward.
    void set_k2_tail_split(bool on) { k2_tail_split_ = on; }
    bool k2_tail_split() const { return k2_tail_split_; }
    // K2 x-info taps (the exact-fallback counters, device/vulkan_forward.h kCountExactFallback): tap i armed on
    // (x, m, k) makes the first vkk2::gemm recorded on exactly that x, m and k write its x-row info words (the x scan
    // every GEMM records, k2_gemm_vk.cpp) at `info` instead of its workspace, so the caller counts those rows without
    // scanning x a second time. The words and the GEMM's reads of them are the same either way: no output bit changes.
    // Recording-time state only; clear_k2_info() disarms every tap.
    static constexpr uint32_t kK2InfoTaps = 2;
    void arm_k2_info(uint32_t i, uint64_t x, uint32_t m, uint32_t k, uint64_t info) {
        k2_taps_[i] = K2InfoTap{x, info, m, k, false};
    }
    bool k2_info_hit(uint32_t i) const { return k2_taps_[i].hit; }
    void clear_k2_info() {
        for (K2InfoTap &t : k2_taps_) t = K2InfoTap{};
    }
    // vkk2::gemm: the info address of the armed tap on (x, m, k) not hit yet (it is hit now), else 0.
    uint64_t take_k2_info(uint64_t x, uint32_t m, uint32_t k) {
        for (K2InfoTap &t : k2_taps_) {
            if (t.x != 0 && !t.hit && t.x == x && t.m == m && t.k == k) {
                t.hit = true;
                return t.info;
            }
        }
        return 0;
    }

    // --- probes --------------------------------------------------------------------------------------------------
    ProbeState probe(Probe p) const { return probes_[uint32_t(p)]; }
    void set_probe(Probe p, ProbeState s) { probes_[uint32_t(p)] = s; }

    // --- recording -----------------------------------------------------------------------------------------------
    // begin() opens a command buffer on the lowest free slot (ARGUMENT if one is already open, the open one kept
    // untouched, or if every slot has a submit in flight: wait() first) and resets that slot's table ring; dispatch()
    // binds + pushes + dispatches; barrier() orders compute writes before later compute/transfer/host reads; stamp(i)
    // writes timestamp i (i < kStamps) after all earlier work. submit() ends and submits the open command buffer without
    // waiting (*ticket = its ticket) and closes it whatever the outcome; a latched recording error is returned and
    // nothing is submitted. wait(t) retires every in-flight submit with ticket <= t in submission order (see the header
    // comment for the kill ordering); a ticket already retired is OK, 0 or one never submitted is ARGUMENT. With
    // spin_from_ns >= 0 (a steady_clock time, ns) the wait for ticket t itself blocks only until that time, then polls
    // the timeline value (vkGetSemaphoreCounterValue) for at most kMaxSpinNs before it blocks again: the caller's
    // expected GPU end minus a margin, so the host sees the end without the ~0.3 ms sleeping-thread wake-up. Same
    // bound, same checks. wait_all() retires everything in flight. submit_wait() = submit() + wait(that ticket).
    // Recording calls outside begin..submit return ARGUMENT and record nothing.
    static constexpr uint32_t kStamps = 256;
    static constexpr int64_t kMaxSpinNs = 3000000;
    cere_status begin();
    cere_status dispatch(const Pipeline &p, const void *push, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
    cere_status barrier();
    cere_status stamp(uint32_t i);
    cere_status submit(uint64_t *ticket = nullptr);
    cere_status wait(uint64_t ticket, int64_t spin_from_ns = -1);
    cere_status wait_all();
    cere_status submit_wait();
    bool recording() const { return recording_; }
    // The first error latched into the open command buffer (CERE_OK if none).
    cere_status record_status() const { return rec_status_; }
    uint32_t max_in_flight() const { return nslots_; }
    uint32_t in_flight() const;                                 // submitted, not yet retired
    uint64_t last_submitted() const { return timeline_value_; }  // ticket of the last submit (0: none)
    uint64_t last_retired() const { return retired_value_; }     // highest ticket retired and verified
    // Milliseconds between stamps a and b of the most recently RETIRED command buffer (0 for an unwritten stamp, a bad
    // index or a faulted context). Valid until the next begin().
    double stamp_ms(uint32_t a, uint32_t b);
    // GPU time of the last retired submit (begin .. end of its command buffer, GPU timestamps), ms; < 0 unknown.
    double last_submit_gpu_ms() const { return last_gpu_ms_; }
    bool last_submit_over_budget() const { return last_gpu_ms_ > kMaxSubmitGpuMs; }
    // The nonce the last retired submit wrote and the host verified (0 before the first).
    uint64_t last_completed_nonce() const { return last_nonce_; }
    // Host / GPU timing of the last retired submit (diagnostics). The calibrated fields are filled only after
    // set_calibrated_timing(true) on a device with info().calibrated (one vkGetCalibratedTimestampsEXT per retire).
    const SubmitTiming &last_submit_timing() const { return last_timing_; }
    void set_calibrated_timing(bool on) { calibrated_timing_ = on; }

    // --- descriptor-table ring -----------------------------------------------------------------------------------
    // Bump allocation of `bytes` (align: a power of two, 1..kMaxTableAlign; the device address is aligned) from the
    // open command buffer's slot of the ring, valid until that command buffer's submit is retired. Only while
    // recording (else ARGUMENT); overflow: CAPACITY (nothing is latched: the caller decides). Reset by begin().
    cere_status table_alloc(uint64_t bytes, uint64_t align, TableRegion *out);
    uint64_t table_ring_bytes() const { return ring_slot_bytes_; }  // per command buffer
    uint64_t table_ring_used() const { return ring_used_; }

private:
    Context() = default;
    cere_status init(uint32_t ordinal, const CreateOptions &opts, std::string *why);
    cere_status vk_fail(VkResult r, const std::string &what, std::string *why);  // maps r; DEVICE_LOST -> fault
    void set_fault(VkResult r, const std::string &why);
    cere_status latch(cere_status s);
    cere_status report(cere_status s);
    cere_status begin_impl();
    cere_status dispatch_impl(const Pipeline &p, const void *push, uint32_t gx, uint32_t gy, uint32_t gz);
    cere_status barrier_impl();
    cere_status stamp_impl(uint32_t i);
    cere_status submit_impl(uint64_t *ticket);
    cere_status wait_impl(uint64_t ticket, int64_t spin_from_ns);
    cere_status retire_one(uint32_t slot, int64_t spin_from_ns);
    void load_stamps(uint32_t slot);
    cere_status recording_gate();
    cere_status load_module(const std::string &name, VkShaderModule *mod, std::string *why);
    cere_status check_spec(const PipelineSpec &s, std::string *why) const;
    // Layout + compute pipeline of `s` from module `mod` into *p; touches no member (warm() calls it from threads).
    VkResult create_pipeline(const PipelineSpec &s, VkShaderModule mod, Pipeline *p) const;
    void append_fault_info();  // VK_EXT_device_fault description, when enabled, into fault_reason_

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice pd_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue q_ = VK_NULL_HANDLE;
    uint32_t qf_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    // One command-buffer slot: its command buffer, caller stamps, frame stamps (begin / end), completion-marker word
    // (marker_ + marker_off) and table-ring part (ring_ + ring_off); the submit it carries while busy.
    struct Slot {
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkQueryPool stamps = VK_NULL_HANDLE;
        VkQueryPool frame = VK_NULL_HANDLE;
        uint64_t marker_off = 0, ring_off = 0;
        bool busy = false;
        uint64_t ticket = 0, nonce = 0, expected = 0;
        int64_t host_submit_end_ns = 0;
        double submit_call_ms = -1;
    };
    Slot slots_[kMaxInFlight];
    uint32_t nslots_ = 1;
    uint32_t cur_ = 0;                 // the open command buffer's slot (while recording_)
    int32_t stamp_slot_ = -1;          // the slot of the last retired submit (stamp_ms reads it)
    uint64_t retired_value_ = 0;
    uint64_t prev_end_tick_ = 0, prev_end_ticket_ = 0;
    SubmitTiming last_timing_;
    PFN_vkGetCalibratedTimestampsEXT calibrate_ = nullptr;
    bool omit_marker_ = false;
    bool calibrated_timing_ = false;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    uint64_t timeline_value_ = 0;
    uint64_t wait_timeout_ns_ = kDefaultWaitTimeoutNs;
    bool stamps_read_ = false;
    std::vector<uint64_t> stamp_vals_;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mp_{};
    DeviceInfo info_;
    std::string shader_dir_;
    bool shader_dir_set_ = false;
    bool k2_tail_split_ = false;
    struct K2InfoTap {
        uint64_t x = 0, info = 0;
        uint32_t m = 0, k = 0;
        bool hit = false;
    };
    K2InfoTap k2_taps_[kK2InfoTaps];
    using Key = std::tuple<std::string, uint32_t, uint32_t, std::vector<uint32_t>>;
    std::map<Key, Pipeline> pipes_;
    std::map<std::string, VkShaderModule> modules_;
    Pipeline null_pipe_;
    cere_status last_pipeline_status_ = CERE_OK;
    std::string last_error_;
    bool recording_ = false;
    cere_status rec_status_ = CERE_OK;
    bool faulted_ = false;
    VkResult fault_result_ = VK_SUCCESS;
    std::string fault_reason_;
    uint64_t device_commands_ = 0;
    ErrorHook error_hook_ = nullptr;
    ProbeState probes_[uint32_t(Probe::kCount)] = {};
    Buffer ring_;                                // nslots_ parts of ring_slot_bytes_
    uint64_t ring_slot_bytes_ = 0;
    uint64_t ring_used_ = 0;
    Buffer marker_;                              // completion marker words (host-visible, coherent), 64 B per slot
    uint64_t ts_mask_ = ~0ull;
    uint64_t nonce_salt_ = 0, last_nonce_ = 0, nonce_skew_ = 0;
    double last_gpu_ms_ = -1;
};

}  // namespace cere::vk
