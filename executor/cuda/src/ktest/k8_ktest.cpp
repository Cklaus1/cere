// K8 kernel-test entry points (executor/tests/gpu/test_model_k8_forward.py, suite `model_k8`): the A4 model through
// the real ABI v4 runtime of this library (cere_executor_create, cere_model_*, cere_session_*, submit/poll/cancel)
// on a ktest-owned executor on GPU0, plus a teacher-forced forward with model::ForwardTaps (forward_trace, and
// forward_trace_detail with the m22p design note D3 detail names), a decode-mode teacher forcing (forced_decode,
// B8 `test_numerics_b8_layers.py`, m22n design note D8/D9) and its tapped twin (forced_decode_trace,
// m22d design note D2), the D5 patched prompt forwards (forward_patch, forward_patch_screen), and the M2.6 rows
// (m26 design note I7: forward_decode_rows, the batched decode step; generate_rows, several requests through the
// executor's rounds; ledger CR4: generate_rows_sampled with sampled rows, sampler_plan_bytes). The M2.11 session
// entries (session_open/feed/snapshot/restore/close, state_digest; m211 design note Interfaces, K15
// `test_model_k15_snapshot.py`) drive long-lived sessions of the same handle. Host buffers are CPU memory, `values` of
// the traces and of forward_patch is device memory; the trailing stream is the caller's (torch) stream, which is
// synchronized before the executor's own stream touches caller memory. 0 = success, else a cere_status.
// O47 (build-loop-ple design note, `test_model_host_share.py`): model_create_device (a device_id with the host-share
// options), segment_digest and the test-only segment_poke on a shared-mode model's segment. D12 (m23 design note,
// `test_profile_forward.py`): forward_profile, one prefill and greedy decode steps split by a marker kernel.
#include "cere_executor.h"
#include "device/cuda_device.h"
#include "host/prefix_cache.h"
#include "host/sha256_ni.h"
#include "host/snapshot_format.h"
#include "host/state.h"
#include "kernels/k6_attention.h"
#include "ktest/profile_mark.h"
#include "ktest/row_digest.h"
#include "ktest/segment_poke.h"
#include "model/model.h"
#include "p3b/p3b_tactic.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

namespace cere::ktest {
namespace {

constexpr uint64_t kWorkerEpoch = 1;

// One ktest handle: an executor of its own and the model created on it. trace_passes: the forward calls the last
// cere_ktest_forward_trace on this handle ran (its prefill passes; m22c D7), read by cere_ktest_forward_trace_passes.
struct K8Model {
    cere_executor *executor = nullptr;
    cere_model *model = nullptr;
    uint64_t trace_passes = 0;
};

K8Model *handle(uint64_t h) { return reinterpret_cast<K8Model *>(static_cast<uintptr_t>(h)); }

device::CudaDevice &cuda_device(const K8Model &k) { return static_cast<device::CudaDevice &>(*k.executor->device); }
device::CudaModel &cuda_model(const K8Model &k) { return static_cast<device::CudaModel &>(*k.model->device); }

cere_status sync(cudaStream_t stream) {
    return cudaStreamSynchronize(stream) == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

uint64_t device_bytes_in_use() {
    size_t free = 0;
    size_t total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return uint64_t(total - free);
}

// Sets the model's rows per forward call for the duration of one request (0 = the executor's own policy).
class ChunkScope {
public:
    ChunkScope(device::CudaModel &model, uint64_t chunk) : model_(model) { model_.set_prefill_chunk(chunk); }
    ~ChunkScope() { model_.set_prefill_chunk(0); }
    ChunkScope(const ChunkScope &) = delete;
    ChunkScope &operator=(const ChunkScope &) = delete;

private:
    device::CudaModel &model_;
};

// A session of the model's context limit (or of `context` tokens), released on scope exit.
class SessionScope {
public:
    explicit SessionScope(const K8Model &k) : k_(k) {}
    ~SessionScope() {
        if (session_ != nullptr) cere_session_release(k_.executor, session_);
    }
    SessionScope(const SessionScope &) = delete;
    SessionScope &operator=(const SessionScope &) = delete;

    cere_status create(uint64_t context = 0) {
        const uint64_t tokens = context != 0 ? context : k_.model->plan.max_context_tokens;
        return cere_session_create(k_.executor, k_.model, tokens, &session_);
    }
    cere_session *get() const { return session_; }

private:
    const K8Model &k_;
    cere_session *session_ = nullptr;
};

// forward_trace's sink: B8 entries of the reported rows, assembled across prefill chunks. Every entry has `rows`
// rows in total; a chunk contributes rows [row0, row0 + chunk rows) of each entry, in the same entry order.
struct TraceSink {
    uint8_t *names = nullptr;
    uint64_t names_capacity = 0;
    int64_t *shapes = nullptr;
    uint64_t entry_capacity = 0;
    uint8_t *values = nullptr;
    uint64_t values_capacity = 0;
    uint64_t rows = 0;  // reported positions in total
    // filled by the first reporting chunk
    uint64_t entries = 0;
    uint64_t name_bytes = 0;
    uint64_t value_bytes = 0;
    // the current chunk
    bool first = true;
    uint64_t row0 = 0;   // first reported row of this chunk
    uint64_t entry = 0;  // next entry index in this chunk
    uint64_t name_at = 0;  // byte offset of the next entry's name in `names` (later chunks: verified)
    uint64_t value_at = 0;  // byte offset of the next entry's first row in `values`
    cere_status error = CERE_OK;

    static cere_status emit(void *ctx, const char *name, const model::bf16 *v, uint64_t r, uint64_t cols,
                            cudaStream_t stream) {
        auto &s = *static_cast<TraceSink *>(ctx);
        const size_t len = std::strlen(name) + 1;
        const uint64_t row_bytes = cols * sizeof(model::bf16);
        if (s.first) {
            if (s.entries >= s.entry_capacity || len > s.names_capacity - s.name_bytes ||
                s.rows * row_bytes > s.values_capacity - s.value_bytes) {
                return s.error = CERE_ERR_CAPACITY;
            }
            std::memcpy(s.names + s.name_bytes, name, len);
            int64_t *shape = s.shapes + 3 * s.entries;
            shape[0] = 2;
            shape[1] = int64_t(s.rows);
            shape[2] = int64_t(cols);
            s.name_bytes += len;
            s.value_bytes += s.rows * row_bytes;
            ++s.entries;
        } else if (s.entry >= s.entries || len > s.name_bytes - s.name_at ||
                   std::memcmp(s.names + s.name_at, name, len) != 0 ||
                   s.shapes[3 * s.entry + 2] != int64_t(cols)) {
            return s.error = CERE_ERR_INTERNAL;  // a chunk emitted a different entry sequence
        }
        if (s.row0 + r > s.rows) return s.error = CERE_ERR_INTERNAL;
        if (cudaMemcpyAsync(s.values + s.value_at + s.row0 * row_bytes, v, r * row_bytes, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            cudaGetLastError();
            return s.error = CERE_ERR_INTERNAL;
        }
        s.name_at += len;
        s.value_at += s.rows * row_bytes;
        ++s.entry;
        return CERE_OK;
    }
};

// forced_decode_trace's size of one tapped single-row forward: forward_trace's sink sizes, nothing written.
struct CountSink {
    uint64_t entries = 0;
    uint64_t name_bytes = 0;
    uint64_t value_bytes = 0;

    static cere_status emit(void *ctx, const char *name, const model::bf16 *, uint64_t r, uint64_t cols,
                            cudaStream_t) {
        auto &s = *static_cast<CountSink *>(ctx);
        ++s.entries;
        s.name_bytes += std::strlen(name) + 1;
        s.value_bytes += r * cols * sizeof(model::bf16);
        return CERE_OK;
    }
};

// forward_decode_rows's sink: a decode step of R rows emits each B8 name once per row, rows in order (model.h
// forward_decode_rows); row r's entries go to slab[r], laid out as one step of forced_decode_trace (names, shapes
// {2, 1, cols}, values). A single-row DECODE forward emits into slab[0] the same way (rows 1).
struct RowsSink {
    struct Slab {
        uint8_t *names = nullptr;
        int64_t *shapes = nullptr;
        uint8_t *values = nullptr;
        uint64_t name_at = 0;
        uint64_t value_at = 0;
    };
    Slab slab[kernels::GemmPlan::kMaxRows];
    uint32_t rows = 0;
    // one row's step (the probe's sizes)
    uint64_t entries = 0;
    uint64_t name_bytes = 0;
    uint64_t value_bytes = 0;
    uint64_t call = 0;
    cere_status error = CERE_OK;

    static cere_status emit(void *ctx, const char *name, const model::bf16 *v, uint64_t r, uint64_t cols,
                            cudaStream_t stream) {
        auto &s = *static_cast<RowsSink *>(ctx);
        const uint64_t entry = s.call / s.rows;
        Slab &d = s.slab[s.call % s.rows];
        ++s.call;
        const size_t len = std::strlen(name) + 1;
        const uint64_t bytes = cols * sizeof(model::bf16);
        if (r != 1 || entry >= s.entries || len > s.name_bytes - d.name_at || bytes > s.value_bytes - d.value_at) {
            return s.error = CERE_ERR_INTERNAL;  // another entry sequence than the probe's
        }
        std::memcpy(d.names + d.name_at, name, len);
        int64_t *shape = d.shapes + 3 * entry;
        shape[0] = 2;
        shape[1] = 1;
        shape[2] = int64_t(cols);
        if (cudaMemcpyAsync(d.values + d.value_at, v, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            cudaGetLastError();
            return s.error = CERE_ERR_INTERNAL;
        }
        d.name_at += len;
        d.value_at += bytes;
        return CERE_OK;
    }

    // Every row received exactly one step.
    bool complete() const {
        if (call != entries * rows) return false;
        for (uint32_t r = 0; r < rows; ++r) {
            if (slab[r].name_at != name_bytes || slab[r].value_at != value_bytes) return false;
        }
        return true;
    }
};

using PatchFn = const model::bf16 *(*)(void *ctx, const char *name, uint64_t layer, uint64_t rows, uint64_t cols,
                                       const model::bf16 *native, cudaStream_t stream);

// Columns of a forward_patch name: a P point of model.h ForwardTaps that runs once per forward (per layer) on `cfg`'s
// layer kinds; 0 for anything else (unknown names, I/D/E names, the twice-per-layer gdn_ba / moe_shared_gate_up
// GEMM outputs, which only forward_patch_screen's callback can tell apart).
uint64_t patch_cols(const cere_qwen4_exp_config &cfg, std::string_view name) {
    using namespace model;
    if (name == "native.final_mix.out") return kHidden;
    std::string_view rest;
    bool native = false;
    if (name.substr(0, 13) == "model.layers.") {
        rest = name.substr(13);
    } else if (name.substr(0, 14) == "native.layers.") {
        rest = name.substr(14);
        native = true;
    } else {
        return 0;
    }
    uint64_t layer = 0;
    size_t digits = 0;
    while (digits < rest.size() && digits < 3 && rest[digits] >= '0' && rest[digits] <= '9') {
        layer = layer * 10 + uint64_t(rest[digits] - '0');
        ++digits;
    }
    if (digits == 0 || (digits > 1 && rest[0] == '0') || digits >= rest.size() || rest[digits] != '.' ||
        layer >= kLayers) {
        return 0;
    }
    rest = rest.substr(digits + 1);
    const bool full = ((cfg.full_attention_layers >> layer) & 1) != 0;
    if (!native) {
        if (rest == "mlp.experts" || rest == "mlp.shared_expert.down_proj") return kHidden;
        if (rest == (full ? "self_attn.o_proj" : "linear_attn.out_proj")) return kHidden;
        return 0;
    }
    if (rest == "attn_mix.out" || rest == "mlp_mix.out" || rest == "gemm.moe_shared_down.out") return kHidden;
    if (rest == "gemm.moe_router.out") return kExperts;
    if (full) {
        if (rest == "gemm.attn_qkv.out") return kQkvWidth;
        if (rest == "gemm.attn_o.out") return kHidden;
        return 0;
    }
    if (rest == "gemm.gdn_qkv.out") return k5::kConvDim;
    if (rest == "gemm.gdn_z.out") return k5::kValueDim;
    if (rest == "gemm.gdn_out.out") return kHidden;
    return 0;
}

// The patch callback of forward_patch (`list`: name -> device BF16 [n, cols]) or forward_patch_screen (`user`).
// It also takes the target's BF16 logit from the `native.logits` input point (one scored row: the last), which it
// does not pass on.
struct PatchRun {
    std::vector<std::pair<std::string, const model::bf16 *>> list;
    std::vector<bool> used;
    PatchFn user = nullptr;
    void *user_ctx = nullptr;
    uint32_t target = 0;
    uint16_t target_bits = 0;
    uint64_t logit_rows = 0;
    cere_status error = CERE_OK;

    static const model::bf16 *patch(void *ctx, const char *name, uint64_t layer, uint64_t rows, uint64_t cols,
                                    const model::bf16 *native, cudaStream_t stream) {
        auto &p = *static_cast<PatchRun *>(ctx);
        if (std::strcmp(name, "native.logits") == 0) {
            ++p.logit_rows;
            if (rows != 1 || cols <= p.target ||
                cudaMemcpyAsync(&p.target_bits, native + p.target, sizeof(p.target_bits), cudaMemcpyDeviceToHost,
                                stream) != cudaSuccess) {
                cudaGetLastError();
                p.error = CERE_ERR_INTERNAL;
            }
            return nullptr;
        }
        if (p.user != nullptr) return p.user(p.user_ctx, name, layer, rows, cols, native, stream);
        for (size_t i = 0; i < p.list.size(); ++i) {
            if (p.list[i].first == name) {
                p.used[i] = true;
                return p.list[i].second;
            }
        }
        return nullptr;
    }
};

// The prompt alone as one PREFILL forward (n = token_count <= the workspace rows) on a fresh session with `run` as
// ForwardTaps.patch; the last row's top-1 and the target's logit. Outputs are written only on success.
cere_status patched_forward(const K8Model &k, const uint32_t *tokens, uint64_t token_count, PatchRun &run,
                            float *out_target_logit, cere_token_score *out_top1, cudaStream_t stream) {
    device::CudaDevice &device = cuda_device(k);
    device::CudaModel &cmodel = cuda_model(k);
    if (token_count > cmodel.workspace_rows()) return CERE_ERR_ARGUMENT;
    // Patch values may still be written by the caller's stream.
    if (const cere_status st = sync(stream); st != CERE_OK) return st;
    SessionScope session(k);
    if (const cere_status st = session.create(); st != CERE_OK) return st;
    auto &csession = static_cast<device::CudaSession &>(*session.get()->device);
    model::ForwardTaps taps;
    taps.last_rows = 1;
    taps.ctx = &run;
    taps.patch = &PatchRun::patch;
    cere_token_score score{};
    const cere_status st =
        device.forward(cmodel, csession, CERE_OP_PREFILL, tokens, uint32_t(token_count), &taps, &score);
    if (st != CERE_OK) return run.error != CERE_OK ? run.error : st;
    if (run.error != CERE_OK) return run.error;
    if (run.logit_rows != 1) return CERE_ERR_INTERNAL;
    for (bool used : run.used) {
        if (!used) return CERE_ERR_INTERNAL;
    }
    const uint32_t bits = uint32_t(run.target_bits) << 16;
    std::memcpy(out_target_logit, &bits, sizeof(bits));
    *out_top1 = score;
    return CERE_OK;
}

// SelfTune ST1d (cere_ktest_next_prefill_slice): the prefill slice in ms the next create sets on its executor (0 =
// none); the create takes it.
std::atomic<uint32_t> armed_prefill_slice{0};

// cere_ktest_model_create(_device): the executor on `device_id`, then the model.
int create_model(const cere_model_desc *desc, uint64_t memory_budget_bytes, const char *device_id,
                 uint64_t *out_model, cudaStream_t stream) {
    if (desc == nullptr || device_id == nullptr || out_model == nullptr) return CERE_ERR_ARGUMENT;
    *out_model = 0;
    if (const cere_status st = sync(stream); st != CERE_OK) return st;
    auto *k = new (std::nothrow) K8Model;
    if (k == nullptr) return CERE_ERR_CAPACITY;
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = device_id;
    options.worker_epoch = kWorkerEpoch;
    options.memory_budget_bytes = memory_budget_bytes;
    cere_status st = cere_executor_create(&k->executor, &options);
    // An armed prefill slice goes on the executor before the model, so the model keeps the MixedPD park region
    // (cere_executor_set_prefill_slice).
    const uint32_t slice_ms = armed_prefill_slice.exchange(0);
    if (st == CERE_OK && slice_ms != 0) st = cere_executor_set_prefill_slice(k->executor, slice_ms);
    if (st == CERE_OK) st = cere_model_create(k->executor, desc, &k->model);
    if (st != CERE_OK) {
        if (k->executor != nullptr) cere_executor_destroy(k->executor);
        delete k;
        return st;
    }
    *out_model = uint64_t(reinterpret_cast<uintptr_t>(k));
    return CERE_OK;
}
}  // namespace
}  // namespace cere::ktest

using cere::ktest::K8Model;
using cere::model::bf16;

extern "C" __attribute__((visibility("default"))) int cere_ktest_model_create(const cere_model_desc *desc,
                                                                              uint64_t memory_budget_bytes,
                                                                              uint64_t *out_model,
                                                                              cudaStream_t stream) {
    return cere::ktest::create_model(desc, memory_budget_bytes, "0", out_model, stream);
}

// SelfTune ST1d (executor/tests/gpu/test_graph_launches.py): arms the prefill slice (ms; 0 disarms) the next
// cere_ktest_model_create* sets on its executor before the model's create, so that model keeps the MixedPD park
// region (cere_executor_set_prefill_slice); the create takes it, and the one after runs without. Above
// CERE_MAX_PREFILL_SLICE_MS: CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_next_prefill_slice(uint64_t slice_ms,
                                                                                    cudaStream_t) {
    if (slice_ms > CERE_MAX_PREFILL_SLICE_MS) return CERE_ERR_ARGUMENT;
    cere::ktest::armed_prefill_slice.store(uint32_t(slice_ms));
    return CERE_OK;
}

// O47 (build-loop-ple design note): cere_ktest_model_create on the executor device_id `device_id` (a NUL-terminated host
// string, e.g. "0,host_share=/dev/shm/cere-host,artifact=<64 hex>") instead of "0".
extern "C" __attribute__((visibility("default"))) int cere_ktest_model_create_device(const cere_model_desc *desc,
                                                                                     uint64_t memory_budget_bytes,
                                                                                     const char *device_id,
                                                                                     uint64_t *out_model,
                                                                                     cudaStream_t stream) {
    return cere::ktest::create_model(desc, memory_budget_bytes, device_id, out_model, stream);
}

// O47: the content digest of a shared-mode model's segment, recomputed now over the region through this process's CPU
// mapping (host/host_share.h region_digest, the digest the header holds after publish): out_digest[32] (host), out[0]
// the region bytes, out[1] 1 when the region was registered on the rw_mprotect path, 0 on read_only (host).
// CERE_ERR_ARGUMENT for a model without a segment (private mode). Synchronizes the caller's stream first.
extern "C" __attribute__((visibility("default"))) int cere_ktest_segment_digest(uint64_t model, uint8_t *out_digest,
                                                                                uint64_t *out, cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out_digest == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    const cere::host::SharedSegment *segment = cmodel.segment();
    if (segment == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    const std::array<uint8_t, 32> digest = cere::host::region_digest(segment->data(), segment->bytes());
    std::memcpy(out_digest, digest.data(), digest.size());
    out[0] = segment->bytes();
    out[1] = cmodel.host_share_rw() ? 1 : 0;
    return CERE_OK;
}

// O47, test-only: one GPU thread stores the low byte of `value` at byte `offset` of a shared-mode model's region
// through the device alias (the write the rw_mprotect registration no longer blocks on the GPU side), then the
// stream is synchronized: out[0] the byte before, out[1] the byte read back through the CPU mapping after (host).
// CERE_ERR_UNSUPPORTED on the read_only path (the GPU write would fault and poison the context); CERE_ERR_ARGUMENT
// for a model without a segment or an offset outside the region.
extern "C" __attribute__((visibility("default"))) int cere_ktest_segment_poke(uint64_t model, uint64_t offset,
                                                                              uint64_t value, uint64_t *out,
                                                                              cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    const cere::host::SharedSegment *segment = cmodel.segment();
    if (segment == nullptr || offset >= segment->bytes()) return CERE_ERR_ARGUMENT;
    if (!cmodel.host_share_rw()) return CERE_ERR_UNSUPPORTED;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    const volatile uint8_t *cpu = segment->data() + offset;
    out[0] = *cpu;
    if (cere::ktest::poke_byte(cmodel.host_device() + offset, uint8_t(value), stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    out[1] = *cpu;
    return CERE_OK;
}

// M3.2N G-2 (m32n design note D5; each executor of a process maps and registers the segment itself): K4's
// FP8 gather (k4::gather) of `n` >= 1 rows of a shared-mode model's PLE table, row ids[i] mod table_rows (ids: host
// int64 [n], each >= 0), run twice on the model's own device: through the device alias of its segment, the mapping
// this device's own registration gives (out_alias), and over a device copy of the same rows read through this
// process's CPU mapping (out_copy). Both outputs host BF16 [n, 160]; out_rows[0] = table_rows (host). Equal outputs
// mean the device reads the segment's bytes through its registration. CERE_ERR_ARGUMENT for a model without a segment,
// a PLE table outside it, n < 1 or a negative id; CERE_ERR_UNSUPPORTED for an NVFP4 table. Synchronizes the caller's
// stream first; the work runs on the model's device (its legacy stream) and the caller's current device is restored.
extern "C" __attribute__((visibility("default"))) int cere_ktest_segment_ple_gather(uint64_t model, const int64_t *ids,
                                                                                    uint64_t n, uint16_t *out_alias,
                                                                                    uint16_t *out_copy,
                                                                                    int64_t *out_rows,
                                                                                    cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || ids == nullptr || n == 0 || out_alias == nullptr || out_copy == nullptr ||
        out_rows == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    const cere::host::SharedSegment *segment = cmodel.segment();
    const cere::k4::Weights &w = cmodel.weights().ple;
    if (segment == nullptr || w.table == nullptr || w.table_rows < 1) return CERE_ERR_ARGUMENT;
    if (w.format != cere::k4::TableFormat::kFp8) return CERE_ERR_UNSUPPORTED;
    constexpr uint64_t kRowBytes = cere::k4::kHeadDim;
    const uintptr_t base = reinterpret_cast<uintptr_t>(cmodel.host_device());
    const uintptr_t table = reinterpret_cast<uintptr_t>(w.table);
    const uint64_t table_bytes = uint64_t(w.table_rows) * kRowBytes;
    if (table < base || table - base > segment->bytes() || segment->bytes() - (table - base) < table_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    // Device buffer: rows int64 [n] | iota int64 [n] | copied rows [n, 160] | out_alias BF16 [n, 160] | out_copy.
    std::vector<int64_t> index(2 * n);
    std::vector<uint8_t> copy(n * kRowBytes);
    const uint8_t *cpu_table = segment->data() + (table - base);
    for (uint64_t i = 0; i < n; ++i) {
        if (ids[i] < 0) return CERE_ERR_ARGUMENT;
        index[i] = ids[i] % w.table_rows;
        index[n + i] = int64_t(i);
        std::memcpy(copy.data() + i * kRowBytes, cpu_table + uint64_t(index[i]) * kRowBytes, kRowBytes);
    }
    out_rows[0] = w.table_rows;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(cmodel.ordinal()) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    const uint64_t index_bytes = index.size() * sizeof(int64_t);
    const uint64_t out_bytes = n * kRowBytes * sizeof(uint16_t);
    uint8_t *buf = nullptr;
    cere_status st = CERE_ERR_INTERNAL;
    if (cudaMalloc(&buf, index_bytes + copy.size() + 2 * out_bytes) == cudaSuccess) {
        const auto *rows_dev = reinterpret_cast<const int64_t *>(buf);
        const auto *iota_dev = rows_dev + n;
        uint8_t *copy_dev = buf + index_bytes;
        auto *alias_out = reinterpret_cast<__nv_bfloat16 *>(copy_dev + copy.size());
        auto *copy_out = reinterpret_cast<__nv_bfloat16 *>(copy_dev + copy.size() + out_bytes);
        if (cudaMemcpy(buf, index.data(), index_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(copy_dev, copy.data(), copy.size(), cudaMemcpyHostToDevice) == cudaSuccess) {
            st = cere::k4::gather(w.table, w.table_rows, rows_dev, int64_t(n), w.table_scale, alias_out, nullptr);
            if (st == CERE_OK) {
                st = cere::k4::gather(copy_dev, int64_t(n), iota_dev, int64_t(n), w.table_scale, copy_out, nullptr);
            }
            if (st == CERE_OK && (cudaMemcpy(out_alias, alias_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                                  cudaMemcpy(out_copy, copy_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)) {
                st = CERE_ERR_INTERNAL;
            }
        }
        if (cudaFree(buf) != cudaSuccess) st = CERE_ERR_INTERNAL;
    }
    if (cudaSetDevice(previous) != cudaSuccess) st = CERE_ERR_INTERNAL;
    return st;
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_model_put(uint64_t model, uint64_t tensor_index,
                                                                           uint64_t offset, const void *bytes,
                                                                           uint64_t len, cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    return cere_model_put(k->executor, k->model, tensor_index, offset, bytes, len);
}

// M2.16 L4d (the dev loop's kept PLE segment): `cere_model_host_resident` of HOST_PINNED tensor `tensor_index`;
// out_ptr[0] (host) = the address of its read-only resident bytes in the attached segment, valid until the model is
// released. CERE_ERR_UNSUPPORTED when the region is private or being created by this process (put it instead);
// CERE_ERR_ARGUMENT as the ABI function (non-HOST_PINNED or empty tensor, already covered, sealed model). No device
// work, so the stream is not used.
extern "C" __attribute__((visibility("default"))) int cere_ktest_model_host_resident(uint64_t model,
                                                                                     uint64_t tensor_index,
                                                                                     uint64_t *out_ptr,
                                                                                     cudaStream_t stream) {
    (void)stream;
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out_ptr == nullptr) return CERE_ERR_ARGUMENT;
    const void *resident = nullptr;
    const cere_status st = cere_model_host_resident(k->executor, k->model, tensor_index, &resident);
    if (st == CERE_OK) out_ptr[0] = uint64_t(reinterpret_cast<uintptr_t>(resident));
    return st;
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_model_seal(uint64_t model, cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere_model_seal(k->executor, k->model);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_model_release(uint64_t model, cudaStream_t stream) {
    K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (const cere_status st = cere_model_release(k->executor, k->model); st != CERE_OK) return st;
    k->model = nullptr;
    if (const cere_status st = cere_executor_destroy(k->executor); st != CERE_OK) return st;
    delete k;
    return CERE_OK;
}

namespace cere::ktest {
namespace {

// cere_ktest_generate's request with `flags` as its cere_work.flags and `sampling` (nullable: greedy) as its
// cere_work.sampling; `spec` (nullable, [3]) receives the completion's verify_steps and accepted_drafts and the
// session's verified_drafts (m29a design note A5; a fresh session, so this request's).
cere_status generate_work(uint64_t model, const uint32_t *prompt, uint64_t prompt_len, uint64_t max_new_tokens,
                          uint64_t prefill_chunk_tokens, uint64_t cancel_after, uint32_t flags,
                          const cere_sampling_params *sampling, uint32_t *out_tokens, cere_token_score *out_scores,
                          uint64_t *out_result, uint64_t *spec, cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || prompt == nullptr || out_tokens == nullptr || out_scores == nullptr || out_result == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    out_result[0] = out_result[1] = out_result[2] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    if (const cere_status st = cere_executor_capabilities(k->executor, k->model, &caps); st != CERE_OK) return st;
    cere::ktest::SessionScope session(*k);
    if (const cere_status st = session.create(); st != CERE_OK) return st;
    cere::ktest::ChunkScope chunk(cere::ktest::cuda_model(*k), prefill_chunk_tokens);

    cere_work work{};
    work.struct_size = sizeof(work);
    work.abi_version = CERE_EXECUTOR_ABI_VERSION;
    work.operation = CERE_OP_PREFILL;
    work.expected_worker_epoch = caps.worker_epoch;
    work.expected_session_revision = 0;
    work.session = session.get();
    work.tokens = prompt;
    work.token_count = prompt_len;
    work.max_new_tokens = max_new_tokens;
    work.flags = flags;
    work.sampling = sampling;
    work.out_tokens = out_tokens;
    work.out_capacity = max_new_tokens;
    work.out_scores = out_scores;
    cere_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
    batch.expected_worker_epoch = caps.worker_epoch;
    batch.capability_profile_generation = caps.profile_generation;
    batch.items = &work;
    batch.item_count = 1;
    cere_completion_ticket ticket{};
    if (const cere_status st = cere_executor_submit(k->executor, &batch, &ticket); st != CERE_OK) return st;

    cere_completion done{};
    bool cancelled = false;
    for (;;) {
        done = cere_completion{};
        done.struct_size = sizeof(done);
        done.abi_version = CERE_EXECUTOR_ABI_VERSION;
        const cere_status st = cere_executor_poll(k->executor, ticket, &done);
        if (st == CERE_OK) break;
        if (st != CERE_PENDING) return st;
        // The token boundary right after the cancel_after-th committed token.
        if (cancel_after != 0 && !cancelled && done.committed_tokens >= cancel_after) {
            if (const cere_status c = cere_executor_cancel(k->executor, ticket); c != CERE_OK) return c;
            cancelled = true;
        }
    }
    out_result[0] = done.committed_tokens;
    out_result[1] = done.finish;
    out_result[2] = cere::ktest::device_bytes_in_use();
    if (spec != nullptr) {
        spec[0] = done.verify_steps;
        spec[1] = done.accepted_drafts;
        spec[2] = session.get()->verified_drafts;
    }
    if (done.finish == CERE_FINISH_LENGTH || done.finish == CERE_FINISH_STOP_TOKEN ||
        done.finish == CERE_FINISH_CANCELLED) {
        return CERE_OK;
    }
    return done.status != CERE_OK ? done.status : CERE_ERR_INTERNAL;
}

}  // namespace
}  // namespace cere::ktest

extern "C" __attribute__((visibility("default"))) int cere_ktest_generate(
    uint64_t model, const uint32_t *prompt, uint64_t prompt_len, uint64_t max_new_tokens,
    uint64_t prefill_chunk_tokens, uint64_t cancel_after, uint32_t *out_tokens, cere_token_score *out_scores,
    uint64_t *out_result, cudaStream_t stream) {
    return cere::ktest::generate_work(model, prompt, prompt_len, max_new_tokens, prefill_chunk_tokens, cancel_after, 0,
                                      nullptr, out_tokens, out_scores, out_result, nullptr, stream);
}

// M2.9L (m29l design note C6L): cere_ktest_generate (executor prefill policy, never cancelled) with `flags` as the
// work's cere_work.flags (e.g. CERE_WORK_LOOKUP_DRAFTS); out_result [6] = {committed_tokens, finish, device bytes,
// verify_steps, accepted_drafts, verified_drafts (m29a design note A5: drafts verified over the rounds)}.
extern "C" __attribute__((visibility("default"))) int cere_ktest_generate_flags(
    uint64_t model, const uint32_t *prompt, uint64_t prompt_len, uint64_t max_new_tokens, uint64_t flags,
    uint32_t *out_tokens, cere_token_score *out_scores, uint64_t *out_result, cudaStream_t stream) {
    if (out_result == nullptr || flags > UINT32_MAX) return CERE_ERR_ARGUMENT;
    out_result[3] = out_result[4] = out_result[5] = 0;
    return cere::ktest::generate_work(model, prompt, prompt_len, max_new_tokens, 0, 0, uint32_t(flags), nullptr,
                                      out_tokens, out_scores, out_result, out_result + 3, stream);
}

// Review M1 (m29a design note A2 [REVISED: review-wave-fg M1]): cere_ktest_generate_flags with `sampling` (host,
// one cere_sampling_params, non-null) as the work's cere_work.sampling: a seeded sampled request under any flags.
extern "C" __attribute__((visibility("default"))) int cere_ktest_generate_sampled_flags(
    uint64_t model, const uint32_t *prompt, uint64_t prompt_len, uint64_t max_new_tokens, uint64_t flags,
    const cere_sampling_params *sampling, uint32_t *out_tokens, cere_token_score *out_scores, uint64_t *out_result,
    cudaStream_t stream) {
    if (out_result == nullptr || sampling == nullptr || flags > UINT32_MAX) return CERE_ERR_ARGUMENT;
    out_result[3] = out_result[4] = out_result[5] = 0;
    return cere::ktest::generate_work(model, prompt, prompt_len, max_new_tokens, 0, 0, uint32_t(flags), sampling,
                                      out_tokens, out_scores, out_result, out_result + 3, stream);
}

// PREFILL tokens[0, P) with max_new_tokens 1 (scores[0] predicts tokens[P]), then for j = 1 .. C - 1 the session's
// last committed, not yet fed token is overwritten with tokens[P + j - 1] and one DECODE with max_new_tokens 1 is
// run (scores[j] predicts tokens[P + j]): the runtime's own prefill-then-decode path, fed the given tokens instead of
// its own picks. The overwrite touches only host session state that no ABI call exposes (ktest-only forcing).
extern "C" __attribute__((visibility("default"))) int cere_ktest_forced_decode(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, uint64_t prompt_tokens, cere_token_score *scores,
    cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || tokens == nullptr || scores == nullptr || prompt_tokens == 0 || prompt_tokens >= token_count ||
        token_count > k->model->plan.max_context_tokens) {
        return CERE_ERR_ARGUMENT;
    }
    // The forced tokens tokens[P, T - 1) bypass the submit's vocab check (only the prompt is submitted), so they are
    // checked here; tokens[T - 1] is only predicted, never fed.
    const uint32_t vocab = k->model->plan.config.vocab_size;
    for (uint64_t i = prompt_tokens; i + 1 < token_count; ++i) {
        if (tokens[i] >= vocab) return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    if (const cere_status st = cere_executor_capabilities(k->executor, k->model, &caps); st != CERE_OK) return st;
    cere::ktest::SessionScope session(*k);
    if (const cere_status st = session.create(); st != CERE_OK) return st;
    cere::ktest::ChunkScope chunk(cere::ktest::cuda_model(*k), 0);
    cere_session &s = *session.get();

    uint64_t revision = 0;
    // One single-token request: PREFILL of the prompt (j = 0) or DECODE of the forced token; polled to terminal.
    auto request = [&](uint64_t j) -> cere_status {
        uint32_t out_token = 0;
        cere_work work{};
        work.struct_size = sizeof(work);
        work.abi_version = CERE_EXECUTOR_ABI_VERSION;
        work.operation = j == 0 ? CERE_OP_PREFILL : CERE_OP_DECODE;
        work.expected_worker_epoch = caps.worker_epoch;
        work.expected_session_revision = revision;
        work.session = session.get();
        work.tokens = j == 0 ? tokens : nullptr;
        work.token_count = j == 0 ? prompt_tokens : 0;
        work.max_new_tokens = 1;
        work.out_tokens = &out_token;
        work.out_capacity = 1;
        work.out_scores = scores + j;
        cere_batch batch{};
        batch.struct_size = sizeof(batch);
        batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
        batch.expected_worker_epoch = caps.worker_epoch;
        batch.capability_profile_generation = caps.profile_generation;
        batch.items = &work;
        batch.item_count = 1;
        cere_completion_ticket ticket{};
        if (const cere_status st = cere_executor_submit(k->executor, &batch, &ticket); st != CERE_OK) return st;
        cere_completion done{};
        for (;;) {
            done = cere_completion{};
            done.struct_size = sizeof(done);
            done.abi_version = CERE_EXECUTOR_ABI_VERSION;
            const cere_status st = cere_executor_poll(k->executor, ticket, &done);
            if (st == CERE_OK) break;
            if (st != CERE_PENDING) return st;
        }
        revision = done.session_revision;
        if (done.committed_tokens != 1) return done.status != CERE_OK ? done.status : CERE_ERR_INTERNAL;
        // A stop token is accepted: the next overwrite replaces it.
        if (done.finish == CERE_FINISH_LENGTH || done.finish == CERE_FINISH_STOP_TOKEN) return CERE_OK;
        return done.status != CERE_OK ? done.status : CERE_ERR_INTERNAL;
    };

    const uint64_t completion = token_count - prompt_tokens;
    for (uint64_t j = 0; j < completion; ++j) {
        if (j != 0) {
            if (s.in_flight != 0 || s.tokens.size() != prompt_tokens + j || s.fed + 1 != s.tokens.size()) {
                return CERE_ERR_INTERNAL;
            }
            s.tokens.back() = tokens[prompt_tokens + j - 1];
        }
        if (const cere_status st = request(j); st != CERE_OK) return st;
    }
    return CERE_OK;
}

namespace cere::ktest {
namespace {

// The body of cere_ktest_forward_trace, cere_ktest_forward_trace_detail and cere_ktest_forward_attn_inputs: PREFILL
// of tokens in chunks through device.forward with taps on the passes that hold the last `rows` positions.
// scores == nullptr: block inputs only (ForwardTaps.attn_inputs_only), nothing scored. `detail` is ForwardTaps.detail
// (0 = forward_trace itself): detail entries go through the same sink.
cere_status trace(K8Model *k, const uint32_t *tokens, uint64_t token_count, uint64_t prefill_chunk_tokens,
                  uint64_t last_rows, uint32_t detail, uint8_t *names, uint64_t names_capacity, int64_t *shapes,
                  uint64_t entry_capacity, void *values, uint64_t values_capacity_bytes, cere_token_score *scores,
                  uint64_t *out_counts, cudaStream_t stream) {
    if (k == nullptr || tokens == nullptr || token_count == 0 || names == nullptr || shapes == nullptr ||
        values == nullptr || out_counts == nullptr || detail > 1) {
        return CERE_ERR_ARGUMENT;
    }
    const bool attn_inputs_only = scores == nullptr;
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    k->trace_passes = 0;
    const uint64_t rows = (last_rows == 0 || last_rows > token_count) ? token_count : last_rows;
    // `values` may still be written by the caller's stream (e.g. its NaN fill).
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere::ktest::SessionScope session(*k);
    if (const cere_status st = session.create(); st != CERE_OK) return st;
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    auto &csession = static_cast<cere::device::CudaSession &>(*session.get()->device);

    const uint64_t ws_rows = cmodel.workspace_rows();
    const uint64_t chunk = (prefill_chunk_tokens == 0 || prefill_chunk_tokens > ws_rows) ? ws_rows
                                                                                         : prefill_chunk_tokens;
    cere::ktest::TraceSink sink;
    sink.names = names;
    sink.names_capacity = names_capacity;
    sink.shapes = shapes;
    sink.entry_capacity = entry_capacity;
    sink.values = static_cast<uint8_t *>(values);
    sink.values_capacity = values_capacity_bytes;
    sink.rows = rows;
    const uint64_t first_reported = token_count - rows;
    cere_token_score last{};
    cere_token_score unscored{};  // forward's out_top2 when nothing is scored (attn_inputs_only copies none)
    for (uint64_t p0 = 0; p0 < token_count;) {
        const uint64_t n = std::min(chunk, token_count - p0);
        const uint64_t end = p0 + n;
        cere_status st;
        if (end <= first_reported) {
            st = device.forward(cmodel, csession, CERE_OP_PREFILL, tokens + p0, uint32_t(n), nullptr, &last);
        } else {
            const uint64_t start = std::max(p0, first_reported);
            cere::model::ForwardTaps taps;
            taps.last_rows = end - start;
            taps.ctx = &sink;
            taps.emit = &cere::ktest::TraceSink::emit;
            if (detail != 0) {
                taps.detail = detail;
                taps.detail_sink = &cere::ktest::TraceSink::emit;
            }
            taps.attn_inputs_only = attn_inputs_only;
            sink.row0 = start - first_reported;
            sink.entry = 0;
            sink.name_at = 0;
            sink.value_at = 0;
            st = device.forward(cmodel, csession, CERE_OP_PREFILL, tokens + p0, uint32_t(n), &taps,
                                attn_inputs_only ? &unscored : scores + sink.row0);
            if (st == CERE_OK && !sink.first && sink.entry != sink.entries) st = CERE_ERR_INTERNAL;
            sink.first = false;
        }
        if (st != CERE_OK) return sink.error != CERE_OK ? sink.error : st;
        ++k->trace_passes;
        p0 = end;
    }
    out_counts[0] = sink.entries;
    out_counts[1] = sink.name_bytes;
    out_counts[2] = sink.value_bytes;
    return CERE_OK;
}

}  // namespace
}  // namespace cere::ktest

extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_trace(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, uint64_t prefill_chunk_tokens, uint64_t last_rows,
    uint8_t *names, uint64_t names_capacity, int64_t *shapes, uint64_t entry_capacity, void *values,
    uint64_t values_capacity_bytes, cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    if (scores == nullptr) return CERE_ERR_ARGUMENT;
    return cere::ktest::trace(cere::ktest::handle(model), tokens, token_count, prefill_chunk_tokens, last_rows, 0,
                              names, names_capacity, shapes, entry_capacity, values, values_capacity_bytes, scores,
                              out_counts, stream);
}

// forward_trace with the m22p design note D3 detail names (`detail` 1; 0 = forward_trace): every reporting chunk
// emits, per layer in forward order, the B8 entries with the detail entries of model.h ForwardTaps between them,
// into the same names / shapes / values buffers (each entry [rows, its cols]). detail > 1 is CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_trace_detail(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, uint64_t prefill_chunk_tokens, uint64_t last_rows,
    uint32_t detail, uint8_t *names, uint64_t names_capacity, int64_t *shapes, uint64_t entry_capacity, void *values,
    uint64_t values_capacity_bytes, cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    if (scores == nullptr) return CERE_ERR_ARGUMENT;
    return cere::ktest::trace(cere::ktest::handle(model), tokens, token_count, prefill_chunk_tokens, last_rows, detail,
                              names, names_capacity, shapes, entry_capacity, values, values_capacity_bytes, scores,
                              out_counts, stream);
}

// m22p design note D5: the prompt alone as one PREFILL forward (token_count <= the workspace rows) on a fresh
// session, with every named P point of model.h ForwardTaps replaced. `names`: name_bytes host bytes of NUL-terminated
// distinct names, each a P point run once per forward (the B8 blocks `model.layers.L.{self_attn.o_proj |
// linear_attn.out_proj, mlp.experts, mlp.shared_expert.down_proj}` of L's kind, `native.layers.L.{attn,mlp}_mix.out`,
// `native.final_mix.out`, `native.layers.L.gemm.<plan>.out` for attn_qkv / attn_o or gdn_qkv / gdn_z / gdn_out,
// moe_router, moe_shared_down); `values`: device BF16, per name in order [token_count, its cols] back to back,
// values_bytes in total. Outputs: the last position's BF16 logit of `target_token` (as float) and its top-2 entry
// (`out_top1`, host). CERE_ERR_ARGUMENT, nothing written: a null or zero argument, an unknown or repeated name, a
// values_bytes other than the names' sum, target_token >= vocab, token_count above the workspace rows.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_patch(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, const uint8_t *names, uint64_t name_bytes,
    const void *values, uint64_t values_bytes, uint32_t target_token, float *out_target_logit,
    cere_token_score *out_top1, cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || tokens == nullptr || token_count == 0 || names == nullptr || name_bytes == 0 ||
        values == nullptr || values_bytes == 0 || out_target_logit == nullptr || out_top1 == nullptr ||
        names[name_bytes - 1] != 0) {
        return CERE_ERR_ARGUMENT;
    }
    const cere_qwen4_exp_config &cfg = k->model->plan.config;
    if (target_token >= cfg.vocab_size) return CERE_ERR_ARGUMENT;
    cere::ktest::PatchRun run;
    run.target = target_token;
    const auto *base = static_cast<const cere::model::bf16 *>(values);
    uint64_t at = 0;  // elements
    for (uint64_t i = 0; i < name_bytes;) {
        const std::string_view name(reinterpret_cast<const char *>(names + i));
        const uint64_t cols = cere::ktest::patch_cols(cfg, name);
        if (cols == 0) return CERE_ERR_ARGUMENT;
        for (const auto &entry : run.list) {
            if (entry.first == name) return CERE_ERR_ARGUMENT;
        }
        run.list.emplace_back(std::string(name), base + at);
        at += token_count * cols;
        i += name.size() + 1;
    }
    if (at * sizeof(cere::model::bf16) != values_bytes) return CERE_ERR_ARGUMENT;
    run.used.assign(run.list.size(), false);
    return cere::ktest::patched_forward(*k, tokens, token_count, run, out_target_logit, out_top1, stream);
}

// forward_patch with `patch` itself as ForwardTaps.patch (every I and P point of model.h ForwardTaps, `patch_ctx`
// passed through; a returned replacement must stay valid until the next callback). The callback never sees
// native.logits (forward_patch_screen reads the target logit there). CERE_ERR_ARGUMENT, nothing written: a null or
// zero argument, target_token >= vocab, token_count above the workspace rows; a non-OK forward status is returned.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_patch_screen(
    uint64_t model, const uint32_t *tokens, uint64_t token_count,
    const bf16 *(*patch)(void *ctx, const char *name, uint64_t layer, uint64_t rows, uint64_t cols, const bf16 *native,
                         cudaStream_t stream),
    void *patch_ctx,
    uint32_t target_token, float *out_target_logit, cere_token_score *out_top1, cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || tokens == nullptr || token_count == 0 || patch == nullptr || out_target_logit == nullptr ||
        out_top1 == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (target_token >= k->model->plan.config.vocab_size) return CERE_ERR_ARGUMENT;
    cere::ktest::PatchRun run;
    run.target = target_token;
    run.user = patch;
    run.user_ctx = patch_ctx;
    return cere::ktest::patched_forward(*k, tokens, token_count, run, out_target_logit, out_top1, stream);
}

// cere_ktest_forward_trace with every row reported and ForwardTaps.attn_inputs_only: one entry
// "model.layers.L.self_attn.input" [token_count, 2,560] BF16 per full-attention layer (the block input after the HC
// mix), no scores; same chunking, CAPACITY rules and out_counts {entries, name bytes, value bytes} (K9 oracle on real
// activations, m22c Z2AC-04).
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_attn_inputs(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, uint64_t prefill_chunk_tokens, uint8_t *names,
    uint64_t names_capacity, int64_t *shapes, uint64_t entry_capacity, void *values, uint64_t values_capacity_bytes,
    uint64_t *out_counts, cudaStream_t stream) {
    return cere::ktest::trace(cere::ktest::handle(model), tokens, token_count, prefill_chunk_tokens, token_count, 0,
                              names, names_capacity, shapes, entry_capacity, values, values_capacity_bytes, nullptr,
                              out_counts, stream);
}

namespace cere::ktest {
namespace {

// cere_ktest_forward_layout_probe's ForwardTaps.patch callback state. Entries are the point occurrences of one forward
// call in forward order (the first call of layout A names them; every later call must repeat names and columns).
struct LayoutSink {
    static constexpr uint64_t kNoSlot = ~uint64_t(0);
    uint64_t w0 = 0;     // the window [w0, w0 + width)
    uint64_t width = 0;
    uint8_t *names = nullptr;
    uint64_t names_capacity = 0;
    uint64_t entry_capacity = 0;
    uint64_t name_bytes = 0;
    std::vector<uint64_t> name_at;  // per entry: its name's byte offset in `names` (reserved: no allocation)
    std::vector<uint64_t> cols;     // per entry
    std::vector<uint64_t> slot_at;  // per entry: its arena offset (P points), else kNoSlot
    uint8_t *arena = nullptr;       // device, the recorded window rows of A's P points (replace only)
    uint64_t arena_bytes = 0;
    uint64_t arena_need = 0;
    uint64_t *device = nullptr;  // device u64 [entry_capacity][width], the current layout's digests
    bool record = false;         // copy every P point's window rows into the arena (layout A with replace)
    bool replay = false;         // return the arena rows as every P point's replacement (layout B with replace)
    bool first = true;           // the naming call
    uint64_t call_start = 0;
    uint64_t call_rows = 0;
    uint64_t entry = 0;
    cere_status error = CERE_OK;

    // An I point (model.h ForwardTaps: "*.in", "*.normed"): its return is ignored, so it is never recorded.
    static bool input_point(const char *name) {
        const size_t len = std::strlen(name);
        return (len >= 3 && std::strcmp(name + len - 3, ".in") == 0) ||
               (len >= 7 && std::strcmp(name + len - 7, ".normed") == 0);
    }

    static const model::bf16 *patch(void *ctx, const char *name, uint64_t /*layer*/, uint64_t rows, uint64_t cols,
                                    const model::bf16 *native, cudaStream_t stream) {
        auto &s = *static_cast<LayoutSink *>(ctx);
        if (s.error != CERE_OK || std::strcmp(name, "native.logits") == 0) return nullptr;
        if (s.first) {
            const size_t len = std::strlen(name) + 1;
            if (s.name_at.size() >= s.entry_capacity || len > s.names_capacity - s.name_bytes) {
                s.error = CERE_ERR_CAPACITY;
                return nullptr;
            }
            std::memcpy(s.names + s.name_bytes, name, len);
            s.name_at.push_back(s.name_bytes);
            s.name_bytes += len;
            s.cols.push_back(cols);
            uint64_t slot = kNoSlot;
            if (!input_point(name)) {
                slot = s.arena_need;
                s.arena_need += (s.width * cols * sizeof(model::bf16) + 255) / 256 * 256;
            }
            s.slot_at.push_back(slot);
        } else if (s.entry >= s.name_at.size() || s.cols[s.entry] != cols ||
                   std::strcmp(reinterpret_cast<const char *>(s.names + s.name_at[s.entry]), name) != 0) {
            s.error = CERE_ERR_INTERNAL;
            return nullptr;
        }
        if (rows != s.call_rows) {  // every point but native.logits carries the call's rows
            s.error = CERE_ERR_INTERNAL;
            return nullptr;
        }
        const uint64_t e = s.entry++;
        if (s.call_start < s.w0) return nullptr;  // before the window: the same calls in both layouts
        const uint64_t row0 = s.call_start - s.w0;
        if (row_digests(reinterpret_cast<const uint16_t *>(native), rows, cols, s.device + e * s.width + row0,
                        stream) != cudaSuccess) {
            cudaGetLastError();
            s.error = CERE_ERR_INTERNAL;
            return nullptr;
        }
        if (s.slot_at[e] == kNoSlot || (!s.record && !s.replay)) return nullptr;
        // Past the arena only during the naming call: it finishes the count, then the caller reports CAPACITY.
        if (s.slot_at[e] + s.width * cols * sizeof(model::bf16) > s.arena_bytes) return nullptr;
        model::bf16 *slot = reinterpret_cast<model::bf16 *>(s.arena + s.slot_at[e]) + row0 * cols;
        if (s.replay) return slot;
        if (cudaMemcpyAsync(slot, native, rows * cols * sizeof(model::bf16), cudaMemcpyDeviceToDevice, stream) !=
            cudaSuccess) {
            cudaGetLastError();
            s.error = CERE_ERR_INTERNAL;
        }
        return nullptr;
    }
};

// Call ends of one layout: strictly increasing, the last n, each call at most `ws_rows`, none straddling w0.
bool layout_ends(const uint64_t *ends, uint64_t k, uint64_t n, uint64_t w0, uint64_t ws_rows) {
    for (uint64_t i = 0, previous = 0; i < k; previous = ends[i++]) {
        if (ends[i] <= previous || ends[i] - previous > ws_rows || (previous < w0 && ends[i] > w0)) return false;
    }
    return ends[k - 1] == n;
}

}  // namespace
}  // namespace cere::ktest

// K16 (10), m210 design note D15⁗′ (the layout probe): the PREFILL of tokens[0, n) (host) on a fresh session through
// CudaDevice::forward in layout A (calls ending at ends[0, ka)), then on another fresh session in layout B (ends[ka,
// ka + kb)); each layout's ends are strictly increasing, the last n, each call at most the workspace rows and none
// straddling w0 (< n), and both layouts end the same calls at or below w0. ForwardTaps.patch observes every I and P
// point of model.h ForwardTaps (a patch routes GDN layers through the decomposed K5 path, the same launches); per point
// occurrence in forward order (gdn_ba and moe_shared_gate_up run twice per layer: two entries each; native.logits is
// none) the row_digests (ktest/row_digest.h) of the window rows [w0, n) of each layout. With `replace` 1, A records
// every P point's window rows in a device arena of arena_bytes and B's P points are replaced by A's rows after their
// digest, so a B digest that differs from A's at a point whose inputs equal A's is that kernel's own layout dependence
// (I points are never replaced: an attention or GDN core output feeds the following projection as computed).
// names: host [names_capacity], NUL-terminated per entry; digests: host u64 [2][entry_capacity][n - w0] (layout-major,
// then entry, then window row; entries past out_counts[0] untouched); scores: host [2], each layout's last-row top-2;
// out_counts: host [3] = {entries, name bytes, arena bytes a replace needs}. CERE_ERR_CAPACITY: more entries or name
// bytes than the capacities, or (replace) arena_bytes below out_counts[2]; CERE_ERR_INTERNAL: a call's point sequence
// differs from the first call's.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_layout_probe(
    uint64_t model, const uint32_t *tokens, uint64_t n, const uint64_t *ends, uint64_t ka, uint64_t kb, uint64_t w0,
    uint64_t replace, uint64_t arena_bytes, uint8_t *names, uint64_t names_capacity, uint64_t *digests,
    uint64_t entry_capacity, cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    K8Model *km = cere::ktest::handle(model);
    if (km == nullptr || tokens == nullptr || n == 0 || ends == nullptr || ka == 0 || kb == 0 || w0 >= n ||
        replace > 1 || names == nullptr || names_capacity == 0 || digests == nullptr || entry_capacity == 0 ||
        entry_capacity > (uint64_t(1) << 20) || n - w0 > (uint64_t(1) << 24) || scores == nullptr ||
        out_counts == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*km);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*km);
    const uint64_t ws_rows = cmodel.workspace_rows();
    const uint64_t *const layout_end[2] = {ends, ends + ka};
    const uint64_t layout_calls[2] = {ka, kb};
    uint64_t below[2] = {0, 0};  // calls ending at or below w0
    for (int l = 0; l < 2; ++l) {
        if (!cere::ktest::layout_ends(layout_end[l], layout_calls[l], n, w0, ws_rows)) return CERE_ERR_ARGUMENT;
        while (layout_end[l][below[l]] <= w0) ++below[l];
    }
    if (below[0] != below[1] || !std::equal(ends, ends + below[0], ends + ka)) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere::ktest::LayoutSink sink;
    try {
        sink.name_at.reserve(entry_capacity);
        sink.cols.reserve(entry_capacity);
        sink.slot_at.reserve(entry_capacity);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    const uint64_t width = n - w0;
    const size_t digest_bytes = entry_capacity * width * sizeof(uint64_t);
    uint64_t *device_digests = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_digests), 2 * digest_bytes) != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_CAPACITY;
    }
    cere_status st = cudaMemset(device_digests, 0, 2 * digest_bytes) == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
    if (st == CERE_OK && replace != 0 &&
        (arena_bytes == 0 || cudaMalloc(reinterpret_cast<void **>(&sink.arena), arena_bytes) != cudaSuccess)) {
        st = CERE_ERR_CAPACITY;
    }
    sink.w0 = w0;
    sink.width = width;
    sink.names = names;
    sink.names_capacity = names_capacity;
    sink.entry_capacity = entry_capacity;
    sink.arena_bytes = replace != 0 ? arena_bytes : 0;
    cere_token_score score[2] = {};
    for (int l = 0; l < 2 && st == CERE_OK; ++l) {
        cere::ktest::SessionScope session(*km);
        if ((st = session.create()) != CERE_OK) break;
        // D3′: a pooled model's session is lazy; bind its own pages as the first prefill round does (not namespaced,
        // so no lookup: the probe never hits).
        cere::host::PrefixAttach attached;
        if ((st = cere::host::attach_prefix(*km->executor, *session.get(), km->executor->last_submission + 1,
                                            &attached)) != CERE_OK) {
            break;
        }
        auto &csession = static_cast<cere::device::CudaSession &>(*session.get()->device);
        sink.device = device_digests + l * entry_capacity * width;
        sink.record = replace != 0 && l == 0;
        sink.replay = replace != 0 && l == 1;
        for (uint64_t i = 0, start = 0; i < layout_calls[l] && st == CERE_OK; start = layout_end[l][i++]) {
            cere::model::ForwardTaps taps;
            taps.last_rows = 1;
            taps.ctx = &sink;
            taps.patch = &cere::ktest::LayoutSink::patch;
            sink.call_start = start;
            sink.call_rows = layout_end[l][i] - start;
            sink.entry = 0;
            st = device.forward(cmodel, csession, CERE_OP_PREFILL, tokens + start,
                                uint32_t(layout_end[l][i] - start), &taps, &score[l]);
            if (sink.error != CERE_OK) st = sink.error;
            if (st == CERE_OK && sink.entry != sink.name_at.size()) st = CERE_ERR_INTERNAL;
            if (sink.first && replace != 0 && sink.arena_need > arena_bytes) st = CERE_ERR_CAPACITY;
            sink.first = false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess && st == CERE_OK) st = CERE_ERR_INTERNAL;
    }
    const uint64_t entries = sink.name_at.size();
    for (int l = 0; l < 2 && st == CERE_OK; ++l) {
        if (cudaMemcpy(digests + l * entry_capacity * width, device_digests + l * entry_capacity * width,
                       entries * width * sizeof(uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            st = CERE_ERR_INTERNAL;
        }
    }
    if (cudaFree(device_digests) != cudaSuccess && st == CERE_OK) st = CERE_ERR_INTERNAL;
    if (sink.arena != nullptr && cudaFree(sink.arena) != cudaSuccess && st == CERE_OK) st = CERE_ERR_INTERNAL;
    cudaGetLastError();
    out_counts[0] = entries;
    out_counts[1] = sink.name_bytes;
    out_counts[2] = sink.arena_need;
    if (st != CERE_OK) return st;
    scores[0] = score[0];
    scores[1] = score[1];
    return CERE_OK;
}

// The prefill passes (forward calls) of the last cere_ktest_forward_trace on `model` (0 before the first, or after a
// failed one before its first pass): ceil(token_count / chunk), chunk = the clamped prefill_chunk_tokens.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_trace_passes(uint64_t model,
                                                                                      uint64_t *out_passes,
                                                                                      cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out_passes == nullptr) return CERE_ERR_ARGUMENT;
    *out_passes = k->trace_passes;
    return CERE_OK;
}

// The model's device memory plan (host arithmetic, no CUDA call): out[0] workspace rows, out[1] workspace_bytes at the
// model's context, out[2] workspace_bytes for the same rows at the dense limit (k6::kMaxContext: no K9 scratch), out[3]
// derived_bytes, out[4] session_bytes (m22c C10/D5).
extern "C" __attribute__((visibility("default"))) int cere_ktest_model_plan_bytes(uint64_t model, uint64_t *out,
                                                                                  cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const cere::host::ModelPlan &plan = k->model->plan;
    const uint64_t ctx = plan.max_context_tokens;
    const uint64_t rows = cere::model::workspace_rows(ctx);
    out[0] = rows;
    out[1] = cere::model::workspace_bytes(plan.config, rows, ctx);
    out[2] = cere::model::workspace_bytes(plan.config, rows, std::min<uint64_t>(ctx, cere::k6::kMaxContext));
    out[3] = cere::model::derived_bytes(plan.config, ctx);
    out[4] = cere::model::session_bytes(plan.config, ctx);
    return CERE_OK;
}

// The model's ALLOCATE estimate (host arithmetic, no CUDA call; M2.11 R1): out[0] device_bytes, out[1] host_bytes (the
// mapped host block plus the snapshot arena, cere_model_desc.snapshot_host_bytes; memory_plan.h pinned_bytes).
// ALLOCATE takes exactly these in one cudaMalloc, one cudaHostAlloc and the arena's own locked allocation
// (cuda_device.h) and nothing later, so they are the model's peaks.
extern "C" __attribute__((visibility("default"))) int cere_ktest_model_memory(uint64_t model, uint64_t *out,
                                                                              cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    out[0] = k->model->estimate.device_bytes;
    out[1] = k->model->estimate.host_bytes;
    return CERE_OK;
}

// The executor's memory ledger (cuda_device.h, host/memory_ledger.h; M2.11 R1 [REVISED 2026-10-11]), exact bytes
// held by the model's executor right now: out[0] device (its cudaMalloc), out[1] the pinned bounce ring, out[2] the
// mapped host blocks, out[3] the mlock'd arena pages. Host arithmetic, no CUDA call.
extern "C" __attribute__((visibility("default"))) int cere_ktest_model_ledger(uint64_t model, uint64_t *out,
                                                                              cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const cere::host::MemoryLedger &ledger = cere::ktest::cuda_device(*k).ledger();
    out[0] = ledger.held(cere::host::Held::kDevice);
    out[1] = ledger.held(cere::host::Held::kRing);
    out[2] = ledger.held(cere::host::Held::kMapped);
    out[3] = ledger.held(cere::host::Held::kLocked);
    return CERE_OK;
}

// K11 (m25 design note D9, ledger R as amended by the owner 2026-10-10: no ktest session handle before M2.11): arms
// the page order the model's next session create applies to the pages it reserves (host/kv_pool.h PagePermutation).
// `order` (n entries, host) is a permutation of [0, n), n = ceil(context / 64) of that create: with `reserved` the n
// lowest free pages in ascending order, the session's page i becomes reserved[order[i]]. One-shot: the next create
// consumes it whatever its outcome, and model release drops it. CERE_ERR_ARGUMENT for a malformed order.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_set_pages(uint64_t model,
                                                                                   const uint32_t *order, uint64_t n,
                                                                                   cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    return cere::ktest::cuda_model(*k).arm_page_order(order, n);
}

// K11 D9 (3), the pool's create-time reservation through the real ABI: with every session at the model's context,
// out[0] = the free pages before, out[1] = create A, out[2] = the free pages after A, out[3] = create B while A
// lives, out[4] = the free pages after B's attempt, out[5] = create B after A's release (CERE_ERR_INTERNAL when the
// first B already succeeded), out[6] = the free pages after it (statuses as cere_status, sign-extended). The
// sessions are released before returning.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_pool_check(uint64_t model, int64_t *out,
                                                                                    cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    cere::device::CudaModel &m = cere::ktest::cuda_model(*k);
    out[0] = int64_t(m.free_pages());
    {
        cere::ktest::SessionScope b(*k);
        {
            cere::ktest::SessionScope a(*k);
            out[1] = a.create();
            out[2] = int64_t(m.free_pages());
            out[3] = b.create();
            out[4] = int64_t(m.free_pages());
        }
        out[5] = b.get() == nullptr ? b.create() : CERE_ERR_INTERNAL;
        out[6] = int64_t(m.free_pages());
    }
    return CERE_OK;
}

// cere_ktest_forced_decode's requests (PREFILL of tokens[0, P), then for j = 1 .. C - 1 one DECODE of the forced
// token tokens[P + j - 1]) as a direct forward loop doing what submit does with them (CudaDevice::step: the prompt as
// PREFILL in chunks of the workspace rows, the policy forced_decode runs under, then one DECODE row per step; each
// forward carries the session's KV and GDN state), the DECODE steps tapped through forward_trace's sink: entries
// step-major, one TraceSink of one row per step at the offsets of the steps before it. The size of one step comes
// from a counting probe (one tapped single-token forward on a throwaway session), so a capacity below C - 1 steps is
// refused before the PREFILL with nothing written. Tapping routes MoE through the K7 tap path; the bit-equal scores
// rows of test_model_k8_forward.py show that the loop still computes the serving path.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forced_decode_trace(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, uint64_t prompt_tokens, uint8_t *names,
    uint64_t names_capacity, int64_t *shapes, uint64_t entry_capacity, void *values, uint64_t values_capacity_bytes,
    cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || tokens == nullptr || scores == nullptr || names == nullptr || shapes == nullptr ||
        values == nullptr || out_counts == nullptr || prompt_tokens == 0 || prompt_tokens >= token_count ||
        token_count > k->model->plan.max_context_tokens) {
        return CERE_ERR_ARGUMENT;
    }
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    // Every fed token: the prompt (submit's check in forced_decode) and the forced tokens tokens[P, T - 1);
    // tokens[T - 1] is only predicted, never fed.
    const uint32_t vocab = k->model->plan.config.vocab_size;
    for (uint64_t i = 0; i + 1 < token_count; ++i) {
        if (tokens[i] >= vocab) return CERE_ERR_ARGUMENT;
    }
    // `values` may still be written by the caller's stream (e.g. its NaN fill).
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    const uint64_t completion = token_count - prompt_tokens;
    const uint64_t steps = completion - 1;

    cere::ktest::CountSink size;
    if (steps != 0) {
        cere::ktest::SessionScope probe(*k);
        if (const cere_status st = probe.create(); st != CERE_OK) return st;
        auto &psession = static_cast<cere::device::CudaSession &>(*probe.get()->device);
        cere::model::ForwardTaps taps;
        taps.last_rows = 1;
        taps.ctx = &size;
        taps.emit = &cere::ktest::CountSink::emit;
        cere_token_score ignored{};
        if (const cere_status st = device.forward(cmodel, psession, CERE_OP_PREFILL, tokens, 1, &taps, &ignored);
            st != CERE_OK) {
            return st;
        }
        if (size.entries > entry_capacity / steps || size.name_bytes > names_capacity / steps ||
            size.value_bytes > values_capacity_bytes / steps) {
            return CERE_ERR_ARGUMENT;
        }
    }

    cere::ktest::SessionScope session(*k);
    if (const cere_status st = session.create(); st != CERE_OK) return st;
    auto &csession = static_cast<cere::device::CudaSession &>(*session.get()->device);
    const uint64_t rows = cmodel.workspace_rows();
    for (uint64_t p0 = 0; p0 < prompt_tokens;) {
        const uint64_t n = std::min(rows, prompt_tokens - p0);
        if (const cere_status st =
                device.forward(cmodel, csession, CERE_OP_PREFILL, tokens + p0, uint32_t(n), nullptr, scores);
            st != CERE_OK) {
            return st;
        }
        p0 += n;
    }
    auto *const value_base = static_cast<uint8_t *>(values);
    for (uint64_t j = 1; j < completion; ++j) {
        const uint64_t step = j - 1;
        cere::ktest::TraceSink sink;
        sink.names = names + step * size.name_bytes;
        sink.names_capacity = size.name_bytes;
        sink.shapes = shapes + 3 * step * size.entries;
        sink.entry_capacity = size.entries;
        sink.values = value_base + step * size.value_bytes;
        sink.values_capacity = size.value_bytes;
        sink.rows = 1;
        cere::model::ForwardTaps taps;
        taps.last_rows = 1;
        taps.ctx = &sink;
        taps.emit = &cere::ktest::TraceSink::emit;
        const cere_status st = device.forward(cmodel, csession, CERE_OP_DECODE, tokens + prompt_tokens + j - 1, 1,
                                              &taps, scores + j);
        // The probe sized every step: a sink error means this step emitted another entry sequence.
        if (st != CERE_OK) return sink.error != CERE_OK ? CERE_ERR_INTERNAL : st;
        if (sink.entries != size.entries || sink.name_bytes != size.name_bytes ||
            sink.value_bytes != size.value_bytes || std::memcmp(sink.names, names, size.name_bytes) != 0) {
            return CERE_ERR_INTERNAL;
        }
    }
    out_counts[0] = steps * size.entries;
    out_counts[1] = steps * size.name_bytes;
    out_counts[2] = steps * size.value_bytes;
    return CERE_OK;
}

// M2.9 K8 (m29 design note I8, D4): cere_ktest_forced_decode_trace's requests with the DECODE steps fed as verify
// chains: PREFILL of tokens[0, P) as forced_decode_trace, then the C - 1 forced tokens tokens[P, T - 1) in calls of
// CudaDevice::verify_rows of up to chain + 1 rows (tokens[0] the next forced token, drafts the following ones; chain
// in [0, 3]), each committing every row. Row i of a call is DECODE step j and lands where forced_decode_trace puts
// step j: entries step-major (one RowsSink slab per row at the offsets of the steps before it), scores[j] its top-2,
// scores[0] the PREFILL's. Same arguments, capacities, probe and refusals as forced_decode_trace; the model needs
// speculative_draft_tokens > 0 (else CERE_ERR_UNSUPPORTED).
extern "C" __attribute__((visibility("default"))) int cere_ktest_forced_verify_trace(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, uint64_t prompt_tokens, uint64_t chain,
    uint8_t *names, uint64_t names_capacity, int64_t *shapes, uint64_t entry_capacity, void *values,
    uint64_t values_capacity_bytes, cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || tokens == nullptr || scores == nullptr || names == nullptr || shapes == nullptr ||
        values == nullptr || out_counts == nullptr || prompt_tokens == 0 || prompt_tokens >= token_count ||
        token_count > k->model->plan.max_context_tokens || chain > 3) {
        return CERE_ERR_ARGUMENT;
    }
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    const uint32_t vocab = k->model->plan.config.vocab_size;
    for (uint64_t i = 0; i + 1 < token_count; ++i) {
        if (tokens[i] >= vocab) return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    if (!device.draft_supported(cmodel)) return CERE_ERR_UNSUPPORTED;
    const uint64_t completion = token_count - prompt_tokens;
    const uint64_t steps = completion - 1;

    cere::ktest::CountSink size;
    if (steps != 0) {
        cere::ktest::SessionScope probe(*k);
        if (const cere_status st = probe.create(); st != CERE_OK) return st;
        auto &psession = static_cast<cere::device::CudaSession &>(*probe.get()->device);
        cere::model::ForwardTaps taps;
        taps.last_rows = 1;
        taps.ctx = &size;
        taps.emit = &cere::ktest::CountSink::emit;
        cere_token_score ignored{};
        if (const cere_status st = device.forward(cmodel, psession, CERE_OP_PREFILL, tokens, 1, &taps, &ignored);
            st != CERE_OK) {
            return st;
        }
        if (size.entries > entry_capacity / steps || size.name_bytes > names_capacity / steps ||
            size.value_bytes > values_capacity_bytes / steps) {
            return CERE_ERR_ARGUMENT;
        }
    }

    cere::ktest::SessionScope session(*k);
    if (const cere_status st = session.create(); st != CERE_OK) return st;
    cere::device::CudaSession *csession = &static_cast<cere::device::CudaSession &>(*session.get()->device);
    const uint64_t rows = cmodel.workspace_rows();
    for (uint64_t p0 = 0; p0 < prompt_tokens;) {
        const uint64_t n = std::min(rows, prompt_tokens - p0);
        if (const cere_status st =
                device.forward(cmodel, *csession, CERE_OP_PREFILL, tokens + p0, uint32_t(n), nullptr, scores);
            st != CERE_OK) {
            return st;
        }
        p0 += n;
    }
    auto *const value_base = static_cast<uint8_t *>(values);
    for (uint64_t j = 1; j < completion;) {
        const uint32_t count = uint32_t(std::min<uint64_t>(chain + 1, completion - j));  // rows of this call
        cere::ktest::RowsSink sink;
        sink.rows = count;
        sink.entries = size.entries;
        sink.name_bytes = size.name_bytes;
        sink.value_bytes = size.value_bytes;
        for (uint32_t r = 0; r < count; ++r) {
            const uint64_t step = j - 1 + r;
            sink.slab[r].names = names + step * size.name_bytes;
            sink.slab[r].shapes = shapes + 3 * step * size.entries;
            sink.slab[r].values = value_base + step * size.value_bytes;
        }
        cere::model::ForwardTaps taps;
        taps.last_rows = 1;
        taps.ctx = &sink;
        taps.emit = &cere::ktest::RowsSink::emit;
        const uint32_t *chain_tokens = tokens + prompt_tokens + j - 1;
        const uint32_t drafts = count - 1;
        const cere_status st =
            device.verify_rows(cmodel, &csession, &chain_tokens, &drafts, 1, &taps, scores + j, &count);
        if (st != CERE_OK) return sink.error != CERE_OK ? CERE_ERR_INTERNAL : st;
        if (!sink.complete()) return CERE_ERR_INTERNAL;
        for (uint32_t r = 0; r < count; ++r) {
            if (std::memcmp(sink.slab[r].names, names, size.name_bytes) != 0) return CERE_ERR_INTERNAL;
        }
        j += count;
    }
    out_counts[0] = steps * size.entries;
    out_counts[1] = steps * size.name_bytes;
    out_counts[2] = steps * size.value_bytes;
    return CERE_OK;
}

// m213 design note R3: the prompt with its images as one PREFILL forward (token_count <= the workspace rows) on a
// fresh session of a vision model. `images`: host cere_image_input [image_count] (the cere_work rules: spans sorted,
// disjoint, inside the prompt, each vision_start + exactly t*h*w/4 pads); their mRoPE positions are
// cere::mrope::positions. `sink(ctx, name, block, values, rows, cols, stream)` receives every encoder tap of
// model/vision.h (device BF16, readable in `stream` order during the call) and, with block = UINT32_MAX,
// "native.embed.spliced" ([token_count, 2,560], ws.mixed after the splice). The prompt's mRoPE positions are written
// to out_positions3 (host int64 [3, token_count]) with *out_rope_delta. out_top1: the last
// row's top-2 entry. CERE_ERR_ARGUMENT for a null or malformed argument, nothing run.
extern "C" __attribute__((visibility("default"))) int cere_ktest_vision_forward(
    uint64_t model, const uint32_t *tokens, uint64_t token_count, const cere_image_input *images, uint64_t image_count,
    int (*sink)(void *ctx, const char *name, uint32_t block, const void *values, uint64_t rows, uint64_t cols,
                cudaStream_t stream),
    void *ctx, int64_t *out_positions3, int64_t *out_rope_delta, cere_token_score *out_top1, cudaStream_t stream) {
    const cere::ktest::K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || tokens == nullptr || token_count == 0 || images == nullptr || image_count == 0 ||
        sink == nullptr || out_positions3 == nullptr || out_rope_delta == nullptr || out_top1 == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    if (token_count > cmodel.workspace_rows()) return CERE_ERR_ARGUMENT;
    try {
        std::vector<cere::host::SessionImage> owned(image_count);
        std::vector<cere::mrope::ImageSpan> spans;
        for (uint64_t i = 0; i < image_count; ++i) {
            const cere_image_input &in = images[i];
            if (in.pixel_patches == nullptr || in.pixel_bytes == 0) return CERE_ERR_ARGUMENT;
            owned[i].span = {in.token_offset, in.grid_t, in.grid_h, in.grid_w};
            owned[i].pad_id = in.pad_id;
            std::copy(in.pixel_sha256, in.pixel_sha256 + 32, owned[i].pixel_sha256.begin());
            const auto *p = static_cast<const uint8_t *>(in.pixel_patches);
            owned[i].pixels.assign(p, p + in.pixel_bytes);
            spans.push_back(owned[i].span);
        }
        std::vector<int64_t> positions3(3 * token_count);
        int64_t rope_delta = 0;
        if (!cere::mrope::positions(token_count, spans, positions3.data(), &rope_delta)) return CERE_ERR_ARGUMENT;
        const cere::host::StepImages step{owned.data(), image_count, positions3.data(), token_count};

        struct Adapter {
            int (*sink)(void *, const char *, uint32_t, const void *, uint64_t, uint64_t, cudaStream_t);
            void *ctx;
            static cere_status vision(void *c, const char *name, uint32_t block, const cere::model::bf16 *values,
                                      uint64_t rows, uint64_t cols, cudaStream_t s) {
                const auto &a = *static_cast<Adapter *>(c);
                return cere_status(a.sink(a.ctx, name, block, values, rows, cols, s));
            }
            static cere_status detail(void *c, const char *name, const cere::model::bf16 *values, uint64_t rows,
                                      uint64_t cols, cudaStream_t s) {
                if (std::strcmp(name, "native.embed.spliced") != 0) return CERE_OK;
                const auto &a = *static_cast<Adapter *>(c);
                return cere_status(a.sink(a.ctx, name, UINT32_MAX, values, rows, cols, s));
            }
        } adapter{sink, ctx};
        cere::model::vision::Taps vtaps;
        vtaps.ctx = &adapter;
        vtaps.sink = &Adapter::vision;
        cere::model::ForwardTaps taps;
        taps.last_rows = 0;  // every row: the spliced embeddings are reported whole
        taps.ctx = &adapter;
        taps.detail = 1;
        taps.detail_sink = &Adapter::detail;
        taps.vision = &vtaps;

        if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
        cere::ktest::SessionScope session(*k);
        if (const cere_status st = session.create(); st != CERE_OK) return st;
        cere::device::CudaDevice &device = cere::ktest::cuda_device(*k);
        auto &csession = static_cast<cere::device::CudaSession &>(*session.get()->device);
        std::vector<cere_token_score> scores(token_count);
        const cere_status st = device.forward(cmodel, csession, CERE_OP_PREFILL, tokens, uint32_t(token_count), &taps,
                                              scores.data(), nullptr, nullptr, &step);
        if (st != CERE_OK) return st;
        std::memcpy(out_positions3, positions3.data(), positions3.size() * sizeof(int64_t));
        *out_rope_delta = rope_delta;
        *out_top1 = scores.back();
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
}

// ---- M2.11 session entries (m211 design note Interfaces "ktest session entries", D10; ledger R, LR4) -----------
//
// A session or snapshot handle of one model handle: an ABI session (cere_session_create on the handle's executor) or
// an ABI snapshot in the model's arena. Teacher-forced feeds run through CudaDevice::forward as `step` would (the
// prompt as PREFILL in chunks of the workspace rows, then one DECODE row per token) and keep the host session state
// (tokens, fed, submitted) as committed work would, except that they leave every token fed: committed work leaves
// the token it sampled committed and unfed, and cere_session_restore refuses an image without that token
// (token_count != fed + 1, HardenHost H1). cere_ktest_session_commit_next appends it, after which the session is an
// ordinary idle one for cere_session_snapshot / cere_session_restore; the next teacher-forced feed starts with it.

namespace cere::ktest {
namespace {

struct K15Handle {
    K8Model *k = nullptr;
    cere_session *session = nullptr;    // a session handle
    cere_snapshot *snapshot = nullptr;  // a snapshot handle
};

K15Handle *k15(uint64_t h) { return reinterpret_cast<K15Handle *>(static_cast<uintptr_t>(h)); }

// One work item of the session through submit/poll (PREFILL of tokens[0, n) on an empty session, else DECODE of the
// committed unfed token), greedy, cancelled at the token boundary after `cancel_after` commits (0 = never). With
// images (PREFILL only) the item carries them (M2.13 W 152-160). M2.10: the PREFILL carries `capture_position` (W 132;
// 0 = none) and *cached_tokens (nullable) is the terminal completion's cached_tokens (D10). `flags` are the item's
// cere_work.flags; `spec` (nullable, [3]) receives the completion's verify_steps and accepted_drafts and the session's
// verified_drafts.
cere_status session_ticket(K15Handle &h, const uint32_t *tokens, uint64_t n, uint64_t max_new_tokens,
                           uint64_t cancel_after, cere_token_score *scores, uint64_t *out_counts,
                           const cere_image_input *images = nullptr, uint64_t image_count = 0,
                           uint64_t capture_position = 0, uint64_t *cached_tokens = nullptr, uint32_t flags = 0,
                           uint64_t *spec = nullptr) {
    K8Model &k = *h.k;
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    if (const cere_status st = cere_executor_capabilities(k.executor, k.model, &caps); st != CERE_OK) return st;
    std::vector<uint32_t> out_tokens;
    try {
        out_tokens.resize(max_new_tokens);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    const bool prefill = h.session->tokens.empty();
    cere_work work{};
    work.struct_size = sizeof(work);
    work.abi_version = CERE_EXECUTOR_ABI_VERSION;
    work.operation = prefill ? CERE_OP_PREFILL : CERE_OP_DECODE;
    work.expected_worker_epoch = caps.worker_epoch;
    work.expected_session_revision = h.session->revision;
    work.session = h.session;
    work.tokens = prefill ? tokens : nullptr;
    work.token_count = prefill ? n : 0;
    work.max_new_tokens = max_new_tokens;
    work.out_tokens = out_tokens.data();
    work.out_capacity = max_new_tokens;
    work.out_scores = scores;
    work.images = images;
    work.image_count = image_count;
    work.flags = flags;
    work.capture_position = capture_position;
    cere_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
    batch.expected_worker_epoch = caps.worker_epoch;
    batch.capability_profile_generation = caps.profile_generation;
    batch.items = &work;
    batch.item_count = 1;
    cere_completion_ticket ticket{};
    if (const cere_status st = cere_executor_submit(k.executor, &batch, &ticket); st != CERE_OK) return st;
    cere_completion done{};
    bool cancelled = false;
    for (;;) {
        done = cere_completion{};
        done.struct_size = sizeof(done);
        done.abi_version = CERE_EXECUTOR_ABI_VERSION;
        const cere_status st = cere_executor_poll(k.executor, ticket, &done);
        if (st == CERE_OK) break;
        if (st != CERE_PENDING) return st;
        if (cancel_after != 0 && !cancelled && done.committed_tokens >= cancel_after) {
            if (const cere_status c = cere_executor_cancel(k.executor, ticket); c != CERE_OK) return c;
            cancelled = true;
        }
    }
    out_counts[0] = done.committed_tokens;
    out_counts[1] = done.finish;
    if (cached_tokens != nullptr) *cached_tokens = done.cached_tokens;
    if (spec != nullptr) {
        spec[0] = done.verify_steps;
        spec[1] = done.accepted_drafts;
        spec[2] = h.session->verified_drafts;
    }
    if (done.finish == CERE_FINISH_LENGTH || done.finish == CERE_FINISH_STOP_TOKEN ||
        done.finish == CERE_FINISH_CANCELLED) {
        return CERE_OK;
    }
    return done.status != CERE_OK ? done.status : CERE_ERR_INTERNAL;
}

// Teacher-forced: tokens[0, n) fed through CudaDevice::forward (PREFILL chunks on an empty session, else one DECODE
// per token), each forward of a drafting session followed by its MTP extend as CudaDevice::step runs it
// (CudaDevice::extend_draft: next ids the fed tokens shifted by one, then the next token or the scored one; a PREFILL
// chunk before the last leaves the prompt open, no decode steps); `sink`
// (nullable) taps the last fed row of the last forward call. A session holding its committed, unfed token (a
// ticket's or cere_ktest_session_commit_next's) feeds it first: tokens[0] must be that token, else
// CERE_ERR_ARGUMENT.
cere_status session_forced(K15Handle &h, const uint32_t *tokens, uint64_t n, TraceSink *sink,
                           cere_token_score *scores) {
    cere_session &s = *h.session;
    const bool pending = s.fed + 1 == s.tokens.size();
    if (s.in_flight != 0 || (s.fed != s.tokens.size() && !pending)) return CERE_ERR_ARGUMENT;
    if (pending && (n == 0 || tokens[0] != s.tokens.back())) return CERE_ERR_ARGUMENT;
    if (n > s.max_context_tokens - s.fed) return CERE_ERR_CAPACITY;
    const uint32_t vocab = h.k->model->plan.config.vocab_size;
    for (uint64_t i = 0; i < n; ++i) {
        if (tokens[i] >= vocab) return CERE_ERR_ARGUMENT;
    }
    device::CudaDevice &device = cuda_device(*h.k);
    device::CudaModel &cmodel = cuda_model(*h.k);
    auto &csession = static_cast<device::CudaSession &>(*s.device);
    const bool prefill = s.tokens.empty();
    const uint64_t chunk = prefill ? cmodel.workspace_rows() : 1;
    try {
        s.tokens.reserve(s.fed + n);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    if (pending) s.tokens.pop_back();  // tokens[0] is it: the feed appends it as fed
    // The session has consumed every token of a call that succeeded: a later failure keeps the earlier ones.
    s.submitted = true;
    for (uint64_t done = 0; done < n;) {
        const uint64_t m = std::min(chunk, n - done);
        const bool last = done + m == n;
        model::ForwardTaps taps;
        if (last && sink != nullptr) {
            taps.last_rows = 1;
            taps.ctx = sink;
            taps.emit = &TraceSink::emit;
        }
        const cere_status st = device.forward(cmodel, csession, prefill ? CERE_OP_PREFILL : CERE_OP_DECODE,
                                              tokens + done, uint32_t(m), last && sink != nullptr ? &taps : nullptr,
                                              scores + (prefill ? 0 : done));
        if (st != CERE_OK) return sink != nullptr && sink->error != CERE_OK ? sink->error : st;
        s.tokens.insert(s.tokens.end(), tokens + done, tokens + done + m);
        s.fed = s.tokens.size();
        const uint32_t next = last ? scores[prefill ? 0 : done + m - 1].id : tokens[done + m];
        if (const cere_status ex =
                device.extend_draft(cmodel, csession, tokens + done, uint32_t(m), next, prefill && !last);
            ex != CERE_OK) {
            return ex;
        }
        done += m;
    }
    return CERE_OK;
}

}  // namespace
}  // namespace cere::ktest

using cere::ktest::K15Handle;

// A session of max_context_tokens (0 = the model's) on the model handle; out_session (host [1]) is its handle.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_open(uint64_t model,
                                                                              uint64_t max_context_tokens,
                                                                              uint64_t *out_session,
                                                                              cudaStream_t stream) {
    K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out_session == nullptr) return CERE_ERR_ARGUMENT;
    *out_session = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    auto *h = new (std::nothrow) K15Handle;
    if (h == nullptr) return CERE_ERR_CAPACITY;
    h->k = k;
    const uint64_t ctx = max_context_tokens == 0 ? k->model->plan.max_context_tokens : max_context_tokens;
    if (const cere_status st = cere_session_create(k->executor, k->model, ctx, &h->session); st != CERE_OK) {
        delete h;
        return st;
    }
    *out_session = uint64_t(reinterpret_cast<uintptr_t>(h));
    return CERE_OK;
}

// max_new_tokens == 0: teacher-forced feed of tokens[0, n) (host): on an empty session one PREFILL in chunks of the
// workspace rows (scores[0] predicts the next token), else n DECODE forwards of one token (scores[i] predicts the
// token after tokens[i]); with names non-null, forward_trace's sink (the 2,560-column B8 entries) taps the last fed
// row, out_counts (host [3]) = {entries, name bytes, value bytes} ({0, 0, 0} without a sink). Any capacity too small:
// CERE_ERR_CAPACITY. max_new_tokens > 0: one greedy work item through submit/poll (PREFILL of tokens[0, n) on an empty
// session, else DECODE with tokens null and n 0), cancelled after `cancel_after` committed tokens (0 = never);
// scores[0, committed) are its scores, out_counts = {committed tokens, finish, 0}; names must be null.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_feed(
    uint64_t session, const uint32_t *tokens, uint64_t n, uint64_t max_new_tokens, uint64_t cancel_after,
    uint8_t *names, uint64_t names_capacity, int64_t *shapes, uint64_t entry_capacity, void *values,
    uint64_t values_capacity_bytes, cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || scores == nullptr || out_counts == nullptr) return CERE_ERR_ARGUMENT;
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    // `values` may still be written by the caller's stream (e.g. its NaN fill).
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (max_new_tokens != 0) {
        const bool prefill = h->session->tokens.empty();
        if (names != nullptr || (prefill ? tokens == nullptr || n == 0 : tokens != nullptr || n != 0)) {
            return CERE_ERR_ARGUMENT;
        }
        return cere::ktest::session_ticket(*h, tokens, n, max_new_tokens, cancel_after, scores, out_counts);
    }
    if (tokens == nullptr || n == 0 || cancel_after != 0) return CERE_ERR_ARGUMENT;
    if (names == nullptr) return cere::ktest::session_forced(*h, tokens, n, nullptr, scores);
    if (shapes == nullptr || values == nullptr) return CERE_ERR_ARGUMENT;
    cere::ktest::TraceSink sink;
    sink.names = names;
    sink.names_capacity = names_capacity;
    sink.shapes = shapes;
    sink.entry_capacity = entry_capacity;
    sink.values = static_cast<uint8_t *>(values);
    sink.values_capacity = values_capacity_bytes;
    sink.rows = 1;
    if (const cere_status st = cere::ktest::session_forced(*h, tokens, n, &sink, scores); st != CERE_OK) return st;
    out_counts[0] = sink.entries;
    out_counts[1] = sink.name_bytes;
    out_counts[2] = sink.value_bytes;
    return CERE_OK;
}

// Appends `token` (< vocab) as the idle session's committed, unfed token, the shape a PREFILL or DECODE ticket leaves
// and cere_session_restore requires of an image (token_count == fed + 1, HardenHost H1), to a session that has fed
// every one of its tokens (at least one): the token its last teacher-forced feed scored, so a drafting session's MTP
// extend (which ran with that next id) stays the one step would have left. The next teacher-forced feed starts with
// it. A session without a fed token, one that already holds an unfed token, or a token >= vocab: CERE_ERR_ARGUMENT;
// a session at its max_context_tokens: CERE_ERR_CAPACITY.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_commit_next(uint64_t session, uint64_t token,
                                                                                     cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_session &s = *h->session;
    if (s.in_flight != 0 || s.fed == 0 || s.fed != s.tokens.size() ||
        token >= h->k->model->plan.config.vocab_size) {
        return CERE_ERR_ARGUMENT;
    }
    if (s.tokens.size() >= s.max_context_tokens) return CERE_ERR_CAPACITY;
    try {
        s.tokens.push_back(uint32_t(token));
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}

// M2.13 C14 (CR14): one greedy PREFILL work item of tokens[0, n) (host) with images[0, image_count) (host; their
// pixel_patches host memory) through submit/poll on an empty session, marked CERE_WORK_NO_SPECULATION: scores[0,
// committed) are its scores, out_counts (host [3]) = {committed tokens, finish, 0}. The submit's checks (plan_images,
// W 152-160) decide the status of a malformed item.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_prefill_images(
    uint64_t session, const uint32_t *tokens, uint64_t n, const cere_image_input *images, uint64_t image_count,
    uint64_t max_new_tokens, cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || n == 0 || images == nullptr ||
        image_count == 0 || max_new_tokens == 0 || scores == nullptr || out_counts == nullptr ||
        !h->session->tokens.empty()) {
        return CERE_ERR_ARGUMENT;
    }
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere::ktest::session_ticket(*h, tokens, n, max_new_tokens, 0, scores, out_counts, images, image_count, 0,
                                       nullptr, CERE_WORK_NO_SPECULATION);
}

// m213l design note (VL-L3/L5/L7 GPU rows): cere_ktest_session_prefill_images with `flags` (u32) as the work's
// cere_work.flags (0: a drafting PREFILL on a model with a draft, CERE_WORK_NO_SPECULATION: none), the model's rows
// per prefill call `prefill_chunk_tokens` for this ticket (0 = the executor's own, as cere_ktest_generate) and the
// prefix cache's answer: out_counts (host [6]) = {committed tokens, finish, cached_tokens, verify_steps,
// accepted_drafts, verified_drafts}.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_prefill_images_flags(
    uint64_t session, const uint32_t *tokens, uint64_t n, const cere_image_input *images, uint64_t image_count,
    uint64_t max_new_tokens, uint64_t flags, uint64_t prefill_chunk_tokens, cere_token_score *scores,
    uint64_t *out_counts, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || n == 0 || images == nullptr ||
        image_count == 0 || max_new_tokens == 0 || scores == nullptr || out_counts == nullptr ||
        !h->session->tokens.empty() || flags > UINT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    for (int i = 0; i < 6; ++i) out_counts[i] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere::ktest::ChunkScope chunk(cere::ktest::cuda_model(*h->k), prefill_chunk_tokens);
    return cere::ktest::session_ticket(*h, tokens, n, max_new_tokens, 0, scores, out_counts, images, image_count, 0,
                                       &out_counts[2], uint32_t(flags), &out_counts[3]);
}

// m213l design note (VL-L5 GPU rows): one greedy DECODE work item per session of sessions[0, rows) (host; one
// model's, each idle with its committed, unfed token, as a PREFILL ticket leaves it) with `flags` (u32) as its
// cere_work.flags, each submitted as a batch of its own in row order, then the unfinished tickets polled in row order
// until every one completes, so the executor's decode rounds (step_rows / speculate_rows) carry the rows together.
// Row r's committed scores land at scores[r * max_new_tokens]; out_counts (host [rows][4]) = {committed tokens,
// finish, verify_steps, accepted_drafts}. On a poll error the unfinished tickets are cancelled and drained, and that
// error is returned.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_decode_rows(
    const uint64_t *sessions, uint64_t rows, uint64_t max_new_tokens, uint64_t flags, cere_token_score *scores,
    uint64_t *out_counts, cudaStream_t stream) {
    constexpr uint32_t kMaxRows = cere::kernels::GemmPlan::kMaxRows;
    if (sessions == nullptr || rows == 0 || rows > kMaxRows || max_new_tokens == 0 || flags > UINT32_MAX ||
        scores == nullptr || out_counts == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    K15Handle *handles[kMaxRows];
    for (uint64_t r = 0; r < rows; ++r) {
        handles[r] = cere::ktest::k15(sessions[r]);
        if (handles[r] == nullptr || handles[r]->session == nullptr || handles[r]->k != handles[0]->k ||
            handles[r]->session->tokens.empty()) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint64_t i = 0; i < 4; ++i) out_counts[4 * r + i] = 0;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_executor *executor = handles[0]->k->executor;
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    if (const cere_status st = cere_executor_capabilities(executor, handles[0]->k->model, &caps); st != CERE_OK) {
        return st;
    }
    std::vector<uint32_t> out_tokens;
    try {
        out_tokens.resize(rows * max_new_tokens);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    cere_work work[kMaxRows];
    cere_completion_ticket tickets[kMaxRows];
    bool done[kMaxRows] = {};
    uint64_t submitted = 0;
    const auto drain = [&](cere_status error) {
        for (uint64_t r = 0; r < submitted; ++r) {
            if (done[r]) continue;
            cere_executor_cancel(executor, tickets[r]);
            for (;;) {
                cere_completion c{};
                c.struct_size = sizeof(c);
                c.abi_version = CERE_EXECUTOR_ABI_VERSION;
                if (cere_executor_poll(executor, tickets[r], &c) != CERE_PENDING) break;
            }
        }
        return error;
    };
    for (uint64_t r = 0; r < rows; ++r) {
        work[r] = cere_work{};
        work[r].struct_size = sizeof(cere_work);
        work[r].abi_version = CERE_EXECUTOR_ABI_VERSION;
        work[r].operation = CERE_OP_DECODE;
        work[r].expected_worker_epoch = caps.worker_epoch;
        work[r].expected_session_revision = handles[r]->session->revision;
        work[r].session = handles[r]->session;
        work[r].max_new_tokens = max_new_tokens;
        work[r].out_tokens = out_tokens.data() + r * max_new_tokens;
        work[r].out_capacity = max_new_tokens;
        work[r].out_scores = scores + r * max_new_tokens;
        work[r].flags = uint32_t(flags);
        cere_batch batch{};
        batch.struct_size = sizeof(batch);
        batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
        batch.expected_worker_epoch = caps.worker_epoch;
        batch.capability_profile_generation = caps.profile_generation;
        batch.items = &work[r];
        batch.item_count = 1;
        if (const cere_status st = cere_executor_submit(executor, &batch, &tickets[r]); st != CERE_OK) {
            return drain(st);
        }
        ++submitted;
    }
    for (uint64_t remaining = rows; remaining != 0;) {
        for (uint64_t r = 0; r < rows; ++r) {
            if (done[r]) continue;
            cere_completion c{};
            c.struct_size = sizeof(c);
            c.abi_version = CERE_EXECUTOR_ABI_VERSION;
            const cere_status st = cere_executor_poll(executor, tickets[r], &c);
            if (st == CERE_PENDING) continue;
            if (st != CERE_OK) return drain(st);
            done[r] = true;
            --remaining;
            out_counts[4 * r] = c.committed_tokens;
            out_counts[4 * r + 1] = c.finish;
            out_counts[4 * r + 2] = c.verify_steps;
            out_counts[4 * r + 3] = c.accepted_drafts;
            if (c.finish != CERE_FINISH_LENGTH && c.finish != CERE_FINISH_STOP_TOKEN) {
                return drain(c.status != CERE_OK ? c.status : CERE_ERR_INTERNAL);
            }
        }
    }
    return CERE_OK;
}

// cere_session_snapshot of the idle session; out_bytes (host [1]) = the image's length. With bytes non-null the image
// is read into it (capacity < length: CERE_ERR_CAPACITY, the snapshot released). With out_snapshot (host [1])
// non-null the snapshot stays in the arena as a handle (release with cere_ktest_session_close); else it is released.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_snapshot(uint64_t session, void *bytes,
                                                                                  uint64_t capacity,
                                                                                  uint64_t *out_bytes,
                                                                                  uint64_t *out_snapshot,
                                                                                  cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || out_bytes == nullptr) return CERE_ERR_ARGUMENT;
    out_bytes[0] = 0;
    if (out_snapshot != nullptr) out_snapshot[0] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_executor *executor = h->k->executor;
    auto *out = new (std::nothrow) K15Handle;
    if (out == nullptr) return CERE_ERR_CAPACITY;
    out->k = h->k;
    if (const cere_status st = cere_session_snapshot(executor, h->session, h->session->revision, &out->snapshot);
        st != CERE_OK) {
        delete out;
        return st;
    }
    cere_snapshot_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = CERE_EXECUTOR_ABI_VERSION;
    cere_status st = cere_snapshot_get_info(executor, out->snapshot, &info);
    if (st == CERE_OK && bytes != nullptr) {
        st = capacity < info.bytes ? CERE_ERR_CAPACITY
                                   : cere_snapshot_read(executor, out->snapshot, 0, bytes, info.bytes);
    }
    if (st != CERE_OK || out_snapshot == nullptr) {
        const cere_status released = cere_snapshot_release(executor, out->snapshot);
        delete out;
        if (st != CERE_OK) return st;
        if (released != CERE_OK) return released;
    } else {
        out_snapshot[0] = uint64_t(reinterpret_cast<uintptr_t>(out));
    }
    out_bytes[0] = info.bytes;
    return CERE_OK;
}

// Restores the fresh session from the snapshot handle `snapshot` (cere_session_restore), or with snapshot 0 from the
// image bytes[0, len) (host): cere_snapshot_import, cere_session_restore, then the imported handle is released
// whatever the restore returned. The status is the first failing call's (the D14 checks of import and restore).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_restore(uint64_t session, uint64_t snapshot,
                                                                                 const void *bytes, uint64_t len,
                                                                                 cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_executor *executor = h->k->executor;
    if (snapshot != 0) {
        const K15Handle *s = cere::ktest::k15(snapshot);
        if (s->snapshot == nullptr || bytes != nullptr) return CERE_ERR_ARGUMENT;
        return cere_session_restore(executor, h->session, s->snapshot);
    }
    cere_snapshot *imported = nullptr;
    if (const cere_status st = cere_snapshot_import(executor, h->k->model, bytes, len, &imported); st != CERE_OK) {
        return st;
    }
    const cere_status st = cere_session_restore(executor, h->session, imported);
    const cere_status released = cere_snapshot_release(executor, imported);
    return st != CERE_OK ? st : released;
}

// Releases a session handle (cere_session_release) or a snapshot handle (cere_snapshot_release).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_close(uint64_t handle, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(handle);
    if (h == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    const cere_status st = h->session != nullptr ? cere_session_release(h->k->executor, h->session)
                                                 : cere_snapshot_release(h->k->executor, h->snapshot);
    if (st != CERE_OK) return st;
    delete h;
    return CERE_OK;
}

#if defined(CERE_FAULT_INJECTION)
// OH-08 (fault-injection builds only; fault/fault_inject.h, `test_faults_oh08.py`): the host's answers on the model's
// executor after a fault, without touching the caller's stream (a sticky error fails every CUDA call, so the other
// entries' leading synchronize would): `session` (0: none) is a K15 session of the model, released first;
// out (host [4]) = {Device::faulted(), cere_session_create's status (a session it creates is released),
// cere_model_seal's status on the sealed model, the status of releasing the model and then its executor}; the model
// handle is gone when out[3] is 0.
extern "C" __attribute__((visibility("default"))) int cere_ktest_fault_refusals(uint64_t model, uint64_t session,
                                                                                int64_t *out, cudaStream_t) {
    K8Model *k = cere::ktest::handle(model);
    K15Handle *h = cere::ktest::k15(session);
    if (k == nullptr || out == nullptr || (h != nullptr && (h->k != k || h->session == nullptr))) {
        return CERE_ERR_ARGUMENT;
    }
    if (h != nullptr) {
        if (const cere_status st = cere_session_release(k->executor, h->session); st != CERE_OK) return st;
        delete h;
    }
    out[0] = k->executor->device->faulted() ? 1 : 0;
    cere_session *probe = nullptr;
    out[1] = cere_session_create(k->executor, k->model, k->model->plan.max_context_tokens, &probe);
    if (probe != nullptr) cere_session_release(k->executor, probe);
    out[2] = cere_model_seal(k->executor, k->model);
    cere_status st = cere_model_release(k->executor, k->model);
    if (st == CERE_OK) {
        k->model = nullptr;
        st = cere_executor_destroy(k->executor);
    }
    out[3] = st;
    if (st == CERE_OK) delete k;
    return CERE_OK;
}
#endif

namespace cere::ktest {
namespace {

// The session's state image exactly as cere_ktest_state_digest hashes it (the CERESNP1 plan at its position, every
// kind >= 1, plus kind 10 with `draft`: the MTP pool entry and the pending proposals, which the device saves only on a
// model with drafts, else its save refuses the plan). The host's `drafting` flag is not consulted: a ktest session
// drafts through cere_ktest_session_draft (CudaDevice::set_session_draft), not a PREFILL item.
cere_status state_image(uint64_t session, cudaStream_t stream, bool draft, cere::host::snapshot::Plan *plan,
                        std::vector<uint8_t> *image) {
    K15Handle *h = k15(session);
    if (h == nullptr || h->session == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = sync(stream); st != CERE_OK) return st;
    cere_session &s = *h->session;
    if (s.in_flight != 0) return CERE_ERR_BUSY;
    namespace snap = cere::host::snapshot;
    const uint32_t kinds = snap::kRequiredKinds | (s.image_session ? uint32_t(1) << snap::kRopeDelta : 0) |
                           (draft ? uint32_t(1) << snap::kDraft : 0);
    if (!snap::plan(h->k->model->plan.config, s.fed, 0, s.tokens.size(), plan, kinds)) return CERE_ERR_CAPACITY;
    try {
        image->resize(plan->total_bytes);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    if (const cere_status st = cuda_device(*h->k).save_call_end_state(
            cuda_model(*h->k), static_cast<cere::device::CudaSession &>(*s.device), *plan, s.fed, image->data());
        st != CERE_OK) {
        return st;
    }
    snap::write_rope_delta(image->data(), *plan, s.rope_delta);
    return CERE_OK;
}

// cere_ktest_state_digest's body: SHA-256 over state_image's sections except kind 0.
cere_status state_digest(uint64_t session, uint8_t *out, cudaStream_t stream, bool draft) {
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    namespace snap = cere::host::snapshot;
    snap::Plan plan;
    std::vector<uint8_t> image;
    if (const cere_status st = state_image(session, stream, draft, &plan, &image); st != CERE_OK) return st;
    cere::host::Sha256Ni sha;  // SHA-NI when the CPU has it (host/sha256_ni.h): the GPU idles while this runs
    for (const snap::Section &section : plan.sections) {
        if (section.kind != snap::kTokens) sha.update(image.data() + section.offset, section.bytes);
    }
    const std::array<uint8_t, 32> digest = sha.finish();
    std::memcpy(out, digest.data(), digest.size());
    return CERE_OK;
}

}  // namespace
}  // namespace cere::ktest

// ledger M4: SHA-256 over the session's serialized state sections exactly as cere_session_snapshot writes them (every
// kind >= 1 of the CERESNP1 plan at the session's position, table order, logical rows [0, position), valid ring
// entries only; for an image session kind 11, its rope_delta, M2.13 C14; neither the header, nor kind 0, nor the
// padding between sections). M2.10 (m210 design note D5, K16 (1)/(8)): also the state after a call that ended inside
// a prompt (prefill_split's call ends, a prefix hit's attach), which a snapshot refuses. out: host [32].
extern "C" __attribute__((visibility("default"))) int cere_ktest_state_digest(uint64_t session, uint8_t *out,
                                                                              cudaStream_t stream) {
    return cere::ktest::state_digest(session, out, stream, false);
}

// m29 design note M29S2-18b, m23 design note P11R-20: cere_ktest_state_digest of a drafting session with its
// kind 10 section too (the MTP pool entry's K/V, compressed keys and ring, and the pending proposals), so a draft
// graph's MTP state is compared with the eager draft's. On a model without drafts the device's save refuses it.
extern "C" __attribute__((visibility("default"))) int cere_ktest_draft_digest(uint64_t session, uint8_t *out,
                                                                              cudaStream_t stream) {
    return cere::ktest::state_digest(session, out, stream, true);
}

// K16 (10), m210 design note D15⁗′ (the layout probe): the bytes cere_ktest_state_digest (draft 0) or
// cere_ktest_draft_digest (draft 1) hashes, unhashed, so a test compares two sessions' states per layer and row in
// memory. dst: host [capacity]; table: host u64 [3 * table_capacity], per section in plan order {kind, offset, bytes};
// out_counts: host [2] = {sections, image bytes}, written also on CERE_ERR_CAPACITY (capacity or table_capacity too
// small; nothing else written then).
extern "C" __attribute__((visibility("default"))) int cere_ktest_state_image(uint64_t session, uint64_t draft,
                                                                             uint8_t *dst, uint64_t capacity,
                                                                             uint64_t *table, uint64_t table_capacity,
                                                                             uint64_t *out_counts,
                                                                             cudaStream_t stream) {
    if (table == nullptr || out_counts == nullptr || draft > 1 || (dst == nullptr && capacity != 0)) {
        return CERE_ERR_ARGUMENT;
    }
    namespace snap = cere::host::snapshot;
    snap::Plan plan;
    std::vector<uint8_t> image;
    if (const cere_status st = cere::ktest::state_image(session, stream, draft != 0, &plan, &image); st != CERE_OK) {
        return st;
    }
    out_counts[0] = plan.sections.size();
    out_counts[1] = image.size();
    if (image.size() > capacity || plan.sections.size() > table_capacity) return CERE_ERR_CAPACITY;
    std::memcpy(dst, image.data(), image.size());
    for (size_t i = 0; i < plan.sections.size(); ++i) {
        table[3 * i] = plan.sections[i].kind;
        table[3 * i + 1] = plan.sections[i].offset;
        table[3 * i + 2] = plan.sections[i].bytes;
    }
    return CERE_OK;
}

// M2.9 (m29 design note D4, D5, M29S2-20; test_model_k8_forward.py K1', K4, K7): drafting K15 sessions. The model
// needs speculative_draft_tokens > 0 (else CERE_ERR_UNSUPPORTED from the device).
//
// cere_ktest_session_draft: CudaDevice::set_session_draft(on != 0) on the session (an empty one, else ARGUMENT).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_draft(uint64_t session, uint64_t on,
                                                                               cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (h->session->in_flight != 0) return CERE_ERR_BUSY;
    return cere::ktest::cuda_device(*h->k).set_session_draft(*h->k->model->device, *h->session->device, on != 0);
}

// cere_ktest_session_verify (K1'): one verify chain on a decode-ready session (fed every token, at least one): host
// tokens[0] (fed at the session's position p) and the drafts tokens[1, n], n <= 3, through CudaDevice::verify_rows,
// committing `commit` in [1, n + 1] rows whatever their scores; scores (host [n + 1]) = every row's top-2. The host
// session then holds tokens[0, commit) as fed, as session_forced keeps a forced feed of them, so
// cere_ktest_state_digest compares the two.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_verify(uint64_t session,
                                                                                const uint32_t *tokens, uint64_t n,
                                                                                uint64_t commit,
                                                                                cere_token_score *scores,
                                                                                cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || scores == nullptr || n > 3 || commit == 0 ||
        commit > n + 1) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_session &s = *h->session;
    if (s.in_flight != 0 || s.fed != s.tokens.size() || s.tokens.empty()) return CERE_ERR_ARGUMENT;
    if (n + 1 > s.max_context_tokens - s.tokens.size()) return CERE_ERR_CAPACITY;
    try {
        s.tokens.reserve(s.tokens.size() + commit);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    cere::device::CudaSession *csession = &static_cast<cere::device::CudaSession &>(*s.device);
    const uint32_t drafts = uint32_t(n);
    const uint32_t a = uint32_t(commit);
    s.submitted = true;
    const cere_status st = cere::ktest::cuda_device(*h->k).verify_rows(cere::ktest::cuda_model(*h->k), &csession,
                                                                         &tokens, &drafts, 1, nullptr, scores, &a);
    if (st != CERE_OK) return st;
    s.tokens.insert(s.tokens.end(), tokens, tokens + commit);
    s.fed = s.tokens.size();
    return CERE_OK;
}

namespace cere::ktest {
namespace {

// The ItemSampling of one ktest row: `params` with struct_size 0 = no sampling (greedy); top < 0 = no logprobs,
// else the item's top_logprobs. Returns false when the row has neither (sampling stays null then).
bool item_sampling(const cere_sampling_params *params, int32_t top, cere::host::ItemSampling *out) {
    const bool sampled = params != nullptr && params->struct_size != 0;
    if (!sampled && top < 0) return false;
    *out = cere::host::ItemSampling{};
    out->sampled = sampled;
    out->top_logprobs = top < 0 ? 0 : uint32_t(top);
    if (sampled) {
        out->params = *params;
        const auto *bias = static_cast<const cere_logit_bias *>(params->logit_bias);
        if (bias != nullptr) out->bias.assign(bias, bias + params->logit_bias_count);
        out->params.logit_bias = nullptr;
    }
    return true;
}

// cere_ktest_session_speculate(_sampled|_mtp|_chain|_masked)'s body; sampling / tops / logprobs null = every row
// greedy; drafts null = every row drafts from its pending MTP proposals (draft_counts[r] of them at most). Row r's
// caller drafts start at drafts[draft_stride r] and its committed scores land at scores[score_stride r] (score_stride
// <= the SpecRowResult's entries: only that many are copied). masks (GrammarSpec GS-02; null = unconstrained): row
// r's 1 + CERE_MAX_CONSTRAINED_DRAFTS masks of ceil(vocab / 32) words at masks[r (1 + 3) words], its sampler's
// vocab_mask (a sampled row only).
int session_speculate(const uint64_t *sessions, uint64_t rows, const uint32_t *tokens, const uint32_t *drafts,
                      const uint32_t *draft_counts, const uint32_t *max_commit, const cere_sampling_params *sampling,
                      const int32_t *tops, int32_t *row_status, uint32_t *results, cere_token_score *scores,
                      cere_token_logprobs *logprobs, cudaStream_t stream, uint32_t draft_stride = 3,
                      uint32_t score_stride = 4, const uint32_t *masks = nullptr) {
    constexpr uint32_t kMaxRows = cere::kernels::GemmPlan::kMaxRows;
    if (sessions == nullptr || rows == 0 || rows > kMaxRows || tokens == nullptr ||
        draft_counts == nullptr || max_commit == nullptr || row_status == nullptr || results == nullptr ||
        scores == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = sync(stream); st != CERE_OK) return st;
    K15Handle *handles[kMaxRows];
    cere::host::SpecRowArgs args[kMaxRows];
    cere::host::ItemSampling items[kMaxRows];
    for (uint64_t r = 0; r < rows; ++r) {
        K15Handle *h = k15(sessions[r]);
        if (h == nullptr || h->session == nullptr || h->k != k15(sessions[0])->k) return CERE_ERR_ARGUMENT;
        cere_session &s = *h->session;
        if (s.in_flight != 0 || s.fed != s.tokens.size()) return CERE_ERR_ARGUMENT;
        try {
            s.tokens.reserve(s.tokens.size() + score_stride);
            if (sampling != nullptr && tops != nullptr && item_sampling(sampling + r, tops[r], &items[r])) {
                args[r].sampling = &items[r];
                if (tops[r] >= 0) {
                    if (logprobs == nullptr) return CERE_ERR_ARGUMENT;
                    args[r].logprobs = logprobs + 4 * r;
                }
                if (masks != nullptr) {
                    if (!items[r].sampled) return CERE_ERR_ARGUMENT;
                    const uint64_t words = (uint64_t(h->k->model->plan.config.vocab_size) + 31) / 32;
                    items[r].vocab_mask = masks + r * (1 + CERE_MAX_CONSTRAINED_DRAFTS) * words;
                }
            } else if (masks != nullptr) {
                return CERE_ERR_ARGUMENT;
            }
        } catch (const std::bad_alloc &) {
            return CERE_ERR_CAPACITY;
        }
        handles[r] = h;
        args[r].session = s.device.get();
        args[r].token = tokens[r];
        args[r].drafts = drafts != nullptr && draft_counts[r] != 0 ? drafts + uint64_t(draft_stride) * r : nullptr;
        args[r].draft_count = draft_counts[r];
        args[r].max_commit = max_commit[r];
        handles[r]->session->submitted = true;
    }
    cere::host::SpecRowResult out[kMaxRows];
    cere_status status[kMaxRows];
    K8Model &k = *handles[0]->k;
    const cere_status st = cuda_device(k).speculate_rows(*k.model->device, args, uint32_t(rows), out, status);
    if (st == CERE_ERR_ARGUMENT) return st;  // malformed: row_status untouched
    for (uint64_t r = 0; r < rows; ++r) {
        row_status[r] = status[r];
        if (status[r] != CERE_OK) continue;
        cere_session &s = *handles[r]->session;
        const uint32_t c = out[r].committed;
        results[2 * r] = c;
        results[2 * r + 1] = out[r].accepted;
        std::memcpy(scores + uint64_t(score_stride) * r, out[r].scores, score_stride * sizeof(cere_token_score));
        // The committed ids: tokens[r], then the ids of rows 0 .. c - 2 (the accepted drafts).
        s.tokens.push_back(tokens[r]);
        for (uint32_t i = 0; i + 1 < c; ++i) s.tokens.push_back(out[r].scores[i].id);
        s.fed = s.tokens.size();
    }
    return st;
}

}  // namespace
}  // namespace cere::ktest

// cere_ktest_session_speculate (K4, K7): one CudaDevice::speculate_rows call of `rows` decode-ready drafting sessions
// of one model (host arrays): row r feeds tokens[r] with the caller drafts drafts[3 r, 3 r + draft_counts[r])
// (draft_counts[r] = 0: none) under max_commit[r], greedy, no stop token. row_status (host int32 [rows]) and
// results (host uint32 [rows, 2] = committed, accepted; cere_token_score [rows, 4] of the committed tokens) are the
// device's. Each OK row's host session then holds its committed tokens (tokens[r] and the accepted drafts) as fed.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_speculate(
    const uint64_t *sessions, uint64_t rows, const uint32_t *tokens, const uint32_t *drafts,
    const uint32_t *draft_counts, const uint32_t *max_commit, int32_t *row_status, uint32_t *results,
    cere_token_score *scores, cudaStream_t stream) {
    return cere::ktest::session_speculate(sessions, rows, tokens, drafts, draft_counts, max_commit, nullptr, nullptr,
                                          row_status, results, scores, nullptr, stream);
}

// cere_ktest_session_speculate_mtp (K4 with MTP drafts, I1): cere_ktest_session_speculate with every row drafting
// from its session's pending MTP proposals (draft_counts[r] <= 3 of them at most, CudaDevice::speculate_rows with
// drafts null), greedy, no stop token.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_speculate_mtp(
    const uint64_t *sessions, uint64_t rows, const uint32_t *tokens, const uint32_t *draft_counts,
    const uint32_t *max_commit, int32_t *row_status, uint32_t *results, cere_token_score *scores,
    cudaStream_t stream) {
    return cere::ktest::session_speculate(sessions, rows, tokens, nullptr, draft_counts, max_commit, nullptr, nullptr,
                                          row_status, results, scores, nullptr, stream);
}

// M2.9 phase S (m29 design note D6, M29S2-13/-14): cere_ktest_session_speculate with row r's sampling
// sampling[r] (host cere_sampling_params [rows]; struct_size 0 = greedy) and tops[r] (host int32 [rows]: -1 = no
// logprobs, else the item's top_logprobs; logprobs host cere_token_logprobs [rows, 4] then receives row r's
// committed tokens' entries at [r, :committed]).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_speculate_sampled(
    const uint64_t *sessions, uint64_t rows, const uint32_t *tokens, const uint32_t *drafts,
    const uint32_t *draft_counts, const uint32_t *max_commit, const cere_sampling_params *sampling,
    const int32_t *tops, int32_t *row_status, uint32_t *results, cere_token_score *scores,
    cere_token_logprobs *logprobs, cudaStream_t stream) {
    if (sampling == nullptr || tops == nullptr) return CERE_ERR_ARGUMENT;
    return cere::ktest::session_speculate(sessions, rows, tokens, drafts, draft_counts, max_commit, sampling, tops,
                                          row_status, results, scores, logprobs, stream);
}

// GrammarSpec GS-02: cere_ktest_session_speculate_sampled with row r's verify masks (host uint32 [rows, 1 +
// CERE_MAX_CONSTRAINED_DRAFTS, ceil(vocab / 32)]: mask j for verify row j, the rows past draft_counts[r] unread);
// every row sampled (struct_size != 0), its drafts the caller's.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_speculate_masked(
    const uint64_t *sessions, uint64_t rows, const uint32_t *tokens, const uint32_t *drafts,
    const uint32_t *draft_counts, const uint32_t *max_commit, const cere_sampling_params *sampling,
    const int32_t *tops, const uint32_t *masks, int32_t *row_status, uint32_t *results, cere_token_score *scores,
    cere_token_logprobs *logprobs, cudaStream_t stream) {
    if (sampling == nullptr || tops == nullptr || masks == nullptr) return CERE_ERR_ARGUMENT;
    return cere::ktest::session_speculate(sessions, rows, tokens, drafts, draft_counts, max_commit, sampling, tops,
                                          row_status, results, scores, logprobs, stream, 3, 4, masks);
}

// cere_ktest_session_speculate_chain (DraftAccept deep chains): cere_ktest_session_speculate with up to
// model::layout::kMaxChainDrafts caller drafts per row: drafts host uint32 [rows, kMaxChainDrafts], scores host
// cere_token_score [rows, kMaxChainDrafts + 1] of the committed tokens. Greedy, no stop token.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_speculate_chain(
    const uint64_t *sessions, uint64_t rows, const uint32_t *tokens, const uint32_t *drafts,
    const uint32_t *draft_counts, const uint32_t *max_commit, int32_t *row_status, uint32_t *results,
    cere_token_score *scores, cudaStream_t stream) {
    constexpr uint32_t kChain = uint32_t(cere::model::layout::kMaxChainDrafts);
    static_assert(sizeof(cere::host::SpecRowResult::scores) == (kChain + 1) * sizeof(cere_token_score),
                  "SpecRowResult holds a full chain");
    if (drafts == nullptr) return CERE_ERR_ARGUMENT;
    return cere::ktest::session_speculate(sessions, rows, tokens, drafts, draft_counts, max_commit, nullptr, nullptr,
                                          row_status, results, scores, nullptr, stream, kChain, kChain + 1);
}

namespace cere::ktest {
namespace {

// cere_ktest_session_step_sampled / _masked's body (`mask` null = unconstrained).
int session_step(uint64_t session, uint32_t token, const cere_sampling_params *sampling, int32_t top,
                 const uint32_t *mask, cere_token_score *score, cere_token_logprobs *logprobs, float *processed,
                 cudaStream_t stream) {
    K15Handle *h = k15(session);
    if (h == nullptr || h->session == nullptr || score == nullptr || (top >= 0 && logprobs == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = sync(stream); st != CERE_OK) return st;
    cere_session &s = *h->session;
    if (s.in_flight != 0 || s.fed != s.tokens.size() || s.tokens.empty()) return CERE_ERR_ARGUMENT;
    cere::host::ItemSampling item;
    cere::host::RowArgs row{s.device.get(), token};
    try {
        s.tokens.reserve(s.tokens.size() + 1);
        if (item_sampling(sampling, top, &item)) row.sampling = &item;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    if (processed != nullptr && row.sampling == nullptr) return CERE_ERR_ARGUMENT;
    if (mask != nullptr) {
        if (!item.sampled) return CERE_ERR_ARGUMENT;
        item.vocab_mask = mask;
    }
    if (top >= 0) row.logprobs = logprobs;
    s.submitted = true;
    K8Model &k = *h->k;
    cere_status row_status = CERE_OK;
    const cere_status st = cuda_device(k).step_rows(*k.model->device, &row, 1, score, &row_status);
    if (st != CERE_OK) return st;
    if (row_status != CERE_OK) return row_status;
    s.tokens.push_back(token);
    s.fed = s.tokens.size();
    if (processed == nullptr) return CERE_OK;
    const cere::model::Workspace &ws = cuda_model(k).workspace();
    const uint64_t vocab = k.model->plan.config.vocab_size;
    return cudaMemcpy(processed, ws.proc_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess
               ? CERE_OK
               : CERE_ERR_INTERNAL;
}

}  // namespace
}  // namespace cere::ktest

// M2.9 phase S reference: plain sampled decode. One CudaDevice::step_rows row on a decode-ready session (fed every
// token, at least one) feeding `token` with `sampling` (struct_size 0 = greedy) and top (-1 = no logprobs): *score,
// *logprobs (when top >= 0) and, when `processed` (host fp32 [vocab]) is non-null, the sampler stage's processed row
// (Workspace::proc_logits; a sampled or logprobs row only). The host session then holds `token` as fed.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_step_sampled(
    uint64_t session, uint32_t token, const cere_sampling_params *sampling, int32_t top, cere_token_score *score,
    cere_token_logprobs *logprobs, float *processed, cudaStream_t stream) {
    return cere::ktest::session_step(session, token, sampling, top, nullptr, score, logprobs, processed, stream);
}

// GrammarSpec GS-02 reference: plain constrained decode. cere_ktest_session_step_sampled with the sampler's vocab
// mask (host uint32 [ceil(vocab / 32)], m28 design note D9; `sampling` sampled, struct_size != 0).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_step_masked(
    uint64_t session, uint32_t token, const cere_sampling_params *sampling, int32_t top, const uint32_t *mask,
    cere_token_score *score, cere_token_logprobs *logprobs, float *processed, cudaStream_t stream) {
    if (mask == nullptr) return CERE_ERR_ARGUMENT;
    return cere::ktest::session_step(session, token, sampling, top, mask, score, logprobs, processed, stream);
}

// M2.9 phase S: the verify rows' sampler stages of the model's last speculate_rows call (model::SpecSampler, rows
// [0, rows) in chain order), copied to host (each nullable): processed fp32 [rows, vocab], probs fp32 [rows, vocab]
// (rows of temperature > 0 chains only), keep uint32 [rows, ceil(vocab / 32)]. CERE_ERR_UNSUPPORTED for a model
// without drafts; rows above the bound rows: CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_spec_stages(uint64_t session, uint64_t rows,
                                                                            float *processed, float *probs,
                                                                            uint32_t *keep, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || rows == 0) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    const cere::model::SpecSampler &sp = cere::ktest::cuda_model(*h->k).workspace().spec;
    if (sp.rows == 0) return CERE_ERR_UNSUPPORTED;
    if (rows > sp.rows) return CERE_ERR_ARGUMENT;
    const uint64_t vocab = h->k->model->plan.config.vocab_size;
    const uint64_t words = (vocab + 31) / 32;
    const struct {
        void *dst;
        const void *src;
        uint64_t bytes;
    } copies[] = {{processed, sp.proc, rows * vocab * sizeof(float)},
                  {probs, sp.probs, rows * vocab * sizeof(float)},
                  {keep, sp.keep, rows * words * sizeof(uint32_t)}};
    for (const auto &c : copies) {
        if (c.dst == nullptr) continue;
        if (cudaMemcpy(c.dst, c.src, c.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return CERE_ERR_INTERNAL;
    }
    return CERE_OK;
}

// M2.9 phase S (M29S2-14): the session's sampler output counts (kind 8) copied to host uint32 [vocab].
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_counts(uint64_t session, uint32_t *out,
                                                                               cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    const auto &s = static_cast<const cere::device::CudaSession &>(*h->session->device);
    const uint64_t vocab = h->k->model->plan.config.vocab_size;
    return cudaMemcpy(out, s.out_counts(), vocab * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess
               ? CERE_OK
               : CERE_ERR_INTERNAL;
}

// m26 design note I7 (K3 test_decode_rows_bit_identical, K4 test_decode_rows_vs_dump): S = `sessions` sessions of
// the model (S in [1, the plan's max_sessions]) run rounds t = 0 .. T - 1, T = the largest leave[s]. In round t
// every session with join[s] == t first prefills its prompt alone, in session order (the single-row PREFILL forward
// in chunks of the workspace rows, as forced_decode_trace; scores[s * (T + 1)] predicts its first token), then every
// session with join[s] <= t < leave[s] feeds one token in one decode step, rows in session order (two or more rows:
// CudaDevice::forward_rows, i.e. model::forward_decode_rows; one row: the single-row DECODE forward; as step_rows
// runs them). Session s's k-th decode (k = 1 .. leave[s] - join[s]) feeds forced[s * T + k - 1] (`forced` non-null)
// or the top-1 id of scores[s * (T + 1) + k - 1] (greedy) and writes scores[s * (T + 1) + k]; the other score rows
// are untouched. With `names` non-null, the B8 taps of that decode (its row's slice) are step s * T + k - 1 of a
// trace laid out as forced_decode_trace's steps (every step the entries of a counting probe, one tapped single-token
// forward on a throwaway session: names, shapes {2, 1, cols}, values); out_counts = the entries, name bytes and
// value bytes of one step (zero without taps), set before the first prefill. `prompts` = the S prompts concatenated,
// prompt_lens[s] >= 1, join[s] < leave[s], prompt_lens[s] + leave[s] - join[s] <= the plan's max_context_tokens,
// every fed token < vocab_size; else, or with a trace capacity below S * T steps, CERE_ERR_ARGUMENT before any
// forward.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_decode_rows(
    uint64_t model, uint64_t sessions, const uint32_t *prompts, const uint64_t *prompt_lens, const uint64_t *join,
    const uint64_t *leave, const uint32_t *forced, cere_token_score *scores, uint8_t *names, uint64_t names_capacity,
    int64_t *shapes, uint64_t entry_capacity, void *values, uint64_t values_capacity_bytes, uint64_t *out_counts,
    cudaStream_t stream) {
    constexpr uint32_t kMaxRows = cere::kernels::GemmPlan::kMaxRows;
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || prompts == nullptr || prompt_lens == nullptr || join == nullptr || leave == nullptr ||
        scores == nullptr || out_counts == nullptr || sessions == 0 || sessions > k->model->plan.max_sessions ||
        sessions > kMaxRows || (names != nullptr && (shapes == nullptr || values == nullptr))) {
        return CERE_ERR_ARGUMENT;
    }
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    const uint64_t ctx = k->model->plan.max_context_tokens;
    const uint32_t vocab = k->model->plan.config.vocab_size;
    uint64_t rounds = 0;
    uint64_t offset[kMaxRows];
    uint64_t prompt_total = 0;
    for (uint64_t s = 0; s < sessions; ++s) {
        if (prompt_lens[s] == 0 || join[s] >= leave[s] || prompt_lens[s] > ctx ||
            leave[s] - join[s] > ctx - prompt_lens[s]) {
            return CERE_ERR_ARGUMENT;
        }
        rounds = std::max(rounds, leave[s]);
        offset[s] = prompt_total;
        prompt_total += prompt_lens[s];
    }
    for (uint64_t i = 0; i < prompt_total; ++i) {
        if (prompts[i] >= vocab) return CERE_ERR_ARGUMENT;
    }
    if (forced != nullptr) {
        for (uint64_t s = 0; s < sessions; ++s) {
            for (uint64_t j = 0; j < leave[s] - join[s]; ++j) {
                if (forced[s * rounds + j] >= vocab) return CERE_ERR_ARGUMENT;
            }
        }
    }
    // `values` may still be written by the caller's stream (e.g. its NaN fill).
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);

    cere::ktest::CountSink size;
    if (names != nullptr) {
        cere::ktest::SessionScope probe(*k);
        if (const cere_status st = probe.create(); st != CERE_OK) return st;
        auto &psession = static_cast<cere::device::CudaSession &>(*probe.get()->device);
        cere::model::ForwardTaps taps;
        taps.last_rows = 1;
        taps.ctx = &size;
        taps.emit = &cere::ktest::CountSink::emit;
        cere_token_score ignored{};
        if (const cere_status st = device.forward(cmodel, psession, CERE_OP_PREFILL, prompts, 1, &taps, &ignored);
            st != CERE_OK) {
            return st;
        }
        const uint64_t steps = sessions * rounds;
        if (size.entries > entry_capacity / steps || size.name_bytes > names_capacity / steps ||
            size.value_bytes > values_capacity_bytes / steps) {
            return CERE_ERR_ARGUMENT;
        }
        out_counts[0] = size.entries;
        out_counts[1] = size.name_bytes;
        out_counts[2] = size.value_bytes;
    }

    std::unique_ptr<cere::ktest::SessionScope> scopes[kMaxRows];
    cere::device::CudaSession *csessions[kMaxRows];
    for (uint64_t s = 0; s < sessions; ++s) {
        scopes[s].reset(new (std::nothrow) cere::ktest::SessionScope(*k));
        if (scopes[s] == nullptr) return CERE_ERR_CAPACITY;
        if (const cere_status st = scopes[s]->create(); st != CERE_OK) return st;
        csessions[s] = &static_cast<cere::device::CudaSession &>(*scopes[s]->get()->device);
    }
    const uint64_t ws_rows = cmodel.workspace_rows();
    auto *const value_base = static_cast<uint8_t *>(values);
    const uint8_t *first_names = nullptr;  // the first decode's names: every later row repeats them
    for (uint64_t t = 0; t < rounds; ++t) {
        for (uint64_t s = 0; s < sessions; ++s) {
            if (join[s] != t) continue;
            for (uint64_t p0 = 0; p0 < prompt_lens[s];) {
                const uint64_t n = std::min(ws_rows, prompt_lens[s] - p0);
                if (const cere_status st = device.forward(cmodel, *csessions[s], CERE_OP_PREFILL,
                                                          prompts + offset[s] + p0, uint32_t(n), nullptr,
                                                          scores + s * (rounds + 1));
                    st != CERE_OK) {
                    return st;
                }
                p0 += n;
            }
        }
        cere::device::CudaSession *row_sessions[kMaxRows];
        uint32_t tokens[kMaxRows];
        uint64_t at[kMaxRows];  // the score row of each decode row
        cere::ktest::RowsSink sink;
        sink.entries = size.entries;
        sink.name_bytes = size.name_bytes;
        sink.value_bytes = size.value_bytes;
        uint32_t count = 0;
        for (uint64_t s = 0; s < sessions; ++s) {
            if (t < join[s] || t >= leave[s]) continue;
            const uint64_t j = t - join[s];  // this decode is k = j + 1
            const uint64_t row0 = s * (rounds + 1);
            row_sessions[count] = csessions[s];
            tokens[count] = forced != nullptr ? forced[s * rounds + j] : scores[row0 + j].id;
            at[count] = row0 + j + 1;
            if (names != nullptr) {
                const uint64_t step = s * rounds + j;
                sink.slab[count].names = names + step * size.name_bytes;
                sink.slab[count].shapes = shapes + 3 * step * size.entries;
                sink.slab[count].values = value_base + step * size.value_bytes;
            }
            ++count;
        }
        if (count == 0) continue;
        sink.rows = count;
        cere::model::ForwardTaps taps;
        taps.last_rows = 1;
        taps.ctx = &sink;
        taps.emit = &cere::ktest::RowsSink::emit;
        cere::model::ForwardTaps *const tapped = names != nullptr ? &taps : nullptr;
        cere_token_score out[kMaxRows];
        const cere_status st = count == 1
                                   ? device.forward(cmodel, *row_sessions[0], CERE_OP_DECODE, tokens, 1, tapped, out)
                                   : device.forward_rows(cmodel, row_sessions, count, tokens, tapped, out);
        if (st != CERE_OK) return sink.error != CERE_OK ? CERE_ERR_INTERNAL : st;
        if (names != nullptr) {
            if (!sink.complete()) return CERE_ERR_INTERNAL;
            if (first_names == nullptr) first_names = sink.slab[0].names;
            for (uint32_t r = 0; r < count; ++r) {
                if (std::memcmp(sink.slab[r].names, first_names, size.name_bytes) != 0) return CERE_ERR_INTERNAL;
            }
        }
        for (uint32_t r = 0; r < count; ++r) {
            scores[at[r]] = out[r];
        }
    }
    return CERE_OK;
}

namespace cere::ktest {
namespace {

// cere_ktest_generate_rows's body, plus per-request sampling (cere_ktest_generate_rows_sampled).
cere_status generate_rows(const K8Model *k, uint64_t requests, const uint32_t *prompts, const uint64_t *prompt_lens,
                          const uint64_t *max_new_tokens, uint64_t prefill_chunk_tokens,
                          const cere_sampling_params *sampling, const uint32_t *top_logprobs, uint32_t *out_tokens,
                          cere_token_score *out_scores, cere_token_logprobs *out_logprobs, uint64_t out_stride,
                          uint64_t *out_results, uint64_t *out_device_bytes, cudaStream_t stream) {
    constexpr uint32_t kMaxRows = kernels::GemmPlan::kMaxRows;
    if (k == nullptr || prompts == nullptr || prompt_lens == nullptr || max_new_tokens == nullptr ||
        out_tokens == nullptr || out_scores == nullptr || out_results == nullptr || requests == 0 ||
        requests > kMaxRows) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint64_t i = 0; i < requests; ++i) {
        if (max_new_tokens[i] > out_stride) return CERE_ERR_ARGUMENT;
        out_results[2 * i] = out_results[2 * i + 1] = 0;
    }
    if (const cere_status st = sync(stream); st != CERE_OK) return st;
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    if (const cere_status st = cere_executor_capabilities(k->executor, k->model, &caps); st != CERE_OK) return st;
    std::unique_ptr<SessionScope> scopes[kMaxRows];
    for (uint64_t i = 0; i < requests; ++i) {
        scopes[i].reset(new (std::nothrow) SessionScope(*k));
        if (scopes[i] == nullptr) return CERE_ERR_CAPACITY;
        if (const cere_status st = scopes[i]->create(); st != CERE_OK) return st;
    }
    ChunkScope chunk(cuda_model(*k), prefill_chunk_tokens);
    if (out_device_bytes != nullptr) out_device_bytes[0] = device_bytes_in_use();

    cere_work work[kMaxRows];
    cere_completion_ticket tickets[kMaxRows];
    bool done[kMaxRows] = {};
    uint64_t submitted = 0;
    // Cancels and drains every submitted, unfinished ticket (an error path: the sessions are released after).
    const auto drain = [&](cere_status error) {
        for (uint64_t i = 0; i < submitted; ++i) {
            if (done[i]) continue;
            cere_executor_cancel(k->executor, tickets[i]);
            for (;;) {
                cere_completion c{};
                c.struct_size = sizeof(c);
                c.abi_version = CERE_EXECUTOR_ABI_VERSION;
                if (cere_executor_poll(k->executor, tickets[i], &c) != CERE_PENDING) break;
            }
        }
        return error;
    };
    uint64_t offset = 0;
    for (uint64_t i = 0; i < requests; ++i) {
        work[i] = cere_work{};
        work[i].struct_size = sizeof(cere_work);
        work[i].abi_version = CERE_EXECUTOR_ABI_VERSION;
        work[i].operation = CERE_OP_PREFILL;
        work[i].expected_worker_epoch = caps.worker_epoch;
        work[i].expected_session_revision = 0;
        work[i].session = scopes[i]->get();
        work[i].tokens = prompts + offset;
        work[i].token_count = prompt_lens[i];
        work[i].max_new_tokens = max_new_tokens[i];
        work[i].out_tokens = out_tokens + i * out_stride;
        work[i].out_capacity = max_new_tokens[i];
        work[i].out_scores = out_scores + i * out_stride;
        // A request whose sampling entry has struct_size 0 is greedy (null sampling).
        if (sampling != nullptr && sampling[i].struct_size != 0) {
            work[i].sampling = &sampling[i];
            if (out_logprobs != nullptr) {
                work[i].out_logprobs = out_logprobs + i * out_stride;
                work[i].top_logprobs = top_logprobs != nullptr ? top_logprobs[i] : 0;
            }
        }
        offset += prompt_lens[i];
        cere_batch batch{};
        batch.struct_size = sizeof(batch);
        batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
        batch.expected_worker_epoch = caps.worker_epoch;
        batch.capability_profile_generation = caps.profile_generation;
        batch.items = &work[i];
        batch.item_count = 1;
        if (const cere_status st = cere_executor_submit(k->executor, &batch, &tickets[i]); st != CERE_OK) {
            return drain(st);
        }
        ++submitted;
    }
    for (uint64_t remaining = requests; remaining != 0;) {
        for (uint64_t i = 0; i < requests; ++i) {
            if (done[i]) continue;
            cere_completion c{};
            c.struct_size = sizeof(c);
            c.abi_version = CERE_EXECUTOR_ABI_VERSION;
            const cere_status st = cere_executor_poll(k->executor, tickets[i], &c);
            if (st == CERE_PENDING) continue;
            if (st != CERE_OK) return drain(st);
            done[i] = true;
            --remaining;
            out_results[2 * i] = c.committed_tokens;
            out_results[2 * i + 1] = c.finish;
            if (c.finish != CERE_FINISH_LENGTH && c.finish != CERE_FINISH_STOP_TOKEN) {
                return drain(c.status != CERE_OK ? c.status : CERE_ERR_INTERNAL);
            }
        }
    }
    if (out_device_bytes != nullptr) out_device_bytes[1] = device_bytes_in_use();
    return CERE_OK;
}

}  // namespace
}  // namespace cere::ktest

// m26 design note K5 (test_prefill_chunk_rounds_bit_identical): `requests` greedy requests through the executor's
// rounds (M2.6 D2: a poll runs at most one prefill chunk or batched decode round, S1 G3), each on a session of its own:
// request i (prompt i of the concatenated `prompts`, prompt_lens[i] tokens, max_new_tokens[i]) is submitted in
// order i, then the unfinished tickets are polled in request order until every one completes, with
// prefill_chunk_tokens rows per prefill forward (cere_ktest_generate's semantics; 0 = the executor's own policy).
// Request i's committed tokens and scores go to out_tokens / out_scores + i * out_stride (out_stride >= every
// max_new_tokens[i]); out_results[2 * i] = its committed tokens, out_results[2 * i + 1] = its finish reason. On a
// poll error the unfinished tickets are cancelled and drained, and that error is returned.
extern "C" __attribute__((visibility("default"))) int cere_ktest_generate_rows(
    uint64_t model, uint64_t requests, const uint32_t *prompts, const uint64_t *prompt_lens,
    const uint64_t *max_new_tokens, uint64_t prefill_chunk_tokens, uint32_t *out_tokens,
    cere_token_score *out_scores, uint64_t out_stride, uint64_t *out_results, cudaStream_t stream) {
    return cere::ktest::generate_rows(cere::ktest::handle(model), requests, prompts, prompt_lens, max_new_tokens,
                                      prefill_chunk_tokens, nullptr, nullptr, out_tokens, out_scores, nullptr,
                                      out_stride, out_results, nullptr, stream);
}

// Ledger CR4 (I10 rows 2-4 sampler invariance, rows-4 sampler memory): cere_ktest_generate_rows with per-request
// sampling, so the decode rounds put sampled rows in step_rows (CudaDevice::step_rows -> model::forward_decode_rows
// running K12 per sampled row). sampling[i] (host, `requests` entries; null = every request greedy) is request i's
// cere_work.sampling, struct_size 0 = greedy; with out_logprobs non-null a sampled request i also gets out_logprobs +
// i * out_stride and top_logprobs[i] (null = 0). out_device_bytes (host [2], nullable): the device bytes in use
// (cudaMemGetInfo) after the sessions are created and before the first submit, and after the last completion.
extern "C" __attribute__((visibility("default"))) int cere_ktest_generate_rows_sampled(
    uint64_t model, uint64_t requests, const uint32_t *prompts, const uint64_t *prompt_lens,
    const uint64_t *max_new_tokens, uint64_t prefill_chunk_tokens, const cere_sampling_params *sampling,
    const uint32_t *top_logprobs, uint32_t *out_tokens, cere_token_score *out_scores,
    cere_token_logprobs *out_logprobs, uint64_t out_stride, uint64_t *out_results, uint64_t *out_device_bytes,
    cudaStream_t stream) {
    return cere::ktest::generate_rows(cere::ktest::handle(model), requests, prompts, prompt_lens, max_new_tokens,
                                      prefill_chunk_tokens, sampling, top_logprobs, out_tokens, out_scores,
                                      out_logprobs, out_stride, out_results, out_device_bytes, stream);
}

// Ledger W / CR4 rows-4 sampler memory (host arithmetic, no CUDA call): out[0] the K12 sampler scratch
// (model::sampler_scratch_bytes, one per model, inside the shared scratch), out[1] the processed-logits row
// (Workspace::proc_logits, one per model), out[2] one session's out_counts, out[3] the plan's max_sessions (R).
extern "C" __attribute__((visibility("default"))) int cere_ktest_sampler_plan_bytes(uint64_t model, uint64_t *out,
                                                                                    cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const cere::host::ModelPlan &plan = k->model->plan;
    out[0] = cere::model::sampler_scratch_bytes(plan.config);
    out[1] = uint64_t(plan.config.vocab_size) * sizeof(float);
    out[2] = uint64_t(plan.config.vocab_size) * sizeof(uint32_t);
    out[3] = plan.max_sessions;
    return CERE_OK;
}

// m23 design note P1 (M23P1-07/08): the model's decode graphs (model.h DecodeGraphs). mode 0 or 1 switches them off
// or on for the model's later steps (no step may run concurrently); -1 leaves them as they are. out (host, [4]) after
// the switch: {enabled (the decode_step lever until switched), graphs instantiated, executables updated in
// place, steps run by a graph} (the counters count from the model's INSTANTIATE). Any other mode, a null out or an
// unknown handle: CERE_ERR_ARGUMENT, nothing changed.
extern "C" __attribute__((visibility("default"))) int cere_ktest_decode_graph(uint64_t model, int32_t mode,
                                                                              uint64_t *out, cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr || mode < -1 || mode > 1) return CERE_ERR_ARGUMENT;
    cere::model::DecodeGraphs &g = cere::ktest::cuda_model(*k).decode_graphs();
    if (mode != -1) g.enabled = mode == 1;
    out[0] = g.enabled ? 1 : 0;
    out[1] = g.counts.captures;
    out[2] = g.counts.updates;
    out[3] = g.counts.replays;
    return CERE_OK;
}

// m23 design note P11R-19: the model's verify_attention lever. mode 0 (per_row) or 1 (batched) switches it for the
// model's later VERIFY calls (no call may run concurrently); -1 leaves it. out (host, [2]) after the switch: {batched
// (the lever until switched), attention_chain calls since the model's INSTANTIATE}. Any other mode, a null out or an
// unknown handle: CERE_ERR_ARGUMENT, nothing changed.
extern "C" __attribute__((visibility("default"))) int cere_ktest_verify_attention(uint64_t model, int32_t mode,
                                                                                  uint64_t *out, cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr || mode < -1 || mode > 1) return CERE_ERR_ARGUMENT;
    auto &cm = cere::ktest::cuda_model(*k);
    if (mode != -1) cm.verify_batched() = mode == 1;
    out[0] = cm.verify_batched() ? 1 : 0;
    out[1] = cm.verify_chains();
    return CERE_OK;
}

// m29 design note M29S2-18b, m23 design note P11R-20: the model's spec graphs (decode_step = spec_graph, model.h
// DecodeGraphs::verify and ::draft). mode 0 or 1 switches them off or on for the model's later VERIFY calls and MTP
// drafts (no call may run concurrently; the plain decode graphs stay as they are); -1 leaves them. out (host, [7])
// after the switch: {on (the decode_step lever until switched), verify graphs instantiated, verify executables
// updated in place, VERIFY calls run by a graph, draft graphs instantiated, draft executables updated in place, MTP
// drafts run by a graph} (counters since the model's INSTANTIATE). Any other mode, a null out or an unknown handle:
// CERE_ERR_ARGUMENT, nothing changed.
extern "C" __attribute__((visibility("default"))) int cere_ktest_spec_graph(uint64_t model, int32_t mode, uint64_t *out,
                                                                            cudaStream_t) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr || mode < -1 || mode > 1) return CERE_ERR_ARGUMENT;
    cere::model::DecodeGraphs &g = cere::ktest::cuda_model(*k).decode_graphs();
    if (mode != -1) g.spec = mode == 1;
    out[0] = g.spec ? 1 : 0;
    out[1] = g.verify_counts.captures;
    out[2] = g.verify_counts.updates;
    out[3] = g.verify_counts.replays;
    out[4] = g.draft_counts.captures;
    out[5] = g.draft_counts.updates;
    out[6] = g.draft_counts.replays;
    return CERE_OK;
}

namespace cere::ktest {
namespace {

// batch-concurrency design note BC0 (BC-D1 (a)): the round the next cere_executor_poll of `polled` runs, predicted
// from the executor's tickets as the poll and run_round select it: none when a round ran since `polled` was last
// reported (S1 G3, beat-sglang design note: the poll reports that round first), else run_round's (the LENGTH / ERROR
// ends run_round applies first count as no row; with a decode share set, its share_round policy):
// row[0] 0 = none, 1 = a prefill chunk, 2 = a decode round; row[1] / row[2] the decode-ready plain / drafting rows
// (the decode round's rows; beside a prefill chunk, the rows it holds back);
// row[3] 1 while a prompt is pending.
void predict_round(const cere_executor &e, const host::Ticket &polled, uint32_t *row) {
    uint32_t prompts = 0;
    uint32_t plain = 0;
    uint32_t drafting = 0;
    for (const auto &entry : e.tickets) {
        const host::Ticket &t = *entry.second;
        if (t.finish_pending != CERE_FINISH_NONE || t.error_deferred != CERE_OK || t.cancel_requested) continue;
        const cere_session &s = *t.session;
        if (t.committed.load(std::memory_order_relaxed) >= t.max_new_tokens ||
            s.tokens.size() >= s.max_context_tokens || !t.mask_present()) {
            continue;
        }
        if (s.fed < s.prompt_tokens) {
            prompts += 1;
        } else if (s.tokens.size() > s.fed) {
            (s.drafting ? drafting : plain) += 1;
        }
    }
    const bool prompt = prompts != 0;
    const bool decode = plain + drafting != 0;
    const bool chunk = e.decode_share_pct != 0 ? prompt && (!decode || prompts > 1 || e.decode_owed_ns <= 0)
                                               : prompt && (!decode || e.decode_rounds_owed == 0);
    row[0] = polled.seen_round != e.rounds ? 0 : chunk ? 1 : decode ? 2 : 0;
    row[1] = plain;
    row[2] = drafting;
    row[3] = prompt ? 1 : 0;
}

// Restores the model's decode and spec graph switches on scope exit.
class GraphScope {
public:
    explicit GraphScope(model::DecodeGraphs &g) : g_(g), enabled_(g.enabled), spec_(g.spec) {}
    ~GraphScope() {
        g_.enabled = enabled_;
        g_.spec = spec_;
    }
    GraphScope(const GraphScope &) = delete;
    GraphScope &operator=(const GraphScope &) = delete;

private:
    model::DecodeGraphs &g_;
    const bool enabled_;
    const bool spec_;
};

}  // namespace
}  // namespace cere::ktest

// batch-concurrency design note BC0 (BC-D1 (a)), the batch invariance gate's composition driver
// (executor/tests/gpu/test_batch_inv_gate.py): `requests` work items through the real rounds (cere_executor_submit /
// cere_executor_poll; a poll runs at most one run_round, S1 G3), request i a PREFILL of prompt i (the concatenated
// `prompts`, prompt_lens[i] tokens) on the empty K15 session sessions[i] (cere_ktest_session_open; its namespace, if
// any, set by the caller), with cere_work.flags flags[i], max_new_tokens[i], sampling[i] (struct_size 0 = greedy) and,
// with out_logprobs non-null and sampling, top_logprobs[i] (null = 0). Request i is submitted before poll join[i] (polls
// count from 0; requests joining at one poll are submitted in index order; with no ticket pending the count jumps to
// the next join). Each poll targets a pending ticket picked by a xorshift64 of poll_seed (the target decides when a
// terminal ticket is collected and, S1 G3, whether the poll runs a round: none for a ticket with a round's report
// pending; it never picks the round). Before poll p, graph_modes[p % graph_mode_count] (null or
// count 0 = never) switches the model's graphs: -1 leaves them, else bit 0 = the decode graphs, bit 1 = the spec
// graphs (both restored on return). prefill_chunk_tokens: the model's rows per prefill forward for the call (0 = the
// executor's policy). Outputs: out_tokens / out_scores / out_logprobs + i * out_stride (out_stride >= every
// max_new_tokens[i]); out_results [6 * requests]: {committed, finish, cached_tokens, verify_steps, accepted_drafts,
// the poll that returned it terminal}; out_rounds (host [4 * max_polls], nullable) the predicted round of each poll
// (predict_round); out_polls [1] the polls run. More than max_polls polls: CERE_ERR_CAPACITY. A poll error or a finish
// other than LENGTH / STOP_TOKEN cancels and drains the pending tickets and returns that error.
extern "C" __attribute__((visibility("default"))) int cere_ktest_compose(
    uint64_t model, uint64_t requests, const uint64_t *sessions, const uint32_t *prompts, const uint64_t *prompt_lens,
    const uint64_t *max_new_tokens, const uint32_t *flags, const cere_sampling_params *sampling,
    const uint32_t *top_logprobs, const uint64_t *join, uint64_t poll_seed, const int8_t *graph_modes,
    uint64_t graph_mode_count, uint64_t prefill_chunk_tokens, uint64_t max_polls, uint32_t *out_tokens,
    cere_token_score *out_scores, cere_token_logprobs *out_logprobs, uint64_t out_stride, uint64_t *out_results,
    uint32_t *out_rounds, uint64_t *out_polls, cudaStream_t stream) {
    using cere::ktest::K15Handle;
    cere::ktest::K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || requests == 0 || sessions == nullptr || prompts == nullptr || prompt_lens == nullptr ||
        max_new_tokens == nullptr || flags == nullptr || join == nullptr || out_tokens == nullptr ||
        out_scores == nullptr || out_results == nullptr || out_polls == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    std::vector<cere_session *> session(requests);
    for (uint64_t i = 0; i < requests; ++i) {
        const K15Handle *h = cere::ktest::k15(sessions[i]);
        if (h == nullptr || h->k != k || h->session == nullptr || !h->session->tokens.empty() ||
            max_new_tokens[i] > out_stride) {
            return CERE_ERR_ARGUMENT;
        }
        session[i] = h->session;
        for (uint64_t j = 0; j < 6; ++j) out_results[6 * i + j] = 0;
    }
    *out_polls = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    if (const cere_status st = cere_executor_capabilities(k->executor, k->model, &caps); st != CERE_OK) return st;
    cere::model::DecodeGraphs &graphs = cere::ktest::cuda_model(*k).decode_graphs();
    cere::ktest::GraphScope graph_scope(graphs);
    cere::ktest::ChunkScope chunk(cere::ktest::cuda_model(*k), prefill_chunk_tokens);

    std::vector<uint64_t> offset(requests, 0);
    for (uint64_t i = 1; i < requests; ++i) offset[i] = offset[i - 1] + prompt_lens[i - 1];
    std::vector<cere_work> work(requests);
    std::vector<cere_completion_ticket> tickets(requests);
    std::vector<uint8_t> state(requests, 0);  // 0 not submitted, 1 pending, 2 done
    std::vector<uint64_t> pending;
    const auto drain = [&](cere_status error) {
        for (uint64_t i = 0; i < requests; ++i) {
            if (state[i] != 1) continue;
            cere_executor_cancel(k->executor, tickets[i]);
            for (;;) {
                cere_completion c{};
                c.struct_size = sizeof(c);
                c.abi_version = CERE_EXECUTOR_ABI_VERSION;
                if (cere_executor_poll(k->executor, tickets[i], &c) != CERE_PENDING) break;
            }
        }
        return error;
    };
    uint64_t rng = poll_seed != 0 ? poll_seed : 0x9E3779B97F4A7C15ull;
    uint64_t done = 0;
    uint64_t p = 0;
    while (done < requests) {
        if (pending.empty()) {  // nothing to poll: jump to the next join
            uint64_t next = UINT64_MAX;
            for (uint64_t i = 0; i < requests; ++i) {
                if (state[i] == 0) next = std::min(next, join[i]);
            }
            p = std::max(p, next);
        }
        for (uint64_t i = 0; i < requests; ++i) {
            if (state[i] != 0 || join[i] > p) continue;
            cere_work &w = work[i];
            w = cere_work{};
            w.struct_size = sizeof(cere_work);
            w.abi_version = CERE_EXECUTOR_ABI_VERSION;
            w.operation = CERE_OP_PREFILL;
            w.expected_worker_epoch = caps.worker_epoch;
            w.expected_session_revision = session[i]->revision;
            w.session = session[i];
            w.tokens = prompts + offset[i];
            w.token_count = prompt_lens[i];
            w.max_new_tokens = max_new_tokens[i];
            w.out_tokens = out_tokens + i * out_stride;
            w.out_capacity = max_new_tokens[i];
            w.out_scores = out_scores + i * out_stride;
            w.flags = flags[i];
            if (sampling != nullptr && sampling[i].struct_size != 0) {
                w.sampling = &sampling[i];
                if (out_logprobs != nullptr) {
                    w.out_logprobs = out_logprobs + i * out_stride;
                    w.top_logprobs = top_logprobs != nullptr ? top_logprobs[i] : 0;
                }
            }
            cere_batch batch{};
            batch.struct_size = sizeof(batch);
            batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
            batch.expected_worker_epoch = caps.worker_epoch;
            batch.capability_profile_generation = caps.profile_generation;
            batch.items = &w;
            batch.item_count = 1;
            if (const cere_status st = cere_executor_submit(k->executor, &batch, &tickets[i]); st != CERE_OK) {
                return drain(st);
            }
            state[i] = 1;
            pending.push_back(i);
        }
        if (p >= max_polls) return drain(CERE_ERR_CAPACITY);
        if (graph_modes != nullptr && graph_mode_count != 0) {
            const int8_t mode = graph_modes[p % graph_mode_count];
            if (mode >= 0) {
                graphs.enabled = (mode & 1) != 0;
                graphs.spec = (mode & 2) != 0;
            }
        }
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        const size_t pick = size_t(rng % pending.size());
        const uint64_t i = pending[pick];
        if (out_rounds != nullptr) {
            const cere::host::Ticket &polled = *k->executor->tickets.at(tickets[i].submission_id);
            cere::ktest::predict_round(*k->executor, polled, out_rounds + 4 * p);
        }
        cere_completion c{};
        c.struct_size = sizeof(c);
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        const cere_status st = cere_executor_poll(k->executor, tickets[i], &c);
        ++p;
        *out_polls = p;
        if (st == CERE_PENDING) continue;
        if (st != CERE_OK) return drain(st);
        state[i] = 2;
        pending.erase(pending.begin() + std::ptrdiff_t(pick));
        ++done;
        uint64_t *r = out_results + 6 * i;
        r[0] = c.committed_tokens;
        r[1] = c.finish;
        r[2] = c.cached_tokens;
        r[3] = c.verify_steps;
        r[4] = c.accepted_drafts;
        r[5] = p - 1;
        if (c.finish != CERE_FINISH_LENGTH && c.finish != CERE_FINISH_STOP_TOKEN) {
            return drain(c.status != CERE_OK ? c.status : CERE_ERR_INTERNAL);
        }
    }
    return CERE_OK;
}

// BC0 (the gate's planted k2_gemm mutant, batch-concurrency design note M1): cere_model_set_selection on the model
// (selection: host, a cere_slot_selection); BUSY while a ticket is in flight, as the hot swap.
extern "C" __attribute__((visibility("default"))) int cere_ktest_model_set_selection(
    uint64_t model, const cere_slot_selection *selection, cudaStream_t stream) {
    cere::ktest::K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || selection == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere_model_set_selection(k->executor, k->model, selection);
}

// BC0 (the gate's prod profile: -v52 serves prefix_alt_edges = true): cere_model_set_prefix_alt_edges on the model
// (before its first ticket, else CERE_ERR_BUSY).
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_alt_edges(uint64_t model, uint32_t on,
                                                                                  cudaStream_t stream) {
    cere::ktest::K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere_model_set_prefix_alt_edges(k->executor, k->model, on);
}

// BC0 under S1 G4 (beat-sglang design note): cere_executor_set_decode_share on the model's executor (the shipped
// native-g0-spec serves decode_share_pct 200; 0 = off, the decode_rounds_per_chunk policy). Takes effect at the next
// round.
extern "C" __attribute__((visibility("default"))) int cere_ktest_decode_share(uint64_t model, uint32_t share_pct,
                                                                              cudaStream_t stream) {
    cere::ktest::K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere_executor_set_decode_share(k->executor, share_pct);
}

// ---- M2.10 K16 prefix entries (m210 design note Interfaces, K16 in R1; ledger M4) --------------------------------
// The K16 rows (executor/tests/gpu/test_model_k16_prefix.py) on K15 session handles of a cache-on model
// (cere_model_desc.prefix_state_slots > 0): the cold-split reference (prefill_split: CudaDevice::step with explicit
// call ends through D4′'s `limit`), a prefix hit without a step (prefix_attach: host::attach_prefix, the D3 lookup and
// apply of a ticket's first prefill round), copy_fixed_state -> load_fixed_state into a second session (state_copy,
// K16 (6′)), the device session's position and open prompt (session_device_state, K16 (2′)), the tree's counters
// (prefix_stats), and the ABI's namespace, flush and W 132 / cached_tokens (D10) on a ticket. A session that consumed
// tokens[0, f) of a prompt here holds them as `tokens` with fed = f (the rest of the prompt is passed again).

namespace cere::ktest {
namespace {

// D3′: binds an unbound lazy session's pages the way the first prefill round does (host::attach_prefix over
// tokens[0, n), a namespaced session may hit); the session has then consumed tokens[0, *h).
cere_status attach(K15Handle &h, const uint32_t *tokens, uint64_t n, host::PrefixAttach *out) {
    cere_session &s = *h.session;
    try {
        s.tokens.assign(tokens, tokens + n);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    s.prompt_tokens = n;
    s.submitted = true;
    cere_executor &executor = *h.k->executor;
    if (const cere_status st = host::attach_prefix(executor, s, executor.last_submission + 1, out); st != CERE_OK) {
        s.tokens.clear();
        s.prompt_tokens = 0;
        s.submitted = false;
        return st;
    }
    s.tokens.resize(s.fed);
    return CERE_OK;
}

// The checks shared by prefill_split and prefix_attach: an idle text session, tokens[0, n) below its context (as a
// PREFILL submit) and within the vocabulary.
cere_status check_prompt(const K15Handle &h, const uint32_t *tokens, uint64_t n) {
    const cere_session &s = *h.session;
    if (s.in_flight != 0) return CERE_ERR_BUSY;
    if (s.image_session) return CERE_ERR_ARGUMENT;
    if (n >= s.max_context_tokens) return CERE_ERR_CAPACITY;
    const uint32_t vocab = h.k->model->plan.config.vocab_size;
    for (uint64_t i = 0; i < n; ++i) {
        if (tokens[i] >= vocab) return CERE_ERR_ARGUMENT;
    }
    return CERE_OK;
}

}  // namespace
}  // namespace cere::ktest

// The cold-split reference of K16 (1)/(2)/(8) (D4 (b′), D5, D8a): the session's prompt tokens[0, n) (host) through
// CudaDevice::step in calls ending at call_ends[0, k) (host; strictly increasing in (fed, n]): each call passes
// count = n - position and limit = its end - position, so the device's own 4,096-token chunk still splits a longer
// call exactly as a ticket's would. A fresh unbound session (un-namespaced) is bound first as its first prefill round
// binds it; a session that consumed tokens[0, f) (an earlier split or prefix_attach, so its prompt is open) continues
// at f. With the last end n the call scores the next token into scores[0] (host [1]), which stays uncommitted (as
// after cere_ktest_session_feed's teacher-forced PREFILL: the session's DECODE rows continue with teacher-forced
// feeds); with an end below n the prompt stays open and scores[0] is untouched.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_prefill_split(
    uint64_t session, const uint32_t *tokens, uint64_t n, const uint64_t *call_ends, uint64_t k,
    cere_token_score *scores, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || n == 0 || call_ends == nullptr || k == 0 ||
        scores == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (const cere_status st = cere::ktest::check_prompt(*h, tokens, n); st != CERE_OK) return st;
    cere_session &s = *h->session;
    const uint64_t fed = s.fed;
    if (s.tokens.size() != fed || fed >= n || !std::equal(s.tokens.begin(), s.tokens.end(), tokens)) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint64_t i = 0, previous = fed; i < k; previous = call_ends[i++]) {
        if (call_ends[i] <= previous || call_ends[i] > n) return CERE_ERR_ARGUMENT;
    }
    if (!s.bound) {
        if (fed != 0 || s.namespaced) return CERE_ERR_ARGUMENT;  // a reference never hits
        cere::host::PrefixAttach a;
        if (const cere_status st = cere::ktest::attach(*h, tokens, n, &a); st != CERE_OK) return st;
    }
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*h->k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*h->k);
    auto &csession = static_cast<cere::device::CudaSession &>(*s.device);
    if (fed != 0 && !csession.prompt_open()) return CERE_ERR_ARGUMENT;
    try {
        s.tokens.reserve(n);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    s.prompt_tokens = n;
    s.submitted = true;
    cere_token_score score{};
    uint64_t position = fed;
    cere_status st = CERE_OK;
    for (uint64_t i = 0; i < k && st == CERE_OK; ++i) {
        while (position < call_ends[i]) {
            uint64_t consumed = 0;
            st = device.step(cmodel, csession, tokens + position, n - position, call_ends[i] - position, nullptr,
                             &consumed, nullptr, &score, nullptr, nullptr, nullptr);
            if (st != CERE_OK) break;
            position += consumed;
        }
    }
    // The tokens every successful call consumed.
    s.tokens.assign(tokens, tokens + position);
    s.fed = position;
    if (st == CERE_OK && position == n) scores[0] = score;
    return st;
}

// SelfTune ST1d (executor/tests/gpu/test_graph_launches.py): one MixedPD PREFILL step of the session's prompt
// tokens[0, n) (host) as the scheduler's solo_step makes it, through CudaDevice::step with a StepSlice of budget_ms
// (0: no slice, so a paused call resumes to its end). Before the step the model's park measurement is fixed at
// ms_per_layer_us / 1,000 ms per layer of a call of any length (CudaModel::set_park_cost), so the segment cut
// (model::segment::segment_layers) is the same in every run. The session continues at its fed tokens (bound first as
// cere_ktest_session_prefill_split binds it); a paused call leaves them, and the next call of this entry point with
// the same arguments resumes it. out (host uint64 [4]) = {tokens the step consumed, paused, the paused call's next
// layer (0: none), the session's fed tokens}; when the step ends the prompt, scores[0] (host [1]) is its uncommitted
// next-token score. A session without the model's park runs whole (paused 0).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_prefill_sliced(
    uint64_t session, const uint32_t *tokens, uint64_t n, uint64_t budget_ms, uint64_t ms_per_layer_us,
    uint64_t *out, cere_token_score *scores, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || n == 0 || out == nullptr || scores == nullptr ||
        budget_ms > UINT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    for (int i = 0; i < 4; ++i) out[i] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (const cere_status st = cere::ktest::check_prompt(*h, tokens, n); st != CERE_OK) return st;
    cere_session &s = *h->session;
    const uint64_t fed = s.fed;
    if (s.tokens.size() != fed || fed >= n || !std::equal(s.tokens.begin(), s.tokens.end(), tokens)) {
        return CERE_ERR_ARGUMENT;
    }
    if (!s.bound) {
        if (fed != 0 || s.namespaced) return CERE_ERR_ARGUMENT;
        cere::host::PrefixAttach a;
        if (const cere_status st = cere::ktest::attach(*h, tokens, n, &a); st != CERE_OK) return st;
    }
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*h->k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*h->k);
    auto &csession = static_cast<cere::device::CudaSession &>(*s.device);
    if (fed != 0 && !csession.prompt_open()) return CERE_ERR_ARGUMENT;
    try {
        s.tokens.reserve(n);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    s.prompt_tokens = n;
    s.submitted = true;
    cmodel.set_park_cost(double(ms_per_layer_us) / 1000.0, 0);
    cere::host::StepSlice slice{uint32_t(budget_ms), false};
    cere_token_score score{};
    uint64_t consumed = 0;
    const cere_status st = device.step(cmodel, csession, tokens + fed, n - fed, n - fed, nullptr, &consumed, nullptr,
                                       &score, nullptr, nullptr, budget_ms != 0 ? &slice : nullptr);
    if (st != CERE_OK) return st;
    const uint64_t position = fed + consumed;
    s.tokens.assign(tokens, tokens + position);
    s.fed = position;
    if (position == n) scores[0] = score;
    out[0] = consumed;
    out[1] = slice.paused ? 1 : 0;
    out[2] = cmodel.parked_layer();
    out[3] = position;
    return CERE_OK;
}

// K16 (2′), (8): a prefix hit without a step. The fresh namespaced session's prompt tokens[0, n) (host) goes through
// host::attach_prefix (the first prefill round's D3 lookup, lock, pages and load_slot); the session has then consumed
// tokens[0, h) with its prompt open (h 0 on a miss: bound, nothing consumed). out (host [2]) = {h, k} (k: the
// tokens of whole pages that match the tree, D4 (d)).
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_attach(uint64_t session,
                                                                               const uint32_t *tokens, uint64_t n,
                                                                               uint64_t *out, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || n == 0 || out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    out[0] = out[1] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (h->k->model->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
    if (const cere_status st = cere::ktest::check_prompt(*h, tokens, n); st != CERE_OK) return st;
    const cere_session &s = *h->session;
    if (s.submitted || s.bound || !s.tokens.empty()) return CERE_ERR_ARGUMENT;
    cere::host::PrefixAttach a;
    if (const cere_status st = cere::ktest::attach(*h, tokens, n, &a); st != CERE_OK) return st;
    out[0] = a.h;
    out[1] = a.k;
    return CERE_OK;
}

// K16 (2′): the device session's out (host [4]) = {position, prompt open (the next step is PREFILL), pages bound,
// lazy}.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_device_state(uint64_t session, uint64_t *out,
                                                                                      cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    const cere::device::CudaDevice &device = cere::ktest::cuda_device(*h->k);
    const auto &csession = static_cast<const cere::device::CudaSession &>(*h->session->device);
    out[0] = device.session_position(csession);
    out[1] = csession.prompt_open() ? 1 : 0;
    out[2] = device.session_page_list(csession).size();
    out[3] = csession.lazy() ? 1 : 0;
    return CERE_OK;
}

// K16 (6′) (ledger M4): copy_fixed_state of the source session (it consumed tokens[0, p) of an open prompt, p a
// multiple of 64; CudaDevice::capture_state into a slot the tree hands out, host::PrefixCache::take_slot) ->
// load_fixed_state into the fresh unbound destination (bind_pages with the source's pages [0, p / 64) shared, then
// CudaDevice::load_slot): the destination has then consumed the same tokens[0, p) with its prompt open. The slot goes
// back to the tree (stream order puts the load before any later capture into it). CERE_ERR_CAPACITY when no slot is
// obtainable (every state locked) or the destination's own pages are not free.
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_state_copy(uint64_t source,
                                                                                   uint64_t destination,
                                                                                   cudaStream_t stream) {
    K15Handle *a = cere::ktest::k15(source);
    K15Handle *b = cere::ktest::k15(destination);
    if (a == nullptr || b == nullptr || a == b || a->session == nullptr || b->session == nullptr || a->k != b->k) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    cere::host::PrefixCache *cache = a->k->model->prefix.get();
    if (cache == nullptr) return CERE_ERR_UNSUPPORTED;
    cere_session &s = *a->session;
    cere_session &d = *b->session;
    if (s.in_flight != 0 || d.in_flight != 0) return CERE_ERR_BUSY;
    const uint64_t p = s.fed;
    if (s.image_session || p == 0 || p % cere::host::kKvPageTokens != 0 || s.tokens.size() != p || d.submitted ||
        d.bound || p >= d.max_context_tokens) {
        return CERE_ERR_ARGUMENT;
    }
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*a->k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*a->k);
    const auto &csource = static_cast<const cere::device::CudaSession &>(*s.device);
    auto &cdestination = static_cast<cere::device::CudaSession &>(*d.device);
    if (!csource.prompt_open()) return CERE_ERR_ARGUMENT;
    std::vector<uint32_t> pages;
    try {
        pages = device.session_page_list(csource);
        d.tokens.reserve(p);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    uint32_t slot = 0;
    if (!cache->take_slot(&slot)) return CERE_ERR_CAPACITY;
    cere_status st = device.capture_state(cmodel, csource, slot);
    if (st == CERE_OK) st = device.bind_pages(cmodel, cdestination, pages.data(), p / cere::host::kKvPageTokens);
    if (st == CERE_OK) {
        d.bound = true;
        d.model->unbound_pages -= cere::host::kv_pages(d.max_context_tokens);
        st = device.load_slot(cmodel, cdestination, slot, p);
    }
    cache->free_slot(slot);
    if (st != CERE_OK) return st;
    d.tokens = s.tokens;
    d.fed = p;
    d.prompt_tokens = s.prompt_tokens;
    d.submitted = true;
    return CERE_OK;
}

// The tree's counters (host::PrefixStats: hits, captures, capture_skipped, inserts, state_evictions, leaf_evictions,
// host_spills, host_loads, host_drops), the KV pool's free pages and capture_page_mismatch (M210S-39): out (host [11]).
// CERE_ERR_UNSUPPORTED on a model with prefix_state_slots 0.
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_stats(uint64_t model, uint64_t *out,
                                                                              cudaStream_t stream) {
    const K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (k->model->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
    const cere::host::PrefixStats &stats = k->model->prefix->stats();
    out[0] = stats.hits;
    out[1] = stats.captures;
    out[2] = stats.capture_skipped;
    out[3] = stats.inserts;
    out[4] = stats.state_evictions;
    out[5] = stats.leaf_evictions;
    out[6] = stats.host_spills;
    out[7] = stats.host_loads;
    out[8] = stats.host_drops;
    out[9] = cere::ktest::cuda_device(*k).free_pages(*k->model->device);
    out[10] = stats.capture_page_mismatch;
    return CERE_OK;
}

// cere_session_set_cache_namespace(namespace_id: host [16]) of the session (D2; before its first submit).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_set_namespace(uint64_t session,
                                                                                       const uint8_t *namespace_id,
                                                                                       cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere_session_set_cache_namespace(h->k->executor, h->session, namespace_id);
}

// cere_model_flush_prefix_cache of the model handle's model (D11).
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_flush(uint64_t model, cudaStream_t stream) {
    K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere_model_flush_prefix_cache(k->executor, k->model);
}

// cere_model_set_prefix_end_state of the model handle's model (m210 D15⁗′ stage 2a; K16 (11) turns it on).
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_end_state(uint64_t model, uint32_t on,
                                                                                  cudaStream_t stream) {
    K8Model *k = cere::ktest::handle(model);
    if (k == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere_model_set_prefix_end_state(k->executor, k->model, on);
}

// K16 (11), m210 D15⁗′ (the κ mutant): the end state at exactly tokens[0, len) (host) under namespace_id (host [16])
// gets class `cls` (> 0) in the tree; old (host [1]) = its class before. The device state is untouched, so a drafting
// ticket whose own class is `cls` hits it and resumes it with that class. CERE_ERR_ARGUMENT when no end state is
// there.
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_end_class(uint64_t model,
                                                                                  const uint8_t *namespace_id,
                                                                                  const uint32_t *tokens, uint64_t len,
                                                                                  uint64_t cls, uint64_t *old,
                                                                                  cudaStream_t stream) {
    K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || namespace_id == nullptr || tokens == nullptr || len == 0 || cls == 0 || old == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    if (k->model->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
    cere::host::CacheNamespace ns;
    std::memcpy(ns.data(), namespace_id, ns.size());
    const cere::host::PrefixNode *node = k->model->prefix->root(ns);
    while (node != nullptr && node->depth < len) {
        const cere::host::PrefixNode *next = nullptr;
        for (const auto &c : node->children) {
            const cere::host::PrefixNode &child = *c.second;
            if (child.depth <= len && std::equal(child.key.begin(), child.key.end(), tokens + child.start())) {
                next = &child;
                break;
            }
        }
        node = next;
    }
    if (node == nullptr || node->depth != len || node->cls == 0) return CERE_ERR_ARGUMENT;
    *old = node->cls;
    const_cast<cere::host::PrefixNode *>(node)->cls = cls;
    return CERE_OK;
}

// K16 (13), m210 D15⁗″ (ResumeExact): out (host [1]) = the device's row-bit signature σ(s, cls) of the model handle's
// model (CudaDevice::row_sig: two classes share a hit iff their σ are equal).
extern "C" __attribute__((visibility("default"))) int cere_ktest_prefix_row_sig(uint64_t model, uint64_t s,
                                                                                uint64_t cls, uint64_t *out,
                                                                                cudaStream_t stream) {
    K8Model *k = cere::ktest::handle(model);
    if (k == nullptr || cls == 0 || out == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    *out = cere::ktest::cuda_device(*k).row_sig(*k->model->device, s, cls);
    return CERE_OK;
}

// K16 (13) K16RowSig: the row counts in [CudaDevice::kRowSigMinRows, cap] at which some per-call choice of a
// PREFILL call of that many rows changes, ascending: each K2 bucket's first m (k2_gemm.cpp bucket_lower), each pinned
// plan's pin range ends (m_lo and m_hi + 1, kSharedGateUpPins / kAttnQkvPins), each P3b bucket's first m
// (p3b_tactic.h kPinnedBuckets) and the first row count above the dense K6 limit (k6::kMaxContext + 1), plus
// kRowSigMinRows itself. out (host [capacity]) holds the first *count (host [1]) values; CERE_ERR_CAPACITY when more.
extern "C" __attribute__((visibility("default"))) int cere_ktest_row_choice_bounds(uint64_t cap, uint64_t *out,
                                                                                   uint64_t capacity,
                                                                                   uint64_t *count,
                                                                                   cudaStream_t stream) {
    if (out == nullptr || count == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    const uint64_t lo = cere::device::CudaDevice::kRowSigMinRows;
    std::vector<uint64_t> bounds{lo, uint64_t(cere::k6::kMaxContext) + 1};
    for (uint32_t b = 1; b < cere::kernels::GemmPlan::kMaxBuckets; ++b) bounds.push_back((uint64_t{1} << (b - 1)) + 1);
    for (const uint32_t n : {cere::model::kSharedInter, cere::model::kQkvWidth}) {
        const cere::kernels::GemmPins *pins =
            cere::kernels::gemm_pins_for(n, cere::model::kHidden, cere::kernels::GemmOut::kBF16);
        for (uint32_t i = 0; pins != nullptr && i < pins->count; ++i) {
            bounds.push_back(pins->ranges[i].m_lo);
            bounds.push_back(uint64_t(pins->ranges[i].m_hi) + 1);
        }
    }
    for (const cere::p3b::PinnedBucket &b : cere::p3b::kPinnedBuckets) {
        if (b.max_m != UINT32_MAX) bounds.push_back(uint64_t(b.max_m) + 1);
    }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    uint64_t written = 0;
    for (const uint64_t m : bounds) {
        if (m < lo || m > cap) continue;
        if (written == capacity) return CERE_ERR_CAPACITY;
        out[written++] = m;
    }
    *count = written;
    return CERE_OK;
}

// One greedy PREFILL work item of tokens[0, n) (host) through submit/poll on an empty session with W 132's
// capture_position (0 = none): scores[0, committed) (host) are its scores, out_counts (host [3]) = {committed tokens,
// finish, cached_tokens} (D10: the prefix the ticket took from the cache).
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_prefill_cached(
    uint64_t session, const uint32_t *tokens, uint64_t n, uint64_t max_new_tokens, uint64_t capture_position,
    cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || n == 0 || max_new_tokens == 0 ||
        scores == nullptr || out_counts == nullptr || !h->session->tokens.empty()) {
        return CERE_ERR_ARGUMENT;
    }
    out_counts[0] = out_counts[1] = out_counts[2] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere::ktest::session_ticket(*h, tokens, n, max_new_tokens, 0, scores, out_counts, nullptr, 0,
                                       capture_position, &out_counts[2]);
}

// m210 D15⁗ (ledger CR8, K16 (9)): cere_ktest_session_prefill_cached without W 132 and with `flags` (u32) as the
// work's cere_work.flags (CERE_WORK_NO_SPECULATION or 0: a drafting PREFILL on a model with a draft): out_counts
// (host [6]) = {committed tokens, finish, cached_tokens, verify_steps, accepted_drafts, verified_drafts}.
extern "C" __attribute__((visibility("default"))) int cere_ktest_session_prefill_cached_flags(
    uint64_t session, const uint32_t *tokens, uint64_t n, uint64_t max_new_tokens, uint64_t flags,
    cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream) {
    K15Handle *h = cere::ktest::k15(session);
    if (h == nullptr || h->session == nullptr || tokens == nullptr || n == 0 || max_new_tokens == 0 ||
        scores == nullptr || out_counts == nullptr || !h->session->tokens.empty() || flags > UINT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    for (int i = 0; i < 6; ++i) out_counts[i] = 0;
    if (const cere_status st = cere::ktest::sync(stream); st != CERE_OK) return st;
    return cere::ktest::session_ticket(*h, tokens, n, max_new_tokens, 0, scores, out_counts, nullptr, 0, 0,
                                       &out_counts[2], uint32_t(flags), &out_counts[3]);
}

// m23 design note D12 (the forward profile; evidence, never a verdict): `rows` sessions (rows in [1, min(the plan's
// max_sessions, GemmPlan::kMaxRows)]), each created at context prompt_len + steps, prefill the same `prompt`
// (prompt_len >= 1 tokens, each < vocab_size) one after another in session order (the single-row PREFILL forward in
// chunks of the workspace rows), then run `steps` greedy decode steps of all rows at once (one row: the single-row
// DECODE forward; more: CudaDevice::forward_rows, as step_rows runs them), row r feeding its own top-1. The ktest
// marker kernel (ktest/profile_mark.h) runs on `stream`, synchronized after it, before session 0's prefill, after
// it, after the last prefill and after every decode step (steps + 3 marks), so a CUPTI kernel trace of the call (the
// test's torch.profiler) splits into session 0's prefill, the other sessions' prefills and the decode steps.
// out_ms (host, float [1 + steps]): [0] session 0's prefill, [1 + s] decode step s: the host steady-clock ms around
// the forward calls, each of which synchronizes the executor's stream. Sessions are released before return.
extern "C" __attribute__((visibility("default"))) int cere_ktest_forward_profile(uint64_t model, uint64_t rows,
                                                                                const uint32_t *prompt,
                                                                                uint64_t prompt_len, uint64_t steps,
                                                                                float *out_ms, cudaStream_t stream) {
    constexpr uint32_t kMaxRows = cere::kernels::GemmPlan::kMaxRows;
    const K8Model *k = cere::ktest::handle(model);
    const uint64_t ctx = k == nullptr ? 0 : k->model->plan.max_context_tokens;
    if (k == nullptr || prompt == nullptr || out_ms == nullptr || rows == 0 || rows > k->model->plan.max_sessions ||
        rows > kMaxRows || prompt_len == 0 || steps == 0 || prompt_len > ctx || steps > ctx - prompt_len) {
        return CERE_ERR_ARGUMENT;
    }
    const uint32_t vocab = k->model->plan.config.vocab_size;
    for (uint64_t i = 0; i < prompt_len; ++i) {
        if (prompt[i] >= vocab) return CERE_ERR_ARGUMENT;
    }
    cere::device::CudaDevice &device = cere::ktest::cuda_device(*k);
    cere::device::CudaModel &cmodel = cere::ktest::cuda_model(*k);
    std::unique_ptr<cere::ktest::SessionScope> scopes[kMaxRows];
    cere::device::CudaSession *sessions[kMaxRows];
    for (uint64_t r = 0; r < rows; ++r) {
        scopes[r].reset(new (std::nothrow) cere::ktest::SessionScope(*k));
        if (scopes[r] == nullptr) return CERE_ERR_CAPACITY;
        if (const cere_status st = scopes[r]->create(prompt_len + steps); st != CERE_OK) return st;
        sessions[r] = &static_cast<cere::device::CudaSession &>(*scopes[r]->get()->device);
    }
    const auto mark = [stream]() {
        if (cere::ktest::profile_mark(stream) != cudaSuccess) return CERE_ERR_INTERNAL;
        return cere::ktest::sync(stream);
    };
    using Clock = std::chrono::steady_clock;
    const auto ms_since = [](Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    };
    const uint64_t ws_rows = cmodel.workspace_rows();
    uint32_t tokens[kMaxRows];
    if (const cere_status st = mark(); st != CERE_OK) return st;
    for (uint64_t r = 0; r < rows; ++r) {
        const Clock::time_point t0 = Clock::now();
        cere_token_score last{};
        for (uint64_t p0 = 0; p0 < prompt_len;) {
            const uint64_t n = std::min(ws_rows, prompt_len - p0);
            if (const cere_status st = device.forward(cmodel, *sessions[r], CERE_OP_PREFILL, prompt + p0,
                                                      uint32_t(n), nullptr, &last);
                st != CERE_OK) {
                return st;
            }
            p0 += n;
        }
        tokens[r] = last.id;
        if (r == 0) {
            out_ms[0] = ms_since(t0);
            if (const cere_status st = mark(); st != CERE_OK) return st;
        }
    }
    if (const cere_status st = mark(); st != CERE_OK) return st;
    for (uint64_t s = 0; s < steps; ++s) {
        cere_token_score out[kMaxRows];
        const Clock::time_point t0 = Clock::now();
        const cere_status st = rows == 1
                                   ? device.forward(cmodel, *sessions[0], CERE_OP_DECODE, tokens, 1, nullptr, out)
                                   : device.forward_rows(cmodel, sessions, uint32_t(rows), tokens, nullptr, out);
        out_ms[1 + s] = ms_since(t0);
        if (st != CERE_OK) return st;
        for (uint64_t r = 0; r < rows; ++r) tokens[r] = out[r].id;
        if (const cere_status mst = mark(); mst != CERE_OK) return mst;
    }
    return CERE_OK;
}
