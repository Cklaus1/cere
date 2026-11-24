// SelfTune ST4 S3c (selftune design note ST4): the CUDA `Device` (host/device.h) of an sdar library (DICE-8B).
// One model of max_sessions 1, greedy only, eager, BF16 weights and KV; the per-token `step` contract runs on the
// block-diffusion schedule (model/sdar/schedule.h) over the model's forward (model/sdar/sdar_model.h). Everything
// DICE-8B runs without (prefix cache, snapshots, drafts, research, LoRA, images, sampling, logprobs) keeps the
// Device defaults (UNSUPPORTED) and is refused by name before it gets here (api.cpp plan_sdar_model, host/family.h).
//
// Device id: the decimal CUDA ordinal among the visible devices, as the Qwen CUDA device (device/cuda_device.h):
// empty, a sign, a leading zero or more than 9 digits is ARGUMENT; a non-decimal id or any ",option" is UNSUPPORTED;
// a missing device, one other than compute capability 12.0, or a process without eager module loading
// (CUDA_MODULE_LOADING=EAGER) is UNSUPPORTED.
#include "host/device.h"
#include "host/support.h"
#include "model/sdar/sdar_model.h"

#include "cere_model_constants.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace cere::sdar {

namespace {

constexpr uint64_t kMaxContextTokens = gen::kMaxPositionEmbeddings;
constexpr size_t kMaxOrdinalDigits = 9;

cere_status alloc_status(cudaError_t e) {
    if (e == cudaSuccess) return CERE_OK;
    cudaGetLastError();
    return e == cudaErrorMemoryAllocation ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
}

const host::Support &dice_support() {
    static const host::Support support{
        "dice-8b",
        cere_qwen4_exp_config{},
        1,                                    // max_sessions: one session (ST4 plan S3, D3)
        kMaxContextTokens,                    // max_context_tokens: config max_position_embeddings
        kMaxContextTokens,                    // max_kv_pool_tokens (no paged pool; host/kv_pool.h bound only)
        host::dtype_bit(CERE_DTYPE_BF16),
        nullptr,
    };
    return support;
}

struct DeviceBuffer {
    void *ptr = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() {
        if (ptr != nullptr) cudaFree(ptr);
    }
    cere_status allocate(uint64_t bytes) {
        return bytes == 0 ? CERE_OK : alloc_status(cudaMalloc(&ptr, bytes));
    }
    uint8_t *bytes() const { return static_cast<uint8_t *>(ptr); }
};

class SdarModel final : public host::DeviceModel {
public:
    std::vector<TensorSlot> slots;
    std::vector<int64_t> slot_of;  // plan tensor index -> slots index, -1 for a discarded tensor
    uint64_t arena_bytes = 0;
    DeviceBuffer arena;
    DeviceBuffer workspace;
    Workspace ws;
    Weights weights;
    Gemms gemms;
    bool instantiated = false;
};

class SdarSession final : public host::DeviceSession {
public:
    Session schedule;
    DeviceBuffer kv;  // K of every layer, then V of every layer
    uint64_t rows = 0;
};

class SdarDevice final : public host::Device {
public:
    static cere_status open(const char *device_id, std::unique_ptr<host::Device> *out);
    ~SdarDevice() override {
        if (stream_ != nullptr) {
            cudaSetDevice(ordinal_);
            cudaStreamDestroy(stream_);
        }
        if (handle_ != nullptr) cublasLtDestroy(handle_);
    }

    std::vector<const host::Support *> supported() const override { return {}; }
    uint64_t host_budget_bytes() const override { return 0; }

    host::MemoryEstimate estimate(const host::ModelPlan &plan) const override {
        std::vector<TensorSlot> slots = device_slots(plan, nullptr);
        const uint64_t arena = layout_tensors(&slots);
        return {arena + workspace_bytes() + gemm_workspace_bytes(plan.sdar_gemm) +
                    plan.max_sessions * kv_bytes(plan.max_context_tokens),
                0};
    }

    cere_status allocate(const host::ModelPlan &plan, std::unique_ptr<host::DeviceModel> *out) override {
        if (const cere_status st = current(); st != CERE_OK) return st;
        auto model = std::make_unique<SdarModel>();
        model->slots = device_slots(plan, &model->slot_of);
        model->arena_bytes = layout_tensors(&model->slots);
        if (const cere_status st = model->arena.allocate(model->arena_bytes); st != CERE_OK) return st;
        if (const cere_status st = model->workspace.allocate(workspace_bytes()); st != CERE_OK) return st;
        model->ws = carve_workspace(model->workspace.bytes());
        if (const cere_status st = model->gemms.init(handle_, plan.sdar_gemm); st != CERE_OK) return st;
        *out = std::move(model);
        return CERE_OK;
    }

    cere_status write(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                      uint64_t len) override {
        auto &m = static_cast<SdarModel &>(model);
        if (tensor_index >= m.slot_of.size() || m.slot_of[tensor_index] < 0 || bytes == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        const TensorSlot &slot = m.slots[size_t(m.slot_of[tensor_index])];
        if (offset > slot.bytes || len > slot.bytes - offset) return CERE_ERR_ARGUMENT;
        if (const cere_status st = current(); st != CERE_OK) return st;
        const cudaError_t e = cudaMemcpy(m.arena.bytes() + slot.offset + offset, bytes, len, cudaMemcpyHostToDevice);
        return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
    }

    uint32_t moe_lever() const override { return CERE_LEVER_MOE_P3B; }

    cere_status set_levers(host::DeviceModel &, const host::Levers &) override {
        // The forward has one implementation per slot; api.cpp refused every non-default choice by name
        // (host/family.h sdar_selection_refusal), so the resolved levers select nothing here.
        return CERE_OK;
    }

    cere_status instantiate(host::DeviceModel &model) override {
        auto &m = static_cast<SdarModel &>(model);
        if (m.instantiated) return CERE_ERR_ARGUMENT;
        if (const cere_status st = bind_weights(m.slots, m.arena.bytes(), &m.weights); st != CERE_OK) return st;
        m.instantiated = true;
        return CERE_OK;
    }

    bool forward_supported() const override { return true; }
    bool sampler_supported() const override { return false; }

    cere_status create_session(host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                               std::unique_ptr<host::DeviceSession> *out) override {
        if (lazy || max_context_tokens == 0 || max_context_tokens > kMaxContextTokens) return CERE_ERR_UNSUPPORTED;
        if (!static_cast<SdarModel &>(model).instantiated) return CERE_ERR_ARGUMENT;
        if (const cere_status st = current(); st != CERE_OK) return st;
        auto session = std::make_unique<SdarSession>();
        session->rows = kv_rows(max_context_tokens);
        if (const cere_status st = session->kv.allocate(kv_bytes(max_context_tokens)); st != CERE_OK) return st;
        session->schedule.capacity = max_context_tokens;
        *out = std::move(session);
        return CERE_OK;
    }

    cere_status step(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *tokens, uint64_t count,
                     uint64_t limit, const host::StepImages *images, uint64_t *consumed,
                     const host::ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                     const host::EndStep *end, host::StepSlice *slice) override {
        if (images != nullptr || sampling != nullptr || logprobs != nullptr || end != nullptr) {
            return CERE_ERR_UNSUPPORTED;
        }
        if (slice != nullptr) slice->paused = false;
        auto &m = static_cast<SdarModel &>(model);
        auto &s = static_cast<SdarSession &>(session);
        if (const cere_status st = current(); st != CERE_OK) return st;
        bf16 *k = reinterpret_cast<bf16 *>(s.kv.bytes());
        bf16 *v = k + kv_half_elements(s.rows);
        CudaForward forward(m.weights, m.gemms, m.ws, k, v, s.rows, stream_);
        return cere::sdar::step(forward, s.schedule, tokens, count, limit, gen::kDecodeMaskTokenId, consumed, out);
    }

    cere_status step_rows(host::DeviceModel &model, const host::RowArgs *rows, uint32_t row_count,
                          cere_token_score *out, cere_status *row_status) override {
        if (rows == nullptr || row_count == 0 || out == nullptr || row_status == nullptr) return CERE_ERR_ARGUMENT;
        for (uint32_t r = 0; r < row_count; ++r) {
            const host::RowArgs &row = rows[r];
            if (row.session == nullptr) {
                row_status[r] = CERE_ERR_ARGUMENT;
                continue;
            }
            if (row.sampling != nullptr || row.logprobs != nullptr) {
                row_status[r] = CERE_ERR_UNSUPPORTED;
                continue;
            }
            uint64_t consumed = 0;
            row_status[r] = step(model, *row.session, &row.token, 1, 0, nullptr, &consumed, nullptr, &out[r],
                                 nullptr, nullptr, nullptr);
            if (row_status[r] == CERE_OK && consumed != 1) row_status[r] = CERE_ERR_INTERNAL;
        }
        return CERE_OK;
    }

    cere_status set_rope_delta(host::DeviceModel &, host::DeviceSession &, int64_t rope_delta) override {
        return rope_delta == 0 ? CERE_OK : CERE_ERR_UNSUPPORTED;
    }

private:
    SdarDevice(int ordinal, cudaStream_t stream, cublasLtHandle_t handle)
        : ordinal_(ordinal), stream_(stream), handle_(handle) {}

    cere_status current() const { return cudaSetDevice(ordinal_) == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; }

    // Elements of one of the session's two halves (K or V): every layer's rows.
    static uint64_t kv_half_elements(uint64_t rows) {
        return uint64_t(gen::kNumHiddenLayers) * rows * gen::kNumKeyValueHeads * uint64_t(kHeadDim);
    }

    // The plan's device tensors as slots (offsets unset), and the plan index -> slot index map.
    static std::vector<TensorSlot> device_slots(const host::ModelPlan &plan, std::vector<int64_t> *slot_of) {
        std::vector<TensorSlot> slots;
        if (slot_of != nullptr) slot_of->assign(plan.tensors.size(), -1);
        for (size_t i = 0; i < plan.tensors.size(); ++i) {
            const host::TensorPlan &t = plan.tensors[i];
            if (t.placement == CERE_PLACEMENT_DISCARD) continue;
            if (slot_of != nullptr) (*slot_of)[i] = int64_t(slots.size());
            TensorSlot slot{t.name, t.dtype, t.ndim, {}, t.bytes, 0};
            std::copy(t.shape, t.shape + CERE_TENSOR_MAX_DIMS, slot.shape);
            slots.push_back(std::move(slot));
        }
        return slots;
    }

    int ordinal_;
    cudaStream_t stream_;
    cublasLtHandle_t handle_;
};

cere_status SdarDevice::open(const char *device_id, std::unique_ptr<host::Device> *out) {
    if (device_id == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const std::string s(device_id);
    if (s.find(',') != std::string::npos) return CERE_ERR_UNSUPPORTED;
    if (s.empty()) return CERE_ERR_ARGUMENT;
    const size_t sign = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    const std::string digits = s.substr(sign);
    const bool decimal =
        !digits.empty() && std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; });
    if (!decimal) return sign != 0 ? CERE_ERR_ARGUMENT : CERE_ERR_UNSUPPORTED;
    if (sign != 0 || (digits.size() > 1 && digits[0] == '0') || digits.size() > kMaxOrdinalDigits) {
        return CERE_ERR_ARGUMENT;
    }
    const char *loading = std::getenv("CUDA_MODULE_LOADING");
    if (loading == nullptr || std::strcmp(loading, "EAGER") != 0) return CERE_ERR_UNSUPPORTED;
    const int ordinal = std::stoi(digits);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_UNSUPPORTED;
    }
    if (ordinal >= count) return CERE_ERR_UNSUPPORTED;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) return CERE_ERR_INTERNAL;
    if (prop.major != 12 || prop.minor != 0) return CERE_ERR_UNSUPPORTED;
    if (cudaSetDevice(ordinal) != cudaSuccess) return CERE_ERR_INTERNAL;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return CERE_ERR_INTERNAL;
    cublasLtHandle_t handle = nullptr;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream);
        return CERE_ERR_INTERNAL;
    }
    out->reset(new SdarDevice(ordinal, stream, handle));
    return CERE_OK;
}

}  // namespace

}  // namespace cere::sdar

namespace cere::host {

cere_status create_device(const char *device_id, std::unique_ptr<Device> *out) {
    return cere::sdar::SdarDevice::open(device_id, out);
}

const Support *sdar_support(const Device &) { return &cere::sdar::dice_support(); }

}  // namespace cere::host
