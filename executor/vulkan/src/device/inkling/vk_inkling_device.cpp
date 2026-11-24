// Inkling 2 Mini (inkling design note S11): the Vulkan `Device` (host/device.h) of an inkling2_loop library
// (executor/vulkan/build_lib.sh CERE_MODEL=inkling-2-mini). It opens a Vulkan device as the A4 device does (the decimal
// ordinal among the selectable devices, the embedded SPIR-V table) and runs the text forward of ki_model.h: greedy
// only, checkpoint tensors as stored (BF16, FP32, NVFP4 / MXFP8 experts), each tensor in its own host-visible buffer
// (UMA: AMD's RADV). A session owns its state (ki_model.h Session); `step` prefills up to 4,096 tokens per call and
// decodes a single token through the same Forward::step, and `step_rows` runs its rows one session at a time through
// that same 1-row step, so a session's step and a step_rows row are bit-identical (G4) at any session count.
// Dense projections run on K2 when the device has the cooperative-matrix GEMM (AMD), else on ki_dense (CPU devices,
// tests only). The scores are computed on the host from the mapped FP32 logits with CUDA's order (model/inkling/
// logits.cu top2_rows: NaN beats everything but an earlier NaN, ties to the lower id).
// Vision (S7), the MTP drafter (S9) and the prefix cache (S6) are refused by name before a model reaches the device
// (api.cpp plan_inkling2_model, host/family.h inkling2_refusal); sampling, logprobs, images and every other optional
// call keep the Device defaults (UNSUPPORTED).
#include "host/device.h"
#include "host/support.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "device/inkling/ki_model.h"
#include "launch/vk_launch.h"

namespace cere::vkink {
namespace {

constexpr size_t kMaxOrdinalDigits = 9;
constexpr uint64_t kMaxSessions = 64;
constexpr uint64_t kPrefillRows = kMaxRows;

const host::Support &inkling_support() {
    static const host::Support support{
        "inkling-2-mini",
        cere_qwen4_exp_config{},  // unused: plan_inkling2_model checks cere_inkling2_config itself
        kMaxSessions,
        kMaxContext,
        kMaxSessions * kMaxContext,  // no paged pool: host/kv_pool.h bound only (KV is per session)
        host::dtype_bit(CERE_DTYPE_BF16) | host::dtype_bit(CERE_DTYPE_F32) | host::dtype_bit(CERE_DTYPE_F8_E4M3) |
            host::dtype_bit(CERE_DTYPE_U8),
        nullptr,
    };
    return support;
}

bool beats(float va, uint32_t ia, float vb, uint32_t ib) {
    const bool na = va != va, nb = vb != vb;
    if (na || nb) return na && (!nb || ia < ib);
    return va > vb || (va == vb && ia < ib);
}

cere_token_score top2(const float *row) {
    uint32_t b = 0, r = 1;
    if (beats(row[1], 1, row[0], 0)) std::swap(b, r);
    for (uint32_t c = 2; c < kVocab; ++c) {
        if (beats(row[c], c, row[b], b)) {
            r = b;
            b = c;
        } else if (beats(row[c], c, row[r], r)) {
            r = c;
        }
    }
    return cere_token_score{b, r, row[b], row[r]};
}

struct Slot {
    std::string name;
    uint64_t bytes = 0;
    vk::Buffer buf;
};

class InklingModel final : public host::DeviceModel {
public:
    explicit InklingModel(vk::Context &ctx) : ctx_(ctx), forward(ctx) {}
    ~InklingModel() override {
        for (Slot &s : slots)
            if (s.buf.addr != 0) ctx_.free(s.buf);
    }

private:
    vk::Context &ctx_;

public:
    std::vector<Slot> slots;
    std::vector<int64_t> slot_of;
    Forward forward;
    bool instantiated = false;
};

class InklingSession final : public host::DeviceSession {
public:
    std::unique_ptr<Session> s;
    uint64_t pos = 0;     // positions [0, pos) are in the session's state
    bool broken = false;  // a failed call: every later step is CERE_ERR_INTERNAL
};

class InklingDevice final : public host::Device {
public:
    static cere_status open(const char *device_id, std::unique_ptr<host::Device> *out);

    std::vector<const host::Support *> supported() const override { return {}; }
    uint64_t host_budget_bytes() const override { return 0; }

    bool faulted() const override {
        if (!ctx_->faulted()) return false;
        if (!poisoned_reported_.exchange(true)) {
            std::fprintf(stderr, "cere: device_poisoned vk=%u error=%s reason=%s\n", ctx_->info().ordinal,
                         vk::result_name(ctx_->fault_result()), ctx_->fault_reason().c_str());
        }
        return true;
    }

    host::MemoryEstimate estimate(const host::ModelPlan &plan) const override {
        uint64_t n = Forward::scratch_bytes(options());
        for (const host::TensorPlan &t : plan.tensors)
            if (t.placement != CERE_PLACEMENT_DISCARD) n += t.bytes + 256;
        return {n + plan.max_sessions * Forward::session_bytes(plan.max_context_tokens), 0};
    }

    cere_status allocate(const host::ModelPlan &plan, std::unique_ptr<host::DeviceModel> *out) override {
        if (plan.max_sessions > kMaxSessions || plan.max_context_tokens > kMaxContext) return CERE_ERR_UNSUPPORTED;
        auto m = std::make_unique<InklingModel>(*ctx_);
        m->slot_of.assign(plan.tensors.size(), -1);
        for (size_t i = 0; i < plan.tensors.size(); ++i) {
            const host::TensorPlan &t = plan.tensors[i];
            if (t.placement == CERE_PLACEMENT_DISCARD) continue;
            m->slot_of[i] = int64_t(m->slots.size());
            Slot s;
            s.name = t.name;
            s.bytes = t.bytes;
            // + 256: the kernels' vector loads may read past a tensor's end
            if (const cere_status st = ctx_->allocate(t.bytes + 256, s.buf, false); st != CERE_OK) return st;
            m->slots.push_back(std::move(s));
            if (m->slots.back().buf.map == nullptr) return CERE_ERR_UNSUPPORTED;  // UMA only
            std::memset(static_cast<uint8_t *>(m->slots.back().buf.map) + t.bytes, 0, 256);
        }
        *out = std::move(m);
        return CERE_OK;
    }

    cere_status write(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                      uint64_t len) override {
        auto &m = static_cast<InklingModel &>(model);
        if (tensor_index >= m.slot_of.size() || m.slot_of[tensor_index] < 0 || bytes == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        const Slot &slot = m.slots[size_t(m.slot_of[tensor_index])];
        if (offset > slot.bytes || len > slot.bytes - offset) return CERE_ERR_ARGUMENT;
        if (ctx_->faulted()) return CERE_ERR_INTERNAL;
        std::memcpy(static_cast<uint8_t *>(slot.buf.map) + offset, bytes, len);
        return CERE_OK;
    }

    uint32_t moe_lever() const override { return CERE_LEVER_MOE_P3B; }

    cere_status set_levers(host::DeviceModel &, const host::Levers &) override {
        // One implementation per slot; api.cpp refused every non-default choice by name (inkling2_refusal).
        return CERE_OK;
    }

    cere_status instantiate(host::DeviceModel &model) override {
        auto &m = static_cast<InklingModel &>(model);
        if (m.instantiated) return CERE_ERR_ARGUMENT;
        const Find find = [&m](const std::string &name, TensorView *out) {
            for (const Slot &s : m.slots) {
                if (s.name != name) continue;
                *out = TensorView{s.buf.addr, s.buf.map, s.bytes};
                return true;
            }
            return false;
        };
        std::string why;
        if (const cere_status st = m.forward.init(find, options(), &why); st != CERE_OK) {
            std::fprintf(stderr, "cere: inkling vulkan forward: %s %s\n", why.c_str(), ctx_->last_error().c_str());
            return st;
        }
        m.instantiated = true;
        return CERE_OK;
    }

    bool forward_supported() const override { return true; }
    bool sampler_supported() const override { return false; }

    cere_status create_session(host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                               std::unique_ptr<host::DeviceSession> *out) override {
        if (lazy || max_context_tokens == 0 || max_context_tokens > kMaxContext) return CERE_ERR_UNSUPPORTED;
        auto &m = static_cast<InklingModel &>(model);
        if (!m.instantiated) return CERE_ERR_ARGUMENT;
        auto session = std::make_unique<InklingSession>();
        if (const cere_status st = m.forward.create_session(uint32_t(max_context_tokens), &session->s); st != CERE_OK) {
            return st;
        }
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
        if (tokens == nullptr || count == 0 || consumed == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
        if (slice != nullptr) slice->paused = false;
        auto &m = static_cast<InklingModel &>(model);
        auto &s = static_cast<InklingSession &>(session);
        if (s.broken) return CERE_ERR_INTERNAL;
        uint64_t n = std::min<uint64_t>(count, kPrefillRows);
        if (limit != 0) n = std::min(n, limit);
        for (uint64_t i = 0; i < n; ++i)
            if (tokens[i] >= kVocab) return CERE_ERR_ARGUMENT;
        if (s.pos + n > s.s->max_ctx()) return CERE_ERR_CAPACITY;
        cere_token_score score{};
        if (const cere_status st = run(m, s, tokens, uint32_t(n), n == count ? &score : nullptr); st != CERE_OK) {
            return st;
        }
        *consumed = n;
        if (n == count) *out = score;
        return CERE_OK;
    }

    cere_status step_rows(host::DeviceModel &model, const host::RowArgs *rows, uint32_t row_count,
                          cere_token_score *out, cere_status *row_status) override {
        if (rows == nullptr || row_count == 0 || row_count > kMaxSessions || out == nullptr || row_status == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        std::set<const host::DeviceSession *> seen;
        for (uint32_t r = 0; r < row_count; ++r) {
            if (rows[r].session == nullptr || !seen.insert(rows[r].session).second) return CERE_ERR_ARGUMENT;
            if (rows[r].logprobs != nullptr && rows[r].sampling == nullptr) return CERE_ERR_ARGUMENT;
        }
        auto &m = static_cast<InklingModel &>(model);
        // Every participant runs the 1-row step in row order; a failure after the first launch breaks every
        // participant (host/device.h step_rows contract), including the rows that already ran.
        std::vector<uint32_t> part;
        for (uint32_t r = 0; r < row_count; ++r) {
            auto *s = static_cast<InklingSession *>(rows[r].session);
            if (rows[r].sampling != nullptr || rows[r].logprobs != nullptr) {
                row_status[r] = CERE_ERR_UNSUPPORTED;
            } else if (s->broken) {
                row_status[r] = CERE_ERR_INTERNAL;
            } else if (s->pos + 1 > s->s->max_ctx()) {
                row_status[r] = CERE_ERR_CAPACITY;
            } else if (rows[r].token >= kVocab) {
                row_status[r] = CERE_ERR_ARGUMENT;
            } else {
                row_status[r] = CERE_OK;
                part.push_back(r);
            }
        }
        for (const uint32_t r : part) {
            const cere_status st = run(m, *static_cast<InklingSession *>(rows[r].session), &rows[r].token, 1, &out[r]);
            if (st == CERE_OK) continue;
            for (const uint32_t p : part) {
                static_cast<InklingSession *>(rows[p].session)->broken = true;
                row_status[p] = st;
            }
            return st;
        }
        return CERE_OK;
    }

    cere_status set_rope_delta(host::DeviceModel &, host::DeviceSession &, int64_t rope_delta) override {
        return rope_delta == 0 ? CERE_OK : CERE_ERR_UNSUPPORTED;
    }

private:
    InklingDevice() = default;

    Forward::Options options() const {
        Forward::Options o;
        o.max_rows = uint32_t(kPrefillRows);
        o.logit_rows = 1;
        o.k2 = k2_;
        return o;
    }

    // n tokens of one session at its position; scores the last row into *score when non-null.
    cere_status run(InklingModel &m, InklingSession &s, const uint32_t *tokens, uint32_t n, cere_token_score *score) {
        cere_status st = m.forward.step(*s.s, tokens, n, uint32_t(s.pos));
        if (st == CERE_OK && score != nullptr) {
            st = m.forward.logits(n - 1, 1);
            if (st == CERE_OK) *score = top2(m.forward.logits_host());
        }
        if (st != CERE_OK) {
            s.broken = true;
            return st;
        }
        s.pos += n;
        return CERE_OK;
    }

    std::unique_ptr<vk::Context> ctx_;
    bool k2_ = false;
    mutable std::atomic<bool> poisoned_reported_{false};
};

cere_status InklingDevice::open(const char *device_id, std::unique_ptr<host::Device> *out) {
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
    size_t shaders = 0;
    vk::embedded_shaders(&shaders);
    if (shaders == 0) {
        std::fprintf(stderr, "cere: vk_open refused: the library was built without its embedded SPIR-V table\n");
        return CERE_ERR_UNSUPPORTED;
    }
    std::unique_ptr<InklingDevice> d(new (std::nothrow) InklingDevice());
    if (d == nullptr) return CERE_ERR_CAPACITY;
    std::string why;
    vk::CreateOptions copt;
    copt.max_in_flight = kSubmitsInFlight;  // the forward's pipelined submits (ki_model.h)
    if (const cere_status st = vk::Context::create(uint32_t(std::stoul(digits)), &d->ctx_, &why, copt); st != CERE_OK) {
        std::fprintf(stderr, "cere: vk_open refused ordinal=%s status=%d: %s\n", digits.c_str(), int(st), why.c_str());
        return st;
    }
    d->k2_ = d->ctx_->info().coopmat;
    *out = std::move(d);
    return CERE_OK;
}

}  // namespace
}  // namespace cere::vkink

namespace cere::host {

// Text only on Vulkan (the vision tower runs in ki_vision's harness, not in this device): no entry for `vision`.
const Support *inkling2_support(const Device &, bool vision) { return vision ? nullptr : &cere::vkink::inkling_support(); }

cere_status create_device(const char *device_id, std::unique_ptr<Device> *out) {
    return cere::vkink::InklingDevice::open(device_id, out);
}

}  // namespace cere::host
