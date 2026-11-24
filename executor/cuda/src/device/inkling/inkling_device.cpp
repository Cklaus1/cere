// Inkling 2 Mini (inkling design note S0, S5, S6g): the CUDA `Device` (host/device.h) of an inkling2_loop library.
// It opens the GPU as every CERE CUDA device does (the decimal CUDA ordinal among the visible devices, compute
// capability 12.0, eager module loading) and runs the text forward of model/inkling/forward.h: eager, checkpoint
// tensors as stored (BF16, FP32, NVFP4 / MXFP8 experts), up to 64 sessions. Each session owns its state (forward.h
// Session: 40 sliding KV rings of 1,024 slots, the 8 full visits' KV for its whole context, the 48 visits' conv
// states; D-I3, D-I4) and its sampler output counts, allocated at create and admitted by KV bytes: the open sessions'
// contexts stay within the plan's kv_pool_tokens (S10, section 9.2). `step` prefills a prompt in chunks of up to
// 4,096 tokens and decodes a single token with the decode path, so a session's step and a step_rows row run the same
// kernels and are bit-identical (G4). The scored row's softcapped FP32 logits go to the greedy top-2 (sampling null)
// or to the sampler (S6g): process_logits (model/inkling/logits.h, K12's D2 on FP32) then K12's top-2,
// probabilities, keep mask, seeded draw, finalize, top logprobs and output counts (kernels/k12_sampler.h), over the
// 200,058 unpadded ids (the padded ids are -inf in the reference: never drawn, never in a top list). A model planned
// with the vision tower (inkling2_support(device, true): its `model.visual.*` tensors on the device) takes images in
// a PREFILL (S6g): each prompt chunk encodes every image whose placeholder rows it holds (model/inkling/vision.h, one
// image per pass) and splices those rows over the placeholders (forward.h prefill). The MTP drafter (S9), the prefix
// cache, snapshots and vocab masks (grammar) are refused (api.cpp plan_inkling2_model, host/family.h
// inkling2_refusal, D-I5); every other optional call keeps the Device defaults (UNSUPPORTED).
#include "host/device.h"
#include "host/mrope.h"
#include "host/support.h"
#include "kernels/k12_sampler.h"
#include "model/inkling/drafter.h"
#include "model/inkling/forward.h"
#include "model/inkling/logits.h"
#include "model/inkling/verify.h"
#include "model/inkling/vision.h"
#include "model/inkling/weights.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace cere::inkling {

namespace {

constexpr size_t kMaxOrdinalDigits = 9;
constexpr uint64_t kAlignBytes = 256;
constexpr uint64_t kMaxSessions = uint64_t(kMaxDecodeRows);       // one decode call holds every session
constexpr uint64_t kMaxContextTokens = uint64_t(1) << 20;          // config max_position_embeddings

uint64_t align_up(uint64_t x) { return (x + kAlignBytes - 1) / kAlignBytes * kAlignBytes; }

cere_status alloc_status(cudaError_t e) {
    if (e == cudaSuccess) return CERE_OK;
    cudaGetLastError();
    return e == cudaErrorMemoryAllocation ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
}

namespace iv = cere::model::inkling_vision;
// api.cpp's span checks and this file's placeholder counts (cere::mrope::image_tokens) use the tower's merge size.
static_assert(cere::mrope::kSpatialMergeSize == uint32_t(cere::inkling::vision::kMerge));

// The vision tower the library runs, as cere-runtime maps config.json's vision_config (upload.rs
// inkling_vision_config; Inkling2VisionConfig defaults where the checkpoint is silent, kernels/inkling/vision.h):
// one image = 1 x h x w patches of 3 x 20 x 20 values (temporal 1: images only), one placeholder `<|unused_200054|>`
// per 2 x 2 patches, each image's placeholders right after `<|content_image|>` (200005, the template's) and before
// `<|end_message|>` (200010); no video token, no deepstack; the erf GELU.
cere_qwen4_exp_vision_config make_vision_config() {
    namespace kv = cere::inkling::vision;
    cere_qwen4_exp_vision_config v{};
    v.struct_size = sizeof(v);
    v.abi_version = CERE_EXECUTOR_ABI_VERSION;
    v.depth = kv::kLayers;
    v.hidden_size = kv::kHidden;
    v.intermediate_size = kv::kInter;
    v.num_heads = kv::kHeads;
    v.patch_size = kv::kPatch;
    v.temporal_patch_size = 1;
    v.spatial_merge_size = kv::kMerge;
    v.in_channels = 3;
    v.out_hidden_size = kv::kOutHidden;
    v.num_position_embeddings = kv::kPosSide * kv::kPosSide;
    v.deepstack_count = 0;
    v.hidden_act = CERE_ACT_GELU_ERF;
    v.image_token_id = kImageToken;
    v.video_token_id = 0;
    v.vision_start_token_id = 200005;
    v.vision_end_token_id = 200010;
    v.layer_norm_eps = 1e-5;
    return v;
}

// The text model, and the same with the vision tower (identical bounds, so the two never drift apart).
const host::Support &inkling_support(bool vision) {
    static const host::Support text{
        "inkling-2-mini",
        cere_qwen4_exp_config{},  // unused: plan_inkling2_model checks cere_inkling2_config itself
        kMaxSessions,
        kMaxContextTokens,
        kMaxSessions * kMaxContextTokens,  // host/kv_pool.h bound of kv_pool_tokens: the open sessions' contexts
        host::dtype_bit(CERE_DTYPE_BF16) | host::dtype_bit(CERE_DTYPE_F32) | host::dtype_bit(CERE_DTYPE_F8_E4M3) |
            host::dtype_bit(CERE_DTYPE_U8),
        nullptr,
    };
    static const cere_qwen4_exp_vision_config config = make_vision_config();
    static const host::Support with_vision = [] {
        host::Support s = text;
        s.name = "inkling-2-mini-vision";
        s.vision = &config;
        return s;
    }();
    return vision ? with_vision : text;
}

// Device bytes of the vision tower's buffers (a vision model only), carved in this order: one image's patches (at
// most iv::kMaxPatches rows of 1,200 BF16), its encoded rows, the image rows of one prefill chunk and the encoder's
// scratch for one image.
struct VisionBuffers {
    iv::bf16 *pixels = nullptr;
    iv::bf16 *encoded = nullptr;
    iv::bf16 *rows = nullptr;
    void *scratch = nullptr;
    uint64_t scratch_bytes = 0;
};

uint64_t carve_vision(uint8_t *base, VisionBuffers *out) {
    namespace kv = cere::inkling::vision;
    uint64_t off = 0;
    const auto take = [&](uint64_t bytes) {
        void *p = base == nullptr ? nullptr : base + off;
        off += align_up(bytes);
        return p;
    };
    VisionBuffers v;
    v.pixels = static_cast<iv::bf16 *>(take(uint64_t(iv::kMaxPatches) * kv::kPatchRow * sizeof(iv::bf16)));
    v.encoded = static_cast<iv::bf16 *>(take(uint64_t(iv::kMaxPatches) / 4 * kv::kOutHidden * sizeof(iv::bf16)));
    v.rows = static_cast<iv::bf16 *>(take(uint64_t(kMaxPrefillRows) * kv::kOutHidden * sizeof(iv::bf16)));
    v.scratch_bytes = iv::scratch_bytes(iv::kMaxPatches, 1);
    v.scratch = take(v.scratch_bytes);
    if (out != nullptr) *out = v;
    return off;
}

uint64_t vision_bytes(const host::ModelPlan &plan) {
    return plan.support != nullptr && plan.support->vision != nullptr ? carve_vision(nullptr, nullptr) : 0;
}

struct DeviceBuffer {
    void *ptr = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() {
        if (ptr != nullptr) cudaFree(ptr);
    }
    cere_status allocate(uint64_t bytes) { return bytes == 0 ? CERE_OK : alloc_status(cudaMalloc(&ptr, bytes)); }
    uint8_t *bytes() const { return static_cast<uint8_t *>(ptr); }
};

struct HostBuffer {
    void *ptr = nullptr;
    HostBuffer() = default;
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer &operator=(const HostBuffer &) = delete;
    ~HostBuffer() {
        if (ptr != nullptr) cudaFreeHost(ptr);
    }
    cere_status allocate(uint64_t bytes) { return alloc_status(cudaMallocHost(&ptr, bytes)); }
};

// One non-DISCARD plan tensor in the arena.
struct Slot {
    TensorRef ref;
    uint64_t bytes = 0;
    uint64_t offset = 0;
};

// The sampler stage's device buffers (S6g), one set for the rows of a call: their stages run one after another on
// the stream. Sizes follow K12's contract over the kUnpaddedVocab ids (kernels/k12_sampler.h).
struct Sampler {
    float *proc = nullptr;                    // [kLogitLd] processed logits of the row being sampled
    float *probs = nullptr;                   // [kLogitLd] probabilities of proc / T
    uint32_t *keep = nullptr;                 // [kMaskWords] the D3 keep mask
    double *lse = nullptr;                    // [1]
    void *scratch = nullptr;                  // k12::scratch_bytes()
    uint32_t *top2_ids = nullptr;             // [2]
    float *top2_scores = nullptr;             // [2]
    uint32_t *drawn = nullptr;                // [1]
    cere_logit_bias *bias = nullptr;          // [kUnpaddedVocab] the row's logit_bias, uploaded per sampled row
    cere_token_logprobs *logprobs = nullptr;  // [kMaxDecodeRows] the D7 outputs before their copy to the host
};

constexpr uint64_t kMaskWords = (uint64_t(kUnpaddedVocab) + 31) / 32;
constexpr uint64_t kCountsBytes = uint64_t(kUnpaddedVocab) * sizeof(uint32_t);  // a session's D5 output counts

// Device bytes besides the tensors and the sessions, carved from `base` in this order (base null: the size only):
// the Forward's workspace, the FP32 logits of one decode call, the scores of its rows and the sampler's buffers.
uint64_t carve_scratch(uint8_t *base, float **logits, cere_token_score **scores, Sampler *sampler) {
    uint64_t off = 0;
    const auto take = [&](uint64_t bytes) {
        void *p = base == nullptr ? nullptr : base + off;
        off += align_up(bytes);
        return p;
    };
    take(Forward::workspace_bytes());
    float *l = static_cast<float *>(take(uint64_t(kMaxDecodeRows) * uint64_t(kLogitLd) * sizeof(float)));
    auto *sc = static_cast<cere_token_score *>(take(uint64_t(kMaxDecodeRows) * sizeof(cere_token_score)));
    Sampler s;
    s.proc = static_cast<float *>(take(uint64_t(kLogitLd) * sizeof(float)));
    s.probs = static_cast<float *>(take(uint64_t(kLogitLd) * sizeof(float)));
    s.keep = static_cast<uint32_t *>(take(kMaskWords * sizeof(uint32_t)));
    s.lse = static_cast<double *>(take(sizeof(double)));
    s.scratch = take(k12::scratch_bytes());
    s.top2_ids = static_cast<uint32_t *>(take(2 * sizeof(uint32_t)));
    s.top2_scores = static_cast<float *>(take(2 * sizeof(float)));
    s.drawn = static_cast<uint32_t *>(take(sizeof(uint32_t)));
    s.bias = static_cast<cere_logit_bias *>(take(uint64_t(kUnpaddedVocab) * sizeof(cere_logit_bias)));
    s.logprobs = static_cast<cere_token_logprobs *>(take(uint64_t(kMaxDecodeRows) * sizeof(cere_token_logprobs)));
    if (logits != nullptr) *logits = l;
    if (scores != nullptr) *scores = sc;
    if (sampler != nullptr) *sampler = s;
    return off;
}

uint64_t scratch_bytes() { return carve_scratch(nullptr, nullptr, nullptr, nullptr); }

// One scored row's sampler request (S6g): `sampling` null = greedy without logprobs. `key` is the position of the
// row's token (D4's draw key) and `counts` its session's D5 output counts.
struct RowSampling {
    const host::ItemSampling *sampling = nullptr;
    bool logprobs = false;
    uint32_t key = 0;
    uint32_t *counts = nullptr;
};

// The plan's device tensors in arena order (offsets 256-aligned), the plan index -> slot map and the arena size.
uint64_t layout(const host::ModelPlan &plan, std::vector<Slot> *slots, std::vector<int64_t> *slot_of) {
    uint64_t off = 0;
    if (slot_of != nullptr) slot_of->assign(plan.tensors.size(), -1);
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        const host::TensorPlan &t = plan.tensors[i];
        if (t.placement == CERE_PLACEMENT_DISCARD) continue;
        if (slots != nullptr) {
            if (slot_of != nullptr) (*slot_of)[i] = int64_t(slots->size());
            Slot s;
            s.ref.name = t.name;
            s.ref.dtype = t.dtype;
            s.ref.ndim = t.ndim;
            for (uint32_t d = 0; d < 4 && d < CERE_TENSOR_MAX_DIMS; ++d) s.ref.shape[d] = t.shape[d];
            s.bytes = t.bytes;
            s.offset = off;
            slots->push_back(std::move(s));
        }
        off += align_up(t.bytes);
    }
    return off;
}

// Device bytes of a model with drafts beyond the plain model's (estimate, allocate, create_session): the verify
// buffers, the drafter's workspace and proposals, and every session's drafter state (its fixed part per session, the
// full layer's context per pool token like the target's full visits). 0 without drafts.
uint64_t draft_bytes(const host::ModelPlan &plan) {
    if (plan.speculative_draft_tokens == 0) return 0;
    return align_up(Forward::verify_bytes()) + align_up(Drafter::workspace_bytes()) +
           align_up(uint64_t(kMaxDraftSessions) * kMaxBlockDrafts * sizeof(uint32_t)) +
           plan.max_sessions * draft_session_bytes(0) +
           (draft_session_bytes(int64_t(plan.kv_pool_tokens)) - draft_session_bytes(0));
}

// The host arrays of one drafter call (extend): per context row its session and position, whether it is the
// session's last (f_rnn); per draft block its session index, anchor and start position (the first kMaxDraftSessions
// entries of sess and pos).
struct DraftCall {
    const DraftSession *sess[kMaxDraftCtxRows];
    int64_t pos[kMaxDraftCtxRows];
    bool frnn[kMaxDraftCtxRows];
    int64_t block[kMaxDraftSessions];
    uint32_t anchor[kMaxDraftSessions];
};

class InklingModel final : public host::DeviceModel {
public:
    std::vector<Slot> slots;
    std::vector<int64_t> slot_of;
    DeviceBuffer arena;
    DeviceBuffer scratch;
    HostBuffer scores_host;    // cere_token_score [kMaxDecodeRows], pinned
    HostBuffer logprobs_host;  // cere_token_logprobs [kMaxDecodeRows], pinned
    float *logits = nullptr;
    cere_token_score *scores = nullptr;
    Sampler sampler;
    std::vector<cere_logit_bias> bias;  // a sampled row's logit_bias below kUnpaddedVocab (host staging)
    Weights weights;
    Forward forward;
    // A vision model's tower (S6g): its buffers (carve_vision), `model.visual.*` tensors and K2 plans.
    DeviceBuffer vision_mem;
    VisionBuffers vision;
    iv::Weights vision_weights;
    iv::Plans vision_plans;
    bool has_vision = false;
    // S9 phase B (D-I-S9B): a model with drafts (window = plan.speculative_draft_tokens, the verify rows of a round:
    // the unfed token + window - 1 drafts) carves the verify buffers (attach_verify), the drafter's workspace and
    // its proposals (device ids + their pinned copy) and binds the MTP drafter's tensors at instantiate.
    uint32_t window = 0;
    DeviceBuffer verify_mem;
    DeviceBuffer draft_ws;
    DeviceBuffer draft_ids;    // uint32 [kMaxDraftSessions][kMaxBlockDrafts]
    HostBuffer draft_ids_host;
    DraftWeights draft_weights;
    Drafter drafter;
    std::unique_ptr<DraftCall> call;
    bool instantiated = false;
    uint64_t kv_pool_tokens = 0;  // plan.kv_pool_tokens: the open sessions' contexts never exceed it (S10, R-I4)
    uint64_t kv_tokens = 0;       // the open sessions' contexts
};

// The 422 `model.visual.*` tensors by checkpoint name, each BF16 of the tower's shape (vision.h tensor_shape).
bool bind_vision(const std::vector<Slot> &slots, iv::Weights *out, std::string *error) {
    std::map<std::string, const TensorRef *> by_name;
    for (const Slot &s : slots) {
        if (s.ref.name.rfind("model.visual.", 0) == 0) by_name.emplace(s.ref.name, &s.ref);
    }
    for (uint32_t i = 0; i < iv::kTensorCount; ++i) {
        const std::string name = iv::tensor_name(i);
        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            *error = name + ": missing";
            return false;
        }
        uint64_t shape[2] = {0, 0};
        const uint32_t ndim = iv::tensor_shape(i, shape);
        const TensorRef &t = *it->second;
        if (t.dtype != CERE_DTYPE_BF16 || t.ndim != ndim || t.shape[0] != shape[0] ||
            (ndim == 2 && t.shape[1] != shape[1])) {
            *error = name + ": dtype or shape differs from the tower's";
            return false;
        }
        out->tensor[i] = static_cast<const iv::bf16 *>(t.ptr);
        by_name.erase(it);
    }
    if (!by_name.empty()) {
        *error = by_name.begin()->first + ": not read by the tower";
        return false;
    }
    return true;
}

// Positions [lo, hi) of image `im`'s placeholders that a call at positions [pos, pos + n) holds (lo >= hi: none). A
// session's image prompt starts at position 0 (api.cpp: a PREFILL of an empty session), so token_offset is a position.
std::pair<uint64_t, uint64_t> held_rows(const host::SessionImage &im, int64_t pos, int64_t n) {
    const uint64_t first = im.span.token_offset;
    const uint64_t last = first + cere::mrope::image_tokens(im.span);
    return {std::max(first, uint64_t(pos)), std::min(last, uint64_t(pos + n))};
}

// An image prompt's call (S6g), before any launch: every image is one frame (t 1) of an even h x w grid of at most
// iv::kMaxPatches patches with pixels BF16 [h * w, 1,200] (the processor's merge-block order) and its placeholders
// inside the prompt, and the call's kImageToken rows are exactly the placeholders it holds (forward.h prefill
// splices the tower's rows over them in order). CERE_ERR_ARGUMENT otherwise.
cere_status check_images(const host::StepImages &images, int64_t pos, const uint32_t *tokens, int64_t n) {
    namespace kv = cere::inkling::vision;
    if (images.image == nullptr || images.count == 0 || images.prompt_len < uint64_t(pos + n)) {
        return CERE_ERR_ARGUMENT;
    }
    uint64_t held = 0;
    for (uint64_t i = 0; i < images.count; ++i) {
        const host::SessionImage &im = images.image[i];
        const uint64_t patches = uint64_t(im.span.grid_h) * im.span.grid_w;
        if (im.span.grid_t != 1 || im.span.grid_h < 2 || im.span.grid_w < 2 || im.span.grid_h % 2 != 0 ||
            im.span.grid_w % 2 != 0 || patches > iv::kMaxPatches ||
            im.pixels.size() != patches * kv::kPatchRow * sizeof(iv::bf16) ||
            im.span.token_offset + patches / 4 > images.prompt_len) {
            return CERE_ERR_ARGUMENT;
        }
        const auto [lo, hi] = held_rows(im, pos, n);
        for (uint64_t r = lo; r < hi; ++r) {
            if (tokens[r - uint64_t(pos)] != kImageToken) return CERE_ERR_ARGUMENT;
        }
        if (lo < hi) held += hi - lo;
    }
    if (uint64_t(std::count(tokens, tokens + n, kImageToken)) != held) return CERE_ERR_ARGUMENT;
    return CERE_OK;
}

class InklingSession final : public host::DeviceSession {
public:
    ~InklingSession() override {
        if (pool != nullptr) *pool -= ctx;
    }
    DeviceBuffer state;
    DeviceBuffer counts;  // uint32 [kUnpaddedVocab]: D5 output counts of the session's sampled steps
    Session s;
    int64_t pos = 0;      // positions [0, pos) are in the session's state
    // S9 phase B (D-I-S9B): a drafting session's drafter state (set_session_draft) and its pending proposals, the
    // drafts of the block anchored at the last committed token (host copies).
    DeviceBuffer draft_state;
    DraftSession ds;
    bool drafting = false;
    uint32_t pending[kMaxBlockDrafts] = {};
    uint32_t pending_count = 0;
    bool broken = false;  // a failure after the first launch of a call: every later step is CERE_ERR_INTERNAL
    // The model's kv_tokens, which counts this session's context `ctx` until its release.
    uint64_t *pool = nullptr;
    uint64_t ctx = 0;
};

class InklingDevice final : public host::Device {
public:
    static cere_status open(const char *device_id, std::unique_ptr<host::Device> *out);
    ~InklingDevice() override {
        cudaSetDevice(ordinal_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
        if (handle_ != nullptr) cublasLtDestroy(handle_);
    }

    std::vector<const host::Support *> supported() const override { return {}; }
    uint64_t host_budget_bytes() const override { return 0; }

    // Admission by KV bytes (inkling design note section 9.2): the arena, the scratch (with the sampler's buffers),
    // the vision buffers of an image model, every session's fixed part (sliding rings, conv states, D5 output counts)
    // plus the full visits' KV of kv_pool_tokens tokens, shared by the open sessions (create_session).
    host::MemoryEstimate estimate(const host::ModelPlan &plan) const override {
        return {layout(plan, nullptr, nullptr) + scratch_bytes() + vision_bytes(plan) +
                    plan.max_sessions * (session_bytes(0) + align_up(kCountsBytes)) +
                    (session_bytes(int64_t(plan.kv_pool_tokens)) - session_bytes(0)) + draft_bytes(plan),
                0};
    }

    cere_status allocate(const host::ModelPlan &plan, std::unique_ptr<host::DeviceModel> *out) override {
        if (plan.max_sessions > kMaxSessions || plan.max_context_tokens > kMaxContextTokens ||
            plan.speculative_draft_tokens == 1 || plan.speculative_draft_tokens > kMaxBlockDrafts + 1) {
            return CERE_ERR_UNSUPPORTED;
        }
        if (const cere_status st = current(); st != CERE_OK) return st;
        auto model = std::make_unique<InklingModel>();
        const uint64_t arena = layout(plan, &model->slots, &model->slot_of);
        if (const cere_status st = model->arena.allocate(arena); st != CERE_OK) return st;
        if (const cere_status st = model->scratch.allocate(scratch_bytes()); st != CERE_OK) return st;
        if (const cere_status st = model->scores_host.allocate(kMaxDecodeRows * sizeof(cere_token_score));
            st != CERE_OK) {
            return st;
        }
        if (const cere_status st = model->logprobs_host.allocate(kMaxDecodeRows * sizeof(cere_token_logprobs));
            st != CERE_OK) {
            return st;
        }
        carve_scratch(model->scratch.bytes(), &model->logits, &model->scores, &model->sampler);
        if (vision_bytes(plan) != 0) {
            if (const cere_status st = model->vision_mem.allocate(vision_bytes(plan)); st != CERE_OK) return st;
            carve_vision(model->vision_mem.bytes(), &model->vision);
            model->has_vision = true;
        }
        if (plan.speculative_draft_tokens != 0) {
            const uint64_t ids = uint64_t(kMaxDraftSessions) * kMaxBlockDrafts * sizeof(uint32_t);
            if (const cere_status st = model->verify_mem.allocate(Forward::verify_bytes()); st != CERE_OK) return st;
            if (const cere_status st = model->draft_ws.allocate(Drafter::workspace_bytes()); st != CERE_OK) return st;
            if (const cere_status st = model->draft_ids.allocate(ids); st != CERE_OK) return st;
            if (const cere_status st = model->draft_ids_host.allocate(ids); st != CERE_OK) return st;
            model->window = plan.speculative_draft_tokens;
            model->call = std::make_unique<DraftCall>();
        }
        model->bias.reserve(size_t(kUnpaddedVocab));
        for (Slot &s : model->slots) s.ref.ptr = model->arena.bytes() + s.offset;
        model->kv_pool_tokens = plan.kv_pool_tokens;
        *out = std::move(model);
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
        if (const cere_status st = current(); st != CERE_OK) return st;
        const cudaError_t e = cudaMemcpy(m.arena.bytes() + slot.offset + offset, bytes, len, cudaMemcpyHostToDevice);
        return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
    }

    uint32_t moe_lever() const override { return CERE_LEVER_MOE_P3B; }

    cere_status set_levers(host::DeviceModel &, const host::Levers &) override {
        // One implementation per slot; api.cpp refused every non-default choice by name (inkling2_refusal).
        return CERE_OK;
    }

    cere_status instantiate(host::DeviceModel &model) override {
        auto &m = static_cast<InklingModel &>(model);
        if (m.instantiated) return CERE_ERR_ARGUMENT;
        if (const cere_status st = current(); st != CERE_OK) return st;
        std::vector<TensorRef> refs;
        refs.reserve(m.slots.size());
        for (const Slot &s : m.slots) refs.push_back(s.ref);
        std::string error;
        if (!bind_weights(refs, &m.weights, &error)) {
            std::fprintf(stderr, "cere: inkling weights: %s\n", error.c_str());
            return CERE_ERR_ARGUMENT;
        }
        if (m.has_vision && !bind_vision(m.slots, &m.vision_weights, &error)) {
            std::fprintf(stderr, "cere: inkling vision weights: %s\n", error.c_str());
            return CERE_ERR_ARGUMENT;
        }
        if (const cere_status st = m.forward.init(&m.weights, m.scratch.bytes(), Forward::workspace_bytes(), stream_);
            st != CERE_OK) {
            return st;
        }
        if (m.has_vision) {
            if (const cere_status st = iv::init_plans(&m.vision_plans, handle_, iv::kMaxPatches); st != CERE_OK) {
                return st;
            }
        }
        if (m.window != 0) {
            if (!bind_draft_weights(refs, &m.draft_weights, &error)) {
                std::fprintf(stderr, "cere: inkling drafter weights: %s\n", error.c_str());
                return CERE_ERR_ARGUMENT;
            }
            if (const cere_status st = m.forward.attach_verify(m.verify_mem.ptr, Forward::verify_bytes());
                st != CERE_OK) {
                return st;
            }
            if (const cere_status st = m.drafter.init(&m.draft_weights, &m.weights, m.draft_ws.ptr,
                                                      Drafter::workspace_bytes(), handle_, stream_);
                st != CERE_OK) {
                return st;
            }
        }
        m.instantiated = true;
        return CERE_OK;
    }

    bool forward_supported() const override { return true; }
    bool sampler_supported() const override { return true; }

    // CERE_ERR_CAPACITY, before allocating, when the open sessions' contexts plus this one exceed kv_pool_tokens
    // (cered queues such a create, ledger CR1); the session's release returns its context to the pool.
    cere_status create_session(host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                               std::unique_ptr<host::DeviceSession> *out) override {
        if (lazy || max_context_tokens == 0 || max_context_tokens > kMaxContextTokens) return CERE_ERR_UNSUPPORTED;
        auto &m = static_cast<InklingModel &>(model);
        if (!m.instantiated) return CERE_ERR_ARGUMENT;
        if (max_context_tokens > m.kv_pool_tokens - m.kv_tokens) return CERE_ERR_CAPACITY;
        if (const cere_status st = current(); st != CERE_OK) return st;
        auto session = std::make_unique<InklingSession>();
        const int64_t ctx = int64_t(max_context_tokens);
        if (const cere_status st = session->state.allocate(session_bytes(ctx)); st != CERE_OK) return st;
        if (const cere_status st = session->counts.allocate(kCountsBytes); st != CERE_OK) return st;
        session->s = carve_session(session->state.ptr, ctx);
        if (m.window != 0) {
            if (const cere_status st = session->draft_state.allocate(draft_session_bytes(ctx)); st != CERE_OK) {
                return st;
            }
            session->ds = carve_draft_session(session->draft_state.ptr, ctx);
        }
        if (reset_session(session->s, stream_) != cudaSuccess ||
            cudaMemsetAsync(session->counts.ptr, 0, kCountsBytes, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        m.kv_tokens += max_context_tokens;
        session->pool = &m.kv_tokens;
        session->ctx = max_context_tokens;
        *out = std::move(session);
        return CERE_OK;
    }

    cere_status step(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *tokens, uint64_t count,
                     uint64_t limit, const host::StepImages *images, uint64_t *consumed,
                     const host::ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                     const host::EndStep *end, host::StepSlice *slice) override {
        if (end != nullptr) return CERE_ERR_UNSUPPORTED;
        if (sampling != nullptr && sampling->vocab_mask != nullptr) return CERE_ERR_UNSUPPORTED;  // grammar (D-I5)
        if (tokens == nullptr || count == 0 || consumed == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
        if (logprobs != nullptr && sampling == nullptr) return CERE_ERR_ARGUMENT;
        if (slice != nullptr) slice->paused = false;
        auto &m = static_cast<InklingModel &>(model);
        auto &s = static_cast<InklingSession &>(session);
        if (s.broken) return CERE_ERR_INTERNAL;
        if (images != nullptr && !m.has_vision) return CERE_ERR_UNSUPPORTED;
        uint64_t n = std::min<uint64_t>(count, uint64_t(kMaxPrefillRows));
        if (limit != 0) n = std::min(n, limit);
        for (uint64_t i = 0; i < n; ++i) {
            if (tokens[i] >= uint32_t(kUnpaddedVocab)) return CERE_ERR_ARGUMENT;
        }
        if (uint64_t(s.pos) + n > uint64_t(s.s.max_ctx)) return CERE_ERR_CAPACITY;
        if (images != nullptr) {
            if (const cere_status st = check_images(*images, s.pos, tokens, int64_t(n)); st != CERE_OK) return st;
        }
        if (const cere_status st = current(); st != CERE_OK) return st;
        const RowSampling rs{sampling, logprobs != nullptr, uint32_t(s.pos + int64_t(n) - 1),
                             static_cast<uint32_t *>(s.counts.ptr)};
        InklingSession *one[1] = {&s};
        // A drafting session's rows also write their drafter taps (D-I-S9B: the prompt's context, then the first
        // block's proposals once the prompt's last row is scored).
        Taps dt;
        dt.draft = m.forward.draft_taps();
        const Taps *taps = s.drafting ? &dt : nullptr;
        // An image prompt's chunks all take the prefill path (a 1-row chunk too: its row may be a placeholder).
        cere_status st = n == 1 && images == nullptr
                             ? decode(m, one, 1, tokens, &rs, taps)
                             : prefill(m, s, tokens, int64_t(n), n == count, rs, images, taps);
        const auto *scores = static_cast<const cere_token_score *>(m.scores_host.ptr);
        if (st == CERE_OK && s.drafting) {
            const int64_t first = s.pos - int64_t(n), rows = int64_t(n);
            const uint32_t anchor = scores[0].id;
            st = extend(m, one, &first, &rows, 1, m.forward.draft_taps(), n == count ? &anchor : nullptr);
        }
        if (st != CERE_OK) {
            s.broken = true;
            return st;
        }
        *consumed = n;
        if (n == count) {
            *out = scores[0];
            if (logprobs != nullptr) *logprobs = staged_logprobs(m, 0, out->id, sampling->top_logprobs);
        }
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
        InklingSession *part[kMaxDecodeRows];
        uint32_t tok[kMaxDecodeRows];
        RowSampling rs[kMaxDecodeRows];
        uint32_t idx[kMaxDecodeRows];
        int64_t k = 0;
        for (uint32_t r = 0; r < row_count; ++r) {
            auto *s = static_cast<InklingSession *>(rows[r].session);
            if (rows[r].sampling != nullptr && rows[r].sampling->vocab_mask != nullptr) {
                row_status[r] = CERE_ERR_UNSUPPORTED;
            } else if (s->broken) {
                row_status[r] = CERE_ERR_INTERNAL;
            } else if (s->pos + 1 > s->s.max_ctx) {
                row_status[r] = CERE_ERR_CAPACITY;
            } else if (rows[r].token >= uint32_t(kUnpaddedVocab)) {
                row_status[r] = CERE_ERR_ARGUMENT;
            } else {
                row_status[r] = CERE_OK;
                part[k] = s;
                tok[k] = rows[r].token;
                rs[k] = RowSampling{rows[r].sampling, rows[r].logprobs != nullptr, uint32_t(s->pos),
                                    static_cast<uint32_t *>(s->counts.ptr)};
                idx[k] = r;
                ++k;
            }
        }
        if (k == 0) return CERE_OK;
        if (const cere_status st = current(); st != CERE_OK) return st;
        const cere_status st = decode(m, part, k, tok, rs);
        if (st != CERE_OK) {
            for (int64_t i = 0; i < k; ++i) {
                part[i]->broken = true;
                row_status[idx[i]] = st;
            }
            return st;
        }
        const auto *scores = static_cast<const cere_token_score *>(m.scores_host.ptr);
        for (int64_t i = 0; i < k; ++i) {
            out[idx[i]] = scores[i];
            if (rows[idx[i]].logprobs != nullptr) {
                *rows[idx[i]].logprobs = staged_logprobs(m, i, scores[i].id, rows[idx[i]].sampling->top_logprobs);
            }
        }
        return CERE_OK;
    }

    // Inkling has no rotary positions (relative position bias only, REF): an image prompt's mRoPE delta, set by the
    // host for every PREFILL (api.cpp), changes nothing.
    cere_status set_rope_delta(host::DeviceModel &, host::DeviceSession &, int64_t) override { return CERE_OK; }

    // S9 phase B (inkling design note D-I-S9B): a model created with a draft window (2..8) runs the MTP drafter; a
    // drafting session's prompt steps write the drafter's context and its last one drafts the first block, then every
    // speculate_rows round verifies, accepts the target's own greedy chain, commits and drafts the next block.
    bool draft_supported(const host::DeviceModel &model) const override {
        return static_cast<const InklingModel &>(model).window != 0;
    }

    // Set before the session's first step (CERE_ERR_ARGUMENT after it).
    cere_status set_session_draft(host::DeviceModel &model, host::DeviceSession &session, bool on) override {
        const auto &m = static_cast<const InklingModel &>(model);
        auto &s = static_cast<InklingSession &>(session);
        if (on && m.window == 0) return CERE_ERR_UNSUPPORTED;
        if (s.pos != 0) return CERE_ERR_ARGUMENT;
        s.drafting = on;
        s.pending_count = 0;
        return CERE_OK;
    }

    // The window's drafts (the verify rows of a round less the unfed token); the interface default without drafts.
    uint32_t max_chain_drafts(const host::DeviceModel &model) const override {
        const uint32_t window = static_cast<const InklingModel &>(model).window;
        return window != 0 ? window - 1 : 3;
    }

    uint32_t pending_drafts(const host::DeviceSession &session, uint32_t *ids, uint32_t max) const override {
        const auto &s = static_cast<const InklingSession &>(session);
        const uint32_t n = s.drafting ? std::min(max, s.pending_count) : 0;
        if (ids != nullptr) std::copy(s.pending, s.pending + n, ids);
        return n;
    }

    // One round per row (host/device.h): row r's run is its unfed token then n drafts (the caller's, else its pending
    // proposals; none for a sampled row, D-I-S9B; at most max_commit - 1). The runs verify together in calls of at
    // most kVerifyRows rows (Forward::verify: every row's logits are the M = 1 decode's), each row is scored as plain
    // decode scores it, the session keeps the greedy-accepted chain plus the bonus token (cut after a stop token and
    // at max_commit), commits it and drafts the next block (extend).
    cere_status speculate_rows(host::DeviceModel &model, const host::SpecRowArgs *rows, uint32_t row_count,
                               host::SpecRowResult *out, cere_status *row_status) override {
        if (rows == nullptr || row_count == 0 || row_count > kMaxSessions || out == nullptr || row_status == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        std::set<const host::DeviceSession *> seen;
        for (uint32_t r = 0; r < row_count; ++r) {
            if (rows[r].session == nullptr || !seen.insert(rows[r].session).second) return CERE_ERR_ARGUMENT;
            if (rows[r].logprobs != nullptr && rows[r].sampling == nullptr) return CERE_ERR_ARGUMENT;
        }
        auto &m = static_cast<InklingModel &>(model);
        if (m.window == 0) return CERE_ERR_UNSUPPORTED;
        Round rd;
        for (uint32_t r = 0; r < row_count; ++r) {
            const host::SpecRowArgs &a = rows[r];
            auto *s = static_cast<InklingSession *>(a.session);
            uint32_t n = a.drafts != nullptr ? a.draft_count : std::min(a.draft_count, s->pending_count);
            const uint32_t *drafts = a.drafts != nullptr ? a.drafts : s->pending;
            if (a.sampling != nullptr && a.sampling->sampled) n = 0;
            if (a.max_commit != 0) n = std::min(n, a.max_commit - 1);
            bool drafts_ok = true;
            for (uint32_t j = 0; j < n; ++j) drafts_ok = drafts_ok && drafts[j] < uint32_t(kUnpaddedVocab);
            cere_status st = CERE_OK;
            if (s->broken) {
                st = CERE_ERR_INTERNAL;
            } else if (a.sampling != nullptr && a.sampling->vocab_mask != nullptr) {
                st = CERE_ERR_UNSUPPORTED;  // grammar (D-I5)
            } else if (!s->drafting || s->pos == 0 || a.token >= uint32_t(kUnpaddedVocab) || a.max_commit == 0 ||
                       a.draft_count > m.window - 1 || (a.drafts != nullptr && a.draft_count == 0) || !drafts_ok) {
                st = CERE_ERR_ARGUMENT;
            } else if (s->pos + int64_t(n) + 1 > s->s.max_ctx) {
                st = CERE_ERR_CAPACITY;
            }
            row_status[r] = st;
            if (st != CERE_OK) continue;
            Run &run = rd.run[rd.k++];
            run.s = s;
            run.r = r;
            run.n = n;
            run.ids[0] = a.token;
            std::copy(drafts, drafts + n, run.ids + 1);
        }
        if (rd.k == 0) return CERE_OK;
        if (const cere_status st = current(); st != CERE_OK) return st;
        for (int64_t first = 0; first < rd.k;) {
            int64_t last = first, total = 0;
            while (last < rd.k && total + int64_t(rd.run[last].n) + 1 <= kVerifyRows) total += rd.run[last++].n + 1;
            if (const cere_status st = verify_group(m, rows, rd.run + first, last - first, out, row_status);
                st != CERE_OK) {
                return st;
            }
            first = last;
        }
        return CERE_OK;
    }

private:
    // One participant of a speculate_rows call: its session, row index, drafts n and run ids (token, drafts).
    struct Run {
        InklingSession *s = nullptr;
        uint32_t r = 0;
        uint32_t n = 0;
        uint32_t ids[kMaxBlockDrafts + 1] = {};
    };
    struct Round {
        Run run[kMaxDecodeRows];
        int64_t k = 0;
    };

    // Verify, score, accept, commit and extend S runs whose rows fit one verify call (speculate_rows). A failure
    // before the commit breaks the runs' sessions with nothing consumed; after it, out holds the committed tokens.
    cere_status verify_group(InklingModel &m, const host::SpecRowArgs *rows, Run *runs, int64_t S,
                             host::SpecRowResult *out, cere_status *row_status) {
        Session ss[kMaxDecodeRows];
        int64_t pos[kMaxDecodeRows], len[kMaxDecodeRows], keep[kMaxDecodeRows];
        uint32_t ids[kVerifyRows], anchors[kMaxDecodeRows];
        RowSampling rs[kVerifyRows];
        InklingSession *part[kMaxDecodeRows];
        int64_t total = 0;
        for (int64_t i = 0; i < S; ++i) {
            const Run &run = runs[i];
            const host::SpecRowArgs &a = rows[run.r];
            part[i] = run.s;
            ss[i] = run.s->s;
            pos[i] = run.s->pos;
            len[i] = int64_t(run.n) + 1;
            for (int64_t j = 0; j < len[i]; ++j, ++total) {
                ids[total] = run.ids[j];
                rs[total] = a.sampling == nullptr ? RowSampling{}
                                                  : RowSampling{a.sampling, a.logprobs != nullptr, uint32_t(pos[i] + j),
                                                                static_cast<uint32_t *>(run.s->counts.ptr)};
            }
        }
        const auto fail = [&](cere_status st) {
            for (int64_t i = 0; i < S; ++i) {
                part[i]->broken = true;
                row_status[runs[i].r] = st;
            }
            return st;
        };
        if (const cere_status st = m.forward.verify(ss, pos, len, S, ids, stream_); st != CERE_OK) return fail(st);
        if (const cere_status st = score(m, 0, total, rs); st != CERE_OK) return fail(st);
        const auto *scores = static_cast<const cere_token_score *>(m.scores_host.ptr);
        int64_t row0 = 0;
        for (int64_t i = 0; i < S; ++i) {
            const Run &run = runs[i];
            const host::SpecRowArgs &a = rows[run.r];
            const cere_token_score *sc = scores + row0;
            // Greedy accept: draft j is kept iff it equals row j's committed token (D5).
            uint32_t accepted = 0;
            while (accepted < run.n && run.ids[accepted + 1] == sc[accepted].id) ++accepted;
            uint32_t committed = accepted + 1;
            for (uint32_t j = 0; j < committed && a.is_stop != nullptr; ++j) {
                if (a.is_stop(a.stop_ctx, sc[j].id)) committed = j + 1;
            }
            committed = std::min(committed, a.max_commit);
            host::SpecRowResult &res = out[run.r];
            res.committed = committed;
            res.accepted = committed - 1;
            for (uint32_t j = 0; j < committed; ++j) {
                res.scores[j] = sc[j];
                if (a.logprobs != nullptr) {
                    a.logprobs[j] = staged_logprobs(m, row0 + j, sc[j].id, a.sampling->top_logprobs);
                }
            }
            keep[i] = committed;
            anchors[i] = sc[committed - 1].id;
            row0 += len[i];
        }
        const bf16 *taps = nullptr;
        cere_status st = m.forward.commit(keep, &taps, stream_);
        for (int64_t i = 0; i < S; ++i) part[i]->pos += keep[i];
        if (st == CERE_OK) st = extend(m, part, pos, keep, S, taps, anchors);
        return st == CERE_OK ? CERE_OK : fail(st);
    }

    InklingDevice(int ordinal, cudaStream_t stream, cublasLtHandle_t handle)
        : ordinal_(ordinal), stream_(stream), handle_(handle) {}

    cere_status current() const { return cudaSetDevice(ordinal_) == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; }

    // The committed token's logprobs of scored row r from the staged entry: the device wrote logprob and the top
    // entries; id, top_count and the unused entries come from here.
    static cere_token_logprobs staged_logprobs(const InklingModel &m, int64_t r, uint32_t id, uint32_t top_count) {
        const cere_token_logprobs &staged = static_cast<const cere_token_logprobs *>(m.logprobs_host.ptr)[r];
        cere_token_logprobs lp{};
        lp.id = id;
        lp.logprob = staged.logprob;
        lp.top_count = top_count;
        for (uint32_t i = 0; i < top_count; ++i) lp.top[i] = staged.top[i];
        return lp;
    }

    // The K12 sampler stage (m27 design note I9, the Qwen forward's sample_row) of scored row r: m.logits row r
    // (softcapped FP32 over the kUnpaddedVocab ids; the padded ids are -inf in the reference, so they carry no
    // mass, are never drawn and never ranked: dropping them, a top_k above kUnpaddedVocab and their logit_bias
    // entries changes nothing). A sampled row: process_logits (D2) -> top2_f32 -> for temperature > 0
    // probabilities, keep_mask (D3) and the seeded draw (D4) -> finalize into m.scores[r]; a logprobs request:
    // top_logprobs (D7) into m.sampler.logprobs[r]; a sampled row then counts its committed token (D5).
    cere_status sample_row(InklingModel &m, int64_t r, const RowSampling &rs) {
        const Sampler &sc = m.sampler;
        const host::ItemSampling &s = *rs.sampling;
        const cere_sampling_params &p = s.params;
        const float *proc = m.logits + r * kLogitLd;
        cere_token_score *score = m.scores + r;
        const auto ok = [](cudaError_t e) { return e == cudaSuccess; };
        const float temperature = s.sampled ? p.temperature : 0.0f;
        if (s.sampled) {
            m.bias.clear();
            for (const cere_logit_bias &b : s.bias) {
                if (b.id < uint32_t(kUnpaddedVocab)) m.bias.push_back(b);
            }
            // Pageable source: the copy is staged before cudaMemcpyAsync returns, so the next row may reuse m.bias.
            if (!m.bias.empty() && !ok(cudaMemcpyAsync(sc.bias, m.bias.data(), m.bias.size() * sizeof(cere_logit_bias),
                                                       cudaMemcpyHostToDevice, stream_))) {
                return CERE_ERR_INTERNAL;
            }
            if (!ok(process_logits(proc, kUnpaddedVocab, rs.counts, p.frequency_penalty, p.presence_penalty,
                                   p.repetition_penalty, sc.bias, int64_t(m.bias.size()), sc.proc, stream_)) ||
                !ok(k12::top2_f32(sc.proc, kUnpaddedVocab, sc.top2_ids, sc.top2_scores, stream_))) {
                return CERE_ERR_INTERNAL;
            }
            proc = sc.proc;
            const uint32_t *drawn = nullptr;
            if (temperature != 0.0f) {
                const int64_t top_k = std::min<int64_t>(int64_t(p.top_k), kUnpaddedVocab);
                if (!ok(k12::probabilities(proc, kUnpaddedVocab, temperature, sc.probs, sc.lse, sc.scratch, stream_)) ||
                    !ok(k12::keep_mask(sc.probs, kUnpaddedVocab, top_k, p.top_p, p.min_p, sc.keep, sc.scratch,
                                       stream_)) ||
                    !ok(k12::draw_seeded(sc.probs, sc.keep, kUnpaddedVocab, p.seed, 1, rs.key, sc.drawn, sc.scratch,
                                         stream_))) {
                    return CERE_ERR_INTERNAL;
                }
                drawn = sc.drawn;
            }
            if (!ok(k12::finalize(proc, sc.top2_ids, drawn, score, stream_))) return CERE_ERR_INTERNAL;
        }
        if (rs.logprobs) {
            cere_token_logprobs *lp = sc.logprobs + r;
            if (!ok(k12::top_logprobs(proc, kUnpaddedVocab, temperature, &score->id, 0, int64_t(s.top_logprobs),
                                      &lp->logprob, &lp->top[0].id, &lp->top[0].logprob, 2, stream_))) {
                return CERE_ERR_INTERNAL;
            }
        }
        if (s.sampled && !ok(k12::count_commit(rs.counts, &score->id, 1, stream_))) return CERE_ERR_INTERNAL;
        return CERE_OK;
    }

    // Scores rows [0, rows) of the last forward call into m.scores_host (and the logprobs of the rows that ask into
    // m.logprobs_host); rs[i] is row i's sampler request (rs null: every row greedy). Synchronizes the stream.
    cere_status score(InklingModel &m, int64_t first, int64_t rows, const RowSampling *rs) {
        if (const cere_status st = m.forward.logits(first, rows, m.logits, kLogitLd, stream_); st != CERE_OK) {
            return st;
        }
        if (top2_rows(m.logits, rows, kUnpaddedVocab, kLogitLd, m.scores, stream_) != cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        bool logprobs = false;
        for (int64_t i = 0; rs != nullptr && i < rows; ++i) {
            if (rs[i].sampling == nullptr) continue;
            logprobs = logprobs || rs[i].logprobs;
            if (const cere_status st = sample_row(m, i, rs[i]); st != CERE_OK) {
                cudaGetLastError();
                return st;
            }
        }
        if (cudaMemcpyAsync(m.scores_host.ptr, m.scores, uint64_t(rows) * sizeof(cere_token_score),
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            (logprobs && cudaMemcpyAsync(m.logprobs_host.ptr, m.sampler.logprobs,
                                         uint64_t(rows) * sizeof(cere_token_logprobs), cudaMemcpyDeviceToHost,
                                         stream_) != cudaSuccess) ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        return CERE_OK;
    }

    // The rows of an image prompt's chunk (S6g; checked by check_images): one encoder pass per image whose
    // placeholders the chunk holds, so an image's rows never depend on how its prompt is chunked, then the held rows to
    // m.vision.rows in placeholder order (*rows null when the chunk holds none). Stream-ordered: the next image's
    // pixels land after the previous pass read them.
    cere_status encode_images(InklingModel &m, const host::StepImages &images, int64_t pos, int64_t n,
                              const iv::bf16 **rows) {
        namespace kv = cere::inkling::vision;
        uint64_t placed = 0;
        for (uint64_t i = 0; i < images.count; ++i) {
            const host::SessionImage &im = images.image[i];
            const auto [lo, hi] = held_rows(im, pos, n);
            if (lo >= hi) continue;
            if (cudaMemcpyAsync(m.vision.pixels, im.pixels.data(), im.pixels.size(), cudaMemcpyHostToDevice,
                                stream_) != cudaSuccess) {
                cudaGetLastError();
                return CERE_ERR_INTERNAL;
            }
            const iv::Grid grid{im.span.grid_h, im.span.grid_w};
            if (const cere_status st = iv::encode(m.vision_weights, m.vision_plans, m.vision.pixels, &grid, 1,
                                                  m.vision.scratch, m.vision.scratch_bytes, m.vision.encoded, stream_);
                st != CERE_OK) {
                return st;
            }
            if (cudaMemcpyAsync(m.vision.rows + placed * kv::kOutHidden,
                                m.vision.encoded + (lo - im.span.token_offset) * kv::kOutHidden,
                                (hi - lo) * kv::kOutHidden * sizeof(iv::bf16), cudaMemcpyDeviceToDevice,
                                stream_) != cudaSuccess) {
                cudaGetLastError();
                return CERE_ERR_INTERNAL;
            }
            placed += hi - lo;
        }
        *rows = placed == 0 ? nullptr : m.vision.rows;
        return CERE_OK;
    }

    // One prefill chunk of n tokens (n >= 2, or any n of an image prompt); scores its last row (with `rs`) when
    // `last`. `images`: the prompt's (null for a text prompt). `taps`: a drafting session's (Taps::draft), else null.
    cere_status prefill(InklingModel &m, InklingSession &s, const uint32_t *tokens, int64_t n, bool last,
                        const RowSampling &rs, const host::StepImages *images, const Taps *taps) {
        const iv::bf16 *rows = nullptr;
        if (images != nullptr) {
            if (const cere_status st = encode_images(m, *images, s.pos, n, &rows); st != CERE_OK) return st;
        }
        if (const cere_status st = m.forward.prefill(s.s, s.pos, tokens, n, taps, rows, stream_);
            st != CERE_OK) {
            return st;
        }
        if (last) {
            if (const cere_status st = score(m, n - 1, 1, &rs); st != CERE_OK) return st;
        } else if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        s.pos += n;
        return CERE_OK;
    }

    // One decode call: row i is tokens[i] of sessions[i] at its position; scores every row (rs[i] its request).
    // `taps`: a drafting session's prompt row (Taps::draft), else null (the graph replay).
    cere_status decode(InklingModel &m, InklingSession *const *sessions, int64_t rows, const uint32_t *tokens,
                       const RowSampling *rs, const Taps *taps = nullptr) {
        Session ss[kMaxDecodeRows];
        int64_t pos[kMaxDecodeRows];
        for (int64_t i = 0; i < rows; ++i) {
            ss[i] = sessions[i]->s;
            pos[i] = sessions[i]->pos;
        }
        if (const cere_status st = m.forward.decode(ss, pos, tokens, rows, taps, stream_); st != CERE_OK) {
            return st;
        }
        if (const cere_status st = score(m, 0, rows, rs); st != CERE_OK) return st;
        for (int64_t i = 0; i < rows; ++i) ++sessions[i]->pos;
        return CERE_OK;
    }

    // The drafter's extend (D-I-S9B) after S sessions' committed rows: session i's rows[i] rows at positions
    // first[i] .. first[i] + rows[i] - 1, their drafter taps compacted in session order at `taps` (kDraftTapLd),
    // become its context (the last row also its f_rnn). With `anchors` (the last committed, unfed token of each
    // session, at its position), each session with room for one more token drafts the next block's window - 1
    // proposals into its pending copies; else its pending list is empty. Synchronizes the stream when it drafts.
    cere_status extend(InklingModel &m, InklingSession *const *sessions, const int64_t *first, const int64_t *rows,
                       int64_t S, const bf16 *taps, const uint32_t *anchors) {
        DraftCall &c = *m.call;
        int64_t n = 0;
        for (int64_t i = 0; i < S; ++i) {
            for (int64_t j = 0; j < rows[i]; ++j, ++n) {
                c.sess[n] = &sessions[i]->ds;
                c.pos[n] = first[i] + j;
                c.frnn[n] = j == rows[i] - 1;
            }
            sessions[i]->pending_count = 0;
        }
        if (const cere_status st = m.drafter.context(c.sess, c.pos, c.frnn, taps, kDraftTapLd, n, stream_);
            st != CERE_OK) {
            return st;
        }
        if (anchors == nullptr) return CERE_OK;
        int64_t B = 0;
        for (int64_t i = 0; i < S; ++i) {
            if (sessions[i]->pos >= sessions[i]->s.max_ctx) continue;
            c.block[B] = i;
            c.sess[B] = &sessions[i]->ds;
            c.anchor[B] = anchors[i];
            c.pos[B] = sessions[i]->pos;
            ++B;
        }
        if (B == 0) return CERE_OK;
        const int drafts = int(m.window) - 1;
        auto *ids = static_cast<uint32_t *>(m.draft_ids.ptr);
        if (const cere_status st = m.drafter.draft(c.sess, c.anchor, c.pos, B, drafts, ids, stream_); st != CERE_OK) {
            return st;
        }
        if (cudaMemcpyAsync(m.draft_ids_host.ptr, ids, uint64_t(B) * kMaxBlockDrafts * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        const auto *host_ids = static_cast<const uint32_t *>(m.draft_ids_host.ptr);
        for (int64_t b = 0; b < B; ++b) {
            InklingSession &s = *sessions[c.block[b]];
            std::copy(host_ids + b * kMaxBlockDrafts, host_ids + b * kMaxBlockDrafts + drafts, s.pending);
            s.pending_count = uint32_t(drafts);
        }
        return CERE_OK;
    }

    int ordinal_;
    cudaStream_t stream_;
    cublasLtHandle_t handle_;  // the vision tower's K2 plans (iv::init_plans); the text forward runs no cuBLASLt
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
    out->reset(new InklingDevice(ordinal, stream, handle));
    return CERE_OK;
}

}  // namespace
}  // namespace cere::inkling

namespace cere::host {

const Support *inkling2_support(const Device &, bool vision) { return &cere::inkling::inkling_support(vision); }

cere_status create_device(const char *device_id, std::unique_ptr<Device> *out) {
    return cere::inkling::InklingDevice::open(device_id, out);
}

}  // namespace cere::host
