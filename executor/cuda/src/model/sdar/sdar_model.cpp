// SelfTune ST4 S3c: sdar_model.h.
#include "model/sdar/sdar_model.h"

#include "cere_model_constants.h"
#if defined(CERE_SDAR_MUTANT_FP32TIE)
#include "kernels/sdar/sdar_mutants.h"
#endif

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string_view>

namespace cere::sdar {

namespace {

constexpr uint64_t kHidden = gen::kHiddenSize;
constexpr uint64_t kInter = gen::kIntermediateSize;
constexpr uint64_t kHeads = gen::kNumAttentionHeads;
constexpr uint64_t kKvHeads = gen::kNumKeyValueHeads;
constexpr uint64_t kVocab = gen::kVocabSize;
constexpr uint64_t kLayers = gen::kNumHiddenLayers;
constexpr uint64_t kQDim = kHeads * kHeadDim;
constexpr uint64_t kKvDim = kKvHeads * kHeadDim;
constexpr float kEps = float(gen::kRmsNormEps);
constexpr float kTheta = float(gen::kRopeTheta);
constexpr uint64_t kAlign = 256;

// SelfTune ST4 S4a: the fail-first mutants of a test-only `native_dev.py build --sdar-mutant <name>` library
// (executor/CMakeLists.txt CERE_SDAR_MUTANT; lock_native.py refuses it), which scripts/live/dice_judge.py must judge
// FAIL. A plain build defines neither, and its code is the unmutated one.
// - causal: every forward attends token-causally inside its block (a denoise row no longer sees the later rows).
// - fp32tie: a denoise picks with kernels/sdar/sdar_mutants.h top_tie_fp32 (FP32 logits compared as given).
#if defined(CERE_SDAR_MUTANT_CAUSAL)
constexpr Mask kAttnMask = Mask::kCausal;
#else
constexpr Mask kAttnMask = Mask::kBlockCausal;
#endif

static_assert(gen::kHeadDim == uint32_t(kHeadDim), "the sdar kernels are compiled for head_dim 128");
static_assert(gen::kDecodeBlockSize == uint32_t(kBlock) && kBlock == int64_t(kScheduleBlock),
              "the sdar kernels and schedule are compiled for block_size 4");
static_assert(gen::kRotaryDim == gen::kHeadDim, "family sdar rotates the whole head");
static_assert(!gen::kWeightsTieWordEmbeddings, "family sdar binds its own lm_head");
static_assert(kQDim == kHidden, "q_proj and o_proj share one GEMM plan");
static_assert(kMaxRows % kScheduleBlock == 0, "prefill passes start on block boundaries");

constexpr uint64_t align(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

cere_status cuda(cudaError_t e) { return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; }

// "model.layers.<L>.<rest>" -> L and rest; false for any other name.
bool layer_name(std::string_view name, uint64_t *layer, std::string_view *rest) {
    constexpr std::string_view prefix = "model.layers.";
    if (name.substr(0, prefix.size()) != prefix) return false;
    name.remove_prefix(prefix.size());
    const size_t dot = name.find('.');
    if (dot == 0 || dot == std::string_view::npos || dot > 3) return false;
    uint64_t v = 0;
    for (char c : name.substr(0, dot)) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + uint64_t(c - '0');
    }
    if (dot > 1 && name[0] == '0') return false;
    *layer = v;
    *rest = name.substr(dot + 1);
    return true;
}

}  // namespace

uint64_t layout_tensors(std::vector<TensorSlot> *slots) {
    // gate_proj index per layer, to place up_proj right after it.
    std::map<uint64_t, size_t> gate, up;
    for (size_t i = 0; i < slots->size(); ++i) {
        uint64_t layer = 0;
        std::string_view rest;
        if (!layer_name((*slots)[i].name, &layer, &rest)) continue;
        if (rest == "mlp.gate_proj.weight") gate[layer] = i;
        if (rest == "mlp.up_proj.weight") up[layer] = i;
    }
    std::vector<bool> placed(slots->size(), false);
    uint64_t at = 0;
    const auto place = [&](size_t i) {
        (*slots)[i].offset = at;
        at = align(at + (*slots)[i].bytes);
        placed[i] = true;
    };
    for (size_t i = 0; i < slots->size(); ++i) {
        if (placed[i]) continue;
        place(i);
        uint64_t layer = 0;
        std::string_view rest;
        if (layer_name((*slots)[i].name, &layer, &rest) && rest == "mlp.gate_proj.weight") {
            const auto u = up.find(layer);
            if (u != up.end() && !placed[u->second]) place(u->second);
        }
    }
    return at;
}

cere_status bind_weights(const std::vector<TensorSlot> &slots, const uint8_t *arena, Weights *out) {
    Weights w;
    w.layers.assign(kLayers, LayerWeights{});
    std::map<std::string, const TensorSlot *> by_name;
    for (const TensorSlot &s : slots) by_name[s.name] = &s;
    const auto refuse = [](const std::string &name, const char *why) {
        std::fprintf(stderr, "cere: sdar tensor %s: %s\n", name.c_str(), why);
        return CERE_ERR_ARGUMENT;
    };
    std::set<std::string> read;
    const auto take = [&](const std::string &name, uint64_t d0, uint64_t d1, const bf16 **ptr) -> cere_status {
        const auto it = by_name.find(name);
        if (it == by_name.end()) return refuse(name, "missing");
        const TensorSlot &s = *it->second;
        const uint32_t ndim = d1 == 0 ? 1 : 2;
        if (s.dtype != CERE_DTYPE_BF16) return refuse(name, "dtype is not BF16");
        if (s.ndim != ndim || s.shape[0] != d0 || (ndim == 2 && s.shape[1] != d1)) return refuse(name, "shape");
        *ptr = reinterpret_cast<const bf16 *>(arena + s.offset);
        read.insert(name);
        return CERE_OK;
    };
    cere_status st = CERE_OK;
    const auto all = [&](cere_status s) {
        if (st == CERE_OK) st = s;
    };
    all(take("model.embed_tokens.weight", kVocab, kHidden, &w.embed));
    all(take("model.norm.weight", kHidden, 0, &w.norm));
    all(take("lm_head.weight", kVocab, kHidden, &w.lm_head));
    for (uint64_t l = 0; l < kLayers && st == CERE_OK; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        LayerWeights &lw = w.layers[l];
        const bf16 *gate = nullptr, *up = nullptr;
        all(take(p + "input_layernorm.weight", kHidden, 0, &lw.input_norm));
        all(take(p + "post_attention_layernorm.weight", kHidden, 0, &lw.post_norm));
        all(take(p + "self_attn.q_norm.weight", kHeadDim, 0, &lw.q_norm));
        all(take(p + "self_attn.k_norm.weight", kHeadDim, 0, &lw.k_norm));
        all(take(p + "self_attn.q_proj.weight", kQDim, kHidden, &lw.q));
        all(take(p + "self_attn.k_proj.weight", kKvDim, kHidden, &lw.k));
        all(take(p + "self_attn.v_proj.weight", kKvDim, kHidden, &lw.v));
        all(take(p + "self_attn.o_proj.weight", kHidden, kQDim, &lw.o));
        all(take(p + "mlp.gate_proj.weight", kInter, kHidden, &gate));
        all(take(p + "mlp.up_proj.weight", kInter, kHidden, &up));
        all(take(p + "mlp.down_proj.weight", kHidden, kInter, &lw.down));
        if (st == CERE_OK && up != gate + kInter * kHidden) {
            return refuse(p + "mlp.up_proj.weight", "not placed right after gate_proj");
        }
        lw.gate_up = gate;
    }
    if (st != CERE_OK) return st;
    for (const TensorSlot &s : slots) {
        if (read.count(s.name) == 0) return refuse(s.name, "not read by the forward");
    }
    *out = std::move(w);
    return CERE_OK;
}

namespace {

struct WorkspaceLayout {
    uint64_t x, h, q, k, v, attn, tmp, gate_up, act, logits, ids, top, masked, index, total;
};

constexpr WorkspaceLayout workspace_layout() {
    WorkspaceLayout l{};
    uint64_t at = 0;
    const auto take = [&at](uint64_t bytes) {
        const uint64_t off = at;
        at = align(at + bytes);
        return off;
    };
    l.x = take(kMaxRows * kHidden * 2);
    l.h = take(kMaxRows * kHidden * 2);
    l.q = take(kMaxRows * kQDim * 2);
    l.k = take(kMaxRows * kKvDim * 2);
    l.v = take(kMaxRows * kKvDim * 2);
    l.attn = take(kMaxRows * kQDim * 2);
    l.tmp = take(kMaxRows * kHidden * 2);
    l.gate_up = take(kMaxRows * 2 * kInter * 2);
    l.act = take(kMaxRows * kInter * 2);
    l.logits = take(kScheduleBlock * kVocab * 4);
    l.ids = take(kMaxRows * 4);
    l.top = take(kScheduleBlock * sizeof(TopTie));
    l.masked = take(kScheduleBlock);
    l.index = take(sizeof(int32_t));
    l.total = at;
    return l;
}

constexpr WorkspaceLayout kWs = workspace_layout();

}  // namespace

uint64_t workspace_bytes() { return kWs.total; }

uint64_t kv_bytes(uint64_t max_context_tokens) {
    return 2 * kLayers * kv_rows(max_context_tokens) * kKvDim * sizeof(bf16);
}

Workspace carve_workspace(uint8_t *base) {
    Workspace w;
    w.x = reinterpret_cast<bf16 *>(base + kWs.x);
    w.h = reinterpret_cast<bf16 *>(base + kWs.h);
    w.q = reinterpret_cast<bf16 *>(base + kWs.q);
    w.k = reinterpret_cast<bf16 *>(base + kWs.k);
    w.v = reinterpret_cast<bf16 *>(base + kWs.v);
    w.attn = reinterpret_cast<bf16 *>(base + kWs.attn);
    w.tmp = reinterpret_cast<bf16 *>(base + kWs.tmp);
    w.gate_up = reinterpret_cast<bf16 *>(base + kWs.gate_up);
    w.act = reinterpret_cast<bf16 *>(base + kWs.act);
    w.logits = reinterpret_cast<float *>(base + kWs.logits);
    w.ids = reinterpret_cast<uint32_t *>(base + kWs.ids);
    w.top = reinterpret_cast<TopTie *>(base + kWs.top);
    w.masked = base + kWs.masked;
    w.index = reinterpret_cast<int32_t *>(base + kWs.index);
    return w;
}

GemmWorkspace &GemmWorkspace::operator=(GemmWorkspace &&other) noexcept {
    if (this != &other) {
        if (ptr != nullptr) cudaFree(ptr);
        ptr = other.ptr;
        other.ptr = nullptr;
    }
    return *this;
}

GemmWorkspace::~GemmWorkspace() {
    if (ptr != nullptr) cudaFree(ptr);
}

uint64_t gemm_workspace_bytes(uint32_t impl) {
    return impl == CERE_LEVER_SDAR_GEMM_CHANGING ? kChangingWorkspaceBytes : 0;
}

cere_status Gemms::init(cublasLtHandle_t handle, uint32_t impl) {
    if (impl != CERE_LEVER_SDAR_GEMM_IDENTICAL && impl != CERE_LEVER_SDAR_GEMM_CHANGING) return CERE_ERR_ARGUMENT;
    const struct {
        Gemm *g;
        uint64_t n, k, max_m;
        GemmOut out;
    } plans[] = {
        {&qo, kHidden, kHidden, kMaxRows, GemmOut::kBF16},
        {&kv, kKvDim, kHidden, kMaxRows, GemmOut::kBF16},
        {&gate_up, 2 * kInter, kHidden, kMaxRows, GemmOut::kBF16},
        {&down, kHidden, kInter, kMaxRows, GemmOut::kBF16},
        {&lm_head, kVocab, kHidden, kScheduleBlock, GemmOut::kF32},
    };
    for (const auto &p : plans) {
        if (const cere_status st = p.g->init(handle, uint32_t(p.n), uint32_t(p.k), uint32_t(p.max_m), p.out);
            st != CERE_OK) {
            return st;
        }
    }
    if (impl != CERE_LEVER_SDAR_GEMM_CHANGING) return CERE_OK;
    PinTarget process{};
    if (const cere_status st = runtime_target(&process); st != CERE_OK) return st;
    if (workspace.ptr == nullptr && cudaMalloc(&workspace.ptr, kChangingWorkspaceBytes) != cudaSuccess) {
        workspace.ptr = nullptr;
        return CERE_ERR_CAPACITY;
    }
    for (const auto &p : plans) {
        if (const cere_status st = p.g->set_changing(changing_pins(), process, workspace.ptr, kChangingWorkspaceBytes);
            st != CERE_OK) {
            return st;
        }
    }
    return CERE_OK;
}

cere_status CudaForward::layers(const uint32_t *tokens, uint64_t n, uint64_t pos) {
    if (n == 0 || n > kMaxRows || pos % kScheduleBlock != 0 || pos + n > kv_rows_) return CERE_ERR_INTERNAL;
    const int64_t rows = int64_t(n);
    cere_status st = cuda(cudaMemcpyAsync(ws_.ids, tokens, n * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_));
    const auto k = [&st](cudaError_t e) {
        if (st == CERE_OK) st = cuda(e);
    };
    const auto g = [&st](cere_status s) {
        if (st == CERE_OK) st = s;
    };
    k(embed(w_.embed, ws_.ids, rows, int(kHidden), ws_.x, stream_));
    for (uint64_t l = 0; l < kLayers && st == CERE_OK; ++l) {
        const LayerWeights &lw = w_.layers[l];
        bf16 *kc = k_cache_ + l * kv_rows_ * kKvDim;
        bf16 *vc = v_cache_ + l * kv_rows_ * kKvDim;
        k(rmsnorm(ws_.x, kHidden, lw.input_norm, kEps, rows, int(kHidden), ws_.h, kHidden, stream_));
        g(gemms_.qo.run(ws_.h, lw.q, uint32_t(n), ws_.q, stream_));
        g(gemms_.kv.run(ws_.h, lw.k, uint32_t(n), ws_.k, stream_));
        g(gemms_.kv.run(ws_.h, lw.v, uint32_t(n), ws_.v, stream_));
        k(rmsnorm(ws_.q, kHeadDim, lw.q_norm, kEps, rows * int64_t(kHeads), kHeadDim, ws_.q, kHeadDim, stream_));
        k(rmsnorm(ws_.k, kHeadDim, lw.k_norm, kEps, rows * int64_t(kKvHeads), kHeadDim, ws_.k, kHeadDim, stream_));
        k(rope(ws_.q, kQDim, rows, int(kHeads), int64_t(pos), kTheta, stream_));
        k(rope(ws_.k, kKvDim, rows, int(kKvHeads), int64_t(pos), kTheta, stream_));
        k(kv_store(ws_.k, kKvDim, ws_.v, kKvDim, rows, int(kKvHeads), kc, vc, int64_t(pos), stream_));
        k(attention(ws_.q, kQDim, kc, vc, int64_t(pos), kc + pos * kKvDim, kKvDim, vc + pos * kKvDim, kKvDim, rows,
                    int(kHeads), int(kKvHeads), ws_.attn, kQDim, kAttnMask, stream_));
        g(gemms_.qo.run(ws_.attn, lw.o, uint32_t(n), ws_.tmp, stream_));
        k(add(ws_.x, ws_.tmp, rows * int64_t(kHidden), stream_));
        k(rmsnorm(ws_.x, kHidden, lw.post_norm, kEps, rows, int(kHidden), ws_.h, kHidden, stream_));
        g(gemms_.gate_up.run(ws_.h, lw.gate_up, uint32_t(n), ws_.gate_up, stream_));
        k(silu_mul(ws_.gate_up, rows, int(kInter), ws_.act, stream_));
        g(gemms_.down.run(ws_.act, lw.down, uint32_t(n), ws_.tmp, stream_));
        k(add(ws_.x, ws_.tmp, rows * int64_t(kHidden), stream_));
    }
    return st;
}

cere_status CudaForward::prefill(const uint32_t *tokens, uint64_t n, uint64_t pos) {
    if (n % kScheduleBlock != 0) return CERE_ERR_INTERNAL;
    for (uint64_t done = 0; done < n;) {
        const uint64_t pass = std::min<uint64_t>(n - done, kMaxRows);
        if (const cere_status st = layers(tokens + done, pass, pos + done); st != CERE_OK) return st;
        done += pass;
    }
    return cuda(cudaStreamSynchronize(stream_));
}

cere_status CudaForward::denoise(const uint32_t *block, const bool *masked, uint64_t base, RowPick *picks,
                                 int32_t *index) {
    if (const cere_status st = layers(block, kScheduleBlock, base); st != CERE_OK) return st;
    uint8_t mask_bytes[kScheduleBlock];
    for (uint64_t r = 0; r < kScheduleBlock; ++r) mask_bytes[r] = masked[r] ? 1 : 0;
    const int64_t rows = int64_t(kScheduleBlock);
    cere_status st = cuda(rmsnorm(ws_.x, kHidden, w_.norm, kEps, rows, int(kHidden), ws_.h, kHidden, stream_));
    if (st == CERE_OK) st = gemms_.lm_head.run(ws_.h, w_.lm_head, uint32_t(kScheduleBlock), ws_.logits, stream_);
#if defined(CERE_SDAR_MUTANT_FP32TIE)
    if (st == CERE_OK) {
        st = cuda(mutant::top_tie_fp32(ws_.logits, int64_t(kVocab), rows, int64_t(kVocab), ws_.top, stream_));
    }
#else
    if (st == CERE_OK) st = cuda(top_tie(ws_.logits, int64_t(kVocab), rows, int64_t(kVocab), ws_.top, stream_));
#endif
    if (st == CERE_OK) {
        st = cuda(cudaMemcpyAsync(ws_.masked, mask_bytes, sizeof(mask_bytes), cudaMemcpyHostToDevice, stream_));
    }
    if (st == CERE_OK) st = cuda(unmask_select(ws_.top, ws_.masked, int(kScheduleBlock), ws_.index, stream_));
    TopTie top[kScheduleBlock];
    if (st == CERE_OK) st = cuda(cudaMemcpyAsync(top, ws_.top, sizeof(top), cudaMemcpyDeviceToHost, stream_));
    if (st == CERE_OK) st = cuda(cudaMemcpyAsync(index, ws_.index, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
    if (st == CERE_OK) st = cuda(cudaStreamSynchronize(stream_));
    if (st != CERE_OK) return st;
    for (uint64_t r = 0; r < kScheduleBlock; ++r) {
        picks[r].id = top[r].id;
        picks[r].ties = top[r].ties;
        picks[r].runner_up = top[r].runner_up;
        picks[r].logit = top[r].logit;
        picks[r].runner_up_logit = top[r].runner_up_logit;
    }
    return CERE_OK;
}

cere_status CudaForward::commit(const uint32_t *block, uint64_t base) {
    if (const cere_status st = layers(block, kScheduleBlock, base); st != CERE_OK) return st;
    return cuda(cudaStreamSynchronize(stream_));
}

}  // namespace cere::sdar
