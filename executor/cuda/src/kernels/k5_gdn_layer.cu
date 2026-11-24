// K5 (m22 design note, m22n design note D7): composition of the GDN linear-attention mixer (see k5_gdn_layer.h).
//
// math_reference: flash-linear-attention
// math_reference: flashinfer
// math_reference: sglang
//
// The order of operations follows Qwen3_5GatedDeltaNet.forward (srt/models/qwen3_5.py) and GDNAttnBackend
// forward_extend (PREFILL: Triton conv, fused_gdn_gating, FLA chunk_gated_delta_rule) / forward_decode (DECODE: Triton
// conv update, FlashInfer gated_delta_rule_decode_pretranspose) in srt/layers/attention/linear/gdn_backend.py.
#include "kernels/k5_gdn_layer.h"

#include "kernels/k1_primitives.h"
#include "kernels/k5_gdn.h"
#include "kernels/k5_gdn_chunk.h"
#include "kernels/k5_gdn_decode.h"

namespace cere::k5 {
namespace {

constexpr size_t kAlign = 256;
constexpr float kScale = 0.08838834764831845f;  // 128^-0.5 (SGLang passes no scale: k.shape[-1] ** -0.5)

size_t round_up(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// Consecutive 256-byte aligned regions from `base` (nullptr: sizing only).
class Carver {
public:
    explicit Carver(void *base) : base_(reinterpret_cast<uintptr_t>(base)) {}
    template <typename T>
    T *take(size_t bytes) {
        const uintptr_t addr = base_ + off_;
        off_ += round_up(bytes);
        return reinterpret_cast<T *>(addr);
    }
    size_t bytes() const { return off_; }

private:
    uintptr_t base_;
    size_t off_ = 0;
};

struct CoreWorkspace {
    __nv_bfloat16 *conv_out;  // [T, 10240]: PREFILL q [T, 2048] | k [T, 2048] | v [T, 6144] planes (each starts a
                              // multiple of 4096 bytes in, so the 16-byte alignment k5::l2norm needs holds); DECODE
                              // one q | k | v row
    __nv_bfloat16 *o;         // [T, 6144]
    float *g;                 // [T, 48]
    float *beta;              // [T, 48]
    void *chunk;              // chunk_workspace_bytes(T)
    size_t bytes;
};

CoreWorkspace carve_core(Carver &c, int64_t tokens) {
    const size_t t = static_cast<size_t>(tokens);
    CoreWorkspace ws{};
    const size_t start = c.bytes();
    ws.conv_out = c.take<__nv_bfloat16>(t * kConvDim * sizeof(__nv_bfloat16));
    ws.o = c.take<__nv_bfloat16>(t * kValueDim * sizeof(__nv_bfloat16));
    ws.g = c.take<float>(t * kValueHeads * sizeof(float));
    ws.beta = c.take<float>(t * kValueHeads * sizeof(float));
    ws.chunk = c.take<void>(chunk_workspace_bytes(tokens));
    ws.bytes = c.bytes() - start;
    return ws;
}

// The projections' outputs; `mixed` is dead once the conv has run and then holds y.
struct LayerWorkspace {
    __nv_bfloat16 *mixed;  // [T, 10240], then y [T, 6144]
    __nv_bfloat16 *z;      // [T, 6144]
    __nv_bfloat16 *b;      // [T, 48]
    __nv_bfloat16 *a;      // [T, 48]
    void *core;            // core_workspace_bytes(T)
    size_t core_bytes;
    size_t bytes;
};

LayerWorkspace carve_layer(void *base, int64_t tokens) {
    const size_t t = static_cast<size_t>(tokens);
    Carver c(base);
    LayerWorkspace ws{};
    ws.mixed = c.take<__nv_bfloat16>(t * kConvDim * sizeof(__nv_bfloat16));
    ws.z = c.take<__nv_bfloat16>(t * kValueDim * sizeof(__nv_bfloat16));
    ws.b = c.take<__nv_bfloat16>(t * kValueHeads * sizeof(__nv_bfloat16));
    ws.a = c.take<__nv_bfloat16>(t * kValueHeads * sizeof(__nv_bfloat16));
    ws.core = c.take<void>(0);
    ws.core_bytes = carve_core(c, tokens).bytes;
    ws.bytes = c.bytes();
    return ws;
}

bool tokens_valid(int64_t tokens) { return tokens >= 1 && tokens <= kChunkMaxTokens; }

bool plan_is(const kernels::GemmPlan *plan, int64_t n, int64_t k, int64_t tokens) {
    return plan != nullptr && plan->n() == n && plan->k() == k && plan->out() == kernels::GemmOut::kBF16 &&
           static_cast<int64_t>(plan->max_m()) >= tokens;
}

cere_status from_cuda(cudaError_t err) {
    if (err == cudaSuccess) {
        return CERE_OK;
    }
    return err == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
}

// An EndState a PREFILL of `tokens` rows can apply: a capture row on the 64 grid strictly inside the call with both
// outputs, or no capture at all.
bool end_valid(const EndState &end, uint32_t operation, int64_t tokens) {
    if (operation != CERE_OP_PREFILL) return end.ssm_in32 == nullptr && end.row < 0;
    if (end.row < 0) return end.ssm_out32 == nullptr && end.conv_out == nullptr;
    return end.ssm_out32 != nullptr && end.conv_out != nullptr && end.row >= kChunkTokens && end.row < tokens &&
           end.row % kChunkTokens == 0;
}

// forward_extend: the Triton conv over the channel ranges of q, k and v (depthwise, so per range it is the same
// arithmetic as one call over all 10240 channels) into the contiguous planes delta_rule_chunk reads, as
// fused_qkv_split_gdn_prefill hands chunk_gated_delta_rule separate q, k, v; fused_gdn_gating; the chunk rule. With
// `end`, its capture (the conv state read from `mixed` before the conv) and FP32 resume / capture (ChunkArgs).
cere_status core_prefill(const LayerWeights &w, LayerState state, const __nv_bfloat16 *mixed, const __nv_bfloat16 *b,
                         const __nv_bfloat16 *a, int64_t tokens, const CoreWorkspace &ws, cudaStream_t stream,
                         const EndState *end) {
    if (end != nullptr && end->row >= 0) {
        const cere_status st = from_cuda(conv1d_state_at(mixed, kConvDim, end->conv_out, end->row, kConvDim, stream));
        if (st != CERE_OK) {
            return st;
        }
    }
    const int64_t offsets[] = {0, kKeyDim, 2 * kKeyDim};
    const int64_t widths[] = {kKeyDim, kKeyDim, kValueDim};
    __nv_bfloat16 *planes[3];
    __nv_bfloat16 *at = ws.conv_out;
    for (int i = 0; i < 3; ++i) {
        planes[i] = at;
        const cere_status st =
            from_cuda(conv1d_silu(mixed + offsets[i], kConvDim, w.conv1d + offsets[i] * kConvWidth,
                                  state.conv + offsets[i] * kConvState, at, widths[i], tokens, widths[i], stream));
        if (st != CERE_OK) {
            return st;
        }
        at += tokens * widths[i];
    }
    cere_status st = from_cuda(
        gating(a, kValueHeads, b, kValueHeads, w.A_log, w.dt_bias, ws.g, ws.beta, tokens, kValueHeads, stream));
    if (st != CERE_OK) {
        return st;
    }
    ChunkArgs ca{};
    ca.q = planes[0];
    ca.k = planes[1];
    ca.v = planes[2];
    ca.g = ws.g;
    ca.beta = ws.beta;
    ca.state = state.ssm;
    ca.o = ws.o;
    ca.tokens = tokens;
    ca.scale = kScale;
    if (end != nullptr) {
        ca.state_in32 = end->ssm_in32;
        ca.capture32 = end->ssm_out32;
        ca.capture_chunk = end->row >= 0 ? end->row / kChunkTokens : -1;
        ca.state_in32_hi = end->ssm_in32_hi;
        ca.capture32_hi = end->ssm_out32_hi;
    }
    return from_cuda(delta_rule_chunk(ca, ws.chunk, stream));
}

// forward_decode: the Triton conv update of the one token, then FlashInfer's decode step (gating and L2 norm
// in-kernel) on the q | k | v row.
cere_status core_decode(const LayerWeights &w, LayerState state, const __nv_bfloat16 *mixed, const __nv_bfloat16 *b,
                        const __nv_bfloat16 *a, const CoreWorkspace &ws, cudaStream_t stream) {
    const cere_status st = from_cuda(
        conv1d_silu(mixed, kConvDim, w.conv1d, state.conv, ws.conv_out, kConvDim, 1, kConvDim, stream));
    if (st != CERE_OK) {
        return st;
    }
    DecodeArgs da{};
    da.q = ws.conv_out;
    da.k = ws.conv_out + kKeyDim;
    da.v = ws.conv_out + 2 * kKeyDim;
    da.a = a;
    da.b = b;
    da.A_log = w.A_log;
    da.dt_bias = w.dt_bias;
    da.state = state.ssm;
    da.o = ws.o;
    da.scale = kScale;
    return from_cuda(delta_rule_decode(da, stream));
}

struct ChainWorkspace {
    __nv_bfloat16 *conv_out;  // [rows, 10240]: row t's q | k | v
    __nv_bfloat16 *o;         // [rows, 6144]
    size_t bytes;
};

ChainWorkspace carve_chain(void *base, int64_t rows) {
    const size_t t = static_cast<size_t>(rows);
    Carver c(base);
    ChainWorkspace ws{};
    ws.conv_out = c.take<__nv_bfloat16>(t * kConvDim * sizeof(__nv_bfloat16));
    ws.o = c.take<__nv_bfloat16>(t * kValueDim * sizeof(__nv_bfloat16));
    ws.bytes = c.bytes();
    return ws;
}

bool chain_rows_valid(int64_t rows) { return rows >= 1 && rows <= kChainMaxRows; }

}  // namespace

size_t core_workspace_bytes(int64_t max_tokens) {
    if (!tokens_valid(max_tokens)) {
        return 0;
    }
    Carver c(nullptr);
    return carve_core(c, max_tokens).bytes;
}

size_t layer_workspace_bytes(int64_t max_tokens) {
    return tokens_valid(max_tokens) ? carve_layer(nullptr, max_tokens).bytes : 0;
}

size_t chain_core_workspace_bytes(int64_t rows) { return chain_rows_valid(rows) ? carve_chain(nullptr, rows).bytes : 0; }

cere_status layer_core(const LayerWeights &w, LayerState state, uint32_t operation, const __nv_bfloat16 *mixed,
                       const __nv_bfloat16 *z, const __nv_bfloat16 *b, const __nv_bfloat16 *a, int64_t tokens,
                       __nv_bfloat16 *y, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                       const EndState *end) {
    if (operation == CERE_OP_VERIFY) {
        return CERE_ERR_UNSUPPORTED;
    }
    if ((operation != CERE_OP_PREFILL && operation != CERE_OP_DECODE) || mixed == nullptr || z == nullptr ||
        b == nullptr || a == nullptr || y == nullptr || workspace == nullptr || state.conv == nullptr ||
        state.ssm == nullptr || w.conv1d == nullptr || w.A_log == nullptr || w.dt_bias == nullptr ||
        w.norm == nullptr || !tokens_valid(tokens) || (operation == CERE_OP_DECODE && tokens != 1) ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || (end != nullptr && !end_valid(*end, operation, tokens))) {
        return CERE_ERR_ARGUMENT;
    }
    Carver c(workspace);
    const CoreWorkspace ws = carve_core(c, tokens);
    if (ws.bytes > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    const cere_status st = operation == CERE_OP_PREFILL ? core_prefill(w, state, mixed, b, a, tokens, ws, stream, end)
                                                        : core_decode(w, state, mixed, b, a, ws, stream);
    if (st != CERE_OK) {
        return st;
    }
    return from_cuda(k1::rmsnorm_gated_sigmoid(ws.o, kHeadDim, z, kHeadDim, w.norm, kNormEps, tokens * kValueHeads,
                                               kHeadDim, y, kHeadDim, stream));
}

cere_status chain_core(const LayerWeights &w, const Chains &chains, const __nv_bfloat16 *mixed,
                       const __nv_bfloat16 *z, const __nv_bfloat16 *b, const __nv_bfloat16 *a, int64_t rows,
                       __nv_bfloat16 *y, void *workspace, size_t workspace_bytes, cudaStream_t stream) {
    static_assert(kChainMaxRows == kConvMaxRows && kChainMaxRows == kDecodeMaxRows, "one chain bound");
    if (mixed == nullptr || z == nullptr || b == nullptr || a == nullptr || y == nullptr || workspace == nullptr ||
        w.conv1d == nullptr || w.A_log == nullptr || w.dt_bias == nullptr || w.norm == nullptr ||
        !chain_rows_valid(rows) || chains.count < 1 || chains.count > kChainMaxRows ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0) {
        return CERE_ERR_ARGUMENT;
    }
    const ChainWorkspace ws = carve_chain(workspace, rows);
    if (ws.bytes > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    ConvChains conv{};
    DecodeRowsArgs da{};
    conv.chains = da.chains = static_cast<int>(chains.count);
    int64_t next = 0;
    for (int64_t c = 0; c < chains.count; ++c) {
        const LayerState &s = chains.states[c];
        if (s.conv == nullptr || s.ssm == nullptr || chains.rows[c] < 1 || chains.rows[c] > rows - next) {
            return CERE_ERR_ARGUMENT;
        }
        for (int64_t o = 0; o < c; ++o) {
            const LayerState &t = chains.states[o];
            if (s.conv == t.conv || s.ssm == t.ssm || s.conv == t.ssm || s.ssm == t.conv) {
                return CERE_ERR_ARGUMENT;
            }
        }
        conv.first[c] = da.first[c] = static_cast<int>(next);
        conv.rows[c] = da.rows[c] = static_cast<int>(chains.rows[c]);
        conv.states[c] = s.conv;
        da.states[c] = s.ssm;
        next += chains.rows[c];
        for (int64_t r = conv.first[c]; r + 1 < next; ++r) {
            const LayerState &save = chains.saves[r];
            if (save.conv == nullptr || save.ssm == nullptr) {
                return CERE_ERR_ARGUMENT;
            }
            for (int64_t o = 0; o < chains.count; ++o) {
                const LayerState &t = chains.states[o];
                if (save.conv == t.conv || save.ssm == t.ssm || save.conv == t.ssm || save.ssm == t.conv) {
                    return CERE_ERR_ARGUMENT;
                }
            }
            conv.saves[r] = save.conv;
            da.saves[r] = save.ssm;
        }
    }
    if (next != rows) {
        return CERE_ERR_ARGUMENT;
    }
    cere_status st =
        from_cuda(conv1d_silu_rows(mixed, kConvDim, w.conv1d, conv, ws.conv_out, kConvDim, kConvDim, stream));
    if (st != CERE_OK) {
        return st;
    }
    da.step.q = ws.conv_out;
    da.step.k = ws.conv_out + kKeyDim;
    da.step.v = ws.conv_out + 2 * kKeyDim;
    da.step.a = a;
    da.step.b = b;
    da.step.A_log = w.A_log;
    da.step.dt_bias = w.dt_bias;
    da.step.o = ws.o;
    da.step.scale = kScale;
    da.qkv_stride = kConvDim;
    da.ab_stride = kValueHeads;
    da.o_stride = kValueDim;
    st = from_cuda(delta_rule_decode_rows(da, stream));
    if (st != CERE_OK) {
        return st;
    }
    return from_cuda(k1::rmsnorm_gated_sigmoid(ws.o, kHeadDim, z, kHeadDim, w.norm, kNormEps, rows * kValueHeads,
                                               kHeadDim, y, kHeadDim, stream));
}

cere_status layer_forward(const LayerGemms &gemms, const LayerWeights &w, LayerState state, uint32_t operation,
                          const __nv_bfloat16 *hidden, int64_t tokens, int64_t hidden_size, __nv_bfloat16 *out,
                          void *workspace, size_t workspace_bytes, cudaStream_t stream, uint32_t cls,
                          const EndState *end) {
    if (operation == CERE_OP_VERIFY) {
        return CERE_ERR_UNSUPPORTED;
    }
    if (hidden == nullptr || out == nullptr || workspace == nullptr || w.in_proj_qkv == nullptr ||
        w.in_proj_z == nullptr || w.in_proj_b == nullptr || w.in_proj_a == nullptr || w.out_proj == nullptr ||
        (operation != CERE_OP_PREFILL && operation != CERE_OP_DECODE) ||
        (operation == CERE_OP_DECODE && tokens != 1) || !tokens_valid(tokens) || hidden_size < 1 ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || !plan_is(gemms.qkv, kConvDim, hidden_size, tokens) ||
        !plan_is(gemms.z, kValueDim, hidden_size, tokens) || !plan_is(gemms.ba, kValueHeads, hidden_size, tokens) ||
        !plan_is(gemms.out, hidden_size, kValueDim, tokens)) {
        return CERE_ERR_ARGUMENT;
    }
    const LayerWorkspace ws = carve_layer(workspace, tokens);
    if (ws.bytes > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    const auto m = static_cast<uint32_t>(tokens);

    // The four in-projections share `hidden`: one small-M launch at m <= 16 (each output bit-equal to its own call),
    // else four GEMMs.
    const kernels::GemmPlan *plans[4] = {gemms.qkv, gemms.z, gemms.ba, gemms.ba};
    const __nv_bfloat16 *weights[4] = {w.in_proj_qkv, w.in_proj_z, w.in_proj_b, w.in_proj_a};
    void *const outs[4] = {ws.mixed, ws.z, ws.b, ws.a};
    cere_status st = kernels::gemm_segments(plans, weights, outs, 4, hidden, m, stream, cls);
    if (st == CERE_OK) {
        st = layer_core(w, state, operation, ws.mixed, ws.z, ws.b, ws.a, tokens, ws.mixed, ws.core, ws.core_bytes,
                        stream, end);
    }
    if (st != CERE_OK) {
        return st;
    }
    return gemms.out->run(ws.mixed, w.out_proj, m, out, stream, cls);
}

}  // namespace cere::k5
