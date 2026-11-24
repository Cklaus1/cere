// K5R kernel-test entry points (libcere_ktest.so; selftune design note phase STP stage a), the contract of
// executor/tests/gpu/test_kernels_k5_region.py: device pointers to contiguous row-major buffers, int64_t counts, the
// stream last; 0 or a cere_status. Both entry points run one GDN layer's decode mixer on `rows` (1..4) rows, updating
// conv_state / ssm_state in place and writing the state after row t (t < rows - 1) to conv_saves[t] / ssm_saves[t]
// (null at rows 1):
//   cere_ktest_k5r_region     the K5R region kernel (ktest/k5r_region.h), `variant` 0 exact, 1 the fail-first mutant
//   cere_ktest_k5r_reference  today's ops: rows 1 k5::layer_forward(DECODE); rows 2..4 the verify composition of
//                             model forward (kernels::gemm_segments, k5::chain_core, the out_proj plan's run)
// with small-M GEMM plans (max_m = rows, the shipped `k2_gemm = "small_m"` lever). timed 0: one call, then the stream
// is synchronized. timed > 0: ktest::graph_spans (warmup, then `timed` replays of a captured call, after an l2_flush
// each when cold != 0) writes each replay's span in microseconds to `us` (F32 [timed]); the state advances per replay.
#include "cere_executor.h"
#include "kernels/k2_gemm.h"
#include "kernels/k5_gdn.h"
#include "kernels/k5_gdn_layer.h"
#include "ktest/k2_ktest.h"
#include "ktest/k5r_region.h"
#include "ktest/l2_flush.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>

namespace {

using bf16 = __nv_bfloat16;
using cere::kernels::GemmOut;
using cere::kernels::GemmPlan;
namespace k5 = cere::k5;
namespace k5r = cere::ktest::k5r;

constexpr uint32_t kHidden = 2560;  // A4 hidden_size
constexpr size_t kAlign = 256;
constexpr size_t kConvStateElems = size_t{k5::kConvDim} * k5::kConvState;
constexpr size_t kSsmStateElems = size_t{k5::kValueHeads} * k5::kHeadDim * k5::kHeadDim;

size_t round_up(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

struct Buffers {
    const void *hidden;
    k5::LayerWeights w;
    bf16 *conv_state;
    bf16 *ssm_state;
    bf16 *conv_saves;  // [rows - 1, 10240, 3]
    bf16 *ssm_saves;   // [rows - 1, 48, 128, 128]
    bf16 *out;
    int64_t rows;
};

bool buffers_ok(const Buffers &b) {
    return b.hidden != nullptr && b.w.in_proj_qkv != nullptr && b.w.in_proj_z != nullptr && b.w.in_proj_b != nullptr &&
           b.w.in_proj_a != nullptr && b.w.conv1d != nullptr && b.w.A_log != nullptr && b.w.dt_bias != nullptr &&
           b.w.norm != nullptr && b.w.out_proj != nullptr && b.conv_state != nullptr && b.ssm_state != nullptr &&
           b.out != nullptr && b.rows >= 1 && b.rows <= k5r::kMaxRows &&
           (b.rows == 1 || (b.conv_saves != nullptr && b.ssm_saves != nullptr));
}

// One call (timed 0) or graph_spans of `run`, with `bytes` of workspace (zeroed when `zero`) freed afterwards.
cere_status drive(size_t bytes, bool zero, int64_t warmup, int64_t timed, int64_t cold, float *us,
                  cudaStream_t stream, const std::function<cere_status(void *, cudaStream_t)> &run) {
    if (warmup < 0 || timed < 0 || (timed > 0 && us == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    void *workspace = nullptr;
    if (cudaMalloc(&workspace, bytes) != cudaSuccess) {
        return CERE_ERR_CAPACITY;
    }
    cere_status s = CERE_OK;
    if (zero && cudaMemsetAsync(workspace, 0, bytes, stream) != cudaSuccess) {
        s = CERE_ERR_INTERNAL;
    }
    if (s == CERE_OK) {
        s = timed == 0 ? run(workspace, stream)
                       : cere::ktest::graph_spans([&](cudaStream_t st) { return run(workspace, st); }, warmup, timed,
                                                  cold != 0, us, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    cudaFree(workspace);
    return s;
}

cere_status region(const Buffers &b, int64_t variant, int64_t warmup, int64_t timed, int64_t cold, float *us,
                   cudaStream_t stream) {
    if (!buffers_ok(b) || (variant != 0 && variant != 1)) {
        return CERE_ERR_ARGUMENT;
    }
    k5r::Args args{};
    args.hidden = static_cast<const bf16 *>(b.hidden);
    args.w = b.w;
    args.conv_state = b.conv_state;
    args.ssm_state = b.ssm_state;
    for (int64_t t = 0; t + 1 < b.rows; ++t) {
        args.conv_saves[t] = b.conv_saves + t * kConvStateElems;
        args.ssm_saves[t] = b.ssm_saves + t * kSsmStateElems;
    }
    args.out = b.out;
    args.rows = static_cast<int>(b.rows);
    const size_t bytes = k5r::workspace_bytes(args.rows);
    const auto v = variant == 0 ? k5r::Variant::kExact : k5r::Variant::kMutantSliceSum;
    return drive(bytes, true, warmup, timed, cold, us, stream, [&](void *ws, cudaStream_t st) {
        return k5r::forward(args, ws, bytes, v, st);
    });
}

// The verify composition's GDN scratch (forward.cpp carve_gdn_rows): mixed, z, b, a, y, then chain_core's workspace.
struct Rows {
    bf16 *mixed;
    bf16 *z;
    bf16 *b;
    bf16 *a;
    bf16 *y;
    void *core;
    size_t core_bytes;
    size_t bytes;
};

Rows carve_rows(void *base, int64_t rows) {
    const auto r = static_cast<size_t>(rows);
    auto *at = static_cast<char *>(base);
    size_t off = 0;
    const auto take = [&](size_t bytes) {
        char *p = at == nullptr ? nullptr : at + off;
        off += round_up(bytes);
        return p;
    };
    Rows g{};
    g.mixed = reinterpret_cast<bf16 *>(take(r * k5::kConvDim * sizeof(bf16)));
    g.z = reinterpret_cast<bf16 *>(take(r * k5::kValueDim * sizeof(bf16)));
    g.b = reinterpret_cast<bf16 *>(take(r * k5::kValueHeads * sizeof(bf16)));
    g.a = reinterpret_cast<bf16 *>(take(r * k5::kValueHeads * sizeof(bf16)));
    g.y = reinterpret_cast<bf16 *>(take(r * k5::kValueDim * sizeof(bf16)));
    g.core_bytes = k5::chain_core_workspace_bytes(rows);
    g.core = take(g.core_bytes);
    g.bytes = off;
    return g;
}

cere_status reference(const Buffers &b, int64_t warmup, int64_t timed, int64_t cold, float *us, cudaStream_t stream) {
    if (!buffers_ok(b)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const auto m = static_cast<uint32_t>(b.rows);
    GemmPlan qkv, z, ba, out;
    void *pin_ws[4] = {};
    GemmPlan *plans[4] = {&qkv, &z, &ba, &out};
    const uint32_t ns[4] = {k5::kConvDim, k5::kValueDim, k5::kValueHeads, kHidden};
    const uint32_t ks[4] = {kHidden, kHidden, kHidden, k5::kValueDim};
    cere_status s = CERE_OK;
    for (int i = 0; i < 4 && s == CERE_OK; ++i) {
        s = cere::ktest::plan_init(plans[i], handle, ns[i], ks[i], m, GemmOut::kBF16, &pin_ws[i], stream);
        if (s == CERE_OK) {
            plans[i]->use_small_m(true);
            if (!plans[i]->small_m()) {
                s = CERE_ERR_UNSUPPORTED;
            }
        }
    }
    const k5::LayerState state{b.conv_state, b.ssm_state};
    const auto *hidden = static_cast<const bf16 *>(b.hidden);
    if (s == CERE_OK && b.rows == 1) {
        const k5::LayerGemms gemms{&qkv, &z, &ba, &out};
        const size_t bytes = k5::layer_workspace_bytes(1);
        s = drive(bytes, false, warmup, timed, cold, us, stream, [&](void *ws, cudaStream_t st) {
            return k5::layer_forward(gemms, b.w, state, CERE_OP_DECODE, hidden, 1, kHidden, b.out, ws, bytes, st);
        });
    } else if (s == CERE_OK) {
        k5::Chains chain;
        chain.count = 1;
        chain.rows[0] = b.rows;
        chain.states[0] = state;
        for (int64_t t = 0; t + 1 < b.rows; ++t) {
            chain.saves[t] = k5::LayerState{b.conv_saves + t * kConvStateElems, b.ssm_saves + t * kSsmStateElems};
        }
        const GemmPlan *in_plans[4] = {&qkv, &z, &ba, &ba};
        const bf16 *weights[4] = {b.w.in_proj_qkv, b.w.in_proj_z, b.w.in_proj_b, b.w.in_proj_a};
        const size_t bytes = carve_rows(nullptr, b.rows).bytes;
        s = drive(bytes, false, warmup, timed, cold, us, stream, [&](void *ws, cudaStream_t st) {
            const Rows g = carve_rows(ws, b.rows);
            void *const outs[4] = {g.mixed, g.z, g.b, g.a};
            cere_status r = cere::kernels::gemm_segments(in_plans, weights, outs, 4, hidden, m, st);
            if (r == CERE_OK) {
                r = k5::chain_core(b.w, chain, g.mixed, g.z, g.b, g.a, b.rows, g.y, g.core, g.core_bytes, st);
            }
            return r == CERE_OK ? out.run(g.y, b.w.out_proj, m, b.out, st) : r;
        });
    }
    for (void *p : pin_ws) {
        if (p != nullptr) {
            cudaFreeAsync(p, stream);
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

Buffers buffers(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b, const void *w_a,
                const void *conv_weight, const void *A_log, const void *dt_bias, const void *norm_weight,
                const void *w_out, void *conv_state, void *ssm_state, void *conv_saves, void *ssm_saves, void *out,
                int64_t rows) {
    Buffers b{};
    b.hidden = hidden;
    b.w = k5::LayerWeights{static_cast<const bf16 *>(w_qkv),     static_cast<const bf16 *>(w_z),
                           static_cast<const bf16 *>(w_b),       static_cast<const bf16 *>(w_a),
                           static_cast<const bf16 *>(conv_weight), static_cast<const float *>(A_log),
                           static_cast<const bf16 *>(dt_bias),   static_cast<const bf16 *>(norm_weight),
                           static_cast<const bf16 *>(w_out)};
    b.conv_state = static_cast<bf16 *>(conv_state);
    b.ssm_state = static_cast<bf16 *>(ssm_state);
    b.conv_saves = static_cast<bf16 *>(conv_saves);
    b.ssm_saves = static_cast<bf16 *>(ssm_saves);
    b.out = static_cast<bf16 *>(out);
    b.rows = rows;
    return b;
}

}  // namespace

#pragma GCC visibility push(default)
extern "C" {

int cere_ktest_k5r_region(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b, const void *w_a,
                          const void *conv_weight, const void *A_log, const void *dt_bias, const void *norm_weight,
                          const void *w_out, void *conv_state, void *ssm_state, void *conv_saves, void *ssm_saves,
                          void *out, int64_t rows, int64_t variant, int64_t warmup, int64_t timed, int64_t cold,
                          void *us, cudaStream_t stream) {
    return region(buffers(hidden, w_qkv, w_z, w_b, w_a, conv_weight, A_log, dt_bias, norm_weight, w_out, conv_state,
                          ssm_state, conv_saves, ssm_saves, out, rows),
                  variant, warmup, timed, cold, static_cast<float *>(us), stream);
}

int cere_ktest_k5r_reference(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b, const void *w_a,
                             const void *conv_weight, const void *A_log, const void *dt_bias, const void *norm_weight,
                             const void *w_out, void *conv_state, void *ssm_state, void *conv_saves, void *ssm_saves,
                             void *out, int64_t rows, int64_t warmup, int64_t timed, int64_t cold, void *us,
                             cudaStream_t stream) {
    return reference(buffers(hidden, w_qkv, w_z, w_b, w_a, conv_weight, A_log, dt_bias, norm_weight, w_out,
                             conv_state, ssm_state, conv_saves, ssm_saves, out, rows),
                     warmup, timed, cold, static_cast<float *>(us), stream);
}

}  // extern "C"
#pragma GCC visibility pop
