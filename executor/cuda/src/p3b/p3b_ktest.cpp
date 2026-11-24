// P3b kernel-test entry points (m23 design note P3, M23-D2; executor/tests/gpu/test_p3b_moe.py, lab suite `p3b`;
// the tactic sweep executor/tests/gpu/test_p3blab_sweep.py, lab suite `p3blab`).
// Linked into libcere_ktest.so of a -DCERE_MOE_BACKEND=p3b build only (executor/CMakeLists.txt). Device pointers,
// int64 sizes, the stream last; 0 = success, else a cere_status.
#include "device/memory_plan.h"
#include "host/device.h"
#include "ktest/k2_ktest.h"
#include "ktest/l2_flush.h"
#include "model/model.h"
#include "p3b/p3b_moe.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <vector>

namespace cere::ktest {
namespace {

bool u32(int64_t v) { return v > 0 && v <= std::numeric_limits<int32_t>::max(); }

// The body of cere_ktest_p3b_moe_block_taps, cere_ktest_p3b_moe_block_rows and cere_ktest_p3b_moe_block_fork. rows 0:
// the block's K2 plans pinned at max_m = m; rows > 0 (m <= rows): the executor's rows MoePlans (m26 design note
// D6), GemmPlan::init_rows(rows) of each instantiated plan (max_m 1: its bucket-0 algorithm is the executor plan's),
// as k7_ktest.cpp's moe_block. fork 1 (m23 design note P11R-23): the plans carry a model::MoeFork and the block must
// fork. `warmup` / `timed` / `us` / `cold` as cere_ktest_p3b_moe_block_fork's (timed 0: one call on `stream`).
int p3b_block(const void *x, const void *router_weight, const void *w13_weight, const void *w13_scale,
              const void *down_weight, const void *down_scale, const void *g1_alphas, const void *g2_alphas,
              const void *act_scales, const void *shared_gate_weight, const void *shared_up_weight,
              const void *shared_down_weight, const void *shared_expert_gate, void *out, void *routed, void *topk_ids,
              void *topk_weights, const void *routed_in, const void *shared_in, void *shared, int64_t m,
              int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter, int64_t gemm1,
              int64_t gemm2, int64_t rows, int64_t fork, int64_t warmup, int64_t timed, float *us, int64_t cold,
              cudaStream_t stream) {
    const bool pinned = gemm1 == -1 && gemm2 == -1;
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || !u32(shared_inter) ||
        (!pinned && (gemm1 < 0 || gemm2 < 0)) || rows < 0 || rows > kernels::GemmPlan::kMaxRows ||
        (rows > 0 && m > rows) || warmup < 0 || timed < 0 || (timed > 0 && us == nullptr) || cold < 0 || cold > 1) {
        return CERE_ERR_ARGUMENT;
    }
    const p3b::TacticIndices pair{size_t(gemm1), size_t(gemm2)};
    kernels::MoeDims d;
    d.hidden = uint32_t(hidden);
    d.num_experts = uint32_t(num_experts);
    d.top_k = uint32_t(top_k);
    d.inter = uint32_t(inter);
    d.shared_inter = uint32_t(shared_inter);
    kernels::MoeLayer L;
    L.router = static_cast<const __nv_bfloat16 *>(router_weight);
    L.up_weight = static_cast<const uint8_t *>(w13_weight);
    L.up_scale = static_cast<const uint8_t *>(w13_scale);
    L.gate_weight = L.up_weight == nullptr ? nullptr : L.up_weight + size_t(inter) * size_t(hidden / 2);
    L.gate_scale = L.up_scale == nullptr ? nullptr : L.up_scale + size_t(inter) * size_t(hidden / 16);
    L.down_weight = static_cast<const uint8_t *>(down_weight);
    L.down_scale = static_cast<const uint8_t *>(down_scale);
    L.g1_alphas = static_cast<const float *>(g1_alphas);
    L.g2_alphas = static_cast<const float *>(g2_alphas);
    L.act_scales = static_cast<const float *>(act_scales);
    L.shared_gate = static_cast<const __nv_bfloat16 *>(shared_gate_weight);
    L.shared_up = static_cast<const __nv_bfloat16 *>(shared_up_weight);
    L.shared_down = static_cast<const __nv_bfloat16 *>(shared_down_weight);
    L.shared_expert_gate = static_cast<const __nv_bfloat16 *>(shared_expert_gate);

    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    const uint32_t mm = uint32_t(m);
    const uint32_t max_m = rows == 0 ? mm : 1;
    kernels::GemmPlan router;
    kernels::GemmPlan shared_gate_up;
    kernels::GemmPlan shared_down;
    void *gemm_workspace = nullptr;
    cere_status s;
    if ((s = router.init(handle, d.num_experts, d.hidden, max_m, kernels::GemmOut::kBF16)) != CERE_OK ||
        (s = plan_init(&shared_gate_up, handle, d.shared_inter, d.hidden, max_m, kernels::GemmOut::kBF16,
                       &gemm_workspace, stream)) != CERE_OK) {
        return s;
    }
    s = shared_down.init(handle, d.hidden, d.shared_inter, max_m, kernels::GemmOut::kBF16);
    kernels::GemmPlan rows_router;
    kernels::GemmPlan rows_gate_up;
    kernels::GemmPlan rows_down;
    if (s == CERE_OK && rows > 0) {
        const uint32_t r = uint32_t(rows);
        if ((s = rows_router.init_rows(handle, router, r)) == CERE_OK &&
            (s = rows_gate_up.init_rows(handle, shared_gate_up, r)) == CERE_OK) {
            s = rows_down.init_rows(handle, shared_down, r);
        }
    }
    const size_t bytes = p3b::moe_workspace_bytes(d, mm);
    void *workspace = nullptr;
    if (s == CERE_OK && bytes == 0) s = CERE_ERR_ARGUMENT;
    if (s == CERE_OK && cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess) s = CERE_ERR_CAPACITY;
    model::MoeFork moe_fork;
    if (s == CERE_OK && fork == 1) s = moe_fork.create();
    if (s == CERE_OK) {
        const kernels::MoePlans plans = rows == 0 ? moe_fork.plans(&router, &shared_gate_up, &shared_down)
                                                  : moe_fork.plans(&rows_router, &rows_gate_up, &rows_down);
        kernels::MoeTaps taps;
        taps.routed = static_cast<__nv_bfloat16 *>(routed);
        taps.routed_in = static_cast<const __nv_bfloat16 *>(routed_in);
        taps.shared_in = static_cast<const __nv_bfloat16 *>(shared_in);
        taps.shared = static_cast<__nv_bfloat16 *>(shared);
        const bool tapped = routed != nullptr || routed_in != nullptr || shared_in != nullptr || shared != nullptr;
        const auto block = [&](cudaStream_t on) {
            return p3b::moe_block(d, L, plans, static_cast<const __nv_bfloat16 *>(x), mm,
                                  static_cast<__nv_bfloat16 *>(out), workspace, bytes, on, tapped ? &taps : nullptr,
                                  static_cast<int32_t *>(topk_ids), static_cast<float *>(topk_weights),
                                  pinned ? nullptr : &pair);
        };
        if (fork == 1 && (routed_in != nullptr || shared_in != nullptr || !kernels::moe_shared_forks(plans, mm))) {
            s = CERE_ERR_ARGUMENT;
        } else {
            s = timed == 0 ? block(stream) : graph_spans(block, warmup, timed, cold == 1, us, stream);
        }
        if (cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    }
    if (gemm_workspace != nullptr && cudaFreeAsync(gemm_workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

}  // namespace
}  // namespace cere::ktest

// The P3b MoE block of one layer in its arena layout (p3b/p3b_moe.h): `w13_weight` [E, 2I, H/2] (per expert
// [up; gate]), `w13_scale` [E, 2I, H/16] and `down_scale` [E, H, I/16] already swizzled (cere_ktest_p3b_swizzle),
// `act_scales` device FP32 {a1, a2}. Plans as cere_ktest_moe_block (rows 0: max_m = m). `routed` (nullable, BF16
// [m, H]) receives the finalized routed rows, `topk_ids` / `topk_weights` (nullable) the router's output.
// The taps of kernels::MoeTaps (all nullable, BF16 [m, H]; the P3b numerics diagnosis, test_p3blab_numerics.py):
// `routed_in` given rows replace the router and the routed experts (K7's routed_in path), `shared_in` given rows
// replace the shared expert, `shared` receives the shared expert's rows (before the sigmoid gate). `gemm1` / `gemm2`:
// the routed experts' tactic pair as indices into the runner's lists, both -1 = m's pinned pair.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_moe_block_taps(
    const void *x, const void *router_weight, const void *w13_weight, const void *w13_scale, const void *down_weight,
    const void *down_scale, const void *g1_alphas, const void *g2_alphas, const void *act_scales,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, void *routed, void *topk_ids, void *topk_weights, const void *routed_in,
    const void *shared_in, void *shared, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
    int64_t shared_inter, int64_t gemm1, int64_t gemm2, cudaStream_t stream) {
    return cere::ktest::p3b_block(x, router_weight, w13_weight, w13_scale, down_weight, down_scale, g1_alphas,
                                  g2_alphas, act_scales, shared_gate_weight, shared_up_weight, shared_down_weight,
                                  shared_expert_gate, out, routed, topk_ids, topk_weights, routed_in, shared_in, shared,
                                  m, hidden, num_experts, top_k, inter, shared_inter, gemm1, gemm2, 0, 0, 0, 0,
                                  nullptr, 0, stream);
}

// cere_ktest_p3b_moe_block_taps without the given rows and the shared tap, on m's pinned pair.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_moe_block(
    const void *x, const void *router_weight, const void *w13_weight, const void *w13_scale, const void *down_weight,
    const void *down_scale, const void *g1_alphas, const void *g2_alphas, const void *act_scales,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, void *routed, void *topk_ids, void *topk_weights, int64_t m,
    int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter, cudaStream_t stream) {
    return cere_ktest_p3b_moe_block_taps(x, router_weight, w13_weight, w13_scale, down_weight, down_scale, g1_alphas,
                                         g2_alphas, act_scales, shared_gate_weight, shared_up_weight,
                                         shared_down_weight, shared_expert_gate, out, routed, topk_ids, topk_weights,
                                         nullptr, nullptr, nullptr, m, hidden, num_experts, top_k, inter,
                                         shared_inter, -1, -1, stream);
}

// Row invariance (test_p3b_moe.py test_row_invariance): cere_ktest_p3b_moe_block on m's pinned pair with the
// executor's rows MoePlans (rows in [1, GemmPlan::kMaxRows], m <= rows; the batched decode and verify path), or with
// rows 0 the plans of cere_ktest_p3b_moe_block. `gemm1` / `gemm2` as cere_ktest_p3b_moe_block_taps'.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_moe_block_rows(
    const void *x, const void *router_weight, const void *w13_weight, const void *w13_scale, const void *down_weight,
    const void *down_scale, const void *g1_alphas, const void *g2_alphas, const void *act_scales,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, void *routed, void *topk_ids, void *topk_weights, int64_t m,
    int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter, int64_t gemm1,
    int64_t gemm2, int64_t rows, cudaStream_t stream) {
    return cere::ktest::p3b_block(x, router_weight, w13_weight, w13_scale, down_weight, down_scale, g1_alphas,
                                  g2_alphas, act_scales, shared_gate_weight, shared_up_weight, shared_down_weight,
                                  shared_expert_gate, out, routed, topk_ids, topk_weights, nullptr, nullptr, nullptr,
                                  m, hidden, num_experts, top_k, inter, shared_inter, gemm1, gemm2, rows, 0, 0, 0,
                                  nullptr, 0, stream);
}

// The shared-expert fork (m23 design note P11R-23; test_p3b_moe.py test_shared_fork_*): cere_ktest_p3b_moe_block_rows
// (gemm1 / gemm2 / rows as its) with `fork` 1: the plans carry a fork (model::MoeFork) and the block must take it
// (kernels::moe_shared_forks: m <= 16, else CERE_ERR_ARGUMENT), `fork` 0: no fork (the serial block). `timed` 0: one
// call on `stream`. `timed` > 0: one call captured in a CUDA graph on a stream of its own (the fork one branch of it),
// replayed `warmup` then `timed` times, each replay's span in microseconds into `us` (host FP32 [timed]); `cold` 1:
// each timed replay after an L2 flush (ktest/l2_flush.h, outside its span), 0: L2-hot back-to-back replays. `out` /
// `routed` / `topk_ids` / `topk_weights` hold the last replay's.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_moe_block_fork(
    const void *x, const void *router_weight, const void *w13_weight, const void *w13_scale, const void *down_weight,
    const void *down_scale, const void *g1_alphas, const void *g2_alphas, const void *act_scales,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, void *routed, void *topk_ids, void *topk_weights, int64_t m,
    int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter, int64_t gemm1,
    int64_t gemm2, int64_t rows, int64_t fork, int64_t warmup, int64_t timed, float *us, int64_t cold,
    cudaStream_t stream) {
    if (fork < 0 || fork > 1) return CERE_ERR_ARGUMENT;
    return cere::ktest::p3b_block(x, router_weight, w13_weight, w13_scale, down_weight, down_scale, g1_alphas,
                                  g2_alphas, act_scales, shared_gate_weight, shared_up_weight, shared_down_weight,
                                  shared_expert_gate, out, routed, topk_ids, topk_weights, nullptr, nullptr, nullptr,
                                  m, hidden, num_experts, top_k, inter, shared_inter, gemm1, gemm2, rows, fork, warmup,
                                  timed, us, cold, stream);
}

// p3b::swizzle_scales in place on `batches` row-major FP8 [rows, cols] matrices (the INSTANTIATE swizzle), with a
// scratch of one matrix allocated on `stream`.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_swizzle(void *scales, int64_t batches,
                                                                             int64_t rows, int64_t cols,
                                                                             cudaStream_t stream) {
    using cere::ktest::u32;
    if (scales == nullptr || !u32(batches) || !u32(rows) || !u32(cols)) return CERE_ERR_ARGUMENT;
    const size_t bytes = (size_t(rows) * size_t(cols) + 255) / 256 * 256;
    void *scratch = nullptr;
    if (cudaMallocAsync(&scratch, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    cere_status s = cere::p3b::swizzle_scales(static_cast<uint8_t *>(scales), uint64_t(batches), uint32_t(rows),
                                              uint32_t(cols), scratch, bytes, stream);
    if (cudaFreeAsync(scratch, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    return s;
}

// The runner's tactic lists and m's pinned pair (p3b_tactic.h kPinnedBuckets): out[0] / out[1] = GEMM1 / GEMM2 list
// sizes, out[2] / out[3] = the pinned indices in them at `m`. INSTANTIATE's status when it fails
// (CERE_ERR_UNSUPPORTED: a pin matches 0 or >1 tactics).
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_tactics(int64_t *out, int64_t m, cudaStream_t) {
    if (out == nullptr || !cere::ktest::u32(m)) return CERE_ERR_ARGUMENT;
    cere_status s = cere::p3b::instantiate();
    if (s != CERE_OK) return s;
    size_t n1 = 0;
    size_t n2 = 0;
    cere::p3b::TacticIndices pair;
    if ((s = cere::p3b::tactic_lists(nullptr, &n1, nullptr, &n2, 0)) != CERE_OK ||
        (s = cere::p3b::pinned_indices(uint32_t(m), &pair)) != CERE_OK) {
        return s;
    }
    out[0] = int64_t(n1);
    out[1] = int64_t(n2);
    out[2] = int64_t(pair.gemm1);
    out[3] = int64_t(pair.gemm2);
    return CERE_OK;
}

// D4 (m32n design note): p3b::runner_binding of the runner slot `ordinal` with `device` current (set for the call,
// the caller's current device restored after): out (host int64 [2]) out[0] = its status there (0: that device's own
// runner, built on first use; CERE_ERR_INTERNAL: built on another ordinal, a runner shared across cards, refused),
// out[1] = the ordinal it was built on (-1: none). CERE_ERR_ARGUMENT for a null `out` or a non-int ordinal/device;
// CERE_ERR_UNSUPPORTED when `device` cannot be made current.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_runner_binding(int64_t ordinal, int64_t device,
                                                                                    int64_t *out, cudaStream_t) {
    if (out == nullptr || ordinal < INT32_MIN || ordinal > INT32_MAX || device < 0 || device > INT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(int(device)) != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_UNSUPPORTED;
    }
    int built = -1;
    out[0] = int64_t(cere::p3b::runner_binding(int(ordinal), &built));
    out[1] = int64_t(built);
    return cudaSetDevice(previous) == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// The runner's tactic lists by configuration (p3b_tactic.h Tactic): out (host int64 [2 * cap * 5]) row i < cap of
// GEMM1 then row i < cap of GEMM2, each {tile_m, tile_n, tile_k_bytes, swap_ab, finalize}; rows past a list's size
// are -1. The list sizes come from cere_ktest_p3b_tactics.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_tactic_configs(int64_t *out, int64_t cap,
                                                                                    cudaStream_t) {
    if (out == nullptr || !cere::ktest::u32(cap)) return CERE_ERR_ARGUMENT;
    cere_status s = cere::p3b::instantiate();
    if (s != CERE_OK) return s;
    try {
        std::vector<cere::p3b::Tactic> g1(static_cast<size_t>(cap)), g2(static_cast<size_t>(cap));
        size_t n1 = 0;
        size_t n2 = 0;
        if ((s = cere::p3b::tactic_lists(g1.data(), &n1, g2.data(), &n2, size_t(cap))) != CERE_OK) return s;
        for (int64_t list = 0; list < 2; ++list) {
            const std::vector<cere::p3b::Tactic> &tactics = list == 0 ? g1 : g2;
            const size_t n = list == 0 ? n1 : n2;
            for (size_t i = 0; i < size_t(cap); ++i) {
                int64_t *row = out + (size_t(list) * size_t(cap) + i) * 5;
                const cere::p3b::Tactic &t = tactics[i];
                const bool have = i < n;
                row[0] = have ? t.tile_m : -1;
                row[1] = have ? t.tile_n : -1;
                row[2] = have ? t.tile_k_bytes : -1;
                row[3] = have ? int64_t(t.swap_ab) : -1;
                row[4] = have ? int64_t(t.finalize) : -1;
            }
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}


// The routed experts alone (p3b::routed_experts; the P3b tactic sweep, m23 design note P3): FlashInfer's runner on
// the given routing (`topk_ids` int32 / `topk_weights` FP32 [m, top_k], e.g. cere_ktest_p3b_moe_block's), weights as
// cere_ktest_p3b_moe_block's, the finalized BF16 rows into `out` [m, H]. `gemm1` / `gemm2`: indices into the runner's
// lists, both -1 = the pins. `timed` 0: one call on `stream`. `timed` > 0: one call captured in a CUDA graph on a
// stream of its own, replayed `warmup` then `timed` times, each replay's span in microseconds into `us` (host FP32
// [timed]); `out` holds the last replay's rows. A tactic the runner cannot run: CERE_ERR_INTERNAL.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_routed(
    const void *x, const void *w13_weight, const void *w13_scale, const void *down_weight, const void *down_scale,
    const void *g1_alphas, const void *g2_alphas, const void *act_scales, const void *topk_ids,
    const void *topk_weights, void *out, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
    int64_t inter, int64_t gemm1, int64_t gemm2, int64_t warmup, int64_t timed, float *us, cudaStream_t stream) {
    using cere::ktest::u32;
    const bool pinned = gemm1 == -1 && gemm2 == -1;
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || warmup < 0 || timed < 0 ||
        (timed > 0 && us == nullptr) || (!pinned && (gemm1 < 0 || gemm2 < 0))) {
        return CERE_ERR_ARGUMENT;
    }
    cere::kernels::MoeDims d;
    d.hidden = uint32_t(hidden);
    d.num_experts = uint32_t(num_experts);
    d.top_k = uint32_t(top_k);
    d.inter = uint32_t(inter);
    cere::kernels::MoeLayer L;
    L.up_weight = static_cast<const uint8_t *>(w13_weight);
    L.up_scale = static_cast<const uint8_t *>(w13_scale);
    L.down_weight = static_cast<const uint8_t *>(down_weight);
    L.down_scale = static_cast<const uint8_t *>(down_scale);
    L.g1_alphas = static_cast<const float *>(g1_alphas);
    L.g2_alphas = static_cast<const float *>(g2_alphas);
    L.act_scales = static_cast<const float *>(act_scales);
    const cere::p3b::TacticIndices pair{size_t(gemm1), size_t(gemm2)};
    const uint32_t mm = uint32_t(m);
    const size_t bytes = cere::p3b::routed_workspace_bytes(d, mm);
    if (bytes == 0) return CERE_ERR_ARGUMENT;
    void *workspace = nullptr;
    if (cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const auto run = [&](cudaStream_t on) {
        return cere::p3b::routed_experts(d, L, static_cast<const __nv_bfloat16 *>(x),
                                         static_cast<const int32_t *>(topk_ids), static_cast<const float *>(topk_weights),
                                         mm, static_cast<__nv_bfloat16 *>(out), workspace, bytes, on,
                                         pinned ? nullptr : &pair);
    };
    cere_status s = timed == 0 ? run(stream) : cere::ktest::graph_spans(run, warmup, timed, false, us, stream);
    if (cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    return s;
}

// P14: whether the model's routed_rows (RowsPath::kAuto) runs an m-token call of these dims on the decode GEMMs, into
// `*decoded` (host int32, 1 / 0). Host only; the stream (oracle.call's) is unused.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_decode_routed(int64_t m, int64_t hidden,
                                                                                   int64_t num_experts, int64_t top_k,
                                                                                   int64_t inter, int32_t *decoded,
                                                                                   cudaStream_t) {
    using cere::ktest::u32;
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || decoded == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cere::kernels::MoeDims d;
    d.hidden = uint32_t(hidden);
    d.num_experts = uint32_t(num_experts);
    d.top_k = uint32_t(top_k);
    d.inter = uint32_t(inter);
    *decoded = cere::p3b::decode_routed(d, uint32_t(m)) ? 1 : 0;
    return CERE_OK;
}

// P14 (p3b/p3b_decode.h): the decode GEMMs against FlashInfer's grouped GEMMs on one routing, weights as
// cere_ktest_p3b_routed's, m's pinned pair. Outputs (device): `rows` BF16 [m * top_k, H] (permuted expert rows),
// `act` FP4 [m * top_k, I / 2] and `act_sf` UE4M3 [m * top_k, I / 16] (GEMM2's A and its scale factors, compact).
// `mode` 0: routed_rows on the grouped GEMMs (the reference; its swizzled scale factors gathered compact). 1:
// routed_rows on the decode GEMMs with MMA chain `order` (0 forward, 1 last k64 block dropped: the fail-first
// perturbation). 2: GEMM2 alone: mode 0, then decode::gemm2 on its GEMM2 A (the group table built from its expert
// offsets) into `rows`. 3: GEMM1 + SwiGLU + FP4 quantize alone: mode 1, then decode::gemm1_act on its GEMM1 A into
// `act` / `act_sf`. `timed` > 0: the mode's measured launches (0 / 1 routed_rows, 2 gemm2, 3 gemm1_act) captured
// once in a CUDA graph and replayed `warmup` then `timed` times, each replay's span in microseconds into `us` (host
// FP32 [timed]); `cold` 1: each timed replay after an L2 flush (ktest/l2_flush.h), 0: L2-hot back-to-back replays.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_decode(
    const void *x, const void *w13_weight, const void *w13_scale, const void *down_weight, const void *down_scale,
    const void *g1_alphas, const void *g2_alphas, const void *act_scales, const void *topk_ids,
    const void *topk_weights, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
    int64_t mode, int64_t order, void *rows, void *act, void *act_sf, int64_t warmup, int64_t timed, int64_t cold,
    float *us, cudaStream_t stream) {
    using cere::ktest::u32;
    namespace dec = cere::p3b::decode;
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || mode < 0 || mode > 3 ||
        order < 0 || order > 1 || warmup < 0 || timed < 0 || cold < 0 || cold > 1 || (timed > 0 && us == nullptr) ||
        rows == nullptr || act == nullptr || act_sf == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cere::kernels::MoeDims d;
    d.hidden = uint32_t(hidden);
    d.num_experts = uint32_t(num_experts);
    d.top_k = uint32_t(top_k);
    d.inter = uint32_t(inter);
    cere::kernels::MoeLayer L;
    L.up_weight = static_cast<const uint8_t *>(w13_weight);
    L.up_scale = static_cast<const uint8_t *>(w13_scale);
    L.down_weight = static_cast<const uint8_t *>(down_weight);
    L.down_scale = static_cast<const uint8_t *>(down_scale);
    L.g1_alphas = static_cast<const float *>(g1_alphas);
    L.g2_alphas = static_cast<const float *>(g2_alphas);
    L.act_scales = static_cast<const float *>(act_scales);
    const uint32_t mm = uint32_t(m);
    const int64_t mk = m * top_k;
    const dec::Order k_order = order == 1 ? dec::Order::kDropLast : dec::Order::kForward;
    const dec::Shape shape{m, hidden, inter, num_experts, top_k};
    if ((mode != 0 && !dec::supported(shape)) || cere::p3b::instantiate() != CERE_OK) {
        return CERE_ERR_ARGUMENT;
    }
    const size_t bytes = cere::p3b::routed_workspace_bytes(d, mm);
    if (bytes == 0) return CERE_ERR_ARGUMENT;
    const size_t groups_bytes = sizeof(dec::Group) * size_t(dec::slots(shape));
    void *workspace = nullptr;
    void *groups = nullptr;
    if (cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    if (cudaMallocAsync(&groups, groups_bytes, stream) != cudaSuccess) {
        (void)cudaFreeAsync(workspace, stream);
        return CERE_ERR_CAPACITY;
    }
    const auto *xb = static_cast<const __nv_bfloat16 *>(x);
    const auto *ids = static_cast<const int32_t *>(topk_ids);
    const auto *wts = static_cast<const float *>(topk_weights);
    auto *rows_out = static_cast<__nv_bfloat16 *>(rows);
    auto *act_out = static_cast<uint8_t *>(act);
    auto *sf_out = static_cast<uint8_t *>(act_sf);
    cere::p3b::Rows r;
    const auto path = [&](cudaStream_t on) {
        const auto rows_path = mode == 0 || mode == 2 ? cere::p3b::RowsPath::kGrouped : cere::p3b::RowsPath::kDecode;
        return cere::p3b::routed_rows_view(d, L, xb, ids, wts, mm, rows_path, k_order, workspace, bytes, on, &r);
    };
    const auto gemm2 = [&](cudaStream_t on) {
        return cere::p3b::decode_gemm2(d, L, mm, static_cast<const dec::Group *>(groups), r.act, sf_out, rows_out,
                                       k_order, on);
    };
    const auto gemm1 = [&](cudaStream_t on) {
        return cere::p3b::decode_gemm1(d, L, mm, r.groups, r.x, r.x_sf, act_out, sf_out, k_order, on);
    };
    const auto ck = [](cudaError_t e) { return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; };
    const bool flush = cold == 1;
    cere_status s =
        timed > 0 && mode < 2 ? cere::ktest::graph_spans(path, warmup, timed, flush, us, stream) : path(stream);
    // The path's rows and GEMM2 A (compact scale factors) into the outputs.
    if (s == CERE_OK) s = ck(cudaMemcpyAsync(rows_out, r.rows, size_t(mk) * hidden * sizeof(__nv_bfloat16),
                                             cudaMemcpyDeviceToDevice, stream));
    if (s == CERE_OK) s = ck(cudaMemcpyAsync(act_out, r.act, size_t(mk) * inter / 2, cudaMemcpyDeviceToDevice, stream));
    if (s == CERE_OK) {
        s = r.decoded ? ck(cudaMemcpyAsync(sf_out, r.act_sf, size_t(mk) * inter / 16, cudaMemcpyDeviceToDevice, stream))
                      : dec::compact_act_sf(r.offsets, r.act_sf, mk, inter, num_experts, sf_out, stream);
    }
    if (s == CERE_OK && mode == 2) {
        // The group table of the reference's expert offsets (host-built: active experts in increasing id).
        std::vector<int64_t> offsets(size_t(num_experts) + 1);
        s = ck(cudaMemcpyAsync(offsets.data(), r.offsets, offsets.size() * sizeof(int64_t), cudaMemcpyDeviceToHost,
                               stream));
        if (s == CERE_OK) s = ck(cudaStreamSynchronize(stream));
        std::vector<dec::Group> table;
        for (int64_t e = 0; e < num_experts && s == CERE_OK; ++e) {
            if (offsets[e + 1] > offsets[e]) {
                table.push_back(dec::Group{int32_t(e), int32_t(offsets[e]), int32_t(offsets[e + 1] - offsets[e]), 0});
            }
        }
        const int32_t active = int32_t(table.size());
        for (auto &g : table) g.active = active;
        table.resize(size_t(dec::slots(shape)), dec::Group{-1, 0, 0, active});
        if (s == CERE_OK) {
            s = ck(cudaMemcpyAsync(groups, table.data(), groups_bytes, cudaMemcpyHostToDevice, stream));
        }
        if (s == CERE_OK) s = ck(cudaStreamSynchronize(stream));
        if (s == CERE_OK) {
            s = timed > 0 ? cere::ktest::graph_spans(gemm2, warmup, timed, flush, us, stream) : gemm2(stream);
        }
    }
    if (s == CERE_OK && mode == 3) {
        s = r.decoded ? (timed > 0 ? cere::ktest::graph_spans(gemm1, warmup, timed, flush, us, stream) : gemm1(stream))
                      : CERE_ERR_ARGUMENT;
    }
    if (s == CERE_OK) s = ck(cudaGetLastError());
    if (cudaFreeAsync(groups, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    if (cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    return s;
}

// M23-D2 arena size (host arithmetic, no CUDA call): the DEVICE weight bytes of the tensor table `tensors` (count
// entries, cere_model_desc.tensors) in the P3b arena order (out[0]) and in the K7 order (out[1]); out[2] / out[3] =
// the P3b / K7 MoE workspace bytes at workspace_rows(ctx) rows of the A4 MoE; out[4] / out[5] = the Workspace's
// shared scratch (model::scratch_bytes: its only MoE-dependent region) of config `cfg` at those rows and ctx, P3b / K7.
extern "C" __attribute__((visibility("default"))) int cere_ktest_p3b_arena_bytes(const cere_tensor_desc *tensors,
                                                                                 uint64_t count,
                                                                                 const cere_qwen4_exp_config *cfg,
                                                                                 int64_t ctx, uint64_t *out,
                                                                                 cudaStream_t) {
    if (tensors == nullptr || count == 0 || cfg == nullptr || out == nullptr || !cere::ktest::u32(ctx))
        return CERE_ERR_ARGUMENT;
    const uint64_t rows = cere::model::workspace_rows(uint64_t(ctx));
    try {
        cere::host::ModelPlan plan{};
        plan.tensors.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            const cere_tensor_desc &t = tensors[i];
            if (t.name == nullptr || t.ndim > CERE_TENSOR_MAX_DIMS) return CERE_ERR_ARGUMENT;
            cere::host::TensorPlan &p = plan.tensors[i];
            p.name = t.name;
            p.dtype = t.dtype;
            p.placement = t.placement;
            p.ndim = t.ndim;
            std::memcpy(p.shape, t.shape, sizeof(p.shape));
            p.bytes = t.bytes;
        }
        const cere::model::ArenaOrder p3b = cere::model::arena_order(plan);
        const cere::model::ArenaOrder k7 = cere::model::arena_order_k7(plan);
        out[0] = cere::device::weight_bytes(plan, p3b.order, p3b.packed);
        out[1] = cere::device::weight_bytes(plan, k7.order, k7.packed);
        const cere::kernels::MoeDims d{cere::model::kHidden, cere::model::kExperts, cere::model::kTopK,
                                       cere::model::kMoeInter, cere::model::kSharedInter};
        out[2] = cere::p3b::moe_workspace_bytes(d, uint32_t(rows));
        out[3] = cere::kernels::moe_workspace_bytes(d, uint32_t(rows));
        uint64_t scratch[2] = {0, 0};
        cere::model::scratch_bytes_p3b_k7(*cfg, rows, uint64_t(ctx), scratch);
        out[4] = scratch[0];
        out[5] = scratch[1];
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}
