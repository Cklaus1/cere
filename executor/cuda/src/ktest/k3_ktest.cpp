// K3 kernel-test entry points (executor/tests/gpu/test_kernels_k3_hc.py, test_kernels_k3_chain.py): hyper-connection
// mix (also the final mix-only mixer), combine and the combine -> mix chain. Device pointers, int64 sizes, the stream
// last; 0 = success, else a cere_status.
#include "device/context_generation.h"
#include "kernels/k2_small_m.h"
#include "kernels/k3_hc.h"
#include "ktest/k2_ktest.h"
#include "ktest/l2_flush.h"

#include <limits>

namespace cere::ktest {
namespace {

bool fits(int64_t v) { return v > 0 && v <= std::numeric_limits<int32_t>::max(); }

// The mix's path codes (small_m > 0): 1 hc_mix's default (kAuto), 2 kUnfused, 3 kEpilogue, 4 kEarly (K3Chain).
constexpr int64_t kMaxPath = 4;

kernels::HcMixPath path_of(int64_t small_m) {
    switch (small_m) {
        case 2:
            return kernels::HcMixPath::kUnfused;
        case 3:
            return kernels::HcMixPath::kEpilogue;
        case 4:
            return kernels::HcMixPath::kEarly;
        default:
            return kernels::HcMixPath::kAuto;
    }
}

// The plans of one call and the mix workspace (cudaMalloc: 256-byte aligned, as hc_mix requires; cudaFree
// synchronizes). rows 0: plans made by HcPlans::init at max_m = m; rows > 0: HcPlans::init_rows(rows, fused_allowed)
// of the instantiated plans (max_m = 1: their bucket-0 algorithms are the executor's), m <= rows. small_m 0: the
// plans as made (cuBLASLt); else both plans on the small-M kernel (the k2_gemm lever's small_m).
struct MixPlans {
    kernels::HcPlans instantiated;
    kernels::HcPlans rows_plans;
    kernels::HcPlans *plans = nullptr;
    void *workspace = nullptr;

    cere_status init(const kernels::HcShape &shape, int64_t m, int64_t rows, int64_t fused_allowed, int64_t small_m) {
        cublasLtHandle_t handle = gemm_handle();
        if (handle == nullptr) {
            return CERE_ERR_INTERNAL;
        }
        cere_status s = instantiated.init(handle, shape, rows == 0 ? static_cast<uint32_t>(m) : 1);
        if (s != CERE_OK) {
            return s;
        }
        if (rows > 0) {
            s = rows_plans.init_rows(handle, instantiated, static_cast<uint32_t>(rows),
                                     static_cast<uint32_t>(fused_allowed));
            if (s != CERE_OK) {
                return s;
            }
        }
        plans = rows == 0 ? &instantiated : &rows_plans;
        plans->down.use_small_m(small_m != 0);
        plans->up.use_small_m(small_m != 0);
        if (cudaMalloc(&workspace, kernels::hc_mix_workspace_bytes(shape, plans->down.max_m())) != cudaSuccess) {
            workspace = nullptr;
            return CERE_ERR_CAPACITY;
        }
        return CERE_OK;
    }

    // Frees the workspace: `s`, or CERE_ERR_INTERNAL when it was CERE_OK and the free failed.
    cere_status release(cere_status s) {
        if (workspace != nullptr && cudaFree(workspace) != cudaSuccess && s == CERE_OK) {
            s = CERE_ERR_INTERNAL;
        }
        workspace = nullptr;
        return s;
    }
};

bool mix_args_ok(int64_t m, int64_t hc, int64_t hidden, int64_t lowrank, int64_t rows, int64_t small_m) {
    return fits(m) && fits(hc) && fits(hidden) && fits(lowrank) && rows >= 0 && rows <= kernels::GemmPlan::kMaxRows &&
           (rows == 0 || m <= rows) && small_m >= 0 && small_m <= kMaxPath;
}

}  // namespace

// MixPlans (rows, fused_allowed, small_m) and hc_mix by path_of(small_m) (m23 design note P11R-21, K3Chain).
cere_status hc_mix(const void *hyper, const void *norm_weight, const void *w_down, const void *w_up, int64_t m,
                   int64_t hc, int64_t hidden, int64_t lowrank, float eps, void *normed, void *mixed, int64_t rows,
                   int64_t fused_allowed, int64_t small_m, cudaStream_t stream) {
    if (!mix_args_ok(m, hc, hidden, lowrank, rows, small_m)) {
        return CERE_ERR_ARGUMENT;
    }
    const kernels::HcShape shape{static_cast<uint32_t>(hc), static_cast<uint32_t>(hidden),
                                 static_cast<uint32_t>(lowrank), eps};
    MixPlans p;
    cere_status s = p.init(shape, m, rows, fused_allowed, small_m);
    if (s == CERE_OK) {
        s = kernels::hc_mix(*p.plans, static_cast<const __nv_bfloat16 *>(hyper),
                            static_cast<const __nv_bfloat16 *>(norm_weight), static_cast<const __nv_bfloat16 *>(w_down),
                            static_cast<const __nv_bfloat16 *>(w_up), static_cast<uint32_t>(m),
                            static_cast<__nv_bfloat16 *>(normed), static_cast<__nv_bfloat16 *>(mixed), p.workspace,
                            stream, path_of(small_m));
    }
    return p.release(s);
}

cere_status hc_combine(const void *block_out, const void *hyper, const void *normed, const void *w_inject, int64_t m,
                       int64_t hc, int64_t hidden, void *out, cudaStream_t stream) {
    if (!fits(m) || !fits(hc) || !fits(hidden)) {
        return CERE_ERR_ARGUMENT;
    }
    const kernels::HcShape shape{static_cast<uint32_t>(hc), static_cast<uint32_t>(hidden), 0, 0.0f};
    return kernels::hc_combine(shape, static_cast<const __nv_bfloat16 *>(block_out),
                               static_cast<const __nv_bfloat16 *>(hyper), static_cast<const __nv_bfloat16 *>(normed),
                               static_cast<const __nv_bfloat16 *>(w_inject), static_cast<uint32_t>(m),
                               static_cast<__nv_bfloat16 *>(out), stream);
}

cere_status hc_combine_norm(const void *block_out, const void *hyper, void *normed, const void *w_inject,
                            const void *next_norm, int64_t m, int64_t hc, int64_t hidden, float eps, void *out,
                            cudaStream_t stream) {
    if (!fits(m) || !fits(hc) || !fits(hidden)) {
        return CERE_ERR_ARGUMENT;
    }
    const kernels::HcShape shape{static_cast<uint32_t>(hc), static_cast<uint32_t>(hidden), 0, eps};
    return kernels::hc_combine_norm(shape, static_cast<const __nv_bfloat16 *>(block_out),
                                    static_cast<const __nv_bfloat16 *>(hyper), static_cast<__nv_bfloat16 *>(normed),
                                    static_cast<const __nv_bfloat16 *>(w_inject),
                                    static_cast<const __nv_bfloat16 *>(next_norm), static_cast<uint32_t>(m),
                                    static_cast<__nv_bfloat16 *>(out), stream);
}

// K3Chain (k3chain design note KC-02, KC-03): the forward's combine -> mix at m rows. hc_combine_norm of block_out
// into `out` (may alias hyper) and `normed` (read as the combine gate, then overwritten with the next mix's norm,
// weight next_norm), then hc_mix_normed of `normed` into `mixed` by path_of(path) (1..kMaxPath), on MixPlans(rows,
// fused_allowed -1, small-M). `chains` such chains in a row, chain c on the c-th weights of the stacks w_inject
// [chains, hc, hc * hidden], next_norm [chains, hc * hidden], w_down [chains, lowrank, hc * hidden] and w_up [chains,
// hc * hidden, lowrank] (the forward's successive mixes, each on its own weights). timed 0: one call. timed > 0:
// `timed` CUDA-graph replays of the chains after 3 warm-up replays (cold != 0: each after an l2_flush), their spans in
// µs into `us` (host float [timed]).
cere_status hc_chain(const void *block_out, const void *hyper, void *normed, const void *w_inject,
                     const void *next_norm, const void *w_down, const void *w_up, int64_t m, int64_t hc,
                     int64_t hidden, int64_t lowrank, float eps, void *out, void *mixed, int64_t rows, int64_t path,
                     int64_t chains, int64_t timed, int64_t cold, float *us, cudaStream_t stream) {
    if (!mix_args_ok(m, hc, hidden, lowrank, rows, path) || path < 1 || !fits(chains) || timed < 0 ||
        (timed > 0 && us == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    const kernels::HcShape shape{static_cast<uint32_t>(hc), static_cast<uint32_t>(hidden),
                                 static_cast<uint32_t>(lowrank), eps};
    const size_t width = static_cast<size_t>(hc) * static_cast<size_t>(hidden);
    const size_t mix_weight = width * static_cast<size_t>(lowrank);
    const size_t inject = width * static_cast<size_t>(hc);
    MixPlans p;
    cere_status s = p.init(shape, m, rows, -1, path);
    const auto chain = [&](cudaStream_t on) {
        for (int64_t c = 0; c < chains; ++c) {
            const size_t i = static_cast<size_t>(c);
            cere_status st = kernels::hc_combine_norm(
                shape, static_cast<const __nv_bfloat16 *>(block_out), static_cast<const __nv_bfloat16 *>(hyper),
                static_cast<__nv_bfloat16 *>(normed), static_cast<const __nv_bfloat16 *>(w_inject) + i * inject,
                static_cast<const __nv_bfloat16 *>(next_norm) + i * width, static_cast<uint32_t>(m),
                static_cast<__nv_bfloat16 *>(out), on);
            if (st == CERE_OK) {
                st = kernels::hc_mix_normed(*p.plans, static_cast<const __nv_bfloat16 *>(normed),
                                            static_cast<const __nv_bfloat16 *>(w_down) + i * mix_weight,
                                            static_cast<const __nv_bfloat16 *>(w_up) + i * mix_weight,
                                            static_cast<uint32_t>(m), static_cast<__nv_bfloat16 *>(mixed),
                                            p.workspace, on, path_of(path));
            }
            if (st != CERE_OK) {
                return st;
            }
        }
        return CERE_OK;
    };
    if (s == CERE_OK) {
        s = timed == 0 ? chain(stream) : graph_spans(chain, 3, timed, cold != 0, us, stream);
    }
    return p.release(s);
}

}  // namespace cere::ktest

extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_mix(
    const void *hyper, const void *norm_weight, const void *w_down, const void *w_up, int64_t m, int64_t hc,
    int64_t hidden, int64_t lowrank, float eps, void *normed, void *mixed, cudaStream_t stream) {
    return cere::ktest::hc_mix(hyper, norm_weight, w_down, w_up, m, hc, hidden, lowrank, eps, normed, mixed, 0, 0, 0,
                               stream);
}

// m26 design note D6 / I6 (K1 test_rows_plan_invariant): cere_ktest_hc_mix through HcPlans::init_rows(rows,
// fused_allowed) of the instantiated plans, m in [1, rows] (rows in [1, GemmPlan::kMaxRows]).
extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_mix_rows(
    const void *hyper, const void *norm_weight, const void *w_down, const void *w_up, int64_t m, int64_t hc,
    int64_t hidden, int64_t lowrank, float eps, void *normed, void *mixed, int64_t rows, int64_t fused_allowed,
    cudaStream_t stream) {
    if (rows < 1) return CERE_ERR_ARGUMENT;
    return cere::ktest::hc_mix(hyper, norm_weight, w_down, w_up, m, hc, hidden, lowrank, eps, normed, mixed, rows,
                               fused_allowed, 0, stream);
}

// m23 design note P11R-21: cere_ktest_hc_mix_rows with both plans on the small-M kernel (the k2_gemm lever's
// small_m); path 1 hc_mix's default, 2 HcMixPath::kUnfused, 3 kEpilogue (the fused hc epilogues where they apply),
// 4 kEarly (K3Chain: kEpilogue with early weights). rows 0 (K3Chain): the plans made by HcPlans::init at max_m = m.
extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_mix_small_m(
    const void *hyper, const void *norm_weight, const void *w_down, const void *w_up, int64_t m, int64_t hc,
    int64_t hidden, int64_t lowrank, float eps, void *normed, void *mixed, int64_t rows, int64_t path,
    cudaStream_t stream) {
    if (path < 1) return CERE_ERR_ARGUMENT;
    return cere::ktest::hc_mix(hyper, norm_weight, w_down, w_up, m, hc, hidden, lowrank, eps, normed, mixed, rows,
                               -1, path, stream);
}

// K3Chain (k3chain design note KC-02): hc_combine_norm then hc_mix_normed by `path` (as cere_ktest_hc_mix_small_m),
// one call, on the small-M plans of `rows` rows (0: init at max_m = m). `out` may alias `hyper`.
extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_chain(
    const void *block_out, const void *hyper, void *normed, const void *w_inject, const void *next_norm,
    const void *w_down, const void *w_up, int64_t m, int64_t hc, int64_t hidden, int64_t lowrank, float eps, void *out,
    void *mixed, int64_t rows, int64_t path, cudaStream_t stream) {
    return cere::ktest::hc_chain(block_out, hyper, normed, w_inject, next_norm, w_down, w_up, m, hc, hidden, lowrank,
                                 eps, out, mixed, rows, path, 1, 0, 0, nullptr, stream);
}

// KGM-4: out[0] = k2s::early_opt_ins(), the dynamic shared-memory raises of the early hc launches so far. With
// next_generation != 0 the current ordinal's context generation is first advanced, as a primary-context reset records
// it (cuda_device.cpp), so every cached opt-in (these, k2_fast's and p3b's) raises again on its next launch.
extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_early_opt_ins(int64_t next_generation,
                                                                                  int64_t *out, cudaStream_t) {
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    if (next_generation != 0) {
        int id = -1;
        if (cudaGetDevice(&id) != cudaSuccess || id < 0 || id >= cere::device::kTrackedDevices)
            return CERE_ERR_INTERNAL;
        cere::device::context_generation(id).fetch_add(1);
    }
    out[0] = static_cast<int64_t>(cere::kernels::k2s::early_opt_ins());
    return CERE_OK;
}

// K3Chain (k3chain design note KC-03): `chains` chains in a row (hc_chain: the weights are stacks of `chains` sets;
// the forward's successive mixes) as `timed` CUDA-graph replays (3 untimed first; cold != 0: an l2_flush before each,
// outside its span); the spans in µs into `us` (host float [timed]). Outputs unspecified.
extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_chain_us(
    const void *block_out, const void *hyper, void *normed, const void *w_inject, const void *next_norm,
    const void *w_down, const void *w_up, int64_t m, int64_t hc, int64_t hidden, int64_t lowrank, float eps, void *out,
    void *mixed, int64_t rows, int64_t path, int64_t chains, int64_t timed, int64_t cold, float *us,
    cudaStream_t stream) {
    if (timed < 1) return CERE_ERR_ARGUMENT;
    return cere::ktest::hc_chain(block_out, hyper, normed, w_inject, next_norm, w_down, w_up, m, hc, hidden, lowrank,
                                 eps, out, mixed, rows, path, chains, timed, cold, us, stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_combine(const void *block_out, const void *hyper,
                                                                            const void *normed, const void *w_inject,
                                                                            int64_t m, int64_t hc, int64_t hidden,
                                                                            void *out, cudaStream_t stream) {
    return cere::ktest::hc_combine(block_out, hyper, normed, w_inject, m, hc, hidden, out, stream);
}

// m23 design note P5R: hc_combine, then the next mix's per-stream norm (weight next_norm [hc * hidden]) of `out`
// into `normed` (read by the combine first), one launch at m <= 16. `out` may alias `hyper`.
extern "C" __attribute__((visibility("default"))) int cere_ktest_hc_combine_norm(
    const void *block_out, const void *hyper, void *normed, const void *w_inject, const void *next_norm, int64_t m,
    int64_t hc, int64_t hidden, float eps, void *out, cudaStream_t stream) {
    return cere::ktest::hc_combine_norm(block_out, hyper, normed, w_inject, next_norm, m, hc, hidden, eps, out,
                                        stream);
}
