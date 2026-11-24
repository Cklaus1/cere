// Track LoRA kernel-test entry point (executor/tests/gpu/test_kernels_lora.py, suite `kernels_lora`; kernels/lora.h):
// GemmPlan::run / gemm_segments with an adapter table next to the same plans without one. Device pointers, int64
// sizes (host arrays where named), the stream last; 0 = success, else a cere_status.
#include "kernels/lora.h"
#include "ktest/k2_ktest.h"
#include "ktest/l2_flush.h"

#include <cstdint>
#include <vector>

namespace {

namespace lora = cere::kernels::lora;
using cere::kernels::GemmOut;
using cere::kernels::GemmPlan;

bool dim_ok(int64_t v) { return v > 0 && v <= INT32_MAX; }

}  // namespace

// y = x · w^T (+ the active adapters) through `count` (1..3) BF16 weights w_ptrs[i] [ns[i], k] sharing x [m, k]
// (host int64 arrays w_ptrs, ns, y_ptrs; y_ptrs[i] BF16 [m, ns[i]]): count 1 runs plan.run, count > 1
// kernels::gemm_segments. Plans as the executor's: rows 0 an init plan of max_m = m (k2_ktest.h plan_init), rows in
// [1, GemmPlan::kMaxRows] (m <= rows) the rows plan init_rows(rows); small_m 1 / 0 use_small_m.
// Targets: packed 1 (count 1): the weight's columns split into `segments` (1..3) targets of seg_n (host int64,
// summing to ns[0]: q | k | v); packed 0: one target per weight. table 0: no table (the base call). Else a table of
// `adapters` slots at max rank `rank` (padded to 8 by the table) over those targets; slot s holds ranks[s] (0: empty)
// at scales[s] (host float) with host BF16 A a_ptrs[s * T + t] [ranks[s], k] and B b_ptrs[s * T + t] [n_t, ranks[s]]
// (0 / 0: the adapter leaves target t alone). per_row 0: every row runs slots[0] (-1: none); 1: row r runs slots[r]
// (host int64 [m], m <= 16). iters 1: one call; > 1: one untimed call then iters - 1 cold-L2 calls, us host double [2]
// = median and minimum µs per call.
extern "C" __attribute__((visibility("default"))) int cere_ktest_lora_plan(
    const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs, const int64_t *ns,
    const int64_t *y_ptrs, int64_t rows, int64_t small_m, int64_t packed, int64_t segments, const int64_t *seg_n,
    int64_t table, int64_t adapters, int64_t rank, const int64_t *ranks, const float *scales, const int64_t *a_ptrs,
    const int64_t *b_ptrs, int64_t per_row, const int64_t *slots, int64_t iters, void *us, cudaStream_t stream) {
    constexpr int64_t kMax = lora::kMaxSegments;
    if (!dim_ok(m) || !dim_ok(k) || count < 1 || count > kMax || w_ptrs == nullptr || ns == nullptr ||
        y_ptrs == nullptr || rows < 0 || rows > GemmPlan::kMaxRows || (rows > 0 && m > rows) || small_m < 0 ||
        small_m > 1 || packed < 0 || packed > 1 || (packed == 1 && (count != 1 || segments < 1 || segments > kMax ||
                                                                    seg_n == nullptr)) ||
        table < 0 || table > 1 || per_row < 0 || per_row > 1 || (per_row == 1 && m > int64_t(lora::kMaxRows)) ||
        iters < 1 || iters > 10'000 || (iters > 1 && us == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    const int64_t targets = packed == 1 ? segments : count;
    if (table == 1 && (adapters < 1 || adapters > 64 || rank < 1 || rank > 256 || ranks == nullptr ||
                       scales == nullptr || a_ptrs == nullptr || b_ptrs == nullptr || slots == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    GemmPlan instantiated[kMax];
    GemmPlan rows_plans[kMax];
    void *workspaces[kMax] = {};
    const GemmPlan *pp[kMax] = {};
    const __nv_bfloat16 *w[kMax] = {};
    void *y[kMax] = {};
    lora::Table lt;
    cere_status s = CERE_OK;
    for (int64_t i = 0; i < count && s == CERE_OK; ++i) {
        if (!dim_ok(ns[i])) {
            s = CERE_ERR_ARGUMENT;
            break;
        }
        if (rows == 0) {
            s = cere::ktest::plan_init(&instantiated[i], handle, uint32_t(ns[i]), uint32_t(k), uint32_t(m),
                                       GemmOut::kBF16, &workspaces[i], stream);
        } else if ((s = instantiated[i].init(handle, uint32_t(ns[i]), uint32_t(k), 1, GemmOut::kBF16)) == CERE_OK) {
            s = rows_plans[i].init_rows(handle, instantiated[i], uint32_t(rows));
        }
        if (s != CERE_OK) break;
        GemmPlan &plan = rows == 0 ? instantiated[i] : rows_plans[i];
        plan.use_small_m(small_m != 0);
        if (small_m != 0 && !plan.small_m()) s = CERE_ERR_UNSUPPORTED;
        if (table == 1) plan.use_lora(&lt);
        pp[i] = &plan;
        w[i] = reinterpret_cast<const __nv_bfloat16 *>(w_ptrs[i]);
        y[i] = reinterpret_cast<void *>(y_ptrs[i]);
    }
    if (s == CERE_OK && table == 1) {
        const uint32_t padded = uint32_t((rank + 7) / 8 * 8);
        std::vector<lora::TargetLayout> layouts;
        std::vector<lora::Entry> entries;
        uint64_t offset = 0;
        for (int64_t t = 0; t < targets && s == CERE_OK; ++t) {
            const int64_t n = packed == 1 ? seg_n[t] : ns[t];
            if (!dim_ok(n)) s = CERE_ERR_ARGUMENT;
            layouts.push_back(lora::TargetLayout{offset, offset + uint64_t{padded} * uint64_t(k), uint32_t(k),
                                                 uint32_t(n)});
            offset += uint64_t{padded} * uint64_t(k + n);
        }
        if (packed == 1) {
            lora::Entry e;
            e.w = w[0];
            e.k = uint32_t(k);
            e.segments = uint32_t(segments);
            uint32_t col = 0;
            for (int64_t t = 0; t < segments; ++t) {
                e.seg[t] = lora::Segment{uint32_t(t), col, layouts[size_t(t)].n};
                col += layouts[size_t(t)].n;
            }
            if (int64_t{col} != ns[0]) s = CERE_ERR_ARGUMENT;
            entries.push_back(e);
        } else {
            for (int64_t i = 0; i < count; ++i) {
                lora::Entry e;
                e.w = w[i];
                e.k = uint32_t(k);
                e.segments = 1;
                e.seg[0] = lora::Segment{uint32_t(i), 0, uint32_t(ns[i])};
                entries.push_back(e);
            }
        }
        if (s == CERE_OK) {
            s = lt.init(uint32_t(adapters), padded, uint32_t(m), std::move(layouts), std::move(entries));
        }
        for (int64_t a = 0; a < adapters && s == CERE_OK; ++a) {
            if (ranks[a] == 0) continue;
            if (ranks[a] < 0 || ranks[a] > rank) {
                s = CERE_ERR_ARGUMENT;
                break;
            }
            std::vector<uint32_t> index;
            std::vector<const void *> ap, bp;
            for (int64_t t = 0; t < targets; ++t) {
                const int64_t ai = a_ptrs[a * targets + t], bi = b_ptrs[a * targets + t];
                if (ai == 0 && bi == 0) continue;
                index.push_back(uint32_t(t));
                ap.push_back(reinterpret_cast<const void *>(ai));
                bp.push_back(reinterpret_cast<const void *>(bi));
            }
            s = lt.load(uint32_t(a), uint32_t(ranks[a]), scales[a], index.size(), index.data(), ap.data(), bp.data());
        }
        if (s == CERE_OK) {
            if (per_row == 1) {
                int32_t row_slots[lora::kMaxRows];
                for (int64_t r = 0; r < m; ++r) row_slots[r] = int32_t(slots[r]);
                lt.activate_rows(row_slots, uint32_t(m));
            } else {
                lt.activate(int32_t(slots[0]));
            }
        }
    }
    if (s == CERE_OK) {
        void *flush = nullptr;
        if (iters > 1 && cudaMallocAsync(&flush, cere::ktest::kL2FlushBytes, stream) != cudaSuccess) {
            flush = nullptr;
            s = CERE_ERR_CAPACITY;
        }
        const auto call = [&]() -> cere_status {
            const auto *xb = static_cast<const __nv_bfloat16 *>(x);
            return count == 1 ? pp[0]->run(xb, w[0], uint32_t(m), y[0], stream)
                              : cere::kernels::gemm_segments(pp, w, y, uint32_t(count), xb, uint32_t(m), stream);
        };
        if (s == CERE_OK) {
            s = iters == 1 ? call() : cere::ktest::cold_us(flush, iters - 1, static_cast<double *>(us), stream, call);
        }
        if (flush != nullptr && cudaFreeAsync(flush, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    }
    lt.deactivate();
    for (void *ws : workspaces) {
        if (ws != nullptr && cudaFreeAsync(ws, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    }
    // The table's pool is freed with lt: finish the stream's reads of it first.
    if (cudaStreamSynchronize(stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    return s;
}
