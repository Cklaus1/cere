// K2 (m22 design note): the cuBLASLt BF16 GEMM wrapper of the native executor and the last-row-only lm_head.
//
//   y[m, n] = x[m, k] · w[n, k]^T
//
// x: BF16, row-major [m, k] (leading dimension k); w: BF16 in the checkpoint (torch nn.Linear) layout [n, k];
// y: row-major [m, n] in BF16 or F32 (GemmOut); FP32 accumulation and FP32 alpha = 1, beta = 0.
//
// Determinism (spec "Model-specific kernel decisions", dense GEMM row): the algorithm is pinned when the plan is
// created (the CUDA device creates its plans at INSTANTIATE), never chosen at run time. M is split into power-of-two
// buckets (1, 2, 3-4, 5-8, ..., up to next_pow2(max_m)); each bucket pins the first cuBLASLt heuristic result for M =
// its upper bound, restricted to reduction scheme NONE (no split-K) and zero workspace (no split-K/stream-K partials,
// no counters, no atomics), and checked with cublasLtMatmulAlgoCheck at both ends of the bucket. The pinned
// configuration is verified to have split-K 1 and reduction scheme NONE. The same (n, k, out, m) therefore always runs
// the same kernel with the same reduction order: repeated calls are bit-identical. Buckets depend on m only (not on
// max_m), so every unpinned plan of one (n, k, out) shape runs the same algorithm for the same m.
//
// Prefill pins (m22p design note D7 lever 2): a plan may instead run, for listed M ranges (all M > 16, so every
// call with m <= 16 keeps its bucket algorithm and bits), an explicit cuBLASLt configuration (GemmPins) recorded from
// the algorithm torch's cuBLAS dispatches for the same GEMM in the pinned image. Such a configuration may split K
// with a separate, deterministic reduction kernel (reduction scheme COMPUTE_TYPE or OUTPUT_TYPE: partials in a
// workspace, no atomics); INPLACE (workspace counters) is refused as malformed pins. Its workspace is device memory the
// plan's owner passes at init (gemm_pins_workspace_bytes); the configuration is verified with cublasLtMatmulAlgoCheck
// at both ends of each range it serves. The pins name the cuBLASLt version they were recorded with
// (cublasLtGetVersion()); a plan refuses them under any other version, whose kernels and split order may differ even
// where AlgoCheck accepts the configuration. Bucket and pin selection depend on m only: one (n, k, out, pins) runs one
// configuration per m. Executor plans and kernel tests take their pins from gemm_pins_for.
//
// Bucket re-pins (r3-prefill design note, R3 PrefillCompute): inside the prefill bucket m 1,025-2,048 (the 1.2K and
// 18K cells' last chunks) seven plan shapes pin, over measured m ranges, a faster zero-workspace, split-K-free
// configuration that gives the bucket algorithm's bits (kGemmPinsTables from kFirstBucketRepin); prefill output is
// unchanged, only its time.
//
// Small-M path (m23 design note P11): selected per plan at run time by the k2_gemm slot (cere_slot_selection impl
// CERE_LEVER_K2_SMALL_M; model::apply_levers switches every plan of a model, at INSTANTIATE and on a hot swap): every
// call with m <= k2s::kMaxM (16) then runs the in-tree kernel of k2_small_m.h instead of cuBLASLt, so y[r, :] is the
// same bits for every m in [1, 16] (decode, batched rows and verify agree with single-row decode by construction) and
// m > 16 keeps the bucket and pin configurations above, bit for bit. init() leaves it off (the default lever,
// CERE_LEVER_K2_CUBLASLT: cuBLASLt for every m); kernel tests switch it per plan (use_small_m).
//
// W8A8 path (k2-prefill design note P3, the lever value CERE_LEVER_K2_SMALL_M_A8: small_m plus this; numerics-
// changing, opt-in): an MXFP8 weight on an init() plan with a row choice above k2s::kMaxM runs cuBLASLt's MXFP8 GEMM
// on an E4M3 copy of x (GemmPlan::init_a8 / use_a8) instead of the BF16 GEMM on the weight's upcast. Its algorithms
// are pinned per bucket the same atomics-free way, so it is deterministic and every m <= 16 keeps its bits.
//
// run() does no allocation, no heuristic query and no synchronization; it enqueues one cublasLtMatmul on `stream`
// (split-K configurations: cuBLASLt's GEMM kernel plus its split-K reduce kernel) or one small-M kernel (an MXFP8
// weight: a dequantize or, on the W8A8 path, the x quantize and scale repack kernels first).
#pragma once

#include "cere_executor.h"
#include "kernels/k2_fp8.h"
#include "kernels/k2_gemv.h"
#include "kernels/k2_mxfp8.h"
#include "kernels/k2_small_m.h"
#include "kernels/lora.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::kernels {

enum class GemmOut : uint32_t {
    kBF16 = 0,  // y is __nv_bfloat16 (one rounding of the FP32 accumulator)
    kF32 = 1,   // y is float (the FP32 accumulator)
};

// One cuBLASLt handle per CUDA device (cublasLtCreate); plans keep the handle they were created with.
cere_status gemm_handle_create(cublasLtHandle_t *out);
void gemm_handle_destroy(cublasLtHandle_t handle);

// One cuBLASLt configuration pinned by value: cublasLtMatmulAlgoInit(algo_id) with these attributes and every other
// configuration attribute 0 (CTA swizzling, inner and cluster shape).
struct GemmConfig {
    int32_t algo_id;
    cublasLtMatmulTile_t tile;
    cublasLtMatmulStages_t stages;
    int32_t splitk;                       // CUBLASLT_ALGO_CONFIG_SPLITK_NUM (1: no split-K)
    cublasLtReductionScheme_t reduction;  // NONE, COMPUTE_TYPE or OUTPUT_TYPE
    // Workspace of one call: `partials` * m * n * element bytes (COMPUTE_TYPE: 4, OUTPUT_TYPE: the output's);
    // 0 for NONE. Checked against cublasLtMatmulAlgoCheck at init.
    uint32_t partials;
    uint32_t custom_option = 0;  // CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
};

// `config` serves every m in [m_lo, m_hi].
struct GemmPinRange {
    uint32_t m_lo;
    uint32_t m_hi;
    GemmConfig config;
};

// The pinned configurations of one plan shape: ranges ascending, disjoint, every m_lo > kPinMinM - 1.
struct GemmPins {
    static constexpr uint32_t kPinMinM = 17;  // m <= 16 (decode, verify rows) never leaves its bucket algorithm
    uint32_t n;
    uint32_t k;
    GemmOut out;
    const GemmPinRange *ranges;
    uint32_t count;
    size_t cublaslt_version;  // cublasLtGetVersion() of the cuBLASLt the configurations were recorded with

    bool applies(uint32_t plan_n, uint32_t plan_k, GemmOut plan_out) const {
        return plan_n == n && plan_k == k && plan_out == out;
    }
};

// Device workspace bytes a plan with `pins` and max_m needs (the largest call it can run; 0 when none splits K).
size_t gemm_pins_workspace_bytes(const GemmPins &pins, uint32_t max_m);

// Lever 2 on the moe_shared_gate_up plan (n 640, k 2,560, BF16 out; model.h Workspace): SGLang runs the shared
// expert's gate and up projections as ONE F.linear over the merged [gate; up] weight [1,280, 2,560] (reduced-precision
// BF16 reduction allowed, torch's default). For m in [17, 4,096] these are the configurations torch 2.13's cuBLAS
// dispatches for x[m, 2,560] · W[1,280, 2,560]^T on GPU0 in the image (cuBLASLt 13.1.1, cublasLtGetVersion() 130,101),
// as recorded by `native_dev.py cublaslt --plan moe_shared_gate_up` (CUBLASLT_LOG_MASK=2 trace of every m; the report
// the table was checked against is named in k2_gemm.cpp and tests/test_k2_gemm_pins.py checks the table against it),
// except the m ranges where torch's pick is INPLACE (1,153-1,536 and 2,369-3,200), which keep the bucket algorithm.
// Each configuration computes every output element from k alone (tile and split boundaries along n do not change an
// element's reduction order), so running it on the gate and up weights separately (n 640 each) gives bit for bit the
// columns of the merged GEMM; executor/tests/gpu/test_kernels_k2_gemm.py checks that at both ends of every range.
extern const GemmPins kSharedGateUpPins;

// Lever 2 on the attn_qkv plan (n 13,312, k 2,560, BF16 out; m22p design note M22PS-R3): SGLang's qkv_proj
// (QKVParallelLinear) is ONE F.linear over the packed [q; k; v] weight, the native plan's own shape. For m in
// [17, 4,096] these are the configurations torch dispatches for x[m, 2,560] · W[13,312, 2,560]^T, recorded the same
// way (`native_dev.py cublaslt --plan attn_qkv`, the report named in k2_gemm.cpp), except the m ranges where torch's
// pick is INPLACE (65-96, 193-256, 385-512, 641-768, 897-1,024, 1,281-1,536 and 1,793-2,048), which keep the bucket
// algorithm.
extern const GemmPins kAttnQkvPins;

// Bucket re-pins (R3 PrefillCompute, r3-prefill design note; the tables in k2_gemm.cpp): per plan shape, one or two
// ranges inside the bucket m 1,025-2,048 whose configuration (reduction NONE, split-K 1, no workspace) is faster than
// the bucket algorithm there and gives its bits at every m measured (the lab's 2-seed sweep, every 32nd m;
// executor/tests/gpu/test_kernels_k2_gemm.py test_bucket_repins at both ends of each range and m 1,217 / 1,633 inside).
// Shapes: gdn_qkv and ple_key (10,240 x 2,560), gdn_z (6,144 x 2,560), gdn_out and attn_o (2,560 x 6,144), moe_router
// (512 x 2,560), moe_shared_down (2,560 x 640), all BF16; hc down (320 x 10,240) and up (10,240 x 320), F32.
// Every static pins table, in the order gemm_pins_for searches them: kSharedGateUpPins, kAttnQkvPins, then the bucket
// re-pins from index kFirstBucketRepin.
extern const GemmPins *const kGemmPinsTables[];
extern const uint32_t kGemmPinsTableCount;
constexpr uint32_t kFirstBucketRepin = 2;

// The pins of an executor plan with shape (n, k, out) (kGemmPinsTables), or nullptr. The executor (model/forward.cpp,
// model/model.cpp, k3_hc.cu) and the kernel tests select pins only through this function, so the tests run the
// executor's configurations. A plan of a pinned shape that the executor runs on the bucket algorithm passes no pins
// (attn_index_qk has moe_shared_gate_up's shape and keeps its buckets).
const GemmPins *gemm_pins_for(uint32_t n, uint32_t k, GemmOut out);

// The m = 1 customOption swap (m23 design note P11R-21): bucket 0's heuristic pick (m = 1, every single-row decode)
// is cuBLASLt's gemvx (algoId 13) in customOption 37 on the attention/GDN out and GDN z shapes; customOption 75, a
// later result of the same heuristic query, sums in the same order (k2v::order_of: T 32, S 1 at k 2,560, 6,144 and
// 10,240) in a faster kernel variant (K2 test_m1_swap reports both timings). Returns the customOption bucket 0 runs for
// the heuristic's pick (algo_id, custom_option) at inner dimension k under libcublasLt `cublaslt_version`: 75 where
// the pick is 37 and k2v identifies both variants as the same order for that version and k, else custom_option
// unchanged. init() then takes the heuristic's result of that algoId and customOption (atomics-free and
// AlgoCheck-accepted like any pick) and refuses the plan (CERE_ERR_UNSUPPORTED) when the query returned none. K2
// test_m1_swap checks the two variants bit for bit (BF16 and F32 outputs) on every decode shape the swap applies to.
uint32_t gemm_m1_custom_option(size_t cublaslt_version, int algo_id, uint32_t custom_option, uint32_t k);

class GemmPlan {
public:
    static constexpr uint32_t kMaxBuckets = 16;  // m up to 2^15 = 32,768
    static constexpr uint32_t kMaxPins = 32;

    // Pins one algorithm per M bucket for m in [1, max_m], and with `pins` (which must apply to (n, k, out)) one
    // configuration per pin range that starts at or below max_m. `pin_workspace` (device, 256-byte aligned) holds at
    // least gemm_pins_workspace_bytes(*pins, max_m) bytes and outlives the plan; it may be null when that is 0.
    // CERE_ERR_ARGUMENT for zero or oversized dimensions, pins that do not apply or are malformed (unordered or
    // overlapping ranges, a range below kPinMinM, a reduction scheme other than NONE/COMPUTE_TYPE/OUTPUT_TYPE such as
    // INPLACE, split-K and partials that disagree with the scheme), or a missing or short workspace;
    // CERE_ERR_UNSUPPORTED when cuBLASLt offers no atomics-free algorithm for a bucket, the running cuBLASLt is not
    // the pins' cublaslt_version, or cuBLASLt refuses a pinned configuration (AlgoInit or AlgoCheck failure, more
    // workspace than `partials` gives); CERE_ERR_INTERNAL for other cuBLASLt failures. Host only (no device work).
    // `m1_swap` false (kernel tests only) keeps bucket 0's heuristic pick as it is (no gemm_m1_custom_option swap):
    // the algorithm main ran before the swap, the reference the swap is checked against.
    cere_status init(cublasLtHandle_t handle, uint32_t n, uint32_t k, uint32_t max_m, GemmOut out,
                     const GemmPins *pins = nullptr, void *pin_workspace = nullptr,
                     size_t pin_workspace_bytes = 0, bool m1_swap = true);
    // y = x · w^T for 1 <= m <= max_m. Pointers must be device pointers aligned to the plan's alignment (16 bytes
    // satisfies every plan; cudaMalloc gives 256). CERE_ERR_ARGUMENT for a bad m or a misaligned pointer,
    // CERE_ERR_INTERNAL when cublasLtMatmul fails. `cls` (m210 design note D15⁗′; 0 = m): the rows of the call this
    // one stands for (m <= cls <= max_m), so every row-count choice (the FP8 kernel, small-M, the pin range, the
    // bucket) is cls's while the GEMM runs m rows; a rows plan ignores it.
    cere_status run(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, void *y, cudaStream_t stream,
                    uint32_t cls = 0) const;

    // Rows plans (m26 design note D6, I6): the plan of a batched decode step whose row r must be bit-equal to the
    // same row run alone. Copies `instantiated`'s shape, descriptor and its m = 1 algorithm (algo_[0], the one every
    // single-row decode of that (n, k, out) runs; no heuristic query) and runs cublasLtMatmulAlgoCheck with that
    // algorithm at every m in [2, rows], requiring SUCCESS and zero workspace. Each m that passes runs as one GEMM of m
    // rows with the m = 1 algorithm. Each m that fails runs as one launch of m rows on one of these `paths` that
    // serves it, every row bit-equal to the m = 1 call (m23 design note P11R-21): kRowsGemv, where that algorithm
    // is a cuBLASLt gemvx variant of a known summation order (k2v::order_of), one k2v::gemv launch in that order
    // (gemv_rows(m)); kRowsBatched, where AlgoCheck accepts the m = 1 algorithm itself with zero workspace on a
    // strided batch of m single-row GEMMs (BATCH_COUNT m, batch strides w 0, x k, y n: the rows buffers as they are),
    // that one cublasLtMatmul (batched_rows(m)). Where both serve, the faster one as measured per (n, k, out) and m
    // (k2_gemm.cpp rows_gemv_from: the rows kernel from a measured m on, the strided batch below it and on every
    // shape not measured). Otherwise (and only that m) it runs as m GEMMs of one row (per_row(m)), so every row is
    // summed in the single-row algorithm's order. The plan has max_m = rows, no pins (rows <= kMaxRows <
    // GemmPins::kPinMinM) and run()'s contract. `fused_allowed` (kernel tests and benches only; default all): bit
    // m - 1 clear makes m per-row even where AlgoCheck passes (the fallback path under test). `paths` (kernel tests
    // and benches only; default both): the one-launch paths init_rows may take; a path left out is never taken. The
    // small-M state is copied from `instantiated`; both paths' masks are built either way, so use_small_m may switch
    // the plan afterwards (with small-M on, every allowed m runs as one small-M launch). CERE_ERR_ARGUMENT for a null
    // handle, an uninitialized `instantiated`, rows outside [1, kMaxRows] or `paths` outside kRowsAllPaths;
    // CERE_ERR_INTERNAL for a cuBLASLt layout or algorithm attribute failure. Host only.
    static constexpr uint32_t kMaxRows = 16;
    static constexpr uint32_t kRowsGemv = 1;
    static constexpr uint32_t kRowsBatched = 2;
    static constexpr uint32_t kRowsAllPaths = kRowsGemv | kRowsBatched;
    cere_status init_rows(cublasLtHandle_t handle, const GemmPlan &instantiated, uint32_t rows,
                          uint32_t fused_allowed = ~uint32_t{0}, uint32_t paths = kRowsAllPaths);
    // A rows plan whose m-row call runs as m single-row calls (cuBLASLt: AlgoCheck refused the m = 1 algorithm at m
    // and no one-launch path serves it; either path: fused_allowed cleared m). False for m = 1, m outside
    // [1, max_m] and every plan made by init().
    bool per_row(uint32_t m) const;
    // A rows plan whose m-row call (cuBLASLt path) runs as one k2v::gemv launch in the m = 1 algorithm's order.
    bool gemv_rows(uint32_t m) const {
        return !small_m_ && m >= 2 && m <= max_m_ && (rows_gemv_ & (uint32_t{1} << (m - 1))) != 0;
    }
    // A rows plan whose m-row call (cuBLASLt path) runs as one strided-batched cublasLtMatmul of m single-row GEMMs
    // with the m = 1 algorithm.
    bool batched_rows(uint32_t m) const {
        return !small_m_ && m >= 2 && m <= max_m_ && (rows_batched_ & (uint32_t{1} << (m - 1))) != 0;
    }
    // The id and customOption of the plan's m = 1 cuBLASLt algorithm (bucket 0), the pairing k2v::order_of maps to a
    // gemvx summation order. CERE_ERR_ARGUMENT for an uninitialized plan, CERE_ERR_INTERNAL when cuBLASLt refuses the
    // attribute read.
    cere_status m1_algo(int *algo_id, uint32_t *custom_option) const;
    // True when m <= k2s::kMaxM runs the small-M kernel (switched on by use_small_m where k2s::supported(k)).
    bool small_m() const { return small_m_; }
    // The k2_gemm lever's switch (model::apply_levers; kernel tests per plan): `on` false runs every m on cuBLASLt
    // (init()'s state, the default lever); true runs m <= k2s::kMaxM on the small-M kernel where the plan's k allows
    // it. A plain bool read by run(): the hot path does no lookup. Hot-safe (cere_model_set_selection between
    // requests): the small-M kernel takes no workspace (its split-K partials stay in shared memory/cluster DSMEM,
    // k2_small_m.h) and the batched lm_head's rows buffer (model.h Workspace::rows_logits) is carved whatever the
    // lever.
    void use_small_m(bool on) { small_m_ = on && k2s::supported(k_); }
    // Early weights (BeatSglang S4, k2_small_m.h): `on` runs the plan's small-M launches as k2s::gemm's early launch
    // (the first round of w issued before the dependency wait; the same bits). Only for a plan whose w is a model
    // weight that no kernel before it on the stream writes (the MoE router); init() clears it, init_rows copies it.
    void use_early_weights(bool on) { early_ = on; }
    // The k2_weights lever (track Fp8W, model::apply_levers): `table` lists the weights INSTANTIATE converted to FP8
    // (kernels/k2_fp8.h; null: none). run() looks w up there (host, sorted vector): a converted w of this plan's (n,
    // k) runs k2f::gemm for m <= k2f::kMaxM (whatever small_m and the rows path say: its rows are batch invariant),
    // else k2f::dequantize into table->staging() and this plan's BF16 path on that copy. Any other w (the MTP
    // layer's BF16 weights on the shared plans) runs as before.
    void use_fp8(const k2f::Table *table) { fp8_ = table; }
    const k2f::Table *fp8() const { return fp8_; }
    // MXFP8 weights (track Fp4S3, kernels/k2_mxfp8.h; set by the model binder when the artifact carries them; null:
    // none). run() looks w up there (host, sorted vector): an MXFP8 w of this plan's (n, k) returns the bits the same
    // call returns on its BF16 upcast: where the BF16 call would run the small-M kernel, the small-M kernel's MXFP8
    // operand variant (same tiles, split and reduction order); on every other path k2m::dequantize into
    // table->staging() and this plan's BF16 path on that copy. Any other w (the MTP layer's BF16 weights on the shared
    // plans) runs as before. Independent of use_fp8 (the model refuses k2_weights = fp8 on an MXFP8 artifact).
    void use_mxfp8(const k2m::Table *table) { mxfp8_ = table; }
    const k2m::Table *mxfp8() const { return mxfp8_; }
    // LoRA adapters (track LoRA, kernels/lora.h; set by model::apply_levers when the model reserved a pool; null:
    // none). While the table is active, run() follows the base GEMM with table->apply on a w the table registers;
    // inactive or a w it misses, run() is the base call, bit for bit. gemm_segments applies it per segment.
    void use_lora(lora::Table *table) { lora_ = table; }
    lora::Table *lora() const { return lora_; }
    // The k2_gemm lever's small_m_a8 (k2-prefill design note P3; numerics-changing, opt-in profile -v56): MXFP8
    // W8A8 for an MXFP8 weight (use_mxfp8's table) on an init() plan when the row choice (cls, else m) exceeds
    // k2s::kMaxM. init_a8 (model INSTANTIATE for the init() plans with BF16 output whose (n, k) is an MXFP8 table
    // entry, whatever the lever; kernel tests) pins per bucket above k2s::kMaxM up to max_m cuBLASLt's first heuristic
    // algorithm for D BF16 [m, n] = E4M3 x [m, k] times E4M3 w [n, k]^T with VEC32_UE8M0 scales on both operands
    // (FP32 compute), under init()'s atomics-free rule (zero workspace, no split-K, reduction NONE, AlgoCheck at both
    // bucket ends). CERE_ERR_ARGUMENT for a plan init() did not make, a rows plan or F32 output; CERE_ERR_UNSUPPORTED
    // when k is no multiple of 32, max_m <= k2s::kMaxM, or cuBLASLt offers no such algorithm for some bucket (or
    // refuses the FP8 descriptor or layouts): the plan then never takes the path (a8_ready() false). Host only.
    cere_status init_a8();
    bool a8_ready() const { return a8_ready_; }
    // The lever's switch (model::apply_levers with use_small_m; kernel tests): on (and a8_ready()), run() on an MXFP8
    // w of this plan's shape with a row choice above k2s::kMaxM writes x's E4M3 copy and its blocked scales
    // (k2m::quantize_x) and the weight's blocked scales (k2m::repack_scales) into the table's staging buffer, then
    // makes one cublasLtMatmul with the bucket's algorithm; CERE_ERR_UNSUPPORTED when the staging buffer cannot hold
    // those operands (a8_staging_bytes). Every other call is unchanged. Hot-safe: no allocation, the algorithms are
    // host state from INSTANTIATE and the staging buffer exists for every MXFP8 model.
    void use_a8(bool on) { a8_ = on && a8_ready_; }
    bool a8() const { return a8_; }
    // Staging bytes the W8A8 path takes for m rows on an [n, k] weight (k a multiple of 32): x's E4M3 copy, x's and
    // the weight's blocked scales, each starting 256-byte aligned.
    static size_t a8_staging_bytes(uint32_t m, uint32_t n, uint32_t k);

    uint32_t n() const { return n_; }
    uint32_t k() const { return k_; }
    uint32_t max_m() const { return max_m_; }
    GemmOut out() const { return out_; }
    // The pinned bucket algorithms (prefill selection signature, model/prefill_signature.h): sets *algos to the plan's
    // per-bucket algorithms (index b: the heuristic's pick for m = 2^b, serving m in (2^(b-1), 2^b]) and returns
    // their count (0 for an uninitialized plan). A const read only.
    uint32_t bucket_algos(const cublasLtMatmulAlgo_t **algos) const {
        *algos = algo_;
        return buckets_;
    }

private:
    cublasLtHandle_t handle_ = nullptr;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t max_m_ = 0;
    GemmOut out_ = GemmOut::kBF16;
    uint32_t align_x_ = 16;  // required byte alignment of x, w and y
    uint32_t align_w_ = 16;
    uint32_t align_y_ = 16;
    uint32_t buckets_ = 0;
    cublasLtMatmulDescOpaque_t desc_{};
    cublasLtMatmulAlgo_t algo_[kMaxBuckets]{};
    // Pin ranges of `pins` that start at or below max_m (pin_algo_[i] serves pin_range_[i]).
    uint32_t pin_count_ = 0;
    const GemmPinRange *pin_range_ = nullptr;
    cublasLtMatmulAlgo_t pin_algo_[kMaxPins]{};
    void *workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    // Rows plans: bit m - 1 set = m rows run as one GEMM (AlgoCheck passed at m; bit 0 always set); 0 for init() plans.
    // rows_allowed_: bit m - 1 set = fused_allowed allows m (the mask while small_m_ is on: one small-M launch).
    uint32_t rows_fused_ = 0;
    uint32_t rows_allowed_ = 0;
    bool small_m_ = false;
    bool early_ = false;
    // Rows plans: the bits of rows_fused_ whose m runs as one k2v::gemv launch (AlgoCheck refused m), in gemv_order_.
    uint32_t rows_gemv_ = 0;
    k2v::Order gemv_order_{};
    // Rows plans: the bits of rows_fused_ whose m runs as one strided-batched cublasLtMatmul (AlgoCheck refused m on
    // the rows layout and accepted it on the batched one; not rows_gemv_).
    uint32_t rows_batched_ = 0;
    // The fused mask of the active path (the k2_gemm lever may switch small_m_ after init_rows).
    uint32_t rows_mask() const { return small_m_ ? rows_allowed_ : rows_fused_; }
    const k2f::Table *fp8_ = nullptr;
    const k2m::Table *mxfp8_ = nullptr;
    lora::Table *lora_ = nullptr;
    // run() without the LoRA pass.
    cere_status run_base(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, void *y, cudaStream_t stream,
                         uint32_t cls) const;
    // W8A8 (init_a8 / use_a8): algo8_[b] serves bucket b for every bucket above k2s::kMaxM (bucket_lower(b) > 16).
    bool a8_ready_ = false;
    bool a8_ = false;
    uint32_t align_q_ = 16;  // byte alignment of the E4M3 x and w (leading dimension k bytes)
    cublasLtMatmulAlgo_t algo8_[kMaxBuckets]{};
    cere_status run_fp8(const k2f::Entry &e, const __nv_bfloat16 *x, uint32_t m, void *y, cudaStream_t stream,
                        uint32_t cls) const;
    cere_status run_mxfp8(const k2m::Entry &e, const __nv_bfloat16 *x, uint32_t m, void *y, cudaStream_t stream,
                          uint32_t cls) const;
    // The W8A8 GEMM of an MXFP8 weight on x's E4M3 copy `xq` and its blocked scales `xs` (staging, quantize_x),
    // with the weight's blocked scales written to `ws` first (the row choice sel > k2s::kMaxM).
    cere_status matmul_a8(const k2m::Entry &e, const uint8_t *xq, const uint8_t *xs, uint8_t *ws, uint32_t m,
                          void *y, cudaStream_t stream, uint32_t sel) const;
    // The MXFP8 entry of w when run(x, w, m, y, stream, cls) takes the W8A8 path with valid arguments, else nullptr
    // (gemm_segments' shared quantization).
    const k2m::Entry *a8_entry(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, const void *y,
                               uint32_t cls) const;
    friend cere_status gemm_segments(const GemmPlan *const *plans, const __nv_bfloat16 *const *w, void *const *y,
                                     uint32_t count, const __nv_bfloat16 *x, uint32_t m, cudaStream_t stream,
                                     uint32_t cls);
};

// The last-row-only lm_head: logits[vocab] = hidden[m - 1, :] · w^T, where hidden is row-major [m, k] and w is
// lm_head.weight [vocab, k]. Only row m - 1 is read; it runs `plan` with m = 1, so prefill (m > 1) and decode (m = 1)
// produce bit-identical logits for the same hidden row. `plan` has n = vocab and the logits dtype (the reference keeps
// BF16 logits: torch.matmul in BF16, sglang/srt/layers/logits_processor.py:740-742).
cere_status lm_head_last_row(const GemmPlan &plan, const __nv_bfloat16 *hidden, uint32_t m, const __nv_bfloat16 *w,
                             void *logits, cudaStream_t stream);

// y_i = x · w_i^T for `count` plans of the same k and output type that share x (GDN's in-projections), in one
// small-M launch when every plan runs the small-M kernel and m <= k2s::kMaxM (count <= k2s::kMaxSegments), else as
// count run() calls, except that two or more segments on the W8A8 path (GemmPlan::use_a8; plans sharing one MXFP8
// table) share one k2m::quantize_x of x, after the other segments ran. Each y_i is bit-equal to plans[i]->run(x, w[i],
// m, y[i], stream, cls). Arguments as run().
cere_status gemm_segments(const GemmPlan *const *plans, const __nv_bfloat16 *const *w, void *const *y, uint32_t count,
                          const __nv_bfloat16 *x, uint32_t m, cudaStream_t stream, uint32_t cls = 0);

}  // namespace cere::kernels
