// P3b MoE block (p3b_moe.h): K7's router, FlashInfer's CUTLASS fused-MoE launches up to GEMM2 for the routed
// experts (p3b_rows.h; pinned tactic pair, fused finalize off), K7's combine on the per-slot expert rows (MoeTaps
// routed_rows: finalizeMoeRoutingKernel's arithmetic) with the shared expert. routed_experts keeps FlashInfer's
// whole runMoe (finalize included) as the reference the ktest compares the block against.
//
// math_reference: flashinfer
// FlashInfer 0.6.17 csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu (FusedMoeRunner: the
// runner's construction, `use_fused_finalize_` before getTactics, the default tactics = the front of each list,
// runMoe's arguments, the NVFP4 QuantParams::FP4 mapping of `quant_scales`, workspace = getWorkspaceSize + the
// source-to-destination map, each 128-byte aligned).
// math_reference: sglang
// SGLang python/sglang/srt/layers/quantization/utils.py (`swizzle_blockscale`), python/sglang/srt/models/qwen2_moe.py
// (`fused_gate_sigmoid_mul_add` onto the finalized routed output).
#include "p3b/p3b_moe.h"

#include "fault/fault_inject.h"

#include "moe_kernels.h"
#include "p3b/p3b_rows.h"
#include "tensorrt_llm/common/envUtils.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace cere::p3b {
namespace {

namespace tk = tensorrt_llm::kernels::cutlass_kernels;
namespace ce = tensorrt_llm::cutlass_extensions;

constexpr size_t kAlign = 256;
size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

Tactic to_tactic(const ce::CutlassGemmConfig &c) {
    // CutlassTileConfigSM120 values are shape_tuple_to_enum(m, n, k) = m * 10^6 + n * 10^3 + k (gemm_configs.h).
    const int v = static_cast<int>(c.tile_config_sm120);
    return Tactic{v / 1000000, (v / 1000) % 1000, v % 1000, c.swap_ab,
                  c.epilogue_fusion_type == ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE};
}

// One runner per device ordinal (m32n design note D4): CutlassMoeFCRunner queries the device at construction and
// runMoe stores per-call workspace pointers in the object, so a device's runner serves every layer on that device and
// its calls are serialized by its own `mutex`. Two executors on two cards never share one: `device` is the ordinal the
// runner was built on, and bind() refuses a runner whose ordinal is not the current device's.
struct Runner {
    std::unique_ptr<FcRunner> fc;
    std::unique_ptr<GemmRunner> gemm;  // routed_rows' grouped GEMMs (the same MoeGemmRunner type runMoe uses)
    std::vector<ce::CutlassGemmConfig> g1, g2;
    TacticIndices pinned[kPinnedBucketCount];  // per bucket of kPinnedBuckets
    cere_status status = CERE_ERR_INTERNAL;
    std::atomic<int> device{-1};  // set once the build is done
    std::once_flag once;
    std::mutex mutex;
};

// The CUDA ordinals a process may hold a runner for (the deployment uses two).
constexpr int kMaxDevices = 16;

int current_device() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return device;
}

// The runner slot of `ordinal` (null outside [0, kMaxDevices)); built only by bind() on that device.
Runner *runner_of(int ordinal) {
    static Runner runners[kMaxDevices];
    return ordinal >= 0 && ordinal < kMaxDevices ? &runners[ordinal] : nullptr;
}

cere_status build(Runner &r);

// The runner slot of `ordinal` in `*out` and its status on `device` (the current device): built on first use when
// `ordinal` is `device` (once; the first result is kept); CERE_ERR_INTERNAL for a missing slot or a runner of another
// ordinal than `device` (D4: a runner is never shared across cards).
cere_status bind(int ordinal, int device, Runner **out) {
    Runner *r = runner_of(ordinal);
    *out = r;
    if (r == nullptr || device < 0) {
        return CERE_ERR_INTERNAL;
    }
    if (ordinal == device) {
        std::call_once(r->once, [r, device] {
            r->status = build(*r);
            r->device.store(device);
        });
    }
    return r->device.load() == device ? r->status : CERE_ERR_INTERNAL;
}

// The current device's runner in `*out` and its status.
cere_status acquire(Runner **out) {
    const int device = current_device();
    return bind(device, device, out);
}

cere_status build(Runner &r) {
    try {
        r.fc = std::make_unique<FcRunner>();
        r.gemm = std::make_unique<GemmRunner>();
        // Must precede getTactics: it gates the GEMM2 finalize-fusion tactics and the workspace sizing.
        r.fc->use_fused_finalize_ = false;
        r.g1 = r.fc->getTactics(tk::MoeGemmId::GEMM_1);
        r.g2 = r.fc->getTactics(tk::MoeGemmId::GEMM_2);
    } catch (const std::exception &) {
        return CERE_ERR_INTERNAL;
    }
    std::vector<Tactic> t1, t2;
    for (const auto &c : r.g1) t1.push_back(to_tactic(c));
    for (const auto &c : r.g2) t2.push_back(to_tactic(c));
    return resolve_pins(t1.data(), t1.size(), t2.data(), t2.size(), kPinnedBuckets, kPinnedBucketCount, r.pinned);
}

size_t runner_workspace(Runner &r, const kernels::MoeDims &d, uint32_t m) {
    try {
        return r.fc->getWorkspaceSize(m, d.hidden, d.inter, static_cast<int>(d.num_experts),
                                      static_cast<int>(d.top_k), tk::ActivationType::Swiglu,
                                      tk::MOEParallelismConfig{}, /*use_lora=*/false,
                                      /*use_deepseek_fp8_block_scale=*/false, /*use_mxfp8_act_scaling=*/false,
                                      /*min_latency_mode=*/false, /*use_awq=*/false);
    } catch (const std::exception &) {
        return 0;
    }
}

RowsProblem rows_problem(const kernels::MoeDims &d, uint32_t m) {
    return RowsProblem{m, d.hidden, d.inter, d.num_experts, d.top_k};
}

// The NVFP4 QuantParams of a layer in the P3b arena layout (act_scales = [fc1 input, fc2 input] global scales).
tk::QuantParams fp4_quant(const kernels::MoeLayer &L) {
    using Sf = tk::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF;
    const float *a = L.act_scales;
    return tk::QuantParams::FP4(a, reinterpret_cast<const Sf *>(L.up_scale), L.g1_alphas, a + 1,
                                reinterpret_cast<const Sf *>(L.down_scale), L.g2_alphas);
}

// Workspace bytes of the routed experts: routed_experts' (the runner's workspace, then the source-to-destination map)
// or routed_rows', whichever is larger (the two share one region).
size_t routed_bytes(Runner &r, const kernels::MoeDims &d, uint32_t m) {
    const size_t runner_bytes = runner_workspace(r, d, m);
    if (runner_bytes == 0) {
        return 0;
    }
    const size_t rows_bytes = rows_workspace_bytes(*r.gemm, rows_problem(d, m));
    if (rows_bytes == 0) {
        return 0;
    }
    return std::max(align_up(runner_bytes) + align_up(sizeof(int32_t) * m * d.top_k), rows_bytes);
}

struct Workspace {
    void *k7;               // K7's workspace (shared expert and combine on the routed_in path)
    size_t k7_bytes;
    __nv_bfloat16 *logits;  // [m, E]
    int32_t *topk_ids;      // [m, K]
    float *topk_weights;    // [m, K]
    void *experts;          // routed_bytes(m): routed_rows' workspace (routed_experts': the runner's and its map)
    size_t experts_bytes;
    size_t bytes;
};

Workspace carve(Runner &r, const kernels::MoeDims &d, uint32_t m, uint8_t *base) {
    Workspace w{};
    size_t off = 0;
    auto take = [&](size_t bytes) {
        uint8_t *p = base == nullptr ? nullptr : base + off;
        off += align_up(bytes);
        return p;
    };
    w.k7_bytes = kernels::moe_workspace_bytes(d, m);
    w.k7 = take(w.k7_bytes);
    w.logits = reinterpret_cast<__nv_bfloat16 *>(take(sizeof(__nv_bfloat16) * m * d.num_experts));
    w.topk_ids = reinterpret_cast<int32_t *>(take(sizeof(int32_t) * m * d.top_k));
    w.topk_weights = reinterpret_cast<float *>(take(sizeof(float) * m * d.top_k));
    w.experts_bytes = routed_bytes(r, d, m);
    w.experts = take(w.experts_bytes);
    w.bytes = off;
    return w;
}

// One thread per 4-byte group (k4 is contiguous on both sides): in [b][mt][r4][r32][kt][k4] -> out
// [b][mt][kt][r32][r4][k4].
__global__ void swizzle_kernel(const uint32_t *__restrict__ in, uint32_t *__restrict__ out, int64_t groups,
                               int rows, int col_groups) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= groups) {
        return;
    }
    const int64_t per = int64_t{rows} * col_groups;
    const int64_t b = i / per;
    const int64_t rem = i % per;
    const int row = static_cast<int>(rem / col_groups);
    const int kt = static_cast<int>(rem % col_groups);
    const int mt = row / 128, r4 = (row % 128) / 32, r32 = row % 32;
    const int64_t o = b * per + ((((int64_t{mt} * col_groups + kt) * 32 + r32) * 4) + r4);
    out[o] = in[i];
}

}  // namespace

cere_status instantiate() {
    Runner *r = nullptr;
    return acquire(&r);
}

cere_status runner_binding(int ordinal, int *device) {
    Runner *r = nullptr;
    const cere_status s = bind(ordinal, current_device(), &r);
    if (device != nullptr) {
        *device = r == nullptr ? -1 : r->device.load();
    }
    return s;
}

cere_status tactic_lists(Tactic *g1, size_t *n1, Tactic *g2, size_t *n2, size_t cap) {
    Runner *rp = nullptr;
    if (const cere_status s = acquire(&rp); s != CERE_OK) {
        return s;
    }
    const Runner &r = *rp;
    if (n1 == nullptr || n2 == nullptr || (cap > 0 && (g1 == nullptr || g2 == nullptr))) {
        return CERE_ERR_ARGUMENT;
    }
    *n1 = r.g1.size();
    *n2 = r.g2.size();
    for (size_t i = 0; i < cap && i < r.g1.size(); ++i) g1[i] = to_tactic(r.g1[i]);
    for (size_t i = 0; i < cap && i < r.g2.size(); ++i) g2[i] = to_tactic(r.g2[i]);
    return CERE_OK;
}

cere_status pinned_indices(uint32_t m, TacticIndices *pair) {
    Runner *rp = nullptr;
    if (const cere_status s = acquire(&rp); s != CERE_OK) {
        return s;
    }
    const Runner &r = *rp;
    const size_t bucket = bucket_of(kPinnedBuckets, kPinnedBucketCount, m);
    if (pair == nullptr || bucket == kPinnedBucketCount) {
        return CERE_ERR_ARGUMENT;
    }
    *pair = r.pinned[bucket];
    return CERE_OK;
}

cere_status swizzle_scales(uint8_t *scales, uint64_t batches, uint32_t rows, uint32_t cols, void *scratch,
                           size_t scratch_bytes, cudaStream_t stream) {
    if (scales == nullptr || scratch == nullptr || rows == 0 || cols == 0 || batches == 0 ||
        reinterpret_cast<uintptr_t>(scales) % 4 != 0 || reinterpret_cast<uintptr_t>(scratch) % kAlign != 0) {
        return CERE_ERR_ARGUMENT;
    }
    if (rows % 128 != 0 || cols % 4 != 0) {
        return CERE_ERR_UNSUPPORTED;
    }
    const size_t matrix = size_t{rows} * cols;
    const uint64_t per_chunk = scratch_bytes / matrix;
    if (per_chunk == 0) {
        return CERE_ERR_UNSUPPORTED;
    }
    for (uint64_t b = 0; b < batches; b += per_chunk) {
        const uint64_t n = batches - b < per_chunk ? batches - b : per_chunk;
        uint8_t *at = scales + b * matrix;
        if (cudaMemcpyAsync(scratch, at, n * matrix, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
        const int64_t groups = static_cast<int64_t>(n * matrix / 4);
        swizzle_kernel<<<static_cast<unsigned>((groups + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint32_t *>(scratch), reinterpret_cast<uint32_t *>(at), groups,
            static_cast<int>(rows), static_cast<int>(cols / 4));
        if (cudaGetLastError() != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
    }
    return CERE_OK;
}

cere_status prepare_layer(const kernels::MoeDims &d, const kernels::MoeLayer &L, void *scratch, size_t scratch_bytes,
                          cudaStream_t stream) {
    if (L.up_scale == nullptr || L.down_scale == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cere_status s = swizzle_scales(const_cast<uint8_t *>(L.up_scale), d.num_experts, 2 * d.inter, d.hidden / 16,
                                   scratch, scratch_bytes, stream);
    if (s == CERE_OK) {
        s = swizzle_scales(const_cast<uint8_t *>(L.down_scale), d.num_experts, d.hidden, d.inter / 16, scratch,
                           scratch_bytes, stream);
    }
    return s;
}

size_t moe_workspace_bytes(const kernels::MoeDims &d, uint32_t max_m) {
    Runner *r = nullptr;
    if (acquire(&r) != CERE_OK || kernels::moe_workspace_bytes(d, max_m) == 0 ||
        runner_workspace(*r, d, max_m) == 0) {
        return 0;
    }
    return carve(*r, d, max_m, nullptr).bytes;
}

size_t routed_workspace_bytes(const kernels::MoeDims &d, uint32_t max_m) {
    Runner *r = nullptr;
    if (acquire(&r) != CERE_OK || d.top_k == 0) {
        return 0;
    }
    return routed_bytes(*r, d, max_m);
}

cere_status routed_experts(const kernels::MoeDims &d, const kernels::MoeLayer &L, const __nv_bfloat16 *x,
                           const int32_t *topk_ids, const float *topk_weights, uint32_t m, __nv_bfloat16 *out,
                           void *workspace, size_t workspace_bytes, cudaStream_t stream, const TacticIndices *tactics) {
    Runner *rp = nullptr;
    cere_status s = acquire(&rp);
    if (s != CERE_OK) {
        return s;
    }
    if (m == 0 || x == nullptr || topk_ids == nullptr || topk_weights == nullptr || out == nullptr ||
        workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || L.up_weight == nullptr ||
        L.up_scale == nullptr || L.down_weight == nullptr || L.down_scale == nullptr || L.g1_alphas == nullptr ||
        L.g2_alphas == nullptr || L.act_scales == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    Runner &r = *rp;
    const size_t runner_bytes = runner_workspace(r, d, m);
    const size_t need = routed_bytes(r, d, m);
    if (need == 0 || need > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    // m > 0 here, so its bucket exists (the last one is open-ended).
    const TacticIndices pair =
        tactics == nullptr ? r.pinned[bucket_of(kPinnedBuckets, kPinnedBucketCount, m)] : *tactics;
    if (pair.gemm1 >= r.g1.size() || pair.gemm2 >= r.g2.size()) {
        return CERE_ERR_ARGUMENT;
    }
    char *runner_ws = static_cast<char *>(workspace);
    int32_t *src_to_dest = reinterpret_cast<int32_t *>(runner_ws + align_up(runner_bytes));
    const tk::QuantParams quant = fp4_quant(L);
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        try {
            r.fc->setTactic(r.g1[pair.gemm1], r.g2[pair.gemm2]);
            tensorrt_llm::kernels::LoraParams lora{};
            tk::MoeMinLatencyParams min_latency{};
            r.fc->runMoe(x, /*input_sf=*/nullptr, /*swizzled_input_sf=*/true, topk_ids, topk_weights, L.up_weight,
                         /*fc1_expert_biases=*/nullptr, tk::ActivationParams(tk::ActivationType::Swiglu),
                         L.down_weight, /*fc2_expert_biases=*/nullptr, quant, m, d.hidden, d.hidden, d.inter,
                         static_cast<int>(d.num_experts), static_cast<int>(d.top_k), runner_ws, out, src_to_dest,
                         tk::MOEParallelismConfig{}, /*enable_alltoall=*/false, /*use_lora=*/false, lora,
                         /*use_deepseek_fp8_block_scale=*/false, /*use_mxfp8_act_scaling=*/false,
                         /*min_latency_mode=*/false, min_latency, /*enable_pdl=*/true, stream);
        } catch (const std::exception &) {
            return CERE_ERR_INTERNAL;
        }
    }
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status routed_rows_view(const kernels::MoeDims &d, const kernels::MoeLayer &L, const __nv_bfloat16 *x,
                             const int32_t *topk_ids, const float *topk_weights, uint32_t m, RowsPath path,
                             decode::Order order, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                             Rows *out) {
    Runner *rp = nullptr;
    cere_status s = acquire(&rp);
    if (s != CERE_OK) {
        return s;
    }
    if (m == 0 || L.up_scale == nullptr || L.down_scale == nullptr || L.g1_alphas == nullptr ||
        L.g2_alphas == nullptr || L.act_scales == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    Runner &r = *rp;
    const TacticIndices pair = r.pinned[bucket_of(kPinnedBuckets, kPinnedBucketCount, m)];
    RowsProblem p = rows_problem(d, m);
    p.path = path;
    p.order = order;
    std::lock_guard<std::mutex> lock(r.mutex);
    return routed_rows(*r.fc, *r.gemm, r.g1[pair.gemm1], r.g2[pair.gemm2], p, x, topk_ids, topk_weights,
                       L.up_weight, L.down_weight, fp4_quant(L), workspace, workspace_bytes, stream, out);
}

bool decode_routed(const kernels::MoeDims &d, uint32_t m) { return rows_decoded(rows_problem(d, m)); }

cere_status decode_gemm1(const kernels::MoeDims &d, const kernels::MoeLayer &L, uint32_t m,
                         const decode::Group *groups, const uint8_t *x, const uint8_t *x_sf, uint8_t *act,
                         uint8_t *act_sf, decode::Order order, cudaStream_t stream) {
    const decode::Shape shape{m, d.hidden, d.inter, d.num_experts, d.top_k};
    if (L.act_scales == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    return decode::gemm1_act(groups, x, x_sf, L.up_weight, L.up_scale, L.g1_alphas, /*quant_scale=*/nullptr,
                             L.act_scales + 1, /*per_expert_act_scale=*/false,
                             tensorrt_llm::common::getEnvDisableFP4QuantFastMath(), act, act_sf, /*ready=*/nullptr,
                             shape, order, /*pdl=*/true, stream);
}

cere_status decode_gemm2(const kernels::MoeDims &d, const kernels::MoeLayer &L, uint32_t m,
                         const decode::Group *groups, const uint8_t *act, const uint8_t *act_sf, __nv_bfloat16 *out,
                         decode::Order order, cudaStream_t stream) {
    const decode::Shape shape{m, d.hidden, d.inter, d.num_experts, d.top_k};
    return decode::gemm2(groups, act, act_sf, L.down_weight, L.down_scale, L.g2_alphas, out, /*ready=*/nullptr, shape,
                         order, /*pdl=*/true, stream);
}

cere_status moe_block(const kernels::MoeDims &d, const kernels::MoeLayer &L, const kernels::MoePlans &plans,
                      const __nv_bfloat16 *x, uint32_t m, __nv_bfloat16 *out, void *workspace, size_t workspace_bytes,
                      cudaStream_t stream, const kernels::MoeTaps *taps, int32_t *topk_ids, float *topk_weights,
                      const TacticIndices *tactics) {
    Runner *rp = nullptr;
    cere_status s = acquire(&rp);
    if (s != CERE_OK) {
        return s;
    }
    if (m == 0 || x == nullptr || out == nullptr || workspace == nullptr ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || plans.router == nullptr || L.router == nullptr ||
        L.up_weight == nullptr || L.up_scale == nullptr || L.down_weight == nullptr || L.down_scale == nullptr ||
        L.g1_alphas == nullptr || L.g2_alphas == nullptr || L.act_scales == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const size_t k7_bytes = kernels::moe_workspace_bytes(d, m);
    if (k7_bytes == 0) {
        return CERE_ERR_ARGUMENT;
    }
    if (taps != nullptr && taps->routed_in != nullptr) {
        // The routed rows are given: the block is K7's (no routed expert runs).
        return kernels::moe_block(d, L, plans, x, m, out, workspace, workspace_bytes, stream, taps);
    }
    Runner &r = *rp;
    const Workspace w = carve(r, d, m, static_cast<uint8_t *>(workspace));
    if (w.bytes > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    const bool hooked = taps != nullptr && taps->point != nullptr;
    const auto point = [&](const char *name, const __nv_bfloat16 *values, uint32_t cols) {
        return taps->point(taps->ctx, name, const_cast<__nv_bfloat16 *>(values), m, cols, stream);
    };
    // The pinned tactic pair (or the caller's) must name tactics before anything forks.
    const TacticIndices pair =
        tactics == nullptr ? r.pinned[bucket_of(kPinnedBuckets, kPinnedBucketCount, plans.cls != 0 ? plans.cls : m)]
                           : *tactics;
    if (pair.gemm1 >= r.g1.size() || pair.gemm2 >= r.g2.size()) {
        return CERE_ERR_ARGUMENT;
    }
    // The shared expert (K7's, into w.k7) forks before the router when it may (k7_moe.h MoePlans), joined before the
    // combine (the scope joins it on every error return in between); else K7's block below runs it serially.
    const __nv_bfloat16 *shared = nullptr;
    const bool forked =
        !hooked && (taps == nullptr || taps->shared_in == nullptr) && kernels::moe_shared_forks(plans, m);
    kernels::SharedForkScope fork_scope(plans, stream);
    if (forked) {
        if ((s = kernels::moe_shared_fork(d, L, plans, x, m, w.k7, w.k7_bytes, stream, &shared)) != CERE_OK) {
            return s;
        }
        fork_scope.arm();
        CERE_FAULT_POINT(moe_fork, return CERE_ERR_INTERNAL);  // OH-08: an error after the fork (KGM-2)
    }
    // Router (K7's, with its hook points).
    if (hooked && (s = point("gemm.moe_router.in", x, d.hidden)) != CERE_OK) {
        return s;
    }
    if ((s = plans.router->run(x, L.router, m, w.logits, stream, plans.cls)) != CERE_OK) {
        return s;
    }
    if (hooked && (s = point("gemm.moe_router.out", w.logits, d.num_experts)) != CERE_OK) {
        return s;
    }
    if (hooked && taps->detail && (s = point("mlp.gate", w.logits, d.num_experts)) != CERE_OK) {
        return s;
    }
    if ((s = kernels::moe_route(w.logits, w.topk_ids, w.topk_weights, m, d.num_experts, d.top_k, stream)) !=
        CERE_OK) {
        return s;
    }
    if (taps != nullptr && taps->route != nullptr &&
        (s = taps->route(taps->ctx, w.topk_ids, w.topk_weights, m, d.top_k, stream)) != CERE_OK) {
        return s;
    }
    const size_t route_n = size_t(m) * d.top_k;
    if (topk_ids != nullptr && cudaMemcpyAsync(topk_ids, w.topk_ids, route_n * sizeof(int32_t),
                                               cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    if (topk_weights != nullptr && cudaMemcpyAsync(topk_weights, w.topk_weights, route_n * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    Rows rows;
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        if ((s = routed_rows(*r.fc, *r.gemm, r.g1[pair.gemm1], r.g2[pair.gemm2], rows_problem(d, m), x, w.topk_ids,
                             w.topk_weights, L.up_weight, L.down_weight, fp4_quant(L), w.experts, w.experts_bytes,
                             stream, &rows)) != CERE_OK) {
            return s;
        }
    }
    // Shared expert and combine (M23-D3): K7's combine forms the routed row from the expert rows as
    // finalizeMoeRoutingKernel does (bit-equal to routed_experts' finalized row), into the routed tap when given.
    kernels::MoeTaps given;
    if (taps != nullptr) {
        given = *taps;
    }
    given.routed_rows = rows.rows;
    given.routed_pos = rows.pos;
    given.routed_scales = w.topk_weights;
    if (forked) {
        if ((s = fork_scope.join()) != CERE_OK) {
            return s;
        }
        given.shared_in = shared;
    }
    return kernels::moe_block(d, L, plans, x, m, out, w.k7, w.k7_bytes, stream, &given);
}

}  // namespace cere::p3b
