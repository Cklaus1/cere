// P3b (m23 design note M23-D1): FlashInfer 0.6.17's CUTLASS fused-MoE runner for NVFP4 weights and activations
// with BF16 input and output, the one specialization p3b_moe.cu drives (FlashInfer's binding instantiates the same,
// flashinfer_cutlass_fused_moe_binding.cu), and routed_rows (p3b_rows.h, P13): runMoe's launches up to GEMM2 in
// the TU that sees FlashInfer's kernel templates.
#include "fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"
#include "moe_kernels.h"
#include "p3b/p3b_rows.h"

#include <algorithm>
#include <atomic>
#include <exception>

namespace tensorrt_llm::kernels::cutlass_kernels {
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
}

namespace cere::p3b {
namespace {

namespace tk = tensorrt_llm::kernels::cutlass_kernels;
using Tma = tk::TmaWarpSpecializedGroupedGemmInput;
using Fp4 = __nv_fp4_e2m1;

constexpr size_t kAlign = 256;
size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

// runMoe's workspace buffers (configureWsPtrs) for the non-min-latency NVFP4 SwiGLU path, as separate regions.
// FlashInfer's aliasing is kept: GEMM1's A (permuted_data) and GEMM2's A (fc1_result) share `inputs`, GEMM1's D
// (glu_inter_result) and GEMM2's D (fc2_result) share `outputs`, and both GEMMs' activation scale factors share
// `act_sf`. Every size is monotone in m, so a workspace carved at m serves every smaller m.
struct RowsWorkspace {
    int *permuted_row_to_unpermuted_row;
    int *permuted_token_selected_experts;
    int *blocked_expert_counts;
    int *blocked_expert_counts_cumsum;
    int *blocked_row_to_unpermuted_row;
    int *unpermuted_row_to_permuted_row;
    int64_t *expert_first_token_offset;
    Fp4 *inputs;
    __nv_bfloat16 *outputs;
    Tma::ElementSF *act_sf;
    float const **alpha_fc1;
    float const **alpha_fc2;
    int8_t *tma_gemm1;
    int8_t *tma_gemm2;
    int8_t *gemm;
    size_t gemm_bytes;
    // The decode GEMMs (p3b_decode.h; over the TMA inputs): the group table, the per-slot gemm1_act CTA counts and
    // GEMM2's compact FP4 A and scale factors.
    decode::Group *groups;
    uint32_t *decode_ready;
    uint8_t *decode_act;
    uint8_t *decode_act_sf;
    size_t bytes;
};

RowsWorkspace carve_rows(const GemmRunner &gemm, const RowsProblem &p, uint8_t *base) {
    RowsWorkspace w{};
    size_t off = 0;
    auto take = [&](size_t bytes) {
        uint8_t *q = base == nullptr ? nullptr : base + off;
        off += align_up(bytes);
        return q;
    };
    const int64_t mk = p.m * p.top_k;
    const int64_t e = p.experts;
    // The three-step prologue's blocked counts: E * ceil(m / tokens_per_block) with computeNumTokensPerBlock's
    // tokens_per_block in [32, 1024] and E * blocks <= tokens_per_block unless it is 1024, bounded by the smaller of
    // E * ceil(m / 32) and max(1024, E * ceil(m / 1024)) (both monotone in m).
    const int64_t blocked = std::min(e * ((p.m + 31) / 32), std::max<int64_t>(1024, e * ((p.m + 1023) / 1024)));
    w.permuted_row_to_unpermuted_row = reinterpret_cast<int *>(take(sizeof(int) * mk));
    w.permuted_token_selected_experts = reinterpret_cast<int *>(take(sizeof(int) * mk));
    w.blocked_expert_counts = reinterpret_cast<int *>(take(sizeof(int) * blocked));
    w.blocked_expert_counts_cumsum = reinterpret_cast<int *>(take(sizeof(int) * blocked));
    w.blocked_row_to_unpermuted_row = reinterpret_cast<int *>(take(sizeof(int) * e * p.m));
    w.unpermuted_row_to_permuted_row = reinterpret_cast<int *>(take(sizeof(int) * mk));
    w.expert_first_token_offset = reinterpret_cast<int64_t *>(take(sizeof(int64_t) * (e + 1)));
    // FP4 rows: two values per byte.
    w.inputs = reinterpret_cast<Fp4 *>(take(static_cast<size_t>(mk * std::max(p.hidden, p.inter) / 2)));
    w.outputs =
        reinterpret_cast<__nv_bfloat16 *>(take(sizeof(__nv_bfloat16) * mk * std::max(2 * p.inter, p.hidden)));
    const int64_t sf_rows = std::min(mk, p.m * e);
    const int64_t sf = std::max(tk::getOffsetActivationSF(e, sf_rows, p.hidden, Tma::FpXBlockScalingType::NVFP4),
                                tk::getOffsetActivationSF(e, sf_rows, p.inter, Tma::FpXBlockScalingType::NVFP4));
    w.act_sf = reinterpret_cast<Tma::ElementSF *>(take(sizeof(Tma::NVFP4ElementSF) * sf));
    w.alpha_fc1 = reinterpret_cast<float const **>(take(sizeof(float *) * e));
    w.alpha_fc2 = reinterpret_cast<float const **>(take(sizeof(float *) * e));
    // The two GEMMs' TMA inputs, or (decode GEMMs: no TMA inputs) the group table, the per-slot counts and GEMM2's
    // compact FP4 A and scale factors, in one region (the larger of the two layouts).
    const size_t tma_bytes = align_up(Tma::workspaceSize(static_cast<int>(e), Tma::FpXBlockScalingType::NVFP4));
    const int64_t decode_rows = std::min(mk, decode::kMaxRows);
    const size_t groups_bytes = align_up(sizeof(decode::Group) * static_cast<size_t>(decode_rows));
    const size_t ready_bytes = align_up(sizeof(uint32_t) * static_cast<size_t>(decode_rows));
    const size_t act_bytes = align_up(static_cast<size_t>(decode_rows * p.inter / 2));
    const size_t act_sf_bytes = align_up(static_cast<size_t>(decode_rows * p.inter / 16));
    uint8_t *shared = take(std::max(2 * tma_bytes, groups_bytes + ready_bytes + act_bytes + act_sf_bytes));
    w.tma_gemm1 = reinterpret_cast<int8_t *>(shared);
    w.tma_gemm2 = reinterpret_cast<int8_t *>(shared == nullptr ? nullptr : shared + tma_bytes);
    w.groups = reinterpret_cast<decode::Group *>(shared);
    w.decode_ready = reinterpret_cast<uint32_t *>(shared == nullptr ? nullptr : shared + groups_bytes);
    w.decode_act = shared == nullptr ? nullptr : shared + groups_bytes + ready_bytes;
    w.decode_act_sf = shared == nullptr ? nullptr : shared + groups_bytes + ready_bytes + act_bytes;
    w.gemm_bytes = gemm.getMaxWorkspaceSize(static_cast<int>(e));
    w.gemm = reinterpret_cast<int8_t *>(take(w.gemm_bytes));
    w.bytes = off;
    return w;
}

// The decode / verify prologue (m * top_k <= kSmallRows): runMoe's expert-map prologue and expandInputRowsKernel as
// one launch. Every CTA reads the m * top_k routing into shared memory and derives the same maps the three-step
// sort writes (permuted rows grouped by expert in increasing expert id, within an expert in increasing token):
// the permuted row of (token t, slot k, expert e) is #{entries with expert < e} + #{entries with expert e and token
// < t}, counted, not sorted, so no atomics and no order dependence. CTA 0 stores the maps runMoe's later launches
// read (permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, expert_first_token_offset); each CTA then
// expands its permuted rows exactly as expandInputRowsKernel does (the same grid, block, row loop and
// quantizePackedFPXValue / writeSF calls, so the FP4 rows and scale factors are bit-equal to it).
constexpr int64_t kSmallRows = tk::EXPAND_THREADS_PER_BLOCK;

template <bool kDisableFastMath, typename Config>
__global__ void __launch_bounds__(tk::EXPAND_THREADS_PER_BLOCK)
    small_prologue_kernel(const __nv_bfloat16 *__restrict__ x, Fp4 *__restrict__ permuted_output,
                          const int32_t *__restrict__ topk_ids, int *__restrict__ permuted_row_to_unpermuted_row,
                          int *__restrict__ unpermuted_row_to_permuted_row,
                          int64_t *__restrict__ expert_first_token_offset, int64_t num_tokens, int64_t hidden_size,
                          int64_t k, const float *__restrict__ fc1_act_global_scale, bool use_per_expert_act_scale,
                          Tma::ElementSF *__restrict__ fc1_act_sf_flat, int64_t num_experts,
                          decode::Group *__restrict__ groups, uint32_t *__restrict__ ready, int group_slots) {
    __shared__ int s_expert[kSmallRows];   // by (token, slot) entry t * k + slot
    __shared__ int s_unpermuted[kSmallRows];  // by permuted row
    __shared__ int s_row_expert[kSmallRows];  // by permuted row
    __shared__ int s_before[kSmallRows];      // by permuted row: entries routed to a smaller expert id
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    const int n = static_cast<int>(num_tokens * k);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_expert[i] = topk_ids[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int expert = s_expert[i];
        const int slot = i % static_cast<int>(k);
        const int token_start = i - slot;  // the entries of earlier tokens are j < token_start
        int before = 0;
        int rank = 0;
#pragma unroll 8
        for (int j = 0; j < n; ++j) {
            const int other = s_expert[j];
            before += other < expert;
            rank += (other == expert) & (j < token_start);
        }
        const int permuted = before + rank;
        const int unpermuted = slot * static_cast<int>(num_tokens) + i / static_cast<int>(k);
        s_unpermuted[permuted] = unpermuted;
        s_row_expert[permuted] = expert;
        s_before[permuted] = before;
        if (blockIdx.x == 0) {
            permuted_row_to_unpermuted_row[permuted] = unpermuted;
            unpermuted_row_to_permuted_row[unpermuted] = permuted;
        }
    }
    __syncthreads();
    if (blockIdx.x == 0) {
        // s_row_expert holds the experts in permuted (ascending) order: #entries with expert < e is its lower bound.
        for (int64_t e = threadIdx.x; e <= num_experts; e += blockDim.x) {
            int lo = 0;
            int hi = n;
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                if (s_row_expert[mid] < e) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            expert_first_token_offset[e] = lo;
        }
    }
    if (blockIdx.x == 0 && groups != nullptr) {
        // The decode GEMMs' group table: slot s = the s-th active expert (increasing id), past the active count -1.
        int active = n > 0 ? 1 : 0;
        for (int j = 1; j < n; ++j) {
            active += s_row_expert[j] != s_row_expert[j - 1];
        }
        for (int p = threadIdx.x; p < n; p += blockDim.x) {
            const int expert = s_row_expert[p];
            if (p == 0 || s_row_expert[p - 1] != expert) {
                int slot = 0;
                for (int j = 1; j <= p; ++j) {
                    slot += s_row_expert[j] != s_row_expert[j - 1];
                }
                int end = p + 1;
                while (end < n && s_row_expert[end] == expert) {
                    ++end;
                }
                groups[slot] = decode::Group{expert, p, end - p, active};
            }
        }
        for (int s = active + static_cast<int>(threadIdx.x); s < group_slots; s += blockDim.x) {
            groups[s] = decode::Group{-1, 0, 0, active};
        }
        for (int s = static_cast<int>(threadIdx.x); s < group_slots; s += blockDim.x) {
            ready[s] = 0;
        }
    }
    // expandInputRowsKernel<__nv_bfloat16, __nv_fp4_e2m1, NVFP4, false, kDisableFastMath, Config>'s body.
    constexpr int VecSize = Tma::NVFP4BlockScaleVectorSize;
    constexpr int64_t ELEM_PER_THREAD = tk::CVT_ELTS_PER_THREAD;
    using DataElem = cutlass::Array<__nv_bfloat16, ELEM_PER_THREAD>;
    using OutputElem = uint32_t;
    const int64_t padded_hidden_size = Tma::alignToSfDim(hidden_size, Tma::MinKDimAlignmentNVFP4);
    for (int64_t permuted_row = blockIdx.x; permuted_row < n; permuted_row += gridDim.x) {
        const int64_t unpermuted_row = s_unpermuted[permuted_row];
        const int64_t source_row = unpermuted_row % num_tokens;
        const auto *source_row_ptr = reinterpret_cast<const DataElem *>(x + source_row * hidden_size);
        auto *dest_row_ptr = reinterpret_cast<OutputElem *>(permuted_output) + permuted_row * hidden_size / ELEM_PER_THREAD;
        const int64_t start_offset = threadIdx.x;
        const int64_t stride = tk::EXPAND_THREADS_PER_BLOCK;
        const int64_t num_elems_in_col = hidden_size / ELEM_PER_THREAD;
        const int64_t expert = s_row_expert[permuted_row];
        const size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
        const float global_scale_val = fc1_act_global_scale ? fc1_act_global_scale[act_scale_idx] : 1.0f;
        const int64_t num_tokens_before_expert = s_before[permuted_row];
        for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
            auto in_vec = source_row_ptr[elem_index];
            auto res = tk::quantizePackedFPXValue<__nv_bfloat16, Fp4, DataElem, VecSize, kDisableFastMath, Config>(
                in_vec, global_scale_val, num_tokens_before_expert, expert, permuted_row, elem_index,
                padded_hidden_size, fc1_act_sf_flat, Tma::FpXBlockScalingType::NVFP4);
            static_assert(sizeof(res) == sizeof(*dest_row_ptr), "Quantized value must be the same size as the output");
            dest_row_ptr[elem_index] = res;
        }
        const size_t padding_start_offset = hidden_size / VecSize + start_offset;
        const size_t padding_elems_in_col = padded_hidden_size / VecSize;
        for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col; elem_index += stride) {
            tk::writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_token_id=*/-1, permuted_row,
                                          elem_index, padded_hidden_size, fc1_act_sf_flat, /*input_sf=*/nullptr);
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// The current device's SM count, queried once per ordinal (m32n design note D4: no process-global device state; the
// deployment's two cards are one SKU, so every slot holds the same count there). 0 when the device is not known.
int64_t device_sm_count() {
    constexpr int kDevices = 16;
    static std::atomic<int64_t> counts[kDevices]{};
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kDevices) {
        cudaGetLastError();
        return 0;
    }
    int64_t count = counts[device].load();
    if (count == 0) {
        count = tensorrt_llm::common::getMultiProcessorCount();
        counts[device].store(count);
    }
    return count;
}

// expandInputRowsKernelLauncher's launch (grid min(8 * SMs, m * top_k), EXPAND_THREADS_PER_BLOCK threads, PDL, the
// kernel variant dispatchNVFP44Over6Config picks) for small_prologue_kernel.
void small_prologue(const __nv_bfloat16 *x, Fp4 *permuted_output, const int32_t *topk_ids,
                    int *permuted_row_to_unpermuted_row, int *unpermuted_row_to_permuted_row,
                    int64_t *expert_first_token_offset, int64_t num_rows, int64_t hidden_size, int64_t k,
                    const tk::QuantParams &quant, Tma::ElementSF *fc1_act_sf_flat, int64_t num_experts,
                    decode::Group *groups, uint32_t *ready, int64_t group_slots, bool enable_pdl,
                    cudaStream_t stream) {
    const int64_t blocks = std::min(std::max<int64_t>(device_sm_count(), 1) * 8, std::max(num_rows * k, int64_t{1}));
    auto func = tk::dispatchNVFP44Over6Config([&](auto disable_fast_math_tag, auto config_tag) {
        return &small_prologue_kernel<decltype(disable_fast_math_tag)::value, decltype(config_tag)>;
    });
    cudaLaunchConfig_t config;
    config.gridDim = static_cast<unsigned>(blocks);
    config.blockDim = tk::EXPAND_THREADS_PER_BLOCK;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, func, x, permuted_output, topk_ids, permuted_row_to_unpermuted_row,
                       unpermuted_row_to_permuted_row, expert_first_token_offset, num_rows, hidden_size, k,
                       quant.fp4.fc1.act_global_scale, quant.fp4.fc1.use_per_expert_act_scale, fc1_act_sf_flat,
                       num_experts, groups, ready, static_cast<int>(group_slots));
}

// The prefill expansion and activation (m * top_k > kSmallRows). expandInputRowsKernel and doActivationKernel give
// each permuted row one 256-thread CTA, so hidden / 8 = 320 vectors run in two passes (the second 64 threads wide)
// and inter / 8 = 80 vectors keep 80 of 256 threads busy, and expandInputRowsKernel reads and quantizes every token
// top_k times. These kernels split the flat (row, vector) index space into chunks of kChunkItems vectors, kTileItems
// per thread, every thread busy, with the chunk's per-row table (expert, its first permuted row, scales) in shared
// memory; expand_tiles quantizes each token once and stores the FP4 vector and its scale factor at every permuted
// row of the token. Identity: each 8-element vector goes through quantizePackedFPXValue's body (the PackedVec
// packing, cvt_quant_get_sf_out_offset, cvt_warp_fp16_to_fp4 with FlashInfer's template arguments) on the same
// values, global scale, expert, expert start and row, and the scale factor's xor-1 shuffle pairs the same two
// vectors: the vectors per row are even and chunk items start on even thread ids, so vector j of a row sits on a
// thread of j's parity with j ^ 1 of the same row on its xor-1 lane, and the even thread stores the scale factor as
// in FlashInfer. Lanes past the last row convert a clamped vector and store nothing (every shuffle sees the whole
// warp); the rows written, and the K-dim scale-factor padding, are the rows FlashInfer's kernels visit.
constexpr int kTileThreads = 256;
constexpr int kTileItems = 4;  // vectors per thread per chunk
constexpr int64_t kChunkItems = int64_t{kTileThreads} * kTileItems;
constexpr size_t kMaxTileSmem = 48 * 1024;

// A chunk spans at most kChunkItems / vecs + 2 rows of `vecs` vectors.
int64_t chunk_rows(int64_t vecs) { return kChunkItems / vecs + 2; }

// Up to 8 CTAs per SM (FlashInfer's bandwidth sizing), at most one per chunk of `items` (the host's row bound).
int64_t tile_blocks(int64_t items) {
    const int64_t chunks = std::max<int64_t>((items + kChunkItems - 1) / kChunkItems, 1);
    return std::min(std::max<int64_t>(device_sm_count(), 1) * 8, chunks);
}

// One (token, slot) copy of an expand chunk: its permuted row (-1: not expanded), expert and the expert's first row.
struct ExpandCopy {
    int64_t before;
    int permuted;
    int expert;
};

// expandInputRowsKernel<__nv_bfloat16, __nv_fp4_e2m1, NVFP4, false, kDisableFastMath, std::false_type> with one act
// global scale (not per expert). The permuted rows of token t are those p < num_valid_tokens with
// permuted_row_to_unpermuted_row[p] = slot * num_tokens + t: both prologues write that map and
// unpermuted_row_to_permuted_row as inverse pairs for every expanded row (the three-step sort leaves the inverse
// stale for a repeated expert of a token, which the check rejects), so the rows written are FlashInfer's, each once
// with source row t. FP4 bytes and the scale-factor value depend only on the token's vector and the global scale.
template <bool kDisableFastMath>
__global__ void __launch_bounds__(kTileThreads)
    expand_tiles_kernel(const __nv_bfloat16 *__restrict__ x, Fp4 *__restrict__ permuted_output,
                        const int *__restrict__ permuted_row_to_unpermuted_row,
                        const int *__restrict__ unpermuted_row_to_permuted_row, int64_t num_tokens,
                        int64_t hidden_size, int64_t k, const float *__restrict__ fc1_act_global_scale,
                        const int64_t *__restrict__ expert_first_token_offset,
                        Tma::ElementSF *__restrict__ fc1_act_sf_flat, int64_t num_experts_per_node) {
    extern __shared__ ExpandCopy s_copy[];  // by (token - chunk's first token) * k + slot
    constexpr int VecSize = Tma::NVFP4BlockScaleVectorSize;
    constexpr int ELEM_PER_THREAD = tk::CVT_ELTS_PER_THREAD;
    using DataElem = cutlass::Array<__nv_bfloat16, ELEM_PER_THREAD>;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    const int64_t padded_hidden_size = Tma::alignToSfDim(hidden_size, Tma::MinKDimAlignmentNVFP4);
    const int64_t num_valid_tokens = expert_first_token_offset[num_experts_per_node];
    const float global_scale_val = fc1_act_global_scale ? fc1_act_global_scale[0] : 1.0f;
    const int64_t vecs = hidden_size / ELEM_PER_THREAD;
    const int64_t total = num_tokens * vecs;
    const int64_t pad = padded_hidden_size / VecSize - hidden_size / VecSize;
    auto *dest = reinterpret_cast<uint32_t *>(permuted_output);
    for (int64_t base = int64_t{blockIdx.x} * kChunkItems; base < total; base += int64_t{gridDim.x} * kChunkItems) {
        const int64_t first = base / vecs;
        const int64_t last = ((base + kChunkItems < total ? base + kChunkItems : total) - 1) / vecs;
        const int64_t copies = (last - first + 1) * k;
        DataElem in_vec[kTileItems];
#pragma unroll
        for (int u = 0; u < kTileItems; ++u) {
            const int64_t item = base + u * kTileThreads + threadIdx.x;
            const int64_t at = item < total ? item : total - 1;
            const int64_t token = at / vecs;
            in_vec[u] = reinterpret_cast<const DataElem *>(x + token * hidden_size)[at - token * vecs];
        }
        __syncthreads();  // the previous chunk's reads of s_copy
        for (int64_t j = threadIdx.x; j < copies; j += kTileThreads) {
            const int64_t unpermuted = (j % k) * num_tokens + first + j / k;
            const int permuted = unpermuted_row_to_permuted_row[unpermuted];
            ExpandCopy c{0, -1, 0};
            if (permuted >= 0 && permuted < num_valid_tokens &&
                permuted_row_to_unpermuted_row[permuted] == unpermuted) {
                const int64_t expert = tk::findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node,
                                                                       int64_t{permuted} + 1) -
                                       1;
                c = ExpandCopy{expert_first_token_offset[expert], permuted, static_cast<int>(expert)};
            }
            s_copy[j] = c;
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < kTileItems; ++u) {
            const int64_t item = base + u * kTileThreads + threadIdx.x;
            const bool valid = item < total;
            const int64_t at = valid ? item : total - 1;
            const int64_t token = at / vecs;
            const int64_t elem_index = at - token * vecs;
            // quantizePackedFPXValue's body, its scale factor kept for every copy's own (expert, row) offset.
            tensorrt_llm::kernels::PackedVec<__nv_bfloat16> packed_vec{};
            for (int i = 0; i < ELEM_PER_THREAD / 2; i++) {
                packed_vec.elts[i].x = static_cast<__nv_bfloat16>(in_vec[u][i * 2 + 0]);
                packed_vec.elts[i].y = static_cast<__nv_bfloat16>(in_vec[u][i * 2 + 1]);
            }
            uint8_t sf = 0;
            const uint32_t res =
                tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, VecSize, ELEM_PER_THREAD, false,
                                                            kDisableFastMath, std::false_type>(
                    packed_vec, global_scale_val, &sf, 0.0f);
            if (!valid) {
                continue;
            }
            for (int64_t slot = 0; slot < k; ++slot) {
                const ExpandCopy c = s_copy[(token - first) * k + slot];
                if (c.permuted < 0) {
                    continue;
                }
                dest[c.permuted * vecs + elem_index] = res;
                auto *act_sf_expert = fc1_act_sf_flat + tk::getOffsetActivationSF(c.expert, c.before, padded_hidden_size,
                                                                                  Tma::FpXBlockScalingType::NVFP4);
                uint8_t *sf_out =
                    tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<Tma::ElementSF, VecSize / ELEM_PER_THREAD>(
                        std::nullopt, c.permuted - c.before, elem_index, std::nullopt, padded_hidden_size / VecSize,
                        act_sf_expert, flashinfer::QuantizationSFLayout::SWIZZLED_128x4);
                if (sf_out) {
                    *sf_out = sf;
                }
            }
        }
        // K-dim scale-factor padding, by the chunk holding the token's first vector.
        const int64_t pad_first = first * vecs < base ? first + 1 : first;
        for (int64_t j = threadIdx.x; j < (last - pad_first + 1) * k * pad; j += kTileThreads) {
            const ExpandCopy c = s_copy[(pad_first - first) * k + j / pad];
            if (c.permuted >= 0) {
                tk::writeSF<VecSize, VecSize>(c.before, c.expert, /*source_token_id=*/-1, c.permuted,
                                              hidden_size / VecSize + j % pad, padded_hidden_size, fc1_act_sf_flat,
                                              /*input_sf=*/nullptr);
            }
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// One permuted row of an activation chunk: its expert, the expert's first row and the row's scales.
struct ActRow {
    int64_t before;
    int expert;
    float quant_scale;
    float global_scale;
};

// doActivationKernel<__nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16, GLUAdaptor<SiLu>, NVFP4, kDisableFastMath,
// std::false_type> without bias or swiglu alpha / beta / limit (GLUAdaptor reads none of them): per vector the same
// arrayConvert, fn(gate, linear), * quant_scale and quantizePackedFPXValue body (num_cols = inter_size), per row the
// same expert search, scales and K-dim padding (num_cols = padded_inter_size).
template <bool kDisableFastMath>
__global__ void __launch_bounds__(kTileThreads)
    act_tiles_kernel(Fp4 *__restrict__ output, const __nv_bfloat16 *__restrict__ gemm_result,
                     const float *__restrict__ fp8_quant, const int64_t *__restrict__ expert_first_token_offset,
                     int num_experts_per_node, int64_t inter_size, const float *__restrict__ fc2_act_global_scale,
                     bool use_per_expert_act_scale, Tma::ElementSF *__restrict__ fc2_act_sf_flat) {
    extern __shared__ ActRow s_row[];  // by row - chunk's first row
    using ActFn = tk::GLUAdaptor<cutlass::epilogue::thread::SiLu>;
    constexpr int VecSize = Tma::NVFP4BlockScaleVectorSize;
    constexpr int ELEM_PER_THREAD = tk::CVT_ELTS_PER_THREAD;
    using GemmResultElem = cutlass::Array<__nv_bfloat16, ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ELEM_PER_THREAD>;
    constexpr int64_t min_k_dim_alignment = Tma::MinKDimAlignmentNVFP4;
    const int64_t padded_inter_size = (inter_size + min_k_dim_alignment - 1) / min_k_dim_alignment * min_k_dim_alignment;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    const int64_t num_valid_tokens = expert_first_token_offset[num_experts_per_node];
    const int64_t vecs = inter_size / ELEM_PER_THREAD;  // also the gate half's offset (gated_off_vec)
    const int64_t total = num_valid_tokens * vecs;
    const int64_t pad = padded_inter_size / VecSize - inter_size / VecSize;
    auto *output_vec = reinterpret_cast<uint32_t *>(output);
    for (int64_t base = int64_t{blockIdx.x} * kChunkItems; base < total; base += int64_t{gridDim.x} * kChunkItems) {
        const int64_t first = base / vecs;
        const int64_t last = ((base + kChunkItems < total ? base + kChunkItems : total) - 1) / vecs;
        GemmResultElem gate_in[kTileItems];
        GemmResultElem linear_in[kTileItems];
#pragma unroll
        for (int u = 0; u < kTileItems; ++u) {
            const int64_t item = base + u * kTileThreads + threadIdx.x;
            const int64_t at = item < total ? item : total - 1;
            const int64_t token = at / vecs;
            const auto *gemm_result_vec = reinterpret_cast<const GemmResultElem *>(gemm_result + token * inter_size * 2);
            gate_in[u] = gemm_result_vec[at - token * vecs + vecs];
            linear_in[u] = gemm_result_vec[at - token * vecs];
        }
        __syncthreads();  // the previous chunk's reads of s_row
        for (int64_t r = threadIdx.x; r <= last - first; r += kTileThreads) {
            const int64_t expert =
                tk::findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node, first + r + 1) - 1;
            const size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
            s_row[r] = ActRow{expert_first_token_offset[expert], static_cast<int>(expert),
                              fp8_quant ? fp8_quant[act_scale_idx] : 1.f,
                              fc2_act_global_scale ? fc2_act_global_scale[act_scale_idx] : 1.0f};
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < kTileItems; ++u) {
            const int64_t item = base + u * kTileThreads + threadIdx.x;
            const bool valid = item < total;
            const int64_t at = valid ? item : total - 1;
            const int64_t token = at / vecs;
            const int64_t elem_index = at - token * vecs;
            const ActRow row = s_row[token - first];
            const int64_t expert = row.expert;
            const int64_t num_tokens_before_expert = row.before;
            ActFn fn{};
            auto fc1_value = tk::arrayConvert<GemmResultElem, ComputeElem>(gate_in[u]);
            auto linear_value = tk::arrayConvert<GemmResultElem, ComputeElem>(linear_in[u]);
            auto gate_act = fn(fc1_value, linear_value);
            auto post_act_val = gate_act * row.quant_scale;
            // quantizePackedFPXValue's body, the scale-factor store dropped on the clamped lanes.
            tensorrt_llm::kernels::PackedVec<__nv_bfloat16> packed_vec{};
            for (int i = 0; i < ELEM_PER_THREAD / 2; i++) {
                packed_vec.elts[i].x = static_cast<__nv_bfloat16>(post_act_val[i * 2 + 0]);
                packed_vec.elts[i].y = static_cast<__nv_bfloat16>(post_act_val[i * 2 + 1]);
            }
            auto *act_sf_expert = fc2_act_sf_flat + tk::getOffsetActivationSF(expert, num_tokens_before_expert,
                                                                              inter_size, Tma::FpXBlockScalingType::NVFP4);
            uint8_t *sf_out =
                tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<Tma::ElementSF, VecSize / ELEM_PER_THREAD>(
                    std::nullopt, token - num_tokens_before_expert, elem_index, std::nullopt, inter_size / VecSize,
                    act_sf_expert, flashinfer::QuantizationSFLayout::SWIZZLED_128x4);
            const uint32_t res =
                tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, VecSize, ELEM_PER_THREAD, false,
                                                            kDisableFastMath, std::false_type>(
                    packed_vec, row.global_scale, valid ? sf_out : nullptr, 0.0f);
            if (valid) {
                output_vec[token * vecs + elem_index] = res;
            }
        }
        // K-dim scale-factor padding, by the chunk holding the row's first vector.
        const int64_t pad_first = first * vecs < base ? first + 1 : first;
        for (int64_t j = threadIdx.x; j < (last - pad_first + 1) * pad; j += kTileThreads) {
            const int64_t r = pad_first - first + j / pad;
            const ActRow row = s_row[r];
            tk::writeSF<VecSize, VecSize>(row.before, row.expert, /*source_token_id=*/-1, first + r,
                                          inter_size / VecSize + j % pad, padded_inter_size, fc2_act_sf_flat,
                                          /*input_sf=*/nullptr);
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename Kernel, typename... Args>
void launch_tiles(Kernel kernel, int64_t blocks, size_t smem, bool enable_pdl, cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t config;
    config.gridDim = static_cast<unsigned>(blocks);
    config.blockDim = kTileThreads;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, args...);
}

// The tiled kernels reproduce FlashInfer's default NVFP4 quantization only (FLASHINFER_NVFP4_4OVER6 keeps
// FlashInfer's kernels), with even vectors per row.
bool tiles_shape(const RowsProblem &p) {
    return p.hidden % (2 * tk::CVT_ELTS_PER_THREAD) == 0 && p.inter % (2 * tk::CVT_ELTS_PER_THREAD) == 0 &&
           !tensorrt_llm::common::getEnvNVFP4Use4Over6();
}

// expand_tiles for expandInputRowsKernelLauncher's prefill launch: one act global scale (else, or without FC1 weight
// block scales, which the launcher refuses, the launcher itself).
bool expand_tiled(const RowsProblem &p, const tk::QuantParams &quant) {
    return tiles_shape(p) && !quant.fp4.fc1.use_per_expert_act_scale && quant.fp4.fc1.weight_block_scale != nullptr &&
           static_cast<size_t>(chunk_rows(p.hidden / tk::CVT_ELTS_PER_THREAD) * p.top_k) * sizeof(ExpandCopy) <=
               kMaxTileSmem;
}

void expand_tiles(const __nv_bfloat16 *x, const RowsWorkspace &w, const RowsProblem &p, const tk::QuantParams &quant,
                  bool enable_pdl, cudaStream_t stream) {
    const int64_t vecs = p.hidden / tk::CVT_ELTS_PER_THREAD;
    auto kernel = tensorrt_llm::common::getEnvDisableFP4QuantFastMath() ? &expand_tiles_kernel<true>
                                                                        : &expand_tiles_kernel<false>;
    launch_tiles(kernel, tile_blocks(p.m * vecs), static_cast<size_t>(chunk_rows(vecs) * p.top_k) * sizeof(ExpandCopy),
                 enable_pdl, stream, x, w.inputs, static_cast<const int *>(w.permuted_row_to_unpermuted_row),
                 static_cast<const int *>(w.unpermuted_row_to_permuted_row), p.m, p.hidden, p.top_k,
                 quant.fp4.fc1.act_global_scale, static_cast<const int64_t *>(w.expert_first_token_offset), w.act_sf,
                 p.experts);
}

// FcRunner::gemm1's TMA warp-specialized branch with act_tiles for doActivation: that branch requires
// config.is_tma_warp_specialized and no fused gated activation, doActivation FC2 weight block scales (else gemm1
// itself, which throws as runMoe does).
bool act_tiled(GemmRunner &gemm, const tensorrt_llm::cutlass_extensions::CutlassGemmConfig &config,
               const RowsProblem &p, const tk::QuantParams &quant) {
    return tiles_shape(p) && gemm.isTmaWarpSpecialized(config) && config.is_tma_warp_specialized &&
           !gemm.isFusedGatedActivation(config, tk::ActivationType::Swiglu, static_cast<int>(p.inter),
                                        static_cast<int>(p.hidden)) &&
           quant.fp4.fc2.weight_block_scale != nullptr &&
           static_cast<size_t>(chunk_rows(p.inter / tk::CVT_ELTS_PER_THREAD)) * sizeof(ActRow) <= kMaxTileSmem;
}

// gemm1 (NVFP4 x NVFP4, gated SwiGLU: GEMM1 into glu_inter_result (`outputs`), no w4afp8 alpha rewrite), then the
// activation into fc1_result (`inputs`) and the FC2 scale factors.
void gemm1_tiles(GemmRunner &gemm, const tensorrt_llm::cutlass_extensions::CutlassGemmConfig &config, const Tma &tma1,
                 const RowsWorkspace &w, const RowsProblem &p, const tk::QuantParams &quant, bool enable_pdl,
                 cudaStream_t stream) {
    const int e = static_cast<int>(p.experts);
    auto universal_input = tk::GroupedGemmInput<Fp4, Fp4, __nv_bfloat16, __nv_bfloat16>{
        w.inputs,
        w.expert_first_token_offset + 1,  // total_tokens_including_expert
        /*weights*/ nullptr,
        /*scales*/ nullptr,
        /*zeros*/ nullptr,
        /*biases*/ nullptr,
        /*C*/ nullptr,
        w.alpha_fc1,
        /*occupancy*/ nullptr,
        tk::ActivationType::Swiglu,
        p.m,
        /*N*/ 2 * p.inter,
        /*K*/ p.hidden,
        e,
        quant.groupwise.group_size,
        /*bias_is_broadcast*/ true,
        /*use_fused_moe*/ false,
        stream,
        config};
    gemm.moeGemm(universal_input, tma1);
    sync_check_cuda_error(stream);
    const int64_t vecs = p.inter / tk::CVT_ELTS_PER_THREAD;
    auto kernel =
        tensorrt_llm::common::getEnvDisableFP4QuantFastMath() ? &act_tiles_kernel<true> : &act_tiles_kernel<false>;
    launch_tiles(kernel, tile_blocks(p.m * p.top_k * vecs), static_cast<size_t>(chunk_rows(vecs)) * sizeof(ActRow),
                 enable_pdl, stream, w.inputs, static_cast<const __nv_bfloat16 *>(w.outputs), quant.fp8.quant_fc2,
                 static_cast<const int64_t *>(w.expert_first_token_offset), e, p.inter,
                 quant.fp4.fc2.act_global_scale, quant.fp4.fc2.use_per_expert_act_scale, w.act_sf);
    sync_check_cuda_error(stream);
}

}  // namespace

size_t rows_workspace_bytes(const GemmRunner &gemm, const RowsProblem &p) {
    if (p.m <= 0 || p.hidden <= 0 || p.inter <= 0 || p.experts <= 0 || p.top_k <= 0) {
        return 0;
    }
    try {
        return carve_rows(gemm, p, nullptr).bytes;
    } catch (const std::exception &) {
        return 0;
    }
}

bool rows_decoded(const RowsProblem &p) {
    // The decode GEMMs (P14) unless the ktest asks for the grouped ones, the shape is past kRoutedMaxRows (kAuto) or
    // FLASHINFER_NVFP4_4OVER6 picks the 4over6 quantization (doActivation's, which they do not reproduce).
    const int64_t mk = p.m * p.top_k;
    const decode::Shape shape{p.m, p.hidden, p.inter, p.experts, p.top_k};
    return mk <= kSmallRows && p.path != RowsPath::kGrouped && decode::supported(shape) &&
           (p.path == RowsPath::kDecode || mk <= decode::kRoutedMaxRows) &&
           !tensorrt_llm::common::getEnvNVFP4Use4Over6();
}

cere_status routed_rows(FcRunner &fc, GemmRunner &gemm, const tensorrt_llm::cutlass_extensions::CutlassGemmConfig &gemm1,
                        const tensorrt_llm::cutlass_extensions::CutlassGemmConfig &gemm2, const RowsProblem &p,
                        const __nv_bfloat16 *x, const int32_t *topk_ids, const float *topk_weights,
                        const void *w13_weight, const void *down_weight, const tk::QuantParams &quant, void *workspace,
                        size_t workspace_bytes, cudaStream_t stream, Rows *out) {
    const size_t need = rows_workspace_bytes(gemm, p);
    if (need == 0 || need > workspace_bytes || workspace == nullptr ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || x == nullptr || topk_ids == nullptr ||
        topk_weights == nullptr || w13_weight == nullptr || down_weight == nullptr || out == nullptr ||
        !gemm.isTmaWarpSpecialized(gemm1) || !gemm.isTmaWarpSpecialized(gemm2) ||
        gemm2.epilogue_fusion_type == tensorrt_llm::cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE) {
        return CERE_ERR_ARGUMENT;
    }
    constexpr bool kPdl = true;  // routed_experts' runMoe(enable_pdl = true)
    const RowsWorkspace w = carve_rows(gemm, p, static_cast<uint8_t *>(workspace));
    const int64_t mk = p.m * p.top_k;
    const int e = static_cast<int>(p.experts);
    const int k = static_cast<int>(p.top_k);
    const auto *w13 = static_cast<const Fp4 *>(w13_weight);
    const auto *down = static_cast<const Fp4 *>(down_weight);
    const decode::Shape shape{p.m, p.hidden, p.inter, p.experts, p.top_k};
    const bool decoded = rows_decoded(p);
    try {
        if (mk <= kSmallRows) {
            // Decode / verify: the maps, the group table and the expanded FP4 rows in one launch.
            small_prologue(x, w.inputs, topk_ids, w.permuted_row_to_unpermuted_row, w.unpermuted_row_to_permuted_row,
                           w.expert_first_token_offset, p.m, p.hidden, p.top_k, quant, w.act_sf, p.experts,
                           decoded ? w.groups : nullptr, w.decode_ready, decode::slots(shape), kPdl, stream);
        } else {
            // Prologue (runMoe: the fused kernel when it supports the shape, else the three-step sort).
            if (!tk::fusedBuildExpertMapsSortFirstToken(topk_ids, w.permuted_row_to_unpermuted_row,
                                                        w.unpermuted_row_to_permuted_row, w.expert_first_token_offset,
                                                        p.m, e, k, /*start_expert=*/0, /*end_expert=*/e, kPdl,
                                                        stream)) {
                tk::threeStepBuildExpertMapsSortFirstToken(
                    topk_ids, w.permuted_token_selected_experts, w.permuted_row_to_unpermuted_row,
                    w.unpermuted_row_to_permuted_row, w.expert_first_token_offset, w.blocked_expert_counts,
                    w.blocked_expert_counts_cumsum, w.blocked_row_to_unpermuted_row, p.m, p.experts, p.top_k,
                    /*start_expert_id=*/0, kPdl, stream);
            }
            // Expand + FP4 quantize into GEMM1's A (permuted_token_final_scales_ is null without fused finalize):
            // expand_tiles writes the launcher's bytes, the launcher itself where expand_tiles does not apply.
            if (expand_tiled(p, quant)) {
                expand_tiles(x, w, p, quant, kPdl, stream);
            } else {
                tk::expandInputRowsKernelLauncher<__nv_bfloat16, Fp4>(
                    x, w.inputs, topk_weights, /*permuted_scales=*/nullptr, w.permuted_row_to_unpermuted_row, p.m,
                    p.hidden, k, e, quant, quant.fp4.fc1.use_per_expert_act_scale, w.expert_first_token_offset,
                    w.act_sf, /*input_sf=*/nullptr, /*swizzled_input_sf=*/true, /*prequant_scales=*/nullptr, kPdl,
                    stream);
            }
        }
        if (decoded) {
            // P14: GEMM1 + SwiGLU + FP4 quantize, then GEMM2, on the group table.
            const auto *x_sf = reinterpret_cast<const uint8_t *>(w.act_sf);
            cere_status s = decode::gemm1_act(
                w.groups, reinterpret_cast<const uint8_t *>(w.inputs), x_sf, static_cast<const uint8_t *>(w13_weight),
                reinterpret_cast<const uint8_t *>(quant.fp4.fc1.weight_block_scale), quant.fp4.fc1.global_scale,
                quant.fp8.quant_fc2, quant.fp4.fc2.act_global_scale, quant.fp4.fc2.use_per_expert_act_scale,
                tensorrt_llm::common::getEnvDisableFP4QuantFastMath(), w.decode_act, w.decode_act_sf, w.decode_ready,
                shape, p.order, kPdl, stream);
            if (s == CERE_OK) {
                s = decode::gemm2(w.groups, w.decode_act, w.decode_act_sf, static_cast<const uint8_t *>(down_weight),
                                  reinterpret_cast<const uint8_t *>(quant.fp4.fc2.weight_block_scale),
                                  quant.fp4.fc2.global_scale, w.outputs, w.decode_ready, shape, p.order, kPdl, stream);
            }
            if (s != CERE_OK) {
                return s;
            }
        } else {
            // setupTmaWarpSpecializedInputs (non-min-latency, no fused finalize).
            Tma tma1{};
            Tma tma2{};
            tma1.configureWorkspace(w.tma_gemm1, e, w.gemm, w.gemm_bytes, Tma::FpXBlockScalingType::NVFP4);
            tma2.configureWorkspace(w.tma_gemm2, e, w.gemm, w.gemm_bytes, Tma::FpXBlockScalingType::NVFP4);
            tma1.enable_pdl = kPdl;
            tma2.enable_pdl = kPdl;
            tma1.fusion = Tma::EpilogueFusion::NONE;
            tma2.fusion = Tma::EpilogueFusion::NONE;
            tma1.swap_ab = gemm1.swap_ab;
            tma2.swap_ab = gemm2.swap_ab;
            const auto strides = fc.computeStridesTmaWarpSpecializedDispatch(
                w.expert_first_token_offset, tma1, tma2, p.m, mk, /*gemm1_n=*/2 * p.inter, /*gemm1_k=*/p.hidden,
                /*gemm2_n=*/p.hidden, /*gemm2_k=*/p.inter, e, w.inputs, w.inputs, w13, down, quant.fp8.dequant_fc1,
                quant.fp8.dequant_fc2, w.act_sf, w.act_sf, quant, /*bias1=*/nullptr, /*bias2=*/nullptr, w.outputs,
                w.outputs, /*router_scales=*/nullptr, w.permuted_row_to_unpermuted_row, kPdl, stream);
            tma1 = strides.first;
            tma2 = strides.second;
            // GEMM1 + SwiGLU + FP4 quantize of GEMM2's A (fc1_result over permuted_data, as FlashInfer aliases them):
            // at prefill gemm1's grouped GEMM with act_tiles in place of doActivation (the same bytes).
            if (mk > kSmallRows && act_tiled(gemm, gemm1, p, quant)) {
                gemm1_tiles(gemm, gemm1, tma1, w, p, quant, kPdl, stream);
            } else {
                FcRunner::gemm1(gemm, /*fp8_blockscale_gemm_runner=*/nullptr, w.inputs, w.inputs, w.outputs,
                                w.expert_first_token_offset, tma1, w13, /*fc1_expert_biases=*/nullptr,
                                /*num_valid_tokens_ptr=*/nullptr,
                                reinterpret_cast<__nv_bfloat16 const *>(quant.wo.fc1_weight_scales),
                                quant.fp8.dequant_fc1, quant.fp8.quant_fc2, w.act_sf, w.act_sf, quant, p.m, mk,
                                p.hidden, p.inter, e, tk::ActivationParams(tk::ActivationType::Swiglu), w.alpha_fc1,
                                /*bias_is_broadcast=*/true, stream, gemm1, /*min_latency_mode=*/false, nullptr,
                                nullptr, kPdl);
            }
            // GEMM2 into fc2_result (over glu_inter_result). min_latency_mode = true returns right after the grouped
            // GEMM (TMA warp-specialized, FP4: the mode's only other effects are those two checks), which skips
            // finalizeMoeRoutingKernel; every argument the GEMM reads is runMoe's.
            FcRunner::gemm2(gemm, /*fp8_blockscale_gemm_runner=*/nullptr, w.inputs, w.outputs,
                            /*final_output=*/nullptr, w.expert_first_token_offset, tma2, down,
                            /*fc2_expert_biases=*/nullptr,
                            reinterpret_cast<__nv_bfloat16 const *>(quant.wo.fc2_weight_scales),
                            quant.fp8.dequant_fc2, w.act_sf, quant, topk_weights, /*permuted_final_scales=*/nullptr,
                            w.unpermuted_row_to_permuted_row, w.permuted_row_to_unpermuted_row, topk_ids,
                            /*num_valid_tokens_ptr=*/nullptr, p.m, mk, p.hidden, p.hidden, p.inter, e, p.top_k,
                            w.alpha_fc2, /*use_lora=*/false, /*fc2_lora=*/nullptr, stream,
                            tk::MOEParallelismConfig{}, /*enable_alltoall=*/false, gemm2, /*min_latency_mode=*/true,
                            nullptr, nullptr, kPdl);
        }
    } catch (const std::exception &) {
        return CERE_ERR_INTERNAL;
    }
    if (cudaGetLastError() != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    out->rows = w.outputs;
    out->pos = w.unpermuted_row_to_permuted_row;
    out->x = reinterpret_cast<const uint8_t *>(w.inputs);
    out->x_sf = reinterpret_cast<const uint8_t *>(w.act_sf);
    out->act = decoded ? w.decode_act : reinterpret_cast<const uint8_t *>(w.inputs);
    out->act_sf = decoded ? w.decode_act_sf : reinterpret_cast<const uint8_t *>(w.act_sf);
    out->offsets = w.expert_first_token_offset;
    out->groups = decoded ? w.groups : nullptr;
    out->decoded = decoded;
    return CERE_OK;
}

}  // namespace cere::p3b
