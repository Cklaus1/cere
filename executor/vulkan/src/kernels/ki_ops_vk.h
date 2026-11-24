// Inkling S11: the Inkling-specific elementwise / row kernels on Vulkan (shaders/ki_*.comp), bit-equal to the CPU references
// executor/ref/inkling/*_ref.h (spec inkling design note section 5, G1; ktest/ki_test.cpp). Conventions of the Vulkan layer:
// addresses are buffer device addresses, calls record into the context's open command buffer (Context::begin ..
// submit_wait), the caller puts a barrier after a call before reading its outputs, nothing is allocated.
// Reduction lengths are the refs' fixed L (256 for 2304-wide rows, 32 for head_dim 128, 16 for d_rel 16): rows of any other
// width are CERE_ERR_UNSUPPORTED.
#pragma once

#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vki {

enum class Dt : uint32_t { kF32 = 0, kBF16 = 1 };

struct Tensor {  // a device tensor view: address and element type
    uint64_t addr = 0;
    Dt dt = Dt::kF32;
};

// Every pipeline of this header; with `coopmat` (moe_cm_device) also the int8 cooperative-matrix MoE tiles and their
// layout probe (never the mutant tiles).
std::vector<vk::PipelineSpec> pipeline_specs(bool coopmat);

// RMSNorm of `rows` rows of width n (2304 or 128): y = x / sqrt(mean(x^2) + eps), times w[i] when w.addr != 0
// (Inkling2LoopRMSNorm / UnweightedRMSNorm). out may alias x.
cere_status rms_norm(vk::Context &ctx, Tensor x, Tensor w, Tensor out, uint32_t n, uint32_t rows, float eps);

// L2 clip of `rows` head rows of width n (128 or 16), head = row % heads: y = x (* scale[head]); y *= min(thr / (|y| + 1e-6),
// 1); with tau != 0 (FP32 per token, token = row / heads): y = bf16(y) * tau[token] (Inkling2LoopL2Clip / ScaledL2Clip /
// LogScaledL2Clip). out may alias x.
cere_status l2_clip(vk::Context &ctx, Tensor x, Tensor scale, uint64_t tau, Tensor out, uint32_t n, uint32_t rows, uint32_t heads,
                    float thr);

// Causal short convolution (kernel 4) over T tokens of C channels, FP32 carried state [C][3] (oldest first) read from
// st_in and the next state written to st_out (distinct buffers). Static: w [C][4] (Inkling2LoopShortConvolution).
// Dynamic: w = static_weight [4][C], dyn [T][heads * 4] (Inkling2LoopDynamicShortConvolution).
cere_status conv_static(vk::Context &ctx, Tensor x, Tensor w, Tensor out, uint64_t st_in, uint64_t st_out, uint32_t T, uint32_t C);
cere_status conv_dynamic(vk::Context &ctx, Tensor x, Tensor static_w, Tensor dyn, Tensor out, uint64_t st_in, uint64_t st_out,
                         uint32_t T, uint32_t C, uint32_t heads);

// Fail-first mutants (ki_forward --mutant / ki_test only; the device runs none). G3, CUDA kernels/inkling/moe.h
// MoeMutant: kSoftmaxRouting: router weights = softmax of the 8 winners' logits x scaling; kGlobalActScale: NVFP4
// activation amax = 64 for every row (one global scale) instead of the per-token amax. D-I-EPB (ki_test only):
// kMoeBlockOrder: the cooperative-matrix NVFP4 grouped tiles add each K chunk's 16 block terms in reverse order.
// D-I-EPB (a2) (ki_test only): kAttnKeyOrder: the tiled attention spans (AttnRoute::kTiled) walk each 64-key tile's
// P x V fma chain in reverse key order; kMsgGateOrder: the tiled message gate (MsgRoute::kTiled) runs each d row's
// chain backwards.
enum class Mutant : uint32_t {
    kNone = 0, kSoftmaxRouting = 1, kGlobalActScale = 2, kMoeBlockOrder = 3, kAttnKeyOrder = 4, kMsgGateOrder = 5
};

// Per-token activation quantizers of the expert inputs (routed_quant.cuh is the contract, spec section 5), one row of K
// values (BF16 or FP32) per token. MXFP8 (K % 32 == 0): q E4M3 u8 [rows][K], s UE8M0 u8 [rows][K / 32]. NVFP4
// fouroversix_mae (K % 16 == 0, K <= 4096): q packed E2M1 u8 [rows][K / 2] (element 2i low nibble), s E4M3 u8
// [rows][K / 16], amax FP32 [rows]; c4 != 0: u32 [rows] += the count of blocks that chose the /4 scale (diagnostics).
cere_status quant_mxfp8(vk::Context &ctx, Tensor x, uint64_t q, uint64_t s, uint32_t K, uint32_t rows);
cere_status quant_nvfp4(vk::Context &ctx, Tensor x, uint64_t q, uint64_t s, uint64_t amax, uint64_t c4, uint32_t K, uint32_t rows,
                        Mutant mutant = Mutant::kNone);

// MoE router (Inkling2LoopTopKRouter): x BF16 [m][H], w BF16 [E][H], bias FP32 [E] -> ids int32 [m][8] (ascending
// expert id), weights FP32 [m][8] (score / (sum + 1e-20) * scaling). lg: FP32 [m][E] scratch (the logits between the
// two dispatches). H % 256 == 0, 8 <= E <= 128.
cere_status router(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t bias, uint64_t lg, uint64_t ids, uint64_t wts, uint32_t m,
                   uint32_t E, uint32_t H, float scaling, Mutant mutant = Mutant::kNone);

// Routed-expert products (shaders/ki_moe.comp; moe_ref.h order). fmt 0 = NVFP4 W4A4, 1 = MXFP8. A pair p = token * 8 +
// slot; ids int32 [pairs], wts FP32 [pairs] (the router's output). Activations as ki_quant writes them (aamax FP32 per
// row, NVFP4 only). Weights of all experts: w [E][N][K / 2 | K], ws [E][N][K / 16 | K / 32], wamax FP32 [E] (NVFP4),
// den = the layer's outer_scale_denominator (NVFP4).
struct ExpertW {
    uint64_t w = 0, ws = 0, wamax = 0;
    float den = 1.0f;
};
struct ActQ {
    uint64_t q = 0, s = 0, amax = 0;
};
// v BF16 [pairs][N / 2] = gamma * capped SwiGLU(bf16(x . w13[2i]), bf16(x . w13[2i + 1])), x row = p / 8. N = 2304.
// tab: 0 = per-pair tiling (decode); else moe_bucket's table for these ids (prefill: an expert's weight rows are read
// once per tile of its pairs). route (tab != 0, NVFP4): kAuto = each block's integer sum on int8 cooperative matrices
// when moe_coopmat(ctx), else (and kVector) by sdot4x8. Every tiling and route gives the same bits.
// mutant: kMoeBlockOrder only (cooperative-matrix route, else CERE_ERR_UNSUPPORTED); the others are ignored here.
enum class MoeRoute : uint32_t { kAuto = 0, kVector = 1 };
cere_status moe_gate_up(vk::Context &ctx, uint32_t fmt, ActQ x, ExpertW w13, uint64_t ids, uint64_t wts, uint64_t v, uint32_t N,
                        uint32_t K, uint32_t pairs, uint64_t tab = 0, MoeRoute route = MoeRoute::kAuto,
                        Mutant mutant = Mutant::kNone);
// y BF16 [pairs][N] = v[p] . w2[e][c] (v quantized per pair). tab, route, mutant as moe_gate_up.
cere_status moe_down(vk::Context &ctx, uint32_t fmt, ActQ v, ExpertW w2, uint64_t ids, uint64_t y, uint32_t N, uint32_t K,
                     uint32_t pairs, uint64_t tab = 0, MoeRoute route = MoeRoute::kAuto, Mutant mutant = Mutant::kNone);
// The cooperative-matrix route's preconditions: the device (moe_cm_device: VK_KHR_cooperative_matrix and 32-lane
// subgroups), then, once per context, the int32 16x16 accumulator's element -> (row, column) map, element i of lane l
// = (2 i + l / 16, l % 16) (the map K7 also checks: it shares vk::Probe::kK7MoeLayout). moe_probe: CERE_OK when it
// holds, CERE_ERR_UNSUPPORTED when it does not or the device cannot; records and submits (no command buffer may be
// open). moe_coopmat: the route is on (the probe passed).
bool moe_cm_device(const vk::Context &ctx);
cere_status moe_probe(vk::Context &ctx);
bool moe_coopmat(const vk::Context &ctx);
// Expert-grouped table of ids int32 [pairs] (E <= 128 experts) into tab (moe_tab_bytes(pairs) bytes); a barrier must
// separate it from the products that read it.
uint64_t moe_tab_bytes(uint32_t pairs);
cere_status moe_bucket(vk::Context &ctx, uint64_t ids, uint64_t tab, uint32_t pairs, uint32_t E);
// out BF16 [tokens][N] = sequential sum of the token's 8 y rows.
cere_status moe_combine(vk::Context &ctx, uint64_t y, uint64_t out, uint32_t N, uint32_t tokens);

// Residual-stream ops (shaders/ki_mix.comp; executor/ref/inkling depth_ref.h, msg_ref.h, embed_ref.h, softcap_ref.h).
// Rows are 2304 wide. depth_mix: S <= 5 states (FP32 or BF16 [M][2304]), queries FP32 [nq][2304]; out FP32
// [nq][M][2304].
cere_status depth_mix(vk::Context &ctx, const Tensor *states, uint32_t S, uint64_t queries, uint32_t nq, uint64_t out, uint32_t M,
                      float eps);
// out FP32 [M][2304] = message gate of h, msg (FP32 [M][2304]); down BF16 [64][4608], forget / gain BF16 [2304][64];
// d FP32 [M][64] scratch. Two dispatches with a barrier between (the caller still barriers before reading out). Route:
// kRow = a workgroup per (4 d rows, token), then per (256 columns, token); kTiled (D-I-EPB (a2)) = the same arithmetic
// with a workgroup per kMsgTileRows7 tokens (all 64 d rows), then per (256 columns, kMsgTileRows8 tokens), forget /
// gain 16-byte aligned (else CERE_ERR_ARGUMENT); kAuto = kTiled from kMsgTiledRows tokens when aligned, else kRow.
// Every route gives the same bits. mutant: kMsgGateOrder only, with kTiled (else CERE_ERR_UNSUPPORTED).
enum class MsgRoute : uint32_t { kAuto = 0, kRow = 1, kTiled = 2 };
constexpr uint32_t kMsgTileRows7 = 4, kMsgTileRows8 = 16;  // TT7 / TT8 of ki_mix.comp
constexpr uint32_t kMsgTiledRows = 16;
cere_status msg_gate(vk::Context &ctx, uint64_t h, uint64_t msg, uint64_t down, uint64_t forget, uint64_t gain, uint64_t d,
                     uint64_t out, uint32_t M, float eps, MsgRoute route = MsgRoute::kAuto, Mutant mutant = Mutant::kNone);
// out FP32 [n] = 30 tanh(x / 30), x BF16 or FP32.
cere_status softcap(vk::Context &ctx, Tensor x, uint64_t out, uint32_t n);
// out BF16 [M][2304] = weighted RMSNorm(table[ids[m]]); table BF16 [V][2304], ids int32 [M] (< V: the caller checks).
cere_status embed_norm(vk::Context &ctx, uint64_t table, uint64_t ids, uint64_t w, uint64_t out, uint32_t M, float eps);
// out FP32 [n] = boundary ? o : h + o (h, o BF16).
cere_status attn_residual(vk::Context &ctx, uint64_t h, uint64_t o, uint64_t out, uint32_t n, bool boundary);
// out BF16 [n] = r + d (r FP32, d BF16).
cere_status layer_out(vk::Context &ctx, uint64_t r, uint64_t d, uint64_t out, uint32_t n);

// Attention core (shaders/ki_attn.comp; executor/ref/inkling/attn_ref.h order), one sequence per call: n query rows at
// positions pos0 .. pos0 + n - 1. kind 0 = sliding (window and ring of 1024: key j in cache slot j % 1024), 1 = full
// (pool row slots[j], int32, or j when slots == 0). k / v: the call's new rows (post-processed BF16 [n][ks | vs],
// 8 * 128 used); kc / vc: the cache [slots][8 * 128] BF16, read for keys before pos0.
struct AttnSeq {
    uint32_t kind = 0, pos0 = 0, n = 0, ks = 1024, vs = 1024;
    uint64_t k = 0, v = 0, kc = 0, vc = 0, slots = 0;
};
// The query side (BF16, after the q / rel posts): q [n][qs] (24 * 128), r [n][rs] (24 * 16), sink [n][ss] (24), gate
// [n][gs] (24 * 128), proj [16][1024]; out BF16 [n][3072] = bf16(bf16(softmax-with-sink . v) * sigmoid(gate)).
// part (optional, FP32 scratch of part_bytes): the split path's span partials, kAttnPartBytes per (row, head, span);
// attention falls back to one workgroup per (row, head) when they do not fit.
struct AttnRows {
    uint64_t q = 0, r = 0, proj = 0, sink = 0, gate = 0, out = 0, part = 0;
    uint32_t qs = 3072, rs = 384, ss = 24, gs = 3072;
    uint64_t part_bytes = 0;
};
constexpr uint64_t kAttnPartBytes = 130 * 4;
// Query rows [r0, r0 + rows) of the call only (rows are independent: a long prefill's attention is split into row
// blocks, one submit each, under AMD's ring timeout; same bits as one dispatch). With part: 256-key spans in parallel,
// then their in-order fold (same bits); it records a barrier between the two. The span partials come from one
// workgroup per (row, head, span) (kSplit; kAuto below kAttnTiledRows rows) or, D-I-EPB (a2), per (kAttnTileRows rows,
// KV head, span) with every key / value load shared by the block's 3 x kAttnTileRows query heads (kTiled; kAuto from
// kAttnTiledRows rows). Every route gives the same bits; kSplit / kTiled need part (else CERE_ERR_ARGUMENT).
// mutant: kAttnKeyOrder only, with kTiled (else CERE_ERR_UNSUPPORTED).
enum class AttnRoute : uint32_t { kAuto = 0, kSplit = 1, kTiled = 2 };
constexpr uint32_t kAttnTileRows = 4;    // RB of ki_attn_m4.spv
constexpr uint32_t kAttnTiledRows = 16;  // kAuto's first tiled row count (prefill; decode and verify keep kSplit)
cere_status attention(vk::Context &ctx, const AttnSeq &s, const AttnRows &q, uint32_t r0 = 0, uint32_t rows = UINT32_MAX,
                      AttnRoute route = AttnRoute::kAuto, Mutant mutant = Mutant::kNone);
// Store the call's new k / v rows into the cache (sliding: the last <= 1024 rows). Record it after the call's attention
// (and a barrier): a ring slot it overwrites may hold a key that attention reads.
cere_status kv_store(vk::Context &ctx, const AttnSeq &s);

// Portable dense projection (shaders/ki_dense.comp): y[r][j] = sum_i x[r][i] w[j][i], one fma chain per output in
// ascending i; x BF16 [m][k], w BF16 [n][k] (k even, n * k and m * k < 2^31), y FP32 (out_f32) or BF16 [m][n].
// Row-independent. Where K2 (vkk2::run) cannot run, and the forward's deterministic baseline.
cere_status dense(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t y, uint32_t m, uint32_t n, uint32_t k, bool out_f32);

}  // namespace cere::vki
