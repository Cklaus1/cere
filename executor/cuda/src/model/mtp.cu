// M2.9 MTP draft layer (model/mtp.h): sizing, INSTANTIATE and the forward, composed from the K1-K3 and K13 kernels,
// the target's attention (forward.cpp attention_rows: K6, K9, P4 per build) plus three small kernels of its own (the
// 10,240-wide Gemma RMSNorm of pre_fc_norm_hidden, the embedding broadcast add of the fusion, and argmax + token map
// over the hot logits).
#include "model/mtp.h"

#include "kernels/k1_primitives.h"
#include "kernels/k6_attention.h"
#include "kernels/k9_qsa.h"
#include "model/layout.h"
#include "model/plan.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace cere::model::mtp {
namespace {

constexpr uint32_t kNormThreads = 256;
constexpr uint32_t kArgmaxThreads = 1024;
constexpr uint32_t kAddThreads = 256;

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

cere_status cuda_status(cudaError_t e) { return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; }

#define MTP_TRY(expr)                         \
    do {                                      \
        const cere_status s_ = (expr);        \
        if (s_ != CERE_OK) return s_;         \
    } while (0)
#define MTP_TRY_CUDA(expr) MTP_TRY(cuda_status(expr))

// The MTP MoE block's dims (the target's: H, E, K, I, S).
constexpr kernels::MoeDims kMoeDims{kHidden, kExperts, kTopK, kMoeInter, kSharedInter};

// Rows of one hot-logits chunk: the lm_head runs `rows` rows at a time (the rows plan) into the shared scratch.
uint64_t lm_chunk_rows(uint64_t max_m) { return std::min<uint64_t>(kRows, max_m); }

// pre_fc_norm_hidden: y[r] = bf16(x[r] * rsqrt(mean(x[r]^2) + eps) * (1 + w)) over the whole 10,240-wide row (one
// GemmaRMSNorm, sglang/srt/layers/layernorm.py), F32 arithmetic; one block per row.
__global__ void norm_hidden_kernel(const bf16 *__restrict__ x, const bf16 *__restrict__ w, float eps,
                                   bf16 *__restrict__ y) {
    const uint64_t row = blockIdx.x;
    x += row * kHcHidden;
    y += row * kHcHidden;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < kHcHidden; i += kNormThreads) {
        const float v = __bfloat162float(x[i]);
        ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float part[kNormThreads / 32];
    __shared__ float total;
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (uint32_t i = 0; i < kNormThreads / 32; ++i) t += part[i];
        total = t;
    }
    __syncthreads();
    const float r = rsqrtf(total / float(kHcHidden) + eps);
    for (uint32_t i = threadIdx.x; i < kHcHidden; i += kNormThreads) {
        y[i] = __float2bfloat16(__bfloat162float(x[i]) * r * (1.0f + __bfloat162float(w[i])));
    }
}

// The fusion's broadcast add (_fuse_residual_linear_shared): hyper[r, s, :] = bf16(hyper[r, s, :] + e[r, :]).
__global__ void add_embedding_kernel(bf16 *__restrict__ hyper, const bf16 *__restrict__ e, uint64_t elements) {
    const uint64_t i = uint64_t(blockIdx.x) * kAddThreads + threadIdx.x;
    if (i >= elements) return;
    const uint64_t row = i / kHcHidden;
    const uint64_t col = i % kHidden;
    hyper[i] = __float2bfloat16(__bfloat162float(hyper[i]) + __bfloat162float(e[row * kHidden + col]));
}

// draft[r] = map[argmax(logits[r, :count])], the lowest index among equal maxima (torch.argmax); one block per row.
// A NaN logit counts as -inf (a NaN would stick as a thread's best and as the tree's winner): the draft is the best
// number of the row, id 0 when there is none.
__global__ void argmax_map_kernel(const bf16 *__restrict__ logits, uint32_t count, const uint32_t *__restrict__ map,
                                  uint32_t *__restrict__ draft) {
    const bf16 *row = logits + uint64_t(blockIdx.x) * count;
    float best = -INFINITY;
    uint32_t at = 0xffffffffu;
    for (uint32_t i = threadIdx.x; i < count; i += kArgmaxThreads) {
        const float raw = __bfloat162float(row[i]);
        const float v = raw == raw ? raw : -INFINITY;
        if (v > best || at == 0xffffffffu) {
            best = v;
            at = i;
        }
    }
    __shared__ float vals[kArgmaxThreads];
    __shared__ uint32_t ids[kArgmaxThreads];
    vals[threadIdx.x] = best;
    ids[threadIdx.x] = at;
    __syncthreads();
    for (uint32_t s = kArgmaxThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = vals[threadIdx.x + s];
            const uint32_t oi = ids[threadIdx.x + s];
            const bool take = oi != 0xffffffffu &&
                              (ids[threadIdx.x] == 0xffffffffu || ov > vals[threadIdx.x] ||
                               (ov == vals[threadIdx.x] && oi < ids[threadIdx.x]));
            if (take) {
                vals[threadIdx.x] = ov;
                ids[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) draft[blockIdx.x] = map[ids[0]];
}

// The rows of one session in a forward: n rows at positions p0 .. p0 + n - 1 (device_positions: the same, read on the
// device by the draft graph's attention_rows_at).
struct Segment {
    SessionState *s = nullptr;
    uint64_t p0 = 0;
    uint32_t n = 0;
    const int64_t *device_positions = nullptr;
};

bool has_entry(const SessionState &s) {
    return s.mtp_k_cache != nullptr && s.mtp_v_cache != nullptr && s.mtp_qsa_keys != nullptr &&
           s.mtp_qsa_ring != nullptr && s.slot_map != nullptr;
}

// DS-3 (draft-span design note): the device positions of attention_segments, row t's from the host by value.
struct RowPositions {
    int64_t p[k1::kKvRowsMax];
};

__global__ void row_positions_kernel(RowPositions r, uint32_t m, int64_t *__restrict__ out) {
    if (threadIdx.x < m) out[threadIdx.x] = r.p[threadIdx.x];
}

// Whether run's attention goes through attention_segments: eager rows (no device positions, no class pin) of >= 2
// segments, at most k1::kKvRowsMax rows, every segment dense with a prefix (1 <= p0, p0 + n <= k6::kMaxContext: its
// attention_rows call is K6's dense one, fewer than p4::kMinRows rows so never P4) of a session at rope_delta 0 (one
// table for every row; m213l design note C4), the QSA plan's index-key GEMM on the small-M kernel at m rows, and
// every segment one row unless the caller reads each segment's last row alone (`last_only`, run's tails).
bool segments_batched(const Workspace &ws, const Segment *segs, uint32_t segments, uint32_t m, uint32_t cls,
                      bool last_only) {
    if (segments < 2 || cls != 0 || m > uint32_t(k1::kKvRowsMax)) return false;
    if (ws.max_context_tokens > uint64_t(k6::kMaxContext) &&
        (!ws.attn_index_qk.small_m() || m > ws.attn_index_qk.max_m())) {
        return false;
    }
    for (uint32_t i = 0; i < segments; ++i) {
        const Segment &g = segs[i];
        if (g.device_positions != nullptr || g.p0 == 0 || g.p0 + g.n > uint64_t(k6::kMaxContext) ||
            g.s->rope_delta != 0) {
            return false;
        }
        if (g.n > 1 && !last_only) return false;
    }
    return true;
}

// DS-3: the attention of every segment at once (segments_batched), on the m rows of ws.qkv / ws.mixed into
// ws.attn_out: the QSA plan's index keys in one small-M GEMM over the m rows (row r bit-equal to its row of a
// segment's call) and each segment's compressed-key store as attention_rows makes it, then one k6::prologue_chains_at
// and one k6::dense_attention_chains_at over the m rows, row t on its session's MTP pool entry at position p0 + j
// (every row's K/V bit for bit the segment's prologue; row t's output the one-token dense call at its position).
// A one-row segment's row is therefore its attention_rows call bit for bit; a longer segment's last row is the last
// row of its own multi-token dense call (the same split count and keys per split, kv_len p0 + n, and the per-row
// arithmetic of the one-token tile loop), its other rows are not.
cere_status attention_segments(const ModelWeights &w, Workspace &ws, const Segment *segs, uint32_t segments,
                               uint32_t m, cudaStream_t stream) {
    const AttentionWeights &aw = w.mtp.attn;
    const float eps = float(w.config.rms_norm_eps);
    const size_t t = m;
    uint8_t *ptr = static_cast<uint8_t *>(ws.scratch);
    bf16 *q = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kAttnOutWidth * sizeof(bf16));
    bf16 *k_rot = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kKvWidth * sizeof(bf16));
    bf16 *gate = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kAttnOutWidth * sizeof(bf16));
    const bool qsa = ws.max_context_tokens > uint64_t(k6::kMaxContext);
    bf16 *qk = reinterpret_cast<bf16 *>(ptr);
    if (qsa) ptr += align256(t * k9::kQkWidth * sizeof(bf16));
    int64_t *positions = reinterpret_cast<int64_t *>(ptr);
    ptr += align256(t * sizeof(int64_t));
    const size_t used = size_t(ptr - static_cast<uint8_t *>(ws.scratch));
    if (used > ws.scratch_bytes) return CERE_ERR_INTERNAL;
    const size_t rest = ws.scratch_bytes - used;
    if (qsa) MTP_TRY(ws.attn_index_qk.run(ws.mixed, aw.indexer_qk, m, qk, stream));
    RowPositions pos{};
    k1::KvRows kv{};
    for (uint32_t i = 0, at = 0; i < segments; at += segs[i].n, ++i) {
        const Segment &g = segs[i];
        const SessionState &s = *g.s;
        if (qsa) {
            MTP_TRY(k9::qsa_compress_store(qk + uint64_t(at) * k9::kQkWidth + k9::kIndexHeads * k9::kIndexHeadDim,
                                           k9::kQkWidth, aw.indexer_k_norm, eps, w.rope_cos_sin,
                                           k9_bounds(w, ws, w.rope_cos_sin, s.max_context_tokens), nullptr,
                                           s.slot_map, int64_t(g.p0), int64_t(g.n), s.mtp_qsa_ring, s.mtp_qsa_keys,
                                           stream));
        }
        for (uint32_t j = 0; j < g.n; ++j) {
            kv.k_cache[at + j] = s.mtp_k_cache;
            kv.v_cache[at + j] = s.mtp_v_cache;
            kv.slots[at + j] = s.slot_map;
            pos.p[at + j] = int64_t(g.p0 + j);
        }
    }
    row_positions_kernel<<<1, k1::kKvRowsMax, 0, stream>>>(pos, m, positions);
    MTP_TRY_CUDA(cudaGetLastError());
    const bf16 *k = ws.qkv + kQGateWidth;
    const bf16 *v = k + kKvWidth;
    MTP_TRY_CUDA(k6::prologue_chains_at(ws.qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm, aw.k_norm, eps,
                                        w.rope_cos_sin, kv, positions, int64_t(m), q, k_rot, gate, stream));
    MTP_TRY_CUDA(k6::dense_attention_chains_at(q, gate, int64_t(m), positions, kv, ptr, rest, ws.attn_out,
                                               kAttnOutWidth, stream));
    return CERE_OK;
}

// The final mixer (plan::Slot::kNorm) of n hc rows `hyper` [n, 10,240] on `mixer` into ws.mixed; `sink` gets its
// output as "mixer.y" of `stage_rows` rows.
cere_status final_mixer(const ModelWeights &w, Workspace &ws, const kernels::HcPlans &mixer, const bf16 *hyper,
                        uint32_t n, StageSink sink, void *ctx, uint32_t stage_rows, cudaStream_t stream) {
    const MtpWeights &mw = w.mtp;
    MTP_TRY(kernels::hc_mix(mixer, hyper, mw.mixer.norm, mw.mixer.down, mw.mixer.up, n, ws.normed, ws.mixed,
                            ws.scratch, stream));
    if (sink != nullptr) MTP_TRY(sink(ctx, "mixer.y", ws.mixed, stage_rows, kHidden, stream));
    return CERE_OK;
}

// The hot-logits draft (plan::Slot::kLmHead) of ws.mixed's n rows: the hot lm_head in chunks of mp.rows rows (rows
// plan) into the scratch, argmax and the map into draft_ids [n] (device); `logits` (when non-null) gets the hot logits.
cere_status hot_lm_head(const ModelWeights &w, Workspace &ws, uint32_t n, uint32_t *draft_ids, bf16 *logits,
                        cudaStream_t stream) {
    const MtpWeights &mw = w.mtp;
    const Workspace::Mtp &mp = ws.mtp;
    bf16 *hot = static_cast<bf16 *>(ws.scratch);
    const uint64_t count = mw.count;
    for (uint32_t i = 0; i < n; i += mp.rows) {
        const uint32_t c = std::min(mp.rows, n - i);
        MTP_TRY(mp.lm_head_rows.run(ws.mixed + uint64_t(i) * kHidden, mw.lm_head, c, hot, stream));
        argmax_map_kernel<<<c, kArgmaxThreads, 0, stream>>>(hot, mw.count, mw.map, draft_ids + i);
        MTP_TRY_CUDA(cudaGetLastError());
        if (logits != nullptr) {
            MTP_TRY_CUDA(cudaMemcpyAsync(logits + uint64_t(i) * count, hot, uint64_t(c) * count * sizeof(bf16),
                                         cudaMemcpyDeviceToDevice, stream));
        }
    }
    return CERE_OK;
}

// The hot-logits draft of n hc rows `hyper` [n, 10,240]: the draft plan's last two ops, final_mixer then hot_lm_head.
// On the rows plans (n <= mp.rows) row r equals the call on that row alone.
cere_status score_rows(const ModelWeights &w, Workspace &ws, const kernels::HcPlans &mixer, const bf16 *hyper,
                       uint32_t n, uint32_t *draft_ids, bf16 *logits, cudaStream_t stream) {
    MTP_TRY(final_mixer(w, ws, mixer, hyper, n, nullptr, nullptr, n, stream));
    return hot_lm_head(w, ws, n, draft_ids, logits, stream);
}

// A draft plan op mtp.cu has a launch for: every slot but embed.ple and the linear (GDN) mixer.
constexpr bool implemented(const plan::Plan &p) {
    for (const plan::Op &op : p) {
        if (op.slot == plan::Slot::kEmbedPle || op.slot == plan::Slot::kAttentionLinearGdn) return false;
    }
    return true;
}
static_assert(implemented(plan::kDraft), "the draft head's plan has a slot mtp.cu does not implement");

// The GatedResidual weights of a mix / combine op: the layer's attn or mlp mix, or the final mixer.
const HcWeights &hc_weights(const MtpWeights &mw, plan::Mix mix) {
    return mix == plan::Mix::kAttn ? mw.attn_hc : mix == plan::Mix::kMlp ? mw.mlp_hc : mw.mixer;
}

// The forward of call.m = Σ segs[i].n rows, segment i's rows after segment i - 1's: every launch but the attention
// runs once on all m rows, the attention once per segment on its session's MTP pool entry (the launches of a forward
// of that segment alone) or, when segments_batched, once over every segment (attention_segments: each row read
// bit-equal to its segment's call). Without `score` it ends at hc_out (no draft). `tails` (non-null; no score, no
// stage sink, the rows plans): the caller reads each segment's last row alone, so from the MoE on the forward runs on
// those rows (gathered in segment order, each bit-equal to its row of the full call: every launch from there is per
// row on the rows plans) and writes them to tails [segments, 10,240] instead of hc_out. The caller checked the call
// and the segments.
cere_status run(const ModelWeights &w, Workspace &ws, const Segment *segs, uint32_t segments, const MtpCall &call,
                bf16 *logits, bool score, bf16 *tails, cudaStream_t stream) {
    const uint32_t m = call.m;
    // The rows every row-count choice is made for (m210 D15⁗′): call.cls when the call stands for a longer one.
    const uint32_t cls = call.cls;
    const MtpWeights &mw = w.mtp;
    const Workspace::Mtp &mp = ws.mtp;
    const bool rows = (cls != 0 ? cls : m) <= mp.rows;
    const kernels::HcPlans &hc = rows ? mp.hc_rows : ws.hc;
    const kernels::GemmPlan &fc = rows ? mp.fc_rows : mp.fc;
    const kernels::GemmPlan &qkv = rows ? mp.attn_qkv_rows : ws.attn_qkv;
    const kernels::GemmPlan &o = rows ? mp.attn_o_rows : ws.attn_o;
    const kernels::MoePlans moe{rows ? &mp.router_rows : &ws.moe_router,
                                rows ? &mp.shared_gate_up_rows : &ws.moe_shared_gate_up,
                                rows ? &mp.shared_down_rows : &ws.moe_shared_down, cls};
    const float eps = float(w.config.rms_norm_eps);
    const uint64_t hc_bytes = uint64_t(m) * kHcHidden * sizeof(bf16);
    const auto stage = [&](const char *name, const bf16 *values, uint64_t cols) {
        return call.stage_sink == nullptr ? CERE_OK : call.stage_sink(call.stage_ctx, name, values, m, cols, stream);
    };

    // The draft plan (plan::kDraft, ST1e) in its launch order: embed.tokens (the MTP fusion), the layer's attn mix,
    // attention, combine, mlp mix, K13 MoE and combine on ws.hyper, then the final mixer and the hot lm_head.
    const uint32_t first = call.last_only ? m - 1 : 0;  // last_only: the draft of row m - 1 alone (rows plans)
    for (const plan::Op &op : plan::kDraft) {
        switch (op.slot) {
        case plan::Slot::kEmbedTokens: {
            // Fusion: e = fc_embedding(norm_e(embed(next))) in ws.mixed; ws.hyper = fc_hidden(norm_h(hc_in) per
            // stream) + e.
            MTP_TRY_CUDA(k1::embedding_gather(w.embed, w.config.vocab_size, kHidden,
                                              reinterpret_cast<const int32_t *>(call.next_ids), m, ws.block_out,
                                              stream));
            MTP_TRY_CUDA(k1::rmsnorm(ws.block_out, kHidden, mw.norm_embedding, 1, true, eps, m, kHidden,
                                     ws.block_out, kHidden, stream));
            MTP_TRY(fc.run(ws.block_out, mw.fc_embedding, m, ws.mixed, stream, cls));
            MTP_TRY(stage("fusion.e", ws.mixed, kHidden));
            norm_hidden_kernel<<<m, kNormThreads, 0, stream>>>(call.hc_in, mw.norm_hidden, eps, ws.normed);
            MTP_TRY_CUDA(cudaGetLastError());
            const uint32_t streams = kHc * m;
            if (rows) {
                for (uint32_t i = 0; i < streams; i += mp.rows) {
                    MTP_TRY(fc.run(ws.normed + uint64_t(i) * kHidden, mw.fc_hidden, std::min(mp.rows, streams - i),
                                   ws.hyper + uint64_t(i) * kHidden, stream));
                }
            } else {
                MTP_TRY(fc.run(ws.normed, mw.fc_hidden, streams, ws.hyper, stream, kHc * cls));
            }
            const uint64_t elements = uint64_t(m) * kHcHidden;
            add_embedding_kernel<<<uint32_t((elements + kAddThreads - 1) / kAddThreads), kAddThreads, 0, stream>>>(
                ws.hyper, ws.mixed, elements);
            MTP_TRY_CUDA(cudaGetLastError());
            MTP_TRY(stage("fusion.hyper", ws.hyper, kHcHidden));
            break;
        }
        case plan::Slot::kMixHc: {
            // A mix runs normed when the combine before it wrote its norm (hc_combine_norm).
            const HcWeights &g = hc_weights(mw, op.mix);
            if (op.fused) {
                MTP_TRY(kernels::hc_mix_normed(hc, ws.normed, g.down, g.up, m, ws.mixed, ws.scratch, stream,
                                               kernels::HcMixPath::kAuto, cls));
            } else {
                MTP_TRY(kernels::hc_mix(hc, ws.hyper, g.norm, g.down, g.up, m, ws.normed, ws.mixed, ws.scratch,
                                        stream, kernels::HcMixPath::kAuto, cls));
            }
            const bool attn = op.mix == plan::Mix::kAttn;
            MTP_TRY(stage(attn ? "attn.normed" : "mlp.normed", ws.normed, kHcHidden));
            MTP_TRY(stage(attn ? "attn.mixed" : "mlp.mixed", ws.mixed, kHidden));
            break;
        }
        case plan::Slot::kAttentionFull: {
            MTP_TRY(qkv.run(ws.mixed, mw.attn.qkv, m, ws.qkv, stream, cls));
            MTP_TRY(stage("attn.qkv", ws.qkv, kQkvWidth));
            // The MTP layer's attention, per segment: forward.cpp attention_rows on the session's MTP pool entry
            // (new rows at p0 .. p0 + n - 1) with the call's rotary inputs (one segment) or the session's position
            // + rope_delta table, or its device-position twin attention_rows_at (the draft graph); every segment at
            // once through attention_segments when segments_batched.
            const bool batched = segments_batched(ws, segs, segments, m, cls, tails != nullptr);
            if (batched) MTP_TRY(attention_segments(w, ws, segs, segments, m, stream));
            for (uint32_t i = 0, at = 0; !batched && i < segments; at += segs[i].n, ++i) {
                const Segment &g = segs[i];
                const SessionState &s = *g.s;
                const AttnKv kv{s.mtp_k_cache, s.mtp_v_cache, s.mtp_qsa_ring, s.mtp_qsa_keys, s.slot_map,
                                int64_t(s.max_context_tokens)};
                bf16 *qkv_rows = ws.qkv + uint64_t(at) * kQkvWidth;
                const bf16 *mixed_rows = ws.mixed + uint64_t(at) * kHidden;
                bf16 *out_rows = ws.attn_out + uint64_t(at) * kAttnOutWidth;
                if (g.device_positions != nullptr) {
                    MTP_TRY(attention_rows_at(w, mw.attn, ws, kv, g.device_positions, qkv_rows, mixed_rows, out_rows,
                                              g.n, stream));
                } else {
                    const Rope rope = segments == 1 && call.rope.table != nullptr ? call.rope : Rope{rope_table(w, s)};
                    MTP_TRY(attention_rows(w, mw.attn, ws, kv, int64_t(g.p0), qkv_rows, mixed_rows, out_rows, g.n,
                                           rope, stream, cls, segments == 1 ? call.end_kv : nullptr));
                }
            }
            MTP_TRY(stage("attn.out", ws.attn_out, kAttnOutWidth));
            MTP_TRY(o.run(ws.attn_out, mw.attn.o_proj, m, ws.block_out, stream, cls));
            MTP_TRY(stage("attn.block_out", ws.block_out, kHidden));
            break;
        }
        case plan::Slot::kCombineHc: {
            // A fused combine also writes the next mix's norm into ws.normed (kernels::hc_combine_norm,
            // bit-identical).
            const HcWeights &g = hc_weights(mw, op.mix);
            if (op.fused) {
                MTP_TRY(kernels::hc_combine_norm(ws.hc.shape, ws.block_out, ws.hyper, ws.normed, g.inject,
                                                 hc_weights(mw, op.target_mix).norm, m, ws.hyper, stream));
            } else {
                MTP_TRY(kernels::hc_combine(ws.hc.shape, ws.block_out, ws.hyper, ws.normed, g.inject, m, ws.hyper,
                                            stream));
            }
            if (op.mix == plan::Mix::kAttn) MTP_TRY(stage("attn.hyper", ws.hyper, kHcHidden));
            break;
        }
        case plan::Slot::kFfnMoe: {
            if (tails != nullptr) {
                if (score || call.stage_sink != nullptr || !rows || segments > uint32_t(k1::kMaxCopySegments) / 3) {
                    return CERE_ERR_ARGUMENT;
                }
                // Each segment's last row p: its MoE input (ws.mixed) to row p of ws.attn_out as [segments, 2,560],
                // its mlp normed streams to row p of ws.qkv as [segments, 10,240] (both dead after the attention),
                // its hyper row to row p of tails; then the MoE and the combine on those `segments` rows, and the
                // call ends there (the caller scores the tails).
                k1::CopySegment copies[k1::kMaxCopySegments];
                for (uint32_t i = 0, at = 0; i < segments; ++i) {
                    at += segs[i].n;
                    const uint64_t a = at - 1;
                    copies[3 * i] = k1::CopySegment{ws.attn_out + uint64_t(i) * kHidden, 0, ws.mixed + a * kHidden,
                                                    0, kHidden * sizeof(bf16), 1};
                    copies[3 * i + 1] = k1::CopySegment{ws.qkv + uint64_t(i) * kHcHidden, 0,
                                                        ws.normed + a * kHcHidden, 0, kHcHidden * sizeof(bf16), 1};
                    copies[3 * i + 2] = k1::CopySegment{tails + uint64_t(i) * kHcHidden, 0, ws.hyper + a * kHcHidden,
                                                        0, kHcHidden * sizeof(bf16), 1};
                }
                MTP_TRY_CUDA(k1::copy_segments(copies, int(3 * segments), stream));
                MTP_TRY(kernels::moe_bf16_block(kMoeDims, mw.moe, moe, ws.attn_out, segments, 0, ws.block_out,
                                                ws.scratch, ws.scratch_bytes, stream));
                return kernels::hc_combine(ws.hc.shape, ws.block_out, tails, ws.qkv, mw.mlp_hc.inject, segments,
                                           tails, stream);
            }
            // A last_only call reads row m - 1 alone (its draft and hc_out row; the callers drop hc_out rows
            // [0, m - 1)), so the routed experts run on that row only (bit-equal, kernels/k13_moe_bf16.h); a staged
            // call keeps every row.
            const uint32_t moe_first = call.last_only && call.stage_sink == nullptr ? m - 1 : 0;
            MTP_TRY(kernels::moe_bf16_block(kMoeDims, mw.moe, moe, ws.mixed, m, moe_first, ws.block_out, ws.scratch,
                                            ws.scratch_bytes, stream));
            MTP_TRY(stage("mlp.block_out", ws.block_out, kHidden));
            break;
        }
        case plan::Slot::kNorm:
            // The layers' output rows to hc_out; without `score` the call ends there (no draft).
            MTP_TRY_CUDA(cudaMemcpyAsync(call.hc_out, ws.hyper, hc_bytes, cudaMemcpyDeviceToDevice, stream));
            if (!score) return CERE_OK;
            MTP_TRY(final_mixer(w, ws, call.last_only ? mp.hc_rows : hc, ws.hyper + uint64_t(first) * kHcHidden,
                                m - first, call.stage_sink, call.stage_ctx, m, stream));
            break;
        case plan::Slot::kLmHead:
            MTP_TRY(hot_lm_head(w, ws, m - first, call.draft_ids, logits, stream));
            break;
        case plan::Slot::kEmbedPle:
        case plan::Slot::kAttentionLinearGdn:
            return CERE_ERR_INTERNAL;  // not in a draft plan (static_assert implemented)
        }
    }
    return CERE_OK;
}

cere_status forward_impl(const ModelWeights &w, Workspace &ws, SessionState &s, const MtpCall &call, bf16 *logits,
                         cudaStream_t stream) {
    const uint32_t m = call.m;
    if (!w.has_draft || w.mtp.lm_head == nullptr || w.mtp.map == nullptr || w.mtp.count == 0 || ws.mtp.rows == 0 ||
        !has_entry(s) || m == 0 || m > ws.max_tokens || call.positions == nullptr || call.next_ids == nullptr ||
        call.hc_in == nullptr || call.hc_out == nullptr || call.draft_ids == nullptr ||
        (call.cls != 0 && (call.cls < m || call.cls > ws.max_tokens)) ||
        (call.device_positions != nullptr && (call.rope.table != nullptr || s.rope_delta != 0 ||
                                              call.end_kv != nullptr))) {
        return CERE_ERR_ARGUMENT;
    }
    const uint64_t p0 = call.positions[0];
    for (uint32_t r = 1; r < m; ++r) {
        if (call.positions[r] != p0 + r) return CERE_ERR_ARGUMENT;
    }
    if (p0 + m > s.max_context_tokens || p0 + m > w.max_context_tokens) return CERE_ERR_ARGUMENT;
    const Segment seg{&s, p0, m, call.device_positions};
    return run(w, ws, &seg, 1, call, logits, true, nullptr, stream);
}

}  // namespace

uint64_t derived_bytes(const cere_qwen4_exp_config &cfg, uint64_t count) { return layout::mtp_derived_bytes(cfg, count); }

uint64_t scratch_bytes(const cere_qwen4_exp_config &, uint64_t max_m, uint64_t, uint64_t count) {
    const uint64_t hot = align256(size_t(lm_chunk_rows(max_m) * count * sizeof(bf16)));
    return std::max<uint64_t>(kernels::moe_bf16_workspace_bytes(kMoeDims, uint32_t(max_m)), hot);
}

uint64_t hc_rows_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_m) { return layout::mtp_hc_rows_bytes(cfg, max_m); }

uint64_t workspace_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_m, uint64_t max_context_tokens,
                         uint64_t count) {
    return model::workspace_bytes(cfg, max_m, max_context_tokens, count) -
           model::workspace_bytes(cfg, max_m, max_context_tokens);
}

cere_status upload_map(ModelWeights *w, const std::vector<uint32_t> &map, cudaStream_t stream) {
    if (w == nullptr || !w->has_draft || w->mtp.map == nullptr || map.size() != w->mtp.count) return CERE_ERR_ARGUMENT;
    MTP_TRY_CUDA(cudaMemcpyAsync(w->mtp.map, map.data(), map.size() * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                 stream));
    return cuda_status(cudaStreamSynchronize(stream));
}

cere_status instantiate(ModelWeights *w, Workspace *ws, cublasLtHandle_t gemm, cudaStream_t stream) {
    if (w == nullptr || ws == nullptr || gemm == nullptr || !w->has_draft || w->mtp.lm_head == nullptr ||
        w->mtp.map == nullptr || w->mtp.count == 0 || ws->max_tokens == 0) {
        return CERE_ERR_ARGUMENT;
    }
    using kernels::GemmOut;
    // A failing step is named on stderr (as the device's other INSTANTIATE failures): the status alone does not say
    // which of the plans cuBLASLt refused.
    const auto step = [](const char *name, cere_status st) {
        if (st != CERE_OK) std::fprintf(stderr, "cere: mtp instantiate: %s failed (status %d)\n", name, int(st));
        return st;
    };
    // The hot lm_head: target lm_head rows at the map (a row gather, bit-exact), in chunks of kGatherRows rows
    // (k1::embedding_gather's row count is a grid dimension, at most 65,535; the map has up to vocab entries).
    constexpr uint64_t kGatherRows = 32768;
    for (uint64_t r0 = 0; r0 < w->mtp.count; r0 += kGatherRows) {
        const uint64_t n = std::min<uint64_t>(kGatherRows, w->mtp.count - r0);
        MTP_TRY(step("hot lm_head gather",
                     cuda_status(k1::embedding_gather(w->lm_head, w->config.vocab_size, kHidden,
                                                      reinterpret_cast<const int32_t *>(w->mtp.map) + r0,
                                                      int64_t(n), w->mtp.lm_head + r0 * kHidden, stream))));
    }
    Workspace::Mtp &p = ws->mtp;
    const uint32_t rows = uint32_t(lm_chunk_rows(ws->max_tokens));
    MTP_TRY(step("fc plan", p.fc.init(gemm, kHidden, kHidden, kHc * ws->max_tokens, GemmOut::kBF16)));
    MTP_TRY(step("lm_head plan", p.lm_head.init(gemm, w->mtp.count, kHidden, rows, GemmOut::kBF16)));
    MTP_TRY(step("hc rows plan", p.hc_rows.init_rows(gemm, ws->hc, rows)));
    MTP_TRY(step("fc rows plan", p.fc_rows.init_rows(gemm, p.fc, rows)));
    MTP_TRY(step("attn_qkv rows plan", p.attn_qkv_rows.init_rows(gemm, ws->attn_qkv, rows)));
    MTP_TRY(step("attn_o rows plan", p.attn_o_rows.init_rows(gemm, ws->attn_o, rows)));
    MTP_TRY(step("router rows plan", p.router_rows.init_rows(gemm, ws->moe_router, rows)));
    MTP_TRY(step("shared gate_up rows plan", p.shared_gate_up_rows.init_rows(gemm, ws->moe_shared_gate_up, rows)));
    MTP_TRY(step("shared down rows plan", p.shared_down_rows.init_rows(gemm, ws->moe_shared_down, rows)));
    MTP_TRY(step("lm_head rows plan", p.lm_head_rows.init_rows(gemm, p.lm_head, rows)));
    p.rows = rows;
    return step("synchronize", cuda_status(cudaStreamSynchronize(stream)));
}

cere_status forward(const ModelWeights &w, Workspace &ws, SessionState &s, const MtpCall &call, cudaStream_t stream) {
    return forward_impl(w, ws, s, call, nullptr, stream);
}

cere_status forward_logits(const ModelWeights &w, Workspace &ws, SessionState &s, const MtpCall &call, bf16 *logits,
                           cudaStream_t stream) {
    return forward_impl(w, ws, s, call, logits, stream);
}

cere_status draft_rows(const ModelWeights &w, Workspace &ws, SessionState *const *sessions, const uint64_t *p0s,
                       const uint32_t *ns, uint32_t count, const uint32_t *next_ids, bf16 *hc, bf16 *last,
                       uint32_t *drafts, cudaStream_t stream) {
    constexpr uint32_t kSteps = uint32_t(layout::kDraftSteps);
    if (!w.has_draft || w.mtp.lm_head == nullptr || w.mtp.map == nullptr || w.mtp.count == 0 || ws.mtp.rows == 0 ||
        sessions == nullptr || p0s == nullptr || ns == nullptr || count < 2 || count > kRows ||
        count > ws.mtp.rows || next_ids == nullptr || hc == nullptr || last == nullptr || drafts == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    Segment segs[kRows];
    uint32_t m = 0;
    for (uint32_t p = 0; p < count; ++p) {
        SessionState *s = sessions[p];
        if (s == nullptr || !has_entry(*s) || ns[p] == 0) return CERE_ERR_ARGUMENT;
        const uint64_t end = p0s[p] + ns[p] + kSteps;  // the extend's rows, then one row per decode step
        if (end > s->max_context_tokens || end > w.max_context_tokens) return CERE_ERR_ARGUMENT;
        for (uint32_t q = 0; q < p; ++q) {
            if (sessions[q] == s) return CERE_ERR_ARGUMENT;
        }
        segs[p] = Segment{s, p0s[p], ns[p], nullptr};
        m += ns[p];
    }
    if (m > ws.mtp.rows) return CERE_ERR_ARGUMENT;

    // The extends: one forward of every session's rows, no draft (the rows plans: row r as in that session's
    // extend alone), each extend's last hc row to row p of `last` (run's tails), scored there: the extend's last_only
    // draft (d1) into drafts[count + p].
    MtpCall call;
    call.m = m;
    call.next_ids = next_ids;
    call.hc_in = hc;
    call.hc_out = hc;
    MTP_TRY(run(w, ws, segs, count, call, nullptr, false, last, stream));
    MTP_TRY(score_rows(w, ws, ws.mtp.hc_rows, last, count, drafts + count, nullptr, stream));
    // The decode steps: every session's next row in one forward each, step k fed drafts row k and its own hc row in
    // `last`, drafting into row k + 1.
    for (uint32_t k = 1; k <= kSteps; ++k) {
        for (uint32_t p = 0; p < count; ++p) segs[p] = Segment{sessions[p], p0s[p] + ns[p] + k - 1, 1, nullptr};
        call.m = count;
        call.next_ids = drafts + uint64_t(k) * count;
        call.hc_in = last;
        call.hc_out = last;
        call.draft_ids = drafts + uint64_t(k + 1) * count;
        MTP_TRY(run(w, ws, segs, count, call, nullptr, true, nullptr, stream));
    }
    return CERE_OK;
}

}  // namespace cere::model::mtp
