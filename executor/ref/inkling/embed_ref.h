#pragma once
// Inkling 2 Mini (inkling design note 3.2, G1): the normed embedding, CPU reference (kernel:
// executor/cuda/src/kernels/inkling/norm.h embed_norm). REF Inkling2LoopNormedEmbedding (modeling_inkling2_loop.py
// :1240-1246, eps config rms_norm_eps :1281): gather the BF16 row, then the weighted RMSNorm of norm_ref.h with the
// input dtype BF16, so the output is rounded once to BF16:
//   x[i] = float(table[id * kHidden + i]);  y = encode_bf16(rms_norm<256> weighted(x, embed_norm.weight, eps)).
#include "norm_ref.h"

namespace ref {

// out[0..kHidden) = the normed embedding row of token `id` (the caller validates id < vocab).
inline void embed_norm_row(const uint16_t *table, uint32_t id, const uint16_t *w, float eps, uint16_t *out) {
    float x[kHidden];
    const uint16_t *row = table + uint64_t(id) * kHidden;
    for (int i = 0; i < kHidden; ++i) x[i] = decode_bf16(row[i]);
    rms_norm<kReduceHidden>(x, kHidden, eps, w, x);
    for (int i = 0; i < kHidden; ++i) out[i] = encode_bf16(x[i]);
}

}  // namespace ref
