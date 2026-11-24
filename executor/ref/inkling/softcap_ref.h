#pragma once
// Inkling 2 Mini (inkling design note 3.1, G1): the final logit softcap, CPU reference (kernel:
// executor/cuda/src/kernels/inkling/softcap.h). REF Inkling2LoopSoftcap (modeling_inkling2_loop.py:751-757) applied
// at :1535 with CFG.final_logit_softcapping 30 to `lm_head(h).float()`: the unembed's BF16 logits widened to FP32.
// Exact order per logit x (FP32; a BF16 logit is widened exactly first):
//   y = mul(tanh(div(x, 30)), 30)          tanh = common_ref.h ref::tanh
// so |y| <= 30, y(+-0) = +0 (ref::tanh(-0) = +0), y(+-inf) = +-30, NaN stays NaN.
#include "common_ref.h"

namespace ref {

constexpr float kLogitSoftcap = 30.0f;

CERE_HD inline float softcap(float x) { return mul(ref::tanh(div(x, kLogitSoftcap)), kLogitSoftcap); }

}  // namespace ref
