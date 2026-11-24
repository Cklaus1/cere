// mRoPE positions of an image prompt (m213 design note D11, C8): a port of SGLang's image-only path
// `QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets` (`multimodal/processors/qwen_vl.py`), which
// equals `MRotaryEmbedding.get_rope_index` on these prompts (tests/fixtures/native/mrope_cases.json). Pure host code.
#pragma once

#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "cere_model_constants.h"

namespace cere::mrope {

#if CERE_GEN_FAMILY == CERE_FAMILY_QWEN4_EXP
constexpr uint32_t kSpatialMergeSize = gen::kVisionSpatialMergeSize;  // vision_config.spatial_merge_size
#elif CERE_GEN_FAMILY == CERE_FAMILY_INKLING2
// Inkling 2 Mini (S6g): the tower's adapter_merge_size, one placeholder per 2 x 2 patches (kernels/inkling/vision.h
// kMerge; device/inkling/inkling_device.cpp asserts they are equal). Its forward has no mRoPE, so api.cpp's
// positions are unused there, but the span checks and placeholder counts need the merged grid.
constexpr uint32_t kSpatialMergeSize = 2;
#elif CERE_GEN_FAMILY == CERE_FAMILY_SDAR
// No vision tower in the sdar family (SelfTune ST4: a vision_config is refused by name, so no image span gets here);
// 1 keeps the text-only path, which never divides by it.
constexpr uint32_t kSpatialMergeSize = 1;
#else
#error "host/mrope.h: no spatial merge size for this family"
#endif

// One image of the prompt: `token_offset` is the row of its first `<|image_pad|>`; the vision grid (t, h, w) is in
// patches, so the image has t * (h / 2) * (w / 2) pad rows.
struct ImageSpan {
    uint64_t token_offset;
    uint32_t grid_t;
    uint32_t grid_h;
    uint32_t grid_w;
};

// Pad rows of `span` (t * (h / 2) * (w / 2)); 0 when that overflows 64 bits.
uint64_t image_tokens(const ImageSpan &span);

// Fills `positions3` ([3][input_len], row-major: t, h, w) and `*rope_delta` = max position + 1 - input_len (<= 0).
// Text rows advance all three axes together; an image's pad rows get (start + t_i, start + h_i, start + w_i) over its
// merged grid, and the next row starts at start + max(t, h / 2, w / 2). No images: positions are 0..input_len-1 on
// every axis, delta 0. `spans` must be in prompt order, non-overlapping, inside the prompt, with t >= 1 and h, w
// non-zero multiples of 2 (the only grids the processor produces); otherwise returns false and writes nothing.
bool positions(uint64_t input_len, const std::vector<ImageSpan> &spans, int64_t *positions3, int64_t *rope_delta);

}  // namespace cere::mrope
