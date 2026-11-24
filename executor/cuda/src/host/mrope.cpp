#include "host/mrope.h"

#include <algorithm>

namespace cere::mrope {

uint64_t image_tokens(const ImageSpan &span) {
    const uint64_t h = span.grid_h / kSpatialMergeSize, w = span.grid_w / kSpatialMergeSize;
    const uint64_t hw = h * w;  // < 2^62: both factors < 2^31
    if (hw != 0 && span.grid_t > UINT64_MAX / hw) return 0;
    return span.grid_t * hw;
}

bool positions(uint64_t input_len, const std::vector<ImageSpan> &spans, int64_t *positions3, int64_t *rope_delta) {
    if (input_len == 0 || input_len > uint64_t(INT64_MAX)) return false;
    uint64_t st = 0;  // first row after the previous image, as SGLang's `st`
    for (const ImageSpan &span : spans) {
        const uint64_t tokens = image_tokens(span);
        if (span.grid_t == 0 || span.grid_h % kSpatialMergeSize != 0 || span.grid_w % kSpatialMergeSize != 0 ||
            tokens == 0 || span.token_offset < st || span.token_offset >= input_len ||
            tokens > input_len - span.token_offset) {
            return false;
        }
        st = span.token_offset + tokens;
    }

    int64_t *const t_row = positions3, *const h_row = positions3 + input_len, *const w_row = positions3 + 2 * input_len;
    uint64_t row = 0;
    int64_t next = 0;  // SGLang's `next_pos`
    auto text = [&](uint64_t end) {
        for (; row < end; ++row, ++next) t_row[row] = h_row[row] = w_row[row] = next;
    };
    for (const ImageSpan &span : spans) {
        text(span.token_offset);
        const uint64_t gh = span.grid_h / kSpatialMergeSize, gw = span.grid_w / kSpatialMergeSize;
        for (uint64_t t = 0; t < span.grid_t; ++t) {
            for (uint64_t h = 0; h < gh; ++h) {
                for (uint64_t w = 0; w < gw; ++w, ++row) {
                    t_row[row] = next + int64_t(t);
                    h_row[row] = next + int64_t(h);
                    w_row[row] = next + int64_t(w);
                }
            }
        }
        next += int64_t(std::max<uint64_t>({span.grid_t, gh, gw}));
    }
    text(input_len);
    // The largest position is the last row's text position or, when the prompt ends in pads, next - 1: either way
    // next - 1, since an image's rows stay below its start + max(t, h, w).
    *rope_delta = next - int64_t(input_len);
    return true;
}

}  // namespace cere::mrope
