// Inkling 2 Mini S6g (inkling design note S6g): host/mrope.h compiled for family inkling2_loop (the
// executor/models/inkling-2-mini.json header, Makefile INKLING_HOST_TESTS). cere_executor_submit's image path
// (api.cpp: cere::mrope::positions over the item's spans) and the device's placeholder rows (inkling_device.cpp
// held_rows) count one `<|unused_200054|>` per 2 x 2 patches, the tower's merge (kernels/inkling/vision.h kMerge).
// Golden p08's image is a 12 x 16 grid at token 14 of a 76-token prompt: 48 placeholders. With a merge size of 1
// (db9e01f8b) it counted 192, positions refused the span and the submit returned ARGUMENT.
#include "../check.h"

#include "cere_model_constants.h"
#include "host/mrope.h"

#include <cstdint>
#include <vector>

static_assert(cere::gen::kFamily == CERE_FAMILY_INKLING2, "built against the inkling-2-mini descriptor's header");

namespace {

using cere::mrope::ImageSpan;

bool span_ok(uint64_t input_len, const ImageSpan &span) {
    std::vector<int64_t> positions3(3 * input_len);
    int64_t rope_delta = 0;
    return cere::mrope::positions(input_len, {span}, positions3.data(), &rope_delta);
}

void test_p08() {
    CHECK_EQ(cere::mrope::kSpatialMergeSize, 2);
    const ImageSpan p08{14, 1, 12, 16};
    CHECK_EQ(cere::mrope::image_tokens(p08), 48);
    CHECK(span_ok(76, p08));
    CHECK(span_ok(62, p08));   // the prompt ends at the last placeholder
    CHECK(!span_ok(61, p08));  // one placeholder past the prompt
}

void test_odd_grid_refused() {
    CHECK(!span_ok(76, ImageSpan{14, 1, 12, 15}));
    CHECK(!span_ok(76, ImageSpan{14, 1, 11, 16}));
}

}  // namespace

int main() {
    test_p08();
    test_odd_grid_refused();
    return cere::test::finish("test_mrope_inkling");
}
