// BcSegment (batch-concurrency design note "BcSegment (O283)"): model/segment.h's segment_layers and
// fit_ms_per_layer. A sliced prefill call whose remaining layers plus the tail fit the slice budget runs whole (one
// segment); one that does not fit splits exactly as before BcSegment (the reference below is that formula, kept here
// only as the test's oracle).
#include "check.h"

#include "model/segment.h"

#include <algorithm>
#include <cstdint>

namespace {

namespace seg = cere::model::segment;

// segment_layers before BcSegment: `most` capped at kLayers, so a call from layer 0 always split in two.
uint32_t before_bc_segment(uint32_t next, double ms_per_layer, uint32_t budget_ms) {
    const uint32_t remaining = seg::kLayers - next;
    uint32_t most = seg::kDefaultSegmentLayers;
    if (ms_per_layer > 0.0) {
        most = uint32_t(std::clamp(double(budget_ms) / ms_per_layer, 1.0, double(seg::kLayers)));
    }
    const uint32_t segments = (remaining + seg::kTailLayers + most - 1) / most;
    return segments <= 1 ? remaining : (remaining + segments - 1) / segments;
}

// The call measured at its own length (fit_ms == ms_per_layer).
uint32_t layers(uint32_t next, double ms_per_layer, uint32_t budget_ms) {
    return seg::segment_layers(next, ms_per_layer, ms_per_layer, budget_ms);
}

}  // namespace

int main() {
    // A short prompt's call from layer 0 at 1 ms per layer: 50 x 1 <= 200, whole (before: 24 + 24).
    CHECK_EQ(layers(0, 1.0, 200), 48);
    // Does not fit (50 x 4.1 = 205 > 200): two segments, unchanged.
    CHECK_EQ(layers(0, 4.1, 200), 24);
    // No measurement yet: 4-layer segments, unchanged.
    CHECK_EQ(layers(0, 0.0, 200), 4);
    // A resume runs its remainder, unchanged.
    CHECK_EQ(layers(24, 1.0, 200), 24);
    // The boundary, (remaining + tail) x ms_per_layer == budget, fits: whole, as a resume at the boundary already was.
    CHECK_EQ(layers(0, 4.0, 200), 48);
    CHECK_EQ(layers(8, 2.0, 84), 40);
    CHECK_EQ(before_bc_segment(8, 2.0, 84), 40);
    // Just past the boundary: splits.
    CHECK_EQ(layers(8, 2.0, 83), 20);

    // fit_ms_per_layer: a larger call than the measured one costs proportionally more, a smaller one the same, and
    // without a measurement the (zero) measurement stands.
    CHECK(seg::fit_ms_per_layer(1.0, 4096, 85) > 48.0);
    CHECK_EQ(seg::fit_ms_per_layer(8.0, 85, 4096), 8.0);
    CHECK_EQ(seg::fit_ms_per_layer(1.0, 85, 85), 1.0);
    CHECK_EQ(seg::fit_ms_per_layer(0.0, 85, 0), 0.0);
    CHECK_EQ(seg::fit_ms_per_layer(2.0, 4096, 0), 2.0);
    // A 4,096-token chunk after an 85-token prompt measured at 1 ms per layer does not run whole: it splits exactly as
    // before, by the measurement itself.
    CHECK_EQ(seg::segment_layers(0, 1.0, seg::fit_ms_per_layer(1.0, 4096, 85), 200), 24);
    // A slightly longer short prompt still fits (87 vs 78 tokens at 1 ms per layer: 50 x 1.12 <= 200).
    CHECK_EQ(seg::segment_layers(0, 1.0, seg::fit_ms_per_layer(1.0, 87, 78), 200), 48);

    // Every call: whole when it fits at the estimate, else exactly the old split by the measurement.
    for (uint32_t next = 0; next < seg::kLayers; ++next) {
        const uint32_t remaining = seg::kLayers - next;
        for (const uint32_t budget : {1u, 20u, 50u, 83u, 84u, 100u, 200u, 1000u}) {
            for (const double ms : {0.0, 0.25, 0.5, 1.0, 2.0, 3.9, 4.0, 4.1, 8.0, 40.0, 500.0}) {
                for (const double scale : {1.0, 1.5, 48.0}) {
                    const double fit = ms * scale;
                    const uint32_t got = seg::segment_layers(next, ms, fit, budget);
                    const bool fits = fit > 0.0 && double(remaining + seg::kTailLayers) * fit <= double(budget);
                    CHECK_EQ(got, fits ? remaining : before_bc_segment(next, ms, budget));
                    CHECK(got >= 1 && got <= remaining);
                }
            }
        }
    }
    return cere::test::finish("test_segment");
}
