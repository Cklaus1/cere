// MixedPD (mixed-pd design note): the layers of a sliced prefill call's next segment. CUDA-free so the host tests
// check it (executor/tests/host/test_segment.cpp, batch-concurrency design note "BcSegment (O283)"); the CUDA
// device's step (device/cuda_device.cpp) is its one caller.
#pragma once

#include <algorithm>
#include <cstdint>

#include "cere_model_constants.h"

namespace cere::model::segment {

inline constexpr uint32_t kLayers = gen::kNumHiddenLayers;  // model::kLayers (static_assert in cuda_device.cpp)
inline constexpr uint32_t kDefaultSegmentLayers = 4;  // before the model's first measurement
inline constexpr uint32_t kTailLayers = 2;            // the call's tail: final mixer, scoring, MTP extend

// The cost per layer of a call of `tokens` prompt tokens, from a measurement of ms_per_layer on a call of
// `measured_tokens` (0: no measurement yet). A larger call is assumed to cost proportionally more, an overestimate
// (a layer's fixed cost does not grow with the tokens); a smaller one costs the same, also an overestimate. So a short
// prompt's measurement never lets a 4,096-token chunk of a long prompt run whole.
inline double fit_ms_per_layer(double ms_per_layer, uint64_t tokens, uint64_t measured_tokens) {
    if (measured_tokens == 0 || tokens <= measured_tokens) return ms_per_layer;
    return ms_per_layer * (double(tokens) / double(measured_tokens));
}

// The layers of the next segment from layer `next`. BcSegment (O283): a call whose remaining layers plus the tail fit
// the budget at the estimated cost ((remaining + kTailLayers) x fit_ms <= budget_ms, fit_ms from fit_ms_per_layer)
// runs whole. Before, every call from layer 0 split in two however short the prompt (the split below caps a segment at
// kLayers), and the parked first half let a decode round run inside each prompt's call: a burst of N short prompts
// paid N extra decode rounds before its last first token. A call that does not fit splits exactly as before: balanced
// segments of at most budget_ms / ms_per_layer layers each (kDefaultSegmentLayers before the model's first
// measurement, ms_per_layer == 0), the call's tail counted as kTailLayers more.
inline uint32_t segment_layers(uint32_t next, double ms_per_layer, double fit_ms, uint32_t budget_ms) {
    const uint32_t remaining = kLayers - next;
    if (fit_ms > 0.0 && double(remaining + kTailLayers) * fit_ms <= double(budget_ms)) return remaining;
    uint32_t most = kDefaultSegmentLayers;
    if (ms_per_layer > 0.0) {
        most = uint32_t(std::clamp(double(budget_ms) / ms_per_layer, 1.0, double(kLayers)));
    }
    const uint32_t segments = (remaining + kTailLayers + most - 1) / most;
    return segments <= 1 ? remaining : (remaining + segments - 1) / segments;
}

}  // namespace cere::model::segment
