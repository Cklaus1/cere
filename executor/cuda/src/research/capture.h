// WeightLab research bindings and captures (weightlab design note WL-02, WL-03): what one session runs under its
// plan (cere_session_set_research) and the capture it writes. Host-only C++ shared by the CUDA library and the toy.
//
// A capture's bytes are one region of the model's host ring (pinned on the CUDA device), `bytes` = the request bound
// (research/plan.h bound()), laid out tensor-major: for each capture op, layer (ascending, 48 last) and part, a block
// of rows x part bytes, row i the i-th position the op selects. The forward writes a selected row into its rank's slot
// (a recomputed or rejected-then-recomputed position overwrites it in stream order), so a capture never outgrows its
// bound. At the session's release the capture keeps the rows below the session's fed count (the rows whose state the
// session kept: a rejected draft's row is never below it); those are a prefix of every op's ranks, since the session
// computes every selected position below it (a capture plan that names prompt positions skips the prefix lookup, and
// generated positions start at prompt_tokens - 1, which no prefix hit covers).
#pragma once

#include "cere_executor.h"
#include "research/plan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cere::research {

// One [rows, cols] tensor of a capture: op, layer, part and its offset in the capture's bytes.
struct Block {
    uint32_t op = 0;
    uint32_t layer = 0;
    uint32_t part = 0;
    uint32_t dtype = 0;
    uint32_t cols = 0;
    uint64_t row_bytes = 0;
    uint64_t offset = 0;
    std::string name;  // "<point>/L<layer>" or "<point><suffix>/L<layer>" (".ids", ".weights", ".logprobs")
};

struct Binding {
    std::shared_ptr<const Plan> plan;
    uint32_t slot = 0;
    uint64_t prompt_tokens = 0;
    uint64_t max_tokens = 0;
    std::vector<std::vector<Interval>> selection;  // per op
    std::vector<std::vector<uint64_t>> base;       // per op and interval: the rank of the interval's first position
    std::vector<uint64_t> rows;                    // per op: |selection|
    // The capture (plan->captures; id 0 otherwise).
    uint64_t id = 0;
    uint8_t *data = nullptr;  // `bytes` of the host ring (null when bytes is 0)
    uint64_t ring_offset = 0;
    uint64_t bytes = 0;
    std::vector<Block> blocks;          // in op, layer, part order
    std::vector<uint32_t> first_block;  // per op: its first block (capture ops), else blocks.size()
    bool lookup_skipped = false;
    bool live = true;        // its session lives (cere_model_research_capture_read: CERE_ERR_BUSY)
    uint64_t watermark = 0;  // at release: the session's fed count
    // cere_model_research_capture_read's arrays (built once, at the first read after release).
    std::vector<cere_research_tensor> tensors;
    std::vector<std::vector<uint64_t>> positions;  // per op

    // The rank of `position` among op `op`'s selected positions; false when not selected.
    bool rank(uint32_t op, uint64_t position, uint64_t *out) const;
    // Whether some op selects `position`.
    bool selects(uint64_t position) const;
    // The slot of row `rank` of `block` in the capture.
    uint8_t *row(const Block &block, uint64_t rank) const { return data + block.offset + rank * block.row_bytes; }
};

// The binding of a session to `plan` for a request of `prompt_tokens` (>= 1) and `max_tokens`: the selections, and
// for a capturing plan its blocks (bytes = bound(plan, prompt_tokens, max_tokens); the caller places `data`).
// CERE_ERR_ARGUMENT for prompt_tokens 0, or a replace op whose vector has more than one row but fewer rows than the
// positions it selects (its row i edits the op's i-th position).
cere_status bind(std::shared_ptr<const Plan> plan, uint32_t slot, uint64_t prompt_tokens, uint64_t max_tokens,
                 Binding *out);

// The toy device's capture rows of `position` (weightlab design note WL-A contract "toy device"): for each capture op
// that selects it, each layer L (48 for the final points) and column j, BF16 parts bf16(float(position % 251) +
// float(L) / 64), I32 parts (position + L + j) % 512, F32 parts 1 / (j + 1).
void write_toy_row(const Binding &binding, uint64_t position);

// The session released with `fed` rows consumed: the capture keeps its rows below `fed`.
void finalise(Binding &binding, uint64_t fed);

// Fills *out from a finalised capture (its arrays stay valid while the binding lives).
void read(Binding &binding, cere_research_capture *out);

// BF16 bits of `f`, round to nearest even (NaN kept quiet).
uint16_t bf16_bits(float f);

}  // namespace cere::research
