#include "research/capture.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace cere::research {

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

bool Binding::rank(uint32_t op, uint64_t position, uint64_t *out) const {
    const std::vector<Interval> &sel = selection[op];
    // The last interval whose first is <= position.
    auto it = std::upper_bound(sel.begin(), sel.end(), position,
                               [](uint64_t p, const Interval &r) { return p < r.first; });
    if (it == sel.begin()) return false;
    --it;
    if (position >= it->end) return false;
    *out = base[op][size_t(it - sel.begin())] + (position - it->first);
    return true;
}

bool Binding::selects(uint64_t position) const {
    uint64_t r;
    for (uint32_t op = 0; op < selection.size(); ++op) {
        if (rank(op, position, &r)) return true;
    }
    return false;
}

cere_status bind(std::shared_ptr<const Plan> plan, uint32_t slot, uint64_t prompt_tokens, uint64_t max_tokens,
                 Binding *out) {
    if (plan == nullptr || prompt_tokens == 0) return CERE_ERR_ARGUMENT;
    Binding b;
    b.slot = slot;
    b.prompt_tokens = prompt_tokens;
    b.max_tokens = max_tokens;
    const size_t ops = plan->ops.size();
    b.selection.resize(ops);
    b.base.resize(ops);
    b.rows.resize(ops);
    b.first_block.assign(ops, 0);
    for (uint32_t i = 0; i < ops; ++i) {
        const Op &op = plan->ops[i];
        b.selection[i] = select(op, prompt_tokens, max_tokens);
        uint64_t rank = 0;
        for (const Interval &r : b.selection[i]) {
            b.base[i].push_back(rank);
            rank += r.end - r.first;
        }
        b.rows[i] = rank;
        if (op.kind == CERE_RESEARCH_REPLACE && op.vector_rows > 1 && op.vector_rows < rank) return CERE_ERR_ARGUMENT;
    }
    // A bound past u64 (bound() saturates) fits no ring; below it no block offset overflows.
    if (bound(*plan, prompt_tokens, max_tokens) == UINT64_MAX) return CERE_ERR_CAPACITY;
    uint64_t offset = 0;
    for (uint32_t i = 0; i < ops; ++i) {
        const Op &op = plan->ops[i];
        b.first_block[i] = uint32_t(b.blocks.size());
        if (!op.captures()) continue;
        for (const uint32_t layer : layers_of(op.layers)) {
            for (uint32_t part = 0; part < op.point->parts; ++part) {
                Block block;
                block.op = i;
                block.layer = layer;
                block.part = part;
                block.dtype = op.point->part[part].dtype;
                block.cols = part_cols(*op.point, part, op.top_k);
                block.row_bytes = part_bytes(*op.point, part, op.top_k);
                block.offset = offset;
                block.name = std::string(op.point->name) + op.point->part[part].suffix + "/L" + std::to_string(layer);
                offset += b.rows[i] * block.row_bytes;
                b.blocks.push_back(std::move(block));
            }
        }
    }
    b.bytes = offset;  // bound(*plan, prompt_tokens, max_tokens)
    b.plan = std::move(plan);
    *out = std::move(b);
    return CERE_OK;
}

void write_toy_row(const Binding &binding, uint64_t position) {
    if (binding.data == nullptr) return;
    uint64_t rank = 0;
    uint32_t op = UINT32_MAX;
    bool selected = false;
    for (const Block &block : binding.blocks) {
        if (block.op != op) {
            op = block.op;
            selected = binding.rank(op, position, &rank);
        }
        if (!selected) continue;
        uint8_t *row = binding.row(block, rank);
        for (uint32_t j = 0; j < block.cols; ++j) {
            switch (block.dtype) {
                case CERE_RESEARCH_DTYPE_BF16: {
                    const uint16_t v = bf16_bits(float(position % 251) + float(block.layer) / 64.0f);
                    std::memcpy(row + uint64_t(j) * 2, &v, 2);
                    break;
                }
                case CERE_RESEARCH_DTYPE_I32: {
                    const int32_t v = int32_t((position + block.layer + j) % 512);
                    std::memcpy(row + uint64_t(j) * 4, &v, 4);
                    break;
                }
                default: {
                    const float v = 1.0f / float(j + 1);
                    std::memcpy(row + uint64_t(j) * 4, &v, 4);
                    break;
                }
            }
        }
    }
}

void finalise(Binding &binding, uint64_t fed) {
    binding.live = false;
    binding.watermark = fed;
}

void read(Binding &binding, cere_research_capture *out) {
    if (binding.tensors.size() != binding.blocks.size()) {
        binding.positions.assign(binding.selection.size(), {});
        for (uint32_t op = 0; op < binding.selection.size(); ++op) {
            if (!binding.plan->ops[op].captures()) continue;
            std::vector<uint64_t> &pos = binding.positions[op];
            for (const Interval &r : binding.selection[op]) {
                for (uint64_t p = r.first; p < r.end && p < binding.watermark; ++p) pos.push_back(p);
            }
        }
        binding.tensors.clear();
        for (const Block &block : binding.blocks) {
            const std::vector<uint64_t> &pos = binding.positions[block.op];
            cere_research_tensor t{};
            t.name = block.name.c_str();
            t.op = block.op;
            t.layer = block.layer;
            t.dtype = block.dtype;
            t.cols = block.cols;
            t.rows = pos.size();
            t.positions = pos.empty() ? nullptr : pos.data();
            t.data = binding.data == nullptr ? nullptr : binding.data + block.offset;
            binding.tensors.push_back(t);
        }
    }
    *out = cere_research_capture{};
    out->tensors = binding.tensors.empty() ? nullptr : binding.tensors.data();
    uint64_t bytes = 0;
    for (size_t i = 0; i < binding.tensors.size(); ++i) bytes += binding.tensors[i].rows * binding.blocks[i].row_bytes;
    out->tensor_count = binding.tensors.size();
    out->bytes = bytes;
    out->prefix_lookup_skipped = binding.lookup_skipped ? 1u : 0u;
    std::memcpy(out->plan_sha256, binding.plan->sha256.data(), sizeof(out->plan_sha256));
}

}  // namespace cere::research
