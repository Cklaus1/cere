// Track LoRA (lora design note): per-request PEFT LoRA adapters on the target's dense projections.
//
//   y[r, col + c] += scale(s) · Σ_j B_s[c, j] · (Σ_i x[r, i] · A_s[j, i])     for row r's adapter slot s (none: skip)
//
// GemmPlan::run applies it after its base GEMM when the plan has a Table, the Table is active and the call's weight
// is one of its entries (the target's q|k|v, o_proj and shared-expert gate / up / down weights; the MTP layer's and
// every other weight miss it). The device pool holds `adapters` slots, each a copy of every target's A [R, k] and B
// [n, R] (BF16, R = the reserved rank padded to host::lora::kRankAlign; an adapter of rank r < R is zero past r); the
// host keeps each (slot, target)'s padded rank (0: the adapter leaves the target alone) and each slot's scale, and
// passes the ones a call needs as kernel arguments (a call whose rows need none launches nothing).
//
// Two launches per hit: shrink (t[r, seg·R + j] = Σ_i x[r, i] · A[j, i] as a double-float (hi, lo): BF16 products are
// exact in FP32 and summed compensated (TwoSum, Sum2), one warp per (row block, j), lanes over k in a fixed order,
// xor-shuffle reduction) and expand (y = bf16(float(y) + scale · Σ_j t · B), the sum a double-float with j ascending
// (TwoProd), the result rounded once; a zero product leaves y untouched) per output column. The sums carry about twice
// FP32's precision (Sum2's error is O(n · 2^-48) of the term magnitudes), so the result stays within 1 bf16 ulp of an
// FP64 reference even where it cancels against the base (an FP32 sum misses there by many ulps: lora design note
// F8). Every row's bits depend on its x row, its slot's weights and k only: batch invariant (any m, any
// mix of slots and of rows without an adapter) by construction.
//
// Active state (set by the model's forward entry points, cleared on return): a uniform slot (one session: forward), or
// per-row slots (forward_decode_rows / forward_verify_rows: row r of each call). Host-side state, like the rest of a
// model's forward: one caller at a time.
#pragma once

#include "cere_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace cere::kernels::lora {

constexpr uint32_t kMaxSegments = 3;  // q | k | v of the packed qkv weight
constexpr uint32_t kMaxRows = 16;     // per-row slots (GemmPlan::kMaxRows)

// One target's columns of an entry's output: pool target index `target` (host::lora::targets order) at [col, col + n).
struct Segment {
    uint32_t target = 0;
    uint32_t col = 0;
    uint32_t n = 0;
};

// A registered weight: its plan's k and the targets of its output columns.
struct Entry {
    const void *w = nullptr;
    uint32_t k = 0;
    uint32_t segments = 0;
    Segment seg[kMaxSegments]{};
};

// Device layout of one target inside a slot: A at a_offset ([R, k]), B at b_offset ([n, R]), in elements.
struct TargetLayout {
    uint64_t a_offset = 0;
    uint64_t b_offset = 0;
    uint32_t k = 0;
    uint32_t n = 0;
};

class Table {
public:
    Table() = default;
    ~Table();
    Table(const Table &) = delete;
    Table &operator=(const Table &) = delete;

    // Device bytes of a table of `adapters` slots at padded rank `rank` over targets of `elements_per_rank` (A + B
    // elements per unit of rank, host::lora::elements_per_rank) and a scratch of `max_tokens` rows.
    static uint64_t bytes(uint32_t adapters, uint32_t rank, uint64_t elements_per_rank, uint32_t max_tokens);
    // Allocates the pool (zeroed: every slot empty) and the table's own copy stream for `targets` (layouts by target
    // index) and `entries` (any order; sorted here). CERE_ERR_CAPACITY when cudaMalloc fails, CERE_ERR_ARGUMENT for a
    // rank not a multiple of 8, above 256, or an entry segment out of range.
    cere_status init(uint32_t adapters, uint32_t rank, uint32_t max_tokens, std::vector<TargetLayout> targets,
                     std::vector<Entry> entries);
    // Copies an adapter into `slot` (host BF16 A [rank, k] and B [n, rank] per listed target index; the slot's other
    // targets zero) on the table's copy stream, synchronizes it, then records its padded ranks and scale: a step on the
    // executor stream (even one capturing a graph meanwhile) never waits on or orders with it, and the next forward
    // that reads the slot starts after the call returned. On error the slot stays empty.
    cere_status load(uint32_t slot, uint32_t rank, float scale, uint64_t count, const uint32_t *target_index,
                     const void *const *a, const void *const *b);
    // Clears `slot`'s ranks: every target then adds nothing (the device bytes stay until the next load).
    void unload(uint32_t slot);

    const Entry *find(const void *w) const;
    bool active() const { return active_; }
    // Forward entry points: one slot for every row, or row r's slot (-1: none) for r < rows. activate_rows with no
    // adapter row leaves the table inactive.
    void activate(int32_t slot);
    void activate_rows(const int32_t *slots, uint32_t rows);
    void deactivate();
    // y[m, n_total] (BF16, row stride = the plan's n) += the entry's adapters on x[m, e.k].
    cere_status apply(const Entry &e, const __nv_bfloat16 *x, uint32_t m, __nv_bfloat16 *y, uint32_t ld_y,
                      cudaStream_t stream) const;

private:
    uint32_t adapters_ = 0;
    uint32_t rank_ = 0;  // R
    uint32_t max_tokens_ = 0;
    uint64_t slot_elems_ = 0;
    std::vector<TargetLayout> targets_;
    std::vector<Entry> entries_;  // sorted by w
    cudaStream_t copy_ = nullptr;  // init / load (non-blocking)
    void *memory_ = nullptr;      // one cudaMalloc: pool, then scratch
    __nv_bfloat16 *pool_ = nullptr;
    float2 *scratch_ = nullptr;   // [max_tokens, kMaxSegments · R] double-float (hi, lo)
    std::vector<uint16_t> ranks_;  // host [adapters, targets]: padded rank, 0 = none
    std::vector<float> scales_;    // host [adapters]
    bool active_ = false;
    bool per_row_ = false;
    int32_t uniform_ = -1;
    int8_t slots_[kMaxRows]{};
};

// The scope of one forward entry point: activates on construction (when `table` is non-null), deactivates on exit.
class Scope {
public:
    Scope(Table *table, int32_t slot) : table_(table) {
        if (table_ != nullptr) table_->activate(slot);
    }
    Scope(Table *table, const int32_t *slots, uint32_t rows) : table_(table) {
        if (table_ != nullptr) table_->activate_rows(slots, rows);
    }
    ~Scope() {
        if (table_ != nullptr) table_->deactivate();
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

private:
    Table *table_;
};

}  // namespace cere::kernels::lora
