// Prefill selection signature (BC1 leftover, batch-concurrency design note "Prefill selection signature";
// cere_model_prefill_signature, format v1 in cere_executor.h): the CUDA device's lines of the canonical text. A sealed
// model's prefill kernels and their bits are picked by the cuBLASLt version, the static GEMM pins (k2_gemm.h
// GemmPins), the bucket algorithm cuBLASLt's heuristic gave each prefill GEMM plan at INSTANTIATE, the P3b tactic
// table (p3b/p3b_tactic.h, on a P3b build), the K5 chunk length and the K6 dense threshold; PrefillSelection holds
// them as plain values and selection_text prints them, one fact per line in a fixed order. This header is CUDA-free
// (host-testable: executor/tests/host/test_prefill_signature.cpp); collect, in prefill_signature.cpp, reads the
// values from a model's Workspace on the CUDA build.
#pragma once

#include "cere_executor.h"
#include "p3b/p3b_tactic.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cere::model {

struct Workspace;

namespace prefill_sig {

// One static cuBLASLt configuration (kernels::GemmConfig as plain values).
struct PinConfig {
    int32_t algo_id = 0;
    uint32_t tile = 0;       // cublasLtMatmulTile_t
    uint32_t stages = 0;     // cublasLtMatmulStages_t
    int32_t splitk = 0;
    uint32_t reduction = 0;  // cublasLtReductionScheme_t
    uint32_t partials = 0;
    uint32_t custom = 0;     // CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
};

// kernels::GemmPinRange.
struct PinRange {
    uint32_t m_lo = 0;
    uint32_t m_hi = 0;
    PinConfig config;
};

// One static pins table (kernels::GemmPins) of shape (n, k, out).
struct PinTable {
    uint32_t n = 0;
    uint32_t k = 0;
    std::string out;  // "bf16" or "f32"
    uint64_t cublaslt_version = 0;
    std::vector<PinRange> ranges;
};

// The algorithm bucket b of a plan pinned (GemmPlan::bucket_algos): the cuBLASLt algorithm configuration
// attributes (cublasLtMatmulAlgoConfigGetAttribute) of the heuristic's pick for m = m_hi = 2^b.
struct BucketAlgo {
    uint32_t m_hi = 0;
    int32_t algo_id = 0;        // CUBLASLT_ALGO_CONFIG_ID
    uint32_t tile = 0;          // CUBLASLT_ALGO_CONFIG_TILE_ID
    uint32_t stages = 0;        // CUBLASLT_ALGO_CONFIG_STAGES_ID
    int32_t splitk = 0;         // CUBLASLT_ALGO_CONFIG_SPLITK_NUM
    uint32_t reduction = 0;     // CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME
    uint32_t custom = 0;        // CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
    uint32_t swizzle = 0;       // CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
    uint32_t inner = 0;         // CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID
    uint32_t cluster = 0;       // CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID
};

// One prefill GEMM plan (built at the workspace rows by model::instantiate_model).
struct PlanSelection {
    std::string name;
    uint32_t n = 0;
    uint32_t k = 0;
    std::string out;  // "bf16" or "f32"
    uint32_t max_m = 0;
    std::vector<BucketAlgo> buckets;
};

struct PrefillSelection {
    uint64_t cublaslt_version = 0;   // cublasLtGetVersion() of the running library
    std::vector<PinTable> pins;      // every static pins table, in gemm_pins_for's order
    std::vector<PlanSelection> plans;  // the prefill plans, in a fixed order; a plan not built is left out
    bool p3b = false;                // the MoE backend is P3b (a CERE_MOE_P3B build)
    std::vector<p3b::PinnedBucket> p3b_buckets;  // p3b::kPinnedBuckets when p3b
    uint32_t k5_chunk_tokens = 0;    // k5::kChunkTokens
    uint32_t k6_max_context = 0;     // k6::kMaxContext
};

inline std::string tactic_text(const p3b::Tactic &t) {
    return std::to_string(t.tile_m) + " " + std::to_string(t.tile_n) + " " + std::to_string(t.tile_k_bytes) + " " +
           (t.swap_ab ? "1" : "0") + " " + (t.finalize ? "1" : "0");
}

// The device lines of format v1 ('\n'-terminated, this order):
//   cublaslt <version>
//   per pins table: pins <n> <k> <out> cublaslt <version> ranges <count>
//     per range:    pin <m_lo> <m_hi> algo <id> tile <t> stages <s> splitk <k> reduction <r> partials <p> custom <c>
//   per plan:       plan <name> <n> <k> <out> max_m <M> buckets <B>
//     per bucket:   bucket <m_hi> algo <id> tile <t> stages <s> splitk <k> reduction <r> custom <c> swizzle <w>
//                   inner <i> cluster <u>
//   p3b <count>, then per bucket: p3b_bucket <max_m> gemm1 <tile_m tile_n tile_k_bytes swap_ab finalize> gemm2 <...>
//     (or the one line "p3b none" when the backend is not P3b)
//   k5 chunk_tokens <n>
//   k6 max_context <n>
inline std::string selection_text(const PrefillSelection &s) {
    std::string t = "cublaslt " + std::to_string(s.cublaslt_version) + "\n";
    for (const PinTable &p : s.pins) {
        t += "pins " + std::to_string(p.n) + " " + std::to_string(p.k) + " " + p.out + " cublaslt " +
             std::to_string(p.cublaslt_version) + " ranges " + std::to_string(p.ranges.size()) + "\n";
        for (const PinRange &r : p.ranges) {
            t += "pin " + std::to_string(r.m_lo) + " " + std::to_string(r.m_hi) + " algo " +
                 std::to_string(r.config.algo_id) + " tile " + std::to_string(r.config.tile) + " stages " +
                 std::to_string(r.config.stages) + " splitk " + std::to_string(r.config.splitk) + " reduction " +
                 std::to_string(r.config.reduction) + " partials " + std::to_string(r.config.partials) + " custom " +
                 std::to_string(r.config.custom) + "\n";
        }
    }
    for (const PlanSelection &p : s.plans) {
        t += "plan " + p.name + " " + std::to_string(p.n) + " " + std::to_string(p.k) + " " + p.out + " max_m " +
             std::to_string(p.max_m) + " buckets " + std::to_string(p.buckets.size()) + "\n";
        for (const BucketAlgo &b : p.buckets) {
            t += "bucket " + std::to_string(b.m_hi) + " algo " + std::to_string(b.algo_id) + " tile " +
                 std::to_string(b.tile) + " stages " + std::to_string(b.stages) + " splitk " +
                 std::to_string(b.splitk) + " reduction " + std::to_string(b.reduction) + " custom " +
                 std::to_string(b.custom) + " swizzle " + std::to_string(b.swizzle) + " inner " +
                 std::to_string(b.inner) + " cluster " + std::to_string(b.cluster) + "\n";
        }
    }
    if (s.p3b) {
        t += "p3b " + std::to_string(s.p3b_buckets.size()) + "\n";
        for (const p3b::PinnedBucket &b : s.p3b_buckets) {
            t += "p3b_bucket " + std::to_string(b.max_m) + " gemm1 " + tactic_text(b.gemm1) + " gemm2 " +
                 tactic_text(b.gemm2) + "\n";
        }
    } else {
        t += "p3b none\n";
    }
    t += "k5 chunk_tokens " + std::to_string(s.k5_chunk_tokens) + "\n";
    t += "k6 max_context " + std::to_string(s.k6_max_context) + "\n";
    return t;
}

// The CUDA device's selection of a model instantiated on `ws` (prefill_signature.cpp; CUDA build only).
// CERE_ERR_INTERNAL when cuBLASLt refuses an algorithm attribute read.
cere_status collect(const Workspace &ws, PrefillSelection *out);

}  // namespace prefill_sig
}  // namespace cere::model
