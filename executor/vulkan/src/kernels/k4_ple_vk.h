// VulkanAmd P2: K4, the PLE n-gram path (hash, table gather FP8 / NVFP4, embed, layer forward) on Vulkan
// (shaders/k4_*.comp). Same contracts as executor/cuda/src/kernels/k4_ple.h and k4_nvfp4.h (the HIP build is the
// reference): the n-gram hash and every gather are bit-exact; the forward is FP64-bounded with any split of a step
// (and every row of a rows call) bit-equal to one call. Buffer device addresses instead of pointers; the functions
// record into the context's open command buffer (Context::begin .. submit_wait) with the barriers they need between
// their own dispatches; the caller orders separate calls with Context::barrier. CERE_ERR_ARGUMENT outside the
// contract, before anything is recorded.
//
// The split table (spec G0 consequences 1 and 2). The PLE table (28.8 GB in S3: 320,001,446 NVFP4 rows (the 16 head vocabularies) of 80 packed bytes
// + 10 E4M3 block scales) is larger than any Vulkan buffer (4 GiB - 4 on RADV here), so it lives in `blocks` buffers
// split on row boundaries: block b holds rows [b * rows_per_block, (b + 1) * rows_per_block). A small host-visible
// "block table" (kMaxBlocks table bases, then kMaxBlocks scale bases, u64 device addresses) is read by the shaders: for
// a valid id (0 <= id < rows, rows < 2^32), r = uint(id), b = r / rows_per_block, the row's bytes are at
// table_base[b] + (r - b * rows_per_block) * row_bytes (a 32-bit in-block byte offset from a 64-bit base, P1's rule) and
// its NVFP4 scales at scale_base[b] + (r - b * rows_per_block) * 10. Nothing in the kernels depends on where the
// blocks live or how large they are, so the same descriptor serves a one-block table (bit-equal results, proven in
// k4_embed_test) and the real 7-block one. SplitTable below allocates and fills such a table.
//
// The K2 projections (key, value) run vkk2::run with HIP's row-class routing (k4::Plans::cls, `cls` of forward here; 0 =
// t): class <= 16 the small-M GEMV (rows independent of the call), class >= 17 the large-m K2 GEMM (k2_gemm_vk.h: shift
// and row-count invariant). So a split of a step is bit-equal to the step when every part passes the step's class (the
// HIP k4_forward_test's plans.cls = 16 rule), and forward_rows (HIP: rows plans) is class <= 16. Rows are not bit-equal
// across the 16 / 17 class boundary (k2_gemm_vk.h, as on HIP). Classes >= 17 need vkk2::gemm_probe on the context and
// cooperative matrices (CERE_ERR_UNSUPPORTED otherwise; no chunked-GEMV fallback). The conv runs token-parallel at
// every t (one kernel, the CUDA conv_tile_kernel's arithmetic): decode and prefill share every output's arithmetic.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk4 {

// The A4 / Flash-Next text config (executor/cuda/src/kernels/k4_ple.h takes them from the generated
// cere_model_constants.h; the values are those of the generated HIP header build/hip/gen/kernels/k4_ple.h).
constexpr int kHidden = 2560;
constexpr int kHcCount = 4;
constexpr int kChannels = kHidden * kHcCount;  // 10,240
constexpr int kNgramSize = 3;
constexpr int kHeadsPerNgram = 8;
constexpr int kNgramHeads = (kNgramSize - 1) * kHeadsPerNgram;  // 16
constexpr int kEmbedDim = 2560;
constexpr int kHeadDim = kEmbedDim / kNgramHeads;  // 160
constexpr int kConvKernel = 4;
constexpr int kConvDilation = kNgramSize;
constexpr int kConvStateLen = (kConvKernel - 1) * kConvDilation;  // 9
constexpr int kHistoryLen = kNgramSize - 1;
constexpr int64_t kEosTokenId = 248044;
constexpr uint32_t kMaxRows = 16;  // rows of one forward_rows call (k4::kMaxRows)

// NVFP4 row geometry (k4_nvfp4.h).
constexpr uint32_t kNvfp4RowBytes = 80;
constexpr uint32_t kNvfp4RowScales = 10;

enum class TableFormat : uint32_t { kFp8 = 0, kNvfp4 = 1 };

constexpr uint32_t kMaxBlocks = 64;              // block-table entries (1 KiB): 64 x 4 GiB covers any PLE table
constexpr uint32_t kBlockTableBytes = 16 * kMaxBlocks;

// Bytes per row of the packed table, and of its block scales (0 for FP8).
inline uint32_t row_bytes(TableFormat f) { return f == TableFormat::kNvfp4 ? kNvfp4RowBytes : uint32_t(kHeadDim); }
inline uint32_t scale_bytes(TableFormat f) { return f == TableFormat::kNvfp4 ? kNvfp4RowScales : 0u; }

// What the kernels see of a split table.
struct TableView {
    uint64_t desc = 0;            // device address of the block table (u64 table_base[kMaxBlocks], scale_base[kMaxBlocks])
    int64_t rows = 0;             // table_rows; ids outside [0, rows) read zero bytes
    uint32_t rows_per_block = 0;  // >= 1
    uint32_t blocks = 0;          // 1 .. kMaxBlocks, blocks * rows_per_block >= rows
};

// Where SplitTable's blocks are allocated. kDefault: the launch layer's Context::alloc (device-local + host-visible on
// RADV/UMA, heap 1). kHostHeap: a host-visible memory type that is NOT device-local (heap 0, P0-5 "path B", memory type
// 2 on AMD): the real table must live there because heap 1 holds the 76.6 GB arena. Allocated through the launch
// layer's Context::allocate(bytes, out, host_heap = true).
// Every pipeline the functions below can request: the hash, gate, conv and state kernels, the embed (+ gather) kernel of
// each table format asked, K1's rmsnorm of width kHidden, K2's BF16 GEMV (k = kEmbedDim, m 1..16) and, when `gemm`, K2's
// GEMM + probe (a forward over more than 16 rows, or at class >= 17). Model: the PLE table is NVFP4 (S3 artifact) or FP8:
// {fp8_table, nvfp4_table} as deployed, gemm true.
std::vector<vk::PipelineSpec> pipeline_specs(bool fp8_table, bool nvfp4_table, bool gemm);

enum class Memory { kDefault, kHostHeap };

// A PLE table split into <= max_block_bytes buffers on row boundaries, plus its block table. Block b's buffer holds its
// rows' packed bytes, then (NVFP4) at a 256-byte aligned offset its rows' block scales. Every block is mapped: the
// upload writes rows through row()/scales() (the host layer's "upload writes, device reads" contract).
class SplitTable {
public:
    // false (and *why) on a refusal: rows <= 0 or >= 2^32, a block geometry above the limits, too many blocks, or an
    // allocation failure (everything allocated so far is freed).
    bool create(vk::Context &ctx, TableFormat format, int64_t rows, uint64_t max_block_bytes, Memory memory, std::string *why);
    void destroy(vk::Context &ctx);
    TableView view() const { return view_; }
    TableFormat format() const { return format_; }
    uint8_t *row(int64_t r) const;     // host mapping of packed row r
    uint8_t *scales(int64_t r) const;  // host mapping of row r's 10 block scales (NVFP4 only)
    const std::vector<vk::Buffer> &buffers() const { return blocks_; }
    uint32_t memory_type() const { return memory_type_; }

private:
    TableFormat format_ = TableFormat::kFp8;
    TableView view_{};
    uint64_t scale_off_ = 0;
    std::vector<vk::Buffer> blocks_;
    vk::Buffer desc_{};
    Memory memory_ = Memory::kDefault;
    uint32_t memory_type_ = UINT32_MAX;
};

// A host-visible, not device-local buffer with a device address (heap 0): Context::allocate(.., host_heap = true);
// refuses (false, *why) sizes above the device limits and a device without such a memory type. Freed by Context::free.
bool alloc_host_heap(vk::Context &ctx, uint64_t bytes, vk::Buffer &out, uint32_t *memory_type, std::string *why);

// Device-accessible weights and tables of one PLE layer (k4::Weights with addresses).
struct Weights {
    TableFormat format = TableFormat::kFp8;
    TableView table;
    uint64_t table_scale = 0;   // kFp8: BF16 [1]; kNvfp4: 0
    uint64_t global_scale = 0;  // kNvfp4: F32 [1]; kFp8: 0
    uint64_t multipliers = 0;       // int64 [3]
    uint64_t head_vocab_sizes = 0;  // int64 [16]
    uint64_t head_offsets = 0;      // int64 [16]
    uint64_t key_proj = 0;    // BF16 [10240, 2560]
    uint64_t value_proj = 0;  // BF16 [2560, 2560]
    uint64_t norm_key = 0, norm_query = 0, norm_conv = 0;  // BF16 [10240]
    uint64_t conv_weight = 0;  // BF16 [10240, 4]
    uint64_t proj_info = 0;    // optional: key_proj / value_proj K2 row info (proj_info_bytes, prepare_projections); 0 = scan in the call
    int64_t eos_token_id = kEosTokenId;
    float eps = 1e-6f;
};

struct State {
    uint64_t history = 0;     // int64 [2]
    uint64_t conv_state = 0;  // BF16 [10240, 9], columns oldest -> newest
};

struct Capture {
    int64_t row = 0;
    uint64_t history = 0;
    uint64_t conv_state = 0;
};

struct RowsCall {
    uint32_t sequences = 0;
    State states[kMaxRows] = {};
    uint32_t first[kMaxRows] = {};
    uint32_t count[kMaxRows] = {};
    State saves[kMaxRows] = {};
};

// Per-call table (segments and state jobs) the embed / conv / state shaders read. embed, forward and forward_rows write
// it at record time into the caller's host-visible `call_table` at byte offset `table_off` (16-byte aligned,
// kCallTableBytes long); that region must not be rewritten before the command buffer has executed (K1's rule).
constexpr uint32_t kCallTableBytes = 4096;

// Ids of `n` windows int64 [n, 3] -> int64 [n, 16], as `_hash_contexts` (bit-exact 64-bit integer arithmetic).
cere_status ngram_hash_windows(vk::Context &ctx, uint64_t windows, int64_t n, uint64_t multipliers, uint64_t head_vocab_sizes,
                               uint64_t head_offsets, int64_t eos_token_id, uint64_t ids);

// FP8 gather: out[i] = bf16(e4m3 row ids[i]) (times *scale when scale != 0), BF16 [n, 160]; ids outside [0, rows) give
// zero rows. The table's view must be an FP8 (160-byte row) split table.
cere_status gather(vk::Context &ctx, const TableView &table, uint64_t ids, int64_t n, uint64_t scale, uint64_t out);

// NVFP4 gather: out[i] = row ids[i] decoded as k4_nvfp4.h (E2M1 x E4M3 exact, x F32 global, BF16 RNE, NaN 0x7FFF).
cere_status gather_nvfp4(vk::Context &ctx, const TableView &table, uint64_t global_scale, uint64_t ids, int64_t n, uint64_t out);

cere_status check_table(const Weights &w);

// emb [t, 2560] of t >= 1 tokens after `history` (unchanged), each token's 16 n-gram rows decoded per w.format.
cere_status embed(vk::Context &ctx, const Weights &w, uint64_t history, uint64_t tokens, int64_t t, uint64_t emb,
                  vk::Buffer &call_table, uint32_t table_off);

// Includes the two K2 GEMM workspaces (vkk2::gemm_workspace_bytes(t, 10,240) and (t, 2,560): key and value run
// without a barrier between them).
size_t workspace_bytes(int64_t max_tokens);

// Bytes of Weights::proj_info (key_proj's 10,240 then value_proj's 2,560 K2 row-info words).
size_t proj_info_bytes();
// Records the K2 weight scans (vkk2::gemm_prepare_weight) of w.key_proj and w.value_proj into w.proj_info (16-byte
// aligned), once per layer at load; the caller puts a barrier after it. CERE_ERR_ARGUMENT for null / misaligned.
cere_status prepare_projections(vk::Context &ctx, const Weights &w);

// One PLE step over t >= 1 tokens (k4::forward without the K2 plans: the projections are recorded on vkk2::run with
// row class `cls`, k4::Plans::cls: 0 = t, else t <= cls, CERE_ERR_ARGUMENT otherwise).
// out = query + delta (add_query; out may alias query) else delta; advances state in place; capture as k4::Capture.
cere_status forward(vk::Context &ctx, const Weights &w, const State &state, uint64_t tokens, int64_t t, uint64_t query,
                    bool add_query, uint64_t out, uint64_t workspace, size_t workspace_size, vk::Buffer &call_table,
                    uint32_t table_off, const Capture *capture = nullptr, uint32_t cls = 0);

// k4::forward_rows: m <= 16 rows of several sequences; row r bit-equal to forward(t = 1) on it after its sequence's
// earlier rows; saves[r] the state after row r. The projections are class <= 16 (HIP's rows plans): the K2 GEMV.
cere_status forward_rows(vk::Context &ctx, const Weights &w, const RowsCall &call, uint64_t tokens, int64_t m, uint64_t query,
                         bool add_query, uint64_t out, uint64_t workspace, size_t workspace_size, vk::Buffer &call_table,
                         uint32_t table_off);

}  // namespace cere::vkk4
