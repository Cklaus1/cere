// The `CERESNP1` snapshot byte format (m211 design note D2, D3, D14; ledger "Snapshot section kinds"): section
// sizes, the writer's header and section table, `layout_id`, and the restore/import byte checks. No CUDA includes:
// the host layer writes the header and the host kinds (0, 11), the device writes the state sections in place.
#pragma once

#include "cere_executor.h"

#include <array>
#include <cstdint>
#include <vector>

namespace cere::host::snapshot {

constexpr uint8_t kMagic[8] = {'C', 'E', 'R', 'E', 'S', 'N', 'P', '1'};
constexpr uint32_t kFormat = 1;
constexpr uint64_t kHeaderBytes = 128;
constexpr uint64_t kEntryBytes = 24;
constexpr uint64_t kSectionAlign = 256;  // every section offset (DMA)
constexpr uint32_t kMaxSections = 16;

// Section kinds (ledger "Snapshot section kinds"); 12-15 unassigned.
enum Kind : uint32_t {
    kTokens = 0,     // u32 [token_count]
    kAttnK = 1,      // FP8 E4M3 [attention layers][position - first_row][kv_heads * head_dim]
    kAttnV = 2,      // as kind 1
    kQsaKeys = 3,    // BF16 [attention layers][floor(position / r) - floor(first_row / r)][indexer_head_dim]
    kQsaRing = 4,    // BF16 [attention layers][position mod r][indexer_head_dim]
    kGdnConv = 5,    // BF16 [GDN layers][conv channels][conv_kernel - 1]
    kGdnSsm = 6,     // BF16 [GDN layers][value heads][key dim][value dim]
    kPle = 7,        // i64 [ngram - 1] + BF16 [hc * hidden][(ple_conv_kernel - 1) * ngram]
    kSampler = 8,    // u32 [vocab]: the session's out_counts (M2.7 D5; the counter RNG has no state, D4)
    kGrammar = 9,    // M2.8: no executor bytes, never written
    kDraft = 10,     // M2.9
    kRopeDelta = 11, // M2.13
};

// Kinds this build writes, as a bit mask (bit k = kind k). Ledger writer rule (rows CR12/CR13/CR14): kinds 8, 10 and
// 11 are written by whichever of M2.11 and their owner reaches `main` second. M2.7 merges after M2.11 (CR13), M2.13
// (kind 11, CR14) and M2.9 (kind 10, CR12) land after M2.11, so this build writes and reads kinds 0-8, 10 and 11.
// Kinds 0-8 are in every image; kind 10 only in a drafting session's (a model with drafts, M29S2-07); kind 11 only in
// an image session's. A model without drafts writes kWritableKinds, a model with drafts kDraftWritableKinds: the
// layout_id (and so every snapshot byte) of a model without drafts is unchanged by M2.9.
constexpr uint32_t kWritableKinds = 0x1FF | 1u << kRopeDelta;
constexpr uint32_t kDraftWritableKinds = kWritableKinds | 1u << kDraft;
constexpr uint32_t kRequiredKinds = 0x1FF;  // kinds 0-8: always present

// Kind 10 (M2.9 D8, M29S2-07): the MTP layer's state for [first_row, position), packed in this order without padding
// (M = mtp_num_hidden_layers, kv_width = num_key_value_heads * head_dim, r = indexer_compress_ratio): FP8 E4M3 K
// [M][position - first_row][kv_width], FP8 E4M3 V (as K), BF16 QSA compressed keys [M][floor(position / r) -
// floor(first_row / r)][indexer_head_dim], BF16 QSA ring [M][position mod r][indexer_head_dim], then the session's
// pending proposals: u32 LE [4] = count (0..3), then 3 ids (< vocab; the unused ones 0).
struct DraftPart {
    uint64_t offset = 0;  // inside the kind 10 section
    uint64_t bytes = 0;
};
struct DraftParts {
    DraftPart attn_k;
    DraftPart attn_v;
    DraftPart qsa_keys;
    DraftPart qsa_ring;
    DraftPart proposals;
    uint64_t total = 0;  // the section's bytes
};
constexpr uint64_t kDraftProposalBytes = 16;
// False on u64 overflow or first_row > position.
bool draft_parts(const cere_qwen4_exp_config &cfg, uint64_t position, uint64_t first_row, DraftParts *out);
// Kinds the host layer writes and reads (tokens, rope_delta); the device writes and loads the others.
constexpr uint32_t kHostKinds = 1u << kTokens | 1u << kRopeDelta;

struct Section {
    uint32_t kind = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

// The header fields after magic/format/header_bytes (D2).
struct Header {
    std::array<uint8_t, 32> layout_id{};
    uint64_t position = 0;
    uint64_t token_count = 0;
    uint64_t fed = 0;
    uint64_t flags = 0;
    uint64_t first_row = 0;
    uint64_t total_bytes = 0;
};

// Sections in table order at their offsets; `total_bytes` = end of the last section (no trailing pad).
struct Plan {
    std::vector<Section> sections;
    uint64_t total_bytes = 0;
};

// The size of `kind` (0-8, 10, 11) derived from the config and the header's rows; false on u64 overflow, an unknown
// kind or first_row > position.
bool section_bytes(const cere_qwen4_exp_config &cfg, uint32_t kind, uint64_t position, uint64_t first_row,
                   uint64_t token_count, uint64_t *out);

// Section table of the given sizes (ascending kind), each offset 256-B aligned after the table; false on overflow.
bool layout(const std::vector<Section> &sized, Plan *out);

// The plan of a snapshot this build writes: the kinds of `kinds` (a session's state: kinds 0-8, plus 10 for a drafting
// session and 11 for an image session) at their derived sizes; false on overflow, or for `kinds` without 0-8 or with a
// kind this build cannot write.
bool plan(const cere_qwen4_exp_config &cfg, uint64_t position, uint64_t first_row, uint64_t token_count, Plan *out,
          uint32_t kinds = kRequiredKinds);

// SHA-256 over format u32 LE, the cere_qwen4_exp_config bytes, the KV dtype (CERE_DTYPE_F8_E4M3) u32 LE, the number
// of writable kinds u32 LE and each writable kind u32 LE ascending (D3; placement never enters it, ledger X19).
std::array<uint8_t, 32> layout_id(const cere_qwen4_exp_config &cfg, uint32_t writable_kinds = kWritableKinds);

// Writes magic, format, header_bytes, the header fields, the section table and zeroes every other byte outside the
// sections of `plan` (dst holds plan.total_bytes); section bodies are the caller's.
void write_header(uint8_t *dst, const Header &header, const Plan &plan);

// What a restore or import accepts.
struct Expect {
    const cere_qwen4_exp_config *config = nullptr;
    std::array<uint8_t, 32> layout_id{};
    uint64_t model_max_context_tokens = 0;
    uint64_t session_max_context_tokens = 0;  // restore: the target session's; import: 0 (not checked)
    uint64_t known_flags = 0;                 // CERE_WORK_KNOWN_FLAGS of the build
    // A model without vision cannot take kind 11 (no image sessions); a model with drafts reads kind 10.
    uint32_t readable_kinds = kWritableKinds;
};

// D14's byte checks after the handle and session checks, in D14's order; on CERE_OK fills `header` and `plan`.
// ARGUMENT: length, magic, header_bytes, non-zero reserved or pad, section table, kinds 9/12+, a missing kind 0-8, a
// size off its derived value, first_row != 0, the idle invariant, a token id >= vocab, a kind 11 rope_delta outside
// (-position, 0], an unknown flag bit, and with kind 10 readable: kind 10 present while flags carry
// CERE_WORK_NO_SPECULATION or absent while they do not (a drafting session's image always has it), or its proposals
// (count > 3, an id >= vocab, an unused id non-zero). UNSUPPORTED: format != 1, another layout_id, a kind 10/11
// outside `readable_kinds`. CAPACITY: position 0 or above the model's or the session's context.
cere_status validate(const uint8_t *bytes, uint64_t len, const Expect &expect, Header *header, Plan *plan);

// Kind 10's pending proposals (count, then 3 ids; its last kDraftProposalBytes) in the image `bytes` of `plan`; false
// when `plan` has no kind 10. The writer stores `count` (<= 3) ids and zeroes the unused ones.
bool read_proposals(const uint8_t *bytes, const Plan &plan, uint32_t *count, uint32_t ids[3]);
bool write_proposals(uint8_t *dst, const Plan &plan, uint32_t count, const uint32_t *ids);

// Kind 11 of an image session's snapshot (M2.13 D14, ledger CR14): rope_delta i64 LE. The writer fills it in `dst`
// (no-op when `plan` has no kind 11); the reader returns false when the validated `plan` has none (a text session).
void write_rope_delta(uint8_t *dst, const Plan &plan, int64_t rope_delta);
bool read_rope_delta(const uint8_t *bytes, const Plan &plan, int64_t *out);

// The header's first_row of an image write_header wrote (no check: the host tier's own images, Sessions O211).
uint64_t read_first_row(const uint8_t *bytes);

}  // namespace cere::host::snapshot
