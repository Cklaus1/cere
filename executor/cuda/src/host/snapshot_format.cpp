#include "host/snapshot_format.h"

#include "host/sha256.h"
#include "model/layout.h"

#include <cstring>

namespace cere::host::snapshot {

namespace {

// Header field offsets (D2).
constexpr uint64_t kAtFormat = 8, kAtHeaderBytes = 12, kAtSectionCount = 16, kAtReserved = 20, kAtLayout = 24,
                   kAtPosition = 56, kAtTokenCount = 64, kAtFed = 72, kAtFlags = 80, kAtFirstRow = 88,
                   kAtTotal = 96, kAtReserved24 = 104;

uint32_t get32(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

uint64_t get64(const uint8_t *p) { return uint64_t(get32(p)) | uint64_t(get32(p + 4)) << 32; }

void put32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}

void put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i));
}

bool all_zero(const uint8_t *p, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        if (p[i] != 0) return false;
    }
    return true;
}

// a * b * c * d without u64 overflow.
bool product(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t *out) {
    uint64_t ab = 0, abc = 0;
    return !__builtin_mul_overflow(a, b, &ab) && !__builtin_mul_overflow(ab, c, &abc) &&
           !__builtin_mul_overflow(abc, d, out);
}

bool align_up(uint64_t x, uint64_t *out) {
    if (__builtin_add_overflow(x, kSectionAlign - 1, out)) return false;
    *out = *out / kSectionAlign * kSectionAlign;
    return true;
}

}  // namespace

bool draft_parts(const cere_qwen4_exp_config &cfg, uint64_t position, uint64_t first_row, DraftParts *out) {
    if (first_row > position) return false;
    const uint64_t layers = cfg.mtp_num_hidden_layers;
    const uint64_t ratio = cfg.indexer_compress_ratio == 0 ? 1 : cfg.indexer_compress_ratio;
    const uint64_t kv_width = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
    const uint64_t bf16 = model::layout::kBf16;
    DraftParts parts;
    uint64_t cursor = 0;
    const auto place = [&](DraftPart *part, uint64_t bytes) {
        part->offset = cursor;
        part->bytes = bytes;
        return !__builtin_add_overflow(cursor, bytes, &cursor);
    };
    uint64_t kv = 0;
    uint64_t keys = 0;
    uint64_t ring = 0;
    if (!product(layers, position - first_row, kv_width, 1, &kv) ||
        !product(layers, position / ratio - first_row / ratio, cfg.indexer_head_dim, bf16, &keys) ||
        !product(layers, position % ratio, cfg.indexer_head_dim, bf16, &ring) || !place(&parts.attn_k, kv) ||
        !place(&parts.attn_v, kv) || !place(&parts.qsa_keys, keys) || !place(&parts.qsa_ring, ring) ||
        !place(&parts.proposals, kDraftProposalBytes)) {
        return false;
    }
    parts.total = cursor;
    *out = parts;
    return true;
}

bool section_bytes(const cere_qwen4_exp_config &cfg, uint32_t kind, uint64_t position, uint64_t first_row,
                   uint64_t token_count, uint64_t *out) {
    if (first_row > position) return false;
    uint64_t attention = 0;
    const uint64_t layers = cfg.num_hidden_layers < 64 ? cfg.num_hidden_layers : 64;
    for (uint32_t l = 0; l < layers; ++l) attention += model::layout::full_attention(cfg, l) ? 1 : 0;
    const uint64_t gdn = layers - attention;
    const uint64_t ratio = cfg.indexer_compress_ratio == 0 ? 1 : cfg.indexer_compress_ratio;
    // The per-layer shapes of model::layout::carve_session (one session slot).
    const uint64_t kv_width = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
    const uint64_t conv_channels = 2 * uint64_t(cfg.linear_num_key_heads) * cfg.linear_key_head_dim +
                                   uint64_t(cfg.linear_num_value_heads) * cfg.linear_value_head_dim;
    const uint64_t conv_taps = cfg.linear_conv_kernel_dim == 0 ? 0 : cfg.linear_conv_kernel_dim - 1;
    const uint64_t ple_history = cfg.ngram_size == 0 ? 0 : cfg.ngram_size - 1;
    const uint64_t ple_taps =
        uint64_t(cfg.ple_conv_kernel_size == 0 ? 0 : cfg.ple_conv_kernel_size - 1) * cfg.ngram_size;
    const uint64_t bf16 = model::layout::kBf16;
    switch (kind) {
        case kTokens:
            return product(token_count, sizeof(uint32_t), 1, 1, out);
        case kAttnK:
        case kAttnV:
            return product(attention, position - first_row, kv_width, 1, out);
        case kQsaKeys:
            return product(attention, position / ratio - first_row / ratio, cfg.indexer_head_dim, bf16, out);
        case kQsaRing:
            return product(attention, position % ratio, cfg.indexer_head_dim, bf16, out);
        case kGdnConv:
            return product(gdn, conv_channels, conv_taps, bf16, out);
        case kGdnSsm:
            return product(gdn, uint64_t(cfg.linear_num_value_heads) * cfg.linear_key_head_dim,
                           cfg.linear_value_head_dim, bf16, out);
        case kPle: {
            uint64_t conv = 0;
            if (!product(uint64_t(cfg.hc_count) * cfg.hidden_size, ple_taps, bf16, 1, &conv)) return false;
            return !__builtin_add_overflow(ple_history * model::layout::kI64, conv, out);
        }
        case kSampler:
            return product(cfg.vocab_size, sizeof(uint32_t), 1, 1, out);
        case kDraft: {
            DraftParts parts;
            if (!draft_parts(cfg, position, first_row, &parts)) return false;
            *out = parts.total;
            return true;
        }
        case kRopeDelta:
            return product(1, sizeof(int64_t), 1, 1, out);
        default:
            return false;
    }
}

bool layout(const std::vector<Section> &sized, Plan *out) {
    Plan plan;
    uint64_t cursor = kHeaderBytes + kEntryBytes * sized.size();
    for (const Section &s : sized) {
        Section placed = s;
        if (!align_up(cursor, &placed.offset) || __builtin_add_overflow(placed.offset, s.bytes, &cursor)) return false;
        plan.sections.push_back(placed);
    }
    plan.total_bytes = cursor;
    *out = std::move(plan);
    return true;
}

bool plan(const cere_qwen4_exp_config &cfg, uint64_t position, uint64_t first_row, uint64_t token_count, Plan *out,
          uint32_t kinds) {
    if ((kinds & kRequiredKinds) != kRequiredKinds || (kinds & ~kDraftWritableKinds) != 0) return false;
    std::vector<Section> sized;
    for (uint32_t kind = 0; kind < 32; ++kind) {
        if (((kinds >> kind) & 1) == 0) continue;
        Section s;
        s.kind = kind;
        if (!section_bytes(cfg, kind, position, first_row, token_count, &s.bytes)) return false;
        sized.push_back(s);
    }
    return layout(sized, out);
}

std::array<uint8_t, 32> layout_id(const cere_qwen4_exp_config &cfg, uint32_t writable_kinds) {
    Sha256 h;
    uint8_t word[4];
    auto u32 = [&](uint32_t v) {
        put32(word, v);
        h.update(word, sizeof(word));
    };
    u32(kFormat);
    h.update(&cfg, sizeof(cfg));
    u32(CERE_DTYPE_F8_E4M3);
    u32(uint32_t(__builtin_popcount(writable_kinds)));
    for (uint32_t kind = 0; kind < 32; ++kind) {
        if (((writable_kinds >> kind) & 1) != 0) u32(kind);
    }
    return h.finish();
}

void write_header(uint8_t *dst, const Header &header, const Plan &plan) {
    uint64_t end = kHeaderBytes + kEntryBytes * plan.sections.size();
    std::memset(dst, 0, end);
    std::memcpy(dst, kMagic, sizeof(kMagic));
    put32(dst + kAtFormat, kFormat);
    put32(dst + kAtHeaderBytes, uint32_t(kHeaderBytes));
    put32(dst + kAtSectionCount, uint32_t(plan.sections.size()));
    std::memcpy(dst + kAtLayout, header.layout_id.data(), header.layout_id.size());
    put64(dst + kAtPosition, header.position);
    put64(dst + kAtTokenCount, header.token_count);
    put64(dst + kAtFed, header.fed);
    put64(dst + kAtFlags, header.flags);
    put64(dst + kAtFirstRow, header.first_row);
    put64(dst + kAtTotal, plan.total_bytes);
    for (size_t i = 0; i < plan.sections.size(); ++i) {
        uint8_t *entry = dst + kHeaderBytes + kEntryBytes * i;
        const Section &s = plan.sections[i];
        put32(entry, s.kind);
        put64(entry + 8, s.offset);
        put64(entry + 16, s.bytes);
        std::memset(dst + end, 0, s.offset - end);
        end = s.offset + s.bytes;
    }
}

cere_status validate(const uint8_t *bytes, uint64_t len, const Expect &expect, Header *header, Plan *plan) {
    // len >= 128 and len == total_bytes.
    if (bytes == nullptr || expect.config == nullptr || len < kHeaderBytes || get64(bytes + kAtTotal) != len) {
        return CERE_ERR_ARGUMENT;
    }
    // Magic, header_bytes, the header's reserved bytes (entry reserved and pads: with the table below, H2).
    if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0 || get32(bytes + kAtHeaderBytes) != kHeaderBytes ||
        get32(bytes + kAtReserved) != 0 || !all_zero(bytes + kAtReserved24, kHeaderBytes - kAtReserved24)) {
        return CERE_ERR_ARGUMENT;
    }
    if (get32(bytes + kAtFormat) != kFormat) return CERE_ERR_UNSUPPORTED;
    if (std::memcmp(bytes + kAtLayout, expect.layout_id.data(), expect.layout_id.size()) != 0) {
        return CERE_ERR_UNSUPPORTED;
    }
    const uint32_t count = get32(bytes + kAtSectionCount);
    if (count > kMaxSections || kHeaderBytes + kEntryBytes * count > len) return CERE_ERR_ARGUMENT;
    // Per entry: offset + bytes without wrap and within len, 256-B aligned, ascending and non-overlapping (also with
    // the table), entry reserved zero, every byte between the table and the sections zero, no trailing bytes.
    Plan parsed;
    uint64_t end = kHeaderBytes + kEntryBytes * count;
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t *entry = bytes + kHeaderBytes + kEntryBytes * i;
        Section s;
        s.kind = get32(entry);
        s.offset = get64(entry + 8);
        s.bytes = get64(entry + 16);
        uint64_t stop = 0;
        if (get32(entry + 4) != 0 || __builtin_add_overflow(s.offset, s.bytes, &stop) || stop > len ||
            s.offset % kSectionAlign != 0 || s.offset < end || !all_zero(bytes + end, s.offset - end)) {
            return CERE_ERR_ARGUMENT;
        }
        end = stop;
        parsed.sections.push_back(s);
    }
    if (end != len) return CERE_ERR_ARGUMENT;
    parsed.total_bytes = len;
    // Kinds strictly ascending; 9 and 12+ unknown; 10/11 only when this build reads them; 0-8 all present.
    for (uint32_t i = 1; i < count; ++i) {
        if (parsed.sections[i].kind <= parsed.sections[i - 1].kind) return CERE_ERR_ARGUMENT;
    }
    uint32_t present = 0;
    for (const Section &s : parsed.sections) {
        if (s.kind >= 12 || s.kind == kGrammar) return CERE_ERR_ARGUMENT;
        present |= uint32_t(1) << s.kind;
    }
    if ((present & ~expect.readable_kinds) != 0) return CERE_ERR_UNSUPPORTED;
    if ((present & kRequiredKinds) != kRequiredKinds) return CERE_ERR_ARGUMENT;
    Header h;
    std::memcpy(h.layout_id.data(), bytes + kAtLayout, h.layout_id.size());
    h.position = get64(bytes + kAtPosition);
    h.token_count = get64(bytes + kAtTokenCount);
    h.fed = get64(bytes + kAtFed);
    h.flags = get64(bytes + kAtFlags);
    h.first_row = get64(bytes + kAtFirstRow);
    h.total_bytes = len;
    for (const Section &s : parsed.sections) {
        uint64_t want = 0;
        if (!section_bytes(*expect.config, s.kind, h.position, h.first_row, h.token_count, &want) || want != s.bytes) {
            return CERE_ERR_ARGUMENT;
        }
    }
    if (h.first_row != 0) return CERE_ERR_ARGUMENT;  // M2.10's own-edge entries (ledger M3)
    if (h.position == 0 || h.position > expect.model_max_context_tokens ||
        (expect.session_max_context_tokens != 0 && h.position > expect.session_max_context_tokens)) {
        return CERE_ERR_CAPACITY;
    }
    if (h.fed != h.position || h.token_count < h.fed || h.token_count > h.fed + 1) return CERE_ERR_ARGUMENT;
    const uint8_t *tokens = bytes + parsed.sections[0].offset;
    for (uint64_t i = 0; i < h.token_count; ++i) {
        if (get32(tokens + 4 * i) >= expect.config->vocab_size) return CERE_ERR_ARGUMENT;
    }
    // Kind 11 (M2.13, ledger CR14): the next DECODE's rotary position, position + rope_delta, is in [1, position].
    for (const Section &s : parsed.sections) {
        if (s.kind != kRopeDelta) continue;
        const int64_t delta = int64_t(get64(bytes + s.offset));
        if (delta > 0 || delta <= -int64_t(h.position)) return CERE_ERR_ARGUMENT;
    }
    if ((h.flags & ~expect.known_flags) != 0) return CERE_ERR_ARGUMENT;
    // Kind 10 (M2.9, M29S2-07): exactly a drafting session's image has it, and its proposals are well formed.
    if (((expect.readable_kinds >> kDraft) & 1) != 0) {
        const bool draft = ((present >> kDraft) & 1) != 0;
        if (draft == ((h.flags & CERE_WORK_NO_SPECULATION) != 0)) return CERE_ERR_ARGUMENT;
        uint32_t count = 0;
        uint32_t ids[3] = {};
        if (read_proposals(bytes, parsed, &count, ids)) {
            if (count > 3) return CERE_ERR_ARGUMENT;
            for (uint32_t i = 0; i < 3; ++i) {
                if (i < count ? ids[i] >= expect.config->vocab_size : ids[i] != 0) return CERE_ERR_ARGUMENT;
            }
        }
    }
    *header = h;
    *plan = std::move(parsed);
    return CERE_OK;
}

void write_rope_delta(uint8_t *dst, const Plan &plan, int64_t rope_delta) {
    for (const Section &s : plan.sections) {
        if (s.kind == kRopeDelta) put64(dst + s.offset, uint64_t(rope_delta));
    }
}

bool read_rope_delta(const uint8_t *bytes, const Plan &plan, int64_t *out) {
    for (const Section &s : plan.sections) {
        if (s.kind != kRopeDelta) continue;
        *out = int64_t(get64(bytes + s.offset));
        return true;
    }
    return false;
}

uint64_t read_first_row(const uint8_t *bytes) { return get64(bytes + kAtFirstRow); }

bool read_proposals(const uint8_t *bytes, const Plan &plan, uint32_t *count, uint32_t ids[3]) {
    for (const Section &s : plan.sections) {
        if (s.kind != kDraft) continue;
        const uint8_t *p = bytes + s.offset + s.bytes - kDraftProposalBytes;  // the last part
        *count = get32(p);
        for (uint32_t i = 0; i < 3; ++i) ids[i] = get32(p + 4 + 4 * i);
        return true;
    }
    return false;
}

bool write_proposals(uint8_t *dst, const Plan &plan, uint32_t count, const uint32_t *ids) {
    for (const Section &s : plan.sections) {
        if (s.kind != kDraft) continue;
        uint8_t *p = dst + s.offset + s.bytes - kDraftProposalBytes;
        put32(p, count);
        for (uint32_t i = 0; i < 3; ++i) put32(p + 4 + 4 * i, i < count ? ids[i] : 0);
        return true;
    }
    return false;
}

}  // namespace cere::host::snapshot
