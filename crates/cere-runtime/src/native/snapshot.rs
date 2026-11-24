//! The `CERESNP1` snapshot size the D8 config bound needs (m211 design note D2, D8): the executor's section plan
//! (`executor/cuda/src/host/snapshot_format.cpp` `plan`) for the kinds this build writes (0-8, ledger "Snapshot
//! section kinds" writer rule, row CR13; a model with drafts adds kind 10, M2.9), on the Qwen3.8-Flash-Next config
//! `cere-native` describes (`support.cpp`). Both sides pin the same totals (`executor/tests/host/test_state.cpp`,
//! `tests/config_native.rs`).

/// Header (128 B) and one 24 B section-table entry per written kind (D2).
const HEADER_BYTES: u64 = 128;
const ENTRY_BYTES: u64 = 24;
/// Every section offset is 256-B aligned (D2).
const SECTION_ALIGN: u64 = 256;
/// Full-attention layers (3, 7, ..., 47) and GDN layers (the other 36).
const ATTN_LAYERS: u64 = 12;
const GDN_LAYERS: u64 = 36;
/// One FP8 E4M3 K or V row of a full-attention layer: 2 KV heads x 256.
const KV_ROW_BYTES: u64 = 512;
/// QSA compressed key / ring entry: BF16 [128]; one compressed key per 4 positions.
const QSA_ENTRY_BYTES: u64 = 128 * 2;
const QSA_BLOCK: u64 = 4;
/// GDN conv state BF16 [10,240][3] and ssm state BF16 [48][128][128] per layer.
const GDN_CONV_BYTES: u64 = 10_240 * 3 * 2;
const GDN_SSM_BYTES: u64 = 48 * 128 * 128 * 2;
/// PLE history i64 [2] and conv BF16 [10,240][9].
const PLE_BYTES: u64 = 2 * 8 + 10_240 * 9 * 2;
/// Sampler state (kind 8, m27 I12): the session's penalty `out_counts` u32 [248,320 = vocab].
const SAMPLER_BYTES: u64 = 248_320 * 4;
/// Kinds this build writes without drafts: 0-8.
const KINDS: usize = 9;
/// Kind 10 (M2.9 drafts, `snapshot_format.cpp` `draft_parts`): the MTP layer's FP8 K and V rows, its QSA keys and
/// ring, then the 16-B pending proposals.
const MTP_LAYERS: u64 = 1;
const DRAFT_PROPOSAL_BYTES: u64 = 16;

/// Section sizes of kinds 0-8 for a snapshot at `position` (`first_row` 0) holding `token_count` tokens.
fn section_bytes(position: u64, token_count: u64) -> Option<[u64; KINDS]> {
    let rows = |n: u64, row: u64| ATTN_LAYERS.checked_mul(n)?.checked_mul(row);
    Some([
        token_count.checked_mul(4)?,
        rows(position, KV_ROW_BYTES)?,
        rows(position, KV_ROW_BYTES)?,
        rows(position / QSA_BLOCK, QSA_ENTRY_BYTES)?,
        rows(position % QSA_BLOCK, QSA_ENTRY_BYTES)?,
        GDN_LAYERS * GDN_CONV_BYTES,
        GDN_LAYERS * GDN_SSM_BYTES,
        PLE_BYTES,
        SAMPLER_BYTES,
    ])
}

/// Kind 10's bytes at `position` (`first_row` 0).
fn draft_bytes(position: u64) -> Option<u64> {
    let rows = |n: u64, row: u64| MTP_LAYERS.checked_mul(n)?.checked_mul(row);
    rows(position, KV_ROW_BYTES)?
        .checked_mul(2)?
        .checked_add(rows(position / QSA_BLOCK, QSA_ENTRY_BYTES)?)?
        .checked_add(rows(position % QSA_BLOCK, QSA_ENTRY_BYTES)?)?
        .checked_add(DRAFT_PROPOSAL_BYTES)
}

/// Bytes of a snapshot at `position` holding `token_count` tokens (U1 sizes a victim's snapshot from its position,
/// D8); `draft` adds kind 10 (a model with drafts, M2.9). `None` on u64 overflow.
pub fn snapshot_bytes(position: u64, token_count: u64, draft: bool) -> Option<u64> {
    let sections = section_bytes(position, token_count)?;
    let extra = if draft {
        Some(draft_bytes(position)?)
    } else {
        None
    };
    let entries = KINDS as u64 + u64::from(draft);
    let mut end = HEADER_BYTES + ENTRY_BYTES * entries;
    for bytes in sections.into_iter().chain(extra) {
        end = end
            .checked_next_multiple_of(SECTION_ALIGN)?
            .checked_add(bytes)?;
    }
    Some(end)
}

/// Bytes of the largest snapshot a session of `max_context_tokens` can produce: `position` = `token_count` = the
/// context (the size grows with `position`, and `token_count` ≤ the context). `None` on u64 overflow.
pub fn max_snapshot_bytes(max_context_tokens: u64, draft: bool) -> Option<u64> {
    snapshot_bytes(max_context_tokens, max_context_tokens, draft)
}

/// Arena bytes one parked snapshot of a session of `max_context_tokens` may take: [`max_snapshot_bytes`] rounded
/// up to the arena's 256-B sub-allocation alignment (D2; `executor/cuda/src/host/api.cpp` `add_snapshot`). The
/// D8 bound and `cere-native generate --preempt-every` size the arena with it. `None` on u64 overflow.
pub fn arena_slot_bytes(max_context_tokens: u64, draft: bool) -> Option<u64> {
    max_snapshot_bytes(max_context_tokens, draft)?.checked_next_multiple_of(SECTION_ALIGN)
}
