"""K15 (tasks/m211-spec.md D10, ledger R): native session snapshot/restore on the real A4 model, GPU0, container only.

Suite `model` (`native_dev.py test --suite model`). One model is created, uploaded and sealed as in
`test_model_k10_long.py` (`oracle.create_model`), here with the build's `max_context_tokens` 200,000, three sessions
over a paged KV pool of 200,000 tokens (the shipped profiles' `kv_pool_tokens`, M2.5 D4/D6: the derived pool of
three full-context sessions is refused) and the v3.L `cere_model_desc.snapshot_host_bytes` arena of 4 GiB (M2.11 D5,
owner N1 = (a); one snapshot at the build max fits). Each session is opened at the context its case needs, so the
pool holds two or three of them; where two sessions of a case do not fit (positions near the build max) the source's
own run is taken first and the source released before its copy is opened (reported per position). Every case drives
the ktest session entries of `k8_ktest.cpp` (spec Interfaces "ktest session entries"); prompts are synthetic and
deterministic (the chat template around numbered records, cut in token space; token IDs stay in memory, nothing is
written).

Restore-then-decode is a native-vs-native gate (bit-exact against the uninterrupted run of the same library); it is
not SGLang parity and no K15 result is a G1 result.

Cases (D10 (1)-(8)):
* `test_restore_then_decode` (1), (7): for snapshots at positions 1, 63, 64, 65, 66, 4,096, 4,097, 4,098, 9,000
  and 199,936 (build max - 64; `position mod 4` takes 0-3): the source session is prefilled teacher-forced and its
  predicted token committed unfed (`cere_ktest_session_commit_next`: the shape a ticket leaves, which restore
  requires, HardenHost H1), its state digest taken, a snapshot taken (the image length equals D2's derived size of
  `position` + 1 tokens, reported per position); a fresh
  session in another slot (at 199,936: the source's, released first, as the pool holds one such session) restores
  it, its digest equals the source's, and 64 greedy DECODE steps of the restored copy equal the source's own 64 steps
  bit for bit: token, every `cere_token_score` field, and the
  `forward_trace`-contract taps (B8 names, shapes and 2,560-column values of the step's row). At 65 and 4,097 a
  never-snapshotted session's 64 greedy steps equal the restored copy's as well.
* `test_restore_into_reused_slot` (2): the source is released and a new session (in its freed slot) restores the
  snapshot; decoding is bit-exact. Permuted pages (D10 (2), M2.5 is in the build): the copy's pages are armed in
  reverse order (`cere_ktest_session_set_pages`, K11's one-shot order, stated in `test_model_k11_paged.py`) before it
  is opened; it restores the snapshot, decodes bit-exactly, and its own snapshot is byte-identical to the source's
  (pages never enter the bytes, D4 / ledger X19).
* `test_read_release_import_restore` (3): the image is read to host memory, the snapshot released, the bytes
  imported and restored into a fresh session; decoding is bit-exact.
* `test_fork_two_slots` (4): two copies restored from one snapshot and the original decode step-interleaved in three
  slots; all three runs are bit-identical.
* `test_d14_rejections` (5): every D14 byte rejection of a corrupted image returns its status through import or
  restore, the target session stays fresh (digest unchanged; a valid restore afterwards decodes bit-exactly), a
  valid image into a smaller session is CAPACITY, and the arena is unchanged: then filling it holds exactly
  arena / aligned image snapshots and the next one is CAPACITY.
* `test_snapshots_byte_identical` (6): two snapshots of one idle session, and the snapshot of a restored copy, are
  byte-identical.
* `test_cancel_snapshot_continue` (8): a greedy DECODE ticket cancelled after k = 17 tokens, snapshot, restore into
  another slot and continue for the remaining 47 tokens equals the uninterrupted 64-token run (scores bit-exact).
* `test_snapshot_restore_timing` (R2, D11b): at 2,048, 73,728 and 200,000 (the build max), wall time and GB/s of
  snapshot and restore and the cold prefill of the session's tokens but its committed unfed last one (raw
  observation, no bound); the restored digest equals the source's.
* `test_restore_requires_unfed_token` (HardenHost H1): the image of a session that fed every token (token_count ==
  fed) is refused by restore (CERE_ERR_ARGUMENT, the copy unchanged); `cere_ktest_session_commit_next` refuses a
  fresh session, a token >= vocab and a second unfed token; a teacher-forced feed starting with another token than
  the unfed one is refused (the session unchanged); after the commit the image (`position` + 1 tokens) restores,
  its digest equals the source's and 64 greedy steps equal the source's.
* `K15Image.test_image_session_round_trip` (tasks/m213-spec.md C14, ledger CR14): a vision model (context 2,048,
  within `indexer_budget`) prefills a synthetic prompt with one seeded-random image (grid 1 x 16 x 16, 64 pad rows;
  pixels in memory only) and commits one token; its snapshot holds 9 sections, kind 11 last with the 8-B
  `rope_delta` equal to the mRoPE value; restored into a fresh session its digest equals the source's and 32 greedy
  DECODE steps are bit-identical to the source's own; the same image with kind 11 zeroed restores to another digest
  and other scores (so the DECODE reads the restored `rope_delta`).
* `K15Memory.test_memory_vs_merge_base` (R1 [REVISED 2026-10-11]): this build at arena 0 and 4 GiB and the
  merge-base library (`$CERE_BASE_KTEST_LIB`, default `build/native-l0/libcere_ktest.so`; it must accept the paged
  K15 desc, so it is an M2.5 build), exact: the executor's ledger device bytes (`cere_ktest_model_ledger`) equal the
  device estimate at both arena sizes and, less `p3b_scratch_delta` on a P3b build against the K7 merge-base,
  bracket the merge-base's (CAPACITY at - 1 B, taken at the bytes); VmLck
  grows by exactly the arena's locked pages; ring and mapped block unchanged; mapped + locked = host estimate.
  cudaMemGetInfo and MemAvailable deltas are recorded only.

Native entry points (the contract with the kernel track; `oracle.call` appends the stream; 0 = success, otherwise a
`cere_status`; "host" buffers are CPU memory, "device" buffers GPU0 memory; types of `cere_executor.h`). The model
handle comes from `cere_ktest_model_create/put/seal` (stated in `test_model_k10_long.py`, called by
`oracle.create_model`); released by `cere_ktest_model_release`.

    int cere_ktest_session_open(uint64_t model, uint64_t max_context_tokens, uint64_t *out_session /*host [1]*/,
                                cudaStream_t stream);
        A session of max_context_tokens (0 = the model's) on the handle's executor.
    int cere_ktest_session_feed(uint64_t session, const uint32_t *tokens /*host*/, uint64_t n,
                                uint64_t max_new_tokens, uint64_t cancel_after, uint8_t *names /*host*/,
                                uint64_t names_capacity, int64_t *shapes /*host [entry_capacity][3]*/,
                                uint64_t entry_capacity, void *values /*device*/, uint64_t values_capacity_bytes,
                                cere_token_score *scores /*host*/, uint64_t *out_counts /*host [3]*/,
                                cudaStream_t stream);
        max_new_tokens 0: teacher-forced feed (an empty session: one PREFILL in workspace-row chunks, scores[0]
        predicts the next token; else n DECODE rows, scores[i] after tokens[i]; it leaves every token fed, and a
        session holding a committed, unfed token feeds it first: tokens[0] must be it, else CERE_ERR_ARGUMENT); with
        names non-null the `forward_trace` sink taps the last fed row and out_counts = {entries, name bytes, value
        bytes}. Otherwise one
        greedy work item through submit/poll (PREFILL of tokens on an empty session, else DECODE with tokens null and
        n 0), cancelled after cancel_after commits (0 = never): scores of the committed tokens, out_counts =
        {committed, finish, 0}.
    int cere_ktest_session_commit_next(uint64_t session, uint64_t token, cudaStream_t stream);
        Appends `token` (< vocab; the one the last teacher-forced feed scored) as the committed, unfed token of an
        idle session that fed every one of its (>= 1) tokens: the shape a ticket leaves and cere_session_restore
        requires of an image (token_count == fed + 1). The next teacher-forced feed must start with it. Else
        CERE_ERR_ARGUMENT; a session at its max_context_tokens: CERE_ERR_CAPACITY.
    int cere_ktest_session_snapshot(uint64_t session, void *bytes /*host*/, uint64_t capacity,
                                    uint64_t *out_bytes /*host [1]*/, uint64_t *out_snapshot /*host [1]*/,
                                    cudaStream_t stream);
        cere_session_snapshot; the image's length; bytes non-null: the image read into it; out_snapshot non-null:
        the snapshot handle stays in the arena, else it is released.
    int cere_ktest_session_restore(uint64_t session, uint64_t snapshot, const void *bytes /*host*/, uint64_t len,
                                   cudaStream_t stream);
        cere_session_restore from the snapshot handle, or (snapshot 0) import bytes[0, len), restore, release.
    int cere_ktest_session_close(uint64_t handle, cudaStream_t stream);
        Releases a session or a snapshot handle.
    int cere_ktest_state_digest(uint64_t session, uint8_t *out /*host [32]*/, cudaStream_t stream);
        SHA-256 over the session's serialized device sections (kinds >= 1, table order, logical rows, valid ring
        entries; no header, no kind 0; ledger M4).
    int cere_ktest_session_set_pages(uint64_t model, const uint32_t *order /*host [n]*/, uint64_t n,
                                     cudaStream_t /*stream*/);
        K11's one-shot page order for the model's next session create (stated in `test_model_k11_paged.py`).
    int cere_ktest_session_prefill_images(uint64_t session, const uint32_t *tokens, uint64_t n,
                                          const cere_image_input *images, uint64_t image_count,
                                          uint64_t max_new_tokens, cere_token_score *scores, uint64_t *out_counts,
                                          cudaStream_t stream);
        One greedy PREFILL work item with images (NO_SPECULATION) on an empty session (M2.13 C14);
        `cere_ktest_session_prefill_images_flags` (stated in `test_model_vl43_long.py`) takes the work flags.
"""
from __future__ import annotations

import contextlib
import ctypes
import functools
import hashlib
import importlib
import json
import os
import time
import unittest
from pathlib import Path

import torch
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

import oracle

ROW = "K15"
MAX_CONTEXT = 200_000  # the build max (support.cpp)
SESSIONS = 3
PAGE_TOKENS = 64  # host/kv_pool.h kKvPageTokens
POOL_TOKENS = 200_000  # the shipped profiles' kv_pool_tokens: 3,125 pages
ARENA_BYTES = 4 << 30  # owner N1 = (a)
STEPS = 64
CASE_CONTEXT = 16_384  # a session's context unless its case needs more: three fit the pool with room to spare
POSITIONS = (1, 63, 64, 65, 66, 4_096, 4_097, 4_098, 9_000, MAX_CONTEXT - STEPS)
NEVER_SNAPSHOTTED = (65, 4_097)
CANCEL_PROMPT = 4_096
CANCEL_AFTER = 17
NAMES_CAPACITY = 1 << 16
ENTRY_CAPACITY = 512
VALUES_CAPACITY = 4 << 20
# R2 (m211 design note Tier 2): 2,048 / 73,728 / the build max.
TIMING_POSITIONS = (2_048, 73_728, MAX_CONTEXT)
# R1: the merge-base library (as test_model_k8_forward.py) and the bounce ring pinned at executor creation
# (cuda_device.h: 4 x 64 MiB).
SRC = Path(__file__).resolve().parents[3]
BASE_KTEST_LIB = Path(os.environ.get("CERE_BASE_KTEST_LIB", str(SRC / "build/native-l0/libcere_ktest.so")))
RING_BYTES = 4 * (64 << 20)

CERE_ERR_ARGUMENT = -1  # cere_executor.h
CERE_ERR_UNSUPPORTED = -2
CERE_ERR_CAPACITY = -3
CERE_FINISH_LENGTH = 1
CERE_FINISH_CANCELLED = 3

SENTINEL = "<<RECORDS>>"
QUESTION = "\n\nCopy every record above, one per line, in the same order, without any commentary."
COLORS = ("red", "green", "blue", "amber", "violet", "silver", "black")
ANIMALS = ("otter", "heron", "lynx", "badger", "falcon", "marten", "ibex", "crane", "viper", "bison", "gecko")

# CERESNP1 header offsets (m211 design note D2).
AT_FORMAT, AT_HEADER_BYTES, AT_SECTION_COUNT, AT_RESERVED = 8, 12, 16, 20
AT_LAYOUT, AT_POSITION, AT_TOKEN_COUNT, AT_FED, AT_FLAGS, AT_FIRST_ROW, AT_TOTAL, AT_RESERVED24 = (
    24, 56, 64, 72, 80, 88, 96, 104)
HEADER_BYTES, ENTRY_BYTES, SECTION_ALIGN = 128, 24, 256


@functools.cache
def tokenizer():
    return get_tokenizer(str(oracle.MODEL_DIR))


def synthetic_ids(total: int) -> list[int]:
    """Exactly `total` prompt IDs: the chat template around numbered records (cut in token space) and a request to
    copy them (a long answer, so greedy decoding meets no stop token early)."""
    tok = tokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": SENTINEL + QUESTION}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    if text.count(SENTINEL) != 1:
        raise oracle.OracleError("the rendered chat template does not hold the record sentinel exactly once")
    head, tail = text.split(SENTINEL)
    head_ids = list(tok.encode(head, add_special_tokens=False))
    tail_ids = list(tok.encode(tail, add_special_tokens=False))
    need = total - len(head_ids) - len(tail_ids)
    records = "\n".join(f"Record {i:05d}: the {COLORS[i % len(COLORS)]} {ANIMALS[i % len(ANIMALS)]} weighs "
                         f"{(i * 37) % 101 + 1} kg." for i in range(max(need // 8, 1)))
    body_ids = list(tok.encode(records, add_special_tokens=False))
    if need <= 0 or len(body_ids) < need:
        raise oracle.OracleError(f"synthetic prompt of {total} tokens: {len(body_ids)} record tokens for {need}")
    return head_ids + body_ids[:need] + tail_ids


def _align(x: int) -> int:
    return (x + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN


def image_bytes(tc: dict, position: int, token_count: int, *, rope_delta: bool = False) -> int:
    """D2's derived length of a full snapshot (first_row 0) with kinds 0-8 (kind 8 the sampler's u32 [vocab]
    out_counts, ledger CR13; and with `rope_delta` an image session's kind 11, 8 B, M2.13 CR14): header, one entry
    per kind, 256-B aligned sections in kind order, no trailing padding."""

    def get(key: str) -> int:
        return int(tc.get(key, 0))  # as oracle's cere_qwen4_exp_config (absent = 0)

    attention = sum(1 for kind in tc["layer_types"] if kind == "full_attention")
    gdn = get("num_hidden_layers") - attention
    kv = get("num_key_value_heads") * get("head_dim")
    key_row = get("indexer_head_dim") * 2
    ratio = get("indexer_compress_ratio") or 1
    conv = 2 * get("linear_num_key_heads") * get("linear_key_head_dim") + \
        get("linear_num_value_heads") * get("linear_value_head_dim")
    taps = max(get("linear_conv_kernel_dim") - 1, 0)
    ple_taps = max(get("ple_conv_kernel_size") - 1, 0) * get("ngram_size")
    sizes = [
        4 * token_count,
        attention * position * kv,
        attention * position * kv,
        attention * (position // ratio) * key_row,
        attention * (position % ratio) * key_row,
        gdn * conv * taps * 2,
        gdn * get("linear_num_value_heads") * get("linear_key_head_dim") * get("linear_value_head_dim") * 2,
        max(get("ngram_size") - 1, 0) * 8 + get("hc_count") * get("hidden_size") * ple_taps * 2,
        4 * get("vocab_size"),
    ]
    if rope_delta:
        sizes.append(8)
    cursor = HEADER_BYTES + ENTRY_BYTES * len(sizes)
    for size in sizes:
        cursor = _align(cursor) + size
    return cursor


# -- ktest session entries ---------------------------------------------------------------------------------------


def session_open(model: int, max_context_tokens: int = 0) -> int:
    out = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_open", model, max_context_tokens, out)
    return int(out.item())


def close(handle: int) -> None:
    oracle.call("cere_ktest_session_close", handle)


def digest(session: int) -> torch.Tensor:
    out = torch.zeros(32, dtype=torch.uint8)
    oracle.call("cere_ktest_state_digest", session, out)
    return out


def feed(session: int, tokens: list[int], *, trace: bool = False) -> tuple[torch.Tensor, tuple | None]:
    """Teacher-forced feed; the score rows (int32 [n][4]: id, runner_up, logit bits, runner_up_logit bits) and, with
    `trace`, the tapped row's (names, shapes, values) as host tensors."""
    ids = torch.tensor(tokens, dtype=torch.int32)
    scores = torch.full((len(tokens), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    if not trace:
        oracle.call("cere_ktest_session_feed", session, ids, len(tokens), 0, 0, None, 0, None, 0, None, 0, scores,
                    counts)
        return scores, None
    names = torch.zeros(NAMES_CAPACITY, dtype=torch.uint8)
    shapes = torch.zeros((ENTRY_CAPACITY, 3), dtype=torch.int64)
    values = oracle.empty((VALUES_CAPACITY,), torch.uint8)
    oracle.call("cere_ktest_session_feed", session, ids, len(tokens), 0, 0, names, NAMES_CAPACITY, shapes,
                ENTRY_CAPACITY, values, VALUES_CAPACITY, scores, counts)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if entries == 0:
        raise oracle.OracleError("a traced feed tapped no entry")
    return scores, (names[:name_bytes].clone(), shapes[:entries].clone(), values[:value_bytes].cpu())


def feed_status(session: int, tokens: list[int]) -> int:
    """An untraced teacher-forced feed; the raw status."""
    ids = torch.tensor(tokens, dtype=torch.int32)
    scores = torch.full((len(tokens), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    return oracle.call_status("cere_ktest_session_feed", session, ids, len(tokens), 0, 0, None, 0, None, 0, None, 0,
                              scores, counts)


def commit_next(session: int, token: int) -> None:
    """The fully fed session's committed, unfed `token` (the shape a ticket leaves; restore requires it)."""
    oracle.call("cere_ktest_session_commit_next", session, token)


def commit_next_status(session: int, token: int) -> int:
    """cere_ktest_session_commit_next; the raw status."""
    return oracle.call_status("cere_ktest_session_commit_next", session, token)



def ticket(session: int, tokens: list[int] | None, max_new: int, cancel_after: int = 0) -> tuple[torch.Tensor, int]:
    """One greedy work item (PREFILL of `tokens` on an empty session, else DECODE): (score rows of the committed
    tokens, finish)."""
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    if tokens is None:
        oracle.call("cere_ktest_session_feed", session, None, 0, max_new, cancel_after, None, 0, None, 0, None, 0,
                    scores, counts)
    else:
        ids = torch.tensor(tokens, dtype=torch.int32)
        oracle.call("cere_ktest_session_feed", session, ids, len(tokens), max_new, cancel_after, None, 0, None, 0,
                    None, 0, scores, counts)
    committed, finish = int(counts[0].item()), int(counts[1].item())
    return scores[:committed].clone(), finish


def snapshot(session: int, *, keep: bool, capacity: int = 0) -> tuple[int, int, torch.Tensor | None]:
    """(snapshot handle or 0, image length, image bytes or None)."""
    image = torch.zeros(capacity, dtype=torch.uint8) if capacity else None
    length = torch.zeros(1, dtype=torch.int64)
    handle = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_snapshot", session, image, capacity, length, handle if keep else None)
    n = int(length.item())
    return int(handle.item()), n, None if image is None else image[:n].clone()


def pages(context: int) -> int:
    """KV pages a session of `context` tokens reserves at create (tasks/m25-spec.md D3)."""
    return -(-context // PAGE_TOKENS)


def two_fit(context: int) -> bool:
    """Whether two sessions of `context` fit the pool at once."""
    return 2 * pages(context) <= POOL_TOKENS // PAGE_TOKENS


def arm_pages(model: int, order: list[int]) -> None:
    """Arms the page order of the model's next session create (one-shot, as test_model_k11_paged.py)."""
    oracle.call("cere_ktest_session_set_pages", model, torch.tensor(order, dtype=torch.int32), len(order))


def restore(session: int, snapshot_handle: int) -> None:
    oracle.call("cere_ktest_session_restore", session, snapshot_handle, None, 0)


def restore_bytes_status(session: int, image: torch.Tensor) -> int:
    """cere_ktest_session_restore from host bytes (import, restore, release); the raw status."""
    fn = oracle.ktest("cere_ktest_session_restore")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    rc = fn(ctypes.c_uint64(session), ctypes.c_uint64(0), ctypes.c_void_p(image.data_ptr()),
            ctypes.c_uint64(image.numel()), ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(rc)


def snapshot_status(session: int) -> tuple[int, int]:
    """A kept snapshot through cere_ktest_session_snapshot; (status, handle)."""
    fn = oracle.ktest("cere_ktest_session_snapshot")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    length = torch.zeros(1, dtype=torch.int64)
    handle = torch.zeros(1, dtype=torch.int64)
    rc = fn(ctypes.c_uint64(session), ctypes.c_void_p(0), ctypes.c_uint64(0), ctypes.c_void_p(length.data_ptr()),
            ctypes.c_void_p(handle.data_ptr()), ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(rc), int(handle.item())


class Run:
    """A greedy decode of `steps` teacher-forced DECODE rows: score rows and the taps of every step."""

    def __init__(self):
        self.scores: list[torch.Tensor] = []
        self.names: list[torch.Tensor] = []
        self.shapes: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def step(self, session: int, token: int) -> int:
        scores, (names, shapes, values) = feed(session, [token], trace=True)
        self.scores.append(scores[0])
        self.names.append(names)
        self.shapes.append(shapes)
        self.values.append(values)
        return int(scores[0, 0].item())


def greedy(session: int, first: int, steps: int = STEPS) -> Run:
    """`steps` greedy DECODE rows starting from the token `first` (the prediction of the state's last row)."""
    run = Run()
    token = first
    for _ in range(steps):
        token = run.step(session, token)
    return run


def check_runs(tc: unittest.TestCase, case: str, run: Run, reference: Run) -> None:
    oracle.check_count(tc, ROW, f"{case} steps", len(run.scores), expected=len(reference.scores))
    oracle.check_bit_exact(tc, ROW, f"{case} scores", torch.stack(run.scores), torch.stack(reference.scores))
    oracle.check_bit_exact(tc, ROW, f"{case} tap names", torch.cat(run.names), torch.cat(reference.names))
    oracle.check_bit_exact(tc, ROW, f"{case} tap shapes", torch.cat(run.shapes), torch.cat(reference.shapes))
    oracle.check_bit_exact(tc, ROW, f"{case} tap values", torch.cat(run.values), torch.cat(reference.values))


def _put(image: torch.Tensor, at: int, value: int, width: int) -> torch.Tensor:
    out = image.clone()
    out[at:at + width] = torch.tensor(list(value.to_bytes(width, "little")), dtype=torch.uint8)
    return out


def _get(image: torch.Tensor, at: int, width: int) -> int:
    return int.from_bytes(bytes(image[at:at + width].tolist()), "little")


class K15Snapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.prompt = synthetic_ids(MAX_CONTEXT)
        cls.model = oracle.create_model(MAX_CONTEXT, max_sessions=SESSIONS, kv_pool_tokens=POOL_TOKENS,
                                        snapshot_host_bytes=ARENA_BYTES)

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    def setUp(self):
        self.handles: list[int] = []

    def tearDown(self):
        # Snapshots before sessions, newest first (the ABI's release order for a shutdown).
        for handle in reversed(self.handles):
            close(handle)

    def open(self, max_context_tokens: int = CASE_CONTEXT) -> int:
        session = session_open(self.model, max_context_tokens)
        self.handles.append(session)
        return session

    def keep(self, session: int) -> int:
        handle, _, _ = snapshot(session, keep=True)
        self.handles.append(handle)
        return handle

    def drop(self, handle: int) -> None:
        self.handles.remove(handle)
        close(handle)

    def prefilled(self, position: int, context: int = CASE_CONTEXT) -> tuple[int, int]:
        """A session of `context` teacher-forced through prompt[0, position) with the predicted next token committed
        unfed (as a ticket leaves it); (session, that token)."""
        session = self.open(context)
        scores, _ = feed(session, self.prompt[:position])
        first = int(scores[0, 0].item())
        commit_next(session, first)
        return session, first

    def test_restore_then_decode(self):
        for position in POSITIONS:
            with self.subTest(position=position):
                context = max(CASE_CONTEXT, position + STEPS)
                source, first = self.prefilled(position, context)
                source_digest = digest(source)
                started = time.perf_counter()
                handle, length, _ = snapshot(source, keep=True)
                snapshot_s = time.perf_counter() - started
                self.handles.append(handle)
                oracle.report_row(ROW, f"bytes p{position}", phase="diag", position=position, bytes=length,
                                  snapshot_seconds=snapshot_s)
                oracle.check_count(self, ROW, f"image bytes p{position}", length,
                                   expected=image_bytes(self.tc, position, position + 1))
                other_slot = two_fit(context)
                if not other_slot:
                    # The pool holds one session of this context: the source's run first, then its pages go.
                    uninterrupted = greedy(source, first)
                    self.drop(source)
                    source = None
                copy = self.open(context)
                started = time.perf_counter()
                restore(copy, handle)
                oracle.report_row(ROW, f"restore p{position}", phase="diag", position=position, context=context,
                                  other_slot=other_slot, restore_seconds=time.perf_counter() - started)
                oracle.check_bit_exact(self, ROW, f"digest p{position}", digest(copy), source_digest)
                restored = greedy(copy, first)
                if other_slot:
                    uninterrupted = greedy(source, first)
                check_runs(self, f"restore p{position}", restored, uninterrupted)
                self.drop(copy)
                self.drop(handle)
                if position in NEVER_SNAPSHOTTED:
                    fresh, fresh_first = self.prefilled(position, context)
                    oracle.check_count(self, ROW, f"never-snapshotted first p{position}", fresh_first,
                                       expected=first)
                    check_runs(self, f"never-snapshotted p{position}", greedy(fresh, fresh_first), restored)
                    self.drop(fresh)
                if source is not None:
                    self.drop(source)

    def test_restore_into_reused_slot(self):
        position = 4_098
        source, first = self.prefilled(position)
        capacity = image_bytes(self.tc, position, position + 1)
        _, _, image = snapshot(source, keep=False, capacity=capacity)
        handle = self.keep(source)
        reference = greedy(source, first)
        self.drop(source)
        n = pages(CASE_CONTEXT)
        arm_pages(self.model, list(range(n - 1, -1, -1)))
        copy = self.open()
        restore(copy, handle)
        _, _, again = snapshot(copy, keep=False, capacity=capacity)
        oracle.check_bit_exact(self, ROW, "permuted pages: snapshot of the restored copy", again, image)
        check_runs(self, "reused slot, permuted pages", greedy(copy, first), reference)
        oracle.report_row(ROW, "permuted pages", phase="diag", pages=n, order="reversed")

    def test_read_release_import_restore(self):
        position = 4_097
        source, first = self.prefilled(position)
        capacity = image_bytes(self.tc, position, position + 1)
        _, length, image = snapshot(source, keep=False, capacity=capacity)
        oracle.check_count(self, ROW, "read bytes", length, expected=capacity)
        copy = self.open()
        oracle.check_count(self, ROW, "import status", restore_bytes_status(copy, image), expected=0)
        check_runs(self, "import", greedy(copy, first), greedy(source, first))

    def test_fork_two_slots(self):
        position = 9_000
        source, first = self.prefilled(position)
        handle = self.keep(source)
        copies = [self.open(), self.open()]
        for copy in copies:
            restore(copy, handle)
        runs = [Run(), Run(), Run()]
        tokens = [first, first, first]
        for _ in range(STEPS):
            for i, session in enumerate([source] + copies):
                tokens[i] = runs[i].step(session, tokens[i])
        check_runs(self, "fork copy 1", runs[1], runs[0])
        check_runs(self, "fork copy 2", runs[2], runs[0])

    def test_snapshots_byte_identical(self):
        position = 4_097
        source, _ = self.prefilled(position)
        capacity = image_bytes(self.tc, position, position + 1)
        _, _, first = snapshot(source, keep=False, capacity=capacity)
        _, _, second = snapshot(source, keep=False, capacity=capacity)
        oracle.check_bit_exact(self, ROW, "two snapshots", second, first)
        copy = self.open()
        oracle.check_count(self, ROW, "import status", restore_bytes_status(copy, first), expected=0)
        _, _, again = snapshot(copy, keep=False, capacity=capacity)
        oracle.check_bit_exact(self, ROW, "snapshot of the restored copy", again, first)

    def corruptions(self, image: torch.Tensor) -> list[tuple[str, torch.Tensor, int]]:
        """(name, corrupted image, D14 status) for every byte rejection of tasks/m211-spec.md D14."""
        count = _get(image, AT_SECTION_COUNT, 4)
        entry = [HEADER_BYTES + ENTRY_BYTES * i for i in range(count)]
        table_end = HEADER_BYTES + ENTRY_BYTES * count
        tokens_at = _get(image, entry[0] + 8, 8)
        n = image.numel()
        longer = torch.cat([image, torch.zeros(SECTION_ALIGN, dtype=torch.uint8)])
        longer = _put(longer, AT_TOTAL, n + SECTION_ALIGN, 8)
        flipped_magic = image.clone()
        flipped_magic[0] ^= 1
        flipped_layout = image.clone()
        flipped_layout[AT_LAYOUT] ^= 1
        position = _get(image, AT_POSITION, 8)
        return [
            ("short of total_bytes", image[:n - 1].clone(), CERE_ERR_ARGUMENT),
            ("below the header", image[:HEADER_BYTES - 1].clone(), CERE_ERR_ARGUMENT),
            ("trailing bytes", longer, CERE_ERR_ARGUMENT),
            ("magic", flipped_magic, CERE_ERR_ARGUMENT),
            ("header_bytes", _put(image, AT_HEADER_BYTES, HEADER_BYTES + 8, 4), CERE_ERR_ARGUMENT),
            ("header reserved", _put(image, AT_RESERVED, 1, 4), CERE_ERR_ARGUMENT),
            ("header reserved 24", _put(image, AT_RESERVED24 + 23, 1, 1), CERE_ERR_ARGUMENT),
            ("format 2", _put(image, AT_FORMAT, 2, 4), CERE_ERR_UNSUPPORTED),
            ("layout_id", flipped_layout, CERE_ERR_UNSUPPORTED),
            ("section_count 17", _put(image, AT_SECTION_COUNT, 17, 4), CERE_ERR_ARGUMENT),
            ("entry reserved", _put(image, entry[3] + 4, 1, 4), CERE_ERR_ARGUMENT),
            ("pad after the table", _put(image, table_end, 1, 1), CERE_ERR_ARGUMENT),
            ("misaligned offset", _put(image, entry[5] + 8, _get(image, entry[5] + 8, 8) + 1, 8), CERE_ERR_ARGUMENT),
            ("kinds not ascending", _put(image, entry[2], 1, 4), CERE_ERR_ARGUMENT),
            ("kind 9", _put(image, entry[8], 9, 4), CERE_ERR_ARGUMENT),
            ("kind 12", _put(image, entry[8], 12, 4), CERE_ERR_ARGUMENT),
            ("kind 10 unreadable", _put(image, entry[8], 10, 4), CERE_ERR_UNSUPPORTED),
            ("section size", _put(image, entry[0] + 16, _get(image, entry[0] + 16, 8) - 4, 8), CERE_ERR_ARGUMENT),
            ("first_row 4", _put(image, AT_FIRST_ROW, 4, 8), CERE_ERR_ARGUMENT),
            ("fed not position", _put(image, AT_FED, position - 1, 8), CERE_ERR_ARGUMENT),
            ("token id at vocab", _put(image, tokens_at, int(self.tc["vocab_size"]), 4), CERE_ERR_ARGUMENT),
            ("unknown flag bit", _put(image, AT_FLAGS, 1 << 62, 8), CERE_ERR_ARGUMENT),
        ]

    def test_d14_rejections(self):
        position = 4_097
        source, first = self.prefilled(position)
        capacity = image_bytes(self.tc, position, position + 1)
        _, _, image = snapshot(source, keep=False, capacity=capacity)
        reference = greedy(source, first, 8)
        self.drop(source)
        target = self.open()
        fresh = digest(target)
        for name, corrupted, status in self.corruptions(image):
            with self.subTest(corruption=name):
                oracle.check_count(self, ROW, f"d14 {name}", restore_bytes_status(target, corrupted), expected=status)
                oracle.check_bit_exact(self, ROW, f"d14 {name} session unchanged", digest(target), fresh)
        small = self.open(position - 1)
        oracle.check_count(self, ROW, "d14 position above the session context",
                           restore_bytes_status(small, image), expected=CERE_ERR_CAPACITY)
        oracle.check_bit_exact(self, ROW, "d14 small session unchanged", digest(small), fresh)
        self.drop(small)
        oracle.check_count(self, ROW, "d14 valid restore after the rejections", restore_bytes_status(target, image),
                           expected=0)
        check_runs(self, "d14 restored", greedy(target, first, 8), reference)
        # Arena unchanged by every rejection: a 66-position image fills it exactly arena // aligned length times.
        holder, _ = self.prefilled(66)
        length = image_bytes(self.tc, 66, 67)
        expected = ARENA_BYTES // _align(length)
        held: list[int] = []
        while True:
            status, handle = snapshot_status(holder)
            if status != 0:
                break
            held.append(handle)
            self.handles.append(handle)
            if len(held) > expected:
                break
        oracle.check_count(self, ROW, "arena full: snapshots held", len(held), expected=expected)
        oracle.check_count(self, ROW, "arena full: status", status, expected=CERE_ERR_CAPACITY)
        for handle in held:
            self.drop(handle)

    def test_cancel_snapshot_continue(self):
        prompt = synthetic_ids(CANCEL_PROMPT)
        straight = self.open()
        head, finish = ticket(straight, prompt, 1)
        oracle.check_count(self, ROW, "uninterrupted prefill finish", finish, expected=CERE_FINISH_LENGTH)
        whole, finish = ticket(straight, None, STEPS)
        oracle.check_count(self, ROW, "uninterrupted finish", finish, expected=CERE_FINISH_LENGTH)
        self.drop(straight)
        source = self.open()
        source_head, _ = ticket(source, prompt, 1)
        oracle.check_bit_exact(self, ROW, "cancel prefill scores", source_head, head)
        part, finish = ticket(source, None, STEPS, CANCEL_AFTER)
        oracle.check_count(self, ROW, "cancel finish", finish, expected=CERE_FINISH_CANCELLED)
        oracle.check_count(self, ROW, "cancel committed", part.shape[0], expected=CANCEL_AFTER)
        handle = self.keep(source)
        copy = self.open()
        restore(copy, handle)
        rest, finish = ticket(copy, None, STEPS - CANCEL_AFTER)
        oracle.check_count(self, ROW, "continued finish", finish, expected=CERE_FINISH_LENGTH)
        oracle.check_bit_exact(self, ROW, "cancel-snapshot-restore-continue scores", torch.cat([part, rest]), whole)

    def test_restore_requires_unfed_token(self):
        """HardenHost H1 (cere_session_restore: token_count == fed + 1) and cere_ktest_session_commit_next's
        refusals, then the committed image's round trip."""
        position = 66
        vocab = int(self.tc["vocab_size"])
        source = self.open()
        oracle.check_count(self, ROW, "commit_next on a fresh session", commit_next_status(source, 0),
                           expected=CERE_ERR_ARGUMENT)
        scores, _ = feed(source, self.prompt[:position])
        first = int(scores[0, 0].item())
        fed_bytes = image_bytes(self.tc, position, position)
        _, length, image = snapshot(source, keep=False, capacity=fed_bytes)
        oracle.check_count(self, ROW, "token_count == fed image bytes", length, expected=fed_bytes)
        copy = self.open()
        fresh = digest(copy)
        oracle.check_count(self, ROW, "restore of a token_count == fed image", restore_bytes_status(copy, image),
                           expected=CERE_ERR_ARGUMENT)
        oracle.check_bit_exact(self, ROW, "refused copy unchanged", digest(copy), fresh)
        oracle.check_count(self, ROW, "commit_next of token vocab", commit_next_status(source, vocab),
                           expected=CERE_ERR_ARGUMENT)
        commit_next(source, first)
        oracle.check_count(self, ROW, "second commit_next", commit_next_status(source, first),
                           expected=CERE_ERR_ARGUMENT)
        source_digest = digest(source)
        oracle.check_count(self, ROW, "feed of another token than the unfed one",
                           feed_status(source, [(first + 1) % vocab]), expected=CERE_ERR_ARGUMENT)
        oracle.check_bit_exact(self, ROW, "refused feed: source unchanged", digest(source), source_digest)
        committed_bytes = image_bytes(self.tc, position, position + 1)
        _, length, image = snapshot(source, keep=False, capacity=committed_bytes)
        oracle.check_count(self, ROW, "committed image bytes", length, expected=committed_bytes)
        oracle.check_count(self, ROW, "restore of the committed image", restore_bytes_status(copy, image),
                           expected=0)
        oracle.check_bit_exact(self, ROW, "committed image digest", digest(copy), source_digest)
        check_runs(self, "committed image", greedy(copy, first), greedy(source, first))

    def test_snapshot_restore_timing(self):
        """R2 (D11b raw observation): at 2,048, 73,728 and 200,000 (the build max, support.cpp), the cold prefill of
        position - 1 tokens into a fresh session of exactly that context (its predicted token then committed unfed, so
        it holds `position` tokens as a ticket leaves it), the snapshot into the arena and the restore into another
        fresh session (at the build max the pool holds one such session, so the source is released first), wall time
        each (stream-synchronized ktest calls) and GB/s of the image; the restored digest equals the source's."""
        for position in TIMING_POSITIONS:
            with self.subTest(position=position):
                source = self.open(position)
                torch.cuda.synchronize(oracle.DEVICE)
                started = time.perf_counter()
                scores, _ = feed(source, self.prompt[:position - 1])
                prefill_s = time.perf_counter() - started
                commit_next(source, int(scores[0, 0].item()))  # the session holds `position` tokens
                source_digest = digest(source)
                started = time.perf_counter()
                handle, length, _ = snapshot(source, keep=True)
                snapshot_s = time.perf_counter() - started
                self.handles.append(handle)
                other_slot = two_fit(position)
                if not other_slot:
                    self.drop(source)
                    source = None
                copy = self.open(position)
                started = time.perf_counter()
                restore(copy, handle)
                restore_s = time.perf_counter() - started
                oracle.report_row(ROW, f"timing p{position}", phase="diag", position=position, bytes=length,
                                  build_max=MAX_CONTEXT, other_slot=other_slot, prefill_seconds=prefill_s,
                                  snapshot_seconds=snapshot_s,
                                  snapshot_gb_per_s=length / snapshot_s / 1e9, restore_seconds=restore_s,
                                  restore_gb_per_s=length / restore_s / 1e9,
                                  restore_over_prefill=restore_s / prefill_s)
                oracle.check_count(self, ROW, f"timing image bytes p{position}", length,
                                   expected=image_bytes(self.tc, position - 1, position))
                oracle.check_bit_exact(self, ROW, f"timing digest p{position}", digest(copy), source_digest)
                self.drop(copy)
                self.drop(handle)
                if source is not None:
                    self.drop(source)


# -- M2.13 C14 (CR14): an image session's snapshot carries kind 11 (rope_delta) ---------------------------------------

IMAGE_CONTEXT = 2_048  # an image session's context stays within indexer_budget (ledger CR19)
IMAGE_TEXT = 512  # text tokens of the image prompt
IMAGE_AT = 24  # the image span (vision_start, pads, vision_end) is inserted before text token 24
IMAGE_GRID = (1, 16, 16)  # (t, h, w) patches: 64 pad rows after the 2 x 2 merge
IMAGE_PAD_ID = 1_000_001  # the PLE id of the pad rows (SGLang pad_value analog)
IMAGE_SEED = 1513
IMAGE_STEPS = 32
KIND_ROPE_DELTA = 11


class ImageInput(ctypes.Structure):
    """struct cere_image_input (80 B, ledger N; `pixel_sha256` = SHA-256 of the pixel_bytes at pixel_patches,
    tasks/m213l-spec.md VL-L7: an all-zero digest is ARGUMENT)."""
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32), ("grid_t", ctypes.c_uint32),
                ("grid_h", ctypes.c_uint32), ("grid_w", ctypes.c_uint32), ("pad_id", ctypes.c_uint32),
                ("token_offset", ctypes.c_uint64), ("pixel_patches", ctypes.c_void_p), ("pixel_bytes", ctypes.c_uint64),
                ("pixel_sha256", ctypes.c_uint8 * 32)]


def image_prompt() -> tuple[list[int], int, torch.Tensor, int]:
    """(prompt ids, the image's token_offset, its BF16 [t*h*w][patch elements] CPU patches from a seeded generator,
    the expected rope_delta): synthetic text with one image span (vision_start, t*h*w/4 image_token_id, vision_end)
    inserted at IMAGE_AT. Nothing is read from or written to disk but the model's config."""
    config = json.loads((oracle.MODEL_DIR / "config.json").read_text())
    vc = config["vision_config"]
    t, h, w = IMAGE_GRID
    merge = int(vc["spatial_merge_size"])
    pads = t * h * w // (merge * merge)
    text = synthetic_ids(IMAGE_TEXT)
    ids = (text[:IMAGE_AT] + [int(config["vision_start_token_id"])] + [int(config["image_token_id"])] * pads
           + [int(config["vision_end_token_id"])] + text[IMAGE_AT:])
    elements = int(vc["in_channels"]) * int(vc["temporal_patch_size"]) * int(vc["patch_size"]) ** 2
    gen = torch.Generator().manual_seed(IMAGE_SEED)
    pixels = torch.randn((t * h * w, elements), generator=gen).to(torch.bfloat16).contiguous()
    # mRoPE (D11): the image advances max(t, h/merge, w/merge) positions instead of its pad rows.
    rope_delta = max(t, h // merge, w // merge) - pads
    return ids, IMAGE_AT + 1, pixels, rope_delta


def prefill_images(session: int, tokens: list[int], token_offset: int, pixels: torch.Tensor,
                   max_new: int) -> tuple[torch.Tensor, int]:
    """One greedy PREFILL work item with the image (NO_SPECULATION): (score rows of the committed tokens, finish)."""
    buf, (image,) = oracle.struct_buffer(ImageInput)
    image.struct_size = ctypes.sizeof(ImageInput)
    image.abi_version = oracle.ABI_VERSION
    image.grid_t, image.grid_h, image.grid_w = IMAGE_GRID
    image.pad_id = IMAGE_PAD_ID
    image.token_offset = token_offset
    image.pixel_patches = pixels.data_ptr()
    image.pixel_bytes = pixels.numel() * pixels.element_size()
    image.pixel_sha256[:] = hashlib.sha256(ctypes.string_at(image.pixel_patches, image.pixel_bytes)).digest()
    ids = torch.tensor(tokens, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_session_prefill_images", session, ids, len(tokens), buf, 1, max_new, scores, counts)
    committed, finish = int(counts[0].item()), int(counts[1].item())
    return scores[:committed].clone(), finish


class K15Image(unittest.TestCase):
    """tasks/m213-spec.md C14 (ledger CR14, kind 11) on the real A4 vision model: snapshot -> restore -> DECODE of an
    image session gives an equal state digest (which covers kind 11) and bit-identical greedy scores with rope_delta
    restored; the same image with kind 11 zeroed restores (0 is a valid rope_delta) but to another digest and other
    scores, so the restored rope_delta is what the DECODE reads."""

    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.ids, cls.offset, cls.pixels, cls.rope_delta = image_prompt()
        # One CERESNP1 image of the prompt position (kinds 0-7 and 11) at a time, plus slack.
        cls.capacity = 2 * image_bytes(cls.tc, IMAGE_CONTEXT, IMAGE_CONTEXT, rope_delta=True)
        cls.model = oracle.create_model(IMAGE_CONTEXT, max_sessions=2, snapshot_host_bytes=cls.capacity, vision=True)

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    def setUp(self):
        self.handles: list[int] = []

    def tearDown(self):
        for handle in reversed(self.handles):
            close(handle)

    def open(self) -> int:
        session = session_open(self.model, IMAGE_CONTEXT)
        self.handles.append(session)
        return session

    def drop(self, handle: int) -> None:
        self.handles.remove(handle)
        close(handle)

    def test_image_session_round_trip(self):
        position = len(self.ids)
        source = self.open()
        first, _ = prefill_images(source, self.ids, self.offset, self.pixels, 1)
        oracle.check_count(self, ROW, "image prefill committed", first.shape[0], expected=1)
        source_digest = digest(source)
        _, length, image = snapshot(source, keep=False, capacity=self.capacity)
        oracle.check_count(self, ROW, "image session bytes", length,
                           expected=image_bytes(self.tc, position, position + 1, rope_delta=True))
        oracle.check_count(self, ROW, "image session sections", _get(image, AT_SECTION_COUNT, 4), expected=10)
        last = HEADER_BYTES + ENTRY_BYTES * 9
        oracle.check_count(self, ROW, "kind 11 entry", _get(image, last, 4), expected=KIND_ROPE_DELTA)
        oracle.check_count(self, ROW, "kind 11 bytes", _get(image, last + 16, 8), expected=8)
        at = _get(image, last + 8, 8)
        stored = int.from_bytes(bytes(image[at:at + 8].tolist()), "little", signed=True)
        oracle.check_count(self, ROW, "kind 11 rope_delta", stored, expected=self.rope_delta)
        uninterrupted, _ = ticket(source, None, IMAGE_STEPS)
        uninterrupted_digest = digest(source)
        self.drop(source)

        copy = self.open()
        oracle.check_count(self, ROW, "image restore status", restore_bytes_status(copy, image), expected=0)
        oracle.check_bit_exact(self, ROW, "image digest after restore", digest(copy), source_digest)
        restored, _ = ticket(copy, None, IMAGE_STEPS)
        oracle.check_bit_exact(self, ROW, "image restore scores", restored, uninterrupted)
        oracle.check_bit_exact(self, ROW, "image digest after decode", digest(copy), uninterrupted_digest)
        self.drop(copy)

        # Control: kind 11 zeroed (rope_delta 0, inside (-position, 0]) restores to another state.
        control = self.open()
        oracle.check_count(self, ROW, "zeroed kind 11 restore status",
                           restore_bytes_status(control, _put(image, at, 0, 8)), expected=0)
        control_digest = digest(control)
        moved, _ = ticket(control, None, IMAGE_STEPS)
        digest_differs = not torch.equal(control_digest, source_digest)
        scores_differ = moved.shape != uninterrupted.shape or not torch.equal(moved, uninterrupted)
        oracle.report_row(ROW, "zeroed kind 11 control", phase="check", digest_differs=digest_differs,
                          scores_differ=scores_differ, rope_delta=self.rope_delta,
                          verdict="pass" if digest_differs and scores_differ else "fail")
        self.assertTrue(digest_differs, "a zeroed kind 11 gives the source's digest")
        self.assertTrue(scores_differ, "a zeroed kind 11 decodes the source's scores: rope_delta is not read")
        self.drop(control)


def mem_available() -> int:
    """MemAvailable of /proc/meminfo, bytes."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise oracle.OracleError("/proc/meminfo has no MemAvailable")


@contextlib.contextmanager
def merge_base_library():
    """Routes oracle.ktest to the merge-base libcere_ktest.so for the duration (as test_model_k8_forward.py); yields
    its SHA-256."""
    if not BASE_KTEST_LIB.is_file():
        raise oracle.OracleError(f"{BASE_KTEST_LIB} is absent (the merge-base build)")
    if BASE_KTEST_LIB.resolve() == oracle.KTEST_LIB.resolve():
        raise oracle.OracleError(f"{BASE_KTEST_LIB} is this build's library")
    with BASE_KTEST_LIB.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    lib = ctypes.CDLL(str(BASE_KTEST_LIB), mode=ctypes.RTLD_LOCAL)
    current = oracle.load_ktest
    oracle.load_ktest = lambda: lib
    try:
        yield sha
    finally:
        oracle.load_ktest = current


def warm_library() -> int:
    """`cere_ktest_model_create` with a 1-byte device budget: the executor is created (the library's modules, its
    cuBLASLt handle and the bounce ring) and destroyed again on the model's CAPACITY, so the measured ALLOCATE
    below holds no one-time library load. The raw status."""
    return create(0, 1)[0]


def p3b_scratch_delta() -> int:
    """Bytes by which this build's device plan exceeds the K7 merge-base's at MAX_CONTEXT: 0 on a K7 build; on a
    `CERE_MOE_BACKEND=p3b` build (the shipped default since M23-D7) the P3b - K7 shared scratch of
    `cere_ktest_p3b_arena_bytes` (tasks/review-m23.md "P3b arena delta"), whose weight delta must be 0."""
    if not hasattr(oracle.load_ktest(), "cere_ktest_p3b_arena_bytes"):
        return 0
    k8 = importlib.import_module("test_model_k8_forward")
    entries, desc_buf, _names = k8.a4_tensor_table()
    out = torch.zeros(6, dtype=torch.int64)
    oracle.call("cere_ktest_p3b_arena_bytes", desc_buf, len(entries), k8.a4_config(oracle.text_config()), MAX_CONTEXT,
                out)
    weights_p3b, weights_k7, _ws_p3b, _ws_k7, scratch_p3b, scratch_k7 = (int(v) for v in out.tolist())
    if weights_p3b != weights_k7:
        raise oracle.OracleError(f"P3b weight bytes {weights_p3b} != K7 {weights_k7}")
    return scratch_p3b - scratch_k7


def vm_locked() -> int:
    """VmLck of /proc/self/status (this process's mlock'd bytes), bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1]) * 1024
    raise oracle.OracleError("/proc/self/status has no VmLck")


def create(arena_bytes: int, budget: int) -> tuple[int, int]:
    """`cere_ktest_model_create` (ALLOCATE, no upload) of the K15 model desc with `arena_bytes` and a device budget
    of `budget` bytes: (raw status, handle; 0 unless the status is 0)."""
    model_buf, _, keep = oracle.a4_model_desc(MAX_CONTEXT, max_sessions=SESSIONS, kv_pool_tokens=POOL_TOKENS,
                                              snapshot_host_bytes=arena_bytes)
    out = torch.zeros(1, dtype=torch.int64)
    fn = oracle.ktest("cere_ktest_model_create")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    rc = fn(ctypes.c_void_p(model_buf.data_ptr()), ctypes.c_uint64(budget), ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(rc), int(out[0])


def allocation(arena_bytes: int, *, budget: int | None = None, ledger: bool) -> dict:
    """One ALLOCATE of the K15 model desc with `arena_bytes` (device budget `budget`, default the free device memory)
    and, across it, this process's VmLck growth (`locked_taken`) and, recorded only, GPU0 bytes taken
    (cudaMemGetInfo) and host bytes taken (MemAvailable); with `ledger` (this build) also the model's MemoryEstimate
    and the executor's memory ledger (cere_ktest_model_ledger) while the model lives. Released before return."""
    torch.cuda.synchronize(oracle.DEVICE)
    free_before, _ = torch.cuda.mem_get_info(oracle.DEVICE)
    avail_before = mem_available()
    locked_before = vm_locked()
    rc, handle = create(arena_bytes, free_before if budget is None else budget)
    if rc != 0 or handle == 0:
        raise oracle.OracleError(f"cere_ktest_model_create returned status {rc}, handle {handle}")
    try:
        free_after, _ = torch.cuda.mem_get_info(oracle.DEVICE)
        avail_after = mem_available()
        out = {"device_taken": free_before - free_after, "host_taken": avail_before - avail_after,
               "locked_taken": vm_locked() - locked_before}
        if ledger:
            memory = torch.zeros(2, dtype=torch.int64)
            oracle.call("cere_ktest_model_memory", handle, memory)
            out["device_estimate"], out["host_estimate"] = (int(v) for v in memory.tolist())
            held = torch.zeros(4, dtype=torch.int64)
            oracle.call("cere_ktest_model_ledger", handle, held)
            out["ledger_device"], out["ledger_ring"], out["ledger_mapped"], out["ledger_locked"] = (
                int(v) for v in held.tolist())
    finally:
        oracle.release_model(handle)
    return out


class K15Memory(unittest.TestCase):
    """R1 [REVISED 2026-10-11, tasks/m211-spec.md]: ALLOCATE takes the whole memory plan and nothing later
    (cuda_device.h), so the bytes it takes are the model's peaks. Per library, after a warm-up, the K15 desc is
    allocated alone (no other model alive): this build without and with ARENA_BYTES, then the merge-base library
    without an arena. Gated, exact (± 0 B):

    * device: this build's ledger device bytes (its only device allocation, the ALLOCATE cudaMalloc) equal its device
      estimate and are identical at arena 0 and ARENA_BYTES; the merge-base library, whose ALLOCATE likewise takes
      exactly its device estimate in one cudaMalloc and has no ledger, refuses the desc at a device budget of those
      bytes less `p3b_scratch_delta` (0 on a K7 build; the P3b - K7 shared scratch on a P3b build against the K7
      merge-base, M23-D7) - 1 (CAPACITY) and takes it at that budget, so its device estimate is that number;
    * host: VmLck grows by exactly the ledger's locked bytes (the arena's whole pages: ARENA_BYTES with the arena,
      0 without, 0 on the merge-base); the ledger's ring is RING_BYTES and its mapped block is the same at both arena
      sizes; mapped + locked equal the host estimate, which grows by exactly the arena.

    Recorded, not gated: cudaMemGetInfo and MemAvailable deltas (process-external; they move by up to ±0.85 MiB and
    miss a 1 GiB band with an unchanged library, bench/reports/native-k15-memory-diag-20261011T084519Z.json)."""

    def test_memory_vs_merge_base(self):
        oracle.check_count(self, ROW, "warm-up status", warm_library(), expected=CERE_ERR_CAPACITY)
        plain = allocation(0, ledger=True)
        arena = allocation(ARENA_BYTES, ledger=True)
        device = plain["ledger_device"]
        moe_delta = p3b_scratch_delta()
        base_device = device - moe_delta
        with merge_base_library() as base_sha:
            oracle.check_count(self, ROW, "merge-base warm-up status", warm_library(), expected=CERE_ERR_CAPACITY)
            under, handle = create(0, base_device - 1)
            if handle != 0:
                oracle.release_model(handle)
            base = allocation(0, budget=base_device, ledger=False)
        oracle.report_row(ROW, "memory P3b - K7 scratch", phase="diag", p3b_scratch_delta=moe_delta,
                          merge_base_device=base_device)
        for name, a in (("merge-base", base), ("arena 0", plain), (f"arena {ARENA_BYTES}", arena)):
            oracle.report_row(ROW, f"memory {name}", phase="diag", merge_base_library=base_sha, ring_bytes=RING_BYTES,
                              **a)
        # Device, exact.
        oracle.check_count(self, ROW, "ledger device: arena 0 vs device estimate", device,
                           expected=plain["device_estimate"])
        oracle.check_count(self, ROW, "ledger device: arena vs arena 0", arena["ledger_device"], expected=device)
        oracle.check_count(self, ROW, "device estimate: arena vs arena 0", arena["device_estimate"],
                           expected=plain["device_estimate"])
        oracle.check_count(self, ROW, "merge-base at device budget ledger - P3b scratch delta - 1: status", under,
                           expected=CERE_ERR_CAPACITY)
        # Host, exact.
        for name, a, locked in (("arena 0", plain, 0), (f"arena {ARENA_BYTES}", arena, ARENA_BYTES)):
            oracle.check_count(self, ROW, f"ledger locked {name}", a["ledger_locked"], expected=locked)
            oracle.check_count(self, ROW, f"VmLck growth {name} vs ledger locked", a["locked_taken"],
                               expected=a["ledger_locked"])
            oracle.check_count(self, ROW, f"ledger ring {name}", a["ledger_ring"], expected=RING_BYTES)
            oracle.check_count(self, ROW, f"ledger mapped + locked {name} vs host estimate",
                               a["ledger_mapped"] + a["ledger_locked"], expected=a["host_estimate"])
        oracle.check_count(self, ROW, "VmLck growth merge-base", base["locked_taken"], expected=0)
        oracle.check_count(self, ROW, "ledger mapped: arena vs arena 0", arena["ledger_mapped"],
                           expected=plain["ledger_mapped"])
        oracle.check_count(self, ROW, "host estimate: arena vs arena 0 + arena", arena["host_estimate"],
                           expected=plain["host_estimate"] + ARENA_BYTES)


if __name__ == "__main__":
    unittest.main()
