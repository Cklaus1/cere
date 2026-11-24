"""K11 (tasks/m25-spec.md D9, C5): the paged KV pool on the real A4 model, GPU0, container only.

Suite `model` (`native_dev.py test --suite model`). Models are created, uploaded and sealed through
`oracle.create_model` (ABI v5 `cere_model_desc` from `/model` read-only, families placement, concurrent puts, seal)
with the context, session count and `kv_pool_tokens` each case names (0 = the derived pool, D4). Two models never
coexist on GPU0: each class owns one model at a time and releases it before the next class starts. The prompts are
synthetic and deterministic: the chat template (`enable_thinking` false) around a generated list of numbered records
and one question, cut in token space to the exact length (no randomness, no seed). Token IDs and text stay in memory;
traces are kept as per-name SHA-256 digests plus the top-2 score rows.

Every comparison here is native-vs-native and bit-exact; none is a SGLang-parity or G1 result.

Cases (D9; the clean-pool reference is the first session of K11Paged's 200,000-token model, taken in setUpClass
before any other session writes a page):
* `test_permuted_pages_bit_identical` (1): the 9,000-token prompt with reversed and with interleaved pages (armed by
  `cere_ktest_session_set_pages` for the next session create) gives `forward_trace` (last 64 rows, B8 names) and 32
  greedy tokens bit-identical to the identity-page reference.
* `test_paged_equals_merge_base` (2, I12): the same prompt's `forward_trace` on this build at a 73,728-token context
  (paged, derived pool) equals the merge-base library's (`CERE_BASE_KTEST_LIB`, default
  `build/native-l0/libcere_ktest.so` under `/src`, the m22p L0 convention) at the same context, digests and scores.
  [INFERENCE: placement does not enter the arithmetic.]
* `test_pool_capacity_at_create` (3, D3): desc max_sessions 2, context 9,088 (142 pages), pool 12,288 (192 pages):
  create A OK, create B CAPACITY with nothing taken, release A, create B OK; the free pages after each step.
* `test_dirty_pool_bit_identical` (6, D3 stale-page rule): a 12,000-token session of another prompt writes pages
  first, then the 9,000-token prompt gives `forward_trace` and 32 greedy tokens bit-identical to the reference.
* `test_device_use_at_200000` (4): a 199,999-token prompt plus one generated token fills the 200,000-token session;
  the executor's device bytes in use at that point are at most 80 GiB. The plan bytes are recorded with it.
* `test_long_prefill_repeat_bit_identical` (5): 2 repeats of a 190,000-token prefill give bit-identical
  `forward_trace` digests and scores.

Native entry points (K8's `cere_ktest_model_*`, `cere_ktest_generate`, `cere_ktest_forward_trace` and
`cere_ktest_model_plan_bytes` as stated in `test_model_k10_long.py`; `oracle.call` appends the stream; 0 = success,
otherwise a `cere_status`; "host" buffers are CPU memory). K11's own (ledger R, M2.5 row, as amended by the owner
2026-10-10: no ktest session handle before M2.11):

    int cere_ktest_session_set_pages(uint64_t model, const uint32_t *order /*host [n]*/, uint64_t n,
                                     cudaStream_t /*stream*/);
        Arms a one-shot page order for the model's next session create: with `reserved` the n = ceil(context / 64)
        lowest free pages in ascending order, the session's page i becomes reserved[order[i]]. The next create
        consumes it whatever its outcome; model release drops it. CERE_ERR_ARGUMENT unless `order` is a permutation
        of [0, n).
    int cere_ktest_session_pool_check(uint64_t model, int64_t *out /*host [7]*/, cudaStream_t /*stream*/);
        With every session at the model's context: out = {free pages before, create A, free after A, create B while
        A lives, free after B's attempt, create B after A's release, free after it} (statuses as cere_status). The
        sessions are released before return.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import hashlib
import os
import unittest
from pathlib import Path
from typing import NamedTuple

import torch
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

import oracle

ROW = "K11"
SRC = Path(__file__).resolve().parents[3]
BASE_KTEST_LIB = Path(os.environ.get("CERE_BASE_KTEST_LIB", str(SRC / "build/native-l0/libcere_ktest.so")))
PAGE_TOKENS = 64  # D2
MAX_CONTEXT = 200_000  # D6
BASE_CONTEXT = 73_728  # the merge-base support limit (m22c D6)
PROMPT_TOKENS = 9_000
DIRTY_TOKENS = 12_000  # the prior session's prompt: longer than PROMPT_TOKENS, so stale rows lie past it
DIRTY_FIRST_RECORD = 50_000  # its records start here, so its text differs from the reference prompt's
LONG_PROMPT_TOKENS = 190_000
STEPS = 32
REPEATS = 2
TRACE_ROWS = 64  # one page of last positions; every one of them reads the whole context's KV
CHECK_SESSIONS = 2
CHECK_CONTEXT = 9_088  # 142 pages
CHECK_POOL = 12_288  # 192 pages: one session fits, two do not
DEVICE_BOUND = 80 << 30
FINISH_LENGTH = 1
FINISH_STOP_TOKEN = 2
CERE_OK = 0
CERE_ERR_CAPACITY = -3
SENTINEL = "<<RECORDS>>"
QUESTION = "\n\nWhich record lists the heaviest animal? Answer with its record number only."
COLORS = ("red", "green", "blue", "amber", "violet", "silver", "black")
ANIMALS = ("otter", "heron", "lynx", "badger", "falcon", "marten", "ibex", "crane", "viper", "bison", "gecko")


# -- prompts -----------------------------------------------------------------------------------------------------


@functools.cache
def tokenizer():
    return get_tokenizer(str(oracle.MODEL_DIR))


def synthetic_ids(total: int, first_record: int = 0) -> list[int]:
    """Exactly `total` prompt IDs: the chat template around numbered records (cut in token space) and a question."""
    tok = tokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": SENTINEL + QUESTION}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    if text.count(SENTINEL) != 1:
        raise oracle.OracleError("the rendered chat template does not hold the record sentinel exactly once")
    head, tail = text.split(SENTINEL)
    head_ids = list(tok.encode(head, add_special_tokens=False))
    tail_ids = list(tok.encode(tail, add_special_tokens=False))
    need = total - len(head_ids) - len(tail_ids)
    records = "\n".join(f"Record {i:06d}: the {COLORS[i % len(COLORS)]} {ANIMALS[i % len(ANIMALS)]} weighs "
                        f"{(i * 37) % 101 + 1} kg." for i in range(first_record, first_record + max(need // 8, 1)))
    body_ids = list(tok.encode(records, add_special_tokens=False))
    if need <= 0 or len(body_ids) < need:
        raise oracle.OracleError(f"synthetic prompt of {total} tokens: {len(body_ids)} record tokens for {need}")
    return head_ids + body_ids[:need] + tail_ids


# -- runtime entry points ----------------------------------------------------------------------------------------


class Generation(NamedTuple):
    tokens: torch.Tensor  # int32 [committed] (uint32 IDs)
    scores: torch.Tensor  # int32 [committed, 4]: cere_token_score bytes {id, runner_up, logit, runner_up_logit}
    committed: int
    finish: int
    device_bytes: int


def generate(handle: int, ids: list[int], max_new: int) -> Generation:
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((max_new,), -1, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    result = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_generate", handle, prompt, len(ids), max_new, 0, 0, tokens, scores, result)
    committed, finish, device_bytes = (int(v) for v in result.tolist())
    if not 0 <= committed <= max_new:
        raise oracle.OracleError(f"committed_tokens {committed} outside [0, {max_new}]")
    return Generation(tokens[:committed].clone(), scores[:committed].clone(), committed, finish, device_bytes)


class Trace(NamedTuple):
    digests: dict  # B8 name -> SHA-256 of its BF16 [rows, cols] values
    scores: torch.Tensor  # int32 [rows, 4] (cere_token_score bytes), host


def b8_names(tc: dict) -> list[str]:
    """The B8 tensor names: per layer N the attention output projection, the routed experts and the shared expert's
    down projection, then the final `model` output."""
    names = []
    for layer, kind in enumerate(tc["layer_types"]):
        attn = "self_attn.o_proj" if kind == "full_attention" else "linear_attn.out_proj"
        names += [f"model.layers.{layer}.{attn}", f"model.layers.{layer}.mlp.experts",
                  f"model.layers.{layer}.mlp.shared_expert.down_proj"]
    names.append("model")
    return names


def forward_trace(handle: int, ids: list[int], tc: dict) -> Trace:
    """`cere_ktest_forward_trace` of the last TRACE_ROWS positions (the executor's own prefill chunking), as per-name
    digests; every B8 name must be present and every score row written."""
    hidden, layers = int(tc["hidden_size"]), int(tc["num_hidden_layers"])
    capacity = 4 * layers
    prompt = torch.tensor(ids, dtype=torch.int32)
    names = torch.zeros(capacity * 128, dtype=torch.uint8)
    shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
    values = oracle.empty((capacity * TRACE_ROWS * hidden,), torch.bfloat16)
    scores = torch.full((TRACE_ROWS, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_trace", handle, prompt, len(ids), 0, TRACE_ROWS, names, names.numel(), shapes,
                capacity, values, values.numel() * values.element_size(), scores, counts)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 <= entries <= capacity and 0 <= name_bytes <= names.numel()
            and 0 <= value_bytes <= values.numel() * values.element_size()):
        raise oracle.OracleError(f"forward_trace counts {entries}, {name_bytes}, {value_bytes} exceed capacities")
    emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"forward_trace wrote {len(emitted) - 1} names for {entries} entries")
    digests: dict = {}
    element = 0
    for i in range(entries):
        name = emitted[i].decode()
        ndim, r, c = (int(v) for v in shapes[i].tolist())
        if ndim != 2 or r != TRACE_ROWS or c < 0:
            raise oracle.OracleError(f"forward_trace entry {name}: shape record {(ndim, r, c)}")
        if name in digests:
            raise oracle.OracleError(f"forward_trace emitted {name} twice")
        block = values[element:element + r * c]
        digests[name] = hashlib.sha256(block.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        element += r * c
    if element * values.element_size() != value_bytes:
        raise oracle.OracleError(f"forward_trace value bytes {value_bytes} != the entries' {element * 2}")
    missing = [n for n in b8_names(tc) if n not in digests]
    if missing:
        raise oracle.OracleError(f"forward_trace lacks {missing}")
    if int((scores[:, 0] < 0).sum().item()) != 0:
        raise oracle.OracleError("forward_trace left score rows unwritten")
    return Trace(digests, scores)


def arm_pages(handle: int, order: list[int]) -> None:
    """Arms the page order of the model's next session create (one-shot)."""
    oracle.call("cere_ktest_session_set_pages", handle, torch.tensor(order, dtype=torch.int32), len(order))


def page_orders(context: int) -> dict[str, list[int]]:
    """The permutations of the n = ceil(context / 64) reserved pages: reversed, and interleaved (the even logical
    pages take the lower half, the odd ones the upper half)."""
    n = -(-context // PAGE_TOKENS)
    half = -(-n // 2)
    return {"reversed": list(range(n - 1, -1, -1)),
            "interleaved": [(i % 2) * half + i // 2 for i in range(n)]}


def plan_bytes(handle: int) -> list[int]:
    out = torch.zeros(5, dtype=torch.int64)
    oracle.call("cere_ktest_model_plan_bytes", handle, out)
    return [int(v) for v in out.tolist()]


def release(handle: int) -> None:
    oracle.release_model(handle)
    torch.cuda.synchronize(oracle.DEVICE)
    torch.cuda.empty_cache()


class _Checks(unittest.TestCase):
    def check_trace(self, case: str, trace: Trace, reference: Trace) -> None:
        self.assertEqual(sorted(trace.digests), sorted(reference.digests), f"{ROW} {case}: traced names differ")
        differing = sum(trace.digests[n] != reference.digests[n] for n in reference.digests)
        oracle.check_count(self, ROW, f"{case} trace names differing", differing, expected=0)
        oracle.check_bit_exact(self, ROW, f"{case} trace scores", trace.scores, reference.scores)

    def check_greedy(self, case: str, g: Generation, reference: Generation) -> None:
        self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} {case}: finish {g.finish}")
        self.assertTrue(torch.equal(g.scores[:, 0], g.tokens), f"{ROW} {case}: score id != committed token")
        oracle.check_count(self, ROW, f"{case} committed", g.committed, expected=reference.committed)
        oracle.check_bit_exact(self, ROW, f"{case} tokens", g.tokens, reference.tokens)
        oracle.check_bit_exact(self, ROW, f"{case} scores", g.scores, reference.scores)


class K11Paged(_Checks):
    """One 200,000-token session over the derived pool (3,125 pages); the clean-pool reference first."""

    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.prompt = synthetic_ids(PROMPT_TOKENS)
        cls.handle = oracle.create_model(MAX_CONTEXT)
        cls.addClassCleanup(release, cls.handle)
        cls.reference_trace = forward_trace(cls.handle, cls.prompt, cls.tc)
        cls.reference = generate(cls.handle, cls.prompt, STEPS)
        if cls.reference.finish not in (FINISH_LENGTH, FINISH_STOP_TOKEN) or cls.reference.committed < 1:
            raise oracle.OracleError(f"reference run: finish {cls.reference.finish}, {cls.reference.committed} tokens")

    def dirty_pool(self) -> None:
        """A session of another, longer prompt writes the lowest free pages (those the next session reserves)."""
        dirty = generate(self.handle, synthetic_ids(DIRTY_TOKENS, DIRTY_FIRST_RECORD), 1)
        self.assertIn(dirty.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} dirty run: finish {dirty.finish}")

    def test_permuted_pages_bit_identical(self):
        for label, order in page_orders(MAX_CONTEXT).items():
            with self.subTest(order=label):
                arm_pages(self.handle, order)
                self.check_trace(f"9000 {label} pages", forward_trace(self.handle, self.prompt, self.tc),
                                 self.reference_trace)
                arm_pages(self.handle, order)
                self.check_greedy(f"9000 {label} pages", generate(self.handle, self.prompt, STEPS), self.reference)

    def test_dirty_pool_bit_identical(self):
        self.dirty_pool()
        self.check_trace("9000 dirty pool", forward_trace(self.handle, self.prompt, self.tc), self.reference_trace)
        self.dirty_pool()
        self.check_greedy("9000 dirty pool", generate(self.handle, self.prompt, STEPS), self.reference)

    def test_device_use_at_200000(self):
        rows, workspace, dense_workspace, derived, session = plan_bytes(self.handle)
        ids = synthetic_ids(MAX_CONTEXT - 1)
        g = generate(self.handle, ids, 1)
        self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} 200000: finish {g.finish}")
        oracle.check_count(self, ROW, "200000 committed", g.committed, expected=1)
        oracle.report_row(ROW, "device use 200000", context=len(ids) + g.committed, device_bytes=g.device_bytes,
                          bound=DEVICE_BOUND, workspace_rows=rows, workspace_bytes=workspace,
                          dense_workspace_bytes=dense_workspace, derived_bytes=derived, session_bytes=session,
                          passed=0 < g.device_bytes <= DEVICE_BOUND)
        self.assertTrue(0 < g.device_bytes <= DEVICE_BOUND,
                        f"{ROW} device use {g.device_bytes} bytes at 200,000 tokens exceeds {DEVICE_BOUND}")

    def test_long_prefill_repeat_bit_identical(self):
        ids = synthetic_ids(LONG_PROMPT_TOKENS)
        runs = [forward_trace(self.handle, ids, self.tc) for _ in range(REPEATS)]
        for k in range(1, REPEATS):
            self.check_trace(f"190000 run {k}", runs[k], runs[0])


@contextlib.contextmanager
def merge_base_library():
    """Routes oracle.ktest to the merge-base libcere_ktest.so for the duration; yields its SHA-256."""
    if not BASE_KTEST_LIB.is_file():
        raise oracle.OracleError(f"{BASE_KTEST_LIB} is absent (the merge-base build, tasks/m25-spec.md D9 (2), I12)")
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


def trace_at(ids: list[int], tc: dict) -> Trace:
    """The prompt's trace on a fresh BASE_CONTEXT model of the current library, released before return."""
    handle = oracle.create_model(BASE_CONTEXT)
    try:
        return forward_trace(handle, ids, tc)
    finally:
        release(handle)


class K11MergeBase(_Checks):
    def test_paged_equals_merge_base(self):
        tc = oracle.text_config()
        ids = synthetic_ids(PROMPT_TOKENS)
        with merge_base_library() as sha:
            base = trace_at(ids, tc)
        paged = trace_at(ids, tc)
        oracle.report_row(ROW, "merge-base library", sha256=sha, context=BASE_CONTEXT)
        self.check_trace("9000 paged vs merge-base", paged, base)


class K11PoolCapacity(_Checks):
    def test_pool_capacity_at_create(self):
        handle = oracle.create_model(CHECK_CONTEXT, max_sessions=CHECK_SESSIONS, kv_pool_tokens=CHECK_POOL)
        try:
            out = torch.zeros(7, dtype=torch.int64)
            oracle.call("cere_ktest_session_pool_check", handle, out)
        finally:
            release(handle)
        free0, a, free_a, b, free_b, retry, free_retry = (int(v) for v in out.tolist())
        pool_pages, session_pages = CHECK_POOL // PAGE_TOKENS, -(-CHECK_CONTEXT // PAGE_TOKENS)
        oracle.check_count(self, ROW, "pool pages before", free0, expected=pool_pages)
        oracle.check_count(self, ROW, "create A", a, expected=CERE_OK)
        oracle.check_count(self, ROW, "free after A", free_a, expected=pool_pages - session_pages)
        oracle.check_count(self, ROW, "create B while A lives", b, expected=CERE_ERR_CAPACITY)
        oracle.check_count(self, ROW, "free after the B refusal", free_b, expected=pool_pages - session_pages)
        oracle.check_count(self, ROW, "create B after the A release", retry, expected=CERE_OK)
        oracle.check_count(self, ROW, "free after B", free_retry, expected=pool_pages - session_pages)


if __name__ == "__main__":
    unittest.main()
