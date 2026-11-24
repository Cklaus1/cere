"""VL43 (tasks/m213l-spec.md VL-L2..VL-L7): native image input at the -v43 shape on the real A4 vision model, GPU0,
container only.

Suite `model_vl43` (`native_dev.py test --suite model_vl43`; `--suite model` runs it too). One model (`VL43Long`):
the build max context 200,000 (-v43's context, C3) over the shipped 200,000-token KV pool, four sessions (rows 1..4),
STATE_SLOTS prefix state slots, the vision tower (`oracle.create_model` vision=True, M2.13 C7) and the MTP draft layer
with the served draft token map (test_model_k16_prefix.DRAFT_MAP, mounted by native_dev.py: DRAFT_MAP_MODULES), at
-v29's device levers (configs/runtime/native-g0-spec.toml [deployments.native.levers]: k2_gemm small_m, decode_step
spec_graph, verify_attention batched; test_model_v22_s3.V22_LEVERS). Every speculating ticket carries -v29's runtime
flags as cered sets them (drafter mtp+lookup, draft_policy adaptive: LOOKUP_DRAFTS | ADAPTIVE_DRAFTS,
test_model_v22_s3.V22_FLAGS); every plain reference is NO_SPECULATION without a namespace (cache off). -v43 is -v29
plus vision (C3); the checkpoint is /model's (it holds the vision tower), not the S3 artifact.

Prompts are synthetic (test_model_k16_prefix.synthetic_ids records, cut in token space) with image spans inserted in
token space (vision_start, t*h*w/4 image_token_id, vision_end); pixels are seeded BF16 patches in memory; each image's
pixel_sha256 is the SHA-256 of its patch bytes (C1) and its pad_id the sidecar's rule over them (1,000,000 +
SHA-256[:8] mod 2^30). Nothing is read from or written to disk but the model's config and the draft map. Every gate is
native-vs-native on this model, bit for bit (token and every cere_token_score field): speculation vs plain, a cache
hit vs cold, rows together vs alone; prefill calls of other sizes vs one call compare the token ids only (another call
split rounds the carried state differently, test_model_k10_long's chunk sizes). No CPU reference of the vision tower
is involved (K15Image and the vision kernel suites hold those rows).

Grids ((t, h, w) patches of 16 px, 2 x 2 merge): SCREEN_GRID (1, 56, 80), a 1280x900 screenshot as the processor
resizes it (1,280 x 896 px, 1,120 rows) [INFERENCE: smart_resize keeps that size within the processor's max_pixels];
CENSUS_GRID (1, 98, 98), 2,401 rows (the largest image of the traffic census); CAP_GRID (1, 128, 128), 4,096 rows
(16,384 patches, the per-image cap, C2); WIDE_GRID (1, 64, 256), CAP_GRID's patches in another shape; SMALL_GRID
(1, 16, 16), K15Image's 64 rows.

Cases:
* `test_long_prompt_spec_equals_plain` (C2-C4): a 184,453-token prompt in a 200,000 session, LONG_TEXT text tokens
  with LONG_IMAGES images, one every 8,900 text tokens from text token 1,000 (CAP_GRID first, its pad rows straddling
  the first 4,096-row prefill call; CENSUS_GRID tenth; SCREEN_GRID otherwise; 24,417 image rows, rope_delta -23,664),
  so K9's indexer and compressed keys take mRoPE rows (the K9 twins) from 1,001 to 176,751 and every decode, verify and
  MTP row runs at logical + rope_delta: LONG_STEPS greedy tokens with speculation equal the plain ticket's bit for bit,
  drafts verified (accept length reported).
* `test_prefix_keys_hit_and_miss` (C6, VL-L7): a 10,098-token prompt with one CAP_GRID image at rows 3,001-7,096
  (across the 4,096 draft state, before the 8,192 one). Cold drafting (namespace A) == plain and inserts the two draft
  states; hot hits 8,192 (past the image: the resumed rows are text at logical + rope_delta) == plain == cold. Other
  pixels sent with this image's pixel_sha256 hit 8,192 too and commit the hot run's scores (the key is the digest,
  which cered computes, C1). Other pixels with their own digest, and the same pixels in WIDE_GRID, hit nothing (no
  state below the first pad row) and equal their own plain runs. A branch (the prompt's first 8,192 rows, another
  tail) hits 4,096, inside the image (the image is encoded again for the resumed rows, VL-L3), == its plain == its
  cold run in namespace B.
* `test_rows_equal_alone` (C4, VL-L5): four sessions: the prefix case's prompt (rope_delta -4,032), a 7,244-token
  prompt with two SCREEN_GRID images (-2,160), a 6,000-token text prompt (0, on the decode graphs alone) and a
  578-token prompt with one SMALL_GRID image (-56) in a 2,048 context (K6 attention); each is prefilled by a PREFILL
  ticket of one token, then ROWS_STEPS greedy DECODE tokens; all four together (one decode round per poll carries the
  rows: step_rows plain, speculate_rows drafting) equal each alone bit for bit, plain and drafting.
* `test_chunked_prefill_equals_one_call` (VL-L3, C5): a 3,988-token prompt with a SMALL_GRID image (pad rows 101-164)
  and a SCREEN_GRID screenshot (1,267-2,386), prefilled in one call (ONE_CALL rows) and in calls of each CHUNK_SIZES
  rows (the screenshot's rows split across calls, so it is encoded once per call holding its rows; the sizes are not
  multiples of 4, so K9's compressed-key groups begun in one call end in the next, positions3's p3_lead):
  CHUNK_STEPS greedy tokens with speculation equal the plain ticket's bit for bit at every size, and every size's
  token ids equal the one call's.

Native entry points (the contract with the executor; `oracle.call` appends the stream; 0 = success, otherwise a
`cere_status`; "host" buffers are CPU memory; types of `cere_executor.h`). `cere_ktest_session_open`,
`cere_ktest_session_close` and `cere_ktest_state_digest` are stated in `test_model_k15_snapshot.py`;
`cere_ktest_session_set_namespace`, `cere_ktest_prefix_stats` and `cere_ktest_session_prefill_cached_flags` in
`test_model_k16_prefix.py`; the model comes from `oracle.create_model`.

    int cere_ktest_session_prefill_images_flags(uint64_t session, const uint32_t *tokens /*host*/, uint64_t n,
                                                const cere_image_input *images /*host*/, uint64_t image_count,
                                                uint64_t max_new_tokens, uint64_t flags,
                                                uint64_t prefill_chunk_tokens, cere_token_score *scores /*host*/,
                                                uint64_t *out_counts /*host [6]*/, cudaStream_t stream);
        One greedy PREFILL work item with the images and `flags` (u32 cere_work.flags) on an empty session, its prompt
        in calls of prefill_chunk_tokens rows (0 = the executor's own: ONE_CALL); out_counts = {committed, finish,
        cached_tokens, verify_steps, accepted_drafts, verified_drafts}.
    int cere_ktest_session_decode_rows(const uint64_t *sessions /*host*/, uint64_t rows, uint64_t max_new_tokens,
                                       uint64_t flags, cere_token_score *scores /*host [rows][max_new_tokens]*/,
                                       uint64_t *out_counts /*host [rows][4]*/, cudaStream_t stream);
        One greedy DECODE work item per session (each idle after a PREFILL ticket, one model's) with `flags`, all
        submitted in row order before any poll, then polled in row order until every one completes; out_counts row r
        = {committed, finish, verify_steps, accepted_drafts}.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import importlib
import json
import unittest

import torch

import oracle

k15 = importlib.import_module("test_model_k15_snapshot")
k16 = importlib.import_module("test_model_k16_prefix")

ROW = "VL43"
MAX_CONTEXT = 200_000  # the build max and -v43's context (C3)
SESSIONS = 4  # rows 1..4 (C3)
POOL_TOKENS = 200_000  # the shipped profiles' kv_pool_tokens: one build-max session fills it
STATE_SLOTS = 16  # every state the prefix case inserts stays resident (no eviction between its tickets)
LEVERS = {"k2_gemm": 1, "decode_step": 2, "verify_attention": 1}  # -v29's small_m, spec_graph, batched
CERE_WORK_NO_SPECULATION = 0x2  # cere_executor.h
CERE_WORK_LOOKUP_DRAFTS = 0x4
CERE_WORK_ADAPTIVE_DRAFTS = 0x10
SPEC_FLAGS = CERE_WORK_LOOKUP_DRAFTS | CERE_WORK_ADAPTIVE_DRAFTS  # -v29: drafter mtp+lookup, draft_policy adaptive
FINISHES = (1, 2)  # CERE_FINISH_LENGTH, CERE_FINISH_STOP_TOKEN
MM_PAD_SHIFT = 1_000_000  # tokenizer_sidecar.py: pad_id = 1,000,000 + SHA-256[:8] (big-endian) mod 2^30

SCREEN_GRID = (1, 56, 80)
CENSUS_GRID = (1, 98, 98)
CAP_GRID = (1, 128, 128)
WIDE_GRID = (1, 64, 256)
SMALL_GRID = (1, 16, 16)

LONG_TEXT = 160_000
LONG_IMAGES = 18
LONG_AT = tuple(1_000 + 8_900 * i for i in range(LONG_IMAGES))  # text offsets; the last at 152,300
LONG_STEPS = 64
LONG_SEED = 4_300

PREFIX_TEXT = 6_000
PREFIX_AT = 3_000  # the CAP_GRID span starts before text token 3,000: pad rows 3,001 .. 7,096
PREFIX_STATES = (4_096, 8_192)  # a drafting PREFILL's call ends below n (model.h kMaxChunkTokens 4,096)
PREFIX_CONTEXT = 16_384
PREFIX_STEPS = 64
PREFIX_SEED = 4_310
BRANCH_TAIL = 1_200
BRANCH_SKIP = 200  # the variant's leading chat-template rows are skipped (as K16 (9))

ROWS_CONTEXT = 16_384
SMALL_CONTEXT = 2_048  # k6::kMaxContext: K6 attention
ROWS_STEPS = 32
ROWS_SEED = 4_320

ONE_CALL = 4_096  # model.h kMaxChunkTokens: the workspace's rows per prefill call (prefill_chunk_tokens 0)
CHUNK_TEXT = 2_800
CHUNK_SPANS = ((100, SMALL_GRID), (1_200, SCREEN_GRID))  # pad rows 101-164 and 1,267-2,386: 3,988 tokens
CHUNK_SIZES = (1_001, 333)  # not multiples of 4 (K9's compressed-key groups straddle the calls)
CHUNK_STEPS = 32
CHUNK_SEED = 4_330


@functools.cache
def model_config() -> dict:
    return json.loads((oracle.MODEL_DIR / "config.json").read_text())


def merge() -> int:
    return int(model_config()["vision_config"]["spatial_merge_size"])


def image_rows(grid: tuple[int, int, int]) -> int:
    t, h, w = grid
    return t * h * w // (merge() * merge())


def rope_delta(grid: tuple[int, int, int]) -> int:
    """mRoPE (M2.13 D11): the image advances max(t, h/merge, w/merge) positions instead of its pad rows."""
    t, h, w = grid
    return max(t, h // merge(), w // merge()) - image_rows(grid)


def seeded_pixels(grid: tuple[int, int, int], seed: int) -> torch.Tensor:
    """BF16 [t*h*w][patch elements] CPU patches from a seeded generator."""
    vc = model_config()["vision_config"]
    elements = int(vc["in_channels"]) * int(vc["temporal_patch_size"]) * int(vc["patch_size"]) ** 2
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((grid[0] * grid[1] * grid[2], elements), generator=gen).to(torch.bfloat16).contiguous()


class Image:
    """One image as cered sends it: grid, BF16 patches, pixel_sha256 (default: of the patches) and pad_id."""

    def __init__(self, grid: tuple[int, int, int], pixels: torch.Tensor, sha: bytes | None = None):
        self.grid = grid
        self.pixels = pixels
        self.sha = sha if sha is not None else hashlib.sha256(
            ctypes.string_at(pixels.data_ptr(), pixels.numel() * pixels.element_size())).digest()
        self.pad_id = MM_PAD_SHIFT + int.from_bytes(self.sha[:8], "big") % (1 << 30)


def compose(text: list[int], spans: list[tuple[int, Image]]) -> tuple[list[int], list[tuple[int, Image]]]:
    """(prompt ids, [(token_offset, image)]): `text` with each image's span inserted before text[at], `at`
    ascending."""
    config = model_config()
    start, pad, end = (int(config[k]) for k in ("vision_start_token_id", "image_token_id", "vision_end_token_id"))
    ids: list[int] = []
    placed: list[tuple[int, Image]] = []
    last = 0
    for at, image in spans:
        ids += text[last:at]
        ids.append(start)
        placed.append((len(ids), image))
        ids += [pad] * image_rows(image.grid)
        ids.append(end)
        last = at
    ids += text[last:]
    return ids, placed


class Run:
    """A ticket's committed score rows (int32 [committed, 4]), finish, cached_tokens and speculation counters."""

    def __init__(self, scores: torch.Tensor, finish: int, cached: int, verify_steps: int, accepted: int):
        self.scores = scores
        self.finish = finish
        self.cached = cached
        self.verify_steps = verify_steps
        self.accepted = accepted

    def accept_length(self) -> float:
        """Committed tokens per verify step; 0.0 without a verify step."""
        return round(len(self.scores) / self.verify_steps, 4) if self.verify_steps else 0.0


def prefill(session: int, ids: list[int], placed: list[tuple[int, Image]], max_new: int, flags: int, *,
            chunk: int = 0) -> Run:
    """One greedy PREFILL ticket of `ids` with the placed images (none: the text entry, which takes no `chunk`) and
    `flags`, its prompt in calls of `chunk` rows (0: the executor's own)."""
    tokens = torch.tensor(ids, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    counts = torch.zeros(6, dtype=torch.int64)
    if not placed:
        oracle.call("cere_ktest_session_prefill_cached_flags", session, tokens, len(ids), max_new, flags, scores,
                    counts)
    else:
        buf, inputs = oracle.struct_buffer(k15.ImageInput, len(placed))
        for view, (offset, image) in zip(inputs, placed):
            view.struct_size = ctypes.sizeof(k15.ImageInput)
            view.abi_version = oracle.ABI_VERSION
            view.grid_t, view.grid_h, view.grid_w = image.grid
            view.pad_id = image.pad_id
            view.token_offset = offset
            view.pixel_patches = image.pixels.data_ptr()
            view.pixel_bytes = image.pixels.numel() * image.pixels.element_size()
            view.pixel_sha256[:] = image.sha
        oracle.call("cere_ktest_session_prefill_images_flags", session, tokens, len(ids), buf, len(placed), max_new,
                    flags, chunk, scores, counts)
    committed, finish, cached, steps, accepted, _ = (int(v) for v in counts.tolist())
    return Run(scores[:committed].clone(), finish, cached, steps, accepted)


def decode_rows(sessions: list[int], steps: int, flags: int) -> list[Run]:
    """`steps` greedy DECODE tokens of every session at once (one ticket each, polled together)."""
    handles = torch.tensor(sessions, dtype=torch.int64)
    scores = torch.full((len(sessions), steps, 4), -1, dtype=torch.int32)
    counts = torch.zeros((len(sessions), 4), dtype=torch.int64)
    oracle.call("cere_ktest_session_decode_rows", handles, len(sessions), steps, flags, scores, counts)
    out = []
    for r in range(len(sessions)):
        committed, finish, verify_steps, accepted = (int(v) for v in counts[r].tolist())
        out.append(Run(scores[r, :committed].clone(), finish, 0, verify_steps, accepted))
    return out


def long_grid(i: int) -> tuple[int, int, int]:
    return CAP_GRID if i == 0 else CENSUS_GRID if i == 9 else SCREEN_GRID


def rows_prompts() -> list[tuple[list[int], list[tuple[int, Image]], int]]:
    """The rows case's four (prompt, placed images, session context)."""
    prefix = compose(list(k16.synthetic_ids(PREFIX_TEXT)),
                     [(PREFIX_AT, Image(CAP_GRID, seeded_pixels(CAP_GRID, PREFIX_SEED)))])
    screens = compose(list(k16.synthetic_ids(5_000, 3_000)),
                      [(200, Image(SCREEN_GRID, seeded_pixels(SCREEN_GRID, ROWS_SEED))),
                       (2_500, Image(SCREEN_GRID, seeded_pixels(SCREEN_GRID, ROWS_SEED + 1)))])
    text = (list(k16.synthetic_ids(6_000, 6_000)), [])
    small = compose(list(k16.synthetic_ids(512, 9_000)),
                    [(24, Image(SMALL_GRID, seeded_pixels(SMALL_GRID, ROWS_SEED + 2)))])
    return [(*prefix, ROWS_CONTEXT), (*screens, ROWS_CONTEXT), (*text, ROWS_CONTEXT), (*small, SMALL_CONTEXT)]


class VL43Long(unittest.TestCase):
    """VL-L2..VL-L7 on one vision + MTP model at -v29's levers (module docstring)."""

    @classmethod
    def setUpClass(cls):
        with open(k16.DRAFT_MAP, "rb") as f:
            draft_map = torch.frombuffer(bytearray(f.read()), dtype=torch.int32)
        cls.model = oracle.create_model(MAX_CONTEXT, max_sessions=SESSIONS, kv_pool_tokens=POOL_TOKENS,
                                        prefix_state_slots=STATE_SLOTS, draft_token_map=draft_map, vision=True,
                                        model_levers=oracle.levers(**LEVERS))
        oracle.report_row(ROW, "model", phase="setup", max_context=MAX_CONTEXT, sessions=SESSIONS,
                          kv_pool_tokens=POOL_TOKENS, prefix_state_slots=STATE_SLOTS, work_flags=SPEC_FLAGS, **LEVERS)

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    def setUp(self):
        self.handles: list[int] = []

    def tearDown(self):
        for handle in reversed(self.handles):
            k15.close(handle)

    def open(self, context: int, namespace: bytes | None = None) -> int:
        session = k15.session_open(self.model, context)
        self.handles.append(session)
        if namespace is not None:
            k16.set_namespace(session, namespace)
        return session

    def drop(self, handle: int) -> None:
        self.handles.remove(handle)
        k15.close(handle)

    def request(self, ids: list[int], placed: list[tuple[int, Image]], namespace: bytes | None, flags: int, *,
                context: int = PREFIX_CONTEXT, max_new: int = PREFIX_STEPS, chunk: int = 0) -> Run:
        """One PREFILL ticket on a fresh session (namespaced unless `namespace` is None), released after."""
        session = self.open(context, namespace)
        run = prefill(session, ids, placed, max_new, flags, chunk=chunk)
        self.drop(session)
        return run

    def same(self, case: str, run: Run, reference: Run) -> None:
        oracle.check_count(self, ROW, f"{case} committed", len(run.scores), expected=len(reference.scores))
        oracle.check_bit_exact(self, ROW, f"{case} scores", run.scores, reference.scores)

    def test_long_prompt_spec_equals_plain(self):
        text = list(k16.synthetic_ids(LONG_TEXT, 40_000))
        spans = [(at, Image(long_grid(i), seeded_pixels(long_grid(i), LONG_SEED + i))) for i, at in enumerate(LONG_AT)]
        prompt, placed = compose(text, spans)
        pad_rows = sum(image_rows(image.grid) for _, image in placed)
        oracle.check_count(self, ROW, "long prompt tokens", len(prompt),
                           expected=LONG_TEXT + pad_rows + 2 * LONG_IMAGES)
        oracle.check_count(self, ROW, "long prompt fits", int(len(prompt) + LONG_STEPS <= MAX_CONTEXT), expected=1)
        plain = self.request(prompt, placed, None, CERE_WORK_NO_SPECULATION, context=MAX_CONTEXT, max_new=LONG_STEPS)
        spec = self.request(prompt, placed, None, SPEC_FLAGS, context=MAX_CONTEXT, max_new=LONG_STEPS)
        oracle.check_count(self, ROW, "long plain finish", int(plain.finish in FINISHES), expected=1)
        oracle.check_count(self, ROW, "long spec finish", spec.finish, expected=plain.finish)
        self.same("long spec == plain", spec, plain)
        oracle.check_count(self, ROW, "long drafts verified", int(spec.verify_steps > 0), expected=1)
        oracle.report_row(ROW, "long", phase="diag", prompt_tokens=len(prompt), images=len(placed),
                          image_rows=pad_rows, rope_delta=sum(rope_delta(image.grid) for _, image in placed),
                          committed=len(spec.scores), verify_steps=spec.verify_steps, accepted_drafts=spec.accepted,
                          accept_length=spec.accept_length())

    def test_prefix_keys_hit_and_miss(self):
        image = Image(CAP_GRID, seeded_pixels(CAP_GRID, PREFIX_SEED))
        prompt, placed = compose(list(k16.synthetic_ids(PREFIX_TEXT)), [(PREFIX_AT, image)])
        first_pad = placed[0][0]
        last_pad = first_pad + image_rows(CAP_GRID) - 1
        oracle.check_count(self, ROW, "prefix image across the first state",
                           int(first_pad < PREFIX_STATES[0] <= last_pad < PREFIX_STATES[1]), expected=1)
        plain = self.request(prompt, placed, None, CERE_WORK_NO_SPECULATION)
        before = k16.stats(self.model)
        cold = self.request(prompt, placed, k16.NS_A, SPEC_FLAGS)
        seeded = k16.stats(self.model)
        oracle.check_count(self, ROW, "prefix cold cached_tokens", cold.cached, expected=0)
        oracle.check_count(self, ROW, "prefix cold draft states", seeded[k16.INSERTS] - before[k16.INSERTS],
                           expected=len(PREFIX_STATES))
        oracle.check_count(self, ROW, "prefix cold drafts verified", int(cold.verify_steps > 0), expected=1)
        self.same("prefix cold == plain", cold, plain)
        hot = self.request(prompt, placed, k16.NS_A, SPEC_FLAGS)
        oracle.check_count(self, ROW, "prefix hot cached_tokens", hot.cached, expected=PREFIX_STATES[-1])
        oracle.check_count(self, ROW, "prefix hot hits", k16.stats(self.model)[k16.HITS_AT] - seeded[k16.HITS_AT],
                           expected=1)
        self.same("prefix hot == plain", hot, plain)
        self.same("prefix hot == cold", hot, cold)
        oracle.check_count(self, ROW, "prefix hot accepted_drafts", hot.accepted, expected=cold.accepted)

        # The key is pixel_sha256 (C1, C6): other pixels sent under this image's digest hit this image's state.
        other = seeded_pixels(CAP_GRID, PREFIX_SEED + 1)
        lied = self.request(prompt, [(first_pad, Image(CAP_GRID, other, sha=image.sha))], k16.NS_A, SPEC_FLAGS)
        oracle.check_count(self, ROW, "prefix digest is the key cached_tokens", lied.cached, expected=PREFIX_STATES[-1])
        self.same("prefix digest is the key == hot", lied, hot)

        # Other pixels (their own digest) and the same pixels in another grid: the same token ids, no hit.
        for case, other_image in (("other pixels", Image(CAP_GRID, other)),
                                  ("other grid", Image(WIDE_GRID, image.pixels))):
            other_placed = [(first_pad, other_image)]
            other_plain = self.request(prompt, other_placed, None, CERE_WORK_NO_SPECULATION)
            other_hot = self.request(prompt, other_placed, k16.NS_A, SPEC_FLAGS)
            oracle.check_count(self, ROW, f"prefix {case} cached_tokens", other_hot.cached, expected=0)
            self.same(f"prefix {case} == its plain", other_hot, other_plain)
            oracle.report_row(ROW, f"prefix {case}", phase="diag",
                              equals_first_image=bool(torch.equal(other_plain.scores, plain.scores)))

        # A branch past the image: the hit falls back to 4,096, inside the image (encoded again for rows 4,096+).
        branch = prompt[:PREFIX_STATES[-1]] + k16.variant(71, BRANCH_TAIL + BRANCH_SKIP)[BRANCH_SKIP:]
        oracle.check_count(self, ROW, "prefix branch token differs",
                           int(branch[PREFIX_STATES[-1]] != prompt[PREFIX_STATES[-1]]), expected=1)
        branch_plain = self.request(branch, placed, None, CERE_WORK_NO_SPECULATION)
        branch_cold = self.request(branch, placed, k16.NS_B, SPEC_FLAGS)
        branch_hot = self.request(branch, placed, k16.NS_A, SPEC_FLAGS)
        oracle.check_count(self, ROW, "prefix branch cold cached_tokens", branch_cold.cached, expected=0)
        oracle.check_count(self, ROW, "prefix branch cached_tokens", branch_hot.cached, expected=PREFIX_STATES[0])
        self.same("prefix branch hot == plain", branch_hot, branch_plain)
        self.same("prefix branch hot == cold", branch_hot, branch_cold)
        oracle.check_count(self, ROW, "prefix branch accepted_drafts", branch_hot.accepted,
                           expected=branch_cold.accepted)
        oracle.report_row(ROW, "prefix accept length", phase="diag", cold_verify_steps=cold.verify_steps,
                          cold_accepted_drafts=cold.accepted, hot_accept_length=hot.accept_length(),
                          branch_accept_length=branch_hot.accept_length())

    def test_rows_equal_alone(self):
        prompts = rows_prompts()
        for case, flags in (("plain", CERE_WORK_NO_SPECULATION), ("drafting", SPEC_FLAGS)):
            alone = []
            for ids, placed, context in prompts:
                session = self.open(context)
                first = prefill(session, ids, placed, 1, flags)
                alone.append((first, decode_rows([session], ROWS_STEPS, flags)[0]))
                self.drop(session)
            sessions, firsts = [], []
            for ids, placed, context in prompts:
                session = self.open(context)
                sessions.append(session)
                firsts.append(prefill(session, ids, placed, 1, flags))
            together = decode_rows(sessions, ROWS_STEPS, flags)
            for r, ((first, run), first_together, run_together) in enumerate(zip(alone, firsts, together)):
                self.same(f"rows {case} row {r} prefill", first_together, first)
                oracle.check_count(self, ROW, f"rows {case} row {r} finish", int(run.finish in FINISHES), expected=1)
                oracle.check_count(self, ROW, f"rows {case} row {r} together finish", run_together.finish,
                                   expected=run.finish)
                self.same(f"rows {case} row {r} together == alone", run_together, run)
            if flags == SPEC_FLAGS:
                oracle.check_count(self, ROW, f"rows {case} drafts verified",
                                   int(all(run.verify_steps > 0 for run in together)), expected=1)
            oracle.report_row(ROW, f"rows {case}", phase="diag", rows=len(sessions),
                              verify_steps=sum(run.verify_steps for run in together),
                              accepted_drafts=sum(run.accepted for run in together),
                              rope_deltas=",".join(str(sum(rope_delta(image.grid) for _, image in placed))
                                                   for _, placed, _ in prompts))
            for session in reversed(sessions):
                self.drop(session)

    def test_chunked_prefill_equals_one_call(self):
        text = list(k16.synthetic_ids(CHUNK_TEXT, 12_000))
        prompt, placed = compose(text, [(at, Image(grid, seeded_pixels(grid, CHUNK_SEED + i)))
                                        for i, (at, grid) in enumerate(CHUNK_SPANS)])
        first = placed[-1][0]
        last = first + image_rows(SCREEN_GRID) - 1
        oracle.check_count(self, ROW, "chunk prompt in one call", int(len(prompt) <= ONE_CALL), expected=1)
        for chunk in CHUNK_SIZES:
            oracle.check_count(self, ROW, f"chunk {chunk} splits the screenshot", int(first // chunk < last // chunk),
                               expected=1)
        one = self.request(prompt, placed, None, CERE_WORK_NO_SPECULATION, max_new=CHUNK_STEPS)
        oracle.check_count(self, ROW, "chunk one call finish", int(one.finish in FINISHES), expected=1)
        for chunk in (0, *CHUNK_SIZES):
            plain = one if chunk == 0 else self.request(prompt, placed, None, CERE_WORK_NO_SPECULATION,
                                                        max_new=CHUNK_STEPS, chunk=chunk)
            spec = self.request(prompt, placed, None, SPEC_FLAGS, max_new=CHUNK_STEPS, chunk=chunk)
            oracle.check_count(self, ROW, f"chunk {chunk} spec finish", spec.finish, expected=plain.finish)
            self.same(f"chunk {chunk} spec == plain", spec, plain)
            oracle.check_count(self, ROW, f"chunk {chunk} drafts verified", int(spec.verify_steps > 0), expected=1)
            if chunk == 0:
                continue
            oracle.check_count(self, ROW, f"chunk {chunk} finish", plain.finish, expected=one.finish)
            oracle.check_count(self, ROW, f"chunk {chunk} committed", len(plain.scores), expected=len(one.scores))
            oracle.check_bit_exact(self, ROW, f"chunk {chunk} token ids == one call", plain.scores[:, 0].clone(),
                                   one.scores[:, 0].clone())
            logits = (plain.scores[:, 2].clone().view(torch.float32) - one.scores[:, 2].clone().view(torch.float32))
            oracle.report_row(ROW, f"chunk {chunk} vs one call", phase="diag", prompt_tokens=len(prompt),
                              scores_equal=bool(torch.equal(plain.scores, one.scores)),
                              max_abs_logit_diff=float(logits.abs().max()), accept_length=spec.accept_length())


if __name__ == "__main__":
    unittest.main()
