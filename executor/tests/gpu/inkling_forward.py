"""Inkling 2 Mini S5 (tasks/inkling-spec.md section 7, G2-G4): the whole text forward of family inkling2_loop on the
checkpoint, the GPU half of scripts/live/inkling_judge.py. Not a test module (unittest never discovers it): the judge
runs it in the GPU0 dev container with /model = Inkling-2-Mini@b85c3cfa, /out = the inkling-2-mini build, /golden =
S1's host run directory (taps/, golden-v2/Q/), /attn = attn_spread.json's host_dir, /p08 = p08_taps.json's host_dir
(when that fixture exists) and /gfp32 = the FP32-attention golden's run directory (Q/; row tf-fp32ids only), all
read-only, /work read-write.

usage (in the container): python3 inkling_forward.py --out FILE ROW [ROW ...]
  ROW: g2 | g2-softmax | g2-globalscale | g2-nomsg | g3 | g3-softmax | g3-globalscale | g3-nomsg | g4 | perf | device
       | spec | spec-classes | verify | verify-time | sampling | admission | g6-long32k | tf-fp32ids
  env INKLING_SPEC_WINDOW (S9 phase B): perf's device rates with that verify window (default 0, plain); spec's and
       spec-classes' window (default 4). spec-classes reads S9r's prompts at /s9r/prompts.json.

- g2: per tap prompt (p01, p03, p07 of tests/fixtures/inkling2_reference/taps.json), CERE's forward over the prompt's
  recorded sequence: positions < decode_from prefilled (chunks of 4,096), the rest decoded one token per call; per
  visit the relative L2 of CERE's visit output against Q's (taps.rel_l2 of run_taps.py: over all positions, Q the
  reference): decoded positions vs Q's decode taps against bar[v] (taps.json g2.bar), prefilled positions vs Q's
  whole-prefill taps against 2 x max(g2.spread_max[v], Q's chunk-512-vs-whole spread on p07), each bar widened per
  prompt to 1.25x REF's eager-vs-FP32-attention rel-L2 (D-I-G2b, attn_spread.json); and, unwidened prefill bar,
  CERE's whole prompt prefill vs REF arm Q with FP32 attention (/attn). The final block states are reported alongside.
  p08 (image; p08_taps.py, /p08) the same way with the reference tower's rows in its placeholders (G2 judges the text
  forward; CERE's own tower rows are reported beside), its bars additionally at least 2x its own decode spread.
- g2-softmax, g2-globalscale, g2-nomsg (the fail-first mutants, D-I-G3d: each must fail G1 or G2): g2 with every
  forward of the row on softmax routing (MoeMutant::kSoftmaxRouting), on the MoE's NVFP4 activation quantizer with one
  global amax (MoeMutant::kGlobalActScale), or with passes 2-3 run without messages; the row passes when G2 FAILS.
- g3 (and the mutants' g3-* rows; report only, D-I-G3d): per golden prompt (p08 with CERE's vision tower rows, S7),
  against golden v2 arm Q: prefill (chunks of 4,096) argmax agreement over every position vs Q's top-1, KL(Q || CERE)
  of the last 16 rows; decode teacher-forced on Q's greedy ids (24 steps, all prompts in one batch of 8 sessions) KL
  mean and argmax agreement vs Q's cached decode logits; the greedy common prefix; and the text prompts' prefill
  against REF arm Q with FP32 attention (/attn tf_fp32.safetensors, equal to golden_fp32.json's prefill), which the
  judge scores with tf-fp32ids against golden_fp32.json's bars. `pass` / `failed` record the superseded D-I-G3 /
  D-I-G3b bars (decode; prefill: p07 taps.json g3_prefill, the others ref_cpu_spread.json).
- g4: bit-identical logits: two runs (p07 prefill, generate); 1 vs 4 sessions (session 0 of a 4-session teacher-forced
  generate vs alone); prefill chunk 512 vs 4,096 (p07: argmax of every position, last 16 rows, every visit output);
  decode graph replays vs eager decode (p07's first 1,040 ids decoded from 1,016, last 24 rows).
- perf: decode tok/s at 4,096 tokens of context (p07's first 4,096 ids prefilled), reported: c1 on the device (three
  reps of 1 vs 257 tokens on one sealed model) and the forward ktest (65 vs 1), c4 on the forward ktest (4 sessions).
- device: the CUDA device (text prompts; the device refuses images, D-I5) against the forward ktest's greedy ids.
- sampling: the device's sampler (S6g) on p01, p03, p07: temperature 0 with logprobs equals the greedy device, a
  seeded temperature-1 run is bit-identical twice, logprobs well formed (row_sampling).
- admission (S10, tasks/inkling-spec.md section 9.2): the device's KV-byte admission through the ABI
  (cere_ktest_inkling_device_admission): a kv_pool_tokens beyond the budget is CAPACITY at model create; 16 sessions of
  up to 262,144 tokens on a 524,288-token pool load, and session creates fill the pool, get CAPACITY, release and
  refill (ADMIT_OPS), each create's device bytes session_bytes(context) (cere_ktest_inkling_sizes).
- g6-long32k (report only, D-I-G3d; G6's G3 on S1's 32K prompt, /golden/long32k/Q.safetensors): g3's measures on the
  32,753 tokens (prefill argmax over every position, KL16, teacher-forced decode, greedy prefix) and the prefill wall;
  p07's former bars applied for information.
- tf-fp32ids (G3's decode half, report only, D-I-G3d): CERE's teacher-forced decode on the FP32-attention golden's
  own greedy ids (golden_fp32.json, InkG3c's arm Q with FP32 attention, /gfp32/Q), all 8 prompts in one batch (p08
  with CERE's tower rows), per prompt and step KL(golden-FP32 || CERE) of the cached decode logits and both argmaxes;
  the same for the three G3 mutants (softmax routing, one global activation scale, passes 2-3 without messages).

Entry points (libcere_ktest.so, executor/cuda/src/ktest/inkling/{forward,vision}_ktest.cpp):
    int cere_ktest_inkling_forward(const void *tensors, const void *names, const void *meta, int64_t count,
                                   const void *ids /*host int32 [n]*/, int64_t n, int64_t chunk, int64_t decode_from,
                                   int64_t moe_mutant, int64_t no_messages, void *visits /*bf16 [48, n, 2304]*/,
                                   void *blocks /*bf16 [n, 4, 2304]*/, void *argmax /*int32 [n]*/,
                                   void *logits /*f32 [n - logit_first, 200058]*/, int64_t logit_first,
                                   const void *image /*bf16 [image_rows, 2304]*/, int64_t image_rows,
                                   cudaStream_t stream);
    int cere_ktest_inkling_generate(const void *tensors, const void *names, const void *meta, int64_t count,
                                    const void *prompts /*host int32*/, const void *lens /*host int64 [S]*/,
                                    int64_t sessions, int64_t gen, const void *forced /*host int32 [S, gen]*/,
                                    int64_t chunk, int64_t moe_mutant, int64_t no_messages,
                                    void *ids /*host int32 [S, gen]*/, void *logits /*f32 [S, gen, 200058]*/,
                                    const void *image /*bf16 [image_rows, 2304]*/, int64_t image_rows,
                                    cudaStream_t stream);
    int cere_ktest_inkling_verify(const void *tensors, const void *names, const void *meta, int64_t count,
                                  const void *prompts, const void *lens, int64_t sessions, int64_t gen,
                                  const void *forced /*host int32 [S, gen]*/, int64_t chunk, int64_t run,
                                  int64_t seed, int64_t mutant, void *ids /*host int32 [S, gen]*/,
                                  void *logits /*f32 [S, gen, 200058]*/, cudaStream_t stream);
    int cere_ktest_inkling_verify_time(const void *tensors, const void *names, const void *meta, int64_t count,
                                       const void *ids /*host int32 [n]*/, int64_t n, int64_t sessions,
                                       int64_t max_m, int64_t reps, void *out /*host f64 [max_m + 1]*/,
                                       cudaStream_t stream);
    int cere_ktest_inkling_device_generate_draft(const void *tensors, const void *names, const void *meta,
                                                 int64_t count, const void *prompts, const void *lens, int64_t n,
                                                 int64_t gen, const void *gens, int64_t context, int64_t budget,
                                                 int64_t window, int64_t concurrent, void *ids, void *result,
                                                 void *bytes, void *seconds /*f64 [n]*/, void *stats /*[n, 2]*/,
                                                 cudaStream_t stream);
    int cere_ktest_inkling_vision_encode(const void *const *tensors, const void *pixels, const int32_t *grids,
                                         int64_t count, int64_t mutant, void *out, cudaStream_t stream);
    int cere_ktest_inkling_vision_tensor(int64_t i, char *name, int64_t cap, int64_t *dims);
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

import oracle

GOLDEN = Path(os.environ.get("INKLING_GOLDEN", "/golden"))
ATTN = Path(os.environ.get("INKLING_ATTN", "/attn"))  # attn_spread.json host_dir (D-I-G2b REF FP32-attention taps)
P08 = Path(os.environ.get("INKLING_P08", "/p08"))  # p08_taps.json host_dir (p08_taps.py: the image prompt's taps)
GFP32 = Path(os.environ.get("INKLING_GFP32", "/gfp32"))  # the FP32-attention golden run (run_golden_fp32.py, InkG3c)
FIXTURES = Path("/src/tests/fixtures/inkling2_reference")
SIDECAR = Path("/src/scripts/native/tokenizer_sidecar.py")
H, V, VISITS = 2304, 200058, 48
EOS = 200006
IMAGE_TOKEN = 200054  # `<|unused_200054|>`, config image_token_id
VISION_TENSORS = 422
IMAGES = {"p08_image": "p08"}  # golden prompt -> its image (tests/fixtures/inkling2_reference/vision/<name>.png)
CHUNK = 4096
LASTN = 16
GEN = 24
MUT_SOFTMAX, MUT_GLOBAL_SCALE = 1, 2
# cere_ktest_inkling_device_generate's params, in order (cere_sampling_params fields; seed of prompt p = seed + p).
SAMPLING_KEYS = ("temperature", "top_p", "min_p", "top_k", "frequency_penalty", "presence_penalty",
                 "repetition_penalty", "seed")
# cere_token_logprobs in 4-byte words: id, logprob, top_count, reserved, then CERE_MAX_TOP_LOGPROBS (id, logprob).
MAX_TOP_LOGPROBS = 20
LOGPROB_WORDS = 4 + 2 * MAX_TOP_LOGPROBS
# inkling design note D-I-G3 (binding): teacher-forced decode KL mean <= 0.02, argmax >= 22 / 24; prefill bars per
# D-I-G3b (g3_prefill_bars).
G3_DECODE_KL_MEAN = 0.02
G3_DECODE_ARGMAX = 22
CAPACITY = -3  # CERE_ERR_CAPACITY
# Row admission (S10, inkling design note section 9.2): GPU0's executor limit (86,016 MiB), 16 sessions of up to
# 262,144 tokens sharing a 524,288-token pool (the pre-S10 estimate, 16 x session_bytes(262,144) ~ 140 GB, refused
# it), and a 2,883,584-token pool (88 GiB of full-visit KV) the estimate must refuse. Ops: +n creates a session of n
# tokens, -i releases the session op i (1-based) created; ADMIT_WANT = each op's status.
ADMIT = {"budget": 90_194_313_216, "sessions": 16, "context": 262_144, "pool": 524_288, "refused_pool": 2_883_584}
ADMIT_OPS = (262_144, 262_144, 1_024, -1, 1_024, 262_144, -2, 262_144)
ADMIT_WANT = (0, 0, CAPACITY, 0, 0, CAPACITY, 0, 0)
ADMIT_SLACK = 64 << 20  # cudaMemGetInfo noise (other GPU0 processes) and cudaMalloc rounding per op
DTYPES = {torch.bfloat16: 1, torch.float32: 3, torch.float8_e4m3fn: 5, torch.uint8: 7}
DEV = oracle.DEVICE
# S9 phase B (D-I-S9B): speculative_draft_tokens (the verify window: the unfed token + window - 1 drafts) of row
# perf's device rates (0 = the plain path) and of row spec (0 there = SPEC_DEFAULT_WINDOW, the deployment's).
SPEC_WINDOW = int(os.environ.get("INKLING_SPEC_WINDOW", "0"))
SPEC_DEFAULT_WINDOW = 4
SPEC_GEN = 256


def _vision_weights():
    """The vision tower's 422 checkpoint tensors in cere_ktest_inkling_vision_tensor order and their pointer table."""
    fn = oracle.ktest("cere_ktest_inkling_vision_tensor")
    name = ctypes.create_string_buffer(256)
    dims = (ctypes.c_int64 * 2)()
    names = []
    for i in range(VISION_TENSORS):
        if fn(ctypes.c_int64(i), name, ctypes.c_int64(len(name)), dims) <= 0:
            raise oracle.OracleError(f"cere_ktest_inkling_vision_tensor({i}) failed")
        names.append(name.value.decode())
    weights = [oracle.real_tensor(n).contiguous() for n in names]
    return weights, torch.tensor([t.data_ptr() for t in weights], dtype=torch.int64)


def _processor():
    """The tokenizer sidecar's InklingProcessor (bit-equal to the reference processor, S7's processor test)."""
    spec = importlib.util.spec_from_file_location("inkling_tokenizer_sidecar", SIDECAR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((oracle.MODEL_DIR / "processor_config.json").read_text())["image_processor"]
    return module.InklingProcessor(torch, config)


def image_arg(image):
    return (None, 0) if image is None else (image, image.shape[0])


def _table(names, tensors):
    """The ktest's tensor table of (names, tensors): (pointers int64, NUL-terminated names uint8, meta int64 [n, 6],
    count)."""
    ptrs = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64)
    blob = torch.tensor(list(b"".join(n.encode() + b"\0" for n in names)), dtype=torch.uint8)
    meta = torch.tensor([[DTYPES[t.dtype], t.dim()] + list(t.shape) + [0] * (4 - t.dim()) for t in tensors],
                        dtype=torch.int64)
    return ptrs, blob, meta, len(tensors)


class Model:
    """Every `model.llm.*` tensor of the checkpoint on GPU0 and the ktest's tensor table."""

    def __init__(self):
        path = oracle.MODEL_DIR / "model-00001-of-00001.safetensors"
        self.tensors = []
        self.names = []
        t0 = time.time()
        with safe_open(str(path), framework="pt", device=str(DEV)) as f:
            for name in sorted(f.keys()):
                if not name.startswith("model.llm."):
                    continue
                t = f.get_tensor(name).contiguous()
                if t.dtype not in DTYPES or t.dim() > 4:
                    raise oracle.OracleError(f"{name}: {t.dtype} {tuple(t.shape)}")
                self.tensors.append(t)
                self.names.append(name)
        self.load_s = time.time() - t0
        self.count = len(self.tensors)
        self._head = _table(self.names, self.tensors)
        self._draft_head = None
        self.draft_tensors = []
        self.images = {}

    def head(self):
        return self._head

    def draft_head(self):
        """head() plus mtp.safetensors' drafter tensors (names as in that file, no prefix: model/inkling/drafter.h
        bind_draft_weights), loaded on first use."""
        if self._draft_head is None:
            names = list(self.names)
            with safe_open(str(oracle.MODEL_DIR / "mtp.safetensors"), framework="pt", device=str(DEV)) as f:
                for name in sorted(f.keys()):
                    t = f.get_tensor(name).contiguous()
                    if t.dtype not in DTYPES or t.dim() > 4:
                        raise oracle.OracleError(f"{name}: {t.dtype} {tuple(t.shape)}")
                    self.draft_tensors.append(t)
                    names.append(name)
            self._draft_head = _table(names, self.tensors + self.draft_tensors)
        return self._draft_head

    def image(self, pid):
        """CERE's vision tower (S7, model/inkling/vision.h) over golden prompt pid's image: BF16 [placeholders,
        2304] on GPU0 (None for a text prompt). The tower's weights live only for the call."""
        if pid not in IMAGES:
            return None
        if pid not in self.images:
            from PIL import Image
            img = Image.open(io.BytesIO((FIXTURES / "vision" / f"{IMAGES[pid]}.png").read_bytes()))
            img.load()
            proc = _processor()
            cells, rows, cols = proc.cells(img)
            pixels = proc.patch_rows(cells).to(torch.bfloat16).to(DEV).contiguous()
            grid = torch.tensor([2 * rows, 2 * cols], dtype=torch.int32)
            out = oracle.empty((rows * cols, H), torch.bfloat16)
            weights, table = _vision_weights()
            oracle.call("cere_ktest_inkling_vision_encode", table, pixels, grid, 1, 0, out)
            del weights
            torch.cuda.empty_cache()
            self.images[pid] = out
        return self.images[pid]

    def forward(self, ids, chunk=CHUNK, decode_from=None, taps=False, logit_first=None, moe_mutant=0,
                no_messages=0, image=None):
        n = len(ids)
        decode_from = n if decode_from is None else decode_from
        logit_first = max(0, n - LASTN) if logit_first is None else logit_first  # p05_raw has 5 tokens
        visits = oracle.empty((VISITS, n, H), torch.bfloat16) if taps else None
        blocks = oracle.empty((n, 4, H), torch.bfloat16) if taps else None
        argmax = oracle.empty((n,), torch.int32)
        logits = oracle.empty((n - logit_first, V), torch.float32) if logit_first < n else None
        oracle.call("cere_ktest_inkling_forward", *self.head(), torch.tensor(ids, dtype=torch.int32), n, chunk,
                    decode_from, moe_mutant, no_messages, visits, blocks, argmax, logits, logit_first,
                    *image_arg(image))
        return {"visits": visits, "blocks": blocks, "argmax": argmax, "logits": logits}

    def generate(self, prompts, gen=GEN, forced=None, chunk=CHUNK, moe_mutant=0, no_messages=0, logits=True,
                 image=None):
        """image: the rows of every prompt's placeholders in prompt order (torch.cat of Model.image), or None."""
        s = len(prompts)
        flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
        lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
        ftok = None if forced is None else torch.tensor(forced, dtype=torch.int32).contiguous()
        ids = torch.full((s, gen), -1, dtype=torch.int32)
        lg = oracle.empty((s, gen, V), torch.float32) if logits else None
        oracle.call("cere_ktest_inkling_generate", *self.head(), flat, lens, s, gen, ftok, chunk, moe_mutant,
                    no_messages, ids, lg, *image_arg(image))
        return ids, lg

    def verify(self, prompts, gen, forced, run, seed=0, mutant=0, chunk=CHUNK):
        """cere_ktest_inkling_verify: generate's teacher-forced run with steps 1.. from speculative verify rounds of
        up to `run` rows per session (rejected rows included), then a commit. Returns (ids, logits) like generate."""
        s = len(prompts)
        flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
        lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
        ftok = torch.tensor(forced, dtype=torch.int32).contiguous()
        ids = torch.full((s, gen), -1, dtype=torch.int32)
        lg = oracle.empty((s, gen, V), torch.float32)
        oracle.call("cere_ktest_inkling_verify", *self.head(), flat, lens, s, gen, ftok, chunk, run, seed, mutant,
                    ids, lg)
        return ids, lg

    def verify_time(self, ids, sessions, max_m, reps):
        """cere_ktest_inkling_verify_time: [plain decode ms, verify of m rows per session ms for m = 1..max_m]."""
        out = torch.zeros(max_m + 1, dtype=torch.float64)
        oracle.call("cere_ktest_inkling_verify_time", *self.head(), torch.tensor(ids, dtype=torch.int32), len(ids),
                    sessions, max_m, reps, out)
        return out.tolist()

    def images_of(self, pids):
        """Model.image of pids, concatenated in order (None when no prompt has an image)."""
        rows = [self.image(p) for p in pids if p in IMAGES]
        return torch.cat(rows).contiguous() if rows else None

    def device_generate(self, prompts, gen=GEN, context=8192, budget=40 << 30, params=None, top_logprobs=None,
                        gens=None):
        """The device end to end (cere_ktest_inkling_device_generate: ABI v5 executor, model, one session per prompt,
        stop token eos; greedy when params is None, else SAMPLING_KEYS values, prompt p seeded seed + p; gens[p]
        tokens for prompt p when given, else gen): (ids [n, gen] (0 past a stop), result [n, 2] (committed, finish),
        device bytes, logprobs, seconds [n] (each prompt's session wall time)) where logprobs is None unless
        top_logprobs is given (needs params): int32 [n, gen, LOGPROB_WORDS], cere_token_logprobs."""
        n = len(prompts)
        flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
        lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
        budgets = None if gens is None else torch.tensor(gens, dtype=torch.int64)
        ids = torch.zeros((n, gen), dtype=torch.int32)
        res = torch.zeros((n, 2), dtype=torch.int64)
        used = torch.zeros((1,), dtype=torch.int64)
        pv = None if params is None else torch.tensor([float(params[k]) for k in SAMPLING_KEYS], dtype=torch.float64)
        lp = None if top_logprobs is None else torch.zeros((n, gen, LOGPROB_WORDS), dtype=torch.int32)
        seconds = torch.zeros((n,), dtype=torch.float64)
        oracle.call("cere_ktest_inkling_device_generate", *self.head(), flat, lens, n, gen, budgets, context, budget,
                    pv, top_logprobs or 0, ids, res, used, lp, seconds)
        return ids, res, int(used[0]), lp, seconds.tolist()

    def device_generate_draft(self, prompts, gen, window, concurrent=1, context=8192, budget=60 << 30, gens=None):
        """cere_ktest_inkling_device_generate_draft: greedy on a model with speculative_draft_tokens `window` (0 =
        plain, no drafter tensors), max_sessions `concurrent`, prompts in groups of `concurrent` submitted together:
        (ids [n, gen] (0 past a stop), result [n, 2] (committed, finish), device bytes, seconds [n] (each prompt's
        wall from its group's start), stats [n, 2] (verify_steps, accepted_drafts))."""
        n = len(prompts)
        flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
        lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
        budgets = None if gens is None else torch.tensor(gens, dtype=torch.int64)
        ids = torch.zeros((n, gen), dtype=torch.int32)
        res = torch.zeros((n, 2), dtype=torch.int64)
        used = torch.zeros((1,), dtype=torch.int64)
        seconds = torch.zeros((n,), dtype=torch.float64)
        stats = torch.zeros((n, 2), dtype=torch.int64)
        head = self.draft_head() if window else self.head()
        oracle.call("cere_ktest_inkling_device_generate_draft", *head, flat, lens, n, gen, budgets, context, budget,
                    window, concurrent, ids, res, used, seconds, stats)
        return ids, res, int(used[0]), seconds.tolist(), stats.tolist()

    def device_admission(self, sessions, context, pool, refused_pool, budget, ops):
        """The device's KV-byte admission (cere_ktest_inkling_device_admission): (the status of a model created with
        kv_pool_tokens refused_pool, each op's status (op > 0 creates a session of op tokens, op < 0 releases the
        session op index -op - 1 created), device bytes in use after the seal and after each op)."""
        n = len(ops)
        status = torch.full((n,), 1, dtype=torch.int64)
        used = torch.zeros((n + 1,), dtype=torch.int64)
        refused = torch.full((1,), 1, dtype=torch.int64)
        oracle.call("cere_ktest_inkling_device_admission", *self.head(), sessions, context, pool, refused_pool,
                    budget, torch.tensor(ops, dtype=torch.int64), n, status, used, refused)
        return int(refused[0]), status.tolist(), used.tolist()


def kl(p_logits, q_logits):
    """KL(p || q) per row (run_golden.py kl), FP32 logits."""
    lp = torch.log_softmax(p_logits.float(), -1)
    lq = torch.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def rel_l2(a, b):
    """run_taps.py rel_l2: per leading index (visit), over every other axis; a the reference."""
    a, b = a.float(), b.float()
    d = (a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1).clamp(min=1e-30)
    return d.tolist()


def golden_prompts():
    """Prompt ids of golden.json; a prompt longer than 256 tokens (p07) is stored without its ids there, so they are
    the first n_tokens ids of its taps file (checked against golden.json's input_ids_sha256)."""
    g = json.loads((FIXTURES / "golden.json").read_text())
    out = {}
    for pid, p in g["prompts"].items():
        ids = p["input_ids"]
        if ids is None:
            with safe_open(str(GOLDEN / "taps" / f"{pid}.safetensors"), framework="pt", device="cpu") as f:
                ids = f.get_tensor("input_ids")[: p["n_tokens"]].tolist()
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != p["input_ids_sha256"]:
            raise oracle.OracleError(f"{pid}: input ids differ from golden.json's input_ids_sha256")
        out[pid] = ids
    return out


def golden_q(pid):
    with safe_open(str(GOLDEN / "golden-v2" / "Q" / f"{pid}.safetensors"), framework="pt", device=str(DEV)) as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def taps_json():
    return json.loads((FIXTURES / "taps.json").read_text())


# -- rows ---------------------------------------------------------------------------------------------------------


def g2_prompt(model, pid, q, start, bars, factor, attn, fp32, eager, image=None, mutant=None):
    """One G2 prompt. q: the reference's taps (input_ids [prompt + greedy[:-1]], whole_visits, decode_visits,
    whole_block_states, decode_block_states); start: decode_from; bars: (decode bar, prefill bar) per visit before the
    attention widening; attn: REF's eager-vs-FP32-attention rel-L2 per visit; fp32 / eager: REF's whole prompt prefill
    taps with FP32 / eager attention; image: the rows of the placeholders (all below start); mutant: the forward's
    moe_mutant / no_messages keywords (row_g2). -> (record, pass)."""
    mutant = mutant or {}
    bar, prefill_bar = bars
    ids = q["input_ids"].tolist()
    bar_dec = [max(b, factor * a) for b, a in zip(bar, attn)]
    bar_pre = [max(b, factor * a) for b, a in zip(prefill_bar, attn)]
    r = model.forward(ids, decode_from=start, taps=True, logit_first=len(ids), image=image, **mutant)
    dec = rel_l2(q["decode_visits"], r["visits"][:, start:])
    pre = rel_l2(q["whole_visits"][:, :start], r["visits"][:, :start])
    whole = rel_l2(q["whole_visits"], r["visits"])
    blk_dec = rel_l2(q["decode_block_states"].transpose(0, 1), r["blocks"][start:].transpose(0, 1))
    del r
    # Binding (D-I-G2b): the whole prompt prefilled in one call against REF with FP32 attention, unwidened bar.
    rw = model.forward(ids, taps=True, logit_first=len(ids), image=image, **mutant)
    n_pre = fp32.shape[1]  # the prompt only (the attention runs prefill the prompt, the taps the whole sequence)
    vs_fp32 = rel_l2(fp32, rw["visits"][:, :n_pre])
    # Diagnostics (report only): CERE's whole prefill against golden, the attn run's eager taps against golden's
    # whole prefill over the prompt (REF's own prefill-length spread; 0 when bit-equal), Q's decode-vs-whole spread.
    diag = {"prefill_all_rel_l2": rel_l2(q["whole_visits"], rw["visits"]),
            "attn_eager_vs_golden": rel_l2(q["whole_visits"][:, :n_pre], eager),
            "q_spread": rel_l2(q["whole_visits"][:, start:], q["decode_visits"])}
    del rw
    dec_fail = [v for v in range(VISITS) if not dec[v] <= bar_dec[v]]
    pre_fail = [v for v in range(VISITS) if not pre[v] <= bar_pre[v]]
    fp32_fail = [v for v in range(VISITS) if not vs_fp32[v] <= prefill_bar[v]]
    p = {"n": len(ids), "decode_from": start, "bar_decode": bar_dec, "bar_prefill": bar_pre,
         "decode_rel_l2": dec, "prefill_rel_l2": pre, "whole_rel_l2": whole, "decode_block_rel_l2": blk_dec,
         "fp32_attn_rel_l2": vs_fp32, "decode_fail_visits": dec_fail, "prefill_fail_visits": pre_fail,
         "fp32_attn_fail_visits": fp32_fail, "decode_ratio_max": max(d / b for d, b in zip(dec, bar_dec)),
         "prefill_ratio_max": max(d / b for d, b in zip(pre, bar_pre)),
         "fp32_attn_ratio_max": max(d / b for d, b in zip(vs_fp32, prefill_bar)), "diag": diag}
    print(f"g2 {pid}: decode ratio max {p['decode_ratio_max']:.3f} fail {dec_fail}; prefill ratio max "
          f"{p['prefill_ratio_max']:.3f} fail {pre_fail}; vs FP32 attention ratio max "
          f"{p['fp32_attn_ratio_max']:.3f} fail {fp32_fail}", flush=True)
    return p, not dec_fail and not pre_fail and not fp32_fail


def row_g2(model, moe_mutant=0, no_messages=0):
    """D-I-G2b: against golden (REF eager attention) per prompt and visit max(the decode / prefill bar, 1.25x REF's
    eager-vs-FP32-attention rel-L2 of attn_spread.json); CERE's whole prefill (prefill positions) against REF with FP32
    attention (/attn) within the unwidened prefill bar 2 x max(decode spread, Q's chunk-512 spread). p08 (image) from
    p08_taps.json (/p08) with the reference tower's placeholder rows; pending when that fixture is absent. moe_mutant /
    no_messages: the mutant of every forward of the row (g2-softmax, g2-globalscale, g2-nomsg)."""
    mutant = {"moe_mutant": moe_mutant, "no_messages": no_messages}
    tj = taps_json()
    aj = json.loads((FIXTURES / "attn_spread.json").read_text())
    bar = tj["g2"]["bar"]
    spread = tj["g2"]["spread_max"]
    chunk_spread = tj["g3_chunk_visit_rel_l2"]["512"]["visit_rel_l2"]
    prefill_bar = [2 * max(a, b) for a, b in zip(spread, chunk_spread)]
    out = {"bar_decode": bar, "bar_prefill": prefill_bar, "attn_factor": aj["factor"], "prompts": {}}
    ok = True
    keys = ("input_ids", "whole_visits", "decode_visits", "whole_block_states", "decode_block_states")
    for pid, rec in tj["prompts"].items():
        with safe_open(str(GOLDEN / "taps" / f"{pid}.safetensors"), framework="pt", device=str(DEV)) as f:
            q = {k: f.get_tensor(k) for k in keys}
        with safe_open(str(ATTN / "attn_spread.safetensors"), framework="pt", device=str(DEV)) as f:
            fp32, eager = f.get_tensor(f"{pid}.fp32"), f.get_tensor(f"{pid}.eager")
        out["prompts"][pid], passed = g2_prompt(model, pid, q, rec["decode_from"], (bar, prefill_bar), aj["factor"],
                                                aj["prompts"][pid]["visit_rel_l2"], fp32, eager, mutant=mutant)
        ok = ok and passed
        del q, fp32, eager
        torch.cuda.empty_cache()
    p08 = FIXTURES / "p08_taps.json"
    if p08.is_file():
        pj = json.loads(p08.read_text())
        pid = pj["prompt"]
        with safe_open(str(P08 / f"{pid}.safetensors"), framework="pt", device=str(DEV)) as f:
            q = {k: f.get_tensor(k) for k in keys + ("whole_embeds", "attn_fp32", "attn_eager")}
        first, count = pj["placeholders"]
        if int((q["input_ids"] == IMAGE_TOKEN).sum()) != count or not bool(
                (q["input_ids"][first:first + count] == IMAGE_TOKEN).all()):
            raise oracle.OracleError(f"{pid}: placeholders differ from p08_taps.json's {pj['placeholders']}")
        ref_rows = q["whole_embeds"][first:first + count].contiguous()
        own = pj["visit_rel_l2"]  # p08's own decode-vs-whole spread: its bars are at least 2x it
        bars = ([max(b, 2 * s) for b, s in zip(bar, own)], [max(b, 2 * s) for b, s in zip(prefill_bar, own)])
        p, passed = g2_prompt(model, pid, q, pj["decode_from"], bars, aj["factor"], pj["attn_visit_rel_l2"],
                              q["attn_fp32"], q["attn_eager"], image=ref_rows, mutant=mutant)
        # Report only: CERE's tower rows against the reference tower's, and CERE's whole prompt prefill with its own
        # tower against golden's whole prefill.
        mine = model.image(pid)
        n_pre = q["attn_fp32"].shape[1]
        rw = model.forward(q["input_ids"][:n_pre].tolist(), taps=True, logit_first=n_pre, image=mine, **mutant)
        p["diag"]["cere_tower_rel_l2"] = rel_l2(ref_rows[None], mine[None])[0]
        p["diag"]["cere_tower_prefill_rel_l2"] = rel_l2(q["whole_visits"][:, :n_pre], rw["visits"])
        p["image_rows"] = "reference tower (whole_embeds placeholder rows)"
        out["prompts"][pid] = p
        ok = ok and passed
        del q, rw
        torch.cuda.empty_cache()
    else:
        out["pending"] = {"p08_image": "tests/fixtures/inkling2_reference/p08_taps.json absent (p08_taps.py)"}
    out["pass"] = ok
    return out


def g3_prefill_bars(pids):
    """D-I-G3b (maintainer, 2026-11-23): p07 keeps S1's bars (taps.json g3_prefill); every other prompt has its own
    bars from REF arm Q's cross-hardware spread (CPU vs golden v2), ref_cpu_spread.json (KL mean and max x 1.25,
    argmax 1 - 1.25 x (1 - REF's))."""
    s1 = taps_json()["g3_prefill"]["bars"]
    path = FIXTURES / "ref_cpu_spread.json"
    spread = json.loads(path.read_text())["prompts"] if path.is_file() else {}
    bars = {}
    for pid in pids:
        if pid == "p07_long_recall":
            bars[pid] = {"source": "taps.json g3_prefill", **{k: s1[k] for k in
                                                               ("argmax_agree_min", "kl_last16_mean_max",
                                                                "kl_last16_max_max")}}
        elif pid in spread:
            r = spread[pid]
            bars[pid] = {"source": "ref_cpu_spread.json", "argmax_agree_min": r["bar_argmax"],
                         "kl_last16_mean_max": r["bar_kl16_mean"], "kl_last16_max_max": r["bar_kl16_max"]}
        else:
            bars[pid] = None  # pending: ref_cpu_spread.json (InklingVulkan, build/inkling-s11) has no bars for it yet
    return bars


def row_g3(model, moe_mutant=0, no_messages=0):
    prompts = golden_prompts()
    pids = list(prompts)
    pbars = g3_prefill_bars(pids)
    out = {"bars": {"prefill": pbars, "decode_kl_mean_max": G3_DECODE_KL_MEAN, "decode_argmax_min": G3_DECODE_ARGMAX},
           "image_rows": {p: "CERE vision tower (S7)" for p in pids if p in IMAGES}, "prompts": {}}
    qs = {}
    tf = ATTN / "tf_fp32.safetensors"  # diagnostic: REF arm Q with FP32 attention (no cache), tf_fp32.py
    ref = safe_open(str(tf), framework="pt", device=str(DEV)) if tf.is_file() else None
    ok = True
    failed = False
    for pid in pids:
        q = golden_q(pid)
        qs[pid] = {"decode_logits": q["decode_logits"], "decode_ids": q["decode_ids"]}
        ids = prompts[pid]
        r = model.forward(ids, moe_mutant=moe_mutant, no_messages=no_messages, image=model.image(pid))
        agree = float((r["argmax"].long() == q["prefill_topk_ids"][:, 0].long()).float().mean())
        k = kl(q["prefill_last16"], r["logits"][-LASTN:])
        # Report only: golden's top-1 among exact ties is topk's arbitrary pick; count CERE's id as agreeing when its
        # golden logit ties golden's maximum.
        tied = (q["prefill_topk_logits"] == q["prefill_topk_logits"][:, :1]) & (
            q["prefill_topk_ids"].long() == r["argmax"].long()[:, None])
        p = {"n": len(ids), "prefill_argmax_agree": agree, "kl_last16_mean": float(k.mean()),
             "kl_last16_max": float(k.max()), "prefill_argmax_agree_ties": float(tied.any(-1).float().mean())}
        b = pbars[pid]
        p["prefill_pass"] = None if b is None else (
            agree >= b["argmax_agree_min"] and p["kl_last16_mean"] <= b["kl_last16_mean_max"]
            and p["kl_last16_max"] <= b["kl_last16_max_max"])
        if ref is not None and f"{pid}.fp32.argmax" in ref.keys():  # tf_fp32.py covers the text prompts
            ra = ref.get_tensor(f"{pid}.fp32.argmax").long()
            kr = kl(ref.get_tensor(f"{pid}.fp32.last16"), r["logits"][-LASTN:])
            kg = kl(q["prefill_last16"], ref.get_tensor(f"{pid}.fp32.last16"))
            p["diag_ref_fp32"] = {
                "cere_vs_ref_fp32": {"argmax_agree": float((r["argmax"].long() == ra).float().mean()),
                                     "kl16_mean": float(kr.mean()), "kl16_max": float(kr.max())},
                "ref_fp32_vs_golden": {"argmax_agree": float((ra == q["prefill_topk_ids"][:, 0].long()).float().mean()),
                                       "kl16_mean": float(kg.mean()), "kl16_max": float(kg.max())}}
        out["prompts"][pid] = p
        del r, q
    forced = [qs[pid]["decode_ids"].tolist() for pid in pids]
    image = model.images_of(pids)
    ids_tf, lg_tf = model.generate([prompts[p] for p in pids], forced=forced, moe_mutant=moe_mutant,
                                   no_messages=no_messages, image=image)
    ids_gr, _ = model.generate([prompts[p] for p in pids], moe_mutant=moe_mutant, no_messages=no_messages,
                               logits=False, image=image)
    prefix_ok = 0
    for i, pid in enumerate(pids):
        ql = qs[pid]["decode_logits"]
        k = kl(ql, lg_tf[i])
        argmax = int((ql.argmax(-1) == lg_tf[i].argmax(-1)).sum())
        qids = qs[pid]["decode_ids"].tolist()
        gr = ids_gr[i].tolist()
        prefix = next((j for j in range(GEN) if gr[j] != qids[j]), GEN)
        prefix_ok += prefix >= 16
        p = out["prompts"][pid]
        p.update({"decode_kl_mean": float(k.mean()), "decode_kl_max": float(k.max()), "decode_argmax": argmax,
                  "greedy_common_prefix": prefix, "greedy_ids": gr})
        if "diag_ref_fp32" in p:
            rl = ref.get_tensor(f"{pid}.fp32.decode")
            kr, kg = kl(rl, lg_tf[i]), kl(ql, rl)
            p["diag_ref_fp32"]["cere_vs_ref_fp32"].update(
                {"tf_kl_mean": float(kr.mean()), "tf_kl_max": float(kr.max()),
                 "tf_argmax": int((rl.argmax(-1) == lg_tf[i].argmax(-1)).sum())})
            p["diag_ref_fp32"]["ref_fp32_vs_golden"].update(
                {"tf_kl_mean": float(kg.mean()), "tf_kl_max": float(kg.max()),
                 "tf_argmax": int((ql.argmax(-1) == rl.argmax(-1)).sum())})
        p["decode_pass"] = p["decode_kl_mean"] <= G3_DECODE_KL_MEAN and argmax >= G3_DECODE_ARGMAX
        p["pass"] = bool(p["prefill_pass"]) and p["decode_pass"]
        failed = failed or p["prefill_pass"] is False or not p["decode_pass"]
        ok = ok and p["pass"]
        print(f"g3 {pid}: {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()})}",
              flush=True)
    out["greedy_prefix_ge16"] = prefix_ok
    out["bars_pending"] = [pid for pid in pids if pbars[pid] is None]
    out["failed"] = failed  # some prompt fails a bar it has: a mutant row's verdict
    out["pass"] = ok
    return out


def row_g4(model):
    prompts = golden_prompts()
    p07 = prompts["p07_long_recall"]
    res = {}
    a = model.forward(p07, chunk=4096, taps=True)
    b = model.forward(p07, chunk=4096)
    res["two_runs_prefill"] = bool(torch.equal(a["logits"], b["logits"]) and torch.equal(a["argmax"], b["argmax"]))
    c = model.forward(p07, chunk=512, taps=True)
    res["chunk_512_vs_4096_logits"] = bool(torch.equal(a["logits"], c["logits"]) and torch.equal(a["argmax"],
                                                                                                  c["argmax"]))
    res["chunk_512_vs_4096_visits"] = bool(torch.equal(a["visits"], c["visits"]))
    if not res["chunk_512_vs_4096_visits"]:
        res["chunk_first_differing_visit"] = next(v for v in range(VISITS) if not torch.equal(a["visits"][v],
                                                                                                c["visits"][v]))
    # Decode graph replays vs eager decode (taps force the eager path), across a kGraphPosStep (1,024) bound.
    d = model.forward(p07[:1040], decode_from=1016, taps=True, logit_first=1016)
    e = model.forward(p07[:1040], decode_from=1016, logit_first=1016)
    res["decode_graph_vs_eager"] = bool(torch.equal(d["logits"], e["logits"]) and torch.equal(d["argmax"],
                                                                                               e["argmax"]))
    del a, b, c, d, e
    torch.cuda.empty_cache()
    four = ["p01_chat_sky", "p02_chat_arith", "p03_chat_code", "p04_chat_fr"]
    forced = [golden_q(p)["decode_ids"].tolist() for p in four]
    one_ids, one = model.generate([prompts[four[0]]], forced=forced[:1])
    one2_ids, one2 = model.generate([prompts[four[0]]], forced=forced[:1])
    res["two_runs_generate"] = bool(torch.equal(one, one2) and torch.equal(one_ids, one2_ids))
    many_ids, many = model.generate([prompts[p] for p in four], forced=forced)
    res["sessions_1_vs_4"] = bool(torch.equal(one[0], many[0]) and torch.equal(one_ids[0], many_ids[0]))
    # Prefill vs decode of the same positions (reported; D9 needs equal logits only within one path).
    gr_ids, gr = model.generate([prompts[four[0]]], forced=forced[:1], chunk=1)
    res["prefill_chunk1_vs_4096_generate"] = bool(torch.equal(one, gr))
    res["pass"] = all(res[k] for k in ("two_runs_prefill", "chunk_512_vs_4096_logits", "chunk_512_vs_4096_visits",
                                       "decode_graph_vs_eager", "two_runs_generate", "sessions_1_vs_4"))
    print(f"g4: {res}", flush=True)
    return res


def row_perf(model, steps=256, reps=3):
    """Decode tok/s at 4K context (spec section 9: c1 >= 160 target, 100 ship minimum). Primary: the device
    (cere_ktest_inkling_device_generate: ABI v5 step, greedy argmax on the device, only the committed id read back,
    as cered drives it), c1 on p07's first 4,096 ids: one sealed model runs a warm-up prompt, then `reps` pairs of a
    1-token and a (1 + steps)-token session; each pair's extra committed tokens over its session wall-time difference
    (prefill cancels), median over the pairs. The 4K prefill rate (spec section 9: >= 5,000 tok/s): 4,096 over each
    pair's 1-token session time (the whole prompt in one 4,096-row call and its first token), median over the pairs.
    Secondary: the forward ktest's generate (host argmax over a logits readback every step), c1 and c4 (4 sessions
    of the same ids, one row each per step), 65 vs 1 tokens, two pairs."""
    ids = golden_prompts()["p07_long_recall"][:4096]
    gens = [1] + [1, 1 + steps] * reps
    stats = None
    if SPEC_WINDOW:
        _, r, _, sec, stats = model.device_generate_draft([ids] * len(gens), 1 + steps, SPEC_WINDOW, gens=gens)
    else:
        _, r, _, _, sec = model.device_generate([ids] * len(gens), gen=1 + steps, gens=gens)
    dev = [(int(r[2 * i + 2, 0]) - int(r[2 * i + 1, 0])) / (sec[2 * i + 2] - sec[2 * i + 1]) for i in range(reps)]
    pre = [len(ids) / sec[2 * i + 1] for i in range(reps)]
    c4 = None
    if has_ktest("cere_ktest_inkling_device_generate_draft"):
        # Groups of 4 sessions submitted together: a warm-up group, then per rep a 1-token and a (1 + steps)-token
        # group; each pair's extra committed tokens over its group wall-time difference.
        g4 = [1] * 4 + ([1] * 4 + [1 + steps] * 4) * reps
        _, r4, _, s4, _ = model.device_generate_draft([ids] * len(g4), 1 + steps, SPEC_WINDOW, 4, gens=g4)

        def group(k):
            return int(r4[4 * k:4 * k + 4, 0].sum()), max(s4[4 * k:4 * k + 4])

        c4 = []
        for i in range(reps):
            (n1, t1), (n2, t2) = group(2 * i + 1), group(2 * i + 2)
            c4.append((n2 - n1) / (t2 - t1))

    def wall(sessions, gen):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate([ids] * sessions, gen=gen, logits=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    wall(1, 2)  # warm
    kt = {c: [c * 64 / (wall(c, 65) - wall(c, 1)) for _ in range(2)] for c in (1, 4)}
    res = {"context": 4096, "decode_steps": steps, "decode_tok_s": statistics.median(dev), "decode_reps": dev,
           "prefill_tok_s": statistics.median(pre), "prefill_reps": pre,
           "finish": int(r[len(gens) - 1, 1]), "ktest_decode_steps": 64,
           "ktest_decode_tok_s": statistics.median(kt[1]), "ktest_c4_decode_tok_s": statistics.median(kt[4]),
           "ktest_reps": kt, "spec_window": SPEC_WINDOW,
           "note": "device (ABI v5 step, device argmax, committed id read back per step), c1, p07's first 4,096 "
                   "ids; prefill excluded by the difference of a (1 + steps)-token and a 1-token session on one model"}
    if c4 is not None:
        res.update(device_c4_decode_tok_s=statistics.median(c4), device_c4_reps=c4)
    if stats is not None:
        # (verify rounds, accepted drafts) of each (1 + steps)-token session
        res["spec_stats"] = [stats[2 * i + 2] for i in range(reps)]
    print(f"perf: {res}", flush=True)
    return res


def has_ktest(name):
    """Whether the library under test exports `name` (an older baseline library may not)."""
    try:
        oracle.ktest(name)
    except oracle.OracleError:
        return False
    return True


def row_spec(model):
    """S9 phase B (D-I-S9B bar (1)): with drafts, the device's greedy ids equal its plain greedy ids. Cases: the
    golden text prompts p01-p07 and p07's first 4,096 ids, SPEC_GEN tokens each, c1 (one session at a time) and c4
    (four sessions submitted together, their rounds batched); plain is the same library and ktest with window 0
    (no drafter). Each case also records its verify rounds, accepted drafts and committed tokens per round. Plain c4
    must equal plain c1 too (G4: a row's ids do not depend on the batch)."""
    w = SPEC_WINDOW or SPEC_DEFAULT_WINDOW
    prompts = golden_prompts()
    cases = [(p, prompts[p]) for p in prompts if p not in IMAGES] + [("p07_4k", prompts["p07_long_recall"][:4096])]
    ps = [c[1] for c in cases]
    context = max(len(p) for p in ps) + SPEC_GEN
    out = {"window": w, "gen": SPEC_GEN, "cases": {}}
    ok = True
    plain = {}
    for conc in (1, 4):
        want, wres, _, wsec, _ = model.device_generate_draft(ps, SPEC_GEN, 0, conc, context)
        plain[conc] = (want, wres)
        torch.cuda.empty_cache()
        ids, res, used, sec, st = model.device_generate_draft(ps, SPEC_GEN, w, conc, context)
        torch.cuda.empty_cache()
        for i, (pid, _) in enumerate(cases):
            n, finish = int(res[i, 0]), int(res[i, 1])
            equal = (n == int(wres[i, 0]) and finish == int(wres[i, 1]) and n >= 1
                     and bool(torch.equal(ids[i, :n], want[i, :n])))
            rounds, acc = st[i]
            out["cases"][f"{pid}_c{conc}"] = {
                "committed": n, "finish": finish, "equal": equal, "verify_steps": rounds, "accepted_drafts": acc,
                "tokens_per_round": (rounds + acc) / rounds if rounds else None, "seconds": sec[i],
                "plain_seconds": wsec[i]}
            ok = ok and equal and rounds > 0
        out[f"device_bytes_c{conc}"] = used
    out["plain_c4_equals_c1"] = bool(torch.equal(plain[1][0], plain[4][0]) and torch.equal(plain[1][1], plain[4][1]))
    out["pass"] = ok and out["plain_c4_equals_c1"]
    print(f"spec: pass {out['pass']} window {w} " + json.dumps(
        {k: (v["equal"], v["committed"], v["tokens_per_round"] and round(v["tokens_per_round"], 3))
         for k, v in out["cases"].items()}), flush=True)
    return out


def row_spec_classes(model):
    """S9 phase B (D-I-S9B bar (1) on S9r's prompt classes, and the accept report): S9r's rendered prompts
    (/s9r/prompts.json: class, ids, gen; the golden ones, ids null there, are row spec's) one session at a time,
    greedy with drafts (window SPEC_WINDOW, else SPEC_DEFAULT_WINDOW) against the same library's plain path. Every
    prompt's ids must equal plain's; per class: committed tokens per verify round (sum over the class) and the wall
    ratio plain / drafts (each prompt's prefill included)."""
    w = SPEC_WINDOW or SPEC_DEFAULT_WINDOW
    rec = json.loads(Path("/s9r/prompts.json").read_text())
    cases = [(pid, r) for pid, r in rec.items() if r.get("ids")]
    ps = [r["ids"] for _, r in cases]
    gens = [int(r["gen"]) for _, r in cases]
    gen = max(gens)
    context = max(len(p) + g for p, g in zip(ps, gens))
    want, wres, _, wsec, _ = model.device_generate_draft(ps, gen, 0, 1, context, gens=gens)
    torch.cuda.empty_cache()
    ids, res, used, sec, st = model.device_generate_draft(ps, gen, w, 1, context, gens=gens)
    torch.cuda.empty_cache()
    out = {"window": w, "prompts": {}, "classes": {}, "device_bytes": used}
    ok = True
    for i, (pid, r) in enumerate(cases):
        n, finish = int(res[i, 0]), int(res[i, 1])
        equal = (n == int(wres[i, 0]) and finish == int(wres[i, 1]) and n >= 1
                 and bool(torch.equal(ids[i, :n], want[i, :n])))
        rounds, acc = st[i]
        ok = ok and equal and rounds > 0
        out["prompts"][pid] = {"class": r["class"], "committed": n, "finish": finish, "equal": equal,
                               "verify_steps": rounds, "accepted_drafts": acc, "seconds": sec[i],
                               "plain_seconds": wsec[i]}
        c = out["classes"].setdefault(r["class"], {"prompts": 0, "verify_steps": 0, "accepted_drafts": 0,
                                                   "seconds": 0.0, "plain_seconds": 0.0, "equal": True})
        c["prompts"] += 1
        c["verify_steps"] += rounds
        c["accepted_drafts"] += acc
        c["seconds"] += sec[i]
        c["plain_seconds"] += wsec[i]
        c["equal"] = c["equal"] and equal
    for c in out["classes"].values():
        c["tokens_per_round"] = (c["verify_steps"] + c["accepted_drafts"]) / c["verify_steps"] if c["verify_steps"] \
            else None
        c["wall_ratio"] = c["plain_seconds"] / c["seconds"] if c["seconds"] else None
    out["pass"] = ok
    print(f"spec-classes: pass {ok} window {w} " + json.dumps(
        {k: (v["equal"], v["tokens_per_round"] and round(v["tokens_per_round"], 3),
             v["wall_ratio"] and round(v["wall_ratio"], 3)) for k, v in out["classes"].items()}), flush=True)
    return out


def row_verify(model):
    """S9 phase B (D-I-S9B bar (2), fail-first): speculative verify rounds (cere_ktest_inkling_verify: k rows per
    session cycling 1..run, rejected rows included, then a commit of the accepted ones) give every step the M = 1
    decode's ids and FP32 logits bit for bit. Cases: p07's first 4,096 ids, one session, runs of up to 16 rows over 48
    steps (the 4K position); the four chat prompts p01-p04 together, runs of up to 16 (64 rows a call), GEN steps;
    two seeds each. Mutants (each must break bit-equality on some case; the ring one only past 1,024 positions, where a
    rejected row's slot still holds a key inside the window): 1 = every row's conv window from the state alone (no
    in-run rows), 2 = the sliding K/V stored in place (rejected rows left in the ring)."""
    prompts = golden_prompts()
    p07 = prompts["p07_long_recall"][:4096]
    long_ids, long_lg = model.generate([p07], gen=48)
    four = ["p01_chat_sky", "p02_chat_arith", "p03_chat_code", "p04_chat_fr"]
    forced = [golden_q(p)["decode_ids"].tolist() for p in four]
    want_ids, want_lg = model.generate([prompts[p] for p in four], forced=forced)
    cases = {"p07_4k_c1": ([p07], 48, long_ids.tolist(), long_ids, long_lg),
             "chat_c4": ([prompts[p] for p in four], GEN, forced, want_ids, want_lg)}
    res = {"cases": {}, "mutants": {}}
    ok = True
    for name, (ps, gen, fc, wi, wl) in cases.items():
        for seed in (0, 1):
            ids, lg = model.verify(ps, gen, fc, run=16, seed=seed)
            eq_ids, eq_lg = bool(torch.equal(ids, wi)), bool(torch.equal(lg, wl))
            res["cases"][f"{name}_seed{seed}"] = {"ids_equal": eq_ids, "logits_equal": eq_lg}
            ok = ok and eq_ids and eq_lg
            del lg
    for mutant in (1, 2):
        per = {}
        for name, (ps, gen, fc, wi, wl) in cases.items():
            ids, lg = model.verify(ps, gen, fc, run=16, seed=0, mutant=mutant)
            per[name] = {"fails": not bool(torch.equal(lg, wl)), "ids_equal": bool(torch.equal(ids, wi))}
            del lg
        res["mutants"][str(mutant)] = {"cases": per, "fails": any(v["fails"] for v in per.values())}
    torch.cuda.empty_cache()
    res["mutants_fail"] = all(v["fails"] for v in res["mutants"].values())
    res["pass"] = ok and res["mutants_fail"]
    print(f"verify: {res}", flush=True)
    return res


def row_verify_time(model, reps=20):
    """S9 pricing (D-I-S9B: the chain length from measured M-row time): at p07's 4,096-token position, the ms of a
    plain decode step and of a verify of M rows per session (M = 1..16 at c1, 1..16 at c4), each with every row's
    logits and argmax (cere_ktest_inkling_verify_time; graph replays, mean of `reps` after a warm-up)."""
    ids = golden_prompts()["p07_long_recall"][:4096]
    res = {"context": 4096, "reps": reps}
    for c in (1, 4):
        t = model.verify_time(ids, c, 16, reps)
        res[f"c{c}"] = {"decode_ms": t[0], "verify_ms": {str(m): t[m] for m in range(1, 17)}}
    print(f"verify-time: {res}", flush=True)
    return res


def row_device(model):
    """The CUDA device through the ABI v5 runtime (cere_ktest_inkling_device_generate) against the forward ktest's
    greedy generation of the same prompts: equal ids over the device's committed tokens, which are GEN unless the
    device committed the stop token (eos). Text prompts only: the device refuses images (D-I5)."""
    prompts = golden_prompts()
    pids = [p for p in prompts if p not in IMAGES]
    want, _ = model.generate([prompts[p] for p in pids], logits=False)
    torch.cuda.empty_cache()
    t0 = time.time()
    ids, res, used, _, _ = model.device_generate([prompts[p] for p in pids])
    out = {"wall_device_s": time.time() - t0, "device_bytes_after_seal": used, "prompts": {}}
    ok = True
    for i, pid in enumerate(pids):
        n, finish = int(res[i, 0]), int(res[i, 1])
        stopped = finish == 2 and n >= 1 and int(ids[i, n - 1]) == EOS  # CERE_FINISH_STOP_TOKEN
        equal = bool(torch.equal(ids[i, :n], want[i, :n])) and (n == GEN or stopped)
        out["prompts"][pid] = {"committed": n, "finish": finish, "equal": equal, "ids": ids[i, :n].tolist(),
                               "ktest_ids": want[i].tolist()}
        ok = ok and equal
    out["pass"] = ok
    print(f"device: pass {ok} " + json.dumps({p: (v["committed"], v["finish"], v["equal"])
                                               for p, v in out["prompts"].items()}), flush=True)
    return out


def logprob_rows(lp, n):
    """The first n cere_token_logprobs of one prompt (int32 [gen, LOGPROB_WORDS]) as (id, logprob, top) with top the
    top_count (id, logprob) pairs."""
    f = lp.view(torch.float32)
    rows = []
    for j in range(n):
        k = int(lp[j, 2])
        top = [(int(lp[j, 4 + 2 * t]), float(f[j, 5 + 2 * t])) for t in range(k)]
        rows.append((int(lp[j, 0]), float(f[j, 1]), top))
    return rows


def logprobs_sane(rows, ids, top_n):
    """Per committed step: id is the committed token, top_n entries in descending logprob order (ties: lower id
    first), every logprob <= 0, and the committed token's logprob equals its entry when it is in the list."""
    for (lid, lpv, top), tid in zip(rows, ids):
        if lid != tid or len(top) != top_n or not lpv <= 0.0:
            return False
        for (ia, la), (ib, lb) in zip(top, top[1:]):
            if not (la > lb or (la == lb and ia < ib)):
                return False
        if any(not lv <= 0.0 for _, lv in top):
            return False
        if any(i == lid and lv != lpv for i, lv in top):
            return False
    return True


SAMPLED = {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0, "top_k": 50, "frequency_penalty": 0.0,
           "presence_penalty": 0.0, "repetition_penalty": 1.0, "seed": 1234}


def row_sampling(model, pids=("p01_chat_sky", "p03_chat_code", "p07_long_recall"), top_n=5):
    """The device's sampler (S6g; kernels/k12_sampler.h on the softcapped FP32 logits, ABI v5 step), text prompts:
    (1) sampling params at temperature 0 with logprobs = the greedy device's ids, and the logprobs sane
    (logprobs_sane; the committed token first in its list); (2) temperature 1, top_p 0.95, top_k 50, seed 1234 twice
    = bit-identical ids and logprobs (per-seed determinism, D4), logprobs sane; (3) seed 1235 (report only: how many
    prompts differ from seed 1234)."""
    prompts = golden_prompts()
    seqs = [prompts[p] for p in pids]
    greedy, gres, _, _, _ = model.device_generate(seqs)
    zero = dict(SAMPLED, temperature=0.0)
    z, zres, _, zlp, _ = model.device_generate(seqs, params=zero, top_logprobs=top_n)
    a, ares, _, alp, _ = model.device_generate(seqs, params=SAMPLED, top_logprobs=top_n)
    b, bres, _, blp, _ = model.device_generate(seqs, params=SAMPLED, top_logprobs=top_n)
    c, cres, _, _, _ = model.device_generate(seqs, params=dict(SAMPLED, seed=SAMPLED["seed"] + 1))
    out = {"params": SAMPLED, "top_logprobs": top_n, "prompts": {}}
    ok = True
    for i, pid in enumerate(pids):
        n0, nz, na = int(gres[i, 0]), int(zres[i, 0]), int(ares[i, 0])
        zrows = logprob_rows(zlp[i], nz)
        arows = logprob_rows(alp[i], na)
        temp0_greedy = nz == n0 and bool(torch.equal(z[i, :nz], greedy[i, :n0]))
        temp0_top1 = all(top and top[0][0] == lid for lid, _, top in zrows)
        temp0_sane = logprobs_sane(zrows, z[i, :nz].tolist(), top_n)
        seeded_equal = (bool(torch.equal(ares[i], bres[i])) and bool(torch.equal(a[i], b[i])) and
                        bool(torch.equal(alp[i], blp[i])))
        sampled_sane = logprobs_sane(arows, a[i, :na].tolist(), top_n)
        r = {"temp0_equals_greedy": temp0_greedy, "temp0_committed_is_top1": temp0_top1,
             "temp0_logprobs_sane": temp0_sane, "seed_bit_identical": seeded_equal,
             "sampled_logprobs_sane": sampled_sane, "sampled_ids": a[i, :na].tolist(),
             "sampled_finish": int(ares[i, 1]), "other_seed_differs": not bool(torch.equal(a[i], c[i])),
             "greedy_ids": greedy[i, :n0].tolist(),
             "sampled_logprobs": [round(lv, 4) for _, lv, _ in arows]}
        out["prompts"][pid] = r
        ok = ok and temp0_greedy and temp0_top1 and temp0_sane and seeded_equal and sampled_sane
    out["pass"] = ok
    print(f"sampling: pass {ok} " + json.dumps({p: [v[k] for k in ("temp0_equals_greedy", "temp0_committed_is_top1",
                                                                    "temp0_logprobs_sane", "seed_bit_identical",
                                                                    "sampled_logprobs_sane", "other_seed_differs")]
                                                 for p, v in out["prompts"].items()}), flush=True)
    return out


def sizes(ctx):
    """(Forward::workspace_bytes(), session_bytes(ctx)) of the build (cere_ktest_inkling_sizes, no device work)."""
    out = (ctypes.c_int64 * 2)()
    if oracle.ktest("cere_ktest_inkling_sizes")(ctypes.c_int64(ctx), out) != 0:
        raise oracle.OracleError(f"cere_ktest_inkling_sizes({ctx}) failed")
    return int(out[0]), int(out[1])


def row_admission(model):
    """S10 KV-byte admission of the device (tasks/inkling-spec.md section 9.2): the model's estimate counts the
    kv_pool_tokens' full-visit KV once (a pool beyond the budget is CAPACITY at create; 16 x 262,144-token sessions,
    refused by the pre-S10 estimate, fit), and cere_session_create admits a session only while the open sessions'
    contexts stay within the pool (ADMIT_OPS: fill, CAPACITY, release, refill). Device bytes: a create adds
    session_bytes(context), a refused create none, a release returns it (each within ADMIT_SLACK)."""
    a = ADMIT
    workspace, sb_max = sizes(a["context"])
    refused, status, used = model.device_admission(a["sessions"], a["context"], a["pool"], a["refused_pool"],
                                                   a["budget"], list(ADMIT_OPS))
    torch.cuda.empty_cache()
    deltas = [used[i + 1] - used[i] for i in range(len(ADMIT_OPS))]
    want = []
    for op, st in zip(ADMIT_OPS, ADMIT_WANT):
        size = sizes(op)[1] if op > 0 else -sizes(ADMIT_OPS[-op - 1])[1]
        want.append(size if st == 0 else 0)
    bytes_ok = [abs(d - w) <= ADMIT_SLACK for d, w in zip(deltas, want)]
    out = {**a, "ops": list(ADMIT_OPS), "want": list(ADMIT_WANT), "status": status, "refused_pool_status": refused,
           "device_bytes": used, "deltas": deltas, "want_deltas": want, "bytes_ok": bytes_ok,
           "workspace_bytes": workspace, "session_bytes_context": sb_max,
           "pre_s10_kv_bytes": a["sessions"] * sb_max}
    out["pass"] = (refused == CAPACITY and status == list(ADMIT_WANT) and all(bytes_ok)
                   and a["sessions"] * sb_max > a["budget"])
    print(f"admission: pass {out['pass']} refused {refused} status {status} deltas {deltas}", flush=True)
    return out


def row_g6_long32k(model):
    """G6's G3 on S1's 32K prompt (taps.json long32k, /golden/long32k/Q.safetensors; report only until the owner
    rules on its bars): prefill (chunks of 4,096) argmax agreement over every position vs Q's top-1 and KL(Q || CERE)
    of the last 16 rows, teacher-forced decode on Q's 24 greedy ids vs Q's cached decode logits, the greedy common
    prefix; `bars_p07` applies p07's prefill bars (taps.json g3_prefill) and D-I-G3's decode bars, for information.
    prefill_s: the forward ktest's prefill wall (G6 ticket plan's 32K point)."""
    rec = taps_json()["long32k"]
    path = GOLDEN / rec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != rec["file_sha256"]:
        raise oracle.OracleError(f"{path}: sha256 differs from taps.json long32k file_sha256")
    keys = ("input_ids", "prefill_topk_ids", "prefill_topk_logits", "prefill_last16", "decode_ids", "decode_logits")
    with safe_open(str(path), framework="pt", device=str(DEV)) as f:
        q = {k: f.get_tensor(k) for k in keys}
    ids = q["input_ids"].tolist()
    if len(ids) != rec["n_tokens"] or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != rec["input_ids_sha256"]:
        raise oracle.OracleError("long32k: input ids differ from taps.json's n_tokens / input_ids_sha256")
    torch.cuda.synchronize()
    t0 = time.time()
    r = model.forward(ids)
    prefill_s = time.time() - t0
    agree = float((r["argmax"].long() == q["prefill_topk_ids"][:, 0].long()).float().mean())
    tied = (q["prefill_topk_logits"] == q["prefill_topk_logits"][:, :1]) & (
        q["prefill_topk_ids"].long() == r["argmax"].long()[:, None])
    k = kl(q["prefill_last16"], r["logits"][-LASTN:])
    del r
    torch.cuda.empty_cache()
    qids = q["decode_ids"].tolist()
    _, lg = model.generate([ids], forced=[qids])
    kd = kl(q["decode_logits"], lg[0])
    dec_argmax = int((q["decode_logits"].argmax(-1) == lg[0].argmax(-1)).sum())
    del lg
    gr, _ = model.generate([ids], logits=False)
    gr = gr[0].tolist()
    b = taps_json()["g3_prefill"]["bars"]
    out = {"n": len(ids), "needle_record": rec["needle_record"], "expect": rec["expect"], "prefill_s": prefill_s,
           "prefill_argmax_agree": agree, "prefill_argmax_agree_ties": float(tied.any(-1).float().mean()),
           "kl_last16_mean": float(k.mean()), "kl_last16_max": float(k.max()), "decode_kl_mean": float(kd.mean()),
           "decode_kl_max": float(kd.max()), "decode_argmax": dec_argmax,
           "greedy_common_prefix": next((j for j in range(GEN) if gr[j] != qids[j]), GEN), "greedy_ids": gr,
           "q_greedy_ids": qids, "bars_p07": {**b, "decode_kl_mean_max": G3_DECODE_KL_MEAN,
                                              "decode_argmax_min": G3_DECODE_ARGMAX}}
    out["bars_p07"]["pass"] = (agree >= b["argmax_agree_min"] and out["kl_last16_mean"] <= b["kl_last16_mean_max"]
                               and out["kl_last16_max"] <= b["kl_last16_max_max"]
                               and out["decode_kl_mean"] <= G3_DECODE_KL_MEAN and dec_argmax >= G3_DECODE_ARGMAX)
    out["pass"] = True  # report only (owner decision on the 32K bars); complete
    print("g6-long32k: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()
                                       if not k.endswith("ids")}), flush=True)
    return out


TF_VARIANTS = (("cere", 0, 0), ("softmax", MUT_SOFTMAX, 0), ("globalscale", MUT_GLOBAL_SCALE, 0), ("nomsg", 0, 1))


def row_tf_fp32ids(model):
    """G3's decode half (report only, D-I-G3d; the judge scores it against golden_fp32.json's bars): teacher-forced
    decode on the FP32-attention golden's greedy ids, per prompt and step KL(golden-FP32 || CERE) and both argmaxes,
    for CERE and its three G3 mutants."""
    prompts = golden_prompts()
    pids = list(prompts)
    gf = json.loads((FIXTURES / "golden_fp32.json").read_text())
    if gf["meta"].get("attention_precision") != "fp32":
        raise oracle.OracleError("golden_fp32.json is not the FP32-attention golden")
    forced, gl = [], {}
    for pid in pids:
        if gf["prompts"][pid]["input_ids_sha256"] != hashlib.sha256(json.dumps(prompts[pid]).encode()).hexdigest():
            raise oracle.OracleError(f"{pid}: golden_fp32.json's prompt ids differ from golden.json's")
        with safe_open(str(GFP32 / "Q" / f"{pid}.safetensors"), framework="pt", device=str(DEV)) as f:
            ids = f.get_tensor("decode_ids").tolist()
            gl[pid] = f.get_tensor("decode_logits")
        if ids != gf["arms"]["Q"][pid]["greedy_ids"]:
            raise oracle.OracleError(f"{pid}: /gfp32/Q decode_ids differ from golden_fp32.json's greedy_ids")
        forced.append(ids)
    image = model.images_of(pids)
    seqs = [prompts[p] for p in pids]
    out = {"golden": "tests/fixtures/inkling2_reference/golden_fp32.json arm Q (FP32 attention), /gfp32/Q",
           "image_rows": {p: "CERE vision tower (S7)" for p in pids if p in IMAGES}, "steps": GEN, "variants": {}}
    for name, moe_mutant, no_messages in TF_VARIANTS:
        _, lg = model.generate(seqs, forced=forced, moe_mutant=moe_mutant, no_messages=no_messages, image=image)
        v = {}
        for i, pid in enumerate(pids):
            k = kl(gl[pid], lg[i])
            ga, ca = gl[pid].argmax(-1), lg[i].argmax(-1)
            v[pid] = {"forced_ids": forced[i], "kl": k.tolist(), "golden_argmax": ga.tolist(),
                      "cere_argmax": ca.tolist(), "argmax_agree": int((ga == ca).sum()), "kl_mean": float(k.mean()),
                      "kl_max": float(k.max())}
        out["variants"][name] = {"moe_mutant": moe_mutant, "no_messages": no_messages, "prompts": v}
        print(f"tf-fp32ids {name}: " + json.dumps({p: (round(x["kl_mean"], 5), round(x["kl_max"], 4),
                                                       x["argmax_agree"]) for p, x in v.items()}), flush=True)
        del lg
        torch.cuda.empty_cache()
    out["pass"] = all(len(out["variants"][n]["prompts"]) == len(pids) for n, _, _ in TF_VARIANTS)  # complete
    return out


ROWS = {
    "g2": row_g2,
    "g2-softmax": lambda m: row_g2(m, moe_mutant=MUT_SOFTMAX),
    "g2-globalscale": lambda m: row_g2(m, moe_mutant=MUT_GLOBAL_SCALE),
    "g2-nomsg": lambda m: row_g2(m, no_messages=1),
    "g3": row_g3,
    "g3-softmax": lambda m: row_g3(m, moe_mutant=MUT_SOFTMAX),
    "g3-globalscale": lambda m: row_g3(m, moe_mutant=MUT_GLOBAL_SCALE),
    "g3-nomsg": lambda m: row_g3(m, no_messages=1),
    "g4": row_g4,
    "perf": row_perf,
    "device": row_device,
    "spec": row_spec,
    "spec-classes": row_spec_classes,
    "verify": row_verify,
    "verify-time": row_verify_time,
    "sampling": row_sampling,
    "admission": row_admission,
    "g6-long32k": row_g6_long32k,
    "tf-fp32ids": row_tf_fp32ids,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("rows", nargs="+", choices=sorted(ROWS))
    args = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    model = Model()
    print(f"loaded {model.count} tensors in {model.load_s:.1f}s", flush=True)
    res = {"schema": "inkling-forward-v1", "load_s": model.load_s, "rows": {}}
    for row in args.rows:
        t0 = time.time()
        r = ROWS[row](model)
        r["wall_s"] = time.time() - t0
        if row.startswith("g2-"):
            r["mutant_fails_g2"] = not r["pass"]
        if row.startswith("g3-"):
            r["mutant_fails_g3"] = r["failed"]
        res["rows"][row] = r
        Path(args.out).write_text(json.dumps(res, indent=1))
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
