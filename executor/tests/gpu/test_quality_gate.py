"""D9Q (b): teacher-forced quality of a numerics-changing lever set against a common reference (tasks/m23-spec.md
D9Q, [REVISED: M23-D9Q-b3v2], [REVISED: M23-D9Q-b2v3], [REVISED: M23-D9Q-screen], [REVISED: owner 2026-10-16 screen
tiers]; suite `quality`, run through `native_dev.py test --suite quality --shape L|S [--lever-set S [--screen |
--triage | --prompt-ids IDS]]`).

One process. The predecessor P is a model with null levers (every lever at the library default). The stdin frame
PROMPT_FRAME (native_dev.py regenerates quality_heldout_v2 from git objects and checks its digests; the container has
carries the shape's main prompts, the long-context suite (k <= 16 prompts, detection-only) and the run mode:
* baseline (`lever_set` null): P, SELF_CONSISTENCY (a second null-lever model) and the controls; the shape decision
  (`quality_measure.shape_decision`) reads these runs;
* candidate (`lever_set` S): the same plus the lever set S of the lever registry fixture
  (`tests/fixtures/lever_registry.json`; `quality_measure.lever_sets`: one d9_batch implementation, or a bundle of
  them joined by `+` in registry order), at the committed quality_measure.SHAPE and CLUSTER_LEVEL;
* screen (`lever_set` S and the frame's `screen`, [REVISED: M23-D9Q-screen]): the candidate run on the SCREEN_N main
  prompts native_dev.py selected (`quality_measure.screen_select`), the whole long suite and every control (P and
  SELF_CONSISTENCY run, nothing reused); the frame's screen must be `quality_measure.screen_frame()` at the pinned
  SCREEN_PREREG_COMMIT and the corpus row records it (mode `screen`);
* triage (`--triage`, [REVISED: owner 2026-10-16 screen tiers]): the same on the TRIAGE_N main prompts of
  `quality_measure.triage_select`, frame screen `quality_measure.triage_frame()`, mode `triage` (never a verdict);
* subset (`--prompt-ids`, the attribution rerun): P and S only (no control, no long suite) on exactly the frame's
  main prompt ids (`quality_measure.subset_frame`), mode `subset`, diagnostic only (`quality_measure.subset_check`).
The frame's mode is `quality_measure.run_mode` (any other frame fails before a model loads).
A candidate checkpoint (the frame's `checkpoint`, `--candidate-checkpoint NAME`; tasks/fp4local-spec.md D11, FL-16,
report-only) replaces the lever set in any mode: the set NAME is a null-lever model of the checkpoint in
quality_measure.CANDIDATE_DIR_ENV's directory, placed by quality_measure.CHECKPOINTS[NAME]'s families file
(`k8.checkpoint_dir`); its config must give P's cere_qwen4_exp_config and its lm_head must equal P's (the logits use
P's lm_head). P, SELF_CONSISTENCY and the controls stay on /model. A frame `reference` (`--reference-checkpoint NAME`,
tasks/k2-prefill-spec.md G4, opt-in) names /model's checkpoint (native_dev.py mounts it there): P, SELF_CONSISTENCY,
the controls and every lever set are placed by quality_measure.REFERENCES[NAME]'s families file (`k8.checkpoint_dir`
on /model for the class).
Each set gets its own model (created, measured and released before the next one), at CONTEXT tokens and one session.

Window and framing (D9Q-b2v3 (2)), identical for R, P, C and every control: sequence = `chat_ids([user])` (SGLang's
chat path, generation prompt, `enable_thinking` false) followed by the continuation encoded without specials, as the
assistant reply. The scored rows are the last WINDOW positions: the dec arm's PREFILL length is s = T - WINDOW and the
rows are [s, T - 1); row r predicts token r + 1. Every frame token count is checked, the continuation holds at least
MIN_CONT tokens and no special or added token.
* pre arm: `model` rows of one teacher-forced prefill (`cere_ktest_forward_trace`, the last T - s rows);
* dec arm: PREFILL of s tokens, then forced DECODE of the rest (`cere_ktest_forced_decode_trace`), one row per step.
Distributions: `model` rows x lm_head^T accumulated in fp32, log-softmax in fp64 (D1's `tf_logprobs`). R = P's pre arm.

Controls (dev-only switches of this suite's logit computation; never levers): NEGATIVE = P's rows through an lm_head
with BF16 partial sums (fp32 dot products over each LOGIT_TILE-wide k tile, the running sum rounded to BF16 after every
tile); SENSITIVITY = P's fp32 logits rounded to BF16 before the softmax (reported only); INJ_TARGET = P's rows with
the target's probability scaled by e^-INJ_C and the rest renormalised (dNLL = +INJ_C on every row). The spread
controls are computed from the per-prompt null values in `quality_measure.run_check`.

Report rows (scalars only): the corpus row (prompt set, PREREG_COMMIT, the fixture's sha256, shape, mode, sizes), the
framing rows, per set, part and arm the candidate and predecessor summaries and the paired row, one `prompt <id>` row
per prompt (file and directory index, x_i, every set's d_i) and one verdict row per set. `native_dev.py
quality-baseline` and `quality-verdict` replay everything from these rows; the suite fails if a required control
misbehaves or the framing check fails (the run is invalid). Prompt text, token ids, logits and `model` rows stay in
memory."""
from __future__ import annotations

import contextlib
import gc
import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from typing import NamedTuple

import torch

import oracle

k8 = importlib.import_module("test_model_k8_forward")
p4 = importlib.import_module("test_model_k8p4_spec")
b8 = importlib.import_module("test_numerics_b8_layers")
qm = importlib.import_module("quality_measure")

ROW = qm.ROW
PROMPT_FRAME = "cere-quality-heldout-v2"
PROMPT_SET = qm.PROMPT_SET
WINDOW = 128
MIN_CONT = 160
CONTEXT = 73_728  # above the set's longest prompt (scripts/bench/quality_heldout.py CONTEXT)
REGISTRY_FILE = oracle.SRC / "tests/fixtures/lever_registry.json"
# cere_executor.h CERE_LEVER_* of the d9_batch implementations (a new one fails setUpClass until it is listed here)
LEVER_VALUES = {("k2_gemm", "small_m"): k8.CERE_LEVER_K2_SMALL_M,
                ("k2_gemm", "small_m_a8"): k8.CERE_LEVER_K2_SMALL_M_A8,
                ("prefill_attention", "p4"): p4.CERE_LEVER_PREFILL_P4,
                ("k2_weights", "fp8"): k8.CERE_LEVER_K2_WEIGHTS_FP8}
LOGIT_TILE = 32
NEGATIVE = qm.NEGATIVE
SENSITIVITY = qm.SENSITIVITY
INJ_TARGET = qm.INJ_TARGET
SELF_CONSISTENCY = qm.SELF_CONSISTENCY


class Seq(NamedTuple):
    pid: str
    part: str  # quality_measure.PARTS
    file: int  # the fixture's file index (-1: the long suite)
    dir: int  # the fixture's directory index (-1: the long suite)
    ids: list  # token ids (memory only)
    start: int  # s: the dec arm's PREFILL length; the scored rows are [start, len(ids) - 1)

    @property
    def rows(self) -> int:
        return len(self.ids) - 1 - self.start

    def targets(self) -> list:
        return [self.ids[r + 1] for r in range(self.start, len(self.ids) - 1)]


def candidate_sets(registry: dict) -> dict:
    """{name: slot choices} of `quality_measure.lever_sets` (every d9_batch implementation alone and every
    bundle of them, [REVISED: owner 2026-10-16 screen tiers])."""
    try:
        sets = qm.lever_sets(registry)
    except qm.QualityError as e:
        raise oracle.OracleError(f"{ROW}: {e}") from None
    missing = sorted({key for chosen in sets.values() for key in chosen} - set(LEVER_VALUES))
    if missing:
        raise oracle.OracleError(f"{ROW}: d9_batch implementations {missing} have no CERE_LEVER value here")
    return {name: {lever: LEVER_VALUES[(lever, impl)] for lever, impl in chosen} for name, chosen in sets.items()}


def special_ids() -> frozenset:
    """The tokenizer's special and added token ids (framing check; never reported)."""
    tok = k8.tokenizer()
    return frozenset(tok.get_added_vocab().values()) | frozenset(tok.all_special_ids)


def sequence(p: dict, special: frozenset) -> Seq:
    """One frame prompt as a Seq: the chat ids, then the continuation encoded without specials; counts checked."""
    ctx = k8.chat_ids(p["messages"])
    cont = list(k8.tokenizer().encode(p["cont"], add_special_tokens=False))
    if (len(ctx), len(cont)) != (p.get("ctx"), p.get("cont_tokens")):
        raise oracle.OracleError(f"{PROMPT_SET} {p['id']} encodes to {len(ctx)} + {len(cont)} tokens, the frame says "
                                 f"{p.get('ctx')} + {p.get('cont_tokens')}")
    if len(cont) < MIN_CONT or special.intersection(cont):
        raise oracle.OracleError(f"{PROMPT_SET} {p['id']}: continuation of {len(cont)} tokens (>= {MIN_CONT}) or a "
                                 "special token in it")
    ids = ctx + cont
    if not len(ids) < CONTEXT:
        raise oracle.OracleError(f"{PROMPT_SET} {p['id']}: {len(ids)} tokens, not below {CONTEXT}")
    return Seq(p["id"], p["part"], p["file"], p["dir"], ids, len(ids) - WINDOW)


def read_frame(special: frozenset) -> dict:
    """{lever_set, shape, set_sha256, mode, checkpoint, families, reference, reference_families, seqs} from the stdin
    frame (memory only); set_sha256 is the sha256 of the set's fixture file (native_dev.py); mode:
    `quality_measure.run_mode` of the frame's screen, lever set and prompts; checkpoint / families: the candidate
    checkpoint and its families file (`quality_measure.checkpoint_families`; None for a lever set); reference /
    reference_families: P's checkpoint and its families file (`quality_measure.reference_families`; None: the pinned)."""
    raw = sys.stdin.read()
    if not raw:
        raise oracle.OracleError(f"no {PROMPT_FRAME} frame on stdin (run through native_dev.py test --suite quality)")
    frame = json.loads(raw)
    prompts = frame.get("prompts")
    lever_set, shape, set_sha256, screen = (frame.get(k) for k in ("lever_set", "shape", "set_sha256", "screen"))
    keys = ("id", "part", "messages", "cont", "ctx", "cont_tokens", "file", "dir")
    if ((frame.get("frame"), frame.get("prompt_set"), frame.get("prereg_commit"))
            != (PROMPT_FRAME, PROMPT_SET, qm.PREREG_COMMIT) or shape not in qm.SHAPES
            or not (lever_set is None or isinstance(lever_set, str))
            or not (isinstance(set_sha256, str) and len(set_sha256) == 64)
            or not isinstance(prompts, list)
            or not all(isinstance(p, dict) and all(k in p for k in keys) and p["part"] in qm.PARTS for p in prompts)):
        raise oracle.OracleError(f"stdin frame {(frame.get('frame'), frame.get('prompt_set'))} is not "
                                 f"{(PROMPT_FRAME, PROMPT_SET)} at {qm.PREREG_COMMIT} with a shape, a lever_set, the "
                                 f"set's sha256 and prompts {keys}")
    # the long suite's count and the screen tier's main prompts (raises on any other frame)
    mode = qm.run_mode(screen, lever_set, [p["id"] for p in prompts if p["part"] == "main"],
                       sum(p["part"] == "long" for p in prompts))
    checkpoint = frame.get("checkpoint")
    families = qm.checkpoint_families(checkpoint, lever_set)
    reference = frame.get("reference")
    reference_families = qm.reference_families(reference)
    seqs = [sequence(p, special) for p in prompts]
    return {"lever_set": lever_set, "shape": shape, "set_sha256": set_sha256, "mode": mode, "checkpoint": checkpoint,
            "families": families, "reference": reference, "reference_families": reference_families, "seqs": seqs}


def candidate_checkpoint(families: str, tc: dict, weight: torch.Tensor) -> tuple[Path, Path]:
    """(directory, families file) of the frame's candidate checkpoint: the directory in CANDIDATE_DIR_ENV, no
    dev-loop share (it covers /model's files), a text config giving P's cere_qwen4_exp_config and an lm_head equal to
    P's `weight` (the suite's logits use P's lm_head)."""
    where = os.environ.get(qm.CANDIDATE_DIR_ENV)
    if not where or not Path(where).is_dir():
        raise oracle.OracleError(f"{qm.CANDIDATE_DIR_ENV}={where!r} is not the mounted candidate checkpoint "
                                 "(run through native_dev.py test --suite quality --candidate-checkpoint)")
    if oracle.dev_loop_share() is not None:
        raise oracle.OracleError("a candidate checkpoint run takes no dev-loop share (it covers /model's files)")
    out = (Path(where), oracle.SRC / families)
    with k8.checkpoint_dir(*out):
        if not torch.equal(k8.a4_config(oracle.text_config()), k8.a4_config(tc)):
            raise oracle.OracleError(f"{where}: its text config gives another cere_qwen4_exp_config than P's")
        if not torch.equal(oracle.real_tensor(b8.LM_HEAD_NAME, device=oracle.DEVICE), weight):
            raise oracle.OracleError(f"{where}: its {b8.LM_HEAD_NAME} differs from P's")
    return out


def bf16_partial_logprobs(rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """NEGATIVE: rows x weight^T with a BF16 accumulator (fp32 products summed per LOGIT_TILE-wide k tile, the running
    sum rounded to BF16 after every tile), log-softmax in fp64."""
    if torch.backends.cuda.matmul.allow_tf32:
        raise oracle.OracleError("TF32 matmul is enabled: the tile products must be fp32")
    x = rows.to(oracle.DEVICE).float()
    n, hidden = (int(v) for v in x.shape)
    tiles = hidden // LOGIT_TILE
    if tiles * LOGIT_TILE != hidden:
        raise oracle.OracleError(f"hidden {hidden} is not a multiple of {LOGIT_TILE}")
    xt = x.view(n, tiles, LOGIT_TILE).transpose(0, 1)  # [tiles, n, tile]
    vocab = int(weight.shape[0])
    logits = torch.empty((n, vocab), dtype=torch.float32, device=oracle.DEVICE)
    for s in range(0, vocab, b8.TF_KL_VOCAB_SLICE):
        w = weight[s:s + b8.TF_KL_VOCAB_SLICE].float()
        parts = torch.bmm(xt, w.view(-1, tiles, LOGIT_TILE).permute(1, 2, 0))  # [tiles, n, v] fp32
        acc = torch.zeros(parts.shape[1:], dtype=torch.bfloat16, device=oracle.DEVICE)
        for t in range(tiles):
            acc = (acc.float() + parts[t]).to(torch.bfloat16)
        logits[:, s:s + int(w.shape[0])] = acc.float()
        del w, parts, acc
    out = logits.double()
    del logits
    out -= torch.logsumexp(out, dim=-1, keepdim=True)
    return out


def bf16_logit_logprobs(rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """SENSITIVITY: D1's fp32 logits (tf_logprobs' product) rounded to BF16 before the fp64 log-softmax."""
    x = rows.to(oracle.DEVICE).float()
    vocab = int(weight.shape[0])
    logits = torch.empty((int(x.shape[0]), vocab), dtype=torch.float32, device=oracle.DEVICE)
    for s in range(0, vocab, b8.TF_KL_VOCAB_SLICE):
        logits[:, s:s + b8.TF_KL_VOCAB_SLICE] = x @ weight[s:s + b8.TF_KL_VOCAB_SLICE].float().T
    out = logits.to(torch.bfloat16).double()
    del logits
    out -= torch.logsumexp(out, dim=-1, keepdim=True)
    return out


def inject_target(lp: torch.Tensor, targets: torch.Tensor, c: float) -> torch.Tensor:
    """INJ_TARGET on fp64 log-probs [n, vocab] (in place): the target's probability scaled by e^-c, the other tokens
    renormalised to the rest, so -log p(target) grows by exactly c on every row."""
    at = targets.view(-1, 1)
    lt = lp.gather(1, at)
    others = lp.scatter(1, at, float("-inf"))
    rest = torch.logsumexp(others, dim=-1, keepdim=True)  # log of the non-target mass, without 1 - p cancellation
    lp += torch.log1p(-torch.exp(lt - c)) - rest
    lp.scatter_(1, at, lt - c)
    return lp


class InjectedLogprobs:
    """INJ_TARGET as `tf_kl_rows`' cand_logprobs for one sequence: `tf_logprobs` then `inject_target`, the chunks
    consumed in order (TF_KL_CHUNK rows at a time); `done` checks every row was seen."""

    def __init__(self, targets: list, c: float) -> None:
        self.targets, self.c, self.at = targets, c, 0

    def __call__(self, rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        n = int(rows.shape[0])
        chunk = self.targets[self.at:self.at + n]
        if len(chunk) != n or any(t is None for t in chunk):
            raise oracle.OracleError(f"{INJ_TARGET}: rows {self.at}..{self.at + n} have no target")
        self.at += n
        tgt = torch.tensor(chunk, dtype=torch.long, device=oracle.DEVICE)
        return inject_target(b8.tf_logprobs(rows, weight), tgt, self.c)

    def done(self) -> None:
        if self.at != len(self.targets):
            raise oracle.OracleError(f"{INJ_TARGET}: {self.at} of {len(self.targets)} rows injected")


def prefill_model_rows(handle: int, seq: Seq, *, hidden: int, layers: int) -> torch.Tensor:
    """The pre arm: `cere_ktest_forward_trace` of the whole sequence keeping its last seq.rows + 1 rows (k8's
    `forward_trace` buffers); the `model` rows [seq.rows, hidden] BF16 on the host."""
    rows = seq.rows + 1
    capacity = 4 * layers
    prompt = torch.tensor(seq.ids, dtype=torch.int32)
    names = torch.zeros(capacity * 128, dtype=torch.uint8)
    shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
    values = oracle.empty((capacity * rows * hidden,), torch.bfloat16)
    scores = torch.full((rows, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_trace", handle, prompt, len(seq.ids), 0, rows, names, names.numel(), shapes,
                capacity, values, values.numel() * values.element_size(), scores, counts)
    trace = k8.parse_trace("forward_trace", names, shapes, values, scores, counts, capacity)
    if "model" not in trace.values:
        raise oracle.OracleError(f"{ROW} {seq.pid}: the prefill trace lacks `model`")
    return trace.values["model"][:seq.rows].cpu()


def arm_rows(handle: int, seq: Seq, *, hidden: int, layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(pre, dec) `model` rows [seq.rows, hidden] BF16 on the host for one sequence on one model."""
    pre = prefill_model_rows(handle, seq, hidden=hidden, layers=layers)
    dec_trace = k8.forced_decode_trace(handle, seq.ids, seq.start, hidden=hidden, layers=layers)
    if len(dec_trace.steps) != seq.rows or any("model" not in step for step in dec_trace.steps):
        raise oracle.OracleError(f"{ROW} {seq.pid}: {len(dec_trace.steps)} decode steps with `model`, expected "
                                 f"{seq.rows}")
    dec = torch.cat([step["model"] for step in dec_trace.steps]).cpu()
    del dec_trace
    gc.collect()
    torch.cuda.empty_cache()
    if tuple(pre.shape) != (seq.rows, hidden) or tuple(dec.shape) != (seq.rows, hidden):
        raise oracle.OracleError(f"{ROW} {seq.pid}: rows {tuple(pre.shape)} / {tuple(dec.shape)}, expected "
                                 f"{(seq.rows, hidden)}")
    return pre, dec


class QualityGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.special = special_ids()
        frame = read_frame(cls.special)
        cls.lever_set, cls.shape, cls.set_sha256, cls.mode, cls.checkpoint, families, cls.seqs = (
            frame[k] for k in ("lever_set", "shape", "set_sha256", "mode", "checkpoint", "families", "seqs"))
        cls.reference = frame["reference"]
        cls.placement = contextlib.ExitStack()
        if frame["reference_families"] is not None:  # k2-prefill design note G4: P's placement is the reference's
            if oracle.dev_loop_share() is not None:
                raise oracle.OracleError("a reference checkpoint run takes no dev-loop share (it covers the pinned "
                                         "/model's files)")
            cls.placement.enter_context(k8.checkpoint_dir(oracle.MODEL_DIR, oracle.SRC / frame["reference_families"]))
        cls.tc = oracle.text_config()
        cls.hidden = int(cls.tc["hidden_size"])
        cls.layers = int(cls.tc["num_hidden_layers"])
        cls.weight = oracle.real_tensor(b8.LM_HEAD_NAME, device=oracle.DEVICE)
        cls.candidate = None
        # a subset run (the attribution rerun) measures P and the lever set only: no control, no long suite
        cls.sets = {} if cls.mode == qm.SUBSET_MODE else {SELF_CONSISTENCY: {}}
        cls.parts = ("main",) if cls.mode == qm.SUBSET_MODE else qm.PARTS
        if cls.lever_set is None:
            cls.level = None
        else:
            if (qm.SHAPE, qm.CLUSTER_LEVEL) == (None, None) or cls.shape != qm.SHAPE:
                raise oracle.OracleError(f"{ROW}: a candidate run needs the committed shape decision "
                                         f"(quality_measure.SHAPE {qm.SHAPE}, frame shape {cls.shape})")
            if cls.checkpoint is None:
                registry = candidate_sets(json.loads(REGISTRY_FILE.read_text()))
                if cls.lever_set not in registry:
                    raise oracle.OracleError(f"{ROW}: lever set {cls.lever_set!r} is not a quality lever set "
                                             f"({'|'.join(registry)})")
                cls.sets[cls.lever_set] = registry[cls.lever_set]
            else:  # fp4local design note FL-16: the checkpoint is the set, null levers
                cls.candidate = candidate_checkpoint(families, cls.tc, cls.weight)
                cls.sets[cls.lever_set] = {}
            cls.level = qm.CLUSTER_LEVEL
        cls.emitted = []

    @classmethod
    def tearDownClass(cls):
        cls.placement.close()
        del cls.weight
        gc.collect()
        torch.cuda.empty_cache()

    def emit(self, row: str, case: str, **fields) -> None:
        oracle.report_row(row, case, **fields)
        self.emitted.append({"row": row, "case": case, **fields})

    def model_rows(self, levers: dict, candidate: bool = False) -> dict:
        """{pid: (pre, dec)} of every sequence on a model with `levers` (null: every lever at the default); `candidate`:
        of the candidate checkpoint (`k8.checkpoint_dir`), else of /model."""
        with k8.checkpoint_dir(*self.candidate) if candidate else contextlib.nullcontext():
            return self._model_rows(levers)

    def _model_rows(self, levers: dict) -> dict:
        model = k8.create_model(CONTEXT, 1, model_levers=oracle.levers(**levers) if levers else None)
        try:
            k8.upload_and_seal(model)
            out = {s.pid: arm_rows(model.handle, s, hidden=self.hidden, layers=self.layers) for s in self.seqs}
        finally:
            k8.release_model(model.handle)
            gc.collect()
            torch.cuda.empty_cache()
        return out

    def stats(self, ref: dict, cand: dict, logprobs=None, inject: float | None = None) -> dict:
        """{(part, arm): [per-prompt RowStats lists]} against R = ref[pid][0]; `inject`: INJ_TARGET with that c."""
        out: dict = {(part, arm): [] for part in qm.PARTS for arm in qm.ARMS}
        for seq in self.seqs:
            for k, arm in enumerate(qm.ARMS):
                lp = logprobs
                if inject is not None:
                    lp = InjectedLogprobs(seq.targets(), inject)
                rows = b8.tf_kl_rows(ref[seq.pid][0], cand[seq.pid][k], self.weight, seq.targets(), cand_logprobs=lp)
                if inject is not None:
                    lp.done()
                out[(seq.part, arm)].append(rows)
        return out

    def test_d9q_quality(self):
        seqs = self.seqs
        main = [s for s in seqs if s.part == "main"]
        tokens = [len(s.ids) for s in seqs]
        self.emit(ROW, "corpus", prompt_set=PROMPT_SET, prereg_commit=qm.PREREG_COMMIT,
                  set_sha256=self.set_sha256, shape=self.shape, mode=self.mode, sequences=len(main),
                  long=len(seqs) - len(main), rows=sum(s.rows for s in seqs), tokens_min=min(tokens),
                  tokens_max=max(tokens), tokens_sum=sum(tokens), window=WINDOW, context=CONTEXT,
                  sets=len(self.sets), **qm.tier_corpus(self.mode, len(main)),
                  **({} if self.checkpoint is None else {"checkpoint": self.checkpoint}),
                  **({} if self.reference is None else {"reference": self.reference}))
        pred = self.model_rows({})
        p_stats = self.stats(pred, pred)
        for part in self.parts:
            ref_rows = [r for rows in p_stats[(part, "pre")] for r in rows]
            self.emit(ROW, f"framing {part}", **qm.framing(ref_rows, self.special))
        measured = {} if self.mode == qm.SUBSET_MODE else {
            NEGATIVE: self.stats(pred, pred, bf16_partial_logprobs),
            SENSITIVITY: self.stats(pred, pred, bf16_logit_logprobs),
            INJ_TARGET: self.stats(pred, pred, inject=qm.INJ_C)}
        for name, levers in self.sets.items():
            measured[name] = self.stats(pred, self.model_rows(levers, candidate=name == self.checkpoint))
        del pred
        gc.collect()
        arms_by_part: dict = {part: {} for part in self.parts}
        for name, st in measured.items():
            for part in self.parts:
                arms = {arm: qm.arm_input(st[(part, arm)], p_stats[(part, arm)]) for arm in qm.ARMS}
                arms_by_part[part][name] = arms
                for r in qm.report_rows(name, part, arms):
                    self.emit(r.pop("row"), r.pop("case"), **r)
        for part in self.parts:
            ps = [s for s in seqs if s.part == part]
            x = qm.null_values(p_stats[(part, "dec")])
            for r in qm.prompt_rows(part, [s.pid for s in ps], [s.file for s in ps], [s.dir for s in ps], x,
                                    arms_by_part[part]):
                self.emit(r.pop("row"), r.pop("case"), **r)
        if self.mode == qm.SUBSET_MODE:  # the attribution rerun: diagnostic only, no gate and no verdict row
            sub = qm.subset_check(self.emitted, self.set_sha256)
            print(f"{ROW} {sub['set']} on {len(sub['ids'])} prompts: mean dNLL "
                  + ", ".join(f"{k} {v:+.6f}" for k, v in sub["mean"].items()), file=sys.stderr)
            self.assertEqual(sub["problems"], [], f"{ROW} the run is invalid: " + "; ".join(sub["problems"]))
            return
        checked = qm.run_check(self.emitted, self.set_sha256, self.level)
        for name in measured:
            g = checked["gates"].get(name) or qm.gate(
                qm.gate_input(checked["cases"], checked["prompts"], name, "main"), checked["labels"],
                checked["files"], checked["dirs"])
            lg = checked["longs"].get(name) or qm.long_gate(
                qm.gate_input(checked["cases"], checked["prompts"], name, "long"))
            oracle.report_row(ROW, f"{name} verdict", level=checked["level"], gate_pass=g["pass"],
                              failures=len(g["failures"]), long_outcome=lg["outcome"],
                              **{f"{arm}_{k}": v for arm in qm.ARMS for k, v in g["arms"][arm].items()
                                 if isinstance(v, (int, float, bool, str))})
            print(f"{ROW} {name}: {'PASS' if g['pass'] else 'FAIL'} (long: {lg['outcome']}) "
                  + "; ".join(g["failures"]), file=sys.stderr)
        oracle.report_row(ROW, "controls", level=checked["level"], problems=len(checked["problems"]))
        # D9Q-b2v3 (7): a required control that misbehaves, or a failed framing check, makes the run invalid
        self.assertEqual(checked["problems"], [], f"{ROW} the run is invalid, no verdict (fix, record, rerun; never "
                         "retune): " + "; ".join(checked["problems"]))


if __name__ == "__main__":
    unittest.main()
