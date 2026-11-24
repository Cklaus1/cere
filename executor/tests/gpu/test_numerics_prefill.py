"""M2.2p prefill numerics (tasks/m22p-spec.md D1a, D2-D5; R1, R2, R4 rows): the native PREFILL of a prompt (n = P, the
served shape) against the reference's prompt-pass dumps of several frames, patch attribution and lever screens.

Run only by `CERE_LIVE=1 python3 scripts/live/native_dev.py numerics --suite prefill [--prompts ids] --frames N`:
after the last reference frame, the dev container (build at `/out`, model at `/model`, the pruned dumps at
`/refdump-prefill`, all read-only) runs `python3 -m unittest discover -s /src/executor/tests/gpu -p
'test_numerics_prefill.py' -v` once with the frame below on stdin. The model is created, uploaded and sealed exactly as
in `test_numerics_b8_layers.py` (its helpers are reused). The calculus (spread, limit, first-exceeding, excess, margin,
L*, attribution, screen ranking) lives in `prefill_measure.py`, covered on the CPU by `tests/test_prefill_measure.py`.

Stdin frame (`cere-prefill-v1`; in memory only):

    {"frame": "cere-prefill-v1",
     "dump_prompts": ["p01", ...],          # all 24 parity ids (served order): a dump dir per frame for each
     "prompts": ["p01", ...],               # the --prompts ids (the D4 per-tap profile set), a subset of them
     "frame_prompts": [[<cere-b8-v2 entry>, ...], ...],  # every frame's 24 entries, frame 1 first
     "prefill_dumps": ["/refdump-prefill/f1", ...],  # one dir per frame, frame 1 first
     "frames": N}                           # N >= 2 (D4's spread needs two frames)

Dump layout per frame dir, per dump prompt pid: `<dir>/<pid>/prompt.pt` (the chat completion's prompt prefill pass,
P rows), `<dir>/<pid>/after_prompt.pt` (the next pass, holding the prompt pass's `model` output), `forced.pt` (the
teacher-forced prefill, T rows) and `after_forced.pt` (its `model`; absent for the last served prompt, no pass follows
it). The forced passes give only SGLang's M-spread c, reported `unavailable` where absent. A pass must embed the
frame's prompt (or teacher-forced) rows bit for bit (`model.embed_tokens`).

Cases (report rows `M22P`; diagnostics, never a G1 gate):
* `test_a_locator` (D3(a), L1): per D3(a) name and B8 name, `found` (the dump holds it at every layer native emits it),
  `partial` or `absent` (not a failure); fails if a B8 name is not found or a found tensor's shape differs.
* `test_b_prefill_step` (D4, R1): per dump prompt (all 24) and tap of `cere_ktest_forward_trace_detail(detail 1)` in
  forward order (B8 and D3(a) names, `model`). Over all 24 (`measure_plan`): `last-row d` (row P-1, per prompt and
  name, limit from the row's own frame spread), `margin projection` (at `model`, row P-1, on the reference's top-2
  lm_head rows), `routed-set first divergence` (native `mlp.gate` vs the dumped `mlp.gate`, top-k =
  num_experts_per_tok). Over the profile set (`prompts`) only: d = rel-L2(native, frame N's prompt pass), e = max over
  frame pairs of the reference's own rel-L2, limit = `oracle.calibrated_limit(e, FLOOR_BLOCK)`; rows `prefill-step
  first-exceeding` (per prompt), `prefill-step per-name` (with max e and SGLang's M-spread c), `prefill-step excess`
  (p12/p13 over the other profiled prompts).
* `test_c_attribution` (D5, R2; p12 then p13 when dumped): margins m from `cere_ktest_forward_patch` with frame N's
  B8 block outputs: P0, P-all, P-prefix(0..48), P-class, P-single (every (layer, block); top 10 reported), L*, the
  attributed class.
* `test_d_screens` (D5 lever screens, R2): through `cere_ktest_forward_patch_screen`'s callback, P-mix (every HC mix
  output := SGLang's own GatedResidual (built as X1 builds it) running its torch.compile'd `_mix_compute` on native's
  normed input, `$S/layers/hyperconnection.py:195-211,230,281-288`; `fused_hc_mix` where SGLang dispatches it),
  P-gemm(plan) (each call of one K2 plan := the GEMM SGLang dispatches, `F.linear` of native's own input with reduced
  precision True; the GDN in_proj and shared gate_up calls are sliced from ONE fused GEMM,
  `test_prefill_isolation_x.dispatched_fused`), P-moe(tactic) (routed sum := `cutlass_fused_moe` at tactic -1 and at
  the bench's logged tactic read by `prefill_tactics.read_tactic`, `test_prefill_isolation_x.routed_reference`); dm
  on p12/p13, the first-token margin of all 24 prompts, the ranking.
* `test_e_first_token` (D1a, R4): per frame and every one of the 24 prompts, native's row P-1 margin m for that
  frame's reference greedy first token; m > 0 = picks it.

Token IDs and text stay in memory; rows carry names, layers, distances, margins and counts only.

Native entry points (PHooks, tasks/m22p-spec.md Interfaces; `oracle.call` appends the stream):

    int cere_ktest_forward_trace_detail(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                        uint64_t prefill_chunk_tokens, uint64_t last_rows, uint32_t detail,
                                        uint8_t *names /*host*/, uint64_t names_capacity,
                                        int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                        void *values /*device*/, uint64_t values_capacity_bytes,
                                        cere_token_score *scores /*host [rows]*/, uint64_t *out_counts /*host [3]*/,
                                        cudaStream_t stream);
        detail 1: the D3 names (kind (a) and `native.*` kind (b)) after each layer's B8 entries, forward order.
    int cere_ktest_forward_patch(uint64_t model, const uint32_t *tokens, uint64_t token_count, const uint8_t *names,
                                 uint64_t name_bytes, const void *values /*device BF16 [P, cols] per name*/,
                                 uint64_t values_bytes, uint32_t target_token, float *out_target_logit,
                                 cere_token_score *out_top1, cudaStream_t stream);
    int cere_ktest_forward_patch_screen(uint64_t model, const uint32_t *tokens, uint64_t token_count,
                                        const bf16 *(*patch)(void *ctx, const char *name, uint64_t layer,
                                                             uint64_t rows, uint64_t cols, const bf16 *native,
                                                             cudaStream_t stream),
                                        void *patch_ctx, uint32_t target_token, float *out_target_logit,
                                        cere_token_score *out_top1, cudaStream_t stream);
        Patch points (forward order): the four B8 block outputs, `native.layers.L.{attn,mlp}_mix.out`,
        `native.final_mix.out`, `native.layers.L.gemm.<plan>.out`; each preceded by its input point
        (`... .normed` for a mix, `native.layers.L.gemm.<plan>.in`, `native.layers.L.mlp.in`) whose return is ignored.
"""
from __future__ import annotations

import ctypes
import functools
import gc
import importlib
import json
import math
import statistics
import sys
import traceback
import unittest
from pathlib import Path
from typing import NamedTuple

import torch

import oracle

# Local helpers of this directory (on sys.path in the container); loaded by name: suite files import only stdlib,
# torch, safetensors, sglang, flashinfer and oracle (tests/test_native_oracles_static.py ALLOWED_TOP).
pm = importlib.import_module("prefill_measure")
b8 = importlib.import_module("test_numerics_b8_layers")

ROW = "M22P"
FRAME = "cere-prefill-v1"
DUMP_ROOT = "/refdump-prefill"
GPU_DIR = Path(__file__).resolve().parent
TACTICS_FILE = GPU_DIR / "prefill_isolation_tactics.json"
DETAIL_COLS_PER_LAYER = 1 << 17  # value capacity per layer and row of a detail trace (every D3/B8 tap fits)
DETAIL_ENTRIES_PER_LAYER = 32
SINGLES_REPORTED = 10
B8_CLASSES = ("self_attn.o_proj", "linear_attn.out_proj", "mlp.experts", "mlp.shared_expert.down_proj")
D3A = ("attn_hyper_connection.hc_norm", "mlp_hyper_connection.hc_norm", "mlp.gate", "mlp.shared_expert.gate_up_proj",
       "self_attn.qkv_proj", "linear_attn.in_proj_qkvz", "linear_attn.in_proj_ba", "linear_attn.norm")
FINAL_MIXER = "model.language_model.hyper_connection_mixer."
LM_HEAD = "lm_head.weight"
# SGLang's gated RMSNorm runs on [P * heads, head_dim] ($S/models/qwen3_5.py:736-747), so `linear_attn.norm` is dumped
# per head; native emits it per token [P, heads * head_dim] (model.h).
GDN_NORM = "linear_attn.norm"
GDN_NORM_COLS = 6144
# K2 plan (model.h) -> per call within a layer, the checkpoint weights (layer-relative, concatenated by rows) it runs.
# Only plans with patch points: the PLE GEMMs run inside K4 (model.h "Not points").
GEMM_WEIGHTS = {
    "attn_qkv": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),),
    "attn_o": (("self_attn.o_proj",),),
    "gdn_qkv": (("linear_attn.in_proj_qkv",),),
    "gdn_z": (("linear_attn.in_proj_z",),),
    "gdn_ba": (("linear_attn.in_proj_b",), ("linear_attn.in_proj_a",)),
    "gdn_out": (("linear_attn.out_proj",),),
    "moe_router": (("mlp.gate",),),
    "moe_shared_gate_up": (("mlp.shared_expert.gate_proj",), ("mlp.shared_expert.up_proj",)),
    "moe_shared_down": (("mlp.shared_expert.down_proj",),),
}
# The plans whose calls SGLang runs as ONE fused dispatched GEMM (Z2A1-02); the fused weight and each call's columns
# come from `test_prefill_isolation_x.dispatched_fused` (shared with X2). GDN: `finalize_fused_in_proj` stacks
# [qkv; z; b; a] (n 16,480) for M <= FUSED_IN_PROJ_MAX_M, above which in_proj_qkvz [qkv; z] and in_proj_ba [b; a] run
# apart as row views of it ($S/models/qwen3_5.py:507-531, 560-613). Shared expert: one [gate; up] GEMM (n 1,280,
# $S/models/qwen2_moe.py:185-214).
FUSED_PLANS = ("gdn_qkv", "gdn_z", "gdn_ba", "moe_shared_gate_up")
FUSED_PARTS = {"gdn_ba": ("b", "a"), "moe_shared_gate_up": ("gate", "up")}  # native call order per layer (PHooks)
INPUT_SUFFIXES = (".normed", ".in")

PATCH_FN = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64,
                            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)


# -- the stdin frame -----------------------------------------------------------------------------------------------


class Frame(NamedTuple):
    prompts: tuple  # the last frame's b8.Prompt entries, served order (== dump_ids order)
    frame_prompts: tuple  # per frame: pid -> b8.Prompt
    dump_ids: tuple  # all 24 parity ids, served order (the --prompts ids first)
    profile_ids: tuple  # the --prompts ids: D4's per-tap profile set
    dumps: tuple  # per frame: Path
    frames: int


def parse_prompts(items, vocab: int, parity: list, what: str) -> tuple:
    """One frame's 24 entries, validated as the B8 frame's (`test_numerics_b8_layers.read_frame`)."""
    if not isinstance(items, list):
        raise oracle.OracleError(f"frame {what} is not a list")
    prompts = []
    for item in items:
        pid = item.get("id") if isinstance(item, dict) else None
        if not isinstance(pid, str):
            raise oracle.OracleError(f"frame {what}: prompt without a string id")
        prompt_ids = b8.token_list(item.get("prompt_ids"), vocab, f"{what} {pid} prompt_ids")
        completion_ids = b8.token_list(item.get("completion_ids"), vocab, f"{what} {pid} completion_ids")
        if not prompt_ids or not completion_ids:
            raise oracle.OracleError(f"frame {what} {pid}: empty prompt or completion")
        total = len(prompt_ids) + len(completion_ids)
        top2 = item.get("top2")
        if not (isinstance(top2, list) and len(top2) == total - 1):
            raise oracle.OracleError(f"frame {what} {pid}: top2 has no entry per position 0..T-2 (T = {total})")
        if total > b8.MAX_CONTEXT:
            raise oracle.OracleError(f"frame {what} {pid}: {total} tokens exceed the {b8.MAX_CONTEXT}-token context")
        entries = [b8.top2_entry(e, vocab, f"{what} {pid} top2[{k}]") for k, e in enumerate(top2)]
        prompts.append(b8.Prompt(pid, prompt_ids, completion_ids, entries))
    order = [p.pid for p in prompts]
    if sorted(order) != sorted(parity):
        raise oracle.OracleError(f"frame {what}: prompts {sorted(order)} are not the {len(parity)} parity prompts")
    return tuple(prompts)


def parse_frame(doc, vocab: int, parity: list) -> Frame:
    """The `cere-prefill-v1` document, validated (ids stay in memory; messages name prompts and fields only)."""
    if not (isinstance(doc, dict) and doc.get("frame") == FRAME):
        raise oracle.OracleError(f"stdin is not a {FRAME} frame")
    frames = doc.get("frames")
    if type(frames) is not int or frames < 2:
        raise oracle.OracleError(f"frame frames {frames!r} is not an int >= 2 (D4's spread needs two frames)")
    per_frame = doc.get("frame_prompts")
    if not (isinstance(per_frame, list) and len(per_frame) == frames):
        raise oracle.OracleError(f"frame frame_prompts does not hold {frames} frames")
    served = [parse_prompts(items, vocab, parity, f"frame_prompts[{k}]") for k, items in enumerate(per_frame)]
    frame_prompts = tuple({p.pid: p for p in items} for items in served)
    dumps = doc.get("prefill_dumps")
    if not (isinstance(dumps, list) and len(dumps) == frames and all(isinstance(d, str) for d in dumps)
            and len(set(dumps)) == frames and all(Path(d).parent == Path(DUMP_ROOT) for d in dumps)):
        raise oracle.OracleError(f"frame prefill_dumps is not {frames} distinct dirs under {DUMP_ROOT}")
    dump_ids = doc.get("dump_prompts")
    if not (isinstance(dump_ids, list) and all(isinstance(i, str) for i in dump_ids)
            and len(set(dump_ids)) == len(dump_ids) == len(parity) and set(dump_ids) == set(parity)):
        raise oracle.OracleError(f"frame dump_prompts is not the {len(parity)} distinct parity ids")
    if [p.pid for p in served[-1]] != dump_ids:
        raise oracle.OracleError("frame dump_prompts is not the last frame's served order")
    profile = doc.get("prompts")
    if not (isinstance(profile, list) and profile and all(isinstance(i, str) for i in profile)
            and len(set(profile)) == len(profile) and dump_ids[:len(profile)] == profile):
        raise oracle.OracleError("frame prompts is not a non-empty list of the first-served dump_prompts ids")
    return Frame(tuple(served[-1]), frame_prompts, tuple(dump_ids), tuple(profile),
                 tuple(Path(d) for d in dumps), frames)


def measure_plan(frame: Frame) -> list:
    """(pid, profiled) for every dumped prompt in served order: D4's last-row, margin-projection and routed-set rows
    cover all of them; the per-tap profile (first-exceeding, per-name, excess) only the --prompts ids."""
    profile = set(frame.profile_ids)
    return [(pid, pid in profile) for pid in frame.dump_ids]


@functools.cache
def read_frame() -> Frame:
    raw = sys.stdin.buffer.read()
    if not raw:
        raise oracle.OracleError(f"no {FRAME} frame on stdin (run through native_dev.py numerics --suite prefill)")
    return parse_frame(json.loads(raw.decode("utf-8")), int(oracle.text_config()["vocab_size"]), b8.parity_ids())


# -- names --------------------------------------------------------------------------------------------------------


def tap_key(name: str) -> tuple:
    """(layer, layer-free name): `model.layers.L.rest` -> (L, rest); `native.layers.L.rest` -> (L, native.rest);
    anything else (`model`, `native.final_mix.*`) -> (-1, name)."""
    parts = name.split(".", 3)
    if len(parts) == 4 and parts[1] == "layers" and parts[2].isdigit():
        return int(parts[2]), ("native." + parts[3]) if parts[0] == "native" else parts[3]
    return -1, name


def full_name(key: tuple) -> str:
    layer, name = key
    return name if layer < 0 else f"model.layers.{layer}.{name}"


def b8_block_keys(tc: dict) -> list:
    return [tap_key(n) for n in b8.b8_names(tc) if n != "model"]


# -- the reference dump -------------------------------------------------------------------------------------------


class RefPass(NamedTuple):
    prompt: dict
    after: dict
    forced: dict | None
    after_forced: dict | None


def load_ref(dump_dir: Path, p, embed: torch.Tensor, frame: int) -> RefPass:
    """A frame's prompt pass of p (and the teacher-forced pass when kept); the prompt pass must embed p's prompt."""
    base = dump_dir / p.pid
    files = {k: base / f"{k}.pt" for k in ("prompt", "after_prompt", "forced", "after_forced")}
    for k in ("prompt", "after_prompt"):
        if not files[k].is_file():
            raise oracle.OracleError(f"frame {frame} {p.pid}: {k}.pt is missing")
    prompt = b8.load_pass(files["prompt"])
    if not b8.embeds_match(prompt, embed, p.prompt_ids):
        raise oracle.OracleError(f"frame {frame} {p.pid}: prompt.pt does not embed the prompt rows")
    forced = after_forced = None  # after_forced.pt is absent for the last served prompt (no pass follows it)
    if files["forced"].is_file():
        forced = b8.load_pass(files["forced"])
        if not b8.embeds_match(forced, embed, p.ids):
            raise oracle.OracleError(f"frame {frame} {p.pid}: forced.pt does not embed the teacher-forced rows")
        if files["after_forced"].is_file():
            after_forced = b8.load_pass(files["after_forced"])
    return RefPass(prompt, b8.load_pass(files["after_prompt"]), forced, after_forced)


def ref_value(passes: RefPass, key: tuple, forced: bool = False):
    """The dumped tensor of a tap (None when absent): `model` is read one pass late."""
    if forced:
        dump = passes.after_forced if key == (-1, "model") else passes.forced
    else:
        dump = passes.after if key == (-1, "model") else passes.prompt
    if dump is None:
        return None
    if key == (-1, "model"):
        value = dump.get("model")
    else:
        dk = b8.dump_key(dump, full_name(key))
        value = None if dk is None else dump[dk]
    if not isinstance(value, torch.Tensor):
        return None
    return value.reshape(-1, GDN_NORM_COLS) if key[1] == GDN_NORM else value


# -- native entry points ------------------------------------------------------------------------------------------


def parse_entries(fn_name: str, names, shapes, values, counts, capacity: int) -> dict:
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 <= entries <= capacity and 0 <= name_bytes <= names.numel()
            and 0 <= value_bytes <= values.numel() * values.element_size()):
        raise oracle.OracleError(f"{fn_name} counts {entries}, {name_bytes}, {value_bytes} exceed capacities")
    emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"{fn_name} wrote {len(emitted) - 1} names for {entries} entries")
    out: dict = {}
    element = 0
    for i in range(entries):
        name = emitted[i].decode()
        ndim, r, c = (int(v) for v in shapes[i].tolist())
        if ndim != 2 or r < 0 or c < 0:
            raise oracle.OracleError(f"{fn_name} entry {name}: shape record {(ndim, r, c)}")
        if name in out:
            raise oracle.OracleError(f"{fn_name} emitted {name} twice")
        out[name] = values[element:element + r * c].view(r, c)
        element += r * c
    if element * values.element_size() != value_bytes:
        raise oracle.OracleError(f"{fn_name} value bytes {value_bytes} != the entries' {element * 2}")
    return out


def trace_detail(handle: int, ids: list, layers: int) -> dict:
    """Native prefill of `ids` (n = P, chunk 0, every row) with the D3 detail taps: name -> BF16 [P, cols] on GPU0,
    in forward order."""
    rows = len(ids)
    capacity = DETAIL_ENTRIES_PER_LAYER * layers + 8
    prompt = torch.tensor(ids, dtype=torch.int32)
    names = torch.zeros(capacity * 128, dtype=torch.uint8)
    shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
    values = oracle.empty((rows * layers * DETAIL_COLS_PER_LAYER,), torch.bfloat16)
    scores = torch.full((rows, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_trace_detail", handle, prompt, rows, 0, 0, 1, names, names.numel(), shapes,
                capacity, values, values.numel() * values.element_size(), scores, counts)
    return parse_entries("forward_trace_detail", names, shapes, values, counts, capacity)


def score_tuple(top1: torch.Tensor) -> tuple:
    """cere_token_score int32[4] -> (id, runner_up, logit, runner_up_logit)."""
    ids = top1[:2].tolist()
    logits = top1[2:4].contiguous().view(torch.float32).tolist()
    return ids[0], ids[1], logits[0], logits[1]


def forward_patch(handle: int, ids: list, target: int, patches: list) -> float:
    """Signed margin m of `target` at row P-1 with each (name, BF16 [P, cols]) block output replaced."""
    if not patches:
        return screen_margin(handle, ids, target, null_screen())
    blob = b"".join(n.encode() + b"\0" for n, _ in patches)
    names = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    values = torch.cat([v.to(oracle.DEVICE, torch.bfloat16).reshape(-1) for _, v in patches]).contiguous()
    tokens = torch.tensor(ids, dtype=torch.int32)
    logit = torch.full((1,), math.nan, dtype=torch.float32)
    top1 = torch.full((4,), -1, dtype=torch.int32)
    oracle.call("cere_ktest_forward_patch", handle, tokens, len(ids), names, names.numel(), values,
                values.numel() * values.element_size(), target, logit, top1)
    return pm.signed_margin(float(logit[0]), score_tuple(top1))


class _DeviceView:
    """`__cuda_array_interface__` over a native device pointer (int16 words; viewed as BF16)."""

    def __init__(self, ptr: int, rows: int, cols: int):
        self.__cuda_array_interface__ = {"shape": (rows, cols), "typestr": "<i2", "data": (ptr, False),
                                         "version": 3, "strides": None}


def device_view(ptr: int, rows: int, cols: int) -> torch.Tensor:
    return torch.as_tensor(_DeviceView(ptr, rows, cols), device=oracle.DEVICE).view(torch.bfloat16)


class Screen:
    """One `forward_patch_screen` callback: input points are copied into `inputs`; every other patch point goes to
    `replace(screen, name, layer, native)`, which returns None for a point it does not own, a BF16 replacement, or
    raises when it owns the point but cannot compute it. Replacements live until the forward ends. Fail closed: a
    Python exception cannot cross ctypes, and a raising ctypes callback does NOT return null (the return slot is left
    unwritten, so native would read a garbage pointer), so `deliver` catches it into `error` and returns null; from
    then on every point returns null without computing anything (the forward finishes natively, nothing else runs)
    and `check` raises it as an OracleError naming the
    screen (a GPU OOM included: never a retry or a CPU fallback). `check` also fails a screen that `expects`
    replacements and made none."""

    def __init__(self, replace, expects: bool = True, label: str = "screen"):
        self.replace = replace
        self.expects = expects
        self.label = label
        self.inputs: dict = {}
        self.keep: list = []
        self.replaced = 0
        self.error = None
        self.fn = PATCH_FN(self._callback)

    def _callback(self, ctx, name, layer, rows, cols, native, stream):
        return self.deliver(name, layer, rows, cols, native, stream)

    def deliver(self, name: bytes, layer: int, rows: int, cols: int, native: int, stream: int):
        """The ctypes callback body: a pointer to the replacement, or null. The only broad handler of the suite
        (tests/test_native_oracles_static.py allows exactly this form): store the error, return null; `check`
        re-raises it after the forward and the forward's result is discarded."""
        if self.error is not None:
            return None
        try:
            with torch.cuda.stream(torch.cuda.ExternalStream(stream or 0, device=oracle.DEVICE)):
                return self._patch(name.decode(), int(layer), rows, cols, device_view(native, rows, cols))
        except BaseException as e:
            self.error = e
            return None

    def _patch(self, name: str, layer: int, rows: int, cols: int, value):
        if name.endswith(INPUT_SUFFIXES):
            self.inputs[name] = value.clone()
            return None
        out = self.replace(self, name, layer, value)
        if out is None:
            return None
        out = _as_patch(out, rows, cols, name)
        self.keep.append(out)
        self.replaced += 1
        return _out_ptr(out)

    def check(self) -> None:
        if self.error is not None:
            # Screen -> error -> traceback -> deliver's frame -> Screen is a reference cycle that keeps every tensor
            # the failing callback's frames hold (P-moe's weights, `keep`, `inputs`) resident until a gc pass; the
            # next test then OOMs (M2.2p R4 attempt 2: test_e at 18.85 GiB). The callback's frames are finished:
            # drop their locals (the report keeps their lines).
            traceback.clear_frames(self.error.__traceback__)
            raise oracle.OracleError(f"screen {self.label}: callback raised {type(self.error).__name__}: "
                                     f"{self.error}") from self.error
        if self.expects and not self.replaced:
            raise oracle.OracleError(f"screen {self.label} found none of its patch points (patch point names changed?)")


def _as_patch(out, rows: int, cols: int, name: str):
    out = out.to(torch.bfloat16).contiguous()
    if tuple(out.shape) != (rows, cols):
        raise oracle.OracleError(f"{name}: replacement shape {tuple(out.shape)} != {(rows, cols)}")
    return out


def _out_ptr(out) -> int:
    return out.data_ptr()


def _bits_equal(a, b) -> bool:
    return bool(torch.equal(a, b))


def keep_native(screen: Screen, name: str, layer: int, native: torch.Tensor):
    return None


def null_screen() -> Screen:
    """No replacement (P0 and the first-token rows)."""
    return Screen(keep_native, expects=False, label="null")


def screen_margin(handle: int, ids: list, target: int, screen: Screen) -> float:
    """Signed margin m of `target` with `screen`'s callback as ForwardTaps.patch."""
    fn = oracle.ktest("cere_ktest_forward_patch_screen")
    tokens = torch.tensor(ids, dtype=torch.int32)
    logit = torch.full((1,), math.nan, dtype=torch.float32)
    top1 = torch.full((4,), -1, dtype=torch.int32)
    stream = torch.cuda.current_stream(oracle.DEVICE)
    rc = fn(ctypes.c_uint64(handle), ctypes.c_void_p(tokens.data_ptr()), ctypes.c_uint64(len(ids)), screen.fn,
            ctypes.c_void_p(0), ctypes.c_uint32(target), ctypes.c_void_p(logit.data_ptr()),
            ctypes.c_void_p(top1.data_ptr()), ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    screen.keep.clear()
    screen.check()
    if rc != 0:
        raise oracle.OracleError(f"screen {screen.label}: cere_ktest_forward_patch_screen returned status {rc}")
    return pm.signed_margin(float(logit[0]), score_tuple(top1))


# -- lever screens (D5) -------------------------------------------------------------------------------------------


@functools.cache
def hc_module(prefix: str):
    """SGLang's own GatedResidual for the checkpoint prefix (as X1 builds it: `test_kernels_k3_hc._gated_residual`),
    whose `_mix_compute` is the torch.compile'd function SGLang dispatches."""
    return importlib.import_module("test_kernels_k3_hc")._gated_residual(prefix, use_combine=False)


def mix_path(module, normed: torch.Tensor) -> str:
    """The branch `GatedResidual.mix` takes for `normed` ($S/layers/hyperconnection.py:249-288; X1's `path`)."""
    from sglang.srt.layers.hc_mix_triton import fused_hc_mix_supported
    if (module._jit_mix_ok and normed.is_cuda and normed.dtype in (torch.bfloat16, torch.float16)
            and normed.shape[0] <= 24):
        return "cute_hc_mix"
    if fused_hc_mix_supported(normed, module.input_mix_weight_down.weight, module.input_mix_weight_up.weight):
        return "triton_fused_hc_mix"
    return "compiled_mix_compute"


def mix_prefix(name: str) -> str:
    """Checkpoint prefix of the GatedResidual behind a mix patch point."""
    if name == "native.final_mix.out":
        return FINAL_MIXER
    layer, rest = tap_key(name)
    which = {"native.attn_mix.out": "attn", "native.mlp_mix.out": "mlp"}[rest]
    return oracle.layer_prefix(layer) + f"{which}_hyper_connection."


class Weights:
    """Real BF16 weights on GPU0, cached for one screen."""

    def __init__(self):
        self.cache: dict = {}

    def get(self, names: tuple) -> torch.Tensor:
        if names not in self.cache:
            parts = [oracle.real_tensor(n + ".weight") for n in names]
            self.cache[names] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0).contiguous()
        return self.cache[names]

    def fused(self, layer: int, plan: str) -> tuple:
        """`dispatched_fused(layer, plan)` (which re-reads the checkpoint on every call), read once per fused weight."""
        key = ("fused", layer, "gdn" if plan.startswith("gdn_") else "shared")
        if key not in self.cache:
            self.cache[key] = isolation().dispatched_fused(layer, plan)
        return self.cache[key]


def mix_screen():
    """P-mix: every HC mix output := SGLang's dispatched mix of native's own normed input, computed by the module's
    compiled `_mix_compute` (or `fused_hc_mix` where SGLang dispatches it); the CuTe path (sm_100 only) is refused."""
    def replace(screen: Screen, name: str, layer: int, native: torch.Tensor):
        if not (name == "native.final_mix.out" or name.endswith("_mix.out") and name.startswith("native.layers.")):
            return None
        normed = screen.inputs.get(name[:-len(".out")] + ".normed")
        if normed is None:
            raise oracle.OracleError(f"{name}: its .normed input point was not delivered")
        module = hc_module(mix_prefix(name))
        down, up = module.input_mix_weight_down.weight, module.input_mix_weight_up.weight
        path = mix_path(module, normed)
        with torch.no_grad():
            if path == "compiled_mix_compute":
                out = module._mix_compute(normed, down, up, module.hc_count, module.hidden_size)
            elif path == "triton_fused_hc_mix":
                from sglang.srt.layers.hc_mix_triton import fused_hc_mix
                out = fused_hc_mix(normed, down, up, module.hc_count, module.hidden_size)
            else:
                raise oracle.OracleError(f"{name}: SGLang dispatches {path} at M={normed.shape[0]}; P-mix does not "
                                         "mirror it")
        return out.to(torch.bfloat16)

    return replace


def dispatched_columns(plan: str, occurrence: int, m: int, cols: dict, fused_max_m: int) -> tuple:
    """For a native call of a FUSED_PLANS plan (the `occurrence`-th in its layer): ((row_lo, row_hi) of the fused
    weight SGLang's one GEMM uses at M = m, (col_lo, col_hi) of the native call's columns in that GEMM's output).
    `cols` is `dispatched_fused`'s column map over the whole fused weight."""
    parts = FUSED_PARTS.get(plan)
    a, b = cols[plan if parts is None else f"{plan}.{parts[occurrence % len(parts)]}"]
    if plan.startswith("gdn_") and m > fused_max_m:
        rows = (cols["gdn_ba.b"][0], cols["gdn_ba.a"][1]) if plan == "gdn_ba" else (0, cols["gdn_z"][1])
    else:
        rows = (0, max(hi for _, hi in cols.values()))
    return rows, (a - rows[0], b - rows[0])


def gemm_screen(plan: str, weights: Weights):
    """P-gemm(plan): every call of K2 `plan` := the GEMM SGLang dispatches for it, `F.linear` (reduced precision
    True, torch's default) of native's own input; for FUSED_PLANS the one fused GEMM, sliced to the call's columns."""
    calls: dict = {}

    def replace(screen: Screen, name: str, layer: int, native: torch.Tensor):
        if not name.endswith(f".gemm.{plan}.out"):
            return None
        x = screen.inputs.get(name[:-len(".out")] + ".in")
        if x is None:
            raise oracle.OracleError(f"{name}: its .in input point was not delivered")
        occurrence = calls.get(name, 0)
        calls[name] = occurrence + 1
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # torch's default (F7)
        if plan in FUSED_PLANS:
            iso = isolation()
            w, cols = weights.fused(layer, plan)
            (lo, hi), (a, b) = dispatched_columns(plan, occurrence, x.shape[0], cols, iso.FUSED_IN_PROJ_MAX_M)
            return torch.nn.functional.linear(x, w[lo:hi])[:, a:b].contiguous()
        prefix = oracle.layer_prefix(layer)
        w = weights.get(tuple(prefix + n for n in GEMM_WEIGHTS[plan][occurrence % len(GEMM_WEIGHTS[plan])]))
        return torch.nn.functional.linear(x, w)

    return replace


@functools.cache
def isolation():
    """PIsolate's helper module (`routed_reference(layer, x, tactic)`, `dispatched_fused(layer, plan)`), loaded by
    name like the others."""
    return importlib.import_module("test_prefill_isolation_x")


@functools.cache
def tactics():
    """PHarness's shared tactics-file reader (`prefill_tactics.read_tactic`: one parser and one bucket rule)."""
    return importlib.import_module("prefill_tactics")


def moe_margins(prompts, tactic_of, layers: int, run, reference, release, label: str) -> dict:
    """P-moe: every layer's routed sum := `reference(layer, x, tactic)` (PIsolate's `routed_reference`) of native's
    own `native.layers.L.mlp.in`, computed LAYER-MAJOR so each MoE layer's weights are loaded once per screen (a
    prompt-major loop reloads all 48 layers per prompt: R1 attempt 2 ran 27 min and OOMed). Pass L: `release()` drops
    the previous layer's weights, then each prompt's forward replaces layers < L with the replacements fixed in
    earlier passes (their inputs must be bit-equal to the ones recorded then, else the forward is not deterministic
    and the screen fails), computes layer L and leaves layers > L native. The last pass replaces every layer, and its
    margin is the screen's. `run(p, screen)` -> margin; returns {p: margin}."""
    fixed = {p: {} for p in prompts}  # layer -> (input, replacement)
    margins: dict = {}
    try:
        for current in range(layers):
            release()
            for p in prompts:
                def replace(screen: Screen, name: str, layer: int, native, p=p, current=current):
                    if not (name.startswith("model.layers.") and name.endswith(".mlp.experts")) or layer > current:
                        return None
                    x = screen.inputs.get(f"native.layers.{layer}.mlp.in")
                    if x is None:
                        raise oracle.OracleError(f"{name}: native.layers.{layer}.mlp.in was not delivered")
                    if layer < current:
                        seen, out = fixed[p][layer]
                        if not _bits_equal(seen, x):
                            raise oracle.OracleError(f"layer {layer} mlp.in differs from the earlier pass "
                                                     "(native forward not deterministic)")
                        return out
                    out = reference(layer, x, tactic_of(p))
                    fixed[p][layer] = (x, out)
                    return out
                margins[p] = run(p, Screen(replace, label=label))
    finally:
        # A failing screen (e.g. an OOM inside the callback) must not leave its replacements and MoE buffers
        # resident: the later tests (test_e's P-1 rows) run on the same GPU (M2.2p R4 attempt 1 OOMed there).
        fixed.clear()
        release()
    return margins


def release_screen_memory(weights: "Weights") -> None:
    """Bound GPU residency between screens: drop the screen's cached weights, PIsolate's MoE buffers and caches
    (`release_moe`), and return torch's freed blocks to the driver."""
    weights.cache.clear()
    isolation().release_moe()
    torch.cuda.empty_cache()


def bench_tactic(prompt_tokens: int) -> tuple:
    """(tactic, status) of the bench's logged MoE tactic for a P-token prefill: tactic -1 or (gemm1, gemm2) when
    status is "logged", else None with status "unavailable[: why]"."""
    return tactics().read_tactic(TACTICS_FILE, prompt_tokens)


def target_of(p) -> int:
    """The reference's greedy first token of a frame entry (in memory only)."""
    return p.completion_ids[0]


# -- the suite ----------------------------------------------------------------------------------------------------


class PrefillNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.hidden = int(cls.tc["hidden_size"])
        cls.layers = int(cls.tc["num_hidden_layers"])
        cls.frame = read_frame()
        cls.by_pid = {p.pid: p for p in cls.frame.prompts}
        model = b8.create_model(b8.MAX_CONTEXT)
        cls.addClassCleanup(b8.release_model, model.handle)
        b8.upload_and_seal(model)
        cls.handle = model.handle
        cls.embed = oracle.real_tensor(b8.EMBED_NAME, device=None)

    def tearDown(self):
        # A failed test's leftovers (any reference cycle its traceback made) must not stay resident for the next
        # test on the same GPU: collect them and return torch's freed blocks to the driver.
        gc.collect()
        torch.cuda.empty_cache()

    def dump_prompts(self) -> list:
        return [self.by_pid[pid] for pid in self.frame.dump_ids]

    def refs(self, p) -> list:
        """Every frame's RefPass of dump prompt p (frame 1 first)."""
        return [load_ref(d, self.frame.frame_prompts[k][p.pid], self.embed, k + 1)
                for k, d in enumerate(self.frame.dumps)]

    def test_a_locator(self):
        oracle.report_row(ROW, "layer prompts", prompts=",".join(self.frame.dump_ids))
        p = self.dump_prompts()[0]
        native = trace_detail(self.handle, p.prompt_ids, self.layers)
        last = self.refs(p)[-1]
        b8_suffixes = {k[1] for k in b8_block_keys(self.tc)} | {"model"}
        status: dict = {}  # suffix -> [found layers, absent layers]
        shape_errors = []
        for name, value in native.items():
            key = tap_key(name)
            if key[1].startswith("native.") or (key[1] not in D3A and key[1] not in b8_suffixes):
                continue
            ref = ref_value(last, key)
            counts = status.setdefault(key[1], [0, 0])
            counts[ref is None] += 1
            if ref is not None and tuple(ref.shape) != tuple(value.shape):
                shape_errors.append(f"{name} dump {tuple(ref.shape)} vs native {tuple(value.shape)}")
        for suffix in D3A:
            status.setdefault(suffix, [0, 0])
        missing_b8 = []
        for suffix, (found, absent) in status.items():
            state = "absent" if not found else "found" if not absent else "partial"
            oracle.report_row(ROW, "locator", name=suffix, status=state, layers_found=found, layers_absent=absent,
                              kind="b8" if suffix in b8_suffixes else "d3a")
            if suffix in b8_suffixes and state != "found":
                missing_b8.append(suffix)
        del native
        self.assertFalse(missing_b8, f"{ROW} locator: B8 names not found in the dump: {missing_b8}")
        self.assertFalse(shape_errors, f"{ROW} locator: shape mismatches: {shape_errors[:3]}")

    def test_b_prefill_step(self):
        top_k = int(self.tc["num_experts_per_tok"])
        order: list = []  # (layer, name, pid) in forward order, prompt by prompt
        d: dict = {}
        limits: dict = {}
        spread: dict = {}  # name -> [max e, max c or None]
        ratios: dict = {}  # pid -> {(layer, name): d/limit}
        last: dict = {}  # pid -> {(layer, name): (d, limit)}
        lm_head = oracle.real_tensor(LM_HEAD, device=None)
        for pid, profiled in measure_plan(self.frame):
            p = self.by_pid[pid]
            refs = self.refs(p)
            native = trace_detail(self.handle, p.prompt_ids, self.layers)
            rows = len(p.prompt_ids)
            last[p.pid] = {}
            if profiled:
                ratios[p.pid] = {}
            routed = []
            first_order = []
            for name, value in native.items():
                key = tap_key(name)
                if key[1].startswith("native."):
                    continue
                full = (key[0], key[1], p.pid)
                frames = [ref_value(r, key) for r in refs]
                present = not any(f is None or tuple(f.shape) != tuple(value.shape) for f in frames)
                if profiled:
                    first_order.append(full)
                    order.append(full)
                    if not present:
                        d[full] = limits[full] = None
                        ratios[p.pid][key] = None
                if not present:
                    continue
                frames = [f.to(oracle.DEVICE) for f in frames]
                if profiled:
                    e = pm.pairwise_spread(frames, dist=oracle.rel_l2)
                    limit = pm.limit(e)
                    dist = oracle.rel_l2(value, frames[-1])
                    d[full], limits[full] = dist, limit
                    ratios[p.pid][key] = dist / limit
                    forced = ref_value(refs[-1], key, forced=True)
                    c = oracle.rel_l2(forced.to(oracle.DEVICE)[:rows], frames[-1]) if forced is not None else None
                    entry = spread.setdefault(key[1], [0.0, None])
                    entry[0] = max(entry[0], e)
                    if c is not None:
                        entry[1] = c if entry[1] is None else max(entry[1], c)
                e_last = pm.pairwise_spread([f[-1] for f in frames], dist=oracle.rel_l2)
                last[p.pid][key] = (pm.last_row_d(value, frames[-1], dist=oracle.rel_l2), pm.limit(e_last))
                if key[1] == "mlp.gate":
                    routed.append((key[0], value.float().tolist(), frames[-1].float().tolist()))
                if key == (-1, "model"):
                    (a, _), (b, _) = p.top2[rows - 1]
                    projection = pm.margin_projection(value[-1].float().tolist(), frames[-1][-1].float().tolist(),
                                                      lm_head[a].float().tolist(), lm_head[b].float().tolist())
                    (_, lp_a), (_, lp_b) = p.top2[rows - 1]
                    oracle.report_row(ROW, "margin projection", prompt=p.pid, projection=projection,
                                      ref_margin=lp_a - lp_b)
                del frames
            del native, refs
            if profiled:
                first = pm.first_exceeding(first_order, d, limits)
                if first is None:
                    oracle.report_row(ROW, "prefill-step first-exceeding", prompt=p.pid, none=True)
                else:
                    oracle.report_row(ROW, "prefill-step first-exceeding", prompt=p.pid, layer=first["layer"],
                                      name=first["name"], d=first["d"], limit=first["limit"], ratio=first["ratio"])
            div = pm.routed_set_first_divergence(routed, k=top_k)
            if div is None:
                oracle.report_row(ROW, "routed-set first divergence", prompt=p.pid, none=True, layers=len(routed))
            else:
                oracle.report_row(ROW, "routed-set first divergence", prompt=p.pid, layer=div["layer"],
                                  div_row=div["row"], differing=div["differing"], gap=div["gap"], layers=len(routed))
            by_name: dict = {}
            for key, (dist, limit) in last[p.pid].items():
                worst = by_name.get(key[1])
                if worst is None or dist / limit > worst[0]:
                    by_name[key[1]] = (dist / limit, key[0], dist, limit)
            for name, (ratio, layer, dist, limit) in by_name.items():
                oracle.report_row(ROW, "last-row d", prompt=p.pid, name=name, layer=layer, d=dist, limit=limit,
                                  ratio=ratio)
        del lm_head
        for name, row in pm.per_name(order, d, limits).items():
            e, c = spread.get(name, [None, None])
            oracle.report_row(ROW, "prefill-step per-name", name=name, exceeding=row["exceeding"],
                              measured=row["measured"], absent=row["absent"], max_ratio=row["max_ratio"], max_e=e,
                              max_c=c if c is not None else "unavailable")
        for target, profile in pm.excess(ratios).items():
            worst: dict = {}
            for key, value in profile.items():
                if value is not None and (key[1] not in worst or value > worst[key[1]][0]):
                    worst[key[1]] = (value, key[0])
            for name, (value, layer) in worst.items():
                oracle.report_row(ROW, "prefill-step excess", prompt=target, name=name, layer=layer, excess=value,
                                  others=len(ratios) - sum(t in ratios for t in pm.TARGETS))
        oracle.report_row(ROW, "prefill-step prompts", profiled=len(ratios), last_row=len(last),
                          frames=self.frame.frames)

    def test_c_attribution(self):
        keys = b8_block_keys(self.tc)
        for pid in pm.TARGETS:
            p = self.by_pid[pid]
            last = self.refs(p)[-1]
            blocks = {}
            for key in keys:
                value = ref_value(last, key)
                self.assertIsNotNone(value, f"{ROW} {pid}: the dump lacks {full_name(key)}")
                blocks[key] = value.to(oracle.DEVICE)
            del last
            ids, target = p.prompt_ids, target_of(p)

            def margin(selected) -> float:
                return forward_patch(self.handle, ids, target, [(full_name(k), blocks[k]) for k in selected])

            m0 = margin([])
            m_all = margin(keys)
            oracle.report_row(ROW, "P0", prompt=pid, m=m0)
            oracle.report_row(ROW, "P-all", prompt=pid, m=m_all, dm=m_all - m0)
            prefix = {0: m0}
            for L in range(1, self.layers + 1):
                prefix[L] = margin([k for k in keys if k[0] < L])
                oracle.report_row(ROW, "P-prefix", prompt=pid, layers=L, m=prefix[L], dm=prefix[L] - m0)
            classes = {}
            for cls_name in B8_CLASSES:
                selected = [k for k in keys if k[1] == cls_name]
                if selected:
                    classes[cls_name] = margin(selected)
                    oracle.report_row(ROW, "P-class", prompt=pid, name=cls_name, m=classes[cls_name],
                                      dm=classes[cls_name] - m0)
            singles = {k: margin([k]) for k in keys}
            for rank, row in enumerate(pm.top_singles(singles, m0, n=SINGLES_REPORTED), start=1):
                oracle.report_row(ROW, "P-single", prompt=pid, rank=rank, layer=row["key"][0], name=row["key"][1],
                                  m=row["m"], dm=row["dm"])
            l_star = pm.l_star(prefix)
            oracle.report_row(ROW, "L*", prompt=pid, layers=l_star if l_star is not None else "none")
            oracle.report_row(ROW, "attributed class", prompt=pid, name=pm.attributed_class(m0, m_all, classes))
            del blocks

    def test_d_screens(self):
        prompts = self.frame.prompts
        null = null_screen()
        p0 = {p.pid: screen_margin(self.handle, p.prompt_ids, target_of(p), null) for p in prompts}
        weights = Weights()
        screens = [("P-mix", lambda p: mix_screen())]
        for plan in GEMM_WEIGHTS:
            screens.append((f"P-gemm({plan})", lambda p, plan=plan: gemm_screen(plan, weights)))
        screens += [("P-moe(-1)", "moe"), ("P-moe(bench)", "moe")]
        targets = {}
        for name, make in screens:
            release_screen_memory(weights)
            torch.cuda.reset_peak_memory_stats(oracle.DEVICE)
            free_before = torch.cuda.mem_get_info(oracle.DEVICE)[0]
            unavailable: dict = {}  # bench tactic status -> prompts
            if make == "moe":
                tactic_of = {}  # pid -> tactic
                for p in prompts:
                    if name == "P-moe(-1)":
                        tactic_of[p.pid] = -1
                        continue
                    tactic, status = bench_tactic(len(p.prompt_ids))
                    if tactic is None:
                        unavailable[status] = unavailable.get(status, 0) + 1
                    else:
                        tactic_of[p.pid] = tactic
                by_pid = self.by_pid

                def run(pid, screen):
                    return screen_margin(self.handle, by_pid[pid].prompt_ids, target_of(by_pid[pid]), screen)
                margins = moe_margins(list(tactic_of), tactic_of.get, self.layers, run,
                                      isolation().routed_reference, lambda: release_screen_memory(weights), name)
            else:
                try:
                    margins = {p.pid: screen_margin(self.handle, p.prompt_ids, target_of(p),
                                                    Screen(make(p), label=name)) for p in prompts}
                finally:
                    release_screen_memory(weights)
            release_screen_memory(weights)
            oracle.report_row(ROW, "screen memory", screen=name, free_before=free_before,
                              peak_allocated=torch.cuda.max_memory_allocated(oracle.DEVICE),
                              peak_reserved=torch.cuda.max_memory_reserved(oracle.DEVICE))
            if not margins:
                oracle.report_row(ROW, "screen", screen=name, status="unavailable",
                                  bench_tactic="; ".join(sorted(unavailable)))
                continue
            unavailable = sum(unavailable.values())
            flips_up = sum(p0[pid] <= 0 < m for pid, m in margins.items())
            flips_down = sum(m <= 0 < p0[pid] for pid, m in margins.items())
            oracle.report_row(ROW, "screen", screen=name, prompts=len(margins), unavailable=unavailable,
                              positive=sum(m > 0 for m in margins.values()), turned_positive=flips_up,
                              turned_non_positive=flips_down, min_m=min(margins.values()),
                              median_dm=statistics.median(margins[pid] - p0[pid] for pid in margins))
            targets[name] = {t: margins[t] for t in pm.TARGETS if t in margins}
            for t, m in targets[name].items():
                oracle.report_row(ROW, "screen target", screen=name, prompt=t, m=m, dm=m - p0[t])
        for rank, row in enumerate(pm.rank_screens(targets, {t: p0[t] for t in pm.TARGETS}), start=1):
            oracle.report_row(ROW, "screen rank", rank=rank, screen=row["screen"], best_dm=row["best_dm"],
                              turns_positive=row["turns_positive"], positive=row["positive"],
                              **{f"dm_{t}": v for t, v in row["dm"].items()})

    def test_e_first_token(self):
        null = null_screen()
        for k, entries in enumerate(self.frame.frame_prompts, start=1):
            agree = 0
            for p in self.frame.prompts:
                entry = entries[p.pid]
                m = screen_margin(self.handle, entry.prompt_ids, target_of(entry), null)
                agree += m > 0
                oracle.report_row(ROW, "P-1 first token", frame=k, prompt=p.pid, m=m, picks=m > 0)
            oracle.report_row(ROW, "P-1 first token summary", frame=k, prompts=len(self.frame.prompts), agree=agree)


if __name__ == "__main__":
    unittest.main()
