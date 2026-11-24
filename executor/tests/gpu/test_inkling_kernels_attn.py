"""Inkling 2 Mini S2 (tasks/inkling-spec.md 3.3, G1/G4): the attention kernels (executor/cuda/src/kernels/inkling/
attn.h) vs their CPU reference (executor/ref/inkling/attn_ref.h, through libcere_ktest.so), bit for bit, and the
reference vs the PyTorch REF op (transformers modeling_inkling2_loop.py @046bdb8c) under the tolerance rule.

Suite `inkling_kernels` (GPU0-only): `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite inkling_kernels`
against build/native-inkling-2-mini. Synthetic inputs (the post ops run on raw projections drawn per row at amplitudes
that hit and miss the L2 clips); /model is not read. Every case reports row "INKLING_ATTN".

Cases:
- post (q scale + clip 22.627 + log scale on full layers, rel scale + clip 8 + log scale, k clip, v unweighted RMSNorm):
  bit-exact vs the reference at positions around the log-scale floor (8,191 / 8,192) up to 1,048,575, rows of zeros,
  small rows (clip not hit) and large ones (clip hit), in place too; vs the torch REF op (tolerance rule).
- attention prefill (sliding ring / full paged pool with shuffled pages, keys before the call from the cache): every
  checked row bit-exact vs the reference; rerun bit-identical; strided (fused-buffer) inputs equal contiguous ones;
  PA's long full visits (4,096 rows past 16K keys and across the 4,096-key group edges, rows on and off its 16-row
  blocks) and a 4,096-row sliding call past the ring.
- attention decode: 4 sessions x 1 row, 2 sessions x 3 draft rows, 16 sessions x 4 rows (64), full layers past
  16,384 keys (two decode passes): bit-exact vs the reference.
- G4: chunked prefill 512 vs 4,096 (with kv_store between chunks) bit-identical, and equal to the reference on sampled
  rows, at 4,608 positions and on a 20,000-position full visit; decode of rows (1 and 8 per session) bit-identical to
  the prefill of the same rows (past 16,384 keys too: two decode passes); 1 vs 4 sessions in one call (prefill and
  decode) bit-identical.
- kv_store: ring (wrap, a call longer than the ring keeps its last 1,024 rows) and pool (slot map) bit copies.
- reference vs torch REF eager attention (BF16 scores and probabilities as REF; FP64 calibration).
- fail-first: the kNoWindow, kNoSink, kBiasOffByOne, kMutantLaneSum (prefill: q.k lane chains summed in lane order,
  not reduce_dot<32>'s tree) (attention) and kNoLogScale (post) mutants differ from the reference (the bit-exact check
  fails on them).
- decode time of one visit (full at 4K and 128K context, sliding), M 1 and 64; prefill time of a 4,096-row call (full
  ending at 4K, 128K and 1M, sliding): reported, no verdict.
- argument refusals (CERE_ERR_ARGUMENT).

Entry points (libcere_ktest.so, executor/cuda/src/ktest/inkling/attn_ktest.cu; int64 integers, the stream last; a
sequence table is int64 [n, 6] = (row, rows, pos0, k_cache, v_cache, slots), device memory for GPU entries and host
memory with host cache pointers for the `_ref_` entries):
    int cere_ktest_inkling_attn_post(int64_t kind, int64_t rows, const void *positions, const void *q_in,
                                     const void *q_scale, void *q_out, const void *r_in, const void *rel_scale,
                                     void *r_out, const void *k_in, void *k_out, const void *v_in, void *v_out,
                                     int64_t mutant, cudaStream_t stream);
    int cere_ktest_inkling_attn_prefill(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                        const void *q, int64_t q_stride, const void *r, int64_t r_stride,
                                        const void *rel_proj, const void *k, int64_t k_stride, const void *v,
                                        int64_t v_stride, const void *sink, int64_t sink_stride, const void *gate,
                                        int64_t gate_stride, void *out, int64_t out_stride, int64_t mutant,
                                        cudaStream_t stream);
    int64_t cere_ktest_inkling_attn_decode_workspace_bytes(int64_t total_rows);
    int cere_ktest_inkling_attn_decode(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                       int64_t max_pos, int64_t total_rows, <q .. out as prefill>, void *workspace,
                                       int64_t workspace_bytes, int64_t mutant, cudaStream_t stream);
    int cere_ktest_inkling_attn_decode_time(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                            int64_t max_pos, int64_t total_rows, const void *q, const void *r,
                                            const void *rel_proj, const void *k, const void *v, const void *sink,
                                            const void *gate, void *out, void *workspace, int64_t workspace_bytes,
                                            int64_t iters, void *ms /*host float [1]*/, cudaStream_t stream);
    int cere_ktest_inkling_attn_prefill_time(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                             const void *q, const void *r, const void *rel_proj, const void *k,
                                             const void *v, const void *sink, const void *gate, void *out,
                                             int64_t iters, void *ms /*host float [1]*/, cudaStream_t stream);
    int cere_ktest_inkling_attn_kv_store(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                         const void *k, int64_t k_stride, const void *v, int64_t v_stride,
                                         cudaStream_t stream);
    int cere_ktest_inkling_attn_ref_post(<as attn_post, host memory, no mutant>);
    int cere_ktest_inkling_attn_ref_attention(int64_t kind, const void *seqs, int64_t n_seqs, const void *q,
                                              const void *r, const void *rel_proj, const void *k, const void *v,
                                              const void *sink, const void *gate, void *out,
                                              const void *check_rows /*int64 [n_check] or null*/, int64_t n_check,
                                              cudaStream_t stream);
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import math
import unittest

import torch
import torch.nn.functional as F

import oracle

ROW = "INKLING_ATTN"
HEADS, KV_HEADS, HEAD_DIM, D_REL, REL_EXTENT = 24, 8, 128, 16, 1024  # config.json text_config
Q_ROW, KV_ROW, REL_ROW = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, HEADS * D_REL
WINDOW = RING = 1024  # sliding_window_size; the ring of D-I3
PAGE = 64  # host/kv_pool.h kKvPageTokens
QK_CLIP, REL_CLIP, V_EPS = 22.627416997969522, 8.0, 1e-6  # qk/rel_l2_clip_threshold, rms_norm_eps
LOG_ALPHA, LOG_FLOOR = 0.1, 8192.0  # log_scaling_alpha, log_scaling_n_floor
SCALE = HEAD_DIM ** -0.5
SLIDING, FULL = 0, 1
NONE, NO_WINDOW, NO_SINK, BIAS_OFF_BY_ONE, NO_LOG_SCALE, LANE_SUM = 0, 1, 2, 3, 4, 5
MAX_PREFILL_ROWS, MAX_DECODE_ROWS = 4096, 64
# S1's recorded per-op tensors of REF (arm Q, GPU torch; tests/fixtures/inkling2_reference/ops): the first 4 positions
# of a 26-token prompt, causal from position 0, so rows 0..3 attend to keys 0..3 only.
FIXTURES = oracle.SRC / "tests/fixtures/inkling2_reference/ops"
RECORDED = (("v19_L07_pass2_sliding", SLIDING), ("v23_L11_pass2_full", FULL))
FIXTURE_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32, "i64": torch.int64}


@functools.cache
def _fixture_index() -> dict:
    return json.loads((FIXTURES / "index.json").read_text())


def _recorded(name: str, key: str) -> torch.Tensor:
    """One recorded tensor of set `name` (sha256-checked against the index), on GPU0."""
    e = _fixture_index()["sets"][name]["tensors"][key]
    data = (FIXTURES / e["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != e["sha256"]:
        raise oracle.OracleError(f"{e['file']}: sha256 differs from index.json")
    return torch.frombuffer(bytearray(data), dtype=FIXTURE_DTYPES[e["dtype"]]).reshape(e["shape"]).to(DEV)


BF = torch.bfloat16
DEV = oracle.DEVICE
KIND_NAME = {SLIDING: "sliding", FULL: "full"}


def _uniform(shape, seed: int, lo: float, hi: float) -> torch.Tensor:
    g = oracle.generator(seed)
    return (torch.rand(shape, generator=g, device=DEV) * (hi - lo) + lo).to(BF)


def _amplitudes(rows: int, seed: int) -> torch.Tensor:
    """Per-row amplitudes in [0.05, 5]: rows below and above the clip thresholds; row 0 is all zero."""
    a = torch.rand((rows, 1), generator=oracle.generator(seed), device=DEV) * 4.95 + 0.05
    a[0] = 0.0
    return a


def _raw(rows: int, width: int, seed: int) -> torch.Tensor:
    x = torch.rand((rows, width), generator=oracle.generator(seed), device=DEV) * 2.0 - 1.0
    return (x * _amplitudes(rows, seed + 1)).to(BF)


def _weights(seed: int) -> dict:
    return {"q_scale": _uniform((HEADS, HEAD_DIM), seed, 0.5, 1.5),
            "rel_scale": _uniform((HEADS, D_REL), seed + 1, 0.5, 1.5),
            "proj": _uniform((D_REL, REL_EXTENT), seed + 2, -1.0, 1.0)}


def _post(kind: int, positions: torch.Tensor, w: dict, q=None, r=None, k=None, v=None, mutant: int = NONE):
    outs = [None if x is None else oracle.empty(tuple(x.shape), BF) for x in (q, r, k, v)]
    oracle.call("cere_ktest_inkling_attn_post", kind, positions.shape[0], positions, q, w["q_scale"], outs[0], r,
                w["rel_scale"], outs[1], k, outs[2], v, outs[3], mutant)
    return outs


def _ref_post(kind: int, positions: torch.Tensor, w: dict, q=None, r=None, k=None, v=None):
    host = [None if x is None else x.cpu().contiguous() for x in (q, r, k, v)]
    outs = [None if x is None else torch.full(tuple(x.shape), float("nan"), dtype=BF) for x in host]
    oracle.call("cere_ktest_inkling_attn_ref_post", kind, positions.shape[0], positions.cpu().contiguous(), host[0],
                w["q_scale"].cpu(), outs[0], host[1], w["rel_scale"].cpu(), outs[1], host[2], outs[2], host[3],
                outs[3])
    return outs


class Session:
    """`total` positions of one session's visit: post outputs q, r, k, v (from the GPU post, which the post cases
    check bit for bit), sink and gate rows, and a cache (sliding: ring; full: pool with shuffled pages)."""

    def __init__(self, kind: int, total: int, seed: int, w: dict):
        positions = torch.arange(total, device=DEV, dtype=torch.int64)
        q, r, k, v = _post(kind, positions, w, _raw(total, Q_ROW, seed), _raw(total, REL_ROW, seed + 2),
                           _raw(total, KV_ROW, seed + 4), _raw(total, KV_ROW, seed + 6))
        self._setup(kind, (q, r, k, v, _uniform((total, HEADS), seed + 8, -2.0, 6.0),
                           _uniform((total, Q_ROW), seed + 9, -4.0, 4.0)), seed)

    @classmethod
    def of(cls, kind: int, rows: tuple) -> "Session":
        """A session of given rows (q, r, k, v, sink, gate) at positions 0 .."""
        s = cls.__new__(cls)
        s._setup(kind, rows, 0)
        return s

    def _setup(self, kind: int, rows: tuple, seed: int) -> None:
        self.q, self.r, self.k, self.v, self.sink, self.gate = (t.contiguous() for t in rows)
        self.kind, self.total = kind, self.q.shape[0]
        self.positions = torch.arange(self.total, device=DEV, dtype=torch.int64)
        pages = (self.total + PAGE - 1) // PAGE
        perm = torch.randperm(pages, generator=torch.Generator().manual_seed(seed)).to(DEV)
        self.slots = (perm[self.positions // PAGE] * PAGE + self.positions % PAGE).contiguous()
        cache_rows = RING if kind == SLIDING else pages * PAGE
        self.k_cache = torch.full((cache_rows, KV_ROW), float("nan"), dtype=BF, device=DEV)
        self.v_cache = torch.full((cache_rows, KV_ROW), float("nan"), dtype=BF, device=DEV)

    def fill(self, pos0: int) -> None:
        """The cache as after storing positions < pos0 (the ring keeps the last 1,024)."""
        self.k_cache.fill_(float("nan"))
        self.v_cache.fill_(float("nan"))
        if self.kind == SLIDING:
            j = torch.arange(max(0, pos0 - RING), pos0, device=DEV)
            self.k_cache[j % RING] = self.k[j]
            self.v_cache[j % RING] = self.v[j]
        else:
            self.k_cache[self.slots[:pos0]] = self.k[:pos0]
            self.v_cache[self.slots[:pos0]] = self.v[:pos0]

    def entry(self, pos0: int, rows: int) -> tuple:
        return (pos0, rows, self.k_cache, self.v_cache, self.slots if self.kind == FULL else None)


class Call:
    """One attention call: sequences (session entries) stacked into contiguous rows."""

    def __init__(self, kind: int, parts: list):
        """parts: (session, pos0, rows); each session's cache must hold its positions < pos0."""
        self.kind = kind
        self.parts = parts
        self.entries = []
        row = 0
        for s, pos0, rows in parts:
            self.entries.append((row, rows, pos0) + s.entry(pos0, rows)[2:])
            row += rows
        self.rows = row
        cat = lambda name: torch.cat([getattr(s, name)[p:p + n] for s, p, n in parts]).contiguous()
        self.q, self.r, self.k, self.v, self.sink, self.gate = (cat(n) for n in ("q", "r", "k", "v", "sink", "gate"))
        self.max_rows = max(n for _, _, n in parts)
        self.max_pos = max(p + n for _, p, n in parts)

    def table(self, host: bool) -> tuple[torch.Tensor, list]:
        keep, rows = [], []
        for row, rows_, pos0, kc, vc, slots in self.entries:
            ptrs = []
            for t in (kc, vc, slots):
                if t is None:
                    ptrs.append(0)
                    continue
                t = t.cpu().contiguous() if host else t
                keep.append(t)
                ptrs.append(t.data_ptr())
            rows.append([row, rows_, pos0] + ptrs)
        return torch.tensor(rows, dtype=torch.int64, device="cpu" if host else DEV), keep


def _prefill(c: Call, proj: torch.Tensor, mutant: int = NONE) -> torch.Tensor:
    table, _ = c.table(False)
    out = oracle.empty((c.rows, Q_ROW), BF)
    oracle.call("cere_ktest_inkling_attn_prefill", c.kind, table, len(c.entries), c.max_rows, c.q, Q_ROW, c.r,
                REL_ROW, proj, c.k, KV_ROW, c.v, KV_ROW, c.sink, HEADS, c.gate, Q_ROW, out, Q_ROW, mutant)
    return out


def _ws_bytes(rows: int) -> int:
    fn = oracle.ktest("cere_ktest_inkling_attn_decode_workspace_bytes")
    fn.restype = ctypes.c_int64
    return int(fn(ctypes.c_int64(rows)))


def _decode(c: Call, proj: torch.Tensor, mutant: int = NONE) -> torch.Tensor:
    table, _ = c.table(False)
    out = oracle.empty((c.rows, Q_ROW), BF)
    nbytes = _ws_bytes(c.rows)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    oracle.call("cere_ktest_inkling_attn_decode", c.kind, table, len(c.entries), c.max_rows, c.max_pos, c.rows, c.q,
                Q_ROW, c.r, REL_ROW, proj, c.k, KV_ROW, c.v, KV_ROW, c.sink, HEADS, c.gate, Q_ROW, out, Q_ROW, ws,
                nbytes, mutant)
    return out


def _ref(c: Call, proj: torch.Tensor, check: list | None = None) -> torch.Tensor:
    table, keep = c.table(True)
    out = torch.full((c.rows, Q_ROW), float("nan"), dtype=BF)
    rows = None if check is None else torch.tensor(check, dtype=torch.int64)
    host = [t.cpu().contiguous() for t in (c.q, c.r, proj, c.k, c.v, c.sink, c.gate)]
    oracle.call("cere_ktest_inkling_attn_ref_attention", c.kind, table, len(c.entries), *host, out, rows,
                0 if check is None else len(check))
    del keep
    return out


def _kv_store(kind: int, c: Call) -> None:
    table, _ = c.table(False)
    oracle.call("cere_ktest_inkling_attn_kv_store", kind, table, len(c.entries), c.max_rows, c.k, KV_ROW, c.v, KV_ROW)


def _differing_rows(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.cpu().view(torch.int16) != b.cpu().view(torch.int16)).any(dim=-1).sum().item())


# -- the torch REF op ---------------------------------------------------------------------------------------------
def _tau(positions: torch.Tensor, dtype) -> torch.Tensor:
    return 1.0 + LOG_ALPHA * torch.log(((positions + 1).to(dtype) / LOG_FLOOR).clamp(min=1.0))


def _torch_clip(x: torch.Tensor, scale, threshold: float, heads: int, dim: int, positions, dtype) -> torch.Tensor:
    """REF:183-236 (Inkling2Loop[LogScaled|Scaled]L2Clip); dtype float32 = REF's rounding points, float64 = none."""
    h = x.view(-1, heads, dim).to(dtype)
    if scale is not None:
        h = h * scale.to(dtype)
    norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    y = h * (threshold / (norm + 1e-6)).clamp(max=1.0)
    if dtype == torch.float32:
        y = y.to(BF)
    if positions is not None:
        y = y.to(dtype) * _tau(positions, dtype)[:, None, None]
        if dtype == torch.float32:
            y = y.to(BF)
    return y.reshape(x.shape[0], heads * dim)


def _torch_vnorm(x: torch.Tensor, dtype) -> torch.Tensor:
    """REF:72-76 (Inkling2LoopUnweightedRMSNorm)."""
    h = x.view(-1, KV_HEADS, HEAD_DIM).to(dtype)
    y = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + V_EPS)
    return (y.to(BF) if dtype == torch.float32 else y).reshape(x.shape[0], KV_ROW)


def _torch_attention(kind: int, c: Call, proj: torch.Tensor, dtype) -> torch.Tensor:
    """REF eager attention (REF:111-142) with the relative bias (REF:266-270), sink and gate (REF:540-542) for a call
    of one sequence at pos0 0 (keys = the call's rows). float32: REF's BF16 tensors (scores, bias, probabilities, PV
    output); float64: no rounding."""
    n = c.rows
    work = BF if dtype == torch.float32 else torch.float64
    q = c.q.to(work).view(n, HEADS, HEAD_DIM).transpose(0, 1)
    k = c.k.to(work).view(n, KV_HEADS, HEAD_DIM).transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, dim=0)
    v = c.v.to(work).view(n, KV_HEADS, HEAD_DIM).transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, dim=0)
    pos = torch.arange(n, device=DEV)
    dist = pos[:, None] - pos[None, :]
    rel = (c.r.to(work).view(n, HEADS, D_REL) @ proj.to(work)).transpose(0, 1)  # [heads, n, extent]
    bias = rel.gather(-1, dist.clamp(0, REL_EXTENT - 1).expand(HEADS, n, n))
    bias = bias.masked_fill((dist < 0) | (dist >= REL_EXTENT), 0.0)
    scores = (q @ k.transpose(1, 2)) * SCALE + bias
    visible = dist >= 0
    if kind == SLIDING:
        visible &= dist < WINDOW
    scores = scores.masked_fill(~visible, float("-inf"))
    soft = torch.float32 if dtype == torch.float32 else torch.float64
    combined = torch.cat([scores, c.sink.to(work).t()[:, :, None]], dim=-1).to(soft)
    p = F.softmax(combined, dim=-1)[..., :-1].to(work)
    o = (p @ v).transpose(0, 1).reshape(n, Q_ROW)
    gated = o.to(soft) * torch.sigmoid(c.gate.to(soft))
    return gated.to(BF) if dtype == torch.float32 else gated


class InklingAttention(unittest.TestCase):
    def setUp(self):
        self.w = _weights(7)

    # -- post ------------------------------------------------------------------------------------------------------
    def test_post_bit_exact(self):
        """q/r/k/v post bit-exact vs the reference (clip hit and not hit, zero rows, log-scale edges), in place too."""
        positions = torch.tensor([0, 1, 8190, 8191, 8192, 8193, 20000, 131071, 1048575] + list(range(100, 155)),
                                 dtype=torch.int64, device=DEV)
        rows = positions.shape[0]
        raws = (_raw(rows, Q_ROW, 11), _raw(rows, REL_ROW, 13), _raw(rows, KV_ROW, 15), _raw(rows, KV_ROW, 17))
        for kind in (SLIDING, FULL):
            case = f"post {KIND_NAME[kind]} rows {rows}"
            native = _post(kind, positions, self.w, *raws)
            again = _post(kind, positions, self.w, *raws)
            reference = _ref_post(kind, positions, self.w, *raws)
            for name, a, b, c in zip("qrkv", native, again, reference):
                oracle.check_bit_exact(self, ROW, f"{case} {name}", a.cpu(), c)
                oracle.check_bit_exact(self, ROW, f"{case} {name} rerun", b, a)
            inplace = [x.clone() for x in raws]
            oracle.call("cere_ktest_inkling_attn_post", kind, rows, positions, inplace[0], self.w["q_scale"],
                        inplace[0], inplace[1], self.w["rel_scale"], inplace[1], inplace[2], inplace[2], inplace[3],
                        inplace[3], NONE)
            for name, a, b in zip("qrkv", inplace, native):
                oracle.check_bit_exact(self, ROW, f"{case} {name} in place", a, b)
            clipped = int((torch.linalg.vector_norm(raws[2].float().view(rows, KV_HEADS, HEAD_DIM), dim=-1)
                           > QK_CLIP).sum().item())
            oracle.check_count(self, ROW, f"{case} k heads clipped and not", int(0 < clipped < rows * KV_HEADS),
                               expected=1)

    def test_post_vs_torch(self):
        """The reference post ops vs the torch REF ops (tolerance rule: FP32 with REF's rounding vs FP64)."""
        rows = 256
        positions = torch.arange(30000, 30000 + rows, dtype=torch.int64, device=DEV)
        raws = (_raw(rows, Q_ROW, 21), _raw(rows, REL_ROW, 23), _raw(rows, KV_ROW, 25), _raw(rows, KV_ROW, 27))
        for kind in (SLIDING, FULL):
            reference = _ref_post(kind, positions, self.w, *raws)
            pos = positions if kind == FULL else None
            ops = {
                "q": lambda d: _torch_clip(raws[0], self.w["q_scale"], QK_CLIP, HEADS, HEAD_DIM, pos, d),
                "r": lambda d: _torch_clip(raws[1], self.w["rel_scale"], REL_CLIP, HEADS, D_REL, pos, d),
                "k": lambda d: _torch_clip(raws[2], None, QK_CLIP, KV_HEADS, HEAD_DIM, None, d),
                "v": lambda d: _torch_vnorm(raws[3], d),
            }
            for (name, op), ours in zip(ops.items(), reference):
                case = f"post {KIND_NAME[kind]} {name} ref vs torch"
                torch_op = op(torch.float32)
                c = oracle.calibrate(ROW, case, torch_op, op(torch.float64), oracle.FLOOR_OP)
                limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                oracle.check_within(self, ROW, case, oracle.rel_l2(ours.to(DEV), torch_op), c=c, limit=limit)

    # -- attention vs the reference ---------------------------------------------------------------------------------
    def _check(self, case: str, native: torch.Tensor, c: Call, proj: torch.Tensor, check: list | None = None):
        reference = _ref(c, proj, check)
        rows = list(range(c.rows)) if check is None else check
        oracle.check_bit_exact(self, ROW, case, native.cpu()[rows], reference[rows])

    def test_prefill_bit_exact(self):
        """Prefill vs the reference: from position 0, after a ring / pool history, across the 4,096-key group edge."""
        proj = self.w["proj"]
        cases = [(SLIDING, 1, 0, 1), (SLIDING, 70, 0, 70), (SLIDING, 1300, 0, 1300), (SLIDING, 2600, 1100, 1500),
                 (FULL, 600, 0, 600), (FULL, 5000, 3900, 1100)]
        for kind, total, pos0, n in cases:
            with self.subTest(kind=kind, total=total, pos0=pos0):
                case = f"prefill {KIND_NAME[kind]} pos0 {pos0} n {n}"
                s = Session(kind, total, 100 + total + pos0, self.w)
                s.fill(pos0)
                c = Call(kind, [(s, pos0, n)])
                native = _prefill(c, proj)
                check = None if n <= 1500 else list(range(0, n, 7)) + [n - 1]
                self._check(case, native, c, proj, check)
                oracle.check_bit_exact(self, ROW, f"{case} rerun", _prefill(c, proj), native)

    def test_prefill_long_bit_exact(self):
        """PA on long visits vs the reference on sampled rows: full 4,096 rows from position 15,904 (keys past 16K,
        across the 16,384 group edge), full 4,096 rows from 3 (16-row blocks off the call's rows, across 4,096),
        sliding 4,096 rows from 1,903 (window wholly in the call and in the ring)."""
        proj = self.w["proj"]
        for kind, total, pos0, n in ((FULL, 20000, 15904, 4096), (FULL, 4099, 3, 4096), (SLIDING, 6000, 1903, 4096)):
            with self.subTest(kind=kind, total=total, pos0=pos0):
                case = f"prefill long {KIND_NAME[kind]} pos0 {pos0} n {n}"
                s = Session(kind, total, 7100 + total + pos0, self.w)
                s.fill(pos0)
                c = Call(kind, [(s, pos0, n)])
                native = _prefill(c, proj)
                edges = {0, 1, 15, 16, 17, 31, 32, 1023, 1024, 1025, n - 17, n - 16, n - 2, n - 1}
                edges |= {x - pos0 + d for x in range(4096, pos0 + n, 4096) for d in (-1, 0, 1)
                          if 0 <= x - pos0 + d < n}
                check = sorted(edges | set(range(0, n, 113)))
                self._check(case, native, c, proj, check)
                oracle.check_bit_exact(self, ROW, f"{case} rerun", _prefill(c, proj), native)

    def test_prefill_strided_inputs(self):
        """Inputs at the strides of fused projection buffers give the contiguous result bit for bit."""
        proj = self.w["proj"]
        for kind in (SLIDING, FULL):
            s = Session(kind, 1200, 300 + kind, self.w)
            s.fill(900)
            c = Call(kind, [(s, 900, 300)])
            want = _prefill(c, proj)
            n = c.rows
            wide = torch.cat([c.q, c.gate, c.k, c.v, c.r, c.sink, torch.zeros((n, 8), dtype=BF, device=DEV)], 1)
            stride = wide.shape[1]
            offs = [0, Q_ROW, 2 * Q_ROW, 2 * Q_ROW + KV_ROW, 2 * Q_ROW + 2 * KV_ROW, 2 * Q_ROW + 2 * KV_ROW + REL_ROW]
            out = torch.full((n, Q_ROW + 64), float("nan"), dtype=BF, device=DEV)
            table, _ = c.table(False)
            flat = wide.view(-1)
            oracle.call("cere_ktest_inkling_attn_prefill", kind, table, 1, n, flat[offs[0]:], stride, flat[offs[4]:],
                        stride, proj, flat[offs[2]:], stride, flat[offs[3]:], stride, flat[offs[5]:], stride,
                        flat[offs[1]:], stride, out, Q_ROW + 64, NONE)
            oracle.check_bit_exact(self, ROW, f"prefill {KIND_NAME[kind]} strided", out[:, :Q_ROW].contiguous(), want)

    def test_decode_bit_exact(self):
        """Decode vs the reference: sessions at the window / group edges, draft rows, 64 rows, two decode passes."""
        proj = self.w["proj"]
        layouts = {
            SLIDING: [[(1, 1), (1023, 1), (1024, 1), (3000, 1)], [(500, 3), (2047, 3)]],
            FULL: [[(1, 1), (4095, 1), (4096, 1), (9000, 1)], [(700, 3), (16383, 3)], [(20000, 1), (16380, 8)]],
        }
        for kind, calls in layouts.items():
            for i, layout in enumerate(calls):
                with self.subTest(kind=kind, layout=i):
                    sessions = []
                    for j, (pos0, n) in enumerate(layout):
                        s = Session(kind, pos0 + n, 500 + 10 * j + 100 * i + kind, self.w)
                        s.fill(pos0)
                        sessions.append((s, pos0, n))
                    c = Call(kind, sessions)
                    case = f"decode {KIND_NAME[kind]} {layout}"
                    native = _decode(c, proj)
                    self._check(case, native, c, proj)
                    oracle.check_bit_exact(self, ROW, f"{case} rerun", _decode(c, proj), native)

    def test_decode_64_rows(self):
        """16 sessions x 4 draft rows (M 64) vs the reference and vs each session alone."""
        proj = self.w["proj"]
        for kind in (SLIDING, FULL):
            parts = []
            for j in range(16):
                pos0 = 37 + 211 * j
                s = Session(kind, pos0 + 4, 900 + j + 50 * kind, self.w)
                s.fill(pos0)
                parts.append((s, pos0, 4))
            c = Call(kind, parts)
            case = f"decode {KIND_NAME[kind]} 16 x 4"
            native = _decode(c, proj)
            self._check(case, native, c, proj)
            alone = torch.cat([_decode(Call(kind, [p]), proj) for p in parts])
            oracle.check_bit_exact(self, ROW, f"{case} vs alone", native, alone)

    # -- G4 ----------------------------------------------------------------------------------------------------------
    def _chunked(self, s: Session, chunk: int, end: int, proj) -> torch.Tensor:
        """Prefill of positions [0, end) in chunks from an empty cache, kv_store after each chunk."""
        s.fill(0)
        outs = []
        for a in range(0, end, chunk):
            c = Call(s.kind, [(s, a, min(chunk, end - a))])
            outs.append(_prefill(c, proj))
            _kv_store(s.kind, c)
        return torch.cat(outs)

    def test_chunk_and_decode_invariance(self):
        """Chunk 512 vs 4,096 bit-identical (and = the reference on sampled rows); decode of the last rows (8 drafts,
        and 1) equals their prefill rows."""
        proj = self.w["proj"]
        total, tail = 4608, 4600
        for kind in (SLIDING, FULL):
            s = Session(kind, total, 1300 + kind, self.w)
            small = self._chunked(s, 512, total, proj)
            big = self._chunked(s, MAX_PREFILL_ROWS, total, proj)
            case = f"{KIND_NAME[kind]} chunk 512 vs 4096"
            oracle.check_bit_exact(self, ROW, case, small, big)
            whole = Call(kind, [(s, 0, total)])
            s.fill(0)
            check = list(range(0, total, 97)) + [1023, 1024, 4095, 4096, total - 1]
            reference = _ref(whole, proj, check)
            oracle.check_bit_exact(self, ROW, f"{case} vs reference", big.cpu()[check], reference[check])
            self._chunked(s, MAX_PREFILL_ROWS, tail, proj)  # the cache as after positions < 4600
            drafts = _decode(Call(kind, [(s, tail, total - tail)]), proj)
            oracle.check_bit_exact(self, ROW, f"{KIND_NAME[kind]} decode 8 drafts vs prefill rows", drafts, big[tail:])
            one = _decode(Call(kind, [(s, tail, 1)]), proj)
            oracle.check_bit_exact(self, ROW, f"{KIND_NAME[kind]} decode 1 row vs prefill row", one, big[tail:tail + 1])

    def test_chunk_long_full(self):
        """A 20,000-position full visit: chunk 512 vs 4,096 bit-identical, = the reference on sampled rows; decode of
        the last row (two decode passes) equals its prefill row."""
        proj = self.w["proj"]
        total = 20000
        s = Session(FULL, total, 1400, self.w)
        small = self._chunked(s, 512, total, proj)
        big = self._chunked(s, MAX_PREFILL_ROWS, total, proj)
        case = "full 20000 chunk 512 vs 4096"
        oracle.check_bit_exact(self, ROW, case, small, big)
        whole = Call(FULL, [(s, 0, total)])
        s.fill(0)
        check = list(range(0, total, 1013)) + [4095, 4096, 8191, 8192, 16383, 16384, total - 1]
        oracle.check_bit_exact(self, ROW, f"{case} vs reference", big.cpu()[check], _ref(whole, proj, check)[check])
        self._chunked(s, MAX_PREFILL_ROWS, total - 1, proj)
        one = _decode(Call(FULL, [(s, total - 1, 1)]), proj)
        oracle.check_bit_exact(self, ROW, "full 20000 decode 1 row vs prefill row", one, big[total - 1:])

    def test_sessions_invariance(self):
        """1 vs 4 sessions in one call (prefill and decode) bit-identical."""
        proj = self.w["proj"]
        for kind in (SLIDING, FULL):
            parts = []
            for j, (pos0, n) in enumerate([(0, 300), (1500, 200), (900, 513), (4090, 100)]):
                s = Session(kind, pos0 + n, 1700 + j + 10 * kind, self.w)
                s.fill(pos0)
                parts.append((s, pos0, n))
            batch = _prefill(Call(kind, parts), proj)
            alone = torch.cat([_prefill(Call(kind, [p]), proj) for p in parts])
            oracle.check_bit_exact(self, ROW, f"prefill {KIND_NAME[kind]} 4 sessions vs 1", batch, alone)
            dparts = [(s, pos0 + n - 1, 1) for s, pos0, n in parts]
            for s, pos0, _ in dparts:
                s.fill(pos0)
            batch = _decode(Call(kind, dparts), proj)
            alone = torch.cat([_decode(Call(kind, [p]), proj) for p in dparts])
            oracle.check_bit_exact(self, ROW, f"decode {KIND_NAME[kind]} 4 sessions vs 1", batch, alone)

    # -- KV store ----------------------------------------------------------------------------------------------------
    def test_kv_store(self):
        """Ring (wrap; a call longer than the ring keeps its last 1,024 rows) and paged pool bit copies."""
        for kind, pos0, n in ((SLIDING, 300, 1500), (SLIDING, 1000, 50), (FULL, 130, 700)):
            s = Session(kind, pos0 + n, 2000 + pos0, self.w)
            s.fill(0)
            k_before, v_before = s.k_cache.clone(), s.v_cache.clone()
            _kv_store(kind, Call(kind, [(s, pos0, n)]))
            want_k, want_v = k_before, v_before
            j = torch.arange(max(pos0, pos0 + n - RING), pos0 + n, device=DEV) if kind == SLIDING else \
                torch.arange(pos0, pos0 + n, device=DEV)
            dst = j % RING if kind == SLIDING else s.slots[j]
            want_k[dst] = s.k[j]
            want_v[dst] = s.v[j]
            case = f"kv_store {KIND_NAME[kind]} pos0 {pos0} n {n}"
            oracle.check_bit_exact(self, ROW, f"{case} k", s.k_cache, want_k)
            oracle.check_bit_exact(self, ROW, f"{case} v", s.v_cache, want_v)

    # -- the reference vs torch REF -----------------------------------------------------------------------------------
    def test_reference_vs_torch(self):
        """The CPU reference vs REF's eager attention (tolerance rule)."""
        proj = self.w["proj"]
        for kind, n in ((SLIDING, 1300), (FULL, 700)):
            s = Session(kind, n, 2500 + kind, self.w)
            s.fill(0)
            c = Call(kind, [(s, 0, n)])
            case = f"attention {KIND_NAME[kind]} n {n} ref vs torch"
            torch_op = _torch_attention(kind, c, proj, torch.float32)
            cal = oracle.calibrate(ROW, case, torch_op, _torch_attention(kind, c, proj, torch.float64),
                                   oracle.FLOOR_OP)
            limit = oracle.calibrated_limit(cal, oracle.FLOOR_OP)
            ours = _ref(c, proj).to(DEV)
            oracle.check_within(self, ROW, case, oracle.rel_l2(ours, torch_op), c=cal, limit=limit)
            native = _prefill(c, proj)
            oracle.check_bit_exact(self, ROW, f"{case} native", native.cpu(), ours.cpu())

    # -- recorded inputs (S1) --------------------------------------------------------------------------------------------
    def test_recorded(self):
        """S1's recorded REF tensors (layer 7 sliding, layer 11 full; pass 2) with the checkpoint's weights: GPU = the
        reference bit for bit (post, prefill, decode), and the reference vs REF's recorded outputs (tolerance rule)."""
        for name, kind in RECORDED:
            rec = lambda key: _recorded(name, key)
            layer = _fixture_index()["sets"][name]["layer"]
            pre = f"model.llm.layers.{layer}.attn."
            w = {"q_scale": oracle.real_tensor(pre + "q_l2_clip.scale").to(BF).contiguous(),
                 "rel_scale": oracle.real_tensor(pre + "rel_l2_clip.scale").to(BF).contiguous(),
                 "proj": oracle.real_tensor(pre + "rel_logits_proj.proj").to(BF).contiguous()}
            positions = rec("self_attn:position_ids").reshape(-1).contiguous()
            n = positions.shape[0]
            heads_first = lambda t, h, d: t.reshape(h, n, d).transpose(0, 1).reshape(n, h * d).contiguous()
            raws = (rec("self_attn.q_proj.linear:out").reshape(n, Q_ROW).contiguous(),
                    rec("self_attn.relative_position_bias.linear:out").reshape(n, REL_ROW).contiguous(),
                    rec("self_attn.k_proj.sconv:out").reshape(n, KV_ROW).contiguous(),
                    rec("self_attn.v_proj.sconv:out").reshape(n, KV_ROW).contiguous())
            recorded = (heads_first(rec("self_attn.q_proj:out"), HEADS, HEAD_DIM),
                        rec("self_attn.relative_position_bias.norm:out").reshape(n, REL_ROW).contiguous(),
                        heads_first(rec("self_attn.k_proj:out"), KV_HEADS, HEAD_DIM),
                        heads_first(rec("self_attn.v_proj:out"), KV_HEADS, HEAD_DIM))
            native = _post(kind, positions, w, *raws)
            reference = _ref_post(kind, positions, w, *raws)
            pos = positions if kind == FULL else None
            exact = {
                "q": lambda d: _torch_clip(raws[0], w["q_scale"], QK_CLIP, HEADS, HEAD_DIM, pos, d),
                "r": lambda d: _torch_clip(raws[1], w["rel_scale"], REL_CLIP, HEADS, D_REL, pos, d),
                "k": lambda d: _torch_clip(raws[2], None, QK_CLIP, KV_HEADS, HEAD_DIM, None, d),
                "v": lambda d: _torch_vnorm(raws[3], d),
            }
            for part, ours, gpu, theirs in zip("qrkv", reference, native, recorded):
                case = f"recorded {name} post {part}"
                oracle.check_bit_exact(self, ROW, f"{case} native", gpu.cpu(), ours)
                oracle.report_row(ROW, f"{case} vs REF", phase="recorded_elements",
                                  differing=int((ours.to(DEV).view(torch.int16) != theirs.view(torch.int16)).sum()),
                                  elements=theirs.numel())
                c = oracle.calibrate(ROW, f"{case} vs REF", theirs, exact[part](torch.float64), oracle.FLOOR_OP)
                limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                oracle.check_within(self, ROW, f"{case} vs REF", oracle.rel_l2(ours.to(DEV), theirs), c=c, limit=limit)
            # Attention on REF's own post outputs, so this checks the attention alone.
            s = Session.of(kind, recorded + (rec("self_attn.sink_proj:out").reshape(n, HEADS),
                                             rec("self_attn.o_proj.gate_proj:out").reshape(n, Q_ROW)))
            s.fill(0)
            call = Call(kind, [(s, 0, n)])
            case = f"recorded {name} attention"
            ours = _ref(call, w["proj"])
            oracle.check_bit_exact(self, ROW, f"{case} prefill native", _prefill(call, w["proj"]).cpu(), ours)
            oracle.check_bit_exact(self, ROW, f"{case} decode native", _decode(call, w["proj"]).cpu(), ours)
            theirs = rec("self_attn.o_proj.linear:in0").reshape(n, Q_ROW)
            oracle.report_row(ROW, f"{case} vs REF", phase="recorded_elements",
                              differing=int((ours.to(DEV).view(torch.int16) != theirs.view(torch.int16)).sum()),
                              elements=theirs.numel())
            c = oracle.calibrate(ROW, f"{case} vs REF", theirs, _torch_attention(kind, call, w["proj"], torch.float64),
                                 oracle.FLOOR_OP)
            limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
            oracle.check_within(self, ROW, f"{case} vs REF", oracle.rel_l2(ours.to(DEV), theirs), c=c, limit=limit)

    # -- fail-first ----------------------------------------------------------------------------------------------------
    def test_mutants_fail(self):
        """Each mutant differs from the reference, so the bit-exact checks above fail on it."""
        proj = self.w["proj"]
        s = Session(SLIDING, 1300, 3000, self.w)
        s.fill(0)
        c = Call(SLIDING, [(s, 0, 1300)])
        reference = _ref(c, proj)
        for name, mutant, rows in (("no window", NO_WINDOW, range(1024, 1300)), ("no sink", NO_SINK, range(1300)),
                                   ("bias off by one", BIAS_OFF_BY_ONE, range(1300)),
                                   ("lane sum", LANE_SUM, range(1300))):
            differ = _differing_rows(_prefill(c, proj, mutant)[list(rows)], reference[list(rows)])
            oracle.report_row(ROW, f"mutant {name}", phase="fail_first", differing_rows=differ)
            oracle.check_count(self, ROW, f"mutant {name} differs from the reference", int(differ > 0), expected=1)
        positions = torch.arange(9000, 9064, dtype=torch.int64, device=DEV)  # past the 8,192 floor: tau > 1
        raws = (_raw(64, Q_ROW, 31), _raw(64, REL_ROW, 33))
        want = _ref_post(FULL, positions, self.w, *raws)
        got = _post(FULL, positions, self.w, *raws, mutant=NO_LOG_SCALE)
        differ = _differing_rows(got[0], want[0]) + _differing_rows(got[1], want[1])
        oracle.report_row(ROW, "mutant no log scale", phase="fail_first", differing_rows=differ)
        oracle.check_count(self, ROW, "mutant no log scale differs from the reference", int(differ > 0), expected=1)

    # -- timing --------------------------------------------------------------------------------------------------------
    def test_decode_time(self):
        """Decode time of one visit (reported, no verdict): full at 4K and 128K context, sliding; M 1 and 64."""
        proj = self.w["proj"]
        for kind, context in ((FULL, 4096), (FULL, 131072), (SLIDING, 131072)):
            keys = context if kind == FULL else RING
            k_cache = _uniform((keys, KV_ROW), 41, -0.2, 0.2)
            v_cache = _uniform((keys, KV_ROW), 42, -1.0, 1.0)
            for m in (1, MAX_DECODE_ROWS):
                pos0 = context - 1
                slots = torch.arange(context, dtype=torch.int64, device=DEV)  # full: identity slot map
                slot_ptr = slots.data_ptr() if kind == FULL else 0
                rows = [[i, 1, pos0, k_cache.data_ptr(), v_cache.data_ptr(), slot_ptr] for i in range(m)]
                table = torch.tensor(rows, dtype=torch.int64, device=DEV)
                q, r = _uniform((m, Q_ROW), 43, -1.0, 1.0), _uniform((m, REL_ROW), 44, -1.0, 1.0)
                k, v = _uniform((m, KV_ROW), 45, -0.2, 0.2), _uniform((m, KV_ROW), 46, -1.0, 1.0)
                sink, gate = _uniform((m, HEADS), 47, -2.0, 2.0), _uniform((m, Q_ROW), 48, -2.0, 2.0)
                out = oracle.empty((m, Q_ROW), BF)
                nbytes = _ws_bytes(m)
                ws = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
                ms = torch.zeros(1, dtype=torch.float32)
                oracle.call("cere_ktest_inkling_attn_decode_time", kind, table, m, 1, context, m, q, r, proj, k, v,
                            sink, gate, out, ws, nbytes, 20, ms)
                oracle.report_row(ROW, f"decode time {KIND_NAME[kind]} context {context} M {m}", phase="timing",
                                  ms=float(ms.item()), note="one visit (one layer); M sessions share one cache")
                oracle.check_count(self, ROW, f"decode time {KIND_NAME[kind]} {context} M {m} finite",
                                   int(math.isfinite(float(ms.item())) and bool(torch.isfinite(out.float()).all())),
                                   expected=1)

    def test_prefill_time(self):
        """Prefill time of one visit's 4,096-row call (reported, no verdict): full from 0 (4K), full ending at 128K and
        1M context (the G6 visits' last chunks), sliding past the ring; with the (row, key, query head) pairs per
        second."""
        proj = self.w["proj"]
        n = MAX_PREFILL_ROWS
        for kind, context in ((FULL, n), (FULL, 131072), (FULL, 1048576), (SLIDING, 131072)):
            pos0 = context - n
            keys = pos0 if kind == FULL else RING
            k_cache = _uniform((max(keys, 1), KV_ROW), 51, -0.2, 0.2)
            v_cache = _uniform((max(keys, 1), KV_ROW), 52, -1.0, 1.0)
            slots = torch.arange(max(keys, 1), dtype=torch.int64, device=DEV)  # full: identity slot map
            slot_ptr = slots.data_ptr() if kind == FULL else 0
            table = torch.tensor([[0, n, pos0, k_cache.data_ptr(), v_cache.data_ptr(), slot_ptr]], dtype=torch.int64,
                                 device=DEV)
            q, r = _uniform((n, Q_ROW), 53, -1.0, 1.0), _uniform((n, REL_ROW), 54, -1.0, 1.0)
            k, v = _uniform((n, KV_ROW), 55, -0.2, 0.2), _uniform((n, KV_ROW), 56, -1.0, 1.0)
            sink, gate = _uniform((n, HEADS), 57, -2.0, 2.0), _uniform((n, Q_ROW), 58, -2.0, 2.0)
            out = oracle.empty((n, Q_ROW), BF)
            ms = torch.zeros(1, dtype=torch.float32)
            iters = 1 if context > 131072 else 5
            oracle.call("cere_ktest_inkling_attn_prefill_time", kind, table, 1, n, q, r, proj, k, v, sink, gate, out,
                        iters, ms)
            pairs = HEADS * sum(min(p + 1, WINDOW) if kind == SLIDING else p + 1 for p in range(pos0, context))
            t = float(ms.item())
            oracle.report_row(ROW, f"prefill time {KIND_NAME[kind]} context {context}", phase="timing", ms=t,
                              gpairs_s=pairs / t / 1e6, note="one visit (one layer), 4,096 rows ending at context")
            oracle.check_count(self, ROW, f"prefill time {KIND_NAME[kind]} {context} finite",
                               int(math.isfinite(t) and bool(torch.isfinite(out.float()).all())), expected=1)

    # -- refusals ------------------------------------------------------------------------------------------------------
    def test_argument_refusals(self):
        proj = self.w["proj"]
        s = Session(FULL, 8, 4000, self.w)
        c = Call(FULL, [(s, 0, 8)])
        table, _ = c.table(False)
        out = oracle.empty((8, Q_ROW), BF)
        ws = torch.empty(_ws_bytes(8), dtype=torch.uint8, device=DEV)

        def prefill(kind=FULL, max_rows=8, q_stride=Q_ROW):
            return oracle.call_status("cere_ktest_inkling_attn_prefill", kind, table, 1, max_rows, c.q, q_stride, c.r,
                                      REL_ROW, proj, c.k, KV_ROW, c.v, KV_ROW, c.sink, HEADS, c.gate, Q_ROW, out,
                                      Q_ROW, NONE)

        def decode(total_rows=8, nbytes=ws.numel(), max_pos=8, mutant=NONE):
            return oracle.call_status("cere_ktest_inkling_attn_decode", FULL, table, 1, 8, max_pos, total_rows, c.q,
                                      Q_ROW, c.r, REL_ROW, proj, c.k, KV_ROW, c.v, KV_ROW, c.sink, HEADS, c.gate,
                                      Q_ROW, out, Q_ROW, ws, nbytes, mutant)

        cases = {
            "prefill kind 2": lambda: prefill(kind=2),
            "prefill max_rows 4097": lambda: prefill(max_rows=MAX_PREFILL_ROWS + 1),
            "prefill q_stride 3071": lambda: prefill(q_stride=Q_ROW - 1),
            "decode total_rows 65": lambda: decode(total_rows=MAX_DECODE_ROWS + 1),
            "decode workspace short": lambda: decode(nbytes=ws.numel() - 1),
            "decode max_pos 0": lambda: decode(max_pos=0),
            "decode mutant lane sum (prefill only)": lambda: decode(mutant=LANE_SUM),
            "post q without scale": lambda: oracle.call_status(
                "cere_ktest_inkling_attn_post", FULL, 8, s.positions, c.q, None, out, None, None, None, None, None,
                None, None, NONE),
            "kv_store max_rows 0": lambda: oracle.call_status(
                "cere_ktest_inkling_attn_kv_store", FULL, table, 1, 0, c.k, KV_ROW, c.v, KV_ROW),
        }
        for name, fn in cases.items():
            oracle.check_count(self, ROW, f"refuse {name}", fn(), expected=oracle.CERE_ERR_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
