"""KRES: WeightLab interventions (tasks/weightlab-spec.md WL-05..08, WL-10, track WL-B; research/patch.h,
research/patch_check.h and the MoE post-route hook kernels::MoeTaps.route), as kernel-test entry points only.

Suite `kernels_research` (GPU0 dev loop: `native_dev.py test --suite kernels_research`):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_research_*.py' -v` (this file also runs in
`--suite kernels`). It needs the shipped `p3b` build (`cere_ktest_research_p3b_moe_block` is CERE_ERR_UNSUPPORTED in
a `k7` build, which fails the P3b rows).

Contract (research/patch.h), every row asserted bit for bit against a reference computed here:
  add          y = bf16_rn(f32(y) + f32(alpha * v[c])), the y bits kept where the FP32 delta is +-0 (alpha 0 is an
               exact no-op);
  ablate_zero  y = +0; ablate_mean y = v[c]; replace y = v[src + i, c] (a one-row vector broadcasts);
  expert_bias  logit[e] = bf16_rn(f32(logit[e]) + b_e), kept where b_e is 0; expert_mask logit[e] = 0xFF7F (the most
               negative finite BF16);
  expert_force per selected row: slots 0..k-1 the forced (ids, weights); then the router's picks in slot order that
               are not forced, the first K - k, each weight f32(w_j * r), r = f32((1 - sum forced) / sum kept) (FP32
               sums in slot order; 0 when either is not positive);
  every kind leaves the rows outside its row ranges and the columns outside [col_begin, col_end) bit-unchanged.

Rows (row key KRES):
  point kinds, asserted (`test_point_kinds_bit_exact`): every dense kind on POINT_SHAPES (cols of the mix outputs,
      gemm.attn_qkv.out and the router logits) with POINT_RANGES (several disjoint ranges, a replace source offset),
      a column range and the whole point; the logit kinds on the router logits [n, 512];
  no-op, asserted (`test_noop_edits_keep_bits`): add with alpha 0 (and with a vector of +-0 / denormal products) and
      an empty expert_mask / expert_bias leave every bit (-0 and NaN payloads included) unchanged (WL-16's no-op plan);
  validation, asserted (`test_check_edit_refusals`): research::check_edit accepts each kind's valid operands and
      refuses every out-of-contract field (CHECK_CASES);
  expert force, asserted (`test_route_force_bit_exact`): research::apply_route on router outputs of real logits
      (`cere_ktest_moe_router`) against the reference above, k in {1, 2, K}, partial forced weight sums and a full
      one (r = 0), selected row ranges only;
  post-route hook, asserted (WL-08), on HOOK_LAYERS x HOOK_M, the K7 block (`cere_ktest_research_moe_block`) and the
      shipped P3b block (`cere_ktest_research_p3b_moe_block`):
      - `test_hook_noop_bit_equal_*`: a hook that only reads the routing gives the base call's output (no taps at
        all) bit for bit, serial and with the shared-expert fork (m <= FORK_ROWS); the routing it reads is the
        router's (`cere_ktest_moe_router` on the block's logits is not reachable here, so P3b's own topk copy-out is
        the witness for that backend);
      - `test_hook_forced_routing_reaches_experts_*`: the block run with the router rows permuted (a seeded
        permutation of the experts: other ids and weights) records its routing; the block with the original router
        and a hook that writes that routing gives the permuted block's output and routed rows bit for bit (the
        reference routes the same ids/weights through the unchanged expert path). Fail-first
        (`test_hook_forced_routing_detects_weight_perturbation`): writing the same routing with one row's slot-0 weight
        2^-7 relative up (its FP32 bits + 2^16; one FP32 ulp can vanish in the BF16 routed rows) is flagged in exactly
        that row's routed row;
      - `test_hook_expert_force_p3b_equals_runmoe`: expert_force through the hook (apply_route) yields the reference
        routing above, and the block's routed rows and output equal runMoe's finalize on that routing fed through
        K7's routed_in path (`test_p3b_moe._runmoe_block`), bit for bit.

Native entry points (C prototypes; 0 = success, else a cere_status; row ranges are host int64 [count, 3] {begin, end,
src}):

  int cere_ktest_research_check(int64_t kind, int64_t router, int64_t point_cols, int64_t top_k, float alpha,
                                int64_t col_begin, int64_t col_end, int64_t vector_rows, int64_t vector_cols,
                                const void *experts /* host int32 */, const void *weights /* host FP32 */,
                                int64_t count, cudaStream_t stream /* unused */);
  int cere_ktest_research_point(int64_t kind, float alpha, int64_t col_begin, int64_t col_end, const void *vector,
                                int64_t vector_rows, const void *experts, const void *weights, int64_t count,
                                void *values /* BF16 [n, cols], in place */, int64_t n, int64_t cols,
                                const int64_t *ranges, int64_t range_count, cudaStream_t stream);
  int cere_ktest_research_route(const void *experts, const void *weights, int64_t count, void *topk_ids,
                                void *topk_weights /* [n, top_k], in place */, int64_t n, int64_t top_k,
                                const int64_t *ranges, int64_t range_count, cudaStream_t stream);
  int cere_ktest_research_moe_block(... cere_ktest_moe_block's weights up to shared_expert_gate ..., void *out,
                                    void *routed, void *ids_out, void *weights_out, int64_t mode, const void *set_ids,
                                    const void *set_weights, const void *force_experts, const void *force_weights,
                                    int64_t force_count, const int64_t *ranges, int64_t range_count, int64_t fork,
                                    int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
                                    int64_t shared_inter, cudaStream_t stream);
      mode 0: no taps (routed null); 1: a recording hook; 2: the hook writes set_ids / set_weights [m, top_k];
      3: expert_force on `ranges`. Modes 1-3 record the routing the experts run into ids_out / weights_out.
  int cere_ktest_research_p3b_moe_block(... cere_ktest_p3b_moe_block's weights up to shared_expert_gate ...,
                                        then as cere_ktest_research_moe_block from `out`);
"""
from __future__ import annotations

import importlib
import struct
import unittest

import torch
import torch.nn.functional as F

import oracle

k7 = importlib.import_module("test_kernels_k7_moe")  # the K7 suite's inputs and weights (F8)
p3b = importlib.import_module("test_p3b_moe")  # P3b's arena operands and runMoe reference (P13)

ROW = "KRES"
ADD, ABLATE_ZERO, ABLATE_MEAN, REPLACE, EXPERT_BIAS, EXPERT_MASK, EXPERT_FORCE = range(1, 8)
MOST_NEGATIVE_BF16 = 0xFF7F
POINT_SEED = 57_000
POINT_ROWS = 40
POINT_SHAPES = (2560, 13312, 512)  # mix outputs, gemm.attn_qkv.out, router logits
# Row ranges {begin, end, src}: disjoint, ordered; src is the replace vector row of `begin`.
POINT_RANGES = ((0, 3, 0), (7, 8, 3), (20, 39, 4))
POINT_COLS = ((0, 0), (128, 384))  # every column; a head-sized column range
ALPHAS = (1.0, -3.5, 0.0078125)
LOGIT_EXPERTS = (0, 7, 100, 511)
LOGIT_BIAS = (2.5, -1.0, 0.0, 1e-3)
HOOK_LAYERS = (0, 23, 47)
HOOK_M = (1, 4, 16, 64, 2048)
FORK_ROWS = 16
PERMUTE_SEED = 57_100
PERTURB = (23, 16, 5)  # fail-first: (layer, m, row) whose written slot-0 weight is 2^-7 relative up
PERTURB_BITS = 1 << 16  # added to the FP32 weight's bits: 2^-7 relative, above the BF16 routed rows' 2^-8 rounding
# k = K (10) forces the whole routing (kept weights scaled by 0); 0.0625 x 10 sums exactly (check_edit: <= 1).
FORCE_CASES = (((3,), (0.5,)), ((3, 400), (0.25, 0.25)), (tuple(range(10)), (0.0625,) * 10))
FORCE_RANGES = ((0, 2, 0), (5, 16, 0))


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _ranges(ranges) -> torch.Tensor:
    return torch.tensor([v for r in ranges for v in r], dtype=torch.int64)


def _selected(ranges) -> list[tuple[int, int]]:
    """(row, offset into the range) for every selected row."""
    return [(row, row - b) for b, e, _ in ranges for row in range(b, e)]


def _values(n: int, cols: int, seed: int) -> torch.Tensor:
    """Seeded BF16 rows with -0, a NaN payload and +-Inf planted (they must survive untouched where not edited)."""
    v = torch.randn((n, cols), generator=oracle.generator(seed), device=oracle.DEVICE).mul_(4).to(torch.bfloat16)
    bits = v.view(torch.int16)
    bits[1, 1] = -32768  # -0
    bits[5, 2] = 0x7FC1  # NaN with a payload, in a row no range selects (an edited NaN's payload is not specified)
    v[3, 3] = float("inf")
    v[4, 4] = float("-inf")
    return v.contiguous()


def _point(kind: int, values: torch.Tensor, ranges, *, alpha: float = 0.0, cols=(0, 0), vector=None,
           experts=None, weights=None) -> torch.Tensor:
    out = values.clone()
    n, c = out.shape
    count = 0 if experts is None else experts.numel()
    oracle.call("cere_ktest_research_point", kind, float(alpha), cols[0], cols[1], vector,
                0 if vector is None else vector.shape[0], experts, weights, count, out, n, c, _ranges(ranges),
                len(ranges))
    return out


def _dense_reference(kind: int, values: torch.Tensor, ranges, alpha: float, cols, vector) -> torch.Tensor:
    ref = values.clone()
    b, e = (0, values.shape[1]) if cols == (0, 0) else cols
    for row, off in _selected(ranges):
        src = next(r[2] for r in ranges if r[0] <= row < r[1]) + off
        seg = ref[row, b:e]
        if kind == ADD:
            d = vector[0, b:e].float() * alpha
            keep = d == 0
            new = (seg.float() + d).to(torch.bfloat16)
            ref[row, b:e] = torch.where(keep, seg, new)
        elif kind == ABLATE_ZERO:
            ref[row, b:e] = torch.zeros_like(seg)
        elif kind == ABLATE_MEAN:
            ref[row, b:e] = vector[0, b:e]
        else:
            ref[row, b:e] = vector[src if vector.shape[0] > 1 else 0, b:e]
    return ref


def _force_reference(ids: torch.Tensor, weights: torch.Tensor, experts, forced, ranges):
    ids = ids.cpu().clone()
    weights = weights.cpu().clone()
    k = len(experts)
    top_k = ids.shape[1]
    forced_sum = 0.0
    for w in forced:
        forced_sum = _f32(forced_sum + _f32(w))
    rest = _f32(1.0 - forced_sum)
    for row, _ in _selected(ranges):
        kept = []
        kept_sum = 0.0
        for j in range(top_k):
            if len(kept) + k >= top_k:
                break
            e = int(ids[row, j])
            if e in experts:
                continue
            w = float(weights[row, j])
            kept.append((e, w))
            kept_sum = _f32(kept_sum + w)
        scale = _f32(rest / kept_sum) if rest > 0 and kept_sum > 0 else 0.0
        for f in range(k):
            ids[row, f] = experts[f]
            weights[row, f] = _f32(forced[f])
        for t, (e, w) in enumerate(kept):
            ids[row, k + t] = e
            weights[row, k + t] = _f32(w * scale)
    return ids, weights


def _k7_args(w: dict, s: dict) -> tuple:
    return (w["router"], w["gate_weight"], w["gate_weight_scale"], w["up_weight"], w["up_weight_scale"],
            w["down_weight"], w["down_weight_scale"], s["g1_alphas"], s["g2_alphas"],
            float(s["w13_input_scale_quant"].item()), float(s["w2_input_scale_quant"].item()), w["shared_gate"],
            w["shared_up"], w["shared_down"], w["shared_expert_gate"])


def _p3b_args(P: dict) -> tuple:
    return (P["router"], P["w13"], P["w13_scale"], P["down"], P["down_scale"], P["g1"], P["g2"], P["act"],
            P["shared_gate"], P["shared_up"], P["shared_down"], P["shared_expert_gate"])


class Layer:
    """One MoE layer's operands for both backends; `permuted` swaps in a router with its expert rows permuted."""

    def __init__(self, layer: int, top_k: int) -> None:
        self.w = k7._load_moe_layer(layer)
        self.s = k7._reference_scales(self.w)
        self.P = p3b._operands(self.w, self.s, top_k)
        self.num_experts, self.inter, half = self.w["gate_weight"].shape
        self.hidden = 2 * half
        self.shared_inter = self.w["shared_gate"].shape[0]
        self.top_k = top_k
        perm = torch.randperm(self.num_experts, generator=torch.Generator().manual_seed(PERMUTE_SEED + layer))
        self.router_permuted = self.w["router"][perm.to(self.w["router"].device)].contiguous()


def _block(L: Layer, backend: str, x: torch.Tensor, mode: int, *, permuted: bool = False, routed: bool = True,
           set_ids=None, set_weights=None, force=None, ranges=(), fork: int = 0) -> dict:
    """One hooked block call (`cere_ktest_research_moe_block` K7 / `cere_ktest_research_p3b_moe_block` P3b), mode as
    the entry points'; `routed` allocates MoeTaps.routed (modes 1-3)."""
    m = x.shape[0]
    r = {"out": oracle.empty((m, L.hidden), torch.bfloat16),
         "routed": oracle.empty((m, L.hidden), torch.bfloat16) if routed and mode else None,
         "ids": oracle.empty((m, L.top_k), torch.int32) if mode else None,
         "weights": oracle.empty((m, L.top_k), torch.float32) if mode else None}
    experts, forced = (None, None) if force is None else force
    tail = (r["out"], r["routed"], r["ids"], r["weights"], mode, set_ids, set_weights, experts, forced,
            0 if experts is None else experts.numel(), _ranges(ranges) if ranges else None, len(ranges), fork,
            m, L.hidden, L.num_experts, L.top_k, L.inter, L.shared_inter)
    if backend == "k7":
        args = list(_k7_args(L.w, L.s))
        if permuted:
            args[0] = L.router_permuted
        oracle.call("cere_ktest_research_moe_block", x, *args, *tail)
    else:
        args = list(_p3b_args(L.P))
        if permuted:
            args[0] = L.router_permuted
        oracle.call("cere_ktest_research_p3b_moe_block", x, *args, *tail)
    return r


def _x(layer: int, m: int) -> torch.Tensor:
    return p3b._x(layer, m)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _differing_rows(a: torch.Tensor, b: torch.Tensor) -> list[int]:
    av = a.contiguous().view(torch.uint8).reshape(a.shape[0], -1)
    bv = b.contiguous().view(torch.uint8).reshape(b.shape[0], -1)
    return [int(i) for i in (av != bv).any(dim=1).nonzero().flatten().tolist()]


class ResearchKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = oracle.text_config()
        cls.top_k = cfg["num_experts_per_tok"]
        cls.num_experts = cfg["num_experts"]

    def test_point_kinds_bit_exact(self) -> None:
        cases = 0
        for cols in POINT_SHAPES:
            values = _values(POINT_ROWS, cols, POINT_SEED + cols)
            gen = oracle.generator(POINT_SEED + 1 + cols)
            vec = torch.randn((1, cols), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
            rows = torch.randn((32, cols), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
            for col in POINT_COLS:
                plans = [(ADD, a, vec) for a in ALPHAS] + [(ABLATE_ZERO, 0.0, None), (ABLATE_MEAN, 0.0, vec),
                                                          (REPLACE, 0.0, rows), (REPLACE, 0.0, vec)]
                for kind, alpha, vector in plans:
                    case = f"kind {kind} cols {cols} [{col[0]},{col[1]}) alpha {alpha} vrows " \
                           f"{0 if vector is None else vector.shape[0]}"
                    native = _point(kind, values, POINT_RANGES, alpha=alpha, cols=col, vector=vector)
                    oracle.check_bit_exact(self, ROW, case, native,
                                           _dense_reference(kind, values, POINT_RANGES, alpha, col, vector))
                    cases += 1
        logits = _values(POINT_ROWS, self.num_experts, POINT_SEED + 9)
        experts = torch.tensor(LOGIT_EXPERTS, dtype=torch.int32, device=oracle.DEVICE)
        bias = torch.tensor(LOGIT_BIAS, dtype=torch.float32, device=oracle.DEVICE)
        ref_bias = logits.clone()
        ref_mask = logits.clone()
        for row, _ in _selected(POINT_RANGES):
            for e, b in zip(LOGIT_EXPERTS, LOGIT_BIAS):
                if b != 0.0:
                    ref_bias[row, e] = (logits[row, e].float() + torch.tensor(b, dtype=torch.float32)).to(
                        torch.bfloat16)
                ref_mask.view(torch.int16)[row, e] = MOST_NEGATIVE_BF16 - 0x10000
        oracle.check_bit_exact(self, ROW, "expert_bias", _point(EXPERT_BIAS, logits, POINT_RANGES, experts=experts,
                                                                weights=bias), ref_bias)
        oracle.check_bit_exact(self, ROW, "expert_mask", _point(EXPERT_MASK, logits, POINT_RANGES, experts=experts),
                               ref_mask)
        oracle.report_row(ROW, "point kinds", phase="assert", dense_cases=cases, logit_cases=2)

    def test_noop_edits_keep_bits(self) -> None:
        for cols in POINT_SHAPES:
            values = _values(POINT_ROWS, cols, POINT_SEED + 20 + cols)
            vec = torch.randn((1, cols), generator=oracle.generator(POINT_SEED + 21), device=oracle.DEVICE)
            vec = vec.to(torch.bfloat16)
            vec.view(torch.int16)[0, :8] = torch.tensor([-32768, 0, 1, -32767, 0x0080, 0x8080 - 0x10000, 2, 3],
                                                        dtype=torch.int16)  # +-0, denormals, tiny normals
            planted = values.clone()
            planted.view(torch.int16)[0, 5] = 0x7FC1  # a NaN payload in a selected row: alpha 0 keeps it too
            alpha_zero = _point(ADD, planted, POINT_RANGES, alpha=0.0, vector=vec)
            self.assertTrue(_bits_equal(alpha_zero, planted), f"{ROW} add alpha 0 cols {cols} changed bits")
            tiny = _point(ADD, values, POINT_RANGES, alpha=1e-30, vector=vec)
            ref = _dense_reference(ADD, values, POINT_RANGES, 1e-30, (0, 0), vec)
            oracle.check_bit_exact(self, ROW, f"add underflowing delta cols {cols}", tiny, ref)
        logits = _values(POINT_ROWS, self.num_experts, POINT_SEED + 30)
        logits.view(torch.int16)[0, 5] = 0x7FC1
        empty_i = torch.empty(0, dtype=torch.int32, device=oracle.DEVICE)
        empty_f = torch.empty(0, dtype=torch.float32, device=oracle.DEVICE)
        mask = _point(EXPERT_MASK, logits, POINT_RANGES, experts=empty_i)
        bias = _point(EXPERT_BIAS, logits, POINT_RANGES, experts=empty_i, weights=empty_f)
        zero_bias = _point(EXPERT_BIAS, logits, POINT_RANGES,
                           experts=torch.tensor(LOGIT_EXPERTS, dtype=torch.int32, device=oracle.DEVICE),
                           weights=torch.zeros(len(LOGIT_EXPERTS), dtype=torch.float32, device=oracle.DEVICE))
        for name, t in (("empty mask", mask), ("empty bias", bias), ("zero bias", zero_bias)):
            self.assertTrue(_bits_equal(t, logits), f"{ROW} {name} changed bits")
        oracle.report_row(ROW, "no-op edits", phase="assert", shapes=len(POINT_SHAPES), logit_cases=3)

    def _check(self, kind, router, point_cols, alpha=0.0, cols=(0, 0), vrows=0, vcols=0, experts=None,
               weights=None) -> int:
        e = None if experts is None else torch.tensor(experts, dtype=torch.int32)
        w = None if weights is None else torch.tensor(weights, dtype=torch.float32)
        return oracle.call_status("cere_ktest_research_check", kind, int(router), point_cols, self.top_k,
                                  float(alpha), cols[0], cols[1], vrows, vcols, e, w,
                                  0 if experts is None else len(experts))

    def test_check_edit_refusals(self) -> None:
        E = self.num_experts
        accepted = [
            (ADD, False, 2560, 1.0, (0, 0), 1, 2560), (ADD, True, E, 0.0, (0, 0), 1, E),
            (ABLATE_ZERO, False, 13312, 0.0, (128, 256), 0, 0), (ABLATE_MEAN, False, 2560, 0.0, (0, 0), 1, 2560),
            (REPLACE, False, 2560, 0.0, (0, 2560), 7, 2560),
        ]
        refused = [
            (0, False, 2560, 0.0, (0, 0), 1, 2560), (8, False, 2560, 0.0, (0, 0), 1, 2560),
            (ADD, False, 2560, float("inf"), (0, 0), 1, 2560), (ADD, False, 2560, float("nan"), (0, 0), 1, 2560),
            (ADD, False, 2560, 1.0, (0, 0), 1, 2559), (ADD, False, 2560, 1.0, (0, 0), 2, 2560),
            (ADD, False, 2560, 1.0, (5, 5), 1, 2560), (ADD, False, 2560, 1.0, (0, 2561), 1, 2560),
            (ADD, False, 2560, 1.0, (10, 5), 1, 2560), (ADD, False, 0, 1.0, (0, 0), 1, 0),
            (ABLATE_ZERO, False, 2560, 0.0, (0, 0), 1, 2560), (ABLATE_ZERO, False, 2560, 1.0, (0, 0), 0, 0),
            (ABLATE_MEAN, False, 2560, 0.0, (0, 0), 0, 2560), (ABLATE_MEAN, False, 2560, 2.0, (0, 0), 1, 2560),
            (REPLACE, False, 2560, 0.0, (0, 0), 0, 2560), (REPLACE, False, 2560, 0.0, (0, 0), 4097, 2560),
        ]
        statuses = {}
        for case in accepted:
            statuses[("ok",) + case] = self._check(*case)
        statuses[("ok", "mask")] = self._check(EXPERT_MASK, True, E, experts=[1, 2, 3])
        statuses[("ok", "bias")] = self._check(EXPERT_BIAS, True, E, experts=[1, 2], weights=[1.0, -2.0])
        statuses[("ok", "empty mask")] = self._check(EXPERT_MASK, True, E, experts=[])
        statuses[("ok", "force")] = self._check(EXPERT_FORCE, False, E, experts=[4, 5], weights=[0.5, 0.5])
        statuses[("ok", "force full")] = self._check(EXPERT_FORCE, False, E, experts=list(range(10)),
                                                     weights=[0.0625] * 10)
        for name, status in statuses.items():
            self.assertEqual(status, 0, f"{ROW} check_edit refused valid {name}")
        refusals = {case: self._check(*case) for case in refused}
        refusals["mask router false"] = self._check(EXPERT_MASK, False, E, experts=[1])
        refusals["force router true"] = self._check(EXPERT_FORCE, True, E, experts=[1], weights=[1.0])
        refusals["mask duplicate"] = self._check(EXPERT_MASK, True, E, experts=[1, 1])
        refusals["mask out of range"] = self._check(EXPERT_MASK, True, E, experts=[E])
        refusals["mask negative"] = self._check(EXPERT_MASK, True, E, experts=[-1])
        refusals["mask with weights"] = self._check(EXPERT_MASK, True, E, experts=[1], weights=[1.0])
        refusals["mask leaves < top_k"] = self._check(EXPERT_MASK, True, E, experts=list(range(E - self.top_k + 1)))
        refusals["mask alpha"] = self._check(EXPERT_MASK, True, E, alpha=1.0, experts=[1])
        refusals["mask cols"] = self._check(EXPERT_MASK, True, E, cols=(0, 4), experts=[1])
        refusals["mask vector"] = self._check(EXPERT_MASK, True, E, vrows=1, vcols=E, experts=[1])
        refusals["bias no weights"] = self._check(EXPERT_BIAS, True, E, experts=[1])
        refusals["bias inf"] = self._check(EXPERT_BIAS, True, E, experts=[1], weights=[float("inf")])
        refusals["force empty"] = self._check(EXPERT_FORCE, False, E, experts=[], weights=[])
        refusals["force > top_k"] = self._check(EXPERT_FORCE, False, E, experts=list(range(self.top_k + 1)),
                                                weights=[0.0] * (self.top_k + 1))
        refusals["force sum > 1"] = self._check(EXPERT_FORCE, False, E, experts=[1, 2], weights=[0.75, 0.5])
        refusals["force negative"] = self._check(EXPERT_FORCE, False, E, experts=[1, 2], weights=[-0.1, 0.5])
        refusals["force nan"] = self._check(EXPERT_FORCE, False, E, experts=[1], weights=[float("nan")])
        refusals["force no weights"] = self._check(EXPERT_FORCE, False, E, experts=[1])
        accepted_wrongly = [str(k) for k, s in refusals.items() if s != -1]
        oracle.report_row(ROW, "check_edit", phase="assert", accepted=len(statuses), refused=len(refusals),
                          accepted_wrongly=len(accepted_wrongly))
        self.assertEqual(accepted_wrongly, [], f"{ROW} check_edit did not refuse (CERE_ERR_ARGUMENT)")

    def test_route_force_bit_exact(self) -> None:
        m = 16
        x = _x(23, m)
        logits = F.linear(x, k7._router_weight(23)).contiguous()
        cases = 0
        for experts, forced in FORCE_CASES:
            ids = oracle.empty((m, self.top_k), torch.int32)
            weights = oracle.empty((m, self.top_k), torch.float32)
            oracle.call("cere_ktest_moe_router", logits, ids, weights, m, self.num_experts, self.top_k)
            ref_ids, ref_weights = _force_reference(ids, weights, experts, forced, FORCE_RANGES)
            oracle.call("cere_ktest_research_route",
                        torch.tensor(experts, dtype=torch.int32, device=oracle.DEVICE),
                        torch.tensor(forced, dtype=torch.float32, device=oracle.DEVICE), len(experts), ids, weights,
                        m, self.top_k, _ranges(FORCE_RANGES), len(FORCE_RANGES))
            oracle.check_bit_exact(self, ROW, f"force k{len(experts)} ids", ids.cpu(), ref_ids)
            oracle.check_bit_exact(self, ROW, f"force k{len(experts)} weights", weights.cpu(), ref_weights)
            cases += 1
        oracle.report_row(ROW, "expert force", phase="assert", cases=cases, m=m)

    def _hook_noop(self, backend: str) -> None:
        failed = []
        cases = 0
        for layer in HOOK_LAYERS:
            L = Layer(layer, self.top_k)
            for m in HOOK_M:
                x = _x(layer, m)
                for fork in ((0, 1) if m <= FORK_ROWS else (0,)):
                    base = _block(L, backend, x, 0, fork=fork)
                    hooked = _block(L, backend, x, 1, routed=False, fork=fork)
                    cases += 1
                    if not _bits_equal(base["out"], hooked["out"]):
                        failed.append(f"L{layer} M{m} fork{fork}")
            del L
            torch.cuda.empty_cache()
        oracle.report_row(ROW, f"hook no-op {backend}", phase="assert", cases=cases, differing=len(failed))
        self.assertEqual(failed, [], f"{ROW} {backend}: a read-only route hook changed the block output")

    def test_hook_noop_bit_equal_k7(self) -> None:
        self._hook_noop("k7")

    def test_hook_noop_bit_equal_p3b(self) -> None:
        self._hook_noop("p3b")

    def _hook_forced(self, backend: str) -> None:
        failed = []
        cases = 0
        for layer in HOOK_LAYERS:
            L = Layer(layer, self.top_k)
            for m in HOOK_M:
                x = _x(layer, m)
                ref = _block(L, backend, x, 1, permuted=True)
                base = _block(L, backend, x, 1)
                forced = _block(L, backend, x, 2, set_ids=ref["ids"], set_weights=ref["weights"])
                cases += 1
                moved = not _bits_equal(base["ids"], ref["ids"])
                for field in ("ids", "weights", "routed", "out"):
                    if not _bits_equal(forced[field], ref[field]):
                        failed.append(f"L{layer} M{m} {field}")
                        break
                if not moved:
                    failed.append(f"L{layer} M{m} permutation left the routing unchanged")
            del L
            torch.cuda.empty_cache()
        oracle.report_row(ROW, f"hook forced routing {backend}", phase="assert", cases=cases, failed=len(failed))
        self.assertEqual(failed, [], f"{ROW} {backend}: forced routing differs from the block that routes it")

    def test_hook_forced_routing_reaches_experts_k7(self) -> None:
        self._hook_forced("k7")

    def test_hook_forced_routing_reaches_experts_p3b(self) -> None:
        self._hook_forced("p3b")

    def test_hook_forced_routing_detects_weight_perturbation(self) -> None:
        layer, m, row = PERTURB
        L = Layer(layer, self.top_k)
        x = _x(layer, m)
        flagged = {}
        for backend in ("k7", "p3b"):
            ref = _block(L, backend, x, 1, permuted=True)
            weights = ref["weights"].clone()
            weights.view(torch.int32)[row, 0] += PERTURB_BITS
            perturbed = _block(L, backend, x, 2, set_ids=ref["ids"], set_weights=weights)
            clean = _block(L, backend, x, 2, set_ids=ref["ids"], set_weights=ref["weights"])
            flagged[backend] = (_differing_rows(perturbed["routed"], ref["routed"]),
                                _differing_rows(clean["routed"], ref["routed"]))
        oracle.report_row(ROW, f"hook weight perturbation L{layer} M{m} row {row}", phase="assert",
                          k7_flagged=len(flagged["k7"][0]), p3b_flagged=len(flagged["p3b"][0]),
                          k7_clean=len(flagged["k7"][1]), p3b_clean=len(flagged["p3b"][1]))
        for backend, (perturbed, clean) in flagged.items():
            self.assertEqual(perturbed, [row], f"{ROW} {backend}: the perturbed weight flagged rows {perturbed}")
            self.assertEqual(clean, [], f"{ROW} {backend}: the clean routing flagged rows {clean}")

    def test_hook_expert_force_p3b_equals_runmoe(self) -> None:
        failed = []
        cases = 0
        for layer in HOOK_LAYERS:
            L = Layer(layer, self.top_k)
            for m in (16, 64):
                x = _x(layer, m)
                base = _block(L, "p3b", x, 1)
                for experts, forced in FORCE_CASES:
                    e = torch.tensor(experts, dtype=torch.int32, device=oracle.DEVICE)
                    f = torch.tensor(forced, dtype=torch.float32, device=oracle.DEVICE)
                    r = _block(L, "p3b", x, 3, force=(e, f), ranges=FORCE_RANGES)
                    ref_ids, ref_weights = _force_reference(base["ids"], base["weights"], experts, forced,
                                                            FORCE_RANGES)
                    runmoe = p3b._runmoe_block(x, L.P, r["ids"], r["weights"])
                    cases += 1
                    checks = (("ids", r["ids"].cpu(), ref_ids), ("weights", r["weights"].cpu(), ref_weights),
                              ("routed", r["routed"], runmoe["routed"]), ("out", r["out"], runmoe["out"]))
                    bad = [name for name, a, b in checks if not _bits_equal(a, b)]
                    if bad:
                        failed.append(f"L{layer} M{m} k{len(experts)} {bad[0]}")
            del L
            torch.cuda.empty_cache()
        oracle.report_row(ROW, "expert force p3b vs runMoe", phase="assert", cases=cases, failed=len(failed))
        self.assertEqual(failed, [], f"{ROW} expert_force through the hook differs from runMoe on its routing")


if __name__ == "__main__":
    unittest.main()
