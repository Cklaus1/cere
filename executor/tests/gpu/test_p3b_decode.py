"""P3b decode GEMMs (tasks/m23-spec.md P14, executor/cuda/src/p3b/p3b_decode.h) as kernel-test entry points.

Suite `p3b` (GPU1 kernel lab): `python3 scripts/live/native_dev.py --gpu 1 test --suite p3b` against a
`--moe-backend p3b` build. The decode GEMMs serve m * top_k <= 256 (decode, verify); the model's routed_rows takes
them at m * top_k <= 200 (kRoutedMaxRows: m <= 20, where they beat the grouped GEMMs with a cold L2) in place of
FlashInfer's TMA warp-specialized grouped GEMMs, doActivation and the stride setup; mode 1 forces them at every
served m.
Each asserted bit for bit against FlashInfer's launches on the same routing (`cere_ktest_p3b_decode` mode 0, the
grouped GEMMs, m's pinned pair) on layers 0, 23, 47 with real expert weights:

  * rows: the permuted BF16 expert rows (GEMM2's D, what K7's combine reads), GEMM2's FP4 A and its UE4M3 scale
    factors (FlashInfer's swizzled ones gathered compact), routed_rows on the decode GEMMs (mode 1) vs mode 0, on
    seeded random routings (distinct experts per token) at every DECODE_M and on the block's own routing;
  * GEMM2 alone (mode 2: on FlashInfer's GEMM2 A) and GEMM1 + SwiGLU + FP4 quantize alone (mode 3);
  * determinism (two runs) and a CUDA-graph replay equal to the eager call;
  * the model's gate: routed_rows (RowsPath::kAuto) on the decode GEMMs exactly at m <= 20, the grouped GEMMs at m 25
    (`cere_ktest_p3b_decode_routed`);
  * fail-first: the k64 chain without its last block (Order::kDropLast) is flagged in each of rows, GEMM2 alone and
    GEMM1's FP4 activations. (A reversed k64 order is not a usable perturbation: the 2026-10-16 GPU1 run gave rows,
    GEMM2 and GEMM1 bit-equal to FlashInfer's with the chain reversed on layers 0, 23, 47 at m = 16, consistent with
    the P3 sweep's tile / swap invariance: the FP32 chain of these block-scaled FP4 products rounds the same in any
    order.)

Speed (`test_speed`, reported, not asserted): graph-replay spans of routed_rows (grouped vs decode) and of each decode
GEMM alone, L2-hot (back-to-back replays: one layer's routed weights stay in the 128 MB L2) and cold (each replay
after `cere_ktest_l2_flush`'s 512 MB fill + L2 discard: the model's case, where a layer's weights were last read a full
step earlier), with the bytes moved as TB/s; and each kernel's CUPTI span in cold eager calls of both paths. Only the
cold numbers predict the model: the L2-hot replays put the old moe-gemm2 decode GEMMs ahead of FlashInfer's, the model
ran them 10 % slower (tasks/m23-spec.md P14).
"""
from __future__ import annotations

import importlib
import json
import statistics
import unittest
from pathlib import Path

import torch

import oracle

p3b = importlib.import_module("test_p3b_moe")  # weights, operands, the P3b layers

ROW = "P3B-DECODE"
LAYERS = (0, 23, 47)
DECODE_M = (1, 2, 4, 5, 8, 16, 20, 25)  # decode (1), MTP-draft / verify shapes, the m * top_k <= 256 bound (25 x 10)
ROUTED_MAX_M = 20  # p3b_decode.h kRoutedMaxRows 200 at top_k 10
SEED = 47_000
WARMUP = 10
TIMED = 50
PROFILED = 9  # cold eager calls per path under the CUDA profiler
# The achievable DRAM read stream (scripts/bench/dram_peak.json `peak_bps`, O249), in TB/s.
PEAK_TBS = json.loads((Path(__file__).resolve().parents[3] / "scripts/bench/dram_peak.json").read_text())[
    "peak_bps"] / 1e12
MODES = {"grouped": 0, "decode": 1, "gemm2": 2, "gemm1": 3}


def _routing(m: int, num_experts: int, top_k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded random routing: top_k distinct experts per token, softmax weights."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    ids = torch.stack([torch.randperm(num_experts, generator=g)[:top_k] for _ in range(m)]).to(torch.int32)
    weights = torch.softmax(torch.randn((m, top_k), generator=g), dim=1).to(torch.float32)
    return ids.to(oracle.DEVICE).contiguous(), weights.to(oracle.DEVICE).contiguous()


def _decode(x: torch.Tensor, P: dict, ids: torch.Tensor, weights: torch.Tensor, mode: str, order: int = 0,
            timed: int = 0, cold: bool = False) -> dict:
    m = x.shape[0]
    mk = m * P["top_k"]
    r = {"rows": oracle.empty((mk, P["hidden"]), torch.bfloat16),
         "act": oracle.empty((mk, P["inter"] // 2), torch.uint8),
         "act_sf": oracle.empty((mk, P["inter"] // 16), torch.uint8)}
    us = torch.zeros(max(timed, 1), dtype=torch.float32)
    oracle.call("cere_ktest_p3b_decode", x, P["w13"], P["w13_scale"], P["down"], P["down_scale"], P["g1"], P["g2"],
                P["act"], ids, weights, m, P["hidden"], P["num_experts"], P["top_k"], P["inter"], MODES[mode], order,
                r["rows"], r["act"], r["act_sf"], WARMUP if timed else 0, timed, int(cold), us if timed else None)
    if timed:
        r["us"] = us.tolist()
    return r


def _differs(a: dict, b: dict, fields=("rows", "act", "act_sf")) -> list[str]:
    return [f for f in fields if not torch.equal(a[f].view(torch.uint8), b[f].view(torch.uint8))]


def _cases(P: dict, layer: int):
    """(name, x, ids, weights): seeded random routings at every DECODE_M, then the block's own routing at 16."""
    for m in DECODE_M:
        ids, weights = _routing(m, P["num_experts"], P["top_k"], SEED + 1000 * layer + m)
        yield f"M{m} random", p3b._x(layer, m), ids, weights
    x = p3b._x(layer, 16)
    block = p3b._p3b_block(x, P)
    yield "M16 block", x, block["ids"], block["weights"]


class P3bDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.top_k = oracle.text_config()["num_experts_per_tok"]

    def _layers(self, layers=LAYERS):
        for layer in layers:
            w, s = p3b._load(layer)
            yield layer, p3b._operands(w, s, self.top_k)
            del w, s
            torch.cuda.empty_cache()

    def test_decode_equals_grouped(self) -> None:
        """routed_rows on the decode GEMMs gives FlashInfer's rows, GEMM2 A and scale factors bit for bit, and each
        decode GEMM alone gives its FlashInfer counterpart's output."""
        failed = []
        for layer, P in self._layers():
            equal = cases = 0
            for name, x, ids, weights in _cases(P, layer):
                ref = _decode(x, P, ids, weights, "grouped")
                got = {mode: _decode(x, P, ids, weights, mode) for mode in ("decode", "gemm2", "gemm1")}
                differs = [f"{mode} {f}" for mode, fields in (("decode", ("rows", "act", "act_sf")),
                                                              ("gemm2", ("rows",)), ("gemm1", ("act", "act_sf")))
                           for f in _differs(got[mode], ref, fields)]
                cases += 1
                if differs:
                    failed.append(f"L{layer} {name} {differs[0]}")
                else:
                    equal += 1
            oracle.report_row(ROW, f"bit-equal L{layer}", phase="assert", layer=layer, cases=cases,
                              bit_equal_cases=equal)
        self.assertEqual(failed, [], f"{ROW}: the decode GEMMs differ from FlashInfer's at {failed}")

    def test_routed_gate(self) -> None:
        """The model's routed_rows takes the decode GEMMs at m <= 20 (m * top_k <= 200) and the grouped GEMMs at m 25
        (fail-first: the 256-row gate takes them at every DECODE_M)."""
        for layer, P in self._layers((23,)):
            got = {}
            for m in DECODE_M:
                decoded = torch.full((1,), -1, dtype=torch.int32)
                oracle.call("cere_ktest_p3b_decode_routed", m, P["hidden"], P["num_experts"], P["top_k"], P["inter"],
                            decoded)
                got[m] = int(decoded.item())
            oracle.report_row(ROW, f"routed gate L{layer}", phase="assert", layer=layer,
                              decoded_cases=sum(got.values()), cases=len(got))
            self.assertEqual(got, {m: int(m <= ROUTED_MAX_M) for m in DECODE_M}, f"{ROW} routed gate L{layer}")

    def test_dropped_k_block_is_flagged(self) -> None:
        """Fail-first: the k64 chain without its last block differs from FlashInfer's in the rows (routed_rows and
        GEMM2 alone) and in GEMM1's FP4 activations, on every layer at m = 16."""
        missed = []
        for layer, P in self._layers():
            ids, weights = _routing(16, P["num_experts"], P["top_k"], SEED + 1000 * layer + 16)
            x = p3b._x(layer, 16)
            ref = _decode(x, P, ids, weights, "grouped")
            flagged = {"decode": _differs(_decode(x, P, ids, weights, "decode", order=1), ref, ("rows",)),
                       "gemm2": _differs(_decode(x, P, ids, weights, "gemm2", order=1), ref, ("rows",)),
                       "gemm1": _differs(_decode(x, P, ids, weights, "gemm1", order=1), ref, ("act",))}
            clean = _differs(_decode(x, P, ids, weights, "decode"), ref)
            oracle.report_row(ROW, f"drop-last L{layer} M16", phase="assert", layer=layer,
                              decode_flagged=bool(flagged["decode"]), gemm2_flagged=bool(flagged["gemm2"]),
                              gemm1_flagged=bool(flagged["gemm1"]), clean_flagged_fields=len(clean))
            missed += [f"L{layer} {mode}" for mode, fields in flagged.items() if not fields]
            self.assertEqual(clean, [], f"{ROW} drop-last: the full chain flagged {clean} at L{layer}")
        self.assertEqual(missed, [], f"{ROW} drop-last: the dropped k64 block was not flagged at {missed}")

    def test_repeat_and_graph_equal_eager(self) -> None:
        """Two eager runs, and a CUDA-graph replay of each decode mode, are bit-equal."""
        failed = []
        for layer, P in self._layers():
            for m in (1, 16, 25):
                ids, weights = _routing(m, P["num_experts"], P["top_k"], SEED + 1000 * layer + m)
                x = p3b._x(layer, m)
                for mode in ("decode", "gemm2", "gemm1"):
                    a = _decode(x, P, ids, weights, mode)
                    b = _decode(x, P, ids, weights, mode)
                    g = _decode(x, P, ids, weights, mode, timed=2)
                    if _differs(a, b):
                        failed.append(f"L{layer} M{m} {mode} repeat {_differs(a, b)[0]}")
                    if _differs(a, g):
                        failed.append(f"L{layer} M{m} {mode} graph {_differs(a, g)[0]}")
            oracle.report_row(ROW, f"repeat graph L{layer}", phase="assert", layer=layer,
                              failed_cases=sum(f.startswith(f"L{layer} ") for f in failed))
        self.assertEqual(failed, [], f"{ROW} determinism: {failed}")

    def test_speed(self) -> None:
        """Reported: graph-replay mean spans of routed_rows (grouped vs decode) and of each decode GEMM alone, L2-hot
        and cold, the GEMMs' as TB/s of the weight, scale-factor and activation bytes they move (means: the event
        timer ticks in microseconds); each path's two GEMM kernels' CUPTI medians over cold eager calls."""
        for layer, P in self._layers((23,)):
            H, inter, E, k = P["hidden"], P["inter"], P["num_experts"], P["top_k"]
            for m in DECODE_M:
                ids, weights = _routing(m, E, k, SEED + 1000 * layer + m)
                x = p3b._x(layer, m)
                mk = m * k
                active = int(torch.unique(ids).numel())
                g1_bytes = active * (2 * inter * H // 2 + 2 * inter * H // 16) + mk * (H // 2 + H // 16 + inter // 2
                                                                                      + inter // 16)
                g2_bytes = active * (H * inter // 2 + H * inter // 16) + mk * (inter // 2 + inter // 16 + 2 * H)
                fields = {}
                for cold in (False, True):
                    tag = "cold" if cold else "hot"
                    for mode in MODES:
                        us = _decode(x, P, ids, weights, mode, timed=TIMED, cold=cold)["us"]
                        fields[f"{mode}_{tag}_us"] = round(statistics.mean(us), 2)
                    fields[f"gemm1_{tag}_tbs"] = round(g1_bytes / fields[f"gemm1_{tag}_us"] / 1e6, 3)
                    fields[f"gemm2_{tag}_tbs"] = round(g2_bytes / fields[f"gemm2_{tag}_us"] / 1e6, 3)
                # Each path's GEMM kernels in cold eager calls (CUPTI spans, profiled): GEMM1 moves more bytes than
                # GEMM2, so the longer of a path's two GEMM kernels' median spans is its GEMM1's.
                for path in ("grouped", "decode"):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(PROFILED):
                            oracle.call("cere_ktest_l2_flush")
                            _decode(x, P, ids, weights, path)
                    by_name: dict[str, list[float]] = {}
                    for e in prof.events():
                        if e.device_type == torch.autograd.DeviceType.CUDA and "gemm" in e.name.lower():
                            by_name.setdefault(e.name, []).append(e.time_range.elapsed_us())
                    spans = sorted((statistics.median(v) for v in by_name.values()), reverse=True)
                    if len(spans) == 2:
                        fields[f"{path}_gemm1_cold_us"] = round(spans[0], 2)
                        fields[f"{path}_gemm2_cold_us"] = round(spans[1], 2)
                        fields[f"{path}_gemm1_cold_tbs"] = round(g1_bytes / spans[0] / 1e6, 3)
                        fields[f"{path}_gemm2_cold_tbs"] = round(g2_bytes / spans[1] / 1e6, 3)
                    else:
                        fields[f"{path}_gemm_kernels"] = len(spans)
                for name in [f for f in fields if f.endswith("_tbs")]:
                    fields[name.replace("_tbs", "_peak_pct")] = round(100 * fields[name] / PEAK_TBS, 1)
                oracle.report_row(ROW, f"speed L{layer} M{m}", phase="speed", layer=layer, m=m, active_experts=active,
                                  gemm1_bytes=g1_bytes, gemm2_bytes=g2_bytes, **fields)


if __name__ == "__main__":
    unittest.main()
