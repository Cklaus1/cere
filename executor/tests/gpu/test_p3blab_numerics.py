"""P3b numerics diagnosis (tasks/m23-spec.md D9, M23-D3): where P3b's MoE block differs from K7's and from SGLang's.

Suite `p3blab` (GPU1 kernel lab): `python3 scripts/live/native_dev.py --gpu 1 test --suite p3blab` against a
`--moe-backend p3b` build. Nothing here is a G1 result and nothing is asserted beyond the calls succeeding: the rows
are the evidence of the D9 diagnosis (tasks/review-m23.md).

Per-op isolation at M = 1 (each token its own call, as a decode step runs it) on real expert weights of every layer,
K7's isolation inputs (`test_kernels_k7_moe._isolate_input`: layer 0 the parity prompts' MoE input, other layers
seeded RMS-matched synthetic rows), against SGLang's own ops on the same inputs (`test_kernels_k7_moe`'s mirror of
`Qwen2MoeSparseMoeBlock.forward`) and the FP64 K-oracle (`_high_precision_block`):
  router:  P3b's routing (K7's router GEMM and `moe_route`) vs SGLang's `F.linear` + `fused_topk`: expert sets, slot
           order, weights bit for bit (aligned by expert);
  routed:  P3b's finalized routed rows vs `cutlass_fused_moe` untuned (tactics (0, 20), fused finalize off) on
           SGLang's routing (rows whose expert sets agree);
  shared:  P3b's shared expert rows (K2 GEMMs + SiLU-mul, the `shared` tap) vs SGLang's (`F.linear` on [gate; up],
           `silu_and_mul`, `F.linear`), each also against FP64;
  combine: P3b's sigmoid-gated combine on SGLang's routed rows and shared rows (`routed_in` / `shared_in` taps) vs
           SGLang's `fused_gate_sigmoid_mul_add` on the same rows;
  block:   P3b's, K7's (`cere_ktest_moe_block`), SGLang (0, 20)'s (untuned, finalize off: the server's prefill path)
           and SGLang tuned's (autotune(True) at M = 1, fused finalize per SGLANG_FLASHINFER_MOE_FUSED_FINALIZE: the
           server's decode path; TUNED_RUNS calls per token) block outputs, pairwise and against FP64.
Then the block rows at prefill M (64, 2048) on layers 0/23/47 (P3b, K7, SGLang (0, 20) against FP64).

Rows (row key P3BNUM, counts and distances only): `<op> L<layer>` per layer and op, `<op> all` over the layers,
`prefill L<layer> M<m>`.

Native entry point (C prototype; 0 = success, else a cere_status):

  int cere_ktest_p3b_moe_block_taps(const void *x, const void *router_weight, const void *w13_weight,
                                    const void *w13_scale, const void *down_weight, const void *down_scale,
                                    const void *g1_alphas, const void *g2_alphas, const void *act_scales,
                                    const void *shared_gate_weight, const void *shared_up_weight,
                                    const void *shared_down_weight, const void *shared_expert_gate, void *out,
                                    void *routed, void *topk_ids, void *topk_weights, const void *routed_in,
                                    const void *shared_in, void *shared, int64_t m, int64_t hidden,
                                    int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
                                    int64_t gemm1, int64_t gemm2 /* both -1: m's pinned pair */,
                                    cudaStream_t stream);
"""
from __future__ import annotations

import importlib
import statistics
import unittest

import torch
import torch.nn.functional as F
from flashinfer.autotuner import AutoTuner, autotune
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.srt.layers.activation import silu_and_mul

import oracle

k7 = importlib.import_module("test_kernels_k7_moe")
p3b = importlib.import_module("test_p3b_moe")

ROW = "P3BNUM"
TUNED_RUNS = 5
PREFILL_LAYERS = (0, 23, 47)
PREFILL_M = (64, 2048)


def _taps(x: torch.Tensor, P: dict, routed_in: torch.Tensor | None = None,
          shared_in: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """`cere_ktest_p3b_moe_block_taps`: the block output, its routing, routed rows and shared rows."""
    m = x.shape[0]
    given = routed_in is not None
    r = {"out": oracle.empty((m, P["hidden"]), torch.bfloat16),
         "shared": oracle.empty((m, P["hidden"]), torch.bfloat16)}
    if not given:
        r["routed"] = oracle.empty((m, P["hidden"]), torch.bfloat16)
        r["route"] = oracle.empty((m, P["top_k"]), torch.int32)
        r["weights"] = oracle.empty((m, P["top_k"]), torch.float32)
    oracle.call("cere_ktest_p3b_moe_block_taps", x, P["router"], P["w13"], P["w13_scale"], P["down"],
                P["down_scale"], P["g1"], P["g2"], P["act"], P["shared_gate"], P["shared_up"], P["shared_down"],
                P["shared_expert_gate"], r["out"], r.get("routed"), r.get("route"), r.get("weights"), routed_in,
                shared_in, r["shared"], m, P["hidden"], P["num_experts"], P["top_k"], P["inter"],
                P["shared_inter"], -1, -1)
    return r


def _sglang_shared(x: torch.Tensor, w: dict) -> torch.Tensor:
    """SGLang's shared expert rows before the gate (`qwen2_moe.py:624-629`)."""
    gate_up = F.linear(x, torch.cat([w["shared_gate"], w["shared_up"]], dim=0))
    return F.linear(silu_and_mul(gate_up), w["shared_down"]).contiguous()


def _fp64_shared(x: torch.Tensor, w: dict) -> torch.Tensor:
    xd = x.double()
    return (F.silu(xd @ w["shared_gate"].double().T) * (xd @ w["shared_up"].double().T)) @ w["shared_down"].double().T


def _sglang_combine(x: torch.Tensor, w: dict, routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    final = routed.clone()
    fused_gate_sigmoid_mul_add(x, w["shared_expert_gate"], shared, final)
    return final


def _bits(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int]:
    """(bit-equal, differing elements) of two BF16 tensors."""
    differing = int((a.view(torch.int16) != b.view(torch.int16)).sum().item())
    return differing == 0, differing


class Acc:
    """Per-op tallies over tokens: counts summed, distances as mean and max."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.values: dict[str, list[float]] = {}

    def count(self, key: str, value: int | bool) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(value)

    def value(self, key: str, value: float) -> None:
        self.values.setdefault(key, []).append(float(value))

    def merge(self, other: "Acc") -> None:
        for k, v in other.counts.items():
            self.count(k, v)
        for k, v in other.values.items():
            self.values.setdefault(k, []).extend(v)

    def fields(self) -> dict:
        out: dict = dict(self.counts)
        for k, v in self.values.items():
            out[f"{k}_mean"] = statistics.fmean(v)
            out[f"{k}_max"] = max(v)
        return out


OPS = ("router", "routed", "shared", "combine", "block")


class P3bNumerics(unittest.TestCase):
    def test_decode_isolation(self) -> None:
        cfg = oracle.text_config()
        top_k = cfg["num_experts_per_tok"]
        totals = {op: Acc() for op in OPS}
        for layer in range(cfg["num_hidden_layers"]):
            w, s = p3b._load(layer)
            ops = k7._cutlass_operands(w, s)
            P = p3b._operands(w, s, top_k)
            acc = {op: Acc() for op in OPS}
            rows = k7._isolate_input(layer)
            kept = []
            AutoTuner.get().clear_cache()
            for t in range(rows.shape[0]):
                x = rows[t:t + 1].contiguous()
                native = _taps(x, P)
                logits = F.linear(x, w["router"])
                sw, sids = k7._sglang_route(x, logits, top_k)
                sids = sids.to(torch.int32)
                # router
                a = acc["router"]
                a.count("tokens_compared", 1)
                n_ids, n_w = native["route"][0], native["weights"][0]
                same_set = torch.equal(n_ids.sort().values, sids[0].sort().values)
                a.count("expert_set_equal", same_set)
                a.count("slot_order_equal", torch.equal(n_ids, sids[0]))
                if same_set:
                    nw = n_w[n_ids.argsort()]
                    rw = sw[0][sids[0].argsort()]
                    a.count("weights_bit_equal", torch.equal(nw.view(torch.int32), rw.view(torch.int32)))
                    a.value("weight_abs_diff", float((nw - rw).abs().max().item()))
                # routed: cutlass (0, 20) on SGLang's routing
                s_routed = p3b._cutlass(x, ops, sids, sw)
                a = acc["routed"]
                a.count("tokens_compared", 1)
                if same_set:
                    equal, differing = _bits(native["routed"], s_routed)
                    a.count("bit_equal_vs_sglang_0_20", equal)
                    a.count("differing_elements", differing)
                own = p3b._cutlass(x, ops, native["route"], native["weights"])
                a.count("bit_equal_on_own_routing", _bits(native["routed"], own)[0])
                # shared expert
                s_shared = _sglang_shared(x, w)
                fp64_shared = _fp64_shared(x, w)
                a = acc["shared"]
                a.count("tokens_compared", 1)
                equal, differing = _bits(native["shared"], s_shared)
                a.count("bit_equal", equal)
                a.count("differing_elements", differing)
                a.value("rel_l2_p3b_vs_sglang", oracle.rel_l2(native["shared"], s_shared))
                a.value("rel_l2_p3b_vs_fp64", oracle.rel_l2(native["shared"].double(), fp64_shared))
                a.value("rel_l2_sglang_vs_fp64", oracle.rel_l2(s_shared.double(), fp64_shared))
                # combine on SGLang's routed and shared rows
                given = _taps(x, P, routed_in=s_routed, shared_in=s_shared)
                s_block = _sglang_combine(x, w, s_routed, s_shared)
                a = acc["combine"]
                a.count("tokens_compared", 1)
                equal, differing = _bits(given["out"], s_block)
                a.count("bit_equal", equal)
                a.count("differing_elements", differing)
                a.value("rel_l2_p3b_vs_sglang", oracle.rel_l2(given["out"], s_block))
                # block
                k7_out = k7._native_block(x, w, s, top_k)
                fp64 = k7._high_precision_block(x, w, s, sw, sids.long())
                a = acc["block"]
                a.count("tokens_compared", 1)
                equal, differing = _bits(native["out"], s_block)
                a.count("p3b_bit_equal_sglang_0_20", equal)
                a.count("p3b_differing_elements_vs_sglang_0_20", differing)
                a.count("k7_bit_equal_sglang_0_20", _bits(k7_out, s_block)[0])
                a.count("k7_differing_elements_vs_sglang_0_20", _bits(k7_out, s_block)[1])
                a.value("rel_l2_p3b_vs_sglang_0_20", oracle.rel_l2(native["out"], s_block))
                a.value("rel_l2_k7_vs_sglang_0_20", oracle.rel_l2(k7_out, s_block))
                a.value("rel_l2_p3b_vs_k7", oracle.rel_l2(native["out"], k7_out))
                for name, value in (("p3b", native["out"]), ("k7", k7_out), ("sglang_0_20", s_block)):
                    a.value(f"rel_l2_{name}_vs_fp64", oracle.rel_l2(value.double(), fp64))
                kept.append((x, native["out"], k7_out, s_block, fp64))
                del native, given, s_routed, s_shared, own
            # SGLang tuned (the server's decode path): tuned once at M = 1 on this layer, TUNED_RUNS calls per token.
            with autotune(True):
                k7._sglang_block(kept[0][0], w, ops, top_k)
            torch.cuda.synchronize(oracle.DEVICE)
            a = acc["block"]
            for x, p3b_out, k7_out, s_block, fp64 in kept:
                runs = [k7._sglang_block(x, w, ops, top_k)[0] for _ in range(TUNED_RUNS)]
                a.count("tuned_runs", len(runs))
                a.count("tuned_runs_differing_from_first", sum(not _bits(r, runs[0])[0] for r in runs[1:]))
                a.count("tuned_bit_equal_sglang_0_20", sum(_bits(r, s_block)[0] for r in runs))
                for r in runs:
                    a.value("rel_l2_p3b_vs_sglang_tuned", oracle.rel_l2(p3b_out, r))
                    a.value("rel_l2_k7_vs_sglang_tuned", oracle.rel_l2(k7_out, r))
                    a.value("rel_l2_sglang_0_20_vs_sglang_tuned", oracle.rel_l2(s_block, r))
                    a.value("rel_l2_sglang_tuned_vs_fp64", oracle.rel_l2(r.double(), fp64))
                for r in runs[1:]:
                    a.value("rel_l2_sglang_tuned_run_vs_first", oracle.rel_l2(r, runs[0]))
            for op in OPS:
                oracle.report_row(ROW, f"{op} L{layer}", phase="isolate", op=op, layer=layer, m=1,
                                  input=k7._input_label(layer), **acc[op].fields())
                totals[op].merge(acc[op])
            del w, s, ops, P, kept
            AutoTuner.get().clear_cache()
            torch.cuda.empty_cache()
        for op in OPS:
            oracle.report_row(ROW, f"{op} all", phase="isolate", op=op, layers=cfg["num_hidden_layers"], m=1,
                              **totals[op].fields())

    def test_prefill_blocks(self) -> None:
        top_k = oracle.text_config()["num_experts_per_tok"]
        AutoTuner.get().clear_cache()
        for layer in PREFILL_LAYERS:
            w, s = p3b._load(layer)
            ops = k7._cutlass_operands(w, s)
            P = p3b._operands(w, s, top_k)
            for m in PREFILL_M:
                x = p3b._x(layer, m)
                native = _taps(x, P)
                logits = F.linear(x, w["router"])
                sw, sids = k7._sglang_route(x, logits, top_k)
                s_block = _sglang_combine(x, w, p3b._cutlass(x, ops, sids.to(torch.int32), sw), _sglang_shared(x, w))
                k7_out = k7._native_block(x, w, s, top_k)
                fp64 = k7._high_precision_block(x, w, s, sw, sids.long())
                oracle.report_row(ROW, f"prefill L{layer} M{m}", phase="prefill", layer=layer, m=m,
                                  input=k7._input_label(layer),
                                  p3b_differing_elements_vs_sglang_0_20=_bits(native["out"], s_block)[1],
                                  k7_differing_elements_vs_sglang_0_20=_bits(k7_out, s_block)[1],
                                  elements=s_block.numel(),
                                  rel_l2_p3b_vs_sglang_0_20=oracle.rel_l2(native["out"], s_block),
                                  rel_l2_k7_vs_sglang_0_20=oracle.rel_l2(k7_out, s_block),
                                  rel_l2_p3b_vs_k7=oracle.rel_l2(native["out"], k7_out),
                                  rel_l2_p3b_vs_fp64=oracle.rel_l2(native["out"].double(), fp64),
                                  rel_l2_k7_vs_fp64=oracle.rel_l2(k7_out.double(), fp64),
                                  rel_l2_sglang_0_20_vs_fp64=oracle.rel_l2(s_block.double(), fp64))
                del x, native, s_block, k7_out, fp64
            del w, s, ops, P
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
