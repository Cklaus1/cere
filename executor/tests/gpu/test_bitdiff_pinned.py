"""OH-10 (tasks/ops-hardening-spec.md): bit identity of two libraries on finite inputs, container only.

Suite `bitdiff`, run by `native_dev.py bitdiff --base <executor sha256 | build dir> [--build-dir D]`: the base build's
libcere_ktest.so is mounted read-only and named by `$CERE_BASE_KTEST_LIB` (the L0 convention of
`test_levers_off_identity.py` and `test_model_k8_forward.py`); its absence, or this build's own library as the base,
fails the suite (never a skip). Every case runs the same entry point on identical finite inputs through this build's
library and then through the base's (`oracle.load_ktest` swapped for the call) and reports, per output, the number
of differing elements (`phase` "bitdiff"); every output is compared and reported before the one verdict per test:
the number of outputs that differ, which must be 0 (`oracle.check_count`). `native_dev.py bitdiff` writes the rows
into its report, so a relock records which outputs moved (bench/reports).

The inputs are those of the pinned suites' finite cases (seeded, no NaN/Inf: the non-finite contracts KGM-1/SS-1
changed on purpose, the `fuzz` suite covers them):
* `BitDiffKernels.test_kernels_finite_inputs`: K2 (`cere_ktest_gemm_rows`, the instantiated plan, every planned
  dense GEMM shape of `test_levers_off_identity._k2_shapes` at m in K2_M), K3 (`cere_ktest_hc_mix`, the three
  hyper-connection sites at m in HC_M), the MoE router (`cere_ktest_moe_router`, finite BF16 logits at m in
  ROUTER_M), the sampler (`cere_ktest_sample_probs`, `_draw`, `_top2_f32`, `_top_logprobs` on finite fp32 rows of
  SAMPLER_VOCABS under SAMPLER_CASES).
* `BitDiffModel.test_forward_digests`: the K8 merge-base digests (`test_model_k8_forward.merge_base_digests`: the L1
  `cere_ktest_forward_trace` digests of p12/p13, every row, and the L2 `cere_ktest_forced_decode` decode-invariance
  digests) on a model of each library in turn.
Entry points as stated in `test_levers_off_identity.py`, `test_kernels_k7_moe.py`, `test_kernels_k12_sampler.py` and
`test_model_k8_forward.py`.
"""
from __future__ import annotations

import hashlib
import unittest

import torch

import oracle
from test_kernels_k12_sampler import _native_draw, _native_logprobs, _native_probs, _native_top2
from test_levers_off_identity import (_activations, _base_library, _both, _gemm_rows, _hc_mix, _hc_sites, _hyper,
                                      _k2_shapes, _weight)
from test_model_k8_forward import (MAX_CONTEXT, PREFILL_IDS, create_model, decode_invariance_digests, forward_trace,
                                   merge_base_digests, parity_prompts, release_model, trace_digest, upload_and_seal)

ROW = "BITDIFF"
SEED = 1_010_000
K2_M = (1, 4, 16, 17, 517, 2048)
HC_M = (1, 4, 16, 17, 517)
ROUTER_M = (1, 4, 16, 517)
SAMPLER_VOCABS = (248_320, 4_099)
SAMPLER_CASES = ({"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0},
                 {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "min_p": 0.05},
                 {"temperature": 1.3, "top_k": 1, "top_p": 1.0, "min_p": 0.0})
DRAWS = 64
TOP_LOGPROBS = 8


def differing(current: torch.Tensor, base: torch.Tensor) -> int:
    """Elements whose bytes differ (every element when the dtype or shape differs)."""
    if current.dtype != base.dtype or tuple(current.shape) != tuple(base.shape):
        return max(current.numel(), base.numel(), 1)
    if current.numel() == 0:
        return 0
    a = current.detach().contiguous().view(torch.uint8).reshape(current.numel(), -1)
    b = base.detach().contiguous().view(torch.uint8).reshape(base.numel(), -1)
    return int((a != b).any(dim=-1).sum().item())


class Diff:
    """Every output's differing-element count, reported as it is compared; `check` is the one verdict."""

    def __init__(self, tc: unittest.TestCase, group: str):
        self.tc, self.group = tc, group
        self.outputs = 0
        self.differ: list[str] = []

    def add(self, case: str, current, base) -> None:
        for i, (a, b) in enumerate(zip(current, base, strict=True)):
            name = f"{case}/out{i}"
            bad = differing(a, b)
            oracle.report_row(ROW, name, phase="bitdiff", elements=a.numel(), differing=bad, passed=bad == 0)
            self.outputs += 1
            if bad:
                self.differ.append(name)

    def add_digest(self, case: str, current: str, base: str) -> None:
        oracle.report_row(ROW, case, phase="bitdiff", digest=current, base_digest=base, passed=current == base)
        self.outputs += 1
        if current != base:
            self.differ.append(case)

    def check(self) -> None:
        oracle.report_row(ROW, f"{self.group}/summary", phase="bitdiff_summary", outputs=self.outputs,
                          differing_outputs=len(self.differ), first_differing=self.differ[0] if self.differ else None)
        oracle.check_count(self.tc, ROW, f"{self.group} differing outputs of {self.outputs}", len(self.differ),
                           expected=0)


def library_sha256() -> str:
    with oracle.KTEST_LIB.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def router(logits: torch.Tensor, num_experts: int, top_k: int) -> list[torch.Tensor]:
    m = logits.shape[0]
    ids = oracle.empty((m, top_k), torch.int32)
    weights = oracle.empty((m, top_k), torch.float32)
    oracle.call("cere_ktest_moe_router", logits, ids, weights, m, num_experts, top_k)
    return [ids, weights]


def sampler(processed: torch.Tensor, case: dict, us: torch.Tensor) -> list[torch.Tensor]:
    probs, mask, lse = _native_probs(processed, case)
    drawn = _native_draw(processed, case, us)
    top2_ids, top2_scores = _native_top2(processed)
    committed, top_ids, top_lps = _native_logprobs(processed, float(case["temperature"]), 0, TOP_LOGPROBS)
    return [probs, mask, lse, drawn, top2_ids, top2_scores, committed, top_ids, top_lps]


def current_digests(tc: dict, prompts: list) -> dict:
    """`merge_base_digests` on this build's library: the L1 trace and L2 decode digests on a model of its own."""
    hidden, layers = int(tc["hidden_size"]), int(tc["num_hidden_layers"])
    model = create_model(MAX_CONTEXT)
    try:
        upload_and_seal(model)
        traces = {pid: trace_digest(forward_trace(model.handle, ids, hidden=hidden, layers=layers))
                  for pid, ids, _ in prompts}
        decode = decode_invariance_digests(model.handle, int(tc["vocab_size"]))
    finally:
        release_model(model.handle)
    torch.cuda.synchronize(oracle.DEVICE)
    torch.cuda.empty_cache()
    return {"trace": traces, "decode": decode}


class _BitDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, sha = _base_library()
        oracle.report_row(ROW, "libraries", phase="info", base_library_sha256=sha, library_sha256=library_sha256())


class BitDiffKernels(_BitDiff):
    def test_kernels_finite_inputs(self):
        diff = Diff(self, "kernels")
        tc = oracle.text_config()
        hidden, eps = int(tc["hidden_size"]), float(tc["rms_norm_eps"])
        for s, (label, load, out_f32) in enumerate(_k2_shapes()):
            w = load()
            for m in K2_M:
                x = _activations(m, w.shape[1], SEED + 1000 * s + m)
                current, base = _both(lambda: _gemm_rows(x, w, 0, out_f32)[0])
                diff.add(f"k2/{label}/M{m}", current, base)
        hc, lowrank = int(tc["hc_count"]), int(tc["hc_lowrank"])
        for s, (site, prefix, _) in enumerate(_hc_sites()):
            w = {"norm": _weight(prefix + "hc_norm.weight", dims=1),
                 "down": _weight(prefix + "input_mix_weight_down.weight"),
                 "up": _weight(prefix + "input_mix_weight_up.weight")}
            for m in HC_M:
                hyper = _hyper(m, hc, hidden, SEED + 100_000 + 1000 * s + m)
                current, base = _both(lambda: _hc_mix(hyper, w, hc, hidden, lowrank, eps, 0))
                diff.add(f"k3/{site}/M{m}", current, base)
        num_experts, top_k = int(tc["num_experts"]), int(tc["num_experts_per_tok"])
        for m in ROUTER_M:
            gen = oracle.generator(SEED + 200_000 + m)
            logits = (torch.randn((m, num_experts), generator=gen, device=oracle.DEVICE) * 4.0).to(torch.bfloat16)
            current, base = _both(lambda: router(logits.contiguous(), num_experts, top_k))
            diff.add(f"router/M{m}", current, base)
        for v, vocab in enumerate(SAMPLER_VOCABS):
            for c, case in enumerate(SAMPLER_CASES):
                gen = oracle.generator(SEED + 300_000 + 10 * v + c)
                processed = (torch.randn(vocab, generator=gen, device=oracle.DEVICE) * 3.0).contiguous()
                us = torch.rand(DRAWS, generator=gen, device=oracle.DEVICE, dtype=torch.float64)
                current, base = _both(lambda: sampler(processed, case, us))
                diff.add(f"sampler/V{vocab}/case{c}", current, base)
        diff.check()


class BitDiffModel(_BitDiff):
    def test_forward_digests(self):
        diff = Diff(self, "model")
        tc = oracle.text_config()
        prompts = parity_prompts(PREFILL_IDS)
        current = current_digests(tc, prompts)
        base = merge_base_digests(tc, prompts)
        for pid, _, _ in prompts:
            diff.add_digest(f"model/L1 forward_trace {pid}", current["trace"][pid], base["trace"][pid])
        for case in sorted(current["decode"]):
            diff.add_digest(f"model/L2 {case}", current["decode"][case], base["decode"].get(case, "absent"))
        diff.check()


if __name__ == "__main__":
    unittest.main()
