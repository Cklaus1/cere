"""K8 speculation under the P4 prefill attention lever (S1 of the 2026-10-18 test audit), out of the per-merge
`model_k8`: no configured profile pairs `prefill_attention = "p4"` (`-v12`, configs/runtime/native-g0-p4.toml) with
speculation, so these classes run on demand (`native_dev.py --gpu {gpu} test --suite model_k8p4`, pooled) when a
change touches a path of `native_dev.ON_DEMAND_SUITES["model_k8p4"]`: the P4 kernels and lever, and the shared
speculation code the classes exercise (`native_dev.py triggered` lists the suites a diff selects). GPU0 or GPU1,
container only, as test_model_k8_forward.py.

K8SpeculationP4 and K8SpeculationSmallMP4 (tasks/m23-spec.md D9Q (c)): every K8Speculation test
(test_model_k8_forward.py) on a model created with the `prefill_attention` lever at `p4` (CERE_LEVER_PREFILL_P4),
alone and with `small_m`: a numerics-changing lever set flips to default only if speculation still equals plain
greedy on it bit for bit (`native_dev.py quality-verdict` reads this suite's report for the p4 lever sets).

Token IDs and text stay in memory; report rows carry counts, scalars and digests only.
"""
from __future__ import annotations

import unittest

import test_model_k8_forward as k8

CERE_LEVER_PREFILL_P4 = 1  # the prefill_attention slot: M2.3 P4 tensor-core prefill attention


class K8SpeculationP4(k8.K8Speculation):
    """K8Speculation on a model created with levers.prefill_attention p4 (module docstring)."""

    LEVERS = {"prefill_attention": CERE_LEVER_PREFILL_P4}


class K8SpeculationSmallMP4(k8.K8Speculation):
    """K8Speculation on a model created with both D9C batch levers, small_m and p4 (tasks/m23-spec.md D9Q (c))."""

    LEVERS = {"k2_gemm": k8.CERE_LEVER_K2_SMALL_M, "prefill_attention": CERE_LEVER_PREFILL_P4}


if __name__ == "__main__":
    unittest.main()
