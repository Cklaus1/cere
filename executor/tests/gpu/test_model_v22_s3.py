"""V22 (test audit 2026-10-18, coverage gap): the shipped default profile on its own artifact, container only.

Suite `model_v22` (`native_dev.py [--gpu 1] test --suite model_v22`, pooled like model_k8; `--suite model` runs it
too). Every other model-level suite loads /model, the NVFP4 checkpoint; the default
(configs/runtime/native-g0-spec.toml: -v22 until the DraftTrain flip, tasks/v29default-spec.md VD-03, -v29 since,
-v22's device set on the S3 variant with self-distilled MTP draft weights and draft token map) serves the S3 FP4-Local
artifact variant of PIN. native_dev.py mounts PIN's `path` read-only at /candidate, named by
quality_measure.CANDIDATE_DIR_ENV (native_dev.ARTIFACT_MODULES; the mount `quality --candidate-checkpoint` uses).
setUpModule refuses unless that directory's manifest.json hashes to PIN's manifest_sha256 (any other mount, /model's
NVFP4 checkpoint included, fails every test), reports row V22 `artifact` (artifact id, manifest SHA-256, families
file, levers, flags), then swaps the oracle's and k8's checkpoint to it for the module (`k8.checkpoint_dir`): its
tensors placed by FAMILIES, whose rules are the artifact's family_rules (MXFP8 dense K2, NVFP4 PLE, NVFP4 W4A16 MTP
experts; tests/test_native_oracles_static.py pins the equality with the S3 inventory).

Every model is created with the -v22 device set V22_DEVICE_SET (the profile's [deployments.native.levers], pinned by
tests/test_native_oracles_static.py): the slot choices k2_gemm small_m, decode_step spec_graph and verify_attention
batched (LEVER_VALUES -> V22_LEVERS); the runtime levers drafter mtp+lookup and draft_policy adaptive are each greedy
ticket's cere_work.flags as cered sets them (FLAG_VALUES -> V22_FLAGS). Drafting models get the artifact's draft token
map (DRAFT_MAP, the map the profile serves). Four models, one per class, each created and released in turn:

* V22Forward: k8.K8Forward on the S3 model at V22_LEVERS: SMOKE, cancel, forced decode and its trace, the
  forward_trace and patch contracts, rows-4 decode and sampling, chunked prefill, the SMOKE memory floor and CR4
  control rows. Not run (set to None, so never collected): `test_forward_trace_equals_merge_base` and
  `test_decode_invariance_equals_merge_base` (L1/L2), whose reference is the merge-base library's run
  (k8.BASE_KTEST_LIB, ABI v3), which predates the S3 tensor formats; model_k8 keeps them on /model.
* V22Speculation: k8.K8Speculation on the drafting S3 model at V22_LEVERS: spec == plain greedy bit for bit with
  caller, MTP, lookup, adaptive and long-lookup drafts, undo, QSA rollback, deep chains, sampled D6 rows; the graph and
  batched-verify contracts; the report-only cycle and window-cost rows. `test_v22_device_set`: the model runs the set
  (decode graphs, spec graphs and batched verify attention on at create, queried, nothing switched) and a greedy
  V22_FLAGS ticket commits plain greedy's tokens and scores with its VERIFY calls run by a graph, batched.
* V22End: test_model_k16_prefix.K16End (K16 (11), the profile's `prefix_end_state`) on the drafting S3 model at
  V22_LEVERS, every ticket with V22_FLAGS.
* V56End: V22End at V56_LEVERS, the opt-in -v56 profile's device set (BeatSglang S7: k2_gemm small_m_a8 and
  prefill_attention p4), so end states captured under P4 resume bit for bit.

No row compares with a CPU reference of the weights: each K8/K16 gate is native-vs-native on one model (cache-on vs
cache-off, speculation vs plain, graph vs eager), and D6's reference (`oracle.d6_ref_row`) samples the model's own
logits. The oracle's references (the NVFP4 checkpoint's BF16 dequantization, families qwen4_exp-nvfp4) are not these
tensors; the S3 formats' references are the kernel suites kernels_k2m, kernels_k13w and kernels_k4n.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import unittest
from pathlib import Path

import torch

import oracle

k8 = importlib.import_module("test_model_k8_forward")
k16 = importlib.import_module("test_model_k16_prefix")
qm = importlib.import_module("quality_measure")

ROW = "V22"
# Repository-relative: the default profile's pin (its `path` is native-g0-spec.toml's artifact_dir) and the families
# file equal to that artifact's manifest family_rules (tests/test_native_oracles_static.py pins both).
PIN_FILE = "configs/artifacts/qwen38-flash-next-fp4local-cere-s3-draft.pin.json"
FAMILIES_FILE = "configs/artifacts/qwen4_exp-nvfp4-plenvfp4-mxfp8.families.json"
PIN = oracle.SRC / PIN_FILE
FAMILIES = oracle.SRC / FAMILIES_FILE
DRAFT_MAP = "draft/token_map.u32le"  # in the artifact: the served draft token map (u32 ids)
# configs/runtime/native-g0-spec.toml [deployments.native.levers], lever -> implementation id.
V22_DEVICE_SET = {"decode_step": "spec_graph", "verify_attention": "batched", "draft_policy": "adaptive",
                  "k2_gemm": "small_m", "drafter": "mtp+lookup"}
CERE_LEVER_DECODE_SPEC_GRAPH = 2  # cere_executor.h: graph + VERIFY / MTP draft steps replay per-shape CUDA graphs
CERE_LEVER_VERIFY_BATCHED = 1  # cere_executor.h: a chain's verify rows' attention cores batched per layer
CERE_LEVER_PREFILL_P4 = 1  # cere_executor.h: prefill attention calls of >= 17 rows (by class) on the P4 kernels
# The model's slot choices (crates/cere-runtime/src/levers.rs, the values cered puts in the desc).
LEVER_VALUES = {("k2_gemm", "small_m"): k8.CERE_LEVER_K2_SMALL_M,
                ("k2_gemm", "small_m_a8"): k8.CERE_LEVER_K2_SMALL_M_A8,
                ("decode_step", "spec_graph"): CERE_LEVER_DECODE_SPEC_GRAPH,
                ("verify_attention", "batched"): CERE_LEVER_VERIFY_BATCHED,
                ("prefill_attention", "p4"): CERE_LEVER_PREFILL_P4}
# The runtime levers: cere_work.flags bits of every greedy ticket that may speculate (native/serve.rs draft_flags).
FLAG_VALUES = {("drafter", "mtp+lookup"): k8.WORK_LOOKUP_DRAFTS,
               ("draft_policy", "adaptive"): k8.WORK_ADAPTIVE_DRAFTS}
V22_LEVERS = {name: LEVER_VALUES[(name, impl)] for name, impl in V22_DEVICE_SET.items()
              if (name, impl) in LEVER_VALUES}
V22_FLAGS = sum(FLAG_VALUES[(name, impl)] for name, impl in V22_DEVICE_SET.items() if (name, impl) not in LEVER_VALUES)
V22_SEED = 20261018
# BeatSglang S7: the opt-in -v56 (configs/runtime/native-g0-spec-v56.toml) is the -v29 device set with k2_gemm
# small_m_a8 and prefill_attention p4 (pinned by tests/test_native_oracles_static.py).
V56_OVERRIDES = {"k2_gemm": "small_m_a8", "prefill_attention": "p4"}
V56_DEVICE_SET = {**V22_DEVICE_SET, **V56_OVERRIDES}
V56_LEVERS = {name: LEVER_VALUES[(name, impl)] for name, impl in V56_DEVICE_SET.items()
              if (name, impl) in LEVER_VALUES}


def setUpModule():
    where = os.environ.get(qm.CANDIDATE_DIR_ENV)
    if where is None:
        raise oracle.OracleError(f"{qm.CANDIDATE_DIR_ENV} is unset (run through native_dev.py test --suite "
                                 "model_v22, which mounts the pinned artifact)")
    pin = json.loads(PIN.read_text())
    manifest = Path(where) / "manifest.json"
    raw = manifest.read_bytes() if manifest.is_file() else b""
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin["manifest_sha256"]:
        raise oracle.OracleError(f"{where} is not the pinned artifact {pin['artifact_id']} (manifest.json SHA-256 "
                                 f"{digest}, pin {pin['manifest_sha256']})")
    if json.loads(raw)["family_rules"] != json.loads(FAMILIES.read_text()):
        raise oracle.OracleError(f"{FAMILIES.name} is not the family_rules of {pin['artifact_id']}")
    if oracle.dev_loop_share() is not None:
        raise oracle.OracleError("the -v22 suite takes no dev-loop share (it covers /model's files)")
    stack = contextlib.ExitStack()
    unittest.addModuleCleanup(stack.close)
    stack.enter_context(k8.checkpoint_dir(Path(where), FAMILIES))
    oracle.report_row(ROW, "artifact", artifact_id=pin["artifact_id"], manifest_sha256=digest,
                      families=FAMILIES.name, work_flags=V22_FLAGS, **V22_LEVERS)


def artifact_draft_map() -> torch.Tensor:
    """The mounted artifact's draft token map (int32 ids; oracle.MODEL_DIR is the artifact inside the module)."""
    with open(oracle.MODEL_DIR / DRAFT_MAP, "rb") as f:
        return torch.frombuffer(bytearray(f.read()), dtype=torch.int32)


def device_set(handle: int) -> tuple:
    """The model's device levers as it runs them, queried (mode -1, nothing switched): (decode graphs on, spec graphs
    on, verify attention batched, VERIFY calls run by a graph, attention_chain calls, MTP drafts run by a graph)."""
    graph = torch.zeros(4, dtype=torch.int64)
    oracle.call("cere_ktest_decode_graph", handle, -1, graph)
    verify = torch.zeros(2, dtype=torch.int64)
    oracle.call("cere_ktest_verify_attention", handle, -1, verify)
    spec = torch.zeros(7, dtype=torch.int64)
    oracle.call("cere_ktest_spec_graph", handle, -1, spec)
    return int(graph[0]), int(spec[0]), int(verify[0]), int(spec[3]), int(verify[1]), int(spec[6])


class V22Forward(k8.K8Forward):
    """K8Forward on the S3 model at the -v22 levers (module docstring)."""

    LEVERS = V22_LEVERS
    MERGE_BASE = False
    test_forward_trace_equals_merge_base = None  # the merge-base library predates the S3 tensor formats
    test_decode_invariance_equals_merge_base = None  # same


class V22Speculation(k8.K8Speculation):
    """K8Speculation on the drafting S3 model at the -v22 levers with the artifact's draft token map."""

    LEVERS = V22_LEVERS

    @classmethod
    def draft_token_map(cls) -> torch.Tensor:
        return artifact_draft_map()

    def test_v22_device_set(self):
        """-v22's device set is what the model runs: decode graphs, spec graphs and batched verify attention on from
        the create; a greedy ticket with -v22's flags (MTP + lookup drafts, adaptive policy) on a repeated-block
        prompt commits plain greedy's tokens and scores bit for bit, verifies drafts, and its VERIFY calls run by a
        graph, batched."""
        start = device_set(self.handle)
        self.assertEqual(start[:3], (1, 1, 1), f"{ROW} device set: (decode graphs, spec graphs, verify batched)")
        gen = torch.Generator().manual_seed(V22_SEED)
        prefix, block, repeats = k8.SPEC_LOOKUP_SHAPES[0]
        prompt = (k8.synthetic_ids(gen, self.vocab, prefix)
                  + k8.synthetic_ids(gen, self.vocab, block) * repeats)
        truth = k8.plain_greedy(self.handle, prompt, k8.SPEC_DECODE - 1)
        want_scores = torch.cat([truth.prefill.unsqueeze(0), truth.scores])
        g, steps, accepted, verified = k8.generate_flags(self.handle, prompt, k8.SPEC_DECODE, V22_FLAGS)
        end = device_set(self.handle)
        self.assertTrue(g.committed >= 1, f"{ROW} v22 ticket: nothing committed")
        self.assertEqual(g.tokens.tolist(), truth.tokens[:g.committed], f"{ROW} v22 ticket: tokens")
        oracle.check_bit_exact(self, ROW, "v22 ticket scores", g.scores, want_scores[:g.committed])
        self.assertEqual(end[:3], start[:3], f"{ROW} v22 ticket: the device set moved")
        self.assertTrue(verified > 0, f"{ROW} v22 ticket: no draft verified")
        self.assertTrue(end[3] > start[3], f"{ROW} v22 ticket: no VERIFY call ran by a graph")
        self.assertTrue(end[4] > start[4], f"{ROW} v22 ticket: no verify chain ran batched")
        oracle.report_row(ROW, "v22 ticket", committed=g.committed, verify_steps=steps, accepted=accepted,
                          verified=verified, graph_verifies=end[3] - start[3], batched_chains=end[4] - start[4],
                          graph_drafts=end[5] - start[5])


class V22End(k16.K16End):
    """K16End (prefix_end_state) on the drafting S3 model at the -v22 levers, every ticket with -v22's flags."""

    LEVERS = V22_LEVERS
    WORK_FLAGS = V22_FLAGS

    @classmethod
    def draft_map_path(cls) -> str:
        return str(oracle.MODEL_DIR / DRAFT_MAP)


class V56End(V22End):
    """K16End on the drafting S3 model at the opt-in -v56 levers (BeatSglang S7): its chunk-0 end states are captured
    under P4 prefill attention and the MXFP8 W8A8 prefill GEMMs, and every hot ticket equals cache-off (the class-pinned
    hot tail runs P4 when its cold call did; it was CERE_ERR_UNSUPPORTED on the K9 path before). The κ mutant resumes
    the wrong class bit for bit: P4 and the MXFP8 GEMMs (one non-split-K algorithm in every bucket) are row-partition
    invariant at both classes, so it equals cache-off (measured on library edfa10c7, `model_v22` 2026-11-01). K16 (13):
    σ is the class at these levers (CudaDevice::row_sig covers the default prefill only), so the chains hit the grid
    after a class change and the σ mutant's resume (another split range at the default levers) equals cache-off too,
    by the same row-partition invariance (both mutants 0 differing; library 06025b77, `model_v22` 2026-11-02T09:38Z)."""

    LEVERS = V56_LEVERS
    KAPPA_MUTANT_DIFFERS = 0
    SIGMA_MUTANT_DIFFERS = 0


if __name__ == "__main__":
    unittest.main()
