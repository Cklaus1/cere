"""Pin the native executor build in the native deployment (tasks/m22-spec.md R6, Q15).

    python3 scripts/native/lock_native.py [--build-json build/native/build.json] [--root REPO]
                                          [--check [--stale SHA ...]]

Reads the `build.json` that `scripts/live/native_dev.py build` writes and sets the SHA-256 of its
`libcere_executor.so` as `backend_build_sha256` in `configs/deployments/qwen38-flash-next-native.lock.json` and as
`[deployments.native].library_sha256` in `configs/runtime/native-g0.toml` (the two must be equal for the config to load),
and the same pair of the dev/live snapshot deployment (`configs/deployments/qwen38-flash-next-native-snapshot.lock.json`
and `.toml`, tasks/m211-spec.md D5/LR5), of the vision profile (`configs/deployments/qwen38-flash-next-native-vision
.lock.json` and `configs/runtime/native-g0-vision.toml`, tasks/m213-spec.md D9, Q2 b), of the GPU1 lane
(`configs/deployments/qwen38-flash-next-native-g1.lock.json` and `configs/runtime/native-g1.toml`, docs/RUNBOOK.md
"GPU1 native lane") and of the opt-in prefix-cache profiles (`configs/deployments/qwen38-flash-next-native-prefix
.lock.json` and `configs/runtime/native-g0-prefix.toml`, `configs/deployments/qwen38-flash-next-native-g1-prefix
.lock.json` and `configs/runtime/native-g1-prefix.toml`, tasks/m210-spec.md D9C), so the profiles never pin different
libraries. Since the perf integration (build/perf-int), the
M2.3 P4 `-v12` (`configs/deployments/qwen38-flash-next-native-p4.lock.json`, `configs/runtime/native-g0-p4.toml`)
the P11 K2 small-M `-v14` (`configs/deployments/qwen38-flash-next-native-k2s.lock.json`,
`configs/runtime/native-g0-k2s.toml`) and its speculation sibling `-v16`
(`configs/deployments/qwen38-flash-next-native-spec-k2s.lock.json`, `configs/runtime/native-g0-spec-k2s.toml`),
the FP8 dense-weights `-v20` (`configs/deployments/qwen38-flash-next-native-spec-fp8w.lock.json`,
`configs/runtime/native-g0-spec-fp8w.toml`), the FP4-Local S2 dev `-v19-fp4local`
(`configs/deployments/qwen38-flash-next-native-fp4local.lock.json`, `configs/runtime/native-g0-spec-fp4local.toml`,
tasks/fp4local-spec.md FL-13), the FP4-Local S3 `-v21`
(`configs/deployments/qwen38-flash-next-native-fp4local-s3.lock.json`, `configs/runtime/native-g0-spec-fp4local-s3.toml`,
FL-30), the NVFP4 speculation `-v19` (the default before the FP4-Local flip, FL-31;
`configs/deployments/qwen38-flash-next-native-spec-nvfp4.lock.json`, `configs/runtime/native-g0-spec-nvfp4.toml`), the
S3 headroom `-v22` (the default from FL-31 to the DraftTrain flip, tasks/v29default-spec.md VD-03;
`configs/deployments/qwen38-flash-next-native-spec-v22.lock.json`, `configs/runtime/native-g0-spec-v22.toml`; the
default `configs/runtime/native-g0-spec.toml` is DraftTrain's `-v29`, tasks/drafttrain-spec.md D7), the TreeSpec
opt-in `-v28` (`configs/deployments/qwen38-flash-next-native-spec-tree.lock.json`,
`configs/runtime/native-g0-spec-tree.toml`, tasks/tree-spec-spec.md TS-03), the Chains opt-in `-v36`
(`configs/deployments/qwen38-flash-next-native-spec-chains.lock.json`, `configs/runtime/native-g0-spec-chains.toml`,
tasks/chains-spec.md), the Sessions opt-in `-v45`
(`configs/deployments/qwen38-flash-next-native-sessions.lock.json`, `configs/runtime/native-g0-sessions.toml`,
tasks/sessions-spec.md: `-v29` plus the M2.10 host tier), the VisionLong opt-in `-v43`
(`configs/deployments/qwen38-flash-next-native-spec-vision.lock.json`, `configs/runtime/native-g0-spec-vision.toml`,
tasks/m213l-spec.md) and its dev-only diagnostic `-v54`
(`configs/deployments/qwen38-flash-next-native-spec-vision-v54.lock.json`,
`configs/runtime/native-g0-spec-vision-v54.toml`, Amendment A2), the Ctx262 diagnostic `-v55`
(`configs/deployments/qwen38-flash-next-native-spec-vision-v55.lock.json`,
`configs/runtime/native-g0-spec-vision-v55.toml`, tasks/ctx262-spec.md CX-R18), the PrefixHit opt-in `-v51`
(`configs/deployments/qwen38-flash-next-native-spec-v51.lock.json`, `configs/runtime/native-g0-spec-v51.toml`,
tasks/m210-spec.md D17), the Ctx262 opt-in `-v44` (`configs/deployments/qwen38-flash-next-native-spec-262k.lock.json`,
`configs/runtime/native-g0-spec-262k.toml`, tasks/ctx262-spec.md CX-02), the Ctx262 x PrefixHit `-v52`
(`configs/deployments/qwen38-flash-next-native-spec-v52.lock.json`, `configs/runtime/native-g0-spec-v52.toml`,
tasks/ctx262-spec.md CX-09), the Sessions composition `-v53`
(`configs/deployments/qwen38-flash-next-native-sessions-v53.lock.json`, `configs/runtime/native-g0-sessions-v53.toml`,
tasks/sessions-spec.md SE-11: `-v52` plus the M2.10 host tier), the MixedPD opt-in `-v47`
(`configs/deployments/qwen38-flash-next-native-spec-v47.lock.json`, `configs/runtime/native-g0-spec-v47.toml`,
tasks/mixed-pd-spec.md), the GrammarSpec opt-in `-v49`
(`configs/deployments/qwen38-flash-next-native-spec-grammar.lock.json`, `configs/runtime/native-g0-spec-grammar.toml`,
tasks/grammar-spec-spec.md), the Track LoRA opt-in `-v50`
(`configs/deployments/qwen38-flash-next-native-spec-lora.lock.json`, `configs/runtime/native-g0-spec-lora.toml`), the
WeightLab research `-v57` (`configs/deployments/qwen38-flash-next-native-spec-research.lock.json`,
`configs/runtime/native-g0-spec-research.toml`, tasks/weightlab-spec.md WL-01), the
BeatSglang S7 opt-in `-v56` (`configs/deployments/qwen38-flash-next-native-spec-v56.lock.json`,
`configs/runtime/native-g0-spec-v56.toml`, tasks/beat-sglang-spec.md S7), the BatchConc BC2 dev profile `-v58`
(`configs/deployments/qwen38-flash-next-native-rows16-v58.lock.json`, `configs/runtime/native-g0-rows16-v58.toml`,
tasks/batch-concurrency-spec.md BC2: the `-v53` composition at 16 rows), the BatchConcurrency BC4 opt-in `-v59`
(`configs/deployments/qwen38-flash-next-native-sessions-v59.lock.json`, `configs/runtime/native-g0-sessions-v59.toml`,
tasks/batch-concurrency-spec.md BC4: `-v53` plus prefill_slice_ms 50), the M3.2N two-card deployment (`configs/deployments/qwen38-flash-next-native-2gpu.lock.json`, `configs/runtime/native-2gpu.toml`) and the FP4-Local
S2 eval configs (`configs/runtime/native-g0-spec-eval.toml` with the `-v19` lock, `native-g0-spec-fp4local-eval.toml`
with the `-v19-fp4local` one; FL-14) and the S3 eval config (`native-g0-spec-fp4local-s3-eval.toml` with the `-v21`
lock; the tool-eval-bench pair of the default flip) and the production T4 config (`configs/runtime/prod-t4.toml`
with the `-v58` 16-row lock since production window 11, tasks/batch-concurrency-spec.md BC7) profiles pin
the same library too: every lever variant is compiled into it and each profile selects its kernels with
`[deployments.native.levers]` (crates/cere-runtime/src/levers.rs). The build's
MoE backend (build.json `flags.moe_backend`, absent `k7`) selects the pairs: the shipped `CERE_MOE_BACKEND=p3b` build
(tasks/m23-spec.md M23-D7) pins all of them. A K7 build (`--moe-backend k7`, the reference backend, M23-D7) is
accepted, but no profile ships it, so it pins no pair (exit 0, nothing written) and its `--check` gates only the
library, the `--stale` SHAs and the executor tree. A pre-lever build (build.json `flags.prefill_attention` or
`flags.k2_small_m`, the retired CMake selections) and any unknown value are refused. A library built for another
model (build.json `flags.model`, SelfTune ST4) pins only that model's pairs: a DICE-8B build (`flags.model` dice-8b)
the `configs/deployments/dice-8b-native.lock.json` / `configs/runtime/native-g0-dice8b.toml` pair (ST4 S3d), whatever
its backend, and never the Qwen ones.
Every other byte of the pinned files is kept, and the result is validated before anything is written. A build from a
dirty tree (`git_dirty: true`) is refused: its library cannot be rebuilt from a commit. So is a build whose compiles
went through ccache (`ccache` not `off`, `native_dev.py build --ccache`, build/ccache): a relock is a clean, uncached
build. So is a fault-injection build (OH-08, `native_dev.py build --fault-injection`: build.json `flags.cxx` or
`flags.cuda` defines CERE_FAULT_INJECTION), and `--check` also refuses a library file that holds the fault-injection
trigger's strings (`CERE_FAULT_INJECT`). So is a fail-first sdar mutant (SelfTune ST4 S4a, `native_dev.py build
--sdar-mutant NAME`: build.json `mutant`, or flags defining CERE_SDAR_MUTANT_<NAME>), by name. A Qwen relock that
pins a library other than the last recorded one appends the build's digests to `bench/reports/native-locks.jsonl`
(OH-10: `executor_sha256`, `ktest_sha256`, `git_head`, `moe_backend` and the `replaced_executor_sha256` the shipped
lock pinned before), so the previous locked libraries
stay known to `native_dev.py bitdiff --base <digest>` after the relock; another model's relock replaces no shipped
library and records nothing there. Exit 0 when pinned (or, for an unshipped
variant, accepted), 2 on refusal.

`--check` is the relock finish gate (tasks/m22c-spec.md D6, R1; outside `make ci`, which has no build): it writes
nothing and exits 0 only if every lock and config pin the clean `build.json`'s library, that SHA is the one of
the `libcere_executor.so` next to `build.json`, the build's `git_head` is a commit of `--root` whose build inputs under
`executor/` equal the checkout's (no later, uncommitted, deleted or untracked change to a file `executor/CMakeLists.txt`
builds the libraries from, `build_inputs`, O52; GPU oracles, docs and host tests are not inputs), and the library is
none of the `--stale` SHAs (the milestone's previous build, e.g. M2.2c passes the `-v1` library). Else exit 2. Run it
again after the final build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from cere_contracts import ContractError, validate  # noqa: E402

LOCK = "configs/deployments/qwen38-flash-next-native.lock.json"
CONFIG = "configs/runtime/native-g0.toml"
# (lock, runtime config) pairs pinned together: the shipped profile, the dev/live snapshot profile (M2.11 LR5), the
# vision profile (M2.13 D9, Q2 b), the GPU1 lane's profile (docs/RUNBOOK.md "GPU1 native lane"), the default
# speculation profile (`-v29` on the S3 draft-variant artifact since the DraftTrain flip, v29default design note
# VD-03), the S3 headroom `-v22` (the default from the FP4-Local flip, fp4local design note FL-31, to the DraftTrain
# flip), the NVFP4 speculation profile (m29 design note D12, `-v19`, the default before FL-31; [REVISED: M29S2-06d]
# on P3b, row-invariant per M23-D8), the
# opt-in prefix-cache profiles of both lanes (M2.10 D9C), the M2.9L lookup profile (m29l design note D7, `-v15`),
# the M2.3 P4 profile (`-v12`, levers.prefill_attention p4), the P11 K2 small-M opt-in profile (P11R-07, `-v14`,
# levers.k2_gemm small_m), its speculation sibling (`-v16`), the FP8 dense-weights profile (Fp8W, `-v20`,
# levers.k2_weights fp8), the M3.2N two-card deployment (m32n design note D1, one replica per GPU) and the
# FP4-Local S2 eval configs (fp4local design note FL-14: `-v19` and `-v19-fp4local` with the eval [api] limits,
# each with its base profile's lock) and the S3 one (`-v21` with the eval [api] limits, the default flip's
# tool-eval-bench pair), so no two profiles ever pin different libraries.
PAIRS = ((LOCK, CONFIG),
         ("configs/deployments/qwen38-flash-next-native-snapshot.lock.json",
          "configs/deployments/qwen38-flash-next-native-snapshot.toml"),
         ("configs/deployments/qwen38-flash-next-native-vision.lock.json", "configs/runtime/native-g0-vision.toml"),
         ("configs/deployments/qwen38-flash-next-native-g1.lock.json", "configs/runtime/native-g1.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec.lock.json", "configs/runtime/native-g0-spec.toml"),
         ("configs/deployments/qwen38-flash-next-native-rows16-v58.lock.json", "configs/runtime/prod-t4.toml"),
         ("configs/deployments/qwen38-flash-next-native-prefix.lock.json", "configs/runtime/native-g0-prefix.toml"),
         ("configs/deployments/qwen38-flash-next-native-g1-prefix.lock.json", "configs/runtime/native-g1-prefix.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-lookup.lock.json",
          "configs/runtime/native-g0-spec-lookup.toml"),
         ("configs/deployments/qwen38-flash-next-native-p4.lock.json", "configs/runtime/native-g0-p4.toml"),
         ("configs/deployments/qwen38-flash-next-native-k2s.lock.json", "configs/runtime/native-g0-k2s.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-k2s.lock.json", "configs/runtime/native-g0-spec-k2s.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-fp8w.lock.json",
          "configs/runtime/native-g0-spec-fp8w.toml"),
         ("configs/deployments/qwen38-flash-next-native-fp4local.lock.json",
          "configs/runtime/native-g0-spec-fp4local.toml"),
         ("configs/deployments/qwen38-flash-next-native-fp4local-s3.lock.json",
          "configs/runtime/native-g0-spec-fp4local-s3.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-tree.lock.json",
          "configs/runtime/native-g0-spec-tree.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-v22.lock.json",
          "configs/runtime/native-g0-spec-v22.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-chains.lock.json",
          "configs/runtime/native-g0-spec-chains.toml"),
         ("configs/deployments/qwen38-flash-next-native-sessions.lock.json",
          "configs/runtime/native-g0-sessions.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-vision.lock.json",
          "configs/runtime/native-g0-spec-vision.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-vision-v54.lock.json",
          "configs/runtime/native-g0-spec-vision-v54.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-vision-v55.lock.json",
          "configs/runtime/native-g0-spec-vision-v55.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-262k.lock.json",
          "configs/runtime/native-g0-spec-262k.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-v51.lock.json",
          "configs/runtime/native-g0-spec-v51.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-v47.lock.json",
          "configs/runtime/native-g0-spec-v47.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-v52.lock.json",
          "configs/runtime/native-g0-spec-v52.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-grammar.lock.json",
          "configs/runtime/native-g0-spec-grammar.toml"),
         ("configs/deployments/qwen38-flash-next-native-sessions-v53.lock.json",
          "configs/runtime/native-g0-sessions-v53.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-lora.lock.json",
          "configs/runtime/native-g0-spec-lora.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-research.lock.json",
          "configs/runtime/native-g0-spec-research.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-v56.lock.json",
          "configs/runtime/native-g0-spec-v56.toml"),
         ("configs/deployments/qwen38-flash-next-native-rows16-v58.lock.json",
          "configs/runtime/native-g0-rows16-v58.toml"),
         ("configs/deployments/qwen38-flash-next-native-sessions-v59.lock.json",
          "configs/runtime/native-g0-sessions-v59.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-nvfp4.lock.json",
          "configs/runtime/native-g0-spec-nvfp4.toml"),
         ("configs/deployments/qwen38-flash-next-native-2gpu.lock.json", "configs/runtime/native-2gpu.toml"),
         ("configs/deployments/qwen38-flash-next-native-spec-nvfp4.lock.json",
          "configs/runtime/native-g0-spec-eval.toml"),
         ("configs/deployments/qwen38-flash-next-native-fp4local.lock.json",
          "configs/runtime/native-g0-spec-fp4local-eval.toml"),
         ("configs/deployments/qwen38-flash-next-native-fp4local-s3.lock.json",
          "configs/runtime/native-g0-spec-fp4local-s3-eval.toml"))
# M23-D1 / M23-D7: the build's MoE backend (`flags.moe_backend`, absent `k7`) selects the pairs that pin it: the P3b
# build (the shipped default since M23-D7) PAIRS; the K7 reference build is a supported build no profile ships.
VARIANT_PAIRS = {"p3b": PAIRS, "k7": ()}
# SelfTune ST4 S3a: a library is built for one model (executor/CMakeLists.txt CERE_MODEL); build.json `flags.model`
# names it, absent for the Qwen descriptor (whose pairs the MoE backend selects). Another model's build pins only its
# own pairs (DICE-8B's GPU0 dev deployment, ST4 S3d); an unknown model is refused.
DEFAULT_MODEL = "qwen38-flash-next"
# SdarOptIn (O320): the opt-in `sdar_gemm = "changing"` twin shares the lock and pins the same library.
MODEL_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "dice-8b": (("configs/deployments/dice-8b-native.lock.json", "configs/runtime/native-g0-dice8b.toml"),
                ("configs/deployments/dice-8b-native.lock.json", "configs/runtime/native-g0-dice8b-gemm-changing.toml")),
    # Inkling S13: the opt-in G7 twin (default output 4,096, its own ledger) shares S5's lock and library with S5's
    # lane, drafts off since D-I-S9B (native-g0-inkling-plain.toml); the default lane (MTP drafter on,
    # inkling design note D-I-S9B) and D-I-C32's opt-in batch twin (`-v69`, 32 rows) each have their own deployment
    # and lock on the same library.
    "inkling-2-mini": (("configs/deployments/inkling-2-mini-native.lock.json",
                        "configs/runtime/native-g0-inkling-plain.toml"),
                       ("configs/deployments/inkling-2-mini-native.lock.json",
                        "configs/runtime/native-g0-inkling-s13.toml"),
                       ("configs/deployments/inkling-2-mini-native-mtp.lock.json",
                        "configs/runtime/native-g0-inkling.toml"),
                       ("configs/deployments/inkling-2-mini-native-batch.lock.json",
                        "configs/runtime/native-g0-inkling-batch.toml"))}
# The retired CMake kernel selections (CERE_PREFILL_ATTENTION, CERE_K2_SMALL_M); a build.json naming one is a
# pre-lever library that selects kernels at build time, which no profile pins.
RETIRED_FLAGS = ("prefill_attention", "k2_small_m")
LIBRARY = "libcere_executor.so"
KTEST_LIBRARY = "libcere_ktest.so"
SHA256 = re.compile(r"[0-9a-f]{64}")
CONFIG_LINE = re.compile(r'^library_sha256 = "[0-9a-f]{64}"$', re.MULTILINE)
# OH-08: the define a fault-injection build adds to both flag strings (executor/CMakeLists.txt) and the bytes its
# trigger (executor/cuda/src/fault/fault_inject.cu) puts in the library; neither is in any other build.
FAULT_INJECTION_FLAG = "-DCERE_FAULT_INJECTION"
FAULT_INJECTION_BYTES = b"CERE_FAULT_INJECT"
# SelfTune ST4 S4a: the define a `native_dev.py build --sdar-mutant NAME` library adds to both flag strings
# (executor/CMakeLists.txt CERE_SDAR_MUTANT); no other build carries it.
SDAR_MUTANT_FLAG = re.compile(r"-DCERE_SDAR_MUTANT_([A-Z0-9_]+)")
# OH-10: the relocks' digests, one JSON object per line, oldest first.
LOCK_HISTORY = "bench/reports/native-locks.jsonl"


def history_record(root: Path, build: dict, sha: str) -> str | None:
    """The LOCK_HISTORY line of a relock to `build` (library `sha`), or None when the last recorded relock pinned
    `sha` already. Read before the shipped lock is written: its pin is the replaced library."""
    path = root / LOCK_HISTORY
    try:
        lines = path.read_text().splitlines() if path.exists() else []
        last = json.loads(lines[-1])["executor_sha256"] if lines else None
        replaced = json.loads((root / LOCK).read_text())["backend_build_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused(f"cannot read {LOCK_HISTORY} or {LOCK}: {type(e).__name__}: {e}") from None
    if last == sha:
        return None
    ktest = [o.get("sha256") for o in build.get("outputs", []) if isinstance(o, dict)
             and o.get("file") == KTEST_LIBRARY]
    if len(ktest) != 1 or not isinstance(ktest[0], str) or not SHA256.fullmatch(ktest[0]):
        raise Refused(f"build.json has no single {KTEST_LIBRARY} output with a lowercase hex SHA-256 (OH-10 keeps it "
                      f"in {LOCK_HISTORY})")
    return json.dumps({"executor_sha256": sha, "ktest_sha256": ktest[0], "git_head": build.get("git_head"),
                       "moe_backend": build["flags"]["moe_backend"], "replaced_executor_sha256": replaced},
                      sort_keys=True) + "\n"


class Refused(Exception):
    """The build or the deployment files cannot be pinned; nothing was written."""


def library_sha256(build: dict) -> str:
    if build.get("schema_version") != "1":
        raise Refused("build.json schema_version is not \"1\"")
    if build.get("git_dirty") is not False:
        raise Refused("build.json is from a dirty tree (git_dirty is not false); rebuild from a commit")
    if build.get("ccache") != "off":
        raise Refused(f"build.json records ccache {build.get('ccache')!r}, not \"off\": a relock is a clean build "
                      "(native_dev.py build without --ccache)")
    found = [o.get("sha256") for o in build.get("outputs", []) if isinstance(o, dict) and o.get("file") == LIBRARY]
    if len(found) != 1 or not isinstance(found[0], str) or not SHA256.fullmatch(found[0]):
        raise Refused(f"build.json has no single {LIBRARY} output with a lowercase hex SHA-256")
    return found[0]


def variant_pairs(build: dict) -> tuple[tuple[str, str], ...]:
    """The (lock, config) pairs that pin the build's model (`flags.model`, absent the Qwen descriptor) and, for the
    Qwen descriptor, its MoE backend (`flags.moe_backend`, absent `k7`)."""
    flags = build.get("flags", {})
    if not isinstance(flags, dict):
        raise Refused(f"build.json flags {flags!r} is not an object")
    retired = [f for f in RETIRED_FLAGS if f in flags]
    if retired:
        raise Refused(f"build.json flags {', '.join(retired)} name a pre-lever build (kernels selected by CMake); "
                      "rebuild: the library compiles every lever and profiles select them in "
                      "[deployments.native.levers]")
    injected = [k for k in ("cxx", "cuda") if FAULT_INJECTION_FLAG in str(flags.get(k, ""))]
    if injected:
        raise Refused(f"build.json flags.{'/'.join(injected)} define CERE_FAULT_INJECTION: a test-only fault-injection "
                      "build (OH-08) is never pinned; rebuild without --fault-injection")
    if "mutant" in build:
        raise Refused(f"build.json names the sdar mutant {build['mutant']} (`mutant`): a test-only fail-first build "
                      "(SelfTune ST4 S4a) is never pinned; rebuild without --sdar-mutant")
    mutant = sorted({m.group(1).lower() for k in ("cxx", "cuda")
                     for m in SDAR_MUTANT_FLAG.finditer(str(flags.get(k, "")))})
    if mutant:
        raise Refused(f"build.json flags define the sdar mutant {', '.join(mutant)} (CERE_SDAR_MUTANT_*): a test-only "
                      "fail-first build (SelfTune ST4 S4a) is never pinned; rebuild without --sdar-mutant")
    model = flags.get("model", DEFAULT_MODEL)
    if model != DEFAULT_MODEL:
        pairs = MODEL_PAIRS.get(model) if isinstance(model, str) else None
        if pairs is None:
            raise Refused(f"build.json flags.model {model!r} is none of {sorted([DEFAULT_MODEL, *MODEL_PAIRS])}")
        return pairs
    backend = flags.get("moe_backend", "k7")
    pairs = VARIANT_PAIRS.get(backend) if isinstance(backend, str) else None
    if pairs is None:
        raise Refused(f"build.json flags.moe_backend {backend!r} is none of {sorted(VARIANT_PAIRS)}")
    return pairs


def pin(root: Path, sha: str, lock_rel: str, config_rel: str) -> tuple[str, str]:
    """The new lock and config texts of one pair, validated together."""
    lock = json.loads((root / lock_rel).read_text())
    lock["backend_build_sha256"] = sha
    lock_text = json.dumps(lock, indent=2, ensure_ascii=False) + "\n"
    config_text = (root / config_rel).read_text()
    config_text, count = CONFIG_LINE.subn(f'library_sha256 = "{sha}"', config_text)
    if count != 1:
        raise Refused(f"{config_rel} needs exactly one `library_sha256 = \"<hex>\"` line, found {count}")
    validate("deployment_lock", lock, root)
    config = tomllib.loads(config_text)
    entries = config["deployments"]
    if len(entries) != 1 or entries[0]["deployment"] + ".lock.json" != lock_rel:
        raise Refused(f"{config_rel} does not use {lock_rel} in its single [[deployments]] entry")
    return lock_text, config_text


def check(root: Path, build_json: Path, build: dict, sha: str, stale: list[str],
          pairs: tuple[tuple[str, str], ...]) -> None:
    """Refused unless every committed lock and config of `pairs` pin `sha`, `sha` is the library file's and not stale,
    and the build's commit has the checkout's `executor/`."""
    if sha in stale:
        raise Refused(f"build.json's {LIBRARY} {sha} is stale (a --stale build); rebuild from the milestone's commit")
    library = build_json.parent / LIBRARY
    try:
        data = library.read_bytes()
    except OSError as e:
        raise Refused(f"cannot read {library}: {e}") from None
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha:
        raise Refused(f"{library} has SHA-256 {actual}, not build.json's {sha}; rebuild")
    if FAULT_INJECTION_BYTES in data:
        raise Refused(f"{library} holds {FAULT_INJECTION_BYTES.decode()}: a fault-injection library (OH-08) is never "
                      "pinned")
    check_executor_tree(root, build.get("git_head"))
    for lock_rel, config_rel in pairs:
        check_pair(root, sha, lock_rel, config_rel)


def check_pair(root: Path, sha: str, lock_rel: str, config_rel: str) -> None:
    try:
        lock = json.loads((root / lock_rel).read_text())["backend_build_sha256"]
        config = tomllib.loads((root / config_rel).read_text())["deployments"][0]["native"]["library_sha256"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        raise Refused(f"cannot read the pins of {lock_rel} and {config_rel}: {type(e).__name__}: {e}") from None
    if (lock, config) != (sha, sha):
        raise Refused(f"not relocked: {lock_rel} backend_build_sha256 {lock} and {config_rel} library_sha256 {config} "
                      f"must both equal build.json's {LIBRARY} {sha}")


def check_executor_tree(root: Path, head: object) -> None:
    """Refused unless `head` is a commit of `root` and every build input of the pinned libraries under `executor/`
    (`build_inputs` of the build's commit and of the checkout, tracked or not) equals that commit's."""
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise Refused(f"build.json git_head {head!r} is not a commit id")
    if git(root, "cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        raise Refused(f"build.json git_head {head} is not a commit of {root}")
    diff = git(root, "diff", "--no-renames", "--name-only", "-z", head, "--", EXECUTOR)
    untracked = git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", EXECUTOR)
    for name, proc in (("diff", diff), ("ls-files", untracked)):
        if proc.returncode != 0:
            raise Refused(f"git {name} {head} -- {EXECUTOR} failed: {proc.stderr.decode(errors='replace').strip()}")
    changed = sorted({path(p) for p in (diff.stdout + untracked.stdout).split(b"\0") if p})
    if not changed:
        return
    try:
        inputs = [build_inputs(git_tree(root, head)), build_inputs(checkout_tree(root))]
        stale, scope = [p for p in changed if any(is_input(p, i) for i in inputs)], "build inputs"
    except Underivable as e:
        stale, scope = changed, f"files (all are build inputs: {e})"
    if stale:
        shown = ", ".join(EXECUTOR + p for p in stale[:10]) + (f" and {len(stale) - 10} more" if len(stale) > 10 else "")
        raise Refused(f"{EXECUTOR} {scope} differ from the build's commit {head} (changed, deleted or untracked): "
                      f"{shown}; rebuild from the checkout")


# Build inputs (opportunities design note O52). Both pinned libraries are built by `cmake -S executor` from
# `executor/CMakeLists.txt` alone, so the files they can read are derived from that build definition, never from a
# list of exclusions:
#   1. every path the definition spells under executor/ (a source, a glob's literal directory, an include directory,
#      a configure_file or LINK_DEPENDS input), as `${CMAKE_CURRENT_SOURCE_DIR}/...` after expanding its one-value
#      `set()` variables, or relative to executor/: that file, or every file under that directory, present or future;
#   2. closure: the compiler reads a file only through a path ending in the file's name (#include, #embed,
#      __has_include, .incbin; a computed `#include MACRO` through the macro's #define), so every file whose name a
#      preprocessor line or .incbin of an input spells is an input too, wherever it is (`../../tests/host/check.h`).
# The `if(CMAKE_SCRIPT_MODE_FILE)` block is skipped: it runs only under `cmake -P` (the build.json writer, run after both
# libraries are linked; its tree walk feeds `source_tree_sha256`, which build.json records and no library contains).
# Whatever could let the build read a file without spelling it makes the derivation Underivable, and every executor/
# file an input again (the pre-O52 check): no build definition, or one spread over other CMake files or bracket
# arguments, or an `else` in the script-mode block; a use of executor/ itself other than the two that read nothing
# into a library; a path ascending out of an unexpanded variable or out of executor/; a symlink or submodule; a path
# holding a delimiter character; a computed include whose macro no input #defines without `##`. (Recording in
# build.json the files the compiler read would miss a new file that a glob or an include directory picks up, and helps
# only after every pinned library is rebuilt.)
EXECUTOR = "executor/"
BUILD_DEFINITION = "CMakeLists.txt"
ROOT_VARS = ("CMAKE_CURRENT_SOURCE_DIR", "CMAKE_CURRENT_LIST_DIR", "CMAKE_SOURCE_DIR", "PROJECT_SOURCE_DIR",
             "CMAKE_HOME_DIRECTORY", "cere_executor_SOURCE_DIR")
# "\0" stands for executor/ itself. `-ffile-prefix-map=<executor>=` rewrites recorded paths and reads nothing;
# `-DCERE_SOURCE_DIR=<executor>` hands the tree to the build.json writer.
NON_READING_ROOT_USES = ("-ffile-prefix-map=\0=", "-DCERE_SOURCE_DIR=\0")
DELIMITERS = "\\s\"'();,=:\0"
CMAKE_SET = re.compile(r"\bset\(\s*(\w+)\s+(\"?)([^\s\"()]+)\2\s*\)")
CMAKE_PATH = re.compile(f"\0(/[^{DELIMITERS}]*)?")
CMAKE_SPREAD = re.compile(r"\[=*\[|\b(?:include|add_subdirectory)\s*\(")
CMAKE_IF = re.compile(r"\b(if|endif|else|elseif)\s*\(")
SCRIPT_MODE_IF = re.compile(r"\bif\s*\(\s*CMAKE_SCRIPT_MODE_FILE\s*\)")
READ_LINE = re.compile(r"^[ \t]*#.*$|^.*\bincbin\b.*$", re.MULTILINE)
COMPUTED_INCLUDE = re.compile(r"^[ \t]*#[ \t]*include[ \t]+([A-Za-z_]\w*)", re.MULTILINE)


class Underivable(Exception):
    """The build inputs cannot be bounded below the whole of executor/."""


class Tree:
    """The files under executor/ of a commit or of the checkout, by path relative to executor/, and the paths that are
    neither a file nor a directory of files (symlinks, submodules)."""

    def __init__(self, files: dict[str, bytes], links: list[str]):
        self.files, self.links = files, links


def git(root: Path, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True)


def path(raw: bytes) -> str:
    """A `git -z` path relative to executor/."""
    return raw.decode("utf-8", "surrogateescape").removeprefix(EXECUTOR)


def git_tree(root: Path, rev: str) -> Tree:
    listing = git(root, "ls-tree", "-r", "-z", rev, "--", EXECUTOR)
    if listing.returncode != 0:
        raise Refused(f"git ls-tree {rev} -- {EXECUTOR} failed: {listing.stderr.decode(errors='replace').strip()}")
    blobs, links = {}, []
    for entry in filter(None, listing.stdout.split(b"\0")):
        meta, raw = entry.split(b"\t", 1)
        mode, kind, oid = meta.split()
        if kind == b"blob" and mode != b"120000":
            blobs[path(raw)] = oid
        else:
            links.append(path(raw))
    batch = git(root, "cat-file", "--batch", stdin=b"".join(oid + b"\n" for oid in blobs.values()))
    if batch.returncode != 0:
        raise Refused(f"git cat-file --batch of {rev} failed: {batch.stderr.decode(errors='replace').strip()}")
    files, out, at = {}, batch.stdout, 0
    for rel in blobs:
        start = out.index(b"\n", at) + 1
        end = start + int(out[at:start].split()[2])
        files[rel], at = out[start:end], end + 1
    return Tree(files, links)


def checkout_tree(root: Path) -> Tree:
    listing = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", EXECUTOR)
    if listing.returncode != 0:
        raise Refused(f"git ls-files -- {EXECUTOR} failed: {listing.stderr.decode(errors='replace').strip()}")
    files, links = {}, []
    for raw in sorted(set(filter(None, listing.stdout.split(b"\0")))):
        full = root / raw.decode("utf-8", "surrogateescape")
        if full.is_symlink() or full.is_dir():
            links.append(path(raw))
        elif full.exists():
            files[path(raw)] = full.read_bytes()
    return Tree(files, links)


def is_input(rel: str, inputs: frozenset[str]) -> bool:
    """`rel` (relative to executor/) is an input file or lies under an input directory."""
    return rel in inputs or any(rel[:i] in inputs for i, c in enumerate(rel) if c == "/")


def build_inputs(tree: Tree) -> frozenset[str]:
    """The paths relative to executor/ the pinned libraries can read (a directory: every file under it)."""
    if tree.links:
        raise Underivable(f"symlink or submodule {EXECUTOR}{tree.links[0]}")
    bad = next((p for p in tree.files if re.search(f"[{DELIMITERS}]", p)), None)
    if bad is not None:
        raise Underivable(f"path with a delimiter character {bad!r}")
    if BUILD_DEFINITION not in tree.files:
        raise Underivable(f"no {EXECUTOR}{BUILD_DEFINITION}")
    text = expand_cmake(skip_script_mode(strip_cmake_comments(tree.files[BUILD_DEFINITION].decode("latin-1"))))
    spread = CMAKE_SPREAD.search(text)
    if spread:
        raise Underivable(f"{BUILD_DEFINITION} has {spread.group(0)!r}")
    paths = [m.group(1) or "" for m in CMAKE_PATH.finditer(text)]
    entries = "|".join(map(re.escape, sorted({p.split("/", 1)[0] for p in tree.files}, key=len, reverse=True)))
    relative = re.compile(f"(?<![\\w.${{}}/\0-])((?:\\.\\.?|{entries})(?:/[^{DELIMITERS}]*)?)(?![^{DELIMITERS}])")
    paths += ["/" + m.group(1) for m in relative.finditer(text)]
    anchors = frozenset({BUILD_DEFINITION} | {literal_prefix(p) for p in paths})
    names: dict[str, list[str]] = {}
    for rel in tree.files:
        names.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
    todo = [rel for rel in tree.files if is_input(rel, anchors)]
    done: set[str] = set()
    while todo:
        rel = todo.pop()
        if rel in done:
            continue
        done.add(rel)
        if rel == BUILD_DEFINITION:
            continue
        name = rel.rsplit("/", 1)[-1]
        if name == BUILD_DEFINITION or name.endswith(".cmake"):
            raise Underivable(f"CMake file {EXECUTOR}{rel} among the inputs")
        text = tree.files[rel].decode("latin-1").replace("\\\r\n", " ").replace("\\\n", " ")
        for line in READ_LINE.findall(text):
            for token in re.findall(r"[^\s\"'<>(),;]+", line):
                todo += names.get(token.rsplit("/", 1)[-1], [])
    check_computed_includes(tree, done - {BUILD_DEFINITION})
    return anchors | done


def strip_cmake_comments(text: str) -> str:
    """CMake `text` without its line comments (quoted arguments, with their escapes, kept)."""
    out, i, quoted = [], 0, False
    while i < len(text):
        c = text[i]
        if quoted and c == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        if c == '"':
            quoted = not quoted
        elif c == "#" and not quoted:
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
            continue
        out.append(c)
        i += 1
    return "".join(out)


def skip_script_mode(text: str) -> str:
    """`text` without its `if(CMAKE_SCRIPT_MODE_FILE)` ... `endif()` blocks (never run by `cmake -S`)."""
    while start := SCRIPT_MODE_IF.search(text):
        depth = 0
        for m in CMAKE_IF.finditer(text, start.start()):
            depth += {"if": 1, "endif": -1}.get(m.group(1), 0)
            if m.group(1) in ("else", "elseif") and depth == 1:
                raise Underivable("its script-mode block has an else branch")
            if depth == 0:
                text = text[:start.start()] + text[m.end():]
                break
        else:
            raise Underivable("its script-mode block has no endif()")
    return text


def expand_cmake(text: str) -> str:
    """`text` with its one-value `set()` variables expanded, executor/ itself (ROOT_VARS, the directory of
    CMAKE_CURRENT_LIST_FILE) written "\\0", and the NON_READING_ROOT_USES removed."""
    values: dict[str, set[str]] = {}
    for m in CMAKE_SET.finditer(text):
        values.setdefault(m.group(1), set()).add(m.group(3))
    single = {k: v.pop() for k, v in values.items() if len(v) == 1}
    for _ in range(len(single) + 1):
        expanded = re.sub(r"\$\{(\w+)\}", lambda m: single.get(m.group(1), m.group(0)), text)
        if expanded == text:
            break
        text = expanded
    else:
        raise Underivable("its set() variables do not expand to a fixed point")
    text = text.replace("${CMAKE_CURRENT_LIST_FILE}", f"\0/{BUILD_DEFINITION}")
    for var in ROOT_VARS:
        text = text.replace(f"${{{var}}}", "\0")
    for use in NON_READING_ROOT_USES:
        text = text.replace(use, "")
    ascent = re.search(f"\\}}/\\.\\.(?![^/{DELIMITERS}])", text)
    if ascent:
        raise Underivable(f"a path ascends from an unexpanded variable: {text[max(0, ascent.start() - 40):ascent.end()]!r}")
    return text


def literal_prefix(spelled: str) -> str:
    """The executor/-relative file or directory a "/"-rooted path of the build definition names, cut before its first
    glob, variable or generator-expression component."""
    parts: list[str] = []
    for part in spelled.split("/")[1:]:
        if re.search(r"[*?\[\]$<>@{}]", part):
            break
        if part == "..":
            if not parts:
                raise Underivable(f"path {spelled!r} ascends out of {EXECUTOR}")
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    if not parts:
        raise Underivable(f"it reads {EXECUTOR} itself ({spelled or '/'!r})")
    return "/".join(parts)


def check_computed_includes(tree: Tree, inputs: set[str]) -> None:
    """Every `#include MACRO` of an input names a macro that inputs #define, none of them with token pasting."""
    texts = [tree.files[rel].decode("latin-1") for rel in sorted(inputs)]
    for macro in sorted({m for text in texts for m in COMPUTED_INCLUDE.findall(text)}):
        define = re.compile(rf"^[ \t]*#[ \t]*define[ \t]+{macro}\b.*$", re.MULTILINE)
        definitions = [d for text in texts for d in define.findall(text)]
        if not definitions or any("##" in d for d in definitions):
            raise Underivable(f"computed #include {macro} is not #defined by an input without ##")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-json", type=Path, default=ROOT / "build/native/build.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="verify the pins against build.json; write nothing")
    parser.add_argument("--stale", action="append", default=[], metavar="SHA",
                        help="with --check: a library SHA-256 the pins must not be (repeatable)")
    args = parser.parse_args(argv)
    if args.stale and not args.check:
        parser.error("--stale needs --check")
    bad = [s for s in args.stale if not SHA256.fullmatch(s)]
    if bad:
        parser.error(f"--stale takes lowercase hex SHA-256 values, not {bad}")
    root = args.root
    try:
        try:
            build = json.loads(args.build_json.read_text())
        except (OSError, ValueError) as e:
            raise Refused(f"cannot read {args.build_json}: {e}") from None
        sha = library_sha256(build)
        pairs = variant_pairs(build)
        if args.check:
            check(root, args.build_json, build, sha, args.stale, pairs)
            if not pairs:
                print(f"lock_native: checked: no deployment pins this variant; {LIBRARY} {sha} of {args.build_json} "
                      f"(git_head {build.get('git_head')}) is the library file's, not stale, and its executor tree")
                return 0
            print(f"lock_native: relocked: {', '.join(p for pair in pairs for p in pair)} pin {LIBRARY} {sha} of "
                  f"{args.build_json} (git_head {build.get('git_head')})")
            return 0
        texts = [(root / lock_rel, root / config_rel, *pin(root, sha, lock_rel, config_rel))
                 for lock_rel, config_rel in pairs]
        shipped = build.get("flags", {}).get("model", DEFAULT_MODEL) == DEFAULT_MODEL  # OH-10 is Qwen's lineage
        record = history_record(root, build, sha) if pairs and shipped else None
        old_locks = [(lock_path, lock_path.read_text()) for lock_path, _, _, _ in texts]
        try:
            for lock_path, _, lock_text, config_text in texts:
                lock_path.write_text(lock_text)
                validate("runtime_config", tomllib.loads(config_text), root)
        except ContractError:
            for lock_path, old in old_locks:
                lock_path.write_text(old)
            raise
        for _, config_path, _, config_text in texts:
            config_path.write_text(config_text)
        if record is not None:
            (root / LOCK_HISTORY).parent.mkdir(parents=True, exist_ok=True)
            with (root / LOCK_HISTORY).open("a") as history:
                history.write(record)
    except (Refused, ContractError) as e:
        print(f"lock_native: {e}", file=sys.stderr)
        return 2
    if not pairs:
        print(f"lock_native: no deployment pins this variant (flags {build['flags']}): nothing written; {LIBRARY} "
              f"{sha} from {args.build_json} (git_head {build.get('git_head')})")
        return 0
    print(f"lock_native: backend_build_sha256={sha} from {args.build_json} (git_head {build.get('git_head')})"
          + (f"; {LOCK_HISTORY} records it after {json.loads(record)['replaced_executor_sha256']}" if record else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
