# Building and testing CERE

## CPU build and test suite

Everything below runs without a GPU.

```sh
cargo build --release --workspace --locked   # cered, cerec, cere-worker, cere-native, cere-llamacpp-shim, ...
make ci                                      # the full CPU gate
```

`make ci` = `make test fmt clippy`, and `make test` runs, in order:

| Target | What |
|---|---|
| `validate` | `scripts/check_third_party.py` (vendored and linked code vs `executor/THIRD_PARTY.json`) and `scripts/pin_configs.py --check` (every runtime config's pinned hash) |
| `test-python` | `python3 -m unittest discover -s tests -t .` (needs `jsonschema` and `PyYAML`, `requirements-dev.txt`) |
| `test-c` | C11 and C++17 compile checks of `executor/include/cere_executor.h` against the ABI layout fixture |
| `test-executor-host` | builds the toy executors (the real host layer over a deterministic CPU "device") and runs `executor/tests/host/*.cpp` |
| `test-rust` | `cargo test --workspace --locked` with the toy executors exported as `CERE_TOY_EXECUTOR*`, so `cered` is exercised end to end |
| `test-proto` | compiles `proto/cere_worker.proto` with `protoc` and runs the `cere-proto` wire tests |

`make deny` runs `cargo deny check licenses bans sources` with [deny.toml](../deny.toml) (needs `cargo-deny`);
`cargo deny check advisories` additionally fetches the RustSec advisory database.

Tests that need hardware or model-derived reference data are not part of `make ci`; see "GPU tests" below.

## CUDA executor

Requirements: CUDA 13.0 toolkit (nvcc 13.0.88 is the toolchain of record), CMake ≥ 3.28, Ninja, g++ 13, `readelf`.

```sh
make executor-cuda CERE_MODEL=qwen38-flash-next   # or inkling-2-mini, dice-8b
```

This configures `executor/CMakeLists.txt` into `build/native-<model>/` and builds:

- `libcere_executor.so`: the host layer (`executor/cuda/src/host/`), the CUDA device and the kernels, for
  `sm_120a` only (no PTX), with cudart / cuBLASLt / culibos linked statically, hidden visibility and the ABI version
  script. It loads only on compute capability 12.0 (RTX PRO 6000 Blackwell, RTX 5090).
- `libcere_ktest.so`: the kernels with `cere_ktest_*` entry points, for the GPU kernel tests.
- `build.json`: provenance (compiler versions, flags, the source tree hash, output SHA-256s and `NEEDED`).
  `CERE_IMAGE_DIGEST` names the container image the build ran in; the Makefile passes all zeros for a host build.
  `scripts/check_third_party.py --build-json build/native-<model>/build.json` checks the linked libraries.

`CERE_MODEL` selects the model descriptor `executor/models/<model>.json`, from which the build generates
`cere_model_constants.h`; a library serves exactly one model family.

## Vulkan executor (AMD)

Requirements: g++ 13, `glslc` (shaderc 2023.8, glslang 14.0), `libvulkan-dev` 1.3.275; at run time Mesa RADV 25.2 or
newer. The K2 GEMM path needs `VK_KHR_cooperative_matrix`.

```sh
CERE_MODEL=inkling-2-mini sh executor/vulkan/build_lib.sh    # or qwen38-flash-next (the default)
```

`build_lib.sh` compiles the shaders and kernels (`executor/vulkan/build.sh`), links the unchanged host layer with the
Vulkan device into `build/vulkan/libcere_executor_vulkan.so` and prints the library's SHA-256. The kernel and device
tests it builds (`build/vulkan-t2b/`) need the GPU: see `executor/vulkan/README.md`.

## HIP port (AMD)

`sh executor/hip/build.sh` (ROCm 7.2, `hipcc --offload-arch=gfx1151`, `hipblaslt`) writes
`build/hip/libcere_executor_hip.so`. It is a parity baseline for the Vulkan executor, not a serving path.

## Pinning a build into a profile

A runtime profile pins the library it loads (`[deployments.native] library_sha256`) and its deployment lock pins the
same digest (`backend_build_sha256`); `cered` refuses a mismatch. After building:

```sh
python3 scripts/native/lock_native.py --build-json build/native-<model>/build.json   # re-pins config + lock
python3 scripts/pin_configs.py --write                                               # refreshes tests/fixtures/config_pins.json
```

Set `library`, `artifact_dir`, `expected_manifest_sha256`, `device_uuid` and the sidecar path in the profile to your
host's values. Paths under `/var/lib/cere/` and `/opt/cere/` in the shipped profiles are placeholders.

## GPU tests

`executor/tests/gpu/` holds the kernel and model tests that run against `libcere_ktest.so` /
`libcere_executor.so` on a real GPU, with SGLang, FlashInfer and PyTorch as numerical oracles
(`python3 -m unittest discover -s executor/tests/gpu -p 'test_<suite>_*.py'`). They expect the source tree at
`/src`, the build under `build/native/` and the model artifact mounted read-only, as in a container. They are not
part of `make ci`.

### Reference data not included

These data sets were recorded from the real models' reference implementations and are **not** in this repository
(they are tensors and logits computed from model weights):

| Data set | Used by | How to regenerate |
|---|---|---|
| `tests/fixtures/inkling2_reference/` (per-op tensors, golden logits, tap traces, drafter rows, vision tensors) | the Inkling GPU suites (`test_inkling_kernels_*`, `inkling_forward.py`, `inkling_drafter_golden.py`), `executor/vulkan/tools/ki_g3_compare.py` | Run the Hugging Face `transformers` 5.24.0 Inkling 2 Mini implementation (`modeling_inkling2_loop.py`) on `thinkingmachines/Inkling-2-Mini` at the revision in `tests/fixtures/model_configs/inkling-2-mini/source.json`, recording the tensors in the schemas the tests read (`cere-inkling2-ops-v1` and the files named in each test's docstring) |
| `tests/fixtures/dice8b_reference/primary.json`, `perturbed.json`, `thresholds.json`; `tests/fixtures/dice8b_identity/` | `test_sdar_kernels_model.py`, `sdar_generate.py` | Run the DICE-8B reference code (`DeadlyKitt3n/DICE-8B`, pinned in `tests/fixtures/model_configs/dice-8b/source.json`) over `tests/fixtures/dice8b_reference/prompts.json` (included), recording per-step tokens and confidences |

Without them those GPU tests stop with a missing-file error naming the path; nothing in `make ci` reads them.

## Model artifacts

`cerec` (`crates/cere-compiler`) prepares and verifies artifacts:

```sh
cerec inventory <src> --families configs/artifacts/<family>.families.json --out <inventory.json>
cerec convert <fp4-local checkpoint> --out <dir> --converter-git-sha $(git rev-parse HEAD) --threads 16 --format s3   # Qwen3.8-Flash-Next only
cerec prepare <src> --name <id> --families <families.json> --profile configs/artifacts/profiles/<profile>.json --out <artifacts dir>
cerec verify <artifacts dir>/<id> --expected-manifest-sha256 <sha256> --mode full    # then --mode stamp on later starts
```

| Model | Families | Profile |
|---|---|---|
| Qwen3.8-Flash-Next (S3) | `qwen4_exp-nvfp4-plenvfp4-mxfp8.families.json` | `rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1.json` |
| Inkling 2 Mini | `inkling2_loop-nvfp4-mxfp8.families.json` | `rtxpro6000bw-inkling-text-v1.json` / `-vision-v1.json` |
| DICE-8B | `sdar-bf16.families.json` | `rtxpro6000bw-bf16-text-v1.json` |

`scripts/native/model_source.py` downloads a checkpoint at a pinned revision and checks every file against its
`source.json`. A verify stamp is bound to the host: after copying an artifact, run `--mode full` on the new host.
