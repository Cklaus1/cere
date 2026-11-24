# CERE

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/tag/Cklaus1/cere?label=release)](https://github.com/Cklaus1/cere/tags)
[![Issues](https://img.shields.io/github/issues/Cklaus1/cere)](https://github.com/Cklaus1/cere/issues)
[![Discussions](https://img.shields.io/github/discussions/Cklaus1/cere)](https://github.com/Cklaus1/cere/discussions)

CERE is an LLM inference server with an OpenAI-compatible HTTP API. A Rust control plane (`cered`) owns admission,
scheduling, sessions, auth, metrics and the API surface; a native executor library loaded through a versioned C ABI
(`executor/include/cere_executor.h`) runs the model on the GPU. There are three executors:

- **CUDA** (`executor/cuda/`): Blackwell `sm_120a` (RTX PRO 6000 Blackwell, RTX 5090).
- **Vulkan** (`executor/vulkan/`): AMD RDNA 3.5 through Mesa RADV (tested on the Radeon 8060S / gfx1151).
- **HIP** (`executor/hip/`): a port of the CUDA executor to ROCm on gfx1151; a parity baseline, not a serving path.

Supported models: **Qwen3.8-Flash-Next** (NVFP4 / FP4-Local MoE with MTP speculation, vision, tool calls,
structured output) and **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`, with MTP drafts and vision on CUDA).
DICE-8B runs as a dev profile. CERE can also front SGLang or llama.cpp servers (`sglang_http`,
`cere-llamacpp-shim`) and federate workers across hosts.

**How this was built:** CERE was developed with AI coding agents under human review.

## Support matrix

Status words: **production** = serving real traffic continuously; **tested** = measured end to end through `cered`
on that hardware; **partial** = runs with the limits listed; **untested** = no measurement exists. The numbers come
from the development repository's benchmark reports (not published here), dated; they are single-host measurements,
not guarantees.

| Model | Hardware / backend | Status | Evidence (date) | Config |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | RTX PRO 6000 Blackwell (96 GB), CUDA | **production** since 2026-11-11 | 4.26 ms per output token (234.8 tok/s per request) on served traffic, 16 decode rows, 262,144-token context (2026-11-21). Default dev profile `-v29`: 264.1 tok/s single-stream decode with speculation, 534.3 tok/s at 4 streams; TTFT 155 ms at 1.2K tokens (2026-10-20) | `configs/runtime/prod-t4.toml`, `native-g0-spec.toml` |
| Inkling 2 Mini | RTX PRO 6000 Blackwell, CUDA | **tested** (dev lane, not in production) | 198.2 tok/s single-stream with MTP drafts vs 119.6 without; 336.2 tok/s at 4 streams; 785.5 tok/s aggregate at 64 streams without drafts (2026-11-24). Quality gate is report-only | `configs/runtime/native-g0-inkling.toml`, `native-g0-inkling-plain.toml` |
| Inkling 2 Mini | Radeon 8060S (gfx1151), Vulkan | **partial**: one session, greedy only, text only, no speculation | 26.0 tok/s decode at short context, 23.8 tok/s at 4.8K; prefill 556.8 tok/s at 4.8K (2026-11-24) | `configs/runtime/native-amd-inkling-vk.toml` |
| Qwen3.8-Flash-Next | Radeon 8060S (gfx1151), Vulkan | **partial**: 32K context; tool calls refused | 17.6 tok/s plain, 19.3 tok/s with speculation on served bodies; 48/48 outputs equal the reference (2026-11-21) | `configs/runtime/amd-serve.toml`, `amd-serve-spec.toml` |
| Qwen3.8-Flash-Next | Radeon 8060S, HIP | **partial**: parity baseline only | 23.6 tok/s plain decode through `cered` (2026-11-21) | `configs/runtime/native-amd-s3.toml` |
| Qwen3.8-Flash-Next (IQ3_XXS GGUF) | Radeon 8060S, llama.cpp via `cere-llamacpp-shim` | **tested** (lab only) | llama-bench tg128: 28.7 t/s Vulkan, 25.5 t/s HIP (2026-11-11) | `configs/runtime/amd-flashnext.toml` |
| Inkling 2 Mini | RTX 5090 (32 GB), CUDA | **untested** | Planned: the 20 GB pack leaves about 10 GiB for KV (one 256K or two 128K sessions; 1M context does not fit). No measurement yet | none |
| Qwen3.5-4B | RTX 5090, CUDA | **untested** (port in progress, not in this release) | The port passes its numerical gates on an RTX PRO 6000; acceptance on the RTX 5090 has not passed | none |
| Qwen3.8-Flash-Next | RTX 5090 (32 GB) | **not supported** | Needs about 77 GB of VRAM plus a 28.8 GB PLE table in pinned host RAM; no 32 GB profile exists | none |
| DICE-8B | RTX PRO 6000, CUDA | **partial**: dev only, one session, BF16, greedy | Served requests all returned 200 (2026-11-23); no throughput claim | `configs/runtime/native-g0-dice8b.toml` |

The AMD numbers need a unified-memory part with enough GTT: Qwen3.8-Flash-Next uses about 104 GB, so a 128 GB
Ryzen AI Max+ class machine. The Vulkan K2 path needs `VK_KHR_cooperative_matrix` (without it Inkling decode drops
to 7.5 tok/s).

## Requirements

- **Build (CPU only):** Rust 1.85.1 (pinned by `rust-toolchain.toml`), `protoc` (protobuf-compiler), a C++17
  compiler, Python 3.11+ with `jsonschema` and `PyYAML` for the tests (`requirements-dev.txt`).
- **CUDA executor:** CUDA 13.0 toolkit (nvcc 13.0.88 is the toolchain of record), CMake ≥ 3.28, Ninja, g++ 13,
  `readelf`. The library targets `sm_120a` only (no PTX): it runs on compute capability 12.0 Blackwell cards. An
  NVIDIA driver with CUDA 13 support.
- **Vulkan executor:** g++ 13, `glslc` (shaderc 2023.8 / glslang 14), `libvulkan-dev` 1.3.275, Mesa RADV 25.2 or
  newer with `VK_KHR_cooperative_matrix`.
- **HIP port:** ROCm 7.2 (`hipcc --offload-arch=gfx1151`, `hipblaslt`).
- **Tokenizer sidecar (at serve time):** Python with `tokenizers` and `transformers`; for images, tool calls and
  grammars also `torch`, `torchvision`, `Pillow`, `xgrammar` and `sglang`.
- **Host RAM:** Qwen3.8-Flash-Next pins its PLE table in host RAM (28.8 GB for the S3 artifact); the production
  profile wants MemAvailable ≥ PLE + 16 GiB + an 8 GiB snapshot arena at start. `cerec prepare` and `cerec verify`
  refuse below 16.5 GiB free.

## Quick start

### 1. Build and test (no GPU)

```sh
cargo build --release --workspace --locked
make ci            # validate, fmt, clippy, toy executors, C ABI checks, host tests, Rust and Python suites
```

`make ci` runs everything on the CPU: the Rust tests drive `cered` against toy executors built from the real host
layer (`make toy-executors`). Tests that need GPU hardware or model-derived reference data skip with a message.

### 2. Build an executor

```sh
make executor-cuda CERE_MODEL=qwen38-flash-next     # or CERE_MODEL=inkling-2-mini; writes build/native-<model>/
CERE_MODEL=inkling-2-mini sh executor/vulkan/build_lib.sh   # Vulkan; writes build/vulkan/
```

See [docs/BUILDING.md](docs/BUILDING.md) for toolchains, provenance (`build.json`) and pinning.

### 3. Prepare a model artifact

`cerec` turns a Hugging Face checkpoint into a verified CERE artifact (a manifest of every tensor with its SHA-256):

```sh
cargo build --release --bin cerec
# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, Apache-2.0): download and pin the source
python3 scripts/native/model_source.py tests/fixtures/model_configs/inkling-2-mini/source.json "$SRC" --mode download --hash-weights
target/release/cerec prepare "$SRC" --name inkling-2-mini-nvfp4-mxfp8 \
  --families configs/artifacts/inkling2_loop-nvfp4-mxfp8.families.json \
  --profile configs/artifacts/profiles/rtxpro6000bw-inkling-text-v1.json --out "$ARTIFACTS"
target/release/cerec verify "$ARTIFACTS/<artifact id>" --expected-manifest-sha256 <sha256> --mode full
```

Qwen3.8-Flash-Next is converted from its FP4-Local checkpoint first (`cerec convert ... --format s3`), then prepared
with `configs/artifacts/qwen4_exp-nvfp4-plenvfp4-mxfp8.families.json`. [docs/ARTIFACT_FORMAT.md](docs/ARTIFACT_FORMAT.md)
and [docs/MODEL_COMPILER.md](docs/MODEL_COMPILER.md) describe the format and the compiler. A verify stamp does not
travel between hosts: run `--mode full` again after copying an artifact.

### 4. Configure and run `cered`

Start from a shipped profile (for example `configs/runtime/native-g0-inkling.toml`) and set, under
`[deployments.native]`: `library` and `library_sha256` (your build), `artifact_dir` and `expected_manifest_sha256`
(your artifact), `device_uuid` (`nvidia-smi -L`; `"vk:0"` on Vulkan) and the `sidecar` path. After a rebuild,
`python3 scripts/native/lock_native.py --build-json build/native-<model>/build.json` re-pins the library in the
config and its deployment lock, and `python3 scripts/pin_configs.py --write` refreshes the config hashes.

```sh
export CERE_OPERATOR_TOKEN=... CERE_TOKEN_NATIVE_A=...
CUDA_VISIBLE_DEVICES=<the GPU UUID> target/release/cered --config configs/runtime/native-g0-inkling.toml
```

`cered` refuses to become ready when any pin (library, artifact, tokenizer, template, device) does not match:
mismatches are reported by name on `/readyz`.

### 5. Call the API

```sh
curl -s http://127.0.0.1:18309/v1/chat/completions \
  -H "Authorization: Bearer $CERE_TOKEN_NATIVE_A" -H 'Content-Type: application/json' \
  -d '{"model": "Inkling-2-Mini", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

Endpoints: `/v1/chat/completions`, `/v1/responses` and `/v1/models` (OpenAI-compatible, streaming included),
`/v1/messages` (Anthropic Messages-compatible), `/v1/cere/*` (sessions, state export/import, request events, stats),
`/metrics` (Prometheus), `/healthz` and `/readyz`. The model name is the profile's `served_model_name`. See
[docs/API.md](docs/API.md), [docs/CONFIGURATION.md](docs/CONFIGURATION.md) and
[docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).

Opt-in research hooks (activation capture, interventions, steering vectors; Qwen3.8-Flash-Next on CUDA, off by
default and bit-identical when unused) are described in [docs/RESEARCH_HOOKS.md](docs/RESEARCH_HOOKS.md).

## Repository layout

| Path | What |
|---|---|
| `crates/cere-runtime` | `cered` (server), `cere-worker` (federated worker), `cere-native` (executor probe), `cere-rpc-probe`; the scheduler and API |
| `crates/cere-compiler` | `cerec`: artifact inventory, conversion, prepare, verify |
| `crates/cere-core`, `cere-proto` | shared types; the worker gRPC protocol (`proto/`) |
| `crates/cere-llamacpp-shim`, `cere-mock-sglang` | the llama.cpp adapter; a mock SGLang server for tests |
| `executor/` | the C ABI, the CUDA / Vulkan / HIP executors, model descriptors, host tests |
| `configs/` | runtime profiles, deployment locks, artifact families and profiles |
| `schemas/`, `examples/` | JSON Schemas for every contract and example documents |
| `scripts/` | the tokenizer sidecar, artifact and lock tooling, fixture generators |

Some documents and comments refer to development scripts, design notes and benchmark reports that are not part of
this repository.

## Model data in this repository

No model weights and no tensors computed from them are included. The safetensors under `tests/fixtures/model_src/`
and `examples/artifact/` are synthetic (written by `scripts/fixtures/make_model_fixture.py`). Tests that compare
against reference outputs of real models (the Inkling 2 Mini and DICE-8B reference sets, the GPU oracles under
`executor/tests/gpu/`) skip without their data; [docs/BUILDING.md](docs/BUILDING.md) says how to regenerate it.
The API, tool-call, grammar and detokenizer fixtures under `tests/fixtures/native/` hold token IDs produced with the
Qwen3.8-Flash-Next tokenizer (Qwen Community License 1.0); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Community

- **Questions, ideas, show-and-tell:** [GitHub Discussions](https://github.com/Cklaus1/cere/discussions).
- **Bugs and feature requests:** [GitHub Issues](https://github.com/Cklaus1/cere/issues).
- **Pull requests** are welcome; every commit needs a DCO sign-off (`git commit -s`), see
  [CONTRIBUTING.md](CONTRIBUTING.md).
- **Security reports:** privately, through GitHub's private vulnerability reporting; see [SECURITY.md](SECURITY.md).

## Releases

Releases are tagged snapshots of the development repository: each one is a fresh export, tagged `v<version>`
(the first is `v0.6.0`). See [CHANGELOG.md](CHANGELOG.md).

## Licensing

- **CERE's own code** is MIT-licensed, see [LICENSE](LICENSE).
- **Vendored third-party code** (FlashInfer with its TensorRT-LLM and CUTLASS headers, a Strata adaptation) keeps its
  own license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Model weights are not included.** Qwen3.8-Flash-Next (Qwen Community License 1.0; the NVIDIA FP4-Local
  checkpoint also under the NVIDIA Open Model Agreement), Inkling 2 Mini (Apache-2.0) and DICE-8B (Apache-2.0) are
  licensed by their publishers: obtain them yourself and accept their terms.
- **CUDA:** executor binaries built from this repository statically link NVIDIA CUDA libraries (`cudart_static`,
  `cublasLt_static`, `culibos`) and are subject to the NVIDIA CUDA Toolkit EULA when you distribute them.
