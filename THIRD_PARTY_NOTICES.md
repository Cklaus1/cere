# Third-party notices

CERE itself is MIT-licensed (see [LICENSE](LICENSE)). This file lists the third-party code that is part of this
repository, the code CERE links against when built, and the projects whose published math CERE's kernels follow.
`executor/THIRD_PARTY.json` is the machine-readable form; `scripts/check_third_party.py` (run by `make validate`)
checks it against the tree and against a CUDA build's `build.json`.

## Vendored source

### FlashInfer 0.6.17 (Apache-2.0), with TensorRT-LLM and CUTLASS headers it bundles

`executor/third_party/flashinfer-0.6.17/` holds verbatim files from the `flashinfer_python` 0.6.17 wheel (pinned by
its `RECORD`): the SM120 NVFP4 CUTLASS fused-MoE sources used by the CUDA executor's `p3b` MoE backend.

- FlashInfer: Apache License 2.0, full text in
  `executor/third_party/flashinfer-0.6.17/flashinfer_python-0.6.17.dist-info/licenses/LICENSE`. The wheel ships no
  `NOTICE` file.
- TensorRT-LLM kernels bundled in FlashInfer (`flashinfer/data/csrc/nv_internal/`): NVIDIA, Apache-2.0 (per-file
  headers).
- CUTLASS 4.5.0 and `cutlass_extensions` (as bundled in the wheel): NVIDIA, BSD-3-Clause; each file carries the full
  license text in its header.
- `flashinfer/data/csrc/nv_internal/include/tensorrt_llm/common/NvInferRuntime.h` has no license header; it is
  distributed inside the FlashInfer wheel and covered by the wheel's license.

No file under `executor/third_party/` is modified.

### Strata (MIT)

`executor/cuda/src/host/spec_lookup.h` (the suffix-lookup drafter and the per-request draft policy) is adapted from
[Niko1221/Strata](https://github.com/Niko1221/Strata) at `fb58e0d` (`include/strata/spec/suffix_drafter.hpp`,
`include/strata/spec/draft_policy.hpp`):

```
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Linked when built (not in this repository)

The CUDA executor (`libcere_executor.so`) links statically against `cudart_static`, `cublasLt_static` and `culibos`
from the CUDA 13.0 toolkit, under the NVIDIA CUDA Toolkit EULA. A built library that you redistribute contains those
libraries; the EULA's redistribution terms apply to it. The Vulkan executor links `libvulkan` dynamically; the HIP
port links ROCm's `hipblaslt` dynamically.

## Math references (no code copied)

The kernels and the CPU reference paths follow the published math of these projects, which the GPU test oracles also
run against. No source from them is included.

| Project | License | Version |
|---|---|---|
| SGLang | Apache-2.0 | 0.5.19.dev3+gd4689328e, and upstream main at 61915e54 |
| FlashInfer | Apache-2.0 | 0.6.17 |
| flash-linear-attention (as adapted in SGLang) | MIT | the copy in SGLang 0.5.19.dev3+gd4689328e |
| Hugging Face transformers | Apache-2.0 | 5.24.0 |
| DICE-8B reference code | Apache-2.0 | DeadlyKitt3n/DICE-8B, main on 2026-11-11 |

## Rust crates

`cargo deny check licenses` (policy in [deny.toml](deny.toml)) checks every crate in `Cargo.lock`. At this release
the 203 third-party crates are under these licenses (an SPDX expression per crate, as declared):

| Crates | License expression |
|---|---|
| 127 | MIT OR Apache-2.0 (incl. `Apache-2.0 OR MIT`, `MIT/Apache-2.0`, `Apache-2.0 / MIT`) |
| 33 | MIT |
| 18 | Unicode-3.0 |
| 5 | Apache-2.0 (`prost*`, `sync_wrapper`) |
| 3 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT (`linux-raw-sys`, `rustix`, `wasi`) |
| 2 | Unlicense OR MIT (`aho-corasick`, `memchr`) |
| 2 | ISC (`rustls-webpki`, `untrusted`) |
| 1 each | MIT AND BSD-3-Clause (`matchit`); MIT OR Apache-2.0 OR LGPL-2.1-or-later (`r-efi`); Apache-2.0 AND ISC (`ring`); Apache-2.0 OR ISC OR MIT (`rustls`); Apache-2.0 OR BSL-1.0 (`ryu`); BSD-3-Clause (`subtle`); (MIT OR Apache-2.0) AND Unicode-3.0 (`unicode-ident`) |

Each crate's own license text ships in its source package on crates.io.

## Python tooling

The Python scripts use only the standard library at test time, plus `jsonschema` and `PyYAML`
(`requirements-dev.txt`). The tokenizer sidecar (`scripts/native/tokenizer_sidecar.py`) imports `tokenizers`,
`transformers` and, for images, tool calls and grammars, `torch`, `torchvision`, `Pillow`, `xgrammar` and `sglang`
at run time; none of them is included.

## Models

No model weights, and no tensors derived from them, are included; obtain each model from its publisher and accept its
license yourself. The models CERE serves are licensed by their publishers, not by this repository:

- **Qwen3.8-Flash-Next**: Qwen Community License 1.0 (the NVIDIA FP4-Local checkpoint is additionally governed by
  the NVIDIA Open Model Agreement).
- **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`): Apache-2.0.
- **DICE-8B**: Apache-2.0.

The `.safetensors` files under `tests/fixtures/model_src/` and `examples/artifact/` are synthetic:
`scripts/fixtures/make_model_fixture.py` writes them from a fixed seed with the standard library.
`tests/fixtures/model_configs/*/config.json` and `hf_quant_config.json` are the upstream checkpoints' configuration
files (Inkling 2 Mini and DICE-8B, both Apache-2.0), committed byte for byte so the descriptor checks can run offline.

The token-ID test fixtures under `tests/fixtures/native/` (the API, detokenizer, grammar, stop, thinking, tool and
`stop_tokenizer_mini.json` cases) were produced with the Qwen3.8-Flash-Next tokenizer and chat template. They hold
token IDs, special-token entries and rendered test prompts, no weights. The tokenizer is covered by:

```
Qwen Community License 1.0

Copyright (c) 2026 Qwen
```

See the full license on the model's Hugging Face page. `tests/fixtures/native/inkling_parser_cases.json` records the
outputs of SGLang's Inkling reasoning and tool-call parsers (Apache-2.0) on test strings.
