# Changelog

All notable changes to CERE are listed here. The project follows [Semantic Versioning](https://semver.org/) from its
first public release; the internal package version (`PACKAGE_VERSION`) is 0.6.

## 0.6.0 - 2026-11-24

First public release, MIT-licensed.

- `cered`: OpenAI-compatible (`/v1/chat/completions`, `/v1/responses`, `/v1/models`, streaming) and Anthropic
  Messages-compatible (`/v1/messages`) API; bearer-token scopes with weighted fair queuing; sessions with snapshot,
  restore and fork; state export/import; prefix cache with a host-RAM spill tier; structured output and tool calls;
  Prometheus metrics; config hot reload; readiness that refuses any unpinned library, artifact, tokenizer or template.
- Native CUDA executor for Blackwell `sm_120a`: Qwen3.8-Flash-Next (NVFP4 / FP4-Local MoE, MTP speculation, vision,
  grammar-constrained drafts, 16 decode rows, 262,144-token context) and Inkling 2 Mini (MTP drafts, vision).
- Vulkan executor for AMD RDNA 3.5 (Radeon 8060S): Inkling 2 Mini and Qwen3.8-Flash-Next; HIP port as a parity
  baseline.
- `cerec`: checkpoint inventory, FP4-Local to S3 conversion, artifact prepare and verify.
- Federation: remote workers over mutual-TLS gRPC (`cere-worker`), placement and session migration.
- Backends besides the native executors: SGLang over HTTP and llama.cpp through `cere-llamacpp-shim`.
- Opt-in research hooks on the CUDA Qwen3.8-Flash-Next path: activation capture, interventions and steering vectors
  under `/v1/cere/research/*`, off by default ([docs/RESEARCH_HOOKS.md](docs/RESEARCH_HOOKS.md)).

See the support matrix in [README.md](README.md) for what is measured on which hardware.
