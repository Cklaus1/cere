# CERE model preparation (`cerec`)

## Decision

Retain the source/serving split, but implement a **verified prepared-weight cache** before designing a novel tensor container. Reuse a backend-supported format where possible; a manifest can index safetensors, GGUF or future packed sections. A file's extension does not prove kernel-ready layout.

SourceModel -> validated/prepared ExecutionArtifact -> hardware ExecutionProfile -> resident ModelInstance.

M2.1 implements the first step: `cerec` (`crates/cere-compiler`, library + binary) prepares, verifies and stages a **layout_only** artifact of one model ([ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md)). It copies the source payload unchanged and indexes it; it does not repack, requantize, deduplicate tensors, upload to a device or run a model. No native deployment lock, executor, GPU upload or INSTANTIATE/SMOKE stage exists yet (M2.2). `scripts/verify_artifact.py` (stdlib + jsonschema) is the independent reference inspector, `scripts/model_inventory.py` the reference header dump and `scripts/token_map.py` the reference token-map parser; Rust and Python agree on the shared table `tests/fixtures/artifact_invalid.json`.

## `cerec` commands

Every command prints one canonical JSON report on stdout. A failure prints `<error_class>: <message>` as the first stderr line and nothing on stdout. Exit codes: 0 success, 1 I/O error, 2 rejected input or usage, 3 `prepare --no-wait` found `ROOT/.cerec.lock` held, 4 the stamp does not cover the artifact ("full verify required"), 5 `MemAvailable` below the floor.

| Command | Does |
|---|---|
| `cerec convert SRC --out DIR --converter-git-sha SHA [--threads N] [--limits JSON]` | One-time offline FP4-Local -> CERE-layout converter (fp4local design note D2-D10, FL-10). Reads a modelopt `MIXED_PRECISION` checkpoint through its index and writes a new flat directory with the shipped NVFP4 checkpoint's tensor names, dtypes and shapes (PLE stays NVFP4, D8): MXFP8 -> BF16 exact, NVFP4-CSF routed scales -> plain E4M3 byte-exact (non-canonical streams refused), MTP experts dequantized in F32 and fused `[gate; up]` in BF16 (RNE), F32 norms -> BF16 RNE (inexact count reported), vision never written, everything else byte-copied. Writes `CONVERSION.json` and the HF tree `prepare` needs; refuses an existing `--out`. `--threads` defaults to 8. |
| `cerec inventory SRC --families F [--out FILE] [--limits JSON]` | Reads shard headers and HF tree metadata only. Writes the per-shard, per-dtype, per-family and per-placement tables and the header digest, byte-identical to `scripts/model_inventory.py`. |
| `cerec prepare SRC [--token-map PATH] --name NAME [--lock LOCK] --families F --profile P --out ROOT [--no-wait] [--threads N] [--min-mem-available-gib G] [--limits JSON]` | Builds or reuses the artifact `ROOT/<NAME>-<16 hex>` (below). `--token-map` is required when the profile's `speculative` is `nextn` and refused (class `token_map`, exit 2) when it is `none` (SelfTune ST4 S3b: DICE-8B's BF16 artifact has no draft). `--threads` defaults to 8 (1..256), the floor to 16 GiB. |
| `cerec verify DIR --expected-manifest-sha256 H --mode full\|stamp [--cache cold] [--limits JSON]` | `full` hashes every file through the held descriptors (8 threads) and rewrites the stamp as `full_verify`; `stamp` reads no payload. `--cache cold` evicts the artifact's files and records residency and a label first. |
| `cerec probe-load DIR --expected-manifest-sha256 H --class device\|host\|all --cache cold\|warm --hash on\|off [--baseline] [--threads N] [--buffer-mib M] [--min-mem-available-gib G]` | Stages one placement class into discarded buffers (`Artifact::stage`) and reports stage times, residency before and after, the cache label, both byte counters, the staging high-water mark, peak RSS (`VmHWM`) and `MemAvailable`. Needs a `full_verify` stamp. `--baseline` is the staged-copy baseline: the same files, ranges and buffers with no walk, size, header-hash, family or stamp check and no hashing (`--hash on` is refused). Defaults: 8 threads, 64 MiB buffers (1..1024). |
| `cerec evict DIR` | `fsync` + `POSIX_FADV_DONTNEED` on every file the manifest lists, plus `manifest.json`/`manifest.sha256`; reports the file count and the residency afterwards. |
| `cerec --version` | `cerec <crate version>; package v<PACKAGE_VERSION>` |

The source directory is read-only to every command. It is opened once (`O_RDONLY|O_DIRECTORY`); each file via `openat(..., O_RDONLY|O_NOFOLLOW|O_CLOEXEC)`, must be regular and is size-checked against `fstat`. Source reads use `POSIX_FADV_SEQUENTIAL` and `POSIX_FADV_NOREUSE`; eviction (`DONTNEED`) is applied only to artifact files. `--out` equal to, inside or containing SRC (after `realpath`) exits 2. `prepare`, `probe-load` and `verify` enforce the memory floor `MemAvailable ≥ G GiB + threads × buffer` (16 GiB + 512 MiB by default), checked at start and after every 1 GiB of payload I/O (`verify --mode full`: every hash chunk). No command `mlock`s or pins host memory.

### Cache labels

`--cache cold` evicts first; the run is labelled `cold` only if at most 1 % of the class's pages were resident before reading, else `cold_unverified`. `--cache warm` is an immediate repeat, labelled `warm` at 95 % or more resident, else `partial_warm`. There is no global page-cache drop.

## `cerec prepare` outcomes

Prepare selects the payload roles (shards, index, config, quant config, generation config, tokenizer, template, preprocessor configs); every other source file is listed in the report as `excluded`. It requires the HF tree metadata (`.cache/huggingface/trees/*.json`). Under the flock `ROOT/.cerec.lock` it removes stale `ROOT/.tmp-*` directories, then decides among the existing artifacts named `<NAME>-<16 hex>` that have a stamp matching their files:

| `outcome` | When | Payload read | Written |
|---|---|---|---|
| `reused` | the stamp's source fingerprint (stat-only: `path size mtime_ns header_sha256` per source file, plus the token map when there is one) matches, and the rebuilt manifest is that artifact's | 0 | nothing |
| `relabelled` | the content matches an existing artifact, but the rules or profile give a different manifest | 0 after a fingerprint match, else one hash pass | a new artifact directory with the payload hardlinked from the old one, its `manifest.json`, `manifest.sha256` and `prepare_hash` stamp; linking changes the shared inodes' ctime, so the stamp of every existing artifact sharing those inodes (not only the link source) is rewritten when its other `fstat` fields are unchanged |
| `reused_after_rehash` | no fingerprint match, but an artifact has the same `(path, bytes)` file set and a hash-only pass over the source reproduces its manifest | every source byte once, no copy | the stamp, with the new source fingerprint |
| `prepared` | none of the above | every source byte once (twice when the hash-only pass found changed content: `rehash_pass` true) | the full copy, `manifest.json`, `manifest.sha256` and a `prepare_hash` stamp |

A full prepare does one copy+hash pass into `ROOT/.tmp-*` with 1 GiB chunked `sync_file_range` + `POSIX_FADV_DONTNEED` on the written range. It checks the upstream metadata: LFS files must match `lfs_sha256`, non-LFS files are compared by size and a difference is reported as `local_modification`. It computes the lock-definition hashes and compares them with `--lock`; any difference exits 2 and publishes nothing. With a token map it converts it (`pickle_int_list_to_u32le:v1`, a stored-zip PyTorch pickle to `draft/token_map.u32le`); a `speculative: "none"` artifact has no `draft/` directory and an empty `transforms` list. It writes the canonical manifest and publishes with a no-replace rename.

### Byte counters

- `payload_bytes_read`: tensor data regions of shards, whole non-safetensors payload files and the token-map source.
- `metadata_bytes_read`: safetensors header regions (`8+n`), manifests, stamps, HF tree JSON and other small inputs.
- `bytes_written`: every byte `cerec` writes: payload, `manifest.json`, `manifest.sha256` and the stamp. A reuse writes at most the stamp.

Live run (report): `prepared` read 135,213,501,385 payload bytes and wrote 135,252,909,233 bytes in 56,315 ms; the repeat was `reused` with 0 payload bytes read in 2,575 ms.

## What goes offline

Validate tensor names/shapes and source identities; perform reversible packing/alignment/scale ordering; generate exact layout manifests; deduplicate genuinely identical shared tensors; write atomically. Requantization is a separate transformation with a separate quality gate and new model fingerprint. Retain original source identifiers and licenses.

## What stays process-local

Allocate actual device memory; validate capacity/capabilities; stream bytes through bounded staging; instantiate/update graph recipes using current pointers; initialize backend handles and run readiness smoke. A tuning profile is reusable data, not a serialized live CUDA graph execution object.

## Loader choices

Current SGLang docs describe `fastsafetensors` and `sharded_state` as different loader selections. `sharded_state` loads pre-sharded per-TP-rank state; this does not by itself establish that every NVFP4 layout transform or draft pass is skipped. Confirm on the exact image/model. Loader thread options are format-dependent. Do not stack incompatible loader settings or infer speedups from names.

The operator's reported startup breakdown motivates profiling, not proof of CPU causality. Read from files, CPU transform, copy, draft loading, graph construction and warmup need independent instrumentation. Warm cache and different resident bytes/quant layouts invalidate a direct 16-second-vs-minutes equivalence claim.

## Loader policy

Pinned buffers, buffered I/O, direct I/O and cache dropping are optional measured strategies. O_DIRECT requires alignment and may be worse when multiple replicas reuse warm files. Preserve useful host embeddings and immutable replica-shared bytes. No machine-wide cache purge or power changes by default.

For two replicas, stream/read once and fan out only when host-memory/PCIe topology and budgets support it. A failure on GPU1 must not corrupt GPU0's ready instance. Never declare readiness with a partial artifact.

External grounding: docs/RESEARCH_SOURCES.md (SGLang loading; NVIDIA graph semantics). Artifact contracts: docs/ARTIFACT_FORMAT.md.
