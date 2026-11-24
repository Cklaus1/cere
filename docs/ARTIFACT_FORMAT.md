# ExecutionArtifact contract v2

A prepared artifact is a **layout_only copy** of a source model's payload files, indexed by a v2 manifest. It is not a new tensor container: the shards stay safetensors files with their own headers, and nothing is repacked or requantized. `cerec prepare` is the only producer ([MODEL_COMPILER.md](MODEL_COMPILER.md)). Two verifiers read it: the Rust loader `cere_compiler::artifact` and the Python reference inspector `scripts/verify_artifact.py`. Both share the invalid-case table `tests/fixtures/artifact_invalid.json`. `examples/artifact/` is a tiny synthetic v2 artifact used by the tests.

## Layout

```
ROOT/<artifact_id>/                 0555, files 0444
  <payload files, HF names>         shards (*.safetensors), index, config, quant config, generation config,
                                    tokenizer, template, preprocessor configs
  draft/token_map.u32le             converted NEXTN token map (only when the profile is `speculative: "nextn"`)
  manifest.json                     canonical JSON
  manifest.sha256                   "<hex>\n"
ROOT/<artifact_id>.verified.json    stamp sidecar (outside the artifact directory)
ROOT/.cerec.lock                    prepare/stamp writer lock
```

`artifact_id` is `<name>-<first 16 hex of the manifest SHA-256>`. The real artifact lives under `/var/lib/cere/artifacts/`, outside every checkout; the bytes are never tracked in git.

## Identity

The manifest SHA-256 (over the exact `manifest.json` bytes) is the artifact's identity. It is pinned out of band: in the repository by the pin file `configs/artifacts/<name>.pin.json` ([CONFIGURATION.md](CONFIGURATION.md#prepared-artifacts)), and every reader takes it as `--expected-manifest-sha256`. Per-file checksums detect corruption but are not authenticity if an attacker can replace the manifest too. Optional signatures require a real trust policy; do not claim signing because hashes exist.

The manifest is a pure function of content, family rules, profile and transforms: no mtime, tool version or time. It also carries `model_hash`, `tokenizer_hash` and `template_hash`, computed with the deployment lock's own definitions ([CONFIGURATION.md](CONFIGURATION.md#deployment-identity)). `cerec prepare --lock` refuses to publish when they differ from the lock's `model_content_sha256`, `tokenizer_sha256` and `template_sha256`.

**Native-lock convention.** The existing `backend_managed` lock and capability profile are unchanged: their `artifact_sha256` describes the source directory SGLang loads. A native deployment lock sets `artifact_sha256` to the prepared artifact's manifest SHA-256. None exists yet; the first one is M2.2b. Until then `cere-native probe` takes the manifest SHA-256 from the artifact pin (`--expected-manifest-sha256`).

## Manifest v2

Canonical JSON: UTF-8, ASCII only, keys sorted, 2-space indent, `,`/`: ` separators, integers only, trailing newline. Both verifiers re-encode the parsed manifest and require the result to equal the file bytes, so a non-canonical manifest is rejected. Unknown fields are rejected. The size cap is 4 MiB.

| Field | Content |
|---|---|
| `schema_version` | `"2"` |
| `tensor_layout_abi` | `1`: safetensors little-endian, row-major, as stored. The executor ABI (`cere_executor.h`) is separate. |
| `quality_mode` | `layout_only` |
| `source` | `{revision, tree_sha256}` of the HF tree metadata |
| `profile_id`, `profile` | the embedded capability profile |
| `family_rules` | the embedded family rules |
| `model_hash`, `tokenizer_hash`, `template_hash` | lock-definition hashes |
| `transforms` | `[{name: "pickle_int_list_to_u32le:v1", output: "draft/token_map.u32le", source_sha256, source_bytes}]`; `[]` for a `speculative: "none"` profile (SelfTune ST4 S3b) |
| `files[]` | `{path, bytes, sha256, role, upstream?, safetensors?}` |
| `families[]` | `{family, component, role, dtype, placement, count, bytes, misaligned}` |
| `totals` | `by_dtype` and `by_placement`, `{count, bytes}` each |

File roles: `weights`, `weight_index`, `config`, `quant_config`, `generation_config`, `tokenizer`, `template`, `preprocessor`, `draft_token_map`. `upstream` is `{lfs_sha256}` for an LFS file, or `{size, blob_id, local_modification}` for a git-stored one. `safetensors` on a shard is `{header_bytes, header_sha256, tensors}`.

### Headers as the index

The manifest has no per-tensor list. Each shard's own safetensors header is the tensor index, pinned by `header_sha256` over bytes `[0, 8+n)`. A verifier parses every header, classifies every tensor, and must reproduce the manifest's per-file tensor counts and per-family counts, bytes and misaligned counts exactly.

The header parser rejects:

- `n` < 2, `n` > 16 MiB, or `8+n` beyond the file size;
- non-UTF-8 bytes, JSON other than an object, duplicate keys;
- per-tensor fields other than `dtype`, `shape`, `data_offsets`; `__metadata__` other than a string→string map;
- more than 1,000,000 tensors per file (4,000,000 in total; at most 16,384 files);
- `end` < `begin`, `end` past the data region, overlaps, gaps, or a data region that does not reach the end of the file;
- overflowing `prod(shape) × itemsize`, a byte count that disagrees with the offsets, or a dtype outside `BF16 F16 F32 F64 I8 I16 I32 I64 U8 U16 U32 U64 F8_E4M3 F8_E5M2 BOOL`.

### Families and placement

A committed, model-specific rules file (`configs/artifacts/*.families.json`) lists ordered rules `{family, pattern, component, role, dtype, shape_rule, placement}` plus `groups`. Every tensor must match exactly one rule's anchored pattern; zero or two matches is a rejection. Group checks cover the NVFP4 quad (weight U8, block scale F8_E4M3, two F32 scalars, all four present, input dimension divisible by `nvfp4_group_size` 16), the PLE table shards (`ple_table_shards`), the PLE module and index members, and the shards' expert-range `__metadata__` (`layer`, `expert_start`, `expert_end`).

Placement comes from the profile, by family role: NVFP4, dense, the PLE module and the MTP draft go to `device`; the PLE table, scale and index to `host_pinned`; vision is `unused` in the text-only profile. Another profile over the same bytes is a relabel (new manifest, hardlinked payload), not a new copy.

The vision profile `rtxpro6000bw-plehost-vision-v1` (m213 design note D9) is `rtxpro6000bw-plehost-text-v1` with `vision: device` and `modalities ["text", "image"]`: the 333 `model.visual.*` BF16 tensors (897,862,112 B) enter the desc as DEVICE tensors and the desc carries the `vision_config` of `config.json`, which the CUDA executor accepts only as its compiled A4 vision tower (`a4_vision_support`; capability 61 `native_image_input`). Its artifact is a relabel of the text artifact (`cerec prepare --profile configs/artifacts/profiles/rtxpro6000bw-plehost-vision-v1.json` → outcome `relabelled`, payload bytes read 0), pinned in `configs/artifacts/qwen38-flash-next-nvfp4-vision.pin.json` and served by `configs/runtime/native-g0-vision.toml`; `native-g0.toml` stays on the text artifact.

The FP4-Local profile `rtxpro6000bw-plehost-nvfp4-text-v1` (fp4local design note D8, FL-12) has the text profile's placement and is used with `configs/artifacts/qwen4_exp-nvfp4-plenvfp4.families.json` (rules `qwen4_exp-nvfp4-plenvfp4-v1`) over the `cerec convert` output. That file differs from `qwen4_exp-nvfp4.families.json` only in the PLE table: the 128 shards are U8 `[rows, 80]` (two E2M1 values per byte), each with its own `shard_<k>.weight_scale` F8_E4M3 `[rows, 10]` (one scale per 16 values, role `ple_scale`), plus exactly one F32 `[1]` `weight_scale_2` global. The group check (Rust `families.rs`, Python `model_inventory.py`) requires one scale per shard with that shape and exactly one global for a U8 table; the FP8 table keeps its single `[1]` `weight_scale` check.

### Alignment

Safetensors permits tensor data offsets that are not a multiple of the element size, and the real model has 444 of them (BF16 and I64 tensors). v2 does not reject them. The loader records `element_aligned` per tensor, and the manifest reports a `misaligned` count per family. "Aligned offsets" means alignment to the declared layout's requirement, which is 1 byte for `tensor_layout_abi` 1. A loader copies tensor bytes into aligned staging buffers and never forms typed views on file offsets.

## Verification

Checks run in one order, shared by both verifiers:

1. manifest size, digest against the pin, strict parse, canonical bytes, structure, semantic rules;
2. the directory walk (no symlinks, special files, unlisted entries or unsafe paths);
3. file opens and sizes;
4. shard headers (bounds, pinned hash, parse, totals);
5. families, expert metadata, groups, then the per-file and per-family counts;
6. by mode:
   - `headers` (library and Python only): stops here, reading no payload byte;
   - `stamp`: the stamp must cover the held files (below), reading no payload byte;
   - `full`: the SHA-256 of every listed file.

### Stamp

The stamp `ROOT/<artifact_id>.verified.json` is a sidecar outside the artifact directory: `{format: 2, manifest_sha256, stamp_source, source_fingerprint, files: {path: {fsid|dev, ino, size, mtime_ns, ctime_ns, mode}}, verified_utc}`. `fsid` (16 hex digits of `fstatfs` `f_fsid`) names the filesystem where ext2/3/4 derive it from the superblock UUID, so a reboot that renumbers `st_dev` keeps the stamp; elsewhere the file is named by `dev`. A stamp without `format` (format 1, `dev` per file) is accepted only while every `st_dev` matches (m21 design note Q7, owner 2026-10-15).

| `stamp_source` | Written by | Accepted by |
|---|---|---|
| `prepare_hash` | `cerec prepare`, from the hashes computed while copying | only prepare's reuse decision |
| `full_verify` | `cerec verify --mode full` | stamp mode (`artifact::open(Mode::Stamp)`, `cerec verify --mode stamp`, `verify_artifact.py --mode stamp`, `cerec probe-load`), and prepare's reuse decision |

Stamp mode requires `full_verify`, the same manifest SHA-256, the same file set, and every file's recorded identity equal to `fstat` of the descriptor actually held. A `touch`, `chmod`, rewrite or replaced inode fails it with "full verify required" (`cerec` exit 4). The stamp caches a verification result against file identity. It is **not** an authenticity proof: anyone who can write the root can write a stamp.

### Descriptor-held loading

`cere_compiler::artifact::open(dir, expected_manifest_sha256, mode)` opens `dir` once as `O_DIRECTORY` and opens every listed file with `openat(dirfd, name, O_RDONLY|O_NOFOLLOW|O_CLOEXEC)`, requiring a regular file. It holds those descriptors for the `Artifact`'s lifetime. The stamp is compared with `fstat` of the held descriptors, full mode hashes through them, and `Artifact::stage` reads only through them (`pread`). A file renamed over after `open` is never read. Any rejection is a typed error, and there is no partial artifact.

`Artifact::stage(class, buffers, sink, options)` streams every tensor of a placement class (`device`, `host` = `host_pinned`, `all`) that the sink wants (`Sink::wants(tensor)`, default all) once. Buffers are caller-provided, 4096-byte aligned and at least 64 KiB. A declined tensor is never read, unless its file is hashed (`options.hash`): then it is read and hashed but not delivered, so a hashed file is still hashed whole. The work is a plan of units taken by one thread per buffer from one shared queue (m216 design note L2): a hashed file whose tensors are all in the class is one unit (file-offset order, for the hash); elsewhere, consecutive wanted tensors that are contiguous in the file and together fit one buffer form one run read by one `pread`, and a tensor larger than a buffer is a unit of its own. Large units go first, longest first. `Sink::accept(tensor, offset_in_tensor, data, is_last)` gets the pieces of one tensor from one thread, in increasing offset order, tiling it exactly. Different tensors may arrive concurrently. The sink must not keep the slice, and an error from it aborts staging. `StageReport.reads` counts the `pread`s of tensor data.

## Publication and rollback

`cerec prepare` writes into `ROOT/.tmp-*` on the destination filesystem, fsyncs payload and manifest, sets the modes, then renames the directory into place with no-replace semantics. A failure removes the temp directory and never replaces a published artifact. Stale `.tmp-*` directories are removed by the next prepare under `ROOT/.cerec.lock`. Registry pointers, reference counting of active instances and garbage collection do not exist yet (M2.2 and later).

## Load stages

DISCOVER -> VALIDATE -> ALLOCATE -> UPLOAD -> INSTANTIATE -> SMOKE -> READY.
M2.1 implements DISCOVER, VALIDATE and the host READ (and optional HASH) of staging, measured by `cerec probe-load`. A stage failure closes local resources and leaves the instance FAILED, with no partial service. Record cold/warm cache evidence, resident weights, disk-only tables, staging high-water mark, checksumming and graph-init time. First-request/page-fault work is included in readiness/first-token metrics.

As implemented in M2.2a (`crates/cere-runtime/src/native/upload.rs`, `executor/cuda/src/device/`, `cere-native probe`):

- **ALLOCATE** (`cere_model_create`): the model desc is built from the opened artifact. Placement comes from the families: `device` → DEVICE, `host_pinned` → HOST_PINNED, the MTP draft family → DISCARD; `unused` tensors are not in the desc. The A4 config comes from the artifact's `config.json` `text_config`, the stop tokens from the `eos_token_id` values of `generation_config.json` and `text_config`; both files are read by path and checked against the manifest's SHA-256. A config value or shape outside the compiled support table is `CERE_ERR_UNSUPPORTED` before any put; a plan that exceeds the memory budgets is `CERE_ERR_CAPACITY` with nothing allocated. The CUDA device takes the whole plan in one device allocation (DEVICE tensors, workspace, session slots) and one mapped pinned host allocation (HOST_PINNED tensors).
- **UPLOAD**: one `Artifact::stage(All)` pass with 32 × 16 MiB staging buffers (m216 design note L2; 4 × 64 MiB and a device pass before a host pass until M2.16); each staging thread calls `cere_model_put` directly. The sink declines DISCARD tensors (the MTP draft family when speculation is off) and tensors outside the desc, so their bytes are never read (L1). DEVICE ranges go through a pinned bounce-ring slot and are stream-synchronized before the put returns; HOST_PINNED ranges are copied into the mapped host arena. Counters: `uploaded_device_bytes`, `uploaded_host_bytes`, `discarded_bytes` (the skipped DISCARD bytes). A sink error aborts staging and nothing is sealed.
- **INSTANTIATE** (`cere_model_seal`): requires exact tiling of every non-discarded tensor, drains the device and checks for errors. Every CUDA module was already loaded at executor creation (`CUDA_MODULE_LOADING=EAGER`); nothing is allocated afterwards.
- **SMOKE**: one greedy token after a fixed 4-token sequence on a fresh session. The CUDA library has no forward pass yet (M2.2b), so SMOKE reports `unsupported` and `ready: false`; it is never reported as passed.

Measured on the real artifact (X5, native-probe-20261006T214923Z.md, library `87277dd2...`, cold, warm, cold, warm): every run uploaded exactly 77,843,711,744 device bytes and 51,200,246,042 pinned host bytes and discarded 5,214,301,696 MTP bytes; the executor used 75,996 MiB of GPU0 (weights, workspace, one session slot, CUDA context and modules), leaving 19,397 MiB free; probe peak RSS 49.5 GiB (the pinned PLE table). Stage times: validate about 3.4 s, allocate 8.0-15.9 s, read + upload 34.5-36.4 s, instantiate ≤ 5 ms, total 51-61 s. SMOKE was `unsupported` in every run. Both warm runs were `partial_warm` (16.6 % and 20.3 % resident before the run). No time threshold applies yet.

## NVFP4 layout and scale semantics

Proven on the real checkpoint in the pinned image (X2, suite `layout`, report native-test-layout-20261006T213912Z.json; kernels `executor/cuda/src/kernels/k0_nvfp4.h`):

- A weight `[n, k]` is stored as `weight` U8 `[n, k/2]`, row-major; element `2j` is the **low** nibble and `2j+1` the high nibble of byte `j`. A nibble is an E2M1 code: bit 3 sign, bits 0-2 index `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
- `weight_scale` is FP8 E4M3 `[n, k/16]`, row-major (the linear, not swizzled, layout): one scale per 16 consecutive `k` elements. `weight_scale_2` is one F32 per tensor.
- Value = `e2m1 × (float(weight_scale) × weight_scale_2)`. The native dequant equals FlashInfer `e2m1_and_ufp8sf_scale_to_float(..., is_sf_swizzled_layout=False)` bit for bit on 18/18 tensors (layers 0/23/47 × gate/up/down × experts 0/511); the swapped nibble order does not match on any of them.
- Activations are quantized with one global scale per layer and GEMM, `1 / max(input_scale)` over experts (and gate/up shards); the native codes and scales equal `fp4_quantize(x, 1/max_input_scale, 16, is_sf_swizzled_layout=False)` bit for bit for the gate/up (w13) and down (w2) inputs of layers 0/23/47 at M ∈ {1, 64, 2,048} (`fp4_quantize` defaults to the swizzled scale layout; the native quantizer writes the linear one).
- `weight_scale_2` of up and gate differ for 0 of 48 × 512 experts.

The MoE block built on these semantics (K7) is compared with SGLang's `flashinfer_cutlass` MoE in the kernel suite ([CORRECTNESS.md](CORRECTNESS.md#native-kernel-oracles-m22a)).
