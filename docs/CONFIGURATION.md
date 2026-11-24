# Typed configuration

Runtime configuration is TOML validated against `schemas/runtime_config.schema.json`, **revision 2**. Two
validators enforce the same contract and must agree:

- Python CPU tooling: `validate("runtime_config", …)` in `scripts/cere_contracts.py` (schema plus cross-field
  rules), run by `make validate` for every `configs/runtime/*.toml`.
- Rust runtime: `cere_runtime::config::RuntimeConfig` (strict serde with unknown keys rejected, every schema
  pattern/range/const/conditional, and the same semantic rules). `cered --check-config <path>` prints
  `config ok <hash>` (plus one `deployment served_model_name=… recipe_id=… lock_sha256=… profile_id=…` line per
  entry with a pinned deployment, in config order) and exits 0, or prints the error and exits 2. It does not read
  auth secrets or touch `state_dir`.

The shared case table `tests/fixtures/runtime_config_invalid.json` lists invalid variants of the shipped TOMLs
(`base` + RFC 7396 JSON merge `patch`, then the optional `entry_patch` merged into `deployments[0]`;
`deployment_patch` patches the recipe/lock/capabilities files in a scratch root; `${REPO_ROOT}` expands to the
checkout). A case whose `patch` has a top-level `backend` key (the pre-M5 shape) is refused by every reader.
`tests/test_contracts.py`, `tests/test_cere_contracts.py`, `crates/cere-runtime/tests/config_rev2.rs` and
`config_federation.rs` (the `fed_`/`operator_` rows) must refuse every `expect: "invalid"` row, accept every
`expect: "valid"` row and accept every shipped TOML.

Schema revision 1 fails closed. Unknown keys, wrong types (a float or string where an integer is required, an
integer where a boolean is required), out-of-range numbers and any failed semantic rule reject the whole file.
Nothing is clamped or defaulted.

Optional keys added after revision 2 shipped (`deployments.load_endpoint`, `deployments.replicas`, `api.resume_*`,
`admission.aging_ms`, `[[auth.scopes]] weight`, `[[auth.scopes]] allowed_tiers`, `server.tier`, `ledger.orphan_load_poll_ms`, `ledger.evict_terminal_when_full`,
`scheduler.worker_lost_after_ms`, `[usage]`, `admission.batch_floor`, `admission.owner_wait_cap_ms`, `[[auth.scopes]]`
`token_envs`/`class`/`per_scope_inflight`/`max_sessions_per_scope`/`max_batch_inflight`, `[[federation.workers]]`
`owner`/`tier`/`owner_only`)
have documented defaults when absent and are then left out of the effective config, so an existing file keeps its
behaviour and its config hash.

## Shipped profiles

| File | Kind | Purpose |
|---|---|---|
| `configs/runtime/dev-single-gpu.toml` | `unconfigured`, `planning_only = true` | Planning profile; contacts nothing. |
| `configs/runtime/dev-two-replica.toml` | `unconfigured`, `planning_only = true` | Planning profile for two replicas. |
| `configs/runtime/sglang-t4-qwen38.toml` | `sglang_http`, `planning_only = false` | Live M1 profile in front of the operator's shared SGLang server `sglang-t4`. |
| `configs/runtime/sglang-bench-g0.toml` | `sglang_http`, `planning_only = false` | Benchmark profile in front of `cere-bench-g0`, the dedicated SGLang benchmark instance on GPU0. |
| `configs/runtime/multimodel-dev.toml` | two `sglang_http` entries, `planning_only = false` | M5 mock profile (bind `127.0.0.1:18396`): `Qwen3.8-Flash-Next` on mock `gpu0` plus remote `worker-a`, `gemma-4-e4b-it` on mock `gpu1`; used by `multimodel_live.py --backend mock`. |
| `configs/runtime/multimodel-g0g1.toml` | two `sglang_http` entries, `planning_only = false` | M5 real profile (bind `127.0.0.1:18397`, no federation): Qwen on `cere-bench-g0` (GPU0), Gemma on `cere-gemma-g1` (GPU1); used by `multimodel_live.py --backend real`. |

The live profile's `ledger.state_dir` is `/var/lib/cere/sglang-t4`. Smoke tests use a copy with a temporary
state directory. `RuntimeConfig::load` finds the repository root from the config file's location, so such a
copy must live inside the checkout (for example under the gitignored `bench/results/`), or the caller passes the
root explicitly to `RuntimeConfig::from_toml_str(text, repo_root)`.

The benchmark profile binds `127.0.0.1:18391`, talks to `http://127.0.0.1:18308`, keeps its ledger in
`/var/lib/cere/bench-g0` (so it runs beside the live profile) and pins the bench server's probed
`/get_server_info`, including `enable_cache_report = true`. It serves two scopes, `bench-a`
(`CERE_TOKEN_BENCH_A`) and `bench-b` (`CERE_TOKEN_BENCH_B`), with `prefix_cache_isolation = "cache_salt"` and
`allow_cross_scope_prefix_cache = false`. Every admission knob is explicit:
`max_backend_inflight = 4`, `per_scope_inflight = 4`, `queue_depth = 16`; the decode rows are the profile's
`decode_batch_max`. `shared_backend_tenancy = "exclusive"`: only CERE benchmarks use that server.

## Sections

### Top level

- `schema_version = "2"`.
- `planning_only`: `false` only for a live backend kind (`sglang_http`). A live kind requires `false`.

### `[server]`

- `bind`: `127.0.0.1:<port>`, port 1..65535. Loopback only.
- `tier` (optional, `"own"`, `"team"` or `"provider"`, default `"own"`): this install's trust tier for its owner
  (net-teleport design note 4.1). State teleport and team-pool placement compare it with the person scope's
  `allowed_tiers`.

### `[[deployments]]`

One entry per served model (M5 D1, m5 design note): 1..16 entries; the request's `model` selects the entry
whose `served_model_name` equals it. A pre-M5 `[backend]` table is an unknown key (no alias). Every entry has the
keys below (the old `[backend]` keys); messages about an entry field name `deployments[<i>].<field>` (0-based);
`expect.*`, uniqueness and deployment-file messages do not name the index.

Multi-deployment rules (D2; one entry keeps the single-deployment rules unchanged):

- (b) With more than one entry every entry is `sglang_http` with `served_model_name`, `deployment` and `expect`;
  `base_url` and an entry-level `load_endpoint` are refused (use `replicas`, `load_endpoint` per replica);
  `replicas` is optional (an entry may be remote-only), but at least one entry has a local replica.
- (c) `served_model_name` and `deployment` are unique across entries.
- (d) Replica devices are disjoint across entries and their union equals `scheduler.devices`.
- (e) Every `[[federation.workers]]` entry belongs to the entry whose `served_model_name` equals its
  `expect.served_model_name` (also with one entry).
- (f) Every entry has at least one worker (local replica or remote).
- (g) Effective `shared_backend_tenancy`, `allow_cross_scope_prefix_cache` and `prefix_cache_isolation`
  (absent = `none`) are equal across entries.
- (h) `native_cuda` and the planning kinds (`unconfigured`, `cpu_reference`) allow exactly one entry.
- (i) Each entry's deployment files are checked against that entry (profile, lock, context, cap).
- (j) `admission.max_backend_inflight` ≤ the sum over entries of (local + remote workers of the entry) ×
  min(`expect.max_running_requests`, every `max_inflight` of the entry's remote workers).

`cered --check-config` prints `config ok <hash>` then one
`deployment served_model_name=<n> recipe_id=<r> lock_sha256=<h> profile_id=<p>` line per entry with a pinned
deployment, in config order.

- `mode`: `backend_managed` | `native`. `sglang_http` requires `backend_managed`.
- `kind`: `unconfigured` | `cpu_reference` | `sglang_http` | `native_cuda` (M2.2b; one entry, D2(h)). `native_cuda`
  requires the `[deployments.native]` table (library path and SHA-256, artifact directory, expected manifest
  SHA-256, device UUID, context limit, sidecar; `configs/runtime/native-g0.toml`) and a native deployment lock (its
  `artifact_sha256` is the prepared artifact's manifest SHA-256, [ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md#identity));
  request behaviour: [API.md](API.md). M2.11 adds optional keys to `[deployments.native]`, each absent by default
  (absent keeps the config hash): `snapshot_host_bytes` (host-RAM snapshot arena, 0; above 0 exactly when the
  profile declares `native_state_snapshot`), `preempt` (false; needs the arena and `[state] host_spill_enabled`;
  since SE-09 also with `decode_batch_max` > 1: a row leaves its batch by its own cancel and a restored session joins
  the next round), `preempt_min_tokens` (1..16777216, default 512), `max_parked_sessions` (0;
  with `preempt`, `admission.max_backend_inflight` may exceed `decode_batch_max` by it, and it times the largest
  snapshot at `max_context_tokens` (with the MTP rows of a drafting profile), rounded up to the arena's 256-B
  alignment, must fit the arena), `preempt_every` (0; a dev knob only the `<profile id>-snapshot` profile with
  `preempt = true` admits: `cered` preempts the oldest running generation after every N committed tokens, with or
  without a competing request) and `reserve_output_tokens` (0 = off; sessions design note SE-09 optimistic
  reservation: a multiple of 64, needs `preempt` and `max_parked_sessions` ≥ `admission.max_backend_inflight`; a
  generation's first session holds `round64(prompt + reserve_output_tokens)` tokens instead of its whole budget, and
  one that uses it up parks itself and resumes in a session of its whole budget, counted as `native_grows` in
  `/v1/cere/stats`; output unchanged).
  The shipped `native-g0.toml` sets none of them. Operator requirement: the arena is page-aligned anonymous memory,
  zeroed, `mlock`'d and advised `MADV_DONTDUMP` + `MADV_DONTFORK` (snapshot bytes never reach disk or swap), so the
  process's `RLIMIT_MEMLOCK` must cover `snapshot_host_bytes`; otherwise ALLOCATE fails closed with `CAPACITY` and
  an error naming `RLIMIT_MEMLOCK` and the bytes needed. The live scripts run such a container with
  `--ulimit memlock=<arena + 64 MiB>` (soft = hard; [RUNBOOK.md](RUNBOOK.md)); arena 0 keeps the default.
- `[deployments.native].kv_pool_tokens` (optional, M2.5): the paged KV pool in tokens, passed as
  `cere_model_desc.kv_pool_tokens`. Absent or 0 is the derived pool: the plan's rows (the profile's `decode_batch_max`)
  × `max_context_tokens` rounded up to 64-token pages (4 × 200,000 = 800,000 at `native-g0`, past the GPU0 floor:
  needs-human, ledger M1/CR1). A non-zero value must be a multiple of 64 and at least `max_context_tokens`
  (`crates/cere-runtime/src/config.rs:2050-2063`); the upper bound is the executor's support table (200,000,
  `executor/cuda/src/host/support.cpp:69`), so a larger pool passes config validation and is refused at model create
  (not ready, worker problem `load`). An absent key leaves the config hash unchanged. Readiness requires the executor's
  reported `kv_pool_tokens` to equal the configured or derived pool (`crates/cere-runtime/src/readiness.rs:838-843`).
  `native-g0` sets 200,000, its context, shared by its 16 rows (ledger CR1; 4 before m23 design note P8): each
  session reserves ceil(min(`max_context_tokens`, round64(prompt + `max_tokens`)) / 64) pages at create, and a
  request whose create is short of pages waits in `cered` (counting against its deadline) until a session is released.
- `[deployments.native].sidecar_deadline_ms` (optional, audit N1): the longest one tokenizer-sidecar call (the request
  write, then the reply; an image attachment frame gets its own wait) may take before `cered` kills the child. Absent
  or 0 is 120,000 (`native::sidecar::DEFAULT_IO_DEADLINE`); otherwise 1,000 to 3,600,000. A child that stops
  answering without exiting used to pin one blocking thread per request and freeze `/readyz` at its last value;
  now the call fails `Gone` (503 `not_ready`) and the child is killed. A lost child (deadline kill or exit) is
  respawned (ops-hardening design note OH-01): at once after a served call, while `/readyz` lists an advisory
  `sidecar` problem ("respawning"; the worker stays ready and requests wait for the new child), else after a bounded
  backoff (250 ms doubling to 30 s) while `sidecar` makes the worker not ready. A call whose child exits under it
  runs once more on the successor started at once. Every health change republishes `/readyz` at once; `/readyz`
  never waits for a call in flight. An absent key leaves the config hash unchanged; restart class.
- `[deployments.native].executor_memory_limit_bytes` (optional, ledger M1): the executor's device-byte limit; absent =
  80 GiB. A whole MiB, at least 80 GiB and at most 86,432 MiB (m23 design note P8: GPU0 total 97,887 MiB - the
  desktop baseline L0 1,215 MiB - the invariant 10,240 MiB free floor, `readiness::NATIVE_MEMORY_CEILING_BYTES`);
  a larger value is refused at config load, so `cered` never becomes ready on it. The live gates still check GPU0
  free >= 10,240 MiB at the measured peak.
- `[deployments.native].decode_rounds_per_prefill_chunk` (optional, opportunity O79, m210 design note PF1): decode
  rounds the executor runs after each prefill chunk while decode rows are ready, 1..64
  (`CERE_MAX_DECODE_ROUNDS_PER_CHUNK`); absent = 1, today's alternation of one prefill chunk and one decode round,
  and the config hash is unchanged. Set at load through `cere_executor_set_decode_rounds` right after the executor is
  created (restart to change it); a library without that export refuses the load only when the key is set. Output is
  bit-identical (row invariance); a higher value trades a prefill's TTFT for the running rows' tok/s.
- `[deployments.native].prefill_slice_ms` (optional, MixedPD, mixed-pd design note): layer-sliced prefill, 1..10000
  (`CERE_MAX_PREFILL_SLICE_MS`); a prefill call pauses at a layer boundary once its segment has run about this many
  ms while decode rows are ready, so decode rounds run between the call's layer segments instead of after the whole
  chunk. Absent = every call runs whole, and the config hash is unchanged. Set at load through
  `cere_executor_set_prefill_slice` right after the executor is created, before the model (restart to change it); a
  library without that export refuses the load only when the key is set. Output is bit-identical.
- `[deployments.native].decode_share_pct` (optional, S1 G4, beat-sglang design note): decode share, 1..1000
  (`CERE_MAX_DECODE_SHARE_PCT`); while two or more prompts wait they prefill back to back, and with one left the decode
  rows get this percent of its prefill wall time (it replaces `decode_rounds_per_prefill_chunk`'s count). Absent = the
  decode rounds per chunk stay, and the config hash is unchanged. Set at load through `cere_executor_set_decode_share`
  right after the executor is created (restart to change it); a library without that export refuses the load only
  when the key is set. Output is bit-identical (only the order of rounds changes).
- `[deployments.native].prefill_age_calls` (optional, BC4 (b), batch-concurrency design note): prefill aging cap,
  1..1024 (`CERE_MAX_PREFILL_AGE_CALLS`); without a decode share, prefill rounds take the prompt with the fewest tokens
  left, and one overtaken this many times by younger prompts' prefill calls runs next. Absent = the call is not made
  and the executor default 8 (`CERE_DEFAULT_PREFILL_AGE_CALLS`) stays, and the config hash is unchanged. Set at load
  through `cere_executor_set_prefill_age` right after the executor is created (restart to change it); a library
  without that export refuses the load only when the key is set. Output is bit-identical (only the order of calls
  changes).
- `[deployments.native].host_share_dir` (optional, opportunity O47, build-loop-ple design note):
  an absolute host tmpfs directory without commas (`/dev/shm/cere-host`) where the executor keeps the artifact's
  HOST_PINNED region (PLE table, scale and index, 51,200,246,042 B for the A4 artifact) as one shared segment
  `cere-host-<sha256 of layout tag, artifact sha256 and region description>` that every native process of that
  artifact on the host maps and pins with `cudaHostRegister`, so a second lane costs its private memory only.
  The registration path follows the device's `cudaDevAttrHostRegisterReadOnlySupported` (never a caught error):
  `read_only` (a `PROT_READ` mapping registered `cudaHostRegisterReadOnly`) where it is 1, `rw_mprotect` (an
  `O_RDWR` `MAP_SHARED` mapping registered Mapped|Portable, then `mprotect(PROT_READ)`) where it is 0 (both GPUs on
  this host); the process prints `cere: host_share_registration=read_only|rw_mprotect` on stderr.
  Readiness passes it as the device-id options `,host_share=<dir>,artifact=<expected_manifest_sha256>`
  (`cere-native` takes the same through `--device-id`). The first process creates, fills and publishes the segment;
  later ones check its header at ALLOCATE and verify its content digest before the seal (in the background
  during the upload, m216 design note L4c), take the weights files whose tensors are all HOST_PINNED (the PLE
  files) from the segment instead of reading them, checking sha256(header prefix ‖ segment bytes) of each against
  the manifest (`cere_model_host_resident`, m216 design note L4), and compare every other put byte with it.
  Any share failure (directory,
  digest, put mismatch, register) fails ALLOCATE, the put or the seal closed; there is no fallback to a private copy. Absent
  keeps the private block and the config hash; no shipped profile sets it yet. Every lane container mounts the same
  host directory (`-v /dev/shm/cere-host:/dev/shm/cere-host`); the creator's memory cgroup is charged for the
  segment, so the first lane to start must be the one without a memory limit below the region
  ([RUNBOOK.md](RUNBOOK.md) "Shared PLE host segment").
- Native decode graph (m23 design note P1, PI-02): the `[deployments.native.levers] decode_step` lever,
  `"eager"` (default), `"graph"` or `"spec_graph"`. `graph` runs the greedy, unconstrained, text-only decode step (K7 or P3b MoE,
  M23P1-10) as a CUDA graph, bit-identical to the eager step; VERIFY, draft-extend and MTP steps stay eager.
  `spec_graph` (M29S2-18b) also replays VERIFY and the MTP draft-extend/draft steps from CUDA graphs captured per
  shape (greedy rounds without taps/logprobs, every row position below 2,048; others eager), bit-identical. P1's
  former device-id option `,decode_graph=on|off` is an unknown option (`CERE_ERR_ARGUMENT`). The lever is part of the
  config hash and hot (dev profiles: `POST /v1/cere/levers`).
- `[deployments.native].keep_host_share` (optional, default false, m216 design note L4; requires
  `host_share_dir`): the device-id option `,host_share_keep=1`, so the published segment outlives this process when
  it is the last mapper and a restart attaches it instead of reading and pinning the PLE again. False is not
  serialized, so configs without it keep their `config_hash`. Removal is the operator's ([RUNBOOK.md](RUNBOOK.md)
  "Warm restart and the segment lifecycle").
- `[[deployments.native.replicas]]` (optional, m32n design note D1/D2): exactly two entries, each exactly
  `device` (one of `scheduler.devices`; together they are exactly `scheduler.devices`) and `device_uuid`
  (`GPU-<8-4-4-4-12 lowercase hex>`, distinct). Refused together with `native.device_uuid` and with
  `lever_hot_swap`; `CUDA_VISIBLE_DEVICES` must list the UUIDs in replica order. Each replica is one worker in
  this `cered` (its own readiness, SMOKE, epoch, KV pool of `kv_pool_tokens`); `admission.max_backend_inflight`
  may reach replicas x the profile's rows. Absent (the single form) keeps every existing `config_hash`. Example:
  `configs/runtime/native-2gpu.toml`. Both keys are restart-class.
- `[deployments.native].spec_drafts` (optional, m29l design note D6): which drafts a speculating ticket may
  verify. `"mtp"` (absent, not serialized, so configs keep their `config_hash`) means the MTP proposals only.
  `"lookup"` means suffix-lookup drafts only (a measurement arm). `"mtp+lookup"` lets each request's draft policy
  pick MTP, lookup or none per round. cered sets `cere_work.flags` bits 2/3 (`CERE_WORK_LOOKUP_DRAFTS`,
  `CERE_WORK_NO_MTP_DRAFTS`) on every ticket that may speculate. The lookup values need
  `speculative_draft_tokens = 4`; anything else is refused at load. VERIFY is unchanged, so greedy output stays
  bit-identical to plain.
- `[deployments.native.levers]` (optional table, m29a design note A1): the swappable algorithms of a native
  deployment. An absent table or key means the lever's default, and defaults are not serialized, so configs keep their
  `config_hash`. Unknown keys are refused.
  - `draft_policy`: `"fixed"` (default) verifies all `speculative_draft_tokens - 1` pending MTP proposals every round.
    `"adaptive"` lets each request pick its draft count per round from its own acceptance and a fixed calibrated cost
    table (`cere_work.flags` bit 4, `CERE_WORK_ADAPTIVE_DRAFTS`); with `spec_drafts = "mtp+lookup"` it also picks the
    source. It needs `speculative_draft_tokens = 4`. Greedy output stays bit-identical to plain either way. The table
    is timed at model seal only as deep as the load's `drafter` can pick (LoadCrit LC-01,
    `cere_levers.calibrate_windows`): `"mtp"`, `"lookup"` and `"mtp+lookup"` time windows 1..4 (about 1.5 s less load
    [INFERENCE until the LC-01 rows]); the tail, long and chain drafters time every window. A hot swap to a deeper
    drafter keeps the load's table.
  - `verify_attention`: `"per_row"` (default) runs each verify row's full-attention core as its own one-token call;
    `"batched"` runs one K6 prologue and one `k6::dense_attention_rows` launch over a chain's drafts + 1 verify rows
    per full-attention layer (`cere_levers.verify_attention`); a chain whose every row is past 2,048 tokens (QSA plans)
    runs K9's index query, top-512 selection and sparse attention over up to 16 rows per call instead (eager only,
    long-ctx design note LK-01). Bit-identical to `per_row`. Hot (dev profiles:
    `POST /v1/cere/levers`).
  - `k2_weights` (Fp8W): `"bf16"` (default) keeps the checkpoint's BF16 dense K2 weights; `"fp8"` makes the executor
    quantize the attention qkv/o/indexer, GDN in/out projections, shared expert and lm_head in place at model create
    to E4M3 with one FP32 scale per output row (`cere_levers.k2_weights`). Numerics-changing (the registry lists it
    for the D9Q screen) and restart-only: the dev hot swap refuses a change with 409 `lever_restart_only`. The opt-in
    profile `configs/runtime/native-g0-spec-fp8w.toml` (`native-g0-qwen38-flash-next-v20`) is the shipped speculation
    profile `-v19` plus `k2_weights = "fp8"`; the default stays `-v19`. The opt-in FP4-Local S3 profile
    `configs/runtime/native-g0-spec-fp4local-s3.toml` (`-v21`) has `-v19`'s levers and serves the S3 artifact (artifact
    profile `rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1`); the weight formats come from the artifact, no lever selects them.
  - `k2_gemm` (K2Prefill P3 value): `"small_m_a8"` (`cere_levers.k2_gemm` 2) runs `"small_m"` for every call of
    m <= 16 and W8A8 MXFP8 GEMMs (E4M3 activations, UE8M0 per 32 k) at m > 16 on MXFP8 weights; a BF16 weight runs
    small_m's path. Numerics-changing (D9 batch), hot; it prices adaptive rounds with small_m's table. Opt-in only:
    `configs/runtime/native-g0-spec-v56.toml` (`native-g0-qwen38-flash-next-v56`, BeatSglang S7) is `-v29` plus
    `k2_gemm = "small_m_a8"` and `prefill_attention = "p4"`.
  - Registry v2 (SelfTune ST2, D3): the table is checked against `crates/cere-runtime/src/levers.rs` `REGISTRY`
    (mirrored by `tests/fixtures/lever_registry.json`). Every selected implementation, defaults included, must be
    proven on the arch of each bound device (`hip:<ordinal>` is the AMD port's `gfx1151`, `vk:<ordinal>` the Vulkan
    executor's `vulkan`, a `GPU-` UUID `sm_120a`),
    get what it `requires` (`speculative_draft_tokens = 4` for the speculation levers), and cover the profile's
    `decode_batch_max` and `max_context_tokens`. Typed params are not config keys: `build` params are fixed by the
    library, and `launch` params are set only by the dev hot swap below. The implementation contract is in
    [EXECUTION_PLUGIN.md](EXECUTION_PLUGIN.md) "Slots, implementations and the model descriptor".
- `[deployments.native].device_uuid` forms (single form; refused together with `replicas`, which take `GPU-` UUIDs
  only): `GPU-<8-4-4-4-12 lowercase hex>` binds a CUDA card (`CUDA_VISIBLE_DEVICES`, lever arch `sm_120a`);
  `hip:<ordinal>` an AMD GPU through the AMD HIP port (`HIP_VISIBLE_DEVICES`, arch `gfx1151`);
  `vk:<ordinal>` an AMD GPU through the Vulkan executor (`executor/vulkan/`, `libcere_executor_vulkan.so`, arch
  `vulkan`; vulkan-amd design note D2). An ordinal is decimal, 1-3 digits, no leading zero (`vk:0`, `vk:12`; not
  `vk:00`, `vk:x`, `VK:0`). `kind` stays `native_cuda` for all three. For `vk:<ordinal>` the ordinal is the executor's
  device id (Vulkan physical device `<ordinal>`); neither `CUDA_VISIBLE_DEVICES` nor `HIP_VISIBLE_DEVICES` is read,
  and `cered` does not check that the device exists (the library's create does). A Vulkan deployment must pin
  `moe_gemm = "k7"` (the default `p3b` is not proven on `vulkan`) and may select only the eager set the registry lists
  for `vulkan`; `small_m`, `small_m_a8`, `graph`, `spec_graph`, `batched`, `fp8` and the lookup / adaptive /
  `draft_tokens` 2-3 / `grammar_drafts = "on"` values are refused at config validation by name. `prefill_attention = "p4"` is selectable on `vulkan` and is the FP16 cooperative-matrix prefill attention for calls of 17 or more rows: it changes prefill numerics within the kernel contract bounds (`numerics: changing`; `cuda_core`, the FP32 kernels, stays the registry default and stays available), and it needs `VK_KHR_cooperative_matrix` plus a seal-time probe, which the device refuses otherwise. On `gfx1151` (the HIP port) `p4` stays refused. See an internal benchmark report.
  The default Vulkan profiles select `moe_gemm = "k7"`, `k2_gemm = "tail_split"` and `prefill_attention = "p4"`
  (every other lever at its registry default): `configs/runtime/native-amd-s3-vk.toml`
  (`native-amd-qwen38-flash-next-vk-v62`, no speculation) and `configs/runtime/native-amd-s3-vk-spec.toml`
  (`...-vk-spec-v63`, MTP, 4 drafts). `k2_gemm = "tail_split"` (Vulkan only, `numerics: changing`; every other
  library refuses it at create) runs the K2 GEMM's (m > 16) rows whose few values are inexact as FP16 through the
  matrix unit plus an exact FP32 chain over those values instead of the whole row on the FP32 route. Both
  numerics-changing levers are on in `-v62` / `-v63` since their reduced AMD screens passed (vulkan-amd design note
  D8, amending D7; `tail_split` and `p4`, 2026-11-21). AMD deployments default to the Vulkan library (D5; the HIP profiles `-v60` /
  `-v61` stay available). All are written by
  `scripts/live/amd/make_native_profile.py --backend vulkan` and run by `scripts/live/amd/run_native_vk.sh`
  ([RUNBOOK.md](RUNBOOK.md) "Vulkan executor on AMD"). Limits, as measured and declared (vulkan-amd design note
  "P3 result and gate G3", "D15 result"): one decode row, 32,768 tokens, text only; the device-tier prefix cache
  (`prefix_state_slots`, split prefill bit-identical to one call since D15, tested at the ABI only, set by no shipped
  profile; refused with speculative drafts; the host tier not ported); measured only on AMD's Radeon 8060S with Mesa
  RADV 25.2.8; no other GPU or Mesa has run it.
- `[deployments.native].lever_hot_swap` (optional, default false, m23 design note PI-04): admits
  `POST /v1/cere/levers` (operator token) on a dev/bench copy of a profile. No committed profile sets it, and it is
  refused with `replicas` and with `family = "sdar"`. The body is `{"levers": {lever: impl, ...}}` (a partial set
  over the active one) with an optional `"params": {lever: {param: unsigned integer, ...}}` (SelfTune D6, O292 D-9).
  Each param is checked against the lever's next implementation: a `launch` param inside its registry range is
  accepted, and a `build` param is refused with 409 `lever_build_param`. Other refusals change nothing: 403
  `lever_hot_swap_disabled`, 409 `requests_in_flight`, 409 `lever_restart_only`, and 400 `invalid_lever` (a malformed
  body, an unknown lever, impl or param, or an out-of-range value). False is not serialized.
- `[deployments.native].family` (optional, SelfTune ST4 S3a): the model family the library is built for (its
  descriptor's `model_type`). `"qwen4_exp"` is the default; it is not serialized, so configs keep their `config_hash`.
  `"sdar"` is DICE-8B (`configs/runtime/native-g0-dice8b.toml`). On `sdar`, every lever must be its default, and the
  config refuses by name `speculative_draft_tokens` other than 0, `prefix_state_slots`, `snapshot_host_bytes` (or a
  snapshot profile), `preempt`, `lora`, `research`, `lever_hot_swap`, a `hip:<ordinal>` or `vk:<ordinal>` device, and a profile
  `decode_batch_max` above 1.
- `[deployments.native.ab]` (optional, ab-split design note AB-01..AB-06): a live A/B traffic split of the
  runtime-layer levers, for dev/experiments; no shipped profile has it. Absent means no split, and an absent section is
  not serialized, so configs keep their `config_hash` (a config with it has another hash, which names the experiment).

  ```toml
  [deployments.native.ab]
  salt = "draft-policy-1"            # identifier; keys the session hash (a label, not a secret)
  [[deployments.native.ab.arms]]
  id = "fixed"                       # identifier, unique; the only name the stats report
  weight = 1                         # 0..=1_000_000; the share of new sessions is weight / sum
  [[deployments.native.ab.arms]]
  id = "adaptive"
  weight = 1
  levers = { draft_policy = "adaptive" }   # over the profile's [deployments.native.levers]; absent = profile's
  ```

  - `arms`: 1..=16. Each arm's `levers` names runtime-layer levers only (`drafter`, `draft_policy`) at
    `bit_identical` implementations, so every arm's greedy output equals plain decoding. Refused at load (Rust
    `validate_ab`, Python `cere_contracts.check_ab`): an unknown lever or implementation, an executor-layer lever
    (process-global, switched only between requests), a `changing` implementation, a speculation lever other than its
    default without `speculative_draft_tokens = 4`, weights summing to 0, a duplicate arm id, and unknown keys.
  - Assignment: `HMAC-SHA256(salt, scope + X-Cere-Session-Id)` for a session turn, else of the request id, mapped
    onto the weights. A session keeps its arm while that arm is configured; a new salt reshuffles every session.
  - Every key is `live` (docs/RUNBOOK.md "Live config reload"): a reload reaches new sessions; existing sessions keep
    their arm until it is removed.
- `[deployments.native.research]` (optional, off by default; docs/RESEARCH_HOOKS.md): opt-in capture and
  intervention plans (docs/API.md "Research (WeightLab)"). Only the dev research profile
  `configs/runtime/native-g0-spec-research.toml` (`native-g0-qwen38-flash-next-v57`: `-v50` plus this table,
  `api_compat_anthropic = true`, `[api] default_deadline_ms = 900000` and the arm scopes `wl-a`/`wl-b`/`wl-p`) has
  it; absent means no research routes and today's code paths, and is not serialized, so configs keep their
  `config_hash`. All five keys are required when the table is present; unknown keys are refused; every key is
  `restart` (the staging is reserved at model load).

  ```toml
  [deployments.native.research]
  device_staging_bytes = 67108864           # 1 MiB..=1 GiB: device capture staging, reserved at model create, counted in fits()
  host_buffer_bytes = 1073741824            # 1 MiB..=16 GiB: pinned host capture ring, all requests together
  max_capture_bytes_per_request = 268435456 # 4096..=host_buffer_bytes: a plan whose bound exceeds it is refused
  max_plans = 8                             # 1..=64 registered plans
  max_vectors = 64                          # 1..=1024 uploaded vectors
  ```

  Isolation: with the table, cered refuses to start when `[server] bind` uses port 18080 (the production port
  constant), when any deployment `device_uuid` equals the `uuid` in the production marker file
  (`CERE_GPU1_PRODUCTION_MARKER`, default `/var/lib/cere/gpu1.production`; a missing marker skips this check), or
  when `[auth] operator_token_env` is unset.
- `[deployments.native.loop_guard]` (optional, loop-guard design note LG-03): the native repetition guard. No
  shipped profile has it; absent means off and is not serialized, so configs keep their `config_hash`. Present turns
  both detectors on; every key is optional (defaults shown), and a detector's 0 turns it off.

  ```toml
  [deployments.native.loop_guard]
  min_period_bytes = 16        # LG-D1: shortest repeating unit (1..=4096); shorter periods (=====, 0, 0, 0) never count
  max_period_bytes = 1024      # LG-D1: longest unit checked (1..=4096, >= min_period_bytes)
  min_repeats = 4              # LG-D1: exact copies ending the output (0 = off, else 2..=64)
  min_span_bytes = 512         # LG-D1: bytes the copies cover together (0..=65536)
  tool_repeat_run = 10         # LG-D2: identical tool call -> result pairs ending the conversation (0 = off, else 2..=10000)
  ignore_arg_keys = ["description"]   # LG-D2: top-level argument keys the call comparison skips (<= 16, 1..=64 chars)
  ```

  - LG-D1: after each committed token, when the output text (reasoning and tool markup included) ends with at least
    `min_repeats` exact copies of a unit of `min_period_bytes..=max_period_bytes` bytes covering `min_span_bytes`,
    the output finishes on that token with `finish_reason` `repetition` (`matched_stop` null; a success, like
    `stop`; Anthropic `stop_reason` `end_turn`). The text through that token is kept and the generation is
    cancelled.
  - LG-D2: a request whose messages end in `tool_repeat_run` identical (assistant `tool_calls`, `tool` results)
    pairs is refused `400 invalid_request_error` code `loop_detected` (param `messages`) before any tokenizer or
    executor work. Calls compare by name and parsed arguments without `ignore_arg_keys`; call ids and prose are not
    compared.
  - Refused at load (Rust `validate_loop_guard`, Python `cere_contracts.check_loop_guard`, schema): both detectors
    off, `min_period_bytes > max_period_bytes` or `min_repeats x max_period_bytes > 65536` with LG-D1 on, values out
    of range, and unknown keys. Every key is `restart` (docs/RUNBOOK.md "Live config reload").
- `profile_id`: capability profile id. For `sglang_http` it must equal the deployment profile's `profile_id`.
- `shared_backend_tenancy`: `exclusive` | `shared_acknowledged`. The live profile sets `shared_acknowledged`
  because `sglang-t4` (and its radix prefix cache) also serves non-CERE clients.
- `allow_cross_scope_prefix_cache`: on `sglang_http`, more than one auth scope requires `true` or
  `prefix_cache_isolation = "cache_salt"`, because the backend's prefix cache is shared across scopes.
- `prefix_cache_isolation` (optional): `none` | `cache_salt`; absent means `none`, and an absent key leaves
  the config hash unchanged. With `cache_salt`, every backend chat request carries
  `"cache_salt": hex(HMAC-SHA256(server key, "cere-cache-salt\0" || scope_id))[..32]`, so scopes cannot hit
  each other's cached prefixes. Clients cannot send `cache_salt` themselves (unknown field).
- `api_compat_anthropic`, `api_compat_responses` (optional booleans, default `false`; M2.14, m214 design note
  D3/D4): enable the Anthropic `POST /v1/messages` + `POST /v1/messages/count_tokens` and the stateless OpenAI
  `POST /v1/responses` (+ `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`) surfaces for this entry. A route
  is registered only when some entry sets its key; a request whose `model` selects an entry without it is
  `404 not_found`. No shipped config sets either (exposure is needs-human N6). See [API.md](API.md).
- `load_endpoint` (optional, `sglang_http` only): the backend's load path, `/` followed by up to 127 of
  `[A-Za-z0-9_./-]` (the benchmark profile sets `"/get_load"`). Set, `cered` polls it while recovered orphans
  hold slots and releases an orphan early when the load shows no work beyond `cered`'s own confirmed streams (see
  [RUNBOOK.md](RUNBOOK.md#crash-recovery-and-orphaned-backend-requests)). Absent: orphans are held for the
  retention bound only. The live `sglang-t4` profile leaves it absent. Refused together with
  `deployments.replicas` or with more than one entry: there it is set per replica.
- `sglang_http` only (and required there; forbidden for other kinds), exactly one of `base_url` and
  `[[deployments.replicas]]` (below):
  - `base_url`: exactly `http://<host>:<port>`. The host is a literal IPv4 address in `127.0.0.0/8` written in
    canonical dotted-quad form (no leading zeros) or `[::1]`. The port is explicit, 1..65535, without leading
    zeros. No hostname (`localhost` is rejected), userinfo, path, query or fragment. **A trailing `/` is
    rejected**: `base_url` is an origin, and clients append request paths to it.
  - `served_model_name`: the model name the backend serves (`Qwen3.8-Flash-Next`).
  - `deployment`: repo-relative prefix `configs/deployments/<name>` of the deployment identity files (below).
  - `[deployments.expect]`: the pinned `GET /get_server_info` values. Keys use SGLang's names: `version`,
    `model_path`, `tokenizer_path`, `chat_template`, `quantization`, `kv_cache_dtype`, `speculative_algorithm`,
    `speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens`, `reasoning_parser`,
    `tool_call_parser`, `allow_auto_truncate` (must be `false`), `enable_cache_report`, `context_length`,
    `max_running_requests`. TOML has no null, so an **omitted nullable key pins `null`**: the backend must report
    `null` for each of `chat_template` (the model's bundled template), `quantization`,
    `speculative_algorithm`, `speculative_num_steps`, `speculative_eagle_topk` and `speculative_num_draft_tokens`;
    a reported value fails readiness naming the key, and so does a nullable key missing from `/get_server_info` (`readiness.rs:183-194,379-388`; `scripts/live/_common.py` `pin_drift` applies the same rule). A present `chat_template` must be a non-empty string; present
    values of the others follow the same rules as before. The four `speculative_*` keys are **all present or all
    absent**; a partial set is a config error naming the missing keys. `image_digest` (`sha256:<hex>`) is recorded
    evidence of the container image; the backend does not report it.

- `[[deployments.replicas]]` (optional, `sglang_http` only, 1..64 entries; M3.1): independent SGLang servers of the
  one deployment, one worker per device. Each entry is `device`, `base_url` (same form as `deployments.base_url`) and
  an optional `load_endpoint` (same form as `deployments.load_endpoint`, for that replica). All replicas share
  `kind`, `profile_id`, `served_model_name`, `deployment`, `[deployments.expect]` and the tenancy keys. Rules:
  `deployments.base_url` and `deployments.load_endpoint` are refused with `replicas`; `device` values are unique and each
  is in `scheduler.devices`, which lists exactly the replica devices; `base_url` values are unique after
  normalisation (`127.0.0.1` and `127.000.0.1`-style spellings are already rejected, so equal host and port means
  the same server). `native_cuda` refuses `replicas` (one executor binds one device per process). Without
  `replicas` (the single form) the backend is one replica on `scheduler.devices[0]`, and `scheduler.devices` must
  then have exactly one entry. A replica's `device` is its `worker_id` in the API, readiness and logs
  ([MULTI_GPU.md](MULTI_GPU.md#implemented-m31)).

  ```toml
  [[deployments.replicas]]
  device = "gpu0"
  base_url = "http://127.0.0.1:18308"
  load_endpoint = "/get_load"

  [[deployments.replicas]]
  device = "gpu1"
  base_url = "http://127.0.0.1:18310"
  ```

### `[model_management]`

Policy for CERE-managed model residency (v0.4): `load_on_demand`, `single_flight_loads` (must be `true`),
`unload_idle`, `idle_unload_seconds` (not below `min_resident_seconds` when `unload_idle`),
`min_resident_seconds`, `max_loaded_models_per_device`, `max_concurrent_loads`, `eviction_policy`
(`lru_priority` | `manual_only`), `drain_timeout_ms`, `preload_recipe_ids` (at most
`max_loaded_models_per_device × devices`).

`sglang_http` requires `load_on_demand = false`, `unload_idle = false`, `eviction_policy = "manual_only"` and
`preload_recipe_ids = []`: the backend owns its model and `cered` never loads, unloads or restarts it. This applies
to `cered`'s own backends only; a managed remote worker's model is loaded and unloaded by its `cere-worker` agent
(`[backend.launch]`, [below](#agent-config-cere-worker-configsworkertoml)).

### `[scheduler]`

`mode = "replica"`, `devices`. There is no rows or prefill-quantum key: a worker's decode rows cap is its
deployment profile's `decode_batch_max` alone (batch-concurrency design note BC-D5). A native deployment's
`admission.max_backend_inflight` is at most workers x (`decode_batch_max` + `max_parked_sessions` with `preempt`) and
readiness refuses an executor reporting another `decode_batch_max`; an SGLang deployment's
`expect.max_running_requests` is at most `decode_batch_max`.

`worker_lost_after_ms` (optional, 1000..3600000, default 60000; absent keeps the config hash): a worker that
stays Suspect (probe failure, pin drift, missed probes) for this long becomes Lost. A probe that matches every pin
under the worker's own `instance_id` counts as seen (its re-smoke may still wait for one of that worker's slots),
so it keeps the worker Suspect, never Lost. Each readiness smoke (its wait for a slot included) gives up after 60 s
or half of `3 × probe interval (15 s) + worker_lost_after_ms`, whichever is shorter, and the probe then counts as
seen again, so a long wait never ages the worker Lost. A Suspect worker gets no new grants either way; a Lost
worker rejoins only at a new worker epoch.

### `[state]`, `[telemetry]`, `[safety]`

`allow_cross_device_resume` must be `false`. Telemetry is `level = "metadata"` with `raw_content = false`.
Safety is fixed: fail on unknown capability; no tool execution, power changes or cloud fallback.

State teleport (NetTeleport S1, `POST /v1/cere/state/export` and `/import`, [API.md](API.md#state-teleport-netteleport-s1);
optional live keys; a config without them keeps its `config_hash`):

| Key | Range | Meaning |
|---|---|---|
| `state.max_blob_bytes` | 65,536..68,719,476,736; absent → 4,294,967,296 | The largest state image an export or import moves (`413 state_oversize` past it). Set explicitly, it also raises the dev containers' `--ulimit memlock` by the same bytes (`scripts/live/native_dev.py` `memlock_args`): the image is decrypted into one locked buffer. |
| `state.busy_wait_ms` | 0..600,000; absent → 30,000 | How long an export or import waits for an idle executor (the library refuses prefix-cache export/import while a ticket is in flight) before it refuses `409 state_busy`. |

Warm restart (NetTeleport S3, L1, [STATE_FABRIC.md](STATE_FABRIC.md); optional keys, a config without them keeps its
`config_hash` and parks nothing):

| Key | Range | Meaning |
|---|---|---|
| `state.park_on_stop` | bool; absent → false | Restart key. A graceful stop (SIGTERM, after the drain) exports the deepest cached prefix state of each recently active session and writes it sealed into `<ledger.state_dir>/parked/` (0700 directory, 0600 files, fsync'd); the next start imports every image whose library, profile, layout, compat id and tier match strictly before the backend serves, and deletes every file. Needs `deployments[0]` `native_cuda` in the single form. |
| `state.park_max_sessions` | 1..256; absent → 8 | Restart key. How many recent distinct prompts (namespace and token ids, in memory only) the recorder keeps; a stop parks at most this many. |
| `state.park_max_bytes` | 65,536..1,099,511,627,776; absent → 17,179,869,184 | Live key, read by the stop. The image bytes one stop parks at most (each image also within `state.max_blob_bytes`). |
| `state.park_deadline_ms` | 0..120,000; absent → 10,000 | Live key, read by the stop. How long the stop parks after the drain, most recent session first; no export starts after it. Budget `api.shutdown_drain_ms` + the cancel grace + this below the supervisor's stop timeout (`docker stop --time 30` on the dev lanes). |

`[[telemetry.clients]]` (optional, roadmap v0.7 A2; restart keys) labels each chat request with a client and a
workload in its log lines and `GET /metrics` ([OBSERVABILITY.md](OBSERVABILITY.md#client-and-workload-labels-roadmap-v07-a2)).
At most 16 rules, first match wins, none matches → `other`. A config without rules keeps its `config_hash`.

| Key | Range | Meaning |
|---|---|---|
| `client`, `workload` | `^[a-z0-9][a-z0-9._-]{0,31}$`, not `other` | The labels of a request the rule matches. |
| `scope` | a configured `auth.scopes[].scope_id`; optional | The rule matches requests of this scope only (every scope when absent). |
| `priority_header` | bool; optional | The rule matches requests that did (`true`) or did not (`false`) send `X-Cere-Priority` (either when absent). |

A request naming a configured client in `X-Cere-Client` takes that client's first rule whose `scope` matches, before
the derived rules.

### `[api]`

| Key | Range | Meaning |
|---|---|---|
| `max_request_body_bytes` | 1024..64 MiB; at most 8323072 with `[federation]` | Request body cap (`413 request_too_large`). Also the total byte budget of unary response bodies retained in memory for idempotent replay. With `[federation]` the cap is `REMOTE_BODY_MAX` (8 MiB SubmitWork message cap minus a 64 KiB envelope), so any admitted body fits one SubmitWork. |
| `max_messages` | 1..10000 | Messages per chat request. |
| `default_max_output_tokens` | 1..131072, ≤ `max_output_tokens` | Budget forwarded when the client sends none. |
| `max_output_tokens` | 1..131072 | Hard output cap; larger requests are rejected. |
| `default_deadline_ms` | 1..86400000, ≤ `max_deadline_ms` | Deadline when the client sends none. |
| `max_deadline_ms` | 1..86400000 | Upper bound for `X-Cere-Deadline-Ms`; larger header values are reduced to it. |
| `stream_buffer_events` | 1..65536 | Per-stream buffer bound. |
| `stall_timeout_ms` | 1..600000 | A full stream buffer past this cancels with `client_backpressure`. |
| `shutdown_drain_ms` | 0..600000 | Graceful-shutdown drain window. |
| `resume_buffer_bytes` (optional) | 4096..256 MiB, default 4 MiB | Per continue-mode stream: bytes of SSE events kept for resume; overflow drops the oldest events. |
| `resume_buffer_total_bytes` (optional) | 64 KiB..4 GiB, default 64 MiB, ≥ the effective `resume_buffer_bytes` × `admission.max_backend_inflight` | All resume buffers together; overflow releases the oldest terminal buffers first. |
| `resume_retention_ms` (optional) | 1000..3600000, default 60000 | How long a resume buffer outlives its request's terminal. |

The three resume keys only matter for streams sent with `X-Cere-On-Disconnect: continue` (see
[API.md](API.md#stream-resume)).

### `[admission]`

- `max_backend_inflight` (1..4096): global cap on requests in flight across all backend workers. On
  `sglang_http` it must be ≤ the D2(j) sum (one entry without federation: N × `deployments.expect.max_running_requests`,
  N the number of replicas, 1 in the single form); each worker's own cap is `expect.max_running_requests`, which must be ≤ the deployment profile's
  `decode_batch_max` (the validated row range). On `native_cuda` (one worker) it must be ≤ `decode_batch_max`.
  The live profile uses 2 of the backend's 4, leaving headroom for the operator's own traffic; the benchmark
  profile uses all 4. `per_scope_inflight`, `queue_depth` and the `api.resume_buffer_total_bytes` rule stay global.
- `per_scope_inflight` (1..4096): ≤ `max_backend_inflight`.
- `queue_depth` (1..10000): bounded admission queue.
- `aging_ms` (optional, 1000..3600000, default 30000): a waiter queued longer than this is served as `high`
  priority. See [ADMISSION_CONTROL.md](ADMISSION_CONTROL.md#what-shipped-m1).
- `batch_floor` (optional, 0.0..=0.5, default 0.1): team pools (net-teleport design note 4.4, C17): while both
  classes have a grantable waiter, at least this share of the grants goes to `batch`; otherwise `interactive` is
  served first.
- `owner_wait_cap_ms` (optional, 1000..600000, default 120000): team pools (Q-T6, Q-T8): a non-owner's work on an
  owned worker, and in-flight non-owner work when an `owner_only` window starts, is bounded by this: its deadline
  is at most this minus 500 ms after its grant, the 500 ms left for its abort and the slot's release.

Native rows and admission (batch-concurrency design note BC2/BC6). A `native_cuda` deployment's rows are its
profile's `decode_batch_max`, and `max_backend_inflight` is at most workers x (`decode_batch_max` +
`max_parked_sessions` with `preempt`). Production `configs/runtime/prod-t4.toml` (`-v53`) runs 4 rows, inflight 4. The
opt-in 16-row profile `configs/runtime/native-g0-rows16-v58.toml` (`-v58`) sets `max_backend_inflight = 16`,
`per_scope_inflight = 16`, `queue_depth = 16`, `kv_pool_tokens = 262144` (= its context, so one 256K request still
fits) and `prefill_slice_ms = 200`; the opt-in `-v59` (`native-g0-sessions-v59.toml`, BC4) is the 4-row `-v53` at
`prefill_slice_ms = 50`. Shortest-remaining-prompt-first prefill with aging cap 8 is the executor default on library
`24497fd0` and later (`prefill_age_calls` above), with no config key set.

### `[auth]`

`[[auth.scopes]]` entries of `scope_id` (`^[a-z0-9][a-z0-9_-]{0,63}$`) and `token_env`, the **name** of an
environment variable (`^[A-Z][A-Z0-9_]{0,63}$`). Token values never appear in config. Scope ids and env
names are unique. At startup the runtime reads each variable (`RuntimeConfig::resolve_auth`). A missing,
empty or non-UTF-8 value, or two scopes resolving to the same token, is a startup failure.

`weight` (optional, 1..100, default 1): the scope's deficit-round-robin quantum in the admission queue, in
requests (under saturation a scope of weight 3 gets three grants per round to a weight-1 scope's one).

`allowed_tiers` (optional, 1..3 distinct of `"own"`, `"team"`, `"provider"`, default `["own"]`): the trust tiers this
person's cached state and work may go to (net-teleport design note 4.1, Q-T5); `team` and `provider` are opt-in.

Team pools (net-teleport design note 4.4, S6), all optional:

- `token_envs` (1..16 env names, same pattern and uniqueness as `token_env`): further tokens of the same person scope
  (one per device, C14); `token_env` stays required and keeps working.
- `class` (`"interactive"` or `"batch"`, default `"interactive"`): the scope's scheduling class when a request sends no
  `X-Cere-Class` header (docs/API.md "Team pools").
- `per_scope_inflight` (1..4096, at most `admission.max_backend_inflight`): this scope's in-flight cap instead of
  `admission.per_scope_inflight`.
- `max_sessions_per_scope` (1..1000000): this scope's live-session cap instead of `ledger.max_sessions_per_scope`.
- `max_batch_inflight` (1..4096): at most this many of the scope's rows in flight in class `batch` (the per-person
  batch-row cap, Q-T7). No token budgets.

### `[ledger]`

- `state_dir`: absolute, normalized path (no `.`/`..`/empty components, no trailing `/`, not `/`), outside the
  repository. Rust checks against the repository root of the config file (the nearest ancestor containing
  `PACKAGE_VERSION`); Python checks against the checkout it runs from.
- `max_records`: request records held (non-terminal plus terminal within retention). A full ledger rejects new
  chat requests with `503 overloaded` before admission. It also bounds the expired-idempotency-key tombstones.
  Size it for the request rate × `terminal_retention_ms` (rejected requests take a record too): about 0.7 KiB of
  RSS per record, and compaction briefly holds a second copy of the live set. Production uses 100,000.
- `terminal_retention_ms`: the documented client retry horizon. A terminal record (status, idempotency key)
  stays for this long after its terminal; a later retry with the same key gets `idempotency_key_expired`.
  `cered` also uses it after a crash: each request of the previous process is failed `worker_restart` and
  aborted at startup, and its admission slot stays charged until `terminal_retention_ms` after its admission,
  because the backend may keep generating after the abort, unless `deployments.load_endpoint` releases it earlier
  (see [RUNBOOK.md](RUNBOOK.md)). The live profile's 3600000 (1 h) means one such orphan holds one of the 2 live
  slots for up to an hour after it was admitted.
- `max_sessions_per_scope`, `session_ttl_ms` (a session without an active writer expires this long after its
  last commit or creation), `max_revisions_per_session`.
- `orphan_load_poll_ms` (optional, 1000..600000, default 5000): period of the orphan load poll; used only with
  `deployments.load_endpoint` and only while recovered orphans hold slots.
- `evict_terminal_when_full` (optional boolean, default false): a full ledger evicts its oldest terminal record
  (earliest terminal) to admit the new request instead of rejecting it `503 overloaded` (`ledger_full`). It never
  evicts an in-flight record or a recovered orphan whose slot is still charged; with only those left the request
  is rejected as before. Eviction writes no new journal record type, so the release before it still opens the
  journal: a keyed victim is journaled as `key_expired` before the new admission, and a restart evicts unkeyed
  terminal records again, oldest first, until the ledger is within `max_records` (an unkeyed victim can therefore
  come back after a restart that finds room for it, until its retention ends). An evicted request id loses
  idempotent replay: its status answers as unknown (404) and a retry with
  its idempotency key gets `idempotency_key_expired` (the key is tombstoned as at the end of retention), so
  `terminal_retention_ms` is then a best-effort horizon under pressure. Production sets it to true. A rollback to
  a release without the key needs that release's config (it rejects the unknown key), whose other config hash
  turns every key retried across the rollback into `idempotency_conflict`. That release drops records only at the
  end of their retention: it restores the evicted records the journal still holds (none after a restart, which
  compacts), and a ledger that was full opens full, so it answers new requests `503 overloaded` until its oldest
  record expires, up to `terminal_retention_ms` (`crates/cere-runtime/tests/server_ledger_rollback.rs`).
- Files in `state_dir` (created `0700`, files `0600`): `journal.jsonl` (fsync'd request/session journal, also
  the epoch store), `server_secret` (the key material for request and session digests, created atomically on
  first start) and `lock` (exclusive `flock` held for the process lifetime; a second `cered` on the directory
  fails at startup). None holds prompts or outputs.
  With the usage journal on (`[usage]` below) it also holds `usage/` (`0700`, day files `0600`).

### `[usage]`

Optional table (net-teleport design note 4.4, S7); absent keeps the defaults and the config hash.

- `enabled` (optional boolean, default true): append one row per request terminal to
  `<state_dir>/usage/YYYY-MM-DD.jsonl` ([OBSERVABILITY.md](OBSERVABILITY.md#usage-journal-netteleport-s7)).
- `retention_days` (optional, 1..3650, default 90, Q-T9): day files older than this many days (today included)
  are deleted at start and at each new UTC day.

## Migrating from revision 1

Set `schema_version = "2"`. Move `scheduler.max_queued_requests` to `admission.queue_depth` and
`scheduler.max_output_tokens` to `api.max_output_tokens`. Add `backend.shared_backend_tenancy`,
`backend.allow_cross_scope_prefix_cache`, `[api]`, `[admission]`, `[[auth.scopes]]` and `[ledger]`.

## Deployment identity

`configs/deployments/<name>.{recipe,lock,capabilities}.json` conform to `model_recipe`, `deployment_lock` and
`execution_capabilities`. `make validate` checks every file there (an unknown file name fails) and cross-checks
each set: the lock's `recipe_sha256` is the SHA-256 of the recipe file bytes; recipe, lock and profile name
the same profile id; profile and lock share `artifact_sha256`; context lengths agree; the lock provides every
operation the recipe requires. A runtime config referencing a deployment additionally requires that the
profile's `profile_id` and `execution_mode` equal `deployments.profile_id` and `deployments.mode`, that the recipe's
`model_ref` is `local://` + `expect.model_path` without its leading `/`, that `sha256:` + the lock's
`backend_build_sha256` equals `expect.image_digest`, that the profile context equals `expect.context_length`,
that the lock provides `chat`, and that `admission.max_backend_inflight` ≤ the profile's `decode_batch_max`.
The Rust loader exposes `recipe_id`, `lock_sha256` (SHA-256 of the lock file bytes) and the profile id.

`qwen38-flash-next` pins Qwen3.8-Flash-Next as served by `sglang-t4`. The lock's digests were computed on the
host copy of the model directory (`/var/lib/cere/models/qwen38-flash-next/Qwen3.8-Flash-Next-NVFP4`, mounted as
`/models/...` in the container). Each is the SHA-256 of `sha256sum` output lines (`<hex>  <file>`) for the
listed top-level files, sorted bytewise by file name (`LC_ALL=C`, not locale order). The lock's
`tokenizer_sha256` was corrected in M2.1 to this order (`5cba89bb…405b`); the old value used locale order.

| Field | Files |
|---|---|
| `artifact_sha256` | every regular top-level file (the `.cache/` directory excluded) |
| `model_content_sha256` | `*.safetensors`, `model.safetensors.index.json`, `config.json`, `hf_quant_config.json` |
| `tokenizer_sha256` | `merges.txt`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json` |
| `template_sha256` | SHA-256 of `chat_template.jinja` itself |
| `backend_build_sha256` | the container image digest |

The capability profile is `backend_managed` with `evidence = "declared"`: SGLang owns batching, KV cache and
EAGLE speculation, and CERE has measured none of it. Rows are concurrent CERE requests (1..4, up to the
backend's `max_running_requests`); rows 3–4 are exercised only on `cere-bench-g0` (see `bench/reports`), and
`sglang-t4` stays capped at 2 by its config. Context 200000, operations `prefill`/`decode`/`cancel`, text output
only (no `json_schema`, grammar or tool calls). Decode is validated for `greedy` and `sampled` samplers; seeded
sampling is not claimed.

### Gemma deployment / lock digest file sets

`gemma-4-e4b-it` pins `google/gemma-4-e4b-it` (HF snapshot `d6436b3d…8e85`, bf16, no quantization, no
speculation) as served by the pinned SGLang image on the GPU1 twin `cere-gemma-g1` and on node-b's RTX 5090
(n3 design note D2-D9). The digest rules above apply to the files present: `tokenizer_sha256` covers the present
members of the tokenizer set, `model_content_sha256` the present members of the model-content set, and
`artifact_sha256` every top-level file of the snapshot with symlinks followed (an HF snapshot links into `blobs/`).
Gemma has `tokenizer.json` and `tokenizer_config.json` but no `merges.txt`/`vocab.json`, and `model.safetensors`
plus `config.json` as model content. `python3 scripts/live/gemma_deploy.py digests DIR` prints all four (it refuses a
directory without `chat_template.jinja`); on `/var/lib/cere/models/models--google--gemma-4-e4b-it/snapshots/d6436b3d…8e85`:

| Field | Value |
|---|---|
| `artifact_sha256` | `2395797c6c8f5d308f2127cf7a97e801642aa6f60c7b5dafd6012d6a4fe096c1` |
| `model_content_sha256` | `25c694e2b120d4d5529a2074dccf6db899254912a22ae2a7ba5edeab1e0495d3` |
| `tokenizer_sha256` | `f2b7feb72dc2b3698b0a38982866db5d63a0be59303f6229e923ad50bb918d9a` |
| `template_sha256` | `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082` |

`backend_build_sha256` is the image's manifest digest `6af86bea…8c69`, the Id under which this host's docker
(containerd image store) knows it. node-b's docker (overlay2) knows the same image (same layers and config) by its
config digest `614d9261…401c`; `gemma_deploy.py node-b-prepare` tags that Id `cere-sglang:6af86bea` and node-b
containers run the tag after its Id is checked. The capability profile is `backend_managed`, `evidence =
"declared"`, rows 1..4, context 32768, text only; SGLang kernels are unmeasured by CERE.

## Prepared artifacts

`configs/artifacts/**` holds the committed metadata of the M2.1 prepared artifact ([ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md)).
`make validate` maps each file name to a schema (an unknown name or directory fails):

| File | Schema | Content |
|---|---|---|
| `<rules_id>.families.json` (`qwen4_exp-nvfp4.families.json`) | `artifact_families` | ordered family rules `{family, pattern, component, role, dtype, shape_rule, placement}` and `groups` |
| `profiles/<profile_id>.json` (`rtxpro6000bw-plehost-text-v1.json`) | `artifact_profile` | `{profile_id, device_model, device_memory_mib, compute_capability, compute_capability_source, modalities, speculative, placement}`; placement maps each family role to `device`, `host_pinned` or `unused` |
| `<name>.inventory.json` (`qwen38-flash-next-nvfp4.inventory.json`) | `artifact_inventory` | `cerec inventory` output: per-shard header metadata, per-dtype/family/placement tables, upstream match, header digest; no tensor data |
| `<name>.pin.json` (`qwen38-flash-next-nvfp4.pin.json`) | `artifact_pin` | `{schema_version, artifact_id, path, manifest_sha256, profile_id, lock_id, model_hash, tokenizer_hash, template_hash, source_revision, created_utc}`, written by `scripts/live/artifact_live.py` after `prepare` |

Cross-checks: a profile file is named after its `profile_id`; family `rules_id`s are unique; an inventory's
`family_rules` names a committed families file by `rules_id` and SHA-256; a pin's `lock_id` names a committed
deployment lock whose `model_content_sha256`, `tokenizer_sha256` and `template_sha256` equal the pin's
`model_hash`, `tokenizer_hash` and `template_hash`, and its `profile_id` has a committed profile file.

The artifact root is `/var/lib/cere/artifacts` (`cerec prepare --out ROOT`, outside every checkout). The pin's
`manifest_sha256` is what every reader passes as `--expected-manifest-sha256`. The real pin is
`qwen38-flash-next-nvfp4-b580d9bd2713aba7` with manifest `b580d9bd…04e7`. The profile's compute capability
`12.0` stays marked `inferred` in the profile; the CUDA executor refuses at creation any device that is not compute
capability 12.0 (`CERE_ERR_UNSUPPORTED`).

The inventory's header digest uses the lock-digest rule: SHA-256 over `<hex sha256 of bytes [0, 8+n)>  <file>\n`
lines, one per shard, ordered bytewise by file name. On the real model (206 shards, 296,475 tensors) it is
`2b680a69…22d3`.

## Config hash

`RuntimeConfig::config_hash()` is the SHA-256 of the canonical JSON (sorted keys, no whitespace) of the
effective config. The effective config serializes to exactly the TOML document, so the value equals
`canonical_digest(tomllib.loads(text))` in `scripts/cere_contracts.py`. It changes with any value, not with key
order or comments.

A live reload ([RUNBOOK.md](RUNBOOK.md#live-config-reload)) never moves it: the `cere` identity and the journal
keep the startup config's hash, and `GET /v1/cere/stats` reports the applied config's hash as
`effective_config_hash`. The file a config was loaded from is not serialized, so it is not part of the hash.

## Policy

Defaults are conservative: loopback, metadata-only traces, no external tools, no online learning, no cloud
egress and no automatic power changes. Budget decreases drain in-flight work rather than reclaiming leased
memory.

Hot-reloadable at a safe scheduling boundary: queue policy/weights, bounded admission ceiling, prefill token
cap, trace sampling. Restart/rebuild: devices/topology, backend mode, model/quant/layout, cache format and
graph allocation policy. Reject the whole update if any field is invalid.

Tunable policy numbers and thresholds belong in versioned config. ABI tags, protocol constants, mathematical
invariants and compile-time kernel shapes are not arbitrary runtime knobs.

In backend-managed M1 the external backend owns the physical allocation mechanism; its reported state must be
reconciled with CERE before a model transitions to READY or UNLOADED.

## Cluster configuration

`configs/cluster/dev-tailnet.yaml` is a planning/default contract for the current Tailscale environment. It defines the control transport, RPC port, heartbeat/suspect/lost thresholds and explicitly disables cross-node state migration/cooperative inference. Runtime secrets and Tailscale credentials do not belong in repository config. Tailnet policy is deployed outside CERE; the config records that such policy is required.

`allow_cross_node_state_migration` (`false` in the examples) refers to physical state (KV or recurrent state copied between workers), which is not implemented. It does not gate the M4 session-owner migration (`POST /v1/cere/sessions/{id}/migrate`, replay mode, no state copied), which needs only `[auth] operator_token_env`.

## Federation (M3.5)

### `[federation]` (coordinator, runtime config)

Optional; requires `deployments.kind = "sglang_http"` (m35 design note Q16). Committed example:
`configs/runtime/federation-dev.toml`.

| Key | Rule | Default |
|---|---|---|
| `cluster_id`, `coordinator_id` | identifier pattern | required |
| `listen` | canonical `<ip>:<port>` literal, host loopback or 100.64.0.0/10, port 1..65535, never a wildcard | required |
| `principal` | lowercase DNS name, at most 253 bytes; the only dNSName SAN of the coordinator leaf | required |
| `ca_file`, `cert_file`, `key_file` | absolute; outside the checkout; not under `ledger.state_dir` | required |
| `heartbeat_interval_ms` | 100..600000 | 1000 |
| `suspect_after_ms`, `lost_after_ms`, `admission_lease_ms` | 100..3600000 | 5000, 15000, 4000 |
| `rpc_timeout_ms` | 100..600000 | 5000 |
| `rpc_quota_per_principal` | 1..4096 concurrent registry calls per principal | 64 |

Timings must satisfy `heartbeat_interval_ms < admission_lease_ms < lost_after_ms`, `suspect_after_ms >
heartbeat_interval_ms`, `lost_after_ms > suspect_after_ms` and `admission_lease_ms < suspect_after_ms`. Absent timing
keys are left out of the effective config, so the config hash is unchanged.

`[[federation.workers]]` (1..64 entries): `worker_id` (identifier, unique, not a `scheduler.devices` id),
`endpoint` (`https://<literal>:<port>`, host loopback or 100.64.0.0/10, at most 128 characters), `principal`
(unique, not the coordinator's), `max_inflight` (1..4096) and `[federation.workers.expect]` with
`served_model_name`, `tp_size` (1..64) and `kv_cache_dtype`: the inventory pins a remote worker must report before
Ready. `admission.max_backend_inflight` must not exceed (local + remote workers) × min(`expect.max_running_requests`,
every `max_inflight`). `cered` reads the PEMs at start; an unreadable or invalid PKI exits 2.

Team pools (net-teleport design note 4.4, S6), all optional per worker: `owner` (a configured `[[auth.scopes]]`
`scope_id`; never `cere-smoke`, which is then refused as a scope id; absent = unowned), `tier` (`own`, `team` or
`provider`, default `own`; the owner always sees its worker as `own`) and `owner_only` (1..16 windows, needs `owner`):
`"<days> <HH:MM>-<HH:MM> <zone>"`, e.g. `"Mon-Fri 09:00-17:00 Europe/Berlin"`. Days are `Mon`..`Sun`, single days,
ranges (`Fri-Mon` wraps) or comma lists; the start is before the end (`24:00` ends at midnight; an overnight window
is two entries); the zone is `UTC` or an IANA name read from the system zone database (`$TZDIR`, else
`/usr/share/zoneinfo`; RFC 8536 TZif with its POSIX rule). Inside a window only the owner is granted the worker;
in-flight non-owner work finishes, bounded by `admission.owner_wait_cap_ms`.

### `[auth] operator_token_env`

Optional name of the environment variable holding the operator bearer token for the drain routes (M3.5 Q13).
Uppercase env-var pattern, at most 64 characters, different from every scope `token_env`. Absent: the operator
routes are disabled. A missing, empty or non-UTF-8 value, or one equal to a scope token, is a startup failure.

### `[auth] keyless_scope`

Optional id of one configured scope (g9-prod design note F2): a client request without an `Authorization` header
is served as that scope (its admission share, cache salt, request and session visibility). A present header is
judged as without the setting (a wrong or malformed token is `401`), and the operator routes ignore it (no header is
`401`). `server.bind` is always `127.0.0.1`, so only local processes reach a keyless scope. Absent: every client
request needs a bearer token. The production T4 config `configs/runtime/prod-t4.toml` sets it for frontend's
substitute proxy, which sends no `Authorization` header.

### Agent config (`cere-worker`, `configs/worker/*.toml`)

Strict TOML (`schemas/worker_agent_config.schema.json`); `cere-worker --check-config <toml>` checks shape only and
opens no PEM and no launch program. Examples: `configs/worker/node-b-mock.toml` (attached: the agent serves a backend
someone else runs) and `configs/worker/node-b-managed.toml` (managed: the agent launches and supervises its backend,
m36 design note Q3, Q13).

| Key | Rule | Default |
|---|---|---|
| `cluster_id`, `worker_id` | identifier pattern | required |
| `listen` | canonical `<ip>:<port>` literal: loopback, 100.64.0.0/10 or RFC1918 (a bridge container), never a wildcard | required |
| `principal` | the worker's DNS-name principal; not `coordinator.principal` | required |
| `ca_file`, `cert_file`, `key_file` | absolute; outside any checkout; not under `state_dir` | required |
| `state_dir` | absolute; outside any repository checkout | required |
| `[coordinator] endpoint`, `principal`, `coordinator_id` | endpoint as for `[[federation.workers]] endpoint` | required |
| `[backend] base_url` | `http://` with a 127.0.0.0/8 or `[::1]` literal and an explicit port | required |
| `[backend] load_endpoint` | a load path such as `/get_load` | absent |
| `[backend] served_model_name` | served model name pattern | required |
| `[backend] lock_sha256` | 64 lowercase hex, the approved deployment lock; set exactly when `[backend.launch]` is | absent (attached) |
| `[backend.launch] argv` | 1..64 entries, each 1..4096 bytes without NUL; `argv[0]` absolute (`--check-config`); at agent start it must be an executable regular file, else exit 2 | required in the table |
| `[backend.launch] env` | at most 64 extra variables; key `^[A-Za-z_][A-Za-z0-9_]{0,127}$`, not `PATH` (fixed); value at most 4096 bytes without NUL | empty |
| `[backend.launch] startup_timeout_ms` | 1000..3600000, spawn to a confirmed identity | 600000 |
| `[backend.launch] stop_timeout_ms` | 100..600000: TERM to KILL, and again KILL to the group gone | 30000 |
| `[backend.launch] uid`, `gid` | 1..4294967294, both or neither; required when `cere-worker` runs as root, refused when it does not (checked at start, exit 2; not by `--check-config`) | absent |
| `[journal] max_operations` | 1..1000000, live + retained operations | 4096 |
| `[journal] retry_horizon_ms` | 600000..86400000, tombstone horizon (at least the 600 s work cap) | 600000 |
| `[limits] heartbeat_interval_ms` | 100..600000, below `lost_after_ms` | 1000 |
| `[limits] lost_after_ms` | 100..3600000 | 15000 |
| `[limits] rpc_timeout_ms` | 100..600000 | 5000 |
| `[limits] control_quota` | 1..4096 concurrent unary calls per principal | 64 |
| `[limits] output_buffer_bytes` | 1..1048576 per operation | 1048576 |
| `[limits] output_stall_ms` | 1..3600000, at most `lost_after_ms` | `rpc_timeout_ms` |
| `owner` | the person scope owning this worker, scope-id pattern, not `cere-smoke` | absent (unowned) |
| `owner_only` | 1..16 windows as `[[federation.workers]] owner_only`; needs `owner` | absent |

Journal sizing: `max_operations` must cover the expected operations per second × `retry_horizon_ms`; at the bound a
new operation is refused `journal_full` and nothing is evicted (m35 design note Q12).

The runtime config's loopback and attach rules for `[[deployments]]` (F8) apply to `cered`'s own backends only. An agent
backend keeps its own loopback `base_url` rule; in managed mode it is launched by the agent, and the coordinator's
`cered` config is unchanged.

Managed mode (`[backend.launch]`): the agent spawns `argv` as its own process group with an environment of exactly
`env` plus `PATH=/usr/bin:/bin`, stdin null, stdout/stderr inherited, dropping to `uid`/`gid` (supplementary groups
cleared) when set. `argv` must make the backend listen on `[backend] base_url` and must not daemonize or call
`setsid` (the agent tracks and kills only its recorded group). The spawned group is recorded in
`<state_dir>/backend_child.json` until its stop is confirmed.

Team pools (net-teleport design note 4.4, F2): an owned agent refuses a new `SubmitWork` of another scope with
`UNAVAILABLE` `owner_only` while it is owner-only: inside one of its `owner_only` windows, or while its local
switch is on. `cere-worker owner-only on|off --config <toml>` sets the switch (a marker file `owner_only` in
`state_dir`); the running agent reads it at every new `SubmitWork` and reports it in its next heartbeat, so the
coordinator stops granting the worker to other scopes.

## v0.6 cluster limits

Cluster schema revision 2 includes heartbeat/loss/admission-lease intervals, RPC timeouts, separate admin/inference message limits, output-buffer limits and explicit coordinator/worker principal bindings. These are planning limits, not measured network recommendations. Secret certificate/key material is external; the example does not deploy mTLS or tailnet policy. Local transport cannot simultaneously enable remote workers.

The Python canonical-digest helper is a deterministic local encoding, not a claim of universal RFC canonical JSON. Live cross-language hashing must pin normalization/serialization vectors or hash the agreed immutable bytes, with compatibility tests before federation.
