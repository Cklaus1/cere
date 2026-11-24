# executor/vulkan (VulkanAmd, vulkan-amd design note)

P1: the small-M GEMV (K2, m <= 16, BF16 and MXFP8 weights) as standalone Vulkan ktests, plus a minimal launch layer.

- `src/launch/`: device/queue selection, required features, <= 4 GiB buffers, SPIR-V pipeline cache, timeline-semaphore submit, GPU timestamps.
- `src/kernels/k2_mgemv_vk.*` + `shaders/k2_mgemv.comp`, `shaders/k2_mxfp8_dequant.comp`: the kernel and its mapping table.
- `ktest/k2v_test.cpp`, `ktest/k2mx_test.cpp`: the HIP tests' contract rows on Vulkan; `ktest/hip_gemv_timing.cpp`: same-method HIP reference timing.
- `p0/`: P0 measurements (not part of the build).

Build (from the worktree root; outputs into `build/vulkan/`):

    docker run --rm -v "$PWD":/w -w /w llama-amd:rocm7.2.4 sh executor/vulkan/build.sh

Run (GPU, under the shared lock):

    R="docker run --rm --device /dev/dri --device /dev/kfd --group-add 990 --group-add 44 -v $PWD:/w -w /w llama-amd:rocm7.2.4"
    flock /tmp/cere-gpu.lock $R build/vulkan/k2v_test
    flock /tmp/cere-gpu.lock $R build/vulkan/k2mx_test            # --no-timing: contract rows only; --tune: mapping sweep
    flock /tmp/cere-gpu.lock $R build/vulkan/hip_gemv_timing      # HIP reference, same timing method

Each test prints PASS/FAIL per row and exits non-zero on any failure. Toolchain: the image's glslc (shaderc 2023.8,
glslang 14.0.0), libvulkan-dev 1.3.275, g++ 13.3; Mesa RADV 25.2.8. No third-party code is vendored.

## Launch layer as a library (P3 T1)

`src/launch/vk_launch.{h,cpp}` is the library path (built as `$OUT/vk_launch_core.o`): `Context::create(ordinal, &ctx,
&why)` selects Vulkan device `ordinal` among the integrated/discrete GPUs (llvmpipe only with
`CreateOptions::allow_software`); every fallible call returns a `cere_status`; nothing exits or prints. Recording
errors latch into the open command buffer (`submit_wait` returns them, nothing is submitted); a failed submit, a failed
or timed-out wait (`CreateOptions::wait_timeout_ns`, default 120 s) or `VK_ERROR_DEVICE_LOST` faults the context for
good (`faulted()`; every later call is `CERE_ERR_INTERNAL` without touching the device). One thread per context.
Also: `warm(specs)` / `has_pipeline` (pipeline warm-up; `vkk1::pipeline_specs` is the worked example; `warm` compiles
the uncached specs on up to `kWarmThreads` = 8 threads, since RADV/ACO compiles on the calling thread, and reports the
first failure in `specs` order, as the serial loop did), `table_alloc`
(the per-command-buffer descriptor-table ring, 1 MiB by default, reset at `begin`), per-context probe state
(`probe` / `set_probe`, used by `vkk2::gemm_probe` / `vkk7::probe`), `allocate(bytes, out, host_heap, &why)`.

`src/launch/vk_launch_testing.{h,cpp}` (`vk::testing::create_or_die`) is the ktests' exit-on-error convenience; the
ktests link `$OUT/vk_launch.o` = core + testing. The library must link `vk_launch_core.o` (or compile
`vk_launch.cpp`) only.

SPIR-V is embedded (OD-4): after all fragments, `build.sh` runs `tools/embed_spv.py $OUT/spv $OUT/gen/vk_spv_table.cpp`
(`.incbin` of every module, sorted, with its sha256) and compiles `$OUT/vk_spv_table.o` (`-fPIC`). `Context::pipeline`
resolves a name from that table first and from the shader directory only when `set_shader_dir` was called (the ktests
pass argv[1]; they do not link the table). `$OUT/vk_launch_test <spv>` checks the launch layer's contract (it links
the core object and the table, not the testing convenience).

## Library: `libcere_executor_vulkan.so` (P3 T2)

`build_lib.sh` (in the image, from the worktree root: `docker run --rm -v "$PWD":/w -w /w llama-amd:rocm7.2.4 sh
executor/vulkan/build_lib.sh`) runs `build.sh` with `CXXFLAGS_EXTRA=-fPIC` into `build/vulkan-t2b/k`, then links
`build/vulkan/libcere_executor_vulkan.so` from the unchanged host layer (`executor/cuda/src/host/*.cpp`,
`executor/cuda/src/research/*.cpp`), the Vulkan device (`src/device/`), every kernel object, `vk_launch_core.o` and
`vk_spv_table.o` (`-fPIC -fvisibility=hidden`, `executor/cuda/src/host/exports.map`, `-lvulkan`, `-z defs`); never a
ktest or `vk_launch_testing.o`. It prints the library sha256 and the exported function count (the ABI's 50), and
builds three tests into `build/vulkan-t2b/`:

- `smoke LIB executor/include/cere_executor.h executor/cuda/src/host/exports.map`: dlopen; exported set == the ABI
  functions; device id grammar; model-create refusals through the ABI.
- `device_test [ARTIFACT]`: binding of the real S3 headers (host only), levers, and on the GPU a synthetic model's
  blocks, upload / read-back, sessions, refusals, ledger, a simulated fault.
- `load_s3 ARTIFACT [WINDOWS]`: the full S3 load through the ABI (~108 GB; check memory, hold `/tmp/cere-gpu.lock`).

The device (`src/device/vulkan_device.h`) loads, seals and opens sessions, and (T5) runs the text forward.

## Forward (P3 T5)

`src/device/vulkan_forward.{h,cpp}`: a Vulkan copy of `executor/cuda/src/model/forward.cpp`'s eager text path (the
`plan::kDecode` op walk, `attention_rows` for the > 2,048 context plan with dense K6 up to 2,048 positions and K9 QSA
above, the K5 / K7 / K3 / K4 kernels, lm_head + top-2, the K12 sampler), every block citing its CUDA line. `step` runs
one PREFILL chunk (<= 4,096 rows) or one DECODE token per call; `step_rows` the single-row decode (max_sessions 1).
A decode step is two command buffers (the first after 4 layers, both in flight: "Decode submits" below); a prefill of
>= 256 rows is submitted every 4 layers (the amdgpu job timeout); every command buffer ends with a completion canary
the host checks. MXFP8 weights (S3) reach every kernel with their scales. Refused: images, graphs (speculation:
below); prefix end states run since D15 (below). `forward_supported()` / `sampler_supported()` are true once a model
is sealed. Per-model statistics (`VulkanModel::forward_stats()`: record / GPU / wait ms, command counts, the K2 GEMM
exact-fallback counters per consumer) are logged at model release.

`build_lib.sh` also builds `forward_s3` (dlopen any library of the ABI family: ABI v5 here, ABI v4 for the amd-amd HIP
library) and `forward_s3_static` (the Vulkan objects linked in, the forward compiled with `CERE_VK_FORWARD_TAPS`: a
per-layer hidden-state dump; split-prefill rows; statistics): `forward_s3 LIB ARTIFACT PROMPTS TRACE [N] [--no-sampler]
[--prompts a,b] [--consistency name:k,..] [--taps FILE]`. Prompts: `tests/data/forward_s3_prompts.txt` (token ids made by
`tests/data/forward_s3_tokenize.py` with the artifact's tokenizer). Traces compare with
`tests/forward_s3_compare.py REF CAND [--identical]`. Run the Vulkan library with `--device /dev/dri --group-add 990`
(no `/dev/kfd`); the HIP library with `--device /dev/kfd --device /dev/dri --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0`.

## Prefix cache and end states (P4 owner item 3, D15)

A model created with `cere_model_desc.prefix_state_slots > 0` (and no speculative drafts: that combination is refused
UNSUPPORTED at create) carves `prefix_state_slots + max_sessions` fixed-state slots (`VkRegionKind::kStateSlots`, one
block) and runs the host's M2.10 prefix cache: lazy sessions, `bind_pages` / `retain_pages` / `release_pages` /
`take_pages`, `capture_state` / `load_slot` (one k1 copy of the fixed state, `vk_bind_fixed_state` restating
`model.cpp:1076-1105`). `step` takes an `EndStep` (CUDA's `cuda_device.cpp:1279-1347`): every row-count choice of the
call is made at its class `cls`; a capture at a multiple of 64 writes the state as of that row into a slot pair (K5
`EndState` FP32 SSM, K4 `Capture` PLE, and for a chunk-0 end state the BF16 K/V run of `end_kv_slots`, CUDA's rule);
a resume starts pinned to the class from the pair. Not ported: the snapshot arena / host tier (`save_prefix`,
`load_prefix`). Test: `forward_s3_static ... --prefix-slots 8` (bar B9 rows, vulkan-amd design note).

## Speculation (P3 T6)

`src/device/vulkan_spec.cpp` (the `VulkanDevice` speculation methods) + `vulkan_forward.cpp` (`vk_verify`,
`vk_commit_draft`, `vk_extend_draft`, the MTP draft layer): MTP speculation as `executor/cuda` runs it, for one session.
A model created with `speculative_draft_tokens` 4 and the artifact's `draft/token_map.u32le` binds the `mtp.*` tensors;
INSTANTIATE uploads the token map, gathers the hot lm_head (65,536 rows) and prepares the K13 row info. A drafting
session's PREFILL steps run the MTP extend over the chunk (its own command buffer); each decode round is (1) verify: the
last token + up to 3 MTP drafts as 4 rows through the target model in ONE command buffer (undo snapshots written
in-kernel by K4 `forward_rows` / K5 `chain_core`, QSA ring / key rows copied; per-row attention; one lm_head GEMV over
the rows; phase S per row and `k13s::spec_accept` for a sampled chain), (2) the accept on the host, (3) commit + draft
in ONE command buffer (undo restore of the rejected rows, sampled commit counts, MTP extend over the committed rows and
two MTP decode steps). Greedy speculation is lossless: committed ids and their top-2 logits equal plain decoding bit for
bit (`spec_s3`, report an internal benchmark report). `times_spec_rounds()` is false
(OD-6: untimed cost tables; fixed-window tickets need none). Refused: `decode_step` graph / spec_graph,
`verify_attention` batched, draft values other than 4. A constrained row with the caller's drafts (grammar drafts,
GrammarSpec GS-02) runs since P4-F with one mask per verify row (below). `kDraftForwardImplemented`
(vulkan_device.h) switches the whole path off (drafts refused at allocate).

`build_lib.sh` builds `spec_s3` (dlopen; the Vulkan library or the amd-amd HIP library, ABI v4) and `spec_s3_static`
(device-level rows and per-round GPU statistics): `spec_s3 LIB ARTIFACT PROMPTS TRACE [N] [--prompts a,b]
[--long name:N] [--quick]`. Per prompt it runs plain (`CERE_WORK_NO_SPECULATION`) and speculating greedy decodes in one
model (~107 GB) and compares them bit for bit; then a stop-token prompt, the sampler rows, a cancel + DECODE
continuation, VERIFY and the negative rows.

Completion marker (P3 T1b): RADV can return VK_SUCCESS from the wait for a job amdgpu killed at its ring timeout (about
2 s here; DEVICE_LOST only at the next submit). `submit_wait` therefore ends every command buffer with a barrier, an
8-byte `vkCmdUpdateBuffer` of a per-submit nonce into a Context-owned word, and an end timestamp; a wrong word after a
successful wait faults the context ("completion marker missing ..." in `fault_reason()` / `last_error()`). No pipeline
is involved, so warm-up counts are unchanged. `last_submit_gpu_ms()` reports each submit's GPU time; callers should
keep submits under `vk::kMaxSubmitGpuMs` (1,000 ms; a hint, never enforced). VK_EXT_device_fault is enabled when
offered and only enriches the fault reason.

## Prefill attention selection (P4-E)

The model's `prefill_attention` lever picks the > 16-row attention path of K6 dense and K9 sparse prefill calls:
`cuda_core` (default) = the FP32 kernels of P2, `p4` = the cooperative-matrix kernels of P4-A (`vkk6` / `vkk9`
`Prefill::kCoopmat`, with their exact FP32 fallback for flagged rows; report
an internal benchmark report). Calls of <= 16 rows (decode, verify, MTP decode) run the
split path under either value, bit for bit. The device accepts `p4` only on a device with `VK_KHR_cooperative_matrix`
(at create and `set_selection`) whose seal-time `vkk6::coopmat_probe` returns FP16 subnormals kept (`0x0010`,
`0x37800000`); a failed probe fails the seal of a `p4` model (UNSUPPORTED) and keeps `cuda_core` models unchanged. The
probe passing warms the coopmat pipelines too, so either selection runs without a pipeline compiled after the seal;
the memory plan charges the coopmat workspace (the flags), so a hot `set_selection` needs no new buffers. Reported:
`cere_model_selection` (the active lever), the seal log line (`prefill_attention=coopmat|fp32 coopmat_probe=...`),
and the release log line (`prefill_calls_coopmat/fp32`, per-kernel flag counters `attn_flags=[k6:..;k9:..]`, the K2
split-route counts `split=[..]`, `prefill_max_submit_gpu_ms`). `forward_s3` / `spec_s3` take
`--prefill-attention cuda_core|p4`; `forward_s3_static` prints per-prompt prefill GPU time and counters, and with
`CERE_VK_FORWARD_PROFILE=1` (test build only) the prefill GPU time by op family. Report:
an internal benchmark report.

## Pipelined submits (P4-G)

Launch layer (`src/launch/vk_launch.h`): `CreateOptions::max_in_flight` (1..`kMaxInFlight` = 4, default 1) command-buffer
slots, each with its own command buffer, stamp pools, completion-marker word and table-ring part (`table_ring_bytes`
per command buffer). `submit(&ticket)` ends and submits the open command buffer without waiting; `begin()` opens the next
one on the lowest free slot (ARGUMENT when every slot is in flight) and, when another submit is in flight, starts it
with a full memory barrier, so command buffers run back to back in submission order. `wait(ticket)` retires every
in-flight submit up to `ticket` strictly in submission order (bounded wait, then its marker); `wait_all()`;
`submit_wait()` = `submit` + `wait`. Kill ordering: the first failed retire (wait error / timeout / DEVICE_LOST or a
missing marker) faults the context; that ticket and every later in-flight one are never reported complete
(`last_retired()` stays below it, the reason names the ticket and the discarded count), whatever their own markers say.
`stamp_ms` / `last_submit_gpu_ms` refer to the last retired submit; `last_submit_timing()` adds the host submit / wait
times and the GPU idle gap before it (and, with `set_calibrated_timing(true)` and VK_EXT_calibrated_timestamps, the
GPU start / host wake-up latencies). Test hooks: `omit_completion_marker_for_test()` (the command buffer really does not
write its marker), `skew_completion_nonce_for_test()`. The amdgpu job timeout is per job: 10 x 296 ms command buffers
back to back (2.96 s, two in flight) ran with no ring timeout (`vk_launch_test SPV --queue-probe 300 10`).

Forward: the device creates its context with `kPrefillSubmitsInFlight` (2) slots; a segmented PREFILL call submits
segment k + 1 while segment k runs (`Rec::open_segment` retires the oldest only when both are in flight; the call's end
drains). Each in-flight segment keeps its own canary words. The exact-fallback / coopmat-flag counters are summed per
segment at its retire (overlapped with the next segment's GPU work) from one streaming copy of its stats words, instead
of per-word reads of write-combined memory after the call (that read was the ~18 ms "per submit" of P4-E: ~70 us per
prompt row). Verify, commit and MTP extend stay one command buffer, waited at once, commands unchanged. Test
build only: `CERE_VK_FORWARD_SERIAL=1` (depth 1: the P4-E order), `CERE_VK_SUBMIT_TIMING=1`; `forward_s3_static`
prints per-prompt and total submit breakdowns. Report: an internal benchmark report.

## Decode submits (VulkanDecode, bar B1)

A DECODE step (`vk_forward`, one token) is submitted as two command buffers in flight (`Rec::split_at` =
`kDecodeFirstSubmitLayers` = 4, `vulkan_forward.h`): the GPU runs the first 4 layers (~3 ms) while the host records
the other 44 layers, lm_head and top-2 (~0.6-1 ms), so that recording no longer sits in front of the GPU; same
kernels, same order, the decode traces are bit-identical. The step's last wait is `Context::wait(ticket, spin_from_ns)`:
it blocks until the previous decode step's GPU time minus `kDecodeSpinMarginMs` (0.75 ms) after the first submit, then
polls the timeline value for at most `kMaxSpinNs` (3 ms) before the bounded blocking wait; the retire (marker,
stamps, kill ordering) is unchanged. A recording refusal after the first submit breaks the session (`*launched`), as a
segmented prefill's. Test build only: `CERE_VK_DECODE_ONE_SUBMIT=1` (one command buffer, the old order),
`CERE_VK_DECODE_NO_SPIN=1` (blocking wait); `forward_s3_static` prints `decode segments:` (GPU idle between a step's
two command buffers; with `CERE_VK_SUBMIT_TIMING=1` the host return after the step's GPU end). Report:
an internal benchmark report.

## Coverage harness and grammar drafts (P4-F)

`build_lib.sh` also builds `coverage_s3` (dlopen, ABI v5) and `coverage_s3_static` (the library's objects; device rows,
KV page and ledger checks, a GPU-time guard): `coverage_s3 LIB ARTIFACT PROMPTS TRACE --model plain|drafts --parts
long,mask,cancel,levers,window,r5,r5grammar,longspec [--prefill-attention p4] [--ladder N,..] [--longspec N,..]
[--r5-samples S] [--r5-tokens T] [--r5-prompts a,b] [--guard-ms G]` (PROMPTS: `tests/data/spec_s3_prompts.txt`). It
covers what first light left unexercised: a full 4,096-row chunk and the 32,768 limit (32,767-token prompt; past-limit
prompts CAPACITY; device step at the last KV slot), vocab masks (`cere_executor_constrain`), cancel mid-prefill and
mid-decode with bit-identical later sessions, hot lever swaps and refusals, the draft_tokens window 2 / 3 / 4, the m29 R5
sampled-speculation distribution test (pooled two-sample chi-square, Bonferroni 1e-3 / m) and greedy speculation at
long context. Grammar drafts (a constrained row with the caller's drafts, GrammarSpec GS-02) run: the verify recorder
uploads the chain's n + 1 masks (the IO block holds `verify_rows` masks) and row r's K12 stage reads mask r
(`forward.cpp:1488-1494`); `grammar_drafts = "on"` greedy equals "off" bit for bit. Report:
an internal benchmark report.

## Prefill vs HIP (P4-I)

The K2 GEMM's exact-fallback counters take the x-row info words the GEMM's own scan writes (`vk::Context::arm_k2_info` /
`take_k2_info`: the forward arms a tap before a counted consumer's GEMM, the GEMM writes its scan there), instead of a
second `k2_gemm_scan` over x. `k7_gemm` keeps each lane's weight scales in registers and stages K in chunks of 64
(less LDS per workgroup: more workgroups per WGP); the per-element FMA order is unchanged, so outputs are bit-equal.
`allocate_storage` creates the device-local non-weight blocks first (workspace, state, KV pool, IO), then the rest in
index order: on AMD the kernel driver puts a buffer in the 2 GiB VRAM carve-out while it fits, and the forward's
workspace in GTT made every prefill family of a model with drafts ~11 % slower. Layout, offsets and digest are
unchanged. `forward_s3 --drafts` allocates the model with drafts (the speculating profile's allocation) and submits
every item with `CERE_WORK_NO_SPECULATION`; `spec_s3` takes `--k2-gemm`, and `spec_s3_static` prints per prompt the
plain and speculating prefill GPU time and the MTP extends (with `CERE_VK_FORWARD_PROFILE=1` by op family). The K3 /
K5 / K7 ktests and their HIP timing twins time m = 1,205; `k5_chunk_test` prints FNV-1a digests of its outputs
(`bits:`). Report:
an internal benchmark report.
