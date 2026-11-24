"""K8 (tasks/m22-spec.md, M2.2b Tier 2): the CUDA forward pass and runtime on the real A4 model, GPU0, container only.

Suite `model` (`native_dev.py test --suite model`). The test owns nothing but the bytes: it reads every checkpoint
tensor from `/model` (read-only) with safetensors and uploads it through ABI v3's staged creation, exactly as the
Rust upload sink does (the executor never opens a file):

* the model descriptor is the ABI v5 `cere_model_desc` of `executor/include/cere_executor.h` (mirrored below with
  ctypes): `CERE_MODEL_QWEN4_EXP_NVFP4`, the typed `cere_qwen4_exp_config` filled from `config.json` `text_config`
  by the header's rules, one `cere_tensor_desc` per checkpoint tensor in name order (name, safetensors dtype, shape,
  bytes) with the placement of `configs/artifacts/qwen4_exp-nvfp4.families.json` (target `device` -> DEVICE, target
  `host_pinned` -> HOST_PINNED, MTP draft and vision -> DISCARD, never put), `max_sessions` ROWS_R = 4 (M2.6 rows,
  tasks/m26-spec.md D3; the merge-base model of L1/L2 keeps 1), the context limit of the milestone (2,048 for
  M2.2b), and the stop tokens of `generation_config.json` (248046, 248044);
* puts run from 4 threads at once (one safetensors file at a time per thread, pieces of 64 MiB), the concurrency
  ABI v3 allows between create and seal (Q13); seal is INSTANTIATE (eager module loading, workspace preallocation);
* SMOKE is the 1-token greedy generation of a fixed chat prompt right after seal (R3's smoke).

Cases (K8 acceptance):
* `test_cancel_at_token_10`: a cancel at the token boundary after token 10 completes CANCELLED with exactly 10
  committed tokens, which equal the uncancelled run's first 10 bit for bit; a later run is unaffected.
* `test_forward_trace_names_and_shapes`: `cere_ktest_forward_trace` returns every B8 tensor name for all 48 layers
  (`model.layers.N.linear_attn.out_proj` on GDN layers or `model.layers.N.self_attn.o_proj` on full-attention layers,
  `model.layers.N.mlp.experts`, `model.layers.N.mlp.shared_expert.down_proj`, and the final `model`), each with
  SGLang's shape [tokens, hidden_size] (RowParallelLinear `o_proj`/`out_proj` output_size = hidden,
  `$S181/sglang/srt/models/qwen3_5.py:380,974`; FusedMoE and the shared `Qwen2MoeMLP.down_proj` return hidden,
  `qwen2_moe.py:281,318`; `Qwen4ExpVLModel.forward` returns the final mixer's [tokens, hidden],
  `qwen4_exp.py:1691-1712`) and finite values, plus one top-2 entry per position whose last top-1 is the first
  greedy token of `cere_ktest_generate` on the same prompt.
* `test_greedy_repeat_bit_identical`: greedy generation of 4 parity prompts (`bench/prompts/parity_v1.jsonl` p01, p02,
  p13, p24; chat template with `enable_thinking` false as the reference serves them, Q8; `max_tokens` from the
  file) repeated 3x is bit-identical in token IDs and top-2 scores (`cere_token_score` bytes).
* `test_forced_decode_rejects_out_of_vocab_forced_token`: `cere_ktest_forced_decode` (declared in
  `test_numerics_b8_layers.py`) returns CERE_ERR_ARGUMENT for a forced token >= vocab_size and writes no score row;
  an out-of-vocab last token (predicted, never fed) runs.
* `test_forced_decode_trace_scores_bit_equal` (tasks/m22d-spec.md C1, D2): on 3 parity prompts (p01, p02, p13;
  forced tokens = the native greedy completion) the scores of `cere_ktest_forced_decode_trace` equal
  `cere_ktest_forced_decode`'s bit for bit (the tapped direct forward loop computes the serving path); C - 1 DECODE
  steps, each with the same names in the same order, every entry [1, hidden] and finite.
* `test_forced_decode_trace_model_vs_prefill`: every step's names equal `cere_ktest_forward_trace`'s, in its order;
  report-only row per prompt: rel-L2 of step j's `model` tap against the PREFILL forward_trace row at the same
  position (max with its step, median, first, last) and the decode-vs-prefill top-1 agreement over the steps.
* `test_forced_decode_trace_refusals`: CERE_ERR_ARGUMENT for a forced token >= vocab_size and for each capacity
  (names, entries, values) one unit below the trace's size, with nothing written (no score row, out_counts zero,
  names, shapes and values untouched); at exactly the size and with an out-of-vocab last token the call runs.
* `test_memory_after_smoke_equals_post_generation`: device memory in use (cudaMemGetInfo through
  `torch.cuda.mem_get_info`, torch's cache emptied) after SMOKE equals the value after every earlier test's
  generation, traces and cancel plus one more pass over the 4 prompts, within +-256 MiB.

Prefill tap rows (tasks/m22p-spec.md L1, L2; the merge-base library is `build/native-l0/libcere_ktest.so` under
`/src`, built at merge-base(main, HEAD) (spec L0); its digests come from the merge-base digest cache (O186,
base_digests.py: keyed on that library, the pinned manifest, the prompt set, this harness and the runtime) or, on a
miss, are computed in setUpClass on a model of its own that is released before this build's model is created; only
SHA-256 digests of its outputs are kept; the `merge-base digests` row names the source and key):
* `test_forward_trace_equals_merge_base` (L1): `cere_ktest_forward_trace` of p12 and p13 (every row: names, shapes,
  values, scores) is byte-identical to the merge-base library's.
* `test_forward_trace_detail_zero_equals_forward_trace` (L1): `cere_ktest_forward_trace_detail` with detail 0 is
  byte-identical to `cere_ktest_forward_trace` on p12 and p13.
* `test_forward_trace_detail_one_keeps_top2` (L1, M22PS-19): on p12 and p13 (last 64 rows) detail 1 emits exactly the
  model.h ForwardTaps names in forward order (D3 kinds (a) and (b) between the B8 entries), each [rows, its cols] and
  finite; its scores and B8 values equal detail 0's bit for bit.
* `test_forward_patch_identity` (L1): `cere_ktest_forward_patch` with every B8 block output of p12 and p13 replaced by
  its own forward_trace value gives the target logit and top-1 of the unpatched forward bit for bit (target = the
  native top-1).
* `test_forward_patch_screen_identity` (L1): `cere_ktest_forward_patch_screen` with no replacement gives
  forward_trace's last top-2 bit for bit; replacing every patch point by a copy of its own value (one per point: per
  layer the 2 mix outputs, the 3 B8 blocks, the attention / GDN and MoE GEMM outputs, then the final mixer) gives the
  same target logit and top-1 bit for bit.
* `test_forward_patch_non_identity` (L1, Z2A1-03): on p12 and p13, `cere_ktest_forward_patch` with ONE of
  `model.layers.L.mlp.experts`, `model.layers.L.mlp.shared_expert.down_proj`, `native.layers.L.attn_mix.out`,
  `native.layers.L.gemm.gdn_qkv.out` (L = the first GDN layer) or `native.final_mix.out` replaced by its own value
  negated (exact in BF16) gives a (target logit, top-2) different from the unpatched forward: a replacement is
  consumed downstream, not only accepted.
* `test_forward_patch_screen_non_identity` (L1, Z2A1-03): the same five negations through
  `cere_ktest_forward_patch_screen`'s callback each move (target logit, top-2); for the two MoE blocks the next
  layer's input point `native.layers.L+1.attn_mix.normed` also differs from the unpatched one.
* `test_forward_patch_refusals`: CERE_ERR_ARGUMENT with nothing written (logit NaN, top-1 -1) for a zero
  token_count, names without a final NUL, an unknown, repeated, wrong-kind or twice-per-layer patch name, values_bytes
  off by one element either way, target_token >= vocab_size, a null screen callback, and detail 2 to
  `cere_ktest_forward_trace_detail`; the same call with one valid name runs.
* `test_decode_invariance_equals_merge_base` (L2): the SHA-256 of the `cere_ktest_forced_decode` top-2 rows of
  prefills at n = 1, 2, 4, 8, 16 (every K2 bucket) and of a 16-token prefill plus 16 forced decode steps, on seeded
  tokens, equals the merge-base library's.
* `test_decode_rows_bit_identical` (tasks/m26-spec.md K3): 4 sessions of seeded synthetic token ids (no prompt
  text) of distinct lengths, each prefilled alone, greedy-decode ROWS_STEPS = 64 rounds through
  `cere_ktest_forward_decode_rows` at up to R = 4 rows (session 1 joins at step 8, session 3 leaves at step 40, so
  the rounds run 3, 4 and 3 rows): every session's scores (hence tokens) equal the same session decoded alone (S = 1)
  bit for bit, and its `model.layers.1.*` taps (the PLE layer, per row) equal the solo taps bit for bit.
* `test_prefill_chunk_rounds_bit_identical` (tasks/m26-spec.md K5, D2a): `cere_ktest_generate_rows` with two short
  parity prompts submitted first (p01, p02: real chat prompts, so greedy decoding outlasts the chunk rounds; random
  synthetic ids hit a stop token after 2 tokens on GPU0), then a 3-chunk synthetic prompt (K5_CHUNK-token chunks): its
  chunks run one per executor round with the two short requests decoding between them; its tokens, scores and
  finish equal the same prompt generated alone at the same chunk size (`cere_ktest_generate`) bit for bit, and each
  short request equals its own solo generation.
* `test_sampled_rows_equal_alone` (ledger CR4, I10 rows 2-4 sampler invariance): `cere_ktest_generate_rows_sampled`
  with the first R = 2, 3, 4 of four parity prompts as requests (CR4_SAMPLING: request 0 sampled at T 1.0 / top_p
  0.95 / top_k 20 with logprobs, request 1 greedy, request 2 sampled with penalties, request 3 temperature 0 with
  logprobs), so the decode rounds put sampled rows into one batched step: each request's committed tokens, finish,
  scores and logprobs equal the same request generated alone (R = 1) bit for bit (the M8 counter key is the row's
  own position, its out_counts its own).
* `test_rows4_sampler_memory` (ledger W, CR4): `cere_ktest_sampler_plan_bytes`: the K12 sampler scratch plus the
  processed-logits row plus R = 4 sessions' out_counts is at most CR4_SAMPLER_BYTES (8 MiB), with R = max_sessions =
  ROWS_R; and a 4-request sampled `cere_ktest_generate_rows_sampled` moves this process's device memory (NVML per-PID
  used memory, not the device-wide figure: GPU0 drives the desktop) by at most that (the rows' sampler stages
  allocate nothing), while a second process holds CR4_FOREIGN_BYTES on GPU0 across the run (the control: NVML shows
  it at least that, and the device-wide delta, reported, includes it).

K8Speculation (tasks/m29-spec.md D4, D5, M29S2-20, I8; M2.9 phase G): a model of its own with
`speculative_draft_tokens` = 4 (draft map = every vocab id; the `mtp.*` tensors DEVICE: every drafting session runs
the MTP draft-extend after PREFILL and after every commit, whether its rows take caller drafts or the proposals),
SPEC_SESSIONS sessions and SPEC_CONTEXT (above 2,048, so attention runs the K9 QSA path with its ring and compressed
keys). "Plain" is the same ids fed one DECODE per token through `cere_ktest_session_feed` (teacher-forced) on a
session that does not draft; every comparison is bit for bit:
* `test_spec_undo_round_trip` (K1'): for every a in {1, 2, 3, 4}, verify chains of n = 3 random (wrong) drafts on a
  drafting session committing a rows (`cere_ktest_session_verify`) leave the `cere_ktest_state_digest` (GDN, PLE,
  QSA ring + compressed keys, KV, out_counts) of a plain session fed the same a tokens, and the a committed rows'
  scores equal plain's, over SPEC_UNDO_ROUNDS chains (every position mod 4).
* `test_spec_greedy_bit_identical` (K4): R in {1, 4} drafting sessions, SPEC_DECODE greedy tokens each through
  `cere_ktest_session_speculate` (one `speculate_rows` call per round) with drafts = the true next tokens, one draft
  made wrong at a rotating row (or none): committed counts as the greedy accept predicts, scores equal the plain run.
* `test_spec_mtp_greedy_bit_identical` (K4 with MTP drafts, I1): the same R and prompts through
  `cere_ktest_session_speculate_mtp` (the rows draft from their pending MTP proposals): every committed score equals
  the plain run, accepted = committed - 1 per round, at least one draft accepted; the accept length is reported.
* `test_spec_lookup_greedy_bit_identical` (tasks/m29l-spec.md C6L): R = 1 greedy requests through
  `cere_ktest_generate_flags` on prompts that repeat a seeded block (so the suffix lookup matches), with
  cere_work.flags 0 (MTP only), CERE_WORK_LOOKUP_DRAFTS (the policy picks MTP or lookup per round) and
  CERE_WORK_LOOKUP_DRAFTS | CERE_WORK_NO_MTP_DRAFTS (lookup only): every committed token and score equals the plain
  run; the lookup-only arm accepts at least one draft; verify steps and accepted drafts are reported per arm.
* `test_spec_adaptive_greedy_bit_identical` (tasks/m29a-spec.md A4): R = 1 greedy requests through
  `cere_ktest_generate_flags` on the parity prompts and a repeated-block prompt with cere_work.flags 0 (the fixed
  policy), CERE_WORK_ADAPTIVE_DRAFTS (the adaptive policy picks 0..3 MTP drafts per round) and that with
  CERE_WORK_LOOKUP_DRAFTS: every committed token and score equals the plain run; the fixed arm verifies 3 drafts in
  every round but the last 3; verify steps, accepted and verified drafts and drafts per round per arm.
* `test_spec_long_lookup_greedy_bit_identical` (TreeSpec, tasks/tree-spec-spec.md TS-02): R = 1 greedy requests through
  `cere_ktest_generate_flags` on the repeated-block prompts with the `drafter = "mtp+lookup+long"` flags (bits 2, 4, 5
  and 6) and without bit 5: every committed token and score equals the plain run, and per arm at least one request
  verifies more than 3 drafts per round on average (chains widened past choose's window ran); per arm verify steps,
  accepted and verified drafts.
* `test_spec_adaptive_sampled_seeded_equals_fixed` (tasks/m29a-spec.md A6, T7 [REVISED: review-wave-fg M1]):
  a seeded sampled request (T 0.8) through `cere_ktest_generate_sampled_flags` with the flags the runtime sends for a
  sampled ticket under `draft_policy = "adaptive"` (bit 4 dropped) commits exactly the fixed policy's (flags 0) tokens
  and scores; per prompt, whether the executor's bit 4 on that sampled request changes its tokens is reported (rows
  "M1 bit4"; on `main` `3fa2aefd` the runtime sent bit 4 and the test failed: 5/5 requests diverged by default and
  under `prefill_attention = "p4"`, 1/5 under `small_m`, GPU1 `native-test-model_k8-20261015T093646Z`).
* `test_spec_qsa_rollback_crosses_2048` (K7): one session, a SPEC_QSA_PROMPT-token prompt, chains placed to start at
  positions = 2, 3 (mod 4) so they straddle a ring group, rejecting at each row in turn, to SPEC_QSA_END positions:
  the state digest equals plain's after every commit and the scores equal plain's.
* `test_forced_verify_trace` (K8, I8): `cere_ktest_forced_verify_trace` (the decode steps fed as verify chains of 4
  rows) writes the same names, shapes, per-row tap values and scores as `cere_ktest_forced_decode_trace`.
* `test_spec_sampled_d6_exact` (phase S, D6, M29S2-13/-14): two sampled drafting sessions (SPEC_SAMPLED: temperature
  > 0 under top_k/top_p, one with logprobs) speculate SPEC_SAMPLED_ROUNDS rounds in one call each, drafts = a probe
  session's greedy continuation with one made wrong at a rotating row. Per chain the committed ids equal the D6
  rule computed on the host (`oracle.D6RowRef`, ledger M8 coins at keys p + j, bonus on channels
  (0, 1)) from the verify rows' own probabilities and keep masks (`cere_ktest_spec_stages`); every consulted row's
  processed logits equal a plain sampled session's (`cere_ktest_session_step_sampled`, fed the committed chains after
  the speculating sessions close, so at most 3 of the SPEC_SESSIONS slots are open) at
  that position; each committed token's score is the plain processed row's (its logit, runner-up and runner-up logit)
  and its logprobs `cere_ktest_sample_top_logprobs` of that row; a bonus token equals plain's own draw; the session's
  out_counts (`cere_ktest_session_counts`) gain exactly the committed ids.
* `test_spec_sampled_top1_greedy` (D6 with p(d) = 1): SPEC_SESSIONS sessions, rows 0 and 2 sampled with top_k 1 and
  rows 1 and 3 greedy in the same calls, drafts as K4: committed counts and scores equal the plain greedy run.
* `test_spec_sampled_t0_digest` (M29S2-14): a sampled temperature-0 item with penalties and logprobs, drafts = a plain
  sampled run's tokens with one made wrong: committed counts, scores and logprobs equal the plain run's and the state
  digest (out_counts included) equals plain's after every (rejecting) round.
* `test_spec_masked_t0_bit_identical` (GrammarSpec GS-02, tasks/grammar-spec-spec.md B1's device half): a constrained
  greedy item (SPEC_MASKED_T0: temperature 0 on the sampled path) whose verify row j samples under the vocab mask
  `position_mask` of its position (`cere_ktest_session_speculate_masked`), drafts = the plain masked run's tokens
  (`cere_ktest_session_step_masked`) with one made wrong at a rotating row and the chain ended at the first draft the
  mask bans: committed counts and scores equal the plain masked run's, its state digest after every round too; the
  masks change the run and some drafts are accepted, some refused.
* `test_spec_masked_sampled_d6_exact` (GS-02, B2's device half): `test_spec_sampled_d6_exact` under the same masks
  (chains of 0..3 drafts): the committed ids follow D6 on the masked rows, each allowed by its row's mask, and the
  processed rows, scores and logprobs equal plain masked sampled decode's.
* `test_spec_masked_refuses_mtp_drafts` (GS-03): a masked row drafting from its pending MTP proposals is refused,
  its state digest unchanged; with no drafts it commits one allowed token.
* `test_rows_lm_head_one_launch` (tasks/m23-spec.md P11R-21, RowsPlans::lm_head): one speculate_rows call of
  SPEC_SESSIONS chains of 3 caller drafts (a verify of m = 16 rows) runs one K1 top-2 launch (the lm_head once at m,
  on the default cuBLASLt levers too, not once per row), and a 2-round forward_decode_rows of SPEC_SESSIONS rows one
  per prefill and one per batched step (torch.profiler kernel names).
* `test_spec_round_never_enters_graph` (M2.3 P1 x M2.9, tasks/m23-spec.md M23P1-11, M29S2-18): the decode graph
  serves plain decode only. The plain greedy run of the K4 prompts with the model's decode graphs off (every step
  eager) equals the run with them on bit for bit, and that run replayed one graph per decode step (the counters
  work on this model); K4's speculation (caller drafts) and K4m's (MTP drafts), R in {1, SPEC_SESSIONS}, with the
  graphs on commit the plain scores while no decode-graph counter (captures, in-place updates, replays) moves.
* `test_verify_attention_batched_bit_identical` (tasks/m23-spec.md P11R-19, the verify_attention lever): with the
  model's lever switched to batched (`cere_ktest_verify_attention` mode 1), K4's speculation (caller drafts with a
  wrong row per round) and K4m's (MTP drafts), R in {1, SPEC_SESSIONS}, K1''s undo round trip and K7's QSA rollback
  across 2,048 (a chain straddling 2,048 stays per row, one wholly past it runs K9 batched) commit plain greedy's
  tokens and scores and leave plain's state digest; the attention_chain counter moves under batched (the lever
  reached the verify rows) and not under per_row.
* `test_verify_attention_batched_sparse` (tasks/long-ctx-spec.md LK-01): under batched, DraftAccept deep chains
  (R 1 and 2) on SPEC_SPARSE_PROMPT-id prompts, every row past 2,048 with more than 512 visible blocks: committed
  counts, scores and the state digest after every round equal plain greedy's, and the attention_chain counter moves
  by the full-attention layers times the chains of more than one row (K9 selection and attention batched per chain).
* `test_spec_graph_bit_identical` (tasks/m29-spec.md M29S2-18b, tasks/m23-spec.md P11R-20, decode_step =
  spec_graph): with the model's verify graphs switched on (`cere_ktest_spec_graph` mode 1), SPEC_GRAPH_ROUNDS caller
  verify chains (every draft count n in 0..3, so every m = n + 1 in 1..SPEC_DRAFT_TOKENS, each shape captured once
  and replayed at later positions) from a short prompt and from SPEC_QSA_PROMPT (whose chains past 2,048 fall back
  to eager), under verify_attention per_row and batched, give the eager run's scores and state digest after every
  round bit for bit, and exactly the eligible rounds (p + n + 1 <= 2,048) replay a graph; the MTP draft graphs:
  K4m speculation (R in {1, SPEC_SESSIONS}, max_commit cycling 1..SPEC_DRAFT_TOKENS so every draft-extend row count
  a in 1..4 runs) with the graphs on gives the eager run's committed counts, accepted drafts, scores and draft digests
  (`cere_ktest_draft_digest`: state plus the MTP pool entry and pending proposals) after every round, and some
  drafts replay a graph; K4's and K4m's speculation (R in {1, SPEC_SESSIONS}, m up to 16) commit plain greedy's
  tokens and scores through the graphs, K1''s undo round trip and K7's QSA rollback across 2,048 leave plain's
  state digest, and sampled rounds (T0 digest) and tapped verify calls (the K8 forced verify trace) run eagerly (no
  verify-graph counter moves).

K8SpeculationSmallM (tasks/m23-spec.md P11R-18, PI-02): every K8Speculation test on a model created with the
`k2_gemm` lever at `small_m` (slot impl CERE_LEVER_K2_SMALL_M): VERIFY then runs the four GDN in-projections as one
`gemm_segments` launch and the lm_head once at m = Σ(drafts + 1) into the rows logits, and speculation must still
equal plain greedy on that same model bit for bit (the small-M kernel is batch invariant).

K8Speculation's lever variants and timing tests outside the per-merge suite (2026-10-18 test audit S1, S4):
K8SpeculationP4 and K8SpeculationSmallMP4 (the `prefill_attention` lever at `p4`) are test_model_k8p4_spec.py (suite
`model_k8p4`, on demand), the report-only SpecCosts timing tests on the small-M model are test_model_k8cal_costs.py
(suite `model_k8cal`, nightly); `native_dev.ON_DEMAND_SUITES` holds when each runs.

The -v22 suite (`test_model_v22_s3.py`, suite `model_v22`) reruns K8Forward and K8Speculation on the shipped profile's
S3 artifact through the class hooks: `LEVERS` (the model's slot selection), K8Forward's `MERGE_BASE` (False: no
merge-base digests, the merge-base library cannot load that checkpoint) and K8Speculation's `draft_token_map` (the
served map), with the checkpoint swapped by `checkpoint_dir`.

Token IDs and text stay in memory; report rows carry counts, scalars and digests only.

Native entry points (the contract with the kernel track; `oracle.call` appends the stream; 0 = success, otherwise a
`cere_status`; "host" buffers are CPU memory, "device" buffers GPU0 memory; the ABI v5 types are those of
`executor/include/cere_executor.h`):

    int cere_ktest_model_create(const cere_model_desc *desc /*host*/, uint64_t memory_budget_bytes,
                                uint64_t *out_model /*host [1]*/, cudaStream_t stream);
        A ktest-owned executor on GPU0 (`cere_executor_create`, budget as given) and `cere_model_create(desc)`;
        writes an opaque handle for the pair. Status as `cere_model_create` (UNSUPPORTED, CAPACITY, ...).
    int cere_ktest_model_put(uint64_t model, uint64_t tensor_index, uint64_t offset, const void *bytes /*host*/,
                             uint64_t len, cudaStream_t stream);
        `cere_model_put`; callable from several threads at once for different tensor_index values.
    int cere_ktest_model_seal(uint64_t model, cudaStream_t stream);
        `cere_model_seal` (exact tiling of every non-discarded tensor, then INSTANTIATE).
    int cere_ktest_model_release(uint64_t model, cudaStream_t stream);
        `cere_model_release` then `cere_executor_destroy`; frees every device and pinned byte of the pair.
    int cere_ktest_generate(uint64_t model, const uint32_t *prompt /*host*/, uint64_t prompt_len,
                            uint64_t max_new_tokens, uint64_t prefill_chunk_tokens, uint64_t cancel_after,
                            uint32_t *out_tokens /*host [max_new_tokens]*/,
                            cere_token_score *out_scores /*host [max_new_tokens]*/,
                            uint64_t *out_result /*host [3]*/, cudaStream_t stream);
        One greedy request through the ABI v5 runtime: `cere_session_create` (the model's context limit), one
        PREFILL `cere_work` carrying the prompt, max_new_tokens and the leased out_tokens/out_scores, submit, poll to
        the terminal completion, `cere_session_release`. prefill_chunk_tokens 0 = the executor's own prefill policy
        (M2.2b: one prefill; M2.2c: chunks of 4,096, Q8), N > 0 = chunks of N tokens (N >= prompt_len: one
        prefill). cancel_after 0 = never; k > 0 = `cere_executor_cancel` is
        issued at the token boundary right after the k-th token is committed (a deterministic stand-in for a client
        cancel arriving between tokens k and k+1). out_result = {committed_tokens, finish (CERE_FINISH_*), device
        bytes in use (cudaMemGetInfo total - free) after the last committed token, before the session is
        released}. Returns 0 for LENGTH, STOP_TOKEN and CANCELLED completions, the completion status otherwise.
    int cere_ktest_generate_flags(uint64_t model, const uint32_t *prompt /*host*/, uint64_t prompt_len,
                                  uint64_t max_new_tokens, uint64_t flags,
                                  uint32_t *out_tokens /*host [max_new_tokens]*/,
                                  cere_token_score *out_scores /*host [max_new_tokens]*/,
                                  uint64_t *out_result /*host [6]*/, cudaStream_t stream);
        M2.9L (tasks/m29l-spec.md C6L): `cere_ktest_generate` (prefill_chunk_tokens 0, cancel_after 0) whose work
        carries `flags` as cere_work.flags; out_result = {committed_tokens, finish, device bytes, verify_steps,
        accepted_drafts} (those two from the terminal completion), then verified_drafts (tasks/m29a-spec.md A5: the
        drafts the session's rounds verified).
    int cere_ktest_generate_sampled_flags(uint64_t model, const uint32_t *prompt /*host*/, uint64_t prompt_len,
                                          uint64_t max_new_tokens, uint64_t flags,
                                          const cere_sampling_params *sampling /*host [1]*/,
                                          uint32_t *out_tokens /*host [max_new_tokens]*/,
                                          cere_token_score *out_scores /*host [max_new_tokens]*/,
                                          uint64_t *out_result /*host [6]*/, cudaStream_t stream);
        Review M1: `cere_ktest_generate_flags` whose work also carries `sampling` (non-null) as cere_work.sampling.
    int cere_ktest_forward_trace(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                 uint64_t prefill_chunk_tokens, uint64_t last_rows,
                                 uint8_t *names /*host*/, uint64_t names_capacity,
                                 int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                 void *values /*device*/, uint64_t values_capacity_bytes,
                                 cere_token_score *scores /*host [rows]*/, uint64_t *out_counts /*host [3]*/,
                                 cudaStream_t stream);
        Teacher-forced forward of `tokens` in a fresh ktest-owned session (released before return), prefilled with
        the same prefill_chunk_tokens semantics as `cere_ktest_generate`. Emits one entry per B8 tensor: its
        NUL-terminated UTF-8 name appended to `names`, {ndim = 2, rows, cols} in `shapes`, and its BF16 row-major
        values appended back to back to `values`. rows = last_rows (the last positions only) or token_count when
        last_rows is 0. scores[r] is the top-2 of the logits at the r-th returned position (that position predicts
        the next token). out_counts = {entries, name bytes, value bytes}. Any capacity too small: CERE_ERR_CAPACITY.
    int cere_ktest_forward_trace_detail(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                        uint64_t prefill_chunk_tokens, uint64_t last_rows, uint32_t detail,
                                        uint8_t *names /*host*/, uint64_t names_capacity,
                                        int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                        void *values /*device*/, uint64_t values_capacity_bytes,
                                        cere_token_score *scores /*host [rows]*/, uint64_t *out_counts /*host [3]*/,
                                        cudaStream_t stream);
        `cere_ktest_forward_trace` with ForwardTaps.detail: 0 = forward_trace; 1 = the model.h ForwardTaps detail
        names too, in forward order between the B8 entries, each {2, rows, its cols}; > 1: CERE_ERR_ARGUMENT.
    int cere_ktest_forward_patch(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                 const uint8_t *names /*host, NUL-separated*/, uint64_t name_bytes,
                                 const void *values /*device BF16, back to back [token_count, cols] per name*/,
                                 uint64_t values_bytes, uint32_t target_token, float *out_target_logit /*host [1]*/,
                                 cere_token_score *out_top1 /*host [1]*/, cudaStream_t stream);
        The prompt alone as one PREFILL (token_count <= the workspace rows) on a fresh session with each named
        patch point (a model.h ForwardTaps P point run once per forward) replaced; the last row's BF16 logit of
        target_token and its top-2. A null or zero argument, an unknown or repeated name, values_bytes other than
        the names' sum or target_token >= vocab_size: CERE_ERR_ARGUMENT, nothing written.
    int cere_ktest_forward_patch_screen(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                        const bf16 *(*patch)(void *ctx, const char *name, uint64_t layer,
                                                             uint64_t rows, uint64_t cols, const bf16 *native,
                                                             cudaStream_t stream),
                                        void *patch_ctx, uint32_t target_token, float *out_target_logit /*host [1]*/,
                                        cere_token_score *out_top1 /*host [1]*/, cudaStream_t stream);
        forward_patch with `patch` as ForwardTaps.patch: called at every I and P point of model.h ForwardTaps
        (native [rows = token_count, cols] on GPU0); a non-null return at a P point replaces it and must stay valid
        until the next callback.
    int cere_ktest_forced_decode_trace(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                       uint64_t prompt_tokens,
                                       uint8_t *names /*host*/, uint64_t names_capacity,
                                       int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                       void *values /*device*/, uint64_t values_capacity_bytes,
                                       cere_token_score *scores /*host [token_count - prompt_tokens]*/,
                                       uint64_t *out_counts /*host [3]: entries, name_bytes, value_bytes*/,
                                       cudaStream_t stream);
        `cere_ktest_forced_decode`'s requests (PREFILL of tokens[0, P), then for j = 1 .. C - 1, C = token_count - P,
        one DECODE of the forced token tokens[P + j - 1]) run by a direct forward loop on one ktest-owned session
        (the session's KV and GDN state carried as submit carries them), the DECODE steps tapped. scores[0] from the
        PREFILL, scores[j] from DECODE step j, bit-equal to `cere_ktest_forced_decode`. Entries are step-major (step
        1 first); each step holds forward_trace's names in forward_trace's order, shape {2, 1, cols}, values as
        forward_trace lays them out; out_counts sum over the steps. The trace's size per step comes from one tapped
        single-token probe forward on a throwaway session before the requests. Refusals as
        `cere_ktest_forced_decode` (incl. a forced token >= vocab_size) plus any capacity below the trace's size:
        CERE_ERR_ARGUMENT before the PREFILL, out_counts zero, nothing written.
    int cere_ktest_forward_decode_rows(uint64_t model, uint64_t sessions, const uint32_t *prompts /*host*/,
                                       const uint64_t *prompt_lens /*host [S]*/, const uint64_t *join /*host [S]*/,
                                       const uint64_t *leave /*host [S]*/, const uint32_t *forced /*host [S * T]*/,
                                       cere_token_score *scores /*host [S * (T + 1)]*/,
                                       uint8_t *names /*host*/, uint64_t names_capacity,
                                       int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                       void *values /*device*/, uint64_t values_capacity_bytes,
                                       uint64_t *out_counts /*host [3]*/, cudaStream_t stream);
        tasks/m26-spec.md I7: S = `sessions` ktest-owned sessions (1 <= S <= the plan's max_sessions) run rounds
        t = 0 .. T - 1, T = max(leave). In round t each session with join[s] == t prefills its prompt (prompts =
        the S prompts back to back) alone, then every session with join[s] <= t < leave[s] feeds one token in ONE
        batched decode step, rows in session order (CudaDevice::forward_rows = model::forward_decode_rows for two
        or more rows, the single-row DECODE forward for one, as step_rows runs them). Its k-th decode (k = 1 ..
        leave - join) feeds forced[s * T + k - 1] (forced non-null) or its own greedy top-1 and writes
        scores[s * (T + 1) + k]; scores[s * (T + 1)] comes from the prefill; other rows stay untouched. With names
        non-null its decode k's B8 taps (its row's slice) are step s * T + k - 1 of a trace laid out as
        forced_decode_trace's (per step: names, {2, 1, cols} shapes, values; out_counts = one step's entries, name
        bytes, value bytes). Malformed sizes, a token >= vocab_size or a capacity below S * T steps:
        CERE_ERR_ARGUMENT before any forward.
    int cere_ktest_generate_rows(uint64_t model, uint64_t requests, const uint32_t *prompts /*host*/,
                                 const uint64_t *prompt_lens /*host [N]*/, const uint64_t *max_new_tokens /*host [N]*/,
                                 uint64_t prefill_chunk_tokens, uint32_t *out_tokens /*host [N][stride]*/,
                                 cere_token_score *out_scores /*host [N][stride]*/, uint64_t out_stride,
                                 uint64_t *out_results /*host [N][2]*/, cudaStream_t stream);
        N = `requests` greedy requests (cere_ktest_generate's, each on a session of its own) submitted in order,
        then polled in request order until all complete, so the executor's rounds (M2.6 D2) interleave one prefill
        chunk of prefill_chunk_tokens (cere_ktest_generate's semantics) with one batched decode round of every
        decode-ready request. out_results[i] = {committed_tokens, finish}. A completion other than LENGTH or
        STOP_TOKEN cancels and drains the rest and returns its status.
    int cere_ktest_generate_rows_sampled(uint64_t model, uint64_t requests, const uint32_t *prompts /*host*/,
                                         const uint64_t *prompt_lens /*host [N]*/,
                                         const uint64_t *max_new_tokens /*host [N]*/, uint64_t prefill_chunk_tokens,
                                         const cere_sampling_params *sampling /*host [N]*/,
                                         const uint32_t *top_logprobs /*host [N]*/,
                                         uint32_t *out_tokens /*host [N][stride]*/,
                                         cere_token_score *out_scores /*host [N][stride]*/,
                                         cere_token_logprobs *out_logprobs /*host [N][stride]*/, uint64_t out_stride,
                                         uint64_t *out_results /*host [N][2]*/,
                                         uint64_t *out_device_bytes /*host [2]*/, cudaStream_t stream);
        cere_ktest_generate_rows with per-request sampling (ledger CR4): sampling[i] is request i's
        cere_work.sampling (struct_size 0 = greedy); with out_logprobs non-null a sampled request also gets its
        out_logprobs row and top_logprobs[i]. out_device_bytes = the device bytes in use after the sessions are
        created (before the first submit) and after the last completion.
    int cere_ktest_sampler_plan_bytes(uint64_t model, uint64_t *out /*host [4]*/, cudaStream_t stream);
        Host arithmetic: out = {K12 sampler scratch bytes (one per model), processed-logits row bytes (one per
        model), one session's out_counts bytes, the plan's max_sessions}.
    int cere_ktest_forced_verify_trace(uint64_t model, const uint32_t *tokens, uint64_t token_count,
                                       uint64_t prompt_tokens, uint64_t chain, uint8_t *names,
                                       uint64_t names_capacity, int64_t *shapes, uint64_t entry_capacity,
                                       void *values, uint64_t values_capacity_bytes, cere_token_score *scores,
                                       uint64_t *out_counts, cudaStream_t stream);
        M2.9 K8: `cere_ktest_forced_decode_trace` with the C - 1 DECODE steps fed as CudaDevice::verify_rows calls of
        up to chain + 1 rows (chain <= 3), every row committed; same layout, capacities and refusals. Needs a model
        with speculative_draft_tokens (else CERE_ERR_UNSUPPORTED).
    int cere_ktest_session_open(uint64_t model, uint64_t max_context_tokens, uint64_t *out_session,
                                cudaStream_t stream);
    int cere_ktest_session_feed(uint64_t session, const uint32_t *tokens, uint64_t n, uint64_t max_new_tokens,
                                uint64_t cancel_after, uint8_t *names, uint64_t names_capacity, int64_t *shapes,
                                uint64_t entry_capacity, void *values, uint64_t values_capacity_bytes,
                                cere_token_score *scores, uint64_t *out_counts, cudaStream_t stream);
    int cere_ktest_session_close(uint64_t handle, cudaStream_t stream);
    int cere_ktest_state_digest(uint64_t session, uint8_t *out, cudaStream_t stream);
        The K15 session entries (test_model_k15_snapshot.py): K8Speculation uses the teacher-forced feed
        (max_new_tokens 0, no names) and the state digest.
    int cere_ktest_session_draft(uint64_t session, uint64_t on, cudaStream_t stream);
        CudaDevice::set_session_draft on an empty session.
    int cere_ktest_session_verify(uint64_t session, const uint32_t *tokens, uint64_t n, uint64_t commit,
                                  cere_token_score *scores, cudaStream_t stream);
        One verify chain (tokens[0] fed at the session's position, drafts tokens[1, n], n <= 3) committing `commit`
        in [1, n + 1] rows whatever their scores; scores host [n + 1].
    int cere_ktest_session_speculate(const uint64_t *sessions, uint64_t rows, const uint32_t *tokens,
                                     const uint32_t *drafts, const uint32_t *draft_counts,
                                     const uint32_t *max_commit, int32_t *row_status, uint32_t *results,
                                     cere_token_score *scores, cudaStream_t stream);
        One CudaDevice::speculate_rows call (greedy, no stop token) of drafting sessions of one model (host arrays;
        drafts [rows, 3]); results [rows, 2] = committed, accepted; scores [rows, 4] the committed tokens'.
    int cere_ktest_session_speculate_sampled(<cere_ktest_session_speculate's arguments up to max_commit>,
                                             const cere_sampling_params *sampling, const int32_t *tops,
                                             int32_t *row_status, uint32_t *results, cere_token_score *scores,
                                             cere_token_logprobs *logprobs, cudaStream_t stream);
        As cere_ktest_session_speculate with row r's sampling[r] (struct_size 0 = greedy) and tops[r] (-1 = no
        logprobs, else top_logprobs; logprobs host [rows, 4] then gets the committed tokens' entries).
    int cere_ktest_session_speculate_masked(<cere_ktest_session_speculate_sampled's arguments up to tops>,
                                            const uint32_t *masks, int32_t *row_status, uint32_t *results,
                                            cere_token_score *scores, cere_token_logprobs *logprobs,
                                            cudaStream_t stream);
        As cere_ktest_session_speculate_sampled (every row sampled) with row r's verify row j under the vocab mask
        masks[r, j] (host [rows, 4, ceil(vocab / 32)]; GrammarSpec GS-02); drafts null = the pending MTP proposals.
    int cere_ktest_session_speculate_mtp(const uint64_t *sessions, uint64_t rows, const uint32_t *tokens,
                                         const uint32_t *draft_counts, const uint32_t *max_commit,
                                         int32_t *row_status, uint32_t *results, cere_token_score *scores,
                                         cudaStream_t stream);
        As cere_ktest_session_speculate with every row drafting from its pending MTP proposals (at most
        draft_counts[r] of them).
    int cere_ktest_session_speculate_chain(<cere_ktest_session_speculate's arguments>);
        As cere_ktest_session_speculate with deep chains (DraftAccept): drafts [rows, 8], draft_counts[r] <= 8,
        scores [rows, 9].
    int cere_ktest_session_step_sampled(uint64_t session, uint32_t token, const cere_sampling_params *sampling,
                                        int32_t top, cere_token_score *score, cere_token_logprobs *logprobs,
                                        float *processed, cudaStream_t stream);
        One CudaDevice::step_rows row (plain decode) feeding `token` with `sampling`: its score, logprobs (top >= 0)
        and the sampler stage's processed logits (host fp32 [vocab], nullable).
    int cere_ktest_session_step_masked(uint64_t session, uint32_t token, const cere_sampling_params *sampling,
                                       int32_t top, const uint32_t *mask, cere_token_score *score,
                                       cere_token_logprobs *logprobs, float *processed, cudaStream_t stream);
        As cere_ktest_session_step_sampled (sampled) under the vocab mask `mask` (host [ceil(vocab / 32)]).
    int cere_ktest_spec_stages(uint64_t session, uint64_t rows, float *processed, float *probs, uint32_t *keep,
                               cudaStream_t stream);
        The verify rows' sampler stages of the model's last speculate_rows call (host, nullable each).
    int cere_ktest_session_counts(uint64_t session, uint32_t *out, cudaStream_t stream);
        The session's out_counts (kind 8), host uint32 [vocab].
    int cere_ktest_decode_graph(uint64_t model, int32_t mode, uint64_t *out /*host [4]*/,
                                cudaStream_t /*stream*/);
        As stated in test_graph_decode.py (M2.3 P1): mode 0 / 1 switches the model's decode graphs off / on for its
        later steps, -1 leaves them; out = {enabled, graphs instantiated, executables updated in place, steps run
        by a graph}.
    int cere_ktest_verify_attention(uint64_t model, int32_t mode, uint64_t *out /*host [2]*/,
                                    cudaStream_t /*stream*/);
        tasks/m23-spec.md P11R-19: mode 0 / 1 switches the model's verify_attention lever to per_row / batched for its
        later VERIFY calls, -1 leaves it; out = {batched, attention_chain calls (one per batched chain and
        full-attention layer) since INSTANTIATE}.
    int cere_ktest_spec_graph(uint64_t model, int32_t mode, uint64_t *out /*host [7]*/, cudaStream_t /*stream*/);
        tasks/m29-spec.md M29S2-18b, tasks/m23-spec.md P11R-20: mode 0 / 1 switches the model's spec graphs
        (decode_step = spec_graph: verify and MTP draft graphs) off / on for its later VERIFY calls and drafts, -1
        leaves them; out = {on, verify graphs instantiated, verify executables updated in place, VERIFY calls run by
        a graph, draft graphs instantiated, draft executables updated in place, drafts run by a graph}.
    int cere_ktest_draft_digest(uint64_t session, uint8_t *out, cudaStream_t stream);
        cere_ktest_state_digest of a drafting session plus its kind 10 section (the MTP pool entry and the pending
        proposals); on a model without drafts the device's save refuses it.
"""
from __future__ import annotations

import contextlib
import ctypes
import functools
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import NamedTuple

import pynvml
import torch
from safetensors import safe_open
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from torch.profiler import ProfilerActivity, profile

import base_digests
import oracle

ROW = "K8"
SRC = Path(__file__).resolve().parents[3]
PARITY_FILE = SRC / "bench/prompts/parity_v1.jsonl"
FAMILIES_FILE = SRC / "configs/artifacts/qwen4_exp-nvfp4.families.json"
PINNED_MODEL_DIR = oracle.MODEL_DIR  # the mounted pinned checkpoint, before any `checkpoint_dir` swap
PARITY_IDS = ("p01", "p02", "p13", "p24")
REPEATS = 3
MAX_CONTEXT = 2048
CANCEL_AT = 10
TRACE_PROMPTS = 3
REFUSAL_COMPLETION = 3
CERE_ERR_ARGUMENT = -1  # cere_executor.h
MEMORY_SLACK = 256 << 20
SMOKE_MESSAGES = [{"role": "user", "content": "Say OK."}]
# Prefill tap rows (m22p design note L1, L2).
PREFILL_IDS = ("p12", "p13")
BASE_KTEST_LIB = Path(os.environ.get("CERE_BASE_KTEST_LIB", str(SRC / "build/native-l0/libcere_ktest.so")))
DETAIL_ROWS = 64  # rows reported by the detail-1 row (its taps cost ~7.5 MB per row)
REFUSAL_PROMPT = 32  # tokens of p12 the refusal row runs
L2_SEED = 20261010
L2_PREFILLS = (1, 2, 4, 8, 16)
L2_DECODE_PROMPT = 16
L2_DECODE_STEPS = 16
# ForwardTaps.patch: const bf16 *(*)(void *ctx, const char *name, uint64_t layer, uint64_t rows, uint64_t cols,
#                                    const bf16 *native, cudaStream_t stream)
PATCH_FN = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64,
                            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)
INPUT_SUFFIXES = (".in", ".normed")  # model.h ForwardTaps I points; every other patch callback is a P point

# M2.6 rows (m26 design note D3, K3, K5).
ROWS_R = 4  # the K8 model's max_sessions: the batched decode step runs up to 4 rows
ROWS_SEED = 20261026
ROWS_STEPS = 64
ROWS_PROMPT_LENS = (37, 211, 512, 1029)  # distinct histories; + ROWS_STEPS stays within MAX_CONTEXT
ROWS_JOIN_LEAVE = ((0, 64), (8, 64), (0, 64), (0, 40))  # K3: join at step 8, leave at step 40
ROWS_TAP_PREFIX = "model.layers.1."  # the PLE layer (K3: per-row PLE taps equal solo)
K5_CHUNK = 512
K5_LONG = 3 * K5_CHUNK - 100  # three chunks
K5_SHORT_IDS = ("p01", "p02")  # real prompts: random synthetic ones stopped after 2 tokens (R1 20261010T215720Z)
K5_SHORT_NEW = 32  # the short requests decode through the long prompt's chunk rounds
K5_LONG_NEW = 16
K5_MIN_INTERLEAVED = 4  # committed tokens a short request needs to decode between all three chunks
# Ledger CR4 (M2.6 x M2.7 sampling rows): I10 rows 2-4 sampler invariance and rows-4 sampler memory (ledger W).
CR4_NEW = 32
CR4_MIN_COMMITTED = 4  # tokens a request needs alone so its sampled rows share decode rounds
CR4_TOP_LOGPROBS = 5
CR4_SAMPLER_BYTES = 8 << 20  # ledger W: sampler memory at rows 4
# The rows-4 memory run's control: a second process on GPU0 allocates this much inside the measured window, as the
# desktop's processes do (GPU0 drives the display), so a measure that sees other processes fails.
CR4_FOREIGN_BYTES = 64 << 20
# Per request (PARITY_IDS order): (temperature, top_p, top_k (0 = vocab), frequency, presence, repetition, seed,
# logprobs), or None for a greedy request (null sampling).
CR4_SAMPLING = (
    (1.0, 0.95, 20, 0.0, 0.0, 1.0, 20261011, True),
    None,
    (0.8, 0.9, 0, 0.5, 0.3, 1.1, 20261012, False),
    (0.0, 1.0, 0, 0.0, 0.0, 1.0, 20261013, True),
)
# M2.9 K8Speculation (m29 design note D4, D5, M29S2-20).
CERE_LEVER_K2_SMALL_M = 1  # the k2_gemm slot: the P11 small-M kernel for m <= 16 (K8SpeculationSmallM)
# the k2_gemm slot: small_m plus MXFP8 W8A8 (E4M3 activations, UE8M0 per 32 k) on MXFP8 weights at m > 16
# (numerics-changing, opt-in -v56; k2-prefill design note P3)
CERE_LEVER_K2_SMALL_M_A8 = 2
CERE_LEVER_K2_WEIGHTS_FP8 = 1  # the k2_weights slot: E4M3 dense K2 weights + FP32 per-row scales (Fp8W, restart)
SPEC_DRAFT_TOKENS = 4  # cere_model_desc.speculative_draft_tokens: 1 + 3 drafts per verify chain
SPEC_SESSIONS = 4
SPEC_CONTEXT = 4096  # above k6::kMaxContext (2,048): the K9 QSA path with its ring and compressed keys
SPEC_SEED = 20261012
SPEC_UNDO_PROMPT = 61
SPEC_UNDO_ROUNDS = 6
SPEC_DECODE = 256
SPEC_PROMPT_LENS = (37, 211, 512, 1029)
SPEC_WRONG_ROWS = (0, 1, 2, None)  # the draft made wrong (rejection at row w + 1), None: every draft right
SPEC_QSA_PROMPT = 2040
SPEC_QSA_END = 2112  # positions the K7 run reaches (past 2,048 and 2,100)
SPEC_CHAIN_PROMPT = 1029  # DraftAccept deep chains below 2,048 (dense K6; the verify graph on a spec_graph model)
SPEC_CHAIN_STEPS = 128
# drafts per chain, cycled (3: a shallow chain between deep ones); Chains (O123): up to CudaModel::max_chain_drafts,
# 12 on the K8 model (max_sessions 4: a 12-snapshot undo pool), so chains complete up to three compressed key groups
SPEC_CHAIN_LENS = (12, 7, 10, 5, 8, 3, 11, 9, 6)
SPEC_CHAIN_WRONG = (None, 7, 5, 4, 6, None, 0, 2, 3, 10, 1, 4, 9, 11, 8)  # the draft made wrong, cycled (None: right)
SPEC_SPARSE_PROMPT = 3001  # LK-01: every verify row past 2,048 with > 512 visible blocks (K9's top-512 radix select)
SPEC_SPARSE_STEPS = 96
SPEC_TRACE_PROMPT = 40
SPEC_TRACE_COMPLETION = 12  # 11 decode steps: verify calls of 4, 4 and 3 rows
# P11R-20 (decode_step = spec_graph): (draft count n, commit) per caller verify round; each n in 0..3 at least
# three times, so every verify-graph shape is captured and replayed at later positions.
SPEC_GRAPH_ROUNDS = ((0, 1), (1, 1), (1, 2), (2, 3), (3, 4), (3, 1), (2, 1), (0, 1), (3, 2), (1, 2), (2, 2), (3, 3),
                     (3, 4), (0, 1), (2, 3), (1, 1))
# M2.9 phase S (m29 design note D6, M29S2-13/-14): CR4_SAMPLING-shaped entries.
SPEC_SAMPLED = (
    (0.7, 0.95, 20, 0.0, 0.0, 1.0, 0x2545F4914F6CDD1D, True),
    (1.0, 0.9, 50, 0.0, 0.0, 1.0, 20261012, False),
)
SPEC_SAMPLED_ROUNDS = 8
SPEC_TOP1 = (1.0, 1.0, 1, 0.0, 0.0, 1.0, 20261014, False)  # p(d) = 1 for d = the argmax
SPEC_T0 = (0.0, 1.0, 0, 0.5, 0.3, 1.1, 20261013, True)  # temperature 0 with penalties: the processed top-1
SPEC_T0_STEPS = 64
# GrammarSpec GS-02 (grammar-spec design note): constrained (masked) verify rows. A constrained greedy item is a
# temperature-0 sampled one with neutral processors (every constrained item takes the sampled path); `position_mask`
# bans a third of the vocab per position.
SPEC_MASKED_T0 = (0.0, 1.0, 0, 0.0, 0.0, 1.0, 20261016, False)
SPEC_MASKED_STEPS = 64
SPEC_MASK_MODULUS = 3

# M2.9L C6L (m29l design note): cere_work.flags bits 2 and 3, and the repeated-block prompts the suffix lookup matches.
WORK_LOOKUP_DRAFTS = 0x4
WORK_NO_MTP_DRAFTS = 0x8
SPEC_LOOKUP_ARMS = (("mtp", 0), ("mtp+lookup", WORK_LOOKUP_DRAFTS), ("lookup", WORK_LOOKUP_DRAFTS | WORK_NO_MTP_DRAFTS))
SPEC_LOOKUP_SHAPES = ((29, 61, 5), (211, 97, 4))  # (random prefix, block, repeats): prompts of 334 and 599 ids
# m29a design note: cere_work.flags bit 4 (the adaptive draft policy) and its arms.
WORK_ADAPTIVE_DRAFTS = 0x10
SPEC_ADAPTIVE_ARMS = (("fixed", 0), ("adaptive", WORK_ADAPTIVE_DRAFTS),
                      ("adaptive+lookup", WORK_ADAPTIVE_DRAFTS | WORK_LOOKUP_DRAFTS))
# TreeSpec (tree-spec design note TS-02): bit 5 (DraftAccept tail) and bit 6 (long lookup chains, up to 8 drafts).
WORK_TAIL_DRAFTS = 0x20
WORK_LONG_DRAFTS = 0x40
# Chains (chains design note, O123): bit 7, the chains of bits 5 and 6 up to the model's max_chain_drafts (12 here).
WORK_CHAIN_DRAFTS = 0x80
SPEC_TAIL_MAX = 8  # DraftPolicy::kMaxTailDrafts: the longest chain of bits 5 and 6 without bit 7
SPEC_LONG_ARMS = (("long", WORK_ADAPTIVE_DRAFTS | WORK_LOOKUP_DRAFTS | WORK_LONG_DRAFTS),
                  ("mtp+lookup+long", WORK_ADAPTIVE_DRAFTS | WORK_LOOKUP_DRAFTS | WORK_TAIL_DRAFTS | WORK_LONG_DRAFTS),
                  ("mtp+lookup+chain", WORK_ADAPTIVE_DRAFTS | WORK_LOOKUP_DRAFTS | WORK_TAIL_DRAFTS | WORK_LONG_DRAFTS
                   | WORK_CHAIN_DRAFTS))
# Review M1 (m29a design note A6 [REVISED: review-wave-fg M1]): a seeded sampled request (CR4_SAMPLING-shaped,
# neutral processors, so it speculates), and the cere_work.flags the runtime sends for such a ticket under
# `draft_policy = "adaptive"` (crates/cere-runtime/src/native/serve.rs draft_flags: bit 4 is set on greedy tickets
# only, so a sampled one gets the fixed policy's 0; `main` `3fa2aefd` sent WORK_ADAPTIVE_DRAFTS).
SPEC_SAMPLED_SEEDED = (0.8, 0.95, 0, 0.0, 0.0, 1.0, 20261015, False)
SPEC_SAMPLED_ADAPTIVE_RUNTIME_FLAGS = 0

# -- ABI v5 (executor/include/cere_executor.h), mirrored with ctypes ---------------------------------------------

ABI_VERSION = 5
MODEL_QWEN4_EXP_NVFP4 = 1
PLACEMENT_DEVICE = 1
PLACEMENT_HOST_PINNED = 2
PLACEMENT_DISCARD = 3
DTYPES = {"BF16": (1, 2), "F16": (2, 2), "F32": (3, 4), "F64": (4, 8), "F8_E4M3": (5, 1), "F8_E5M2": (6, 1),
          "U8": (7, 1), "I8": (8, 1), "I16": (9, 2), "I32": (10, 4), "I64": (11, 8), "BOOL": (12, 1)}
SSM_DTYPES = {"bfloat16": 1, "float16": 2, "float32": 3}
OUTPUT_GATES = {"sigmoid": 1}
FINISH_LENGTH = 1
FINISH_STOP_TOKEN = 2
FINISH_CANCELLED = 3
MAX_DIMS = 8
PUT_PIECE = 64 << 20

U32, U64, F64 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_double


class Qwen4ExpConfig(ctypes.Structure):
    _fields_ = [("struct_size", U32), ("abi_version", U32), ("vocab_size", U32), ("hidden_size", U32),
                ("num_hidden_layers", U32), ("full_attention_interval", U32), ("full_attention_layers", U64),
                ("num_attention_heads", U32), ("num_key_value_heads", U32), ("head_dim", U32),
                ("output_gate_type", U32), ("rope_theta", F64), ("partial_rotary_factor", F64),
                ("mrope_interleaved", U32), ("mrope_section", U32 * 3), ("indexer_n_heads", U32),
                ("indexer_head_dim", U32), ("indexer_kv_heads", U32), ("indexer_compress_ratio", U32),
                ("indexer_budget", U32), ("linear_num_key_heads", U32), ("linear_num_value_heads", U32),
                ("linear_key_head_dim", U32), ("linear_value_head_dim", U32), ("linear_conv_kernel_dim", U32),
                ("mamba_ssm_dtype", U32), ("num_experts", U32), ("num_experts_per_tok", U32),
                ("moe_intermediate_size", U32), ("shared_expert_intermediate_size", U32), ("norm_topk_prob", U32),
                ("hc_count", U32), ("hc_lowrank", U32), ("ple_layer_ids", U64), ("ple_embed_dim", U32),
                ("heads_per_ngram", U32), ("ngram_size", U32), ("ple_conv_kernel_size", U32),
                ("ngram_vocab_size_base", U64), ("split_ngram_parts", U32),
                ("make_ngram_vocab_size_divisible_by", U32), ("rms_norm_eps", F64), ("tie_word_embeddings", U32),
                ("mtp_num_hidden_layers", U32), ("max_position_embeddings", U64)]


class TensorDesc(ctypes.Structure):
    _fields_ = [("struct_size", U32), ("abi_version", U32), ("name", ctypes.c_void_p), ("dtype", U32),
                ("placement", U32), ("ndim", U32), ("reserved", U32), ("shape", U64 * MAX_DIMS), ("bytes", U64)]


class ModelDesc(ctypes.Structure):
    _fields_ = [("struct_size", U32), ("abi_version", U32), ("model_kind", U32), ("speculative_draft_tokens", U32),
                ("config", ctypes.c_void_p), ("tensors", ctypes.c_void_p), ("tensor_count", U64),
                ("max_sessions", U64), ("max_context_tokens", U64), ("stop_token_ids", ctypes.c_void_p),
                ("stop_token_count", U64), ("kv_pool_tokens", U64), ("draft_token_map", ctypes.c_void_p),
                ("draft_token_map_count", U64), ("snapshot_host_bytes", U64), ("prefix_state_slots", U64),
                ("vision_config", ctypes.c_void_p), ("selection", ctypes.c_void_p), ("family", U32),
                ("calibrate_windows", U32)]


class SamplingParams(ctypes.Structure):
    _fields_ = [("struct_size", U32), ("abi_version", U32), ("temperature", ctypes.c_float),
                ("top_p", ctypes.c_float), ("min_p", ctypes.c_float), ("top_k", U32),
                ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
                ("repetition_penalty", ctypes.c_float), ("flags", U32), ("seed", U64), ("logit_bias", ctypes.c_void_p),
                ("logit_bias_count", U64), ("suppress_token_ids", ctypes.c_void_p), ("suppress_token_count", U64),
                ("mask_until_committed", U64)]


TOKEN_LOGPROBS_BYTES = 16 + 8 * 20  # cere_token_logprobs: id, logprob, top_count, reserved, top[CERE_MAX_TOP_LOGPROBS]


# Plain integer fields of cere_qwen4_exp_config, copied from text_config (an absent key is 0, header rule).
CONFIG_INT_KEYS = (
    "vocab_size", "hidden_size", "num_hidden_layers", "full_attention_interval", "num_attention_heads",
    "num_key_value_heads", "head_dim", "indexer_n_heads", "indexer_head_dim", "indexer_kv_heads",
    "indexer_compress_ratio", "indexer_budget", "linear_num_key_heads", "linear_num_value_heads",
    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok",
    "moe_intermediate_size", "shared_expert_intermediate_size", "hc_count", "hc_lowrank", "ple_embed_dim",
    "heads_per_ngram", "ngram_size", "ple_conv_kernel_size", "ngram_vocab_size_base", "split_ngram_parts",
    "make_ngram_vocab_size_divisible_by", "tie_word_embeddings", "mtp_num_hidden_layers", "max_position_embeddings",
)


def struct_buffer(ctype, count: int = 1):
    """A zeroed CPU tensor holding `count` ctypes `ctype` values, and the ctypes view over it (the tensor is what
    `oracle.call` passes, so the struct bytes and the pointer stay one object)."""
    buf = torch.zeros(ctypes.sizeof(ctype) * count, dtype=torch.uint8)
    view = (ctype * count).from_address(buf.data_ptr())
    return buf, view


def a4_config(tc: dict):
    """cere_qwen4_exp_config from text_config by the header's rules (lists as bitmasks, strings as enums)."""
    buf, (cfg,) = struct_buffer(Qwen4ExpConfig)
    cfg.struct_size = ctypes.sizeof(Qwen4ExpConfig)
    cfg.abi_version = ABI_VERSION
    for key in CONFIG_INT_KEYS:
        setattr(cfg, key, int(tc.get(key, 0)))
    cfg.full_attention_layers = sum(1 << i for i, kind in enumerate(tc["layer_types"]) if kind == "full_attention")
    gate = tc.get("output_gate_type")
    if gate is not None and gate not in OUTPUT_GATES:
        raise oracle.OracleError(f"output_gate_type {gate!r} has no CERE_OUTPUT_GATE_* value")
    cfg.output_gate_type = OUTPUT_GATES.get(gate, 0)
    rope = tc["rope_parameters"]
    cfg.rope_theta = float(rope["rope_theta"])
    cfg.partial_rotary_factor = float(rope["partial_rotary_factor"])
    cfg.mrope_interleaved = int(bool(rope.get("mrope_interleaved", False)))
    for i, value in enumerate(rope["mrope_section"]):
        cfg.mrope_section[i] = int(value)
    ssm = tc.get("mamba_ssm_dtype")
    if ssm is not None and ssm not in SSM_DTYPES:
        raise oracle.OracleError(f"mamba_ssm_dtype {ssm!r} has no CERE_DTYPE_* value")
    cfg.mamba_ssm_dtype = SSM_DTYPES.get(ssm, 0)
    cfg.norm_topk_prob = int(bool(tc.get("norm_topk_prob", True)))
    cfg.ple_layer_ids = sum(1 << int(n) for n in tc.get("ple_layer_ids", []))
    cfg.rms_norm_eps = float(tc["rms_norm_eps"])
    return buf


@contextlib.contextmanager
def checkpoint_dir(model_dir: Path, families: Path):
    """The checkpoint swapped for the block: the directory the oracle and k8 read (oracle.MODEL_DIR) and the families
    file that places its tensors (oracle.FAMILIES_FILE and k8's FAMILIES_FILE), their cached reads (text config,
    weight map, MTP weights and rope, tokenizer) cleared on entry and on exit. A candidate checkpoint (suite
    `quality`) and the shipped profile's artifact (suite `model_v22`) load through it."""
    global FAMILIES_FILE
    saved = oracle.MODEL_DIR, oracle.FAMILIES_FILE, FAMILIES_FILE

    def clear():
        for cached in (oracle.text_config, oracle._weight_map, oracle.mtp_weights, oracle.mtp_rope, tokenizer):
            cached.cache_clear()
    oracle.MODEL_DIR, oracle.FAMILIES_FILE, FAMILIES_FILE = model_dir, families, families
    clear()
    try:
        yield
    finally:
        oracle.MODEL_DIR, oracle.FAMILIES_FILE, FAMILIES_FILE = saved
        clear()


def tensor_placements(draft: bool = False) -> list:
    """(compiled pattern, placement) per rule of the artifact's family file, in rule order; with `draft` the draft
    family (`mtp.*`) is DEVICE, as cered places it for `speculative_draft_tokens` 4 (tasks/m29-spec.md D3)."""
    rules = json.loads(FAMILIES_FILE.read_text())["rules"]
    out = []
    for rule in rules:
        placement = PLACEMENT_DISCARD
        if rule["placement"] == "unused":
            # Out of the desc, as cered's model spec leaves it (a text-only desc carries no vision tensor, C7).
            placement = None
        elif draft and rule["component"] == "draft":
            placement = PLACEMENT_DEVICE
        elif rule["component"] == "target":
            placement = {"device": PLACEMENT_DEVICE, "host_pinned": PLACEMENT_HOST_PINNED}[rule["placement"]]
        out.append((re.compile(rule["pattern"]), placement))
    return out


class TableEntry(NamedTuple):
    name: str
    file: str
    placement: int
    nbytes: int


def a4_tensor_table(draft: bool = False):
    """Every checkpoint tensor in index (name) order but the family file's `unused` ones: table entries and the
    cere_tensor_desc array over them (`draft`: tensor_placements')."""
    weight_map = json.loads((oracle.MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    rules = tensor_placements(draft)
    by_file: dict[str, list[str]] = {}
    for name in weight_map:
        by_file.setdefault(weight_map[name], []).append(name)
    header: dict[str, tuple[str, list[int]]] = {}
    for file, names in by_file.items():
        with safe_open(str(oracle.MODEL_DIR / file), framework="pt", device="cpu") as f:
            for name in names:
                piece = f.get_slice(name)
                header[name] = (piece.get_dtype(), list(piece.get_shape()))
    placements = {}
    for name in sorted(weight_map):
        match = next(((p,) for pattern, p in rules if pattern.fullmatch(name)), None)
        if match is None:
            raise oracle.OracleError(f"{name} matches no rule of {FAMILIES_FILE.name}")
        if match[0] is not None:
            placements[name] = match[0]
    names = list(placements)
    blob = b"".join(n.encode() + b"\0" for n in names)
    name_buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    desc_buf, descs = struct_buffer(TensorDesc, len(names))
    entries = []
    offset = 0
    for i, name in enumerate(names):
        placement = placements[name]
        dtype, shape = header[name]
        if dtype not in DTYPES:
            raise oracle.OracleError(f"{name}: dtype {dtype} has no CERE_DTYPE_* value")
        if len(shape) > MAX_DIMS:
            raise oracle.OracleError(f"{name}: rank {len(shape)} exceeds CERE_TENSOR_MAX_DIMS")
        code, size = DTYPES[dtype]
        nbytes = size
        for dim in shape:
            nbytes *= dim
        d = descs[i]
        d.struct_size = ctypes.sizeof(TensorDesc)
        d.abi_version = ABI_VERSION
        d.name = name_buf.data_ptr() + offset
        d.dtype = code
        d.placement = placement
        d.ndim = len(shape)
        for k, dim in enumerate(shape):
            d.shape[k] = dim
        d.bytes = nbytes
        offset += len(name.encode()) + 1
        entries.append(TableEntry(name, weight_map[name], placement, nbytes))
    return entries, desc_buf, name_buf


def stop_token_ids() -> list[int]:
    eos = json.loads((oracle.MODEL_DIR / "generation_config.json").read_text())["eos_token_id"]
    return [int(t) for t in (eos if isinstance(eos, list) else [eos])]


def upload_file(handle: int, file: str, items: list) -> None:
    """Puts every non-discarded tensor of one safetensors file in pieces of PUT_PIECE bytes."""
    with safe_open(str(oracle.MODEL_DIR / file), framework="pt", device="cpu") as f:
        for index, entry in items:
            flat = f.get_tensor(entry.name).contiguous().reshape(-1).view(torch.uint8)
            if flat.numel() != entry.nbytes:
                raise oracle.OracleError(f"{entry.name}: {flat.numel()} bytes read, header says {entry.nbytes}")
            for offset in range(0, entry.nbytes, PUT_PIECE):
                piece = flat[offset:offset + PUT_PIECE]
                oracle.call("cere_ktest_model_put", handle, index, offset, piece, piece.numel())


class A4Model(NamedTuple):
    handle: int
    keep: tuple  # the descriptor buffers (the executor copies them, they only need to outlive create)
    desc_size: int  # the cere_model_desc struct_size the library accepted (recorded in the merge-base digests)


def create_model(max_context_tokens: int, max_sessions: int = 1, *, drafts: int = 0,
                 model_levers: oracle.Levers | None = None, draft_token_map: torch.Tensor | None = None) -> A4Model:
    """ALLOCATE (create), UPLOAD (concurrent puts), INSTANTIATE (seal) of the real A4 model on GPU0, with this
    build's ABI v5 desc (a merge-base library is of the same ABI). `drafts`: the desc's speculative_draft_tokens, with
    `draft_token_map` (int32 ids, the served map) as the draft token map, every vocab id when None. `model_levers`:
    the desc's cere_slot_selection (ABI v5; None = null, every slot at its default)."""
    tc = oracle.text_config()
    config_buf = a4_config(tc)
    entries, desc_buf, name_buf = a4_tensor_table(drafts != 0)
    stops = torch.tensor(stop_token_ids(), dtype=torch.int32)
    model_buf, (desc,) = struct_buffer(ModelDesc)
    desc.abi_version = ABI_VERSION
    desc.model_kind = MODEL_QWEN4_EXP_NVFP4
    desc.config = config_buf.data_ptr()
    desc.tensors = desc_buf.data_ptr()
    desc.tensor_count = len(entries)
    desc.max_sessions = max_sessions
    desc.max_context_tokens = max_context_tokens
    desc.stop_token_ids = stops.data_ptr()
    desc.stop_token_count = stops.numel()
    if not drafts:
        draft_map = torch.zeros(0, dtype=torch.int32)
    elif draft_token_map is None:
        draft_map = torch.arange(int(tc["vocab_size"]), dtype=torch.int32)
    else:
        draft_map = draft_token_map
    if drafts:
        desc.speculative_draft_tokens = drafts
        desc.draft_token_map = draft_map.data_ptr()
        desc.draft_token_map_count = draft_map.numel()
    if model_levers is not None:
        desc.selection = ctypes.addressof(model_levers)
    out = torch.zeros(1, dtype=torch.int64)
    free, _ = torch.cuda.mem_get_info(oracle.DEVICE)
    desc.struct_size = ctypes.sizeof(ModelDesc)
    status = oracle.create_status(model_buf, free, out)
    if status != 0:
        raise oracle.OracleError(f"cere_ktest_model_create failed (status {status}; desc size {desc.struct_size}, "
                                 f"abi {desc.abi_version})")
    handle = int(out[0])
    if handle == 0:
        raise oracle.OracleError("cere_ktest_model_create returned a null handle")
    return A4Model(handle, (config_buf, desc_buf, name_buf, stops, model_buf, draft_map, model_levers, entries),
                   desc.struct_size)


def upload_and_seal(model: A4Model) -> None:
    oracle.upload_and_seal(model.handle, model.keep[-1], upload_file)


def release_model(handle: int) -> None:
    oracle.release_model(handle)


# -- runtime entry points ----------------------------------------------------------------------------------------


class Generation(NamedTuple):
    tokens: torch.Tensor  # int32 [committed] (uint32 IDs)
    scores: torch.Tensor  # int32 [committed, 4]: cere_token_score bytes {id, runner_up, logit, runner_up_logit}
    committed: int
    finish: int
    device_bytes: int


def generate(handle: int, ids: list[int], max_new: int, *, chunk: int = 0, cancel_after: int = 0) -> Generation:
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((max_new,), -1, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    result = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_generate", handle, prompt, len(ids), max_new, chunk, cancel_after, tokens, scores,
                result)
    committed, finish, device_bytes = (int(v) for v in result.tolist())
    if not 0 <= committed <= max_new:
        raise oracle.OracleError(f"committed_tokens {committed} outside [0, {max_new}]")
    return Generation(tokens[:committed].clone(), scores[:committed].clone(), committed, finish, device_bytes)


def generate_flags(handle: int, ids: list[int], max_new: int, flags: int) -> tuple[Generation, int, int, int]:
    """`cere_ktest_generate_flags`: the generation, its verify steps, accepted drafts and verified drafts."""
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((max_new,), -1, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    result = torch.zeros(6, dtype=torch.int64)
    oracle.call("cere_ktest_generate_flags", handle, prompt, len(ids), max_new, flags, tokens, scores, result)
    committed, finish, device_bytes, steps, accepted, verified = (int(v) for v in result.tolist())
    if not 0 <= committed <= max_new:
        raise oracle.OracleError(f"committed_tokens {committed} outside [0, {max_new}]")
    generation = Generation(tokens[:committed].clone(), scores[:committed].clone(), committed, finish, device_bytes)
    return generation, steps, accepted, verified


def synthetic_ids(gen: torch.Generator, vocab: int, count: int) -> list[int]:
    """`count` seeded token ids in [0, vocab) (no prompt text)."""
    return torch.randint(0, vocab, (count,), generator=gen, dtype=torch.int64).tolist()


class RowsDecode(NamedTuple):
    scores: torch.Tensor  # int32 [S, T + 1, 4] (cere_token_score bytes), -1 where unwritten
    steps: list  # per session: per decode k = 1 .. leave - join, {B8 name: BF16 [1, cols] on GPU0} (empty untapped)


def forward_decode_rows(handle: int, prompts: list, join: list, leave: list, *, taps: bool, hidden: int,
                        layers: int) -> RowsDecode:
    """`cere_ktest_forward_decode_rows` greedy (no forced tokens), with forward_trace's headroom per tapped step (4
    entries per layer, 128 name bytes per entry)."""
    sessions, rounds = len(prompts), max(leave)
    flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
    lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
    scores = torch.full((sessions * (rounds + 1), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    steps = sessions * rounds
    per_step = 4 * layers if taps else 0
    names = torch.zeros(steps * per_step * 128, dtype=torch.uint8) if taps else None
    shapes = torch.full((steps * per_step, 3), -1, dtype=torch.int64) if taps else None
    values = oracle.empty((steps * per_step * hidden,), torch.bfloat16) if taps else None
    oracle.call("cere_ktest_forward_decode_rows", handle, sessions, flat, lens,
                torch.tensor(join, dtype=torch.int64), torch.tensor(leave, dtype=torch.int64), None, scores,
                names, names.numel() if taps else 0, shapes, steps * per_step, values,
                values.numel() * values.element_size() if taps else 0, counts)
    scores = scores.view(sessions, rounds + 1, 4)
    out: list = [[] for _ in range(sessions)]
    if not taps:
        return RowsDecode(scores, out)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 < entries <= per_step and 0 < name_bytes <= per_step * 128
            and 0 < value_bytes <= per_step * hidden * values.element_size()):
        raise oracle.OracleError(f"forward_decode_rows counts {entries}, {name_bytes}, {value_bytes} per step")
    for s in range(sessions):
        for k in range(1, leave[s] - join[s] + 1):
            step = s * rounds + k - 1
            emitted = bytes(names[step * name_bytes:(step + 1) * name_bytes].tolist()).split(b"\0")
            if len(emitted) != entries + 1 or emitted[-1] != b"":
                raise oracle.OracleError(f"forward_decode_rows session {s} decode {k}: {len(emitted) - 1} names")
            element = step * value_bytes // values.element_size()
            entry: dict = {}
            for i in range(entries):
                name = emitted[i].decode()
                ndim, r, c = (int(v) for v in shapes[step * entries + i].tolist())
                if ndim != 2 or r != 1 or c <= 0 or name in entry:
                    raise oracle.OracleError(f"forward_decode_rows session {s} decode {k} entry {name}: "
                                             f"shape record {(ndim, r, c)}")
                entry[name] = values[element:element + c].view(1, c)
                element += c
            if (element - step * value_bytes // values.element_size()) * values.element_size() != value_bytes:
                raise oracle.OracleError(f"forward_decode_rows session {s} decode {k}: value bytes != the entries'")
            out[s].append(entry)
    return RowsDecode(scores, out)


def generate_rows(handle: int, prompts: list, max_new: list, chunk: int) -> list:
    """`cere_ktest_generate_rows`: one Generation per request (device_bytes 0: not measured here)."""
    n, stride = len(prompts), max(max_new)
    flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
    lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
    tokens = torch.full((n, stride), -1, dtype=torch.int32)
    scores = torch.full((n, stride, 4), -1, dtype=torch.int32)
    results = torch.zeros(2 * n, dtype=torch.int64)
    oracle.call("cere_ktest_generate_rows", handle, n, flat, lens, torch.tensor(max_new, dtype=torch.int64), chunk,
                tokens, scores, stride, results)
    out = []
    for i in range(n):
        committed, finish = int(results[2 * i]), int(results[2 * i + 1])
        if not 0 <= committed <= max_new[i]:
            raise oracle.OracleError(f"generate_rows request {i}: committed_tokens {committed} outside "
                                     f"[0, {max_new[i]}]")
        out.append(Generation(tokens[i, :committed].clone(), scores[i, :committed].clone(), committed, finish, 0))
    return out


class SampledGeneration(NamedTuple):
    gen: Generation
    logprobs: torch.Tensor  # int32 [committed, TOKEN_LOGPROBS_BYTES / 4]: cere_token_logprobs bytes


def sampling_bytes(specs: list, vocab: int) -> torch.Tensor:
    """The cere_sampling_params array of `specs` (CR4_SAMPLING entries) as host bytes; None is struct_size 0."""
    array = (SamplingParams * len(specs))()
    for p, spec in zip(array, specs):
        if spec is None:
            continue
        temperature, top_p, top_k, frequency, presence, repetition, seed, _ = spec
        p.struct_size = ctypes.sizeof(SamplingParams)
        p.abi_version = ABI_VERSION
        p.temperature, p.top_p, p.min_p, p.top_k = temperature, top_p, 0.0, top_k or vocab
        p.frequency_penalty, p.presence_penalty, p.repetition_penalty = frequency, presence, repetition
        p.seed = seed
    return torch.frombuffer(bytearray(bytes(array)), dtype=torch.uint8)


def generate_rows_sampled(handle: int, prompts: list, specs: list, max_new: int,
                          vocab: int) -> tuple[list, list[int]]:
    """`cere_ktest_generate_rows_sampled` (executor policy chunks): one SampledGeneration per request (its logprobs
    rows empty unless the spec asks for them) and the device bytes in use before the first submit and after."""
    n = len(prompts)
    flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
    lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
    tops = torch.tensor([CR4_TOP_LOGPROBS if s is not None and s[7] else 0 for s in specs], dtype=torch.int32)
    tokens = torch.full((n, max_new), -1, dtype=torch.int32)
    scores = torch.full((n, max_new, 4), -1, dtype=torch.int32)
    logprobs = torch.full((n, max_new, TOKEN_LOGPROBS_BYTES // 4), -1, dtype=torch.int32)
    results = torch.zeros(2 * n, dtype=torch.int64)
    device_bytes = torch.zeros(2, dtype=torch.int64)
    oracle.call("cere_ktest_generate_rows_sampled", handle, n, flat, lens,
                torch.tensor([max_new] * n, dtype=torch.int64), 0, sampling_bytes(specs, vocab), tops, tokens, scores,
                logprobs, max_new, results, device_bytes)
    out = []
    for i in range(n):
        committed, finish = int(results[2 * i]), int(results[2 * i + 1])
        if not 0 <= committed <= max_new:
            raise oracle.OracleError(f"generate_rows_sampled request {i}: committed_tokens {committed} outside "
                                     f"[0, {max_new}]")
        gen = Generation(tokens[i, :committed].clone(), scores[i, :committed].clone(), committed, finish, 0)
        out.append(SampledGeneration(gen, logprobs[i, :committed].clone()))
    return out, [int(v) for v in device_bytes.tolist()]


def generate_sampled_flags(handle: int, ids: list[int], max_new: int, flags: int, spec,
                           vocab: int) -> tuple[Generation, int, int, int]:
    """`cere_ktest_generate_sampled_flags`: `generate_flags` of a request sampled per `spec` (CR4_SAMPLING-shaped,
    not None): the generation, its verify steps, accepted drafts and verified drafts."""
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((max_new,), -1, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    result = torch.zeros(6, dtype=torch.int64)
    oracle.call("cere_ktest_generate_sampled_flags", handle, prompt, len(ids), max_new, flags,
                sampling_bytes([spec], vocab), tokens, scores, result)
    committed, finish, device_bytes, steps, accepted, verified = (int(v) for v in result.tolist())
    if not 0 <= committed <= max_new:
        raise oracle.OracleError(f"committed_tokens {committed} outside [0, {max_new}]")
    gen = Generation(tokens[:committed].clone(), scores[:committed].clone(), committed, finish, device_bytes)
    return gen, steps, accepted, verified


class Trace(NamedTuple):
    values: dict  # B8 name -> BF16 [rows, cols] on GPU0
    scores: torch.Tensor  # int32 [rows, 4] (cere_token_score bytes)


def forward_trace(handle: int, ids: list[int], *, hidden: int, layers: int, chunk: int = 0,
                  last_rows: int = 0) -> Trace:
    rows = last_rows or len(ids)
    capacity = 4 * layers
    prompt = torch.tensor(ids, dtype=torch.int32)
    names = torch.zeros(capacity * 128, dtype=torch.uint8)
    shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
    values = oracle.empty((capacity * rows * hidden,), torch.bfloat16)
    scores = torch.full((rows, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_trace", handle, prompt, len(ids), chunk, last_rows, names, names.numel(),
                shapes, capacity, values, values.numel() * values.element_size(), scores, counts)
    return parse_trace("forward_trace", names, shapes, values, scores, counts, capacity)


def parse_trace(fn_name: str, names: torch.Tensor, shapes: torch.Tensor, values: torch.Tensor, scores: torch.Tensor,
                counts: torch.Tensor, capacity: int) -> Trace:
    """A forward_trace-style output (names, {2, rows, cols} shapes, values back to back) as {name: [rows, cols]}."""
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 <= entries <= capacity and 0 <= name_bytes <= names.numel()
            and 0 <= value_bytes <= values.numel() * values.element_size()):
        raise oracle.OracleError(f"{fn_name} counts {entries}, {name_bytes}, {value_bytes} exceed capacities")
    emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"{fn_name} wrote {len(emitted) - 1} names for {entries} entries")
    out: dict = {}
    element = 0
    for i in range(entries):
        name = emitted[i].decode()
        ndim, r, c = (int(v) for v in shapes[i].tolist())
        if ndim != 2 or r < 0 or c < 0:
            raise oracle.OracleError(f"{fn_name} entry {name}: shape record {(ndim, r, c)}")
        if name in out:
            raise oracle.OracleError(f"{fn_name} emitted {name} twice")
        out[name] = values[element:element + r * c].view(r, c)
        element += r * c
    if element * values.element_size() != value_bytes:
        raise oracle.OracleError(f"{fn_name} value bytes {value_bytes} != the entries' {element * 2}")
    return Trace(out, scores)


def tap_names(tc: dict, detail: int) -> list[tuple[str, int]]:
    """(name, cols) of every forward_trace_detail entry in forward order (model.h ForwardTaps): detail 0 = the B8
    names; detail 1 adds the D3 names, kind (a) `model.layers.L.*` and kind (b) `native.*`."""
    h = int(tc["hidden_size"])
    hc = int(tc["hc_count"]) * h
    qkv = 2 * int(tc["num_attention_heads"]) * int(tc["head_dim"]) + 2 * int(tc["num_key_value_heads"]) * int(
        tc["head_dim"])
    value_dim = int(tc["linear_num_value_heads"]) * int(tc["linear_value_head_dim"])
    conv = 2 * int(tc["linear_num_key_heads"]) * int(tc["linear_key_head_dim"]) + value_dim
    ba = 2 * int(tc["linear_num_value_heads"])
    out = []
    for layer, kind in enumerate(tc["layer_types"]):
        m, n = f"model.layers.{layer}.", f"native.layers.{layer}."
        if detail:
            out += [(n + "attn_mix.in", hc), (m + "attn_hyper_connection.hc_norm", hc), (n + "block.in", h)]
        if kind == "full_attention":
            if detail:
                out.append((m + "self_attn.qkv_proj", qkv))
            out.append((m + "self_attn.o_proj", h))
        else:
            if detail:
                out += [(m + "linear_attn.in_proj_qkvz", conv + value_dim), (m + "linear_attn.in_proj_ba", ba),
                        (m + "linear_attn.norm", value_dim)]
            out.append((m + "linear_attn.out_proj", h))
        if detail:
            out += [(n + "mlp_mix.in", hc), (m + "mlp_hyper_connection.hc_norm", hc), (n + "mlp.in", h),
                    (m + "mlp.gate", int(tc["num_experts"])),
                    (m + "mlp.shared_expert.gate_up_proj", 2 * int(tc["shared_expert_intermediate_size"]))]
        out += [(m + "mlp.experts", h), (m + "mlp.shared_expert.down_proj", h)]
    if detail:
        out += [("native.final_mix.in", hc), ("native.final_mix.normed", hc)]
    out.append(("model", h))
    return out


def trace_detail_buffers(tc: dict, detail: int, rows: int):
    """names, shapes, values (NaN), scores (-1), counts and the entry capacity of a forward_trace_detail call, sized
    for tap_names plus 16 entries of headroom."""
    expected = tap_names(tc, detail)
    capacity = len(expected) + 16
    elements = (sum(c for _, c in expected) + 16 * int(tc["hc_count"]) * int(tc["hidden_size"])) * rows
    return (torch.zeros(capacity * 128, dtype=torch.uint8), torch.full((capacity, 3), -1, dtype=torch.int64),
            oracle.empty((elements,), torch.bfloat16), torch.full((rows, 4), -1, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int64), capacity)


def trace_detail(handle: int, ids: list[int], tc: dict, detail: int, *, last_rows: int = 0) -> Trace:
    rows = last_rows or len(ids)
    names, shapes, values, scores, counts, capacity = trace_detail_buffers(tc, detail, rows)
    oracle.call("cere_ktest_forward_trace_detail", handle, torch.tensor(ids, dtype=torch.int32), len(ids), 0,
                last_rows, detail, names, names.numel(), shapes, capacity, values,
                values.numel() * values.element_size(), scores, counts)
    return parse_trace("forward_trace_detail", names, shapes, values, scores, counts, capacity)


def trace_detail_refusal(handle: int, ids: list[int], tc: dict, detail: int) -> tuple[int, bool]:
    """(status, nothing written) of one forward_trace_detail call over every row with tap level `detail`."""
    names, shapes, values, scores, counts, capacity = trace_detail_buffers(tc, 0, len(ids))
    fn = oracle.ktest("cere_ktest_forward_trace_detail")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    tokens = torch.tensor(ids, dtype=torch.int32)
    status = fn(*[oracle._arg(a) for a in (handle, tokens, len(ids), 0, 0)], ctypes.c_uint32(detail),
                *[oracle._arg(a) for a in (names, names.numel(), shapes, capacity, values,
                                           values.numel() * values.element_size(), scores, counts)],
                ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    untouched = (not bool(counts.any()) and not bool(names.any()) and bool((shapes == -1).all())
                 and bool(torch.isnan(values).all()) and bool((scores == -1).all()))
    return int(status), untouched


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def trace_digest(trace: Trace) -> str:
    """SHA-256 over a trace's entries in order (name, shape, value bytes), then its score bytes."""
    h = hashlib.sha256()
    for name, value in trace.values.items():
        h.update(name.encode() + b"\0" + repr(tuple(value.shape)).encode())
        h.update(tensor_bytes(value))
    h.update(tensor_bytes(trace.scores))
    return h.hexdigest()


def decode_invariance_digests(handle: int, vocab: int) -> dict:
    """L2: SHA-256 of the forced_decode top-2 rows of prefills at n in L2_PREFILLS and of an L2_DECODE_PROMPT-token
    prefill plus L2_DECODE_STEPS forced decode steps, on tokens drawn from a seeded generator (kept in memory)."""
    gen = torch.Generator().manual_seed(L2_SEED)
    seeded = torch.randint(0, vocab, (L2_DECODE_PROMPT + L2_DECODE_STEPS + 1,), generator=gen).tolist()
    out = {}
    for n in L2_PREFILLS:
        out[f"prefill n={n}"] = hashlib.sha256(tensor_bytes(forced_decode(handle, seeded[:n + 1], n))).hexdigest()
    decode = forced_decode(handle, seeded, L2_DECODE_PROMPT)
    out[f"decode {L2_DECODE_STEPS} steps"] = hashlib.sha256(tensor_bytes(decode)).hexdigest()
    return out


def base_library_sha256() -> str:
    """The merge-base libcere_ktest.so's (spec L0) SHA-256; OracleError when it is absent or this build's library."""
    if not BASE_KTEST_LIB.is_file():
        raise oracle.OracleError(f"{BASE_KTEST_LIB} is absent (the merge-base build, tasks/m22p-spec.md L0)")
    if BASE_KTEST_LIB.resolve() == oracle.KTEST_LIB.resolve():
        raise oracle.OracleError(f"{BASE_KTEST_LIB} is this build's library")
    with BASE_KTEST_LIB.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@contextlib.contextmanager
def merge_base_library():
    """Routes oracle.ktest to the merge-base libcere_ktest.so (spec L0) for the duration; yields its SHA-256."""
    sha = base_library_sha256()
    lib = ctypes.CDLL(str(BASE_KTEST_LIB), mode=ctypes.RTLD_LOCAL)
    current = oracle.load_ktest
    oracle.load_ktest = lambda: lib
    try:
        yield sha
    finally:
        oracle.load_ktest = current


def merge_base_digests(tc: dict, prompts: list) -> dict:
    """The merge-base library's L1 trace digests (every row of each prompt) and L2 decode digests, on a model of its
    own, released before return."""
    hidden, layers = int(tc["hidden_size"]), int(tc["num_hidden_layers"])
    with merge_base_library() as sha:
        model = create_model(MAX_CONTEXT)
        try:
            upload_and_seal(model)
            traces = {pid: trace_digest(forward_trace(model.handle, ids, hidden=hidden, layers=layers))
                      for pid, ids, _ in prompts}
            decode = decode_invariance_digests(model.handle, int(tc["vocab_size"]))
        finally:
            release_model(model.handle)
    torch.cuda.synchronize(oracle.DEVICE)
    torch.cuda.empty_cache()
    return {"library_sha256": sha, "desc_size": model.desc_size, "trace": traces, "decode": decode}


def cached_merge_base_digests(tc: dict, prompts: list) -> dict:
    """O186: merge_base_digests through the merge-base digest cache (base_digests.py: its key, modes and atomic
    store), with the `merge-base digests` row: `base_digests` = "<cached|fresh> <key>" ("fresh uncached" without a
    cache), the mode, whether this run stored the entry or verified it, and the seconds the phase took. The cache's
    manifest digest names the pinned checkpoint (PINNED_MODEL_DIR), so under a swapped one (`checkpoint_dir`), or
    when the pinned artifact's manifest.json (base_digests.MANIFEST_MOUNT; PINNED_MODEL_DIR holds none) cannot be
    read, does not hash to the pin's or does not name PINNED_MODEL_DIR's weights files (`base_digests.pinned`),
    nothing is read or stored."""
    start = time.monotonic()
    cfg = base_digests.pinned(base_digests.config(os.environ), base_digests.MANIFEST_MOUNT, PINNED_MODEL_DIR)
    if oracle.MODEL_DIR != PINNED_MODEL_DIR:
        cfg = cfg._replace(directory=None, manifest_sha256=None)
    library = base_library_sha256()
    inputs = {"library_sha256": library, "manifest_sha256": cfg.manifest_sha256,
              "prompts_sha256": base_digests.prompts_sha256([(pid, ids) for pid, ids, _ in prompts]),
              "harness_sha256": base_digests.harness_sha256(Path(__file__), (FAMILIES_FILE,)),
              "runtime": f"torch {torch.__version__} cuda {torch.version.cuda} "
                         f"{torch.cuda.get_device_name(oracle.DEVICE)}"}
    result = base_digests.resolve(cfg, inputs, [pid for pid, _, _ in prompts],
                                  lambda: merge_base_digests(tc, prompts))
    oracle.report_row(ROW, "merge-base digests", base_digests=f"{result.source} {result.key or 'uncached'}",
                      mode=cfg.mode, stored=result.stored, verified=result.verified, merge_base_library=library,
                      seconds=round(time.monotonic() - start, 1))
    return result.digests


def score_fields(top1: torch.Tensor) -> tuple:
    """cere_token_score int32[4] -> (id, runner_up, logit bits, runner-up logit bits)."""
    return tuple(int(v) for v in top1.tolist())


def float_bits(value: torch.Tensor) -> int:
    return int(value.view(torch.int32)[0])


def patch_status(handle: int, ids: list[int], names: torch.Tensor, name_bytes: int, values: torch.Tensor,
                 values_bytes: int, target: int, logit: torch.Tensor, top1: torch.Tensor, *,
                 count: int | None = None) -> int:
    """The status of one `cere_ktest_forward_patch` call (not raised); `count` overrides token_count."""
    fn = oracle.ktest("cere_ktest_forward_patch")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    tokens = torch.tensor(ids, dtype=torch.int32)
    token_count = len(ids) if count is None else count
    status = fn(*[oracle._arg(a) for a in (handle, tokens, token_count, names, name_bytes, values, values_bytes)],
                ctypes.c_uint32(target), *[oracle._arg(a) for a in (logit, top1)],
                ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(status)


def name_blob(names: list[str]) -> torch.Tensor:
    return torch.frombuffer(bytearray(b"".join(n.encode() + b"\0" for n in names)), dtype=torch.uint8)


def forward_patch(handle: int, ids: list[int], target: int, patches: list) -> tuple[int, tuple]:
    """(target logit bits, top-1 fields) of `cere_ktest_forward_patch` with each (name, BF16 [n, cols]) replaced."""
    names = name_blob([n for n, _ in patches])
    values = torch.cat([v.reshape(-1) for _, v in patches]).contiguous()
    logit = torch.full((1,), math.nan, dtype=torch.float32)
    top1 = torch.full((4,), -1, dtype=torch.int32)
    oracle.call("cere_ktest_forward_patch", handle, torch.tensor(ids, dtype=torch.int32), len(ids), names,
                names.numel(), values, values.numel() * values.element_size(), target, logit, top1)
    return float_bits(logit), score_fields(top1)


class _DeviceView:
    """`__cuda_array_interface__` over a native device pointer (int16 words; viewed as BF16)."""

    def __init__(self, ptr: int, rows: int, cols: int):
        self.__cuda_array_interface__ = {"shape": (rows, cols), "typestr": "<i2", "data": (ptr, False),
                                         "version": 3, "strides": None}


class Screen:
    """A `forward_patch_screen` callback: with `copy` every P point is replaced by a fresh copy of its own value (kept
    until the next callback, model.h), else none is; `replace` (name -> BF16 [n, cols] on GPU0) replaces the named P
    points instead; `capture` names the I and P points whose native value is cloned into `captured` (a P point's
    value before its replacement). Counts the I and P callbacks by name; a Python exception cannot cross ctypes
    (ctypes prints it and returns null), so `started` - `finished` counts the callbacks that did not finish."""

    def __init__(self, copy: bool = False, *, capture: tuple = (), replace: dict | None = None):
        self.copy = copy
        self.capture = set(capture)
        self.replace = dict(replace or {})
        self.captured: dict = {}
        self.inputs = 0
        self.points: list = []
        self.started = 0
        self.finished = 0
        self.keep = None
        self.fn = PATCH_FN(self._callback)

    def _callback(self, ctx, name, layer, rows, cols, native, stream):
        self.started += 1
        name = name.decode()
        on = torch.cuda.ExternalStream(stream, device=oracle.DEVICE) if stream else torch.cuda.default_stream(
            oracle.DEVICE)
        if name in self.capture:
            with torch.cuda.stream(on):
                self.captured[name] = torch.as_tensor(_DeviceView(native, rows, cols),
                                                      device=oracle.DEVICE).clone().view(torch.bfloat16)
        if name.endswith(INPUT_SUFFIXES):
            self.inputs += 1
            self.finished += 1
            return None
        self.points.append(name)
        if name in self.replace:
            value = self.replace[name]
            if tuple(value.shape) != (rows, cols) or value.dtype != torch.bfloat16 or not value.is_contiguous():
                return None  # unfinished: forward_patch_screen raises
            on.wait_stream(torch.cuda.current_stream(oracle.DEVICE))
            self.finished += 1
            return value.data_ptr()
        if not self.copy:
            self.finished += 1
            return None
        with torch.cuda.stream(on):
            value = torch.as_tensor(_DeviceView(native, rows, cols), device=oracle.DEVICE).clone()
        self.keep = value
        self.finished += 1
        return value.data_ptr()


def screen_status(handle: int, ids: list[int], fn, target: int, logit: torch.Tensor, top1: torch.Tensor) -> int:
    """The status of one `cere_ktest_forward_patch_screen` call with callback `fn` (not raised)."""
    call = oracle.ktest("cere_ktest_forward_patch_screen")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    tokens = torch.tensor(ids, dtype=torch.int32)
    status = call(ctypes.c_uint64(handle), ctypes.c_void_p(tokens.data_ptr()), ctypes.c_uint64(len(ids)), fn,
                  ctypes.c_void_p(0), ctypes.c_uint32(target), ctypes.c_void_p(logit.data_ptr()),
                  ctypes.c_void_p(top1.data_ptr()), ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(status)


def forward_patch_screen(handle: int, ids: list[int], target: int, screen: Screen) -> tuple[int, tuple]:
    logit = torch.full((1,), math.nan, dtype=torch.float32)
    top1 = torch.full((4,), -1, dtype=torch.int32)
    status = screen_status(handle, ids, screen.fn, target, logit, top1)
    screen.keep = None
    if status != 0:
        raise oracle.OracleError(f"cere_ktest_forward_patch_screen returned status {status}")
    if screen.finished != screen.started:
        raise oracle.OracleError(f"screen callback failed in {screen.started - screen.finished} of {screen.started} "
                                 "calls")
    return float_bits(logit), score_fields(top1)


def non_identity_points(tc: dict) -> tuple[list, str]:
    """(Z2A1-03) The five P points whose negated replacement must reach the logits, and the next layer's input point
    the two MoE blocks must reach: L = the first GDN layer (gdn_qkv.out exists only there)."""
    layer = oracle.linear_attention_layers()[0]
    if layer + 1 >= len(tc["layer_types"]):
        raise oracle.OracleError(f"first GDN layer {layer} has no next layer")
    names = [f"model.layers.{layer}.mlp.experts", f"model.layers.{layer}.mlp.shared_expert.down_proj",
             f"native.layers.{layer}.attn_mix.out", f"native.layers.{layer}.gemm.gdn_qkv.out", "native.final_mix.out"]
    return names, f"native.layers.{layer + 1}.attn_mix.normed"


def native_points(handle: int, ids: list[int], target: int, names: list) -> tuple[tuple, dict]:
    """The unpatched (target logit bits, top-1 fields) and native value of each named point of one screened forward."""
    screen = Screen(capture=tuple(names))
    result = forward_patch_screen(handle, ids, target, screen)
    missing = [n for n in names if n not in screen.captured]
    if missing:
        raise oracle.OracleError(f"forward_patch_screen never called back {missing}")
    return result, screen.captured


def negated(value: torch.Tensor) -> torch.Tensor:
    """-value, exact in BF16; refuses an all-zero tensor (its negation is the identity)."""
    if not bool(value.ne(0).any()):
        raise oracle.OracleError("a non-identity patch needs a nonzero native value")
    return value.neg().contiguous()


def patch_point_count(tc: dict) -> int:
    """P callbacks of one forward (model.h ForwardTaps): per layer 2 mix outputs, 3 B8 blocks, the MoE GEMM outputs
    (router, shared gate and up, shared down) and the attention (qkv, o) or GDN (qkv, z, b, a, out) GEMM outputs; then
    the final mixer."""
    return sum(2 + 3 + 4 + (2 if kind == "full_attention" else 5) for kind in tc["layer_types"]) + 1


def forced_decode(handle: int, ids: list[int], prompt_tokens: int) -> torch.Tensor:
    """int32 [T - P, 4] cere_token_score rows of `cere_ktest_forced_decode` (as B8's helper); unwritten rows stay -1."""
    tokens = torch.tensor(ids, dtype=torch.int32)
    scores = torch.full((len(ids) - prompt_tokens, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_forced_decode", handle, tokens, len(ids), prompt_tokens, scores)
    return scores


class TraceBuffers(NamedTuple):
    names: torch.Tensor  # uint8, zero-filled
    shapes: torch.Tensor  # int64 [entries, 3], -1-filled
    values: torch.Tensor  # BF16 on GPU0, NaN-filled
    scores: torch.Tensor  # int32 [completion, 4], -1-filled
    counts: torch.Tensor  # int64 [3], zero-filled


def trace_buffers(name_bytes: int, entries: int, value_elements: int, completion: int) -> TraceBuffers:
    """Output buffers of `cere_ktest_forced_decode_trace`, filled so that unwritten bytes are visible."""
    return TraceBuffers(torch.zeros(name_bytes, dtype=torch.uint8), torch.full((entries, 3), -1, dtype=torch.int64),
                        oracle.empty((value_elements,), torch.bfloat16),
                        torch.full((completion, 4), -1, dtype=torch.int32), torch.zeros(3, dtype=torch.int64))


def trace_status(handle: int, ids: list[int], prompt_tokens: int, buf: TraceBuffers, names_capacity: int,
                 entry_capacity: int, values_capacity_bytes: int) -> int:
    """The status of one `cere_ktest_forced_decode_trace` call with the given capacities (not raised)."""
    fn = oracle.ktest("cere_ktest_forced_decode_trace")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    tokens = torch.tensor(ids, dtype=torch.int32)
    args = (handle, tokens, len(ids), prompt_tokens, buf.names, names_capacity, buf.shapes, entry_capacity,
            buf.values, values_capacity_bytes, buf.scores, buf.counts)
    status = fn(*[oracle._arg(a) for a in args], ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(status)


class DecodeTrace(NamedTuple):
    steps: list  # per DECODE step j = 1 .. C - 1: B8 name -> BF16 [1, cols] on GPU0, in emission order
    scores: torch.Tensor  # int32 [C, 4] (cere_token_score bytes): row 0 from the PREFILL, row j from step j
    counts: tuple  # out_counts: entries, name bytes, value bytes


def forced_decode_trace(handle: int, ids: list[int], prompt_tokens: int, *, hidden: int, layers: int,
                        chain: int | None = None) -> DecodeTrace:
    """`cere_ktest_forced_decode_trace` (or, with `chain`, `cere_ktest_forced_verify_trace` at that chain) with
    forward_trace's headroom per step (4 entries per layer, 128 name bytes per entry), parsed step-major into one
    {name: [1, cols]} dict per DECODE step."""
    completion = len(ids) - prompt_tokens
    steps = completion - 1
    if steps < 1:
        raise oracle.OracleError(f"a decode-step trace needs at least 2 completion tokens, got {completion}")
    capacity = steps * 4 * layers
    buf = trace_buffers(capacity * 128, capacity, capacity * hidden, completion)
    tokens = torch.tensor(ids, dtype=torch.int32)
    if chain is None:
        oracle.call("cere_ktest_forced_decode_trace", handle, tokens, len(ids), prompt_tokens, buf.names,
                    buf.names.numel(), buf.shapes, capacity, buf.values,
                    buf.values.numel() * buf.values.element_size(), buf.scores, buf.counts)
    else:
        oracle.call("cere_ktest_forced_verify_trace", handle, tokens, len(ids), prompt_tokens, chain, buf.names,
                    buf.names.numel(), buf.shapes, capacity, buf.values,
                    buf.values.numel() * buf.values.element_size(), buf.scores, buf.counts)
    entries, name_bytes, value_bytes = (int(v) for v in buf.counts.tolist())
    if not (0 <= entries <= capacity and 0 <= name_bytes <= buf.names.numel()
            and 0 <= value_bytes <= buf.values.numel() * buf.values.element_size()):
        raise oracle.OracleError(f"forced_decode_trace counts {entries}, {name_bytes}, {value_bytes} exceed "
                                 "capacities")
    if entries == 0 or entries % steps:
        raise oracle.OracleError(f"forced_decode_trace wrote {entries} entries for {steps} steps")
    emitted = bytes(buf.names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"forced_decode_trace wrote {len(emitted) - 1} names for {entries} entries")
    per_step = entries // steps
    out = []
    element = 0
    for s in range(steps):
        step: dict = {}
        for i in range(s * per_step, (s + 1) * per_step):
            name = emitted[i].decode()
            ndim, r, c = (int(v) for v in buf.shapes[i].tolist())
            if ndim != 2 or r != 1 or c <= 0:
                raise oracle.OracleError(f"forced_decode_trace step {s + 1} entry {name}: shape record {(ndim, r, c)}")
            if name in step:
                raise oracle.OracleError(f"forced_decode_trace step {s + 1} emitted {name} twice")
            step[name] = buf.values[element:element + r * c].view(r, c)
            element += r * c
        out.append(step)
    if element * buf.values.element_size() != value_bytes:
        raise oracle.OracleError(f"forced_decode_trace value bytes {value_bytes} != the entries' {element * 2}")
    return DecodeTrace(out, buf.scores, (entries, name_bytes, value_bytes))


def b8_names(tc: dict) -> list[str]:
    """The B8 tensor names: per layer N the attention output projection, the routed experts and the shared expert's
    down projection, then the final `model` output."""
    names = []
    for layer, kind in enumerate(tc["layer_types"]):
        attn = "self_attn.o_proj" if kind == "full_attention" else "linear_attn.out_proj"
        names += [f"model.layers.{layer}.{attn}", f"model.layers.{layer}.mlp.experts",
                  f"model.layers.{layer}.mlp.shared_expert.down_proj"]
    names.append("model")
    return names


# -- prompts -----------------------------------------------------------------------------------------------------


@functools.cache
def tokenizer():
    return get_tokenizer(str(oracle.MODEL_DIR))


def chat_ids(messages: list) -> list[int]:
    """Prompt IDs as SGLang's chat path builds them (render with the template, `enable_thinking` false, then encode
    without extra specials; `serving_chat.py:1324-1366`)."""
    tok = tokenizer()
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return list(tok.encode(text, add_special_tokens=False))


def parity_prompts(pids: tuple = PARITY_IDS) -> list[tuple[str, list[int], int]]:
    rows = {}
    for line in PARITY_FILE.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["id"]] = item
    missing = [pid for pid in pids if pid not in rows]
    if missing:
        raise oracle.OracleError(f"{PARITY_FILE.name} lacks {missing}")
    return [(pid, chat_ids(rows[pid]["messages"]), int(rows[pid]["max_tokens"])) for pid in pids]


def device_bytes_in_use() -> int:
    torch.cuda.synchronize(oracle.DEVICE)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(oracle.DEVICE)
    return total - free


def process_device_bytes(pid: int) -> int:
    """Device memory NVML reports in use by process `pid` on GPU0 (the container sees GPU0 only, as NVML index 0;
    pids are the container's). One process's own figure: GPU0 also drives the desktop, whose processes allocate and
    free at any time."""
    pynvml.nvmlInit()
    try:
        if pynvml.nvmlDeviceGetCount() != 1:
            raise oracle.OracleError(f"NVML sees {pynvml.nvmlDeviceGetCount()} GPUs, expected GPU0 only")
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(0))
    finally:
        pynvml.nvmlShutdown()
    used = [p.usedGpuMemory for p in procs if p.pid == pid]
    if len(used) != 1 or used[0] is None:
        raise oracle.OracleError(f"NVML reports {used} for pid {pid}, expected one used-memory figure")
    return int(used[0])


@contextlib.contextmanager
def foreign_allocation(nbytes: int):
    """A second process on GPU0 holding `nbytes` of device memory (plus its own CUDA context) until the block exits;
    yields its pid once the allocation is resident."""
    script = (f"import sys, torch\nx = torch.empty({nbytes}, dtype=torch.uint8, device='cuda:0')\n"
              "torch.cuda.synchronize()\nprint('ready', flush=True)\nsys.stdin.read()\n")
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        if line != "ready\n":
            raise oracle.OracleError(f"foreign allocation process printed {line!r} (exit {proc.poll()})")
        yield proc.pid
    finally:
        proc.stdin.close()
        proc.wait(timeout=60)
        proc.stdout.close()


class K8Forward(unittest.TestCase):
    LEVERS: dict[str, int] = {}  # the model's slot choices, lever -> CERE_LEVER_* (empty: null selection, every default)
    MERGE_BASE = True  # L1/L2: the merge-base library's digests (BASE_KTEST_LIB) of the same checkpoint

    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.hidden = int(cls.tc["hidden_size"])
        cls.layers = int(cls.tc["num_hidden_layers"])
        cls.stops = set(stop_token_ids())
        cls.prompts = parity_prompts()
        cls.prefill_prompts = parity_prompts(PREFILL_IDS)
        # The merge-base library's digests first (cached or fresh, O186): two models do not fit GPU0 at once.
        cls.merge_base = cached_merge_base_digests(cls.tc, cls.prefill_prompts) if cls.MERGE_BASE else None
        model = create_model(MAX_CONTEXT, ROWS_R, model_levers=oracle.levers(**cls.LEVERS) if cls.LEVERS else None)
        cls.addClassCleanup(release_model, model.handle)
        upload_and_seal(model)
        cls.handle = model.handle
        smoke = generate(cls.handle, chat_ids(SMOKE_MESSAGES), 1)
        if smoke.committed != 1:
            raise oracle.OracleError(f"SMOKE committed {smoke.committed} tokens, expected 1")
        cls.after_smoke = device_bytes_in_use()
        oracle.report_row(ROW, "smoke", device_bytes=cls.after_smoke, finish=smoke.finish)

    def check_terminal(self, case: str, g: Generation, max_new: int) -> None:
        """A finished greedy request: LENGTH at max_new or STOP_TOKEN on its last committed token; no stop token
        earlier; each score's id is the committed token and outranks its runner-up."""
        self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} {case}: finish {g.finish}")
        self.assertTrue(g.committed >= 1, f"{ROW} {case}: no token committed")
        ids = g.tokens.tolist()
        if g.finish == FINISH_LENGTH:
            self.assertEqual(g.committed, max_new, f"{ROW} {case}: LENGTH before max_new_tokens")
        else:
            self.assertTrue(ids[-1] in self.stops, f"{ROW} {case}: STOP_TOKEN without a stop token last")
        self.assertFalse(any(t in self.stops for t in ids[:-1]), f"{ROW} {case}: generation ran past a stop token")
        self.assertTrue(torch.equal(g.scores[:, 0], g.tokens), f"{ROW} {case}: score id != committed token")
        logits = g.scores.view(torch.float32)
        self.assertTrue(bool((g.scores[:, 1] != g.scores[:, 0]).all()), f"{ROW} {case}: runner-up equals top-1")
        self.assertTrue(bool((logits[:, 2] >= logits[:, 3]).all()), f"{ROW} {case}: runner-up logit above top-1")

    def test_cancel_at_token_10(self):
        pid, ids, max_new = self.prompts[0]
        full = generate(self.handle, ids, max_new)
        self.check_terminal(f"{pid} uncancelled", full, max_new)
        if full.committed <= CANCEL_AT:
            raise oracle.OracleError(f"{pid} completes in {full.committed} tokens; the cancel case needs more")
        cancelled = generate(self.handle, ids, max_new, cancel_after=CANCEL_AT)
        oracle.check_count(self, ROW, f"{pid} cancel committed", cancelled.committed, expected=CANCEL_AT)
        self.assertEqual(cancelled.finish, FINISH_CANCELLED, f"{ROW} {pid}: cancel finish {cancelled.finish}")
        oracle.check_bit_exact(self, ROW, f"{pid} cancel tokens", cancelled.tokens, full.tokens[:CANCEL_AT])
        oracle.check_bit_exact(self, ROW, f"{pid} cancel scores", cancelled.scores, full.scores[:CANCEL_AT])
        again = generate(self.handle, ids, max_new)
        oracle.check_count(self, ROW, f"{pid} after cancel committed", again.committed, expected=full.committed)
        oracle.check_bit_exact(self, ROW, f"{pid} after cancel tokens", again.tokens, full.tokens)
        oracle.check_bit_exact(self, ROW, f"{pid} after cancel scores", again.scores, full.scores)

    def test_forced_decode_rejects_out_of_vocab_forced_token(self):
        """`cere_ktest_forced_decode` refuses a forced token >= vocab_size with CERE_ERR_ARGUMENT before any request
        (review M22NI-05): no score row is written. The last token is only predicted, so an out-of-vocab id there
        is not fed and the call runs."""
        pid, ids, _ = self.prompts[0]
        vocab = int(self.tc["vocab_size"])
        fn = oracle.ktest("cere_ktest_forced_decode")
        stream = torch.cuda.current_stream(oracle.DEVICE)
        for case, forced, expected in (("forced", [vocab, 0], CERE_ERR_ARGUMENT), ("last", [0, vocab], 0)):
            tokens = torch.tensor(ids + forced, dtype=torch.int32)
            scores = torch.full((len(forced), 4), -1, dtype=torch.int32)
            status = fn(*[oracle._arg(a) for a in (self.handle, tokens, len(tokens), len(ids), scores)],
                        ctypes.c_void_p(stream.cuda_stream))
            stream.synchronize()
            oracle.check_count(self, ROW, f"{pid} {case} out-of-vocab status", int(status), expected=expected)
            written = int((scores[:, 0] >= 0).sum())
            oracle.check_count(self, ROW, f"{pid} {case} out-of-vocab score rows", written,
                               expected=0 if expected else len(forced))

    def decode_case(self, pid: str, ids: list[int], max_new: int) -> list[int]:
        """Prompt + the native greedy completion (the forced tokens of the decode-step trace cases)."""
        g = generate(self.handle, ids, max_new)
        self.check_terminal(f"{pid} trace completion", g, max_new)
        if g.committed < 2:
            raise oracle.OracleError(f"{pid} completes in {g.committed} token(s); a decode-step trace needs 2")
        return ids + g.tokens.tolist()

    def test_forced_decode_trace_scores_bit_equal(self):
        """C1 (tasks/m22d-spec.md D2): the decode-step trace's scores equal `cere_ktest_forced_decode`'s bit for bit
        on 3 prompts (the tapped direct loop computes the serving path); C - 1 steps, each with the same names, each
        entry [1, hidden] and finite."""
        for pid, ids, max_new in self.prompts[:TRACE_PROMPTS]:
            full = self.decode_case(pid, ids, max_new)
            steps = len(full) - len(ids) - 1
            reference = forced_decode(self.handle, full, len(ids))
            trace = forced_decode_trace(self.handle, full, len(ids), hidden=self.hidden, layers=self.layers)
            oracle.check_bit_exact(self, ROW, f"{pid} decode trace scores", trace.scores, reference)
            oracle.check_count(self, ROW, f"{pid} decode trace steps", len(trace.steps), expected=steps)
            first = list(trace.steps[0])
            other = sum(list(step) != first for step in trace.steps)
            oracle.check_count(self, ROW, f"{pid} decode trace steps naming other entries", other, expected=0)
            bad = sum(tuple(v.shape) != (1, self.hidden) or not bool(torch.isfinite(v).all())
                      for step in trace.steps for v in step.values())
            oracle.check_count(self, ROW, f"{pid} decode trace entries not finite or not (1, hidden)", bad, expected=0)

    def test_forced_decode_trace_model_vs_prefill(self):
        """C1 report-only row: per DECODE step j, rel-L2 of the step's `model` tap against the PREFILL
        `cere_ktest_forward_trace` row at the same position (prompt + completion minus its last token, last C rows:
        row j is position P + j - 1, the input position of step j), and the top-1 agreement of the two. Asserted:
        every step's names equal forward_trace's names in forward_trace's order."""
        for pid, ids, max_new in self.prompts[:TRACE_PROMPTS]:
            full = self.decode_case(pid, ids, max_new)
            completion = len(full) - len(ids)
            trace = forced_decode_trace(self.handle, full, len(ids), hidden=self.hidden, layers=self.layers)
            prefill = forward_trace(self.handle, full[:-1], hidden=self.hidden, layers=self.layers,
                                    last_rows=completion)
            expected = list(prefill.values)
            other = sum(list(step) != expected for step in trace.steps)
            oracle.check_count(self, ROW, f"{pid} decode trace steps naming other entries than forward_trace", other,
                               expected=0)
            d = [oracle.rel_l2(step["model"], prefill.values["model"][j:j + 1])
                 for j, step in enumerate(trace.steps, start=1)]
            agree = int((trace.scores[1:, 0] == prefill.scores[1:, 0]).sum())
            worst = max(range(len(d)), key=d.__getitem__)
            oracle.report_row(ROW, f"{pid} decode-step model vs prefill", steps=len(d), rel_l2_max=d[worst],
                              rel_l2_max_step=worst + 1, rel_l2_median=statistics.median(d), rel_l2_first=d[0],
                              rel_l2_last=d[-1], top1_agree=agree)

    def test_forced_decode_trace_refusals(self):
        """`cere_ktest_forced_decode_trace` returns CERE_ERR_ARGUMENT and writes nothing (no score row, out_counts
        zero, names, shapes and values untouched) for a forced token >= vocab_size and for each capacity one unit
        below the trace's size; at exactly that size, and with an out-of-vocab last token (predicted, never fed),
        the call runs and returns that size."""
        pid, ids, _ = self.prompts[0]
        vocab = int(self.tc["vocab_size"])
        forced = [0] * REFUSAL_COMPLETION
        entries, name_bytes, value_bytes = forced_decode_trace(self.handle, ids + forced, len(ids),
                                                               hidden=self.hidden, layers=self.layers).counts
        size = (name_bytes, entries, value_bytes)  # capacity argument order: names, entries, values
        cases = (("exact", forced, size, 0),
                 ("names capacity", forced, (name_bytes - 1, entries, value_bytes), CERE_ERR_ARGUMENT),
                 ("entry capacity", forced, (name_bytes, entries - 1, value_bytes), CERE_ERR_ARGUMENT),
                 ("values capacity", forced, (name_bytes, entries, value_bytes - 1), CERE_ERR_ARGUMENT),
                 ("out-of-vocab forced", [vocab] + forced[1:], size, CERE_ERR_ARGUMENT),
                 ("out-of-vocab last", forced[:-1] + [vocab], size, 0))
        for case, tail, capacities, expected in cases:
            buf = trace_buffers(name_bytes, entries, value_bytes // 2, len(tail))
            status = trace_status(self.handle, ids + tail, len(ids), buf, *capacities)
            oracle.check_count(self, ROW, f"{pid} trace {case} status", status, expected=expected)
            written = int((buf.scores[:, 0] >= 0).sum())
            oracle.check_count(self, ROW, f"{pid} trace {case} score rows", written,
                               expected=len(tail) if expected == 0 else 0)
            counts = [int(v) for v in buf.counts.tolist()]
            oracle.check_count(self, ROW, f"{pid} trace {case} counts as expected",
                               int(counts == ([entries, name_bytes, value_bytes] if expected == 0 else [0, 0, 0])),
                               expected=1)
            if expected != 0:
                touched = (int(buf.names.count_nonzero()) + int((buf.shapes != -1).sum())
                           + int((~torch.isnan(buf.values)).sum()))
                oracle.check_count(self, ROW, f"{pid} trace {case} tap elements written", touched, expected=0)

    def test_forward_trace_names_and_shapes(self):
        pid, ids, max_new = self.prompts[0]
        expected = b8_names(self.tc)
        trace = forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers)
        oracle.check_count(self, ROW, f"{pid} trace entries", len(trace.values), expected=len(expected))
        missing = [n for n in expected if n not in trace.values]
        extra = sorted(set(trace.values) - set(expected))
        self.assertEqual(missing, [], f"{ROW} forward_trace lacks {missing}")
        self.assertEqual(extra, [], f"{ROW} forward_trace emits names outside B8: {extra}")
        for name in expected:
            value = trace.values[name]
            self.assertEqual(tuple(value.shape), (len(ids), self.hidden), f"{ROW} {name}: shape")
            self.assertTrue(bool(torch.isfinite(value).all()), f"{ROW} {name}: non-finite values")
        oracle.check_count(self, ROW, f"{pid} trace score rows", trace.scores.shape[0], expected=len(ids))
        logits = trace.scores.view(torch.float32)
        self.assertTrue(bool((trace.scores[:, 1] != trace.scores[:, 0]).all()), f"{ROW} trace: runner-up = top-1")
        self.assertTrue(bool((logits[:, 2] >= logits[:, 3]).all()), f"{ROW} trace: runner-up logit above top-1")
        first = generate(self.handle, ids, 1)
        self.assertEqual(first.committed, 1, f"{ROW} {pid}: 1-token generation")
        self.assertTrue(int(trace.scores[-1, 0]) == int(first.tokens[0]),
                        f"{ROW} {pid}: trace's last top-1 differs from the first greedy token")
        oracle.report_row(ROW, f"{pid} trace", entries=len(trace.values), rows=len(ids))

    def test_greedy_repeat_bit_identical(self):
        for pid, ids, max_new in self.prompts:
            runs = [generate(self.handle, ids, max_new) for _ in range(REPEATS)]
            for k, run in enumerate(runs):
                self.check_terminal(f"{pid} run {k}", run, max_new)
            for k in range(1, REPEATS):
                oracle.check_count(self, ROW, f"{pid} run {k} committed", runs[k].committed,
                                   expected=runs[0].committed)
                oracle.check_bit_exact(self, ROW, f"{pid} run {k} tokens", runs[k].tokens, runs[0].tokens)
                oracle.check_bit_exact(self, ROW, f"{pid} run {k} scores", runs[k].scores, runs[0].scores)
            oracle.report_row(ROW, f"{pid} repeat", runs=REPEATS, committed=runs[0].committed,
                              finish=runs[0].finish)

    # -- prefill taps (m22p design note L1, L2) ----------------------------------------------------------------

    def test_forward_trace_equals_merge_base(self):
        # Every prefill prompt is compared and reported before the assert, so one prompt's failure never hides
        # another's row (Z2A3-01).
        differs = []
        for pid, ids, _ in self.prefill_prompts:
            digest = trace_digest(forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers))
            base = self.merge_base["trace"][pid]
            oracle.report_row(ROW, "L1 forward_trace vs merge-base", prompt=pid, prompt_tokens=len(ids), digest=digest,
                              merge_base=base, merge_base_library=self.merge_base["library_sha256"],
                              merge_base_desc_size=self.merge_base["desc_size"], passed=digest == base)
            if digest != base:
                differs.append(pid)
        self.assertEqual(differs, [], f"{ROW} {differs}: forward_trace differs from the merge-base library's")

    def test_forward_trace_detail_zero_equals_forward_trace(self):
        for pid, ids, _ in self.prefill_prompts:
            trace = trace_digest(forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers))
            detail = trace_digest(trace_detail(self.handle, ids, self.tc, 0))
            oracle.report_row(ROW, "L1 forward_trace_detail 0", prompt=pid, prompt_tokens=len(ids), digest=detail,
                              forward_trace=trace, passed=detail == trace)
            self.assertEqual(detail, trace, f"{ROW} {pid}: forward_trace_detail(detail 0) != forward_trace")

    def test_forward_trace_detail_one_keeps_top2(self):
        expected = tap_names(self.tc, 1)
        for pid, ids, _ in self.prefill_prompts:
            rows = min(DETAIL_ROWS, len(ids))
            plain = trace_detail(self.handle, ids, self.tc, 0, last_rows=rows)
            detail = trace_detail(self.handle, ids, self.tc, 1, last_rows=rows)
            names = [(name, int(v.shape[1])) for name, v in detail.values.items()]
            bad_rows = sum(int(v.shape[0]) != rows for v in detail.values.values())
            not_finite = sum(not bool(torch.isfinite(v).all()) for v in detail.values.values())
            b8_differ = [n for n, v in plain.values.items()
                         if n not in detail.values or not torch.equal(detail.values[n].view(torch.int16),
                                                                      v.view(torch.int16))]
            same = torch.equal(detail.scores, plain.scores)
            oracle.report_row(ROW, "L1 forward_trace_detail 1 top-2", prompt=pid, rows=rows, entries=len(names),
                              expected_entries=len(expected), b8_differ=len(b8_differ), not_finite=not_finite,
                              passed=same and names == expected and bad_rows == 0 and not b8_differ
                              and not_finite == 0)
            self.assertEqual(names, expected, f"{ROW} {pid}: detail 1 names/cols differ from model.h's forward order")
            oracle.check_count(self, ROW, f"{pid} detail 1 entries with other than {rows} rows", bad_rows, expected=0)
            oracle.check_count(self, ROW, f"{pid} detail 1 entries not finite", not_finite, expected=0)
            self.assertTrue(same, f"{ROW} {pid}: detail 1 top-2 differs from detail 0")
            self.assertEqual(b8_differ, [], f"{ROW} {pid}: detail 1 B8 values differ from detail 0")

    def test_forward_patch_identity(self):
        for pid, ids, _ in self.prefill_prompts:
            trace = forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers)
            last = score_fields(trace.scores[-1])
            null = Screen(copy=False)
            logit0, top0 = forward_patch_screen(self.handle, ids, last[0], null)
            blocks = [(name, value) for name, value in trace.values.items() if name != "model"]
            logit1, top1 = forward_patch(self.handle, ids, last[0], blocks)
            passed = top0 == last and logit0 == last[2] and (logit1, top1) == (logit0, top0)
            oracle.report_row(ROW, "L1 identity patch", prompt=pid, prompt_tokens=len(ids), blocks=len(blocks),
                              passed=passed)
            self.assertEqual(top0, last, f"{ROW} {pid}: unpatched forward_patch_screen top-2 != forward_trace's")
            self.assertEqual(logit0, last[2], f"{ROW} {pid}: target logit != the top-1 logit")
            self.assertEqual((logit1, top1), (logit0, top0),
                             f"{ROW} {pid}: identity patch of {len(blocks)} blocks moved the target logit or top-2")

    def test_forward_patch_screen_identity(self):
        points = patch_point_count(self.tc)
        for pid, ids, _ in self.prefill_prompts:
            null, copy = Screen(copy=False), Screen(copy=True)
            reference = forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers, last_rows=1)
            target = score_fields(reference.scores[0])[0]
            base = forward_patch_screen(self.handle, ids, target, null)
            patched = forward_patch_screen(self.handle, ids, target, copy)
            self.assertEqual(base[1], score_fields(reference.scores[0]),
                             f"{ROW} {pid}: unpatched screen top-2 != forward_trace's last row")
            oracle.report_row(ROW, "L1 identity screen", prompt=pid, points=len(copy.points), expected_points=points,
                              inputs=copy.inputs, passed=patched == base and len(copy.points) == points)
            oracle.check_count(self, ROW, f"{pid} screen P points", len(copy.points), expected=points)
            self.assertEqual(copy.points, null.points, f"{ROW} {pid}: P points depend on the replacement")
            self.assertTrue(copy.inputs > 0, f"{ROW} {pid}: no input point reached the callback")
            self.assertEqual(patched, base, f"{ROW} {pid}: replacing every P point by itself moved the result")

    def test_forward_patch_non_identity(self):
        names, _ = non_identity_points(self.tc)
        for pid, ids, _ in self.prefill_prompts:
            target = forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers, last_rows=1).scores[0]
            target = score_fields(target)[0]
            base, native = native_points(self.handle, ids, target, names)
            unmoved = []
            for name in names:
                patched = forward_patch(self.handle, ids, target, [(name, negated(native[name]))])
                if patched == base:
                    unmoved.append(name)
            oracle.report_row(ROW, "L1 non-identity patch", prompt=pid, points=len(names), unmoved=len(unmoved),
                              passed=not unmoved)
            self.assertEqual(unmoved, [], f"{ROW} {pid}: a negated forward_patch replacement left the target logit "
                             "and top-2 unchanged")

    def test_forward_patch_screen_non_identity(self):
        names, downstream = non_identity_points(self.tc)
        moe = names[:2]
        for pid, ids, _ in self.prefill_prompts:
            target = forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers, last_rows=1).scores[0]
            target = score_fields(target)[0]
            base, native = native_points(self.handle, ids, target, names + [downstream])
            unmoved, unreached = [], []
            for name in names:
                screen = Screen(capture=(downstream,), replace={name: negated(native[name])})
                if forward_patch_screen(self.handle, ids, target, screen) == base:
                    unmoved.append(name)
                if name in moe and torch.equal(screen.captured[downstream].view(torch.int16),
                                               native[downstream].view(torch.int16)):
                    unreached.append(name)
            oracle.report_row(ROW, "L1 non-identity screen", prompt=pid, points=len(names), unmoved=len(unmoved),
                              moe_blocks=len(moe), downstream_unchanged=len(unreached),
                              passed=not unmoved and not unreached)
            self.assertEqual(unmoved, [], f"{ROW} {pid}: a negated screen replacement left the target logit and top-2 "
                             "unchanged")
            self.assertEqual(unreached, [], f"{ROW} {pid}: a negated MoE block left {downstream} unchanged")

    def test_forward_patch_refusals(self):
        ids = self.prefill_prompts[0][1][:REFUSAL_PROMPT]
        vocab = int(self.tc["vocab_size"])
        block = "model.layers.0.mlp.experts"
        one = torch.zeros(len(ids) * self.hidden, dtype=torch.bfloat16, device=oracle.DEVICE)
        valid = name_blob([block])

        def run(blob: torch.Tensor, name_bytes: int, values: torch.Tensor, target: int = 0,
                count: int | None = None) -> tuple[int, bool, torch.Tensor]:
            logit = torch.full((1,), math.nan, dtype=torch.float32)
            top1 = torch.full((4,), -1, dtype=torch.int32)
            status = patch_status(self.handle, ids, blob, name_bytes, values, values.numel() * values.element_size(),
                                  target, logit, top1, count=count)
            return status, bool(torch.isnan(logit).all()) and bool((top1 == -1).all()), top1

        def names(*n: str) -> tuple[torch.Tensor, int]:
            blob = name_blob(list(n))
            return blob, blob.numel()

        statuses = {
            "zero token_count": run(valid, valid.numel(), one, count=0),
            "zero name_bytes": run(valid, 0, one),
            "names without a final NUL": run(valid, valid.numel() - 1, one),
            "unknown name": run(*names(block.replace("experts", "expert")), one),
            "E name model": run(*names("model"), one),
            "I point": run(*names("native.layers.0.mlp.in"), one),
            "twice-per-layer GEMM output": run(*names("native.layers.0.gemm.moe_shared_gate_up.out"), one),
            "layer out of range": run(*names(f"model.layers.{self.layers}.mlp.experts"), one),
            "repeated name": run(*names(block, block), torch.cat([one, one])),
            "values_bytes one element short": run(valid, valid.numel(), one[:-1]),
            "values_bytes one element long": run(valid, valid.numel(), torch.cat([one, one[:1]])),
            "target = vocab": run(valid, valid.numel(), one, target=vocab),
        }
        logit = torch.full((1,), math.nan, dtype=torch.float32)
        top1 = torch.full((4,), -1, dtype=torch.int32)
        screen = Screen(copy=False)
        statuses["screen target = vocab"] = (screen_status(self.handle, ids, screen.fn, vocab, logit, top1),
                                             bool(torch.isnan(logit).all()) and bool((top1 == -1).all()), top1)
        statuses["screen null callback"] = (screen_status(self.handle, ids, PATCH_FN(), 0, logit, top1),
                                            bool(torch.isnan(logit).all()) and bool((top1 == -1).all()), top1)
        statuses["trace_detail detail 2"] = (*trace_detail_refusal(self.handle, ids, self.tc, 2), top1)
        for case, (status, untouched, _) in statuses.items():
            oracle.report_row(ROW, f"prefill tap refusal: {case}", status=status, untouched=untouched,
                              passed=status == CERE_ERR_ARGUMENT and untouched)
            self.assertEqual(status, CERE_ERR_ARGUMENT, f"{ROW} {case}: status {status}")
            self.assertTrue(untouched, f"{ROW} {case}: an output was written")
        status, untouched, top1 = run(valid, valid.numel(), one)
        oracle.report_row(ROW, "prefill tap refusal: one valid name", status=status, written=not untouched,
                          passed=status == 0 and not untouched)
        self.assertEqual(status, 0, f"{ROW} one valid patch name: status {status}")
        self.assertTrue(0 <= int(top1[0]) < vocab, f"{ROW} one valid patch name: top-1 not written")

    def test_decode_invariance_equals_merge_base(self):
        digests = decode_invariance_digests(self.handle, int(self.tc["vocab_size"]))
        base = self.merge_base["decode"]
        for case, digest in digests.items():
            oracle.report_row(ROW, f"L2 decode invariance: {case}", digest=digest, merge_base=base.get(case),
                              merge_base_library=self.merge_base["library_sha256"],
                              merge_base_desc_size=self.merge_base["desc_size"], passed=digest == base.get(case))
        self.assertEqual(digests, base, f"{ROW} n <= 16 forward differs from the merge-base library's")

    def test_memory_after_smoke_equals_post_generation(self):
        for pid, ids, max_new in self.prompts:
            self.check_terminal(f"{pid} memory pass", generate(self.handle, ids, max_new), max_new)
        after = device_bytes_in_use()
        delta = after - self.after_smoke
        oracle.report_row(ROW, "memory", after_smoke=self.after_smoke, after_generation=after, delta=delta,
                          bound=MEMORY_SLACK, passed=abs(delta) <= MEMORY_SLACK)
        self.assertTrue(abs(delta) <= MEMORY_SLACK,
                        f"{ROW} device memory moved {delta} bytes between SMOKE and post-generation")

    def test_decode_rows_bit_identical(self):
        """K3 (tasks/m26-spec.md D5/D6): a batched decode step's row r equals session r's decode alone."""
        vocab = int(self.tc["vocab_size"])
        gen = torch.Generator().manual_seed(ROWS_SEED)
        prompts = [synthetic_ids(gen, vocab, n) for n in ROWS_PROMPT_LENS]
        join = [j for j, _ in ROWS_JOIN_LEAVE]
        leave = [lv for _, lv in ROWS_JOIN_LEAVE]
        rows = forward_decode_rows(self.handle, prompts, join, leave, taps=True, hidden=self.hidden,
                                   layers=self.layers)
        for s, ids in enumerate(prompts):
            decodes = leave[s] - join[s]
            solo = forward_decode_rows(self.handle, [ids], [0], [decodes], taps=True, hidden=self.hidden,
                                       layers=self.layers)
            case = f"session {s} (join {join[s]}, leave {leave[s]})"
            oracle.check_bit_exact(self, ROW, f"K3 {case} scores", rows.scores[s, :decodes + 1],
                                   solo.scores[0, :decodes + 1])
            self.assertTrue(bool((rows.scores[s, decodes + 1:] == -1).all()),
                            f"{ROW} K3 {case}: a score row past its last decode was written")
            names = [n for n in solo.steps[0][0] if n.startswith(ROWS_TAP_PREFIX)]
            self.assertTrue(names, f"{ROW} K3 {case}: no {ROWS_TAP_PREFIX}* tap")
            self.assertEqual([list(step) for step in rows.steps[s]], [list(step) for step in solo.steps[0]],
                             f"{ROW} K3 {case}: the batched taps' names differ from the solo taps'")
            batched = torch.cat([step[n] for step in rows.steps[s] for n in names], dim=1)
            alone = torch.cat([step[n] for step in solo.steps[0] for n in names], dim=1)
            oracle.check_bit_exact(self, ROW, f"K3 {case} taps under {ROWS_TAP_PREFIX}", batched, alone)
            del solo, batched, alone
        del rows
        torch.cuda.empty_cache()

    def test_prefill_chunk_rounds_bit_identical(self):
        """K5 (tasks/m26-spec.md D2a): one prefill chunk per round, decode rounds between, equals the prompt alone."""
        vocab = int(self.tc["vocab_size"])
        gen = torch.Generator().manual_seed(ROWS_SEED + 5)
        shorts = [ids for _, ids, _ in parity_prompts(K5_SHORT_IDS)]
        long_ids = synthetic_ids(gen, vocab, K5_LONG)
        solo_shorts = [generate(self.handle, ids, K5_SHORT_NEW) for ids in shorts]
        # A data precondition, checked on the solo runs: a mixed-run shortfall is the implementation's and fails the
        # committed/token comparisons below instead.
        for i, g in enumerate(solo_shorts):
            if g.committed < K5_MIN_INTERLEAVED:
                raise oracle.OracleError(f"K5 short prompt {K5_SHORT_IDS[i]} commits {g.committed} tokens alone; the "
                                         f"rounds need {K5_MIN_INTERLEAVED} to interleave with every chunk")
        solo_long = generate(self.handle, long_ids, K5_LONG_NEW, chunk=K5_CHUNK)
        mixed = generate_rows(self.handle, shorts + [long_ids], [K5_SHORT_NEW] * len(shorts) + [K5_LONG_NEW],
                              K5_CHUNK)
        for case, alone, rounds in [(f"short {i}", solo_shorts[i], mixed[i]) for i in range(len(shorts))] + [
                ("3-chunk prompt", solo_long, mixed[-1])]:
            oracle.check_count(self, ROW, f"K5 {case} committed", rounds.committed, expected=alone.committed)
            self.assertEqual(rounds.finish, alone.finish, f"{ROW} K5 {case}: finish")
            oracle.check_bit_exact(self, ROW, f"K5 {case} tokens", rounds.tokens, alone.tokens)
            oracle.check_bit_exact(self, ROW, f"K5 {case} scores", rounds.scores, alone.scores)

    def test_sampled_rows_equal_alone(self):
        """Ledger CR4 / I10 rows 2-4: a sampled row in a batched step equals the same request alone."""
        vocab = int(self.tc["vocab_size"])
        prompts = [ids for _, ids, _ in parity_prompts(PARITY_IDS)]
        specs = list(CR4_SAMPLING)
        alone = [generate_rows_sampled(self.handle, [ids], [spec], CR4_NEW, vocab)[0][0]
                 for ids, spec in zip(prompts, specs)]
        # A data precondition, checked on the solo runs (as K5): a request that stops at once never shares a round.
        for i, s in enumerate(alone):
            if s.gen.committed < CR4_MIN_COMMITTED:
                raise oracle.OracleError(f"CR4 prompt {PARITY_IDS[i]} commits {s.gen.committed} tokens alone; rows "
                                         f"need {CR4_MIN_COMMITTED}")
        for rows in range(2, ROWS_R + 1):
            batched, _ = generate_rows_sampled(self.handle, prompts[:rows], specs[:rows], CR4_NEW, vocab)
            for i in range(rows):
                case = f"rows {rows} request {i} ({'greedy' if specs[i] is None else 'sampled'})"
                got, ref = batched[i], alone[i]
                oracle.check_count(self, ROW, f"CR4 {case} committed", got.gen.committed, expected=ref.gen.committed)
                self.assertEqual(got.gen.finish, ref.gen.finish, f"{ROW} CR4 {case}: finish")
                oracle.check_bit_exact(self, ROW, f"CR4 {case} tokens", got.gen.tokens, ref.gen.tokens)
                oracle.check_bit_exact(self, ROW, f"CR4 {case} scores", got.gen.scores, ref.gen.scores)
                oracle.check_bit_exact(self, ROW, f"CR4 {case} logprobs", got.logprobs, ref.logprobs)

    def test_rows4_sampler_memory(self):
        """Ledger W / CR4: the sampler's device memory at rows 4 is at most 8 MiB and a sampled rows-4 run
        allocates nothing beyond it."""
        plan = torch.zeros(4, dtype=torch.int64)
        oracle.call("cere_ktest_sampler_plan_bytes", self.handle, plan)
        scratch, proc_row, counts, max_sessions = (int(v) for v in plan.tolist())
        total = scratch + proc_row + ROWS_R * counts
        oracle.report_row(ROW, "CR4 rows-4 sampler memory (plan)", scratch=scratch, proc_logits=proc_row,
                          out_counts=counts, rows=ROWS_R, total=total, bound=CR4_SAMPLER_BYTES,
                          passed=max_sessions == ROWS_R and total <= CR4_SAMPLER_BYTES)
        self.assertEqual(max_sessions, ROWS_R, f"{ROW} CR4: the model's max_sessions")
        self.assertTrue(total <= CR4_SAMPLER_BYTES, f"{ROW} CR4: sampler memory {total} bytes at rows {ROWS_R}")
        vocab = int(self.tc["vocab_size"])
        prompts = [ids for _, ids, _ in parity_prompts(PARITY_IDS)]
        sampled = [CR4_SAMPLING[0], CR4_SAMPLING[2], CR4_SAMPLING[0], CR4_SAMPLING[3]]
        wide_before = device_bytes_in_use()  # synchronizes and empties torch's cache first
        own_before = process_device_bytes(os.getpid())
        with foreign_allocation(CR4_FOREIGN_BYTES) as foreign_pid:
            runs, _ = generate_rows_sampled(self.handle, prompts, sampled, CR4_NEW, vocab)
            wide_after = device_bytes_in_use()
            own_after = process_device_bytes(os.getpid())
            foreign = process_device_bytes(foreign_pid)
        for i, s in enumerate(runs):
            self.assertIn(s.gen.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} CR4 memory request {i}: finish")
        delta = own_after - own_before
        oracle.report_row(ROW, "CR4 rows-4 sampler memory (run)", before=own_before, after=own_after, delta=delta,
                          foreign_bytes=foreign, device_wide_delta=wide_after - wide_before, bound=CR4_SAMPLER_BYTES,
                          passed=abs(delta) <= CR4_SAMPLER_BYTES and foreign >= CR4_FOREIGN_BYTES)
        self.assertTrue(foreign >= CR4_FOREIGN_BYTES,
                        f"{ROW} CR4 control: the foreign process holds {foreign} bytes, not its allocation")
        self.assertTrue(abs(delta) <= CR4_SAMPLER_BYTES,
                        f"{ROW} CR4: this process's device memory moved {delta} bytes over a sampled "
                        f"rows-{ROWS_R} run")


# -- M2.9 speculation (K8Speculation) -------------------------------------------------------------------------------


def session_open(handle: int) -> int:
    out = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_open", handle, 0, out)
    return int(out[0])


def session_feed(session: int, ids: list[int]) -> torch.Tensor:
    """Teacher-forced feed (PREFILL on an empty session, else one DECODE per token): int32 [n, 4] scores (a
    PREFILL writes row 0 only)."""
    scores = torch.full((len(ids), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_session_feed", session, torch.tensor(ids, dtype=torch.int32), len(ids), 0, 0, None, 0,
                None, 0, None, 0, scores, counts)
    return scores


def state_digest(session: int) -> bytes:
    out = torch.zeros(32, dtype=torch.uint8)
    oracle.call("cere_ktest_state_digest", session, out)
    return bytes(out.tolist())


def draft_digest(session: int) -> bytes:
    out = torch.zeros(32, dtype=torch.uint8)
    oracle.call("cere_ktest_draft_digest", session, out)
    return bytes(out.tolist())


def tearDownModule() -> None:
    """Reported only (no verdict reads it): the wall of the digest calls (the D2H state save and its SHA-256, while
    the GPU idles), from oracle.CALL_WALL."""
    fields = {}
    for kind in ("state", "draft"):
        calls, secs = oracle.CALL_WALL.get(f"cere_ktest_{kind}_digest", (0, 0.0))
        fields[f"{kind}_calls"] = calls
        fields[f"{kind}_ms_mean"] = round(1000 * secs / calls, 2) if calls else 0.0
        fields[f"{kind}_s_total"] = round(secs, 1)
    oracle.report_row(ROW, "digest wall", **fields)


def session_verify(session: int, chain: list[int], commit: int) -> torch.Tensor:
    """One verify chain (chain[0] then len(chain) - 1 drafts) committing `commit` rows: int32 [len(chain), 4]."""
    scores = torch.full((len(chain), 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_verify", session, torch.tensor(chain, dtype=torch.int32), len(chain) - 1, commit,
                scores)
    return scores


class Speculated(NamedTuple):
    committed: list  # per row
    accepted: list
    scores: torch.Tensor  # int32 [rows, 4, 4]: row r's committed tokens' scores in [r, :committed]
    logprobs: torch.Tensor | None = None  # int32 [rows, 4, TOKEN_LOGPROBS_BYTES / 4] (sampled calls)


def session_speculate(sessions: list, tokens: list, drafts: list, max_commit: list, specs: list | None = None,
                      vocab: int = 0, masks: list | None = None) -> Speculated:
    """One speculate_rows call; `specs` (CR4_SAMPLING-shaped entries, None = greedy) selects the sampled entry, and
    `masks` (per row 4 `position_mask`-shaped masks, verify row j's at [j]; GrammarSpec GS-02) the masked one."""
    rows = len(sessions)
    flat = torch.zeros((rows, 3), dtype=torch.int32)
    for r, d in enumerate(drafts):
        flat[r, :len(d)] = torch.tensor(d, dtype=torch.int32)
    status = torch.full((rows,), 1, dtype=torch.int32)
    results = torch.zeros((rows, 2), dtype=torch.int32)
    scores = torch.full((rows, 4, 4), -1, dtype=torch.int32)
    args = (torch.tensor(sessions, dtype=torch.int64), rows, torch.tensor(tokens, dtype=torch.int32), flat,
            torch.tensor([len(d) for d in drafts], dtype=torch.int32), torch.tensor(max_commit, dtype=torch.int32))
    logprobs = None
    if specs is None:
        name = "cere_ktest_session_speculate"
        oracle.call(name, *args, status, results, scores)
    else:
        tops = torch.tensor([logprobs_top(s) for s in specs], dtype=torch.int32)
        logprobs = torch.full((rows, 4, TOKEN_LOGPROBS_BYTES // 4), -1, dtype=torch.int32)
        if masks is None:
            name = "cere_ktest_session_speculate_sampled"
            oracle.call(name, *args, sampling_bytes(specs, vocab), tops, status, results, scores, logprobs)
        else:
            name = "cere_ktest_session_speculate_masked"
            stacked = torch.stack([torch.stack(m) for m in masks]).contiguous()
            oracle.call(name, *args, sampling_bytes(specs, vocab), tops, stacked, status, results, scores, logprobs)
    if status.tolist() != [0] * rows:
        raise oracle.OracleError(f"{name} row status {status.tolist()}")
    return Speculated(results[:, 0].tolist(), results[:, 1].tolist(), scores, logprobs)


SPEC_CHAIN_MAX = 15  # model::layout::kMaxChainDrafts: drafts of one deep verify chain (DraftAccept; Chains, O123)


def session_speculate_chain(sessions: list, tokens: list, drafts: list, max_commit: list) -> Speculated:
    """One greedy speculate_rows call with caller drafts of up to SPEC_CHAIN_MAX per row (deep chains)."""
    rows = len(sessions)
    flat = torch.zeros((rows, SPEC_CHAIN_MAX), dtype=torch.int32)
    for r, d in enumerate(drafts):
        flat[r, :len(d)] = torch.tensor(d, dtype=torch.int32)
    status = torch.full((rows,), 1, dtype=torch.int32)
    results = torch.zeros((rows, 2), dtype=torch.int32)
    scores = torch.full((rows, SPEC_CHAIN_MAX + 1, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_speculate_chain", torch.tensor(sessions, dtype=torch.int64), rows,
                torch.tensor(tokens, dtype=torch.int32), flat,
                torch.tensor([len(d) for d in drafts], dtype=torch.int32),
                torch.tensor(max_commit, dtype=torch.int32), status, results, scores)
    if status.tolist() != [0] * rows:
        raise oracle.OracleError(f"cere_ktest_session_speculate_chain row status {status.tolist()}")
    return Speculated(results[:, 0].tolist(), results[:, 1].tolist(), scores)



def session_speculate_mtp(sessions: list, tokens: list, draft_counts: list, max_commit: list) -> Speculated:
    """One greedy speculate_rows call whose rows draft from their sessions' pending MTP proposals (I1)."""
    rows = len(sessions)
    status = torch.full((rows,), 1, dtype=torch.int32)
    results = torch.zeros((rows, 2), dtype=torch.int32)
    scores = torch.full((rows, 4, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_speculate_mtp", torch.tensor(sessions, dtype=torch.int64), rows,
                torch.tensor(tokens, dtype=torch.int32), torch.tensor(draft_counts, dtype=torch.int32),
                torch.tensor(max_commit, dtype=torch.int32), status, results, scores)
    if status.tolist() != [0] * rows:
        raise oracle.OracleError(f"cere_ktest_session_speculate_mtp row status {status.tolist()}")
    return Speculated(results[:, 0].tolist(), results[:, 1].tolist(), scores)


def logprobs_top(spec) -> int:
    """A CR4_SAMPLING-shaped entry's top_logprobs: -1 without logprobs."""
    return CR4_TOP_LOGPROBS if spec is not None and spec[7] else -1


class Stepped(NamedTuple):
    score: torch.Tensor  # int32 [4]
    logprobs: torch.Tensor  # int32 [TOKEN_LOGPROBS_BYTES / 4] (untouched without logprobs)
    processed: torch.Tensor  # fp32 [vocab]: the sampler stage's processed logits


def session_step_sampled(session: int, token: int, spec, vocab: int, mask: torch.Tensor | None = None) -> Stepped:
    """Plain sampled decode: one step_rows row feeding `token` with `spec` (CR4_SAMPLING-shaped, not None); `mask`
    (`position_mask`-shaped) selects the masked entry (GrammarSpec GS-02)."""
    score = torch.full((4,), -1, dtype=torch.int32)
    logprobs = torch.full((TOKEN_LOGPROBS_BYTES // 4,), -1, dtype=torch.int32)
    processed = torch.empty(vocab, dtype=torch.float32)
    sampling = sampling_bytes([spec], vocab)
    if mask is None:
        oracle.call("cere_ktest_session_step_sampled", session, token, sampling, logprobs_top(spec), score, logprobs,
                    processed)
    else:
        oracle.call("cere_ktest_session_step_masked", session, token, sampling, logprobs_top(spec), mask, score,
                    logprobs, processed)
    return Stepped(score, logprobs, processed)


def position_mask(position: int, vocab: int) -> torch.Tensor:
    """GrammarSpec GS-02's grammar stand-in: the vocab mask (int32 [ceil(vocab / 32)], bit i % 32 of word i // 32 =
    id i allowed; tasks/m28-spec.md D9) of the token at `position`: id allowed unless (7 id + position) %
    SPEC_MASK_MODULUS == 0, a different third of the vocab banned at every position."""
    words = (vocab + 31) // 32
    ids = torch.arange(words * 32, dtype=torch.int64)
    allowed = ((7 * ids + position) % SPEC_MASK_MODULUS != 0) & (ids < vocab)
    bits = (allowed.view(words, 32).to(torch.int64) << torch.arange(32, dtype=torch.int64)).sum(dim=1)
    return torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32)


def mask_allows(mask: torch.Tensor, token: int) -> bool:
    return (int(mask[token // 32]) >> (token % 32)) & 1 == 1


def spec_stages(session: int, rows: int, vocab: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The last speculate_rows call's verify-row stages: processed and probs fp32 [rows, vocab], keep int32 words."""
    processed = torch.empty((rows, vocab), dtype=torch.float32)
    probs = torch.empty((rows, vocab), dtype=torch.float32)
    keep = torch.empty((rows, (vocab + 31) // 32), dtype=torch.int32)
    oracle.call("cere_ktest_spec_stages", session, rows, processed, probs, keep)
    return processed, probs, keep


def session_counts(session: int, vocab: int) -> torch.Tensor:
    out = torch.zeros(vocab, dtype=torch.int32)
    oracle.call("cere_ktest_session_counts", session, out)
    return out.long()


def committed_score(processed: torch.Tensor, token: int) -> torch.Tensor:
    """k12::finalize's cere_token_score of `token` on a processed row: its logit, the runner-up (the most probable
    other id, ties to the lower id) and its logit, as int32 [4] (float bits)."""
    top1 = int(torch.argmax(processed))
    runner = top1
    if token == top1:
        rest = processed.clone()
        rest[top1] = -math.inf
        runner = int(torch.argmax(rest))
    logits = torch.tensor([processed[token], processed[runner]], dtype=torch.float32).view(torch.int32)
    return torch.tensor([token, runner, int(logits[0]), int(logits[1])], dtype=torch.int32)


def expected_logprobs(processed: torch.Tensor, temperature: float, token: int, top: int) -> torch.Tensor:
    """order 16's cere_token_logprobs of `token` on a processed row (K12 top_logprobs): id, logprob, top_count and the
    top entries, as int32 [TOKEN_LOGPROBS_BYTES / 4] with the reserved word and unused entries zero."""
    logprob = oracle.empty((1,), torch.float32)
    ids = oracle.empty((top,), torch.int32)
    values = oracle.empty((top,), torch.float32)
    oracle.call("cere_ktest_sample_top_logprobs", processed.to(oracle.DEVICE).contiguous(), processed.numel(),
                temperature, token, top, logprob, ids, values)
    out = torch.zeros(TOKEN_LOGPROBS_BYTES // 4, dtype=torch.int32)
    out[0] = token
    out[1] = logprob.cpu().view(torch.int32)[0]
    out[2] = top
    out[4:4 + 2 * top:2] = ids.cpu()
    out[5:5 + 2 * top:2] = values.cpu().view(torch.int32)
    return out


def d6_chain(probs: torch.Tensor, keep: torch.Tensor, drafts: list, seed: int, key0: int) -> list[int]:
    """The D6 rule on one chain's verify-row stages (rows 0..n, n = len(drafts)): row j accepts d_{j+1} with the
    coin at key key0 + j (channels 4, 5), else commits its resample (6, 7) and stops; after n accepts the bonus is
    row n's plain draw (0, 1) at key0 + n."""
    ids = []
    for j, d in enumerate(drafts):
        ref = oracle.D6RowRef(probs[j], oracle.d6_kept(keep[j]), d)
        got, flags = oracle.d6_ref_row(ref, [seed], [key0 + j])
        ids.append(got[0])
        if not flags[0]:
            return ids
    bonus = oracle.D6RowRef(probs[len(drafts)], oracle.d6_kept(keep[len(drafts)]), None)
    ids.append(bonus.draw(oracle.d6_uniforms([seed], [key0 + len(drafts)], 0, 1)[0]))
    return ids


class PlainRun(NamedTuple):
    tokens: list  # tokens[0] = the PREFILL's greedy token, tokens[i + 1] = the greedy token after feeding tokens[i]
    prefill: torch.Tensor  # int32 [4]: the PREFILL's score
    scores: torch.Tensor  # int32 [steps, 4]: scores[i] after feeding tokens[i]
    digests: dict  # fed tokens -> state digest (when asked)


def plain_greedy(handle: int, prompt: list[int], steps: int, *, digests: bool = False) -> PlainRun:
    """Greedy decode of `steps` tokens, one teacher-forced DECODE per token on a session that does not draft."""
    with opened_session(handle) as session:
        prefill = session_feed(session, prompt)[0].clone()
        tokens = [int(prefill[0])]
        rows = []
        states = {}
        for _ in range(steps):
            row = session_feed(session, tokens[-1:])[0].clone()
            rows.append(row)
            tokens.append(int(row[0]))
            if digests:
                states[len(prompt) + len(rows)] = state_digest(session)
        return PlainRun(tokens, prefill, torch.stack(rows), states)


def session_close(session: int) -> None:
    oracle.call("cere_ktest_session_close", session)


@contextlib.contextmanager
def opened_session(handle: int, *, draft: bool = False):
    """A session of the model (drafting with `draft`, set while it is empty), closed on exit."""
    session = session_open(handle)
    try:
        if draft:
            oracle.call("cere_ktest_session_draft", session, 1)
        yield session
    finally:
        session_close(session)


def wrong(token: int, vocab: int) -> int:
    return (token + 1) % vocab


def decode_graph(handle: int, mode: int) -> tuple:
    """`cere_ktest_decode_graph` (M2.3 P1): switch (mode 0 / 1) or query (-1) the model's decode graphs; (enabled,
    graphs instantiated, executables updated in place, steps run by a graph)."""
    out = torch.zeros(4, dtype=torch.int64)
    oracle.call("cere_ktest_decode_graph", handle, mode, out)
    return tuple(int(v) for v in out.tolist())


def verify_attention(handle: int, mode: int) -> tuple:
    """`cere_ktest_verify_attention` (tasks/m23-spec.md P11R-19): switch (mode 0 per_row / 1 batched) or query (-1)
    the model's verify_attention lever; (batched, attention_chain calls since INSTANTIATE)."""
    out = torch.zeros(2, dtype=torch.int64)
    oracle.call("cere_ktest_verify_attention", handle, mode, out)
    return tuple(int(v) for v in out.tolist())


def spec_graph(handle: int, mode: int) -> tuple:
    """`cere_ktest_spec_graph` (tasks/m23-spec.md P11R-20): switch (mode 0 off / 1 on) or query (-1) the model's
    spec graphs: (on, verify graphs instantiated, verify executables updated in place, VERIFY calls run by a graph,
    draft graphs instantiated, draft executables updated in place, drafts run by a graph)."""
    out = torch.zeros(7, dtype=torch.int64)
    oracle.call("cere_ktest_spec_graph", handle, mode, out)
    return tuple(int(v) for v in out.tolist())


def kernel_launches(parts: tuple[str, ...], fn) -> int:
    """The launches of the CUDA kernels whose name holds every one of `parts` that fn() ran, under torch.profiler."""
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise oracle.OracleError("torch.profiler has no CUDA activity (CUPTI) in this image")
    torch.cuda.synchronize(oracle.DEVICE)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize(oracle.DEVICE)
    return sum(all(p in e.name() for p in parts) for e in prof.profiler.kineto_results.events()
               if e.device_type() == torch.autograd.DeviceType.CUDA)


def top2_launches(fn) -> int:
    """The K1 top-2 kernel launches fn() ran (one per lm_head call: its rows' top-2)."""
    return kernel_launches(("cere::k1::", "top2_kernel"), fn)


class K8Speculation(unittest.TestCase):
    """M2.9 phase G on its own drafting model (module docstring K8Speculation)."""

    LEVERS: dict[str, int] = {}  # the model's slot choices, lever -> CERE_LEVER_* (empty: null selection, every default)

    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.hidden = int(cls.tc["hidden_size"])
        cls.layers = int(cls.tc["num_hidden_layers"])
        cls.vocab = int(cls.tc["vocab_size"])
        levers = oracle.levers(**cls.LEVERS) if cls.LEVERS else None
        model = create_model(SPEC_CONTEXT, SPEC_SESSIONS, drafts=SPEC_DRAFT_TOKENS, model_levers=levers,
                             draft_token_map=cls.draft_token_map())
        cls.addClassCleanup(release_model, model.handle)
        upload_and_seal(model)
        cls.handle = model.handle

    @classmethod
    def draft_token_map(cls) -> torch.Tensor | None:
        """The drafting model's draft token map (None: every vocab id)."""
        return None

    def drafting_session(self, stack: contextlib.ExitStack, prompt: list[int]) -> tuple[int, torch.Tensor]:
        """A drafting session fed `prompt` (closed with `stack`) and its PREFILL score."""
        session = stack.enter_context(opened_session(self.handle, draft=True))
        return session, session_feed(session, prompt)[0].clone()

    def test_spec_undo_round_trip(self):
        """K1' (M29S2-20): verify(n = 3) + commit(a) leaves the state of a plain feed of the same a tokens."""
        gen = torch.Generator().manual_seed(SPEC_SEED)
        prompt = synthetic_ids(gen, self.vocab, SPEC_UNDO_PROMPT)
        for a in range(1, 5):
            with self.subTest(commit=a), contextlib.ExitStack() as stack:
                spec, spec_prefill = self.drafting_session(stack, prompt)
                plain = stack.enter_context(opened_session(self.handle))
                plain_prefill = session_feed(plain, prompt)[0]
                oracle.check_bit_exact(self, ROW, f"K1p a={a} prefill score", spec_prefill, plain_prefill)
                token = int(spec_prefill[0])
                for round_ in range(SPEC_UNDO_ROUNDS):
                    chain = [token] + synthetic_ids(gen, self.vocab, 3)
                    verified = session_verify(spec, chain, a)
                    fed = session_feed(plain, chain[:a])
                    case = f"K1p a={a} round {round_}"
                    oracle.check_bit_exact(self, ROW, f"{case} scores", verified[:a], fed)
                    self.assertEqual(state_digest(spec), state_digest(plain), f"{ROW} {case}: state digest")
                    token = int(verified[a - 1, 0])

    def speculate_run(self, prompts: list, truth: list) -> None:
        """Every session decodes SPEC_DECODE tokens by speculate_rows rounds (all live sessions in one call) with
        drafts = the true next tokens, the one at SPEC_WRONG_ROWS[(round + r) % 4] made wrong."""
        with contextlib.ExitStack() as stack:
            self.speculate_sessions(stack, prompts, truth)

    def speculate_sessions(self, stack: contextlib.ExitStack, prompts: list, truth: list, specs: list | None = None,
                           label: str = "K4") -> None:
        """`specs` (one CR4_SAMPLING-shaped entry or None per session): the rows' sampling, else every row greedy."""
        rows = len(prompts)
        sessions = []
        for r, ids in enumerate(prompts):
            session, prefill = self.drafting_session(stack, ids)
            oracle.check_bit_exact(self, ROW, f"{label} R={rows} session {r} prefill", prefill, truth[r].prefill)
            sessions.append(session)
        at = [0] * rows  # truth[r].tokens[at[r]]: the session's committed, unfed token
        round_ = 0
        while any(i < SPEC_DECODE for i in at):
            live = [r for r in range(rows) if at[r] < SPEC_DECODE]
            tokens, drafts, max_commit, expect = [], [], [], []
            for r in live:
                i = at[r]
                commit = min(4, SPEC_DECODE - i)
                d = truth[r].tokens[i + 1:i + commit]
                w = SPEC_WRONG_ROWS[(round_ + r) % len(SPEC_WRONG_ROWS)]
                if w is not None and w < len(d):
                    d[w] = wrong(d[w], self.vocab)
                    expect.append(w + 1)
                else:
                    expect.append(len(d) + 1)
                tokens.append(truth[r].tokens[i])
                drafts.append(d)
                max_commit.append(commit)
            live_specs = None if specs is None else [specs[r] for r in live]
            got = session_speculate([sessions[r] for r in live], tokens, drafts, max_commit, live_specs, self.vocab)
            for p, r in enumerate(live):
                case = f"{label} R={rows} session {r} round {round_} (position {len(prompts[r]) + at[r]})"
                c = got.committed[p]
                self.assertEqual((c, got.accepted[p]), (expect[p], expect[p] - 1), f"{ROW} {case}: committed")
                oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[p, :c],
                                       truth[r].scores[at[r]:at[r] + c])
                at[r] += c
            round_ += 1
        oracle.report_row(ROW, f"{label} R={rows}", rounds=round_, committed=rows * SPEC_DECODE)

    def test_spec_greedy_bit_identical(self):
        """K4 (D4, D5): greedy speculation with caller drafts commits exactly the plain greedy tokens and scores."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 1)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        truth = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        for rows in range(1, SPEC_SESSIONS + 1):  # every row count (P11R-21: lossless at rows 1 .. 4)
            with self.subTest(rows=rows):
                self.speculate_run(prompts[:rows], truth[:rows])

    def mtp_sessions(self, stack: contextlib.ExitStack, prompts: list, truth: list, label: str = "K4m",
                     digests: list | None = None) -> int:
        """Every session decodes SPEC_DECODE greedy tokens by speculate_rows rounds (all live sessions in one call)
        drafting from its pending MTP proposals; returns the accepted drafts. With `digests` (one list per session)
        each live session's draft_digest is appended after every round."""
        rows = len(prompts)
        sessions = []
        for r in range(rows):
            session, prefill = self.drafting_session(stack, prompts[r])
            oracle.check_bit_exact(self, ROW, f"{label} R={rows} session {r} prefill", prefill, truth[r].prefill)
            sessions.append(session)
        at = [0] * rows
        round_, accepted = 0, 0
        while any(i < SPEC_DECODE for i in at):
            live = [r for r in range(rows) if at[r] < SPEC_DECODE]
            commits = [min(4, SPEC_DECODE - at[r]) for r in live]
            got = session_speculate_mtp([sessions[r] for r in live], [truth[r].tokens[at[r]] for r in live],
                                        [c - 1 for c in commits], commits)
            for p, r in enumerate(live):
                case = f"{label} R={rows} session {r} round {round_} (position {len(prompts[r]) + at[r]})"
                c, a = got.committed[p], got.accepted[p]
                self.assertTrue(1 <= c <= commits[p] and a == c - 1, f"{ROW} {case}: committed {c} accepted {a}")
                oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[p, :c],
                                       truth[r].scores[at[r]:at[r] + c])
                at[r] += c
                accepted += a
                if digests is not None:
                    digests[r].append((state_digest(sessions[r]), draft_digest(sessions[r])))
            round_ += 1
        oracle.report_row(ROW, f"{label} R={rows}", rounds=round_, committed=rows * SPEC_DECODE, accepted=accepted,
                          accept_length=round(rows * SPEC_DECODE / round_, 4))
        return accepted

    def test_spec_mtp_greedy_bit_identical(self):
        """K4 with MTP drafts (I1, D2, D5): R in 1 .. SPEC_SESSIONS drafting sessions, SPEC_DECODE greedy tokens each
        through speculate_rows rounds drafting from the pending MTP proposals (draft-extend after PREFILL and after
        every commit, 2 MTP decode steps): every committed token and score equals the plain run, and some drafts are
        accepted (the proposals reach the verifier). The accept length (committed / rounds) is reported."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 1)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        truth = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        for rows in range(1, SPEC_SESSIONS + 1):
            with self.subTest(rows=rows), contextlib.ExitStack() as stack:
                accepted = self.mtp_sessions(stack, prompts[:rows], truth[:rows])
                self.assertTrue(accepted > 0, f"{ROW} K4m R={rows}: no MTP draft accepted")

    def test_spec_mtp_rows_draft_bit_identical(self):
        """P11R-21 (CudaDevice::draft_rows): R in 2 .. SPEC_SESSIONS drafting sessions speculating together draft in
        one MTP forward per pass over all their rows; after every round each session's draft digest (state, MTP pool
        entry, pending proposals) equals the one of the same session speculating alone (CudaDevice::draft), and its
        committed scores equal plain greedy's."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 1)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        truth = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        alone = []
        for r in range(SPEC_SESSIONS):
            with contextlib.ExitStack() as stack:
                digests = [[]]
                self.mtp_sessions(stack, prompts[r:r + 1], truth[r:r + 1], label=f"K4m alone {r}", digests=digests)
                alone.append(digests[0])
        for rows in range(2, SPEC_SESSIONS + 1):
            with self.subTest(rows=rows), contextlib.ExitStack() as stack:
                digests = [[] for _ in range(rows)]
                self.mtp_sessions(stack, prompts[:rows], truth[:rows], label="K4m rows", digests=digests)
                for r in range(rows):
                    # Round by round first, so a mismatch names the first round that differs and whether the target
                    # state (state_digest) or only the draft part (the MTP pool entry, the proposals) does.
                    for j, (got, want) in enumerate(zip(digests[r], alone[r])):
                        part = "target state" if got[0] != want[0] else "MTP entry / proposals"
                        self.assertEqual(got, want, f"{ROW} P11R-21 R={rows} session {r} round {j}: {part} digest")
                    self.assertEqual(len(digests[r]), len(alone[r]), f"{ROW} P11R-21 R={rows} session {r}: rounds")

    def test_spec_mtp_feed_extends(self):
        """P11R-21 (CudaDevice::extend_draft): the kernel-test feed (session_feed) leaves a drafting session in the
        draft state a PREFILL work item (CudaDevice::step) leaves it in: equal PREFILL scores and draft digests (MTP
        pool entry, pending proposals) per SPEC_PROMPT_LENS prompt. Without the extend the entry's prompt rows keep a
        pool page's earlier bytes, which the MTP attends to, so a session's drafts depended on the pages' history."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 31)
        for n in SPEC_PROMPT_LENS:
            ids = synthetic_ids(gen, self.vocab, n)
            with self.subTest(n=n), contextlib.ExitStack() as stack:
                fed, prefill = self.drafting_session(stack, ids)
                stepped = stack.enter_context(opened_session(self.handle, draft=True))
                scores = torch.full((1, 4), -1, dtype=torch.int32)
                counts = torch.zeros(3, dtype=torch.int64)
                oracle.call("cere_ktest_session_feed", stepped, torch.tensor(ids, dtype=torch.int32), n, 1, 0, None, 0,
                            None, 0, None, 0, scores, counts)
                self.assertEqual(int(counts[0]), 1, f"{ROW} P11R-21 feed n={n}: PREFILL item commits")
                oracle.check_bit_exact(self, ROW, f"P11R-21 feed n={n} prefill score", prefill, scores[0])
                same = draft_digest(fed) == draft_digest(stepped)
                oracle.report_row(ROW, f"P11R-21 feed n={n} draft digest", phase="bit_exact", passed=same)
                self.assertTrue(same, f"{ROW} P11R-21 feed n={n}: draft digest of the fed session != the stepped one")

    def test_rows_draft_one_mtp_pass(self):
        """P11R-21 (CudaDevice::draft_rows): one speculate_rows call of SPEC_SESSIONS drafting sessions drafts them
        together: 1 + 2 MTP argmax launches (the extends' drafts, then each decode step's), not 3 per session."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 21)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        with contextlib.ExitStack() as stack:
            sessions, tokens = [], []
            for ids in prompts:
                session, prefill = self.drafting_session(stack, ids)
                sessions.append(session)
                tokens.append(int(prefill[0]))
            rows = len(sessions)
            argmax = kernel_launches(("argmax_map_kernel",), lambda: session_speculate_mtp(
                sessions, tokens, [3] * rows, [4] * rows))
        oracle.report_row(ROW, "P11R-21 draft rows", rows=rows, mtp_argmax_launches=argmax)
        self.assertEqual(argmax, 3, f"{ROW} P11R-21: MTP argmax launches of a {rows}-session round")

    def test_rows_lm_head_one_launch(self):
        """P11R-21 (RowsPlans::lm_head): one speculate_rows call of SPEC_SESSIONS chains of 3 caller drafts (a verify
        of m = 16 rows) runs one K1 top-2 launch, so the lm_head once at m, on the model's levers (the default
        cuBLASLt plan too); a 2-round forward_decode_rows of SPEC_SESSIONS rows runs one per prefill and one per
        batched step. K4 and the K8 rows tests hold the scores to plain greedy's / the solo step's."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 21)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        with contextlib.ExitStack() as stack:
            sessions, tokens = [], []
            for ids in prompts:
                session, prefill = self.drafting_session(stack, ids)
                sessions.append(session)
                tokens.append(int(prefill[0]))
            drafts = [synthetic_ids(gen, self.vocab, 3) for _ in sessions]
            verify = top2_launches(lambda: session_speculate(sessions, tokens, drafts, [4] * len(sessions)))
        rounds = 2
        decode = top2_launches(lambda: forward_decode_rows(self.handle, prompts, [0] * len(prompts),
                                                           [rounds] * len(prompts), taps=False, hidden=self.hidden,
                                                           layers=self.layers))
        oracle.report_row(ROW, "P11R-21 lm_head rows", rows=len(prompts), verify_top2_launches=verify,
                          decode_top2_launches=decode)
        self.assertEqual(verify, 1, f"{ROW} P11R-21: verify of m = {4 * len(prompts)} top-2 launches")
        self.assertEqual(decode, len(prompts) + rounds, f"{ROW} P11R-21: decode rows top-2 launches")

    def test_spec_round_never_enters_graph(self):
        """M2.3 P1 x M2.9 (tasks/m23-spec.md M23P1-11, M29S2-18): the decode graph serves plain decode only. Plain
        greedy decode with the model's decode graphs off equals the run with them on bit for bit, and that run
        replayed one graph per decode step; speculation with caller drafts (K4) and MTP drafts (K4m), R in
        {1, SPEC_SESSIONS}, with the graphs on commits the plain scores while no decode-graph counter moves; nor do
        M2.9L's policy-picked rounds (tasks/m29l-spec.md D5: none, MTP or suffix-lookup drafts, C6L's arms), which a
        drafting session also runs through speculate_rows."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 1)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        self.addCleanup(decode_graph, self.handle, decode_graph(self.handle, -1)[0])
        decode_graph(self.handle, 0)
        eager = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        before = decode_graph(self.handle, 1)
        truth = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        graphed = decode_graph(self.handle, -1)
        for r, ids in enumerate(prompts):
            case = f"P1 plain session {r} ({len(ids)} + {SPEC_DECODE})"
            oracle.check_bit_exact(self, ROW, f"{case} prefill: graph vs eager", truth[r].prefill, eager[r].prefill)
            oracle.check_bit_exact(self, ROW, f"{case} scores: graph vs eager", truth[r].scores, eager[r].scores)
        self.assertEqual(graphed[3] - before[3], len(prompts) * SPEC_DECODE,
                         f"{ROW} P1 plain: every plain decode step runs by a graph")
        for rows in (1, SPEC_SESSIONS):
            with self.subTest(rows=rows):
                start = decode_graph(self.handle, -1)
                with contextlib.ExitStack() as stack:
                    self.speculate_sessions(stack, prompts[:rows], truth[:rows], label="P1 K4")
                with contextlib.ExitStack() as stack:
                    self.mtp_sessions(stack, prompts[:rows], truth[:rows], label="P1 K4m")
                end = decode_graph(self.handle, -1)
                self.assertEqual(end[1:], start[1:], f"{ROW} P1 R={rows}: speculation moved the decode-graph "
                                                     f"counters (captures, updates, replays) {start[1:]} -> {end[1:]}")
        prefix, block, repeats = SPEC_LOOKUP_SHAPES[0]
        prompt = synthetic_ids(gen, self.vocab, prefix) + synthetic_ids(gen, self.vocab, block) * repeats
        start = decode_graph(self.handle, -1)
        lookup_steps = 0
        for _, flags in SPEC_LOOKUP_ARMS[1:]:
            lookup_steps += generate_flags(self.handle, prompt, SPEC_DECODE, flags)[1]
        self.assertTrue(lookup_steps > 0, f"{ROW} P1 M2.9L: no policy-picked round ran")
        end = decode_graph(self.handle, -1)
        self.assertEqual(end[1:], start[1:], f"{ROW} P1 M2.9L: policy-picked rounds moved the decode-graph counters "
                                             f"(captures, updates, replays) {start[1:]} -> {end[1:]}")
        oracle.report_row(ROW, "P1 spec never graphed", plain_steps=len(prompts) * SPEC_DECODE,
                          replays=graphed[3] - before[3], captures=graphed[1] - before[1],
                          updates=graphed[2] - before[2], lookup_rounds=lookup_steps)

    def test_verify_attention_batched_bit_identical(self):
        """P11R-19 (the verify_attention lever): with the lever at batched, speculation with caller drafts (K4) and
        MTP drafts (K4m), R in {1, SPEC_SESSIONS}, commits plain greedy's tokens and scores, and K1''s undo round trip
        and K7's QSA rollback across 2,048 leave plain's state digest; the attention_chain counter moves under batched
        and stays under per_row. Fails on a library without the lever (no cere_ktest_verify_attention)."""
        self.addCleanup(verify_attention, self.handle, verify_attention(self.handle, -1)[0])
        gen = torch.Generator().manual_seed(SPEC_SEED + 1)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        verify_attention(self.handle, 0)
        truth = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        start = verify_attention(self.handle, -1)
        with contextlib.ExitStack() as stack:
            self.speculate_sessions(stack, prompts[:1], truth[:1], label="P11R-19 per_row K4")
        self.assertEqual(verify_attention(self.handle, -1), start,
                         f"{ROW} P11R-19: per_row ran a batched verify chain")
        verify_attention(self.handle, 1)
        chains = 0
        for rows in (1, SPEC_SESSIONS):
            with self.subTest(rows=rows):
                start = verify_attention(self.handle, -1)
                with contextlib.ExitStack() as stack:
                    self.speculate_sessions(stack, prompts[:rows], truth[:rows], label="P11R-19 K4")
                with contextlib.ExitStack() as stack:
                    accepted = self.mtp_sessions(stack, prompts[:rows], truth[:rows], label="P11R-19 K4m")
                self.assertTrue(accepted > 0, f"{ROW} P11R-19 K4m R={rows}: no MTP draft accepted")
                end = verify_attention(self.handle, -1)
                self.assertEqual(end[0], 1, f"{ROW} P11R-19 R={rows}: the lever left batched")
                self.assertTrue(end[1] > start[1], f"{ROW} P11R-19 R={rows}: no verify chain ran batched")
                chains += end[1] - start[1]
        with self.subTest(case="K1p undo"):
            self.test_spec_undo_round_trip()
        with self.subTest(case="K7 qsa rollback"):
            self.test_spec_qsa_rollback_crosses_2048()
        oracle.report_row(ROW, "P11R-19 verify_attention batched", chains=chains)

    def spec_graph_trace(self, prompt: list[int], chains: list) -> tuple:
        """`prompt` fed to a drafting session, then one session_verify per (chain, commit) of `chains`: the PREFILL
        score, every round's scores and the state digest after every round."""
        with contextlib.ExitStack() as stack:
            session, prefill = self.drafting_session(stack, prompt)
            scores, digests = [], []
            for chain, commit in chains:
                scores.append(session_verify(session, chain, commit))
                digests.append(state_digest(session))
        return prefill, scores, digests

    def spec_graph_mtp_trace(self, stack: contextlib.ExitStack, prompts: list, truth: list) -> list:
        """K4m on drafting sessions of `prompts` fed `truth`'s tokens, max_commit cycling 1..SPEC_DRAFT_TOKENS (so
        every draft-extend row count runs), every committed score checked against `truth`: per round (committed,
        accepted, every live row's committed scores, every session's draft digest)."""
        rows = len(prompts)
        sessions = [self.drafting_session(stack, prompts[r])[0] for r in range(rows)]
        at, trace = [0] * rows, []
        while any(i < SPEC_DECODE for i in at):
            live = [r for r in range(rows) if at[r] < SPEC_DECODE]
            cycle = len(trace) % SPEC_DRAFT_TOKENS + 1
            commits = [min(cycle, SPEC_DECODE - at[r]) for r in live]
            got = session_speculate_mtp([sessions[r] for r in live], [truth[r].tokens[at[r]] for r in live],
                                        [c - 1 for c in commits], commits)
            scores = []
            for p, r in enumerate(live):
                c = got.committed[p]
                oracle.check_bit_exact(self, ROW, f"P11R-20 K4m trace R={rows} session {r} round {len(trace)}",
                                       got.scores[p, :c], truth[r].scores[at[r]:at[r] + c])
                scores.append(got.scores[p, :c].clone())
                at[r] += c
            trace.append((got.committed, got.accepted, scores, [draft_digest(s) for s in sessions]))
        return trace

    def test_spec_graph_bit_identical(self):
        """P11R-20 / M29S2-18b (decode_step = spec_graph): verify graphs vs eager VERIFY, bit for bit. Caller chains
        of every m in 1..SPEC_DRAFT_TOKENS (SPEC_GRAPH_ROUNDS) from a short prompt and from SPEC_QSA_PROMPT (rounds
        past 2,048 fall back to eager), under per_row and batched verify attention: equal scores and state digests
        per round, and exactly the eligible rounds replay. MTP draft graphs vs eager drafts: K4m rounds of every
        draft-extend row count give equal committed counts, accepted drafts, scores and draft digests, and drafts
        replay a graph. K4 / K4m speculation (R in {1, SPEC_SESSIONS}) commits plain greedy's tokens and scores
        through the graphs; K1''s undo and K7's QSA rollback leave plain's digest; sampled and tapped verify calls stay
        eager. Fails on a library without the lever (no cere_ktest_spec_graph)."""
        self.addCleanup(spec_graph, self.handle, spec_graph(self.handle, -1)[0])
        self.addCleanup(verify_attention, self.handle, verify_attention(self.handle, -1)[0])
        gen = torch.Generator().manual_seed(SPEC_SEED + 6)
        cases = []
        for length in (SPEC_PROMPT_LENS[1], SPEC_QSA_PROMPT):
            prompt = synthetic_ids(gen, self.vocab, length)
            chains, position, eligible = [], length, 0
            for n, commit in SPEC_GRAPH_ROUNDS:
                chains.append((synthetic_ids(gen, self.vocab, n + 1), commit))
                eligible += 1 if position + n + 1 <= 2048 else 0  # p > 0: a prefilled session
                position += commit
            self.assertTrue(0 < eligible, f"{ROW} P11R-20: no eligible round from {length}")
            cases.append((prompt, chains, eligible))
        self.assertTrue(cases[1][2] < len(SPEC_GRAPH_ROUNDS), f"{ROW} P11R-20: no fallback round past 2,048")
        rounds = replays = 0
        for batched in (0, 1):
            verify_attention(self.handle, batched)
            for prompt, chains, eligible in cases:
                case = f"P11R-20 {('per_row', 'batched')[batched]} from {len(prompt)}"
                spec_graph(self.handle, 0)
                start = spec_graph(self.handle, -1)
                eager = self.spec_graph_trace(prompt, chains)
                self.assertEqual(spec_graph(self.handle, 1)[1:], start[1:], f"{ROW} {case}: eager moved the counters")
                graphed = self.spec_graph_trace(prompt, chains)
                end = spec_graph(self.handle, -1)
                oracle.check_bit_exact(self, ROW, f"{case} prefill", graphed[0], eager[0])
                for i, (n, commit) in enumerate(SPEC_GRAPH_ROUNDS):
                    label = f"{case} round {i} (n {n}, commit {commit})"
                    oracle.check_bit_exact(self, ROW, f"{label} scores", graphed[1][i], eager[1][i])
                    self.assertEqual(graphed[2][i], eager[2][i], f"{ROW} {label}: state digest")
                self.assertEqual(end[3] - start[3], eligible, f"{ROW} {case}: eligible rounds replayed")
                self.assertTrue(end[1] > 0, f"{ROW} {case}: no verify graph instantiated")
                rounds += len(chains)
                replays += end[3] - start[3]
        gen = torch.Generator().manual_seed(SPEC_SEED + 1)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        truth = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        draft_rounds = draft_replays = 0
        for batched, rows in ((0, 1), (1, SPEC_SESSIONS)):
            verify_attention(self.handle, batched)
            case = f"P11R-20 draft {('per_row', 'batched')[batched]} R={rows}"
            spec_graph(self.handle, 0)
            start = spec_graph(self.handle, -1)
            with contextlib.ExitStack() as stack:
                eager = self.spec_graph_mtp_trace(stack, prompts[:rows], truth[:rows])
            self.assertEqual(spec_graph(self.handle, 1)[1:], start[1:], f"{ROW} {case}: eager moved the counters")
            with contextlib.ExitStack() as stack:
                graphed = self.spec_graph_mtp_trace(stack, prompts[:rows], truth[:rows])
            end = spec_graph(self.handle, -1)
            self.assertEqual(len(graphed), len(eager), f"{ROW} {case}: rounds")
            for i, (g, e) in enumerate(zip(graphed, eager)):
                label = f"{case} round {i}"
                self.assertEqual(g[:2], e[:2], f"{ROW} {label}: committed / accepted")
                for p, (gs, es) in enumerate(zip(g[2], e[2])):
                    oracle.check_bit_exact(self, ROW, f"{label} row {p} scores", gs, es)
                self.assertEqual(g[3], e[3], f"{ROW} {label}: draft digests")
            self.assertTrue(sum(sum(a) for _, a, _, _ in eager) > 0, f"{ROW} {case}: no MTP draft accepted")
            # R > 1: rounds with 2+ live sessions draft together (draft_rows, P11R-21), eager.
            if rows == 1:
                self.assertTrue(end[4] > 0 and end[6] > start[6], f"{ROW} {case}: no draft ran by a graph")
            self.assertTrue(end[3] > start[3], f"{ROW} {case}: no VERIFY ran by a graph")
            draft_rounds += len(graphed)
            draft_replays += end[6] - start[6]
        spec_graph(self.handle, 1)
        for batched, rows in ((0, SPEC_SESSIONS), (1, 1), (1, SPEC_SESSIONS)):
            with self.subTest(batched=batched, rows=rows):
                verify_attention(self.handle, batched)
                start = spec_graph(self.handle, -1)
                with contextlib.ExitStack() as stack:
                    self.speculate_sessions(stack, prompts[:rows], truth[:rows], label="P11R-20 K4")
                with contextlib.ExitStack() as stack:
                    accepted = self.mtp_sessions(stack, prompts[:rows], truth[:rows], label="P11R-20 K4m")
                self.assertTrue(accepted > 0, f"{ROW} P11R-20 K4m R={rows}: no MTP draft accepted")
                end = spec_graph(self.handle, -1)
                self.assertTrue(end[3] > start[3], f"{ROW} P11R-20 R={rows}: no VERIFY ran by a graph")
                replays += end[3] - start[3]
        with self.subTest(case="K1p undo"):
            self.test_spec_undo_round_trip()
        with self.subTest(case="K7 qsa rollback"):
            self.test_spec_qsa_rollback_crosses_2048()
        start = spec_graph(self.handle, -1)
        with self.subTest(case="T0 sampled"):
            self.test_spec_sampled_t0_digest()
        with self.subTest(case="K8 forced verify trace"):
            self.test_forced_verify_trace()
        self.assertEqual(spec_graph(self.handle, -1)[:4], start[:4],
                         f"{ROW} P11R-20: a sampled or tapped VERIFY moved the verify-graph counters")
        oracle.report_row(ROW, "P11R-20 spec_graph", caller_rounds=rounds, replays=replays, captures=start[1],
                          updates=start[2], draft_rounds=draft_rounds, draft_replays=draft_replays,
                          draft_captures=start[4], draft_updates=start[5])

    def test_spec_lookup_greedy_bit_identical(self):
        """C6L (tasks/m29l-spec.md D1, D2, D4, D5): greedy requests whose cere_work.flags let the suffix lookup's drafts
        be verified (with or without the MTP proposals) commit exactly the plain greedy tokens and scores."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 5)
        accepted_lookup = 0
        for prefix, block, repeats in SPEC_LOOKUP_SHAPES:
            unit = synthetic_ids(gen, self.vocab, block)
            prompt = synthetic_ids(gen, self.vocab, prefix) + unit * repeats
            truth = plain_greedy(self.handle, prompt, SPEC_DECODE - 1)
            want_scores = torch.cat([truth.prefill.unsqueeze(0), truth.scores])
            for arm, flags in SPEC_LOOKUP_ARMS:
                case = f"C6L prompt {len(prompt)} {arm}"
                with self.subTest(case=case):
                    g, steps, accepted, _ = generate_flags(self.handle, prompt, SPEC_DECODE, flags)
                    self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} {case}: finish {g.finish}")
                    self.assertTrue(g.committed >= 1, f"{ROW} {case}: nothing committed")
                    self.assertEqual(g.tokens.tolist(), truth.tokens[:g.committed], f"{ROW} {case}: tokens")
                    oracle.check_bit_exact(self, ROW, f"{case} scores", g.scores, want_scores[:g.committed])
                    if arm == "lookup":
                        accepted_lookup += accepted
                    oracle.report_row(ROW, case, committed=g.committed, verify_steps=steps, accepted=accepted,
                                      accept_length=round(g.committed / max(steps, 1), 4))
        self.assertTrue(accepted_lookup > 0, f"{ROW} C6L: no lookup draft accepted")

    def test_spec_adaptive_greedy_bit_identical(self):
        """A4 (tasks/m29a-spec.md A2, A4): greedy requests under the fixed policy (flags 0) and the adaptive one (bit 4,
        alone and with bit 2) commit exactly the plain greedy tokens and scores. The fixed arm verifies all 3 pending
        proposals in every round whose budget fits them (every round but at most the last 3)."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 6)
        prefix, block, repeats = SPEC_LOOKUP_SHAPES[0]
        prompts = [(pid, ids) for pid, ids, _ in parity_prompts()]
        prompts.append(("block", synthetic_ids(gen, self.vocab, prefix) + synthetic_ids(gen, self.vocab, block) * repeats))
        totals = {arm: [0, 0, 0, 0] for arm, _ in SPEC_ADAPTIVE_ARMS}  # committed, steps, accepted, verified
        for name, prompt in prompts:
            truth = plain_greedy(self.handle, prompt, SPEC_DECODE - 1)
            want_scores = torch.cat([truth.prefill.unsqueeze(0), truth.scores])
            for arm, flags in SPEC_ADAPTIVE_ARMS:
                case = f"A4 {name} {arm}"
                with self.subTest(case=case):
                    g, steps, accepted, verified = generate_flags(self.handle, prompt, SPEC_DECODE, flags)
                    self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} {case}: finish {g.finish}")
                    self.assertTrue(g.committed >= 1, f"{ROW} {case}: nothing committed")
                    self.assertEqual(g.tokens.tolist(), truth.tokens[:g.committed], f"{ROW} {case}: tokens")
                    oracle.check_bit_exact(self, ROW, f"{case} scores", g.scores, want_scores[:g.committed])
                    self.assertTrue(accepted <= verified <= 3 * steps, f"{ROW} {case}: {accepted}/{verified}/{steps}")
                    if arm == "fixed":
                        self.assertTrue(verified >= 3 * (steps - 3), f"{ROW} {case}: fixed verified {verified}")
                    for i, v in enumerate((g.committed, steps, accepted, verified)):
                        totals[arm][i] += v
                    oracle.report_row(ROW, case, committed=g.committed, verify_steps=steps, accepted=accepted,
                                      verified=verified, accept_length=round(g.committed / max(steps, 1), 4),
                                      drafts_per_round=round(verified / max(steps, 1), 4))
        for arm, (committed, steps, accepted, verified) in totals.items():
            oracle.report_row(ROW, f"A4 all {arm}", committed=committed, verify_steps=steps, accepted=accepted,
                              verified=verified, accept_length=round(committed / max(steps, 1), 4),
                              drafts_per_round=round(verified / max(steps, 1), 4))

    def test_spec_long_lookup_greedy_bit_identical(self):
        """TreeSpec TS-02 (module docstring): greedy requests whose flags let the policy verify up to 8 lookup drafts in
        one chain commit exactly the plain greedy tokens and scores, and the long chains run. Chains (O123, bit 7):
        with up to 12 drafts per chain they still commit plain's tokens and scores, and on some prompt verify a
        different number of rounds or drafts than the 8-draft cap (so chains past 8 run; equal counts everywhere = bit 7
        ignored). Fewer rounds is not guaranteed: a long chain the target rejects early verifies more drafts."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 5)
        deep = {arm: 0 for arm, _ in SPEC_LONG_ARMS}
        changed = 0
        for prefix, block, repeats in SPEC_LOOKUP_SHAPES:
            unit = synthetic_ids(gen, self.vocab, block)
            prompt = synthetic_ids(gen, self.vocab, prefix) + unit * repeats
            truth = plain_greedy(self.handle, prompt, SPEC_DECODE - 1)
            want_scores = torch.cat([truth.prefill.unsqueeze(0), truth.scores])
            rounds = {}
            for arm, flags in SPEC_LONG_ARMS:
                case = f"TS-02 prompt {len(prompt)} {arm}"
                with self.subTest(case=case):
                    g, steps, accepted, verified = generate_flags(self.handle, prompt, SPEC_DECODE, flags)
                    self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} {case}: finish {g.finish}")
                    self.assertTrue(g.committed >= 1, f"{ROW} {case}: nothing committed")
                    self.assertEqual(g.tokens.tolist(), truth.tokens[:g.committed], f"{ROW} {case}: tokens")
                    oracle.check_bit_exact(self, ROW, f"{case} scores", g.scores, want_scores[:g.committed])
                    cap = SPEC_CHAIN_MAX if flags & WORK_CHAIN_DRAFTS else SPEC_TAIL_MAX
                    self.assertTrue(accepted <= verified <= cap * steps, f"{ROW} {case}: {accepted}/{verified}/{steps}")
                    if verified > 3 * steps:
                        deep[arm] += 1
                    rounds[arm] = (steps, verified)
                    oracle.report_row(ROW, case, committed=g.committed, verify_steps=steps, accepted=accepted,
                                      verified=verified, accept_length=round(g.committed / max(steps, 1), 4),
                                      drafts_per_round=round(verified / max(steps, 1), 4))
            changed += rounds.get("mtp+lookup+chain") != rounds.get("mtp+lookup+long")
        for arm, n in deep.items():
            self.assertTrue(n > 0, f"{ROW} TS-02 {arm}: no request verified more than 3 drafts per round")
        self.assertTrue(changed > 0, f"{ROW} TS-02: bit 7 verified the same rounds and drafts as the 8-draft cap")

    def test_spec_adaptive_sampled_seeded_equals_fixed(self):
        """Review M1 (tasks/m29a-spec.md A6, T7 [REVISED: review-wave-fg M1]): a seeded sampled request
        (SPEC_SAMPLED_SEEDED, temperature 0.8) sent with the cere_work.flags the runtime gives a sampled ticket under
        `draft_policy = "adaptive"` (SPEC_SAMPLED_ADAPTIVE_RUNTIME_FLAGS) commits exactly the tokens and scores of the
        fixed policy (flags 0), which verifies 3 drafts in every round but the last 3. The executor's bit 4 on a
        sampled row is not bit-identical (a round's draft count moves the seeded draws of the positions after it):
        rows "M1 bit4" report, per prompt, whether flags 16 commit the fixed tokens and the first position that
        differs (-1: none). Token ids are compared in memory only."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 7)
        prefix, block, repeats = SPEC_LOOKUP_SHAPES[0]
        prompts = [(pid, ids) for pid, ids, _ in parity_prompts()]
        prompts.append(("block", synthetic_ids(gen, self.vocab, prefix) + synthetic_ids(gen, self.vocab, block) * repeats))

        def run(prompt, flags):
            return generate_sampled_flags(self.handle, prompt, SPEC_DECODE, flags, SPEC_SAMPLED_SEEDED, self.vocab)

        def first_divergence(a: Generation, b: Generation) -> int:
            n = min(a.committed, b.committed)
            differs = (a.tokens[:n] != b.tokens[:n]).nonzero()
            if differs.numel():
                return int(differs[0, 0])
            return -1 if a.committed == b.committed else n

        diverged = 0
        for name, prompt in prompts:
            fixed, steps, accepted, verified = run(prompt, 0)
            case = f"M1 {name}"
            self.assertIn(fixed.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} {case}: finish {fixed.finish}")
            self.assertTrue(fixed.committed >= 1, f"{ROW} {case}: nothing committed")
            self.assertTrue(accepted <= verified <= 3 * steps, f"{ROW} {case}: {accepted}/{verified}/{steps}")
            self.assertTrue(verified >= 3 * (steps - 3), f"{ROW} {case}: fixed verified {verified}")
            bit4 = run(prompt, WORK_ADAPTIVE_DRAFTS)[0]
            at = first_divergence(bit4, fixed)
            diverged += at != -1
            oracle.report_row(ROW, f"M1 bit4 {name}", equal=int(at == -1), first_divergence=at,
                              committed_fixed=fixed.committed, committed_bit4=bit4.committed)
            with self.subTest(case=case):
                got = run(prompt, SPEC_SAMPLED_ADAPTIVE_RUNTIME_FLAGS)[0]
                at = first_divergence(got, fixed)
                self.assertEqual(at, -1, f"{ROW} {case}: flags {SPEC_SAMPLED_ADAPTIVE_RUNTIME_FLAGS} committed "
                                         f"{got.committed} vs fixed {fixed.committed}, first divergence at {at}")
                oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores, fixed.scores)
        oracle.report_row(ROW, "M1 bit4 all", requests=len(prompts), diverged=diverged)

    def chain_run(self, label: str, prompt_len: int, steps: int, seed: int, rows: int = 1) -> int:
        """Deep chains (DraftAccept): `rows` sessions on `prompt_len`-token prompts decode `steps` tokens through
        cere_ktest_session_speculate_chain rounds (all live rows in one call) with SPEC_CHAIN_LENS drafts = the true
        next tokens, the one at SPEC_CHAIN_WRONG made wrong; committed counts, scores and the state digest after every
        round equal the plain run's, and chains longer than 4 start at every position mod 4. Returns the verified
        chains of more than one row (those verify_attention = batched runs as attention_chain)."""
        gen = torch.Generator().manual_seed(seed)
        prompts = [synthetic_ids(gen, self.vocab, prompt_len + r) for r in range(rows)]
        truth = [plain_greedy(self.handle, ids, steps, digests=True) for ids in prompts]
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        sessions = []
        for r in range(rows):
            session, prefill = self.drafting_session(stack, prompts[r])
            oracle.check_bit_exact(self, ROW, f"{label} session {r} prefill", prefill, truth[r].prefill)
            sessions.append(session)
        at = [0] * rows
        round_, offsets, deep, multi = 0, set(), 0, 0
        while any(i < steps for i in at):
            live = [r for r in range(rows) if at[r] < steps]
            tokens, drafts, commits, expect = [], [], [], []
            for k, r in enumerate(live):
                n = SPEC_CHAIN_LENS[(round_ + k) % len(SPEC_CHAIN_LENS)]
                commit = min(n + 1, steps - at[r])
                d = truth[r].tokens[at[r] + 1:at[r] + commit]
                w = SPEC_CHAIN_WRONG[(round_ + 3 * k) % len(SPEC_CHAIN_WRONG)]
                if w is not None and w < len(d):
                    d[w] = wrong(d[w], self.vocab)
                    expect.append(w + 1)
                else:
                    expect.append(len(d) + 1)
                if len(d) > 4:
                    offsets.add((len(prompts[r]) + at[r]) % 4)
                    deep += 1
                tokens.append(truth[r].tokens[at[r]])
                drafts.append(d)
                multi += 1 if d else 0
                commits.append(commit)
            got = session_speculate_chain([sessions[r] for r in live], tokens, drafts, commits)
            for p, r in enumerate(live):
                case = f"{label} session {r} round {round_} (position {len(prompts[r]) + at[r]}, n {len(drafts[p])})"
                c = got.committed[p]
                self.assertEqual((c, got.accepted[p]), (expect[p], expect[p] - 1), f"{ROW} {case}: committed")
                oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[p, :c],
                                       truth[r].scores[at[r]:at[r] + c])
                at[r] += c
                self.assertEqual(state_digest(sessions[r]), truth[r].digests[len(prompts[r]) + at[r]],
                                 f"{ROW} {case}: state digest after commit")
            round_ += 1
        self.assertEqual(offsets, {0, 1, 2, 3}, f"{ROW} {label}: deep chain start offsets mod 4")
        oracle.report_row(ROW, label, rows=rows, rounds=round_, deep_chains=deep, committed=rows * steps)
        return multi

    def test_spec_deep_chain_graph(self):
        """DraftAccept S1 / Chains (O123): chains of 5..12 drafts below 2,048 (the verify graph on a spec_graph model,
        dense K6) commit plain greedy's tokens, scores and state; R 1 and 2 (two deep chains share the undo pool, split
        into two calls when they do not fit it together)."""
        self.chain_run("DC dense R=1", SPEC_CHAIN_PROMPT, SPEC_CHAIN_STEPS, SPEC_SEED + 21)
        self.chain_run("DC dense R=2", SPEC_CHAIN_PROMPT, SPEC_CHAIN_STEPS, SPEC_SEED + 22, rows=2)

    def test_spec_deep_chain_qsa(self):
        """DraftAccept S1 / Chains (O123): chains of 5..12 drafts past 2,048 (the K9 QSA path: a chain completes up to
        three compressed key groups, restored from as many snapshots) commit plain greedy's tokens, scores and state."""
        self.chain_run("DC qsa R=1", SPEC_QSA_PROMPT, SPEC_QSA_END - SPEC_QSA_PROMPT + 32, SPEC_SEED + 23)

    def test_verify_attention_batched_sparse(self):
        """LK-01 (tasks/long-ctx-spec.md, verify_attention = batched past 2,048): deep chains (3..12 drafts, rejections
        at every row in turn) of R 1 and 2 sessions whose every row attends past the dense limit (prompts of
        SPEC_SPARSE_PROMPT ids: more than 512 visible blocks, so the top-512 radix select runs) commit plain greedy's
        tokens and scores and leave plain's state digest with K9's selection and sparse attention batched over each
        chain's rows: the attention_chain counter moves by one per full-attention layer and verified chain of more
        than one row. Fails on a library whose sparse chains stay per row (the counter does not move)."""
        self.addCleanup(verify_attention, self.handle, verify_attention(self.handle, -1)[0])
        full_layers = sum(1 for kind in self.tc["layer_types"] if kind == "full_attention")
        verify_attention(self.handle, 1)
        for rows in (1, 2):
            with self.subTest(rows=rows):
                start = verify_attention(self.handle, -1)[1]
                chains = self.chain_run(f"LK-01 sparse R={rows}", SPEC_SPARSE_PROMPT, SPEC_SPARSE_STEPS,
                                        SPEC_SEED + 30 + rows, rows=rows)
                self.assertTrue(chains > 0, f"{ROW} LK-01 R={rows}: no chain of more than one row")
                self.assertEqual(verify_attention(self.handle, -1)[1] - start, full_layers * chains,
                                 f"{ROW} LK-01 R={rows}: sparse verify chains not batched")
                oracle.report_row(ROW, f"LK-01 sparse batched R={rows}", chains=chains)

    def test_spec_qsa_rollback_crosses_2048(self):
        """K7 (M29S2-20): rejections inside chains that straddle a QSA ring group, past 2,048 positions, leave the
        state digest and scores of the plain run."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 2)
        prompt = synthetic_ids(gen, self.vocab, SPEC_QSA_PROMPT)
        steps = SPEC_QSA_END - SPEC_QSA_PROMPT
        truth = plain_greedy(self.handle, prompt, steps, digests=True)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        session, prefill = self.drafting_session(stack, prompt)
        oracle.check_bit_exact(self, ROW, "K7 prefill", prefill, truth.prefill)
        at = 0
        straddling = 0
        while at < steps:
            position = SPEC_QSA_PROMPT + at
            commit = min(4, steps - at)
            d = truth.tokens[at + 1:at + commit]
            if position % 4 in (0, 1):
                # Placement: commit exactly enough right drafts that the next chain starts at 2 (mod 4).
                d = d[:(2 - position) % 4 - 1]
                expect = len(d) + 1
                commit = min(commit, expect)
            else:
                w = SPEC_WRONG_ROWS[straddling % len(SPEC_WRONG_ROWS)]
                straddling += 1
                if w is not None and w < len(d):
                    d[w] = wrong(d[w], self.vocab)
                    expect = w + 1
                else:
                    expect = len(d) + 1
            got = session_speculate([session], [truth.tokens[at]], [d], [commit])
            case = f"K7 position {position}"
            self.assertEqual(got.committed[0], expect, f"{ROW} {case}: committed")
            oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[0, :expect],
                                   truth.scores[at:at + expect])
            at += expect
            self.assertEqual(state_digest(session), truth.digests[SPEC_QSA_PROMPT + at],
                             f"{ROW} {case}: state digest after commit")
        self.assertTrue(straddling >= 2 * len(SPEC_WRONG_ROWS), f"{ROW} K7: only {straddling} straddling chains")
        oracle.report_row(ROW, "K7", straddling=straddling, end=SPEC_QSA_PROMPT + at)

    def probe_drafts(self, fed: list[int], token: int) -> list[int]:
        """Three drafts: the greedy continuation after `fed` + [token] on a throwaway plain session (PREFILL of the
        whole sequence, then two DECODE steps); plausible drafts only, the checks never depend on them."""
        with opened_session(self.handle) as probe:
            drafts = [int(session_feed(probe, fed + [token])[0, 0])]
            for _ in range(2):
                drafts.append(int(session_feed(probe, drafts[-1:])[0, 0]))
        return drafts

    def test_spec_sampled_d6_exact(self):
        """Phase S (D6, M29S2-13/-14): sampled chains commit the D6 rule's ids on their own rows' distributions, which
        are plain sampled decode's; scores, logprobs and out_counts are plain's for the committed tokens."""
        self.sampled_d6_run("D6", SPEC_SEED + 4, masked=False)

    def test_spec_masked_sampled_d6_exact(self):
        """GrammarSpec GS-02 (B2's device half): test_spec_sampled_d6_exact with every verify row under the
        `position_mask` of the position it samples and the drafts ended at the first one that mask bans (a grammar's
        refusal: chains of 0..3 drafts); the plain sessions sample under the same masks. The committed ids follow D6
        on the masked rows, every one allowed, and the processed rows, scores and logprobs are plain masked decode's."""
        self.sampled_d6_run("GS-02 D6", SPEC_SEED + 9, masked=True)

    def sampled_d6_run(self, label: str, seed: int, masked: bool) -> None:
        """The D6 run of the SPEC_SAMPLED sessions; `masked`: under GS-02's masks (test_spec_masked_sampled_d6_exact)."""
        self.assertEqual(self.vocab, oracle.D6_VOCAB, f"{ROW} {label}: the D6 reference's vocab")
        gen = torch.Generator().manual_seed(seed)
        rows = len(SPEC_SAMPLED)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS[:rows]]
        stats = {"accepted": 0, "rejected": 0, "bonus": 0, "committed": 0}
        refused = 0  # drafts the masks banned (masked: the chain ends before them)
        # The speculating sessions plus the draft probe hold 3 of the model's SPEC_SESSIONS slots; the plain sampled
        # sessions replay the committed chains after they close (rounds keep the rows they compare).
        rounds = [[] for _ in range(rows)]  # (case, position, chain, drafts, want, processed rows, scores, logprobs)
        with contextlib.ExitStack() as stack:
            spec, fed, token, counts, prefills = [], [], [], [], []
            for ids in prompts:
                session, prefill = self.drafting_session(stack, ids)
                spec.append(session)
                prefills.append(prefill)
                fed.append(list(ids))
                token.append(int(prefill[0]))
                counts.append(torch.zeros(self.vocab, dtype=torch.int64))
            for round_ in range(SPEC_SAMPLED_ROUNDS):
                drafts, masks = [], []
                for r in range(rows):
                    d = self.probe_drafts(fed[r], token[r])
                    w = SPEC_WRONG_ROWS[(round_ + r) % len(SPEC_WRONG_ROWS)]
                    if w is not None:
                        d[w] = wrong(d[w], self.vocab)
                    if masked:
                        # Verify row j feeds the chain's token j at len(fed) + j and samples the next position.
                        masks.append([position_mask(len(fed[r]) + j + 1, self.vocab) for j in range(4)])
                        n = next((j for j, x in enumerate(d) if not mask_allows(masks[r][j], x)), len(d))
                        refused += len(d) - n
                        d = d[:n]
                    drafts.append(d)
                got = session_speculate(spec, token, drafts, [4] * rows, list(SPEC_SAMPLED), self.vocab,
                                        masks if masked else None)
                spans = [len(d) + 1 for d in drafts]  # the stage rows of each chain, in call order
                processed, probs, keep = spec_stages(spec[0], sum(spans), self.vocab)
                at = 0
                for r, s in enumerate(SPEC_SAMPLED):
                    p = len(fed[r])
                    case = f"{label} session {r} round {round_} (position {p})"
                    rows_r = slice(at, at + spans[r])
                    want = d6_chain(probs[rows_r], keep[rows_r], drafts[r], s[6], p)
                    c = got.committed[r]
                    self.assertEqual(got.scores[r, :c, 0].tolist(), want, f"{ROW} {case}: committed ids")
                    self.assertEqual((c, got.accepted[r]), (len(want), len(want) - 1), f"{ROW} {case}: counts")
                    if masked:
                        for j, x in enumerate(want):
                            self.assertTrue(mask_allows(masks[r][j], x), f"{ROW} {case}: row {j} commits a banned id")
                    counts[r] += torch.bincount(torch.tensor(want), minlength=self.vocab)
                    self.assertTrue(torch.equal(session_counts(spec[r], self.vocab), counts[r]),
                                    f"{ROW} {case}: out_counts gain exactly the committed ids")
                    chain = [token[r]] + drafts[r][:c - 1]
                    rounds[r].append((case, p, chain, len(drafts[r]), want, processed[at:at + c].clone(),
                                      got.scores[r, :c].clone(), got.logprobs[r, :c].clone()))
                    stats["accepted"] += c - 1
                    stats["rejected"] += int(c <= len(drafts[r]))
                    stats["bonus"] += int(c == len(drafts[r]) + 1)
                    stats["committed"] += c
                    at += spans[r]
                    fed[r] += chain
                    token[r] = want[-1]
        for r, s in enumerate(SPEC_SAMPLED):
            with opened_session(self.handle) as plain:
                oracle.check_bit_exact(self, ROW, f"{label} session {r} prefill", session_feed(plain, prompts[r])[0],
                                       prefills[r])
                for case, p, chain, n, want, processed, scores, logprobs in rounds[r]:
                    for j, x in enumerate(chain):
                        mask = position_mask(p + j + 1, self.vocab) if masked else None
                        step = session_step_sampled(plain, x, s, self.vocab, mask)
                        oracle.check_bit_exact(self, ROW, f"{case} row {j} processed", processed[j], step.processed)
                        oracle.check_bit_exact(self, ROW, f"{case} row {j} score", scores[j],
                                               committed_score(step.processed, want[j]))
                        if s[7]:
                            oracle.check_bit_exact(self, ROW, f"{case} row {j} logprobs", logprobs[j],
                                                   expected_logprobs(step.processed, s[0], want[j], CR4_TOP_LOGPROBS))
                        if j == n:
                            # The bonus row draws as plain sampling does at that position (channels (0, 1)).
                            oracle.check_bit_exact(self, ROW, f"{case} bonus vs plain draw", scores[j], step.score)
        self.assertTrue(stats["accepted"] > 0, f"{ROW} {label}: no draft accepted")
        self.assertTrue(stats["rejected"] > 0, f"{ROW} {label}: no draft rejected")
        if masked:
            self.assertTrue(refused > 0, f"{ROW} {label}: the masks never refuse a draft")
            stats["refused"] = refused
        oracle.report_row(ROW, f"{label} sampled", rounds=SPEC_SAMPLED_ROUNDS, sessions=rows, **stats)

    def test_spec_masked_t0_bit_identical(self):
        """GrammarSpec GS-02 (B1's device half): a constrained greedy item (SPEC_MASKED_T0) speculating with plain's
        tokens as drafts, one made wrong at a rotating row and the chain ended at the first draft its mask bans, every
        verify row under the `position_mask` of the position it samples, commits exactly the plain masked run's tokens
        and scores, with plain's state digest after every round; the masks change the run (it differs from the
        unmasked one) and drafts are accepted."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 7)
        prompt = synthetic_ids(gen, self.vocab, SPEC_UNDO_PROMPT)
        n0 = len(prompt)
        with opened_session(self.handle) as session:
            prefill = session_feed(session, prompt)[0].clone()
            tokens = [int(prefill[0])]
            steps, digests = [], {}
            for k in range(SPEC_MASKED_STEPS):
                mask = position_mask(n0 + k + 1, self.vocab)
                step = session_step_sampled(session, tokens[-1], SPEC_MASKED_T0, self.vocab, mask)
                steps.append(step)
                tokens.append(int(step.score[0]))
                digests[k + 1] = state_digest(session)
        with opened_session(self.handle) as session:
            free = [int(session_feed(session, prompt)[0, 0])]
            for _ in range(SPEC_MASKED_STEPS):
                free.append(int(session_step_sampled(session, free[-1], SPEC_MASKED_T0, self.vocab).score[0]))
        self.assertNotEqual(free, tokens, f"{ROW} GS-02 T0: the masks never change the run")
        with contextlib.ExitStack() as stack:
            spec, spec_prefill = self.drafting_session(stack, prompt)
            oracle.check_bit_exact(self, ROW, "GS-02 T0 prefill", spec_prefill, prefill)
            at = round_ = accepted = refused = 0
            while at < SPEC_MASKED_STEPS:
                commit = min(4, SPEC_MASKED_STEPS - at)
                masks = [position_mask(n0 + at + j + 1, self.vocab) for j in range(4)]
                d = tokens[at + 1:at + commit]
                w = SPEC_WRONG_ROWS[round_ % len(SPEC_WRONG_ROWS)]
                if w is not None and w < len(d):
                    d[w] = wrong(d[w], self.vocab)
                n = next((j for j, x in enumerate(d) if not mask_allows(masks[j], x)), len(d))
                refused += len(d) - n
                d = d[:n]
                expect = next((j + 1 for j, x in enumerate(d) if x != tokens[at + j + 1]), len(d) + 1)
                got = session_speculate([spec], [tokens[at]], [d], [commit], [SPEC_MASKED_T0], self.vocab, [masks])
                case = f"GS-02 T0 round {round_} (position {n0 + at})"
                self.assertEqual((got.committed[0], got.accepted[0]), (expect, expect - 1), f"{ROW} {case}: committed")
                oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[0, :expect],
                                       torch.stack([s.score for s in steps[at:at + expect]]))
                at += expect
                accepted += expect - 1
                self.assertEqual(state_digest(spec), digests[at], f"{ROW} {case}: state digest")
                round_ += 1
        self.assertTrue(accepted > 0, f"{ROW} GS-02 T0: no draft accepted")
        self.assertTrue(refused > 0, f"{ROW} GS-02 T0: the masks never refuse a draft")
        oracle.report_row(ROW, "GS-02 T0 masked", rounds=round_, committed=at, accepted=accepted, refused=refused)

    def test_spec_masked_refuses_mtp_drafts(self):
        """GrammarSpec GS-03 (CudaDevice::speculate_group): a masked row with drafts verifies the caller's
        grammar-checked drafts only; one drafting from its pending MTP proposals (drafts null) is refused (call or row
        status CERE_ERR_ARGUMENT) and its session keeps its state digest; the same row with no drafts runs."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 8)
        prompt = synthetic_ids(gen, self.vocab, SPEC_UNDO_PROMPT)
        masks = [position_mask(len(prompt) + j + 1, self.vocab) for j in range(4)]
        with contextlib.ExitStack() as stack:
            spec, prefill = self.drafting_session(stack, prompt)
            before = state_digest(spec)
            status = torch.full((1,), 1, dtype=torch.int32)
            results = torch.zeros((1, 2), dtype=torch.int32)
            scores = torch.full((1, 4, 4), -1, dtype=torch.int32)
            logprobs = torch.full((1, 4, TOKEN_LOGPROBS_BYTES // 4), -1, dtype=torch.int32)
            rc = oracle.call_status("cere_ktest_session_speculate_masked", torch.tensor([spec], dtype=torch.int64), 1,
                                    torch.tensor([int(prefill[0])], dtype=torch.int32), None,
                                    torch.tensor([3], dtype=torch.int32), torch.tensor([4], dtype=torch.int32),
                                    sampling_bytes([SPEC_MASKED_T0], self.vocab), torch.tensor([-1], dtype=torch.int32),
                                    torch.stack([torch.stack(masks)]).contiguous(), status, results, scores, logprobs)
            self.assertTrue(rc == CERE_ERR_ARGUMENT or (rc == 0 and status.tolist() == [CERE_ERR_ARGUMENT]),
                            f"{ROW} GS-03: MTP drafts under masks gave status {rc}, row status {status.tolist()}")
            self.assertEqual(state_digest(spec), before, f"{ROW} GS-03: the refused row changed its session")
            got = session_speculate([spec], [int(prefill[0])], [[]], [4], [SPEC_MASKED_T0], self.vocab, [masks])
            self.assertEqual((got.committed[0], got.accepted[0]), (1, 0), f"{ROW} GS-03: a 0-draft masked round")
            self.assertTrue(mask_allows(masks[0], int(got.scores[0, 0, 0])), f"{ROW} GS-03: a banned id committed")

    def test_spec_sampled_top1_greedy(self):
        """D6 with p(d) = 1: top_k 1 sampled rows (beside greedy rows in the same calls) commit exactly as greedy."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 1)
        prompts = [synthetic_ids(gen, self.vocab, n) for n in SPEC_PROMPT_LENS]
        truth = [plain_greedy(self.handle, ids, SPEC_DECODE) for ids in prompts]
        specs = [SPEC_TOP1 if r % 2 == 0 else None for r in range(SPEC_SESSIONS)]
        with contextlib.ExitStack() as stack:
            self.speculate_sessions(stack, prompts, truth, specs, "D6 top1")

    def test_spec_sampled_t0_digest(self):
        """M29S2-14: a rejecting sampled (temperature 0, penalties) round leaves plain sampled decode's state digest
        (out_counts included), scores and logprobs."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 5)
        prompt = synthetic_ids(gen, self.vocab, SPEC_UNDO_PROMPT)
        with opened_session(self.handle) as session:
            prefill = session_feed(session, prompt)[0].clone()
            tokens = [int(prefill[0])]
            steps, digests = [], {}
            for _ in range(SPEC_T0_STEPS):
                step = session_step_sampled(session, tokens[-1], SPEC_T0, self.vocab)
                steps.append(step)
                tokens.append(int(step.score[0]))
                digests[len(prompt) + len(steps)] = state_digest(session)
        with contextlib.ExitStack() as stack:
            spec, spec_prefill = self.drafting_session(stack, prompt)
            oracle.check_bit_exact(self, ROW, "T0 prefill", spec_prefill, prefill)
            at = 0
            round_ = 0
            rejecting = 0
            while at < SPEC_T0_STEPS:
                commit = min(4, SPEC_T0_STEPS - at)
                d = tokens[at + 1:at + commit]
                w = SPEC_WRONG_ROWS[round_ % len(SPEC_WRONG_ROWS)]
                expect = len(d) + 1
                if w is not None and w < len(d):
                    d[w] = wrong(d[w], self.vocab)
                    expect = w + 1
                    rejecting += 1
                got = session_speculate([spec], [tokens[at]], [d], [commit], [SPEC_T0], self.vocab)
                case = f"T0 round {round_} (position {len(prompt) + at})"
                self.assertEqual((got.committed[0], got.accepted[0]), (expect, expect - 1), f"{ROW} {case}: committed")
                oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[0, :expect],
                                       torch.stack([s.score for s in steps[at:at + expect]]))
                oracle.check_bit_exact(self, ROW, f"{case} logprobs", got.logprobs[0, :expect],
                                       torch.stack([s.logprobs for s in steps[at:at + expect]]))
                at += expect
                self.assertEqual(state_digest(spec), digests[len(prompt) + at], f"{ROW} {case}: state digest")
                round_ += 1
        self.assertTrue(rejecting > 0, f"{ROW} T0: no rejecting round")
        oracle.report_row(ROW, "T0 sampled digest", rounds=round_, rejecting=rejecting, committed=at)

    def test_forced_verify_trace(self):
        """K8 (I8): the verify rows' per-name taps and scores equal the forced decode steps' at the same positions."""
        gen = torch.Generator().manual_seed(SPEC_SEED + 3)
        ids = synthetic_ids(gen, self.vocab, SPEC_TRACE_PROMPT + SPEC_TRACE_COMPLETION)
        decode = forced_decode_trace(self.handle, ids, SPEC_TRACE_PROMPT, hidden=self.hidden, layers=self.layers)
        verify = forced_decode_trace(self.handle, ids, SPEC_TRACE_PROMPT, hidden=self.hidden, layers=self.layers,
                                     chain=3)
        oracle.check_bit_exact(self, ROW, "K8 verify trace scores", verify.scores, decode.scores)
        self.assertEqual(verify.counts, decode.counts, f"{ROW} K8: trace counts")
        for step, (v, d) in enumerate(zip(verify.steps, decode.steps, strict=True)):
            self.assertEqual(list(v), list(d), f"{ROW} K8 step {step + 1}: names")
            oracle.check_bit_exact(self, ROW, f"K8 step {step + 1} taps", torch.cat(list(v.values()), dim=1),
                                   torch.cat(list(d.values()), dim=1))
        del decode, verify
        torch.cuda.empty_cache()


class K8SpeculationSmallM(K8Speculation):
    """K8Speculation on a model created with levers.k2_gemm small_m (module docstring K8SpeculationSmallM)."""

    LEVERS = {"k2_gemm": CERE_LEVER_K2_SMALL_M}


class K8DeepChainV19(unittest.TestCase):
    """DraftAccept S1: the deep-chain tests of K8Speculation on a model with the -v19 device levers (k2_gemm small_m,
    decode_step spec_graph, verify_attention batched): chains below 2,048 run the verify graph (its key holds the undo
    range), chains past it the eager K9 path."""

    LEVERS = {"k2_gemm": CERE_LEVER_K2_SMALL_M, "decode_step": 2, "verify_attention": 1}
    setUpClass = classmethod(K8Speculation.setUpClass.__func__)
    draft_token_map = classmethod(K8Speculation.draft_token_map.__func__)
    drafting_session = K8Speculation.drafting_session
    chain_run = K8Speculation.chain_run
    test_spec_deep_chain_graph = K8Speculation.test_spec_deep_chain_graph
    test_spec_deep_chain_qsa = K8Speculation.test_spec_deep_chain_qsa



if __name__ == "__main__":
    unittest.main()
