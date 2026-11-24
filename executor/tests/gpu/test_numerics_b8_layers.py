"""B8 numerics (tasks/m22-spec.md, M2.2b; X8 -> X9): the native forward pass against SGLang's tensor dump and teacher-forced
top-2, layer by layer, on the real A4 model, GPU0, container only.

Run only by `CERE_LIVE=1 python3 scripts/live/native_dev.py numerics [--prompts ids]` step 4: the dev container with
the build at `/out`, the A4 model at `/model` and the reference dump at `/refdump` (all read-only), running
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_numerics_b8_layers.py' -v` with the frame below on
stdin (the exact file: `test_numerics_prefill.py` reads its own frame and runs only under `numerics --suite prefill`).
The model is created, uploaded and sealed exactly as in `test_model_k8_forward.py` (ABI v5 `cere_model_desc` from
`/model`, families placement, concurrent puts, seal).

Reference dump (`/refdump`, read-only): `cere-ref-g0` ran with `--debug-tensor-dump-output-folder /refdump` and no
`--debug-tensor-dump-layers`, no speculation (so no `draft`/`target` subdirectory), which disables CUDA graphs and
server warmup (`$S181/sglang/srt/server_args.py:8576-8582`). `TensorDumper`
(`$S181/sglang/srt/debug_utils/tensor_dump_forward_hook.py`) writes one directory `TP0_PP0_Rank0_pid<PID>/` holding
`Pass<NNNNN>.pt` per forward pass (`torch.save` of a dict name -> CPU tensor, read here with `torch.load(mmap=True)`).
Keys are module attribute paths of the hooked leaves (modules without children) plus the top-level `model`, whose
hook first writes the pass file and then records its own output, so the `model` output of pass k is in pass k+1.
`Qwen4ExpForConditionalGeneration` calls its `model` with keyword arguments only (`$S181/sglang/srt/managers/
mm_utils.py:130-136`), so no `model.forward_batch_info.*` keys exist: a pass is identified by its
`model.embed_tokens` output, which must equal the checkpoint's embedding rows of the expected token IDs bit for bit.
The full-attention decoder layer owns its projection directly (`self.o_proj`, `$S181/sglang/srt/models/
qwen3_5.py:974`; `self_attn` exists only in weight names), so the B8 name `model.layers.N.self_attn.o_proj` is dumped
as `model.layers.N.o_proj`; the lookup accepts either key. Any B8 name absent from a dump pass or from the native
trace fails the test naming it.

Stdin frame (`cere-b8-v2`; the runner's in-memory IDs and top-2, never written to disk): one UTF-8 JSON document read
to EOF:

    {"frame": "cere-b8-v2",
     "dump_prompts": ["p01", "p13"],
     "prompts": [{"id": "p01",
                  "prompt_ids": [int, ...],
                  "completion_ids": [int, ...],
                  "top2": [[[int, float], [int, float]], ...]},
                 ...]}

* `dump_prompts`: the runner's `--prompts` ids (default p01,p13; any parity ids), non-empty and distinct: the prompts
  whose dump passes are compared layer by layer.
* `prompts`: all 24 ids of `bench/prompts/parity_v1.jsonl`, each once, the `dump_prompts` first in that order (the
  order the reference served them, so their passes are the first dump files), the others after them in any order.
* `prompt_ids`: the chat completion's prompt token IDs (`return_prompt_token_ids`); `completion_ids`: its greedy
  completion (`temperature` 0, `max_tokens` from the file, `return_token_ids`), at least one token.
* `top2`: the teacher-forced `/generate` on input_ids = prompt_ids + completion_ids (T tokens) with `return_logprob`,
  `logprob_start_len` 0, `top_logprobs_num` 2: exactly T - 1 entries, entry k = `meta_info.input_top_logprobs[k + 1]`
  rewritten best-first as [[token_id, logprob], [token_id, logprob]], i.e. SGLang's top-2 of the distribution computed
  at position k (predicting input token k + 1).
* The reference's radix cache is flushed (`/flush_cache`) before every request (as Q7 does), so the chat completion's
  first pass prefills exactly prompt_ids and the teacher-forced pass prefills all T tokens; the test checks both row
  counts in the dump.

Long frame (`cere-b8-long-v1`, tasks/m22c-spec.md C7/D8b, tasks/m25-spec.md D8b): as above plus `context` (the model is
created at it), `dump_layers`, `d8b_floor_pp` (the paired gate's floor: 1.0 on parity_long_v1, 0.5 on parity_long_v2)
and per prompt `logprob_start` (top2 covers positions logprob_start .. T - 2) and, for P > 2,048, `short` {tokens
2,048, logprob_start, top2} (A_short). Layer tensors compare the prompt's last 4,096-chunk rows of the dumped layers;
teacher-forced top-1 gates the paired d_i = A_long,i - A_short,i; decode steps are skipped.

Cases (B8/X9; diagnostic, never a G1 gate):
* `test_layer_tensors_vs_dump` (the `dump_prompts`): first the row `layer prompts` naming them (native_dev.py requires
  it to equal its `--prompts`), then per B8 name and layer, rel-L2(native trace of the T teacher-forced tokens,
  SGLang's teacher-forced dump pass) <= max(1e-2, 2c); the report lists the worst layer per name (pattern), each case
  carries its own c and limit. Every row is reported; the test then fails once if any exceeded (tasks/m22d-spec.md
  D9: no early stop).
* `test_decode_steps_vs_dump` (the `dump_prompts`; tasks/m22d-spec.md D3/D4, current build only: native_dev.py
  excludes it from the baseline run): locator: strictly between a prompt's prompt pass and its teacher-forced pass
  the dump holds exactly C = len(completion_ids) one-token passes; pass prompt + j embeds tokens[P + j - 1] bit for bit
  (j = 1 .. C - 1 the decode steps, j = C the trailing pass), else the test fails naming the prompt and the pass
  number; the `model` output of step j is read from the next pass. Measure: native `cere_ktest_forced_decode_trace`
  step j against decode pass j per B8 name, d = rel-L2 and limit = max(1e-2, 2c) with c = rel-L2(decode pass j,
  SGLang's teacher-forced row P + j - 1) (decode vs extend of the reference itself). Report rows only:
  `decode-step first-exceeding` per prompt (the first (step, layer, name) in forward order with d > limit: fields
  prompt, step, layer, name, d, limit, or none) and `decode-step per-name` per B8 name (exceeding_steps, steps,
  max_ratio = max d/limit). Fails only on a locator failure, a frame without decode steps or a ktest error.
* `test_decode_rows_vs_dump` (tasks/m26-spec.md K4; the `dump_prompts`, in pairs; current build only: the baseline
  libraries lack `cere_ktest_forward_decode_rows`, so native_dev.py must exclude it from the baseline run as it
  excludes `test_decode_steps_vs_dump`): the pair's prompts decoded as rows 0 and 2 of one batched decode step of
  R = ROWS_R rows (rows 1 and 3: the first two non-dump frame prompts with a decode step), every row forced with its
  completion as `cere_ktest_forced_decode_trace` forces it, through `cere_ktest_forward_decode_rows`: per decode step
  and B8 name the row's tensor, and every score row, equal the R = 1 `cere_ktest_forced_decode_trace` bit for bit
  (native vs native, the consequence of D6; not an SGLang measure, and not run on the long frame, whose model has
  one session). Report rows `decode-rows first-differing` per dump prompt (step, name, or none) and the step and
  tensor counts; the test fails once at the end if any differed.
* `test_teacher_forced_top1` (all 24 prompts): the native top-1 equals SGLang's at >= 99.9 % of the T - 1 positions
  summed over all prompts, and every disagreement has SGLang's top-2 log-prob margin <= 0.1 (native logits and
  SGLang log-probs share the softmax normalizer, so only SGLang's margin is used).
* `test_decode_forced_top1` (all 24 prompts; tasks/m22n-spec.md D8/D9, the serving path): native PREFILL of the prompt
  then forced single-token DECODE steps of the reference's greedy completion (`cere_ktest_forced_decode`); counts
  only, no agreement threshold: per completion position j, native decode top-1 vs the completion token (F of D9),
  native prefill-mode top-1 (the teacher-forced trace) vs it, SGLang's teacher-forced top-1 vs it, their overlaps and
  the first-token misses; `misses_over_tau` counts the decode misses where SGLang's teacher-forced top-2 margin is
  >= 0.25 (tasks/m22d-spec.md D8; ties and smaller margins are not counted). Checks: every forced position yields a
  score row, and the forced scores equal `cere_ktest_generate`'s greedy scores bit for bit up to the first position
  where native greedy leaves the completion (24/24 prompts).
* `test_teacher_forced_kl` (all 24 frame prompts, independent of `dump_prompts`; tasks/m214-spec.md D1, a measurement
  that changes no math and is never a G1 verdict): the prompt pass and the teacher-forced pass of every frame prompt
  are located as above (the pass after each holds its `model` rows; native_dev.py serves a closing request so the
  last served prompt's teacher-forced pass is followed too). Logits = `model` rows x the checkpoint's BF16
  `lm_head.weight`^T, accumulated in fp32 (the BF16 values widened exactly, vocab slices), log-softmax in fp64, in
  chunks of TF_KL_CHUNK rows; the same function for SGLang's and native's rows (this measures the trunk; the lm_head
  kernel's rounding is covered by the executor's top-2). Arms: (y) the yardstick, KL(teacher-forced pass || prompt
  pass) over rows [0, P), computed for every prompt before the first native trace; (a) KL(teacher-forced pass ||
  native `forward_trace`) over prompt rows [0, P) and, separately, completion rows [P, T). The calculus (same-top-1,
  KL mean/median/p99, flips and flips with SGLang margin < 0.5 nats, dNLL over prompt rows 0 .. P-2 with paired SE,
  ratio to the yardstick with YARD_FLOOR, the descriptive label) is `kl_measure.py` (covered on the CPU by
  `tests/test_numerics_kl_offline.py`). Rows, numbers only: `<pid> tf-kl` and `<pid> tf-kl last-prompt-row` (KL,
  both margins, top1_equal: the first-token row) per prompt, then `tf-kl aggregate`. Fails only on a locator failure
  (a missing pass or `model` row is never fabricated), a NaN or a ktest error. The long frame reports one `tf-kl` row
  with `applicable` false.

Calibration (tolerance rule, explained): c = rel-L2(SGLang op, high-precision reference) needs a reference that does not
exist for a 48-layer NVFP4 forward (no FP64 or FP32-dequantized whole-model run fits the comparison, and the kernel
rows' per-op c values live in other processes). The whole-block analogue used here is SGLang's own arithmetic spread
on identical inputs: the chat completion's prompt prefill and the teacher-forced prefill compute the same first P =
len(prompt_ids) positions of every named tensor (causal model, same tokens, same weights) under different batch shapes
(GEMM M, MoE token counts, attention lengths), so c = rel-L2(teacher-forced rows [0, P), prompt-pass rows) per name,
layer and prompt is how far SGLang moves that tensor by reordering its own arithmetic. Every c is computed and written
(`oracle.calibrate`) before the native model is traced; the limit is max(FLOOR_BLOCK, 2c), the block floor because
each tensor is a whole layer's output.

Token IDs and text stay in memory (stdin, process memory); report rows carry counts, rel-L2 values and margins only,
and no failure message contains an ID.

Native entry points (shared with K8, `test_model_k8_forward.py`; the contract with the kernel track; `oracle.call`
appends the stream; 0 = success, otherwise a
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
    int cere_ktest_generate(uint64_t model, const uint32_t *prompt /*host*/, uint64_t prompt_len,
                            uint64_t max_new_tokens, uint64_t prefill_chunk_tokens, uint64_t cancel_after,
                            uint32_t *out_tokens /*host [max_new_tokens]*/,
                            cere_token_score *out_scores /*host [max_new_tokens]*/,
                            uint64_t *out_result /*host [3]*/, cudaStream_t stream);
        One greedy request through the ABI v5 runtime (as `test_model_k8_forward.py`): one PREFILL `cere_work`
        with max_new_tokens, polled to its terminal completion. out_result = {committed_tokens, finish, device bytes
        in use}. Returns 0 for LENGTH, STOP_TOKEN and CANCELLED completions, the completion status otherwise.
    int cere_ktest_forced_decode(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                 uint64_t prompt_tokens,
                                 cere_token_score *scores /*host [token_count - prompt_tokens]*/,
                                 cudaStream_t stream);
        Decode-mode teacher forcing through the same runtime, in a fresh session at the executor's own prefill policy:
        PREFILL tokens[0, P) (P = prompt_tokens) with max_new_tokens 1, then for j = 1 .. C - 1 (C = token_count - P)
        the session's last committed (not yet fed) token is overwritten with tokens[P + j - 1] and one DECODE with
        max_new_tokens 1 is submitted and polled. scores[j] is the top-2 predicting tokens[P + j] after j forced M=1
        DECODE steps (the server's prefill-then-decode arithmetic). A STOP_TOKEN finish is accepted (the next
        overwrite replaces it). 0 < prompt_tokens < token_count <= max_context_tokens and every forced token
        tokens[P, T - 1) < vocab_size, else CERE_ERR_ARGUMENT before any request.
    int cere_ktest_forced_decode_trace(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                       uint64_t prompt_tokens,
                                       uint8_t *names /*host*/, uint64_t names_capacity,
                                       int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                       void *values /*device*/, uint64_t values_capacity_bytes,
                                       cere_token_score *scores /*host [token_count - prompt_tokens]*/,
                                       uint64_t *out_counts /*host [3]: entries, name_bytes, value_bytes*/,
                                       cudaStream_t stream);
        `cere_ktest_forced_decode`'s requests with the B8 taps of every DECODE step (tasks/m22d-spec.md D2): scores
        as `cere_ktest_forced_decode` writes them (scores[0] from the PREFILL, scores[j] from DECODE step j); entries
        step-major (step 1 first), each step the same names in the same order as `cere_ktest_forward_trace` with one
        row ({2, 1, cols}), names, shapes and values laid out as there. Refusals as `cere_ktest_forced_decode` plus
        any capacity too small: CERE_ERR_ARGUMENT, out_counts zero, no score row written.
    int cere_ktest_forward_decode_rows(uint64_t model, uint64_t sessions, const uint32_t *prompts /*host*/,
                                       const uint64_t *prompt_lens /*host [S]*/, const uint64_t *join /*host [S]*/,
                                       const uint64_t *leave /*host [S]*/, const uint32_t *forced /*host [S * T]*/,
                                       cere_token_score *scores /*host [S * (T + 1)]*/,
                                       uint8_t *names /*host*/, uint64_t names_capacity,
                                       int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                       void *values /*device*/, uint64_t values_capacity_bytes,
                                       uint64_t *out_counts /*host [3]*/, cudaStream_t stream);
        tasks/m26-spec.md I7, documented in `test_model_k8_forward.py`: S sessions, each prefilled alone, decoded in
        rounds of one batched decode step (row order = session order); session s's k-th decode feeds
        forced[s * T + k - 1] and writes scores[s * (T + 1) + k] (k = 0: the prefill); its tapped decode k is step
        s * T + k - 1 of a forced_decode_trace-layout trace; out_counts = one step's entries, name and value bytes.
"""
from __future__ import annotations

import ctypes
import functools
import importlib
import json
import math
import re
import statistics
import sys
import unittest
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open

import oracle

ROW = "B8"
SRC = Path(__file__).resolve().parents[3]
PARITY_FILE = SRC / "bench/prompts/parity_v1.jsonl"
FAMILIES_FILE = SRC / "configs/artifacts/qwen4_exp-nvfp4.families.json"
REFDUMP = Path("/refdump")
FRAME = "cere-b8-v2"
MAX_CONTEXT = 2048
# m22c design note C7, D8b: the long frame carries its context; the model is created at it (M22CS-10)
LONG_FRAME = "cere-b8-long-v1"
LONG_CHUNK = 4096  # the reference's --chunked-prefill-size: dump passes are chunks from position 0
SHORT_TOKENS = 2048  # A_short: the prompt truncated to one dense chunk (M22CS-04)
# D8b paired gate: mean(d) >= -max(floor, t * sd / sqrt(n)), the floor (pp) carried by the long frame
# one-sided t_{0.95, df} for df = 1 .. 30; above, the normal quantile
T95 = (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746,
       1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697)
AGREE_PER_MILLE = 999
MAX_DISAGREE_MARGIN = 0.1
EMBED_NAME = "model.language_model.embed_tokens.weight"
TAU = 0.25  # m22d design note D8: a decode miss counts in misses_over_tau iff SGLang's margin >= TAU
ROWS_R = 4  # m26 design note D3/K4: the short frame's model has 4 sessions, a batched decode step up to 4 rows
ROWS_DUMP = (0, 2)  # K4: the dump prompts' rows; the other rows run non-dump frame prompts

# m214 design note D1: the checkpoint's BF16 lm_head (excluded from quantization), logits in chunks of TF_KL_CHUNK
# rows. R1 bounds the oracle's GPU0 peak (executor + 2 GiB): at 128 rows the per-chunk transients were ~1 GiB on top of
# the resident lm_head (1,213 MiB) and R1 missed by 558 MiB, so the chunk is sized to TF_KL_DEVICE_BUDGET.
LM_HEAD_NAME = "lm_head.weight"
TF_KL_CHUNK = 32
TF_KL_VOCAB_SLICE = 16384  # lm_head rows widened to fp32 at a time (a 160 MiB transient)
TF_KL_DEVICE_BUDGET = 512 * 2**20  # bytes of KL transients beyond the resident lm_head


# -- ABI v5 (executor/include/cere_executor.h), mirrored with ctypes ---------------------------------------------

ABI_VERSION = 5
CERE_ERR_ARGUMENT = -1  # cere_executor.h
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


def tensor_placements() -> list:
    """(compiled pattern, placement) per rule of the artifact's family file, in rule order."""
    rules = json.loads(FAMILIES_FILE.read_text())["rules"]
    out = []
    for rule in rules:
        placement = PLACEMENT_DISCARD
        if rule["placement"] == "unused":
            # Out of the desc, as cered's model spec leaves it (a text-only desc carries no vision tensor, C7).
            placement = None
        elif rule["component"] == "target":
            placement = {"device": PLACEMENT_DEVICE, "host_pinned": PLACEMENT_HOST_PINNED}[rule["placement"]]
        out.append((re.compile(rule["pattern"]), placement))
    return out


class TableEntry(NamedTuple):
    name: str
    file: str
    placement: int
    nbytes: int


def a4_tensor_table():
    """Every checkpoint tensor in index (name) order but the family file's `unused` ones: table entries and the
    cere_tensor_desc array over them."""
    weight_map = json.loads((oracle.MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    rules = tensor_placements()
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


# The descs a `--baseline-build-dir` library may take (m22p design note L0): (cere_model_desc struct_size,
# abi_version stamped on the desc, its config and every tensor desc), newest first. ABI v5 (SelfTune ST2) puts
# `selection` (null here, the defaults) where v4's `levers` was and appends `family` and `calibrate_windows` (0, 0: v4's
# defaults); ABI v4 (PerfInt levers) appended `levers` to the v3.L desc of 120 bytes, and the v3.L desc only appends
# fields after the pre-L0 prefix of 72 (all zero here), so one buffer, restamped, is each older library's desc: an
# older library refuses a newer size with CERE_ERR_ARGUMENT before it reads anything else.
BASELINE_MODEL_DESCS = ((136, ABI_VERSION), (128, 4), (120, 3), (72, 3))


def create_model(max_context_tokens: int, max_sessions: int = 1) -> A4Model:
    """ALLOCATE (create), UPLOAD (concurrent puts), INSTANTIATE (seal) of the real A4 model on GPU0. A
    CERE_ERR_ARGUMENT (-1) is retried at each older desc of BASELINE_MODEL_DESCS (a baseline library may predate
    ABI v5, v4 or L0); the error names every attempt and its size."""
    tc = oracle.text_config()
    config_buf = a4_config(tc)
    entries, desc_buf, name_buf = a4_tensor_table()
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
    out = torch.zeros(1, dtype=torch.int64)
    free, _ = torch.cuda.mem_get_info(oracle.DEVICE)
    assert BASELINE_MODEL_DESCS[0] == (ctypes.sizeof(ModelDesc), ABI_VERSION)
    cfg = Qwen4ExpConfig.from_address(config_buf.data_ptr())
    tensors = (TensorDesc * len(entries)).from_address(desc_buf.data_ptr())
    attempts = []
    for size, abi in BASELINE_MODEL_DESCS:
        desc.struct_size = size
        desc.abi_version = cfg.abi_version = abi
        for t in tensors:
            t.abi_version = abi
        status = oracle.create_status(model_buf, free, out)
        attempts.append(f"desc size {size} abi {abi}: status {status}")
        if status != CERE_ERR_ARGUMENT:  # only the size refusal is retried
            break
    if status != 0:
        raise oracle.OracleError(f"cere_ktest_model_create failed ({'; '.join(attempts)})")
    handle = int(out[0])
    if handle == 0:
        raise oracle.OracleError("cere_ktest_model_create returned a null handle")
    return A4Model(handle, (config_buf, desc_buf, name_buf, stops, model_buf, entries))


def upload_and_seal(model: A4Model) -> None:
    oracle.upload_and_seal(model.handle, model.keep[-1], upload_file)


def release_model(handle: int) -> None:
    oracle.release_model(handle)


# -- runtime entry points ----------------------------------------------------------------------------------------


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
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 <= entries <= capacity and 0 <= name_bytes <= names.numel()
            and 0 <= value_bytes <= values.numel() * values.element_size()):
        raise oracle.OracleError(f"forward_trace counts {entries}, {name_bytes}, {value_bytes} exceed capacities")
    emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"forward_trace wrote {len(emitted) - 1} names for {entries} entries")
    out: dict = {}
    element = 0
    for i in range(entries):
        name = emitted[i].decode()
        ndim, r, c = (int(v) for v in shapes[i].tolist())
        if ndim != 2 or r < 0 or c < 0:
            raise oracle.OracleError(f"forward_trace entry {name}: shape record {(ndim, r, c)}")
        if name in out:
            raise oracle.OracleError(f"forward_trace emitted {name} twice")
        out[name] = values[element:element + r * c].view(r, c)
        element += r * c
    if element * values.element_size() != value_bytes:
        raise oracle.OracleError(f"forward_trace value bytes {value_bytes} != the entries' {element * 2}")
    return Trace(out, scores)


class Generation(NamedTuple):
    tokens: torch.Tensor  # int32 [committed] (uint32 IDs)
    scores: torch.Tensor  # int32 [committed, 4]: cere_token_score bytes {id, runner_up, logit, runner_up_logit}
    committed: int
    finish: int


def generate(handle: int, ids: list[int], max_new: int) -> Generation:
    """Native greedy decoding (`cere_ktest_generate`, the executor's own prefill policy, no cancel)."""
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((max_new,), -1, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    result = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_generate", handle, prompt, len(ids), max_new, 0, 0, tokens, scores, result)
    committed, finish, _ = (int(v) for v in result.tolist())
    if not 0 <= committed <= max_new:
        raise oracle.OracleError(f"committed_tokens {committed} outside [0, {max_new}]")
    return Generation(tokens[:committed].clone(), scores[:committed].clone(), committed, finish)


def forced_decode(handle: int, ids: list[int], prompt_tokens: int) -> torch.Tensor:
    """int32 [T - P, 4] cere_token_score rows: row j predicts ids[P + j] after PREFILL of ids[:P] and j forced DECODE
    steps of ids[P:P + j] (`cere_ktest_forced_decode`); unwritten rows stay -1."""
    tokens = torch.tensor(ids, dtype=torch.int32)
    scores = torch.full((len(ids) - prompt_tokens, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_forced_decode", handle, tokens, len(ids), prompt_tokens, scores)
    return scores


class DecodeTrace(NamedTuple):
    steps: list  # per DECODE step j = 1 .. C - 1: B8 name -> BF16 [1, cols] on GPU0
    scores: torch.Tensor  # int32 [C, 4] (cere_token_score bytes), bit-equal to forced_decode's


def step_layout(records: list, steps: int) -> list:
    """Splits `cere_ktest_forced_decode_trace`'s step-major entries [(name, (ndim, rows, cols))] into one list per
    DECODE step of (name, element offset, cols): every step holds the same names in the same order, one row each."""
    if not steps or not records or len(records) % steps:
        if records or steps:
            raise oracle.OracleError(f"forced_decode_trace wrote {len(records)} entries for {steps} decode steps")
        return []
    per = len(records) // steps
    order = [name for name, _ in records[:per]]
    if len(set(order)) != per:
        raise oracle.OracleError("forced_decode_trace step 1 emits a name twice")
    out: list = []
    offset = 0
    for s in range(steps):
        step = []
        for i, (name, shape) in enumerate(records[s * per:(s + 1) * per]):
            if name != order[i]:
                raise oracle.OracleError(f"forced_decode_trace step {s + 1} entry {i} is {name}, step 1 has {order[i]}")
            ndim, rows, cols = shape
            if ndim != 2 or rows != 1 or cols <= 0:
                raise oracle.OracleError(f"forced_decode_trace step {s + 1} {name}: shape record {tuple(shape)}")
            step.append((name, offset, cols))
            offset += cols
        out.append(step)
    return out


def forced_decode_trace(handle: int, ids: list[int], prompt_tokens: int, *, hidden: int, layers: int) -> DecodeTrace:
    """`cere_ktest_forced_decode_trace`: `forced_decode`'s requests with the B8 taps of every DECODE step."""
    steps = len(ids) - prompt_tokens - 1
    capacity = max(steps, 1) * 4 * layers
    tokens = torch.tensor(ids, dtype=torch.int32)
    names = torch.zeros(capacity * 128, dtype=torch.uint8)
    shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
    values = oracle.empty((capacity * hidden,), torch.bfloat16)
    scores = torch.full((len(ids) - prompt_tokens, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forced_decode_trace", handle, tokens, len(ids), prompt_tokens, names, names.numel(),
                shapes, capacity, values, values.numel() * values.element_size(), scores, counts)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 <= entries <= capacity and 0 <= name_bytes <= names.numel()
            and 0 <= value_bytes <= values.numel() * values.element_size()):
        raise oracle.OracleError(f"forced_decode_trace counts {entries}, {name_bytes}, {value_bytes} exceed capacities")
    emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"forced_decode_trace wrote {len(emitted) - 1} names for {entries} entries")
    if int((scores[:, 0] >= 0).sum()) != len(ids) - prompt_tokens:
        raise oracle.OracleError("forced_decode_trace left a score row unwritten")
    layout = step_layout([(emitted[i].decode(), tuple(int(v) for v in shapes[i].tolist())) for i in range(entries)],
                         steps)
    used = sum(cols for step in layout for _, _, cols in step)
    if used * values.element_size() != value_bytes:
        raise oracle.OracleError(f"forced_decode_trace value bytes {value_bytes} != the entries' {used * 2}")
    return DecodeTrace([{name: values[at:at + cols].view(1, cols) for name, at, cols in step} for step in layout],
                       scores)


class RowsDecode(NamedTuple):
    scores: torch.Tensor  # int32 [S, T + 1, 4] (cere_token_score bytes)
    steps: list  # per session: per decode k = 1 .. C - 1, {B8 name: BF16 [1, cols] on GPU0}, in emission order


def forward_decode_rows(handle: int, prompts: list, *, hidden: int, layers: int) -> RowsDecode:
    """`cere_ktest_forward_decode_rows` of the prompts as rows 0 .. S - 1 (all joining at round 0), each forced with
    its completion as `forced_decode_trace` forces it: prefill prompt_ids, then C - 1 decodes feeding completion[k - 1]
    (C = len(completion_ids) >= 2), every decode tapped with forced_decode_trace's headroom per step."""
    sessions = len(prompts)
    leave = [len(p.completion_ids) - 1 for p in prompts]
    rounds = max(leave)
    flat = torch.tensor([t for p in prompts for t in p.prompt_ids], dtype=torch.int32)
    lens = torch.tensor([len(p.prompt_ids) for p in prompts], dtype=torch.int64)
    forced = torch.zeros(sessions * rounds, dtype=torch.int32)
    for s, p in enumerate(prompts):
        forced[s * rounds:s * rounds + leave[s]] = torch.tensor(p.completion_ids[:leave[s]], dtype=torch.int32)
    per_step = 4 * layers
    steps = sessions * rounds
    names = torch.zeros(steps * per_step * 128, dtype=torch.uint8)
    shapes = torch.full((steps * per_step, 3), -1, dtype=torch.int64)
    values = oracle.empty((steps * per_step * hidden,), torch.bfloat16)
    scores = torch.full((sessions * (rounds + 1), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    join = torch.zeros(sessions, dtype=torch.int64)
    oracle.call("cere_ktest_forward_decode_rows", handle, sessions, flat, lens, join,
                torch.tensor(leave, dtype=torch.int64), forced, scores, names, names.numel(), shapes, steps * per_step,
                values, values.numel() * values.element_size(), counts)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 < entries <= per_step and 0 < name_bytes <= per_step * 128
            and 0 < value_bytes <= per_step * hidden * values.element_size()):
        raise oracle.OracleError(f"forward_decode_rows counts {entries}, {name_bytes}, {value_bytes} per step")
    out: list = []
    for s in range(sessions):
        session: list = []
        for k in range(1, leave[s] + 1):
            step = s * rounds + k - 1
            emitted = bytes(names[step * name_bytes:(step + 1) * name_bytes].tolist()).split(b"\0")
            if len(emitted) != entries + 1 or emitted[-1] != b"":
                raise oracle.OracleError(f"forward_decode_rows row {s} decode {k}: {len(emitted) - 1} names")
            base = step * value_bytes // values.element_size()
            element = base
            entry: dict = {}
            for i in range(entries):
                name = emitted[i].decode()
                ndim, r, c = (int(v) for v in shapes[step * entries + i].tolist())
                if ndim != 2 or r != 1 or c <= 0 or name in entry:
                    raise oracle.OracleError(f"forward_decode_rows row {s} decode {k} {name}: shape {(ndim, r, c)}")
                entry[name] = values[element:element + c].view(1, c)
                element += c
            if (element - base) * values.element_size() != value_bytes:
                raise oracle.OracleError(f"forward_decode_rows row {s} decode {k}: value bytes != the entries'")
            session.append(entry)
        out.append(session)
    return RowsDecode(scores.view(sessions, rounds + 1, 4), out)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


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



# -- the stdin frame -----------------------------------------------------------------------------------------------


class Prompt(NamedTuple):
    pid: str
    prompt_ids: list
    completion_ids: list
    top2: list  # T - 1 entries of ((id, logprob), (id, logprob)); the long frame: None before `top2_start`
    top2_start: int = 0  # the long frame's logprob_start_len (P - window)
    short: tuple = ()  # the long frame's A_short (logprob_start, entries for positions start .. 2,046), or ()

    @property
    def ids(self) -> list:
        return self.prompt_ids + self.completion_ids


class Frame(NamedTuple):
    prompts: tuple
    dump_ids: tuple
    context: int
    long: bool
    dump_layers: tuple  # the long frame's dumped layers; () = every layer
    d8b_floor_pp: float | None  # the long frame's D8b gate floor in pp; None on the short frame


def parity_ids() -> list:
    out = [json.loads(line)["id"] for line in PARITY_FILE.read_text().splitlines() if line.strip()]
    if len(out) != len(set(out)):
        raise oracle.OracleError(f"{PARITY_FILE.name} repeats an id")
    return out


def token_list(value, vocab: int, what: str) -> list:
    if not (isinstance(value, list) and all(type(t) is int and 0 <= t < vocab for t in value)):
        raise oracle.OracleError(f"frame {what} is not a list of token IDs below {vocab}")
    return value


def top2_entry(entry, vocab: int, what: str) -> tuple:
    ok = isinstance(entry, list) and len(entry) == 2 and all(
        isinstance(pair, list) and len(pair) == 2 and type(pair[0]) is int and 0 <= pair[0] < vocab
        and isinstance(pair[1], (int, float)) for pair in entry)
    if not ok:
        raise oracle.OracleError(f"frame {what} is not [[id, logprob], [id, logprob]]")
    (a, la), (b, lb) = entry
    if a == b or la < lb:
        raise oracle.OracleError(f"frame {what} is not two distinct tokens best-first")
    return (a, float(la)), (b, float(lb))


@functools.cache
def read_frame() -> Frame:
    """The `cere-b8-v2` or `cere-b8-long-v1` document from stdin, validated (prompt order, dump prompts, counts,
    shapes). IDs stay in memory."""
    raw = sys.stdin.buffer.read()
    if not raw:
        raise oracle.OracleError(f"no {FRAME} frame on stdin (run through native_dev.py numerics)")
    doc = json.loads(raw.decode("utf-8"))
    if not (isinstance(doc, dict) and doc.get("frame") in (FRAME, LONG_FRAME) and isinstance(doc.get("prompts"), list)):
        raise oracle.OracleError(f"stdin is not a {FRAME} or {LONG_FRAME} frame")
    long = doc["frame"] == LONG_FRAME
    context = doc.get("context") if long else MAX_CONTEXT
    if not (type(context) is int and context >= MAX_CONTEXT):
        raise oracle.OracleError(f"frame context {context!r} is not an integer >= {MAX_CONTEXT}")
    vocab = int(oracle.text_config()["vocab_size"])
    prompts = []
    for item in doc["prompts"]:
        pid = item.get("id") if isinstance(item, dict) else None
        if not isinstance(pid, str):
            raise oracle.OracleError("frame prompt without a string id")
        prompt_ids = token_list(item.get("prompt_ids"), vocab, f"{pid} prompt_ids")
        completion_ids = token_list(item.get("completion_ids"), vocab, f"{pid} completion_ids")
        if not prompt_ids or not completion_ids:
            raise oracle.OracleError(f"frame {pid}: empty prompt or completion")
        top2 = item.get("top2")
        total = len(prompt_ids) + len(completion_ids)
        start = item.get("logprob_start", 0) if long else 0
        if not (type(start) is int and 0 <= start < len(prompt_ids)):
            raise oracle.OracleError(f"frame {pid}: logprob_start {start!r} is not a prompt position")
        if not (isinstance(top2, list) and len(top2) == total - 1 - start):
            raise oracle.OracleError(f"frame {pid}: top2 has no entry per position {start}..T-2 (T = {total})")
        if total > context:
            raise oracle.OracleError(f"frame {pid}: {total} tokens exceed the {context}-token context")
        entries = [None] * start + [top2_entry(e, vocab, f"{pid} top2[{start + k}]") for k, e in enumerate(top2)]
        short = ()
        if long and len(prompt_ids) > SHORT_TOKENS:
            s = item.get("short")
            ok = (isinstance(s, dict) and s.get("tokens") == SHORT_TOKENS and type(s.get("logprob_start")) is int
                  and 0 <= s["logprob_start"] < SHORT_TOKENS and isinstance(s.get("top2"), list)
                  and len(s["top2"]) == SHORT_TOKENS - 1 - s["logprob_start"])
            if not ok:
                raise oracle.OracleError(f"frame {pid}: P > {SHORT_TOKENS} without a valid A_short entry")
            short = (s["logprob_start"],
                     [top2_entry(e, vocab, f"{pid} short top2[{k}]") for k, e in enumerate(s["top2"])])
        prompts.append(Prompt(pid, prompt_ids, completion_ids, entries, start, short))
    order = [p.pid for p in prompts]
    if len(set(order)) != len(order):
        raise oracle.OracleError("frame prompts repeat an id")
    if not long and sorted(order) != sorted(parity_ids()):
        raise oracle.OracleError(f"frame prompts {sorted(order)} are not the {len(parity_ids())} parity prompts")
    dump_ids = doc.get("dump_prompts")
    if not (isinstance(dump_ids, list) and dump_ids and all(isinstance(i, str) for i in dump_ids)
            and len(set(dump_ids)) == len(dump_ids) and set(dump_ids) <= set(order)):
        raise oracle.OracleError(f"frame dump_prompts {dump_ids!r} is not a non-empty list of distinct frame ids")
    if order[:len(dump_ids)] != dump_ids:
        raise oracle.OracleError(f"frame order starts {order[:len(dump_ids)]}, not the dump prompts {dump_ids}")
    layers = doc.get("dump_layers", []) if long else []
    if not (isinstance(layers, list) and all(type(n) is int and n >= 0 for n in layers) and (layers or not long)):
        raise oracle.OracleError(f"frame dump_layers {layers!r} is not a non-empty list of layers")
    floor = doc.get("d8b_floor_pp") if long else None
    if long and not (type(floor) in (int, float) and math.isfinite(floor) and floor > 0):
        raise oracle.OracleError(f"frame d8b_floor_pp {floor!r} is not a positive number of pp")
    return Frame(tuple(prompts), tuple(dump_ids), context, long, tuple(layers),
                 float(floor) if long else None)


def long_names(tc: dict, layers: tuple) -> list[str]:
    """The B8 names the long frame's reference dumped: those of `layers`, then `model`."""
    return [n for n in b8_names(tc) if n == "model" or name_kind(n)[1] in layers]


def chunk_rows(tokens: int) -> list[int]:
    return [min(LONG_CHUNK, tokens - s) for s in range(0, tokens, LONG_CHUNK)]


def find_long_passes(files: list, selected: tuple) -> dict:
    """pid -> (prompt prefill's last chunk pass, the teacher-forced pass of the chunk holding row P - 1), located in
    pass order by the `model.embed_tokens` row counts of consecutive chunk passes (as native_dev.py checks)."""
    shapes: dict = {}

    def rows(i: int):
        if i not in shapes:
            value = load_pass(files[i]).get("model.embed_tokens")
            shapes[i] = int(value.shape[0]) if isinstance(value, torch.Tensor) else None
        return shapes[i]

    found, cursor = {}, 0
    for p in selected:
        located = []
        for what, count in (("prompt prefill", len(p.prompt_ids)), ("teacher-forced prefill", len(p.ids))):
            sizes = chunk_rows(count)
            while cursor + len(sizes) < len(files) and [rows(cursor + k) for k in range(len(sizes))] != sizes:
                cursor += 1
            if cursor + len(sizes) >= len(files):
                raise oracle.OracleError(f"{p.pid}: no {len(sizes)} consecutive dump passes (followed by another) "
                                         f"hold the {what} chunks")
            located.append(cursor)
            cursor += len(sizes)
        chunk = (len(p.prompt_ids) - 1) // LONG_CHUNK
        found[p.pid] = (located[0] + chunk, located[1] + chunk)
    return found


# -- the reference dump --------------------------------------------------------------------------------------------


def pass_files() -> list:
    dirs = sorted(p for p in REFDUMP.glob("TP0_PP0_Rank0_pid*") if p.is_dir())
    if len(dirs) != 1:
        raise oracle.OracleError(f"{REFDUMP} holds {len(dirs)} TP0_PP0_Rank0_pid* directories, expected 1")
    files = [p for p in dirs[0].iterdir() if re.fullmatch(r"Pass[0-9]{5}\.pt", p.name)]
    files.sort(key=lambda p: int(p.name[4:9]))
    if [int(p.name[4:9]) for p in files] != list(range(len(files))):
        raise oracle.OracleError(f"{dirs[0].name}: pass files are not numbered 0..{len(files) - 1} without gaps")
    return files


def load_pass(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)


def dump_key(dump: dict, name: str):
    """The dump key of a B8 name (the attribute path drops `self_attn`), or None."""
    for key in (name, name.replace(".self_attn.", ".")):
        if key in dump:
            return key
    return None


def dump_tensor(dump: dict, name: str, rows: int, hidden: int, where: str) -> torch.Tensor:
    key = dump_key(dump, name)
    if key is None:
        raise oracle.OracleError(f"{where}: the dump lacks {name}")
    value = dump[key]
    if not isinstance(value, torch.Tensor):
        raise oracle.OracleError(f"{where}: {key} holds {type(value).__name__}, not one tensor")
    if tuple(value.shape) != (rows, hidden):
        raise oracle.OracleError(f"{where}: {key} has shape {tuple(value.shape)}, expected {(rows, hidden)}")
    return value.to(oracle.DEVICE)


class DumpPasses(NamedTuple):
    prompt: dict  # the chat completion's prompt prefill (P rows)
    after_prompt: dict  # the next pass: holds the prompt prefill's `model` output
    forced: dict  # the teacher-forced prefill (T rows)
    after_forced: dict  # the next pass: holds the teacher-forced prefill's `model` output
    prompt_index: int  # pass numbers of `prompt` and `forced`
    forced_index: int


def embeds_match(dump: dict, embed: torch.Tensor, ids: list) -> bool:
    value = dump.get("model.embed_tokens")
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (len(ids), embed.shape[1]):
        return False
    expected = embed[torch.tensor(ids, dtype=torch.long)]
    return value.dtype == expected.dtype and torch.equal(value, expected)


def find_passes(files: list, selected: tuple, embed: torch.Tensor) -> dict:
    """pid -> DumpPasses for the selected (dump) prompts, located in pass order by their bit-exact embedding rows."""
    found = {}
    cursor = 0
    for p in selected:
        located = []
        for ids, what in ((p.prompt_ids, "prompt prefill"), (p.ids, "teacher-forced prefill")):
            while cursor < len(files) and not embeds_match(load_pass(files[cursor]), embed, ids):
                cursor += 1
            if cursor + 1 >= len(files):
                raise oracle.OracleError(f"{p.pid}: no dump pass (followed by another) embeds the {what} rows "
                                         f"(was the radix cache flushed before the request?)")
            located.append(cursor)
            cursor += 1
        prompt_pass, forced_pass = located
        found[p.pid] = DumpPasses(load_pass(files[prompt_pass]), load_pass(files[prompt_pass + 1]),
                                  load_pass(files[forced_pass]), load_pass(files[forced_pass + 1]),
                                  prompt_pass, forced_pass)
    return found


def decode_passes(files: list, located: DumpPasses, p: Prompt, embed: torch.Tensor) -> list:
    """[(step pass, pass holding its `model`)] for decode steps j = 1 .. C - 1 of prompt p (tasks/m22d-spec.md D3).
    Strictly between the prompt pass and the teacher-forced pass sit exactly C one-token passes: pass prompt + j
    embeds tokens[P + j - 1] for j = 1 .. C (j = C is the trailing pass); step j is pass prompt + j, and its `model`
    output sits in the next pass (the trailing pass for j = C - 1). Any other count or embedding fails naming p and
    the pass number (no token IDs)."""
    prompt_len, completion = len(p.prompt_ids), len(p.completion_ids)
    between = located.forced_index - located.prompt_index - 1
    if between != completion:
        raise oracle.OracleError(f"{p.pid}: {between} dump passes between the prompt pass {located.prompt_index} and "
                                 f"the teacher-forced pass {located.forced_index}, expected C = {completion} "
                                 "one-token passes (C - 1 decode steps and a trailing pass)")
    passes = []
    for j in range(1, completion + 1):
        index = located.prompt_index + j
        dump = load_pass(files[index])
        if not embeds_match(dump, embed, [p.ids[prompt_len + j - 1]]):
            what = "the trailing pass" if j == completion else f"decode step {j}"
            raise oracle.OracleError(f"{p.pid}: dump pass {index} ({what}) does not embed the forced token's row")
        passes.append(dump)
    return [(passes[j - 1], passes[j]) for j in range(1, completion)]


def name_kind(name: str) -> tuple:
    """(name pattern, layer) of a B8 name: `model.layers.N.rest` -> (rest, N); `model` -> (model, -1)."""
    if name == "model":
        return name, -1
    _, _, layer, rest = name.split(".", 3)
    return rest, int(layer)


# -- D1: teacher-forced KL (m214 design note D1) -------------------------------------------------------------------


def kl_measure():
    """The D1 calculus (`kl_measure.py` in this directory, on sys.path in the container); loaded by name: suite files
    import only stdlib, torch, safetensors, sglang, flashinfer and oracle."""
    return importlib.import_module("kl_measure")


def tf_kl_transient_bytes(vocab: int, hidden: int) -> int:
    """The worst concurrent D1 transients of one chunk: the reference log-probs (fp64) held while the candidate's are
    built from fp32 logits, an fp64 copy and logsumexp's fp64 temporary, plus one widened lm_head slice (fp32)."""
    return TF_KL_CHUNK * vocab * (8 + 4 + 8 + 8) + TF_KL_VOCAB_SLICE * hidden * 4


def release_device_cache() -> None:
    """Returns the caching allocator's free blocks (earlier tests' tensors) to the device and restarts its peak."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(oracle.DEVICE)


def device_peak_reserved() -> int:
    """The caching allocator's peak reserved bytes since `release_device_cache`."""
    return int(torch.cuda.max_memory_reserved(oracle.DEVICE))


def tf_logprobs(rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """D1 log-probs of `model` rows [n, hidden] (BF16): rows x weight^T (BF16 [vocab, hidden]) accumulated in fp32
    (both widened exactly to fp32, TF32 off, TF_KL_VOCAB_SLICE lm_head rows at a time), log-softmax in fp64."""
    if torch.backends.cuda.matmul.allow_tf32:
        raise oracle.OracleError("TF32 matmul is enabled: D1 logits must accumulate in fp32")
    x = rows.to(oracle.DEVICE).float()
    vocab = int(weight.shape[0])
    logits = torch.empty((int(x.shape[0]), vocab), dtype=torch.float32, device=oracle.DEVICE)
    for s in range(0, vocab, TF_KL_VOCAB_SLICE):
        logits[:, s:s + TF_KL_VOCAB_SLICE] = x @ weight[s:s + TF_KL_VOCAB_SLICE].float().T
    out = logits.double()
    del logits
    out -= torch.logsumexp(out, dim=-1, keepdim=True)
    return out


def tf_kl_rows(ref: torch.Tensor, cand: torch.Tensor, weight: torch.Tensor, targets: list, *,
               cand_logprobs=None) -> list:
    """`kl_measure.RowStats` per row of the (reference, candidate) `model` rows [n, hidden], TF_KL_CHUNK rows at a
    time through the one `tf_logprobs`: KL(ref || cand), both top-1 and top-2 margins, and where targets[r] is not
    None both NLLs of that token. Token ids stay in this process. `cand_logprobs(rows, weight)` replaces
    `tf_logprobs` on the candidate side only (tasks/m23-spec.md D9Q's negative controls: a degraded lm_head)."""
    cand_logprobs = cand_logprobs or tf_logprobs
    km = kl_measure()
    rows = int(ref.shape[0])
    if int(cand.shape[0]) != rows or len(targets) != rows:
        raise oracle.OracleError(f"D1: {rows} reference rows, {int(cand.shape[0])} candidate rows, "
                                 f"{len(targets)} targets")
    out = []
    for s in range(0, rows, TF_KL_CHUNK):
        e = min(s + TF_KL_CHUNK, rows)
        picked = [r - s for r in range(s, e) if targets[r] is not None]
        at = (torch.tensor(picked, dtype=torch.long, device=oracle.DEVICE),
              torch.tensor([targets[s + r] for r in picked], dtype=torch.long, device=oracle.DEVICE))
        lr = tf_logprobs(ref[s:e], weight)
        r_top = lr.topk(2, dim=-1)
        r_nll = (-lr[at]).tolist() if picked else []
        lc = cand_logprobs(cand[s:e], weight)
        c_top = lc.topk(2, dim=-1)
        c_nll = (-lc[at]).tolist() if picked else []
        lc.neg_().add_(lr)  # log p_ref - log p_cand, in place
        kl = lr.exp_().mul_(lc).sum(dim=-1).tolist()
        r_ids, r_vals = r_top.indices[:, 0].tolist(), r_top.values.tolist()
        c_ids, c_vals = c_top.indices[:, 0].tolist(), c_top.values.tolist()
        del lr, lc, r_top, c_top
        nll = dict(zip(picked, zip(r_nll, c_nll)))
        for k in range(e - s):
            ref_nll, cand_nll = nll.get(k, (None, None))
            out.append(km.RowStats(kl[k], r_ids[k], c_ids[k], r_vals[k][0] - r_vals[k][1], c_vals[k][0] - c_vals[k][1],
                                   ref_nll, cand_nll))
    return out


class B8Numerics(unittest.TestCase):
    long = False  # setUpClass sets it from the frame (cere-b8-long-v1 = True)

    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.hidden = int(cls.tc["hidden_size"])
        cls.layers = int(cls.tc["num_hidden_layers"])
        frame = read_frame()
        cls.prompts, cls.dump_ids, cls.long = frame.prompts, frame.dump_ids, frame.long
        cls.dump_layers, cls.d8b_floor_pp = frame.dump_layers, frame.d8b_floor_pp
        # m26 design note K4: the short frame's model takes ROWS_R sessions (the long frame's context keeps one).
        model = create_model(frame.context, 1 if cls.long else ROWS_R)
        cls.addClassCleanup(release_model, model.handle)
        upload_and_seal(model)
        cls.handle = model.handle

    def trace(self, p: Prompt) -> Trace:
        trace = forward_trace(self.handle, p.ids, hidden=self.hidden, layers=self.layers)
        missing = [n for n in b8_names(self.tc) if n not in trace.values]
        self.assertTrue(not missing, f"{ROW} {p.pid}: the native trace lacks {missing}")
        oracle.check_count(self, ROW, f"{p.pid} trace score rows", int(trace.scores.shape[0]), expected=len(p.ids))
        return trace

    def test_decode_forced_top1(self):
        # j indexes completion positions: dec[j] is native's decode-mode top-2 after PREFILL(prompt) and j forced
        # DECODE steps, pre[j] native's prefill-mode (one forward of all T tokens) top-2 at position P - 1 + j, and
        # p.top2[P - 1 + j] SGLang's teacher-forced top-2; all three predict completion token j.
        positions = decode_misses = prefill_misses = ref_misses = 0
        both = decode_only = prefill_only = modes_differ = first_token_misses = exact_prompts = over_tau = 0
        miss_margins: list = []
        for p in self.prompts:
            prompt_len, completion_len = len(p.prompt_ids), len(p.completion_ids)
            dec = forced_decode(self.handle, p.ids, prompt_len)
            oracle.check_count(self, ROW, f"{p.pid} decode-forced score rows", int((dec[:, 0] >= 0).sum()),
                               expected=completion_len)
            # Self-consistency: up to the first position where native greedy leaves the completion, greedy and
            # forced decoding feed identical tokens, so their scores must agree in all four words.
            greedy = generate(self.handle, p.prompt_ids, completion_len)
            same = next((i for i in range(greedy.committed) if int(greedy.tokens[i]) != p.completion_ids[i]),
                        greedy.committed)
            prefix = min(same + 1, greedy.committed)
            prefix_exact = torch.equal(dec[:prefix], greedy.scores[:prefix])
            exact_prompts += prefix_exact
            if self.long:
                # m22c design note D8b (Z2AC-02): only the completion rows P - 1 .. T - 2 are needed; a full trace
                # of a 72K frame would hold 4 * layers * T hidden rows on GPU0
                trace = forward_trace(self.handle, p.ids, hidden=self.hidden, layers=self.layers,
                                      last_rows=completion_len + 1)
                # Z2AC2-02: score rows start at -1; every row of the bounded window must have been written
                oracle.check_count(self, ROW, f"{p.pid} long prefill score rows",
                                   int((trace.scores[:completion_len + 1, 0] >= 0).sum()),
                                   expected=completion_len + 1)
                pre = trace.scores[:completion_len, 0].tolist()
            else:
                trace = self.trace(p)
                pre = trace.scores[prompt_len - 1:len(p.ids) - 1, 0].tolist()
            del trace
            dec_top1 = dec[:, 0].tolist()
            dec_runner = dec[:, 1].tolist()
            dec_logits = dec[:, 2:4].contiguous().view(torch.float32).tolist()
            misses = []
            counts = [0, 0, 0]  # decode, prefill, SGLang teacher-forced vs the completion
            for j, token in enumerate(p.completion_ids):
                (ref_id, ref_lp), (_, runner_lp) = p.top2[prompt_len - 1 + j]
                d_miss, p_miss, r_miss = dec_top1[j] != token, pre[j] != token, ref_id != token
                counts[0] += d_miss
                counts[1] += p_miss
                counts[2] += r_miss
                both += d_miss and p_miss
                decode_only += d_miss and not p_miss
                prefill_only += p_miss and not d_miss
                modes_differ += dec_top1[j] != pre[j]
                if d_miss:
                    native_margin = dec_logits[j][0] - dec_logits[j][1]
                    miss_margins.append(native_margin)
                    first_token_misses += j == 0
                    over_tau += ref_lp - runner_lp >= TAU
                    # j: SGLang's teacher-forced top-2 margin / native decode top-2 logit margin, r when native's
                    # runner-up is the completion token, P when prefill mode also missed; no IDs or text
                    misses.append(f"{j}:{ref_lp - runner_lp:.3f}/{native_margin:.3f}"
                                  f"{'r' if dec_runner[j] == token else 'x'}{'P' if p_miss else '-'}")
            positions += completion_len
            decode_misses += counts[0]
            prefill_misses += counts[1]
            ref_misses += counts[2]
            oracle.report_row(ROW, f"{p.pid} decode-forced", phase="diagnostic", prompt_tokens=prompt_len,
                              completion_tokens=completion_len, misses=" ".join(misses),
                              native_decode_vs_ref_greedy=counts[0], native_prefill_vs_ref_greedy=counts[1],
                              ref_vs_own_greedy=counts[2], greedy_committed=greedy.committed,
                              greedy_prefix=prefix, prefix_bit_exact=prefix_exact)
        miss_margins.sort()
        oracle.report_row(ROW, "decode-forced top-1", prompts=len(self.prompts), completion_positions=positions,
                          native_decode_vs_ref_greedy=decode_misses, native_prefill_vs_ref_greedy=prefill_misses,
                          ref_vs_own_greedy=ref_misses, both=both, decode_only=decode_only,
                          prefill_only=prefill_only, decode_vs_prefill_top1=modes_differ,
                          first_token_misses=first_token_misses,
                          worst_miss_native_margin=miss_margins[-1] if miss_margins else 0.0,
                          median_miss_native_margin=statistics.median(miss_margins) if miss_margins else 0.0,
                          misses_over_tau=over_tau, prefix_bit_exact_prompts=exact_prompts)
        oracle.check_count(self, ROW, "decode-forced prefix bit-exact prompts", exact_prompts,
                           expected=len(self.prompts))

    def long_layers_vs_dump(self):
        """tasks/m22c-spec.md D8b: per dumped name, rel-L2 at the prompt's last chunk rows [s, P): native trace of
        the prompt (last rows) vs SGLang's prompt-prefill chunk pass, c = rel-L2(SGLang's teacher-forced chunk rows
        [s, P), the prompt chunk). Report rows only."""
        names = long_names(self.tc, self.dump_layers)
        selected = self.prompts[:len(self.dump_ids)]
        oracle.report_row(ROW, "layer prompts", prompts=",".join(p.pid for p in selected))
        files = pass_files()
        located = find_long_passes(files, selected)
        exceeded = total = 0
        for p in selected:
            prompt_pass, forced_pass = located[p.pid]
            start = (len(p.prompt_ids) - 1) // LONG_CHUNK * LONG_CHUNK
            rows = len(p.prompt_ids) - start
            forced_rows = chunk_rows(len(p.ids))[start // LONG_CHUNK]
            trace = forward_trace(self.handle, p.prompt_ids, hidden=self.hidden, layers=self.layers, last_rows=rows)
            for name in names:
                after = name == "model"
                where = f"{p.pid} chunk {start // LONG_CHUNK}{' + 1' if after else ''}"
                ref = dump_tensor(load_pass(files[prompt_pass + after]), name, rows, self.hidden, f"{where} prompt")
                forced = dump_tensor(load_pass(files[forced_pass + after]), name, forced_rows, self.hidden,
                                     f"{where} forced")
                self.assertIn(name, trace.values, f"{ROW} {p.pid}: the native trace lacks {name}")
                c = oracle.calibrate(ROW, f"{p.pid} {name}", forced[:rows], ref, oracle.FLOOR_BLOCK)
                limit = oracle.calibrated_limit(c, oracle.FLOOR_BLOCK)
                value = oracle.rel_l2(trace.values[name], ref)
                passed = math.isfinite(value) and value <= limit
                total += 1
                exceeded += not passed
                oracle.report_row(ROW, f"{p.pid} {name}", phase="result", chunk_start=start, rows=rows, c=c,
                                  limit=limit, measured=value, passed=passed)
            del trace
        oracle.report_row(ROW, "long layer tensors", prompts=len(selected), names=total, exceeded=exceeded)

    def long_teacher_forced(self):
        """tasks/m22c-spec.md D8b, tasks/m25-spec.md D8b: A_long,i over the prompt positions logprob_start .. P - 2
        (the last 1,024 prompt positions, completion positions excluded), A_short,i over the 2,048-token truncation's
        window, d_i = A_long,i - A_short,i for P > 2,048; gate mean(d) >= -max(floor, t * sd / sqrt n), the floor the
        frame's d8b_floor_pp."""
        d = []
        for p in self.prompts:
            positions = len(p.prompt_ids) - 1 - p.top2_start
            trace = forward_trace(self.handle, p.prompt_ids, hidden=self.hidden, layers=self.layers,
                                  last_rows=positions + 1)
            native = trace.scores[:positions, 0].tolist()
            del trace
            a_long = sum(native[k] == p.top2[p.top2_start + k][0][0] for k in range(positions)) / positions
            row = {"tokens": len(p.ids), "prompt_tokens": len(p.prompt_ids), "positions": positions,
                   "a_long": a_long}
            if p.short:
                start, entries = p.short
                trace = forward_trace(self.handle, p.prompt_ids[:SHORT_TOKENS], hidden=self.hidden,
                                      layers=self.layers, last_rows=SHORT_TOKENS - start)
                native = trace.scores[:len(entries), 0].tolist()
                del trace
                a_short = sum(native[k] == e[0][0] for k, e in enumerate(entries)) / len(entries)
                d.append(100.0 * (a_long - a_short))
                row |= {"short_positions": len(entries), "a_short": a_short, "d_pp": d[-1]}
            oracle.report_row(ROW, f"{p.pid} long teacher-forced top-1", **row)
        n = len(d)
        self.assertTrue(n >= 2, f"{ROW} long teacher-forced: {n} prompts with P > {SHORT_TOKENS}, the gate needs 2")
        mean, sd = statistics.fmean(d), statistics.stdev(d)
        t = T95[n - 2] if n - 1 <= len(T95) else 1.645
        bound = -max(self.d8b_floor_pp, t * sd / math.sqrt(n))
        oracle.report_row(ROW, "long teacher-forced paired", n=n, mean_d_pp=mean, sd_d_pp=sd, t=t, bound_pp=bound,
                          passed=mean >= bound)
        self.assertTrue(mean >= bound, f"{ROW} long teacher-forced mean(d) {mean:.3f} pp < {bound:.3f} pp")

    def test_layer_tensors_vs_dump(self):
        if self.long:
            return self.long_layers_vs_dump()
        names = b8_names(self.tc)
        selected = self.prompts[:len(self.dump_ids)]
        # native_dev.py numerics requires exactly this row to name its --prompts (the frame's dump_prompts)
        oracle.report_row(ROW, "layer prompts", prompts=",".join(p.pid for p in selected))
        embed = oracle.real_tensor(EMBED_NAME, device=None)
        passes = find_passes(pass_files(), selected, embed)
        del embed
        refs: dict = {}
        cals: dict = {}
        for p in selected:
            d = passes[p.pid]
            rows, total = len(p.prompt_ids), len(p.ids)
            for name in names:
                if name == "model":
                    prompt_value = dump_tensor(d.after_prompt, name, rows, self.hidden, f"{p.pid} prompt pass + 1")
                    forced_value = dump_tensor(d.after_forced, name, total, self.hidden, f"{p.pid} forced pass + 1")
                else:
                    prompt_value = dump_tensor(d.prompt, name, rows, self.hidden, f"{p.pid} prompt pass")
                    forced_value = dump_tensor(d.forced, name, total, self.hidden, f"{p.pid} forced pass")
                cals[p.pid, name] = oracle.calibrate(ROW, f"{p.pid} {name}", forced_value[:rows], prompt_value,
                                                     oracle.FLOOR_BLOCK)
                refs[p.pid, name] = forced_value
        del passes
        measured: dict = {}
        for p in selected:
            trace = self.trace(p)
            for name in names:
                measured[p.pid, name] = oracle.rel_l2(trace.values[name], refs[p.pid, name])
            del trace
        worst: dict = {}
        for (pid, name), value in measured.items():
            kind, layer = name_kind(name)
            limit = oracle.calibrated_limit(cals[pid, name], oracle.FLOOR_BLOCK)
            if kind not in worst or value / limit > worst[kind][0]:
                worst[kind] = (value / limit, pid, layer, value, cals[pid, name], limit)
        for kind, (ratio, pid, layer, value, c, limit) in sorted(worst.items()):
            oracle.report_row(ROW, f"worst {kind}", prompt=pid, layer=layer, rel_l2=value, c=c, limit=limit,
                              ratio=ratio)
        # m22d design note D9: every row is reported, then the test fails once if any exceeded its limit
        exceeded = []
        for (pid, name), value in measured.items():
            c = cals[pid, name]
            limit = oracle.calibrated_limit(c, oracle.FLOOR_BLOCK)
            passed = math.isfinite(value) and value <= limit
            oracle.report_row(ROW, f"{pid} {name}", phase="result", c=c, limit=limit, measured=value, passed=passed)
            if not passed:
                exceeded.append(f"{pid} {name} (rel-L2 {value:.6g} > {limit:.6g})")
        self.assertFalse(exceeded, f"{ROW} {len(exceeded)} of {len(measured)} layer tensors exceed their limits, "
                                   f"first {exceeded[0] if exceeded else None}")

    def test_decode_steps_vs_dump(self):
        # m22d design note D3/D4: per dump prompt and DECODE step j = 1 .. C - 1, per B8 name, d = rel-L2(native
        # step j, SGLang's decode pass j) against limit max(FLOOR_BLOCK, 2c), c = rel-L2(SGLang's decode pass j,
        # SGLang's teacher-forced row P + j - 1). Report rows only; fails on a locator failure, an empty frame or a
        # ktest error.
        if self.long:  # cere-b8-long-v1: decode steps are not dumped per step (m22c design note D8b)
            oracle.report_row(ROW, "decode-step", applicable=False, reason="long frame dumps no decode step")
            return
        names = b8_names(self.tc)
        selected = self.prompts[:len(self.dump_ids)]
        embed = oracle.real_tensor(EMBED_NAME, device=None)
        files = pass_files()
        located = find_passes(files, selected, embed)
        steps = {p.pid: decode_passes(files, located[p.pid], p, embed) for p in selected}
        del embed
        self.assertTrue(any(steps.values()), f"{ROW} decode-step: the dump prompts hold no decode step")
        per_name = {name: [0, 0, 0.0] for name in names}  # exceeding steps, steps, max d/limit
        for p in selected:
            d = located[p.pid]
            prompt_len, total = len(p.prompt_ids), len(p.ids)
            forced = {name: dump_tensor(d.after_forced if name == "model" else d.forced, name, total, self.hidden,
                                        f"{p.pid} forced pass{' + 1' if name == 'model' else ''}") for name in names}
            native = forced_decode_trace(self.handle, p.ids, prompt_len, hidden=self.hidden, layers=self.layers)
            self.assertEqual(len(native.steps), len(steps[p.pid]), f"{ROW} {p.pid}: native decode steps")
            first = None
            for j, ((step_pass, model_pass), values) in enumerate(zip(steps[p.pid], native.steps), start=1):
                missing = [n for n in names if n not in values]
                self.assertTrue(not missing, f"{ROW} {p.pid}: native decode step {j} lacks {missing}")
                for name in names:
                    where = f"{p.pid} dump pass {d.prompt_index + j + (name == 'model')}"
                    ref = dump_tensor(model_pass if name == "model" else step_pass, name, 1, self.hidden, where)
                    c = oracle.rel_l2(ref, forced[name][prompt_len + j - 1:prompt_len + j])
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_BLOCK)
                    dist = oracle.rel_l2(values[name], ref)
                    exceeds = not (math.isfinite(dist) and dist <= limit)
                    counts = per_name[name]
                    counts[0] += exceeds
                    counts[1] += 1
                    counts[2] = max(counts[2], dist / limit)
                    if exceeds and first is None:
                        first = (j, name_kind(name)[1], name, dist, limit)
            del native, forced
            if first is None:
                oracle.report_row(ROW, "decode-step first-exceeding", prompt=p.pid, none=True)
            else:
                step, layer, name, dist, limit = first
                oracle.report_row(ROW, "decode-step first-exceeding", prompt=p.pid, step=step, layer=layer, name=name,
                                  d=dist, limit=limit)
        for name, (exceeding, count, ratio) in per_name.items():
            oracle.report_row(ROW, "decode-step per-name", name=name, exceeding_steps=exceeding, steps=count,
                              max_ratio=ratio)

    def test_decode_rows_vs_dump(self):
        # m26 design note K4 (native vs native, D6): the dump prompts as rows ROWS_DUMP of an R = ROWS_R batched
        # decode step equal their R = 1 forced decode trace bit for bit, per decode step and B8 name, and per score
        # row. Every prompt is reported; the test fails once at the end.
        if self.long:  # cere-b8-long-v1: the model has one session
            oracle.report_row(ROW, "decode-rows", applicable=False, reason="long frame model has one session")
            return
        dump = [p for p in self.prompts[:len(self.dump_ids)] if len(p.completion_ids) >= 2]
        others = [p for p in self.prompts[len(self.dump_ids):] if len(p.completion_ids) >= 2]
        self.assertTrue(dump, f"{ROW} decode-rows: no dump prompt has a decode step")
        if len(others) < ROWS_R - 1:
            raise oracle.OracleError(f"decode-rows needs {ROWS_R - 1} non-dump prompts with a decode step")
        differing: list = []
        for i in range(0, len(dump), len(ROWS_DUMP)):
            group = dump[i:i + len(ROWS_DUMP)]
            fill = iter(others)
            rows = [None] * ROWS_R
            for slot, p in zip(ROWS_DUMP, group):
                rows[slot] = p
            rows = [p if p is not None else next(fill) for p in rows]
            batched = forward_decode_rows(self.handle, rows, hidden=self.hidden, layers=self.layers)
            for slot, p in zip(ROWS_DUMP, group):
                completion = len(p.completion_ids)
                ref = forced_decode_trace(self.handle, p.ids, len(p.prompt_ids), hidden=self.hidden,
                                          layers=self.layers)
                score_rows = sum(not same_bits(batched.scores[slot, k], ref.scores[k]) for k in range(completion))
                steps = batched.steps[slot]
                self.assertEqual(len(steps), len(ref.steps), f"{ROW} {p.pid}: batched decode steps")
                first = None
                tensors = bad = 0
                for j, (got, want) in enumerate(zip(steps, ref.steps), start=1):
                    self.assertEqual(list(got), list(want), f"{ROW} {p.pid}: decode step {j} names differ")
                    for name in want:
                        tensors += 1
                        if not same_bits(got[name], want[name]):
                            bad += 1
                            if first is None:
                                first = (j, name)
                if first is None:
                    oracle.report_row(ROW, "decode-rows first-differing", prompt=p.pid, slot=slot, none=True)
                else:
                    oracle.report_row(ROW, "decode-rows first-differing", prompt=p.pid, slot=slot, step=first[0],
                                      name=first[1])
                oracle.report_row(ROW, "decode-rows counts", prompt=p.pid, slot=slot, rows=ROWS_R, steps=len(steps),
                                  tensors=tensors, differing_tensors=bad, score_rows=completion,
                                  differing_score_rows=score_rows, passed=bad == 0 and score_rows == 0)
                if bad or score_rows:
                    differing.append(p.pid)
                del ref
            del batched
            torch.cuda.empty_cache()
        self.assertFalse(differing, f"{ROW} decode-rows: {differing} differ from their R = 1 trace")

    def test_teacher_forced_kl(self):
        # m214 design note D1: every frame prompt (not only the dump prompts); the yardstick for every prompt is
        # computed before the first native trace; fails only on a locator failure, a NaN or a ktest error.
        if self.long:
            oracle.report_row(ROW, "tf-kl", applicable=False, reason="D1 runs on the cere-b8-v2 frame")
            return
        km = kl_measure()
        release_device_cache()
        embed = oracle.real_tensor(EMBED_NAME, device=None)
        located = find_passes(pass_files(), self.prompts, embed)
        del embed
        weight = oracle.real_tensor(LM_HEAD_NAME, device=oracle.DEVICE)
        vocab, hidden = (int(n) for n in weight.shape)
        refs, yards = {}, {}
        for p in self.prompts:
            d = located[p.pid]
            prompt_len, total = len(p.prompt_ids), len(p.ids)
            prompt_model = dump_tensor(d.after_prompt, "model", prompt_len, self.hidden, f"{p.pid} prompt pass + 1")
            forced_model = dump_tensor(d.after_forced, "model", total, self.hidden, f"{p.pid} forced pass + 1")
            yards[p.pid] = tf_kl_rows(forced_model[:prompt_len], prompt_model, weight, [None] * prompt_len)
            refs[p.pid] = forced_model
        del located
        arms = []
        for p in self.prompts:
            prompt_len, total = len(p.prompt_ids), len(p.ids)
            trace = self.trace(p)
            # row r predicts token r + 1: NLL only over prompt rows 0 .. P-2 (the prompt's own next tokens)
            targets = p.ids[1:prompt_len] + [None] * (total - prompt_len + 1)
            rows = tf_kl_rows(refs.pop(p.pid), trace.values["model"], weight, targets)
            del trace
            arms.append((rows[:prompt_len], rows[prompt_len:], yards.pop(p.pid)))
        del weight
        oracle.report_row(ROW, "tf-kl device", phase="diagnostic", torch_peak_reserved_mib=device_peak_reserved() >> 20,
                          transient_mib=tf_kl_transient_bytes(vocab, hidden) >> 20,
                          transient_budget_mib=TF_KL_DEVICE_BUDGET >> 20, chunk_rows=TF_KL_CHUNK)
        for p, arm in zip(self.prompts, arms):
            s = km.prompt_summary(*arm)
            last = s.pop("last")
            oracle.report_row(ROW, f"{p.pid} tf-kl", phase="diagnostic", **s)
            oracle.report_row(ROW, f"{p.pid} tf-kl last-prompt-row", phase="diagnostic", **last)
        oracle.report_row(ROW, "tf-kl aggregate", phase="diagnostic", **km.aggregate(arms))

    def test_teacher_forced_top1(self):
        if self.long:
            return self.long_teacher_forced()
        total = 0
        agree = 0
        worst_margin = 0.0
        over_margin = 0
        completion_positions = ref_self_total = native_self_total = 0
        for p in self.prompts:
            trace = self.trace(p)
            native_top1 = trace.scores[:len(p.ids) - 1, 0].tolist()
            native_runner = trace.scores[:len(p.ids) - 1, 1].tolist()
            native_logits = trace.scores[:len(p.ids) - 1, 2:4].contiguous().view(torch.float32).tolist()
            misses = []
            for k, ((ref_id, ref_lp), (_, runner_lp)) in enumerate(p.top2):
                total += 1
                if native_top1[k] == ref_id:
                    agree += 1
                    continue
                margin = ref_lp - runner_lp
                native_margin = native_logits[k][0] - native_logits[k][1]
                second = "r" if native_runner[k] == ref_id else "x"
                misses.append(f"{k}:{margin:.3f}/{native_margin:.3f}{second}")
                worst_margin = max(worst_margin, margin)
                over_margin += margin > MAX_DISAGREE_MARGIN
            # Noise floor: at completion positions (k >= P - 1 predicts completion token k + 1 - P), how often
            # SGLang's own teacher-forced top-1 differs from its greedy decode, and how often native's does.
            first = len(p.prompt_ids) - 1
            completion = p.ids[first + 1:]
            ref_self = sum(p.top2[first + j][0][0] != t for j, t in enumerate(completion))
            native_self = sum(native_top1[first + j] != t for j, t in enumerate(completion))
            completion_positions += len(completion)
            ref_self_total += ref_self
            native_self_total += native_self
            if misses or ref_self or native_self:
                # positions (k predicts token k+1), SGLang's top-2 margin / native's top-2 logit margin, r when
                # native's runner-up is SGLang's top-1; counts only: no IDs or text
                oracle.report_row(ROW, f"{p.pid} top-1 disagreements", phase="diagnostic", tokens=len(p.ids),
                                  prompt_tokens=len(p.prompt_ids), misses=" ".join(misses),
                                  ref_vs_own_greedy=ref_self, native_vs_ref_greedy=native_self)
            del trace
        oracle.report_row(ROW, "teacher-forced top-1", prompts=len(self.prompts), positions=total, agree=agree,
                          disagree=total - agree, worst_disagree_margin=worst_margin,
                          disagree_over_margin=over_margin, completion_positions=completion_positions,
                          ref_vs_own_greedy=ref_self_total, native_vs_ref_greedy=native_self_total)
        self.assertTrue(agree * 1000 >= AGREE_PER_MILLE * total,
                        f"{ROW} top-1 agreement {agree}/{total} is below 99.9 %")
        oracle.check_count(self, ROW, "disagreements with SGLang margin > 0.1", over_margin, expected=0)


if __name__ == "__main__":
    unittest.main()
