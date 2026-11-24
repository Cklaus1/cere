"""K4 (tasks/m22-spec.md): the PLE n-gram path of layer 1 vs SGLang on real weights.

Suite `kernels` (inside the pinned SGLang image on GPU0):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_*.py' -v`.
Every case reports row "K4".

What is compared ($S181 = the reference image's site-packages, `sglang/srt/models/qwen4_exp.py` unless named):
- hash, bit-exact: n-gram head IDs vs `Qwen4ExpNGramEmbedding._hash_contexts` (`:603`) - the eager torch path
  (`decode_sized=False`, prefill) and the fused Triton path it routes decode-sized calls to
  (`sglang/kernels/ops/qwen4_ple.py:98`, `fused_qwen4_ngram_hash`) - on 100,000 seeded random token windows (EOS
  injected at rate 1/8 per position so every EOS-segment branch of `_shift_right_ignore_eos` (`:658`) is hit) and on
  every token window of the 24 parity prompts (`bench/prompts/parity_v1.jsonl`, rendered with the chat template,
  each a fresh request whose history is two EOS tokens as `NGramPool` initialises it; token IDs stay in memory).
- gather, bit-exact: rows of the mapped pinned PLE table vs `Qwen4ExpPinnedHostEmbedding.gather` (`:750-870`, the
  Triton UVA kernel `_gather_ple_embedding_from_pinned_kernel` `:717`) for the IDs of the parity windows, 65,536
  seeded uniform IDs, every shard boundary, and out-of-range IDs (which SGLang turns into zero rows).
- block: `Qwen4ExpPLELayer.forward` (`:1162`) built as `_init_qwen4_exp_layer_extensions` builds it (`:1241-1255`,
  layer 1 because `ple_layer_ids` = [2] is 1-based) with `--ple-offload-embedding` (the bench server's flag), driven
  the way `Qwen4ExpModel.forward` drives it (`:1620-1668`): `_prepare_ple_batch` (`:108`), `start_prefetch` (`:1106`),
  the layer forward, `_commit_ple_batch` (`:276`). The query is the 4-stream HC state: `ple_query = hidden_states`
  because the residual entering every layer is None (`_postprocess_qwen4_exp_layer` returns `(hidden, None)`,
  `:1378-1385`; `_prepare_qwen4_exp_attn` `:1276-1306`). One sequence: a prefill of all parity tokens but the last 16,
  then 16 decode steps on those 16 tokens (teacher-forced), carrying the short-conv state ([10240, 9] per slot,
  `ShortConvPool`, `sglang/srt/mem_cache/ple_state_pool.py:42`) and the 2-token n-gram history (`NGramPool`, `:132`)
  between steps. Decode takes SGLang's decode fast path (fused hash, gate/value and short-conv-state kernels) when
  `SGLANG_ENABLE_QWEN4_PLE_FUSION` is on (its default); the report records which path ran.

Tolerance rule (block): `c` = rel-L2(SGLang op vs an FP64 torch reference of the same block on the same inputs: the
same token windows and query, the FP8 table rows times the BF16 scale, and every BF16 weight in FP64), written to
the report before the native call; limit = max(FLOOR_OP, 2c); measured = rel-L2(native vs SGLang op). Per step it
checks the block output and the carried conv state that way, and the n-gram history bit-exact. The FP64 reference
carries its own FP64 conv state, the native code its own buffers (state is never copied from SGLang into native).

Memory: the gather and the block read the whole table. Every real token produces 16 IDs, one per n-gram head, and
head h hashes into its own ~20M-row range (`ngram_heads_offsets`), i.e. about 8 of the 128 checkpoint shards each,
uniformly; any real window therefore touches rows across all 128 shards, and a partial table would turn real gathers
into zero rows. So the test loads all 128 shards (320,001,536 x 160 FP8 = 51,200,245,760 bytes = 47.68 GiB) once
per process into the one pinned allocation SGLang's own `Qwen4ExpPinnedHostEmbedding` makes, and the native entry
points read that same mapped allocation (zero-copy, as the executor's mapped pinned table). The pinned allocator's
power-of-two rounding is limited to <= 1 GiB first (`pinned_max_round_threshold_mb`), so the table pins 47.68 GiB,
not 64 GiB; the spec's host floor (MemAvailable >= 47.68 GiB + 16 GiB) is checked before allocating. The hash test
does not need the table and builds its own index-only module.

Native entry points (each returns 0 or a nonzero cere_status; the stream is the last argument; `table` is the host
pointer of the pinned table, device-accessible through UVA like SGLang's gather kernel uses it):

    int cere_ktest_ple_hash(const void *windows, const void *multipliers, const void *head_vocab_sizes,
                            const void *head_offsets, void *ids, int64_t n, int64_t eos_token_id,
                            cudaStream_t stream);
        windows int64 [n, 3] = tokens (t-2, t-1, t), EOS where the sequence has none; multipliers int64 [3]
        (`layer_multipliers`); head_vocab_sizes, head_offsets int64 [16]; ids int64 [n, 16] out. Same function as
        `_hash_contexts`: shift-1/-2 sources become EOS across an EOS at t-1 (shift 1) or t-1/t-2 (shift 2);
        2-gram heads 0-7 hash t*m0 ^ s1*m1, 3-gram heads 8-15 additionally ^ s2*m2 (int64 wrap-around
        multiply); id = floor-mod(mix, vocab_size[h]) + offset[h].

    int cere_ktest_ple_gather(const void *table, int64_t n_table_rows, const void *ids, int64_t n, void *out,
                              cudaStream_t stream);
        table FP8 E4M3 [n_table_rows, 160]; ids int64 [n]; out BF16 [n, 160] = the rows converted exactly, no
        scale; an id outside [0, n_table_rows) gives a zero row.

    int cere_ktest_ple_block(const void *query, const void *tokens, int64_t n_tokens, void *ngram_history,
                             void *conv_state, const void *table, int64_t n_table_rows, const void *table_scale,
                             const void *multipliers, const void *head_vocab_sizes, const void *head_offsets,
                             const void *key_proj, const void *value_proj, const void *norm_key,
                             const void *norm_query, const void *norm_conv, const void *conv_weight,
                             int64_t eos_token_id, float eps, void *out, cudaStream_t stream);
        One sequence step (prefill: n_tokens = prompt length; decode: 1). query BF16 [n_tokens, 10240] (the HC
        state); tokens int64 [n_tokens]; ngram_history int64 [2] in/out (the two previous tokens, EOS-initialised);
        conv_state BF16 [10240, 9] in/out (the last 9 short-conv inputs per channel, oldest first, zero-initialised;
        SGLang's `ShortConvPool` slot layout); table_scale BF16 [1]; key_proj BF16 [10240, 2560]; value_proj BF16
        [2560, 2560]; norm_* BF16 [10240] (Gemma `x * rsqrt(mean(x^2) + eps) * (1 + w)` per 2,560-wide group);
        conv_weight BF16 [10240, 4] (the checkpoint's [10240, 1, 4]); out BF16 [n_tokens, 10240] = the
        `Qwen4ExpPLELayer.forward` output (the PLE delta, before the decoder layer adds it to the hidden state).
        Block: emb = gather(hash(windows)) * scale; key = emb Wk^T; value = emb Wv^T; gate =
        sigmoid(sign(g) sqrt(max(|g|, 1e-6))) with g = <norm_key(key), norm_query(query)> / sqrt(2560) per stream;
        gv = gate * value; y = depthwise conv (kernel 4, dilation 3) over [state | norm_conv(gv)]; out = gv + silu(y).

    int cere_ktest_ple_block_planned(<the arguments of cere_ktest_ple_block up to and including out>,
                                     int64_t plan_max_m, int64_t workspace_tokens, cudaStream_t stream);
        The same step with the caller's capacities: the key/value K2 plans are pinned for max_m = plan_max_m, the
        workspace holds `k4::workspace_bytes(workspace_tokens)` bytes (0 = no workspace, a null pointer). A step the
        capacities cannot hold (n_tokens > plan_max_m, or a missing/short workspace) returns CERE_ERR_ARGUMENT (-1)
        before any launch: ngram_history, conv_state and out are left byte-identical (review M22a P3).

    int cere_ktest_ple_rows(const void *query, const void *tokens, int64_t m, int64_t sequences, const void *counts,
                            void *histories, void *conv_states, void *save_histories, void *save_convs,
                            const void *save_mask, <table .. eps as cere_ktest_ple_block>, int64_t add_query,
                            void *out, int64_t plan_max_m, int64_t mode, int64_t small_m, cudaStream_t stream);
        tasks/rows16-spec.md R16-A: m rows of `sequences` sequences back to back (host int64 counts), their states in
        histories [sequences, 2] / conv_states [sequences, 10240, 9] (in/out), row r's state after it into
        save_histories [m, 2] / save_convs [m, 10240, 9] where host int64 save_mask[r] is 1. mode 0: k4::forward
        (t = 1) per row on plans of max_m plan_max_m (the previous executor path); mode 1: one k4::forward_rows on their
        rows plans. small_m 1: both plans on the small-M kernel.
"""
from __future__ import annotations

import contextlib
import ctypes
import functools
import json
import math
import types
import unittest
from pathlib import Path

import torch
from torch import nn

import oracle
from sglang.srt.configs.mamba_utils import mamba2_state_dtype
from sglang.srt.environ import envs
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLELayer,
    _commit_ple_batch,
    _prepare_ple_batch,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils.hf_transformers import get_tokenizer

ROW = "K4"
PLE_LAYER = 1
PLE_PREFIX = f"model.language_model.layers.{PLE_LAYER}.ple."
SHARD_NAME = PLE_PREFIX + "ple_embedding.ngram_embedding.shard_{}.weight"
TABLE_PARAM = "ple_embedding.ngram_embedding.weight"
PARITY_PROMPTS = Path(__file__).resolve().parents[3] / "bench" / "prompts" / "parity_v1.jsonl"
RANDOM_WINDOWS = 100_000
EOS_RATE = 1.0 / 8.0
RANDOM_GATHER_IDS = 65_536
DECODE_STEPS = 16
# Per-stream query scales: the grouped PLE norms must normalise each 2,560-wide stream on its own.
STREAM_SCALES = (1.0, 0.5, 2.0, 4.0)
SEED_BASE = 4_000
CERE_ERR_ARGUMENT = -1  # cere_executor.h
# Bookkeeping for the one sequence: request-pool row 0 maps to state slot 1 (slot 0 is the pools' reserved dummy).
REQ_INDEX = 0
STATE_SLOT = 1
GIB = 1 << 30
HOST_HEADROOM = 16 * GIB
PINNED_ROUND_THRESHOLD_MB = 1024
EXPECTED_CONFIG = {
    "hidden_size": 2560,
    "hc_count": 4,
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ple_layer_ids": [2],
    "split_ngram_parts": 128,
    "eos_token_id": 248044,
    "rms_norm_eps": 1e-6,
    "ple_embedding_dtype": "float8_e4m3fn",
}


# -- config and SGLang module construction -------------------------------------------------------------------------


@functools.cache
def _config() -> types.SimpleNamespace:
    """The text config as attributes (what the PLE modules read), with the bench server's `--ple-offload-embedding`
    applied the way `load_model_utils.py:290` applies it."""
    cfg = oracle.text_config()
    actual = {k: cfg.get(k) for k in EXPECTED_CONFIG}
    if actual != EXPECTED_CONFIG:
        raise oracle.OracleError(f"text_config PLE fields {actual} != {EXPECTED_CONFIG}")
    if sorted(set(cfg["ple_layer_ids"]))[0] - 1 != PLE_LAYER:
        raise oracle.OracleError(f"ple_layer_ids {cfg['ple_layer_ids']} do not put the PLE on layer {PLE_LAYER}")
    return types.SimpleNamespace(**{**cfg, "ple_offload_embedding": True})


@contextlib.contextmanager
def _meta_build():
    """Construction on the meta device with BF16 default dtype at TP 1: every tensor the checkpoint provides is
    materialised afterwards from the checkpoint, and the pinned table (`device="cpu", pin_memory=True` is explicit in
    `Qwen4ExpPinnedHostEmbedding.__init__`) is the only real allocation, so the 51 GB table never lands on GPU0."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with get_parallel().override(tp_size=1, tp_rank=0), torch.device("meta"):
            yield
    finally:
        torch.set_default_dtype(previous)


def _materialise(module: nn.Module, prefix: str, exclude: set[str]) -> None:
    """Replace every meta parameter/buffer `name` of `module` by the checkpoint tensor `prefix + name` on GPU0."""
    tensors = [(n, t, True) for n, t in module.named_parameters()] + [(n, t, False) for n, t in module.named_buffers()]
    for name, meta, is_param in tensors:
        if name in exclude:
            continue
        real = oracle.real_tensor(prefix + name).contiguous()
        if tuple(real.shape) != tuple(meta.shape) or real.dtype != meta.dtype:
            raise oracle.OracleError(
                f"{prefix + name}: checkpoint {real.dtype} {tuple(real.shape)} != module {meta.dtype} {tuple(meta.shape)}"
            )
        owner_name, _, leaf = name.rpartition(".")
        owner = module.get_submodule(owner_name) if owner_name else module
        if is_param:
            owner._parameters[leaf] = nn.Parameter(real, requires_grad=False)
        else:
            owner._buffers[leaf] = real
    leftovers = [n for n, t in list(module.named_parameters()) + list(module.named_buffers())
                 if t.is_meta and n not in exclude]
    if leftovers:
        raise oracle.OracleError(f"tensors without checkpoint data: {leftovers}")


def _check_index(ngram: Qwen4ExpNGramEmbedding) -> None:
    """The checkpoint's hash index (`layer_multipliers`, head sizes/offsets) equals what SGLang derives from the config
    (splitmix64 multipliers, primes after `ngram_vocab_size_base`), and the table size is the padded head total."""
    multipliers = ngram._build_layer_multipliers(ngram.ngram_size)
    sizes, offsets, total = ngram._build_head_vocab_and_offsets()
    derived = {
        "layer_multipliers": multipliers.to(torch.long),
        "ngram_heads_vocab_sizes": torch.tensor(sizes, dtype=torch.long),
        "ngram_heads_offsets": torch.tensor(offsets, dtype=torch.long),
    }
    for name, value in derived.items():
        loaded = getattr(ngram, name).cpu()
        if not torch.equal(loaded, value):
            raise oracle.OracleError(f"checkpoint {name} differs from SGLang's derivation")
    divisor = ngram.make_ngram_vocab_size_divisible_by
    padded = (total + divisor - 1) // divisor * divisor
    if ngram.ngram_embedding.num_embeddings != padded:
        raise oracle.OracleError(f"table rows {ngram.ngram_embedding.num_embeddings} != padded head total {padded}")


@functools.cache
def _hash_module() -> Qwen4ExpNGramEmbedding:
    """Index-only `Qwen4ExpNGramEmbedding` of layer 1 (ple_layer_index 0): real hash buffers on GPU0, table on meta."""
    with _meta_build():
        ngram = Qwen4ExpNGramEmbedding(_config(), int(_config().ple_embed_dim), ple_layer_index=0, quant_config=None)
    _materialise(ngram, PLE_PREFIX + "ple_embedding.", exclude={"ngram_embedding.weight"})
    _check_index(ngram)
    return ngram


def _mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise oracle.OracleError("/proc/meminfo has no MemAvailable")


@functools.cache
def _ple_layer() -> Qwen4ExpPLELayer:
    """Layer 1's `Qwen4ExpPLELayer` with `--ple-offload-embedding`: all 128 shards in its pinned host table, every
    other tensor from the checkpoint on GPU0 (quant_config None = the NVFP4 config's `*.ple.*` exclusion)."""
    cfg = _config()
    divisor = int(cfg.make_ngram_vocab_size_divisible_by)
    # Table bytes from the checkpoint index alone (before allocating): rows x 160 FP8.
    with _meta_build():
        probe = Qwen4ExpNGramEmbedding(cfg, int(cfg.ple_embed_dim), ple_layer_index=0, quant_config=None)
    table_bytes = probe.ngram_embedding.num_embeddings * probe.head_dim_per_ngram
    if probe.ngram_embedding.num_embeddings % divisor:
        raise oracle.OracleError("table rows are not a multiple of make_ngram_vocab_size_divisible_by")
    available = _mem_available()
    if available < table_bytes + HOST_HEADROOM:
        raise oracle.OracleError(
            f"MemAvailable {available} B < table {table_bytes} B + {HOST_HEADROOM} B (spec host-memory floor)"
        )
    torch._C._accelerator_setAllocatorSettings(f"pinned_max_round_threshold_mb:{PINNED_ROUND_THRESHOLD_MB}")
    with _meta_build():
        layer = Qwen4ExpPLELayer(cfg, quant_config=None, prefix=PLE_PREFIX.rstrip("."), layer_id=PLE_LAYER,
                                 ple_layer_index=0)
    table_module = layer.ple_embedding.ngram_embedding
    if not isinstance(table_module, Qwen4ExpPinnedHostEmbedding):
        raise oracle.OracleError("the PLE table is not offloaded to pinned host memory")
    table = table_module.weight
    if table.device.type != "cpu" or not table.is_pinned() or table.dtype != torch.float8_e4m3fn:
        raise oracle.OracleError(f"PLE table is {table.dtype} on {table.device}, pinned={table.is_pinned()}")
    _materialise(layer, PLE_PREFIX, exclude={TABLE_PARAM})
    _check_index(layer.ple_embedding)
    # The shard loader `load_qwen4_exp_ple_shard` (`qwen4_exp.py:1863-1926`) at TP 1: shard k holds rows
    # [k * shard_size, k * shard_size + rows_k) with shard_size = ceil(rows / split_ngram_parts).
    parts = int(cfg.split_ngram_parts)
    rows = table_module.org_vocab_size
    shard_size = (rows + parts - 1) // parts
    covered = 0
    with torch.no_grad():
        for k in range(parts):
            shard = oracle.real_tensor(SHARD_NAME.format(k), device=None)
            if shard.dtype != torch.float8_e4m3fn or shard.shape[1] != table.shape[1]:
                raise oracle.OracleError(f"shard {k} is {shard.dtype} {tuple(shard.shape)}")
            start = k * shard_size
            if start != covered or start + shard.shape[0] > rows:
                raise oracle.OracleError(f"shard {k} rows [{start}, {start + shard.shape[0]}) break the table layout")
            table.data[start : start + shard.shape[0]].copy_(shard)
            covered = start + shard.shape[0]
            del shard
    if covered != rows:
        raise oracle.OracleError(f"shards cover {covered} of {rows} table rows")
    oracle.report_row(ROW, "table", phase="setup", rows=rows, bytes=table.numel(), mem_available=available)
    return layer


# -- SGLang runtime state -------------------------------------------------------------------------------------------


class _PLEStatePool:
    """The PLE half of `HybridReqToTokenPool` (`sglang/srt/mem_cache/memory_pool.py:1446-1497`) for one request:
    SGLang's own `ShortConvPool` and `NGramPool` with the same accessors, request row 0 -> state slot 1."""

    def __init__(self, cfg: types.SimpleNamespace):
        conv_dtype = mamba2_state_dtype().conv
        if conv_dtype != torch.bfloat16:
            raise oracle.OracleError(f"short-conv state dtype is {conv_dtype}; the native contract is BF16")
        state_len = (int(cfg.ple_conv_kernel_size) - 1) * int(cfg.ngram_size)
        channels = int(cfg.hidden_size) * int(cfg.hc_count)
        self.ple_window_cache = None
        self.short_conv_pool = ShortConvPool(
            size=STATE_SLOT, state_shape=(channels, state_len), layer_ids=[PLE_LAYER], dtype=conv_dtype,
            device=str(oracle.DEVICE),
        )
        self.ngram_pool = NGramPool(
            size=STATE_SLOT, context_len=int(cfg.ngram_size) - 1, eos_token_id=int(cfg.eos_token_id),
            device=str(oracle.DEVICE),
        )
        self.req_index_to_mamba_index_mapping = torch.full((REQ_INDEX + 1,), STATE_SLOT, dtype=torch.long,
                                                           device=oracle.DEVICE)

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def short_conv_layer_cache(self, layer_id: int) -> torch.Tensor:
        return self.short_conv_pool.layer_cache(layer_id)

    def short_conv_layer_intermediate_cache(self, layer_id: int):
        return self.short_conv_pool.layer_intermediate_cache(layer_id)

    def get_ngram_context(self, ngram_indices: torch.Tensor) -> torch.Tensor:
        return self.ngram_pool.get_context(ngram_indices)

    def set_ngram_context(self, ngram_indices: torch.Tensor, context: torch.Tensor) -> None:
        self.ngram_pool.set_context(ngram_indices, context)

    def set_ngram_intermediate_context(self, context: torch.Tensor) -> None:
        self.ngram_pool.set_intermediate_context(context)

    def conv_state(self) -> torch.Tensor:
        return self.short_conv_layer_cache(PLE_LAYER)[STATE_SLOT]

    def history(self) -> torch.Tensor:
        return self.ngram_pool.context[STATE_SLOT]


def _sglang_forward(pool: _PLEStatePool):
    """The forward context the PLE code reads its pool from (`get_req_to_token_pool()`, `forward_context.py:74`)."""
    return forward_context(ForwardContext(attn_backend=types.SimpleNamespace(req_to_token_pool=pool)))


def _forward_batch(tokens: torch.Tensor, prefill: bool) -> types.SimpleNamespace:
    """The `ForwardBatch` fields the PLE path reads (`_prepare_ple_batch`, `_ple_track_targets`), one request."""
    n = tokens.shape[0]
    return types.SimpleNamespace(
        input_ids=tokens,
        forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
        _original_forward_mode=None,
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=None,
        num_token_non_padded_cpu=None,
        extend_seq_lens=torch.tensor([n], dtype=torch.int32, device=oracle.DEVICE) if prefill else None,
        extend_seq_lens_cpu=[n] if prefill else None,
        req_pool_indices=torch.tensor([REQ_INDEX], dtype=torch.long, device=oracle.DEVICE),
        out_cache_loc=None if prefill else torch.ones(1, dtype=torch.long, device=oracle.DEVICE),
        mamba_track_indices=None,
        mamba_track_mask=None,
    )


# -- inputs ---------------------------------------------------------------------------------------------------------


@functools.cache
def _parity_prompt_ids() -> tuple[tuple[int, ...], ...]:
    """Each parity prompt rendered with the chat template (enable_thinking false), in file order; memory only."""
    tokenizer = get_tokenizer(str(oracle.MODEL_DIR))
    prompts = []
    for line in PARITY_PROMPTS.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(tuple(tokenizer.encode(text, add_special_tokens=False)))
    if len(prompts) != 24 or not all(prompts):
        raise oracle.OracleError(f"{PARITY_PROMPTS} gave {len(prompts)} prompts; expected 24 non-empty")
    return tuple(prompts)


def _windows(history: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """One int64 [3] window (t-2, t-1, t) per token of `tokens` following the 2-token `history`."""
    return torch.cat([history, tokens]).unfold(0, 3, 1).contiguous()


def _parity_windows(eos: int) -> torch.Tensor:
    """Every token window of every parity prompt, each prompt a fresh request (history = two EOS)."""
    fresh = torch.full((2,), eos, dtype=torch.long, device=oracle.DEVICE)
    parts = [_windows(fresh, torch.tensor(ids, dtype=torch.long, device=oracle.DEVICE)) for ids in _parity_prompt_ids()]
    return torch.cat(parts).contiguous()


def _random_windows(vocab: int, eos: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    windows = torch.randint(0, vocab, (RANDOM_WINDOWS, 3), generator=g, dtype=torch.long, device=oracle.DEVICE)
    eos_mask = torch.rand((RANDOM_WINDOWS, 3), generator=g, device=oracle.DEVICE) < EOS_RATE
    return torch.where(eos_mask, torch.full_like(windows, eos), windows).contiguous()


def _query(n_tokens: int, hidden: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    x = torch.randn((n_tokens, len(STREAM_SCALES), hidden), generator=g, dtype=torch.float32, device=oracle.DEVICE)
    scales = torch.tensor(STREAM_SCALES, dtype=torch.float32, device=oracle.DEVICE).view(1, -1, 1)
    return (x * scales).reshape(n_tokens, -1).to(torch.bfloat16).contiguous()


# -- FP64 reference of the block ------------------------------------------------------------------------------------


def _weights64(layer: Qwen4ExpPLELayer) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        key=layer.key_proj.weight.to(torch.float64),
        value=layer.value_proj.weight.to(torch.float64),
        norm_key=layer.norm_key.weight.to(torch.float64),
        norm_query=layer.norm_query.weight.to(torch.float64),
        norm_conv=layer.norm_conv.weight.to(torch.float64),
        conv=layer.conv1d.weight.to(torch.float64).reshape(layer.conv_channels, layer.conv_kernel_size),
        scale=float(layer.ple_embedding.ngram_embedding.weight_scale.to(torch.float64).item()),
    )


def _norm64(x: torch.Tensor, weight: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    xs = x.unflatten(-1, (-1, group))
    y = xs * torch.rsqrt(xs.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y.flatten(-2) * (1.0 + weight)


def _table_rows64(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """FP8 table rows for `ids` (any shape) in FP64 on GPU0 (byte gather on the host, exact conversion)."""
    flat = ids.reshape(-1).cpu()
    rows = table.view(torch.uint8).index_select(0, flat).to(oracle.DEVICE)
    return rows.view(torch.float8_e4m3fn).to(torch.float32).to(torch.float64).reshape(*ids.shape, table.shape[1])


def _reference_step(layer: Qwen4ExpPLELayer, w: types.SimpleNamespace, ids: torch.Tensor, query: torch.Tensor,
                    state: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 block for one step: returns (out [n, 10240], next state [10240, 9])."""
    n = query.shape[0]
    hidden, hc = layer.hidden_size, layer.hc_count
    emb = (_table_rows64(layer.ple_embedding.ngram_embedding.weight, ids) * w.scale).reshape(n, -1)
    key = emb @ w.key.T
    value = emb @ w.value.T
    key_n = _norm64(key, w.norm_key, hidden, eps).reshape(n, hc, hidden)
    query_n = _norm64(query.to(torch.float64), w.norm_query, hidden, eps).reshape(n, hc, hidden)
    g = (key_n * query_n).sum(dim=-1, keepdim=True) / math.sqrt(hidden)
    gate = torch.sigmoid(g.abs().clamp_min(1e-6).sqrt() * g.sign())
    gated = (gate * value.unsqueeze(-2)).reshape(n, -1)
    gated_n = _norm64(gated, w.norm_conv, hidden, eps)
    seq = torch.cat([state, gated_n.T], dim=1)
    dilation = layer.short_conv_dilation
    conv = sum(w.conv[:, k : k + 1] * seq[:, k * dilation : k * dilation + n] for k in range(layer.conv_kernel_size))
    out = gated + torch.nn.functional.silu(conv.T)
    return out, seq[:, n:].contiguous()


def _planned_block(inputs: types.SimpleNamespace, history: torch.Tensor, state: torch.Tensor, out: torch.Tensor,
                   eos: int, eps: float, plan_max_m: int, workspace_tokens: int) -> int:
    """`cere_ktest_ple_block_planned` on the current stream; returns its status (a rejection is the expected result
    here, so this does not go through `oracle.call`, which raises on any nonzero status)."""
    fn = oracle.ktest("cere_ktest_ple_block_planned")
    args = (inputs.query, inputs.tokens, inputs.tokens.shape[0], history, state, inputs.table, inputs.rows,
            inputs.scale, inputs.multipliers, inputs.vocab_sizes, inputs.offsets, inputs.key, inputs.value,
            inputs.norm_key, inputs.norm_query, inputs.norm_conv, inputs.conv, eos, eps, out, plan_max_m,
            workspace_tokens)
    stream = torch.cuda.current_stream(oracle.DEVICE)
    status = fn(*[oracle._arg(a) for a in args], ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(status)


# -- suite ----------------------------------------------------------------------------------------------------------


class K4PLE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # SGLang modules read the published config at construction/dispatch; publish a dummy-boundary one
        # (`runtime_context.py:968`, the sanctioned test entry).
        cls.sglang_config = get_context().override_server_args()
        cls.sglang_config.install()
        cfg = _config()
        cls.eos = int(cfg.eos_token_id)
        cls.vocab = int(cfg.vocab_size)
        cls.eps = float(cfg.rms_norm_eps)
        cls.hidden = int(cfg.hidden_size)
        cls.fusion = bool(envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get())

    @classmethod
    def tearDownClass(cls):
        cls.sglang_config.restore()

    # -- hash -----------------------------------------------------------------------------------------------------

    def _check_hash(self, case: str, windows: torch.Tensor) -> torch.Tensor:
        ngram = _hash_module()
        pool = _PLEStatePool(_config())
        with torch.no_grad(), _sglang_forward(pool):
            eager = ngram._hash_contexts(windows, decode_sized=False)
            routed = ngram._hash_contexts(windows, decode_sized=True)
        n = windows.shape[0]
        native = oracle.empty((n, ngram.ngram_heads), torch.long)
        oracle.call(
            "cere_ktest_ple_hash",
            windows,
            ngram.layer_multipliers,
            ngram.ngram_heads_vocab_sizes,
            ngram.ngram_heads_offsets,
            native,
            n,
            self.eos,
        )
        oracle.report_row(ROW, case, phase="setup", windows=n, fused_hash=self.fusion,
                          eos_positions=int((windows == self.eos).sum().item()))
        oracle.check_bit_exact(self, ROW, f"{case}_vs_hash_contexts", native, eager)
        oracle.check_bit_exact(self, ROW, f"{case}_vs_decode_sized_hash", native, routed)
        return eager

    def test_hash_random_windows(self):
        self._check_hash("hash_random", _random_windows(self.vocab, self.eos, SEED_BASE + 1))

    def test_hash_parity_windows(self):
        self._check_hash("hash_parity", _parity_windows(self.eos))

    # -- gather ---------------------------------------------------------------------------------------------------

    def test_gather_bit_exact(self):
        layer = _ple_layer()
        table_module = layer.ple_embedding.ngram_embedding
        table = table_module.weight
        rows = table_module.org_vocab_size
        pool = _PLEStatePool(_config())
        with torch.no_grad(), _sglang_forward(pool):
            parity_ids = layer.ple_embedding._hash_contexts(_parity_windows(self.eos)).reshape(-1)
        g = oracle.generator(SEED_BASE + 2)
        random_ids = torch.randint(0, rows, (RANDOM_GATHER_IDS,), generator=g, dtype=torch.long, device=oracle.DEVICE)
        parts = int(_config().split_ngram_parts)
        shard_size = (rows + parts - 1) // parts
        boundary = sorted({b for k in range(1, parts) for b in (k * shard_size - 1, k * shard_size)} | {0, rows - 1})
        out_of_range = [-1, rows, rows + 1, -(1 << 40), 1 << 40]
        cases = {
            "gather_parity": parity_ids,
            "gather_random": random_ids,
            "gather_boundaries": torch.tensor(boundary, dtype=torch.long, device=oracle.DEVICE),
            "gather_out_of_range": torch.tensor(out_of_range, dtype=torch.long, device=oracle.DEVICE),
        }
        for case, ids in cases.items():
            with self.subTest(case=case):
                ids = ids.contiguous()
                with torch.no_grad():
                    reference = table_module.gather(ids)
                torch.cuda.synchronize(oracle.DEVICE)
                native = oracle.empty((ids.numel(), table_module.embedding_dim), torch.bfloat16)
                oracle.call("cere_ktest_ple_gather", table, rows, ids, ids.numel(), native)
                oracle.report_row(ROW, case, phase="setup", n_ids=ids.numel())
                oracle.check_bit_exact(self, ROW, case, native, reference)

    # -- full block -----------------------------------------------------------------------------------------------

    def test_block_prefill_then_decode(self):
        layer = _ple_layer()
        ngram = layer.ple_embedding
        table_module = ngram.ngram_embedding
        w = _weights64(layer)
        stream = [t for ids in _parity_prompt_ids() for t in ids]
        if len(stream) <= DECODE_STEPS:
            raise oracle.OracleError(f"parity stream has {len(stream)} tokens; need more than {DECODE_STEPS}")
        all_tokens = torch.tensor(stream, dtype=torch.long, device=oracle.DEVICE)
        prefill_len = len(stream) - DECODE_STEPS
        steps = [("block_prefill", all_tokens[:prefill_len])]
        steps += [(f"block_decode_{i:02d}", all_tokens[prefill_len + i : prefill_len + i + 1])
                  for i in range(DECODE_STEPS)]
        oracle.report_row(ROW, "block", phase="setup", prefill_tokens=prefill_len, decode_steps=DECODE_STEPS,
                          decode_fast_path=self.fusion)

        pool = _PLEStatePool(_config())
        ref_history = torch.full((2,), self.eos, dtype=torch.long, device=oracle.DEVICE)
        ref_state = torch.zeros(layer.conv_channels, layer.short_conv_state_len, dtype=torch.float64,
                                device=oracle.DEVICE)
        native_history = torch.full((2,), self.eos, dtype=torch.long, device=oracle.DEVICE)
        native_state = torch.zeros(layer.conv_channels, layer.short_conv_state_len, dtype=torch.bfloat16,
                                   device=oracle.DEVICE)
        for index, (case, tokens) in enumerate(steps):
            prefill = index == 0
            n = tokens.shape[0]
            query = _query(n, self.hidden, SEED_BASE + 100 + index)
            batch_info = _forward_batch(tokens, prefill)
            with torch.no_grad(), _sglang_forward(pool):
                # As `Qwen4ExpModel.forward` runs it: batch prep, prefetch from the pinned table, the layer, commit.
                batch = _prepare_ple_batch(tokens, batch_info, ngram_size=ngram.ngram_size,
                                           ngram_eos_token_id=self.eos)
                layer.start_prefetch(batch, batch_info)
                sglang_out = layer(query, batch_info, batch)
                _commit_ple_batch(batch, batch_info)
                ref_ids = ngram._hash_contexts(_windows(ref_history, tokens))
            torch.cuda.synchronize(oracle.DEVICE)
            sglang_state = pool.conv_state().clone()
            sglang_history = pool.history().clone()
            reference, ref_state = _reference_step(layer, w, ref_ids, query, ref_state, self.eps)
            ref_history = torch.cat([ref_history, tokens])[-2:]

            c_out = oracle.calibrate(ROW, f"{case}_out", sglang_out, reference, oracle.FLOOR_OP)
            c_state = oracle.calibrate(ROW, f"{case}_conv_state", sglang_state, ref_state, oracle.FLOOR_OP)
            native_out = oracle.empty((n, layer.conv_channels), torch.bfloat16)
            oracle.call(
                "cere_ktest_ple_block",
                query,
                tokens,
                n,
                native_history,
                native_state,
                table_module.weight,
                table_module.org_vocab_size,
                table_module.weight_scale,
                ngram.layer_multipliers,
                ngram.ngram_heads_vocab_sizes,
                ngram.ngram_heads_offsets,
                layer.key_proj.weight,
                layer.value_proj.weight,
                layer.norm_key.weight,
                layer.norm_query.weight,
                layer.norm_conv.weight,
                layer.conv1d.weight,
                self.eos,
                self.eps,
                native_out,
            )
            limit_out = oracle.calibrated_limit(c_out, oracle.FLOOR_OP)
            oracle.check_within(self, ROW, f"{case}_out", oracle.rel_l2(native_out, sglang_out), c=c_out,
                                limit=limit_out)
            limit_state = oracle.calibrated_limit(c_state, oracle.FLOOR_OP)
            oracle.check_within(self, ROW, f"{case}_conv_state", oracle.rel_l2(native_state, sglang_state),
                                c=c_state, limit=limit_state)
            oracle.check_bit_exact(self, ROW, f"{case}_ngram_history", native_history, sglang_history)

    # -- capacity validation (review M22a P3) ---------------------------------------------------------------------

    def test_block_rejects_oversized_step_before_any_launch(self):
        """A step the plans/workspace cannot hold returns CERE_ERR_ARGUMENT and leaves history, conv state and out
        byte-identical; the same step with enough capacity runs. Synthetic inputs (no table pinning needed): a
        1,024-row FP8 table with 16 heads of 64 rows each, small random BF16 weights."""
        n = 4
        channels = self.hidden * len(STREAM_SCALES)
        g = oracle.generator(SEED_BASE + 7_000)
        inputs = _synthetic_block_inputs(g, self.hidden, self.vocab, n, SEED_BASE + 7_001)
        dev = oracle.DEVICE
        history = torch.full((2,), self.eos, dtype=torch.long, device=dev)
        state = torch.randn((channels, 9), generator=g, device=dev).to(torch.bfloat16)
        out = oracle.empty((n, channels), torch.bfloat16)
        # (case, plan_max_m, workspace_tokens): the first rejection must come first so that a build without the
        # up-front checks fails on the state comparison before a null/short workspace is ever written.
        rejected = (("reject_plan_max_m", n - 1, n), ("reject_no_workspace", n, 0), ("reject_short_workspace", n, n - 1))
        for case, plan_max_m, workspace_tokens in rejected:
            before = (history.clone(), state.clone(), out.clone())
            status = _planned_block(inputs, history, state, out, self.eos, self.eps, plan_max_m, workspace_tokens)
            oracle.check_count(self, ROW, f"{case}_status", status, expected=CERE_ERR_ARGUMENT)
            oracle.check_bit_exact(self, ROW, f"{case}_ngram_history_unchanged", history, before[0])
            oracle.check_bit_exact(self, ROW, f"{case}_conv_state_unchanged", state, before[1])
            oracle.check_bit_exact(self, ROW, f"{case}_out_unchanged", out, before[2])
        status = _planned_block(inputs, history, state, out, self.eos, self.eps, n, n)
        oracle.check_count(self, ROW, "fits_status", status, expected=0)
        oracle.check_bit_exact(self, ROW, "fits_ngram_history", history, inputs.tokens[-2:].contiguous())
        oracle.check_count(self, ROW, "fits_out_nonfinite", int((~torch.isfinite(out)).sum().item()), expected=0)

    def test_block_rows_bit_exact(self):
        """tasks/rows16-spec.md R16-A: k4::forward_rows (one call over several sequences on the rows plans) vs the
        executor's previous path, k4::forward(t = 1) once per row in sequence order on the single-row plans with a D2D
        copy of the state after each saved row: out, every sequence's history and conv state, and every save,
        bit-exact, for the verify call's shapes (r4 chains of 4, r1 chain of 4, DraftAccept chains up to 9, decode
        rows of one token each), with the query added (the model path) and without, cuBLASLt and small-M plans.
        Synthetic inputs (EOS injected into the tokens and histories)."""
        channels = self.hidden * len(STREAM_SCALES)
        dev = oracle.DEVICE
        cases = (("r4_chains", (4, 4, 4, 4)), ("r1_chain", (4,)), ("decode_rows", (1, 1, 1, 1)),
                 ("mixed_chains", (9, 3, 1, 3)), ("one_chain_16", (16,)), ("rows_16", (1,) * 16))
        for index, (case, counts) in enumerate(cases):
            m, sequences = sum(counts), len(counts)
            g = oracle.generator(SEED_BASE + 8_000 + index)
            inputs = _synthetic_block_inputs(g, self.hidden, self.vocab, m, SEED_BASE + 8_100 + index)
            eos_tokens = torch.rand((m,), generator=g, device=dev) < EOS_RATE
            tokens = torch.where(eos_tokens, torch.full_like(inputs.tokens, self.eos), inputs.tokens)
            histories = torch.randint(0, self.vocab, (sequences, 2), generator=g, dtype=torch.long, device=dev)
            histories[0, 0] = self.eos
            conv_states = torch.randn((sequences, channels, 9), generator=g, device=dev).to(torch.bfloat16)
            # Row j of a chain saves when j < count - 1 (the verify undo snapshots).
            mask = [1 if j + 1 < c else 0 for c in counts for j in range(c)]
            save_mask = torch.tensor(mask, dtype=torch.long)
            counts_host = torch.tensor(counts, dtype=torch.long)
            for add_query in (True, False):
                for small_m in (0, 1):
                    results = []
                    for mode in (0, 1):
                        hist = histories.clone()
                        conv = conv_states.clone()
                        save_hist = torch.full((m, 2), -7, dtype=torch.long, device=dev)
                        save_conv = oracle.empty((m, channels, 9), torch.bfloat16)
                        out = oracle.empty((m, channels), torch.bfloat16)
                        oracle.call("cere_ktest_ple_rows", inputs.query, tokens, m, sequences, counts_host, hist,
                                    conv, save_hist, save_conv, save_mask, inputs.table, inputs.rows, inputs.scale,
                                    inputs.multipliers, inputs.vocab_sizes, inputs.offsets, inputs.key, inputs.value,
                                    inputs.norm_key, inputs.norm_query, inputs.norm_conv, inputs.conv, self.eos,
                                    self.eps, add_query, out, 4096, mode, small_m)
                        saved = save_mask.to(dev).bool()
                        results.append((out, hist, conv, save_hist[saved], save_conv[saved]))
                    label = f"rows_{case}_q{int(add_query)}_s{small_m}"
                    names = ("out", "history", "conv_state", "save_history", "save_conv")
                    for name, native, reference in zip(names, results[1], results[0]):
                        if native.numel() == 0:  # rows of one token each: nothing saved
                            oracle.check_count(self, ROW, f"{label}_{name}_rows", reference.numel(), expected=0)
                            continue
                        oracle.check_bit_exact(self, ROW, f"{label}_{name}", native, reference)
                    oracle.check_count(self, ROW, f"{label}_out_nonfinite",
                                       int((~torch.isfinite(results[1][0])).sum().item()), expected=0)


def _synthetic_block_inputs(g: torch.Generator, hidden: int, vocab: int, n: int, query_seed: int):
    """Synthetic block inputs (no table pinning needed): a 1,024-row FP8 table with 16 heads of 64 rows each, small
    random BF16 weights, n random tokens and a seeded query."""
    heads, head_rows = 16, 64
    channels = hidden * len(STREAM_SCALES)
    dev = oracle.DEVICE
    table = torch.randint(0, 256, (heads * head_rows, 160), generator=g, dtype=torch.uint8, device=dev)
    table = torch.where((table & 0x7F) == 0x7F, torch.zeros_like(table), table)  # no FP8 NaN encodings
    return types.SimpleNamespace(
        query=_query(n, hidden, query_seed),
        tokens=torch.randint(0, vocab, (n,), generator=g, dtype=torch.long, device=dev),
        table=table.view(torch.float8_e4m3fn).contiguous(),
        rows=heads * head_rows,
        scale=torch.ones(1, dtype=torch.bfloat16, device=dev),
        multipliers=torch.randint(1, 1 << 62, (3,), generator=g, dtype=torch.long, device=dev),
        vocab_sizes=torch.full((heads,), head_rows, dtype=torch.long, device=dev),
        offsets=torch.arange(heads, dtype=torch.long, device=dev) * head_rows,
        key=(torch.randn((channels, hidden), generator=g, device=dev) * 0.02).to(torch.bfloat16),
        value=(torch.randn((hidden, hidden), generator=g, device=dev) * 0.02).to(torch.bfloat16),
        norm_key=(torch.randn((channels,), generator=g, device=dev) * 0.1).to(torch.bfloat16),
        norm_query=(torch.randn((channels,), generator=g, device=dev) * 0.1).to(torch.bfloat16),
        norm_conv=(torch.randn((channels,), generator=g, device=dev) * 0.1).to(torch.bfloat16),
        conv=(torch.randn((channels, 4), generator=g, device=dev) * 0.5).to(torch.bfloat16),
    )


if __name__ == "__main__":
    unittest.main()
