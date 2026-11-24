"""SelfTune ST4 S3c: family sdar's kernels (executor/cuda/src/kernels/sdar/sdar_kernels.h; DICE-8B, block diffusion
over Qwen3-8B dense layers) vs torch FP32 oracles of the reference forward (DICE modeling_sdar.py at the S2 pin).

Suite `sdar_kernels` (GPU0-only, scripts/live/native_dev.py GPU0_ONLY_SUITES): `CERE_LIVE=1 python3
scripts/live/native_dev.py test --suite sdar_kernels` runs it against the dice-8b library (SUITE_MODELS: --build-dir
defaults to build/native-dice-8b, built by `native_dev.py build --model dice-8b`). Synthetic inputs only: /model (the
DICE-8B checkpoint, read by test_sdar_kernels_model.py's replay) is never read here.

Every case reports row "SDAR". Tolerance cases follow the tolerance rule: `c` = rel-L2(torch op in FP32 with the
reference's BF16 rounding points vs an FP64 torch reference on the same inputs), reported before the native call;
limit = max(4e-3, 2c); measured = rel-L2(native vs the torch op). Embedding, KV store, the residual add, top_tie and
unmask_select are bit-exact.

Cases:
- attention: GQA 32/8, head_dim 128, block-causal mask (key j visible to the query at p iff j // 4 <= p // 4), past in
  {0, 8, 52} x n in {1, 3, 4, 8, 64}, plus a long prefill (n 4,096 over 32,768 keys, the oracle on its first and last
  64 rows); each native case runs twice, bit-identical. Fail-first: the Mask::kCausal mutant (query p sees j <= p)
  fails the block-causal oracle check at every n > 1.
- tail slot vs scratch: attention reading the new K/V from the cache tail (after kv_store) equals, bit for bit, the
  same call reading them from a separate scratch buffer (strided, as in a fused projection) while the cache tail holds
  NaN.
- kv_store bit-exact (aligned and misaligned sources); embed bit-exact (vectorized and scalar widths); add bit-exact
  vs torch's BF16 add (FP32 add, one RNE rounding; aliased y = x too); silu_mul vs F.silu(gate) * up on BF16 tensors;
  rmsnorm per row (strided, 4,096 wide) and per head (q [m, 32, 128] / k [m, 8, 128] at stride 128, in place too) vs
  flash_attn's rms_norm_fn arithmetic; rope vs the HF formula (FP32 inv_freq and angle, cos/sin cast to BF16, each
  product and the sum rounded to BF16).
- top_tie: random rows (stride > vocab too); explicit ties (lowest id, the second-lowest as runner-up); condition 1:
  two FP32 logits that differ but round to one BF16 value tie (ties 2, the lowest id wins), and the FP32-comparing
  mutant fails that case; a highest-id tie pick is detected (differs from the native result).
- unmask_select: fewest ties wins, the lowest index among equals, unmasked rows skipped, -1 when none is masked (and
  at rows 0); a highest-index pick and a mask-ignoring pick are each detected; chained after top_tie.
- argument refusals (CERE_ERR_ARGUMENT).

Entry points (libcere_ktest.so, executor/cuda/src/ktest/sdar/sdar_ktest.cu; device pointers, int64 sizes, the stream
last; 0 on success):
    int cere_ktest_sdar_embed(const void *table, const void *ids /*uint32 [n]*/, int64_t n, int64_t hidden,
                              void *out, cudaStream_t stream);
    int cere_ktest_sdar_rmsnorm(const void *x, int64_t x_stride, const void *w, float eps, int64_t rows, int64_t dim,
                                void *y, int64_t y_stride, cudaStream_t stream);
    int cere_ktest_sdar_rope(void *x, int64_t stride, int64_t rows, int64_t heads, int64_t pos0, float theta,
                             cudaStream_t stream);
    int cere_ktest_sdar_kv_store(const void *k, int64_t k_stride, const void *v, int64_t v_stride, int64_t n,
                                 int64_t kv_heads, void *k_cache, void *v_cache, int64_t pos, cudaStream_t stream);
    int cere_ktest_sdar_attention(const void *q, int64_t q_stride, const void *k_cache, const void *v_cache,
                                  int64_t past, const void *new_k, int64_t new_k_stride, const void *new_v,
                                  int64_t new_v_stride, int64_t n, int64_t heads, int64_t kv_heads, void *out,
                                  int64_t out_stride, int64_t mask /*0 block-causal, 1 causal mutant*/,
                                  cudaStream_t stream);
    int cere_ktest_sdar_silu_mul(const void *gate_up, int64_t rows, int64_t inter, void *out, cudaStream_t stream);
    int cere_ktest_sdar_add(void *x, const void *y, int64_t n, cudaStream_t stream);
    int cere_ktest_sdar_top_tie(const void *logits, int64_t stride, int64_t rows, int64_t vocab,
                                void *out /*TopTie [rows]: uint32 id, ties, runner_up; float logit, runner_up_logit*/,
                                int64_t fp32 /*1: the condition-1 mutant*/, cudaStream_t stream);
    int cere_ktest_sdar_unmask_select(const void *top /*TopTie [rows]*/, const void *masked /*uint8 [rows]*/,
                                      int64_t rows, void *index /*int32 [1]*/, cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

import oracle

ROW = "SDAR"
HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
Q_ROW, KV_ROW = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
QKV_ROW = Q_ROW + 2 * KV_ROW  # a fused projection row: the strided K/V source of the scratch and kv_store cases
HIDDEN, INTER, VOCAB = 4096, 12288, 151936
EPS = 1e-6  # config.json rms_norm_eps
THETA = 1_000_000  # config.json rope_theta
BLOCK = 4  # config.json block_size
PASTS = (0, 8, 52)
NEWS = (1, 3, 4, 8, 64)
LONG_PAST, LONG_N, LONG_EDGE = 28_672, 4_096, 64  # 32,768 keys (max_position_embeddings)
MASK_BLOCK_CAUSAL, MASK_CAUSAL = 0, 1
NONE_ID = 0xFFFFFFFF


def _randn(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = oracle.generator(seed)
    return (torch.randn(shape, generator=g, device=oracle.DEVICE, dtype=torch.float32) * scale).to(torch.bfloat16)


# -- attention --------------------------------------------------------------------------------------------------


def _visible(positions: torch.Tensor, keys: int, causal: bool) -> torch.Tensor:
    """[m, keys] bool: key j visible to the query at absolute position p."""
    p = positions[:, None]
    j = torch.arange(keys, device=positions.device)[None, :]
    return j <= p if causal else (j // BLOCK) <= (p // BLOCK)


def _attention_ref(q: torch.Tensor, positions: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
                   dtype: torch.dtype) -> torch.Tensor:
    """q [m, HEADS, HEAD_DIM] at `positions`, k/v [keys, KV_HEADS, HEAD_DIM] -> [m, Q_ROW] in `dtype`: the reference's
    scaled_dot_product_attention (enable_gqa, scale head_dim^-0.5, additive mask) with scores, softmax and P V in
    `dtype` (FP32: the torch op; FP64: calibration)."""
    m, keys = q.shape[0], k.shape[0]
    group = HEADS // KV_HEADS
    mask = _visible(positions, keys, causal)
    out = torch.empty((m, HEADS, HEAD_DIM), dtype=dtype, device=q.device)
    for g in range(KV_HEADS):
        kg = k[:, g].to(dtype)
        vg = v[:, g].to(dtype)
        qg = q[:, g * group:(g + 1) * group].to(dtype).transpose(0, 1)  # [group, m, D]
        scores = (qg @ kg.T) * HEAD_DIM ** -0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        out[:, g * group:(g + 1) * group] = (torch.softmax(scores, dim=-1) @ vg).transpose(0, 1)
    return out.reshape(m, Q_ROW)


def _native_attention(q, q_stride: int, k_cache, v_cache, past: int, new_k, new_v, new_stride: int, n: int,
                      mask: int) -> torch.Tensor:
    out = oracle.empty((n, Q_ROW), torch.bfloat16)
    oracle.call("cere_ktest_sdar_attention", q, q_stride, k_cache, v_cache, past, new_k, new_stride, new_v, new_stride,
                n, HEADS, KV_HEADS, out, Q_ROW, mask)
    return out


# -- rmsnorm, rope, silu ------------------------------------------------------------------------------------------


def _rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """flash_attn rms_norm_fn's arithmetic: var = mean(x^2), rstd = 1 / sqrt(var + eps), y = x * rstd * w (FP32 for
    the torch op, rounded to BF16 once; FP64 for calibration)."""
    xf = x.to(dtype)
    rstd = 1.0 / torch.sqrt(xf.square().mean(dim=-1, keepdim=True) + EPS)
    y = xf * rstd * w.to(dtype)
    return y.to(torch.bfloat16) if dtype == torch.float32 else y


def _rope_hf(x: torch.Tensor, pos0: int) -> torch.Tensor:
    """SDARRotaryEmbedding + apply_rotary_pos_emb on x [m, heads, HEAD_DIM] BF16: inv_freq on the CPU in FP32, the
    angle as an FP32 matmul, cos/sin cast to BF16, BF16 products and sum."""
    m = x.shape[0]
    inv_freq = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).to(dtype=torch.float) / HEAD_DIM))
    positions = torch.arange(pos0, pos0 + m, device=x.device)[None, None, :].float()
    freqs = (inv_freq.to(x.device)[None, :, None].float() @ positions).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)[0][:, None, :]
    sin = emb.sin().to(x.dtype)[0][:, None, :]
    rot = torch.cat((-x[..., HEAD_DIM // 2:], x[..., :HEAD_DIM // 2]), dim=-1)
    return (x * cos) + (rot * sin)


def _rope_fp64(x: torch.Tensor, pos0: int) -> torch.Tensor:
    m = x.shape[0]
    inv_freq = 1.0 / (float(THETA) ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device=x.device) / HEAD_DIM))
    angle = torch.arange(pos0, pos0 + m, dtype=torch.float64, device=x.device)[:, None] * inv_freq[None, :]
    emb = torch.cat((angle, angle), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    x64 = x.to(torch.float64)
    rot = torch.cat((-x64[..., HEAD_DIM // 2:], x64[..., :HEAD_DIM // 2]), dim=-1)
    return x64 * cos + rot * sin


# -- top_tie and unmask_select ------------------------------------------------------------------------------------


def _top_tie_row(values: torch.Tensor, *, highest: bool) -> torch.Tensor:
    """One TopTie as int32 [5] from the compared values (BF16-rounded logits as FP32). highest: the wrong-pick mutant
    (the highest id holding the maximum, the next-highest as runner-up)."""
    top = values.max()
    holders = (values == top).nonzero().flatten().tolist()
    ties = len(holders)
    pick = holders[-1] if highest else holders[0]
    if ties > 1:
        runner_up, runner_up_logit = (holders[-2] if highest else holders[1]), top
    else:
        below = values.masked_fill(values == top, float("-inf"))
        second = below.max()
        runner_up = int((below == second).nonzero().flatten()[0]) if values.numel() > 1 else NONE_ID
        runner_up_logit = second
    ids = torch.tensor([pick, ties, runner_up], dtype=torch.int64, device=values.device)
    ids = torch.where(ids >= 2 ** 31, ids - 2 ** 32, ids).to(torch.int32)
    floats = torch.stack([top, runner_up_logit]).to(torch.float32).view(torch.int32)
    return torch.cat([ids, floats])


def _top_tie_ref(logits: torch.Tensor, vocab: int, *, highest: bool = False, fp32: bool = False) -> torch.Tensor:
    """int32 [rows, 5]: the contract (every logit rounded to BF16 first), or with fp32 the raw FP32 comparison."""
    rows = []
    for r in range(logits.shape[0]):
        x = logits[r, :vocab]
        rows.append(_top_tie_row(x if fp32 else x.to(torch.bfloat16).float(), highest=highest))
    return torch.stack(rows)


def _native_top_tie(logits: torch.Tensor, stride: int, vocab: int, fp32: int = 0) -> torch.Tensor:
    out = oracle.empty((logits.shape[0], 5), torch.int32)
    oracle.call("cere_ktest_sdar_top_tie", logits, stride, logits.shape[0], vocab, out, fp32)
    return out


def _select(ties: list[int], masked: list[int], *, lowest: bool = True, honor_mask: bool = True) -> int:
    """low_confidence_static, top_k 1: the masked row with the fewest ties, the lowest index among equals (lowest
    False: the highest-index mutant; honor_mask False: the mask-ignoring mutant); -1 when none."""
    rows = [r for r in range(len(ties)) if masked[r] or not honor_mask]
    if not rows:
        return -1
    fewest = min(ties[r] for r in rows)
    equal = [r for r in rows if ties[r] == fewest]
    return equal[0] if lowest else equal[-1]


def _native_select(top: torch.Tensor, masked: torch.Tensor, rows: int) -> int:
    index = oracle.empty((1,), torch.int32)
    oracle.call("cere_ktest_sdar_unmask_select", top, masked, rows, index)
    return int(index.item())


class SdarKernels(unittest.TestCase):

    # -- attention ----------------------------------------------------------------------------------------------

    def _attention_inputs(self, past: int, n: int, seed: int):
        q = _randn((n, HEADS, HEAD_DIM), seed)
        k = _randn((past + n, KV_HEADS, HEAD_DIM), seed + 1)
        v = _randn((past + n, KV_HEADS, HEAD_DIM), seed + 2)
        return q, k, v

    def test_attention_block_causal(self):
        """Block-causal GQA vs the FP32 oracle over past x n; the kCausal mutant fails the same check (fail-first)."""
        for past in PASTS:
            for n in NEWS:
                with self.subTest(past=past, n=n):
                    case = f"attention past {past} n {n}"
                    q, k, v = self._attention_inputs(past, n, 1000 + 100 * past + n)
                    positions = torch.arange(past, past + n, device=oracle.DEVICE)
                    reference = _attention_ref(q, positions, k, v, False, torch.float32).to(torch.bfloat16)
                    c = oracle.calibrate(ROW, case, reference, _attention_ref(q, positions, k, v, False, torch.float64),
                                         oracle.FLOOR_OP)
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                    native = _native_attention(q, Q_ROW, k, v, past, k[past:], v[past:], KV_ROW, n, MASK_BLOCK_CAUSAL)
                    oracle.check_within(self, ROW, case, oracle.rel_l2(native, reference), c=c, limit=limit)
                    again = _native_attention(q, Q_ROW, k, v, past, k[past:], v[past:], KV_ROW, n, MASK_BLOCK_CAUSAL)
                    oracle.check_bit_exact(self, ROW, f"{case} rerun", again, native)
                    if n == 1:
                        continue  # one query at a block start: causal and block-causal see the same keys
                    mutant = _native_attention(q, Q_ROW, k, v, past, k[past:], v[past:], KV_ROW, n, MASK_CAUSAL)
                    measured = oracle.rel_l2(mutant, reference)
                    caught = not measured <= limit
                    oracle.report_row(ROW, f"{case} causal mutant", phase="fail_first", measured=measured, limit=limit,
                                      passed=caught)
                    self.assertTrue(caught, f"{case}: the kCausal mutant passes the block-causal oracle "
                                            f"(rel-L2 {measured:.6g} <= {limit:.6g})")

    def test_attention_long_prefill(self):
        """n 4,096 queries over 32,768 keys (the context limit) in one call; the oracle on the first and last 64
        rows."""
        past, n = LONG_PAST, LONG_N
        case = f"attention past {past} n {n}"
        q, k, v = self._attention_inputs(past, n, 7000)
        rows = torch.cat([torch.arange(0, LONG_EDGE), torch.arange(n - LONG_EDGE, n)]).to(oracle.DEVICE)
        positions = past + rows
        reference = _attention_ref(q[rows], positions, k, v, False, torch.float32).to(torch.bfloat16)
        c = oracle.calibrate(ROW, case, reference, _attention_ref(q[rows], positions, k, v, False, torch.float64),
                             oracle.FLOOR_OP)
        limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
        native = _native_attention(q, Q_ROW, k, v, past, k[past:], v[past:], KV_ROW, n, MASK_BLOCK_CAUSAL)
        oracle.check_within(self, ROW, case, oracle.rel_l2(native[rows], reference), c=c, limit=limit)

    def test_attention_tail_slot_equals_scratch(self):
        """New K/V read from the cache tail (written by kv_store) vs from a strided scratch buffer with NaN in the
        cache tail: bit-equal outputs."""
        for past, n in ((0, 8), (8, 3), (52, 64)):
            with self.subTest(past=past, n=n):
                case = f"tail vs scratch past {past} n {n}"
                qkv = _randn((n * QKV_ROW,), 4000 + past + n)
                new_k, new_v = qkv[Q_ROW:], qkv[Q_ROW + KV_ROW:]
                old_k = _randn((past, KV_ROW), 4100 + past + n)
                old_v = _randn((past, KV_ROW), 4200 + past + n)
                garbage = torch.full((n, KV_ROW), float("nan"), dtype=torch.bfloat16, device=oracle.DEVICE)
                k_tail = torch.cat([old_k, garbage]).contiguous()
                v_tail = torch.cat([old_v, garbage]).contiguous()
                oracle.call("cere_ktest_sdar_kv_store", new_k, QKV_ROW, new_v, QKV_ROW, n, KV_HEADS, k_tail, v_tail,
                            past)
                tail = _native_attention(qkv, QKV_ROW, k_tail, v_tail, past, k_tail[past:], v_tail[past:], KV_ROW, n,
                                         MASK_BLOCK_CAUSAL)
                k_scratch = torch.cat([old_k, garbage]).contiguous()
                v_scratch = torch.cat([old_v, garbage]).contiguous()
                scratch = _native_attention(qkv, QKV_ROW, k_scratch, v_scratch, past, new_k, new_v, QKV_ROW, n,
                                            MASK_BLOCK_CAUSAL)
                oracle.check_count(self, ROW, f"{case} finite", int(bool(torch.isfinite(scratch.float()).all())),
                                   expected=1)
                oracle.check_bit_exact(self, ROW, case, tail, scratch)

    # -- kv_store, embed, add -------------------------------------------------------------------------------------

    def test_kv_store_bit_exact(self):
        """K/V rows from a fused projection buffer into cache rows [pos, pos + n); the other rows untouched."""
        n, pos, cache_rows = 5, 12, 24
        for offset in (0, 1):  # 1: a 2-byte-misaligned source (the scalar path)
            with self.subTest(offset=offset):
                case = f"kv_store offset {offset}"
                src = _randn((n * QKV_ROW + 1,), 5000 + offset)
                rows = src[offset:offset + n * QKV_ROW].view(n, QKV_ROW)
                k_cache = _randn((cache_rows, KV_ROW), 5100 + offset)
                v_cache = _randn((cache_rows, KV_ROW), 5200 + offset)
                want_k, want_v = k_cache.clone(), v_cache.clone()
                want_k[pos:pos + n] = rows[:, Q_ROW:Q_ROW + KV_ROW]
                want_v[pos:pos + n] = rows[:, Q_ROW + KV_ROW:]
                oracle.call("cere_ktest_sdar_kv_store", src[offset + Q_ROW:], QKV_ROW, src[offset + Q_ROW + KV_ROW:],
                            QKV_ROW, n, KV_HEADS, k_cache, v_cache, pos)
                oracle.check_bit_exact(self, ROW, f"{case} k", k_cache, want_k)
                oracle.check_bit_exact(self, ROW, f"{case} v", v_cache, want_v)

    def test_embed_bit_exact(self):
        vocab, n = 1000, 9
        for hidden in (HIDDEN, 100):  # 100: not a multiple of 8 (the scalar path)
            with self.subTest(hidden=hidden):
                table = _randn((vocab, hidden), 6000 + hidden)
                ids = torch.tensor([0, vocab - 1, 17, 17, 512, 3, 998, 1, 250], dtype=torch.int32,
                                   device=oracle.DEVICE)
                out = oracle.empty((n, hidden), torch.bfloat16)
                oracle.call("cere_ktest_sdar_embed", table, ids, n, hidden, out)
                oracle.check_bit_exact(self, ROW, f"embed hidden {hidden}", out, table[ids.long()])

    def test_add_bit_exact(self):
        """The residual add: x = bf16(float(x) + float(y)), torch's BF16 `residual + hidden_states`."""
        n = 7 * HIDDEN + 3
        x = _randn((n,), 6500, 4.0)
        y = _randn((n,), 6501, 0.5)
        want = (x.float() + y.float()).to(torch.bfloat16)
        oracle.check_bit_exact(self, ROW, "add torch op", x + y, want)
        native = x.clone()
        oracle.call("cere_ktest_sdar_add", native, y, n)
        oracle.check_bit_exact(self, ROW, "add", native, want)
        aliased = x.clone()
        oracle.call("cere_ktest_sdar_add", aliased, aliased, n)
        oracle.check_bit_exact(self, ROW, "add aliased", aliased, (x.float() * 2.0).to(torch.bfloat16))

    # -- rmsnorm, rope, silu_mul ----------------------------------------------------------------------------------

    def _rmsnorm_case(self, case: str, x: torch.Tensor, w: torch.Tensor, native: torch.Tensor) -> None:
        reference = _rmsnorm_ref(x, w, torch.float32)
        c = oracle.calibrate(ROW, case, reference, _rmsnorm_ref(x, w, torch.float64), oracle.FLOOR_OP)
        limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
        oracle.check_within(self, ROW, case, oracle.rel_l2(native, reference), c=c, limit=limit)

    def test_rmsnorm_rows(self):
        """input/post-attention/final norm: 4,096-wide rows at strides 4,160 (x) and 4,128 (y); y's padding kept."""
        rows, x_stride, y_stride = 7, HIDDEN + 64, HIDDEN + 32
        x = _randn((rows, x_stride), 7100, 3.0)
        w = (1.0 + _randn((HIDDEN,), 7101, 0.1).float()).to(torch.bfloat16)
        y = oracle.empty((rows, y_stride), torch.bfloat16)
        oracle.call("cere_ktest_sdar_rmsnorm", x, x_stride, w, EPS, rows, HIDDEN, y, y_stride)
        self._rmsnorm_case("rmsnorm rows", x[:, :HIDDEN], w, y[:, :HIDDEN])
        oracle.check_count(self, ROW, "rmsnorm rows padding", int(torch.isnan(y[:, HIDDEN:].float()).sum()),
                           expected=rows * (y_stride - HIDDEN))

    def test_rmsnorm_per_head(self):
        """q_norm / k_norm on contiguous q [m, 32, 128] and k [m, 8, 128] (rows m * heads, stride 128), separate and in
        place."""
        w = (1.0 + _randn((HEAD_DIM,), 7200, 0.1).float()).to(torch.bfloat16)
        for heads in (HEADS, KV_HEADS):
            for m in (1, 7, 64):
                with self.subTest(heads=heads, m=m):
                    case = f"rmsnorm heads {heads} m {m}"
                    x = _randn((m, heads, HEAD_DIM), 7300 + heads + m, 2.0)
                    rows = m * heads
                    y = oracle.empty((m, heads, HEAD_DIM), torch.bfloat16)
                    oracle.call("cere_ktest_sdar_rmsnorm", x, HEAD_DIM, w, EPS, rows, HEAD_DIM, y, HEAD_DIM)
                    self._rmsnorm_case(case, x.reshape(rows, HEAD_DIM), w, y.reshape(rows, HEAD_DIM))
                    inplace = x.clone()
                    oracle.call("cere_ktest_sdar_rmsnorm", inplace, HEAD_DIM, w, EPS, rows, HEAD_DIM, inplace,
                                HEAD_DIM)
                    oracle.check_bit_exact(self, ROW, f"{case} in place", inplace, y)

    def test_rope(self):
        """In-place NeoX RoPE on q [m, 32, 128] and k [m, 8, 128] vs the HF formula, positions up to 32,767."""
        m = 64
        for heads in (HEADS, KV_HEADS):
            for pos0 in (0, 52, 32_768 - m):
                with self.subTest(heads=heads, pos0=pos0):
                    case = f"rope heads {heads} pos0 {pos0}"
                    x = _randn((m, heads, HEAD_DIM), 7400 + heads + pos0, 2.0)
                    reference = _rope_hf(x, pos0)
                    c = oracle.calibrate(ROW, case, reference, _rope_fp64(x, pos0), oracle.FLOOR_OP)
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                    native = x.clone()
                    oracle.call("cere_ktest_sdar_rope", native, heads * HEAD_DIM, m, heads, pos0, float(THETA))
                    oracle.check_within(self, ROW, case, oracle.rel_l2(native, reference), c=c, limit=limit)

    def test_silu_mul(self):
        """SDARMLP without liger: F.silu(gate) * up on BF16 tensors (silu rounded to BF16, then the product)."""
        for rows in (1, 5):
            with self.subTest(rows=rows):
                case = f"silu_mul rows {rows}"
                gate_up = _randn((rows, 2 * INTER), 7500 + rows, 3.0)
                gate, up = gate_up[:, :INTER], gate_up[:, INTER:]
                reference = F.silu(gate) * up
                g64, u64 = gate.to(torch.float64), up.to(torch.float64)
                c = oracle.calibrate(ROW, case, reference, g64 / (1.0 + torch.exp(-g64)) * u64, oracle.FLOOR_OP)
                limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                native = oracle.empty((rows, INTER), torch.bfloat16)
                oracle.call("cere_ktest_sdar_silu_mul", gate_up, rows, INTER, native)
                oracle.check_within(self, ROW, case, oracle.rel_l2(native, reference), c=c, limit=limit)
                oracle.check_ulp(self, ROW, f"{case} ulp", native, reference.contiguous(), max_fraction=1e-3)

    # -- top_tie ------------------------------------------------------------------------------------------------

    def test_top_tie_random(self):
        rows = 4
        for stride in (VOCAB, VOCAB + 64):
            with self.subTest(stride=stride):
                g = oracle.generator(8000 + stride)
                logits = torch.randn((rows, stride), generator=g, device=oracle.DEVICE, dtype=torch.float32) * 4.0
                native = _native_top_tie(logits, stride, VOCAB)
                oracle.check_bit_exact(self, ROW, f"top_tie random stride {stride}", native,
                                       _top_tie_ref(logits, VOCAB))

    def test_top_tie_explicit_ties(self):
        """Ties: the lowest id wins, the second-lowest is the runner-up; a unique maximum's runner-up is the lowest id
        of the next value. A highest-id pick is detected."""
        vocab = 1000
        g = oracle.generator(8100)
        logits = (torch.randn((3, vocab), generator=g, device=oracle.DEVICE, dtype=torch.float32) * 0.5).clamp(-3, 3)
        logits[0, [7, 100, vocab - 1]] = 8.0  # ties 3: id 7, runner-up 100
        logits[1] = 0.0  # ties 1,000: id 0, runner-up 1
        logits[2, 500] = 8.0  # unique: id 500, runner-up 30 (the lowest id at 4.0)
        logits[2, [30, 800]] = 4.0
        native = _native_top_tie(logits, vocab, vocab)
        oracle.check_bit_exact(self, ROW, "top_tie explicit ties", native, _top_tie_ref(logits, vocab))
        oracle.check_count(self, ROW, "top_tie explicit ids", int(native[0, 0]) * 10_000 + int(native[1, 0]) * 1_000
                           + int(native[2, 0]), expected=7 * 10_000 + 0 + 500)
        oracle.check_count(self, ROW, "top_tie explicit ties counts", int(native[0, 1]) * 10_000 + int(native[1, 1]),
                           expected=3 * 10_000 + vocab)
        wrong = _top_tie_ref(logits, vocab, highest=True)
        oracle.check_count(self, ROW, "top_tie highest id pick detected", int(torch.equal(native, wrong)), expected=0)

    def test_top_tie_bf16_rounding_ties(self):
        """Condition 1: two FP32 logits that differ but round to one BF16 value are a tie (ties 2, the lowest id wins);
        the FP32-comparing mutant fails it."""
        vocab = 1000
        g = oracle.generator(8200)
        logits = (torch.randn((2, vocab), generator=g, device=oracle.DEVICE, dtype=torch.float32) * 0.25).clamp(-3, 0.5)
        logits[0, 50] = 1.0
        logits[0, 900] = 1.0 + 2.0 ** -10  # larger in FP32, the higher id; rounds to 1.0 in BF16 (ulp 2^-7)
        logits[1, 10] = 1.0 - 2.0 ** -10  # rounds up to 1.0 (BF16 ulp below 1.0 is 2^-8)
        logits[1, 20] = 1.0 + 2.0 ** -10
        oracle.check_count(self, ROW, "top_tie fp32 logits differ",
                           int(bool(logits[0, 50] != logits[0, 900])) + int(bool(logits[1, 10] != logits[1, 20])),
                           expected=2)
        native = _native_top_tie(logits, vocab, vocab)
        oracle.check_bit_exact(self, ROW, "top_tie bf16 ties", native, _top_tie_ref(logits, vocab))
        oracle.check_count(self, ROW, "top_tie bf16 ties ids", int(native[0, 0]) * 1_000 + int(native[1, 0]),
                           expected=50 * 1_000 + 10)
        oracle.check_count(self, ROW, "top_tie bf16 ties counts", int(native[0, 1]) * 10 + int(native[1, 1]),
                           expected=22)
        mutant = _native_top_tie(logits, vocab, vocab, fp32=1)
        oracle.check_bit_exact(self, ROW, "top_tie fp32 mutant vs fp32 oracle", mutant,
                               _top_tie_ref(logits, vocab, fp32=True))
        oracle.check_count(self, ROW, "top_tie fp32 mutant fails condition 1", int(torch.equal(mutant, native)),
                           expected=0)
        oracle.check_count(self, ROW, "top_tie fp32 mutant picks", int(mutant[0, 0]) * 1_000 + int(mutant[0, 1]),
                           expected=900 * 1_000 + 1)

    # -- unmask_select --------------------------------------------------------------------------------------------

    def test_unmask_select(self):
        """Fewest ties among masked rows, the lowest index among equals, -1 when none is masked; the highest-index
        and mask-ignoring mutants are each detected."""
        cases = (([5, 2, 7, 2], [1, 1, 1, 1]), ([1, 3, 3, 9], [0, 1, 1, 1]), ([2, 2], [0, 0]),
                 ([4, 1, 1], [1, 0, 1]), ([3], [1]))
        detected = {"highest index": 0, "mask ignored": 0}
        for i, (ties, masked) in enumerate(cases):
            with self.subTest(case=i):
                top = torch.zeros((len(ties), 5), dtype=torch.int32, device=oracle.DEVICE)
                top[:, 0] = torch.arange(len(ties), dtype=torch.int32)
                top[:, 1] = torch.tensor(ties, dtype=torch.int32)
                mask = torch.tensor(masked, dtype=torch.uint8, device=oracle.DEVICE)
                native = _native_select(top, mask, len(ties))
                oracle.check_count(self, ROW, f"unmask_select case {i}", native, expected=_select(ties, masked))
                detected["highest index"] += int(native != _select(ties, masked, lowest=False))
                detected["mask ignored"] += int(native != _select(ties, masked, honor_mask=False))
        for name, count in detected.items():
            oracle.check_count(self, ROW, f"unmask_select {name} mutant detected", int(count > 0), expected=1)
        empty = torch.zeros((1, 5), dtype=torch.int32, device=oracle.DEVICE)
        none = torch.zeros((1,), dtype=torch.uint8, device=oracle.DEVICE)
        oracle.check_count(self, ROW, "unmask_select rows 0", _native_select(empty, none, 0), expected=-1)

    def test_unmask_select_after_top_tie(self):
        """The denoise step's chain: top_tie on 4 rows (ties 3, 1, 2, 1), rows 0, 2, 3 masked -> row 3."""
        vocab = 1000
        logits = torch.zeros((4, vocab), dtype=torch.float32, device=oracle.DEVICE)
        logits[0, [3, 9, 27]] = 5.0
        logits[1, 4] = 5.0
        logits[2, [1, 2]] = 5.0
        logits[3, 600] = 5.0
        top = _native_top_tie(logits, vocab, vocab)
        oracle.check_count(self, ROW, "chain ties", int((top[:, 1].long() * torch.tensor([1000, 100, 10, 1],
                                                                                          device=oracle.DEVICE)).sum()),
                           expected=3121)
        mask = torch.tensor([1, 0, 1, 1], dtype=torch.uint8, device=oracle.DEVICE)
        oracle.check_count(self, ROW, "chain unmask_select", _native_select(top, mask, 4), expected=3)

    # -- argument refusals ------------------------------------------------------------------------------------------

    def test_argument_refusals(self):
        q, k, v = self._attention_inputs(8, 4, 9000)
        out = oracle.empty((4, Q_ROW), torch.bfloat16)
        refusals = {
            "attention past not a block multiple": lambda: oracle.call_status(
                "cere_ktest_sdar_attention", q, Q_ROW, k, v, 6, k[6:], KV_ROW, v[6:], KV_ROW, 4, HEADS, KV_HEADS, out,
                Q_ROW, MASK_BLOCK_CAUSAL),
            "attention mask 2": lambda: oracle.call_status(
                "cere_ktest_sdar_attention", q, Q_ROW, k, v, 8, k[8:], KV_ROW, v[8:], KV_ROW, 4, HEADS, KV_HEADS, out,
                Q_ROW, 2),
            "attention heads not a kv multiple": lambda: oracle.call_status(
                "cere_ktest_sdar_attention", q, Q_ROW, k, v, 8, k[8:], KV_ROW, v[8:], KV_ROW, 4, HEADS, 7, out, Q_ROW,
                MASK_BLOCK_CAUSAL),
            "rope stride below heads": lambda: oracle.call_status(
                "cere_ktest_sdar_rope", q, Q_ROW - 1, 4, HEADS, 0, float(THETA)),
            "rmsnorm stride below dim": lambda: oracle.call_status(
                "cere_ktest_sdar_rmsnorm", q, HEAD_DIM - 1, q, EPS, 4, HEAD_DIM, out, HEAD_DIM),
            "top_tie fp32 flag 2": lambda: oracle.call_status("cere_ktest_sdar_top_tie", q, 8, 1, 8, out, 2),
            "add n negative": lambda: oracle.call_status("cere_ktest_sdar_add", out, q, -1),
        }
        for name, refused in refusals.items():
            with self.subTest(refusal=name):
                oracle.check_count(self, ROW, f"refusal {name}", refused(), expected=oracle.CERE_ERR_ARGUMENT)
        oracle.check_count(self, ROW, "refusal leaves out unwritten", int(torch.isnan(out.float()).sum()),
                           expected=out.numel())


if __name__ == "__main__":
    unittest.main()
