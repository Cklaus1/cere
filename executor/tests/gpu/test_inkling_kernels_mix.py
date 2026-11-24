"""Inkling 2 Mini slice S3 (tasks/inkling-spec.md 3.1-3.2, gate G1): the residual-stream kernels of family
inkling2_loop (executor/cuda/src/kernels/inkling/{norm,conv,depth,msg,softcap}.h) vs their CPU references
(executor/ref/inkling/*_ref.h, the shared contract common_ref.h).

Suite `inkling_kernels` (GPU0 only): `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite inkling_kernels` on
the inkling-2-mini library (build/native-inkling-2-mini) with the checkpoint read-only at /model (real weights: the
embedding table and norm weights, conv weights, depth-mix queries, loop controllers; activations are synthetic).
Recorded per-op vectors (S1, tests/fixtures/inkling2_reference/ops/, schema cere-inkling2-ops-v1: the REF forward on
GPU over the first 4 positions of a prompt, loop visits 19 (layer 7, pass 2, sliding) and 23 (layer 11, pass 2, full)
and the final visit): each kernel on the recorded inputs is bit-exact vs the CPU reference; the convs and the residual
helpers are also bit-exact vs the recorded REF outputs, every other op within max(4e-3, 2c) of the FP64 op, c =
rel-L2(recorded REF output, FP64 op).

Every case reports row "INKLING_MIX". G1 has three parts per op:
- bit-exact: the kernel's output equals the CPU reference's (oracle.check_bit_exact) on random inputs with edge rows
  (zeros, large magnitudes, overflowing squares, one-hot softmax, NaN/inf logits), several strides and row counts.
- determinism: every kernel case runs twice, bit-identical; a row's output does not depend on the batch (a row subset
  run alone equals the same rows inside the batch) and the conv prefill split at any boundary with the state carried
  equals the whole prefill (outputs and final state); conv decode equals the n = 1 prefill of each row's own state;
  the S8 fusion (the message gates with their weighted norm, up to 3 gates per call) equals the ops apart; calls of
  more than 64 rows take the prefill-sized message kernels (several rows per block, D-I-CPF): rows 129, 134 (a
  partial last block) and 1,024, and a 3-row subset (the decode-sized kernels) equals the same rows of the larger call.
- the reference is the REF op: c = rel-L2(torch FP32 transcription of the REF module, with its BF16 rounding points,
  vs FP64) is reported first; the CPU reference must be within max(4e-3, 2c) of the FP64 op (the residual add and
  layer output are bit-exact vs torch instead).
Fail-first: the conv without residual and scale, the depth mix with a weighted key norm, the message gate with the
forget and message gates swapped and the fused gate + norm with the prefill-sized kernels' chains reversed each fail
the bit-exact comparison (counted, expected 1).

Entry points (libcere_ktest.so; kernels: executor/cuda/src/ktest/inkling/mix_ktest.cu, device pointers;
references: executor/cuda/src/ktest/inkling/mix_ref_ktest.cpp, host pointers, the stream ignored; int64 sizes, the
stream last; 0 on success):
    int cere_ktest_inkling_rms_norm(const void *x, int64_t x_f32, int64_t ldx, const void *w, float eps, int64_t rows,
                                    void *y, int64_t y_f32, int64_t ldy, cudaStream_t stream);
    int cere_ktest_inkling_embed_norm(const void *table, const void *ids, int64_t rows, const void *w, float eps,
                                      void *out, int64_t ldo, cudaStream_t stream);
    int cere_ktest_inkling_conv_prefill(int64_t dynamic, int64_t mutant, const void *x, int64_t ldx, int64_t n,
                                        int64_t C, const void *w, const void *dyn, int64_t ld_dyn, float *state,
                                        void *y, int64_t ldy, cudaStream_t stream);
    int cere_ktest_inkling_conv_decode(int64_t dynamic, const void *x, int64_t ldx, int64_t M, int64_t C,
                                       const void *w, const void *dyn, int64_t ld_dyn, const void *states, void *y,
                                       int64_t ldy, cudaStream_t stream);
    int cere_ktest_inkling_depth_mix(const void *s0, const void *s1, const void *s2, const void *s3, const void *s4,
                                     int64_t count, int64_t f32_mask, int64_t ld, int64_t rows, const float *queries,
                                     int64_t nq, float *out, int64_t ldo, int64_t out_qstride, const void *mutant_w,
                                     cudaStream_t stream);
    int cere_ktest_inkling_depth_mix_norm(const void *s0, const void *s1, const void *s2, const void *s3,
                                          const void *s4, int64_t count, int64_t f32_mask, int64_t ld, int64_t rows,
                                          const float *queries, int64_t nq, const void *w, float eps, void *y,
                                          int64_t ldy, int64_t y_qstride, cudaStream_t stream);
    int cere_ktest_inkling_attn_residual(int64_t boundary, const void *h, int64_t ldh, const void *o, int64_t ldo,
                                         int64_t rows, float *r, int64_t ldr, cudaStream_t stream);
    int cere_ktest_inkling_layer_out(const float *r, int64_t ldr, const void *d, int64_t ldd, int64_t rows, void *h,
                                     int64_t ldh, cudaStream_t stream);
    int cere_ktest_inkling_msg_gate(int64_t mutant, const float *h, int64_t ldh, const float *msg, int64_t ldm,
                                    int64_t rows, const void *down, const void *forget, const void *gain, float eps,
                                    float *out, int64_t ldo, cudaStream_t stream);
    int cere_ktest_inkling_msg_gate_norm(int64_t mutant, const int64_t *table, int64_t count, int64_t ldh,
                                         int64_t ldm, int64_t ldy, int64_t rows, float eps, cudaStream_t stream);
        (table: host int64 [count, 7] = h, msg, down, forget, gain, norm_w, y device pointers)
    int cere_ktest_inkling_softcap(float *logits, int64_t rows, int64_t vocab, int64_t ld, cudaStream_t stream);
    int cere_ktest_inkling_ref_rms_norm(const void *x, int64_t x_f32, int64_t ldx, const uint16_t *w, float eps,
                                        int64_t rows, void *y, int64_t y_f32, int64_t ldy, void *stream);
    int cere_ktest_inkling_ref_embed_norm(const uint16_t *table, int64_t vocab, const int32_t *ids, int64_t rows,
                                          const uint16_t *w, float eps, uint16_t *out, int64_t ldo, void *stream);
    int cere_ktest_inkling_ref_conv_seq(int64_t dynamic, const uint16_t *x, int64_t ldx, int64_t n, int64_t C,
                                        const uint16_t *w, const uint16_t *dyn, int64_t ld_dyn, float *state,
                                        uint16_t *y, int64_t ldy, void *stream);
    int cere_ktest_inkling_ref_depth_mix(const void *s0, const void *s1, const void *s2, const void *s3,
                                         const void *s4, int64_t count, int64_t f32_mask, int64_t ld, int64_t rows,
                                         const float *queries, int64_t nq, float *out, int64_t ldo,
                                         int64_t out_qstride, void *stream);
    int cere_ktest_inkling_ref_attn_residual(int64_t boundary, const uint16_t *h, int64_t ldh, const uint16_t *o,
                                             int64_t ldo, int64_t rows, float *r, int64_t ldr, void *stream);
    int cere_ktest_inkling_ref_layer_out(const float *r, int64_t ldr, const uint16_t *d, int64_t ldd, int64_t rows,
                                         uint16_t *h, int64_t ldh, void *stream);
    int cere_ktest_inkling_ref_msg_gate(const float *h, int64_t ldh, const float *msg, int64_t ldm, int64_t rows,
                                        const uint16_t *down, const uint16_t *forget, const uint16_t *gain,
                                        float eps, float *out, int64_t ldo, void *stream);
    int cere_ktest_inkling_ref_softcap(float *logits, int64_t rows, int64_t vocab, int64_t ld, void *stream);
"""
from __future__ import annotations

import functools
import json
import math
import unittest

import torch
import torch.nn.functional as F

import oracle

ROW = "INKLING_MIX"
H = 2304
EPS = 1e-6  # config.json rms_norm_eps; also the UnweightedRMSNorm default of the depth mixes
KV = 1024  # 8 KV heads x 128
TAPS = 4
SOFTCAP = 30.0
CPU = torch.device("cpu")


def _randn(shape, seed: int, scale: float = 1.0, dtype=torch.float32) -> torch.Tensor:
    return (torch.randn(shape, generator=oracle.generator(seed), device=oracle.DEVICE) * scale).to(dtype)


def _host(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(CPU).contiguous()


def _f32(dtype) -> int:
    return int(dtype == torch.float32)


def _differ(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose bytes differ (a, b same dtype and shape, on the CPU)."""
    x = a.contiguous().view(torch.uint8).reshape(a.numel(), -1)
    y = b.contiguous().view(torch.uint8).reshape(b.numel(), -1)
    return int((x != y).any(dim=-1).sum().item())


def _check_ref(tc: unittest.TestCase, case: str, ours: torch.Tensor, torch_op: torch.Tensor, fp64: torch.Tensor):
    """The CPU reference is the REF op: within max(4e-3, 2c) of FP64, c = rel-L2(torch FP32 REF op, FP64)."""
    c = oracle.calibrate(ROW, f"{case} ref vs torch", torch_op, fp64, oracle.FLOOR_OP)
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    oracle.check_within(tc, ROW, f"{case} ref vs torch", oracle.rel_l2(ours.to(fp64.device), fp64), c=c, limit=limit)


def _fail_first(tc: unittest.TestCase, case: str, mutant: torch.Tensor, reference: torch.Tensor):
    differing = _differ(_host(mutant), _host(reference))
    oracle.report_row(ROW, f"{case} mutant", phase="fail_first", differing=differing, passed=differing > 0)
    oracle.check_count(tc, ROW, f"{case} mutant refused by the bit-exact check", int(differing > 0), expected=1)


def _canon_nan(t: torch.Tensor) -> torch.Tensor:
    """NaN payloads are not part of the contract (CUDA arithmetic returns the canonical NaN, x86 keeps the input's):
    every NaN becomes torch's NaN, so the byte comparison still requires NaN at the same positions."""
    t = t.clone()
    t[t.isnan()] = math.nan
    return t


# -- torch transcriptions of the REF modules (modeling_inkling2_loop.py) ---------------------------------------------


def _torch_rms(x: torch.Tensor, w: torch.Tensor | None, dtype) -> torch.Tensor:
    """Inkling2LoopRMSNorm :740-745 (w) / Inkling2LoopUnweightedRMSNorm :72-76 (w None), math in `dtype`."""
    x = x.to(dtype)
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return x if w is None else w.to(dtype) * x


def _torch_conv_static(x: torch.Tensor, w: torch.Tensor, state: torch.Tensor, dtype) -> torch.Tensor:
    """Inkling2LoopShortConvolution :362-393 over [state rows; x] (causal_conv1d_fn: y[t] = sum_j w[c, j] x[t-3+j])."""
    xx = torch.cat([state.to(dtype), x.to(dtype)], dim=0)  # [3 + n, C]
    y = F.conv1d(xx.T.unsqueeze(0), w.to(dtype).view(-1, 1, TAPS), groups=x.shape[1])[0].T
    return (y + x.to(dtype)) / math.sqrt(TAPS + 1)


def _torch_conv_dynamic(x: torch.Tensor, static_w: torch.Tensor, dyn: torch.Tensor, state: torch.Tensor,
                        dtype) -> torch.Tensor:
    """Inkling2LoopDynamicShortConvolution :428-455 with the state as the 3 rows before x."""
    n, C = x.shape
    heads = C // 128
    weights = static_w.to(dtype).T.reshape(heads, -1, TAPS) + dyn.to(dtype).view(n, heads, 1, TAPS)
    context = torch.cat([state.to(dtype), x.to(dtype)], dim=0).T.unsqueeze(0)  # [1, C, 3 + n]
    windows = context.unfold(-1, TAPS, 1).flip(-1).transpose(1, 2)  # [1, n, C, 4], [..., lag] = x[t - lag]
    windows = windows.reshape(n, heads, -1, TAPS)
    y = (windows * weights).sum(-1).reshape(n, C)
    return (y + x.to(dtype)) / math.sqrt(2 * TAPS + 1)


def _torch_depth(states: list[torch.Tensor], queries: torch.Tensor, dtype) -> torch.Tensor:
    """Inkling2LoopBlockAttentionResidual :838-853 / MessageAttentionResidual :158-173 (eager): [nq, rows, H]."""
    k = _torch_rms(torch.stack([s.to(dtype) for s in states], dim=1), None, dtype)  # [rows, S, H]
    scores = torch.einsum("qh,rsh->rqs", queries.to(dtype), k) * H ** -0.5
    p = F.softmax(scores, dim=-1, dtype=dtype)
    return torch.einsum("rqs,rsh->qrh", p, k)


def _torch_msg(h, msg, down, forget, gain, dtype) -> torch.Tensor:
    """Inkling2LoopMessageGate :89-96; FP32: the concat cast to BF16 for the BF16 down_proj, as REF."""
    hd, md = h.to(dtype), msg.to(dtype)
    g = torch.cat([_torch_rms(hd, None, dtype), _torch_rms(md, None, dtype)], dim=-1)
    if dtype == torch.float32:
        g = F.linear(g.to(torch.bfloat16), down).float()
    else:
        g = F.linear(g, down.to(dtype))
    f = torch.tanh(F.linear(g, forget.to(dtype)))
    m = torch.tanh(F.linear(g, gain.to(dtype)))
    return (1 - f) * hd + m * md


# -- native and reference calls ----------------------------------------------------------------------------------


def _norm(x: torch.Tensor, w: torch.Tensor | None, y_dtype, ldy: int | None = None) -> torch.Tensor:
    """The kernel on device tensors, the CPU reference on host tensors."""
    rows, ldx = x.shape
    ldy = ldy or H
    if x.is_cuda:
        y = oracle.empty((rows, ldy), y_dtype)
        oracle.call("cere_ktest_inkling_rms_norm", x, _f32(x.dtype), ldx, w, EPS, rows, y, _f32(y_dtype), ldy)
    else:
        y = torch.empty((rows, ldy), dtype=y_dtype)
        oracle.call("cere_ktest_inkling_ref_rms_norm", x, _f32(x.dtype), ldx, w, EPS, rows, y, _f32(y_dtype), ldy)
    return y[:, :H]


def _conv_prefill(x, w, dyn, state, *, dynamic: int, mutant: int = 0) -> torch.Tensor:
    n, C = x.shape
    y = oracle.empty((n, C), torch.bfloat16)
    oracle.call("cere_ktest_inkling_conv_prefill", dynamic, mutant, x, C, n, C, w, dyn,
                dyn.shape[1] if dyn is not None else 0, state, y, C)
    return y


def _conv_ref(x, w, dyn, state, *, dynamic: int) -> torch.Tensor:
    """CPU reference over host copies; `state` (host FP32) is updated in place."""
    n, C = x.shape
    y = torch.empty((n, C), dtype=torch.bfloat16)
    oracle.call("cere_ktest_inkling_ref_conv_seq", dynamic, _host(x), C, n, C, _host(w),
                None if dyn is None else _host(dyn), dyn.shape[1] if dyn is not None else 0, state, y, C)
    return y


def _depth_args(states: list[torch.Tensor]):
    ptrs = list(states) + [None] * (5 - len(states))
    mask = sum(1 << i for i, s in enumerate(states) if s.dtype == torch.float32)
    return ptrs, mask


def _depth(states, queries, *, mutant_w=None) -> torch.Tensor:
    rows = states[0].shape[0]
    nq = queries.shape[0]
    ptrs, mask = _depth_args(states)
    out = oracle.empty((nq, rows, H), torch.float32)
    oracle.call("cere_ktest_inkling_depth_mix", *ptrs, len(states), mask, H, rows, queries, nq, out, H, rows * H,
                mutant_w)
    return out


def _depth_ref(states, queries) -> torch.Tensor:
    host = [_host(s) for s in states]
    rows = host[0].shape[0]
    nq = queries.shape[0]
    ptrs, mask = _depth_args(host)
    out = torch.empty((nq, rows, H), dtype=torch.float32)
    oracle.call("cere_ktest_inkling_ref_depth_mix", *ptrs, len(host), mask, H, rows, _host(queries), nq, out, H,
                rows * H)
    return out


def _depth_norm(states, queries, w) -> torch.Tensor:
    """The fused depth mix + weighted RMSNorm (w BF16 [nq, H]): BF16 [nq, rows, H] at a padded row stride."""
    rows = states[0].shape[0]
    nq = queries.shape[0]
    ldy = H + 64
    ptrs, mask = _depth_args(states)
    y = oracle.empty((nq, rows, ldy), torch.bfloat16)
    oracle.call("cere_ktest_inkling_depth_mix_norm", *ptrs, len(states), mask, H, rows, queries, nq, w, EPS, y, ldy,
                rows * ldy)
    return y[:, :, :H]


def _depth_norm_ref(depth_ref: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The reference of _depth_norm: per query, the CPU RMSNorm (FP32 in, weight w[q], BF16 out) of the reference
    depth mix."""
    wh = _host(w)
    return torch.stack([_norm(depth_ref[q].contiguous(), wh[q].contiguous(), torch.bfloat16)
                        for q in range(depth_ref.shape[0])])


def _msg(h, msg, ws, *, mutant: int = 0, out=None) -> torch.Tensor:
    rows = h.shape[0]
    out = oracle.empty((rows, H), torch.float32) if out is None else out
    oracle.call("cere_ktest_inkling_msg_gate", mutant, h, H, msg, H, rows, *ws, EPS, out, H)
    return out


def _msg_ref(h, msg, ws) -> torch.Tensor:
    rows = h.shape[0]
    out = torch.empty((rows, H), dtype=torch.float32)
    oracle.call("cere_ktest_inkling_ref_msg_gate", _host(h), H, _host(msg), H, rows, *[_host(w) for w in ws], EPS,
                out, H)
    return out


def _edge_rows(x: torch.Tensor, big: float) -> torch.Tensor:
    """Row 0 zeros, row 1 large (squares overflow when big is), row 2 tiny (subnormal squares), row 3 one-hot."""
    x = x.clone()
    n = x.shape[0]
    if n > 0:
        x[0] = 0
    if n > 1:
        x[1] = x[1].sign() * big
    if n > 2:
        x[2] = x[2] * 1e-30
    if n > 3:
        x[3] = 0
        x[3, 7] = 1
    return x


# -- S1's recorded REF vectors (tests/fixtures/inkling2_reference/ops, schema cere-inkling2-ops-v1) -----------------

OPS = oracle.SRC / "tests/fixtures/inkling2_reference/ops"
OPS_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
RECORDED = (("v19_L07_pass2_sliding", 7, False), ("v23_L11_pass2_full", 11, True))  # set, layer, full attention


@functools.cache
def _ops_index() -> dict:
    """set -> {"tensors": key -> {file, dtype, shape}}; the REF forward on GPU over 4 positions of a fresh sequence."""
    return json.loads((OPS / "index.json").read_text())


def _rec(s: str, key: str, *shape: int) -> torch.Tensor:
    """One recorded tensor (raw little-endian file, host) reshaped to `shape`."""
    t = _ops_index()["sets"][s]["tensors"][key]
    data = bytearray((OPS / t["file"]).read_bytes())
    return torch.frombuffer(data, dtype=OPS_DTYPES[t["dtype"]]).reshape(shape)


class InklingMixKernels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.norm_w = oracle.real_tensor("model.llm.norm.weight")
        cls.embed_norm_w = oracle.real_tensor("model.llm.embed_norm.weight")

    # -- RMSNorm ----------------------------------------------------------------------------------------------

    def test_rms_norm(self):
        """Weighted (final norm weight) and unweighted, BF16/FP32 in and out, rows 1..300, row strides > 2304."""
        for x_dtype in (torch.bfloat16, torch.float32):
            for weighted in (True, False):
                for y_dtype in (torch.bfloat16, torch.float32):
                    for rows in (1, 4, 64, 300):
                        case = f"rms_norm x {x_dtype} w {int(weighted)} y {y_dtype} rows {rows}"
                        with self.subTest(case=case):
                            ld = H + 64 * (rows % 2)
                            x = _randn((rows, ld), 10 + rows, 3.0, x_dtype)
                            x[:, :H] = _edge_rows(x[:, :H], 1e30 if x_dtype == torch.float32 else 1e38)
                            w = self.norm_w if weighted else None
                            native = _norm(x, w, y_dtype, H + 32)
                            ref = _norm(_host(x), None if w is None else _host(w), y_dtype)
                            oracle.check_bit_exact(self, ROW, case, _host(native), ref)
                            again = _norm(x, w, y_dtype, H + 32)
                            oracle.check_bit_exact(self, ROW, f"{case} rerun", _host(again), _host(native))
                            if rows >= 8:
                                part = _norm(x[5:8].contiguous(), w, y_dtype)
                                oracle.check_bit_exact(self, ROW, f"{case} rows 5..8 alone", _host(part),
                                                       _host(native[5:8]))
                                body = x[4:, :H].float()  # the random rows (no edge rows)
                                ours = ref[4:].float()
                                wf = None if w is None else w.float()
                                _check_ref(self, case, ours, _torch_rms(body, wf, torch.float32).to(y_dtype),
                                           _torch_rms(body.double(), None if wf is None else wf.double(),
                                                      torch.float64))

    def test_embed_norm(self):
        """Real table and embed_norm weight; ids at the vocab edges (pad, eos, last unpadded, padded rows)."""
        table = oracle.real_tensor("model.llm.embed.weight", device=None)
        vocab = table.shape[0]
        table_gpu = table.to(oracle.DEVICE)
        edge = [0, 1, 199999, 200005, 200006, 200057, 200058, vocab - 1]
        for rows in (1, 8, 64):
            case = f"embed_norm rows {rows}"
            ids = torch.tensor((edge + torch.randint(0, vocab, (rows,), generator=torch.Generator().manual_seed(rows))
                                .tolist())[:rows], dtype=torch.int32)
            if rows == 1:
                ids = torch.tensor([200006], dtype=torch.int32)
            native = oracle.empty((rows, H), torch.bfloat16)
            oracle.call("cere_ktest_inkling_embed_norm", table_gpu, ids.to(oracle.DEVICE), rows, self.embed_norm_w,
                        EPS, native, H)
            ref = torch.empty((rows, H), dtype=torch.bfloat16)
            oracle.call("cere_ktest_inkling_ref_embed_norm", table, vocab, ids, rows, _host(self.embed_norm_w), EPS,
                        ref, H)
            oracle.check_bit_exact(self, ROW, case, _host(native), ref)
            again = oracle.empty((rows, H), torch.bfloat16)
            oracle.call("cere_ktest_inkling_embed_norm", table_gpu, ids.to(oracle.DEVICE), rows, self.embed_norm_w,
                        EPS, again, H)
            oracle.check_bit_exact(self, ROW, f"{case} rerun", _host(again), _host(native))
            if rows == 64:
                rows_x = table[ids.long()].float()
                live = rows_x.square().sum(-1) > 0
                w = self.embed_norm_w.float().cpu()
                _check_ref(self, case, ref[live].float(), _torch_rms(rows_x[live], w, torch.float32).bfloat16(),
                           _torch_rms(rows_x[live].double(), w.double(), torch.float64))

    # -- short convolutions -------------------------------------------------------------------------------------

    def _conv_inputs(self, dynamic: bool, C: int, n: int, seed: int):
        x = _randn((n, C), seed, 2.0, torch.bfloat16)
        x = _edge_rows(x, 1e30)
        state = _randn((3, C), seed + 1, 2.0)
        if not dynamic:
            if C == H:
                w = oracle.real_tensor("model.llm.layers.0.attn_sconv.weight").view(C, TAPS).contiguous()
            elif C == KV:
                w = oracle.real_tensor("model.llm.layers.0.attn.k_sconv.weight").view(C, TAPS).contiguous()
            else:
                w = _randn((C, TAPS), seed + 2, 0.5, torch.bfloat16)
            return x, w, None, state
        if C == KV:
            w = oracle.real_tensor("model.llm.layers.5.attn.k_sconv.static_w").contiguous()
        else:
            w = _randn((TAPS, C), seed + 2, 0.5, torch.bfloat16)
        dyn = _randn((n, C // 128 * TAPS), seed + 3, 0.5, torch.bfloat16)
        return x, w, dyn, state

    def _conv_case(self, dynamic: int, C: int, n: int, seed: int):
        case = f"conv {'dynamic' if dynamic else 'static'} C {C} n {n}"
        x, w, dyn, state0 = self._conv_inputs(bool(dynamic), C, n, seed)
        state = state0.clone()
        native = _conv_prefill(x, w, dyn, state, dynamic=dynamic)
        ref_state = _host(state0)
        ref = _conv_ref(x, w, dyn, ref_state, dynamic=dynamic)
        oracle.check_bit_exact(self, ROW, case, _host(native), ref)
        oracle.check_bit_exact(self, ROW, f"{case} state", _host(state), ref_state)
        state2 = state0.clone()
        again = _conv_prefill(x, w, dyn, state2, dynamic=dynamic)
        oracle.check_bit_exact(self, ROW, f"{case} rerun", _host(again), _host(native))
        oracle.check_bit_exact(self, ROW, f"{case} rerun state", _host(state2), _host(state))
        return case, x, w, dyn, state0, native, ref, ref_state

    def test_conv_prefill(self):
        """Static (real attn_sconv / k_sconv weights, a random C 200) and dynamic (real static_w, random C 256) over n
        1..4,096 with a random FP32 state in; outputs and the state out bit-exact; the ref vs the REF module."""
        for dynamic in (0, 1):
            for C in ((H, KV, 200) if not dynamic else (KV, 256)):
                for n in (1, 2, 3, 4, 5, 31, 32, 33, 300, 4096):
                    with self.subTest(dynamic=dynamic, C=C, n=n):
                        case, x, w, dyn, state0, native, ref, _ = self._conv_case(dynamic, C, n, 100 * n + C)
                        if n == 300:
                            # rows 7.. (random, past the edge rows 0..3): the state is x rows 4..6
                            xb = _host(x)[7:]
                            st = _host(x)[4:7].float()
                            wf = _host(w).float()
                            d = None if dyn is None else _host(dyn)[7:].float()
                            if dynamic:
                                op = _torch_conv_dynamic(xb.float(), wf, d, st, torch.float32).bfloat16()
                                hi = _torch_conv_dynamic(xb.double(), wf.double(), d.double(), st, torch.float64)
                            else:
                                op = _torch_conv_static(xb.float(), wf, st, torch.float32).bfloat16()
                                hi = _torch_conv_static(xb.double(), wf.double(), st, torch.float64)
                            sub_state = st.clone()
                            sub = _conv_ref(x[7:].contiguous(), w, None if dyn is None else dyn[7:].contiguous(),
                                            sub_state, dynamic=dynamic)
                            oracle.check_bit_exact(self, ROW, f"{case} rows 7.. from the carried state", sub, ref[7:])
                            _check_ref(self, case, sub.float(), op, hi)
                        if n >= 4:
                            mut = _conv_prefill(x, w, dyn, state0.clone(), dynamic=dynamic, mutant=1)
                            _fail_first(self, f"{case} no residual/scale", mut, ref)

    def test_conv_chunk_invariance(self):
        """A prefill split at any boundary (state carried) equals the whole prefill: outputs and final state."""
        n = 300
        splits = ([1], [2], [3], [4], [31, 32], [33], [150], [299], [1, 1, 1, 5, 32, 260])
        for dynamic in (0, 1):
            C = KV
            x, w, dyn, state0 = self._conv_inputs(bool(dynamic), C, n, 777 + dynamic)
            whole_state = state0.clone()
            whole = _conv_prefill(x, w, dyn, whole_state, dynamic=dynamic)
            for split in splits:
                case = f"conv {'dynamic' if dynamic else 'static'} chunks {split}"
                with self.subTest(case=case):
                    bounds = [0]
                    for s in split:
                        bounds.append(bounds[-1] + s)
                    if bounds[-1] < n:
                        bounds.append(n)
                    state = state0.clone()
                    parts = []
                    for a, b in zip(bounds, bounds[1:]):
                        parts.append(_conv_prefill(x[a:b].contiguous(), w,
                                                   None if dyn is None else dyn[a:b].contiguous(), state,
                                                   dynamic=dynamic))
                    oracle.check_bit_exact(self, ROW, case, _host(torch.cat(parts)), _host(whole))
                    oracle.check_bit_exact(self, ROW, f"{case} state", _host(state), _host(whole_state))

    def test_conv_decode(self):
        """M rows, each with its own state (pointer array): each equals the n = 1 reference of its state; a decode
        chain over one state equals the prefill of the same rows."""
        for dynamic in (0, 1):
            for C in ((H, KV) if not dynamic else (KV,)):
                for M in (1, 5, 64):
                    case = f"conv decode {'dynamic' if dynamic else 'static'} C {C} M {M}"
                    with self.subTest(case=case):
                        x, w, dyn, _ = self._conv_inputs(bool(dynamic), C, M, 31 * M + C + dynamic)
                        states0 = _randn((M, 3, C), 5 + M, 2.0)
                        states = states0.clone()
                        ptrs = torch.tensor([states[r].data_ptr() for r in range(M)], dtype=torch.int64,
                                            device=oracle.DEVICE)
                        y = oracle.empty((M, C), torch.bfloat16)
                        oracle.call("cere_ktest_inkling_conv_decode", dynamic, x, C, M, C, w, dyn,
                                    0 if dyn is None else dyn.shape[1], ptrs, y, C)
                        ref_states = _host(states0)
                        ref = torch.cat([_conv_ref(x[r:r + 1], w, None if dyn is None else dyn[r:r + 1],
                                                   ref_states[r], dynamic=dynamic) for r in range(M)])
                        oracle.check_bit_exact(self, ROW, case, _host(y), ref)
                        oracle.check_bit_exact(self, ROW, f"{case} states", _host(states), ref_states)
                        states.copy_(states0)
                        y2 = oracle.empty((M, C), torch.bfloat16)
                        oracle.call("cere_ktest_inkling_conv_decode", dynamic, x, C, M, C, w, dyn,
                                    0 if dyn is None else dyn.shape[1], ptrs, y2, C)
                        oracle.check_bit_exact(self, ROW, f"{case} rerun", _host(y2), _host(y))
                        # one state, M decode steps == prefill of M rows
                        chain_state = states0[0].clone()
                        one = torch.tensor([chain_state.data_ptr()], dtype=torch.int64, device=oracle.DEVICE)
                        steps = []
                        for t in range(M):
                            yt = oracle.empty((1, C), torch.bfloat16)
                            oracle.call("cere_ktest_inkling_conv_decode", dynamic, x[t:t + 1].contiguous(), C, 1, C,
                                        w, None if dyn is None else dyn[t:t + 1].contiguous(),
                                        0 if dyn is None else dyn.shape[1], one, yt, C)
                            steps.append(yt)
                        pre_state = states0[0].clone()
                        pre = _conv_prefill(x, w, dyn, pre_state, dynamic=dynamic)
                        oracle.check_bit_exact(self, ROW, f"{case} decode chain", _host(torch.cat(steps)), _host(pre))
                        oracle.check_bit_exact(self, ROW, f"{case} decode chain state", _host(chain_state),
                                               _host(pre_state))

    # -- depth mixes and residual helpers -----------------------------------------------------------------------

    def _stack(self, S: int, rows: int, mask: int, seed: int) -> list[torch.Tensor]:
        states = []
        for s in range(S):
            dtype = torch.float32 if (mask >> s) & 1 else torch.bfloat16
            x = _randn((rows, H), seed + s, 1.0 + 3 * s, dtype)
            if rows > 2 and s == 0:
                x[0] = 0  # a zero state row
            states.append(x)
        if rows > 2 and S > 1:
            states[S - 1][1] = states[0][1].to(states[S - 1].dtype)  # equal-direction states
        return states

    def test_depth_mix(self):
        """Stacks of 1..5 states (BF16 blocks + FP32 partial, all BF16, all FP32) under real queries: attention input
        (1 and 3 queries), MLP input, final, the message mix (4 queries over 2 states); one-hot softmax rows. The fused
        depth mix + weighted RMSNorm (real norm weights, one per query) equals the reference mix then the reference
        RMSNorm, alone or inside the batch."""
        q3 = oracle.real_tensor("model.llm.attn_res.attn_queries.4").contiguous()
        q_mlp = oracle.real_tensor("model.llm.attn_res.mlp_queries.0").view(1, H)
        q_out = oracle.real_tensor("model.llm.attn_res.output_query").view(1, H)
        qm = oracle.real_tensor("model.llm.middle_cycle_cross_pass_queries")
        qkv = oracle.real_tensor("model.llm.middle_cycle_cross_pass_kv_queries")
        q_msg = torch.stack([qm[0], qkv[0], qkv[1], qm[1]]).contiguous()  # loop layer 0: q, k, v, mlp
        cases = [("attention_input full", q3, (1, 2, 3, 4, 5)), ("mlp_input", q_mlp, (2, 5)),
                 ("final_input", q_out, (5,)), ("message", q_msg, (2,))]
        for name, queries, sizes in cases:
            for S in sizes:
                for mask_name, mask in (("partial fp32", 1 << (S - 1)), ("all bf16", 0), ("all fp32", (1 << S) - 1)):
                    for rows in (1, 5, 64, 257):
                        case = f"depth {name} S {S} {mask_name} rows {rows}"
                        with self.subTest(case=case):
                            states = self._stack(S, rows, mask, 1000 * S + rows)
                            native = _depth(states, queries)
                            ref = _depth_ref(states, queries)
                            oracle.check_bit_exact(self, ROW, case, _host(native), ref)
                            again = _depth(states, queries)
                            oracle.check_bit_exact(self, ROW, f"{case} rerun", _host(again), _host(native))
                            w = torch.stack([self.norm_w, self.embed_norm_w, self.norm_w.flip(0),
                                             self.embed_norm_w.flip(0)])[:queries.shape[0]].contiguous()
                            fused = _depth_norm(states, queries, w)
                            oracle.check_bit_exact(self, ROW, f"{case} fused norm", _host(fused),
                                                   _depth_norm_ref(ref, w))
                            if rows == 64:
                                part = _depth([s[9:12].contiguous() for s in states], queries)
                                oracle.check_bit_exact(self, ROW, f"{case} rows 9..12 alone", _host(part),
                                                       _host(native[:, 9:12]))
                                part = _depth_norm([s[9:12].contiguous() for s in states], queries, w)
                                oracle.check_bit_exact(self, ROW, f"{case} fused norm rows 9..12 alone", _host(part),
                                                       _host(fused[:, 9:12]))
                                body = [_host(s)[3:].float() for s in states]
                                qh = _host(queries)
                                _check_ref(self, case, ref[:, 3:], _torch_depth(body, qh, torch.float32),
                                           _torch_depth([b.double() for b in body], qh, torch.float64))
                                if S > 1:
                                    mut = _depth(states, queries, mutant_w=self.norm_w)
                                    _fail_first(self, f"{case} weighted norm", mut, ref)

    def test_depth_mix_one_hot(self):
        """A query aligned with one normed state, scaled until the softmax is one-hot (every other probability
        underflows to 0)."""
        rows, S = 8, 5
        states = self._stack(S, rows, 1 << (S - 1), 4242)
        for target in range(S):
            case = f"depth one-hot state {target}"
            with self.subTest(case=case):
                k = _torch_rms(states[target][2].float(), None, torch.float32)
                queries = (k * 1e4).view(1, H).contiguous()
                native = _depth(states, queries)
                ref = _depth_ref(states, queries)
                oracle.check_bit_exact(self, ROW, case, _host(native), ref)

    def test_residual_helpers(self):
        """attn_residual (boundary and not; at a boundary h is null) and layer_out: bit-exact vs the reference and
        vs torch's REF arithmetic (FP32 add; FP32 add then one BF16 rounding)."""
        for rows in (1, 7, 64):
            h = _edge_rows(_randn((rows, H), 50 + rows, 4.0, torch.bfloat16), 1e38)
            o = _randn((rows, H), 60 + rows, 4.0, torch.bfloat16)
            for boundary in (0, 1):
                case = f"attn_residual boundary {boundary} rows {rows}"
                r = oracle.empty((rows, H), torch.float32)
                oracle.call("cere_ktest_inkling_attn_residual", boundary, None if boundary else h, H, o, H, rows, r, H)
                ref = torch.empty((rows, H), dtype=torch.float32)
                oracle.call("cere_ktest_inkling_ref_attn_residual", boundary, None if boundary else _host(h), H,
                            _host(o), H, rows, ref, H)
                oracle.check_bit_exact(self, ROW, case, _host(r), ref)
                expect = _host(o).float() if boundary else _host(h).float() + _host(o).float()
                oracle.check_bit_exact(self, ROW, f"{case} vs torch", ref, expect)
            r = _randn((rows, H), 70 + rows, 4.0)
            d = _randn((rows, H), 80 + rows, 4.0, torch.bfloat16)
            case = f"layer_out rows {rows}"
            hn = oracle.empty((rows, H), torch.bfloat16)
            oracle.call("cere_ktest_inkling_layer_out", r, H, d, H, rows, hn, H)
            ref = torch.empty((rows, H), dtype=torch.bfloat16)
            oracle.call("cere_ktest_inkling_ref_layer_out", _host(r), H, _host(d), H, rows, ref, H)
            oracle.check_bit_exact(self, ROW, case, _host(hn), ref)
            oracle.check_bit_exact(self, ROW, f"{case} vs torch", ref, (_host(r) + _host(d).float()).bfloat16())
            d_alias = d.clone()
            oracle.call("cere_ktest_inkling_layer_out", r, H, d_alias, H, rows, d_alias, H)
            oracle.check_bit_exact(self, ROW, f"{case} in place", _host(d_alias), ref)

    # -- message gate ---------------------------------------------------------------------------------------------

    def test_msg_gate(self):
        """Real loop-layer-6 controllers (q, k, v, mlp gates); edge rows (zero h, zero msg, large); in place."""
        for gate in ("attn", "k", "v", "mlp"):
            prefix = f"model.llm.layers.6.controllers.{gate}."
            ws = [oracle.real_tensor(prefix + n).contiguous() for n in ("down_weight", "forget_weight", "gain_weight")]
            for rows in (1, 4, 64, 129):
                case = f"msg_gate {gate} rows {rows}"
                with self.subTest(case=case):
                    h = _randn((rows, H), 90 + rows, 3.0)
                    msg = _randn((rows, H), 95 + rows, 3.0)
                    if rows >= 4:
                        h[0] = 0
                        msg[1] = 0
                        h[2] *= 1e15
                        msg[3] *= 1e-30
                    native = _msg(h, msg, ws)
                    ref = _msg_ref(h, msg, ws)
                    oracle.check_bit_exact(self, ROW, case, _host(native), ref)
                    again = _msg(h, msg, ws)
                    oracle.check_bit_exact(self, ROW, f"{case} rerun", _host(again), _host(native))
                    in_place = h.clone()
                    _msg(in_place, msg, ws, out=in_place)
                    oracle.check_bit_exact(self, ROW, f"{case} out = h", _host(in_place), ref)
                    if rows >= 64:
                        part = _msg(h[10:13].contiguous(), msg[10:13].contiguous(), ws)
                        oracle.check_bit_exact(self, ROW, f"{case} rows 10..13 alone", _host(part),
                                               _host(native[10:13]))
                        hb, mb = h[4:], msg[4:]
                        _check_ref(self, case, ref[4:], _host(_torch_msg(hb, mb, *ws, torch.float32)),
                                   _host(_torch_msg(hb, mb, *ws, torch.float64)))
                        mut = _msg(h, msg, ws, mutant=1)
                        _fail_first(self, f"{case} f/m swapped", mut, ref)

    def test_msg_gate_norm(self):
        """S8 fusion: msg_gate_norm over the mlp gate alone, or the q, k, v gates in one call (each its own h: full
        layers; one shared h: sliding layers) equals msg_gate then the weighted rms_norm to BF16, bit for bit; rerun;
        a row subset alone."""
        weights = {}
        for gate in ("attn", "k", "v", "mlp"):
            prefix = f"model.llm.layers.6.controllers.{gate}."
            weights[gate] = [oracle.real_tensor(prefix + n).contiguous()
                             for n in ("down_weight", "forget_weight", "gain_weight")]
        norms = [self.norm_w, self.embed_norm_w, self.norm_w.flip(0).contiguous()]
        calls = (("mlp", ("mlp",), False), ("qkv full", ("attn", "k", "v"), False),
                 ("qkv sliding", ("attn", "k", "v"), True))

        def fused(gates, hs, msgs, mutant=0):
            rows = hs[0].shape[0]
            ys = [oracle.empty((rows, H), torch.bfloat16) for _ in gates]
            table = torch.tensor([[hs[j].data_ptr(), msgs[j].data_ptr(), *[w.data_ptr() for w in weights[g]],
                                   norms[j].data_ptr(), ys[j].data_ptr()] for j, g in enumerate(gates)],
                                 dtype=torch.int64)
            oracle.call("cere_ktest_inkling_msg_gate_norm", mutant, table, len(gates), H, H, H, rows, EPS)
            return ys

        # The reversed chains change FP32 bits that the BF16 roundings (d, then y) mostly hide: the fail-first check
        # pools every call of more than 64 rows (1,024 rows among them).
        mut_all, ys_all = [], []
        for rows in (1, 4, 64, 129, 134, 1024):
            h_all = [_randn((rows, H), 300 + 10 * rows + j, 3.0) for j in range(3)]
            msgs = [_randn((rows, H), 400 + 10 * rows + j, 3.0) for j in range(3)]
            if rows >= 4:
                h_all[0][0] = 0
                msgs[1][1] = 0
                h_all[2][2] *= 1e15
                msgs[0][3] *= 1e-30
            for label, gates, shared in calls:
                hs = [h_all[0]] * len(gates) if shared else h_all[:len(gates)]
                case = f"msg_gate_norm {label} rows {rows}"
                with self.subTest(case=case):
                    ys = fused(gates, hs, msgs)
                    again = fused(gates, hs, msgs)
                    for j, g in enumerate(gates):
                        expect = _norm(_msg(hs[j], msgs[j], weights[g]), norms[j], torch.bfloat16)
                        oracle.check_bit_exact(self, ROW, f"{case} gate {g}", _host(ys[j]), _host(expect))
                        oracle.check_bit_exact(self, ROW, f"{case} gate {g} rerun", _host(again[j]), _host(ys[j]))
                    if rows >= 64:
                        part = fused(gates, [h[10:13].contiguous() for h in hs], [m[10:13].contiguous() for m in msgs])
                        for j, g in enumerate(gates):
                            oracle.check_bit_exact(self, ROW, f"{case} gate {g} rows 10..13 alone", _host(part[j]),
                                                   _host(ys[j][10:13]))
                    if rows > 64:
                        mut_all += fused(gates, hs, msgs, mutant=1)
                        ys_all += ys
        _fail_first(self, "msg_gate_norm prefill kernels' chains reversed (calls of more than 64 rows)",
                    torch.cat(mut_all), torch.cat(ys_all))

    # -- logit softcap --------------------------------------------------------------------------------------------

    def test_softcap(self):
        """30 tanh(x / 30) in place: full vocab rows, a strided small vocab, special values (+-0, +-inf, NaN, huge,
        tiny, the tanh range-reduction neighbourhoods)."""
        special = torch.tensor([0.0, -0.0, math.inf, -math.inf, math.nan, 1e30, -1e30, 30.0, -30.0, 1e-30, -1e-45,
                                88.7, 300.0, 600.0, 1e-3, 15.0, 0.5, -0.5], dtype=torch.float32)
        for rows, vocab, ld in ((1, 201024, 201024), (3, 201024, 201024), (64, 1000, 1024)):
            case = f"softcap rows {rows} vocab {vocab} ld {ld}"
            logits = _randn((rows, ld), 300 + rows, 12.0)
            logits[0, :special.numel()] = special.to(oracle.DEVICE)
            native = logits.clone()
            oracle.call("cere_ktest_inkling_softcap", native, rows, vocab, ld)
            ref = _host(logits)
            oracle.call("cere_ktest_inkling_ref_softcap", ref, rows, vocab, ld)
            oracle.check_bit_exact(self, ROW, case, _canon_nan(_host(native)), _canon_nan(ref))
            again = logits.clone()
            oracle.call("cere_ktest_inkling_softcap", again, rows, vocab, ld)
            oracle.check_bit_exact(self, ROW, f"{case} rerun", _host(again), _host(native))
            body = _host(logits)[:, special.numel():vocab]
            ours = ref[:, special.numel():vocab]
            _check_ref(self, case, ours, SOFTCAP * torch.tanh(body / SOFTCAP),
                       SOFTCAP * torch.tanh(body.double() / SOFTCAP))

    # -- recorded vectors (S1) ----------------------------------------------------------------------------------

    def _recorded_exact(self, case: str, native: torch.Tensor, ref: torch.Tensor, recorded: torch.Tensor):
        """Kernel == CPU reference == the recorded REF output, bit for bit."""
        oracle.check_bit_exact(self, ROW, f"{case} recorded", _host(native), ref)
        oracle.check_bit_exact(self, ROW, f"{case} recorded vs REF", ref, recorded.contiguous())

    def _recorded_within(self, case: str, native: torch.Tensor, ref: torch.Tensor, recorded: torch.Tensor,
                         fp64: torch.Tensor):
        """Kernel == CPU reference bit for bit; the reference within max(4e-3, 2c) of the FP64 op, c = rel-L2(the
        recorded REF output, FP64 op)."""
        oracle.check_bit_exact(self, ROW, f"{case} recorded", _host(native), ref)
        _check_ref(self, f"{case} recorded", ref, recorded.contiguous(), fp64)

    def _recorded_depth(self, s: str, key: str, queries: torch.Tensor, recorded: torch.Tensor):
        """in0 = the block states [rows, S - 1, H] (BF16), in1 = the partial; recorded = [nq, rows, H] FP32."""
        blocks = _rec(s, f"{key}:in0", 4, -1, H)
        states = [blocks[:, i].contiguous() for i in range(blocks.shape[1])] + [_rec(s, f"{key}:in1", 4, H)]
        dev = [t.to(oracle.DEVICE) for t in states]
        self._recorded_within(f"{s} {key}", _depth(dev, queries), _depth_ref(dev, queries), recorded,
                              _torch_depth(states, _host(queries), torch.float64))

    def _recorded_norm(self, s: str, key: str, wname: str | None):
        x = _rec(s, f"{key}:in0", -1, H)
        w = None if wname is None else oracle.real_tensor(wname)
        w_host = None if w is None else _host(w)
        self._recorded_within(f"{s} {key}", _norm(x.to(oracle.DEVICE), w, torch.float32),
                              _norm(x, w_host, torch.float32), _rec(s, f"{key}:out", -1, H),
                              _torch_rms(x, w_host, torch.float64))

    def test_recorded_vectors(self):
        """S1's recorded REF vectors (4 prompt positions from a fresh state): loop visits 19 (layer 7, sliding) and
        23 (layer 11, full) of pass 2, and the final visit. Kernel == CPU reference on every op; the four convs
        (static and dynamic), attn_residual and layer_out == the recorded REF bits; depth mixes (attention_input with
        1 and 3 queries, mlp_input, the 4-query message_input, final_input), the 4 message gates, the norms and the
        softcap within the calibrated bar of the FP64 op. Plus a conv mutant on recorded inputs."""
        index = _ops_index()
        oracle.check_count(self, ROW, "recorded ops schema", int(index["schema"] == "cere-inkling2-ops-v1"),
                           expected=1)
        P = "model.llm."
        qm = oracle.real_tensor(f"{P}middle_cycle_cross_pass_queries")
        qkv = oracle.real_tensor(f"{P}middle_cycle_cross_pass_kv_queries")
        for s, layer, full in RECORDED:
            lp = f"{P}layers.{layer}."
            convs = [("post_attention_convolution", H, lp + "attn_sconv.weight", None),
                     ("post_mlp_convolution", H, lp + "mlp_sconv.weight", None)]
            for kv in ("k", "v"):
                key = f"self_attn.{kv}_proj.sconv"
                convs.append((key, KV, lp + f"attn.{kv}_sconv." + ("static_w" if full else "weight"),
                              f"{key}.weight_proj:out" if full else None))
            for key, C, wname, dyn_key in convs:
                case = f"{s} {key}"
                dynamic = int(dyn_key is not None)
                w = oracle.real_tensor(wname)
                w = w.contiguous() if dynamic else w.view(C, TAPS).contiguous()
                dyn = None if dyn_key is None else _rec(s, dyn_key, 4, -1).to(oracle.DEVICE)
                x = _rec(s, f"{key}:in0", 4, C).to(oracle.DEVICE)
                state = torch.zeros((3, C), dtype=torch.float32, device=oracle.DEVICE)
                native = _conv_prefill(x, w, dyn, state, dynamic=dynamic)
                ref_state = torch.zeros((3, C), dtype=torch.float32)
                ref = _conv_ref(x, w, dyn, ref_state, dynamic=dynamic)
                self._recorded_exact(case, native, ref, _rec(s, f"{key}:out", 4, C))
                oracle.check_bit_exact(self, ROW, f"{case} recorded state", _host(state), ref_state)
                if key == "post_attention_convolution":
                    mut = _conv_prefill(x, w, dyn, torch.zeros_like(state), dynamic=dynamic, mutant=1)
                    _fail_first(self, f"{case} recorded no residual/scale", mut, ref)
            # residual bookkeeping (layers 7 and 11 are not block boundaries)
            h = _rec(s, "layer:in0", 4, H).to(oracle.DEVICE)
            o = _rec(s, "post_attention_convolution:out", 4, H).to(oracle.DEVICE)
            r = oracle.empty((4, H), torch.float32)
            oracle.call("cere_ktest_inkling_attn_residual", 0, h, H, o, H, 4, r, H)
            r_ref = torch.empty((4, H), dtype=torch.float32)
            oracle.call("cere_ktest_inkling_ref_attn_residual", 0, _host(h), H, _host(o), H, 4, r_ref, H)
            self._recorded_exact(f"{s} attn_residual", r, r_ref, _rec(s, "mlp_input:in1", 4, H))
            r_in = _rec(s, "mlp_input:in1", 4, H).to(oracle.DEVICE)
            d = _rec(s, "post_mlp_convolution:out", 4, H).to(oracle.DEVICE)
            hn = oracle.empty((4, H), torch.bfloat16)
            oracle.call("cere_ktest_inkling_layer_out", r_in, H, d, H, 4, hn, H)
            hn_ref = torch.empty((4, H), dtype=torch.bfloat16)
            oracle.call("cere_ktest_inkling_ref_layer_out", _host(r_in), H, _host(d), H, 4, hn_ref, H)
            self._recorded_exact(f"{s} layer_out", hn, hn_ref, _rec(s, "layer:out", 4, H))
            # depth mixes: attention_input = attn_queries.{L-1}, mlp_input = mlp_queries.L, message_input = loop
            # layer i's [q, k, v, mlp] queries
            aq = oracle.real_tensor(f"{P}attn_res.attn_queries.{layer - 1}").view(-1, H).contiguous()
            self._recorded_depth(s, "attention_input", aq,
                                 torch.stack([_rec(s, f"attention_input:out.{q}", 4, H) for q in range(aq.shape[0])]))
            mlp_q = oracle.real_tensor(f"{P}attn_res.mlp_queries.{layer}").view(1, H).contiguous()
            self._recorded_depth(s, "mlp_input", mlp_q, _rec(s, "mlp_input:out.0", 1, 4, H))
            i = layer - 6
            mq = torch.stack([qm[2 * i], qkv[2 * i], qkv[2 * i + 1], qm[2 * i + 1]]).contiguous()
            self._recorded_depth(s, "message_input", mq, _rec(s, "message_input:out", 4, 4, H).transpose(0, 1))
            # message gates
            for g, ctrl in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                key = f"{g}_message_gate"
                ws = [oracle.real_tensor(f"{lp}controllers.{ctrl}.{n}").contiguous()
                      for n in ("down_weight", "forget_weight", "gain_weight")]
                hm, mm = _rec(s, f"{key}:in0", 4, H), _rec(s, f"{key}:in1", 4, H)
                self._recorded_within(f"{s} {key}", _msg(hm.to(oracle.DEVICE), mm.to(oracle.DEVICE), ws),
                                      _msg_ref(hm, mm, ws), _rec(s, f"{key}:out", 4, H),
                                      _torch_msg(hm, mm, *[_host(w) for w in ws], torch.float64))
            # norms (the depth mixes' unweighted key norm over [rows x S] and the weighted pre-norms)
            norms = [("attention_input.norm", None), ("mlp_input.norm", None),
                     ("post_attention_layernorm", lp + "mlp_norm.weight")]
            if full:
                norms += [(f"{g}_input_layernorm", lp + f"{g}_input_norm.weight") for g in ("q", "k", "v")]
            else:
                norms += [(k, lp + "attn_norm.weight") for k in ("input_layernorm", "input_layernorm#1",
                                                                 "input_layernorm#2")]
            for key, wname in norms:
                self._recorded_norm(s, key, wname)
        # the final visit: final_input (output_query), the final norm, the logit softcap
        s = "final"
        out_q = oracle.real_tensor(f"{P}attn_res.output_query").view(1, H).contiguous()
        self._recorded_depth(s, "final_input", out_q, _rec(s, "final_input:out.0", 1, 4, H))
        self._recorded_norm(s, "norm", f"{P}norm.weight")
        logits = _rec(s, "logit_softcap:in0", 1, -1)
        vocab = logits.shape[1]
        native = logits.to(oracle.DEVICE)
        oracle.call("cere_ktest_inkling_softcap", native, 1, vocab, vocab)
        ref = logits.clone()
        oracle.call("cere_ktest_inkling_ref_softcap", ref, 1, vocab, vocab)
        self._recorded_within(f"{s} logit_softcap", native, ref, _rec(s, "logit_softcap:out", 1, -1),
                              SOFTCAP * torch.tanh(logits.double() / SOFTCAP))

    # -- argument refusals --------------------------------------------------------------------------------------

    def test_refusals(self):
        """Malformed arguments return a nonzero status and launch nothing."""
        x = _randn((4, H), 1, 1.0, torch.bfloat16)
        y = oracle.empty((4, H), torch.bfloat16)
        f = oracle.empty((4, H), torch.float32)
        st = torch.zeros((3, 200), dtype=torch.float32, device=oracle.DEVICE)
        w4 = _randn((4, 200), 2, 1.0, torch.bfloat16)
        dyn = _randn((4, 8), 3, 1.0, torch.bfloat16)
        checks = {
            "rms_norm rows 0": oracle.call_status("cere_ktest_inkling_rms_norm", x, 0, H, None, EPS, 0, y, 0, H),
            "rms_norm ldx < 2304": oracle.call_status("cere_ktest_inkling_rms_norm", x, 0, H - 1, None, EPS, 4, y, 0,
                                                      H),
            "conv dynamic C 200": oracle.call_status("cere_ktest_inkling_conv_prefill", 1, 0, x, H, 4, 200, w4, dyn,
                                                     8, st, y, H),
            "depth count 0": oracle.call_status("cere_ktest_inkling_depth_mix", x, None, None, None, None, 0, 0, H, 4,
                                                f, 1, f, H, 0, None),
            "depth null state": oracle.call_status("cere_ktest_inkling_depth_mix", x, None, None, None, None, 2, 0, H,
                                                   4, f, 1, f, H, 0, None),
            "attn_residual null h": oracle.call_status("cere_ktest_inkling_attn_residual", 0, None, H, x, H, 4, f, H),
            "layer_out h = r": oracle.call_status("cere_ktest_inkling_layer_out", f, H, x, H, 4, f, H),
            "softcap ld < vocab": oracle.call_status("cere_ktest_inkling_softcap", f, 4, H, H - 1),
        }
        for name, rc in checks.items():
            oracle.check_count(self, ROW, f"refusal {name}", int(rc != 0), expected=1)


if __name__ == "__main__":
    unittest.main()
