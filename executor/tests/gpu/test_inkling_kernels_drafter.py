"""Inkling 2 Mini S9 phase B (tasks/inkling-spec.md 3.5, D-I-S9B, gate G1): the block drafter's kernels
(executor/cuda/src/kernels/inkling/drafter.h) vs their CPU reference (executor/ref/inkling/drafter_ref.h, through
libcere_ktest.so, ktest/inkling/drafter_ktest.cu), bit for bit, and the whole Drafter (model/inkling/drafter.h):
graph replay == eager, batch-invariant drafts.

Suite `inkling_kernels` (GPU0 only): `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite inkling_kernels`
against build/native-inkling-2-mini with the checkpoint read-only at /model (real weights: mtp.safetensors and the
target's embed / embed_norm / unembed; activations synthetic, at the recorded magnitudes). Every case reports row
"INKLING_DRAFTER". Cases:
- mix_rmsnorm (alpha from the real and a random fuse_weight, 3 planes, row indirection), the residual add + RMSNorm,
  SiLU * up (incl. overflow and NaN), the block q/k head norm + RoPE and the context store (ctx-form RoPE with the
  BF16 round between, ring slots j % 1024 across the wrap, the full cache) at positions up to 2^17;
- block attention: the non-causal block over [lo, L + 8) with lo = max(0, L - 1024) on the local layers (ring) and
  0 on the full layer, at L in {0, 5, 700, 1023, 1024, 1025, 1500, 3000} (local) and up to 5,000 (full: past the
  4,096-key group), single and 64-block batches (several span passes), bound-independent; fail-first mutants: a causal
  block mask and a per-row (target) sliding window must FAIL;
- Markov gates (bias or none), the Markov vocab step (ties -> first id, NaN loses, ids past 200,058 never win) with
  its logits, and the FP64 fold on real unembed rows and the real markov_head.hidden_out (and random).
- Drafter: draft() graph replay == eager; a batch of 3 == each block alone; n_drafts 5 == the first 5 of 7.
"""

from __future__ import annotations

import functools
import unittest

import torch
from safetensors import safe_open

import oracle

ROW = "INKLING_DRAFTER"
H = 2304
QROW, KVROW, HD = 3072, 1024, 128
BLOCK, WINDOW, RANK, PROJ = 8, 1024, 256, 768
VOCAB = 200058
LOGIT_LD = 200064
DEV = oracle.DEVICE
CPU = torch.device("cpu")
BF = torch.bfloat16


def _bf(shape, scale=1.0, gen=None, device=DEV) -> torch.Tensor:
    return (torch.randn(shape, generator=gen) * scale).to(BF).to(device)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _host(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(CPU).contiguous()


def _canon_nan(t: torch.Tensor) -> torch.Tensor:
    """NaN payloads are not part of the contract (CUDA arithmetic returns the canonical NaN, x86 keeps the input's):
    every NaN becomes torch's NaN, so the byte comparison still requires NaN at the same positions."""
    if not t.is_floating_point():
        return t
    t = t.clone()
    t[t.isnan()] = float("nan")
    return t


@functools.cache
def _mtp() -> dict[str, torch.Tensor]:
    """mtp.safetensors (BF16) on the device, by name."""
    out = {}
    with safe_open(str(oracle.MODEL_DIR / "mtp.safetensors"), framework="pt", device="cpu") as f:
        for name in f.keys():
            out[name] = f.get_tensor(name).contiguous().to(DEV)
    return out


@functools.cache
def _target() -> dict[str, torch.Tensor]:
    return {k: oracle.real_tensor(f"model.llm.{k}.weight").contiguous()
            for k in ("embed", "embed_norm", "unembed")}


def _rope(pos: torch.Tensor) -> torch.Tensor:
    """FP32 [rows, 128] cos then sin (drafter_ref.h rope_cos_sin), on the host."""
    p = pos.to(torch.int64).to(CPU).contiguous()
    cs = torch.empty((p.numel(), HD), dtype=torch.float32)
    oracle.call("cere_ktest_inkling_draft_ref_rope", p, p.numel(), cs)
    return cs


class InklingDrafterKernels(unittest.TestCase):
    def _exact(self, case, native, ref):
        oracle.check_bit_exact(self, ROW, case, _canon_nan(_host(native)), _canon_nan(_host(ref)))

    # ---- mix_rmsnorm -------------------------------------------------------------------------------------------
    def _alpha(self, fuse: torch.Tensor) -> torch.Tensor:
        alpha = torch.empty(16, dtype=torch.float32)
        oracle.call("cere_ktest_inkling_draft_ref_fuse_alpha", _host(fuse), alpha)
        return alpha

    def test_mix_rmsnorm(self):
        mtp = _mtp()
        g = _gen(1)
        rows = 37
        taps = torch.cat([_bf((rows, 3, H), 2.0, g), _bf((rows, 1, H), 0.05, g)], 1).contiguous()  # final-norm tap
        taps[3, 0, :8] = torch.tensor([300.0, -300.0, 1e-30, 0.0, -0.0, 7.0, -7.0, 1.0], dtype=BF)
        norms = [mtp[f"layers.{i}.dflare_norm.weight"] for i in range(3)] + [mtp["markov_head.dflare_norm.weight"]]
        for label, fuse in (("real", mtp["fuse_weight"]), ("random", _bf((4, 4), 3.0, g))):
            alpha = self._alpha(fuse)
            src = torch.randperm(rows, generator=g).to(torch.int32)
            # planes 0..2 (the context layers) through the row indirection, plane 3 alone (f_rnn)
            y = oracle.empty((3, rows, H), BF)
            oracle.call("cere_ktest_inkling_draft_mix", taps, 4 * H, src.to(DEV), rows, alpha.to(DEV), 0, 3,
                        norms[0], norms[1], norms[2], y, H, rows * H)
            y3 = oracle.empty((rows, H), BF)
            oracle.call("cere_ktest_inkling_draft_mix", taps, 4 * H, None, rows, alpha.to(DEV), 3, 1, norms[3], None,
                        None, y3, H, 0)
            ht = _host(taps)[src.long()].contiguous()
            for p in range(4):
                ref = torch.empty((rows, H), dtype=BF)
                oracle.call("cere_ktest_inkling_draft_ref_mix", ht if p < 3 else _host(taps), 4 * H, rows, alpha, p,
                            _host(norms[p]), ref, H)
                self._exact(f"mix {label} alpha row {p}", y[p] if p < 3 else y3, ref)

    # ---- residual add + RMSNorm, SiLU * up ---------------------------------------------------------------------
    def test_add_rmsnorm(self):
        g = _gen(2)
        blocks, group, lead, take = 3, 8, 1, 7
        n_in = blocks * group
        add_in, res_in = _bf((n_in, H), 3.0, g), _bf((n_in, H), 5.0, g)
        w = _mtp()["layers.1.input_layernorm.weight"]
        idx = torch.tensor([(i // take) * group + lead + i % take for i in range(blocks * take)])
        for label, a in (("add", add_in), ("copy", None)):
            res_out = oracle.empty((n_in, H), BF)
            y = oracle.empty((blocks * take, H), BF)
            oracle.call("cere_ktest_inkling_draft_add_norm", a, res_in, res_out, w, y, blocks * take, group, lead, take)
            r = _host(res_in) if a is None else (_host(a).float() + _host(res_in).float()).to(BF)
            self._exact(f"add_norm {label} residual", _host(res_out)[idx], r[idx])
            ref = torch.empty((blocks * take, H), dtype=BF)
            oracle.call("cere_ktest_inkling_ref_rms_norm", r[idx].contiguous(), 0, H, _host(w), 1e-6, blocks * take,
                        ref, 0, H)
            self._exact(f"add_norm {label} normed", y, ref)
        # res_out aliasing res_in (the model's in-place residual)
        res = res_in.clone()
        y = oracle.empty((blocks * take, H), BF)
        oracle.call("cere_ktest_inkling_draft_add_norm", add_in, res, res, w, y, blocks * take, group, lead, take)
        r = (_host(add_in).float() + _host(res_in).float()).to(BF)
        self._exact("add_norm in place", _host(res)[idx], r[idx])

    def test_silu_mul(self):
        g = _gen(3)
        rows, n = 9, 9216
        gate, up = _bf((rows, n), 4.0, g), _bf((rows, n), 1.0, g)
        gate[0, :6] = torch.tensor([-100.0, 100.0, float("nan"), float("inf"), float("-inf"), -0.0], dtype=BF)
        act = oracle.empty((rows, n), BF)
        oracle.call("cere_ktest_inkling_draft_silu_mul", gate, up, act, rows, n)
        ref = torch.empty((rows, n), dtype=BF)
        oracle.call("cere_ktest_inkling_draft_ref_silu_mul", _host(gate), _host(up), ref, rows * n)
        self._exact("silu_mul", act, ref)

    # ---- head norm + RoPE --------------------------------------------------------------------------------------
    def test_qk_norm_rope(self):
        mtp = _mtp()
        g = _gen(4)
        rows = 29
        pos = torch.tensor([0, 1, 7, 1023, 1024, 4095, 4096, 4103, 131071, 131072] +
                           torch.randint(0, 200000, (rows - 10,), generator=g).tolist())
        cs = _rope(pos)
        for layer in range(3):
            q, k = _bf((rows, QROW), 3.0, g), _bf((rows, KVROW), 3.0, g)
            q[0, :4] = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=BF)
            qw, kw = mtp[f"layers.{layer}.self_attn.q_norm.weight"], mtp[f"layers.{layer}.self_attn.k_norm.weight"]
            rq, rk = torch.empty((rows, QROW), dtype=BF), torch.empty((rows, KVROW), dtype=BF)
            oracle.call("cere_ktest_inkling_draft_ref_head_norm_rope", _host(q), QROW, rows, 24, _host(qw), cs, 0, rq,
                        QROW)
            oracle.call("cere_ktest_inkling_draft_ref_head_norm_rope", _host(k), KVROW, rows, 8, _host(kw), cs, 0, rk,
                        KVROW)
            oracle.call("cere_ktest_inkling_draft_qk_norm_rope", q, QROW, qw, k, KVROW, kw, cs.to(DEV), rows)
            self._exact(f"qk_norm_rope layer {layer} q", q, rq)
            self._exact(f"qk_norm_rope layer {layer} k", k, rk)

    def test_ctx_store(self):
        mtp = _mtp()
        g = _gen(5)
        max_ctx = 3000
        # two sessions: positions 0..9 then 1020..1030 (ring wrap: 1024..1030 take the slots of 0..6) in one,
        # 2990..2999, 1124 then 100 (1124 keeps slot 100 although row 100 comes later) in the other
        pos = [list(range(10)) + list(range(1020, 1031)), list(range(2990, 3000)) + [1124, 100]]
        rings = [oracle.empty((2, 2, WINDOW, KVROW), BF) for _ in range(2)]
        fulls = [oracle.empty((2, max_ctx, KVROW), BF) for _ in range(2)]
        rows_tab, flat = [], []
        for s, ps in enumerate(pos):
            for p in ps:
                # ring_write: no larger position of the call maps to this slot (Drafter::context's rule)
                keep = int(not any(q > p and q % WINDOW == p % WINDOW for q in ps))
                rows_tab.append([_ptr(rings[s]), _ptr(fulls[s]), p, max_ctx, keep])
                flat.append((s, p, keep))
        n = len(rows_tab)
        k, v = _bf((3, n, KVROW), 4.0, g), _bf((3, n, KVROW), 1.0, g)
        kn = [mtp[f"layers.{i}.self_attn.k_norm_ctx.weight"] for i in range(3)]
        cs = _rope(torch.tensor([p for _, p, _ in flat]))
        exp_r = [_host(r).clone() for r in rings]
        exp_f = [_host(f).clone() for f in fulls]
        for layer in range(3):
            kr = torch.empty((n, KVROW), dtype=BF)
            oracle.call("cere_ktest_inkling_draft_ref_head_norm_rope", _host(k[layer]), KVROW, n, 8, _host(kn[layer]),
                        cs, 1, kr, KVROW)
            for i, (s, p, keep) in enumerate(flat):
                if layer == 1:
                    exp_f[s][0, p], exp_f[s][1, p] = kr[i], _host(v[layer, i])
                elif keep:
                    plane = 0 if layer == 0 else 1
                    exp_r[s][plane, 0, p % WINDOW], exp_r[s][plane, 1, p % WINDOW] = kr[i], _host(v[layer, i])
        tab = torch.tensor(rows_tab, dtype=torch.int64, device=DEV)
        oracle.call("cere_ktest_inkling_draft_ctx_store", k, v, n * KVROW, kn[0], kn[1], kn[2], cs.to(DEV), tab, n)
        for s in range(2):
            self._exact(f"ctx_store ring session {s}", rings[s], exp_r[s])
            self._exact(f"ctx_store full session {s}", fulls[s], exp_f[s])

    # ---- block attention ---------------------------------------------------------------------------------------
    def _attention(self, case: str, Ls: list[int], local: bool, *, bound: int | None = None, mutant: int = 0,
                   seed: int = 6, expect_equal: bool = True, timed: bool = False) -> None:
        g = _gen(seed)
        blocks = len(Ls)
        hist = max(Ls) + 1
        # one shared history per call: positions [0, hist) of K/V (recorded magnitudes: normed + roped keys ~ 1)
        hk, hv = _bf((hist, KVROW), 1.0, g, CPU), _bf((hist, KVROW), 1.0, g, CPU)
        q, k, v = _bf((blocks * BLOCK, QROW), 2.0, g), _bf((blocks * BLOCK, KVROW), 1.0, g), \
            _bf((blocks * BLOCK, KVROW), 1.0, g)
        caches, dev_tab, host_tab = [], [], []
        for L in Ls:
            if local:
                ck = torch.full((WINDOW, KVROW), float("nan"), dtype=BF)
                cv = ck.clone()
                for j in range(max(0, L - WINDOW), L):
                    ck[j % WINDOW], cv[j % WINDOW] = hk[j], hv[j]
            else:
                ck = torch.full((max(L, 1) + 16, KVROW), float("nan"), dtype=BF)
                cv = ck.clone()
                ck[:L], cv[:L] = hk[:L], hv[:L]
            dk, dv = ck.to(DEV), cv.to(DEV)
            caches.append((ck, cv, dk, dv))
            dev_tab.append([_ptr(dk), _ptr(dv), L, int(local)])
            host_tab.append([_ptr(ck), _ptr(cv), L, int(local)])
        bound = bound or max(Ls) + BLOCK
        ws_bytes = int(oracle.ktest("cere_ktest_inkling_draft_attention_ws_bytes")(oracle._arg(blocks)))
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=DEV)
        o = oracle.empty((blocks * BLOCK, QROW), BF)
        oracle.call("cere_ktest_inkling_draft_attention", torch.tensor(dev_tab, dtype=torch.int64, device=DEV), blocks,
                    int(local), q, k, v, o, bound, ws, ws_bytes, mutant)
        ref = torch.empty((blocks * BLOCK, QROW), dtype=BF)
        oracle.call("cere_ktest_inkling_draft_ref_attention", torch.tensor(host_tab, dtype=torch.int64), blocks,
                    0 if local else 1, _host(q), _host(k), _host(v), ref)
        if expect_equal:
            self._exact(case, o, ref)
            if timed:
                res = torch.empty(1, dtype=torch.float64)
                oracle.call("cere_ktest_inkling_draft_attention_time",
                            torch.tensor(dev_tab, dtype=torch.int64, device=DEV), blocks, int(local), q, k, v, o,
                            bound, ws, ws_bytes, 200, res)
                oracle.report_row(ROW, f"{case} time", phase="timing", us=round(float(res[0]) * 1000.0, 2))
            return
        a, b = _host(o).view(torch.int16), ref.view(torch.int16)
        differing = int((a != b).sum())
        oracle.report_row(ROW, case, phase="mutant", elements=a.numel(), differing=differing, passed=differing > 0)
        oracle.check_count(self, ROW, f"{case}: the mutant differs from the reference (the test can see it)",
                           int(differing > 0), expected=1)

    def test_attention_local(self):
        for L in (0, 5, 700, 1023, 1024, 1025, 1500, 3000):
            self._attention(f"attention local L={L}", [L], True, seed=100 + L)
        self._attention("attention local batch", [0, 5, 700, 1023, 1024, 1500, 2047, 3000], True)
        self._attention("attention local bound 8192", [1500, 3], True, bound=8192)

    def test_attention_full(self):
        for L in (0, 5, 700, 1024, 4088, 4089, 4100, 5000):
            self._attention(f"attention full L={L}", [L], False, seed=200 + L)
        self._attention("attention full bound 8192", [5000, 17], False, bound=8192)
        g = _gen(7)
        Ls = [int(x) for x in torch.randint(0, 5001, (64,), generator=g)]
        Ls[:4] = [0, 4096, 4088, 5000]
        self._attention("attention full 64 blocks", Ls, False, seed=8)
        self._attention("attention local 64 blocks", Ls, True, seed=9)

    def test_attention_draft_shape(self):
        # draft()'s shape at L = 4096 (bound 5120, the 1024-step graph bound), timed per layer kind
        self._attention("attention local L=4096 bound 5120", [4096], True, bound=5120, seed=300, timed=True)
        self._attention("attention full L=4096 bound 5120", [4096], False, bound=5120, seed=301, timed=True)

    def test_attention_mutants_fail(self):
        # the causal block mask hides the block's later rows from its earlier ones (any L)
        self._attention("mutant causal (local L=1500)", [1500], True, mutant=1, expect_equal=False)
        self._attention("mutant causal (full L=5)", [5], False, mutant=1, expect_equal=False)
        # the target's per-row window drops keys [L - 1024, L + r - 1023) that the block-anchored window keeps
        self._attention("mutant row window (local L=1500)", [1500], True, mutant=2, expect_equal=False)
        self._attention("mutant row window (local L=3000)", [3000, 1024], True, mutant=2, expect_equal=False)

    # ---- Markov head -------------------------------------------------------------------------------------------
    def test_markov_gates(self):
        g = _gen(10)
        blocks = 7
        acc = (torch.randn((blocks, PROJ), generator=g) * 4.0).to(DEV)
        acc[0, :6] = torch.tensor([100.0, -100.0, 0.0, -0.0, 1e-30, float("nan")])
        bias = _mtp()["markov_head.rnn_proj.bias"]
        for label, b in (("bias", bias), ("no bias", None)):
            state = _bf((blocks, RANK), 0.5, g)
            st0 = _host(state)
            latent = oracle.empty((blocks, RANK), BF)
            oracle.call("cere_ktest_inkling_draft_gates", acc, b, state, latent, blocks)
            proj = (_host(acc).to(BF) if b is None else (_host(acc) + _host(b).float()).to(BF)).contiguous()
            rs, rl = st0.clone(), torch.empty((blocks, RANK), dtype=BF)
            oracle.call("cere_ktest_inkling_draft_ref_gates", proj, rs, rl, blocks)
            self._exact(f"gates {label} state", state, rs)
            self._exact(f"gates {label} latent", latent, rl)

    def _vocab(self, case: str, latent, fold_w, fold_b, base, blocks: int) -> None:
        keys = torch.empty(blocks, dtype=torch.int64, device=DEV)
        logits = oracle.empty((blocks, LOGIT_LD), BF)
        oracle.call("cere_ktest_inkling_draft_vocab", latent, fold_w, fold_b, base, LOGIT_LD, blocks, keys, logits,
                    LOGIT_LD)
        ids = torch.empty(blocks, dtype=torch.int64)
        ref_logits = torch.empty((blocks, LOGIT_LD), dtype=BF)
        oracle.call("cere_ktest_inkling_draft_ref_vocab", _host(latent), _host(fold_w), _host(fold_b), _host(base),
                    LOGIT_LD, blocks, ids, ref_logits, LOGIT_LD)
        self._exact(f"{case} logits", _host(logits)[:, :VOCAB], ref_logits[:, :VOCAB])
        k = _host(keys)
        got = torch.where(k == 0, torch.zeros_like(k), (~k) & 0xFFFFFFFF)
        self._exact(f"{case} ids", got, ids)

    def test_markov_vocab(self):
        g = _gen(11)
        blocks = 5
        latent = torch.tanh(torch.randn((blocks, RANK), generator=g) * 2).to(BF).to(DEV)
        fold_w = _bf((VOCAB, RANK), 0.05, g)
        fold_b = _bf((VOCAB,), 0.5, g)
        base = _bf((blocks, LOGIT_LD), 6.0, g)
        base[:, VOCAB:] = 1e4  # padding past the step vocab never wins (not read)
        self._vocab("vocab random", latent, fold_w, fold_b, base, blocks)
        # block 0: an exact tie at the maximum (identical fold rows, bias and base) -> the first id wins
        fold_w[150000], fold_b[150000] = fold_w[77], fold_b[77]
        base[0, 77] = base[0, 150000] = 200.0
        # block 1: NaN logits lose (scattered NaN bases); block 2: every logit NaN but the last id
        base[1, ::97] = float("nan")
        base[2, :VOCAB - 1] = float("nan")
        # block 3: the maximum at the last step-vocab id; block 4: every logit NaN -> id 0, as markov_step
        base[3, VOCAB - 1] = 500.0
        base[4, :VOCAB] = float("nan")
        self._vocab("vocab ties nan edges", latent, fold_w, fold_b, base, blocks)
        # 64 blocks (one atomic slot per block, both key halves of the lane map)
        g2 = _gen(12)
        lat64 = torch.tanh(torch.randn((64, RANK), generator=g2)).to(BF).to(DEV)
        base64 = _bf((64, LOGIT_LD), 6.0, g2)
        self._vocab("vocab 64 blocks", lat64, fold_w, fold_b, base64, 64)

    def test_fold(self):
        mtp = _mtp()
        unembed = _target()["unembed"]
        hw, hb = mtp["markov_head.hidden_out.weight"], mtp["markov_head.hidden_out.bias"]
        g = _gen(13)
        cases = (("fold real", torch.cat([unembed[:1021], unembed[VOCAB - 6:VOCAB]]).contiguous(), hw, hb),
                 ("fold random", _bf((19, H), 0.03, g), _bf((H, RANK), 0.1, g), _bf((H,), 0.1, g)))
        for case, lm, w, b in cases:
            rows = lm.shape[0]
            fw, fb = oracle.empty((rows, RANK), BF), oracle.empty((rows,), BF)
            oracle.call("cere_ktest_inkling_draft_fold", lm, rows, w, b, fw, fb)
            rw, rb = torch.empty((rows, RANK), dtype=BF), torch.empty((rows,), dtype=BF)
            oracle.call("cere_ktest_inkling_draft_ref_fold", _host(lm), rows, _host(w), _host(b), rw, rb)
            self._exact(f"{case} w", fw, rw)
            self._exact(f"{case} b", fb, rb)


def drafter_sizes(max_ctx: int) -> tuple[int, int]:
    """The open drafter's (workspace bytes, one draft session's bytes at `max_ctx`)."""
    out = torch.empty(2, dtype=torch.int64)
    oracle.call("cere_ktest_inkling_drafter_sizes", max_ctx, out)
    return int(out[0]), int(out[1])


class DrafterHarness:
    """The drafter of the ktest (cere_ktest_inkling_drafter_*) on the real mtp.safetensors and target tensors."""

    def __init__(self):
        mtp = _mtp()
        tgt = _target()
        names = sorted(mtp)
        self.tensors = [mtp[n] for n in names]
        self.ptrs = torch.tensor([t.data_ptr() for t in self.tensors], dtype=torch.int64)
        self.names = torch.tensor(list(b"".join(n.encode() + b"\0" for n in names)), dtype=torch.uint8)
        self.meta = torch.tensor([[1, t.dim()] + list(t.shape) + [0] * (4 - t.dim()) for t in self.tensors],
                                 dtype=torch.int64)
        oracle.call("cere_ktest_inkling_drafter_open", self.ptrs, self.names, self.meta, len(names), tgt["embed"],
                    tgt["embed_norm"], tgt["unembed"])

    def sessions(self, count: int, max_ctx: int) -> None:
        oracle.call("cere_ktest_inkling_drafter_sessions", count, max_ctx)

    def context(self, sess: list[int], pos: list[int], frnn: list[bool], taps: torch.Tensor) -> None:
        """taps BF16 [n, 4, 2304] on the device."""
        n = len(sess)
        for at in range(0, n, 4096):
            s = slice(at, min(n, at + 4096))
            oracle.call("cere_ktest_inkling_drafter_context", torch.tensor(sess[s], dtype=torch.int64),
                        torch.tensor(pos[s], dtype=torch.int64), torch.tensor([int(f) for f in frnn[s]],
                                                                              dtype=torch.int64),
                        taps[s].contiguous(), 4 * H, s.stop - s.start)

    def draft(self, sess: list[int], anchors: list[int], Ls: list[int], n_drafts: int, graphs: bool,
              probe: torch.Tensor | None = None) -> torch.Tensor:
        B = len(sess)
        ids = torch.zeros((B, 7), dtype=torch.int32, device=DEV)
        oracle.call("cere_ktest_inkling_drafter_draft", torch.tensor(sess, dtype=torch.int64),
                    torch.tensor(anchors, dtype=torch.int64), torch.tensor(Ls, dtype=torch.int64), B, n_drafts,
                    int(graphs), ids, probe)
        return _host(ids)[:, :n_drafts]

    def close(self) -> None:
        oracle.call("cere_ktest_inkling_drafter_close")


class InklingDrafter(unittest.TestCase):
    def test_draft_graph_batch(self):
        d = DrafterHarness()
        try:
            ws, sb = drafter_sizes(4096)
            oracle.report_row(ROW, "drafter sizes", phase="info", workspace_bytes=ws, draft_session_bytes_4096=sb)
            g = _gen(20)
            Ls = [1500, 300, 4096]
            d.sessions(3, 4104)
            sess, pos, frnn = [], [], []
            for s, L in enumerate(Ls):
                sess += [s] * L
                pos += list(range(L))
                frnn += [p == L - 1 for p in range(L)]
            taps = torch.cat([_bf((len(sess), 3, H), 2.0, g), _bf((len(sess), 1, H), 0.05, g)], 1).contiguous()
            d.context(sess, pos, frnn, taps)
            anchors = [int(x) for x in torch.randint(0, VOCAB, (3,), generator=g)]
            eager = d.draft([0, 1, 2], anchors, Ls, 7, False)
            graph = d.draft([0, 1, 2], anchors, Ls, 7, True)
            graph2 = d.draft([0, 1, 2], anchors, Ls, 7, True)
            self.assertTrue(torch.equal(eager, graph), f"graph {graph.tolist()} != eager {eager.tolist()}")
            self.assertTrue(torch.equal(graph, graph2), "graph replay is not deterministic")
            for b in range(3):
                alone = d.draft([b], [anchors[b]], [Ls[b]], 7, True)
                self.assertTrue(torch.equal(alone[0], eager[b]), f"block {b}: alone {alone.tolist()} != batch")
            five = d.draft([0, 1, 2], anchors, Ls, 5, True)
            self.assertTrue(torch.equal(five, eager[:, :5]), "n_drafts 5 != the first 5 of 7")
            oracle.report_row(ROW, "drafter graph == eager, batch == alone", phase="result", passed=True)
        finally:
            d.close()


if __name__ == "__main__":
    unittest.main()
