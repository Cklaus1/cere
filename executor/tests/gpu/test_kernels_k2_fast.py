"""K2X (O73, track K2Fast): the fast small-M MXFP8 K2 GEMM kernels/k2_fast.h and its dispatch, m 1..16.

Every case reports row "K2X". The reference is the kernel k2x replaces, k2s::gemm_mxfp8 (FP4-Local S3, impl 0 of the
ktest entry) on the same E4M3 bytes and scales, which is k2s::gemm (impl 3) on their BF16 upcast bf16(q * 2^(s - 127))
(computed here in torch: the FP32 product is exact, one round to BF16); both references are checked. The weights are
the checkpoint's BF16 tensors of every MXFP8 K2 decode shape in its launch grouping (attention qkv packed and o_proj,
the QSA indexer's index_qk_proj of layer 3; GDN in_proj_qkv / z / b / a as one 4-segment launch and out_proj of layer
0; the shared expert gate / up as one 2-segment launch and down of layer 0), quantized per 32 consecutive k of a row:
scale exponent e = ceil(log2(amax / 448)) (0 for a zero block), s = e + 127, q = E4M3(w / 2^e) round to nearest even,
saturated.

- `mxfp8_<shape>`: for every m in 1..16, k2x::gemm in every layout (the default and each Tune (q, lock, stages) of
  `_tunes` the kernel accepts for the shape's k), k2x::resident in every instantiated layout (the default and each
  ResidentTune (q, kc, stages) of RESIDENT_TUNES, at the k it serves: 2,560 and 6,144) and the dispatch
  k2x::gemm_mxfp8 are bit-equal to the reference (count of equal launches; k2s::gemm on the upcast is counted too, one
  launch per m).
- `e4m3_exact`: every finite E4M3 byte (254 codes, signed zeros and subnormals included) times every scale byte in
  E4M3_SCALES (the checkpoint's 59..120 and the contract's ends k2m::kMinScale 10 and kMaxScale 246) through the
  kernel with a one-hot x equals the upcast in F32, bit for bit, and equals both reference launches.
- `dispatch_<case>`: the calls k2x does not serve (k not a multiple of 128, scales not 4-byte aligned, k past
  kFastMaxK) or was not measured faster on (fewer weight rows than k2_fast.cu's kFastRows / kFastRowsLowM) go to
  k2s::gemm_mxfp8 through the dispatch: bit-equal to it.
- `early_<case>` (BeatSglang S4): k2s::gemm's early launch (impl 5, the MoE router plan's launch: its first round
  of weights issued before the dependency wait) is bit-equal to the plain launch (impl 3) for every m in 1..16 and
  both outputs, on the checkpoint's router weight `mlp.gate` of layer 0 (n 512, k 2,560) and on random BF16 weights
  at k 640 and 6,144 (a slice of more than one round, so only its first round is early).
- `refusals`: m 0 and 17, k past one slice group (split_warps(k) > 16), k not a multiple of 128, scales missing or
  not 4-byte aligned, an invalid layout; k2x::resident also at a k it does not serve (640, 4,096) and for a tune that
  names no instantiated layout.
- Report-only bench `bench_<shape>_M<m>`: µs per call (median of BENCH_ITERS launches, each after an l2_flush of
  512 MiB: cold L2) and effective TB/s (weight + scale + x + y bytes) of k2s::gemm_mxfp8, k2x::gemm (default layout),
  k2x::resident (default layout, at the k it serves) and the dispatch, and k2x's and the dispatch's speedup over
  k2s::gemm_mxfp8.

Native entry point (oracle.py calling convention: device pointers `void *`, host int64 arrays `const int64_t *`,
integers `int64_t`, the stream last; 0 = success). All tensors are contiguous and row-major.

    int cere_ktest_k2x_gemm(const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs,
                            const int64_t *scale_ptrs, const int64_t *ns, const int64_t *y_ptrs, int64_t out,
                            int64_t impl, const int64_t *tune, int64_t iters, void *us, cudaStream_t stream);
        One launch over `count` weights sharing x [m, k] (`iters` times): impl 0..2 E4M3 [n, k] + UE8M0 [n, k / 32]
        (0 k2s::gemm_mxfp8, 1 k2x::gemm with layout tune (host int64 [3] (q, lock, stages), nullptr: the default),
        2 k2x::gemm_mxfp8, 4 k2x::resident with layout tune (host int64 [3] (q, kc, stages), nullptr: the default));
        impl 3 k2s::gemm on BF16 [n, k] (scale_ptrs unused), impl 5 the same with `early`. out 0 BF16, 1 F32. us
        nullptr: untimed; else host f64 [iters], each launch timed after an l2_flush.
"""

from __future__ import annotations

import statistics
import unittest

import torch

import oracle

ROW = "K2X"
ATTN_LAYER = 3
GDN_LAYER = 0
MLP_LAYER = 0
MAX_M = 16
STEP_K = 32
BENCH_M = (1, 2, 4, 8, 16)
BENCH_ITERS = 30
E4M3_MAX = 448.0
E4M3_SCALES = (10, 59, 90, 120, 127, 246)  # k2_mxfp8.h kMinScale .. kMaxScale
SEED = 1_300_000
CERE_ERR_ARGUMENT = -1  # cere_executor.h
SMEM_OPTIN = 101_376  # sm_120 dynamic shared memory per CTA with the opt-in (99 KiB)
K2S, K2X, DISPATCH, K2S_BF16, RESIDENT, K2S_BF16_EARLY = 0, 1, 2, 3, 4, 5  # ktest impl
# k2_fast.h ResidentTune: the instantiated (q, kc, stages) per k the resident layout serves.
RESIDENT_TUNES = {2_560: [(2, 1, 8), (1, 5, 4)], 6_144: [(1, 4, 4), (2, 3, 3), (1, 3, 6)]}


def _weight(name: str) -> torch.Tensor:
    w = oracle.real_tensor(name)
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise oracle.OracleError(f"{name} is {w.dtype} {tuple(w.shape)}; expected 2-D BF16")
    return w.contiguous()


def _groups() -> list:
    """(label, loader of the launch's weights) of every MXFP8 K2 decode launch (forward.cpp's grouping)."""
    attn = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
    gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
    mlp = oracle.layer_prefix(MLP_LAYER) + "mlp."
    return [
        ("gdn_in", lambda: [_weight(gdn + p + ".weight") for p in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                                                                   "in_proj_a")]),
        ("attn_qkv", lambda: [torch.cat([_weight(attn + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                                        dim=0).contiguous()]),
        ("attn_o", lambda: [_weight(attn + "o_proj.weight")]),
        ("attn_index_qk", lambda: [_weight(attn + "indexer.index_qk_proj.weight")]),
        ("gdn_out", lambda: [_weight(gdn + "out_proj.weight")]),
        ("shared_gate_up", lambda: [_weight(mlp + "shared_expert." + p + ".weight") for p in ("gate_proj",
                                                                                            "up_proj")]),
        ("shared_down", lambda: [_weight(mlp + "shared_expert.down_proj.weight")]),
    ]


def _split_warps(k: int) -> int:
    """k2s::split_warps for k <= 6,144 (one slice group)."""
    return max(1, min(16, (k // STEP_K) // 4))


def _padded(row_bytes: int) -> int:
    """k2_fast.cu `padded`: a shared row stride."""
    return -(-row_bytes // 128) * 128 + 32


def _tunes(k: int, m: int) -> list:
    """None (the default layout) and every (q, lock, stages), stages 1 to 4, the kernel accepts at inner dimension k
    and m: lock divides the slices, stages at most the chunks, and the ring (k2_fast.cu `launch`: scales, then stages
    of 8 q weight rows and m x rows of one chunk) within SMEM_OPTIN."""
    slices = _split_warps(k)
    steps = k // STEP_K
    chunks = -(-steps // slices)
    out = [None]
    for q in (1, 2, 4):
        tn = 8 * q
        stage = tn * _padded(slices * STEP_K) + m * _padded(slices * STEP_K * 2)
        scales = -(-tn * steps // 128) * 128
        out += [(q, g, st) for g in (1, 2, 4) for st in (1, 2, 3, 4)
                if slices % g == 0 and st <= chunks and scales + st * stage <= SMEM_OPTIN]
    return out


def _activations(m: int, k: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    return torch.randn((m, k), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)


def _mxfp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(q uint8 [n, k], s uint8 [n, k / 32]) by the module docstring's rule."""
    n, k = w.shape
    blocks = w.float().view(n, k // STEP_K, STEP_K)
    amax = blocks.abs().amax(dim=-1)
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / E4M3_MAX)), torch.zeros_like(amax)).clamp(-127, 7)
    q = (blocks / torch.exp2(e)[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).view(n, k).contiguous(), (e + 127).to(torch.uint8).contiguous()


def _upcast(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """bf16(q * 2^(s - 127)): the FP32 product is exact, one round to BF16."""
    n, k = q.shape
    v = q.view(torch.float8_e4m3fn).float().view(n, k // STEP_K, STEP_K) * torch.exp2(s.float() - 127.0)[..., None]
    return v.view(n, k).to(torch.bfloat16).contiguous()


def _pointers(tensors) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64)


def _launch(x: torch.Tensor, weights: list, scales: list | None, impl: int, tune=None, out_f32: bool = False,
            iters: int = 1, timed: bool = False):
    """One launch over `weights` (E4M3 bytes with `scales`, or BF16 for K2S_BF16): (outputs, µs per launch or
    None)."""
    m, k = x.shape
    ns = [w.shape[0] for w in weights]
    ys = [oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16) for n in ns]
    tune_t = None if tune is None else torch.tensor(list(tune), dtype=torch.int64)
    us = torch.zeros((iters,), dtype=torch.float64) if timed else None
    oracle.call("cere_ktest_k2x_gemm", x.contiguous(), m, k, len(weights), _pointers(weights),
                None if scales is None else _pointers(scales), torch.tensor(ns, dtype=torch.int64), _pointers(ys),
                int(out_f32), impl, tune_t, iters, us)
    return ys, (None if us is None else statistics.median(us.tolist()))


def _equal(a: list, b: list) -> bool:
    return all(torch.equal(p.view(torch.uint8), q.view(torch.uint8)) for p, q in zip(a, b))


class K2Fast(unittest.TestCase):
    def test_mxfp8_equals_small_m(self):
        for index, (label, load) in enumerate(_groups()):
            quantized = [_mxfp8(w) for w in load()]
            qs = [q for q, _ in quantized]
            ss = [s for _, s in quantized]
            up = [_upcast(q, s) for q, s in quantized]
            del quantized
            k = qs[0].shape[1]
            equal = total = 0
            for m in range(1, MAX_M + 1):
                x = _activations(m, k, SEED + 1_000 * index + m)
                reference, _ = _launch(x, qs, ss, K2S)
                bf16, _ = _launch(x, up, None, K2S_BF16)
                runs = [((K2S_BF16, None), bf16), ((DISPATCH, None), _launch(x, qs, ss, DISPATCH)[0])]
                runs += [((K2X, tune), _launch(x, qs, ss, K2X, tune)[0]) for tune in _tunes(k, m)]
                runs += [((RESIDENT, tune), _launch(x, qs, ss, RESIDENT, tune)[0])
                         for tune in [None] + RESIDENT_TUNES.get(k, []) if k in RESIDENT_TUNES]
                for (impl, tune), got in runs:
                    with self.subTest(shape=label, m=m, impl=impl, tune=tune):
                        total += 1
                        equal += int(_equal(got, reference))
            oracle.check_count(self, ROW, f"mxfp8_{label}", equal, expected=total)
            del qs, ss, up

    def test_e4m3_exact(self):
        codes = torch.arange(256, dtype=torch.int64, device=oracle.DEVICE)
        codes = torch.where((codes & 0x7F) == 0x7F, torch.zeros_like(codes), codes).to(torch.uint8)
        n, groups = codes.numel(), len(E4M3_SCALES)
        k = STEP_K * -(-groups // 4) * 4  # MXFP8 k: a multiple of 128 (the padding groups are not read)
        q = codes[:, None].expand(n, k).contiguous()
        s = torch.full((n, k // STEP_K), 127, dtype=torch.uint8, device=oracle.DEVICE)
        s[:, :groups] = torch.tensor(E4M3_SCALES, dtype=torch.uint8, device=oracle.DEVICE)[None, :]
        x = torch.zeros((groups, k), dtype=torch.bfloat16, device=oracle.DEVICE)
        for g in range(groups):
            x[g, STEP_K * g] = 1.0  # row g reads scale group g
        up = _upcast(q, s)
        # + 0.0: the FP32 accumulator starts at +0, so a -0 weight sums to +0.
        expected = up[:, : STEP_K * groups : STEP_K].float().t().contiguous() + 0.0
        (got,), _ = _launch(x, [q], [s], K2X, out_f32=True)
        oracle.check_bit_exact(self, ROW, "e4m3_exact", got, expected)
        (reference,), _ = _launch(x, [up], None, K2S_BF16, out_f32=True)
        oracle.check_bit_exact(self, ROW, "e4m3_exact_reference", got, reference)
        (k2m,), _ = _launch(x, [q], [s], K2S, out_f32=True)
        oracle.check_bit_exact(self, ROW, "e4m3_exact_k2m", got, k2m)
        (dispatched,), _ = _launch(x, [q], [s], DISPATCH, out_f32=True)
        oracle.check_bit_exact(self, ROW, "e4m3_exact_dispatch", dispatched, k2m)

    def _random_mxfp8(self, n: int, k: int, seed: int, scale_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """(q, s): random E4M3 bytes (no NaN) and scales in the checkpoint's range; s starts `scale_offset` bytes into
        its own buffer."""
        g = oracle.generator(seed)
        q = torch.randint(0, 256, (n, k), generator=g, device=oracle.DEVICE, dtype=torch.int64)
        q = torch.where((q & 0x7F) == 0x7F, torch.zeros_like(q), q).to(torch.uint8)
        groups = n * (k // STEP_K)
        buf = torch.randint(59, 121, (groups + scale_offset,), generator=g, device=oracle.DEVICE,
                            dtype=torch.int64).to(torch.uint8)
        return q, buf[scale_offset:].view(n, k // STEP_K)

    def test_dispatch_fallback(self):
        cases = {
            "k_not_128": (4, 2_592, 0),
            "scales_unaligned": (4, 2_560, 1),
            "k_past_fast": (4, 6_144, 0),
        }
        for index, (case, (m, k, offset)) in enumerate(cases.items()):
            q, s = self._random_mxfp8(640, k, SEED + 50_000 + index, offset)
            x = _activations(m, k, SEED + 60_000 + index)
            reference, _ = _launch(x, [q], [s], K2S)
            got, _ = _launch(x, [q], [s], DISPATCH)
            oracle.check_count(self, ROW, f"dispatch_{case}", int(_equal(got, reference)), expected=1)

    def test_early_equals_plain(self):
        g = oracle.generator(SEED + 70_000)
        cases = {
            "router": _weight(oracle.layer_prefix(MLP_LAYER) + "mlp.gate.weight"),
            "k640": torch.randn((2_560, 640), generator=g, device=oracle.DEVICE).to(torch.bfloat16),
            "k6144": torch.randn((2_560, 6_144), generator=g, device=oracle.DEVICE).to(torch.bfloat16),
        }
        for index, (case, w) in enumerate(cases.items()):
            k = w.shape[1]
            equal = total = 0
            for m in range(1, MAX_M + 1):
                x = _activations(m, k, SEED + 75_000 + 100 * index + m)
                for out_f32 in (False, True):
                    plain, _ = _launch(x, [w], None, K2S_BF16, out_f32=out_f32)
                    early, _ = _launch(x, [w], None, K2S_BF16_EARLY, out_f32=out_f32)
                    with self.subTest(case=case, m=m, out_f32=out_f32):
                        total += 1
                        equal += int(_equal(early, plain))
            oracle.check_count(self, ROW, f"early_{case}", equal, expected=total)

    def test_refusals(self):
        n = 64
        cases = {
            "m0": (0, 2_560, None, 0),
            "m17": (17, 2_560, None, 0),
            "k_two_groups": (1, 10_240, None, 0),
            "k_not_128": (1, 2_592, None, 0),
            "tune_q3": (1, 2_560, (3, 1, 0), 0),
            "tune_lock3": (1, 2_560, (1, 3, 0), 0),
            "tune_stages_past_chunks": (1, 2_560, (1, 1, 6), 0),
            "tune_smem_past_optin": (16, 6_144, (4, 1, 3), 0),
            "no_scales": (1, 2_560, None, None),
            "scales_unaligned": (1, 2_560, None, 1),
        }
        resident = {
            "resident_m0": (0, 2_560, None, 0),
            "resident_m17": (17, 2_560, None, 0),
            "resident_k640": (1, 640, None, 0),
            "resident_k4096": (1, 4_096, None, 0),
            "resident_tune_not_built": (1, 2_560, (1, 1, 8), 0),
            "resident_tune_kc_not_dividing": (1, 6_144, (1, 5, 4), 0),
            "resident_tune_q3": (1, 6_144, (3, 4, 4), 0),
            "resident_tune_k6144_at_k2560": (1, 2_560, (1, 4, 4), 0),
            "resident_no_scales": (1, 2_560, None, None),
            "resident_scales_unaligned": (1, 6_144, None, 1),
        }
        for case, (m, k, tune, offset) in list(cases.items()) + list(resident.items()):
            impl = RESIDENT if case.startswith("resident_") else K2X
            x = _activations(max(m, 1), k, SEED + 90_000)
            w = torch.zeros((n, k), dtype=torch.uint8, device=oracle.DEVICE)
            s = torch.zeros((n * (k // STEP_K) + 1,), dtype=torch.uint8, device=oracle.DEVICE)
            y = oracle.empty((max(m, 1), n), torch.bfloat16)
            scale_ptrs = (torch.zeros((1,), dtype=torch.int64) if offset is None
                          else torch.tensor([s.data_ptr() + offset], dtype=torch.int64))
            tune_t = None if tune is None else torch.tensor(list(tune), dtype=torch.int64)
            rc = oracle.call_status("cere_ktest_k2x_gemm", x, m, k, 1, _pointers([w]), scale_ptrs,
                                    torch.tensor([n], dtype=torch.int64), _pointers([y]), 0, impl, tune_t, 1, None)
            oracle.check_count(self, ROW, f"refuse_{case}", rc, expected=CERE_ERR_ARGUMENT)

    def test_bench(self):
        """Report-only: cold µs per call and effective TB/s of k2s::gemm_mxfp8, k2x, k2x::resident (where it serves
        k) and the dispatch."""
        for index, (label, load) in enumerate(_groups()):
            quantized = [_mxfp8(w) for w in load()]
            qs, ss = [q for q, _ in quantized], [s for _, s in quantized]
            del quantized
            k = qs[0].shape[1]
            n = sum(q.shape[0] for q in qs)
            for m in BENCH_M:
                x = _activations(m, k, SEED + 80_000 + index)
                moved = n * k + n * k // STEP_K + 2 * m * k + 2 * m * n
                fields = dict(n=n, k=k, m=m)
                impls = [("k2m", K2S), ("k2x", K2X), ("dispatch", DISPATCH)]
                impls += [("resident", RESIDENT)] if k in RESIDENT_TUNES else []
                for name, impl in impls:
                    _, us = _launch(x, qs, ss, impl, iters=BENCH_ITERS, timed=True)
                    fields.update({f"{name}_us": us, f"{name}_tbps": moved / us / 1e6})
                fields.update(k2x_speedup=fields["k2m_us"] / fields["k2x_us"],
                              dispatch_speedup=fields["k2m_us"] / fields["dispatch_us"])
                oracle.report_row(ROW, f"bench_{label}_M{m}", phase="bench", **fields)
            del qs, ss


if __name__ == "__main__":
    unittest.main()
