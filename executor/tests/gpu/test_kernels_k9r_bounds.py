"""K9B device-resident index bounds (audit KernelsAttention F3; kernels/k9_qsa.h `Bounds`): every K9 launch that
reads a rotary position or a logical position from device memory clamps it into the call's bounds (rope-table rows
rope_first .. rope_end - 1, logical positions 0 .. context - 1) and sets the fault word; host scalars outside them are
CERE_ERR_ARGUMENT before any launch.

Suite `kernels_k9r` (synthetic inputs, no checkpoint): `native_dev.py test --suite kernels_k9r`.

- test_valid_positions_unchanged: in-range device positions under tight bounds give the bytes of the open-bounds
  call (q_idx, ring, compressed keys) and leave the fault word 0.
- test_compress_clamps / test_compress_mrope_clamps: device rotary positions past the table's end, below its first
  row and near 2^62 (text positions and each mRoPE axis, at group starts so the compressed keys see them too) give
  exactly the bytes of the same call at the clamped positions, with the fault word set; the clamped call leaves it 0.
- test_decode_step_clamps: attention_core_step's QSA launches (`cere_ktest_qsa_decode_step`: compress_store_step,
  indexer_qk and select_step at one device position) at positions past the context, negative and near 2^62 equal the
  call at the clamped position (ring, compressed keys, q_idx, indices), fault set.
- test_compress_at_clamps: the draft graph's `qsa_compress_store_at` at a device start past context - tokens or below
  0 equals the call at the clamped start, fault set.
- test_chains_clamps: `qsa_chains_step` fused and per row (key_save_step + compress_store_step) with a row's device
  position past its slot map equal the calls at the clamped position, fault set.
- test_host_arguments: host pos0 / tokens outside the rope rows or the context, an empty rotary range, a context
  below 1 and a decode context above ctx are CERE_ERR_ARGUMENT with every output untouched.

C prototypes (libcere_ktest.so; executor/cuda/src/ktest/k9_ktest.cpp; cere_ktest_qsa_compress,
cere_ktest_qsa_compress_mrope and cere_ktest_qsa_chains_step as stated in test_kernels_k9_qsa.py and
test_kernels_k9r_chains.py):

    int cere_ktest_qsa_set_bounds(int64_t rope_first, int64_t rope_end, int64_t context, void* fault,
        cudaStream_t stream);
    int cere_ktest_qsa_decode_step(const void* qk, const void* position, const void* slots, const void* q_norm,
        const void* k_norm, float eps, const void* cos_sin, int64_t ctx, void* ring, void* keys, void* q_idx,
        void* indices, cudaStream_t stream);
    int cere_ktest_qsa_compress_at(const void* qk, int64_t tokens, const void* position, const void* slots,
        const void* k_norm, float eps, const void* cos_sin, void* ring, void* keys, cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch

import oracle

ROW = "K9B"
HEAD_DIM = 128  # k9::kIndexHeadDim
INDEX_HEADS = 4  # k9::kIndexHeads
QK_WIDTH = (INDEX_HEADS + 1) * HEAD_DIM  # k9::kQkWidth
RATIO = 4  # k9::kRatio
ROTARY = 64  # k9::kRotaryDim
ROW_WIDTH = 2051  # k9::kRowWidth
EPS = 1e-6
CTX = 4096  # the context and the table's rows (rope_delta 0: rope_end == context)
FAR = 1 << 62
ERR_ARGUMENT = -1  # CERE_ERR_ARGUMENT
BF16 = torch.bfloat16
I64 = torch.int64


def _bf16(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, generator=oracle.generator(seed), device=oracle.DEVICE) * scale).to(BF16).contiguous()


def _i64(values) -> torch.Tensor:
    return torch.tensor(values, dtype=I64, device=oracle.DEVICE)


def _clamp(values, lo: int, hi: int) -> list[int]:
    return [min(max(v, lo), hi - 1) for v in values]


class K9Bounds(unittest.TestCase):
    def setUp(self) -> None:
        g = oracle.generator(9100)
        self.cos_sin = torch.rand((CTX, ROTARY), generator=g, device=oracle.DEVICE).mul_(2).sub_(1).contiguous()
        self.q_norm = _bf16((HEAD_DIM,), 9101, 0.5)
        self.k_norm = _bf16((HEAD_DIM,), 9102, 0.5)
        self.slots = torch.arange(CTX, dtype=I64, device=oracle.DEVICE)
        self.addCleanup(oracle.call, "cere_ktest_qsa_set_bounds", 0, 0, 0, None)

    def _bounds(self, rope_first: int, rope_end: int, context: int) -> torch.Tensor:
        """Sets the bounds with a fresh zeroed fault word, returned."""
        fault = torch.zeros(1, dtype=torch.int32, device=oracle.DEVICE)
        oracle.call("cere_ktest_qsa_set_bounds", rope_first, rope_end, context, fault)
        return fault

    def _fault(self, case: str, fault: torch.Tensor, expected: int) -> None:
        oracle.check_count(self, ROW, f"{case}/fault", int(fault.item()), expected=expected)

    def _same(self, case: str, got: dict, ref: dict) -> None:
        for name in ref:
            oracle.check_bit_exact(self, ROW, f"{case}/{name}", got[name], ref[name])

    # -- the text prep (indexer_qk + qsa_compress_store at device positions) --------------------------------------

    def _compress(self, qk: torch.Tensor, pos0: int, positions, state: dict) -> dict:
        out = {"ring": state["ring"].clone(), "keys": state["keys"].clone(),
               "q_idx": oracle.empty((qk.shape[0], INDEX_HEADS, HEAD_DIM), BF16)}
        oracle.call("cere_ktest_qsa_compress", qk, qk.shape[0], pos0, _i64(positions), self.slots, self.q_norm,
                    self.k_norm, EPS, self.cos_sin, out["ring"], out["keys"], out["q_idx"])
        return out

    def _state(self, seed: int) -> dict:
        return {"ring": _bf16((RATIO, HEAD_DIM), seed), "keys": _bf16((CTX // RATIO, HEAD_DIM), seed + 1)}

    def test_valid_positions_unchanged(self) -> None:
        tokens, pos0 = 10, 8
        qk = _bf16((tokens, QK_WIDTH), 9110)
        state = self._state(9111)
        positions = [CTX - 1 - t for t in range(tokens)]  # any in-range rotary rows, the table's last included
        ref = self._compress(qk, pos0, positions, state)  # open bounds
        fault = self._bounds(0, CTX, CTX)
        got = self._compress(qk, pos0, positions, state)
        self._same("valid/compress", got, ref)
        self._fault("valid/compress", fault, 0)

    def test_compress_clamps(self) -> None:
        tokens, pos0 = 10, 8  # groups 2, 3 complete (rows 0 and 4 their starts), row 8 starts the ring's group
        qk = _bf16((tokens, QK_WIDTH), 9120)
        state = self._state(9121)
        positions = [pos0 + t for t in range(tokens)]
        positions[4] = CTX + 7  # a group start: index q and compressed key
        positions[6] = -5  # index q only
        positions[8] = FAR
        fault = self._bounds(0, CTX, CTX)
        got = self._compress(qk, pos0, positions, state)
        self._fault("compress/clamped", fault, 1)
        fault = self._bounds(0, CTX, CTX)
        ref = self._compress(qk, pos0, _clamp(positions, 0, CTX), state)
        self._fault("compress/reference", fault, 0)
        self._same("compress", got, ref)

    def test_compress_mrope_clamps(self) -> None:
        tokens, pos0 = 9, 6  # logical 6 .. 14; columns 0 .. 10 are logical 4 .. 14 (p3_lead 2)
        lead = pos0 % RATIO
        stride = lead + tokens
        qk = _bf16((tokens, QK_WIDTH), 9130)
        state = self._state(9131)
        axes = [[pos0 - lead + c for c in range(stride)] for _ in range(3)]
        axes[0][0] = FAR  # group 1's start (its compressed key)
        axes[1][4] = CTX + 3  # group 2's start
        axes[2][7] = -9  # index q of logical 7 only

        def run(p3) -> dict:
            out = {"ring": state["ring"].clone(), "keys": state["keys"].clone(),
                   "q_idx": oracle.empty((tokens, INDEX_HEADS, HEAD_DIM), BF16)}
            oracle.call("cere_ktest_qsa_compress_mrope", qk, tokens, pos0, _i64(p3), stride, lead, self.slots,
                        self.q_norm, self.k_norm, EPS, self.cos_sin, out["ring"], out["keys"], out["q_idx"])
            return out

        fault = self._bounds(0, CTX, CTX)
        got = run(axes)
        self._fault("compress_mrope/clamped", fault, 1)
        fault = self._bounds(0, CTX, CTX)
        ref = run([_clamp(a, 0, CTX) for a in axes])
        self._fault("compress_mrope/reference", fault, 0)
        self._same("compress_mrope", got, ref)

    # -- device-position launches -----------------------------------------------------------------------------

    def _decode(self, qk: torch.Tensor, position: int, state: dict) -> dict:
        out = {"ring": state["ring"].clone(), "keys": state["keys"].clone(),
               "q_idx": oracle.empty((1, INDEX_HEADS, HEAD_DIM), BF16),
               "indices": oracle.empty((1, ROW_WIDTH), torch.int32)}
        oracle.call("cere_ktest_qsa_decode_step", qk, _i64([position]), self.slots, self.q_norm, self.k_norm, EPS,
                    self.cos_sin, CTX, out["ring"], out["keys"], out["q_idx"], out["indices"])
        return out

    def test_decode_step_clamps(self) -> None:
        qk = _bf16((1, QK_WIDTH), 9140)
        state = self._state(9141)
        for position in (CTX, CTX + 3, -2, FAR, -FAR):
            case = f"decode_step/p{'far' if abs(position) == FAR else position}"
            with self.subTest(case=case):
                fault = self._bounds(0, CTX, CTX)
                got = self._decode(qk, position, state)
                self._fault(f"{case}/clamped", fault, 1)
                fault = self._bounds(0, CTX, CTX)
                ref = self._decode(qk, min(max(position, 0), CTX - 1), state)
                self._fault(f"{case}/reference", fault, 0)
                self._same(case, got, ref)

    def _compress_at(self, qk: torch.Tensor, position: int, state: dict) -> dict:
        out = {"ring": state["ring"].clone(), "keys": state["keys"].clone()}
        oracle.call("cere_ktest_qsa_compress_at", qk, qk.shape[0], _i64([position]), self.slots, self.k_norm, EPS,
                    self.cos_sin, out["ring"], out["keys"])
        return out

    def test_compress_at_clamps(self) -> None:
        tokens = 5
        qk = _bf16((tokens, QK_WIDTH), 9150)
        state = self._state(9151)
        for position, clamped in ((CTX - 2, CTX - tokens), (CTX + 9, CTX - tokens), (-1, 0), (FAR, CTX - tokens)):
            case = f"compress_at/p{'far' if position == FAR else position}"
            with self.subTest(case=case):
                fault = self._bounds(0, CTX, CTX)
                got = self._compress_at(qk, position, state)
                self._fault(f"{case}/clamped", fault, 1)
                fault = self._bounds(0, CTX, CTX)
                ref = self._compress_at(qk, clamped, state)
                self._fault(f"{case}/reference", fault, 0)
                self._same(case, got, ref)

    def test_chains_clamps(self) -> None:
        layout = (4,)
        rows = sum(layout)
        k_idx = _bf16((rows, HEAD_DIM), 9160, 2.0)
        rings = _bf16((1, RATIO, HEAD_DIM), 9161)
        keys = _bf16((1, CTX // RATIO, HEAD_DIM), 9162)

        def run(positions, fused: int) -> dict:
            out = {"rings": rings.clone(), "keys": keys.clone(),
                   "undo_rings": oracle.empty((rows, RATIO, HEAD_DIM), BF16),
                   "undo_keys": oracle.empty((rows, HEAD_DIM), BF16)}
            oracle.call("cere_ktest_qsa_chains_step", k_idx, self.k_norm, EPS, self.cos_sin, _i64(positions),
                        torch.tensor(layout, dtype=I64), 1, self.slots, CTX, out["rings"], out["keys"], CTX // RATIO,
                        out["undo_rings"], out["undo_keys"], 1, fused)
            return out

        for fused in (1, 0):
            for name, positions in (("past", [CTX - 4, CTX - 3, CTX - 2, CTX + 6]),
                                    ("far", [CTX - 3, CTX - 2, FAR, FAR]), ("negative", [-7, 1, 2, 3])):
                case = f"chains/{name}/fused{fused}"
                with self.subTest(case=case):
                    fault = self._bounds(0, CTX, CTX)
                    got = run(positions, fused)
                    self._fault(f"{case}/clamped", fault, 1)
                    fault = self._bounds(0, CTX, CTX)
                    ref = run(_clamp(positions, 0, CTX), fused)
                    self._fault(f"{case}/reference", fault, 0)
                    self._same(case, got, ref)

    # -- host scalars -----------------------------------------------------------------------------------------

    def test_host_arguments(self) -> None:
        qk = _bf16((8, QK_WIDTH), 9170)
        state = self._state(9171)

        def compress(tokens: int, pos0: int) -> tuple[int, dict]:
            out = {"ring": state["ring"].clone(), "keys": state["keys"].clone()}
            rc = oracle.call_status("cere_ktest_qsa_compress", qk, tokens, pos0, None, self.slots, self.q_norm,
                                    self.k_norm, EPS, self.cos_sin, out["ring"], out["keys"],
                                    oracle.empty((8, INDEX_HEADS, HEAD_DIM), BF16))
            return rc, out

        def decode(ctx: int, owned: tuple[str, ...]) -> tuple[int, dict]:
            """The outputs of the refusing launch (`owned`): the step's compress runs before select_step's check."""
            out = {"ring": state["ring"].clone(), "keys": state["keys"].clone(),
                   "indices": oracle.empty((1, ROW_WIDTH), torch.int32)}
            rc = oracle.call_status("cere_ktest_qsa_decode_step", qk, _i64([5]), self.slots, self.q_norm,
                                    self.k_norm, EPS, self.cos_sin, ctx, out["ring"], out["keys"],
                                    oracle.empty((1, INDEX_HEADS, HEAD_DIM), BF16), out["indices"])
            return rc, {name: out[name] for name in owned}

        def compress_at(tokens: int) -> tuple[int, dict]:
            out = {"ring": state["ring"].clone(), "keys": state["keys"].clone()}
            rc = oracle.call_status("cere_ktest_qsa_compress_at", qk, tokens, _i64([0]), self.slots, self.k_norm,
                                    EPS, self.cos_sin, out["ring"], out["keys"])
            return rc, out

        cases = {
            "rope_end": ((0, 64, CTX), lambda: compress(8, 60)),  # rotary rows 60 .. 67 past the table's 64
            "rope_first": ((10, CTX, CTX), lambda: compress(4, 9)),  # index rows 9 .., compressed key row 8, below 10
            "context": ((0, CTX, 64), lambda: compress(8, 60)),  # logical 60 .. 67 past the context
            "empty_rope": ((5, 5, CTX), lambda: compress(1, 0)),
            "decode_context_above_ctx": ((0, CTX, CTX), lambda: decode(CTX // 2, ("indices",))),
            "decode_context_below_1": ((0, CTX, -1), lambda: decode(CTX, ("ring", "keys", "indices"))),
            "compress_at_tokens_past_context": ((0, CTX, 4), lambda: compress_at(5)),
        }
        untouched = {**state, "indices": oracle.empty((1, ROW_WIDTH), torch.int32)}
        for name, ((rope_first, rope_end, context), run) in cases.items():
            case = f"arguments/{name}"
            with self.subTest(case=case):
                fault = self._bounds(rope_first, rope_end, context)
                rc, out = run()
                oracle.report_row(ROW, case, phase="result", status=rc)
                self.assertEqual(rc, ERR_ARGUMENT, name)
                self._same(case, out, {k: untouched[k] for k in out})
                self._fault(case, fault, 0)


if __name__ == "__main__":
    unittest.main()
