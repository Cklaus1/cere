"""K9R chains step (track FuseGlue, tasks/fuse-glue-spec.md FG-6): `k9::qsa_chains_step`, one launch for a
full-attention layer's verify QSA work over every chain of the verify graph, against the per-row launches it
replaces in forward.cpp attention_chains (per row in row order: draft rows j >= 1 copy the ring to their undo ring
and, on a QSA plan, run `k9::qsa_key_save_step`; on a QSA plan every row then runs `k9::qsa_compress_store_step`).

Suite `kernels_k9r` (synthetic inputs, no checkpoint): `native_dev.py --gpu 1 test --suite kernels_k9r`.

- test_bit_identical: for each layout of LAYOUTS (chains of 1..16 rows, 16 rows at most) and each chain start
  offset (the four p % 4 phases, so rows complete groups at every j, and key saves hit and miss), on a QSA plan and
  without one (ring copies only), the rings, the compressed keys, the undo rings and the undo keys after the fused
  call equal the per-row launches' bytes (undo buffers start as NaN, so an unwritten element differs nowhere only
  when the reference leaves it unwritten too).
- test_arguments: no chains, more than 16, rows past 16 and a chain of 0 rows are CERE_ERR_ARGUMENT before any launch.

C prototype (libcere_ktest.so; executor/cuda/src/ktest/k9_ktest.cpp):

    int cere_ktest_qsa_chains_step(const void* k_idx, const void* k_norm, float eps, const void* cos_sin,
        const void* positions, const void* layout, int64_t chains, const void* slots, int64_t slots_len,
        void* rings, void* keys, int64_t keys_rows, void* undo_rings, void* undo_keys, int64_t qsa, int64_t fused,
        cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch

import oracle

ROW = "K9R"
HEAD_DIM = 128  # k9::kIndexHeadDim
RATIO = 4  # k9::kRatio
ROTARY = 64  # k9::kRotaryDim
EPS = 1e-6
SLOTS = 4096
KEY_ROWS = SLOTS // RATIO
LAYOUTS = ((4, 4, 4, 4), (16,), (1,) * 16, (9, 3, 1, 3), (2, 5, 1), (4,))
STARTS = (1000, 1001, 1002, 1003)
ERR_ARGUMENT = -1  # CERE_ERR_ARGUMENT
BF16 = torch.bfloat16


def _gen(seed: int) -> torch.Generator:
    return oracle.generator(seed)


def _bf16(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, generator=_gen(seed), device=oracle.DEVICE) * scale).to(BF16).contiguous()


class ChainsStep(unittest.TestCase):
    def _inputs(self, layout, start: int, seed: int):
        chains, m = len(layout), sum(layout)
        g = _gen(seed)
        positions = []
        for c, count in enumerate(layout):
            p = start + 64 * c
            positions += [p + j for j in range(count)]
        n_pos = max(positions) + 8
        cos_sin = torch.rand((n_pos, ROTARY), generator=g, device=oracle.DEVICE).mul_(2).sub_(1).contiguous()
        slots = torch.stack([torch.randperm(SLOTS, generator=g, device=oracle.DEVICE) for _ in range(chains)])
        return dict(
            k_idx=_bf16((m, HEAD_DIM), seed + 1, 2.0),
            k_norm=_bf16((HEAD_DIM,), seed + 2, 0.5),
            cos_sin=cos_sin,
            positions=torch.tensor(positions, dtype=torch.int64, device=oracle.DEVICE),
            layout=torch.tensor(layout, dtype=torch.int64),
            chains=chains,
            slots=slots.to(torch.int64).contiguous(),
            rings=_bf16((chains, RATIO, HEAD_DIM), seed + 3),
            keys=_bf16((chains, KEY_ROWS, HEAD_DIM), seed + 4),
            undo_rings=oracle.empty((m, RATIO, HEAD_DIM), BF16),
            undo_keys=oracle.empty((m, HEAD_DIM), BF16),
        )

    @staticmethod
    def _run(x, qsa: int, fused: int):
        out = {k: x[k].clone() for k in ("rings", "keys", "undo_rings", "undo_keys")}
        oracle.call("cere_ktest_qsa_chains_step", x["k_idx"], x["k_norm"], EPS, x["cos_sin"], x["positions"],
                    x["layout"], x["chains"], x["slots"], SLOTS, out["rings"], out["keys"], KEY_ROWS,
                    out["undo_rings"], out["undo_keys"], qsa, fused)
        return out

    def test_bit_identical(self) -> None:
        for li, layout in enumerate(LAYOUTS):
            for start in STARTS:
                for qsa in (1, 0):
                    case = f"chains_step/{'-'.join(map(str, layout))}/p{start % RATIO}/qsa{qsa}"
                    with self.subTest(case=case):
                        x = self._inputs(layout, start, 7300 + 16 * li + start % RATIO)
                        ref = self._run(x, qsa, 0)
                        got = self._run(x, qsa, 1)
                        for name in ("rings", "keys", "undo_rings", "undo_keys"):
                            oracle.check_bit_exact(self, ROW, f"{case}/{name}", got[name], ref[name])

    def test_arguments(self) -> None:
        x = self._inputs((4, 4), 1000, 7400)
        cases = {"no_chains": ((4, 4), 0), "chains_17": ((1,) * 17, 17), "rows_17": ((9, 8), 2),
                 "empty_chain": ((4, 0), 2)}
        for name, (layout, chains) in cases.items():
            with self.subTest(case=name):
                rings = x["rings"].clone()
                rc = oracle.call_status("cere_ktest_qsa_chains_step", x["k_idx"], x["k_norm"], EPS, x["cos_sin"],
                                        x["positions"], torch.tensor(layout, dtype=torch.int64), chains, x["slots"],
                                        SLOTS, rings, x["keys"].clone(), KEY_ROWS, x["undo_rings"].clone(),
                                        x["undo_keys"].clone(), 1, 1)
                oracle.report_row(ROW, f"chains_step/arguments/{name}", phase="result", status=rc)
                self.assertEqual(rc, ERR_ARGUMENT, name)
                oracle.check_bit_exact(self, ROW, f"chains_step/arguments/{name}/rings", rings, x["rings"])


if __name__ == "__main__":
    unittest.main()
