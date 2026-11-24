"""SdarAlgoPin (O324; tasks/selftune-spec.md ST6, DECISION (maintainer, 2026-11-13, SdarAlgoPin)): cere::sdar::Gemm's
cuBLASLt algorithm pins (executor/cuda/src/kernels/sdar/sdar_gemm.h). Part of suite `sdar_kernels` (GPU0-only:
`CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite sdar_kernels`), with /model = the DICE-8B checkpoint.

- Every pinned (n, k, m) of the built-in table gives the same output bit for bit as today's heuristic pick (no table)
  at the exact n and k of its shape, on 16 seeds of unit-scale synthetic operands, 16 seeds of large-magnitude
  activations (scale 64 with 8 outlier channels at +-2048, near the largest BF16 hidden values of a real decode
  [INFERENCE]) against synthetic weights, and 16 seeds of activations against the checkpoint's own weights of that
  shape (spread over the layers). On this host's target (the table's) each pick is the pin, not a fallback.
- The guard is fail-closed: a cuBLASLt version or a compute capability other than the table's, a config cuBLASLt
  rejects (an unknown algorithm id; a tile algorithm 21 does not offer) and a config that needs a workspace each fall
  back to the heuristic, with today's output. Removing any of the four fallbacks fails its case. (A tile id outside
  cuBLASLt's enum crashes cublasLtMatmulAlgoCheck rather than failing it, so the table holds only probed configs.)
- Shapes and m with no entry keep the heuristic.
- SdarOptIn (O320, lever `sdar_gemm` = changing, opt-in only): every entry of the changing table (split-K through a
  workspace; at an (n, k, m) the built-in table also pins, kv at m = 2..4, the changing entry replaces that pin) is
  picked as such on this host's target, gives the same bytes on a repeat and stays within the tolerance rule against
  an FP64 reference (the limit calibrated on the identical pick's own error, floor 4e-3), on the same three input
  families. Its output differs from the identical pick by design (split-K rounds otherwise: the first GPU0 run saw up
  to 250 BF16 ulp on outputs near zero, bench/reports/native-test-sdar_kernels-20261113T212251Z.json), so no
  per-element ulp bound applies. It never falls back: a cuBLASLt version or compute capability other than the
  table's, or a workspace smaller than an entry needs, is UNSUPPORTED. The `changing_k_short` mutant (its layouts drop
  the last 128 k channels) must fail the tolerance cases.

Entry points (libcere_ktest.so, executor/cuda/src/ktest/sdar/sdar_gemm_pin_ktest.cpp; modes in its header):
    int cere_ktest_sdar_gemm_pins(void *out /*int64 [cap, 4]: n, k, m, f32*/, int64_t cap,
                                  void *target /*int64 [6]: table version, cc major, minor, then the process's*/,
                                  cudaStream_t stream);
    int cere_ktest_sdar_gemm_changing_pins(void *out, int64_t cap, void *target, cudaStream_t stream);
    int cere_ktest_sdar_gemm_pin(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y, int64_t f32,
                                 int64_t mode, void *use /*int64 [1]: PinUse*/, cudaStream_t stream);
"""
from __future__ import annotations

import json
import unittest

import torch
from safetensors import safe_open

import oracle

ROW = "SDAR_PIN"
SEEDS = 16
CAP = 64
OUTLIERS = 8
# cere::sdar::PinUse
HEURISTIC, PINNED, VERSION, ARCH, REJECTED, WORKSPACE, CHANGING = range(7)
# cere_ktest_sdar_gemm_pin modes
(OWN, NO_TABLE, OTHER_VERSION, OTHER_ARCH, BAD_ALGO, UNOFFERED_TILE, SPLIT_K, CHANGING_OWN, CHANGING_VERSION,
 CHANGING_ARCH, CHANGING_SMALL_WORKSPACE) = range(11)
KV = (1024, 4096)  # n, k of the kv GEMM (sdar_model.cpp plans); its split-K config needs a workspace at m = 2
NO_PIN = ((1024, 4096, 1, 0), (4096, 4096, 4, 0), (24576, 4096, 4, 0), (4096, 12288, 4, 0))


def _randn(shape, seed: int, scale: float, dtype=torch.bfloat16) -> torch.Tensor:
    g = oracle.generator(seed)
    return (torch.randn(shape, generator=g, device=oracle.DEVICE, dtype=torch.float32) * scale).to(dtype)


def _large(m: int, k: int, seed: int) -> torch.Tensor:
    """Scale-64 activations with OUTLIERS channels at +-2048 (BF16-exact), the same channels in every row."""
    x = _randn((m, k), seed, 64.0, torch.float32)
    g = oracle.generator(seed + 1000)
    cols = torch.randperm(k, generator=g, device=oracle.DEVICE)[:OUTLIERS]
    x[:, cols] = 2048.0 * torch.where(torch.rand((m, OUTLIERS), generator=g, device=oracle.DEVICE) < 0.5, -1.0, 1.0)
    return x.to(torch.bfloat16)


def _table(symbol: str, count: int, out: torch.Tensor,
           target: torch.Tensor) -> tuple[list[tuple[int, int, int, int]], list[int]]:
    if count < 0:
        raise oracle.OracleError(f"{symbol} returned status {count}")
    return [tuple(int(v) for v in out[i]) for i in range(count)], [int(v) for v in target]


def pins() -> tuple[list[tuple[int, int, int, int]], list[int]]:
    out = torch.zeros((CAP, 4), dtype=torch.int64)
    target = torch.zeros((6,), dtype=torch.int64)
    count = oracle.call_status("cere_ktest_sdar_gemm_pins", out, CAP, target)
    return _table("cere_ktest_sdar_gemm_pins", count, out, target)


def changing_pins() -> tuple[list[tuple[int, int, int, int]], list[int]]:
    out = torch.zeros((CAP, 4), dtype=torch.int64)
    target = torch.zeros((6,), dtype=torch.int64)
    count = oracle.call_status("cere_ktest_sdar_gemm_changing_pins", out, CAP, target)
    return _table("cere_ktest_sdar_gemm_changing_pins", count, out, target)


def gemm(x: torch.Tensor, w: torch.Tensor, n: int, k: int, m: int, f32: int, mode: int) -> tuple[torch.Tensor, int]:
    """(y, PinUse) of one cere_ktest_sdar_gemm_pin call; a non-zero status fails."""
    y = oracle.empty((m, n), torch.float32 if f32 else torch.bfloat16)
    use = torch.full((1,), -1, dtype=torch.int64)
    oracle.call("cere_ktest_sdar_gemm_pin", x, w, m, n, k, y, f32, mode, use)
    return y, int(use[0])


def checkpoint_weights(n: int, k: int) -> list[tuple[str, torch.Tensor]]:
    """Up to SEEDS checkpoint tensors of shape [n, k] (spread over the layers), on GPU0."""
    index = json.loads((oracle.MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    found: list[tuple[str, str]] = []
    for file in sorted(set(index.values())):
        with safe_open(str(oracle.MODEL_DIR / file), framework="pt", device="cpu") as f:
            found += [(name, file) for name in f.keys() if tuple(f.get_slice(name).get_shape()) == (n, k)]
    found.sort()
    step = max(1, len(found) // SEEDS)
    out = []
    for name, file in found[::step][:SEEDS]:
        with safe_open(str(oracle.MODEL_DIR / file), framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        if t.dtype != torch.bfloat16:
            raise oracle.OracleError(f"{name} is {t.dtype}, not BF16")
        out.append((name, t.to(oracle.DEVICE).contiguous()))
    return out


class SdarGemmPin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries, cls.target = pins()
        cls.on_target = cls.target[:3] == cls.target[3:]

    def _same(self, case: str, x: torch.Tensor, w: torch.Tensor, n: int, k: int, m: int, f32: int, mode: int,
              want: int) -> None:
        ref, ref_use = gemm(x, w, n, k, m, f32, NO_TABLE)
        self.assertEqual(ref_use, HEURISTIC, f"{case}: the reference must be today's heuristic pick")
        got, use = gemm(x, w, n, k, m, f32, mode)
        self.assertEqual(use, want, f"{case}: picked by PinUse {use}, expected {want}")
        oracle.check_bit_exact(self, ROW, case, got, ref)

    def test_table(self):
        entries, target = pins()
        self.assertEqual((entries, target), (self.entries, self.target), "the table changed between calls")
        self.assertTrue(len(entries) >= 1, "the built-in table is empty")
        self.assertEqual(len(set(entries)), len(entries), f"duplicate entries {entries}")
        listed = " ".join(f"n{n}/k{k}/m{m}/f32={f32}" for n, k, m, f32 in self.entries)
        oracle.report_row(ROW, "table", entries=listed, table_cublaslt=self.target[0],
                          table_cc=f"{self.target[1]}.{self.target[2]}", process_cublaslt=self.target[3],
                          process_cc=f"{self.target[4]}.{self.target[5]}", on_target=self.on_target)

    def test_pins_are_bit_identical(self):
        want = PINNED if self.on_target else (VERSION if self.target[0] != self.target[3] else ARCH)
        for n, k, m, f32 in self.entries:
            shape = f"n{n}/k{k}/m{m}"
            for seed in range(SEEDS):
                with self.subTest(shape=shape, seed=seed):
                    self._same(f"{shape}/unit/{seed}", _randn((m, k), seed, 1.0), _randn((n, k), seed + 500, 0.02),
                               n, k, m, f32, OWN, want)
                    self._same(f"{shape}/large/{seed}", _large(m, k, seed), _randn((n, k), seed + 600, 0.02),
                               n, k, m, f32, OWN, want)
            weights = checkpoint_weights(n, k)
            self.assertTrue(len(weights) >= 1, f"{shape}: the checkpoint has no [{n}, {k}] tensor")
            for seed, (name, w) in enumerate(weights):
                with self.subTest(shape=shape, weight=name):
                    self._same(f"{shape}/checkpoint/{name}", _large(m, k, seed + 700), w, n, k, m, f32, OWN, want)

    def test_guard_falls_back_to_the_heuristic(self):
        for n, k, m, f32 in self.entries:
            shape = f"n{n}/k{k}/m{m}"
            x, w = _large(m, k, 40), _randn((n, k), 41, 0.02)
            for mode, want, why in ((OTHER_VERSION, VERSION, "version"), (OTHER_ARCH, ARCH, "arch"),
                                    (BAD_ALGO, REJECTED, "algo"), (UNOFFERED_TILE, REJECTED, "tile")):
                with self.subTest(shape=shape, guard=why):
                    self._same(f"{shape}/fallback/{why}", x, w, n, k, m, f32, mode, want)
        n, k = KV
        with self.subTest(guard="workspace"):
            self._same("kv/m2/fallback/workspace", _large(2, k, 42), _randn((n, k), 43, 0.02), n, k, 2, 0, SPLIT_K,
                       WORKSPACE)

    def test_no_entry_keeps_the_heuristic(self):
        for n, k, m, f32 in NO_PIN:
            with self.subTest(shape=(n, k, m)):
                self.assertNotIn((n, k, m, f32), self.entries)
                self._same(f"n{n}/k{k}/m{m}/no_entry", _large(m, k, 50), _randn((n, k), 51, 0.02), n, k, m, f32,
                           OWN, HEURISTIC)

    def test_refusals(self):
        x, w = _randn((2, KV[1]), 60, 1.0), _randn(KV, 61, 0.02)
        y = oracle.empty((2, KV[0]), torch.bfloat16)
        use = torch.zeros((1,), dtype=torch.int64)
        for mode in (-1, 11):
            self.assertEqual(oracle.call_status("cere_ktest_sdar_gemm_pin", x, w, 2, KV[0], KV[1], y, 0, mode, use),
                             oracle.CERE_ERR_ARGUMENT)
        self.assertEqual(oracle.call_status("cere_ktest_sdar_gemm_pin", x, w, 0, KV[0], KV[1], y, 0, OWN, use),
                         oracle.CERE_ERR_ARGUMENT)
        out = torch.zeros((1, 4), dtype=torch.int64)
        target = torch.zeros((6,), dtype=torch.int64)
        self.assertEqual(oracle.call_status("cere_ktest_sdar_gemm_pins", out, 0, target), oracle.CERE_ERR_ARGUMENT)


class SdarGemmChanging(unittest.TestCase):
    """SdarOptIn's changing table (lever sdar_gemm = changing)."""

    @classmethod
    def setUpClass(cls):
        cls.entries, cls.target = changing_pins()

    def test_changing_table(self):
        self.assertTrue(len(self.entries) >= 1, "the changing table is empty")
        self.assertEqual(len(set(self.entries)), len(self.entries), f"duplicate entries {self.entries}")
        self.assertEqual(self.target[:3], self.target[3:], "this host is not the changing table's target")
        listed = " ".join(f"n{n}/k{k}/m{m}/f32={f32}" for n, k, m, f32 in self.entries)
        replaced = " ".join(f"n{n}/k{k}/m{m}" for n, k, m, _ in sorted(set(self.entries) & set(pins()[0])))
        oracle.report_row(ROW, "changing_table", entries=listed, table_cublaslt=self.target[0],
                          table_cc=f"{self.target[1]}.{self.target[2]}", process_cublaslt=self.target[3],
                          process_cc=f"{self.target[4]}.{self.target[5]}", replaced_pins=replaced)

    def _within(self, case: str, x: torch.Tensor, w: torch.Tensor, n: int, k: int, m: int, f32: int) -> None:
        ident, ident_use = gemm(x, w, n, k, m, f32, NO_TABLE)
        self.assertEqual(ident_use, HEURISTIC, f"{case}: the identical reference must be the heuristic pick")
        high = x.to(torch.float64) @ w.to(torch.float64).T
        c = oracle.calibrate(ROW, case, ident, high, oracle.FLOOR_OP)
        got, use = gemm(x, w, n, k, m, f32, CHANGING_OWN)
        self.assertEqual(use, CHANGING, f"{case}: picked by PinUse {use}, expected the changing entry")
        again, _ = gemm(x, w, n, k, m, f32, CHANGING_OWN)
        oracle.check_bit_exact(self, ROW, f"{case}/repeat", again, got)
        oracle.check_within(self, ROW, case, oracle.rel_l2(got, high), c=c,
                            limit=oracle.calibrated_limit(c, oracle.FLOOR_OP))
        oracle.report_row(ROW, f"{case}/vs_identical", differing=int((got != ident).sum()), elements=got.numel())

    def test_changing_within_bound(self):
        for n, k, m, f32 in self.entries:
            shape = f"n{n}/k{k}/m{m}"
            for seed in range(SEEDS):
                with self.subTest(shape=shape, seed=seed):
                    self._within(f"{shape}/changing/unit/{seed}", _randn((m, k), seed, 1.0),
                                 _randn((n, k), seed + 500, 0.02), n, k, m, f32)
                    self._within(f"{shape}/changing/large/{seed}", _large(m, k, seed),
                                 _randn((n, k), seed + 600, 0.02), n, k, m, f32)
            weights = checkpoint_weights(n, k)
            self.assertTrue(len(weights) >= 1, f"{shape}: the checkpoint has no [{n}, {k}] tensor")
            for seed, (name, w) in enumerate(weights):
                with self.subTest(shape=shape, weight=name):
                    self._within(f"{shape}/changing/checkpoint/{name}", _large(m, k, seed + 700), w, n, k, m, f32)

    def test_changing_never_falls_back(self):
        for n, k, m, f32 in self.entries:
            x, w = _large(m, k, 80), _randn((n, k), 81, 0.02)
            y = oracle.empty((m, n), torch.float32 if f32 else torch.bfloat16)
            for mode, why in ((CHANGING_VERSION, "version"), (CHANGING_ARCH, "arch"),
                              (CHANGING_SMALL_WORKSPACE, "workspace")):
                with self.subTest(shape=(n, k, m), guard=why):
                    use = torch.full((1,), -1, dtype=torch.int64)
                    status = oracle.call_status("cere_ktest_sdar_gemm_pin", x, w, m, n, k, y, f32, mode, use)
                    oracle.check_count(self, ROW, f"n{n}/k{k}/m{m}/changing_refused/{why}", status,
                                       expected=oracle.CERE_ERR_UNSUPPORTED)
        out = torch.zeros((1, 4), dtype=torch.int64)
        target = torch.zeros((6,), dtype=torch.int64)
        self.assertEqual(oracle.call_status("cere_ktest_sdar_gemm_changing_pins", out, 0, target),
                         oracle.CERE_ERR_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
