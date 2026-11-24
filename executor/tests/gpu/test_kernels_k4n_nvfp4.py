"""K4N: the K4 NVFP4 PLE row decode (tasks/fp4local-spec.md D8, D14, FL-13; `kernels/k4_nvfp4.h`).

Suite `kernels_k4n` (GPU1 lane: `native_dev.py --gpu 1 test --suite kernels_k4n`; also part of GPU0's `kernels`):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_k4n_*.py' -v`.

The FP4-Local checkpoint stores the PLE table as NVFP4 rows (D8): per 160-element row 80 bytes of packed E2M1 codes
(element 2j in the low nibble of byte j), 10 E4M3 block scales (one per 16 elements) and one F32 global for the whole
table. The executor reads all three from mapped pinned host memory (the artifact profile
`rtxpro6000bw-plehost-nvfp4-text-v1` places them `host_pinned`). Element i of a row is
bf16_rne(f32(e2m1(code_i) * e4m3(scale_{i / 16})) * global).

Reference: `decode_reference`, a CPU torch decode written from the format definitions (E2M1 by table, E4M3 through
torch's float8_e4m3fn, F32 products on the CPU, BF16 through torch's round-to-nearest-even; a NaN element compares
as the canonical 0x7FFF the device writes). The host test (executor/tests/host/test_k4_nvfp4.cpp) checks the header's
functions against a third, ldexp-based reference exhaustively; here the kernel is checked on sampled rows. Inputs are
synthetic and seeded: a table of 1,048,576 random rows (90 MiB pinned) whose first 26 rows hold all 256 scale bytes
(0, subnormals, 448, NaN, negatives), under F32 globals that are 1, not powers of two, a real-scale value, subnormal
products, overflowing and negative.

Native entry points (each returns 0 or a nonzero cere_status; the stream is the last argument; table, block_scales and
global_scale are host pointers of pinned allocations, device-accessible through UVA):

    int cere_ktest_ple_gather_nvfp4(const void *table, const void *block_scales, const void *global_scale,
                                    int64_t n_table_rows, const void *ids, int64_t n, void *out,
                                    cudaStream_t stream);
        table U8 [n_table_rows, 80]; block_scales E4M3 [n_table_rows, 10]; global_scale F32 [1]; ids int64 [n];
        out BF16 [n, 160] = the decoded rows; an id outside [0, n_table_rows) gives a zero row (scale byte 0).

    int cere_ktest_ple_embed(uint32_t format, const void *table, int64_t n_table_rows, const void *table_scale,
                             const void *block_scales, const void *global_scale, const void *multipliers,
                             const void *head_vocab_sizes, const void *head_offsets, const void *ngram_history,
                             const void *tokens, int64_t n_tokens, int64_t eos_token_id, void *emb,
                             cudaStream_t stream);
        The forward's embedding stage (k4::embed): format 0 = FP8 (E4M3 table x BF16 table_scale), 1 = NVFP4;
        emb BF16 [n_tokens, 2,560] = the 16 rows of each token's n-gram ids (from history ++ tokens), concatenated.
        k4::check_table refuses fields of the other format (an FP8 call with block scales or a global, an NVFP4 call
        with a BF16 table_scale, a missing block scale or global, a table not 8-byte aligned) and formats > 1.

    int cere_ktest_ple_hash(...): as in test_kernels_k4_ple.py (ids int64 [n, 16] of windows int64 [n, 3]).

Rows: `K4N` gather bit-exact per global and id case (random, the scale-coverage rows, shard-like boundaries, out of
range); the NVFP4 embed bit-exact vs the reference rows of the hashed ids; each refusal returns nonzero and leaves the
output unwritten, and the FP8 embed with FP8 fields runs.
"""
from __future__ import annotations

import functools
import struct
import unittest

import torch

import oracle

ROW = "K4N"
SEED = 0x4B344E
ROWS = 1 << 20
ROW_ELEMENTS = 160
ROW_BYTES = 80
ROW_SCALES = 10
BLOCK = 16
HEADS = 16
EMBED_DIM = HEADS * ROW_ELEMENTS
RANDOM_IDS = 65_536
SCALE_ROWS = 26  # 260 block scales >= the 256 E4M3 bytes
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float32)
BF16_NAN = 0x7FFF


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_of_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# name -> F32 global: 1, not a power of two, the order of a real NVFP4 global (amax / (6 x 448)), products in the F32
# subnormal range, products that overflow to +-inf, negative, -0.
GLOBALS = {
    "one": 1.0,
    "tenth": f32(0.1),
    "real_scale": f32(1.0 / (6.0 * 448.0)),
    "subnormal": f32_of_bits(0x00800000) / 2.0 ** 7,
    "overflow": f32(1.0e38),
    "negative": -1.0,
    "negative_zero": -0.0,
}


@functools.cache
def table() -> tuple[torch.Tensor, torch.Tensor]:
    """The synthetic NVFP4 table: packed U8 [ROWS, 80] and E4M3 scale bytes [ROWS, 10], both pinned host memory."""
    g = torch.Generator().manual_seed(SEED)
    packed = torch.randint(0, 256, (ROWS, ROW_BYTES), generator=g, dtype=torch.uint8)
    scales = torch.randint(0, 256, (ROWS, ROW_SCALES), generator=g, dtype=torch.uint8)
    scales[:SCALE_ROWS].view(-1)[:256] = torch.arange(256, dtype=torch.uint8)
    return packed.pin_memory(), scales.pin_memory()


def pinned_global(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32).pin_memory()


def decode_reference(packed: torch.Tensor, scales: torch.Tensor, global_value: float, ids: torch.Tensor) -> torch.Tensor:
    """BF16 [n, 160] rows `ids` of the table on the CPU; an id outside the table gives the decode of zero bytes."""
    ids = ids.cpu()
    valid = (ids >= 0) & (ids < packed.shape[0])
    safe = torch.where(valid, ids, torch.zeros_like(ids))
    rows = packed[safe] * valid.unsqueeze(1).to(torch.uint8)
    scale_bytes = scales[safe] * valid.unsqueeze(1).to(torch.uint8)
    codes = torch.stack([rows & 0xF, rows >> 4], dim=-1).reshape(ids.numel(), ROW_ELEMENTS).long()
    block = scale_bytes.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(BLOCK, dim=1)
    value = (E2M1[codes] * block) * torch.tensor(global_value, dtype=torch.float32)
    out = value.to(torch.bfloat16)
    out.view(torch.int16)[torch.isnan(value)] = BF16_NAN
    return out.to(oracle.DEVICE)


def ngram_inputs(g: torch.Generator, n_tokens: int):
    """Seeded hash inputs whose ids fall inside the table: multipliers, vocab sizes and offsets of 16 heads, a history
    of 2 and n_tokens tokens with EOS injected (1/8 per position), all on the device."""
    eos = 7
    vocab = torch.tensor([60_000 + 17 * h for h in range(HEADS)], dtype=torch.long)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), vocab[:-1]]), dim=0)
    if int(offsets[-1] + vocab[-1]) > ROWS:
        raise oracle.OracleError("n-gram heads exceed the synthetic table")
    multipliers = torch.randint(1, 1 << 62, (3,), generator=g, dtype=torch.long) | 1
    seq = torch.randint(0, 248_320, (n_tokens + 2,), generator=g, dtype=torch.long)
    seq[torch.rand(n_tokens + 2, generator=g) < 0.125] = eos
    dev = oracle.DEVICE
    return (multipliers.to(dev), vocab.to(dev), offsets.to(dev), seq[:2].contiguous().to(dev),
            seq[2:].contiguous().to(dev), seq, eos)


class K4NNvfp4(unittest.TestCase):
    def test_gather_bit_exact(self):
        packed, scales = table()
        g = torch.Generator().manual_seed(SEED + 1)
        cases = {
            "random": torch.randint(0, ROWS, (RANDOM_IDS,), generator=g, dtype=torch.long),
            "scale_rows": torch.arange(SCALE_ROWS, dtype=torch.long),
            "boundaries": torch.tensor(sorted({0, 1, ROWS // 128 - 1, ROWS // 128, ROWS // 2, ROWS - 2, ROWS - 1}),
                                       dtype=torch.long),
            "out_of_range": torch.tensor([-1, ROWS, ROWS + 1, -(1 << 40), 1 << 40], dtype=torch.long),
        }
        for name, value in GLOBALS.items():
            global_scale = pinned_global(value)
            for case, ids in cases.items():
                with self.subTest(global_scale=name, case=case):
                    ids_dev = ids.to(oracle.DEVICE).contiguous()
                    native = oracle.empty((ids.numel(), ROW_ELEMENTS), torch.bfloat16)
                    oracle.call("cere_ktest_ple_gather_nvfp4", packed, scales, global_scale, ROWS, ids_dev,
                                ids.numel(), native)
                    oracle.report_row(ROW, f"gather_{case}_{name}", phase="setup", n_ids=ids.numel())
                    oracle.check_bit_exact(self, ROW, f"gather_{case}_{name}", native,
                                           decode_reference(packed, scales, value, ids))

    def test_embed_bit_exact(self):
        packed, scales = table()
        g = torch.Generator().manual_seed(SEED + 2)
        for n_tokens in (1, 2, 3, 61):
            mult, vocab, offsets, history, tokens, seq, eos = ngram_inputs(g, n_tokens)
            windows = torch.stack([seq[t:t + 3] for t in range(n_tokens)]).contiguous().to(oracle.DEVICE)
            ids = oracle.empty((n_tokens, HEADS), torch.long)
            oracle.call("cere_ktest_ple_hash", windows, mult, vocab, offsets, ids, n_tokens, eos)
            for name in ("one", "real_scale"):
                with self.subTest(n_tokens=n_tokens, global_scale=name):
                    global_scale = pinned_global(GLOBALS[name])
                    native = oracle.empty((n_tokens, EMBED_DIM), torch.bfloat16)
                    oracle.call("cere_ktest_ple_embed", 1, packed, ROWS, None, scales, global_scale, mult, vocab,
                                offsets, history, tokens, n_tokens, eos, native)
                    reference = decode_reference(packed, scales, GLOBALS[name], ids.reshape(-1))
                    oracle.check_bit_exact(self, ROW, f"embed_t{n_tokens}_{name}", native,
                                           reference.reshape(n_tokens, EMBED_DIM))

    def test_embed_refuses_the_other_formats_fields(self):
        packed, scales = table()
        g = torch.Generator().manual_seed(SEED + 3)
        mult, vocab, offsets, history, tokens, _, eos = ngram_inputs(g, 4)
        global_scale = pinned_global(1.0)
        fp8_scale = torch.tensor([1.0], dtype=torch.bfloat16, device=oracle.DEVICE)
        fp8_table = torch.zeros((ROWS, ROW_ELEMENTS), dtype=torch.uint8).pin_memory()
        unaligned = packed.view(-1)[1:]
        refused = {
            # An NVFP4 table on the FP8 kernel: format 0 with the block scales and the global.
            "fp8_with_nvfp4_scales": (0, packed, None, scales, global_scale),
            "fp8_with_global_only": (0, fp8_table, fp8_scale, None, global_scale),
            "fp8_with_block_scales_only": (0, fp8_table, fp8_scale, scales, None),
            "nvfp4_with_bf16_scale": (1, packed, fp8_scale, scales, global_scale),
            "nvfp4_without_block_scales": (1, packed, None, None, global_scale),
            "nvfp4_without_global": (1, packed, None, scales, None),
            "nvfp4_unaligned_table": (1, unaligned, None, scales, global_scale),
            "format_2": (2, packed, None, scales, global_scale),
        }
        for case, (fmt, tab, tscale, bscales, gscale) in refused.items():
            with self.subTest(case=case):
                native = oracle.empty((4, EMBED_DIM), torch.bfloat16)
                rc = oracle.call_status("cere_ktest_ple_embed", fmt, tab, ROWS, tscale, bscales, gscale, mult, vocab,
                                        offsets, history, tokens, 4, eos, native)
                untouched = bool(torch.isnan(native).all().item())
                oracle.report_row(ROW, f"refuse_{case}", phase="result", status=rc, untouched=untouched,
                                  passed=rc != 0 and untouched)
                self.assertNotEqual(rc, 0, f"{case}: accepted")
                self.assertTrue(untouched, f"{case}: output written")
        rc = oracle.call_status("cere_ktest_ple_gather_nvfp4", packed, None, global_scale, ROWS,
                                torch.zeros(1, dtype=torch.long, device=oracle.DEVICE), 1,
                                oracle.empty((1, ROW_ELEMENTS), torch.bfloat16))
        oracle.report_row(ROW, "refuse_gather_without_block_scales", phase="result", status=rc, passed=rc != 0)
        self.assertNotEqual(rc, 0)
        native = oracle.empty((4, EMBED_DIM), torch.bfloat16)
        oracle.call("cere_ktest_ple_embed", 0, fp8_table, ROWS, fp8_scale, None, None, mult, vocab, offsets, history,
                    tokens, 4, eos, native)
        written = bool((native.view(torch.int16) == 0).all().item())
        oracle.report_row(ROW, "fp8_embed_runs", phase="result", passed=written)
        self.assertTrue(written, "the FP8 embed of a zero table with FP8 fields is not all +0")


if __name__ == "__main__":
    unittest.main()
