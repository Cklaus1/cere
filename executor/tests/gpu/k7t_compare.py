"""M2.15 D5 comparison helpers (stdlib only): FP32 bit equality, ulp distance and seeded sampling."""

from __future__ import annotations

import math
import random
import struct


def f32_bits(x: float) -> int:
    """FP32 bit pattern of `x` (rounded to FP32 first)."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _ordered(x: float) -> int:
    if math.isnan(x):
        raise ValueError("NaN has no ulp position")
    b = f32_bits(x)
    return -(b & 0x7FFFFFFF) if b & 0x80000000 else b


def ulp_distance(x: float, y: float) -> int:
    """FP32 ulps between x and y; +0 and -0 are 0 apart; NaN raises ValueError."""
    return abs(_ordered(x) - _ordered(y))


def _pairs(xs, ys):
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys) or not xs:
        raise ValueError("vectors must be non-empty and of equal length")
    return xs, ys


def bit_equal_fraction(xs, ys) -> float:
    """Fraction of positions whose FP32 bit patterns are identical; NaN raises ValueError."""
    xs, ys = _pairs(xs, ys)
    same = 0
    for x, y in zip(xs, ys):
        if math.isnan(x) or math.isnan(y):
            raise ValueError("NaN in compared vectors")
        same += f32_bits(x) == f32_bits(y)
    return same / len(xs)


def max_ulp(xs, ys) -> int:
    """Largest per-position ulp distance."""
    xs, ys = _pairs(xs, ys)
    return max(ulp_distance(x, y) for x, y in zip(xs, ys))


def sample_indices(seed: int, total: int, n: int) -> list[int]:
    """Deterministic sorted distinct indices in [0, total), min(n, total) of them."""
    if n >= total:
        return list(range(max(total, 0)))
    return sorted(random.Random(seed).sample(range(total), n))
