"""M2.15 D3: CPU model of one OMMA m16n8k64 (mxf4nvf4, scale_vec::4X) output element.

Rule (`ROUNDING = "trunc35-rz"`, established on GPU1 by D2 (c); it amends the probe's RZ(exact), which only held on
the F3 domain): (1) each K block j's 16 e2m1 products are summed exactly and the sum is multiplied exactly by
sa_j * sb_j, giving the block term T_j. (2) Block j is active when it holds at least one non-zero product
a_k * b_k and both its scales are non-zero; a block whose products cancel to 0 stays active. An active block's
alignment exponent is Ea_j + Eb_j, the unbiased exponents of its two ue4m3 scales (a subnormal scale counts as -6),
whatever T_j's magnitude; C's, when C != 0, is floor(log2 |C|). (3) With Emax the largest alignment exponent, every
T_j and C is truncated toward zero to a multiple of 2^(Emax - 35) (`WINDOW`). (4) The truncated terms are summed
exactly and the sum is rounded toward zero to FP32: subnormal results on the 2^-149 grid, a magnitude at or above
2^128 gives the largest finite FP32 with the sum's sign. An exact-zero result is always +0, whatever the signs of C
and the products (no -0 was ever observed natively).

Exactness: the block terms are exact binary64 values (a 12-bit block sum times an 8-bit scale product) and the
truncation is exact. Each truncated addend is a multiple of 2^(Emax - 35) below 2^(Emax + 12), so the D3 fit test
(sum |addend| < 2^(lo + 53), lo the smallest addend lsb) holds and the float path is exact; `exact_sum` keeps the
scaled-int path as the D3 guard.

Core is stdlib only. `dot64_t`/`chain_t` vectorize the same formulas in torch float64, importing torch lazily.
"""

from __future__ import annotations

import math

ROUNDING = "trunc35-rz"
WINDOW = 35

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_F32_MAX = math.ldexp((1 << 24) - 1, 104)


def e2m1(code: int) -> float:
    if not 0 <= code <= 0xF:
        raise ValueError(f"e2m1 code out of range: {code:#x}")
    v = _E2M1[code & 7]
    return -v if code & 8 else v


def ue4m3(code: int) -> float:
    """Unsigned e4m3 scale byte; only 0x00-0x7E (0x7F is NaN, 0x80-0xFF signed/NaN) are accepted."""
    if not 0 <= code <= 0x7E:
        raise ValueError(f"ue4m3 scale byte refused: {code:#x}")
    e, m = code >> 3, code & 7
    return math.ldexp(m, -9) if e == 0 else math.ldexp(8 + m, e - 10)


def _scale_exp(code: int) -> int:
    """Unbiased exponent of an accepted ue4m3 byte (subnormals: -6)."""
    return max(code >> 3, 1) - 7


def _lsb(x: float) -> int:
    n, d = x.as_integer_ratio()
    return (abs(n) & -abs(n)).bit_length() - d.bit_length()


def round_f32(n: int, e: int) -> float:
    """FP32 value of n * 2^e under round-toward-zero (n == 0 gives +0)."""
    if n == 0:
        return 0.0
    m = abs(n)
    top = m.bit_length() - 1 + e
    if top > 127:
        r = _F32_MAX
    else:
        q = max(top - 23, -149)
        if e < q:
            m >>= q - e
            e = q
        r = math.ldexp(m, e)
    return -r if n < 0 else r


def exact_sum(addends) -> tuple[str, int, int]:
    """Exact sum of binary64 addends as (path, n, e) with value n * 2^e; path "float" or "int" per the fit rule."""
    nz = [x for x in addends if x != 0.0]
    if not nz:
        return "float", 0, 0
    lo = min(_lsb(x) for x in nz)
    scaled = []
    for x in nz:
        n, d = x.as_integer_ratio()
        scaled.append((n << -lo) // d if lo <= 0 else n // (d << lo))
    if sum(abs(s) for s in scaled) < (1 << 53):
        s = addends[0]
        for x in addends[1:]:
            s += x
        n, d = s.as_integer_ratio()
        return "float", n, 1 - d.bit_length()
    return "int", sum(scaled), lo


def _addends(a, b, sa, sb, c: float) -> list[float]:
    """The truncated block terms and C of the rule (steps 1-3)."""
    if len(a) != 64 or len(b) != 64 or len(sa) != 4 or len(sb) != 4:
        raise ValueError("dot64 needs a[64], b[64], sa[4], sb[4]")
    c = float(c)
    terms, anchors = [], []
    for j in range(4):
        s = ue4m3(sa[j]) * ue4m3(sb[j])
        acc, active = 0.0, False
        for k in range(16 * j, 16 * j + 16):
            p = e2m1(a[k]) * e2m1(b[k])
            active = active or (p != 0.0 and s != 0.0)
            acc += p
        terms.append(acc * s)
        if active:
            anchors.append(_scale_exp(sa[j]) + _scale_exp(sb[j]))
    terms.append(c)
    if c != 0.0:
        anchors.append(math.frexp(c)[1] - 1)
    if not anchors:
        return []
    cut = max(anchors) - WINDOW
    return [math.ldexp(math.trunc(math.ldexp(x, -cut)), cut) for x in terms]


def dot64(a, b, sa, sb, c: float) -> float:
    """One OMMA output element: a, b are 64 e2m1 codes, sa, sb 4 ue4m3 bytes, c an FP32 value."""
    _, n, e = exact_sum(_addends(a, b, sa, sb, c) or [0.0])
    return round_f32(n, e)


def chain(a, b, sa, sb, c0: float = 0.0) -> float:
    """K-long chain in 64-wide steps; each step's C is the previous step's FP32 output."""
    k = len(a)
    if k == 0 or k % 64 or len(b) != k or len(sa) != k // 16 or len(sb) != k // 16:
        raise ValueError("chain needs K a multiple of 64 and K/16 scales")
    c = float(c0)
    for s in range(0, k, 64):
        c = dot64(a[s:s + 64], b[s:s + 64], sa[s // 16:s // 16 + 4], sb[s // 16:s // 16 + 4], c)
    return c


def _pow2(torch, e):
    """Exact 2^e as float64 for an int64 tensor e in [-1022, 1023]."""
    return ((e + 1023) << 52).view(torch.float64)


def dot64_t(a, b, sa, sb, c):
    """Vectorized dot64: a, b uint8[N,64] e2m1 codes (one per byte), sa, sb uint8[N,4], c float32[N] -> float32[N]."""
    import torch

    dev = a.device
    n = a.shape[0]
    if tuple(a.shape) != (n, 64) or tuple(b.shape) != (n, 64) or tuple(sa.shape) != (n, 4) \
            or tuple(sb.shape) != (n, 4) or tuple(c.shape) != (n,):
        raise ValueError("dot64_t needs a, b [N,64], sa, sb [N,4], c [N]")
    if bool((a > 0xF).any()) or bool((b > 0xF).any()):
        raise ValueError("e2m1 code out of range")
    if bool((sa > 0x7E).any()) or bool((sb > 0x7E).any()):
        raise ValueError("ue4m3 scale byte refused (> 0x7E)")
    f64, i64 = torch.float64, torch.int64
    ev = torch.tensor([e2m1(i) for i in range(16)], dtype=f64, device=dev)
    sv = torch.tensor([ue4m3(i) for i in range(0x7F)], dtype=f64, device=dev)
    se = torch.tensor([_scale_exp(i) for i in range(0x7F)], dtype=i64, device=dev)
    sai, sbi = sa.long(), sb.long()
    prod = (ev[a.long()] * ev[b.long()]).view(n, 4, 16)
    s = sv[sai] * sv[sbi]
    terms = prod.sum(dim=2) * s                                   # exact block terms [N,4]
    active = (prod != 0).any(dim=2) & (s != 0)
    none = torch.full((n, 4), -(1 << 20), dtype=i64, device=dev)
    anchor = torch.where(active, se[sai] + se[sbi], none).max(dim=1).values
    cd = c.to(f64)
    cexp = torch.frexp(cd)[1].to(i64) - 1
    anchor = torch.maximum(anchor, torch.where(cd != 0, cexp, none[:, 0]))
    cut = torch.where(anchor == none[:, 0], torch.zeros_like(anchor), anchor - WINDOW)
    down, up = _pow2(torch, -cut), _pow2(torch, cut)
    terms = torch.trunc(terms * down[:, None]) * up[:, None]
    cd = torch.trunc(cd * down) * up
    tot = terms.sum(dim=1) + cd
    fit = terms.abs().sum(dim=1) + cd.abs() < _pow2(torch, cut + 53)  # lo >= cut: sufficient for the D3 fit
    r = tot.to(torch.float32)
    r = torch.where(r.to(f64).abs() > tot.abs(), torch.nextafter(r, torch.zeros_like(r)), r)
    r = torch.where(tot == 0, torch.zeros_like(r), r)
    rest = (~fit).nonzero().flatten().tolist()
    if rest:
        al, bl, sal, sbl, cl = (t[rest].tolist() for t in (a, b, sa, sb, c))
        vals = [dot64(*row) for row in zip(al, bl, sal, sbl, cl)]
        r[rest] = torch.tensor(vals, dtype=torch.float32, device=dev)
    return r


def chain_t(a, b, sa, sb, c0=None):
    """Vectorized chain: a, b uint8[N,K], sa, sb uint8[N,K/16], c0 float32[N] or None (zeros) -> float32[N]."""
    import torch

    n, k = a.shape
    if k == 0 or k % 64 or tuple(b.shape) != (n, k) or tuple(sa.shape) != (n, k // 16) \
            or tuple(sb.shape) != (n, k // 16):
        raise ValueError("chain_t needs K a multiple of 64 and K/16 scales")
    c = torch.zeros(n, dtype=torch.float32, device=a.device) if c0 is None else c0.to(torch.float32)
    for s in range(0, k, 64):
        c = dot64_t(a[:, s:s + 64].contiguous(), b[:, s:s + 64].contiguous(), sa[:, s // 16:s // 16 + 4],
                    sb[:, s // 16:s // 16 + 4], c)
    return c
