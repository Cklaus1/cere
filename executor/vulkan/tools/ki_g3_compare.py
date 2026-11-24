#!/usr/bin/env python3
"""Inkling S11 G3 / G5 (tasks/inkling-spec.md section 7): the Vulkan forward (build/vulkan/ki_forward) against arm Q's
golden logits (tests/fixtures/inkling2_reference/golden.json + the golden-v2 Q safetensors S1 recorded).

  ki_g3_compare.py prep GOLDEN_JSON PROMPT DIR      -> DIR/PROMPT.ids (prompt ids), DIR/PROMPT.forced (Q greedy[:-1])
  ki_g3_compare.py compare GOLDEN_JSON GOLDEN_DIR PROMPT LOGITS [--spread REF_CPU_SPREAD_JSON] [--json OUT]
  ki_g3_compare.py g2prep TAPS_JSON TAPS_DIR PROMPT DIR -> DIR/PROMPT.g2.ids, .g2.forced, .g2w.ids (G2's two runs)
  ki_g3_compare.py g2 TAPS_JSON TAPS_DIR ATTN_JSON ATTN_ST PROMPT DEC_DUMP WHOLE_DUMP [--json OUT]  (G2, D-I-G2b)

LOGITS is ki_forward's output for `prep`'s files: T prompt rows then 23 forced rows, float32 [*, 200058]. Metrics as
run_golden.py's compare (kl(ref, ours) = sum p_ref (log p_ref - log p_ours)): prefill argmax agreement over all
positions (golden argmax_all), last-16 KL mean / max (prefill_last16), decode teacher-forced KL mean / argmax agreement
(decode_logits; row 0 is the last prompt row) and the greedy common prefix. Bars: G3 with S1's prefill reset
(bench/reports/inkling-s1-20261123T060843Z.md: argmax >= 0.8805, last-16 KL mean <= 0.1136, max <= 1.373), binding on
p07; with --spread (tests/fixtures/inkling2_reference/ref_cpu_spread.json, spec D-I-G3b) a prompt listed there takes
its own prefill bars (1.25x the reference's CPU-vs-GPU spread). With the A-prime files (D-I-G3c, PROPOSED, report-only:
golden_fp32.json, its golden dir, --spread ref_cpu_spread_fp32.json) a listed prompt also takes its decode bars
(bar_tf_kl_mean, bar_tf_argmax); `prep` against golden_fp32.json writes that golden's own forced ids.
Argmax agreement is tie-aware: a position agrees when our logit at Q's id equals our row max. The final softcap
(30 tanh(x / 30)) maps distinct pre-cap logits near 27 onto one float, so exact ties occur in Q's own rows too (p05_raw
prefill row 2: ids 220 and 149876 both 27.724176 in Q) and a tie's argmax is only the tie-break. The strict counts are
reported next to them (`*_strict`); the decode KL against Q's uncached teacher-forced rows is reported too.
numpy only (the safetensors file is parsed by hand)."""
import argparse
import json
import os
import struct
import sys

import numpy as np

V = 200058
BARS = {"prefill_argmax": 0.8805, "kl_last16_mean": 0.1136, "kl_last16_max": 1.373, "dec_kl_mean": 0.02, "dec_argmax": 22,
        "prefix": 16}


def load_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, e in hdr.items():
            if k == "__metadata__":
                continue
            dt = {"F32": np.float32, "I32": np.int32, "BF16": np.uint16}[e["dtype"]]
            a, b = e["data_offsets"]
            f.seek(base + a)
            x = np.frombuffer(f.read(b - a), dtype=dt).reshape(e["shape"])
            out[k] = (x.astype(np.uint32) << 16).view(np.float32) if e["dtype"] == "BF16" else x
    return out


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(ref, ours):
    lp, lq = log_softmax(ref), log_softmax(ours)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def prompt_rec(golden_json, prompt):
    g = json.load(open(golden_json))
    return g["prompts"][prompt], g["arms"]["Q"][prompt]


def agrees(ours, ids):
    """Tie-aware argmax agreement per row: ours[r, ids[r]] is the row max."""
    ids = np.asarray(ids)
    return ours[np.arange(len(ids)), ids] == ours.max(-1)

def prep(a):
    p, q = prompt_rec(a.golden_json, a.prompt)
    np.asarray(p["input_ids"], dtype="<i4").tofile(f"{a.dir}/{a.prompt}.ids")
    np.asarray(q["greedy_ids"][:-1], dtype="<i4").tofile(f"{a.dir}/{a.prompt}.forced")
    print(f"{a.prompt}: {len(p['input_ids'])} prompt ids, {len(q['greedy_ids']) - 1} forced ids")


def read_amax(path, T):
    """ki_forward --keep's OUT.amax: per prompt row int32 count, then the ids at the row max."""
    a = np.fromfile(path, dtype="<i4")
    rows, i = [], 0
    while i < a.size:
        n = int(a[i])
        rows.append(a[i + 1:i + 1 + n])
        i += 1 + n
    if len(rows) != T:
        sys.exit(f"{path}: expected {T} rows, got {len(rows)}")
    return rows


def compare(a):
    p, q = prompt_rec(a.golden_json, a.prompt)
    T, greedy = p.get("n_tokens") or len(p["input_ids"]), q["greedy_ids"]
    gold = load_st(f"{a.golden_dir}/Q/{a.prompt}.safetensors")
    ours = np.fromfile(a.logits, dtype="<f4")
    if ours.size % V:
        sys.exit("logits file is not a whole number of rows")
    ours = ours.reshape(-1, V)
    ties = read_amax(a.logits + ".amax", T) if os.path.exists(a.logits + ".amax") else None
    K = T if ties is None else ours.shape[0] - (len(greedy) - 1)  # prompt rows kept (--keep: the last K)
    if ours.shape[0] != K + len(greedy) - 1 or K < min(16, T):  # p05_raw: 5 prompt tokens
        sys.exit(f"expected {T + len(greedy) - 1} rows (or --keep >= 16 with .amax), got {ours.shape[0]}")
    if not np.isfinite(ours).all():
        print("NON-FINITE logits")
    pre, dec = ours[:K], np.concatenate([ours[K - 1:K], ours[K:]])
    gam = np.asarray(q["argmax_all"])
    if ties is None:
        pa = agrees(pre, gam)
        strict = pre.argmax(-1) == gam
    else:
        pa = np.array([g in t for g, t in zip(gam, ties)])
        strict = np.array([len(t) > 0 and t[0] == g for g, t in zip(gam, ties)])
    agree = float(pa.mean())
    agree_strict = float(strict.mean())
    k16 = kl(gold["prefill_last16"], pre[-len(gold["prefill_last16"]):])
    kd = kl(gold["decode_logits"], dec)
    kn = kl(gold["nocache_decode_logits"], dec) if "nocache_decode_logits" in gold else None
    da = agrees(dec, greedy)
    dagree = int(da.sum())
    dagree_strict = int((dec.argmax(-1) == np.asarray(greedy)).sum())
    prefix = next((i for i in range(len(greedy)) if not da[i]), len(greedy))
    r = {"prompt": a.prompt, "T": T, "prefill_argmax": round(agree, 5), "prefill_argmax_strict": round(agree_strict, 5),
         "kl_last16_mean": float(k16.mean()), "kl_last16_max": float(k16.max()), "dec_kl_mean": float(kd.mean()),
         "dec_kl_max": float(kd.max()), "dec_argmax": dagree, "dec_argmax_strict": dagree_strict, "prefix": prefix,
         "max_abs_last": float(np.abs(gold["prefill_last16"][-1] - pre[-1]).max())}
    if kn is not None:
        r["dec_kl_nocache_mean"], r["dec_kl_nocache_max"] = float(kn.mean()), float(kn.max())
    bars = dict(BARS)
    if a.spread:
        sp = json.load(open(a.spread))["prompts"].get(a.prompt)
        if sp is not None:
            bars.update(prefill_argmax=sp["bar_argmax"], kl_last16_mean=sp["bar_kl16_mean"], kl_last16_max=sp["bar_kl16_max"])
            if "bar_tf_kl_mean" in sp:  # D-I-G3c (A-prime, PROPOSED): decode bars from the same spread file
                bars.update(dec_kl_mean=sp["bar_tf_kl_mean"], dec_argmax=sp["bar_tf_argmax"])
    r["bars"] = {k: bars[k] for k in ("prefill_argmax", "kl_last16_mean", "kl_last16_max", "dec_kl_mean", "dec_argmax")}
    r["pass"] = {
        "prefill_argmax": r["prefill_argmax"] >= bars["prefill_argmax"],
        "kl_last16_mean": r["kl_last16_mean"] <= bars["kl_last16_mean"],
        "kl_last16_max": r["kl_last16_max"] <= bars["kl_last16_max"],
        "dec_kl_mean": r["dec_kl_mean"] <= bars["dec_kl_mean"],
        "dec_argmax": r["dec_argmax"] >= bars["dec_argmax"],
        "prefix": r["prefix"] >= BARS["prefix"],  # counted over prompts: >= 6 / 8 must hold
    }
    print(json.dumps(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    core = [k for k in r["pass"] if k != "prefix"]
    return 0 if all(r["pass"][k] for k in core) else 1


def st_tensors(path, keys):
    """The named tensors of a safetensors file (BF16 as float32), read lazily (S1's p07 taps are 1.2 GB)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for k in keys:
            e = hdr[k]
            dt = {"F32": np.float32, "I32": np.int32, "BF16": np.uint16}[e["dtype"]]
            a, b = e["data_offsets"]
            f.seek(8 + n + a)
            x = np.frombuffer(f.read(b - a), dtype=dt).reshape(e["shape"])
            out[k] = (x.astype(np.uint32) << 16).view(np.float32) if e["dtype"] == "BF16" else x
    return out


def rel_l2(a, b):
    """run_taps.py rel_l2: per visit (axis 0) ||a - b|| / ||a|| over every other axis, a the reference (one visit at a
    time in float64)."""
    out = []
    for v in range(len(a)):
        x, y = a[v].astype(np.float64), b[v].astype(np.float64)
        out.append(float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30)))
    return np.asarray(out)


def read_dump(path, n, prefill, chunk):
    """ki_forward's KI_DUMP of a run over n positions: steps of [48][M][H] BF16 in order, the first `prefill` positions
    in chunks of `chunk` rows (ki_forward --chunk), then one row per forced id. -> float32 [48][n][H] in position
    order."""
    raw = np.fromfile(path, dtype="<u2")
    per_pos = 48 * 2304
    if raw.size != n * per_pos:
        sys.exit(f"{path}: {raw.size / per_pos} positions, expected {n}")
    ours = np.empty((48, n, 2304), np.float32)
    off, pos = 0, 0
    while pos < n:
        m = min(chunk, prefill - pos) if pos < prefill else 1
        ours[:, pos:pos + m] = (raw[off:off + m * per_pos].astype(np.uint32) << 16).view(np.float32).reshape(48, m, 2304)
        off += m * per_pos
        pos += m
    return ours


def g2_prep(a):
    """Input files of the two G2 runs of a tap prompt: DIR/P.g2.ids (input_ids[:decode_from], prefilled) and
    DIR/P.g2.forced (input_ids[decode_from:], decoded one per step), DIR/P.g2w.ids (the prompt, prefilled whole)."""
    taps = json.load(open(a.taps_json))
    rec = taps["prompts"][a.prompt]
    ids = st_tensors(f"{a.taps_dir}/{a.prompt}.safetensors", ["input_ids"])["input_ids"].astype("<i4")
    start = rec["decode_from"]
    ids[:start].tofile(f"{a.dir}/{a.prompt}.g2.ids")
    ids[start:].tofile(f"{a.dir}/{a.prompt}.g2.forced")
    ids[:rec["n_prompt_tokens"]].tofile(f"{a.dir}/{a.prompt}.g2w.ids")
    print(f"{a.prompt}: {start} prefilled + {len(ids) - start} decoded, whole prompt {rec['n_prompt_tokens']}")


def g2(a):
    """G2 with D-I-G2b (tasks/inkling-spec.md section 7), the S5 judge's row g2 (executor/tests/gpu/inkling_forward.py
    g2_prompt) on ki_forward's KI_DUMPs:
    - DEC: the run over input_ids, decode_from positions prefilled (--chunk DEC_CHUNK), the rest teacher-forced one per
      step. Decoded positions vs Q's decode taps <= max(g2.bar[v], f x attn[v]); prefilled positions vs Q's whole taps
      <= max(pre[v], f x attn[v]), pre[v] = 2 x max(g2.spread_max[v], Q's chunk-512 spread[v]); attn[v] = REF's
      eager-vs-FP32-attention rel-L2 (attn_spread.json, f = 1.25).
    - WHOLE: the prompt prefilled (--chunk WHOLE_CHUNK) vs REF with FP32 attention (attn_spread.safetensors P.fp32)
      <= pre[v] unwidened. Every visit of each, else G2 fails."""
    taps = json.load(open(a.taps_json))
    rec = taps["prompts"][a.prompt]
    aj = json.load(open(a.attn_json))
    factor = aj.get("factor", 1.25)
    attn = np.asarray(aj["prompts"][a.prompt]["visit_rel_l2"] if "prompts" in aj else aj[a.prompt])
    n, start, n_pre = rec["n_tokens"], rec["decode_from"], rec["n_prompt_tokens"]
    bar = np.asarray(taps["g2"]["bar"])
    pre_bar = 2 * np.maximum(np.asarray(taps["g2"]["spread_max"]),
                             np.asarray(taps["g3_chunk_visit_rel_l2"]["512"]["visit_rel_l2"]))
    bar_dec, bar_pre = np.maximum(bar, factor * attn), np.maximum(pre_bar, factor * attn)
    q = st_tensors(f"{a.taps_dir}/{a.prompt}.safetensors", ["decode_visits", "whole_visits"])
    ours = read_dump(a.dec_dump, n, start, a.dec_chunk)
    dec = rel_l2(q["decode_visits"], ours[:, start:])
    pre = rel_l2(q["whole_visits"][:, :start], ours[:, :start]) if start > 0 else np.zeros(48)
    whole_golden = rel_l2(q["whole_visits"], ours)
    del q, ours
    fp32 = st_tensors(a.attn_st, [f"{a.prompt}.fp32"])[f"{a.prompt}.fp32"]
    if fp32.shape[1] != n_pre:
        sys.exit(f"attn taps hold {fp32.shape[1]} positions, prompt {n_pre}")
    vs_fp32 = rel_l2(fp32, read_dump(a.whole_dump, n_pre, n_pre, a.whole_chunk))
    r4 = lambda x: [round(float(v), 5) for v in x]
    fails = {"decode": [int(v) for v in np.nonzero(~(dec <= bar_dec))[0]],
             "prefill": [int(v) for v in np.nonzero(~(pre <= bar_pre))[0]],
             "fp32_attn": [int(v) for v in np.nonzero(~(vs_fp32 <= pre_bar))[0]]}
    r = {"prompt": a.prompt, "n": n, "decode_from": start, "n_prompt": n_pre, "dec_chunk": a.dec_chunk,
         "whole_chunk": a.whole_chunk, "fail_visits": fails,
         "decode_ratio_max": float((dec / bar_dec).max()), "prefill_ratio_max": float((pre / bar_pre).max()),
         "fp32_attn_ratio_max": float((vs_fp32 / pre_bar).max()),
         "decode_rel_l2": r4(dec), "prefill_rel_l2": r4(pre), "fp32_attn_rel_l2": r4(vs_fp32),
         "whole_vs_golden_rel_l2": r4(whole_golden), "bar_decode": r4(bar_dec), "bar_prefill": r4(bar_pre),
         "bar_fp32_attn": r4(pre_bar)}
    r["pass"] = not any(fails.values())
    print(f"g2 {a.prompt}: decode ratio max {r['decode_ratio_max']:.3f} fail {fails['decode']}; prefill ratio max "
          f"{r['prefill_ratio_max']:.3f} fail {fails['prefill']}; vs FP32 attention ratio max "
          f"{r['fp32_attn_ratio_max']:.3f} fail {fails['fp32_attn']} -> {'PASS' if r['pass'] else 'FAIL'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0 if r["pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("prep")
    a.add_argument("golden_json")
    a.add_argument("prompt")
    a.add_argument("dir")
    b = sp.add_parser("compare")
    b.add_argument("golden_json")
    b.add_argument("golden_dir")
    b.add_argument("prompt")
    b.add_argument("logits")
    b.add_argument("--spread")
    b.add_argument("--json")
    d = sp.add_parser("g2prep")
    d.add_argument("taps_json")
    d.add_argument("taps_dir")
    d.add_argument("prompt")
    d.add_argument("dir")
    c = sp.add_parser("g2")
    c.add_argument("taps_json")
    c.add_argument("taps_dir")
    c.add_argument("attn_json", help="attn_spread.json (the fixture or its host_dir copy)")
    c.add_argument("attn_st", help="attn_spread.safetensors (REF FP32-attention prompt taps)")
    c.add_argument("prompt")
    c.add_argument("dec_dump", help="KI_DUMP of P.g2.ids + --forced P.g2.forced")
    c.add_argument("whole_dump", help="KI_DUMP of P.g2w.ids")
    c.add_argument("--dec-chunk", type=int, default=512)
    c.add_argument("--whole-chunk", type=int, default=4096)
    c.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a)
        return 0
    if a.cmd == "g2prep":
        g2_prep(a)
        return 0
    return g2(a) if a.cmd == "g2" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
