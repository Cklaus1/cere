#!/usr/bin/env python3
"""VulkanAmd P3 T5: compare two forward_s3 traces (executor/vulkan/tests/forward_s3.cpp).

usage: forward_s3_compare.py REF.trace CAND.trace [--identical]

Per prompt: (a) the first step where the greedy ids differ; (b) the top-1 logit difference per step while both are on
the same prefix (max and median of |d| and |d| / |ref|); (c) at the first differing step, REF's top-1 / top-2 margin
(a near-tie explains a late divergence; a large margin is a bug to hunt). --identical: every line of the two traces
other than the header and timing lines must be byte-equal (determinism); exit 1 otherwise.
"""
import statistics
import sys


def load(path):
    steps, sampled, other = {}, {}, []
    for line in open(path):
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if f[0] == "step":
            name, k = f[1], int(f[2])
            steps.setdefault(name, []).append(
                dict(k=k, tok=int(f[3]), id=int(f[4]), ru=int(f[5]), logit=float(f[6]), ru_logit=float(f[7]),
                     bits=(f[8], f[9])))
        elif f[0] == "sampled":
            sampled.setdefault(f[1], []).append((int(f[3]), f[4]))
        elif f[0] != "timing":
            other.append(line)
    return steps, sampled


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ref, cand = load(sys.argv[1]), load(sys.argv[2])
    if "--identical" in sys.argv:
        a = [l for l in open(sys.argv[1]) if l.startswith(("step", "sampled"))]
        b = [l for l in open(sys.argv[2]) if l.startswith(("step", "sampled"))]
        same = a == b
        print(f"identical step/sampled lines: {same} ({len(a)} vs {len(b)} lines)")
        return 0 if same else 1
    print(f"{'prompt':8} {'steps':>5} {'first diff':>10} {'max |dlogit|':>12} {'median':>8} {'max rel':>8} "
          f"{'ref margin @diff':>16} {'cand margin @diff':>17} {'min ref margin before':>21}")
    for name, rs in ref[0].items():
        cs = cand[0].get(name, [])
        n = min(len(rs), len(cs))
        first = next((k for k in range(n) if rs[k]["tok"] != cs[k]["tok"]), None)
        upto = first if first is not None else n
        # step k's logits score the token committed at k given the same prefix up to k (inclusive of the prompt)
        d = [abs(rs[k]["logit"] - cs[k]["logit"]) for k in range(upto + (1 if first is not None else 0)) if k < n]
        rel = [abs(rs[k]["logit"] - cs[k]["logit"]) / max(abs(rs[k]["logit"]), 1e-9) for k in range(len(d))]
        margin = lambda s: s["logit"] - s["ru_logit"]
        rm = f"{margin(rs[first]):.4f}" if first is not None else "-"
        cm = f"{margin(cs[first]):.4f}" if first is not None else "-"
        min_before = min((margin(rs[k]) for k in range(upto)), default=float("nan"))
        print(f"{name:8} {n:5d} {('none' if first is None else str(first)):>10} {max(d):12.4f} "
              f"{statistics.median(d):8.4f} {max(rel):8.4f} {rm:>16} {cm:>17} {min_before:21.4f}")
        if first is not None:
            r, c = rs[first], cs[first]
            print(f"    step {first}: ref top-2 {r['id']}/{r['ru']} logits {r['logit']:.4f}/{r['ru_logit']:.4f}; "
                  f"cand top-2 {c['id']}/{c['ru']} logits {c['logit']:.4f}/{c['ru_logit']:.4f}")
    for name, rsamp in ref[1].items():
        csamp = cand[1].get(name, [])
        same = sum(1 for a, b in zip(rsamp, csamp) if a[0] == b[0])
        print(f"sampled {name}: {same}/{min(len(rsamp), len(csamp))} draws equal (cross-device: not a contract)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
