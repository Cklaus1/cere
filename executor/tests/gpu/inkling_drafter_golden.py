"""Inkling 2 Mini S9 phase B (tasks/inkling-spec.md 3.5, D-I-S9B): the drafter-golden row and the drafter's timing.
Not a test module (unittest never discovers it): it runs in the GPU0 dev container with /model = Inkling-2-Mini@b85c3cfa
(mtp.safetensors + the target's embed / embed_norm / unembed), /out = the inkling-2-mini build and /golden = the S9
reference run (raw/<prompt>.safetensors of scripts/native/inkling_reference/drafter_ref.py: ids [T] and the bf16_in taps
[T, 4, 2304] entering visits 5/23/41 + the final norm), all read-only:

    cd /src/executor/tests/gpu && python3 inkling_drafter_golden.py --out /work/drafter_golden.json

Golden: per prompt, CERE's Drafter (model/inkling/drafter.h) replays drafter_ref.py simulate() on the fixture's block
starts (tests/fixtures/inkling2_reference/drafter.json, bf16_in): context() of the rows committed since the previous
block (f_rnn at the last one), then an eager draft() of 7 drafts at L with anchor ids[L], its step logits probed. A
block agrees when its 7 drafts equal the fixture's; a disagreeing block is BF16 noise when the top-2 gap of the step
logits at its first differing draft is within 2 BF16 ulps of the top logit (the fixture's recorded top-2 for the first
3 blocks of a prompt, CERE's probed logits otherwise). Rows "INKLING_DRAFTER_GOLDEN": per prompt the agreement rate,
the noise and beyond-noise counts and the accept length CERE's drafts would get against the target's greedy ids.

Timing (--time): context() of 6 and 4,096 rows, draft() B = 1, n_drafts = 5 at L = 4,096, eager and graph (host wall
per synchronized call, and CUDA-event time per call back to back).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch
from safetensors import safe_open

import oracle
from test_inkling_kernels_drafter import (BF, DEV, H, LOGIT_LD, VOCAB, DrafterHarness, _bf, _gen, _host,
                                          drafter_sizes)

ROW = "INKLING_DRAFTER_GOLDEN"
GAMMA = 7
FIXTURE = oracle.SRC / "tests" / "fixtures" / "inkling2_reference" / "drafter.json"


def bf16_ulp(x: float) -> float:
    a = abs(x)
    return 2.0 ** (math.floor(math.log2(a)) - 7) if a > 0 else 2.0 ** -133


def top2_gap(row: torch.Tensor) -> tuple[float, float]:
    v = row[:VOCAB].float().topk(2).values.tolist()
    return v[0], v[0] - v[1]


def golden(d: DrafterHarness, raw_dir: str, prompts: list[str] | None) -> dict:
    fx = json.loads(FIXTURE.read_text())
    out = {}
    tot = {"blocks": 0, "equal": 0, "noise": 0, "beyond": 0, "acc_cere": 0, "acc_fixture": 0}
    for pid, p in fx["prompts"].items():
        if prompts and pid not in prompts:
            continue
        c = p["bf16_in"]
        with safe_open(f"{raw_dir}/{pid}.safetensors", framework="pt", device="cpu") as f:
            ids = f.get_tensor("ids").tolist()
            taps = f.get_tensor("taps").to(DEV).contiguous()
        T = len(ids)
        d.sessions(1, T + 8)
        prev = 0
        equal = noise = beyond = acc_cere = 0
        worst = []
        probe = torch.empty((GAMMA, LOGIT_LD), dtype=BF, device=DEV)
        for i, L in enumerate(c["L"]):
            n = L - prev
            d.context([0] * n, list(range(prev, L)), [r == L - 1 for r in range(prev, L)], taps[prev:L])
            prev = L
            got = d.draft([0], [ids[L]], [L], GAMMA, False, probe)[0].tolist()
            want = c["drafts"][i]
            a = 0
            while a < GAMMA and got[a] == ids[L + 1 + a]:
                a += 1
            acc_cere += a
            if got == want:
                equal += 1
                continue
            j = next(k for k in range(GAMMA) if got[k] != want[k])
            rec = c["logits"][i]["step_top"] if i < len(c["logits"]) else None
            if rec is not None:
                top, gap = rec["logits"][j][0], rec["logits"][j][0] - rec["logits"][j][1]
                src = "fixture"
            else:
                top, gap = top2_gap(_host(probe[j]))
                src = "cere"
            within = gap <= 2 * bf16_ulp(top)
            noise += within
            beyond += not within
            if not within:
                worst.append({"block": i, "L": L, "draft": j, "cere": got[j], "fixture": want[j], "gap": gap,
                              "gap_from": src})
        blocks = len(c["L"])
        acc_fx = sum(c["accepted"])
        r = {"blocks": blocks, "equal": equal, "noise": noise, "beyond": beyond,
             "agreement": round(equal / blocks, 4), "agreement_or_noise": round((equal + noise) / blocks, 4),
             "mean_accepted_cere": round(acc_cere / blocks, 4), "mean_accepted_fixture": round(acc_fx / blocks, 4),
             "beyond_noise": worst[:8]}
        out[pid] = r
        for k, v in (("blocks", blocks), ("equal", equal), ("noise", noise), ("beyond", beyond),
                     ("acc_cere", acc_cere), ("acc_fixture", acc_fx)):
            tot[k] += v
        oracle.report_row(ROW, pid, phase="golden", blocks=blocks, equal=equal, noise=noise, beyond=beyond,
                          agreement=r["agreement"], mean_accepted_cere=r["mean_accepted_cere"],
                          mean_accepted_fixture=r["mean_accepted_fixture"])
        print(pid, json.dumps(r), flush=True)
    b = max(tot["blocks"], 1)
    out["all"] = {"blocks": tot["blocks"], "agreement": round(tot["equal"] / b, 4),
                  "agreement_or_noise": round((tot["equal"] + tot["noise"]) / b, 4), "beyond": tot["beyond"],
                  "mean_accepted_cere": round(tot["acc_cere"] / b, 4),
                  "mean_accepted_fixture": round(tot["acc_fixture"] / b, 4)}
    oracle.report_row(ROW, "all", phase="golden", **{k: v for k, v in out["all"].items()})
    return out


def timing(d: DrafterHarness, reps: int) -> dict:
    g = _gen(30)
    L = 4096
    d.sessions(1, L + 8)
    taps = torch.cat([_bf((L, 3, H), 2.0, g), _bf((L, 1, H), 0.05, g)], 1).contiguous()
    out = {}
    res = torch.empty(2, dtype=torch.float64)

    def ctx_time(n: int) -> list[float]:
        pos = torch.arange(L - n, L, dtype=torch.int64)
        oracle.call("cere_ktest_inkling_drafter_context_time", torch.zeros(n, dtype=torch.int64), pos,
                    (pos == L - 1).to(torch.int64), taps[L - n:].contiguous(), 4 * H, n, reps, res)
        return [round(float(x), 4) for x in res]

    out["context_4096_ms"] = ctx_time(4096)  # also fills the caches the draft below reads
    out["context_6_ms"] = ctx_time(6)
    ids = torch.zeros((1, GAMMA), dtype=torch.int32, device=DEV)
    for graphs in (0, 1):
        oracle.call("cere_ktest_inkling_drafter_draft_time", torch.zeros(1, dtype=torch.int64),
                    torch.tensor([1234], dtype=torch.int64), torch.tensor([L], dtype=torch.int64), 1, 5, graphs, ids,
                    reps, res)
        out[f"draft_b1_n5_L4096_{'graph' if graphs else 'eager'}_ms"] = [round(float(x), 4) for x in res]
    oracle.report_row(ROW, "timing [host wall per synced call, event per call]", phase="timing",
                      **{k: json.dumps(v) for k, v in out.items()})
    print("timing", json.dumps(out), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/golden/raw")
    ap.add_argument("--prompts", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    d = DrafterHarness()
    res = {}
    try:
        ws, sb = drafter_sizes(4096)
        res["workspace_bytes"], res["draft_session_bytes_4096"] = ws, sb
        print("sizes", ws, sb, flush=True)
        if not args.no_golden:
            res["golden"] = golden(d, args.raw, args.prompts)
        if args.time:
            res["timing"] = timing(d, args.reps)
    finally:
        d.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
