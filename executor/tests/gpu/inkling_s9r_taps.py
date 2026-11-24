"""Inkling 2 Mini S9r (tasks/inkling-spec.md D-I-S9r): CERE's greedy target ids and drafter taps, the GPU half of
scripts/live/inkling_s9r.py `run`. Not a test module: the driver runs it in the GPU0 dev container with /model =
Inkling-2-Mini@b85c3cfa, /out = the inkling-2-mini build (libcere_executor.so bf112fc7), /golden = S1's host run
directory (p07's ids), /work read-write holding prompts.json ({pid: {"class", "ids" | null, "gen"}}; null ids = the
golden prompt's).

Per prompt: the device's greedy generation (cere_ktest_inkling_device_generate: ABI v5 executor, stop eos), then
CERE's forward over prompt + generation (prefill, chunks of 4,096) for the drafter's four taps: visit outputs 4, 22,
40 (target_layer_ids 5, 23, 41 are 1-based visit outputs, executor/cuda/src/model/inkling/plan.h:86) and 48 = the
final stream, hidden = rms_w(depth_mix([h_0, h_6, h_12, h_18, h_24], output_query), norm) (forward.h:20) over the
ktest's final block states and visit 47's output, recomputed here in FP32 from the same BF16 inputs.
Writes /work/cere/<pid>.safetensors (ids int32 [T], taps bf16 [T, 4, 2304], n_prompt) and /work/cere.json.

usage (in the container): python3 inkling_s9r_taps.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import oracle
from inkling_forward import DEV, EOS, H, Model, golden_prompts

WORK = Path("/work")
TAP_VISITS = (4, 22, 40)


def final_stream(model_dir: Path):
    with safe_open(str(model_dir / "model-00001-of-00001.safetensors"), framework="pt", device=str(DEV)) as f:
        q = f.get_tensor("model.llm.attn_res.output_query").float()
        w = f.get_tensor("model.llm.norm.weight").float()

    def run(blocks, h24):  # blocks bf16 [n, 4, H], h24 bf16 [n, H]
        s = torch.cat([blocks, h24[:, None]], 1).float()
        s = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + 1e-6)
        a = torch.softmax((s @ q) * H ** -0.5, -1)
        f = (a[..., None] * s).sum(1)
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (w * f).to(torch.bfloat16)
    return run


def main() -> int:
    torch.set_grad_enabled(False)
    spec = json.loads((WORK / "prompts.json").read_text())
    golden = golden_prompts()
    model = Model()
    final = final_stream(oracle.MODEL_DIR)
    out_dir = WORK / "cere"
    out_dir.mkdir(exist_ok=True)
    doc = {"schema": "inkling-s9r-cere-v1", "load_s": model.load_s, "prompts": {}}
    for pid, p in spec.items():
        ids = golden[pid] if p["ids"] is None else p["ids"]
        t0 = time.time()
        gen_ids, res, _, _, secs = model.device_generate([ids], gen=p["gen"])
        n = int(res[0, 0])
        gen = gen_ids[0, :n].tolist()
        seq = ids + gen
        fw = model.forward(seq, taps=True, logit_first=len(seq))
        taps = torch.stack([fw["visits"][v] for v in TAP_VISITS] + [final(fw["blocks"], fw["visits"][47])], 1)
        argmax_ok = fw["argmax"][len(ids) - 1:len(seq) - 1].cpu().tolist() == gen
        save_file({"ids": torch.tensor(seq, dtype=torch.int32), "taps": taps.contiguous().cpu(),
                   "n_prompt": torch.tensor([len(ids)], dtype=torch.int64)}, str(out_dir / f"{pid}.safetensors"))
        os.chmod(out_dir / f"{pid}.safetensors", 0o644)  # the reference container reads them as nobody
        doc["prompts"][pid] = {"class": p["class"], "n_prompt": len(ids), "n_generated": n,
                               "finish": int(res[0, 1]), "stopped_eos": bool(n and gen[-1] == EOS),
                               "prefill_argmax_equals_device": argmax_ok, "device_s": secs[0],
                               "wall_s": round(time.time() - t0, 2)}
        print(pid, json.dumps(doc["prompts"][pid]), flush=True)
        del fw, taps
        torch.cuda.empty_cache()
    (WORK / "cere.json").write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
