"""SelfTune ST4 S4a (tasks/selftune-spec.md ST4, D-S4-1/2/4): the e2e free run of family sdar on the DICE-8B
checkpoint, the GPU half of scripts/live/dice_judge.py. Not a test module (unittest never discovers it): the judge
runs it under `nsys profile --trace cuda,nvtx` in the GPU0 dev container, with /model = the DICE-8B checkpoint and
/out = the sdar build.

usage (in the container): python3 sdar_generate.py --out FILE [--limit N] [--contexts C,...] [--prompts FILE,...]
                           [--gen G] [--sdar-gemm identical|changing]

Per S2 reference prompt (tests/fixtures/dice8b_reference/primary.json, its recorded input ids) it calls
cere_ktest_sdar_generate (executor/cuda/src/ktest/sdar/sdar_generate_ktest.cpp): the device's schedule over the
device's forward, free to the schedule's full length (the reference's block_plan: no stop token ends it), inside an
NVTX range named by the prompt id. FILE gets, per prompt: the generated ids, per decode block the record of each
denoise (unmask index, then id and tie count of rows 0..3; -1 after the block's last) and the host tally (prefill
forwards, prefill tokens, then per block its denoise and commit forwards). The weights are loaded once, as BF16 on
GPU0 (test_sdar_kernels_model.load_weights).

`--contexts C,...` (SelfTune ST3 S3, D-ST3-7: dice_judge.py `bytes`, a measurement only) adds, after the reference
prompts, one synthetic prompt per C: C - G tokens cut from the recorded prompts' ids concatenated and repeated, then G
generated tokens (`--gen`, default 16), so its decode forwards run at a context of about C tokens. Its row id and
NVTX range are `ctx-<C>`. C and G must be multiples of 4 (the schedule's block) with G < C.

`--prompts FILE,...` (SelfTune ST6 S2, D-ST6-5: a named long-context workload, scripts/tune/dice_longctx.py) adds each
row of each JSONL set after them: its messages encoded as the S2 reference encodes a prompt (the checkpoint's
Qwen2Tokenizer and chat template with the generation prompt, no special tokens added), refused unless the count is the
row's `prompt_tokens`, then `--gen` generated tokens. Its row id and NVTX range are the set's id (`8k-<k>`, `32k-<k>`).

`--sdar-gemm changing` (SdarOptIn, O320; dice_judge.py `--config` of a profile with `sdar_gemm = "changing"`) runs
the Gemms on the lever's changing split-K table instead of the default identical pins.

Entry point (libcere_ktest.so; host arrays except the tensors):
    int cere_ktest_sdar_generate(const void *tensors, const void *names, const void *dims, int64_t count,
                                 const void *prompt /*uint32 [prompt_len]*/, int64_t prompt_len, int64_t gen_len,
                                 int64_t mask_id, int64_t sdar_gemm /*CERE_LEVER_SDAR_GEMM_* */, const char *label,
                                 void *gen /*uint32 [gen_len]*/, void *rec /*int32 [blocks, 4, 9]*/,
                                 void *tally /*int32 [2 + 2 * blocks]*/, cudaStream_t stream);
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import oracle
from test_sdar_kernels_model import BLOCK, RECORD_INTS, RECORDING, load_weights

GEN_LENGTH = 256
SYNTH_GEN = 16
SDAR_GEMM = {"identical": 0, "changing": 1}  # CERE_LEVER_SDAR_GEMM_* (executor/include/cere_executor.h)


def plan(prompt_len: int, gen_length: int = GEN_LENGTH) -> tuple[int, int]:
    """(generated length, decode blocks) of the reference's schedule for a prompt of P tokens (dice_reference.py
    block_plan: the generation fills whole blocks past the prompt's last, partial or not)."""
    total = (prompt_len + gen_length + BLOCK - 1) // BLOCK * BLOCK
    return total - prompt_len, (total - prompt_len // BLOCK * BLOCK) // BLOCK


def synthetic_rows(rows: list[dict], contexts: list[int], gen: int) -> list[dict]:
    """One synthetic row per context C: C - gen prompt ids from the recorded prompts concatenated and repeated."""
    pool = [t for r in rows for t in r["prompt_ids"]]
    out = []
    for c in contexts:
        if c % BLOCK or gen % BLOCK or not 0 < gen < c:
            raise ValueError(f"--contexts {c} / --gen {gen}: need multiples of {BLOCK} with 0 < gen < context")
        n = c - gen
        out.append({"id": f"ctx-{c}", "prompt_ids": (pool * (n // len(pool) + 1))[:n], "gen": gen})
    return out


def encode(tok, messages: list[dict]) -> list[int]:
    """The S2 reference's prompt ids (scripts/native/dice_reference/run_in_container.py)."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def prompt_rows(paths: list[Path], gen: int, model: Path = Path("/model")) -> list[dict]:
    """One row per line of each long-context set, its ids encoded from its messages; refuses a count other than the
    row's `prompt_tokens`."""
    if not paths:
        return []
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import Qwen2Tokenizer
    tok = Qwen2Tokenizer.from_pretrained(str(model), trust_remote_code=False, local_files_only=True)
    out = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            ids = encode(tok, row["messages"])
            if len(ids) != row["prompt_tokens"]:
                raise ValueError(f"{path.name} {row['id']}: {len(ids)} prompt tokens, the set records "
                                 f"{row['prompt_tokens']}")
            if gen % BLOCK or len(ids) + gen + BLOCK > row["context"]:
                raise ValueError(f"{row['id']}: {len(ids)} + --gen {gen} does not fit context {row['context']}")
            out.append({"id": row["id"], "prompt_ids": ids, "gen": gen})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke)")
    ap.add_argument("--contexts", default="", help="synthetic contexts C,... after the prompts (bytes measurement)")
    ap.add_argument("--gen", type=int, default=SYNTH_GEN,
                    help="generated tokens of each synthetic context and long-context prompt")
    ap.add_argument("--prompts", default="", help="long-context JSONL sets FILE,... after them (dice_longctx.py)")
    ap.add_argument("--sdar-gemm", choices=sorted(SDAR_GEMM), default="identical",
                    help="the sdar_gemm lever the Gemms run (default identical)")
    args = ap.parse_args(argv)
    doc = json.loads(RECORDING.read_text())
    mask_id = int(doc["meta"]["mask_id"])
    if doc["meta"]["schedule"]["top_k"] != 1 or doc["meta"]["schedule"]["block_length"] != BLOCK:
        raise oracle.OracleError(f"{RECORDING.name}: schedule {doc['meta']['schedule']} is not top_k 1, block 4")
    rows = doc["rows"][: args.limit] if args.limit else doc["rows"]
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    rows = rows + synthetic_rows(doc["rows"], contexts, args.gen)
    rows = rows + prompt_rows([Path(p) for p in args.prompts.split(",") if p.strip()], args.gen)
    views, names, owners = load_weights()
    ptrs = torch.tensor([v.data_ptr() for v in views], dtype=torch.int64)
    name_bytes = torch.tensor(list(b"".join(n.encode() + b"\0" for n in names)), dtype=torch.uint8)
    dims = torch.tensor([[v.shape[0], v.shape[1] if v.dim() == 2 else 0] for v in views], dtype=torch.int64)
    out = []
    for row in rows:
        plen = len(row["prompt_ids"])
        gen_len, blocks = plan(plen, row.get("gen", GEN_LENGTH))
        prompt = torch.tensor(row["prompt_ids"], dtype=torch.int32)
        label = torch.tensor(list(row["id"].encode() + b"\0"), dtype=torch.uint8)
        gen = torch.zeros(gen_len, dtype=torch.int32)
        rec = torch.full((blocks, BLOCK, RECORD_INTS), -2, dtype=torch.int32)
        tally = torch.full((2 + 2 * blocks,), -2, dtype=torch.int32)
        oracle.call("cere_ktest_sdar_generate", ptrs, name_bytes, dims, len(views), prompt, plen, gen_len, mask_id,
                    SDAR_GEMM[args.sdar_gemm], label, gen, rec, tally)
        out.append({"id": row["id"], "prompt_len": plen, "gen_ids": gen.tolist(), "records": rec.tolist(),
                    "tally": tally.tolist()})
    del owners
    args.out.write_text(json.dumps({"schema": "dice8b-e2e-run-v1", "mask_id": mask_id, "sdar_gemm": args.sdar_gemm,
                                    "rows": out}) + "\n")
    print(f"sdar_generate: {len(out)} prompts -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
