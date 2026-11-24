"""SelfTune ST4 S3c, condition 1 (replay): family sdar's whole forward (model/sdar/sdar_model.h CudaForward, the
device's) on the DICE-8B checkpoint, replayed along the S2 reference recording
`tests/fixtures/dice8b_reference/primary.json` (64 prompts, GPU0, the reference's default SDPA backends).

Suite `sdar_kernels` (GPU0-only), with /model = the DICE-8B checkpoint (native_dev.py SUITE_CHECKPOINTS). The weights
are loaded as BF16 onto GPU0 (each layer's gate_proj and up_proj into one [2 * 12288, 4096] buffer, as the device
places them).

What the recording holds per decode block: the position each denoise step unmasked (`steps`) and, for the steps whose
masked positions' confidences differ (`unequal`), those confidences. With the reference's top_k 1 a confidence is
1 / (the number of BF16-tied maximal logits of that row), so a step's tie counts over its masked rows are
round(1 / confidence) there and 1 on every other step. It holds no logits. So the replay runs teacher-forced
(cere_ktest_sdar_replay): every step sees the reference's block (after each step the recorded position receives the
recorded token, then the block's recorded tokens are committed), and per step CERE's tie counts and unmask index are
compared with the recorded ones.

Bar (condition 1, "at least where the logits agree"): on every step whose CERE tie counts over the masked rows equal
the recorded ones, CERE's unmask index is the recorded position (the selection: fewest ties, the lowest row among
equals). Precondition: such steps exist, multi-tie ones among them; otherwise the recording cannot support the bar and
the test fails saying so. Reported, not judged (the logits differ where the arithmetic does: cuBLASLt GEMMs and
CERE's attention vs torch's): the steps whose tie counts agree, overall and among the recorded multi-tie steps; the
index agreement over every step; the transferred token agreement where the recorded row was untied.

Entry point (libcere_ktest.so, executor/cuda/src/ktest/sdar/sdar_model_ktest.cpp; host arrays except the tensors):
    int cere_ktest_sdar_replay(const void *tensors /*uint64 [count] device pointers*/,
                               const void *names /*NUL-terminated names back to back*/,
                               const void *dims /*int64 [count, 2], 1-D: second 0*/, int64_t count,
                               const void *tokens /*uint32 [total]: prompt, then generation*/, int64_t prompt_len,
                               int64_t total, const void *order /*int32 [blocks, 4]: position per step, -1 after*/,
                               void *out /*int32 [blocks, 4, 9]: index, then id and ties of rows 0..3*/,
                               int64_t mask_id, cudaStream_t stream);
"""
from __future__ import annotations

import json
import unittest

import torch
from safetensors import safe_open

import oracle

ROW = "SDAR"
BLOCK = 4
INTER = 12288
RECORDING = oracle.SRC / "tests/fixtures/dice8b_reference/primary.json"
RECORD_INTS = 1 + 2 * BLOCK


def load_weights() -> tuple[list[torch.Tensor], list[str], list[torch.Tensor]]:
    """(views passed to the entry point, their names, the buffers that own them) of every checkpoint tensor."""
    index = json.loads((oracle.MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for file in sorted(set(index.values())):
        with safe_open(str(oracle.MODEL_DIR / file), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    views, names, owners = [], [], []
    for name in sorted(tensors):
        if name.endswith("mlp.up_proj.weight"):
            continue
        t = tensors[name]
        if t.dtype != torch.bfloat16:
            raise oracle.OracleError(f"{name} is {t.dtype}, not BF16")
        if name.endswith("mlp.gate_proj.weight"):
            up = tensors[name.replace("gate_proj", "up_proj")]
            gate_up = torch.cat([t, up]).to(oracle.DEVICE)
            owners.append(gate_up)
            views += [gate_up[:INTER], gate_up[INTER:]]
            names += [name, name.replace("gate_proj", "up_proj")]
            continue
        d = t.to(oracle.DEVICE).contiguous()
        owners.append(d)
        views.append(d)
        names.append(name)
    return views, names, owners


def _order(steps: list[str]) -> torch.Tensor:
    """int32 [blocks, 4]: the position each step unmasked, -1 after the block's last step."""
    order = torch.full((len(steps), BLOCK), -1, dtype=torch.int32)
    for b, s in enumerate(steps):
        moved = s.split(".")
        if any(len(m) != 1 for m in moved):
            raise oracle.OracleError(f"block {b}: a step unmasked {s!r}, not one position per step")
        order[b, : len(moved)] = torch.tensor([int(m) for m in moved], dtype=torch.int32)
    return order


class SdarReplay(unittest.TestCase):
    """The recorded-step replay of condition 1 over the 64 S2 prompts."""

    def test_replay_reproduces_recorded_selection(self):
        doc = json.loads(RECORDING.read_text())
        mask_id = int(doc["meta"]["mask_id"])
        if doc["meta"]["schedule"]["top_k"] != 1 or doc["meta"]["schedule"]["block_length"] != BLOCK:
            raise oracle.OracleError(f"{RECORDING.name}: schedule {doc['meta']['schedule']} is not top_k 1, block 4")
        views, names, owners = load_weights()
        ptrs = torch.tensor([v.data_ptr() for v in views], dtype=torch.int64)
        name_bytes = torch.tensor(list(b"".join(n.encode() + b"\0" for n in names)), dtype=torch.uint8)
        dims = torch.tensor([[v.shape[0], v.shape[1] if v.dim() == 2 else 0] for v in views], dtype=torch.int64)

        steps = agree = agree_multi = multi = index_ok = index_ok_agree = 0
        untied = untied_ok = 0
        mismatched = []
        for row in doc["rows"]:
            seq = row["prompt_ids"] + row["gen_ids"]
            plen = len(row["prompt_ids"])
            order = _order(row["steps"])
            out = torch.full((len(row["steps"]), BLOCK, RECORD_INTS), -2, dtype=torch.int32)
            tokens = torch.tensor(seq, dtype=torch.int32)
            oracle.call("cere_ktest_sdar_replay", ptrs, name_bytes, dims, len(views), tokens, plen, len(seq), order,
                        out, mask_id)
            unequal = {(u[0], u[1]): u[2] for u in row["unequal"]}
            prefill = plen // BLOCK * BLOCK
            for b in range(len(row["steps"])):
                masked = [r for r in range(BLOCK) if prefill + b * BLOCK + r >= plen]
                for s in range(BLOCK):
                    p = int(order[b, s])
                    if p < 0:
                        break
                    conf = unequal.get((b, s))
                    want = {r: (round(1 / conf[r]) if conf is not None else 1) for r in masked}
                    rec = out[b, s].tolist()
                    got = {r: rec[2 + 2 * r] for r in masked}
                    steps += 1
                    multi += conf is not None
                    index_ok += rec[0] == p
                    if got == want:
                        agree += 1
                        agree_multi += conf is not None
                        index_ok_agree += rec[0] == p
                        if rec[0] != p and len(mismatched) < 5:
                            mismatched.append(f"{row['id']} block {b} step {s}: index {rec[0]}, recorded {p}")
                    if want[p] == 1:
                        untied += 1
                        untied_ok += rec[1 + 2 * p] == seq[prefill + b * BLOCK + p]
                    masked.remove(p)
        del owners
        oracle.report_row(ROW, "replay", phase="replay", prompts=len(doc["rows"]), steps=steps,
                          ties_agree=agree, recorded_multi_tie=multi, multi_tie_agree=agree_multi,
                          index_agree=index_ok, untied_transfers=untied, untied_token_agree=untied_ok)
        if agree == 0 or agree_multi == 0:
            self.fail(f"the recording cannot support the replay bar: {agree} steps with equal tie counts, "
                      f"{agree_multi} of them multi-tie")
        oracle.check_count(self, ROW, "replay index on tie-agreeing steps", index_ok_agree, expected=agree)
        self.assertEqual(mismatched, [])


if __name__ == "__main__":
    unittest.main()
