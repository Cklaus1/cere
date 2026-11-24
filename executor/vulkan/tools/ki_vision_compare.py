#!/usr/bin/env python3
"""Inkling S12: the vision tower on Vulkan (ktest/ki_vision.cpp) against the recorded reference tower
(tests/fixtures/inkling2_reference/vision/, S7's gate, executor/tests/gpu/test_inkling_kernels_vision.py).

  ki_vision_compare.py prep PROCESSOR_CONFIG OUT_DIR   (torch + PIL: the reference image)
      the tokenizer sidecar's InklingProcessor on each recorded PNG; the pixel values must equal the reference
      processor's bit for bit (sha256 of the FP32 [N, 40, 40, 3] bytes) and grid. Writes OUT_DIR/NAME.pix (BF16 patch
      rows) and NAME.grid (int32 h, w in 20-px patches) for ki_vision.
  ki_vision_compare.py compare DIR [--json OUT]   (numpy)
      per image and case (single, batched): cosine >= 0.999 and rel-L2 <= 2 x the reference's own BF16-vs-FP32 rel-L2
      on that image, vs the reference tower in FP32 (stored FP16); each mutant (no_class, swap_rope) must fail that bar
      on at least one image. Exit 0 when every check holds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
FIXTURE = SRC / "tests/fixtures/inkling2_reference/vision"
OUT_HIDDEN = 2304
MIN_COSINE = 0.999
BAR_SPREAD = 2.0
CASES = ("single", "batched")
MUTANTS = ("no_class", "swap_rope")


def record() -> dict:
    return json.loads((FIXTURE / "vision.json").read_text())


def prep(a) -> int:
    import torch
    from PIL import Image

    spec = importlib.util.spec_from_file_location("sidecar", SRC / "scripts/native/tokenizer_sidecar.py")
    sidecar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sidecar)
    config = json.loads(Path(a.processor_config).read_text())["image_processor"]
    proc = sidecar.InklingProcessor(torch, config)
    rec = record()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for n in rec["order"]:
        r = rec["images"][n]
        cells, rows, cols = proc.cells(Image.open(FIXTURE / f"{n}.png"))
        same = (hashlib.sha256(cells.contiguous().numpy().tobytes()).hexdigest() == r["pixel_values_sha256"]
                and [1, rows, cols] == r["grid_thw"])
        print(f"processor {n}: grid {rows}x{cols} {'same' if same else 'DIFFERS'}")
        ok &= same
        pix = proc.patch_rows(cells).to(torch.bfloat16).contiguous()
        (out / f"{n}.pix").write_bytes(pix.view(torch.int16).numpy().tobytes())
        (out / f"{n}.grid").write_bytes(struct.pack("<ii", 2 * rows, 2 * cols))
    (out / "order").write_text(" ".join(rec["order"]) + "\n")
    return 0 if ok else 1


def load_reference() -> dict:
    import numpy as np

    raw = (FIXTURE / "vision.safetensors").read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    out = {}
    for k, e in hdr.items():
        if k == "__metadata__" or not k.startswith("f32."):
            continue
        assert e["dtype"] == "F16", (k, e["dtype"])
        b, c = e["data_offsets"]
        out[k[4:]] = np.frombuffer(raw[8 + n + b:8 + n + c], dtype=np.float16).reshape(e["shape"]).astype(np.float64)
    return out


def load_bf16(path: Path):
    import numpy as np

    u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(-1, OUT_HIDDEN).astype(np.float64)


def compare(a) -> int:
    import numpy as np

    rec, ref = record(), load_reference()
    d = Path(a.dir)
    result = {"cases": {}, "mutants": {}}
    ok = True
    for case in CASES + MUTANTS:
        failed = 0
        for n in rec["order"]:
            ours, r = load_bf16(d / f"{n}.{case}.bf16"), ref[n]
            if ours.shape != r.shape:
                print(f"{case} {n}: shape {ours.shape} vs {r.shape}")
                ok, failed = False, failed + 1
                continue
            cosine = float(np.dot(ours.ravel(), r.ravel()) / (np.linalg.norm(ours) * np.linalg.norm(r)))
            rel = float(np.linalg.norm(ours - r) / np.linalg.norm(r))
            limit = BAR_SPREAD * rec["images"][n]["bf16_vs_f32_cos_rel"][1]
            passed = bool(np.isfinite(ours).all()) and cosine >= MIN_COSINE and rel <= limit
            failed += not passed
            result["cases"][f"{case} {n}"] = {"cosine": cosine, "rel_l2": rel, "limit": limit, "passed": passed}
            print(f"{case:9s} {n:9s} cosine {cosine:.6f} rel-L2 {rel:.5f} (limit {limit:.5f}) "
                  f"{'PASS' if passed else 'FAIL'}")
            if case in CASES:
                ok &= passed
        if case in MUTANTS:
            refused = failed > 0
            result["mutants"][case] = {"failed_images": failed, "refused": refused}
            print(f"mutant {case}: fails {failed} / {len(rec['order'])} images -> {'REFUSED' if refused else 'NOT REFUSED'}")
            ok &= refused
    result["pass"] = ok
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1) + "\n")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("processor_config")
    p.add_argument("out_dir")
    c = sp.add_parser("compare")
    c.add_argument("dir")
    c.add_argument("--json")
    a = ap.parse_args()
    return prep(a) if a.cmd == "prep" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
