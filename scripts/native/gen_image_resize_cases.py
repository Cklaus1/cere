"""Generates the M2.13 C2 resize oracle (tasks/m213-spec.md C2, D4, D18) in the pinned image, CPU only, no network.

    docker run --rm --network none --user UID:GID -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= \
        --entrypoint python3 -v REPO:/src:ro -v MODEL:/model:ro -v REPO/tests/fixtures/native:/out IMAGE \
        /src/scripts/native/gen_image_resize_cases.py --model /model --out /out

Writes `image_resize_cases.json`:

    {"schema": "cere-image-resize-cases-v1", "source": "transformers <version> smart_resize",
     "preprocessor_sha256": <sha256 of MODEL/preprocessor_config.json>,
     "factor": 32, "min_pixels": 65536, "max_pixels": 16777216, "max_ratio": 200,
     "processor_checked": <number of cases also run through AutoImageProcessor>,
     "cases": [{"height", "width", "resized": [h, w] | null, "error": null | "aspect"}]}

`factor` = patch_size x merge_size and the pixel bounds are read from the model's `preprocessor_config.json`
(`size.shortest_edge` / `size.longest_edge`, as the HF processor reads them). `resized` is HF's `smart_resize`
(`image_processing_qwen2_vl.py`) for that factor and those bounds; `error` is `"aspect"` where it raises its aspect
ratio error (and then `resized` is null). The image tokens of a case are (h / factor) x (w / factor). Every case with
at most CHECK_PIXELS input pixels is also run through `AutoImageProcessor.from_pretrained(MODEL)` on a synthetic
zero image of that size, and its `image_grid_thw` must equal [1, h / patch, w / patch] (else the run fails), so the
fixture is the processor's own grid, not only the helper's. The sizes are a fixed list (ties, minimum and maximum
pixel edges, aspect 200/201) plus seeded random sizes; no image content is involved or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import transformers
from transformers import AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

SCHEMA = "cere-image-resize-cases-v1"
SEED = 20261010
TOTAL_CASES = 300
MAX_SIDE = 8192  # D4: header sides above this are refused before any resize
MAX_RATIO = 200
CHECK_PIXELS = 1 << 20

FIXED = [
    # Python `round` ties to even: 48/32 = 1.5 -> 2, 80/32 = 2.5 -> 2, 112/32 = 3.5 -> 4, 144/32 = 4.5 -> 4.
    (48, 2000), (80, 1000), (112, 800), (144, 600), (2000, 48), (1000, 80), (800, 112), (600, 144),
    (48, 4400), (80, 4400), (112, 4400), (144, 4400), (1040, 1040), (1072, 1072), (1104, 1008), (1136, 976),
    # aspect ratio exactly 200 (accepted) and above 200 (refused)
    (32, 6400), (6400, 32), (40, 8000), (8000, 40), (1, 200), (200, 1),
    (32, 6401), (6401, 32), (32, 6432), (6432, 32), (40, 8041), (1, 201), (201, 1), (16, 8192),
    # minimum pixel edge (65,536 = 256 x 256)
    (256, 256), (255, 256), (256, 255), (224, 288), (288, 224), (240, 272), (200, 327), (128, 512),
    (64, 1024), (1, 1), (2, 3), (17, 17), (32, 32), (31, 33), (100, 100), (160, 410), (192, 341), (257, 255),
    # maximum pixel edge (16,777,216 = 4,096 x 4,096)
    (4096, 4096), (4097, 4096), (4096, 4097), (4095, 4097), (4112, 4080), (5000, 4000), (8192, 8192),
    (8192, 2048), (2048, 8192), (8192, 4096), (6000, 3000), (8000, 2100), (4128, 4064), (8191, 8191),
    (8192, 1), (3000, 5593), (4160, 4033), (1024, 1024), (1023, 1025), (2048, 2048), (512, 768),
]


def cases() -> list[tuple[int, int]]:
    rng = random.Random(SEED)
    out = list(dict.fromkeys(FIXED))
    seen = set(out)
    while len(out) < TOTAL_CASES:
        # half log-uniform over the full side range, half within 32..2,048 where most requests sit
        if rng.random() < 0.5:
            h, w = (int(round(2 ** rng.uniform(0, 13))) for _ in range(2))
        else:
            h, w = rng.randint(32, 2048), rng.randint(32, 2048)
        h, w = max(1, min(h, MAX_SIDE)), max(1, min(w, MAX_SIDE))
        if (h, w) not in seen:
            seen.add((h, w))
            out.append((h, w))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_bytes = (args.model / "preprocessor_config.json").read_bytes()
    config = json.loads(config_bytes)
    patch = config["patch_size"]
    factor = patch * config["merge_size"]
    min_pixels = config["size"]["shortest_edge"]
    max_pixels = config["size"]["longest_edge"]
    processor = AutoImageProcessor.from_pretrained(str(args.model), local_files_only=True)

    rows = []
    checked = 0
    for h, w in cases():
        try:
            resized = list(smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels))
            error = None
        except ValueError as e:
            if "aspect ratio" not in str(e):
                raise
            resized, error = None, "aspect"
        if error is None and h * w <= CHECK_PIXELS:
            image = np.zeros((h, w, 3), dtype=np.uint8)
            grid = processor(images=[image], input_data_format="channels_last", return_tensors="pt")
            grid = grid["image_grid_thw"].tolist()
            want = [[1, resized[0] // patch, resized[1] // patch]]
            if grid != want:
                raise SystemExit(f"processor grid {grid} != smart_resize {want} for {h}x{w}")
            checked += 1
        rows.append({"height": h, "width": w, "resized": resized, "error": error})

    document = {
        "schema": SCHEMA,
        "source": f"transformers {transformers.__version__} smart_resize",
        "preprocessor_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "factor": factor,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_ratio": MAX_RATIO,
        "processor_checked": checked,
        "cases": rows,
    }
    path = args.out / "image_resize_cases.json"
    path.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {path}: {len(rows)} cases, {checked} processor-checked")


if __name__ == "__main__":
    main()
