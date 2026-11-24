"""Generates the M2.13 C8 mRoPE oracle (tasks/m213-spec.md C8, D11, M213S-04) in the pinned image, CPU only, no network.

    docker run --rm --network none --user UID:GID -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= \
        --entrypoint python3 -v REPO:/src:ro -v MODEL:/model:ro -v REPO/tests/fixtures/native:/out IMAGE \
        /src/scripts/native/gen_mrope_cases.py --model /model --out /out

Writes `mrope_cases.json`:

    {"schema": "cere-mrope-cases-v1",
     "source": "sglang <version> QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets",
     "torch": <version>, "config_sha256": <sha256 of MODEL/config.json>, "model_type": "qwen4_exp",
     "spatial_merge_size": 2, "image_token_id", "vision_start_token_id", "vision_end_token_id",
     "cases": [{"name", "segments": [["text", n] | ["image", [t, h, w]]], "input_len": L,
                "images": [{"offset": <index of the first image pad>, "grid": [t, h, w], "tokens": n}],
                "path": "image_only" | "get_rope_index", "positions": [[L ints] x 3], "rope_delta": d}]}

A case is a token layout without text: `["text", n]` is n ordinary tokens, `["image", [t, h, w]]` is
`<|vision_start|>`, t*(h/2)*(w/2) `<|image_pad|>` tokens and `<|vision_end|>`, as the processor's image token regex
expands an image (`qwen_vl.py:349-354`). The ids are built from the model's `config.json` (no text is involved or
written); `offsets` come from SGLang's own `get_mm_items_offset` on those ids and each image becomes a real
`MultimodalDataItem` (modality IMAGE, `image_grid_thw` [1, 3]). Image cases record the image-only path the processor
takes (`_compute_image_only_mrope_positions_from_offsets`); the text-only case records `get_rope_index`, the path
SGLang takes when that one returns None. Every case is also run through `MRotaryEmbedding.get_rope_index` on the same
ids and grids, and both must agree exactly (else the run fails). Grids are ones the HF processor produces (factor 32,
>= 65,536 pixels, so >= 64 tokens; 1,024 tokens is the D4 per-image cap); the case list is fixed, so two runs write
the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "cere-mrope-cases-v1"
TEXT_ID = 1000  # an ordinary (non-special) token id; ids are never written

CASES: list[tuple[str, list[tuple[str, object]]]] = [
    ("text_only", [("text", 9)]),
    ("one_image_middle", [("text", 5), ("image", (1, 16, 16)), ("text", 4)]),
    ("image_first", [("image", (1, 8, 32)), ("text", 6)]),
    ("image_last", [("text", 3), ("image", (1, 32, 8))]),
    ("image_only", [("image", (1, 16, 16))]),
    ("wide_strip", [("text", 2), ("image", (1, 2, 128)), ("text", 2)]),
    ("tall_strip", [("text", 1), ("image", (1, 128, 2)), ("text", 3)]),
    ("max_tokens_image", [("text", 10), ("image", (1, 64, 64)), ("text", 5)]),
    ("non_square", [("text", 4), ("image", (1, 24, 40)), ("text", 7)]),
    ("two_adjacent", [("text", 2), ("image", (1, 16, 16)), ("image", (1, 40, 24)), ("text", 3)]),
    ("two_separated", [("text", 6), ("image", (1, 8, 32)), ("text", 11), ("image", (1, 32, 8)), ("text", 1)]),
    ("three_images", [("text", 4), ("image", (1, 4, 64)), ("text", 1), ("image", (1, 64, 4)), ("text", 2),
                      ("image", (1, 20, 20)), ("text", 5)]),
    ("four_images", [("image", (1, 16, 16)), ("image", (1, 8, 32)), ("text", 3), ("image", (1, 48, 16)),
                     ("text", 2), ("image", (1, 16, 16))]),
    ("four_images_last_adjacent", [("text", 7), ("image", (1, 16, 20)), ("text", 1), ("image", (1, 2, 128)),
                                   ("image", (1, 128, 2)), ("image", (1, 32, 32))]),
]


def build_layout(segments, image_token_id: int, start_id: int, end_id: int, merge: int):
    """Token ids and the image spans (offset of the first pad, grid, pad count) of a case layout."""
    ids: list[int] = []
    images = []
    for kind, value in segments:
        if kind == "text":
            ids.extend([TEXT_ID] * value)
        elif kind == "image":
            t, h, w = value
            if h % merge or w % merge:
                raise ValueError(f"grid {value} is not a multiple of the merge size {merge}")
            tokens = t * (h // merge) * (w // merge)
            ids.append(start_id)
            images.append({"offset": len(ids), "grid": [t, h, w], "tokens": tokens})
            ids.extend([image_token_id] * tokens)
            ids.append(end_id)
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
    return ids, images


def main() -> None:
    import sglang
    import torch
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_bytes = (args.model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    model_type = config["model_type"]
    merge = config["vision_config"]["spatial_merge_size"]
    image_id = config["image_token_id"]
    video_id = config["video_token_id"]
    start_id = config["vision_start_token_id"]
    end_id = config["vision_end_token_id"]

    # The processor's own method on a bare instance: it reads only `model_type` and `_spatial_merge_size`, which
    # `QwenVLImageProcessor.__init__` sets from the same config (qwen_vl.py:308, :343).
    processor = object.__new__(QwenVLImageProcessor)
    processor.model_type = model_type
    processor._spatial_merge_size = merge

    rows = []
    for name, segments in CASES:
        ids, images = build_layout(segments, image_id, start_id, end_id, merge)
        input_ids = torch.tensor(ids, dtype=torch.long)
        offsets = BaseMultimodalProcessor.get_mm_items_offset(input_ids, image_id)
        if len(offsets) != len(images):
            raise SystemExit(f"{name}: {len(offsets)} pad runs for {len(images)} images")
        items = []
        for (start, end), image in zip(offsets, images):
            if start != image["offset"] or end - start + 1 != image["tokens"]:
                raise SystemExit(f"{name}: offsets {(start, end)} != layout {image}")
            items.append(MultimodalDataItem(
                modality=Modality.IMAGE, offsets=[(start, end)],
                model_specific_data={"image_grid_thw": torch.tensor([image["grid"]], dtype=torch.long)}))

        grids = torch.tensor([image["grid"] for image in images], dtype=torch.long) if images else None
        reference = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=merge, image_token_id=image_id, video_token_id=video_id,
            vision_start_token_id=start_id, model_type=model_type, input_ids=input_ids.unsqueeze(0),
            image_grid_thw=grids)
        ref_positions = reference[0].squeeze(1)
        ref_delta = int(reference[1].reshape(-1)[0])

        result = processor._compute_image_only_mrope_positions_from_offsets(
            input_len=input_ids.numel(), mm_items=items, dtype=input_ids.dtype, device=input_ids.device)
        if result is None:
            if images:
                raise SystemExit(f"{name}: the image-only path returned None")
            positions, delta, path = ref_positions, ref_delta, "get_rope_index"
        else:
            positions, delta, path = result[0].squeeze(1), int(result[1].reshape(-1)[0]), "image_only"
            if positions.shape != ref_positions.shape or not torch.equal(positions, ref_positions) \
                    or delta != ref_delta:
                raise SystemExit(f"{name}: image-only positions differ from get_rope_index")
        if tuple(positions.shape) != (3, len(ids)):
            raise SystemExit(f"{name}: positions shape {tuple(positions.shape)}")
        rows.append({
            "name": name,
            "segments": [[kind, list(value) if kind == "image" else value] for kind, value in segments],
            "input_len": len(ids),
            "images": images,
            "path": path,
            "positions": positions.tolist(),
            "rope_delta": delta,
        })

    header = {
        "schema": SCHEMA,
        "source": f"sglang {sglang.__version__} "
                  "QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets",
        "torch": torch.__version__,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_type": model_type,
        "spatial_merge_size": merge,
        "image_token_id": image_id,
        "vision_start_token_id": start_id,
        "vision_end_token_id": end_id,
    }
    # One case per line keeps the [3][L] position rows compact.
    head = json.dumps(header, indent=1)[:-2]
    body = ",\n".join("  " + json.dumps(row, separators=(",", ":")) for row in rows)
    path = args.out / "mrope_cases.json"
    path.write_text(f'{head},\n "cases": [\n{body}\n ]\n}}\n')
    json.loads(path.read_text())
    print(f"wrote {path}: {len(rows)} cases, {sum(r['input_len'] for r in rows)} positions")


if __name__ == "__main__":
    main()
