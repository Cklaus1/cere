"""M2.2p X4: FlashInfer's MoE tuning bucket and the reader of the bench's logged tactics (tasks/m22p-spec.md D6, Q3).

One implementation for the writer (`scripts/live/native_dev.py tactics`, which loads this file by path) and the GPU
readers (`test_numerics_prefill.py`, `test_prefill_isolation_x.py`, which import it from this directory). Python stdlib
only: no torch, no CUDA. The tactics file (`prefill_isolation_tactics.json`) is
`{"buckets": {"<bucket>": {"tactics": [<one per launch>], "sources": [...]}}, "source": ..., "launches": N}`, where a
launch's tactic is `[gemm1, gemm2]`, `-1` (both ops fell back) or `"unavailable"`.
"""
from __future__ import annotations

import json
from pathlib import Path


def next_power_of_2(n: int) -> int:
    """SGLang's `next_power_of_2` ($S/utils/common.py:3337)."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


def moe_bucket(m: int, max_tokens: int | None = None) -> int:
    """FlashInfer's MoERunner tuning bucket of M token rows, `map_to_hybrid_bucket(M, max_tokens)` (flashinfer 0.6.17
    `fused_moe/utils.py:184-188, 245-261`): next power of 2 up to 256, ceil to 256 up to 2048, ceil to 512 up to 4096,
    then the next power of 2, clamped to [1, max_tokens]. Dimension 0 of input 0 (the token rows) is tuned. The
    default cap is the runtime one: SGLang passes `tune_max_num_tokens=next_power_of_2(M)`
    (`$S/layers/moe/moe_runner/flashinfer_cutlass.py:236, 390`) and `refine_tuning_config` installs that mapper
    (`fused_moe/core.py:498-507, 552`), not MoERunner's class default 8192; the two agree for M <= 8192."""
    if max_tokens is None:
        max_tokens = next_power_of_2(m)
    if m <= 0:
        return 1
    if m >= max_tokens:
        return max_tokens
    if m <= 256:
        return 1 << (m - 1).bit_length()
    if m <= 2048:
        return -(-m // 256) * 256
    if m <= 4096:
        return -(-m // 512) * 512
    return min(1 << (m - 1).bit_length(), max_tokens)


def _pair(t) -> bool:
    return isinstance(t, tuple) and len(t) == 2 and all(type(v) is int for v in t)


def tactic_status(doc: dict, prompt_tokens: int) -> tuple[object, str]:
    """(tactic, "logged") for bucket moe_bucket(prompt_tokens) when >= 2 logged launches agree on -1 or a
    (gemm1, gemm2) pair, else (None, "unavailable: <why>")."""
    bucket = str(moe_bucket(prompt_tokens))
    entry = (doc.get("buckets") or {}).get(bucket) if isinstance(doc, dict) else None
    launches = (entry.get("tactics") or []) if isinstance(entry, dict) else []
    canon = [tuple(t) if isinstance(t, list) else t for t in launches]
    if len(canon) < 2:
        return None, f"unavailable: {len(canon)} logged launch(es) for bucket {bucket}"
    if any(t != canon[0] for t in canon):
        return None, f"unavailable: the {len(canon)} logged launches of bucket {bucket} disagree"
    tactic = canon[0]
    if not ((type(tactic) is int and tactic == -1) or _pair(tactic)):
        return None, f"unavailable: bucket {bucket} logs {tactic!r}, not -1 or a (gemm1, gemm2) pair"
    return tactic, "logged"


def logged_tactic(doc: dict, prompt_tokens: int):
    """The agreed tactic of `tactic_status`, or None."""
    return tactic_status(doc, prompt_tokens)[0]


def read_tactic(path: Path, prompt_tokens: int) -> tuple[object, str]:
    """`tactic_status` of the tactics file at `path`; an absent file is (None, "unavailable")."""
    path = Path(path)
    if not path.is_file():
        return None, "unavailable"
    return tactic_status(json.loads(path.read_text()), prompt_tokens)
