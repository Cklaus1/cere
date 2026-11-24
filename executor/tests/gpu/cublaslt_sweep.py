"""torch's cuBLASLt dispatch for y = F.linear(x, w), x BF16 [m, k], w BF16 [n, k], at every m in [m_lo, m_hi] on GPU0
(tasks/m22p-spec.md D7 lever 2: the record the pins of executor/cuda/src/kernels/k2_gemm.cpp come from).

    python3 cublaslt_sweep.py N K M_LO M_HI

Run by `native_dev.py cublaslt` in the bench image with CUBLASLT_LOG_MASK=2 and CUBLASLT_LOG_FILE=/dev/stderr: every
`cublasLtMatmul` torch launches is traced on stderr (algorithm id, tile, stages, split-K, reduction scheme, workspace
bytes), and native_dev parses that trace. Reduced-precision BF16 reduction is on, SGLang's (and torch's) default.
Inputs are seeded normals; nothing of a prompt is involved. Prints one line on stdout:

    CERE_CUBLASLT {"torch": ..., "cublaslt_library": <file name>, "cublaslt_version": <cublasLtGetVersion()>}

for the libcublasLt torch has loaded.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PREFIX = "CERE_CUBLASLT "


def loaded_cublaslt() -> str:
    """Path of the libcublasLt mapped into this process (the one torch's cuBLAS dispatches through)."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if "/libcublasLt.so" in path:
                return path
    raise SystemExit("cublaslt_sweep: libcublasLt is not loaded")


def main(argv: list[str]) -> int:
    n, k, m_lo, m_hi = (int(v) for v in argv)
    if not (n > 0 and k > 0 and 1 <= m_lo <= m_hi):
        raise SystemExit("cublaslt_sweep: N K M_LO M_HI with 1 <= M_LO <= M_HI")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((n, k), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    for m in range(m_lo, m_hi + 1):
        x = torch.randn((m, k), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        F.linear(x, w)
    torch.cuda.synchronize()
    path = loaded_cublaslt()
    lib = ctypes.CDLL(path)
    lib.cublasLtGetVersion.restype = ctypes.c_size_t
    print(PREFIX + json.dumps({"torch": torch.__version__, "cublaslt_library": Path(path).name,
                               "cublaslt_version": int(lib.cublasLtGetVersion())}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
