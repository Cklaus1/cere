#!/usr/bin/env python3
"""Inkling S11: fail the build when a ki_*.spv has a floating-point arithmetic instruction without NoContraction.

    python3 executor/vulkan/tools/check_nocontract.py <file.spv>...

The Inkling shaders are bit-equal to executor/ref/inkling by construction (every float op is one IEEE rounding, fma only
where written), which holds only if no driver may fuse a mul into an add. glslang marks the ops of `precise` values
NoContraction; spirv-opt -O drops the decoration from code it copies (seen: the unrolled fp64 TwoSum of ki_fma, fused by
lavapipe's LLVM into a wrong-by-one-ulp chain), so the ki shaders are compiled -O0 and this check keeps it that way.
Exit 0 when every OpFAdd / OpFSub / OpFMul / OpFDiv / OpFRem / OpFMod / OpVectorTimesScalar / OpDot result carries
NoContraction; 1 and a list otherwise.
"""
import struct
import sys

OPS = {129: "OpFAdd", 131: "OpFSub", 133: "OpFMul", 136: "OpFDiv", 138: "OpFRem", 141: "OpFMod",
       142: "OpVectorTimesScalar", 148: "OpDot"}
OP_DECORATE, NO_CONTRACTION = 71, 42


def unmarked(path: str) -> list:
    data = open(path, "rb").read()
    if len(data) < 20 or len(data) % 4:
        raise SystemExit(f"check_nocontract: {path}: not a SPIR-V module")
    end = "<" if struct.unpack_from("<I", data)[0] == 0x07230203 else ">"
    words = struct.unpack(f"{end}{len(data) // 4}I", data)
    nc, arith = set(), []
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xFFFF
        if n == 0:
            raise SystemExit(f"check_nocontract: {path}: zero-length instruction at word {i}")
        if op == OP_DECORATE and n >= 3 and words[i + 2] == NO_CONTRACTION:
            nc.add(words[i + 1])
        elif op in OPS:
            arith.append((words[i + 2], OPS[op]))
        i += n
    return [(rid, name) for rid, name in arith if rid not in nc]


def main() -> int:
    bad = 0
    for path in sys.argv[1:]:
        for rid, name in unmarked(path):
            print(f"check_nocontract: {path}: %{rid} {name} without NoContraction")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
