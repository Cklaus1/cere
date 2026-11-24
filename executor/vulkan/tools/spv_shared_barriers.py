#!/usr/bin/env python3
"""D-I-EPB (a1): move workgroup-memory availability / visibility from every shared access to the barriers.

    python3 executor/vulkan/tools/spv_shared_barriers.py <file.spv>...   (rewrites each file in place)

A shader that uses cooperative matrices is compiled under the Vulkan memory model (glslang turns it on), and glslang
then marks every load / store of a `shared` variable MakePointerVisible / MakePointerAvailable at workgroup scope
(GLSL's shared is coherent). Mesa's SPIR-V front end turns each mark into an acquire / release memory barrier, so on
RADV every LDS access waits for all outstanding ones (`s_waitcnt lgkmcnt(0)` after each ds_load / ds_store), which
serializes the int8 cooperative-matrix MoE tile's loads. The pass drops those marks (keeping NonPrivatePointer, so
the accesses stay non-private) and adds MakeAvailable | MakeVisible to the semantics of every barrier that already
orders workgroup memory with acquire-release: the same guarantees the GLSL450 model gives the vector shaders (writes
reach other invocations through the barriers, an invocation's own accesses stay in program order). No arithmetic
instruction changes, so the shader's bits do not depend on the pass.

Refuses (exit 1, file untouched) a module with a workgroup access the pass does not know (copies, atomics,
cooperative-matrix stores) or a barrier on workgroup memory without acquire-release.
"""
import struct
import sys

OP_VARIABLE, OP_CONSTANT, OP_LOAD, OP_STORE = 59, 43, 61, 62
OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_PTR_ACCESS_CHAIN = 65, 66, 67
OP_COPY_MEMORY, OP_COPY_MEMORY_SIZED = 63, 64
OP_CONTROL_BARRIER, OP_MEMORY_BARRIER = 224, 225
OP_CM_LOAD, OP_CM_STORE = 4457, 4458
ATOMICS = set(range(227, 243)) | {5000, 6035}
WORKGROUP = 4
ALIGNED, AVAILABLE, VISIBLE = 0x2, 0x8, 0x10
ACQ_REL, WG_MEMORY, MAKE_AVAILABLE, MAKE_VISIBLE = 0x8, 0x100, 0x2000, 0x4000


def die(path, msg):
    raise SystemExit(f"spv_shared_barriers: {path}: {msg}")


def strip(path, ops, at):
    """ops[at] is a memory-access mask; return the operand words with the Available / Visible marks removed."""
    mask, rest = ops[at], list(ops[at + 1:])
    out = [mask & ~(AVAILABLE | VISIBLE)]
    if mask & ALIGNED:
        out.append(rest.pop(0))
    if mask & AVAILABLE:
        rest.pop(0)
    if mask & VISIBLE:
        rest.pop(0)
    if rest or mask & ~(ALIGNED | AVAILABLE | VISIBLE | 0x20 | 0x1 | 0x4):
        die(path, f"memory operands {ops[at:]} not understood")
    return list(ops[:at]) + out


def rewrite(path):
    data = open(path, "rb").read()
    if len(data) < 20 or len(data) % 4 or struct.unpack_from("<I", data)[0] != 0x07230203:
        die(path, "not a little-endian SPIR-V module")
    w = list(struct.unpack(f"<{len(data) // 4}I", data))
    insts, i = [], 5
    while i < len(w):
        n = w[i] >> 16
        if n == 0:
            die(path, f"zero-length instruction at word {i}")
        insts.append([w[i] & 0xFFFF] + w[i + 1:i + n])
        i += n
    wg, root, const, const_at = set(), {}, {}, {}
    for k, ins in enumerate(insts):
        op = ins[0]
        if op == OP_VARIABLE and ins[3] == WORKGROUP:
            wg.add(ins[2])
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_PTR_ACCESS_CHAIN):
            root[ins[2]] = ins[3]
        elif op == OP_CONSTANT and len(ins) == 4:
            const[ins[2]], const_at[ins[2]] = ins[3], k

    def shared(p):
        while p in root:
            p = root[p]
        return p in wg

    bound, sem_new, adds, changed = w[3], {}, [], 0
    for ins in insts:
        op = ins[0]
        if op == OP_LOAD and len(ins) > 4 and shared(ins[3]):
            ins[1:] = strip(path, ins[1:], 3)
            changed += 1
        elif op == OP_STORE and len(ins) > 3 and shared(ins[1]):
            ins[1:] = strip(path, ins[1:], 2)
            changed += 1
        elif op == OP_CM_LOAD and shared(ins[3]):
            if len(ins) > 6:  # result type, result, pointer, layout, stride, mask...
                ins[1:] = strip(path, ins[1:], 5)
                changed += 1
        elif op == OP_CM_STORE and shared(ins[1]):
            die(path, "cooperative-matrix store to workgroup memory")
        elif op in (OP_COPY_MEMORY, OP_COPY_MEMORY_SIZED) and (shared(ins[1]) or shared(ins[2])):
            die(path, "memory copy on workgroup memory")
        elif op in ATOMICS and len(ins) > 3 and shared(ins[3]):
            die(path, "atomic on workgroup memory")
        elif op in (OP_CONTROL_BARRIER, OP_MEMORY_BARRIER):
            s = len(ins) - 1  # the semantics id is the last operand of both
            if ins[s] not in const:
                die(path, f"barrier semantics %{ins[s]} is not a scalar constant")
            v = const[ins[s]]
            if v & WG_MEMORY:
                if not v & ACQ_REL:
                    die(path, f"workgroup barrier semantics {v:#x} without AcquireRelease")
                if ins[s] not in sem_new:
                    sem_new[ins[s]] = bound
                    adds.append((const_at[ins[s]], bound, v | MAKE_AVAILABLE | MAKE_VISIBLE))
                    bound += 1
                ins[s] = sem_new[ins[s]]
    if changed == 0:
        return 0
    for at, nid, v in sorted(adds, reverse=True):  # new constants right after the ones they extend (same type)
        old = insts[at]
        insts.insert(at + 1, [OP_CONSTANT, old[1], nid, v])
    w[3] = bound
    out = w[:5]
    for ins in insts:
        out.append((len(ins) << 16) | ins[0])
        out.extend(ins[1:])
    open(path, "wb").write(struct.pack(f"<{len(out)}I", *out))
    return changed


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for p in sys.argv[1:]:
        print(f"spv_shared_barriers: {p}: {rewrite(p)} shared accesses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
