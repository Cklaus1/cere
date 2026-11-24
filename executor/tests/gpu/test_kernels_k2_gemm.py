"""K2 (tasks/m22-spec.md): the cuBLASLt BF16 GEMM wrapper and the last-row-only lm_head path on the real dense weights.

Every case reports row "K2". Reference op: `torch.matmul(x, w.T)` in BF16, i.e. what SGLang runs for unquantized
linears (`UnquantizedLinearMethod.apply` -> `F.linear`, `sglang/srt/layers/quantization/unquant.py:391`) and for the
lm_head (`torch.matmul(hidden_states.to(w.dtype), w.T)`, `sglang/srt/layers/logits_processor.py:741-743`). Tolerance
rule: `c` = rel-L2(reference op vs an FP32 GEMM of the same BF16 values, TF32 off), reported before the native call;
limit = max(4e-3, 2c); measured = rel-L2(native vs reference op). Determinism: 10 runs of the same native call are
bit-identical (the pinned atomics-free algorithm, spec kernel table "Dense GEMM ... lm_head").

Weights (checkpoint layout `[out_features, in_features]`, BF16 in the NVFP4 checkpoint): attention q/k/v/o of layer 3,
GDN in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj of layer 0, shared expert gate/up/down and shared_expert_gate
of layer 0, and lm_head (248,320 rows); M in {1, 4, 7, 16, 2,048}; activations are seeded BF16 normals. The executor
plan runs cuBLASLt for every M under the default k2_gemm lever (slot impl CERE_LEVER_K2_CUBLASLT); the lever's
CERE_LEVER_K2_SMALL_M runs M <= 16 on the P11 small-M kernel (tasks/m23-spec.md P11, `k2_small_m.cu`).

The last-row-only lm_head path: SGLang prunes an extend batch to its last position before the lm_head
(`logits_processor.py:464-465`), so its logits are `torch.matmul(h[-1:], w.T)`; the native path must (a) be within the
limit of that op, (b) equal the native GEMM of the last row alone (`cere_ktest_gemm_bf16` with M = 1) bit for bit, so
prefill and decode produce the same logits for the same hidden row, and (c) repeat bit-identically.

P11 small-M rows (tasks/m23-spec.md P11, D9 "n <= 16 rule"), on every dense decode shape (attention qkv packed and o,
GDN in_proj_qkv/z/b/a and out_proj, router and hc down/up with F32 output, shared expert gate/up/down, lm_head),
with the kernel switched on per plan (the k2_gemm lever's small_m path): the oracle (`p11_<shape>_M<m>_oracle`, m in {1, 4, 7, 16},
against torch.matmul under the rule above), batch invariance (`p11_<shape>_batch_invariant`: for every m in 2..16, each row of the m-row call is bit-equal to the
m = 1 call on that row alone, count of such m = 15), 10 repeats of the M = 16 call bit-identical, prefill unchanged
(`p11_<shape>_M<m>_prefill_unchanged`, m in {17, 2,048}: the plan's call is bit-equal to the same plan with the small-M
kernel off, i.e. cuBLASLt), and fusion equality (`p11_<group>_M<m>_fused`: `kernels::gemm_segments` over the GDN
in-projections and over the shared gate/up equals each weight's own small-M call bit for bit). Layout (P11R-12,
P11R-13): `p11_layout_fills_gpu` counts the dense decode calls whose launch runs one CTA per 16-slice group (a
cluster of slices / 16 CTAs past 16 slices) and gives every SM a CTA (grid >= min(SMs, 8-column tiles x groups)),
`p11_<shape>_M<m>_layout_invariant` (m in {1, 16}) checks each shape's own call against the same weight launched
beside a 16 x SMs-column neighbour (16-column tiles) bit for bit. Report-only bench rows
`p11_bench_<shape>_M<m>` give the µs per call and effective TB/s of the small-M kernel and of cuBLASLt (CUDA events
around 50 calls of one plan after 5 warm-ups, rotating over copies of the weight totalling >= 384 MiB, 3x L2, so each
call streams its weight from DRAM as a decode step does).

Lever 2 pin rows (tasks/m22p-spec.md D7 lever 2, M22PS-R3): for both ends m_lo and min(m_hi, 4,096) of every pin range
of the shared gate/up shape (`cere_ktest_gemm_pin_ranges`), the native gate and up GEMMs of layer 0 are bit-identical
to the columns of torch's one `F.linear(x, [gate; up])` (reduced-precision BF16 reduction on) and repeat 10 times
bit-identically; likewise for every pin range of the attn_qkv shape, the native GEMM over layer 3's packed q|k|v weight
is bit-identical to torch's `F.linear(x, w_qkv)`. Each shape's pins are refused (CERE_ERR_UNSUPPORTED) under any
cuBLASLt version but the recorded one (`cere_ktest_gemm_pins_init` with the version shifted by -1 and +1; 0 is
accepted).

Bucket re-pin rows (tasks/r3-prefill-spec.md, R3 PrefillCompute; k2_gemm.h kFirstBucketRepin): every re-pinned plan
shape (REPIN_SHAPES) has pin ranges, and at both ends of each range and at every registered prefill chunk m inside it
(REPIN_CELL_M) the executor's pinned plan (`cere_ktest_gemm_path`, cuBLASLt) equals the same shape's plan without
pins (`cere_ktest_gemm_unpinned`: the bucket algorithm main runs) bit for bit, on full-mantissa weights and
wide-exponent activations (so equal bits are evidence of the same summation order), and repeats 10 times
bit-identically. Row `repin_<shape>_<m_lo>_<m_hi>_M<m>`.

D6 isolation rows (tasks/m22d-spec.md D6/C7; phase `isolate`, report-only: they assert nothing beyond running), one
decode token (M = 1) on every layer of the block's type, real weights, a seeded BF16 input fed to both sides; the
control is the GEMM call SGLang dispatches for the op: `isolate_L<layer>_qkv_proj_M1` (`F.linear` over the
`QKVParallelLinear`-stacked q|k|v weight; native: one GEMM over the same packed weight, as `forward.cpp` runs it) and
`isolate_L<layer>_o_proj_M1` on the 12 full-attention layers; `isolate_L<layer>_router_M1` (`mlp.gate`),
`isolate_L<layer>_shared_gate_up_M1` (SGLang's one GEMM over [gate; up] vs the native gate and up GEMMs) and
`isolate_L<layer>_shared_down_M1` (on the control's `silu_and_mul` output) on all 48 layers. Each row reports `c` =
rel-L2(control, FP32 GEMM), `limit` = max(FLOOR_OP, 2c), `measured` = rel-L2(native, control), `within`, and the
differing BF16 elements.

Native entry points (oracle.py calling convention: device pointers `void *`, integers `int64_t`, the stream last;
0 = success). All tensors are contiguous and row-major.

    int cere_ktest_gemm_bf16(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                             cudaStream_t stream);
        x bf16 [m, k]; w bf16 [n, k] (checkpoint layout); y bf16 [m, n] = x * w^T with FP32 accumulation, through the
        cuBLASLt algorithm pinned at INSTANTIATE (no split-K atomics; identical bits on every call). The shared
        gate/up shape (n 640, k 2,560) at M >= 17 runs the executor's pins (k2_gemm.h gemm_pins_for: torch's split-K
        configuration with a separate reduce kernel, tasks/m22p-spec.md lever 2).

    int cere_ktest_gemm_pin_ranges(int64_t n, int64_t k, int64_t out_f32, void *ranges, int64_t cap, void *count,
                                   cudaStream_t stream);
        ranges: host int64 [cap, 2] receiving (m_lo, m_hi) of every pin range of gemm_pins_for(n, k, out) (F32 when
        out_f32 is 1, else BF16); count: host int64 [1] receiving the number of ranges. Nonzero status when the
        shape has no pins or cap is short.

    int cere_ktest_gemm_unpinned(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t out_f32,
                                 void *y, cudaStream_t stream);
        y [m, n] (BF16, or F32 when out_f32 is 1) = x [m, k] * w^T through a plan of max_m = m without pins: the
        bucket algorithm GemmPlan::init picks for m (the reference of the bucket re-pins).

    int cere_ktest_gemm_pins_init(int64_t n, int64_t k, int64_t max_m, int64_t version_offset, cudaStream_t stream);
        The status of GemmPlan::init(n, k, max_m, BF16) with gemm_pins_for's pins whose recorded cuBLASLt version
        is shifted by version_offset (0: the executor's pins), given their workspace; no GEMM runs.

    int cere_ktest_lm_head_last(const void *hidden, const void *w, int64_t m, int64_t vocab, int64_t k, void *logits,
                                cudaStream_t stream);
        hidden bf16 [m, k]; w bf16 [vocab, k] (lm_head.weight); logits bf16 [vocab] = hidden[m - 1] * w^T. Only the
        last row is read and computed.

    int cere_ktest_gemm_rows(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t rows,
                             int64_t out_f32, int64_t fused_allowed, void *y, void *per_row, cudaStream_t stream);
        M2.6 K1 (tasks/m26-spec.md D6, I6): y [m, n] (BF16, or F32 when out_f32 is 1) = x [m, k] * w^T through
        GemmPlan::init_rows(instantiated, rows, fused_allowed) at m <= rows (rows 0: the instantiated plan itself at
        max_m = m); per_row: host int64 [1] = per_row(m) of the plan that ran. fused_allowed bit m - 1 clear forces
        the per-row fallback at m (all bits set: the executor's rows plan).

    int cere_ktest_hc_mix_rows(..., as cere_ktest_hc_mix ..., int64_t rows, int64_t fused_allowed,
                               cudaStream_t stream);
        cere_ktest_hc_mix through HcPlans::init_rows(rows, fused_allowed) at m <= rows.

    int cere_ktest_gemm_path(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t out_f32,
                             int64_t small_m, int64_t iters, int64_t copies, void *y, void *us, cudaStream_t stream);
        P11: y [m, n] (BF16, or F32 when out_f32 is 1) = x [m, k] * w^T through a plan of max_m = m with the small-M
        kernel on (small_m 1, the executor's path for m <= 16) or off (0: cuBLASLt, the reference path). iters >= 1
        calls of the same plan, call i on copy i % copies of w ([copies, n, k] back to back); with iters > 5, us
        (host float64 [1]) receives the mean µs per call of the last iters - 5 calls (CUDA events), else it is
        untouched.

    int cere_ktest_gemm_segments(const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs,
                                 const int64_t *ns, const int64_t *y_ptrs, int64_t out_f32, cudaStream_t stream);
        P11 fusion: kernels::gemm_segments over `count` <= 4 weights sharing x [m, k]; w_ptrs / y_ptrs host int64
        device addresses, ns host int64 output widths.

    int cere_ktest_k2s_geometry(int64_t k, int64_t count, const int64_t *ns, void *out, cudaStream_t stream);
        P11R-12: the small-M launch k2s::gemm makes for widths ns (host int64 [count]) at k on the current device;
        out host int64 [9]: grid, block and cluster dims (host arithmetic).

    int cere_ktest_gemm_m1_algo(int64_t n, int64_t k, int64_t out_f32, void *out, cudaStream_t stream);
        P11R-21: out host int64 [5]: the loaded libcublasLt's version, the executor plan's m = 1 algorithm id and
        customOption for (n, k, out), and k2v::order_of's lanes and chunk for them (0 0: no identified order).

    int cere_ktest_k2v_order(int64_t cublaslt_version, int64_t algo_id, int64_t custom_option, int64_t k, void *out,
                             cudaStream_t stream);
        P11R-21: k2v::order_of for an explicit libcublasLt version; out host int64 [2] = lanes, chunk.

    int cere_ktest_k2v_gemv(const void *x, int64_t m, int64_t k, const void *w, int64_t n, int64_t out_f32,
                            int64_t lanes, int64_t chunk, void *y, cudaStream_t stream);
        P11R-21: one gemvx-order rows launch (k2v::gemv) in an explicit order; CERE_ERR_ARGUMENT outside its contract.

    int cere_ktest_gemm_rows_path(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t rows,
                                  int64_t out_f32, int64_t paths, int64_t iters, int64_t copies, void *y, void *info,
                                  void *us, cudaStream_t stream);
        P11R-21: y through a rows plan built at `rows` whose one-launch paths are `paths` (0 none, 1 the rows kernel,
        2 the strided batch, 3 both, -1 init_rows' default); info host int64 [3]: how m ran (0 one GEMM, 1 per row,
        2 rows kernel, 3 strided batch) and the plan's m = 1 algorithm (id, customOption); iters calls over `copies`
        weights back to back, us host double: mean µs per call after 5 warm-ups (iters > 5).

    int cere_ktest_gemm_m1_swap(const void *x, const void *w, int64_t n, int64_t k, int64_t out_f32, int64_t swap,
                                int64_t iters, int64_t copies, void *y, void *ids, void *us, cudaStream_t stream);
        P11R-21: y [1, n] through a max_m 1 plan initialised with m1_swap = `swap` (0: bucket 0's heuristic pick as
        main ran it); ids host int64 [2]: its m = 1 algorithm (id, customOption); iters, copies, us as above.

    int cere_ktest_plan_layout(int64_t max_sessions, int64_t max_context_tokens, int64_t kv_pool_tokens,
                               int64_t drafts, void *out, cudaStream_t stream);
        M2.9 (tasks/m29-spec.md M29S2-19/20, A1): device::plan_layout of a weight-less A4 text plan (host
        arithmetic); out host uint64 [6]: device_bytes, pinned_bytes, pool end (pool_offset + pool_bytes),
        undo_offset, undo_bytes (0 = no draft-only region), workspace_rows.

M2.9 K5 (tasks/m29-spec.md M29S2-05): `test_rows_plan_invariant_m16` repeats `test_rows_plan_invariant`'s fused
path with the rows plans built at 16 (the verify call's min(16, 4 x max_sessions)): for every m in 1..16 and every
row r < m, row r of the m-row call is bit-equal to the m = 1 call on that row alone. `DraftRegionTest` reports the
native-g0 plan's (R 4, context 200,000, pool 200,000) device estimate without and with drafts: drafts 0 ends at the
KV pool (no M2.9 region, A1) and drafts 4 adds exactly R x 3 undo snapshots after it.

tasks/m23-spec.md P11R-21: `test_rows_plan_lm_head` runs the same fused checks on the lm_head shape (n 248,320) with
the rows plan built at 16 and requires per_row(m) 0 for every m (one GEMM: the executor's RowsPlans::lm_head reads
the weight once per verify call or batched decode step). `test_rows_plan_mtp` runs them on the MTP draft layer's rows
plans (fc halves, attention, router, shared expert, the 65,536-row hot lm_head). `test_k2v_orders` reports, for every
target and MTP shape, the loaded libcublasLt version and the m = 1 algorithm (id, customOption) with the gemvx order
the rows kernel (k2_gemv.h) runs it in; requires libcublasLt 130101, no order under any other version, the rows kernel
bit-equal per row to the m = 1 call at m = 16, and another order giving different raw FP32 sums (perturbation, F32
output). `test_m1_swap` checks the m = 1 customOption swap (37 -> 75) bit for bit against the unswapped plan in BF16
and F32 output. `test_rows_paths` pins every shape's m = 1 algorithm (M1_PINS) and checks, on each rows path alone
(per row, the rows kernel, the strided batch) at m = 1..16, every row against the m = 1 call; `test_rows_paths_bench`
reports each path's µs per call (the per-(shape, m) choice).
"""
from __future__ import annotations

import ctypes
import unittest

import torch
import torch.nn.functional as F

import oracle
from sglang.srt.layers.activation import silu_and_mul

ROW = "K2"
M_VALUES = (1, 4, 7, 16, 2048)
REPEATS = 10
ATTN_LAYER = 3
GDN_LAYER = 0
SHARED_LAYER = 0
PLE_LAYER = 1  # model.h kPleLayer (0-based)
ISOLATE_SEED = 500_000
PIN_SEED = 600_000
PIN_MAX_M = 4096  # the largest forward call (model.h workspace_rows: kMaxChunkTokens)
MAX_PIN_RANGES = 32  # GemmPlan::kMaxPins
# k2_gemm.h bucket re-pins: plan shape -> (n, k, out_f32); attn_o shares gdn_out's shape and ple_key gdn_qkv's.
REPIN_SHAPES = {"gdn_qkv": (10240, 2560, False), "gdn_z": (6144, 2560, False), "gdn_out": (2560, 6144, False),
                "moe_router": (512, 2560, False), "moe_shared_down": (2560, 640, False),
                "hc_down": (320, 10240, True), "hc_up": (10240, 320, True)}
REPIN_CELL_M = (1217, 1633)  # the prefill chunks of ttft_cold_1.2K (1,217) and ttft_cold_18K's last (1,633)
REPIN_SEED = 1_300_000
CERE_OK = 0
CERE_ERR_UNSUPPORTED = -2  # cere_executor.h
ROWS_R = 4  # M2.6 rows R (m26 design note D3: plan max_sessions <= 4)
VERIFY_ROWS = 16  # M2.9 rows plans with drafts: min(16, 4 x max_sessions) (m29 design note M29S2-05)
ROWS_SEED = 700_000
ROWS_ALL_FUSED = -1  # fused_allowed: every m may run fused (the executor's rows plans)
ROWS_FORCE_PER_ROW = 1  # fused_allowed: only m = 1; every m >= 2 runs the per-row fallback
# m23 design note P11R-21: the rows-plan shapes that run every m in 1..16 as one launch on the executor's path (the
# m = 1 algorithm fused where AlgoCheck accepts m, else the gemvx-order rows kernel, k2_gemv.h, or the strided batch of
# the m = 1 algorithm): every target rows-plan shape.
GEMVX_ROWS_SHAPES = frozenset({"attn_qkv", "attn_o", "gdn_qkv", "gdn_z", "gdn_b", "gdn_a", "gdn_out", "moe_router",
                               "moe_shared_gate", "moe_shared_up", "moe_shared_down", "hc_down", "hc_up"})
WIDE_MIN_EXP = -20  # _wide_activations: magnitudes 2^-20 .. 2^0 times randn


def _weight(name: str, dims: int = 2) -> torch.Tensor:
    """A BF16 checkpoint weight of `dims` dimensions (2 for GEMM weights; 1 for hc_mix's hc_norm)."""
    w = oracle.real_tensor(name)
    if w.dtype != torch.bfloat16 or w.dim() != dims:
        raise oracle.OracleError(f"{name} is {w.dtype} {tuple(w.shape)}; expected {dims}-D BF16")
    return w.contiguous()


def _activations(m: int, k: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    return torch.randn((m, k), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)


def _wide_activations(m: int, k: int, seed: int) -> torch.Tensor:
    """x [m, k] BF16 of randn values each scaled by 2^e, e uniform in [WIDE_MIN_EXP, 0]: tasks/m23-spec.md P11R-21.
    On the checkpoint's dense weights (few significant bits) most FP32 sums of randn activations are exact, so every
    summation order gives the same bits (test_k2v_orders' other-lane-count launch matched the identified order on
    every element of gdn_b, gdn_a, moe_shared_up and the MTP router). Products spread over ~2^20 leave bits below a
    partial sum's last place, so each FP32 add rounds and the sum's bits depend on the order: a bit-equal row is
    evidence of the same order, not of exact arithmetic."""
    g = oracle.generator(seed)
    base = torch.randn((m, k), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    e = torch.randint(WIDE_MIN_EXP, 1, (m, k), generator=g, device=oracle.DEVICE).to(torch.float32)
    return (base * torch.exp2(e)).to(torch.bfloat16)


def _native_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = w.shape[0]
    y = oracle.empty((m, n), torch.bfloat16)
    oracle.call("cere_ktest_gemm_bf16", x, w, m, n, k, y)
    return y


def _native_lm_head_last(hidden: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = hidden.shape
    vocab = w.shape[0]
    logits = oracle.empty((vocab,), torch.bfloat16)
    oracle.call("cere_ktest_lm_head_last", hidden, w, m, vocab, k, logits)
    return logits


def _pin_ranges(n: int, k: int, out_f32: bool = False) -> list[tuple[int, int]]:
    """(m_lo, m_hi) of every pin range of the executor's (n, k, out) plan (k2_gemm.h gemm_pins_for)."""
    ranges = torch.zeros((MAX_PIN_RANGES, 2), dtype=torch.int64)
    count = torch.zeros((1,), dtype=torch.int64)
    oracle.call("cere_ktest_gemm_pin_ranges", n, k, int(out_f32), ranges, MAX_PIN_RANGES, count)
    return [(int(lo), int(hi)) for lo, hi in ranges[:int(count[0])].tolist()]


def _native_gemm_unpinned(x: torch.Tensor, w: torch.Tensor, out_f32: bool) -> torch.Tensor:
    """`cere_ktest_gemm_unpinned`: y [m, n] through a plan of max_m = m without pins (the bucket algorithm)."""
    m, k = x.shape
    n = w.shape[0]
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    oracle.call("cere_ktest_gemm_unpinned", x.contiguous(), w, m, n, k, int(out_f32), y)
    return y


def _full_mantissa_weight(n: int, k: int, seed: int) -> torch.Tensor:
    """w [n, k] BF16 of U(-1, 1) x 2^-5: every bit of each product's mantissa is set at random, so the FP32 sums round
    and their bits depend on the summation order (the checkpoint's dense weights have few significant bits)."""
    g = oracle.generator(seed)
    u = torch.rand((n, k), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    return ((u * 2.0 - 1.0) * 0.03125).to(torch.bfloat16).contiguous()


def _pins_init_status(n: int, k: int, max_m: int, version_offset: int) -> int:
    """The status of `cere_ktest_gemm_pins_init` (a refusal is the expected result, so it is returned, not raised)."""
    fn = oracle.ktest("cere_ktest_gemm_pins_init")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    status = fn(ctypes.c_int64(n), ctypes.c_int64(k), ctypes.c_int64(max_m), ctypes.c_int64(version_offset),
                ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(status)


def _identical_runs(first: torch.Tensor, run, repeats: int) -> int:
    """Number of runs (including `first`) whose bytes equal `first`'s."""
    reference = first.view(torch.int16)
    same = 1
    for _ in range(repeats - 1):
        same += int(torch.equal(run().view(torch.int16), reference))
    return same


def _native_gemm_rows(x: torch.Tensor, w: torch.Tensor, m: int, rows: int, out_f32: bool, fused_allowed: int):
    """`cere_ktest_gemm_rows` on x[:m]: (y [m, n], per_row(m) of the plan that ran)."""
    n, k = w.shape
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    per_row = torch.full((1,), -1, dtype=torch.int64)
    oracle.call("cere_ktest_gemm_rows", x[:m].contiguous(), w, m, n, k, rows, int(out_f32), fused_allowed, y,
                per_row)
    return y, int(per_row[0])


def _native_hc_mix_rows(hyper: torch.Tensor, mix: tuple, m: int, rows: int, fused_allowed: int) -> torch.Tensor:
    """hc_mix of hyper[:m] (`cere_ktest_hc_mix_rows`, rows > 0; `cere_ktest_hc_mix`, rows 0): normed | mixed per
    row as one BF16 [m, hc * hidden + hidden] tensor."""
    norm, down, up, streams, hidden, lowrank, eps = mix
    normed = oracle.empty((m, streams * hidden), torch.bfloat16)
    mixed = oracle.empty((m, hidden), torch.bfloat16)
    x = hyper[:m].contiguous()
    if rows == 0:
        oracle.call("cere_ktest_hc_mix", x, norm, down, up, m, streams, hidden, lowrank, float(eps), normed, mixed)
    else:
        oracle.call("cere_ktest_hc_mix_rows", x, norm, down, up, m, streams, hidden, lowrank, float(eps), normed,
                    mixed, rows, fused_allowed)
    return torch.cat([normed, mixed], dim=1)


def _m1_algo(n: int, k: int, out_f32: bool) -> dict[str, int]:
    """`cere_ktest_gemm_m1_algo`: the loaded libcublasLt's version, the executor plan's m = 1 algorithm (id and
    customOption) for (n, k, out) and the gemvx order k2v::order_of gives it (lanes T, chunk S; 0 0 = none)."""
    out = torch.zeros(5, dtype=torch.int64)
    oracle.call("cere_ktest_gemm_m1_algo", n, k, int(out_f32), out)
    return dict(zip(("cublaslt", "algo_id", "custom_option", "lanes", "chunk"), (int(v) for v in out.tolist())))


def _k2v_order(version: int, algo_id: int, custom_option: int, k: int) -> tuple[int, int]:
    """`cere_ktest_k2v_order`: k2v::order_of's (lanes, chunk) for an explicit libcublasLt version."""
    out = torch.zeros(2, dtype=torch.int64)
    oracle.call("cere_ktest_k2v_order", version, algo_id, custom_option, k, out)
    return int(out[0]), int(out[1])


def _k2v_gemv(x: torch.Tensor, w: torch.Tensor, out_f32: bool, lanes: int, chunk: int) -> torch.Tensor:
    """`cere_ktest_k2v_gemv`: one gemvx-order rows launch on x [m, k] in an explicit order."""
    m, k = x.shape
    n = w.shape[0]
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    oracle.call("cere_ktest_k2v_gemv", x.contiguous(), m, k, w, n, int(out_f32), lanes, chunk, y)
    return y


K2V_CUBLASLT = 130_101  # k2_gemv.h: the libcublasLt whose gemvx orders were identified (k2_gemm.cpp kPinsCublasLt)
MTP_LAYER = "mtp.layers.0."
MTP_HOT_ROWS = 65_536  # the -v10 artifact's draft token map (draft/token_map.u32le): the MTP hot lm_head's rows
MTP_HOT_SEED = 950_000


def _target_rows_shapes() -> list:
    """(label, loader, out_f32) of the target's rows-plan shapes (M2.6 D6: forward.cpp RowsPlans): attention qkv packed
    and o, GDN in-projections and out, PLE key/value (tasks/rows16-spec.md R16-A), router, shared gate/up/down, hc
    down/up with F32 output."""
    attn = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
    gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
    ple = oracle.layer_prefix(PLE_LAYER) + "ple."
    mlp = oracle.layer_prefix(SHARED_LAYER) + "mlp."
    hc = oracle.layer_prefix(GDN_LAYER) + "attn_hyper_connection."
    return [
        ("attn_qkv", lambda: torch.cat([_weight(attn + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                                       dim=0).contiguous(), False),
        ("attn_o", lambda: _weight(attn + "o_proj.weight"), False),
        ("gdn_qkv", lambda: _weight(gdn + "in_proj_qkv.weight"), False),
        ("gdn_z", lambda: _weight(gdn + "in_proj_z.weight"), False),
        ("gdn_b", lambda: _weight(gdn + "in_proj_b.weight"), False),
        ("gdn_a", lambda: _weight(gdn + "in_proj_a.weight"), False),
        ("gdn_out", lambda: _weight(gdn + "out_proj.weight"), False),
        ("ple_key", lambda: _weight(ple + "key_proj.weight"), False),
        ("ple_value", lambda: _weight(ple + "value_proj.weight"), False),
        ("moe_router", lambda: _weight(mlp + "gate.weight"), False),
        ("moe_shared_gate", lambda: _weight(mlp + "shared_expert.gate_proj.weight"), False),
        ("moe_shared_up", lambda: _weight(mlp + "shared_expert.up_proj.weight"), False),
        ("moe_shared_down", lambda: _weight(mlp + "shared_expert.down_proj.weight"), False),
        ("hc_down", lambda: _weight(hc + "input_mix_weight_down.weight"), True),
        ("hc_up", lambda: _weight(hc + "input_mix_weight_up.weight"), True),
    ]


def _mtp_rows_shapes() -> list:
    """(label, loader, out_f32) of the MTP draft layer's rows plans (mtp.cu instantiate: Workspace::Mtp): fc (its two
    halves, k 2,560), attention qkv packed and o, router, shared gate/up/down and the hot lm_head (the target lm_head's
    rows at the draft map; seeded distinct rows here, MTP_HOT_ROWS of them: the plan's algorithm depends on the
    shape, not on which rows)."""
    attn = MTP_LAYER + "self_attn."
    mlp = MTP_LAYER + "mlp."

    def hot_lm_head() -> torch.Tensor:
        full = _weight("lm_head.weight")
        g = torch.Generator().manual_seed(MTP_HOT_SEED)
        ids = torch.randperm(full.shape[0], generator=g)[:MTP_HOT_ROWS].sort().values.to(full.device)
        return full.index_select(0, ids).contiguous()

    return [
        ("mtp_fc_embedding", lambda: _weight("mtp.fc_embedding.weight"), False),
        ("mtp_fc_hidden", lambda: _weight("mtp.fc_hidden.weight"), False),
        ("mtp_attn_qkv", lambda: torch.cat([_weight(attn + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                                           dim=0).contiguous(), False),
        ("mtp_attn_o", lambda: _weight(attn + "o_proj.weight"), False),
        ("mtp_moe_router", lambda: _weight(mlp + "gate.weight"), False),
        ("mtp_moe_shared_gate", lambda: _weight(mlp + "shared_expert.gate_proj.weight"), False),
        ("mtp_moe_shared_up", lambda: _weight(mlp + "shared_expert.up_proj.weight"), False),
        ("mtp_moe_shared_down", lambda: _weight(mlp + "shared_expert.down_proj.weight"), False),
        ("mtp_lm_head_hot", hot_lm_head, False),
    ]


P11_MAX_M = 16  # k2s::kMaxM
P11_SEED = 800_000
P11_PREFILL_M = (17, 2048)
P11_ORACLE_M = (1, 4, 7, 16)
P11_BENCH_M = (1, 4, 16)
P11_BENCH_ITERS = 55  # 5 warm-ups + 50 timed
P11_GROUP = 16  # k2_small_m.cu kGroup: slices per CTA past 16 slices (one CTA per group, a cluster per column tile)
P11_COLD_BYTES = 384 << 20  # 3x the 128 MiB L2: every timed call streams its weight copy from DRAM


def _native_gemm_path(x: torch.Tensor, w: torch.Tensor, out_f32: bool, small_m: bool) -> torch.Tensor:
    """`cere_ktest_gemm_path`: y [m, n] through a plan of max_m = m, small-M kernel on or off (cuBLASLt)."""
    m, k = x.shape
    n = w.shape[0]
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    us = torch.zeros((1,), dtype=torch.float64)
    oracle.call("cere_ktest_gemm_path", x.contiguous(), w, m, n, k, int(out_f32), int(small_m), 1, 1, y, us)
    return y


def _time_gemm_path(x: torch.Tensor, w: torch.Tensor, out_f32: bool, small_m: bool) -> float:
    """Mean µs per call of one plan (`cere_ktest_gemm_path` with P11_BENCH_ITERS calls), cold: the calls rotate over
    enough copies of w that none is still in L2 when it is read again (a decode step reads each weight once)."""
    m, k = x.shape
    n = w.shape[0]
    copies = min(P11_BENCH_ITERS, -(-P11_COLD_BYTES // (w.numel() * w.element_size())))
    ws = w.unsqueeze(0).repeat(copies, 1, 1).contiguous() if copies > 1 else w
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    us = torch.zeros((1,), dtype=torch.float64)
    oracle.call("cere_ktest_gemm_path", x.contiguous(), ws, m, n, k, int(out_f32), int(small_m), P11_BENCH_ITERS,
                copies, y, us)
    del ws
    return float(us[0])


# m23 design note P11R-21: the rows plans' one-launch paths (GemmPlan::init_rows `paths`) and how an m ran
# (`cere_ktest_gemm_rows_path` info[0]).
PATHS_NONE = 0  # no one-launch path: an m AlgoCheck refuses on the rows layout runs per row
PATHS_GEMV = 1  # GemmPlan::kRowsGemv: the gemvx-order rows kernel (k2_gemv.h)
PATHS_BATCHED = 2  # GemmPlan::kRowsBatched: the m = 1 algorithm on a strided batch of m single-row GEMMs
PATHS_DEFAULT = -1  # init_rows' default paths (the executor's rows plans)
RAN_FUSED, RAN_PER_ROW, RAN_GEMV, RAN_BATCHED = 0, 1, 2, 3
GEMVX_ALGO = 13  # cuBLASLt's gemvx algoId
# The m = 1 algorithm (algoId, customOption) every K2 decode shape's executor plan runs under K2V_CUBLASLT, after
# the m = 1 customOption swap (k2_gemm.h gemm_m1_custom_option: 37 -> 75 at an identified equal order). A plan that
# picks anything else fails test_rows_paths: every rows path below is checked against the m = 1 call of this one.
M1_PINS = {
    "attn_qkv": (21, 0), "attn_o": (13, 75), "gdn_qkv": (21, 0), "gdn_z": (13, 75), "gdn_b": (13, 75),
    "gdn_a": (13, 75), "gdn_out": (13, 75), "moe_router": (13, 75), "moe_shared_gate": (13, 75),
    "moe_shared_up": (13, 75), "moe_shared_down": (13, 82), "hc_down": (13, 75), "hc_up": (21, 0),
    "ple_key": (21, 0), "ple_value": (13, 75),
    "lm_head": (13, 89), "mtp_fc_embedding": (13, 75), "mtp_fc_hidden": (13, 75), "mtp_attn_qkv": (21, 0),
    "mtp_attn_o": (13, 75), "mtp_moe_router": (13, 75), "mtp_moe_shared_gate": (13, 75),
    "mtp_moe_shared_up": (13, 75), "mtp_moe_shared_down": (13, 82), "mtp_lm_head_hot": (13, 71),
}
M1_SWAP = (37, 75)  # gemm_m1_custom_option: the heuristic's pick and the equal-order variant bucket 0 runs instead
ROWS_PATH_SEED = 1_100_000
# k2_gemm.cpp kRowsGemvFrom: per (n, k, out_f32), the first m at which init_rows' default paths take the rows kernel
# over the strided batch (the measured faster one; test_rows_paths_bench); below it and on every other shape the
# strided batch.
ROWS_GEMV_FROM = {
    (2560, 6144, False): 2, (6144, 2560, False): 2, (2560, 640, False): 2, (2560, 2560, False): 2,
    (248320, 2560, False): 2, (65536, 2560, False): 2, (512, 2560, False): 4, (512, 2560, True): 4,
    (640, 2560, False): 4, (320, 10240, True): 6,
}


def _k2_decode_shapes() -> list:
    """(label, loader, out_f32) of every K2 decode shape: the target's rows plans, the lm_head and the MTP draft
    layer's rows plans."""
    return _target_rows_shapes() + [("lm_head", lambda: _weight("lm_head.weight"), False)] + _mtp_rows_shapes()


def _cold_copies(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(w repeated back to back, copies): enough copies that none is still in L2 when it is read again."""
    copies = min(P11_BENCH_ITERS, -(-P11_COLD_BYTES // (w.numel() * w.element_size())))
    return (w.unsqueeze(0).repeat(copies, 1, 1).contiguous() if copies > 1 else w), copies


def _native_rows_path(x: torch.Tensor, w: torch.Tensor, m: int, rows: int, out_f32: bool, paths: int,
                      iters: int = 1, copies: int = 1):
    """`cere_ktest_gemm_rows_path` on x[:m] (w: `copies` weights back to back): (y [m, n], how m ran, the rows plan's
    m = 1 (algoId, customOption), µs per call when iters > 5)."""
    k = x.shape[1]
    n = w.shape[-2]
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    info = torch.full((3,), -1, dtype=torch.int64)
    us = torch.zeros((1,), dtype=torch.float64)
    oracle.call("cere_ktest_gemm_rows_path", x[:m].contiguous(), w, m, n, k, rows, int(out_f32), paths, iters, copies,
                y, info, us)
    return y, int(info[0]), (int(info[1]), int(info[2])), float(us[0])


def _native_m1_swap(x: torch.Tensor, w: torch.Tensor, out_f32: bool, swap: bool, iters: int = 1, copies: int = 1):
    """`cere_ktest_gemm_m1_swap` on x [1, k]: (y [1, n], the plan's m = 1 (algoId, customOption), µs per call when
    iters > 5) of a max_m 1 plan with GemmPlan::init's m1_swap = `swap`."""
    k = x.shape[1]
    n = w.shape[-2]
    y = oracle.empty((1, n), torch.float32 if out_f32 else torch.bfloat16)
    ids = torch.full((2,), -1, dtype=torch.int64)
    us = torch.zeros((1,), dtype=torch.float64)
    oracle.call("cere_ktest_gemm_m1_swap", x.contiguous(), w, n, k, int(out_f32), int(swap), iters, copies, y, ids,
                us)
    return y, (int(ids[0]), int(ids[1])), float(us[0])


def _native_gemm_segments(x: torch.Tensor, weights: list, out_f32: bool) -> list:
    """`cere_ktest_gemm_segments`: one gemm_segments call over `weights` sharing x; the outputs in order."""
    m, k = x.shape
    ys = [oracle.empty((m, w.shape[0]), torch.float32 if out_f32 else torch.bfloat16) for w in weights]
    w_ptrs = torch.tensor([w.data_ptr() for w in weights], dtype=torch.int64)
    ns = torch.tensor([w.shape[0] for w in weights], dtype=torch.int64)
    y_ptrs = torch.tensor([y.data_ptr() for y in ys], dtype=torch.int64)
    oracle.call("cere_ktest_gemm_segments", x.contiguous(), m, k, len(weights), w_ptrs, ns, y_ptrs, int(out_f32))
    return ys


def _weight_shape(name: str) -> tuple[int, int]:
    return tuple(_weight(name).shape)


def _k2s_geometry(k: int, ns: list) -> tuple[list, list, list]:
    """`cere_ktest_k2s_geometry`: the (grid, block, cluster) dims of the small-M launch over widths ns at k."""
    dims = torch.zeros(9, dtype=torch.int64)
    oracle.call("cere_ktest_k2s_geometry", k, len(ns), torch.tensor(ns, dtype=torch.int64), dims)
    v = dims.tolist()
    return v[0:3], v[3:6], v[6:9]


def _p11_shapes() -> list:
    """(label, loader, out_f32) of every dense decode GEMM shape (forward.cpp / k7_moe.cu / k3_hc.cu plans)."""
    attn = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
    gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
    mlp = oracle.layer_prefix(SHARED_LAYER) + "mlp."
    hc = oracle.layer_prefix(GDN_LAYER) + "attn_hyper_connection."
    return [
        ("attn_qkv", lambda: torch.cat([_weight(attn + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                                       dim=0).contiguous(), False),
        ("attn_o", lambda: _weight(attn + "o_proj.weight"), False),
        ("gdn_qkv", lambda: _weight(gdn + "in_proj_qkv.weight"), False),
        ("gdn_z", lambda: _weight(gdn + "in_proj_z.weight"), False),
        ("gdn_b", lambda: _weight(gdn + "in_proj_b.weight"), False),
        ("gdn_a", lambda: _weight(gdn + "in_proj_a.weight"), False),
        ("gdn_out", lambda: _weight(gdn + "out_proj.weight"), False),
        ("moe_router", lambda: _weight(mlp + "gate.weight"), True),
        ("moe_shared_gate", lambda: _weight(mlp + "shared_expert.gate_proj.weight"), False),
        ("moe_shared_up", lambda: _weight(mlp + "shared_expert.up_proj.weight"), False),
        ("moe_shared_down", lambda: _weight(mlp + "shared_expert.down_proj.weight"), False),
        ("hc_down", lambda: _weight(hc + "input_mix_weight_down.weight"), True),
        ("hc_up", lambda: _weight(hc + "input_mix_weight_up.weight"), True),
        ("lm_head", lambda: _weight("lm_head.weight"), False),
    ]


def _check_rows(tc: unittest.TestCase, case: str, w, seed: int, out_f32: bool, fused_allowed: int, run_rows,
                run_single, *, width: int | None = None, rows: int = ROWS_R) -> None:
    """K1 for one shape and path: x [rows, k] of distinct seeded rows; for each m in 1..rows (and for m >= 2 again
    with row m - 1 holding NaN/Inf), every finite row r < m of run_rows(x, m, rows, fused_allowed) equals
    run_single(x[r:r + 1]) bit for bit; per_row(m) reported, and 1 for m >= 2 on the forced per-row path."""
    k = width if w is None else w.shape[1]
    x = _activations(rows, k, seed)
    singles = [run_single(x[r:r + 1].contiguous()) for r in range(rows)]
    poisoned = x.clone()
    poisoned[:, : k // 2] = float("nan")
    poisoned[:, k // 2:] = float("inf")
    for m in range(1, rows + 1):
        inputs = [("clean", x)]
        if m >= 2:
            dirty = x.clone()
            dirty[m - 1] = poisoned[m - 1]
            inputs.append(("poisoned", dirty))
        for label, data in inputs:
            y, per_row = run_rows(data, m, rows, fused_allowed)
            if per_row is not None:
                oracle.report_row(ROW, f"{case}_M{m}_{label}_per_row", per_row=per_row)
                if fused_allowed == ROWS_FORCE_PER_ROW:
                    oracle.check_count(tc, ROW, f"{case}_M{m}_{label}_forced_per_row", per_row,
                                       expected=1 if m >= 2 else 0)
                elif m == 1:
                    oracle.check_count(tc, ROW, f"{case}_M1_{label}_fused", per_row, expected=0)
            finite_rows = m - 1 if label == "poisoned" else m
            for r in range(finite_rows):
                oracle.check_bit_exact(tc, ROW, f"{case}_M{m}_{label}_row{r}", y[r:r + 1], singles[r])


def _isolation_row(layer: int, op: str, native: torch.Tensor, control: torch.Tensor, high_precision: torch.Tensor,
                   control_kernel: str) -> None:
    """One report-only D6 isolation row: `c` = rel-L2(control, FP32 GEMM), `limit` = max(FLOOR_OP, 2c), `measured` =
    rel-L2(native, control) and the differing elements on identical inputs; `within` = measured <= limit. Asserts
    nothing (tasks/m22d-spec.md C7)."""
    c = oracle.rel_l2(control, high_precision)
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    measured = oracle.rel_l2(native, control)
    differing = int((native.view(torch.int16) != control.view(torch.int16)).sum().item())
    oracle.report_row(ROW, f"isolate_L{layer}_{op}_M1", phase="isolate", layer=layer, op=op, m=1,
                      control=control_kernel, c=c, limit=limit, measured=measured, within=measured <= limit,
                      elements=native.numel(), differing=differing)


def _isolate_gemm(layer: int, op: str, x: torch.Tensor, w: torch.Tensor, control_kernel: str) -> None:
    """Isolation row of one GEMM: the control's `F.linear` (`UnquantizedLinearMethod.apply`) vs `cere_ktest_gemm_bf16`
    on the same BF16 input and real weight."""
    control = F.linear(x, w)
    high_precision = torch.matmul(x.float(), w.float().t())
    _isolation_row(layer, op, _native_gemm(x, w), control, high_precision, control_kernel)


def _check_gemm(tc: unittest.TestCase, label: str, w: torch.Tensor, seed: int) -> None:
    n, k = w.shape
    w32 = w.float()
    for m in M_VALUES:
        with tc.subTest(weight=label, m=m):
            case = f"{label}_M{m}"
            x = _activations(m, k, seed + m)
            reference_op = torch.matmul(x, w.t())
            high_precision = torch.matmul(x.float(), w32.t())
            c = oracle.calibrate(ROW, case, reference_op, high_precision, oracle.FLOOR_OP)
            del high_precision
            limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
            y = _native_gemm(x, w)
            oracle.check_within(tc, ROW, case, oracle.rel_l2(y, reference_op), c=c, limit=limit)
            del reference_op
            same = _identical_runs(y, lambda: _native_gemm(x, w), REPEATS)
            oracle.check_count(tc, ROW, f"{case}_repeats_identical", same, expected=REPEATS)


class K2Gemm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The FP32 calibration GEMM must be true FP32 (no TF32).
        cls.previous_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        cls.cfg = oracle.text_config()

    @classmethod
    def tearDownClass(cls):
        torch.set_float32_matmul_precision(cls.previous_precision)

    def test_attention_projections_layer3(self):
        prefix = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
        for i, proj in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
            _check_gemm(self, f"L{ATTN_LAYER}_{proj}", _weight(prefix + proj + ".weight"), 10_000 * (i + 1))

    def test_gdn_projections_layer0(self):
        prefix = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
        for i, proj in enumerate(("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")):
            _check_gemm(self, f"L{GDN_LAYER}_{proj}", _weight(prefix + proj + ".weight"), 100_000 + 10_000 * i)

    def test_shared_expert_layer0(self):
        prefix = oracle.layer_prefix(SHARED_LAYER) + "mlp."
        names = ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert_gate")
        for i, proj in enumerate(names):
            _check_gemm(self, f"L{SHARED_LAYER}_{proj}", _weight(prefix + proj + ".weight"), 200_000 + 10_000 * i)

    def test_lm_head_full(self):
        w = _weight("lm_head.weight")
        self.assertEqual(tuple(w.shape), (self.cfg["vocab_size"], self.cfg["hidden_size"]))
        _check_gemm(self, "lm_head", w, 300_000)

    def test_lm_head_last_row_only(self):
        w = _weight("lm_head.weight")
        vocab, k = w.shape
        self.assertEqual((vocab, k), (self.cfg["vocab_size"], self.cfg["hidden_size"]))
        w32 = w.float()
        for m in M_VALUES:
            with self.subTest(m=m):
                case = f"lm_head_last_M{m}"
                hidden = _activations(m, k, 400_000 + m)
                last = hidden[-1:].contiguous()
                reference_op = torch.matmul(last, w.t())[0]
                high_precision = torch.matmul(last.float(), w32.t())[0]
                c = oracle.calibrate(ROW, case, reference_op, high_precision, oracle.FLOOR_OP)
                limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                logits = _native_lm_head_last(hidden, w)
                oracle.check_within(self, ROW, case, oracle.rel_l2(logits, reference_op), c=c, limit=limit)
                oracle.check_bit_exact(self, ROW, f"{case}_equals_gemm_M1", logits, _native_gemm(last, w)[0])
                same = _identical_runs(logits, lambda: _native_lm_head_last(hidden, w), REPEATS)
                oracle.check_count(self, ROW, f"{case}_repeats_identical", same, expected=REPEATS)

    # -- lever 2 pins (m22p design note D7 lever 2): every pin range, and the cuBLASLt version guard ---------------

    def test_shared_gate_up_pin_ranges(self):
        """Both ends of every pin range of the shared gate/up shape (up to PIN_MAX_M): the native gate and up GEMMs
        equal torch's one F.linear over [gate; up] (reduced-precision BF16 reduction on, SGLang's default) column for
        column, bit for bit, and repeat bit-identically."""
        prefix = oracle.layer_prefix(SHARED_LAYER) + "mlp."
        w_gate = _weight(prefix + "shared_expert.gate_proj.weight")
        w_up = _weight(prefix + "shared_expert.up_proj.weight")
        n, k = w_gate.shape
        self.assertEqual((n, k), (self.cfg["shared_expert_intermediate_size"], self.cfg["hidden_size"]))
        w_gate_up = torch.cat([w_gate, w_up], dim=0).contiguous()
        ranges = _pin_ranges(n, k)
        self.assertTrue(ranges, "the shared gate/up shape has no pin ranges")
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        try:
            for m_lo, m_hi in ranges:
                ends = sorted({m_lo, min(m_hi, PIN_MAX_M)}) if m_lo <= PIN_MAX_M else []
                for m in ends:
                    with self.subTest(m=m):
                        case = f"L{SHARED_LAYER}_shared_gate_up_pin_{m_lo}_{m_hi}_M{m}"
                        x = _activations(m, k, PIN_SEED + m)
                        dispatched = F.linear(x, w_gate_up)
                        gate = _native_gemm(x, w_gate)
                        up = _native_gemm(x, w_up)
                        oracle.check_bit_exact(self, ROW, f"{case}_gate_equals_fused", gate,
                                               dispatched[:, :n].contiguous())
                        oracle.check_bit_exact(self, ROW, f"{case}_up_equals_fused", up,
                                               dispatched[:, n:].contiguous())
                        same = min(_identical_runs(gate, lambda: _native_gemm(x, w_gate), REPEATS),
                                   _identical_runs(up, lambda: _native_gemm(x, w_up), REPEATS))
                        oracle.check_count(self, ROW, f"{case}_repeats_identical", same, expected=REPEATS)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous

    def test_attn_qkv_pin_ranges(self):
        """Both ends of every pin range of the attn_qkv shape (up to PIN_MAX_M; tasks/m22p-spec.md M22PS-R3): the
        native GEMM over the packed q|k|v weight of layer ATTN_LAYER equals torch's F.linear over the same weight
        (reduced-precision BF16 reduction on, SGLang's default) bit for bit, and repeats bit-identically."""
        prefix = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
        w_qkv = torch.cat([_weight(prefix + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                          dim=0).contiguous()
        n, k = w_qkv.shape
        ranges = _pin_ranges(n, k)
        self.assertTrue(ranges, "the attn_qkv shape has no pin ranges")
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        try:
            for m_lo, m_hi in ranges:
                ends = sorted({m_lo, min(m_hi, PIN_MAX_M)}) if m_lo <= PIN_MAX_M else []
                for m in ends:
                    with self.subTest(m=m):
                        case = f"L{ATTN_LAYER}_attn_qkv_pin_{m_lo}_{m_hi}_M{m}"
                        x = _activations(m, k, PIN_SEED + m)
                        native = _native_gemm(x, w_qkv)
                        oracle.check_bit_exact(self, ROW, f"{case}_equals_dispatched", native, F.linear(x, w_qkv))
                        same = _identical_runs(native, lambda: _native_gemm(x, w_qkv), REPEATS)
                        oracle.check_count(self, ROW, f"{case}_repeats_identical", same, expected=REPEATS)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous

    def test_bucket_repins(self):
        """tasks/r3-prefill-spec.md: each bucket re-pin gives the bucket algorithm's bits (module docstring)."""
        for shape, (n, k, out_f32) in REPIN_SHAPES.items():
            ranges = _pin_ranges(n, k, out_f32)
            self.assertTrue(ranges, f"{ROW} {shape} ({n} x {k}) has no bucket re-pins")
            w = _full_mantissa_weight(n, k, REPIN_SEED + n + k)
            for m_lo, m_hi in ranges:
                for m in sorted({m_lo, m_hi} | {m for m in REPIN_CELL_M if m_lo <= m <= m_hi}):
                    with self.subTest(shape=shape, m=m):
                        case = f"repin_{shape}_{m_lo}_{m_hi}_M{m}"
                        x = _wide_activations(m, k, REPIN_SEED + m)
                        pinned = _native_gemm_path(x, w, out_f32, small_m=False)
                        oracle.check_bit_exact(self, ROW, f"{case}_equals_bucket", pinned,
                                               _native_gemm_unpinned(x, w, out_f32))
                        same = _identical_runs(pinned, lambda: _native_gemm_path(x, w, out_f32, small_m=False),
                                               REPEATS)
                        oracle.check_count(self, ROW, f"{case}_repeats_identical", same, expected=REPEATS)

    def test_pins_refuse_other_cublaslt(self):
        """GemmPlan::init takes each pinned shape's pins under the cuBLASLt version they name and refuses them
        (CERE_ERR_UNSUPPORTED) under any other: the pins with their recorded version shifted by -1 and +1."""
        hidden = int(self.cfg["hidden_size"])
        prefix = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
        qkv_n = sum(int(oracle.real_tensor(prefix + p + ".weight", device="cpu").shape[0])
                    for p in ("q_proj", "k_proj", "v_proj"))
        shapes = {"shared_gate_up": int(self.cfg["shared_expert_intermediate_size"]), "attn_qkv": qkv_n}
        for plan, n in shapes.items():
            for offset, expected in ((0, CERE_OK), (-1, CERE_ERR_UNSUPPORTED), (1, CERE_ERR_UNSUPPORTED)):
                with self.subTest(plan=plan, offset=offset):
                    status = _pins_init_status(n, hidden, PIN_MAX_M, offset)
                    oracle.report_row(ROW, f"{plan}_pins_version_offset_{offset}", status=status,
                                      expected=expected, passed=status == expected)
                    self.assertEqual(status, expected, f"{ROW} {plan} pins init with cuBLASLt version offset {offset}")

    # -- D6 isolation rows (m22d design note C7): decode M = 1, every layer of the block's type, report-only -------

    def test_isolate_decode_attention_projections(self):
        """qkv_proj (one GEMM over the packed q|k|v weight) and o_proj of every full-attention layer at M = 1."""
        layers = [i for i, t in enumerate(self.cfg["layer_types"]) if t == "full_attention"]
        for layer in layers:
            with self.subTest(layer=layer):
                prefix = oracle.layer_prefix(layer) + "self_attn."
                # QKVParallelLinear stacking; the native forward runs the same packed weight (forward.cpp
                # full_attention, ws.attn_qkv).
                w_qkv = torch.cat([_weight(prefix + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                                  dim=0).contiguous()
                x = _activations(1, w_qkv.shape[1], ISOLATE_SEED + 10 * layer)
                _isolate_gemm(layer, "qkv_proj", x, w_qkv, "F.linear(x, w_qkv)")
                del w_qkv
                w_o = _weight(prefix + "o_proj.weight")
                _isolate_gemm(layer, "o_proj", _activations(1, w_o.shape[1], ISOLATE_SEED + 10 * layer + 1), w_o,
                              "F.linear(attn, w_o)")

    def test_isolate_decode_moe_gemms(self):
        """Router logits (mlp.gate), shared expert gate/up and down of every layer at M = 1."""
        for layer in range(int(self.cfg["num_hidden_layers"])):
            with self.subTest(layer=layer):
                prefix = oracle.layer_prefix(layer) + "mlp."
                seed = ISOLATE_SEED + 100_000 + 10 * layer
                x = _activations(1, int(self.cfg["hidden_size"]), seed)
                _isolate_gemm(layer, "router", x, _weight(prefix + "gate.weight"), "F.linear(x, mlp.gate)")
                w_gate = _weight(prefix + "shared_expert.gate_proj.weight")
                w_up = _weight(prefix + "shared_expert.up_proj.weight")
                # SGLang: one MergedColumnParallelLinear GEMM over [gate; up]; native: the shared_gate_up plan run
                # once with each weight (k7_moe.h MoePlans).
                w_gate_up = torch.cat([w_gate, w_up], dim=0).contiguous()
                control = F.linear(x, w_gate_up)
                high_precision = torch.matmul(x.float(), w_gate_up.float().t())
                native = torch.cat([_native_gemm(x, w_gate), _native_gemm(x, w_up)], dim=-1)
                _isolation_row(layer, "shared_gate_up", native, control, high_precision,
                               "F.linear(x, cat(gate, up))")
                # down_proj on the control's own activation (silu_and_mul of its gate_up), the same bytes both sides.
                act = silu_and_mul(control).contiguous()
                _isolate_gemm(layer, "shared_down", act, _weight(prefix + "shared_expert.down_proj.weight"),
                              "F.linear(silu_and_mul(gate_up), down)")

    # -- M2.6 K1 (m26 design note D6, I6): rows plans -----------------------------------------------------------

    def _rows_invariant(self, rows: int, paths: tuple, prefix: str, one_launch: frozenset = frozenset(),
                        shapes: list | None = None, hc_mix: bool = True) -> None:
        """`_check_rows` at `rows` for every rows-plan shape (`shapes`, default the target's) and, with `hc_mix`,
        hc_mix itself, on each (path, fused_allowed) of `paths`; the shapes in `one_launch` must run every m as one
        launch on the ROWS_ALL_FUSED path (per_row(m) 0)."""
        hc = oracle.layer_prefix(GDN_LAYER) + "attn_hyper_connection."
        for index, (label, load, out_f32) in enumerate(_target_rows_shapes() if shapes is None else shapes):
            w = load()
            for path, fused_allowed in paths:
                per_rows = []

                def run_rows(x, m, r, f, w=w, out_f32=out_f32, per_rows=per_rows):
                    y, per_row = _native_gemm_rows(x, w, m, r, out_f32, f)
                    per_rows.append((m, per_row))
                    return y, per_row

                with self.subTest(shape=label, path=path, rows=rows):
                    _check_rows(self, f"{prefix}_{label}_{path}", w, ROWS_SEED + 1_000 * index, out_f32,
                                fused_allowed, run_rows,
                                lambda x: _native_gemm_rows(x, w, 1, 0, out_f32, ROWS_ALL_FUSED)[0], rows=rows)
                    if label in one_launch and fused_allowed == ROWS_ALL_FUSED:
                        for m, per_row in per_rows:
                            oracle.check_count(self, ROW, f"{prefix}_{label}_M{m}_one_launch", per_row, expected=0)
            del w
        if not hc_mix:
            return
        norm = _weight(hc + "hc_norm.weight", dims=1)
        down = _weight(hc + "input_mix_weight_down.weight")
        up = _weight(hc + "input_mix_weight_up.weight")
        streams, lowrank = int(self.cfg["hc_count"]), int(self.cfg["hc_lowrank"])
        hidden, eps = int(self.cfg["hidden_size"]), float(self.cfg["rms_norm_eps"])
        mix = (norm, down, up, streams, hidden, lowrank, eps)
        for path, fused_allowed in paths:
            with self.subTest(shape="hc_mix", path=path, rows=rows):
                _check_rows(self, f"{prefix}_hc_mix_{path}", None, ROWS_SEED + 90_000, False, fused_allowed,
                            lambda x, m, r, f: (_native_hc_mix_rows(x, mix, m, r, f), None),
                            lambda x: _native_hc_mix_rows(x, mix, 1, 0, ROWS_ALL_FUSED),
                            width=streams * hidden, rows=rows)

    def test_rows_plan_invariant(self):
        """K1: every D6 rows-plan shape (attn_qkv packed, attn_o, gdn qkv/z/b/a/out, moe_router, shared gate/up/down,
        hc down/up with F32 output, and hc_mix itself on HcPlans::init_rows): for m in 1..ROWS_R and every row r < m,
        row r of the rows plan's m-row call is bit-equal to the instantiated plan's m = 1 call on that row alone,
        on the fused path and on the forced per-row fallback; the rows hold distinct random data and the last row of
        each m >= 2 call one NaN/Inf row (no cross-row contamination); per_row(m) is reported per m (the per-m
        bound: an m whose AlgoCheck fails, and only that m, runs per row) and is 1 for every m >= 2 when forced."""
        self._rows_invariant(ROWS_R, (("fused", ROWS_ALL_FUSED), ("per_row", ROWS_FORCE_PER_ROW)), "rows")

    def test_rows_plan_invariant_m16(self):
        """M2.9 K5 (M29S2-05): test_rows_plan_invariant's checks with the rows plans built at VERIFY_ROWS = 16, m in
        1..16, on the executor's path (every m may run fused; the per-row fallback is the m = 1 plan per row).
        tasks/m23-spec.md P11R-21: every target rows-plan shape (GEMVX_ROWS_SHAPES) runs every m as one launch (the
        m = 1 algorithm fused, or where AlgoCheck refuses m the K2 gemvx-order rows kernel or the strided batch of the
        m = 1 algorithm), still bit-equal per row to the m = 1 call."""
        self._rows_invariant(VERIFY_ROWS, (("fused", ROWS_ALL_FUSED),), "rows16", GEMVX_ROWS_SHAPES)

    def test_rows_plan_lm_head(self):
        """tasks/m23-spec.md P11R-21 (RowsPlans::lm_head): test_rows_plan_invariant_m16's checks on the lm_head shape
        (n 248,320, k 2,560, BF16 logits) with the rows plan built at VERIFY_ROWS, and every m in 2..16 runs as ONE
        launch (per_row(m) 0: AlgoCheck refuses the m = 1 algorithm, a gemvx variant, at m >= 2 on the rows layout, so
        the K2 gemvx-order rows kernel or the strided batch of that algorithm runs it), so a verify call or a batched
        decode step makes one lm_head call instead of m."""
        w = _weight("lm_head.weight")
        per_rows = []

        def run_rows(x, m, rows, fused_allowed):
            y, per_row = _native_gemm_rows(x, w, m, rows, False, fused_allowed)
            per_rows.append((m, per_row))
            return y, per_row

        _check_rows(self, "rows16_lm_head_fused", w, ROWS_SEED + 95_000, False, ROWS_ALL_FUSED, run_rows,
                    lambda x: _native_gemm_rows(x, w, 1, 0, False, ROWS_ALL_FUSED)[0], rows=VERIFY_ROWS)
        for m, per_row in per_rows:
            oracle.check_count(self, ROW, f"rows16_lm_head_M{m}_one_gemm", per_row, expected=0)

    def test_rows_plan_mtp(self):
        """tasks/m23-spec.md P11R-21: test_rows_plan_invariant_m16's checks on the MTP draft layer's rows plans
        (_mtp_rows_shapes: fc halves, attention qkv/o, router, shared gate/up/down, the hot lm_head) at VERIFY_ROWS:
        for every m in 1..16 each row of the m-row call is bit-equal to the m = 1 call on that row alone, and every
        shape runs every m as one launch."""
        shapes = _mtp_rows_shapes()
        self._rows_invariant(VERIFY_ROWS, (("fused", ROWS_ALL_FUSED),), "rows16",
                             frozenset(label for label, _, _ in shapes), shapes, hc_mix=False)

    def test_k2v_orders(self):
        """tasks/m23-spec.md P11R-21: every K2 decode shape of the target (rows plans and lm_head) and of the MTP draft
        layer reports the loaded libcublasLt's version and its m = 1 algorithm (id, customOption) with the gemvx
        order the rows kernel runs it in (`k2v_<shape>`). The loaded library must be K2V_CUBLASLT (the identified
        one); every identified order maps to no order under any other version (k2v::order_of refuses: the rows
        plans then keep cuBLASLt's per-row calls, never a guessed order). Per identified shape, 16 seeded rows of
        _wide_activations (sums that round, so the bits depend on the order): the rows kernel in the identified order
        is bit-equal per row to the m = 1 call (`k2v_<shape>_M16_row<r>`). The perturbation runs in F32 output, the
        raw FP32 sums (a BF16 output can round two orders' sums to the same bits on every element of a narrow shape:
        gdn_b/gdn_a's 48 columns): the identified order's F32 sums are bit-equal per row to the F32 plan's m = 1 call
        where that plan runs the same order (`k2v_<shape>_f32_M16_row<r>`), and the same launch in another order
        (another chunk at the same lanes for a blocked order, else the other lane count) differs from them in at
        least one element (`k2v_<shape>_perturbed`: the bit check tells the orders apart on every shape)."""
        shapes = _k2_decode_shapes()
        identified = 0
        for index, (label, load, out_f32) in enumerate(shapes):
            w = load()
            n, k = w.shape
            with self.subTest(shape=label):
                info = _m1_algo(n, k, out_f32)
                oracle.report_row(ROW, f"k2v_{label}", n=n, k=k, out_f32=out_f32, **info)
                self.assertEqual(info["cublaslt"], K2V_CUBLASLT, f"{ROW} k2v_{label}: libcublasLt version")
                order = (info["lanes"], info["chunk"])
                self.assertEqual(_k2v_order(info["cublaslt"], info["algo_id"], info["custom_option"], k), order)
                if info["lanes"] == 0:
                    continue
                identified += 1
                for version in (info["cublaslt"] - 1, info["cublaslt"] + 1, 120_000, 130_200):
                    self.assertEqual(_k2v_order(version, info["algo_id"], info["custom_option"], k), (0, 0),
                                     f"{ROW} k2v_{label}: order under libcublasLt {version}")
                x = _wide_activations(VERIFY_ROWS, k, ROWS_SEED + 200_000 + 1_000 * index)
                y = _k2v_gemv(x, w, out_f32, *order)
                singles = [_native_gemm_rows(x[r:r + 1], w, 1, 0, out_f32, ROWS_ALL_FUSED)[0]
                           for r in range(VERIFY_ROWS)]
                for r in range(VERIFY_ROWS):
                    oracle.check_bit_exact(self, ROW, f"k2v_{label}_M{VERIFY_ROWS}_row{r}", y[r:r + 1], singles[r])
                sums = _k2v_gemv(x, w, True, *order)
                info32 = _m1_algo(n, k, True)
                same_order = (info32["lanes"], info32["chunk"]) == order
                oracle.report_row(ROW, f"k2v_{label}_f32", same_order=same_order, algo_id=info32["algo_id"],
                                  custom_option=info32["custom_option"])
                if same_order:
                    for r in range(VERIFY_ROWS):
                        single = _native_gemm_rows(x[r:r + 1], w, 1, 0, True, ROWS_ALL_FUSED)[0]
                        oracle.check_bit_exact(self, ROW, f"k2v_{label}_f32_M{VERIFY_ROWS}_row{r}", sums[r:r + 1],
                                               single)
                other = (info["lanes"], 1) if info["chunk"] != 1 else (8 if info["lanes"] == 32 else 32, 1)
                perturbed = _k2v_gemv(x, w, True, *other)
                differing = int((perturbed.view(torch.int32) != sums.view(torch.int32)).sum())
                oracle.report_row(ROW, f"k2v_{label}_perturbed", lanes=other[0], chunk=other[1], differing=differing,
                                  elements=sums.numel())
                self.assertTrue(differing != 0, f"{ROW} k2v_{label}: order {other} gave the identified order's sums")
            del w
        oracle.report_row(ROW, "k2v_identified", shapes=len(shapes), identified=identified)
        self.assertTrue(identified != 0, f"{ROW} k2v_identified: no shape runs an identified gemvx order")

    def test_m1_swap(self):
        """tasks/m23-spec.md P11R-21 (k2_gemm.h gemm_m1_custom_option): for every K2 decode shape, in BF16 and in F32
        output, the plan initialised without the swap (m1_swap false: bucket 0's heuristic pick, main's algorithm)
        and the executor's plan run the same m = 1 algorithm except where the pick is gemvx customOption 37, which
        the executor's plan runs as customOption 75; there both plans' m = 1 calls are bit-equal on 16 seeded rows of
        _wide_activations (F32: the raw FP32 sums). Report-only: µs per call of both plans with cold weights
        (`m1_swap_<shape>_bench`)."""
        swapped = 0
        for index, (label, load, _) in enumerate(_k2_decode_shapes()):
            w = load()
            k = w.shape[1]
            x = _wide_activations(VERIFY_ROWS, k, ROWS_PATH_SEED + 500_000 + 1_000 * index)
            for out_f32 in (False, True):
                case = f"m1_swap_{label}_{'f32' if out_f32 else 'bf16'}"
                with self.subTest(shape=label, out_f32=out_f32):
                    before = _native_m1_swap(x[:1], w, out_f32, False)[1]
                    after = _native_m1_swap(x[:1], w, out_f32, True)[1]
                    expected = (GEMVX_ALGO, M1_SWAP[1]) if before == (GEMVX_ALGO, M1_SWAP[0]) else before
                    oracle.report_row(ROW, case, pick_algo_id=before[0], pick_custom_option=before[1],
                                      algo_id=after[0], custom_option=after[1])
                    self.assertEqual(after, expected, f"{ROW} {case}: the executor plan's m = 1 algorithm")
                    if after == before:
                        continue
                    swapped += 1
                    ref = torch.cat([_native_m1_swap(x[r:r + 1], w, out_f32, False)[0] for r in range(VERIFY_ROWS)])
                    got = torch.cat([_native_m1_swap(x[r:r + 1], w, out_f32, True)[0] for r in range(VERIFY_ROWS)])
                    oracle.check_bit_exact(self, ROW, f"{case}_rows{VERIFY_ROWS}", got, ref)
                    ws, copies = _cold_copies(w)
                    us = [_native_m1_swap(x[:1], ws, out_f32, s, P11_BENCH_ITERS, copies)[2] for s in (False, True)]
                    oracle.report_row(ROW, f"{case}_bench", phase="bench", us_pick=us[0], us_swapped=us[1])
                    del ws
            del w
        oracle.report_row(ROW, "m1_swap_shapes", swapped=swapped)
        self.assertTrue(swapped != 0, f"{ROW} m1_swap_shapes: no shape's m = 1 pick is swapped")

    def _rows_paths_case(self, label: str, w: torch.Tensor, out_f32: bool, seed: int, pin: tuple | None) -> None:
        """test_rows_paths for one shape and output type; `pin` the required m = 1 algorithm (None: the plan's own,
        reported)."""
        n, k = w.shape
        dtype = "f32" if out_f32 else "bf16"
        info = _m1_algo(n, k, out_f32)
        m1 = (info["algo_id"], info["custom_option"])
        oracle.report_row(ROW, f"paths_{label}_{dtype}", algo_id=m1[0], custom_option=m1[1], lanes=info["lanes"],
                          chunk=info["chunk"])
        if pin is not None:
            self.assertEqual(m1, pin, f"{ROW} paths_{label}_{dtype}: the executor plan's m = 1 algorithm")
        x = _wide_activations(VERIFY_ROWS, k, seed)
        singles = torch.cat([_native_gemm_rows(x[r:r + 1], w, 1, 0, out_f32, ROWS_ALL_FUSED)[0]
                             for r in range(VERIFY_ROWS)])
        gemvx = m1[0] == GEMVX_ALGO
        gemv_from = ROWS_GEMV_FROM.get((n, k, out_f32), VERIFY_ROWS + 1) if info["lanes"] > 0 else VERIFY_ROWS + 1
        expected_ran = {PATHS_NONE: lambda m: RAN_PER_ROW,
                        PATHS_GEMV: lambda m: RAN_GEMV if info["lanes"] > 0 else RAN_PER_ROW,
                        PATHS_BATCHED: lambda m: RAN_BATCHED,
                        PATHS_DEFAULT: lambda m: RAN_GEMV if m >= gemv_from else RAN_BATCHED}
        for name, paths in (("per_row", PATHS_NONE), ("gemv", PATHS_GEMV), ("batched", PATHS_BATCHED),
                            ("default", PATHS_DEFAULT)):
            for m in range(1, VERIFY_ROWS + 1):
                case = f"paths_{label}_{dtype}_{name}_M{m}"
                y, ran, rows_m1, _ = _native_rows_path(x, w, m, VERIFY_ROWS, out_f32, paths)
                self.assertEqual(rows_m1, m1, f"{ROW} {case}: the rows plan's m = 1 algorithm")
                if m == 1:
                    oracle.check_count(self, ROW, f"{case}_ran", ran, expected=RAN_FUSED)
                elif gemvx:
                    oracle.check_count(self, ROW, f"{case}_ran", ran, expected=expected_ran[paths](m))
                else:
                    oracle.report_row(ROW, f"{case}_ran", ran=ran)
                oracle.check_bit_exact(self, ROW, case, y, singles[:m])

    def test_rows_paths(self):
        """tasks/m23-spec.md P11R-21: every K2 decode shape (_k2_decode_shapes) runs its pinned m = 1 algorithm
        (M1_PINS, in the instantiated plan and the rows plan: an algorithm selection that falls back to another fails
        here, and GemmPlan::init refuses the plan when the swap's variant is unavailable), and on each one-launch path
        alone (PATHS_NONE: per row; PATHS_GEMV: the gemvx-order rows kernel; PATHS_BATCHED: the strided batch) and on
        init_rows' default paths (PATHS_DEFAULT, the executor's), with the rows plan built at VERIFY_ROWS, for every m
        in 1..16 every row of the m-row call is bit-equal to the m = 1 call on that row alone (rows of
        _wide_activations, sums that round), in the executor's output type and, for every gemvx shape, in F32 output
        (the raw FP32 sums, where a different summation order shows; that plan's own m = 1 algorithm). How m ran is
        required for the gemvx shapes: m = 1 one GEMM; m >= 2 per row on PATHS_NONE, one rows-kernel launch on
        PATHS_GEMV where the order is identified (else per row), one strided-batched cublasLtMatmul on PATHS_BATCHED,
        and on PATHS_DEFAULT the rows kernel from ROWS_GEMV_FROM's m on (where the order is identified), else the
        strided batch."""
        for index, (label, load, out_f32) in enumerate(_k2_decode_shapes()):
            w = load()
            seed = ROWS_PATH_SEED + 1_000 * index
            with self.subTest(shape=label, out_f32=out_f32):
                self._rows_paths_case(label, w, out_f32, seed, M1_PINS[label])
            if M1_PINS[label][0] == GEMVX_ALGO and not out_f32:
                with self.subTest(shape=label, out_f32=True):
                    self._rows_paths_case(label, w, True, seed, None)
            del w

    def test_rows_paths_bench(self):
        """Report-only (tasks/m23-spec.md P11R-21: the per-(shape, m) path choice): µs per call with cold weights of
        every gemvx K2 decode shape's rows plan, in the executor's output type, at m in 2..16 on each path alone
        (per row, the rows kernel where its order is identified, the strided batch) and on the executor's default
        paths (`paths_bench_<shape>_M<m>`: us_<path>, how the default ran)."""
        for index, (label, load, out_f32) in enumerate(_k2_decode_shapes()):
            if M1_PINS[label][0] != GEMVX_ALGO:
                continue
            w = load()
            n, k = w.shape
            lanes = _m1_algo(n, k, out_f32)["lanes"]
            ws, copies = _cold_copies(w)
            x = _activations(VERIFY_ROWS, k, ROWS_PATH_SEED + 300_000 + 1_000 * index)
            for m in range(2, VERIFY_ROWS + 1):
                fields = {}
                for name, paths in (("per_row", PATHS_NONE), ("gemv", PATHS_GEMV), ("batched", PATHS_BATCHED),
                                    ("default", PATHS_DEFAULT)):
                    if paths == PATHS_GEMV and lanes == 0:
                        continue
                    _, ran, _, us = _native_rows_path(x, ws, m, VERIFY_ROWS, out_f32, paths, P11_BENCH_ITERS, copies)
                    fields[f"us_{name}"] = us
                    fields[f"ran_{name}"] = ran
                oracle.report_row(ROW, f"paths_bench_{label}_M{m}", phase="bench", n=n, k=k, **fields)
            del ws, w

    # -- P11 (m23 design note P11): the small-M kernel -------------------------------------------------------------

    def test_p11_small_m_batch_invariant(self):
        """Every dense shape: row r of each m-row call (m in 2..16) is bit-equal to the m = 1 call on row r alone; the
        M = 16 call repeats 10 times bit-identically."""
        for index, (label, load, out_f32) in enumerate(_p11_shapes()):
            w = load()
            k = w.shape[1]
            bits = torch.int32 if out_f32 else torch.int16
            with self.subTest(shape=label):
                x = _activations(P11_MAX_M, k, P11_SEED + 1_000 * index)
                singles = torch.cat([_native_gemm_path(x[r:r + 1], w, out_f32, True) for r in range(P11_MAX_M)])
                invariant = 0
                for m in range(2, P11_MAX_M + 1):
                    y = _native_gemm_path(x[:m], w, out_f32, True)
                    same = torch.equal(y.view(bits), singles[:m].view(bits))
                    oracle.report_row(ROW, f"p11_{label}_M{m}_rows_equal_M1", equal=same)
                    invariant += int(same)
                oracle.check_count(self, ROW, f"p11_{label}_batch_invariant", invariant, expected=P11_MAX_M - 1)
                same = _identical_runs(_native_gemm_path(x, w, out_f32, True),
                                       lambda: _native_gemm_path(x, w, out_f32, True), REPEATS)
                oracle.check_count(self, ROW, f"p11_{label}_M16_repeats_identical", same, expected=REPEATS)
            del w

    def test_p11_small_m_oracle(self):
        """The small-M kernel (switched on per plan, the k2_gemm lever's small_m path) against torch.matmul on every dense decode
        shape at m in P11_ORACLE_M under the unchanged tolerance rule (limit = max(4e-3, 2c))."""
        for index, (label, load, out_f32) in enumerate(_p11_shapes()):
            w = load()
            k = w.shape[1]
            w32 = w.float()
            for m in P11_ORACLE_M:
                with self.subTest(shape=label, m=m):
                    case = f"p11_{label}_M{m}_oracle"
                    x = _activations(m, k, P11_SEED + 70_000 + 1_000 * index + m)
                    reference_op = torch.matmul(x, w.t())
                    high_precision = torch.matmul(x.float(), w32.t())
                    c = oracle.calibrate(ROW, case, reference_op, high_precision, oracle.FLOOR_OP)
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                    y = _native_gemm_path(x, w, out_f32, True)
                    oracle.check_within(self, ROW, case, oracle.rel_l2(y.float(), reference_op.float()), c=c,
                                        limit=limit)
            del w, w32

    def test_p11_prefill_unchanged(self):
        """m > 16 keeps cuBLASLt: the plan's call equals the same plan with the small-M kernel off, bit for bit."""
        for index, (label, load, out_f32) in enumerate(_p11_shapes()):
            w = load()
            k = w.shape[1]
            for m in P11_PREFILL_M:
                with self.subTest(shape=label, m=m):
                    x = _activations(m, k, P11_SEED + 50_000 + 1_000 * index + m)
                    oracle.check_bit_exact(self, ROW, f"p11_{label}_M{m}_prefill_unchanged",
                                           _native_gemm_path(x, w, out_f32, True),
                                           _native_gemm_path(x, w, out_f32, False))
            del w

    def test_p11_fused_segments(self):
        """gemm_segments over the GDN in-projections (qkv, z, b, a) and the shared gate/up equals each weight's own
        small-M call bit for bit, at m in {1, 4, 16}."""
        gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
        mlp = oracle.layer_prefix(SHARED_LAYER) + "mlp."
        groups = [
            ("gdn_in", [_weight(gdn + p + ".weight") for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]),
            ("shared_gate_up", [_weight(mlp + p + ".weight") for p in ("shared_expert.gate_proj",
                                                                       "shared_expert.up_proj")]),
        ]
        for index, (label, weights) in enumerate(groups):
            for m in (1, 4, P11_MAX_M):
                with self.subTest(group=label, m=m):
                    x = _activations(m, weights[0].shape[1], P11_SEED + 90_000 + 1_000 * index + m)
                    fused = torch.cat(_native_gemm_segments(x, weights, False), dim=1)
                    alone = torch.cat([_native_gemm_path(x, w, False, True) for w in weights], dim=1)
                    oracle.check_bit_exact(self, ROW, f"p11_{label}_M{m}_fused", fused, alone)

    def test_p11_layout_fills_gpu(self):
        """P11R-12/P11R-13: every dense decode call (each P11 shape alone, the GDN in-projections and the shared
        gate/up as their fused launches) runs one CTA per group of 16 slices (cluster = slices / 16 past 16 slices)
        and gives every SM a CTA when its 8-column tiles times its groups can: grid >= min(SMs, that bound)."""
        sms = torch.cuda.get_device_properties(oracle.DEVICE).multi_processor_count
        gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
        mlp = oracle.layer_prefix(SHARED_LAYER) + "mlp."
        calls = [(label, w.shape[1], [w.shape[0]]) for label, w in
                 ((label, load()) for label, load, _ in _p11_shapes() if label != "lm_head")]
        calls.append(("gdn_in", oracle.text_config()["hidden_size"],
                      [_weight_shape(gdn + p + ".weight")[0] for p in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                                                                        "in_proj_a")]))
        calls.append(("shared_gate_up", oracle.text_config()["hidden_size"],
                      [_weight_shape(mlp + f"shared_expert.{p}_proj.weight")[0] for p in ("gate", "up")]))
        filled = 0
        for label, k, ns in calls:
            grid, block, cluster = _k2s_geometry(k, ns)
            slices = block[0] // 32 * cluster[0]
            groups = slices // P11_GROUP if slices > P11_GROUP else 1
            bound = min(sms, sum(-(-n // 8) for n in ns) * groups)
            ok = cluster[0] == groups and grid[0] >= bound and grid[0] % cluster[0] == 0
            oracle.report_row(ROW, f"p11_{label}_layout", k=k, n=sum(ns), sms=sms, ctas=grid[0], warps=block[0] // 32,
                              cluster=cluster[0], slices=slices, bound=bound, passed=ok)
            filled += int(ok)
        oracle.check_count(self, ROW, "p11_layout_fills_gpu", filled, expected=len(calls))

    def test_p11_layout_invariant(self):
        """P11R-12: every P11 shape's own call (its layout: 8-column tiles when it has few columns) equals the same
        weight launched with a neighbour wide enough for 16-column tiles, bit for bit, at m in {1, 16}."""
        sms = torch.cuda.get_device_properties(oracle.DEVICE).multi_processor_count
        for index, (label, load, out_f32) in enumerate(_p11_shapes()):
            if label == "lm_head":  # its own layout is already the wide one
                continue
            w = load()
            k = w.shape[1]
            wide = _activations(16 * sms, k, P11_SEED + 120_000 + index)
            for m in (1, P11_MAX_M):
                with self.subTest(shape=label, m=m):
                    x = _activations(m, k, P11_SEED + 110_000 + 1_000 * index + m)
                    fused = _native_gemm_segments(x, [w, wide], out_f32)[0]
                    alone = _native_gemm_path(x, w, out_f32, True)
                    oracle.check_bit_exact(self, ROW, f"p11_{label}_M{m}_layout_invariant", alone, fused)
            del w, wide

    def test_p11_bench(self):
        """Report-only: µs per call and effective TB/s (weight + activation + output bytes) of the small-M kernel and
        of cuBLASLt per dense shape at m in {1, 4, 16}."""
        for index, (label, load, out_f32) in enumerate(_p11_shapes()):
            w = load()
            n, k = w.shape
            for m in P11_BENCH_M:
                x = _activations(m, k, P11_SEED + 70_000 + index)
                moved = 2 * n * k + 2 * m * k + (4 if out_f32 else 2) * m * n
                small = _time_gemm_path(x, w, out_f32, True)
                cublas = _time_gemm_path(x, w, out_f32, False)
                oracle.report_row(ROW, f"p11_bench_{label}_M{m}", phase="bench", n=n, k=k, m=m, small_us=small,
                                  cublaslt_us=cublas, small_tbps=moved / small / 1e6, cublaslt_tbps=moved / cublas / 1e6)
            del w


NATIVE_G0_SESSIONS = 4  # configs/runtime/native-g0.toml rows
NATIVE_G0_CONTEXT = 200_000  # its max_context_tokens and kv_pool_tokens
NATIVE_G0_POOL = 200_000
UNDO_SNAPSHOTS = 3  # model.h UndoState: undo[j - 1] for verify rows j = 1..3
REGION_ALIGN = 256  # memory_plan.h kRegionAlignment


def _plan_layout(drafts: int) -> dict[str, int]:
    out = torch.zeros(9, dtype=torch.int64)
    oracle.call("cere_ktest_plan_layout", NATIVE_G0_SESSIONS, NATIVE_G0_CONTEXT, NATIVE_G0_POOL, drafts, out)
    keys = ("device_bytes", "pinned_bytes", "pool_end", "undo_offset", "undo_bytes", "workspace_rows",
            "spec_sampler_offset", "spec_sampler_bytes", "verify_rows")
    return dict(zip(keys, (int(v) for v in out.tolist())))


def _align(v: int) -> int:
    return (v + REGION_ALIGN - 1) // REGION_ALIGN * REGION_ALIGN


class DraftRegionTest(unittest.TestCase):
    def test_draft_region_estimate(self):
        """M2.9 (M29S2-19/20, A1): the weight-less native-g0 plan without drafts carves no draft-only region (its
        device arena ends at the KV pool, as before M2.9); with drafts 4 the region of R x 3 undo snapshots starts at
        the next 256-byte boundary, then the verify rows' sampler stages (phase S, D6: min(16, 4 R) rows of fp32
        processed logits and probabilities plus keep masks) end the arena."""
        off, on = _plan_layout(0), _plan_layout(4)
        oracle.report_row(ROW, "draft region estimate", phase="m29", sessions=NATIVE_G0_SESSIONS,
                          context=NATIVE_G0_CONTEXT, device_bytes_drafts0=off["device_bytes"],
                          device_bytes_drafts4=on["device_bytes"], delta=on["device_bytes"] - off["device_bytes"],
                          undo_bytes=on["undo_bytes"], spec_sampler_bytes=on["spec_sampler_bytes"],
                          verify_rows=on["verify_rows"], pinned_drafts0=off["pinned_bytes"],
                          pinned_drafts4=on["pinned_bytes"])
        for key in ("undo_bytes", "undo_offset", "spec_sampler_offset", "spec_sampler_bytes", "verify_rows"):
            self.assertEqual(off[key], 0, key)
        self.assertEqual(off["device_bytes"], _align(off["pool_end"]))
        self.assertEqual(on["undo_bytes"], 59_034_880)
        self.assertEqual(on["undo_offset"], _align(on["pool_end"]))
        self.assertEqual(on["verify_rows"], min(16, 4 * NATIVE_G0_SESSIONS))
        self.assertEqual(on["spec_sampler_offset"],
                         _align(on["undo_offset"] + NATIVE_G0_SESSIONS * UNDO_SNAPSHOTS * on["undo_bytes"]))
        # Per row: processed logits and probabilities fp32 [vocab], the keep mask, then the shared buffers.
        vocab = 248_320
        self.assertTrue(on["spec_sampler_bytes"] >= on["verify_rows"] * (2 * 4 * vocab + vocab // 8))
        self.assertEqual(on["device_bytes"], _align(on["spec_sampler_offset"] + on["spec_sampler_bytes"]))


if __name__ == "__main__":
    unittest.main()
