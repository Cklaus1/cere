"""Levers-off identity (ABI v5, PerfInt lever registry `crates/cere-runtime/src/levers.rs`): with every perf lever at
its default (null slot selection: k2_gemm CERE_LEVER_K2_CUBLASLT, prefill_attention CERE_LEVER_PREFILL_CUDA_CORE), this
build's kernels are byte-identical to the merge-base library's. The registry names these methods as the contracts of
its bit-identical default impls (`k2_gemm/cublaslt`, `prefill_attention/cuda_core`; K3 has no lever, its row guards
the HcPlans GEMMs the k2_gemm lever switches): making the selections runtime levers (one library, the K2 small-M GEMM
and the P4 prefill kernel compiled into every build, per-plan `use_small_m` and `Workspace::prefill_p4`) must not
move a default bit.

Suite `levers` (`native_dev.py [--gpu 1] test --suite levers`; pattern `test_levers_*.py`). The reference is the
merge-base libcere_ktest.so of the L0 convention (tasks/m22p-spec.md L0, as test_model_k8_forward.py
`test_forward_trace_equals_merge_base`): `$CERE_BASE_KTEST_LIB`, default `build/native-l0/libcere_ktest.so` under
`/src`, a default build of merge-base(main, HEAD) (pre-v4 main: CERE_K2_SMALL_M off, CERE_PREFILL_ATTENTION
cuda_core). Its absence, or the build's own library given as the reference, fails the test (never a skip). Every
case runs the same entry point on identical inputs, first through this build's library and then through the
reference (`oracle.load_ktest` swapped for the call), and `oracle.check_bit_exact` compares every output byte.

Cases (ROW "LEV"; seeded BF16 activations, real checkpoint weights):
* `test_k2_default_equals_main`: every dense GEMM shape the model plans (layer 3's packed attn q|k|v, o_proj and QSA
  index_qk_proj; layer 0's GDN in-projections qkv/z/b/a and out_proj, MoE router (F32), shared expert gate/up/down,
  attn HC down/up (F32)) at m in 1..16, 17, 64, 517, 2,048 and 4,096 (the pin ranges up to the largest forward
  call): `cere_ktest_gemm_bf16` (BF16 shapes) and `cere_ktest_gemm_rows` with rows 0 (the instantiated plan,
  max_m = m), and the executor's rows plans (`cere_ktest_gemm_rows`, rows 4 and 16, every m <= rows, all m fused
  allowed: M2.6 rows and the M2.9 verify rows) with their per_row(m), which must be 0 (tasks/m23-spec.md P11R-21:
  every rows plan runs m rows as one launch, the k2v gemvx-order kernel or the m = 1 algorithm's strided batch,
  where the merge-base library before P11R-21 ran m single-row calls; the merge-base's per_row(m) is reported);
  `cere_ktest_lm_head_last` at m in 1, 4, 16,
  517 (the default lm_head path: one m = 1 call per row; `run_rows` takes its batched m = R call of
  Workspace::lm_head into rows_logits only under small_m) and the lm_head shape through the rows plans (rows 4 and 16
  at m in 1, 4, 16, the MTP lm_head rows plans' path; cuBLASLt under the default lever).
* `test_k2_small_m_equals_main` (tasks/m23-spec.md P11R-21): the k2_gemm lever's `small_m` impl keeps the numerics
  the quality gate judged. `cere_ktest_gemm_path` with small_m 1 on every shape above and lm_head, m in 1..16, and
  `cere_ktest_gemm_segments` (one small-M launch) over layer 0's GDN in-projections (qkv, z, b, a) and shared expert
  gate|up, m in 1..16, equal the merge-base library's small-M kernel byte for byte.
* `test_k3_default_equals_main`: `cere_ktest_hc_mix` (HcPlans, the instantiated plans) and `cere_ktest_hc_mix_rows`
  (rows 4 and 16) on layer 0's attn and mlp hyper-connections and the final mixer, m in 1..16, 17 and 517, and
  `cere_ktest_hc_combine` on both per-layer sites.
* `test_hc_combine_norm_equals_main` (tasks/m23-spec.md P5R): `cere_ktest_hc_combine_norm` (this build only: the
  combine fused with the next mix's per-stream norm) at both per-layer sites into the next mix (attn -> mlp,
  mlp -> the final mixer), m in 1..16 (the fused kernel), 17 and 517 (two launches), out separate and in place:
  [out, normed] equal the merge-base library's `cere_ktest_hc_combine` output and the normed of its
  `cere_ktest_hc_mix` over that output, byte for byte.
* `test_prefill_default_equals_main`: `cere_ktest_attn_block` (K6, the default prefill_attention lever: CUDA-core
  kernels for every call) on layer 3's weights, without a prefix (T in 1, 4, 16, 17, 517, 2,048) and through the
  FP8 cache after a prefix (P/T 517/4, 480/37, 2,031/17, 1,024/1,024, 2,047/1): attn_out, out and both caches;
  `cere_ktest_qsa_attention` (K9) of R rows at the end of a 4,500-token sequence (R in 1, 16, 17, 128, 517) with
  BF16 K/V and with the FP8 cache at permuted slots.
* `test_decode_attention_equals_main` (tasks/m23-spec.md P5R): K6's decode/verify attention through the FP8 cache,
  M query rows (1..16) after a prefix of P in 1, 15, 127, 1,200 and 2,032 (one key split up to 16 splits):
  `cere_ktest_attn_block` (eager dense_attention: split_rows_kernel for M 1-3, attention_kernel<4> above, and the split
  combine) of this build equals the merge-base library's (attn_out, out, both caches). The decode graph's step / MTP
  extend (`dense_attention_at`) and the verify chain (`dense_attention_rows`, `_rows_at`) run the same arithmetic with
  the position on the device; on this build, through `cere_ktest_attn_dense` over the caches the block left, each equals
  the eager call it replaces (M rows at P; per row t the one-token call at P + t) byte for byte, so with the base
  library's own contracts (the same equalities) each equals the merge-base kernel.
* `test_decode_attention_edges_equals_main` (AttnSplit: the one-token FP8 calls run k6's split_rows_kernel, query-head
  groups of a KV head per block): `cere_ktest_attn_dense` in all four modes (eager, at, rows, rows_at) of M rows in
  1, 2, 3, 4, 5, 8, 16 after a prefix of P in 1, 127, 128, 129, 1,023, 1,200, 1,535, 2,032 (one, two and 9..16 key
  splits), over FP8 caches holding signed zeros, denormals, +-448 and one NaN key and value byte, equals the
  merge-base library's byte for byte.
* `test_decode_attention_split_blocks` (AttnSplit, fail-first on the merge-base library: symbol absent):
  `cere_ktest_attn_split_blocks` reports 6 blocks per key split for a decode step and 4 per row and split for a
  verify chain of 1..4 rows (the merge-base tile kernel runs one block per KV head, 2).
* `test_attention_chains_equals_per_chain` (tasks/rows16-spec.md R16-C, this build only: symbol absent on the
  merge-base library): `cere_ktest_attn_chains` mode 1 (one k6::prologue_chains_at + k6::dense_attention_chains_at
  over the verify graph's chains of several sequences, each on its own FP8 caches and slot map at its own prefix from
  the edge prefixes above, caches with the edge bytes) equals mode 0 (attention_chain's per-chain prologue_rows_at +
  dense_attention_rows_at) byte for byte: attention output and both caches, layouts of 1..16 rows in 1..6 chains.
* `test_top2_equals_main` (tasks/m23-spec.md P5R): `cere_ktest_top2` (K1 top2, the greedy sampler's argmax and
  runner-up) of R rows (1..16) at the model vocab 248,320 (16-byte aligned rows), at 248,321 (rows after the first
  misaligned, a tail) and at 7 equals the merge-base library's ids and scores; the rows cycle through massive ties,
  one and several NaNs, -inf rows, mostly -inf rows and signed zeros.
* `test_k1_pdl_equals_main` (tasks/m23-spec.md P5R, SmallFuse): the K1 kernels the decode path launches as
  programmatic dependent launches, m in 1, 4, 16 (`cere_ktest_embedding`, `cere_ktest_rmsnorm` Gemma over the hidden
  row and plain per head, `cere_ktest_rmsnorm_gated` per GDN head, `cere_ktest_sigmoid_gate`,
  `cere_ktest_kv_store_fp8`, `cere_ktest_attn_prologue` = qk_norm_rope_gate then the KV store, and
  `cere_ktest_copy_segments`): one input is written by `cere_ktest_pdl_late_copy` (this build only), which lets the
  stream's next kernel launch at once and writes its destination PDL_LATE_NS later, the K1 call enqueued right behind
  it with no wait between; every output equals the merge-base library's plain launch on the written input byte for
  byte. A K1 kernel that touched its input before its griddepcontrol.wait reads the stale bytes and fails the case.
* `test_gdn_chain_equals_main` (tasks/m23-spec.md DG-02): the GDN mixer core of a verify chain (no lever) on layer
  0's weights, after a GDN_PREFILL-token PREFILL, for chains of GDN_CHAIN_ROWS (1..4) rows:
  `cere_ktest_gdn_layer_core` DECODE once per row (the merge-base verify's GDN step) of this build equals the
  merge-base library's, and this build's `cere_ktest_gdn_chain_core` (the whole chain in one conv, one delta-rule and
  one norm launch, the undo snapshots written in-kernel) equals the merge-base library's per-row calls: y, both final
  states and the state after every row but the last (what the merge-base verify copied to its undo snapshots), byte
  for byte.

Native entry points (oracle.py calling convention; prototypes in test_kernels_k1_primitives.py,
test_kernels_k2_gemm.py, test_kernels_k3_hc.py, test_kernels_k6_attn.py and test_kernels_k9_qsa.py):

    int cere_ktest_gemm_bf16(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                             cudaStream_t stream);
    int cere_ktest_gemm_rows(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t rows,
                             int64_t out_f32, int64_t fused_allowed, void *y, void *per_row, cudaStream_t stream);
    int cere_ktest_gemm_path(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t out_f32,
                             int64_t small_m, int64_t iters, int64_t copies, void *y, void *us, cudaStream_t stream);
    int cere_ktest_gemm_segments(const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs,
                                 const int64_t *ns, const int64_t *y_ptrs, int64_t out_f32, cudaStream_t stream);
    int cere_ktest_lm_head_last(const void *hidden, const void *w, int64_t m, int64_t vocab, int64_t k, void *logits,
                                cudaStream_t stream);
    int cere_ktest_hc_mix(const void *hyper, const void *norm_weight, const void *w_down, const void *w_up, int64_t m,
                          int64_t hc, int64_t hidden, int64_t lowrank, float eps, void *normed, void *mixed,
                          cudaStream_t stream);
    int cere_ktest_hc_mix_rows(..., as cere_ktest_hc_mix ..., int64_t rows, int64_t fused_allowed,
                               cudaStream_t stream);
    int cere_ktest_hc_combine(const void *block_out, const void *hyper, const void *normed, const void *w_inject,
                              int64_t m, int64_t hc, int64_t hidden, void *out, cudaStream_t stream);
    int cere_ktest_hc_combine_norm(const void *block_out, const void *hyper, void *normed, const void *w_inject,
                                   const void *next_norm, int64_t m, int64_t hc, int64_t hidden, float eps,
                                   void *out, cudaStream_t stream);  /* this build only */
    int cere_ktest_attn_block(const void *hidden, const void *w_q, const void *w_k, const void *w_v, const void *w_o,
                              const void *q_norm, const void *k_norm, const void *cos_sin, int64_t n_pos,
                              const void *positions, const void *slots, int64_t prefix_len, int64_t T, void *k_cache,
                              void *v_cache, int64_t n_slots, int64_t kv_from_cache, int64_t hidden_size,
                              int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim, int64_t rotary_dim, float eps,
                              float kv_scale, float softmax_scale, void *attn_out, void *out, cudaStream_t stream);
    int cere_ktest_qsa_attention(const void *q, int64_t rows, const void *indices, const void *k, const void *v,
                                 const void *k_cache, const void *v_cache, const void *slots, float scale, void *out,
                                 cudaStream_t stream);
    int cere_ktest_attn_dense(const void *q, const void *gate, int64_t tokens, int64_t pos0, const void *pos,
                              const void *k_cache, const void *v_cache, const void *slots, int64_t mode, void *out,
                              cudaStream_t stream);  /* modes in ktest/k6_ktest.cu */
    int cere_ktest_attn_split_blocks(int64_t rows, int64_t chain, void *blocks /* int64 [1], host */,
                                     cudaStream_t stream);  /* this build only */
    int cere_ktest_attn_chains(const void *q_gate, const void *k, const void *v, const void *q_norm,
                               const void *k_norm, const void *cos_sin, const void *positions, const void *slots,
                               void *k_caches, void *v_caches, int64_t n_slots, const int64_t *chain_rows /* host */,
                               int64_t chains, float eps, int64_t mode, void *out,
                               cudaStream_t stream);  /* this build only; modes in ktest/k6_ktest.cu */
    int cere_ktest_top2(const void *logits, int64_t n_rows, int64_t vocab, void *ids, void *scores,
                        cudaStream_t stream);
    int cere_ktest_pdl_late_copy(const void *src, void *dst, int64_t bytes, int64_t spin_ns,
                                 cudaStream_t stream);  /* this build only: dst = src after spin_ns, see above */
    int cere_ktest_gdn_layer_core(const void *mixed_qkv, const void *z, const void *b, const void *a,
                                  const void *conv_weight, const void *A_log, const void *dt_bias,
                                  const void *norm_weight, void *conv_state, void *ssm_state, void *y,
                                  int64_t tokens, int64_t operation, cudaStream_t stream);
    int cere_ktest_gdn_chain_core(..., as cere_ktest_gdn_layer_core through ssm_state ..., void *conv_saves,
                                  void *ssm_saves, void *y, int64_t rows, cudaStream_t stream);  /* this build only */
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import unittest
from pathlib import Path

import torch

import oracle

ROW = "LEV"
BASE_KTEST_LIB = Path(os.environ.get("CERE_BASE_KTEST_LIB", str(oracle.SRC / "build/native-l0/libcere_ktest.so")))
SEED = 900_000
ATTN_LAYER = 3
GDN_LAYER = 0
SHARED_LAYER = 0
HC_LAYER = 0
K2_M = tuple(range(1, 17)) + (17, 64, 517, 2048, 4096)
SMALL_M = tuple(range(1, 17))  # k2s::kMaxM: every m the small-M kernel serves
ROWS_PLANS = (4, 16)  # M2.6 rows R (max_sessions 4) and the M2.9 verify rows min(16, 4 x max_sessions)
ROWS_ALL_FUSED = -1  # fused_allowed: every m may run fused (the executor's rows plans)
LM_HEAD_M = (1, 4, 16, 517)
HC_M = tuple(range(1, 17)) + (17, 517)
HC_STREAM_SCALES = (1.0, 0.5, 2.0, 4.0)
# K6 (k6_attention.h): 24 Q heads, 2 KV heads, head_dim 256, rotary_dim 64, scale 256^-0.5, kv_scale 1.0, dense
# contexts <= 2,048.
N_Q_HEADS = 24
N_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
SOFTMAX_SCALE = HEAD_DIM ** -0.5
K6_CONTEXT = 2048
K6_NO_PREFIX = (1, 4, 16, 17, 517, 2048)
K6_PREFIX = ((517, 4), (480, 37), (2031, 17), (1024, 1024), (2047, 1))
K6_DECODE_PREFIX = (1, 15, 127, 1200, 2032)
K6_DECODE_M = tuple(range(1, 17))
# AttnSplit: one-token FP8-cache attention at key-split edges (128 keys per split: 1, 2 and 9 .. 16 splits) and verify
# widths around the chain kernel's row-group switch (rows <= 4 / > 4); special E4M3 bytes in the caches.
K6_EDGE_PREFIX = (1, 127, 128, 129, 1023, 1200, 1535, 2032)
K6_EDGE_M = (1, 2, 3, 4, 5, 8, 16)
# R16-C: verify graph row layouts (rows per sequence's chain, 1..16 rows; both row-group widths, one-row chains).
K6_CHAINS_LAYOUTS = ((1,), (2, 1), (1, 1, 1, 1), (4, 4, 4, 4), (1, 3, 2, 5, 1, 4), (5, 3, 7, 1), (8, 8), (16,))
K6_EDGE_SPECIALS = (0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x7E, 0xFE)  # +-0, +-min denormal, +-max denormal, +-448
K6_NAN = 0x7F
# Query-head blocks per (row, key split) of the one-token FP8 attention: a decode step / MTP extend and a verify
# chain of the speculation window (rows <= 4); the merge-base kernel runs one block per KV head (2).
K6_STEP_BLOCKS = 6
K6_CHAIN_BLOCKS = 4
K6_WINDOW_ROWS = (1, 2, 3, 4)
TOP2_VOCABS = (248_320, 248_321, 7)
TOP2_ROWS = tuple(range(1, 17))
# The rotary table's values select no kernel; a NeoX cos|sin table of this theta keeps the rotated rows finite.
ROPE_THETA = 10_000.0
# K9 (k9_qsa.h): 2,051 selected logical positions per query row (2,048 + ratio 4 - 1), -1 padding after.
K9_ROW_WIDTH = 2051
K9_LENGTH = 4500
K9_ROWS = (1, 16, 17, 128, 517)
# K5 (k5_gdn_layer.h): the GDN mixer core's widths; a verify chain is the committed token and up to 3 drafts.
GDN_CONV_DIM = 10_240
GDN_VALUE_DIM = 6_144
GDN_HEADS = 48
GDN_HEAD_DIM = 128
GDN_PREFILL = 27
GDN_CHAIN_ROWS = (1, 2, 3, 4)
OP_PREFILL = 1  # CERE_OP_PREFILL (cere_executor.h)
OP_DECODE = 2  # CERE_OP_DECODE
# K1 programmatic dependent launches (m23 design note P5R, SmallFuse): m rows of the decode/verify widths; the late
# writer spins this long on the global timer before it writes (ample for the host to enqueue the K1 call behind it).
K1_PDL_M = (1, 4, 16)
PDL_LATE_NS = 2_000_000
PDL_EMBED_VOCAB = 1024
PDL_PREFIX = 37


@functools.cache
def _base_library() -> tuple[ctypes.CDLL, str]:
    """The merge-base libcere_ktest.so (loaded once) and its SHA-256."""
    if not BASE_KTEST_LIB.is_file():
        raise oracle.OracleError(f"{BASE_KTEST_LIB} is absent (the merge-base build, tasks/m22p-spec.md L0)")
    if BASE_KTEST_LIB.resolve() == oracle.KTEST_LIB.resolve():
        raise oracle.OracleError(f"{BASE_KTEST_LIB} is this build's library")
    with BASE_KTEST_LIB.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    return ctypes.CDLL(str(BASE_KTEST_LIB), mode=ctypes.RTLD_LOCAL), sha


def _base(run):
    """run() through the merge-base library's ktest (oracle.ktest routed to it for the call)."""
    lib, _ = _base_library()
    build = oracle.load_ktest
    oracle.load_ktest = lambda: lib
    try:
        return run()
    finally:
        oracle.load_ktest = build


def _both(run):
    """run() through this build's library, then through the merge-base library's: (current, base)."""
    return run(), _base(run)


def _check_tensors(tc: unittest.TestCase, case: str, current, base) -> None:
    for i, (a, b) in enumerate(zip(current, base, strict=True)):
        oracle.check_bit_exact(tc, ROW, f"{case}/out{i}", a, b)


def _check_one_launch(tc: unittest.TestCase, case: str, per_row: int, base_per_row: int) -> None:
    """A rows plan's per_row(m) is 0 (P11R-21: one launch for the m rows, bit-equal to the merge-base library's
    per-row calls by _check_tensors); the merge-base library's own per_row(m) is reported beside it."""
    oracle.report_row(ROW, f"{case}/merge_base", phase="info", per_row=base_per_row)
    oracle.check_count(tc, ROW, case, per_row, expected=0)


def _weight(name: str, dims: int = 2) -> torch.Tensor:
    w = oracle.real_tensor(name)
    if w.dtype != torch.bfloat16 or w.dim() != dims:
        raise oracle.OracleError(f"{name} is {w.dtype} {tuple(w.shape)}; expected {dims}-D BF16")
    return w.contiguous()


def _activations(m: int, k: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    return torch.randn((m, k), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)


def _unit_rms(m: int, k: int, seed: int) -> torch.Tensor:
    """Seeded rows at unit RMS (a block input is the output of a norm)."""
    g = oracle.generator(seed)
    x = torch.randn((m, k), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))).to(torch.bfloat16).contiguous()


# -- K2 ---------------------------------------------------------------------------------------------------------


def _k2_shapes() -> list:
    """(label, loader, out_f32) of every dense GEMM shape the model plans (forward.cpp / k7_moe.cu / k3_hc.cu)."""
    attn = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
    gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
    mlp = oracle.layer_prefix(SHARED_LAYER) + "mlp."
    hc = oracle.layer_prefix(HC_LAYER) + "attn_hyper_connection."
    return [
        ("attn_qkv", lambda: torch.cat([_weight(attn + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                                       dim=0).contiguous(), False),
        ("attn_o", lambda: _weight(attn + "o_proj.weight"), False),
        ("attn_index_qk", lambda: _weight(attn + "indexer.index_qk_proj.weight"), False),
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
    ]


def _gemm_bf16(x: torch.Tensor, w: torch.Tensor) -> list[torch.Tensor]:
    m, k = x.shape
    n = w.shape[0]
    y = oracle.empty((m, n), torch.bfloat16)
    oracle.call("cere_ktest_gemm_bf16", x, w, m, n, k, y)
    return [y]


def _gemm_rows(x: torch.Tensor, w: torch.Tensor, rows: int, out_f32: bool) -> tuple[list[torch.Tensor], int]:
    """`cere_ktest_gemm_rows` on all of x (m = x rows; rows 0: the instantiated plan at max_m = m): ([y], per_row)."""
    m, k = x.shape
    n = w.shape[0]
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    per_row = torch.full((1,), -1, dtype=torch.int64)
    oracle.call("cere_ktest_gemm_rows", x, w, m, n, k, rows, int(out_f32), ROWS_ALL_FUSED, y, per_row)
    return [y], int(per_row[0])


def _lm_head_last(hidden: torch.Tensor, w: torch.Tensor) -> list[torch.Tensor]:
    m, k = hidden.shape
    vocab = w.shape[0]
    logits = oracle.empty((vocab,), torch.bfloat16)
    oracle.call("cere_ktest_lm_head_last", hidden, w, m, vocab, k, logits)
    return [logits]


def _gemm_small_m(x: torch.Tensor, w: torch.Tensor, out_f32: bool) -> list[torch.Tensor]:
    """`cere_ktest_gemm_path` with the small-M kernel on (a plan of max_m = m): [y]."""
    m, k = x.shape
    n = w.shape[0]
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    us = torch.zeros(1, dtype=torch.float64)
    oracle.call("cere_ktest_gemm_path", x, w, m, n, k, int(out_f32), 1, 1, 1, y, us)
    return [y]


def _gemm_segments(x: torch.Tensor, weights: list) -> list[torch.Tensor]:
    """`cere_ktest_gemm_segments` (small-M, one launch) over BF16 `weights` sharing x: the outputs in order."""
    m, k = x.shape
    ys = [oracle.empty((m, w.shape[0]), torch.bfloat16) for w in weights]
    w_ptrs = torch.tensor([w.data_ptr() for w in weights], dtype=torch.int64)
    ns = torch.tensor([w.shape[0] for w in weights], dtype=torch.int64)
    y_ptrs = torch.tensor([y.data_ptr() for y in ys], dtype=torch.int64)
    oracle.call("cere_ktest_gemm_segments", x, m, k, len(weights), w_ptrs, ns, y_ptrs, 0)
    return ys


# -- K3 ---------------------------------------------------------------------------------------------------------


def _hc_sites() -> list:
    """(label, prefix, has_combine): layer HC_LAYER's attn and mlp hyper-connections and the final mixer."""
    layer = oracle.layer_prefix(HC_LAYER)
    return [("attn", layer + "attn_hyper_connection.", True), ("mlp", layer + "mlp_hyper_connection.", True),
            ("final_mixer", "model.language_model.hyper_connection_mixer.", False)]


def _hyper(m: int, hc: int, hidden: int, seed: int) -> torch.Tensor:
    """Seeded BF16 hyper states [m, hc * hidden], one scale per stream (as test_kernels_k3_hc.py)."""
    g = oracle.generator(seed)
    x = torch.randn((m, hc, hidden), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    scales = torch.tensor(HC_STREAM_SCALES, device=oracle.DEVICE, dtype=torch.float32).view(1, hc, 1)
    return (x * scales).reshape(m, hc * hidden).to(torch.bfloat16).contiguous()


def _hc_mix(hyper: torch.Tensor, w: dict, hc: int, hidden: int, lowrank: int, eps: float,
            rows: int) -> list[torch.Tensor]:
    """hc_mix of hyper: `cere_ktest_hc_mix` (rows 0) or `cere_ktest_hc_mix_rows` (rows > 0): [normed, mixed]."""
    m = hyper.shape[0]
    normed = oracle.empty((m, hc * hidden), torch.bfloat16)
    mixed = oracle.empty((m, hidden), torch.bfloat16)
    if rows == 0:
        oracle.call("cere_ktest_hc_mix", hyper, w["norm"], w["down"], w["up"], m, hc, hidden, lowrank, float(eps),
                    normed, mixed)
    else:
        oracle.call("cere_ktest_hc_mix_rows", hyper, w["norm"], w["down"], w["up"], m, hc, hidden, lowrank,
                    float(eps), normed, mixed, rows, ROWS_ALL_FUSED)
    return [normed, mixed]


def _hc_combine(block_out: torch.Tensor, hyper: torch.Tensor, normed: torch.Tensor, w_inject: torch.Tensor, hc: int,
                hidden: int) -> list[torch.Tensor]:
    m = hyper.shape[0]
    out = oracle.empty((m, hc * hidden), torch.bfloat16)
    oracle.call("cere_ktest_hc_combine", block_out, hyper, normed, w_inject, m, hc, hidden, out)
    return [out]


def _hc_combine_norm(block_out: torch.Tensor, hyper: torch.Tensor, normed: torch.Tensor, w_inject: torch.Tensor,
                     next_norm: torch.Tensor, hc: int, hidden: int, eps: float, in_place: bool) -> list[torch.Tensor]:
    """`cere_ktest_hc_combine_norm` on a copy of normed (and of hyper as out when in_place): [out, normed]."""
    m = hyper.shape[0]
    normed = normed.clone()
    out = hyper.clone() if in_place else oracle.empty((m, hc * hidden), torch.bfloat16)
    oracle.call("cere_ktest_hc_combine_norm", block_out, out if in_place else hyper, normed, w_inject, next_norm, m,
                hc, hidden, float(eps), out)
    return [out, normed]


# -- prefill attention ------------------------------------------------------------------------------------------


def _rope_table(n_pos: int) -> torch.Tensor:
    """f32 [n_pos, ROTARY_DIM]: row p = [cos(p * inv_freq) | sin(p * inv_freq)] (SGLang's cos_sin_cache layout)."""
    inv_freq = ROPE_THETA ** (-torch.arange(0, ROTARY_DIM, 2, dtype=torch.float64) / ROTARY_DIM)
    angles = torch.arange(n_pos, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=1).to(torch.float32).to(oracle.DEVICE).contiguous()


def _fp8_rows(rows: int, width: int, seed: int) -> torch.Tensor:
    """Seeded FP8 E4M3 rows [rows, width] as uint8 (K/V cache content of a stored prefix)."""
    g = oracle.generator(seed)
    x = torch.randn((rows, width), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    return x.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def _fp8_edge_rows(rows: int, width: int, seed: int, nan_row: int) -> torch.Tensor:
    """_fp8_rows with 1 in 32 bytes replaced by one of K6_EDGE_SPECIALS (signed zeros, denormals, +-448) and one NaN
    byte in the first KV head's half of row nan_row (the outputs it reaches are NaN)."""
    x = _fp8_rows(rows, width, seed)
    g = oracle.generator(seed + 1)
    pick = torch.randint(0, 32 * len(K6_EDGE_SPECIALS), (rows, width), generator=g, device=oracle.DEVICE)
    specials = torch.tensor(K6_EDGE_SPECIALS, dtype=torch.uint8, device=oracle.DEVICE)
    x = torch.where(pick < len(K6_EDGE_SPECIALS), specials[pick.clamp(max=len(K6_EDGE_SPECIALS) - 1)], x)
    x[nan_row, HEAD_DIM // 3] = K6_NAN
    return x.contiguous()


def _attn_block(w: dict, hidden: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                prefix: int, t: int, kv_from_cache: int, k_cache0: torch.Tensor, v_cache0: torch.Tensor,
                eps: float) -> list[torch.Tensor]:
    """`cere_ktest_attn_block` of rows prefix .. prefix + t on fresh copies of the caches: [attn_out, out, k_cache,
    v_cache]."""
    k_cache, v_cache = k_cache0.clone(), v_cache0.clone()
    attn_out = oracle.empty((t, N_Q_HEADS * HEAD_DIM), torch.bfloat16)
    out = oracle.empty((t, hidden.shape[1]), torch.bfloat16)
    oracle.call("cere_ktest_attn_block", hidden[prefix:prefix + t].contiguous(), w["w_q"], w["w_k"], w["w_v"],
                w["w_o"], w["q_norm"], w["k_norm"], cos_sin, cos_sin.shape[0],
                positions[prefix:prefix + t].contiguous(), slots[:prefix + t].contiguous(), prefix, t, k_cache,
                v_cache, k_cache.shape[0], kv_from_cache, hidden.shape[1], N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
                ROTARY_DIM, float(eps), 1.0, SOFTMAX_SCALE, attn_out, out)
    return [attn_out, out, k_cache, v_cache]


def _attn_dense(q: torch.Tensor, gate: torch.Tensor, pos0: int, k_cache: torch.Tensor, v_cache: torch.Tensor,
                slots: torch.Tensor, mode: int) -> torch.Tensor:
    """`cere_ktest_attn_dense` (this build) of q's rows at pos0 in `mode` (0 eager, 1 at, 2 rows, 3 rows_at)."""
    tokens = q.shape[0]
    pos = torch.full((1,), pos0, dtype=torch.int64, device=oracle.DEVICE)
    out = oracle.empty((tokens, N_Q_HEADS * HEAD_DIM), torch.bfloat16)
    oracle.call("cere_ktest_attn_dense", q, gate, tokens, pos0, pos, k_cache, v_cache, slots, mode, out)
    return out


def _qsa_indices(rows: int, length: int, seed: int) -> torch.Tensor:
    """int32 [rows, 2051]: row t (logical position length - rows + t) selects min(2051, p + 1) distinct positions
    <= p in ascending order, then -1 padding (qsa_select's output form)."""
    g = torch.Generator()
    g.manual_seed(seed)
    out = torch.full((rows, K9_ROW_WIDTH), -1, dtype=torch.int32)
    for t in range(rows):
        p = length - rows + t
        count = min(K9_ROW_WIDTH, p + 1)
        out[t, :count] = torch.sort(torch.randperm(p + 1, generator=g)[:count]).values.to(torch.int32)
    return out.to(oracle.DEVICE).contiguous()


def _qsa_attention(q: torch.Tensor, indices: torch.Tensor, kv: tuple) -> list[torch.Tensor]:
    """`cere_ktest_qsa_attention` of q's rows; kv = (k, v, k_cache, v_cache, slots), one pair null: [out]."""
    rows = q.shape[0]
    out = oracle.empty((rows, N_Q_HEADS * HEAD_DIM), torch.bfloat16)
    oracle.call("cere_ktest_qsa_attention", q, rows, indices, *kv, SOFTMAX_SCALE, out)
    return [out]


def _top2_logits(rows: int, vocab: int, seed: int) -> torch.Tensor:
    """BF16 [rows, vocab]: row r is pattern r % 6 - 0 ties (values on a 0.5 grid), 1 ties plus two NaNs, 2 one NaN,
    3 all -inf, 4 -inf but three entries (two tied), 5 signed zeros over negatives."""
    g = oracle.generator(seed)
    x = torch.round(torch.randn((rows, vocab), generator=g, device=oracle.DEVICE, dtype=torch.float32) * 2) / 2
    idx = torch.randint(0, vocab, (rows, 3), generator=g, device=oracle.DEVICE)
    for r in range(rows):
        kind = r % 6
        a, b, c = (int(i) for i in idx[r])
        if kind in (1, 2):
            x[r, a] = float("nan")
            if kind == 1:
                x[r, b] = float("nan")
        elif kind == 3:
            x[r] = float("-inf")
        elif kind == 4:
            x[r] = float("-inf")
            x[r, a], x[r, b], x[r, c] = 1.5, 1.5, -2.0
        elif kind == 5:
            sign = torch.randint(0, 2, (vocab,), generator=g, device=oracle.DEVICE) * 2.0 - 1.0
            zeros = torch.rand((vocab,), generator=g, device=oracle.DEVICE) < 0.3
            x[r] = torch.where(zeros, 0.0 * sign, -x[r].abs() - 1)
    return x.to(torch.bfloat16).contiguous()


def _top2(logits: torch.Tensor) -> list[torch.Tensor]:
    """`cere_ktest_top2` of logits' rows: [ids int32 [rows, 2], scores f32 [rows, 2]]."""
    rows, vocab = logits.shape
    ids = oracle.empty((rows, 2), torch.int32)
    scores = oracle.empty((rows, 2), torch.float32)
    oracle.call("cere_ktest_top2", logits, rows, vocab, ids, scores)
    return [ids, scores]


def _gdn_weights() -> dict:
    """Layer GDN_LAYER's mixer-core weights as the K5 entry points take them: conv bf16 [10,240, 4], A_log f32 [48],
    dt_bias bf16 [48], norm bf16 [128], and the in-projections (qkv, z, b, a) that make the core's inputs."""
    raw = oracle.gdn_layer_weights(GDN_LAYER)
    return {"conv": raw["conv1d.weight"].reshape(GDN_CONV_DIM, 4).contiguous(),
            "A_log": raw["A_log"].float().contiguous(), "dt_bias": raw["dt_bias"].contiguous(),
            "norm": raw["norm.weight"].contiguous(),
            **{p: raw[f"in_proj_{p}.weight"].contiguous() for p in ("qkv", "z", "b", "a")}}


def _gdn_core(wt: dict, x: list, conv: torch.Tensor, ssm: torch.Tensor, operation: int) -> torch.Tensor:
    """`cere_ktest_gdn_layer_core` on x = [mixed_qkv, z, b, a] (tokens rows): y; conv and ssm updated in place."""
    tokens = x[0].shape[0]
    y = oracle.empty((tokens, GDN_VALUE_DIM), torch.bfloat16)
    oracle.call("cere_ktest_gdn_layer_core", *x, wt["conv"], wt["A_log"], wt["dt_bias"], wt["norm"], conv, ssm, y,
                tokens, operation)
    return y


def _gdn_per_row(wt: dict, x: list, conv0: torch.Tensor, ssm0: torch.Tensor) -> list[torch.Tensor]:
    """The chain x one DECODE layer_core per row from copies of (conv0, ssm0): [y, conv, ssm, then conv and ssm after
    every row but the last]."""
    conv, ssm = conv0.clone(), ssm0.clone()
    ys, saves = [], []
    rows = x[0].shape[0]
    for t in range(rows):
        ys.append(_gdn_core(wt, [v[t:t + 1] for v in x], conv, ssm, OP_DECODE))
        if t + 1 < rows:
            saves += [conv.clone(), ssm.clone()]
    return [torch.cat(ys), conv, ssm, *saves]


def _gdn_chain(wt: dict, x: list, conv0: torch.Tensor, ssm0: torch.Tensor) -> list[torch.Tensor]:
    """`cere_ktest_gdn_chain_core` (this build) on the chain x from copies of (conv0, ssm0), as _gdn_per_row lists."""
    conv, ssm = conv0.clone(), ssm0.clone()
    rows = x[0].shape[0]
    y = oracle.empty((rows, GDN_VALUE_DIM), torch.bfloat16)
    conv_saves = oracle.empty((rows - 1, GDN_CONV_DIM, 3), torch.bfloat16)
    ssm_saves = oracle.empty((rows - 1, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM), torch.bfloat16)
    oracle.call("cere_ktest_gdn_chain_core", *x, wt["conv"], wt["A_log"], wt["dt_bias"], wt["norm"], conv, ssm,
                conv_saves if rows > 1 else None, ssm_saves if rows > 1 else None, y, rows)
    saves = []
    for t in range(rows - 1):
        saves += [conv_saves[t], ssm_saves[t]]
    return [y, conv, ssm, *saves]



def _late(src: torch.Tensor, stale: torch.Tensor) -> torch.Tensor:
    """A copy of `stale` that `cere_ktest_pdl_late_copy` (this build) overwrites with src's bytes PDL_LATE_NS after
    it starts. The copy lets the stream's next kernel launch at once and is not waited on here, so the next ktest
    call's K1 kernel is resident while it spins: only its griddepcontrol.wait keeps it from reading the stale bytes."""
    if src.shape != stale.shape or src.dtype != stale.dtype or not src.is_contiguous():
        raise oracle.OracleError("the late source and its stale tensor differ")
    dst = stale.clone()
    stream = torch.cuda.current_stream(oracle.DEVICE)
    rc = oracle.ktest("cere_ktest_pdl_late_copy")(
        ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()),
        ctypes.c_int64(src.numel() * src.element_size()), ctypes.c_int64(PDL_LATE_NS),
        ctypes.c_void_p(stream.cuda_stream))
    if rc != 0:
        raise oracle.OracleError(f"cere_ktest_pdl_late_copy returned status {rc}")
    return dst


def _k1_pdl_cases(m: int, hidden: int, eps: float, seed: int, norms: dict, cos_sin: torch.Tensor) -> list:
    """(label, inputs, late input, its stale bytes, outputs factory, call(inputs, outputs)) of every K1 kernel the
    decode path launches as a programmatic dependent launch, at m rows (the prologue: qk_norm_rope_gate, then the KV
    store through the FP8 cache after PDL_PREFIX tokens)."""
    q_width, kv_width = N_Q_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM
    n_slots = K6_CONTEXT + 1  # row 0 is the pad row
    g = oracle.generator(seed)
    ids = torch.randint(0, PDL_EMBED_VOCAB, (m,), generator=g, device=oracle.DEVICE, dtype=torch.int32)
    slots = (torch.randperm(K6_CONTEXT, generator=g, device=oracle.DEVICE)[:PDL_PREFIX + m] + 1)
    slots = slots.to(torch.int64).contiguous()

    def act(rows: int, width: int, s: int) -> torch.Tensor:
        return _activations(rows, width, seed + s)

    def zeros(*shapes, dtype=torch.bfloat16):
        return lambda: [torch.zeros(s, dtype=dtype, device=oracle.DEVICE) for s in shapes]

    def caches_and(*shapes):
        return lambda: zeros((n_slots, kv_width), (n_slots, kv_width), dtype=torch.uint8)() + zeros(*shapes)()

    def copy(i: dict, o: list) -> None:
        pitch = hidden * 2
        ranges = torch.tensor([[o[0].data_ptr(), pitch, i["src"].data_ptr(), pitch, pitch, m]], dtype=torch.int64)
        oracle.call("cere_ktest_copy_segments", ranges, 1)

    x = act(m, hidden, 2)
    xq = act(m, q_width, 4)
    xg = act(m * GDN_HEADS, GDN_HEAD_DIM, 6)
    k, kv_v = act(m, kv_width, 10), act(m, kv_width, 11)
    q_gate = act(m, 2 * q_width, 12)
    return [
        ("embedding", {"table": act(PDL_EMBED_VOCAB, hidden, 1), "ids": ids}, "ids", (ids + 1) % PDL_EMBED_VOCAB,
         zeros((m, hidden)),
         lambda i, o: oracle.call("cere_ktest_embedding", i["table"], i["ids"], m, PDL_EMBED_VOCAB, hidden, o[0])),
        ("rmsnorm_gemma", {"x": x, "w": act(1, hidden, 3).reshape(-1)}, "x", torch.zeros_like(x), zeros((m, hidden)),
         lambda i, o: oracle.call("cere_ktest_rmsnorm", i["x"], i["w"], m, hidden, hidden, eps, 1.0, o[0])),
        ("rmsnorm_heads", {"x": xq, "w": act(1, q_width, 5).reshape(-1)}, "x", torch.zeros_like(xq),
         zeros((m, q_width)),
         lambda i, o: oracle.call("cere_ktest_rmsnorm", i["x"], i["w"], m, q_width, HEAD_DIM, eps, 0.0, o[0])),
        ("rmsnorm_gated",
         {"x": xg, "z": act(m * GDN_HEADS, GDN_HEAD_DIM, 7), "w": act(1, GDN_HEAD_DIM, 8).reshape(-1)},
         "x", torch.zeros_like(xg), zeros((m * GDN_HEADS, GDN_HEAD_DIM)),
         lambda i, o: oracle.call("cere_ktest_rmsnorm_gated", i["x"], i["z"], i["w"], m * GDN_HEADS, GDN_HEAD_DIM,
                                  eps, o[0])),
        ("sigmoid_mul", {"x": xq, "gate": act(m, q_width, 9)}, "x", torch.zeros_like(xq), zeros((m, q_width)),
         lambda i, o: oracle.call("cere_ktest_sigmoid_gate", i["x"], i["gate"], m, q_width, o[0])),
        ("kv_store", {"k": k, "v": kv_v}, "k", torch.zeros_like(k), caches_and(),
         lambda i, o: oracle.call("cere_ktest_kv_store_fp8", i["k"], i["v"], slots[:m].contiguous(), m, kv_width,
                                  o[0], o[1], n_slots)),
        ("qk_norm_rope_gate", {"q_gate": q_gate, "k": k, "v": kv_v}, "q_gate", torch.zeros_like(q_gate),
         caches_and((m, q_width), (m, kv_width), (m, q_width)),
         lambda i, o: oracle.call("cere_ktest_attn_prologue", i["q_gate"], i["k"], i["v"], norms["q_norm"],
                                  norms["k_norm"], cos_sin, K6_CONTEXT, None, slots, PDL_PREFIX, m, o[0], o[1],
                                  n_slots, eps, o[2], o[3], o[4])),
        ("copy_segments", {"src": x}, "src", torch.zeros_like(x), zeros((m, hidden)), copy),
    ]


class LeversOffIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, sha = _base_library()
        oracle.report_row(ROW, "merge_base_library", phase="info", library_sha256=sha)
        tc = oracle.text_config()
        cls.hidden = int(tc["hidden_size"])
        cls.eps = float(tc["rms_norm_eps"])

    def test_k2_default_equals_main(self):
        """k2_gemm/cublaslt: every planned GEMM shape at every m bucket, the rows plans and the last-row lm_head of
        this build equal the merge-base library's byte for byte (GemmPlan::init leaves small-M off)."""
        for s, (label, load, out_f32) in enumerate(_k2_shapes()):
            w = load()
            k = w.shape[1]
            for m in K2_M:
                with self.subTest(shape=label, m=m):
                    x = _activations(m, k, SEED + 100 * s + m)
                    if not out_f32:
                        current, base = _both(lambda: _gemm_bf16(x, w))
                        _check_tensors(self, f"k2/{label}/M{m}/gemm_bf16", current, base)
                    for rows in (0,) + ROWS_PLANS:
                        if rows != 0 and m > rows:
                            continue
                        (current, per_row), (base, base_per_row) = _both(lambda: _gemm_rows(x, w, rows, out_f32))
                        _check_tensors(self, f"k2/{label}/M{m}/rows{rows}", current, base)
                        _check_one_launch(self, f"k2/{label}/M{m}/rows{rows}/per_row", per_row, base_per_row)
            del w
        w = _weight("lm_head.weight")
        for m in LM_HEAD_M:
            with self.subTest(shape="lm_head", m=m):
                hidden = _activations(m, w.shape[1], SEED + 50_000 + m)
                current, base = _both(lambda: _lm_head_last(hidden, w))
                _check_tensors(self, f"k2/lm_head/M{m}/last_row", current, base)
                for rows in ROWS_PLANS:
                    if m > rows:
                        continue
                    (current, per_row), (base, base_per_row) = _both(lambda: _gemm_rows(hidden, w, rows, False))
                    _check_tensors(self, f"k2/lm_head/M{m}/rows{rows}", current, base)
                    _check_one_launch(self, f"k2/lm_head/M{m}/rows{rows}/per_row", per_row, base_per_row)
        del w
        torch.cuda.empty_cache()

    def test_k2_small_m_equals_main(self):
        """k2_gemm/small_m (P11R-21): the small-M kernel of this build, alone and fused over segments, equals the
        merge-base library's byte for byte at every m it serves."""
        for s, (label, load, out_f32) in enumerate([*_k2_shapes(), ("lm_head", lambda: _weight("lm_head.weight"),
                                                                     False)]):
            w = load()
            k = w.shape[1]
            for m in SMALL_M:
                with self.subTest(shape=label, m=m):
                    x = _activations(m, k, SEED + 80_000 + 100 * s + m)
                    current, base = _both(lambda: _gemm_small_m(x, w, out_f32))
                    _check_tensors(self, f"k2_small_m/{label}/M{m}", current, base)
            del w
        gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
        mlp = oracle.layer_prefix(SHARED_LAYER) + "mlp.shared_expert."
        fused = (("gdn_in", [_weight(gdn + f"in_proj_{p}.weight") for p in ("qkv", "z", "b", "a")]),
                 ("shared_gate_up", [_weight(mlp + f"{p}_proj.weight") for p in ("gate", "up")]))
        for s, (label, weights) in enumerate(fused):
            k = weights[0].shape[1]
            for m in SMALL_M:
                with self.subTest(fused=label, m=m):
                    x = _activations(m, k, SEED + 90_000 + 100 * s + m)
                    current, base = _both(lambda: _gemm_segments(x, weights))
                    _check_tensors(self, f"k2_small_m/{label}_fused/M{m}", current, base)
        del fused
        torch.cuda.empty_cache()

    def test_k3_default_equals_main(self):
        """K3 (no lever): hc_mix through HcPlans (instantiated and rows plans) and hc_combine of this build equal the
        merge-base library's byte for byte."""
        tc = oracle.text_config()
        hc, lowrank = int(tc["hc_count"]), int(tc["hc_lowrank"])
        for s, (site, prefix, has_combine) in enumerate(_hc_sites()):
            w = {"norm": _weight(prefix + "hc_norm.weight", dims=1),
                 "down": _weight(prefix + "input_mix_weight_down.weight"),
                 "up": _weight(prefix + "input_mix_weight_up.weight")}
            if has_combine:
                w["inject"] = _weight(prefix + "block_inject_weight.weight")
            for m in HC_M:
                with self.subTest(site=site, m=m):
                    hyper = _hyper(m, hc, self.hidden, SEED + 60_000 + 1000 * s + m)
                    for rows in (0,) + ROWS_PLANS:
                        if rows != 0 and m > rows:
                            continue
                        current, base = _both(lambda: _hc_mix(hyper, w, hc, self.hidden, lowrank, self.eps, rows))
                        _check_tensors(self, f"k3/{site}/M{m}/mix_rows{rows}", current, base)
                    if has_combine:
                        normed = current[0]
                        block_out = _activations(m, self.hidden, SEED + 70_000 + 1000 * s + m)
                        current, base = _both(lambda: _hc_combine(block_out, hyper, normed, w["inject"], hc,
                                                                  self.hidden))
                        _check_tensors(self, f"k3/{site}/M{m}/combine", current, base)

    def test_hc_combine_norm_equals_main(self):
        """K3 hc_combine_norm (no lever): this build's combine fused with the next mix's norm equals the merge-base
        library's hc_combine, then hc_mix's norm of its output, byte for byte (out and normed)."""
        tc = oracle.text_config()
        hc, lowrank = int(tc["hc_count"]), int(tc["hc_lowrank"])
        sites = _hc_sites()
        mix_w = [{"norm": _weight(prefix + "hc_norm.weight", dims=1),
                  "down": _weight(prefix + "input_mix_weight_down.weight"),
                  "up": _weight(prefix + "input_mix_weight_up.weight")} for _, prefix, _ in sites]
        for s in (0, 1):  # attn -> mlp mix, mlp -> the final mixer
            site, prefix, _ = sites[s]
            inject = _weight(prefix + "block_inject_weight.weight")
            nxt = mix_w[s + 1]
            for m in HC_M:
                with self.subTest(site=site, m=m):
                    hyper = _hyper(m, hc, self.hidden, SEED + 80_000 + 1000 * s + m)
                    normed = _hc_mix(hyper, mix_w[s], hc, self.hidden, lowrank, self.eps, 0)[0]
                    block_out = _activations(m, self.hidden, SEED + 90_000 + 1000 * s + m)
                    out = _base(lambda: _hc_combine(block_out, hyper, normed, inject, hc, self.hidden))[0]
                    base = [out, _base(lambda: _hc_mix(out, nxt, hc, self.hidden, lowrank, self.eps, 0))[0]]
                    for label, in_place in (("out", False), ("in_place", True)):
                        current = _hc_combine_norm(block_out, hyper, normed, inject, nxt["norm"], hc, self.hidden,
                                                   self.eps, in_place)
                        _check_tensors(self, f"k3/{site}/M{m}/combine_norm_{label}", current, base)

    def test_prefill_default_equals_main(self):
        """prefill_attention/cuda_core: K6's full-attention block (with and without a cached prefix) and K9's sparse
        attention (BF16 K/V and the FP8 cache) of this build equal the merge-base library's byte for byte."""
        p = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
        w = {name: _weight(p + proj + ".weight", dims=1 if proj.endswith("norm") else 2) for name, proj in
             (("w_q", "q_proj"), ("w_k", "k_proj"), ("w_v", "v_proj"), ("w_o", "o_proj"), ("q_norm", "q_norm"),
              ("k_norm", "k_norm"))}
        cos_sin = _rope_table(K6_CONTEXT)
        kv_width = N_KV_HEADS * HEAD_DIM
        n_slots = K6_CONTEXT + 1  # row 0 is the pad row
        cases = [(0, t) for t in K6_NO_PREFIX] + list(K6_PREFIX)
        for i, (prefix, t) in enumerate(cases):
            with self.subTest(attention="k6", P=prefix, T=t):
                seed = SEED + 80_000 + 10 * i
                hidden = _unit_rms(prefix + t, self.hidden, seed)
                positions = torch.arange(prefix + t, device=oracle.DEVICE, dtype=torch.int64)
                g = oracle.generator(seed + 1)
                slots = (torch.randperm(K6_CONTEXT, generator=g, device=oracle.DEVICE)[:prefix + t] + 1)
                slots = slots.to(torch.int64).contiguous()
                k_cache = torch.zeros((n_slots, kv_width), dtype=torch.uint8, device=oracle.DEVICE)
                v_cache = torch.zeros_like(k_cache)
                if prefix:
                    k_cache[slots[:prefix]] = _fp8_rows(prefix, kv_width, seed + 2)
                    v_cache[slots[:prefix]] = _fp8_rows(prefix, kv_width, seed + 3)
                current, base = _both(lambda: _attn_block(w, hidden, cos_sin, positions, slots, prefix, t,
                                                          1 if prefix else 0, k_cache, v_cache, self.eps))
                _check_tensors(self, f"prefill/k6/P{prefix}_T{t}", current, base)
        del w
        k_all = _activations(K9_LENGTH, kv_width, SEED + 90_000)
        v_all = _activations(K9_LENGTH, kv_width, SEED + 90_001)
        g = oracle.generator(SEED + 90_002)
        slots = torch.randperm(K9_LENGTH, generator=g, device=oracle.DEVICE).to(torch.int64).contiguous()
        k_cache = torch.empty((K9_LENGTH, kv_width), dtype=torch.uint8, device=oracle.DEVICE)
        v_cache = torch.empty_like(k_cache)
        k_cache[slots] = k_all.to(torch.float8_e4m3fn).view(torch.uint8)
        v_cache[slots] = v_all.to(torch.float8_e4m3fn).view(torch.uint8)
        for rows in K9_ROWS:
            q = _activations(rows, N_Q_HEADS * HEAD_DIM, SEED + 91_000 + rows)
            indices = _qsa_indices(rows, K9_LENGTH, SEED + 92_000 + rows)
            for label, kv in (("bf16_kv", (k_all, v_all, None, None, None)),
                              ("fp8_cache", (None, None, k_cache, v_cache, slots))):
                with self.subTest(attention="k9", rows=rows, kv=label):
                    current, base = _both(lambda: _qsa_attention(q, indices, kv))
                    _check_tensors(self, f"prefill/k9/n{K9_LENGTH}/R{rows}/{label}", current, base)
        torch.cuda.empty_cache()

    def test_decode_attention_equals_main(self):
        """K6 decode/verify attention (no lever): the eager block of this build equals the merge-base library's at
        every verify width and split count, and the step / chain entry points equal the eager calls they replace."""
        p = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
        w = {name: _weight(p + proj + ".weight", dims=1 if proj.endswith("norm") else 2) for name, proj in
             (("w_q", "q_proj"), ("w_k", "k_proj"), ("w_v", "v_proj"), ("w_o", "o_proj"), ("q_norm", "q_norm"),
              ("k_norm", "k_norm"))}
        cos_sin = _rope_table(K6_CONTEXT)
        kv_width = N_KV_HEADS * HEAD_DIM
        q_width = N_Q_HEADS * HEAD_DIM
        n_slots = K6_CONTEXT + 1  # row 0 is the pad row
        for i, prefix in enumerate(K6_DECODE_PREFIX):
            seed = SEED + 95_000 + 100 * i
            hidden = _unit_rms(prefix + max(K6_DECODE_M), self.hidden, seed)
            positions = torch.arange(hidden.shape[0], device=oracle.DEVICE, dtype=torch.int64)
            g = oracle.generator(seed + 1)
            slots = (torch.randperm(K6_CONTEXT, generator=g, device=oracle.DEVICE)[:hidden.shape[0]] + 1)
            slots = slots.to(torch.int64).contiguous()
            k_cache = torch.zeros((n_slots, kv_width), dtype=torch.uint8, device=oracle.DEVICE)
            v_cache = torch.zeros_like(k_cache)
            k_cache[slots[:prefix]] = _fp8_rows(prefix, kv_width, seed + 2)
            v_cache[slots[:prefix]] = _fp8_rows(prefix, kv_width, seed + 3)
            for m in K6_DECODE_M:
                with self.subTest(P=prefix, M=m):
                    current, base = _both(lambda: _attn_block(w, hidden, cos_sin, positions, slots, prefix, m, 1,
                                                              k_cache, v_cache, self.eps))
                    _check_tensors(self, f"decode/k6/P{prefix}_M{m}/block", current, base)
                    kc, vc = current[2], current[3]
                    sl = slots[:prefix + m].contiguous()
                    q = _activations(m, q_width, seed + 10 + m)
                    gate = _activations(m, q_width, seed + 50 + m)
                    eager = _attn_dense(q, gate, prefix, kc, vc, sl, 0)
                    per_row = torch.cat([_attn_dense(q[t:t + 1].contiguous(), gate[t:t + 1].contiguous(),
                                                     prefix + t, kc, vc, sl, 0) for t in range(m)])
                    for mode, label, expected in ((1, "at", eager), (2, "rows", per_row), (3, "rows_at", per_row)):
                        oracle.check_bit_exact(self, ROW, f"decode/k6/P{prefix}_M{m}/{label}",
                                               _attn_dense(q, gate, prefix, kc, vc, sl, mode), expected)
        del w
        torch.cuda.empty_cache()

    def test_decode_attention_edges_equals_main(self):
        """K6 one-token FP8 attention (AttnSplit's query-head groups, no lever): `cere_ktest_attn_dense` eager, at,
        rows and rows_at of this build equal the merge-base library's, byte for byte, at key-split edges and verify
        widths 1..16 on caches with signed zeros, denormals, +-448 and a NaN key and value."""
        q_width = N_Q_HEADS * HEAD_DIM
        kv_width = N_KV_HEADS * HEAD_DIM
        n_slots = K6_CONTEXT + 1  # row 0 is the pad row
        for i, prefix in enumerate(K6_EDGE_PREFIX):
            seed = SEED + 96_000 + 100 * i
            keys = prefix + max(K6_EDGE_M)
            g = oracle.generator(seed)
            slots = (torch.randperm(K6_CONTEXT, generator=g, device=oracle.DEVICE)[:keys] + 1).to(torch.int64)
            k_cache = torch.zeros((n_slots, kv_width), dtype=torch.uint8, device=oracle.DEVICE)
            v_cache = torch.zeros_like(k_cache)
            k_cache[slots] = _fp8_edge_rows(keys, kv_width, seed + 2, nan_row=prefix // 3)
            v_cache[slots] = _fp8_edge_rows(keys, kv_width, seed + 4, nan_row=2 * prefix // 3)
            for m in K6_EDGE_M:
                q = _activations(m, q_width, seed + 10 + m)
                gate = _activations(m, q_width, seed + 50 + m)
                sl = slots[:prefix + m].contiguous()
                for mode, label in ((0, "eager"), (1, "at"), (2, "rows"), (3, "rows_at")):
                    with self.subTest(P=prefix, M=m, mode=label):
                        current, base = _both(lambda: [_attn_dense(q, gate, prefix, k_cache, v_cache, sl, mode)])
                        _check_tensors(self, f"decode/k6_edges/P{prefix}_M{m}/{label}", current, base)
        torch.cuda.empty_cache()

    def test_decode_attention_split_blocks(self):
        """AttnSplit (fail-first: the merge-base library has no cere_ktest_attn_split_blocks; its tile kernel runs one
        block per KV head and key split): a decode step / MTP extend runs K6_STEP_BLOCKS query-head blocks per key
        split and a verify chain of the speculation window K6_CHAIN_BLOCKS per row and split."""
        blocks = torch.full((1,), -1, dtype=torch.int64)
        oracle.call("cere_ktest_attn_split_blocks", 1, 0, blocks)
        oracle.check_count(self, ROW, "decode/k6_blocks/step", int(blocks.item()), expected=K6_STEP_BLOCKS)
        for rows in K6_WINDOW_ROWS:
            with self.subTest(rows=rows):
                blocks.fill_(-1)
                oracle.call("cere_ktest_attn_split_blocks", rows, 1, blocks)
                oracle.check_count(self, ROW, f"decode/k6_blocks/chain_r{rows}", int(blocks.item()),
                                   expected=K6_CHAIN_BLOCKS)

    def test_attention_chains_equals_per_chain(self):
        """R16-C (tasks/rows16-spec.md; this build only, the merge-base library has no cere_ktest_attn_chains): the
        verify graph's full-attention core over every sequence's chain at once (k6::prologue_chains_at +
        k6::dense_attention_chains_at, mode 1) equals attention_chain's per-chain calls (prologue_rows_at +
        dense_attention_rows_at, mode 0) byte for byte (attention output and both caches), on K6_CHAINS_LAYOUTS with
        each sequence at its own prefix from K6_EDGE_PREFIX over its own caches holding the edge bytes."""
        q_width = N_Q_HEADS * HEAD_DIM
        kv_width = N_KV_HEADS * HEAD_DIM
        n_slots = K6_CONTEXT + 1  # row 0 is the pad row
        p = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
        norms = {name: _weight(p + name + ".weight", dims=1) for name in ("q_norm", "k_norm")}
        cos_sin = _rope_table(K6_CONTEXT)
        for i, layout in enumerate(K6_CHAINS_LAYOUTS):
            seed = SEED + 98_000 + 100 * i
            chains, rows = len(layout), sum(layout)
            g = oracle.generator(seed)
            prefixes = [min(K6_EDGE_PREFIX[(i + c) % len(K6_EDGE_PREFIX)], K6_CONTEXT - n)
                        for c, n in enumerate(layout)]
            slots = torch.zeros((chains, n_slots), dtype=torch.int64, device=oracle.DEVICE)
            k_cache = torch.zeros((chains, n_slots, kv_width), dtype=torch.uint8, device=oracle.DEVICE)
            v_cache = torch.zeros_like(k_cache)
            for c, prefix in enumerate(prefixes):
                slots[c, :K6_CONTEXT] = torch.randperm(K6_CONTEXT, generator=g, device=oracle.DEVICE) + 1
                sl = slots[c, :prefix]
                k_cache[c, sl] = _fp8_edge_rows(prefix, kv_width, seed + 2 + c, nan_row=prefix // 3)
                v_cache[c, sl] = _fp8_edge_rows(prefix, kv_width, seed + 40 + c, nan_row=2 * prefix // 3)
            positions = torch.tensor([prefix + j for prefix, n in zip(prefixes, layout, strict=True)
                                      for j in range(n)], dtype=torch.int64, device=oracle.DEVICE)
            q_gate = _activations(rows, 2 * q_width, seed + 80)
            k = _activations(rows, kv_width, seed + 81)
            v = _activations(rows, kv_width, seed + 82)
            chain_rows = torch.tensor(layout, dtype=torch.int64)  # host

            def run(mode: int) -> list[torch.Tensor]:
                kc, vc = k_cache.clone(), v_cache.clone()
                out = oracle.empty((rows, q_width), torch.bfloat16)
                oracle.call("cere_ktest_attn_chains", q_gate, k, v, norms["q_norm"], norms["k_norm"], cos_sin,
                            positions, slots, kc, vc, n_slots, chain_rows, chains, self.eps, mode, out)
                return [out, kc, vc]

            with self.subTest(layout=layout, prefixes=prefixes):
                _check_tensors(self, f"verify/k6_chains/{'_'.join(map(str, layout))}", run(1), run(0))
        torch.cuda.empty_cache()

    def test_top2_equals_main(self):
        """K1 top2 (no lever): ids and scores of this build equal the merge-base library's on ties, NaNs, -inf and
        signed zeros at aligned, misaligned and short rows."""
        for i, vocab in enumerate(TOP2_VOCABS):
            for rows in TOP2_ROWS:
                with self.subTest(vocab=vocab, rows=rows):
                    logits = _top2_logits(rows, vocab, SEED + 97_000 + 100 * i + rows)
                    current, base = _both(lambda: _top2(logits))
                    _check_tensors(self, f"top2/V{vocab}/R{rows}", current, base)
        torch.cuda.empty_cache()

    def test_k1_pdl_equals_main(self):
        """K1 programmatic dependent launches (P5R, SmallFuse; no lever): each decode-path K1 kernel of this build,
        reading an input that cere_ktest_pdl_late_copy writes late, equals the merge-base library's plain launch on
        the written input, byte for byte (a kernel that read before its dependency wait would see the stale bytes)."""
        p = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
        norms = {name: _weight(p + name + ".weight", dims=1) for name in ("q_norm", "k_norm")}
        cos_sin = _rope_table(K6_CONTEXT)
        for m in K1_PDL_M:
            for label, ins, key, stale, outs, run in _k1_pdl_cases(m, self.hidden, self.eps, SEED + 99_000 + 100 * m,
                                                                    norms, cos_sin):
                with self.subTest(kernel=label, m=m):
                    base = outs()
                    _base(lambda: run(ins, base))
                    current = outs()
                    late = dict(ins)
                    late[key] = _late(ins[key], stale)
                    run(late, current)  # nothing is enqueued between the late copy and the K1 call
                    _check_tensors(self, f"k1_pdl/{label}/M{m}", current, base)
        torch.cuda.empty_cache()

    def test_gdn_chain_equals_main(self):
        """K5 verify chain (DG-02, no lever): per-row DECODE layer_core of this build equals the merge-base library's,
        and this build's chain_core equals the merge-base library's per-row calls, snapshots included."""
        wt = _gdn_weights()
        tokens = GDN_PREFILL + max(GDN_CHAIN_ROWS)
        hidden = _unit_rms(tokens, self.hidden, SEED + 98_000)
        x = [(hidden.float() @ wt[p].float().t()).to(torch.bfloat16).contiguous() for p in ("qkv", "z", "b", "a")]
        conv = torch.zeros((GDN_CONV_DIM, 3), dtype=torch.bfloat16, device=oracle.DEVICE)
        ssm = torch.zeros((GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM), dtype=torch.bfloat16, device=oracle.DEVICE)
        _base(lambda: _gdn_core(wt, [v[:GDN_PREFILL] for v in x], conv, ssm, OP_PREFILL))
        for rows in GDN_CHAIN_ROWS:
            with self.subTest(rows=rows):
                chain = [v[GDN_PREFILL:GDN_PREFILL + rows] for v in x]
                current, base = _both(lambda: _gdn_per_row(wt, chain, conv, ssm))
                _check_tensors(self, f"gdn/L{GDN_LAYER}/P{GDN_PREFILL}_R{rows}/per_row", current, base)
                _check_tensors(self, f"gdn/L{GDN_LAYER}/P{GDN_PREFILL}_R{rows}/chain",
                               _gdn_chain(wt, chain, conv, ssm), base)
        del wt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
