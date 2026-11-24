"""D12 (tasks/m23-spec.md D12): the native forward profile on the real A4 model, GPU0, container only.

Evidence, never a verdict: it ranks the P rows (P1 launch overhead, P9 per-row attention/GDN launches). Suite
`profile` (`native_dev.py test --suite profile`, GPU0-only: a timing suite); its report is
`bench/reports/native-test-profile-<UTC>.json`.

One model through `oracle.create_model` at the native-g0 plan (`configs/runtime/native-g0.toml`): context 200,000,
4 sessions (decode_batch_max), pool 200,000. Cells: the G6 1.2K prompt (1,200 tokens) at rows 1 and 4, and the 18K
(18,000 tokens), 72K (72,000 tokens) and 200K prompts at rows 1; the 200K prompt is 199,000 tokens, so prompt + 64
steps (199,064) stays within the context and the pool's 3,125 pages of 64 (tasks/m23-spec.md P4 D12 cells). Rows 4
at 72K needs 4 x 72,064 pool tokens, above the native-g0 plan's 200,000-token pool, so it is
not a cell. The prompts are synthetic and deterministic (K11's records template); token IDs stay in memory and no ID
or text is reported.

Per cell `cere_ktest_forward_profile` runs on fresh sessions three times: under torch.profiler with CUDA activity
only (CUPTI kernel, memcpy and memset records of the whole process; nothing is linked into either library), whose
records the ktest marker kernel splits into session 0's prefill, the other rows' prefills and the decode steps; then
unprofiled (the host wall ms of session 0's prefill and of each of the 64 greedy decode steps) with the model's
decode_step lever as created (tasks/m23-spec.md PI-02: eager by default), then unprofiled on the other arm of it (P1:
the decode graph after one unmeasured run that captures, or eager). The profiled run goes first. A launch is one
kernel, memcpy or memset record (a node of a replayed graph is one too). Families by kernel name:
`cere::k1|k4|k5|k6|k9|k12::` are K1/K4/K5/K6/K9/K12, `cere::kernels::` is K2 (the in-tree K2 kernels:
d12_spec.K2_NAMESPACES), K0 (quantize/dequant), K3 (hc_*) or K7 (the rest), `tensorrt_llm`/`cere::p3b::` is P3b, any
other kernel is cuBLASLt's (K2's GemmPlan::run). Rows
(`oracle.report_row`, ROW "D12"):

* `decode-<size>-r<rows>`: per decode step, launches (and kernels / copies), busy ms (the sum of the record
  durations), the unprofiled step ms (mean, median) and rows * 1000 / mean as tok/s, the launch overhead share
  1 - busy / unprofiled mean (the share of the step not covered by GPU work: launch and host time [INFERENCE]), the
  attention/GDN launches (K5 + K6 + K9) per step, their share and per row (P9), and per family launches, us (summed
  durations) and exclusive us (`<family>_excl_us`, d12_spec.exclusive_by: each covered instant goes to the
  earliest-started record still running, so a programmatic dependent launch's wait for its predecessor (P11R-21) is
  the predecessor's);
  P1: `decode_graph` (1 = the graphs were on for the timed run), `graph_steps` (decode steps of the timed run a graph
  ran), `graph_captures` (graphs instantiated or updated during it), the eager and graph arms' step ms (mean, median)
  and tok/s, and `graph_speedup` = eager mean / graph mean.
* `prefill-<size>`: session 0's prefill: launches, busy ms, the unprofiled wall ms, per family launches and ms, and
  the prefill attention split (P4): `attention_ms` = the attention cores (K6's and K9's `attention_kernel` and
  `combine_kernel`, every `cere::p4::` kernel), `qsa_select_ms` = K9's other kernels (index q, compressed keys, ring,
  logits, top-k), and the attention cores' share of the busy ms and of the unprofiled wall ms (cold TTFT's prefill).
* `decode-<size>-r<rows>-k2s-<shapes>` (tasks/m23-spec.md P11R-12): the K2 small-M kernel's records of the decode
  steps grouped by launch geometry (CUPTI's grid and block, read from the chrome trace; `cere_ktest_k2s_geometry` of
  every decode call shape, `decode_k2_calls`; calls of one geometry share a row, e.g. gdn_out+attn_o), per step:
  launches (and the expected count), µs, MB moved (weights, x and y) and the effective TB/s;
  `decode-<size>-r<rows>-k2s-total` sums them and gives the other K2-family records (cuBLASLt, P3b's CUTLASS GEMMs)
  apart.
* `spec-<size>-r1` (tasks/m23-spec.md P11R-16, the 1.2K prompt): speculation as the `-v10` profile serves it on its
  own drafting model (`draft_token_map` = the pinned artifact's MTP draft map, mounted at DRAFT_MAP by native_dev.py),
  one session: rounds of `cere_ktest_session_speculate_mtp` with 3 MTP drafts (a verify step of 4 rows, then the
  draft-extend and 2 MTP decode steps) until 64 tokens are committed, unprofiled (host ms per round) and profiled
  (each round cut at a torch fill, d12_spec.py). The committed tokens must equal plain greedy's. Per round: span,
  busy and idle ms; per phase (verify, commit, mtp0..2, tail) span and busy ms, launches, us per group (k2_dense,
  lm_head, moe, k3, attn_gdn, sampler, copy, other; summed `_us` and exclusive `_excl_us`) and the gap before it; host
  ms per round (the unprofiled round minus the GPU span); accept length; and the mean distinct routed experts over 4
  consecutive positions against 1 (the teacher-forced `mlp.gate` top-k of the fed positions, mean over layers): what
  one verify step's MoE reads against a plain decode step's. `levers` names the model's slot selection: `default` (null
  levers), `spec-<size>-r1-small_m-graph` (`D12SpeculationFast`) repeats the cell on its own model created with the
  PI-06 sweep's best overall executor levers (k2_gemm = small_m, decode_step = graph), `spec-<size>-r1-batched`
  (`D12SpeculationBatched`) with verify_attention = batched (tasks/m23-spec.md P11R-19) alone and
  `spec-<size>-r1-small_m-graph-batched` (`D12SpeculationFastBatched`) with the three;
  `spec-<size>-r1-small_m-spec_graph` (`D12SpeculationSpecGraph`) and `spec-<size>-r1-small_m-spec_graph-batched`
  (`D12SpeculationSpecGraphBatched`) put decode_step at spec_graph (tasks/m23-spec.md P11R-20: the verify steps
  replay per-shape CUDA graphs) on k2_gemm = small_m, without and with verify_attention = batched;
  `spec-<size>-r1-spec_graph-batched` (`D12SpeculationShippedSet`) puts decode_step at spec_graph and
  verify_attention at batched on the default k2_gemm (the shipped speculation set's bit-identical executor levers).
* `spec-<size>-r4<suffix>` (tasks/m23-spec.md P11R-21), every class above: the cell with the 4 sessions speculating
  together (one `cere_ktest_session_speculate_mtp` call of 4 rows per round, until every row has 64 tokens; row r's
  prompt starts at record r x SPEC_ROW_RECORDS, and each row's tokens must equal its own plain greedy decode), with
  `rows` and the draft as one window (`draft_span_ms`, `draft_busy_ms`, `draft_idle_ms`, `draft_launches`,
  `draft_<group>_us`: the chains' MTP forwards together, CudaDevice::draft_rows, d12_spec.draft_fields); the distinct
  experts are row 0's.

Native entry points (K8, `oracle.call` appends the stream; 0 = success, else a `cere_status`):

    int cere_ktest_forward_profile(uint64_t model, uint64_t rows, const uint32_t *prompt /*host*/,
                                   uint64_t prompt_len, uint64_t steps, float *out_ms /*host [1 + steps]*/,
                                   cudaStream_t stream);
    int cere_ktest_decode_graph(uint64_t model, int32_t mode, uint64_t *out /*host [4]*/, cudaStream_t /*stream*/);
    int cere_ktest_session_speculate_mtp(const uint64_t *sessions, uint64_t rows, const uint32_t *tokens,
                                         const uint32_t *draft_counts, const uint32_t *max_commit,
                                         int32_t *row_status, uint32_t *results, cere_token_score *scores,
                                         cudaStream_t stream);

The speculation cell also opens, drafts, feeds and closes K8 sessions (`cere_ktest_session_*`, the M2.9 exports
test_model_k8_forward.py states) and reads the `mlp.gate` taps of `cere_ktest_forward_trace_detail` (detail 1).
"""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import os
import re
import statistics
import sys
import tempfile
import time
import unittest

import torch
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

import d12_spec
import oracle

ROW = "D12"
MAX_CONTEXT = 200_000  # native-g0.toml max_context_tokens
SESSIONS = 4  # native-g0.toml decode_batch_max
POOL = 200_000  # native-g0.toml kv_pool_tokens
STEPS = 64
SPEC_DRAFTS = 4  # native-g0-spec.toml speculative_draft_tokens: a verify chain of 1 + 3 rows
SPEC_CHAIN = SPEC_DRAFTS - 1  # drafts per verify chain
SPEC_FORWARDS = 3  # MTP forwards per round: the draft-extend and 2 decode steps (CudaDevice::draft)
SPEC_ROW_RECORDS = 5_000  # spec-<size>-r<rows>: row r's synthetic prompt starts at record r * this (another prompt)
SPEC_ROW_CONTEXT = 16_384  # spec-<size>-r<rows>, rows > 1: each session's context (256 pages of 64; 4 fit the pool)
DRAFT_MAP = "/draft_token_map.u32le"  # native_dev.py DRAFT_MAP_MOUNT: the pinned artifact's MTP draft token map
TRACE_ROWS = 16  # rows per forward_trace_detail call of the routed-experts count (bounds its value buffer)
# scripts/bench/frontend_prompts.py SIZES; 200K: 199,000 so that prompt + STEPS fits the context and the pool
SIZES = {"1.2K": 1_200, "18K": 18_000, "72K": 72_000, "200K": 199_000}
ATTENTION_CORES = ("attention_kernel", "combine_kernel")  # K6's and K9's softmax(q k^T) v kernels (P4 split)
MARKER = "profile_mark_kernel"
ATTN_GDN = ("k5", "k6", "k9")
SENTINEL = "<<RECORDS>>"
QUESTION = "\n\nWhich record lists the heaviest animal? Answer with its record number only."
COLORS = ("red", "green", "blue", "amber", "violet", "silver", "black")
ANIMALS = ("otter", "heron", "lynx", "badger", "falcon", "marten", "ibex", "crane", "viper", "bison", "gecko")
KERNEL = re.compile(r"cere::(\w+)::(?:\(anonymous namespace\)::)?(\w+)")
# The K2 small-M kernels as CUPTI demangles them (`16u` or `(unsigned int)16`): small_m_kernel<O, TN, Hi> (O: 0 BF16,
# 1 F32, 2 K3's activation fused, gemm_hc_act) and hc_gate_kernel<TN, Hi> (K3's gate fused, gemm_hc_gate; P11R-21)
_UINT = r"(?:\(unsigned int\))?(\d+)u?"
K2S_SMALL_M = re.compile(rf"small_m_kernel<{_UINT}, {_UINT}, (?:true|false)>")
K2S_HC_GATE = re.compile(rf"hc_gate_kernel<{_UINT}, (?:true|false)>")
K2S_OUT_HC_GATE = 3  # the launch key's output code of hc_gate_kernel (small_m_kernel's are its O)
K2S_EPILOGUES = ("bf16", "f32", "hc_act", "hc_gate")  # by output code
K9_INDEX_HEADS = 4  # k9_qsa.h kIndexHeads: index_qk_proj rows = (heads + 1) * head dim
K9_INDEX_HEAD_DIM = 128  # k9_qsa.h kIndexHeadDim
PLE_EMBED = 2560  # k4_ple.h kEmbedDim (ple_embed_dim): the PLE key/value projections' k


@functools.cache
def tokenizer():
    return get_tokenizer(str(oracle.MODEL_DIR))


def synthetic_ids(total: int, first: int = 0) -> list[int]:
    """Exactly `total` prompt IDs: the chat template around numbered records (cut in token space; records `first`
    on, so another `first` is another prompt of the same length) and a question."""
    tok = tokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": SENTINEL + QUESTION}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    if text.count(SENTINEL) != 1:
        raise oracle.OracleError("the rendered chat template does not hold the record sentinel exactly once")
    head, tail = text.split(SENTINEL)
    head_ids = list(tok.encode(head, add_special_tokens=False))
    tail_ids = list(tok.encode(tail, add_special_tokens=False))
    need = total - len(head_ids) - len(tail_ids)
    records = "\n".join(f"Record {i:06d}: the {COLORS[i % len(COLORS)]} {ANIMALS[i % len(ANIMALS)]} weighs "
                        f"{(i * 37) % 101 + 1} kg." for i in range(first, first + max(need // 8, 1)))
    body_ids = list(tok.encode(records, add_special_tokens=False))
    if need <= 0 or len(body_ids) < need:
        raise oracle.OracleError(f"synthetic prompt of {total} tokens: {len(body_ids)} record tokens for {need}")
    return head_ids + body_ids[:need] + tail_ids


def family(name: str) -> str:
    """The kernel family of one CUPTI record name (module docstring)."""
    if name.startswith("Memcpy"):
        return "memcpy"
    if name.startswith("Memset"):
        return "memset"
    if "tensorrt_llm" in name or "cere::p3b::" in name:
        return "p3b"
    m = KERNEL.search(name)
    if m is None:
        return "k2"
    namespace, kernel = m.groups()
    if namespace == "ktest":
        return "marker" if kernel == MARKER else "ktest"
    if namespace == "kernels":
        if kernel in ("quantize_kernel", "dequant_kernel"):
            return "k0"
        if kernel in d12_spec.K2_NAMESPACES:  # the in-tree K2 kernels (m23 design note P11, P11R-21, P11R-24; O144)
            return "k2"
        return "k3" if kernel.startswith("hc_") else "k7"
    return namespace


def run(handle: int, ids: list[int], rows: int) -> list[float]:
    """One cere_ktest_forward_profile call: [session 0's prefill ms, decode step ms ...]."""
    prompt = torch.tensor(ids, dtype=torch.int32)
    out = torch.zeros(1 + STEPS, dtype=torch.float32)
    oracle.call("cere_ktest_forward_profile", handle, rows, prompt, len(ids), STEPS, out)
    return out.tolist()


def decode_graph(handle: int, mode: int) -> list[int]:
    """P1's switch (mode 0 off, 1 on, -1 query): [enabled, captures, updates, replays] after it."""
    out = torch.zeros(4, dtype=torch.int64)
    oracle.call("cere_ktest_decode_graph", handle, mode, out)
    return out.tolist()



def launch_dims(prof) -> dict[int, tuple[int, ...]]:
    """(grid x, y, z, block x, y, z) of every kernel record by CUPTI correlation id: Kineto writes them into the
    chrome trace's kernel args only (`_KinetoEvent.metadata_json` is empty for GPU records)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    return {int(ev["args"]["correlation"]): tuple(int(v) for v in ev["args"]["grid"] + ev["args"]["block"])
            for ev in events if ev.get("cat") == "kernel"}


def is_k2s(name: str) -> bool:
    m = KERNEL.search(name)
    return m is not None and m.groups() == ("kernels", "k2s")


def k2s_launch(name: str, dims: tuple[int, ...]) -> tuple[int, ...]:
    """A K2 small-M record's launch key: (grid x, y, z, block x, y, z, output code, tile columns), the last two from
    the kernel's template arguments (`small_m_kernel<O, TN, Hi>`, `hc_gate_kernel<TN, Hi>`: K2S_EPILOGUES), so two
    shapes of one grid and block stay apart."""
    m = K2S_SMALL_M.search(name)
    if m is not None:
        return dims + (int(m.group(1)), int(m.group(2)))
    m = K2S_HC_GATE.search(name)
    if m is not None:
        return dims + (K2S_OUT_HC_GATE, int(m.group(1)))
    raise oracle.OracleError(f"no small_m_kernel<O, TN, Hi> or hc_gate_kernel<TN, Hi> template arguments in "
                             f"{name[:120]}")


def gpu_records(prof) -> list[tuple[int, int, str, tuple[int, ...] | None]]:
    """(start ns, duration ns, name, k2s_launch key of a K2 small-M kernel record else None) of every GPU record, in
    start order."""
    dims = launch_dims(prof)
    out = []
    for e in prof.profiler.kineto_results.events():
        if e.device_type() == torch.autograd.DeviceType.CUDA:
            name = e.name()
            launch = None
            if is_k2s(name):
                launch = dims.get(e.correlation_id())
                if launch is None:
                    raise oracle.OracleError(f"no chrome-trace launch dims for {name[:80]}")
                launch = k2s_launch(name, launch)
            out.append((e.start_ns(), e.duration_ns(), name, launch))
    out.sort(key=lambda r: (r[0], r[1], r[2]))
    return out


def segments(records: list, marks: int) -> list[list[tuple[int, int, str, tuple[int, ...] | None]]]:
    """The records between consecutive markers (the marks - 1 segments) as (start, duration, family, k2s launch);
    exactly `marks` markers or OracleError."""
    at = [i for i, r in enumerate(records) if family(r[2]) == "marker"]
    if len(at) != marks:
        raise oracle.OracleError(f"{len(at)} profile markers in the trace, expected {marks}")
    return [[(s, d, family(n), g) for s, d, n, g in records[a + 1:b]] for a, b in zip(at, at[1:])]


def by_family(records: list) -> tuple[collections.Counter, collections.Counter]:
    count: collections.Counter = collections.Counter()
    ns: collections.Counter = collections.Counter()
    for _, duration, fam, _ in records:
        count[fam] += 1
        ns[fam] += duration
    return count, ns


def attention_split(records: list) -> tuple[float, float]:
    """(attention-core ms, QSA-selection ms) of session 0's prefill: the records between the first two markers."""
    at = [i for i, r in enumerate(records) if family(r[2]) == "marker"]
    core = select = 0
    for _, duration, name, _ in records[at[0] + 1:at[1]]:
        m = KERNEL.search(name)
        if m is None:
            continue
        namespace, kernel = m.groups()
        if namespace == "p4" or (namespace in ("k6", "k9") and kernel in ATTENTION_CORES):
            core += duration
        elif namespace == "k9":
            select += duration
    return core / 1e6, select / 1e6


def shape_of(name: str) -> tuple[int, int]:
    """(out_features, in_features) of one checkpoint weight, from its safetensors header (no tensor is read)."""
    with safe_open(str(oracle.MODEL_DIR / oracle._weight_map()[name]), framework="pt", device="cpu") as f:
        n, k = f.get_slice(name).get_shape()
    return int(n), int(k)


@functools.cache
def decode_k2_calls(rows: int) -> tuple[tuple[str, int, tuple[int, ...], int, int, int | None], ...]:
    """The K2 small-M calls of one decode step at `rows` (forward.cpp run_rows, k3_hc.cu hc_mix, k7_moe.cu,
    p3b_moe.cu, k4_ple.cu): (label, k, segment widths, m, output bytes per element, launches per step or None when
    the count is not fixed here: the K9 indexer's qk projection runs only past K6's context, K3's hc mixes per layer
    and before the final norm)."""
    cfg = oracle.text_config()
    layers = int(cfg["num_hidden_layers"])
    p = oracle.layer_prefix
    full = [i for i in range(layers) if oracle.has_tensor(p(i) + "self_attn.o_proj.weight")]
    gdn = [i for i in range(layers) if i not in full]
    attn, lin, mlp = p(full[0]) + "self_attn.", p(gdn[0]) + "linear_attn.", p(0) + "mlp."
    hc = p(0) + "attn_hyper_connection."
    qkv = [shape_of(attn + f"{x}_proj.weight") for x in ("q", "k", "v")]
    gdn_in = [shape_of(lin + f"in_proj_{x}.weight") for x in ("qkv", "z", "b", "a")]
    hidden = int(cfg["hidden_size"])
    calls = [
        ("hc_down", shape_of(hc + "input_mix_weight_down.weight"), 4, None),
        ("hc_up", shape_of(hc + "input_mix_weight_up.weight"), 4, None),
        ("attn_qkv", (sum(n for n, _ in qkv), qkv[0][1]), 2, len(full)),
        ("attn_o", shape_of(attn + "o_proj.weight"), 2, len(full)),
        ("attn_index_qk", ((K9_INDEX_HEADS + 1) * K9_INDEX_HEAD_DIM, hidden), 2, None),
        ("gdn_out", shape_of(lin + "out_proj.weight"), 2, len(gdn)),
        ("moe_router", shape_of(mlp + "gate.weight"), 2, layers),  # BF16 logits (forward.cpp moe_router plan)
        ("moe_shared_down", shape_of(mlp + "shared_expert.down_proj.weight"), 2, layers),
        ("lm_head", (int(cfg["vocab_size"]), hidden), 2, 1),
    ]
    out = [(label, k, (n,), rows, size, count) for label, (n, k), size, count in calls]
    out.append(("gdn_in", gdn_in[0][1], tuple(n for n, _ in gdn_in), rows, 2, len(gdn)))
    gate_up = [shape_of(mlp + f"shared_expert.{x}_proj.weight") for x in ("gate", "up")]
    out.append(("moe_shared_gate_up", gate_up[0][1], tuple(n for n, _ in gate_up), rows, 2, layers))
    # PLE (layer 1) runs per row at m = 1: key [kHidden * hc, embed] and value [hidden, embed] (k4_ple.h).
    out.append(("ple_key", PLE_EMBED, (hidden * int(cfg["hc_count"]),), 1, 2, rows))
    out.append(("ple_value", PLE_EMBED, (hidden,), 1, 2, rows))
    return tuple(out)


def mb_moved(calls: list, counts: list) -> float:
    """MB one step moves through the given calls: BF16 weights [sum(ns), k] and x [m, k], y [m, sum(ns)]."""
    return sum(n_calls * (2 * k * sum(ns) + 2 * m * k + size * m * sum(ns))
               for n_calls, (_, k, ns, m, size, _) in zip(counts, calls)) / 1e6


def k2s_rows(size: str, rows: int, decode: list) -> None:
    """Per-shape K2 rows of a decode cell (P11R-12): the small-M kernel records grouped by launch key (k2s_launch;
    `cere_ktest_k2s_geometry` of each call in decode_k2_calls, its output type and tile width; calls of one key share
    a row), per step: launches (measured, expected), µs, MB moved (weights, x and y) and the effective TB/s; then the
    k2s total and the other K2-family records (cuBLASLt, P3b's CUTLASS GEMMs). The hc mix's GEMMs also map under
    their fused launches (P11R-21, hc_mix's small-M path): hc_down as `hc_down_act` (BF16 t) and hc_up as
    `hc_up_gate` (`cere_ktest_k2s_hc_gate_geometry`; its y term counts the normed streams it reads)."""
    groups: dict[tuple[int, ...], list] = collections.defaultdict(list)
    for call in decode_k2_calls(rows):
        label, k, ns, m, out_bytes, n_calls = call
        dims = torch.zeros(9, dtype=torch.int64)  # grid, block, cluster: CUPTI's records give grid and block
        oracle.call("cere_ktest_k2s_geometry", k, len(ns), torch.tensor(ns, dtype=torch.int64), dims)
        v = [int(x) for x in dims.tolist()]
        tile_n = 16 if v[0] == v[6] * sum(-(-n // 16) for n in ns) else 8
        groups[tuple(v[:6]) + (int(out_bytes == 4), tile_n)].append(call)
        if label == "hc_down":
            groups[tuple(v[:6]) + (2, tile_n)].append(("hc_down_act", k, ns, m, 2, n_calls))
        elif label == "hc_up":
            hidden = int(oracle.text_config()["hidden_size"])  # hc_up's n is hc_count streams of hidden
            oracle.call("cere_ktest_k2s_hc_gate_geometry", k, hidden, dims)
            g = [int(x) for x in dims.tolist()]
            gate_tile = 16 if g[0] == -(-hidden // 16) else 8
            groups[tuple(g[:6]) + (K2S_OUT_HC_GATE, gate_tile)].append(("hc_up_gate", k, ns, m, 2, n_calls))
    count: collections.Counter = collections.Counter()
    ns_by: collections.Counter = collections.Counter()
    other_n = other_ns = 0
    for _, duration, fam, launch in decode:
        if launch is not None:
            count[launch] += 1
            ns_by[launch] += duration
        elif fam == "k2":
            other_n += 1
            other_ns += duration
    total_us = total_mb = 0.0
    for launch in sorted(count, key=lambda g: -ns_by[g]):
        calls = groups.get(launch, [])
        launches = count[launch] / STEPS
        us = ns_by[launch] / 1e3 / STEPS
        unknown = [c for c in calls if c[5] is None]
        known = sum(c[5] for c in calls if c[5] is not None)
        mb = None
        if len(calls) == 1:  # one call shape: the measured launches
            mb = mb_moved(calls, [launches])
        elif calls and len(unknown) <= 1:  # a shared geometry: the fixed counts, the rest to the unfixed one
            mb = mb_moved(calls, [c[5] if c[5] is not None else launches - known for c in calls])
        label = "+".join(c[0] for c in calls) or "unmapped_" + "x".join(str(v) for v in launch)
        total_us += us
        total_mb += mb or 0.0
        oracle.report_row(ROW, f"decode-{size}-r{rows}-k2s-{label}", rows=rows, grid="x".join(map(str, launch[:3])),
                          block="x".join(map(str, launch[3:6])), f32=launch[6] == 1,
                          epilogue=K2S_EPILOGUES[launch[6]], tile_n=launch[7],
                          launches_per_step=launches,
                          launches_expected=known if not unknown else None, us_per_step=us,
                          us_per_launch=us / launches, mb_per_step=mb,
                          tbps=None if mb is None else mb / us)
    oracle.report_row(ROW, f"decode-{size}-r{rows}-k2s-total", rows=rows,
                      launches_per_step=sum(count.values()) / STEPS, us_per_step=total_us, mb_per_step=total_mb,
                      tbps=total_mb / total_us if total_us else None, k2_other_launches_per_step=other_n / STEPS,
                      k2_other_us_per_step=other_ns / 1e3 / STEPS)


class D12Profile(unittest.TestCase):
    handle: int = 0

    @classmethod
    def setUpClass(cls):
        if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise oracle.OracleError("torch.profiler has no CUDA activity (CUPTI) in this image")
        cls.handle = oracle.create_model(MAX_CONTEXT, max_sessions=SESSIONS, kv_pool_tokens=POOL)

    @classmethod
    def tearDownClass(cls):
        if cls.handle:
            oracle.release_model(cls.handle)
            cls.handle = 0

    def profile_cell(self, size: str, rows: int) -> None:
        ids = synthetic_ids(SIZES[size])
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            profiled = run(self.handle, ids, rows)
        before = decode_graph(self.handle, -1)
        wall = run(self.handle, ids, rows)
        after = decode_graph(self.handle, -1)
        # The other arm of the decode_step lever (graph when the model runs eager, the default, else eager); a graph
        # arm first runs once unmeasured, so its captures fall outside the timed run.
        decode_graph(self.handle, 1 - before[0])
        try:
            if not before[0]:
                run(self.handle, ids, rows)
            other = run(self.handle, ids, rows)
        finally:
            decode_graph(self.handle, before[0])
        graph, eager = (wall, other) if before[0] else (other, wall)
        records = gpu_records(prof)
        seg = segments(records, STEPS + 3)
        prefill, decode = seg[0], [r for step in seg[2:] for r in step]
        for s, step in enumerate(seg[2:]):
            self.assertTrue(step, f"decode step {s} holds no GPU record")
        self.assertNotIn("ktest", {r[2] for r in prefill + decode}, "ktest kernels inside the forward")
        for name in sorted({r[2] for r in records if family(r[2]) == "k2"}):
            print(f"D12 k2 kernel: {name[:160]}", file=sys.stderr)

        count, ns = by_family(prefill)
        busy_ms = sum(ns.values()) / 1e6
        attention_ms, select_ms = attention_split(records)
        oracle.report_row(ROW, f"prefill-{size}", prompt_tokens=len(ids), launches=sum(count.values()),
                          busy_ms=busy_ms, wall_ms=wall[0], profiled_wall_ms=profiled[0],
                          attention_ms=attention_ms, qsa_select_ms=select_ms,
                          attention_share_busy=attention_ms / busy_ms, attention_share_wall=attention_ms / wall[0],
                          **{f"{f}_launches": c for f, c in count.items()},
                          **{f"{f}_ms": t / 1e6 for f, t in ns.items()})

        count, ns = by_family(decode)
        steps = wall[1:]
        launches = sum(count.values()) / STEPS
        copies = (count["memcpy"] + count["memset"]) / STEPS
        busy_ms = sum(ns.values()) / 1e6 / STEPS
        mean_ms = statistics.fmean(steps)
        eager_ms = statistics.fmean(eager[1:])
        graph_ms = statistics.fmean(graph[1:])
        attn_gdn = sum(count[f] for f in ATTN_GDN) / STEPS
        spans = [(max(s + d for s, d, _, _ in step) - step[0][0]) / 1e6 for step in seg[2:]]
        oracle.report_row(ROW, f"decode-{size}-r{rows}", prompt_tokens=len(ids), rows=rows, steps=STEPS,
                          launches_per_step=launches, kernels_per_step=launches - copies, copies_per_step=copies,
                          busy_ms_per_step=busy_ms, gpu_span_ms_per_step=statistics.fmean(spans),
                          step_ms_mean=mean_ms, step_ms_median=statistics.median(steps),
                          profiled_step_ms_median=statistics.median(profiled[1:]),
                          tok_s=rows * 1000.0 / mean_ms, overhead_share=1.0 - busy_ms / mean_ms,
                          attn_gdn_launches_per_step=attn_gdn, attn_gdn_share=attn_gdn / launches,
                          attn_gdn_launches_per_row=attn_gdn / rows,
                          **{f"{f}_launches": c / STEPS for f, c in count.items()},
                          **{f"{f}_us": t / 1e3 / STEPS for f, t in ns.items()},
                          **{f"{f}_excl_us": t / 1e3 / STEPS
                             for f, t in d12_spec.exclusive_by([(s, d, f) for s, d, f, _ in decode]).items()},
                          decode_graph=before[0], graph_steps=after[3] - before[3],
                          graph_captures=(after[1] + after[2]) - (before[1] + before[2]),
                          eager_step_ms_mean=eager_ms, eager_step_ms_median=statistics.median(eager[1:]),
                          eager_tok_s=rows * 1000.0 / eager_ms, graph_step_ms_mean=graph_ms,
                          graph_step_ms_median=statistics.median(graph[1:]), graph_tok_s=rows * 1000.0 / graph_ms,
                          graph_speedup=eager_ms / graph_ms)
        k2s_rows(size, rows, decode)

    def test_profile_1k2_rows1(self):
        self.profile_cell("1.2K", 1)

    def test_profile_1k2_rows4(self):
        self.profile_cell("1.2K", 4)

    def test_profile_18k_rows1(self):
        self.profile_cell("18K", 1)

    def test_profile_72k_rows1(self):
        self.profile_cell("72K", 1)

    def test_profile_200k_rows1(self):
        self.profile_cell("200K", 1)


@contextlib.contextmanager
def opened_session(handle: int, *, draft: bool, context: int = 0):
    """A session of the model (drafting with `draft`, set while it is empty) of `context` tokens (0: the model's),
    closed on exit."""
    out = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_open", handle, context, out)
    session = int(out[0])
    try:
        if draft:
            oracle.call("cere_ktest_session_draft", session, 1)
        yield session
    finally:
        oracle.call("cere_ktest_session_close", session)


def session_feed(session: int, ids: list[int]) -> int:
    """Teacher-forced feed (PREFILL on an empty session, else one DECODE per token): row 0's top-1 (a PREFILL scores
    row 0 only)."""
    scores = torch.full((len(ids), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_session_feed", session, torch.tensor(ids, dtype=torch.int32), len(ids), 0, 0, None, 0,
                None, 0, None, 0, scores, counts)
    return int(scores[0, 0])


def speculate_mtp(sessions: list[int], tokens: list[int]) -> list[tuple[int, int, list[int]]]:
    """One greedy speculate_rows call of `sessions` (row r fed tokens[r]), each drafting SPEC_CHAIN pending MTP
    proposals and committing up to SPEC_DRAFTS tokens: per row (committed, accepted, the committed tokens)."""
    rows = len(sessions)
    status = torch.full((rows,), 1, dtype=torch.int32)
    results = torch.zeros((rows, 2), dtype=torch.int32)
    scores = torch.full((rows, 4, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_speculate_mtp", torch.tensor(sessions, dtype=torch.int64), rows,
                torch.tensor(tokens, dtype=torch.int32), torch.full((rows,), SPEC_CHAIN, dtype=torch.int32),
                torch.full((rows,), SPEC_DRAFTS, dtype=torch.int32), status, results, scores)
    out = []
    for r in range(rows):
        if int(status[r]) != 0:
            raise oracle.OracleError(f"cere_ktest_session_speculate_mtp row {r} status {int(status[r])}")
        c = int(results[r, 0])
        out.append((c, int(results[r, 1]), [int(v) for v in scores[r, :c, 0].tolist()]))
    return out


def router_top(handle: int, ids: list[int], rows: int) -> list[list[list[int]]]:
    """Per MoE layer, per position of `ids`'s last `rows` (a multiple of TRACE_ROWS): the top num_experts_per_tok
    router ids of the teacher-forced forward's `mlp.gate` logits (`cere_ktest_forward_trace_detail` detail 1, BF16;
    ties may order differently from K7's FP32 softmax), TRACE_ROWS positions per call."""
    tc = oracle.text_config()
    hidden, layers = int(tc["hidden_size"]), int(tc["num_hidden_layers"])
    value_dim = int(tc["linear_num_value_heads"]) * int(tc["linear_value_head_dim"])
    head = int(tc["head_dim"])
    cols = max(int(tc["hc_count"]) * hidden, int(tc["num_experts"]), 2 * int(tc["shared_expert_intermediate_size"]),
               2 * int(tc["linear_num_key_heads"]) * int(tc["linear_key_head_dim"]) + 2 * value_dim,
               2 * (int(tc["num_attention_heads"]) + int(tc["num_key_value_heads"])) * head)
    capacity = 16 * (layers + 1)  # detail 1 emits at most 14 entries per layer and 3 more
    top = int(tc["num_experts_per_tok"])
    out: list[list[list[int]]] = [[] for _ in range(layers)]
    for end in range(len(ids) - rows + TRACE_ROWS, len(ids) + 1, TRACE_ROWS):
        names = torch.zeros(capacity * 128, dtype=torch.uint8)
        shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
        values = oracle.empty((capacity * TRACE_ROWS * cols,), torch.bfloat16)
        scores = torch.full((TRACE_ROWS, 4), -1, dtype=torch.int32)
        counts = torch.zeros(3, dtype=torch.int64)
        oracle.call("cere_ktest_forward_trace_detail", handle, torch.tensor(ids[:end], dtype=torch.int32), end, 0,
                    TRACE_ROWS, 1, names, names.numel(), shapes, capacity, values,
                    values.numel() * values.element_size(), scores, counts)
        entries, name_bytes = int(counts[0]), int(counts[1])
        emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
        if not 0 < entries <= capacity or len(emitted) != entries + 1:
            raise oracle.OracleError(f"forward_trace_detail wrote {len(emitted) - 1} names for {entries} entries")
        gates, at = {}, 0
        for name, (_, r, c) in zip(emitted, shapes[:entries].tolist()):
            gates[name.decode()] = values[at:at + r * c].view(r, c)
            at += r * c
        for layer in range(layers):
            logits = gates[f"model.layers.{layer}.mlp.gate"].float()
            out[layer] += torch.topk(logits, top, dim=1).indices.tolist()
    return out



class D12Speculation(unittest.TestCase):
    """The speculation cell (module docstring `spec-<size>-r1`) on its own drafting model, at the levers' defaults."""

    handle: int = 0
    # The model's slot choices (lever -> CERE_LEVER_* value; empty: null, every lever at its default) and the row's
    # case suffix and `levers` field naming them.
    LEVERS: dict[str, int] = {}
    LEVERS_NAME = "default"
    CASE_SUFFIX = ""

    @classmethod
    def setUpClass(cls):
        if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise oracle.OracleError("torch.profiler has no CUDA activity (CUPTI) in this image")
        with open(DRAFT_MAP, "rb") as f:
            draft_map = torch.frombuffer(bytearray(f.read()), dtype=torch.int32)
        cls.model_levers = oracle.levers(**cls.LEVERS) if cls.LEVERS else None
        cls.handle = oracle.create_model(MAX_CONTEXT, max_sessions=SESSIONS, kv_pool_tokens=POOL,
                                         draft_token_map=draft_map, model_levers=cls.model_levers)

    @classmethod
    def tearDownClass(cls):
        if cls.handle:
            oracle.release_model(cls.handle)
            cls.handle = 0

    def spec_run(self, prompts: list[list[int]], mark: torch.Tensor | None) -> tuple[list[list[int]],
                                                                                      list[list[int]], list[float]]:
        """One drafting session per prompt, each fed its prompt, then speculation rounds of every row together (one
        speculate_rows call; a full chain per row: SPEC_CHAIN MTP drafts, up to SPEC_DRAFTS committed) until every
        row has STEPS tokens committed: (per row the PREFILL token then every committed token, per round the rows'
        committed counts, the host ms of each round). With `mark`, a fill of it precedes the sessions' open, each
        round's call and the sessions' close: the records between the second and the last fill are the rounds'."""
        if mark is not None:
            mark.fill_(0.0)
        with contextlib.ExitStack() as stack:
            # A session of the model's context takes the whole pool: rows > 1 open SPEC_ROW_CONTEXT tokens each.
            context = 0 if len(prompts) == 1 else SPEC_ROW_CONTEXT
            sessions = [stack.enter_context(opened_session(self.handle, draft=True, context=context))
                        for _ in prompts]
            tokens = [[session_feed(s, ids)] for s, ids in zip(sessions, prompts)]
            committed: list[list[int]] = []
            ms = []
            while min(len(t) for t in tokens) <= STEPS:
                if mark is not None:
                    mark.fill_(0.0)
                t0 = time.perf_counter()
                got = speculate_mtp(sessions, [t[-1] for t in tokens])
                ms.append((time.perf_counter() - t0) * 1e3)
                for r, (c, accepted, new) in enumerate(got):
                    self.assertTrue(1 <= c <= SPEC_DRAFTS and accepted == c - 1,
                                    f"{ROW} spec round {len(committed)} row {r}: committed {c} accepted {accepted}")
                    tokens[r] += new
                committed.append([c for c, _, _ in got])
            if mark is not None:
                mark.fill_(0.0)
        return tokens, committed, ms

    def spec_cell(self, size: str, rows: int = 1) -> None:
        """`spec-<size>-r<rows>`: row r's prompt is synthetic_ids(size, first=r * SPEC_ROW_RECORDS) (row 0's is the
        r1 cell's), every row's committed tokens must equal its plain greedy decode; the distinct experts are row
        0's."""
        prompts = [synthetic_ids(SIZES[size], first=r * SPEC_ROW_RECORDS) for r in range(rows)]
        case = f"spec-{size}-r{rows}{self.CASE_SUFFIX}"
        tokens, committed, ms = self.spec_run(prompts, None)
        mark = torch.zeros(1, device=oracle.DEVICE)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            profiled_tokens, profiled_committed, _ = self.spec_run(prompts, mark)
        self.assertEqual((profiled_tokens, profiled_committed), (tokens, committed),
                         f"{ROW} {case}: the profiled run committed other tokens")
        for r, ids in enumerate(prompts):
            with opened_session(self.handle, draft=False) as session:
                plain = [session_feed(session, ids)]
                for _ in range(STEPS):
                    plain.append(session_feed(session, plain[-1:]))
            self.assertEqual(tokens[r][:len(plain)], plain, f"{ROW} {case}: row {r}'s speculation differs from plain "
                             "greedy")
        records = [(s, d, n) for s, d, n, _ in gpu_records(prof)]  # d12_spec's Record: (start, duration, name)
        end = max(i for i, (_, _, n) in enumerate(records) if d12_spec.ROUND_MARK in n)
        cut = d12_spec.rounds(records[:end], 1 + len(committed))[1:]
        # The rows of a round draft together (CudaDevice::draft_rows: one MTP forward per pass over every chain's rows).
        phases = [d12_spec.phases(r, SPEC_FORWARDS) for r in cut]
        fields = d12_spec.round_fields(phases)
        round_ms = statistics.fmean(ms)
        total = sum(map(sum, committed))
        layers = router_top(self.handle, prompts[0] + tokens[0][:STEPS], STEPS)
        experts = {f"distinct_experts_w{w}": statistics.fmean(d12_spec.distinct_experts(pos, w) for pos in layers)
                   for w in (1, SPEC_DRAFTS)}
        oracle.report_row(ROW, case, levers=self.LEVERS_NAME, prompt_tokens=len(prompts[0]), rows=rows,
                          rounds=len(committed), committed=total, accept_length=total / (rows * len(committed)),
                          verify_rows=rows * SPEC_DRAFTS, mtp_forwards=SPEC_FORWARDS, round_ms_mean=round_ms,
                          round_ms_median=statistics.median(ms), tok_s=total * 1000.0 / sum(ms),
                          host_ms_per_round=round_ms - fields["span_ms"], **experts,
                          distinct_experts_ratio=experts[f"distinct_experts_w{SPEC_DRAFTS}"]
                          / experts["distinct_experts_w1"], **fields, **d12_spec.draft_fields(phases))

    def test_profile_spec_1k2_rows1(self):
        self.spec_cell("1.2K")

    def test_profile_spec_1k2_rows4(self):
        self.spec_cell("1.2K", SESSIONS)


class D12SpeculationFast(D12Speculation):
    """The speculation cell on the PI-06 sweep's best overall executor levers (bench/reports/
    perf_sweep-20261013T230019Z.json: k2_gemm = small_m, decode_step = graph; draft_policy is cered's, and the cell's
    rounds always draft the full chain)."""

    LEVERS = {"k2_gemm": 1, "decode_step": 1}  # CERE_LEVER_K2_SMALL_M, CERE_LEVER_DECODE_GRAPH
    LEVERS_NAME = "k2_gemm=small_m,decode_step=graph"
    CASE_SUFFIX = "-small_m-graph"


class D12SpeculationBatched(D12Speculation):
    """The speculation cell with verify_attention = batched (tasks/m23-spec.md P11R-19), every other lever at its
    default: the bit-identical set."""

    LEVERS = {"verify_attention": 1}  # CERE_LEVER_VERIFY_BATCHED
    LEVERS_NAME = "verify_attention=batched"
    CASE_SUFFIX = "-batched"


class D12SpeculationFastBatched(D12Speculation):
    """The speculation cell on D12SpeculationFast's levers plus verify_attention = batched (P11R-19)."""

    LEVERS = {"k2_gemm": 1, "decode_step": 1, "verify_attention": 1}
    LEVERS_NAME = "k2_gemm=small_m,decode_step=graph,verify_attention=batched"
    CASE_SUFFIX = "-small_m-graph-batched"


class D12SpeculationSpecGraph(D12Speculation):
    """The speculation cell on k2_gemm = small_m and decode_step = spec_graph (tasks/m23-spec.md P11R-20)."""

    LEVERS = {"k2_gemm": 1, "decode_step": 2}  # CERE_LEVER_K2_SMALL_M, CERE_LEVER_DECODE_SPEC_GRAPH
    LEVERS_NAME = "k2_gemm=small_m,decode_step=spec_graph"
    CASE_SUFFIX = "-small_m-spec_graph"


class D12SpeculationSpecGraphBatched(D12Speculation):
    """D12SpeculationSpecGraph's levers plus verify_attention = batched: D12SpeculationFastBatched with spec_graph."""

    LEVERS = {"k2_gemm": 1, "decode_step": 2, "verify_attention": 1}
    LEVERS_NAME = "k2_gemm=small_m,decode_step=spec_graph,verify_attention=batched"
    CASE_SUFFIX = "-small_m-spec_graph-batched"


class D12SpeculationShippedSet(D12Speculation):
    """The speculation cell on the bit-identical executor levers of the shipped speculation set (tasks/m23-spec.md
    P11R-21: decode_step = spec_graph, verify_attention = batched, k2_gemm at its default cublaslt; draft_policy is
    cered's, and the cell's rounds always draft the full chain)."""

    LEVERS = {"decode_step": 2, "verify_attention": 1}  # CERE_LEVER_DECODE_SPEC_GRAPH, CERE_LEVER_VERIFY_BATCHED
    LEVERS_NAME = "decode_step=spec_graph,verify_attention=batched"
    CASE_SUFFIX = "-spec_graph-batched"


if __name__ == "__main__":
    unittest.main()
