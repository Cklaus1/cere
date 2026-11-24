"""K7: the NVFP4 MoE block (tasks/m22-spec.md K7, the NVFP4 MoE row of "Model-specific kernel decisions", Q14).

Suite `kernels` (inside the pinned SGLang image on GPU0):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_*.py' -v`.

Reference: SGLang's Qwen3Next/qwen4_exp MoE block (`Qwen2MoeSparseMoeBlock`, `sglang/srt/models/qwen2_moe.py`) with
`--moe-runner-backend flashinfer_cutlass` and the ModelOpt NVFP4 MoE method, rebuilt here from its pieces exactly as
the server runs them (no distributed state is needed for that; the server's code path is mirrored call by call):
  * router logits: the unquantized BF16 `mlp.gate` linear (`F.linear`);
  * routing: `fused_topk(..., topk=10, renormalize=True)` (`sglang/srt/layers/moe/topk.py:827`; `norm_topk_prob`
    absent from the config, Qwen3Next default True, `sglang/srt/configs/qwen3_next.py:214`), which on CUDA runs the
    Triton `moe_fused_gate` softmax router (`sglang/kernels/ops/moe/moe_fused_gate.py:255`): FP32 softmax over all
    experts, slot k = the k-th largest logit (ties: lowest expert id), weights = softmax probabilities renormalized
    by their sum over the selected 10;
  * experts: `flashinfer.fused_moe.cutlass_fused_moe` called as `_run_flashinfer_cutlass` does
    (`sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:178-244`) with the weights and scales prepared as
    `ModelOptNvFp4FusedMoEMethod.process_weights_after_loading` prepares them for flashinfer_cutlass
    (`sglang/srt/layers/quantization/modelopt_quant.py`): w13 = [up; gate] per expert (`load_up_proj_weight_first`,
    `:2696-2701`, `moe/fused_moe_triton/layer.py:653-660`), block scales `swizzle_blockscale`d (`:2600`,
    `quantization/utils.py:597`), one activation global scale per layer and GEMM = FP32 1/max `input_scale` (w13: over
    all experts and both gate/up shards; w2: over all experts; `:2425-2427,2478-2483`), `g1_alphas` =
    max_w13_input_scale x gate.weight_scale_2 for both GEMM1 halves (`:2112-2135`), `g2_alphas` =
    max_w2_input_scale x down.weight_scale_2, quant_scales passed as in `apply` (`:2872-2899`);
  * shared expert: BF16 `gate_up_proj` (gate rows first), `silu_and_mul`, `down_proj`; sigmoid-gated and added by
    `fused_gate_sigmoid_mul_add` (`qwen2_moe.py:624-631`).

Calibration reference (tolerance rule): FP32-dequantized weights with the same quantized activations, accumulated in
FP64: the input quantized by FlashInfer `fp4_quantize` with the layer's w13 scale and dequantized; per expert the
FP32 dequantized gate/up (both with the gate's `weight_scale_2`, the reference's `g1_alphas` semantics) and down
weights; the SwiGLU intermediate re-quantized with the layer's w2 scale; the reference's routing (ids and weights);
the BF16 shared expert and its sigmoid gate on the unquantized input. Router weights calibrate against an FP64
softmax at the reference's ids. Floors: FLOOR_OP for router weights, FLOOR_BLOCK for the block.

Inputs (same definition as `test_layout_k0_nvfp4.py`), M in {1, 64, 2048} (the first M rows):
  (i) layer 0: the MoE input of the parity prompts' tokens. The checkpoint has no `post_attention_layernorm`
      (`Qwen4ExpLayerExtensionMixin` deletes it, `sglang/srt/models/qwen4_exp.py:1235-1239`); its place before the MLP
      is `mlp_hyper_connection.mix` (`qwen4_exp.py:1318`): the per-branch Gemma RMSNorm `hc_norm` followed by the
      low-rank input mix (`sglang/srt/layers/hyperconnection.py` `GroupedGemmaRMSNorm.forward`, `_mix_compute`).
      Input (i) is that function, on real layer-0 weights, applied to the hyper-connection state the model builds
      from the `embed_tokens` rows (`hc_count` copies, `qwen4_exp.py:1288`). Token IDs: every prompt of
      `bench/prompts/parity_v1.jsonl` rendered with the model's chat template (`add_generation_prompt`,
      `enable_thinking=False`), concatenated in file order and cycled to 2,048 rows;
  (ii) layers 23/47: synthetic BF16 (seeded) with each row's RMS matched to the same row of (i).

D6 isolation rows (tasks/m22d-spec.md D6/C7, phase `isolate`; report-only: nothing is asserted beyond running), on
every layer 0-47 with its real weights, 4 decode tokens per layer (each its own M = 1 call; layer 0: the first 4 rows
of (i), else RMS-matched seeded rows as in (ii)), aggregated per layer:
  * `isolate_L<layer>_router_topk_M1`: `cere_ktest_moe_router` vs `fused_topk` on the same BF16 logits; differing
    ids (slot order) and differing id sets;
  * `isolate_L<layer>_router_weights_M1`: the renormalized weights of the same call (floor FLOOR_OP);
  * `isolate_L<layer>_experts_combine_M1`: routed experts and the weighted combine alone (the shared expert zeroed on
    both sides) vs `cutlass_fused_moe` as above (floor FLOOR_BLOCK). The control here runs FlashInfer's default
    tactics at `tune_max_num_tokens` 1, not the served server's autotuned ones;
  * `isolate_L<layer>_shared_expert_gated_M1`: the sigmoid-gated shared expert alone (native: the block with every
    `g2_alphas` 0, so each routed term is exactly 0) vs SGLang's shared GEMMs, `silu_and_mul` and
    `fused_gate_sigmoid_mul_add` onto zeros (floor FLOOR_OP);
  * `isolate_L<layer>_block_M1`: the whole block (floor FLOOR_BLOCK).
Each tolerance row reports `c` = rel-L2(control, FP64 reference above), `limit` = max(floor, 2c), `measured` =
rel-L2(native, control), `within`, and the differing elements.

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_moe_router(const void *logits,        /* bfloat16 [m, num_experts] */
                            void *topk_ids,            /* int32 [m, top_k]; slot k = k-th largest logit,
                                                          ties broken by the lowest expert id */
                            void *topk_weights,        /* float32 [m, top_k]; softmax probability over all
                                                          experts, renormalized over the selected top_k */
                            int64_t m, int64_t num_experts, int64_t top_k,
                            cudaStream_t stream);

  int cere_ktest_moe_block(const void *x,              /* bfloat16 [m, hidden] (the MLP input) */
                           const void *router_weight,  /* bfloat16 [num_experts, hidden] (mlp.gate.weight) */
                           const void *gate_weight,    /* uint8 [num_experts, inter, hidden/2], checkpoint bytes */
                           const void *gate_scale,     /* FP8 E4M3 [num_experts, inter, hidden/16], linear layout */
                           const void *up_weight,      /* uint8 [num_experts, inter, hidden/2] */
                           const void *up_scale,       /* FP8 E4M3 [num_experts, inter, hidden/16] */
                           const void *down_weight,    /* uint8 [num_experts, hidden, inter/2] */
                           const void *down_scale,     /* FP8 E4M3 [num_experts, hidden, inter/16] */
                           const void *g1_alphas,      /* float32 [num_experts]: max_w13_input_scale x
                                                          gate.weight_scale_2, for both the gate and the up GEMM */
                           const void *g2_alphas,      /* float32 [num_experts]: max_w2_input_scale x
                                                          down.weight_scale_2 */
                           float a1_global_scale,      /* FP32 1/max_w13_input_scale (GEMM1 activation quant) */
                           float a2_global_scale,      /* FP32 1/max_w2_input_scale (GEMM2 activation quant) */
                           const void *shared_gate_weight,  /* bfloat16 [shared_inter, hidden] */
                           const void *shared_up_weight,    /* bfloat16 [shared_inter, hidden] */
                           const void *shared_down_weight,  /* bfloat16 [hidden, shared_inter] */
                           const void *shared_expert_gate,  /* bfloat16 [hidden] (mlp.shared_expert_gate.weight) */
                           void *out,                  /* bfloat16 [m, hidden] */
                           int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
                           int64_t inter, int64_t shared_inter,
                           cudaStream_t stream);
      out = sum_k w_k * down_e(quant_a2(silu(gate_e(quant_a1(x))) * up_e(quant_a1(x))))
            + sigmoid(x . shared_expert_gate) * shared_down(silu(shared_gate(x)) * shared_up(x)),
      with the router of `cere_ktest_moe_router` on the BF16 logits x @ router_weight^T and the activation
      quantizer of `cere_ktest_fp4_quantize` (K0).

  int cere_ktest_moe_block_rows(const void *x, ... the cere_ktest_moe_block arguments up to `out` ...,
                                void *routed,         /* nullable bfloat16 [m, hidden]: MoeTaps.routed (the
                                                         block's tap path) */
                                int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
                                int64_t inter, int64_t shared_inter,
                                int64_t rows,         /* 0: cere_ktest_moe_block's plans; [1, 16] with m <= rows:
                                                         the executor's rows MoePlans (GemmPlan::init_rows) */
                                cudaStream_t stream);

M2.6 K2 (tasks/m26-spec.md D6, `test_moe_rows_invariant`): on layers 0, 23 and 47, moe_block with the rows MoePlans
at m = ROWS_R is bit-equal per row to the same row alone (m = 1 on the single-row plans), output and `routed` tap,
for rows sharing experts, rows with pairwise disjoint expert sets and the same token on every row. The row
`moe rows path` reports `moe_path` = `fused` when every case is bit-equal, else `per_row` (the D6 fallback
`kMoeRowsPerRow` in model/forward.cpp is then required; the test still fails).

P11R-23 (tasks/m23-spec.md, the shared-expert fork): `test_shared_fork_bit_equal` asserts that the block forking its
shared expert onto the side stream (model::MoeFork) gives the serial block's output and routed tap bit for bit at
every m in [1, 16], on the single-row plans pinned at m and on the rows MoePlans at 16, layers 0, 23, 47;
`test_shared_fork_bound` that the fork is taken at m = 16 and refused (CERE_ERR_ARGUMENT) at m = 17:

  int cere_ktest_moe_block_fork(... the cere_ktest_moe_block_rows arguments up to `rows` ...,
                                int64_t fork,         /* 1: the plans carry a fork the block must take, else
                                                         CERE_ERR_ARGUMENT; 0: cere_ktest_moe_block_rows */
                                cudaStream_t stream);

BeatSglang S4 D7 (tasks/beat-sglang-spec.md, `test_router_equals_block_router`): `cere_ktest_moe_router` (k7_moe.cu
router_kernel: the block softmax, then warp 0 picks the top k from per-lane sorted 32-bit keys) is bit-equal in ids
and weights to the 128-thread block-argmax router it replaced (`cere_ktest_moe_router_block`, variant 0, kept verbatim
in ktest/k7_router_ktest.cu): on every MoE layer's real router at ROUTER_M rows, on ROUTER_TIE_ROWS synthetic rows of
tied, signed-zero, NaN and partly -inf logits, and on ROUTER_SHAPES synthetic (num_experts, top_k) shapes (both
template widths, top_k up to 32). Fail-first (`test_router_equals_block_router_detects_sum_order`): variant 1 (the
block sum adds its 4 warp partials in reverse) is flagged in the weights at ROUTER_PERTURB_M rows of layer 0; variant
0 in nothing.

  int cere_ktest_moe_router_block(const void *logits, void *topk_ids, void *topk_weights, int64_t m,
                                  int64_t num_experts, int64_t top_k, int64_t variant /* 0 verbatim, 1 perturbed */,
                                  cudaStream_t stream);
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from safetensors import safe_open
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.srt.environ import envs
from sglang.srt.layers.activation import silu_and_mul
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import get_activation_type
from sglang.srt.layers.moe.topk import fused_topk
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.utils.common import next_power_of_2
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

import oracle

ROW = "K7"
LAYERS = (0, 23, 47)
M_CASES = (1, 64, 2048)
M_MAX = 2048
SYNTHETIC_SEED = 7_000
PARITY_PROMPTS = Path(__file__).resolve().parents[3] / "bench" / "prompts" / "parity_v1.jsonl"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# D6 isolation (m22d design note C7): decode tokens per layer, each its own M = 1 call; the synthetic input seed.
ISOLATE_TOKENS = 4
ISOLATE_SEED = 7_500
# S4 D7 router equivalence: the M cases, the synthetic tied / signed-zero / NaN rows (seed), the synthetic
# (num_experts, top_k) shapes, the fail-first M.
ROUTER_M = (1, 2, 4, 8, 16, 64, 2048)
ROUTER_TIE_ROWS = 256
ROUTER_TIE_SEED = 7_700
ROUTER_SHAPES = ((512, 10), (512, 32), (100, 1), (513, 10), (1024, 32), (1024, 8), (33, 32))
ROUTER_PERTURB_M = 2048


# -- checkpoint access (one open per shard file) ------------------------------------------------------------------


@functools.cache
def _weight_map() -> dict:
    index = oracle.MODEL_DIR / "model.safetensors.index.json"
    if not index.is_file():
        raise oracle.OracleError(f"{index} is absent (the model directory must be mounted at {oracle.MODEL_DIR})")
    return json.loads(index.read_text())["weight_map"]


def _load_named(names: list[str]) -> dict[str, torch.Tensor]:
    """CPU tensors by safetensors name, each shard file opened once."""
    weight_map = _weight_map()
    by_file: dict[str, list[str]] = {}
    for name in names:
        if name not in weight_map:
            raise oracle.OracleError(f"tensor {name} is not in the checkpoint index")
        by_file.setdefault(weight_map[name], []).append(name)
    out: dict[str, torch.Tensor] = {}
    for file_name, file_names in by_file.items():
        with safe_open(str(oracle.MODEL_DIR / file_name), framework="pt", device="cpu") as f:
            for name in file_names:
                out[name] = f.get_tensor(name)
    return out


def _expert_name(layer: int, expert: int, proj: str, field: str) -> str:
    return f"{oracle.layer_prefix(layer)}mlp.experts.{expert}.{proj}.{field}"


# -- inputs (i) and (ii) ------------------------------------------------------------------------------------------


def _parity_token_ids() -> list[int]:
    """All parity prompts rendered with the chat template (enable_thinking false), concatenated in file order."""
    tokenizer = get_tokenizer(str(oracle.MODEL_DIR))
    ids: list[int] = []
    for line in PARITY_PROMPTS.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise oracle.OracleError(f"{PARITY_PROMPTS} produced no tokens")
    return ids


@functools.cache
def _layer0_input() -> torch.Tensor:
    """Input (i): bfloat16 [M_MAX, hidden], layer 0's MLP input function on the parity prompts' embedding rows."""
    cfg = oracle.text_config()
    hidden = cfg["hidden_size"]
    hc = cfg["hc_count"]
    eps = cfg["rms_norm_eps"]
    ids = _parity_token_ids()
    oracle.report_row(ROW, "input_i", prompt_tokens=len(ids), rows=M_MAX)
    rows = torch.tensor([ids[r % len(ids)] for r in range(M_MAX)], dtype=torch.int64, device=oracle.DEVICE)
    embed = oracle.real_tensor("model.language_model.embed_tokens.weight")
    h = embed[rows].contiguous()
    del embed
    prefix = oracle.layer_prefix(0) + "mlp_hyper_connection."
    norm_w = oracle.real_tensor(prefix + "hc_norm.weight").float()
    down_w = oracle.real_tensor(prefix + "input_mix_weight_down.weight").float()
    up_w = oracle.real_tensor(prefix + "input_mix_weight_up.weight").float()
    hyper = torch.cat([h for _ in range(hc)], dim=-1).float().reshape(M_MAX, hc, hidden)
    normed = hyper * torch.rsqrt(hyper.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = (normed.flatten(-2) * (1.0 + norm_w)).to(torch.bfloat16).float()
    mix = torch.sigmoid(F.linear(F.silu(F.linear(normed, down_w) / hc), up_w))
    x = (mix.unflatten(-1, (hc, hidden)) * normed.unflatten(-1, (hc, hidden))).mean(dim=-2)
    return x.to(torch.bfloat16).contiguous()


def _row_rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()


@functools.cache
def _moe_input(layer: int) -> torch.Tensor:
    """K7's input of `layer`: (i) for layer 0, (ii) synthetic RMS-matched BF16 otherwise; [M_MAX, hidden]."""
    x0 = _layer0_input()
    if layer == 0:
        return x0
    z = torch.randn(x0.shape, generator=oracle.generator(SYNTHETIC_SEED + layer), device=oracle.DEVICE,
                    dtype=torch.float32)
    z = z / _row_rms(z) * _row_rms(x0)
    return z.to(torch.bfloat16).contiguous()


def _input_label(layer: int) -> str:
    return "i" if layer == 0 else "ii"


# -- one MoE layer's weights --------------------------------------------------------------------------------------


def _router_weight(layer: int) -> torch.Tensor:
    return oracle.real_tensor(oracle.layer_prefix(layer) + "mlp.gate.weight").contiguous()


def _load_moe_layer(layer: int) -> dict[str, torch.Tensor]:
    """Every tensor of one MoE block on GPU0: routed experts stacked per projection in the checkpoint layout,
    per-expert F32 scalars stacked to [num_experts], the router and the BF16 shared expert."""
    num_experts = oracle.text_config()["num_experts"]
    projs = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    fields = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    names = [_expert_name(layer, e, p, f) for p in projs.values() for f in fields for e in range(num_experts)]
    loaded = _load_named(names)
    w: dict[str, torch.Tensor] = {}
    for short, proj in projs.items():
        for field in fields:
            parts = [loaded.pop(_expert_name(layer, e, proj, field)) for e in range(num_experts)]
            if field in ("weight_scale_2", "input_scale"):
                stacked = torch.stack([p.reshape(()).to(torch.float32) for p in parts])
            else:
                # FP8 scales are stacked as raw bytes and viewed back (byte-identical).
                stacked = torch.stack([p.view(torch.uint8) for p in parts]).view(parts[0].dtype)
            w[f"{short}_{field}"] = stacked.to(oracle.DEVICE).contiguous()
    prefix = oracle.layer_prefix(layer) + "mlp."
    w["router"] = _router_weight(layer)
    w["shared_gate"] = oracle.real_tensor(prefix + "shared_expert.gate_proj.weight").contiguous()
    w["shared_up"] = oracle.real_tensor(prefix + "shared_expert.up_proj.weight").contiguous()
    w["shared_down"] = oracle.real_tensor(prefix + "shared_expert.down_proj.weight").contiguous()
    w["shared_expert_gate"] = oracle.real_tensor(prefix + "shared_expert_gate.weight").reshape(-1).contiguous()
    return w


def _reference_scales(w: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The flashinfer_cutlass scale set of `process_weights_after_loading` (`modelopt_quant.py:2424-2483`)."""
    w13_input_scale = torch.stack([w["gate_input_scale"], w["up_input_scale"]], dim=1)  # [E, 2]: col 0 gate
    w13_max = w13_input_scale.max().to(torch.float32)
    w2_max = w["down_input_scale"].max().to(torch.float32)
    w13_weight_scale_2 = torch.stack([w["gate_weight_scale_2"], w["up_weight_scale_2"]], dim=1)
    return {
        "w13_max": w13_max,
        "w2_max": w2_max,
        "g1_alphas": (w13_max * w13_weight_scale_2[:, 0]).to(torch.float32).contiguous(),
        "g2_alphas": (w2_max * w["down_weight_scale_2"]).to(torch.float32).contiguous(),
        "w13_input_scale_quant": (1 / w13_max).to(torch.float32),
        "w2_input_scale_quant": (1 / w2_max).to(torch.float32),
    }


def _cutlass_operands(w: dict[str, torch.Tensor], s: dict[str, torch.Tensor]) -> dict:
    """w13 ([up; gate] per expert) and the swizzled block scales as the flashinfer_cutlass path holds them."""
    w13_weight = torch.cat([w["up_weight"], w["gate_weight"]], dim=1).contiguous()
    w13_weight_scale = torch.cat(
        [w["up_weight_scale"].view(torch.uint8), w["gate_weight_scale"].view(torch.uint8)], dim=1
    ).view(torch.float8_e4m3fn)
    w13_blockscale_swizzled = swizzle_blockscale(w13_weight_scale.cpu())
    w2_blockscale_swizzled = swizzle_blockscale(w["down_weight_scale"].cpu())
    if w13_blockscale_swizzled.size(1) != w13_weight.size(1):
        raise oracle.OracleError("w13 block scales need intermediate padding; the reference path asserts it does not")
    return {
        "w13_weight": w13_weight.view(torch.long),
        "w2_weight": w["down_weight"].view(torch.long),
        "quant_scales": [
            s["w13_input_scale_quant"],
            w13_blockscale_swizzled.view(torch.int32),
            s["g1_alphas"],
            s["w2_input_scale_quant"],
            w2_blockscale_swizzled.view(torch.int32),
            s["g2_alphas"],
        ],
    }


# -- the reference op (SGLang) and the calibration reference ------------------------------------------------------


def _sglang_route(x: torch.Tensor, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    topk_weights, topk_ids = fused_topk(hidden_states=x, gating_output=logits, topk=top_k, renormalize=True)
    return topk_weights, topk_ids


def _sglang_routed(x: torch.Tensor, w: dict[str, torch.Tensor], ops: dict, top_k: int):
    """SGLang's routing and routed-expert output (`cutlass_fused_moe`, finalize included; no shared expert)."""
    m = x.shape[0]
    logits = F.linear(x, w["router"])
    topk_weights, topk_ids = _sglang_route(x, logits, top_k)
    output = torch.empty(m, x.shape[1], dtype=torch.bfloat16, device=x.device)
    final = cutlass_fused_moe(
        output=output,
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=ops["w13_weight"],
        fc2_expert_weights=ops["w2_weight"],
        output_dtype=torch.bfloat16,
        input_sf=None,
        quant_scales=ops["quant_scales"],
        ep_size=1,
        ep_rank=0,
        tp_size=1,
        tp_rank=0,
        tune_max_num_tokens=next_power_of_2(m),
        activation_type=ActivationType(get_activation_type("silu", is_gated=True)),
        enable_alltoall=False,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )[0]
    return final, topk_weights, topk_ids


def _sglang_shared_add(x: torch.Tensor, w: dict[str, torch.Tensor], final: torch.Tensor) -> None:
    """SGLang's shared expert, sigmoid-gated and added onto `final` in place (`qwen2_moe.py:624-631`)."""
    gate_up = F.linear(x, torch.cat([w["shared_gate"], w["shared_up"]], dim=0))
    shared_output = F.linear(silu_and_mul(gate_up), w["shared_down"]).contiguous()
    fused_gate_sigmoid_mul_add(x, w["shared_expert_gate"], shared_output, final)


def _sglang_block(x: torch.Tensor, w: dict[str, torch.Tensor], ops: dict, top_k: int):
    """SGLang's MoE block output (bfloat16 [m, hidden]) and its routing."""
    final, topk_weights, topk_ids = _sglang_routed(x, w, ops, top_k)
    _sglang_shared_add(x, w, final)
    return final, topk_weights, topk_ids


@functools.cache
def _e2m1_table() -> torch.Tensor:
    return torch.tensor(E2M1_VALUES, dtype=torch.float32, device=oracle.DEVICE)


def _dequant(codes: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """FP32 dequant of linear-layout NVFP4 (low nibble = even element): e2m1 * (fp8 scale * global scale)."""
    rows, half = codes.shape
    codes = codes.view(torch.uint8)
    nibbles = torch.stack([codes & 0x0F, codes >> 4], dim=-1).reshape(rows, 2 * half).long()
    values = _e2m1_table()[nibbles].reshape(rows, -1, 16)
    sf = scales.view(torch.float8_e4m3fn).reshape(rows, -1).float() * global_scale.to(torch.float32)
    return (values * sf.unsqueeze(-1)).reshape(rows, 2 * half)


def _quant_dequant(x: torch.Tensor, global_scale_quant: torch.Tensor, max_input_scale: torch.Tensor) -> torch.Tensor:
    codes, scales = fp4_quantize(
        x.contiguous(), global_scale_quant.reshape(1), 16, sf_use_ue8m0=False, is_sf_swizzled_layout=False
    )
    return _dequant(codes.reshape(x.shape[0], -1), scales.reshape(x.shape[0], -1), max_input_scale)


def _high_precision_block(x: torch.Tensor, w: dict[str, torch.Tensor], s: dict[str, torch.Tensor],
                          topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """FP64 accumulation over FP32-dequantized weights and the reference's quantized activations (float64)."""
    x_q = _quant_dequant(x, s["w13_input_scale_quant"], s["w13_max"]).double()
    routed = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for expert in topk_ids.unique().tolist():
        tokens, slots = (topk_ids == expert).nonzero(as_tuple=True)
        gate_ws2 = w["gate_weight_scale_2"][expert]
        wg = _dequant(w["gate_weight"][expert], w["gate_weight_scale"][expert], gate_ws2).double()
        wu = _dequant(w["up_weight"][expert], w["up_weight_scale"][expert], gate_ws2).double()
        wd = _dequant(w["down_weight"][expert], w["down_weight_scale"][expert],
                      w["down_weight_scale_2"][expert]).double()
        xe = x_q[tokens]
        act = F.silu(xe @ wg.T) * (xe @ wu.T)
        act_q = _quant_dequant(act.to(torch.bfloat16), s["w2_input_scale_quant"], s["w2_max"]).double()
        y = act_q @ wd.T
        routed.index_add_(0, tokens, y * topk_weights[tokens, slots].double().unsqueeze(-1))
    xd = x.double()
    shared = (F.silu(xd @ w["shared_gate"].double().T) * (xd @ w["shared_up"].double().T)) @ w["shared_down"].double().T
    gate = torch.sigmoid(xd @ w["shared_expert_gate"].double())
    return routed + gate.unsqueeze(-1) * shared


def _native_block(x: torch.Tensor, w: dict[str, torch.Tensor], s: dict[str, torch.Tensor], top_k: int) -> torch.Tensor:
    m, hidden = x.shape
    num_experts, inter, _ = w["gate_weight"].shape
    shared_inter = w["shared_gate"].shape[0]
    out = oracle.empty((m, hidden), torch.bfloat16)
    oracle.call(
        "cere_ktest_moe_block",
        x, w["router"],
        w["gate_weight"], w["gate_weight_scale"],
        w["up_weight"], w["up_weight_scale"],
        w["down_weight"], w["down_weight_scale"],
        s["g1_alphas"], s["g2_alphas"],
        float(s["w13_input_scale_quant"].item()), float(s["w2_input_scale_quant"].item()),
        w["shared_gate"], w["shared_up"], w["shared_down"], w["shared_expert_gate"],
        out,
        m, hidden, num_experts, top_k, inter, shared_inter,
    )
    return out


ROWS_R = 4  # M2.6 rows R (m26 design note D3: plan max_sessions <= 4)
VERIFY_ROWS = 16  # M2.9 rows plans with drafts: min(16, 4 x max_sessions) (m29 design note M29S2-05)


def _native_block_rows(x: torch.Tensor, w: dict[str, torch.Tensor], s: dict[str, torch.Tensor], top_k: int,
                       rows: int, routed: torch.Tensor | None) -> torch.Tensor:
    """`cere_ktest_moe_block_rows` on x [m, hidden] (rows 0: the single-row plans; else the rows MoePlans)."""
    m, hidden = x.shape
    num_experts, inter, _ = w["gate_weight"].shape
    shared_inter = w["shared_gate"].shape[0]
    out = oracle.empty((m, hidden), torch.bfloat16)
    oracle.call(
        "cere_ktest_moe_block_rows",
        x, w["router"],
        w["gate_weight"], w["gate_weight_scale"],
        w["up_weight"], w["up_weight_scale"],
        w["down_weight"], w["down_weight_scale"],
        s["g1_alphas"], s["g2_alphas"],
        float(s["w13_input_scale_quant"].item()), float(s["w2_input_scale_quant"].item()),
        w["shared_gate"], w["shared_up"], w["shared_down"], w["shared_expert_gate"],
        out, routed,
        m, hidden, num_experts, top_k, inter, shared_inter, rows,
    )
    return out


FORK_ROWS = 16  # P11R-23: the fork's bound (k2s::kMaxM, the rows MoePlans' VERIFY_ROWS)


def _native_block_fork(x: torch.Tensor, w: dict[str, torch.Tensor], s: dict[str, torch.Tensor], top_k: int,
                       rows: int, fork: int, routed: torch.Tensor | None, status_only: bool = False):
    """`cere_ktest_moe_block_fork` on x [m, hidden]: the output (fork 1: the block must fork), or with
    `status_only` the call's status."""
    m, hidden = x.shape
    num_experts, inter, _ = w["gate_weight"].shape
    shared_inter = w["shared_gate"].shape[0]
    out = oracle.empty((m, hidden), torch.bfloat16)
    args = (x, w["router"], w["gate_weight"], w["gate_weight_scale"], w["up_weight"], w["up_weight_scale"],
            w["down_weight"], w["down_weight_scale"], s["g1_alphas"], s["g2_alphas"],
            float(s["w13_input_scale_quant"].item()), float(s["w2_input_scale_quant"].item()),
            w["shared_gate"], w["shared_up"], w["shared_down"], w["shared_expert_gate"], out, routed,
            m, hidden, num_experts, top_k, inter, shared_inter, rows, fork)
    if status_only:
        return oracle.call_status("cere_ktest_moe_block_fork", *args)
    oracle.call("cere_ktest_moe_block_fork", *args)
    return out


def _rows_cases(layer: int, router: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
    """K2's [ROWS_R, hidden] inputs of `layer`, picked from K7's input rows by their top-k expert sets: `shared`
    (every pair of rows shares an expert), `disjoint` (pairwise disjoint expert sets), `same` (one row repeated)."""
    x_all = _moe_input(layer)
    sets = [set(r) for r in F.linear(x_all, router).float().topk(top_k, dim=-1).indices.tolist()]

    def pick(compatible) -> list[int]:
        chosen: list[int] = []
        for i in range(len(sets)):
            if all(compatible(sets[i], sets[j]) for j in chosen):
                chosen.append(i)
                if len(chosen) == ROWS_R:
                    return chosen
        raise oracle.OracleError(f"layer {layer}: fewer than {ROWS_R} rows with the requested expert overlap")

    shared = pick(lambda a, b: bool(a & b))
    disjoint = pick(lambda a, b: not (a & b))
    return {"shared": x_all[shared].contiguous(), "disjoint": x_all[disjoint].contiguous(),
            "same": x_all[:1].repeat(ROWS_R, 1).contiguous()}


def _isolate_input(layer: int) -> torch.Tensor:
    """ISOLATE_TOKENS decode rows of `layer`: the first rows of input (i) for layer 0, otherwise seeded BF16 with each
    row's RMS matched to the same row of (i) (input (ii)'s definition, a separate seed)."""
    x0 = _layer0_input()[:ISOLATE_TOKENS]
    if layer == 0:
        return x0
    z = torch.randn(x0.shape, generator=oracle.generator(ISOLATE_SEED + layer), device=oracle.DEVICE,
                    dtype=torch.float32)
    return (z / _row_rms(z) * _row_rms(x0)).to(torch.bfloat16).contiguous()


def _isolation_row(layer: int, op: str, native: torch.Tensor, control: torch.Tensor, high_precision: torch.Tensor,
                   control_kernel: str, floor: float) -> None:
    """One report-only D6 isolation row: `c` = rel-L2(control, FP64), `limit` = max(floor, 2c), `measured` =
    rel-L2(native, control) and the differing elements on identical inputs; `within` = measured <= limit. Asserts
    nothing (tasks/m22d-spec.md C7)."""
    c = oracle.rel_l2(control, high_precision)
    limit = oracle.calibrated_limit(c, floor)
    measured = oracle.rel_l2(native, control)
    oracle.report_row(ROW, f"isolate_L{layer}_{op}_M1", phase="isolate", layer=layer, op=op, m=1,
                      n_tokens=ISOLATE_TOKENS, input=_input_label(layer), control=control_kernel, c=c, limit=limit,
                      measured=measured, within=measured <= limit, elements=native.numel(),
                      differing=int((native != control).sum().item()))


# -- tests --------------------------------------------------------------------------------------------------------


class MoeTest(unittest.TestCase):
    def test_router_bit_exact_on_equal_logits(self) -> None:
        """Router indices bit-exact vs SGLang's softmax top-10 renormalize on the same logits; weights calibrated."""
        cfg = oracle.text_config()
        num_experts = cfg["num_experts"]
        top_k = cfg["num_experts_per_tok"]
        for layer in LAYERS:
            router = _router_weight(layer)
            x_all = _moe_input(layer)
            for m in M_CASES:
                case = f"router L{layer} ({_input_label(layer)}) M{m}"
                with self.subTest(case=case):
                    x = x_all[:m].contiguous()
                    logits = F.linear(x, router).contiguous()
                    ref_weights, ref_ids = _sglang_route(x, logits, top_k)
                    ref_ids = ref_ids.to(torch.int32).contiguous()
                    probs = torch.softmax(logits.double(), dim=-1).gather(1, ref_ids.long())
                    high_precision = probs / probs.sum(dim=-1, keepdim=True)
                    c = oracle.calibrate(ROW, case + " weights", ref_weights, high_precision, oracle.FLOOR_OP)
                    ids = oracle.empty((m, top_k), torch.int32)
                    weights = oracle.empty((m, top_k), torch.float32)
                    oracle.call("cere_ktest_moe_router", logits, ids, weights, m, num_experts, top_k)
                    oracle.check_bit_exact(self, ROW, case + " ids", ids, ref_ids)
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                    oracle.check_within(self, ROW, case + " weights", oracle.rel_l2(weights, ref_weights),
                                        c=c, limit=limit)

    def test_router_non_finite_rows(self) -> None:
        """Hardening (KGM-1): a row with NaN / Inf logits still gets K distinct expert ids in [0, E) (NaN ranks last)
        and finite weights; the other rows of the batch are bit-equal to the same rows routed alone."""
        cfg = oracle.text_config()
        num_experts = cfg["num_experts"]
        top_k = cfg["num_experts_per_tok"]
        gen = oracle.generator(20260515)
        normal = torch.randn((2, num_experts), generator=gen, device=oracle.DEVICE).to(torch.bfloat16).cpu()
        nan = float("nan")
        inf = float("inf")
        rows = torch.empty((6, num_experts), dtype=torch.bfloat16)
        rows[0] = normal[0]
        rows[1] = nan  # all NaN
        rows[2] = nan  # one number among NaNs
        rows[2, 7] = 5.0
        rows[3] = -inf  # all -inf
        rows[4] = normal[1]  # a +Inf logit among numbers
        rows[4, 3] = inf
        rows[5] = normal[1]
        logits = rows.to(oracle.DEVICE).contiguous()
        ids = oracle.empty((6, top_k), torch.int32)
        weights = oracle.empty((6, top_k), torch.float32)
        oracle.call("cere_ktest_moe_router", logits, ids, weights, 6, num_experts, top_k)
        ids, weights = ids.cpu(), weights.cpu()
        for r in range(6):
            with self.subTest(row=r):
                row_ids = ids[r].tolist()
                self.assertTrue(all(0 <= e < num_experts for e in row_ids), row_ids)
                self.assertEqual(len(set(row_ids)), top_k, row_ids)
                self.assertTrue(bool(torch.isfinite(weights[r]).all()), weights[r])
                self.assertTrue(bool((weights[r] >= 0).all()), weights[r])
        self.assertEqual(ids[2, 0].item(), 7)  # the only number leads, NaNs follow
        self.assertEqual(ids[4, 0].item(), 3)  # +Inf ranks first
        self.assertEqual(ids[3].tolist(), list(range(top_k)))  # all equal: lowest ids first
        for r in (1, 2, 3, 4):  # softmax denominator not finite: weights 0, never NaN
            self.assertTrue(bool((weights[r] == 0).all()), (r, weights[r]))
        for r in (0, 5):
            oracle.check_count(self, ROW, f"router_weight_sum_row{r}", int(abs(weights[r].sum().item() - 1.0) > 1e-5),
                               expected=0)
        alone = oracle.empty((2, top_k), torch.int32)
        alone_w = oracle.empty((2, top_k), torch.float32)
        oracle.call("cere_ktest_moe_router", normal.to(oracle.DEVICE).contiguous(), alone, alone_w, 2, num_experts,
                    top_k)
        alone, alone_w = alone.cpu(), alone_w.cpu()
        self.assertTrue(torch.equal(ids[0], alone[0]) and torch.equal(weights[0], alone_w[0]))
        self.assertTrue(torch.equal(ids[5], alone[1]) and torch.equal(weights[5], alone_w[1]))
        self.assertEqual(len(set(ids[0].tolist())), top_k)

    def _router_pair(self, logits: torch.Tensor, variant: int, num_experts: int | None = None,
                     top_k: int | None = None) -> tuple[torch.Tensor, ...]:
        """(ids, weights) of `cere_ktest_moe_router` and (ids, weights) of the block router `variant` on `logits`."""
        cfg = oracle.text_config()
        num_experts = cfg["num_experts"] if num_experts is None else num_experts
        top_k = cfg["num_experts_per_tok"] if top_k is None else top_k
        m = logits.shape[0]
        out = [oracle.empty((m, top_k), dt) for dt in (torch.int32, torch.float32, torch.int32, torch.float32)]
        oracle.call("cere_ktest_moe_router", logits, out[0], out[1], m, num_experts, top_k)
        oracle.call("cere_ktest_moe_router_block", logits, out[2], out[3], m, num_experts, top_k, variant)
        return tuple(out)

    def _tied_rows(self, num_experts: int, seed: int) -> torch.Tensor:
        """ROUTER_TIE_ROWS BF16 rows of tied and signed-zero logits; the third quarter has NaNs and an Inf, the last
        quarter all but 5 logits -inf (top-k past the finite ones re-picks -inf ids)."""
        g = oracle.generator(seed)
        values = torch.tensor([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], device=oracle.DEVICE)
        pick = torch.randint(0, len(values), (ROUTER_TIE_ROWS, num_experts), generator=g, device=oracle.DEVICE)
        tied = values[pick]
        half, quarter = ROUTER_TIE_ROWS // 2, ROUTER_TIE_ROWS * 3 // 4
        nan = torch.rand((quarter - half, num_experts), generator=g, device=oracle.DEVICE) < 0.25
        tied[half:quarter][nan] = float("nan")
        tied[half + 1, 0] = float("inf")
        tied[half + 2] = float("nan")
        tied[quarter:, 5:] = float("-inf")
        return tied.to(torch.bfloat16).contiguous()

    def test_router_equals_block_router(self) -> None:
        """S4 D7: the sorted-key router's ids and weights are bit-equal to the 128-thread block router it replaced,
        on every MoE layer's router at ROUTER_M rows, on tied / signed-zero / NaN / partly -inf synthetic logits and
        on the ROUTER_SHAPES synthetic shapes."""
        cfg = oracle.text_config()
        for layer in range(cfg["num_hidden_layers"]):
            router = _router_weight(layer)
            x_all = _moe_input(layer)
            for m in ROUTER_M:
                case = f"router sorted L{layer} M{m}"
                with self.subTest(case=case):
                    logits = F.linear(x_all[:m].contiguous(), router).contiguous()
                    ids, weights, ref_ids, ref_weights = self._router_pair(logits, 0)
                    oracle.check_bit_exact(self, ROW, case + " ids", ids, ref_ids)
                    oracle.check_bit_exact(self, ROW, case + " weights", weights, ref_weights)
        ids, weights, ref_ids, ref_weights = self._router_pair(self._tied_rows(cfg["num_experts"], ROUTER_TIE_SEED), 0)
        oracle.check_bit_exact(self, ROW, "router sorted tied ids", ids, ref_ids)
        oracle.check_bit_exact(self, ROW, "router sorted tied weights", weights, ref_weights)
        for i, (num_experts, top_k) in enumerate(ROUTER_SHAPES):
            g = oracle.generator(ROUTER_TIE_SEED + 1 + i)
            normal = torch.randn((ROUTER_TIE_ROWS, num_experts), generator=g, device=oracle.DEVICE)
            for label, logits in (("normal", normal.to(torch.bfloat16).contiguous()),
                                  ("tied", self._tied_rows(num_experts, ROUTER_TIE_SEED + 100 + i))):
                case = f"router sorted E{num_experts} K{top_k} {label}"
                with self.subTest(case=case):
                    ids, weights, ref_ids, ref_weights = self._router_pair(logits, 0, num_experts, top_k)
                    oracle.check_bit_exact(self, ROW, case + " ids", ids, ref_ids)
                    oracle.check_bit_exact(self, ROW, case + " weights", weights, ref_weights)

    def test_router_equals_block_router_detects_sum_order(self) -> None:
        """Fail-first for test_router_equals_block_router: the block router with its softmax sum adding the 4 warp
        partials in reverse (variant 1) is flagged in the weights of layer 0 at ROUTER_PERTURB_M rows; variant 0 is
        not."""
        logits = F.linear(_moe_input(0)[:ROUTER_PERTURB_M].contiguous(), _router_weight(0)).contiguous()
        ids, weights, ref_ids, ref_weights = self._router_pair(logits, 1)
        flagged = int((weights.view(torch.int32) != ref_weights.view(torch.int32)).any(dim=-1).sum().item())
        ids_equal = torch.equal(ids, ref_ids)
        _, weights0, _, ref_weights0 = self._router_pair(logits, 0)
        clean = int((weights0.view(torch.int32) != ref_weights0.view(torch.int32)).any(dim=-1).sum().item())
        oracle.report_row(ROW, f"router sorted sum order M{ROUTER_PERTURB_M}", phase="assert",
                          perturbed_rows_flagged=flagged, clean_rows_flagged=clean, ids_equal=ids_equal)
        self.assertNotEqual(flagged, 0, f"{ROW} router sum order: the reversed warp sum is not flagged")
        self.assertEqual(clean, 0, f"{ROW} router sum order: the verbatim block router flagged {clean} rows")

    def _check_block(self, layer: int) -> None:
        top_k = oracle.text_config()["num_experts_per_tok"]
        w = _load_moe_layer(layer)
        s = _reference_scales(w)
        ops = _cutlass_operands(w, s)
        x_all = _moe_input(layer)
        for m in M_CASES:
            case = f"block L{layer} ({_input_label(layer)}) M{m}"
            with self.subTest(case=case):
                x = x_all[:m].contiguous()
                reference, topk_weights, topk_ids = _sglang_block(x, w, ops, top_k)
                high_precision = _high_precision_block(x, w, s, topk_weights, topk_ids)
                c = oracle.calibrate(ROW, case, reference, high_precision, oracle.FLOOR_BLOCK)
                native = _native_block(x, w, s, top_k)
                limit = oracle.calibrated_limit(c, oracle.FLOOR_BLOCK)
                oracle.check_within(self, ROW, case, oracle.rel_l2(native, reference), c=c, limit=limit)

    def test_block_layer0_parity_inputs(self) -> None:
        self._check_block(0)

    def test_block_layer23_synthetic(self) -> None:
        self._check_block(23)

    def test_block_layer47_synthetic(self) -> None:
        self._check_block(47)

    # -- D6 isolation rows (m22d design note C7): decode M = 1 on every layer, report-only -----------------------

    def _isolate_layer(self, layer: int) -> None:
        cfg = oracle.text_config()
        num_experts = cfg["num_experts"]
        top_k = cfg["num_experts_per_tok"]
        w = _load_moe_layer(layer)
        s = _reference_scales(w)
        ops = _cutlass_operands(w, s)
        # Routed experts alone: the shared expert zeroed on both sides (its gated contribution is exactly 0).
        w_routed = dict(w, shared_gate=torch.zeros_like(w["shared_gate"]),
                        shared_up=torch.zeros_like(w["shared_up"]),
                        shared_down=torch.zeros_like(w["shared_down"]))
        # Shared expert alone on the native side: g2_alphas 0 scales every routed down GEMM to exactly 0.
        s_shared = dict(s, g2_alphas=torch.zeros_like(s["g2_alphas"]))
        x_all = _isolate_input(layer)
        parts: dict[str, list[torch.Tensor]] = {}

        def add(key: str, value: torch.Tensor) -> None:
            parts.setdefault(key, []).append(value)

        for t in range(ISOLATE_TOKENS):  # each token is its own M = 1 call, as a decode step runs it
            x = x_all[t:t + 1].contiguous()
            logits = F.linear(x, w["router"]).contiguous()
            ref_weights, ref_ids = _sglang_route(x, logits, top_k)
            ref_ids = ref_ids.to(torch.int32).contiguous()
            probs = torch.softmax(logits.double(), dim=-1).gather(1, ref_ids.long())
            ids = oracle.empty((1, top_k), torch.int32)
            weights = oracle.empty((1, top_k), torch.float32)
            oracle.call("cere_ktest_moe_router", logits, ids, weights, 1, num_experts, top_k)
            add("router_ids", ids)
            add("router_ids_ref", ref_ids)
            add("router_weights", (weights, ref_weights, probs / probs.sum(dim=-1, keepdim=True)))
            routed, topk_weights, topk_ids = _sglang_routed(x, w, ops, top_k)
            add("experts_combine", (_native_block(x, w_routed, s, top_k), routed.clone(),
                                    _high_precision_block(x, w_routed, s, topk_weights, topk_ids)))
            shared = torch.zeros_like(x)
            _sglang_shared_add(x, w, shared)
            xd = x.double()
            act = F.silu(xd @ w["shared_gate"].double().T) * (xd @ w["shared_up"].double().T)
            shared_hp = torch.sigmoid(xd @ w["shared_expert_gate"].double()).unsqueeze(-1) * (
                act @ w["shared_down"].double().T)
            add("shared_expert_gated", (_native_block(x, w, s_shared, top_k), shared, shared_hp))
            _sglang_shared_add(x, w, routed)  # the served block: routed + gated shared, in place
            add("block", (_native_block(x, w, s, top_k), routed,
                          _high_precision_block(x, w, s, topk_weights, topk_ids)))
        ids, ref_ids = torch.cat(parts["router_ids"]), torch.cat(parts["router_ids_ref"])
        oracle.report_row(ROW, f"isolate_L{layer}_router_topk_M1", phase="isolate", layer=layer, op="router_topk",
                          m=1, n_tokens=ISOLATE_TOKENS, input=_input_label(layer),
                          control="fused_topk (moe_fused_gate) on the same BF16 logits", elements=ids.numel(),
                          ids_differing=int((ids != ref_ids).sum().item()),
                          id_sets_differing=int((ids.sort(dim=-1).values != ref_ids.sort(dim=-1).values)
                                                .any(dim=-1).sum().item()))
        controls = {
            "router_weights": ("fused_topk renormalized weights", oracle.FLOOR_OP),
            "experts_combine": ("cutlass_fused_moe (tune_max_num_tokens 1, untuned tactics), shared expert zeroed",
                                oracle.FLOOR_BLOCK),
            "shared_expert_gated": ("F.linear(x, cat(gate, up)) + silu_and_mul + F.linear(down) + "
                                    "fused_gate_sigmoid_mul_add onto zeros, native g2_alphas 0", oracle.FLOOR_OP),
            "block": ("served block (_sglang_block): routed + gated shared", oracle.FLOOR_BLOCK),
        }
        for op, (control_kernel, floor) in controls.items():
            native, control, high_precision = (torch.cat(z) for z in zip(*parts[op]))
            _isolation_row(layer, op, native, control, high_precision, control_kernel, floor)

    def test_isolate_decode_every_layer(self) -> None:
        """D6 isolation (tasks/m22d-spec.md C7): report-only M = 1 rows on every MoE layer."""
        for layer in range(oracle.text_config()["num_hidden_layers"]):
            with self.subTest(layer=layer):
                self._isolate_layer(layer)
                torch.cuda.empty_cache()

    # -- M2.6 K2 (m26 design note D6): rows MoePlans ------------------------------------------------------------

    def test_moe_rows_invariant(self) -> None:
        """K2: moe_block at m = ROWS_R on the rows MoePlans equals each row alone (m = 1), output and routed tap."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        hidden = oracle.text_config()["hidden_size"]
        results: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for layer in LAYERS:
            w = _load_moe_layer(layer)
            s = _reference_scales(w)
            for label, x in _rows_cases(layer, w["router"], top_k).items():
                solo_out, solo_routed, solo_plain = [], [], []
                for r in range(ROWS_R):
                    routed = oracle.empty((1, hidden), torch.bfloat16)
                    solo_out.append(_native_block_rows(x[r:r + 1].contiguous(), w, s, top_k, 0, routed))
                    solo_routed.append(routed)
                    solo_plain.append(_native_block_rows(x[r:r + 1].contiguous(), w, s, top_k, 0, None))
                routed = oracle.empty((ROWS_R, hidden), torch.bfloat16)
                tapped = _native_block_rows(x, w, s, top_k, ROWS_R, routed)
                plain = _native_block_rows(x, w, s, top_k, ROWS_R, None)
                case = f"moe rows L{layer} {label} R{ROWS_R}"
                results.append((case + " out", plain, torch.cat(solo_plain)))
                results.append((case + " tapped out", tapped, torch.cat(solo_out)))
                results.append((case + " routed", routed, torch.cat(solo_routed)))
            del w, s
            torch.cuda.empty_cache()
        fused = all(bool(torch.equal(a.view(torch.int16), b.view(torch.int16))) for _, a, b in results)
        oracle.report_row(ROW, "moe rows path", phase="rows", rows=ROWS_R, moe_path="fused" if fused else "per_row")
        for case, native, solo in results:
            with self.subTest(case=case):
                oracle.check_bit_exact(self, ROW, case, native, solo)

    def test_moe_rows_invariant_m16(self) -> None:
        """M2.9 K5 (tasks/m29-spec.md M29S2-05): moe_block on the rows MoePlans built at VERIFY_ROWS = 16 (the verify
        call's rows), for every m in 2..16: row r of the m-row call equals row r alone (m = 1 on the single-row plans),
        output and routed tap. Inputs per layer: K7's first 16 input rows (mixed expert overlap) and its first row
        repeated 16 times. The row `moe rows16 path` reports `fused` when every case is bit-equal, else `per_row`."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        hidden = oracle.text_config()["hidden_size"]
        results: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for layer in LAYERS:
            w = _load_moe_layer(layer)
            s = _reference_scales(w)
            x_all = _moe_input(layer)
            cases = {"first": x_all[:VERIFY_ROWS].contiguous(),
                     "same": x_all[:1].repeat(VERIFY_ROWS, 1).contiguous()}
            for label, x in cases.items():
                solo_out, solo_routed, solo_plain = [], [], []
                for r in range(VERIFY_ROWS):
                    routed = oracle.empty((1, hidden), torch.bfloat16)
                    solo_out.append(_native_block_rows(x[r:r + 1].contiguous(), w, s, top_k, 0, routed))
                    solo_routed.append(routed)
                    solo_plain.append(_native_block_rows(x[r:r + 1].contiguous(), w, s, top_k, 0, None))
                for m in range(2, VERIFY_ROWS + 1):
                    xm = x[:m].contiguous()
                    routed = oracle.empty((m, hidden), torch.bfloat16)
                    tapped = _native_block_rows(xm, w, s, top_k, VERIFY_ROWS, routed)
                    plain = _native_block_rows(xm, w, s, top_k, VERIFY_ROWS, None)
                    case = f"moe rows16 L{layer} {label} M{m}"
                    results.append((case + " out", plain, torch.cat(solo_plain[:m])))
                    results.append((case + " tapped out", tapped, torch.cat(solo_out[:m])))
                    results.append((case + " routed", routed, torch.cat(solo_routed[:m])))
            del w, s, x_all
            torch.cuda.empty_cache()
        fused = all(bool(torch.equal(a.view(torch.int16), b.view(torch.int16))) for _, a, b in results)
        oracle.report_row(ROW, "moe rows16 path", phase="rows", rows=VERIFY_ROWS,
                          moe_path="fused" if fused else "per_row")
        for case, native, solo in results:
            with self.subTest(case=case):
                oracle.check_bit_exact(self, ROW, case, native, solo)

    # -- P11R-23 (m23 design note): the shared-expert fork -------------------------------------------------------

    def test_shared_fork_bit_equal(self) -> None:
        """P11R-23: the block that forks its shared expert onto the side stream gives the serial block's bits, output
        and routed tap, at every m in [1, FORK_ROWS] on the single-row plans pinned at m (rows 0) and on the rows
        MoePlans built at FORK_ROWS; K7's first input rows of layers 0, 23, 47."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        hidden = oracle.text_config()["hidden_size"]
        results: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for layer in LAYERS:
            w = _load_moe_layer(layer)
            s = _reference_scales(w)
            x_all = _moe_input(layer)
            for rows in (0, FORK_ROWS):
                for m in range(1, FORK_ROWS + 1):
                    x = x_all[:m].contiguous()
                    serial_routed = oracle.empty((m, hidden), torch.bfloat16)
                    forked_routed = oracle.empty((m, hidden), torch.bfloat16)
                    serial = _native_block_fork(x, w, s, top_k, rows, 0, serial_routed)
                    forked = _native_block_fork(x, w, s, top_k, rows, 1, forked_routed)
                    case = f"shared fork L{layer} R{rows} M{m}"
                    results.append((case + " out", forked, serial))
                    results.append((case + " routed", forked_routed, serial_routed))
            del w, s, x_all
            torch.cuda.empty_cache()
        equal = sum(bool(torch.equal(a.view(torch.int16), b.view(torch.int16))) for _, a, b in results)
        oracle.report_row(ROW, "shared fork", phase="fork", cases=len(results), bit_equal_cases=equal)
        for case, forked, serial in results:
            with self.subTest(case=case):
                oracle.check_bit_exact(self, ROW, case, forked, serial)

    def test_shared_fork_bound(self) -> None:
        """P11R-23: the fork is taken at m <= FORK_ROWS (status 0) and refused one row past it (CERE_ERR_ARGUMENT:
        kernels::moe_shared_forks is false, the block would run serially)."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        hidden = oracle.text_config()["hidden_size"]
        w = _load_moe_layer(0)
        s = _reference_scales(w)
        statuses = {}
        for m in (FORK_ROWS, FORK_ROWS + 1):
            statuses[m] = _native_block_fork(_moe_input(0)[:m].contiguous(), w, s, top_k, 0, 1, None,
                                             status_only=True)
        oracle.report_row(ROW, "shared fork bound", phase="fork", status_at_bound=statuses[FORK_ROWS],
                          status_past_bound=statuses[FORK_ROWS + 1], hidden=hidden)
        self.assertEqual(statuses, {FORK_ROWS: 0, FORK_ROWS + 1: oracle.CERE_ERR_ARGUMENT})


if __name__ == "__main__":
    unittest.main()
