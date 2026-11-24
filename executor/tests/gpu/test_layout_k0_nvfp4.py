"""K0 / X2: NVFP4 layout and scale semantics on the real checkpoint (tasks/m22-spec.md K0, X2, Q14).

Suite `layout` (run by X2 inside the pinned SGLang image on GPU0):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_layout_*.py' -v`.

Q14 items proven here:
  (a) native dequant of the raw checkpoint bytes equals FlashInfer
      `e2m1_and_ufp8sf_scale_to_float(..., is_sf_swizzled_layout=False)` bit-exactly for 18 tensors:
      layers 0/23/47 x gate/up/down x experts 0/511 (row K0). Each case also reports whether the native output
      equals the reference applied to nibble-swapped bytes, so a failed (a) with the swapped order passing is
      recorded as the semantics, never patched over;
  (b) native activation quantization equals `fp4_quantize(x, 1/max_input_scale, 16, is_sf_swizzled_layout=False)`
      bit-exactly (codes and scales) on layers 0/23/47 with K7's inputs, for M in {1, 64, 2048} and both per-layer
      global scales (w13 = max over all experts and the gate/up shards of `input_scale`; w2 = max over all experts
      of `down_proj.input_scale`; `modelopt_quant.py:2425-2427`, inverted in FP32 at `:2478-2483`) (row K0);
  (d) the number of experts (48 x 512) whose `up_proj.weight_scale_2 != gate_proj.weight_scale_2`, reported
      (row X2). The reference uses the gate's `weight_scale_2` for both GEMM1 halves (`g1_alphas`,
      `modelopt_quant.py:2112-2135`), so (d) also dequantizes one up_proj (an up != gate expert when one exists)
      with the gate's `weight_scale_2`, the scale the reference applies to it.
(c), the MoE layer, is K7 (`test_kernels_k7_moe.py`), checked by X3.

K7's inputs (shared definition with `test_kernels_k7_moe.py`):
  (i) layer 0: the MoE input of the parity prompts' tokens. The checkpoint has no `post_attention_layernorm`
      (`Qwen4ExpLayerExtensionMixin` deletes it, `sglang/srt/models/qwen4_exp.py:1235-1239`); its place before the MLP
      is `mlp_hyper_connection.mix` (`qwen4_exp.py:1318`): the per-branch Gemma RMSNorm `hc_norm` followed by the
      low-rank input mix (`sglang/srt/layers/hyperconnection.py` `GroupedGemmaRMSNorm.forward`, `_mix_compute`).
      Input (i) is that function, on real layer-0 weights, applied to the hyper-connection state the model builds
      from the `embed_tokens` rows (`hc_count` copies, `qwen4_exp.py:1288`). Token IDs: every prompt of
      `bench/prompts/parity_v1.jsonl` rendered with the model's chat template (`add_generation_prompt`,
      `enable_thinking=False`), concatenated in file order and cycled to 2,048 rows;
  (ii) layers 23/47: synthetic BF16 (seeded) with each row's RMS matched to the same row of (i).

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_nvfp4_dequant(const void *weight,      /* uint8 [n, k/2], checkpoint bytes, row-major;
                                                            element 2j = low nibble, 2j+1 = high nibble */
                               const void *scale,       /* FP8 E4M3 [n, k/16], row-major (linear layout); one
                                                            scale per 16 consecutive k elements */
                               float global_scale,      /* the tensor's F32 weight_scale_2 */
                               void *out,               /* float32 [n, k] */
                               int64_t n, int64_t k,
                               cudaStream_t stream);
      out[r][j] = e2m1(code(r, j)) * (float(scale[r][j/16]) * global_scale), in FP32 with exactly that product
      order (FlashInfer `E2M1AndUFP8SFScaleToFloatV2`, `thop/fp4Op.cpp:345-357`).

  int cere_ktest_fp4_quantize(const void *x,            /* bfloat16 [m, k], row-major */
                              float global_scale,       /* FP32 1/max_input_scale, the value fp4_quantize gets */
                              void *codes,              /* uint8 [m, k/2]; element 2j in the low nibble */
                              void *scales,             /* FP8 E4M3 [m, k/16], row-major (linear layout) */
                              int64_t m, int64_t k,
                              cudaStream_t stream);
      Bit-identical to FlashInfer `fp4_quantize(x, global_scale, 16, sf_use_ue8m0=False,
      is_sf_swizzled_layout=False)` (`flashinfer/quantization/fp4_quantization.py:899`).
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
import unittest

import torch
from flashinfer import e2m1_and_ufp8sf_scale_to_float, fp4_quantize
from safetensors import safe_open
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

import oracle

ROW = "K0"
ROW_X2 = "X2"
LAYERS = (0, 23, 47)
EXPERTS = (0, 511)
PROJS = ("gate_proj", "up_proj", "down_proj")
M_CASES = (1, 64, 2048)
M_MAX = 2048
SYNTHETIC_SEED = 7_000
PARITY_PROMPTS = Path(__file__).resolve().parents[3] / "bench" / "prompts" / "parity_v1.jsonl"


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


def _expert_scalars(layer: int, num_experts: int, proj: str, field: str) -> torch.Tensor:
    """float32 [num_experts] on GPU0: one per-tensor scalar (`input_scale` / `weight_scale_2`) of every expert."""
    names = [_expert_name(layer, e, proj, field) for e in range(num_experts)]
    loaded = _load_named(names)
    return torch.stack([loaded[n].reshape(()).to(torch.float32) for n in names]).to(oracle.DEVICE)


def _input_scale_quant(layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's FP32 activation global scales of one layer: (1/max w13 input_scale, 1/max w2 input_scale),
    each a float32 [1] tensor on GPU0 (`modelopt_quant.py:2425-2427,2478-2483`)."""
    num_experts = oracle.text_config()["num_experts"]
    w13 = torch.stack(
        [_expert_scalars(layer, num_experts, "gate_proj", "input_scale"),
         _expert_scalars(layer, num_experts, "up_proj", "input_scale")],
        dim=1,
    )
    w2 = _expert_scalars(layer, num_experts, "down_proj", "input_scale")
    w13_quant = (1 / w13.max().to(torch.float32)).to(torch.float32).reshape(1)
    w2_quant = (1 / w2.max().to(torch.float32)).to(torch.float32).reshape(1)
    return w13_quant, w2_quant


# -- K7's inputs --------------------------------------------------------------------------------------------------


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
    mix = torch.sigmoid(torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(normed, down_w) / hc), up_w))
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


# -- tests --------------------------------------------------------------------------------------------------------


def _reference_dequant(weight_u8: torch.Tensor, scale_u8: torch.Tensor, global_scale: float) -> torch.Tensor:
    """FlashInfer's linear-layout dequant (a host op over CPU tensors); float32 [n, k] moved to GPU0."""
    gs = torch.tensor([global_scale], dtype=torch.float32)
    out = e2m1_and_ufp8sf_scale_to_float(
        weight_u8.contiguous(), scale_u8.contiguous(), gs, 16, 1, is_sf_swizzled_layout=False
    )
    return out.to(oracle.DEVICE)


def _native_dequant(weight_u8: torch.Tensor, scale_u8: torch.Tensor, global_scale: float) -> torch.Tensor:
    n, k_half = weight_u8.shape
    k = 2 * k_half
    weight = weight_u8.to(oracle.DEVICE).contiguous()
    scale = scale_u8.to(oracle.DEVICE).contiguous()
    out = oracle.empty((n, k), torch.float32)
    oracle.call("cere_ktest_nvfp4_dequant", weight, scale, float(global_scale), out, n, k)
    return out


class NvFp4LayoutTest(unittest.TestCase):
    def test_a_weight_dequant_bit_exact(self) -> None:
        """Q14 (a): 18 real tensors, native dequant == FlashInfer linear-layout dequant, bit for bit."""
        cases = 0
        for layer in LAYERS:
            for proj in PROJS:
                for expert in EXPERTS:
                    case = f"a L{layer} {proj} e{expert}"
                    with self.subTest(case=case):
                        names = [_expert_name(layer, expert, proj, f)
                                 for f in ("weight", "weight_scale", "weight_scale_2")]
                        t = _load_named(names)
                        weight_u8 = t[names[0]]
                        scale_u8 = t[names[1]].view(torch.uint8)
                        ws2 = float(t[names[2]].reshape(()).to(torch.float32).item())
                        self.assertEqual(weight_u8.dtype, torch.uint8, f"{case}: weight dtype")
                        self.assertEqual(scale_u8.shape[1] * 16, weight_u8.shape[1] * 2, f"{case}: scale shape")
                        reference = _reference_dequant(weight_u8, scale_u8, ws2)
                        swapped_bytes = ((weight_u8 << 4) | (weight_u8 >> 4)).to(torch.uint8)
                        swapped_reference = _reference_dequant(swapped_bytes, scale_u8, ws2)
                        native = _native_dequant(weight_u8, scale_u8, ws2)
                        oracle.report_row(
                            ROW, case, phase="nibble_order",
                            equals_reference=bool(torch.equal(native.view(torch.int32), reference.view(torch.int32))),
                            equals_swapped_order=bool(
                                torch.equal(native.view(torch.int32), swapped_reference.view(torch.int32))),
                        )
                        oracle.check_bit_exact(self, ROW, case, native, reference)
                        cases += 1
        oracle.check_count(self, ROW, "a tensors", cases, expected=len(LAYERS) * len(PROJS) * len(EXPERTS))

    def test_b_activation_quantize_bit_exact(self) -> None:
        """Q14 (b): native FP4 quantization of K7's inputs == fp4_quantize(linear layout), codes and scales."""
        hidden = oracle.text_config()["hidden_size"]
        for layer in LAYERS:
            x_all = _moe_input(layer)
            w13_quant, w2_quant = _input_scale_quant(layer)
            for scale_name, global_scale in (("w13", w13_quant), ("w2", w2_quant)):
                for m in M_CASES:
                    case = f"b L{layer} {scale_name} M{m}"
                    with self.subTest(case=case):
                        x = x_all[:m].contiguous()
                        ref_codes, ref_scales = fp4_quantize(
                            x, global_scale, 16, sf_use_ue8m0=False, is_sf_swizzled_layout=False
                        )
                        ref_codes = ref_codes.view(torch.uint8).reshape(m, hidden // 2)
                        ref_scales = ref_scales.view(torch.uint8).reshape(m, hidden // 16)
                        codes = oracle.empty((m, hidden // 2), torch.uint8)
                        scales = oracle.empty((m, hidden // 16), torch.uint8)
                        oracle.call("cere_ktest_fp4_quantize", x, float(global_scale.item()), codes, scales,
                                    m, hidden)
                        oracle.check_bit_exact(self, ROW, case + " codes", codes, ref_codes)
                        oracle.check_bit_exact(self, ROW, case + " scales", scales, ref_scales)

    def test_d_weight_scale_2_up_vs_gate(self) -> None:
        """Q14 (d): count of experts with up.weight_scale_2 != gate.weight_scale_2 over all layers and experts."""
        cfg = oracle.text_config()
        num_layers = cfg["num_hidden_layers"]
        num_experts = cfg["num_experts"]
        differing = 0
        visited = 0
        layers_with_difference = 0
        first: tuple[int, int] | None = None
        for layer in range(num_layers):
            gate = _expert_scalars(layer, num_experts, "gate_proj", "weight_scale_2")
            up = _expert_scalars(layer, num_experts, "up_proj", "weight_scale_2")
            ne = gate != up
            count = int(ne.sum().item())
            differing += count
            visited += num_experts
            if count:
                layers_with_difference += 1
                if first is None:
                    first = (layer, int(ne.nonzero()[0].item()))
        oracle.report_row(ROW_X2, "d up_ne_gate_weight_scale_2", count=differing, experts=visited,
                          layers_with_difference=layers_with_difference)
        oracle.check_count(self, ROW_X2, "d experts visited", visited, expected=num_layers * num_experts)

        layer, expert = first if first is not None else (0, 0)
        case = f"d L{layer} up_proj e{expert} gate_weight_scale_2"
        names = [
            _expert_name(layer, expert, "up_proj", "weight"),
            _expert_name(layer, expert, "up_proj", "weight_scale"),
            _expert_name(layer, expert, "gate_proj", "weight_scale_2"),
        ]
        t = _load_named(names)
        weight_u8 = t[names[0]]
        scale_u8 = t[names[1]].view(torch.uint8)
        gate_ws2 = float(t[names[2]].reshape(()).to(torch.float32).item())
        reference = _reference_dequant(weight_u8, scale_u8, gate_ws2)
        native = _native_dequant(weight_u8, scale_u8, gate_ws2)
        oracle.check_bit_exact(self, ROW_X2, case, native, reference)


if __name__ == "__main__":
    unittest.main()
