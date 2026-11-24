"""K5 (tasks/m22n-spec.md C6, D6): the native chunked GDN prefill (`k5::delta_rule_chunk`) against the kernel the
no-speculation control dispatches for an extend: FLA's `chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`
called as SGLang's Triton GDN backend calls it (`TritonGDNKernel.extend`: one request in pool slot 0, `cu_seqlens`,
scale None = 128^-0.5), with g, beta from `fused_gdn_gating` on the same a, b.

Container only (pinned SGLang image, GPU0, `/out/libcere_ktest.so`, `/model` read-only); suite `kernels_k5`.

Inputs are real activations of each GDN layer of LAYERS (tasks/m22d-spec.md D5: first, second, middle and last
`linear_attention` layer, each with its own weights; `oracle.gdn_layer_weights` refuses any other id; case names
prefixed `layer<id>/`): RMS-normalised embedding rows of seeded random token ids (the same seeds on every layer)
through the layer's in_proj_qkv / in_proj_b / in_proj_a, SGLang's prefill conv (`causal_conv1d_fn`, Triton route) and
`fused_qkv_split_gdn_prefill`. Token ids never leave the process. Prompt lengths L in {1, 16, 17, 21, 24, 27, 32, 33,
48, 64, 65, 128, 2048, 4096}, each with a fresh (zero) initial state and with the state a previous 64-token SGLang
prefill left behind (carried).

Per case: `c` = rel-L2(SGLang, FP64 token recurrence written here) is written first (calibration rows, o and final
state); then the native entry point runs and rel-L2(native, SGLang) of o and of the final state is checked against the
pre-registered port limit (D14) 3e-4 for L <= 64 and 5e-4 above (the layer-0 limits, unchanged on every layer).
Report-only rows: the fraction of elements that are bit-equal to SGLang's, and the Triton autotune choice of FLA's
only autotuned kernel in this path (`chunk_gated_delta_rule_fwd_kkt_solve_kernel`, configs BK x num_warps,
chunk_fwd.py:24-37). Two identical native calls must be bit-identical (o and state), per layer.

Native entry point (every buffer a contiguous row-major device pointer, `int64_t` a count, `float` a scalar, the
stream last, the return value 0 or a nonzero `cere_status`):

    int cere_ktest_gdn_chunk_prefill(const void *q, const void *k, const void *v, const void *g, const void *beta,
                                     void *state, void *o, int64_t tokens, float scale, cudaStream_t stream);
        q, k bf16 [tokens, 16, 128] raw (L2-normalised inside, eps 1e-6); v bf16 [tokens, 48, 128]; g, beta f32
        [tokens, 48] (fused_gdn_gating's outputs); state bf16 [48, 128 (v), 128 (k)] in/out (SGLang's K-last layout;
        zeros = fresh sequence); o bf16 [tokens, 48, 128]; 1 <= tokens <= 4096. One sequence, chunk 64, FLA's
        rounding points (D6); the entry point allocates and frees its own workspace (`chunk_workspace_bytes`).

    int cere_ktest_gdn_chunk_prefill_fp32(const void *q, const void *k, const void *v, const void *g, const void *beta,
                                          void *state, void *o, int64_t tokens, float scale, const void *state_in32,
                                          void *capture32, int64_t capture_chunk, cudaStream_t stream);
        m210 D15⁗′: the same call with the FP32 state (ChunkArgs): state_in32 f32 [48, 128, 128] (or null) is the
        state the recurrence starts from; capture32 f32 [48, 128, 128] (or null) receives the FP32 state at the start
        of chunk capture_chunk (0 .. ceil(tokens / 64); the last is the final state before its BF16 rounding).
        `test_fp32_resume_bit_exact`: a call from row 64 c started from the capture at chunk c of a longer call
        continues it bit for bit (o rows and final state), and the capture changes nothing.
"""
from __future__ import annotations

import functools
import hashlib
import types
import unittest

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.fla import chunk_fwd
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn

import oracle

torch.set_grad_enabled(False)

ROW = "K5"
CASE = "delta_rule_chunk"
LAYERS = (0, 1, 18, 46)  # D5 (m22d design note): first, second, middle and last linear_attention layer
HIDDEN = 2560
HK = 16  # key (and query) heads
HV = 48  # value heads
DK = 128
DV = 128
CONV_K = 4
KEY_DIM = HK * DK
VALUE_DIM = HV * DV
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
RMS_EPS = 1e-6
L2_EPS = 1e-6  # FLA l2norm_fwd default (the in-kernel L2 norm of q and k)
SCALE = DK ** -0.5  # SGLang passes no scale: k.shape[-1] ** -0.5
CARRY_TOKENS = 64  # the previous prefill that leaves the carried state behind
BF16 = torch.bfloat16
EMBED = "model.language_model.embed_tokens.weight"


# -- model facts and weights ------------------------------------------------------------------------------------


@functools.cache
def _checked_config() -> None:
    cfg = oracle.text_config()
    expected = {"hidden_size": HIDDEN, "linear_num_key_heads": HK, "linear_num_value_heads": HV,
                "linear_key_head_dim": DK, "linear_value_head_dim": DV, "linear_conv_kernel_dim": CONV_K,
                "hidden_act": "silu", "rms_norm_eps": RMS_EPS}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise oracle.OracleError(f"text_config {key}={cfg.get(key)!r}, K5 is written for {value!r}")


def _expect(name: str, t: torch.Tensor, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    if tuple(t.shape) != shape or t.dtype != dtype:
        raise oracle.OracleError(f"{name}: {t.dtype} {tuple(t.shape)}, expected {dtype} {shape}")
    return t.contiguous()


@functools.lru_cache(maxsize=len(LAYERS))
def _weights(layer: int) -> types.SimpleNamespace:
    """The `linear_attn` weights of one GDN layer the inputs need, in the dtypes SGLang's Qwen3_5GatedDeltaNet holds
    them (`oracle.gdn_layer_weights` refuses a non-`linear_attention` id)."""
    _checked_config()
    raw = oracle.gdn_layer_weights(layer)
    p = oracle.layer_prefix(layer) + "linear_attn."

    def get(name, shape):
        return _expect(p + name, raw[name], shape, BF16)

    w = types.SimpleNamespace(layer=layer)
    w.qkv = get("in_proj_qkv.weight", (CONV_DIM, HIDDEN))
    w.b = get("in_proj_b.weight", (HV, HIDDEN))
    w.a = get("in_proj_a.weight", (HV, HIDDEN))
    w.conv = get("conv1d.weight", (CONV_DIM, 1, CONV_K)).reshape(CONV_DIM, CONV_K).contiguous()
    w.A_log = get("A_log", (HV,)).float().contiguous()  # SGLang's A_log parameter is float32
    w.dt_bias = get("dt_bias", (HV,))  # SGLang's dt_bias parameter is the model dtype (bf16)
    return w


@functools.cache
def _embedding_cpu() -> torch.Tensor:
    return oracle.real_tensor(EMBED, device=None)


def _hidden(seed: int, tokens: int) -> torch.Tensor:
    """bf16 [tokens, HIDDEN]: real embedding rows of seeded random token ids, RMS-normalised."""
    table = _embedding_cpu()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    ids = torch.randint(0, table.shape[0], (tokens,), generator=gen)
    x = table[ids].to(oracle.DEVICE, torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return x.to(BF16).contiguous()


def _slot() -> torch.Tensor:
    return torch.zeros(1, dtype=torch.int32, device=oracle.DEVICE)


def _cu(tokens: int) -> torch.Tensor:
    return torch.tensor([0, tokens], dtype=torch.int32, device=oracle.DEVICE)


# -- SGLang's extend path (one request in slot 0) -----------------------------------------------------------------


def _sglang_inputs(w: types.SimpleNamespace, hidden: torch.Tensor, conv_state: torch.Tensor, has_initial_state: bool):
    """q, k, v [1, T, H, D] bf16 and g, beta [1, T, HV] f32 exactly as `forward_extend` feeds the GDN kernel."""
    tokens = hidden.shape[0]
    mixed_qkv = F.linear(hidden, w.qkv)
    b = F.linear(hidden, w.b)
    a = F.linear(hidden, w.a)
    conv_out = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        w.conv,
        None,
        activation="silu",
        conv_states=conv_state,
        has_initial_state=torch.tensor([has_initial_state], device=oracle.DEVICE),
        cache_indices=_slot(),
        query_start_loc=_cu(tokens),
        seq_lens_cpu=[tokens],
    ).transpose(0, 1)[:tokens]
    q, k, v = fused_qkv_split_gdn_prefill(conv_out, HK, HK, HV, DK, DK, DV)
    g, beta = fused_gdn_gating(w.A_log, a, b, w.dt_bias)
    return q, k, v, g, beta


def _sglang_chunk(q, k, v, g, beta, ssm_pool: torch.Tensor) -> torch.Tensor:
    """`TritonGDNKernel.extend`: o [1, T, HV, DV]; the slot-0 state of `ssm_pool` [1, HV, DV, DK] is updated in place."""
    o, _, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=ssm_pool,
                                     initial_state_indices=_slot(), cu_seqlens=_cu(q.shape[1]), head_first=False,
                                     use_qk_l2norm_in_kernel=True)
    return o


def _kkt_autotune_choice() -> str:
    """The Triton autotune choice of FLA's fused kkt + solve kernel in this process (heuristics -> autotuner)."""
    tuner = chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel.fn
    return str(tuner.best_config)


# -- FP64 reference (calibration) -------------------------------------------------------------------------------


def _ref_l2norm64(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + L2_EPS)


def _ref_delta_rule64(q, k, v, g, beta, state):
    """q, k [T, HK, DK]; v [T, HV, DV]; g, beta [T, HV]; state [HV, DV, DK] -> (o [T, HV, DV], state), FP64 token
    recurrence with the in-kernel L2 norm of q and k (value head h uses key head h // (HV // HK))."""
    rep = HV // HK
    q = _ref_l2norm64(q).repeat_interleave(rep, dim=1) * SCALE
    k = _ref_l2norm64(k).repeat_interleave(rep, dim=1)
    v, g, beta = v.double(), g.double(), beta.double()
    s = state.double().transpose(1, 2).contiguous()  # [HV, DK, DV]
    decay = torch.exp(g)
    out = torch.empty(q.shape[0], HV, DV, dtype=torch.float64, device=q.device)
    for t in range(q.shape[0]):
        s.mul_(decay[t].view(HV, 1, 1))
        kv = torch.bmm(k[t].unsqueeze(1), s).squeeze(1)  # [HV, DV]
        u = (v[t] - kv) * beta[t].unsqueeze(-1)
        s.add_(k[t].unsqueeze(2) * u.unsqueeze(1))
        out[t] = torch.bmm(q[t].unsqueeze(1), s).squeeze(1)
    return out, s.transpose(1, 2).contiguous()


# -- native entry point -----------------------------------------------------------------------------------------


def _native_chunk(q, k, v, g, beta, state: torch.Tensor) -> torch.Tensor:
    """q, k [T, HK, DK]; v [T, HV, DV]; g, beta [T, HV]; state [HV, DV, DK] in/out -> o [T, HV, DV]."""
    tokens = q.shape[0]
    o = oracle.empty((tokens, HV, DV), BF16)
    oracle.call("cere_ktest_gdn_chunk_prefill", q, k, v, g, beta, state, o, tokens, SCALE)
    return o


def _sha256(t: torch.Tensor) -> str:
    """The hex SHA-256 of a tensor's bytes (contiguous, host copy)."""
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _report_bit_equal(case: str, native: torch.Tensor, reference: torch.Tensor) -> None:
    """Report-only: the fraction of elements whose bytes equal the reference's."""
    a = native.contiguous().view(torch.int16)
    b = reference.contiguous().view(torch.int16)
    elements = a.numel()
    equal = int((a == b).sum().item())
    oracle.report_row(ROW, case, phase="bit_equal", elements=elements, equal=equal, fraction=equal / elements)


# -- cases ------------------------------------------------------------------------------------------------------


def _inputs(w: types.SimpleNamespace, seed: int, tokens: int, carried: bool):
    """(q, k, v [T, H, D] bf16, g, beta [T, HV] f32, initial state [HV, DV, DK] bf16), all contiguous."""
    conv_state = torch.zeros(1, CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
    ssm_pool = torch.zeros(1, HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
    if carried:  # the conv and ssm state a previous 64-token prefill leaves behind
        _sglang_chunk(*_sglang_inputs(w, _hidden(seed + 1, CARRY_TOKENS), conv_state, False), ssm_pool)
    q, k, v, g, beta = _sglang_inputs(w, _hidden(seed, tokens), conv_state, carried)
    return q, k, v, g, beta, ssm_pool[0].clone()


class GdnChunkPrefill(unittest.TestCase):
    """`delta_rule_chunk` vs FLA's chunked gated delta rule on real activations of each layer of LAYERS (C6, D5)."""

    def _case(self, w: types.SimpleNamespace, tokens: int, seed: int, carried: bool):
        case = f"layer{w.layer}/{CASE}_L{tokens}_{'carried' if carried else 'fresh'}"
        q, k, v, g, beta, state0 = _inputs(w, seed, tokens, carried)
        ssm_pool = state0.clone().unsqueeze(0)
        sg_o = _sglang_chunk(q, k, v, g, beta, ssm_pool)[0]
        sg_state = ssm_pool[0]
        oracle.report_row(ROW, f"{case}/autotune", phase="report_only",
                          kkt_solve_config=_kkt_autotune_choice())
        ref_o, ref_state = _ref_delta_rule64(q[0], k[0], v[0], g[0], beta[0], state0)
        c_o = oracle.calibrate(ROW, f"{case}/o", sg_o, ref_o, oracle.FLOOR_BLOCK)
        c_s = oracle.calibrate(ROW, f"{case}/ssm_state", sg_state, ref_state, oracle.FLOOR_BLOCK)
        nat_state = state0.clone()
        nat_o = _native_chunk(q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), g[0].contiguous(),
                              beta[0].contiguous(), nat_state)
        _report_bit_equal(f"{case}/o", nat_o, sg_o)
        _report_bit_equal(f"{case}/ssm_state", nat_state, sg_state)
        if tokens <= 64:
            limit = oracle.port_limit(3e-4, "C6/D14: FLA chunk kernel is the control's prefill; one chunk, rounding "
                                            "points matched, FP32 sum order inside MMAs and Triton reductions only")
        else:
            limit = oracle.port_limit(5e-4, "C6/D14: FLA chunk kernel is the control's prefill; several chunks, "
                                            "rounding points matched, BF16 state flips compound across chunks")
        with self.subTest(quantity="o"):
            oracle.check_within(self, ROW, f"{case}/o", oracle.rel_l2(nat_o, sg_o), c=c_o, limit=limit)
        with self.subTest(quantity="ssm_state"):
            oracle.check_within(self, ROW, f"{case}/ssm_state", oracle.rel_l2(nat_state, sg_state), c=c_s,
                                limit=limit)

    def _both(self, tokens: int, seed: int):
        for layer in LAYERS:
            for carried in (False, True):
                with self.subTest(layer=layer, carried=carried):
                    self._case(_weights(layer), tokens, seed, carried)

    def test_L1(self):
        self._both(1, 5401)

    def test_L16(self):
        self._both(16, 5403)

    def test_L17(self):
        self._both(17, 5405)

    def test_L21(self):
        self._both(21, 5407)

    def test_L24(self):
        self._both(24, 5409)

    def test_L27(self):
        self._both(27, 5411)

    def test_L32(self):
        self._both(32, 5413)

    def test_L33(self):
        self._both(33, 5415)

    def test_L48(self):
        self._both(48, 5417)

    def test_L64(self):
        self._both(64, 5419)

    def test_L65(self):
        self._both(65, 5421)

    def test_L128(self):
        self._both(128, 5423)

    def test_L2048(self):
        self._both(2048, 5425)

    def test_L4096(self):
        self._both(4096, 5427)

    def test_repeat_bit_identical(self):
        """Two identical native calls (L 2048, carried state) give identical o and state bytes, per layer."""
        tokens = 2048
        for layer in LAYERS:
            with self.subTest(layer=layer):
                q, k, v, g, beta, state0 = _inputs(_weights(layer), 5429, tokens, True)
                args = (q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), g[0].contiguous(),
                        beta[0].contiguous())
                state_a, state_b = state0.clone(), state0.clone()
                o_a = _native_chunk(*args, state_a)
                o_b = _native_chunk(*args, state_b)
                case = f"layer{layer}/{CASE}_repeat_L{tokens}"
                oracle.check_bit_exact(self, ROW, f"{case}/o", o_a, o_b)
                oracle.check_bit_exact(self, ROW, f"{case}/ssm_state", state_a, state_b)

    def test_fp32_resume_bit_exact(self):
        """m210 D15⁗′: per layer (carried state, L 1,000: a partial last chunk), the capture at chunk c changes no byte
        of the call; a call over rows [64 c, L) from that FP32 capture gives the long call's o rows and final state bit
        for bit (c 1, 7, 15: the last resumes over 40 rows); the capture at chunk 0 is the initial state and the one
        after the last chunk rounds to the final state. The mutant (resume at c 7 from the BF16-rounded capture, as a
        call end stores it) must differ. The plain call's digests (report_only, computed first) compare libraries."""
        tokens = 1000
        last = (tokens + 63) // 64
        f32 = torch.float32
        plain = {}
        for layer in LAYERS:
            q, k, v, g, beta, state0 = _inputs(_weights(layer), 5431, tokens, True)
            args = (q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), g[0].contiguous(), beta[0].contiguous())
            state_p = state0.clone()
            o_p = _native_chunk(*args, state_p)
            plain[layer] = (args, state0, o_p, state_p)
            oracle.report_row(ROW, f"layer{layer}/{CASE}_fp32_L{tokens}/plain_digest", phase="report_only",
                              o=_sha256(o_p), ssm_state=_sha256(state_p))
        for layer, (args, state0, o_p, state_p) in plain.items():
            with self.subTest(layer=layer):
                for chunk in (0, 1, 7, 15, last):
                    case = f"layer{layer}/{CASE}_fp32_L{tokens}_c{chunk}"
                    cap = oracle.empty((HV, DV, DK), f32)
                    state_a = state0.clone()
                    o_a = oracle.empty((tokens, HV, DV), BF16)
                    oracle.call("cere_ktest_gdn_chunk_prefill_fp32", *args, state_a, o_a, tokens, SCALE, None, cap,
                                chunk)
                    oracle.check_bit_exact(self, ROW, f"{case}/capture_o", o_a, o_p)
                    oracle.check_bit_exact(self, ROW, f"{case}/capture_state", state_a, state_p)
                    if chunk == 0:
                        oracle.check_bit_exact(self, ROW, f"{case}/initial", cap, state0.to(f32))
                        continue
                    if chunk == last:
                        oracle.check_bit_exact(self, ROW, f"{case}/final", cap.to(BF16), state_p)
                        continue
                    r0 = 64 * chunk
                    rest = tokens - r0
                    tail = tuple(x[r0:].contiguous() for x in args)
                    state_b = oracle.empty((HV, DV, DK), BF16)
                    o_b = oracle.empty((rest, HV, DV), BF16)
                    oracle.call("cere_ktest_gdn_chunk_prefill_fp32", *tail, state_b, o_b, rest, SCALE, cap, None, -1)
                    oracle.check_bit_exact(self, ROW, f"{case}/resume_o", o_b, o_p[r0:])
                    oracle.check_bit_exact(self, ROW, f"{case}/resume_state", state_b, state_p)
                    if chunk == 7:
                        state_m = cap.to(BF16)
                        o_m = _native_chunk(*tail, state_m)
                        differing = int((o_m.view(torch.int16) != o_p[r0:].view(torch.int16)).sum())
                        oracle.report_row(ROW, f"{case}/bf16_mutant", phase="report_only", differing=differing)
                        # A BF16 resume must differ from the FP32 one (S1).
                        oracle.check_count(self, ROW, f"{case}/bf16_mutant_differs", int(differing > 0), expected=1)


if __name__ == "__main__":
    unittest.main()
