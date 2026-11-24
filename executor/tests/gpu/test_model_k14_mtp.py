"""K14 (tasks/m29-spec.md D3, K-table row K3, C11): the native MTP (NEXTN) draft layer (executor/cuda/src/model/mtp.h)
on the real `mtp.*` weights vs `oracle.mtp_reference`, the torch port of SGLang's Qwen4ExpForCausalLMMTP forward
(sglang/srt/models/qwen4_exp_mtp.py:73-216: fusion, the one full-attention decoder layer, the mixer, hot lm_head and
token map).

Suite `model` (GPU0; `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite model`, or `--suite model_k14`
for this file alone).

Setup: the real A4 model with the draft layer (`oracle.create_model(..., draft_token_map=...)`, `mtp.*` DEVICE) at a
CONTEXT-token context, one session; the draft token map is a seeded set of MAP_COUNT distinct ids (strictly
increasing). Inputs are synthetic: seeded token ids (no prompt text). The native entry PREFILLs the tokens on the
target, takes each position's pre-final-mixer hc row (the hc rows the MTP layer reads, `native.final_mix.in`), and
runs the MTP layer over positions 0 .. n - 1 in calls of `mtp_chunk` rows, row p reading hc[p] and next_ids[p] (the
token at p + 1; the last row's is a seeded id). The oracle runs on the native hc rows (the target is not under test
here) with the same call partition (the attention's KV source per call, `oracle._mtp_attention`).

Cases (ROW "K14"):
- `draft_rows`: ROWS = 512 rows in calls of DRAFT_CHUNK = 4 rows (the draft-extend/verify row count; every call but
  the first reads the MTP layer's FP8 KV cache).
- `extend_m4096`: one call of EXTEND_ROWS = 4,096 rows (one prefill chunk, the MTP extend after a target PREFILL
  chunk; above 2,048 keys the QSA selection, the oracle's being SGLang's QSA indexer selection).
Per case, gated [REVISED: M29S-K14R, gate changed after a miss]: `draft_map`, the rows whose native draft id is not
map[argmax(native hot logits)] (count, expected 0); `stage/<name>` for each oracle.MTP_GATE_STAGES step (the MTP_STAGES,
the pre-mixer `hc_out` rows and the hot `logits`): rel-L2(native stage, the oracle step on the native inputs of that
step, `oracle.mtp_local_stages`) <= calibrated_limit(c, oracle.MTP_STAGE_FLOORS[name]) with c = rel-L2(the control's
dispatched op on the same inputs, the oracle step) (`oracle.mtp_control_stage`; c = 0 where none is callable).
Reported, not gated (the round 1-3 misses: the chain amplifies stage roundings ~100x into top-10/512 routing flips,
and the control's own floors sit at the old 0.99 / 2e-2 bars): `chain/<name>` (the whole oracle chain from the native
hc rows), `top1` and `logits` of that chain, `seeded/<stage>` (the oracle chain from the native stages up to <stage>),
`floor/fp32_spread` (the oracle chain vs the same chain with cuBLAS FP32-output GEMMs), `floor/control_repeat` (two
runs of the chain with SGLang's dispatched mixes), `floor/control_vs_oracle` and `native_vs_control`. The outcome gate
of draft quality is the acceptance-length row (M2.9 spec K3) and correctness the spec == plain greedy identity.
- `d10/*` (spec D10', A1): `cere_ktest_mtp_plan_bytes` of this handle's plan at R = 4, context and pool 200,000,
  spec-off (no map, `mtp.*` DISCARD) and drafting; reported, with the workspace growth equal to
  model::mtp::workspace_bytes and the same workspace rows (counts).

Native entry points (executor/cuda/src/ktest/k14_ktest.cpp):

    int cere_ktest_mtp_forward(uint64_t model, const uint32_t *tokens /*host [n]*/, uint64_t n,
                               const uint32_t *next_ids /*device [n]*/, uint64_t mtp_chunk,
                               void *hc /*device BF16 [n, 10,240]*/, void *hc_out /*device BF16 [n, 10,240]*/,
                               void *logits /*device BF16 [n, draft count]*/, uint32_t *drafts /*device [n]*/,
                               cudaStream_t stream);
    int cere_ktest_mtp_plan_bytes(uint64_t model, uint64_t max_context_tokens, uint64_t max_sessions,
                                  uint64_t kv_pool_tokens, uint64_t draft, uint64_t *out /*host [12]*/,
                                  cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch

import oracle
from sglang.srt.runtime_context import get_context

ROW = "K14"
CONTEXT = 8192
ROWS = 512
DRAFT_CHUNK = 4
EXTEND_ROWS = 4096
MAP_COUNT = 65_536
SEED = 2914
PERTURB: tuple[str, float] | None = None
"""Test-only fail-first hook: (MTP stage, scale) multiplies that native stage before the gates. None when committed."""
BF16 = torch.bfloat16
# D10' (cere_ktest_mtp_plan_bytes): the native-g0 shape (configs/runtime/native-g0.toml, R = max_sessions 4).
D10_CONTEXT = 200_000
D10_SESSIONS = 4
D10_POOL_TOKENS = 200_000
PLAN_FIELDS = ("weight_bytes", "workspace_rows", "workspace_bytes", "derived_bytes", "session_bytes", "pool_bytes",
               "device_bytes", "pinned_bytes", "draft_count", "mtp_derived_bytes", "mtp_hc_rows_bytes",
               "mtp_workspace_growth")


def _draft_map(vocab: int) -> torch.Tensor:
    """MAP_COUNT distinct seeded ids < vocab, strictly increasing (int32, CPU)."""
    g = torch.Generator()
    g.manual_seed(SEED)
    return torch.sort(torch.randperm(vocab, generator=g)[:MAP_COUNT]).values.to(torch.int32).contiguous()


def _tokens(n: int, vocab: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded synthetic ids: (tokens int32 CPU [n], next_ids int32 DEVICE [n]); next_ids[p] = tokens[p + 1], the last
    row's a further seeded id."""
    g = torch.Generator()
    g.manual_seed(seed)
    ids = torch.randint(0, vocab, (n + 1,), generator=g, dtype=torch.int64).to(torch.int32)
    return ids[:n].contiguous(), ids[1:].contiguous().to(oracle.DEVICE)


class K14MtpDraft(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The rotary table and the QSA indexer of the oracle read the published config bags (as the K6/K9 suites).
        cls.context_override = get_context().override_server_args(kv_cache_dtype="fp8_e4m3")
        cls.context_override.install()
        cls.addClassCleanup(cls.context_override.restore)
        tc = oracle.text_config()
        cls.vocab = int(tc["vocab_size"])
        cls.hc_hidden = int(tc["hc_count"]) * int(tc["hidden_size"])
        hidden, heads, kv_heads, head_dim = (int(tc[k]) for k in ("hidden_size", "num_attention_heads",
                                                                  "num_key_value_heads", "head_dim"))
        wide = {"fusion.hyper", "attn.normed", "attn.hyper", "mlp.normed"}
        cls.stage_cols = {name: cls.hc_hidden if name in wide else hidden for name in oracle.MTP_STAGES}
        cls.stage_cols["attn.qkv"] = (2 * heads + 2 * kv_heads) * head_dim
        cls.stage_cols["attn.out"] = heads * head_dim
        cls.map = _draft_map(cls.vocab)
        cls.map_dev = cls.map.to(oracle.DEVICE).to(torch.int64)
        cls.model = oracle.create_model(CONTEXT, draft_token_map=cls.map)
        cls.addClassCleanup(oracle.release_model, cls.model)

    def _native(self, tokens: torch.Tensor, next_ids: torch.Tensor, chunk: int):
        n = tokens.numel()
        hc = oracle.empty((n, self.hc_hidden), BF16)
        hc_out = oracle.empty((n, self.hc_hidden), BF16)
        logits = oracle.empty((n, MAP_COUNT), BF16)
        drafts = oracle.empty((n,), torch.int32)
        stages = {name: oracle.empty((n, self.stage_cols[name]), BF16) for name in oracle.MTP_STAGES}
        pointers = torch.tensor([stages[name].data_ptr() for name in oracle.MTP_STAGES], dtype=torch.int64)
        oracle.call("cere_ktest_mtp_forward", self.model, tokens, n, next_ids, chunk, hc, hc_out, logits, drafts,
                    pointers)
        return hc, hc_out, logits, drafts, stages

    def _case(self, case: str, n: int, chunk: int, seed: int) -> None:
        tokens, next_ids = _tokens(n, self.vocab, seed)
        hc, hc_out, logits, drafts, stages = self._native(tokens, next_ids, chunk)
        if PERTURB is not None:
            name, scale = PERTURB
            stages[name].copy_((stages[name].float() * scale).to(BF16))
        drafts = drafts.to(torch.int64)
        native_pick = self.map_dev[logits.float().argmax(dim=-1)]
        oracle.check_count(self, ROW, f"{case}/draft_map", int((drafts != native_pick).sum().item()), expected=0)

        # Gates: each stage's oracle step on the native inputs of that step, calibrated against the control's op.
        for name, ref in oracle.mtp_local_stages(stages, hc, next_ids, hc_out, self.map_dev, chunk):
            native = logits if name == "logits" else hc_out if name == "hc_out" else stages[name]
            floor = oracle.MTP_STAGE_FLOORS[name]
            control = oracle.mtp_control_stage(name, stages, hc, next_ids, hc_out, self.map_dev, chunk)
            if control is None:
                c = 0.0
                oracle.report_row(ROW, f"{case}/stage/{name}", phase="calibration", c=c, floor=floor,
                                  limit=oracle.calibrated_limit(c, floor), control="none")
            else:
                c = oracle.calibrate(ROW, f"{case}/stage/{name}", control, ref, floor)
            del control
            limit = oracle.calibrated_limit(c, floor)
            with self.subTest(stage=name):
                oracle.check_within(self, ROW, f"{case}/stage/{name}", oracle.rel_l2(native, ref), c=c, limit=limit)
            del ref

        # Evidence (reported, not gated; [REVISED: M29S-K14R]): the whole oracle chain from the native hc rows, the
        # chains seeded with native stages, and the control's own floors measured on the same rows.
        chain: dict[str, float] = {}

        def stage(name: str, ref: torch.Tensor) -> None:
            chain[name] = oracle.rel_l2(stages[name], ref)

        ref_hc_out, ref_logits = oracle.mtp_reference(hc, next_ids, self.map_dev, chunk, stage)
        chain["hc_out"] = oracle.rel_l2(hc_out, ref_hc_out)
        del ref_hc_out
        for name in oracle.MTP_STAGES:
            oracle.report_row(ROW, f"{case}/chain/{name}", phase="report", measured=chain[name])
        oracle.report_row(ROW, f"{case}/chain/hc_out", phase="report", measured=chain["hc_out"])
        agree = int((drafts == self.map_dev[ref_logits.float().argmax(dim=-1)]).sum().item())
        oracle.report_row(ROW, f"{case}/top1", phase="report", rows=n, agree=agree, fraction=agree / n)
        oracle.report_row(ROW, f"{case}/logits", phase="report", measured=oracle.rel_l2(logits, ref_logits))
        # The report-only chains below keep their hot logits on the CPU (4,096 x 65,536 BF16 = 512 MiB each; the
        # extend case holds four next to the model); rel_l2 moves the CPU operand's chunks to the device of `a`.
        ref_logits = ref_logits.cpu()

        def pair(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
            top1 = self._top1(a, b)
            oracle.report_row(ROW, f"{case}/{label}", phase="report", logits=oracle.rel_l2(a, b), top1=top1)

        for start in ("attn.qkv", "attn.out", "mlp.mixed"):
            seeded = oracle.mtp_reference(hc, next_ids, self.map_dev, chunk, seed=stages, start=start)[1].cpu()
            pair(f"seeded/{start}", logits, seeded)
            del seeded
        spread = oracle.mtp_reference(hc, next_ids, self.map_dev, chunk, gemm="mm32")[1].cpu()
        pair("floor/fp32_spread", spread, ref_logits)
        del spread
        first = oracle.mtp_reference(hc, next_ids, self.map_dev, chunk, mix="sglang")[1].cpu()
        again = oracle.mtp_reference(hc, next_ids, self.map_dev, chunk, mix="sglang")[1].cpu()
        pair("floor/control_repeat", again, first)
        pair("floor/control_vs_oracle", first, ref_logits)
        pair("native_vs_control", logits, first)

    def _top1(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """The fraction of rows whose hot argmax ids agree (each argmax on its operand's device, compared on CPU)."""
        return float((a.float().argmax(dim=-1).cpu() == b.float().argmax(dim=-1).cpu()).float().mean().item())

    def test_draft_rows_vs_oracle(self):
        self._case("draft_rows", ROWS, DRAFT_CHUNK, SEED + 1)

    def test_extend_one_prefill_chunk_vs_oracle(self):
        self._case("extend_m4096", EXTEND_ROWS, EXTEND_ROWS, SEED + 2)

    def _plan(self, draft: int) -> dict[str, int]:
        out = torch.zeros(len(PLAN_FIELDS), dtype=torch.int64)
        oracle.call("cere_ktest_mtp_plan_bytes", self.model, D10_CONTEXT, D10_SESSIONS, D10_POOL_TOKENS, draft, out)
        return dict(zip(PLAN_FIELDS, (int(v) for v in out.tolist())))

    def test_d10_plan_bytes(self):
        """D10' from the layout functions at R = 4, context and pool 200,000 (the native-g0 shape, text-only desc):
        the spec-off plan and the drafting plan, and the MTP parts as their differences."""
        off = self._plan(0)
        on = self._plan(1)
        oracle.report_row(ROW, "d10/spec_off", **off)
        oracle.report_row(ROW, "d10/drafts4", **on)
        delta = {f: on[f] - off[f] for f in PLAN_FIELDS[:8]}
        oracle.report_row(ROW, "d10/delta", **delta, mtp_session_bytes_per_slot=delta["session_bytes"],
                          mtp_pool_bytes=delta["pool_bytes"], workspace_growth=delta["workspace_bytes"])
        oracle.check_count(self, ROW, "d10/spec_off_draft_count", off["draft_count"], expected=0)
        oracle.check_count(self, ROW, "d10/draft_count", on["draft_count"], expected=MAP_COUNT)
        oracle.check_count(self, ROW, "d10/workspace_growth", delta["workspace_bytes"],
                           expected=on["mtp_workspace_growth"])
        oracle.check_count(self, ROW, "d10/workspace_rows", delta["workspace_rows"], expected=0)


if __name__ == "__main__":
    unittest.main()
