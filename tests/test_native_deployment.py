"""Native deployment files (tasks/m22-spec.md R6, Q15; tasks/m22c-spec.md D6, C3; tasks/m25-spec.md D6, C4):
`configs/deployments/qwen38-flash-next-native.*`, `configs/runtime/native-g0.toml` and `scripts/native/lock_native.py`,
which pins the executor library from a clean `build.json`."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cere_contracts import GRAMMAR_OUTPUTS, ContractError, load_json, validate, validate_deployment
import pin_configs

PREFIX = 'configs/deployments/qwen38-flash-next-native'
SGLANG_PREFIX = 'configs/deployments/qwen38-flash-next'
TOML = 'configs/runtime/native-g0.toml'
PIN = 'configs/artifacts/qwen38-flash-next-nvfp4.pin.json'
PROFILE_ID = 'native-g0-qwen38-flash-next-v8'
# M2.8 D10 `-v8` (ledger P1, maintainer 2026-10-11: `-v6` = M2.13, `-v7` = main/M2.7): the grammar outputs.
V7_PROFILE_ID = 'native-g0-qwen38-flash-next-v7'
GRAMMAR_OUTPUTS = ['json_schema', 'regex', 'ebnf', 'tool_calls_constrained']
# M2.12 A1 (m212 design note D2, ledger P1): `-v5` = main's `-v4` (M2.6 D9/D10) plus the three M2.12 capability
# fields and the D9 library; P1 numbers it 1 + the highest on `main` at the merge.
V4_PROFILE_ID = 'native-g0-qwen38-flash-next-v4'
M212_CAPABILITIES = {'thinking_toggle': True, 'token_controls': ['stop_token_ids', 'ignore_eos'], 'vocab_size': 248320}
# M2.7 D11 takes `-v7` (ledger P1: `-v5` is M2.12's on `main`, `-v6` is claimed by M2.13).
V5_PROFILE_ID = 'native-g0-qwen38-flash-next-v5'
CONTEXT = 200000
LOCK_NATIVE = ROOT / 'scripts/native/lock_native.py'
LIBRARY_SHA = 'a' * 64
# m213 design note D9, Q2 b): the vision deployment, its own runtime config and lock; native-g0 stays text-only.
VISION_PREFIX = 'configs/deployments/qwen38-flash-next-native-vision'
VISION_TOML = 'configs/runtime/native-g0-vision.toml'
VISION_PIN = 'configs/artifacts/qwen38-flash-next-nvfp4-vision.pin.json'
VISION_PROFILE_ID = 'native-g0-qwen38-flash-next-v6'  # ledger P1: claimed for M2.13 on main (M2.12 -v5, M2.7 -v7)
VISION_CONTEXT = 2048  # D5: until K9 takes positions3, image sessions stay within indexer_budget
IMAGE_COMBINATION = ('decode', 'image', 'text', 'optimized', 'greedy', (1,), VISION_CONTEXT)
# M2.11 D5/LR5 (owner N1 = (a)): the dev/live `<profile id>-snapshot` deployment. Not shipped: it lives beside the
# deployment parts, never in configs/runtime/ (the shipped runtime set, test_contracts.py).
DEV_PREFIX = 'configs/deployments/qwen38-flash-next-native-snapshot'
DEV_TOML = f'{DEV_PREFIX}.toml'
DEV_PROFILE_ID = f'{PROFILE_ID}-snapshot'
DEV_ARENA = 4 << 30
# M23-D7 (m23 design note): the shipped profiles pin the `CERE_MOE_BACKEND=p3b` library; the opt-in `-v9` P3b
# deployment (M23-D1) is retired. A K7 build (`--moe-backend k7`, no flag) is the reference and pins no pair.
RETIRED_P3B = ('configs/runtime/native-g0-p3b.toml',
               *(f'configs/deployments/qwen38-flash-next-native-p3b.{p}.json'
                 for p in ('capabilities', 'recipe', 'lock')))
P3B_BUILD_FLAGS = {'cuda': '-O3', 'cxx': '-O3', 'link': '', 'moe_backend': 'p3b', 'p3b_cuda': '-use_fast_math'}
K7_BUILD_FLAGS = {'cuda': '-O3', 'cxx': '-O3', 'link': ''}
# M23-D7: the sentence the shipped (and GPU1, vision, snapshot) capabilities notes start with; the P4 profile's notes
# end with the whole shipped notes (P4 is the one change on top of them).
P3B_NOTE = ('MoE backend P3b (tasks/m23-spec.md M23-D7, shipped default): the CERE_MOE_BACKEND=p3b library, routed '
            'experts on the vendored FlashInfer 0.6.17 SM120 CUTLASS fused MoE with per-m-bucket pinned deterministic '
            'GEMM1/GEMM2 tactic pairs (M23-D6), each bit-equal to the K7 reference at block level; promoted by the '
            "identity gate (native_hashes identical to K7's on the D9 parity set). ")
# P11 (m23 design note P11R-07, D9C): the opt-in `-v14` deployment, the shipped profile plus exactly one lever,
# `[deployments.native.levers] k2_gemm = "small_m"` (m23 design note PI-02: one library, levers in config).
K2S_PREFIX = 'configs/deployments/qwen38-flash-next-native-k2s'
K2S_TOML = 'configs/runtime/native-g0-k2s.toml'
K2S_PROFILE_ID = 'native-g0-qwen38-flash-next-v14'
# M2.3 P4 (m23 design note P4, P4R-01..07): the opt-in `-v12` deployment, the shipped profile plus exactly one
# lever, `prefill_attention = "p4"` (in-tree tensor-core prefill attention).
P4_PREFIX = 'configs/deployments/qwen38-flash-next-native-p4'
P4_TOML = 'configs/runtime/native-g0-p4.toml'
P4_PROFILE_ID = 'native-g0-qwen38-flash-next-v12'
# The retired CMake selections (CERE_PREFILL_ATTENTION, CERE_K2_SMALL_M) as a pre-lever build.json recorded them;
# lock_native refuses every one of them (any value, any backend).
P4_PRE_LEVER_FLAGS = dict(P3B_BUILD_FLAGS, prefill_attention='p4')
PRE_LEVER_BUILD_FLAGS = (P4_PRE_LEVER_FLAGS, dict(P3B_BUILD_FLAGS, k2_small_m=True),
                         dict(K7_BUILD_FLAGS, prefill_attention='p4'), dict(P3B_BUILD_FLAGS, prefill_attention='p5'),
                         dict(P3B_BUILD_FLAGS, k2_small_m=False), {'prefill_attention': 'cuda_core'})
# m23 design note P8: the shipped `-v8` pair (native-g0, native-g1) decodes 16 rows; the profiles built on it that
# P8 left alone (`-v12` P4, `-v14` K2 small-M) keep 4. `rows_masked` drops exactly those rows figures, so the rest
# still compares equal.
ROWS = tuple(range(1, 17))
PRE_P8_ROWS = (1, 2, 3, 4)
P8_ROWS_NOTE = 'up to 16 rows (tasks/m23-spec.md P8; 4 before)'


def rows_masked(kind, value):
    """`value` (capabilities or a runtime TOML) without the P8 rows figures."""
    value = json.loads(json.dumps(value))
    if kind == 'capabilities':
        del value['decode_batch_max']
        for c in value['validated_combinations']:
            if c['operation'] in ('decode', 'verify'):
                del c['row_counts']
    else:
        del value['admission']['max_backend_inflight'], value['admission']['per_scope_inflight']
    return value


# The GPU1 native lane (docs/RUNBOOK.md "GPU1 native lane"): a shipped runtime profile and deployment that equal the
# GPU0 ones except the lane's identity: its own port, GPU, profile id, deployment and ledger state_dir.
G1_PREFIX = 'configs/deployments/qwen38-flash-next-native-g1'
G1_TOML = 'configs/runtime/native-g1.toml'
G1_PROFILE_ID = PROFILE_ID.replace('native-g0-', 'native-g1-', 1)  # follows every relock of PROFILE_ID
GPU0_UUID = 'GPU-aaaaaaaa-0000-4000-8000-000000000000'
GPU1_UUID = 'GPU-bbbbbbbb-1111-4111-8111-111111111111'
# m29 design note D11/D12 (M29S2-16, OQ8): the MTP speculation profile, the shipped `-v8` one plus 4 draft tokens,
# `verify` and the verify combinations; a shipped runtime config pinned with the shipped library (PAIRS). `-v19` (owner
# decision 2026-10-15, every opt-in except p4; was -v18) adds SPEC_LEVERS (incl. the numerics-changing k2_gemm small_m
# and the lossless mtp+lookup drafter) and the prefix cache (PREFIX_TOML_DELTA); `-v19` without its levers table and
# PREFIX_TOML_DELTA is the lever-free speculation reference (`-v10`'s shape, now retired; m23 design note PI-02).
# The NVFP4 opt-in since the FP4-Local flip (fp4local design note FL-31); the default is DEFAULT_SPEC_*.
SPEC_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-nvfp4'
SPEC_TOML = 'configs/runtime/native-g0-spec-nvfp4.toml'
SPEC_PROFILE_ID = 'native-g0-qwen38-flash-next-v19'
SPEC_LEVERS = {'decode_step': 'spec_graph', 'verify_attention': 'batched', 'draft_policy': 'adaptive',
               'k2_gemm': 'small_m', 'drafter': 'mtp+lookup'}
# The sentence the `-v19` capabilities notes start with; the rest are the lever-free speculation notes `-v16` ends with,
# with PREFIX_NOTES in place of PRE_PREFIX_NOTES (spec_rest_notes).
SPEC_LEVERS_NOTE = ('Speculation levers and prefix cache (tasks/m23-spec.md PI-02, DG rows; owner decision '
                    '2026-10-15): levers.decode_step spec_graph, levers.verify_attention batched and levers.draft_policy '
                    'adaptive (bit-identical), levers.k2_gemm small_m (numerics-changing: tokens and scores equal the '
                    'small_m plain run, -v14) and levers.drafter mtp+lookup (lossless); the prefix cache (cache_salt, '
                    '16 slots) is never looked up by a drafting session (CR8), only by NO_SPECULATION requests (-v11 '
                    'cache-on numerics). ')
# M2.10 (m210 design note D11, N2 (a); owner/maintainer D9C 2026-10-12): the shipped native-g0 stays the cache-off
# `-v8` profile and gains only the operator route (the flush answers prefix_cache "off" at slots 0); the GPU1 lane,
# P3b, vision and snapshot profiles stay main's (without it).
OPERATOR_DELTA = {('auth', 'operator_token_env'): 'CERE_OPERATOR_TOKEN'}
# The cache is the opt-in `-v11` profile: native-g0-prefix.toml = native-g0.toml + PREFIX_TOML_DELTA on its own profile
# id and deployment, whose capabilities differ from the shipped ones only in profile_id and PREFIX_NOTES replacing
# PRE_PREFIX_NOTES. native-g1-prefix.toml mirrors it on the GPU1 lane (native-g1 + PREFIX_TOML_DELTA + OPERATOR_DELTA).
PREFIX_TOML_DELTA = {
    ('deployments', 0, 'prefix_cache_isolation'): 'cache_salt',
    ('deployments', 0, 'native', 'prefix_state_slots'): 16,
}
PREFIX_NOTES = ('a native prefix cache (M2.10: 64-token pages and GDN state checkpoints in 16 state slots, each scope '
                'namespaced by its cache_salt; a hit is bit-exact with the cold run split at the hit, not with a '
                'single-call cold run; image sessions never use it; usage.prompt_tokens_details.cached_tokens reports '
                'the hit)')
PRE_PREFIX_NOTES = 'no prefix cache'
G0_PREFIX_PREFIX = 'configs/deployments/qwen38-flash-next-native-prefix'
G0_PREFIX_TOML = 'configs/runtime/native-g0-prefix.toml'
G0_PREFIX_PROFILE_ID = 'native-g0-qwen38-flash-next-v11'
G1_PREFIX_PREFIX = 'configs/deployments/qwen38-flash-next-native-g1-prefix'
G1_PREFIX_TOML = 'configs/runtime/native-g1-prefix.toml'
G1_PREFIX_PROFILE_ID = 'native-g1-qwen38-flash-next-v11'
# P11 lever 2 (m23 design note P11): the opt-in `-v16` profile, the `-v19` speculation one without its levers table
# and PREFIX_TOML_DELTA
# plus exactly one lever, levers.k2_gemm small_m (its VERIFY on the small-M kernel), on the shipped library (PAIRS).
SPEC_K2S_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-k2s'
SPEC_K2S_TOML = 'configs/runtime/native-g0-spec-k2s.toml'
SPEC_K2S_PROFILE_ID = 'native-g0-qwen38-flash-next-v16'
# m29l design note D7: the opt-in `-v15` profile, `-v19` without its levers table and PREFIX_TOML_DELTA (the `drafter`
# lever absent, "mtp").
SPEC_LOOKUP_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-lookup'
SPEC_LOOKUP_TOML = 'configs/runtime/native-g0-spec-lookup.toml'
SPEC_LOOKUP_PROFILE_ID = 'native-g0-qwen38-flash-next-v15'
# Fp8W: the opt-in FP8 dense-weights `-v20` profile, the `-v19` speculation one (levers, prefix cache) plus exactly one
# lever, levers.k2_weights fp8 (numerics-changing, restart-only), on the shipped library (PAIRS); the default stays -v19.
SPEC_FP8W_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-fp8w'
SPEC_FP8W_TOML = 'configs/runtime/native-g0-spec-fp8w.toml'
SPEC_FP8W_PROFILE_ID = 'native-g0-qwen38-flash-next-v20'
# M3.2N (m32n design note D1/D2/D5/D16): the two-card deployment, the `-v19` speculation one served by two in-process
# native replicas (gpu0, gpu1) on the shipped library (PAIRS), one capability set per card.
TWO_GPU_PREFIX = 'configs/deployments/qwen38-flash-next-native-2gpu'
TWO_GPU_TOML = 'configs/runtime/native-2gpu.toml'
TWO_GPU_PROFILE_ID = 'native-2gpu-qwen38-flash-next-v1'
# FP4-Local FL-13 (fp4local design note D12, D16): the S2 dev `-v19-fp4local` profile, the `-v19` speculation one
# (levers, prefix cache, library) on the converted FP4-Local artifact; never the default.
SPEC_FP4LOCAL_PREFIX = 'configs/deployments/qwen38-flash-next-native-fp4local'
SPEC_FP4LOCAL_TOML = 'configs/runtime/native-g0-spec-fp4local.toml'
SPEC_FP4LOCAL_PROFILE_ID = 'native-g0-qwen38-flash-next-v19-fp4local'
FP4LOCAL_PIN = 'configs/artifacts/qwen38-flash-next-fp4local-cere.pin.json'
# FP4-Local S3 (fp4local design note S3, D12, FL-30): the opt-in `-v21` profile (id tentative until Main assigns it),
# the S2 dev one on the S3 artifact (MXFP8 dense + NVFP4 MTP experts kept); never the default.
S3_PREFIX = 'configs/deployments/qwen38-flash-next-native-fp4local-s3'
S3_TOML = 'configs/runtime/native-g0-spec-fp4local-s3.toml'
S3_PROFILE_ID = 'native-g0-qwen38-flash-next-v21'
S3_PIN = 'configs/artifacts/qwen38-flash-next-fp4local-cere-s3.pin.json'
# Headroom's `-v22` (m210 design note D15''''' O112; the default speculation profile from the FP4-Local flip,
# fp4local design note FL-31, to the DraftTrain flip, v29default design note VD-03; opt-in since): `-v21` with
# the GPU0 memory S3 frees spent on 32 prefix state slots, prefix end states and a 300,032-token KV pool.
V22_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-v22'
V22_TOML = 'configs/runtime/native-g0-spec-v22.toml'
V22_PROFILE_ID = 'native-g0-qwen38-flash-next-v22'
# TreeSpec (tree-spec design note TS-03): the opt-in `-v28`, the `-v22` deployment (artifact, library,
# capabilities) under its own id with exactly one lever changed, drafter "mtp+lookup+long" (long lookup chains, lossless).
TREE_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-tree'
TREE_TOML = 'configs/runtime/native-g0-spec-tree.toml'
TREE_PROFILE_ID = 'native-g0-qwen38-flash-next-v28'
# The default speculation profile since the DraftTrain flip (v29default design note VD-03): DraftTrain's `-v29`
# (drafttrain design note D7), the `-v22` deployment (library, levers, capabilities) on the S3 artifact variant
# whose BF16 `mtp.*` draft weights were self-distilled and whose draft token map was re-derived (its own pin).
DEFAULT_SPEC_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec'
DEFAULT_SPEC_TOML = 'configs/runtime/native-g0-spec.toml'
DEFAULT_SPEC_PROFILE_ID = 'native-g0-qwen38-flash-next-v29'
DRAFT_PIN = 'configs/artifacts/qwen38-flash-next-fp4local-cere-s3-draft.pin.json'
# Chains (chains design note): the opt-in `-v36`, the TreeSpec `-v28` deployment (the default `-v22` artifact,
# library, capabilities) under its own id with exactly one lever changed, drafter "mtp+lookup+chain" (chains up to the
# model's max chain drafts, lossless).
CHAINS_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-chains'
CHAINS_TOML = 'configs/runtime/native-g0-spec-chains.toml'
CHAINS_PROFILE_ID = 'native-g0-qwen38-flash-next-v36'
# Sessions (sessions design note O211): the opt-in `-v45`, the default `-v29` deployment (artifact, library, levers,
# rows, pool, slots) under its own id plus the M2.10 prefix-cache host tier: a 16 GiB pinned snapshot arena and
# `[state] host_spill_enabled`, so its capabilities add snapshot/restore and native_state_snapshot (the arena).
SESSIONS_PREFIX = 'configs/deployments/qwen38-flash-next-native-sessions'
SESSIONS_TOML = 'configs/runtime/native-g0-sessions.toml'
SESSIONS_PROFILE_ID = 'native-g0-qwen38-flash-next-v45'
SESSIONS_ARENA_BYTES = 16 << 30

# Ctx262 CX-02 (ctx262 design note): the opt-in `-v44`, the default `-v29` deployment (artifact, levers, library)
# under its own id with the context raised to the model's max_position_embeddings 262,144 (every context field of the
# capabilities, recipe and lock) and its runtime config's KV pool 524,288.
CTX262_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-262k'
CTX262_TOML = 'configs/runtime/native-g0-spec-262k.toml'
CTX262_PROFILE_ID = 'native-g0-qwen38-flash-next-v44'
# VisionLong (m213l design note): the opt-in `-v43`, the default `-v29` deployment (library, levers, text
# capabilities) on the vision variant of its artifact (every `-v29` tensor plus the BF16 vision tower on the device and
# the preprocessor config; its own pin and artifact profile) with image combinations at the text shape and sidecar
# feature `image`.
VISION_LONG_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-vision'
VISION_LONG_TOML = 'configs/runtime/native-g0-spec-vision.toml'
VISION_LONG_PROFILE_ID = 'native-g0-qwen38-flash-next-v43'
VISION_LONG_PIN = 'configs/artifacts/qwen38-flash-next-fp4local-cere-s3-draft-vision.pin.json'
TEXT_ARTIFACT_PROFILE = 'configs/artifacts/profiles/rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1.json'
VISION_ARTIFACT_PROFILE = 'configs/artifacts/profiles/rtxpro6000bw-plehost-nvfp4-mxfp8-vision-v1.json'
# VisionLong Amendment A2 (m213l design note): the dev-only diagnostic `-v54`, the `-v43` deployment (artifact,
# library, capabilities) under its own id with exactly one key added, `native.prefix_alt_edges` (PrefixHit's D17).
VISION_V54_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-vision-v54'
VISION_V54_TOML = 'configs/runtime/native-g0-spec-vision-v54.toml'
VISION_V54_PROFILE_ID = 'native-g0-qwen38-flash-next-v54'
# Ctx262 CX-R-vision (ctx262 design note CX-R18): the dev-only diagnostic `-v55`, the `-v54` deployment under its own
# id with every context field 262,144; its runtime config `-v54`'s at context 262,144, pool 524,288 and a raised limit.
VISION_V55_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-vision-v55'
VISION_V55_TOML = 'configs/runtime/native-g0-spec-vision-v55.toml'
VISION_V55_PROFILE_ID = 'native-g0-qwen38-flash-next-v55'
# PrefixHit (m210 design note D17): the opt-in `-v51`, the default `-v29` deployment (artifact, library,
# capabilities) under its own id with exactly one key added, `native.prefix_alt_edges` (alternative prefix-cache edges).
V51_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-v51'
V51_TOML = 'configs/runtime/native-g0-spec-v51.toml'
V51_PROFILE_ID = 'native-g0-qwen38-flash-next-v51'
# Ctx262 CX-09 (ctx262 design note): `-v52` = the Ctx262 `-v44` deployment (artifact, levers, library, capabilities,
# context 262,144) under its own id, its runtime config `-v44`'s plus `native.prefix_alt_edges`; production since CX-09.
V52_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-v52'
V52_TOML = 'configs/runtime/native-g0-spec-v52.toml'
V52_PROFILE_ID = 'native-g0-qwen38-flash-next-v52'
# MixedPD (mixed-pd design note): the opt-in `-v47`, the default `-v29` deployment (artifact, library, capabilities)
# under its own id with exactly one runtime key added, `[deployments.native] prefill_slice_ms` 200.
MIXEDPD_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-v47'
MIXEDPD_TOML = 'configs/runtime/native-g0-spec-v47.toml'
MIXEDPD_PROFILE_ID = 'native-g0-qwen38-flash-next-v47'
# GrammarSpec (grammar-spec design note): the opt-in `-v49`, the default `-v29` deployment (artifact, library,
# capabilities) under its own id with exactly one lever added, grammar_drafts "on" (constrained requests speculate with
# grammar-checked drafts; changing for sampled constrained tickets only).
GRAMMAR_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-grammar'
GRAMMAR_TOML = 'configs/runtime/native-g0-spec-grammar.toml'
GRAMMAR_PROFILE_ID = 'native-g0-qwen38-flash-next-v49'
# Sessions SE-11 (sessions design note): `-v53` = the `-v52` deployment (artifact, levers, library, context 262,144)
# under its own id plus `-v45`'s host tier (snapshot/restore and native_state_snapshot declared; runtime config `-v52`'s
# plus the 16 GiB arena and `[state] host_spill_enabled`).
V53_PREFIX = 'configs/deployments/qwen38-flash-next-native-sessions-v53'
V53_TOML = 'configs/runtime/native-g0-sessions-v53.toml'
V53_PROFILE_ID = 'native-g0-qwen38-flash-next-v53'

# Track LoRA (Wave P): the opt-in `-v50`, the default `-v29` deployment (artifact, library, levers) under its own id
# plus `[deployments.native.lora]` (the adapter pool and its root); requests without an adapter are unchanged.
LORA_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-lora'
LORA_TOML = 'configs/runtime/native-g0-spec-lora.toml'
LORA_PROFILE_ID = 'native-g0-qwen38-flash-next-v50'
# WeightLab (weightlab design note WL-01): the research `-v57`, the `-v50` deployment under its own id; its runtime
# config is `-v50`'s plus `[deployments.native.research]`, api_compat_anthropic, production's 900 s default deadline and
# the three WeightLab arm scopes. Dev only (GPU0); absent from every production and release list.
RESEARCH_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-research'
RESEARCH_TOML = 'configs/runtime/native-g0-spec-research.toml'
RESEARCH_PROFILE_ID = 'native-g0-qwen38-flash-next-v57'
# BeatSglang S7 (beat-sglang design note S7, owner OQ1): the opt-in `-v56`, the default `-v29` deployment (artifact,
# library, capabilities) under its own id, its runtime config the default's with two numerics-changing prefill levers,
# k2_gemm `small_m_a8` (W8A8 MXFP8 dense prefill GEMMs) and prefill_attention `p4`.
V56_PREFIX = 'configs/deployments/qwen38-flash-next-native-spec-v56'
V56_TOML = 'configs/runtime/native-g0-spec-v56.toml'
V56_PROFILE_ID = 'native-g0-qwen38-flash-next-v56'
# BatchConc BC2 (batch-concurrency design note BC2, owner Q1 (b), Q2 (a), Q8): the dev `-v58`, the `-v53`
# deployment under its own id at 16 decode rows; its runtime config is prod-t4.toml retargeted to the GPU0 lane
# (qe1_tickets.RETARGET). Production serves it since window 11 (BC7, owner 2026-11-11).
V58_PREFIX = 'configs/deployments/qwen38-flash-next-native-rows16-v58'
V58_TOML = 'configs/runtime/native-g0-rows16-v58.toml'
V58_PROFILE_ID = 'native-g0-qwen38-flash-next-v58'
# BatchConcurrency BC4 (batch-concurrency design note BC4): the opt-in `-v59`, the production Sessions `-v53`
# deployment (artifact, library, capabilities) under its own id, its runtime config `-v53`'s with MixedPD's
# `[deployments.native] prefill_slice_ms` 50 (mixed-pd design note; scheduling only).
V59_PREFIX = 'configs/deployments/qwen38-flash-next-native-sessions-v59'
V59_TOML = 'configs/runtime/native-g0-sessions-v59.toml'
V59_PROFILE_ID = 'native-g0-qwen38-flash-next-v59'
# FP4-Local FL-14 (fp4local design note "S2 quality plan"): the S2 eval configs, one per arm (base config, profile),
# each its base config with only `[api] max_output_tokens` 131,072 and `default_deadline_ms` = `max_deadline_ms`; the
# S3 one (`-v21`) is the FP4L arm of the default flip's tool-eval-bench pair.
SPEC_EVAL_TOML = 'configs/runtime/native-g0-spec-eval.toml'
SPEC_FP4LOCAL_EVAL_TOML = 'configs/runtime/native-g0-spec-fp4local-eval.toml'
S3_EVAL_TOML = 'configs/runtime/native-g0-spec-fp4local-s3-eval.toml'
EVAL_TOMLS = {SPEC_EVAL_TOML: (SPEC_TOML, SPEC_PROFILE_ID),
              SPEC_FP4LOCAL_EVAL_TOML: (SPEC_FP4LOCAL_TOML, SPEC_FP4LOCAL_PROFILE_ID),
              S3_EVAL_TOML: (S3_TOML, S3_PROFILE_ID)}
# g9-prod design note: the production T4 config, the 16-row `-v58` on GPU1 behind cere-t4.service (production
# window 11, batch-concurrency design note BC7); lock_native.py pins it with the `-v58` lock.
PROD_T4_TOML = 'configs/runtime/prod-t4.toml'
GPU1_UUID = 'GPU-bbbbbbbb-1111-4111-8111-111111111111'
# Roadmap v0.7 A2: production's `[[telemetry.clients]]` rules (deliberately prod-only; scripts/live/census.py reads
# their `cere_client_*` families), popped before a contract compares prod-t4.toml with its base.
PROD_T4_CLIENTS = [
    {'client': 'coding-agent', 'workload': 'interactive', 'scope': 'frontend', 'priority_header': True},
    {'client': 'judge', 'workload': 'batch-eval', 'scope': 'frontend', 'priority_header': False},
    {'client': 'axon', 'workload': 'eval-repair', 'scope': 'ae'},
]
# SelfTune ST4 S3d: DICE-8B's GPU0 dev deployment, pinned only by a DICE-8B build (build.json `flags.model` dice-8b).
DICE_PREFIX = 'configs/deployments/dice-8b-native'
DICE_TOML = 'configs/runtime/native-g0-dice8b.toml'
# SdarOptIn (O320): its opt-in `sdar_gemm = "changing"` twin, the same lock and library.
DICE_CHANGING_TOML = 'configs/runtime/native-g0-dice8b-gemm-changing.toml'
# Inkling S5: Inkling 2 Mini's GPU0 dev deployment, pinned only by an Inkling build (`flags.model` inkling-2-mini);
# since D-I-S9B its drafts-off lane (opt-in).
INKLING_PREFIX = 'configs/deployments/inkling-2-mini-native'
INKLING_PLAIN_TOML = 'configs/runtime/native-g0-inkling-plain.toml'
# Inkling S13: the plain lane's opt-in G7 twin (default output 4,096, its own ledger), the same lock and library.
INKLING_S13_TOML = 'configs/runtime/native-g0-inkling-s13.toml'
# Inkling S9B (inkling design note D-I-S9B): the default lane, MTP drafter on, its own lock on the same library.
INKLING_TOML = 'configs/runtime/native-g0-inkling.toml'
# Inkling D-I-C32: its opt-in batch twin (`-v69`, 32 rows), its own deployment and lock on the same library.
INKLING_BATCH_PREFIX = 'configs/deployments/inkling-2-mini-native-batch'
INKLING_BATCH_TOML = 'configs/runtime/native-g0-inkling-batch.toml'
# Every runtime config lock_native.py pins (its PAIRS and MODEL_PAIRS outside the deployment directory): a lock_native
# test root needs every one of them.
PINNED_TOMLS = (TOML, VISION_TOML, G1_TOML, P4_TOML, K2S_TOML, G0_PREFIX_TOML, G1_PREFIX_TOML, SPEC_TOML,
                SPEC_LOOKUP_TOML, SPEC_K2S_TOML, SPEC_FP8W_TOML, SPEC_FP4LOCAL_TOML, S3_TOML, DEFAULT_SPEC_TOML,
                SPEC_EVAL_TOML, SPEC_FP4LOCAL_EVAL_TOML, S3_EVAL_TOML, TREE_TOML, V22_TOML, CHAINS_TOML, V51_TOML,
                SESSIONS_TOML, VISION_LONG_TOML, VISION_V54_TOML, VISION_V55_TOML,
                TWO_GPU_TOML, PROD_T4_TOML, CTX262_TOML, V52_TOML, V53_TOML, MIXEDPD_TOML, GRAMMAR_TOML,
                LORA_TOML, V56_TOML, RESEARCH_TOML, V58_TOML, V59_TOML, DICE_TOML, DICE_CHANGING_TOML,
                INKLING_TOML, INKLING_S13_TOML, INKLING_PLAIN_TOML, INKLING_BATCH_TOML)

# C4 (m25 D6; ledger P1): the committed `-v2` files (main `39ae7d4`, relocked by L0). `-v3`/`-v4` equal them except the
# context fields, the profile id, the new `deployments.native.kv_pool_tokens` (= the context, m25 D4) and the lock's
# build/library SHA fields (`backend_build_sha256` and the TOML's `library_sha256`, relocked by `lock_native.py` after
# R1 to the `build/m25` library), plus the lock's `recipe_sha256`, which is the recipe file's own hash
# (pinned by test_lock_identity_equals_the_sglang_lock_and_the_pin), and `api.default_deadline_ms`, which may only
# rise after R1 measures (asserted never lower). M2.6 D9/D10 (`-v4`, ledger P1: `-v3` is M2.5's) adds the rows
# fields: the profile's `decode_batch_max`, `native_decode_batch`, `notes` and every combination's `row_counts`
# (asserted by test_capabilities_are_exactly_q15), and the TOML's `[admission]` (the rows are the profile's
# `decode_batch_max`, BC-D5); the pool stays 200,000 tokens, shared by the rows (ledger CR1).
# M2.7 D11 (`-v7`, ledger P1: after M2.12's `-v5` on `main` and M2.13's claimed `-v6`) adds the sampled decode rows,
# `max_top_logprobs`, `sampling_params` and rewrites `notes` (asserted by test_d11_sampled_profile). M2.7 Q3 (a) moves
# the recipe's `sampling_defaults` to the model's 1.0/0.95 (D10; asserted below).
PREV_PROFILE_ID = 'native-g0-qwen38-flash-next-v2'
PREV_CONTEXT = 73728
PREV_BUILD_SHA = 'd95f1278989dfd2f0ada9f74918973c62fb5ae45286ad5719eea5ae64001e761'
PREV_LOCK = {
    'schema_version': '1',
    'lock_id': 'qwen38-flash-next-native-lock-v1',
    'recipe_sha256': '57914fd3d945012f122eda7dac7a3e806df30e276967eeb0703900f150ee75a8',
    'artifact_sha256': 'b580d9bd2713aba7dc09be32a02e856ef73d1de1783e11a085b29bbbd4dc04e7',
    'model_content_sha256': '6be3034c56ff1acf2559a52556236d15be0b809d015815d177d37b5047aa461d',
    'tokenizer_sha256': '5cba89bb97f70720773d6148b3e86aa3f22e914acbad16930f559d107996405b',
    'template_sha256': 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041',
    'backend_build_sha256': PREV_BUILD_SHA,
    'backend_profile_id': PREV_PROFILE_ID,
    'effective_context_tokens': PREV_CONTEXT,
    'effective_operations': ['chat'],
    'adapter_sha256s': [],
}
PREV_RECIPE = {
    'schema_version': '1',
    'recipe_id': 'qwen38-flash-next-native',
    'model_ref': 'local://models/qwen38-flash-next/Qwen3.8-Flash-Next-NVFP4',
    'backend_profile_id': PREV_PROFILE_ID,
    'context_tokens': PREV_CONTEXT,
    'required_operations': ['chat'],
    'adapters': [],
    'residency': {'priority': 1000, 'keep_warm_seconds': 0, 'pinned': True},
    'sampling_defaults': {'temperature': 0.0, 'top_p': 1.0},
}
# canonical_json digests of the committed `-v2` capabilities and runtime TOML (`git show 39ae7d4:...`) with the excepted
# fields of `-v3`, `-v4`, `-v5`, `-v7` and `-v8` removed (`masked`), so `-v2` -> `-v3` -> `-v4` -> `-v5` -> `-v7` -> `-v8`
# changed nothing else (`-v5`'s M2.12 fields masked; `-v8`'s grammar fields reset to `-v2`'s none). The TOML one is
# tests/fixtures/config_pins.json `masked_toml_sha256` (`masked_toml_pin`), which `scripts/pin_configs.py --write`
# generates with `masked` (since BC-D5, `-v2`'s TOML without its two deleted `[scheduler]` keys), so the pin is read at test time, never at import.
PREV_MASKED_CAPABILITIES_SHA256 = '74acd83d695d6aa000190e9534cb5b98208a5bde20937a6e6f297284d9c651c3'
PREV_DEFAULT_DEADLINE_MS = 300_000
LOCK_EXCEPT = ('backend_profile_id', 'effective_context_tokens', 'backend_build_sha256', 'recipe_sha256')
RECIPE_EXCEPT = ('backend_profile_id', 'context_tokens', 'sampling_defaults')


def pop_path(config, path):
    """Removes `config[path[0]][path[1]]...` and returns its value (None when absent)."""
    *parents, key = path
    for p in parents:
        config = config[p]
    return config.pop(key, None)


def without(test, config, delta):
    """A copy of `config` without the keys of `delta`, after asserting each holds its value there."""
    config = json.loads(json.dumps(config))
    for path, value in delta.items():
        with test.subTest(delta=path):
            test.assertEqual(pop_path(config, path), value)
    return config


def shipped_toml(test):
    """native-g0.toml without OPERATOR_DELTA: the shape main's cache-off profiles derived from it share."""
    return without(test, tomllib.loads((ROOT / TOML).read_text()), OPERATOR_DELTA)


# Keys of native-g0-spec's [deployments.native] that BeatSglang flipped on the default after the opt-in profiles were
# derived from it (beat-sglang design note S5, OQ2/OQ3; S1 C4Sched G4): key -> (value before, value now; None =
# absent before).
DEFAULT_SPEC_FLIPS = {'prefix_state_slots': (32, 48), 'prefill_slice_ms': (None, 200), 'decode_share_pct': (None, 200),
                      'host_share_dir': (None, '/dev/shm/cere-host-g0'), 'keep_host_share': (None, True)}

# L8 (beat-sglang design note, maintainer 2026-11-01): the GPU0 dev speculation profiles (-v29, -v52, -v49, -v47)
# share their PLE region in a kept O47 segment under a GPU0 dev directory, never /dev/shm/cere-host; prod-t4 sets none.
L8_SHARE = {'host_share_dir': '/dev/shm/cere-host-g0', 'keep_host_share': True}


def pop_l8_share(test, config):
    """`config` (a parsed -v29, -v52, -v49 or -v47 runtime config) without L8_SHARE, after asserting it holds it."""
    native = config['deployments'][0]['native']
    test.assertEqual({k: native.pop(k, None) for k in L8_SHARE}, L8_SHARE)
    return config


def default_spec_before_flips(test, config):
    """`config` (the parsed native-g0-spec.toml) with DEFAULT_SPEC_FLIPS reverted, after asserting each holds its new
    value: the default the derived opt-in profiles still equal."""
    native = config['deployments'][0]['native']
    for key, (before, now) in DEFAULT_SPEC_FLIPS.items():
        with test.subTest(flip=key):
            test.assertEqual(native.pop(key, None), now)
        if before is not None:
            native[key] = before
    return config


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def masked_toml_pin():
    """native-g0.toml's pinned `canonical_sha256(masked('toml', ...))` (tests/fixtures/config_pins.json)."""
    return pin_configs.load()['masked_toml_sha256'][TOML]


def masked(kind, value):
    """`value` without the fields `-v3` (C4), `-v4` (M2.6 D9/D10), `-v5` (M2.12), `-v7` (M2.7 D11) and `-v8` (M2.8
    D10) may change, and M2.10's OPERATOR_DELTA (the `-v8` profile's operator route)."""
    value = json.loads(json.dumps(value))
    if kind == 'capabilities':
        del value['profile_id'], value['max_context_tokens']
        del value['decode_batch_max'], value['native_decode_batch'], value['notes']
        value.pop('max_top_logprobs', None)
        value.pop('sampling_params', None)
        value['validated_combinations'] = [c for c in value['validated_combinations']
                                           if c['sampler_mode'] != 'sampled' and c['output'] not in GRAMMAR_OUTPUTS]
        value['structured_outputs'] = [o for o in value['structured_outputs'] if o not in GRAMMAR_OUTPUTS]
        value['json_schema_dialect'], value['json_schema_keywords'] = None, []
        for key in M212_CAPABILITIES:
            value.pop(key, None)
        for c in value['validated_combinations']:
            del c['max_context_tokens'], c['row_counts']
    elif kind == 'toml':
        entry = value['deployments'][0]
        del entry['profile_id'], entry['native']['max_context_tokens'], entry['native']['library_sha256']
        entry['native'].pop('kv_pool_tokens', None)
        del value['api']['default_deadline_ms']
        # api-lenient design note AL-5 raised the cap from `-v2`'s 2,048 to 32,768: reset, like `-v8`'s grammar.
        value['api']['max_output_tokens'] = 2048
        del value['admission']
        for path in OPERATOR_DELTA:
            pop_path(value, path)
    else:
        for key in {'lock': LOCK_EXCEPT, 'recipe': RECIPE_EXCEPT}[kind]:
            del value[key]
    return value


def part(prefix, name, root=ROOT):
    return load_json(root / f'{prefix}.{name}.json')


def spec_rest_notes(test):
    """The `-v19` capabilities notes after SPEC_LEVERS_NOTE with PRE_PREFIX_NOTES back in place of PREFIX_NOTES: the
    lever-free, cache-off speculation notes `-v15` and `-v16` end with."""
    notes = part(SPEC_PREFIX, 'capabilities')['notes']
    test.assertTrue(notes.startswith(SPEC_LEVERS_NOTE))
    rest = notes.removeprefix(SPEC_LEVERS_NOTE)
    test.assertEqual(rest.count(PREFIX_NOTES), 1)
    test.assertNotIn(PRE_PREFIX_NOTES, rest)
    return rest.replace(PREFIX_NOTES, PRE_PREFIX_NOTES)


def lever_free_spec_toml(test):
    """native-g0-spec.toml without its levers table (asserted SPEC_LEVERS) and PREFIX_TOML_DELTA (asserted): the
    lever-free, cache-off speculation config `-v15` and `-v16` derive from."""
    spec = without(test, tomllib.loads((ROOT / SPEC_TOML).read_text()), PREFIX_TOML_DELTA)
    test.assertEqual(spec['deployments'][0]['native'].pop('levers'), SPEC_LEVERS)
    return spec


def build_json(dirty=False, outputs=None, ccache='off'):
    if outputs is None:
        outputs = [{'file': 'libcere_executor.so', 'sha256': LIBRARY_SHA, 'bytes': 1, 'needed': []},
                   {'file': 'libcere_ktest.so', 'sha256': 'b' * 64, 'bytes': 1, 'needed': []}]
    return {'schema_version': '1', 'git_head': 'c' * 40, 'git_dirty': dirty, 'outputs': outputs,
            'image_digest': 'sha256:' + 'd' * 64, 'flags': P3B_BUILD_FLAGS, 'ccache': ccache}


class NativeDeployment(unittest.TestCase):
    def test_deployment_validates(self):
        d = validate_deployment(ROOT, PREFIX)
        self.assertEqual(d['profile']['profile_id'], PROFILE_ID)
        self.assertEqual(d['lock']['effective_context_tokens'], CONTEXT)
        self.assertEqual(d['lock']['backend_profile_id'], PROFILE_ID)
        self.assertEqual(d['recipe']['backend_profile_id'], PROFILE_ID)
        self.assertEqual(d['recipe']['context_tokens'], CONTEXT)

    def test_lock_identity_equals_the_sglang_lock_and_the_pin(self):
        lock, sglang = part(PREFIX, 'lock'), part(SGLANG_PREFIX, 'lock')
        for key in ('model_content_sha256', 'tokenizer_sha256', 'template_sha256'):
            with self.subTest(key=key):
                self.assertEqual(lock[key], sglang[key])
        pin = load_json(ROOT / PIN)
        self.assertEqual(lock['artifact_sha256'], pin['manifest_sha256'])
        self.assertEqual(part(PREFIX, 'capabilities')['artifact_sha256'], pin['manifest_sha256'])
        self.assertEqual(lock['recipe_sha256'], hashlib.sha256((ROOT / f'{PREFIX}.recipe.json').read_bytes()).hexdigest())

    def test_capabilities_are_exactly_q15(self):
        c = part(PREFIX, 'capabilities')
        expected = {
            'profile_id': PROFILE_ID, 'execution_mode': 'native', 'evidence': 'measured',
            'operations': ['prefill', 'decode', 'cancel'], 'decode_batch_min': 1, 'decode_batch_max': 16,
            'max_context_tokens': CONTEXT, 'native_decode_batch': True, 'native_mixed_prefill_decode': False,
            'native_speculative_batch': False, 'native_moe_grouping': False, 'native_state_snapshot': False,
            'structured_outputs': ['text', 'tool_calls', *GRAMMAR_OUTPUTS],
            'json_schema_dialect': 'json-schema-2020-12/xgrammar-0.2.1',
            'numerical_modes': ['optimized'], 'checkpoint_mode': 'none', 'modalities': ['text'],
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(c[key], value)
        combinations = {(x['operation'], x['modality'], x['output'], x['numerical_mode'], x['sampler_mode'],
                         tuple(x['row_counts']), x['max_context_tokens']) for x in c['validated_combinations']}
        self.assertEqual(len(combinations), len(c['validated_combinations']))
        self.assertEqual(combinations, {
            ('prefill', 'text', 'text', 'optimized', 'not_applicable', (1,), CONTEXT),
            ('decode', 'text', 'text', 'optimized', 'greedy', ROWS, CONTEXT),
            ('decode', 'text', 'text', 'optimized', 'sampled', ROWS, CONTEXT),
            ('decode', 'text', 'tool_calls', 'optimized', 'greedy', ROWS, CONTEXT),
            ('decode', 'text', 'tool_calls', 'optimized', 'sampled', ROWS, CONTEXT),
            *(('decode', 'text', output, 'optimized', sampler, ROWS, CONTEXT)
              for output in GRAMMAR_OUTPUTS for sampler in ('greedy', 'sampled')),
            ('cancel', 'text', 'text', 'optimized', 'not_applicable', (1,), CONTEXT),
        })
        # M2.6 D10: rows are concurrently decoded sessions, bit-identical to one row; prefill one chunk per round.
        for words in ('deterministic at rows = 1 without cache', 'NVFP4 W4A4', 'fp8_e4m3 KV', 'flashinfer_cutlass',
                      'concurrently decoded sessions in one batched decode step', 'bit-identical',
                      'prefill is serialized, one 4,096-token chunk'):
            self.assertIn(words.lower(), c['notes'].lower())
        # D6: every combination carries the profile's context, M2.4's tool_calls included.
        self.assertEqual({x['max_context_tokens'] for x in c['validated_combinations']}, {CONTEXT})

    def test_d11_sampled_profile(self):
        """M2.7 D11: sampled decode rows, `max_top_logprobs: 20`, the seven `sampling_params`, and notes that name
        the device sampler, its start check and the missing SGLang parity claim; the id bumps per ledger P1."""
        c = part(PREFIX, 'capabilities')
        self.assertEqual(c['max_top_logprobs'], 20)
        self.assertEqual(c['sampling_params'], ['top_k', 'min_p', 'seed', 'frequency_penalty', 'presence_penalty',
                                                'repetition_penalty', 'logit_bias'])
        self.assertEqual(V7_PROFILE_ID, 'native-g0-qwen38-flash-next-v7')
        self.assertNotEqual(PROFILE_ID, V7_PROFILE_ID)
        for words in ('device sampler', 'native_sampler 1, checked at start', 'not an sglang parity claim',
                      'logprobs with top_logprobs up to 20', 'its own sampling, output counts and counter key',
                      'distinct per-choice seed'):
            self.assertIn(words, c['notes'].lower())
        self.assertNotIn('Sampled decoding, logprobs', c['notes'])
        # Ledger CR4: the interim (a sampled session decoding alone, greedy-only n) is gone.
        for words in ('decodes alone', 'sampled row_counts [1]', '(greedy, n <= 4'):
            self.assertNotIn(words, c['notes'])

    def test_d10_grammar_profile(self):
        """M2.8 D10 (C3): the four grammar outputs with the SGLang profile's dialect and keywords, greedy and sampled
        decode rows 1-16 for each (CR5: a row waits for its mask; 1-4 before tasks/m23-spec.md P8), none with `verify` (I3); the dev snapshot profile
        carries the same grammar set at row 1; the id is `-v8` (ledger P1)."""
        self.assertEqual(PROFILE_ID, 'native-g0-qwen38-flash-next-v8')
        c, sglang = part(PREFIX, 'capabilities'), part(SGLANG_PREFIX, 'capabilities')
        self.assertEqual(c['structured_outputs'], sglang['structured_outputs'])
        self.assertEqual(c['structured_outputs'], ['text', 'tool_calls', *GRAMMAR_OUTPUTS])
        self.assertEqual(c['json_schema_dialect'], 'json-schema-2020-12/xgrammar-0.2.1')
        self.assertEqual(c['json_schema_dialect'], sglang['json_schema_dialect'])
        self.assertEqual(c['json_schema_keywords'], sglang['json_schema_keywords'])
        self.assertNotIn('verify', c['operations'])
        for prefix, rows, profile_id in ((PREFIX, list(ROWS), PROFILE_ID), (DEV_PREFIX, [1], DEV_PROFILE_ID)):
            caps = part(prefix, 'capabilities')
            self.assertEqual(caps['profile_id'], profile_id)
            self.assertEqual({k: caps[k] for k in ('structured_outputs', 'json_schema_dialect', 'json_schema_keywords')},
                             {k: c[k] for k in ('structured_outputs', 'json_schema_dialect', 'json_schema_keywords')})
            grammar = [x for x in caps['validated_combinations'] if x['output'] in GRAMMAR_OUTPUTS]
            self.assertEqual(sorted((x['output'], x['sampler_mode']) for x in grammar),
                             sorted((o, s) for o in GRAMMAR_OUTPUTS for s in ('greedy', 'sampled')))
            for x in grammar:
                with self.subTest(prefix=prefix, output=x['output'], sampler=x['sampler_mode']):
                    self.assertEqual((x['operation'], x['modality'], x['numerical_mode'], x['row_counts'],
                                      x['max_context_tokens']), ('decode', 'text', 'optimized', rows, CONTEXT))
            self.assertNotIn('verify', {x['operation'] for x in caps['validated_combinations']})
            for words in ('grammar process (xgrammar 0.2.1, sidecar --role grammar)', 'per-position vocab masks',
                          'after penalties and before logit_bias', 'constrained tickets never speculate',
                          'one constraint per request', 'not post-validated', 'native_token_mask 1'):
                self.assertIn(words, caps['notes'])
            self.assertNotIn('tool_choice required/named, json_schema, verify', caps['notes'])

    def test_runtime_config_binds_18309_with_bench_style_tokens(self):
        x = tomllib.loads((ROOT / TOML).read_text())
        validate('runtime_config', x)
        self.assertEqual(x['server']['bind'], '127.0.0.1:18309')
        self.assertEqual(x['deployments'][0]['deployment'], PREFIX)
        self.assertEqual(x['deployments'][0]['profile_id'], PROFILE_ID)
        self.assertEqual([(s['scope_id'], s['token_env']) for s in x['auth']['scopes']],
                         [('native-a', 'CERE_TOKEN_NATIVE_A'), ('native-b', 'CERE_TOKEN_NATIVE_B')])
        self.assertEqual(x['ledger']['state_dir'], '/var/lib/cere/native-g0')
        native = x['deployments'][0]['native']
        lock = part(PREFIX, 'lock')
        self.assertEqual(native['library_sha256'], lock['backend_build_sha256'])
        self.assertEqual(native['expected_manifest_sha256'], lock['artifact_sha256'])
        self.assertEqual(native['artifact_dir'], load_json(ROOT / PIN)['path'])
        self.assertEqual(native['max_context_tokens'], CONTEXT)
        # m25 D4/D6, ledger CR1: one 200,000-token pool (= the context) shared by the profile's 4 rows.
        self.assertEqual(native['kv_pool_tokens'], CONTEXT)
        # ApiLenient AL-5 (D7): the output cap is omp's 32,768; the comment above [api] names the served context.
        self.assertEqual(x['api']['max_output_tokens'], 32768)
        text = (ROOT / TOML).read_text()
        self.assertIn('200,000-token context', text)
        self.assertNotIn('73,728-token context', text)

    def test_v8_differs_from_the_committed_v2_only_in_context_profile_pool_sampler_grammar_and_build_pins(self):
        """C4 (m25 D6), M2.7 D11/Q3 and M2.8 D10: the `-v2` values are held here; everything else is unchanged."""
        lock, recipe = part(PREFIX, 'lock'), part(PREFIX, 'recipe')
        self.assertEqual(masked('lock', lock), masked('lock', PREV_LOCK))
        self.assertEqual(masked('recipe', recipe), masked('recipe', PREV_RECIPE))
        # M2.7 Q3 (a): the recipe states native's effective default (D10, `generation_config.json`).
        self.assertEqual(recipe['sampling_defaults'], {'temperature': 1.0, 'top_p': 0.95})
        self.assertEqual(canonical_sha256(masked('capabilities', part(PREFIX, 'capabilities'))),
                         PREV_MASKED_CAPABILITIES_SHA256)
        x = tomllib.loads((ROOT / TOML).read_text())
        self.assertEqual(canonical_sha256(masked('toml', x)), masked_toml_pin())
        # The excepted fields moved from their `-v2` values to `-v8` ones.
        self.assertEqual((lock['backend_profile_id'], lock['effective_context_tokens']), (PROFILE_ID, CONTEXT))
        self.assertEqual((recipe['backend_profile_id'], recipe['context_tokens']), (PROFILE_ID, CONTEXT))
        self.assertNotEqual(PROFILE_ID, PREV_PROFILE_ID)
        self.assertNotEqual(PROFILE_ID, V4_PROFILE_ID)
        self.assertNotEqual(PROFILE_ID, V5_PROFILE_ID)
        self.assertNotEqual(PROFILE_ID, V7_PROFILE_ID)
        # M2.10 (D9C): the shipped `-v8` runs cache off (no slots, no isolation) and gained only the operator route.
        for path in PREFIX_TOML_DELTA:
            self.assertIsNone(pop_path(json.loads(json.dumps(x)), path), path)
        without(self, x, OPERATOR_DELTA)
        self.assertIn(PRE_PREFIX_NOTES, part(PREFIX, 'capabilities')['notes'])
        self.assertEqual(x['deployments'][0]['native']['kv_pool_tokens'], CONTEXT)
        self.assertEqual(x['deployments'][0]['native']['library_sha256'], lock['backend_build_sha256'])
        # R1 relocked (lock_native.py after the GPU build): the lock and the TOML pin a library other than `-v2`'s.
        self.assertNotEqual(lock['backend_build_sha256'], PREV_BUILD_SHA)
        self.assertGreaterEqual(x['api']['default_deadline_ms'], PREV_DEFAULT_DEADLINE_MS)

    def test_v4_declares_the_m212_controls(self):
        """M2.12 A1 (D2, Q6 b): thinking (native context 200,000 >= 73,728), `stop_token_ids` and `ignore_eos` (the D9
        library); `min_tokens` waits on M2.7; `vocab_size` is the model's `text_config.vocab_size` (D5)."""
        c = part(PREFIX, 'capabilities')
        for key, value in M212_CAPABILITIES.items():
            with self.subTest(key=key):
                self.assertEqual(c.get(key), value)
        self.assertGreaterEqual(c['max_context_tokens'], 73728)
        self.assertNotIn('min_tokens', c['token_controls'])

    def test_a_d6_deadline_raise_keeps_c3(self):
        """Z2AC-14: raising `api.default_deadline_ms` (D6, M22CS-12) is an excepted field, not a C3 failure."""
        x = tomllib.loads((ROOT / TOML).read_text())
        x['api']['default_deadline_ms'] = 2 * PREV_DEFAULT_DEADLINE_MS
        self.assertEqual(canonical_sha256(masked('toml', x)), masked_toml_pin())


def combination_set(capabilities):
    return {(x['operation'], x['modality'], x['output'], x['numerical_mode'], x['sampler_mode'],
             tuple(x['row_counts']), x['max_context_tokens']) for x in capabilities['validated_combinations']}


class NativeVisionDeployment(unittest.TestCase):
    """tasks/m213-spec.md D5, D9, Q2 b): `native-g0-vision.toml` serves the vision artifact (relabel to
    `rtxpro6000bw-plehost-vision-v1`) at 2,048 tokens and one row (D14, ledger CR19: native image rows 1) with one
    greedy image decode combination; everything else is native-g0's."""

    def test_deployment_validates(self):
        d = validate_deployment(ROOT, VISION_PREFIX)
        self.assertEqual(d['profile']['profile_id'], VISION_PROFILE_ID)
        self.assertEqual((d['lock']['backend_profile_id'], d['lock']['effective_context_tokens']),
                         (VISION_PROFILE_ID, VISION_CONTEXT))
        self.assertEqual((d['recipe']['backend_profile_id'], d['recipe']['context_tokens']),
                         (VISION_PROFILE_ID, VISION_CONTEXT))

    def test_lock_pins_the_vision_artifact_and_the_text_identity(self):
        lock, text = part(VISION_PREFIX, 'lock'), part(PREFIX, 'lock')
        self.assertEqual(lock['lock_id'], 'qwen38-flash-next-native-vision-lock-v1')
        for key in ('model_content_sha256', 'tokenizer_sha256', 'template_sha256', 'effective_operations',
                    'adapter_sha256s', 'schema_version'):
            with self.subTest(key=key):
                self.assertEqual(lock[key], text[key])
        pin = load_json(ROOT / VISION_PIN)
        self.assertEqual(pin['profile_id'], 'rtxpro6000bw-plehost-vision-v1')
        self.assertEqual(lock['artifact_sha256'], pin['manifest_sha256'])
        self.assertNotEqual(lock['artifact_sha256'], text['artifact_sha256'])
        self.assertEqual(part(VISION_PREFIX, 'capabilities')['artifact_sha256'], pin['manifest_sha256'])
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{VISION_PREFIX}.recipe.json').read_bytes()).hexdigest())
        recipe = part(VISION_PREFIX, 'recipe')
        self.assertEqual(recipe['recipe_id'], 'qwen38-flash-next-native-vision')
        self.assertEqual({k: v for k, v in recipe.items() if k not in ('recipe_id', *RECIPE_EXCEPT)},
                         {k: v for k, v in part(PREFIX, 'recipe').items() if k not in ('recipe_id', *RECIPE_EXCEPT)})

    def test_capabilities_are_native_g0_at_2048_plus_one_image_combination(self):
        c, text = part(VISION_PREFIX, 'capabilities'), part(PREFIX, 'capabilities')
        self.assertEqual(c['modalities'], ['text', 'image'])
        self.assertEqual(c['max_context_tokens'], VISION_CONTEXT)
        same = ('schema_version', 'plugin', 'execution_mode', 'evidence', 'operations', 'decode_batch_min',
                'native_mixed_prefill_decode', 'native_speculative_batch',
                'native_moe_grouping', 'native_state_snapshot', 'numerical_modes', 'checkpoint_mode')
        for key in same:
            with self.subTest(key=key):
                self.assertEqual(c[key], text[key])
        # M2.8 grammar (m28 design note D10) is native-g0's only: the vision profile declares no grammar output.
        self.assertEqual(c['structured_outputs'], [o for o in text['structured_outputs'] if o not in GRAMMAR_OUTPUTS])
        self.assertEqual((c.get('json_schema_dialect'), c.get('json_schema_keywords') or []), (None, []))
        self.assertEqual((c['native_decode_batch'], c['decode_batch_max']), (False, 1))
        self.assertEqual(set(c), set(text))
        text_at_2048 = {x[:5] + ((1,), VISION_CONTEXT) for x in combination_set(text) if x[2] not in GRAMMAR_OUTPUTS}
        self.assertEqual(len(combination_set(c)), len(c['validated_combinations']))
        self.assertEqual(combination_set(c), text_at_2048 | {IMAGE_COMBINATION})
        self.assertIn('image', c['notes'].lower())
        # native-g0 stays text-only (Q2 b).
        self.assertEqual(text['modalities'], ['text'])
        self.assertNotIn('image', {x[1] for x in combination_set(text)})

    def test_runtime_config_is_native_g0_on_the_vision_deployment(self):
        x = tomllib.loads((ROOT / VISION_TOML).read_text())
        validate('runtime_config', x)
        entry, native = x['deployments'][0], x['deployments'][0]['native']
        self.assertEqual((entry['deployment'], entry['profile_id']), (VISION_PREFIX, VISION_PROFILE_ID))
        self.assertEqual(x['ledger']['state_dir'], '/var/lib/cere/native-g0-vision')
        lock, pin = part(VISION_PREFIX, 'lock'), load_json(ROOT / VISION_PIN)
        self.assertEqual(native['library_sha256'], lock['backend_build_sha256'])
        self.assertEqual(native['expected_manifest_sha256'], lock['artifact_sha256'])
        self.assertEqual(native['artifact_dir'], pin['path'])
        self.assertEqual((native['max_context_tokens'], native['kv_pool_tokens']), (VISION_CONTEXT, VISION_CONTEXT))
        self.assertEqual(native['sidecar'], ['python3', '/src/scripts/native/tokenizer_sidecar.py', '--artifact',
                                             pin['path'], '--feature', 'image'])
        text = tomllib.loads((ROOT / TOML).read_text())
        self.assertEqual(text['deployments'][0]['native']['sidecar'][-2:], ['--artifact', load_json(ROOT / PIN)['path']])
        # One row (D14): one decode row (the profile's decode_batch_max 1, above), one request in flight.
        self.assertEqual((x['admission']['max_backend_inflight'], x['admission']['per_scope_inflight']), (1, 1))
        # Every other field is native-g0's (same bind and container: one GPU0 backend at a time).
        def rest(config):
            config = json.loads(json.dumps(config))
            e = config['deployments'][0]
            del e['deployment'], e['profile_id'], config['ledger']['state_dir']
            # The cap is the 2,048-token context here; native-g0 serves 32,768 (api-lenient design note AL-5).
            del config['api']['max_output_tokens']
            del config['admission']['max_backend_inflight'], config['admission']['per_scope_inflight']
            for key in ('library_sha256', 'expected_manifest_sha256', 'artifact_dir', 'max_context_tokens',
                        'kv_pool_tokens', 'sidecar'):
                del e['native'][key]
            return config
        self.assertEqual(rest(x), rest(shipped_toml(self)))
        self.assertIn('2,048-token context', (ROOT / VISION_TOML).read_text())

    def test_lock_native_pins_the_vision_pair_with_the_shipped_one(self):
        """lock_native.py pins one library into every native pair (PAIRS), the vision lock and TOML included; every
        other byte of the vision lock is kept."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel in ('schemas', 'configs/deployments'):
            shutil.copytree(ROOT / rel, root / rel)
        (root / 'configs/runtime').mkdir()
        for rel in PINNED_TOMLS:
            shutil.copy(ROOT / rel, root / rel)
        (root / 'build.json').write_text(json.dumps(build_json()))
        r = subprocess.run([sys.executable, str(LOCK_NATIVE), '--root', str(root), '--build-json',
                            str(root / 'build.json')], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        for prefix, toml in ((PREFIX, TOML), (VISION_PREFIX, VISION_TOML)):
            with self.subTest(toml=toml):
                self.assertEqual(part(prefix, 'lock', root)['backend_build_sha256'], LIBRARY_SHA)
                x = tomllib.loads((root / toml).read_text())
                self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
                validate('runtime_config', x, root)
        before = part(VISION_PREFIX, 'lock')
        self.assertEqual({k: v for k, v in part(VISION_PREFIX, 'lock', root).items() if k != 'backend_build_sha256'},
                         {k: v for k, v in before.items() if k != 'backend_build_sha256'})


class SnapshotDevDeployment(unittest.TestCase):
    """tasks/m211-spec.md D5, D8, D13, LR5: the dev/live profile is the shipped one plus snapshot/restore, and its
    runtime config the shipped one plus a 4 GiB arena, preempt and one parked session (inflight 2 = rows 1 + parked
    1). `preempt_every` is set per row by the runners (R3 64, R4 16), so it is absent here."""

    def test_parts_differ_from_the_shipped_ones_only_in_profile_and_snapshot(self):
        d = validate_deployment(ROOT, DEV_PREFIX)
        self.assertEqual(d['profile']['profile_id'], DEV_PROFILE_ID)
        caps, shipped = part(DEV_PREFIX, 'capabilities'), part(PREFIX, 'capabilities')
        self.assertTrue(caps['native_state_snapshot'])
        self.assertEqual(caps['operations'], ['prefill', 'decode', 'snapshot', 'restore', 'cancel'])
        self.assertEqual(caps['checkpoint_mode'], 'none')  # OQ6
        self.assertIn('not shipped', caps['notes'])
        # M2.11 x M2.6: U1 preemption needed decode_batch_max 1 when this profile was cut (sessions design note
        # SE-09 lifted the rule), so the dev profile serves one row of the shipped four: rows 1, no batched decode,
        # row_counts [1].
        self.assertEqual((caps['decode_batch_max'], caps['native_decode_batch']), (1, False))
        self.assertEqual([c['row_counts'] for c in caps['validated_combinations']],
                         [[1] for _ in shipped['validated_combinations']])
        changed = ('profile_id', 'native_state_snapshot', 'operations', 'notes', 'decode_batch_max',
                   'native_decode_batch', 'validated_combinations')
        self.assertEqual({k: v for k, v in caps.items() if k not in changed},
                         {k: v for k, v in shipped.items() if k not in changed})
        self.assertEqual([{k: v for k, v in c.items() if k != 'row_counts'} for c in caps['validated_combinations']],
                         [{k: v for k, v in c.items() if k != 'row_counts'} for c in shipped['validated_combinations']])
        recipe, lock = part(DEV_PREFIX, 'recipe'), part(DEV_PREFIX, 'lock')
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (DEV_PROFILE_ID, DEV_PROFILE_ID))
        self.assertEqual({k: v for k, v in recipe.items() if k not in ('recipe_id', 'backend_profile_id')},
                         {k: v for k, v in part(PREFIX, 'recipe').items() if k not in ('recipe_id', 'backend_profile_id')})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{DEV_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_ = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_},
                         {k: v for k, v in part(PREFIX, 'lock').items() if k not in except_})

    def test_runtime_config_is_the_shipped_one_plus_the_4_gib_arena_and_preempt(self):
        self.assertFalse((ROOT / 'configs/runtime' / Path(DEV_TOML).name).exists())
        x = tomllib.loads((ROOT / DEV_TOML).read_text())
        validate('runtime_config', x)
        entry, native = x['deployments'][0], x['deployments'][0]['native']
        self.assertEqual((entry['deployment'], entry['profile_id']), (DEV_PREFIX, DEV_PROFILE_ID))
        self.assertEqual((native['snapshot_host_bytes'], native['preempt'], native['max_parked_sessions']),
                         (DEV_ARENA, True, 1))
        self.assertNotIn('preempt_every', native)
        self.assertTrue(x['state']['host_spill_enabled'])
        self.assertEqual((x['admission']['max_backend_inflight'], x['admission']['per_scope_inflight']), (2, 1))
        self.assertEqual(part(DEV_PREFIX, 'capabilities')['decode_batch_max'], 1)  # its one row (BC-D5)
        shipped = shipped_toml(self)
        self.assertEqual(native['library_sha256'], shipped['deployments'][0]['native']['library_sha256'])
        for doc in (x, shipped):
            doc['deployments'][0].pop('deployment'), doc['deployments'][0].pop('profile_id')
            for key in ('snapshot_host_bytes', 'preempt', 'max_parked_sessions'):
                doc['deployments'][0]['native'].pop(key, None)
            doc['state'].pop('host_spill_enabled'), doc['admission'].pop('max_backend_inflight')
            doc['admission'].pop('per_scope_inflight')
        self.assertEqual(x, shipped)

    def test_lock_native_pins_the_dev_pair_with_the_shipped_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ('schemas', 'configs/deployments'):
                shutil.copytree(ROOT / rel, root / rel)
            (root / 'configs/runtime').mkdir()
            for rel in PINNED_TOMLS:
                shutil.copy(ROOT / rel, root / rel)
            path = root / 'build.json'
            path.write_text(json.dumps(build_json()))
            r = subprocess.run([sys.executable, str(LOCK_NATIVE), '--root', str(root), '--build-json', str(path)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(part(DEV_PREFIX, 'lock', root)['backend_build_sha256'], LIBRARY_SHA)
            x = tomllib.loads((root / DEV_TOML).read_text())
            self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
            validate('runtime_config', x, root)


class P3bShipped(unittest.TestCase):
    """M23-D7: P3b is the shipped backend through the D9C-compatible identity gate; the `-v9` opt-in deployment is
    retired (no second profile id for the same library and capabilities) and the shipped profile ids stay `-v8`."""

    def test_the_v9_deployment_is_retired(self):
        for rel in RETIRED_P3B:
            self.assertFalse((ROOT / rel).exists(), rel)

    def test_the_shipped_profile_is_p3b(self):
        caps = part(PREFIX, 'capabilities')
        self.assertEqual(caps['profile_id'], PROFILE_ID)
        for prefix in (PREFIX, G1_PREFIX, VISION_PREFIX, DEV_PREFIX):
            with self.subTest(prefix=prefix):
                self.assertTrue(part(prefix, 'capabilities')['notes'].startswith(P3B_NOTE))


class LeverProfiles(unittest.TestCase):
    """tasks/m23-spec.md PI-02 (build/perf-int): the K2 small-M kernel (P11R-07, `-v14`) and P4 prefill attention
    (M2.3 P4, `-v12`) are selected by a deployment's `[deployments.native.levers]`, never by a build: each profile is
    the shipped `-v8` one on the shipped library plus exactly its one lever (and its ids); the shipped profile keeps
    every lever at the registry default."""

    PROFILES = ((K2S_PREFIX, K2S_TOML, K2S_PROFILE_ID, 'k2s', {'k2_gemm': 'small_m'}, 'levers.k2_gemm small_m'),
                (P4_PREFIX, P4_TOML, P4_PROFILE_ID, 'p4', {'prefill_attention': 'p4'},
                 'levers.prefill_attention p4'))

    def test_parts_are_the_shipped_ones_with_the_lever_profile(self):
        for prefix, _, profile_id, name, _, note in self.PROFILES:
            with self.subTest(profile=profile_id):
                d = validate_deployment(ROOT, prefix)
                self.assertEqual(d['profile']['profile_id'], profile_id)
                caps, shipped = part(prefix, 'capabilities'), part(PREFIX, 'capabilities')
                self.assertIn(note, caps['notes'])
                self.assertNotIn('CERE_K2_SMALL_M', caps['notes'])
                self.assertNotIn('CERE_PREFILL_ATTENTION', caps['notes'])
                self.assertEqual(caps['decode_batch_max'], len(PRE_P8_ROWS))
                self.assertEqual({k: v for k, v in rows_masked('capabilities', caps).items()
                                  if k not in ('profile_id', 'notes')},
                                 {k: v for k, v in rows_masked('capabilities', shipped).items()
                                  if k not in ('profile_id', 'notes')})
                recipe, lock = part(prefix, 'recipe'), part(prefix, 'lock')
                self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                                 (f'qwen38-flash-next-native-{name}', f'qwen38-flash-next-native-{name}-lock-v1'))
                self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (profile_id, profile_id))
                self.assertEqual({k: v for k, v in recipe.items() if k not in ('recipe_id', 'backend_profile_id')},
                                 {k: v for k, v in part(PREFIX, 'recipe').items()
                                  if k not in ('recipe_id', 'backend_profile_id')})
                self.assertEqual(lock['recipe_sha256'],
                                 hashlib.sha256((ROOT / f'{prefix}.recipe.json').read_bytes()).hexdigest())
                except_ = ('lock_id', 'backend_profile_id', 'recipe_sha256')
                self.assertEqual({k: v for k, v in lock.items() if k not in except_},
                                 {k: v for k, v in part(PREFIX, 'lock').items() if k not in except_})

    def test_runtime_config_is_the_shipped_one_plus_its_lever(self):
        shipped = shipped_toml(self)
        self.assertNotIn('levers', shipped['deployments'][0]['native'])
        for prefix, toml, profile_id, _, levers, _ in self.PROFILES:
            with self.subTest(profile=profile_id):
                x = tomllib.loads((ROOT / toml).read_text())
                validate('runtime_config', x)
                entry = x['deployments'][0]
                self.assertEqual((entry['deployment'], entry['profile_id']), (prefix, profile_id))
                self.assertEqual(entry['native']['library_sha256'], part(prefix, 'lock')['backend_build_sha256'])
                self.assertEqual(entry['native']['library_sha256'],
                                 shipped['deployments'][0]['native']['library_sha256'])
                self.assertEqual(entry['native'].pop('levers'), levers)
                base = copy.deepcopy(shipped)
                for doc in (x, base):
                    e = doc['deployments'][0]
                    del e['deployment'], e['profile_id']
                self.assertEqual((x['admission']['max_backend_inflight'], x['admission']['per_scope_inflight']),
                                 (4, 4))
                self.assertEqual(rows_masked('toml', x), rows_masked('toml', base))


class GpuLaneDeployment(unittest.TestCase):
    """The GPU1 lane's profile (native_backend.py --gpu 1): the GPU0 profile with only the lane's identity changed,
    so a GPU1 functional row runs the same served contract; lock_native pins its library with the GPU0 pair."""

    def test_runtime_config_is_native_g0_except_the_lane_identity(self):
        g1 = tomllib.loads((ROOT / G1_TOML).read_text())
        g0 = shipped_toml(self)
        validate('runtime_config', g1)
        entry, native = g1['deployments'][0], g1['deployments'][0]['native']
        self.assertEqual(g1['server']['bind'], '127.0.0.1:18329')
        self.assertEqual((entry['profile_id'], entry['deployment']), (G1_PROFILE_ID, G1_PREFIX))
        self.assertEqual(native['device_uuid'], GPU1_UUID)
        self.assertEqual(g0['deployments'][0]['native']['device_uuid'], GPU0_UUID)
        self.assertEqual(g1['scheduler']['devices'], ['gpu1'])
        self.assertEqual(g1['ledger']['state_dir'], '/var/lib/cere/native-g1')
        # never native-g0's port or state_dir, never sglang-t4's 18080
        self.assertNotIn(g1['server']['bind'].rsplit(':', 1)[1], ('18080', g0['server']['bind'].rsplit(':', 1)[1]))
        self.assertNotEqual(g1['ledger']['state_dir'], g0['ledger']['state_dir'])
        for doc in (g1, g0):
            del doc['server']['bind'], doc['scheduler']['devices'], doc['ledger']['state_dir']
            e = doc['deployments'][0]
            del e['profile_id'], e['deployment'], e['native']['device_uuid']
        self.assertEqual(g1, g0)

    def test_deployment_is_the_shipped_one_except_its_ids(self):
        validate_deployment(ROOT, G1_PREFIX)
        recipe, shipped_recipe = part(G1_PREFIX, 'recipe'), part(PREFIX, 'recipe')
        self.assertEqual((recipe.pop('recipe_id'), recipe.pop('backend_profile_id')),
                         ('qwen38-flash-next-native-g1', G1_PROFILE_ID))
        del shipped_recipe['recipe_id'], shipped_recipe['backend_profile_id']
        self.assertEqual(recipe, shipped_recipe)
        lock, shipped_lock = part(G1_PREFIX, 'lock'), part(PREFIX, 'lock')
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{G1_PREFIX}.recipe.json').read_bytes()).hexdigest())
        self.assertEqual((lock.pop('lock_id'), lock.pop('backend_profile_id')),
                         ('qwen38-flash-next-native-g1-lock-v1', G1_PROFILE_ID))
        for doc in (lock, shipped_lock):
            doc.pop('recipe_sha256')
        del shipped_lock['lock_id'], shipped_lock['backend_profile_id']
        self.assertEqual(lock, shipped_lock)
        caps, shipped_caps = part(G1_PREFIX, 'capabilities'), part(PREFIX, 'capabilities')
        self.assertEqual(caps.pop('profile_id'), G1_PROFILE_ID)
        del shipped_caps['profile_id']
        self.assertEqual(caps, shipped_caps)
        g1 = tomllib.loads((ROOT / G1_TOML).read_text())
        self.assertEqual(g1['deployments'][0]['native']['library_sha256'], part(G1_PREFIX, 'lock')['backend_build_sha256'])

    def test_lock_native_pins_the_gpu1_pair_with_the_shipped_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ('schemas', 'configs/deployments'):
                shutil.copytree(ROOT / rel, root / rel)
            (root / 'configs/runtime').mkdir()
            for rel in PINNED_TOMLS:
                shutil.copy(ROOT / rel, root / rel)
            path = root / 'build.json'
            path.write_text(json.dumps(build_json()))
            r = subprocess.run([sys.executable, str(LOCK_NATIVE), '--root', str(root), '--build-json', str(path)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(part(G1_PREFIX, 'lock', root)['backend_build_sha256'], LIBRARY_SHA)
            x = tomllib.loads((root / G1_TOML).read_text())
            self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
            validate('runtime_config', x, root)


class NativePrefixDeployment(unittest.TestCase):
    """M2.10 (tasks/m210-spec.md D2, D6, D11; owner/maintainer D9C 2026-10-12): a cache hit's tokens equal the cold run
    split at the hit, so the prefix cache is the opt-in `-v11` profile. native-g0-prefix is native-g0 plus exactly
    PREFIX_TOML_DELTA on its own profile id and deployment; native-g1-prefix is native-g1 plus the same keys and the
    operator route (OPERATOR_DELTA, for the D11 flush), i.e. native-g0-prefix on the GPU1 lane. Their deployments
    are the cache-off ones with the ids and the capabilities notes naming the cache."""

    LANES = ((G0_PREFIX_PREFIX, G0_PREFIX_TOML, G0_PREFIX_PROFILE_ID, PREFIX, TOML, PROFILE_ID, {}),
             (G1_PREFIX_PREFIX, G1_PREFIX_TOML, G1_PREFIX_PROFILE_ID, G1_PREFIX, G1_TOML, G1_PROFILE_ID,
              OPERATOR_DELTA))

    def test_profile_ids_are_v11(self):
        self.assertEqual((G0_PREFIX_PROFILE_ID, G1_PREFIX_PROFILE_ID),
                         ('native-g0-qwen38-flash-next-v11', 'native-g1-qwen38-flash-next-v11'))
        self.assertEqual(PROFILE_ID, 'native-g0-qwen38-flash-next-v8')

    def test_runtime_config_is_the_cache_off_one_plus_exactly_the_prefix_keys(self):
        for prefix, toml, profile_id, base_prefix, base_toml, _, extra in self.LANES:
            with self.subTest(toml=toml):
                x = tomllib.loads((ROOT / toml).read_text())
                validate('runtime_config', x)
                entry = x['deployments'][0]
                self.assertEqual((entry['deployment'], entry['profile_id']), (prefix, profile_id))
                self.assertEqual(entry['native']['library_sha256'], part(prefix, 'lock')['backend_build_sha256'])
                base = tomllib.loads((ROOT / base_toml).read_text())
                rest = without(self, x, {**PREFIX_TOML_DELTA, **extra})
                for doc in (rest, base):
                    del doc['deployments'][0]['deployment'], doc['deployments'][0]['profile_id']
                self.assertEqual(rest, base)

    def test_g1_prefix_is_g0_prefix_except_the_lane_identity(self):
        g0, g1 = (tomllib.loads((ROOT / t).read_text()) for t in (G0_PREFIX_TOML, G1_PREFIX_TOML))
        self.assertEqual((g1['server']['bind'], g1['deployments'][0]['native']['device_uuid'], g1['scheduler']['devices'],
                          g1['ledger']['state_dir']), ('127.0.0.1:18329', GPU1_UUID, ['gpu1'], '/var/lib/cere/native-g1'))
        for doc in (g0, g1):
            del doc['server']['bind'], doc['scheduler']['devices'], doc['ledger']['state_dir']
            e = doc['deployments'][0]
            del e['profile_id'], e['deployment'], e['native']['device_uuid']
        self.assertEqual(g1, g0)

    def test_deployment_is_the_cache_off_one_with_its_ids_and_cache_notes(self):
        for prefix, _, profile_id, base_prefix, _, base_profile_id, _ in self.LANES:
            with self.subTest(prefix=prefix):
                d = validate_deployment(ROOT, prefix)
                self.assertEqual(d['profile']['profile_id'], profile_id)
                name = prefix.rsplit('/', 1)[1]
                caps, base_caps = part(prefix, 'capabilities'), part(base_prefix, 'capabilities')
                self.assertEqual((caps.pop('profile_id'), base_caps.pop('profile_id')), (profile_id, base_profile_id))
                self.assertIn(PREFIX_NOTES, caps['notes'])
                self.assertNotIn(PRE_PREFIX_NOTES, caps['notes'])
                self.assertEqual(caps.pop('notes'), base_caps.pop('notes').replace(PRE_PREFIX_NOTES, PREFIX_NOTES))
                self.assertEqual(caps, base_caps)
                recipe, base_recipe = part(prefix, 'recipe'), part(base_prefix, 'recipe')
                self.assertEqual((recipe.pop('recipe_id'), recipe.pop('backend_profile_id')), (name, profile_id))
                del base_recipe['recipe_id'], base_recipe['backend_profile_id']
                self.assertEqual(recipe, base_recipe)
                lock, base_lock = part(prefix, 'lock'), part(base_prefix, 'lock')
                self.assertEqual(lock.pop('recipe_sha256'),
                                 hashlib.sha256((ROOT / f'{prefix}.recipe.json').read_bytes()).hexdigest())
                self.assertEqual((lock.pop('lock_id'), lock.pop('backend_profile_id')),
                                 (f'{name}-lock-v1', profile_id))
                for key in ('recipe_sha256', 'lock_id', 'backend_profile_id'):
                    del base_lock[key]
                self.assertEqual(lock, base_lock)

    def test_lock_native_pins_the_prefix_pairs_with_the_shipped_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ('schemas', 'configs/deployments'):
                shutil.copytree(ROOT / rel, root / rel)
            (root / 'configs/runtime').mkdir()
            for rel in PINNED_TOMLS:
                shutil.copy(ROOT / rel, root / rel)
            path = root / 'build.json'
            path.write_text(json.dumps(build_json()))
            r = subprocess.run([sys.executable, str(LOCK_NATIVE), '--root', str(root), '--build-json', str(path)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            for prefix, toml in ((PREFIX, TOML), (G0_PREFIX_PREFIX, G0_PREFIX_TOML), (G1_PREFIX_PREFIX, G1_PREFIX_TOML)):
                with self.subTest(toml=toml):
                    self.assertEqual(part(prefix, 'lock', root)['backend_build_sha256'], LIBRARY_SHA)
                    x = tomllib.loads((root / toml).read_text())
                    self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
                    validate('runtime_config', x, root)


class SpecDeployment(unittest.TestCase):
    """tasks/m29-spec.md D12: the `-v19` deployment (the NVFP4 opt-in since the FP4-Local flip) is the shipped one
    plus speculation (drafts 4, `verify`, verify combinations, notes); its runtime config is native-g0.toml plus the
    drafts, the raised executor limit, SPEC_LEVERS and PREFIX_TOML_DELTA (owner decision 2026-10-15: every opt-in
    except p4)."""

    def test_deployment_is_the_shipped_one_plus_speculation(self):
        d = validate_deployment(ROOT, SPEC_PREFIX)
        self.assertEqual(d['profile']['profile_id'], SPEC_PROFILE_ID)
        recipe, shipped_recipe = part(SPEC_PREFIX, 'recipe'), part(PREFIX, 'recipe')
        self.assertEqual((recipe.pop('recipe_id'), recipe.pop('backend_profile_id')),
                         ('qwen38-flash-next-native-spec-nvfp4', SPEC_PROFILE_ID))
        del shipped_recipe['recipe_id'], shipped_recipe['backend_profile_id']
        self.assertEqual(recipe, shipped_recipe)
        lock, shipped_lock = part(SPEC_PREFIX, 'lock'), part(PREFIX, 'lock')
        self.assertEqual((lock.pop('lock_id'), lock.pop('backend_profile_id')),
                         ('qwen38-flash-next-native-spec-nvfp4-lock-v1', SPEC_PROFILE_ID))
        for doc in (lock, shipped_lock):
            doc.pop('recipe_sha256')
        del shipped_lock['lock_id'], shipped_lock['backend_profile_id']
        self.assertEqual(lock, shipped_lock)
        caps = part(SPEC_PREFIX, 'capabilities')
        self.assertEqual(caps['operations'], ['prefill', 'decode', 'verify', 'cancel'])
        self.assertTrue(caps['native_speculative_batch'])
        x = tomllib.loads((ROOT / SPEC_TOML).read_text())
        native = x['deployments'][0]['native']
        self.assertEqual(native['speculative_draft_tokens'], 4)
        self.assertEqual(native['library_sha256'], part(SPEC_PREFIX, 'lock')['backend_build_sha256'])
        # [REVISED: M29S2-06d] -v19 pins the shipped P3b library (row-invariant at m 1..16, M23-D8).
        self.assertEqual(native['library_sha256'], part(PREFIX, 'lock')['backend_build_sha256'])

    def test_runtime_config_is_the_shipped_one_plus_speculation_and_its_levers(self):
        x = tomllib.loads((ROOT / SPEC_TOML).read_text())
        validate('runtime_config', x)
        entry = x['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (SPEC_PREFIX, SPEC_PROFILE_ID))
        self.assertEqual(entry['native'].pop('levers'), SPEC_LEVERS)
        without(self, x, PREFIX_TOML_DELTA)
        caps = part(SPEC_PREFIX, 'capabilities')
        self.assertTrue(caps['notes'].startswith(SPEC_LEVERS_NOTE))
        self.assertIn(PREFIX_NOTES, caps['notes'])
        self.assertNotIn(PRE_PREFIX_NOTES, caps['notes'])

    def test_the_v10_id_is_retired(self):
        """D9 batch flip: `-v10` became `-v18` (its levers table added), which became `-v19` (owner decision
        2026-10-15); no config carries either old id."""
        for path in sorted((ROOT / 'configs').rglob('*')):
            if path.suffix == '.json':
                doc = load_json(path)
                ids = {doc.get(k) for k in ('profile_id', 'backend_profile_id')} if isinstance(doc, dict) else set()
            elif path.suffix == '.toml':
                ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            else:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn('native-g0-qwen38-flash-next-v10', ids)
                self.assertNotIn('native-g0-qwen38-flash-next-v18', ids)

    def test_lock_native_pins_the_spec_pair_with_the_shipped_one(self):
        """[REVISED: M29S2-06d] a P3b build pins the `-v19` pair with the shipped ones; a K7 build pins neither."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ('schemas', 'configs/deployments'):
                shutil.copytree(ROOT / rel, root / rel)
            (root / 'configs/runtime').mkdir()
            for rel in PINNED_TOMLS:
                shutil.copy(ROOT / rel, root / rel)
            path = root / 'build.json'
            spec_before = (root / SPEC_TOML).read_bytes()
            path.write_text(json.dumps(dict(build_json(), flags=K7_BUILD_FLAGS)))
            r = subprocess.run([sys.executable, str(LOCK_NATIVE), '--root', str(root), '--build-json', str(path)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((root / SPEC_TOML).read_bytes(), spec_before)
            path.write_text(json.dumps(build_json()))
            r = subprocess.run([sys.executable, str(LOCK_NATIVE), '--root', str(root), '--build-json', str(path)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(part(SPEC_PREFIX, 'lock', root)['backend_build_sha256'], LIBRARY_SHA)
            x = tomllib.loads((root / SPEC_TOML).read_text())
            self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
            validate('runtime_config', x, root)
            self.assertEqual(part(SPEC_LOOKUP_PREFIX, 'lock', root)['backend_build_sha256'], LIBRARY_SHA)
            x = tomllib.loads((root / SPEC_LOOKUP_TOML).read_text())
            self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
            validate('runtime_config', x, root)

    def test_lookup_deployment_is_the_spec_one_with_its_shipped_drafts(self):
        """tasks/m29l-spec.md D7: `-v15` equals `-v19` without its levers table and PREFIX_TOML_DELTA but for its
        ids and the notes; the `drafter` lever is absent ("mtp", M1 ship-mtp)."""
        d = validate_deployment(ROOT, SPEC_LOOKUP_PREFIX)
        self.assertEqual(d['profile']['profile_id'], SPEC_LOOKUP_PROFILE_ID)
        recipe, spec_recipe = part(SPEC_LOOKUP_PREFIX, 'recipe'), part(SPEC_PREFIX, 'recipe')
        self.assertEqual((recipe.pop('recipe_id'), recipe.pop('backend_profile_id')),
                         ('qwen38-flash-next-native-spec-lookup', SPEC_LOOKUP_PROFILE_ID))
        del spec_recipe['recipe_id'], spec_recipe['backend_profile_id']
        self.assertEqual(recipe, spec_recipe)
        lock, spec_lock = part(SPEC_LOOKUP_PREFIX, 'lock'), part(SPEC_PREFIX, 'lock')
        self.assertEqual((lock.pop('lock_id'), lock.pop('backend_profile_id')),
                         ('qwen38-flash-next-native-spec-lookup-lock-v1', SPEC_LOOKUP_PROFILE_ID))
        for doc in (lock, spec_lock):
            doc.pop('recipe_sha256')
        del spec_lock['lock_id'], spec_lock['backend_profile_id']
        self.assertEqual(lock, spec_lock)
        x = tomllib.loads((ROOT / SPEC_LOOKUP_TOML).read_text())
        native = x['deployments'][0]['native']
        self.assertEqual((native['speculative_draft_tokens'], native.get('levers')), (4, None))
        self.assertEqual(native['library_sha256'], part(SPEC_LOOKUP_PREFIX, 'lock')['backend_build_sha256'])
        spec = lever_free_spec_toml(self)
        for doc in (x, spec):
            e = doc['deployments'][0]
            del e['deployment'], e['profile_id']
        self.assertEqual(x, spec)


class SpecK2sDeployment(unittest.TestCase):
    """P11 lever 2 (tasks/m23-spec.md PI-02): the `-v16` deployment is the `-v19` speculation one without its levers
    table and PREFIX_TOML_DELTA plus exactly one lever, levers.k2_gemm small_m, on the shipped library (never a build of its own)."""

    def test_deployment_is_the_spec_one_with_the_small_m_lever(self):
        d = validate_deployment(ROOT, SPEC_K2S_PREFIX)
        self.assertEqual(d['profile']['profile_id'], SPEC_K2S_PROFILE_ID)
        caps, spec_caps = part(SPEC_K2S_PREFIX, 'capabilities'), part(SPEC_PREFIX, 'capabilities')
        self.assertIn('levers.k2_gemm small_m', caps['notes'])
        self.assertNotIn('CERE_K2_SMALL_M', caps['notes'])
        self.assertTrue(caps['notes'].endswith(spec_rest_notes(self)))
        self.assertNotIn(SPEC_LEVERS_NOTE, caps['notes'])
        self.assertEqual({k: v for k, v in caps.items() if k not in ('profile_id', 'notes')},
                         {k: v for k, v in spec_caps.items() if k not in ('profile_id', 'notes')})
        recipe, lock = part(SPEC_K2S_PREFIX, 'recipe'), part(SPEC_K2S_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-k2s', 'qwen38-flash-next-native-spec-k2s-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (SPEC_K2S_PROFILE_ID, SPEC_K2S_PROFILE_ID))
        self.assertEqual({k: v for k, v in recipe.items() if k not in ('recipe_id', 'backend_profile_id')},
                         {k: v for k, v in part(SPEC_PREFIX, 'recipe').items()
                          if k not in ('recipe_id', 'backend_profile_id')})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{SPEC_K2S_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_ = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_},
                         {k: v for k, v in part(SPEC_PREFIX, 'lock').items() if k not in except_})

    def test_runtime_config_is_the_spec_one_plus_its_lever(self):
        x = tomllib.loads((ROOT / SPEC_K2S_TOML).read_text())
        validate('runtime_config', x)
        entry = x['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (SPEC_K2S_PREFIX, SPEC_K2S_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(SPEC_K2S_PREFIX, 'lock')['backend_build_sha256'])
        spec = lever_free_spec_toml(self)
        self.assertEqual(entry['native'].pop('levers'), {'k2_gemm': 'small_m'})
        for doc in (x, spec):
            e = doc['deployments'][0]
            del e['deployment'], e['profile_id']
        self.assertEqual(x, spec)


class SpecFp8wDeployment(unittest.TestCase):
    """Fp8W: the opt-in `-v20` deployment is the `-v19` speculation one (its levers and prefix cache) plus exactly one
    lever, levers.k2_weights fp8, on the shipped library (never a build of its own); the default stays `-v19`."""

    def test_deployment_is_the_spec_one_with_the_fp8_weights_lever(self):
        d = validate_deployment(ROOT, SPEC_FP8W_PREFIX)
        self.assertEqual(d['profile']['profile_id'], SPEC_FP8W_PROFILE_ID)
        caps, spec_caps = part(SPEC_FP8W_PREFIX, 'capabilities'), part(SPEC_PREFIX, 'capabilities')
        self.assertIn('levers.k2_weights fp8', caps['notes'])
        self.assertTrue(caps['notes'].endswith(spec_caps['notes']))
        self.assertEqual({k: v for k, v in caps.items() if k not in ('profile_id', 'notes')},
                         {k: v for k, v in spec_caps.items() if k not in ('profile_id', 'notes')})
        recipe, lock = part(SPEC_FP8W_PREFIX, 'recipe'), part(SPEC_FP8W_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-fp8w', 'qwen38-flash-next-native-spec-fp8w-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (SPEC_FP8W_PROFILE_ID, SPEC_FP8W_PROFILE_ID))
        self.assertEqual({k: v for k, v in recipe.items() if k not in ('recipe_id', 'backend_profile_id')},
                         {k: v for k, v in part(SPEC_PREFIX, 'recipe').items()
                          if k not in ('recipe_id', 'backend_profile_id')})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{SPEC_FP8W_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_ = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_},
                         {k: v for k, v in part(SPEC_PREFIX, 'lock').items() if k not in except_})

    def test_runtime_config_is_the_spec_one_plus_its_lever(self):
        x = tomllib.loads((ROOT / SPEC_FP8W_TOML).read_text())
        validate('runtime_config', x)
        entry = x['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (SPEC_FP8W_PREFIX, SPEC_FP8W_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(SPEC_FP8W_PREFIX, 'lock')['backend_build_sha256'])
        spec = tomllib.loads((ROOT / SPEC_TOML).read_text())
        self.assertEqual(entry['native'].pop('levers'), {**SPEC_LEVERS, 'k2_weights': 'fp8'})
        self.assertEqual(spec['deployments'][0]['native'].pop('levers'), SPEC_LEVERS)
        for doc in (x, spec):
            e = doc['deployments'][0]
            del e['deployment'], e['profile_id']
        self.assertEqual(x, spec)


class SpecFp4localDeployment(unittest.TestCase):
    """FP4-Local FL-13 (tasks/fp4local-spec.md D12, D16): the S2 dev `-v19-fp4local` deployment is the `-v19`
    speculation one (levers, prefix cache, library, capabilities declared from it) on the converted FP4-Local artifact:
    its lock and runtime config name the artifact the committed pin names, whose artifact profile is the NVFP4-PLE one."""

    def test_deployment_is_the_spec_one_on_the_converted_artifact(self):
        pin = json.loads((ROOT / FP4LOCAL_PIN).read_text())
        self.assertEqual(pin['profile_id'], 'rtxpro6000bw-plehost-nvfp4-text-v1')
        d = validate_deployment(ROOT, SPEC_FP4LOCAL_PREFIX)
        self.assertEqual(d['profile']['profile_id'], SPEC_FP4LOCAL_PROFILE_ID)
        caps, spec_caps = part(SPEC_FP4LOCAL_PREFIX, 'capabilities'), part(SPEC_PREFIX, 'capabilities')
        self.assertEqual((caps['artifact_sha256'], caps['evidence']), (pin['manifest_sha256'], 'declared'))
        except_caps = ('profile_id', 'notes', 'artifact_sha256', 'evidence')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in spec_caps.items() if k not in except_caps})
        recipe, lock = part(SPEC_FP4LOCAL_PREFIX, 'recipe'), part(SPEC_FP4LOCAL_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']), ('qwen38-flash-next-native-fp4local', pin['lock_id']))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (SPEC_FP4LOCAL_PROFILE_ID, SPEC_FP4LOCAL_PROFILE_ID))
        self.assertEqual(recipe['model_ref'], 'local://models/qwen38-flash-next/Qwen3.8-Flash-Next-FP4-Local-cere')
        except_recipe = ('recipe_id', 'backend_profile_id', 'model_ref')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{SPEC_FP4LOCAL_PREFIX}.recipe.json').read_bytes()).hexdigest())
        self.assertEqual((lock['artifact_sha256'], lock['model_content_sha256'], lock['tokenizer_sha256'],
                          lock['template_sha256']),
                         (pin['manifest_sha256'], pin['model_hash'], pin['tokenizer_hash'], pin['template_hash']))
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256', 'artifact_sha256', 'model_content_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_spec_one_on_the_converted_artifact(self):
        pin = json.loads((ROOT / FP4LOCAL_PIN).read_text())
        x = tomllib.loads((ROOT / SPEC_FP4LOCAL_TOML).read_text())
        validate('runtime_config', x)
        entry = x['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (SPEC_FP4LOCAL_PREFIX, SPEC_FP4LOCAL_PROFILE_ID))
        self.assertEqual((entry['native']['artifact_dir'], entry['native']['expected_manifest_sha256']),
                         (pin['path'], pin['manifest_sha256']))
        self.assertEqual(entry['native']['library_sha256'], part(SPEC_FP4LOCAL_PREFIX, 'lock')['backend_build_sha256'])
        spec = tomllib.loads((ROOT / SPEC_TOML).read_text())
        cur = spec['deployments'][0]['native']['artifact_dir']
        self.assertNotEqual(cur, pin['path'])
        self.assertEqual(json.dumps(x).count(pin['path']), 2)  # the native artifact and the tokenizer sidecar
        for doc in (x, spec):
            e = doc['deployments'][0]
            del e['deployment'], e['profile_id'], e['native']['artifact_dir'], e['native']['expected_manifest_sha256']
        self.assertEqual(json.loads(json.dumps(x).replace(pin['path'], cur)), spec)

class S3Deployment(unittest.TestCase):
    """FP4-Local S3 (tasks/fp4local-spec.md S3, D12, FL-30): the opt-in `-v21` deployment is the S2 dev `-v19-fp4local`
    one (so `-v19`'s levers, prefix cache, library, capabilities declared from it) on the S3 artifact: its lock and
    runtime config name the artifact the committed pin names, whose artifact profile is the MXFP8 one."""

    def test_deployment_is_the_s2_one_on_the_s3_artifact(self):
        pin = json.loads((ROOT / S3_PIN).read_text())
        self.assertEqual(pin['profile_id'], 'rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1')
        d = validate_deployment(ROOT, S3_PREFIX)
        self.assertEqual(d['profile']['profile_id'], S3_PROFILE_ID)
        caps, s2_caps = part(S3_PREFIX, 'capabilities'), part(SPEC_FP4LOCAL_PREFIX, 'capabilities')
        self.assertEqual((caps['artifact_sha256'], caps['evidence']), (pin['manifest_sha256'], 'declared'))
        except_caps = ('profile_id', 'notes', 'artifact_sha256')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in s2_caps.items() if k not in except_caps})
        recipe, lock = part(S3_PREFIX, 'recipe'), part(S3_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']), ('qwen38-flash-next-native-fp4local-s3', pin['lock_id']))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (S3_PROFILE_ID, S3_PROFILE_ID))
        self.assertEqual(recipe['model_ref'], 'local://models/qwen38-flash-next/Qwen3.8-Flash-Next-FP4-Local-cere-s3')
        except_recipe = ('recipe_id', 'backend_profile_id', 'model_ref')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(SPEC_FP4LOCAL_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'], hashlib.sha256((ROOT / f'{S3_PREFIX}.recipe.json').read_bytes()).hexdigest())
        self.assertEqual((lock['artifact_sha256'], lock['model_content_sha256'], lock['tokenizer_sha256'],
                          lock['template_sha256']),
                         (pin['manifest_sha256'], pin['model_hash'], pin['tokenizer_hash'], pin['template_hash']))
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256', 'artifact_sha256', 'model_content_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(SPEC_FP4LOCAL_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_s2_one_on_the_s3_artifact(self):
        pin, s2_pin = json.loads((ROOT / S3_PIN).read_text()), json.loads((ROOT / FP4LOCAL_PIN).read_text())
        x = tomllib.loads((ROOT / S3_TOML).read_text())
        validate('runtime_config', x)
        entry = x['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (S3_PREFIX, S3_PROFILE_ID))
        self.assertEqual((entry['native']['artifact_dir'], entry['native']['expected_manifest_sha256']),
                         (pin['path'], pin['manifest_sha256']))
        self.assertEqual(entry['native']['library_sha256'], part(S3_PREFIX, 'lock')['backend_build_sha256'])
        s2 = tomllib.loads((ROOT / SPEC_FP4LOCAL_TOML).read_text())
        self.assertNotEqual(s2_pin['path'], pin['path'])
        self.assertEqual(json.dumps(x).count(pin['path']), 2)  # the native artifact and the tokenizer sidecar
        for doc in (x, s2):
            e = doc['deployments'][0]
            del e['deployment'], e['profile_id'], e['native']['artifact_dir'], e['native']['expected_manifest_sha256']
        self.assertEqual(json.loads(json.dumps(x).replace(pin['path'], s2_pin['path'])), s2)

    def test_never_the_default_profile(self):
        for toml in (TOML, DEFAULT_SPEC_TOML, V22_TOML, SPEC_TOML):
            with self.subTest(toml=toml):
                self.assertNotIn(S3_PROFILE_ID, (ROOT / toml).read_text())


class V22SpecDeployment(unittest.TestCase):
    """The FP4-Local flip (tasks/fp4local-spec.md FL-31, owner decision 2026-10-17) made the S3 `-v21` deployment
    (artifact, levers, library, capabilities) under Headroom's `-v22` id (tasks/m210-spec.md D15''''' O112) the
    default; the DraftTrain flip (tasks/v29default-spec.md VD-03) moved it to the opt-in `native-g0-spec-v22.toml`.
    Its runtime config differs from `-v21`'s only in the deployment, the profile id and the three headroom keys
    (prefix_state_slots 32, prefix_end_state, kv_pool_tokens 300,032). Clean cutover: no other config carries `-v22`,
    and neither the retired `fp4local-s3-v22` files nor DraftTrain's `spec-draft` ones exist."""

    def test_deployment_is_the_s3_one_under_its_own_id(self):
        d = validate_deployment(ROOT, V22_PREFIX)
        self.assertEqual(d['profile']['profile_id'], V22_PROFILE_ID)
        caps, s3_caps = part(V22_PREFIX, 'capabilities'), part(S3_PREFIX, 'capabilities')
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in s3_caps.items() if k not in except_caps})
        recipe, lock = part(V22_PREFIX, 'recipe'), part(V22_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-v22', 'qwen38-flash-next-native-spec-v22-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (V22_PROFILE_ID, V22_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(S3_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{V22_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(S3_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_s3_plus_the_headroom_keys(self):
        x = tomllib.loads((ROOT / V22_TOML).read_text())
        validate('runtime_config', x)
        s3 = tomllib.loads((ROOT / S3_TOML).read_text())
        entry, s3e = x['deployments'][0], s3['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (V22_PREFIX, V22_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(V22_PREFIX, 'lock')['backend_build_sha256'])
        n, s3n = entry['native'], s3e['native']
        self.assertEqual((n['prefix_state_slots'], n['prefix_end_state'], n['kv_pool_tokens']), (32, True, 300032))
        self.assertEqual((s3n['prefix_state_slots'], s3n.get('prefix_end_state'), s3n['kv_pool_tokens']),
                         (16, None, 200000))
        for e in (entry, s3e):
            del e['deployment'], e['profile_id']
            for key in ('prefix_state_slots', 'prefix_end_state', 'kv_pool_tokens'):
                e['native'].pop(key, None)
        self.assertEqual(x, s3)

    def test_the_only_v22_config_and_no_retired_files(self):
        self.assertFalse(list((ROOT / 'configs').rglob('*fp4local-s3-v22*')))
        self.assertFalse(list((ROOT / 'configs').rglob('*spec-draft*')))
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(V22_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == V22_TOML)


class TreeSpecDeployment(unittest.TestCase):
    """TreeSpec (tasks/tree-spec-spec.md TS-03): the opt-in `-v28` deployment is the `-v22` one (artifact, library,
    capabilities) under its own id; its runtime config differs from `-v22`'s only in the deployment, the profile id and
    levers.drafter "mtp+lookup+long". No other config carries `-v28`."""

    def test_deployment_is_the_v22_one_under_its_own_id(self):
        d = validate_deployment(ROOT, TREE_PREFIX)
        self.assertEqual(d['profile']['profile_id'], TREE_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(TREE_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(V22_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(TREE_PREFIX, 'recipe'), part(TREE_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-tree', 'qwen38-flash-next-native-spec-tree-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (TREE_PROFILE_ID, TREE_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(V22_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{TREE_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(V22_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_v22_plus_the_long_drafter(self):
        x = tomllib.loads((ROOT / TREE_TOML).read_text())
        validate('runtime_config', x)
        base = tomllib.loads((ROOT / V22_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (TREE_PREFIX, TREE_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(TREE_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual((entry['native']['levers']['drafter'], b['native']['levers']['drafter']),
                         ('mtp+lookup+long', 'mtp+lookup'))
        for e in (entry, b):
            del e['deployment'], e['profile_id'], e['native']['levers']['drafter']
        self.assertEqual(x, base)

    def test_the_only_v28_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(TREE_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == TREE_TOML)


class ChainsDeployment(unittest.TestCase):
    """Chains (tasks/chains-spec.md): the opt-in `-v36` deployment is the TreeSpec `-v28` one (the `-v22`
    artifact, library, capabilities) under its own id; its runtime config differs from `-v28`'s only in the deployment,
    the profile id and levers.drafter "mtp+lookup+chain". No other config carries `-v36`."""

    def test_deployment_is_the_tree_one_under_its_own_id(self):
        d = validate_deployment(ROOT, CHAINS_PREFIX)
        self.assertEqual(d['profile']['profile_id'], CHAINS_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(CHAINS_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(TREE_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(CHAINS_PREFIX, 'recipe'), part(CHAINS_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-chains', 'qwen38-flash-next-native-spec-chains-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (CHAINS_PROFILE_ID, CHAINS_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(TREE_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{CHAINS_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(TREE_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_tree_one_plus_the_chain_drafter(self):
        x = tomllib.loads((ROOT / CHAINS_TOML).read_text())
        validate('runtime_config', x)
        base = tomllib.loads((ROOT / TREE_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (CHAINS_PREFIX, CHAINS_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(CHAINS_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual((entry['native']['levers']['drafter'], b['native']['levers']['drafter']),
                         ('mtp+lookup+chain', 'mtp+lookup+long'))
        for e in (entry, b):
            del e['deployment'], e['profile_id'], e['native']['levers']['drafter']
        self.assertEqual(x, base)

    def test_the_only_v36_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(CHAINS_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == CHAINS_TOML)


class SessionsDeployment(unittest.TestCase):
    """Sessions (tasks/sessions-spec.md O211): the opt-in `-v45` deployment is the default `-v29` one (artifact,
    library) under its own id with snapshot/restore and native_state_snapshot declared (the host arena); its runtime
    config differs from `-v29`'s only in the deployment, the profile id, snapshot_host_bytes and host_spill_enabled.
    No other config carries `-v45`."""

    def test_deployment_is_the_default_one_plus_the_arena(self):
        d = validate_deployment(ROOT, SESSIONS_PREFIX)
        self.assertEqual(d['profile']['profile_id'], SESSIONS_PROFILE_ID)
        caps, base_caps = part(SESSIONS_PREFIX, 'capabilities'), part(DEFAULT_SPEC_PREFIX, 'capabilities')
        self.assertEqual(caps['operations'], ['prefill', 'decode', 'verify', 'snapshot', 'restore', 'cancel'])
        self.assertEqual(base_caps['operations'], ['prefill', 'decode', 'verify', 'cancel'])
        self.assertEqual((caps['native_state_snapshot'], base_caps['native_state_snapshot']), (True, False))
        except_caps = ('profile_id', 'notes', 'operations', 'native_state_snapshot')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in base_caps.items() if k not in except_caps})
        recipe, lock = part(SESSIONS_PREFIX, 'recipe'), part(SESSIONS_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-sessions', 'qwen38-flash-next-native-sessions-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (SESSIONS_PROFILE_ID, SESSIONS_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{SESSIONS_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_default_one_plus_the_host_tier(self):
        x = tomllib.loads((ROOT / SESSIONS_TOML).read_text())
        validate('runtime_config', x)
        base = default_spec_before_flips(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (SESSIONS_PREFIX, SESSIONS_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(SESSIONS_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual(entry['native'].pop('snapshot_host_bytes'), SESSIONS_ARENA_BYTES)
        self.assertNotIn('snapshot_host_bytes', b['native'])
        self.assertEqual((x['state']['host_spill_enabled'], base['state']['host_spill_enabled']), (True, False))
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        del x['state']['host_spill_enabled'], base['state']['host_spill_enabled']
        self.assertEqual(x, base)

    def test_the_only_v45_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(SESSIONS_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == SESSIONS_TOML)


class SessionsV53Deployment(unittest.TestCase):
    """Sessions SE-11 (tasks/sessions-spec.md): the `-v53` deployment is Ctx262's `-v52` one (artifact, library, context
    262,144) under its own id with snapshot/restore and native_state_snapshot declared (the host arena); its runtime
    config differs from `-v52`'s only in the deployment, the profile id, snapshot_host_bytes and host_spill_enabled.
    No other config carries `-v53` (production served it from window 10 until window 11 moved prod-t4.toml to
    `-v58`, tasks/batch-concurrency-spec.md BC7)."""

    def test_deployment_is_the_v52_one_plus_the_arena(self):
        d = validate_deployment(ROOT, V53_PREFIX)
        self.assertEqual(d['profile']['profile_id'], V53_PROFILE_ID)
        caps, base_caps = part(V53_PREFIX, 'capabilities'), part(V52_PREFIX, 'capabilities')
        self.assertEqual(caps['operations'], ['prefill', 'decode', 'verify', 'snapshot', 'restore', 'cancel'])
        self.assertEqual(base_caps['operations'], ['prefill', 'decode', 'verify', 'cancel'])
        self.assertEqual((caps['native_state_snapshot'], base_caps['native_state_snapshot']), (True, False))
        except_caps = ('profile_id', 'notes', 'operations', 'native_state_snapshot')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in base_caps.items() if k not in except_caps})
        recipe, lock = part(V53_PREFIX, 'recipe'), part(V53_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-sessions-v53', 'qwen38-flash-next-native-sessions-v53-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (V53_PROFILE_ID, V53_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(V52_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{V53_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(V52_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_v52_one_plus_the_host_tier(self):
        x = tomllib.loads((ROOT / V53_TOML).read_text())
        validate('runtime_config', x)
        base = pop_l8_share(self, tomllib.loads((ROOT / V52_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (V53_PREFIX, V53_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(V53_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual(entry['native'].pop('snapshot_host_bytes'), SESSIONS_ARENA_BYTES)
        self.assertNotIn('snapshot_host_bytes', b['native'])
        self.assertEqual((x['state']['host_spill_enabled'], base['state']['host_spill_enabled']), (True, False))
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        del x['state']['host_spill_enabled'], base['state']['host_spill_enabled']
        self.assertEqual(x, base)

    def test_the_only_v53_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(V53_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == V53_TOML)


class Rows16V58Deployment(unittest.TestCase):
    """BatchConc BC2 (tasks/batch-concurrency-spec.md BC2, owner Q1 (b), Q2 (a), Q8): the dev `-v58` deployment is the
    `-v53` one (artifact, library, context 262,144, snapshot/restore) under its own id at `decode_batch_max` 16; its
    runtime config is production's (prod-t4.toml, `-v58` since window 11, BC7) on the GPU0 lane (qe1_tickets.RETARGET's
    fields): 16 rows of admission and a KV pool of one context, against `-v53`'s 4 rows and 524,288. No other config
    but prod-t4.toml carries `-v58`."""

    def test_deployment_is_the_v53_one_at_16_rows(self):
        d = validate_deployment(ROOT, V58_PREFIX)
        self.assertEqual(d['profile']['profile_id'], V58_PROFILE_ID)
        caps, base_caps = part(V58_PREFIX, 'capabilities'), part(V53_PREFIX, 'capabilities')
        self.assertEqual((caps['decode_batch_max'], base_caps['decode_batch_max']), (16, 4))
        # Every decode and verify combination validates rows 1..16 (the base's 1..4); the rest is the base's.
        rows = lambda c, n: [dict(x, row_counts=list(range(1, n + 1))) if x['operation'] in ('decode', 'verify') else x
                             for x in c['validated_combinations']]
        self.assertEqual(caps['validated_combinations'], rows(base_caps, 16))
        self.assertEqual(base_caps['validated_combinations'], rows(base_caps, 4))
        except_caps = ('profile_id', 'notes', 'decode_batch_max', 'validated_combinations')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in base_caps.items() if k not in except_caps})
        recipe, lock = part(V58_PREFIX, 'recipe'), part(V58_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-rows16-v58', 'qwen38-flash-next-native-rows16-v58-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (V58_PROFILE_ID, V58_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(V53_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{V58_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(V53_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_production_on_the_gpu0_lane(self):
        x = tomllib.loads((ROOT / V58_TOML).read_text())
        validate('runtime_config', x)
        prod = tomllib.loads((ROOT / PROD_T4_TOML).read_text())
        lane = tomllib.loads((ROOT / V53_TOML).read_text())
        entry, p = x['deployments'][0], prod['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (V58_PREFIX, V58_PROFILE_ID))
        self.assertEqual((p['deployment'], p['profile_id']), (V58_PREFIX, V58_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(V58_PREFIX, 'lock')['backend_build_sha256'])
        # The GPU0 lane's fields (the -v53 dev config's) for production's GPU1 ones, and native_backend's two scopes for
        # production's frontend and ae (frontend's DRR weight kept on native-a).
        self.assertEqual(x['server'], lane['server'])
        self.assertEqual(entry['native'].pop('device_uuid'), lane['deployments'][0]['native']['device_uuid'])
        self.assertEqual(p['native'].pop('device_uuid'), GPU1_UUID)
        self.assertEqual(x['scheduler'], lane['scheduler'])
        self.assertEqual(x['ledger'].pop('state_dir'), lane['ledger']['state_dir'])
        self.assertEqual(x['auth']['keyless_scope'], 'native-a')
        self.assertEqual([(s['scope_id'], s.get('weight', 1)) for s in x['auth']['scopes']],
                         [('native-a', 4), ('native-b', 1)])
        self.assertEqual(prod['auth']['keyless_scope'], 'frontend')
        for doc in (x, prod):
            del doc['server'], doc['scheduler'], doc['auth']['keyless_scope'], doc['auth']['scopes']
        self.assertEqual(prod['telemetry'].pop('clients'), PROD_T4_CLIENTS)
        del prod['ledger']['state_dir']
        # 16 rows (Q1 (b), Q8), 16 queued (Q1), against -v53's 4 rows.
        self.assertEqual((x['admission']['max_backend_inflight'], x['admission']['per_scope_inflight'],
                          x['admission']['queue_depth']), (16, 16, 16))
        self.assertEqual(lane['admission']['max_backend_inflight'], 4)
        # Q2 (a): the pool shrinks to the context (one 262,144-token session still fits; -v53 has 524,288).
        pool = entry['native']['kv_pool_tokens']
        self.assertEqual(pool, entry['native']['max_context_tokens'])
        self.assertLess(pool, lane['deployments'][0]['native']['kv_pool_tokens'])
        self.assertEqual(x, prod)

    def test_the_only_v58_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(V58_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() in (V58_TOML, PROD_T4_TOML))



class PrefixHitDeployment(unittest.TestCase):
    """PrefixHit (tasks/m210-spec.md D17): the opt-in `-v51` deployment is the default `-v29` one (artifact, library,
    capabilities) under its own id; its runtime config differs from `-v29`'s only in the deployment, the profile id and
    `native.prefix_alt_edges = true`. No other config carries `-v51`."""

    def test_deployment_is_the_default_one_under_its_own_id(self):
        d = validate_deployment(ROOT, V51_PREFIX)
        self.assertEqual(d['profile']['profile_id'], V51_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(V51_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(V51_PREFIX, 'recipe'), part(V51_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-v51', 'qwen38-flash-next-native-spec-v51-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (V51_PROFILE_ID, V51_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{V51_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_default_one_plus_alt_edges(self):
        x = tomllib.loads((ROOT / V51_TOML).read_text())
        validate('runtime_config', x)
        base = default_spec_before_flips(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (V51_PREFIX, V51_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(V51_PREFIX, 'lock')['backend_build_sha256'])
        self.assertIs(entry['native'].pop('prefix_alt_edges'), True)
        self.assertNotIn('prefix_alt_edges', b['native'])
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)

    def test_the_only_v51_and_v52_configs(self):
        """`-v51` only in its own config; `-v52` (Ctx262 CX-09) only in its own (production left it for `-v53` in window
        10); alternative edges only in those two, production's, the Sessions `-v53` (SE-11), BatchConc's `-v58` (BC2)
        and the diagnostics `-v54` (VisionLong) and `-v55` (Ctx262 CX-R18)."""
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            entries = tomllib.loads(path.read_text()).get('deployments', [])
            ids = {e.get('profile_id') for e in entries}
            alt = any(e.get('native', {}).get('prefix_alt_edges') for e in entries)
            rel = path.relative_to(ROOT).as_posix()
            with self.subTest(path=path.name):
                self.assertEqual(V51_PROFILE_ID in ids, rel == V51_TOML)
                self.assertEqual(V52_PROFILE_ID in ids, rel == V52_TOML)
                self.assertEqual(alt, rel in (V51_TOML, V52_TOML, PROD_T4_TOML, V53_TOML, V58_TOML, V59_TOML,
                                              VISION_V54_TOML, VISION_V55_TOML))


class OptInPrefillDeployment(unittest.TestCase):
    """BeatSglang S7 (tasks/beat-sglang-spec.md S7): the opt-in `-v56` deployment is the default `-v29` one (artifact,
    library, capabilities) under its own id; its runtime config differs from the default's only in the deployment, the
    profile id and the levers k2_gemm `small_m_a8` (the default's `small_m`) and prefill_attention `p4` (the default's
    registry default). No other config carries `-v56`, and it is no default (owner OQ1)."""

    def test_deployment_is_the_default_one_under_its_own_id(self):
        d = validate_deployment(ROOT, V56_PREFIX)
        self.assertEqual(d['profile']['profile_id'], V56_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(V56_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(V56_PREFIX, 'recipe'), part(V56_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-v56', 'qwen38-flash-next-native-spec-v56-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (V56_PROFILE_ID, V56_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{V56_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_default_one_plus_the_prefill_levers(self):
        x = tomllib.loads((ROOT / V56_TOML).read_text())
        validate('runtime_config', x)
        base = pop_l8_share(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))  # L8: -v56 shares no segment
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (V56_PREFIX, V56_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(V56_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual(entry['native']['levers'].pop('k2_gemm'), 'small_m_a8')
        self.assertEqual(b['native']['levers'].pop('k2_gemm'), 'small_m')
        self.assertEqual(entry['native']['levers'].pop('prefill_attention'), 'p4')
        self.assertNotIn('prefill_attention', b['native']['levers'])
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)

    def test_the_only_v56_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(V56_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == V56_TOML)


class BcPrefillDeployment(unittest.TestCase):
    """BatchConcurrency BC4 (tasks/batch-concurrency-spec.md BC4): the opt-in `-v59` deployment is the production
    Sessions `-v53` one (artifact, library, capabilities) under its own id; its runtime config differs from `-v53`'s only
    in the deployment, the profile id and MixedPD's `[deployments.native] prefill_slice_ms` 50 (tasks/mixed-pd-spec.md).
    No other config carries `-v59`."""

    def test_deployment_is_the_v53_one_under_its_own_id(self):
        d = validate_deployment(ROOT, V59_PREFIX)
        self.assertEqual(d['profile']['profile_id'], V59_PROFILE_ID)
        except_caps = ('profile_id', 'notes', 'prefill_signature')  # the BC1 pin: `-v59` only (test_cere_contracts)
        self.assertEqual({k: v for k, v in part(V59_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(V53_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(V59_PREFIX, 'recipe'), part(V59_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-sessions-v59', 'qwen38-flash-next-native-sessions-v59-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (V59_PROFILE_ID, V59_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(V53_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{V59_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(V53_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_v53_one_plus_prefill_slice_50(self):
        x = tomllib.loads((ROOT / V59_TOML).read_text())
        validate('runtime_config', x)
        base = tomllib.loads((ROOT / V53_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (V59_PREFIX, V59_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(V59_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual((entry['native'].pop('prefill_slice_ms'), b['native'].get('prefill_slice_ms')), (50, None))
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)

    def test_the_only_v59_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(V59_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == V59_TOML)



class Ctx262PrefixHitDeployment(unittest.TestCase):
    """Ctx262 CX-09 (tasks/ctx262-spec.md; maintainer 2026-10-21): the `-v52` deployment is the `-v44` one (artifact,
    levers, library, capabilities, every context field 262,144) under its own id; its runtime config differs from
    `-v44`'s only in the deployment, the profile id and `native.prefix_alt_edges = true`, as `-v51`'s from `-v29`'s."""

    def test_deployment_is_the_v44_one_under_its_own_id(self):
        d = validate_deployment(ROOT, V52_PREFIX)
        self.assertEqual(d['profile']['profile_id'], V52_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(V52_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(CTX262_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(V52_PREFIX, 'recipe'), part(V52_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-v52', 'qwen38-flash-next-native-spec-v52-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']), (V52_PROFILE_ID, V52_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(CTX262_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{V52_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(CTX262_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_v44_one_plus_alt_edges(self):
        x = tomllib.loads((ROOT / V52_TOML).read_text())
        validate('runtime_config', x)
        pop_l8_share(self, x)
        base = tomllib.loads((ROOT / CTX262_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (V52_PREFIX, V52_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(V52_PREFIX, 'lock')['backend_build_sha256'])
        self.assertIs(entry['native'].pop('prefix_alt_edges'), True)
        self.assertNotIn('prefix_alt_edges', b['native'])
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)


class MixedPDDeployment(unittest.TestCase):
    """MixedPD (tasks/mixed-pd-spec.md): the opt-in `-v47` deployment is the default `-v29` one (artifact, library,
    capabilities) under its own id; its runtime config differs from `-v29`'s only in the deployment, the profile id and
    `[deployments.native] prefill_slice_ms` 200. No other config carries `-v47`."""

    def test_deployment_is_the_default_one_under_its_own_id(self):
        d = validate_deployment(ROOT, MIXEDPD_PREFIX)
        self.assertEqual(d['profile']['profile_id'], MIXEDPD_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(MIXEDPD_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(MIXEDPD_PREFIX, 'recipe'), part(MIXEDPD_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-v47', 'qwen38-flash-next-native-spec-v47-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (MIXEDPD_PROFILE_ID, MIXEDPD_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{MIXEDPD_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_default_one_plus_the_prefill_slice(self):
        x = tomllib.loads((ROOT / MIXEDPD_TOML).read_text())
        validate('runtime_config', x)
        pop_l8_share(self, x)  # L8: -v47 shares like -v29 (the default before its flips has no share)
        base = default_spec_before_flips(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (MIXEDPD_PREFIX, MIXEDPD_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(MIXEDPD_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual((entry['native']['prefill_slice_ms'], b['native'].get('prefill_slice_ms')), (200, None))
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        del entry['native']['prefill_slice_ms']
        self.assertEqual(x, base)

    def test_the_only_v47_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(MIXEDPD_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == MIXEDPD_TOML)


class LoraDeployment(unittest.TestCase):
    """Track LoRA (Wave P): the opt-in `-v50` deployment is the default `-v29` one under its own id, and its runtime
    config is `native-g0-spec.toml` plus the `[deployments.native.lora]` table."""

    def test_deployment_is_the_default_one_under_its_own_id(self):
        d = validate_deployment(ROOT, LORA_PREFIX)
        self.assertEqual(d['profile']['profile_id'], LORA_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(LORA_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(LORA_PREFIX, 'recipe'), part(LORA_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-lora', 'qwen38-flash-next-native-spec-lora-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (LORA_PROFILE_ID, LORA_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{LORA_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_default_one_plus_the_lora_table(self):
        x = tomllib.loads((ROOT / LORA_TOML).read_text())
        validate('runtime_config', x)
        base = default_spec_before_flips(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (LORA_PREFIX, LORA_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(LORA_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual(entry['native'].pop('lora'),
                         {'max_adapters': 4, 'max_rank': 64, 'adapter_root': '/var/lib/cere/lora'})
        self.assertNotIn('lora', b['native'])
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)

    def test_the_only_v50_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(LORA_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == LORA_TOML)


class ResearchDeployment(unittest.TestCase):
    """WeightLab WL-01: the research `-v57` deployment is the `-v50` one under its own id, and its runtime config is
    `native-g0-spec-lora.toml` plus `[deployments.native.research]` (the spec's values), api_compat_anthropic, `[api]
    default_deadline_ms` 900,000 and the WeightLab arm scopes wl-a/wl-b/wl-p; no other config carries the table."""

    def test_deployment_is_the_lora_one_under_its_own_id(self):
        d = validate_deployment(ROOT, RESEARCH_PREFIX)
        self.assertEqual(d['profile']['profile_id'], RESEARCH_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(RESEARCH_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(LORA_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(RESEARCH_PREFIX, 'recipe'), part(RESEARCH_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-research', 'qwen38-flash-next-native-spec-research-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (RESEARCH_PROFILE_ID, RESEARCH_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(LORA_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{RESEARCH_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(LORA_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_lora_one_plus_the_research_table(self):
        x = tomllib.loads((ROOT / RESEARCH_TOML).read_text())
        validate('runtime_config', x)
        base = tomllib.loads((ROOT / LORA_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (RESEARCH_PREFIX, RESEARCH_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(RESEARCH_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual(entry['native'].pop('research'),
                         {'device_staging_bytes': 67108864, 'host_buffer_bytes': 1073741824,
                          'max_capture_bytes_per_request': 268435456, 'max_plans': 8, 'max_vectors': 64})
        self.assertIs(entry.pop('api_compat_anthropic'), True)
        self.assertNotIn('api_compat_anthropic', b)
        self.assertEqual((x['api'].pop('default_deadline_ms'), base['api'].pop('default_deadline_ms')),
                         (900000, 300000))
        self.assertEqual(x['server']['bind'], '127.0.0.1:18309')
        self.assertEqual(x['auth']['operator_token_env'], 'CERE_OPERATOR_TOKEN')
        self.assertNotIn('keyless_scope', x['auth'])
        wl = [s for s in x['auth']['scopes'] if s['scope_id'].startswith('wl-')]
        self.assertEqual(wl, [{'scope_id': 'wl-a', 'token_env': 'CERE_TOKEN_WL_A'},
                              {'scope_id': 'wl-b', 'token_env': 'CERE_TOKEN_WL_B'},
                              {'scope_id': 'wl-p', 'token_env': 'CERE_TOKEN_WL_P'}])
        x['auth']['scopes'] = [s for s in x['auth']['scopes'] if s not in wl]
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)

    def test_the_only_v57_config_and_the_only_research_table(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            entries = tomllib.loads(path.read_text()).get('deployments', [])
            ids = {e.get('profile_id') for e in entries}
            research = any('research' in (e.get('native') or {}) for e in entries)
            only = path.relative_to(ROOT).as_posix() == RESEARCH_TOML
            with self.subTest(path=path.name):
                self.assertEqual(RESEARCH_PROFILE_ID in ids, only)
                self.assertEqual(research, only)



class DefaultSpecDeployment(unittest.TestCase):
    """The DraftTrain flip (tasks/v29default-spec.md VD-03): the default speculation deployment is DraftTrain's `-v29`
    (tasks/drafttrain-spec.md D7), the `-v22` one (library, capabilities) on its own artifact, the S3 variant with
    trained BF16 `mtp.*` draft weights and a re-derived draft token map: same tokenizer, template and source revision
    as S3. Its runtime config differs from `-v22`'s only in the deployment, the profile id and the artifact
    (directory, manifest digest, the sidecar's artifact). No other config carries `-v29`."""

    def test_pin_is_an_s3_variant(self):
        pin, s3 = load_json(ROOT / DRAFT_PIN), load_json(ROOT / S3_PIN)
        for key in ('profile_id', 'tokenizer_hash', 'template_hash', 'source_revision', 'schema_version'):
            with self.subTest(key=key):
                self.assertEqual(pin[key], s3[key])
        self.assertNotEqual((pin['manifest_sha256'], pin['model_hash']), (s3['manifest_sha256'], s3['model_hash']))
        self.assertTrue(pin['artifact_id'].endswith('-' + pin['manifest_sha256'][:16]))
        self.assertEqual(pin['path'], '/var/lib/cere/artifacts/' + pin['artifact_id'])

    def test_deployment_is_the_v22_one_on_the_draft_artifact(self):
        d = validate_deployment(ROOT, DEFAULT_SPEC_PREFIX)
        self.assertEqual(d['profile']['profile_id'], DEFAULT_SPEC_PROFILE_ID)
        pin = load_json(ROOT / DRAFT_PIN)
        caps = part(DEFAULT_SPEC_PREFIX, 'capabilities')
        self.assertEqual(caps['artifact_sha256'], pin['manifest_sha256'])
        except_caps = ('profile_id', 'notes', 'artifact_sha256')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in part(V22_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(DEFAULT_SPEC_PREFIX, 'recipe'), part(DEFAULT_SPEC_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id'], pin['lock_id']),
                         ('qwen38-flash-next-native-spec', 'qwen38-flash-next-native-spec-lock-v1',
                          'qwen38-flash-next-native-spec-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (DEFAULT_SPEC_PROFILE_ID, DEFAULT_SPEC_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id', 'model_ref')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(V22_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{DEFAULT_SPEC_PREFIX}.recipe.json').read_bytes()).hexdigest())
        self.assertEqual((lock['artifact_sha256'], lock['model_content_sha256']),
                         (pin['manifest_sha256'], pin['model_hash']))
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256', 'artifact_sha256', 'model_content_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(V22_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_v22_on_the_draft_artifact(self):
        x = tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text())
        validate('runtime_config', x)
        default_spec_before_flips(self, x)
        base = tomllib.loads((ROOT / V22_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        pin = load_json(ROOT / DRAFT_PIN)
        self.assertEqual((entry['deployment'], entry['profile_id']), (DEFAULT_SPEC_PREFIX, DEFAULT_SPEC_PROFILE_ID))
        self.assertEqual((entry['native']['artifact_dir'], entry['native']['expected_manifest_sha256']),
                         (pin['path'], pin['manifest_sha256']))
        self.assertEqual(entry['native']['sidecar'][-1], pin['path'])
        self.assertEqual(entry['native']['library_sha256'], part(DEFAULT_SPEC_PREFIX, 'lock')['backend_build_sha256'])
        for e in (entry, b):
            n = e['native']
            del e['deployment'], e['profile_id'], n['artifact_dir'], n['expected_manifest_sha256'], n['sidecar'][-1]
        self.assertEqual(x, base)

    def test_the_only_v29_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(DEFAULT_SPEC_PROFILE_ID in ids,
                                 path.relative_to(ROOT).as_posix() == DEFAULT_SPEC_TOML)


class GrammarSpecDeployment(unittest.TestCase):
    """GrammarSpec (tasks/grammar-spec-spec.md): the opt-in `-v49` deployment is the default `-v29` one (artifact,
    library, capabilities) under its own id; its runtime config differs from `-v29`'s only in the deployment, the
    profile id and the added lever grammar_drafts "on". No other config carries `-v49`."""

    def test_deployment_is_the_default_one_under_its_own_id(self):
        d = validate_deployment(ROOT, GRAMMAR_PREFIX)
        self.assertEqual(d['profile']['profile_id'], GRAMMAR_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(GRAMMAR_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(GRAMMAR_PREFIX, 'recipe'), part(GRAMMAR_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-grammar', 'qwen38-flash-next-native-spec-grammar-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (GRAMMAR_PROFILE_ID, GRAMMAR_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{GRAMMAR_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_default_one_plus_grammar_drafts(self):
        x = tomllib.loads((ROOT / GRAMMAR_TOML).read_text())
        validate('runtime_config', x)
        pop_l8_share(self, x)
        base = default_spec_before_flips(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (GRAMMAR_PREFIX, GRAMMAR_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(GRAMMAR_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual(entry['native']['levers']['grammar_drafts'], 'on')
        self.assertNotIn('grammar_drafts', b['native']['levers'])
        del entry['native']['levers']['grammar_drafts']
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)

    def test_the_only_v49_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(GRAMMAR_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == GRAMMAR_TOML)


class L8SharedSegment(unittest.TestCase):
    """L8 (tasks/beat-sglang-spec.md, maintainer 2026-11-01): exactly the GPU0 dev speculation profiles -v29, -v52,
    -v49 and -v47 (MixedPD, -v29 plus the prefill slice) serve the O47 shared PLE segment, kept, in the GPU0 dev directory /dev/shm/cere-host-g0 (never the dev loop's
    /dev/shm/cere-host); prod-t4 shares no segment until a production window decides it."""

    def test_only_the_gpu0_speculation_profiles_share_and_keep_in_the_dev_directory(self):
        want = {DEFAULT_SPEC_TOML, V52_TOML, GRAMMAR_TOML, MIXEDPD_TOML}
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            rel = path.relative_to(ROOT).as_posix()
            for entry in tomllib.loads(path.read_text()).get('deployments', []):
                native = entry.get('native', {})
                with self.subTest(path=rel):
                    got = {k: native[k] for k in L8_SHARE if k in native}
                    if rel in want:
                        self.assertEqual(got, L8_SHARE)
                    elif rel != 'configs/runtime/native-2gpu.toml':  # the two-card replicas' own share (D21)
                        self.assertEqual(got, {})


class ProdT4Config(unittest.TestCase):
    """tasks/g9-prod-spec.md "Production shape": `prod-t4.toml` is the Sessions config (`-v53`, tasks/sessions-spec.md
    SE-11: Ctx262 x PrefixHit's `-v52` with 262,144 context and alternative prefix-cache edges plus the M2.10 host
    tier; production window 10, SE-07 (b)) at 16 rows since production window 11 (BatchConc BC7, owner 2026-11-11):
    the 16-row deployment `-v58` and its profile id, 16 in flight and 16 per scope, a 262,144-token KV pool (the
    base's 524,288 for 4 rows); otherwise only the bind (sglang-t4's :18080), GPU1, the state dir, the
    `frontend` scope (also the keyless scope, F2; DRR weight 4) and the bearer-only `ae` scope (weight 1; Axon
    authoring-eval and unattended-repair runs, owner 2026-11-10), the output cap and the default deadline changed
    (ApiFit AF-10: 32,768 tokens, 900 s; it was frontend's 16,384, F3), the Anthropic compat routes on (production
    window 1: frontend's count_tokens_upstream; ApiFit owner item 5), `ledger.max_records` 100,000 (the base's
    10,000 filled at ~167 requests/min with the 1 h retention: a full ledger is 503 for every scope) without
    `ledger.evict_terminal_when_full` (tasks/ops-hardening-spec.md OH-03: a rollback to a release without eviction
    refuses fresh requests on a full ledger until retention frees records) and an 8 GiB host arena (the base's 16 GiB;
    SE-07 (b)). Its `api.max_messages` 4,096 is the base's own (tasks/max-messages-spec.md MM-02: every
    `native-g0-spec*.toml`). Production window 5 (owner 2026-10-25) added four opt-ins on the same deployment and
    profile id: prefix_state_slots 48 (PT0b; 40 is the window's fallback commit when GPU1's free at peak is below
    10,240 MiB), MixedPD's prefill_slice_ms 200, GrammarSpec's lever grammar_drafts "on" (off 2026-10-28 by
    quality-eval D18, back on after GdFix #283 made it greedy-identical) and the loop guard table at its defaults."""

    def test_is_the_v53_config_at_16_rows_on_gpu1_with_the_frontend_api(self):
        prod = tomllib.loads((ROOT / PROD_T4_TOML).read_text())
        validate('runtime_config', prod)
        base = tomllib.loads((ROOT / V53_TOML).read_text())
        for key in L8_SHARE:  # production shares no PLE segment (L8 left prod-t4 to a production window)
            self.assertNotIn(key, prod['deployments'][0]['native'])
            self.assertNotIn(key, base['deployments'][0]['native'])
        self.assertEqual((prod['state']['host_spill_enabled'], base['state']['host_spill_enabled']), (True, True))
        self.assertEqual(prod['server']['bind'], '127.0.0.1:18080')
        self.assertEqual(prod['deployments'][0]['native']['device_uuid'], GPU1_UUID)
        self.assertEqual(prod['scheduler']['devices'], ['gpu1'])
        self.assertEqual(prod['ledger']['state_dir'], '/var/lib/cere/prod-t4')
        self.assertEqual((prod['ledger']['max_records'], base['ledger']['max_records']), (100000, 10000))
        self.assertEqual(prod['ledger']['terminal_retention_ms'], 3600000)
        self.assertNotIn('evict_terminal_when_full', prod['ledger'])
        self.assertEqual((prod['api']['max_output_tokens'], prod['api']['default_deadline_ms']), (32768, 900000))
        self.assertIs(prod['deployments'][0]['api_compat_anthropic'], True)
        self.assertEqual((prod['api']['max_messages'], base['api']['max_messages']), (4096, 4096))
        self.assertNotIn('api_compat_anthropic', base['deployments'][0])
        # priority design note: the starvation bound of frontend's low class (subagents, helpers) is pinned
        # to the runtime default rather than inherited from it.
        self.assertEqual((prod['admission'].get('aging_ms'), base['admission'].get('aging_ms')), (30000, None))
        self.assertEqual(prod['auth'], {'operator_token_env': 'CERE_OPERATOR_TOKEN', 'keyless_scope': 'frontend',
                                        'scopes': [{'scope_id': 'frontend', 'token_env': 'CERE_TOKEN_FRONTEND',
                                                    'weight': 4},
                                                   {'scope_id': 'ae', 'token_env': 'CERE_TOKEN_AE', 'weight': 1}]})
        self.assertEqual(prod['telemetry'].pop('clients'), PROD_T4_CLIENTS)
        self.assertNotIn('clients', base['telemetry'])
        for x in (prod, base):
            del x['server']['bind'], x['deployments'][0]['native']['device_uuid'], x['scheduler']['devices']
            del x['ledger']['state_dir'], x['ledger']['max_records']
            del x['api']['max_output_tokens'], x['api']['default_deadline_ms'], x['auth']
        del prod['deployments'][0]['api_compat_anthropic'], prod['admission']['aging_ms']
        # Window 11 (BC7): the 16-row deployment, its admission and the pool of one context.
        for x, want in ((prod, (V58_PREFIX, V58_PROFILE_ID, 16, 262144)),
                        (base, (V53_PREFIX, V53_PROFILE_ID, 4, 524288))):
            d = x['deployments'][0]
            self.assertEqual((d.pop('deployment'), d.pop('profile_id'), x['admission'].pop('max_backend_inflight'),
                              d['native'].pop('kv_pool_tokens')), want)
            self.assertEqual(x['admission'].pop('per_scope_inflight'), want[2])
        native, base_native = prod['deployments'][0]['native'], base['deployments'][0]['native']
        self.assertEqual((native.pop('snapshot_host_bytes'), base_native.pop('snapshot_host_bytes')),
                         (8 << 30, SESSIONS_ARENA_BYTES))
        self.assertIn(native['prefix_state_slots'], (40, 48))
        self.assertEqual(base_native['prefix_state_slots'], 32)
        self.assertEqual(native.pop('prefill_slice_ms'), 200)
        self.assertEqual(native['levers'].pop('grammar_drafts'), 'on')
        self.assertEqual(native.pop('loop_guard'), {'min_period_bytes': 16, 'max_period_bytes': 1024, 'min_repeats': 4,
                                                    'min_span_bytes': 512, 'tool_repeat_run': 10,
                                                    'ignore_arg_keys': ['description']})
        for key in ('prefill_slice_ms', 'loop_guard', 'lora'):
            self.assertNotIn(key, base_native)
            self.assertNotIn(key, native)
        self.assertNotIn('grammar_drafts', base_native['levers'])
        native['prefix_state_slots'] = base_native['prefix_state_slots']
        self.assertEqual(prod, base)

    def test_a_config_without_the_keyless_scope_bound_to_a_scope_is_refused(self):
        prod = tomllib.loads((ROOT / PROD_T4_TOML).read_text())
        prod['auth']['keyless_scope'] = 'native-a'
        with self.assertRaises(ContractError):
            validate('runtime_config', prod)


class Ctx262Deployment(unittest.TestCase):
    """Ctx262 CX-02 (tasks/ctx262-spec.md): the opt-in `-v44` deployment is the default `-v29` one (artifact, levers,
    library) under its own id with every context field 262,144; its runtime config differs from `-v29`'s only in the
    deployment, the profile id, max_context_tokens 262,144 and kv_pool_tokens 524,288. No other config carries
    `-v44`."""

    @staticmethod
    def contexts(value, path=''):
        """{json path: value} of every `max_context_tokens` / `effective_context_tokens` / `context_tokens` field."""
        out = {}
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ('max_context_tokens', 'effective_context_tokens', 'context_tokens'):
                    out[f'{path}.{k}'] = v
                else:
                    out |= Ctx262Deployment.contexts(v, f'{path}.{k}')
        elif isinstance(value, list):
            for i, v in enumerate(value):
                out |= Ctx262Deployment.contexts(v, f'{path}[{i}]')
        return out

    def without_contexts(self, value):
        if isinstance(value, dict):
            return {k: self.without_contexts(v) for k, v in value.items() if k not in
                    ('max_context_tokens', 'effective_context_tokens', 'context_tokens')}
        if isinstance(value, list):
            return [self.without_contexts(v) for v in value]
        return value

    def test_deployment_is_the_v29_one_at_262144(self):
        d = validate_deployment(ROOT, CTX262_PREFIX)
        self.assertEqual(d['profile']['profile_id'], CTX262_PROFILE_ID)
        for kind in ('capabilities', 'recipe', 'lock'):
            with self.subTest(kind=kind):
                mine, base = part(CTX262_PREFIX, kind), part(DEFAULT_SPEC_PREFIX, kind)
                self.assertEqual(set(self.contexts(mine).values()), {262144})
                self.assertEqual(set(self.contexts(base).values()), {200000})
                self.assertEqual(self.contexts(mine).keys(), self.contexts(base).keys())
        caps = part(CTX262_PREFIX, 'capabilities')
        self.assertEqual(len(self.contexts(caps)), 19)
        self.assertIn('262,144', caps['notes'])
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in self.without_contexts(caps).items() if k not in except_caps},
                         {k: v for k, v in self.without_contexts(part(DEFAULT_SPEC_PREFIX, 'capabilities')).items()
                          if k not in except_caps})
        recipe, lock = part(CTX262_PREFIX, 'recipe'), part(CTX262_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-262k', 'qwen38-flash-next-native-spec-262k-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (CTX262_PROFILE_ID, CTX262_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id', 'context_tokens')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{CTX262_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256', 'effective_context_tokens')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_v29_at_262144_with_a_524288_pool(self):
        x = tomllib.loads((ROOT / CTX262_TOML).read_text())
        validate('runtime_config', x)
        base = default_spec_before_flips(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (CTX262_PREFIX, CTX262_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(CTX262_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual((entry['native']['max_context_tokens'], b['native']['max_context_tokens']), (262144, 200000))
        self.assertEqual((entry['native']['kv_pool_tokens'], b['native']['kv_pool_tokens']), (524288, 300032))
        self.assertEqual(entry['native']['kv_pool_tokens'] % 64, 0)
        self.assertLessEqual(entry['native']['kv_pool_tokens'], 1048576)  # the CX-01 A4 support bound
        for e in (entry, b):
            del e['deployment'], e['profile_id'], e['native']['max_context_tokens'], e['native']['kv_pool_tokens']
        self.assertEqual(x, base)

    def test_the_only_v44_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(CTX262_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == CTX262_TOML)




class VisionLongDeployment(unittest.TestCase):
    """VisionLong (tasks/m213l-spec.md): the opt-in `-v43` deployment is the default `-v29` one (lock, recipe, text
    capabilities) on the vision variant of the `-v29` artifact, plus image combinations at the text shape (prefill;
    decode and verify of text and tool_calls, greedy and sampled, rows 1-4, the 200,000-token context); its runtime
    config differs from `-v29`'s only in the deployment, the profile id, the artifact, the sidecar feature `image` (its
    processor on the CPU: 4 threads, 2 GiB result cache) and the ledger state_dir. No other config carries `-v43`."""

    def test_artifact_profile_is_the_text_one_with_the_vision_tower_on_the_device(self):
        text, vision = load_json(ROOT / TEXT_ARTIFACT_PROFILE), load_json(ROOT / VISION_ARTIFACT_PROFILE)
        self.assertEqual((vision['profile_id'], vision['modalities'], vision['placement']['vision']),
                         ('rtxpro6000bw-plehost-nvfp4-mxfp8-vision-v1', ['text', 'image'], 'device'))
        for d in (text, vision):
            del d['profile_id'], d['modalities'], d['placement']['vision']
        self.assertEqual(vision, text)

    def test_pin_is_the_draft_artifact_plus_vision(self):
        pin, draft = load_json(ROOT / VISION_LONG_PIN), load_json(ROOT / DRAFT_PIN)
        for key in ('tokenizer_hash', 'template_hash', 'source_revision', 'schema_version'):
            with self.subTest(key=key):
                self.assertEqual(pin[key], draft[key])
        self.assertEqual(pin['profile_id'], load_json(ROOT / VISION_ARTIFACT_PROFILE)['profile_id'])
        self.assertNotEqual((pin['manifest_sha256'], pin['model_hash']), (draft['manifest_sha256'], draft['model_hash']))
        self.assertTrue(pin['artifact_id'].endswith('-' + pin['manifest_sha256'][:16]))
        self.assertEqual(pin['path'], '/var/lib/cere/artifacts/' + pin['artifact_id'])
        self.assertEqual(pin['lock_id'], 'qwen38-flash-next-native-spec-vision-lock-v1')

    def test_recipe_is_the_v29_one_under_its_own_ids(self):
        recipe = part(VISION_LONG_PREFIX, 'recipe')
        self.assertEqual((recipe['recipe_id'], recipe['backend_profile_id']),
                         ('qwen38-flash-next-native-spec-vision', VISION_LONG_PROFILE_ID))
        self.assertTrue(recipe['model_ref'].endswith('/draft-data/artifact-src/r2m-vision'))
        except_recipe = ('recipe_id', 'backend_profile_id', 'model_ref')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'recipe').items() if k not in except_recipe})

    def test_capabilities_are_v29_plus_image_combinations_at_the_text_shape(self):
        caps, base = part(VISION_LONG_PREFIX, 'capabilities'), part(DEFAULT_SPEC_PREFIX, 'capabilities')
        validate('execution_capabilities', caps)
        pin = load_json(ROOT / VISION_LONG_PIN)
        self.assertEqual((caps['profile_id'], caps['artifact_sha256'], caps['modalities']),
                         (VISION_LONG_PROFILE_ID, pin['manifest_sha256'], ['text', 'image']))
        combos = caps['validated_combinations']
        self.assertEqual([c for c in combos if c['modality'] == 'text'], base['validated_combinations'])
        images = sorted((c['operation'], c['output'], c['sampler_mode'], tuple(c['row_counts']), c['max_context_tokens'])
                        for c in combos if c['modality'] == 'image')
        self.assertEqual(images, sorted([('prefill', 'text', 'not_applicable', (1,), 200000)] + [
            (op, out, s, (1, 2, 3, 4), 200000) for op in ('decode', 'verify') for out in ('text', 'tool_calls')
            for s in ('greedy', 'sampled')]))
        except_caps = ('profile_id', 'notes', 'artifact_sha256', 'modalities', 'validated_combinations')
        self.assertEqual({k: v for k, v in caps.items() if k not in except_caps},
                         {k: v for k, v in base.items() if k not in except_caps})

    def test_deployment_is_the_v29_one_on_the_vision_artifact(self):
        """Needs the `-v43` lock (written with the relocked library): the v29 lock under its own ids and artifact."""
        d = validate_deployment(ROOT, VISION_LONG_PREFIX)
        self.assertEqual(d['profile']['profile_id'], VISION_LONG_PROFILE_ID)
        pin, lock = load_json(ROOT / VISION_LONG_PIN), part(VISION_LONG_PREFIX, 'lock')
        self.assertEqual((lock['lock_id'], lock['backend_profile_id']), (pin['lock_id'], VISION_LONG_PROFILE_ID))
        self.assertEqual((lock['artifact_sha256'], lock['model_content_sha256'], lock['tokenizer_sha256'],
                          lock['template_sha256']),
                         (pin['manifest_sha256'], pin['model_hash'], pin['tokenizer_hash'], pin['template_hash']))
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{VISION_LONG_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256', 'artifact_sha256', 'model_content_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(DEFAULT_SPEC_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_v29_on_the_vision_artifact_with_the_image_feature(self):
        x = tomllib.loads((ROOT / VISION_LONG_TOML).read_text())
        validate('runtime_config', x)
        base = default_spec_before_flips(self, tomllib.loads((ROOT / DEFAULT_SPEC_TOML).read_text()))
        entry, b = x['deployments'][0], base['deployments'][0]
        pin = load_json(ROOT / VISION_LONG_PIN)
        self.assertEqual((entry['deployment'], entry['profile_id']), (VISION_LONG_PREFIX, VISION_LONG_PROFILE_ID))
        self.assertEqual((entry['native']['artifact_dir'], entry['native']['expected_manifest_sha256']),
                         (pin['path'], pin['manifest_sha256']))
        self.assertEqual(entry['native']['sidecar'][3:], [pin['path'], '--feature', 'image', '--image-device', 'cpu',
                                                          '--image-threads', '4', '--image-cache-mib', '2048'])
        self.assertEqual(entry['native']['library_sha256'], b['native']['library_sha256'])
        self.assertEqual(x['ledger']['state_dir'], '/var/lib/cere/native-g0-spec-vision')
        for e in (entry, b):
            n = e['native']
            del e['deployment'], e['profile_id'], n['artifact_dir'], n['expected_manifest_sha256'], n['sidecar'][3:]
        for c in (x, base):
            del c['ledger']['state_dir']
        self.assertEqual(x, base)

    def test_the_only_v43_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(VISION_LONG_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == VISION_LONG_TOML)


class VisionDiagnosticDeployment(unittest.TestCase):
    """VisionLong Amendment A2 (tasks/m213l-spec.md): the dev-only diagnostic `-v54` deployment is the `-v43` one
    (artifact, library, capabilities) under its own id; its runtime config differs from `-v43`'s only in the deployment,
    the profile id and `native.prefix_alt_edges = true`. No other config carries `-v54`."""

    def test_deployment_is_the_v43_one_under_its_own_id(self):
        d = validate_deployment(ROOT, VISION_V54_PREFIX)
        self.assertEqual(d['profile']['profile_id'], VISION_V54_PROFILE_ID)
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in part(VISION_V54_PREFIX, 'capabilities').items() if k not in except_caps},
                         {k: v for k, v in part(VISION_LONG_PREFIX, 'capabilities').items() if k not in except_caps})
        recipe, lock = part(VISION_V54_PREFIX, 'recipe'), part(VISION_V54_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-vision-v54', 'qwen38-flash-next-native-spec-vision-v54-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (VISION_V54_PROFILE_ID, VISION_V54_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(VISION_LONG_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{VISION_V54_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(VISION_LONG_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_the_v43_one_plus_alt_edges(self):
        x = tomllib.loads((ROOT / VISION_V54_TOML).read_text())
        validate('runtime_config', x)
        base = tomllib.loads((ROOT / VISION_LONG_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (VISION_V54_PREFIX, VISION_V54_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(VISION_V54_PREFIX, 'lock')['backend_build_sha256'])
        self.assertIs(entry['native'].pop('prefix_alt_edges'), True)
        self.assertNotIn('prefix_alt_edges', b['native'])
        for e in (entry, b):
            del e['deployment'], e['profile_id']
        self.assertEqual(x, base)

    def test_the_only_v54_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(VISION_V54_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == VISION_V54_TOML)


class VisionContextDiagnosticDeployment(unittest.TestCase):
    """Ctx262 CX-R-vision (tasks/ctx262-spec.md CX-R18): the dev-only diagnostic `-v55` deployment is VisionLong's `-v54`
    one (artifact, library, capabilities) under its own id with every context field 262,144; its runtime config differs
    from `-v54`'s only in the deployment, the profile id, max_context_tokens 262,144, kv_pool_tokens 479,168 (`-v44`'s
    524,288 cut by the CX-R-vision shrink rule, revision CX-R18-limit) and executor_memory_limit_bytes (the schema
    bound, 86,432 MiB). No other config carries `-v55`."""

    without_contexts = Ctx262Deployment.without_contexts

    def test_deployment_is_the_v54_one_at_262144(self):
        d = validate_deployment(ROOT, VISION_V55_PREFIX)
        self.assertEqual(d['profile']['profile_id'], VISION_V55_PROFILE_ID)
        contexts, without = Ctx262Deployment.contexts, self.without_contexts
        for kind in ('capabilities', 'recipe', 'lock'):
            with self.subTest(kind=kind):
                mine, base = part(VISION_V55_PREFIX, kind), part(VISION_V54_PREFIX, kind)
                self.assertEqual(set(contexts(mine).values()), {262144})
                self.assertEqual(set(contexts(base).values()), {200000})
                self.assertEqual(contexts(mine).keys(), contexts(base).keys())
        except_caps = ('profile_id', 'notes')
        self.assertEqual({k: v for k, v in without(part(VISION_V55_PREFIX, 'capabilities')).items()
                          if k not in except_caps},
                         {k: v for k, v in without(part(VISION_V54_PREFIX, 'capabilities')).items()
                          if k not in except_caps})
        recipe, lock = part(VISION_V55_PREFIX, 'recipe'), part(VISION_V55_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-spec-vision-v55', 'qwen38-flash-next-native-spec-vision-v55-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (VISION_V55_PROFILE_ID, VISION_V55_PROFILE_ID))
        except_recipe = ('recipe_id', 'backend_profile_id', 'context_tokens')
        self.assertEqual({k: v for k, v in recipe.items() if k not in except_recipe},
                         {k: v for k, v in part(VISION_V54_PREFIX, 'recipe').items() if k not in except_recipe})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{VISION_V55_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_lock = ('lock_id', 'backend_profile_id', 'recipe_sha256', 'effective_context_tokens')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_lock},
                         {k: v for k, v in part(VISION_V54_PREFIX, 'lock').items() if k not in except_lock})

    def test_runtime_config_is_v54_at_262144_with_a_shrunk_pool(self):
        x = tomllib.loads((ROOT / VISION_V55_TOML).read_text())
        validate('runtime_config', x)
        base = tomllib.loads((ROOT / VISION_V54_TOML).read_text())
        entry, b = x['deployments'][0], base['deployments'][0]
        self.assertEqual((entry['deployment'], entry['profile_id']), (VISION_V55_PREFIX, VISION_V55_PROFILE_ID))
        self.assertEqual(entry['native']['library_sha256'], part(VISION_V55_PREFIX, 'lock')['backend_build_sha256'])
        n, bn = entry['native'], b['native']
        self.assertEqual((n['max_context_tokens'], n['kv_pool_tokens']), (262144, 479168))
        # the shrink rule: -v44's pool minus ceil(608 MiB / 14,144 B) tokens in whole 64-token pages, where 608 MiB is
        # the projected 87,040 MiB estimate over the schema bound
        ctx_pool = tomllib.loads((ROOT / CTX262_TOML).read_text())['deployments'][0]['native']['kv_pool_tokens']
        cut = -(-((87040 - 86432) << 20) // 14144)
        self.assertEqual(n['kv_pool_tokens'], ctx_pool - -(-cut // 64) * 64)
        self.assertEqual((n['executor_memory_limit_bytes'], bn['executor_memory_limit_bytes']),
                         (86432 << 20, 86016 << 20))
        for e in (entry, b):
            del e['deployment'], e['profile_id']
            for k in ('max_context_tokens', 'kv_pool_tokens', 'executor_memory_limit_bytes'):
                del e['native'][k]
        self.assertEqual(x, base)

    def test_the_only_v55_config(self):
        for path in sorted((ROOT / 'configs/runtime').glob('*.toml')):
            ids = {e.get('profile_id') for e in tomllib.loads(path.read_text()).get('deployments', [])}
            with self.subTest(path=path.name):
                self.assertEqual(VISION_V55_PROFILE_ID in ids, path.relative_to(ROOT).as_posix() == VISION_V55_TOML)


class S2EvalConfigs(unittest.TestCase):
    """FP4-Local FL-14 (tasks/fp4local-spec.md "S2 quality plan"): one eval config per S2 arm, each its base profile's
    config (CUR `-v19`, FP4L `-v19-fp4local`, and the default flip's S3 FP4L `-v21`: profile, deployment, levers,
    artifact, library) with only `[api] max_output_tokens` 131,072 (AIME26/GPQA ask 130,000; `-v19` caps 32,768 since
    tasks/api-lenient-spec.md AL-5) and
    `default_deadline_ms` = `max_deadline_ms` (tool-eval-bench sends no deadline header). lock_native.py pins them
    (LockNative)."""

    def test_each_is_its_base_config_with_the_eval_api_limits(self):
        for rel, (base_rel, profile_id) in EVAL_TOMLS.items():
            with self.subTest(toml=rel):
                x = tomllib.loads((ROOT / rel).read_text())
                validate('runtime_config', x)
                base = tomllib.loads((ROOT / base_rel).read_text())
                self.assertEqual(x['deployments'][0]['profile_id'], profile_id)
                api = x['api']
                self.assertEqual((api['max_output_tokens'], api['default_deadline_ms']),
                                 (131_072, api['max_deadline_ms']))
                self.assertEqual((base['api']['max_output_tokens'], base['api']['max_deadline_ms']),
                                 (32_768, api['max_deadline_ms']))
                for doc in (x, base):
                    del doc['api']['max_output_tokens'], doc['api']['default_deadline_ms']
                self.assertEqual(x, base)

    def test_the_eval_limits_fit_the_context_and_the_lock_pins_the_library(self):
        for rel in EVAL_TOMLS:
            with self.subTest(toml=rel):
                x = tomllib.loads((ROOT / rel).read_text())
                entry = x['deployments'][0]
                self.assertLess(x['api']['max_output_tokens'], entry['native']['max_context_tokens'])
                self.assertEqual(entry['native']['library_sha256'],
                                 part(entry['deployment'], 'lock')['backend_build_sha256'])


class TwoGpuDeployment(unittest.TestCase):
    """M3.2N (tasks/m32n-spec.md D1/D2/D5/D16): the two-card deployment is the `-v19` speculation one (levers, prefix
    cache, drafts) with two native replicas instead of `native.device_uuid`, both cards in scheduler.devices, one
    shared PLE segment (host_share_dir) and the global cap replicas x rows, on the shipped library."""

    def test_deployment_is_the_spec_one_under_its_own_ids(self):
        d = validate_deployment(ROOT, TWO_GPU_PREFIX)
        self.assertEqual(d['profile']['profile_id'], TWO_GPU_PROFILE_ID)
        caps, spec_caps = part(TWO_GPU_PREFIX, 'capabilities'), part(SPEC_PREFIX, 'capabilities')
        self.assertIn('M3.2N', caps['notes'])
        self.assertEqual({k: v for k, v in caps.items() if k not in ('profile_id', 'notes')},
                         {k: v for k, v in spec_caps.items() if k not in ('profile_id', 'notes')})
        recipe, lock = part(TWO_GPU_PREFIX, 'recipe'), part(TWO_GPU_PREFIX, 'lock')
        self.assertEqual((recipe['recipe_id'], lock['lock_id']),
                         ('qwen38-flash-next-native-2gpu', 'qwen38-flash-next-native-2gpu-lock-v1'))
        self.assertEqual((recipe['backend_profile_id'], lock['backend_profile_id']),
                         (TWO_GPU_PROFILE_ID, TWO_GPU_PROFILE_ID))
        self.assertEqual({k: v for k, v in recipe.items() if k not in ('recipe_id', 'backend_profile_id')},
                         {k: v for k, v in part(SPEC_PREFIX, 'recipe').items()
                          if k not in ('recipe_id', 'backend_profile_id')})
        self.assertEqual(lock['recipe_sha256'],
                         hashlib.sha256((ROOT / f'{TWO_GPU_PREFIX}.recipe.json').read_bytes()).hexdigest())
        except_ = ('lock_id', 'backend_profile_id', 'recipe_sha256')
        self.assertEqual({k: v for k, v in lock.items() if k not in except_},
                         {k: v for k, v in part(SPEC_PREFIX, 'lock').items() if k not in except_})

    def test_runtime_config_is_the_spec_one_on_two_replicas(self):
        x = tomllib.loads((ROOT / TWO_GPU_TOML).read_text())
        validate('runtime_config', x)
        spec = tomllib.loads((ROOT / SPEC_TOML).read_text())
        entry, native = x['deployments'][0], x['deployments'][0]['native']
        self.assertEqual((entry.pop('deployment'), entry.pop('profile_id')), (TWO_GPU_PREFIX, TWO_GPU_PROFILE_ID))
        self.assertEqual(native['library_sha256'], part(TWO_GPU_PREFIX, 'lock')['backend_build_sha256'])
        self.assertEqual(native.pop('replicas'), [
            {'device': 'gpu0', 'device_uuid': 'GPU-aaaaaaaa-0000-4000-8000-000000000000'},
            {'device': 'gpu1', 'device_uuid': 'GPU-bbbbbbbb-1111-4111-8111-111111111111'}])
        self.assertEqual(native.pop('host_share_dir'), '/dev/shm/cere-host')
        self.assertEqual(x['scheduler'].pop('devices'), ['gpu0', 'gpu1'])
        self.assertEqual(x.pop('admission'), {'max_backend_inflight': 8, 'per_scope_inflight': 8, 'queue_depth': 32})
        self.assertEqual(x['server'].pop('bind'), '127.0.0.1:18339')
        self.assertEqual(x['ledger'].pop('state_dir'), '/var/lib/cere/native-2gpu')
        s = spec['deployments'][0]
        self.assertEqual(s['native'].pop('device_uuid'), 'GPU-aaaaaaaa-0000-4000-8000-000000000000')
        del s['deployment'], s['profile_id'], spec['scheduler']['devices'], spec['admission']
        del spec['server']['bind'], spec['ledger']['state_dir']
        self.assertEqual(x, spec)


class LockNative(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for rel in ('schemas', 'configs/deployments'):
            shutil.copytree(ROOT / rel, self.root / rel)
        (self.root / 'configs/runtime').mkdir()
        for rel in PINNED_TOMLS:
            shutil.copy(ROOT / rel, self.root / rel)

    def tearDown(self):
        self.tmp.cleanup()

    def files(self):
        return {rel: (self.root / rel).read_bytes() for rel in (f'{PREFIX}.lock.json', TOML)}

    def k2s_files(self):
        return {rel: (self.root / rel).read_bytes()
                for rel in (f'{K2S_PREFIX}.lock.json', K2S_TOML, f'{SPEC_K2S_PREFIX}.lock.json', SPEC_K2S_TOML)}

    def p4_files(self):
        return {rel: (self.root / rel).read_bytes() for rel in (f'{P4_PREFIX}.lock.json', P4_TOML)}

    def run_lock(self, build, *extra):
        path = self.root / 'build.json'
        path.write_text(json.dumps(build))
        return subprocess.run([sys.executable, str(LOCK_NATIVE), '--root', str(self.root), '--build-json', str(path),
                               *extra], capture_output=True, text=True, timeout=60)

    def test_clean_build_pins_the_library_in_the_lock_and_the_config(self):
        r = self.run_lock(build_json())
        self.assertEqual(r.returncode, 0, r.stderr)
        lock = part(PREFIX, 'lock', self.root)
        self.assertEqual(lock['backend_build_sha256'], LIBRARY_SHA)
        x = tomllib.loads((self.root / TOML).read_text())
        self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
        validate('runtime_config', x, self.root)
        before = part(PREFIX, 'lock')
        self.assertEqual({k: v for k, v in lock.items() if k != 'backend_build_sha256'},
                         {k: v for k, v in before.items() if k != 'backend_build_sha256'})
        self.assertIn(LIBRARY_SHA, r.stdout)

    def test_the_build_backend_selects_the_pinned_pair(self):
        """M23-D7: the P3b build (build.json `flags.moe_backend` p3b) pins the shipped pairs; a K7 reference build (no
        flag, or `k7`) pins no pair (exit 0, nothing written); any other backend is refused and writes nothing."""
        before = self.files()
        for flags in (K7_BUILD_FLAGS, dict(K7_BUILD_FLAGS, moe_backend='k7')):
            with self.subTest(flags=flags):
                r = self.run_lock(dict(build_json(), flags=flags))
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn('no deployment pins', r.stdout)
                self.assertEqual(self.files(), before)
        r = self.run_lock(build_json())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotEqual(self.files(), before)
        self.assertEqual(part(PREFIX, 'lock', self.root)['backend_build_sha256'], LIBRARY_SHA)
        pinned = self.files()
        r = self.run_lock(dict(build_json(), flags={'moe_backend': 'k8'}))
        self.assertEqual(r.returncode, 2)
        self.assertIn('moe_backend', r.stderr)
        self.assertEqual(self.files(), pinned)

    def test_the_build_model_selects_its_own_pairs(self):
        """SelfTune ST4 S3a/S3d: build.json `flags.model` names the descriptor a library is built for (absent: the
        Qwen one, whose pairs the backend selects). A DICE-8B build pins only DICE-8B's pairs (the default and
        SdarOptIn's opt-in twin) whatever its backend,
        never a Qwen one, and records nothing in the OH-10 history; an unknown model is refused and writes
        nothing."""
        before = self.files()
        history = self.root / 'bench/reports/native-locks.jsonl'
        for flags in (dict(K7_BUILD_FLAGS, model='dice-8b'), dict(build_json()['flags'], model='dice-8b')):
            with self.subTest(flags=flags):
                r = self.run_lock(dict(build_json(), flags=flags))
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(LIBRARY_SHA, r.stdout)
                self.assertEqual(self.files(), before)
                self.assertFalse(history.exists())
                self.assertEqual(part(DICE_PREFIX, 'lock', self.root)['backend_build_sha256'], LIBRARY_SHA)
                for toml in (DICE_TOML, DICE_CHANGING_TOML):
                    x = tomllib.loads((self.root / toml).read_text())
                    self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
                    validate('runtime_config', x, self.root)
        dice = (self.root / DICE_TOML).read_bytes()
        r = self.run_lock(build_json())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.root / DICE_TOML).read_bytes(), dice)
        r = self.run_lock(dict(build_json(), flags=dict(build_json()['flags'], model='llama-9b')))
        self.assertEqual(r.returncode, 2)
        self.assertIn("flags.model 'llama-9b' is none of ['dice-8b', 'inkling-2-mini', 'qwen38-flash-next']", r.stderr)
        r = self.run_lock(dict(build_json(), flags=dict(build_json()['flags'], model='inkling-2-mini')))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(part(INKLING_PREFIX, 'lock', self.root)['backend_build_sha256'], LIBRARY_SHA)
        self.assertEqual(part(INKLING_PREFIX + '-mtp', 'lock', self.root)['backend_build_sha256'], LIBRARY_SHA)
        self.assertEqual(part(INKLING_BATCH_PREFIX, 'lock', self.root)['backend_build_sha256'], LIBRARY_SHA)
        for toml in (INKLING_TOML, INKLING_S13_TOML, INKLING_PLAIN_TOML, INKLING_BATCH_TOML):
            x = tomllib.loads((self.root / toml).read_text())
            self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
            validate('runtime_config', x, self.root)
        self.assertEqual((self.root / DICE_TOML).read_bytes(), dice)

    def test_a_p3b_build_pins_every_lever_profile_with_the_shipped_one(self):
        """tasks/m23-spec.md PI-02: every lever is compiled into the one library, so the P3b build pins the `-v14` and
        `-v16` (levers.k2_gemm small_m), `-v12` (levers.prefill_attention p4) and `-v20` (levers.k2_weights fp8) pairs
        and the M3.2N two-card deployment together with the shipped one; FP4-Local FL-14 the two S2 eval configs
        with their base profiles' locks; Ctx262 CX-02 the `-v44` (262,144) pair and CX-09 the `-v52` and production
        pairs."""
        r = self.run_lock(build_json())
        self.assertEqual(r.returncode, 0, r.stderr)
        for prefix, toml in ((PREFIX, TOML), (K2S_PREFIX, K2S_TOML), (P4_PREFIX, P4_TOML),
                             (SPEC_K2S_PREFIX, SPEC_K2S_TOML), (SPEC_FP8W_PREFIX, SPEC_FP8W_TOML),
                             (SPEC_FP4LOCAL_PREFIX, SPEC_FP4LOCAL_TOML), (TWO_GPU_PREFIX, TWO_GPU_TOML),
                             (SPEC_PREFIX, SPEC_EVAL_TOML), (SPEC_FP4LOCAL_PREFIX, SPEC_FP4LOCAL_EVAL_TOML),
                             (S3_PREFIX, S3_EVAL_TOML), (DEFAULT_SPEC_PREFIX, DEFAULT_SPEC_TOML),
                             (SPEC_PREFIX, SPEC_TOML), (TREE_PREFIX, TREE_TOML), (V22_PREFIX, V22_TOML),
                             (CHAINS_PREFIX, CHAINS_TOML), (V51_PREFIX, V51_TOML), (V52_PREFIX, PROD_T4_TOML),
                             (SESSIONS_PREFIX, SESSIONS_TOML),
                             (VISION_LONG_PREFIX, VISION_LONG_TOML), (VISION_V54_PREFIX, VISION_V54_TOML),
                             (VISION_V55_PREFIX, VISION_V55_TOML),
                             (CTX262_PREFIX, CTX262_TOML), (V52_PREFIX, V52_TOML), (V53_PREFIX, V53_TOML),
                             (GRAMMAR_PREFIX, GRAMMAR_TOML),
                             (LORA_PREFIX, LORA_TOML), (V56_PREFIX, V56_TOML),
                             (RESEARCH_PREFIX, RESEARCH_TOML), (V58_PREFIX, V58_TOML), (V59_PREFIX, V59_TOML)):
            with self.subTest(toml=toml):
                self.assertEqual(part(prefix, 'lock', self.root)['backend_build_sha256'], LIBRARY_SHA)
                x = tomllib.loads((self.root / toml).read_text())
                self.assertEqual(x['deployments'][0]['native']['library_sha256'], LIBRARY_SHA)
                validate('runtime_config', x, self.root)

    def test_a_pre_lever_build_is_refused_and_writes_nothing(self):
        """A build.json with the retired CMake selections (`flags.prefill_attention`, `flags.k2_small_m`) is a library
        that selects kernels at build time, which no profile pins: refused (exit 2) whatever the value or backend."""
        before = (self.files(), self.p4_files(), self.k2s_files())
        for flags in PRE_LEVER_BUILD_FLAGS:
            with self.subTest(flags=flags):
                r = self.run_lock(dict(build_json(), flags=flags))
                self.assertEqual(r.returncode, 2)
                self.assertIn('pre-lever build', r.stderr)
                self.assertEqual((self.files(), self.p4_files(), self.k2s_files()), before)

    def test_check_of_a_p3b_build_reads_every_lever_pair(self):
        self.init_repo()
        build, sha = self.checked_build()
        self.assertEqual(self.run_lock(build, '--check').returncode, 2)
        self.assertEqual(self.run_lock(build).returncode, 0)
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 0, r.stderr)
        for toml in (TOML, K2S_TOML, P4_TOML, SPEC_K2S_TOML, SPEC_FP8W_TOML, SPEC_FP4LOCAL_TOML, TWO_GPU_TOML,
                     SPEC_EVAL_TOML, SPEC_FP4LOCAL_EVAL_TOML, S3_EVAL_TOML, DEFAULT_SPEC_TOML, SPEC_TOML, TREE_TOML,
                     V22_TOML, CHAINS_TOML, V51_TOML, SESSIONS_TOML, VISION_LONG_TOML, VISION_V54_TOML,
                     VISION_V55_TOML, PROD_T4_TOML, CTX262_TOML, V52_TOML, V53_TOML, GRAMMAR_TOML, LORA_TOML,
                     V56_TOML, RESEARCH_TOML, V58_TOML, V59_TOML):
            self.assertIn(toml, r.stdout)
        self.assertIn(sha, r.stdout)
        build['flags'] = P4_PRE_LEVER_FLAGS
        self.assertEqual(self.run_lock(build, '--check').returncode, 2)

    def test_check_of_a_k7_build_gates_the_library_and_the_tree(self):
        """`--check` of a K7 build reads no pair, still refuses a stale library, a library file other than
        build.json's and an executor/ tree other than the build's commit."""
        self.init_repo()
        build, sha = self.checked_build()
        build['flags'] = K7_BUILD_FLAGS
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('no deployment pins', r.stdout)
        for toml in (TOML, P4_TOML, K2S_TOML):
            self.assertNotIn(toml, r.stdout)
        self.assertIn(sha, r.stdout)
        self.assertEqual(self.run_lock(build, '--check', '--stale', sha).returncode, 2)
        bad, _ = self.checked_build(sha=PREV_BUILD_SHA)
        bad['flags'] = K7_BUILD_FLAGS
        self.assertEqual(self.run_lock(bad, '--check').returncode, 2)
        build, _ = self.checked_build()
        build['flags'] = K7_BUILD_FLAGS
        self.commit('executor/src.cpp', 'v2\n')
        build['git_head'] = self.git('rev-list', '--max-parents=0', 'HEAD')
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn('src.cpp', r.stderr)

    def test_refuses_a_dirty_build(self):
        before = self.files()
        r = self.run_lock(build_json(dirty=True))
        self.assertEqual(r.returncode, 2)
        self.assertIn('dirty', r.stderr)
        self.assertEqual(self.files(), before)

    def test_refuses_a_ccache_build(self):
        """build/ccache: a build whose compiles went through ccache (or whose build.json predates the record) is
        refused, by the relock and by `--check`; a relock is a clean, uncached build."""
        before = self.files()
        for ccache in ('4.12.3', None):
            with self.subTest(ccache=ccache):
                build = build_json(ccache=ccache)
                if ccache is None:
                    del build['ccache']
                for extra in ((), ('--check',)):
                    r = self.run_lock(build, *extra)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertIn('ccache', r.stderr)
                    self.assertEqual(self.files(), before)

    def test_refuses_a_build_without_the_executor_library(self):
        before = self.files()
        for outputs in ([], [{'file': 'libcere_executor.so', 'sha256': 'XYZ'}]):
            with self.subTest(outputs=outputs):
                r = self.run_lock(build_json(outputs=outputs))
                self.assertEqual(r.returncode, 2)
                self.assertEqual(self.files(), before)

    def test_refuses_a_fault_injection_build(self):
        """OH-08: a `build --fault-injection` library (build.json flags.cxx/cuda define CERE_FAULT_INJECTION) is
        refused by the relock and by `--check`, and `--check` refuses a library file holding the trigger's
        `CERE_FAULT_INJECT` bytes whatever its flags; the same build without them relocks."""
        before = self.files()
        for key in ('cxx', 'cuda'):
            flags = dict(P3B_BUILD_FLAGS, **{key: P3B_BUILD_FLAGS[key] + ' -DCERE_FAULT_INJECTION=1'})
            for extra in ((), ('--check',)):
                with self.subTest(key=key, extra=extra):
                    r = self.run_lock(dict(build_json(), flags=flags), *extra)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertIn('CERE_FAULT_INJECTION', r.stderr)
                    self.assertEqual(self.files(), before)
        self.assertFalse((self.root / 'bench/reports/native-locks.jsonl').exists())
        self.init_repo()
        build, _ = self.checked_build(library=b'lib getenv("CERE_FAULT_INJECT") lib')
        self.assertEqual(self.run_lock(build).returncode, 0)
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn('fault-injection library', r.stderr)
        build, _ = self.checked_build()
        self.assertEqual(self.run_lock(build).returncode, 0)
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_refuses_an_sdar_mutant_build(self):
        """SelfTune ST4 S4a: a `build --sdar-mutant NAME` library (build.json `mutant` NAME, flags.cxx/cuda define
        CERE_SDAR_MUTANT_<NAME>) is refused by name by the relock and by `--check`, writing nothing, whether
        build.json shows it by its `mutant` alone, by a define alone or by both; the same DICE-8B build without either
        relocks."""
        before = self.files()
        dice = dict(P3B_BUILD_FLAGS, model='dice-8b')
        for name in ('causal', 'fp32tie', 'changing_k_short'):
            define = f' -DCERE_SDAR_MUTANT_{name.upper()}=1'
            for build, says in (
                    (dict(build_json(), flags=dice, mutant=name), f'names the sdar mutant {name} (`mutant`)'),
                    (dict(build_json(), flags=dict(dice, cxx=dice['cxx'] + define, cuda=dice['cuda'] + define),
                          mutant=name), f'names the sdar mutant {name} (`mutant`)'),
                    (dict(build_json(), flags=dict(dice, cxx=dice['cxx'] + define)), f'sdar mutant {name} '),
                    (dict(build_json(), flags=dict(dice, cuda=dice['cuda'] + define)), f'sdar mutant {name} ')):
                for extra in ((), ('--check',)):
                    with self.subTest(name=name, mutant='mutant' in build, flags=build['flags'], extra=extra):
                        r = self.run_lock(build, *extra)
                        self.assertEqual(r.returncode, 2, r.stderr)
                        self.assertIn(says, r.stderr)
                        self.assertEqual(self.files(), before)
        r = self.run_lock(dict(build_json(), flags=dice))
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_relock_records_the_replaced_and_the_new_digests(self):
        """OH-10: a relock appends the new build's executor and ktest digests, git_head and backend and the library
        the shipped lock pinned before to bench/reports/native-locks.jsonl; relocking the same library again adds
        nothing, an unshipped variant or a refused relock nothing either."""
        history = self.root / 'bench/reports/native-locks.jsonl'
        shipped = part(PREFIX, 'lock', self.root)['backend_build_sha256']
        self.assertEqual(self.run_lock(dict(build_json(), flags=K7_BUILD_FLAGS)).returncode, 0)
        self.assertFalse(history.exists())
        r = self.run_lock(build_json())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f'native-locks.jsonl records it after {shipped}', r.stdout)
        first = {'executor_sha256': LIBRARY_SHA, 'ktest_sha256': 'b' * 64, 'git_head': 'c' * 40,
                 'moe_backend': 'p3b', 'replaced_executor_sha256': shipped}
        self.assertEqual([json.loads(line) for line in history.read_text().splitlines()], [first])
        self.assertEqual(self.run_lock(build_json()).returncode, 0)
        self.assertEqual(self.run_lock(build_json(dirty=True)).returncode, 2)
        self.assertEqual([json.loads(line) for line in history.read_text().splitlines()], [first])
        second = [{'file': 'libcere_executor.so', 'sha256': 'e' * 64, 'bytes': 1, 'needed': []},
                  {'file': 'libcere_ktest.so', 'sha256': 'f' * 64, 'bytes': 1, 'needed': []}]
        self.assertEqual(self.run_lock(build_json(outputs=second)).returncode, 0)
        self.assertEqual([json.loads(line) for line in history.read_text().splitlines()],
                         [first, dict(first, executor_sha256='e' * 64, ktest_sha256='f' * 64,
                                      replaced_executor_sha256=LIBRARY_SHA)])
        r = self.run_lock(build_json(outputs=second[:1] + [{'file': 'libcere_executor.so', 'sha256': 'a' * 64}]))
        self.assertEqual(r.returncode, 2)
        r = self.run_lock(build_json(outputs=[dict(second[0], sha256='9' * 64)]))
        self.assertEqual(r.returncode, 2)
        self.assertIn('libcere_ktest.so', r.stderr)
        self.assertEqual(len(history.read_text().splitlines()), 2)

    def test_the_committed_lock_history_ends_with_the_shipped_pin(self):
        """OH-10: bench/reports/native-locks.jsonl's last relock is the library the shipped lock pins, and each
        record replaced the one before it."""
        records = [json.loads(line) for line in (ROOT / 'bench/reports/native-locks.jsonl').read_text().splitlines()]
        self.assertEqual(records[-1]['executor_sha256'], part(PREFIX, 'lock')['backend_build_sha256'])
        for before, after in zip(records, records[1:]):
            self.assertEqual(after['replaced_executor_sha256'], before['executor_sha256'])

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=t', '-c', 'user.email=t@t',
                               '-c', 'commit.gpgsign=false', *args],
                              check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, rel, text):
        """Writes `rel` under the temp repo and commits it; the new HEAD."""
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_text(text)
        self.git('add', '-A')
        self.git('commit', '-qm', rel)
        return self.git('rev-parse', 'HEAD')

    def checked_build(self, library=b'executor library', head=None, **kw):
        """A clean build.json of `head` (default: the temp repo's HEAD) whose library file holds `library`; its
        `outputs[].sha256` is the file's hash unless `kw['sha']` overrides it."""
        (self.root / 'libcere_executor.so').write_bytes(library)
        sha = kw.get('sha', hashlib.sha256(library).hexdigest())
        build = build_json(outputs=[{'file': 'libcere_executor.so', 'sha256': sha, 'bytes': len(library),
                                     'needed': []},
                                    {'file': 'libcere_ktest.so', 'sha256': 'b' * 64, 'bytes': 1, 'needed': []}])
        build['git_head'] = head or self.git('rev-parse', 'HEAD')
        return build, sha

    def init_repo(self):
        self.git('init', '-q')
        return self.commit('executor/src.cpp', 'v1\n')

    def test_check_is_the_relock_finish_gate(self):
        """Z2AC-11 (D6, R1): `--check` writes nothing and exits 0 only when the lock and the TOML pin the library of
        the clean `build.json` and that library is none of the `--stale` ones (the `-v2` build at m25 R1)."""
        self.init_repo()
        stale = ('--stale', PREV_BUILD_SHA)
        build, sha = self.checked_build()
        before = self.files()
        # The committed files pin a library other than this build's: not relocked.
        r = self.run_lock(build, '--check', *stale)
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn(sha, r.stderr)
        self.assertEqual(self.files(), before)
        r = self.run_lock(self.checked_build(sha=PREV_BUILD_SHA)[0], '--check', *stale)
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertEqual(self.files(), before)
        # After the relock the gate passes, and fails again for a dirty build or a stale relocked library.
        build, sha = self.checked_build()
        self.assertEqual(self.run_lock(build).returncode, 0)
        relocked = self.files()
        r = self.run_lock(build, '--check', *stale)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(sha, r.stdout)
        self.assertEqual(self.run_lock(dict(build, git_dirty=True), '--check', *stale).returncode, 2)
        r = self.run_lock(build, '--check', '--stale', sha)
        self.assertEqual(r.returncode, 2)
        self.assertIn('stale', r.stderr)
        self.assertEqual(self.files(), relocked)
        # The TOML disagreeing with the lock fails the gate too.
        toml = self.root / TOML
        toml.write_text(toml.read_text().replace(f'library_sha256 = "{sha}"', f'library_sha256 = "{"e" * 64}"'))
        self.assertEqual(self.run_lock(build, '--check', *stale).returncode, 2)

    def test_check_ties_the_build_to_the_checked_out_executor(self):
        """Z2AC2-05: `--check` hashes the library file and refuses a build of a commit whose `executor/` differs from
        the checkout's (a later executor fix after R1), or of a commit the repository does not have."""
        old = self.init_repo()
        build, sha = self.checked_build()
        self.assertEqual(self.run_lock(build).returncode, 0)
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 0, r.stderr)
        # A later commit outside executor/ keeps the build valid.
        self.commit('tasks/notes.md', 'x\n')
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 0, r.stderr)
        # A later executor/ commit, or an uncommitted executor/ change, makes it a stale build.
        self.commit('executor/src.cpp', 'v2\n')
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn('executor/', r.stderr)
        self.assertIn(old, r.stderr)
        build, sha = self.checked_build()
        self.assertEqual(self.run_lock(build).returncode, 0)
        self.assertEqual(self.run_lock(build, '--check').returncode, 0)
        (self.root / 'executor/src.cpp').write_text('v3\n')
        self.assertEqual(self.run_lock(build, '--check').returncode, 2)
        (self.root / 'executor/src.cpp').write_text('v2\n')
        (self.root / 'executor/new.cpp').write_text('untracked\n')
        self.assertEqual(self.run_lock(build, '--check').returncode, 2)
        (self.root / 'executor/new.cpp').unlink()
        self.assertEqual(self.run_lock(build, '--check').returncode, 0)
        for name, b in (('unknown commit', dict(build, git_head='c' * 40)),
                        ('no git_head', {k: v for k, v in build.items() if k != 'git_head'})):
            with self.subTest(case=name):
                r = self.run_lock(b, '--check')
                self.assertEqual(r.returncode, 2, r.stdout)
                self.assertNotIn('Traceback', r.stderr)
        # The recorded SHA must be the library file's: a rebuilt (or missing) file next to an old build.json fails.
        (self.root / 'libcere_executor.so').write_bytes(b'another library')
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn('libcere_executor.so', r.stderr)
        (self.root / 'libcere_executor.so').unlink()
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertNotIn('Traceback', r.stderr)

    def init_build_tree(self, cmake=None):
        """A temp repo whose executor/ has the real build definition (or `cmake`), sources it consumes and files it
        does not: GPU oracles, docs, a host test. `cuda/src/host/support.cpp` reaches `tests/host/check.h` (and a
        not-yet-written `tests/host/gen.h`) through `..`, so both are build inputs although `tests/` is not named."""
        self.git('init', '-q')
        files = {
            'executor/CMakeLists.txt': cmake or (ROOT / 'executor/CMakeLists.txt').read_text(),
            'executor/include/cere_executor.h': '#pragma once\n',
            'executor/cuda/src/kernels/k1.cu': '#include "k1.h"\n',
            'executor/cuda/src/kernels/k1.h': '#pragma once\n',
            'executor/cuda/src/host/support.cpp': '#include "../../../tests/host/check.h"\n'
                                                  '#if __has_include("../../../tests/host/gen.h")\n#endif\n',
            'executor/cuda/README.md': 'kernels\n',
            'executor/README.md': 'abi\n',
            'executor/tests/gpu/test_kernels_k1.py': 'ORACLE = 1\n',
            'executor/tests/host/check.h': '#pragma once\n',
            'executor/tests/host/test_state.cpp': '#include "check.h"\n',
        }
        for rel, text in files.items():
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(text)
        self.git('add', '-A')
        self.git('commit', '-qm', 'executor')
        return self.relocked_build()

    def relocked_build(self):
        """The checked build of HEAD, relocked; the relock is committed so `git reset --hard` keeps it."""
        build, _ = self.checked_build()
        self.assertEqual(self.run_lock(build).returncode, 0)
        self.git('add', '-A')
        self.git('commit', '-qm', 'relock')
        self.assertEqual(self.run_lock(build, '--check').returncode, 0)
        return build

    def assert_check(self, build, rel, change, refused):
        """`--check` after `change` ('edit', 'commit', 'new' untracked, 'delete') of `rel`, then the tree restored."""
        path = self.root / rel
        old = path.read_bytes() if path.exists() else None
        if change == 'delete':
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((old or b'') + b'// changed\n')
        if change == 'commit':
            self.git('add', '--', 'executor/')
            self.git('commit', '-qm', rel)
        try:
            r = self.run_lock(build, '--check')
            self.assertEqual(r.returncode, 2 if refused else 0, f'{rel} {change}: {r.stdout}{r.stderr}')
            self.assertNotIn('Traceback', r.stderr)
            if refused:
                self.assertIn('executor/', r.stderr)
        finally:
            if change == 'commit':
                self.git('reset', '-q', '--hard', 'HEAD~1')
            elif old is None:
                path.unlink()
            else:
                path.write_bytes(old)
        self.assertEqual(self.run_lock(build, '--check').returncode, 0)

    def test_check_refuses_a_changed_build_input(self):
        """(a) A change to a file the build definition consumes (a source, a header, the CMakeLists, a header reached
        through `..`) or its deletion keeps refusing the build."""
        build = self.init_build_tree()
        for rel in ('executor/cuda/src/kernels/k1.cu', 'executor/cuda/src/kernels/k1.h',
                    'executor/include/cere_executor.h', 'executor/CMakeLists.txt', 'executor/tests/host/check.h'):
            for change in ('edit', 'commit', 'delete'):
                with self.subTest(rel=rel, change=change):
                    self.assert_check(build, rel, change, refused=True)

    def test_check_refuses_a_new_untracked_build_input(self):
        """(b) A new untracked file the build would consume (a globbed source, a header under an include directory, a
        `__has_include` target) refuses the build."""
        build = self.init_build_tree()
        for rel in ('executor/cuda/src/kernels/k2.cu', 'executor/cuda/src/ktest/k2_ktest.cpp',
                    'executor/include/cere_next.h', 'executor/tests/host/gen.h'):
            with self.subTest(rel=rel):
                self.assert_check(build, rel, 'new', refused=True)

    def test_check_ignores_executor_files_the_build_does_not_consume(self):
        """(c) GPU oracles, docs and host tests are not inputs of the pinned libraries: changing, committing or adding
        them keeps the build valid (M2.3: a `tests/gpu/*.py` fix forced a non-reproducible P3b rebuild)."""
        build = self.init_build_tree()
        for rel in ('executor/tests/gpu/test_kernels_k1.py', 'executor/README.md', 'executor/cuda/README.md',
                    'executor/tests/host/test_state.cpp'):
            for change in ('edit', 'commit', 'delete'):
                with self.subTest(rel=rel, change=change):
                    self.assert_check(build, rel, change, refused=False)
        with self.subTest(rel='new oracle'):
            self.assert_check(build, 'executor/tests/gpu/oracle_new.py', 'new', refused=False)

    def test_check_compares_all_of_executor_when_the_definition_reads_its_root(self):
        """Fail closed: a build definition that reads executor/ itself (a glob of the root) makes every executor/ file
        an input again."""
        cmake = (ROOT / 'executor/CMakeLists.txt').read_text() + 'file(GLOB_RECURSE all ${CMAKE_CURRENT_SOURCE_DIR}/*)\n'
        build = self.init_build_tree(cmake)
        self.assert_check(build, 'executor/tests/gpu/test_kernels_k1.py', 'edit', refused=True)

    def test_check_compares_all_of_executor_when_it_holds_a_symlink(self):
        """Fail closed: a symlink under executor/ can make any file a build input, so every executor/ file is one."""
        self.init_build_tree()
        (self.root / 'executor/cuda/src/kernels/k1_link.h').symlink_to('../../../tests/gpu/test_kernels_k1.py')
        self.git('add', '-A')
        self.git('commit', '-qm', 'symlink')
        build = self.relocked_build()
        self.assert_check(build, 'executor/tests/gpu/test_kernels_k1.py', 'edit', refused=True)

    def test_build_inputs_of_the_repository(self):
        """The derived input set of this checkout's HEAD: every library source and header is in, the GPU oracles and
        the executor docs are out."""
        sys.path.insert(0, str(LOCK_NATIVE.parent))
        import lock_native
        tree = lock_native.git_tree(ROOT, 'HEAD')
        inputs = lock_native.build_inputs(tree)
        self.assertIsNotNone(inputs)
        for rel in tree.files:
            if rel.startswith(('cuda/src/', 'include/')) or rel == 'CMakeLists.txt':
                self.assertTrue(lock_native.is_input(rel, inputs), rel)
            if rel.startswith('tests/gpu/') or rel in ('README.md', 'cuda/README.md', 'PROVENANCE.json'):
                self.assertFalse(lock_native.is_input(rel, inputs), rel)
        self.assertTrue(any(r.startswith('tests/gpu/') and r.endswith('.py') for r in tree.files))

    def test_check_validates_stale_values_and_refuses_malformed_pin_files(self):
        """Z2AC2-06: a `--stale` value that is not a lowercase 64-hex SHA-256 is a usage error (it could never match),
        and an unreadable or malformed lock or TOML is a refusal (exit 2), not a traceback."""
        self.init_repo()
        build, sha = self.checked_build()
        self.assertEqual(self.run_lock(build).returncode, 0)
        for value in (sha.upper(), sha[:63], 'sha256:' + sha, sha + '0', ''):
            with self.subTest(stale=value):
                r = self.run_lock(build, '--check', '--stale', value)
                self.assertEqual(r.returncode, 2, r.stdout)
                self.assertIn('--stale', r.stderr)
                self.assertNotIn('Traceback', r.stderr)
        lock_path, toml_path = self.root / f'{PREFIX}.lock.json', self.root / TOML
        lock_text, toml_text = lock_path.read_text(), toml_path.read_text()
        for name, path, text in (
                ('lock not json', lock_path, '{'),
                ('lock without backend_build_sha256', lock_path,
                 json.dumps({k: v for k, v in json.loads(lock_text).items() if k != 'backend_build_sha256'})),
                ('lock a list', lock_path, '[]'),
                ('toml syntax', toml_path, 'deployments = ['),
                ('toml no deployments', toml_path, 'x = 1\n'),
                ('toml empty deployments', toml_path, 'deployments = []\n'),
                ('toml no native table', toml_path, '[[deployments]]\nmode = "native"\n')):
            with self.subTest(case=name):
                path.write_text(text)
                r = self.run_lock(build, '--check')
                self.assertEqual(r.returncode, 2, r.stdout)
                self.assertIn('lock_native:', r.stderr)
                self.assertNotIn('Traceback', r.stderr)
                lock_path.write_text(lock_text)
                toml_path.write_text(toml_text)
        lock_path.unlink()
        r = self.run_lock(build, '--check')
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertNotIn('Traceback', r.stderr)


if __name__ == '__main__':
    unittest.main()
