//! Perf levers (m23 design note PI-01..PI-05): every perf lever is a runtime-selected implementation in the one
//! executor library, chosen per profile by `[deployments.native.levers]`. [`REGISTRY`] is the single source of
//! truth: the config validation, the ABI v5 `cere_slot_selection` block ([`LeverSet::ffi`]), the dev hot swap
//! (`POST /v1/cere/levers`), `GET /v1/cere/stats` `levers` and, through `tests/fixtures/lever_registry.json`, the
//! Python profile contract all read it. A lever's selection resolves to a [`LeverSet`] of plain enums at load or at a
//! hot swap (PI-03, PI-04), so a hot path never looks a lever up by name.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::native::ffi;

/// Which side runs a lever: the executor (a `cere_slot_selection` slot, ABI v5) or `cered` itself (e.g. the drafter's work flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    Executor,
    Runtime,
}

/// PI-04: a `hot` lever may change between requests (`cere_model_set_selection`); a `restart` one needs another
/// library or load, so the hot swap refuses changing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LeverClass {
    Hot,
    Restart,
}

/// Whether an implementation's output equals the default's bit for bit; a `changing` one ships only through a
/// batched D9 default flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Numerics {
    BitIdentical,
    Changing,
}

/// The GPU architecture an implementation is proven on (SelfTune D3): the CUDA library is built for `sm_120a`
/// (`executor/CMakeLists.txt` `CMAKE_CUDA_ARCHITECTURES`), the AMD HIP port (`executor/hip/build.sh`) for `gfx1151`.
/// The Vulkan executor (vulkan-amd design note, one arch per API, not per GPU) is `vulkan`.
/// A deployment's arch follows its `device_uuid`: `hip:<ordinal>` is `gfx1151`, `vk:<ordinal>` is `vulkan`, a `GPU-`
/// UUID `sm_120a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arch {
    #[serde(rename = "sm_120a")]
    Sm120a,
    Gfx1151,
    Vulkan,
}

impl Arch {
    pub fn name(self) -> &'static str {
        match self {
            Arch::Sm120a => "sm_120a",
            Arch::Gfx1151 => "gfx1151",
            Arch::Vulkan => "vulkan",
        }
    }
}

/// The model family a library is built for (SelfTune ST4: the descriptor's `model_type`, `gen::kFamily`,
/// `cere_model_desc.family`): `qwen4_exp` (Qwen3.8-Flash-Next, every shared impl), `sdar` (DICE-8B: block
/// diffusion over dense layers, one session, eager, BF16) or `inkling2_loop` (Inkling 2 Mini, inkling design note:
/// a looped NVFP4/MXFP8 MoE). An `sdar` or `inkling2_loop` library runs only each shared lever's default
/// ([`LeverSet::check_target`]); the registry's other shared impls are Qwen kernels it is not built with. A
/// family-scoped lever ([`LeverSpec::family`], e.g. `sdar_gemm`) is built in full on its family only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    #[default]
    Qwen4Exp,
    Sdar,
    #[serde(rename = "inkling2_loop")]
    Inkling2Loop,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Qwen4Exp => "qwen4_exp",
            Family::Sdar => "sdar",
            Family::Inkling2Loop => "inkling2_loop",
        }
    }

    /// The ABI value (`CERE_FAMILY_*`).
    pub fn ffi(self) -> u32 {
        match self {
            Family::Qwen4Exp => ffi::FAMILY_QWEN4_EXP,
            Family::Sdar => ffi::FAMILY_SDAR,
            Family::Inkling2Loop => ffi::FAMILY_INKLING2,
        }
    }

    /// serde `skip_serializing_if`: the default family is not serialized, so configs keep their `config_hash`.
    pub fn is_qwen4_exp(&self) -> bool {
        *self == Family::Qwen4Exp
    }

    /// SelfTune ST4: whether a library of this family is built with `lever`'s impl `id`. A family-scoped lever
    /// (SdarOptIn) has every impl on its own family and only its default elsewhere; for a shared lever a
    /// `qwen4_exp` library has every registered impl and an `sdar` library each lever's default only.
    pub fn admits(self, lever: &LeverSpec, id: &str) -> bool {
        id == lever.default || self == lever.family.unwrap_or(Family::Qwen4Exp)
    }
}

/// Every model family, in ABI order (the registry fixture's `families`).
pub const FAMILIES: &[Family] = &[Family::Qwen4Exp, Family::Sdar, Family::Inkling2Loop];

/// The shapes an implementation runs (SelfTune D3): at most `max_rows` decode rows (the profile's
/// `decode_batch_max`) and `max_context` tokens (`native.max_context_tokens`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Domain {
    pub max_rows: u64,
    pub max_context: u64,
}

/// The executor's model entry (`executor/cuda/src/host/support.cpp` `a4_support`): 16 rows (GemmPlan::kMaxRows rows
/// per call) and 262,144 tokens. Every registered implementation runs that whole domain today.
pub const A4_DOMAIN: Domain = Domain {
    max_rows: 16,
    max_context: 262_144,
};

/// Where an implementation's device memory comes from (SelfTune D3, ST-R10). `arena`: inside the workspace the model
/// reserves at create for every hot implementation, so a hot swap allocates nothing; `create`: allocated at create
/// for this selection (e.g. the FP8 weight copies), which only a restart-class lever may need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Workspace {
    Arena,
    Create,
}

/// What a deployment must provide for an implementation to run (checked by [`LeverSet::from_table`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Requires {
    /// `native.speculative_draft_tokens` must equal this value: the implementation acts on drafts.
    SpeculativeDraftTokens(u32),
}

/// A typed parameter's kind (SelfTune D3, ST-R21): `launch` is a plan value inside an inclusive range (hot, a plan
/// rebuild); `build` is a compiled template argument, one of at most 4 instantiations, so a hot swap refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParamKind {
    Launch { min: u64, max: u64 },
    Build { values: &'static [u64] },
}

/// One typed parameter of an implementation. `source` anchors the value by symbol (`<repo path>#<symbol>`); for a
/// build parameter the Python registry check reads the compiled value there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ParamSpec {
    pub name: &'static str,
    pub unit: &'static str,
    pub source: &'static str,
    #[serde(flatten)]
    pub kind: ParamKind,
}

/// One implementation of a lever.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImplSpec {
    pub id: &'static str,
    pub numerics: Numerics,
    /// The test that holds the implementation to its numerics: `<repo path>::<Class>::<test>` or
    /// `<repo path>::<test>`.
    pub contract: &'static str,
    /// The Vulkan executor's evidence (vulkan-amd design note "P3 result"): `<repo path>::<test>` of the ktests and
    /// library tests under `executor/vulkan/` that exercise the implementation on that library. Empty (not
    /// serialized) unless `arch` has `Vulkan`; an implementation on `Vulkan` with none is listed in
    /// [`VULKAN_UNPROVEN`]. `contract` stays the CUDA test; the arch list is where each was proven.
    #[serde(skip_serializing_if = "no_contracts")]
    pub vulkan_contracts: &'static [&'static str],
    /// A `changing` implementation is a candidate of a batched D9 default flip.
    pub d9_batch: bool,
    /// Non-empty only for a shape-dependent "auto" implementation: the implementations it picks between.
    pub picks: &'static [&'static str],
    /// The architectures its contract was proven on.
    pub arch: &'static [Arch],
    pub domain: Domain,
    pub workspace: Workspace,
    pub requires: &'static [Requires],
    pub params: &'static [ParamSpec],
    /// SelfTune E4: the numerics classes ([`CLASSES`]) it is proven in; the tuner moves only between implementations
    /// proven in the profile's class.
    pub proven_in: &'static [&'static str],
}

/// One lever: its D2 slot, its implementations and the default (today's library behaviour).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LeverSpec {
    pub name: &'static str,
    /// The inventory slot (selftune-inventory design note section 1) whose implementation the lever selects.
    pub slot: &'static str,
    pub layer: Layer,
    pub class: LeverClass,
    pub default: &'static str,
    /// SdarOptIn: the one family whose library has the lever's non-default impls (a family-scoped slot past
    /// `CERE_SLOT_COUNT`); `None` (not serialized) for a shared lever.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<Family>,
    pub impls: &'static [ImplSpec],
}

/// The inventory slots (selftune-inventory design note section 1) a lever may name.
pub const SLOTS: &[&str] = &[
    "gemm",
    "quant",
    "sample",
    "ffn.moe",
    "infra",
    "norm",
    "attention.linear_gdn",
    "attention.full",
    "attention.sparse_qsa",
    "mix.hc",
    "embed.ple",
    "kv.store",
    "rope",
    "ffn.dense",
    "spec.accept",
    "embed.token",
    "lm_head",
    "vision",
];

/// One lever selection that makes a numerics class differ from the library defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Selection {
    pub lever: &'static str,
    #[serde(rename = "impl")]
    pub id: &'static str,
}

/// SelfTune E4: a numerics class, named by a reference profile. Its `changing` selections are the reference
/// profile's `changing` implementations; every other lever's selection is bit-identical to its default, so two lever
/// sets with the same `changing` selections produce the same output and share the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NumericsClass {
    pub name: &'static str,
    pub reference: &'static str,
    pub changing: &'static [Selection],
}

/// The registered numerics classes: the library defaults, and production `-v58` (O279 D-b, D-b2).
pub static CLASSES: &[NumericsClass] = &[
    NumericsClass {
        name: "default",
        reference: "configs/runtime/native-g0.toml",
        changing: &[],
    },
    NumericsClass {
        name: "-v58",
        reference: "configs/runtime/native-g0-rows16-v58.toml",
        changing: &[
            Selection {
                lever: "k2_gemm",
                id: "small_m",
            },
            Selection {
                lever: "grammar_drafts",
                id: "on",
            },
        ],
    },
];

/// The levers-off identity rows (one test file, one row per default kernel).
const K2_DEFAULT_CONTRACT: &str =
    "executor/tests/gpu/test_levers_off_identity.py::LeversOffIdentity::test_k2_default_equals_main";
const PREFILL_DEFAULT_CONTRACT: &str =
    "executor/tests/gpu/test_levers_off_identity.py::LeversOffIdentity::test_prefill_default_equals_main";
const SPEC_LOOKUP_CONTRACT: &str =
    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_spec_lookup_greedy_bit_identical";
const DRAFT_TOKENS_CONTRACT: &str =
    "executor/tests/host/test_state.cpp::test_draft_tokens_lever_greedy_equals_plain";
const SPEC_MTP_CONTRACT: &str =
    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_spec_mtp_greedy_bit_identical";
/// DraftAccept: deep verify chains (MTP drafts plus a lookup tail, up to 8) commit plain decoding's tokens.
const SPEC_TAIL_CONTRACT: &str =
    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_spec_deep_chain_graph";
/// TreeSpec: long lookup chains (up to 8 lookup drafts, with or without the tail; Chains: up to the model's max chain
/// drafts, 15) commit plain decoding's tokens.
const SPEC_LONG_CONTRACT: &str =
    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_spec_long_lookup_greedy_bit_identical";

/// Proven only in the library defaults' class.
const DEFAULT_CLASS: &[&str] = &["default"];
/// Proven in both registered classes: the default contract, and the same test run under `-v58`'s changing
/// selections (`K8SpeculationSmallM` reruns every `K8Speculation` test with k2_gemm small_m; `grammar_drafts` "on"
/// changes only constrained sampled tickets, which `test_spec_masked_sampled_d6_exact` covers in the same class).
const BOTH_CLASSES: &[&str] = &["default", "-v58"];
/// Proven only in `-v58`: an implementation that is one of the class's changing selections.
const V58_CLASS: &[&str] = &["-v58"];

const CUDA: &[Arch] = &[Arch::Sm120a];
/// The AMD HIP port serves the defaults and the k7 MoE build (`configs/runtime/native-amd-s3*.toml`); the Vulkan
/// executor (vulkan-amd-p3-design design note 5.4) the same eager set: `cublaslt` is the device's K2 routing, no
/// graphs, no P4, per-row verify, BF16 weights, the MTP drafter at fixed policy.
const CUDA_HIP_AND_VULKAN: &[Arch] = &[Arch::Sm120a, Arch::Gfx1151, Arch::Vulkan];
/// `prefill_attention = "p4"`: the CUDA tensor-core path and the Vulkan FP16 cooperative-matrix path (needs
/// `VK_KHR_cooperative_matrix` and a seal-time probe; the device refuses otherwise). The HIP port has no P4.
const CUDA_AND_VULKAN: &[Arch] = &[Arch::Sm120a, Arch::Vulkan];
/// `k2_gemm = "tail_split"`: the Vulkan K2 GEMM's SPLIT x rows (vulkan-amd design note P4-B, D7); no other device
/// has that route.
const VULKAN: &[Arch] = &[Arch::Vulkan];

/// The implementations that act on drafts (m29l design note D6: the drafts ride the MTP verify chain).
const NEEDS_DRAFTS: &[Requires] = &[Requires::SpeculativeDraftTokens(4)];

const fn build(name: &'static str, values: &'static [u64], source: &'static str) -> ParamSpec {
    ParamSpec {
        name,
        unit: "count",
        source,
        kind: ParamKind::Build { values },
    }
}

/// The verify window a `draft_tokens` implementation caps a round at (D-6: today's runtime value as a launch param).
const fn window(min: u64, max: u64) -> ParamSpec {
    ParamSpec {
        name: "window",
        unit: "tokens",
        source: "executor/cuda/src/host/api.cpp#resolve_selection",
        kind: ParamKind::Launch { min, max },
    }
}

/// The small-M kernel's compiled shape (selftune-inventory design note section 13).
const SMALL_M_PARAMS: &[ParamSpec] = &[
    build(
        "k_step",
        &[32],
        "executor/cuda/src/kernels/k2_small_m.cu#kStepK",
    ),
    build(
        "group",
        &[16],
        "executor/cuda/src/kernels/k2_small_m.cu#kGroup",
    ),
    build(
        "max_cluster",
        &[8],
        "executor/cuda/src/kernels/k2_small_m.cu#kMaxCluster",
    ),
    build(
        "max_m",
        &[16],
        "executor/cuda/src/kernels/k2_small_m.h#kMaxM",
    ),
    build(
        "max_segments",
        &[4],
        "executor/cuda/src/kernels/k2_small_m.h#kMaxSegments",
    ),
];
const FP8_PARAMS: &[ParamSpec] = &[
    build("k_step", &[64], "executor/cuda/src/kernels/k2_fp8.h#kStepK"),
    build("max_m", &[16], "executor/cuda/src/kernels/k2_fp8.h#kMaxM"),
    build(
        "max_segments",
        &[4],
        "executor/cuda/src/kernels/k2_fp8.h#kMaxSegments",
    ),
];
const LOOKUP_PARAMS: &[ParamSpec] = &[
    build(
        "min_match",
        &[3],
        "executor/cuda/src/host/spec_lookup.h#kMinMatch",
    ),
    build(
        "max_match",
        &[32],
        "executor/cuda/src/host/spec_lookup.h#kMaxMatch",
    ),
    build("ways", &[4], "executor/cuda/src/host/spec_lookup.h#kWays"),
];
const TAIL_PARAMS: &[ParamSpec] = &[
    build(
        "min_match",
        &[3],
        "executor/cuda/src/host/spec_lookup.h#kMinMatch",
    ),
    build(
        "max_match",
        &[32],
        "executor/cuda/src/host/spec_lookup.h#kMaxMatch",
    ),
    build("ways", &[4], "executor/cuda/src/host/spec_lookup.h#kWays"),
    build(
        "max_tail_drafts",
        &[8],
        "executor/cuda/src/host/spec_lookup.h#kMaxTailDrafts",
    ),
];
const CHAIN_PARAMS: &[ParamSpec] = &[
    build(
        "min_match",
        &[3],
        "executor/cuda/src/host/spec_lookup.h#kMinMatch",
    ),
    build(
        "max_match",
        &[32],
        "executor/cuda/src/host/spec_lookup.h#kMaxMatch",
    ),
    build("ways", &[4], "executor/cuda/src/host/spec_lookup.h#kWays"),
    build(
        "max_chain_drafts",
        &[15],
        "executor/cuda/src/model/layout.h#kMaxChainDrafts",
    ),
];

fn no_contracts(c: &&[&str]) -> bool {
    c.is_empty()
}

/// Implementations available on `Vulkan` that have no Vulkan test evidence yet (reported, not hidden). Empty since
/// P4-F: `grammar_drafts` "off" (the last one) is backed by the vocab-mask rows of
/// `executor/vulkan/tests/coverage_s3.cpp` (an internal benchmark report).
pub const VULKAN_UNPROVEN: &[(&str, &str)] = &[];

pub(crate) const fn bit_identical(id: &'static str, contract: &'static str) -> ImplSpec {
    ImplSpec {
        id,
        numerics: Numerics::BitIdentical,
        contract,
        vulkan_contracts: &[],
        d9_batch: false,
        picks: &[],
        arch: CUDA,
        domain: A4_DOMAIN,
        workspace: Workspace::Arena,
        requires: &[],
        params: &[],
        proven_in: DEFAULT_CLASS,
    }
}

pub(crate) const fn changing(id: &'static str, contract: &'static str, d9_batch: bool) -> ImplSpec {
    ImplSpec {
        id,
        numerics: Numerics::Changing,
        contract,
        vulkan_contracts: &[],
        d9_batch,
        picks: &[],
        arch: CUDA,
        domain: A4_DOMAIN,
        workspace: Workspace::Arena,
        requires: &[],
        params: &[],
        proven_in: &[],
    }
}

/// PI-01: every lever, in fixture order. The defaults are the library's behaviour before levers, so a profile
/// without `[deployments.native.levers]` runs bit-identically to today's default library.
pub static REGISTRY: &[LeverSpec] = &[
    LeverSpec {
        name: "k2_gemm",
        slot: "gemm",
        layer: Layer::Executor,
        class: LeverClass::Hot,
        default: "cublaslt",
        family: None,
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/ktest/k2_test.cpp::main",
                    "executor/vulkan/ktest/k2w_test.cpp::main",
                    "executor/vulkan/ktest/k2mx_test.cpp::main",
                    "executor/vulkan/ktest/k2v_test.cpp::main",
                    "executor/vulkan/tests/forward_s3.cpp::main",
                ],
                ..bit_identical("cublaslt", K2_DEFAULT_CONTRACT)
            },
            ImplSpec {
                params: SMALL_M_PARAMS,
                proven_in: V58_CLASS,
                ..changing(
                    "small_m",
                    "executor/tests/gpu/test_kernels_k2_gemm.py::K2Gemm::test_p11_small_m_oracle",
                    true,
                )
            },
            // K2Prefill P3 (k2-prefill design note): small_m plus MXFP8 W8A8 at m > 16 on MXFP8 weights (opt-in -v56).
            ImplSpec {
                params: SMALL_M_PARAMS,
                ..changing(
                    "small_m_a8",
                    "executor/tests/gpu/test_kernels_k2m_a8.py::K2mA8::test_a8_oracle",
                    true,
                )
            },
            // VulkanAmd P4-B (vulkan-amd design note D7): the Vulkan GEMM (m > 16) carries a BF16 x row that is
            // inexact as FP16 only through <= 64 tiny tail values on the matrix unit plus an exact FP32 fma chain over
            // the tail, instead of the whole row on the FP32 route (within the K2 contract bounds). Gated by the
            // reduced AMD screen (D8, PASS 2026-11-21), it is selected by the default Vulkan profiles `-v62` / `-v63`
            // (the CUDA quality suite skips it: quality_measure.lever_sets).
            ImplSpec {
                arch: VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/ktest/k2w_test.cpp::main",
                    "executor/vulkan/ktest/k2_gemm_sites_test.cpp::main",
                    "executor/vulkan/tests/forward_s3.cpp::main",
                    "executor/vulkan/tests/coverage_s3.cpp::main",
                ],
                ..changing("tail_split", "executor/vulkan/ktest/k2w_test.cpp::main", true)
            },
        ],
    },
    LeverSpec {
        name: "prefill_attention",
        slot: "attention.full",
        layer: Layer::Executor,
        class: LeverClass::Hot,
        default: "cuda_core",
        family: None,
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/ktest/k6_test.cpp::main",
                    "executor/vulkan/ktest/k9_test.cpp::main",
                    "executor/vulkan/tests/forward_s3.cpp::main",
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical("cuda_core", PREFILL_DEFAULT_CONTRACT)
            },
            // The FP16 cooperative-matrix prefill attention (calls of 17+ rows). On Vulkan it is gated by the reduced
            // AMD screen (vulkan-amd design note D8, `p4` PASS 2026-11-21) and selected by the default Vulkan
            // profiles `-v62` / `-v63`; the registry default stays `cuda_core`.
            ImplSpec {
                arch: CUDA_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/ktest/k6_test.cpp::main",
                    "executor/vulkan/ktest/k9_test.cpp::main",
                    "executor/vulkan/tests/forward_s3.cpp::main",
                    "executor/vulkan/tests/spec_s3.cpp::main",
                ],
                params: &[build(
                    "min_rows",
                    &[17],
                    "executor/cuda/src/kernels/p4_attention.h#kMinRows",
                )],
                ..changing(
                    "p4",
                    "executor/tests/gpu/test_kernels_k6_attn.py::K6AttentionBlock::test_p4_extend",
                    true,
                )
            },
        ],
    },
    LeverSpec {
        name: "decode_step",
        slot: "infra",
        layer: Layer::Executor,
        class: LeverClass::Hot,
        default: "eager",
        family: None,
        impls: &[
            // K8SpeculationSmallM's tests decode eagerly under small_m.
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/tests/forward_s3.cpp::main",
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical(
                    "eager",
                    "executor/tests/gpu/test_model_k8_forward.py::K8Forward::test_decode_invariance_equals_merge_base",
                )
            },
            // P1 (m23 design note P1, M23P1-01..08): the graph replays the eager step's launches. Its contract
            // runs at the default k2_gemm only.
            ImplSpec {
                params: &[build(
                    "max_rows",
                    &[16],
                    "executor/cuda/src/kernels/k2_gemm.h#kMaxRows",
                )],
                ..bit_identical(
                    "graph",
                    "executor/tests/gpu/test_graph_decode.py::GraphRowsK6::test_rows_1",
                )
            },
            // M29S2-18b (m29 design note): `graph` plus the speculation cycle replayed from per-shape graphs.
            ImplSpec {
                params: &[
                    build(
                        "max_rows",
                        &[16],
                        "executor/cuda/src/model/layout.h#kDraftGraphRows",
                    ),
                    build(
                        "dense_context",
                        &[2048],
                        "executor/cuda/src/model/layout.h#kDenseContext",
                    ),
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical(
                    "spec_graph",
                    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_spec_graph_bit_identical",
                )
            },
        ],
    },
    LeverSpec {
        name: "moe_gemm",
        slot: "ffn.moe",
        layer: Layer::Executor,
        // Each MoE backend is another library build (CERE_MOE_BACKEND).
        class: LeverClass::Restart,
        default: "p3b",
        family: None,
        impls: &[
            ImplSpec {
                proven_in: BOTH_CLASSES,
                ..bit_identical(
                    "p3b",
                    "executor/tests/gpu/test_p3b_moe.py::P3bMoe::test_block_within_oracle",
                )
            },
            // The reference build, never shipped on CUDA: restart-class, so no D9 batch. The AMD HIP library is a k7
            // build.
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/ktest/k7_test.cpp::main",
                    "executor/vulkan/tests/forward_s3.cpp::main",
                ],
                ..changing(
                    "k7",
                    "executor/tests/gpu/test_kernels_k7_moe.py::MoeTest::test_block_layer0_parity_inputs",
                    false,
                )
            },
        ],
    },
    LeverSpec {
        name: "draft_tokens",
        slot: "spec.accept",
        layer: Layer::Executor,
        class: LeverClass::Hot,
        default: "max",
        family: None,
        // Greedy speculation equals plain decoding at every verify window (the host contract, device-independent),
        // and K8SpeculationSmallM verifies every chain length m in 1..4 bit-identically under small_m. "max" verifies
        // the model's speculative_draft_tokens (0 or 4, config.rs).
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/tests/spec_s3.cpp::main",
                ],
                params: &[window(4, 4)],
                proven_in: BOTH_CLASSES,
                ..bit_identical("max", DRAFT_TOKENS_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: &[window(3, 3)],
                proven_in: BOTH_CLASSES,
                ..bit_identical("3", DRAFT_TOKENS_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: &[window(2, 2)],
                proven_in: BOTH_CLASSES,
                ..bit_identical("2", DRAFT_TOKENS_CONTRACT)
            },
        ],
    },
    // The spec graph (m23 design note): per full-attention layer, a chain's m = drafts + 1 verify rows run as m
    // one-token attention cores ("per_row") or as one K6 prologue and one `k6::dense_attention_rows` launch
    // ("batched"; a chain wholly past 2,048 tokens: K9 selection and sparse attention over its rows,
    // long-ctx design note LK-01), which computes every row exactly as its one-token call.
    LeverSpec {
        name: "verify_attention",
        slot: "attention.full",
        layer: Layer::Executor,
        class: LeverClass::Hot,
        default: "per_row",
        family: None,
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/tests/spec_s3.cpp::main",
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical("per_row", SPEC_MTP_CONTRACT)
            },
            ImplSpec {
                proven_in: BOTH_CLASSES,
                ..bit_identical(
                    "batched",
                    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_verify_attention_batched_bit_identical",
                )
            },
        ],
    },
    // Fp8W: the dense K2 weights (attention qkv/o/indexer, GDN in/out projections, shared expert, lm_head) as the
    // checkpoint's BF16 ("bf16") or quantized in place at INSTANTIATE to E4M3 with FP32 per-row scales ("fp8").
    // Restart-class: the conversion happens once at create, so the hot swap refuses changing it.
    LeverSpec {
        name: "k2_weights",
        slot: "gemm",
        layer: Layer::Executor,
        class: LeverClass::Restart,
        default: "bf16",
        family: None,
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/ktest/k2v_test.cpp::main",
                    "executor/vulkan/ktest/k2w_test.cpp::main",
                    "executor/vulkan/tests/forward_s3.cpp::main",
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical("bf16", K2_DEFAULT_CONTRACT)
            },
            // The FP8 copies are allocated at create (model.cpp `instantiate_model`).
            ImplSpec {
                workspace: Workspace::Create,
                params: FP8_PARAMS,
                ..changing(
                    "fp8",
                    "executor/tests/gpu/test_kernels_k2_fp8.py::K2Fp8::test_fp8_oracle",
                    true,
                )
            },
        ],
    },
    LeverSpec {
        name: "drafter",
        slot: "spec.accept",
        layer: Layer::Runtime,
        class: LeverClass::Hot,
        default: "mtp",
        family: None,
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/tests/spec_s3.cpp::main",
                    "executor/vulkan/ktest/mtp_kernels_test.cpp::main",
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical("mtp", SPEC_MTP_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: LOOKUP_PARAMS,
                proven_in: BOTH_CLASSES,
                ..bit_identical("lookup", SPEC_LOOKUP_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: LOOKUP_PARAMS,
                proven_in: BOTH_CLASSES,
                ..bit_identical("mtp+lookup", SPEC_LOOKUP_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: TAIL_PARAMS,
                proven_in: BOTH_CLASSES,
                ..bit_identical("mtp+lookup+tail", SPEC_TAIL_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: TAIL_PARAMS,
                proven_in: BOTH_CLASSES,
                ..bit_identical("mtp+lookup+long", SPEC_LONG_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: CHAIN_PARAMS,
                proven_in: BOTH_CLASSES,
                ..bit_identical("mtp+lookup+chain", SPEC_LONG_CONTRACT)
            },
        ],
    },
    // m29a design note A1/A2 (build/adapt-draft): the per-request draft policy. "fixed" verifies the drafter's full
    // window (M2.9L's policy with the model's measured round costs when a lookup bit is set); "adaptive"
    // (`CERE_WORK_ADAPTIVE_DRAFTS`) lets the session's policy pick the window from a fixed calibrated cost table.
    // Greedy speculation commits plain decoding's tokens whatever the window; a sampled ticket's seeded draws move
    // with it, so "adaptive" reaches greedy tickets only and sampled ones keep "fixed" (A6, [REVISED: review-wave-fg
    // M1]: `native::serve::draft_flags`). Both are bit-identical for every served ticket.
    LeverSpec {
        name: "draft_policy",
        slot: "spec.accept",
        layer: Layer::Runtime,
        class: LeverClass::Hot,
        default: "fixed",
        family: None,
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &[
                    "executor/vulkan/tests/spec_s3.cpp::main",
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical("fixed", SPEC_MTP_CONTRACT)
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                params: &[
                    build(
                        "max_drafts",
                        &[3],
                        "executor/cuda/src/host/spec_lookup.h#kMaxDrafts",
                    ),
                    build(
                        "max_window",
                        &[16],
                        "executor/cuda/src/host/spec_lookup.h#kMaxWindow",
                    ),
                ],
                proven_in: BOTH_CLASSES,
                ..bit_identical(
                    "adaptive",
                    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_spec_adaptive_greedy_bit_identical",
                )
            },
        ],
    },
    // grammar-spec design note GS-04 (build/grammar-spec): constrained requests (json_schema, regex, ebnf,
    // tool_calls_constrained) and speculation. "off" serves them by plain masked decoding (`CERE_WORK_NO_SPECULATION`);
    // "on" lets them speculate with grammar-checked drafts (`cere_executor_round_drafts` / `_constrain_drafts`, the
    // grammar process's `grammar_drafts`). Greedy constrained output equals "off"'s, with the prefix cache on too: a
    // drafting session whose PREFILL carried a mask keeps the non-drafting prefix-cache layout (build/gd-fix,
    // `test_state.cpp` `test_grammar_spec_prefix_layout`). A sampled constrained ticket's seeded draws move with the
    // rounds (distribution-equal, M2.9 R5), so "on" is changing: opt-in per profile; an A/B arm may only set "off". A
    // request that is not constrained is unchanged either way, so the batched D9 (whose prompts are unconstrained) has
    // nothing to measure: not d9_batch, which restart-class allows. Restart-class: "on" needs a library with the two
    // symbols and a grammar process with feature `grammar_drafts`, both checked at load.
    LeverSpec {
        name: "grammar_drafts",
        slot: "sample",
        layer: Layer::Runtime,
        class: LeverClass::Restart,
        default: "off",
        family: None,
        impls: &[
            ImplSpec {
                arch: CUDA_HIP_AND_VULKAN,
                vulkan_contracts: &["executor/vulkan/tests/coverage_s3.cpp::main"],
                ..bit_identical(
                    "off",
                    "crates/cere-runtime/tests/server_native.rs::grammar_drafts_off_serves_constrained_requests_plain",
                )
            },
            ImplSpec {
                requires: NEEDS_DRAFTS,
                proven_in: V58_CLASS,
                ..changing(
                    "on",
                    "executor/tests/gpu/test_model_k8_forward.py::K8Speculation::test_spec_masked_sampled_d6_exact",
                    false,
                )
            },
        ],
    },
    // SdarOptIn (O320/O325): the sdar library's dense GEMMs (`executor/cuda/src/kernels/sdar/sdar_gemm.cpp`).
    // "identical" is today's cuBLASLt pin table, bit-identical to the merge base; "changing" takes split-K cuBLASLt
    // configs, which change output bits: opt-in per profile, never a tuner candidate, outside the batched D9
    // (restart class allows it). Family-scoped (slot `CERE_SLOT_SDAR_GEMM`): only an sdar library has the slot; on
    // qwen4_exp the lever stays at its default and is never sent. Restart-class: fixed at create.
    LeverSpec {
        name: "sdar_gemm",
        slot: "gemm",
        layer: Layer::Executor,
        class: LeverClass::Restart,
        default: "identical",
        family: Some(Family::Sdar),
        impls: &[
            ImplSpec {
                proven_in: BOTH_CLASSES,
                ..bit_identical(
                    "identical",
                    "executor/tests/gpu/test_sdar_kernels_pin.py::SdarGemmPin::test_pins_are_bit_identical",
                )
            },
            ImplSpec {
                workspace: Workspace::Create,
                ..changing(
                    "changing",
                    "executor/tests/gpu/test_sdar_kernels_pin.py::SdarGemmChanging::test_changing_within_bound",
                    false,
                )
            },
        ],
    },
];

/// The registry's contract rules (PI-01, SelfTune D3), run by a test on [`REGISTRY`]: lever names and impl ids unique
/// and non-empty; the slot an inventory slot; the default registered, bit-identical, needing nothing of the
/// deployment; every impl names a contract test and at least one arch; a `changing` impl is in a D9 batch or
/// restart-class, and never the default; an "auto" impl picks only registered bit-identical impls; a hot lever's
/// impls run in the create-time arena (a larger workspace is restart class); a launch param's range is ordered and a
/// build param lists 1..=4 instantiations; param names are unique per impl.
pub fn check_registry(registry: &[LeverSpec]) -> Result<(), String> {
    let mut names = Vec::new();
    for lever in registry {
        let name = lever.name;
        if name.is_empty() || names.contains(&name) {
            return Err(format!("lever name {name:?} is empty or not unique"));
        }
        names.push(name);
        if !SLOTS.contains(&lever.slot) {
            return Err(format!(
                "{name}: slot {:?} is not an inventory slot",
                lever.slot
            ));
        }
        let mut ids = Vec::new();
        for i in lever.impls {
            if i.id.is_empty() || ids.contains(&i.id) {
                return Err(format!("{name}: impl id {:?} is empty or not unique", i.id));
            }
            ids.push(i.id);
        }
        let find = |id: &str| lever.impls.iter().find(|i| i.id == id);
        match find(lever.default) {
            None => {
                return Err(format!(
                    "{name}: default {:?} is not a registered impl",
                    lever.default
                ))
            }
            Some(d) if d.numerics != Numerics::BitIdentical => {
                return Err(format!(
                    "{name}: default {:?} is not bit_identical",
                    lever.default
                ))
            }
            Some(d) if !d.requires.is_empty() => {
                return Err(format!(
                    "{name}: default {:?} must run on every deployment (requires nothing)",
                    lever.default
                ))
            }
            Some(_) => {}
        }
        for i in lever.impls {
            if i.contract.is_empty() {
                return Err(format!("{name}.{}: no contract test", i.id));
            }
            if i.arch.is_empty() {
                return Err(format!("{name}.{}: proven on no arch", i.id));
            }
            if i.numerics == Numerics::Changing && !i.d9_batch && lever.class != LeverClass::Restart
            {
                return Err(format!(
                    "{name}.{}: a changing impl needs d9_batch or a restart-class lever",
                    i.id
                ));
            }
            if lever.class == LeverClass::Hot && i.workspace != Workspace::Arena {
                return Err(format!(
                    "{name}.{}: a hot impl must run in the create-time arena; a larger workspace is restart class",
                    i.id
                ));
            }
            for pick in i.picks {
                if !find(pick).is_some_and(|p| p.numerics == Numerics::BitIdentical) {
                    return Err(format!(
                        "{name}.{}: picks {pick:?}, which is not a registered bit_identical impl",
                        i.id
                    ));
                }
            }
            let mut params = Vec::new();
            for p in i.params {
                if p.name.is_empty() || params.contains(&p.name) {
                    return Err(format!(
                        "{name}.{}: param {:?} is empty or not unique",
                        i.id, p.name
                    ));
                }
                params.push(p.name);
                match p.kind {
                    ParamKind::Launch { min, max } if min > max => {
                        return Err(format!(
                            "{name}.{}.{}: launch range {min}..={max} is empty",
                            i.id, p.name
                        ))
                    }
                    ParamKind::Build { values } if values.is_empty() || values.len() > 4 => {
                        return Err(format!(
                            "{name}.{}.{}: a build param lists 1..=4 instantiations",
                            i.id, p.name
                        ))
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// SelfTune E4's rules over `classes` (run by a test on [`CLASSES`]): class names unique; a class's changing
/// selections name registered `changing` impls, at most one per lever; every impl's `proven_in` names registered
/// classes; a `changing` impl is proven only in a class that selects it; and each class's own selection of every
/// lever (its changing one, else the default) is proven in it.
pub fn check_classes(registry: &[LeverSpec], classes: &[NumericsClass]) -> Result<(), String> {
    let mut names = Vec::new();
    for c in classes {
        if c.name.is_empty() || names.contains(&c.name) {
            return Err(format!("class name {:?} is empty or not unique", c.name));
        }
        names.push(c.name);
        let mut levers = Vec::new();
        for s in c.changing {
            let changing = registry
                .iter()
                .find(|l| l.name == s.lever)
                .and_then(|l| l.impls.iter().find(|i| i.id == s.id))
                .is_some_and(|i| i.numerics == Numerics::Changing);
            if !changing || levers.contains(&s.lever) {
                return Err(format!(
                    "class {}: {}.{} is not a registered changing impl, or its lever is selected twice",
                    c.name, s.lever, s.id
                ));
            }
            levers.push(s.lever);
        }
    }
    for lever in registry {
        for i in lever.impls {
            for name in i.proven_in {
                let Some(c) = classes.iter().find(|c| c.name == *name) else {
                    return Err(format!(
                        "{}.{}: proven in unknown class {name:?}",
                        lever.name, i.id
                    ));
                };
                if i.numerics == Numerics::Changing && c.selection(lever) != i.id {
                    return Err(format!(
                        "{}.{}: a changing impl is proven only in a class that selects it, not {name:?}",
                        lever.name, i.id
                    ));
                }
            }
        }
        for c in classes {
            let own = c.selection(lever);
            if !lever
                .impls
                .iter()
                .any(|i| i.id == own && i.proven_in.contains(&c.name))
            {
                return Err(format!(
                    "class {}: its own {}.{own} is not proven in it",
                    c.name, lever.name
                ));
            }
        }
    }
    Ok(())
}

impl NumericsClass {
    /// The impl id the class's reference profile selects for `lever` (its changing selection, else the default).
    pub fn selection(&self, lever: &LeverSpec) -> &'static str {
        self.changing
            .iter()
            .find(|s| s.lever == lever.name)
            .map_or(lever.default, |s| s.id)
    }
}

/// The registered numerics class whose changing selections are exactly `set`'s, if any.
pub fn class_of(set: &LeverSet) -> Option<&'static NumericsClass> {
    let changing: Vec<(&str, &str)> = REGISTRY
        .iter()
        .filter_map(|l| {
            let id = set.get(l.name)?;
            l.impls
                .iter()
                .any(|i| i.id == id && i.numerics == Numerics::Changing)
                .then_some((l.name, id))
        })
        .collect();
    CLASSES.iter().find(|c| {
        c.changing.len() == changing.len()
            && c.changing
                .iter()
                .all(|s| changing.contains(&(s.lever, s.id)))
    })
}

/// SelfTune E4: whether the tuner may select `lever` = `id` in numerics class `class`. Refused: an unknown lever,
/// impl or class, a restart-class lever (the tuner moves hot levers only), and an impl not proven in the class.
pub fn tuner_candidate(class: &str, lever_name: &str, id: &str) -> Result<(), String> {
    if !CLASSES.iter().any(|c| c.name == class) {
        return Err(format!("unknown numerics class {class:?}"));
    }
    let spec = lever(lever_name).ok_or_else(|| format!("{lever_name}: unknown lever"))?;
    let i = spec
        .impls
        .iter()
        .find(|i| i.id == id)
        .ok_or_else(|| format!("{lever_name}: unknown implementation {id:?}"))?;
    if spec.class == LeverClass::Restart {
        return Err(format!(
            "{lever_name}: a restart-class lever is not a tuner candidate"
        ));
    }
    if !i.proven_in.contains(&class) {
        return Err(format!(
            "{lever_name}.{id}: not proven in numerics class {class:?}"
        ));
    }
    Ok(())
}

/// Why [`check_hot_param`] refused a value: `Build` (a build param, 409 `lever_build_param` on the route) or `Invalid`
/// (an unknown lever, impl or param, or a launch value outside its range: 400 `invalid_lever`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRefusal {
    Invalid(String),
    Build(String),
}

impl ParamRefusal {
    pub fn message(&self) -> &str {
        match self {
            ParamRefusal::Invalid(m) | ParamRefusal::Build(m) => m,
        }
    }
}

/// SelfTune D3/D6: whether a hot swap may set `lever`.`id`'s param `param` to `value`. A build param is fixed at
/// library build, so the swap refuses it; a launch param must lie in its range (`POST /v1/cere/levers` `params`,
/// O292 D-9).
pub fn check_hot_param(
    lever_name: &str,
    id: &str,
    param: &str,
    value: u64,
) -> Result<(), ParamRefusal> {
    let invalid = |m: String| ParamRefusal::Invalid(m);
    let spec = lever(lever_name).ok_or_else(|| invalid(format!("{lever_name}: unknown lever")))?;
    let i = spec
        .impls
        .iter()
        .find(|i| i.id == id)
        .ok_or_else(|| invalid(format!("{lever_name}: unknown implementation {id:?}")))?;
    let p = i
        .params
        .iter()
        .find(|p| p.name == param)
        .ok_or_else(|| invalid(format!("{lever_name}.{id}: unknown param {param:?}")))?;
    match p.kind {
        ParamKind::Build { .. } => Err(ParamRefusal::Build(format!(
            "{lever_name}.{id}.{param}: a build param is fixed at library build; the hot swap refuses it"
        ))),
        ParamKind::Launch { min, max } if !(min..=max).contains(&value) => Err(invalid(format!(
            "{lever_name}.{id}.{param}: {value} is outside {min}..={max}"
        ))),
        ParamKind::Launch { .. } => Ok(()),
    }
}

/// What a deployment provides (SelfTune D3): the library's arch (from `device_uuid`), the profile's decode rows and
/// context, the model's drafts and the model family the library is built for (ST4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub arch: Arch,
    pub max_rows: u64,
    pub max_context: u64,
    pub speculative_draft_tokens: u32,
    pub family: Family,
}

impl Target {
    /// The arch a `device_uuid` names: `hip:<ordinal>` is the AMD HIP port's `gfx1151`, `vk:<ordinal>` the Vulkan
    /// executor's `vulkan`, anything else (a `GPU-` UUID) the CUDA library's `sm_120a`.
    pub fn arch_of(device_uuid: &str) -> Arch {
        if device_uuid.starts_with("hip:") {
            Arch::Gfx1151
        } else if device_uuid.starts_with("vk:") {
            Arch::Vulkan
        } else {
            Arch::Sm120a
        }
    }
}

/// The registry entry of `name`.
pub fn lever(name: &str) -> Option<&'static LeverSpec> {
    REGISTRY.iter().find(|l| l.name == name)
}

/// `tests/fixtures/lever_registry.json`: `{"levers": [...], "classes": [...], "families": [...]}` in registry order,
/// 2-space JSON, trailing newline. Each `families` entry (SelfTune ST3) names the impls a library of that family is
/// built with ([`Family::admits`]) and, per `drafter` impl, the `calibrate_windows` its model is created with
/// ([`LeverSet::calibrate_windows`]: `mtp` or `all`), so the tuner (E8) reads them instead of mirroring the rule.
pub fn registry_json() -> String {
    #[derive(Serialize)]
    struct FamilyEntry {
        family: Family,
        impls: BTreeMap<&'static str, Vec<&'static str>>,
        calibrate_windows: BTreeMap<&'static str, &'static str>,
    }
    #[derive(Serialize)]
    struct Fixture {
        levers: &'static [LeverSpec],
        classes: &'static [NumericsClass],
        families: Vec<FamilyEntry>,
    }
    let drafter = lever("drafter").expect("the drafter lever is registered");
    let families = FAMILIES
        .iter()
        .map(|&family| FamilyEntry {
            family,
            impls: REGISTRY
                .iter()
                .map(|l| {
                    let ids = l.impls.iter().map(|i| i.id);
                    (l.name, ids.filter(|id| family.admits(l, id)).collect())
                })
                .collect(),
            calibrate_windows: drafter
                .impls
                .iter()
                .map(|i| {
                    let set = LeverSet {
                        drafter: Drafter::parse(i.id).expect("a registered drafter impl parses"),
                        ..LeverSet::default()
                    };
                    let name = match set.calibrate_windows(family) {
                        ffi::CALIBRATE_MTP => "mtp",
                        _ => "all",
                    };
                    (i.id, name)
                })
                .collect(),
        })
        .collect();
    let mut text = serde_json::to_string_pretty(&Fixture {
        levers: REGISTRY,
        classes: CLASSES,
        families,
    })
    .expect("the registry serializes");
    text.push('\n');
    text
}

/// One typed lever: its impl ids in registry order and the enum each selects.
macro_rules! lever_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident = $id:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant),+
        }

        impl $name {
            /// The registry impl id.
            pub fn id(self) -> &'static str {
                match self {
                    $($name::$variant => $id),+
                }
            }

            fn parse(id: &str) -> Option<$name> {
                match id {
                    $($id => Some($name::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

lever_enum!(
    /// `k2_gemm` (slot `CERE_SLOT_K2_GEMM`).
    K2Gemm {
        #[default]
        Cublaslt = "cublaslt",
        SmallM = "small_m",
        /// K2Prefill P3: small_m, plus W8A8 MXFP8 prefill GEMMs (m > 16) on MXFP8 weights.
        SmallMA8 = "small_m_a8",
        /// VulkanAmd P4-B: the Vulkan GEMM's SPLIT x rows (Vulkan only).
        TailSplit = "tail_split",
    }
);
lever_enum!(
    /// `prefill_attention` (slot `CERE_SLOT_PREFILL_ATTENTION`).
    PrefillAttention {
        #[default]
        CudaCore = "cuda_core",
        P4 = "p4",
    }
);
lever_enum!(
    /// `decode_step` (slot `CERE_SLOT_DECODE_STEP`): eager launches, P1's decode CUDA graph, or that plus the
    /// speculation cycle's graphs.
    DecodeStep {
        #[default]
        Eager = "eager",
        Graph = "graph",
        SpecGraph = "spec_graph",
    }
);
lever_enum!(
    /// `moe_gemm` (slot `CERE_SLOT_MOE_GEMM`): the backend the library must have been built with.
    MoeGemm {
        #[default]
        P3b = "p3b",
        K7 = "k7",
    }
);
lever_enum!(
    /// `draft_tokens` (slot `CERE_SLOT_DRAFT_TOKENS`): the verify window cap per speculation round.
    DraftTokens {
        #[default]
        Max = "max",
        Three = "3",
        Two = "2",
    }
);
lever_enum!(
    /// `verify_attention` (slot `CERE_SLOT_VERIFY_ATTENTION`): a chain's verify rows' full-attention cores one call per
    /// row, or batched into one launch per layer.
    VerifyAttention {
        #[default]
        PerRow = "per_row",
        Batched = "batched",
    }
);
lever_enum!(
    /// `k2_weights` (slot `CERE_SLOT_K2_WEIGHTS`): the dense K2 weights' storage, fixed at model create.
    K2Weights {
        #[default]
        Bf16 = "bf16",
        Fp8 = "fp8",
    }
);
lever_enum!(
    /// `drafter` (runtime): which drafts a speculating ticket may verify (`cere_work.flags` bits 2 and 3,
    /// m29l design note D5/D6; "mtp+lookup+tail" also bit 5, `CERE_WORK_TAIL_DRAFTS`, on greedy tickets: the
    /// lookup's continuation past three agreeing MTP drafts, priced by the adaptive policy; "mtp+lookup+long" bits 5
    /// and 6, `CERE_WORK_LONG_DRAFTS`, on greedy tickets: also a long lookup proposal, up to 8 drafts, on its own,
    /// tree-spec design note TS-01; "mtp+lookup+chain" bits 5, 6 and 7, `CERE_WORK_CHAIN_DRAFTS`, on greedy
    /// tickets: those chains up to the model's max chain drafts, chains design note). The lookup values need
    /// `speculative_draft_tokens = 4`.
    Drafter {
        #[default]
        Mtp = "mtp",
        Lookup = "lookup",
        MtpLookup = "mtp+lookup",
        MtpLookupTail = "mtp+lookup+tail",
        MtpLookupLong = "mtp+lookup+long",
        MtpLookupChain = "mtp+lookup+chain",
    }
);
lever_enum!(
    /// `draft_policy` (runtime): "adaptive" adds `CERE_WORK_ADAPTIVE_DRAFTS` (`cere_work.flags` bit 4,
    /// m29a design note A2) to every ticket that may speculate. Needs `speculative_draft_tokens = 4`.
    DraftPolicy {
        #[default]
        Fixed = "fixed",
        Adaptive = "adaptive",
    }
);
lever_enum!(
    /// `grammar_drafts` (runtime): "on" drops `CERE_WORK_NO_SPECULATION` from constrained tickets (penalties,
    /// logit_bias, ignore_eos and min_tokens keep it) and constrains each round's drafts with the grammar. Needs
    /// `speculative_draft_tokens = 4` and a library with the constrained-draft symbols.
    GrammarDrafts {
        #[default]
        Off = "off",
        On = "on",
    }
);
lever_enum!(
    /// `sdar_gemm` (family-scoped slot `CERE_SLOT_SDAR_GEMM`, family sdar): the sdar library's bit-identical
    /// cuBLASLt pin table, or its split-K configs (numerics-changing, opt-in), fixed at model create.
    SdarGemm {
        #[default]
        Identical = "identical",
        Changing = "changing",
    }
);

/// The resolved selection of every registry lever (defaults filled): plain enums, `Copy`, so a request reads it
/// once without a lookup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LeverSet {
    pub k2_gemm: K2Gemm,
    pub prefill_attention: PrefillAttention,
    pub decode_step: DecodeStep,
    pub moe_gemm: MoeGemm,
    pub draft_tokens: DraftTokens,
    pub verify_attention: VerifyAttention,
    pub k2_weights: K2Weights,
    pub drafter: Drafter,
    pub draft_policy: DraftPolicy,
    pub grammar_drafts: GrammarDrafts,
    pub sdar_gemm: SdarGemm,
}

impl LeverSet {
    /// PI-02: `table` (lever name -> impl id; absent levers take their default) checked against [`REGISTRY`] and
    /// against what a model with `speculative_draft_tokens` can run. The error starts with the lever name (callers
    /// prefix their path, e.g. `deployments[0].native.levers.`).
    pub fn from_table(
        table: &BTreeMap<String, String>,
        speculative_draft_tokens: u32,
    ) -> Result<LeverSet, String> {
        let mut set = LeverSet::default();
        for (name, id) in table {
            let spec = lever(name).ok_or_else(|| format!("{name}: unknown lever"))?;
            if !spec.impls.iter().any(|i| i.id == id) {
                let known: Vec<&str> = spec.impls.iter().map(|i| i.id).collect();
                return Err(format!(
                    "{name}: unknown implementation {id:?} (registered: {})",
                    known.join(", ")
                ));
            }
            if !set.select(name, id) {
                return Err(format!("{name}: impl {id:?} has no typed value"));
            }
        }
        // The registry's `requires` (SelfTune D3): the drafts ride the MTP verify chain (m29l design note D6), and
        // the window cap needs drafts to cap.
        for (spec, i) in set.impls() {
            for r in i.requires {
                let Requires::SpeculativeDraftTokens(n) = *r;
                if speculative_draft_tokens != n {
                    return Err(format!(
                        "{}: {:?} needs speculative_draft_tokens = {n}",
                        spec.name, i.id
                    ));
                }
            }
        }
        Ok(set)
    }

    /// Every registry lever with its selected impl's registry entry, in registry order.
    fn impls(&self) -> impl Iterator<Item = (&'static LeverSpec, &'static ImplSpec)> + '_ {
        REGISTRY.iter().filter_map(|l| {
            let id = self.get(l.name)?;
            l.impls.iter().find(|i| i.id == id).map(|i| (l, i))
        })
    }

    /// SelfTune D3, fail-first at config validation: every selected impl was proven on `target`'s arch, covers its
    /// decode rows and context (the impl's shape domain), gets what it requires, and is built for `target`'s family
    /// ([`Family::admits`]): on `sdar` (ST4) every shared lever is its default. A lever scoped to another family
    /// (SdarOptIn: `sdar_gemm` on qwen4_exp) must be its default, which the library never receives (its slot is not
    /// sent, [`LeverSet::ffi`]), so no other rule applies to it (e.g. the AMD HIP port runs no sdar GEMM). An
    /// `inkling2_loop` library (Inkling S11/S13) runs its own kernels on every arch it is built for (CUDA, Vulkan) and
    /// ignores the shared slots (each its default, `set_levers` a no-op), so the arch and requirement of those Qwen
    /// impls do not apply to it. The shared impls' domain ([`A4_DOMAIN`]) is the Qwen model entry's; an
    /// `inkling2_loop` library's own forward bounds rows and context (`inkling_support`: 64 sessions, 1,048,576
    /// tokens), so the domain does not bind it either (inkling design note section 9.2, S10).
    pub fn check_target(&self, target: &Target) -> Result<(), String> {
        for (spec, i) in self.impls() {
            let (name, id) = (spec.name, i.id);
            if !target.family.admits(spec, id) {
                return Err(format!(
                    "{name}: {id:?} is not built for family {}",
                    target.family.name()
                ));
            }
            if spec.family.is_some_and(|f| f != target.family) {
                continue;
            }
            if target.family == Family::Inkling2Loop {
                continue;
            }
            if !i.arch.contains(&target.arch) {
                return Err(format!(
                    "{name}: {id:?} is not proven on {}",
                    target.arch.name()
                ));
            }
            // A shared impl on inkling2_loop: a Qwen kernel that library does not run.
            let unbound = spec.family.is_none() && target.family == Family::Inkling2Loop;
            if !unbound
                && (target.max_rows > i.domain.max_rows
                    || target.max_context > i.domain.max_context)
            {
                return Err(format!(
                    "{name}: {id:?} runs at most {} rows and {} context tokens, not {} and {}",
                    i.domain.max_rows, i.domain.max_context, target.max_rows, target.max_context
                ));
            }
            for r in i.requires {
                let Requires::SpeculativeDraftTokens(n) = *r;
                if target.speculative_draft_tokens != n {
                    return Err(format!(
                        "{name}: {id:?} needs speculative_draft_tokens = {n}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Sets lever `name` to impl `id`; false when the typed set has no such value.
    fn select(&mut self, name: &str, id: &str) -> bool {
        fn put<T>(slot: &mut T, value: Option<T>) -> bool {
            value.map(|v| *slot = v).is_some()
        }
        match name {
            "k2_gemm" => put(&mut self.k2_gemm, K2Gemm::parse(id)),
            "prefill_attention" => put(&mut self.prefill_attention, PrefillAttention::parse(id)),
            "decode_step" => put(&mut self.decode_step, DecodeStep::parse(id)),
            "moe_gemm" => put(&mut self.moe_gemm, MoeGemm::parse(id)),
            "draft_tokens" => put(&mut self.draft_tokens, DraftTokens::parse(id)),
            "verify_attention" => put(&mut self.verify_attention, VerifyAttention::parse(id)),
            "k2_weights" => put(&mut self.k2_weights, K2Weights::parse(id)),
            "drafter" => put(&mut self.drafter, Drafter::parse(id)),
            "draft_policy" => put(&mut self.draft_policy, DraftPolicy::parse(id)),
            "grammar_drafts" => put(&mut self.grammar_drafts, GrammarDrafts::parse(id)),
            "sdar_gemm" => put(&mut self.sdar_gemm, SdarGemm::parse(id)),
            _ => false,
        }
    }

    /// The impl id of lever `name`, `None` for an unknown lever.
    pub fn get(&self, name: &str) -> Option<&'static str> {
        Some(match name {
            "k2_gemm" => self.k2_gemm.id(),
            "prefill_attention" => self.prefill_attention.id(),
            "decode_step" => self.decode_step.id(),
            "moe_gemm" => self.moe_gemm.id(),
            "draft_tokens" => self.draft_tokens.id(),
            "verify_attention" => self.verify_attention.id(),
            "k2_weights" => self.k2_weights.id(),
            "drafter" => self.drafter.id(),
            "draft_policy" => self.draft_policy.id(),
            "grammar_drafts" => self.grammar_drafts.id(),
            "sdar_gemm" => self.sdar_gemm.id(),
            _ => return None,
        })
    }

    /// Every registry lever with its selected impl id (`GET /v1/cere/stats` `levers`, PI-05).
    pub fn to_table(&self) -> BTreeMap<String, String> {
        REGISTRY
            .iter()
            .filter_map(|l| {
                self.get(l.name)
                    .map(|id| (l.name.to_owned(), id.to_owned()))
            })
            .collect()
    }

    /// PI-03: the executor-layer part as the ABI v5 `cere_slot_selection` of a `family` library: every shared slot
    /// (1..=SLOT_COUNT) listed once, then the family's scoped slots (sdar: `CERE_SLOT_SDAR_GEMM`), in slot order
    /// (the form `cere_model_selection` reads back, so a round trip compares equal). A qwen4_exp block never lists a
    /// family-scoped slot (its library refuses one); [`LeverSet::check_target`] keeps such a lever at its default.
    pub fn ffi(&self, family: Family) -> ffi::SlotSelection {
        let impls = [
            (
                ffi::SLOT_K2_GEMM,
                match self.k2_gemm {
                    K2Gemm::Cublaslt => ffi::LEVER_K2_CUBLASLT,
                    K2Gemm::SmallM => ffi::LEVER_K2_SMALL_M,
                    K2Gemm::SmallMA8 => ffi::LEVER_K2_SMALL_M_A8,
                    K2Gemm::TailSplit => ffi::LEVER_K2_TAIL_SPLIT,
                },
                0,
            ),
            (
                ffi::SLOT_PREFILL_ATTENTION,
                match self.prefill_attention {
                    PrefillAttention::CudaCore => ffi::LEVER_PREFILL_CUDA_CORE,
                    PrefillAttention::P4 => ffi::LEVER_PREFILL_P4,
                },
                0,
            ),
            (
                ffi::SLOT_DECODE_STEP,
                match self.decode_step {
                    DecodeStep::Eager => ffi::LEVER_DECODE_EAGER,
                    DecodeStep::Graph => ffi::LEVER_DECODE_GRAPH,
                    DecodeStep::SpecGraph => ffi::LEVER_DECODE_SPEC_GRAPH,
                },
                0,
            ),
            (
                ffi::SLOT_MOE_GEMM,
                match self.moe_gemm {
                    MoeGemm::P3b => ffi::LEVER_MOE_P3B,
                    MoeGemm::K7 => ffi::LEVER_MOE_K7,
                },
                0,
            ),
            (
                ffi::SLOT_DRAFT_TOKENS,
                ffi::LEVER_DRAFT_WINDOW,
                match self.draft_tokens {
                    DraftTokens::Max => 0,
                    DraftTokens::Three => 3,
                    DraftTokens::Two => 2,
                },
            ),
            (
                ffi::SLOT_VERIFY_ATTENTION,
                match self.verify_attention {
                    VerifyAttention::PerRow => ffi::LEVER_VERIFY_PER_ROW,
                    VerifyAttention::Batched => ffi::LEVER_VERIFY_BATCHED,
                },
                0,
            ),
            (
                ffi::SLOT_K2_WEIGHTS,
                match self.k2_weights {
                    K2Weights::Bf16 => ffi::LEVER_K2_WEIGHTS_BF16,
                    K2Weights::Fp8 => ffi::LEVER_K2_WEIGHTS_FP8,
                },
                0,
            ),
        ];
        let mut s = ffi::SlotSelection {
            struct_size: ffi::struct_size::<ffi::SlotSelection>(),
            abi_version: ffi::ABI_VERSION,
            block_version: ffi::SLOT_BLOCK_VERSION,
            count: impls.len() as u32,
            ..Default::default()
        };
        for (choice, (slot, r#impl, window)) in s.choices.iter_mut().zip(impls) {
            choice.slot = slot;
            choice.r#impl = r#impl;
            choice.params[0] = window;
        }
        if family == Family::Sdar {
            s.choices[s.count as usize] = ffi::SlotChoice {
                slot: ffi::SLOT_SDAR_GEMM,
                r#impl: match self.sdar_gemm {
                    SdarGemm::Identical => ffi::LEVER_SDAR_GEMM_IDENTICAL,
                    SdarGemm::Changing => ffi::LEVER_SDAR_GEMM_CHANGING,
                },
                params: [0; ffi::SLOT_PARAMS],
            };
            s.count += 1;
        }
        s
    }

    /// LoadCrit LC-01: `cere_model_desc.calibrate_windows` of a `family` model, derived from `drafter` (not a slot):
    /// a drafter whose tickets never carry `CERE_WORK_TAIL_DRAFTS`/`LONG`/`CHAIN` (bits 5-7, serve.rs
    /// `draft_flags`) only ever has windows 1..4 priced, so the seal times only those (MTP); the tail, long and chain
    /// drafters keep every window (ALL). An `sdar` model has no speculation windows and its library accepts only
    /// ALL (executor `host/family.h` `sdar_refusal`; SelfTune ST4 S3d, D-n), whatever the drafter; so does an
    /// `inkling2_loop` model until its drafter lands (`inkling2_refusal`, inkling design note D-I5, S9).
    pub fn calibrate_windows(&self, family: Family) -> u32 {
        if family != Family::Qwen4Exp {
            return ffi::CALIBRATE_ALL;
        }
        match self.drafter {
            Drafter::Mtp | Drafter::Lookup | Drafter::MtpLookup => ffi::CALIBRATE_MTP,
            Drafter::MtpLookupTail | Drafter::MtpLookupLong | Drafter::MtpLookupChain => {
                ffi::CALIBRATE_ALL
            }
        }
    }

    /// PI-05: the selection the executor reports (`cere_model_selection`) with the runtime-layer levers (`drafter`,
    /// `draft_policy`, `grammar_drafts`) of `runtime`. A malformed block (block version, count, a slot outside
    /// 1..=SLOT_COUNT and not the family-scoped `SLOT_SDAR_GEMM`, or listed twice, a param the impl does not have) or
    /// a value outside the registry (e.g. `CERE_LEVER_MOE_BUILD`, which `cered` never sends) is an error; an unlisted
    /// slot is its default (impl 0; a qwen4_exp library never lists `SLOT_SDAR_GEMM`).
    pub fn from_ffi(
        selection: &ffi::SlotSelection,
        runtime: &LeverSet,
    ) -> Result<LeverSet, String> {
        let bad = |name: &str, value: u32| Err(format!("{name}: the executor reports {value}"));
        if selection.block_version != ffi::SLOT_BLOCK_VERSION {
            return bad("block_version", selection.block_version);
        }
        if selection.count as usize > ffi::SLOT_MAX {
            return bad("count", selection.count);
        }
        let mut impls = [0u32; ffi::SLOT_SDAR_GEMM as usize + 1];
        let mut window = 0;
        let mut seen = [false; ffi::SLOT_SDAR_GEMM as usize + 1];
        for c in &selection.choices[..selection.count as usize] {
            if c.slot == 0 || c.slot > ffi::SLOT_SDAR_GEMM || seen[c.slot as usize] {
                return bad("slot", c.slot);
            }
            seen[c.slot as usize] = true;
            let params = if c.slot == ffi::SLOT_DRAFT_TOKENS {
                1
            } else {
                0
            };
            if let Some(p) = c.params[params..].iter().find(|&&p| p != 0) {
                return bad("params", *p);
            }
            impls[c.slot as usize] = c.r#impl;
            if c.slot == ffi::SLOT_DRAFT_TOKENS {
                window = c.params[0];
            }
        }
        let slot = |s: u32| impls[s as usize];
        if slot(ffi::SLOT_DRAFT_TOKENS) != ffi::LEVER_DRAFT_WINDOW {
            return bad("draft_tokens impl", slot(ffi::SLOT_DRAFT_TOKENS));
        }
        Ok(LeverSet {
            k2_gemm: match slot(ffi::SLOT_K2_GEMM) {
                ffi::LEVER_K2_CUBLASLT => K2Gemm::Cublaslt,
                ffi::LEVER_K2_SMALL_M => K2Gemm::SmallM,
                ffi::LEVER_K2_SMALL_M_A8 => K2Gemm::SmallMA8,
                ffi::LEVER_K2_TAIL_SPLIT => K2Gemm::TailSplit,
                v => return bad("k2_gemm", v),
            },
            prefill_attention: match slot(ffi::SLOT_PREFILL_ATTENTION) {
                ffi::LEVER_PREFILL_CUDA_CORE => PrefillAttention::CudaCore,
                ffi::LEVER_PREFILL_P4 => PrefillAttention::P4,
                v => return bad("prefill_attention", v),
            },
            decode_step: match slot(ffi::SLOT_DECODE_STEP) {
                ffi::LEVER_DECODE_EAGER => DecodeStep::Eager,
                ffi::LEVER_DECODE_GRAPH => DecodeStep::Graph,
                ffi::LEVER_DECODE_SPEC_GRAPH => DecodeStep::SpecGraph,
                v => return bad("decode_step", v),
            },
            moe_gemm: match slot(ffi::SLOT_MOE_GEMM) {
                ffi::LEVER_MOE_P3B => MoeGemm::P3b,
                ffi::LEVER_MOE_K7 => MoeGemm::K7,
                v => return bad("moe_gemm", v),
            },
            draft_tokens: match window {
                0 => DraftTokens::Max,
                3 => DraftTokens::Three,
                2 => DraftTokens::Two,
                v => return bad("draft_tokens", v),
            },
            verify_attention: match slot(ffi::SLOT_VERIFY_ATTENTION) {
                ffi::LEVER_VERIFY_PER_ROW => VerifyAttention::PerRow,
                ffi::LEVER_VERIFY_BATCHED => VerifyAttention::Batched,
                v => return bad("verify_attention", v),
            },
            k2_weights: match slot(ffi::SLOT_K2_WEIGHTS) {
                ffi::LEVER_K2_WEIGHTS_BF16 => K2Weights::Bf16,
                ffi::LEVER_K2_WEIGHTS_FP8 => K2Weights::Fp8,
                v => return bad("k2_weights", v),
            },
            sdar_gemm: match slot(ffi::SLOT_SDAR_GEMM) {
                ffi::LEVER_SDAR_GEMM_IDENTICAL => SdarGemm::Identical,
                ffi::LEVER_SDAR_GEMM_CHANGING => SdarGemm::Changing,
                v => return bad("sdar_gemm", v),
            },
            drafter: runtime.drafter,
            draft_policy: runtime.draft_policy,
            grammar_drafts: runtime.grammar_drafts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K2: LeverSpec = LeverSpec {
        name: "k2_gemm",
        slot: "gemm",
        layer: Layer::Executor,
        class: LeverClass::Hot,
        default: "a",
        family: None,
        impls: &[
            bit_identical("a", "x.py::A::t"),
            changing("b", "x.py::B::t", true),
        ],
    };

    fn with_impls(impls: &'static [ImplSpec]) -> [LeverSpec; 1] {
        [LeverSpec { impls, ..K2 }]
    }

    #[test]
    fn the_registry_keeps_its_contract_rules() {
        assert_eq!(check_registry(REGISTRY), Ok(()));
        check_registry(&[K2]).unwrap();
    }

    #[test]
    fn an_impl_without_a_contract_is_refused() {
        static IMPLS: [ImplSpec; 2] = [bit_identical("a", "x.py::A::t"), changing("b", "", true)];
        let err = check_registry(&with_impls(&IMPLS)).unwrap_err();
        assert!(err.contains("k2_gemm.b: no contract test"), "{err}");
    }

    #[test]
    fn a_changing_default_is_refused() {
        let err = check_registry(&[LeverSpec { default: "b", ..K2 }]).unwrap_err();
        assert!(err.contains("default \"b\" is not bit_identical"), "{err}");
    }

    #[test]
    fn a_changing_impl_needs_a_d9_batch_unless_restart_class() {
        static IMPLS: [ImplSpec; 2] = [
            bit_identical("a", "x.py::A::t"),
            changing("b", "x.py::B::t", false),
        ];
        let err = check_registry(&with_impls(&IMPLS)).unwrap_err();
        assert!(err.contains("needs d9_batch"), "{err}");
        check_registry(&[LeverSpec {
            class: LeverClass::Restart,
            impls: &IMPLS,
            ..K2
        }])
        .unwrap();
    }

    #[test]
    fn an_auto_impl_picks_only_registered_bit_identical_impls() {
        const fn auto(picks: &'static [&'static str]) -> ImplSpec {
            ImplSpec {
                picks,
                ..bit_identical("auto", "x.py::Auto::t")
            }
        }
        static ONE: [ImplSpec; 3] = [K2.impls[0], K2.impls[1], auto(&["a"])];
        static CHANGING: [ImplSpec; 3] = [K2.impls[0], K2.impls[1], auto(&["a", "b"])];
        static UNKNOWN: [ImplSpec; 3] = [K2.impls[0], K2.impls[1], auto(&["c"])];
        check_registry(&with_impls(&ONE)).unwrap();
        for impls in [&CHANGING, &UNKNOWN] {
            let err = check_registry(&with_impls(impls)).unwrap_err();
            assert!(err.contains("not a registered bit_identical impl"), "{err}");
        }
    }

    #[test]
    fn an_unknown_default_or_duplicate_id_is_refused() {
        let err = check_registry(&[LeverSpec { default: "z", ..K2 }]).unwrap_err();
        assert!(
            err.contains("default \"z\" is not a registered impl"),
            "{err}"
        );
        static DUPLICATE: [ImplSpec; 2] = [
            bit_identical("a", "x.py::A::t"),
            bit_identical("a", "x.py::A::u"),
        ];
        let err = check_registry(&with_impls(&DUPLICATE)).unwrap_err();
        assert!(err.contains("not unique"), "{err}");
        let err = check_registry(&[K2, K2]).unwrap_err();
        assert!(err.contains("lever name \"k2_gemm\""), "{err}");
    }

    /// SelfTune D3 fail-first: the registry v2 rules on slot, arch, workspace, requires and typed params.
    #[test]
    fn registry_v2_refuses_a_bad_slot_arch_workspace_requires_or_param() {
        let err = check_registry(&[LeverSpec {
            slot: "gemmm",
            ..K2
        }])
        .unwrap_err();
        assert!(
            err.contains("slot \"gemmm\" is not an inventory slot"),
            "{err}"
        );
        static NO_ARCH: [ImplSpec; 2] = [
            bit_identical("a", "x.py::A::t"),
            ImplSpec {
                arch: &[],
                ..changing("b", "x.py::B::t", true)
            },
        ];
        let err = check_registry(&with_impls(&NO_ARCH)).unwrap_err();
        assert!(err.contains("k2_gemm.b: proven on no arch"), "{err}");
        static CREATE: [ImplSpec; 2] = [
            bit_identical("a", "x.py::A::t"),
            ImplSpec {
                workspace: Workspace::Create,
                ..changing("b", "x.py::B::t", true)
            },
        ];
        let err = check_registry(&with_impls(&CREATE)).unwrap_err();
        assert!(
            err.contains("a hot impl must run in the create-time arena"),
            "{err}"
        );
        check_registry(&[LeverSpec {
            class: LeverClass::Restart,
            impls: &CREATE,
            ..K2
        }])
        .unwrap();
        static NEEDY_DEFAULT: [ImplSpec; 1] = [ImplSpec {
            requires: NEEDS_DRAFTS,
            ..bit_identical("a", "x.py::A::t")
        }];
        let err = check_registry(&with_impls(&NEEDY_DEFAULT)).unwrap_err();
        assert!(err.contains("must run on every deployment"), "{err}");
        static EMPTY_RANGE: [ImplSpec; 1] = [ImplSpec {
            params: &[window(4, 3)],
            ..bit_identical("a", "x.py::A::t")
        }];
        let err = check_registry(&with_impls(&EMPTY_RANGE)).unwrap_err();
        assert!(
            err.contains("k2_gemm.a.window: launch range 4..=3 is empty"),
            "{err}"
        );
        static WIDE_BUILD: [ImplSpec; 1] = [ImplSpec {
            params: &[build("tile", &[1, 2, 3, 4, 5], "x.h#kTile")],
            ..bit_identical("a", "x.py::A::t")
        }];
        let err = check_registry(&with_impls(&WIDE_BUILD)).unwrap_err();
        assert!(
            err.contains("a build param lists 1..=4 instantiations"),
            "{err}"
        );
        static NO_BUILD: [ImplSpec; 1] = [ImplSpec {
            params: &[build("tile", &[], "x.h#kTile")],
            ..bit_identical("a", "x.py::A::t")
        }];
        let err = check_registry(&with_impls(&NO_BUILD)).unwrap_err();
        assert!(
            err.contains("a build param lists 1..=4 instantiations"),
            "{err}"
        );
        static TWICE: [ImplSpec; 1] = [ImplSpec {
            params: &[window(2, 4), window(2, 4)],
            ..bit_identical("a", "x.py::A::t")
        }];
        let err = check_registry(&with_impls(&TWICE)).unwrap_err();
        assert!(
            err.contains("param \"window\" is empty or not unique"),
            "{err}"
        );
    }

    /// SelfTune E4: the registered classes keep their rules, and each rule refuses its violation.
    #[test]
    fn numerics_classes_keep_their_rules() {
        assert_eq!(check_classes(REGISTRY, CLASSES), Ok(()));
        // A class selecting a bit-identical impl as "changing", or one lever twice.
        static BIT: [NumericsClass; 1] = [NumericsClass {
            name: "x",
            reference: "x.toml",
            changing: &[Selection {
                lever: "k2_gemm",
                id: "a",
            }],
        }];
        let err = check_classes(&[K2], &BIT).unwrap_err();
        assert!(err.contains("not a registered changing impl"), "{err}");
        static TWICE: [NumericsClass; 1] = [NumericsClass {
            name: "x",
            reference: "x.toml",
            changing: &[
                Selection {
                    lever: "k2_gemm",
                    id: "b",
                },
                Selection {
                    lever: "k2_gemm",
                    id: "b",
                },
            ],
        }];
        let err = check_classes(&[K2], &TWICE).unwrap_err();
        assert!(err.contains("selected twice"), "{err}");
        // K2's "b" is proven nowhere and "a" only in "default": a class without proof of its own selection fails.
        const DEFAULT: [NumericsClass; 1] = [NumericsClass {
            name: "default",
            reference: "x.toml",
            changing: &[],
        }];
        check_classes(&[K2], &DEFAULT).unwrap();
        static B: [NumericsClass; 2] = [
            DEFAULT[0],
            NumericsClass {
                name: "b",
                reference: "x.toml",
                changing: &[Selection {
                    lever: "k2_gemm",
                    id: "b",
                }],
            },
        ];
        let err = check_classes(&[K2], &B).unwrap_err();
        assert!(
            err.contains("class b: its own k2_gemm.b is not proven in it"),
            "{err}"
        );
        // A changing impl proven in a class that does not select it, and an unknown class.
        static WRONG: [ImplSpec; 2] = [
            bit_identical("a", "x.py::A::t"),
            ImplSpec {
                proven_in: DEFAULT_CLASS,
                ..changing("b", "x.py::B::t", true)
            },
        ];
        let err = check_classes(&with_impls(&WRONG), &DEFAULT).unwrap_err();
        assert!(
            err.contains("proven only in a class that selects it"),
            "{err}"
        );
        static UNKNOWN: [ImplSpec; 1] = [ImplSpec {
            proven_in: &["nope"],
            ..bit_identical("a", "x.py::A::t")
        }];
        let err = check_classes(&with_impls(&UNKNOWN), &DEFAULT).unwrap_err();
        assert!(err.contains("proven in unknown class \"nope\""), "{err}");
    }

    /// Production `-v58` (and `-v53`, the same changing selections without grammar drafts) resolve to their classes.
    #[test]
    fn a_lever_set_resolves_to_its_numerics_class() {
        let v58 = LeverSet {
            k2_gemm: K2Gemm::SmallM,
            decode_step: DecodeStep::SpecGraph,
            verify_attention: VerifyAttention::Batched,
            drafter: Drafter::MtpLookup,
            draft_policy: DraftPolicy::Adaptive,
            grammar_drafts: GrammarDrafts::On,
            ..LeverSet::default()
        };
        assert_eq!(class_of(&v58).map(|c| c.name), Some("-v58"));
        assert_eq!(
            class_of(&LeverSet::default()).map(|c| c.name),
            Some("default")
        );
        let v53 = LeverSet {
            grammar_drafts: GrammarDrafts::Off,
            ..v58
        };
        assert_eq!(class_of(&v53), None);
    }

    /// SelfTune E4 fail-first: the tuner moves only between hot impls proven in the profile's class.
    #[test]
    fn a_tuner_candidate_must_be_hot_and_proven_in_the_class() {
        tuner_candidate("-v58", "decode_step", "spec_graph").unwrap();
        tuner_candidate("-v58", "k2_gemm", "small_m").unwrap();
        for (class, lever, id, want) in [
            (
                "-v58",
                "decode_step",
                "graph",
                "decode_step.graph: not proven in numerics class \"-v58\"",
            ),
            (
                "-v58",
                "k2_gemm",
                "cublaslt",
                "k2_gemm.cublaslt: not proven in numerics class \"-v58\"",
            ),
            (
                "default",
                "k2_gemm",
                "small_m",
                "k2_gemm.small_m: not proven in numerics class \"default\"",
            ),
            (
                "-v58",
                "prefill_attention",
                "p4",
                "prefill_attention.p4: not proven in numerics class \"-v58\"",
            ),
            (
                "-v58",
                "moe_gemm",
                "p3b",
                "moe_gemm: a restart-class lever is not a tuner candidate",
            ),
            (
                "-v58",
                "grammar_drafts",
                "on",
                "grammar_drafts: a restart-class lever is not a tuner candidate",
            ),
            (
                "default",
                "sdar_gemm",
                "changing",
                "sdar_gemm: a restart-class lever is not a tuner candidate",
            ),
            (
                "-v99",
                "k2_gemm",
                "small_m",
                "unknown numerics class \"-v99\"",
            ),
            ("-v58", "turbo", "on", "turbo: unknown lever"),
            (
                "-v58",
                "k2_gemm",
                "fast",
                "k2_gemm: unknown implementation \"fast\"",
            ),
        ] {
            assert_eq!(tuner_candidate(class, lever, id), Err(want.to_owned()));
        }
    }

    /// SelfTune D3/D6 fail-first: a hot swap refuses a build param (`Build`, 409 on the route) and an unknown or
    /// out-of-range launch param (`Invalid`, 400).
    #[test]
    fn a_hot_param_must_be_a_launch_param_in_range() {
        check_hot_param("draft_tokens", "3", "window", 3).unwrap();
        check_hot_param("draft_tokens", "max", "window", 4).unwrap();
        assert_eq!(
            check_hot_param("k2_gemm", "small_m", "k_step", 64),
            Err(ParamRefusal::Build(
                "k2_gemm.small_m.k_step: a build param is fixed at library build; the hot swap refuses it"
                    .to_owned()
            ))
        );
        for (lever, id, param, value, want) in [
            (
                "draft_tokens",
                "3",
                "window",
                4,
                "draft_tokens.3.window: 4 is outside 3..=3",
            ),
            (
                "draft_tokens",
                "3",
                "tile",
                1,
                "draft_tokens.3: unknown param \"tile\"",
            ),
            (
                "draft_tokens",
                "5",
                "window",
                5,
                "draft_tokens: unknown implementation \"5\"",
            ),
            ("turbo", "on", "window", 1, "turbo: unknown lever"),
        ] {
            let got = check_hot_param(lever, id, param, value);
            assert_eq!(got, Err(ParamRefusal::Invalid(want.to_owned())));
            assert_eq!(got.unwrap_err().message(), want);
        }
    }

    /// SelfTune D3 fail-first: a selection outside its impl's arch, shape domain or requirements is refused.
    #[test]
    fn check_target_refuses_an_arch_domain_or_requirement_outside_the_impl() {
        let cuda = Target {
            arch: Arch::Sm120a,
            max_rows: 16,
            max_context: 262_144,
            speculative_draft_tokens: 4,
            family: Family::Qwen4Exp,
        };
        let hip = Target {
            arch: Target::arch_of("hip:0"),
            ..cuda
        };
        assert_eq!(
            Target::arch_of("GPU-aaaaaaaa-0000-4000-8000-000000000000"),
            Arch::Sm120a
        );
        LeverSet::default().check_target(&cuda).unwrap();
        let k7 = LeverSet {
            moe_gemm: MoeGemm::K7,
            ..LeverSet::default()
        };
        k7.check_target(&hip).unwrap();
        assert_eq!(
            LeverSet::default().check_target(&hip),
            Err("moe_gemm: \"p3b\" is not proven on gfx1151".to_owned())
        );
        let small_m = LeverSet {
            k2_gemm: K2Gemm::SmallM,
            ..k7
        };
        assert_eq!(
            small_m.check_target(&hip),
            Err("k2_gemm: \"small_m\" is not proven on gfx1151".to_owned())
        );
        for target in [
            Target {
                max_rows: 17,
                ..cuda
            },
            Target {
                max_context: 262_145,
                ..cuda
            },
        ] {
            let err = LeverSet::default().check_target(&target).unwrap_err();
            assert!(
                err.starts_with(
                    "k2_gemm: \"cublaslt\" runs at most 16 rows and 262144 context tokens"
                ),
                "{err}"
            );
        }
        let lookup = LeverSet {
            drafter: Drafter::Lookup,
            ..LeverSet::default()
        };
        lookup.check_target(&cuda).unwrap();
        assert_eq!(
            lookup.check_target(&Target {
                speculative_draft_tokens: 0,
                ..cuda
            }),
            Err("drafter: \"lookup\" needs speculative_draft_tokens = 4".to_owned())
        );
    }

    /// SelfTune ST4 fail-first: an `sdar` library runs every lever's default and refuses any other impl by name.
    #[test]
    fn check_target_on_family_sdar_refuses_every_impl_but_the_default() {
        let sdar = Target {
            arch: Arch::Sm120a,
            max_rows: 1,
            max_context: 4096,
            speculative_draft_tokens: 0,
            family: Family::Sdar,
        };
        LeverSet::default().check_target(&sdar).unwrap();
        // The shared levers only: a family-scoped lever of family sdar (sdar_gemm) is built in full there.
        for l in REGISTRY.iter().filter(|l| l.family.is_none()) {
            for i in l.impls.iter().filter(|i| i.id != l.default) {
                let set = LeverSet::from_table(
                    &BTreeMap::from([(l.name.to_owned(), i.id.to_owned())]),
                    4,
                )
                .unwrap();
                assert_eq!(
                    set.check_target(&sdar),
                    Err(format!(
                        "{}: {:?} is not built for family sdar",
                        l.name, i.id
                    )),
                );
            }
        }
        assert_eq!(Family::default(), Family::Qwen4Exp);
        assert_eq!(
            (Family::Qwen4Exp.ffi(), Family::Sdar.ffi()),
            (ffi::FAMILY_QWEN4_EXP, ffi::FAMILY_SDAR)
        );
        assert_eq!(
            (Family::Qwen4Exp.name(), Family::Sdar.name()),
            ("qwen4_exp", "sdar")
        );
    }

    /// Inkling 2 Mini S0 (fail-first): an `inkling2_loop` library, like an `sdar` one, runs every shared lever's
    /// default only and refuses every other impl and every family-scoped lever by name; it calibrates ALL windows.
    /// S13 AMD: the defaults pass on Vulkan too (the CUDA-only `moe_gemm = "p3b"` is a Qwen kernel the library never
    /// runs), and every other impl is still refused there.
    /// S10: the shared impls' A4 domain does not bind it (a 1M context with 64 rows passes; on qwen4_exp it fails).
    #[test]
    fn check_target_on_family_inkling2_loop_refuses_every_impl_but_the_default() {
        for arch in [Arch::Sm120a, Arch::Vulkan] {
            let inkling = Target {
                arch,
                max_rows: 1,
                max_context: 4096,
                speculative_draft_tokens: 0,
                family: Family::Inkling2Loop,
            };
            LeverSet::default().check_target(&inkling).unwrap();
            let long = Target {
                max_rows: 64,
                max_context: 1 << 20,
                ..inkling
            };
            LeverSet::default().check_target(&long).unwrap();
            let qwen = Target {
                family: Family::Qwen4Exp,
                ..long
            };
            assert!(LeverSet::default().check_target(&qwen).is_err());
            for l in REGISTRY {
                for i in l.impls.iter().filter(|i| i.id != l.default) {
                    let set = LeverSet::from_table(
                        &BTreeMap::from([(l.name.to_owned(), i.id.to_owned())]),
                        4,
                    )
                    .unwrap();
                    assert!(set.check_target(&inkling).is_err(), "{}.{}", l.name, i.id);
                }
            }
        }
        assert_eq!(Family::Inkling2Loop.ffi(), ffi::FAMILY_INKLING2);
        assert_eq!(Family::Inkling2Loop.name(), "inkling2_loop");
        assert_eq!(
            serde_json::to_value(Family::Inkling2Loop).unwrap(),
            "inkling2_loop"
        );
        assert_eq!(
            LeverSet::default().calibrate_windows(Family::Inkling2Loop),
            ffi::CALIBRATE_ALL
        );
    }

    /// Every registered impl has a typed value, and the typed set has no lever outside the registry.
    #[test]
    fn the_typed_set_covers_the_registry_exactly() {
        for l in REGISTRY {
            assert_eq!(
                LeverSet::default().get(l.name),
                Some(l.default),
                "{}",
                l.name
            );
            for i in l.impls {
                let mut set = LeverSet::default();
                assert!(set.select(l.name, i.id), "{}.{}", l.name, i.id);
                assert_eq!(set.get(l.name), Some(i.id));
            }
        }
        let names: Vec<&str> = REGISTRY.iter().map(|l| l.name).collect();
        let table = LeverSet::default().to_table();
        assert_eq!(table.len(), names.len());
        assert!(names.iter().all(|n| table.contains_key(*n)));
    }

    #[test]
    fn from_table_validates_against_the_registry_and_the_drafts() {
        let table = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        assert_eq!(
            LeverSet::from_table(&table(&[]), 0),
            Ok(LeverSet::default())
        );
        let set = LeverSet::from_table(
            &table(&[("k2_gemm", "small_m"), ("prefill_attention", "p4")]),
            0,
        )
        .unwrap();
        assert_eq!(
            (set.k2_gemm, set.prefill_attention),
            (K2Gemm::SmallM, PrefillAttention::P4)
        );
        for (pairs, want) in [
            (&[("turbo", "on")][..], "turbo: unknown lever"),
            (
                &[("k2_gemm", "fast")][..],
                "k2_gemm: unknown implementation \"fast\" (registered: cublaslt, small_m, small_m_a8, tail_split)",
            ),
            (
                &[("drafter", "mtp+lookup")][..],
                "drafter: \"mtp+lookup\" needs speculative_draft_tokens = 4",
            ),
            (
                &[("draft_tokens", "3")][..],
                "draft_tokens: \"3\" needs speculative_draft_tokens = 4",
            ),
            (
                &[("draft_policy", "adaptive")][..],
                "draft_policy: \"adaptive\" needs speculative_draft_tokens = 4",
            ),
            (
                &[("grammar_drafts", "on")][..],
                "grammar_drafts: \"on\" needs speculative_draft_tokens = 4",
            ),
        ] {
            assert_eq!(LeverSet::from_table(&table(pairs), 0), Err(want.to_owned()));
        }
        let spec = LeverSet::from_table(
            &table(&[
                ("drafter", "lookup"),
                ("draft_tokens", "2"),
                ("draft_policy", "adaptive"),
                ("grammar_drafts", "on"),
            ]),
            4,
        )
        .unwrap();
        assert_eq!(
            (
                spec.drafter,
                spec.draft_tokens,
                spec.draft_policy,
                spec.grammar_drafts
            ),
            (
                Drafter::Lookup,
                DraftTokens::Two,
                DraftPolicy::Adaptive,
                GrammarDrafts::On
            )
        );
        // Restart-class levers are settable in a config (only the hot swap refuses changing them).
        let k7 = LeverSet::from_table(&table(&[("moe_gemm", "k7")]), 0).unwrap();
        assert_eq!(k7.moe_gemm, MoeGemm::K7);
        let fp8 = LeverSet::from_table(&table(&[("k2_weights", "fp8")]), 0).unwrap();
        assert_eq!(fp8.k2_weights, K2Weights::Fp8);
        assert_eq!(
            LeverSet::from_table(&table(&[("k2_weights", "fp4")]), 0),
            Err("k2_weights: unknown implementation \"fp4\" (registered: bf16, fp8)".to_owned())
        );
        let restart: Vec<&str> = REGISTRY
            .iter()
            .filter(|l| l.class == LeverClass::Restart)
            .map(|l| l.name)
            .collect();
        assert_eq!(
            restart,
            ["moe_gemm", "k2_weights", "grammar_drafts", "sdar_gemm"]
        );
    }

    /// SelfTune ST4 S3d (D-n): a qwen4_exp model keeps the drafter's windows, an sdar model is ALL whatever the
    /// drafter (its library refuses any other value).
    #[test]
    fn calibrate_windows_follow_the_drafter_for_qwen_and_are_all_for_sdar() {
        for (drafter, qwen) in [
            (Drafter::Mtp, ffi::CALIBRATE_MTP),
            (Drafter::Lookup, ffi::CALIBRATE_MTP),
            (Drafter::MtpLookup, ffi::CALIBRATE_MTP),
            (Drafter::MtpLookupTail, ffi::CALIBRATE_ALL),
            (Drafter::MtpLookupLong, ffi::CALIBRATE_ALL),
            (Drafter::MtpLookupChain, ffi::CALIBRATE_ALL),
        ] {
            let levers = LeverSet {
                drafter,
                ..LeverSet::default()
            };
            assert_eq!(
                levers.calibrate_windows(Family::Qwen4Exp),
                qwen,
                "{drafter:?}"
            );
            assert_eq!(
                levers.calibrate_windows(Family::Sdar),
                ffi::CALIBRATE_ALL,
                "{drafter:?}"
            );
        }
    }

    /// SelfTune ST3 (E8): the fixture's `families` carry, per family, the impls its library is built with and each
    /// drafter's `calibrate_windows`, from the same rules [`LeverSet::check_target`] and
    /// [`LeverSet::calibrate_windows`] apply.
    #[test]
    fn the_fixture_families_follow_check_target_and_calibrate_windows() {
        let doc: serde_json::Value = serde_json::from_str(&registry_json()).unwrap();
        let families = doc["families"].as_array().unwrap();
        assert_eq!(families.len(), FAMILIES.len());
        for (entry, &family) in families.iter().zip(FAMILIES) {
            assert_eq!(entry["family"], family.name());
            let target = Target {
                arch: Arch::Sm120a,
                max_rows: 1,
                max_context: 4096,
                speculative_draft_tokens: 4,
                family,
            };
            for l in REGISTRY {
                let listed: Vec<&str> = entry["impls"][l.name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                for i in l.impls {
                    let mut set = LeverSet::default();
                    assert!(set.select(l.name, i.id));
                    let built = set
                        .check_target(&target)
                        .map_or_else(|e| !e.contains("is not built for family"), |()| true);
                    assert_eq!(
                        listed.contains(&i.id),
                        built,
                        "{} {}.{}",
                        family.name(),
                        l.name,
                        i.id
                    );
                }
            }
            let windows = &entry["calibrate_windows"];
            assert_eq!(windows.as_object().unwrap().len(), 6);
            assert_eq!(
                windows["mtp+lookup"],
                if family == Family::Qwen4Exp {
                    "mtp"
                } else {
                    "all"
                }
            );
            assert_eq!(windows["mtp+lookup+tail"], "all");
        }
        for family in &families[1..] {
            assert_eq!(family["impls"]["k2_gemm"], serde_json::json!(["cublaslt"]));
        }
    }

    #[test]
    fn ffi_round_trips_and_the_defaults_map_to_the_shipped_values() {
        let d = LeverSet::default().ffi(Family::Qwen4Exp);
        assert_eq!(
            (d.struct_size, d.abi_version, d.block_version, d.count),
            (
                400,
                ffi::ABI_VERSION,
                ffi::SLOT_BLOCK_VERSION,
                ffi::SLOT_COUNT
            )
        );
        let listed: Vec<(u32, u32, [u32; ffi::SLOT_PARAMS])> = d.choices[..d.count as usize]
            .iter()
            .map(|c| (c.slot, c.r#impl, c.params))
            .collect();
        assert_eq!(
            listed,
            [
                (ffi::SLOT_K2_GEMM, ffi::LEVER_K2_CUBLASLT, [0; 4]),
                (
                    ffi::SLOT_PREFILL_ATTENTION,
                    ffi::LEVER_PREFILL_CUDA_CORE,
                    [0; 4]
                ),
                (ffi::SLOT_DECODE_STEP, ffi::LEVER_DECODE_EAGER, [0; 4]),
                (ffi::SLOT_MOE_GEMM, ffi::LEVER_MOE_P3B, [0; 4]),
                (ffi::SLOT_DRAFT_TOKENS, ffi::LEVER_DRAFT_WINDOW, [0; 4]),
                (
                    ffi::SLOT_VERIFY_ATTENTION,
                    ffi::LEVER_VERIFY_PER_ROW,
                    [0; 4]
                ),
                (ffi::SLOT_K2_WEIGHTS, ffi::LEVER_K2_WEIGHTS_BF16, [0; 4]),
            ]
        );
        assert!(d.choices[d.count as usize..]
            .iter()
            .all(|c| *c == ffi::SlotChoice::default()));
        assert_eq!(
            LeverSet::default().calibrate_windows(Family::Qwen4Exp),
            ffi::CALIBRATE_MTP
        );
        let all = LeverSet {
            k2_gemm: K2Gemm::SmallM,
            prefill_attention: PrefillAttention::P4,
            decode_step: DecodeStep::Graph,
            moe_gemm: MoeGemm::K7,
            draft_tokens: DraftTokens::Three,
            verify_attention: VerifyAttention::Batched,
            k2_weights: K2Weights::Fp8,
            drafter: Drafter::MtpLookup,
            draft_policy: DraftPolicy::Adaptive,
            grammar_drafts: GrammarDrafts::On,
            sdar_gemm: SdarGemm::Identical,
        };
        let qwen = |s: &LeverSet| s.ffi(Family::Qwen4Exp);
        assert_eq!(LeverSet::from_ffi(&qwen(&all), &all), Ok(all));
        let slot = |s: &ffi::SlotSelection, slot: u32| s.choices[slot as usize - 1];
        assert_eq!(
            slot(&qwen(&all), ffi::SLOT_K2_WEIGHTS).r#impl,
            ffi::LEVER_K2_WEIGHTS_FP8
        );
        assert_eq!(
            slot(&qwen(&all), ffi::SLOT_DRAFT_TOKENS).params,
            [3, 0, 0, 0]
        );
        // K2Prefill P3: small_m_a8 is value 2 both ways.
        let a8 = LeverSet {
            k2_gemm: K2Gemm::SmallMA8,
            ..all
        };
        assert_eq!(
            slot(&qwen(&a8), ffi::SLOT_K2_GEMM).r#impl,
            ffi::LEVER_K2_SMALL_M_A8
        );
        assert_eq!(LeverSet::from_ffi(&qwen(&a8), &a8), Ok(a8));
        // A partial block in any order: the unlisted slots are their defaults (impl 0, window 0).
        let mut partial = ffi::SlotSelection { count: 2, ..d };
        partial.choices = [ffi::SlotChoice::default(); ffi::SLOT_MAX];
        partial.choices[0] = ffi::SlotChoice {
            slot: ffi::SLOT_MOE_GEMM,
            r#impl: ffi::LEVER_MOE_K7,
            params: [0; 4],
        };
        partial.choices[1] = ffi::SlotChoice {
            slot: ffi::SLOT_K2_GEMM,
            r#impl: ffi::LEVER_K2_SMALL_M,
            params: [0; 4],
        };
        assert_eq!(
            LeverSet::from_ffi(&partial, &LeverSet::default()),
            Ok(LeverSet {
                k2_gemm: K2Gemm::SmallM,
                moe_gemm: MoeGemm::K7,
                ..LeverSet::default()
            })
        );
        let with = |slot: u32, r#impl: u32, params: [u32; 4]| {
            let mut s = d;
            s.choices[slot as usize - 1] = ffi::SlotChoice {
                slot,
                r#impl,
                params,
            };
            s
        };
        for (s, want) in [
            (
                with(ffi::SLOT_K2_GEMM, ffi::LEVER_K2_TAIL_SPLIT + 1, [0; 4]),
                "k2_gemm: the executor reports 4",
            ),
            (
                with(ffi::SLOT_MOE_GEMM, ffi::LEVER_MOE_BUILD, [0; 4]),
                "moe_gemm: the executor reports 0",
            ),
            (
                with(ffi::SLOT_K2_WEIGHTS, ffi::LEVER_K2_WEIGHTS_FP8 + 1, [0; 4]),
                "k2_weights: the executor reports 2",
            ),
            (
                with(
                    ffi::SLOT_DRAFT_TOKENS,
                    ffi::LEVER_DRAFT_WINDOW,
                    [4, 0, 0, 0],
                ),
                "draft_tokens: the executor reports 4",
            ),
            (
                with(ffi::SLOT_DRAFT_TOKENS, ffi::LEVER_DRAFT_WINDOW + 1, [0; 4]),
                "draft_tokens impl: the executor reports 1",
            ),
            (
                with(
                    ffi::SLOT_DRAFT_TOKENS,
                    ffi::LEVER_DRAFT_WINDOW,
                    [3, 1, 0, 0],
                ),
                "params: the executor reports 1",
            ),
            (
                with(ffi::SLOT_K2_GEMM, 0, [1, 0, 0, 0]),
                "params: the executor reports 1",
            ),
            (
                with(ffi::SLOT_K2_GEMM, 0, [0; 4]).with_slot(1, ffi::SLOT_K2_GEMM),
                "slot: the executor reports 1",
            ),
            (
                with(ffi::SLOT_K2_GEMM, 0, [0; 4]).with_slot(0, ffi::SLOT_SDAR_GEMM + 1),
                "slot: the executor reports 9",
            ),
            (
                ffi::SlotSelection {
                    block_version: ffi::SLOT_BLOCK_VERSION + 1,
                    ..d
                },
                "block_version: the executor reports 2",
            ),
            (
                ffi::SlotSelection {
                    count: ffi::SLOT_MAX as u32 + 1,
                    ..d
                },
                "count: the executor reports 17",
            ),
        ] {
            assert_eq!(
                LeverSet::from_ffi(&s, &LeverSet::default()),
                Err(want.to_owned())
            );
        }
    }

    /// SdarOptIn fail-first: `sdar_gemm` is family-scoped to sdar. Its family admits every impl, another family only
    /// the default; a shared lever keeps the ST4 rule (qwen4_exp every impl, sdar the default only).
    #[test]
    fn a_family_scoped_lever_is_built_in_full_on_its_family_only() {
        let sdar_gemm = lever("sdar_gemm").unwrap();
        assert_eq!(sdar_gemm.family, Some(Family::Sdar));
        assert_eq!(
            (sdar_gemm.class, sdar_gemm.layer, sdar_gemm.default),
            (LeverClass::Restart, Layer::Executor, "identical")
        );
        for (family, id, admitted) in [
            (Family::Sdar, "identical", true),
            (Family::Sdar, "changing", true),
            (Family::Qwen4Exp, "identical", true),
            (Family::Qwen4Exp, "changing", false),
        ] {
            assert_eq!(family.admits(sdar_gemm, id), admitted, "{family:?} {id}");
        }
        let k2 = lever("k2_gemm").unwrap();
        assert_eq!(k2.family, None);
        for (family, id, admitted) in [
            (Family::Qwen4Exp, "cublaslt", true),
            (Family::Qwen4Exp, "small_m", true),
            (Family::Sdar, "cublaslt", true),
            (Family::Sdar, "small_m", false),
        ] {
            assert_eq!(family.admits(k2, id), admitted, "{family:?} {id}");
        }
        let scoped: Vec<&str> = REGISTRY
            .iter()
            .filter(|l| l.family.is_some())
            .map(|l| l.name)
            .collect();
        assert_eq!(scoped, ["sdar_gemm"]);
        // The fixture names a lever's family only where it is scoped, so the shared entries keep their form.
        let doc: serde_json::Value = serde_json::from_str(&registry_json()).unwrap();
        for l in doc["levers"].as_array().unwrap() {
            let want = (l["name"] == "sdar_gemm").then(|| serde_json::json!("sdar"));
            assert_eq!(l.get("family").cloned(), want, "{}", l["name"]);
        }
        assert_eq!(
            doc["families"][1]["impls"]["sdar_gemm"],
            serde_json::json!(["identical", "changing"])
        );
        assert_eq!(
            doc["families"][0]["impls"]["sdar_gemm"],
            serde_json::json!(["identical"])
        );
        // "changing" changes output bits: opt-in only, in no numerics class and outside the batched D9.
        let changing = sdar_gemm.impls.iter().find(|i| i.id == "changing").unwrap();
        assert_eq!(
            (
                changing.numerics,
                changing.d9_batch,
                changing.workspace,
                changing.proven_in.len()
            ),
            (Numerics::Changing, false, Workspace::Create, 0)
        );
    }

    /// SdarOptIn fail-first: an sdar profile may opt into `sdar_gemm = "changing"`; a qwen4_exp one is refused by
    /// name, and neither impl is proven on the HIP port (gfx1151).
    #[test]
    fn check_target_admits_sdar_gemm_changing_on_family_sdar_only() {
        let sdar = Target {
            arch: Arch::Sm120a,
            max_rows: 1,
            max_context: 4096,
            speculative_draft_tokens: 0,
            family: Family::Sdar,
        };
        let changing = LeverSet::from_table(
            &BTreeMap::from([("sdar_gemm".to_owned(), "changing".to_owned())]),
            0,
        )
        .unwrap();
        assert_eq!(changing.sdar_gemm, SdarGemm::Changing);
        changing.check_target(&sdar).unwrap();
        // In no registered numerics class: a profile selecting it is never a tuner input.
        assert_eq!(class_of(&changing), None);
        assert_eq!(
            changing.check_target(&Target {
                family: Family::Qwen4Exp,
                max_rows: 16,
                ..sdar
            }),
            Err("sdar_gemm: \"changing\" is not built for family qwen4_exp".to_owned())
        );
        LeverSet::default()
            .check_target(&Target {
                family: Family::Qwen4Exp,
                ..sdar
            })
            .unwrap();
        // On gfx1151 an sdar set already fails at moe_gemm's CUDA-only default (check_target reports the first lever
        // in registry order), so the arch rule on sdar_gemm is held by its registry entry.
        let gfx = Target {
            arch: Arch::Gfx1151,
            ..sdar
        };
        assert_eq!(
            changing.check_target(&gfx),
            Err("moe_gemm: \"p3b\" is not proven on gfx1151".to_owned())
        );
        for i in lever("sdar_gemm").unwrap().impls {
            assert_eq!(i.arch, CUDA, "{}", i.id);
            assert!(
                !i.arch.contains(&gfx.arch),
                "{} is not proven on gfx1151",
                i.id
            );
        }
    }

    /// The Vulkan executor's arch (`vk:<ordinal>`, vulkan-amd-p3-design design note 5.4): the k7 MoE build with the eager
    /// defaults is proven; the registry default `p3b` and every lever value the Vulkan library refuses at first
    /// light (graphs, P4, batched verify, FP8 weights, the small-M GEMMs) fail at config validation, naming the
    /// lever first.
    #[test]
    fn check_target_on_vulkan_accepts_the_eager_k7_set_and_refuses_the_rest() {
        assert_eq!(Target::arch_of("vk:0"), Arch::Vulkan);
        assert_eq!(Target::arch_of("vk:12"), Arch::Vulkan);
        assert_eq!(Arch::Vulkan.name(), "vulkan");
        assert_eq!(Target::arch_of("hip:0"), Arch::Gfx1151);
        assert_eq!(
            Target::arch_of("GPU-aaaaaaaa-0000-4000-8000-000000000000"),
            Arch::Sm120a
        );
        let vk = Target {
            arch: Target::arch_of("vk:0"),
            max_rows: 16,
            max_context: 262_144,
            speculative_draft_tokens: 4,
            family: Family::Qwen4Exp,
        };
        let cuda = Target {
            arch: Arch::Sm120a,
            ..vk
        };
        let k7 = LeverSet {
            moe_gemm: MoeGemm::K7,
            ..LeverSet::default()
        };
        k7.check_target(&vk).unwrap();
        // `p4` is the Vulkan FP16 cooperative-matrix prefill attention: valid on vulkan and on CUDA, refused on HIP.
        let p4 = LeverSet {
            prefill_attention: PrefillAttention::P4,
            ..k7
        };
        p4.check_target(&vk).unwrap();
        p4.check_target(&Target {
            arch: Arch::Sm120a,
            ..vk
        })
        .unwrap();
        assert_eq!(
            p4.check_target(&Target {
                arch: Arch::Gfx1151,
                ..vk
            }),
            Err("prefill_attention: \"p4\" is not proven on gfx1151".to_owned())
        );
        assert_eq!(
            LeverSet::default().check_target(&vk),
            Err("moe_gemm: \"p3b\" is not proven on vulkan".to_owned())
        );
        for (set, want) in [
            (
                LeverSet {
                    k2_gemm: K2Gemm::SmallM,
                    ..k7
                },
                "k2_gemm: \"small_m\" is not proven on vulkan",
            ),
            (
                LeverSet {
                    k2_gemm: K2Gemm::SmallMA8,
                    ..k7
                },
                "k2_gemm: \"small_m_a8\" is not proven on vulkan",
            ),
            (
                LeverSet {
                    decode_step: DecodeStep::Graph,
                    ..k7
                },
                "decode_step: \"graph\" is not proven on vulkan",
            ),
            (
                LeverSet {
                    decode_step: DecodeStep::SpecGraph,
                    ..k7
                },
                "decode_step: \"spec_graph\" is not proven on vulkan",
            ),
            (
                LeverSet {
                    verify_attention: VerifyAttention::Batched,
                    ..k7
                },
                "verify_attention: \"batched\" is not proven on vulkan",
            ),
            (
                LeverSet {
                    k2_weights: K2Weights::Fp8,
                    ..k7
                },
                "k2_weights: \"fp8\" is not proven on vulkan",
            ),
        ] {
            assert_eq!(set.check_target(&vk), Err(want.to_owned()));
            // The same value is valid on the CUDA arch: the refusal is the arch's.
            set.check_target(&cuda).unwrap();
        }
        // VulkanAmd P4-B / D7: `k2_gemm` `tail_split` is the Vulkan GEMM's route alone.
        let split = LeverSet {
            k2_gemm: K2Gemm::TailSplit,
            ..k7
        };
        split.check_target(&vk).unwrap();
        assert_eq!(
            split.check_target(&cuda),
            Err("k2_gemm: \"tail_split\" is not proven on sm_120a".to_owned())
        );
        // Vulkan serves HIP's eager set plus `prefill_attention` `p4` (the FP16 cooperative-matrix path) and
        // `k2_gemm` `tail_split` (the Vulkan GEMM's SPLIT rows).
        for (l, i) in REGISTRY
            .iter()
            .flat_map(|l| l.impls.iter().map(move |i| (l, i)))
        {
            assert_eq!(
                i.arch.contains(&Arch::Vulkan),
                i.arch.contains(&Arch::Gfx1151)
                    || [("prefill_attention", "p4"), ("k2_gemm", "tail_split")]
                        .contains(&(l.name, i.id)),
                "{}.{}",
                l.name,
                i.id
            );
        }
    }

    /// The Vulkan evidence rows (vulkan-amd design note "P3 result"): every implementation available on `Vulkan`
    /// cites tests under `executor/vulkan/` that exist, bar the named unproven one; no other arch carries any.
    #[test]
    fn vulkan_implementations_cite_vulkan_evidence() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut on_vulkan = 0;
        for (l, i) in REGISTRY
            .iter()
            .flat_map(|l| l.impls.iter().map(move |i| (l, i)))
        {
            let what = format!("{}.{}", l.name, i.id);
            let unproven = VULKAN_UNPROVEN.contains(&(l.name, i.id));
            if !i.arch.contains(&Arch::Vulkan) {
                assert!(i.vulkan_contracts.is_empty(), "{what}");
                assert!(!unproven, "{what}");
                continue;
            }
            on_vulkan += 1;
            assert_eq!(i.vulkan_contracts.is_empty(), unproven, "{what}");
            for c in i.vulkan_contracts {
                let (file, test) = c.split_once("::").expect(c);
                assert!(file.starts_with("executor/vulkan/"), "{what}: {c}");
                let text = std::fs::read_to_string(root.join(file))
                    .unwrap_or_else(|e| panic!("{what}: {c}: {e}"));
                assert!(text.contains(&format!("{test}(")), "{what}: {c}");
            }
        }
        assert_eq!(on_vulkan, 12);
        assert_eq!(VULKAN_UNPROVEN, &[] as &[(&str, &str)]);
    }

    /// SdarOptIn fail-first: an sdar block lists the shared slots then `SLOT_SDAR_GEMM` (8 choices, slot order) and
    /// round-trips both impls; a qwen4_exp block never lists slot 8, whatever `sdar_gemm` holds, so a Qwen library
    /// gets today's bytes.
    #[test]
    fn ffi_lists_the_sdar_gemm_slot_on_family_sdar_only() {
        let qwen_default = LeverSet::default().ffi(Family::Qwen4Exp);
        for (sdar_gemm, value) in [
            (SdarGemm::Identical, ffi::LEVER_SDAR_GEMM_IDENTICAL),
            (SdarGemm::Changing, ffi::LEVER_SDAR_GEMM_CHANGING),
        ] {
            let set = LeverSet {
                sdar_gemm,
                ..LeverSet::default()
            };
            let s = set.ffi(Family::Sdar);
            assert_eq!(s.count, ffi::SLOT_COUNT + 1, "{sdar_gemm:?}");
            let slots: Vec<u32> = s.choices[..s.count as usize]
                .iter()
                .map(|c| c.slot)
                .collect();
            let want: Vec<u32> = (1..=ffi::SLOT_COUNT).chain([ffi::SLOT_SDAR_GEMM]).collect();
            assert_eq!(slots, want, "{sdar_gemm:?}");
            assert_eq!(
                s.choices[ffi::SLOT_COUNT as usize],
                ffi::SlotChoice {
                    slot: ffi::SLOT_SDAR_GEMM,
                    r#impl: value,
                    params: [0; ffi::SLOT_PARAMS],
                }
            );
            assert_eq!(
                s.choices[..ffi::SLOT_COUNT as usize],
                qwen_default.choices[..ffi::SLOT_COUNT as usize]
            );
            assert!(s.choices[s.count as usize..]
                .iter()
                .all(|c| *c == ffi::SlotChoice::default()));
            assert_eq!(LeverSet::from_ffi(&s, &set), Ok(set));
            let q = set.ffi(Family::Qwen4Exp);
            assert_eq!(q, qwen_default, "{sdar_gemm:?}");
            assert_eq!(q.count, ffi::SLOT_COUNT);
            assert!(q.choices.iter().all(|c| c.slot != ffi::SLOT_SDAR_GEMM));
        }
        // An unlisted slot 8 (a qwen4_exp library's readback) is "identical".
        assert_eq!(
            LeverSet::from_ffi(&qwen_default, &LeverSet::default()).map(|l| l.sdar_gemm),
            Ok(SdarGemm::Identical)
        );
        let sdar = LeverSet::default().ffi(Family::Sdar);
        let mut unknown = sdar;
        unknown.choices[ffi::SLOT_COUNT as usize].r#impl = ffi::LEVER_SDAR_GEMM_CHANGING + 1;
        let mut param = sdar;
        param.choices[ffi::SLOT_COUNT as usize].params[0] = 1;
        for (s, want) in [
            (unknown, "sdar_gemm: the executor reports 2"),
            (param, "params: the executor reports 1"),
            (
                sdar.with_slot(0, ffi::SLOT_SDAR_GEMM),
                "slot: the executor reports 8",
            ),
        ] {
            assert_eq!(
                LeverSet::from_ffi(&s, &LeverSet::default()),
                Err(want.to_owned())
            );
        }
    }

    /// SdarOptIn: the family-scoped slot lies past every shared slot and inside the block's capacity.
    #[test]
    fn the_sdar_gemm_slot_is_past_the_shared_slots() {
        let shared = [
            ffi::SLOT_K2_GEMM,
            ffi::SLOT_PREFILL_ATTENTION,
            ffi::SLOT_DECODE_STEP,
            ffi::SLOT_MOE_GEMM,
            ffi::SLOT_DRAFT_TOKENS,
            ffi::SLOT_VERIFY_ATTENTION,
            ffi::SLOT_K2_WEIGHTS,
        ];
        assert_eq!(shared, [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(ffi::SLOT_COUNT, 7);
        assert!((1..=ffi::SLOT_COUNT).all(|s| s != ffi::SLOT_SDAR_GEMM));
        assert!(!shared.contains(&ffi::SLOT_SDAR_GEMM));
        assert_eq!(ffi::SLOT_SDAR_GEMM, 8);
        assert!(ffi::SLOT_SDAR_GEMM < ffi::SLOT_MAX as u32);
    }

    trait WithSlot {
        fn with_slot(self, index: usize, slot: u32) -> Self;
    }

    impl WithSlot for ffi::SlotSelection {
        /// `self` with choice `index`'s slot replaced (a duplicate or an unknown slot).
        fn with_slot(mut self, index: usize, slot: u32) -> Self {
            self.choices[index].slot = slot;
            self
        }
    }

    /// PI-01: `tests/fixtures/lever_registry.json` is the serialized registry (the Python profile contract reads
    /// it). On a registry change, write [`registry_json`]'s output (printed below) to the fixture.
    #[test]
    fn the_fixture_equals_the_registry() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/lever_registry.json");
        let fixture = std::fs::read_to_string(&path).unwrap();
        let want = registry_json();
        assert!(
            fixture == want,
            "{} differs from the registry:\n{want}",
            path.display()
        );
    }
}
