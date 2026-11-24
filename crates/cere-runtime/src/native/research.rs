//! WeightLab research plans (`-v57`, weightlab design note WL-01..WL-04, WL-11, WL-14, WL-20; docs/API.md
//! "Research (WeightLab)"): opt-in capture and intervention on the native executor's sessions.
//!
//! [`parse_plan`] turns a `POST /v1/cere/research/plans` body into a validated [`Plan`]: capture ops (a point, its
//! layers, prompt ranges and generated positions) and intervention ops (track WL-B's kinds at the P points), each
//! checked against the executor's point table ([`POINTS`], the Rust mirror of `executor/cuda/src/research/points.cpp`)
//! and the field rules of `research/patch_check.cpp`, its vectors resolved in the deployment's [`VectorStore`]. The
//! plan's `plan_sha256` is the SHA-256 of its canonical JSON (sorted keys, no whitespace); its registration bound
//! ([`Plan::capture_bound_bytes`]) is checked against `max_capture_bytes_per_request`.
//!
//! [`Research`] is the deployment's registry: plans in executor slots (`cere_model_research_plan_load`), scope
//! bindings, the host-buffer reservations of running and undrained captures (the admission bound,
//! [`Plan::request_bound_bytes`], equal to the executor's `bound()`), and the finished captures (manifest plus
//! safetensors), each drained once. A request of a bound scope holds its plan ([`Applied`]) until it ends, so a
//! plan in use cannot be deleted.
use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::app::NativeBackend;
use crate::config::ResearchConfig;
use crate::digest::hex;
use crate::native::driver::{CaptureData, Driver, DriverError, Model};
use crate::native::ffi::{self, StatusError};
use crate::native::research_vectors::{Vector, VectorStore};

/// WL-20: the registered targets a plan may name (the pre-registered first target only).
pub const TARGETS: &[&str] = &["coding_output_volume"];

/// The model's MoE top_k (the routing point's ids per row; the expert kinds' limits). The executor applies it itself:
/// `cere_research_op.top_k` is for `native.logits.topk` only (0 for every other op, research/plan.cpp `load_op`).
pub const MOE_TOP_K: u32 = 10;
/// Routed experts (the router-logit point's columns).
pub const EXPERTS: u32 = 512;
/// Most rows a replace vector may have (research/patch_check.h kMaxVectorRows).
const MAX_VECTOR_ROWS: u32 = 4096;

/// Which layers a point exists at (points.h `Where`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Where {
    All,
    FullAttention,
    Gdn,
    Final,
}

/// One tensor of a capture row: name suffix, `CERE_RESEARCH_DTYPE_*`, columns (0 = the op's `top_k`).
#[derive(Debug)]
pub struct Part {
    pub suffix: &'static str,
    pub dtype: u32,
    pub cols: u32,
}

/// One research point (points.h `PointInfo`).
#[derive(Debug)]
pub struct Point {
    pub name: &'static str,
    pub at: Where,
    pub parts: &'static [Part],
    /// Columns an intervention edits (the expert count for the router and routing points).
    pub edit_cols: u32,
    /// Takes add / ablate_* / replace.
    pub dense: bool,
    /// Takes expert_bias / expert_mask.
    pub router: bool,
    /// Takes expert_force.
    pub route: bool,
    /// Scored rows only: generated positions only.
    pub logits: bool,
}

const BF16: u32 = ffi::RESEARCH_DTYPE_BF16;
const I32: u32 = ffi::RESEARCH_DTYPE_I32;
const F32: u32 = ffi::RESEARCH_DTYPE_F32;

const fn one(cols: u32) -> &'static [Part] {
    match cols {
        10_240 => &[Part {
            suffix: "",
            dtype: BF16,
            cols: 10_240,
        }],
        13_312 => &[Part {
            suffix: "",
            dtype: BF16,
            cols: 13_312,
        }],
        6_144 => &[Part {
            suffix: "",
            dtype: BF16,
            cols: 6_144,
        }],
        512 => &[Part {
            suffix: "",
            dtype: BF16,
            cols: 512,
        }],
        640 => &[Part {
            suffix: "",
            dtype: BF16,
            cols: 640,
        }],
        248_320 => &[Part {
            suffix: "",
            dtype: BF16,
            cols: 248_320,
        }],
        _ => &[Part {
            suffix: "",
            dtype: BF16,
            cols: 2_560,
        }],
    }
}

const fn dense(name: &'static str, at: Where, cols: u32) -> Point {
    Point {
        name,
        at,
        parts: one(cols),
        edit_cols: cols,
        dense: true,
        router: false,
        route: false,
        logits: false,
    }
}

/// The executor's research points, in `PointId` order (points.cpp `kPoints`).
pub const POINTS: &[Point] = &[
    Point {
        edit_cols: 0,
        dense: false,
        ..dense("native.layers.L.attn_mix.in", Where::All, 10_240)
    },
    dense("native.layers.L.attn_mix.out", Where::All, 2_560),
    Point {
        edit_cols: 0,
        dense: false,
        ..dense("native.layers.L.mlp_mix.in", Where::All, 10_240)
    },
    dense("native.layers.L.mlp_mix.out", Where::All, 2_560),
    dense(
        "native.layers.L.gemm.attn_qkv.out",
        Where::FullAttention,
        13_312,
    ),
    dense("native.layers.L.gemm.gdn_qkv.out", Where::Gdn, 10_240),
    dense("native.layers.L.gemm.gdn_z.out", Where::Gdn, 6_144),
    dense(
        "model.layers.L.self_attn.o_proj",
        Where::FullAttention,
        2_560,
    ),
    dense("model.layers.L.linear_attn.out_proj", Where::Gdn, 2_560),
    Point {
        router: true,
        ..dense("native.layers.L.gemm.moe_router.out", Where::All, 512)
    },
    Point {
        name: "native.layers.L.moe_route",
        at: Where::All,
        parts: &[
            Part {
                suffix: ".ids",
                dtype: I32,
                cols: MOE_TOP_K,
            },
            Part {
                suffix: ".weights",
                dtype: F32,
                cols: MOE_TOP_K,
            },
        ],
        edit_cols: EXPERTS,
        dense: false,
        router: false,
        route: true,
        logits: false,
    },
    dense("model.layers.L.mlp.experts", Where::All, 2_560),
    dense(
        "model.layers.L.mlp.shared_expert.down_proj",
        Where::All,
        2_560,
    ),
    Point {
        edit_cols: 0,
        dense: false,
        ..dense("native.layers.L.gemm.moe_shared_down.in", Where::All, 640)
    },
    dense("native.final_mix.out", Where::Final, 2_560),
    Point {
        name: "native.logits.topk",
        at: Where::Final,
        parts: &[
            Part {
                suffix: ".ids",
                dtype: I32,
                cols: 0,
            },
            Part {
                suffix: ".logprobs",
                dtype: F32,
                cols: 0,
            },
        ],
        edit_cols: 0,
        dense: false,
        router: false,
        route: false,
        logits: true,
    },
    Point {
        edit_cols: 0,
        dense: false,
        logits: true,
        ..dense("native.logits", Where::Final, 248_320)
    },
];

impl Point {
    /// The point named `name` (exact match).
    pub fn find(name: &str) -> Option<&'static Point> {
        POINTS.iter().find(|p| p.name == name)
    }

    /// The layer mask the point exists at (bit L, L < 48; bit 48 for the final points).
    pub fn layers(&self) -> u64 {
        let full = |l: u32| l % 4 == 3;
        match self.at {
            Where::Final => 1u64 << ffi::RESEARCH_FINAL_LAYER,
            at => (0..ffi::RESEARCH_FINAL_LAYER)
                .filter(|&l| match at {
                    Where::All => true,
                    Where::FullAttention => full(l),
                    _ => !full(l),
                })
                .fold(0, |m, l| m | (1u64 << l)),
        }
    }

    /// Bytes of one row (every part; `top_k` for `native.logits.topk`).
    pub fn row_bytes(&self, top_k: u32) -> u64 {
        self.parts
            .iter()
            .map(|p| {
                let cols = if p.cols == 0 { top_k } else { p.cols };
                u64::from(cols) * if p.dtype == BF16 { 2 } else { 4 }
            })
            .sum()
    }
}

/// Why a research operation was refused: HTTP status, error code and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResearchError {
    /// 400 `invalid_research_plan`: a body, point, layer, position, kind, field, vector or target the plan may not
    /// have, or an executor refusal of the plan.
    Invalid(String),
    /// 400 `research_budget_exceeded`: the registration bound is above `max_capture_bytes_per_request`.
    Budget { bound: u64, cap: u64 },
    /// 400 `invalid_request`: a request the plan cannot run (`n` > 1).
    Request(String),
    /// 404 with its code (`research_plan_not_found`, `research_scope_not_found`, `research_binding_not_found`,
    /// `research_capture_not_found`).
    NotFound(&'static str, String),
    /// 409 with its code (`research_scope_bound`, `research_plan_busy`).
    Conflict(&'static str, String),
    /// 503 `research_buffer_full`: the request's bound does not fit the host buffer (nothing ran).
    BufferFull { bound: u64, free: u64 },
    /// 503 `not_ready`: the native backend is not loaded.
    NotReady,
    /// 507 `research_plans_full`: `max_plans` plans are registered.
    PlansFull(u32),
    /// 500 `research_failed`: the executor failed otherwise.
    Internal(String),
}

impl ResearchError {
    /// HTTP status and error code.
    pub fn status_code(&self) -> (u16, &'static str) {
        match self {
            Self::Invalid(_) => (400, "invalid_research_plan"),
            Self::Budget { .. } => (400, "research_budget_exceeded"),
            Self::Request(_) => (400, "invalid_request"),
            Self::NotFound(code, _) => (404, code),
            Self::Conflict(code, _) => (409, code),
            Self::BufferFull { .. } => (503, "research_buffer_full"),
            Self::NotReady => (503, "not_ready"),
            Self::PlansFull(_) => (507, "research_plans_full"),
            Self::Internal(_) => (500, "research_failed"),
        }
    }

    /// The API error of a chat request refused by its plan (admission, `n`).
    pub fn api_error(&self) -> crate::api_types::ApiError {
        use crate::api_types::{ApiError, INVALID_REQUEST, SERVER_ERROR};
        let (status, code) = self.status_code();
        let kind = if status >= 500 {
            SERVER_ERROR
        } else {
            INVALID_REQUEST
        };
        ApiError::new(status, kind, code, self.to_string(), None)
    }
}

impl std::fmt::Display for ResearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(m) | Self::Request(m) | Self::Internal(m) => f.write_str(m),
            Self::NotFound(_, m) | Self::Conflict(_, m) => f.write_str(m),
            Self::Budget { bound, cap } => write!(
                f,
                "the plan's capture bound {bound} bytes exceeds max_capture_bytes_per_request {cap}"
            ),
            Self::BufferFull { bound, free } => write!(
                f,
                "the request's capture bound {bound} bytes does not fit the research host buffer ({free} free)"
            ),
            Self::NotReady => f.write_str("the native backend is not loaded"),
            Self::PlansFull(max) => write!(f, "{max} plans are registered (max_plans); delete one first"),
        }
    }
}

impl std::error::Error for ResearchError {}

fn invalid(message: impl Into<String>) -> ResearchError {
    ResearchError::Invalid(message.into())
}

// ---------------------------------------------------------------------------------------------------------
// The plan body

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanBody {
    #[serde(default)]
    capture: Vec<OpBody>,
    #[serde(default)]
    interventions: Vec<OpBody>,
    seed: Option<u64>,
    target: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpBody {
    kind: Option<String>,
    point: String,
    layers: Option<LayersBody>,
    positions: PositionsBody,
    top_k: Option<u32>,
    vector_id: Option<String>,
    alpha: Option<f64>,
    cols: Option<[u32; 2]>,
    experts: Option<Vec<i64>>,
    weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum LayersBody {
    Named(String),
    List(Vec<u32>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PositionsBody {
    #[serde(default)]
    prompt: Vec<[u64; 2]>,
    generated: Option<GeneratedBody>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GeneratedBody {
    Every(bool),
    First(u64),
}

/// Which generated rows an op selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generated {
    None,
    /// Every generated row (`generated: true`, `generated_max` 0 = max_tokens).
    Every,
    /// The first N (`generated: N`, N >= 1).
    First(u64),
}

/// One validated op in `cere_research_op` form (owned; [`Plan::ffi_ops`] points into it).
#[derive(Debug)]
pub struct Op {
    pub kind: u32,
    pub point: &'static Point,
    point_c: CString,
    pub layers: u64,
    pub prompt: Vec<ffi::ResearchRange>,
    pub generated: Generated,
    pub top_k: u32,
    pub alpha: f32,
    pub cols: (u32, u32),
    pub vector: Option<Arc<Vector>>,
    pub experts: Vec<i32>,
    pub weights: Vec<f32>,
}

impl Op {
    /// The op's capture rows for a request of `prompt_tokens` P and `max_tokens` M (the executor's row selection):
    /// prompt positions in the ranges ∩ [0, P), generated positions [P - 1, P - 1 + g) with g = min(N or M, M),
    /// their union.
    fn request_rows(&self, prompt_tokens: u64, max_tokens: u64) -> u64 {
        let in_prompt = |first: u64, end: u64| -> u64 {
            self.prompt
                .iter()
                .map(|r| r.end.min(end).saturating_sub(r.first.max(first)))
                .sum()
        };
        let prompt = in_prompt(0, prompt_tokens);
        let g = match self.generated {
            Generated::None => 0,
            Generated::Every => max_tokens,
            Generated::First(n) => n.min(max_tokens),
        };
        if g == 0 || prompt_tokens == 0 {
            return prompt;
        }
        // Only position P - 1 can be in both (the prompt rows end at P).
        let start = prompt_tokens - 1;
        prompt + g - in_prompt(start, prompt_tokens)
    }

    /// The op's registration rows: every prompt range's length plus N, or the context limit for `true`.
    fn plan_rows(&self, context: u64) -> u64 {
        let prompt: u64 = self.prompt.iter().map(|r| r.end - r.first).sum();
        prompt
            + match self.generated {
                Generated::None => 0,
                Generated::Every => context,
                Generated::First(n) => n,
            }
    }

    fn bytes(&self, rows: u64) -> u64 {
        if self.kind != ffi::RESEARCH_CAPTURE {
            return 0;
        }
        rows.saturating_mul(u64::from(self.layers.count_ones()))
            .saturating_mul(self.point.row_bytes(self.top_k))
    }
}

/// A validated research plan.
#[derive(Debug)]
pub struct Plan {
    pub ops: Vec<Op>,
    pub seed: Option<u64>,
    pub target: String,
    pub sha256: [u8; 32],
    /// The plan's canonical JSON (sorted keys, no whitespace) whose SHA-256 is [`Plan::sha256`].
    pub canonical: String,
    /// The deployment's context limit (`generated: true`'s registration rows).
    context: u64,
}

impl Plan {
    pub fn sha256_hex(&self) -> String {
        hex(&self.sha256)
    }

    /// Any intervention op: its sessions get their own prefix-cache namespace (WL-11).
    pub fn intervenes(&self) -> bool {
        self.ops.iter().any(|o| o.kind != ffi::RESEARCH_CAPTURE)
    }

    /// The prefix-cache namespace of a request under this plan (WL-11): an intervening plan's is its own
    /// ([`namespace`] of `base` and the plan digest); a capture-only plan keeps `base` (the executor skips the
    /// prefix lookup for a prompt capture).
    pub fn cache_namespace(&self, base: [u8; 16]) -> [u8; 16] {
        match self.intervenes() {
            true => namespace(base, &self.sha256),
            false => base,
        }
    }

    /// Any capture op.
    pub fn captures(&self) -> bool {
        self.ops.iter().any(|o| o.kind == ffi::RESEARCH_CAPTURE)
    }

    /// The registration bound: over capture ops and their layers, rows x row bytes, rows = the prompt ranges'
    /// lengths + N (or the context limit for `generated: true`).
    pub fn capture_bound_bytes(&self) -> u64 {
        self.ops
            .iter()
            .map(|o| o.bytes(o.plan_rows(self.context)))
            .fold(0, u64::saturating_add)
    }

    /// The admission bound of a request of `prompt_tokens` and `max_tokens` (the executor's `bound()`).
    pub fn request_bound_bytes(&self, prompt_tokens: u64, max_tokens: u64) -> u64 {
        self.ops
            .iter()
            .map(|o| o.bytes(o.request_rows(prompt_tokens, max_tokens)))
            .fold(0, u64::saturating_add)
    }

    /// The ops in `cere_research_op` form; every pointer is into `self`, valid while it lives.
    pub fn ffi_ops(&self) -> Vec<ffi::ResearchOp> {
        let ptr_or_null = |empty: bool, p: *const u8| if empty { std::ptr::null() } else { p };
        self.ops
            .iter()
            .map(|o| {
                let (generated, generated_max) = match o.generated {
                    Generated::None => (0, 0),
                    Generated::Every => (1, 0),
                    Generated::First(n) => (1, n),
                };
                let (vector_rows, vector_cols, vector) = match &o.vector {
                    Some(v) => (v.rows, v.cols, v.data.as_ptr().cast()),
                    None => (0, 0, std::ptr::null()),
                };
                ffi::ResearchOp {
                    kind: o.kind,
                    top_k: o.top_k,
                    point: o.point_c.as_ptr(),
                    layers: o.layers,
                    prompt: ptr_or_null(o.prompt.is_empty(), o.prompt.as_ptr().cast()).cast(),
                    prompt_count: o.prompt.len() as u32,
                    generated,
                    generated_max,
                    alpha: o.alpha,
                    col_begin: o.cols.0,
                    col_end: o.cols.1,
                    vector_rows,
                    vector,
                    vector_cols,
                    expert_count: o.experts.len() as u32,
                    experts: ptr_or_null(o.experts.is_empty(), o.experts.as_ptr().cast()).cast(),
                    weights: ptr_or_null(o.weights.is_empty(), o.weights.as_ptr().cast()).cast(),
                }
            })
            .collect()
    }
}

fn kind_of(name: &str) -> Option<u32> {
    Some(match name {
        "add" => ffi::RESEARCH_ADD,
        "ablate_zero" => ffi::RESEARCH_ABLATE_ZERO,
        "ablate_mean" => ffi::RESEARCH_ABLATE_MEAN,
        "replace" => ffi::RESEARCH_REPLACE,
        "expert_bias" => ffi::RESEARCH_EXPERT_BIAS,
        "expert_mask" => ffi::RESEARCH_EXPERT_MASK,
        "expert_force" => ffi::RESEARCH_EXPERT_FORCE,
        _ => return None,
    })
}

fn kind_name(kind: u32) -> &'static str {
    match kind {
        ffi::RESEARCH_ADD => "add",
        ffi::RESEARCH_ABLATE_ZERO => "ablate_zero",
        ffi::RESEARCH_ABLATE_MEAN => "ablate_mean",
        ffi::RESEARCH_REPLACE => "replace",
        ffi::RESEARCH_EXPERT_BIAS => "expert_bias",
        ffi::RESEARCH_EXPERT_MASK => "expert_mask",
        ffi::RESEARCH_EXPERT_FORCE => "expert_force",
        _ => "capture",
    }
}

/// Parses and validates a plan body (see the module doc). `context` is the deployment's context limit,
/// `vectors` the store intervention ops resolve their `vector_id` in.
pub fn parse_plan(body: &[u8], context: u64, vectors: &VectorStore) -> Result<Plan, ResearchError> {
    let body: PlanBody =
        serde_json::from_slice(body).map_err(|e| invalid(format!("the plan body: {e}")))?;
    if !TARGETS.contains(&body.target.as_str()) {
        return Err(invalid(format!(
            "target {:?} is not a registered target (allowed: {})",
            body.target,
            TARGETS.join(", ")
        )));
    }
    let count = body.capture.len() + body.interventions.len();
    if count == 0 || count > ffi::RESEARCH_MAX_OPS as usize {
        return Err(invalid(format!(
            "a plan has 1..={} ops (capture + interventions), not {count}",
            ffi::RESEARCH_MAX_OPS
        )));
    }
    let mut ops = Vec::with_capacity(count);
    for (i, op) in body.capture.into_iter().enumerate() {
        ops.push(parse_op(&format!("capture[{i}]"), op, false, vectors)?);
    }
    for (i, op) in body.interventions.into_iter().enumerate() {
        ops.push(parse_op(&format!("interventions[{i}]"), op, true, vectors)?);
    }
    let canonical = canonical_json(&ops, body.seed, &body.target);
    let sha256 = Sha256::digest(canonical.as_bytes()).into();
    Ok(Plan {
        ops,
        seed: body.seed,
        target: body.target,
        sha256,
        canonical,
        context,
    })
}

fn parse_op(
    p: &str,
    op: OpBody,
    intervention: bool,
    vectors: &VectorStore,
) -> Result<Op, ResearchError> {
    let point = Point::find(&op.point)
        .ok_or_else(|| invalid(format!("{p}.point {:?} is not a research point", op.point)))?;
    let kind = match (&op.kind, intervention) {
        (None, false) => ffi::RESEARCH_CAPTURE,
        (Some(_), false) => return Err(invalid(format!("{p}: a capture op takes no kind"))),
        (None, true) => return Err(invalid(format!("{p}.kind is required"))),
        (Some(k), true) => kind_of(k)
            .ok_or_else(|| invalid(format!("{p}.kind {k:?} is not an intervention kind")))?,
    };
    // Layers: a list of layers the point exists at, or "all"; final points take none (or [48]).
    let exists = point.layers();
    let layers = match (&op.layers, point.at) {
        (None, Where::Final) => exists,
        (None, _) => {
            return Err(invalid(format!(
                "{p}.layers is required for {}",
                point.name
            )))
        }
        (Some(LayersBody::Named(s)), _) if s == "all" => exists,
        (Some(LayersBody::Named(s)), _) => {
            return Err(invalid(format!(
                "{p}.layers {s:?}: a list of layers or \"all\""
            )))
        }
        (Some(LayersBody::List(list)), _) => {
            if list.is_empty() {
                return Err(invalid(format!("{p}.layers is empty")));
            }
            let mut mask = 0u64;
            for &l in list {
                let bit = if l <= ffi::RESEARCH_FINAL_LAYER {
                    1u64 << l
                } else {
                    0
                };
                if bit & exists == 0 {
                    return Err(invalid(format!(
                        "{p}.layers: {} has no layer {l}",
                        point.name
                    )));
                }
                if mask & bit != 0 {
                    return Err(invalid(format!("{p}.layers: layer {l} is listed twice")));
                }
                mask |= bit;
            }
            mask
        }
    };
    // Positions: sorted disjoint [first, end) prompt ranges (at most 64) and/or generated rows.
    let prompt = &op.positions.prompt;
    if prompt.len() > ffi::RESEARCH_MAX_RANGES as usize {
        return Err(invalid(format!(
            "{p}.positions.prompt has {} ranges (at most {})",
            prompt.len(),
            ffi::RESEARCH_MAX_RANGES
        )));
    }
    let mut last_end = 0u64;
    for (i, [first, end]) in prompt.iter().copied().enumerate() {
        if end <= first {
            return Err(invalid(format!(
                "{p}.positions.prompt[{i}]: end must exceed first"
            )));
        }
        if i > 0 && first < last_end {
            return Err(invalid(format!(
                "{p}.positions.prompt[{i}]: ranges must be sorted and disjoint"
            )));
        }
        last_end = end;
    }
    let generated = match op.positions.generated {
        None | Some(GeneratedBody::Every(false)) => Generated::None,
        Some(GeneratedBody::Every(true)) => Generated::Every,
        Some(GeneratedBody::First(0)) => {
            return Err(invalid(format!(
                "{p}.positions.generated: N must be at least 1"
            )))
        }
        Some(GeneratedBody::First(n)) => Generated::First(n),
    };
    if prompt.is_empty() && generated == Generated::None {
        return Err(invalid(format!("{p}.positions selects no position")));
    }
    if point.logits {
        if !prompt.is_empty() {
            return Err(invalid(format!(
                "{p}: {} takes generated positions only",
                point.name
            )));
        }
        if point.name == "native.logits" {
            let max = u64::from(ffi::RESEARCH_MAX_LOGIT_POSITIONS);
            if !matches!(generated, Generated::First(n) if n <= max) {
                return Err(invalid(format!(
                    "{p}: native.logits takes generated: N with N <= {max}"
                )));
            }
        }
    }
    let top_k = match (point.name == "native.logits.topk", op.top_k) {
        (true, Some(k)) if (1..=ffi::RESEARCH_MAX_TOPK).contains(&k) => k,
        (true, _) => {
            return Err(invalid(format!(
                "{p}.top_k in [1, {}] is required for native.logits.topk",
                ffi::RESEARCH_MAX_TOPK
            )))
        }
        (false, Some(_)) => {
            return Err(invalid(format!("{p}.top_k is for native.logits.topk only")))
        }
        (false, None) => 0,
    };
    let ranges: Vec<ffi::ResearchRange> = prompt
        .iter()
        .map(|&[first, end]| ffi::ResearchRange { first, end })
        .collect();
    let mut out = Op {
        kind,
        point,
        point_c: CString::new(point.name).expect("no NUL in the table"),
        layers,
        prompt: ranges,
        generated,
        top_k,
        alpha: 0.0,
        cols: (0, 0),
        vector: None,
        experts: Vec::new(),
        weights: Vec::new(),
    };
    if kind == ffi::RESEARCH_CAPTURE {
        if op.vector_id.is_some()
            || op.alpha.is_some()
            || op.cols.is_some()
            || op.experts.is_some()
            || op.weights.is_some()
        {
            return Err(invalid(format!(
                "{p}: a capture op takes no vector_id, alpha, cols, experts or weights"
            )));
        }
        return Ok(out);
    }
    intervention_fields(p, &mut out, op, vectors)?;
    Ok(out)
}

/// The kind's fields (research/patch_check.cpp `check_edit`, the executor's check at plan_load).
fn intervention_fields(
    p: &str,
    out: &mut Op,
    op: OpBody,
    vectors: &VectorStore,
) -> Result<(), ResearchError> {
    let point = out.point;
    let kind = kind_name(out.kind);
    let dense = matches!(
        out.kind,
        ffi::RESEARCH_ADD
            | ffi::RESEARCH_ABLATE_ZERO
            | ffi::RESEARCH_ABLATE_MEAN
            | ffi::RESEARCH_REPLACE
    );
    let fits = match out.kind {
        _ if dense => point.dense,
        ffi::RESEARCH_EXPERT_FORCE => point.route,
        _ => point.router,
    };
    if !fits {
        return Err(invalid(format!(
            "{p}: {kind} does not apply at {}",
            point.name
        )));
    }
    if out.kind == ffi::RESEARCH_ADD {
        let alpha = op
            .alpha
            .ok_or_else(|| invalid(format!("{p}.alpha is required for add")))?;
        let a = alpha as f32;
        if !a.is_finite() {
            return Err(invalid(format!("{p}.alpha must be a finite f32")));
        }
        out.alpha = a;
    } else if op.alpha.is_some() {
        return Err(invalid(format!("{p}.alpha is for add only")));
    }
    if dense {
        if op.experts.is_some() || op.weights.is_some() {
            return Err(invalid(format!("{p}: {kind} takes no experts or weights")));
        }
        if let Some([b, e]) = op.cols {
            if b >= e || e > point.edit_cols {
                return Err(invalid(format!(
                    "{p}.cols [{b}, {e}) must be a non-empty range within the point's {} columns",
                    point.edit_cols
                )));
            }
            out.cols = (b, e);
        }
        let needs_vector = out.kind != ffi::RESEARCH_ABLATE_ZERO;
        match (&op.vector_id, needs_vector) {
            (Some(_), false) => {
                return Err(invalid(format!("{p}: ablate_zero takes no vector_id")))
            }
            (None, true) => return Err(invalid(format!("{p}.vector_id is required for {kind}"))),
            (None, false) => return Ok(()),
            (Some(id), true) => {
                let v = vectors.get(id).ok_or_else(|| {
                    invalid(format!("{p}.vector_id {id:?} is not a stored vector"))
                })?;
                if v.cols != point.edit_cols {
                    return Err(invalid(format!(
                        "{p}: vector {id} has {} columns, {} has {}",
                        v.cols, point.name, point.edit_cols
                    )));
                }
                let positions = match out.generated {
                    Generated::Every => None,
                    Generated::None => Some(0),
                    Generated::First(n) => Some(n),
                }
                .map(|g| g + out.prompt.iter().map(|r| r.end - r.first).sum::<u64>());
                let rows_ok = match out.kind {
                    ffi::RESEARCH_REPLACE => {
                        v.rows == 1
                            || (v.rows <= MAX_VECTOR_ROWS && positions == Some(u64::from(v.rows)))
                    }
                    _ => v.rows == 1,
                };
                if !rows_ok {
                    return Err(invalid(format!(
                        "{p}: vector {id} has {} rows; {kind} takes 1{}",
                        v.rows,
                        if out.kind == ffi::RESEARCH_REPLACE {
                            " or one per planned position"
                        } else {
                            ""
                        }
                    )));
                }
                out.vector = Some(v);
            }
        }
        return Ok(());
    }
    // Expert kinds: ids in [0, 512), unique; no vector or columns.
    if op.vector_id.is_some() || op.cols.is_some() {
        return Err(invalid(format!("{p}: {kind} takes no vector_id or cols")));
    }
    let experts = op.experts.unwrap_or_default();
    let mut seen = vec![false; EXPERTS as usize];
    for &e in &experts {
        if !(0..i64::from(EXPERTS)).contains(&e) || std::mem::replace(&mut seen[e as usize], true) {
            return Err(invalid(format!(
                "{p}.experts: ids are distinct and in [0, {EXPERTS}), not {e}"
            )));
        }
    }
    out.experts = experts.iter().map(|&e| e as i32).collect();
    let weights: Vec<f32> = op
        .weights
        .unwrap_or_default()
        .iter()
        .map(|&w| w as f32)
        .collect();
    match out.kind {
        ffi::RESEARCH_EXPERT_MASK => {
            if !weights.is_empty() {
                return Err(invalid(format!("{p}: expert_mask takes no weights")));
            }
            if EXPERTS - (out.experts.len() as u32) < MOE_TOP_K {
                return Err(invalid(format!(
                    "{p}: expert_mask must leave at least {MOE_TOP_K} experts unmasked"
                )));
            }
        }
        ffi::RESEARCH_EXPERT_BIAS => {
            if weights.len() != out.experts.len() || weights.iter().any(|w| !w.is_finite()) {
                return Err(invalid(format!(
                    "{p}: expert_bias takes one finite weight per expert"
                )));
            }
        }
        _ => {
            let n = out.experts.len();
            let sum: f64 = weights.iter().map(|&w| f64::from(w)).sum();
            if n == 0
                || n > MOE_TOP_K as usize
                || weights.len() != n
                || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
                || sum > 1.0
            {
                return Err(invalid(format!(
                    "{p}: expert_force takes 1..={MOE_TOP_K} experts with one finite weight >= 0 each, \
                     summing to at most 1"
                )));
            }
        }
    }
    out.weights = weights;
    Ok(())
}

/// The canonical JSON of a parsed plan: ops in order with resolved layers (a list), positions, and the kind's
/// fields (vectors by id and SHA-256), `seed` and `target`; serde_json's sorted maps, no whitespace.
fn canonical_json(ops: &[Op], seed: Option<u64>, target: &str) -> String {
    let op_value = |o: &Op| {
        let mut m = Map::new();
        m.insert("point".into(), json!(o.point.name));
        let layers: Vec<u32> = (0..=ffi::RESEARCH_FINAL_LAYER)
            .filter(|l| o.layers & (1u64 << l) != 0)
            .collect();
        m.insert("layers".into(), json!(layers));
        let generated = match o.generated {
            Generated::None => json!(false),
            Generated::Every => json!(true),
            Generated::First(n) => json!(n),
        };
        let prompt: Vec<[u64; 2]> = o.prompt.iter().map(|r| [r.first, r.end]).collect();
        m.insert(
            "positions".into(),
            json!({"prompt": prompt, "generated": generated}),
        );
        if o.kind == ffi::RESEARCH_CAPTURE {
            if o.top_k > 0 {
                m.insert("top_k".into(), json!(o.top_k));
            }
            return Value::Object(m);
        }
        m.insert("kind".into(), json!(kind_name(o.kind)));
        if o.kind == ffi::RESEARCH_ADD {
            m.insert("alpha".into(), json!(o.alpha));
        }
        if o.cols != (0, 0) {
            m.insert("cols".into(), json!([o.cols.0, o.cols.1]));
        }
        if let Some(v) = &o.vector {
            m.insert("vector_id".into(), json!(v.id));
            m.insert("vector_sha256".into(), json!(v.sha256));
        }
        if !matches!(
            o.kind,
            ffi::RESEARCH_ADD
                | ffi::RESEARCH_ABLATE_ZERO
                | ffi::RESEARCH_ABLATE_MEAN
                | ffi::RESEARCH_REPLACE
        ) {
            m.insert("experts".into(), json!(o.experts));
            if o.kind != ffi::RESEARCH_EXPERT_MASK {
                m.insert("weights".into(), json!(o.weights));
            }
        }
        Value::Object(m)
    };
    let (capture, interventions): (Vec<&Op>, Vec<&Op>) =
        ops.iter().partition(|o| o.kind == ffi::RESEARCH_CAPTURE);
    let mut plan = Map::new();
    plan.insert(
        "capture".into(),
        Value::Array(capture.into_iter().map(op_value).collect()),
    );
    plan.insert(
        "interventions".into(),
        Value::Array(interventions.into_iter().map(op_value).collect()),
    );
    if let Some(seed) = seed {
        plan.insert("seed".into(), json!(seed));
    }
    plan.insert("target".into(), json!(target));
    serde_json::to_string(&Value::Object(plan)).expect("plain JSON")
}

/// The plan seed of an unseeded request: the first 8 bytes (LE) of SHA-256(plan seed LE || session id UTF-8 ||
/// revision LE), the session being the request's CERE session (id, expected revision) or ("", its message count).
pub fn request_seed(plan_seed: u64, session_id: &str, revision: u64) -> u64 {
    let mut h = Sha256::new();
    h.update(plan_seed.to_le_bytes());
    h.update(session_id.as_bytes());
    h.update(revision.to_le_bytes());
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().expect("8 bytes"))
}

/// The prefix-cache namespace of an intervening plan's request: the base namespace mixed with the plan digest, so
/// no cached state crosses between edited and unedited computation (WL-11). Capture-only plans keep the base.
pub fn namespace(base: [u8; 16], plan_sha256: &[u8; 32]) -> [u8; 16] {
    let mut h = Sha256::new();
    h.update(b"cere-research-namespace-v1\0");
    h.update(base);
    h.update(plan_sha256);
    let d = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&d[..16]);
    out
}

// ---------------------------------------------------------------------------------------------------------
// The registry

/// A registered plan in its executor slot.
#[derive(Debug)]
pub struct LoadedPlan {
    pub id: String,
    pub slot: u32,
    pub plan: Plan,
}

/// A request running under its scope's plan: holds the plan (so it cannot be deleted) until the request ends.
#[derive(Debug)]
pub struct Applied {
    pub plan: Arc<LoadedPlan>,
    pub scope: String,
    pub request_id: String,
    /// The plan seed for an unseeded request ([`request_seed`]); `None` when the plan has none.
    pub seed: Option<u64>,
    pub research: Arc<Research>,
}

impl Applied {
    /// [`Research::reserve`] for this request's plan.
    pub fn reserve(
        &self,
        prompt_tokens: u64,
        max_tokens: u64,
        n: u32,
    ) -> Result<Option<Reservation>, ResearchError> {
        self.research
            .reserve(&self.plan.plan, prompt_tokens, max_tokens, n)
    }
}

/// A request's research run, admitted ([`Applied::reserve`]): its plan slot for `cere_session_set_research`
/// and its host-buffer reservation until the capture is finished ([`Run::end`]).
#[derive(Debug)]
pub struct Run {
    pub applied: Arc<Applied>,
    reservation: Mutex<Option<Reservation>>,
}

impl Run {
    pub fn new(applied: Arc<Applied>, reservation: Option<Reservation>) -> Run {
        Run {
            applied,
            reservation: Mutex::new(reservation),
        }
    }

    /// The plan's executor slot.
    pub fn slot(&self) -> u32 {
        self.applied.plan.slot
    }

    /// After the session was released: reads the capture (`capture` > 0), lists it until drained and frees the
    /// executor's copy. A failed generation (`ok` false) or read frees the capture and its reservation, listing
    /// nothing.
    pub fn end(&self, driver: &Driver, model: &Model, capture: u64, ok: bool) {
        let held = lock(&self.reservation).take();
        if capture == 0 {
            return;
        }
        let read = || -> Result<CaptureData, DriverError> {
            let view = driver.research_capture_read(model, capture)?;
            view.copy()
        };
        let data = if ok { Some(read()) } else { None };
        if let Err(e) = driver.research_capture_free(model, capture) {
            eprintln!(
                "cered research: capture_free rid={} capture={capture}: {e}",
                self.applied.request_id
            );
        }
        match (data, held) {
            (Some(Ok(data)), Some(held)) => self.applied.research.finish(&self.applied, held, data),
            (Some(Err(e)), _) => eprintln!(
                "cered research: capture_read rid={} capture={capture}: {e}",
                self.applied.request_id
            ),
            _ => {}
        }
    }
}

/// The host-buffer bytes a request holds from admission until its capture is drained (or the request fails).
#[derive(Debug)]
pub struct Reservation {
    research: Arc<Research>,
    bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut s = self.research.lock();
        s.reserved = s.reserved.saturating_sub(self.bytes);
    }
}

/// A finished capture, listed until drained.
struct Finished {
    seq: u64,
    request_id: String,
    scope: String,
    manifest: Value,
    file: Vec<u8>,
    /// Its bytes in the host buffer, released at drain.
    _held: Reservation,
}

#[derive(Default)]
struct State {
    plans: BTreeMap<String, Arc<LoadedPlan>>,
    bindings: BTreeMap<String, Arc<LoadedPlan>>,
    next_plan: u64,
    /// Host bytes of running and undrained captures.
    reserved: u64,
    captures: Vec<Finished>,
    next_seq: u64,
    /// request id -> finish sequence, for `after` (kept after the drain).
    finished: HashMap<String, u64>,
    /// request id -> plan sha256 hex, for the request's log line.
    applied: HashMap<String, String>,
}

/// The deployment's research registry (see the module doc).
pub struct Research {
    config: ResearchConfig,
    /// The deployment's context limit.
    context: u64,
    /// The deployment's scope ids (bindings name one).
    scopes: Vec<String>,
    /// Track WL-B's vector store.
    pub vectors: VectorStore,
    /// Serializes executor-side plan changes: register, delete and [`Research::attach`].
    ops: Mutex<()>,
    state: Mutex<State>,
}

impl std::fmt::Debug for Research {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Research")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn plan_load_error(e: DriverError) -> ResearchError {
    match &e {
        DriverError::Status {
            status: StatusError::Argument | StatusError::Unsupported,
            ..
        }
        | DriverError::Argument(_)
        | DriverError::Missing(_) => invalid(format!("the executor refused the plan: {e}")),
        _ => ResearchError::Internal(format!("loading the plan: {e}")),
    }
}

impl Research {
    pub fn new(config: ResearchConfig, context: u64, scopes: Vec<String>) -> Research {
        Research {
            vectors: VectorStore::new(config.max_vectors),
            config,
            context,
            scopes,
            ops: Mutex::default(),
            state: Mutex::default(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    pub fn config(&self) -> &ResearchConfig {
        &self.config
    }

    /// At a backend load, after the model was sealed: every registered plan is loaded again into its slot (a
    /// reloaded model). An error fails the load.
    pub fn attach(&self, driver: &Driver, model: &Model) -> Result<(), String> {
        let _ops = lock(&self.ops);
        let plans: Vec<Arc<LoadedPlan>> = self.lock().plans.values().cloned().collect();
        for p in plans {
            driver
                .research_plan_load(model, p.slot, &p.plan.ffi_ops(), p.plan.sha256)
                .map_err(|e| format!("research plan {}: {e}", p.id))?;
        }
        Ok(())
    }

    /// `POST /v1/cere/research/plans`: validates the plan, checks its registration bound and loads it into a free
    /// slot of the `current` backend.
    pub fn register(
        &self,
        body: &[u8],
        current: impl FnOnce() -> Option<Arc<NativeBackend>>,
    ) -> Result<Arc<LoadedPlan>, ResearchError> {
        let plan = parse_plan(body, self.context, &self.vectors)?;
        let bound = plan.capture_bound_bytes();
        let cap = self.config.max_capture_bytes_per_request;
        if bound > cap {
            return Err(ResearchError::Budget { bound, cap });
        }
        let _ops = lock(&self.ops);
        let slot = {
            let s = self.lock();
            (0..self.config.max_plans)
                .find(|slot| !s.plans.values().any(|p| p.slot == *slot))
                .ok_or(ResearchError::PlansFull(self.config.max_plans))?
        };
        let backend = current().ok_or(ResearchError::NotReady)?;
        backend
            .driver
            .research_plan_load(&backend.model, slot, &plan.ffi_ops(), plan.sha256)
            .map_err(plan_load_error)?;
        let mut s = self.lock();
        s.next_plan += 1;
        let loaded = Arc::new(LoadedPlan {
            id: format!("plan_{}", s.next_plan),
            slot,
            plan,
        });
        s.plans.insert(loaded.id.clone(), loaded.clone());
        eprintln!(
            "cered research: registered plan {} slot={} plan_sha256={} capture_bound_bytes={bound}",
            loaded.id,
            slot,
            loaded.plan.sha256_hex()
        );
        Ok(loaded)
    }

    /// `DELETE /v1/cere/research/plans/{id}`: unloads the plan; 409 while bound or a request runs under it.
    pub fn delete_plan(
        &self,
        id: &str,
        current: impl FnOnce() -> Option<Arc<NativeBackend>>,
    ) -> Result<(), ResearchError> {
        let _ops = lock(&self.ops);
        let plan = {
            let mut s = self.lock();
            let Some(plan) = s.plans.get(id) else {
                return Err(ResearchError::NotFound(
                    "research_plan_not_found",
                    format!("no plan {id:?}"),
                ));
            };
            let busy =
                || ResearchError::Conflict("research_plan_busy", format!("plan {id} is in use"));
            if s.bindings.values().any(|b| Arc::ptr_eq(b, plan)) {
                return Err(ResearchError::Conflict(
                    "research_plan_busy",
                    format!("plan {id} is bound to a scope"),
                ));
            }
            // Requests clone the plan only under this lock: one count is the registry's own.
            if Arc::strong_count(plan) > 1 {
                return Err(busy());
            }
            s.plans.remove(id).expect("present")
        };
        let reinsert = |e: ResearchError| {
            self.lock().plans.insert(id.to_owned(), plan.clone());
            Err(e)
        };
        let Some(backend) = current() else {
            return reinsert(ResearchError::NotReady);
        };
        match backend
            .driver
            .research_plan_unload(&backend.model, plan.slot)
        {
            Ok(()) => Ok(()),
            Err(e) if e.status() == Some(StatusError::Busy) => reinsert(ResearchError::Conflict(
                "research_plan_busy",
                format!("plan {id} is in use by a session"),
            )),
            Err(e) => reinsert(ResearchError::Internal(format!("unloading {id}: {e}"))),
        }
    }

    /// `POST /v1/cere/research/bindings {scope, plan_id}`.
    pub fn bind(&self, scope: &str, plan_id: &str) -> Result<(), ResearchError> {
        if !self.scopes.iter().any(|s| s == scope) {
            return Err(ResearchError::NotFound(
                "research_scope_not_found",
                format!("no scope {scope:?} in [[auth.scopes]]"),
            ));
        }
        let mut s = self.lock();
        let Some(plan) = s.plans.get(plan_id).cloned() else {
            return Err(ResearchError::NotFound(
                "research_plan_not_found",
                format!("no plan {plan_id:?}"),
            ));
        };
        if let Some(bound) = s.bindings.get(scope) {
            return Err(ResearchError::Conflict(
                "research_scope_bound",
                format!("scope {scope} is bound to plan {}", bound.id),
            ));
        }
        s.bindings.insert(scope.to_owned(), plan);
        Ok(())
    }

    /// `DELETE /v1/cere/research/bindings/{scope}`.
    pub fn unbind(&self, scope: &str) -> Result<(), ResearchError> {
        match self.lock().bindings.remove(scope) {
            Some(_) => Ok(()),
            None => Err(ResearchError::NotFound(
                "research_binding_not_found",
                format!("scope {scope:?} is not bound"),
            )),
        }
    }

    /// The plan request `request_id` of `scope` runs under, if its scope is bound; `session` is the request's CERE
    /// session (id, expected revision) or `None` with its message count for the plan seed.
    pub fn apply(
        self: &Arc<Self>,
        scope: &str,
        request_id: &str,
        session: Option<(&str, u64)>,
        message_count: u64,
    ) -> Option<Arc<Applied>> {
        let mut s = self.lock();
        let plan = s.bindings.get(scope)?.clone();
        s.applied
            .insert(request_id.to_owned(), plan.plan.sha256_hex());
        let (id, revision) = session.unwrap_or(("", message_count));
        Some(Arc::new(Applied {
            seed: plan.plan.seed.map(|seed| request_seed(seed, id, revision)),
            plan,
            scope: scope.to_owned(),
            request_id: request_id.to_owned(),
            research: self.clone(),
        }))
    }

    /// The plan sha256 request `request_id` ran under (once, for its log line).
    pub fn take_applied(&self, request_id: &str) -> Option<String> {
        self.lock().applied.remove(request_id)
    }

    /// Admission: reserves the request's bound in the host buffer (503 `research_buffer_full` when it does not
    /// fit). A plan without capture ops reserves nothing.
    pub fn reserve(
        self: &Arc<Self>,
        plan: &Plan,
        prompt_tokens: u64,
        max_tokens: u64,
        n: u32,
    ) -> Result<Option<Reservation>, ResearchError> {
        if n > 1 {
            return Err(ResearchError::Request(
                "n must be 1 under a research plan (one capture per request)".to_owned(),
            ));
        }
        if !plan.captures() {
            return Ok(None);
        }
        let bound = plan.request_bound_bytes(prompt_tokens, max_tokens);
        let mut s = self.lock();
        let free = self.config.host_buffer_bytes.saturating_sub(s.reserved);
        if bound > free {
            return Err(ResearchError::BufferFull { bound, free });
        }
        s.reserved += bound;
        Ok(Some(Reservation {
            research: self.clone(),
            bytes: bound,
        }))
    }

    /// The finished capture of `applied`: its manifest and safetensors file, listed until drained. The
    /// reservation shrinks to the capture's bytes.
    pub fn finish(&self, applied: &Applied, mut held: Reservation, data: CaptureData) {
        let (manifest, file) = build_capture(applied, &data);
        let bytes = file.len() as u64;
        let mut s = self.lock();
        if bytes < held.bytes {
            s.reserved -= held.bytes - bytes;
            held.bytes = bytes;
        }
        s.next_seq += 1;
        let seq = s.next_seq;
        s.finished.insert(applied.request_id.clone(), seq);
        s.captures.push(Finished {
            seq,
            request_id: applied.request_id.clone(),
            scope: applied.scope.clone(),
            manifest,
            file,
            _held: held,
        });
    }

    /// `GET /v1/cere/research/captures?scope=&after=`: the manifests of undrained captures (of `scope`, finished
    /// after request `after`'s capture), in finish order.
    pub fn list(
        &self,
        scope: Option<&str>,
        after: Option<&str>,
    ) -> Result<Vec<Value>, ResearchError> {
        let s = self.lock();
        let after = match after {
            None => 0,
            Some(rid) => *s.finished.get(rid).ok_or_else(|| {
                ResearchError::NotFound(
                    "research_capture_not_found",
                    format!("no capture of request {rid:?}"),
                )
            })?,
        };
        Ok(s.captures
            .iter()
            .filter(|c| c.seq > after && scope.is_none_or(|sc| c.scope == sc))
            .map(|c| c.manifest.clone())
            .collect())
    }

    /// `GET /v1/cere/research/captures/{request_id}`: the capture's safetensors file, once; its bytes are freed.
    pub fn drain(&self, request_id: &str) -> Result<Vec<u8>, ResearchError> {
        let finished = {
            let mut s = self.lock();
            let i = s
                .captures
                .iter()
                .position(|c| c.request_id == request_id)
                .ok_or_else(|| {
                    ResearchError::NotFound(
                        "research_capture_not_found",
                        format!("no undrained capture of request {request_id:?}"),
                    )
                })?;
            s.captures.remove(i)
        };
        // The reservation drops (outside the lock) with the record.
        Ok(finished.file)
    }
}

fn dtype_name(dtype: u32) -> &'static str {
    match dtype {
        I32 => "I32",
        F32 => "F32",
        _ => "BF16",
    }
}

/// The manifest and safetensors file of a capture (tensor names as the executor gives them, `<point>/L<layer>`;
/// shape `[rows, cols]`; metadata `request_id` and `plan_sha256`).
fn build_capture(applied: &Applied, data: &CaptureData) -> (Value, Vec<u8>) {
    let plan = &applied.plan;
    let mut header = Map::new();
    let mut tensors = Vec::with_capacity(data.tensors.len());
    let mut offset = 0usize;
    for t in &data.tensors {
        let end = offset + t.data.len();
        let shape = [t.positions.len() as u64, u64::from(t.cols)];
        header.insert(
            t.name.clone(),
            json!({"dtype": dtype_name(t.dtype), "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
        let point = t.name.rsplit_once("/L").map_or(t.name.as_str(), |(p, _)| p);
        tensors.push(json!({
            "name": t.name,
            "point": point,
            "layer": t.layer,
            "positions": t.positions,
            "dtype": dtype_name(t.dtype),
            "shape": shape,
            "sha256": hex(&Sha256::digest(&t.data)),
        }));
    }
    header.insert(
        "__metadata__".into(),
        json!({"request_id": applied.request_id, "plan_sha256": plan.plan.sha256_hex()}),
    );
    let mut head = serde_json::to_vec(&Value::Object(header)).expect("plain JSON");
    while head.len() % 8 != 0 {
        head.push(b' ');
    }
    let mut file = Vec::with_capacity(8 + head.len() + offset);
    file.extend_from_slice(&(head.len() as u64).to_le_bytes());
    file.extend_from_slice(&head);
    for t in &data.tensors {
        file.extend_from_slice(&t.data);
    }
    let manifest = json!({
        "request_id": applied.request_id,
        "scope": applied.scope,
        "plan_id": plan.id,
        "plan_sha256": plan.plan.sha256_hex(),
        "prefix_lookup": if data.prefix_lookup_skipped { "skipped" } else { "used" },
        "bytes": file.len(),
        "created": unix_seconds(),
        "tensors": tensors,
    });
    (manifest, file)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTEXT: u64 = 200_000;

    fn config() -> ResearchConfig {
        ResearchConfig {
            device_staging_bytes: 64 << 20,
            host_buffer_bytes: 1 << 20,
            max_capture_bytes_per_request: 256 << 10,
            max_plans: 2,
            max_vectors: 8,
        }
    }

    fn research() -> Arc<Research> {
        Arc::new(Research::new(
            config(),
            CONTEXT,
            vec!["wl-a".into(), "wl-b".into()],
        ))
    }

    /// A stored BF16 `[rows, cols]` vector of 1.0s; its id.
    fn vector(store: &VectorStore, name: &str, rows: u64, cols: u64) -> String {
        let bytes: Vec<u8> = (0..rows * cols)
            .flat_map(|_| 0x3f80u16.to_le_bytes())
            .collect();
        let header = json!({"v": {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [0, bytes.len()]}});
        let mut head = serde_json::to_vec(&header).unwrap();
        while head.len() % 8 != 0 {
            head.push(b' ');
        }
        let mut body = (head.len() as u64).to_le_bytes().to_vec();
        body.extend_from_slice(&head);
        body.extend_from_slice(&bytes);
        store.insert(name, &body).unwrap().vector_id
    }

    fn parse(body: Value) -> Result<Plan, ResearchError> {
        parse_plan(body.to_string().as_bytes(), CONTEXT, &VectorStore::new(8))
    }

    fn capture(op: Value) -> Value {
        json!({"capture": [op], "target": "coding_output_volume"})
    }

    fn refused(body: Value, needle: &str) {
        match parse(body.clone()) {
            Err(ResearchError::Invalid(m)) => assert!(m.contains(needle), "{body}: {m}"),
            other => panic!("{body}: {other:?}"),
        }
    }

    #[test]
    fn point_table_matches_the_header() {
        let p = Point::find("native.layers.L.gemm.attn_qkv.out").unwrap();
        assert_eq!(p.layers().count_ones(), 12);
        assert_eq!(p.layers() & 0b1000, 0b1000);
        assert_eq!(
            Point::find("native.layers.L.gemm.gdn_z.out")
                .unwrap()
                .layers()
                .count_ones(),
            36
        );
        assert_eq!(Point::find("native.logits").unwrap().layers(), 1 << 48);
        let row = |n: &str, k| Point::find(n).unwrap().row_bytes(k);
        assert_eq!(row("native.layers.L.attn_mix.in", 0), 20_480);
        assert_eq!(row("native.layers.L.attn_mix.out", 0), 5_120);
        assert_eq!(row("native.layers.L.gemm.moe_router.out", 0), 1_024);
        assert_eq!(row("native.layers.L.moe_route", 0), 80);
        assert_eq!(row("native.logits.topk", 5), 40);
        assert_eq!(row("native.logits", 0), 496_640);
        assert_eq!(row("native.layers.L.gemm.moe_shared_down.in", 0), 1_280);
        assert_eq!(POINTS.len(), 17);
    }

    #[test]
    fn plan_sha256_is_canonical() {
        let a = parse(json!({"target": "coding_output_volume", "capture": [{
            "point": "native.layers.L.gemm.attn_qkv.out", "layers": "all",
            "positions": {"prompt": [[0, 8]]}}]}))
        .unwrap();
        let all: Vec<u32> = (0..48).filter(|l| l % 4 == 3).collect();
        let b = parse(json!({"capture": [{"positions": {"prompt": [[0, 8]], "generated": false},
            "layers": all, "point": "native.layers.L.gemm.attn_qkv.out"}], "target": "coding_output_volume"}))
        .unwrap();
        assert_eq!(a.sha256, b.sha256);
        assert_eq!(a.sha256_hex(), hex(&Sha256::digest(a.canonical.as_bytes())));
        assert!(!a.canonical.contains(' '));
        assert!(a.canonical.starts_with(r#"{"capture":[{"layers":[3,7,"#));
        let c = parse(
            json!({"target": "coding_output_volume", "seed": 1, "capture": [{
            "point": "native.layers.L.gemm.attn_qkv.out", "layers": "all",
            "positions": {"prompt": [[0, 8]]}}]}),
        )
        .unwrap();
        assert_ne!(a.sha256, c.sha256);
        assert!(!a.intervenes() && a.captures());
    }

    #[test]
    fn target_outside_the_allowlist_is_refused() {
        refused(
            json!({"capture": [{"point": "native.final_mix.out", "positions": {"generated": 1}}],
                   "target": "something_else"}),
            "not a registered target",
        );
        refused(
            json!({"capture": [{"point": "native.final_mix.out", "positions": {"generated": 1}}]}),
            "target",
        );
    }

    #[test]
    fn body_refusals_name_the_rule() {
        let ok_pos = json!({"generated": 1});
        let cases = [
            (json!({"target": "coding_output_volume"}), "1..=128 ops"),
            (
                json!({"target": "coding_output_volume", "extra": 1, "capture": []}),
                "unknown field",
            ),
            (
                capture(json!({"point": "native.nope", "layers": [0], "positions": ok_pos})),
                "not a research point",
            ),
            (
                capture(json!({"point": "native.layers.L.attn_mix.out", "positions": ok_pos})),
                "layers is required",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.gemm.attn_qkv.out", "layers": [0], "positions": ok_pos}),
                ),
                "has no layer 0",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [48], "positions": ok_pos}),
                ),
                "has no layer 48",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [1, 1], "positions": ok_pos}),
                ),
                "listed twice",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [], "positions": ok_pos}),
                ),
                "layers is empty",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": "some", "positions": ok_pos}),
                ),
                "\"all\"",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [0],
                            "positions": {"prompt": [[4, 4]]}}),
                ),
                "end must exceed first",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [0],
                            "positions": {"prompt": [[4, 8], [6, 9]]}}),
                ),
                "sorted and disjoint",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [0],
                            "positions": {"prompt": (0..65).map(|i| [2 * i, 2 * i + 1]).collect::<Vec<_>>()}}),
                ),
                "at most 64",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [0], "positions": {}}),
                ),
                "selects no position",
            ),
            (
                capture(
                    json!({"point": "native.layers.L.attn_mix.out", "layers": [0],
                            "positions": {"generated": 0}}),
                ),
                "at least 1",
            ),
            (
                capture(
                    json!({"point": "native.logits", "positions": {"prompt": [[0, 1]], "generated": 1}}),
                ),
                "generated positions only",
            ),
            (
                capture(json!({"point": "native.logits", "positions": {"generated": 9}})),
                "N <= 8",
            ),
            (
                capture(json!({"point": "native.logits", "positions": {"generated": true}})),
                "N <= 8",
            ),
            (
                capture(json!({"point": "native.logits.topk", "positions": ok_pos})),
                "top_k in [1, 64]",
            ),
            (
                capture(json!({"point": "native.logits.topk", "top_k": 65, "positions": ok_pos})),
                "top_k in [1, 64]",
            ),
            (
                capture(json!({"point": "native.final_mix.out", "top_k": 4, "positions": ok_pos})),
                "native.logits.topk only",
            ),
            (
                capture(
                    json!({"point": "native.final_mix.out", "kind": "add", "positions": ok_pos}),
                ),
                "takes no kind",
            ),
            (
                capture(
                    json!({"point": "native.final_mix.out", "alpha": 1.0, "positions": ok_pos}),
                ),
                "takes no vector_id",
            ),
        ];
        for (body, needle) in cases {
            refused(body, needle);
        }
        assert!(parse(capture(
            json!({"point": "native.logits", "layers": [48], "positions": {"generated": 8}})
        ))
        .is_ok());
    }

    fn intervention(store: &VectorStore, op: Value) -> Result<Plan, ResearchError> {
        let body = json!({"interventions": [op], "target": "coding_output_volume"});
        parse_plan(body.to_string().as_bytes(), CONTEXT, store)
    }

    #[test]
    fn intervention_refusals_follow_check_edit() {
        let store = VectorStore::new(8);
        let row = vector(&store, "row", 1, 2_560);
        let narrow = vector(&store, "narrow", 1, 512);
        let rows3 = vector(&store, "rows3", 3, 2_560);
        let pos = json!({"generated": true});
        let at = "native.layers.L.attn_mix.out";
        let router = "native.layers.L.gemm.moe_router.out";
        let route = "native.layers.L.moe_route";
        let cases = [
            (
                json!({"point": at, "layers": [0], "positions": pos}),
                "kind is required",
            ),
            (
                json!({"kind": "scale", "point": at, "layers": [0], "positions": pos}),
                "not an intervention kind",
            ),
            (
                json!({"kind": "add", "point": "native.layers.L.attn_mix.in", "layers": [0], "positions": pos,
                    "vector_id": row, "alpha": 1.0}),
                "does not apply",
            ),
            (
                json!({"kind": "add", "point": "native.logits", "positions": {"generated": 1}, "vector_id": row,
                    "alpha": 1.0}),
                "does not apply",
            ),
            (
                json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "vector_id": row}),
                "alpha is required",
            ),
            (
                json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "vector_id": row, "alpha": 1e39}),
                "finite f32",
            ),
            (
                json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "alpha": 1.0}),
                "vector_id is required",
            ),
            (
                json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "alpha": 1.0,
                    "vector_id": "vec_nope"}),
                "not a stored vector",
            ),
            (
                json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "alpha": 1.0,
                    "vector_id": narrow}),
                "512 columns",
            ),
            (
                json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "alpha": 1.0,
                    "vector_id": rows3}),
                "3 rows",
            ),
            (
                json!({"kind": "ablate_mean", "point": at, "layers": [0], "positions": pos, "alpha": 1.0,
                    "vector_id": row}),
                "alpha is for add only",
            ),
            (
                json!({"kind": "ablate_zero", "point": at, "layers": [0], "positions": pos, "vector_id": row}),
                "takes no vector_id",
            ),
            (
                json!({"kind": "ablate_zero", "point": at, "layers": [0], "positions": pos, "cols": [5, 5]}),
                "cols [5, 5)",
            ),
            (
                json!({"kind": "ablate_zero", "point": at, "layers": [0], "positions": pos, "cols": [0, 2_561]}),
                "cols [0, 2561)",
            ),
            (
                json!({"kind": "ablate_zero", "point": at, "layers": [0], "positions": pos, "experts": [1]}),
                "no experts",
            ),
            (
                json!({"kind": "replace", "point": at, "layers": [0], "positions": {"prompt": [[0, 2]]},
                    "vector_id": rows3}),
                "one per planned position",
            ),
            (
                json!({"kind": "expert_force", "point": router, "layers": [0], "positions": pos,
                    "experts": [1], "weights": [0.5]}),
                "does not apply",
            ),
            (
                json!({"kind": "expert_bias", "point": route, "layers": [0], "positions": pos,
                    "experts": [1], "weights": [0.5]}),
                "does not apply",
            ),
            (
                json!({"kind": "expert_bias", "point": router, "layers": [0], "positions": pos,
                    "experts": [1, 1], "weights": [0.5, 0.5]}),
                "distinct",
            ),
            (
                json!({"kind": "expert_bias", "point": router, "layers": [0], "positions": pos,
                    "experts": [512], "weights": [0.5]}),
                "distinct",
            ),
            (
                json!({"kind": "expert_bias", "point": router, "layers": [0], "positions": pos,
                    "experts": [1, 2], "weights": [0.5]}),
                "one finite weight",
            ),
            (
                json!({"kind": "expert_bias", "point": router, "layers": [0], "positions": pos,
                    "experts": [1], "weights": [0.5], "vector_id": row}),
                "no vector_id or cols",
            ),
            (
                json!({"kind": "expert_mask", "point": router, "layers": [0], "positions": pos,
                    "experts": (0..503).collect::<Vec<_>>()}),
                "at least 10 experts unmasked",
            ),
            (
                json!({"kind": "expert_mask", "point": router, "layers": [0], "positions": pos,
                    "experts": [1], "weights": [1.0]}),
                "takes no weights",
            ),
            (
                json!({"kind": "expert_force", "point": route, "layers": [0], "positions": pos,
                    "experts": [], "weights": []}),
                "1..=10 experts",
            ),
            (
                json!({"kind": "expert_force", "point": route, "layers": [0], "positions": pos,
                    "experts": [1, 2], "weights": [0.7, 0.7]}),
                "summing to at most 1",
            ),
            (
                json!({"kind": "expert_force", "point": route, "layers": [0], "positions": pos,
                    "experts": [1], "weights": [-0.1]}),
                "1..=10 experts",
            ),
            (
                json!({"kind": "expert_force", "point": route, "layers": [0], "positions": pos,
                    "experts": (0..11).collect::<Vec<_>>(), "weights": vec![0.01; 11]}),
                "1..=10 experts",
            ),
        ];
        for (op, needle) in cases {
            match intervention(&store, op.clone()) {
                Err(ResearchError::Invalid(m)) => assert!(m.contains(needle), "{op}: {m}"),
                other => panic!("{op}: {other:?}"),
            }
        }
        // Accepted forms: one per kind.
        let ok = [
            json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "vector_id": row, "alpha": 2.0}),
            json!({"kind": "ablate_mean", "point": at, "layers": "all", "positions": pos, "vector_id": row,
                   "cols": [0, 100]}),
            json!({"kind": "ablate_zero", "point": "native.final_mix.out", "positions": pos}),
            json!({"kind": "replace", "point": at, "layers": [0], "positions": {"prompt": [[0, 2]], "generated": 1},
                   "vector_id": rows3}),
            json!({"kind": "expert_bias", "point": router, "layers": [0], "positions": pos,
                   "experts": [1], "weights": [-3.0]}),
            json!({"kind": "expert_mask", "point": router, "layers": [0], "positions": pos, "experts": [1, 2]}),
            json!({"kind": "expert_force", "point": route, "layers": [0], "positions": pos,
                   "experts": [1, 2], "weights": [0.5, 0.5]}),
            // WL-16's identity plans: an empty expert set (count 0) is a valid no-op.
            json!({"kind": "expert_mask", "point": router, "layers": "all", "positions": pos, "experts": []}),
            json!({"kind": "expert_bias", "point": router, "layers": [0], "positions": pos,
                   "experts": [], "weights": []}),
        ];
        for op in ok {
            let plan = intervention(&store, op.clone()).unwrap_or_else(|e| panic!("{op}: {e}"));
            assert!(plan.intervenes() && !plan.captures(), "{op}");
            assert_eq!(plan.capture_bound_bytes(), 0);
            let ops = plan.ffi_ops();
            assert_eq!(ops[0].kind, kind_of(op["kind"].as_str().unwrap()).unwrap());
            // research/plan.cpp `load_op` refuses a nonzero top_k on every point but native.logits.topk.
            assert_eq!(ops[0].top_k, 0, "{op}");
        }
        // The plan holds its vector: WL-B's store refuses the delete while it lives.
        let plan = intervention(
            &store,
            json!({"kind": "add", "point": at, "layers": [0], "positions": pos, "vector_id": row, "alpha": 2.0}),
        )
        .unwrap();
        assert!(plan.canonical.contains(&row));
        assert!(store.remove(&row).is_err());
        drop(plan);
        store.remove(&row).unwrap();
    }

    #[test]
    fn registration_and_request_bounds() {
        let plan = parse(capture(
            json!({"point": "native.layers.L.attn_mix.out", "layers": [0, 5],
            "positions": {"prompt": [[0, 128]], "generated": 16}}),
        ))
        .unwrap();
        assert_eq!(plan.capture_bound_bytes(), (128 + 16) * 2 * 5_120);
        // P = 100, M = 50: prompt rows [0, 100), generated [99, 115): 100 + 16 - 1 (position 99 once).
        assert_eq!(plan.request_bound_bytes(100, 50), (100 + 15) * 2 * 5_120);
        // M = 4 caps N; P = 200 puts the generated rows past the range.
        assert_eq!(plan.request_bound_bytes(200, 4), (128 + 4) * 2 * 5_120);
        assert_eq!(plan.request_bound_bytes(0, 0), 0);
        let every = parse(capture(json!({"point": "native.logits.topk", "top_k": 4,
            "positions": {"generated": true}})))
        .unwrap();
        assert_eq!(every.capture_bound_bytes(), CONTEXT * 32);
        assert_eq!(every.request_bound_bytes(10, 7), 7 * 32);
        let route = parse(capture(
            json!({"point": "native.layers.L.moe_route", "layers": "all",
            "positions": {"prompt": [[0, 1]]}}),
        ))
        .unwrap();
        assert_eq!(route.capture_bound_bytes(), 48 * 80);
    }

    fn insert_plan(r: &Research, slot: u32, body: Value) -> Arc<LoadedPlan> {
        let plan = parse(body).unwrap();
        let loaded = Arc::new(LoadedPlan {
            id: format!("plan_t{slot}"),
            slot,
            plan,
        });
        r.lock().plans.insert(loaded.id.clone(), loaded.clone());
        loaded
    }

    fn small() -> Value {
        capture(json!({"point": "native.final_mix.out", "positions": {"generated": 4}}))
    }

    #[test]
    fn register_refuses_budget_and_full_before_the_executor() {
        let r = research();
        let big = capture(
            json!({"point": "native.layers.L.attn_mix.in", "layers": "all",
            "positions": {"generated": true}}),
        );
        let err = r.register(big.to_string().as_bytes(), || None).unwrap_err();
        assert_eq!(err.status_code(), (400, "research_budget_exceeded"));
        assert!(matches!(err, ResearchError::Budget { cap, .. } if cap == 256 << 10));
        let bad = r.register(b"{", || None).unwrap_err();
        assert_eq!(bad.status_code(), (400, "invalid_research_plan"));
        let err = r
            .register(small().to_string().as_bytes(), || None)
            .unwrap_err();
        assert_eq!(err, ResearchError::NotReady);
        insert_plan(&r, 0, small());
        insert_plan(&r, 1, small());
        let err = r
            .register(small().to_string().as_bytes(), || None)
            .unwrap_err();
        assert_eq!(err.status_code(), (507, "research_plans_full"));
    }

    #[test]
    fn bindings_refusals() {
        let r = research();
        let plan = insert_plan(&r, 0, small());
        let code = |e: ResearchError| e.status_code();
        assert_eq!(
            code(r.bind("nope", &plan.id).unwrap_err()),
            (404, "research_scope_not_found")
        );
        assert_eq!(
            code(r.bind("wl-a", "plan_x").unwrap_err()),
            (404, "research_plan_not_found")
        );
        r.bind("wl-a", &plan.id).unwrap();
        assert_eq!(
            code(r.bind("wl-a", &plan.id).unwrap_err()),
            (409, "research_scope_bound")
        );
        assert_eq!(
            code(r.delete_plan(&plan.id, || None).unwrap_err()),
            (409, "research_plan_busy")
        );
        assert_eq!(
            code(r.unbind("wl-b").unwrap_err()),
            (404, "research_binding_not_found")
        );
        assert!(r.apply("wl-b", "rid_1", None, 1).is_none());
        let applied = r.apply("wl-a", "rid_2", None, 3).unwrap();
        assert_eq!(r.take_applied("rid_2"), Some(plan.plan.sha256_hex()));
        assert_eq!(r.take_applied("rid_2"), None);
        r.unbind("wl-a").unwrap();
        // A running request still holds the plan.
        assert_eq!(
            code(r.delete_plan(&plan.id, || None).unwrap_err()),
            (409, "research_plan_busy")
        );
        drop(applied);
        drop(plan);
        // Free of holders, the executor unload needs the backend.
        assert_eq!(
            r.delete_plan("plan_t0", || None).unwrap_err(),
            ResearchError::NotReady
        );
        assert!(r.lock().plans.contains_key("plan_t0"));
        assert_eq!(
            code(r.delete_plan("plan_x", || None).unwrap_err()),
            (404, "research_plan_not_found")
        );
    }

    #[test]
    fn admission_reserves_the_request_bound_until_drain() {
        let r = research();
        let plan = insert_plan(
            &r,
            0,
            capture(json!({"point": "native.final_mix.out",
            "positions": {"generated": true}})),
        );
        r.bind("wl-a", &plan.id).unwrap();
        let err = r.reserve(&plan.plan, 10, 10, 2).unwrap_err();
        assert_eq!(err.status_code(), (400, "invalid_request"));
        // 1 MiB buffer, 5,120 B per row: 204 rows fit.
        let held = r.reserve(&plan.plan, 10, 150, 1).unwrap().unwrap();
        let err = r.reserve(&plan.plan, 10, 150, 1).unwrap_err();
        assert_eq!(err.status_code(), (503, "research_buffer_full"));
        let applied = r.apply("wl-a", "rid_7", Some(("sess", 4)), 1).unwrap();
        let data = CaptureData {
            tensors: vec![crate::native::driver::CapturedTensor {
                name: "native.final_mix.out/L48".into(),
                op: 0,
                layer: 48,
                dtype: BF16,
                cols: 2,
                positions: vec![9, 10],
                data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }],
            bytes: 8,
            prefix_lookup_skipped: true,
            plan_sha256: plan.plan.sha256,
        };
        r.finish(&applied, held, data);
        // The reservation shrank to the file: another request fits again.
        drop(r.reserve(&plan.plan, 10, 150, 1).unwrap());
        let list = r.list(Some("wl-a"), None).unwrap();
        assert_eq!(list.len(), 1);
        let m = &list[0];
        assert_eq!(m["request_id"], "rid_7");
        assert_eq!(m["plan_sha256"], plan.plan.sha256_hex());
        assert_eq!(m["prefix_lookup"], "skipped");
        assert_eq!(m["tensors"][0]["point"], "native.final_mix.out");
        assert_eq!(m["tensors"][0]["shape"], json!([2, 2]));
        assert_eq!(m["tensors"][0]["positions"], json!([9, 10]));
        assert!(r.list(Some("wl-b"), None).unwrap().is_empty());
        assert!(r.list(None, Some("rid_7")).unwrap().is_empty());
        assert_eq!(
            r.list(None, Some("rid_0")).unwrap_err().status_code().0,
            404
        );
        let file = r.drain("rid_7").unwrap();
        assert_eq!(m["bytes"], file.len());
        let head = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&file[8..8 + head]).unwrap();
        assert_eq!(header["native.final_mix.out/L48"]["dtype"], "BF16");
        assert_eq!(header["__metadata__"]["request_id"], "rid_7");
        assert_eq!(&file[8 + head..], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            r.drain("rid_7").unwrap_err().status_code(),
            (404, "research_capture_not_found")
        );
        assert_eq!(r.lock().reserved, 0);
        // A capture-free plan reserves nothing.
        let edit = parse(
            json!({"interventions": [{"kind": "ablate_zero", "point": "native.final_mix.out",
            "positions": {"generated": true}}], "target": "coding_output_volume"}),
        )
        .unwrap();
        assert!(r.reserve(&edit, 10, 1 << 30, 1).unwrap().is_none());
    }

    #[test]
    fn seed_and_namespace() {
        let a = request_seed(7, "sess", 3);
        assert_eq!(a, request_seed(7, "sess", 3));
        assert_ne!(a, request_seed(7, "sess", 4));
        assert_ne!(a, request_seed(8, "sess", 3));
        assert_ne!(request_seed(7, "", 3), a);
        let mut h = Sha256::new();
        h.update(7u64.to_le_bytes());
        h.update(b"sess");
        h.update(3u64.to_le_bytes());
        assert_eq!(a.to_le_bytes(), h.finalize()[..8]);
        let base = [1u8; 16];
        assert_ne!(namespace(base, &[0; 32]), base);
        assert_ne!(namespace(base, &[0; 32]), namespace(base, &[1; 32]));
    }

    /// WL-11: an intervening plan's requests get a namespace of their own, different from the base and from
    /// another intervening plan's; a capture-only plan's requests keep the base namespace.
    #[test]
    fn intervening_plans_get_their_own_namespace_and_capture_only_plans_keep_the_base() {
        let base = [3u8; 16];
        let edit = |point: &str| {
            parse(
                json!({"interventions": [{"kind": "ablate_zero", "point": point, "layers": [0],
                "positions": {"generated": true}}], "target": "coding_output_volume"}),
            )
            .unwrap()
        };
        let a = edit("native.layers.L.attn_mix.out");
        let b = edit("native.layers.L.mlp_mix.out");
        assert!(a.intervenes() && b.intervenes());
        assert_ne!(a.cache_namespace(base), base);
        assert_eq!(a.cache_namespace(base), namespace(base, &a.sha256));
        assert_ne!(a.cache_namespace(base), b.cache_namespace(base));
        assert_ne!(a.cache_namespace(base), a.cache_namespace([4u8; 16]));
        let capture_only = parse(capture(json!({"point": "native.final_mix.out",
            "positions": {"prompt": [[0, 4]], "generated": 2}})))
        .unwrap();
        assert!(!capture_only.intervenes());
        assert_eq!(capture_only.cache_namespace(base), base);
    }

    /// The shipped live plans (scripts/research/plans, WL-12 steering and WL-16's no-op plan) parse once wl_live.py
    /// substitutes `@NAME` with the uploaded vector's id.
    #[test]
    fn the_shipped_live_plans_parse() {
        let store = VectorStore::new(8);
        let steer = vector(&store, "steer", 1, 2_560);
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/research/plans");
        for name in ["wl12-steer.json", "wl16-noop.json"] {
            let text = std::fs::read_to_string(dir.join(name)).unwrap();
            let body = text.replace("@steer", &steer);
            let plan = parse_plan(body.as_bytes(), CONTEXT, &store)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(plan.intervenes() && !plan.captures(), "{name}");
        }
    }
}
