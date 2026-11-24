//! The native executor path (spec m22 R1-R3, R5; m24 D8-D11; m28 D7): the ABI v3 projection and library loader
//! ([`ffi`]), the driver thread that owns the executor handle ([`driver`]), the M2.1-driven upload sink ([`upload`]),
//! the request path's tokenizer boundary: the Rust detokenizer ([`detok`]), the Qwen3 reasoning split
//! ([`reasoning`]), the qwen3_coder tool-call detector ([`tool_parser`]), the stop-string matcher ([`stop`]), the
//! tokenizer and grammar sidecar clients ([`sidecar`], masks decoded by [`base64`]), the request's sampling
//! ([`sampling`]), the request path itself ([`serve`]), and M2.11: the snapshot size the config bounds
//! ([`snapshot`]), the U1 preemption bookkeeping ([`preempt`]) and the U3 state export ([`export`]).
pub mod affinity;
pub mod base64;
pub mod detok;
pub mod driver;
pub mod export;
pub mod ffi;
pub mod inkling;
pub mod loop_guard;
pub mod lora;
pub mod park;
pub mod preempt;
pub mod reasoning;
pub mod research;
pub mod research_vectors;
pub mod sampling;
pub(crate) mod serve;
pub mod sidecar;
pub mod snapshot;
pub mod stop;
pub mod teleport;
pub(crate) mod teleport_http;
pub mod tool_parser;
pub mod upload;

use cere_core::capabilities::ExecutionCapabilities;

use crate::native::sidecar::{GrammarSidecar, GRAMMAR_COMPILE_DEADLINE};

/// The structured outputs a grammar constrains (spec m28 D10): a profile advertising one needs the grammar process
/// and an executor with `native_token_mask`.
pub const GRAMMAR_OUTPUTS: &[&str] = &["json_schema", "regex", "ebnf", "tool_calls_constrained"];

/// The profile advertises a grammar output, in `structured_outputs` or in a validated combination.
pub fn advertises_grammar(profile: &ExecutionCapabilities) -> bool {
    profile
        .structured_outputs
        .iter()
        .chain(profile.validated_combinations.iter().map(|c| &c.output))
        .any(|output| GRAMMAR_OUTPUTS.contains(&output.as_str()))
}

/// The grammar process of `profile` (spec m28 D7): `None` when it advertises no grammar output, else `<sidecar>
/// --role grammar` checked against the xgrammar version of the profile's dialect (`json-schema-2020-12/xgrammar-0.2.1`
/// pins `0.2.1`). `Err` is the readiness problem's detail.
pub fn spawn_grammar(
    profile: &ExecutionCapabilities,
    sidecar: &[String],
) -> Result<Option<GrammarSidecar>, String> {
    if !advertises_grammar(profile) {
        return Ok(None);
    }
    let xgrammar = profile
        .json_schema_dialect
        .as_deref()
        .and_then(|dialect| dialect.rsplit_once("/xgrammar-"))
        .map(|(_, version)| version)
        .ok_or_else(|| {
            "the profile advertises a grammar output without an xgrammar json_schema_dialect"
                .to_owned()
        })?;
    GrammarSidecar::spawn(sidecar, xgrammar, GRAMMAR_COMPILE_DEADLINE)
        .map(Some)
        .map_err(|e| e.to_string())
}
