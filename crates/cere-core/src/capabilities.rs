use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCapabilities {
    pub schema_version: String,
    pub plugin: String,
    pub profile_id: String,
    pub artifact_sha256: String,
    pub execution_mode: String,
    pub evidence: String,
    pub operations: Vec<String>,
    pub native_decode_batch: bool,
    pub decode_batch_min: u64,
    pub decode_batch_max: u64,
    pub max_context_tokens: u64,
    pub native_mixed_prefill_decode: bool,
    pub native_speculative_batch: bool,
    pub native_moe_grouping: bool,
    pub native_state_snapshot: bool,
    pub structured_outputs: Vec<String>,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub json_schema_dialect: Option<String>,
    pub json_schema_keywords: Vec<String>,
    pub numerical_modes: Vec<String>,
    pub checkpoint_mode: String,
    pub modalities: Vec<String>,
    /// Largest `top_logprobs` a chat request may ask for (0..=20); absent = logprobs refused (M2.4 D12).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_top_logprobs: Option<u32>,
    /// M2.7 D8/D11: the request sampling fields this profile serves, from [`SAMPLING_PARAMS`]; absent = each of
    /// them is refused (400 `unsupported_parameter`). A native profile with a sampled row must declare it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<Vec<String>>,
    /// Per-request `chat_template_kwargs.enable_thinking` / `separate_reasoning` / `reasoning_effort` accepted;
    /// absent = refused (M2.12 D2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_toggle: Option<bool>,
    /// Accepted token controls, a subset of `stop_token_ids`, `ignore_eos`, `min_tokens`; absent = all refused
    /// (M2.12 D2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_controls: Option<Vec<String>>,
    /// The model's vocabulary size: every `stop_token_ids` item must be below it (M2.12 D5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<u32>,
    /// BC1 leftover (batch-concurrency design note "Prefill selection signature"): the SHA-256 (64 lowercase hex)
    /// the native executor reports for the sealed model's prefill kernel selection (`cere_model_prefill_signature`);
    /// absent = not pinned. Declared, readiness reports `capabilities.prefill_signature` when the loaded model's
    /// differs or the library reports none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_signature: Option<String>,
    pub notes: String,
    pub validated_combinations: Vec<ValidatedCombination>,
}

/// The request sampling fields a profile may declare in `sampling_params` (M2.7 D8).
pub const SAMPLING_PARAMS: [&str; 7] = [
    "top_k",
    "min_p",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "logit_bias",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidatedCombination {
    pub operation: String,
    pub modality: String,
    pub output: String,
    pub numerical_mode: String,
    pub sampler_mode: String,
    pub row_counts: Vec<u64>,
    pub max_context_tokens: u64,
}

/// The xgrammar JSON Schema dialect (M2.8 D5): Draft 2020-12 as xgrammar 0.2.1 (the pinned SGLang image's) enforces
/// it. The native path requires it; see [`json_schema_dialect_keywords`] for every registered dialect.
pub const JSON_SCHEMA_DIALECT: &str = "json-schema-2020-12/xgrammar-0.2.1";

/// M2.8 D5: the keywords a profile's `json_schema_keywords` may list - the enforcement candidates plus the
/// annotation-only `title`, `description` and `default`. `oneOf` is never one (xgrammar cannot enforce
/// exclusivity).
pub const JSON_SCHEMA_KEYWORD_CANDIDATES: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "enum",
    "const",
    "anyOf",
    "allOf",
    "$ref",
    "$defs",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "title",
    "description",
    "default",
];

/// AmdAmd PAR-3 (amd-amd design note): Draft 2020-12 as llama.cpp `162e2bb2e`'s `json_schema_to_grammar` enforces
/// it, for llama-server behind `cere-llamacpp-shim`.
pub const JSON_SCHEMA_DIALECT_LLAMACPP: &str = "json-schema-2020-12/llamacpp-162e2bb2e";

/// The keywords that pass CERE's enforcement variants on that llama.cpp
/// (an internal benchmark report), plus the annotations. `format` is absent: cered's
/// format list is xgrammar's, and llama.cpp enforces 4 of its 13 values. `pattern` and `prefixItems` fail there.
pub const JSON_SCHEMA_KEYWORDS_LLAMACPP: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "allOf",
    "$ref",
    "$defs",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "title",
    "description",
    "default",
];

/// The keywords a profile advertising `dialect` may list; `None` for an unregistered dialect.
pub fn json_schema_dialect_keywords(dialect: &str) -> Option<&'static [&'static str]> {
    match dialect {
        JSON_SCHEMA_DIALECT => Some(JSON_SCHEMA_KEYWORD_CANDIDATES),
        JSON_SCHEMA_DIALECT_LLAMACPP => Some(JSON_SCHEMA_KEYWORDS_LLAMACPP),
        _ => None,
    }
}

/// M2.8 D10: the outputs a grammar constrains (decode only, never `verify`).
pub const GRAMMAR_OUTPUTS: &[&str] = &["json_schema", "regex", "ebnf", "tool_calls_constrained"];

impl ExecutionCapabilities {
    /// Additional semantics AFTER strict JSON Schema validation; not a probe or admission authority.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != "2"
            || self.decode_batch_min == 0
            || self.decode_batch_min > self.decode_batch_max
            || self.max_context_tokens == 0
        {
            return Err("invalid version or batch/context bounds");
        }
        let native = self.native_decode_batch
            || self.native_mixed_prefill_decode
            || self.native_speculative_batch
            || self.native_moe_grouping
            || self.native_state_snapshot;
        if (self.execution_mode == "serial" || self.execution_mode == "unsupported") && native {
            return Err("non-native mode claims native features");
        }
        if self.execution_mode == "serial" && self.decode_batch_max != 1 {
            return Err("serial native row capacity exceeds one");
        }
        if self.execution_mode == "unsupported" && !self.operations.is_empty() {
            return Err("unsupported profile advertises operations");
        }
        let has_op = |name: &str| self.operations.iter().any(|op| op == name);
        if self.native_decode_batch && !has_op("decode") {
            return Err("native decode lacks decode operation");
        }
        if self.native_speculative_batch && !has_op("verify") {
            return Err("speculation lacks verification");
        }
        if self.checkpoint_mode == "exact" && !self.native_state_snapshot {
            return Err("exact checkpoint lacks snapshot support");
        }
        if self.structured_outputs.iter().any(|o| o == "json_schema") {
            let Some(candidates) = self
                .json_schema_dialect
                .as_deref()
                .and_then(json_schema_dialect_keywords)
            else {
                return Err("json_schema output lacks the D5 dialect and keywords");
            };
            if self.json_schema_keywords.is_empty() {
                return Err("json_schema output lacks the D5 dialect and keywords");
            }
            if self
                .json_schema_keywords
                .iter()
                .any(|k| !candidates.contains(&k.as_str()))
            {
                return Err("json_schema keyword outside the D5 candidates");
            }
        } else if self.json_schema_dialect.is_some() || !self.json_schema_keywords.is_empty() {
            return Err("schema dialect/keywords advertised without json_schema output");
        }
        if !matches!(
            self.execution_mode.as_str(),
            "native" | "backend_managed" | "serial" | "unsupported"
        ) {
            return Err("unknown execution mode");
        }
        // M2.7 D11: a native sampled row needs cered's sampler, which a native profile declares with
        // `sampling_params` (the start check against the library's `native_sampler` is dynamic).
        if self.execution_mode == "native"
            && self.sampling_params.as_ref().is_none_or(Vec::is_empty)
            && self
                .validated_combinations
                .iter()
                .any(|c| c.sampler_mode == "sampled")
        {
            return Err("native sampled combination lacks sampling_params");
        }
        for (index, combination) in self.validated_combinations.iter().enumerate() {
            if self.validated_combinations[..index].contains(combination) {
                return Err("duplicate capability combination");
            }
            if !self.operations.contains(&combination.operation)
                || !self.modalities.contains(&combination.modality)
                || !self.structured_outputs.contains(&combination.output)
                || !self.numerical_modes.contains(&combination.numerical_mode)
                || combination.row_counts.is_empty()
                || combination.max_context_tokens == 0
                || combination.max_context_tokens > self.max_context_tokens
                || combination
                    .row_counts
                    .iter()
                    .any(|n| *n < self.decode_batch_min || *n > self.decode_batch_max)
            {
                return Err("combination exceeds declared capabilities");
            }
            // M2.8 D10/I3: constrained tickets never speculate, so no grammar output pairs with `verify`.
            if combination.operation == "verify"
                && GRAMMAR_OUTPUTS.contains(&combination.output.as_str())
            {
                return Err("grammar output combined with verify");
            }
            // M2.4 D13: tool calls are text decoding plus a parser, so each one needs its text twin. M2.13 D5: an
            // image decode combination needs the same text twin.
            let twin = |output: &str| {
                combination.operation == "decode"
                    && combination.output == output
                    && !self.validated_combinations.iter().any(|text| {
                        text.operation == "decode"
                            && text.modality == "text"
                            && text.output == "text"
                            && text.sampler_mode == combination.sampler_mode
                            && text.row_counts == combination.row_counts
                            && text.max_context_tokens == combination.max_context_tokens
                    })
            };
            if twin("tool_calls") {
                return Err(
                    "tool_calls decode combination lacks the matching text decode combination",
                );
            }
            if combination.modality == "image" && twin("text") {
                return Err("image decode combination lacks the matching text decode combination");
            }
        }
        // M2.13 D5: `modalities` lists `image` iff an image decode combination exists (the converse is above).
        if self.modalities.iter().any(|m| m == "image")
            && !self
                .validated_combinations
                .iter()
                .any(|c| c.operation == "decode" && c.modality == "image")
        {
            return Err("modalities lists image without an image decode combination");
        }
        Ok(())
    }
}
