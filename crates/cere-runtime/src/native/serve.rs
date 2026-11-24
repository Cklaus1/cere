//! The native request path (spec m22 R5, "Native path contract"): the prompt is encoded once by the tokenizer
//! sidecar and checked against the context limit, then one fresh session runs PREFILL(prompt, 1) and, unless that
//! token ended the output, DECODE(max_tokens - 1), both with the model's stop IDs. The committed tokens are
//! detokenized incrementally, split by the Qwen3 reasoning parser and emitted as SGLang-shaped chunks into the
//! same channel the SGLang stream reader feeds, so the `Relay` (SSE) and the unary path see one backend shape.
//!
//! Spec m24 D10/D11: every committed token goes through the stop matcher first (a stop string cancels the
//! generation at the next token boundary and finishes it as `stop`), then the reasoning parser, then, for a
//! request with tools and `tool_choice` `auto`, the qwen3_coder tool parser, whose parameter values the sidecar
//! converts on the request's prepared tools.
//!
//! Spec m26 D11: a request with `n` choices runs `n` such sessions (each prefills the prompt), decoded as rows;
//! every choice has its own stop matcher, reasoning and tool parsers and chunks (`index` 0..n-1), and one usage
//! chunk counts the prompt once and sums the completions. A failed choice cancels the others.
//!
//! The session is released on every terminal path: the generation thread owns it from create to release, and a
//! cancel only records intent and calls `cere_executor_cancel`; the item still ends through its terminal poll.
//! A session is opened at `min(context, round64(prompt + max_tokens))` (`context`: `max_context_tokens`, or the
//! image decode combination's for a request with images, m213 D11) and reserves its KV pages at
//! create (ledger CR1); a create the pool cannot cover waits in [`SessionQueue`] for a release instead of failing.
//! With U1 preemption (M2.11 D8) the generation's creates go through [`Preemption`] instead, at the same size.
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use crate::ab::{self, Arm, RequestTally};
use crate::api_types::{ApiError, ContextExcess, INVALID_REQUEST, SERVER_ERROR};
use crate::app::NativeBackend;
use crate::digest;
use crate::levers::{DraftPolicy, Drafter, Family, GrammarDrafts};
use crate::native::detok::Vocab;
use crate::native::driver::{
    Committed, Driver, DriverError, Finish, ImageInput, Job, Model, Progress, RoundDrafts,
    Sampling, Session, WorkRequest,
};
use crate::native::ffi::{self, CompletionTicket, StatusError, TokenLogprobs};
use crate::native::inkling;
use crate::native::loop_guard::{LoopGuard, Repetition};
use crate::native::preempt::{NoSlot, Policy, Sched};
use crate::native::reasoning::{self, split, Reasoning, Split};
use crate::native::sampling;
use crate::native::sidecar::{
    Encoded, GrammarError, GrammarHandle, GrammarOpen, SidecarError, TemplateKwargs,
};
use crate::native::stop::{Finished, Matched, Step, StopMatcher, Text};
use crate::native::tool_parser::{self, Call, Chain, ChainStep, Delta, Item};
use crate::sglang::BackendError;
use crate::stream::BackendItem;
use crate::validate::{first_image, ChatRequest, ToolChoice};

/// An encoded request: the prompt and what the output path needs from the request.
pub(crate) struct Prompt {
    /// The prompt token IDs, image placeholders expanded (one `<|image_pad|>` per merged patch, m213 D8).
    ids: Vec<u32>,
    /// The prompt's images (spans and pixels) for the PREFILL item; empty for a text-only request.
    images: Vec<ImageInput>,
    /// The session's prefix-cache namespace (M2.10 D2); `None` when the executor runs no prefix cache.
    namespace: Option<[u8; 16]>,
    /// The PREFILL's checkpoint hint (M2.10 D4 (c), W 132); 0 = none.
    capture_position: u32,
    /// The session's context limit: the deployment's, or for an image request its image decode combination's
    /// (capped by the deployment's: 2,048 on the -v6 lab profile, the text context on -v43, VisionLong C3).
    context: u64,
    stop: Vec<String>,
    /// Per stop string, its token count (the matcher's window, spec m24 F9).
    stop_token_lens: Vec<u32>,
    /// The prepared tools when the tool parser runs: the request has tools and `tool_choice` is not `none` (D10).
    /// `required`/named outputs are the qwen3_coder structural tag's markup (spec m28 F11/F12), parsed by the same
    /// detector as `auto`, as SGLang does for a detector that supports structural tags (`_process_tool_calls`).
    tools: Option<Arc<Vec<Value>>>,
    /// `cere_work.sampling` of every item (m27 D1): `None` = the greedy path.
    sampling: Option<Sampling>,
    /// `top_logprobs` per committed token when the request has `logprobs: true` (absent = 0), else `None` (m27 D7).
    top_logprobs: Option<u32>,
    /// The model's stop tokens, then the request's `stop_token_ids` not among them (spec m212 D8): every ticket's
    /// item list and the matcher's token list.
    stop_token_ids: Vec<u32>,
    /// The reasoning parser of the output (spec m212 D4): the resolved thinking flag and `separate_reasoning`.
    reasoning: Reasoning,
    /// The resolved thinking flag (D4/D13): SGLang's `require_reasoning`, which gates `usage.reasoning_tokens` (D7).
    thinking: bool,
    /// `ignore_eos` (spec m212 D9): no token finishes the output; the matcher keeps only strings and the budget.
    ignore_eos: bool,
    /// ApiFit AF-L7: the request's `thinking_budget` (validated: thinking on, no logprobs).
    thinking_budget: Option<u64>,
    /// `cere_work.flags` of every ticket of the request ([`work_flags`]).
    flags: u32,
    /// A constrained request (spec m28 D8): one open grammar per choice, its mask that of position 0; empty for a
    /// grammar-free request.
    grammars: OffRuntime<OpenGrammar>,
    /// ab-split design note AB-04: the request's measures on its A/B arm, `None` without one.
    tally: Option<Arc<RequestTally>>,
    /// Track LoRA: the request's adapter lease; every choice's session is bound to its slot before the PREFILL.
    lora: Option<Arc<crate::native::lora::Adapter>>,
    /// WeightLab: the request's research run (plan slot and host-buffer reservation); every choice's session is
    /// bound to the plan before the PREFILL (`n` is 1 under a plan).
    research: Option<Arc<crate::native::research::Run>>,
}

/// Values whose drop blocks (an [`OpenGrammar`] sends `grammar_close` over a pipe that a compile or a hung child can
/// hold for a long time), dropped off the async runtime (audit N2): a [`Prompt`] that is refused or cancelled while
/// its request waits for admission is dropped on a tokio worker, and the values still in it are then handed to the
/// blocking pool. Outside a runtime there is no async worker to protect, and they drop in place.
struct OffRuntime<T: Send + 'static>(Vec<T>);

impl<T: Send + 'static> OffRuntime<T> {
    /// The values, which the caller now drops where it chooses (a blocking thread).
    fn into_vec(mut self) -> Vec<T> {
        std::mem::take(&mut self.0)
    }
}

impl<T: Send + 'static> Drop for OffRuntime<T> {
    fn drop(&mut self) {
        if self.0.is_empty() {
            return;
        }
        let values = std::mem::take(&mut self.0);
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn_blocking(move || drop(values));
            }
            Err(_) => drop(values),
        }
    }
}

impl Prompt {
    /// m32n design note D6/F8: the KV pool tokens one choice reserves, `min(context, ceil64(prompt +
    /// max_tokens))` (the pages its session can use, as [`generate`] opens it).
    pub(crate) fn reservation(&self, max_tokens: u64) -> u64 {
        self.context
            .min((self.ids.len() as u64 + max_tokens).div_ceil(64) * 64)
    }

    /// The prompt's token count (`usage.prompt_tokens`), known before generating (ApiFit AF-L8).
    pub(crate) fn tokens(&self) -> u64 {
        self.ids.len() as u64
    }

    /// D11 (2)/D13: the prompt's prefix-affinity key; `None` without the prefix cache (no namespace) or for a
    /// prompt shorter than one captured page.
    pub(crate) fn affinity_key(&self) -> Option<crate::native::affinity::Key> {
        crate::native::affinity::key(self.namespace.as_ref()?, &self.ids)
    }

    /// D12: a prompt encoded on one replica and granted another: its grammars (each replica runs its own grammar
    /// process) are closed off the async runtime and opened again on `backend`. A grammar-free prompt is unchanged.
    /// A refused re-open ends the request on its A/B arm as an error (AB-04), as a refused encode does.
    pub(crate) async fn reopen_grammars(
        &mut self,
        backend: &Arc<NativeBackend>,
        request: &ChatRequest,
    ) -> Result<(), ApiError> {
        if self.grammars.0.is_empty() {
            return Ok(());
        }
        // The old set drops off the runtime (`OffRuntime`).
        self.grammars = OffRuntime(Vec::new());
        match open_grammars(backend, request).await {
            Ok(grammars) => {
                self.grammars = OffRuntime(grammars);
                Ok(())
            }
            Err(error) => {
                if let Some(tally) = &self.tally {
                    tally.finish(crate::ab::End::Error);
                }
                Err(error)
            }
        }
    }
}

/// A choice's open grammar (spec m28 D8): its handle in the grammar process and the mask of the next position.
/// `grammar_close` runs when it drops, on every exit path of the choice (finish, cancel, error, refusal); the
/// drop is a blocking pipe call, so it is dropped off the async runtime.
pub(crate) struct OpenGrammar {
    backend: Arc<NativeBackend>,
    handle: GrammarHandle,
    mask: Vec<u32>,
}

impl Drop for OpenGrammar {
    fn drop(&mut self) {
        if let Some(grammar) = &self.backend.grammar {
            let _ = grammar.grammar_close(&self.handle);
        }
    }
}

/// The `grammar_open` of a constrained request, owned so a blocking task can send it: the constraint field's
/// raw JSON, else the tools with the raw `tool_choice` (spec m28 D6, D8), and the thinking flag (D15, ledger CR20).
struct GrammarRequest {
    field: Option<(&'static str, String)>,
    tools_json: Option<String>,
    tool_choice: Option<String>,
    parallel_tool_calls: Option<bool>,
    enable_thinking: bool,
}

impl GrammarRequest {
    /// `None` for a grammar-free request.
    fn of(request: &ChatRequest) -> Option<Self> {
        request.grammar_output()?;
        let field = request
            .constraint_json()
            .map(|(name, raw)| (name, raw.get().to_owned()));
        let tools = field.is_none();
        Some(Self {
            field,
            tools_json: request.tools_json().filter(|_| tools).map(str::to_owned),
            tool_choice: request
                .tool_choice_raw
                .as_deref()
                .filter(|_| tools)
                .map(|raw| raw.get().to_owned()),
            parallel_tool_calls: request.parallel_tool_calls.filter(|_| tools),
            enable_thinking: request.thinking,
        })
    }

    fn open(&self) -> GrammarOpen<'_> {
        let field = |name: &str| {
            self.field
                .as_ref()
                .filter(|(n, _)| *n == name)
                .map(|(_, raw)| raw.as_str())
        };
        GrammarOpen {
            response_format: field("response_format"),
            regex: field("regex"),
            ebnf: field("ebnf"),
            tools_json: self.tools_json.as_deref(),
            tool_choice: self.tool_choice.as_deref(),
            parallel_tool_calls: self.parallel_tool_calls,
            enable_thinking: self.enable_thinking,
        }
    }
}

/// One open grammar per choice for a constrained request (on a blocking task; spec m28 D8 step 1), none for a
/// grammar-free one. A grammar process that is not running refuses the request (D7: 503 `not_ready`); a refused
/// open closes the ones already opened.
async fn open_grammars(
    backend: &Arc<NativeBackend>,
    request: &ChatRequest,
) -> Result<Vec<OpenGrammar>, ApiError> {
    let Some(open) = GrammarRequest::of(request) else {
        return Ok(Vec::new());
    };
    if backend.grammar.is_none() {
        return Err(GrammarError::NotReady.api_error());
    }
    let (b, n) = (backend.clone(), request.n);
    tokio::task::spawn_blocking(move || {
        let grammar = b.grammar.as_ref().expect("checked above");
        let mut opened = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let mut mask = Vec::new();
            let handle = grammar
                .grammar_open(&open.open(), &mut mask)
                .map_err(|e| e.api_error())?;
            opened.push(OpenGrammar {
                backend: b.clone(),
                handle,
                mask,
            });
        }
        Ok(opened)
    })
    .await
    .unwrap_or_else(|_| Err(GrammarError::NotReady.api_error()))
}

/// Encodes the request's messages (one sidecar call, which also validates and prepares `tools` and counts the
/// stop strings' tokens) and applies the context check: prompt + `max_tokens` must fit the session's context
/// limit (the image decode combination's for a request with images). A constrained request then opens its
/// grammars (spec m28 D8; a refusal is the request's answer). Nothing is submitted here. `arm` (ab-split design note
/// AB-03) is the request's A/B arm: its lever choices go over the active set for the work flags, and an encoded
/// request starts on it (its arm's `requests` count). loop-guard design note LG-D2: with the loop guard on, a
/// conversation ending in `tool_repeat_run` identical tool call -> result pairs is refused first (400
/// `loop_detected`).
pub(crate) async fn encode(
    backend: &Arc<NativeBackend>,
    request: &ChatRequest,
    cache_salt: Option<&str>,
    arm: Option<&Arm>,
) -> Result<Prompt, ApiError> {
    if let Some(guard) = &backend.loop_guard {
        guard.check_messages(&request.messages)?;
    }
    // Track LoRA: an adapter request's cached states are its adapter's own (the base namespace mixed with the
    // adapter digest); a base request's namespace is unchanged.
    let namespace = cache_namespace(backend, cache_salt)?.map(|base| match &request.adapter {
        Some(a) => crate::native::lora::namespace(base, &a.sha256),
        None => base,
    });
    // WeightLab WL-11: an intervening plan's cached states are its plan's own; a capture-only plan keeps the
    // request's namespace (`Plan::cache_namespace`).
    let namespace = namespace.map(|base| match &request.research {
        Some(r) => r.plan.plan.cache_namespace(base),
        None => base,
    });
    // Before the sidecar call: a refused `logit_bias` id never costs an encode.
    // WeightLab: an unseeded request under a plan with a seed draws the plan's derived seed.
    let seed = match (request.seed, request.research.as_ref().and_then(|r| r.seed)) {
        (Some(seed), _) | (None, Some(seed)) => seed,
        (None, None) => random_seed()?,
    };
    let constrained = request.grammar_output().is_some();
    let sampling = sampling::sampling(
        request,
        &backend.generation_defaults,
        backend.vocab_size,
        seed,
        constrained,
    )?;
    let render_tools = request.render_tools();
    let stop = request.stop.clone();
    let encoded = render(backend, request).await?;
    let context = match request.image_context_tokens {
        Some(image) if !encoded.images.is_empty() => image.min(backend.max_context_tokens),
        _ => backend.max_context_tokens,
    };
    let total = encoded.ids.len() as u64 + request.max_output_tokens;
    if total > context {
        return Err(ContextExcess {
            prompt: encoded.ids.len() as u64,
            max_tokens: request.max_output_tokens,
            context,
        }
        .error());
    }
    let grammars = OffRuntime(open_grammars(backend, request).await?);
    let capture_position = capture_position(
        encoded.last_message_start,
        encoded.ids.len(),
        namespace.is_some() && (encoded.images.is_empty() || request.image_capture),
    );
    // One read of the active set (a hot swap holds the write lock), so the ticket's bits come from one whole set;
    // an A/B arm's runtime-layer choices go over it as the profile's do.
    let active = backend.active_levers();
    let levers = arm.map_or(active, |a| a.levers.apply(active));
    // GrammarSpec GS-04: under `grammar_drafts = "on"` a constraint no longer refuses speculation; its DECODE
    // verifies drafts under the grammar ([`Steer::run`]).
    let flags = family_flags(
        image_flags(
            draft_flags(
                work_flags(
                    request.ignore_eos == Some(true),
                    request.min_tokens.unwrap_or(0),
                    constrained && levers.grammar_drafts == GrammarDrafts::Off,
                    sampling.as_ref(),
                ),
                levers.drafter,
                levers.draft_policy,
                sampling.as_ref(),
            ),
            !encoded.images.is_empty(),
            request.image_speculation,
        ),
        backend.model.family(),
        sampling.is_some() || request.logprobs,
    );
    let prompt_tokens = encoded.ids.len() as u64;
    Ok(Prompt {
        ids: encoded.ids,
        images: encoded.images,
        namespace,
        capture_position,
        context,
        stop,
        stop_token_lens: encoded.stop_token_lens,
        tools: render_tools.then(|| Arc::new(encoded.tools_prepared.unwrap_or_default())),
        sampling,
        top_logprobs: request.logprobs.then(|| request.top_logprobs.unwrap_or(0)),
        stop_token_ids: stop_token_ids(&backend.stop_token_ids, request),
        reasoning: Reasoning::for_family(
            backend.model.family(),
            request.thinking,
            request.separate_reasoning != Some(false),
        ),
        thinking: request.thinking,
        ignore_eos: request.ignore_eos == Some(true),
        thinking_budget: request.thinking_budget,
        flags,
        grammars,
        tally: arm.map(Arm::start),
        lora: request.adapter.clone(),
        // WeightLab admission: the plan's request bound reserved in the host ring (503 `research_buffer_full`).
        research: match &request.research {
            Some(applied) => {
                let reservation = applied
                    .reserve(prompt_tokens, request.max_output_tokens, request.n)
                    .map_err(|e| e.api_error())?;
                Some(Arc::new(crate::native::research::Run::new(
                    applied.clone(),
                    reservation,
                )))
            }
            None => None,
        },
    })
}

/// The session's prefix-cache namespace (M2.10 D2), set before its first PREFILL: none unless the executor runs
/// the cache (`native_prefix_cache`); the scope's `cache_salt` bytes under `prefix_cache_isolation =
/// "cache_salt"`, else all-zero (`allow_cross_scope_prefix_cache` or a single scope).
pub(crate) fn cache_namespace(
    backend: &NativeBackend,
    cache_salt: Option<&str>,
) -> Result<Option<[u8; 16]>, ApiError> {
    if backend.capabilities.native_prefix_cache != 1 {
        return Ok(None);
    }
    let Some(salt) = cache_salt else {
        return Ok(Some([0; 16]));
    };
    digest::unhex(salt)
        .and_then(|bytes| <[u8; 16]>::try_from(bytes).ok())
        .map(Some)
        .ok_or_else(|| {
            ApiError::new(
                500,
                SERVER_ERROR,
                "internal_error",
                "native cache namespace: the scope's cache_salt is not 32 hex digits".to_owned(),
                None,
            )
        })
}

/// The PREFILL's `capture_position` (M2.10 D4 (c), W 132): the sidecar's `last_message_start` floored to 64,
/// sent only for a namespaced text-only session when it is at least 8,192 and the prompt ends within 1,024
/// tokens after it; 0 (none) otherwise.
fn capture_position(last_message_start: u32, prompt_tokens: usize, namespaced: bool) -> u32 {
    let p = last_message_start / 64 * 64;
    let n = prompt_tokens as u64;
    let fits = p >= 8192 && u64::from(p) < n && n - u64::from(p) <= 1024;
    if namespaced && fits {
        p
    } else {
        0
    }
}

/// The `cere_work.flags` every ticket of a request carries, PREFILL and DECODE alike (spec m212 D9 (ii), (iv)):
/// `ignore_eos` sets `CERE_WORK_IGNORE_STOP_TOKENS`; `ignore_eos`, `min_tokens > 0`, a grammar constraint that may
/// not speculate (spec m28 D8, I3; `constrained`: every one unless `grammar_drafts = "on"`, GrammarSpec GS-04), a
/// non-neutral penalty (presence, frequency or repetition) or any `logit_bias` entry (m29 design note D11,
/// M29S2-08) sets `CERE_WORK_NO_SPECULATION` (Q9 a, D14: such requests run without MTP).
fn work_flags(
    ignore_eos: bool,
    min_tokens: u64,
    constrained: bool,
    sampling: Option<&Sampling>,
) -> u32 {
    let mut flags = 0;
    if ignore_eos {
        flags |= ffi::WORK_IGNORE_STOP_TOKENS;
    }
    let processed = sampling.is_some_and(|s| {
        s.presence_penalty != 0.0
            || s.frequency_penalty != 0.0
            || s.repetition_penalty != 1.0
            || !s.logit_bias.is_empty()
    });
    if ignore_eos || min_tokens > 0 || constrained || processed {
        flags |= ffi::WORK_NO_SPECULATION;
    }
    flags
}

/// m213 D14 / VisionLong C3: an image session runs without speculation (`CERE_WORK_NO_SPECULATION` on every item,
/// over the draft bits) unless the profile has the image `verify` combination for its sampler mode; a text-only
/// request's flags pass through untouched.
fn image_flags(flags: u32, images: bool, image_speculation: bool) -> u32 {
    if images && !image_speculation {
        flags | ffi::WORK_NO_SPECULATION
    } else {
        flags
    }
}

/// m29l design note D6: the draft-source bits (ledger F bits 2, 3) a speculating deployment adds to every ticket that
/// may speculate (no `CERE_WORK_NO_SPECULATION`), from the `drafter` lever (m23 design note PI-01): "lookup" and
/// "mtp+lookup" let the executor verify suffix-lookup drafts, "lookup" also never verifies MTP proposals; "mtp" adds
/// nothing. m29a design note A2: the `draft_policy` lever's "adaptive" adds `CERE_WORK_ADAPTIVE_DRAFTS` (bit 4) to a
/// greedy ticket (`sampling` `None`) only, "fixed" nothing: a sampled ticket's seeded draws depend on each round's
/// draft count, so it keeps the fixed policy and its output equals the default's ([REVISED: review-wave-fg M1]).
/// DraftAccept: "mtp+lookup+tail" is "mtp+lookup" plus `CERE_WORK_TAIL_DRAFTS` (bit 5) on a greedy ticket only (the
/// executor appends a lookup tail only under the adaptive policy; a sampled ticket keeps the fixed contract).
/// TreeSpec (tree-spec design note TS-01): "mtp+lookup+long" is "mtp+lookup+tail" plus `CERE_WORK_LONG_DRAFTS`
/// (bit 6, a long lookup proposal on its own) on a greedy ticket only, likewise. Chains (chains design note):
/// "mtp+lookup+chain" is "mtp+lookup+long" plus `CERE_WORK_CHAIN_DRAFTS` (bit 7, those chains up to the model's max
/// chain drafts instead of 8) on a greedy ticket only, likewise.
fn draft_flags(
    flags: u32,
    drafter: Drafter,
    policy: DraftPolicy,
    sampling: Option<&Sampling>,
) -> u32 {
    if flags & ffi::WORK_NO_SPECULATION != 0 {
        return flags;
    }
    let sources = match drafter {
        Drafter::Mtp => 0,
        Drafter::Lookup => ffi::WORK_LOOKUP_DRAFTS | ffi::WORK_NO_MTP_DRAFTS,
        Drafter::MtpLookup => ffi::WORK_LOOKUP_DRAFTS,
        Drafter::MtpLookupTail if sampling.is_none() => {
            ffi::WORK_LOOKUP_DRAFTS | ffi::WORK_TAIL_DRAFTS
        }
        Drafter::MtpLookupTail => ffi::WORK_LOOKUP_DRAFTS,
        Drafter::MtpLookupLong if sampling.is_none() => {
            ffi::WORK_LOOKUP_DRAFTS | ffi::WORK_TAIL_DRAFTS | ffi::WORK_LONG_DRAFTS
        }
        Drafter::MtpLookupLong => ffi::WORK_LOOKUP_DRAFTS,
        Drafter::MtpLookupChain if sampling.is_none() => {
            ffi::WORK_LOOKUP_DRAFTS
                | ffi::WORK_TAIL_DRAFTS
                | ffi::WORK_LONG_DRAFTS
                | ffi::WORK_CHAIN_DRAFTS
        }
        Drafter::MtpLookupChain => ffi::WORK_LOOKUP_DRAFTS,
    };
    let policy = match policy {
        DraftPolicy::Adaptive if sampling.is_none() => ffi::WORK_ADAPTIVE_DRAFTS,
        DraftPolicy::Adaptive | DraftPolicy::Fixed => 0,
    };
    flags | sources | policy
}

/// The draft bits an Inkling 2 Mini ticket may carry (inkling design note D-I-S9B): plain MTP only. A greedy ticket
/// without logprobs verifies the drafter's block (no draft-source or policy bit: the executor's `choose_drafts` then
/// gives the window's 5 drafts); a sampled one or one asking for logprobs (`sampled`) runs without speculation
/// (`CERE_WORK_NO_SPECULATION`: its session never drafts and decodes on plain decode's path). Other families' flags
/// pass through untouched.
fn family_flags(flags: u32, family: Family, sampled: bool) -> u32 {
    if family != Family::Inkling2Loop {
        return flags;
    }
    let plain = flags
        & !(ffi::WORK_LOOKUP_DRAFTS
            | ffi::WORK_NO_MTP_DRAFTS
            | ffi::WORK_ADAPTIVE_DRAFTS
            | ffi::WORK_TAIL_DRAFTS
            | ffi::WORK_LONG_DRAFTS
            | ffi::WORK_CHAIN_DRAFTS);
    if sampled {
        plain | ffi::WORK_NO_SPECULATION
    } else {
        plain
    }
}

/// D4: a request without `seed` draws 8 bytes from the CSPRNG (F8); the toy records it, never the logs.
fn random_seed() -> Result<u64, ApiError> {
    digest::random_bytes::<8>()
        .map(u64::from_le_bytes)
        .map_err(|e| {
            ApiError::new(
                500,
                SERVER_ERROR,
                "internal_error",
                format!("native seed: {e}"),
                None,
            )
        })
}

/// The request's prompt token count, rendered and encoded exactly as [`encode`] does for `usage.prompt_tokens`
/// (M2.14 D3 `count_tokens`); no context check, nothing submitted.
pub(crate) async fn prompt_tokens(
    backend: &Arc<NativeBackend>,
    request: &ChatRequest,
) -> Result<u64, ApiError> {
    render(backend, request)
        .await
        .map(|encoded| encoded.ids.len() as u64)
}

/// NetTeleport S1: the request's prompt token IDs, rendered and encoded exactly as [`encode`] does (the ids a chat
/// of the same body looks up in the prefix cache); nothing submitted. A request with images has no prefix state
/// to move (`None`).
pub(crate) async fn prompt_ids(
    backend: &Arc<NativeBackend>,
    request: &ChatRequest,
) -> Result<Option<Vec<u32>>, ApiError> {
    let encoded = render(backend, request).await?;
    Ok(encoded.images.is_empty().then_some(encoded.ids))
}

/// The sidecar call of [`encode`]: the request's messages, tools and stop strings through the pinned template, with
/// the request's template kwargs (spec m212 D4/D13: the resolved thinking flag and `reasoning_effort`). A named
/// `tool_choice` renders only that tool (spec m28 D4, B3: cered filters `tools`; `encode_chat` is unchanged).
async fn render(backend: &Arc<NativeBackend>, request: &ChatRequest) -> Result<Encoded, ApiError> {
    // m213 D5/C6: image parts passed the profile's image combination gate (validation); the executor must also
    // serve image input, else the request is refused here, before the sidecar sees it.
    if let Some(at) = first_image(&request.messages) {
        if backend.capabilities.native_image_input == 0 {
            return Err(ApiError::unsupported_parameter(
                at,
                "the native executor has no image input",
            ));
        }
    }
    // m213 D2/F4: a text-only parts array renders like its joined string; a message with an image part goes to
    // the sidecar as its parts (feature `image`).
    let messages = request
        .messages
        .iter()
        .map(|m| m.joined().unwrap_or_else(|| m.clone()))
        .collect::<Vec<_>>();
    let tools_json = match (&request.tool_choice, request.tools_json()) {
        (ToolChoice::Named(name), Some(tools)) => Some(named_tool(tools, name)?),
        (_, tools) => tools.map(str::to_owned),
    };
    let render_tools = request.render_tools();
    let stop = request.stop.clone();
    let template = TemplateKwargs {
        enable_thinking: request.thinking,
        reasoning_effort: request.reasoning_effort.clone(),
        preserve_thinking: request.preserve_thinking,
    };
    let b = backend.clone();
    let encoded = tokio::task::spawn_blocking(move || {
        b.sidecar.encode_chat(
            &messages,
            tools_json.as_deref(),
            render_tools,
            &stop,
            &template,
        )
    })
    .await
    .map_err(|_| sidecar_error(&SidecarError::Gone("encode task failed".to_owned())))?;
    encoded.map_err(|e| sidecar_error(&e))
}

/// The tools of a named `tool_choice` (validated to name one of them): only the entries whose `function.name` is
/// `name`, as SGLang's chat path filters them for the template (`serving_chat.py`).
fn named_tool(tools: &str, name: &str) -> Result<String, ApiError> {
    let invalid = || ApiError::invalid_value("tools", "tools is not a JSON array");
    let tools: Vec<Value> = serde_json::from_str(tools).map_err(|_| invalid())?;
    let named: Vec<Value> = tools
        .into_iter()
        .filter(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
        .collect();
    serde_json::to_string(&named).map_err(|_| invalid())
}

/// A sidecar refusal is the request's fault (400): a tool schema SGLang rejects is `invalid_value` at the path
/// the sidecar names (D2), any other refusal `tokenizer_refused`. Anything else leaves the tokenizer unusable
/// (503, and readiness turns not ready once the child is gone).
fn sidecar_error(error: &SidecarError) -> ApiError {
    match error {
        SidecarError::Refused {
            kind,
            message,
            param: Some(param),
        } if kind == "invalid_value" => ApiError::invalid_value(param.clone(), message.clone()),
        SidecarError::Refused { kind, message, .. } => ApiError::new(
            400,
            INVALID_REQUEST,
            "tokenizer_refused",
            format!("the tokenizer refused the messages ({kind}): {message}"),
            Some("messages".to_owned()),
        ),
        other => ApiError::new(503, SERVER_ERROR, "not_ready", other.to_string(), None),
    }
}

/// Why a generation was cancelled; the first reason sticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CancelReason {
    /// The client (or the server for it) cancelled: the stream ends without a finish.
    Client,
    /// A stop string matched (D11): the output is finished; the tokens after the match are dropped.
    StopString,
    /// The output path failed (a tool value conversion): the stream ends with the backend error.
    Failed,
    /// loop-guard design note LG-D1: the loop guard found the output repeating: the output is finished.
    Repetition,
}

/// The cancel state of a choice: the client's (or the output path's) reason, the ticket of the item in flight,
/// and a pending U1 preemption (M2.11 D8), which cancels the item at its token boundary without ending the
/// generation.
#[derive(Default)]
struct CancelState {
    reason: Option<CancelReason>,
    ticket: Option<CompletionTicket>,
    preempt: bool,
}

/// Cancel intent of one choice, shared by its generation thread and its emitter.
struct ChoiceCancel {
    backend: Arc<NativeBackend>,
    state: Mutex<CancelState>,
}

impl ChoiceCancel {
    fn new(backend: Arc<NativeBackend>) -> Self {
        Self {
            backend,
            state: Mutex::new(CancelState::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CancelState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `cere_executor_cancel` for the item in flight (or the next one the moment it is submitted); no further
    /// item is submitted. A round trip to the driver thread: call it off the async runtime.
    fn cancel_for(&self, reason: CancelReason) {
        let ticket = {
            let mut state = self.lock();
            state.reason.get_or_insert(reason);
            state.ticket
        };
        if let Some(ticket) = ticket {
            let _ = self.backend.driver.cancel(ticket);
        }
        // A choice waiting for pages stops waiting.
        self.backend.sessions.wake();
    }

    /// U1: cancels the item in flight (or the next one) at its token boundary so the generation parks.
    fn preempt(&self) {
        let ticket = {
            let mut state = self.lock();
            state.preempt = true;
            state.ticket
        };
        if let Some(ticket) = ticket {
            let _ = self.backend.driver.cancel(ticket);
        }
    }

    /// Whether a preemption is pending (asked for and not yet taken by the generation thread).
    fn preempt_pending(&self) -> bool {
        self.lock().preempt
    }

    /// Takes a pending preemption.
    fn take_preempt(&self) -> bool {
        std::mem::take(&mut self.lock().preempt)
    }

    fn cancelled(&self) -> bool {
        self.lock().reason.is_some()
    }

    fn reason(&self) -> Option<CancelReason> {
        self.lock().reason
    }

    /// Records the submitted item; a cancel or preemption that came first cancels it at once.
    fn submitted(&self, driver: &Driver, ticket: CompletionTicket) {
        let cancelled = {
            let mut state = self.lock();
            state.ticket = Some(ticket);
            state.reason.is_some() || state.preempt
        };
        if cancelled {
            let _ = driver.cancel(ticket);
        }
    }
}

/// The cancel handle of a request's choices (M2.6 D11: one session each), shared by the request executor and
/// the emitters: a failed choice cancels them all.
pub(crate) struct NativeCancel {
    choices: Vec<Arc<ChoiceCancel>>,
}

impl NativeCancel {
    /// Cancels every choice (client intent). A round trip to the driver thread per choice: call it off the async
    /// runtime.
    pub(crate) fn cancel(&self) {
        self.cancel_for(CancelReason::Client, None);
    }

    /// Cancels every choice but `except` with `reason`.
    fn cancel_for(&self, reason: CancelReason, except: Option<usize>) {
        for (index, choice) in self.choices.iter().enumerate() {
            if Some(index) != except {
                choice.cancel_for(reason);
            }
        }
    }
}

/// How a generation ended in the executor, and a grammar failure that ended it (spec m28 D8: 500 `internal_error`).
struct Outcome {
    finish: Finish,
    failure: Option<BackendError>,
    /// The PREFILL's `cached_tokens` (M2.10 D10).
    cached: u32,
}

impl Outcome {
    fn of(finish: Finish) -> Self {
        Self {
            finish,
            failure: None,
            cached: 0,
        }
    }
}

/// Raises the backend's fault flag when dropped during a panic (audit N3): the generation closure panicked, so the
/// executor's state after it is unknown.
struct FaultOnPanic<'a>(&'a AtomicBool);

impl Drop for FaultOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Release);
        }
    }
}

/// A started generation: its cancel handle and the SGLang-shaped chunk stream.
pub(crate) struct Generation {
    pub(crate) cancel: Arc<NativeCancel>,
    pub(crate) chunks: mpsc::Receiver<BackendItem>,
}

/// What the chunks name: the response `id`, `model` and `created`.
pub(crate) struct Head {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) created: u64,
}

impl Head {
    pub(crate) fn new(id: &str, model: &str) -> Self {
        Self {
            id: id.to_owned(),
            model: model.to_owned(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }
}

/// Starts the request's `n` choices and counts the submissions (M2.6 D11): one session each, every one
/// prefilling the prompt, their chunks (`choices[0].index` = the choice) on one channel; once every choice ended,
/// one usage chunk (prompt once, completions summed) and the end. `guard` (the admission slot) is held until
/// every session is released, whatever the reader of the chunks does. `stream` picks the parsers: SGLang's
/// streaming ones for SSE, its non-stream `detect_and_parse` / `parse_non_stream` over the whole text for a
/// unary response.
pub(crate) fn start<G: Send + 'static>(
    backend: Arc<NativeBackend>,
    prompt: Prompt,
    max_tokens: u64,
    n: u32,
    stream: bool,
    head: Head,
    guard: G,
) -> Generation {
    let cancel = Arc::new(NativeCancel {
        choices: (0..n)
            .map(|_| Arc::new(ChoiceCancel::new(backend.clone())))
            .collect(),
    });
    let (tx, chunks) = mpsc::channel(1);
    let head = Arc::new(head);
    let Prompt {
        mut ids,
        mut stop,
        mut stop_token_lens,
        tools,
        mut images,
        namespace,
        capture_position,
        context,
        sampling,
        top_logprobs,
        mut stop_token_ids,
        reasoning,
        thinking,
        ignore_eos,
        thinking_budget,
        flags,
        grammars,
        tally,
        lora,
        research,
    } = prompt;
    // NetTeleport S3: a text-only namespaced prompt without an adapter or plan is what a stop can park and the next
    // start re-import (in memory only; never written in the clear).
    if let (Some(recorder), Some(namespace)) = (&backend.parked, namespace) {
        if images.is_empty() && lora.is_none() && research.is_none() {
            recorder.record(namespace, &ids);
        }
    }
    let mut grammars = grammars.into_vec().into_iter();
    let prompt_tokens = ids.len() as u64;
    let mut choices = Vec::with_capacity(cancel.choices.len());
    for (index, choice) in cancel.choices.iter().enumerate() {
        // The last choice takes the request's vectors; the others copy them.
        let last = index + 1 == cancel.choices.len();
        let ids = share(&mut ids, last);
        let images = share(&mut images, last);
        let matcher_stops = if ignore_eos {
            Vec::new()
        } else {
            stop_token_ids.clone()
        };
        let item_stops = share(&mut stop_token_ids, last);
        let (progress, committed) = mpsc::unbounded_channel::<Committed>();
        let (outcome_tx, outcome) = oneshot::channel();
        // U1: each choice queues for a session here, so arrival order is the start order.
        let slot = backend.preemption.as_ref().map(|p| p.enqueue(choice));
        let (b, c) = (backend.clone(), choice.clone());
        // Set at runtime, by the generation thread, when the grammar terminates on a token the executor does
        // not stop on (spec m28 D8): a `OnceLock`, not a `LazyLock`.
        let grammar = grammars.next().map(|open| Steer {
            open,
            stop: Arc::new(std::sync::OnceLock::new()),
            ignore_eos,
        });
        let grammar_stop = grammar.as_ref().map(|g| g.stop.clone());
        // ApiFit AF-L7: a budget binds a free (grammar-less) thinking choice shorter than its output budget.
        let budget = thinking_budget
            .filter(|b| thinking && grammar.is_none() && *b < max_tokens)
            .and_then(|b| ThinkingBudget::new(&backend.vocab, b, ignore_eos));
        // m26 D11 / ledger CR4 (M8 channels (2, 3)): choice 0 keeps the request's seed, choice i >= 1 samples with
        // `choice_seed(seed, i)`, so sampled choices are distinct.
        let sampling = sampling.clone().map(|mut s| {
            if index > 0 {
                s.seed = sampling::choice_seed(s.seed, index as u32);
            }
            s
        });
        let ticket_tally = tally.clone();
        let ticket_lora = lora.clone();
        let ticket_research = research.clone();
        tokio::task::spawn_blocking(move || {
            // Audit N3: a panic below is a possible device fault; the replica's next probe re-smokes the card.
            let fault = FaultOnPanic(&b.fault);
            // An image session's PREFILL carries the images (their pixels live in the work item until the submit
            // returns).
            let tickets = TicketSettings {
                stop_token_ids: item_stops,
                flags,
                sampling,
                top_logprobs,
                context,
                namespace,
                tally: ticket_tally,
                lora: ticket_lora,
                research: ticket_research,
            };
            let prefill = WorkRequest::prefill(ids, 1)
                .with_images(images)
                .with_capture_position(capture_position);
            let outcome = generate(
                &b, prefill, &tickets, max_tokens, &c, progress, slot, grammar, budget,
            );
            // m32n design note D9 (b): an internal or unknown executor status, or a stopped driver, may be a
            // device fault; the replica's next probe re-smokes the card (a per-request error passes it).
            if let Err(
                DriverError::Stopped
                | DriverError::Status {
                    status: StatusError::Internal | StatusError::Unknown(_),
                    ..
                },
            ) = &outcome
            {
                b.fault.store(true, Ordering::Release);
            }
            drop(fault);
            let _ = outcome_tx.send(outcome);
        });
        let setup = Choice {
            backend: backend.clone(),
            tx: tx.clone(),
            head: head.clone(),
            index,
            cancel: choice.clone(),
            group: cancel.clone(),
            stop: share(&mut stop, last),
            stop_token_lens: share(&mut stop_token_lens, last),
            stop_token_ids: matcher_stops,
            reasoning,
            thinking,
            tools: tools.clone(),
            stream,
            max_tokens,
            top_logprobs,
            slot,
            grammar_stop,
            tally: tally.clone(),
        };
        choices.push(setup.run(committed, outcome));
    }
    // Counted once every choice is queued for a U1 slot, so a reader that sees the count also sees the queue entry.
    backend
        .submissions
        .fetch_add(u64::from(n), Ordering::Release);
    tokio::spawn(async move {
        let ends = futures_util::future::join_all(choices).await;
        // The first failed choice fails the request; a choice without a finish (cancelled) ends it without one.
        // `cached_tokens` is choice 0's (M2.10 D3′: choices 1..n-1 hit choice 0's prefix).
        let ended = ends.into_iter().enumerate().try_fold(
            Some(Tokens::default()),
            |total, (index, end)| {
                Ok::<_, BackendError>(end?.and_then(|tokens| {
                    total.map(|t| Tokens {
                        completion: t.completion + tokens.completion,
                        reasoning: t.reasoning + tokens.reasoning,
                        cached: if index == 0 { tokens.cached } else { t.cached },
                    })
                }))
            },
        );
        // AB-04: folded into the arm's stats before the client sees the end, so a stats read after the
        // response counts the request.
        if let Some(tally) = &tally {
            tally.finish(match &ended {
                Ok(Some(tokens)) => ab::End::Completed {
                    completion_tokens: tokens.completion,
                },
                Ok(None) => ab::End::Cancelled,
                Err(_) => ab::End::Error,
            });
        }
        let end = match ended {
            Ok(Some(tokens)) => {
                let usage = chunk(
                    &head,
                    json!([]),
                    json!({
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens + tokens.completion,
                        "completion_tokens": tokens.completion,
                        "prompt_tokens_details": { "cached_tokens": tokens.cached },
                        "reasoning_tokens": tokens.reasoning,
                    }),
                );
                let _ = tx.send(Ok(Some(usage))).await;
                Ok(None)
            }
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        };
        let _ = tx.send(end).await;
        drop(guard);
    });
    Generation { cancel, chunks }
}

/// `value` itself for the last user, a copy for the others.
fn share<T: Clone + Default>(value: &mut T, last: bool) -> T {
    if last {
        std::mem::take(value)
    } else {
        value.clone()
    }
}

/// A finished choice's `usage` counts: its completion tokens and its `reasoning_tokens` (spec m212 D7), summed
/// over the choices.
#[derive(Clone, Copy, Debug, Default)]
struct Tokens {
    completion: u64,
    reasoning: u64,
    /// The choice's PREFILL `cached_tokens` (M2.10 D10); the usage reports choice 0's.
    cached: u64,
}

/// One choice of a request: what its emitter needs.
struct Choice {
    backend: Arc<NativeBackend>,
    tx: mpsc::Sender<BackendItem>,
    head: Arc<Head>,
    index: usize,
    cancel: Arc<ChoiceCancel>,
    group: Arc<NativeCancel>,
    stop: Vec<String>,
    stop_token_lens: Vec<u32>,
    /// The matcher's token list: [`Prompt::stop_token_ids`], none under `ignore_eos` (D9).
    stop_token_ids: Vec<u32>,
    reasoning: Reasoning,
    thinking: bool,
    tools: Option<Arc<Vec<Value>>>,
    stream: bool,
    max_tokens: u64,
    top_logprobs: Option<u32>,
    /// The choice's U1 generation id (M2.11 D8), when the backend preempts.
    slot: Option<u64>,
    /// A constrained choice's grammar termination on a token the executor did not stop on (spec m28 D8):
    /// `(output index, token)`, set before that token's progress report.
    grammar_stop: Option<GrammarStop>,
    /// ab-split design note AB-04: the request's measures on its A/B arm.
    tally: Option<Arc<RequestTally>>,
}

impl Choice {
    /// Emits the choice's chunks from its committed tokens to its finish chunk: `Ok(Some(tokens))` when it
    /// finished, `Ok(None)` when it was cancelled without a finish.
    async fn run(
        self,
        mut committed: mpsc::UnboundedReceiver<Committed>,
        outcome: oneshot::Receiver<Result<Outcome, DriverError>>,
    ) -> Result<Option<Tokens>, BackendError> {
        let (backend, slot) = (self.backend, self.slot);
        // Specials stay in the text iff the tool parser runs (`serving_chat.py:1080-1081`), and always for the
        // Inkling parsers (`_patch_reasoning_skip_special_tokens`).
        let family = backend.model.family();
        let is_inkling = family == Family::Inkling2Loop;
        let skip_special = self.tools.is_none() && !is_inkling;
        let matcher = StopMatcher::new(
            &backend.vocab,
            self.stop,
            &self.stop_token_lens,
            self.stop_token_ids,
            self.max_tokens,
        );
        let parse = match (self.stream, &self.tools) {
            (false, _) => Parse::Whole,
            (true, None) => Parse::Plain(reasoning::Stream::new(self.reasoning)),
            (true, Some(_)) => Parse::Tools(Chain::new(
                Some(tool_parser::Format::of(family)),
                self.reasoning,
            )),
        };
        // SGLang's `think_end_ids`: Inkling's reasoning is always on and ends at the first `<|end_message|>`.
        let reasoning_tokens = if is_inkling {
            ReasoningTokens::new(true, Some(inkling::END_MESSAGE_ID))
        } else {
            ReasoningTokens::new(self.thinking, backend.vocab.think_end())
        };
        let mut out = Emitter {
            tx: self.tx,
            head: self.head,
            index: self.index,
            backend: backend.clone(),
            cancel: self.cancel,
            group: self.group,
            text: Text::new(backend.vocab.decoder_with(skip_special)),
            matcher,
            sent: 0,
            parse,
            reasoning: self.reasoning,
            reasoning_tokens,
            tools: self.tools,
            called: false,
            failed: None,
            logprobs: self.top_logprobs.map(|_| Vec::new()),
            grammar_stop: self.grammar_stop,
            repetition: backend.loop_guard.as_ref().and_then(LoopGuard::repetition),
        };
        out.delta(
            json!({ "reasoning_content": null, "role": "assistant", "content": "" }),
            false,
        )
        .await;
        // `progress` carries every committed token once; it closes when the generation has ended.
        // AB-04: the first and last committed instants and the token count, for the arm's TTFT and decode rate.
        let mut span: Option<(Instant, Instant)> = None;
        let mut count = 0u64;
        while let Some(c) = committed.recv().await {
            backend
                .committed
                .fetch_add(c.tokens.len() as u64, Ordering::Relaxed);
            if let (Some(p), Some(id)) = (&backend.preemption, slot) {
                p.committed(id, c.tokens.len() as u64);
            }
            if self.tally.is_some() && !c.tokens.is_empty() {
                let now = Instant::now();
                span = Some((span.map_or(now, |(first, _)| first), now));
                count += c.tokens.len() as u64;
            }
            out.tokens(&c).await;
        }
        if let (Some(tally), Some((first, last))) = (&self.tally, span) {
            tally.choice(first, last, count);
        }
        let outcome = outcome.await.unwrap_or(Err(DriverError::Stopped));
        out.finish(outcome).await
    }
}

/// The model's stop tokens followed by the request's `stop_token_ids` not among them (spec m212 D8; the request's
/// list is already a set below the vocabulary, `validate.rs`).
fn stop_token_ids(model: &[u32], request: &ChatRequest) -> Vec<u32> {
    let mut ids = model.to_vec();
    for id in request.stop_token_ids.iter().flatten() {
        if !model.contains(id) {
            ids.push(*id);
        }
    }
    ids
}

/// What every ticket of one choice carries: `stop_token_ids` (the model's and the request's, m212 D8), the request's
/// `flags` (m212 D9), its `sampling` (m27 D5) and, when the request asked for them, `top_logprobs` alternatives
/// (m27 D7); and the context limit its session is opened within (the deployment's, or an image request's image
/// decode combination's, m213 D11).
struct TicketSettings {
    stop_token_ids: Vec<u32>,
    flags: u32,
    sampling: Option<Sampling>,
    top_logprobs: Option<u32>,
    context: u64,
    /// Set on the session before its PREFILL (M2.10 D2).
    namespace: Option<[u8; 16]>,
    /// ab-split design note AB-04: the request's A/B arm measures (each ticket's speculate rounds).
    tally: Option<Arc<RequestTally>>,
    /// Track LoRA: the request's adapter lease, held until the session is released; bound before the PREFILL.
    lora: Option<Arc<crate::native::lora::Adapter>>,
    /// WeightLab: the request's research run, bound before the PREFILL (`cere_session_set_research`); its capture
    /// is read after the session is released.
    research: Option<Arc<crate::native::research::Run>>,
}

impl TicketSettings {
    /// A session the executor refuses to snapshot (track LoRA's adapter, WeightLab's plan): never parked through
    /// a snapshot and never grown, so it holds its whole budget.
    fn pinned(&self) -> bool {
        self.lora.is_some() || self.research.is_some()
    }

    fn apply(&self, work: WorkRequest) -> WorkRequest {
        let mut work = work
            .with_stop_tokens(self.stop_token_ids.clone())
            .with_flags(self.flags);
        if let Some(s) = &self.sampling {
            work = work.with_sampling(s.clone());
        }
        if let Some(top) = self.top_logprobs {
            work = work.with_logprobs(top);
        }
        work
    }
}

/// The generation thread: PREFILL (`prefill`: the prompt with its images, 1 token) then DECODE on one session, from
/// create to release; every ticket carries the choice's `tickets` settings (m27 D5: exactly one PREFILL per request,
/// so the device's output counts cover this request). With U1 (m211 D8) the session may be preempted between token boundaries: snapshotted, released, and
/// later restored into a fresh session that continues the DECODE with the remaining budget; the restore carries the
/// output counts (snapshot kind 8, ledger CR13) and the counter key is the restored position (m27 D4), so a sampled
/// choice continues as if never preempted. The Rust-side state (detokenizer, matcher, parsers) stays in the emitter
/// (D7). With `preempt_every` N (D13) every ticket ends by the next N tokens committed since the last preemption, and
/// there, nothing in flight, the generation asks for its forced preemption, as `cere-native generate
/// --preempt-every` does: a cancel sent while the ticket runs lands only after more tokens, so the ticket, or the
/// whole generation, could end before it.
///
/// A constrained choice (`grammar`, spec m28 D8) submits every ticket with the grammar's mask of its position 0 and
/// steers it ([`Steer::run`]): the PREFILL token's mask becomes the DECODE's position-0 mask, and each committed
/// DECODE token's mask is sent with `Constrain`. The grammar closes when this thread ends, on every path.
///
/// ApiFit AF-L7: a choice with a [`ThinkingBudget`] (never constrained, never under U1) relays its progress through
/// the budget's watch while the thinking is open. A thinking that ends (or stops) within the budget passes through
/// unchanged; one that reaches it is cut: the ticket is cancelled, the template's closing tokens are committed in
/// place of the overshoot, and the session is swapped for a fresh one whose PREFILL is the prompt + the kept
/// thinking + the closing tokens (the prompt's prefix-cache checkpoint makes it a short PREFILL), then DECODE as
/// usual. One output: its tokens, usage and finish are the whole choice's.
#[allow(clippy::too_many_arguments)]
fn generate(
    backend: &NativeBackend,
    prefill: WorkRequest,
    tickets: &TicketSettings,
    max_tokens: u64,
    cancel: &ChoiceCancel,
    progress: Progress,
    slot: Option<u64>,
    mut grammar: Option<Steer>,
    thinking: Option<ThinkingBudget>,
) -> Result<Outcome, DriverError> {
    let driver = &backend.driver;
    // Ledger CR1: the session (and, with U1, every session it is restored into) is opened at this size.
    // M2.5 D4: the pages it can use (prompt + output budget, 64-token pages) within the prompt's context limit (an
    // image request's is its image decode combination's, m213 D11).
    let prompt_tokens = prefill.tokens.len() as u64;
    let full = tickets
        .context
        .min((prompt_tokens + max_tokens).div_ceil(64) * 64);
    let u1 = backend.preemption.as_ref().zip(slot);
    // sessions design note SE-09: with `reserve_output_tokens` R the first session holds the prompt + R output
    // tokens only; a generation that uses it up parks itself and resumes in a session of `full` (U1 only).
    // An adapter or plan session (track LoRA, WeightLab WL-11) is never snapshotted, so it cannot grow: it holds
    // its whole budget.
    let reserve = match tickets.pinned() {
        true => 0,
        false => u1.map_or(0, |(p, _)| p.reserve()),
    };
    let mut context = match reserve {
        0 => full,
        r => full.min((prompt_tokens + max_tokens.min(r)).div_ceil(64) * 64),
    };
    let _leave = u1.map(|(p, id)| Leave(p, id));
    // AF-L7: a U1 session may be parked and restored mid-thinking; budgets bind without U1 only.
    let mut thinking = thinking.filter(|_| u1.is_none() && grammar.is_none());
    // The cut's continuation PREFILL extends the prompt's (its images and checkpoint hint kept).
    let origin = thinking.as_ref().map(|_| prefill.clone());
    let mut held = HeldSession {
        backend,
        u1: u1.is_some(),
        session: match u1 {
            None => backend
                .sessions
                .create(driver, &backend.model, context, cancel)?,
            Some((p, id)) => p.acquire(backend, id, context, prompt_tokens, cancel)?,
        },
    };
    let mut prefill = Some(prefill);
    let mut done = 0u64;
    // D13: `done` at the last preemption (parked, or its snapshot refused); a forced preemption granted at the last
    // ticket's boundary.
    let every = u1.map_or(0, |(p, _)| p.every());
    let mut mark = 0u64;
    let mut forced = false;
    // M2.10 D10: the PREFILL's `cached_tokens`.
    let mut cached = 0u32;
    // AF-L7: the PREFILL after a cut, whose `cached_tokens` is not the request's.
    let mut resumed = false;
    // WeightLab: the session's capture id (0 = nothing captured), from `cere_session_set_research`.
    let mut capture = 0u64;
    let mut run = |session: &mut Option<Session>| loop {
        let Some(held) = session.as_ref() else {
            // Cancelled while waiting for pages, a slot, or parked.
            return Ok(Outcome::of(Finish::Cancelled));
        };
        if let Some((p, id)) = u1 {
            if prefill.is_none() && !cancel.cancelled() {
                let force = std::mem::take(&mut forced);
                if cancel.take_preempt() || force {
                    let parked = session.take().expect("the session is held");
                    *session = p
                        .park(backend, id, context, parked, cancel, tickets.pinned())?
                        .session();
                    mark = done;
                    continue;
                }
                // SE-09: the session is used up and budget is left: grow it to the whole budget.
                if context < full && prompt_tokens + done >= context {
                    let held = session.take().expect("the session is held");
                    let position = prompt_tokens + done;
                    *session = match p.grow(backend, id, full, position, held, cancel)? {
                        Park::Resumed(s) => {
                            context = full;
                            Some(s)
                        }
                        other => other.session(),
                    };
                    continue;
                }
            }
        }
        if cancel.cancelled() {
            return Ok(Outcome::of(Finish::Cancelled));
        }
        let (work, first, budget) = match prefill.take() {
            Some(work) => (work, true, 1),
            None => {
                let mut budget = max_tokens - done;
                if every > 0 {
                    budget = budget.min(every - (done - mark) % every);
                }
                // SE-09: a DECODE never runs past its session (prompt + `done` ≤ `context`, M2.5 D4's bound).
                if context < full {
                    budget = budget.min(context - prompt_tokens - done);
                }
                (WorkRequest::decode(budget), false, budget)
            }
        };
        // M2.10 D2: the namespace goes on the session before its first (PREFILL) submit.
        if let Some(namespace) = tickets.namespace.filter(|_| first) {
            driver.set_cache_namespace(held, namespace)?;
        }
        // Track LoRA: so does the adapter binding.
        if let Some(adapter) = tickets.lora.as_ref().filter(|_| first) {
            driver.set_lora(held, adapter.slot)?;
        }
        // WeightLab: and so does the research plan (one capture per session, positions from the PREFILL on).
        if let Some(research) = tickets.research.as_ref().filter(|_| first) {
            capture = driver.set_research(held, research.slot(), prompt_tokens, max_tokens)?;
        }
        let mut work = tickets.apply(work);
        let (job, failure) = match grammar.as_mut() {
            None => match thinking.as_mut().filter(|b| b.state == Thinking::Open) {
                None => {
                    let job = driver.submit_with_progress(held, work, progress.clone())?;
                    cancel.submitted(driver, job.ticket());
                    (job, None)
                }
                Some(watch) => {
                    let (tx, rx) = mpsc::unbounded_channel();
                    let job = driver.submit_with_progress(held, work, tx)?;
                    cancel.submitted(driver, job.ticket());
                    watch.run(driver, &job, rx, &progress, &tickets.stop_token_ids);
                    (job, None)
                }
            },
            Some(steer) => {
                work = work.with_vocab_mask(&steer.open.mask);
                // GrammarSpec GS-02: a DECODE the constraint lets speculate (`grammar_drafts = "on"`, see
                // `encode`) verifies its rounds' drafts under the grammar.
                if !first && tickets.flags & ffi::WORK_NO_SPECULATION == 0 {
                    work = work.with_grammar_drafts();
                }
                let (tx, rx) = mpsc::unbounded_channel();
                let job = driver.submit_with_progress(held, work, tx)?;
                cancel.submitted(driver, job.ticket());
                let failure = steer.run(backend, &job, first, done, rx, &progress, tickets);
                (job, failure)
            }
        };
        let result = job.wait()?;
        if first && !resumed {
            cached = result.cached_tokens;
            if cached > 0 {
                backend.prefix_hits.fetch_add(1, Ordering::Relaxed);
            }
        }
        // m29 design note D11: the ticket's speculate rounds and accepted drafts (0 without speculation).
        backend
            .spec_verify_steps
            .fetch_add(result.verify_steps, Ordering::Release);
        backend
            .spec_accepted_drafts
            .fetch_add(result.accepted_drafts, Ordering::Release);
        if let Some(tally) = &tickets.tally {
            tally.spec(result.verify_steps, result.accepted_drafts);
        }
        if let Err(status) = result.status {
            return Err(DriverError::Status {
                call: "cere_executor_poll",
                status,
            });
        }
        done += result.committed_tokens;
        if let Some(cut) = thinking.as_mut().filter(|b| b.state == Thinking::Reached) {
            // The budget was reached: the overshoot (and the rest of the ticket) is dropped.
            if cancel.cancelled() {
                return Ok(Outcome::of(Finish::Cancelled));
            }
            cut.state = Thinking::Closed;
            let room = usize::try_from(max_tokens - cut.kept.len() as u64).unwrap_or(usize::MAX);
            let closure: Vec<u32> = cut.closure.iter().copied().take(room).collect();
            done = (cut.kept.len() + closure.len()) as u64;
            let _ = progress.send(Committed {
                tokens: closure.clone(),
                logprobs: Vec::new(),
                drafts: None,
            });
            if done >= max_tokens {
                return Ok(Outcome::of(Finish::Length));
            }
            if let Some(cut_session) = session.take() {
                backend.sessions.release(driver, cut_session)?;
            }
            *session = backend
                .sessions
                .create(driver, &backend.model, context, cancel)?;
            let mut next = origin.clone().expect("a budget keeps its PREFILL");
            next.tokens.extend_from_slice(&cut.kept);
            next.tokens.extend_from_slice(&closure);
            prefill = Some(next);
            resumed = true;
            continue;
        }
        if failure.is_some() {
            return Ok(Outcome {
                finish: result.finish,
                cached,
                failure,
            });
        }
        let ended = grammar.as_ref().is_some_and(|g| g.stop.get().is_some());
        // Only a PREFILL that used its whole 1-token budget continues with DECODE; a DECODE cancelled for a
        // preemption (and nothing else) continues after the park, one that used its budget up to a `preempt_every`
        // boundary continues after the forced preemption, if granted, and one that used its session up (SE-09)
        // continues after the grow. A grammar that terminated ends the choice.
        let preempted =
            result.finish == Finish::Cancelled && !cancel.cancelled() && cancel.preempt_pending();
        let boundary = every > 0
            && result.finish == Finish::Length
            && result.committed_tokens == budget
            && done < max_tokens
            && (done - mark) % every == 0;
        let used_up = context < full
            && result.finish == Finish::Length
            && result.committed_tokens == budget
            && prompt_tokens + done >= context;
        let more = !ended
            && if first {
                done < max_tokens
                    && (result.finish == Finish::Length
                        || (preempted && result.committed_tokens == 1))
            } else {
                (preempted || boundary || used_up) && done < max_tokens
            };
        if !more {
            // A preemption cancel that landed after the last budgeted token: the budget ended it.
            let finish = if preempted && done >= max_tokens {
                Finish::Length
            } else {
                result.finish
            };
            return Ok(Outcome::of(finish));
        }
        if let Some((p, id)) = u1.filter(|_| boundary) {
            forced = p.force(id, done);
        }
    };
    let outcome = run(&mut held.session);
    // A generation that ended with the thinking open (cancelled, stopped by a string) delivers what it held.
    if let Some(rest) = thinking.as_mut().map(ThinkingBudget::flush) {
        if !rest.is_empty() {
            let _ = progress.send(Committed {
                tokens: rest,
                logprobs: Vec::new(),
                drafts: None,
            });
        }
    }
    drop(progress);
    // A U1 generation's sessions bypass the CR1 queue (U1 waits and preempts for them).
    let released = held.release();
    // WeightLab: the capture is finalised by the release; read, listed and freed now (dropped on a failure).
    if let Some(research) = &tickets.research {
        research.end(
            driver,
            &backend.model,
            capture,
            outcome.is_ok() && released.is_ok(),
        );
    }
    let outcome = outcome?;
    released?;
    // `grammar_close` (the `OpenGrammar` drop) after the session is released.
    drop(grammar);
    Ok(Outcome { cached, ..outcome })
}

/// A choice's executor session (audit N3): released through [`HeldSession::release`] when the generation ends, and
/// by the drop when it unwinds, so a panic does not leak the session's KV pages or leave `SessionQueue::open`
/// counting a session that is gone (the creates waiting on the pool would never be woken for them).
struct HeldSession<'a> {
    backend: &'a NativeBackend,
    /// Opened through U1 (its sessions bypass the CR1 queue), so released straight to the driver.
    u1: bool,
    session: Option<Session>,
}

impl HeldSession<'_> {
    fn release(&mut self) -> Result<(), DriverError> {
        match self.session.take() {
            None => Ok(()),
            Some(s) if self.u1 => self.backend.driver.release_session(s),
            Some(s) => self.backend.sessions.release(&self.backend.driver, s),
        }
    }
}

impl Drop for HeldSession<'_> {
    fn drop(&mut self) {
        // Nothing to report on this path: the outcome is the panic.
        let _ = self.release();
    }
}

/// ApiFit AF-L7: where a budgeted choice's thinking is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Thinking {
    /// Watched: tokens past the budget's keep are held.
    Open,
    /// The budget was reached without `</think>`: the ticket is being cancelled, its remaining tokens dropped.
    Reached,
    /// Ended (by `</think>`, a stop token, or a cut): everything passes through.
    Closed,
}

/// ApiFit AF-L7 (api-fit design note): a choice's `thinking_budget` B, the most output tokens its thinking may
/// take, `</think>` included. Tokens 1..=B-c (c: the closing tokens counted as thinking) pass through as they
/// commit; the next c are held until `</think>` or a stop token releases them (the thinking ended within the
/// budget: the output is the unbudgeted one) or token B arrives without one (the cut).
pub(crate) struct ThinkingBudget {
    tokens: u64,
    think_end: u32,
    /// The template's closing tokens: the thinking part (ending with `</think>`), then the token after it.
    closure: Vec<u32>,
    /// How many of `closure` are thinking (count against the budget).
    closing: u64,
    ignore_eos: bool,
    state: Thinking,
    /// Output tokens seen while the thinking was open.
    seen: u64,
    /// The thinking tokens passed through while open (the cut's continuation extends the prompt with them).
    kept: Vec<u32>,
    /// Seen, not yet passed through.
    held: Vec<u32>,
}

impl ThinkingBudget {
    /// `None` without a `</think>` token: the budget cannot be closed.
    fn new(vocab: &Vocab, tokens: u64, ignore_eos: bool) -> Option<Self> {
        let think_end = vocab.think_end()?;
        let (mut closure, after) = vocab.thinking_closure(tokens)?;
        let closing = closure.len() as u64;
        closure.extend(after);
        Some(ThinkingBudget {
            tokens,
            think_end,
            closure,
            closing,
            ignore_eos,
            state: Thinking::Open,
            seen: 0,
            kept: Vec::new(),
            held: Vec::new(),
        })
    }

    /// Relays one ticket's progress `rx` to `progress` (logprobs are refused with a budget, `validate.rs`). Reaching
    /// the budget cancels the ticket.
    fn run(
        &mut self,
        driver: &Driver,
        job: &Job,
        mut rx: mpsc::UnboundedReceiver<Committed>,
        progress: &Progress,
        stop_token_ids: &[u32],
    ) {
        let keep = self.tokens - self.closing;
        while let Some(c) = rx.blocking_recv() {
            debug_assert!(
                c.logprobs.is_empty(),
                "thinking_budget is refused with logprobs"
            );
            let mut forward = Vec::with_capacity(c.tokens.len());
            for &token in &c.tokens {
                match self.state {
                    Thinking::Closed => forward.push(token),
                    Thinking::Reached => break,
                    Thinking::Open => {
                        self.seen += 1;
                        self.held.push(token);
                        let stops = !self.ignore_eos && stop_token_ids.contains(&token);
                        if token == self.think_end || stops {
                            self.state = Thinking::Closed;
                            forward.append(&mut self.held);
                        } else if self.seen >= self.tokens {
                            self.state = Thinking::Reached;
                            self.held.clear();
                            let _ = driver.cancel(job.ticket());
                        } else {
                            // Held positions are seen-len+1..=seen; those within keep pass through.
                            let first = self.seen + 1 - self.held.len() as u64;
                            let ready =
                                (keep + 1).saturating_sub(first).min(self.held.len() as u64);
                            let passed: Vec<u32> = self.held.drain(..ready as usize).collect();
                            self.kept.extend_from_slice(&passed);
                            forward.extend(passed);
                        }
                    }
                }
            }
            if !forward.is_empty() {
                let _ = progress.send(Committed {
                    tokens: forward,
                    logprobs: Vec::new(),
                    drafts: None,
                });
            }
        }
    }

    /// The held tokens of a generation that ended with its thinking open.
    fn flush(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.held)
    }
}

/// The serve loop's grammar latencies (spec m28 D8, `GET /v1/cere/stats` `grammar`): `grammar_step` wall time and
/// the Constrain->step gap (from `Driver::constrain` to the job's next committed report), each over its last
/// [`GRAMMAR_SAMPLES`] samples.
#[derive(Default)]
pub(crate) struct GrammarStats {
    step_ms: Mutex<Samples>,
    gap_ms: Mutex<Samples>,
}

/// How many recent samples each grammar latency (and each A/B arm summary, `crate::ab`) keeps.
const GRAMMAR_SAMPLES: usize = 4096;

/// A ring of the last [`GRAMMAR_SAMPLES`] values.
#[derive(Default)]
pub(crate) struct Samples {
    values: Vec<f64>,
    next: usize,
}

impl Samples {
    pub(crate) fn push(&mut self, value: f64) {
        if self.values.len() < GRAMMAR_SAMPLES {
            self.values.push(value);
        } else {
            self.values[self.next] = value;
        }
        self.next = (self.next + 1) % GRAMMAR_SAMPLES;
    }

    /// p50 and p99 (nearest rank); `null` without samples.
    pub(crate) fn percentiles(&self) -> (Value, Value) {
        if self.values.is_empty() {
            return (Value::Null, Value::Null);
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(f64::total_cmp);
        let rank =
            |p: f64| sorted[((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1];
        (json!(rank(0.50)), json!(rank(0.99)))
    }
}

impl GrammarStats {
    fn lock(samples: &Mutex<Samples>) -> MutexGuard<'_, Samples> {
        samples.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn step(&self, elapsed: Duration) {
        Self::lock(&self.step_ms).push(elapsed.as_secs_f64() * 1e3);
    }

    fn gap(&self, elapsed: Duration) {
        Self::lock(&self.gap_ms).push(elapsed.as_secs_f64() * 1e3);
    }

    /// The `grammar` object: `step_ms_p50`, `step_ms_p99`, `constrain_gap_ms_p50`, `constrain_gap_ms_p99`.
    pub(crate) fn render(&self) -> Value {
        let (step_p50, step_p99) = Self::lock(&self.step_ms).percentiles();
        let (gap_p50, gap_p99) = Self::lock(&self.gap_ms).percentiles();
        json!({
            "step_ms_p50": step_p50,
            "step_ms_p99": step_p99,
            "constrain_gap_ms_p50": gap_p50,
            "constrain_gap_ms_p99": gap_p99,
        })
    }
}

/// A constrained choice's grammar in the generation thread (spec m28 D8): the open grammar, whose `mask` is that
/// of the next ticket's position 0, and the termination the emitter reads.
pub(crate) struct Steer {
    open: OpenGrammar,
    stop: GrammarStop,
    ignore_eos: bool,
}

/// `(output index, token)` of a grammar termination on a token the executor did not stop on.
type GrammarStop = Arc<std::sync::OnceLock<(u64, u32)>>;

impl Steer {
    /// Feeds `job`'s committed tokens (`rx`, until the job's terminal poll closes it) to the grammar on this
    /// blocking thread, then forwards them to the emitter's `progress`. A DECODE token's mask goes to the job's
    /// buffer; after each report, `Constrain(job, committed)` sends the last one ([`Steer::constrain`]: a
    /// speculation round commits several tokens, GrammarSpec GS-02). The PREFILL token's mask stays in `open.mask`
    /// for the DECODE submit, so no `Constrain` targets a PREFILL. `before`: the choice's tokens committed by
    /// earlier tickets.
    ///
    /// Exceptions cancel the ticket (a ticket awaiting its mask then reaches `CERE_FINISH_CANCELLED`): a grammar
    /// failure or a rejected token is returned as the choice's 500; a termination on a token the executor does
    /// not stop on (always under `ignore_eos`, D15) is recorded in `stop` before that token is forwarded.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        backend: &NativeBackend,
        job: &Job,
        prefill: bool,
        before: u64,
        mut rx: mpsc::UnboundedReceiver<Committed>,
        progress: &Progress,
        tickets: &TicketSettings,
    ) -> Option<BackendError> {
        let driver = &backend.driver;
        let grammar = backend
            .grammar
            .as_ref()
            .expect("an open grammar has a grammar process");
        let mut committed = 0u64;
        let mut failure: Option<BackendError> = None;
        let mut constrained: Option<Instant> = None;
        while let Some(c) = rx.blocking_recv() {
            if let Some(at) = constrained.take() {
                backend.grammar_stats.gap(at.elapsed());
            }
            // The last token stepped left the mask of the next position.
            let mut masked = false;
            for &token in &c.tokens {
                committed += 1;
                if failure.is_some() || self.stop.get().is_some() {
                    continue;
                }
                masked = false;
                let started = Instant::now();
                let step = match job.mask() {
                    Some(mut buffer) if !prefill => {
                        grammar.grammar_step(&self.open.handle, token, &mut buffer)
                    }
                    _ => grammar.grammar_step(&self.open.handle, token, &mut self.open.mask),
                };
                backend.grammar_stats.step(started.elapsed());
                let ended = match step {
                    Err(e) => Some(Err(format!("grammar_step failed: {e}"))),
                    Ok(s) if !s.accepted => {
                        eprintln!("cered: the grammar rejected committed token {token}");
                        Some(Err("the grammar rejected a committed token".to_owned()))
                    }
                    Ok(s) if s.terminated => {
                        let stops = !self.ignore_eos && tickets.stop_token_ids.contains(&token);
                        (!stops).then_some(Ok(()))
                    }
                    Ok(s) if !s.has_mask => Some(Err("a grammar step without a mask".to_owned())),
                    Ok(_) => {
                        masked = true;
                        None
                    }
                };
                match ended {
                    None => {}
                    Some(Ok(())) => {
                        let _ = self.stop.set((before + committed - 1, token));
                        let _ = driver.cancel(job.ticket());
                    }
                    Some(Err(message)) => {
                        failure = Some(BackendError::Internal(message));
                        let _ = driver.cancel(job.ticket());
                    }
                }
            }
            if masked && !prefill && failure.is_none() && self.stop.get().is_none() {
                match self.constrain(backend, job, committed, c.drafts.as_ref()) {
                    Ok(()) => constrained = Some(Instant::now()),
                    Err(message) => {
                        failure = Some(BackendError::Internal(message));
                        let _ = driver.cancel(job.ticket());
                    }
                }
            }
            let _ = progress.send(c);
        }
        // A U1 preemption resubmits the DECODE: its position 0 is the job's pending mask (the buffer's first).
        if !prefill {
            if let Some(buffer) = job.mask() {
                self.open.mask.clear();
                self.open
                    .mask
                    .extend_from_slice(&buffer[..grammar.info().mask_words()]);
            }
        }
        failure
    }

    /// `Constrain(job, position)` with the mask of `position` in the job's buffer. With the round's frozen
    /// `drafts` (GrammarSpec GS-02), `grammar_drafts` appends the mask after each leading draft the grammar
    /// accepts (the grammar's state unchanged) and `constrain_drafts` keeps those drafts; the executor verifies
    /// them under those masks and counts the rest as rejected.
    fn constrain(
        &self,
        backend: &NativeBackend,
        job: &Job,
        position: u64,
        drafts: Option<&Result<RoundDrafts, DriverError>>,
    ) -> Result<(), String> {
        let driver = &backend.driver;
        let drafts = match drafts {
            Some(Ok(drafts)) => drafts.as_slice(),
            Some(Err(e)) => return Err(format!("round_drafts failed: {e}")),
            None => &[],
        };
        if drafts.is_empty() {
            return driver
                .constrain(job, position)
                .map_err(|e| format!("constrain failed: {e}"));
        }
        let grammar = backend
            .grammar
            .as_ref()
            .expect("an open grammar has a grammar process");
        let accepted = {
            let mut buffer = job.mask().expect("a constrained job has a mask buffer");
            grammar.grammar_drafts(&self.open.handle, drafts, &mut buffer)
        }
        .map_err(|e| format!("grammar_drafts failed: {e}"))?;
        driver
            .constrain_drafts(job, position, &drafts[..accepted])
            .map_err(|e| format!("constrain failed: {e}"))
    }
}

/// U1 preemption of one backend (D8): the [`Sched`] bookkeeping, the cancel handles of the generations it may
/// preempt, and a condition variable the waiting generation threads poll on.
pub(crate) struct Preemption {
    state: Mutex<U1State>,
    changed: Condvar,
    next_id: AtomicU64,
}

struct U1State {
    sched: Sched,
    cancels: HashMap<u64, Weak<ChoiceCancel>>,
}

/// How long a waiting generation sleeps before it looks at its cancel and the slots again.
const U1_POLL: Duration = Duration::from_millis(20);

/// Ends the generation's U1 bookkeeping on every path (after its session and snapshot are released).
struct Leave<'a>(&'a Preemption, u64);

impl Drop for Leave<'_> {
    fn drop(&mut self) {
        self.0.update(|s| {
            s.sched.ended(self.1);
            s.cancels.remove(&self.1);
        });
    }
}

/// How a [`Preemption::park`] ended.
enum Park {
    /// Snapshotted, released, then restored into a fresh session of the asked context.
    Resumed(Session),
    /// The snapshot was refused (CAPACITY/UNSUPPORTED), or a grow's wait was cancelled: the session is unchanged.
    Kept(Session),
    /// Cancelled while parked: nothing is held.
    Cancelled,
}

impl Park {
    fn session(self) -> Option<Session> {
        match self {
            Park::Resumed(s) | Park::Kept(s) => Some(s),
            Park::Cancelled => None,
        }
    }
}

fn capacity(e: &DriverError) -> bool {
    matches!(
        e,
        DriverError::Status {
            status: StatusError::Capacity,
            ..
        }
    )
}

impl Preemption {
    pub(crate) fn new(policy: Policy) -> Self {
        Preemption {
            state: Mutex::new(U1State {
                sched: Sched::new(policy),
                cancels: HashMap::new(),
            }),
            changed: Condvar::new(),
            next_id: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, U1State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Applies `f` and wakes every waiting generation.
    fn update<R>(&self, f: impl FnOnce(&mut U1State) -> R) -> R {
        let r = f(&mut self.lock());
        self.changed.notify_all();
        r
    }

    /// Sleeps until the bookkeeping changes or [`U1_POLL`] passes.
    fn wait(&self) {
        let guard = self.lock();
        let _ = self.changed.wait_timeout(guard, U1_POLL);
    }

    /// Preemptions done so far (`native_preemptions`).
    pub(crate) fn preemptions(&self) -> u64 {
        self.lock().sched.preemptions()
    }

    /// SE-09 grows done so far (`native_grows`).
    pub(crate) fn grows(&self) -> u64 {
        self.lock().sched.grows()
    }

    /// SE-09 `reserve_output_tokens` (0 = a session holds its whole budget).
    fn reserve(&self) -> u64 {
        self.lock().sched.reserve()
    }

    fn enqueue(&self, cancel: &Arc<ChoiceCancel>) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.update(|s| {
            s.sched.enqueue(id);
            s.cancels.insert(id, Arc::downgrade(cancel));
        });
        id
    }

    /// `n` more tokens committed by `id` (the counts a competing request's victim is chosen by).
    fn committed(&self, id: u64, n: u64) {
        self.lock().sched.committed(id, n);
    }

    /// `preempt_every` (D13); 0 = off.
    fn every(&self) -> u64 {
        self.lock().sched.every()
    }

    /// D13: `id` is at a `preempt_every` boundary with `committed` tokens and nothing in flight; whether it parks
    /// now ([`Sched::force`]).
    fn force(&self, id: u64, committed: u64) -> bool {
        self.lock().sched.force(id, committed)
    }

    /// A session of `context` tokens (ledger CR1) for the fresh generation `id`: its turn in the queue, then
    /// `cere_session_create`; on CAPACITY a victim is preempted for it, or it waits. `None` when it was cancelled
    /// first.
    fn acquire(
        &self,
        backend: &NativeBackend,
        id: u64,
        context: u64,
        prompt_tokens: u64,
        cancel: &ChoiceCancel,
    ) -> Result<Option<Session>, DriverError> {
        let driver = &backend.driver;
        loop {
            if cancel.cancelled() {
                return Ok(None);
            }
            if self.lock().sched.may_create(id) {
                match driver.create_session(&backend.model, context) {
                    Ok(s) => {
                        self.update(|st| st.sched.created(id, prompt_tokens));
                        return Ok(Some(s));
                    }
                    Err(e) if capacity(&e) => {
                        let victim = self.update(|st| match st.sched.no_slot(id) {
                            NoSlot::Preempt(v) => st.cancels.get(&v).and_then(Weak::upgrade),
                            NoSlot::Wait => None,
                        });
                        if let Some(victim) = victim {
                            victim.preempt();
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            self.wait();
        }
    }

    /// SE-09: the generation `id` used its session of `prompt + committed` = `position` tokens up with budget left.
    /// It parks itself and resumes in a session of `context` tokens (its whole budget); while the park list or the
    /// arena is full, or the snapshot is refused, it waits and asks again. [`Park::Kept`] only when cancelled first.
    fn grow(
        &self,
        backend: &NativeBackend,
        id: u64,
        context: u64,
        position: u64,
        session: Session,
        cancel: &ChoiceCancel,
    ) -> Result<Park, DriverError> {
        let mut session = session;
        loop {
            if cancel.cancelled() {
                return Ok(Park::Kept(session));
            }
            if self.update(|st| st.sched.grow(id, position)) {
                match self.park(backend, id, context, session, cancel, false)? {
                    Park::Kept(s) => session = s,
                    done => return Ok(done),
                }
            }
            self.wait();
        }
    }

    /// Parks the victim `id`: snapshot, release, then (FIFO, before any new grant) create a session of `context`
    /// (ledger CR1: the same page reservation; SE-09: a grow's whole budget) + restore. A refused snapshot
    /// (CAPACITY/UNSUPPORTED) keeps the session: restore-free resume, as does an adapter session (`lora`, track
    /// LoRA: never snapshotted, the executor refuses it UNSUPPORTED). On an error nothing is left held.
    fn park(
        &self,
        backend: &NativeBackend,
        id: u64,
        context: u64,
        session: Session,
        cancel: &ChoiceCancel,
        lora: bool,
    ) -> Result<Park, DriverError> {
        let driver = &backend.driver;
        if lora {
            self.update(|st| st.sched.kept(id));
            return Ok(Park::Kept(session));
        }
        let snapshot = match driver.snapshot(&session) {
            Ok(s) => s,
            Err(DriverError::Status {
                status: StatusError::Capacity | StatusError::Unsupported,
                ..
            }) => {
                self.update(|st| st.sched.kept(id));
                return Ok(Park::Kept(session));
            }
            Err(e) => {
                let _ = driver.release_session(session);
                return Err(e);
            }
        };
        let bytes = match driver.snapshot_info(&snapshot) {
            Ok(info) => info.bytes,
            Err(e) => {
                let _ = driver.release_snapshot(snapshot);
                let _ = driver.release_session(session);
                return Err(e);
            }
        };
        if let Err(e) = driver.release_session(session) {
            let _ = driver.release_snapshot(snapshot);
            return Err(e);
        }
        self.update(|st| st.sched.parked(id, bytes));
        loop {
            if cancel.cancelled() {
                driver.release_snapshot(snapshot)?;
                return Ok(Park::Cancelled);
            }
            if self.lock().sched.may_resume(id) {
                match driver.create_session(&backend.model, context) {
                    Ok(s) => match driver.restore(&s, &snapshot) {
                        Ok(()) => {
                            self.update(|st| st.sched.resumed(id));
                            if let Err(e) = driver.release_snapshot(snapshot) {
                                let _ = driver.release_session(s);
                                return Err(e);
                            }
                            return Ok(Park::Resumed(s));
                        }
                        Err(e) if capacity(&e) => {
                            let released = driver.release_session(s);
                            self.update(|st| st.sched.restore_blocked(id));
                            if let Err(e) = released {
                                let _ = driver.release_snapshot(snapshot);
                                return Err(e);
                            }
                        }
                        Err(e) => {
                            let _ = driver.release_session(s);
                            let _ = driver.release_snapshot(snapshot);
                            return Err(e);
                        }
                    },
                    Err(e) if capacity(&e) => {}
                    Err(e) => {
                        let _ = driver.release_snapshot(snapshot);
                        return Err(e);
                    }
                }
            }
            self.wait();
        }
    }
}

/// The cered side of ledger CR1 (m25 D3 and N2 (a), m26 D4 and N1 (d)): a session reserves its KV pages at
/// `cere_session_create`, which answers CAPACITY with nothing kept when the pool is short, so no step ever lacks a
/// page. Such a create waits here, in arrival order, until a session of this backend is released (or its choice is
/// cancelled), then tries again: the request is queued in cered, not failed. A CAPACITY while no session of this
/// backend is open or being created cannot be waited out and is returned. While no create waits on CAPACITY, creates
/// call the executor at once, concurrently (BcAdmit, O282); while one does, the others wait behind it in the FIFO.
#[derive(Default)]
pub(crate) struct SessionQueue {
    state: Mutex<QueueState>,
    changed: Condvar,
    /// Sessions created here and not yet released. An atomic, so [`SessionQueue::load`] (the operator's cards
    /// route, on an async worker) never waits on the queue's mutex (audit N2).
    open: AtomicU64,
    /// Creates waiting in the FIFO: `QueueState::waiting.len()`, mirrored for [`SessionQueue::load`].
    waiting: AtomicU64,
}

#[derive(Default)]
struct QueueState {
    /// Waiting creates in arrival order. While `blocked` is 0 any of them calls the executor; otherwise only the
    /// front one does.
    waiting: VecDeque<u64>,
    next: u64,
    /// Sessions released so far: a create that ran without the lock sees whether a release landed meanwhile.
    released: u64,
    /// Creates in the FIFO that the executor answered CAPACITY.
    blocked: u64,
    /// Creates inside the executor call: one of them may open a session whose release a CAPACITY can wait for.
    calling: u64,
}

/// A create's place in the FIFO, left when it drops, on every path out of [`SessionQueue::admit`] (a panic of the
/// executor call included: the entry would otherwise stay at the front and stall every later create, and its
/// `blocked` or `calling` count would stay raised).
struct Waiting<'q> {
    queue: &'q SessionQueue,
    id: u64,
    /// Counted in `QueueState::blocked`.
    blocked: bool,
    /// Counted in `QueueState::calling`.
    calling: bool,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        let mut state = self.queue.lock();
        state.waiting.retain(|w| *w != self.id);
        state.blocked -= u64::from(self.blocked);
        state.calling -= u64::from(self.calling);
        self.queue
            .waiting
            .store(state.waiting.len() as u64, Ordering::Release);
        self.queue.changed.notify_all();
    }
}

impl SessionQueue {
    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// m32n design note D18: sessions open on the card and creates waiting in its FIFO.
    pub(crate) fn load(&self) -> (u64, u64) {
        (
            self.open.load(Ordering::Acquire),
            self.waiting.load(Ordering::Acquire),
        )
    }

    /// Wakes the waiting creates (a cancel may concern one of them).
    fn wake(&self) {
        let _state = self.lock();
        self.changed.notify_all();
    }

    /// A session of `context` tokens once the pool covers it; `None` if `cancel` fires first. Blocking: call it
    /// off the async runtime. The executor call runs without the queue's mutex (audit N2): a `cere_session_create`
    /// is a round trip to the driver thread, and the mutex is what `wake` and `release` take.
    fn create(
        &self,
        driver: &Driver,
        model: &Model,
        context: u64,
        cancel: &ChoiceCancel,
    ) -> Result<Option<Session>, DriverError> {
        self.admit(
            || cancel.cancelled(),
            || driver.create_session(model, context),
        )
    }

    /// [`SessionQueue::create`] with the executor call `create` (the tests stand a fake pool in for the driver).
    fn admit<S>(
        &self,
        cancelled: impl Fn() -> bool,
        mut create: impl FnMut() -> Result<S, DriverError>,
    ) -> Result<Option<S>, DriverError> {
        let id = {
            let mut state = self.lock();
            let id = state.next;
            state.next += 1;
            state.waiting.push_back(id);
            self.waiting
                .store(state.waiting.len() as u64, Ordering::Release);
            id
        };
        // Declared before `state`, so it drops after the guard it locks.
        let mut waiting = Waiting {
            queue: self,
            id,
            blocked: false,
            calling: false,
        };
        let mut state = self.lock();
        loop {
            if cancelled() {
                return Ok(None);
            }
            // BcAdmit (O282, batch-concurrency design note): the FIFO orders creates only while one waits on
            // CAPACITY (the arrival order of pages that come back, which no later create may jump); otherwise every
            // create calls the executor at once. Before, only the front called, so a burst of N creates with free
            // pages reached the driver thread one round trip apart (each behind the previous one's reply, and that
            // reply behind any round the thread ran first), and their PREFILL submits never queued together.
            let turn =
                state.waiting.front() == Some(&id) || (!waiting.blocked && state.blocked == 0);
            if turn {
                let released = state.released;
                state.calling += 1;
                waiting.calling = true;
                drop(state);
                let created = create();
                state = self.lock();
                state.calling -= 1;
                waiting.calling = false;
                match created {
                    Ok(session) => {
                        self.open.fetch_add(1, Ordering::AcqRel);
                        return Ok(Some(session));
                    }
                    Err(e) if capacity(&e) => {
                        if !waiting.blocked {
                            waiting.blocked = true;
                            state.blocked += 1;
                        }
                        // Pages came back while the executor was asked: ask again (at the front), no release will
                        // wake us for them.
                        if state.released != released {
                            continue;
                        }
                        // Nothing open and no create in flight that may open a session: no release will come.
                        if self.open.load(Ordering::Acquire) == 0 && state.calling == 0 {
                            return Err(e);
                        }
                    }
                    Err(e) => return Err(e),
                }
                // A cancel whose wake came while the executor was asked (the mutex dropped) found no waiter, and
                // only a release wakes this wait again: check before sleeping.
                if cancelled() {
                    return Ok(None);
                }
            }
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Releases `session` (its pages return to the pool) and wakes the waiting creates.
    fn release(&self, driver: &Driver, session: Session) -> Result<(), DriverError> {
        let released = driver.release_session(session);
        self.released();
        released
    }

    /// Counts a session of this queue released and wakes the waiting creates.
    fn released(&self) {
        let mut state = self.lock();
        state.released += 1;
        self.open.fetch_sub(1, Ordering::AcqRel);
        self.changed.notify_all();
    }
}

/// The client sees only `backend_error`; the call and status (never request text) go to cered's log.
fn driver_error(error: &DriverError) -> BackendError {
    eprintln!("cered: native executor failed: {error}");
    BackendError::Status {
        status: 500,
        message: format!("native executor: {error}"),
    }
}

/// A failed tool value conversion: the stream ends with a backend error (D9; SGLang's own outcome varies with
/// the exception type, `serving_chat.py:1723-1727`).
fn tool_error(error: &SidecarError) -> BackendError {
    BackendError::Status {
        status: 500,
        message: format!("native tool parser: {error}"),
    }
}

/// A new tool call id: `call_` + 24 hex digits, as SGLang's `uuid4().hex[:24]` (`serving_chat.py:1970-1989`).
fn call_id() -> Result<String, BackendError> {
    digest::random_bytes::<12>()
        .map(|b| format!("call_{}", digest::hex(&b)))
        .map_err(|e| BackendError::Status {
            status: 500,
            message: format!("native tool call id: {e}"),
        })
}

/// SGLang's `usage.reasoning_tokens` (spec m212 D7, `Req.update_reasoning_tokens`, `schedule_batch.py:1823`):
/// with thinking resolved on and a `</think>` token, every output token up to and including the first `</think>`
/// (all of them while it has not appeared); 0 with thinking off or no `</think>` in the vocabulary
/// (`batch_result_processor.py:1109`).
#[derive(Debug)]
pub(crate) struct ReasoningTokens {
    think_end: Option<u32>,
    over: bool,
    count: u64,
}

impl ReasoningTokens {
    pub(crate) fn new(thinking: bool, think_end: Option<u32>) -> Self {
        Self {
            think_end: think_end.filter(|_| thinking),
            over: false,
            count: 0,
        }
    }

    /// One committed output token.
    pub(crate) fn push(&mut self, id: u32) {
        let Some(end) = self.think_end else { return };
        if self.over {
            return;
        }
        self.count += 1;
        self.over = id == end;
    }
}

/// Turns one choice's committed tokens into SGLang chunks on the reader's channel (a gone reader is ignored: the
/// generation still runs to its terminal poll).
struct Emitter<'v> {
    tx: mpsc::Sender<BackendItem>,
    head: Arc<Head>,
    /// The choice's `index`.
    index: usize,
    backend: Arc<NativeBackend>,
    cancel: Arc<ChoiceCancel>,
    /// Every choice of the request: a failed choice cancels them all.
    group: Arc<NativeCancel>,
    /// The response text of the tokens the matcher ruled on, as decoded for this request (D11).
    text: Text<'v>,
    matcher: StopMatcher<'v>,
    /// How much of `text` went to the parsers (SSE).
    sent: usize,
    parse: Parse,
    /// The unary parser's mode (D4).
    reasoning: Reasoning,
    /// `usage.reasoning_tokens` over the tokens the matcher ruled on (D7).
    reasoning_tokens: ReasoningTokens,
    tools: Option<Arc<Vec<Value>>>,
    /// A tool call went out: a `stop` finish becomes `tool_calls`.
    called: bool,
    /// The output path failed; the generation is being cancelled.
    failed: Option<BackendError>,
    /// `logprobs: true` (m27 D7): the rendered entries not yet sent (SSE: the current step's; unary: all), else
    /// `None`.
    logprobs: Option<Vec<Value>>,
    /// [`Choice::grammar_stop`].
    grammar_stop: Option<GrammarStop>,
    /// loop-guard design note LG-D1: the repetition check of the output text, with the loop guard on.
    repetition: Option<Repetition>,
}

/// How the decoded text is split into reasoning, content and tool calls.
enum Parse {
    /// SSE without the tool parser: the reasoning parser, piece by piece.
    Plain(reasoning::Stream),
    /// SSE with the tool parser (D10).
    Tools(Chain),
    /// Unary: the whole text at the end.
    Whole,
}

/// An SGLang `chat.completion.chunk` named by `head`.
fn chunk(head: &Head, choices: Value, usage: Value) -> Value {
    json!({
        "id": head.id,
        "object": "chat.completion.chunk",
        "created": head.created,
        "model": head.model,
        "choices": choices,
        "usage": usage,
    })
}

impl Emitter<'_> {
    /// One content chunk; `attach` = it takes the step's logprobs not yet sent (SGLang attaches them to the step's
    /// first reasoning or content chunk, never to a tool parser's chunk: `serving_chat.py:694-775`).
    async fn delta(&mut self, delta: Value, attach: bool) {
        let logprobs = if attach {
            self.take_logprobs()
        } else {
            Value::Null
        };
        self.send(delta, logprobs).await;
    }

    async fn send(&mut self, delta: Value, logprobs: Value) {
        let chunk = chunk(
            &self.head,
            json!([{
                "index": self.index,
                "delta": delta,
                "logprobs": logprobs,
                "finish_reason": null,
                "matched_stop": null,
            }]),
            Value::Null,
        );
        let _ = self.tx.send(Ok(Some(chunk))).await;
    }

    /// The pending entries as `{content: [...]}`, `null` when there are none.
    fn take_logprobs(&mut self) -> Value {
        match &mut self.logprobs {
            Some(pending) if !pending.is_empty() => json!({ "content": std::mem::take(pending) }),
            _ => Value::Null,
        }
    }

    /// The end of an SSE step: logprobs no chunk of the step took go out on their own chunk (the reasoning parser
    /// always runs on native, so SGLang's flush applies: `serving_chat.py:777-805`).
    async fn flush_logprobs(&mut self) {
        let logprobs = self.take_logprobs();
        if !logprobs.is_null() {
            let delta = json!({
                "role": null,
                "content": null,
                "reasoning_content": null,
                "tool_calls": null,
            });
            self.send(delta, logprobs).await;
        }
    }

    async fn split(&mut self, split: Split) {
        if split.reasoning.is_empty() && split.normal.is_empty() {
            return;
        }
        // With the tool parser only its reasoning chunk is SGLang's own (`serving_chat.py:707-728`).
        let attach = match self.parse {
            Parse::Plain(_) => true,
            Parse::Tools(_) => split.normal.is_empty(),
            Parse::Whole => false,
        };
        let text = |s: String| if s.is_empty() { Value::Null } else { json!(s) };
        let delta = json!({
            "role": null,
            "content": text(split.normal),
            "reasoning_content": text(split.reasoning),
            "tool_calls": null,
        });
        self.delta(delta, attach).await;
    }

    async fn tool_calls(&mut self, calls: Value) {
        self.called = true;
        let delta = json!({
            "role": null,
            "content": null,
            "reasoning_content": null,
            "tool_calls": calls,
        });
        self.delta(delta, false).await;
    }

    /// Stops the output path with `error`; every choice of the request is cancelled and this one's end reports
    /// `error`.
    fn fail(&mut self, error: BackendError) {
        if self.failed.is_none() {
            self.failed = Some(error);
            let group = self.group.clone();
            tokio::task::spawn_blocking(move || group.cancel_for(CancelReason::Failed, None));
        }
    }

    /// The stop check per committed token (D11): every token goes to the response text with the matcher's verdict
    /// ([`Text`]); unless the window ends with a stop prefix, the text not yet sent goes to the parsers (SSE). A
    /// stop string cancels the generation, as does LG-D1 (loop-guard design note: a running output whose text
    /// now ends repeating finishes with `repetition` on this token). Each token the matcher rules on keeps its
    /// logprobs entry (D7); an SSE step is one token, and the finishing token's step is the terminal piece
    /// ([`Emitter::finish`]).
    async fn tokens(&mut self, committed: &Committed) {
        let stream = !matches!(self.parse, Parse::Whole);
        for (k, &id) in committed.tokens.iter().enumerate() {
            if self.failed.is_some() || self.matcher.finished().is_some() {
                return;
            }
            // A grammar termination on this token (D8) finishes it as a stop token, unless the matcher finishes
            // the output here anyway.
            let at = self.matcher.counted();
            let step = match self.grammar_stop.as_ref().and_then(|s| s.get()) {
                Some(&(index, token)) if index == at && token == id => self.matcher.push_stop(id),
                _ => self.matcher.push(id),
            };
            self.reasoning_tokens.push(id);
            self.text.push(id, &step);
            if let (Some(pending), Some(lp)) = (&mut self.logprobs, committed.logprobs.get(k)) {
                pending.push(logprob_entry(&self.backend.vocab, lp, stream));
            }
            match step {
                Step::Running { should_output } => {
                    if self
                        .repetition
                        .as_mut()
                        .and_then(|r| r.check(self.text.as_str()))
                        .is_some()
                    {
                        self.matcher.finish_repetition();
                        if let Some(guard) = &self.backend.loop_guard {
                            guard.stopped();
                        }
                        let cancel = self.cancel.clone();
                        tokio::task::spawn_blocking(move || {
                            cancel.cancel_for(CancelReason::Repetition)
                        });
                        continue;
                    }
                    if stream {
                        if should_output {
                            let piece = self.text.as_str()[self.sent..].to_owned();
                            self.sent = self.text.as_str().len();
                            self.piece(&piece, false).await;
                        }
                        self.flush_logprobs().await;
                    }
                }
                Step::Finished(finished) => {
                    if matches!(finished.matched, Matched::Str(_)) {
                        let cancel = self.cancel.clone();
                        tokio::task::spawn_blocking(move || {
                            cancel.cancel_for(CancelReason::StopString)
                        });
                    }
                }
            }
        }
    }

    /// One piece of SSE text through the parsers; `last` = the terminal piece of a finished output (the
    /// parsers' end flush follows).
    async fn piece(&mut self, piece: &str, last: bool) {
        match &mut self.parse {
            Parse::Plain(stream) => {
                let split = if piece.is_empty() {
                    Split::default()
                } else {
                    stream.push(piece)
                };
                let end = if last {
                    stream.finish()
                } else {
                    Split::default()
                };
                self.split(split).await;
                self.split(end).await;
            }
            Parse::Tools(chain) => {
                if piece.is_empty() && !last {
                    return;
                }
                let step = chain.push(piece, last);
                self.chain_step(step).await;
            }
            Parse::Whole => {}
        }
    }

    /// SGLang's order (`serving_chat.py:700-748`, :2425-2514): reasoning, then normal text, then one chunk per
    /// call item; a parameter's fragment is the sidecar's `tool_param` on the prepared tool.
    async fn chain_step(&mut self, step: ChainStep) {
        self.split(Split {
            reasoning: step.reasoning,
            normal: String::new(),
        })
        .await;
        self.split(Split {
            reasoning: String::new(),
            normal: step.normal,
        })
        .await;
        for item in step.items {
            if self.failed.is_some() {
                return;
            }
            let delta = if matches!(item, Item::Param { .. }) {
                let b = self.backend.clone();
                let tools = self.tools.clone().unwrap_or_default();
                tokio::task::spawn_blocking(move || {
                    item.resolve(|func, name, raw| {
                        b.sidecar
                            .tool_param(tool_parser::tool_for(&tools, func), func, name, raw)
                    })
                })
                .await
                .unwrap_or_else(|_| Err(SidecarError::Gone("tool task failed".to_owned())))
            } else {
                item.resolve(|_, _, _| Ok(String::new()))
            };
            let delta = match delta {
                Ok(delta) => delta,
                Err(e) => {
                    self.fail(tool_error(&e));
                    return;
                }
            };
            let call = match self.stream_call(delta) {
                Ok(call) => call,
                Err(e) => {
                    self.fail(e);
                    return;
                }
            };
            self.tool_calls(json!([call])).await;
        }
    }

    /// One streamed `ToolCall`: id and name on a call's first item, null after.
    fn stream_call(&self, delta: Delta) -> Result<Value, BackendError> {
        let id = match &delta.name {
            Some(_) => json!(call_id()?),
            None => Value::Null,
        };
        Ok(json!({
            "id": id,
            "index": delta.index,
            "type": "function",
            "function": { "name": delta.name, "arguments": delta.arguments },
        }))
    }

    /// The unary split (D10): reasoning `detect_and_parse`, then, with tools, SGLang's gate and
    /// `parse_non_stream` with the calls' `arguments` from the sidecar's `tool_args` (Inkling: the payload's
    /// `args`, [`inkling::parse_non_stream`]): its `convert_error` is the reference's conversion exception (the
    /// whole text, no calls); any other sidecar failure fails the output.
    async fn whole(&mut self, text: String) {
        let split = split(&text, self.reasoning);
        let Some(tools) = self.tools.clone() else {
            self.split(split).await;
            return;
        };
        let Split { reasoning, normal } = split;
        let b = self.backend.clone();
        let format = tool_parser::Format::of(b.model.family());
        let parsed = tokio::task::spawn_blocking(move || {
            if format == tool_parser::Format::Inkling {
                // The Inkling payload carries its `args` object: no sidecar conversion.
                return Ok(inkling::parse_non_stream(&normal));
            }
            tool_parser::parse_non_stream(&normal, |call| {
                match b.sidecar.tool_args(
                    tool_parser::tool_for(&tools, Some(&call.name)),
                    &call.name,
                    &call.params,
                ) {
                    Ok(arguments) => Ok(Some(arguments)),
                    Err(SidecarError::Refused { kind, .. }) if kind == "convert_error" => Ok(None),
                    Err(e) => Err(e),
                }
            })
        })
        .await
        .unwrap_or_else(|_| Err(SidecarError::Gone("tool task failed".to_owned())));
        let (content, calls) = match parsed {
            Ok(parsed) => parsed,
            Err(e) => {
                self.fail(tool_error(&e));
                return;
            }
        };
        self.split(Split {
            reasoning,
            normal: content,
        })
        .await;
        if calls.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(calls.len());
        for Call {
            index,
            name,
            arguments,
        } in calls
        {
            let id = match call_id() {
                Ok(id) => id,
                Err(e) => {
                    self.fail(e);
                    return;
                }
            };
            out.push(json!({
                "id": id,
                "index": index,
                "type": "function",
                "function": { "name": name, "arguments": arguments },
            }));
        }
        self.tool_calls(Value::Array(out)).await;
    }

    /// The finish (the matcher's, D11) from the executor's outcome: a client cancel ends without one (abort: no
    /// flush, F8), a stop-string cancel ends with the match.
    fn finished(
        &self,
        outcome: Result<Outcome, DriverError>,
    ) -> Result<Option<Finished>, BackendError> {
        let outcome = outcome.map_err(|e| driver_error(&e))?;
        if let Some(failure) = outcome.failure {
            return Err(failure);
        }
        // A grammar termination (D8) cancels the ticket; the matcher holds that finish.
        let grammar_stop = self.cancel.reason().is_none()
            && self
                .grammar_stop
                .as_ref()
                .is_some_and(|s| s.get().is_some());
        match outcome.finish {
            Finish::Cancelled
                if !matches!(
                    self.cancel.reason(),
                    Some(CancelReason::StopString | CancelReason::Repetition)
                ) && !grammar_stop =>
            {
                Ok(None)
            }
            Finish::Cancelled | Finish::StopToken | Finish::Length => match self.matcher.finished()
            {
                Some(finished) => Ok(Some(finished.clone())),
                None => Err(BackendError::Protocol(format!(
                    "the native executor finished with {} before the stop check did",
                    outcome.finish.name()
                ))),
            },
            other => Err(BackendError::Status {
                status: 500,
                message: format!("native executor finished with {}", other.name()),
            }),
        }
    }

    /// The text not yet sent and the finish chunk: `Ok(Some(completion tokens))`. A cancelled item ends without
    /// a finish chunk (`Ok(None)`; the relay settles it as cancelled). A failed choice cancels the others. SSE: the
    /// finishing step's logprobs ride on its terminal piece or on their own chunk; unary: every entry rides on the
    /// finish chunk, which only [`collect_unary`] reads.
    async fn finish(
        &mut self,
        outcome: Result<Outcome, DriverError>,
    ) -> Result<Option<Tokens>, BackendError> {
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        let cached = outcome.as_ref().map_or(0, |o| u64::from(o.cached));
        let finished = match self.finished(outcome) {
            Ok(finished) => finished,
            Err(e) => {
                if self.group.choices.len() > 1 {
                    let (group, index) = (self.group.clone(), self.index);
                    tokio::task::spawn_blocking(move || {
                        group.cancel_for(CancelReason::Failed, Some(index))
                    });
                }
                return Err(e);
            }
        };
        let Some(finished) = finished else {
            return Ok(None);
        };
        self.text.finish(&finished);
        if matches!(self.parse, Parse::Whole) {
            let text = self.text.take();
            self.whole(text).await;
        } else {
            // The terminal piece: the trimmed text from where SSE left off (none when the trim cut into it).
            let text = self.text.as_str();
            let piece = text.get(self.sent..).unwrap_or_default().to_owned();
            self.sent = text.len();
            self.piece(&piece, true).await;
            self.flush_logprobs().await;
        }
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        let (mut reason, mut matched) = match &finished.matched {
            Matched::Str(stop) => ("stop", json!(stop)),
            Matched::Token(id) => ("stop", json!(id)),
            Matched::Length => ("length", Value::Null),
            Matched::Repetition => ("repetition", Value::Null),
        };
        if self.called && reason == "stop" {
            // `serving_chat.py:1625-1628` (stream, `matched_stop` kept), :2035-2038 (unary, cleared).
            reason = "tool_calls";
            if matches!(self.parse, Parse::Whole) {
                matched = Value::Null;
            }
        }
        let logprobs = match (&self.parse, self.logprobs.take()) {
            (Parse::Whole, Some(all)) => json!({ "content": all }),
            _ => Value::Null,
        };
        let finish = chunk(
            &self.head,
            json!([{
                "index": self.index,
                "delta": { "role": null, "content": null, "reasoning_content": null, "tool_calls": null },
                "logprobs": logprobs,
                "finish_reason": reason,
                "matched_stop": matched,
            }]),
            Value::Null,
        );
        let _ = self.tx.send(Ok(Some(finish))).await;
        let completion_tokens = finished.len as u64;
        Ok(Some(Tokens {
            completion: completion_tokens,
            reasoning: self.reasoning_tokens.count.min(completion_tokens),
            cached,
        }))
    }
}

/// One choice of the unary response, folded from its chunks.
#[derive(Default)]
struct Folded {
    content: String,
    reasoning: String,
    tool_calls: Vec<Value>,
    /// `finish_reason`, `matched_stop` and `logprobs` (the finish chunk's).
    finish: Option<(Value, Value, Value)>,
}

/// The unary response: the chunks of the `n` choices folded into one SGLang `chat.completion` body; a choice's
/// `logprobs` is its finish chunk's.
pub(crate) async fn collect_unary(
    mut chunks: mpsc::Receiver<BackendItem>,
    n: u32,
) -> Result<Value, BackendError> {
    let mut head = None;
    let mut folded: Vec<Folded> = (0..n).map(|_| Folded::default()).collect();
    let mut usage = Value::Null;
    loop {
        let chunk = match chunks.recv().await {
            Some(Ok(Some(chunk))) => chunk,
            Some(Ok(None)) => break,
            Some(Err(e)) => return Err(e),
            None => return Err(BackendError::Protocol("native generation ended".to_owned())),
        };
        if head.is_none() {
            head = Some((
                chunk["id"].clone(),
                chunk["model"].clone(),
                chunk["created"].clone(),
            ));
        }
        if !chunk["usage"].is_null() {
            usage = chunk["usage"].clone();
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            continue;
        };
        let index = choice["index"].as_u64();
        let Some(out) = index
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| folded.get_mut(i))
        else {
            return Err(BackendError::Protocol(format!(
                "native choice index {} outside the {n} choices",
                choice["index"]
            )));
        };
        out.content
            .push_str(choice["delta"]["content"].as_str().unwrap_or(""));
        out.reasoning
            .push_str(choice["delta"]["reasoning_content"].as_str().unwrap_or(""));
        if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
            out.tool_calls.extend(calls.iter().cloned());
        }
        if !choice["finish_reason"].is_null() {
            out.finish = Some((
                choice["finish_reason"].clone(),
                choice["matched_stop"].clone(),
                choice["logprobs"].clone(),
            ));
        }
    }
    let cancelled = || {
        // Cancelled: the executor ended an item without a finish.
        BackendError::Protocol("the native generation was cancelled".to_owned())
    };
    let Some((id, model, created)) = head else {
        return Err(cancelled());
    };
    let mut choices = Vec::with_capacity(folded.len());
    for (index, choice) in folded.into_iter().enumerate() {
        let Some((finish_reason, matched_stop, logprobs)) = choice.finish else {
            return Err(cancelled());
        };
        let Folded {
            content,
            reasoning,
            tool_calls,
            ..
        } = choice;
        choices.push(json!({
            "index": index,
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning_content": if reasoning.is_empty() { Value::Null } else { json!(reasoning) },
                "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
            },
            "logprobs": logprobs,
            "finish_reason": finish_reason,
            "matched_stop": matched_stop,
        }));
    }
    Ok(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
        "metadata": { "weight_version": "default" },
    }))
}

/// One committed token's `logprobs.content` entry in SGLang's shape (m27 D7/F7, `serving_chat.py:1918-1958`): the
/// text of an ID is `decode_with(&[id], false)` (`batch_decode([[id]])`), `bytes` its UTF-8; `top_logprobs` is the
/// `{text: logprob}` dict of `utils.py:33-38`, so a later alternative with an earlier one's text keeps that
/// position and takes its logprob. A -inf alternative renders `null` in a stream (SGLang's recorded rendering,
/// `sampler_cases.json` `neg_inf_logprob`) and is omitted from a unary body (SGLang's is not JSON; Q7). Logprobs are
/// the device's f32 widened to f64, as SGLang's Python floats are.
fn logprob_entry(vocab: &Vocab, lp: &TokenLogprobs, stream: bool) -> Value {
    let entry = |text: &str, logprob: f32| json!({ "token": text, "bytes": text.as_bytes(), "logprob": f64::from(logprob) });
    let n = (lp.top_count as usize).min(lp.top.len());
    let mut top: Vec<(String, f32)> = Vec::with_capacity(n);
    for alt in &lp.top[..n] {
        let text = vocab.decode_with(&[alt.id], false);
        match top.iter_mut().find(|(t, _)| *t == text) {
            Some(slot) => slot.1 = alt.logprob,
            None => top.push((text, alt.logprob)),
        }
    }
    let mut out = entry(&vocab.decode_with(&[lp.id], false), lp.logprob);
    out["top_logprobs"] = top
        .iter()
        .filter(|(_, logprob)| stream || logprob.is_finite())
        .map(|(text, logprob)| entry(text, *logprob))
        .collect();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the thread it is dropped on.
    struct Noting(std::sync::mpsc::Sender<std::thread::ThreadId>);

    impl Drop for Noting {
        fn drop(&mut self) {
            let _ = self.0.send(std::thread::current().id());
        }
    }

    /// Audit N2: values in an `OffRuntime` that drops on an async worker are dropped on the blocking pool instead;
    /// handed out with `into_vec` they drop where the caller chooses (and the empty wrapper drops nothing).
    #[tokio::test(flavor = "current_thread")]
    async fn off_runtime_values_drop_on_the_blocking_pool() {
        let here = std::thread::current().id();
        let (tx, rx) = std::sync::mpsc::channel();
        drop(OffRuntime(vec![Noting(tx.clone()), Noting(tx.clone())]));
        let first = rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let second = rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_ne!(first, here);
        assert_ne!(second, here);
        let owned = OffRuntime(vec![Noting(tx.clone())]).into_vec();
        assert!(rx.try_recv().is_err(), "into_vec drops nothing");
        drop(owned);
        assert_eq!(
            rx.try_recv().unwrap(),
            here,
            "dropped by the caller, in place"
        );
        drop(OffRuntime::<Noting>(Vec::new()));
        drop(tx);
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
    }

    /// Audit N2: outside a runtime there is no async worker to protect, and the values drop in place.
    #[test]
    fn off_runtime_values_drop_in_place_without_a_runtime() {
        let (tx, rx) = std::sync::mpsc::channel();
        drop(OffRuntime(vec![Noting(tx)]));
        assert_eq!(rx.try_recv().unwrap(), std::thread::current().id());
    }

    /// Audit N2: the operator's `load` reads atomics; it answers while the queue's mutex is held (a create's
    /// executor call used to hold it), and counts what `create` and `release` record.
    #[test]
    fn session_queue_load_does_not_take_the_queue_mutex() {
        let queue = Arc::new(SessionQueue::default());
        queue.open.store(3, Ordering::Release);
        let held = queue.lock();
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = queue.clone();
        std::thread::spawn(move || tx.send(reader.load()).unwrap());
        let answer = rx.recv_timeout(Duration::from_secs(10));
        drop(held);
        assert_eq!(answer.expect("load waited for the mutex"), (3, 0));
    }

    /// Audit N2/N3: a create's FIFO entry leaves when its guard drops (a panic of the executor call included) and
    /// the mirrored waiter count follows, so a later create is not stalled behind a dead entry; its CAPACITY and
    /// in-call counts (BcAdmit) go with it.
    #[test]
    fn a_waiting_entry_leaves_the_fifo_when_its_guard_drops() {
        let queue = SessionQueue::default();
        {
            let mut state = queue.lock();
            state.waiting.extend([0, 1]);
            state.blocked = 1;
            state.calling = 1;
            queue.waiting.store(2, Ordering::Release);
        }
        assert_eq!(queue.load(), (0, 2));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _waiting = Waiting {
                queue: &queue,
                id: 0,
                blocked: true,
                calling: true,
            };
            panic!("the executor call panicked");
        }));
        assert!(result.is_err());
        let state = queue.lock();
        assert_eq!(state.waiting.front(), Some(&1));
        assert_eq!((state.blocked, state.calling), (0, 0));
        drop(state);
        assert_eq!(queue.load(), (0, 1));
    }

    /// BcAdmit (O282, batch-concurrency design note): with no create waiting on CAPACITY, concurrent creates reach
    /// the executor together. Before, each waited for the previous one's round trip to the driver thread (the FIFO
    /// handoff), so a burst's sessions were created one round trip apart.
    #[test]
    fn creates_with_free_pages_reach_the_executor_together() {
        const N: usize = 4;
        let queue = SessionQueue::default();
        // (creates inside the executor call now, most at once so far)
        let inside = (
            std::sync::Mutex::new((0usize, 0usize)),
            std::sync::Condvar::new(),
        );
        let call = || {
            let (lock, cv) = &inside;
            let mut g = lock.lock().unwrap();
            g.0 += 1;
            g.1 = g.1.max(g.0);
            cv.notify_all();
            // Bounded: a serialized create would wait here for the others forever.
            let (mut g, _) = cv
                .wait_timeout_while(g, Duration::from_secs(2), |g| g.1 < N)
                .unwrap();
            g.0 -= 1;
            Ok(())
        };
        std::thread::scope(|s| {
            for _ in 0..N {
                s.spawn(|| assert!(matches!(queue.admit(|| false, call), Ok(Some(())))));
            }
        });
        assert_eq!(inside.0.lock().unwrap().1, N, "creates serialized");
        assert_eq!(queue.load(), (N as u64, 0));
    }

    fn no_pages() -> DriverError {
        DriverError::Status {
            call: "cere_session_create",
            status: StatusError::Capacity,
        }
    }

    /// Waits (at most 10 s) until `cond` holds.
    fn until(what: &str, cond: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {what}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// CR1 order kept (BcAdmit): while a create waits on CAPACITY, a later one waits behind it without asking the
    /// executor, and each release admits the oldest waiter.
    #[test]
    fn creates_waiting_for_pages_are_admitted_in_arrival_order() {
        let queue = SessionQueue::default();
        let pages = std::sync::Mutex::new(1u32);
        let log = std::sync::Mutex::new(Vec::new());
        let ask = |who: char| {
            let (pages, log) = (&pages, &log);
            move || {
                let mut free = pages.lock().unwrap();
                let got = *free > 0;
                *free -= u32::from(got);
                log.lock().unwrap().push((who, got));
                if got {
                    Ok(who)
                } else {
                    Err(no_pages())
                }
            }
        };
        let asked = |who: char, got: bool| log.lock().unwrap().contains(&(who, got));
        let blocked = || {
            let state = queue.lock();
            state.blocked == 1 && state.calling == 0
        };
        assert_eq!(queue.admit(|| false, ask('A')).unwrap(), Some('A'));
        std::thread::scope(|s| {
            let b = s.spawn(|| queue.admit(|| false, ask('B')));
            until("B's CAPACITY", || asked('B', false) && blocked());
            let c = s.spawn(|| queue.admit(|| false, ask('C')));
            until("C in the FIFO", || queue.load().1 == 2);
            std::thread::sleep(Duration::from_millis(50));
            assert!(
                !asked('C', false),
                "C asked ahead of B, which waits for pages"
            );
            *pages.lock().unwrap() += 1;
            queue.released(); // A's session
            assert_eq!(b.join().unwrap().unwrap(), Some('B'));
            until("C's CAPACITY", || asked('C', false) && blocked());
            *pages.lock().unwrap() += 1;
            queue.released(); // B's session
            assert_eq!(c.join().unwrap().unwrap(), Some('C'));
        });
        assert_eq!(
            *log.lock().unwrap(),
            [
                ('A', true),
                ('B', false),
                ('B', true),
                ('C', false),
                ('C', true)
            ]
        );
        assert_eq!(queue.load(), (1, 0));
    }

    /// BcAdmit: a CAPACITY answered while another create is still inside the executor call (which may open the
    /// session whose release it can wait for) waits, rather than fail as if nothing were open.
    #[test]
    fn a_capacity_beside_a_create_in_flight_waits_for_it() {
        let queue = SessionQueue::default();
        let a_may_return = std::sync::atomic::AtomicBool::new(false);
        let a_inside = std::sync::atomic::AtomicBool::new(false);
        let b_calls = AtomicU64::new(0);
        std::thread::scope(|s| {
            let a = s.spawn(|| {
                queue.admit(
                    || false,
                    || {
                        a_inside.store(true, Ordering::Release);
                        until("B's CAPACITY", || a_may_return.load(Ordering::Acquire));
                        Ok('A')
                    },
                )
            });
            until("A inside", || a_inside.load(Ordering::Acquire));
            let b = s.spawn(|| {
                queue.admit(
                    || false,
                    || match b_calls.fetch_add(1, Ordering::AcqRel) {
                        0 => Err(no_pages()),
                        _ => Ok('B'),
                    },
                )
            });
            until("B waiting beside A", || {
                let state = queue.lock();
                state.blocked == 1 && state.calling == 1
            });
            a_may_return.store(true, Ordering::Release);
            assert_eq!(a.join().unwrap().unwrap(), Some('A'));
            assert_eq!(b.join().unwrap().unwrap(), Some('B'));
        });
        assert_eq!(queue.load(), (2, 0));
    }

    /// CR1 kept: creates answered CAPACITY with no session open and none in flight fail, concurrent ones included
    /// (nothing can release pages for them).
    #[test]
    fn creates_answered_capacity_with_nothing_open_fail() {
        let queue = SessionQueue::default();
        let inside = AtomicU64::new(0);
        let call = || {
            if inside.fetch_add(1, Ordering::AcqRel) < 2 {
                until("both creates inside", || {
                    inside.load(Ordering::Acquire) >= 2
                });
            }
            Err::<(), _>(no_pages())
        };
        std::thread::scope(|s| {
            let a = s.spawn(|| queue.admit(|| false, call));
            let b = s.spawn(|| queue.admit(|| false, call));
            for r in [a.join().unwrap(), b.join().unwrap()] {
                assert!(matches!(r, Err(e) if capacity(&e)));
            }
        });
        assert_eq!(queue.load(), (0, 0));
    }

    /// Audit N3: a drop during a panic raises the fault flag; a normal drop leaves it down.
    #[test]
    fn fault_on_panic_raises_the_flag_only_when_panicking() {
        let flag = AtomicBool::new(false);
        drop(FaultOnPanic(&flag));
        assert!(!flag.load(Ordering::Acquire));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _fault = FaultOnPanic(&flag);
            panic!("the generation panicked");
        }));
        assert!(result.is_err());
        assert!(flag.load(Ordering::Acquire));
    }

    /// M2.12 C5 (D7, D11): the running count equals SGLang's per token, for every `reasoning_tokens` case of
    /// `tests/fixtures/native/thinking_cases.json`; thinking off counts nothing.
    #[test]
    fn reasoning_tokens_follow_the_oracle() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/native/thinking_cases.json"
        );
        let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let ids = |v: &Value| -> Vec<u64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|i| i.as_u64().unwrap())
                .collect()
        };
        let mut checked = 0;
        for case in doc["cases"].as_array().unwrap() {
            if case["kind"] != "reasoning_tokens" {
                continue;
            }
            let end = ids(&case["think_end_ids"]);
            assert_eq!(end.len(), 1, "{case}");
            let mut on = ReasoningTokens::new(true, Some(end[0] as u32));
            let mut off = ReasoningTokens::new(false, Some(end[0] as u32));
            let mut got = Vec::new();
            for id in ids(&case["ids"]) {
                on.push(id as u32);
                off.push(id as u32);
                got.push(on.count);
            }
            assert_eq!(got, ids(&case["reasoning_tokens"]), "{case}");
            assert_eq!(off.count, 0);
            checked += 1;
        }
        assert!(checked >= 5);
    }

    /// M2.10 D4 (c) (W 132): the last message's start floored to 64, only from 8,192 in and within 1,024 of the end.
    #[test]
    fn capture_position_follows_d4c_limits() {
        assert_eq!(capture_position(9_000, 9_100, true), 8_960);
        assert_eq!(capture_position(9_000, 9_100, false), 0);
        assert_eq!(capture_position(9_000, 10_000, true), 0);
        assert_eq!(capture_position(8_100, 8_200, true), 0);
        assert_eq!(capture_position(8_192, 8_192, true), 0);
        assert_eq!(capture_position(8_192, 9_216, true), 8_192);
    }

    /// M2.12 C6b (D9 (ii), D14 M2.9): `ignore_eos` tickets carry bit 0 and `NO_SPECULATION`; `min_tokens > 0`
    /// tickets carry `NO_SPECULATION` only; a plain request carries nothing. M2.8 D8/I3: a constrained request's
    /// tickets carry `NO_SPECULATION` (these landed cases are M2.9's regression guards, M29S2-08).
    #[test]
    fn work_flags_follow_ignore_eos_min_tokens_and_constraint() {
        assert_eq!(work_flags(false, 0, false, None), 0);
        assert_eq!(
            work_flags(true, 0, false, None),
            ffi::WORK_IGNORE_STOP_TOKENS | ffi::WORK_NO_SPECULATION
        );
        assert_eq!(work_flags(false, 3, false, None), ffi::WORK_NO_SPECULATION);
        assert_eq!(
            work_flags(true, 3, false, None),
            ffi::WORK_IGNORE_STOP_TOKENS | ffi::WORK_NO_SPECULATION
        );
        assert_eq!(work_flags(false, 0, true, None), ffi::WORK_NO_SPECULATION);
    }

    /// VisionLong C3: an image session keeps its flags (draft bits included) only when the profile declares the
    /// image `verify` combination; otherwise (the -v6 lab profile) `NO_SPECULATION` joins them. Text passes through.
    #[test]
    fn image_flags_follow_the_image_verify_combination() {
        let drafts = ffi::WORK_LOOKUP_DRAFTS;
        assert_eq!(image_flags(drafts, false, false), drafts);
        assert_eq!(image_flags(drafts, false, true), drafts);
        assert_eq!(image_flags(drafts, true, true), drafts);
        assert_eq!(
            image_flags(drafts, true, false),
            drafts | ffi::WORK_NO_SPECULATION
        );
    }

    /// m29 design note D11 (M29S2-08): a non-neutral presence, frequency or repetition penalty or any
    /// `logit_bias` entry sets `NO_SPECULATION`; neutral sampling (greedy or sampled) does not.
    #[test]
    fn work_flags_refuse_speculation_for_penalties_and_logit_bias() {
        let neutral = Sampling {
            temperature: 0.7,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 248_320,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 1,
            logit_bias: Vec::new(),
        };
        assert_eq!(work_flags(false, 0, false, Some(&neutral)), 0);
        let greedy = Sampling {
            temperature: 0.0,
            ..neutral.clone()
        };
        assert_eq!(work_flags(false, 0, false, Some(&greedy)), 0);
        let processed = [
            Sampling {
                presence_penalty: 0.5,
                ..neutral.clone()
            },
            Sampling {
                frequency_penalty: -0.25,
                ..neutral.clone()
            },
            Sampling {
                repetition_penalty: 1.1,
                ..greedy.clone()
            },
            Sampling {
                logit_bias: vec![ffi::LogitBias { id: 7, bias: 0.0 }],
                ..greedy.clone()
            },
        ];
        for s in &processed {
            assert_eq!(
                work_flags(false, 0, false, Some(s)),
                ffi::WORK_NO_SPECULATION,
                "{s:?}"
            );
            assert_eq!(
                work_flags(true, 0, false, Some(s)),
                ffi::WORK_IGNORE_STOP_TOKENS | ffi::WORK_NO_SPECULATION,
                "{s:?}"
            );
        }
    }

    /// m29l design note C5L (D6): "mtp" adds nothing, "mtp+lookup" bit 2, "lookup" bits 2 and 3, and a ticket that
    /// carries `NO_SPECULATION` gets neither.
    #[test]
    fn draft_flags_follow_the_drafter_lever() {
        let lookup = ffi::WORK_LOOKUP_DRAFTS;
        let only = ffi::WORK_LOOKUP_DRAFTS | ffi::WORK_NO_MTP_DRAFTS;
        let fixed = DraftPolicy::Fixed;
        assert_eq!(draft_flags(0, Drafter::Mtp, fixed, None), 0);
        assert_eq!(draft_flags(0, Drafter::MtpLookup, fixed, None), lookup);
        assert_eq!(draft_flags(0, Drafter::Lookup, fixed, None), only);
        for drafts in [Drafter::Mtp, Drafter::Lookup, Drafter::MtpLookup] {
            for policy in [DraftPolicy::Fixed, DraftPolicy::Adaptive] {
                for flags in [
                    work_flags(true, 0, false, None),
                    work_flags(false, 3, false, None),
                    work_flags(false, 0, true, None),
                ] {
                    assert_eq!(
                        draft_flags(flags, drafts, policy, None),
                        flags,
                        "{drafts:?} {policy:?}"
                    );
                }
            }
        }
    }

    /// m29a design note A2, A4: the draft-policy lever's "adaptive" adds bit 4 to a greedy ticket that may speculate,
    /// with any draft source; the default "fixed" adds nothing (today's tickets unchanged).
    #[test]
    fn draft_flags_follow_draft_policy() {
        let adaptive = ffi::WORK_ADAPTIVE_DRAFTS;
        assert_eq!(DraftPolicy::default(), DraftPolicy::Fixed);
        assert_eq!(draft_flags(0, Drafter::Mtp, DraftPolicy::Fixed, None), 0);
        assert_eq!(
            draft_flags(0, Drafter::Mtp, DraftPolicy::Adaptive, None),
            adaptive
        );
        assert_eq!(
            draft_flags(0, Drafter::MtpLookup, DraftPolicy::Adaptive, None),
            ffi::WORK_LOOKUP_DRAFTS | adaptive
        );
        let stop = ffi::WORK_IGNORE_STOP_TOKENS | ffi::WORK_NO_SPECULATION;
        assert_eq!(
            draft_flags(stop, Drafter::Mtp, DraftPolicy::Adaptive, None),
            stop
        );
        assert_eq!(adaptive & ffi::WORK_KNOWN_FLAGS, adaptive);
    }

    /// Review M1 ([REVISED: review-wave-fg M1], m29a design note A2): a seeded sampled ticket keeps the fixed draft
    /// policy under `draft_policy = "adaptive"` (the seeded stream that picks a sampled token depends on the round's
    /// draft count; greedy accept does not), whatever its drafter.
    #[test]
    fn draft_flags_sampled_ticket_keeps_fixed_policy() {
        let sampled = Sampling {
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 248_320,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 1234,
            logit_bias: Vec::new(),
        };
        let flags = work_flags(false, 0, false, Some(&sampled));
        assert_eq!(flags & ffi::WORK_NO_SPECULATION, 0);
        for drafter in [
            Drafter::Mtp,
            Drafter::Lookup,
            Drafter::MtpLookup,
            Drafter::MtpLookupTail,
            Drafter::MtpLookupLong,
            Drafter::MtpLookupChain,
        ] {
            let fixed = draft_flags(flags, drafter, DraftPolicy::Fixed, Some(&sampled));
            assert_eq!(
                fixed & (ffi::WORK_TAIL_DRAFTS | ffi::WORK_LONG_DRAFTS | ffi::WORK_CHAIN_DRAFTS),
                0,
                "{drafter:?}"
            );
            assert_eq!(fixed & ffi::WORK_ADAPTIVE_DRAFTS, 0, "{drafter:?}");
            assert_eq!(
                draft_flags(flags, drafter, DraftPolicy::Adaptive, Some(&sampled)),
                fixed,
                "{drafter:?}"
            );
            // A greedy ticket adds bit 4 (and bit 5 for the tail drafter, bits 5 and 6 for the long one, bits 5, 6 and
            // 7 for the chain one) to the sampled ticket's sources.
            let deep = match drafter {
                Drafter::MtpLookupTail => ffi::WORK_TAIL_DRAFTS,
                Drafter::MtpLookupLong => ffi::WORK_TAIL_DRAFTS | ffi::WORK_LONG_DRAFTS,
                Drafter::MtpLookupChain => {
                    ffi::WORK_TAIL_DRAFTS | ffi::WORK_LONG_DRAFTS | ffi::WORK_CHAIN_DRAFTS
                }
                _ => 0,
            };
            assert_eq!(
                draft_flags(flags, drafter, DraftPolicy::Adaptive, None),
                fixed | ffi::WORK_ADAPTIVE_DRAFTS | deep,
                "{drafter:?}"
            );
        }
    }

    /// DraftAccept: "mtp+lookup+tail" adds bit 5 next to bit 2 on a greedy ticket that may speculate (bit 4 still from
    /// the policy lever), nothing on a NO_SPECULATION ticket; bit 5 is a known flag.
    #[test]
    fn draft_flags_tail_drafter() {
        let tail = ffi::WORK_LOOKUP_DRAFTS | ffi::WORK_TAIL_DRAFTS;
        assert_eq!(ffi::WORK_TAIL_DRAFTS, 0x20);
        assert_eq!(ffi::WORK_KNOWN_FLAGS, 0xFF);
        assert_eq!(
            draft_flags(0, Drafter::MtpLookupTail, DraftPolicy::Fixed, None),
            tail
        );
        assert_eq!(
            draft_flags(0, Drafter::MtpLookupTail, DraftPolicy::Adaptive, None),
            tail | ffi::WORK_ADAPTIVE_DRAFTS
        );
        let off = ffi::WORK_NO_SPECULATION;
        assert_eq!(
            draft_flags(off, Drafter::MtpLookupTail, DraftPolicy::Adaptive, None),
            off
        );
    }

    /// TreeSpec TS-01: "mtp+lookup+long" adds bits 5 and 6 next to bit 2 on a greedy ticket that may speculate (bit 4
    /// still from the policy lever), nothing on a NO_SPECULATION ticket; bit 6 is a known flag.
    #[test]
    fn draft_flags_long_drafter() {
        let long = ffi::WORK_LOOKUP_DRAFTS | ffi::WORK_TAIL_DRAFTS | ffi::WORK_LONG_DRAFTS;
        assert_eq!(ffi::WORK_LONG_DRAFTS, 0x40);
        assert_eq!(
            ffi::WORK_KNOWN_FLAGS & ffi::WORK_LONG_DRAFTS,
            ffi::WORK_LONG_DRAFTS
        );
        assert_eq!(
            draft_flags(0, Drafter::MtpLookupLong, DraftPolicy::Fixed, None),
            long
        );
        assert_eq!(
            draft_flags(0, Drafter::MtpLookupLong, DraftPolicy::Adaptive, None),
            long | ffi::WORK_ADAPTIVE_DRAFTS
        );
        let off = ffi::WORK_NO_SPECULATION;
        assert_eq!(
            draft_flags(off, Drafter::MtpLookupLong, DraftPolicy::Adaptive, None),
            off
        );
    }

    /// Chains (chains design note): "mtp+lookup+chain" adds bits 5, 6 and 7 next to bit 2 on a greedy ticket that
    /// may speculate (bit 4 still from the policy lever), nothing on a NO_SPECULATION ticket; bit 7 is a known flag.
    #[test]
    fn draft_flags_chain_drafter() {
        let chain = ffi::WORK_LOOKUP_DRAFTS
            | ffi::WORK_TAIL_DRAFTS
            | ffi::WORK_LONG_DRAFTS
            | ffi::WORK_CHAIN_DRAFTS;
        assert_eq!(ffi::WORK_CHAIN_DRAFTS, 0x80);
        assert_eq!(
            ffi::WORK_KNOWN_FLAGS & ffi::WORK_CHAIN_DRAFTS,
            ffi::WORK_CHAIN_DRAFTS
        );
        assert_eq!(
            draft_flags(0, Drafter::MtpLookupChain, DraftPolicy::Fixed, None),
            chain
        );
        assert_eq!(
            draft_flags(0, Drafter::MtpLookupChain, DraftPolicy::Adaptive, None),
            chain | ffi::WORK_ADAPTIVE_DRAFTS
        );
        let off = ffi::WORK_NO_SPECULATION;
        assert_eq!(
            draft_flags(off, Drafter::MtpLookupChain, DraftPolicy::Adaptive, None),
            off
        );
    }

    /// inkling design note D-I-S9B: an Inkling ticket carries plain MTP only (every draft-source and policy bit
    /// cleared), a sampled or logprobs one NO_SPECULATION; the stop bit stays; other families pass through.
    #[test]
    fn family_flags_inkling_plain_mtp() {
        let ink = Family::Inkling2Loop;
        assert_eq!(family_flags(0, ink, false), 0);
        assert_eq!(family_flags(0, ink, true), ffi::WORK_NO_SPECULATION);
        let every = ffi::WORK_KNOWN_FLAGS & !ffi::WORK_NO_SPECULATION;
        assert_eq!(
            family_flags(every, ink, false),
            ffi::WORK_IGNORE_STOP_TOKENS
        );
        assert_eq!(
            family_flags(every, ink, true),
            ffi::WORK_IGNORE_STOP_TOKENS | ffi::WORK_NO_SPECULATION
        );
        for family in [Family::Qwen4Exp, Family::Sdar] {
            assert_eq!(family_flags(every, family, true), every);
            assert_eq!(family_flags(0, family, false), 0);
        }
    }
}
