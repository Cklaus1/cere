//! CI-only fake of the probed SGLang HTTP subset (m1 design note K1).
//!
//! # CI-only; never M1 acceptance evidence
//!
//! This crate imitates the shapes observed on the live `sglang-t4` server on 2026-10-05 closely enough to
//! exercise CERE's backend client, ledger, streaming and cancellation paths deterministically in `make ci`.
//! It does not run a model and does not prove anything about the real server. Every externally observable
//! M1 row still needs its live check.
//!
//! # Library use
//!
//! ```no_run
//! # async fn demo() -> std::io::Result<()> {
//! use cere_mock_sglang::{Fault, MockConfig, MockSglang};
//! use std::time::Duration;
//!
//! let mock = MockSglang::start(MockConfig {
//!     token_latency: Duration::from_millis(20),
//!     ..MockConfig::default()
//! })
//! .await?;
//! let base = mock.base_url(); // "http://127.0.0.1:<ephemeral port>"
//! mock.set_fault(Fault::IgnoreAbort); // applies to requests that arrive from now on
//! // ... drive CERE against `base` ...
//! assert!(mock.max_in_flight() <= 2);
//! mock.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! [`MockSglang::start`] binds `127.0.0.1:<MockConfig::port>` (0 = ephemeral) and serves until
//! [`MockHandle::shutdown`] or until the handle is dropped. The binary `cere-mock-sglang [--port N]
//! [--server-info FILE] [--token-latency-ms N] [--max-running-requests N]` serves the default configuration
//! adjusted by those flags (a probed server info file, per-token latency, scheduler cap), always reports its own
//! launch time as `startup_time`, and prints `cere-mock-sglang listening on http://127.0.0.1:<port>` on stdout;
//! see the binary's docs.
//!
//! # Endpoints
//!
//! - `GET /health` → 200, empty body.
//! - `GET /v1/models` → one model: the effective `served_model_name`, `max_model_len` = `context_length`.
//! - `GET /get_server_info` → the effective [`ServerInfo`] (the pinned subset of SGLang's server args, plus the
//!   per-launch `random_seed` and `startup_time` when set, plus any [`ServerInfo::extra`] keys).
//! - `GET /get_load` → SGLang's per-data-parallel-rank load list with one rank:
//!   `[{"dp_rank":0,"num_reqs":r,"num_waiting_reqs":w}]`, where `r + w` is the number of requests inside the
//!   "scheduler" (`r` at most `max_running_requests`). A request counts from the moment it starts generating
//!   (after any [`MockHandle::set_load_lag`] delay or [`MockHandle::hold_start`] hold) until its generation ends:
//!   finish, abort, injected disconnect or shutdown, or its client going away; with
//!   [`Fault::KeepGeneratingAfterDisconnect`] a client that goes away leaves it counting until it would have
//!   finished (or [`MockHandle::end_kept`]). Requests rejected before generation (injected errors, context
//!   overflow, [`Fault::HangBeforeResponse`]) never count.
//! - `POST /v1/chat/completions` → unary or SSE (`stream:true`). Reads `rid`, `model`, `messages`,
//!   `max_completion_tokens` (wins over `max_tokens`, as in SGLang), `stop` (string or list), `stream`,
//!   `stream_options.include_usage` and `cache_salt`; other fields are ignored. The request `rid` becomes the
//!   response/chunk `id`; without one, a 32-hex id is generated.
//! - `POST /abort_request {"rid": r, "abort_all": b}` → 200, empty body. Aborts every live request whose rid
//!   starts with `r` (the SGLang scheduler's predicate); an empty `r` or `abort_all:true` aborts everything.
//!   Aborting `cere-1` therefore also kills `cere-10`: CERE rids must be prefix-free.
//! - `POST /v1/messages/count_tokens` (Anthropic body) → `{"input_tokens": n}`, `n` counted as a chat request's
//!   prompt tokens over `system` and `messages` (string contents and text blocks' `text`).
//!
//! "Effective" server info is the override set by [`MockHandle::set_server_info_override`] if any, else
//! [`MockConfig::server_info`]. It drives every endpoint (models listing, context check, cache report), so an
//! override simulates a restarted or reconfigured server.
//!
//! # Deterministic generation
//!
//! - Prompt tokens = whitespace-separated words across all string `content` fields of `messages`.
//! - Output tokens are `tok0`, ` tok1`, ` tok2`, …; content is their concatenation (`"tok0 tok1 tok2"`).
//! - Finish: after `natural_output_tokens` tokens → `stop` with `matched_stop` = `eos_token_id`; at the token
//!   budget first → `length` (`matched_stop: null`); when the content contains a `stop` string → `stop`, the
//!   content is cut before the match and `matched_stop` is that string; on abort → `abort`.
//! - `usage.completion_tokens` = output tokens emitted.
//! - Prefix cache (off with `disable_radix_cache`): every chat request that is served (no injected server
//!   error, no context overflow) inserts `(cache_salt, messages)` into a
//!   cache that lives as long as the mock (the salt namespaces it, as SGLang's `cache_salt` does). A request
//!   whose exact pair is already cached is a hit with `cached = prompt_tokens` rounded down to a multiple of
//!   `cache_granularity` (SGLang's mamba checkpoint interval, 64 on the bench backend); anything else is a miss
//!   (`cached = 0`). With `enable_cache_report`, `prompt_tokens_details` is `{"cached_tokens": cached}` when
//!   `cached > 0` and `null` otherwise (live: a miss reports `null`); without it, always `null`.
//! - Context overflow: prompt tokens + token budget > `context_length` (or prompt tokens ≥ `context_length`
//!   without a budget) → HTTP 400 `{"object":"error","message":"… is longer than the model's context length …",
//!   "type":"BadRequestError","param":null,"code":400}`.
//! - Every output token waits `token_latency` first, so tests can act mid-stream.
//!
//! SSE order: a role chunk (`delta: {"reasoning_content":null,"role":"assistant","content":""}`), one chunk per
//! token, one finish chunk (null delta fields, `finish_reason` + `matched_stop`), then with
//! `stream_options.include_usage` a `choices:[]` + `usage` chunk, then `data: [DONE]`.
//!
//! # Faults and observation
//!
//! [`MockHandle::set_fault`] installs a [`Fault`]. Each chat request snapshots the fault when it arrives, so a
//! change affects later requests only. Faults apply to `/v1/chat/completions` only; readiness endpoints are
//! perturbed with [`MockHandle::set_server_info_override`] (drift) or by shutting the mock down (connection
//! loss). Hung requests ignore aborts and end only at shutdown. [`MockHandle::set_abort_delay`] makes
//! `/abort_request` wait before it takes effect and answers (a slow control plane); [`MockHandle::hold_aborts`]
//! makes it wait until the test clears the hold (a control plane stalled for as long as the test needs).
//! [`MockHandle::set_info_delay`] makes `/get_server_info` wait before it reads the effective info and answers (a
//! probe slower than its worker's suspect bound, as on a loaded host). [`MockHandle::set_load_lag`] holds each
//! later chat request "in transit" for a while: no response bytes and no `/get_load` count yet (a request still
//! in SGLang's tokenizer, not in the scheduler); [`MockHandle::hold_start`] holds them there until it is cleared, so
//! a test chooses the moment a live request starts producing output. [`MockHandle::hold_finish`] holds every
//! generation that has emitted its last token before its finish until it is cleared (an abort or a shutdown still
//! ends it), so a test chooses the moment a request ends; [`MockHandle::finish_held`] counts the ones waiting there.
//! [`MockHandle::end_kept`] ends the generations [`Fault::KeepGeneratingAfterDisconnect`] keeps counting, so a test
//! chooses the moment an orphan stops.
//!
//! The handle records every chat request's rid in arrival order ([`MockHandle::dispatched_rids`]) and the
//! `cache_salt` it carried ([`MockHandle::received_cache_salts`], same order), every abort call
//! ([`MockHandle::abort_requests`]) and the current and maximum number of concurrently live chat requests
//! ([`MockHandle::in_flight`], [`MockHandle::max_in_flight`]). A request is live from arrival until its
//! response finishes or its client disconnects.

mod server;

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// The pinned subset of SGLang's `/get_server_info`. Defaults are the live `sglang-t4` values probed on
/// 2026-10-05 (m1 design note "Environment facts"). Deserializing a probed body keeps every key the mock
/// does not model in [`ServerInfo::extra`], so the mock serves that body back unchanged (a null `random_seed`
/// or `startup_time` is omitted).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerInfo {
    pub version: String,
    pub model_path: String,
    pub tokenizer_path: String,
    pub served_model_name: String,
    pub chat_template: Option<String>,
    pub quantization: Option<String>,
    pub kv_cache_dtype: String,
    pub speculative_algorithm: Option<String>,
    pub speculative_num_steps: Option<u32>,
    pub speculative_eagle_topk: Option<u32>,
    pub speculative_num_draft_tokens: Option<u32>,
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
    pub allow_auto_truncate: bool,
    pub enable_cache_report: bool,
    pub context_length: u64,
    pub max_running_requests: u32,
    pub disable_radix_cache: bool,
    /// Per-launch value (random per launch unless SGLang runs with `--random-seed`); omitted when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
    /// Per-launch load durations; omitted when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup_time: Option<serde_json::Value>,
    /// Every other key of a probed body, served verbatim after the modelled ones; empty by default.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for ServerInfo {
    fn default() -> Self {
        let model = "/models/qwen38-flash-next/Qwen3.8-Flash-Next-NVFP4";
        Self {
            version: "0.5.19.dev3+gd4689328e".to_owned(),
            model_path: model.to_owned(),
            tokenizer_path: model.to_owned(),
            served_model_name: "Qwen3.8-Flash-Next".to_owned(),
            chat_template: None,
            quantization: Some("modelopt_fp4".to_owned()),
            kv_cache_dtype: "fp8_e4m3".to_owned(),
            speculative_algorithm: Some("EAGLE".to_owned()),
            speculative_num_steps: Some(3),
            speculative_eagle_topk: Some(1),
            speculative_num_draft_tokens: Some(4),
            reasoning_parser: Some("qwen3".to_owned()),
            tool_call_parser: Some("qwen3_coder".to_owned()),
            allow_auto_truncate: false,
            enable_cache_report: false,
            context_length: 200_000,
            max_running_requests: 4,
            disable_radix_cache: false,
            random_seed: None,
            startup_time: None,
            extra: serde_json::Map::new(),
        }
    }
}

/// Mock configuration. `Default` = ephemeral port, live server info, no latency.
#[derive(Clone, Debug)]
pub struct MockConfig {
    /// Port on 127.0.0.1; 0 picks an ephemeral port.
    pub port: u16,
    pub server_info: ServerInfo,
    /// Delay before each output token.
    pub token_latency: Duration,
    /// Output tokens before the natural EOS (`finish_reason:"stop"`), unless the budget ends first.
    pub natural_output_tokens: u32,
    /// `matched_stop` reported for the natural EOS.
    pub eos_token_id: u64,
    /// Prefix-cache hit granularity in tokens (a hit reports `prompt_tokens` rounded down to a multiple of
    /// it); 0 counts as 1.
    pub cache_granularity: u64,
    /// Keep every chat request's raw body for [`MockHandle::received_bodies`] (tests only: a long dev run would
    /// grow without bound). Default off.
    pub record_bodies: bool,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            port: 0,
            server_info: ServerInfo::default(),
            token_latency: Duration::ZERO,
            natural_output_tokens: 8,
            eos_token_id: 151_645,
            cache_granularity: 64,
            record_bodies: false,
        }
    }
}

/// Injected chat-completion fault. Snapshotted per request on arrival.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Fault {
    #[default]
    None,
    /// Answer the next `remaining` chat requests with `status` and an SGLang-style error object, then revert
    /// to `None`.
    ServerError { status: StatusCode, remaining: u32 },
    /// Drop the connection after `tokens` output tokens (stream: mid-body; unary: 200 headers, broken body).
    DisconnectAfter { tokens: u32 },
    /// Accept the request and never send a response.
    HangBeforeResponse,
    /// Stream the role chunk and `tokens` output tokens, then never send more (unary: never responds).
    HangAfterTokens { tokens: u32 },
    /// Return 200 to `/abort_request` but keep generating to the natural finish or the token budget.
    IgnoreAbort,
    /// Finish with `length` after at most `tokens` output tokens, regardless of the requested budget.
    ClampOutput { tokens: u32 },
    /// Stream: 200, then one in-stream error chunk `data: {"error":{"message","type","param","code"}}` and
    /// `data: [DONE]` (SGLang's scheduler-side rejection, e.g. code 400 "Input length (N tokens) exceeds the
    /// maximum allowed length"). Unary: HTTP `code` with the SGLang error object carrying `message`.
    StreamError { code: u16, message: String },
    /// A client that goes away mid-generation leaves the request generating (and counting in `/get_load`) until
    /// it would have finished or [`MockHandle::end_kept`] ends it; `/abort_request` no longer reaches it (SGLang
    /// drops the rid on disconnect).
    KeepGeneratingAfterDisconnect,
    /// Answer every chat request (unary and stream) with HTTP `code` and the SGLang error object carrying
    /// `message`, before any generation: SGLang's request-conversion refusals, e.g. a tool whose `parameters`
    /// fail the schema check (`Tool 0 function has invalid 'parameters' schema: ...`).
    Reject { code: u16, message: String },
}

/// A scripted chat reply (M2.4 C8): while set, every chat request that passes the context check and the
/// faults is answered with it instead of the generator. Unary: `response` as the 200 body. Stream: each of
/// `chunks` as one `data:` event, then `data: [DONE]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scripted {
    pub response: Value,
    pub chunks: Vec<Value>,
}

/// One recorded `POST /abort_request` call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbortRecord {
    pub rid: String,
    pub abort_all: bool,
}

/// Entry point; see the crate docs.
pub struct MockSglang;

impl MockSglang {
    /// Binds `127.0.0.1:<config.port>` and starts serving on the current Tokio runtime.
    pub async fn start(config: MockConfig) -> std::io::Result<MockHandle> {
        let listener =
            tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, config.port)))
                .await?;
        let addr = listener.local_addr()?;
        let (shutdown, mut shutdown_rx) = watch::channel(false);
        let shared = Arc::new(Shared {
            config,
            created: unix_secs(),
            state: Mutex::new(State::default()),
            shutdown,
            finish_hold: watch::channel(false).0,
            kept_end: watch::channel(0).0,
            abort_hold: watch::channel(false).0,
            start_hold: watch::channel(false).0,
        });
        let app = server::router(Arc::clone(&shared));
        let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = shutdown_rx.wait_for(|stop| *stop).await;
        });
        let task = tokio::spawn(async move { serve.await });
        Ok(MockHandle { addr, shared, task })
    }
}

/// Running mock. Dropping it signals shutdown without waiting.
pub struct MockHandle {
    addr: SocketAddr,
    shared: Arc<Shared>,
    task: JoinHandle<std::io::Result<()>>,
}

impl MockHandle {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// `http://127.0.0.1:<port>`, no trailing slash.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Rids of every chat request received, in arrival order (generated ids for requests without `rid`).
    pub fn dispatched_rids(&self) -> Vec<String> {
        self.shared.lock().dispatched.clone()
    }

    /// The `cache_salt` of every chat request received (`None` when absent or not a string), in the order of
    /// [`MockHandle::dispatched_rids`].
    pub fn received_cache_salts(&self) -> Vec<Option<String>> {
        self.shared.lock().cache_salts.clone()
    }

    /// The raw body bytes of every chat request received, in arrival order; empty unless
    /// [`MockConfig::record_bodies`].
    pub fn received_bodies(&self) -> Vec<Vec<u8>> {
        self.shared.lock().bodies.clone()
    }

    /// Installs (`Some`) or clears (`None`) the scripted reply for chat requests arriving from now on.
    pub fn set_scripted(&self, scripted: Option<Scripted>) {
        self.shared.lock().scripted = scripted;
    }

    /// Every `/abort_request` call, in arrival order.
    pub fn abort_requests(&self) -> Vec<AbortRecord> {
        self.shared.lock().aborts.clone()
    }

    /// Chat requests currently live (arrived and not yet finished or disconnected).
    pub fn in_flight(&self) -> usize {
        self.shared.lock().live.len()
    }

    /// Highest `in_flight` value observed since start.
    pub fn max_in_flight(&self) -> usize {
        self.shared.lock().max_in_flight
    }

    /// Installs `fault` for chat requests arriving from now on.
    pub fn set_fault(&self, fault: Fault) {
        self.shared.lock().fault = fault;
    }

    /// `Some(info)` replaces the effective server info (drift); `None` restores `MockConfig::server_info`.
    pub fn set_server_info_override(&self, info: Option<ServerInfo>) {
        self.shared.lock().server_info_override = info;
    }

    /// Makes `/abort_request` wait `delay` before it takes effect and answers (a slow control plane).
    pub fn set_abort_delay(&self, delay: Duration) {
        self.shared.lock().abort_delay = delay;
    }

    /// While `hold` is set, `/abort_request` waits (after any [`MockHandle::set_abort_delay`]) before it takes
    /// effect and answers; a shutdown ends the wait. Clearing it lets the held calls go on.
    pub fn hold_aborts(&self, hold: bool) {
        self.shared.abort_hold.send_replace(hold);
    }

    /// Every `/abort_request` call that reached the [`MockHandle::hold_aborts`] hold so far, in arrival order
    /// (delivered to the backend, whether or not it took effect yet).
    pub fn held_aborts(&self) -> Vec<AbortRecord> {
        self.shared.lock().held_aborts.clone()
    }

    /// Makes `/get_server_info` wait `delay` before it reads the effective info and answers (a probe slower than
    /// its worker's suspect bound, as on a loaded host).
    pub fn set_info_delay(&self, delay: Duration) {
        self.shared.lock().info_delay = delay;
    }

    /// Holds every chat request arriving from now on for `lag` before it starts generating: no response bytes
    /// and no `/get_load` count until then (aborts and the request's `in_flight` count apply at once).
    pub fn set_load_lag(&self, lag: Duration) {
        self.shared.lock().load_lag = lag;
    }

    /// While `hold` is set, every chat request about to start generating waits in transit as under
    /// [`MockHandle::set_load_lag`], but until the hold is cleared instead of for a fixed time: no response bytes and
    /// no `/get_load` count (its `in_flight` count applies at once). An abort or a shutdown ends the wait. Clearing it
    /// lets the held ones start.
    pub fn hold_start(&self, hold: bool) {
        self.shared.start_hold.send_replace(hold);
    }

    /// While `hold` is set, every generation (unary or stream) that has emitted its last token waits before its
    /// finish; an abort or a shutdown still ends it. Clearing it lets the held ones finish.
    pub fn hold_finish(&self, hold: bool) {
        self.shared.finish_hold.send_replace(hold);
    }

    /// Generations waiting at the [`MockHandle::hold_finish`] hold now.
    pub fn finish_held(&self) -> usize {
        self.shared.lock().finish_held
    }

    /// Ends every generation [`Fault::KeepGeneratingAfterDisconnect`] keeps counting now (as if it reached its
    /// finish): it leaves the `/get_load` total ([`MockHandle::load`]). Generations kept later are not affected.
    pub fn end_kept(&self) {
        self.shared.kept_end.send_modify(|n| *n += 1);
    }

    /// The `/get_load` total: requests generating now (including ones kept by
    /// [`Fault::KeepGeneratingAfterDisconnect`]).
    pub fn load(&self) -> usize {
        self.shared.lock().generating
    }

    /// Ends live and hung requests (their connections drop), stops accepting, and waits for the server task.
    pub async fn shutdown(mut self) -> std::io::Result<()> {
        self.shared.shutdown.send_replace(true);
        (&mut self.task).await.map_err(std::io::Error::other)?
    }
}

impl Drop for MockHandle {
    fn drop(&mut self) {
        self.shared.shutdown.send_replace(true);
    }
}

struct Shared {
    config: MockConfig,
    created: u64,
    state: Mutex<State>,
    shutdown: watch::Sender<bool>,
    /// [`MockHandle::hold_finish`].
    finish_hold: watch::Sender<bool>,
    /// [`MockHandle::end_kept`]: bumped once per call.
    kept_end: watch::Sender<u64>,
    /// [`MockHandle::hold_aborts`].
    abort_hold: watch::Sender<bool>,
    /// [`MockHandle::hold_start`].
    start_hold: watch::Sender<bool>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[derive(Default)]
struct State {
    fault: Fault,
    server_info_override: Option<ServerInfo>,
    dispatched: Vec<String>,
    cache_salts: Vec<Option<String>>,
    bodies: Vec<Vec<u8>>,
    scripted: Option<Scripted>,
    /// Cached `(cache_salt, messages)` pairs, encoded as one JSON array string.
    prefix_cache: HashSet<String>,
    aborts: Vec<AbortRecord>,
    /// Delay before `/abort_request` takes effect and answers.
    abort_delay: Duration,
    /// `/abort_request` calls that reached the [`MockHandle::hold_aborts`] hold so far.
    held_aborts: Vec<AbortRecord>,
    /// Delay before `/get_server_info` reads the effective info and answers.
    info_delay: Duration,
    /// Delay before a chat request starts generating (see [`MockHandle::set_load_lag`]).
    load_lag: Duration,
    live: HashMap<u64, LiveRequest>,
    next_id: u64,
    max_in_flight: usize,
    /// Requests generating now: the `/get_load` total.
    generating: usize,
    /// Generations waiting at the finish hold ([`MockHandle::finish_held`]).
    finish_held: usize,
}

struct LiveRequest {
    rid: String,
    abort: watch::Sender<bool>,
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
