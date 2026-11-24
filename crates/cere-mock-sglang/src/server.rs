//! HTTP handlers and the deterministic token generator.

use crate::{unix_secs, AbortRecord, Fault, LiveRequest, ServerInfo, Shared};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

pub(crate) fn router(shared: Arc<Shared>) -> Router {
    Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/v1/models", get(models))
        .route("/get_server_info", get(server_info))
        .route("/get_load", get(load))
        .route("/v1/chat/completions", post(chat))
        .route("/abort_request", post(abort_request))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .with_state(shared)
}

fn effective_info(shared: &Shared) -> ServerInfo {
    shared
        .lock()
        .server_info_override
        .clone()
        .unwrap_or_else(|| shared.config.server_info.clone())
}

async fn models(State(shared): State<Arc<Shared>>) -> Json<Value> {
    let info = effective_info(&shared);
    Json(json!({
        "object": "list",
        "data": [{
            "id": info.served_model_name,
            "object": "model",
            "created": shared.created,
            "owned_by": "sglang",
            "root": info.served_model_name,
            "parent": null,
            "max_model_len": info.context_length,
        }],
    }))
}

async fn server_info(State(shared): State<Arc<Shared>>) -> Json<ServerInfo> {
    let delay = shared.lock().info_delay;
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
    Json(effective_info(&shared))
}

/// SGLang's per-dp-rank load list (one rank): `num_reqs` up to `max_running_requests`, the rest waiting.
async fn load(State(shared): State<Arc<Shared>>) -> Json<Value> {
    let generating = shared.lock().generating as u64;
    let running = generating.min(u64::from(effective_info(&shared).max_running_requests));
    Json(json!([{
        "dp_rank": 0,
        "num_reqs": running,
        "num_waiting_reqs": generating - running,
    }]))
}

/// SGLang's Anthropic `count_tokens`: the prompt tokens a chat request of the same conversation counts.
async fn count_tokens(Json(body): Json<Value>) -> Json<Value> {
    let words = |content: &Value| -> u64 {
        let count = |text: &str| text.split_whitespace().count() as u64;
        match content {
            Value::String(text) => count(text),
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .map(count)
                .sum(),
            _ => 0,
        }
    };
    let system = body.get("system").map_or(0, words);
    let messages: u64 = body
        .get("messages")
        .and_then(Value::as_array)
        .map_or(0, |messages| {
            messages
                .iter()
                .filter_map(|m| m.get("content"))
                .map(words)
                .sum()
        });
    Json(json!({ "input_tokens": system + messages }))
}

async fn abort_request(State(shared): State<Arc<Shared>>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(err) => {
            return error_response(StatusCode::BAD_REQUEST, "BadRequestError", &err.to_string())
        }
    };
    let rid = body
        .get("rid")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let abort_all = body
        .get("abort_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let delay = shared.lock().abort_delay;
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
    let mut held = shared.abort_hold.subscribe();
    if *held.borrow_and_update() {
        shared.lock().held_aborts.push(AbortRecord {
            rid: rid.clone(),
            abort_all,
        });
        let mut stop = shared.shutdown.subscribe();
        tokio::select! {
            _ = held.wait_for(|held| !*held) => {}
            _ = stop.wait_for(|stop| *stop) => {}
        }
    }
    let mut state = shared.lock();
    for live in state.live.values() {
        // SGLang scheduler predicate: `abort_all or req.rid.startswith(rid)`; "" prefixes everything.
        if abort_all || live.rid.starts_with(&rid) {
            live.abort.send_replace(true);
        }
    }
    state.aborts.push(AbortRecord { rid, abort_all });
    StatusCode::OK.into_response()
}

fn error_response(status: StatusCode, kind: &str, message: &str) -> Response {
    let body = json!({
        "object": "error",
        "message": message,
        "type": kind,
        "param": null,
        "code": status.as_u16(),
    });
    (status, Json(body)).into_response()
}

/// 200 headers whose body fails at once: the client sees the connection drop before a complete response.
fn broken_response() -> Response {
    let body = stream::once(async {
        Err::<String, _>(std::io::Error::other(
            "cere-mock-sglang: connection dropped",
        ))
    });
    Response::new(Body::from_stream(body))
}

/// SGLang's scheduler-side rejection of a stream: 200, one in-stream error chunk, then `[DONE]`.
fn stream_error_response(code: u16, message: &str) -> Response {
    let error = json!({
        "error": {
            "message": message,
            "type": error_type(code),
            "param": null,
            "code": code,
        }
    });
    let body = stream::iter([
        Ok::<_, std::io::Error>(format!("data: {error}\n\n")),
        Ok("data: [DONE]\n\n".to_owned()),
    ]);
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(body),
    )
        .into_response()
}

fn error_type(code: u16) -> &'static str {
    if code == 400 {
        "BadRequestError"
    } else {
        "InternalServerError"
    }
}

struct ChatRequest {
    rid: Option<String>,
    model: Option<String>,
    prompt_tokens: u64,
    max_new_tokens: Option<u64>,
    stops: Vec<String>,
    stream: bool,
    include_usage: bool,
    cache_salt: Option<String>,
    /// `(cache_salt, messages)` as one JSON string: the prefix-cache key.
    cache_key: String,
}

impl ChatRequest {
    fn parse(body: &Value) -> Self {
        let str_field = |key: &str| body.get(key).and_then(Value::as_str).map(str::to_owned);
        let prompt_tokens = body
            .get("messages")
            .and_then(Value::as_array)
            .map_or(0, |messages| {
                messages
                    .iter()
                    .filter_map(|m| m.get("content").and_then(Value::as_str))
                    .map(|content| content.split_whitespace().count() as u64)
                    .sum()
            });
        // SGLang: max_completion_tokens wins over max_tokens.
        let max_new_tokens = body
            .get("max_completion_tokens")
            .and_then(Value::as_u64)
            .or_else(|| body.get("max_tokens").and_then(Value::as_u64));
        let stops = match body.get("stop") {
            Some(Value::String(stop)) => vec![stop.clone()],
            Some(Value::Array(stops)) => stops
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
            _ => Vec::new(),
        };
        let cache_salt = str_field("cache_salt");
        let cache_key = json!([cache_salt, body.get("messages")]).to_string();
        Self {
            rid: str_field("rid"),
            model: str_field("model"),
            prompt_tokens,
            max_new_tokens,
            stops,
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            include_usage: body
                .pointer("/stream_options/include_usage")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            cache_salt,
            cache_key,
        }
    }
}

async fn chat(State(shared): State<Arc<Shared>>, bytes: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(err) => {
            return error_response(StatusCode::BAD_REQUEST, "BadRequestError", &err.to_string())
        }
    };
    let req = ChatRequest::parse(&body);
    let (abort_tx, abort_rx) = watch::channel(false);
    let (rid, fault, lag, info, scripted, guard) = {
        let mut state = shared.lock();
        if shared.config.record_bodies {
            state.bodies.push(bytes.to_vec());
        }
        let scripted = state.scripted.clone();
        let id = state.next_id;
        state.next_id += 1;
        let rid = req.rid.clone().unwrap_or_else(|| format!("{id:032x}"));
        state.dispatched.push(rid.clone());
        state.cache_salts.push(req.cache_salt.clone());
        let fault = state.fault.clone();
        let lag = state.load_lag;
        if let Fault::ServerError { status, remaining } = state.fault {
            state.fault = match remaining {
                0 | 1 => Fault::None,
                _ => Fault::ServerError {
                    status,
                    remaining: remaining - 1,
                },
            };
        }
        let live = LiveRequest {
            rid: rid.clone(),
            abort: abort_tx,
        };
        state.live.insert(id, live);
        state.max_in_flight = state.max_in_flight.max(state.live.len());
        let info = state
            .server_info_override
            .clone()
            .unwrap_or_else(|| shared.config.server_info.clone());
        (
            rid,
            fault,
            lag,
            info,
            scripted,
            LiveGuard {
                shared: Arc::clone(&shared),
                id,
            },
        )
    };

    if let Fault::ServerError { status, remaining } = fault {
        if remaining > 0 {
            return error_response(
                status,
                "InternalServerError",
                "cere-mock-sglang: injected server error",
            );
        }
    }
    let context_length = info.context_length;
    let overflow = match req.max_new_tokens {
        Some(max_new) => req.prompt_tokens.saturating_add(max_new) > context_length,
        None => req.prompt_tokens >= context_length,
    };
    if overflow {
        let message = format!(
            "The input ({} tokens) plus max_new_tokens ({}) is longer than the model's context length ({} tokens).",
            req.prompt_tokens,
            req.max_new_tokens.unwrap_or(0),
            context_length
        );
        return error_response(StatusCode::BAD_REQUEST, "BadRequestError", &message);
    }
    if let Fault::StreamError { code, message } = &fault {
        if req.stream {
            return stream_error_response(*code, message);
        }
        let status = StatusCode::from_u16(*code).unwrap_or(StatusCode::BAD_REQUEST);
        return error_response(status, error_type(*code), message);
    }
    if let Fault::Reject { code, message } = &fault {
        let status = StatusCode::from_u16(*code).unwrap_or(StatusCode::BAD_REQUEST);
        return error_response(status, error_type(*code), message);
    }
    if let Some(scripted) = scripted {
        drop(guard);
        if !req.stream {
            return Json(scripted.response).into_response();
        }
        let events: Vec<Result<String, std::io::Error>> = scripted
            .chunks
            .iter()
            .map(|chunk| Ok(format!("data: {chunk}\n\n")))
            .chain([Ok("data: [DONE]\n\n".to_owned())])
            .collect();
        return (
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            Body::from_stream(stream::iter(events)),
        )
            .into_response();
    }
    // Cached on arrival: a repeat that arrives while the first is still running already hits.
    let hit =
        !info.disable_radix_cache && !shared.lock().prefix_cache.insert(req.cache_key.clone());
    let mut shutdown_rx = shared.shutdown.subscribe();
    if fault == Fault::HangBeforeResponse {
        let _ = shutdown_rx.wait_for(|stop| *stop).await;
        return broken_response();
    }
    if !lag.is_zero() {
        // In transit (SGLang's tokenizer): no bytes and no `/get_load` count until the lag has passed.
        tokio::select! {
            () = tokio::time::sleep(lag) => {}
            _ = shutdown_rx.wait_for(|stop| *stop) => return broken_response(),
        }
    }
    // `MockHandle::hold_start`: in transit until the hold is cleared; an abort ends the wait (the generation then
    // ends at its first step).
    let mut start_hold = shared.start_hold.subscribe();
    let mut aborted = abort_rx.clone();
    tokio::select! {
        _ = start_hold.wait_for(|held| !*held) => {}
        _ = aborted.wait_for(|aborted| *aborted) => {}
        _ = shutdown_rx.wait_for(|stop| *stop) => return broken_response(),
    }

    let mut budget = req
        .max_new_tokens
        .unwrap_or(context_length - req.prompt_tokens);
    if let Fault::ClampOutput { tokens } = fault {
        budget = budget.min(u64::from(tokens));
    }
    let (pieces, planned) = plan(
        budget,
        shared.config.natural_output_tokens,
        &req.stops,
        shared.config.eos_token_id,
    );
    let granularity = shared.config.cache_granularity.max(1);
    let cached = if hit {
        req.prompt_tokens / granularity * granularity
    } else {
        0
    };
    let usage = Usage {
        prompt_tokens: req.prompt_tokens,
        // Live: a miss reports `prompt_tokens_details: null` even with the cache report on.
        cached_tokens: (info.enable_cache_report && cached > 0).then_some(cached),
    };
    let generation = Generation {
        pieces,
        planned,
        emitted: 0,
        latency: shared.config.token_latency,
        ignore_abort: matches!(fault, Fault::IgnoreAbort | Fault::HangAfterTokens { .. }),
        keep_generating: fault == Fault::KeepGeneratingAfterDisconnect,
        disconnect_after: match fault {
            Fault::DisconnectAfter { tokens } => Some(tokens as usize),
            _ => None,
        },
        hang_after: match fault {
            Fault::HangAfterTokens { tokens } => Some(tokens as usize),
            _ => None,
        },
        abort_rx,
        shutdown_rx,
        finish_hold: shared.finish_hold.subscribe(),
        generating: Some(GeneratingGuard::new(Arc::clone(&shared))),
        live: guard,
    };
    let head = Head {
        id: rid,
        created: unix_secs(),
        model: req.model.unwrap_or(info.served_model_name),
    };
    if req.stream {
        sse_response(Sse {
            generation,
            head,
            usage,
            include_usage: req.include_usage,
            phase: Phase::Role,
        })
    } else {
        unary_response(generation, head, usage).await
    }
}

/// Removes the request from the live set when the response finishes or the client goes away.
struct LiveGuard {
    shared: Arc<Shared>,
    id: u64,
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        self.shared.lock().live.remove(&self.id);
    }
}

/// Counts the request in `/get_load` while it generates.
struct GeneratingGuard {
    shared: Arc<Shared>,
}

impl GeneratingGuard {
    fn new(shared: Arc<Shared>) -> Self {
        shared.lock().generating += 1;
        Self { shared }
    }
}

impl Drop for GeneratingGuard {
    fn drop(&mut self) {
        self.shared.lock().generating -= 1;
    }
}

/// Counts a generation in [`crate::MockHandle::finish_held`] while it waits at the finish hold.
struct HeldGuard {
    shared: Arc<Shared>,
}

impl HeldGuard {
    fn new(shared: Arc<Shared>) -> Self {
        shared.lock().finish_held += 1;
        Self { shared }
    }
}

impl Drop for HeldGuard {
    fn drop(&mut self) {
        self.shared.lock().finish_held -= 1;
    }
}

#[derive(Clone)]
enum Finish {
    Stop(Value),
    Length,
    Abort,
}

impl Finish {
    fn reason(&self) -> &'static str {
        match self {
            Finish::Stop(_) => "stop",
            Finish::Length => "length",
            Finish::Abort => "abort",
        }
    }

    fn matched_stop(&self) -> Value {
        match self {
            Finish::Stop(matched) => matched.clone(),
            Finish::Length | Finish::Abort => Value::Null,
        }
    }
}

/// Output pieces `tok0`, ` tok1`, … up to the budget, natural EOS or the first stop-string match.
fn plan(budget: u64, natural: u32, stops: &[String], eos_token_id: u64) -> (Vec<String>, Finish) {
    let mut pieces = Vec::new();
    let mut content = String::new();
    let mut i = 0u64;
    loop {
        if i == budget {
            return (pieces, Finish::Length);
        }
        if i == u64::from(natural) {
            return (pieces, Finish::Stop(json!(eos_token_id)));
        }
        let start = content.len();
        content.push_str(&if i == 0 {
            "tok0".to_owned()
        } else {
            format!(" tok{i}")
        });
        let matched = stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| content.find(stop.as_str()).map(|at| (at, stop)))
            .min_by_key(|(at, _)| *at);
        if let Some((at, stop)) = matched {
            pieces.push(content[start..at.max(start)].to_owned());
            return (pieces, Finish::Stop(json!(stop)));
        }
        pieces.push(content[start..].to_owned());
        i += 1;
    }
}

struct Generation {
    pieces: Vec<String>,
    planned: Finish,
    emitted: usize,
    latency: Duration,
    ignore_abort: bool,
    /// [`Fault::KeepGeneratingAfterDisconnect`].
    keep_generating: bool,
    disconnect_after: Option<usize>,
    hang_after: Option<usize>,
    abort_rx: watch::Receiver<bool>,
    shutdown_rx: watch::Receiver<bool>,
    /// [`crate::MockHandle::hold_finish`].
    finish_hold: watch::Receiver<bool>,
    /// Present until the generation ends (finish, abort, disconnect or shutdown).
    generating: Option<GeneratingGuard>,
    live: LiveGuard,
}

/// Dropped with its generation still counting: the client went away mid-generation.
impl Drop for Generation {
    fn drop(&mut self) {
        let Some(generating) = self.generating.take() else {
            return;
        };
        if !self.keep_generating {
            return;
        }
        let left = u32::try_from(self.pieces.len() - self.emitted).unwrap_or(u32::MAX);
        let remaining = self.latency.saturating_mul(left);
        let mut shutdown_rx = self.shutdown_rx.clone();
        // Subscribed now: only a later `end_kept` ends this one.
        let mut end_rx = self.live.shared.kept_end.subscribe();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                tokio::select! {
                    () = tokio::time::sleep(remaining) => {}
                    _ = shutdown_rx.wait_for(|stop| *stop) => {}
                    _ = end_rx.changed() => {}
                }
                drop(generating);
            });
        }
    }
}

enum Step {
    Token(String),
    Finish(Finish),
    /// Drop the connection (fault, or shutdown while live).
    Disconnect,
}

impl Generation {
    async fn step(&mut self) -> Step {
        let step = self.advance().await;
        if !matches!(step, Step::Token(_)) {
            self.generating = None;
        }
        step
    }

    async fn advance(&mut self) -> Step {
        if !self.ignore_abort && *self.abort_rx.borrow() {
            return Step::Finish(Finish::Abort);
        }
        if self.hang_after == Some(self.emitted) {
            let _ = self.shutdown_rx.wait_for(|stop| *stop).await;
            return Step::Disconnect;
        }
        if self.disconnect_after == Some(self.emitted) {
            return Step::Disconnect;
        }
        if self.emitted == self.pieces.len() {
            return self.finish().await;
        }
        enum Wake {
            Abort,
            Shutdown,
            Tick,
        }
        let ignore_abort = self.ignore_abort;
        let abort_rx = &mut self.abort_rx;
        let aborted = async move {
            if ignore_abort || abort_rx.wait_for(|aborted| *aborted).await.is_err() {
                std::future::pending::<()>().await;
            }
        };
        let wake = tokio::select! {
            biased;
            () = aborted => Wake::Abort,
            _ = self.shutdown_rx.wait_for(|stop| *stop) => Wake::Shutdown,
            () = tokio::time::sleep(self.latency) => Wake::Tick,
        };
        match wake {
            Wake::Abort => Step::Finish(Finish::Abort),
            Wake::Shutdown => Step::Disconnect,
            Wake::Tick => {
                self.emitted += 1;
                Step::Token(self.pieces[self.emitted - 1].clone())
            }
        }
    }

    /// The planned finish, once [`crate::MockHandle::hold_finish`] lets it go; an abort or a shutdown ends a held
    /// generation first.
    async fn finish(&mut self) -> Step {
        if !*self.finish_hold.borrow() {
            return Step::Finish(self.planned.clone());
        }
        let _held = HeldGuard::new(Arc::clone(&self.live.shared));
        let ignore_abort = self.ignore_abort;
        let abort_rx = &mut self.abort_rx;
        let aborted = async move {
            if ignore_abort || abort_rx.wait_for(|aborted| *aborted).await.is_err() {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! {
            biased;
            () = aborted => Step::Finish(Finish::Abort),
            _ = self.shutdown_rx.wait_for(|stop| *stop) => Step::Disconnect,
            _ = self.finish_hold.wait_for(|held| !*held) => Step::Finish(self.planned.clone()),
        }
    }

    fn content(&self) -> String {
        self.pieces[..self.emitted].concat()
    }
}

struct Head {
    id: String,
    created: u64,
    model: String,
}

struct Usage {
    prompt_tokens: u64,
    cached_tokens: Option<u64>,
}

impl Usage {
    fn json(&self, completion_tokens: usize) -> Value {
        let completion_tokens = completion_tokens as u64;
        json!({
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.prompt_tokens + completion_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": self.cached_tokens.map(|cached| json!({ "cached_tokens": cached })),
            "reasoning_tokens": 0,
        })
    }
}

async fn unary_response(mut generation: Generation, head: Head, usage: Usage) -> Response {
    let finish = loop {
        match generation.step().await {
            Step::Token(_) => {}
            Step::Finish(finish) => break finish,
            Step::Disconnect => return broken_response(),
        }
    };
    Json(json!({
        "id": head.id,
        "object": "chat.completion",
        "created": head.created,
        "model": head.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generation.content(),
                "reasoning_content": null,
                "tool_calls": null,
            },
            "logprobs": null,
            "finish_reason": finish.reason(),
            "matched_stop": finish.matched_stop(),
        }],
        "usage": usage.json(generation.emitted),
        "metadata": { "weight_version": "default" },
    }))
    .into_response()
}

enum Phase {
    Role,
    Tokens,
    Usage,
    Done,
    End,
}

struct Sse {
    generation: Generation,
    head: Head,
    usage: Usage,
    include_usage: bool,
    phase: Phase,
}

impl Sse {
    fn chunk(&self, choices: Value, usage: Value) -> String {
        let chunk = json!({
            "id": self.head.id,
            "object": "chat.completion.chunk",
            "created": self.head.created,
            "model": self.head.model,
            "choices": choices,
            "usage": usage,
        });
        format!("data: {chunk}\n\n")
    }

    fn choice(&self, delta: Value, finish: Option<&Finish>) -> Value {
        json!([{
            "index": 0,
            "delta": delta,
            "logprobs": null,
            "finish_reason": finish.map(Finish::reason),
            "matched_stop": finish.map_or(Value::Null, Finish::matched_stop),
        }])
    }

    async fn next(&mut self) -> Option<std::io::Result<String>> {
        match self.phase {
            Phase::Role => {
                self.phase = Phase::Tokens;
                let delta =
                    json!({ "reasoning_content": null, "role": "assistant", "content": "" });
                Some(Ok(self.chunk(self.choice(delta, None), Value::Null)))
            }
            Phase::Tokens => match self.generation.step().await {
                Step::Token(piece) => {
                    let delta = json!({ "role": null, "content": piece, "reasoning_content": null, "tool_calls": null });
                    Some(Ok(self.chunk(self.choice(delta, None), Value::Null)))
                }
                Step::Finish(finish) => {
                    self.phase = if self.include_usage {
                        Phase::Usage
                    } else {
                        Phase::Done
                    };
                    let delta = json!({ "role": null, "content": null, "reasoning_content": null, "tool_calls": null });
                    Some(Ok(
                        self.chunk(self.choice(delta, Some(&finish)), Value::Null)
                    ))
                }
                Step::Disconnect => {
                    self.phase = Phase::End;
                    // One Pending poll lets hyper flush the chunks already produced; the error then drops the
                    // connection, so the client sees exactly the tokens sent before the "crash".
                    tokio::task::yield_now().await;
                    Some(Err(std::io::Error::other(
                        "cere-mock-sglang: connection dropped",
                    )))
                }
            },
            Phase::Usage => {
                self.phase = Phase::Done;
                let usage = self.usage.json(self.generation.emitted);
                Some(Ok(self.chunk(json!([]), usage)))
            }
            Phase::Done => {
                self.phase = Phase::End;
                Some(Ok("data: [DONE]\n\n".to_owned()))
            }
            Phase::End => None,
        }
    }
}

fn sse_response(sse: Sse) -> Response {
    let body = stream::unfold(sse, |mut sse| async move {
        sse.next().await.map(|item| (item, sse))
    });
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(body),
    )
        .into_response()
}
