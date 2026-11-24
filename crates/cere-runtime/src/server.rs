//! HTTP surface of `cered` (m1 design note A2–A4, L2, L4, L6, L7, S1–S6) and the per-request executor.
//!
//! A chat request is authenticated, gated on stopping/readiness, parsed and validated, digested (Q3), routed
//! (S4), and accepted by the ledger (L1/L2/S2, journaled before any backend call). An executor task then owns
//! the request until its terminal: admission (L4), dispatch, cancellation (L6: endpoint, deadline, client
//! disconnect, backpressure, shutdown) and the terminal journal write. It holds the admission slot until the
//! backend response has really ended. The handler only waits for the executor's HTTP response; dropping it
//! (client gone) is observed by the executor as a disconnect.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Path, RawQuery, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use crate::admission::{deadline_budget_ms, AdmissionError, Permit, Priority, Route, Target};
use crate::api_types::{
    ApiError, AttemptIdentity, CereExtension, ReadyzWorker, RequestUsage, INVALID_REQUEST,
    SERVER_ERROR,
};
use crate::app::{
    unix_ms, worker_field, App, Backend, NativeBackend, Replica, Served, REASON_BACKEND_ABORT,
    REASON_CANCEL_REQUESTED, REASON_CANCEL_UNACKNOWLEDGED, REASON_CLIENT_DISCONNECTED,
    REASON_DEADLINE_EXCEEDED, REASON_QUEUE_FULL, REASON_SHUTDOWN, REASON_WORKER_LOST,
    REASON_WORKER_NOT_READY,
};
use crate::digest::{RequestDigestInput, Salt};
use crate::document;
use crate::fed::client::AuthorityBase;
use crate::fed::remote::{Refusal, RemoteCall, RemoteWorker};
use crate::fed::wire;
use crate::ledger::{
    AcceptOutcome, AcceptRequest, CancelOutcome, Completion, LedgerError, RequestState,
    RetainedResponse, SessionClaim, WorkerTarget, REASON_JOURNAL_ERROR,
    REASON_SESSION_OWNER_SUPERSEDED, REASON_WORKER_EPOCH_SUPERSEDED,
};
use crate::native::serve::{self, NativeCancel};
use crate::pool::REASON_NO_ALLOWED_WORKER;
use crate::reload::Settings;
use crate::resume::AttachError;
use crate::session::{Revision, SessionError};
use crate::sglang::{BackendError, Rid};
use crate::surface::{CompatApi, CompatKind, ConvertCtx, ErrorView, OwnedError, Surface};
use crate::validate::{
    assistant_reply, classify_response, model_not_found, BackendBody, ChatMessage, ChatRequest,
    Finish, ToolCallAssembly,
};

pub const REQUEST_ID_HEADER: &str = "x-cere-request-id";
/// ApiLenient D4: the request fields accepted without effect (sorted, comma-separated).
pub const IGNORED_PARAMS_HEADER: &str = "x-cere-ignored-params";
const IDEMPOTENCY_KEY_HEADER: &str = "idempotency-key";
const DEADLINE_HEADER: &str = "x-cere-deadline-ms";
const SESSION_HEADER: &str = "x-cere-session-id";
const EXPECTED_REVISION_HEADER: &str = "x-cere-expected-revision";
const PRIORITY_HEADER: &str = "x-cere-priority";
const ON_DISCONNECT_HEADER: &str = "x-cere-on-disconnect";
/// Roadmap v0.7 A2: the client a request names for its log lines and metrics (`[[telemetry.clients]]`); a value no
/// rule names, or a non-ASCII one, is ignored (the rules then derive the client).
const CLIENT_HEADER: &str = "x-cere-client";
/// ab-split design note AB-03: the arm an operator names for a request (tests), with the operator token in
/// [`OPERATOR_AUTHORIZATION_HEADER`].
const AB_ARM_HEADER: &str = "x-cere-ab-arm";
/// WeightLab WL-14: the plan a reply ran under (`plan_sha256`), present only when a plan applied.
pub(crate) const RESEARCH_PLAN_HEADER: &str = "x-cere-research-plan";
const OPERATOR_AUTHORIZATION_HEADER: &str = "x-cere-operator-authorization";
const LAST_EVENT_ID_HEADER: &str = "last-event-id";
const MAX_IDEMPOTENCY_KEY_BYTES: usize = 255;

/// Routes of the M1 subset; `listen` is the bound address `GET /get_server_info` reports.
pub fn router(app: Arc<App>, listen: std::net::SocketAddr) -> Router {
    let mut router = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/v1/models/{*id}", get(model))
        // server-info design note SI-01: SGLang's server-info route, cered's own configuration.
        .route(
            "/get_server_info",
            get(move |state: State<Arc<App>>, headers: HeaderMap| {
                server_info(state, headers, listen)
            }),
        )
        .route("/v1/cere/requests/{id}", get(request_status))
        .route("/v1/cere/requests/{id}/events", get(request_events))
        .route("/v1/cere/requests/{id}/cancel", post(request_cancel))
        .route("/v1/cere/stats", get(scope_stats))
        .route("/v1/cere/sessions", post(create_session))
        .route("/v1/cere/sessions/{id}", get(get_session))
        .route("/v1/cere/sessions/{id}/fork", post(fork_session))
        .route("/v1/cere/state/export", post(state_export))
        .route("/v1/cere/state/import", post(state_import))
        .route(
            "/v1/cere/sessions/{id}/migrate",
            post(crate::operator::migrate),
        )
        .route(
            "/v1/cere/sessions/{id}/owner",
            get(crate::operator::session_owner),
        )
        .route("/v1/cere/workers/{id}/drain", post(crate::operator::drain))
        .route(
            "/v1/cere/workers/{id}/undrain",
            post(crate::operator::undrain),
        )
        .route("/v1/cere/workers/{id}/load", post(crate::operator::load))
        .route("/v1/cere/cards", get(crate::operator::cards))
        .route("/v1/cere/prefix_cache", get(crate::operator::prefix_cache))
        // ops-hardening design note OH-05: the Prometheus exposition (operator auth).
        .route("/metrics", get(crate::operator::metrics))
        .route(
            "/v1/cere/workers/{id}/unload",
            post(crate::operator::unload),
        )
        // m23 design note PI-04: the lever registry, the active set and the dev hot swap (operator auth).
        .route(
            "/v1/cere/levers",
            get(crate::operator::levers).post(crate::operator::set_levers),
        )
        .route(
            "/v1/cere/workers/{id}/flush_cache",
            post(crate::operator::flush_cache),
        )
        .route(
            "/v1/cere/config/reload",
            post(crate::operator::reload_config),
        );
    // Track LoRA: the vLLM adapter routes, only when `[deployments.native.lora]` is set.
    if app.lora.is_some() {
        router = router
            .route("/v1/load_lora_adapter", post(load_lora_adapter))
            .route("/v1/unload_lora_adapter", post(unload_lora_adapter));
    }
    // WeightLab WL-14 (`-v57`): the research routes (operator bearer), only when `[deployments.native.research]` is
    // set; without the table they do not exist (404 as any unknown path).
    if app.research.is_some() {
        router = router
            .route("/v1/cere/research/plans", post(research_post_plan))
            .route(
                "/v1/cere/research/plans/{id}",
                axum::routing::delete(research_delete_plan),
            )
            .route("/v1/cere/research/bindings", post(research_bind))
            .route(
                "/v1/cere/research/bindings/{scope}",
                axum::routing::delete(research_unbind),
            )
            .route("/v1/cere/research/captures", get(research_captures))
            .route(
                "/v1/cere/research/captures/{id}",
                get(research_drain_capture),
            )
            // WL-B (track WeightLab B, native/research_vectors.rs): the vector routes.
            .route("/v1/cere/research/vectors", post(research_post_vector))
            .route(
                "/v1/cere/research/vectors/{id}",
                axum::routing::delete(research_delete_vector),
            );
    }
    for (kind, routes) in COMPAT_ROUTES {
        if compat_enabled(&app, *kind) {
            router = routes(router);
        }
    }
    // OH-05/OH-06: every route (the fallback too) is counted and access-logged; added last so it wraps them all.
    router
        .layer(axum::middleware::from_fn_with_state(
            app.clone(),
            crate::metrics::track,
        ))
        .with_state(app)
}

/// Serves `listener` until `signal` resolves, then runs the L8 shutdown and stops the HTTP server.
/// Returns whether the shutdown was clean (the process exit status).
pub async fn serve(
    app: Arc<App>,
    listener: tokio::net::TcpListener,
    signal: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<bool> {
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let listen = listener.local_addr()?;
    let server =
        axum::serve(listener, router(app.clone(), listen)).with_graceful_shutdown(async move {
            let _ = stop_rx.await;
        });
    let mut server = tokio::spawn(async move { server.await });
    tokio::select! {
        _ = signal => {}
        joined = &mut server => {
            return match joined {
                Ok(Ok(())) => Err(std::io::Error::other("HTTP server stopped unexpectedly")),
                Ok(Err(e)) => Err(e),
                Err(e) => Err(std::io::Error::other(e)),
            };
        }
    }
    // New requests keep getting 503 `shutting_down` while in-flight ones drain.
    let clean = app.shutdown().await;
    let _ = stop_tx.send(());
    match tokio::time::timeout(Duration::from_secs(5), &mut server).await {
        Ok(Ok(Ok(()))) => Ok(clean),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(e)) => Err(std::io::Error::other(e)),
        Err(_) => {
            server.abort();
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Response helpers

pub(crate) fn json_response(status: StatusCode, body: &Value) -> Response {
    (status, Json(body)).into_response()
}

pub(crate) fn api_error(error: &ApiError) -> Response {
    json_response(status_of(error.status), &error.body())
}

pub(crate) fn status_of(code: u16) -> StatusCode {
    StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

pub(crate) fn with_request_id(mut response: Response, rid: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(rid) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

fn status_url(rid: &str) -> String {
    format!("/v1/cere/requests/{rid}")
}

fn events_url(rid: &str) -> String {
    format!("/v1/cere/requests/{rid}/events")
}

/// 409 for a keyed retry: the error names the original request's status URL (and, for a continue-mode stream,
/// its resume URL).
fn conflict(code: &'static str, message: &str, rid: &str, resumable: bool) -> Response {
    let mut body = ApiError::new(409, INVALID_REQUEST, code, message, None).body();
    body["error"]["status_url"] = json!(status_url(rid));
    if resumable {
        body["error"]["events_url"] = json!(events_url(rid));
    }
    with_request_id(json_response(StatusCode::CONFLICT, &body), rid)
}

fn stream_not_replayable(rid: &str, resumable: bool) -> Response {
    conflict(
        "stream_not_replayable",
        "the original request was streamed and cannot be replayed",
        rid,
        resumable,
    )
}

pub(crate) fn unauthorized() -> Response {
    let mut response = api_error(&ApiError::new(
        401,
        INVALID_REQUEST,
        "unauthorized",
        "missing or invalid bearer token",
        None,
    ));
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

fn request_not_found() -> Response {
    api_error(&ApiError::new(
        404,
        INVALID_REQUEST,
        "request_not_found",
        "request not found",
        None,
    ))
}

fn session_error(e: SessionError) -> ApiError {
    let (status, message) = match e {
        SessionError::NotFound => (404, "session not found"),
        SessionError::SessionLimit => (429, "session limit reached for this scope"),
        SessionError::RevisionLimit => (409, "session revision limit reached"),
        SessionError::RevisionConflict => (
            409,
            "expected revision is not current or the message history does not match it",
        ),
        SessionError::LeaseHeld => (409, "another request is writing this session"),
        SessionError::OwnerSuperseded => (409, OWNER_SUPERSEDED_MESSAGE),
        SessionError::ModelMismatch => {
            return ApiError::new(
                409,
                INVALID_REQUEST,
                e.code(),
                "the session is bound to another model than this request's",
                Some("model".to_owned()),
            )
        }
    };
    ApiError::new(status, INVALID_REQUEST, e.code(), message, None)
}

/// m4 design note Q7: the commit (unary) or the terminal (stream, status) of a turn whose session owner moved.
const OWNER_SUPERSEDED_MESSAGE: &str =
    "the session was migrated to another worker after this turn started; the turn was not committed";

fn internal_error(message: impl Into<String>) -> ApiError {
    ApiError::new(500, SERVER_ERROR, "internal_error", message, None)
}

/// Audit F6: encodes of requests not yet granted, in flight at once, process-wide.
const MAX_PRE_GRANT_ENCODES: usize = 8;

static PRE_GRANT_ENCODES: tokio::sync::Semaphore =
    tokio::sync::Semaphore::const_new(MAX_PRE_GRANT_ENCODES);

/// `503 overloaded` with `Retry-After: 1`: a bound that refuses before admission.
fn overloaded(message: &str) -> Response {
    let mut response = api_error(&ApiError::new(
        503,
        SERVER_ERROR,
        "overloaded",
        message,
        None,
    ));
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    response
}

/// Audit F5: the answer to a request whose document extraction did not run (see [`document::run`]).
pub(crate) fn offload_response(refused: document::Offload) -> Response {
    match refused {
        document::Offload::Busy => overloaded("document extraction is busy; retry later"),
        document::Offload::Failed => api_error(&internal_error("document extraction failed")),
    }
}

/// [`offload_response`] for a compat surface's converter, whose errors are [`OwnedError`]s.
pub(crate) fn offload_owned_error(refused: document::Offload) -> OwnedError {
    let (status, code, message) = match refused {
        document::Offload::Busy => (
            503,
            "overloaded",
            "document extraction is busy; retry later",
        ),
        document::Offload::Failed => (500, "internal_error", "document extraction failed"),
    };
    OwnedError {
        status,
        kind: SERVER_ERROR.to_owned(),
        code: code.to_owned(),
        message: message.to_owned(),
        param: None,
    }
}

/// HTTP view of a ledger refusal (L1 bound, L2 outcomes, S1/S2 session errors).
pub(crate) fn ledger_error(e: &LedgerError) -> Response {
    match e {
        LedgerError::Overloaded => overloaded("request ledger is full; retry later"),
        LedgerError::ResponseNotRetained { rid } => conflict(
            "response_not_retained",
            "the original response is no longer retained; query its status",
            rid,
            false,
        ),
        LedgerError::RequestInProgress { rid } => conflict(
            "request_in_progress",
            "a request with this idempotency key is still in progress",
            rid,
            false,
        ),
        LedgerError::StreamNotReplayable { rid } => stream_not_replayable(rid, false),
        LedgerError::IdempotencyKeyExpired => api_error(&ApiError::new(
            409,
            INVALID_REQUEST,
            "idempotency_key_expired",
            "this idempotency key's record expired; it is never regenerated",
            None,
        )),
        LedgerError::IdempotencyConflict => api_error(&ApiError::new(
            409,
            INVALID_REQUEST,
            "idempotency_conflict",
            "this idempotency key was used with a different request",
            None,
        )),
        LedgerError::Session(s) => api_error(&session_error(*s)),
        LedgerError::NotFound => request_not_found(),
        other => api_error(&internal_error(format!("ledger error: {}", other.code()))),
    }
}

// ---------------------------------------------------------------------------------------------------------
// Auth and serving gates

fn authenticate(app: &App, headers: &HeaderMap) -> Result<String, Response> {
    app.auth
        .authenticate(
            headers
                .get(header::AUTHORIZATION)
                .map(HeaderValue::as_bytes),
        )
        .map(str::to_owned)
        .ok_or_else(unauthorized)
}

pub(crate) fn shutting_down() -> Response {
    api_error(&ApiError::new(
        503,
        SERVER_ERROR,
        "shutting_down",
        "server is shutting down",
        None,
    ))
}

// ---------------------------------------------------------------------------------------------------------
// Health and readiness (A2)

async fn healthz() -> Response {
    json_response(StatusCode::OK, &json!({ "status": "alive" }))
}

/// Q11: process-level fields plus one entry per worker, local and remote (M3.5 P11: the same keys, 15 with M3.6
/// Q11's `model`, null for a local worker); 200 iff at least one worker (local or remote) is ready, the process is
/// not stopping and no process-level problem (a poisoned journal) stands. m32n design note C8/R12: a deployment
/// with native replicas adds `capacity` (the ready cards' share of the configured ones); the single form and every
/// other kind answer without it.
async fn readyz(State(app): State<Arc<App>>) -> Response {
    let (ready, body) = readyz_report(&app);
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    json_response(status, &body)
}

/// The `/readyz` verdict and body ([`readyz`]); `GET /metrics` exports the same verdict (`cere_ready`) and each
/// worker's `ready` (`cere_worker_up`).
pub(crate) fn readyz_report(app: &App) -> (bool, Value) {
    let stopping = app.is_stopping();
    let problems = app.process_problems();
    let mut tenancy = Value::Null;
    let mut workers = Map::new();
    let mut any_ready = false;
    let mut capacity: Option<Capacity> = None;
    for replica in app.replicas() {
        let snapshot = replica.readiness().snapshot();
        let (state, epoch) = app
            .worker_state(replica.worker_id())
            .map_or(("unknown", 0), |(s, e)| (s.as_str(), e));
        // Ready means accepting new work on the latest probe: the probe passed and the registry agrees.
        let ready = snapshot.ready && state == "ready";
        any_ready |= ready;
        if let Some(card) = app.native_card(replica.worker_id()) {
            capacity.get_or_insert_default().add(&card, ready);
        }
        if tenancy.is_null() {
            tenancy = json!(snapshot.tenancy);
        }
        let mut entry = json!(ReadyzWorker::local(snapshot, state, epoch, ready));
        // BC1 leftover: the loaded native model's prefill selection signature (absent when none is reported).
        if let Some(signature) = replica
            .backend
            .native()
            .and_then(|n| n.prefill_signature.clone())
        {
            entry["prefill_signature"] = json!(signature);
        }
        workers.insert(replica.worker_id().to_owned(), entry);
    }
    let now = unix_ms();
    for (config, state) in app.remote_entries(now) {
        let problems = app.remote_problems(&state.worker_id, state.round);
        let entry = ReadyzWorker::remote(config, state, problems, now);
        // P11: a Ready remote worker accepts new work like a ready local one.
        any_ready |= entry.ready;
        workers.insert(config.worker_id.clone(), json!(entry));
    }
    let ready = any_ready && !stopping && problems.is_empty();
    let mut body = json!({
        "ready": ready,
        "stopping": stopping,
        "problems": problems,
        "tenancy": tenancy,
        "workers": workers,
    });
    if let Some(capacity) = capacity {
        body["capacity"] = capacity.render();
    }
    (ready, body)
}

/// m32n design note C8/R12: `/readyz` `capacity` of the native replicas: cards, decode rows and KV pool tokens,
/// ready (on the latest probe, by the registry too) against configured.
#[derive(Default)]
struct Capacity {
    workers: (u64, u64),
    rows: (u64, u64),
    kv_pool_tokens: (u64, u64),
}

impl Capacity {
    fn add(&mut self, card: &crate::app::NativeCard, ready: bool) {
        let ready = u64::from(ready);
        self.workers.0 += ready;
        self.workers.1 += 1;
        self.rows.0 += ready * card.rows_max;
        self.rows.1 += card.rows_max;
        self.kv_pool_tokens.0 += ready * card.kv_pool_tokens;
        self.kv_pool_tokens.1 += card.kv_pool_tokens;
    }

    fn render(&self) -> Value {
        json!({
            "workers_ready": self.workers.0,
            "workers_configured": self.workers.1,
            "rows": { "ready": self.rows.0, "configured": self.rows.1 },
            "kv_pool_tokens": { "ready": self.kv_pool_tokens.0, "configured": self.kv_pool_tokens.1 },
        })
    }
}

// ---------------------------------------------------------------------------------------------------------
// Model discovery (OpenAI `GET /v1/models`)

/// A served model's OpenAI model object. `created` is 0 (no meaningful creation time); `max_model_len` is the
/// deployment's context limit, the field SGLang and vLLM report.
fn model_card(served: &Served) -> Value {
    json!({"id": served.model, "object": "model", "created": 0, "owned_by": "cere",
           "max_model_len": served.max_context_tokens})
}

/// Track LoRA: a loaded adapter's card (vLLM's: `parent` the base model, `root` the adapter directory).
fn adapter_card(app: &App, adapter: &crate::native::lora::Adapter) -> Value {
    let base = app.lora_deployment().map(|d| &app.served()[d]);
    json!({"id": adapter.name, "object": "model", "created": adapter.created, "owned_by": "cere",
           "parent": base.as_ref().map(|s| &s.model), "root": adapter.path,
           "max_model_len": base.as_ref().map(|s| s.max_context_tokens)})
}

/// One card per `[[deployments]]` entry, in config order (M5 D5), then one per loaded LoRA adapter in name order.
async fn models(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(response) = authenticate(&app, &headers) {
        return response;
    }
    let mut data: Vec<Value> = app.served().iter().map(model_card).collect();
    if let Some(lora) = &app.lora {
        data.extend(lora.list().iter().map(|a| adapter_card(&app, a)));
    }
    json_response(StatusCode::OK, &json!({"object": "list", "data": data}))
}

/// server-info design note SI-01: `GET /get_server_info`, behind the same auth as `/v1/models`.
async fn server_info(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    listen: std::net::SocketAddr,
) -> Response {
    if let Err(response) = authenticate(&app, &headers) {
        return response;
    }
    json_response(StatusCode::OK, &crate::server_info::report(&app, listen))
}

/// `{*id}` takes the rest of the path: a served name may contain `/` (`org/model`, sent unencoded by OpenAI clients).
async fn model(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = authenticate(&app, &headers) {
        return response;
    }
    if let Some(d) = app.deployment_of_model(&id) {
        return json_response(StatusCode::OK, &model_card(&app.served()[d]));
    }
    match app.lora_adapter(&id) {
        Some((_, adapter)) => json_response(StatusCode::OK, &adapter_card(&app, &adapter)),
        None => api_error(&model_not_found(
            &id,
            &app.settings().policies[0].served_models,
        )),
    }
}

// ---------------------------------------------------------------------------------------------------------
// NetTeleport S1: state export and import (client auth; crate::native::teleport_http)

async fn state_export(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    if app.is_stopping() {
        return shutting_down();
    }
    crate::native::teleport_http::export(app, scope, headers, body).await
}

async fn state_import(
    State(app): State<Arc<App>>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    if app.is_stopping() {
        return shutting_down();
    }
    crate::native::teleport_http::import(app, scope, headers, query, body).await
}

// ---------------------------------------------------------------------------------------------------------
// LoRA adapters (track LoRA; vLLM's dynamic-adapter routes)

#[derive(serde::Deserialize)]
struct LoraRequest {
    lora_name: String,
    #[serde(default)]
    lora_path: Option<String>,
}

fn lora_error(error: crate::native::lora::LoraError) -> Response {
    use crate::native::lora::LoraError;
    let (status, kind, code) = match &error {
        LoraError::Invalid(_) => (400, INVALID_REQUEST, "invalid_lora_adapter"),
        LoraError::NotFound(_) => (404, INVALID_REQUEST, "lora_adapter_not_found"),
        LoraError::Busy(_) => (409, INVALID_REQUEST, "lora_adapter_busy"),
        LoraError::NotReady => (503, SERVER_ERROR, "not_ready"),
        LoraError::Internal(_) => (500, SERVER_ERROR, "lora_adapter_failed"),
    };
    api_error(&ApiError::new(status, kind, code, error.to_string(), None))
}

/// The body of a load/unload request (client auth, as `/v1/models`: an external deploy tool sends no key).
async fn lora_request(app: &App, headers: &HeaderMap, body: Body) -> Result<LoraRequest, Response> {
    authenticate(app, headers)?;
    if app.is_stopping() {
        return Err(shutting_down());
    }
    let limit = app.config.api.max_request_body_bytes as usize;
    let bytes = axum::body::to_bytes(body, limit).await.map_err(|_| {
        api_error(&ApiError::invalid_json(format!(
            "request body exceeds {limit} bytes"
        )))
    })?;
    serde_json::from_slice(&bytes).map_err(|e| api_error(&ApiError::invalid_json(e.to_string())))
}

/// `POST /v1/load_lora_adapter {lora_name, lora_path}`: parses the PEFT adapter under `adapter_root` and loads it
/// into a free pool slot; requests then name it as their `model`.
async fn load_lora_adapter(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let request = match lora_request(&app, &headers, body).await {
        Ok(r) => r,
        Err(response) => return response,
    };
    let Some(path) = request.lora_path else {
        return api_error(&ApiError::invalid_value(
            "lora_path",
            "lora_path is required",
        ));
    };
    let worker = app.clone();
    let name = request.lora_name;
    let loaded = tokio::task::spawn_blocking(move || {
        let lora = worker.lora.as_ref().expect("routed only with lora");
        let served: Vec<&str> = worker.served().iter().map(|s| s.model.as_str()).collect();
        lora.load(&name, &path, &served, || worker.native())
            .map(|a| a.name.clone())
    })
    .await;
    match loaded {
        Ok(Ok(name)) => (
            StatusCode::OK,
            format!("Success: LoRA adapter '{name}' added successfully."),
        )
            .into_response(),
        Ok(Err(e)) => lora_error(e),
        Err(e) => lora_error(crate::native::lora::LoraError::Internal(e.to_string())),
    }
}

/// `POST /v1/unload_lora_adapter {lora_name}`: frees the adapter's slot; 409 while a request uses it.
async fn unload_lora_adapter(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let request = match lora_request(&app, &headers, body).await {
        Ok(r) => r,
        Err(response) => return response,
    };
    let worker = app.clone();
    let name = request.lora_name.clone();
    let unloaded = tokio::task::spawn_blocking(move || {
        let lora = worker.lora.as_ref().expect("routed only with lora");
        lora.unload(&name, || worker.native())
    })
    .await;
    match unloaded {
        Ok(Ok(())) => (
            StatusCode::OK,
            format!(
                "Success: LoRA adapter '{}' removed successfully.",
                request.lora_name
            ),
        )
            .into_response(),
        Ok(Err(e)) => lora_error(e),
        Err(e) => lora_error(crate::native::lora::LoraError::Internal(e.to_string())),
    }
}

// ---------------------------------------------------------------------------------------------------------
// WeightLab research routes (WL-14, `-v57`, docs/API.md "Research (WeightLab)"): operator bearer, as the other
// operator routes (404 without an operator token configured, 403 a client scope token, 401 none or unknown).

/// A refused research operation: `{"error": {...}}`, plus `bound` and `cap` for `research_budget_exceeded` and
/// `bound` and `free` for `research_buffer_full`.
pub(crate) fn research_error(error: &crate::native::research::ResearchError) -> Response {
    use crate::native::research::ResearchError;
    let (status, code) = error.status_code();
    let kind = if status >= 500 {
        SERVER_ERROR
    } else {
        INVALID_REQUEST
    };
    let mut body = ApiError::new(status, kind, code, error.to_string(), None).body();
    let extra = match error {
        ResearchError::Budget { bound, cap } => json!({"bound": bound, "cap": cap}),
        ResearchError::BufferFull { bound, free } => json!({"bound": bound, "free": free}),
        _ => json!({}),
    };
    if let (Some(e), Value::Object(extra)) =
        (body.get_mut("error").and_then(Value::as_object_mut), extra)
    {
        e.extend(extra);
    }
    json_response(status_of(status), &body)
}

/// The research registry after the operator check; the routes exist only with it.
fn research_of(
    app: &App,
    headers: &HeaderMap,
) -> Result<Arc<crate::native::research::Research>, Response> {
    crate::operator::authorize(app, headers)?;
    if app.is_stopping() {
        return Err(shutting_down());
    }
    Ok(app.research.clone().expect("routed only with research"))
}

async fn research_body(app: &App, body: Body) -> Result<Bytes, Response> {
    let limit = app.config.api.max_request_body_bytes as usize;
    axum::body::to_bytes(body, limit).await.map_err(|_| {
        api_error(&ApiError::invalid_json(format!(
            "request body exceeds {limit} bytes"
        )))
    })
}

/// `?key=value` of a query (the research routes' values are ids and names: no percent-decoding).
fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
    query?
        .split('&')
        .find_map(|pair| pair.strip_prefix(key)?.strip_prefix('='))
}

/// `POST /v1/cere/research/plans`: 200 `{plan_id, plan_sha256, capture_bound_bytes}`.
async fn research_post_plan(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let research = match research_of(&app, &headers) {
        Ok(r) => r,
        Err(response) => return response,
    };
    let bytes = match research_body(&app, body).await {
        Ok(b) => b,
        Err(response) => return response,
    };
    let worker = app.clone();
    let registered =
        tokio::task::spawn_blocking(move || research.register(&bytes, || worker.native())).await;
    match registered {
        Ok(Ok(p)) => json_response(
            StatusCode::OK,
            &json!({
                "plan_id": p.id,
                "plan_sha256": p.plan.sha256_hex(),
                "capture_bound_bytes": p.plan.capture_bound_bytes(),
            }),
        ),
        Ok(Err(e)) => research_error(&e),
        Err(e) => research_error(&crate::native::research::ResearchError::Internal(
            e.to_string(),
        )),
    }
}

/// `DELETE /v1/cere/research/plans/{id}`: 200 `{plan_id, deleted: true}`.
async fn research_delete_plan(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let research = match research_of(&app, &headers) {
        Ok(r) => r,
        Err(response) => return response,
    };
    let worker = app.clone();
    let plan = id.clone();
    let deleted =
        tokio::task::spawn_blocking(move || research.delete_plan(&plan, || worker.native())).await;
    match deleted {
        Ok(Ok(())) => json_response(StatusCode::OK, &json!({"plan_id": id, "deleted": true})),
        Ok(Err(e)) => research_error(&e),
        Err(e) => research_error(&crate::native::research::ResearchError::Internal(
            e.to_string(),
        )),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingRequest {
    scope: String,
    plan_id: String,
}

/// `POST /v1/cere/research/bindings {scope, plan_id}`: 200 `{scope, plan_id}`.
async fn research_bind(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    let research = match research_of(&app, &headers) {
        Ok(r) => r,
        Err(response) => return response,
    };
    let bytes = match research_body(&app, body).await {
        Ok(b) => b,
        Err(response) => return response,
    };
    let request: BindingRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => return api_error(&ApiError::invalid_json(e.to_string())),
    };
    match research.bind(&request.scope, &request.plan_id) {
        Ok(()) => json_response(
            StatusCode::OK,
            &json!({"scope": request.scope, "plan_id": request.plan_id}),
        ),
        Err(e) => research_error(&e),
    }
}

/// `DELETE /v1/cere/research/bindings/{scope}`: 200 `{scope, unbound: true}`.
async fn research_unbind(
    State(app): State<Arc<App>>,
    Path(scope): Path<String>,
    headers: HeaderMap,
) -> Response {
    let research = match research_of(&app, &headers) {
        Ok(r) => r,
        Err(response) => return response,
    };
    match research.unbind(&scope) {
        Ok(()) => json_response(StatusCode::OK, &json!({"scope": scope, "unbound": true})),
        Err(e) => research_error(&e),
    }
}

/// `GET /v1/cere/research/captures?scope=&after=`: 200 `{"captures": [manifest, ...]}` (undrained, finish order).
async fn research_captures(
    State(app): State<Arc<App>>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
) -> Response {
    let research = match research_of(&app, &headers) {
        Ok(r) => r,
        Err(response) => return response,
    };
    let (scope, after) = (
        query_value(query.as_deref(), "scope"),
        query_value(query.as_deref(), "after"),
    );
    match research.list(scope, after) {
        Ok(captures) => json_response(StatusCode::OK, &json!({"captures": captures})),
        Err(e) => research_error(&e),
    }
}

/// `GET /v1/cere/research/captures/{request_id}`: the capture's safetensors file, once (then freed).
async fn research_drain_capture(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let research = match research_of(&app, &headers) {
        Ok(r) => r,
        Err(response) => return response,
    };
    match research.drain(&id) {
        Ok(file) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            file,
        )
            .into_response(),
        Err(e) => research_error(&e),
    }
}

/// `POST /v1/cere/research/vectors?name=<name>` (track WL-B's store and answer).
async fn research_post_vector(
    State(app): State<Arc<App>>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let research = match research_of(&app, &headers) {
        Ok(r) => r,
        Err(response) => return response,
    };
    let bytes = match research_body(&app, body).await {
        Ok(b) => b,
        Err(response) => return response,
    };
    crate::native::research_vectors::post_vector_response(
        &research.vectors,
        query_value(query.as_deref(), "name"),
        &bytes,
    )
}

/// `DELETE /v1/cere/research/vectors/{id}` (track WL-B's store and answer).
async fn research_delete_vector(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    match research_of(&app, &headers) {
        Ok(research) => {
            crate::native::research_vectors::delete_vector_response(&research.vectors, &id)
        }
        Err(response) => response,
    }
}

// ---------------------------------------------------------------------------------------------------------
// Status and cancel (L6, L7)

async fn request_status(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    let Some(s) = app.ledger().status(&scope, &id) else {
        return request_not_found();
    };
    json_response(
        StatusCode::OK,
        &json!({
            "request_id": s.rid,
            "state": s.state.as_str(),
            "reason": s.reason,
            "attempt": s.attempt,
            "epoch": s.epoch,
            "worker_id": s.worker_id,
            "worker_epoch": s.worker_epoch,
            "stream": s.stream,
            "session_id": s.session_id,
            "execution_mode": app.deployment(0).execution_mode,
            "timings": s.timings,
            "usage": s.usage,
        }),
    )
}

/// The resume cursor (m1close Q4): `?after=<seq>` wins over `Last-Event-ID`; neither -> from the first event.
fn resume_cursor(query: Option<&str>, headers: &HeaderMap) -> Result<Option<u64>, ApiError> {
    let after = query
        .into_iter()
        .flat_map(|q| q.split('&'))
        .find_map(|pair| pair.strip_prefix("after="));
    if let Some(after) = after {
        return after
            .parse()
            .map(Some)
            .map_err(|_| ApiError::invalid_value("after", "after must be an unsigned integer"));
    }
    parse_u64_header(headers, LAST_EVENT_ID_HEADER)
}

fn resume_refused(status: u16, code: &'static str, message: &str) -> Response {
    api_error(&ApiError::new(status, INVALID_REQUEST, code, message, None))
}

/// Q3-Q5: replays a continue-mode stream's retained events after the cursor, then follows it live to `[DONE]`.
/// The new reader supersedes the current one.
async fn request_events(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    let cursor = match resume_cursor(query.as_deref(), &headers) {
        Ok(cursor) => cursor,
        Err(e) => return api_error(&e),
    };
    // Ledger-expired → 404 by the ledger's own retention rule. A buffer's tombstone outlives that expiry (the
    // terminal is recorded before the buffer ends, and the tombstone lasts the same retention from its later
    // release), so a rid still known here never lost its tombstone to the sweep.
    let Some(s) = app.ledger().status_at(&scope, &id, unix_ms()) else {
        return request_not_found();
    };
    let not_resumable = || {
        resume_refused(
            409,
            "not_resumable",
            "the request is not a stream sent with X-Cere-On-Disconnect: continue",
        )
    };
    let cursor_expired = || {
        resume_refused(
            410,
            "cursor_expired",
            "the events after this cursor are no longer retained",
        )
    };
    if !s.stream {
        return not_resumable();
    }
    if s.surface.is_some() {
        return stream_not_replayable(&id, false);
    }
    if s.epoch != app.epoch() {
        return cursor_expired();
    }
    let keepalive = app.tuning.keepalive_interval;
    match app.resume.attach(&scope, &id, cursor, keepalive) {
        Ok(reader) => crate::stream::sse_response(reader.into_body(), &id),
        Err(AttachError::NotResumable) => not_resumable(),
        Err(AttachError::CursorExpired) => cursor_expired(),
    }
}

/// Q5: the caller's scope only, since process start. Q11: per worker its state, epoch, cap and the caller scope's
/// in-flight on it; m32n design note D18: a native replica of a multi-replica deployment adds its card
/// (`device_uuid`, `rows_max`, `kv_pool_tokens`) and the caller scope's terminals there (`requests`,
/// `completion_tokens`, `cached_tokens`). The process-wide per-card load, which reveals every scope's traffic
/// (F15), is the operator's `GET /v1/cere/cards`.
async fn scope_stats(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    let (stats, since_unix_ms) = {
        let ledger = app.ledger();
        (ledger.stats(&scope), ledger.opened_ms())
    };
    let workers: Map<String, Value> = app
        .admission
        .worker_loads(&scope)
        .into_iter()
        .map(|w| {
            let (state, epoch) = app
                .worker_state(&w.id)
                .map_or(("unknown", 0), |(s, e)| (s.as_str(), e));
            let mut body = json!({
                "state": state,
                "epoch": epoch,
                "cap": w.cap,
                "inflight": w.scope_inflight,
            });
            if let Some(card) = app.native_card(&w.id) {
                let mine = stats.workers.get(&*w.id).cloned().unwrap_or_default();
                body["device_uuid"] = json!(card.device_uuid);
                body["rows_max"] = json!(card.rows_max);
                body["kv_pool_tokens"] = json!(card.kv_pool_tokens);
                body["requests"] = json!(mine.requests);
                body["completion_tokens"] = json!(mine.completion_tokens);
                body["cached_tokens"] = json!(mine.cached_tokens);
            }
            (w.id.to_string(), body)
        })
        .collect();
    let settings = app.settings();
    let mut body = json!({
        "scope": scope,
        "requests": stats.requests,
        "prompt_tokens_total": stats.prompt_tokens_total,
        "cached_tokens_total": stats.cached_tokens_total,
        "cache_reported_requests": stats.cache_reported_requests,
        "cache_unavailable_requests": stats.cache_unavailable_requests,
        "since_unix_ms": since_unix_ms,
        "native_submissions": app.native_submissions(),
        // m29 design note D11: cumulative speculate rounds and accepted drafts (process-wide, like submissions).
        "native_spec_verify_steps": app.native_spec_verify_steps(),
        "native_spec_accepted_drafts": app.native_spec_accepted_drafts(),
        "native_preemptions": app.native_preemptions(),
        // sessions design note SE-09: generations that grew their session (`reserve_output_tokens`).
        "native_grows": app.native_grows(),
        // m23 design note PI-05: every registry lever's active impl (null without a loaded native backend).
        "levers": app.native_levers(),
        // BC1 leftover: the loaded native model's prefill selection signature (null without one).
        "prefill_signature": app.native_prefill_signature(),
        // Live config reload: the applied settings' generation (0 = as started) and their config's hash (the
        // startup `config_hash` at generation 0).
        "config_generation": settings.generation,
        "effective_config_hash": settings.config_hash,
        "workers": workers,
    });
    // M2.8 D8: only a native backend with a grammar process reports `grammar`.
    if let Some(grammar) = app.native_grammar_stats() {
        body["grammar"] = grammar;
    }
    // loop-guard design note LG-04: only a native backend with the loop guard reports `native_loop_guard`.
    if let Some(guard) = app.native_loop_guard() {
        body["native_loop_guard"] = guard;
    }
    // ab-split design note AB-04: per served model with A/B arms, per arm (process-wide, like the native
    // counters); absent when no deployment has arms.
    let models: Vec<&str> = app.served().iter().map(|s| s.model.as_str()).collect();
    if let Some(ab) = app.ab.render(&settings.ab_plans, &models) {
        body["ab"] = ab;
    }
    json_response(StatusCode::OK, &body)
}

async fn request_cancel(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    let (status, state) = match app.cancel(&scope, &id, REASON_CANCEL_REQUESTED) {
        Ok(CancelOutcome::Cancelled) => (StatusCode::ACCEPTED, RequestState::Cancelled),
        Ok(CancelOutcome::AbortRequired | CancelOutcome::AlreadyPending) => {
            (StatusCode::ACCEPTED, RequestState::CancelPending)
        }
        Ok(CancelOutcome::AlreadyTerminal(state)) => (StatusCode::OK, state),
        Err(LedgerError::NotFound) => return request_not_found(),
        Err(e) => return ledger_error(&e),
    };
    json_response(
        status,
        &json!({ "request_id": id, "state": state.as_str() }),
    )
}

// ---------------------------------------------------------------------------------------------------------
// Sessions (S1, S3)

async fn create_session(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    if app.is_stopping() {
        return shutting_down();
    }
    let created = app.ledger().create_session(&scope, unix_ms());
    match created {
        Ok(view) => json_response(
            StatusCode::CREATED,
            &json!({ "session_id": view.session_id, "revision": view.revision }),
        ),
        Err(e) => ledger_error(&e),
    }
}

async fn get_session(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    let view = app.ledger().session(&scope, &id, unix_ms());
    match view {
        Ok(view) => json_response(
            StatusCode::OK,
            &json!({
                "session_id": view.session_id,
                "revision": view.revision,
                "message_count": view.current.message_count,
                "model": view.model,
                "parent": view.parent.map(|p| json!({
                    "session_id": p.session_id,
                    "revision": p.revision,
                })),
            }),
        ),
        Err(e) => ledger_error(&e),
    }
}

async fn fork_session(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let scope = match authenticate(&app, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    if app.is_stopping() {
        return shutting_down();
    }
    let revision = match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(fields)) => match (fields.len(), fields.get("revision")) {
            (1, Some(r)) => match r.as_u64() {
                Some(r) => r,
                None => {
                    return api_error(&ApiError::invalid_value(
                        "revision",
                        "revision must be a non-negative integer",
                    ))
                }
            },
            (_, None) => return api_error(&ApiError::missing_parameter("revision")),
            _ => {
                let extra = fields.keys().find(|k| *k != "revision").cloned();
                let param = extra.unwrap_or_default();
                return api_error(&ApiError::unsupported_parameter(
                    param.clone(),
                    format!("unsupported fork parameter '{param}'"),
                ));
            }
        },
        Ok(_) => return api_error(&ApiError::invalid_json("body must be a JSON object")),
        Err(e) => return api_error(&ApiError::invalid_json(e.to_string())),
    };
    let forked = app.ledger().fork_session(&scope, &id, revision, unix_ms());
    match forked {
        Ok(fork) => json_response(
            StatusCode::CREATED,
            &json!({
                "session_id": fork.child.session_id,
                "revision": fork.child.revision,
                "parent_session_id": fork.parent_session_id,
                "parent_revision": fork.parent_revision,
            }),
        ),
        Err(e) => ledger_error(&e),
    }
}

// ---------------------------------------------------------------------------------------------------------
// Chat completions (A3, A4, L2, L4, S2)

struct ChatHeaders {
    key: Option<String>,
    deadline_ms: Option<u64>,
    session: Option<(String, u64)>,
    priority: Priority,
    /// `X-Cere-On-Disconnect: continue` (m1close Q3).
    continue_on_disconnect: bool,
    /// Team pools (net-teleport design note 4.4, C17): `X-Cere-Class`; `None` -> the scope's default class.
    class: Option<crate::pool::Class>,
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>, ApiError> {
    headers
        .get(name)
        .map(|v| {
            v.to_str()
                .map_err(|_| ApiError::invalid_value(name, format!("{name} must be ASCII")))
        })
        .transpose()
}

fn parse_u64_header(headers: &HeaderMap, name: &str) -> Result<Option<u64>, ApiError> {
    header_str(headers, name)?
        .map(|v| {
            v.trim().parse::<u64>().map_err(|_| {
                ApiError::invalid_value(name, format!("{name} must be an unsigned integer"))
            })
        })
        .transpose()
}

fn parse_chat_headers(headers: &HeaderMap) -> Result<ChatHeaders, ApiError> {
    let key = header_str(headers, IDEMPOTENCY_KEY_HEADER)?
        .map(|k| {
            let ok = !k.is_empty()
                && k.len() <= MAX_IDEMPOTENCY_KEY_BYTES
                && k.bytes().all(|b| b.is_ascii_graphic());
            if ok {
                Ok(k.to_owned())
            } else {
                Err(ApiError::invalid_value(
                    "Idempotency-Key",
                    "Idempotency-Key must be 1-255 visible ASCII characters",
                ))
            }
        })
        .transpose()?;
    let deadline_ms = parse_u64_header(headers, DEADLINE_HEADER)?;
    if deadline_ms == Some(0) {
        return Err(ApiError::invalid_value(
            "X-Cere-Deadline-Ms",
            "X-Cere-Deadline-Ms must be at least 1",
        ));
    }
    let session_id = header_str(headers, SESSION_HEADER)?;
    let expected = parse_u64_header(headers, EXPECTED_REVISION_HEADER)?;
    let session = match (session_id, expected) {
        (Some(id), Some(revision)) => Some((id.to_owned(), revision)),
        (None, None) => None,
        (Some(_), None) => return Err(ApiError::missing_parameter("X-Cere-Expected-Revision")),
        (None, Some(_)) => return Err(ApiError::missing_parameter("X-Cere-Session-Id")),
    };
    let priority = match header_str(headers, PRIORITY_HEADER)? {
        None => Priority::Normal,
        Some(value) => Priority::parse(value.trim()).ok_or_else(|| {
            ApiError::invalid_value(
                "X-Cere-Priority",
                "X-Cere-Priority must be high, normal or low",
            )
        })?,
    };
    let continue_on_disconnect = match header_str(headers, ON_DISCONNECT_HEADER)? {
        None => false,
        Some(value) if value.trim().eq_ignore_ascii_case("continue") => true,
        Some(_) => {
            return Err(ApiError::invalid_value(
                "X-Cere-On-Disconnect",
                "X-Cere-On-Disconnect must be continue",
            ))
        }
    };
    let class = header_str(headers, crate::pool::CLASS_HEADER)?
        .map(|value| {
            crate::pool::Class::parse(value.trim()).ok_or_else(|| {
                ApiError::invalid_value("X-Cere-Class", "X-Cere-Class must be interactive or batch")
            })
        })
        .transpose()?;
    Ok(ChatHeaders {
        key,
        deadline_ms,
        session,
        priority,
        continue_on_disconnect,
        class,
    })
}

/// ab-split design note AB-03: the arm `X-Cere-Ab-Arm` names, by its index in `plan`; `None` without the header.
/// Naming an arm needs the operator token as `X-Cere-Operator-Authorization: Bearer <token>` (403 `forbidden`
/// otherwise); a deployment without arms or an unconfigured arm is 400 `invalid_value`.
fn named_ab_arm(
    app: &App,
    headers: &HeaderMap,
    plan: Option<&crate::ab::Plan>,
) -> Result<Option<usize>, ApiError> {
    let Some(name) = header_str(headers, AB_ARM_HEADER)? else {
        return Ok(None);
    };
    let operator = headers
        .get(OPERATOR_AUTHORIZATION_HEADER)
        .map(HeaderValue::as_bytes);
    if app.auth.authorize_operator(operator) != crate::auth::OperatorAccess::Operator {
        return Err(ApiError::new(
            403,
            INVALID_REQUEST,
            "forbidden",
            "X-Cere-Ab-Arm needs the operator token in X-Cere-Operator-Authorization",
            None,
        ));
    }
    let Some(plan) = plan else {
        return Err(ApiError::invalid_value(
            "X-Cere-Ab-Arm",
            "the model's deployment has no A/B arms",
        ));
    };
    plan.position(name.trim()).map(Some).ok_or_else(|| {
        ApiError::invalid_value("X-Cere-Ab-Arm", "X-Cere-Ab-Arm names no configured arm")
    })
}

/// A message as session prefix digests take it. O113: `reasoning_content` is left out, so a committed reply (which
/// keeps no reasoning) continues whether the client echoes the reply's reasoning or drops it.
pub(crate) fn message_value(message: &ChatMessage) -> Value {
    let mut value = serde_json::to_value(message).expect("chat messages serialize");
    if let Some(object) = value.as_object_mut() {
        object.remove("reasoning_content");
    }
    value
}

/// Session context of a continuation turn: what the commit needs to digest the new revision, and the
/// session's owner as the turn holds it from acceptance (m4 design note Q2, read under the lease lock; m3 Q8:
/// only this turn, the lease holder, can move it).
pub(crate) struct SessionTurn {
    pub(crate) session_id: String,
    pub(crate) expected_revision: u64,
    pub(crate) salt: Salt,
    pub(crate) owner: Option<String>,
}

async fn chat(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    match chat_inner(app, headers, body).await {
        Ok(response) | Err(response) => response,
    }
}

async fn chat_inner(app: Arc<App>, headers: HeaderMap, body: Body) -> Result<Response, Response> {
    let scope = authenticate(&app, &headers)?;
    let bytes = read_request(&app, body).await?;
    run_chat(app, scope, &headers, Surface::Chat, bytes).await
}

/// The gates before a request body is read: stopping, then Q10 (refused at once unless some worker accepts new
/// work; a request still waits for a slot on any), then the body bound.
async fn read_request(app: &App, body: Body) -> Result<Bytes, Response> {
    if app.is_stopping() {
        return Err(shutting_down());
    }
    if let Some(why) = app.not_serving() {
        return Err(api_error(&ApiError::new(
            503,
            SERVER_ERROR,
            "not_ready",
            format!("backend not ready: {why}"),
            None,
        )));
    }
    let limit = app.config.api.max_request_body_bytes as usize;
    axum::body::to_bytes(body, limit).await.map_err(|_| {
        api_error(&ApiError::new(
            413,
            INVALID_REQUEST,
            "request_too_large",
            format!("request body exceeds {limit} bytes"),
            None,
        ))
    })
}

/// Compat route installers (M2.14 D3/D4): each surface module adds its `routes` here; a surface's routes are
/// registered iff some deployment enables it ([`compat_enabled`]).
#[allow(clippy::type_complexity)]
pub(crate) const COMPAT_ROUTES: &[(CompatKind, fn(Router<Arc<App>>) -> Router<Arc<App>>)] = &[
    (CompatKind::Anthropic, crate::anthropic::routes),
    (CompatKind::Responses, crate::responses::routes),
];

/// Some deployment serves the `kind` surface.
pub fn compat_enabled(app: &App, kind: CompatKind) -> bool {
    app.config.deployments.iter().any(|d| kind.enabled(d))
}

/// A compat surface's front half (M2.14): authentication, the [`read_request`] gates, and the deployment the
/// request's `model` selects. A served model's deployment must enable the surface (else 404 `not_found`); an
/// unserved or missing model selects the first entry and is left to the chat validation (`model_not_found` naming
/// every served model). Every refusal is in the surface's envelope.
pub async fn compat_gate(
    api: &CompatApi,
    app: &App,
    headers: &HeaderMap,
    body: Body,
) -> Result<(String, Bytes, usize), Response> {
    let front = async {
        let scope = authenticate(app, headers)?;
        let bytes = read_request(app, body).await?;
        #[derive(serde::Deserialize)]
        struct Model {
            model: Option<String>,
        }
        // Track LoRA: an adapter name selects its base entry.
        let requested = serde_json::from_slice::<Model>(&bytes)
            .ok()
            .and_then(|m| m.model)
            .and_then(|m| {
                app.deployment_of_model(&m)
                    .or_else(|| app.lora_adapter(&m).map(|(d, _)| d))
            });
        let deployment = requested.unwrap_or(0);
        if requested.is_some() && !api.kind.enabled(&app.config.deployments[deployment]) {
            return Err(api_error(&ApiError::new(
                404,
                INVALID_REQUEST,
                "not_found",
                format!(
                    "model {} is not served on the {} API",
                    app.served()[deployment].model,
                    api.kind.name()
                ),
                None,
            )));
        }
        Ok((scope, bytes, deployment))
    };
    match front.await {
        Ok(front) => Ok(front),
        Err(response) => Err(surface_error(response, api.error_body).await),
    }
}

/// Authentication alone, refused in the surface's envelope (M2.14 D4 retrieve/cancel: no body, model or serving
/// gate applies to a route that never serves).
pub async fn compat_authenticate(
    api: &CompatApi,
    app: &App,
    headers: &HeaderMap,
) -> Result<(), Response> {
    match authenticate(app, headers) {
        Ok(_) => Ok(()),
        Err(response) => Err(surface_error(response, api.error_body).await),
    }
}

/// A compat surface's request (M2.14): [`compat_gate`], the surface's conversion to a chat body, then the chat
/// pipeline, every error rendered in the surface's envelope.
pub async fn serve_compat(
    api: &CompatApi,
    app: Arc<App>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let (scope, bytes, deployment) = match compat_gate(api, &app, &headers, body).await {
        Ok(front) => front,
        Err(response) => return response,
    };
    // Audit F5: the conversion renders the request's documents; with any, it runs off the async workers.
    let converted = {
        let (app, settings, bytes, convert) =
            (app.clone(), app.settings(), bytes.clone(), api.convert);
        let carries = document::bytes_carry_documents(&bytes);
        document::run(carries, move || {
            let ctx = ConvertCtx {
                served_model: &app.served()[deployment].model,
                policy: &settings.policies[deployment],
                deployment: &app.config.deployments[deployment],
            };
            convert(&bytes, &ctx)
        })
        .await
        .unwrap_or_else(|refused| Err(offload_owned_error(refused)))
    };
    let (chat, render) = match converted {
        Ok(converted) => converted,
        Err(e) => {
            let view = e.view();
            return bytes_response(status_of(view.status), (api.error_body)(&view));
        }
    };
    let surface = Surface::Compat(render.clone());
    match run_chat(app, scope, &headers, surface, Bytes::from(chat)).await {
        Ok(response) | Err(response) => {
            let render = move |e: &ErrorView| render.error_body(e);
            surface_error(response, render).await
        }
    }
}

/// The `not_ready` refusal of a request whose deployment has no Ready worker (M5 D5).
fn deployment_not_ready(model: &str) -> ApiError {
    ApiError::new(
        503,
        SERVER_ERROR,
        "not_ready",
        format!("backend not ready: no worker of {model} is ready"),
        None,
    )
}

/// A compat surface's token count (M2.14 D3 `count_tokens`). `chat` (the surface's conversion of the request)
/// is validated as a chat request of `deployment` (greedy: a count samples nothing), then counted where the chat
/// path counts `usage.prompt_tokens`: a native worker renders and encodes it through the sidecar, an
/// `sglang_http` worker answers `backend_path` for the client's `request` bytes. Nothing is admitted, journaled
/// or submitted.
pub(crate) async fn compat_prompt_tokens(
    app: &App,
    deployment: usize,
    chat: &[u8],
    backend_path: &str,
    request: &[u8],
) -> Result<u64, ApiError> {
    let mut value: Value =
        serde_json::from_slice(chat).map_err(|e| ApiError::invalid_json(e.to_string()))?;
    if let Some(object) = value.as_object_mut() {
        object.insert("temperature".to_owned(), json!(0.0));
    }
    let served = &app.served()[deployment];
    // Track LoRA: an adapter shares its base model's tokenizer; it is counted as the base model.
    let adapter = value
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|m| app.deployment_of_model(m).is_none() && app.lora_adapter(m).is_some());
    if adapter {
        value["model"] = Value::String(served.model.clone());
    }
    let validated = app.settings().policies[deployment].validate_bytes(&value, chat)?;
    if !app.deployment_ready(deployment) {
        return Err(deployment_not_ready(&served.model));
    }
    let ready = app.replicas().iter().filter(|r| {
        r.deployment() == deployment
            && matches!(
                app.worker_state(r.worker_id()),
                Some((crate::worker::WorkerState::Ready, _))
            )
    });
    for replica in ready {
        match &replica.backend {
            Backend::Native(slot) => {
                if let Some(native) = slot.get() {
                    return serve::prompt_tokens(&native, &validated).await;
                }
            }
            Backend::SglangHttp(client) => {
                return client
                    .count_tokens(backend_path, request)
                    .await
                    .map_err(|e| ApiError::from(&e));
            }
            Backend::Remote(_) => {}
        }
    }
    Err(ApiError::new(
        503,
        SERVER_ERROR,
        "not_ready",
        format!(
            "backend not ready: no local worker of {} is ready to count tokens",
            served.model
        ),
        None,
    ))
}

/// A JSON response of already rendered bytes.
pub(crate) fn bytes_response(status: StatusCode, body: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// Re-renders a chat error response (`{"error":{type, code, message, param}}`, status >= 400) in a compat
/// surface's envelope, keeping its status and headers; anything else passes unchanged.
async fn surface_error(response: Response, render: impl Fn(&ErrorView) -> Vec<u8>) -> Response {
    if response.status().as_u16() < 400 {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, usize::MAX).await else {
        return Response::from_parts(parts, Body::empty());
    };
    let value: Option<Value> = serde_json::from_slice(&bytes).ok();
    let error = value.as_ref().and_then(|v| v.get("error"));
    let field = |name: &str| error.and_then(|e| e.get(name)).and_then(Value::as_str);
    let (Some(kind), Some(code), Some(message)) = (field("type"), field("code"), field("message"))
    else {
        return Response::from_parts(parts, Body::from(bytes));
    };
    let view = ErrorView {
        status: parts.status.as_u16(),
        kind,
        code,
        message,
        param: field("param"),
    };
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, Body::from(render(&view)))
}

/// The chat pipeline from the request bytes on (chat, or a compat surface's converted body): validation on the
/// bytes, digest (with the surface), ledger acceptance and the executor.
pub(crate) async fn run_chat(
    app: Arc<App>,
    scope: String,
    headers: &HeaderMap,
    surface: Surface,
    bytes: Bytes,
) -> Result<Response, Response> {
    let mut value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| api_error(&ApiError::invalid_json(e.to_string())))?;
    // M5 D5: the request's `model` selects the deployment; an unserved one fails the first entry's validation
    // (404 `model_not_found` listing every served model) at the same point as before. Track LoRA: a loaded
    // adapter's name selects its base entry, validated as a request for the base model and leasing the adapter.
    let requested = value.get("model").and_then(Value::as_str);
    let (deployment, adapter) = match requested.and_then(|m| app.deployment_of_model(m)) {
        Some(d) => (d, None),
        None => match requested.and_then(|m| app.lora_adapter(m)) {
            Some((d, lease)) => (d, Some(lease)),
            None => (0, None),
        },
    };
    // A live reload swaps the settings; this request runs to its terminal on the ones it takes here.
    let settings = app.settings();
    let served = &app.served()[deployment];
    if adapter.is_some() {
        value["model"] = Value::String(served.model.clone());
    }
    // Audit F5: a body with documents is extracted off the async workers, behind a bound and a deadline.
    let mut request = {
        let (settings, bytes) = (settings.clone(), bytes.clone());
        let carries = document::carries_documents(&value);
        document::run(carries, move || {
            settings.policies[deployment].validate_bytes(&value, &bytes)
        })
        .await
        .map_err(offload_response)?
        .map_err(|e| api_error(&e))?
    };
    if let Some(lease) = adapter {
        // Responses name the adapter, as vLLM's do.
        request.model = lease.name.clone();
        request.adapter = Some(lease);
    }
    if !app.deployment_ready(deployment) {
        return Err(api_error(&deployment_not_ready(&served.model)));
    }
    let mut chat_headers = parse_chat_headers(headers).map_err(|e| api_error(&e))?;
    let ab_plan = settings.ab_plans[deployment].clone();
    let named_arm = named_ab_arm(&app, headers, ab_plan.as_deref()).map_err(|e| api_error(&e))?;
    // M2.14: a compat stream renders its own events and is never resumable.
    if !surface.is_chat() {
        chat_headers.continue_on_disconnect = false;
    }
    request
        .check_choices(
            chat_headers.session.is_some(),
            chat_headers.continue_on_disconnect,
        )
        .map_err(|e| api_error(&e))?;
    // ApiLenient D4: counted once per validated request, reported on its response (and on a replay of it).
    let ignored = ignored_params(&request, &surface);
    app.metrics
        .observe_params(surface.name(), &request.mapped_params, &ignored);

    let budget_ms = deadline_budget_ms(chat_headers.deadline_ms, &settings.config.api);
    let deadline = Instant::now() + Duration::from_millis(budget_ms);
    let now = unix_ms();
    let scope_key = app.ledger().digest_keys().scope_key(&scope);
    let mut normalized = request.normalized();
    // Track LoRA: an adapter request's digest covers the adapter's bytes, so a retry against an adapter reloaded
    // with other weights under the same name conflicts instead of replaying; a base request's is unchanged.
    if let Some(adapter) = &request.adapter {
        normalized["lora_adapter_sha256"] = json!(adapter.sha256);
    }
    let digest = scope_key.request_digest(&RequestDigestInput {
        request: &normalized,
        session_id: chat_headers.session.as_ref().map(|s| s.0.as_str()),
        expected_revision: chat_headers.session.as_ref().map(|s| s.1),
        served_model: &served.model,
        deadline_ms: budget_ms,
        config_hash: &app.config_hash,
        surface: surface.tag(),
    });

    // S2: the expected revision's message prefix, digested under the session's salt lineage.
    let mut turn = None;
    let mut claim_prefix = None;
    if let Some((session_id, expected_revision)) = &chat_headers.session {
        let view = app
            .ledger()
            .session(&scope, session_id, now)
            .map_err(|e| ledger_error(&e))?;
        // A shorter history cannot match the current revision; the ledger still checks the idempotency key
        // first (a replayed turn) and then rejects the claim as `revision_conflict`.
        let count = (view.current.message_count as usize).min(request.messages.len());
        let prefix: Vec<Value> = request.messages[..count]
            .iter()
            .map(message_value)
            .collect();
        claim_prefix = Some(scope_key.prefix_digest(&view.salt, &prefix));
        turn = Some(SessionTurn {
            session_id: session_id.clone(),
            expected_revision: *expected_revision,
            salt: view.salt,
            owner: None,
        });
    }

    let rid = app.rids.next_rid();
    // AB-03: the session's arm (a session turn by its scope and id, any other request by its rid), or the named one.
    let ab_arm = ab_plan.as_deref().map(|plan| {
        let index = named_arm.unwrap_or_else(|| {
            let key = match &turn {
                Some(t) => crate::ab::SessionKey::Session {
                    scope: &scope,
                    session_id: &t.session_id,
                },
                None => crate::ab::SessionKey::Request(rid.as_str()),
            };
            app.ab.assign(deployment, plan, key)
        });
        app.ab.arm(deployment, plan, index)
    });
    // A2: the request's client and workload, kept with its ledger record from acceptance; none without rules.
    let labels = (!app.metrics.clients.is_empty()).then(|| {
        app.metrics.clients.resolve(
            &scope,
            headers.get(CLIENT_HEADER).and_then(|v| v.to_str().ok()),
            headers.contains_key(PRIORITY_HEADER),
        )
    });
    let accepted = {
        let mut ledger = app.ledger();
        let accepted = ledger.accept(
            AcceptRequest {
                scope: scope.clone(),
                rid: rid.as_str().to_owned(),
                key: chat_headers.key,
                digest,
                stream: request.stream,
                surface: surface.tag().map(str::to_owned),
                session: turn
                    .as_ref()
                    .zip(claim_prefix)
                    .map(|(t, prefix)| SessionClaim {
                        session_id: t.session_id.clone(),
                        model: request.model.clone(),
                        expected_revision: t.expected_revision,
                        prefix_digest: prefix,
                    }),
            },
            now,
        );
        // M4-00: the admission target is the owner read on the guard that took the lease, never the view above.
        if let (Ok(AcceptOutcome::Accepted), Some(t)) = (&accepted, turn.as_mut()) {
            t.owner = ledger.held_owner(rid.as_str()).and_then(|h| h.worker);
        }
        if let (Ok(AcceptOutcome::Accepted), Some(labels)) = (&accepted, labels) {
            ledger.set_labels(rid.as_str(), labels);
        }
        // Team pools (C17): the class the usage row reports.
        if let Ok(AcceptOutcome::Accepted) = &accepted {
            let class = crate::pool::class_of(&settings.config, &scope, chat_headers.class);
            ledger.set_class(rid.as_str(), class);
        }
        accepted
    };
    match accepted {
        Ok(AcceptOutcome::Accepted) => crate::metrics::note_admitted(rid.as_str()),
        Ok(AcceptOutcome::Replay { rid, response }) => {
            let mut replay = Response::new(Body::from(response.body.clone()));
            *replay.status_mut() = status_of(response.status);
            replay.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            return Ok(with_ignored_params(with_request_id(replay, &rid), &ignored));
        }
        // Q4: a continue-mode original can be resumed instead of replayed, after its terminal and while it
        // still runs (its buffer opens with the `accepted` event; a queued original stays `request_in_progress`).
        Err(LedgerError::StreamNotReplayable { rid }) => {
            let resumable = app.resume.knows(&rid);
            return Err(stream_not_replayable(&rid, resumable));
        }
        Err(LedgerError::RequestInProgress { rid }) if app.resume.knows(&rid) => {
            return Err(stream_not_replayable(&rid, true));
        }
        Err(e) => return Err(ledger_error(&e)),
    }

    // WeightLab: a request of a bound scope runs under its plan (held until the request ends). Only the research
    // entry's requests (the native one with `[deployments.native.research]`).
    let research_entry = app.config.deployments[deployment]
        .native
        .as_ref()
        .is_some_and(|n| n.research.is_some());
    if let Some(research) = app.research.as_ref().filter(|_| research_entry) {
        request.research = research.apply(
            &scope,
            rid.as_str(),
            chat_headers
                .session
                .as_ref()
                .map(|(id, revision)| (id.as_str(), *revision)),
            request.messages.len() as u64,
        );
    }
    let research_plan = request.research.as_ref().map(|a| a.plan.plan.sha256_hex());
    let cancel_rx = app.register_active(rid.as_str());
    let (reply_tx, reply_rx) = oneshot::channel();
    let exec = Exec {
        attempt: AttemptIdentity {
            request_id: rid.as_str().to_owned(),
            attempt: 1,
            worker_id: None,
            worker_epoch: None,
            epoch: app.epoch(),
            instance_id: None,
        },
        settings,
        app,
        scope,
        rid: rid.clone(),
        request,
        deployment,
        priority: chat_headers.priority,
        continue_on_disconnect: chat_headers.continue_on_disconnect,
        class: chat_headers.class,
        surface,
        deadline,
        turn,
        cancel_rx,
        reason: None,
        ack_deadline: None,
        client: Client::Pending(reply_tx),
        native_cancel: None,
        ab_arm,
        remote: None,
        encoded: None,
    };
    tokio::spawn(exec.run());
    match reply_rx.await {
        Ok(mut response) => {
            if let Some(sha) = research_plan.and_then(|s| HeaderValue::from_str(&s).ok()) {
                response.headers_mut().insert(RESEARCH_PLAN_HEADER, sha);
            }
            Ok(with_ignored_params(response, &ignored))
        }
        Err(_) => Err(with_request_id(
            api_error(&internal_error("request executor ended without a response")),
            rid.as_str(),
        )),
    }
}

/// ApiLenient D4: a request's ignored parameters (the chat layer's and its compat conversion's), sorted.
fn ignored_params(request: &ChatRequest, surface: &Surface) -> Vec<String> {
    let mut ignored: Vec<String> = request
        .ignored_params
        .iter()
        .map(|p| (*p).to_owned())
        .collect();
    if let Some(render) = surface.renderer() {
        ignored.extend(render.ignored_params().iter().cloned());
    }
    ignored.sort_unstable();
    ignored.dedup();
    ignored
}

/// ApiLenient D4: `X-Cere-Ignored-Params: <sorted comma-separated paths>`, absent when nothing was ignored.
fn with_ignored_params(mut response: Response, ignored: &[String]) -> Response {
    if let Some(value) = (!ignored.is_empty())
        .then(|| HeaderValue::from_str(&ignored.join(",")).ok())
        .flatten()
    {
        response.headers_mut().insert(IGNORED_PARAMS_HEADER, value);
    }
    response
}

// ---------------------------------------------------------------------------------------------------------
// Executor

/// Where the executor's output goes: the waiting handler, an SSE stream, or nowhere (client gone).
pub(crate) enum Client {
    Pending(oneshot::Sender<Response>),
    Streaming(mpsc::Sender<Bytes>),
    Gone,
}

impl Client {
    /// Resolves when the client has gone away.
    pub(crate) async fn closed(&mut self) {
        match self {
            Client::Pending(tx) => tx.closed().await,
            Client::Streaming(tx) => tx.closed().await,
            Client::Gone => std::future::pending().await,
        }
    }

    /// Delivers the HTTP response (once); later calls and a gone client are no-ops.
    pub(crate) fn respond(&mut self, response: Response) {
        if let Client::Pending(tx) = std::mem::replace(self, Client::Gone) {
            let _ = tx.send(response);
        }
    }
}

/// Holds the admission slot and the worker's active-work count until the backend response has ended. A remote
/// worker's active work is its agent's (Q11): its slot holds only the admission charge, which outlives the slot
/// while the agent may still hold the work (H2, [`App::hold_remote_orphan`]).
pub(crate) struct Slot {
    app: Arc<App>,
    /// The granted worker's backend.
    backend: Backend,
    /// The remote call of a remote grant, built at the grant (Q11).
    call: Option<Arc<RemoteCall>>,
    worker_id: String,
    epoch: u64,
    /// Taken by [`Slot::into_permit`], which consumes the slot, and at the drop.
    permit: Option<Permit>,
}

impl Slot {
    /// The admission permit of a refused remote grant, to re-queue (H2); a remote slot has no `begin_work`.
    fn into_permit(mut self) -> Permit {
        self.permit
            .take()
            .expect("a slot holds its permit until it is consumed")
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let permit = self.permit.take();
        let Backend::Remote(worker) = &self.backend else {
            let _ = self.app.workers().finish_work(&self.worker_id, self.epoch);
            drop(permit);
            return;
        };
        // H2: a remote call that was accepted and did not reach its natural terminal may still hold the agent's
        // work: its slot stays charged to the worker until `cleanup_acknowledged` or the retention bound.
        if let (Some(call), Some(permit)) = (self.call.as_ref().filter(|c| c.held()), permit) {
            self.app.hold_remote_orphan(worker, call.clone(), permit);
        }
    }
}

/// The worker an admission grant goes to, once its own check passed.
enum Granted {
    /// Replica `index`, whose `begin_work` was recorded.
    Local(usize),
    /// A remote worker Ready at the granted epoch: the authority from its registry state at the grant and its
    /// inventory's instance id.
    Remote(RemoteWorker, AuthorityBase, Option<String>),
}

pub(crate) struct Exec {
    pub(crate) app: Arc<App>,
    /// The live settings this request was accepted under ([`crate::reload`]); it keeps them to its terminal.
    pub(crate) settings: Arc<Settings>,
    pub(crate) scope: String,
    pub(crate) rid: Rid,
    pub(crate) request: ChatRequest,
    /// The `[[deployments]]` entry the request's `model` selected (M5 D5): its workers serve it, its identity
    /// fills the `cere` extension.
    pub(crate) deployment: usize,
    /// Admission priority within the scope (`X-Cere-Priority`, m1close Q2).
    pub(crate) priority: Priority,
    /// A continue-mode stream (m1close Q3): the client detaches instead of cancelling once the stream opened.
    pub(crate) continue_on_disconnect: bool,
    /// Team pools (C17): the request's `X-Cere-Class`; `None` -> its scope's default.
    pub(crate) class: Option<crate::pool::Class>,
    /// M2.14: the API surface whose bytes the client gets (chat, or a compat surface's rendering).
    pub(crate) surface: Surface,
    pub(crate) deadline: Instant,
    pub(crate) attempt: AttemptIdentity,
    pub(crate) turn: Option<SessionTurn>,
    pub(crate) cancel_rx: watch::Receiver<Option<&'static str>>,
    /// Cancel reason once cancellation started.
    pub(crate) reason: Option<&'static str>,
    /// After the backend abort was sent: when an unacknowledged cancel fails `cancel_unacknowledged`.
    pub(crate) ack_deadline: Option<Instant>,
    pub(crate) client: Client,
    /// The native generation's cancel handle, once it started.
    pub(crate) native_cancel: Option<Arc<NativeCancel>>,
    /// ab-split design note AB-03: the request's A/B arm when its deployment has `[deployments.native.ab]`.
    pub(crate) ab_arm: Option<crate::ab::Arm>,
    /// The call of the current remote grant (Q11), which a cancel cancels and whose terminal may carry the
    /// request's reason (H2); `None` before a remote grant and while a refused Submit re-queues.
    pub(crate) remote: Option<Arc<RemoteCall>>,
    /// m32n design note D12: a multi-replica native request's prompt, encoded before the grant on `backend`
    /// (any live replica), with its prefix-affinity key; taken by the granted replica's generation.
    encoded: Option<Encoded>,
}

/// D12: a prompt encoded before the grant, the replica whose sidecar (and grammar process) encoded it, and its
/// prefix-affinity key (D13).
struct Encoded {
    prompt: serve::Prompt,
    backend: Arc<NativeBackend>,
    key: Option<crate::native::affinity::Key>,
}

/// A backend call the executor drives; `'static` so an unacknowledged cancel can hand it to a drain task.
pub(crate) type BackendCall<T> = Pin<Box<dyn Future<Output = Result<T, BackendError>> + Send>>;

impl Exec {
    async fn run(mut self) {
        let mut next = self.admit().await;
        // H2: the remote workers that refused this request's Submit; a second refusal from one fails it.
        let mut tried = Vec::new();
        while let Some(slot) = next.take() {
            match slot.backend.clone() {
                Backend::SglangHttp(client) if self.request.stream => {
                    crate::stream::run(&mut self, slot, client).await
                }
                Backend::SglangHttp(client) => {
                    let body = self.backend_body();
                    let deadline = self.backend_deadline();
                    let call: BackendCall<Value> =
                        Box::pin(async move { client.chat(&body, deadline).await });
                    self.run_unary(Some(slot), call).await
                }
                Backend::Native(native) => match native.get() {
                    Some(native) => self.run_native(slot, native).await,
                    // Readiness has not published a loaded executor: fail closed.
                    None => self.fail_not_ready(slot),
                },
                Backend::Remote(worker) => match slot.call.clone() {
                    Some(call) => next = self.run_remote(slot, worker, call, &mut tried).await,
                    None => self.fail_not_ready(slot),
                },
            }
        }
        self.app.unregister_active(self.rid.as_str());
    }

    /// Fails a granted request `worker_not_ready` before any backend call.
    fn fail_not_ready(&mut self, slot: Slot) {
        let reason = REASON_WORKER_NOT_READY;
        self.settle_failure("failed", reason, None);
        let response = self.failure_response(&reason_error(reason), "failed", reason, None);
        self.client.respond(response);
        drop(slot);
    }

    /// Q11/H2: a chat on a remote worker. SubmitWork carries the remaining budget; the work stream (and a unary
    /// call) may run until the deadline plus twice the cancel bound, like a local backend call
    /// ([`Self::backend_deadline`]). A refused Submit re-queues the request: the result is then the slot of its
    /// next grant.
    async fn run_remote(
        &mut self,
        slot: Slot,
        worker: RemoteWorker,
        call: Arc<RemoteCall>,
        tried: &mut Vec<String>,
    ) -> Option<Slot> {
        let budget = self.deadline;
        let until = self.deadline + 2 * self.app.tuning.cancel_ack_timeout;
        let submitting = call.clone();
        let mut submit: BackendCall<Option<Refusal>> =
            Box::pin(async move { submitting.submit(budget).await });
        let Some(submitted) = self.drive(&mut submit).await else {
            self.give_up(Some(slot), submit);
            return None;
        };
        let unreceipted = matches!(submitted, Ok(Some(_)) | Err(BackendError::Timeout(_)));
        if unreceipted && self.reason.is_none() && Instant::now() >= budget {
            // H2: the budget ran out before a receipt.
            self.begin_cancel(Some(REASON_DEADLINE_EXCEEDED));
        }
        match submitted {
            Ok(None) if self.request.stream => {
                crate::stream::run_remote(self, slot, call, until).await;
                None
            }
            Ok(None) => {
                let call: BackendCall<Value> = Box::pin(async move { call.chat(until).await });
                self.run_unary(Some(slot), call).await;
                None
            }
            Ok(Some(refusal)) if self.reason.is_none() => {
                self.requeue_remote(slot, &worker, refusal, tried).await
            }
            Ok(Some(refusal)) => {
                drop(slot);
                self.backend_failed(&BackendError::Connect(format!(
                    "SubmitWork refused ({refusal:?})"
                )));
                None
            }
            Err(e) => {
                drop(slot);
                self.backend_failed(&e);
                None
            }
        }
    }

    /// H2: `worker` refused the request's Submit (class `refusal`). A joining-class refusal takes the worker back
    /// to Joining. A worker in `tried` refused it before: the request fails `worker_not_ready`. Otherwise the
    /// running request waits in admission again (the worker is marked suspect there) until its deadline, and the
    /// result is the slot of its next grant.
    async fn requeue_remote(
        &mut self,
        slot: Slot,
        worker: &RemoteWorker,
        refusal: Refusal,
        tried: &mut Vec<String>,
    ) -> Option<Slot> {
        let app = self.app.clone();
        self.remote = None;
        // Team pools (C19): the worker serves only its owner now; admission stops granting it to other scopes.
        if refusal == Refusal::OwnerOnly {
            app.admission.set_owner_only(worker.worker_id(), true);
        }
        if refusal == Refusal::Joining {
            if let Some(registry) = app.remote_workers() {
                worker.rejoin(registry, &app.admission);
            }
        }
        let id = worker.worker_id();
        if tried.iter().any(|t| t == id) {
            self.fail_not_ready(slot);
            return None;
        }
        tried.push(id.to_owned());
        let permit = slot.into_permit();
        let again = tokio::select! {
            biased;
            _ = self.cancel_rx.changed() => None,
            _ = self.client.closed() => {
                let _ = app.cancel(&self.scope, self.rid.as_str(), REASON_CLIENT_DISCONNECTED);
                None
            }
            permit = app.admission.requeue(permit, self.deadline) => Some(permit),
        };
        let permit = self.granted(again, true)?;
        self.dispatch(permit, true).await
    }

    /// The replica of the granted worker; `None` before the grant.
    pub(crate) fn replica(&self) -> Option<&Replica> {
        self.app.replica(self.attempt.worker_id.as_deref()?)
    }

    pub(crate) fn rid_str(&self) -> &str {
        self.rid.as_str()
    }

    /// Admission (L4) and `queued → running`. `None` when the request ended before dispatch (response sent).
    async fn admit(&mut self) -> Option<Slot> {
        let app = self.app.clone();
        if app.ledger().mark_queued(self.rid_str(), unix_ms()).is_err() {
            self.respond_settled();
            return None;
        }
        // Q8: a session turn goes to the session's owner (as held from acceptance) while it is eligible, else to
        // any; a grant elsewhere re-pins the session in its `dispatched` record (m4 design note Q3a).
        let target = match self.turn.as_ref().and_then(|t| t.owner.as_deref()) {
            Some(worker) => Target::Pinned(Arc::from(worker), self.deployment),
            None => Target::Any(self.deployment),
        };
        let route = Route {
            class: self.class,
            ..self.encode_before_grant().await?
        };
        let acquired = tokio::select! {
            biased;
            _ = self.cancel_rx.changed() => None,
            _ = self.client.closed() => {
                let _ = app.cancel(&self.scope, self.rid.as_str(), REASON_CLIENT_DISCONNECTED);
                None
            }
            permit = app.admission.acquire_routed(
                &self.scope,
                self.priority,
                self.deadline,
                target,
                u64::from(self.request.n),
                route,
            ) => Some(permit),
        };
        let permit = self.granted(acquired, false)?;
        self.dispatch(permit, false).await
    }

    /// m32n design note D11/D12: a multi-replica native request is encoded before its grant, on any live
    /// replica's sidecar, so the grant knows its pool reservation (`n` x [`serve::Prompt::reservation`]) and the
    /// card its prefix-affinity key prefers; the prompt is kept for the granted card. A refused encode is the
    /// request's answer (`None`, response sent) and never reaches admission. Every other request (and one while no
    /// replica is live: its grant then fails closed) routes by rows only.
    async fn encode_before_grant(&mut self) -> Option<Route> {
        let app = self.app.clone();
        if !app.routes_by_pool(self.deployment) {
            return Some(Route::default());
        }
        let Some(backend) = app.live_native(self.deployment) else {
            return Some(Route::default());
        };
        let salt = app.cache_salts.get(&self.scope).map(String::as_str);
        // Audit F6: the encode answers to the same cancel, disconnect and deadline as the queue it precedes, and
        // at most `MAX_PRE_GRANT_ENCODES` run at once (the rest wait here, still cancellable), so a request
        // admission will refuse, or a client that left, does not keep the tokenizer busy. An encode whose request
        // ended has its sidecar call finish on the blocking pool, bounded by the sidecar's I/O deadline.
        let (request, ab_arm) = (&self.request, self.ab_arm.as_ref());
        let encode_start = Instant::now();
        let encoded = tokio::select! {
            biased;
            _ = self.cancel_rx.changed() => Err(None),
            _ = self.client.closed() => {
                let _ = app.cancel(&self.scope, self.rid.as_str(), REASON_CLIENT_DISCONNECTED);
                Err(None)
            }
            _ = tokio::time::sleep_until(self.deadline) => Err(Some(AdmissionError::DeadlineExceeded)),
            encoded = async {
                let _slot = PRE_GRANT_ENCODES
                    .acquire()
                    .await
                    .expect("the encode gate is never closed");
                serve::encode(&backend, request, salt, ab_arm).await
            } => Ok(encoded),
        };
        // OQ4 (beat-sglang design note): the encode stage is `encode_ms`, outside `queue_wait_ms`. A request that
        // ended meanwhile is no longer queued and keeps its wait as measured.
        let encode_ms = u64::try_from(encode_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let _ = app.ledger().record_encode(self.rid_str(), encode_ms);
        let prompt = match encoded {
            Ok(Ok(prompt)) => prompt,
            Ok(Err(error)) => {
                self.refuse_encode(&error);
                return None;
            }
            Err(ended) => {
                // Settled as a request that ended in admission's queue does: cancelled, or past its deadline.
                let _ = self.granted(ended.map(Err), false);
                return None;
            }
        };
        let key = prompt.affinity_key();
        let route = Route {
            tokens: u64::from(self.request.n) * prompt.reservation(self.request.max_output_tokens),
            prefer: key.and_then(|k| app.affinity_worker(self.deployment, &k)),
            class: None,
        };
        self.encoded = Some(Encoded {
            prompt,
            backend,
            key,
        });
        Some(route)
    }

    /// A refused native encode (context, tools, grammar, sampling): the request fails with `error` and never
    /// reaches the executor.
    fn refuse_encode(&mut self, error: &ApiError) {
        // AB-04: a refused encode is a request of the arm that failed.
        if let Some(arm) = &self.ab_arm {
            arm.start().finish(crate::ab::End::Error);
        }
        self.refuse(error);
    }

    /// The request fails with `error` before any executor call.
    fn refuse(&mut self, error: &ApiError) {
        let reason = error.code;
        self.settle_failure("failed", reason, None);
        let response = self.failure_response(error, "failed", reason, None);
        self.client.respond(response);
    }

    /// The dispatch of an admission grant: the granted worker's own check, the session pin, the slot, `queued →
    /// running` (unless the request is `running` already, re-queued after a refused Submit), and the `dispatched`
    /// record. `None` when the request ended here (response sent).
    async fn dispatch(&mut self, mut permit: Permit, running: bool) -> Option<Slot> {
        let app = self.app.clone();
        // The granted worker (Q3); observed again, since the request may have queued for longer than the
        // worker's suspect bound or across an epoch bump (Q7). A rejection re-queues the request once.
        let mut requeued = false;
        let granted = loop {
            let worker_id = permit.worker();
            if app.remote(worker_id).is_some() {
                // Q11: the work's authority is the registry state at the grant; a worker no longer Ready at the
                // granted epoch is rejected like a local `begin_work`.
                if let Some((worker, at, instance_id)) = app.remote_grant(worker_id, permit.epoch())
                {
                    break Granted::Remote(worker, at, instance_id);
                }
            } else {
                let replica = app.replica_index(worker_id);
                let began = app
                    .workers()
                    .begin_work(worker_id, permit.epoch(), unix_ms());
                match (replica, began) {
                    (Some(index), Ok(())) => break Granted::Local(index),
                    (None, Ok(())) => {
                        let _ = app.workers().finish_work(worker_id, permit.epoch());
                    }
                    (_, Err(_)) => {}
                }
            }
            if requeued {
                drop(permit);
                if app
                    .ledger()
                    .fail(self.rid_str(), REASON_WORKER_NOT_READY, None, unix_ms())
                    .is_ok()
                {
                    app.log_terminal(&self.scope, self.rid_str());
                }
                let error = reason_error(REASON_WORKER_NOT_READY);
                self.client
                    .respond(with_request_id(api_error(&error), self.rid_str()));
                return None;
            }
            requeued = true;
            let again = tokio::select! {
                biased;
                _ = self.cancel_rx.changed() => None,
                _ = self.client.closed() => {
                    let _ = app.cancel(&self.scope, self.rid.as_str(), REASON_CLIENT_DISCONNECTED);
                    None
                }
                permit = app.admission.requeue(permit, self.deadline) => Some(permit),
            };
            permit = self.granted(again, running)?;
        };
        // Q-T6/Q-T8 (net-teleport design note 4.4): a non-owner's work on an owned worker ends within
        // `admission.owner_wait_cap_ms` of its grant, so its owner (or an owner-only window) waits at most that long:
        // its deadline keeps `OWNER_CAP_RELEASE` of the cap for its abort and the slot's release.
        if let Some(cap) = permit.team().and_then(|t| t.work_cap) {
            let budget = cap.saturating_sub(crate::pool::OWNER_CAP_RELEASE);
            self.deadline = self.deadline.min(Instant::now() + budget);
        }
        let worker_id = permit.worker().to_owned();
        let worker_epoch = permit.epoch();
        // Q9/Q15: the worker that may hold the backend work, journaled immediately before the backend call.
        let (backend, call, instance_id, target) = match granted {
            Granted::Local(index) => {
                let replica = app.replica_at(index);
                let target = WorkerTarget::Local {
                    base_url: replica.base_url().to_owned(),
                };
                (replica.backend.clone(), None, replica.instance_id(), target)
            }
            Granted::Remote(worker, at, instance_id) => {
                let call = Arc::new(worker.call(
                    at,
                    &self.scope,
                    self.rid_str(),
                    self.request.stream,
                    &self.backend_body(),
                ));
                self.remote = Some(call.clone());
                let target = WorkerTarget::Remote {
                    endpoint: worker.endpoint().to_owned(),
                };
                let backend = Backend::Remote(worker);
                (backend, Some(call), instance_id, target)
            }
        };
        self.attempt.worker_id = Some(worker_id.clone());
        self.attempt.worker_epoch = Some(worker_epoch);
        self.attempt.instance_id = instance_id;
        let slot = Slot {
            app: app.clone(),
            backend,
            call,
            worker_id,
            epoch: worker_epoch,
            permit: Some(permit),
        };
        // OQ4: the queue wait ends at the grant, stamped before the ledger lock, so another request's journal fsync
        // held under that lock is not counted as this one's queue wait.
        let granted_ms = unix_ms();
        if !running
            && app
                .ledger()
                .mark_running(self.rid_str(), granted_ms)
                .is_err()
        {
            // Cancelled between the grant and now.
            self.respond_settled();
            return None;
        }
        // A cancel that raced the transition: `cancel_pending` before any backend call → settle without one.
        let raced = *self.cancel_rx.borrow_and_update();
        if let Some(reason) = raced {
            self.reason = Some(reason);
            self.settle_cancelled(reason);
            return None;
        }
        // m4 design note Q3a: a grant that moves the session's owner rides on this record; the held generation
        // read on the same guard says whether it did.
        let (dispatched, owner_moved) = {
            let mut ledger = app.ledger();
            let before = ledger.held_owner(self.rid_str());
            let dispatched = ledger.dispatched(
                self.rid_str(),
                &slot.worker_id,
                slot.epoch,
                &target,
                unix_ms(),
            );
            let after = ledger.held_owner(self.rid_str());
            let moved = before
                .zip(after)
                .filter(|(b, a)| a.owner_generation > b.owner_generation)
                .map(|(b, a)| (b.owner_generation, a.owner_generation));
            (dispatched, moved)
        };
        if let (Some((before, generation)), Some(turn)) = (owner_moved, &self.turn) {
            let cause = if before == 0 { "pin" } else { "repin" };
            eprintln!(
                "cered event=session_owner session={} worker={} generation={generation} cause={cause}",
                turn.session_id, slot.worker_id
            );
        }
        match dispatched {
            Ok(()) => Some(slot),
            Err(e) => {
                drop(slot);
                match e {
                    // Q10: the worker's epoch moved past the grant; the ledger failed the request.
                    LedgerError::WorkerEpochSuperseded => {
                        app.log_terminal(&self.scope, self.rid_str())
                    }
                    // Already terminal: a cancel or the supersede of this worker epoch won.
                    LedgerError::InvalidTransition { .. } => {}
                    // A journal failure (the ledger is poisoned): fail before any backend sees the request.
                    _ => {
                        if app
                            .ledger()
                            .fail(self.rid_str(), REASON_JOURNAL_ERROR, None, unix_ms())
                            .is_ok()
                        {
                            app.log_terminal(&self.scope, self.rid_str());
                        }
                    }
                }
                self.respond_settled();
                None
            }
        }
    }

    /// The permit of an admission wait, or `None` once the request ended there: cancelled while queued
    /// (settled by the cancel), or refused (queue full, deadline, shutdown), each answered here. A `running`
    /// request (re-queued after a refused Submit) ends as [`Self::requeue_ended`] says.
    fn granted(
        &mut self,
        acquired: Option<Result<Permit, AdmissionError>>,
        running: bool,
    ) -> Option<Permit> {
        let e = match acquired {
            Some(Ok(permit)) => return Some(permit),
            None if running => {
                self.requeue_ended(None);
                return None;
            }
            None => {
                // Cancelled while queued: no backend call was made.
                self.respond_settled();
                return None;
            }
            Some(Err(e)) if running => {
                self.requeue_ended(Some(e));
                return None;
            }
            Some(Err(e)) => e,
        };
        let app = self.app.clone();
        let (result, reason) = match e {
            AdmissionError::QueueFull => (
                app.ledger()
                    .fail(self.rid_str(), REASON_QUEUE_FULL, None, unix_ms()),
                REASON_QUEUE_FULL,
            ),
            AdmissionError::DeadlineExceeded => (
                app.ledger()
                    .cancel(self.rid_str(), REASON_DEADLINE_EXCEEDED, unix_ms()),
                REASON_DEADLINE_EXCEEDED,
            ),
            AdmissionError::Closed => (
                app.ledger()
                    .cancel(self.rid_str(), REASON_SHUTDOWN, unix_ms()),
                REASON_SHUTDOWN,
            ),
            AdmissionError::NoAllowedWorker => (
                app.ledger()
                    .fail(self.rid_str(), REASON_NO_ALLOWED_WORKER, None, unix_ms()),
                REASON_NO_ALLOWED_WORKER,
            ),
        };
        if result.is_ok() {
            app.log_terminal(&self.scope, self.rid_str());
        }
        let error = reason_error(reason);
        self.client
            .respond(with_request_id(api_error(&error), self.rid_str()));
        None
    }

    /// A running request's re-queue ended without a grant, cancelled or refused by admission (`e`). No backend
    /// call is pending: a refusal moves it to `cancel_pending` with its reason, then it settles `cancelled` with
    /// the reason that won.
    fn requeue_ended(&mut self, e: Option<AdmissionError>) {
        let cause = e.map(|e| match e {
            AdmissionError::QueueFull => REASON_QUEUE_FULL,
            AdmissionError::DeadlineExceeded => REASON_DEADLINE_EXCEEDED,
            AdmissionError::Closed => REASON_SHUTDOWN,
            AdmissionError::NoAllowedWorker => REASON_NO_ALLOWED_WORKER,
        });
        if let Some(cause) = cause {
            let _ = self.app.cancel(&self.scope, self.rid.as_str(), cause);
        }
        let signalled = *self.cancel_rx.borrow_and_update();
        let reason = signalled.or(cause).unwrap_or(REASON_CANCEL_REQUESTED);
        self.reason = Some(reason);
        self.settle_cancelled(reason);
    }

    /// Answers a request that already reached its terminal before dispatch.
    fn respond_settled(&mut self) {
        let status = self.app.ledger().status(&self.scope, self.rid_str());
        let reason = status
            .as_ref()
            .and_then(|s| s.reason.clone())
            .unwrap_or_else(|| REASON_CANCEL_REQUESTED.to_owned());
        let error = reason_error(&reason);
        self.client
            .respond(with_request_id(api_error(&error), self.rid_str()));
    }

    /// `cancel_pending → cancelled` with no backend response pending, then answers the client.
    fn settle_cancelled(&mut self, reason: &'static str) {
        if self
            .app
            .ledger()
            .cancel(self.rid_str(), reason, unix_ms())
            .is_ok()
        {
            self.app.log_terminal(&self.scope, self.rid_str());
        }
        let response = self.failure_response(&reason_error(reason), "cancelled", reason, None);
        self.client.respond(response);
    }

    /// Drives `call` while watching for cancellation (signal, deadline, client gone). After a cancel starts,
    /// the backend abort is sent and `call` gets `cancel_ack_timeout` to end; `None` means it did not, or the
    /// request was [`superseded`](Exec::superseded) (it then stops at once).
    pub(crate) async fn drive<T>(
        &mut self,
        call: &mut BackendCall<T>,
    ) -> Option<Result<T, BackendError>> {
        loop {
            if self.superseded() {
                self.begin_cancel(None);
                return None;
            }
            let cancelling = self.ack_deadline.is_some();
            let ack_at = self.ack_deadline.unwrap_or(self.deadline);
            tokio::select! {
                biased;
                out = call.as_mut() => return Some(out),
                _ = self.cancel_rx.changed(), if !cancelling => self.begin_cancel(None),
                _ = tokio::time::sleep_until(self.deadline), if !cancelling => {
                    self.begin_cancel(Some(REASON_DEADLINE_EXCEEDED));
                }
                _ = self.client.closed(), if !cancelling => {
                    self.begin_cancel(Some(REASON_CLIENT_DISCONNECTED));
                }
                _ = tokio::time::sleep_until(ack_at), if cancelling => return None,
            }
        }
    }

    /// Starts cancellation: moves the ledger to `cancel_pending` (for a server-side `cause`), records the
    /// reason, arms the acknowledgement deadline and sends `/abort_request` for this rid (a remote call's
    /// `CancelWork`, H2). The abort call runs on its own task, so the executor keeps watching the backend and the
    /// ack window counts from the cancel even when the control plane is slow.
    pub(crate) fn begin_cancel(&mut self, cause: Option<&'static str>) {
        if self.ack_deadline.is_some() {
            return;
        }
        if let Some(cause) = cause {
            let _ = self.app.cancel(&self.scope, self.rid.as_str(), cause);
        }
        // The signal carries the reason that won (a concurrent endpoint cancel may have come first).
        let signalled = *self.cancel_rx.borrow_and_update();
        self.reason = signalled.or(cause).or(Some(REASON_CANCEL_REQUESTED));
        self.ack_deadline = Some(Instant::now() + self.app.tuning.cancel_ack_timeout);
        if let Some(cancel) = self.native_cancel.clone() {
            // `cere_executor_cancel`; the generation's terminal poll ends the chunk stream.
            tokio::task::spawn_blocking(move || cancel.cancel());
            return;
        }
        if signalled == Some(REASON_WORKER_EPOCH_SUPERSEDED) {
            // Q10: the supersede sent this rid's `/abort_request` already.
            return;
        }
        if let Some(call) = &self.remote {
            // H2: `CancelWork` with the receipt's ticket (once the receipt arrived).
            call.cancel();
            return;
        }
        let Some(client) = self.replica().and_then(|r| r.backend.sglang().cloned()) else {
            return;
        };
        let rid = self.rid.clone();
        let worker = self
            .attempt
            .worker_id
            .clone()
            .zip(self.attempt.worker_epoch);
        tokio::spawn(async move {
            if let Err(e) = client.abort(&rid).await {
                eprintln!(
                    "cered abort rid={rid} worker={} failed: {e}",
                    worker_field(worker.as_ref())
                );
            }
        });
    }

    /// The backend chat body, with the scope's `cache_salt` under `prefix_cache_isolation = "cache_salt"` (Q2).
    pub(crate) fn backend_body(&self) -> BackendBody {
        let mut body = self.request.backend_body(&self.rid);
        if let Some(salt) = self.app.cache_salts.get(&self.scope) {
            body.fields.insert("cache_salt".to_owned(), json!(salt));
        }
        body
    }

    /// The deadline handed to a backend call: the request's remaining budget plus twice the cancel bound. The
    /// ack window of a deadline cancel ends one bound after the deadline, so CERE's own timeout never ends the
    /// call inside it (a timed-out call would otherwise be mistaken for the backend acknowledging the abort).
    pub(crate) fn backend_deadline(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
            + 2 * self.app.tuning.cancel_ack_timeout
    }

    /// Q10: the supersede of this request's worker epoch failed it (under the ledger lock, signalling the
    /// executor there), aborted its backend request and took over its admission slot. The executor then writes
    /// no terminal and sends no abort of its own, and the client gets `worker_epoch_superseded`. The signal is
    /// sticky: once the request is terminal no cancel signals it again.
    pub(crate) fn superseded(&self) -> bool {
        *self.cancel_rx.borrow() == Some(REASON_WORKER_EPOCH_SUPERSEDED)
    }

    /// Writes a failed/cancelled terminal, with the backend's `usage` if it sent one, and logs it; nothing for
    /// a [`superseded`](Exec::superseded) request, whose terminal the supersede wrote.
    pub(crate) fn settle_failure(&self, state: &str, reason: &str, usage: Option<&Value>) {
        if self.superseded() {
            return;
        }
        if usage.is_some() {
            self.record_usage(&self.usage(usage, state));
        }
        let now = unix_ms();
        let result = if state == "cancelled" {
            self.app.ledger().cancel(self.rid_str(), reason, now)
        } else {
            self.app.ledger().fail(self.rid_str(), reason, None, now)
        };
        match result {
            Ok(()) => self.app.log_terminal(&self.scope, self.rid_str()),
            Err(e) => eprintln!(
                "cered request rid={} terminal {state}/{reason} not recorded: {}",
                self.rid,
                e.code()
            ),
        }
    }

    /// Q1 for a request ending in `state`: the pin's zero inference does not apply to a cancelled request.
    pub(crate) fn usage(&self, usage: Option<&Value>, state: &str) -> RequestUsage {
        RequestUsage::from_backend(
            usage,
            self.app.deployment(self.deployment).cache_report_pinned && state != "cancelled",
        )
    }

    /// Hands the request's usage to the ledger, which journals it with the terminal (Q5).
    pub(crate) fn record_usage(&self, usage: &RequestUsage) {
        if let Err(e) = self.app.ledger().record_usage(self.rid_str(), *usage) {
            eprintln!(
                "cered request rid={} usage not recorded: {}",
                self.rid,
                e.code()
            );
        }
    }

    /// The `cere` object of a response or event.
    pub(crate) fn cere(&self, usage: &RequestUsage) -> Map<String, Value> {
        let extension =
            CereExtension::new(self.app.deployment(self.deployment), &self.attempt, usage);
        match serde_json::to_value(extension) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        }
    }

    /// The session revision committed by a completed turn: the request messages plus the assistant reply.
    pub(crate) fn session_commit(&self, reply: ChatMessage) -> Option<(Revision, u64, String)> {
        let turn = self.turn.as_ref()?;
        let mut messages: Vec<Value> = self.request.messages.iter().map(message_value).collect();
        messages.push(message_value(&reply));
        let digest = self
            .app
            .ledger()
            .digest_keys()
            .scope_key(&self.scope)
            .prefix_digest(&turn.salt, &messages);
        Some((
            Revision {
                message_count: messages.len() as u64,
                digest,
            },
            turn.expected_revision + 1,
            turn.session_id.clone(),
        ))
    }

    /// A failure after dispatch: the error body plus `cere` with state and reason; `worker_epoch_superseded`
    /// for a [`superseded`](Exec::superseded) request, whatever the executor was ending it with.
    pub(crate) fn failure_response(
        &self,
        error: &ApiError,
        state: &str,
        reason: &str,
        usage: Option<&Value>,
    ) -> Response {
        let superseded;
        let (error, state, reason, usage) = if self.superseded() {
            superseded = reason_error(REASON_WORKER_EPOCH_SUPERSEDED);
            (&superseded, "failed", REASON_WORKER_EPOCH_SUPERSEDED, None)
        } else {
            (error, state, reason, usage)
        };
        let mut body = error.body();
        let mut cere = self.cere(&self.usage(usage, state));
        cere.insert("state".to_owned(), json!(state));
        cere.insert("reason".to_owned(), json!(reason));
        body["cere"] = Value::Object(cere);
        with_request_id(
            json_response(status_of(error.status), &body),
            self.rid_str(),
        )
    }

    /// The native path: encode and context check (a refusal never reaches the executor), then one generation of
    /// `n` choices whose chunks go through the same relay (SSE) or unary handling as an SGLang response. A
    /// multi-replica request encoded before its grant (m32n design note D12) reuses that prompt, its grammars
    /// opened again when another card was granted, and records its affinity key on the granted card (D13: the
    /// card whose PREFILL captures the prefix).
    async fn run_native(&mut self, slot: Slot, backend: Arc<NativeBackend>) {
        let prompt = match self.encoded.take() {
            Some(encoded) => {
                let mut prompt = encoded.prompt;
                if !Arc::ptr_eq(&encoded.backend, &backend) {
                    if let Err(error) = prompt.reopen_grammars(&backend, &self.request).await {
                        drop(slot);
                        self.refuse(&error);
                        return;
                    }
                }
                if let Some(key) = encoded.key {
                    backend.affinity.insert(key);
                }
                prompt
            }
            None => {
                let salt = self.app.cache_salts.get(&self.scope).map(String::as_str);
                match serve::encode(&backend, &self.request, salt, self.ab_arm.as_ref()).await {
                    Ok(prompt) => prompt,
                    Err(error) => {
                        drop(slot);
                        self.refuse_encode(&error);
                        return;
                    }
                }
            }
        };
        let head = serve::Head::new(self.rid_str(), &self.request.model);
        let prompt_tokens = prompt.tokens();
        let generation = serve::start(
            backend,
            prompt,
            self.request.max_output_tokens,
            self.request.n,
            self.request.stream,
            head,
            slot,
        );
        self.native_cancel = Some(generation.cancel);
        if self.request.stream {
            crate::stream::relay(self, generation.chunks, Some(prompt_tokens)).await;
        } else {
            let call: BackendCall<Value> =
                Box::pin(serve::collect_unary(generation.chunks, self.request.n));
            self.run_unary(None, call).await;
        }
    }

    /// A unary request: `call` returns the backend's `chat.completion` body. `slot` (when the call does not own
    /// it) is held until the call has ended.
    async fn run_unary(&mut self, slot: Option<Slot>, mut call: BackendCall<Value>) {
        let Some(result) = self.drive(&mut call).await else {
            self.give_up(slot, call);
            return;
        };
        drop(slot);
        let value = match result {
            Ok(value) => value,
            Err(e) => {
                self.backend_failed(&e);
                return;
            }
        };
        let usage = value.get("usage").cloned();
        match classify_response(&value, self.request.max_output_tokens, self.request.n) {
            Ok(Finish::Stop | Finish::ToolCalls | Finish::Length | Finish::Repetition) => {
                self.complete_unary(value, usage.as_ref())
            }
            Ok(Finish::Abort) => {
                let (state, reason) = match self.reason {
                    Some(reason) => ("cancelled", reason),
                    None => ("failed", REASON_BACKEND_ABORT),
                };
                self.settle_failure(state, reason, usage.as_ref());
                let response =
                    self.failure_response(&reason_error(reason), state, reason, usage.as_ref());
                self.client.respond(response);
            }
            Ok(Finish::BackendClampedOutput { .. }) => {
                let reason = "backend_clamped_output";
                self.settle_failure("failed", reason, usage.as_ref());
                let response =
                    self.failure_response(&reason_error(reason), "failed", reason, usage.as_ref());
                self.client.respond(response);
            }
            Err(e) => {
                let error = ApiError::from(&e);
                self.settle_failure("failed", error.code, usage.as_ref());
                let response = self.failure_response(&error, "failed", error.code, usage.as_ref());
                self.client.respond(response);
            }
        }
    }

    /// The backend call ended with `e` before any response: after our abort, the backend ending the call in any
    /// way acknowledges the cancel; otherwise the request fails with `e`'s error.
    pub(crate) fn backend_failed(&mut self, e: &BackendError) {
        let (state, error) = match self.reason {
            Some(reason) => ("cancelled", reason_error(reason)),
            None => ("failed", self.backend_error(e)),
        };
        let reason = self.reason.unwrap_or(error.code);
        self.settle_failure(state, reason, None);
        let response = self.failure_response(&error, state, reason, None);
        self.client.respond(response);
    }

    /// The client error of backend failure `e`: the reason a remote work's terminal carried (H2, so the error
    /// code is that reason), else `e`'s own. A tool schema refusal counts only on a request with tools (M2.4 D2);
    /// without them it is an ordinary backend 400 (local `Tool {i} function` text, a remote `invalid_value`).
    /// A grammar compile refusal (M2.8 D6) names the constraint field (`tool_choice` for a tool constraint) and
    /// counts only on a constrained request.
    pub(crate) fn backend_error(&self, e: &BackendError) -> ApiError {
        let plain_400 = || {
            ApiError::from(&BackendError::Status {
                status: 400,
                message: String::new(),
            })
        };
        match (self.remote.as_ref().and_then(|call| call.reason()), e) {
            (Some(reason), _) => reason_error(reason),
            (None, BackendError::ToolSchema { .. }) if self.request.tools.is_none() => plain_400(),
            (None, BackendError::GrammarCompile) => match self.request.constraint_field() {
                Some(field) => ApiError::grammar_compile(field),
                None if self.request.tool_constrained() => ApiError::grammar_compile("tool_choice"),
                None => plain_400(),
            },
            (None, e) => ApiError::from(e),
        }
    }

    /// [`Self::drive`] gave up on `call`: a [`superseded`](Exec::superseded) request answers that (its slot
    /// belongs to the supersede now); otherwise the abort was not acknowledged in time, so the request fails now
    /// and keeps `slot` until the backend answers.
    fn give_up<T: Send + 'static>(&mut self, slot: Option<Slot>, call: BackendCall<T>) {
        if self.superseded() {
            drop(call);
            drop(slot);
            let reason = REASON_WORKER_EPOCH_SUPERSEDED;
            let response = self.failure_response(&reason_error(reason), "failed", reason, None);
            self.client.respond(response);
            return;
        }
        let reason = REASON_CANCEL_UNACKNOWLEDGED;
        self.settle_failure("failed", reason, None);
        let response = self.failure_response(&reason_error(reason), "failed", reason, None);
        self.client.respond(response);
        tokio::spawn(async move {
            let _ = call.await;
            drop(slot);
        });
    }

    fn complete_unary(&mut self, mut value: Value, usage: Option<&Value>) {
        let message = value.pointer("/choices/0/message");
        let content = message
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let mut calls = ToolCallAssembly::default();
        calls.push(message.and_then(|m| m.get("tool_calls")));
        let commit = self.session_commit(assistant_reply(content, calls.into_calls()));
        let reported = self.usage(usage, "completed");
        self.record_usage(&reported);
        let mut cere = self.cere(&reported);
        cere.insert("state".to_owned(), json!("completed"));
        // ApiLenient D4: the inert fields the request carried (absent when none, so valid-today bodies are unchanged).
        if !self.request.ignored_params.is_empty() {
            cere.insert(
                "ignored_params".to_owned(),
                json!(self.request.ignored_params),
            );
        }
        if let Some((_, revision, session_id)) = &commit {
            cere.insert("session_id".to_owned(), json!(session_id));
            cere.insert("revision".to_owned(), json!(revision));
        }
        value["cere"] = Value::Object(cere);
        let bytes = match self.surface.renderer() {
            Some(render) => render.render_unary(&value, self.rid_str()),
            None => serde_json::to_vec(&value).expect("JSON values serialize"),
        };
        let completed = self.app.ledger().complete(
            self.rid_str(),
            Completion {
                retained: Some(RetainedResponse {
                    status: 200,
                    body: bytes.clone(),
                }),
                session_commit: commit.map(|c| c.0),
            },
            unix_ms(),
        );
        if let Err(e) = completed {
            let reason = e.code();
            self.settle_failure("failed", reason, usage);
            let error = match e {
                LedgerError::Session(s) => session_error(s),
                other => internal_error(format!("cannot commit the response: {}", other.code())),
            };
            let response = self.failure_response(&error, "failed", reason, usage);
            self.client.respond(response);
            return;
        }
        self.app.log_terminal(&self.scope, self.rid_str());
        let mut response = Response::new(Body::from(bytes));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        self.client
            .respond(with_request_id(response, self.rid_str()));
    }
}

/// The client error for a server-assigned terminal reason.
pub(crate) fn reason_error(reason: &str) -> ApiError {
    let (status, error_type, code, message): (u16, &'static str, &'static str, &str) = match reason
    {
        REASON_QUEUE_FULL => (429, SERVER_ERROR, "queue_full", "admission queue is full"),
        REASON_DEADLINE_EXCEEDED => (
            504,
            SERVER_ERROR,
            "deadline_exceeded",
            "request deadline exceeded",
        ),
        REASON_SHUTDOWN => (
            503,
            SERVER_ERROR,
            "shutting_down",
            "server is shutting down",
        ),
        REASON_CANCEL_REQUESTED => (
            409,
            INVALID_REQUEST,
            "request_cancelled",
            "request was cancelled",
        ),
        REASON_CLIENT_DISCONNECTED => (
            499,
            INVALID_REQUEST,
            "client_disconnected",
            "client disconnected",
        ),
        "client_backpressure" => (
            503,
            SERVER_ERROR,
            "client_backpressure",
            "client did not read the stream in time",
        ),
        REASON_CANCEL_UNACKNOWLEDGED => (
            502,
            SERVER_ERROR,
            "cancel_unacknowledged",
            "backend did not acknowledge the abort in time",
        ),
        "backend_clamped_output" => (
            502,
            SERVER_ERROR,
            "backend_clamped_output",
            "backend stopped at length below the requested output budget",
        ),
        REASON_BACKEND_ABORT => (
            502,
            SERVER_ERROR,
            "backend_abort",
            "backend aborted the request",
        ),
        REASON_WORKER_NOT_READY => (503, SERVER_ERROR, "not_ready", "worker is not ready"),
        REASON_NO_ALLOWED_WORKER => (
            503,
            SERVER_ERROR,
            REASON_NO_ALLOWED_WORKER,
            "no worker allowed for this scope (trust tier or owner-only window) was available",
        ),
        REASON_WORKER_EPOCH_SUPERSEDED => (
            503,
            SERVER_ERROR,
            "worker_epoch_superseded",
            "the worker's backend was relaunched or rejoined; the request was abandoned",
        ),
        REASON_SESSION_OWNER_SUPERSEDED => (
            409,
            INVALID_REQUEST,
            REASON_SESSION_OWNER_SUPERSEDED,
            OWNER_SUPERSEDED_MESSAGE,
        ),
        REASON_JOURNAL_ERROR => (
            500,
            SERVER_ERROR,
            "journal_error",
            "the request ledger could not be written: journal_error",
        ),
        // H2: a remote work's terminal for a worker or coordinator restart or a model generation change, and
        // (D5) a remote request whose worker turned Lost.
        wire::WORKER_RESTART => (
            503,
            SERVER_ERROR,
            "worker_restart",
            "the remote worker restarted; the request was abandoned",
        ),
        REASON_WORKER_LOST => (
            503,
            SERVER_ERROR,
            "worker_lost",
            "the remote worker was lost; the request was abandoned",
        ),
        wire::MODEL_GENERATION_SUPERSEDED => (
            503,
            SERVER_ERROR,
            "model_generation_superseded",
            "the remote worker's model generation changed; the request was abandoned",
        ),
        wire::COORDINATOR_RESTART => (
            503,
            SERVER_ERROR,
            "coordinator_restart",
            "the coordinator restarted; the request was abandoned",
        ),
        _ => (500, SERVER_ERROR, "internal_error", "request ended"),
    };
    ApiError::new(status, error_type, code, message, None)
}
