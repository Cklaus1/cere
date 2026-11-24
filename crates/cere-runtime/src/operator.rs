//! Operator API of `cered` (m35 design note Q13): drain and undrain a worker. Track OPS
//! (build-loop-m35 design note). Load and unload a remote worker's model (m36 design note Q9).
//!
//! `POST /v1/cere/workers/{worker_id}/drain` and `.../undrain` exist only when `[auth] operator_token_env` is
//! configured (404 otherwise); a client scope token is 403, a missing or unknown bearer 401, a worker id that is
//! neither a local device nor a `[federation.workers]` id 404. Success journals the intent through
//! [`App::set_drain_intent`] and answers `{"worker_id", "draining"}`; repeating it is idempotent.
//!
//! `POST /v1/cere/workers/{worker_id}/load` and `.../unload` (no body) share that auth model and the 404. A local
//! native replica of a multi-replica deployment (m32n design note D10) is unloaded and loaded in process:
//! `unload` needs the operator's drain intent (409 `worker_not_draining`) and nothing admitted on it (409
//! `requests_in_flight`), releases its executor and answers 202 `{"worker_id", "native": "unloaded"}`; `load`
//! makes its next probe run the load again (Ready at epoch + 1) and answers 202 `{"worker_id", "native":
//! "loading"}`, except after a sticky fault whose one reload failed (409 `worker_faulted`, N4: a `cered` restart
//! recovers it). Any other local worker, or a remote one last described in attached mode, is 409
//! `model_unmanaged`; a remote worker not described since this start 409 `worker_not_described`. Success
//! journals a new residency intent through [`App::set_residency`] (each call is one) and answers 202
//! `{"worker_id", "desired", "intent_seq", "model"}` after the admission view is pushed. Every `[[deployments]]`
//! entry of a running `cered` has a pinned deployment (`App::open` refuses a config without one) and each worker
//! belongs to exactly one entry, whose lock its residency uses (m5 design note D8), so Q9's
//! `deployment_unpinned` cannot occur.
//!
//! `GET /v1/cere/prefix_cache` (operator token only) answers the prefix cache counters of every local native worker,
//! the single form included: `{"workers": {<worker_id>: {"loaded", "prefix_cache", "reason"}}}`, `prefix_cache` the
//! cumulative `cere_model_prefix_stats` of the loaded model (`hits`, `captures`, `capture_skipped`,
//! `capture_page_mismatch`, `inserts`, `state_evictions`, `leaf_evictions`, `host_spills`, `host_loads`,
//! `host_drops`, `alt_edges`, `alt_edges_enabled`; they restart at each load, a flush keeps them) plus, from
//! `cere_model_prefix_capture_stats` (prefix-token design note PT0a), `capture_skipped_by_site`
//! (`{"prefill_slot", "decode_track", "end_pair"}`, summing to `capture_skipped`), `hit_shortfall_tokens` and
//! `hit_shortfall_lookups` (each null on a library without that export), or null with a
//! `reason` (`not_loaded`, `cache_off`, `export_missing` for a library from before the export, `read_failed`); every
//! `GET /v1/cere/cards` entry carries the same object as `prefix_cache`.
//!
//! `POST /v1/cere/sessions/{id}/migrate` and `GET /v1/cere/sessions/{id}/owner` (m4 design note Q4-Q6,
//! design-m4 design note §3) answer `503 shutting_down` while stopping, then share that auth model. The migrate
//! body is exactly `{"to_worker": string, "expected_owner_generation": u64, "force"?: bool}`; the move goes
//! through [`App::migrate_session`] and answers `{"session_id", "worker", "owner_generation", "revision",
//! "mode": "replay"}`. Its refusals change nothing: 404 `session_not_found`, 409 `owner_generation_conflict`
//! (the current `worker` and `owner_generation` inside `error`), 404 `worker_not_found`, 409 `model_mismatch`
//! (M5 D8: `to_worker` serves another model than the session's), 409 `worker_not_eligible`, `already_owner`,
//! `owner_healthy`, `session_busy`; a journal failure is 500. The
//! owner read answers `{"session_id", "worker", "owner_generation", "revision"}` for a live session of any scope.
//!
//! `GET /v1/cere/levers` and `POST /v1/cere/levers` (m23 design note PI-04, PI-05) share that auth model. The
//! read answers `{"registry": <tests/fixtures/lever_registry.json levers>, "active": {lever: impl} | null,
//! "hot_swap": bool}` (`active` null and `hot_swap` false until the native backend is loaded). The dev hot swap
//! body is `{"levers": {lever: impl, ...}}` (a partial set over the active one) with an optional `"params": {lever:
//! {param: value, ...}}` (SelfTune D6, O292 D-9: checked against the lever's next impl in the registry) and answers
//! 200 `{"active": {...}}` read back from the executor; its refusals change nothing: 403 `lever_hot_swap_disabled`
//! (the deployment does not set `lever_hot_swap`), 400 `invalid_lever` (a malformed body, an unknown lever, impl or
//! param, a launch param outside its range, or a value the model cannot run), 409 `lever_build_param` (a build
//! param: fixed at library build), 409 `lever_restart_only` (a restart-class lever would change), 409
//! `requests_in_flight` (admission holds a request, or the executor answers `CERE_ERR_BUSY`), 409
//! `native_not_loaded` (no native backend), 500 `executor_error`.
//!
//! `POST /v1/cere/workers/{worker_id}/flush_cache` (m210 design note D11, N2 (a)) shares that auth model and
//! the 404. The body is empty or exactly `{"timeout_s"?: number in [0, 30]}` (default 30). It drops every entry of
//! a local native worker's prefix cache, retrying while the executor is busy (a ticket in flight or a session
//! holding a hit) until `timeout_s`, then 409 `worker_busy`; success answers `{"worker_id", "prefix_cache":
//! "flushed"}`, and `"off"` (nothing to flush) for a worker that runs no prefix cache. A worker that is not
//! local `native_cuda` is 409 `worker_not_native`, one not loaded yet 503 `worker_not_ready`.
//!
//! `GET /v1/cere/cards` (m32n design note D18, F15: it reveals every scope's traffic, so only the operator token
//! reads it) shares that auth model and answers `{"cards": {<worker_id>: {...}}}`, one entry per native replica of
//! a multi-replica deployment (empty for the single form and every other kind): `device_uuid`, `state`, `epoch`,
//! `loaded`, `rows_max`, `active_rows` (admitted requests), `kv_pool_tokens`, `reserved_pool_tokens`,
//! `open_sessions`, `fifo_waiters` (session creates waiting for KV pages), `preemptions`, `prefix_hits`,
//! `committed_tokens`, `decode_tokens_per_s` (committed since the previous read of this route), `spec_verify_steps`,
//! `spec_accepted_drafts`, `spec_accept_length` (accepted drafts per speculate round, null without one),
//! `free_mib` (null: the executor reports no device memory yet; the GPU phase adds it) and `load` (D22 (3), null
//! while not loaded): `background` (a load after the replica's first, the other card serving), `nice` (its staging
//! threads' nice value), `allocate_nice` and `allocate_cpus` (the nice value and CPU count of the thread its
//! ALLOCATE ran on, which the executor threads started there inherit), `staging_slots` (its staging buffers, the cap
//! on copies in flight) and `copies_in_flight_max` (the most buffers' worth of bytes in flight). The load counters
//! are 0 while the card is not loaded and restart at each load.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Map, Value};

use crate::api_types::{ApiError, ReadyzModel, INVALID_REQUEST, SERVER_ERROR};
use crate::app::{
    unix_ms, App, Backend, DrainError, LeverSwapError, NativeBackend, NativeResidencyError,
};
use crate::auth::OperatorAccess;
use crate::fed::registry::{Desired, ResidencyError};
use crate::journal::Residency;
use crate::ledger::{MigrateError, MigrateRequest};
use crate::levers;
use crate::native::driver::DriverError;
use crate::native::ffi::{PrefixCaptureStats, PrefixStats, StatusError};
use crate::reload::ReloadError;
use crate::server::{api_error, json_response, ledger_error, shutting_down, unauthorized};

/// `POST /v1/cere/workers/{worker_id}/drain`.
pub(crate) async fn drain(
    State(app): State<Arc<App>>,
    Path(worker_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    set_drain(&app, &worker_id, &headers, true)
}

/// `POST /v1/cere/workers/{worker_id}/undrain`.
pub(crate) async fn undrain(
    State(app): State<Arc<App>>,
    Path(worker_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    set_drain(&app, &worker_id, &headers, false)
}

/// The longest `timeout_s` a flush waits for an idle executor (SGLang's `/flush_cache?timeout=30`).
const FLUSH_TIMEOUT_MAX_S: f64 = 30.0;
/// Pause between two BUSY flush attempts.
const FLUSH_RETRY: std::time::Duration = std::time::Duration::from_millis(20);

/// `POST /v1/cere/workers/{worker_id}/flush_cache`.
pub(crate) async fn flush_cache(
    State(app): State<Arc<App>>,
    Path(worker_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    let timeout_s = match flush_timeout(&body) {
        Ok(t) => t,
        Err(e) => return api_error(&e),
    };
    let Some(replica) = app.replica(&worker_id) else {
        return not_found("worker_not_found", "no configured worker has this id");
    };
    let Backend::Native(slot) = &replica.backend else {
        return conflict(
            "worker_not_native",
            "only a local native_cuda worker has a native prefix cache",
        );
    };
    let Some(backend) = slot.get() else {
        return api_error(&ApiError::new(
            503,
            SERVER_ERROR,
            "worker_not_ready",
            "the native worker is not loaded",
            None,
        ));
    };
    let flushed = |state: &str| {
        json_response(
            StatusCode::OK,
            &json!({ "worker_id": worker_id, "prefix_cache": state }),
        )
    };
    if backend.capabilities.native_prefix_cache == 0 {
        return flushed("off");
    }
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs_f64(timeout_s);
    loop {
        let b = backend.clone();
        let result = tokio::task::spawn_blocking(move || b.driver.flush_prefix_cache(&b.model))
            .await
            .unwrap_or(Err(DriverError::Stopped));
        match result {
            Ok(()) => return flushed("flushed"),
            Err(DriverError::Status {
                status: StatusError::Busy,
                ..
            }) if tokio::time::Instant::now() + FLUSH_RETRY <= deadline => {
                tokio::time::sleep(FLUSH_RETRY).await;
            }
            Err(DriverError::Status {
                status: StatusError::Busy,
                ..
            }) => {
                return conflict(
                    "worker_busy",
                    "the native executor stayed busy (a ticket in flight or a session holding a cache hit) \
                     until timeout_s",
                )
            }
            Err(e) => {
                return api_error(&ApiError::new(
                    500,
                    SERVER_ERROR,
                    "internal_error",
                    format!("native prefix cache flush: {e}"),
                    None,
                ))
            }
        }
    }
}

/// The flush body: empty, or exactly `{"timeout_s"?: number in [0, 30]}`; 30 when absent.
fn flush_timeout(body: &[u8]) -> Result<f64, ApiError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(FLUSH_TIMEOUT_MAX_S);
    }
    let fields = match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(fields)) => fields,
        Ok(_) => return Err(ApiError::invalid_json("body must be a JSON object")),
        Err(e) => return Err(ApiError::invalid_json(e.to_string())),
    };
    if let Some(extra) = fields.keys().find(|k| k.as_str() != "timeout_s") {
        return Err(ApiError::unsupported_parameter(
            extra.clone(),
            format!("unsupported flush_cache parameter '{extra}'"),
        ));
    }
    match fields.get("timeout_s") {
        None => Ok(FLUSH_TIMEOUT_MAX_S),
        Some(t) => t
            .as_f64()
            .filter(|t| (0.0..=FLUSH_TIMEOUT_MAX_S).contains(t))
            .ok_or_else(|| {
                ApiError::invalid_value("timeout_s", "timeout_s must be a number from 0 to 30")
            }),
    }
}

fn set_drain(app: &App, worker_id: &str, headers: &HeaderMap, draining: bool) -> Response {
    if let Err(refused) = authorize(app, headers) {
        return refused;
    }
    match app.set_drain_intent(worker_id, draining) {
        Ok(()) => json_response(
            StatusCode::OK,
            &json!({ "worker_id": worker_id, "draining": draining }),
        ),
        Err(DrainError::UnknownWorker) => {
            not_found("worker_not_found", "no configured worker has this id")
        }
        Err(DrainError::Journal(e)) => ledger_error(&e),
    }
}

/// `POST /v1/cere/workers/{worker_id}/load`.
pub(crate) async fn load(
    State(app): State<Arc<App>>,
    Path(worker_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    set_residency(&app, &worker_id, &headers, Residency::Resident)
}

/// `POST /v1/cere/workers/{worker_id}/unload`.
pub(crate) async fn unload(
    State(app): State<Arc<App>>,
    Path(worker_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    set_residency(&app, &worker_id, &headers, Residency::Unloaded)
}

/// `POST /v1/cere/config/reload` (no body): a live config reload ([`crate::reload`]) behind the operator token.
/// 200 `{"applied", "generation", "config_hash", "changed"}` (`applied` false: the file has no change);
/// 409 `restart_required` (the changed restart-class keys in the message), `config_invalid`,
/// `reload_hook_refused` or `reload_unavailable` (not started from a file), each with nothing applied;
/// 503 `shutting_down`.
pub(crate) async fn reload_config(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    // The file and the deployment files it names are read and hashed off the async workers.
    let reloaded = match tokio::task::spawn_blocking(move || app.reload()).await {
        Ok(result) => result,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    };
    match reloaded {
        Ok(r) => json_response(
            StatusCode::OK,
            &json!({
                "applied": r.applied,
                "generation": r.generation,
                "config_hash": r.config_hash,
                "changed": r.changed,
            }),
        ),
        Err(e) => {
            let code = match &e {
                ReloadError::Stopping => return shutting_down(),
                ReloadError::Unavailable => "reload_unavailable",
                ReloadError::Invalid(_) => "config_invalid",
                ReloadError::Restart(_) => "restart_required",
                ReloadError::Hook { .. } => "reload_hook_refused",
            };
            conflict(code, &e.to_string())
        }
    }
}

fn set_residency(app: &App, worker_id: &str, headers: &HeaderMap, desired: Residency) -> Response {
    if let Err(refused) = authorize(app, headers) {
        return refused;
    }
    if let Some(native) = app.native_residency(worker_id, desired) {
        return match native {
            Ok(state) => json_response(
                StatusCode::ACCEPTED,
                &json!({ "worker_id": worker_id, "native": state }),
            ),
            Err(NativeResidencyError::NotDraining) => conflict(
                "worker_not_draining",
                "drain the native replica before unloading it",
            ),
            Err(NativeResidencyError::InFlight) => conflict(
                "requests_in_flight",
                "the native replica still holds admitted requests",
            ),
            Err(NativeResidencyError::Faulted) => conflict(
                "worker_faulted",
                "the native replica's reload after a device fault failed; a cered restart recovers it",
            ),
        };
    }
    match app.set_residency(worker_id, desired) {
        Ok(model) => json_response(
            StatusCode::ACCEPTED,
            &json!({
                "worker_id": worker_id,
                "desired": Desired::from(desired),
                "intent_seq": model.intent_seq,
                "model": ReadyzModel::from(model),
            }),
        ),
        Err(ResidencyError::UnknownWorker) => {
            not_found("worker_not_found", "no configured worker has this id")
        }
        Err(ResidencyError::Unmanaged) => conflict(
            "model_unmanaged",
            "the worker's model is not managed by its agent (a local worker, or last described attached)",
        ),
        Err(ResidencyError::NotDescribed) => conflict(
            "worker_not_described",
            "the worker has not described its model since this cered started",
        ),
        Err(ResidencyError::Journal(e)) => ledger_error(&e),
    }
}

/// `GET /v1/cere/sessions/{id}/owner`.
pub(crate) async fn session_owner(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if app.is_stopping() {
        return shutting_down();
    }
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    let owner = app.ledger().session_owner(&id, unix_ms());
    match owner {
        Ok(owner) => json_response(
            StatusCode::OK,
            &json!({
                "session_id": owner.session_id,
                "worker": owner.worker,
                "owner_generation": owner.owner_generation,
                "revision": owner.revision,
            }),
        ),
        Err(e) => ledger_error(&e),
    }
}

/// `POST /v1/cere/sessions/{id}/migrate`.
pub(crate) async fn migrate(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if app.is_stopping() {
        return shutting_down();
    }
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    let fields = match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(fields)) => fields,
        Ok(_) => return api_error(&ApiError::invalid_json("body must be a JSON object")),
        Err(e) => return api_error(&ApiError::invalid_json(e.to_string())),
    };
    let req = match migrate_request(&id, &fields) {
        Ok(req) => req,
        Err(e) => return api_error(&e),
    };
    match app.migrate_session(req) {
        Ok(moved) => json_response(
            StatusCode::OK,
            &json!({
                "session_id": moved.session_id,
                "worker": moved.worker,
                "owner_generation": moved.owner_generation,
                "revision": moved.revision,
                "mode": "replay",
            }),
        ),
        Err(MigrateError::NotFound) => not_found("session_not_found", "session not found"),
        Err(MigrateError::Conflict {
            worker,
            owner_generation,
        }) => {
            let mut body = ApiError::new(
                409,
                INVALID_REQUEST,
                "owner_generation_conflict",
                "the session's owner generation is not the expected one; read the owner and retry",
                None,
            )
            .body();
            body["error"]["worker"] = json!(worker);
            body["error"]["owner_generation"] = json!(owner_generation);
            json_response(StatusCode::CONFLICT, &body)
        }
        Err(MigrateError::WorkerNotFound) => {
            not_found("worker_not_found", "no configured worker has this id")
        }
        Err(MigrateError::ModelMismatch) => api_error(&ApiError::new(
            409,
            INVALID_REQUEST,
            "model_mismatch",
            "the destination worker serves another model than the session's",
            Some("model".to_owned()),
        )),
        Err(MigrateError::WorkerNotEligible) => {
            conflict("worker_not_eligible", "the destination worker is not ready")
        }
        Err(MigrateError::AlreadyOwner) => conflict(
            "already_owner",
            "the destination worker already owns the session",
        ),
        Err(MigrateError::OwnerHealthy) => conflict(
            "owner_healthy",
            "force takes over only from an owner that is not ready",
        ),
        Err(MigrateError::SessionBusy) => conflict(
            "session_busy",
            "a turn is writing the session; retry after it ends",
        ),
        Err(MigrateError::Journal(e)) => ledger_error(&e),
    }
}

/// The migrate body, exactly `{"to_worker": string, "expected_owner_generation": u64, "force"?: bool}`: a
/// missing key, then an unknown one, then a value of the wrong type is refused (the `fork` body's order).
fn migrate_request<'a>(
    session_id: &'a str,
    fields: &'a Map<String, Value>,
) -> Result<MigrateRequest<'a>, ApiError> {
    const KEYS: [&str; 3] = ["to_worker", "expected_owner_generation", "force"];
    let (Some(to_worker), Some(expected)) = (
        fields.get("to_worker"),
        fields.get("expected_owner_generation"),
    ) else {
        let missing = KEYS[..2].iter().find(|k| !fields.contains_key(**k));
        return Err(ApiError::missing_parameter(*missing.unwrap_or(&KEYS[0])));
    };
    if let Some(extra) = fields.keys().find(|k| !KEYS.contains(&k.as_str())) {
        return Err(ApiError::unsupported_parameter(
            extra.clone(),
            format!("unsupported migrate parameter '{extra}'"),
        ));
    }
    let to_worker = to_worker
        .as_str()
        .ok_or_else(|| ApiError::invalid_value("to_worker", "to_worker must be a string"))?;
    let expected_owner_generation = expected.as_u64().ok_or_else(|| {
        ApiError::invalid_value(
            "expected_owner_generation",
            "expected_owner_generation must be a non-negative integer",
        )
    })?;
    let force = match fields.get("force") {
        None => false,
        Some(force) => force
            .as_bool()
            .ok_or_else(|| ApiError::invalid_value("force", "force must be a boolean"))?,
    };
    Ok(MigrateRequest {
        session_id,
        to_worker,
        expected_owner_generation,
        force,
    })
}

/// The operator routes' auth model (m35 design note Q13): the route does not exist without an operator token;
/// a client scope token is 403, a missing or unknown bearer 401.
pub(crate) fn authorize(app: &App, headers: &HeaderMap) -> Result<(), Response> {
    let authorization = headers
        .get(header::AUTHORIZATION)
        .map(HeaderValue::as_bytes);
    match app.auth.authorize_operator(authorization) {
        OperatorAccess::Operator => Ok(()),
        OperatorAccess::Disabled => Err(not_found("not_found", "not found")),
        OperatorAccess::Client => Err(forbidden()),
        OperatorAccess::Unauthenticated => Err(unauthorized()),
    }
}

/// `GET /v1/cere/cards` (m32n design note D18).
pub(crate) async fn cards(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    let loads: BTreeMap<String, (u64, u64)> = app
        .admission
        .worker_loads("")
        .into_iter()
        .map(|w| (w.id.to_string(), (w.inflight, w.reserved)))
        .collect();
    let mut cards = Map::new();
    for replica in app.replicas() {
        let id = replica.worker_id();
        let Some(card) = app.native_card(id) else {
            continue;
        };
        let (state, epoch) = app
            .worker_state(id)
            .map_or(("unknown", 0), |(s, e)| (s.as_str(), e));
        let (active_rows, reserved) = loads.get(id).copied().unwrap_or_default();
        let native = replica.backend.native();
        let read = |f: &dyn Fn(&crate::app::NativeBackend) -> u64| native.as_deref().map_or(0, f);
        let (open_sessions, fifo_waiters) = native.as_deref().map_or((0, 0), |n| n.sessions.load());
        let prefix_cache = match &native {
            Some(n) => prefix_cache_counters(n.clone())
                .await
                .unwrap_or(Value::Null),
            None => Value::Null,
        };
        let verify_steps = read(&|n| n.spec_verify_steps.load(Ordering::Acquire));
        let accepted = read(&|n| n.spec_accepted_drafts.load(Ordering::Acquire));
        cards.insert(
            id.to_owned(),
            json!({
                "device_uuid": card.device_uuid,
                "state": state,
                "epoch": epoch,
                "loaded": native.is_some(),
                "rows_max": card.rows_max,
                "active_rows": active_rows,
                "kv_pool_tokens": card.kv_pool_tokens,
                "reserved_pool_tokens": reserved,
                "open_sessions": open_sessions,
                "fifo_waiters": fifo_waiters,
                "preemptions": read(&|n| n.preemption.as_ref().map_or(0, |p| p.preemptions())),
                "prefix_hits": read(&|n| n.prefix_hits.load(Ordering::Acquire)),
                "prefix_cache": prefix_cache,
                "committed_tokens": read(&|n| n.committed.load(Ordering::Acquire)),
                "decode_tokens_per_s": native.as_deref().map_or(0.0, |n| n.decode_rate()),
                "spec_verify_steps": verify_steps,
                "spec_accepted_drafts": accepted,
                "spec_accept_length": (verify_steps > 0).then(|| accepted as f64 / verify_steps as f64),
                "free_mib": Value::Null,
                "load": native.as_deref().map(|n| json!({
                    "background": n.load.background,
                    "nice": n.load.nice,
                    "allocate_nice": n.load.allocate_nice,
                    "allocate_cpus": n.load.allocate_cpus,
                    "staging_slots": n.load.staging_slots,
                    "copies_in_flight_max": n.load.copies_in_flight_max,
                })),
            }),
        );
    }
    json_response(StatusCode::OK, &json!({ "cards": cards }))
}

/// One native backend's prefix cache counters (`cere_model_prefix_stats`: cumulative since the model was loaded, so
/// they restart at each load like the cards' `load`; a flush keeps them), or the reason there are none: `cache_off`
/// (no `prefix_state_slots`), `export_missing` (a library from before the export) or `read_failed`. The read is a
/// driver round trip, so it runs off the async threads.
pub(crate) async fn read_prefix_stats(
    native: Arc<NativeBackend>,
) -> Result<PrefixStats, &'static str> {
    if native.capabilities.native_prefix_cache == 0 {
        return Err("cache_off");
    }
    let read = tokio::task::spawn_blocking(move || native.driver.prefix_stats(&native.model))
        .await
        .unwrap_or(Err(DriverError::Stopped));
    match read {
        Ok(s) => Ok(s),
        Err(DriverError::Missing(_)) => Err("export_missing"),
        Err(DriverError::Status {
            status: StatusError::Unsupported,
            ..
        }) => Err("cache_off"),
        Err(_) => Err("read_failed"),
    }
}

/// One native backend's capture-skip sites and lost hit tokens (`cere_model_prefix_capture_stats`, PT0a; cumulative
/// like [`read_prefix_stats`]), `None` without a prefix cache, on a library from before the export or when the read
/// fails (the [`read_prefix_stats`] reason covers those).
pub(crate) async fn read_prefix_capture_stats(
    native: Arc<NativeBackend>,
) -> Option<PrefixCaptureStats> {
    if native.capabilities.native_prefix_cache == 0 {
        return None;
    }
    tokio::task::spawn_blocking(move || native.driver.prefix_capture_stats(&native.model))
        .await
        .ok()?
        .ok()
}

/// [`read_prefix_stats`] as JSON. Counters only: no prompt, token or namespace.
async fn prefix_cache_counters(native: Arc<NativeBackend>) -> Result<Value, &'static str> {
    let s = read_prefix_stats(native.clone()).await?;
    let sites = read_prefix_capture_stats(native).await;
    Ok(json!({
        "alt_edges_enabled": s.alt_edges_on != 0,
        "hits": s.hits,
        "captures": s.captures,
        "capture_skipped": s.capture_skipped,
        "capture_skipped_by_site": sites.map(|c| json!({
            "prefill_slot": c.skipped_prefill_slot,
            "decode_track": c.skipped_decode_track,
            "end_pair": c.skipped_end_pair,
        })),
        "capture_page_mismatch": s.capture_page_mismatch,
        "inserts": s.inserts,
        "state_evictions": s.state_evictions,
        "leaf_evictions": s.leaf_evictions,
        "host_spills": s.host_spills,
        "host_loads": s.host_loads,
        "host_drops": s.host_drops,
        "alt_edges": s.alt_edges,
        "hit_shortfall_tokens": sites.map(|c| c.hit_shortfall_tokens),
        "hit_shortfall_lookups": sites.map(|c| c.hit_shortfall_lookups),
    }))
}

/// `GET /metrics` (ops-hardening design note OH-05): the Prometheus text exposition ([`crate::metrics`]). Operator
/// token only, like every operator route: the counters cover every scope's traffic.
pub(crate) async fn metrics(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    let body = crate::metrics::render(&app).await;
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static(crate::metrics::CONTENT_TYPE),
        )],
        body,
    )
        .into_response()
}

/// `GET /v1/cere/prefix_cache`: the prefix cache counters of every local native worker, the single form (production's,
/// where `GET /v1/cere/cards` is `{}`) included. Operator token only: the counters cover every scope's traffic.
/// `{"workers": {<worker_id>: {"loaded": bool, "prefix_cache": object | null, "reason": null | "not_loaded" |
/// "cache_off" | "export_missing" | "read_failed"}}}`.
pub(crate) async fn prefix_cache(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    let mut workers = Map::new();
    for replica in app.replicas() {
        if !matches!(replica.backend, Backend::Native(_)) {
            continue;
        }
        let (loaded, counters, reason) = match replica.backend.native() {
            None => (false, Value::Null, Some("not_loaded")),
            Some(native) => match prefix_cache_counters(native).await {
                Ok(counters) => (true, counters, None),
                Err(reason) => (true, Value::Null, Some(reason)),
            },
        };
        workers.insert(
            replica.worker_id().to_owned(),
            json!({ "loaded": loaded, "prefix_cache": counters, "reason": reason }),
        );
    }
    json_response(StatusCode::OK, &json!({ "workers": workers }))
}

/// `GET /v1/cere/levers` (m23 design note PI-04, PI-05).
pub(crate) async fn levers(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    json_response(
        StatusCode::OK,
        &json!({
            "registry": levers::REGISTRY,
            "active": app.native_levers(),
            "hot_swap": app.native().is_some_and(|n| n.lever_hot_swap),
        }),
    )
}

/// `POST /v1/cere/levers` (PI-04): the dev hot swap between requests.
pub(crate) async fn set_levers(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(refused) = authorize(&app, &headers) {
        return refused;
    }
    let (changes, params) = match lever_changes(&body) {
        Ok(parsed) => parsed,
        Err(message) => return invalid_lever(&message),
    };
    // The swap waits on the driver thread (`cere_model_set_selection`, then `cere_model_selection`).
    let swapped = {
        let app = app.clone();
        tokio::task::spawn_blocking(move || app.swap_levers(&changes, &params)).await
    };
    match swapped {
        Ok(Ok(active)) => json_response(StatusCode::OK, &json!({ "active": active })),
        Ok(Err(LeverSwapError::Disabled)) => api_error(&ApiError::new(
            403,
            INVALID_REQUEST,
            "lever_hot_swap_disabled",
            "the deployment does not set lever_hot_swap",
            None,
        )),
        Ok(Err(LeverSwapError::Invalid(message))) => invalid_lever(&message),
        Ok(Err(LeverSwapError::BuildParam(message))) => conflict("lever_build_param", &message),
        Ok(Err(LeverSwapError::RestartOnly(name))) => conflict(
            "lever_restart_only",
            &format!("{name} is a restart-class lever: it changes only with a restart"),
        ),
        Ok(Err(LeverSwapError::InFlight)) => conflict(
            "requests_in_flight",
            "levers change only while no request is in flight",
        ),
        Ok(Err(LeverSwapError::NotLoaded)) => {
            conflict("native_not_loaded", "no native backend is loaded")
        }
        Ok(Err(LeverSwapError::Executor(message))) => api_error(&ApiError::new(
            500,
            SERVER_ERROR,
            "executor_error",
            message,
            None,
        )),
        Err(join) => api_error(&ApiError::new(
            500,
            SERVER_ERROR,
            "executor_error",
            format!("the lever swap failed: {join}"),
            None,
        )),
    }
}

/// The `POST /v1/cere/levers` body: `{"levers": {lever: impl, ...}}`, optionally with `"params": {lever: {param:
/// unsigned integer, ...}, ...}` (SelfTune D6, O292 D-9); any other key is refused.
type LeverChanges = (
    BTreeMap<String, String>,
    BTreeMap<String, BTreeMap<String, u64>>,
);

fn lever_changes(body: &[u8]) -> Result<LeverChanges, String> {
    const SHAPE: &str =
        "the body must be {\"levers\": {lever: impl, ...}} with an optional \"params\": {lever: {param: value, ...}}";
    let value: Value =
        serde_json::from_slice(body).map_err(|e| format!("the body is not JSON: {e}"))?;
    let body = value
        .as_object()
        .filter(|o| o.keys().all(|k| k == "levers" || k == "params"))
        .ok_or(SHAPE)?;
    let fields = body.get("levers").and_then(Value::as_object).ok_or(SHAPE)?;
    let changes = fields
        .iter()
        .map(|(name, id)| match id {
            Value::String(id) => Ok((name.clone(), id.clone())),
            _ => Err(format!("{name}: the impl id must be a string")),
        })
        .collect::<Result<_, String>>()?;
    let mut params = BTreeMap::new();
    if let Some(levers) = body.get("params") {
        for (name, values) in levers.as_object().ok_or(SHAPE)? {
            let values = values
                .as_object()
                .ok_or_else(|| format!("{name}: params must be an object of param -> value"))?
                .iter()
                .map(|(param, v)| {
                    v.as_u64().map(|v| (param.clone(), v)).ok_or_else(|| {
                        format!("{name}.{param}: the value must be an unsigned integer")
                    })
                })
                .collect::<Result<BTreeMap<_, _>, String>>()?;
            params.insert(name.clone(), values);
        }
    }
    Ok((changes, params))
}

fn invalid_lever(message: &str) -> Response {
    api_error(&ApiError::new(
        400,
        INVALID_REQUEST,
        "invalid_lever",
        message,
        Some("levers".to_owned()),
    ))
}

fn conflict(code: &'static str, message: &str) -> Response {
    api_error(&ApiError::new(409, INVALID_REQUEST, code, message, None))
}

fn not_found(code: &'static str, message: &str) -> Response {
    api_error(&ApiError::new(404, INVALID_REQUEST, code, message, None))
}

fn forbidden() -> Response {
    api_error(&ApiError::new(
        403,
        INVALID_REQUEST,
        "forbidden",
        "the operator routes need the operator token",
        None,
    ))
}
