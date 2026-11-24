//! NetTeleport S1 (net-teleport design note 4.2 L2, 4.3): `POST /v1/cere/state/export` and
//! `POST /v1/cere/state/import` on one native `cered` (client bearer auth; docs/API.md "State teleport").
//!
//! Export seals the deepest cached prefix-cache state of the caller scope's namespace for a token prefix (or a chat
//! body rendered by the deployment's template) and streams the blob. Import opens a blob frame by frame into one
//! locked buffer and inserts it into the caller scope's own namespace (D-T2). Each refusal is a named code and
//! imports nothing; the client then replays (D-T1). One export or import runs at a time per process, so at most one
//! image buffer (`state.max_blob_bytes`) is locked beside the arena. Log lines carry the scope, sizes, timings and
//! the outcome code; never the key, tokens or content (4.1).

use std::sync::{Arc, LazyLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::teleport::{
    admit, tier_list, HeaderReader, Local, Opener, Refusal, SealMeta, Sealer, StateKey, KEY_HEADER,
};
use crate::api_types::{ApiError, SERVER_ERROR};
use crate::app::{App, NativeBackend};
use crate::digest::hex;
use crate::server::{api_error, json_response};
use crate::tier::{allows, Tier, DEFAULT_ALLOWED_TIERS};

/// The media type of a state blob.
pub const BLOB_CONTENT_TYPE: &str = "application/vnd.cere.state-blob";
/// Blob chunks buffered between the sealing (or opening) thread and the HTTP body.
const CHUNKS_IN_FLIGHT: usize = 4;

/// The process-wide one-at-a-time gate of exports and imports.
static GATE: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(1)));

async fn acquire(deadline: Instant) -> Result<OwnedSemaphorePermit, Refusal> {
    tokio::time::timeout_at(deadline.into(), GATE.clone().acquire_owned())
        .await
        .map_err(|_| Refusal::Busy)?
        .map_err(|_| Refusal::Internal("the state gate closed".to_owned()))
}

fn refusal(r: Refusal) -> ApiError {
    r.api_error()
}

fn log(op: &str, scope: &str, started: Instant, outcome: &str, tokens: u64, bytes: u64) {
    eprintln!(
        "cered state {op} scope={scope} outcome={outcome} tokens={tokens} image_bytes={bytes} ms={}",
        started.elapsed().as_millis()
    );
}

fn key(headers: &HeaderMap) -> Result<StateKey, ApiError> {
    StateKey::from_header(headers.get(KEY_HEADER).map(HeaderValue::as_bytes)).map_err(refusal)
}

fn scope_tiers(app: &App, scope: &str) -> Vec<Tier> {
    app.settings()
        .config
        .auth
        .scopes
        .iter()
        .find(|s| s.scope_id == scope)
        .map_or(DEFAULT_ALLOWED_TIERS, |s| s.allowed_tiers())
        .to_vec()
}

/// The deployment `model` names, its one native backend and the scope's prefix-cache namespace on it.
fn target(
    app: &App,
    scope: &str,
    model: Option<&str>,
) -> Result<(usize, Arc<NativeBackend>, [u8; 16]), ApiError> {
    let model = model.ok_or_else(|| ApiError::missing_parameter("model"))?;
    let deployment = app.deployment_of_model(model).ok_or_else(|| {
        crate::validate::model_not_found(model, &app.settings().policies[0].served_models)
    })?;
    if app.routes_by_pool(deployment) {
        return Err(refusal(Refusal::Unsupported(
            "a multi-replica deployment (state goes to one card)".to_owned(),
        )));
    }
    let backend = app.live_native(deployment).ok_or_else(|| {
        if app.settings().config.deployments[deployment]
            .native
            .is_some()
        {
            ApiError::new(
                503,
                SERVER_ERROR,
                "not_ready",
                format!("backend not ready: no worker of {model} is ready"),
                None,
            )
        } else {
            refusal(Refusal::Unsupported(
                "not a native_cuda deployment".to_owned(),
            ))
        }
    })?;
    let salt = app.cache_salts.get(scope).map(String::as_str);
    let namespace = super::serve::cache_namespace(&backend, salt)?.ok_or_else(|| {
        refusal(Refusal::Unsupported(
            "the executor runs no prefix cache".to_owned(),
        ))
    })?;
    Ok((deployment, backend, namespace))
}

fn deadline(app: &App) -> Instant {
    Instant::now() + std::time::Duration::from_millis(app.settings().config.state.busy_wait_ms())
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The prompt ids of an export body: `tokens`, or the chat body (every other field) rendered as a chat request.
async fn export_ids(
    app: &App,
    backend: &Arc<NativeBackend>,
    deployment: usize,
    mut value: Value,
    bytes: &[u8],
) -> Result<Vec<u32>, ApiError> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| ApiError::invalid_json("the body must be a JSON object"))?;
    if let Some(tokens) = object.remove("tokens") {
        if object.contains_key("messages") {
            return Err(ApiError::invalid_value(
                "tokens",
                "give tokens or messages, not both",
            ));
        }
        let ids: Vec<u32> = serde_json::from_value(tokens)
            .map_err(|_| ApiError::invalid_value("tokens", "tokens must be token ids (u32)"))?;
        if ids.is_empty() {
            return Err(ApiError::invalid_value(
                "tokens",
                "tokens must not be empty",
            ));
        }
        return Ok(ids);
    }
    let request = app.settings().policies[deployment].validate_bytes(&value, bytes)?;
    super::serve::prompt_ids(backend, &request)
        .await?
        .ok_or_else(|| refusal(Refusal::Unsupported("a request with images".to_owned())))
}

/// `POST /v1/cere/state/export` for an authenticated `scope`.
pub(crate) async fn export(
    app: Arc<App>,
    scope: String,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let started = Instant::now();
    match export_blob(&app, &scope, &headers, body).await {
        Ok((response, tokens, bytes)) => {
            log("export", &scope, started, "ok", tokens, bytes);
            response
        }
        Err(e) => {
            log("export", &scope, started, e.code, 0, 0);
            api_error(&e)
        }
    }
}

async fn export_blob(
    app: &Arc<App>,
    scope: &str,
    headers: &HeaderMap,
    body: Body,
) -> Result<(Response, u64, u64), ApiError> {
    let key = key(headers)?;
    let limit = app.settings().config.api.max_request_body_bytes as usize;
    let body = axum::body::to_bytes(body, limit)
        .await
        .map_err(|_| ApiError::invalid_json(format!("request body exceeds {limit} bytes")))?;
    let mut value: Value =
        serde_json::from_slice(&body).map_err(|e| ApiError::invalid_json(e.to_string()))?;
    let destination = value
        .as_object_mut()
        .and_then(|o| o.remove("destination_tier"))
        .ok_or_else(|| ApiError::missing_parameter("destination_tier"))?;
    let destination: Tier = serde_json::from_value(destination)
        .map_err(|_| ApiError::invalid_value("destination_tier", "one of own, team, provider"))?;
    let allowed = scope_tiers(app, scope);
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let (deployment, backend, namespace) = target(app, scope, model.as_deref())?;
    // D-T7: the source enforces the scope's tier policy before any state is read.
    if !allows(&allowed, destination) {
        return Err(refusal(Refusal::Tier(format!(
            "destination tier {} is not in this scope's allowed tiers [{}]",
            destination.as_str(),
            tier_list(&allowed)
        ))));
    }
    let ids = export_ids(app, &backend, deployment, value, &body).await?;
    let settings = app.settings();
    let max = settings.config.state.max_blob_bytes();
    let source_tier = settings.config.server.tier();
    let deadline = deadline(app);
    let permit = acquire(deadline).await.map_err(refusal)?;
    let worker = backend.clone();
    let exported = tokio::task::spawn_blocking(move || {
        worker.export_prefix_state(namespace, &ids, max, deadline)
    })
    .await
    .map_err(|e| refusal(Refusal::Internal(e.to_string())))?
    .map_err(refusal)?;
    let (image, tokens) = exported;
    let image_bytes = image.len() as u64;
    let meta = SealMeta {
        library_sha256: backend.identity.library_sha256.clone(),
        profile_id: backend.identity.profile_id.clone(),
        layout_id: backend.identity.layout_id,
        compat_id: backend.state_compat_id(),
        tokens,
        source_tier,
        allowed_tiers: allowed,
        created_unix_ms: unix_ms(),
    };
    let mut sealer = Sealer::new(&key, meta, image).map_err(refusal)?;
    drop(key);
    let blob_bytes = sealer.blob_bytes();
    let compat = sealer.header().state_compat_id.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(CHUNKS_IN_FLIGHT);
    // The sealer (and its locked image) and the gate permit live until the last frame left or the client went.
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        while let Some(chunk) = sealer.next_chunk() {
            if tx.blocking_send(Ok(Bytes::from(chunk))).is_err() {
                break;
            }
        }
    });
    let response = (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, BLOB_CONTENT_TYPE.to_owned()),
            (header::CONTENT_LENGTH, blob_bytes.to_string()),
            (
                header::HeaderName::from_static("x-cere-state-tokens"),
                tokens.to_string(),
            ),
            (
                header::HeaderName::from_static("x-cere-state-image-bytes"),
                image_bytes.to_string(),
            ),
            (
                header::HeaderName::from_static("x-cere-state-compat-id"),
                compat,
            ),
        ],
        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
    )
        .into_response();
    Ok((response, tokens, image_bytes))
}

/// `POST /v1/cere/state/import?model=<model>` for an authenticated `scope`.
pub(crate) async fn import(
    app: Arc<App>,
    scope: String,
    headers: HeaderMap,
    query: Option<String>,
    body: Body,
) -> Response {
    let started = Instant::now();
    match import_blob(&app, &scope, &headers, query.as_deref(), body).await {
        Ok((tokens, bytes, response)) => {
            log("import", &scope, started, "ok", tokens, bytes);
            response
        }
        Err(e) => {
            log("import", &scope, started, e.code, 0, 0);
            api_error(&e)
        }
    }
}

fn query_model(query: Option<&str>) -> Option<String> {
    query?.split('&').find_map(|pair| {
        let value = pair.strip_prefix("model=")?;
        // Model ids are plain; decode `%XX` and `+` for clients that escape anyway.
        let mut out = Vec::with_capacity(value.len());
        let mut bytes = value.bytes();
        while let Some(b) = bytes.next() {
            match b {
                b'+' => out.push(b' '),
                b'%' => {
                    let hi = (bytes.next()? as char).to_digit(16)?;
                    let lo = (bytes.next()? as char).to_digit(16)?;
                    out.push((hi * 16 + lo) as u8);
                }
                _ => out.push(b),
            }
        }
        String::from_utf8(out).ok()
    })
}

async fn import_blob(
    app: &Arc<App>,
    scope: &str,
    headers: &HeaderMap,
    query: Option<&str>,
    body: Body,
) -> Result<(u64, u64, Response), ApiError> {
    let key = key(headers)?;
    let model = query_model(query);
    let (_, backend, namespace) = target(app, scope, model.as_deref())?;
    let settings = app.settings();
    let mut stream = body.into_data_stream();
    let mut reader = HeaderReader::default();
    let parsed = loop {
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(refusal(Refusal::Malformed(format!("body: {e}")))),
            None => return Err(refusal(Refusal::Malformed("truncated header".to_owned()))),
        };
        if let Some(parsed) = reader.push(&chunk).map_err(refusal)? {
            break parsed;
        }
    };
    // Size, library, layout, compat id and tier (defence in depth, D-T7) before any key use or allocation.
    admit(
        &parsed.header,
        &Local {
            identity: &backend.identity,
            compat_id: backend.state_compat_id(),
            tier: settings.config.server.tier(),
            max_blob_bytes: settings.config.state.max_blob_bytes(),
        },
    )
    .map_err(refusal)?;
    let image_bytes = parsed.header.image_bytes;
    let busy_wait = std::time::Duration::from_millis(settings.config.state.busy_wait_ms());
    let permit = acquire(Instant::now() + busy_wait).await.map_err(refusal)?;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Bytes>(CHUNKS_IN_FLIGHT);
    let worker = backend.clone();
    // The opening thread: frames are decrypted off the async workers into one locked buffer, then imported.
    let opened = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let (mut opener, rest) = Opener::new(&key, parsed)?;
        drop(key);
        opener.push(&rest)?;
        while let Some(chunk) = rx.blocking_recv() {
            opener.push(&chunk)?;
        }
        let image = opener.finish()?;
        // The idle wait starts once the image is opened, so a long upload does not use it up.
        worker.import_prefix_state(namespace, &image, Instant::now() + busy_wait)
    });
    let mut body_error = None;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                if tx.send(chunk).await.is_err() {
                    // The opening thread refused; its answer follows.
                    break;
                }
            }
            Err(e) => {
                body_error = Some(e.to_string());
                break;
            }
        }
    }
    drop(tx);
    let opened = opened
        .await
        .map_err(|e| refusal(Refusal::Internal(e.to_string())))?;
    if let Some(e) = body_error {
        return Err(refusal(Refusal::Malformed(format!("body: {e}"))));
    }
    let tokens = opened.map_err(refusal)?;
    let response = json_response(
        StatusCode::OK,
        &json!({
            "imported_tokens": tokens,
            "image_bytes": image_bytes,
            "state_compat_id": hex(&backend.state_compat_id()),
            "tier": settings.config.server.tier(),
        }),
    );
    Ok((tokens, image_bytes, response))
}

#[cfg(test)]
mod tests {
    use super::query_model;

    #[test]
    fn query_model_decodes() {
        assert_eq!(query_model(Some("model=a%2Fb+c")).as_deref(), Some("a/b c"));
        assert_eq!(query_model(Some("x=1&model=m")).as_deref(), Some("m"));
        assert_eq!(query_model(Some("x=1")), None);
        assert_eq!(query_model(None), None);
        assert_eq!(query_model(Some("model=%zz")), None);
    }
}
