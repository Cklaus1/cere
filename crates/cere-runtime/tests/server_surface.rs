//! M2.14 plumbing: a compat surface (here a stub) served through `server::serve_compat` and the chat pipeline:
//! surface-rendered unary bodies (also on a keyed replay), errors in the surface envelope, a rendered stream
//! without CERE frames or `[DONE]`, the per-deployment flag, and a compat stream's refused resume.

mod support;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use cere_mock_sglang::MockConfig;
use cere_runtime::app::App;
use cere_runtime::server::serve_compat;
use cere_runtime::surface::{
    CompatApi, CompatKind, ConvertCtx, ErrorView, OwnedError, SurfaceRender, SurfaceStream,
};
use serde_json::{json, Value};
use support::*;

const REQUEST_ID: &str = cere_runtime::server::REQUEST_ID_HEADER;

struct Stub;

fn stub_error(e: &ErrorView) -> Vec<u8> {
    json!({"stub_error": {"status": e.status, "type": e.kind, "code": e.code}})
        .to_string()
        .into_bytes()
}

impl SurfaceRender for Stub {
    fn name(&self) -> &'static str {
        "anthropic"
    }
    fn render_unary(&self, chat: &Value, rid: &str) -> Vec<u8> {
        let text = &chat["choices"][0]["message"]["content"];
        json!({"stub": rid, "text": text}).to_string().into_bytes()
    }
    fn error_body(&self, e: &ErrorView) -> Vec<u8> {
        stub_error(e)
    }
    fn stream(&self) -> Box<dyn SurfaceStream> {
        Box::new(StubStream)
    }
}

struct StubStream;

impl SurfaceStream for StubStream {
    fn open(&mut self, _prompt_tokens: Option<u64>) -> Vec<u8> {
        b"event: open\ndata: {}\n\n".to_vec()
    }
    fn chunk(&mut self, chunk: &Value) -> Vec<u8> {
        assert!(chunk.get("cere").is_none(), "{chunk}");
        format!("event: chunk\ndata: {}\n\n", chunk["choices"][0]["delta"]).into_bytes()
    }
    fn complete(&mut self, finish: &Value, usage: Option<&Value>) -> Vec<u8> {
        let usage = usage.cloned().unwrap_or(Value::Null);
        let reason = &finish["choices"][0]["finish_reason"];
        format!(
            "event: done\ndata: {}\n\n",
            json!({"reason": reason, "usage": usage})
        )
        .into_bytes()
    }
    fn fail(&mut self, e: &ErrorView) -> Vec<u8> {
        format!("event: fail\ndata: {}\n\n", e.code).into_bytes()
    }
}

/// Passes the body through unchanged unless it says `"refuse": true`.
fn convert(bytes: &[u8], _: &ConvertCtx) -> Result<(Vec<u8>, Arc<dyn SurfaceRender>), OwnedError> {
    let value: Value = serde_json::from_slice(bytes).unwrap();
    if value.get("refuse").is_some() {
        return Err(OwnedError {
            status: 400,
            kind: "invalid_request_error".into(),
            code: "unsupported_parameter".into(),
            message: "refused".into(),
            param: Some("refuse".into()),
        });
    }
    Ok((bytes.to_vec(), Arc::new(Stub)))
}

static STUB: CompatApi = CompatApi {
    kind: CompatKind::Anthropic,
    error_body: stub_error,
    convert,
};

async fn handler(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    serve_compat(&STUB, app, headers, body).await
}

const FLAG: (&str, &str) = (
    "allow_cross_scope_prefix_cache = true",
    "allow_cross_scope_prefix_cache = true\napi_compat_anthropic = true",
);

/// The harness plus the stub surface served on its own listener; returns its URL.
async fn start(tweaks: &[(&str, &str)]) -> (Harness, String) {
    let h = Harness::start(MockConfig::default(), tweaks).await;
    let router = Router::new()
        .route("/stub", post(handler))
        .with_state(h.app.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/stub", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await });
    (h, url)
}

fn stub(h: &Harness, url: &str, body: &Value) -> reqwest::RequestBuilder {
    h.http.post(url).bearer_auth(TOKEN_A).json(body)
}

#[tokio::test(flavor = "multi_thread")]
async fn unary_is_surface_rendered_and_replayed_as_rendered() {
    let (h, url) = start(&[FLAG]).await;
    let send = || stub(&h, &url, &chat_body(false)).header("idempotency-key", "k1");
    let r = send().send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()[REQUEST_ID].to_str().unwrap().to_owned();
    let first: Value = r.json().await.unwrap();
    assert_eq!(first["stub"], rid.as_str(), "{first}");
    assert!(first.get("cere").is_none(), "{first}");
    let replay = send().send().await.unwrap();
    assert_eq!(replay.status(), 200);
    assert_eq!(replay.json::<Value>().await.unwrap(), first);
    // The same key on the chat route is another surface's request.
    let chat = h
        .chat(&chat_body(false))
        .header("idempotency-key", "k1")
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status(), 409);
    let body: Value = chat.json().await.unwrap();
    assert_eq!(body["error"]["code"], "idempotency_conflict", "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_are_in_the_surface_envelope() {
    let (h, url) = start(&[FLAG]).await;
    let mut refused = chat_body(false);
    refused["refuse"] = json!(true);
    let r = stub(&h, &url, &refused).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let body: Value = r.json().await.unwrap();
    assert_eq!(
        body["stub_error"]["code"], "unsupported_parameter",
        "{body}"
    );
    // A chat validation error, rendered by the request's renderer.
    let mut invalid = chat_body(false);
    invalid["max_completion_tokens"] = json!(-1);
    let r = stub(&h, &url, &invalid).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let body: Value = r.json().await.unwrap();
    assert_eq!(
        body["stub_error"]["type"], "invalid_request_error",
        "{body}"
    );
    // Authentication, before any conversion.
    let r = h
        .http
        .post(&url)
        .json(&chat_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["stub_error"]["code"], "unauthorized", "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_deployment_without_the_flag_is_not_found_in_the_surface_envelope() {
    let (h, url) = start(&[]).await;
    let r = stub(&h, &url, &chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["stub_error"]["code"], "not_found", "{body}");
    assert!(!cere_runtime::server::compat_enabled(
        &h.app,
        CompatKind::Anthropic
    ));
    assert!(!cere_runtime::server::compat_enabled(
        &h.app,
        CompatKind::Responses
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn stream_is_surface_rendered_and_never_resumable() {
    let (h, url) = start(&[FLAG]).await;
    let r = stub(&h, &url, &chat_body(true))
        .header("idempotency-key", "s1")
        .header("x-cere-on-disconnect", "continue")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()[REQUEST_ID].to_str().unwrap().to_owned();
    let text = r.text().await.unwrap();
    assert!(text.starts_with("event: open\n"), "{text}");
    assert!(text.contains("event: chunk\n"), "{text}");
    assert!(
        text.trim_end().lines().any(|l| l == "event: done"),
        "{text}"
    );
    assert!(
        !text.contains("[DONE]") && !text.contains("\"cere\""),
        "{text}"
    );
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "completed", "{status}");
    let events = h
        .get(&format!("/v1/cere/requests/{rid}/events"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(events.status(), 409);
    let body: Value = events.json().await.unwrap();
    assert_eq!(body["error"]["code"], "stream_not_replayable", "{body}");
    let retry = stub(&h, &url, &chat_body(true))
        .header("idempotency-key", "s1")
        .header("x-cere-on-disconnect", "continue")
        .send()
        .await
        .unwrap();
    assert_eq!(retry.status(), 409);
    let body: Value = retry.json().await.unwrap();
    assert_eq!(
        body["stub_error"]["code"], "stream_not_replayable",
        "{body}"
    );
}
