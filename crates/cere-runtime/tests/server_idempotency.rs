//! L2 idempotency outcomes over HTTP: replay, conflict, in progress, stream not replayable, response not
//! retained after a restart, expired key.
mod support;

use std::time::Duration;

use cere_mock_sglang::MockConfig;
use serde_json::Value;
use support::*;

async fn keyed(h: &Harness, body: &Value, key: &str) -> reqwest::Response {
    h.chat(body)
        .header("Idempotency-Key", key)
        .send()
        .await
        .unwrap()
}

async fn conflict(response: reqwest::Response, code: &str) -> Value {
    assert_eq!(response.status(), 409);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], code, "{body}");
    body
}

#[tokio::test]
async fn l2_replay_and_conflict() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let first = keyed(&h, &chat_body(false), "k-1").await;
    assert_eq!(first.status(), 200);
    let rid = first.headers()["x-cere-request-id"].clone();
    let first = first.bytes().await.unwrap();
    let dispatched = h.mock.dispatched_rids().len();

    let replay = keyed(&h, &chat_body(false), "k-1").await;
    assert_eq!(replay.status(), 200);
    assert_eq!(replay.headers()["x-cere-request-id"], rid);
    assert_eq!(
        replay.bytes().await.unwrap(),
        first,
        "byte-identical replay"
    );
    assert_eq!(
        h.mock.dispatched_rids().len(),
        dispatched,
        "a replay never executes"
    );

    let mut other = chat_body(false);
    other["max_completion_tokens"] = 4.into();
    conflict(keyed(&h, &other, "k-1").await, "idempotency_conflict").await;
    // The key is per scope: scope b may use it for a different request.
    let b = h
        .post("/v1/chat/completions", TOKEN_B)
        .header("Idempotency-Key", "k-1")
        .json(&other)
        .send()
        .await
        .unwrap();
    assert_eq!(b.status(), 200);
}

#[tokio::test]
async fn l2_in_progress_and_stream_not_replayable() {
    let h = Harness::start(
        MockConfig {
            token_latency: Duration::from_millis(60),
            ..MockConfig::default()
        },
        &[],
    )
    .await;
    let body = chat_body(false);
    let running = {
        let request = h.chat(&body).header("Idempotency-Key", "slow");
        tokio::spawn(async move { request.send().await.unwrap() })
    };
    // The first request streams 8 tokens at 60 ms each; probe while it runs.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let body409 = conflict(keyed(&h, &body, "slow").await, "request_in_progress").await;
    let rid = body409["error"]["status_url"]
        .as_str()
        .unwrap()
        .strip_prefix("/v1/cere/requests/")
        .unwrap()
        .to_owned();
    let done = running.await.unwrap();
    assert_eq!(done.status(), 200);
    assert_eq!(done.headers()["x-cere-request-id"], rid.as_str());

    let stream = chat_body(true);
    let first = keyed(&h, &stream, "s-1").await;
    assert_eq!(first.status(), 200);
    SseReader::new(first).rest().await;
    let again = conflict(keyed(&h, &stream, "s-1").await, "stream_not_replayable").await;
    assert!(again["error"]["status_url"]
        .as_str()
        .unwrap()
        .starts_with("/v1/cere/requests/"));
}

#[tokio::test]
async fn l2_response_not_retained_after_restart() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let first = keyed(&h, &chat_body(false), "r-1").await;
    assert_eq!(first.status(), 200);
    let rid = first.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let epoch = h.app.epoch();
    let h = h.restart().await;
    assert_eq!(h.app.epoch(), epoch + 1);
    let dispatched = h.mock.dispatched_rids().len();
    let body = conflict(
        keyed(&h, &chat_body(false), "r-1").await,
        "response_not_retained",
    )
    .await;
    assert_eq!(
        body["error"]["status_url"],
        format!("/v1/cere/requests/{rid}")
    );
    assert_eq!(h.mock.dispatched_rids().len(), dispatched);
    let (code, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(code, 200);
    assert_eq!(status["state"], "completed");
}

#[tokio::test]
async fn l2_expired_key_is_never_regenerated() {
    let h = Harness::start(
        MockConfig::default(),
        &[(
            "terminal_retention_ms = 3600000",
            "terminal_retention_ms = 1000",
        )],
    )
    .await;
    assert_eq!(keyed(&h, &chat_body(false), "e-1").await.status(), 200);
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let dispatched = h.mock.dispatched_rids().len();
    conflict(
        keyed(&h, &chat_body(false), "e-1").await,
        "idempotency_key_expired",
    )
    .await;
    assert_eq!(h.mock.dispatched_rids().len(), dispatched);
}
