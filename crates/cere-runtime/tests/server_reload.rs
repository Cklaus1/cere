//! Live config reload on a running `cered` (`POST /v1/cere/config/reload`, `crate::reload`): live keys take effect
//! without a restart, restart-class keys refuse the whole reload with nothing applied, requests in flight keep the
//! settings they were accepted under, and `/v1/cere/stats` reports the generation and the effective config hash.
mod support;

use std::time::Duration;

use cere_mock_sglang::MockConfig;
use serde_json::{json, Value};
use support::*;

/// One backend slot, one queued waiter at most, and the operator routes (the reload route needs the token).
const TIGHT: [(&str, &str); 4] = [
    ("max_backend_inflight = 2", "max_backend_inflight = 1"),
    ("per_scope_inflight = 2", "per_scope_inflight = 1"),
    ("queue_depth = 64", "queue_depth = 1"),
    OPERATOR_TWEAK,
];

fn slow() -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(40),
        natural_output_tokens: 8,
        ..MockConfig::default()
    }
}

/// `text` with each `from` replaced by its `to`; every `from` must occur.
fn edited(text: &str, edits: &[(&str, &str)]) -> String {
    let mut text = text.to_owned();
    for (from, to) in edits {
        assert!(text.contains(from), "config has no {from:?}");
        text = text.replacen(from, to, 1);
    }
    text
}

/// A unary request in the background; resolves to (status, body).
fn spawn_chat(h: &Harness, body: Value) -> tokio::task::JoinHandle<(u16, Value)> {
    let request = h.chat(&body);
    tokio::spawn(async move {
        let r = request.send().await.unwrap();
        (r.status().as_u16(), r.json().await.unwrap())
    })
}

async fn wait_inflight(h: &Harness, inflight: u64, queued: u64) {
    for _ in 0..400 {
        let s = h.app.admission_snapshot();
        if s.inflight == inflight && s.queued == queued {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "admission never reached inflight={inflight} queued={queued}: {:?}",
        h.app.admission_snapshot()
    );
}

/// The reload refused with `code`, every name of `keys` in its message; nothing applied (generation 0, the
/// effective hash and the admission settings still the startup ones).
async fn assert_refused(h: &Harness, code: &str, keys: &[&str]) {
    let before = h.stats().await;
    let (status, body) = h.reload().await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], code, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    for key in keys {
        assert!(message.contains(key), "{key} not named: {message}");
    }
    let after = h.stats().await;
    assert_eq!(after["config_generation"], 0, "{after}");
    assert_eq!(
        after["effective_config_hash"],
        before["effective_config_hash"]
    );
    let settings = h.app.settings();
    assert_eq!(settings.generation, 0);
    assert_eq!(
        settings.config.admission.queue_depth, 1,
        "a live key of a refused reload was applied"
    );
}

#[tokio::test]
async fn reload_without_change_is_a_noop() {
    let h = Harness::start(MockConfig::default(), &TIGHT).await;
    let stats = h.stats().await;
    assert_eq!(stats["config_generation"], 0);
    let startup = h.app.deployment(0).config_hash.clone();
    assert_eq!(stats["effective_config_hash"], startup.as_str());
    // A comment is no change.
    h.write_config(&format!("# reloaded\n{}", h.started_config()));
    let (status, body) = h.reload().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body,
        json!({"applied": false, "generation": 0, "config_hash": startup, "changed": []})
    );
    assert_eq!(h.stats().await["config_generation"], 0);
}

#[tokio::test]
async fn reload_route_needs_the_operator_token() {
    let h = Harness::start(MockConfig::default(), &TIGHT).await;
    let client = h
        .post("/v1/cere/config/reload", TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(client.status(), 403);
    let anonymous = h
        .http
        .post(h.url("/v1/cere/config/reload"))
        .send()
        .await
        .unwrap();
    assert_eq!(anonymous.status(), 401);
    let disabled = Harness::start(MockConfig::default(), &[]).await;
    let r = disabled
        .post("/v1/cere/config/reload", OPERATOR_TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        404,
        "no operator token configured: the route does not exist"
    );
}

#[tokio::test]
async fn queue_depth_change_takes_effect_without_restart() {
    let h = Harness::start(slow(), &TIGHT).await;
    let first = spawn_chat(&h, chat_body(false));
    wait_inflight(&h, 1, 0).await;
    let second = spawn_chat(&h, chat_body(false));
    wait_inflight(&h, 1, 1).await;
    let full = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(full.status(), 429, "queue_depth = 1 is full");

    h.write_config(&edited(
        h.started_config(),
        &[("queue_depth = 1", "queue_depth = 2")],
    ));
    let (status, body) = h.reload().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["applied"], true);
    assert_eq!(body["generation"], 1);
    assert_eq!(body["changed"], json!(["admission.queue_depth"]));

    // The same load that was refused now queues and runs.
    let third = spawn_chat(&h, chat_body(false));
    wait_inflight(&h, 1, 2).await;
    for request in [first, second, third] {
        let (status, body) = request.await.unwrap();
        assert_eq!(status, 200, "{body}");
    }
}

#[tokio::test]
async fn raised_backend_cap_grants_the_queued_waiter_at_once() {
    let h = Harness::start(slow(), &TIGHT).await;
    let first = spawn_chat(&h, chat_body(false));
    wait_inflight(&h, 1, 0).await;
    let second = spawn_chat(&h, chat_body(false));
    wait_inflight(&h, 1, 1).await;
    h.write_config(&edited(
        h.started_config(),
        &[
            ("max_backend_inflight = 1", "max_backend_inflight = 2"),
            ("per_scope_inflight = 1", "per_scope_inflight = 2"),
        ],
    ));
    let (status, body) = h.reload().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["changed"],
        json!([
            "admission.max_backend_inflight",
            "admission.per_scope_inflight"
        ])
    );
    // The queued request is granted by the reload itself, while the first still runs.
    wait_inflight(&h, 2, 0).await;
    for request in [first, second] {
        assert_eq!(request.await.unwrap().0, 200);
    }
}

#[tokio::test]
async fn in_flight_request_finishes_on_its_settings_and_stats_bump() {
    let h = Harness::start(slow(), &TIGHT).await;
    let startup = h.app.deployment(0).config_hash.clone();
    // Accepted under max_output_tokens = 32768: asks for 8 tokens.
    let running = spawn_chat(&h, chat_body(false));
    wait_inflight(&h, 1, 0).await;
    h.write_config(&edited(
        h.started_config(),
        &[
            (
                "default_max_output_tokens = 4096",
                "default_max_output_tokens = 4",
            ),
            ("max_output_tokens = 32768", "max_output_tokens = 4"),
        ],
    ));
    let (status, body) = h.reload().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["generation"], 1);
    let effective = body["config_hash"].as_str().unwrap().to_owned();
    assert_ne!(effective, startup, "the effective config changed");

    let (status, done) = running.await.unwrap();
    assert_eq!(status, 200, "{done}");
    assert_eq!(done["usage"]["completion_tokens"], 8, "{done}");
    // The `cere` identity hash is the startup config's: a live key never moves it.
    assert_eq!(done["cere"]["config_hash"], startup.as_str());

    // New requests are validated under the new bound.
    let refused = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(refused.status(), 400);
    let mut small = chat_body(false);
    small["max_completion_tokens"] = json!(4);
    let r = h.chat(&small).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["cere"]["config_hash"], startup.as_str());

    let stats = h.stats().await;
    assert_eq!(stats["config_generation"], 1);
    assert_eq!(stats["effective_config_hash"], effective.as_str());
}

#[tokio::test]
async fn invalid_file_is_refused_with_nothing_applied() {
    let h = Harness::start(MockConfig::default(), &TIGHT).await;
    // A live key changed, but the file fails validation as a whole.
    h.write_config(&edited(
        h.started_config(),
        &[
            ("queue_depth = 1", "queue_depth = 2"),
            ("max_messages = 512", "max_messages = 0"),
        ],
    ));
    assert_refused(&h, "config_invalid", &["api.max_messages"]).await;
    h.write_config("schema_version = \"2\"\n[server\n");
    assert_refused(&h, "config_invalid", &[]).await;
}

#[tokio::test]
async fn restart_keys_refuse_the_whole_reload() {
    let h = Harness::start(MockConfig::default(), &TIGHT).await;
    let started = h.started_config().to_owned();
    // One edit per restart class; each is bundled with a live change that must not be applied either.
    let cases: [(&str, (&str, &str), &str); 7] = [
        (
            "bound socket",
            ("bind = \"127.0.0.1:18390\"", "bind = \"127.0.0.1:18391\""),
            "server.bind",
        ),
        (
            "memory sizing",
            (
                "[admission]",
                "resume_buffer_total_bytes = 134217728\n\n[admission]",
            ),
            "api.resume_buffer_total_bytes",
        ),
        (
            "ledger",
            ("max_records = 10000", "max_records = 20000"),
            "ledger.max_records",
        ),
        (
            "credentials",
            ("token_env = \"TOK_B\"", "token_env = \"TOK_C\""),
            "auth.scopes[1].token_env",
        ),
        (
            "worker registry",
            (
                "devices = [\"gpu1\"]",
                "devices = [\"gpu1\"]\nworker_lost_after_ms = 120000",
            ),
            "scheduler.worker_lost_after_ms",
        ),
        (
            "backend identity",
            ("max_running_requests = 4", "max_running_requests = 3"),
            "deployments[0].expect.max_running_requests",
        ),
        (
            "request body bound",
            ("max_request_body_bytes = ", "max_request_body_bytes = 1"),
            "api.max_request_body_bytes",
        ),
    ];
    for (class, edit, key) in cases {
        let text = edited(&started, &[("queue_depth = 1", "queue_depth = 2"), edit]);
        h.write_config(&text);
        let (status, body) = h.reload().await;
        assert_eq!(status, 409, "{class}: {body}");
        assert_eq!(body["error"]["code"], "restart_required", "{class}: {body}");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains(key), "{class}: {key} not named: {message}");
        assert!(
            !message.contains("queue_depth"),
            "{class}: a live key is not a refusal reason: {message}"
        );
        assert_eq!(h.app.settings().generation, 0, "{class}");
        assert_eq!(h.app.settings().config.admission.queue_depth, 1, "{class}");
    }
    assert_eq!(h.stats().await["config_generation"], 0);
}
