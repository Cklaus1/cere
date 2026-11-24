//! `cered` HTTP surface against the mock: readiness gating (A2) and the backend launch identity (m1close R5),
//! validation on the route (A3), the `cere` object on unary and streamed responses (A4, S6), clamp detection,
//! context overflow (m1close R4), auth (S5) and request status (L7).
mod image_support;
mod support;

use std::time::Duration;

use cere_mock_sglang::{Fault, MockConfig, ServerInfo};
use serde_json::{json, Value};
use support::*;

#[tokio::test]
async fn a2_drift_gates_chat_and_recovers() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let health = h.http.get(h.url("/healthz")).send().await.unwrap();
    assert_eq!(health.status(), 200);
    assert_eq!(
        health.json::<Value>().await.unwrap(),
        json!({"status": "alive"})
    );
    let ready = h.http.get(h.url("/readyz")).send().await.unwrap();
    assert_eq!(ready.status(), 200);
    let body: Value = ready.json().await.unwrap();
    assert_eq!(body["ready"], true);
    assert_eq!(body["stopping"], false);
    assert!(body["tenancy"].is_object());

    h.mock.set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    h.app.readiness(0).probe().await;
    let drift = h.http.get(h.url("/readyz")).send().await.unwrap();
    assert_eq!(drift.status(), 503);
    let body: Value = drift.json().await.unwrap();
    assert!(
        worker(&body)["problems"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["field"] == "kv_cache_dtype"),
        "{body}"
    );
    assert_eq!(body["problems"], json!([]), "{body}");
    assert_eq!(worker(&body)["state"], "suspect", "{body}");
    let dispatched = h.mock.dispatched_rids().len();
    let refused = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(refused.status(), 503);
    let err: Value = refused.json().await.unwrap();
    assert_eq!(err["error"]["code"], "not_ready");
    assert!(err["error"]["message"]
        .as_str()
        .unwrap()
        .contains("kv_cache_dtype"));
    assert_eq!(
        h.mock.dispatched_rids().len(),
        dispatched,
        "no backend dispatch while not ready"
    );

    h.mock.set_server_info_override(None);
    h.app.readiness(0).probe().await;
    assert_eq!(
        h.http.get(h.url("/readyz")).send().await.unwrap().status(),
        200
    );
    assert_eq!(
        h.chat(&chat_body(false)).send().await.unwrap().status(),
        200
    );
}

/// Item 10b: a terminal whose journal write fails poisons the journal and every later admission fails; /readyz
/// must say so (`journal_poisoned`) instead of staying green, and chat is refused as not ready.
#[tokio::test]
async fn a2_poisoned_journal_is_not_ready() {
    let h = Harness::start_tuned(MockConfig::default(), &[], short_ack(test_tuning())).await;
    // One request stays in flight (hung on the backend) while the journal fills up; its cancel ends at the short
    // ack bound (a hung request ignores the abort).
    let before = h.mock.dispatched_rids().len();
    h.mock.set_fault(Fault::HangBeforeResponse);
    let mut held = chat_body(false);
    held["max_completion_tokens"] = 4.into();
    let held_call = tokio::spawn(h.chat(&held).send());
    while h.mock.dispatched_rids().len() == before {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let held_rid = h.mock.dispatched_rids()[before].clone();
    h.mock.set_fault(Fault::None);
    // The compaction temp path is a directory, so the next compaction fails (also for root).
    std::fs::create_dir(h.state_dir.path().join("journal.jsonl.compact.tmp")).unwrap();
    // Fill the journal until a write's compaction fails: an admission (that request alone fails) or a terminal
    // (the journal is poisoned).
    let mut refused = None;
    for _ in 0..5_000 {
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        if r.status() != 200 {
            refused = Some(r.json::<Value>().await.unwrap());
            break;
        }
    }
    let refused = refused.expect("the journal never needed compaction");
    assert!(
        refused["error"]["message"]
            .as_str()
            .unwrap()
            .ends_with("journal_error"),
        "{refused}"
    );
    // The held request's terminal write fails in any case, which poisons the journal.
    let cancel = h
        .post(&format!("/v1/cere/requests/{held_rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    let cancel_status = cancel.status();
    assert!(
        cancel_status.is_success(),
        "{cancel_status} {}",
        cancel.text().await.unwrap()
    );
    held_call.await.unwrap().unwrap();

    let ready = h.http.get(h.url("/readyz")).send().await.unwrap();
    assert_eq!(ready.status(), 503);
    let body: Value = ready.json().await.unwrap();
    assert_eq!(body["ready"], false, "{body}");
    assert!(
        body["problems"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["field"] == "journal_poisoned"),
        "{body}"
    );
    let refused = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(refused.status(), 503);
    let err: Value = refused.json().await.unwrap();
    assert_eq!(err["error"]["code"], "not_ready", "{err}");
    // A probe does not clear it: only a restart recovers from disk.
    h.app.readiness(0).probe().await;
    assert_eq!(
        h.http.get(h.url("/readyz")).send().await.unwrap().status(),
        503
    );
}

/// Opportunity O7: a stop after a lost terminal write is not clean, even when every request ended in memory and
/// no backend stream is open (the journal on disk lacks the terminal).
#[tokio::test]
async fn l8_stop_after_a_poisoned_journal_is_not_clean() {
    let h = Harness::start(
        MockConfig {
            token_latency: std::time::Duration::from_millis(5),
            natural_output_tokens: 1_000_000,
            ..MockConfig::default()
        },
        &[],
    )
    .await;
    // One long generation stays in flight; unlike a hang, it honours the abort, so its backend stream closes.
    let before = h.mock.dispatched_rids().len();
    let mut held = chat_body(false);
    held["max_completion_tokens"] = 30_000.into();
    let held_call = tokio::spawn(h.chat(&held).send());
    while h.mock.dispatched_rids().len() == before {
        assert!(
            !held_call.is_finished(),
            "the held request was not dispatched"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let held_rid = h.mock.dispatched_rids()[before].clone();
    std::fs::create_dir(h.state_dir.path().join("journal.jsonl.compact.tmp")).unwrap();
    let mut short = chat_body(false);
    short["max_completion_tokens"] = 1.into();
    let mut refused = false;
    for _ in 0..5_000 {
        if h.chat(&short).send().await.unwrap().status() != 200 {
            refused = true;
            break;
        }
    }
    assert!(refused, "the journal never needed compaction");
    let cancel = h
        .post(&format!("/v1/cere/requests/{held_rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert!(cancel.status().is_success());
    held_call.await.unwrap().unwrap();
    assert_eq!(h.mock.in_flight(), 0, "the abort closed the backend stream");
    assert!(
        !h.app.shutdown().await,
        "shutdown reported clean on a poisoned journal"
    );
}

#[tokio::test]
async fn a4_unary_carries_the_cere_object() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let response = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let body: Value = response.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "tok0 tok1 tok2 tok3 tok4 tok5 tok6 tok7"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert!(body["choices"][0]["message"]
        .get("reasoning_content")
        .is_some());
    assert_eq!(body["usage"]["completion_tokens"], 8);
    let cere = &body["cere"];
    assert_identity(cere, &h.app, &rid);
    assert_eq!(cere["state"], "completed");
    assert_eq!(cere["cached_tokens"], Value::Null);
    assert_eq!(cere["cache_report"], "unavailable");
    assert!(cere.get("session_id").is_none());
    assert_eq!(h.terminal(&rid).await["state"], "completed");
}

/// m1close R4/Q9: a unary request rejected with SGLang's scheduler text ("... exceeds the maximum allowed
/// length", HTTP 400) is a client 400 `context_overflow`, recorded like the tokenizer-form overflow.
#[tokio::test]
async fn r4_unary_max_allowed_length_is_context_overflow() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let overflow = |body: &Value| {
        let request = h.chat(body);
        async move {
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), 400);
            let rid = response.headers()["x-cere-request-id"]
                .to_str()
                .unwrap()
                .to_owned();
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["error"]["code"], "context_overflow", "{body}");
            assert_eq!(body["cere"]["state"], "failed", "{body}");
            assert_eq!(body["cere"]["reason"], "context_overflow", "{body}");
            (rid, body)
        }
    };
    h.mock.set_fault(Fault::StreamError {
        code: 400,
        message: "Input length (199994 tokens) exceeds the maximum allowed length (199994 tokens)."
            .to_owned(),
    });
    let (rid, body) = overflow(&chat_body(false)).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("exceeds the maximum allowed length"),
        "{body}"
    );
    let record = h.terminal(&rid).await;
    assert_eq!(
        (&record["state"], &record["reason"]),
        (&json!("failed"), &json!("context_overflow")),
        "{record}"
    );

    // The tokenizer form: prompt + budget over the mock's 200000-token context.
    h.mock.set_fault(Fault::None);
    let mut long = chat_body(false);
    long["messages"][0]["content"] = "w ".repeat(200_000).into();
    let (early, _) = overflow(&long).await;
    let early = h.terminal(&early).await;
    assert_eq!(
        (&early["state"], &early["reason"]),
        (&record["state"], &record["reason"]),
        "{early}"
    );
}

#[tokio::test]
async fn a4_stream_carries_the_cere_object() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let mut body = chat_body(true);
    body["stream_options"] = json!({"include_usage": true});
    let response = h.chat(&body).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let events = SseReader::new(response).rest().await;
    let finish = check_stream(&events);

    let accepted = events[0].data();
    assert_eq!(accepted["id"], rid.as_str());
    assert_eq!(accepted["object"], "chat.completion.chunk");
    assert_eq!(accepted["choices"], json!([]));
    assert_identity(&accepted["cere"], &h.app, &rid);
    assert_eq!(accepted["cere"]["cache_report"], "unavailable");

    let content: String = events[1..]
        .iter()
        .filter_map(|e| match e {
            Event::Data(d) => d
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str),
            Event::Done => None,
        })
        .collect();
    assert_eq!(content, "tok0 tok1 tok2 tok3 tok4 tok5 tok6 tok7");
    for e in &events[1..events.len() - 2] {
        if e.data().get("error").is_none() && e.data()["choices"][0]["finish_reason"].is_null() {
            assert_eq!(e.data()["cere"]["committed_tokens"], Value::Null);
        }
    }
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(finish["cere"]["state"], "completed");
    assert_identity(&finish["cere"], &h.app, &rid);
    // include_usage: the usage chunk follows the finish chunk, right before [DONE].
    let usage = events[events.len() - 2].data();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 8);
    assert!(usage["cere"]["seq"].as_u64().unwrap() > finish["cere"]["seq"].as_u64().unwrap());
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "completed");
    assert_eq!(status["stream"], true);
}

/// A mock with SGLang's cache report on (exact-repeat hits at 1-token granularity) behind a config that pins
/// `expect.enable_cache_report = true`.
async fn reporting_harness() -> Harness {
    let mock = MockConfig {
        cache_granularity: 1,
        server_info: ServerInfo {
            enable_cache_report: true,
            ..ServerInfo::default()
        },
        ..MockConfig::default()
    };
    Harness::start(
        mock,
        &[("enable_cache_report = false", "enable_cache_report = true")],
    )
    .await
}

fn cache_fields(cere: &Value) -> (Value, Value) {
    (cere["cached_tokens"].clone(), cere["cache_report"].clone())
}

/// H1/Q1: with the pin true, a miss (usage present, `prompt_tokens_details: null`) is 0/reported and a repeat
/// reports the backend's count, on unary and streamed responses.
#[tokio::test]
async fn h1_pinned_report_miss_is_zero_and_repeat_hits() {
    let h = reporting_harness().await;
    for expected in [0, 2] {
        let unary: Value = h
            .chat(&chat_body(false))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(unary["usage"]["prompt_tokens"], 2);
        assert_eq!(
            cache_fields(&unary["cere"]),
            (json!(expected), json!("reported")),
            "{unary}"
        );
    }

    let mut stream = chat_body(true);
    stream["messages"] = json!([{"role": "user", "content": "three fresh words"}]);
    stream["stream_options"] = json!({"include_usage": false});
    for expected in [0, 3] {
        let response = h.chat(&stream).send().await.unwrap();
        let events = SseReader::new(response).rest().await;
        let finish = check_stream(&events);
        assert_eq!(
            cache_fields(&finish["cere"]),
            (json!(expected), json!("reported")),
            "{finish}"
        );
        // The accepted event precedes any backend usage.
        assert_eq!(
            cache_fields(&events[0].data()["cere"]),
            (Value::Null, json!("unavailable"))
        );
    }
}

/// H1/Q1: a failure without backend `usage` stays null/unavailable even with the pin true.
#[tokio::test]
async fn h1_pinned_failure_without_usage_is_unavailable() {
    let h = reporting_harness().await;
    for stream in [false, true] {
        h.mock.set_fault(Fault::ServerError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            remaining: 1,
        });
        let response = h.chat(&chat_body(stream)).send().await.unwrap();
        assert_eq!(response.status(), 502);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["cere"]["state"], "failed", "{body}");
        assert_eq!(
            cache_fields(&body["cere"]),
            (Value::Null, json!("unavailable"))
        );
    }
}

/// `GET /v1/models` (OpenAI list) and `GET /v1/models/{id}` (retrieve): the one served model with the deployment's
/// context limit, behind the chat bearer token; any other id is `404 model_not_found` like a chat naming it.
#[tokio::test]
async fn models_lists_the_served_model() {
    let mut context = 0;
    let h = Harness::start_seeded(MockConfig::default(), &[], |config| {
        context = config.deployments[0]
            .lock
            .as_ref()
            .unwrap()
            .profile
            .max_context_tokens;
    })
    .await;
    let card = json!({"id": MODEL, "object": "model", "created": 0, "owned_by": "cere",
                      "max_model_len": context});
    assert!(context > 0);

    let list = h.get("/v1/models", TOKEN_A).send().await.unwrap();
    assert_eq!(list.status(), 200);
    assert_eq!(
        list.json::<Value>().await.unwrap(),
        json!({"object": "list", "data": [card.clone()]})
    );
    let one = h
        .get(&format!("/v1/models/{MODEL}"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(one.status(), 200);
    assert_eq!(one.json::<Value>().await.unwrap(), card);

    let other = h
        .get("/v1/models/other-model", TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(other.status(), 404);
    let err: Value = other.json().await.unwrap();
    assert_eq!(err["error"]["code"], "model_not_found");
    assert_eq!(err["error"]["param"], "model");

    for path in ["/v1/models".to_owned(), format!("/v1/models/{MODEL}")] {
        let anonymous = h.http.get(h.url(&path)).send().await.unwrap();
        assert_eq!(anonymous.status(), 401, "{path}");
        let wrong = h.get(&path, "not-a-token").send().await.unwrap();
        assert_eq!(wrong.status(), 401, "{path}");
    }
}

/// A served name may contain `/` (`org/model`, as OpenAI clients send it unencoded): retrieve still matches it, and
/// another id with a `/` is `404 model_not_found` behind the token, not a route miss.
#[tokio::test]
async fn models_retrieve_matches_a_served_name_with_a_slash() {
    let mut mock = MockConfig::default();
    mock.server_info.served_model_name = "org/Qwen3.8-Flash-Next".into();
    let h = Harness::start(
        mock,
        &[(
            "served_model_name = \"Qwen3.8-Flash-Next\"",
            "served_model_name = \"org/Qwen3.8-Flash-Next\"",
        )],
    )
    .await;
    let one = h
        .get("/v1/models/org/Qwen3.8-Flash-Next", TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(one.status(), 200);
    assert_eq!(
        one.json::<Value>().await.unwrap()["id"],
        "org/Qwen3.8-Flash-Next"
    );
    let other = h.get("/v1/models/org/other", TOKEN_A).send().await.unwrap();
    assert_eq!(other.status(), 404);
    let err: Value = other.json().await.unwrap();
    assert_eq!(err["error"]["code"], "model_not_found");
    let anonymous = h
        .http
        .get(h.url("/v1/models/org/other"))
        .send()
        .await
        .unwrap();
    assert_eq!(anonymous.status(), 401);
}

#[tokio::test]
async fn a3_unsupported_parameter_is_rejected_before_dispatch() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let dispatched = h.mock.dispatched_rids();
    let mut body = chat_body(false);
    body["response_format"] = json!({"type": "structural_tag"});
    let response = h.chat(&body).send().await.unwrap();
    assert_eq!(response.status(), 400);
    assert!(response.headers().get("x-cere-request-id").is_none());
    let err: Value = response.json().await.unwrap();
    assert_eq!(err["error"]["code"], "unsupported_parameter");
    assert_eq!(err["error"]["param"], "response_format.type");

    let mut wrong_model = chat_body(false);
    wrong_model["model"] = json!("other-model");
    let response = h.chat(&wrong_model).send().await.unwrap();
    assert_eq!(response.status(), 404);
    assert_eq!(
        response.json::<Value>().await.unwrap()["error"]["code"],
        "model_not_found"
    );

    let response = h
        .post("/v1/chat/completions", TOKEN_A)
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert_eq!(
        h.mock.dispatched_rids(),
        dispatched,
        "nothing reached the backend"
    );
}

#[tokio::test]
async fn a3_clamped_output_fails() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    h.mock.set_fault(Fault::ClampOutput { tokens: 3 });
    let response = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(response.status(), 502);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "backend_clamped_output");
    assert_eq!(body["cere"]["state"], "failed");
    assert_eq!(body["cere"]["reason"], "backend_clamped_output");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed");
    assert_eq!(status["reason"], "backend_clamped_output");

    h.mock.set_fault(Fault::ClampOutput { tokens: 3 });
    let response = h.chat(&chat_body(true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let events = SseReader::new(response).rest().await;
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["state"], "failed");
    assert_eq!(terminal["cere"]["reason"], "backend_clamped_output");
    assert_eq!(terminal["error"]["code"], "backend_clamped_output");
}

#[tokio::test]
async fn s5_auth_is_required_except_for_probes() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let expected = json!({"error": {
        "message": "missing or invalid bearer token",
        "type": "invalid_request_error",
        "code": "unauthorized",
        "param": null,
    }});
    let dispatched = h.mock.dispatched_rids();
    for token in [None, Some("wrong"), Some("")] {
        let mut request = h
            .http
            .post(h.url("/v1/chat/completions"))
            .json(&chat_body(false));
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), 401);
        assert_eq!(response.json::<Value>().await.unwrap(), expected);
    }
    for path in ["/v1/cere/requests/x", "/v1/cere/sessions/x"] {
        let response = h.http.get(h.url(path)).send().await.unwrap();
        assert_eq!(response.status(), 401);
    }
    let response = h
        .http
        .post(h.url("/v1/cere/sessions"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    assert_eq!(h.mock.dispatched_rids(), dispatched);
    assert_eq!(
        h.http.get(h.url("/healthz")).send().await.unwrap().status(),
        200
    );
    assert_eq!(
        h.http.get(h.url("/readyz")).send().await.unwrap().status(),
        200
    );
}

/// `[auth] keyless_scope` (g9-prod design note F2): a request without `Authorization` is served as that scope
/// (its status reads back under the scope's own token), every present header is judged as without the setting
/// (401 when wrong), and the operator routes stay closed to an absent header.
#[tokio::test]
async fn keyless_scope_serves_headerless_requests_only() {
    let tweak = (
        "[[auth.scopes]]\nscope_id = \"a\"",
        "[auth]\nkeyless_scope = \"b\"\noperator_token_env = \"TOK_OP\"\n\n[[auth.scopes]]\nscope_id = \"a\"",
    );
    let h = Harness::start(MockConfig::default(), &[tweak]).await;
    let response = h
        .http
        .post(h.url("/v1/chat/completions"))
        .json(&chat_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    assert_eq!(h.status(&rid, TOKEN_B).await.0, 200);
    assert_eq!(h.status(&rid, TOKEN_A).await.0, 404);
    let response = h
        .http
        .get(h.url(&format!("/v1/cere/requests/{rid}")))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    for token in ["wrong", "", OPERATOR_TOKEN] {
        let response = h
            .http
            .post(h.url("/v1/chat/completions"))
            .bearer_auth(token)
            .json(&chat_body(false))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401, "{token:?}");
    }
    assert_eq!(
        h.chat(&chat_body(false)).send().await.unwrap().status(),
        200
    );
    let response = h
        .http
        .post(h.url("/v1/cere/workers/gpu0/drain"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}

#[tokio::test]
async fn l7_status_timings_and_scope() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let response = h.chat(&chat_body(false)).send().await.unwrap();
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let (code, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(code, 200);
    assert_eq!(status["request_id"], rid.as_str());
    assert_eq!(status["state"], "completed");
    assert_eq!(status["reason"], Value::Null);
    assert_eq!(status["attempt"], 1);
    assert_eq!(status["epoch"], h.app.epoch());
    assert_eq!(status["worker_id"], WORKER);
    assert_eq!(status["worker_epoch"], h.app.replicas()[0].epoch());
    assert_eq!(status["stream"], false);
    assert_eq!(status["session_id"], Value::Null);
    assert_eq!(status["execution_mode"], "backend_managed");
    let timings = &status["timings"];
    for field in ["queue_wait_ms", "total_ms"] {
        assert!(timings[field].is_u64(), "{field}: {timings}");
    }
    assert!(timings.get("first_generated_ms").is_some());
    assert!(timings.get("first_visible_ms").is_some());
    assert_eq!(timings["client_backpressure_ms"], 0);

    let other = h
        .get(&format!("/v1/cere/requests/{rid}"), TOKEN_B)
        .send()
        .await
        .unwrap();
    let unknown = h
        .get("/v1/cere/requests/cere-unknown", TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(other.status(), 404);
    assert_eq!(unknown.status(), 404);
    assert_eq!(
        other.bytes().await.unwrap(),
        unknown.bytes().await.unwrap(),
        "another scope's request is indistinguishable from an unknown one"
    );
    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_B)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 404);
    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 200, "terminal cancel is idempotent");
    assert_eq!(
        cancel.json::<Value>().await.unwrap(),
        json!({"request_id": rid, "state": "completed"})
    );
}

/// H1/Q1: a cancelled stream stays null/unavailable with the pin true. The backend's abort may still send a
/// usage chunk without `prompt_tokens_details`; for a cancelled request the pin infers nothing from that, so
/// only the token counts pass through. m1close Q12: the SSE error event's `cere.usage` is the status API's
/// `usage` (null `cached_tokens`, `unavailable`), and `cere` still has no top-level `cached_tokens`.
#[tokio::test]
async fn h1_pinned_cancelled_stream_is_unavailable() {
    let mock = MockConfig {
        token_latency: std::time::Duration::from_millis(50),
        natural_output_tokens: 2_000,
        server_info: ServerInfo {
            enable_cache_report: true,
            ..ServerInfo::default()
        },
        ..MockConfig::default()
    };
    let h = Harness::start(
        mock,
        &[("enable_cache_report = false", "enable_cache_report = true")],
    )
    .await;
    let mut body = chat_body(true);
    body["max_completion_tokens"] = json!(2_000);
    let response = h.chat(&body).send().await.unwrap();
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut reader = SseReader::new(response);
    reader.next().await;
    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert!(cancel.status().is_success());
    let events = reader.rest().await;
    let terminal = events
        .iter()
        .rev()
        .find(|e| matches!(e, Event::Data(d) if d["cere"]["state"].is_string()))
        .expect("a terminal event")
        .data();
    assert_eq!(terminal["cere"]["state"], "cancelled", "{terminal}");
    assert!(
        terminal["cere"].get("cached_tokens").is_none(),
        "{terminal}"
    );
    let usage = &terminal["cere"]["usage"];
    assert_eq!(usage["cached_tokens"], Value::Null, "{terminal}");
    assert_eq!(usage["cache_report"], "unavailable", "{terminal}");
    let (_, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(status["state"], "cancelled");
    assert_eq!(status["usage"]["cached_tokens"], Value::Null, "{status}");
    assert_eq!(status["usage"]["cache_report"], "unavailable", "{status}");
    assert_eq!(*usage, status["usage"], "{status}");
}

/// H2/Q5: the status API `usage` carries the backend's counts, and `GET /v1/cere/stats` counts only the
/// caller's scope.
#[tokio::test]
async fn h2_status_usage_and_per_scope_stats() {
    let h = reporting_harness().await;
    let mut rids = Vec::new();
    for _ in 0..2 {
        let response = h.chat(&chat_body(false)).send().await.unwrap();
        rids.push(
            response.headers()["x-cere-request-id"]
                .to_str()
                .unwrap()
                .to_owned(),
        );
        assert_eq!(response.status(), 200);
    }
    let (_, miss) = h.status(&rids[0], TOKEN_A).await;
    assert_eq!(
        miss["usage"],
        json!({"prompt_tokens": 2, "completion_tokens": 8, "cached_tokens": 0, "cache_report": "reported"})
    );
    let (_, hit) = h.status(&rids[1], TOKEN_A).await;
    assert_eq!(
        hit["usage"],
        json!({"prompt_tokens": 2, "completion_tokens": 8, "cached_tokens": 2, "cache_report": "reported"})
    );

    let mut b_body = chat_body(false);
    b_body["messages"] = json!([{"role": "user", "content": "scope b says three"}]);
    let response = h
        .post("/v1/chat/completions", TOKEN_B)
        .json(&b_body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    let stats = |token: &'static str| {
        let request = h.get("/v1/cere/stats", token);
        async move {
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), 200);
            response.json::<Value>().await.unwrap()
        }
    };
    let a = stats(TOKEN_A).await;
    let since = a["since_unix_ms"].as_u64().unwrap();
    assert!(since > 0);
    // Q11: the single form's one worker, with the caller scope's in-flight on it.
    let workers = json!({"gpu1": {"state": "ready", "epoch": 1, "cap": 4, "inflight": 0}});
    // As started: generation 0 under the startup config's hash.
    let hash = h.app.deployment(0).config_hash.clone();
    assert_eq!(
        a,
        json!({"scope": "a", "requests": {"cancelled": 0, "completed": 2, "failed": 0},
               "prompt_tokens_total": 4, "cached_tokens_total": 2, "cache_reported_requests": 2,
               "cache_unavailable_requests": 0, "since_unix_ms": since, "native_submissions": 0,
               "native_spec_verify_steps": 0, "native_spec_accepted_drafts": 0, "native_preemptions": 0,
               "native_grows": 0, "levers": null, "prefill_signature": null,
               "config_generation": 0, "effective_config_hash": hash,
               "workers": workers})
    );
    assert_eq!(
        stats(TOKEN_B).await,
        json!({"scope": "b", "requests": {"cancelled": 0, "completed": 1, "failed": 0},
               "prompt_tokens_total": 4, "cached_tokens_total": 0, "cache_reported_requests": 1,
               "cache_unavailable_requests": 0, "since_unix_ms": since, "native_submissions": 0,
               "native_spec_verify_steps": 0, "native_spec_accepted_drafts": 0, "native_preemptions": 0,
               "native_grows": 0, "levers": null, "prefill_signature": null,
               "config_generation": 0, "effective_config_hash": hash,
               "workers": workers})
    );
    let anonymous = h.http.get(h.url("/v1/cere/stats")).send().await.unwrap();
    assert_eq!(anonymous.status(), 401);
    let wrong = h.get("/v1/cere/stats", "not-a-token").send().await.unwrap();
    assert_eq!(wrong.status(), 401);
}

/// H3/Q2: with `prefix_cache_isolation = "cache_salt"` every backend chat call carries the scope's salt:
/// stable per scope, different across scopes, 32 hex chars, never the scope id; the cache is then namespaced.
#[tokio::test]
async fn h3_cache_salt_is_sent_per_scope() {
    let mock = MockConfig {
        cache_granularity: 1,
        server_info: ServerInfo {
            enable_cache_report: true,
            ..ServerInfo::default()
        },
        ..MockConfig::default()
    };
    let h = Harness::start(
        mock,
        &[
            ("enable_cache_report = false", "enable_cache_report = true"),
            (
                "allow_cross_scope_prefix_cache = true",
                "allow_cross_scope_prefix_cache = false\nprefix_cache_isolation = \"cache_salt\"",
            ),
        ],
    )
    .await;
    let mut cached = Vec::new();
    for token in [TOKEN_A, TOKEN_B, TOKEN_A, TOKEN_B] {
        for stream in [false, true] {
            let response = h
                .post("/v1/chat/completions", token)
                .json(&chat_body(stream))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            if !stream {
                let body: Value = response.json().await.unwrap();
                cached.push(body["cere"]["cached_tokens"].clone());
            } else {
                SseReader::new(response).rest().await;
            }
        }
    }
    // Scope b's first request does not hit scope a's cache entry.
    assert_eq!(cached, [json!(0), json!(0), json!(2), json!(2)]);
    let salts: Vec<String> = h
        .mock
        .received_cache_salts()
        .into_iter()
        .skip_while(Option::is_none) // the readiness smoke is not a scope's request
        .map(|s| s.expect("every chat call carries a salt"))
        .collect();
    assert_eq!(salts.len(), 8);
    let (a, b) = (salts[0].clone(), salts[2].clone());
    assert_eq!(salts, [&a, &a, &b, &b, &a, &a, &b, &b].map(String::clone));
    assert_ne!(a, b);
    for salt in [a, b] {
        assert_eq!(salt.len(), 32);
        assert!(salt
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert!(salt != "a" && salt != "b");
    }
}

/// H3: without isolation (absent, as in the shipped t4 config) no salt is sent.
#[tokio::test]
async fn h3_no_salt_without_isolation() {
    for tweak in [
        &[][..],
        &[(
            "allow_cross_scope_prefix_cache = true",
            "allow_cross_scope_prefix_cache = true\nprefix_cache_isolation = \"none\"",
        )][..],
    ] {
        let h = Harness::start(MockConfig::default(), tweak).await;
        for stream in [false, true] {
            let response = h.chat(&chat_body(stream)).send().await.unwrap();
            assert_eq!(response.status(), 200);
            response.bytes().await.unwrap();
        }
        let salts = h.mock.received_cache_salts();
        assert!(salts.len() >= 2);
        assert!(salts.iter().all(Option::is_none), "{salts:?}");
    }
}

/// Server info reporting a launch (`random_seed`, `startup_time`) with every pinned value unchanged.
fn launched(random_seed: Option<u64>, startup_time: Option<Value>) -> ServerInfo {
    ServerInfo {
        random_seed,
        startup_time,
        ..ServerInfo::default()
    }
}

async fn readyz(h: &Harness) -> (u16, Value) {
    let response = h.http.get(h.url("/readyz")).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// The single-form worker: `scheduler.devices[0]` of the t4 profile (M3 Q3).
const WORKER: &str = "gpu1";

/// The Q11 `/readyz` entry of [`WORKER`].
fn worker(body: &Value) -> &Value {
    &body["workers"][WORKER]
}

/// The problem fields of [`WORKER`]'s `/readyz` entry.
fn problem_fields(body: &Value) -> Vec<String> {
    worker(body)["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["field"].as_str().unwrap().to_owned())
        .collect()
}

/// m1close R5/Q10: a backend relaunch with identical pins is a new identity. Before the re-smoke completes
/// (the mock holds the smoke request), `/readyz` is not ready with the smoke pending and chat is refused; then
/// it is ready again under a different `instance_id`, after exactly one more smoke.
#[tokio::test]
async fn r5_launch_change_gates_until_the_new_launch_is_smoked() {
    let h = Harness::start(
        MockConfig {
            server_info: launched(Some(11), Some(json!({"total": 41.5}))),
            ..MockConfig::default()
        },
        &[],
    )
    .await;
    let (status, before) = readyz(&h).await;
    assert_eq!(status, 200, "{before}");
    let old_id = worker(&before)["instance_id"].as_str().unwrap().to_owned();
    let smokes = h.mock.dispatched_rids().len();

    h.mock.set_load_lag(Duration::from_millis(800));
    h.mock
        .set_server_info_override(Some(launched(Some(12), Some(json!({"total": 39.0})))));
    let readiness = h.app.readiness(0).clone();
    let probe = tokio::spawn(async move { readiness.probe().await });
    let gate = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let (status, body) = readyz(&h).await;
            if worker(&body)["smoke"]["state"] == "pending" {
                return (status, body);
            }
            assert!(
                !probe.is_finished(),
                "re-smoke completed before a pending gate was published: {body}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("no not-ready snapshot with the smoke pending");
    assert!(
        !probe.is_finished(),
        "the gate was observed only after the re-smoke"
    );
    let (status, body) = gate;
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["ready"], false);
    assert_eq!(problem_fields(&body), ["smoke"], "{body}");
    let refused = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(refused.status(), 503);
    assert_eq!(
        refused.json::<Value>().await.unwrap()["error"]["code"],
        "not_ready"
    );

    let after = probe.await.unwrap();
    assert!(after.ready, "{:?}", after.problems);
    let (status, body) = readyz(&h).await;
    assert_eq!(status, 200, "{body}");
    let new_id = worker(&body)["instance_id"].as_str().unwrap();
    assert_ne!(new_id, old_id);
    assert_eq!(
        worker(&body)["smoke"],
        json!({"state": "passed", "instance_id": new_id})
    );
    assert_eq!(
        h.mock.dispatched_rids().len(),
        smokes + 1,
        "one re-smoke for the new launch"
    );
}

/// m1close R5/Q10: without both launch fields, `instance_id` is today's value: `sglang-` + the first 8 bytes
/// (hex) of sha256 over the compact JSON `{"base_url", "served_model_name", "server_info": <pins>}`, where the
/// pins are `backend.expect` without `image_digest`, each absent nullable key (`chat_template`, `quantization`,
/// the four `speculative_*`) defaulting to null.
#[tokio::test]
async fn r5_absent_launch_fields_keep_todays_instance_id() {
    use sha2::{Digest, Sha256};
    let h = Harness::start(MockConfig::default(), &[]).await;
    let config = cere_runtime::config::RuntimeConfig::from_toml_str(
        &config_text(&h.mock.base_url(), h.state_dir.path(), 18390, &[]),
        &repo_root(),
    )
    .unwrap();
    let mut pins = serde_json::to_value(config.deployments[0].expect.as_ref().unwrap()).unwrap();
    let pins = pins.as_object_mut().unwrap();
    pins.remove("image_digest");
    for key in [
        "chat_template",
        "quantization",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    ] {
        pins.entry(key).or_insert(Value::Null);
    }
    let identity = json!({
        "base_url": h.mock.base_url(),
        "served_model_name": MODEL,
        "server_info": pins,
    });
    let digest = Sha256::digest(identity.to_string().as_bytes());
    let todays: String = std::iter::once("sglang-".to_owned())
        .chain(digest[..8].iter().map(|b| format!("{b:02x}")))
        .collect();

    for (info, expect_todays) in [
        (launched(None, None), true),
        (launched(Some(11), None), true),
        (launched(None, Some(json!({"total": 41.5}))), true),
        (launched(Some(11), Some(json!({"total": 41.5}))), false),
    ] {
        h.mock.set_server_info_override(Some(info.clone()));
        let snapshot = h.app.readiness(0).probe().await;
        assert!(snapshot.ready, "{info:?}: {:?}", snapshot.problems);
        let (_, body) = readyz(&h).await;
        assert_eq!(
            worker(&body)["instance_id"] == todays.as_str(),
            expect_todays,
            "{info:?}: {body}"
        );
    }
}

/// m1close R5/Q10: launch fields are observations, never pins: reporting them, changing them, or changing
/// them alongside a pin drift never yields a launch-named problem.
#[tokio::test]
async fn r5_launch_fields_never_add_problems() {
    let h = Harness::start(
        MockConfig {
            server_info: launched(Some(1), Some(json!({"total": 10.0}))),
            ..MockConfig::default()
        },
        &[],
    )
    .await;
    let (status, body) = readyz(&h).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(problem_fields(&body), Vec::<String>::new());

    h.mock
        .set_server_info_override(Some(launched(Some(2), Some(json!({"total": 12.0})))));
    let relaunched = h.app.readiness(0).probe().await;
    assert!(relaunched.ready, "{:?}", relaunched.problems);
    assert!(relaunched.problems.is_empty());

    h.mock.set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "bf16".to_owned(),
        ..launched(Some(3), Some(json!({"total": 9.0})))
    }));
    h.app.readiness(0).probe().await;
    let (status, body) = readyz(&h).await;
    assert_eq!(status, 503);
    assert_eq!(problem_fields(&body), ["kv_cache_dtype"], "{body}");
}

/// C2/C8 (D4/D5): the client's `tools` bytes - key order (`type` before `function`), `1.0`, `1E5`, 2^70 and an
/// escaped `\u00e9` - which a `Value` round trip would all change.
const TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather","description":"Weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string","default":"Z\u00e9"}},"required":["city"],"x-scale":1.0,"x-max":1E5,"x-big":1180591620717411303424}}}]"#;

/// The default mock, keeping the raw chat bodies it receives (`received_bodies`).
fn recording() -> MockConfig {
    MockConfig {
        record_bodies: true,
        ..MockConfig::default()
    }
}

/// A chat body with `tools` spliced in as raw bytes; `extra` holds further raw `"key":value` members.
fn tools_body(messages: &Value, extra: &str) -> String {
    format!(
        r#"{{"model":"{MODEL}","messages":{messages},"max_completion_tokens":16,"tools":{TOOLS}{extra}}}"#
    )
}

fn raw_chat_request(h: &Harness, body: String) -> reqwest::RequestBuilder {
    h.post("/v1/chat/completions", TOKEN_A)
        .header("content-type", "application/json")
        .body(body)
}

/// What SGLang answers a tools request with (F3 shape): a call, its logprobs, `finish_reason:"tool_calls"`.
fn tool_call_response() -> Value {
    json!({
        "id": "0c1d2e3f",
        "object": "chat.completion",
        "created": 1_760_000_000,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "reasoning_content": null,
                "tool_calls": [{
                    "id": "call_9f8e7d6c5b4a39281706f5e4",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\",  \"unit\":\"C\"}"}
                }]
            },
            "logprobs": {"content": [
                {"token": "<tool_call>", "logprob": -0.012, "bytes": null,
                 "top_logprobs": [{"token": "<tool_call>", "logprob": -0.012, "bytes": null},
                                  {"token": "The", "logprob": -4.5, "bytes": null}]}
            ]},
            "finish_reason": "tool_calls",
            "matched_stop": null
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52}
    })
}

fn without_cere(mut value: Value) -> Value {
    value.as_object_mut().unwrap().remove("cere");
    value
}

/// C2 + C8 unary (D4-D7): the forwarded body carries the tools bytes verbatim plus the D5 fields; the reply is
/// relayed value-equal (exact `arguments`), ends `completed`, and its verbatim echo continues the session at
/// revision + 1.
#[tokio::test]
async fn c8_unary_tool_call_relays_commits_and_continues() {
    let h = Harness::start(recording(), &[]).await;
    h.mock.set_scripted(Some(cere_mock_sglang::Scripted {
        response: tool_call_response(),
        chunks: Vec::new(),
    }));
    let session = {
        let r = h.post("/v1/cere/sessions", TOKEN_A).send().await.unwrap();
        assert_eq!(r.status(), 201);
        r.json::<Value>().await.unwrap()["session_id"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let before = h.mock.received_bodies().len();
    let first = json!([{"role": "user", "content": "Weather in Paris?"}]);
    let r = raw_chat_request(
        &h,
        tools_body(
            &first,
            r#","parallel_tool_calls":false,"logprobs":true,"top_logprobs":2"#,
        ),
    )
    .header("X-Cere-Session-Id", &session)
    .header("X-Cere-Expected-Revision", "0")
    .send()
    .await
    .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["cere"]["state"], "completed", "{body}");
    assert_eq!(body["cere"]["revision"], 1, "{body}");
    assert_eq!(without_cere(body.clone()), tool_call_response());
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\": \"Paris\",  \"unit\":\"C\"}"
    );
    assert_eq!(h.terminal(&rid).await["state"], "completed");

    let sent = h.mock.received_bodies().split_off(before);
    assert_eq!(sent.len(), 1);
    let text = String::from_utf8(sent[0].clone()).unwrap();
    assert!(text.contains(&format!(r#""tools":{TOOLS}"#)), "{text}");
    let forwarded: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(forwarded["tool_choice"], "auto");
    assert_eq!(forwarded["parallel_tool_calls"], false);
    assert_eq!(forwarded["logprobs"], true);
    assert_eq!(forwarded["top_logprobs"], 2);

    // The OpenAI SDK idiom: append `choices[0].message` as-is, then the tool result.
    let echoed = body["choices"][0]["message"].clone();
    let second = json!([
        {"role": "user", "content": "Weather in Paris?"},
        echoed,
        {"role": "tool", "tool_call_id": "call_9f8e7d6c5b4a39281706f5e4", "content": "18 C, sunny"}
    ]);
    h.mock.set_scripted(None);
    let r = raw_chat_request(&h, tools_body(&second, ""))
        .header("X-Cere-Session-Id", &session)
        .header("X-Cere-Expected-Revision", "1")
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["cere"]["revision"], 2, "{body}");
    let sent = h.mock.received_bodies().split_off(before);
    let forwarded: Value = serde_json::from_slice(&sent[1]).unwrap();
    // D3: `index` and the null `reasoning_content` are dropped, `content: null` forwards as SGLang renders it.
    assert_eq!(
        forwarded["messages"][1],
        json!({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_9f8e7d6c5b4a39281706f5e4", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\",  \"unit\":\"C\"}"}
        }]})
    );
    assert_eq!(
        forwarded["messages"][2],
        json!({"role": "tool", "content": "18 C, sunny", "tool_call_id": "call_9f8e7d6c5b4a39281706f5e4"})
    );
}

/// O113 (owner 2026-10-17): a reply with `reasoning_content`, echoed verbatim (reasoning included, the
/// tool-eval-bench idiom) or with the reasoning dropped, continues the session (session prefix digests leave
/// `reasoning_content` out), and the echoed reasoning is forwarded to SGLang for its chat template. Fails on main
/// 49397403 (400 `unsupported_parameter` `messages[1].reasoning_content`).
#[tokio::test]
async fn o113_echoed_reasoning_continues_the_session_and_is_forwarded() {
    let h = Harness::start(recording(), &[]).await;
    let mut reply = tool_call_response();
    reply["choices"][0]["message"]["reasoning_content"] = json!("Look the weather up.");
    for keep_reasoning in [true, false] {
        h.mock.set_scripted(Some(cere_mock_sglang::Scripted {
            response: reply.clone(),
            chunks: Vec::new(),
        }));
        let session = {
            let r = h.post("/v1/cere/sessions", TOKEN_A).send().await.unwrap();
            assert_eq!(r.status(), 201);
            r.json::<Value>().await.unwrap()["session_id"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        let first = json!([{"role": "user", "content": "Weather in Paris?", "name": "client"}]);
        let r = raw_chat_request(&h, tools_body(&first, ""))
            .header("X-Cere-Session-Id", &session)
            .header("X-Cere-Expected-Revision", "0")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["cere"]["revision"], 1, "{body}");
        let mut echoed = body["choices"][0]["message"].clone();
        assert_eq!(echoed["reasoning_content"], "Look the weather up.");
        if !keep_reasoning {
            echoed.as_object_mut().unwrap().remove("reasoning_content");
        }
        let second = json!([
            {"role": "user", "content": "Weather in Paris?"},
            echoed,
            {"role": "tool", "tool_call_id": "call_9f8e7d6c5b4a39281706f5e4", "content": "18 C, sunny"}
        ]);
        h.mock.set_scripted(None);
        let before = h.mock.received_bodies().len();
        let r = raw_chat_request(&h, tools_body(&second, ""))
            .header("X-Cere-Session-Id", &session)
            .header("X-Cere-Expected-Revision", "1")
            .send()
            .await
            .unwrap();
        let status = r.status();
        let body: Value = r.json().await.unwrap();
        assert_eq!(status, 200, "keep {keep_reasoning}: {body}");
        assert_eq!(body["cere"]["revision"], 2, "{body}");
        let sent = h.mock.received_bodies().split_off(before);
        let forwarded: Value = serde_json::from_slice(&sent[0]).unwrap();
        assert_eq!(
            forwarded["messages"][1].get("reasoning_content"),
            keep_reasoning
                .then(|| json!("Look the weather up."))
                .as_ref(),
            "{forwarded}"
        );
    }
}

/// O113 extended to tool messages (Main 2026-10-17), SGLang path: a tool result carrying a string `name` (the
/// tool-eval-bench `_tool_result_message` shape) is accepted and forwarded without it, as on the other roles; a
/// non-string `name` is 400 `invalid_value` naming it. Fails on main 186deb08 (400 `unsupported_parameter`).
#[tokio::test]
async fn o113_tool_message_name_is_accepted_and_not_forwarded() {
    let h = Harness::start(recording(), &[]).await;
    let messages = |name: Value| {
        json!([
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": Value::Null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18 C, sunny", "name": name}
        ])
    };
    let before = h.mock.received_bodies().len();
    let r = raw_chat_request(&h, tools_body(&messages(json!("get_weather")), ""))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    let sent = h.mock.received_bodies().split_off(before);
    let forwarded: Value = serde_json::from_slice(&sent[0]).unwrap();
    assert_eq!(
        forwarded["messages"][2],
        json!({"role": "tool", "content": "18 C, sunny", "tool_call_id": "call_1"})
    );
    for name in [json!(5), json!({"n": 1})] {
        let r = raw_chat_request(&h, tools_body(&messages(name.clone()), ""))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "{name}");
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_value", "{body}");
        assert_eq!(body["error"]["param"], "messages[2].name", "{body}");
    }
}

/// C1 (D2) SGLang side: a backend 400 `Tool {i} function ...` maps to 400 `invalid_value` naming the tool's
/// parameters, without the backend's text.
#[tokio::test]
async fn c1_backend_tool_schema_refusal_is_invalid_value() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    h.mock.set_fault(Fault::Reject {
        code: 400,
        message: "Tool 0 function has invalid 'parameters' schema: 'objekt' is not valid under any of the given schemas".to_owned(),
    });
    for stream in ["false", "true"] {
        let messages = json!([{"role": "user", "content": "hi"}]);
        let r = raw_chat_request(&h, tools_body(&messages, &format!(r#","stream":{stream}"#)))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "stream {stream}");
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_value", "{body}");
        assert_eq!(
            body["error"]["param"], "tools[0].function.parameters",
            "{body}"
        );
        let message = body["error"]["message"].as_str().unwrap();
        assert!(!message.contains("objekt"), "{message}");
    }
}

/// D2 "on a request with tools" (M24 Z2 R1-12): the same backend 400 on a tool-free request is an ordinary
/// backend failure (502 `backend_error`), not a tool schema refusal.
#[tokio::test]
async fn c1_tool_schema_text_without_tools_is_a_backend_error() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    h.mock.set_fault(Fault::Reject {
        code: 400,
        message: "Tool 0 function has invalid 'parameters' schema: x".to_owned(),
    });
    for stream in [false, true] {
        let r = h.chat(&chat_body(stream)).send().await.unwrap();
        assert_eq!(r.status(), 502, "stream {stream}");
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "backend_error", "{body}");
    }
}

/// C2 (D4): a number outside f64 anywhere is refused as invalid JSON, as before M2.4.
#[tokio::test]
async fn c2_out_of_range_number_is_invalid_json() {
    let h = Harness::start(recording(), &[]).await;
    let before = h.mock.received_bodies().len();
    let messages = json!([{"role": "user", "content": "hi"}]);
    let body = tools_body(&messages, "").replace("1E5", "1e400");
    let r = raw_chat_request(&h, body).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_json", "{body}");
    assert_eq!(h.mock.received_bodies().len(), before);
}

/// C2 (M2.8 D6): `response_format`/`regex`/`ebnf` and `tool_choice` reach SGLang as the client's bytes (key order,
/// `1E5`, an escaped `\u00e9`), which a `Value` round trip would change.
#[tokio::test]
async fn c2_constraints_forward_the_client_bytes() {
    let h = Harness::start(recording(), &[]).await;
    let messages = json!([{"role": "user", "content": "hi"}]);
    let base = |extra: &str| {
        format!(r#"{{"model":"{MODEL}","messages":{messages},"max_completion_tokens":16{extra}}}"#)
    };
    let schema = r#"{"json_schema":{"schema":{"properties":{"n":{"default":1E5,"type":"number","description":"Z\u00e9"}},"type":"object"},"name":"n"},"type":"json_schema"}"#;
    let named = r#"{"function":{"name":"get_weather"},"type":"function"}"#;
    let cases = [
        (
            base(&format!(r#","response_format":{schema}"#)),
            format!(r#""response_format":{schema}"#),
        ),
        (
            base(r#","regex":"\u0061[0-9]+""#),
            r#""regex":"\u0061[0-9]+""#.to_owned(),
        ),
        (
            base(r#","ebnf":"root ::= \"\u00e9\"""#),
            r#""ebnf":"root ::= \"\u00e9\"""#.to_owned(),
        ),
        (
            tools_body(&messages, &format!(r#","tool_choice":{named}"#)),
            format!(r#""tool_choice":{named}"#),
        ),
        (
            tools_body(&messages, r#","tool_choice":"required""#),
            r#""tool_choice":"required""#.to_owned(),
        ),
    ];
    for (body, forwarded) in cases {
        let before = h.mock.received_bodies().len();
        let r = raw_chat_request(&h, body.clone()).send().await.unwrap();
        let status = r.status();
        let reply: Value = r.json().await.unwrap();
        assert_eq!(status, 200, "{body} -> {reply}");
        let sent = h.mock.received_bodies().split_off(before);
        assert_eq!(sent.len(), 1);
        let text = String::from_utf8(sent[0].clone()).unwrap();
        assert!(text.contains(&forwarded), "{text}");
    }
}

/// C2 (M2.8 D6): SGLang's grammar compile 400 maps to 400 `invalid_value` on the request's constraint field (a
/// tool constraint names `tool_choice`) with a fixed message, unary and streamed (the backend refuses before the
/// stream opens); the same text on a grammar-free request stays an ordinary backend failure.
#[tokio::test]
async fn c2_grammar_compile_failure_is_invalid_value() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let tools = serde_json::from_str::<Value>(TOOLS).unwrap();
    let cases = [
        (
            json!({"response_format": {"type": "json_object"}}),
            "response_format",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}}),
            "response_format",
        ),
        (json!({"regex": "[0-9]+"}), "regex"),
        (json!({"ebnf": "root ::= \"a\""}), "ebnf"),
        (
            json!({"tools": tools, "tool_choice": "required"}),
            "tool_choice",
        ),
    ];
    for message in [
        "Failed to compile json grammar: secret",
        "Grammar compilation failed: secret",
    ] {
        h.mock.set_fault(Fault::Reject {
            code: 400,
            message: message.to_owned(),
        });
        for (patch, param) in &cases {
            for stream in [false, true] {
                let mut body = chat_body(stream);
                for (key, value) in patch.as_object().unwrap() {
                    body[key] = value.clone();
                }
                let r = h.chat(&body).send().await.unwrap();
                assert_eq!(r.status(), 400, "{patch} stream {stream}");
                let rid = r.headers()["x-cere-request-id"]
                    .to_str()
                    .unwrap()
                    .to_owned();
                let error: Value = r.json().await.unwrap();
                assert_eq!(error["error"]["code"], "invalid_value", "{error}");
                assert_eq!(error["error"]["param"], *param, "{error}");
                assert!(!error.to_string().contains("secret"), "{error}");
                let record = h.terminal(&rid).await;
                assert_eq!(
                    (&record["state"], &record["reason"]),
                    (&json!("failed"), &json!("invalid_value")),
                    "{record}"
                );
            }
        }
        for stream in [false, true] {
            let r = h.chat(&chat_body(stream)).send().await.unwrap();
            assert_eq!(r.status(), 502, "grammar-free stream {stream}");
            let error: Value = r.json().await.unwrap();
            assert_eq!(error["error"]["code"], "backend_error", "{error}");
        }
    }
    // A 500 with the same text is not a compile refusal.
    h.mock.set_fault(Fault::Reject {
        code: 500,
        message: "Failed to compile json grammar: x".to_owned(),
    });
    let mut body = chat_body(false);
    body["regex"] = json!("a");
    let r = h.chat(&body).send().await.unwrap();
    assert_eq!(r.status(), 502);
}

/// The bodies forwarded since `before`, each with its `rid` value replaced by `<rid>`.
fn forwarded_since(h: &Harness, before: usize) -> Vec<String> {
    h.mock
        .received_bodies()
        .split_off(before)
        .into_iter()
        .map(|bytes| {
            let text = String::from_utf8(bytes).unwrap();
            let rid = serde_json::from_str::<Value>(&text).unwrap()["rid"]
                .as_str()
                .unwrap()
                .to_owned();
            text.replace(&rid, "<rid>")
        })
        .collect()
}

/// M2.12 C9 (D3, D8-D10): unary and streamed, the six fields reach SGLang value-exact (`stop_token_ids` as the
/// sorted set SGLang makes of them), and only when sent.
#[tokio::test]
async fn c9_m212_fields_are_forwarded_value_exact() {
    let h = Harness::start(recording(), &[]).await;
    let before = h.mock.received_bodies().len();
    for stream in [false, true] {
        let mut body = chat_body(stream);
        for (key, value) in [
            ("chat_template_kwargs", json!({"enable_thinking": true})),
            ("separate_reasoning", json!(false)),
            ("reasoning_effort", json!("low")),
            ("stop_token_ids", json!([7, 3, 7])),
            ("min_tokens", json!(2)),
            ("ignore_eos", json!(true)),
        ] {
            body[key] = value;
        }
        let r = h.chat(&body).send().await.unwrap();
        let status = r.status();
        let text = r.text().await.unwrap();
        assert_eq!(status, 200, "{text}");
    }
    let sent = forwarded_since(&h, before);
    assert_eq!(sent.len(), 2);
    for text in &sent {
        let forwarded: Value = serde_json::from_str(text).unwrap();
        assert_eq!(
            forwarded["chat_template_kwargs"],
            json!({"enable_thinking": true})
        );
        assert_eq!(forwarded["separate_reasoning"], false);
        assert_eq!(forwarded["reasoning_effort"], "low");
        assert_eq!(forwarded["stop_token_ids"], json!([3, 7]));
        assert_eq!(forwarded["min_tokens"], 2);
        assert_eq!(forwarded["ignore_eos"], true);
    }

    // One field at a time: only that one is added.
    let before = h.mock.received_bodies().len();
    for (key, value) in [
        ("chat_template_kwargs", json!({"enable_thinking": false})),
        ("reasoning_effort", json!(0.25)),
        ("ignore_eos", json!(false)),
    ] {
        let mut body = chat_body(false);
        body[key] = value.clone();
        if key == "reasoning_effort" {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
        let r = h.chat(&body).send().await.unwrap();
        assert_eq!(r.status(), 200);
    }
    let sent = forwarded_since(&h, before);
    let base: Value = json!({
        "rid": "<rid>", "model": MODEL, "messages": [{"role": "user", "content": "hello there"}],
        "max_completion_tokens": 8, "stream": false,
    });
    let with = |extra: Value| {
        let mut body = base.clone();
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        body
    };
    let got: Vec<Value> = sent
        .iter()
        .map(|t| serde_json::from_str(t).unwrap())
        .collect();
    assert_eq!(
        got,
        [
            with(json!({"chat_template_kwargs": {"enable_thinking": false}})),
            with(
                json!({"reasoning_effort": 0.25, "chat_template_kwargs": {"enable_thinking": false}})
            ),
            with(json!({"ignore_eos": false})),
        ]
    );
}

/// M2.12 C9 regression guard (I12): without the six fields the forwarded bytes, unary and streamed, equal the
/// bodies recorded on the merge-base `f16560b`.
#[tokio::test]
async fn c9_absent_m212_fields_forward_the_merge_base_bytes() {
    let h = Harness::start(recording(), &[]).await;
    let before = h.mock.received_bodies().len();
    for stream in [false, true] {
        let r = h.chat(&chat_body(stream)).send().await.unwrap();
        assert_eq!(r.status(), 200);
        r.text().await.unwrap();
    }
    assert_eq!(
        forwarded_since(&h, before),
        [
            r#"{"max_completion_tokens":8,"messages":[{"content":"hello there","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"<rid>","stream":false}"#,
            r#"{"max_completion_tokens":8,"messages":[{"content":"hello there","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"<rid>","stream":true,"stream_options":{"include_usage":true}}"#,
        ]
    );
}

/// M2.13 C4 (m213 design note D7): unary and streamed image requests reach the mock SGLang with the parts
/// value-exact, the data URL byte-equal and `detail` only when the client sent it; a profile refusal or a bad
/// data URL never reaches the backend.
#[tokio::test]
async fn m213_c4_image_parts_forward_value_exact() {
    let h = Harness::start(recording(), &[]).await;
    let url = image_support::png_url(64, 48);
    let messages = json!([
        {"role": "system", "content": "s"},
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "image_url", "image_url": {"detail": "auto", "url": url}}
        ]}
    ]);
    for stream in [false, true] {
        let before = h.mock.received_bodies().len();
        let body = format!(
            r#"{{"model":"{MODEL}","messages":{messages},"max_completion_tokens":8,"stream":{stream}}}"#
        );
        let r = raw_chat_request(&h, body).send().await.unwrap();
        let status = r.status();
        let text = r.text().await.unwrap();
        assert_eq!(status, 200, "{text}");
        if stream {
            assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
        }
        let sent = h.mock.received_bodies().split_off(before);
        assert_eq!(sent.len(), 1, "stream {stream}");
        let raw = String::from_utf8(sent[0].clone()).unwrap();
        assert_eq!(raw.matches(url.as_str()).count(), 2, "data URL byte-equal");
        let forwarded: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(forwarded["messages"], messages, "stream {stream}");
        assert!(forwarded["messages"][1]["content"][1]["image_url"]
            .get("detail")
            .is_none());
    }

    let before = h.mock.received_bodies().len();
    for (part, param) in [
        (
            json!({"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}),
            "messages[0].content[0].image_url.url",
        ),
        (
            json!({"type": "image_url", "image_url": {"url": url, "detail": "low"}}),
            "messages[0].content[0].image_url.detail",
        ),
    ] {
        let body = json!({"model": MODEL, "messages": [{"role": "user", "content": [part]}]});
        let r = raw_chat_request(&h, body.to_string()).send().await.unwrap();
        assert_eq!(r.status(), 400);
        let error: Value = r.json().await.unwrap();
        assert_eq!(error["error"]["param"], param, "{error}");
    }
    assert_eq!(
        h.mock.received_bodies().len(),
        before,
        "refusals reach no backend"
    );
}

/// server-info design note SI-01, SI-02, SI-05: `GET /get_server_info` answers behind `/v1/models`' auth with the
/// served model and the bound address. The `sglang_http` backend's own logging and dump settings are not cered's to
/// see, so the retention keys cered cannot vouch for are `null` (`log_requests`) or absent, the backend is named under
/// `cere.unverified_workers`, and a vetting client refuses the body.
#[tokio::test]
async fn server_info_reports_an_sglang_backend_as_unverified() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let response = h.get("/get_server_info", TOKEN_A).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let info: Value = response.json().await.unwrap();
    let port: u16 = h.base.rsplit(':').next().unwrap().parse().unwrap();
    assert_eq!(info["served_model_name"], MODEL, "{info}");
    assert_eq!(info["host"], "127.0.0.1", "{info}");
    assert_eq!(info["port"], port, "{info}");
    assert_eq!(info["log_requests"], Value::Null, "{info}");
    for key in std::iter::once("crash_dump_folder").chain(CLIENT_EXPORT_FLAGS) {
        assert!(info.get(key).is_none(), "{key}: {info}");
    }
    assert_eq!(
        info["cere"]["unverified_workers"],
        json!([format!("gpu1: sglang_http {}", h.mock.base_url())]),
        "{info}"
    );
    assert_eq!(info["cere"]["telemetry_level"], "metadata", "{info}");
    assert_eq!(info["cere"]["raw_content"], false, "{info}");
    assert_eq!(
        client_refusal(&info, MODEL, false).as_deref(),
        Some("the server may log requests"),
        "{info}"
    );

    let anonymous = h.http.get(h.url("/get_server_info")).send().await.unwrap();
    assert_eq!(anonymous.status(), 401);
}
