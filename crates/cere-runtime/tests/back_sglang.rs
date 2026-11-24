//! A1 (SglangHttp + prefix-free rids), A3 finish classification and the A4/S6 `cere` extension against the
//! in-process mock. CI evidence only; the live checks are separate (m1 design note ground rules).

#[allow(dead_code)]
#[path = "support/ports.rs"]
mod ports;

use std::time::Duration;

use axum::http::StatusCode;
use cere_mock_sglang::{AbortRecord, Fault, MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::api_types::{
    ApiError, AttemptIdentity, CacheReport, CereExtension, DeploymentIdentity, RequestUsage,
};
use cere_runtime::config::RuntimeConfig;
use cere_runtime::sglang::{BackendError, ChatStream, Rid, RidGenerator, SglangHttp, Timeouts};
use cere_runtime::validate::{classify_finish, classify_response, ChatPolicy, Finish};
use serde_json::{json, Value};

const MODEL: &str = "Qwen3.8-Flash-Next";
const DEADLINE: Duration = Duration::from_secs(10);
const TIMEOUTS: Timeouts = Timeouts {
    connect: Duration::from_secs(1),
    control: Duration::from_secs(2),
    stream_idle: Duration::from_secs(2),
};

async fn mock(config: MockConfig) -> (MockHandle, SglangHttp) {
    let mock = MockSglang::start(config).await.unwrap();
    let client = SglangHttp::new(&mock.base_url(), TIMEOUTS).unwrap();
    (mock, client)
}

fn body(rid: &Rid, stream: bool, max_tokens: u64) -> Value {
    json!({
        "rid": rid.as_str(),
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello there"}],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "stream_options": {"include_usage": true},
    })
}

/// Drains a stream: (content, finish_reason, usage chunk).
async fn drain(
    stream: &mut ChatStream,
) -> Result<(String, Option<String>, Option<Value>), BackendError> {
    let mut content = String::new();
    let mut finish = None;
    let mut usage = None;
    while let Some(chunk) = stream.next_chunk().await? {
        if let Some(choice) = chunk["choices"].get(0) {
            if let Some(piece) = choice["delta"]["content"].as_str() {
                content.push_str(piece);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish = Some(reason.to_owned());
            }
        } else if !chunk["usage"].is_null() {
            usage = Some(chunk["usage"].clone());
        }
    }
    Ok((content, finish, usage))
}

#[test]
fn rid_is_fixed_width_and_round_trips() {
    let rid = Rid::new(3, 42, 0xdead_beef);
    assert_eq!(
        rid.as_str(),
        "cere-00000000000000000003-00000000000000000042-00000000deadbeef"
    );
    assert_eq!(Rid::parse(rid.as_str()).unwrap(), rid);
    assert_eq!((rid.epoch(), rid.seq(), rid.nonce()), (3, 42, 0xdead_beef));
    let max = Rid::new(u64::MAX, u64::MAX, u64::MAX);
    assert_eq!(max.as_str().len(), rid.as_str().len());
    assert_eq!(Rid::parse(max.as_str()).unwrap(), max);
    let json = serde_json::to_string(&rid).unwrap();
    assert_eq!(serde_json::from_str::<Rid>(&json).unwrap(), rid);

    for bad in [
        "",
        "cere-1",
        "cere-00000000000000000003-00000000000000000042-00000000DEADBEEF",
        "cere-00000000000000000003-00000000000000000042-00000000deadbee",
        "cere-00000000000000000003-00000000000000000042-00000000deadbeef0",
        "xere-00000000000000000003-00000000000000000042-00000000deadbeef",
        "cere-0000000000000000000a-00000000000000000042-00000000deadbeef",
        "cere-00000000000000000003_00000000000000000042-00000000deadbeef",
        "cere-99999999999999999999-00000000000000000042-00000000deadbeef",
    ] {
        assert!(Rid::parse(bad).is_err(), "{bad:?}");
    }
    assert!(serde_json::from_str::<Rid>("\"cere-1\"").is_err());
}

#[test]
fn generator_issues_distinct_prefix_free_rids() {
    let rids = RidGenerator::with_nonce(9, 1);
    let issued: Vec<Rid> = (0..12).map(|_| rids.next_rid()).collect();
    for (i, a) in issued.iter().enumerate() {
        assert_eq!((a.epoch(), a.seq(), a.nonce()), (9, i as u64, 1));
        for (j, b) in issued.iter().enumerate() {
            assert!(
                i == j || !b.as_str().starts_with(a.as_str()),
                "{a} prefixes {b}"
            );
        }
    }
    let other = RidGenerator::new(9);
    let mine = RidGenerator::new(9);
    assert_ne!(other.next_rid(), mine.next_rid(), "process nonces differ");
}

/// A1 acceptance: aborting rid N never touches rid N' that differs only in seq (with naive decimal rids,
/// `…-1` would prefix `…-10`).
#[tokio::test]
async fn abort_is_prefix_free_against_the_mock() {
    let (mock, client) = mock(MockConfig {
        token_latency: Duration::from_millis(30),
        ..MockConfig::default()
    })
    .await;
    let victim = Rid::new(1, 1, 0xfeed);
    let bystander = Rid::new(1, 10, 0xfeed);

    let mut victim_stream = client
        .chat_stream(&body(&victim, true, 64), DEADLINE)
        .await
        .unwrap();
    let mut bystander_stream = client
        .chat_stream(&body(&bystander, true, 64), DEADLINE)
        .await
        .unwrap();
    assert_eq!(mock.in_flight(), 2);
    client.abort(&victim).await.unwrap();

    let (_, victim_finish, _) = drain(&mut victim_stream).await.unwrap();
    let (content, bystander_finish, usage) = drain(&mut bystander_stream).await.unwrap();
    assert_eq!(victim_finish.as_deref(), Some("abort"));
    assert_eq!(bystander_finish.as_deref(), Some("stop"));
    assert_eq!(content, "tok0 tok1 tok2 tok3 tok4 tok5 tok6 tok7");
    assert_eq!(usage.unwrap()["completion_tokens"], 8);
    assert_eq!(
        mock.abort_requests(),
        [AbortRecord {
            rid: victim.as_str().to_owned(),
            abort_all: false
        }]
    );
    assert_eq!(
        mock.dispatched_rids(),
        [victim.as_str(), bystander.as_str()]
    );
}

#[tokio::test]
async fn unary_round_trip() {
    let (mock, client) = mock(MockConfig::default()).await;
    let rid = Rid::new(1, 0, 2);
    let response = client.chat(&body(&rid, false, 64), DEADLINE).await.unwrap();
    assert_eq!(response["id"], rid.as_str());
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "tok0 tok1 tok2 tok3 tok4 tok5 tok6 tok7"
    );
    assert_eq!(classify_response(&response, 64, 1).unwrap(), Finish::Stop);
    assert_eq!(mock.dispatched_rids(), [rid.as_str()]);
    client.health().await.unwrap();
    assert_eq!(client.models().await.unwrap(), [MODEL]);
    let info = client.server_info().await.unwrap();
    assert_eq!(info["speculative_algorithm"], "EAGLE");
}

#[tokio::test]
async fn stream_round_trip_ends_at_done() {
    let (_mock, client) = mock(MockConfig::default()).await;
    let rid = Rid::new(1, 0, 3);
    let mut stream = client
        .chat_stream(&body(&rid, true, 3), DEADLINE)
        .await
        .unwrap();
    let first = stream.next_chunk().await.unwrap().unwrap();
    assert_eq!(first["id"], rid.as_str());
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let (content, finish, usage) = drain(&mut stream).await.unwrap();
    assert_eq!(content, "tok0 tok1 tok2");
    assert_eq!(finish.as_deref(), Some("length"));
    let usage = usage.unwrap();
    assert_eq!(
        classify_finish(finish.as_deref(), usage["completion_tokens"].as_u64(), 3).unwrap(),
        Finish::Length
    );
    assert_eq!(
        stream.next_chunk().await.unwrap(),
        None,
        "stays ended after [DONE]"
    );
}

#[tokio::test]
async fn server_error_maps_to_status() {
    let (mock, client) = mock(MockConfig::default()).await;
    mock.set_fault(Fault::ServerError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        remaining: 2,
    });
    let rid = Rid::new(1, 0, 4);
    let unary = client
        .chat(&body(&rid, false, 8), DEADLINE)
        .await
        .unwrap_err();
    assert!(
        matches!(&unary, BackendError::Status { status: 503, message } if message.contains("injected")),
        "{unary:?}"
    );
    let stream = client
        .chat_stream(&body(&rid, true, 8), DEADLINE)
        .await
        .unwrap_err();
    assert!(
        matches!(stream, BackendError::Status { status: 503, .. }),
        "{stream:?}"
    );
    assert_eq!(ApiError::from(&unary).status, 502);
}

#[tokio::test]
async fn disconnect_is_a_protocol_error_not_success() {
    let (mock, client) = mock(MockConfig::default()).await;
    mock.set_fault(Fault::DisconnectAfter { tokens: 2 });
    let rid = Rid::new(1, 0, 5);
    let unary = client
        .chat(&body(&rid, false, 8), DEADLINE)
        .await
        .unwrap_err();
    assert!(matches!(unary, BackendError::Protocol(_)), "{unary:?}");

    let mut stream = client
        .chat_stream(&body(&rid, true, 8), DEADLINE)
        .await
        .unwrap();
    let mut tokens = 0;
    let error = loop {
        match stream.next_chunk().await {
            Ok(Some(chunk)) => {
                if chunk["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("tok"))
                {
                    tokens += 1;
                }
            }
            Ok(None) => panic!("a dropped stream must not end cleanly"),
            Err(error) => break error,
        }
    };
    assert_eq!(tokens, 2);
    assert!(matches!(error, BackendError::Protocol(_)), "{error:?}");
}

#[tokio::test]
async fn context_overflow_is_typed() {
    let (_mock, client) = mock(MockConfig::default()).await;
    let rid = Rid::new(1, 0, 6);
    for stream in [false, true] {
        let error = match stream {
            false => client
                .chat(&body(&rid, false, 200_000), DEADLINE)
                .await
                .unwrap_err(),
            true => client
                .chat_stream(&body(&rid, true, 200_000), DEADLINE)
                .await
                .unwrap_err(),
        };
        assert!(
            matches!(&error, BackendError::ContextOverflow { message } if message.contains("context length")),
            "{error:?}"
        );
        let api = ApiError::from(&error);
        assert_eq!((api.status, api.code), (400, "context_overflow"));
        assert_eq!(api.body()["error"]["code"], "context_overflow");
    }
}

/// Q9: SGLang's scheduler check answers a stream with HTTP 200 and then an in-stream error object (code 400,
/// "... exceeds the maximum allowed length"); a unary request gets the same text as an HTTP 400.
#[tokio::test]
async fn max_allowed_length_overflow_is_typed_in_stream_and_unary() {
    let (mock, client) = mock(MockConfig::default()).await;
    let message =
        "Input length (199994 tokens) exceeds the maximum allowed length (199994 tokens).";
    mock.set_fault(Fault::StreamError {
        code: 400,
        message: message.to_owned(),
    });
    let rid = Rid::new(1, 0, 7);
    let mut stream = client
        .chat_stream(&body(&rid, true, 2), DEADLINE)
        .await
        .unwrap();
    let in_stream = drain(&mut stream).await.unwrap_err();
    let unary = client
        .chat(&body(&rid, false, 2), DEADLINE)
        .await
        .unwrap_err();
    for error in [in_stream, unary] {
        assert_eq!(
            error,
            BackendError::ContextOverflow {
                message: message.to_owned()
            }
        );
        assert_eq!(ApiError::from(&error).code, "context_overflow");
    }
    // Another 400 and a 500 mentioning the context length stay ordinary backend errors.
    for (code, text) in [(400, "Invalid request"), (500, "context length")] {
        mock.set_fault(Fault::StreamError {
            code,
            message: text.to_owned(),
        });
        let mut stream = client
            .chat_stream(&body(&rid, true, 2), DEADLINE)
            .await
            .unwrap();
        let error = drain(&mut stream).await.unwrap_err();
        assert!(
            matches!(&error, BackendError::Status { status, .. } if *status == code),
            "{error:?}"
        );
        assert_eq!(ApiError::from(&error).code, "backend_error");
    }
}

/// C2 (M2.8 D6, F13): an SGLang 400 whose message starts `Failed to compile` or `Grammar compilation failed` is a
/// grammar compile refusal, in stream and unary; the client sees `invalid_value` without the backend's text. A
/// 500 or another prefix stays an ordinary backend error.
#[tokio::test]
async fn grammar_compile_refusal_is_typed_in_stream_and_unary() {
    let (mock, client) = mock(MockConfig::default()).await;
    let rid = Rid::new(1, 0, 9);
    for message in [
        "Failed to compile json grammar: secret",
        "Grammar compilation failed: secret",
    ] {
        mock.set_fault(Fault::StreamError {
            code: 400,
            message: message.to_owned(),
        });
        let mut stream = client
            .chat_stream(&body(&rid, true, 2), DEADLINE)
            .await
            .unwrap();
        let in_stream = drain(&mut stream).await.unwrap_err();
        let unary = client
            .chat(&body(&rid, false, 2), DEADLINE)
            .await
            .unwrap_err();
        for error in [in_stream, unary] {
            assert_eq!(error, BackendError::GrammarCompile);
            let api = ApiError::from(&error);
            assert_eq!((api.status, api.code), (400, "invalid_value"));
            assert!(!api.message.contains("secret"), "{}", api.message);
        }
    }
    for (code, text) in [
        (500, "Failed to compile json grammar: x"),
        (400, "Error: Failed to compile"),
    ] {
        mock.set_fault(Fault::Reject {
            code,
            message: text.to_owned(),
        });
        let error = client
            .chat(&body(&rid, false, 2), DEADLINE)
            .await
            .unwrap_err();
        assert!(
            matches!(&error, BackendError::Status { status, .. } if *status == code),
            "{error:?}"
        );
    }
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn connect_and_timeouts_are_typed() {
    // Reserved for this process and never bound: nothing listens on it.
    let port = ports::free_port();
    let closed = SglangHttp::new(&format!("http://127.0.0.1:{port}"), TIMEOUTS).unwrap();
    let error = closed.server_info().await.unwrap_err();
    assert!(matches!(error, BackendError::Connect(_)), "{error:?}");
    assert_eq!(ApiError::from(&error).status, 503);
    let error = closed.abort(&Rid::new(0, 0, 0)).await.unwrap_err();
    assert!(matches!(error, BackendError::Connect(_)), "{error:?}");

    let (mock, client) = mock(MockConfig::default()).await;
    mock.set_fault(Fault::HangBeforeResponse);
    let rid = Rid::new(1, 0, 7);
    let error = client
        .chat(&body(&rid, false, 8), Duration::from_millis(200))
        .await
        .unwrap_err();
    assert!(matches!(error, BackendError::Timeout(_)), "{error:?}");

    mock.set_fault(Fault::HangAfterTokens { tokens: 1 });
    let idle = SglangHttp::new(
        &mock.base_url(),
        Timeouts {
            stream_idle: Duration::from_millis(200),
            ..TIMEOUTS
        },
    )
    .unwrap();
    let mut stream = idle
        .chat_stream(&body(&rid, true, 8), DEADLINE)
        .await
        .unwrap();
    let error = drain(&mut stream).await.unwrap_err();
    assert!(matches!(error, BackendError::Timeout(_)), "{error:?}");
}

#[tokio::test]
async fn clamped_output_is_classified_for_unary_and_stream() {
    let (mock, client) = mock(MockConfig::default()).await;
    mock.set_fault(Fault::ClampOutput { tokens: 2 });
    let rid = Rid::new(1, 0, 8);
    let response = client.chat(&body(&rid, false, 5), DEADLINE).await.unwrap();
    assert_eq!(response["choices"][0]["finish_reason"], "length");
    let finish = classify_response(&response, 5, 1).unwrap();
    assert_eq!(
        finish,
        Finish::BackendClampedOutput {
            completion_tokens: 2,
            budget: 5
        }
    );
    assert_eq!(finish.as_str(), "backend_clamped_output");

    let mut stream = client
        .chat_stream(&body(&rid, true, 5), DEADLINE)
        .await
        .unwrap();
    let (_, reason, usage) = drain(&mut stream).await.unwrap();
    let completion = usage.unwrap()["completion_tokens"].as_u64();
    assert_eq!(
        classify_finish(reason.as_deref(), completion, 5)
            .unwrap()
            .as_str(),
        "backend_clamped_output"
    );
}

#[test]
fn finish_classification_table() {
    assert_eq!(
        classify_finish(Some("stop"), None, 4).unwrap(),
        Finish::Stop
    );
    assert_eq!(
        classify_finish(Some("abort"), Some(1), 4).unwrap(),
        Finish::Abort
    );
    assert_eq!(
        classify_finish(Some("length"), Some(4), 4).unwrap(),
        Finish::Length
    );
    assert_eq!(
        classify_finish(Some("length"), Some(3), 4).unwrap(),
        Finish::BackendClampedOutput {
            completion_tokens: 3,
            budget: 4
        }
    );
    assert_eq!(
        classify_finish(Some("tool_calls"), Some(1), 4).unwrap(),
        Finish::ToolCalls
    );
    for (reason, tokens) in [(Some("length"), None), (None, Some(1))] {
        assert!(matches!(
            classify_finish(reason, tokens, 4),
            Err(BackendError::Protocol(_))
        ));
    }
    assert!(classify_response(&json!({"choices": []}), 4, 1).is_err());
}

/// M2.6 D11: `n` choices with `usage.completion_tokens` summed over them: the clamp bound is `budget` per `length`
/// choice; `abort` outranks a clamp, a clamp outranks a success; the choice count must be `n`.
#[test]
fn n_choice_classification() {
    let response = |reasons: &[&str], completion: u64| {
        let choices: Vec<Value> = reasons
            .iter()
            .enumerate()
            .map(|(i, r)| json!({"index": i, "finish_reason": r}))
            .collect();
        json!({"choices": choices, "usage": {"completion_tokens": completion}})
    };
    let classify = |reasons: &[&str], completion| {
        classify_response(&response(reasons, completion), 4, reasons.len() as u32)
    };
    assert_eq!(classify(&["length", "length"], 8).unwrap(), Finish::Length);
    assert_eq!(
        classify(&["length", "length"], 7).unwrap(),
        Finish::BackendClampedOutput {
            completion_tokens: 7,
            budget: 8
        }
    );
    assert_eq!(classify(&["stop", "length"], 5).unwrap(), Finish::Stop);
    assert!(matches!(
        classify(&["stop", "length"], 3).unwrap(),
        Finish::BackendClampedOutput { .. }
    ));
    assert_eq!(
        classify(&["length", "abort", "length"], 1).unwrap(),
        Finish::Abort
    );
    assert_eq!(
        classify(&["tool_calls", "stop"], 2).unwrap(),
        Finish::ToolCalls
    );
    assert!(matches!(
        classify(&["stop", "bogus"], 2),
        Err(BackendError::Protocol(_))
    ));
    assert!(matches!(
        classify_response(&response(&["stop"], 1), 4, 2),
        Err(BackendError::Protocol(_))
    ));
}

fn live_config() -> RuntimeConfig {
    RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/sglang-t4-qwen38.toml"),
    )
    .unwrap()
}

/// A4/S6: the `cere` object carries the identity fields; cached_tokens only when the backend reports it.
#[tokio::test]
async fn cere_extension_surfaces_reported_cache_hits_only() {
    let config = live_config();
    let deployment = DeploymentIdentity::from_config(&config, 0).unwrap();
    assert_eq!(deployment.recipe_id, "qwen38-flash-next");
    assert_eq!(deployment.profile_id, "sglang-t4-qwen38-flash-next-v1");
    assert_eq!(deployment.execution_mode, "backend_managed");
    assert_eq!(deployment.config_hash, config.config_hash());
    let attempt = AttemptIdentity {
        request_id: "req-1".to_owned(),
        attempt: 1,
        worker_id: Some("gpu1".to_owned()),
        worker_epoch: Some(2),
        epoch: 4,
        instance_id: Some("sglang-0123456789abcdef".to_owned()),
    };
    let policy = ChatPolicy::from_config(&config, 0).unwrap();
    let request = policy
        .validate(&json!({"model": MODEL, "messages": [{"role": "user", "content": "a b c d"}]}))
        .unwrap();

    let (_plain, client) = mock(MockConfig::default()).await;
    let response = client
        .chat(&request.backend_body(&Rid::new(4, 0, 1)), DEADLINE)
        .await
        .unwrap();
    let usage = RequestUsage::from_backend(response.get("usage"), deployment.cache_report_pinned);
    let ext = CereExtension::new(&deployment, &attempt, &usage);
    assert_eq!(
        (ext.cached_tokens, ext.cache_report),
        (None, CacheReport::Unavailable)
    );
    let json = serde_json::to_value(&ext).unwrap();
    assert_eq!(json["cached_tokens"], Value::Null);
    assert_eq!(json["cache_report"], "unavailable");
    assert_eq!(json["execution_mode"], "backend_managed");
    assert_eq!(json["lock_sha256"], deployment.lock_sha256);
    assert_eq!(json["epoch"], 4);
    assert_eq!(json["worker_id"], "gpu1");
    assert_eq!(json["worker_epoch"], 2);
    assert_eq!(json["instance_id"], "sglang-0123456789abcdef");
    // Before dispatch the worker fields are null.
    let queued = AttemptIdentity {
        worker_id: None,
        worker_epoch: None,
        instance_id: None,
        ..attempt.clone()
    };
    let json = serde_json::to_value(CereExtension::new(&deployment, &queued, &usage)).unwrap();
    assert_eq!(
        (
            &json["worker_id"],
            &json["worker_epoch"],
            &json["instance_id"]
        ),
        (&Value::Null, &Value::Null, &Value::Null)
    );

    // The live config pins `enable_cache_report = false`: unchanged by Q1, a reported count passes through and
    // a missing one stays unavailable.
    let (_reporting, client) = mock(MockConfig {
        server_info: ServerInfo {
            enable_cache_report: true,
            ..ServerInfo::default()
        },
        cache_granularity: 1,
        ..MockConfig::default()
    })
    .await;
    let mut seen = Vec::new();
    for seq in 1..=2 {
        let response = client
            .chat(&request.backend_body(&Rid::new(4, seq, 1)), DEADLINE)
            .await
            .unwrap();
        let usage =
            RequestUsage::from_backend(response.get("usage"), deployment.cache_report_pinned);
        let ext = CereExtension::new(&deployment, &attempt, &usage);
        seen.push((ext.cached_tokens, ext.cache_report));
    }
    assert_eq!(
        seen,
        [
            (None, CacheReport::Unavailable),
            (Some(4), CacheReport::Reported)
        ]
    );
}

/// Q6: `load` sums `num_reqs + num_waiting_reqs` over the `/get_load` dp ranks (the `_common.py` formula) and
/// types a malformed answer as a protocol error.
#[tokio::test]
async fn load_sums_running_and_waiting_requests() {
    let (mock, client) = mock(MockConfig {
        token_latency: Duration::from_millis(20),
        natural_output_tokens: 1_000,
        server_info: ServerInfo {
            max_running_requests: 1,
            ..ServerInfo::default()
        },
        ..MockConfig::default()
    })
    .await;
    assert_eq!(client.load("/get_load").await, Ok(0));
    let mut streams = Vec::new();
    for seq in 0..3 {
        let rid = Rid::new(1, seq, 7);
        streams.push(
            client
                .chat_stream(&body(&rid, true, 1_000), DEADLINE)
                .await
                .unwrap(),
        );
    }
    let until = std::time::Instant::now() + Duration::from_secs(5);
    while client.load("/get_load").await != Ok(3) {
        assert!(std::time::Instant::now() < until, "load never reached 3");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    drop(streams);
    let error = client.load("/v1/models").await.unwrap_err();
    assert!(matches!(error, BackendError::Protocol(_)), "{error:?}");
    let error = client.load("/missing").await.unwrap_err();
    assert!(
        matches!(error, BackendError::Status { status: 404, .. }),
        "{error:?}"
    );
    mock.shutdown().await.unwrap();
}
