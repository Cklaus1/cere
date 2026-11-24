//! Self-tests for the CI-only SGLang mock. A minimal HTTP/1.1 client keeps the crate free of extra deps.

use cere_mock_sglang::{AbortRecord, Fault, MockConfig, MockHandle, MockSglang, ServerInfo};
use serde_json::{json, Value};
use std::future::Future;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const LIMIT: Duration = Duration::from_secs(10);

async fn within<F: Future>(fut: F) -> F::Output {
    tokio::time::timeout(LIMIT, fut)
        .await
        .expect("test step timed out")
}

async fn eventually(what: &str, cond: impl Fn() -> bool) {
    within(async {
        while !cond() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(cond(), "{what}");
}

/// One request on its own connection (`Connection: close`).
struct Conn {
    stream: TcpStream,
    buf: Vec<u8>,
}

#[derive(Debug, PartialEq)]
enum Event {
    Data(String),
    /// Terminating chunk received.
    End,
    /// Connection closed before the terminating chunk.
    Broken,
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

impl Conn {
    async fn send(addr: SocketAddr, method: &str, path: &str, body: Option<&Value>) -> Conn {
        let mut stream = TcpStream::connect(addr).await.expect("connect");
        let body = body.map(Value::to_string).unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).await.expect("write");
        Conn {
            stream,
            buf: Vec::new(),
        }
    }

    /// Reads more bytes; false on EOF or reset.
    async fn fill(&mut self) -> bool {
        let mut chunk = [0u8; 4096];
        match self.stream.read(&mut chunk).await {
            Ok(0) | Err(_) => false,
            Ok(n) => {
                self.buf.extend_from_slice(&chunk[..n]);
                true
            }
        }
    }

    /// Status and lower-cased headers; None if the connection closes first.
    async fn head(&mut self) -> Option<(u16, Vec<(String, String)>)> {
        loop {
            if let Some(end) = find(&self.buf, b"\r\n\r\n") {
                let text = String::from_utf8(self.buf.drain(..end + 4).collect()).unwrap();
                let mut lines = text.split("\r\n");
                let status = lines.next()?.split(' ').nth(1)?.parse().ok()?;
                let headers = lines
                    .filter_map(|line| line.split_once(':'))
                    .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
                    .collect();
                return Some((status, headers));
            }
            if !self.fill().await {
                return None;
            }
        }
    }

    /// Next chunk of a chunked body: Some(data), None at the terminator, Err on a dropped connection.
    async fn chunk(&mut self) -> Result<Option<Vec<u8>>, ()> {
        loop {
            if let Some(line) = find(&self.buf, b"\r\n") {
                let size = std::str::from_utf8(&self.buf[..line]).unwrap().trim();
                let size = usize::from_str_radix(size, 16).expect("chunk size");
                if size == 0 {
                    return Ok(None);
                }
                if self.buf.len() >= line + 2 + size + 2 {
                    let data = self.buf[line + 2..line + 2 + size].to_vec();
                    self.buf.drain(..line + 2 + size + 2);
                    return Ok(Some(data));
                }
            }
            if !self.fill().await {
                return Err(());
            }
        }
    }

    /// Full body after `head`; Err if the connection drops first.
    async fn body(&mut self, headers: &[(String, String)]) -> Result<Vec<u8>, ()> {
        let header = |name: &str| {
            headers
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.as_str())
        };
        if header("transfer-encoding") == Some("chunked") {
            let mut body = Vec::new();
            while let Some(data) = self.chunk().await? {
                body.extend(data);
            }
            return Ok(body);
        }
        let len: usize = header("content-length").map_or(0, |v| v.parse().unwrap());
        while self.buf.len() < len {
            if !self.fill().await {
                return Err(());
            }
        }
        Ok(self.buf.drain(..len).collect())
    }
}

/// Reads SSE `data:` events from a chunked body (events never straddle chunks here, but are re-split anyway).
struct SseReader {
    conn: Conn,
    pending: String,
}

impl SseReader {
    async fn open(addr: SocketAddr, body: &Value) -> SseReader {
        let mut conn = Conn::send(addr, "POST", "/v1/chat/completions", Some(body)).await;
        let (status, headers) = within(conn.head()).await.expect("stream head");
        assert_eq!(status, 200);
        assert!(headers.contains(&("content-type".to_owned(), "text/event-stream".to_owned())));
        SseReader {
            conn,
            pending: String::new(),
        }
    }

    async fn next(&mut self) -> Event {
        loop {
            if let Some(end) = self.pending.find("\n\n") {
                let event: String = self.pending.drain(..end + 2).collect();
                let data = event.trim_end().strip_prefix("data: ").expect("data line");
                return Event::Data(data.to_owned());
            }
            match self.conn.chunk().await {
                Ok(Some(data)) => self.pending.push_str(std::str::from_utf8(&data).unwrap()),
                Ok(None) => return Event::End,
                Err(()) => return Event::Broken,
            }
        }
    }

    async fn json(&mut self) -> Value {
        match within(self.next()).await {
            Event::Data(data) => serde_json::from_str(&data).expect("chunk json"),
            other => panic!("expected a chunk, got {other:?}"),
        }
    }

    /// Next chunk must be a content token; returns its text.
    async fn token(&mut self) -> String {
        let chunk = self.json().await;
        let choice = &chunk["choices"][0];
        assert_eq!(choice["finish_reason"], Value::Null, "{chunk}");
        choice["delta"]["content"]
            .as_str()
            .expect("content delta")
            .to_owned()
    }

    /// Reads the role chunk.
    async fn role(&mut self) -> Value {
        let chunk = self.json().await;
        assert_eq!(
            chunk["choices"][0]["delta"],
            json!({"reasoning_content": null, "role": "assistant", "content": ""})
        );
        chunk
    }

    /// Reads tokens until the finish chunk; returns (tokens, finish chunk).
    async fn until_finish(&mut self) -> (Vec<String>, Value) {
        let mut tokens = Vec::new();
        loop {
            let chunk = self.json().await;
            if chunk["choices"][0]["finish_reason"].is_null() {
                tokens.push(
                    chunk["choices"][0]["delta"]["content"]
                        .as_str()
                        .unwrap()
                        .to_owned(),
                );
            } else {
                return (tokens, chunk);
            }
        }
    }

    async fn done(&mut self) {
        assert_eq!(within(self.next()).await, Event::Data("[DONE]".to_owned()));
        assert_eq!(within(self.next()).await, Event::End);
    }
}

async fn call(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> Result<(u16, Vec<u8>), ()> {
    within(async {
        let mut conn = Conn::send(addr, method, path, body).await;
        let (status, headers) = conn.head().await.ok_or(())?;
        Ok((status, conn.body(&headers).await?))
    })
    .await
}

async fn call_json(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> (u16, Value) {
    let (status, body) = call(addr, method, path, body)
        .await
        .expect("complete response");
    (status, serde_json::from_slice(&body).expect("json body"))
}

async fn abort(addr: SocketAddr, body: Value) {
    let (status, body) = call(addr, "POST", "/abort_request", Some(&body))
        .await
        .unwrap();
    assert_eq!((status, body.len()), (200, 0));
}

fn chat(rid: &str, max_tokens: u64) -> Value {
    json!({
        "rid": rid,
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "say three words"}],
        "max_tokens": max_tokens,
    })
}

fn stream_chat(rid: &str, max_tokens: u64, include_usage: bool) -> Value {
    let mut body = chat(rid, max_tokens);
    body["stream"] = json!(true);
    if include_usage {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

async fn start(config: MockConfig) -> MockHandle {
    MockSglang::start(config).await.expect("mock starts")
}

fn slow(token_latency_ms: u64, natural_output_tokens: u32) -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(token_latency_ms),
        natural_output_tokens,
        ..MockConfig::default()
    }
}

#[tokio::test]
async fn health_models_and_server_info_match_the_live_probe() {
    let mock = start(MockConfig::default()).await;
    assert!(mock.base_url().starts_with("http://127.0.0.1:"));
    let addr = mock.addr();
    assert_eq!(
        call(addr, "GET", "/health", None).await,
        Ok((200, Vec::new()))
    );

    let (status, models) = call_json(addr, "GET", "/v1/models", None).await;
    assert_eq!(status, 200);
    assert_eq!(models["object"], "list");
    let model = &models["data"][0];
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(model["id"], "Qwen3.8-Flash-Next");
    assert_eq!(model["root"], "Qwen3.8-Flash-Next");
    assert_eq!(model["object"], "model");
    assert_eq!(model["owned_by"], "sglang");
    assert_eq!(model["parent"], Value::Null);
    assert_eq!(model["max_model_len"], 200_000);
    assert!(model["created"].is_u64());

    let (status, info) = call_json(addr, "GET", "/get_server_info", None).await;
    assert_eq!(status, 200);
    let path = "/models/qwen38-flash-next/Qwen3.8-Flash-Next-NVFP4";
    let expected = json!({
        "version": "0.5.19.dev3+gd4689328e",
        "model_path": path,
        "tokenizer_path": path,
        "served_model_name": "Qwen3.8-Flash-Next",
        "chat_template": null,
        "quantization": "modelopt_fp4",
        "kv_cache_dtype": "fp8_e4m3",
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "allow_auto_truncate": false,
        "enable_cache_report": false,
        "context_length": 200_000,
        "max_running_requests": 4,
        "disable_radix_cache": false,
    });
    assert_eq!(info, expected);
    mock.shutdown().await.unwrap();
}

/// M24 Z2 R1-16: bodies are kept only behind `record_bodies` (a long dev run must not grow without bound).
#[tokio::test]
async fn chat_bodies_are_recorded_only_when_asked() {
    for record in [false, true] {
        let mock = start(MockConfig {
            record_bodies: record,
            ..MockConfig::default()
        })
        .await;
        let body = chat("cere-a", 1);
        let (status, _) = call_json(mock.addr(), "POST", "/v1/chat/completions", Some(&body)).await;
        assert_eq!(status, 200);
        let recorded = mock.received_bodies();
        assert_eq!(recorded.len(), usize::from(record), "record {record}");
        if record {
            assert_eq!(serde_json::from_slice::<Value>(&recorded[0]).unwrap(), body);
        }
    }
}

#[tokio::test]
async fn unary_completion_shape_and_rid_passthrough() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    let (status, body) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-a", 3)),
    )
    .await;
    assert_eq!(status, 200);
    let expected = json!({
        "id": "cere-a",
        "object": "chat.completion",
        "created": body["created"],
        "model": "Qwen3.8-Flash-Next",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "tok0 tok1 tok2", "reasoning_content": null, "tool_calls": null},
            "logprobs": null,
            "finish_reason": "length",
            "matched_stop": null,
        }],
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 8,
            "completion_tokens": 3,
            "prompt_tokens_details": null,
            "reasoning_tokens": 0,
        },
        "metadata": {"weight_version": "default"},
    });
    assert_eq!(body, expected);
    assert!(body["created"].is_u64());

    // Natural EOS before the budget: stop, matched_stop = eos token id.
    let (_, body) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-b", 100)),
    )
    .await;
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["choices"][0]["matched_stop"], 151_645);
    assert_eq!(body["usage"]["completion_tokens"], 8);

    // max_completion_tokens wins over max_tokens.
    let mut both = chat("cere-c", 100);
    both["max_completion_tokens"] = json!(2);
    let (_, body) = call_json(addr, "POST", "/v1/chat/completions", Some(&both)).await;
    assert_eq!(body["choices"][0]["message"]["content"], "tok0 tok1");
    assert_eq!(body["choices"][0]["finish_reason"], "length");

    // Stop string: content cut before the match.
    let mut stop = chat("cere-d", 100);
    stop["stop"] = json!(["nope", "tok2"]);
    let (_, body) = call_json(addr, "POST", "/v1/chat/completions", Some(&stop)).await;
    assert_eq!(body["choices"][0]["message"]["content"], "tok0 tok1 ");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["choices"][0]["matched_stop"], "tok2");

    // No rid: generated 32-hex id, still recorded.
    let mut anonymous = chat("", 1);
    anonymous.as_object_mut().unwrap().remove("rid");
    let (_, body) = call_json(addr, "POST", "/v1/chat/completions", Some(&anonymous)).await;
    let id = body["id"].as_str().unwrap().to_owned();
    assert!(
        id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "{id}"
    );

    assert_eq!(
        mock.dispatched_rids(),
        ["cere-a", "cere-b", "cere-c", "cere-d", id.as_str()]
    );
    eventually("all requests finished", || mock.in_flight() == 0).await;
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_cache_reports_hits_only_namespaced_by_salt() {
    let mut config = MockConfig {
        cache_granularity: 2,
        ..MockConfig::default()
    };
    config.server_info.enable_cache_report = true;
    let mock = start(config).await;
    let addr = mock.addr();
    let details = |body: &Value| body["usage"]["prompt_tokens_details"].clone();
    // 5 prompt tokens: a miss reports null (as live), a repeat hits 5 rounded down to 2 → 4.
    let (_, miss) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-a", 1)),
    )
    .await;
    assert_eq!(miss["usage"]["prompt_tokens"], 5);
    assert_eq!(details(&miss), Value::Null);
    let (_, hit) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-b", 1)),
    )
    .await;
    assert_eq!(details(&hit), json!({"cached_tokens": 4}));

    // A salt is its own namespace.
    let mut salted = chat("cere-c", 1);
    salted["cache_salt"] = json!("s1");
    let (_, miss) = call_json(addr, "POST", "/v1/chat/completions", Some(&salted)).await;
    assert_eq!(details(&miss), Value::Null);
    salted["rid"] = json!("cere-d");
    let (_, hit) = call_json(addr, "POST", "/v1/chat/completions", Some(&salted)).await;
    assert_eq!(details(&hit), json!({"cached_tokens": 4}));

    let mut sse = SseReader::open(addr, &stream_chat("cere-e", 1, true)).await;
    sse.role().await;
    sse.until_finish().await;
    let usage = sse.json().await;
    assert_eq!(details(&usage), json!({"cached_tokens": 4}));
    sse.done().await;
    assert_eq!(
        mock.received_cache_salts(),
        [
            None,
            None,
            Some("s1".to_owned()),
            Some("s1".to_owned()),
            None
        ]
    );
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_cache_default_granularity_and_report_off() {
    let long = |rid: &str| {
        let words = vec!["w"; 70].join(" ");
        json!({"rid": rid, "messages": [{"role": "user", "content": words}], "max_tokens": 1})
    };
    let mut reporting = MockConfig::default();
    reporting.server_info.enable_cache_report = true;
    let mock = start(reporting).await;
    let addr = mock.addr();
    let (_, miss) = call_json(addr, "POST", "/v1/chat/completions", Some(&long("cere-a"))).await;
    assert_eq!(miss["usage"]["prompt_tokens_details"], Value::Null);
    let (_, hit) = call_json(addr, "POST", "/v1/chat/completions", Some(&long("cere-b"))).await;
    assert_eq!(
        hit["usage"]["prompt_tokens_details"],
        json!({"cached_tokens": 64})
    );
    mock.shutdown().await.unwrap();

    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    for rid in ["cere-a", "cere-b"] {
        let (_, body) = call_json(addr, "POST", "/v1/chat/completions", Some(&long(rid))).await;
        assert_eq!(body["usage"]["prompt_tokens_details"], Value::Null);
    }
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn stream_order_with_and_without_include_usage() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();

    let mut sse = SseReader::open(addr, &stream_chat("cere-s1", 3, true)).await;
    let role = sse.role().await;
    assert_eq!(role["id"], "cere-s1");
    assert_eq!(role["object"], "chat.completion.chunk");
    assert_eq!(role["model"], "Qwen3.8-Flash-Next");
    assert_eq!(role["choices"][0]["finish_reason"], Value::Null);
    assert_eq!(role["usage"], Value::Null);
    assert_eq!(sse.token().await, "tok0");
    assert_eq!(sse.token().await, " tok1");
    assert_eq!(sse.token().await, " tok2");
    let finish = sse.json().await;
    assert_eq!(finish["id"], "cere-s1");
    assert_eq!(
        finish["choices"],
        json!([{
            "index": 0,
            "delta": {"role": null, "content": null, "reasoning_content": null, "tool_calls": null},
            "logprobs": null,
            "finish_reason": "length",
            "matched_stop": null,
        }])
    );
    let usage = sse.json().await;
    assert_eq!(usage["id"], "cere-s1");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(
        usage["usage"],
        json!({"prompt_tokens": 5, "total_tokens": 8, "completion_tokens": 3, "prompt_tokens_details": null, "reasoning_tokens": 0})
    );
    sse.done().await;

    let mut sse = SseReader::open(addr, &stream_chat("cere-s2", 100, false)).await;
    sse.role().await;
    let (tokens, finish) = sse.until_finish().await;
    assert_eq!(tokens.concat(), "tok0 tok1 tok2 tok3 tok4 tok5 tok6 tok7");
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(finish["choices"][0]["matched_stop"], 151_645);
    sse.done().await;

    eventually("streams released", || mock.in_flight() == 0).await;
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn abort_matches_rid_prefixes_so_cere_rids_must_be_prefix_free() {
    let mock = start(slow(20, 10_000)).await;
    let addr = mock.addr();
    let mut one = SseReader::open(addr, &stream_chat("cere-1", 10_000, true)).await;
    let mut ten = SseReader::open(addr, &stream_chat("cere-10", 10_000, false)).await;
    let mut two = SseReader::open(addr, &stream_chat("cere-2", 10_000, false)).await;
    for sse in [&mut one, &mut ten, &mut two] {
        sse.role().await;
        sse.token().await;
    }

    abort(addr, json!({"rid": "cere-1"})).await;
    // "cere-10".startswith("cere-1"): both die.
    let (_, finish) = one.until_finish().await;
    assert_eq!(finish["choices"][0]["finish_reason"], "abort");
    assert_eq!(finish["choices"][0]["matched_stop"], Value::Null);
    let usage = one.json().await;
    assert_eq!(usage["choices"], json!([]));
    one.done().await;
    let (_, finish) = ten.until_finish().await;
    assert_eq!(finish["choices"][0]["finish_reason"], "abort");
    ten.done().await;

    // cere-2 is untouched and keeps streaming.
    two.token().await;
    two.token().await;
    // An empty rid prefixes every rid.
    abort(addr, json!({"rid": ""})).await;
    let (_, finish) = two.until_finish().await;
    assert_eq!(finish["choices"][0]["finish_reason"], "abort");
    two.done().await;

    assert_eq!(
        mock.abort_requests(),
        [
            AbortRecord {
                rid: "cere-1".to_owned(),
                abort_all: false
            },
            AbortRecord {
                rid: String::new(),
                abort_all: false
            },
        ]
    );
    eventually("aborted streams released", || mock.in_flight() == 0).await;
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn unary_abort_returns_partial_output_with_finish_abort() {
    let mock = start(slow(20, 10_000)).await;
    let addr = mock.addr();
    let pending = tokio::spawn(async move {
        call_json(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(&chat("cere-u", 10_000)),
        )
        .await
    });
    eventually("unary request live", || mock.in_flight() == 1).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    // abort_all ignores the rid.
    abort(addr, json!({"rid": "unrelated", "abort_all": true})).await;
    let (status, body) = within(pending).await.unwrap();
    assert_eq!(status, 200);
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "abort");
    let completion = body["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(completion > 0 && completion < 10_000, "{completion}");
    let content = choice["message"]["content"].as_str().unwrap();
    assert_eq!(content.split_whitespace().count() as u64, completion);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn context_overflow_is_an_sglang_style_400() {
    let mut config = MockConfig::default();
    config.server_info.context_length = 10;
    let mock = start(config).await;
    let addr = mock.addr();
    // 5 prompt words + 6 > 10.
    let (status, body) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-o", 6)),
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(body["object"], "error");
    assert_eq!(body["type"], "BadRequestError");
    assert_eq!(body["param"], Value::Null);
    assert_eq!(body["code"], 400);
    let message = body["message"].as_str().unwrap();
    assert!(
        message.contains("is longer than the model's context length"),
        "{message}"
    );
    // 5 + 5 fits exactly.
    let (status, _) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-p", 5)),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(mock.dispatched_rids(), ["cere-o", "cere-p"]);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn fault_server_error_for_next_n_requests() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    mock.set_fault(Fault::ServerError {
        status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        remaining: 2,
    });
    for rid in ["cere-e1", "cere-e2"] {
        let (status, body) =
            call_json(addr, "POST", "/v1/chat/completions", Some(&chat(rid, 1))).await;
        assert_eq!(status, 503);
        assert_eq!(body["object"], "error");
        assert_eq!(body["code"], 503);
    }
    // Readiness endpoints are not affected by chat faults.
    assert_eq!(
        call(addr, "GET", "/health", None).await,
        Ok((200, Vec::new()))
    );
    let (status, _) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-e3", 1)),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(mock.dispatched_rids(), ["cere-e1", "cere-e2", "cere-e3"]);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn fault_mid_stream_disconnect() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    mock.set_fault(Fault::DisconnectAfter { tokens: 2 });
    let mut sse = SseReader::open(addr, &stream_chat("cere-d1", 100, true)).await;
    sse.role().await;
    assert_eq!(sse.token().await, "tok0");
    assert_eq!(sse.token().await, " tok1");
    assert_eq!(within(sse.next()).await, Event::Broken);

    assert_eq!(
        call(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(&chat("cere-d2", 100))
        )
        .await,
        Err(())
    );
    eventually("dropped requests released", || mock.in_flight() == 0).await;
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn fault_hang_before_response_ends_only_at_shutdown() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    mock.set_fault(Fault::HangBeforeResponse);
    let mut conn = Conn::send(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-h", 1)),
    )
    .await;
    eventually("hung request live", || mock.in_flight() == 1).await;
    abort(addr, json!({"rid": "cere-h"})).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(200), conn.head())
            .await
            .is_err(),
        "no response while hung"
    );
    assert_eq!(mock.in_flight(), 1);
    within(mock.shutdown()).await.unwrap();
    let rest = within(async {
        let head = conn.head().await;
        match head {
            Some((_, headers)) => conn.body(&headers).await.map(|_| ()),
            None => Err(()),
        }
    })
    .await;
    assert_eq!(rest, Err(()), "shutdown drops the hung request");
}

#[tokio::test]
async fn fault_hang_after_tokens_stalls_the_stream() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    mock.set_fault(Fault::HangAfterTokens { tokens: 1 });
    let mut sse = SseReader::open(addr, &stream_chat("cere-h2", 100, false)).await;
    sse.role().await;
    assert_eq!(sse.token().await, "tok0");
    abort(addr, json!({"rid": "cere-h2"})).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(200), sse.next())
            .await
            .is_err(),
        "stream stalls"
    );
    assert_eq!(mock.in_flight(), 1);
    within(mock.shutdown()).await.unwrap();
    assert_eq!(within(sse.next()).await, Event::Broken);
}

#[tokio::test]
async fn fault_ignore_abort_keeps_streaming_and_only_affects_later_requests() {
    let mock = start(slow(20, 10_000)).await;
    let addr = mock.addr();
    let mut before = SseReader::open(addr, &stream_chat("cere-x1", 10_000, false)).await;
    before.role().await;
    mock.set_fault(Fault::IgnoreAbort);
    let mut after = SseReader::open(addr, &stream_chat("cere-x2", 5, true)).await;
    after.role().await;
    after.token().await;

    abort(addr, json!({"rid": "cere-x"})).await;
    let (_, finish) = before.until_finish().await;
    assert_eq!(finish["choices"][0]["finish_reason"], "abort");
    before.done().await;

    let (tokens, finish) = after.until_finish().await;
    assert_eq!(tokens.len(), 4, "streams to the budget");
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(after.json().await["usage"]["completion_tokens"], 5);
    after.done().await;
    assert_eq!(mock.abort_requests().len(), 1);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn fault_output_clamp_finishes_length_below_budget() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    mock.set_fault(Fault::ClampOutput { tokens: 2 });
    let (_, body) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-c1", 6)),
    )
    .await;
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 2);

    let mut sse = SseReader::open(addr, &stream_chat("cere-c2", 6, true)).await;
    sse.role().await;
    let (tokens, finish) = sse.until_finish().await;
    assert_eq!(tokens.len(), 2);
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(sse.json().await["usage"]["completion_tokens"], 2);
    sse.done().await;
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn server_info_drift_and_recovery() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    mock.set_server_info_override(Some(ServerInfo {
        served_model_name: "Other-Model".to_owned(),
        context_length: 8,
        kv_cache_dtype: "auto".to_owned(),
        ..ServerInfo::default()
    }));
    let (_, info) = call_json(addr, "GET", "/get_server_info", None).await;
    assert_eq!(
        (
            info["served_model_name"].as_str(),
            info["kv_cache_dtype"].as_str()
        ),
        (Some("Other-Model"), Some("auto"))
    );
    let (_, models) = call_json(addr, "GET", "/v1/models", None).await;
    assert_eq!(models["data"][0]["id"], "Other-Model");
    assert_eq!(models["data"][0]["max_model_len"], 8);
    // The drifted context length governs admission too.
    let (status, _) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-r1", 4)),
    )
    .await;
    assert_eq!(status, 400);

    mock.set_server_info_override(None);
    let (_, info) = call_json(addr, "GET", "/get_server_info", None).await;
    assert_eq!(info["served_model_name"], "Qwen3.8-Flash-Next");
    assert_eq!(info["context_length"], 200_000);
    let (status, _) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-r2", 4)),
    )
    .await;
    assert_eq!(status, 200);
    mock.shutdown().await.unwrap();
}

/// [`MockHandle::set_info_delay`]: `/get_server_info` answers after the delay, with the info in effect once it
/// has passed (an override set while the request waits is the one reported); a cleared delay answers at once.
#[tokio::test]
async fn info_delay_answers_the_info_in_effect_after_the_delay() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    mock.set_info_delay(Duration::from_millis(400));
    let started = std::time::Instant::now();
    let delayed =
        tokio::spawn(async move { call_json(addr, "GET", "/get_server_info", None).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    mock.set_server_info_override(Some(ServerInfo {
        served_model_name: "Other-Model".to_owned(),
        ..ServerInfo::default()
    }));
    let (status, info) = within(delayed).await.unwrap();
    assert_eq!(status, 200);
    assert_eq!(info["served_model_name"], "Other-Model");
    assert!(started.elapsed() >= Duration::from_millis(400));
    mock.set_info_delay(Duration::ZERO);
    mock.set_server_info_override(None);
    let (_, info) = within(call_json(addr, "GET", "/get_server_info", None)).await;
    assert_eq!(
        info["served_model_name"],
        ServerInfo::default().served_model_name
    );
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn max_in_flight_tracks_concurrency_peak() {
    let mock = start(slow(10, 30)).await;
    let addr = mock.addr();
    assert_eq!((mock.in_flight(), mock.max_in_flight()), (0, 0));
    let mut streams = Vec::new();
    for rid in ["cere-m1", "cere-m2", "cere-m3"] {
        let mut sse = SseReader::open(addr, &stream_chat(rid, 100, false)).await;
        sse.role().await;
        streams.push(sse);
    }
    assert_eq!(mock.in_flight(), 3);
    for sse in &mut streams {
        sse.until_finish().await;
        sse.done().await;
    }
    eventually("streams released", || mock.in_flight() == 0).await;
    // Sequential traffic does not raise the peak.
    for rid in ["cere-m4", "cere-m5"] {
        call_json(addr, "POST", "/v1/chat/completions", Some(&chat(rid, 2))).await;
    }
    assert_eq!(mock.max_in_flight(), 3);

    // A client that walks away releases its slot.
    let mut gone = SseReader::open(addr, &stream_chat("cere-m6", 100, false)).await;
    gone.role().await;
    drop(gone);
    eventually("disconnected stream released", || mock.in_flight() == 0).await;
    mock.shutdown().await.unwrap();
}

async fn get_load(addr: SocketAddr) -> Value {
    let (status, load) = call_json(addr, "GET", "/get_load", None).await;
    assert_eq!(status, 200);
    load
}

fn load_of(reqs: u64, waiting: u64) -> Value {
    json!([{"dp_rank": 0, "num_reqs": reqs, "num_waiting_reqs": waiting}])
}

#[tokio::test]
async fn get_load_counts_generating_requests_per_dp_rank() {
    let mut config = slow(20, 10_000);
    config.server_info.max_running_requests = 1;
    let mock = start(config).await;
    let addr = mock.addr();
    assert_eq!(get_load(addr).await, load_of(0, 0));
    let mut first = SseReader::open(addr, &stream_chat("cere-l1", 10_000, false)).await;
    first.role().await;
    first.token().await;
    assert_eq!(get_load(addr).await, load_of(1, 0));
    let mut second = SseReader::open(addr, &stream_chat("cere-l2", 10_000, false)).await;
    second.role().await;
    second.token().await;
    // Beyond max_running_requests a request is waiting; the helpers' sum counts both.
    assert_eq!(get_load(addr).await, load_of(1, 1));
    assert_eq!(mock.load(), 2);
    abort(addr, json!({"rid": "cere-l1"})).await;
    first.until_finish().await;
    eventually("aborted request leaves the load", || mock.load() == 1).await;
    // A client that walks away ends its generation.
    drop(second);
    eventually("disconnected request leaves the load", || mock.load() == 0).await;
    // Rejected before generation: never counted.
    mock.set_fault(Fault::ServerError {
        status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        remaining: 1,
    });
    let (status, _) = call(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-l3", 2)),
    )
    .await
    .unwrap();
    assert_eq!((status, mock.load()), (503, 0));
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn load_lag_holds_a_request_in_transit() {
    let mock = start(slow(10, 3)).await;
    let addr = mock.addr();
    mock.set_load_lag(Duration::from_millis(400));
    let started = std::time::Instant::now();
    let mut conn = Conn::send(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&stream_chat("cere-g1", 10, false)),
    )
    .await;
    eventually("request arrived", || mock.in_flight() == 1).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(150), conn.head())
            .await
            .is_err(),
        "no bytes while in transit"
    );
    assert_eq!(get_load(addr).await, load_of(0, 0));
    let (status, _) = within(conn.head()).await.unwrap();
    assert_eq!(status, 200);
    assert!(started.elapsed() >= Duration::from_millis(400));
    assert_eq!(mock.load(), 1, "counted once it generates");
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn hold_start_holds_requests_in_transit_until_cleared() {
    let mock = start(slow(10, 3)).await;
    let addr = mock.addr();
    mock.hold_start(true);
    let mut conn = Conn::send(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&stream_chat("cere-s1", 10, false)),
    )
    .await;
    eventually("request arrived", || mock.in_flight() == 1).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(150), conn.head())
            .await
            .is_err(),
        "no bytes while held"
    );
    assert_eq!(get_load(addr).await, load_of(0, 0));
    mock.hold_start(false);
    let (status, _) = within(conn.head()).await.unwrap();
    assert_eq!(status, 200);
    assert_eq!(mock.load(), 1, "counted once it generates");
    drop(conn);
    eventually("the first request ended", || mock.in_flight() == 0).await;

    // An abort ends the wait: the request finishes `abort` with no output.
    mock.hold_start(true);
    let pending = tokio::spawn(async move {
        call_json(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(&chat("cere-s2", 10)),
        )
        .await
    });
    eventually("request arrived", || mock.in_flight() == 1).await;
    abort(addr, json!({"rid": "cere-s2"})).await;
    let (status, body) = within(pending).await.unwrap();
    assert_eq!(status, 200);
    assert_eq!(body["choices"][0]["finish_reason"], "abort", "{body}");
    assert_eq!(body["usage"]["completion_tokens"], 0, "{body}");
    assert_eq!(mock.load(), 0);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn hold_finish_keeps_a_generation_at_its_end_until_cleared() {
    let mock = start(slow(10, 3)).await;
    let addr = mock.addr();
    mock.hold_finish(true);
    let mut sse = SseReader::open(addr, &stream_chat("cere-h1", 10, false)).await;
    sse.role().await;
    for _ in 0..3 {
        sse.token().await;
    }
    eventually("held at its finish", || mock.finish_held() == 1).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(150), sse.next())
            .await
            .is_err(),
        "no finish while held"
    );
    assert_eq!(mock.load(), 1, "still generating");
    mock.hold_finish(false);
    let (tokens, finish) = sse.until_finish().await;
    assert!(tokens.is_empty(), "{tokens:?}");
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    sse.done().await;
    assert_eq!((mock.finish_held(), mock.load()), (0, 0));

    // An abort still ends a held generation.
    mock.hold_finish(true);
    let mut sse = SseReader::open(addr, &stream_chat("cere-h2", 10, false)).await;
    sse.role().await;
    eventually("held at its finish", || mock.finish_held() == 1).await;
    abort(addr, json!({"rid": "cere-h2"})).await;
    let (tokens, finish) = sse.until_finish().await;
    assert_eq!(tokens.len(), 3, "{tokens:?}");
    assert_eq!(finish["choices"][0]["finish_reason"], "abort");
    assert_eq!(mock.finish_held(), 0);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn fault_keep_generating_after_disconnect_counts_until_it_would_finish() {
    let mock = start(slow(50, 20)).await;
    let addr = mock.addr();
    mock.set_fault(Fault::KeepGeneratingAfterDisconnect);
    let mut sse = SseReader::open(addr, &stream_chat("cere-k1", 100, false)).await;
    sse.role().await;
    sse.token().await;
    let gone = std::time::Instant::now();
    drop(sse);
    eventually("client gone", || mock.in_flight() == 0).await;
    // SGLang dropped the rid with the connection: the abort is a no-op and the request keeps generating.
    abort(addr, json!({"rid": "cere-k1"})).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(get_load(addr).await, load_of(1, 0));
    eventually("generation ends", || mock.load() == 0).await;
    // 19 tokens were left at 50 ms each.
    assert!(
        gone.elapsed() >= Duration::from_millis(800),
        "{:?}",
        gone.elapsed()
    );

    // An abort while the client is connected still ends it.
    let mut sse = SseReader::open(addr, &stream_chat("cere-k2", 100, false)).await;
    sse.role().await;
    abort(addr, json!({"rid": "cere-k2"})).await;
    let (_, finish) = sse.until_finish().await;
    assert_eq!(finish["choices"][0]["finish_reason"], "abort");
    eventually("aborted request leaves the load", || mock.load() == 0).await;
    mock.shutdown().await.unwrap();
}

/// `end_kept` ends the generations the fault keeps counting at the call; one kept after it counts on.
#[tokio::test]
async fn end_kept_ends_the_generations_kept_at_the_call() {
    // 50 ms per token, 10 000 tokens: a kept generation outlasts the test on its own.
    let mock = start(slow(50, 10_000)).await;
    let addr = mock.addr();
    mock.set_fault(Fault::KeepGeneratingAfterDisconnect);
    let kept = |rid: &'static str| async move {
        let mut sse = SseReader::open(addr, &stream_chat(rid, 10_000, false)).await;
        sse.role().await;
        sse.token().await;
    };
    kept("cere-k3").await;
    eventually("client gone", || mock.in_flight() == 0).await;
    assert_eq!(mock.load(), 1);
    mock.end_kept();
    eventually("the kept generation ends", || mock.load() == 0).await;

    kept("cere-k4").await;
    eventually("client gone", || mock.in_flight() == 0).await;
    assert_eq!(mock.load(), 1, "kept after the call");
    mock.end_kept();
    eventually("the second one ends", || mock.load() == 0).await;
    mock.shutdown().await.unwrap();
}

/// `hold_aborts` holds `/abort_request` until it is cleared: the call is counted as it reaches the hold, the
/// generation streams on and no abort is recorded until then; clearing it lets the abort take effect and answer.
#[tokio::test]
async fn hold_aborts_holds_the_abort_until_cleared() {
    let mock = start(slow(20, 10_000)).await;
    let addr = mock.addr();
    let mut sse = SseReader::open(addr, &stream_chat("cere-h1", 10_000, false)).await;
    sse.role().await;
    sse.token().await;
    mock.hold_aborts(true);
    let held = tokio::spawn(abort(addr, json!({"rid": "cere-h1"})));
    eventually("the abort reached the hold", || {
        mock.held_aborts().len() == 1
    })
    .await;
    assert_eq!(mock.held_aborts()[0].rid, "cere-h1");
    sse.token().await;
    sse.token().await;
    assert!(!held.is_finished(), "the held abort answered");
    assert!(
        mock.abort_requests().is_empty(),
        "the held abort took effect"
    );
    mock.hold_aborts(false);
    held.await.unwrap();
    let (_, finish) = sse.until_finish().await;
    assert_eq!(finish["choices"][0]["finish_reason"], "abort");
    sse.done().await;
    assert_eq!(mock.abort_requests().len(), 1);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn fault_stream_error_is_an_in_stream_error_after_200() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    let message =
        "Input length (199994 tokens) exceeds the maximum allowed length (199994 tokens).";
    mock.set_fault(Fault::StreamError {
        code: 400,
        message: message.to_owned(),
    });
    let mut sse = SseReader::open(addr, &stream_chat("cere-e1", 2, false)).await;
    let error = sse.json().await;
    assert_eq!(
        error,
        json!({"error": {"message": message, "type": "BadRequestError", "param": null, "code": 400}})
    );
    sse.done().await;
    let (status, body) = call_json(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat("cere-e2", 2)),
    )
    .await;
    assert_eq!((status, body["message"].as_str()), (400, Some(message)));
    assert_eq!(
        (body["object"].as_str(), body["code"].as_u64()),
        (Some("error"), Some(400))
    );
    assert_eq!(mock.load(), 0);
    mock.shutdown().await.unwrap();
}

#[tokio::test]
async fn server_info_reports_launch_values_when_set() {
    let mock = start(MockConfig::default()).await;
    let addr = mock.addr();
    let (_, info) = call_json(addr, "GET", "/get_server_info", None).await;
    assert!(info.get("random_seed").is_none() && info.get("startup_time").is_none());
    mock.set_server_info_override(Some(ServerInfo {
        random_seed: Some(42),
        startup_time: Some(json!({"load_weights": 12.5})),
        ..ServerInfo::default()
    }));
    let (_, info) = call_json(addr, "GET", "/get_server_info", None).await;
    assert_eq!(
        (info["random_seed"].clone(), info["startup_time"].clone()),
        (json!(42), json!({"load_weights": 12.5}))
    );
    mock.shutdown().await.unwrap();
}

const EXE: &str = env!("CARGO_BIN_EXE_cere-mock-sglang");

/// A running `cere-mock-sglang` process, killed on drop.
struct MockProcess {
    child: std::process::Child,
    addr: SocketAddr,
}

impl MockProcess {
    /// Spawns the binary with `args` and waits for its banner.
    fn spawn(args: &[&str]) -> MockProcess {
        use std::io::BufRead;
        use std::process::{Command, Stdio};

        let mut child = Command::new(EXE)
            .args(args)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        std::io::BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let base = line
            .trim()
            .strip_prefix("cere-mock-sglang listening on http://")
            .unwrap_or_else(|| panic!("{args:?}: unexpected banner {line:?}"));
        let addr: SocketAddr = base.parse().unwrap();
        assert!(addr.ip().is_loopback() && addr.port() != 0);
        MockProcess { child, addr }
    }
}

impl Drop for MockProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A temporary file removed on drop.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(contents: &str) -> TempFile {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cere-mock-sglang-test-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, contents).unwrap();
        TempFile(path)
    }

    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn unix_secs_f64() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// A probed-looking `/get_server_info` body: every pinned key, the launch keys and keys the mock does not model.
fn probed_server_info() -> Value {
    let mut info = serde_json::to_value(ServerInfo {
        served_model_name: "Replica-Model".to_owned(),
        max_running_requests: 6,
        enable_cache_report: true,
        random_seed: Some(1234),
        startup_time: Some(json!({"total": 40.0})),
        ..ServerInfo::default()
    })
    .unwrap();
    info["tp_size"] = json!(1);
    info["internal_states"] =
        json!([{"last_gen_throughput": 0.0, "memory_usage": {"weight": 12.5}}]);
    info
}

/// `info` without `startup_time`.
fn without_startup_time(mut info: Value) -> Value {
    info.as_object_mut().unwrap().remove("startup_time");
    info
}

#[tokio::test]
async fn binary_prints_its_address_and_stops_on_sigterm() {
    let mut mock = MockProcess::spawn(&["--port", "0"]);
    let addr = mock.addr;
    assert_eq!(
        call(addr, "GET", "/health", None).await,
        Ok((200, Vec::new()))
    );
    let (_, info) = call_json(addr, "GET", "/get_server_info", None).await;
    assert_eq!(info["max_running_requests"], 4, "default cap");

    let killed = std::process::Command::new("kill")
        .args(["-TERM", &mock.child.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    assert!(
        mock.child.wait().unwrap().success(),
        "graceful exit on SIGTERM"
    );
}

#[tokio::test]
async fn binary_serves_the_server_info_file_with_its_own_launch_time() {
    let file_info = probed_server_info();
    let file = TempFile::new(&file_info.to_string());
    let mut startup_times = Vec::new();
    for _ in 0..2 {
        let before = unix_secs_f64();
        let mock = MockProcess::spawn(&["--server-info", file.path()]);
        let after = unix_secs_f64();
        let (status, info) = call_json(mock.addr, "GET", "/get_server_info", None).await;
        assert_eq!(status, 200);
        assert_eq!(
            without_startup_time(info.clone()),
            without_startup_time(file_info.clone()),
            "served info is the file except startup_time"
        );
        let startup_time = info["startup_time"]
            .as_f64()
            .unwrap_or_else(|| panic!("startup_time is the launch time: {info}"));
        assert!(
            (before..=after).contains(&startup_time),
            "{before} <= {startup_time} <= {after}"
        );
        let (_, models) = call_json(mock.addr, "GET", "/v1/models", None).await;
        assert_eq!(
            (&models["data"][0]["id"], &models["data"][0]["root"]),
            (&json!("Replica-Model"), &json!("Replica-Model"))
        );
        startup_times.push(startup_time);
    }
    // Same random_seed, different startup_time: each launch is a new readiness instance_id.
    assert!(startup_times[0] < startup_times[1], "{startup_times:?}");
}

#[tokio::test]
async fn binary_token_latency_spaces_stream_tokens_on_each_mock() {
    const LATENCY: Duration = Duration::from_millis(20);
    const TOKENS: u32 = 5;
    let mocks = [
        MockProcess::spawn(&["--token-latency-ms", "20"]),
        MockProcess::spawn(&["--token-latency-ms", "20"]),
    ];
    // Every output token waits the latency first, so the i-th token cannot arrive before i x 20 ms.
    let stream = |addr: SocketAddr, rid: &'static str| async move {
        let started = std::time::Instant::now();
        let mut reader = SseReader::open(addr, &stream_chat(rid, u64::from(TOKENS), false)).await;
        reader.role().await;
        let mut arrivals = Vec::new();
        for _ in 0..TOKENS {
            reader.token().await;
            arrivals.push(started.elapsed());
        }
        let (rest, finish) = reader.until_finish().await;
        assert!(rest.is_empty());
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        reader.done().await;
        arrivals
    };
    let (a, b) = tokio::join!(
        stream(mocks[0].addr, "cere-lat-a"),
        stream(mocks[1].addr, "cere-lat-b")
    );
    for arrivals in [a, b] {
        for (i, arrival) in (1u32..).zip(&arrivals) {
            assert!(
                *arrival >= LATENCY * i,
                "token {i} at {arrival:?}: {arrivals:?}"
            );
        }
    }
}

#[tokio::test]
async fn binary_reports_and_enforces_max_running_requests() {
    let mock = MockProcess::spawn(&["--max-running-requests", "1", "--token-latency-ms", "100"]);
    let addr = mock.addr;
    let (_, info) = call_json(addr, "GET", "/get_server_info", None).await;
    assert_eq!(info["max_running_requests"], 1);
    let mut first = SseReader::open(addr, &stream_chat("cere-m1", 8, false)).await;
    first.role().await;
    first.token().await;
    let mut second = SseReader::open(addr, &stream_chat("cere-m2", 8, false)).await;
    second.role().await;
    second.token().await;
    assert_eq!(get_load(addr).await, load_of(1, 1));
    drop((first, second));

    // The flag wins over the file's value; without the flag the file's value is served (see the test above).
    let file = TempFile::new(&probed_server_info().to_string());
    let mock = MockProcess::spawn(&["--server-info", file.path(), "--max-running-requests", "2"]);
    let (_, info) = call_json(mock.addr, "GET", "/get_server_info", None).await;
    assert_eq!(info["max_running_requests"], 2);
}

#[tokio::test]
async fn binary_natural_output_tokens_lets_max_tokens_govern() {
    // Default: natural EOS after 8 tokens even with a larger budget.
    let mock = MockProcess::spawn(&[]);
    let mut reader = SseReader::open(mock.addr, &stream_chat("cere-n0", 20, false)).await;
    reader.role().await;
    let (tokens, finish) = reader.until_finish().await;
    assert_eq!(
        (tokens.len(), &finish["choices"][0]["finish_reason"]),
        (8, &json!("stop"))
    );
    // With the flag above the budget, the stream runs to max_tokens and ends `length`.
    let mock = MockProcess::spawn(&["--natural-output-tokens", "100"]);
    let mut reader = SseReader::open(mock.addr, &stream_chat("cere-n1", 20, false)).await;
    reader.role().await;
    let (tokens, finish) = reader.until_finish().await;
    assert_eq!(
        (tokens.len(), &finish["choices"][0]["finish_reason"]),
        (20, &json!("length"))
    );
    reader.done().await;
}

/// The binary must exit 2 with the usage line on stderr.
fn assert_rejected(args: &[&str]) {
    use std::process::{Command, Stdio};

    let mut child = Command::new(EXE)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + LIMIT;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{args:?}: still running, expected exit 2");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut stderr = String::new();
    std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut stderr).unwrap();
    assert_eq!(status.code(), Some(2), "{args:?}: {stderr}");
    assert!(
        stderr.contains("usage: cere-mock-sglang [--port N] [--server-info FILE]"),
        "{args:?}: {stderr}"
    );
}

#[test]
fn binary_rejects_bad_values_and_files_with_exit_2() {
    let invalid_json = TempFile::new("{\"served_model_name\": ");
    let missing_keys = TempFile::new("{\"served_model_name\": \"m\"}");
    let wrong_type = TempFile::new(&{
        let mut info = probed_server_info();
        info["context_length"] = json!("200000");
        info.to_string()
    });
    let not_an_object = TempFile::new("[]");
    let missing_file = format!("{}.absent", invalid_json.path());
    for args in [
        &["--bogus"][..],
        &["--port"],
        &["--port", "70000"],
        &["--port", "1", "--port", "2"],
        &["--token-latency-ms", "x"],
        &["--token-latency-ms", "-1"],
        &["--token-latency-ms"],
        &["--max-running-requests", "0"],
        &["--max-running-requests", "four"],
        &["--natural-output-tokens", "x"],
        &["--natural-output-tokens"],
        &["--server-info"],
        &["--server-info", &missing_file],
        &["--server-info", invalid_json.path()],
        &["--server-info", missing_keys.path()],
        &["--server-info", wrong_type.path()],
        &["--server-info", not_an_object.path()],
    ] {
        assert_rejected(args);
    }
}
