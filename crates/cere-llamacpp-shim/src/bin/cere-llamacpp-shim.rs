//! `cere-llamacpp-shim --port N --upstream URL --server-info FILE [--max-running-requests N]
//! [--trust-single-scope] [--tool-converter PY]`: serves the SGLang subset on `127.0.0.1:N` in front of `llama-server` at URL. See the
//! library docs for the contract. Prints `cere-llamacpp-shim listening on http://127.0.0.1:<port>` on stdout.

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cere_llamacpp_shim as shim;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, watch};

struct Shim {
    client: reqwest::Client,
    upstream: String,
    info: Value,
    model: String,
    context_length: u64,
    cache_report: bool,
    max_running: usize,
    trust_single_scope: bool,
    live: Mutex<HashMap<String, watch::Sender<bool>>>,
    seq: AtomicU64,
    /// The qwen3_coder value converter (`scripts/live/amd/qwen3_coder_convert.py`); `None` refuses tools.
    converter: Option<Arc<Converter>>,
}

/// One long-lived `python3 -I <converter>` child speaking the native sidecar's `tool_args` protocol, one JSON line
/// per request and reply, serialized by the mutex.
struct Converter {
    io: Mutex<(ChildStdin, BufReader<ChildStdout>, Child)>,
}

/// A conversion that SGLang's parse treats as an exception (the whole text, no calls), or a converter failure.
enum ConvertError {
    Exception,
    Gone(String),
}

impl Converter {
    fn spawn(path: &str) -> std::io::Result<Self> {
        let mut child = Command::new("python3")
            .arg("-I")
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(Self {
            io: Mutex::new((stdin, stdout, child)),
        })
    }

    fn tool_args(
        &self,
        tool: Option<&Value>,
        func: &str,
        params: &[(String, String)],
    ) -> Result<String, ConvertError> {
        let req = json!({"op": "tool_args", "tool_json": tool, "func": func, "params": params});
        self.call(&req, "arguments")
    }

    /// The streaming fragment of one closed parameter (`json.dumps(name) + ": " + json.dumps(value)`).
    fn tool_param(
        &self,
        tool: Option<&Value>,
        func: Option<&str>,
        name: &str,
        raw: &str,
    ) -> Result<String, ConvertError> {
        let req =
            json!({"op": "tool_param", "tool_json": tool, "func": func, "name": name, "raw": raw});
        self.call(&req, "fragment")
    }

    fn call(&self, req: &Value, key: &str) -> Result<String, ConvertError> {
        let mut io = self
            .io
            .lock()
            .map_err(|_| ConvertError::Gone("converter lock poisoned".into()))?;
        let (stdin, stdout, _) = &mut *io;
        let mut line = serde_json::to_string(req).map_err(|e| ConvertError::Gone(e.to_string()))?;
        line.push('\n');
        stdin
            .write_all(line.as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|e| ConvertError::Gone(e.to_string()))?;
        let mut reply = String::new();
        if stdout
            .read_line(&mut reply)
            .map_err(|e| ConvertError::Gone(e.to_string()))?
            == 0
        {
            return Err(ConvertError::Gone("converter exited".into()));
        }
        let v: Value =
            serde_json::from_str(&reply).map_err(|e| ConvertError::Gone(e.to_string()))?;
        match (v["ok"].as_bool(), v[key].as_str()) {
            (Some(true), Some(a)) => Ok(a.to_owned()),
            (Some(false), _) if v["error"] != "unknown_op" => Err(ConvertError::Exception),
            _ => Err(ConvertError::Gone(format!("converter reply: {reply}"))),
        }
    }
}

/// A tool call id: `call_` + 24 hex digits, as SGLang's `uuid4().hex[:24]` (the native path's `call_id`).
fn call_id() -> std::io::Result<String> {
    use std::fmt::Write as _;
    let mut b = [0u8; 12];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    let mut id = String::from("call_");
    for x in b {
        let _ = write!(id, "{x:02x}");
    }
    Ok(id)
}

/// The qwen3_coder parse of a unary reply: `(normal text, calls)`, SGLang's `parse_non_stream` with the converter's
/// `tool_args` (a conversion exception = the whole text, no calls).
fn parse_tools(
    conv: &Converter,
    tools: &[Value],
    text: &str,
) -> Result<(String, Vec<shim::ToolCall>), String> {
    let (normal, calls) = cere_runtime::native::tool_parser::parse_non_stream(text, |call| {
        let tool = cere_runtime::native::tool_parser::tool_for(tools, Some(&call.name));
        match conv.tool_args(tool, &call.name, &call.params) {
            Ok(a) => Ok(Some(a)),
            Err(ConvertError::Exception) => Ok(None),
            Err(ConvertError::Gone(e)) => Err(e),
        }
    })?;
    let mut out = Vec::with_capacity(calls.len());
    for c in calls {
        out.push(shim::ToolCall {
            id: call_id().map_err(|e| e.to_string())?,
            index: c.index,
            name: c.name,
            arguments: c.arguments,
        });
    }
    Ok((normal, out))
}

/// Removes a request from the live set when its handler or stream task ends, whatever the reason.
struct LiveGuard {
    shim: Arc<Shim>,
    rid: String,
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        self.shim.live.lock().unwrap().remove(&self.rid);
    }
}

impl Shim {
    fn register(
        self: &Arc<Self>,
        rid: &str,
    ) -> Result<(LiveGuard, watch::Receiver<bool>), Response> {
        let (tx, rx) = watch::channel(false);
        let mut live = self.live.lock().unwrap();
        if live.contains_key(rid) {
            return Err(error(
                StatusCode::BAD_REQUEST,
                &format!("duplicate live rid {rid}"),
            ));
        }
        live.insert(rid.to_owned(), tx);
        Ok((
            LiveGuard {
                shim: Arc::clone(self),
                rid: rid.to_owned(),
            },
            rx,
        ))
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(
            json!({"object": "error", "message": message, "type": "BadRequestError", "param": null,
                         "code": status.as_u16()}),
        ),
    )
        .into_response()
}

fn now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

async fn health(State(s): State<Arc<Shim>>) -> StatusCode {
    match s.client.get(format!("{}/health", s.upstream)).send().await {
        Ok(r) if r.status().is_success() => StatusCode::OK,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    }
}

async fn models(State(s): State<Arc<Shim>>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": s.model, "object": "model", "owned_by": "cere-llamacpp-shim",
                                            "max_model_len": s.context_length}]}),
    )
}

async fn server_info(State(s): State<Arc<Shim>>) -> Json<Value> {
    Json(s.info.clone())
}

async fn get_load(State(s): State<Arc<Shim>>) -> Json<Value> {
    let live = s.live.lock().unwrap().len();
    Json(shim::load(live, s.max_running))
}

async fn abort_request(State(s): State<Arc<Shim>>, Json(body): Json<Value>) -> StatusCode {
    let prefix = body.get("rid").and_then(Value::as_str).unwrap_or("");
    let all = body
        .get("abort_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    for (rid, tx) in s.live.lock().unwrap().iter() {
        if shim::abort_matches(rid, prefix, all) {
            let _ = tx.send(true);
        }
    }
    StatusCode::OK
}

/// Resolves once the request is aborted (or its sender is gone); the borrowed `watch::Ref` never escapes, so
/// callers stay `Send`.
async fn cancelled(rx: &mut watch::Receiver<bool>) {
    let _ = rx.wait_for(|c| *c).await;
}

fn sse(value: &Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

async fn chat(State(s): State<Arc<Shim>>, Json(body): Json<Value>) -> Response {
    let Some(body) = body.as_object().cloned() else {
        return error(StatusCode::BAD_REQUEST, "body is not a JSON object");
    };
    let rid = body
        .get("rid")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            format!(
                "{:032x}",
                s.seq.fetch_add(1, Ordering::Relaxed) as u128 ^ now_s().to_bits() as u128
            )
        });
    if let Some(field) = shim::refused_field(&body) {
        return error(
            StatusCode::BAD_REQUEST,
            &format!("{field} is not supported by this backend (llama-server would ignore it)"),
        );
    }
    let plan = match shim::tool_plan(&body) {
        Ok(p) => p,
        Err(msg) => return error(StatusCode::BAD_REQUEST, &msg),
    };
    let converter = match (&plan, &s.converter) {
        (shim::ToolPlan::Parse(_), None) => {
            return error(
                StatusCode::BAD_REQUEST,
                "tools are not enabled on this shim (no --tool-converter)",
            )
        }
        (_, c) => c.clone(),
    };
    let stream = body.get("stream").and_then(Value::as_bool) == Some(true);
    let client_usage = shim::wants_usage(&body);
    let upstream = shim::apply_tool_plan(shim::upstream_body(body, s.trust_single_scope), &plan);
    let (guard, mut cancel) = match s.register(&rid) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let request = s
        .client
        .post(format!("{}/v1/chat/completions", s.upstream))
        .json(&Value::Object(upstream));

    if !stream {
        let sent = tokio::select! {
            r = request.send() => r,
            _ = cancelled(&mut cancel) => return Json(shim::abort_response(&rid, &s.model, 0)).into_response(),
        };
        let resp = match sent {
            Ok(r) => r,
            Err(e) => return error(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")),
        };
        let status = resp.status();
        let read = tokio::select! {
            r = resp.bytes() => r,
            _ = cancelled(&mut cancel) => return Json(shim::abort_response(&rid, &s.model, 0)).into_response(),
        };
        drop(guard);
        let bytes = match read {
            Ok(b) => b,
            Err(e) => return error(StatusCode::BAD_GATEWAY, &format!("upstream body: {e}")),
        };
        if !status.is_success() {
            return (
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                [(header::CONTENT_TYPE, "application/json")],
                bytes,
            )
                .into_response();
        }
        let v = match serde_json::from_slice::<Value>(&bytes) {
            Ok(v) => shim::reshape_response(v, &rid, &s.model, s.cache_report),
            Err(e) => return error(StatusCode::BAD_GATEWAY, &format!("upstream JSON: {e}")),
        };
        let (shim::ToolPlan::Parse(tools), Some(conv)) = (plan, converter) else {
            return Json(v).into_response();
        };
        let text = v
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let parsed = tokio::task::spawn_blocking(move || parse_tools(&conv, &tools, &text)).await;
        return match parsed {
            Ok(Ok((normal, calls))) => {
                Json(shim::with_tool_calls(v, &normal, &calls)).into_response()
            }
            Ok(Err(e)) => error(StatusCode::BAD_GATEWAY, &format!("tool converter: {e}")),
            Err(e) => error(StatusCode::BAD_GATEWAY, &format!("tool parse task: {e}")),
        };
    }

    let resp = tokio::select! {
        r = request.send() => match r {
            Ok(r) => r,
            Err(e) => return error(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")),
        },
        _ = cancelled(&mut cancel) => {
            let model = s.model.clone();
            let body = format!("data: {}\n\ndata: [DONE]\n\n", shim::abort_chunk(&rid, &model));
            return ([(header::CONTENT_TYPE, "text/event-stream")], body).into_response();
        }
    };
    if !resp.status().is_success() {
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let bytes = resp.bytes().await.unwrap_or_default();
        return (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response();
    }

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let shim_state = Arc::clone(&s);
    // With tools: CERE's qwen3_coder streaming detector over the content, parameters converted by the converter.
    let mut tools_stream = match (plan, converter) {
        (shim::ToolPlan::Parse(tools), Some(conv)) => Some((
            cere_runtime::native::tool_parser::Stream::default(),
            conv,
            Arc::new(tools),
            false, // a call went out
        )),
        _ => None,
    };
    tokio::spawn(async move {
        let _guard = guard; // live until this task ends
        let mut upstream = resp.bytes_stream();
        let mut buf = String::new();
        let mut usage: Option<Value> = None;
        let mut generated: u64 = 0;
        let model = shim_state.model.clone();
        let mut finished = false;
        'read: loop {
            let next = tokio::select! {
                n = upstream.next() => n,
                _ = cancelled(&mut cancel) => {
                    // Dropping `upstream` closes the connection: llama-server cancels the slot.
                    let _ = tx.send(Ok(sse(&shim::abort_chunk(&rid, &model)))).await;
                    if client_usage {
                        let u = usage.take().unwrap_or_else(|| json!({"prompt_tokens": 0,
                            "completion_tokens": generated, "total_tokens": generated, "prompt_tokens_details": null}));
                        let _ = tx.send(Ok(sse(&shim::usage_chunk(&rid, &model, u)))).await;
                    }
                    let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
                    return;
                }
            };
            let Some(piece) = next else { break };
            let Ok(piece) = piece else { break };
            buf.push_str(&String::from_utf8_lossy(&piece));
            while let Some(end) = buf.find("\n\n") {
                let event: String = buf.drain(..end + 2).collect();
                for line in event.lines() {
                    let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                        continue;
                    };
                    if data == "[DONE]" {
                        finished = true;
                        break 'read;
                    }
                    let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };
                    if shim::generated(&chunk) {
                        generated += 1;
                    }
                    let (relay, u) =
                        shim::reshape_chunk(chunk, &rid, &model, shim_state.cache_report);
                    if u.is_some() {
                        usage = u;
                    }
                    let Some(mut relay) = relay else { continue };
                    let Some((parser, conv, tools, called)) = tools_stream.as_mut() else {
                        if tx.send(Ok(sse(&relay))).await.is_err() {
                            return; // client went away: dropping upstream cancels the slot
                        }
                        continue;
                    };
                    // SGLang's order: reasoning, normal text, then one chunk per call item, then the finish.
                    let (content, finish) = shim::take_content_and_finish(&mut relay);
                    let inc = parser.push(&content);
                    if !inc.normal.is_empty() {
                        relay["choices"][0]["delta"]["content"] = json!(inc.normal);
                    }
                    if shim::chunk_has_payload(&relay) && tx.send(Ok(sse(&relay))).await.is_err() {
                        return;
                    }
                    for item in inc.items {
                        let (conv, tools) = (Arc::clone(conv), Arc::clone(tools));
                        let resolved = tokio::task::spawn_blocking(move || {
                            item.resolve(|func, name, raw| {
                                let tool =
                                    cere_runtime::native::tool_parser::tool_for(&tools, func);
                                conv.tool_param(tool, func, name, raw)
                            })
                        })
                        .await;
                        // SGLang's stream fails on any conversion error; so does a dead converter.
                        let Ok(Ok(delta)) = resolved else {
                            let _ = tx
                                .send(Err(std::io::Error::other(
                                    "tool parameter conversion failed",
                                )))
                                .await;
                            return;
                        };
                        let id = match delta.name {
                            Some(_) => match call_id() {
                                Ok(id) => Some(id),
                                Err(e) => {
                                    let _ = tx.send(Err(e)).await;
                                    return;
                                }
                            },
                            None => None,
                        };
                        *called = true;
                        let call = shim::stream_call(id, delta.index, delta.name, delta.arguments);
                        if tx
                            .send(Ok(sse(&shim::chunk_like(&relay, call, Value::Null))))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    if let Some(mut f) = finish {
                        if *called && f == "stop" {
                            f = json!("tool_calls"); // serving_chat.py:1625-1628
                        }
                        let delta = json!({"role": null, "content": null, "reasoning_content": null,
                                           "tool_calls": null});
                        if tx
                            .send(Ok(sse(&shim::chunk_like(&relay, delta, f))))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
        }
        if !finished {
            let _ = tx
                .send(Err(std::io::Error::other(
                    "upstream stream ended without [DONE]",
                )))
                .await;
            return;
        }
        if client_usage {
            if let Some(u) = usage {
                let _ = tx.send(Ok(sse(&shim::usage_chunk(&rid, &model, u)))).await;
            }
        }
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
    });
    let stream =
        futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|v| (v, rx)) });
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn usage_exit(msg: &str) -> ! {
    eprintln!("cere-llamacpp-shim: {msg}\nusage: cere-llamacpp-shim --port N --upstream URL --server-info FILE \
               [--max-running-requests N] [--trust-single-scope] [--tool-converter PY]");
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let (mut port, mut upstream, mut info_file, mut max_running, mut trust) =
        (None, None, None, None, false);
    let mut tool_converter: Option<String> = None;
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .unwrap_or_else(|| usage_exit(&format!("{a} needs a value")))
        };
        match a.as_str() {
            "--port" => {
                port = Some(
                    val()
                        .parse::<u16>()
                        .unwrap_or_else(|_| usage_exit("bad --port")),
                )
            }
            "--upstream" => upstream = Some(val().trim_end_matches('/').to_owned()),
            "--server-info" => info_file = Some(val()),
            "--max-running-requests" => {
                max_running = Some(
                    val()
                        .parse::<usize>()
                        .unwrap_or_else(|_| usage_exit("bad --max-running-requests")),
                )
            }
            "--trust-single-scope" => trust = true,
            "--tool-converter" => tool_converter = Some(val()),
            other => usage_exit(&format!("unknown argument {other}")),
        }
    }
    let port = port.unwrap_or_else(|| usage_exit("--port is required"));
    let upstream = upstream.unwrap_or_else(|| usage_exit("--upstream is required"));
    let info_file = info_file.unwrap_or_else(|| usage_exit("--server-info is required"));
    let text = std::fs::read_to_string(&info_file)
        .unwrap_or_else(|e| usage_exit(&format!("{info_file}: {e}")));
    let mut info: Map<String, Value> = serde_json::from_str(&text)
        .unwrap_or_else(|e| usage_exit(&format!("{info_file}: not a JSON object: {e}")));
    let model = info
        .get("served_model_name")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| usage_exit("server info lacks served_model_name"));
    let context_length = info
        .get("context_length")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| usage_exit("server info lacks an integer context_length"));
    let cache_report = info
        .get("enable_cache_report")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let max_running = max_running
        .or_else(|| {
            info.get("max_running_requests")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
        })
        .unwrap_or_else(|| usage_exit("need max_running_requests (flag or server info)"));
    info.insert("max_running_requests".into(), json!(max_running));
    info.insert("startup_time".into(), json!(now_s()));
    info.entry("random_seed").or_insert_with(|| {
        json!(SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0))
    });
    info.insert(
        "cere_shim".into(),
        json!({"name": "cere-llamacpp-shim", "version": env!("CARGO_PKG_VERSION"),
        "upstream": "llama.cpp llama-server", "trust_single_scope": trust,
        "tool_parser": if tool_converter.is_some() { json!("qwen3_coder (cere detector + SGLang conversion)") } else { Value::Null }}),
    );

    let state = Arc::new(Shim {
        client: reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("http client"),
        upstream,
        info: Value::Object(info),
        model,
        context_length,
        cache_report,
        max_running,
        trust_single_scope: trust,
        live: Mutex::new(HashMap::new()),
        seq: AtomicU64::new(0),
        converter: tool_converter.map(|p| {
            Arc::new(
                Converter::spawn(&p)
                    .unwrap_or_else(|e| usage_exit(&format!("--tool-converter {p}: {e}"))),
            )
        }),
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/get_server_info", get(server_info))
        .route("/get_load", get(get_load))
        .route("/abort_request", post(abort_request))
        .route("/v1/chat/completions", post(chat))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap_or_else(|e| usage_exit(&format!("bind 127.0.0.1:{port}: {e}")));
    println!(
        "cere-llamacpp-shim listening on http://{}",
        listener.local_addr().unwrap()
    );
    let shutdown = async {
        let mut term =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap();
}
