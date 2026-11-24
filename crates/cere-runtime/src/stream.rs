//! Streaming relay (m1 design note A4, L5, L6): backend SSE chunks to the client with the `cere` object.
//!
//! A reader task owns the backend stream and the admission slot: it forwards chunks to the executor and, once
//! the executor stops listening (terminal written), keeps draining until the backend really closes, so the
//! slot stays charged exactly as long as the backend works on the request.
//!
//! Towards the client, events go through a bounded channel of `api.stream_buffer_events + n + 1` (`n` the
//! request's choices, M2.6 D11). Every passthrough event needs `n + 2` free slots, so the error event and `[DONE]`
//! always fit and the `n + 2` terminal events (the finish chunks, the usage chunk, `[DONE]`) fit the channel; the
//! queued passthrough events stay bounded by `api.stream_buffer_events`. A client that leaves no room for
//! `api.stall_timeout_ms` is cancelled `client_backpressure`; the blocked time is recorded in the request timings.
//! A cancel signal or the deadline arriving while the relay waits for room starts that cancel at once.
//!
//! In continue mode (`X-Cere-On-Disconnect: continue`, m1close design note Q3) events go into the request's
//! resume buffer instead ([`crate::resume`]) and the response reads from it: nothing waits for the client, so a
//! stall or disconnect detaches it and the request runs on under its own deadline and output budget. Every
//! event carries `id: <seq>` (its `cere.seq`) in both modes.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderValue};
use axum::response::Response;
use futures_util::stream;
use serde_json::{json, Value};
use tokio::sync::mpsc::{self, error::TrySendError};
use tokio::sync::watch;
use tokio::time::Instant;

use crate::api_types::{ApiError, RequestUsage};
use crate::app::{
    unix_ms, ConfirmedStream, REASON_BACKEND_ABORT, REASON_CANCEL_REQUESTED,
    REASON_CANCEL_UNACKNOWLEDGED, REASON_CLIENT_BACKPRESSURE, REASON_CLIENT_DISCONNECTED,
    REASON_DEADLINE_EXCEEDED,
};
use crate::fed::remote::{RemoteCall, RemoteStream};
use crate::ledger::{Completion, REASON_WORKER_EPOCH_SUPERSEDED};
use crate::resume::{Producer, DONE, KEEPALIVE};
use crate::server::{reason_error, with_request_id, BackendCall, Client, Exec, Slot};
use crate::sglang::{BackendError, ChatStream, SglangHttp};
use crate::surface::{ErrorView, SurfaceStream};
use crate::validate::{assistant_reply, classify_choices, Finish, ToolCallAssembly};

/// Free client-channel slots a passthrough event of a one-choice request needs: itself plus the two terminal
/// events; every further choice adds its finish chunk.
const PASSTHROUGH_SLOTS: usize = 3;

/// One item of a backend chat stream: a chunk, the end (`Ok(None)`), or the error that ended it. The SGLang
/// reader and the native generation (`crate::native::serve`) both feed the relay these.
pub(crate) type BackendItem = Result<Option<Value>, BackendError>;

/// A backend chat stream the relay reads: an SGLang SSE body, or a remote worker's work events, one backend
/// chunk each (Q11).
pub(crate) enum Chunks {
    Sglang(ChatStream),
    Remote(RemoteStream),
}

impl Chunks {
    async fn next_chunk(&mut self) -> BackendItem {
        match self {
            Chunks::Sglang(stream) => stream.next_chunk().await,
            Chunks::Remote(stream) => stream.next_chunk().await,
        }
    }
}

/// Runs a streamed request from dispatch to its terminal.
pub(crate) async fn run(exec: &mut Exec, slot: Slot, client: SglangHttp) {
    let body = exec.backend_body();
    let deadline = exec.backend_deadline();
    let call: BackendCall<Chunks> = Box::pin(async move {
        client
            .chat_stream(&body, deadline)
            .await
            .map(Chunks::Sglang)
    });
    run_call(exec, slot, call).await
}

/// Q11: runs a streamed request on a remote worker from its accepted Submit to its terminal: the work events
/// (open until `until`) relayed exactly like a local backend's chunks.
pub(crate) async fn run_remote(exec: &mut Exec, slot: Slot, call: Arc<RemoteCall>, until: Instant) {
    let call: BackendCall<Chunks> =
        Box::pin(async move { call.open(until).await.map(Chunks::Remote) });
    run_call(exec, slot, call).await
}

async fn run_call(exec: &mut Exec, slot: Slot, mut call: BackendCall<Chunks>) {
    let opened = match exec.drive(&mut call).await {
        None if exec.superseded() => {
            // Q10: the backend request was aborted and its slot belongs to the supersede now.
            drop(call);
            drop(slot);
            let reason = REASON_WORKER_EPOCH_SUPERSEDED;
            let response = exec.failure_response(&reason_error(reason), "failed", reason, None);
            exec.client.respond(response);
            return;
        }
        None => {
            let reason = REASON_CANCEL_UNACKNOWLEDGED;
            exec.settle_failure("failed", reason, None);
            let response = exec.failure_response(&reason_error(reason), "failed", reason, None);
            exec.client.respond(response);
            tokio::spawn(async move {
                if let Ok(mut backend) = call.await {
                    while let Ok(Some(_)) = backend.next_chunk().await {}
                }
                drop(slot);
            });
            return;
        }
        Some(Err(e)) => {
            drop(slot);
            // Nothing was streamed yet: an ordinary HTTP error (e.g. 400 context_overflow).
            exec.backend_failed(&e);
            return;
        }
        Some(Ok(backend)) => backend,
    };
    let (backend_tx, backend_rx) = mpsc::channel(1);
    let signal = exec.cancel_rx.clone();
    tokio::spawn(read_backend(opened, backend_tx, slot, signal));
    relay(exec, backend_rx, None).await;
}

/// Opens the SSE response (unless a cancel already started) and relays `backend` to the terminal. A native
/// stream comes here directly: its generation holds the admission slot until its session is released.
/// `prompt_tokens`: the prompt's token count when known before generating (a local native prompt), for a compat
/// surface's first frames (ApiFit AF-L8).
pub(crate) async fn relay(
    exec: &mut Exec,
    backend: mpsc::Receiver<BackendItem>,
    prompt_tokens: Option<u64>,
) {
    let mut relay = Relay {
        seq: 0,
        content: String::new(),
        tool_calls: ToolCallAssembly::default(),
        finishes: vec![None; exec.request.n as usize],
        usage: None,
        gap: false,
        last_send: Instant::now(),
        first_generated: false,
        first_visible: false,
        created: unix_secs(),
        producer: None,
        confirmed: None,
        surface: exec.surface.stream(),
    };
    if exec.reason.is_none() {
        relay.open(exec, prompt_tokens);
    }
    relay.run(exec, backend).await;
}

/// Forwards backend items until the executor stops listening, then drains the backend to its end. A
/// superseded request (Q10, `signal`) drops the backend stream and its slot at once: the supersede aborted the
/// backend request and took over the slot.
async fn read_backend(
    mut backend: Chunks,
    tx: mpsc::Sender<BackendItem>,
    slot: Slot,
    mut signal: watch::Receiver<Option<&'static str>>,
) {
    let superseded = async move {
        // The executor dropping its sender (it ended) never supersedes the request.
        let superseded = signal
            .wait_for(|reason| *reason == Some(REASON_WORKER_EPOCH_SUPERSEDED))
            .await
            .is_ok();
        if !superseded {
            std::future::pending::<()>().await;
        }
    };
    tokio::pin!(superseded);
    let mut forwarding = true;
    loop {
        let item = tokio::select! {
            item = backend.next_chunk() => item,
            () = &mut superseded => break,
        };
        let end = !matches!(item, Ok(Some(_)));
        if forwarding && tx.send(item).await.is_err() {
            forwarding = false;
        }
        if end {
            break;
        }
    }
    drop(backend);
    drop(slot);
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// One SSE event: `id: <seq>` (the event's `cere.seq`) and its `data:` line.
fn sse(seq: u64, value: &Value) -> Bytes {
    Bytes::from(format!("id: {seq}\ndata: {value}\n\n"))
}

/// A `200 text/event-stream` response over `body` (a chat stream or a resume reader).
pub(crate) fn sse_response(body: Body, rid: &str) -> Response {
    let mut response = Response::new(body);
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    with_request_id(response, rid)
}

/// Waits up to `stall_timeout_ms` for `slots` free events (a passthrough event's, or the terminal events) and
/// records the blocked time. Unless the request is already cancelling, a cancel signal or the deadline ends the
/// wait at once. `Err` carries the cancel cause: client gone, `client_backpressure`, deadline, or `None` for the
/// signal.
async fn wait_for_room<'a>(
    exec: &mut Exec,
    tx: &'a mpsc::Sender<Bytes>,
    slots: usize,
) -> Result<mpsc::PermitIterator<'a, Bytes>, Option<&'static str>> {
    let stall = Duration::from_millis(exec.settings.config.api.stall_timeout_ms);
    let watch = exec.ack_deadline.is_none();
    let deadline = exec.deadline;
    let blocked = Instant::now();
    let waited = tokio::select! {
        biased;
        _ = exec.cancel_rx.changed(), if watch => Err(None),
        _ = tokio::time::sleep_until(deadline), if watch => Err(Some(REASON_DEADLINE_EXCEEDED)),
        reserved = tokio::time::timeout(stall, tx.reserve_many(slots)) => match reserved {
            Ok(Ok(permits)) => Ok(permits),
            Ok(Err(_)) => Err(Some(REASON_CLIENT_DISCONNECTED)),
            Err(_) => Err(Some(REASON_CLIENT_BACKPRESSURE)),
        },
    };
    let ms = blocked.elapsed().as_millis() as u64;
    let _ = exec.app.ledger().add_backpressure(exec.rid_str(), ms);
    waited
}

/// How the relay ended towards the backend.
enum Step {
    Continue,
    Terminal,
}

struct Relay {
    seq: u64,
    /// Assistant content streamed so far (the session commit digests it).
    content: String,
    /// D7: the calls of every chunk's `delta.tool_calls`, the held finish chunk's included.
    tool_calls: ToolCallAssembly,
    /// The backend's finish chunk per choice `index` (M2.6 D11), held until the usage chunk and `[DONE]` decide
    /// the terminal.
    finishes: Vec<Option<Value>>,
    usage: Option<Value>,
    /// Some passthrough chunk did not reach the client (client gone, cancelling, or SSE not opened).
    gap: bool,
    last_send: Instant,
    first_generated: bool,
    first_visible: bool,
    created: u64,
    /// Continue mode: the resume buffer every event goes into (the client reads from it).
    producer: Option<Producer>,
    /// m1close Q6 `C`, a lower bound of cered's share of the backend load: held from a content chunk until the
    /// finish chunk, a cancel or the terminal. A relay about to wait for client room drops it first: while it
    /// waits it reads nothing, so the backend may finish unseen. The next content chunk counts it again.
    confirmed: Option<ConfirmedStream>,
    /// M2.14: a compat surface renders every client event (no `accepted` event, no `[DONE]`).
    surface: Option<Box<dyn SurfaceStream>>,
}

impl Relay {
    /// Sends the 200 SSE response and the `accepted` event. In continue mode the response is the buffer's
    /// first reader and the executor stops watching the client.
    fn open(&mut self, exec: &mut Exec, prompt_tokens: Option<u64>) {
        let mut cere = exec.cere(&RequestUsage::UNAVAILABLE);
        cere.insert("seq".to_owned(), json!(0));
        cere.insert("event".to_owned(), json!("accepted"));
        if !exec.request.ignored_params.is_empty() {
            cere.insert(
                "ignored_params".to_owned(),
                json!(exec.request.ignored_params),
            );
        }
        let accepted = match &mut self.surface {
            Some(surface) => Bytes::from(surface.open(prompt_tokens)),
            None => sse(
                0,
                &json!({
                    "id": exec.rid_str(),
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": exec.request.model,
                    "choices": [],
                    "cere": cere,
                }),
            ),
        };
        self.last_send = Instant::now();
        if exec.continue_on_disconnect {
            let (producer, reader) = exec.app.resume.open(
                exec.rid_str(),
                &exec.scope,
                exec.app.tuning.keepalive_interval,
            );
            producer.push(0, accepted);
            self.producer = Some(producer);
            let response = sse_response(reader.into_body(), exec.rid_str());
            exec.client.respond(response);
            return;
        }
        let capacity =
            exec.settings.config.api.stream_buffer_events as usize + self.passthrough_slots() - 1;
        let (tx, rx) = mpsc::channel::<Bytes>(capacity);
        if !accepted.is_empty() {
            let _ = tx.try_send(accepted);
        }
        let body = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|b| (Ok::<_, Infallible>(b), rx))
        });
        let response = sse_response(Body::from_stream(body), exec.rid_str());
        exec.client.respond(response);
        exec.client = Client::Streaming(tx);
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn streaming(exec: &Exec) -> bool {
        matches!(exec.client, Client::Streaming(_))
    }

    async fn run(&mut self, exec: &mut Exec, mut backend: mpsc::Receiver<BackendItem>) {
        let keepalive = exec.app.tuning.keepalive_interval;
        loop {
            if exec.superseded() {
                // Q10: stop at once; `fail` answers `worker_epoch_superseded` without a terminal of its own.
                exec.begin_cancel(None);
                self.confirmed = None;
                let reason = REASON_WORKER_EPOCH_SUPERSEDED;
                self.fail(exec, "failed", reason, None, None);
                return;
            }
            if exec.ack_deadline.is_some() || self.finished() {
                self.confirmed = None;
            }
            let cancelling = exec.ack_deadline.is_some();
            let ack_at = exec.ack_deadline.unwrap_or(exec.deadline);
            let streaming = Self::streaming(exec);
            let keepalive_at = self.last_send + keepalive;
            let item = tokio::select! {
                biased;
                item = backend.recv() => item,
                _ = exec.cancel_rx.changed(), if !cancelling => {
                    exec.begin_cancel(None);
                    continue;
                }
                _ = tokio::time::sleep_until(exec.deadline), if !cancelling => {
                    exec.begin_cancel(Some(REASON_DEADLINE_EXCEEDED));
                    continue;
                }
                _ = exec.client.closed(), if !cancelling => {
                    exec.begin_cancel(Some(REASON_CLIENT_DISCONNECTED));
                    continue;
                }
                _ = tokio::time::sleep_until(ack_at), if cancelling => {
                    self.fail(exec, "failed", REASON_CANCEL_UNACKNOWLEDGED, None, None);
                    return;
                }
                _ = tokio::time::sleep_until(keepalive_at), if streaming => {
                    self.keepalive(exec);
                    continue;
                }
            };
            let step = match item {
                Some(Ok(Some(chunk))) => self.chunk(exec, chunk).await,
                Some(Ok(None)) => {
                    self.end(exec).await;
                    Step::Terminal
                }
                Some(Err(e)) => {
                    self.backend_error(exec, &e);
                    Step::Terminal
                }
                None => {
                    let e = BackendError::Protocol("backend reader ended".to_owned());
                    self.backend_error(exec, &e);
                    Step::Terminal
                }
            };
            if let Step::Terminal = step {
                return;
            }
        }
    }

    /// Every choice sent its finish chunk.
    fn finished(&self) -> bool {
        self.finishes.iter().all(Option::is_some)
    }

    /// Free client-channel slots a passthrough event needs: itself plus the terminal events after a finish chunk.
    fn passthrough_slots(&self) -> usize {
        self.finishes.len() + PASSTHROUGH_SLOTS - 1
    }

    fn keepalive(&mut self, exec: &Exec) {
        if let Client::Streaming(tx) = &exec.client {
            if tx.capacity() >= self.passthrough_slots() {
                let _ = tx.try_send(Bytes::from_static(KEEPALIVE));
            }
        }
        self.last_send = Instant::now();
    }

    async fn chunk(&mut self, exec: &mut Exec, chunk: Value) -> Step {
        let choices = chunk.get("choices").and_then(Value::as_array);
        if choices.is_some_and(|c| c.is_empty()) {
            if chunk.get("usage").is_some_and(|u| !u.is_null()) {
                self.usage = Some(chunk);
            }
            return Step::Continue;
        }
        // M2.6 D11: every chunk names one of the request's `n` choices, each finishing once; the reply that gets
        // committed (D7) is choice 0's (`n` > 1 never commits).
        let index = chunk.pointer("/choices/0/index").and_then(Value::as_u64);
        let Some(i) = index
            .and_then(|i| usize::try_from(i).ok())
            .filter(|&i| i < self.finishes.len())
        else {
            let e = BackendError::Protocol(format!(
                "chunk of choice {index:?} outside the {} choices",
                self.finishes.len()
            ));
            return self.protocol_error(exec, &e);
        };
        if self.finishes[i].is_some() {
            let e = BackendError::Protocol(format!("chunk of choice {i} after its finish chunk"));
            return self.protocol_error(exec, &e);
        }
        let delta = chunk.pointer("/choices/0/delta");
        let text = |field: &str| {
            delta
                .and_then(|d| d.get(field))
                .and_then(Value::as_str)
                .unwrap_or("")
        };
        let tool_calls = delta.and_then(|d| d.get("tool_calls"));
        let calls = if i == 0 {
            self.content.push_str(text("content"));
            self.tool_calls.push(tool_calls)
        } else {
            tool_calls
                .and_then(Value::as_array)
                .is_some_and(|c| !c.is_empty())
        };
        if chunk
            .pointer("/choices/0/finish_reason")
            .is_some_and(|f| !f.is_null())
        {
            // Relayed at completion, but its delta is part of the reply (D7).
            self.finishes[i] = Some(chunk);
            return Step::Continue;
        }
        if exec.ack_deadline.is_none() && self.confirmed.is_none() {
            self.confirmed = exec.replica().map(|r| r.confirmed.enter());
        }
        let generated =
            !text("content").is_empty() || !text("reasoning_content").is_empty() || calls;
        if generated && !self.first_generated {
            self.first_generated = true;
            let _ = exec
                .app
                .ledger()
                .note_first_generated(exec.rid_str(), unix_ms());
        }
        let mut chunk = chunk;
        let event = match &mut self.surface {
            Some(surface) => Bytes::from(surface.chunk(&chunk)),
            None => {
                let seq = self.seq + 1;
                chunk["cere"] = json!({ "seq": seq, "committed_tokens": null });
                sse(seq, &chunk)
            }
        };
        if event.is_empty() {
            return Step::Continue;
        }
        let seq = self.next_seq();
        if self.send(exec, seq, event).await && generated && !self.first_visible {
            self.first_visible = true;
            let _ = exec
                .app
                .ledger()
                .note_first_visible(exec.rid_str(), unix_ms());
        }
        Step::Continue
    }

    /// Queues one passthrough event, waiting up to `stall_timeout_ms` for room. While cancelling nothing waits.
    /// Returns whether the event was queued; a stall starts the `client_backpressure` cancel, and a cancel signal
    /// or the deadline arriving while blocked starts that cancel at once. In continue mode the event goes into
    /// the resume buffer and nothing waits.
    async fn send(&mut self, exec: &mut Exec, seq: u64, event: Bytes) -> bool {
        if let Some(producer) = &self.producer {
            producer.push(seq, event);
            self.last_send = Instant::now();
            return true;
        }
        let Client::Streaming(tx) = &exec.client else {
            self.gap = true;
            return false;
        };
        let tx = tx.clone();
        let slots = self.passthrough_slots();
        let reserved = match tx.try_reserve_many(slots) {
            Ok(permits) => Ok(permits),
            Err(TrySendError::Closed(())) => Err(Some(REASON_CLIENT_DISCONNECTED)),
            Err(TrySendError::Full(())) if exec.ack_deadline.is_some() => {
                self.gap = true;
                return false;
            }
            Err(TrySendError::Full(())) => {
                self.confirmed = None;
                wait_for_room(exec, &tx, slots).await
            }
        };
        match reserved {
            Ok(mut permits) => {
                if let Some(permit) = permits.next() {
                    permit.send(event);
                }
                self.last_send = Instant::now();
                true
            }
            Err(cause) => {
                self.gap = true;
                exec.begin_cancel(cause);
                false
            }
        }
    }

    fn backend_error(&mut self, exec: &mut Exec, e: &BackendError) {
        // After our abort, the backend ending the stream in any way acknowledges the cancel.
        match exec.reason {
            Some(reason) => self.fail(exec, "cancelled", reason, None, None),
            None => {
                let error = exec.backend_error(e);
                self.fail(exec, "failed", error.code, Some(error), None);
            }
        }
    }

    /// The relay found the backend stream malformed: the request fails and the backend is stopped.
    fn protocol_error(&mut self, exec: &mut Exec, e: &BackendError) -> Step {
        let error = ApiError::from(e);
        self.fail(exec, "failed", error.code, Some(error), None);
        exec.begin_cancel(None);
        Step::Terminal
    }

    /// `[DONE]` arrived: the finish chunks (one per choice) and usage decide the terminal.
    async fn end(&mut self, exec: &mut Exec) {
        if !self.finished() {
            match exec.reason {
                Some(reason) => self.fail(exec, "cancelled", reason, None, None),
                None => {
                    let e =
                        BackendError::Protocol("stream ended without a finish chunk".to_owned());
                    let error = ApiError::from(&e);
                    self.fail(exec, "failed", error.code, Some(error), None);
                }
            }
            return;
        }
        let finishes: Vec<Value> = std::mem::take(&mut self.finishes)
            .into_iter()
            .flatten()
            .collect();
        let usage = self.usage.as_ref().and_then(|u| u.get("usage")).cloned();
        let classified = classify_choices(
            finishes.iter().map(|finish| {
                finish
                    .pointer("/choices/0/finish_reason")
                    .and_then(Value::as_str)
            }),
            usage
                .as_ref()
                .and_then(|u| u.get("completion_tokens"))
                .and_then(Value::as_u64),
            exec.request.max_output_tokens,
        );
        match classified {
            Ok(Finish::Stop | Finish::ToolCalls | Finish::Length | Finish::Repetition) => {
                if let Some(reason) = exec.reason.filter(|_| self.gap) {
                    // The backend finished, but the client did not receive the whole output.
                    self.fail(exec, "cancelled", reason, None, usage.as_ref());
                } else {
                    self.complete(exec, finishes, usage.as_ref()).await;
                }
            }
            Ok(Finish::Abort) => match exec.reason {
                Some(reason) => self.fail(exec, "cancelled", reason, None, usage.as_ref()),
                None => self.fail(exec, "failed", REASON_BACKEND_ABORT, None, usage.as_ref()),
            },
            Ok(Finish::BackendClampedOutput { .. }) => {
                self.fail(
                    exec,
                    "failed",
                    "backend_clamped_output",
                    None,
                    usage.as_ref(),
                );
            }
            Err(e) => {
                let error = ApiError::from(&e);
                self.fail(exec, "failed", error.code, Some(error), usage.as_ref());
            }
        }
    }

    /// The terminal events: the finish chunks in `index` order, the last carrying the terminal `cere`, the
    /// optional usage chunk and `[DONE]`.
    async fn complete(&mut self, exec: &mut Exec, mut finishes: Vec<Value>, usage: Option<&Value>) {
        // Room for the finish chunks, the optional usage chunk and `[DONE]` before anything is committed.
        let Some(mut finish) = finishes.pop() else {
            return self.fail(exec, "failed", "internal_error", None, usage);
        };
        let slots = finishes.len() + PASSTHROUGH_SLOTS;
        let tx = match (&exec.client, &self.producer) {
            (_, Some(_)) => None,
            (Client::Streaming(tx), None) => Some(tx.clone()),
            _ => return self.fail(exec, "failed", "internal_error", None, usage),
        };
        let permits = match &tx {
            None => Ok(None),
            Some(tx) => match tx.try_reserve_many(slots) {
                Ok(permits) => Ok(Some(permits)),
                Err(TrySendError::Closed(())) => Err(Some(REASON_CLIENT_DISCONNECTED)),
                Err(TrySendError::Full(())) => wait_for_room(exec, tx, slots).await.map(Some),
            },
        };
        let permits = match permits {
            Ok(permits) => permits,
            Err(cause) => {
                // The backend is done (no abort needed); the client never got the end of the output.
                let reason = cause
                    .or(*exec.cancel_rx.borrow_and_update())
                    .unwrap_or(REASON_CANCEL_REQUESTED);
                let _ = exec.app.cancel(&exec.scope, exec.rid_str(), reason);
                return self.fail(exec, "cancelled", reason, None, usage);
            }
        };
        let reply = assistant_reply(
            std::mem::take(&mut self.content),
            std::mem::take(&mut self.tool_calls).into_calls(),
        );
        let commit = exec.session_commit(reply);
        let reported = exec.usage(usage, "completed");
        exec.record_usage(&reported);
        let completed = exec.app.ledger().complete(
            exec.rid_str(),
            Completion {
                retained: None,
                session_commit: commit.as_ref().map(|c| c.0),
            },
            unix_ms(),
        );
        if let Err(e) = completed {
            drop(permits);
            let reason = e.code();
            return self.fail(exec, "failed", reason, None, usage);
        }
        exec.app.log_terminal(&exec.scope, exec.rid_str());
        let mut events = Vec::with_capacity(finishes.len() + 2);
        for mut earlier in finishes {
            let seq = self.next_seq();
            let event = match &mut self.surface {
                Some(surface) => Bytes::from(surface.chunk(&earlier)),
                None => {
                    earlier["cere"] = json!({ "seq": seq, "committed_tokens": null });
                    sse(seq, &earlier)
                }
            };
            events.push((seq, event));
        }
        let mut cere = exec.cere(&reported);
        let finish_seq = self.next_seq();
        cere.insert("seq".to_owned(), json!(finish_seq));
        cere.insert("committed_tokens".to_owned(), Value::Null);
        cere.insert("state".to_owned(), json!("completed"));
        if let Some((_, revision, session_id)) = &commit {
            cere.insert("session_id".to_owned(), json!(session_id));
            cere.insert("revision".to_owned(), json!(revision));
        }
        match &mut self.surface {
            Some(surface) => {
                events.push((finish_seq, Bytes::from(surface.complete(&finish, usage))))
            }
            None => {
                finish["cere"] = Value::Object(cere);
                events.push((finish_seq, sse(finish_seq, &finish)));
            }
        }
        if exec.request.include_usage && self.surface.is_none() {
            if let Some(mut chunk) = self.usage.take() {
                let seq = self.next_seq();
                chunk["cere"] = json!({ "seq": seq, "committed_tokens": null });
                events.push((seq, sse(seq, &chunk)));
            }
        }
        match (permits, &mut self.producer) {
            (Some(mut permits), _) => {
                let done = self.surface.is_none().then(|| Bytes::from_static(DONE));
                for event in events.into_iter().map(|(_, e)| e).chain(done) {
                    if let Some(permit) = permits.next() {
                        permit.send(event);
                    }
                }
            }
            (None, Some(producer)) => {
                for (seq, event) in events {
                    producer.push(seq, event);
                }
                producer.finish();
            }
            (None, None) => {}
        };
    }

    /// Writes a failed/cancelled terminal and tells the client: an error event + `[DONE]` on an open stream,
    /// else the HTTP error response. The error event's `cere.usage` is the status API's `usage` (Q12). A
    /// superseded request (Q10) ends `worker_epoch_superseded`, its terminal written by the supersede.
    fn fail(
        &mut self,
        exec: &mut Exec,
        state: &str,
        reason: &str,
        error: Option<ApiError>,
        usage: Option<&Value>,
    ) {
        let (state, reason, error, usage) = if exec.superseded() {
            let reason = REASON_WORKER_EPOCH_SUPERSEDED;
            ("failed", reason, reason_error(reason), None)
        } else {
            (
                state,
                reason,
                error.unwrap_or_else(|| reason_error(reason)),
                usage,
            )
        };
        exec.settle_failure(state, reason, usage);
        if self.producer.is_none() && !Self::streaming(exec) {
            let response = exec.failure_response(&error, state, reason, usage);
            exec.client.respond(response);
            return;
        }
        let seq = self.next_seq();
        let event = match &mut self.surface {
            Some(surface) => {
                let view = ErrorView {
                    status: error.status,
                    kind: error.error_type,
                    code: error.code,
                    message: &error.message,
                    param: error.param.as_deref(),
                };
                Bytes::from(surface.fail(&view))
            }
            None => sse(seq, &error_event(exec, &error, state, reason, seq)),
        };
        if let Some(producer) = &mut self.producer {
            producer.push(seq, event);
            producer.finish();
        } else if let Client::Streaming(tx) = &exec.client {
            // The reservation of every passthrough event leaves room for these two.
            let _ = tx.try_send(event);
            if self.surface.is_none() {
                let _ = tx.try_send(Bytes::from_static(DONE));
            }
        }
    }
}

/// The SSE terminal error event; its `cere.usage` is the status API's `usage` (Q12), its `worker_id` and
/// `worker_epoch` the request's dispatched worker (Q11; `null` before dispatch).
fn error_event(exec: &Exec, error: &ApiError, state: &str, reason: &str, seq: u64) -> Value {
    let status = exec.app.ledger().status(&exec.scope, exec.rid_str());
    let (usage, worker_id, worker_epoch) = status
        .map_or((RequestUsage::UNAVAILABLE, None, None), |s| {
            (s.usage, s.worker_id, s.worker_epoch)
        });
    let mut event = error.body();
    event["cere"] = json!({
        "seq": seq,
        "state": state,
        "reason": reason,
        "usage": usage,
        "worker_id": worker_id,
        "worker_epoch": worker_epoch,
    });
    event
}
