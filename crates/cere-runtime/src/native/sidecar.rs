//! The tokenizer sidecar client (spec m22 Q9, R5): one child process (`scripts/native/tokenizer_sidecar.py` in the
//! native container) that renders the chat template and encodes, so `cered` never tokenizes text itself.
//!
//! Protocol `cere-tokenizer-sidecar-v3` (spec m24 D8; ledger S0): every v2 frame unchanged plus `info.features`, the
//! sorted list of named protocol features the sidecar serves. A frame is a 4-byte little-endian length then that many
//! bytes of UTF-8 JSON; every request gets one reply, in order. `info` returns the composite tokenizer/template
//! hashes, `encode_chat` the prompt token IDs with the stop strings' token counts and the prepared tools,
//! `tool_param`/`tool_args` convert tool-call values with SGLang's own code (D9), and a failure
//! `{"ok": false, "error": {"type", "message", "param"?}}` (the message is fixed text that never quotes input). The
//! client keeps one child at a time (ops-hardening design note OH-01). A child that exits, a closed pipe, a reply
//! outside the protocol or a call that outlasts the I/O deadline (below) stops the child. A child that exits is
//! noticed at once: its pipe reader sees EOF and wakes the respawn thread, so no call or readiness probe has to come
//! first. A respawned child serves once it answered `info` the same as the first. At most one respawn runs at a
//! time: the first after a served call at once, each further one (without a served call in between) after a
//! doubling backoff ([`RESPAWN_BACKOFF_FIRST`] up to [`RESPAWN_BACKOFF_MAX`]), started by the next call or readiness
//! probe. [`Sidecar::health`] follows every change (`Respawning` while a respawn after a served call runs, `Down`
//! while no child serves and none is coming before a backoff ends) and tells the [`Sidecar::watch`] watcher, so
//! readiness publishes it at once. A call whose child died under it (no missed deadline, no reply outside the
//! protocol) runs once more on the successor started at once: every op is pure.
//!
//! I/O deadline (audit N1): a writer thread owns the child's stdin and a reader thread its stdout, so one call
//! (the request write plus the reply, and the attachment frame separately) waits on a channel for at most the
//! deadline (`native.sidecar_deadline_ms`, [`DEFAULT_IO_DEADLINE`]). A child that stops answering without exiting
//! is killed at the deadline, the call fails `Gone`, and the calls queued behind it fail at once while the respawn
//! backoff runs instead of pinning a blocking thread each. [`Sidecar::health`] never waits for a call in flight, so
//! readiness cannot hang.
//!
//! Feature `image` (spec m213 D8/D16/D18, contract `m213-sidecar-image-wire`): started with `--feature image`, the
//! sidecar reports `preprocessor_sha256` in `info` and accepts user messages with text and image parts; an
//! `encode_chat` reply for a request with images carries `images[]` (`grid_thw`, `token_offset`, `pixel_bytes`,
//! `pad_id`) and `attachment_bytes`, then one raw frame of the images' BF16 `pixel_values` concatenated in order.
//!
//! Feature `message_boundary` (spec m210 D4 (c), ledger S): every `encode_chat` reply carries `last_message_start`,
//! the token index where the last message's rendering begins (0 for one message), the source of
//! `cere_work.capture_position` (W 132). The client expects the field exactly when the last `info` advertised it.
//!
//! The grammar process (spec m28 D7, A8) is a second child of the same command with `--role grammar` and its own
//! mutex ([`GrammarSidecar`]), so a compile never blocks `encode_chat`: feature `grammar`, `grammar_open` under a
//! compile deadline, `grammar_step` and `grammar_close` under the same deadline (ops-hardening design note
//! OH-02), masks as base64 little-endian `u32` words. A missed deadline, a dead child or a reply outside the
//! protocol fails only that request; the child is killed and respawned.
use std::fmt;
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, TryLockError, Weak};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::api_types::{ApiError, SERVER_ERROR};
use crate::native::base64;
use crate::native::driver::ImageInput;
use crate::validate::{ChatMessage, Content, ContentPart};

/// The frame protocol the client speaks; any other version (v2 included) is refused.
pub const PROTOCOL: &str = "cere-tokenizer-sidecar-v3";
/// The `info.features` the native profile needs (ledger table S); `info` refuses a sidecar that lacks one. Empty
/// until a feature owner lands.
pub const REQUIRED_FEATURES: &[&str] = &[];
/// The protocol feature that serves image parts (m213 D8).
pub const FEATURE_IMAGE: &str = "image";
/// The protocol feature whose `encode_chat` replies carry `last_message_start` (M2.10 D4 (c), ledger S).
pub const FEATURE_MESSAGE_BOUNDARY: &str = "message_boundary";
/// Longest reply frame read; a longer length is a protocol failure.
const MAX_REPLY_BYTES: usize = 16 << 20;
/// Longest attachment frame read (m213 D8: the images' BF16 pixels): every image at its C2 token cap.
const MAX_ATTACHMENT_BYTES: usize = crate::image::MAX_IMAGES
    * crate::image::MAX_TOKENS_PER_IMAGE as usize
    * crate::image::PIXEL_BYTES_PER_TOKEN as usize;
/// Request frame headroom over `api.max_request_body_bytes` (m213 D18: the JSON envelope around the body).
const REQUEST_FRAME_HEADROOM: u64 = 64 << 10;
/// How long one tokenizer sidecar call may wait for its reply (and its request write) when the config does not say
/// (`native.sidecar_deadline_ms`): far above any encode or image preprocess, so it fires only on a hung child.
pub const DEFAULT_IO_DEADLINE: Duration = Duration::from_secs(120);

/// The sidecar argv `cered` starts: the configured one, plus `--max-request-bytes` = `max_request_body_bytes` +
/// 64 KiB when it serves feature `image` (m213 D18), so any body the API admits fits one request frame.
pub fn argv(configured: &[String], max_request_body_bytes: u64) -> Vec<String> {
    let mut argv = configured.to_vec();
    if serves_image(configured) {
        argv.push("--max-request-bytes".to_owned());
        argv.push((max_request_body_bytes + REQUEST_FRAME_HEADROOM).to_string());
    }
    argv
}

/// `argv` starts the sidecar with `--feature image`.
fn serves_image(argv: &[String]) -> bool {
    argv.windows(2)
        .any(|w| w[0] == "--feature" && w[1] == FEATURE_IMAGE)
        || argv
            .iter()
            .any(|a| a.strip_prefix("--feature=") == Some(FEATURE_IMAGE))
}

/// The `info` reply: the composite tokenizer/template hashes (docs/CONFIGURATION.md lock definitions), the
/// library versions the sidecar loaded and the protocol features it serves (sorted, unique).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SidecarInfo {
    pub tokenizer_sha256: String,
    pub template_sha256: String,
    pub transformers: String,
    pub tokenizers: String,
    pub features: Vec<String>,
    /// With feature `image`: the SHA-256 of the sidecar's `preprocessor_config.json` (checked against the
    /// artifact manifest); `None` without the feature.
    pub preprocessor_sha256: Option<String>,
}

/// Why a sidecar call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SidecarError {
    /// The child could not be started.
    Spawn(String),
    /// The child exited or its pipes closed: not ready.
    Gone(String),
    /// A reply outside the protocol; the child was stopped.
    Protocol(String),
    /// A typed error reply (`type`, fixed message, the request path for `invalid_value`); the child keeps serving.
    Refused {
        kind: String,
        message: String,
        param: Option<String>,
    },
}

impl fmt::Display for SidecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(m) => write!(f, "tokenizer sidecar failed to start: {m}"),
            Self::Gone(m) => write!(f, "tokenizer sidecar is gone: {m}"),
            Self::Protocol(m) => write!(f, "tokenizer sidecar protocol error: {m}"),
            Self::Refused { kind, message, .. } => {
                write!(f, "tokenizer sidecar refused ({kind}): {message}")
            }
        }
    }
}

impl std::error::Error for SidecarError {}

struct Io {
    child: Child,
    /// Request frames for the writer thread, which owns the child's stdin.
    requests: Sender<Vec<u8>>,
    /// Reply frames from the reader thread, which owns the child's stdout (and ends at its EOF or a bad frame).
    replies: Receiver<Result<Vec<u8>, SidecarError>>,
    /// Raised by the reader thread when it ends: the child is gone (or out of sync) even before `try_wait` sees it.
    ended: Arc<AtomicBool>,
    /// How long one request write plus one reply frame may take.
    deadline: Duration,
    /// When the child was found gone or out of sync and stopped; `None` while it serves. Never cleared: a respawn
    /// replaces the whole `Io`.
    stopped_at: Option<Instant>,
    /// A call missed the deadline: the child hung (stopped for that, not for an exit).
    missed_deadline: bool,
    /// The last `info` reply advertised [`FEATURE_MESSAGE_BOUNDARY`]: `encode_chat` replies must carry
    /// `last_message_start`, and replies without the feature must not.
    message_boundary: bool,
}

impl Io {
    /// Starts `argv` (program and arguments) with piped stdin/stdout and its pipe threads; its stderr is inherited
    /// (the sidecar writes only a startup failure there). When the child's stdout ends the reader thread wakes the
    /// respawn thread of `sidecar`, if it still exists.
    fn start(
        argv: &[String],
        deadline: Duration,
        sidecar: Weak<Shared>,
    ) -> Result<Io, SidecarError> {
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| SidecarError::Spawn("the sidecar command is empty".to_owned()))?;
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| SidecarError::Spawn(format!("{program}: {e}")))?;
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let mut stdout = child.stdout.take().expect("stdout is piped");
        // A sidecar without feature `image` never sends an attachment frame: its replies keep the reply cap.
        let reader_cap = if serves_image(argv) {
            MAX_REPLY_BYTES.max(MAX_ATTACHMENT_BYTES)
        } else {
            MAX_REPLY_BYTES
        };
        let (replies_in, replies) = mpsc::channel();
        let (requests, requests_out) = mpsc::channel::<Vec<u8>>();
        let write_failed = replies_in.clone();
        let ended = Arc::new(AtomicBool::new(false));
        let reader_ended = ended.clone();
        let spawned = std::thread::Builder::new()
            .name("cere-sidecar-writer".to_owned())
            .spawn(move || {
                for frame in requests_out {
                    if let Err(e) = stdin.write_all(&frame).and_then(|()| stdin.flush()) {
                        let _ = write_failed
                            .send(Err(SidecarError::Gone(format!("writing a request: {e}"))));
                        break;
                    }
                }
            })
            .and_then(|_| {
                std::thread::Builder::new()
                    .name("cere-sidecar-reader".to_owned())
                    .spawn(move || {
                        loop {
                            let frame = read_frame(&mut stdout, reader_cap);
                            let end = frame.is_err();
                            if replies_in.send(frame).is_err() || end {
                                break;
                            }
                        }
                        // Raised before the wake-up: the respawn thread checks it after it lowers `respawning`, so a
                        // loss that found it running is never missed.
                        reader_ended.store(true, Ordering::SeqCst);
                        if let Some(shared) = sidecar.upgrade() {
                            respawn_in_background(&shared);
                        }
                    })
            });
        if let Err(e) = spawned {
            let _ = child.kill();
            let _ = child.wait();
            return Err(SidecarError::Spawn(format!("the pipe threads: {e}")));
        }
        Ok(Io {
            child,
            requests,
            replies,
            ended,
            deadline,
            stopped_at: None,
            missed_deadline: false,
            message_boundary: false,
        })
    }

    fn is_dead(&self) -> bool {
        self.stopped_at.is_some()
    }

    /// Stopped without missing the deadline: the child exited, was killed, closed a pipe or went out of sync.
    fn exited(&self) -> bool {
        self.is_dead() && !self.missed_deadline
    }

    /// Its reader ended (the child is gone) but nothing stopped it yet.
    fn ended_unseen(&self) -> bool {
        !self.is_dead() && self.ended.load(Ordering::SeqCst)
    }

    /// The child is running and in sync; a child found gone (exited, or its reader ended) is stopped.
    fn running(&mut self) -> bool {
        if !self.is_dead()
            && (self.ended.load(Ordering::SeqCst) || !matches!(self.child.try_wait(), Ok(None)))
        {
            self.stop();
        }
        !self.is_dead()
    }

    /// Stops the child for good (kill is a no-op on one that already exited). Killing it closes its pipes, which
    /// ends the reader (EOF) and unblocks a writer stuck on a full pipe (EPIPE).
    fn stop(&mut self) {
        self.stopped_at.get_or_insert_with(Instant::now);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn exchange(&mut self, request: &Value) -> Result<Value, SidecarError> {
        if self.is_dead() {
            return Err(SidecarError::Gone("the child is not running".to_owned()));
        }
        if self.requests.send(frame_bytes(request)).is_err() {
            self.stop();
            return Err(SidecarError::Gone(
                "writing a request: the writer ended".to_owned(),
            ));
        }
        let reply = self.read_frame(MAX_REPLY_BYTES)?;
        let reply = parse_reply(request, &reply);
        if let Err(SidecarError::Protocol(_)) = reply {
            self.stop();
        }
        reply
    }

    /// Waits at most the deadline for one frame of at most `cap` bytes; a closed pipe, a failed request write, a
    /// missed deadline or a longer frame stops the child.
    fn read_frame(&mut self, cap: usize) -> Result<Vec<u8>, SidecarError> {
        let frame = match self.replies.recv_timeout(self.deadline) {
            Ok(Ok(frame)) if frame.len() <= cap => Ok(frame),
            Ok(Ok(frame)) => Err(SidecarError::Protocol(format!(
                "a reply frame of {} bytes",
                frame.len()
            ))),
            Ok(Err(e)) => Err(e),
            Err(RecvTimeoutError::Timeout) => {
                self.missed_deadline = true;
                Err(SidecarError::Gone(format!(
                    "no reply within {} ms",
                    self.deadline.as_millis()
                )))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(SidecarError::Gone("the reply reader ended".to_owned()))
            }
        };
        if frame.is_err() {
            self.stop();
        }
        frame
    }

    /// The `info` op; a reply that is not this protocol's v3 frame with every required feature stops the child.
    fn info(&mut self) -> Result<SidecarInfo, SidecarError> {
        let reply = self.exchange(&json!({"op": "info"}))?;
        let text = |key: &str| reply.get(key).and_then(Value::as_str).map(str::to_owned);
        if text("protocol").as_deref() != Some(PROTOCOL) {
            self.stop();
            return Err(SidecarError::Protocol(format!(
                "the sidecar does not speak {PROTOCOL}"
            )));
        }
        let Some(features) = features(&reply) else {
            self.stop();
            return Err(SidecarError::Protocol(
                "an info reply lacks a sorted features list".to_owned(),
            ));
        };
        if let Some(missing) = REQUIRED_FEATURES
            .iter()
            .find(|need| !features.iter().any(|f| f == *need))
        {
            self.stop();
            return Err(SidecarError::Protocol(format!(
                "the sidecar lacks the feature {missing}"
            )));
        }
        match (
            text("tokenizer_sha256"),
            text("template_sha256"),
            text("transformers"),
            text("tokenizers"),
        ) {
            (
                Some(tokenizer_sha256),
                Some(template_sha256),
                Some(transformers),
                Some(tokenizers),
            ) => {
                let image = features.iter().any(|f| f == FEATURE_IMAGE);
                let preprocessor_sha256 = text("preprocessor_sha256");
                if image && preprocessor_sha256.is_none() {
                    self.stop();
                    return Err(SidecarError::Protocol(
                        "an info reply with feature image lacks preprocessor_sha256".to_owned(),
                    ));
                }
                self.message_boundary = features.iter().any(|f| f == FEATURE_MESSAGE_BOUNDARY);
                Ok(SidecarInfo {
                    tokenizer_sha256,
                    template_sha256,
                    transformers,
                    tokenizers,
                    features,
                    preprocessor_sha256: preprocessor_sha256.filter(|_| image),
                })
            }
            _ => {
                self.stop();
                Err(SidecarError::Protocol(
                    "an info reply lacks a hash or a version".to_owned(),
                ))
            }
        }
    }
}

impl Drop for Io {
    fn drop(&mut self) {
        self.stop();
    }
}

/// One request as a frame: its 4-byte little-endian length, then the JSON.
fn frame_bytes(request: &Value) -> Vec<u8> {
    let mut frame = vec![0u8; 4];
    serde_json::to_writer(&mut frame, request).expect("a JSON value serializes");
    let length = u32::try_from(frame.len() - 4).expect("requests are far below 4 GiB");
    frame[..4].copy_from_slice(&length.to_le_bytes());
    frame
}

/// Writes `request` as one frame.
fn send(stdin: &mut ChildStdin, request: &Value) -> std::io::Result<()> {
    stdin.write_all(&frame_bytes(request))?;
    stdin.flush()
}

/// Reads one frame of at most `cap` bytes: `Gone` on a closed pipe, `Protocol` on a longer length.
fn read_frame(stdout: &mut ChildStdout, cap: usize) -> Result<Vec<u8>, SidecarError> {
    let mut header = [0u8; 4];
    stdout
        .read_exact(&mut header)
        .map_err(|e| SidecarError::Gone(format!("reading a reply: {e}")))?;
    let length = u32::from_le_bytes(header) as usize;
    if length > cap {
        return Err(SidecarError::Protocol(format!(
            "a reply frame of {length} bytes"
        )));
    }
    let mut reply = vec![0u8; length];
    stdout
        .read_exact(&mut reply)
        .map_err(|e| SidecarError::Gone(format!("reading a reply: {e}")))?;
    Ok(reply)
}

/// The reply to `request`: the ok object for its op, a typed `Refused`, or `Protocol` (the caller stops the child).
fn parse_reply(request: &Value, reply: &[u8]) -> Result<Value, SidecarError> {
    let reply: Value = serde_json::from_slice(reply)
        .map_err(|_| SidecarError::Protocol("a reply frame is not JSON".to_owned()))?;
    match reply.get("ok") {
        Some(Value::Bool(true)) if reply.get("op") == request.get("op") => Ok(reply),
        Some(Value::Bool(false)) => {
            let text = |key: &str| {
                reply
                    .pointer(&format!("/error/{key}"))
                    .and_then(Value::as_str)
            };
            match (text("type"), text("message")) {
                (Some(kind), Some(message)) => Err(SidecarError::Refused {
                    kind: kind.to_owned(),
                    message: message.to_owned(),
                    param: text("param").map(str::to_owned),
                }),
                _ => Err(SidecarError::Protocol(
                    "an error reply without type and message".to_owned(),
                )),
            }
        }
        _ => Err(SidecarError::Protocol(
            "a reply is neither ok for the op nor an error".to_owned(),
        )),
    }
}

/// The chat-template kwargs of an `encode_chat` frame beyond v2 (feature `thinking`, ledger S; spec m212 D4, D13):
/// `enable_thinking` is sent only when true, `reasoning_effort` only when the request has one and
/// `preserve_thinking` only when the request sent false (api-lenient design note AL-2, D3), so a request without
/// them is the v2 frame byte for byte.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateKwargs {
    pub enable_thinking: bool,
    pub reasoning_effort: Option<Value>,
    pub preserve_thinking: Option<bool>,
}

/// An `encode_chat` reply.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    /// The prompt token IDs.
    pub ids: Vec<u32>,
    /// Per stop string, in order: its token count (`encode(s, add_special_tokens=False)`, spec m24 F9).
    pub stop_token_lens: Vec<u32>,
    /// The request's tools after SGLang's validation and normalization (`model_dump()`), when it has tools.
    pub tools_prepared: Option<Vec<Value>>,
    /// The request's images in request order (feature `image`): span, `pad_id` and BF16 pixels; empty without
    /// image parts.
    pub images: Vec<ImageInput>,
    /// Feature `message_boundary` (M2.10 D4 (c), ledger S): the token index where the last message's rendering
    /// begins under the same template kwargs, at most `ids.len()`; 0 for one message, with images, or when the
    /// sidecar does not serve the feature (no checkpoint hint).
    pub last_message_start: u32,
}

/// The image parts of one message.
fn image_count(message: &ChatMessage) -> usize {
    match &message.content {
        Some(Content::Parts(parts)) => parts
            .iter()
            .filter(|p| matches!(p, ContentPart::ImageUrl { .. }))
            .count(),
        _ => 0,
    }
}

/// One `images[]` entry: `grid_thw`, `token_offset`, `pixel_bytes`, `pad_id`.
fn image_entry(entry: &Value) -> Option<([u32; 3], u64, u64, u32)> {
    let grid = u32_list(entry.get("grid_thw"))?;
    let grid_thw: [u32; 3] = grid.try_into().ok()?;
    let token_offset = entry.get("token_offset")?.as_u64()?;
    let pixel_bytes = entry.get("pixel_bytes")?.as_u64()?;
    let pad_id = u32::try_from(entry.get("pad_id")?.as_u64()?).ok()?;
    Some((grid_thw, token_offset, pixel_bytes, pad_id))
}

/// Reads the `images[]` of a reply to a request with `sent` images and the attachment frame that follows it,
/// splitting the frame into each image's pixels. Any mismatch (count, sizes, a frame longer than its cap) is
/// outside the protocol and stops the child.
fn read_images(io: &mut Io, reply: &Value, sent: usize) -> Result<Vec<ImageInput>, SidecarError> {
    let entries = reply
        .get("images")
        .and_then(Value::as_array)
        .and_then(|list| list.iter().map(image_entry).collect::<Option<Vec<_>>>())
        .filter(|list| list.len() == sent);
    let total = entries.as_ref().and_then(|list| {
        list.iter()
            .try_fold(0u64, |sum, entry| sum.checked_add(entry.2))
    });
    let attachment = reply.get("attachment_bytes").and_then(Value::as_u64);
    let (Some(entries), Some(total)) = (entries, total) else {
        io.stop();
        return Err(SidecarError::Protocol(
            "an encode_chat reply lacks one well-formed images entry per image".to_owned(),
        ));
    };
    if attachment != Some(total) {
        io.stop();
        return Err(SidecarError::Protocol(
            "an encode_chat reply's attachment_bytes is not the sum of its pixel_bytes".to_owned(),
        ));
    }
    let mut frame = io.read_frame(MAX_ATTACHMENT_BYTES)?;
    if frame.len() as u64 != total {
        io.stop();
        return Err(SidecarError::Protocol(format!(
            "an attachment frame of {} bytes, the reply announced {total}",
            frame.len()
        )));
    }
    // Split from the back: every later image's pixels are copied out once and the frame, truncated, becomes the
    // first image's (a one-image request moves the frame without a copy).
    let mut images = Vec::with_capacity(entries.len());
    for (grid_thw, token_offset, pixel_bytes, pad_id) in entries.into_iter().rev() {
        let start = frame.len() - pixel_bytes as usize;
        let pixels = if start == 0 {
            std::mem::take(&mut frame)
        } else {
            frame.split_off(start)
        };
        images.push(ImageInput::new(grid_thw, token_offset, pad_id, pixels));
    }
    images.reverse();
    Ok(images)
}

fn u32_list(value: Option<&Value>) -> Option<Vec<u32>> {
    value?
        .as_array()?
        .iter()
        .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
        .collect()
}

/// An `info` reply's `features`: strings, sorted and unique.
fn features(reply: &Value) -> Option<Vec<String>> {
    reply
        .get("features")?
        .as_array()?
        .iter()
        .map(|f| f.as_str().map(str::to_owned))
        .collect::<Option<Vec<String>>>()
        .filter(|list| list.windows(2).all(|w| w[0] < w[1]))
}

/// The wait before the first respawn after the last served call: none, so a child that hung once costs one request
/// and one that died costs none (the call runs again on its successor). Each further respawn without a served call
/// in between waits twice as long after the previous loss, from [`RESPAWN_BACKOFF_FIRST`] up to
/// [`RESPAWN_BACKOFF_MAX`] (ops-hardening design note OH-01).
pub const RESPAWN_BACKOFF_FIRST: Duration = Duration::from_millis(250);
/// The longest respawn backoff: a sidecar whose every child fails is retried this often.
pub const RESPAWN_BACKOFF_MAX: Duration = Duration::from_secs(30);

/// The wait before respawn attempt `attempt` (1-based, counted since the last served call) after the loss of the
/// previous child: 0 for the first, then [`RESPAWN_BACKOFF_FIRST`] doubling up to [`RESPAWN_BACKOFF_MAX`].
fn respawn_backoff(attempt: u32) -> Duration {
    match attempt {
        0 | 1 => Duration::ZERO,
        n => RESPAWN_BACKOFF_FIRST
            .saturating_mul(1 << (n - 2).min(16))
            .min(RESPAWN_BACKOFF_MAX),
    }
}

/// Counters of the tokenizer sidecar (the `/metrics` respawn counter, ops-hardening design note OH-05).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SidecarStats {
    /// Respawn attempts since the sidecar was spawned (one whose child failed to start counts too).
    pub respawns: u64,
}

/// The tokenizer sidecar's health, which readiness publishes (ops-hardening design note OH-01).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarHealth {
    /// A child is running and in sync.
    Ready,
    /// The child was lost after a served call and its successor is starting at once: calls wait for it, and a call
    /// whose child died under it runs again there.
    Respawning,
    /// No child serves and none is coming before a backoff ends: the last respawn failed, its child was lost before
    /// it served, or the sidecar never answered its first `info`. The next call or readiness probe after the backoff
    /// tries again.
    Down,
}

/// The child and its respawn state, under the call lock.
struct Slot {
    /// The current child; `None` after a respawn whose child failed to start.
    io: Option<Io>,
    /// Respawn attempts since the last served call (the backoff exponent).
    attempts: u32,
    /// When the last respawn attempt failed (its child never served).
    failed_at: Option<Instant>,
}

impl Slot {
    /// How long until a respawn may start: the backoff of the next attempt after the last loss.
    fn backoff_left(&self) -> Duration {
        let lost = match &self.io {
            Some(io) => io.stopped_at,
            None => self.failed_at,
        };
        lost.map_or(Duration::ZERO, |at| {
            (at + respawn_backoff(self.attempts + 1)).saturating_duration_since(Instant::now())
        })
    }
}

struct Shared {
    argv: Vec<String>,
    deadline: Duration,
    slot: Mutex<Slot>,
    /// The health of `slot`, set under the call lock (the slot lock first, then this one) whenever it changes and
    /// read without the call lock, which a call in flight holds for up to the deadline.
    health: Mutex<SidecarHealth>,
    /// Called with the new health after every change, in order (changes happen under the call lock).
    watcher: OnceLock<Box<dyn Fn(SidecarHealth) + Send + Sync>>,
    /// A respawn thread has not finished.
    respawning: AtomicBool,
    respawns: AtomicU64,
    /// The first `info` answer; a respawned child must answer the same.
    info: OnceLock<SidecarInfo>,
    /// Set when the client is dropped: a respawn thread that has not started its child yet starts none.
    closed: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lock_health(&self) -> MutexGuard<'_, SidecarHealth> {
        self.health.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The successor of a lost child starts at once: the sidecar answered its first `info` and a call was served
    /// since the last respawn.
    fn respawns_at_once(&self, slot: &Slot) -> bool {
        slot.attempts == 0 && self.info.get().is_some()
    }

    /// Sets the health of `slot` (under the call lock) and tells the watcher when it changed: `Ready` while its
    /// child serves, else `Respawning` when the successor starts at once (on the respawn thread, unless a call
    /// starts it first), else `Down`.
    fn settle(self: &Arc<Self>, slot: &Slot) {
        let health = if slot.io.as_ref().is_some_and(|io| !io.is_dead()) {
            SidecarHealth::Ready
        } else if self.respawns_at_once(slot) {
            SidecarHealth::Respawning
        } else {
            SidecarHealth::Down
        };
        let before = std::mem::replace(&mut *self.lock_health(), health);
        if before == health {
            return;
        }
        if let Some(watcher) = self.watcher.get() {
            watcher(health);
        }
        if health == SidecarHealth::Respawning {
            respawn_in_background(self);
        }
    }
}

/// The live child for the next call: the running one, else a respawned one once the backoff allows (OH-01). A
/// child found gone is stopped and its loss settled first; while the backoff runs the call fails `Gone` at once.
fn revive<'a>(shared: &Arc<Shared>, slot: &'a mut Slot) -> Result<&'a mut Io, SidecarError> {
    if !slot.io.as_mut().is_some_and(Io::running) {
        shared.settle(slot);
        let wait = slot.backoff_left();
        if !wait.is_zero() {
            return Err(SidecarError::Gone(format!(
                "no live child; the next respawn is due in {} ms",
                wait.as_millis()
            )));
        }
        respawn(shared, slot)?;
    }
    slot.io
        .as_mut()
        .ok_or_else(|| SidecarError::Gone("the child is not running".to_owned()))
}

/// Replaces the child in `slot` with a new one that answered `info` equal to the first child's, then settles the
/// health. Runs under the call lock, so at most one respawn is in flight.
fn respawn(shared: &Arc<Shared>, slot: &mut Slot) -> Result<(), SidecarError> {
    slot.io = None;
    slot.attempts += 1;
    let attempt = shared.respawns.fetch_add(1, Ordering::SeqCst) + 1;
    let started =
        Io::start(&shared.argv, shared.deadline, Arc::downgrade(shared)).and_then(|mut io| {
            let info = io.info()?;
            match shared.info.get() {
                Some(first) if *first != info => Err(SidecarError::Protocol(
                    "the respawned tokenizer sidecar reports another info".to_owned(),
                )),
                _ => Ok(io),
            }
        });
    let result = match started {
        Ok(io) => {
            slot.io = Some(io);
            eprintln!("cered: the tokenizer sidecar respawned (respawn {attempt})");
            Ok(())
        }
        Err(e) => {
            slot.failed_at = Some(Instant::now());
            eprintln!("cered: the tokenizer sidecar respawn {attempt} failed: {e}");
            Err(e)
        }
    };
    shared.settle(slot);
    result
}

/// Starts the respawn thread unless one runs or the sidecar was dropped. Once a call in flight let go of the call
/// lock it stops a child found gone, settles the loss and respawns when the backoff allows: at once after a served
/// call; a backoff not over yet waits for the next call or readiness probe.
fn respawn_in_background(shared: &Arc<Shared>) {
    if shared.closed.load(Ordering::SeqCst) || shared.respawning.swap(true, Ordering::SeqCst) {
        return;
    }
    let respawn = shared.clone();
    let spawned = std::thread::Builder::new()
        .name("cere-sidecar-respawn".to_owned())
        .spawn(move || {
            let mut slot = respawn.lock();
            loop {
                if respawn.closed.load(Ordering::SeqCst) {
                    respawn.respawning.store(false, Ordering::SeqCst);
                    return;
                }
                if respawn.info.get().is_some() {
                    // A failed attempt is logged by `respawn`.
                    let _ = revive(&respawn, &mut slot);
                } else if !slot.io.as_mut().is_some_and(Io::running) {
                    // Before its first `info` was answered a lost child is the starter's failure (readiness starts
                    // the sidecar again): settled, never respawned here.
                    respawn.settle(&slot);
                }
                respawn.respawning.store(false, Ordering::SeqCst);
                // A child whose reader ended after `revive` looked found this thread running and started none.
                let missed = slot.io.as_ref().is_some_and(Io::ended_unseen);
                if !missed || respawn.respawning.swap(true, Ordering::SeqCst) {
                    return;
                }
            }
        });
    if let Err(e) = spawned {
        eprintln!("cered: the tokenizer sidecar respawn thread did not start: {e}");
        shared.respawning.store(false, Ordering::SeqCst);
    }
}

/// The sidecar child of a native backend. Calls are serialized (one request on the pipe at a time) and block, each
/// for at most the I/O deadline; async callers run them on a blocking thread. A child that exits, misses the
/// deadline or answers outside the protocol is replaced (ops-hardening design note OH-01): at once after a served
/// call, else by the next call or readiness probe after a bounded backoff ([`RESPAWN_BACKOFF_FIRST`] ..
/// [`RESPAWN_BACKOFF_MAX`]), at most one respawn in flight.
pub struct Sidecar {
    shared: Arc<Shared>,
}

impl fmt::Debug for Sidecar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sidecar")
            .field("health", &*self.shared.lock_health())
            .finish()
    }
}

impl Sidecar {
    /// Starts `argv` (program and arguments) with piped stdin/stdout; its stderr is inherited (the sidecar writes
    /// only a startup failure there). Every call waits at most `deadline` for its request write and then for its
    /// reply (and again for an image attachment frame); a child that misses it is killed. A respawned child is
    /// started from the same `argv` and checked by `info` under the same deadline.
    pub fn spawn(argv: &[String], deadline: Duration) -> Result<Sidecar, SidecarError> {
        let shared = Arc::new(Shared {
            argv: argv.to_vec(),
            deadline,
            slot: Mutex::new(Slot {
                io: None,
                attempts: 0,
                failed_at: None,
            }),
            health: Mutex::new(SidecarHealth::Ready),
            watcher: OnceLock::new(),
            respawning: AtomicBool::new(false),
            respawns: AtomicU64::new(0),
            info: OnceLock::new(),
            closed: AtomicBool::new(false),
        });
        shared.lock().io = Some(Io::start(argv, deadline, Arc::downgrade(&shared))?);
        Ok(Sidecar { shared })
    }

    /// Runs `op` on the live child (respawned first when needed) and settles the health. A served call (a reply, or
    /// a typed refusal) resets the respawn backoff. When the child died under `op` (it exited or went out of sync,
    /// not a missed deadline) and its successor starts at once, `op` runs once more on that successor: every op is
    /// pure, so the caller sees one call.
    fn call<T>(
        &self,
        mut op: impl FnMut(&mut Io) -> Result<T, SidecarError>,
    ) -> Result<T, SidecarError> {
        let mut slot = self.shared.lock();
        let mut retried = false;
        loop {
            let io = revive(&self.shared, &mut slot)?;
            let result = op(io);
            let died = matches!(result, Err(SidecarError::Gone(_))) && io.exited();
            if matches!(result, Ok(_) | Err(SidecarError::Refused { .. })) {
                slot.attempts = 0;
                slot.failed_at = None;
            }
            self.shared.settle(&slot);
            if died && !retried && self.shared.respawns_at_once(&slot) {
                retried = true;
                continue;
            }
            return result;
        }
    }

    /// The health now (readiness: `Respawning` is advisory, `Down` not ready). Never waits for a call in flight;
    /// see [`Sidecar::with_health`].
    pub fn health(&self) -> SidecarHealth {
        self.with_health(|health| health)
    }

    /// Runs `f` on the health with every change held off until it returns, so what `f` publishes is never followed
    /// by an older state; `f` must not call this sidecar. First the readiness probe's part in respawning: unless a
    /// call in flight holds the call lock (that call finds out itself), a child found gone is stopped and its loss
    /// settled, and once the backoff is over its successor starts on the respawn thread.
    pub fn with_health<R>(&self, f: impl FnOnce(SidecarHealth) -> R) -> R {
        let slot = match self.shared.slot.try_lock() {
            Ok(slot) => Some(slot),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        };
        if let Some(mut slot) = slot {
            if !slot.io.as_mut().is_some_and(Io::running) {
                self.shared.settle(&slot);
                if slot.backoff_left().is_zero() {
                    respawn_in_background(&self.shared);
                }
            }
        }
        f(*self.shared.lock_health())
    }

    /// Sets the watcher, called with the new health after every change of [`Sidecar::health`], in order (readiness
    /// probes again at once). Set once; a later call is ignored.
    pub fn watch(&self, watcher: impl Fn(SidecarHealth) + Send + Sync + 'static) {
        let _ = self.shared.watcher.set(Box::new(watcher));
    }

    /// The respawn counter (OH-05 `/metrics`).
    pub fn stats(&self) -> SidecarStats {
        SidecarStats {
            respawns: self.shared.respawns.load(Ordering::SeqCst),
        }
    }

    /// The `info` op. The first answer is recorded (under the call lock, so a child lost right after it is
    /// respawned): every respawned child must answer the same.
    pub fn info(&self) -> Result<SidecarInfo, SidecarError> {
        self.call(|io| {
            let info = io.info()?;
            let _ = self.shared.info.set(info.clone());
            Ok(info)
        })
    }

    /// The `encode_chat` op: the prompt token IDs of `messages` under the chat template (generation prompt, thinking
    /// as `template` asks, thinking disabled by default), `tools_json` (the request's raw `tools`) validated and
    /// prepared, rendered iff `render_tools`, and the token count of each `stop` string. With image parts (feature
    /// `image`) the IDs are expanded (one `<|image_pad|>` per merged patch) and each image comes back with its span
    /// and BF16 pixels, read from the attachment frame that follows the reply. With feature `message_boundary`
    /// (advertised by the last `info`) the reply carries `last_message_start`; without it, that field is refused.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        tools_json: Option<&str>,
        render_tools: bool,
        stop: &[String],
        template: &TemplateKwargs,
    ) -> Result<Encoded, SidecarError> {
        let mut request = json!({
            "op": "encode_chat",
            "messages": messages,
            "render_tools": render_tools,
            "stop": stop,
        });
        if let Some(tools) = tools_json {
            request["tools_json"] = json!(tools);
        }
        if template.enable_thinking {
            request["enable_thinking"] = json!(true);
        }
        if let Some(effort) = &template.reasoning_effort {
            request["reasoning_effort"] = effort.clone();
        }
        if let Some(preserve) = template.preserve_thinking {
            request["preserve_thinking"] = json!(preserve);
        }
        self.call(|io| {
            let reply = io.exchange(&request)?;
            let ids = u32_list(reply.get("ids"));
            let lens = u32_list(reply.get("stop_token_lens")).filter(|l| l.len() == stop.len());
            let tools_prepared = match reply.get("tools_prepared") {
                None | Some(Value::Null) if tools_json.is_none() => Some(None),
                Some(Value::Array(tools)) if tools_json.is_some() => Some(Some(tools.clone())),
                _ => None,
            };
            let boundary = match reply.get("last_message_start") {
                None if !io.message_boundary => Some(0),
                Some(start) if io.message_boundary => start
                    .as_u64()
                    .filter(|&s| ids.as_ref().is_some_and(|ids| s <= ids.len() as u64))
                    .and_then(|s| u32::try_from(s).ok()),
                _ => None,
            };
            let (Some(ids), Some(stop_token_lens), Some(tools_prepared), Some(last_message_start)) =
                (ids, lens, tools_prepared, boundary)
            else {
                io.stop();
                return Err(SidecarError::Protocol(
                    "an encode_chat reply lacks its token IDs, stop token counts, prepared tools or a valid \
                     last_message_start (exactly with feature message_boundary)"
                        .to_owned(),
                ));
            };
            let sent = messages.iter().map(image_count).sum::<usize>();
            let images = if sent == 0 {
                if reply.get("images").is_some() || reply.get("attachment_bytes").is_some() {
                    io.stop();
                    return Err(SidecarError::Protocol(
                        "an encode_chat reply without images carries image fields".to_owned(),
                    ));
                }
                Vec::new()
            } else {
                read_images(io, &reply, sent)?
            };
            Ok(Encoded {
                ids,
                stop_token_lens,
                tools_prepared,
                images,
                last_message_start,
            })
        })
    }

    /// The `tool_param` op: the streaming fragment `json.dumps(name): json.dumps(value)` of one closed parameter
    /// (`qwen3_coder_detector.py:356-365`), its value converted on `tool` (the prepared tool named `func`, `None`
    /// when no tool has that name).
    pub fn tool_param(
        &self,
        tool: Option<&Value>,
        func: Option<&str>,
        name: &str,
        raw: &str,
    ) -> Result<String, SidecarError> {
        let request =
            json!({"op": "tool_param", "tool_json": tool, "func": func, "name": name, "raw": raw});
        self.text_op(&request, "fragment")
    }

    /// The `tool_args` op: the unary `arguments` of one call, `json.dumps` of its converted parameters (:227).
    pub fn tool_args(
        &self,
        tool: Option<&Value>,
        func: &str,
        params: &[(String, String)],
    ) -> Result<String, SidecarError> {
        let request = json!({"op": "tool_args", "tool_json": tool, "func": func, "params": params});
        self.text_op(&request, "arguments")
    }

    fn text_op(&self, request: &Value, key: &str) -> Result<String, SidecarError> {
        self.call(|io| {
            let reply = io.exchange(request)?;
            match reply.get(key).and_then(Value::as_str) {
                Some(text) => Ok(text.to_owned()),
                None => {
                    io.stop();
                    Err(SidecarError::Protocol(format!(
                        "a {} reply lacks its {key}",
                        request["op"].as_str().unwrap_or("tool")
                    )))
                }
            }
        })
    }
}

impl Drop for Sidecar {
    /// The child is killed when the last holder of the shared state lets go: here, at the end of a respawn thread
    /// still running, or in a pipe reader that is waking one.
    fn drop(&mut self) {
        self.shared.closed.store(true, Ordering::SeqCst);
    }
}

/// The `info.features` entry the grammar process serves (spec m28 D7).
pub const GRAMMAR_FEATURE: &str = "grammar";
/// The `info.features` entry of a grammar process serving `grammar_drafts` (GrammarSpec GS-03); `grammar_drafts =
/// "on"` needs it.
pub const GRAMMAR_DRAFTS_FEATURE: &str = "grammar_drafts";
/// Appended to the sidecar command for the grammar process.
const GRAMMAR_ROLE: [&str; 2] = ["--role", "grammar"];
/// `grammar_open`'s compile deadline: SGLang's grammar wait, 10,000 × 5 ms (spec m28 D7, F13). It also bounds a
/// child's startup `info` (the warm-up of `TokenizerInfo` and the compiler) and every `grammar_step` and
/// `grammar_close` (OH-02): a child that misses it is killed.
pub const GRAMMAR_COMPILE_DEADLINE: Duration = Duration::from_secs(50);
/// The fixed text of a `grammar_open` that missed the deadline or lost its child (400 `invalid_value`).
pub const GRAMMAR_TIMEOUT_MESSAGE: &str = "Grammar compilation timed out";

/// The grammar process's `info` beyond the v3 frame: the loaded library versions and the tokenizer facts its
/// matchers were built with (`TokenizerInfo(vocab_size, stop_token_ids)`), for the serve loop's startup check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarInfo {
    pub xgrammar: String,
    pub torch: String,
    pub stop_token_ids: Vec<u32>,
    pub vocab_size: u32,
    /// The process serves `grammar_drafts` ([`GRAMMAR_DRAFTS_FEATURE`]).
    pub drafts: bool,
}

impl GrammarInfo {
    /// The length of every mask, in `u32` words: ⌈vocab_size / 32⌉.
    pub fn mask_words(&self) -> usize {
        (self.vocab_size as usize).div_ceil(32)
    }
}

/// A `grammar_open` request: the request's constraint as raw JSON text (the client's bytes: `response_format`,
/// or the JSON string token of `regex`/`ebnf`), or its tools with the raw `tool_choice`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GrammarOpen<'a> {
    pub response_format: Option<&'a str>,
    pub regex: Option<&'a str>,
    pub ebnf: Option<&'a str>,
    pub tools_json: Option<&'a str>,
    pub tool_choice: Option<&'a str>,
    pub parallel_tool_calls: Option<bool>,
    /// The request enables thinking (spec m28 D15, ledger CR20): the grammar process defers the grammar until the
    /// think-end token (all-ones masks before it). Sent only when true, so other frames are unchanged.
    pub enable_thinking: bool,
}

impl GrammarOpen<'_> {
    /// The request field a failed compile names: the constraint field, else `tool_choice` (a tool constraint).
    pub fn param(&self) -> &'static str {
        if self.response_format.is_some() {
            "response_format"
        } else if self.regex.is_some() {
            "regex"
        } else if self.ebnf.is_some() {
            "ebnf"
        } else {
            "tool_choice"
        }
    }

    fn request(&self) -> Value {
        let mut request = json!({"op": "grammar_open"});
        for (key, value) in [
            ("response_format", self.response_format),
            ("regex", self.regex),
            ("ebnf", self.ebnf),
            ("tools_json", self.tools_json),
            ("tool_choice", self.tool_choice),
        ] {
            if let Some(value) = value {
                request[key] = json!(value);
            }
        }
        if let Some(parallel) = self.parallel_tool_calls {
            request["parallel_tool_calls"] = json!(parallel);
        }
        if self.enable_thinking {
            request["enable_thinking"] = json!(true);
        }
        request
    }
}

/// An open grammar of one grammar child; it dies with that child.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarHandle {
    pub grammar: u64,
    /// SGLang's cache key type (`json`, `regex`, `ebnf`, `structural_tag`) and the SHA-256 of the key.
    pub key_type: String,
    pub key_sha256: String,
    /// The child that owns it.
    generation: u64,
}

/// A `grammar_step` reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrammarStep {
    pub accepted: bool,
    pub terminated: bool,
    /// The next mask was written to the caller's buffer (false once terminated or for a rejected token).
    pub has_mask: bool,
}

/// Why a grammar call failed; the HTTP mapping is the caller's (spec m28 D7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrammarError {
    /// `grammar_open` missed the compile deadline or its child died: 400 `invalid_value` on `param` with
    /// [`GRAMMAR_TIMEOUT_MESSAGE`]. The child was killed and is being respawned.
    TimedOut { param: &'static str },
    /// The child already holds its 4 open grammars: 503 `overloaded`.
    Overloaded,
    /// No grammar child is running (respawning, or the respawn failed).
    NotReady,
    /// The handle's child died or was replaced (or died during this call): the constrained request fails with 500
    /// `internal_error`.
    Lost(String),
    /// A typed error reply (a compile failure is `invalid_value` with the request field as `param`); the child keeps
    /// serving.
    Refused {
        kind: String,
        message: String,
        param: Option<String>,
    },
    /// A `grammar_open` reply outside the protocol; the child was killed and is being respawned.
    Protocol(String),
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedOut { .. } => f.write_str(GRAMMAR_TIMEOUT_MESSAGE),
            Self::Overloaded => f.write_str("the grammar sidecar has no free grammar handle"),
            Self::NotReady => f.write_str("the grammar sidecar is not running"),
            Self::Lost(m) => write!(f, "the grammar sidecar lost the grammar: {m}"),
            Self::Refused { kind, message, .. } => {
                write!(f, "grammar sidecar refused ({kind}): {message}")
            }
            Self::Protocol(m) => write!(f, "grammar sidecar protocol error: {m}"),
        }
    }
}

impl std::error::Error for GrammarError {}

impl GrammarError {
    /// The HTTP answer of a constrained request (spec m28 D7): a missed deadline or a lost child at `grammar_open`
    /// is 400 `invalid_value` on the request field with [`GRAMMAR_TIMEOUT_MESSAGE`], a compile failure 400
    /// `invalid_value` on the field the sidecar names (its fixed text), a 5th handle 503 `overloaded`, no running
    /// child 503 `not_ready`; anything else (a lost handle, a reply outside the protocol, another refusal) is
    /// cered's fault, 500 `internal_error`.
    pub fn api_error(&self) -> ApiError {
        match self {
            Self::TimedOut { param } => ApiError::invalid_value(*param, GRAMMAR_TIMEOUT_MESSAGE),
            Self::Refused {
                kind,
                message,
                param: Some(param),
            } if kind == "invalid_value" => ApiError::invalid_value(param.clone(), message.clone()),
            Self::Overloaded => {
                ApiError::new(503, SERVER_ERROR, "overloaded", self.to_string(), None)
            }
            Self::NotReady => ApiError::new(503, SERVER_ERROR, "not_ready", self.to_string(), None),
            Self::Lost(_) | Self::Refused { .. } | Self::Protocol(_) => {
                ApiError::new(500, SERVER_ERROR, "internal_error", self.to_string(), None)
            }
        }
    }
}

/// A typed refusal of the grammar child (it keeps serving).
fn refused(kind: String, message: String, param: Option<String>) -> GrammarError {
    if kind == "overloaded" {
        GrammarError::Overloaded
    } else {
        GrammarError::Refused {
            kind,
            message,
            param,
        }
    }
}

/// One grammar child. A reader thread owns its stdout and forwards each reply frame, so a reply can be awaited
/// under a deadline; dropping it kills the child (the reader then sees EOF and ends).
struct GrammarIo {
    child: Child,
    stdin: ChildStdin,
    replies: Receiver<Result<Vec<u8>, SidecarError>>,
    generation: u64,
}

impl GrammarIo {
    /// Spawns `argv --role grammar` and checks its `info` within `deadline`.
    fn start(
        argv: &[String],
        generation: u64,
        xgrammar: &str,
        deadline: Duration,
    ) -> Result<(GrammarIo, GrammarInfo), SidecarError> {
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| SidecarError::Spawn("the sidecar command is empty".to_owned()))?;
        let mut child = Command::new(program)
            .args(args)
            .args(GRAMMAR_ROLE)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| SidecarError::Spawn(format!("{program}: {e}")))?;
        let stdin = child.stdin.take().expect("stdin is piped");
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let (sender, replies) = mpsc::channel();
        let reader = std::thread::Builder::new()
            .name("cere-grammar-reader".to_owned())
            .spawn(move || loop {
                let frame = read_frame(&mut stdout, MAX_REPLY_BYTES);
                let end = frame.is_err();
                if sender.send(frame).is_err() || end {
                    break;
                }
            });
        let mut io = GrammarIo {
            child,
            stdin,
            replies,
            generation,
        };
        if let Err(e) = reader {
            return Err(SidecarError::Spawn(format!("the reply reader: {e}")));
        }
        let info = io.info(xgrammar, deadline)?;
        Ok((io, info))
    }

    /// One request and its reply within `deadline`. Any error but `Refused` leaves the child unusable.
    fn exchange(&mut self, request: &Value, deadline: Duration) -> Result<Value, SidecarError> {
        send(&mut self.stdin, request)
            .map_err(|e| SidecarError::Gone(format!("writing a request: {e}")))?;
        let frame = self.replies.recv_timeout(deadline).map_err(|e| match e {
            RecvTimeoutError::Timeout => {
                SidecarError::Gone(format!("no reply within {} ms", deadline.as_millis()))
            }
            RecvTimeoutError::Disconnected => {
                SidecarError::Gone("the reply reader ended".to_owned())
            }
        })??;
        parse_reply(request, &frame)
    }

    /// The v3 frame with feature `grammar` and the dialect's xgrammar version (spec m28 D7 "Versions").
    fn info(&mut self, xgrammar: &str, deadline: Duration) -> Result<GrammarInfo, SidecarError> {
        let reply = self.exchange(&json!({"op": "info"}), deadline)?;
        let text = |key: &str| reply.get(key).and_then(Value::as_str).map(str::to_owned);
        if text("protocol").as_deref() != Some(PROTOCOL) {
            return Err(SidecarError::Protocol(format!(
                "the grammar sidecar does not speak {PROTOCOL}"
            )));
        }
        let Some(features) = features(&reply).filter(|f| f.iter().any(|f| f == GRAMMAR_FEATURE))
        else {
            return Err(SidecarError::Protocol(format!(
                "the grammar sidecar lacks the feature {GRAMMAR_FEATURE}"
            )));
        };
        let reported = text("xgrammar").unwrap_or_default();
        if reported != xgrammar {
            return Err(SidecarError::Protocol(format!(
                "the grammar sidecar reports xgrammar {reported:?}, the profile dialect pins {xgrammar:?}"
            )));
        }
        let vocab_size = reply
            .get("vocab_size")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0);
        match (
            text("torch"),
            u32_list(reply.get("stop_token_ids")),
            vocab_size,
        ) {
            (Some(torch), Some(stop_token_ids), Some(vocab_size)) => Ok(GrammarInfo {
                xgrammar: reported,
                torch,
                stop_token_ids,
                vocab_size,
                drafts: features.iter().any(|f| f == GRAMMAR_DRAFTS_FEATURE),
            }),
            _ => Err(SidecarError::Protocol(
                "a grammar info reply lacks torch, stop_token_ids or vocab_size".to_owned(),
            )),
        }
    }
}

impl Drop for GrammarIo {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct GrammarShared {
    argv: Vec<String>,
    /// The profile dialect's xgrammar version.
    xgrammar: String,
    deadline: Duration,
    /// The first child's info; a respawned child must report the same.
    info: GrammarInfo,
    /// The running child; `None` while respawning or after a failed respawn.
    io: Mutex<Option<GrammarIo>>,
    /// A child is installed and in sync (read without the lock, which a compile holds).
    up: AtomicBool,
    respawning: AtomicBool,
    /// Set when the client is dropped: a respawn in flight discards its child.
    closed: AtomicBool,
    generations: AtomicU64,
}

impl GrammarShared {
    fn lock(&self) -> MutexGuard<'_, Option<GrammarIo>> {
        self.io.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Kills the child in `slot` and respawns one on a background thread (readiness: `Respawning` until it answered
/// `info`). A respawned child that fails its `info` leaves the grammar process `Down` for good.
fn fail(shared: &Arc<GrammarShared>, slot: &mut Option<GrammarIo>) {
    // Raised before `up` drops, so readiness never sees a transient `Down`.
    let respawning = shared.respawning.swap(true, Ordering::SeqCst);
    shared.up.store(false, Ordering::SeqCst);
    *slot = None;
    if respawning || shared.closed.load(Ordering::SeqCst) {
        return;
    }
    let respawn = shared.clone();
    let spawned = std::thread::Builder::new()
        .name("cere-grammar-respawn".to_owned())
        .spawn(move || {
            let generation = respawn.generations.fetch_add(1, Ordering::SeqCst) + 1;
            let started = GrammarIo::start(
                &respawn.argv,
                generation,
                &respawn.xgrammar,
                respawn.deadline,
            )
            .and_then(|(io, info)| {
                if info == respawn.info {
                    Ok(io)
                } else {
                    Err(SidecarError::Protocol(
                        "the respawned grammar sidecar reports another info".to_owned(),
                    ))
                }
            });
            // Audit N4: `respawning` clears in the critical section that installs the child. `fail` runs under
            // this lock too, so it either sees the old state (no child to fail) or the new one (`up`, not
            // `respawning`): it can never find `respawning` still raised for a child that is already running and
            // then skip the respawn, which left the process `Down` for good.
            let mut slot = respawn.lock();
            match started {
                Ok(io) => {
                    if !respawn.closed.load(Ordering::SeqCst) {
                        *slot = Some(io);
                        respawn.up.store(true, Ordering::SeqCst);
                    }
                }
                Err(e) => eprintln!("cered: the grammar sidecar did not respawn: {e}"),
            }
            respawn.respawning.store(false, Ordering::SeqCst);
            drop(slot);
        });
    if let Err(e) = spawned {
        eprintln!("cered: the grammar sidecar did not respawn: {e}");
        shared.respawning.store(false, Ordering::SeqCst);
    }
}

/// The grammar process as readiness reports it (spec m28 D7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrammarHealth {
    /// A child is running and in sync.
    Ready,
    /// The child was killed (deadline, death, protocol error); its successor has not answered `info` yet.
    Respawning,
    /// The respawn failed: no child for good, every constrained request is refused.
    Down,
}

/// The grammar process of a native backend whose profile advertises a grammar output (spec m28 D7): `<sidecar
/// argv> --role grammar`, its own mutex, one request on the pipe at a time; calls block (async callers run them on a
/// blocking thread).
pub struct GrammarSidecar {
    shared: Arc<GrammarShared>,
}

impl fmt::Debug for GrammarSidecar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrammarSidecar")
            .field("health", &self.health())
            .finish()
    }
}

impl GrammarSidecar {
    /// Starts `argv --role grammar` and checks its `info`: protocol v3, feature `grammar`, `xgrammar` equal to
    /// `xgrammar` (the profile dialect's version), `torch`, `stop_token_ids` and `vocab_size` present. `deadline`
    /// bounds every `grammar_open` and every child's startup `info` ([`GRAMMAR_COMPILE_DEADLINE`] in production).
    pub fn spawn(
        argv: &[String],
        xgrammar: &str,
        deadline: Duration,
    ) -> Result<GrammarSidecar, SidecarError> {
        let (io, info) = GrammarIo::start(argv, 1, xgrammar, deadline)?;
        Ok(GrammarSidecar {
            shared: Arc::new(GrammarShared {
                argv: argv.to_vec(),
                xgrammar: xgrammar.to_owned(),
                deadline,
                info,
                io: Mutex::new(Some(io)),
                up: AtomicBool::new(true),
                respawning: AtomicBool::new(false),
                closed: AtomicBool::new(false),
                generations: AtomicU64::new(1),
            }),
        })
    }

    /// The startup `info` (every respawned child reports the same).
    pub fn info(&self) -> &GrammarInfo {
        &self.shared.info
    }

    /// The grammar readiness: [`GrammarHealth::Ready`]. Never waits for a compile in flight; a child found exited is
    /// killed and respawned.
    pub fn ready(&self) -> bool {
        self.health() == GrammarHealth::Ready
    }

    /// The grammar process's state; a child found exited is killed and respawned (never waits for a compile in
    /// flight, which holds the lock).
    pub fn health(&self) -> GrammarHealth {
        let slot = match self.shared.io.try_lock() {
            Ok(slot) => Some(slot),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        };
        if let Some(mut slot) = slot {
            if slot
                .as_mut()
                .is_some_and(|io| !matches!(io.child.try_wait(), Ok(None)))
            {
                fail(&self.shared, &mut slot);
            }
        }
        if self.shared.up.load(Ordering::SeqCst) {
            GrammarHealth::Ready
        } else if self.shared.respawning.load(Ordering::SeqCst) {
            GrammarHealth::Respawning
        } else {
            GrammarHealth::Down
        }
    }

    /// The running child's process id.
    pub fn pid(&self) -> Option<u32> {
        self.shared.lock().as_ref().map(|io| io.child.id())
    }

    /// `grammar_open`: compiles `open` within the deadline and writes the mask of position 0 into `mask_out`
    /// (cleared, its allocation reused).
    pub fn grammar_open(
        &self,
        open: &GrammarOpen<'_>,
        mask_out: &mut Vec<u32>,
    ) -> Result<GrammarHandle, GrammarError> {
        let mut slot = self.shared.lock();
        let Some(io) = slot.as_mut() else {
            return Err(GrammarError::NotReady);
        };
        let generation = io.generation;
        let reply = match io.exchange(&open.request(), self.shared.deadline) {
            Ok(reply) => reply,
            Err(SidecarError::Refused {
                kind,
                message,
                param,
            }) => return Err(refused(kind, message, param)),
            Err(e) => {
                fail(&self.shared, &mut slot);
                return Err(match e {
                    SidecarError::Protocol(m) => GrammarError::Protocol(m),
                    _ => GrammarError::TimedOut {
                        param: open.param(),
                    },
                });
            }
        };
        let text = |key: &str| reply.get(key).and_then(Value::as_str);
        let parsed = match (
            reply.get("grammar").and_then(Value::as_u64),
            text("key_type"),
            text("key_sha256"),
        ) {
            (Some(grammar), Some(key_type), Some(key_sha256))
                if self.decode_mask(reply.get("bitmask"), mask_out) =>
            {
                Some(GrammarHandle {
                    grammar,
                    key_type: key_type.to_owned(),
                    key_sha256: key_sha256.to_owned(),
                    generation,
                })
            }
            _ => None,
        };
        parsed.ok_or_else(|| {
            fail(&self.shared, &mut slot);
            GrammarError::Protocol(
                "a grammar_open reply lacks its handle, key or a whole mask".to_owned(),
            )
        })
    }

    /// `grammar_step`: advances `handle` by `token`; the next mask, when there is one, goes into `mask_out`.
    pub fn grammar_step(
        &self,
        handle: &GrammarHandle,
        token: u32,
        mask_out: &mut Vec<u32>,
    ) -> Result<GrammarStep, GrammarError> {
        let request = json!({"op": "grammar_step", "grammar": handle.grammar, "token": token});
        let mut slot = self.shared.lock();
        let reply = self.handle_op(&mut slot, handle, &request)?;
        let Some(reply) = reply else {
            return Err(GrammarError::Lost(
                "its grammar sidecar child is gone".to_owned(),
            ));
        };
        let flag = |key: &str| reply.get(key).and_then(Value::as_bool);
        let has_mask = match reply.get("bitmask") {
            Some(Value::Null) => Some(false),
            bitmask => self.decode_mask(bitmask, mask_out).then_some(true),
        };
        match (flag("accepted"), flag("terminated"), has_mask) {
            (Some(accepted), Some(terminated), Some(has_mask)) => Ok(GrammarStep {
                accepted,
                terminated,
                has_mask,
            }),
            _ => {
                fail(&self.shared, &mut slot);
                Err(GrammarError::Lost(
                    "a grammar_step reply outside the protocol".to_owned(),
                ))
            }
        }
    }

    /// `grammar_drafts` (GrammarSpec GS-03): how many leading `drafts` the grammar accepts from `handle`'s state
    /// without terminating, the mask after each accepted one appended to `masks_out`; the state is unchanged.
    pub fn grammar_drafts(
        &self,
        handle: &GrammarHandle,
        drafts: &[u32],
        masks_out: &mut Vec<u32>,
    ) -> Result<usize, GrammarError> {
        let request = json!({"op": "grammar_drafts", "grammar": handle.grammar, "drafts": drafts});
        let mut slot = self.shared.lock();
        let reply = self.handle_op(&mut slot, handle, &request)?;
        let Some(reply) = reply else {
            return Err(GrammarError::Lost(
                "its grammar sidecar child is gone".to_owned(),
            ));
        };
        let start = masks_out.len();
        let accepted = reply
            .get("accepted")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n <= drafts.len());
        let decoded = accepted.filter(|&n| {
            reply
                .get("bitmasks")
                .and_then(Value::as_str)
                .is_some_and(|text| base64::append_le_u32(text.as_bytes(), masks_out).is_ok())
                && masks_out.len() - start == n * self.shared.info.mask_words()
        });
        decoded.ok_or_else(|| {
            masks_out.truncate(start);
            fail(&self.shared, &mut slot);
            GrammarError::Lost("a grammar_drafts reply outside the protocol".to_owned())
        })
    }

    /// `grammar_close`: releases `handle`; a no-op when its child is gone.
    pub fn grammar_close(&self, handle: &GrammarHandle) -> Result<(), GrammarError> {
        let request = json!({"op": "grammar_close", "grammar": handle.grammar});
        let mut slot = self.shared.lock();
        self.handle_op(&mut slot, handle, &request).map(|_| ())
    }

    /// A step or close on `handle`'s child, within the compile deadline (ops-hardening design note OH-02): `None`
    /// when that child is no longer the running one; a failed exchange other than a typed refusal (a missed
    /// deadline included) kills and respawns the child and loses the handle.
    fn handle_op(
        &self,
        slot: &mut Option<GrammarIo>,
        handle: &GrammarHandle,
        request: &Value,
    ) -> Result<Option<Value>, GrammarError> {
        let io = match slot.as_mut() {
            Some(io) if io.generation == handle.generation => io,
            _ => return Ok(None),
        };
        match io.exchange(request, self.shared.deadline) {
            Ok(reply) => Ok(Some(reply)),
            Err(SidecarError::Refused {
                kind,
                message,
                param,
            }) => Err(refused(kind, message, param)),
            Err(e) => {
                fail(&self.shared, slot);
                Err(GrammarError::Lost(e.to_string()))
            }
        }
    }

    /// Decodes a base64 mask of exactly [`GrammarInfo::mask_words`] words into `mask_out`.
    fn decode_mask(&self, bitmask: Option<&Value>, mask_out: &mut Vec<u32>) -> bool {
        bitmask
            .and_then(Value::as_str)
            .is_some_and(|text| base64::decode_le_u32(text.as_bytes(), mask_out).is_ok())
            && mask_out.len() == self.shared.info.mask_words()
    }
}

impl Drop for GrammarSidecar {
    fn drop(&mut self) {
        self.shared.closed.store(true, Ordering::SeqCst);
        self.shared.up.store(false, Ordering::SeqCst);
        *self.shared.lock() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OH-01: the first respawn after a served call is immediate, later ones double from 250 ms and stop at 30 s.
    #[test]
    fn respawn_backoff_doubles_up_to_the_cap() {
        let ms = |attempt| respawn_backoff(attempt).as_millis();
        assert_eq!(ms(1), 0);
        assert_eq!(ms(2), 250);
        assert_eq!(ms(3), 500);
        assert_eq!(ms(4), 1_000);
        assert_eq!(ms(8), 16_000);
        assert_eq!(ms(9), 30_000);
        assert_eq!(ms(u32::MAX), 30_000);
    }
}
