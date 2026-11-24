//! `cere-rpc-probe`: one-shot worker-RPC calls for the live federation harness (m35 design note Q18, P10;
//! CLI pinned in build-loop-m35 design note "`cere-rpc-probe` CLI"; `load`/`drain`/`unload`: m36 design note Q15, P7).
//!
//! `cere-rpc-probe --ca PEM --cert PEM --key PEM --endpoint https://HOST:PORT --peer PRINCIPAL [--timeout-ms N]
//! <command> [flags]`. The client identity is used as given (no SAN check on the own leaf), so forged and foreign
//! identities can be presented; the server is always verified against `--ca` and `--peer`. The connection (TCP,
//! mTLS, HTTP/2) is set up once, before the first call. Each call prints one JSON line on stdout
//! `{"call","n","code","reason","accepted","ticket","ms"}` (`describe` adds `"inventory_sha256"`; `submit
//! --stream` adds a line with call `"stream"`); `ms` is the call's wall time on that connection. `code` is the
//! gRPC code name, or `"TRANSPORT"` with reason `"handshake"`/`"connect"` when the call failed in the transport;
//! when the connection cannot be set up at all, every call of the command prints that failure, `ms` timed from
//! the attempt. The submit body's one message is `--content` (default `"probe"`) padded with spaces; prompt or
//! output text is never printed.
//! `load`/`drain`/`unload` send
//! the `--lock` hex as given (any length, so a malformed lock can be presented) with the M3.6 Q6
//! `wire::lifecycle_sha256` digest; the lock is never printed. Exit 0 when every call printed its line, 2 on a
//! usage error, 1 otherwise.
use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use cere_proto::worker::v2 as pb;
use cere_runtime::fed::wire;
use pb::worker_inference_service_client::WorkerInferenceServiceClient;
use pb::worker_registry_service_client::WorkerRegistryServiceClient;
use pb::worker_service_client::WorkerServiceClient;
use serde_json::{json, Value};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};
use tonic::{Code, Request, Status};

const USAGE: &str = "usage: cere-rpc-probe --ca PEM --cert PEM --key PEM --endpoint https://HOST:PORT \
--peer PRINCIPAL [--timeout-ms N] <command> [flags]
commands:
  register  --cluster-id C --worker-id W --epoch E [--revision 2]
  heartbeat --cluster-id C --worker-id W --epoch E --sequence S [--generation G] [--admission-state open]
  describe  AUTHORITY
  list      AUTHORITY
  submit    AUTHORITY --scope S --op-id O --request-id R [--attempt 1] [--model M] [--max-tokens N] [--stream]
            [--pad-bytes N] [--repeat N] [--content TEXT] [--conflict]
  cancel    AUTHORITY --scope S --op-id O --request-id R --ticket K
  oversize  AUTHORITY --service control|inference --bytes N
  load|drain|unload AUTHORITY --op-id O --lock HEX [--scope _cere_control] [--instance-id I]
            [--device-id D]... [--repeat N]
AUTHORITY = --cluster-id C --worker-id W --epoch E --coordinator-id D --term T [--generation G] [--budget-ms N]";

/// The lifecycle scope when `--scope` is not given.
const LIFECYCLE_SCOPE: &str = "_cere_control";

const DEFAULT_TIMEOUT_MS: u64 = 5000;
/// Default message content of a submit; `--conflict` appends CONFLICT_SUFFIX (same op, request and attempt).
const CONTENT: &str = "probe";
const CONFLICT_SUFFIX: &str = " conflict";
/// The `oversize --service inference` submit's scope and op/request id (a CERE-shaped rid is not needed: the
/// message must be refused at decode, before any field is read).
const OVERSIZE_SCOPE: &str = "probe-oversize";
const OVERSIZE_ID: &str = "oversize";

struct Usage(String);

/// `--flag value` and `--switch` arguments of one command; each flag at most once, unknown flags refused.
struct Flags(BTreeMap<String, Option<String>>);

impl Flags {
    fn parse(args: &[String], valued: &[&str], switches: &[&str]) -> Result<Flags, Usage> {
        let mut map = BTreeMap::new();
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            let name = arg
                .strip_prefix("--")
                .ok_or_else(|| Usage(format!("unexpected argument {arg:?}")))?;
            let value = if valued.contains(&name) {
                Some(
                    it.next()
                        .ok_or_else(|| Usage(format!("--{name} needs a value")))?
                        .clone(),
                )
            } else if switches.contains(&name) {
                None
            } else {
                return Err(Usage(format!("unknown flag --{name}")));
            };
            if map.insert(name.to_owned(), value).is_some() {
                return Err(Usage(format!("--{name} given twice")));
            }
        }
        Ok(Flags(map))
    }

    fn opt(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned().flatten()
    }

    fn req(&self, name: &str) -> Result<String, Usage> {
        self.opt(name)
            .ok_or_else(|| Usage(format!("--{name} is required")))
    }

    fn opt_num<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, Usage> {
        self.opt(name)
            .map(|v| {
                v.parse()
                    .map_err(|_| Usage(format!("--{name} {v:?} is not a number")))
            })
            .transpose()
    }

    fn req_num<T: std::str::FromStr>(&self, name: &str) -> Result<T, Usage> {
        self.opt_num(name)?
            .ok_or_else(|| Usage(format!("--{name} is required")))
    }

    fn switch(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

struct Globals {
    ca: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    endpoint: String,
    peer: String,
    timeout: Duration,
}

struct Identity3 {
    cluster_id: String,
    worker_id: String,
    epoch: u64,
}

impl Identity3 {
    const FLAGS: [&'static str; 3] = ["cluster-id", "worker-id", "epoch"];

    fn from(flags: &Flags) -> Result<Identity3, Usage> {
        Ok(Identity3 {
            cluster_id: flags.req("cluster-id")?,
            worker_id: flags.req("worker-id")?,
            epoch: flags.req_num("epoch")?,
        })
    }

    fn pb(&self) -> pb::WorkerIdentity {
        pb::WorkerIdentity {
            cluster_id: self.cluster_id.clone(),
            worker_id: self.worker_id.clone(),
            worker_epoch: self.epoch,
        }
    }
}

/// The authority flags of the worker-service commands.
struct Auth {
    target: Identity3,
    coordinator_id: String,
    term: u64,
    generation: Option<u64>,
    budget_ms: u64,
}

const AUTH_FLAGS: [&str; 7] = [
    "cluster-id",
    "worker-id",
    "epoch",
    "coordinator-id",
    "term",
    "generation",
    "budget-ms",
];

impl Auth {
    fn from(flags: &Flags) -> Result<Auth, Usage> {
        Ok(Auth {
            target: Identity3::from(flags)?,
            coordinator_id: flags.req("coordinator-id")?,
            term: flags.req_num("term")?,
            generation: flags.opt_num("generation")?,
            budget_ms: flags.opt_num("budget-ms")?.unwrap_or(wire::WORK_CAP_MS),
        })
    }

    /// The read form (DescribeResources, ListOperations): no scope, operation, digest or generation.
    fn read(&self) -> pb::Authority {
        pb::Authority {
            target: Some(self.target.pb()),
            coordinator_id: self.coordinator_id.clone(),
            coordinator_term: self.term,
            ..Default::default()
        }
    }

    /// The operation form (SubmitWork, CancelWork, StreamWork).
    fn op(&self, scope: &str, op_id: &str, payload_sha256: Vec<u8>) -> pb::Authority {
        pb::Authority {
            scope_id: scope.to_owned(),
            operation_id: op_id.to_owned(),
            payload_sha256,
            remaining_budget_ms: self.budget_ms,
            model_generation: self.generation,
            ..self.read()
        }
    }
}

struct Submit {
    auth: Auth,
    scope: String,
    op_id: String,
    request_id: String,
    attempt: u64,
    model: Option<String>,
    max_tokens: u64,
    stream: bool,
    pad_bytes: usize,
    repeat: u64,
    content: String,
    conflict: bool,
}

/// `load`/`drain`/`unload` (Q15).
struct Lifecycle {
    call: &'static str,
    method: &'static str,
    auth: Auth,
    scope: String,
    op_id: String,
    lock: Vec<u8>,
    instance_id: String,
    device_ids: Vec<String>,
    repeat: u64,
}

enum Service {
    Control,
    Inference,
}

enum Command {
    Register {
        id: Identity3,
        revision: u32,
    },
    Heartbeat {
        id: Identity3,
        sequence: u64,
        generation: u64,
        admission_state: String,
    },
    Describe(Auth),
    List(Auth),
    Submit(Submit),
    Cancel {
        auth: Auth,
        scope: String,
        op_id: String,
        request_id: String,
        ticket: String,
    },
    Oversize {
        auth: Auth,
        service: Service,
        bytes: usize,
    },
    Lifecycle(Lifecycle),
}

impl Command {
    /// The call name and the number of lines the command prints when none of its calls is sent (no `stream`
    /// line: it only follows an accepted submit).
    fn lines(&self) -> (&'static str, u64) {
        match self {
            Command::Register { .. } => ("register", 1),
            Command::Heartbeat { .. } => ("heartbeat", 1),
            Command::Describe(_) => ("describe", 1),
            Command::List(_) => ("list", 1),
            Command::Submit(s) => ("submit", s.repeat),
            Command::Cancel { .. } => ("cancel", 1),
            Command::Oversize { .. } => ("oversize", 1),
            Command::Lifecycle(l) => (l.call, l.repeat),
        }
    }
}

fn with_auth<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    AUTH_FLAGS
        .iter()
        .copied()
        .chain(extra.iter().copied())
        .collect()
}

/// Lower- or upper-case hex of even length (empty allowed by the parser, refused by the caller).
fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| s.get(i..i + 2).and_then(|b| u8::from_str_radix(b, 16).ok()))
        .collect()
}

fn repeat(f: &Flags) -> Result<u64, Usage> {
    let repeat = f.opt_num("repeat")?.unwrap_or(1);
    if repeat == 0 {
        return Err(Usage("--repeat must be at least 1".to_owned()));
    }
    Ok(repeat)
}

fn parse(args: &[String]) -> Result<(Globals, Command), Usage> {
    let split = args
        .iter()
        .enumerate()
        .step_by(2)
        .find(|(_, a)| !a.starts_with("--"))
        .map(|(i, _)| i)
        .ok_or_else(|| Usage("no command".to_owned()))?;
    let global = Flags::parse(
        &args[..split],
        &["ca", "cert", "key", "endpoint", "peer", "timeout-ms"],
        &[],
    )?;
    let endpoint = global.req("endpoint")?;
    if !endpoint.starts_with("https://") {
        return Err(Usage(format!("--endpoint {endpoint:?} is not https://")));
    }
    let globals = Globals {
        ca: global.req("ca")?.into(),
        cert: global.req("cert")?.into(),
        key: global.req("key")?.into(),
        endpoint,
        peer: global.req("peer")?,
        timeout: Duration::from_millis(global.opt_num("timeout-ms")?.unwrap_or(DEFAULT_TIMEOUT_MS)),
    };
    let rest = &args[split + 1..];
    let command = match args[split].as_str() {
        "register" => {
            let f = Flags::parse(rest, &[&Identity3::FLAGS[..], &["revision"]].concat(), &[])?;
            Command::Register {
                id: Identity3::from(&f)?,
                revision: f
                    .opt_num("revision")?
                    .unwrap_or(cere_proto::PROTOCOL_REVISION),
            }
        }
        "heartbeat" => {
            let valued = [
                &Identity3::FLAGS[..],
                &["sequence", "generation", "admission-state"],
            ]
            .concat();
            let f = Flags::parse(rest, &valued, &[])?;
            Command::Heartbeat {
                id: Identity3::from(&f)?,
                sequence: f.req_num("sequence")?,
                generation: f.opt_num("generation")?.unwrap_or(0),
                admission_state: f
                    .opt("admission-state")
                    .unwrap_or_else(|| wire::AdmissionState::Open.as_str().to_owned()),
            }
        }
        "describe" => Command::Describe(Auth::from(&Flags::parse(rest, &AUTH_FLAGS, &[])?)?),
        "list" => Command::List(Auth::from(&Flags::parse(rest, &AUTH_FLAGS, &[])?)?),
        "submit" => {
            let valued = with_auth(&[
                "scope",
                "op-id",
                "request-id",
                "attempt",
                "model",
                "max-tokens",
                "pad-bytes",
                "repeat",
                "content",
            ]);
            let f = Flags::parse(rest, &valued, &["stream", "conflict"])?;
            let repeat = repeat(&f)?;
            Command::Submit(Submit {
                auth: Auth::from(&f)?,
                scope: f.req("scope")?,
                op_id: f.req("op-id")?,
                request_id: f.req("request-id")?,
                attempt: f.opt_num("attempt")?.unwrap_or(1),
                model: f.opt("model"),
                max_tokens: f.opt_num("max-tokens")?.unwrap_or(1),
                stream: f.switch("stream"),
                pad_bytes: f.opt_num("pad-bytes")?.unwrap_or(0),
                repeat,
                content: f.opt("content").unwrap_or_else(|| CONTENT.to_owned()),
                conflict: f.switch("conflict"),
            })
        }
        "cancel" => {
            let valued = with_auth(&["scope", "op-id", "request-id", "ticket"]);
            let f = Flags::parse(rest, &valued, &[])?;
            Command::Cancel {
                auth: Auth::from(&f)?,
                scope: f.req("scope")?,
                op_id: f.req("op-id")?,
                request_id: f.req("request-id")?,
                ticket: f.req("ticket")?,
            }
        }
        "oversize" => {
            let f = Flags::parse(rest, &with_auth(&["service", "bytes"]), &[])?;
            let service = match f.req("service")?.as_str() {
                "control" => Service::Control,
                "inference" => Service::Inference,
                other => return Err(Usage(format!("--service {other:?}: control|inference"))),
            };
            Command::Oversize {
                auth: Auth::from(&f)?,
                service,
                bytes: f.req_num("bytes")?,
            }
        }
        call @ ("load" | "drain" | "unload") => {
            let (call, method) = match call {
                "load" => ("load", wire::METHOD_LOAD),
                "drain" => ("drain", wire::METHOD_DRAIN),
                _ => ("unload", wire::METHOD_UNLOAD),
            };
            // `--device-id` is the one repeatable flag: taken out before the at-most-once parse.
            let mut device_ids = Vec::new();
            let mut others = Vec::new();
            let mut it = rest.iter();
            while let Some(arg) = it.next() {
                if arg == "--device-id" {
                    device_ids.push(
                        it.next()
                            .ok_or_else(|| Usage("--device-id needs a value".to_owned()))?
                            .clone(),
                    );
                } else {
                    others.push(arg.clone());
                }
            }
            let valued = with_auth(&["scope", "op-id", "lock", "instance-id", "repeat"]);
            let f = Flags::parse(&others, &valued, &[])?;
            let lock_hex = f.req("lock")?;
            let lock = unhex(&lock_hex)
                .filter(|l| !l.is_empty())
                .ok_or_else(|| Usage("--lock is not hex".to_owned()))?;
            Command::Lifecycle(Lifecycle {
                call,
                method,
                auth: Auth::from(&f)?,
                scope: f.opt("scope").unwrap_or_else(|| LIFECYCLE_SCOPE.to_owned()),
                op_id: f.req("op-id")?,
                lock,
                instance_id: f.opt("instance-id").unwrap_or_default(),
                device_ids,
                repeat: repeat(&f)?,
            })
        }
        other => return Err(Usage(format!("unknown command {other:?}"))),
    };
    Ok((globals, command))
}

/// The result of one call, printed as its JSON line.
struct Outcome {
    code: String,
    reason: String,
    accepted: bool,
    ticket: String,
    /// `describe` only: hex SHA-256 of the exact inventory bytes, "" on failure.
    inventory_sha256: Option<String>,
}

impl Outcome {
    fn ok(accepted: bool, reason: &str, ticket: &str) -> Outcome {
        Outcome {
            code: "OK".to_owned(),
            reason: reason.to_owned(),
            accepted,
            ticket: ticket.to_owned(),
            inventory_sha256: None,
        }
    }

    fn receipt(receipt: pb::OperationReceipt) -> Outcome {
        Outcome::ok(
            receipt.accepted,
            &receipt.reason_code,
            &receipt.completion_ticket,
        )
    }

    fn failed(status: &Status) -> Outcome {
        let (code, reason) = match transport_failure(status) {
            Some(reason) => ("TRANSPORT", reason.to_owned()),
            None => (code_name(status.code()), wire::reason(status).to_owned()),
        };
        Outcome {
            code: code.to_owned(),
            reason,
            accepted: false,
            ticket: String::new(),
            inventory_sha256: None,
        }
    }
}

fn code_name(code: Code) -> &'static str {
    match code {
        Code::Ok => "OK",
        Code::Cancelled => "CANCELLED",
        Code::Unknown => "UNKNOWN",
        Code::InvalidArgument => "INVALID_ARGUMENT",
        Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        Code::NotFound => "NOT_FOUND",
        Code::AlreadyExists => "ALREADY_EXISTS",
        Code::PermissionDenied => "PERMISSION_DENIED",
        Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        Code::FailedPrecondition => "FAILED_PRECONDITION",
        Code::Aborted => "ABORTED",
        Code::OutOfRange => "OUT_OF_RANGE",
        Code::Unimplemented => "UNIMPLEMENTED",
        Code::Internal => "INTERNAL",
        Code::Unavailable => "UNAVAILABLE",
        Code::DataLoss => "DATA_LOSS",
        Code::Unauthenticated => "UNAUTHENTICATED",
    }
}

/// `Some("connect" | "handshake")` when the status wraps a `tonic::transport::Error` (the call never reached a
/// service; a status the peer sent has no source). `connect` = the TCP connection was never established (refused,
/// unreachable, timed out); anything after it (a failed TLS handshake, or the server closing the connection
/// because it refused the client certificate, which under TLS 1.3 the client only sees once it writes the
/// request) is `handshake`.
fn transport_failure(status: &Status) -> Option<&'static str> {
    use std::io::ErrorKind;
    let mut source = std::error::Error::source(status);
    let mut transport = false;
    let mut connect = false;
    while let Some(err) = source {
        transport |= err.is::<tonic::transport::Error>();
        // hyper-util's `ConnectError("tcp connect error" | "dns error" | ..., ..)` is the TCP connect step (tonic's
        // own `ConnectError(..)` around it also wraps TLS failures, so only the inner one counts).
        connect |= format!("{err:?}").starts_with("ConnectError(\"");
        if let Some(io) = err.downcast_ref::<std::io::Error>() {
            connect |= matches!(
                io.kind(),
                ErrorKind::ConnectionRefused
                    | ErrorKind::TimedOut
                    | ErrorKind::HostUnreachable
                    | ErrorKind::NetworkUnreachable
                    | ErrorKind::AddrNotAvailable
            );
        }
        source = err.source();
    }
    transport.then_some(if connect { "connect" } else { "handshake" })
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

fn print_line(call: &str, n: u64, outcome: &Outcome, started: Instant) {
    let mut line = json!({
        "call": call,
        "n": n,
        "code": outcome.code,
        "reason": outcome.reason,
        "accepted": outcome.accepted,
        "ticket": outcome.ticket,
        "ms": started.elapsed().as_millis() as u64,
    });
    if call == "describe" {
        line["inventory_sha256"] =
            Value::from(outcome.inventory_sha256.clone().unwrap_or_default());
    }
    println!("{line}");
}

fn request<T>(message: T, timeout: Duration) -> Request<T> {
    let mut req = Request::new(message);
    req.set_timeout(timeout);
    req
}

/// The submitted chat body: the CERE rid, one user message of `content` (`--content`) padded with spaces.
fn submit_body(s: &Submit, content: &str, pad_bytes: usize) -> Vec<u8> {
    let mut content = content.to_owned();
    if content.len() < pad_bytes {
        content.push_str(&" ".repeat(pad_bytes - content.len()));
    }
    let mut body = json!({
        "rid": s.request_id,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": s.max_tokens,
        "temperature": 0,
        "stream": s.stream,
    });
    if let Some(model) = &s.model {
        body["model"] = Value::from(model.as_str());
    }
    serde_json::to_vec(&body).expect("a JSON value serializes")
}

fn submit_request(s: &Submit, body: Vec<u8>) -> pb::SubmitWorkRequest {
    let work_unit_json = serde_json::to_vec(&json!({"kind": "chat", "stream": s.stream}))
        .expect("a JSON value serializes");
    let digest = wire::payload_sha256(
        wire::METHOD_SUBMIT,
        &s.scope,
        &s.request_id,
        s.attempt,
        &work_unit_json,
        &body,
    );
    pb::SubmitWorkRequest {
        authority: Some(s.auth.op(&s.scope, &s.op_id, digest.to_vec())),
        request_id: s.request_id.clone(),
        attempt: s.attempt,
        work_unit_json,
        bounded_inference_input_json: body,
    }
}

/// Reads a `StreamWork` stream to its terminal event; `accepted` = a terminal event arrived, `reason` = its
/// reason code. Bounded by the request budget.
async fn stream(channel: Channel, s: &Submit, ticket: &str) -> Outcome {
    let req = pb::WorkStreamRequest {
        authority: Some(s.auth.op(&s.scope, &s.op_id, Vec::new())),
        request_id: s.request_id.clone(),
        attempt: s.attempt,
        completion_ticket: ticket.to_owned(),
        after_sequence: 0,
    };
    let read = async {
        let mut events = match WorkerServiceClient::new(channel).stream_work(req).await {
            Ok(response) => response.into_inner(),
            Err(status) => return Outcome::failed(&status),
        };
        loop {
            match events.message().await {
                Ok(Some(event)) if event.terminal => {
                    return Outcome::ok(true, &event.reason_code, "")
                }
                Ok(Some(_)) => {}
                Ok(None) => return Outcome::ok(false, "", ""),
                Err(status) => return Outcome::failed(&status),
            }
        }
    };
    match tokio::time::timeout(Duration::from_millis(s.auth.budget_ms), read).await {
        Ok(outcome) => outcome,
        Err(_) => Outcome::failed(&Status::deadline_exceeded("")),
    }
}

fn boot_id() -> std::io::Result<String> {
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(hex(&bytes))
}

/// Faults this process's own executable mappings in, before anything is timed. On a host under memory pressure
/// the page cache drops the probe's code; paging the TLS and gRPC paths back in one fault at a time from a busy disk
/// took over a second, which a line's `ms` (the peer's latency) must not count (fed-load-flake design note F1).
/// One `MADV_POPULATE_READ` per mapping reads them with readahead instead. Best effort: on a kernel without it
/// (before 5.14) the pages fault in as before.
fn prefault_own_code() {
    let (Ok(exe), Ok(maps)) = (
        std::fs::read_link("/proc/self/exe"),
        std::fs::read_to_string("/proc/self/maps"),
    ) else {
        return;
    };
    for line in maps.lines() {
        // `start-end perms offset dev inode path`, the path padded with spaces.
        let mut fields = line.splitn(6, ' ');
        let (Some(range), Some(path)) = (fields.next(), fields.nth(4)) else {
            continue;
        };
        if std::path::Path::new(path.trim_start()) != exe {
            continue;
        }
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        let (Ok(start), Ok(end)) = (
            usize::from_str_radix(start, 16),
            usize::from_str_radix(end, 16),
        ) else {
            continue;
        };
        // SAFETY: advice on a range this process has mapped; POPULATE_READ only faults pages in for reading, it
        // changes no mapping and no content.
        unsafe {
            libc::madvise(
                start as *mut libc::c_void,
                end - start,
                libc::MADV_POPULATE_READ,
            )
        };
    }
}

fn endpoint(g: &Globals) -> Result<Endpoint, String> {
    let read = |p: &PathBuf| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
    let (ca, cert, key) = (read(&g.ca)?, read(&g.cert)?, read(&g.key)?);
    let tls = ClientTlsConfig::new()
        .ca_certificate(Certificate::from_pem(ca))
        .identity(Identity::from_pem(cert, key))
        .domain_name(g.peer.clone());
    Endpoint::from_shared(g.endpoint.clone())
        .map_err(|e| format!("--endpoint: {e}"))?
        .connect_timeout(g.timeout)
        .timeout(g.timeout)
        .tls_config(tls)
        .map_err(|e| format!("TLS config: {e}"))
}

async fn run(g: Globals, command: Command) -> Result<(), String> {
    prefault_own_code();
    let endpoint = endpoint(&g)?;
    // Connected before any call is timed, so a line's `ms` is the peer answering the call, not this process
    // setting up TCP, mTLS and HTTP/2 (fed-load-flake design note F1).
    let attempt = Instant::now();
    let ch = match endpoint.connect().await {
        Ok(ch) => ch,
        Err(e) => {
            let outcome = Outcome::failed(&Status::from_error(Box::new(e)));
            let (call, lines) = command.lines();
            for n in 0..lines {
                print_line(call, n, &outcome, attempt);
            }
            return Ok(());
        }
    };
    let t = g.timeout;
    let started = Instant::now();
    match command {
        Command::Register { id, revision } => {
            let req = pb::RegisterRequest {
                proposed: Some(id.pb()),
                boot_id: boot_id().map_err(|e| format!("boot id: {e}"))?,
                protocol_revision: revision,
                inventory_sha256: Vec::new(),
            };
            let outcome = match WorkerRegistryServiceClient::new(ch)
                .register(request(req, t))
                .await
            {
                Ok(r) => {
                    let r = r.into_inner();
                    Outcome::ok(r.accepted, if r.accepted { "" } else { &r.reason_code }, "")
                }
                Err(status) => Outcome::failed(&status),
            };
            print_line("register", 0, &outcome, started);
        }
        Command::Heartbeat {
            id,
            sequence,
            generation,
            admission_state,
        } => {
            let req = pb::HeartbeatRequest {
                worker: Some(id.pb()),
                sequence,
                observed_coordinator_term: 0,
                inventory_sha256: Vec::new(),
                model_generation: generation,
                admission_state,
                owner_only: false,
            };
            let outcome = match WorkerRegistryServiceClient::new(ch)
                .heartbeat(request(req, t))
                .await
            {
                Ok(r) => {
                    let r = r.into_inner();
                    Outcome::ok(r.accepted, if r.accepted { "" } else { &r.reason_code }, "")
                }
                Err(status) => Outcome::failed(&status),
            };
            print_line("heartbeat", 0, &outcome, started);
        }
        Command::Describe(auth) => {
            let req = pb::InventoryRequest {
                authority: Some(auth.read()),
            };
            let outcome = match WorkerServiceClient::new(ch)
                .describe_resources(request(req, t))
                .await
            {
                Ok(r) => {
                    let sha = wire::Inventory::sha256(&r.get_ref().bounded_resource_inventory_json);
                    Outcome {
                        inventory_sha256: Some(hex(&sha)),
                        ..Outcome::ok(true, "", "")
                    }
                }
                Err(status) => Outcome::failed(&status),
            };
            print_line("describe", 0, &outcome, started);
        }
        Command::List(auth) => {
            let req = pb::ListOperationsRequest {
                authority: Some(auth.read()),
                limit: wire::LIST_PAGE_MAX as u32,
                ..Default::default()
            };
            let outcome = match WorkerServiceClient::new(ch)
                .list_operations(request(req, t))
                .await
            {
                Ok(_) => Outcome::ok(true, "", ""),
                Err(status) => Outcome::failed(&status),
            };
            print_line("list", 0, &outcome, started);
        }
        Command::Submit(s) => {
            let content = if s.conflict {
                format!("{}{CONFLICT_SUFFIX}", s.content)
            } else {
                s.content.clone()
            };
            let req = submit_request(&s, submit_body(&s, &content, s.pad_bytes));
            let mut first_ticket = None;
            for n in 0..s.repeat {
                let started = Instant::now();
                let outcome = match WorkerInferenceServiceClient::new(ch.clone())
                    .submit_work(request(req.clone(), t))
                    .await
                {
                    Ok(r) => Outcome::receipt(r.into_inner()),
                    Err(status) => Outcome::failed(&status),
                };
                if n == 0 && outcome.code == "OK" && outcome.accepted {
                    first_ticket = Some(outcome.ticket.clone());
                }
                print_line("submit", n, &outcome, started);
            }
            if let (true, Some(ticket)) = (s.stream, first_ticket) {
                let started = Instant::now();
                let outcome = stream(ch, &s, &ticket).await;
                print_line("stream", 0, &outcome, started);
            }
        }
        Command::Cancel {
            auth,
            scope,
            op_id,
            request_id,
            ticket,
        } => {
            let attempt = 1;
            let digest = wire::payload_sha256(
                wire::METHOD_CANCEL,
                &scope,
                &request_id,
                attempt,
                b"",
                ticket.as_bytes(),
            );
            let req = pb::CancelWorkRequest {
                authority: Some(auth.op(&scope, &op_id, digest.to_vec())),
                request_id,
                attempt,
                completion_ticket: ticket,
            };
            let outcome = match WorkerServiceClient::new(ch)
                .cancel_work(request(req, t))
                .await
            {
                Ok(r) => Outcome::receipt(r.into_inner()),
                Err(status) => Outcome::failed(&status),
            };
            print_line("cancel", 0, &outcome, started);
        }
        Command::Oversize {
            auth,
            service,
            bytes,
        } => {
            let outcome = match service {
                // A DescribeResources whose authority carries `bytes` of padding: refused at decode.
                Service::Control => {
                    let req = pb::InventoryRequest {
                        authority: Some(pb::Authority {
                            scope_id: "x".repeat(bytes),
                            ..auth.read()
                        }),
                    };
                    match WorkerServiceClient::new(ch)
                        .describe_resources(request(req, t))
                        .await
                    {
                        Ok(_) => Outcome::ok(true, "", ""),
                        Err(status) => Outcome::failed(&status),
                    }
                }
                // A well-formed SubmitWork whose synthetic body is padded to `bytes`.
                Service::Inference => {
                    let s = Submit {
                        auth,
                        scope: OVERSIZE_SCOPE.to_owned(),
                        op_id: OVERSIZE_ID.to_owned(),
                        request_id: OVERSIZE_ID.to_owned(),
                        attempt: 1,
                        model: None,
                        max_tokens: 1,
                        stream: false,
                        pad_bytes: bytes,
                        repeat: 1,
                        content: CONTENT.to_owned(),
                        conflict: false,
                    };
                    let req = submit_request(&s, submit_body(&s, CONTENT, bytes));
                    match WorkerInferenceServiceClient::new(ch)
                        .submit_work(request(req, t))
                        .await
                    {
                        Ok(r) => Outcome::receipt(r.into_inner()),
                        Err(status) => Outcome::failed(&status),
                    }
                }
            };
            print_line("oversize", 0, &outcome, started);
        }
        Command::Lifecycle(l) => {
            let digest =
                wire::lifecycle_sha256(l.method, &l.scope, &l.lock, &l.instance_id, &l.device_ids);
            let req = pb::ModelLifecycleRequest {
                authority: Some(l.auth.op(&l.scope, &l.op_id, digest.to_vec())),
                deployment_lock_sha256: l.lock,
                instance_id: l.instance_id,
                device_ids: l.device_ids,
            };
            let client = WorkerServiceClient::new(ch);
            for n in 0..l.repeat {
                let started = Instant::now();
                let mut c = client.clone();
                let req = request(req.clone(), t);
                let reply = match l.method {
                    wire::METHOD_LOAD => c.load_model(req).await,
                    wire::METHOD_DRAIN => c.drain_model(req).await,
                    _ => c.unload_model(req).await,
                };
                let outcome = match reply {
                    Ok(r) => Outcome::receipt(r.into_inner()),
                    Err(status) => Outcome::failed(&status),
                };
                print_line(l.call, n, &outcome, started);
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (globals, command) = match parse(&args) {
        Ok(parsed) => parsed,
        Err(Usage(message)) => {
            eprintln!("cere-rpc-probe: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("cere-rpc-probe: runtime: {e}");
            return ExitCode::from(1);
        }
    };
    match runtime.block_on(run(globals, command)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("cere-rpc-probe: {message}");
            ExitCode::from(1)
        }
    }
}
