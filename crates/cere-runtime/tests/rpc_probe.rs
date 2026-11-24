//! `cere-rpc-probe` (m35 design note P10, Q18; build-loop-m35 design note "`cere-rpc-probe` CLI"; m36 design note
//! P7, Q15 for load/drain/unload): the built binary against in-test mTLS tonic servers that check the fields the
//! probe sends the way the real peers do (authorization, authority, generation, digest, body rid, lock, dedupe)
//! and record every call.
mod support;

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cere_proto::worker::v2 as pb;
use cere_runtime::fed::tls::{Authz, Pki};
use cere_runtime::fed::wire;
use pb::worker_inference_service_server::{WorkerInferenceService, WorkerInferenceServiceServer};
use pb::worker_registry_service_server::{WorkerRegistryService, WorkerRegistryServiceServer};
use pb::worker_service_server::{WorkerService, WorkerServiceServer};
use serde_json::Value;
use support::pki::TestPki;
use support::StateDir;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_stream::Stream;
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

const COORD: &str = "cere-coordinator";
const WORKER_A: &str = "cere-worker-a";
const CLUSTER: &str = "cluster-1";
const WORKER_ID: &str = "worker-a";
const COORD_ID: &str = "coord-1";
const EPOCH: u64 = 3;
const TERM: u64 = 7;
const GENERATION: u64 = 2;
const RID: &str = "cere-00000000000000000001-00000000000000000002-0123456789abcdef";
const INVENTORY: &[u8] = br#"{"schema_version":1,"worker_id":"worker-a"}"#;
const LINE_KEYS: [&str; 7] = ["accepted", "call", "code", "ms", "n", "reason", "ticket"];
/// The deployment lock the in-test agent approves (`[backend] lock_sha256`).
const LOCK: [u8; 32] = [0xab; 32];

/// Every request that passed authorization, in arrival order per method.
#[derive(Default)]
struct Seen {
    register: Vec<pb::RegisterRequest>,
    heartbeat: Vec<pb::HeartbeatRequest>,
    describe: Vec<pb::InventoryRequest>,
    list: Vec<pb::ListOperationsRequest>,
    submit: Vec<pb::SubmitWorkRequest>,
    cancel: Vec<pb::CancelWorkRequest>,
    stream: Vec<pb::WorkStreamRequest>,
    /// (method, request) of LoadModel/DrainModel/UnloadModel.
    lifecycle: Vec<(&'static str, pb::ModelLifecycleRequest)>,
}

struct Op {
    payload: Vec<u8>,
    ticket: String,
}

#[derive(Default)]
struct State {
    seen: Seen,
    /// (scope_id, operation_id) -> op, as the agent's op journal keys it.
    ops: BTreeMap<(String, String), Op>,
    last_sequence: Option<u64>,
}

#[derive(Clone)]
struct Peer {
    authz: Authz,
    state: Arc<Mutex<State>>,
    tickets: Arc<AtomicUsize>,
}

fn status(code: Code, reason: &str) -> Status {
    wire::status(code, reason)
}

fn malformed() -> Status {
    status(Code::InvalidArgument, wire::MALFORMED_REQUEST)
}

impl Peer {
    fn new(authz: Authz) -> Peer {
        Peer {
            authz,
            state: Arc::default(),
            tickets: Arc::default(),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }

    /// The agent's authority checks: target, coordinator id, term; the generation is required (and must be
    /// current) for Submit/Cancel/Stream and must be unset for the reads.
    fn authority<'a>(
        &self,
        authority: &'a Option<pb::Authority>,
        generation_required: bool,
    ) -> Result<&'a pb::Authority, Status> {
        let a = authority.as_ref().ok_or_else(malformed)?;
        let target = a.target.as_ref().ok_or_else(malformed)?;
        if target.cluster_id != CLUSTER {
            return Err(status(Code::FailedPrecondition, wire::WRONG_CLUSTER));
        }
        if target.worker_id != WORKER_ID {
            return Err(status(Code::FailedPrecondition, wire::WRONG_WORKER));
        }
        if target.worker_epoch != EPOCH {
            return Err(status(Code::FailedPrecondition, wire::STALE_EPOCH));
        }
        if a.coordinator_id != COORD_ID {
            return Err(status(Code::FailedPrecondition, wire::WRONG_COORDINATOR));
        }
        if a.coordinator_term < TERM {
            return Err(status(Code::FailedPrecondition, wire::STALE_TERM));
        }
        if generation_required {
            if a.model_generation != Some(GENERATION) {
                return Err(status(Code::FailedPrecondition, wire::STALE_GENERATION));
            }
            if a.payload_sha256.len() != 32 && !a.payload_sha256.is_empty() {
                return Err(malformed());
            }
        } else if a.model_generation.is_some()
            || !a.payload_sha256.is_empty()
            || !a.operation_id.is_empty()
        {
            return Err(malformed());
        }
        Ok(a)
    }

    fn ticket(&self) -> String {
        format!("ticket-{}", self.tickets.fetch_add(1, Ordering::SeqCst))
    }

    /// LoadModel/DrainModel/UnloadModel as the agent checks them (M3.6 Q6): coordinator caller, authority with
    /// the current generation, scope and op id set, the lifecycle digest, a 32-byte approved lock; the op journal
    /// answers a repeated (scope, op id) with its first ticket.
    fn lifecycle(
        &self,
        method: &'static str,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.authz.coordinator_caller(&req)?;
        let r = req.into_inner();
        let a = self.authority(&r.authority, true)?;
        if a.scope_id.is_empty() || a.operation_id.is_empty() || a.payload_sha256.len() != 32 {
            return Err(malformed());
        }
        let digest = wire::lifecycle_sha256(
            method,
            &a.scope_id,
            &r.deployment_lock_sha256,
            &r.instance_id,
            &r.device_ids,
        );
        if a.payload_sha256 != digest {
            return Err(status(Code::InvalidArgument, wire::PAYLOAD_DIGEST_MISMATCH));
        }
        let mut st = self.state();
        st.seen.lifecycle.push((method, r.clone()));
        if r.deployment_lock_sha256.len() != 32 {
            return Err(malformed());
        }
        if r.deployment_lock_sha256 != LOCK {
            return Err(status(Code::FailedPrecondition, wire::LOCK_NOT_APPROVED));
        }
        let key = (a.scope_id.clone(), a.operation_id.clone());
        let ticket = match st.ops.get(&key) {
            Some(op) if op.payload == digest => op.ticket.clone(),
            Some(_) => return Err(status(Code::AlreadyExists, wire::OPERATION_CONFLICT)),
            None => {
                let ticket = self.ticket();
                st.ops.insert(
                    key,
                    Op {
                        payload: digest.to_vec(),
                        ticket: ticket.clone(),
                    },
                );
                ticket
            }
        };
        Ok(Response::new(pb::OperationReceipt {
            accepted: true,
            operation_id: a.operation_id.clone(),
            completion_ticket: ticket,
            reason_code: String::new(),
        }))
    }
}

#[tonic::async_trait]
impl WorkerRegistryService for Peer {
    async fn register(
        &self,
        req: Request<pb::RegisterRequest>,
    ) -> Result<Response<pb::RegisterReply>, Status> {
        let proposed = req.get_ref().proposed.clone().ok_or_else(malformed)?;
        self.authz.registry_caller(&req, &proposed.worker_id)?;
        let r = req.into_inner();
        if !r.inventory_sha256.is_empty() && r.inventory_sha256.len() != 32 {
            return Err(malformed());
        }
        self.state().seen.register.push(r.clone());
        let refuse = |reason: &str| pb::RegisterReply {
            reason_code: reason.to_owned(),
            ..Default::default()
        };
        let reply = if r.protocol_revision != cere_proto::PROTOCOL_REVISION {
            refuse(wire::UNSUPPORTED_REVISION)
        } else if proposed.cluster_id != CLUSTER {
            refuse(wire::WRONG_CLUSTER)
        } else {
            pb::RegisterReply {
                accepted: true,
                coordinator_term: TERM,
                admission_lease_ms: 4000,
                requires_reconciliation: true,
                ..Default::default()
            }
        };
        Ok(Response::new(reply))
    }

    async fn heartbeat(
        &self,
        req: Request<pb::HeartbeatRequest>,
    ) -> Result<Response<pb::HeartbeatReply>, Status> {
        let worker = req.get_ref().worker.clone().ok_or_else(malformed)?;
        self.authz.registry_caller(&req, &worker.worker_id)?;
        let r = req.into_inner();
        if wire::AdmissionState::parse(&r.admission_state).is_none() {
            return Err(malformed());
        }
        let mut st = self.state();
        st.seen.heartbeat.push(r.clone());
        if st.last_sequence.is_some_and(|last| r.sequence <= last) {
            return Ok(Response::new(pb::HeartbeatReply {
                reason_code: wire::REPLAYED_HEARTBEAT.to_owned(),
                ..Default::default()
            }));
        }
        st.last_sequence = Some(r.sequence);
        Ok(Response::new(pb::HeartbeatReply {
            accepted: true,
            coordinator_term: TERM,
            lease_sequence: 1,
            admission_lease_ms: 4000,
            ..Default::default()
        }))
    }
}

type EventStream = Pin<Box<dyn Stream<Item = Result<pb::WorkEvent, Status>> + Send>>;

#[tonic::async_trait]
impl WorkerService for Peer {
    async fn describe_resources(
        &self,
        req: Request<pb::InventoryRequest>,
    ) -> Result<Response<pb::InventoryReply>, Status> {
        self.authz.coordinator_caller(&req)?;
        self.authority(&req.get_ref().authority, false)?;
        self.state().seen.describe.push(req.into_inner());
        Ok(Response::new(pb::InventoryReply {
            bounded_resource_inventory_json: INVENTORY.to_vec(),
            inventory_generation: 1,
        }))
    }

    async fn load_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(wire::METHOD_LOAD, req)
    }

    async fn drain_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(wire::METHOD_DRAIN, req)
    }

    async fn unload_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(wire::METHOD_UNLOAD, req)
    }

    async fn cancel_work(
        &self,
        req: Request<pb::CancelWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.authz.coordinator_caller(&req)?;
        let r = req.into_inner();
        let a = self.authority(&r.authority, true)?;
        if a.scope_id.is_empty() || a.operation_id.is_empty() || a.payload_sha256.len() != 32 {
            return Err(malformed());
        }
        let digest = wire::payload_sha256(
            wire::METHOD_CANCEL,
            &a.scope_id,
            &r.request_id,
            r.attempt,
            b"",
            r.completion_ticket.as_bytes(),
        );
        if a.payload_sha256 != digest {
            return Err(status(Code::InvalidArgument, wire::PAYLOAD_DIGEST_MISMATCH));
        }
        let mut st = self.state();
        st.seen.cancel.push(r.clone());
        let target = st.ops.get(&(a.scope_id.clone(), r.request_id.clone()));
        match target {
            Some(op) if op.ticket == r.completion_ticket && r.attempt == 1 => {}
            _ => return Err(status(Code::FailedPrecondition, wire::TICKET_MISMATCH)),
        }
        Ok(Response::new(pb::OperationReceipt {
            accepted: true,
            operation_id: a.operation_id.clone(),
            completion_ticket: r.completion_ticket.clone(),
            reason_code: String::new(),
        }))
    }

    async fn get_operation(
        &self,
        _: Request<pb::OperationQuery>,
    ) -> Result<Response<pb::OperationStatus>, Status> {
        Err(Status::unimplemented(""))
    }

    async fn list_operations(
        &self,
        req: Request<pb::ListOperationsRequest>,
    ) -> Result<Response<pb::ListOperationsReply>, Status> {
        self.authz.coordinator_caller(&req)?;
        self.authority(&req.get_ref().authority, false)?;
        let limit = req.get_ref().limit as usize;
        if !(1..=wire::LIST_PAGE_MAX).contains(&limit) {
            return Err(malformed());
        }
        self.state().seen.list.push(req.into_inner());
        Ok(Response::new(pb::ListOperationsReply::default()))
    }

    type StreamWorkStream = EventStream;

    async fn stream_work(
        &self,
        req: Request<pb::WorkStreamRequest>,
    ) -> Result<Response<EventStream>, Status> {
        self.authz.coordinator_caller(&req)?;
        let r = req.into_inner();
        let a = self.authority(&r.authority, true)?;
        let mut st = self.state();
        st.seen.stream.push(r.clone());
        let key = (a.scope_id.clone(), a.operation_id.clone());
        match st.ops.get(&key) {
            Some(op) if op.ticket == r.completion_ticket => {}
            Some(_) => return Err(status(Code::FailedPrecondition, wire::TICKET_MISMATCH)),
            None => return Err(status(Code::NotFound, wire::UNKNOWN_OPERATION)),
        }
        let event = |sequence: u64, terminal: bool| pb::WorkEvent {
            worker: a.target.clone(),
            coordinator_term: TERM,
            scope_id: key.0.clone(),
            request_id: r.request_id.clone(),
            attempt: r.attempt,
            sequence,
            completion_ticket: r.completion_ticket.clone(),
            committed_token_offset: sequence - 1,
            bounded_output_event_json: if terminal {
                Vec::new()
            } else {
                b"{\"choices\":[]}".to_vec()
            },
            terminal,
            reason_code: String::new(),
        };
        let events = vec![Ok(event(1, false)), Ok(event(2, true))];
        Ok(Response::new(Box::pin(tokio_stream::iter(events))))
    }

    type StreamTelemetryStream =
        Pin<Box<dyn Stream<Item = Result<pb::TelemetryEnvelope, Status>> + Send>>;

    async fn stream_telemetry(
        &self,
        _: Request<pb::TelemetryRequest>,
    ) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        Err(Status::unimplemented(""))
    }
}

#[tonic::async_trait]
impl WorkerInferenceService for Peer {
    async fn submit_work(
        &self,
        req: Request<pb::SubmitWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.authz.coordinator_caller(&req)?;
        let r = req.into_inner();
        let a = self.authority(&r.authority, true)?;
        if a.scope_id.is_empty() || a.operation_id.is_empty() || a.payload_sha256.len() != 32 {
            return Err(malformed());
        }
        let unit: Value = serde_json::from_slice(&r.work_unit_json).map_err(|_| malformed())?;
        if unit["kind"] != "chat" || !unit["stream"].is_boolean() {
            return Err(malformed());
        }
        let body: Value =
            serde_json::from_slice(&r.bounded_inference_input_json).map_err(|_| malformed())?;
        if body["rid"] != r.request_id.as_str() {
            return Err(malformed());
        }
        let digest = wire::payload_sha256(
            wire::METHOD_SUBMIT,
            &a.scope_id,
            &r.request_id,
            r.attempt,
            &r.work_unit_json,
            &r.bounded_inference_input_json,
        );
        if a.payload_sha256 != digest {
            return Err(status(Code::InvalidArgument, wire::PAYLOAD_DIGEST_MISMATCH));
        }
        let key = (a.scope_id.clone(), a.operation_id.clone());
        let fresh_ticket = self.ticket();
        let mut st = self.state();
        st.seen.submit.push(r.clone());
        let ticket = match st.ops.get(&key) {
            Some(op) if op.payload == digest => op.ticket.clone(),
            Some(_) => return Err(status(Code::AlreadyExists, wire::OPERATION_CONFLICT)),
            None => {
                st.ops.insert(
                    key,
                    Op {
                        payload: digest.to_vec(),
                        ticket: fresh_ticket.clone(),
                    },
                );
                fresh_ticket
            }
        };
        Ok(Response::new(pb::OperationReceipt {
            accepted: true,
            operation_id: a.operation_id.clone(),
            completion_ticket: ticket,
            reason_code: String::new(),
        }))
    }
}

/// One cluster CA, the coordinator leaf the probe presents, and the PEM files.
struct Cluster {
    pki: TestPki,
    dir: StateDir,
    ca: PathBuf,
    cert: PathBuf,
    key: PathBuf,
}

impl Cluster {
    fn new() -> Cluster {
        let pki = TestPki::new();
        let dir = StateDir::new();
        let ca = pki.write_ca(dir.path());
        let (cert, key) = pki.leaf(COORD).write(dir.path());
        Cluster {
            pki,
            dir,
            ca,
            cert,
            key,
        }
    }

    fn node(&self, principal: &str) -> Pki {
        let (cert, key) = self.pki.leaf(principal).write(self.dir.path());
        Pki::load(&self.ca, &cert, &key, principal).unwrap()
    }
}

async fn listener() -> (tokio::net::TcpListener, SocketAddr) {
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

/// The agent listener: one router, both worker services with their decode limits (harness "Wire behaviour").
async fn serve_agent(cluster: &Cluster) -> (SocketAddr, Peer) {
    let pki = cluster.node(WORKER_A);
    let peer = Peer::new(Authz::agent_side(COORD.to_owned()));
    let (listener, addr) = listener().await;
    let router = Server::builder()
        .tls_config(pki.server_tls())
        .unwrap()
        .add_service(
            WorkerServiceServer::new(peer.clone())
                .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX),
        )
        .add_service(
            WorkerInferenceServiceServer::new(peer.clone())
                .max_decoding_message_size(wire::INFERENCE_MESSAGE_MAX),
        );
    tokio::spawn(router.serve_with_incoming(TcpListenerStream::new(listener)));
    (addr, peer)
}

/// The coordinator's registry listener; the probe presents the worker leaf to it.
async fn serve_registry(cluster: &Cluster) -> (SocketAddr, Peer) {
    let pki = cluster.node(COORD);
    let peer = Peer::new(Authz::coordinator_side(BTreeMap::from([(
        WORKER_A.to_owned(),
        WORKER_ID.to_owned(),
    )])));
    let (listener, addr) = listener().await;
    let router = Server::builder()
        .tls_config(pki.server_tls())
        .unwrap()
        .add_service(
            WorkerRegistryServiceServer::new(peer.clone())
                .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX),
        );
    tokio::spawn(router.serve_with_incoming(TcpListenerStream::new(listener)));
    (addr, peer)
}

struct Run {
    code: i32,
    lines: Vec<serde_json::Map<String, Value>>,
    stdout: String,
}

fn s(x: &str) -> String {
    x.to_owned()
}

/// Global flags for a client presenting `cert`/`key` to `peer` at `addr`.
fn globals(ca: &Path, cert: &Path, key: &Path, addr: SocketAddr, peer: &str) -> Vec<String> {
    vec![
        s("--ca"),
        ca.display().to_string(),
        s("--cert"),
        cert.display().to_string(),
        s("--key"),
        key.display().to_string(),
        s("--endpoint"),
        format!("https://{addr}"),
        s("--peer"),
        s(peer),
        s("--timeout-ms"),
        s("5000"),
    ]
}

fn authority_flags(term: u64, generation: Option<u64>) -> Vec<String> {
    let mut v = vec![
        s("--cluster-id"),
        s(CLUSTER),
        s("--worker-id"),
        s(WORKER_ID),
        s("--epoch"),
        EPOCH.to_string(),
        s("--coordinator-id"),
        s(COORD_ID),
        s("--term"),
        term.to_string(),
    ];
    if let Some(g) = generation {
        v.extend([s("--generation"), g.to_string()]);
    }
    v
}

async fn probe(args: Vec<String>) -> Run {
    tokio::task::spawn_blocking(move || {
        let out = Command::new(env!("CARGO_BIN_EXE_cere-rpc-probe"))
            .args(&args)
            .output()
            .unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let lines = stdout
            .lines()
            .map(|l| match serde_json::from_str::<Value>(l) {
                Ok(Value::Object(m)) => m,
                _ => panic!("not a JSON object line: {l:?}"),
            })
            .collect();
        Run {
            code: out.status.code().unwrap_or(-1),
            lines,
            stdout,
        }
    })
    .await
    .unwrap()
}

/// Exactly the pinned keys with their types (describe adds `inventory_sha256`), as `parse_probe` checks them.
fn check_shape(line: &serde_json::Map<String, Value>) {
    let mut keys: Vec<&str> = line.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut want = LINE_KEYS.to_vec();
    if line["call"] == "describe" {
        want.push("inventory_sha256");
        want.sort_unstable();
        let sha = line["inventory_sha256"].as_str().unwrap();
        assert!(
            sha.is_empty()
                || (sha.len() == 64
                    && sha
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())),
            "{line:?}"
        );
    }
    assert_eq!(keys, want, "{line:?}");
    assert!(line["call"].is_string() && line["code"].is_string() && line["reason"].is_string());
    assert!(line["ticket"].is_string() && line["accepted"].is_boolean());
    assert!(line["n"].is_u64() && line["ms"].is_number());
}

fn one(run: &Run) -> &serde_json::Map<String, Value> {
    assert_eq!(run.lines.len(), 1, "{}", run.stdout);
    check_shape(&run.lines[0]);
    &run.lines[0]
}

fn assert_line(
    line: &serde_json::Map<String, Value>,
    call: &str,
    code: &str,
    reason: &str,
    accepted: bool,
) {
    assert_eq!(line["call"], call, "{line:?}");
    assert_eq!(line["code"], code, "{line:?}");
    assert_eq!(line["reason"], reason, "{line:?}");
    assert_eq!(line["accepted"], accepted, "{line:?}");
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

fn submit_args(cluster: &Cluster, addr: SocketAddr, op: &str, extra: &[&str]) -> Vec<String> {
    let mut args = globals(&cluster.ca, &cluster.cert, &cluster.key, addr, WORKER_A);
    args.push(s("submit"));
    args.extend(authority_flags(TERM, Some(GENERATION)));
    args.extend([
        s("--scope"),
        s("probe"),
        s("--op-id"),
        s(op),
        s("--request-id"),
        s(RID),
        s("--max-tokens"),
        s("4"),
    ]);
    args.extend(extra.iter().map(|x| s(x)));
    args
}

fn lifecycle_args(
    cert: &Path,
    key: &Path,
    cluster: &Cluster,
    addr: SocketAddr,
    call: &str,
    extra: &[&str],
) -> Vec<String> {
    let mut args = globals(&cluster.ca, cert, key, addr, WORKER_A);
    args.push(s(call));
    args.extend(authority_flags(TERM, Some(GENERATION)));
    args.extend(extra.iter().map(|x| s(x)));
    args
}

#[tokio::test(flavor = "multi_thread")]
async fn load_drain_unload_send_lock_instance_devices_and_a_verified_digest() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let lock = hex(&LOCK);
    let cases = [
        ("load", wire::METHOD_LOAD, "op-load", "", vec![]),
        ("drain", wire::METHOD_DRAIN, "op-drain", "inst-1", vec!["0"]),
        (
            "unload",
            wire::METHOD_UNLOAD,
            "op-unload",
            "inst-1",
            vec!["0", "1"],
        ),
    ];
    for (n, (call, method, op, instance, devices)) in cases.iter().enumerate() {
        let mut extra = vec!["--scope", "probe", "--op-id", op, "--lock", &lock];
        if !instance.is_empty() {
            extra.extend(["--instance-id", instance]);
        }
        for d in devices {
            extra.extend(["--device-id", d]);
        }
        let run = probe(lifecycle_args(
            &cluster.cert,
            &cluster.key,
            &cluster,
            addr,
            call,
            &extra,
        ))
        .await;
        assert_eq!(run.code, 0, "{}", run.stdout);
        let line = one(&run);
        assert_line(line, call, "OK", "", true);
        assert_eq!(line["n"], 0);
        assert_eq!(line["ticket"], format!("ticket-{n}"));

        let st = peer.state();
        let (seen_method, r) = st.seen.lifecycle.last().unwrap();
        assert_eq!(seen_method, method);
        assert_eq!(r.deployment_lock_sha256, LOCK);
        assert_eq!(r.instance_id, *instance);
        assert_eq!(
            r.device_ids,
            devices.iter().map(|d| s(d)).collect::<Vec<_>>()
        );
        let a = r.authority.clone().unwrap();
        assert_eq!(
            (a.scope_id.as_str(), a.operation_id.as_str()),
            ("probe", *op)
        );
        assert_eq!(a.model_generation, Some(GENERATION));
    }

    // Default scope `_cere_control`.
    let run = probe(lifecycle_args(
        &cluster.cert,
        &cluster.key,
        &cluster,
        addr,
        "load",
        &["--op-id", "op-default", "--lock", &lock],
    ))
    .await;
    assert_line(one(&run), "load", "OK", "", true);
    let st = peer.state();
    let a = st
        .seen
        .lifecycle
        .last()
        .unwrap()
        .1
        .authority
        .clone()
        .unwrap();
    assert_eq!(a.scope_id, "_cere_control");
}

#[tokio::test(flavor = "multi_thread")]
async fn lifecycle_refusals_print_code_and_reason() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let other = hex(&[0xcd; 32]);
    let run = probe(lifecycle_args(
        &cluster.cert,
        &cluster.key,
        &cluster,
        addr,
        "load",
        &["--scope", "probe", "--op-id", "op-1", "--lock", &other],
    ))
    .await;
    assert_eq!(run.code, 0);
    let line = one(&run);
    assert_line(
        line,
        "load",
        "FAILED_PRECONDITION",
        wire::LOCK_NOT_APPROVED,
        false,
    );
    assert_eq!(line["ticket"], "");

    // A 31-byte lock is sent as given (and digested as given): the agent refuses it as malformed.
    let short = hex(&[0xab; 31]);
    let run = probe(lifecycle_args(
        &cluster.cert,
        &cluster.key,
        &cluster,
        addr,
        "drain",
        &["--scope", "probe", "--op-id", "op-2", "--lock", &short],
    ))
    .await;
    assert_line(
        one(&run),
        "drain",
        "INVALID_ARGUMENT",
        wire::MALFORMED_REQUEST,
        false,
    );
    let st = peer.state();
    assert_eq!(st.seen.lifecycle.len(), 2);
    assert_eq!(st.seen.lifecycle[1].1.deployment_lock_sha256, [0xab; 31]);
}

#[tokio::test(flavor = "multi_thread")]
async fn lifecycle_repeat_numbers_lines_and_sends_the_same_op() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let lock = hex(&LOCK);
    let run = probe(lifecycle_args(
        &cluster.cert,
        &cluster.key,
        &cluster,
        addr,
        "unload",
        &["--op-id", "op-r", "--lock", &lock, "--repeat", "3"],
    ))
    .await;
    assert_eq!(run.code, 0, "{}", run.stdout);
    assert_eq!(run.lines.len(), 3, "{}", run.stdout);
    for (n, line) in run.lines.iter().enumerate() {
        check_shape(line);
        assert_line(line, "unload", "OK", "", true);
        assert_eq!(line["n"], n);
        assert_eq!(line["ticket"], "ticket-0");
    }
    let st = peer.state();
    assert_eq!(st.seen.lifecycle.len(), 3);
    let first = &st.seen.lifecycle[0];
    assert!(st.seen.lifecycle.iter().all(|r| r == first));
    assert_eq!(first.1.authority.as_ref().unwrap().operation_id, "op-r");
}

#[tokio::test(flavor = "multi_thread")]
async fn lifecycle_from_a_worker_principal_is_permission_denied() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let (cert, key) = cluster.pki.leaf("cere-worker-b").write(cluster.dir.path());
    let lock = hex(&LOCK);
    let run = probe(lifecycle_args(
        &cert,
        &key,
        &cluster,
        addr,
        "load",
        &["--op-id", "op-w", "--lock", &lock],
    ))
    .await;
    assert_eq!(run.code, 0);
    assert_line(
        one(&run),
        "load",
        "PERMISSION_DENIED",
        wire::PERMISSION_DENIED_REASON,
        false,
    );
    assert!(peer.state().seen.lifecycle.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn register_and_heartbeat_send_the_identity_and_report_reason_codes() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_registry(&cluster).await;
    let (cert, key) = cluster.pki.leaf(WORKER_A).write(cluster.dir.path());
    let g = || globals(&cluster.ca, &cert, &key, addr, COORD);
    let ident = |cmd: &str| -> Vec<String> {
        let mut v = g();
        v.extend([
            s(cmd),
            s("--cluster-id"),
            s(CLUSTER),
            s("--worker-id"),
            s(WORKER_ID),
            s("--epoch"),
            s("5"),
        ]);
        v
    };

    let run = probe(ident("register")).await;
    assert_eq!(run.code, 0);
    let line = one(&run);
    assert_line(line, "register", "OK", "", true);
    assert_eq!(
        (&line["ticket"], &line["n"]),
        (&Value::from(""), &Value::from(0))
    );

    let mut v = ident("register");
    v.extend([s("--revision"), s("3")]);
    let run = probe(v).await;
    assert_line(
        one(&run),
        "register",
        "OK",
        wire::UNSUPPORTED_REVISION,
        false,
    );
    assert_eq!(run.code, 0);

    let mut v = ident("heartbeat");
    v.extend([s("--sequence"), s("9"), s("--generation"), s("4")]);
    let run = probe(v.clone()).await;
    assert_line(one(&run), "heartbeat", "OK", "", true);
    let run = probe(v).await;
    assert_line(
        one(&run),
        "heartbeat",
        "OK",
        wire::REPLAYED_HEARTBEAT,
        false,
    );

    let mut v = ident("heartbeat");
    v.extend([
        s("--sequence"),
        s("10"),
        s("--admission-state"),
        s("fenced"),
    ]);
    assert_line(one(&probe(v).await), "heartbeat", "OK", "", true);

    let st = peer.state();
    let regs = &st.seen.register;
    assert_eq!(regs.len(), 2);
    let id = regs[0].proposed.clone().unwrap();
    assert_eq!(
        (
            id.cluster_id.as_str(),
            id.worker_id.as_str(),
            id.worker_epoch
        ),
        (CLUSTER, WORKER_ID, 5)
    );
    assert_eq!(
        (regs[0].protocol_revision, regs[1].protocol_revision),
        (2, 3)
    );
    for r in regs {
        assert_eq!(r.boot_id.len(), 32);
        assert!(
            r.boot_id.bytes().all(|b| b.is_ascii_hexdigit()),
            "{}",
            r.boot_id
        );
    }
    assert_ne!(
        regs[0].boot_id, regs[1].boot_id,
        "boot id is random per process"
    );
    let hbs = &st.seen.heartbeat;
    assert_eq!(hbs.len(), 3);
    assert_eq!((hbs[0].sequence, hbs[0].model_generation), (9, 4));
    assert_eq!(hbs[0].admission_state, "open");
    assert_eq!(hbs[0].worker.clone().unwrap().worker_epoch, 5);
    assert_eq!(
        (hbs[2].sequence, hbs[2].admission_state.as_str()),
        (10, "fenced")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn registry_calls_from_an_unbound_principal_are_permission_denied() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_registry(&cluster).await;
    let (cert, key) = cluster.pki.leaf("cere-worker-b").write(cluster.dir.path());
    let mut v = globals(&cluster.ca, &cert, &key, addr, COORD);
    v.extend(
        [
            "register",
            "--cluster-id",
            CLUSTER,
            "--worker-id",
            WORKER_ID,
            "--epoch",
            "1",
        ]
        .map(s),
    );
    let run = probe(v).await;
    assert_eq!(run.code, 0);
    assert_line(
        one(&run),
        "register",
        "PERMISSION_DENIED",
        wire::PERMISSION_DENIED_REASON,
        false,
    );
    assert!(peer.state().seen.register.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn describe_reports_the_inventory_digest_and_list_reads_one_page() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let g = || globals(&cluster.ca, &cluster.cert, &cluster.key, addr, WORKER_A);

    // `--generation` is accepted on reads but never sent (reads carry no generation, creg-notes 2).
    let mut v = g();
    v.push(s("describe"));
    v.extend(authority_flags(TERM, Some(GENERATION)));
    let run = probe(v).await;
    assert_eq!(run.code, 0);
    let line = one(&run);
    assert_line(line, "describe", "OK", "", true);
    let want: [u8; 32] = wire::Inventory::sha256(INVENTORY);
    assert_eq!(line["inventory_sha256"], hex(&want));
    assert_eq!(line["ticket"], "");

    let mut v = g();
    v.push(s("describe"));
    v.extend(authority_flags(TERM - 1, None));
    let run = probe(v).await;
    let line = one(&run);
    assert_line(
        line,
        "describe",
        "FAILED_PRECONDITION",
        wire::STALE_TERM,
        false,
    );
    assert_eq!(line["inventory_sha256"], "");

    let mut v = g();
    v.push(s("list"));
    v.extend(authority_flags(TERM, Some(GENERATION)));
    let run = probe(v).await;
    assert_eq!(run.code, 0);
    assert_line(one(&run), "list", "OK", "", true);

    let st = peer.state();
    assert_eq!(st.seen.describe.len(), 1);
    let a = st.seen.describe[0].authority.clone().unwrap();
    assert_eq!(
        (a.coordinator_id.as_str(), a.coordinator_term),
        (COORD_ID, TERM)
    );
    assert_eq!(a.model_generation, None);
    assert!(a.scope_id.is_empty() && a.operation_id.is_empty() && a.payload_sha256.is_empty());
    assert_eq!(st.seen.list.len(), 1);
    assert_eq!(st.seen.list[0].limit as usize, wire::LIST_PAGE_MAX);
}

#[tokio::test(flavor = "multi_thread")]
async fn submit_repeat_numbers_lines_and_returns_one_ticket() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let run = probe(submit_args(
        &cluster,
        addr,
        RID,
        &["--repeat", "3", "--pad-bytes", "64"],
    ))
    .await;
    assert_eq!(run.code, 0, "{}", run.stdout);
    assert_eq!(run.lines.len(), 3, "{}", run.stdout);
    for (n, line) in run.lines.iter().enumerate() {
        check_shape(line);
        assert_line(line, "submit", "OK", "", true);
        assert_eq!(line["n"], n);
        assert_eq!(line["ticket"], "ticket-0");
    }

    let st = peer.state();
    assert_eq!(st.seen.submit.len(), 3);
    let first = &st.seen.submit[0];
    assert!(
        st.seen.submit.iter().all(|r| r == first),
        "repeats are identical"
    );
    let a = first.authority.clone().unwrap();
    assert_eq!(
        (a.scope_id.as_str(), a.operation_id.as_str()),
        ("probe", RID)
    );
    assert_eq!(a.model_generation, Some(GENERATION));
    assert!(a.remaining_budget_ms > 0);
    assert_eq!((first.request_id.as_str(), first.attempt), (RID, 1));
    assert_eq!(first.work_unit_json, br#"{"kind":"chat","stream":false}"#);
    let body: Value = serde_json::from_slice(&first.bounded_inference_input_json).unwrap();
    assert_eq!(body["rid"], RID);
    assert_eq!(body["stream"], false);
    assert_eq!(body["max_completion_tokens"], 4);
    let content = body["messages"][0]["content"].as_str().unwrap();
    assert_eq!(content.len(), 64);
    assert_eq!(content.trim_end(), "probe", "synthetic content only");
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn submit_conflict_changes_only_the_message_content() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    assert_line(
        one(&probe(submit_args(&cluster, addr, RID, &[])).await),
        "submit",
        "OK",
        "",
        true,
    );
    let run = probe(submit_args(&cluster, addr, RID, &["--conflict"])).await;
    assert_eq!(run.code, 0);
    let line = one(&run);
    assert_line(
        line,
        "submit",
        "ALREADY_EXISTS",
        wire::OPERATION_CONFLICT,
        false,
    );
    assert_eq!(line["ticket"], "");

    let st = peer.state();
    let (a, b) = (&st.seen.submit[0], &st.seen.submit[1]);
    let mut ba: Value = serde_json::from_slice(&a.bounded_inference_input_json).unwrap();
    let mut bb: Value = serde_json::from_slice(&b.bounded_inference_input_json).unwrap();
    assert_ne!(ba["messages"][0]["content"], bb["messages"][0]["content"]);
    ba["messages"][0]["content"] = Value::Null;
    bb["messages"][0]["content"] = Value::Null;
    assert_eq!(ba, bb);
    assert_eq!(a.request_id, b.request_id);
    assert_eq!(a.attempt, b.attempt);
    assert_eq!(a.work_unit_json, b.work_unit_json);
    let (aa, ab) = (a.authority.clone().unwrap(), b.authority.clone().unwrap());
    assert_ne!(aa.payload_sha256, ab.payload_sha256);
    assert_eq!(
        pb::Authority {
            payload_sha256: vec![],
            ..aa
        },
        pb::Authority {
            payload_sha256: vec![],
            ..ab
        }
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn submit_content_is_sent_and_conflict_still_differs() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let text = "Count from 1 to 5.";
    let run = probe(submit_args(&cluster, addr, RID, &["--content", text])).await;
    assert_line(one(&run), "submit", "OK", "", true);
    let run = probe(submit_args(
        &cluster,
        addr,
        RID,
        &["--content", text, "--conflict"],
    ))
    .await;
    assert_line(
        one(&run),
        "submit",
        "ALREADY_EXISTS",
        wire::OPERATION_CONFLICT,
        false,
    );
    assert!(!run.stdout.contains(text), "content is never printed");
    let st = peer.state();
    let content = |r: &pb::SubmitWorkRequest| {
        let body: Value = serde_json::from_slice(&r.bounded_inference_input_json).unwrap();
        body["messages"][0]["content"].as_str().unwrap().to_owned()
    };
    assert_eq!(content(&st.seen.submit[0]), text);
    let conflict = content(&st.seen.submit[1]);
    assert_ne!(conflict, text);
    assert!(conflict.starts_with(text), "{conflict}");
}

#[tokio::test(flavor = "multi_thread")]
async fn submit_stream_adds_a_stream_line_and_model_and_attempt_are_sent() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let run = probe(submit_args(
        &cluster,
        addr,
        RID,
        &["--stream", "--model", "m1", "--attempt", "2"],
    ))
    .await;
    assert_eq!(run.code, 0, "{}", run.stdout);
    assert_eq!(run.lines.len(), 2, "{}", run.stdout);
    run.lines.iter().for_each(check_shape);
    assert_line(&run.lines[0], "submit", "OK", "", true);
    assert_line(&run.lines[1], "stream", "OK", "", true);

    let st = peer.state();
    let r = &st.seen.submit[0];
    assert_eq!(r.work_unit_json, br#"{"kind":"chat","stream":true}"#);
    assert_eq!(r.attempt, 2);
    let body: Value = serde_json::from_slice(&r.bounded_inference_input_json).unwrap();
    assert_eq!(
        (&body["model"], &body["stream"]),
        (&Value::from("m1"), &Value::from(true))
    );
    assert_eq!(st.seen.stream.len(), 1);
    let w = &st.seen.stream[0];
    assert_eq!(
        (w.request_id.as_str(), w.attempt, w.after_sequence),
        (RID, 2, 0)
    );
    assert_eq!(
        w.completion_ticket,
        run.lines[0]["ticket"].as_str().unwrap()
    );
    let a = w.authority.clone().unwrap();
    assert_eq!(
        (a.scope_id.as_str(), a.operation_id.as_str()),
        ("probe", RID)
    );
    assert_eq!(a.model_generation, Some(GENERATION));
}

#[tokio::test(flavor = "multi_thread")]
async fn submit_without_generation_is_stale_generation() {
    let cluster = Cluster::new();
    let (addr, _peer) = serve_agent(&cluster).await;
    let mut v = globals(&cluster.ca, &cluster.cert, &cluster.key, addr, WORKER_A);
    v.push(s("submit"));
    v.extend(authority_flags(TERM, None));
    v.extend(["--scope", "probe", "--op-id", RID, "--request-id", RID].map(s));
    let run = probe(v).await;
    assert_eq!(run.code, 0);
    assert_line(
        one(&run),
        "submit",
        "FAILED_PRECONDITION",
        wire::STALE_GENERATION,
        false,
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_sends_the_given_operation_id_and_the_ticket() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let submit = probe(submit_args(&cluster, addr, RID, &[])).await;
    let ticket = one(&submit)["ticket"].as_str().unwrap().to_owned();
    let cancel_op = format!("{RID}/cancel/1");
    let cancel = |ticket: &str| {
        let mut v = globals(&cluster.ca, &cluster.cert, &cluster.key, addr, WORKER_A);
        v.push(s("cancel"));
        v.extend(authority_flags(TERM, Some(GENERATION)));
        v.extend([
            s("--scope"),
            s("probe"),
            s("--op-id"),
            cancel_op.clone(),
            s("--request-id"),
            s(RID),
            s("--ticket"),
            s(ticket),
        ]);
        v
    };
    let run = probe(cancel(&ticket)).await;
    assert_eq!(run.code, 0, "{}", run.stdout);
    let line = one(&run);
    assert_line(line, "cancel", "OK", "", true);
    assert_eq!(line["ticket"], ticket.as_str());

    let run = probe(cancel("ticket-999")).await;
    assert_line(
        one(&run),
        "cancel",
        "FAILED_PRECONDITION",
        wire::TICKET_MISMATCH,
        false,
    );

    let st = peer.state();
    assert_eq!(st.seen.cancel.len(), 2);
    let r = &st.seen.cancel[0];
    assert_eq!((r.request_id.as_str(), r.attempt), (RID, 1));
    assert_eq!(r.completion_ticket, ticket);
    let a = r.authority.clone().unwrap();
    assert_eq!(
        (a.scope_id.as_str(), a.operation_id.as_str()),
        ("probe", cancel_op.as_str())
    );
    assert_eq!(a.model_generation, Some(GENERATION));
}

#[tokio::test(flavor = "multi_thread")]
async fn oversize_messages_are_out_of_range_on_both_services() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    for (service, bytes) in [
        ("control", wire::CONTROL_MESSAGE_MAX + 1),
        ("inference", wire::INFERENCE_MESSAGE_MAX + 1),
    ] {
        let mut v = globals(&cluster.ca, &cluster.cert, &cluster.key, addr, WORKER_A);
        v.push(s("oversize"));
        v.extend(authority_flags(TERM, Some(GENERATION)));
        v.extend([s("--service"), s(service), s("--bytes"), bytes.to_string()]);
        let run = probe(v).await;
        assert_eq!(run.code, 0, "{}", run.stdout);
        let line = one(&run);
        assert_eq!(line["call"], "oversize");
        assert_eq!(line["code"], "OUT_OF_RANGE", "{service}: {line:?}");
        assert_eq!(line["accepted"], false);
    }
    let st = peer.state();
    assert!(st.seen.describe.is_empty() && st.seen.submit.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn foreign_ca_is_transport_handshake_and_closed_port_is_transport_connect() {
    let cluster = Cluster::new();
    let (addr, peer) = serve_agent(&cluster).await;
    let foreign = TestPki::new();
    let (cert, key) = foreign.leaf(COORD).write(cluster.dir.path());
    let mut v = globals(&cluster.ca, &cert, &key, addr, WORKER_A);
    v.push(s("describe"));
    v.extend(authority_flags(TERM, None));
    let run = probe(v).await;
    assert_eq!(run.code, 0, "{}", run.stdout);
    let line = one(&run);
    assert_line(line, "describe", "TRANSPORT", "handshake", false);
    assert_eq!(line["inventory_sha256"], "");
    assert!(peer.state().seen.describe.is_empty());

    // The server certificate does not name `--peer`: the client refuses the handshake.
    let mut v = globals(
        &cluster.ca,
        &cluster.cert,
        &cluster.key,
        addr,
        "cere-worker-z",
    );
    v.push(s("list"));
    v.extend(authority_flags(TERM, None));
    let run = probe(v).await;
    assert_eq!(run.code, 0, "{}", run.stdout);
    assert_line(one(&run), "list", "TRANSPORT", "handshake", false);
    assert!(peer.state().seen.list.is_empty());

    let closed = {
        let (l, a) = listener().await;
        drop(l);
        a
    };
    let mut v = globals(&cluster.ca, &cluster.cert, &cluster.key, closed, WORKER_A);
    v.push(s("submit"));
    v.extend(authority_flags(TERM, Some(GENERATION)));
    v.extend(["--scope", "probe", "--op-id", RID, "--request-id", RID].map(s));
    let run = probe(v).await;
    assert_eq!(run.code, 0, "{}", run.stdout);
    let line = one(&run);
    assert_line(line, "submit", "TRANSPORT", "connect", false);
    assert_eq!(line["ticket"], "");
}

#[tokio::test(flavor = "multi_thread")]
async fn usage_errors_exit_2_without_lines() {
    let cluster = Cluster::new();
    let addr: SocketAddr = (Ipv4Addr::LOCALHOST, 9).into();
    let g = || globals(&cluster.ca, &cluster.cert, &cluster.key, addr, WORKER_A);
    let mut cases: Vec<Vec<String>> = vec![vec![], g()];
    let mut unknown = g();
    unknown.push(s("frobnicate"));
    cases.push(unknown);
    let mut missing = g();
    missing.push(s("describe"));
    missing.extend(authority_flags(TERM, None)[..4].iter().cloned());
    cases.push(missing);
    let mut bad_service = g();
    bad_service.push(s("oversize"));
    bad_service.extend(authority_flags(TERM, None));
    bad_service.extend(["--service", "bulk", "--bytes", "10"].map(s));
    cases.push(bad_service);
    let mut bad_number = g();
    bad_number.push(s("register"));
    bad_number.extend(
        [
            "--cluster-id",
            CLUSTER,
            "--worker-id",
            WORKER_ID,
            "--epoch",
            "x",
        ]
        .map(s),
    );
    cases.push(bad_number);
    let mut unknown_flag = g();
    unknown_flag.push(s("list"));
    unknown_flag.extend(authority_flags(TERM, None));
    unknown_flag.push(s("--bogus"));
    cases.push(unknown_flag);
    let mut no_global = g()[2..].to_vec();
    no_global.push(s("list"));
    no_global.extend(authority_flags(TERM, None));
    cases.push(no_global);
    let lc = |extra: &[&str]| {
        let mut v = g();
        v.push(s("load"));
        v.extend(authority_flags(TERM, Some(GENERATION)));
        v.extend(extra.iter().map(|x| s(x)));
        v
    };
    cases.push(lc(&["--op-id", "op"]));
    cases.push(lc(&["--op-id", "op", "--lock", "zz"]));
    cases.push(lc(&["--op-id", "op", "--lock", "abc"]));
    cases.push(lc(&["--lock", "ab"]));
    cases.push(lc(&["--op-id", "op", "--lock", "ab", "--repeat", "0"]));
    for args in cases {
        let run = probe(args.clone()).await;
        assert_eq!(run.code, 2, "{args:?}: {}", run.stdout);
        assert!(run.lines.is_empty(), "{args:?}");
    }
}
