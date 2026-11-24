//! In-process stub of the M3.5 coordinator registry (`WorkerRegistryService`) for the agent tests (track LINK,
//! build-loop-m35 design note): it enforces the coordinator's Register/Heartbeat rules (m35 design note Q2, Q4, Q6,
//! Q7, Q7a) so the agent is tested against them, and lets a test script the replies a real coordinator produces in
//! its other states (a restart at a higher term, a Lost worker, a drain intent, finished reconciliation, silence).
//!
//! Rules, in handler order: `Authz::coordinator_side` (the worker principal bound to exactly the claimed id) before
//! any state; a missing identity is `INVALID_ARGUMENT malformed_request`. Register: `protocol_revision != 2` ->
//! `unsupported_revision`; another `cluster_id` -> `wrong_cluster`; an epoch below the journaled one ->
//! `stale_epoch` + `journaled_worker_epoch`; equal and Lost -> `epoch_lost`; equal and not Lost -> reconnect (Joining
//! on the first Register of a term); higher -> a new incarnation (journaled, Joining, heartbeat sequence reset).
//! Heartbeat: not registered in the current term -> `unknown_worker`; `observed_coordinator_term` not the current
//! term -> `stale_term`; a lower epoch -> `stale_epoch`, a higher one -> `unknown_worker`; Lost (no accepted heartbeat
//! for `lost_after`, receive time) -> `epoch_lost`; a sequence not above the last accepted one for that epoch ->
//! `replayed_heartbeat` (no aging update); else accepted with a grant whose `lease_sequence` restarts at 1 per term
//! (the order of `fed::registry`). Journaled epochs survive [`StubCoordinator::restart`] (they are
//! journaled); registrations and the lease sequence do not.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use cere_proto::worker::v2 as pb;
use cere_runtime::fed::tls::{Authz, Pki};
use cere_runtime::fed::wire;
use pb::worker_registry_service_server::{WorkerRegistryService, WorkerRegistryServiceServer};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

/// The stub's fixed configuration.
#[derive(Debug, Clone)]
pub struct StubConfig {
    pub cluster_id: String,
    /// The first coordinator term (the journaled process epoch of `cered`).
    pub term: u64,
    pub admission_lease_ms: u64,
    /// Absolute age of the last accepted heartbeat at which a worker is Lost (`lost_after_ms`).
    pub lost_after: Duration,
    /// Worker principal -> the one worker id it may claim.
    pub bindings: BTreeMap<String, String>,
}

/// One Register the stub answered (after authorization).
#[derive(Debug, Clone)]
pub struct SeenRegister {
    pub at: Instant,
    pub worker_id: String,
    pub epoch: u64,
    pub boot_id: String,
    pub inventory_sha256: Vec<u8>,
    pub accepted: bool,
    pub reason: String,
    pub term: u64,
}

/// One Heartbeat the stub answered (after authorization).
#[derive(Debug, Clone)]
pub struct SeenHeartbeat {
    pub at: Instant,
    pub worker_id: String,
    pub epoch: u64,
    pub sequence: u64,
    pub observed_term: u64,
    pub model_generation: u64,
    pub admission_state: String,
    pub inventory_sha256: Vec<u8>,
    pub accepted: bool,
    pub reason: String,
    pub term: u64,
    pub lease_sequence: u64,
    pub reconciled_generation: Option<u64>,
}

#[derive(Debug, Default)]
struct WorkerSlot {
    journaled_epoch: u64,
    /// `(epoch, term)` of the accepted registration; `None` until a Register is accepted in the current term.
    registered: Option<(u64, u64)>,
    last_sequence: u64,
    last_seen: Option<Instant>,
    forced_lost: bool,
}

impl WorkerSlot {
    fn lost(&self, lost_after: Duration, now: Instant) -> bool {
        self.forced_lost
            || self
                .last_seen
                .is_some_and(|seen| now.duration_since(seen) >= lost_after)
    }
}

/// Test-scripted reply knobs.
#[derive(Debug, Default)]
struct Script {
    reconcile: bool,
    draining: bool,
    silent: bool,
    admission_lease_ms: u64,
    /// The next grant uses this sequence (the counter continues from it): replays, older or jumped grants.
    next_lease_sequence: Option<u64>,
    /// The next reply carries this `coordinator_term` instead of the current one (a stale coordinator).
    next_reply_term: Option<u64>,
}

struct State {
    config: StubConfig,
    term: u64,
    lease_sequence: u64,
    script: Script,
    workers: BTreeMap<String, WorkerSlot>,
    registers: Vec<SeenRegister>,
    heartbeats: Vec<SeenHeartbeat>,
}

#[derive(Clone)]
struct Registry {
    authz: Authz,
    state: Arc<Mutex<State>>,
}

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(|p| p.into_inner())
}

fn malformed() -> Status {
    wire::status(Code::InvalidArgument, wire::MALFORMED_REQUEST)
}

impl Registry {
    async fn silence(&self) {
        if lock(&self.state).script.silent {
            std::future::pending::<()>().await;
        }
    }
}

#[tonic::async_trait]
impl WorkerRegistryService for Registry {
    async fn register(
        &self,
        req: Request<pb::RegisterRequest>,
    ) -> Result<Response<pb::RegisterReply>, Status> {
        let claimed = req
            .get_ref()
            .proposed
            .as_ref()
            .map(|p| p.worker_id.clone())
            .unwrap_or_default();
        self.authz.registry_caller(&req, &claimed)?;
        self.silence().await;
        let req = req.into_inner();
        let proposed = req.proposed.ok_or_else(malformed)?;
        let now = Instant::now();
        let mut guard = lock(&self.state);
        let st = &mut *guard;
        let term = st.term;
        let refuse = |reason: &str, journaled: u64| pb::RegisterReply {
            accepted: false,
            coordinator_term: term,
            reason_code: reason.to_owned(),
            journaled_worker_epoch: journaled,
            ..Default::default()
        };
        let mut reply = if req.protocol_revision != cere_proto::PROTOCOL_REVISION {
            refuse(wire::UNSUPPORTED_REVISION, 0)
        } else if proposed.cluster_id != st.config.cluster_id {
            refuse(wire::WRONG_CLUSTER, 0)
        } else {
            let lost_after = st.config.lost_after;
            let slot = st.workers.entry(proposed.worker_id.clone()).or_default();
            let epoch = proposed.worker_epoch;
            if epoch < slot.journaled_epoch {
                refuse(wire::STALE_EPOCH, slot.journaled_epoch)
            } else if epoch == slot.journaled_epoch && slot.lost(lost_after, now) {
                refuse(wire::EPOCH_LOST, 0)
            } else {
                let joining = if epoch > slot.journaled_epoch {
                    slot.journaled_epoch = epoch;
                    slot.last_sequence = 0;
                    true
                } else {
                    // Reconnect: the state is kept when already registered under this term.
                    slot.registered != Some((epoch, term))
                };
                if slot
                    .registered
                    .is_some_and(|(e, t)| e != epoch || t != term)
                {
                    slot.last_sequence = 0;
                }
                slot.registered = Some((epoch, term));
                slot.last_seen = Some(now);
                slot.forced_lost = false;
                pb::RegisterReply {
                    accepted: true,
                    coordinator_term: term,
                    admission_lease_ms: st.script.admission_lease_ms,
                    requires_reconciliation: joining,
                    ..Default::default()
                }
            }
        };
        if let Some(t) = st.script.next_reply_term.take() {
            reply.coordinator_term = t;
        }
        st.registers.push(SeenRegister {
            at: now,
            worker_id: proposed.worker_id,
            epoch: proposed.worker_epoch,
            boot_id: req.boot_id,
            inventory_sha256: req.inventory_sha256,
            accepted: reply.accepted,
            reason: reply.reason_code.clone(),
            term: reply.coordinator_term,
        });
        Ok(Response::new(reply))
    }

    async fn heartbeat(
        &self,
        req: Request<pb::HeartbeatRequest>,
    ) -> Result<Response<pb::HeartbeatReply>, Status> {
        let claimed = req
            .get_ref()
            .worker
            .as_ref()
            .map(|w| w.worker_id.clone())
            .unwrap_or_default();
        self.authz.registry_caller(&req, &claimed)?;
        self.silence().await;
        let req = req.into_inner();
        let worker = req.worker.ok_or_else(malformed)?;
        let now = Instant::now();
        let mut guard = lock(&self.state);
        let st = &mut *guard;
        let term = st.term;
        let lost_after = st.config.lost_after;
        let refuse = |reason: &str| pb::HeartbeatReply {
            accepted: false,
            coordinator_term: term,
            reason_code: reason.to_owned(),
            ..Default::default()
        };
        let slot = st.workers.entry(worker.worker_id.clone()).or_default();
        let mut reply = match slot.registered {
            _ if worker.cluster_id != st.config.cluster_id => refuse(wire::WRONG_CLUSTER),
            Some((epoch, t)) if t == term => {
                if req.observed_coordinator_term != term {
                    refuse(wire::STALE_TERM)
                } else if worker.worker_epoch < epoch {
                    refuse(wire::STALE_EPOCH)
                } else if worker.worker_epoch > epoch {
                    refuse(wire::UNKNOWN_WORKER)
                } else if slot.lost(lost_after, now) {
                    refuse(wire::EPOCH_LOST)
                } else if req.sequence <= slot.last_sequence {
                    refuse(wire::REPLAYED_HEARTBEAT)
                } else {
                    slot.last_sequence = req.sequence;
                    slot.last_seen = Some(now);
                    let sequence = st
                        .script
                        .next_lease_sequence
                        .take()
                        .unwrap_or(st.lease_sequence + 1);
                    st.lease_sequence = sequence;
                    pb::HeartbeatReply {
                        accepted: true,
                        coordinator_term: term,
                        lease_sequence: sequence,
                        admission_lease_ms: st.script.admission_lease_ms,
                        desired_state: if st.script.draining {
                            wire::DESIRED_DRAINING.to_owned()
                        } else {
                            String::new()
                        },
                        reconciled_generation: st.script.reconcile.then_some(req.model_generation),
                        reason_code: String::new(),
                    }
                }
            }
            _ => refuse(wire::UNKNOWN_WORKER),
        };
        if let Some(t) = st.script.next_reply_term.take() {
            reply.coordinator_term = t;
        }
        st.heartbeats.push(SeenHeartbeat {
            at: now,
            worker_id: worker.worker_id,
            epoch: worker.worker_epoch,
            sequence: req.sequence,
            observed_term: req.observed_coordinator_term,
            model_generation: req.model_generation,
            admission_state: req.admission_state,
            inventory_sha256: req.inventory_sha256,
            accepted: reply.accepted,
            reason: reply.reason_code.clone(),
            term: reply.coordinator_term,
            lease_sequence: reply.lease_sequence,
            reconciled_generation: reply.reconciled_generation,
        });
        Ok(Response::new(reply))
    }
}

/// A running stub registry on an ephemeral loopback port (mTLS with the coordinator's PKI). Dropping it stops the
/// server.
pub struct StubCoordinator {
    addr: SocketAddr,
    state: Arc<Mutex<State>>,
    task: tokio::task::JoinHandle<()>,
}

impl StubCoordinator {
    pub async fn start(pki: &Pki, config: StubConfig) -> StubCoordinator {
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(Mutex::new(State {
            term: config.term,
            lease_sequence: 0,
            script: Script {
                admission_lease_ms: config.admission_lease_ms,
                ..Script::default()
            },
            workers: BTreeMap::new(),
            registers: Vec::new(),
            heartbeats: Vec::new(),
            config: config.clone(),
        }));
        let svc = WorkerRegistryServiceServer::new(Registry {
            authz: Authz::coordinator_side(config.bindings),
            state: Arc::clone(&state),
        })
        .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX);
        let server = Server::builder()
            .tls_config(pki.server_tls())
            .unwrap()
            .add_service(svc)
            .serve_with_incoming(TcpListenerStream::new(listener));
        let task = tokio::spawn(async move {
            let _ = server.await;
        });
        StubCoordinator { addr, state, task }
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// `https://127.0.0.1:<port>` (the agent's `[coordinator] endpoint`).
    pub fn endpoint(&self) -> String {
        format!("https://{}", self.addr)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    pub fn term(&self) -> u64 {
        self.lock().term
    }

    /// A `cered` restart: the term goes up by one, registrations and the lease sequence are forgotten, the
    /// journaled epochs are kept.
    pub fn restart(&self) {
        let mut st = self.lock();
        st.term += 1;
        st.lease_sequence = 0;
        for slot in st.workers.values_mut() {
            slot.registered = None;
            slot.last_sequence = 0;
            slot.last_seen = None;
            slot.forced_lost = false;
        }
    }

    /// Reconciliation finished: every accepted heartbeat reply carries `reconciled_generation` = the reported
    /// generation while set.
    pub fn set_reconcile(&self, on: bool) {
        self.lock().script.reconcile = on;
    }

    /// A drain intent: accepted replies carry `desired_state = "DRAINING"` while set.
    pub fn set_draining(&self, on: bool) {
        self.lock().script.draining = on;
    }

    /// While set, Register and Heartbeat never answer (a partition after the TLS handshake).
    pub fn set_silent(&self, on: bool) {
        self.lock().script.silent = on;
    }

    /// Register and accepted Heartbeat replies from now on grant `ms` leases.
    pub fn set_admission_lease_ms(&self, ms: u64) {
        self.lock().script.admission_lease_ms = ms;
    }

    /// The next grant uses `sequence` and the counter continues from it.
    pub fn force_lease_sequence(&self, sequence: u64) {
        self.lock().script.next_lease_sequence = Some(sequence);
    }

    /// The next grant replays the last granted sequence (the counter continues after it); returns that sequence.
    pub fn replay_last_grant(&self) -> u64 {
        let mut st = self.lock();
        let last = st.lease_sequence;
        st.script.next_lease_sequence = Some(last);
        last
    }

    /// The next reply (Register or Heartbeat) carries `term` as its `coordinator_term`.
    pub fn force_reply_term(&self, term: u64) {
        self.lock().script.next_reply_term = Some(term);
    }

    /// The worker turns Lost now (as if its last heartbeat were `lost_after` old).
    pub fn mark_lost(&self, worker_id: &str) {
        self.lock()
            .workers
            .entry(worker_id.to_owned())
            .or_default()
            .forced_lost = true;
    }

    /// Sets the journaled epoch (a coordinator that saw a later incarnation before this agent's state was lost).
    pub fn set_journaled_epoch(&self, worker_id: &str, epoch: u64) {
        self.lock()
            .workers
            .entry(worker_id.to_owned())
            .or_default()
            .journaled_epoch = epoch;
    }

    pub fn journaled_epoch(&self, worker_id: &str) -> u64 {
        self.lock()
            .workers
            .get(worker_id)
            .map_or(0, |slot| slot.journaled_epoch)
    }

    pub fn registers(&self) -> Vec<SeenRegister> {
        self.lock().registers.clone()
    }

    pub fn heartbeats(&self) -> Vec<SeenHeartbeat> {
        self.lock().heartbeats.clone()
    }
}

impl Drop for StubCoordinator {
    fn drop(&mut self) {
        self.task.abort();
    }
}
