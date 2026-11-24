//! The coordinator's `WorkerRegistryService` and the state of every remote worker (m35 design note Q4, Q6, Q7,
//! Q7a, Q13, Q14; build-loop-m35 design note "CREG split" D2, D5).
//!
//! Remote workers live here, not in [`crate::worker::WorkerRegistry`]: their epochs are minted by their agents,
//! sightings are accepted heartbeats (aged by coordinator receive time), and Joining -> Ready is driven by
//! reconciliation and the readiness smoke. The term is the ledger's process epoch: one [`RemoteRegistry`] lives
//! for one term, so its per-worker lease sequence (starting at 1) never restarts within a term.
//!
//! States: a configured worker reads Joining until its first accepted Register of this term. An accepted Register
//! that starts an incarnation (a higher epoch, journaled through [`Ledger::accept_worker_epoch`] before the reply)
//! or the term (the journaled epoch, first seen in this term) resets the remote state and enters Joining; a same
//! epoch Register while registered is a reconnect and keeps it. Joining ages to Lost at the absolute lost bound
//! since its last accepted heartbeat (or the Register that started the epoch); Ready ages to Suspect (silence,
//! or its lease view expired) and to Lost by [`WorkerView::state_at`] with `Aging { suspect_after_ms,
//! lost_after_ms - suspect_after_ms }`. Lost is recorded the first time a call observes it; a Lost epoch never
//! returns. Each entry into Joining starts a new *round*: the driver's transitions name the round they were
//! computed for, so a stale reconciliation or smoke never promotes a later incarnation. Drain intent belongs to
//! the worker id, not to an epoch: it survives Lost and every new incarnation and only an undrain clears it.
//!
//! Model residency (m36 design note Q8-Q10): the last described model (`managed`, state, lock, and the epoch it
//! was described at) and the operator's journaled residency intent ([`RemoteRegistry::set_residency`]) also belong
//! to the worker id: they survive `join()`, Lost and every re-register (review M36-13). A managed worker is Ready
//! only with its model `ready` under the coordinator's deployment lock and `desired != unloaded`; `unloaded` makes
//! it ineligible at once. An attached worker is not gated by the intent, which stays journaled as it is.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::Duration;

use cere_proto::worker::v2 as pb;
use pb::worker_registry_service_server::{WorkerRegistryService, WorkerRegistryServiceServer};
use serde::Serialize;
use tokio::sync::{oneshot, watch};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

use crate::app::unix_ms;
use crate::config::FederationConfig;
use crate::fed::quota::PrincipalQuota;
use crate::fed::tls::{Authz, Pki};
use crate::fed::wire::{self, AdmissionState, Inventory, ModelState};
use crate::journal::Residency;
use crate::ledger::{Ledger, LedgerError};
use crate::readiness::SmokeStatus;
use crate::worker::{Aging, WorkerHealth, WorkerState, WorkerView};

/// How long [`RegistryListener::stop`] waits for open connections to close before it aborts the server.
const LISTENER_STOP_GRACE: Duration = Duration::from_secs(2);

/// The coordinator-side timings of remote workers (Q7), in milliseconds; `lost_after_ms` is absolute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteTiming {
    pub suspect_after_ms: u64,
    pub lost_after_ms: u64,
    pub lease_ms: u64,
}

impl RemoteTiming {
    pub fn from_config(fed: &FederationConfig) -> RemoteTiming {
        RemoteTiming {
            suspect_after_ms: fed.suspect_after().as_millis() as u64,
            lost_after_ms: fed.lost_after().as_millis() as u64,
            lease_ms: fed.admission_lease().as_millis() as u64,
        }
    }

    /// The per-worker aging rule of a remote worker: Lost at the absolute `lost_after_ms` (Q7).
    pub fn aging(self) -> Aging {
        Aging {
            suspect_after_ms: self.suspect_after_ms,
            lost_after_ms: self.lost_after_ms.saturating_sub(self.suspect_after_ms),
        }
    }
}

/// A copy of one remote worker's state, aged at the instant it was read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteState {
    pub worker_id: String,
    /// Joining, Ready, Draining (Ready with drain intent), Suspect or Lost. A Ready worker whose lease view
    /// expired reads Suspect (no new work).
    pub state: WorkerState,
    /// Operator drain intent of the worker id (Q13).
    pub draining: bool,
    /// The current Joining round; driver transitions for another round are refused.
    pub round: u64,
    /// The term it registered under; `None` before an accepted Register in this term.
    pub term: Option<u64>,
    /// The epoch registered in this term.
    pub epoch: Option<u64>,
    /// The agent's `model_generation` from its last accepted heartbeat.
    pub generation: Option<u64>,
    /// The agent's `admission_state` from its last accepted heartbeat.
    pub admission_state: Option<AdmissionState>,
    /// The `inventory_sha256` the agent last reported (Register or heartbeat).
    pub reported_inventory_sha256: Option<[u8; 32]>,
    /// The inventory read by the driver (`DescribeResources`) in this round, and the SHA-256 of its exact bytes.
    pub inventory: Option<Inventory>,
    pub inventory_sha256: Option<[u8; 32]>,
    /// The last lease grant's sequence (0 = none in this term).
    pub lease_sequence: u64,
    /// The coordinator-side lease view: receive time of the last accepted heartbeat + the lease.
    pub lease_until_ms: Option<u64>,
    /// The generation reconciled in this round, sent in every heartbeat reply while set.
    pub reconciled_generation: Option<u64>,
    /// Receive time of the last accepted heartbeat of this incarnation.
    pub last_heartbeat_ms: Option<u64>,
    /// The readiness smoke of this round.
    pub smoke: SmokeStatus,
    /// Team pools (net-teleport design note 4.4, C19): the agent's last accepted heartbeat said it serves only
    /// its owner (its local switch or an `owner_only` window).
    pub owner_only: bool,
    /// The last described model and the operator's residency intent (m36 design note Q8, Q11); kept across
    /// `join()`, Lost and re-register.
    pub model: ModelView,
    /// The worker epoch the last recorded inventory was described at (the driver acts only on a current one).
    pub inventory_epoch: Option<u64>,
}

/// The operator's residency intent of a remote worker (m36 design note Q8, Q11): `none` until the first
/// `load`/`unload`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Desired {
    None,
    Resident,
    Unloaded,
}

impl From<Residency> for Desired {
    fn from(residency: Residency) -> Desired {
        match residency {
            Residency::Resident => Desired::Resident,
            Residency::Unloaded => Desired::Unloaded,
        }
    }
}

/// A remote worker's model as the coordinator knows it (m36 design note Q9, Q11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelView {
    /// `None` before the first describe since this start; `Some(false)` when last described in attached mode.
    pub managed: Option<bool>,
    pub desired: Desired,
    /// The journaled intent's sequence (0 = no intent yet).
    pub intent_seq: u64,
    /// The last described model state.
    pub state: Option<ModelState>,
    /// The last described model's lock (`None` in attached mode).
    pub lock_sha256: Option<String>,
}

/// Why [`RemoteRegistry::set_residency`] refused (m36 design note Q9).
#[derive(Debug)]
pub enum ResidencyError {
    /// Not a configured remote worker.
    UnknownWorker,
    /// Last described in attached mode: the worker's backend is not the agent's to load or unload.
    Unmanaged,
    /// Never described since this start.
    NotDescribed,
    Journal(LedgerError),
}

/// Why a driver transition was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteError {
    /// Not a configured remote worker.
    UnknownWorker,
    /// No accepted Register in this term.
    NotRegistered,
    /// The transition was computed for another round.
    StaleRound { current: u64 },
    /// The worker is not Joining (Ready, or Lost).
    NotJoining,
    /// No reconciliation of the agent's current generation in this round.
    NotReconciled,
    /// The agent's last heartbeat did not report `open` (or, while draining, `fenced`).
    NotOpen,
    /// The coordinator-side lease view has expired.
    LeaseExpired,
    /// The round's smoke has not passed (required unless draining).
    SmokeNotPassed,
    /// A managed worker's model is not `ready` under the coordinator's deployment lock, or the operator asked
    /// for it unloaded (m36 design note Q10).
    ModelNotReady,
}

#[derive(Debug)]
struct Remote {
    worker_id: String,
    epoch: Option<u64>,
    /// Recorded health: Joining, Ready or Lost (Suspect is only ever aged).
    health: WorkerHealth,
    /// The aging base: the last accepted heartbeat, or the Register that started the round's incarnation.
    seen_ms: u64,
    round: u64,
    heartbeat_sequence: Option<u64>,
    last_heartbeat_ms: Option<u64>,
    generation: Option<u64>,
    admission_state: Option<AdmissionState>,
    reported_sha256: Option<[u8; 32]>,
    inventory: Option<(Inventory, [u8; 32])>,
    lease_sequence: u64,
    lease_until_ms: Option<u64>,
    reconciled_generation: Option<u64>,
    smoke: SmokeStatus,
    owner_only: bool,
    draining: bool,
    /// Kept outside what `join()` and `start()` reset (review M36-13).
    model: ModelView,
    inventory_epoch: Option<u64>,
    /// The lock (hex SHA-256) of the worker's deployment (M5 D8) its managed model must be ready under (Q10).
    deployment_lock: String,
}

impl Remote {
    fn new(
        worker_id: &str,
        draining: bool,
        residency: Option<(Residency, u64)>,
        deployment_lock: &str,
    ) -> Remote {
        let (desired, intent_seq) =
            residency.map_or((Desired::None, 0), |(desired, seq)| (desired.into(), seq));
        Remote {
            worker_id: worker_id.to_owned(),
            epoch: None,
            health: WorkerHealth::Joining,
            seen_ms: 0,
            round: 0,
            heartbeat_sequence: None,
            last_heartbeat_ms: None,
            generation: None,
            admission_state: None,
            reported_sha256: None,
            inventory: None,
            lease_sequence: 0,
            lease_until_ms: None,
            reconciled_generation: None,
            smoke: SmokeStatus::Pending,
            owner_only: false,
            draining,
            model: ModelView {
                managed: None,
                desired,
                intent_seq,
                state: None,
                lock_sha256: None,
            },
            inventory_epoch: None,
            deployment_lock: deployment_lock.to_owned(),
        }
    }

    /// Drain intent makes the worker ineligible, and so does `desired == unloaded` while its model was last
    /// described managed (m36 design note Q10 applies to managed workers only: a journaled unload does not fence
    /// the same worker id described in attached mode later).
    fn view(&self, epoch: u64) -> WorkerView {
        WorkerView {
            health: self.health,
            last_seen_ms: self.seen_ms,
            epoch,
            draining: self.draining
                || (self.model.managed == Some(true) && self.model.desired == Desired::Unloaded),
            lease_until_ms: self.lease_until_ms,
        }
    }

    /// Q7a/Q13: the admission state a healthy agent reports. The agent fences while drain intent is set, so a
    /// draining worker reports `fenced` (or still `open`, before the drain reply reaches it).
    fn healthy_admission(&self, admission: Option<AdmissionState>) -> bool {
        match admission {
            Some(AdmissionState::Open) => true,
            Some(AdmissionState::Fenced) => self.draining,
            _ => false,
        }
    }

    fn state_at(&self, now_ms: u64, timing: RemoteTiming) -> WorkerState {
        let Some(epoch) = self.epoch else {
            return WorkerState::Joining;
        };
        match self.health {
            WorkerHealth::Joining
                if now_ms >= self.seen_ms.saturating_add(timing.lost_after_ms) =>
            {
                WorkerState::Lost
            }
            WorkerHealth::Joining => WorkerState::Joining,
            _ => self.view(epoch).state_at(now_ms, timing.aging()),
        }
    }

    /// Records Lost when it is observed at `now_ms`; true if the worker is Lost.
    fn observe_lost(&mut self, now_ms: u64, timing: RemoteTiming) -> bool {
        if self.state_at(now_ms, timing) == WorkerState::Lost {
            self.health = WorkerHealth::Lost;
        }
        self.health == WorkerHealth::Lost
    }

    /// A new Joining round: reconciliation and the smoke start over.
    fn join(&mut self) {
        self.health = WorkerHealth::Joining;
        self.round += 1;
        self.reconciled_generation = None;
        self.inventory = None;
        self.smoke = SmokeStatus::Pending;
    }

    /// Starts `epoch` in this term (D5: the remote state only); the lease sequence and drain intent stay.
    fn start(&mut self, epoch: u64, reported_sha256: Option<[u8; 32]>, now_ms: u64) {
        self.epoch = Some(epoch);
        self.seen_ms = now_ms;
        self.heartbeat_sequence = None;
        self.last_heartbeat_ms = None;
        self.generation = None;
        self.admission_state = None;
        self.owner_only = false;
        self.reported_sha256 = reported_sha256;
        self.lease_until_ms = None;
        self.join();
    }

    /// A driver transition's precondition: registered, the named round, Joining at `now_ms`.
    fn joining_round(
        &self,
        round: u64,
        now_ms: u64,
        timing: RemoteTiming,
    ) -> Result<(), RemoteError> {
        if self.epoch.is_none() {
            return Err(RemoteError::NotRegistered);
        }
        if round != self.round {
            return Err(RemoteError::StaleRound {
                current: self.round,
            });
        }
        if self.state_at(now_ms, timing) != WorkerState::Joining {
            return Err(RemoteError::NotJoining);
        }
        Ok(())
    }

    fn snapshot(&self, term: u64, now_ms: u64, timing: RemoteTiming) -> RemoteState {
        RemoteState {
            worker_id: self.worker_id.clone(),
            state: self.state_at(now_ms, timing),
            draining: self.draining,
            round: self.round,
            term: self.epoch.map(|_| term),
            epoch: self.epoch,
            generation: self.generation,
            admission_state: self.admission_state,
            reported_inventory_sha256: self.reported_sha256,
            inventory: self
                .inventory
                .as_ref()
                .map(|(inventory, _)| inventory.clone()),
            inventory_sha256: self.inventory.as_ref().map(|&(_, sha256)| sha256),
            lease_sequence: self.lease_sequence,
            lease_until_ms: self.lease_until_ms,
            reconciled_generation: self.reconciled_generation,
            last_heartbeat_ms: self.last_heartbeat_ms,
            smoke: self.smoke.clone(),
            owner_only: self.owner_only,
            model: self.model.clone(),
            inventory_epoch: self.inventory_epoch,
        }
    }
}

/// Every configured remote worker of one coordinator term.
#[derive(Debug)]
pub struct RemoteRegistry {
    term: u64,
    cluster_id: String,
    timing: RemoteTiming,
    quota: PrincipalQuota,
    /// In `[[federation.workers]]` order.
    workers: Mutex<Vec<Remote>>,
    /// Counts round bumps (a worker entering Joining) and the heartbeats that change a Joining worker's checks
    /// (O182), so the readiness driver wakes at once.
    rounds: watch::Sender<u64>,
}

fn invalid(field: &str) -> Status {
    Status::invalid_argument(format!("malformed {field}"))
}

/// An optional SHA-256 field: empty = not reported, else exactly 32 bytes.
fn sha256_field(bytes: &[u8]) -> Result<Option<[u8; 32]>, Status> {
    if bytes.is_empty() {
        return Ok(None);
    }
    <[u8; 32]>::try_from(bytes)
        .map(Some)
        .map_err(|_| invalid("inventory_sha256"))
}

impl RemoteRegistry {
    /// The registry of term `term` with the drain and residency intents restored from the ledger, gating each
    /// managed worker on its deployment's lock (`deployment_locks`: worker id -> hex `lock_sha256`, m36 design note
    /// Q10, M5 D8). A worker without an entry is gated on no lock (never Ready while managed).
    pub fn new(
        fed: &FederationConfig,
        term: u64,
        drain_intents: &BTreeSet<String>,
        residency: &BTreeMap<String, (Residency, u64)>,
        deployment_locks: &BTreeMap<String, String>,
    ) -> RemoteRegistry {
        RemoteRegistry {
            term,
            cluster_id: fed.cluster_id.clone(),
            timing: RemoteTiming::from_config(fed),
            quota: PrincipalQuota::new(fed.rpc_quota()),
            workers: Mutex::new(
                fed.workers
                    .iter()
                    .map(|w| {
                        Remote::new(
                            &w.worker_id,
                            drain_intents.contains(&w.worker_id),
                            residency.get(&w.worker_id).copied(),
                            deployment_locks
                                .get(&w.worker_id)
                                .map_or("", String::as_str),
                        )
                    })
                    .collect(),
            ),
            rounds: watch::Sender::new(0),
        }
    }

    /// The coordinator term (the ledger's process epoch, Q6).
    pub fn term(&self) -> u64 {
        self.term
    }

    pub fn timing(&self) -> RemoteTiming {
        self.timing
    }

    /// The per-principal unary quota of the registry service (Q14).
    pub fn quota(&self) -> &PrincipalQuota {
        &self.quota
    }

    /// Changes on every round bump of any worker (an accepted Register that starts an incarnation or the term, a
    /// Q7a event, [`RemoteRegistry::enter_joining`]) and on every accepted heartbeat that changes what a Joining
    /// worker's checks read (O182: its generation, admission state or inventory hash, or a lease it had let expire).
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.rounds.subscribe()
    }

    fn bump_round(&self) {
        self.rounds.send_modify(|n| *n += 1);
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Remote>> {
        // Every mutation completes before any call that could panic; the state stays consistent.
        self.workers.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn with<T>(&self, worker_id: &str, f: impl FnOnce(&mut Remote) -> T) -> Option<T> {
        self.lock()
            .iter_mut()
            .find(|w| w.worker_id == worker_id)
            .map(f)
    }

    pub fn contains(&self, worker_id: &str) -> bool {
        self.with(worker_id, |_| ()).is_some()
    }

    /// Worker `worker_id`'s state aged at `now_ms`; `None` for an id that is not a configured remote worker.
    pub fn get(&self, worker_id: &str, now_ms: u64) -> Option<RemoteState> {
        self.with(worker_id, |w| w.snapshot(self.term, now_ms, self.timing))
    }

    /// Worker `worker_id`'s state aged at `now_ms` (that of [`RemoteRegistry::get`], without the snapshot).
    pub fn state_at(&self, worker_id: &str, now_ms: u64) -> Option<WorkerState> {
        self.with(worker_id, |w| w.state_at(now_ms, self.timing))
    }

    /// Every remote worker, in configuration order, aged at `now_ms`.
    pub fn list(&self, now_ms: u64) -> Vec<RemoteState> {
        self.lock()
            .iter()
            .map(|w| w.snapshot(self.term, now_ms, self.timing))
            .collect()
    }

    /// The worker as admission judges it (health, aging base, epoch, drain intent and lease view), to be aged with
    /// [`RemoteTiming::aging`]; `None` before an accepted Register in this term.
    pub fn worker_view(&self, worker_id: &str) -> Option<WorkerView> {
        self.with(worker_id, |w| w.epoch.map(|epoch| w.view(epoch)))
            .flatten()
    }

    /// Q13: journals drain intent (or its clearing) through [`Ledger::set_drain`], then stores it for the worker
    /// id in any state, under one lock so the journal and memory agree. `Ok(false)` (nothing journaled) for an
    /// id that is not a configured remote worker; on a journal error the intent is unchanged.
    pub fn set_drain_intent(
        &self,
        ledger: &Mutex<Ledger>,
        worker_id: &str,
        draining: bool,
        now_ms: u64,
    ) -> Result<bool, LedgerError> {
        let mut workers = self.lock();
        let Some(w) = workers.iter_mut().find(|w| w.worker_id == worker_id) else {
            return Ok(false);
        };
        // Lock order: remote workers, then the ledger.
        ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .set_drain(worker_id, draining, now_ms)?;
        w.draining = draining;
        // Undrain re-runs Q10 (the smoke never ran while the agent was fenced by the drain).
        if !draining && w.epoch.is_some() && w.health == WorkerHealth::Ready {
            w.join();
            self.bump_round();
        }
        Ok(true)
    }

    /// m36 design note Q8-Q10: journals a new residency intent through [`Ledger::set_residency`] (fsync), then
    /// stores it, under one lock so the journal and memory agree; works in any state, Lost included (M36-13).
    /// Refused for an unknown worker, one never described since this start, or one last described in attached
    /// mode. `unloaded` takes a Ready worker back to Joining (and its admission view is ineligible at once);
    /// every intent bumps the round so the readiness driver acts on it. On a journal error nothing changes.
    pub fn set_residency(
        &self,
        ledger: &Mutex<Ledger>,
        worker_id: &str,
        desired: Residency,
        now_ms: u64,
    ) -> Result<ModelView, ResidencyError> {
        let mut workers = self.lock();
        let Some(w) = workers.iter_mut().find(|w| w.worker_id == worker_id) else {
            return Err(ResidencyError::UnknownWorker);
        };
        match w.model.managed {
            None => return Err(ResidencyError::NotDescribed),
            Some(false) => return Err(ResidencyError::Unmanaged),
            Some(true) => {}
        }
        // Lock order: remote workers, then the ledger.
        let intent_seq = ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .set_residency(worker_id, desired, now_ms)
            .map_err(ResidencyError::Journal)?;
        w.model.desired = desired.into();
        w.model.intent_seq = intent_seq;
        if desired == Residency::Unloaded && w.epoch.is_some() && w.health == WorkerHealth::Ready {
            w.join();
        }
        self.bump_round();
        eprintln!(
            "cered remote worker={worker_id} event=residency desired={} intent_seq={intent_seq} round={}",
            match desired {
                Residency::Resident => "resident",
                Residency::Unloaded => "unloaded",
            },
            w.round
        );
        Ok(w.model.clone())
    }

    /// Driver: takes the worker back to Joining from round `round` (a failed re-check, leaving Ready); returns the
    /// new round. A Lost worker stays Lost.
    pub fn enter_joining(&self, worker_id: &str, round: u64) -> Result<u64, RemoteError> {
        let now = unix_ms();
        self.with(worker_id, |w| {
            if w.epoch.is_none() {
                return Err(RemoteError::NotRegistered);
            }
            if round != w.round {
                return Err(RemoteError::StaleRound { current: w.round });
            }
            if w.observe_lost(now, self.timing) {
                return Err(RemoteError::NotJoining);
            }
            w.join();
            self.bump_round();
            Ok(w.round)
        })
        .unwrap_or(Err(RemoteError::UnknownWorker))
    }

    /// Driver: reconciliation of round `round` completed for `generation` (Q9); every following heartbeat reply
    /// carries it as `reconciled_generation` until the worker re-enters Joining.
    pub fn mark_reconciled(
        &self,
        worker_id: &str,
        round: u64,
        generation: u64,
    ) -> Result<(), RemoteError> {
        let now = unix_ms();
        self.with(worker_id, |w| {
            w.joining_round(round, now, self.timing)?;
            w.reconciled_generation = Some(generation);
            Ok(())
        })
        .unwrap_or(Err(RemoteError::UnknownWorker))
    }

    /// Driver: the inventory read in round `round` and the SHA-256 of its exact `DescribeResources` bytes. Its
    /// model (`managed` unless attached, state, lock) and the epoch it was described at are kept until the next
    /// describe (m36 design note Q8, Q11).
    pub fn record_inventory(
        &self,
        worker_id: &str,
        round: u64,
        inventory: Inventory,
        sha256: [u8; 32],
    ) -> Result<(), RemoteError> {
        let now = unix_ms();
        self.with(worker_id, |w| {
            w.joining_round(round, now, self.timing)?;
            if let Some(model) = inventory.model() {
                w.model.managed = Some(model.state != ModelState::Attached);
                w.model.state = Some(model.state);
                w.model.lock_sha256 = model.lock_sha256.clone();
            }
            w.inventory_epoch = w.epoch;
            w.inventory = Some((inventory, sha256));
            Ok(())
        })
        .unwrap_or(Err(RemoteError::UnknownWorker))
    }

    /// Driver: the outcome of round `round`'s readiness smoke.
    pub fn set_smoke(
        &self,
        worker_id: &str,
        round: u64,
        smoke: SmokeStatus,
    ) -> Result<(), RemoteError> {
        let now = unix_ms();
        self.with(worker_id, |w| {
            w.joining_round(round, now, self.timing)?;
            w.smoke = smoke;
            Ok(())
        })
        .unwrap_or(Err(RemoteError::UnknownWorker))
    }

    /// Driver: Joining -> Ready for round `round` at `now_ms` (Q7a, Q10). Requires reconciliation of the agent's
    /// current generation, an `open` report, a live lease view and a passed smoke, and for a managed worker its
    /// model `ready` under the deployment lock with `desired != unloaded` (m36 design note Q10); the inventory
    /// pins are the driver's to check before.
    pub fn mark_ready(&self, worker_id: &str, round: u64, now_ms: u64) -> Result<(), RemoteError> {
        self.with(worker_id, |w| {
            w.joining_round(round, now_ms, self.timing)?;
            if w.reconciled_generation.is_none() || w.reconciled_generation != w.generation {
                return Err(RemoteError::NotReconciled);
            }
            if !w.healthy_admission(w.admission_state) {
                return Err(RemoteError::NotOpen);
            }
            if w.lease_until_ms.is_none_or(|until| now_ms >= until) {
                return Err(RemoteError::LeaseExpired);
            }
            if !w.draining && !matches!(w.smoke, SmokeStatus::Passed { .. }) {
                return Err(RemoteError::SmokeNotPassed);
            }
            if w.model.managed == Some(true)
                && (w.model.state != Some(ModelState::Ready)
                    || w.model.lock_sha256.as_deref() != Some(w.deployment_lock.as_str())
                    || w.model.desired == Desired::Unloaded)
            {
                return Err(RemoteError::ModelNotReady);
            }
            w.health = WorkerHealth::Ready;
            Ok(())
        })
        .unwrap_or(Err(RemoteError::UnknownWorker))
    }

    fn register_refusal(&self, reason: &str, journaled_worker_epoch: u64) -> pb::RegisterReply {
        pb::RegisterReply {
            accepted: false,
            coordinator_term: self.term,
            reason_code: reason.to_owned(),
            journaled_worker_epoch,
            ..Default::default()
        }
    }

    /// Q4 Register rules, after authorization. An incarnation above the journaled epoch is journaled through
    /// `ledger` before the reply; refusals change nothing.
    pub fn register(
        &self,
        ledger: &Mutex<Ledger>,
        req: &pb::RegisterRequest,
        now_ms: u64,
    ) -> Result<pb::RegisterReply, Status> {
        let proposed = req.proposed.as_ref().ok_or_else(|| invalid("proposed"))?;
        let reported = sha256_field(&req.inventory_sha256)?;
        let id = proposed.worker_id.as_str();
        let epoch = proposed.worker_epoch;
        if req.protocol_revision != cere_proto::PROTOCOL_REVISION {
            return Ok(self.register_refusal(wire::UNSUPPORTED_REVISION, 0));
        }
        if proposed.cluster_id != self.cluster_id {
            return Ok(self.register_refusal(wire::WRONG_CLUSTER, 0));
        }
        if epoch == 0 {
            return Err(invalid("worker_epoch"));
        }
        let mut workers = self.lock();
        let Some(w) = workers.iter_mut().find(|w| w.worker_id == id) else {
            return Ok(self.register_refusal(wire::UNKNOWN_WORKER, 0));
        };
        // Lock order: remote workers, then the ledger.
        let mut ledger = ledger.lock().unwrap_or_else(PoisonError::into_inner);
        let outcome = match ledger.worker_epoch(id) {
            Some(journaled) if epoch < journaled => {
                return Ok(self.register_refusal(wire::STALE_EPOCH, journaled));
            }
            Some(journaled) if epoch == journaled && w.epoch == Some(epoch) => {
                if w.observe_lost(now_ms, self.timing) {
                    eprintln!(
                        "cered remote worker={id}@{epoch} event=register refused={}",
                        wire::EPOCH_LOST
                    );
                    return Ok(self.register_refusal(wire::EPOCH_LOST, 0));
                }
                "reconnect"
            }
            Some(journaled) if epoch == journaled => {
                w.start(epoch, reported, now_ms);
                self.bump_round();
                "term"
            }
            _ => {
                ledger
                    .accept_worker_epoch(id, epoch, now_ms)
                    .map_err(|e| wire::status(Code::Unavailable, e.code()))?;
                w.start(epoch, reported, now_ms);
                self.bump_round();
                "incarnation"
            }
        };
        drop(ledger);
        eprintln!(
            "cered remote worker={id}@{epoch} event=register outcome={outcome} term={} round={}",
            self.term, w.round
        );
        Ok(pb::RegisterReply {
            accepted: true,
            coordinator_term: self.term,
            admission_lease_ms: self.timing.lease_ms,
            requires_reconciliation: true,
            reason_code: String::new(),
            journaled_worker_epoch: epoch,
        })
    }

    fn heartbeat_refusal(&self, reason: &str) -> pb::HeartbeatReply {
        pb::HeartbeatReply {
            accepted: false,
            coordinator_term: self.term,
            reason_code: reason.to_owned(),
            ..Default::default()
        }
    }

    /// Q7/Q7a heartbeat rules, after authorization: an accepted heartbeat is a sighting at `now_ms` (receive
    /// time) and grants the next lease of this term.
    pub fn heartbeat(
        &self,
        req: &pb::HeartbeatRequest,
        now_ms: u64,
    ) -> Result<pb::HeartbeatReply, Status> {
        let worker = req.worker.as_ref().ok_or_else(|| invalid("worker"))?;
        let admission = AdmissionState::parse(&req.admission_state)
            .ok_or_else(|| invalid("admission_state"))?;
        let reported = sha256_field(&req.inventory_sha256)?;
        if worker.cluster_id != self.cluster_id {
            return Ok(self.heartbeat_refusal(wire::WRONG_CLUSTER));
        }
        let mut workers = self.lock();
        let Some(w) = workers.iter_mut().find(|w| w.worker_id == worker.worker_id) else {
            return Ok(self.heartbeat_refusal(wire::UNKNOWN_WORKER));
        };
        let Some(epoch) = w.epoch else {
            return Ok(self.heartbeat_refusal(wire::UNKNOWN_WORKER));
        };
        if req.observed_coordinator_term != self.term {
            return Ok(self.heartbeat_refusal(wire::STALE_TERM));
        }
        if worker.worker_epoch < epoch {
            return Ok(self.heartbeat_refusal(wire::STALE_EPOCH));
        }
        if worker.worker_epoch > epoch {
            return Ok(self.heartbeat_refusal(wire::UNKNOWN_WORKER));
        }
        if w.observe_lost(now_ms, self.timing) {
            return Ok(self.heartbeat_refusal(wire::EPOCH_LOST));
        }
        if w.heartbeat_sequence
            .is_some_and(|last| req.sequence <= last)
        {
            return Ok(self.heartbeat_refusal(wire::REPLAYED_HEARTBEAT));
        }
        // Q7a: a Ready worker returns to Joining on a new generation or inventory, a non-open report, or after its
        // lease view expired (every return to Ready passes Q10 again). A Joining worker stays in its round:
        // `mark_ready` requires the reconciled generation to equal the reported one.
        let changed = w.generation.is_some_and(|g| g != req.model_generation)
            || reported.is_some_and(|sha| w.reported_sha256.is_some_and(|known| known != sha));
        let live = matches!(
            w.state_at(now_ms, self.timing),
            WorkerState::Ready | WorkerState::Draining
        );
        if w.health == WorkerHealth::Ready
            && (changed || !live || !w.healthy_admission(Some(admission)))
        {
            w.join();
            self.bump_round();
            eprintln!(
                "cered remote worker={}@{epoch} event=joining round={} generation={} admission_state={}",
                w.worker_id,
                w.round,
                req.model_generation,
                admission.as_str()
            );
        }
        // O182: the driver of a Joining worker waits on these (the first generation, the agent opening after the
        // reconciled generation reached it, a new inventory, a renewed lease); without a wake it would wait a period.
        let wake = w.health == WorkerHealth::Joining
            && (w.generation != Some(req.model_generation)
                || w.admission_state != Some(admission)
                || reported.is_some_and(|sha| w.reported_sha256 != Some(sha))
                || w.lease_until_ms.is_none_or(|until| now_ms >= until));
        w.heartbeat_sequence = Some(req.sequence);
        w.last_heartbeat_ms = Some(now_ms);
        w.seen_ms = now_ms;
        w.generation = Some(req.model_generation);
        w.admission_state = Some(admission);
        w.owner_only = req.owner_only;
        if reported.is_some() {
            w.reported_sha256 = reported;
        }
        w.lease_sequence += 1;
        w.lease_until_ms = Some(now_ms.saturating_add(self.timing.lease_ms));
        if wake {
            self.bump_round();
        }
        Ok(pb::HeartbeatReply {
            accepted: true,
            coordinator_term: self.term,
            lease_sequence: w.lease_sequence,
            admission_lease_ms: self.timing.lease_ms,
            desired_state: if w.draining {
                wire::DESIRED_DRAINING.to_owned()
            } else {
                String::new()
            },
            reconciled_generation: w.reconciled_generation,
            reason_code: String::new(),
        })
    }
}

/// The tonic service: authorization (Q2) and the per-principal quota (Q14) before any state.
struct RegistryService {
    registry: Arc<RemoteRegistry>,
    /// Weak: a connection still open after shutdown never keeps the ledger (and its state-dir lock) alive.
    ledger: Weak<Mutex<Ledger>>,
    authz: Authz,
}

#[tonic::async_trait]
impl WorkerRegistryService for RegistryService {
    async fn register(
        &self,
        req: Request<pb::RegisterRequest>,
    ) -> Result<Response<pb::RegisterReply>, Status> {
        let claimed = req
            .get_ref()
            .proposed
            .as_ref()
            .map_or("", |p| p.worker_id.as_str());
        let principal = self.authz.registry_caller(&req, claimed)?;
        let _slot = self.registry.quota.try_acquire(&principal)?;
        let ledger = self
            .ledger
            .upgrade()
            .ok_or_else(|| Status::unavailable("coordinator stopped"))?;
        self.registry
            .register(&ledger, req.get_ref(), unix_ms())
            .map(Response::new)
    }

    async fn heartbeat(
        &self,
        req: Request<pb::HeartbeatRequest>,
    ) -> Result<Response<pb::HeartbeatReply>, Status> {
        let claimed = req
            .get_ref()
            .worker
            .as_ref()
            .map_or("", |w| w.worker_id.as_str());
        let principal = self.authz.registry_caller(&req, claimed)?;
        let _slot = self.registry.quota.try_acquire(&principal)?;
        self.registry
            .heartbeat(req.get_ref(), unix_ms())
            .map(Response::new)
    }
}

/// The registry service on `[federation] listen`, stopped by [`RegistryListener::stop`] or when dropped.
#[derive(Debug)]
pub struct RegistryListener {
    addr: SocketAddr,
    stop: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

impl RegistryListener {
    /// Binds `fed.listen` and serves `registry` over mTLS (`pki`, client certificate required) with the control
    /// decode limit (Q14). Must run inside a tokio runtime.
    pub fn spawn(
        registry: Arc<RemoteRegistry>,
        ledger: Weak<Mutex<Ledger>>,
        fed: &FederationConfig,
        pki: &Pki,
    ) -> Result<RegistryListener, String> {
        let bind_error =
            |e: std::io::Error| format!("cannot bind federation.listen {}: {e}", fed.listen);
        let listener = std::net::TcpListener::bind(fed.listen).map_err(bind_error)?;
        listener.set_nonblocking(true).map_err(bind_error)?;
        let listener = tokio::net::TcpListener::from_std(listener).map_err(bind_error)?;
        let addr = listener.local_addr().map_err(bind_error)?;
        let authz = Authz::coordinator_side(
            fed.workers
                .iter()
                .map(|w| (w.principal.clone(), w.worker_id.clone()))
                .collect(),
        );
        let service = WorkerRegistryServiceServer::new(RegistryService {
            registry,
            ledger,
            authz,
        })
        .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX);
        let router = Server::builder()
            .tls_config(pki.server_tls())
            .map_err(|e| format!("federation TLS: {e}"))?
            .add_service(service);
        let (stop, stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let incoming = TcpListenerStream::new(listener);
            let signal = async {
                let _ = stopped.await;
            };
            if let Err(e) = router.serve_with_incoming_shutdown(incoming, signal).await {
                eprintln!("cered federation: registry listener failed: {e}");
            }
        });
        Ok(RegistryListener { addr, stop, task })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stops accepting, lets open connections close (bounded), then ends the server.
    pub async fn stop(self) {
        let RegistryListener { stop, mut task, .. } = self;
        let _ = stop.send(());
        if tokio::time::timeout(LISTENER_STOP_GRACE, &mut task)
            .await
            .is_err()
        {
            task.abort();
            let _ = task.await;
        }
    }
}
