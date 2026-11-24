//! Local `Worker` record and read-only lifecycle view (m1 design note F4), mirroring the semantics of the CPU
//! oracle `scripts/worker_registry_reference.py` (D1/D2).
//!
//! A worker is identified by `worker_id` + `epoch`. `state()` is `health` while joining/suspect/lost, otherwise
//! the operator's `desired_state`. Only `ready` accepts new work. Health is aged on the observer's clock at
//! every eligibility decision by one rule, [`aged_health`], shared with admission (m3 design note Q7, Q10): a
//! worker not observed healthy for `suspect_after_ms` reads suspect, and for `suspect_after_ms + lost_after_ms`
//! without a healthy observation or a sighting (its backend identity confirmed, [`WorkerRegistry::sighted`])
//! reads lost. Each worker ages by the registry's rule unless it was given its own ([`WorkerRegistry::set_aging`],
//! M3.5 Q7: remote workers). Operator drain intent is stored per `worker_id` outside the epoch record (also for
//! an id not registered yet, [`WorkerRegistry::set_drain_intent`]), so it survives health transitions and
//! re-registration at a new epoch until `undrain`. A lost epoch (marked, or aged) never rejoins; work completions
//! from a stale epoch are rejected.
//!
//! Model lifecycle actions (load/unload/drain) are not offered in backend-managed M1: [`lifecycle_action`]
//! always answers `unsupported`.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    Joining,
    Ready,
    Suspect,
    Lost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesiredState {
    Ready,
    Draining,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Joining,
    Ready,
    Draining,
    Suspect,
    Lost,
}

impl WorkerState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Joining => "joining",
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Suspect => "suspect",
            Self::Lost => "lost",
        }
    }
}

/// Health-aging thresholds: one rule for the registry and admission (m3 design note Q7, Q10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aging {
    /// A ready worker not observed healthy for this long reads suspect.
    pub suspect_after_ms: u64,
    /// How long a worker stays suspect before it reads lost, counted from the moment it would have aged to
    /// suspect: lost once neither observed healthy nor sighted for `suspect_after_ms + lost_after_ms`.
    pub lost_after_ms: u64,
}

impl Aging {
    fn suspect_at(self, last_seen_ms: u64) -> u64 {
        last_seen_ms.saturating_add(self.suspect_after_ms)
    }

    fn lost_at(self, last_seen_ms: u64) -> u64 {
        self.suspect_at(last_seen_ms)
            .saturating_add(self.lost_after_ms)
    }
}

/// The aging rule: `health` as recorded at the last observation, aged at `now_ms` from `last_seen_ms` (the last
/// healthy observation or sighting). A ready or suspect worker reads lost from `last_seen_ms + suspect_after_ms +
/// lost_after_ms`, so a probe failure (explicit suspect, which never moves `last_seen_ms`) and a silent worker
/// (aged suspect) both stay suspect for at least `lost_after_ms` and turn lost at the same bound. A ready worker
/// reads suspect from `last_seen_ms + suspect_after_ms`. Joining and lost never age.
pub fn aged_health(
    health: WorkerHealth,
    last_seen_ms: u64,
    now_ms: u64,
    aging: Aging,
) -> WorkerHealth {
    match health {
        WorkerHealth::Ready | WorkerHealth::Suspect if now_ms >= aging.lost_at(last_seen_ms) => {
            WorkerHealth::Lost
        }
        WorkerHealth::Ready if now_ms >= aging.suspect_at(last_seen_ms) => WorkerHealth::Suspect,
        h => h,
    }
}

/// The first instant after `now_ms` at which [`aged_health`] changes without a new observation, if any.
pub fn next_aging_ms(
    health: WorkerHealth,
    last_seen_ms: u64,
    now_ms: u64,
    aging: Aging,
) -> Option<u64> {
    let deadlines: &[u64] = match health {
        WorkerHealth::Ready => &[aging.suspect_at(last_seen_ms), aging.lost_at(last_seen_ms)],
        WorkerHealth::Suspect => &[aging.lost_at(last_seen_ms)],
        WorkerHealth::Joining | WorkerHealth::Lost => &[],
    };
    deadlines
        .iter()
        .copied()
        .find(|&at| at > now_ms && at < u64::MAX)
}

/// What admission needs to judge a worker (m3 design note Q7): its recorded health, last healthy observation or
/// sighting, epoch, drain intent and (a remote worker, M3.5 Q7) its admission lease, pushed on every readiness
/// publish and aged at each grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerView {
    pub health: WorkerHealth,
    pub last_seen_ms: u64,
    pub epoch: u64,
    pub draining: bool,
    /// When set, the worker takes new work only while `now < lease_until_ms` (its agent refuses work after the
    /// lease it was granted runs out); at and after it the worker reads suspect.
    pub lease_until_ms: Option<u64>,
}

impl WorkerView {
    /// Lifecycle state after aging at `now_ms`; only [`WorkerState::Ready`] accepts new work.
    pub fn state_at(&self, now_ms: u64, aging: Aging) -> WorkerState {
        match aged_health(self.health, self.last_seen_ms, now_ms, aging) {
            WorkerHealth::Joining => WorkerState::Joining,
            WorkerHealth::Suspect => WorkerState::Suspect,
            WorkerHealth::Lost => WorkerState::Lost,
            WorkerHealth::Ready if self.lease_until_ms.is_some_and(|until| now_ms >= until) => {
                WorkerState::Suspect
            }
            WorkerHealth::Ready if self.draining => WorkerState::Draining,
            WorkerHealth::Ready => WorkerState::Ready,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worker {
    pub worker_id: String,
    pub epoch: u64,
    pub health: WorkerHealth,
    pub desired_state: DesiredState,
    pub last_seen_ms: u64,
    pub active: u64,
}

impl Worker {
    /// The record as admission sees it (no lease: the registry's workers are local).
    pub fn view(&self) -> WorkerView {
        WorkerView {
            health: self.health,
            last_seen_ms: self.last_seen_ms,
            epoch: self.epoch,
            draining: self.desired_state == DesiredState::Draining,
            lease_until_ms: None,
        }
    }

    /// Health after aging at `now_ms`.
    pub fn health_at(&self, now_ms: u64, aging: Aging) -> WorkerHealth {
        aged_health(self.health, self.last_seen_ms, now_ms, aging)
    }

    pub fn state_at(&self, now_ms: u64, aging: Aging) -> WorkerState {
        self.view().state_at(now_ms, aging)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerError {
    UnknownWorker,
    /// The epoch is older than (or, for work, different from) the registered one.
    StaleEpoch,
    /// A lost epoch cannot rejoin, heartbeat or complete work.
    LostEpoch,
    /// The worker is not `ready` (joining, draining, suspect or lost).
    NotEligible(WorkerState),
    /// A completion without matching active work.
    NoActiveWork,
}

#[derive(Debug, Clone)]
pub struct WorkerRegistry {
    aging: Aging,
    /// Workers that age by their own rule instead of the registry's (M3.5 Q7: remote workers).
    worker_aging: BTreeMap<String, Aging>,
    workers: BTreeMap<String, Worker>,
    drain_intent: BTreeSet<String>,
}

impl WorkerRegistry {
    pub fn new(aging: Aging) -> Self {
        Self {
            aging,
            worker_aging: BTreeMap::new(),
            workers: BTreeMap::new(),
            drain_intent: BTreeSet::new(),
        }
    }

    /// The registry's aging rule: every worker's unless [`Self::set_aging`] gave it its own.
    pub fn aging(&self) -> Aging {
        self.aging
    }

    /// M3.5 Q7: `worker_id` (registered or not) ages by `aging` from now on.
    pub fn set_aging(&mut self, worker_id: &str, aging: Aging) {
        self.worker_aging.insert(worker_id.to_owned(), aging);
    }

    /// The aging rule `worker_id` is judged by.
    pub fn aging_of(&self, worker_id: &str) -> Aging {
        self.worker_aging
            .get(worker_id)
            .copied()
            .unwrap_or(self.aging)
    }

    /// Registers `worker_id` at `epoch` (joining). The same epoch returns the existing record (a reconnect is
    /// neither renewal nor readiness); a higher epoch replaces it and inherits the drain intent.
    pub fn register(
        &mut self,
        worker_id: &str,
        epoch: u64,
        now_ms: u64,
    ) -> Result<&Worker, WorkerError> {
        if let Some(old) = self.workers.get(worker_id) {
            if epoch < old.epoch {
                return Err(WorkerError::StaleEpoch);
            }
            if epoch == old.epoch {
                if old.health_at(now_ms, self.aging_of(worker_id)) == WorkerHealth::Lost {
                    return Err(WorkerError::LostEpoch);
                }
                return Ok(&self.workers[worker_id]);
            }
        }
        let desired_state = if self.drain_intent.contains(worker_id) {
            DesiredState::Draining
        } else {
            DesiredState::Ready
        };
        self.workers.insert(
            worker_id.to_owned(),
            Worker {
                worker_id: worker_id.to_owned(),
                epoch,
                health: WorkerHealth::Joining,
                desired_state,
                last_seen_ms: now_ms,
                active: 0,
            },
        );
        Ok(&self.workers[worker_id])
    }

    fn same_epoch(&mut self, worker_id: &str, epoch: u64) -> Result<&mut Worker, WorkerError> {
        let w = self
            .workers
            .get_mut(worker_id)
            .ok_or(WorkerError::UnknownWorker)?;
        if w.epoch != epoch {
            return Err(WorkerError::StaleEpoch);
        }
        Ok(w)
    }

    /// A readiness observation (for the local backend: the cached readiness probe incl. pin match and smoke).
    /// Healthy → `ready` and fresh; unhealthy → `suspect` (no new work; recovers on the next healthy one). An
    /// epoch already lost at `now_ms` (marked or aged) is recorded lost and rejected: it needs a new epoch.
    pub fn observe(
        &mut self,
        worker_id: &str,
        epoch: u64,
        healthy: bool,
        now_ms: u64,
    ) -> Result<(), WorkerError> {
        let aging = self.aging_of(worker_id);
        let w = self.same_epoch(worker_id, epoch)?;
        if w.health_at(now_ms, aging) == WorkerHealth::Lost {
            w.health = WorkerHealth::Lost;
            return Err(WorkerError::LostEpoch);
        }
        if healthy {
            w.health = WorkerHealth::Ready;
            w.last_seen_ms = now_ms;
        } else {
            w.health = WorkerHealth::Suspect;
        }
        Ok(())
    }

    /// A sighting (m3 design note Q4 rule 3, review A1): the probe confirmed the backend identity behind this
    /// epoch (every pin matched, the same `instance_id`) but readiness is still pending (the re-smoke waits for a
    /// slot of this worker, or failed). Suspect (no new work) and seen: `last_seen_ms` moves, so the worker does
    /// not age lost while the same backend answers. An epoch already lost at `now_ms` is recorded lost and
    /// rejected, as in [`Self::observe`].
    pub fn sighted(&mut self, worker_id: &str, epoch: u64, now_ms: u64) -> Result<(), WorkerError> {
        let aging = self.aging_of(worker_id);
        let w = self.same_epoch(worker_id, epoch)?;
        if w.health_at(now_ms, aging) == WorkerHealth::Lost {
            w.health = WorkerHealth::Lost;
            return Err(WorkerError::LostEpoch);
        }
        w.health = WorkerHealth::Suspect;
        w.last_seen_ms = now_ms;
        Ok(())
    }

    /// Marks the epoch lost; it never rejoins.
    pub fn mark_lost(&mut self, worker_id: &str, epoch: u64) -> Result<(), WorkerError> {
        self.same_epoch(worker_id, epoch)?.health = WorkerHealth::Lost;
        Ok(())
    }

    /// Operator drain: no new work; the intent outlives epochs until [`Self::undrain`].
    pub fn drain(&mut self, worker_id: &str, epoch: u64) -> Result<(), WorkerError> {
        let w = self.same_epoch(worker_id, epoch)?;
        if w.health == WorkerHealth::Lost {
            return Err(WorkerError::LostEpoch);
        }
        w.desired_state = DesiredState::Draining;
        self.drain_intent.insert(worker_id.to_owned());
        Ok(())
    }

    /// Clears drain intent. Readiness still requires a healthy observation.
    pub fn undrain(&mut self, worker_id: &str) {
        self.drain_intent.remove(worker_id);
        if let Some(w) = self.workers.get_mut(worker_id) {
            w.desired_state = DesiredState::Ready;
        }
    }

    /// M3.5 Q13: stores operator drain intent for `worker_id` (any id, registered or not, in any health) without
    /// an epoch: a registered worker drains now, a later registration (at any epoch) starts draining. Clearing it
    /// is [`Self::undrain`].
    pub fn set_drain_intent(&mut self, worker_id: &str, draining: bool) {
        if !draining {
            self.undrain(worker_id);
            return;
        }
        self.drain_intent.insert(worker_id.to_owned());
        if let Some(w) = self.workers.get_mut(worker_id) {
            w.desired_state = DesiredState::Draining;
        }
    }

    /// Whether the operator's drain intent stands for `worker_id`.
    pub fn drain_intended(&self, worker_id: &str) -> bool {
        self.drain_intent.contains(worker_id)
    }

    pub fn get(&self, worker_id: &str) -> Option<&Worker> {
        self.workers.get(worker_id)
    }

    pub fn state(&self, worker_id: &str, now_ms: u64) -> Option<WorkerState> {
        self.get(worker_id)
            .map(|w| w.state_at(now_ms, self.aging_of(worker_id)))
    }

    /// Workers that accept new work at `now_ms` (aged state `ready`), sorted by id.
    pub fn eligible(&self, now_ms: u64) -> Vec<&Worker> {
        self.workers
            .values()
            .filter(|w| w.state_at(now_ms, self.aging_of(&w.worker_id)) == WorkerState::Ready)
            .collect()
    }

    /// Starts one request on an eligible worker of the given epoch.
    pub fn begin_work(
        &mut self,
        worker_id: &str,
        epoch: u64,
        now_ms: u64,
    ) -> Result<(), WorkerError> {
        let aging = self.aging_of(worker_id);
        let w = self.same_epoch(worker_id, epoch)?;
        let state = w.state_at(now_ms, aging);
        if state != WorkerState::Ready {
            return Err(WorkerError::NotEligible(state));
        }
        w.active += 1;
        Ok(())
    }

    /// Ends one request. A stale-epoch completion is rejected (quarantined), never applied to the new epoch.
    pub fn finish_work(&mut self, worker_id: &str, epoch: u64) -> Result<(), WorkerError> {
        let w = self.same_epoch(worker_id, epoch)?;
        if w.active == 0 {
            return Err(WorkerError::NoActiveWork);
        }
        w.active -= 1;
        Ok(())
    }

    /// Read-only lifecycle view of one worker.
    pub fn lifecycle_view(&self, worker_id: &str, now_ms: u64) -> Option<LifecycleView> {
        let w = self.get(worker_id)?;
        let worker_state = w.state_at(now_ms, self.aging_of(worker_id));
        Some(LifecycleView {
            worker_id: w.worker_id.clone(),
            epoch: w.epoch,
            state: if worker_state == WorkerState::Ready {
                LifecycleState::Ready
            } else {
                LifecycleState::Unavailable
            },
            worker_state,
            active_requests: w.active,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    Ready,
    Unavailable,
}

impl LifecycleState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleView {
    pub worker_id: String,
    pub epoch: u64,
    pub state: LifecycleState,
    pub worker_state: WorkerState,
    pub active_requests: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleOp {
    Load,
    Unload,
    Drain,
}

impl LifecycleOp {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "load" => Some(Self::Load),
            "unload" => Some(Self::Unload),
            "drain" => Some(Self::Drain),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Unload => "unload",
            Self::Drain => "drain",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsupported(pub LifecycleOp);

impl Unsupported {
    pub fn code(self) -> &'static str {
        "unsupported"
    }
}

/// Backend-managed M1 owns no model lifecycle: every action is rejected as `unsupported`.
pub fn lifecycle_action(op: LifecycleOp) -> Result<(), Unsupported> {
    Err(Unsupported(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUSPECT: u64 = 100;
    const LOST: u64 = 1000;
    const AGING: Aging = Aging {
        suspect_after_ms: SUSPECT,
        lost_after_ms: LOST,
    };

    fn ready(reg: &mut WorkerRegistry, id: &str, epoch: u64, now: u64) {
        reg.register(id, epoch, now).unwrap();
        reg.observe(id, epoch, true, now).unwrap();
    }

    #[test]
    fn joining_is_not_ready_until_observed_healthy() {
        let mut reg = WorkerRegistry::new(AGING);
        reg.register("w", 1, 0).unwrap();
        assert_eq!(reg.state("w", 0), Some(WorkerState::Joining));
        assert_eq!(
            reg.begin_work("w", 1, 0),
            Err(WorkerError::NotEligible(WorkerState::Joining))
        );
        reg.observe("w", 1, true, 1).unwrap();
        assert_eq!(reg.eligible(1).len(), 1);
    }

    #[test]
    fn drain_intent_survives_epoch_change_until_undrain() {
        let mut reg = WorkerRegistry::new(AGING);
        ready(&mut reg, "w", 1, 0);
        reg.drain("w", 1).unwrap();
        assert_eq!(reg.state("w", 1), Some(WorkerState::Draining));
        // Unhealthy then recovered at the same epoch: still draining.
        reg.observe("w", 1, false, 2).unwrap();
        reg.observe("w", 1, true, 3).unwrap();
        assert_eq!(reg.state("w", 3), Some(WorkerState::Draining));
        // Restart at a higher epoch (after loss or not): still draining once ready.
        reg.mark_lost("w", 1).unwrap();
        ready(&mut reg, "w", 2, 4);
        assert_eq!(reg.state("w", 4), Some(WorkerState::Draining));
        ready(&mut reg, "w", 3, 5);
        assert_eq!(reg.state("w", 5), Some(WorkerState::Draining));
        assert!(reg.eligible(5).is_empty());
        reg.undrain("w");
        assert_eq!(reg.state("w", 5), Some(WorkerState::Ready));
        // undrain alone does not make a joining worker ready.
        reg.register("w", 4, 6).unwrap();
        assert_eq!(reg.state("w", 6), Some(WorkerState::Joining));
    }

    #[test]
    fn health_is_aged_before_eligibility_and_work() {
        let mut reg = WorkerRegistry::new(AGING);
        ready(&mut reg, "w", 1, 0);
        assert_eq!(reg.eligible(SUSPECT - 1).len(), 1);
        assert!(reg.eligible(SUSPECT).is_empty());
        assert_eq!(
            reg.begin_work("w", 1, SUSPECT),
            Err(WorkerError::NotEligible(WorkerState::Suspect))
        );
        reg.observe("w", 1, true, SUSPECT).unwrap();
        reg.begin_work("w", 1, SUSPECT).unwrap();
    }

    /// m3 Q10: a probe failure (explicit suspect) turns lost no later than `suspect_after + lost_after` after
    /// the last healthy observation, and the lost epoch never comes back.
    #[test]
    fn probe_failure_suspect_turns_lost_by_suspect_plus_lost_after_last_healthy() {
        let mut reg = WorkerRegistry::new(AGING);
        ready(&mut reg, "w", 1, 0);
        reg.observe("w", 1, false, 10).unwrap();
        assert_eq!(reg.state("w", 10), Some(WorkerState::Suspect));
        assert_eq!(
            reg.state("w", SUSPECT + LOST - 1),
            Some(WorkerState::Suspect)
        );
        assert_eq!(reg.state("w", SUSPECT + LOST), Some(WorkerState::Lost));
        assert_eq!(
            reg.begin_work("w", 1, SUSPECT + LOST),
            Err(WorkerError::NotEligible(WorkerState::Lost))
        );
        assert_eq!(
            reg.observe("w", 1, true, SUSPECT + LOST),
            Err(WorkerError::LostEpoch)
        );
        assert_eq!(reg.get("w").unwrap().health, WorkerHealth::Lost);
        assert_eq!(
            reg.register("w", 1, SUSPECT + LOST).unwrap_err(),
            WorkerError::LostEpoch
        );
        ready(&mut reg, "w", 2, SUSPECT + LOST);
        assert_eq!(reg.state("w", SUSPECT + LOST), Some(WorkerState::Ready));
    }

    /// The shared rule: a silent ready worker reads suspect, then lost; `next_aging_ms` names each change.
    #[test]
    fn silent_worker_ages_to_suspect_then_lost_at_the_announced_deadlines() {
        use WorkerHealth::*;
        assert_eq!(aged_health(Ready, 0, SUSPECT - 1, AGING), Ready);
        assert_eq!(aged_health(Ready, 0, SUSPECT, AGING), Suspect);
        assert_eq!(aged_health(Ready, 0, SUSPECT + LOST - 1, AGING), Suspect);
        assert_eq!(aged_health(Ready, 0, SUSPECT + LOST, AGING), Lost);
        assert_eq!(aged_health(Joining, 0, u64::MAX - 1, AGING), Joining);
        assert_eq!(next_aging_ms(Ready, 0, 0, AGING), Some(SUSPECT));
        assert_eq!(
            next_aging_ms(Ready, 0, SUSPECT, AGING),
            Some(SUSPECT + LOST)
        );
        assert_eq!(next_aging_ms(Suspect, 0, 0, AGING), Some(SUSPECT + LOST));
        assert_eq!(next_aging_ms(Ready, 0, SUSPECT + LOST, AGING), None);
        assert_eq!(next_aging_ms(Joining, 0, 0, AGING), None);
        let never_lost = Aging {
            suspect_after_ms: SUSPECT,
            lost_after_ms: u64::MAX,
        };
        assert_eq!(aged_health(Suspect, 0, u64::MAX - 1, never_lost), Suspect);
        assert_eq!(next_aging_ms(Ready, 0, SUSPECT, never_lost), None);
        let view = WorkerView {
            health: Ready,
            last_seen_ms: 0,
            epoch: 1,
            draining: true,
            lease_until_ms: None,
        };
        assert_eq!(view.state_at(0, AGING), WorkerState::Draining);
        assert_eq!(view.state_at(SUSPECT, AGING), WorkerState::Suspect);
    }

    #[test]
    fn epochs_are_fenced() {
        let mut reg = WorkerRegistry::new(AGING);
        ready(&mut reg, "w", 2, 0);
        assert_eq!(
            reg.register("w", 1, 0).unwrap_err(),
            WorkerError::StaleEpoch
        );
        reg.begin_work("w", 2, 0).unwrap();
        reg.mark_lost("w", 2).unwrap();
        assert_eq!(reg.register("w", 2, 1).unwrap_err(), WorkerError::LostEpoch);
        assert_eq!(reg.observe("w", 2, true, 1), Err(WorkerError::LostEpoch));
        ready(&mut reg, "w", 3, 2);
        // The old epoch's completion cannot release the new epoch's work.
        assert_eq!(reg.finish_work("w", 2), Err(WorkerError::StaleEpoch));
        assert_eq!(reg.finish_work("w", 3), Err(WorkerError::NoActiveWork));
        assert_eq!(reg.finish_work("x", 3), Err(WorkerError::UnknownWorker));
    }

    #[test]
    fn lifecycle_view_and_actions() {
        let mut reg = WorkerRegistry::new(AGING);
        ready(&mut reg, "w", 7, 0);
        reg.begin_work("w", 7, 0).unwrap();
        let view = reg.lifecycle_view("w", 1).unwrap();
        assert_eq!(
            view,
            LifecycleView {
                worker_id: "w".into(),
                epoch: 7,
                state: LifecycleState::Ready,
                worker_state: WorkerState::Ready,
                active_requests: 1,
            }
        );
        reg.observe("w", 7, false, 2).unwrap();
        assert_eq!(
            reg.lifecycle_view("w", 2).unwrap().state,
            LifecycleState::Unavailable
        );
        for name in ["load", "unload", "drain"] {
            let op = LifecycleOp::parse(name).unwrap();
            assert_eq!(op.as_str(), name);
            assert_eq!(lifecycle_action(op).unwrap_err().code(), "unsupported");
        }
        assert_eq!(LifecycleOp::parse("pin"), None);
    }

    /// M3.5 Q13: drain intent set for an id not yet registered is applied at its registration; it can be set on
    /// a lost worker (where `drain` refuses), survives a higher-epoch registration, and only clearing it (or
    /// `undrain`) ends it; readiness still needs a healthy observation.
    #[test]
    fn drain_intent_for_any_id_in_any_health_until_cleared() {
        let mut reg = WorkerRegistry::new(AGING);
        reg.set_drain_intent("w", true);
        assert_eq!(reg.state("w", 0), None);
        reg.register("w", 1, 0).unwrap();
        assert_eq!(reg.get("w").unwrap().desired_state, DesiredState::Draining);
        reg.observe("w", 1, true, 0).unwrap();
        assert_eq!(reg.state("w", 0), Some(WorkerState::Draining));
        reg.set_drain_intent("w", false);
        assert_eq!(reg.state("w", 0), Some(WorkerState::Ready));
        reg.mark_lost("w", 1).unwrap();
        assert_eq!(reg.drain("w", 1), Err(WorkerError::LostEpoch));
        reg.set_drain_intent("w", true);
        assert_eq!(reg.state("w", 1), Some(WorkerState::Lost));
        reg.register("w", 2, 2).unwrap();
        assert_eq!(reg.state("w", 2), Some(WorkerState::Joining));
        reg.observe("w", 2, true, 3).unwrap();
        assert_eq!(reg.state("w", 3), Some(WorkerState::Draining));
        reg.observe("w", 2, false, 4).unwrap();
        reg.observe("w", 2, true, 5).unwrap();
        assert_eq!(reg.state("w", 5), Some(WorkerState::Draining));
        assert!(reg.eligible(5).is_empty());
        reg.set_drain_intent("w", false);
        assert_eq!(reg.state("w", 5), Some(WorkerState::Ready));
        ready(&mut reg, "w", 3, 6);
        assert_eq!(reg.state("w", 6), Some(WorkerState::Ready));
        reg.set_drain_intent("w", true);
        assert_eq!(reg.state("w", 6), Some(WorkerState::Draining));
        reg.undrain("w");
        ready(&mut reg, "w", 4, 7);
        assert_eq!(reg.state("w", 7), Some(WorkerState::Ready));
    }

    /// M3.5 Q7: a worker given its own aging (a remote worker: suspect 5 s, lost a further 10 s) is suspect at
    /// 5 s and lost at 15 s after its last sighting, while a worker on the registry's M3 rule (suspect 15 s, lost
    /// a further 60 s) seen at the same time is still ready at 5 s and only suspect at 15 s.
    #[test]
    fn each_worker_ages_by_its_own_rule() {
        let m3 = Aging {
            suspect_after_ms: 15_000,
            lost_after_ms: 60_000,
        };
        let remote = Aging {
            suspect_after_ms: 5_000,
            lost_after_ms: 10_000,
        };
        let mut reg = WorkerRegistry::new(m3);
        reg.set_aging("worker-a", remote);
        assert_eq!(
            (reg.aging_of("worker-a"), reg.aging_of("gpu0")),
            (remote, m3)
        );
        assert_eq!(reg.aging(), m3);
        ready(&mut reg, "gpu0", 1, 0);
        ready(&mut reg, "worker-a", 1, 0);
        let states =
            |reg: &WorkerRegistry, now| (reg.state("gpu0", now), reg.state("worker-a", now));
        use WorkerState::*;
        assert_eq!(states(&reg, 4_999), (Some(Ready), Some(Ready)));
        assert_eq!(states(&reg, 5_000), (Some(Ready), Some(Suspect)));
        assert_eq!(reg.eligible(5_000).len(), 1);
        assert_eq!(
            reg.begin_work("worker-a", 1, 5_000),
            Err(WorkerError::NotEligible(Suspect))
        );
        assert_eq!(states(&reg, 14_999), (Some(Ready), Some(Suspect)));
        assert_eq!(states(&reg, 15_000), (Some(Suspect), Some(Lost)));
        assert_eq!(
            reg.lifecycle_view("worker-a", 15_000).unwrap().worker_state,
            Lost
        );
        assert_eq!(
            reg.observe("worker-a", 1, true, 15_000),
            Err(WorkerError::LostEpoch)
        );
        assert_eq!(
            reg.register("worker-a", 1, 15_000).unwrap_err(),
            WorkerError::LostEpoch
        );
        assert_eq!(states(&reg, 74_999).0, Some(Suspect));
        assert_eq!(states(&reg, 75_000).0, Some(Lost));
    }

    /// M3.5 Q7: a view with a lease is eligible only before `lease_until_ms`; at and after it the worker takes no
    /// new work (suspect) until a newer view renews it. Aging still applies within the lease.
    #[test]
    fn lease_bounds_view_eligibility() {
        let view = WorkerView {
            health: WorkerHealth::Ready,
            last_seen_ms: 0,
            epoch: 1,
            draining: false,
            lease_until_ms: Some(50),
        };
        assert_eq!(view.state_at(49, AGING), WorkerState::Ready);
        assert_eq!(view.state_at(50, AGING), WorkerState::Suspect);
        assert_eq!(view.state_at(51, AGING), WorkerState::Suspect);
        assert_eq!(view.state_at(SUSPECT + LOST, AGING), WorkerState::Lost);
        let renewed = WorkerView {
            lease_until_ms: Some(SUSPECT * 2),
            ..view
        };
        assert_eq!(renewed.state_at(50, AGING), WorkerState::Ready);
        assert_eq!(renewed.state_at(SUSPECT, AGING), WorkerState::Suspect);
        let unleased = WorkerView {
            lease_until_ms: None,
            ..view
        };
        assert_eq!(unleased.state_at(SUSPECT - 1, AGING), WorkerState::Ready);
    }
}
