//! Team pools at the grant (net-teleport design note 4.4, C13-C19, D-T7, D-T8): tier filtering with no spill,
//! owner priority ahead of DRR and the non-owner work cap, classes with the batch floor, per-scope quota overrides,
//! and owner-only windows and the worker's local switch. Admission alone, no server: every worker is ready and
//! never ages.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_runtime::admission::{
    Admission, AdmissionError, AdmissionLimits, Permit, PoolGrants, Priority, Route,
    SchedulePolicy, ScopePolicy, Target, WorkerCap,
};
use cere_runtime::pool::{Class, Window, WorkerPool};
use cere_runtime::tier::Tier;
use cere_runtime::worker::{Aging, WorkerHealth, WorkerView};
use futures_util::FutureExt;
use tokio::time::Instant;

type Acquire = Pin<Box<dyn Future<Output = Result<Permit, AdmissionError>> + Send>>;

const NEVER: Aging = Aging {
    suspect_after_ms: u64::MAX,
    lost_after_ms: u64::MAX,
};

const CAP: Duration = Duration::from_secs(5);

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn limits(max_inflight: u64, per_scope_inflight: u64) -> AdmissionLimits {
    AdmissionLimits {
        max_inflight,
        per_scope_inflight,
        queue_depth: 1000,
    }
}

fn scope(class: Class, tiers: &[Tier]) -> ScopePolicy {
    ScopePolicy {
        class,
        per_scope_inflight: None,
        max_batch_inflight: None,
        allowed_tiers: tiers.to_vec(),
    }
}

fn policy(floor: f64) -> SchedulePolicy {
    SchedulePolicy::new([], Duration::from_secs(3600)).with_pool_limits(floor, CAP)
}

fn owned(owner: &str, tier: Tier, windows: &[&str]) -> WorkerPool {
    WorkerPool {
        owner: Some(Arc::from(owner)),
        tier,
        windows: windows.iter().map(|w| Window::parse(w).unwrap()).collect(),
    }
}

fn unowned(tier: Tier) -> WorkerPool {
    WorkerPool {
        owner: None,
        tier,
        windows: Vec::new(),
    }
}

/// Ready workers `(id, cap, pool)`.
fn admission(
    limits: AdmissionLimits,
    policy: SchedulePolicy,
    workers: Vec<(&str, u64, WorkerPool)>,
) -> Admission {
    let adm = Admission::with_policy(limits, policy);
    adm.set_workers(
        workers
            .iter()
            .map(|(id, cap, _)| WorkerCap {
                id: (*id).to_owned(),
                cap: *cap,
                aging: NEVER,
                deployment: 0,
            })
            .collect(),
    );
    for (id, _, pool) in workers {
        adm.set_worker_pool(id, pool);
        adm.update_worker(
            id,
            WorkerView {
                health: WorkerHealth::Ready,
                last_seen_ms: unix_ms(),
                epoch: 1,
                draining: false,
                lease_until_ms: None,
            },
        );
    }
    adm
}

fn far() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

/// A waiter of `scope` with `class` (`None`: the scope's default) until `deadline`, queued at its first poll.
fn waiter(adm: &Admission, scope: &str, class: Option<Class>, deadline: Instant) -> Acquire {
    let (adm, scope) = (adm.clone(), scope.to_owned());
    let route = Route {
        class,
        ..Route::default()
    };
    Box::pin(async move {
        adm.acquire_routed(&scope, Priority::Normal, deadline, Target::Any(0), 1, route)
            .await
    })
}

/// Polls `w` once: `Some` once it resolved.
fn poll(w: &mut Acquire) -> Option<Result<Permit, AdmissionError>> {
    w.now_or_never()
}

fn queued(adm: &Admission, scope: &str, class: Option<Class>) -> Acquire {
    let mut w = waiter(adm, scope, class, far());
    assert!(poll(&mut w).is_none(), "{scope} is granted at once");
    w
}

fn granted(adm: &Admission, scope: &str, class: Option<Class>) -> Permit {
    poll(&mut waiter(adm, scope, class, far()))
        .expect("granted at once")
        .unwrap()
}

/// D-T7: a scope is granted only on a worker whose tier (for it) it allows; it never spills to another tier, and
/// with no allowed worker at all it queues and then fails `no_allowed_worker`.
#[tokio::test]
async fn tiers_filter_placement_and_never_spill() {
    let policy = policy(0.1)
        .with_scope("own_only", scope(Class::Interactive, &[Tier::Own]))
        .with_scope(
            "teamer",
            scope(Class::Interactive, &[Tier::Own, Tier::Team]),
        )
        .with_scope("provider", scope(Class::Interactive, &[Tier::Provider]));
    let adm = admission(
        limits(8, 8),
        policy,
        vec![
            ("mine", 1, unowned(Tier::Own)),
            ("team", 4, unowned(Tier::Team)),
        ],
    );
    let first = granted(&adm, "own_only", None);
    assert_eq!(first.worker(), "mine");
    assert_eq!(first.team().unwrap().tier, Tier::Own);
    // `mine` is full and `team` is free, but `team` is not an allowed tier: the waiter stays queued.
    let mut second = queued(&adm, "own_only", None);
    let teamer = granted(&adm, "teamer", None);
    assert_eq!(teamer.worker(), "team");
    assert!(poll(&mut second).is_none());
    drop(first);
    assert_eq!(poll(&mut second).unwrap().unwrap().worker(), "mine");

    // No worker of an allowed tier: queued until the deadline, then `no_allowed_worker`; nothing was granted.
    let before = adm.worker_loads("provider");
    let err = waiter(
        &adm,
        "provider",
        None,
        Instant::now() + Duration::from_millis(30),
    )
    .await
    .unwrap_err();
    assert_eq!(err, AdmissionError::NoAllowedWorker);
    assert_eq!(err.code(), "no_allowed_worker");
    assert_eq!(adm.worker_loads("provider"), before);
    assert_eq!(adm.scope_inflight("provider"), 0);
}

/// The owner's tier is `own` on their worker whatever its configured tier; everyone else sees the configured tier.
#[tokio::test]
async fn the_owner_sees_own_on_their_worker() {
    let policy = policy(0.1)
        .with_scope("alice", scope(Class::Interactive, &[Tier::Own]))
        .with_scope("bob", scope(Class::Interactive, &[Tier::Own]));
    let adm = admission(
        limits(8, 8),
        policy,
        vec![("alice-gpu", 4, owned("alice", Tier::Team, &[]))],
    );
    let alice = granted(&adm, "alice", None);
    assert_eq!(alice.team().unwrap().tier, Tier::Own);
    assert_eq!(alice.team().unwrap().owner.as_deref(), Some("alice"));
    let err = waiter(
        &adm,
        "bob",
        None,
        Instant::now() + Duration::from_millis(30),
    )
    .await
    .unwrap_err();
    assert_eq!(err, AdmissionError::NoAllowedWorker);
}

/// Q-T6: the next free slot on an owned worker goes to its owner's waiter ahead of earlier waiters of other scopes
/// (DRR), and a non-owner's grant on an owned worker carries the `owner_wait_cap_ms` work cap.
#[tokio::test]
async fn owner_priority_takes_the_next_free_slot_and_non_owners_are_capped() {
    let policy = policy(0.1).with_scope("bob", scope(Class::Interactive, &[Tier::Own, Tier::Team]));
    let adm = admission(
        limits(8, 8),
        policy,
        vec![("alice-gpu", 1, owned("alice", Tier::Team, &[]))],
    );
    let bob = granted(&adm, "bob", None);
    let team = bob.team().unwrap();
    assert_eq!((team.tier, team.work_cap), (Tier::Team, Some(CAP)));
    let mut bob2 = queued(&adm, "bob", None);
    let mut alice = queued(&adm, "alice", None);
    drop(bob);
    let alice = poll(&mut alice).unwrap().unwrap();
    assert_eq!(alice.worker(), "alice-gpu");
    assert_eq!(alice.team().unwrap().work_cap, None);
    assert!(poll(&mut bob2).is_none());
    assert_eq!(adm.pool_grants().owner_priority, 1);
    drop(alice);
    let bob2 = poll(&mut bob2).unwrap().unwrap();
    assert_eq!(bob2.team().unwrap().work_cap, Some(CAP));
    assert_eq!(adm.pool_grants().owner_priority, 1);
}

/// An unowned worker has no work cap for anyone.
#[tokio::test]
async fn unowned_workers_carry_no_work_cap() {
    let adm = admission(
        limits(8, 8),
        policy(0.1),
        vec![("w", 4, unowned(Tier::Own))],
    );
    let p = granted(&adm, "anyone", None);
    assert_eq!(p.team().unwrap().work_cap, None);
    assert_eq!(p.team().unwrap().owner, None);
}

/// Grants one slot at a time from a queue where both classes always wait; returns the class of each grant.
fn contested_grants(floor: f64, grants: usize) -> (Vec<Class>, PoolGrants) {
    let policy = policy(floor).with_scope("batcher", scope(Class::Batch, &[Tier::Own]));
    let adm = admission(limits(1, 1), policy, vec![("w", 1, unowned(Tier::Own))]);
    let mut held = Some(granted(&adm, "holder", None));
    let mut waiters: Vec<(Class, Acquire)> = Vec::new();
    for _ in 0..grants {
        // The batch scope's default class, and an explicit header class from an interactive scope.
        waiters.push((Class::Batch, queued(&adm, "batcher", None)));
        waiters.push((
            Class::Interactive,
            queued(&adm, "chat", Some(Class::Interactive)),
        ));
    }
    let mut order = Vec::new();
    for _ in 0..grants {
        drop(held.take());
        let resolved: Vec<(usize, Permit)> = waiters
            .iter_mut()
            .enumerate()
            .filter_map(|(i, (_, w))| poll(w).map(|r| (i, r.unwrap())))
            .collect();
        assert_eq!(resolved.len(), 1, "one grant per freed slot");
        let (i, permit) = resolved.into_iter().next().unwrap();
        let (class, _) = waiters.remove(i);
        assert_eq!(permit.class(), class);
        held = Some(permit);
        order.push(class);
    }
    // Counted before the last slot frees (that would grant once more).
    let counts = adm.pool_grants();
    drop(held);
    (order, counts)
}

/// C17: interactive is served before batch, and batch keeps `batch_floor` of the grants while both wait: after
/// every contested grant, batch holds at least ceil(floor x grants) of them (and no more).
#[tokio::test]
async fn batch_keeps_its_floor_while_interactive_saturates() {
    let (order, counts) = contested_grants(0.1, 50);
    let mut batch = 0;
    for (k, class) in order.iter().enumerate() {
        batch += usize::from(*class == Class::Batch);
        assert_eq!(batch, (k + 1).div_ceil(10), "after {} grants", k + 1);
    }
    assert_eq!(counts.contested, [45, 5]);

    let (order, _) = contested_grants(0.25, 40);
    assert_eq!(order.iter().filter(|c| **c == Class::Batch).count(), 10);

    // Floor 0: batch waits for as long as interactive waits.
    let (order, counts) = contested_grants(0.0, 20);
    assert!(order.iter().all(|c| *c == Class::Interactive));
    assert_eq!(counts.contested, [20, 0]);
}

/// Batch alone is served (not contested, not counted).
#[tokio::test]
async fn batch_alone_is_served() {
    let policy = policy(0.1).with_scope("batcher", scope(Class::Batch, &[Tier::Own]));
    let adm = admission(limits(1, 1), policy, vec![("w", 1, unowned(Tier::Own))]);
    let first = granted(&adm, "batcher", None);
    assert_eq!(first.class(), Class::Batch);
    let mut next = queued(&adm, "batcher", None);
    drop(first);
    assert!(poll(&mut next).unwrap().is_ok());
    assert_eq!(adm.pool_grants().contested, [0, 0]);
}

/// C18: a scope's `per_scope_inflight` replaces the global one; `max_batch_inflight` caps its batch rows only.
#[tokio::test]
async fn per_scope_quota_overrides() {
    let mut small = scope(Class::Interactive, &[Tier::Own]);
    small.per_scope_inflight = Some(1);
    let mut capped = scope(Class::Interactive, &[Tier::Own]);
    capped.max_batch_inflight = Some(1);
    let policy = policy(0.1)
        .with_scope("small", small)
        .with_scope("capped", capped);
    let adm = admission(limits(8, 4), policy, vec![("w", 8, unowned(Tier::Own))]);

    let _s1 = granted(&adm, "small", None);
    let mut s2 = queued(&adm, "small", None);
    let others: Vec<Permit> = (0..4).map(|_| granted(&adm, "other", None)).collect();
    assert!(poll(&mut s2).is_none());
    drop(others);

    let _b1 = granted(&adm, "capped", Some(Class::Batch));
    let mut b2 = queued(&adm, "capped", Some(Class::Batch));
    let _i1 = granted(&adm, "capped", Some(Class::Interactive));
    assert!(poll(&mut b2).is_none());
    assert_eq!(adm.scope_inflight("capped"), 2);
}

/// C19 / Q-T8: inside an owner-only window the worker is granted only to its owner; other scopes go elsewhere or
/// wait, and fail `no_allowed_worker` when no worker admits them.
#[tokio::test]
async fn a_window_grants_only_the_owner() {
    let all_week = "Mon-Sun 00:00-24:00 UTC";
    let policy = policy(0.1).with_scope("bob", scope(Class::Interactive, &[Tier::Own, Tier::Team]));
    let adm = admission(
        limits(8, 8),
        policy,
        vec![
            ("alice-gpu", 4, owned("alice", Tier::Team, &[all_week])),
            ("shared", 1, unowned(Tier::Own)),
        ],
    );
    let bob = granted(&adm, "bob", None);
    assert_eq!(bob.worker(), "shared");
    // `alice-gpu` has room, but it is inside its window: bob waits for `shared`.
    let mut bob2 = queued(&adm, "bob", None);
    let alice = granted(&adm, "alice", None);
    assert_eq!(alice.worker(), "alice-gpu");
    assert!(poll(&mut bob2).is_none());
    drop(bob);
    assert_eq!(poll(&mut bob2).unwrap().unwrap().worker(), "shared");

    let (_, states) = adm.owner_only_states();
    let (_, owner, since) = states.iter().find(|s| &*s.0 == "alice-gpu").unwrap();
    assert_eq!(&**owner, "alice");
    assert!(since.is_some());

    // Only the windowed worker: no allowed worker for bob.
    let lone = admission(
        limits(8, 8),
        policy_with_bob(),
        vec![("alice-gpu", 4, owned("alice", Tier::Team, &[all_week]))],
    );
    let err = waiter(
        &lone,
        "bob",
        None,
        Instant::now() + Duration::from_millis(30),
    )
    .await
    .unwrap_err();
    assert_eq!(err, AdmissionError::NoAllowedWorker);
}

fn policy_with_bob() -> SchedulePolicy {
    policy(0.1).with_scope("bob", scope(Class::Interactive, &[Tier::Own, Tier::Team]))
}

/// C19: the worker's local switch (its heartbeat's `owner_only`): in-flight non-owner work keeps its slot and
/// finishes; no new non-owner grant until it is off again.
#[tokio::test]
async fn the_local_switch_stops_new_non_owner_grants_and_keeps_in_flight_work() {
    let adm = admission(
        limits(8, 8),
        policy_with_bob(),
        vec![("alice-gpu", 4, owned("alice", Tier::Team, &[]))],
    );
    let in_flight = granted(&adm, "bob", None);
    adm.set_owner_only("alice-gpu", true);
    assert_eq!(adm.scope_inflight("bob"), 1);
    assert_eq!(in_flight.worker(), "alice-gpu");
    let err = waiter(
        &adm,
        "bob",
        None,
        Instant::now() + Duration::from_millis(30),
    )
    .await
    .unwrap_err();
    assert_eq!(err, AdmissionError::NoAllowedWorker);
    let alice = granted(&adm, "alice", None);
    assert_eq!(alice.worker(), "alice-gpu");
    let mut bob2 = queued(&adm, "bob", None);
    drop(in_flight);
    assert!(poll(&mut bob2).is_none());
    adm.set_owner_only("alice-gpu", false);
    assert_eq!(poll(&mut bob2).unwrap().unwrap().worker(), "alice-gpu");
    let (_, states) = adm.owner_only_states();
    assert_eq!(states[0].2, None);
}
