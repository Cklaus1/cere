//! P6 coordinator registry (m35 design note Q4, Q6, Q7, Q7a, Q13, Q14; build-loop-m35 design note "CREG split"
//! D2, D5): an in-process `cered` App with `[federation]` and an rcgen PKI, a tonic `WorkerRegistryService`
//! client acting as the agent, and the remote-state API the readiness driver (RDY) and the drain routes (OPS)
//! call. Short timings: heartbeat 100 ms, lease 200 ms, suspect 400 ms, lost 800 ms (absolute ages).

mod support;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::MockConfig;
use cere_proto::worker::v2 as pb;
use cere_runtime::app::DrainError;
use cere_runtime::fed::registry::{RemoteError, RemoteRegistry, RemoteState};
use cere_runtime::fed::wire::{self, AdmissionState};
use cere_runtime::readiness::SmokeStatus;
use cere_runtime::worker::WorkerState;
use pb::worker_registry_service_client::WorkerRegistryServiceClient;
use serde_json::Value;
use support::pki::TestPki;
use support::{
    journal, test_tuning, tls_with, Cered, ConfigFile, Federation, Harness, FED_CLUSTER,
    FED_COORDINATOR,
};
use tonic::transport::Channel;
use tonic::Code;

const A: &str = "worker-a";
const B: &str = "worker-b";
const PRINCIPAL_A: &str = "cere-worker-a";
const PRINCIPAL_B: &str = "cere-worker-b";
const LOCAL: &str = "gpu1";
const LEASE_MS: u64 = 200;
const SUSPECT_MS: u64 = 400;
const LOST_MS: u64 = 800;
const TIMINGS: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 200\nsuspect_after_ms = 400\n\
                       lost_after_ms = 800\n";

type Client = WorkerRegistryServiceClient<Channel>;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

async fn start_with(keys: &str) -> (Harness, Federation) {
    let fed = Federation::new();
    let text = fed.toml(keys, &[(A, PRINCIPAL_A), (B, PRINCIPAL_B)]);
    let h = Harness::start_federated(MockConfig::default(), &text, test_tuning()).await;
    (h, fed)
}

async fn start() -> (Harness, Federation) {
    start_with(TIMINGS).await
}

fn agent(fed: &Federation, principal: &str) -> Client {
    WorkerRegistryServiceClient::new(fed.channel(fed.agent_tls(principal)))
}

fn registry(h: &Harness) -> &RemoteRegistry {
    h.app.remote_workers().expect("[federation] configured")
}

fn state(h: &Harness, id: &str) -> RemoteState {
    registry(h)
        .get(id, now_ms())
        .expect("configured remote worker")
}

fn identity(id: &str, epoch: u64) -> Option<pb::WorkerIdentity> {
    Some(pb::WorkerIdentity {
        cluster_id: FED_CLUSTER.into(),
        worker_id: id.into(),
        worker_epoch: epoch,
    })
}

fn register_req(id: &str, epoch: u64) -> pb::RegisterRequest {
    pb::RegisterRequest {
        proposed: identity(id, epoch),
        boot_id: "0123456789abcdef0123456789abcdef".into(),
        protocol_revision: cere_proto::PROTOCOL_REVISION,
        inventory_sha256: vec![7; 32],
    }
}

async fn register(client: &mut Client, id: &str, epoch: u64) -> pb::RegisterReply {
    client
        .register(register_req(id, epoch))
        .await
        .unwrap()
        .into_inner()
}

/// A heartbeat at `term` from `id@epoch` reporting `generation` and `admission`.
fn heartbeat_req(
    id: &str,
    epoch: u64,
    term: u64,
    sequence: u64,
    generation: u64,
    admission: &str,
) -> pb::HeartbeatRequest {
    pb::HeartbeatRequest {
        worker: identity(id, epoch),
        sequence,
        observed_coordinator_term: term,
        inventory_sha256: vec![7; 32],
        model_generation: generation,
        admission_state: admission.into(),
        owner_only: false,
    }
}

async fn heartbeat(client: &mut Client, req: pb::HeartbeatRequest) -> pb::HeartbeatReply {
    client.heartbeat(req).await.unwrap().into_inner()
}

fn worker_epoch_records(state_dir: &std::path::Path, id: &str) -> Vec<u64> {
    journal(state_dir)
        .iter()
        .filter(|r| r["type"] == "worker_epoch" && r["worker"] == id)
        .map(|r| r["epoch"].as_u64().unwrap())
        .collect()
}

fn refused(reply: &pb::RegisterReply, reason: &str) {
    assert!(!reply.accepted, "{reply:?}");
    assert_eq!(reply.reason_code, reason, "{reply:?}");
}

/// Drives `id@epoch` (registered, Joining) to Ready the way the readiness driver does: reconcile generation 1,
/// an `open` heartbeat that carries the grant, a passed smoke, then Ready. Returns the next heartbeat sequence.
async fn drive_ready(h: &Harness, client: &mut Client, id: &str, epoch: u64, sequence: u64) -> u64 {
    let reg = registry(h);
    let round = state(h, id).round;
    reg.mark_reconciled(id, round, 1).unwrap();
    let reply = heartbeat(
        client,
        heartbeat_req(id, epoch, h.app.epoch(), sequence, 1, "open"),
    )
    .await;
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(reply.reconciled_generation, Some(1));
    reg.set_smoke(
        id,
        round,
        SmokeStatus::Passed {
            instance_id: "remote-instance".into(),
        },
    )
    .unwrap();
    reg.mark_ready(id, round, now_ms()).unwrap();
    let ready = state(h, id);
    let expected = if ready.draining {
        WorkerState::Draining
    } else {
        WorkerState::Ready
    };
    assert_eq!(ready.state, expected);
    sequence + 1
}

#[tokio::test(flavor = "multi_thread")]
async fn register_outcomes_follow_q4() {
    let (h, fed) = start().await;
    let mut a = agent(&fed, PRINCIPAL_A);
    let term = h.app.epoch();
    let before = state(&h, A);
    assert_eq!(before.state, WorkerState::Joining);
    assert_eq!((before.epoch, before.term), (None, None));

    let mut wrong_revision = register_req(A, 3);
    wrong_revision.protocol_revision = 1;
    let reply = a.register(wrong_revision).await.unwrap().into_inner();
    refused(&reply, wire::UNSUPPORTED_REVISION);
    let mut wrong_cluster = register_req(A, 3);
    wrong_cluster.proposed.as_mut().unwrap().cluster_id = "other-cluster".into();
    let reply = a.register(wrong_cluster).await.unwrap().into_inner();
    refused(&reply, wire::WRONG_CLUSTER);
    assert!(worker_epoch_records(h.state_dir.path(), A).is_empty());
    assert_eq!(state(&h, A).epoch, None);

    // First registration: the agent's epoch is journaled before the reply; Joining under this term.
    let reply = register(&mut a, A, 3).await;
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(reply.coordinator_term, term);
    assert_eq!(reply.admission_lease_ms, LEASE_MS);
    assert!(reply.requires_reconciliation);
    assert_eq!(reply.reason_code, "");
    assert_eq!(worker_epoch_records(h.state_dir.path(), A), vec![3]);
    let joined = state(&h, A);
    assert_eq!(joined.state, WorkerState::Joining);
    assert_eq!((joined.epoch, joined.term), (Some(3), Some(term)));

    // Lower epoch: stale, with the journaled one.
    let reply = register(&mut a, A, 2).await;
    refused(&reply, wire::STALE_EPOCH);
    assert_eq!(reply.journaled_worker_epoch, 3);

    // Same epoch, not Lost: a reconnect keeps the state (round, reconciled generation, heartbeat sequence).
    let round = joined.round;
    registry(&h).mark_reconciled(A, round, 1).unwrap();
    let reply = heartbeat(&mut a, heartbeat_req(A, 3, term, 1, 1, "fenced")).await;
    assert!(reply.accepted, "{reply:?}");
    let reply = register(&mut a, A, 3).await;
    assert!(reply.accepted && reply.requires_reconciliation, "{reply:?}");
    assert_eq!(worker_epoch_records(h.state_dir.path(), A), vec![3]);
    let kept = state(&h, A);
    assert_eq!(kept.round, round);
    assert_eq!(kept.reconciled_generation, Some(1));
    assert_eq!(kept.generation, Some(1));
    let replay = heartbeat(&mut a, heartbeat_req(A, 3, term, 1, 1, "fenced")).await;
    assert_eq!(replay.reason_code, wire::REPLAYED_HEARTBEAT, "{replay:?}");

    // Higher epoch: a new incarnation, journaled; the remote state resets (D5), Joining in a new round.
    let reply = register(&mut a, A, 7).await;
    assert!(reply.accepted && reply.requires_reconciliation, "{reply:?}");
    assert_eq!(worker_epoch_records(h.state_dir.path(), A), vec![3, 7]);
    let fresh = state(&h, A);
    assert_eq!(fresh.state, WorkerState::Joining);
    assert_eq!(fresh.epoch, Some(7));
    assert!(fresh.round > round);
    assert_eq!(fresh.reconciled_generation, None);
    assert_eq!(fresh.generation, None);
    assert_eq!(fresh.last_heartbeat_ms, None);
    // The heartbeat sequence restarts per epoch.
    let reply = heartbeat(&mut a, heartbeat_req(A, 7, term, 1, 1, "fenced")).await;
    assert!(reply.accepted, "{reply:?}");
    // Another worker is untouched.
    assert_eq!(state(&h, B).epoch, None);
}

#[tokio::test(flavor = "multi_thread")]
async fn joining_ages_lost_at_the_absolute_bound_and_lost_epochs_never_rejoin() {
    let (h, fed) = start().await;
    let mut a = agent(&fed, PRINCIPAL_A);
    let term = h.app.epoch();
    let before = now_ms();
    assert!(register(&mut a, A, 1).await.accepted);
    let after = now_ms();
    let reg = registry(&h);
    assert_eq!(
        reg.get(A, before + LOST_MS - 1).unwrap().state,
        WorkerState::Joining
    );
    assert_eq!(
        reg.get(A, after + LOST_MS).unwrap().state,
        WorkerState::Lost
    );

    tokio::time::sleep(Duration::from_millis(LOST_MS + 100)).await;
    assert_eq!(state(&h, A).state, WorkerState::Lost);
    let reply = heartbeat(&mut a, heartbeat_req(A, 1, term, 1, 1, "fenced")).await;
    assert!(!reply.accepted);
    assert_eq!(reply.reason_code, wire::EPOCH_LOST);
    refused(&register(&mut a, A, 1).await, wire::EPOCH_LOST);
    let reply = register(&mut a, A, 2).await;
    assert!(reply.accepted, "{reply:?}");
    let rejoined = state(&h, A);
    assert_eq!(rejoined.state, WorkerState::Joining);
    assert_eq!(rejoined.epoch, Some(2));
    assert_eq!(worker_epoch_records(h.state_dir.path(), A), vec![1, 2]);
}

#[tokio::test(flavor = "multi_thread")]
async fn refused_callers_change_no_state() {
    let (h, fed) = start_with(&format!("{TIMINGS}rpc_quota_per_principal = 2\n")).await;
    let untouched = |h: &Harness| {
        assert!(worker_epoch_records(h.state_dir.path(), A).is_empty());
        assert!(worker_epoch_records(h.state_dir.path(), B).is_empty());
        assert_eq!(state(h, A).epoch, None);
        assert_eq!(state(h, B).epoch, None);
    };

    // A client certificate from a foreign CA fails the handshake.
    let foreign = TestPki::new().leaf(PRINCIPAL_A);
    let mut client = WorkerRegistryServiceClient::new(fed.channel(tls_with(
        fed.pki.ca_pem(),
        &foreign,
        FED_COORDINATOR,
    )));
    let status = client.register(register_req(A, 1)).await.unwrap_err();
    assert_ne!(status.code(), Code::PermissionDenied, "{status:?}");
    assert!(
        std::error::Error::source(&status).is_some_and(|s| s.is::<tonic::transport::Error>()),
        "not a transport error: {status:?}"
    );

    // A principal bound to no worker, and a worker principal claiming another worker id.
    let mut unbound = agent(&fed, "cere-worker-z");
    for status in [
        unbound.register(register_req(A, 1)).await.unwrap_err(),
        unbound
            .heartbeat(heartbeat_req(A, 1, h.app.epoch(), 1, 1, "open"))
            .await
            .unwrap_err(),
    ] {
        assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");
        assert_eq!(wire::reason(&status), wire::PERMISSION_DENIED_REASON);
    }
    let mut a = agent(&fed, PRINCIPAL_A);
    let status = a.register(register_req(B, 1)).await.unwrap_err();
    assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");

    // Quota: every slot of the principal held -> RESOURCE_EXHAUSTED before the handler.
    let quota = registry(&h).quota();
    let held = [
        quota.try_acquire(PRINCIPAL_A).unwrap(),
        quota.try_acquire(PRINCIPAL_A).unwrap(),
    ];
    let status = a.register(register_req(A, 1)).await.unwrap_err();
    assert_eq!(status.code(), Code::ResourceExhausted, "{status:?}");
    assert_eq!(wire::reason(&status), wire::QUOTA_EXHAUSTED);
    // Another principal has its own pool.
    let mut b = agent(&fed, PRINCIPAL_B);
    let reply = b
        .heartbeat(heartbeat_req(B, 1, h.app.epoch(), 1, 1, "open"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(reply.reason_code, wire::UNKNOWN_WORKER);
    drop(held);

    // A message above the 1 MiB control limit is refused by the decoder.
    let mut oversize = register_req(A, 1);
    oversize.boot_id = "x".repeat(wire::CONTROL_MESSAGE_MAX);
    let status = a.register(oversize).await.unwrap_err();
    assert_eq!(status.code(), Code::OutOfRange, "{status:?}");

    untouched(&h);
    // The same principal is served once its slots are free.
    assert!(register(&mut a, A, 1).await.accepted);
}

#[tokio::test(flavor = "multi_thread")]
async fn heartbeats_are_checked_and_grant_leases() {
    let (h, fed) = start().await;
    let mut a = agent(&fed, PRINCIPAL_A);
    let term = h.app.epoch();

    // Not registered in this term.
    let reply = heartbeat(&mut a, heartbeat_req(A, 1, term, 1, 1, "fenced")).await;
    assert!(!reply.accepted);
    assert_eq!(reply.reason_code, wire::UNKNOWN_WORKER);
    assert_eq!(state(&h, A).last_heartbeat_ms, None);

    assert!(register(&mut a, A, 4).await.accepted);
    let mut wrong_cluster = heartbeat_req(A, 4, term, 1, 1, "fenced");
    wrong_cluster.worker.as_mut().unwrap().cluster_id = "other".into();
    for (req, reason) in [
        (wrong_cluster, wire::WRONG_CLUSTER),
        (
            heartbeat_req(A, 4, term + 1, 1, 1, "fenced"),
            wire::STALE_TERM,
        ),
        (
            heartbeat_req(A, 4, term - 1, 1, 1, "fenced"),
            wire::STALE_TERM,
        ),
        (heartbeat_req(A, 3, term, 1, 1, "fenced"), wire::STALE_EPOCH),
        // An epoch it has not registered: Register first.
        (
            heartbeat_req(A, 5, term, 1, 1, "fenced"),
            wire::UNKNOWN_WORKER,
        ),
    ] {
        let reply = heartbeat(&mut a, req).await;
        assert!(!reply.accepted, "{reply:?}");
        assert_eq!(reply.reason_code, reason, "{reply:?}");
    }
    assert_eq!(state(&h, A).last_heartbeat_ms, None);
    let status = a
        .heartbeat(heartbeat_req(A, 4, term, 1, 1, "sleepy"))
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");

    let received = now_ms();
    let reply = heartbeat(&mut a, heartbeat_req(A, 4, term, 5, 2, "fenced")).await;
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(reply.coordinator_term, term);
    assert_eq!(reply.admission_lease_ms, LEASE_MS);
    assert_eq!(reply.lease_sequence, 1);
    assert_eq!(reply.desired_state, "");
    assert_eq!(reply.reconciled_generation, None);
    assert_eq!(reply.reason_code, "");
    let seen = state(&h, A);
    let at = seen.last_heartbeat_ms.unwrap();
    assert!(at >= received, "aged by receive time");
    assert_eq!(seen.lease_until_ms, Some(at + LEASE_MS));
    assert_eq!(seen.generation, Some(2));
    assert_eq!(seen.admission_state, Some(AdmissionState::Fenced));
    assert_eq!(seen.reported_inventory_sha256, Some([7; 32]));

    // Sequence must strictly increase within the epoch; a replay changes nothing.
    for sequence in [5, 4] {
        let reply = heartbeat(&mut a, heartbeat_req(A, 4, term, sequence, 2, "fenced")).await;
        assert_eq!(reply.reason_code, wire::REPLAYED_HEARTBEAT, "{reply:?}");
        assert!(!reply.accepted);
    }
    assert_eq!(state(&h, A).last_heartbeat_ms, Some(at));

    // Reconciliation done: every following reply carries the generation.
    registry(&h).mark_reconciled(A, seen.round, 2).unwrap();
    let reply = heartbeat(&mut a, heartbeat_req(A, 4, term, 6, 2, "fenced")).await;
    assert_eq!(reply.reconciled_generation, Some(2));
    assert_eq!(reply.lease_sequence, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn ready_ages_by_receive_time_and_q7a_events_return_to_joining() {
    let (h, fed) = start().await;
    let mut a = agent(&fed, PRINCIPAL_A);
    let term = h.app.epoch();
    assert!(register(&mut a, A, 1).await.accepted);
    let reg = registry(&h);

    // Ready needs reconciliation, an open report and a passed smoke.
    let round = state(&h, A).round;
    assert_eq!(
        reg.mark_ready(A, round, now_ms()),
        Err(RemoteError::NotReconciled)
    );
    assert_eq!(
        reg.mark_reconciled(A, round + 1, 1),
        Err(RemoteError::StaleRound { current: round })
    );
    let mut sequence = drive_ready(&h, &mut a, A, 1, 1).await;
    let ready = state(&h, A);
    assert_eq!(
        ready.smoke,
        SmokeStatus::Passed {
            instance_id: "remote-instance".into()
        }
    );

    // Heartbeats keep it Ready past the suspect bound.
    let until = now_ms() + SUSPECT_MS + 200;
    while now_ms() < until {
        let reply = heartbeat(&mut a, heartbeat_req(A, 1, term, sequence, 1, "open")).await;
        assert!(reply.accepted, "{reply:?}");
        sequence += 1;
        assert_eq!(state(&h, A).state, WorkerState::Ready);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Silence: the lease view expires first (Suspect for eligibility), then Suspect, then Lost at the absolute
    // bound, all measured from the last receive time.
    let last = state(&h, A).last_heartbeat_ms.unwrap();
    let at = |ms: u64| reg.get(A, last + ms).unwrap().state;
    assert_eq!(at(LEASE_MS - 1), WorkerState::Ready);
    assert_eq!(at(LEASE_MS), WorkerState::Suspect);
    assert_eq!(at(SUSPECT_MS), WorkerState::Suspect);
    assert_eq!(at(LOST_MS - 1), WorkerState::Suspect);
    assert_eq!(at(LOST_MS), WorkerState::Lost);
    let view = reg.worker_view(A).unwrap();
    assert_eq!(view.lease_until_ms, Some(last + LEASE_MS));
    assert_eq!(view.epoch, 1);

    // A `fenced` report while Ready -> Joining (new round, reconciliation withheld).
    let reply = heartbeat(&mut a, heartbeat_req(A, 1, term, sequence, 1, "fenced")).await;
    sequence += 1;
    assert!(reply.accepted);
    assert_eq!(reply.reconciled_generation, None);
    let joining = state(&h, A);
    assert_eq!(joining.state, WorkerState::Joining);
    assert!(joining.round > ready.round);
    assert_eq!(
        reg.mark_ready(A, ready.round, now_ms()),
        Err(RemoteError::StaleRound {
            current: joining.round
        })
    );

    // A generation change while Ready -> Joining.
    sequence = drive_ready(&h, &mut a, A, 1, sequence).await;
    let reply = heartbeat(&mut a, heartbeat_req(A, 1, term, sequence, 2, "open")).await;
    assert!(reply.accepted);
    assert_eq!(reply.reconciled_generation, None);
    assert_eq!(state(&h, A).state, WorkerState::Joining);

    // The driver may also take a Ready worker back to Joining (a failed re-check).
    sequence = drive_ready(&h, &mut a, A, 1, sequence + 1).await;
    let round = state(&h, A).round;
    let next = reg.enter_joining(A, round).unwrap();
    assert!(next > round);
    assert_eq!(state(&h, A).state, WorkerState::Joining);
    let _ = sequence;
}

#[tokio::test(flavor = "multi_thread")]
async fn lease_sequence_never_restarts_within_a_term() {
    let (h, fed) = start().await;
    let mut a = agent(&fed, PRINCIPAL_A);
    let term = h.app.epoch();
    let mut grants = Vec::new();
    assert!(register(&mut a, A, 1).await.accepted);
    for sequence in 1..=2 {
        grants.push(heartbeat(&mut a, heartbeat_req(A, 1, term, sequence, 1, "fenced")).await);
    }
    // A reconnect at the same epoch and a new incarnation keep counting.
    assert!(register(&mut a, A, 1).await.accepted);
    grants.push(heartbeat(&mut a, heartbeat_req(A, 1, term, 3, 1, "fenced")).await);
    assert!(register(&mut a, A, 2).await.accepted);
    grants.push(heartbeat(&mut a, heartbeat_req(A, 2, term, 1, 1, "fenced")).await);
    let sequences: Vec<u64> = grants.iter().map(|g| g.lease_sequence).collect();
    assert_eq!(sequences, vec![1, 2, 3, 4]);
    assert!(grants
        .iter()
        .all(|g| g.accepted && g.coordinator_term == term));
    assert_eq!(state(&h, A).lease_sequence, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn coordinator_restart_keeps_the_remote_epoch_and_starts_a_new_term() {
    let (h, fed) = start().await;
    let mut a = agent(&fed, PRINCIPAL_A);
    let first_term = h.app.epoch();
    assert!(register(&mut a, A, 5).await.accepted);
    let local_before = *worker_epoch_records(h.state_dir.path(), LOCAL)
        .last()
        .unwrap();
    let mut h = h.restart().await;
    let term = h.app.epoch();
    assert!(term > first_term);
    // Not bumped at open: the coordinator records only epochs an agent registers (the open compacts the
    // journal to the live records; a local worker is bumped).
    assert_eq!(worker_epoch_records(h.state_dir.path(), A), vec![5]);
    assert_eq!(
        worker_epoch_records(h.state_dir.path(), LOCAL),
        vec![local_before + 1]
    );
    let restarted = state(&h, A);
    assert_eq!((restarted.epoch, restarted.term), (None, None));

    // The agent's heartbeat under the old term: not registered in this one.
    let mut a = agent(&fed, PRINCIPAL_A);
    let reply = heartbeat(&mut a, heartbeat_req(A, 5, first_term, 9, 1, "open")).await;
    assert_eq!(reply.reason_code, wire::UNKNOWN_WORKER);
    let reply = register(&mut a, A, 5).await;
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(reply.coordinator_term, term);
    assert!(reply.requires_reconciliation);
    let joined = state(&h, A);
    assert_eq!(joined.state, WorkerState::Joining);
    assert_eq!((joined.epoch, joined.term), (Some(5), Some(term)));
    assert_eq!(worker_epoch_records(h.state_dir.path(), A), vec![5]);
    assert!(h.stop().await);
}

#[tokio::test(flavor = "multi_thread")]
async fn drain_intent_is_journaled_restored_and_sent_as_desired_state() {
    let (h, fed) = start().await;
    let mut a = agent(&fed, PRINCIPAL_A);
    assert!(matches!(
        h.app.set_drain_intent("worker-unknown", true),
        Err(DrainError::UnknownWorker)
    ));
    // Before the first Register, and for a local worker.
    h.app.set_drain_intent(A, true).unwrap();
    h.app.set_drain_intent(LOCAL, true).unwrap();
    assert!(state(&h, A).draining);
    assert_eq!(h.app.worker_state(LOCAL).unwrap().0, WorkerState::Draining);
    let drains: Vec<Value> = journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == "worker_drain")
        .collect();
    assert_eq!(drains.len(), 2, "{drains:?}");

    assert!(register(&mut a, A, 1).await.accepted);
    let reply = heartbeat(&mut a, heartbeat_req(A, 1, h.app.epoch(), 1, 1, "open")).await;
    assert_eq!(reply.desired_state, wire::DESIRED_DRAINING);
    // Drained and Ready reads draining; a higher epoch keeps the intent.
    drive_ready(&h, &mut a, A, 1, 2).await;
    assert_eq!(state(&h, A).state, WorkerState::Draining);
    assert!(register(&mut a, A, 2).await.accepted);
    assert!(state(&h, A).draining);

    // Restored at open, for both.
    let h = h.restart().await;
    let term = h.app.epoch();
    assert!(state(&h, A).draining);
    assert_eq!(h.app.worker_state(LOCAL).unwrap().0, WorkerState::Draining);
    let mut a = agent(&fed, PRINCIPAL_A);
    assert!(register(&mut a, A, 2).await.accepted);
    let reply = heartbeat(&mut a, heartbeat_req(A, 2, term, 1, 1, "open")).await;
    assert_eq!(reply.desired_state, wire::DESIRED_DRAINING);

    // Undrain clears it.
    h.app.set_drain_intent(A, false).unwrap();
    h.app.set_drain_intent(LOCAL, false).unwrap();
    let reply = heartbeat(&mut a, heartbeat_req(A, 2, term, 2, 1, "open")).await;
    assert_eq!(reply.desired_state, "");
    assert!(!state(&h, A).draining);
    assert_eq!(h.app.worker_state(LOCAL).unwrap().0, WorkerState::Ready);
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_workers_are_listed_in_config_order() {
    let (h, _fed) = start().await;
    let ids: Vec<String> = registry(&h)
        .list(now_ms())
        .into_iter()
        .map(|s| s.worker_id)
        .collect();
    assert_eq!(ids, vec![A, B]);
    assert_eq!(registry(&h).term(), h.app.epoch());
    assert_eq!(h.app.federation_addr(), Some(_fed.listen));
}

#[tokio::test(flavor = "multi_thread")]
async fn cered_announces_the_registry_and_refuses_unreadable_pki() {
    let mock = cere_mock_sglang::MockSglang::start(MockConfig::default())
        .await
        .unwrap();
    let state_dir = support::StateDir::new();
    let fed = Federation::new();
    let base = support::config_text(
        &mock.base_url(),
        state_dir.path(),
        support::free_port(),
        &[support::FEDERATION_BODY_TWEAK],
    );
    let text = format!("{base}\n{}", fed.toml(TIMINGS, &[(A, PRINCIPAL_A)]));
    let config = ConfigFile::new(&text);
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    let mut a = agent(&fed, PRINCIPAL_A);
    let reply = register(&mut a, A, 1).await;
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(reply.coordinator_term, cered.epoch);
    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(20)).await;
    assert_eq!(code, Some(0), "{stderr}");
    let line = format!(
        "cered federation listening on {} cluster={FED_CLUSTER} term={}",
        fed.listen, cered.epoch
    );
    assert!(stderr.contains(&line), "{stderr}");

    // An unreadable certificate: refused at startup, exit 2.
    std::fs::remove_file(&fed.ca).unwrap();
    let state_dir = support::StateDir::new();
    let base = support::config_text(
        &mock.base_url(),
        state_dir.path(),
        support::free_port(),
        &[support::FEDERATION_BODY_TWEAK],
    );
    let config = ConfigFile::new(&format!(
        "{base}\n{}",
        fed.toml(TIMINGS, &[(A, PRINCIPAL_A)])
    ));
    let (code, stderr) = Cered::spawn(&config.path, &[]).await.err().unwrap();
    assert_eq!(code, Some(2), "{stderr}");
    assert!(stderr.contains("federation PKI:"), "{stderr}");
}
