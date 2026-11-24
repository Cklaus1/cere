//! Two-node fault matrix on one host (m35 design note P9; build-loop-m35 design note track MATRIX-A): the real
//! `cered` and the real `cere-worker`, each in front of its own `cere_mock_sglang`, with an rcgen cluster PKI
//! (`support::pki`) and shrunk Q7 timings. Both directions of the federation link run through a loopback relay
//! (`support::relay`) that a test black-holes to partition the nodes. `cere-rpc-probe` makes the forged and
//! duplicate calls as the coordinator principal; the stream tests hold `StreamWork` streams with an in-test
//! client. The coordinator's local worker never turns ready (its mock serves another model), so every client
//! request goes to the remote worker.
//!
//! One test per `docs/ADVERSARIAL_TEST_PLAN.md` row the matrix covers (:15 to :22; tracks MATRIX-A and MATRIX-B)
//! plus the happy path. The coordinator-restart test SIGKILLs `cered` and starts it again on the same config and
//! state dir; in-flight client work runs on a backend that generates until it is aborted.
mod support;

use std::time::{Duration, Instant};

use cere_mock_sglang::{Fault, MockConfig};
use cere_proto::worker::v2 as pb;
use cere_runtime::fed::wire;
use pb::worker_inference_service_client::WorkerInferenceServiceClient;
use pb::worker_service_client::WorkerServiceClient;
use serde_json::{json, Value};
use support::fed_world::*;
use support::relay::Block;
use support::*;

/// P7/Q11 over the real binaries: the agent registers, passes reconciliation and the smoke and turns Ready; a
/// client chat through `cered` is served by the remote worker (the agent's backend sees it, the local one does
/// not), and the `dispatched` record names node `remote` and the relay endpoint the coordinator dials.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_chat_is_served_by_the_remote_worker() {
    let w = World::start(Options::default()).await;
    let rid = w.chat(w.agent.epoch).await;
    assert_eq!(w.backend_requests(&rid), 1);
    assert_eq!(w.local.dispatched_rids(), Vec::<String>::new());
    assert_eq!(
        w.agent
            .count(&format!("event=accept op=a/{rid} fresh=true")),
        1
    );

    let records = journal(w.cered_state.path());
    let dispatched: Vec<&Value> = records
        .iter()
        .filter(|r| r["type"] == "dispatched" && r["rid"] == rid.as_str())
        .collect();
    assert_eq!(dispatched.len(), 1, "{records:?}");
    let record = dispatched[0];
    assert_eq!(record["worker"], A, "{record}");
    assert_eq!(record["worker_epoch"], w.agent.epoch, "{record}");
    assert_eq!(record["node"], "remote", "{record}");
    assert_eq!(
        record["endpoint"],
        format!("https://{}", w.to_agent.addr()),
        "{record}"
    );
    assert!(record.get("base_url").is_none(), "{record}");
}

/// ADVERSARIAL :16: a SubmitWork repeated with the same operation returns the original receipt (same ticket) and
/// reaches the backend once; a CancelWork repeated with the same operation returns its original receipt, and a
/// second cancel operation of the same target finds it terminal: one `cancel_requested`, one backend abort.
#[tokio::test(flavor = "multi_thread")]
async fn duplicate_submit_and_cancel_have_one_side_effect() {
    let w = World::start(Options {
        backend: MockConfig {
            server_info: backend_info(1, 4),
            // 20 ms per token: a long budget keeps an operation running while it is cancelled.
            token_latency: Duration::from_millis(20),
            natural_output_tokens: 100_000,
            ..MockConfig::default()
        },
        ..Options::default()
    })
    .await;
    let seen = w.until_ready().await;
    let g = seen.generation;

    let one = rid(1);
    let lines = w
        .submit(
            seen,
            g,
            "op-1",
            &one,
            &["--max-tokens", "1", "--repeat", "3"],
        )
        .await;
    assert_eq!(lines.len(), 3, "{lines:?}");
    let ticket = accepted_ticket(&lines[0], "submit");
    for line in &lines[1..] {
        assert_eq!(accepted_ticket(line, "submit"), ticket, "{lines:?}");
    }
    w.agent
        .until_event(&format!("event=terminal op={SCOPE}/op-1 "), short_bound())
        .await;
    assert_eq!(w.backend_requests(&one), 1);
    assert_eq!(w.agent.count(&format!("op={SCOPE}/op-1 fresh=true")), 1);
    assert_eq!(w.agent.count(&format!("op={SCOPE}/op-1 fresh=false")), 2);

    let two = rid(2);
    let lines = w
        .submit(seen, g, "op-2", &two, &["--max-tokens", "5000"])
        .await;
    let target = accepted_ticket(&lines[0], "submit");
    until("op-2 reaches the backend", short_bound(), || {
        (w.backend().in_flight() == 1 && w.backend_requests(&two) == 1).then_some(())
    })
    .await;
    let cancel_op = format!("{two}/cancel/1");
    let first = w.cancel(seen, g, &cancel_op, &two, &target).await;
    let receipt = accepted_ticket(&first, "cancel");
    let again = w.cancel(seen, g, &cancel_op, &two, &target).await;
    assert_eq!(accepted_ticket(&again, "cancel"), receipt, "{again}");
    let other = w
        .cancel(seen, g, &format!("{two}/cancel/2"), &two, &target)
        .await;
    accepted_ticket(&other, "cancel");
    until("the backend ends op-2", short_bound(), || {
        (w.backend().in_flight() == 0).then_some(())
    })
    .await;
    assert_eq!(
        w.backend_aborts(&two),
        1,
        "{:?}",
        w.backend().abort_requests()
    );
    assert_eq!(
        w.agent.count(&format!(
            "event=terminal op={SCOPE}/op-2 reason=cancel_requested"
        )),
        1,
        "{}",
        w.agent.events()
    );
    assert_eq!(
        w.agent.count(&format!("op={SCOPE}/{cancel_op} fresh=true")),
        1
    );
    assert_eq!(
        w.agent
            .count(&format!("op={SCOPE}/{cancel_op} fresh=false")),
        1
    );
    assert_eq!(w.backend_requests(&two), 1);
}

/// ADVERSARIAL :17: drained by the operator route, then partitioned by the relay: the worker goes Lost and the
/// journal still holds the drain intent. After the heal and a new agent epoch it reads `draining` (no smoke, no
/// grant); undrain runs the smoke and it is Ready again and serves.
#[tokio::test(flavor = "multi_thread")]
async fn a_partition_while_draining_keeps_the_drain_intent() {
    let mut w = World::start(Options::default()).await;
    let seen = w.until_ready().await;
    assert_eq!(
        w.operate("drain").await,
        json!({"worker_id": A, "draining": true})
    );
    w.until_entry("worker-a draining", short_bound(), |e| {
        e["state"] == "draining"
    })
    .await;

    w.to_coordinator.block(Block::Refuse);
    w.to_agent.block(Block::Hold);
    let entry = w
        .until_entry("worker-a lost", Duration::from_millis(LOST_MS * 3), |e| {
            e["state"] == "lost"
        })
        .await;
    assert_eq!(entry["ready"], false, "{entry}");
    let drains: Vec<Value> = journal(w.cered_state.path())
        .into_iter()
        .filter(|r| r["type"] == "worker_drain" && r["worker"] == A)
        .collect();
    assert_eq!(drains.len(), 1, "{drains:?}");
    assert_eq!(drains[0]["draining"], true, "{drains:?}");

    let smokes = w.backend().dispatched_rids().len();
    w.to_coordinator.heal();
    w.to_agent.heal();
    w.restart_agent().await;
    let epoch = w.agent.epoch;
    assert!(epoch > seen.epoch);
    let entry = w
        .until_entry("worker-a draining at the new epoch", ready_bound(), |e| {
            e["state"] == "draining" && e["epoch"] == epoch
        })
        .await;
    assert_eq!(entry["ready"], false, "{entry}");
    assert_eq!(w.backend().dispatched_rids().len(), smokes);

    assert_eq!(
        w.operate("undrain").await,
        json!({"worker_id": A, "draining": false})
    );
    let ready = w.until_ready().await;
    assert_eq!(ready.epoch, epoch);
    assert!(
        w.backend().dispatched_rids().len() > smokes,
        "undrain runs the smoke"
    );
    w.chat(epoch).await;
}

/// ADVERSARIAL :19: the backend restarts (a new instance id) under a live agent: the agent bumps its model
/// generation, the coordinator re-runs Joining (the new backend serves a new smoke) and the worker is Ready at
/// the same epoch and the new generation; the old generation's ticket can neither cancel nor resubmit anything
/// (`stale_generation`, or `ticket_mismatch` under the new generation) and never reaches the new backend.
#[tokio::test(flavor = "multi_thread")]
async fn a_backend_restart_makes_old_generation_tickets_inert() {
    let mut w = World::start(Options::default()).await;
    let seen = w.until_ready().await;
    let old = seen.generation;
    let one = rid(1);
    let lines = w
        .submit(seen, old, "op-1", &one, &["--max-tokens", "1"])
        .await;
    let ticket = accepted_ticket(&lines[0], "submit");
    w.agent
        .until_event(&format!("event=terminal op={SCOPE}/op-1 "), short_bound())
        .await;

    w.restart_backend(backend_info(2, 4)).await;
    let new = old + 1;
    w.agent
        .until_event(
            &format!("event=generation generation={new} "),
            short_bound(),
        )
        .await;
    let entry = w
        .until_entry("worker-a ready at the new generation", ready_bound(), |e| {
            e["state"] == "ready" && e["model_generation"] == new
        })
        .await;
    assert_eq!(entry["epoch"], seen.epoch, "{entry}");
    assert_eq!(entry["smoke"]["state"], "passed", "{entry}");
    assert!(
        !w.backend().dispatched_rids().is_empty(),
        "the new backend served the new smoke"
    );

    let cancel_op = format!("{one}/cancel/1");
    let line = w.cancel(seen, old, &cancel_op, &one, &ticket).await;
    assert_prompt_refusal(&line, "cancel", "FAILED_PRECONDITION", "stale_generation");
    let line = w.cancel(seen, new, &cancel_op, &one, &ticket).await;
    assert_prompt_refusal(&line, "cancel", "FAILED_PRECONDITION", "ticket_mismatch");
    let lines = w
        .submit(seen, old, "op-1", &one, &["--max-tokens", "1"])
        .await;
    assert_prompt_refusal(
        &lines[0],
        "submit",
        "FAILED_PRECONDITION",
        "stale_generation",
    );
    let two = rid(2);
    let lines = w
        .submit(seen, old, "op-2", &two, &["--max-tokens", "1"])
        .await;
    assert_prompt_refusal(
        &lines[0],
        "submit",
        "FAILED_PRECONDITION",
        "stale_generation",
    );
    assert_eq!(w.backend_requests(&one) + w.backend_requests(&two), 0);
    assert_eq!(w.backend_aborts(&one), 0);
    assert_eq!(w.agent.count(&format!("op={SCOPE}/{cancel_op} ")), 0);

    w.chat(seen.epoch).await;
}

/// ADVERSARIAL :21: a leaf from a foreign CA fails the handshake on both services; a worker leaf calling the
/// agent's WorkerService/WorkerInferenceService, registering or heartbeating another worker id, an unbound
/// principal claiming `A` and the coordinator principal calling the registry are `PERMISSION_DENIED` before any
/// state: the coordinator journal, the registry view, the agent's operations and the backend are unchanged.
#[tokio::test(flavor = "multi_thread")]
async fn spoofed_principals_and_worker_ids_are_denied_before_admission() {
    let w = World::start(Options::default()).await;
    let seen = w.until_ready().await;
    w.settled().await;
    let journal_before = journal(w.cered_state.path());
    let accepts = w.agent.count("event=accept ");
    let backend = w.backend().dispatched_rids();

    let foreign = support::pki::TestPki::new();
    let foreign_coordinator = foreign.leaf(FED_COORDINATOR).write(w.fed.dir.path());
    let foreign_worker = foreign.leaf(PRINCIPAL_A).write(w.fed.dir.path());
    let unbound = w.fed.pki.leaf("cere-worker-z").write(w.fed.dir.path());

    let mut describe = vec!["describe".to_owned()];
    describe.extend(authority_flags(seen, seen.generation));
    let line = &w.probe_agent(&foreign_coordinator, describe.clone()).await[0];
    assert_prompt_refusal(line, "describe", "TRANSPORT", "handshake");
    let line = &w.probe_agent(&w.worker_pem, describe).await[0];
    assert_prompt_refusal(line, "describe", "PERMISSION_DENIED", "permission_denied");
    let mut submit = vec!["submit".to_owned()];
    submit.extend(authority_flags(seen, seen.generation));
    submit.extend(args(&[
        "--scope",
        SCOPE,
        "--op-id",
        "op-1",
        "--request-id",
        &rid(1),
    ]));
    let line = &w.probe_agent(&w.worker_pem, submit).await[0];
    assert_prompt_refusal(line, "submit", "PERMISSION_DENIED", "permission_denied");

    let register = |worker: &str| {
        args(&[
            "register",
            "--cluster-id",
            FED_CLUSTER,
            "--worker-id",
            worker,
            "--epoch",
            "1",
        ])
    };
    let line = &w.probe_registry(&foreign_worker, register(A)).await[0];
    assert_prompt_refusal(line, "register", "TRANSPORT", "handshake");
    for (pem, worker) in [
        (&w.worker_pem, "worker-b"),
        (&unbound, A),
        (&w.coordinator_pem, A),
    ] {
        let line = &w.probe_registry(pem, register(worker)).await[0];
        assert_prompt_refusal(line, "register", "PERMISSION_DENIED", "permission_denied");
    }
    let heartbeat = args(&[
        "heartbeat",
        "--cluster-id",
        FED_CLUSTER,
        "--worker-id",
        "worker-b",
        "--epoch",
        "1",
        "--sequence",
        "1",
    ]);
    let line = &w.probe_registry(&w.worker_pem, heartbeat).await[0];
    assert_prompt_refusal(line, "heartbeat", "PERMISSION_DENIED", "permission_denied");

    assert_eq!(journal(w.cered_state.path()), journal_before);
    let entry = w.entry().await;
    assert_eq!(entry["state"], "ready", "{entry}");
    assert_eq!(entry["epoch"], seen.epoch, "{entry}");
    assert_eq!(entry["term"], seen.term, "{entry}");
    let body: Value = w
        .http
        .get(w.cered.url("/readyz"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(body["workers"].get("worker-b").is_none(), "{body}");
    assert_eq!(w.agent.count("event=accept "), accepts);
    assert_eq!(w.backend().dispatched_rids(), backend);
}

/// ADVERSARIAL :22 on a backend of one running request with a 4 KiB per-operation output buffer: a SubmitWork
/// beyond the live-op bound is `RESOURCE_EXHAUSTED queue_full` at once and never reaches the backend; an
/// operation nobody drains stops reading the backend and ends `client_backpressure` after `output_stall_ms`
/// (the backend generation is aborted, nothing keeps generating), which frees the bound. `StreamWork` streams
/// beyond the stream pool (live-op bound + 1) are `RESOURCE_EXHAUSTED quota_exhausted` at once; dropping the held
/// streams frees it.
#[tokio::test(flavor = "multi_thread")]
async fn full_rpc_and_output_queues_refuse_visibly_and_stay_bounded() {
    const STALL_MS: u64 = 1000;
    let w = World::start(Options {
        backend: MockConfig {
            server_info: backend_info(1, 1),
            natural_output_tokens: 100_000,
            ..MockConfig::default()
        },
        output_buffer_bytes: 4096,
        output_stall_ms: STALL_MS,
    })
    .await;
    let seen = w.until_ready().await;
    w.settled().await;
    let channel = w.narrow_channel();
    let mut inference = WorkerInferenceServiceClient::new(channel.clone());

    // Output queue: op-1 holds the only live-op slot. The backend holds it in transit (live, no output) until op-2
    // is refused, so op-2 meets the full queue however long its probe takes to start (fed-load-flake design note
    // F2); then op-1 is never drained, so it holds the slot until its stall bound.
    w.backend().hold_start(true);
    let one = rid(1);
    let receipt = inference
        .submit_work(chat_submit(seen, "op-1", &one, 50_000, true))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted, "{receipt:?}");
    until("op-1 reaches the backend", short_bound(), || {
        (w.backend_requests(&one) == 1).then_some(())
    })
    .await;
    let two = rid(2);
    let lines = w
        .submit(seen, seen.generation, "op-2", &two, &["--max-tokens", "1"])
        .await;
    assert_prompt_refusal(&lines[0], "submit", "RESOURCE_EXHAUSTED", "queue_full");
    assert_eq!(w.backend_requests(&two), 0);
    w.backend().hold_start(false);
    w.agent
        .until_event(
            &format!("event=terminal op={SCOPE}/op-1 reason=client_backpressure"),
            Duration::from_millis(STALL_MS) + short_bound(),
        )
        .await;
    until("the backend stops generating op-1", short_bound(), || {
        (w.backend().in_flight() == 0 && w.backend().load() == 0).then_some(())
    })
    .await;
    assert_eq!(w.backend_aborts(&one), 1);
    assert_eq!(w.backend().max_in_flight(), 1);
    w.settled().await;
    // The refusal left nothing behind: the same operation is accepted fresh now.
    let lines = w
        .submit(seen, seen.generation, "op-2", &two, &["--max-tokens", "1"])
        .await;
    accepted_ticket(&lines[0], "submit");
    assert_eq!(w.agent.count(&format!("op={SCOPE}/op-2 fresh=true")), 1);
    w.agent
        .until_event(&format!("event=terminal op={SCOPE}/op-2 "), short_bound())
        .await;
    w.settled().await;

    // Stream pool: two streams (the bound + 1) stay open unread on ops that ended and were cleaned up.
    let mut control = WorkerServiceClient::new(channel);
    let mut held = Vec::new();
    for n in 3..=4u64 {
        let op = format!("op-{n}");
        let r = rid(n);
        let receipt = inference
            .submit_work(chat_submit(seen, &op, &r, 50_000, true))
            .await
            .unwrap()
            .into_inner();
        assert!(receipt.accepted, "{receipt:?}");
        let stream = control
            .stream_work(pb::WorkStreamRequest {
                authority: Some(op_authority(seen, &op, [0; 32])),
                request_id: r.clone(),
                attempt: 1,
                completion_ticket: receipt.completion_ticket,
                after_sequence: 0,
            })
            .await
            .unwrap()
            .into_inner();
        held.push(stream);
        w.agent
            .until_event(
                &format!("event=terminal op={SCOPE}/{op} reason=client_backpressure"),
                Duration::from_millis(STALL_MS) + short_bound(),
            )
            .await;
        w.settled().await;
    }
    let five = rid(5);
    let stream_five = ["--max-tokens", "1", "--stream"];
    let lines = w
        .submit(seen, seen.generation, "op-5", &five, &stream_five)
        .await;
    assert_eq!(lines.len(), 2, "{lines:?}");
    let ticket = accepted_ticket(&lines[0], "submit");
    assert_prompt_refusal(&lines[1], "stream", "RESOURCE_EXHAUSTED", "quota_exhausted");

    drop(held);
    let deadline = Instant::now() + short_bound();
    loop {
        let lines = w
            .submit(seen, seen.generation, "op-5", &five, &stream_five)
            .await;
        assert_eq!(accepted_ticket(&lines[0], "submit"), ticket);
        if lines[1]["code"] == "OK" {
            assert_eq!(lines[1]["accepted"], true, "{lines:?}");
            break;
        }
        assert_outcome(&lines[1], "stream", "RESOURCE_EXHAUSTED", "quota_exhausted");
        assert!(Instant::now() < deadline, "the stream pool stays full");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(w.backend().max_in_flight(), 1);
}

/// The agent's backend for a client stream that runs until its work is aborted: one token per 20 ms, far more
/// natural output than [`LONG_TOKENS`].
fn generating() -> MockConfig {
    MockConfig {
        server_info: backend_info(1, 4),
        token_latency: Duration::from_millis(20),
        natural_output_tokens: 100_000,
        ..MockConfig::default()
    }
}

/// A client stream's budget on [`generating`]: it outlives every test.
const LONG_TOKENS: u64 = 30_000;

/// The admission bound of scope `a` (`[admission] max_backend_inflight` and `per_scope_inflight` of the live
/// profile), below `A`'s cap of 4: at most this many client requests run on `A` at once.
const ADMISSION_CAP: usize = 2;

/// ADVERSARIAL :15: the network between the nodes is cut while a client streams from the remote worker. Once its
/// lease expires the agent admits nothing (`UNAVAILABLE lease_expired` at once, nothing reaches the backend) while
/// the in-flight work stays allocated (nonterminal, still generating); the client sees the terminal `worker_lost`
/// within lost_after + keepalive of the cut, not a hang. The agent ends the work on its own lost bound (a new
/// incarnation, `worker_restart`) and cleans it up, which the coordinator cannot hear: it keeps the slot. After the
/// heal the agent registers its new epoch and reports the work `cleanup_acknowledged`; only then is the slot released
/// (one `orphan_released`), and the worker is Ready again and serves.
#[tokio::test(flavor = "multi_thread")]
async fn a_partition_during_decode_stops_admission_and_holds_the_slot_until_cleanup() {
    let w = World::start(Options {
        backend: generating(),
        ..Options::default()
    })
    .await;
    let seen = w.until_ready().await;
    w.settled().await;
    let (client, mut sse, mut events) = w.stream_chat(LONG_TOKENS).await;
    assert_eq!(w.inflight().await, 1);
    // Connected before the cut: the calls between the lease expiry and the agent's own lost bound (`LOST_MS -
    // LEASE_MS` later) are single RPCs, not a probe process that a loaded host can start after that window closed
    // (fed-load-flake design note F3).
    let channel = w.connected_channel().await;
    let mut inference = WorkerInferenceServiceClient::new(channel.clone());

    let cut = Instant::now();
    w.to_coordinator.block(Block::Refuse);
    w.to_agent.block(Block::Hold);
    w.agent
        .until_event(
            "event=lease state=expired",
            Duration::from_millis(LEASE_MS) + short_bound(),
        )
        .await;
    let one = rid(1);
    let asked = Instant::now();
    let refused = inference
        .submit_work(chat_submit(seen, "op-1", &one, 1, false))
        .await
        .expect_err("a submit after the lease expired is refused");
    assert_eq!(
        (refused.code(), wire::reason(&refused)),
        (tonic::Code::Unavailable, wire::LEASE_EXPIRED),
        "{refused:?}"
    );
    assert!(
        asked.elapsed() < Duration::from_millis(PROMPT_REFUSAL_MS),
        "a refusal must not wait: {:?}",
        asked.elapsed()
    );
    assert_eq!(w.backend_requests(&one), 0);
    let work = operation_on(channel, seen, "a", &client).await;
    assert_eq!(
        (work.phase.as_str(), work.terminal),
        (wire::PHASE_ACCEPTED, false),
        "{work:?}"
    );
    assert_eq!(w.backend().in_flight(), 1);

    let keepalive = Duration::from_millis(HEARTBEAT_MS + RPC_TIMEOUT_MS);
    let bound = Duration::from_millis(LOST_MS) + keepalive;
    let rest = tokio::time::timeout(bound.saturating_sub(cut.elapsed()), sse.rest())
        .await
        .expect("the client saw no terminal within lost_after + keepalive");
    events.extend(rest);
    let terminal = check_stream(&events);
    assert_eq!(terminal["error"]["code"], "worker_lost", "{terminal}");
    let status = w.status(&client).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_lost", "{status}");

    w.agent
        .until_event(
            &format!(
                "event=terminal op=a/{client} reason={}",
                wire::WORKER_RESTART
            ),
            short_bound(),
        )
        .await;
    w.agent
        .until_line(&format!("event=cleanup op=a/{client}"), short_bound())
        .await;
    until("the backend stops generating", short_bound(), || {
        (w.backend().in_flight() == 0).then_some(())
    })
    .await;
    assert_eq!(
        w.inflight().await,
        1,
        "the slot is held while unacknowledged"
    );
    assert_eq!(w.records("orphan_released", &client), Vec::<Value>::new());

    w.to_coordinator.heal();
    w.to_agent.heal();
    let released = w.until_released(&client, ready_bound()).await;
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    let entry = w
        .until_entry("worker-a ready at a new epoch", ready_bound(), |e| {
            e["state"] == "ready" && e["epoch"].as_u64() > Some(seen.epoch)
        })
        .await;
    let now = Seen {
        epoch: entry["epoch"].as_u64().unwrap(),
        term: entry["term"].as_u64().unwrap(),
        generation: entry["model_generation"].as_u64().unwrap(),
    };
    let work = w.operation(now, "a", &client).await;
    assert!(work.terminal && work.cleanup_acknowledged, "{work:?}");
    assert_eq!(work.reason_code, wire::WORKER_RESTART, "{work:?}");
    w.until_inflight(0, short_bound()).await;
    w.chat(now.epoch).await;
    assert_eq!(w.records("orphan_released", &client).len(), 1);
    assert_eq!(w.backend_requests(&client), 1);
}

/// ADVERSARIAL :18: `cered` is killed while a client streams from the remote worker and started again on the same
/// config and state dir: it journals a higher `Epoch`, its term. The agent fences (`coordinator_restart`), ends the
/// older term's work `coordinator_restart` and aborts it on the backend; an old-term command is refused
/// `stale_term`. No output of the old work reaches a client: the request is journaled failed `worker_restart` and
/// its events cannot be resumed (`cursor_expired`). The old work is terminal before the agent admits anything in
/// the new term (its first admission is the new term's smoke, which needs the reconciled grant), and the worker is
/// Ready at the new term; the old request's slot is released on its `cleanup_acknowledged` (one `orphan_released`)
/// and a new request is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_coordinator_restart_fences_old_term_work_and_reconciles_before_serving() {
    let mut w = World::start(Options {
        backend: generating(),
        ..Options::default()
    })
    .await;
    let seen = w.until_ready().await;
    assert_eq!(seen.term, w.cered.epoch);
    w.settled().await;
    let (client, sse, _) = w.stream_chat(LONG_TOKENS).await;

    w.restart_cered().await;
    drop(sse);
    let term = w.cered.epoch;
    let epochs = journal(w.cered_state.path())
        .into_iter()
        .filter(|r| r["type"] == "epoch")
        .filter_map(|r| r["epoch"].as_u64())
        .max();
    assert_eq!(epochs, Some(term));
    let status = w.status(&client).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], wire::WORKER_RESTART, "{status}");
    let r = w
        .http
        .get(
            w.cered
                .url(&format!("/v1/cere/requests/{client}/events?after=0")),
        )
        .bearer_auth(TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 410);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "cursor_expired", "{body}");

    let fence = format!(
        "event=fence reason={} term={term} previous_term={}",
        wire::COORDINATOR_RESTART,
        seen.term
    );
    w.agent.until_event(&fence, short_bound()).await;
    let ended = format!(
        "event=terminal op=a/{client} reason={}",
        wire::COORDINATOR_RESTART
    );
    w.agent.until_event(&ended, short_bound()).await;
    let one = rid(1);
    let lines = w
        .submit(seen, seen.generation, "op-1", &one, &["--max-tokens", "1"])
        .await;
    assert_prompt_refusal(&lines[0], "submit", "FAILED_PRECONDITION", wire::STALE_TERM);
    assert_eq!(w.backend_requests(&one), 0);

    let ready = w.until_ready().await;
    assert_eq!((ready.epoch, ready.term), (seen.epoch, term));
    let events = w.agent.events();
    let lines: Vec<&str> = events.lines().collect();
    let after = |from: usize, needle: &str| {
        lines[from..]
            .iter()
            .position(|l| l.contains(needle))
            .map(|i| i + from)
    };
    let fenced = after(0, &fence).unwrap();
    let ended_at = after(fenced, &ended).unwrap();
    let admitted = after(fenced, " fresh=true").expect("an admission in the new term");
    assert!(
        lines[admitted].contains(&format!(
            "event=accept op={}/",
            cere_runtime::readiness::REMOTE_SMOKE_SCOPE
        )),
        "{events}"
    );
    assert!(ended_at < admitted, "{events}");

    until("the backend stops generating", short_bound(), || {
        (w.backend().in_flight() == 0 && w.backend_aborts(&client) == 1).then_some(())
    })
    .await;
    let now = Seen { term, ..ready };
    let work = w.operation(now, "a", &client).await;
    assert!(work.terminal, "{work:?}");
    assert_eq!(work.reason_code, wire::COORDINATOR_RESTART, "{work:?}");
    assert_eq!(work.coordinator_term, seen.term, "{work:?}");
    let released = w.until_released(&client, short_bound()).await;
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    w.chat(seen.epoch).await;
    assert_eq!(w.backend_requests(&client), 1);
}

/// ADVERSARIAL :20: duplicate cancel and completion packets in either order end each operation once and free its
/// slot once. (a) Completion first: the client's cancel finds the request `completed` and the coordinator sends no
/// CancelWork; a CancelWork sent twice after the terminal (one operation) gets one receipt and changes nothing (one
/// natural terminal, no abort, nothing held). (b) Cancel first, on a backend that ignores the abort and completes
/// afterwards: one terminal `cancel_requested`, the late completion is no second one; the coordinator's CancelWork
/// repeated returns its receipt, a second client cancel finds the request `cancelled` and sends nothing, and the
/// slot is released once, on the cleanup. Then cap + 1 concurrent requests run at most cap at once and all are
/// served: nothing was freed twice.
#[tokio::test(flavor = "multi_thread")]
async fn duplicate_cancel_and_completion_in_either_order_end_once_and_free_once() {
    let w = World::start(Options {
        backend: generating(),
        ..Options::default()
    })
    .await;
    let seen = w.until_ready().await;
    let g = seen.generation;
    w.settled().await;

    // (a) Completion, then cancel packets.
    let done = w.chat(seen.epoch).await;
    let (code, body) = w.client_cancel(&done).await;
    assert_eq!((code, &body["state"]), (200, &json!("completed")), "{body}");
    let cancel_op = format!("{done}/cancel/1");
    assert_eq!(w.agent.count(&format!("op=a/{cancel_op} ")), 0);
    let work = w.operation(seen, "a", &done).await;
    assert!(work.terminal, "{work:?}");
    assert_eq!(work.reason_code, "", "{work:?}");
    let first = w
        .cancel_in("a", seen, g, &cancel_op, &done, &work.completion_ticket)
        .await;
    let receipt = accepted_ticket(&first, "cancel");
    let again = w
        .cancel_in("a", seen, g, &cancel_op, &done, &work.completion_ticket)
        .await;
    assert_eq!(accepted_ticket(&again, "cancel"), receipt, "{again}");
    assert_eq!(w.agent.count(&format!("op=a/{cancel_op} fresh=true")), 1);
    assert_eq!(w.agent.count(&format!("op=a/{cancel_op} fresh=false")), 1);
    assert_eq!(w.agent.count(&format!("event=terminal op=a/{done} ")), 1);
    let work = w.operation(seen, "a", &done).await;
    assert!(work.terminal, "{work:?}");
    assert_eq!(work.reason_code, "", "{work:?}");
    assert_eq!(w.backend_aborts(&done), 0);
    let terminals = w.records("terminal", &done);
    assert_eq!(terminals.len(), 1, "{terminals:?}");
    assert_eq!(terminals[0]["state"], "completed", "{terminals:?}");
    assert_eq!(w.records("orphan_released", &done), Vec::<Value>::new());
    w.until_inflight(0, short_bound()).await;

    // (b) Cancel, then the completion.
    w.backend().set_fault(Fault::IgnoreAbort);
    let late_tokens = 100;
    let (client, mut sse, mut events) = w.stream_chat(late_tokens).await;
    w.backend().set_fault(Fault::None);
    let (code, body) = w.client_cancel(&client).await;
    assert_eq!(code, 202, "{body}");
    let rest = tokio::time::timeout(short_bound(), sse.rest())
        .await
        .expect("the cancelled stream ended");
    events.extend(rest);
    let terminal = check_stream(&events);
    assert_eq!(terminal["error"]["code"], "request_cancelled", "{terminal}");
    let cancelled = format!(
        "event=terminal op=a/{client} reason={}",
        wire::CANCEL_REQUESTED
    );
    w.agent.until_event(&cancelled, short_bound()).await;
    let cleanup = format!("event=cleanup op=a/{client}");
    w.agent
        .until_line(
            &cleanup,
            Duration::from_millis(20 * late_tokens) + short_bound(),
        )
        .await;
    let agent_events = w.agent.events();
    assert!(
        w.agent.position(&cancelled).unwrap() < w.agent.position(&cleanup).unwrap(),
        "{agent_events}"
    );
    assert_eq!(w.backend().in_flight(), 0);
    assert_eq!(w.backend_aborts(&client), 1);
    assert_eq!(w.agent.count(&format!("event=terminal op=a/{client} ")), 1);
    let work = w.operation(seen, "a", &client).await;
    assert!(work.terminal && work.cleanup_acknowledged, "{work:?}");
    assert_eq!(work.reason_code, wire::CANCEL_REQUESTED, "{work:?}");

    let cancel_op = format!("{client}/cancel/1");
    let original = w.operation(seen, "a", &cancel_op).await;
    let duplicate = w
        .cancel_in("a", seen, g, &cancel_op, &client, &work.completion_ticket)
        .await;
    assert_eq!(
        accepted_ticket(&duplicate, "cancel"),
        original.completion_ticket,
        "{duplicate}"
    );
    let (code, body) = w.client_cancel(&client).await;
    assert_eq!((code, &body["state"]), (200, &json!("cancelled")), "{body}");
    assert_eq!(w.agent.count(&format!("op=a/{cancel_op} fresh=true")), 1);
    assert_eq!(w.agent.count(&format!("op=a/{cancel_op} fresh=false")), 1);
    assert_eq!(
        w.agent
            .count(&format!("event=accept op=a/{client}/cancel/")),
        2
    );
    assert_eq!(w.agent.count(&format!("event=terminal op=a/{client} ")), 1);
    assert_eq!(w.backend_aborts(&client), 1);
    let released = w.until_released(&client, short_bound()).await;
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    let terminals = w.records("terminal", &client);
    assert_eq!(terminals.len(), 1, "{terminals:?}");
    assert_eq!(terminals[0]["state"], "cancelled", "{terminals:?}");
    w.until_inflight(0, short_bound()).await;

    // No double free: cap + 1 concurrent requests, at most cap at once on the backend, every one served. A request
    // alone lasts 8 tokens, so on a loaded host the burst may reach the backend one at a time: the backend holds
    // each generation at its finish until cap of them are there at once and the last one waits `queued` in
    // admission, never dispatched (a slot freed twice would have dispatched it; one leaked would hold one fewer).
    assert!(w.backend().max_in_flight() <= 1);
    let before = w.admitted();
    w.backend().hold_finish(true);
    let burst = futures_util::future::join_all((0..=ADMISSION_CAP).map(|_| w.chat(seen.epoch)));
    let gate = async {
        let queued = w
            .until_held_and_queued(&before, ADMISSION_CAP + 1, ADMISSION_CAP, short_bound())
            .await;
        assert_eq!(w.backend().in_flight(), ADMISSION_CAP);
        assert_eq!(w.inflight().await, ADMISSION_CAP as u64);
        for rid in &queued {
            assert_eq!(w.status(rid).await["state"], "queued", "{rid}");
            assert_eq!(w.backend_requests(rid), 0, "{rid}");
        }
        w.backend().hold_finish(false);
        queued
    };
    let (burst, queued) = tokio::join!(burst, gate);
    assert_eq!(burst.len(), ADMISSION_CAP + 1);
    assert_eq!(queued.len(), 1, "{queued:?}");
    assert!(burst.contains(&queued[0]), "{queued:?} {burst:?}");
    assert_eq!(w.backend_requests(&queued[0]), 1);
    assert_eq!(w.backend().max_in_flight(), ADMISSION_CAP);
    w.until_inflight(0, short_bound()).await;
    assert_eq!(w.records("orphan_released", &client).len(), 1);
}
