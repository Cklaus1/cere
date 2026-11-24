//! CORE track: request ledger (L1, L2 outcomes), durable journal crash-replay (L3), session claims (S1–S3), and
//! M3 worker epochs, dispatch records and orphan classification (Q4, Q9–Q11). A ledger opened without configured
//! workers writes pre-M3 admissions (no dispatch journal), so the M1 tests below exercise the legacy path.

use cere_runtime::api_types::{CacheReport, RequestUsage};
use cere_runtime::digest::{Digest, DigestKeys, RequestDigestInput};
use cere_runtime::journal::{
    DispatchedRecord, Node, OrphanReleaseCause, OrphanReleasedRecord, Record, Residency,
    WorkerEpochRecord, JOURNAL_FILE,
};
use cere_runtime::ledger::{
    AcceptOutcome, AcceptRequest, CancelOutcome, Completion, Ledger, LedgerError, LedgerLimits,
    LedgerStats, Orphan, OrphanCharge, OrphanWorker, Recovery, RequestState, RetainedResponse,
    SessionClaim, WorkerTarget, REASON_WORKER_EPOCH_SUPERSEDED,
};
use cere_runtime::session::{Revision, SessionError, SessionLimits};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SENTINEL_PROMPT: &str = "SENTINEL-PROMPT-7f3a91";
const SENTINEL_OUTPUT: &str = "SENTINEL-OUTPUT-c04e22";

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cere-core-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir.join("state"))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
    }
}

fn limits() -> LedgerLimits {
    LedgerLimits {
        max_records: 8,
        terminal_retention_ms: 1_000,
        max_retained_bytes: 64,
        compaction_slack: 1_024,
        sessions: SessionLimits {
            max_sessions_per_scope: 2,
            session_ttl_ms: 100_000,
            max_revisions_per_session: 8,
        },
        evict_terminal_when_full: false,
    }
}

fn open_full(
    dir: &Path,
    limits: LedgerLimits,
    workers: &BTreeMap<String, WorkerTarget>,
    now: u64,
) -> (Ledger, Recovery) {
    let keys = Arc::new(DigestKeys::open(dir).unwrap());
    Ledger::open(limits, keys, dir, workers, now).unwrap()
}

fn open_with(dir: &Path, limits: LedgerLimits, now: u64) -> (Ledger, Recovery) {
    open_full(dir, limits, &BTreeMap::new(), now)
}

fn open(dir: &Path, now: u64) -> (Ledger, Recovery) {
    open_with(dir, limits(), now)
}

fn rid(n: u64) -> String {
    format!("cere-{:020}-{:020}-0123456789abcdef", 1, n)
}

fn body(text: &str) -> Value {
    json!({"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 8})
}

fn digest(ledger: &Ledger, scope: &str, request: &Value) -> Digest {
    ledger
        .digest_keys()
        .scope_key(scope)
        .request_digest(&RequestDigestInput {
            request,
            session_id: None,
            expected_revision: None,
            served_model: "m",
            deadline_ms: 1_000,
            config_hash: "cfg",
            surface: None,
        })
}

fn req(ledger: &Ledger, scope: &str, n: u64, key: Option<&str>, stream: bool) -> AcceptRequest {
    AcceptRequest {
        scope: scope.into(),
        rid: rid(n),
        key: key.map(str::to_owned),
        digest: digest(ledger, scope, &body("hello")),
        stream,
        surface: None,
        session: None,
    }
}

/// Method form so call sites can build the request from `&ledger` (two-phase borrow).
trait Admit {
    fn admit(&mut self, r: AcceptRequest, now: u64);
}

impl Admit for Ledger {
    fn admit(&mut self, r: AcceptRequest, now: u64) {
        assert_eq!(self.accept(r, now), Ok(AcceptOutcome::Accepted));
    }
}

fn run(ledger: &mut Ledger, n: u64, now: u64) {
    ledger.mark_queued(&rid(n), now).unwrap();
    ledger.mark_running(&rid(n), now).unwrap();
}

fn state(ledger: &Ledger, n: u64) -> RequestState {
    ledger.status("a", &rid(n)).unwrap().state
}

fn retained(text: &str) -> RetainedResponse {
    RetainedResponse {
        status: 200,
        body: text.as_bytes().to_vec(),
    }
}

fn journal_records(dir: &Path) -> Vec<Record> {
    let text = std::fs::read_to_string(dir.join(JOURNAL_FILE)).unwrap();
    assert!(text.ends_with('\n'));
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn l1_happy_path_transitions_and_timings() {
    let dir = TempDir::new("l1-happy");
    let (mut l, _) = open(&dir.0, 0);
    l.admit(req(&l, "a", 1, None, false), 100);
    assert_eq!(state(&l, 1), RequestState::Accepted);
    l.mark_queued(&rid(1), 100).unwrap();
    assert_eq!(state(&l, 1), RequestState::Queued);
    l.mark_running(&rid(1), 105).unwrap();
    assert_eq!(state(&l, 1), RequestState::Running);
    l.note_first_generated(&rid(1), 110).unwrap();
    l.note_first_generated(&rid(1), 111).unwrap();
    l.note_first_visible(&rid(1), 112).unwrap();
    l.add_backpressure(&rid(1), 3).unwrap();
    l.complete(&rid(1), Completion::default(), 120).unwrap();
    let s = l.status("a", &rid(1)).unwrap();
    assert_eq!(
        (s.state, s.reason.as_deref(), s.attempt, s.epoch),
        (RequestState::Completed, None, 1, 1)
    );
    let t = s.timings;
    assert_eq!(
        (
            t.queue_wait_ms,
            t.first_generated_ms,
            t.first_visible_ms,
            t.total_ms,
            t.client_backpressure_ms
        ),
        (Some(5), Some(10), Some(12), Some(20), 3)
    );
    assert!(l.in_flight().is_empty());
}

#[test]
fn l1_cancel_before_dispatch_has_no_running_phase() {
    let dir = TempDir::new("l1-cancel");
    let (mut l, _) = open(&dir.0, 0);
    l.admit(req(&l, "a", 1, None, false), 0);
    assert_eq!(
        l.request_cancel("a", &rid(1), "client_cancel", 5),
        Ok(CancelOutcome::Cancelled)
    );
    l.admit(req(&l, "a", 2, None, false), 0);
    l.mark_queued(&rid(2), 0).unwrap();
    assert_eq!(
        l.request_cancel("a", &rid(2), "deadline_exceeded", 7),
        Ok(CancelOutcome::Cancelled)
    );
    let s = l.status("a", &rid(2)).unwrap();
    assert_eq!(
        (s.state, s.reason.as_deref()),
        (RequestState::Cancelled, Some("deadline_exceeded"))
    );
    assert_eq!(s.timings.queue_wait_ms, Some(7));
    assert_eq!(
        l.request_cancel("a", &rid(2), "client_cancel", 8),
        Ok(CancelOutcome::AlreadyTerminal(RequestState::Cancelled))
    );
}

/// beat-sglang design note OQ4: `queue_wait_ms` is the time in `queued` less the pre-grant encode; the host stages
/// between acceptance and `mark_queued` (digest, the fsynced `admitted` record) and the encode count in the other
/// timings only. Before OQ4 it was `mark_running - accepted` (12 here), which invalidated G6 rounds on host time.
#[test]
fn oq4_queue_wait_is_the_queued_time_less_the_encode() {
    let dir = TempDir::new("oq4-queue-wait");
    let (mut l, _) = open(&dir.0, 0);
    l.admit(req(&l, "a", 1, None, false), 100);
    l.mark_queued(&rid(1), 103).unwrap();
    l.record_encode(&rid(1), 4).unwrap();
    l.mark_running(&rid(1), 112).unwrap();
    l.note_first_generated(&rid(1), 115).unwrap();
    l.complete(&rid(1), Completion::default(), 120).unwrap();
    let t = l.status("a", &rid(1)).unwrap().timings;
    assert_eq!(
        (
            t.queue_wait_ms,
            t.encode_ms,
            t.first_generated_ms,
            t.total_ms
        ),
        (Some(5), Some(4), Some(15), Some(20))
    );
    // Without an encode (every single-form or SGLang request) the wait is the queued time alone.
    l.admit(req(&l, "a", 2, None, false), 200);
    l.mark_queued(&rid(2), 209).unwrap();
    l.mark_running(&rid(2), 210).unwrap();
    let t = l.status("a", &rid(2)).unwrap().timings;
    assert_eq!((t.queue_wait_ms, t.encode_ms), (Some(1), None));
    // A request that ends while queued reports its queued time so far; an encode is refused once it left `queued`.
    l.admit(req(&l, "a", 3, None, false), 300);
    l.mark_queued(&rid(3), 302).unwrap();
    l.request_cancel("a", &rid(3), "client_cancel", 310)
        .unwrap();
    assert!(l.record_encode(&rid(3), 1).is_err());
    let t = l.status("a", &rid(3)).unwrap().timings;
    assert_eq!((t.queue_wait_ms, t.encode_ms), (Some(8), None));
}

#[test]
fn l1_running_cancel_pending_paths() {
    let dir = TempDir::new("l1-pending");
    let (mut l, _) = open(&dir.0, 0);
    for n in 1..=3 {
        l.admit(req(&l, "a", n, None, true), 0);
        run(&mut l, n, 1);
        assert_eq!(
            l.request_cancel("a", &rid(n), "client_cancel", 2),
            Ok(CancelOutcome::AbortRequired)
        );
        assert_eq!(state(&l, n), RequestState::CancelPending);
        assert_eq!(
            l.request_cancel("a", &rid(n), "client_cancel", 2),
            Ok(CancelOutcome::AlreadyPending)
        );
    }
    // Backend acked the abort → cancelled; finished naturally first → completed; never acked → failed.
    l.cancel(&rid(1), "client_cancel", 3).unwrap();
    l.complete(&rid(2), Completion::default(), 3).unwrap();
    l.fail(&rid(3), "cancel_unacknowledged", None, 3).unwrap();
    assert_eq!(
        [state(&l, 1), state(&l, 2), state(&l, 3)],
        [
            RequestState::Cancelled,
            RequestState::Completed,
            RequestState::Failed
        ]
    );
    assert_eq!(
        l.status("a", &rid(3)).unwrap().reason.as_deref(),
        Some("cancel_unacknowledged")
    );
    // Running → failed directly (backend error) is allowed.
    l.admit(req(&l, "a", 4, None, false), 0);
    run(&mut l, 4, 1);
    l.fail(&rid(4), "backend_error", None, 2).unwrap();
    assert_eq!(state(&l, 4), RequestState::Failed);
}

#[test]
fn l1_invalid_transitions_and_duplicate_terminal() {
    let dir = TempDir::new("l1-invalid");
    let (mut l, _) = open(&dir.0, 0);
    l.admit(req(&l, "a", 1, None, false), 0);
    let invalid = |from, to| Err(LedgerError::InvalidTransition { from, to });
    assert_eq!(
        l.mark_running(&rid(1), 0),
        invalid(RequestState::Accepted, RequestState::Running)
    );
    assert_eq!(
        l.complete(&rid(1), Completion::default(), 0),
        invalid(RequestState::Accepted, RequestState::Completed)
    );
    run(&mut l, 1, 0);
    assert_eq!(
        l.mark_queued(&rid(1), 0),
        invalid(RequestState::Running, RequestState::Queued)
    );
    assert_eq!(
        l.cancel(&rid(1), "x", 0),
        invalid(RequestState::Running, RequestState::Cancelled)
    );
    l.complete(&rid(1), Completion::default(), 1).unwrap();
    let dup = LedgerError::AlreadyTerminal {
        state: RequestState::Completed,
    };
    assert_eq!(
        l.complete(&rid(1), Completion::default(), 2),
        Err(dup.clone())
    );
    assert_eq!(l.fail(&rid(1), "x", None, 2), Err(dup.clone()));
    assert_eq!(l.cancel(&rid(1), "x", 2), Err(dup.clone()));
    assert_eq!(dup.code(), "already_terminal");
    // The duplicate never reached the journal: exactly one terminal on disk.
    let terminals = journal_records(&dir.0)
        .into_iter()
        .filter(|r| matches!(r, Record::Terminal(t) if t.rid == rid(1)))
        .count();
    assert_eq!(terminals, 1);
    // Unknown and cross-scope ids look the same.
    assert_eq!(l.mark_queued(&rid(9), 0), Err(LedgerError::NotFound));
    assert_eq!(
        l.request_cancel("b", &rid(1), "x", 2),
        Err(LedgerError::NotFound)
    );
    assert_eq!(l.status("b", &rid(1)), None);
    assert_eq!(
        l.accept(req(&l, "a", 1, None, false), 3),
        Err(LedgerError::DuplicateRid)
    );
}

#[test]
fn l1_record_bound_rejects_before_admission_until_retention_expires() {
    let dir = TempDir::new("l1-bound");
    let (mut l, _) = open(&dir.0, 0);
    for n in 1..=8 {
        l.admit(req(&l, "a", n, None, false), 0);
    }
    let err = l.accept(req(&l, "a", 9, None, false), 0).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (LedgerError::Overloaded, "ledger_full")
    );
    assert!(l.status("a", &rid(9)).is_none());
    // Terminal records still count until their retention passes.
    l.request_cancel("a", &rid(1), "client_cancel", 10).unwrap();
    assert_eq!(
        l.accept(req(&l, "a", 9, None, false), 1_009),
        Err(LedgerError::Overloaded)
    );
    l.admit(req(&l, "a", 9, None, false), 1_010);
    assert!(l.status("a", &rid(1)).is_none());
    assert_eq!(l.len(), 8);
}

fn evicting() -> LedgerLimits {
    LedgerLimits {
        evict_terminal_when_full: true,
        ..limits()
    }
}

fn known_rids(l: &Ledger, upto: u64) -> BTreeSet<u64> {
    (1..=upto)
        .filter(|&n| l.status("a", &rid(n)).is_some())
        .collect()
}

/// The `type` of every record the release before OH-03 (d1ae4a55) parses. Its `Journal::open` refuses any other
/// line, so a journal written after evictions stays within these to keep a rollback startable.
const BASE_RECORD_TYPES: [&str; 12] = [
    "epoch",
    "worker_epoch",
    "admitted",
    "dispatched",
    "terminal",
    "session",
    "session_owner",
    "key_expired",
    "orphan_released",
    "orphan_held",
    "worker_drain",
    "worker_residency",
];

fn assert_base_record_types(dir: &Path) {
    for line in journal_lines(dir) {
        let record: Value = serde_json::from_str(&line).unwrap();
        let kind = record["type"].as_str().unwrap();
        assert!(BASE_RECORD_TYPES.contains(&kind), "{line}");
    }
}

fn expired_keys(dir: &Path) -> Vec<(String, u64)> {
    journal_records(dir)
        .into_iter()
        .filter_map(|r| match r {
            Record::KeyExpired { key, at_ms, .. } => Some((key, at_ms)),
            _ => None,
        })
        .collect()
}

/// Eight records, the first two keyed and terminal: rid 2 completed (retained body) before rid 1 failed.
fn full_with_two_terminal(l: &mut Ledger) {
    l.admit(req(l, "a", 1, Some("k1"), false), 0);
    l.admit(req(l, "a", 2, Some("k2"), false), 0);
    for n in 3..=8 {
        l.admit(req(l, "a", n, None, false), 0);
    }
    run(l, 1, 1);
    run(l, 2, 1);
    l.complete(
        &rid(2),
        Completion {
            retained: Some(retained("two")),
            session_commit: None,
        },
        5,
    )
    .unwrap();
    l.fail(&rid(1), "backend_error", None, 10).unwrap();
}

#[test]
fn oh03_full_ledger_evicts_the_oldest_terminal_record_and_replays_without_it() {
    let dir = TempDir::new("oh03-evict");
    let (mut l, _) = open_with(&dir.0, evicting(), 0);
    full_with_two_terminal(&mut l);
    // rid 2 ended first, so it is the oldest terminal record although rid 1 was admitted earlier.
    l.admit(req(&l, "a", 9, None, false), 20);
    assert!(
        l.status("a", &rid(2)).is_none(),
        "the evicted id is unknown"
    );
    assert_eq!(state(&l, 1), RequestState::Failed);
    assert_eq!(
        l.ledger_stats(),
        LedgerStats {
            live: 7,
            terminal: 1,
            capacity: 8,
            evictions: 1
        }
    );
    // Its key loses idempotent replay: a retry is told the key expired instead of regenerating.
    assert_eq!(
        l.accept(req(&l, "a", 10, Some("k2"), false), 21),
        Err(LedgerError::IdempotencyKeyExpired)
    );
    l.admit(req(&l, "a", 10, None, false), 22);
    assert!(l.status("a", &rid(1)).is_none());
    // An unkeyed victim: nothing is journaled for it.
    run(&mut l, 3, 23);
    l.complete(
        &rid(3),
        Completion {
            retained: None,
            session_commit: None,
        },
        24,
    )
    .unwrap();
    l.admit(req(&l, "a", 11, None, false), 25);
    assert!(l.status("a", &rid(3)).is_none());
    // Only in-flight records are left: nothing is evicted and the request is rejected.
    let err = l.accept(req(&l, "a", 12, None, false), 26).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (LedgerError::Overloaded, "ledger_full")
    );
    let stats = l.ledger_stats();
    assert_eq!((stats.live, stats.terminal, stats.evictions), (8, 0, 3));
    // A keyed victim is journaled as the `key_expired` the previous release already replays; the journal holds no
    // record type that release would refuse.
    assert_eq!(
        expired_keys(&dir.0),
        [("k2".to_owned(), 20), ("k1".to_owned(), 22)]
    );
    assert_base_record_types(&dir.0);
    let before = known_rids(&l, 12);
    assert_eq!(before, (4..=11).collect());
    drop(l);

    // Replay of the uncompacted journal, then of the journal that open compacted, restores the same set: keyed
    // victims by their `key_expired`, the unkeyed one by evicting down to the bound again.
    for now in [30, 40] {
        let (mut l, rec) = open_with(&dir.0, evicting(), now);
        assert_eq!(known_rids(&l, 12), before);
        for key in ["k1", "k2"] {
            assert_eq!(
                l.accept(req(&l, "a", 13, Some(key), false), now),
                Err(LedgerError::IdempotencyKeyExpired)
            );
        }
        // The recovered orphans hold their slots (worker_restart, not released): never evicted.
        assert_eq!(rec.orphans.len() + rec.recharged.len(), 8);
        assert_eq!(
            l.accept(req(&l, "a", 13, None, false), now),
            Err(LedgerError::Overloaded)
        );
        assert_eq!(
            l.ledger_stats(),
            LedgerStats {
                live: 0,
                terminal: 8,
                capacity: 8,
                evictions: 0
            },
            "counted per incarnation"
        );
        drop(l);
        assert_base_record_types(&dir.0);
    }
}

#[test]
fn oh03_full_ledger_of_in_flight_records_rejects_with_eviction_enabled() {
    let dir = TempDir::new("oh03-in-flight");
    let (mut l, _) = open_with(&dir.0, evicting(), 0);
    for n in 1..=8 {
        l.admit(req(&l, "a", n, Some(&format!("k{n}")), false), 0);
    }
    run(&mut l, 1, 1);
    l.request_cancel("a", &rid(1), "client_cancel", 2).unwrap();
    assert_eq!(state(&l, 1), RequestState::CancelPending);
    let journal = std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap();
    let err = l.accept(req(&l, "a", 9, None, false), 3).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (LedgerError::Overloaded, "ledger_full")
    );
    assert_eq!(known_rids(&l, 9), (1..=8).collect());
    assert_eq!(std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap(), journal);
    assert_eq!(
        l.ledger_stats(),
        LedgerStats {
            live: 8,
            terminal: 0,
            capacity: 8,
            evictions: 0
        }
    );
    drop(l);
    // A journal that expires the key of an in-flight record is corrupt: the open fails closed.
    let mut text = std::fs::read_to_string(dir.0.join(JOURNAL_FILE)).unwrap();
    let scope = journal_records(&dir.0)
        .into_iter()
        .find_map(|r| match r {
            Record::Admitted(a) => Some(a.scope),
            _ => None,
        })
        .unwrap();
    text.push_str(&format!(
        r#"{{"type":"key_expired","scope":"{scope}","key":"k2","at_ms":4}}"#
    ));
    text.push('\n');
    std::fs::write(dir.0.join(JOURNAL_FILE), text).unwrap();
    let keys = Arc::new(DigestKeys::open(&dir.0).unwrap());
    let err = Ledger::open(evicting(), keys, &dir.0, &BTreeMap::new(), 5).unwrap_err();
    let why = format!("key_expired for in-flight rid {}", rid(2));
    assert!(
        matches!(&err, LedgerError::Journal(m) if m.ends_with(&why)),
        "{err:?}"
    );
}

#[test]
fn oh03_without_the_flag_a_full_ledger_rejects_and_journals_nothing() {
    let dir = TempDir::new("oh03-flag-off");
    let (mut l, _) = open(&dir.0, 0);
    full_with_two_terminal(&mut l);
    let journal = std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap();
    assert_eq!(
        l.accept(req(&l, "a", 9, None, false), 20),
        Err(LedgerError::Overloaded)
    );
    assert_eq!(std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap(), journal);
    assert_eq!(known_rids(&l, 9), (1..=8).collect());
    match l.accept(req(&l, "a", 9, Some("k2"), false), 21).unwrap() {
        AcceptOutcome::Replay { rid: r, .. } => assert_eq!(r, rid(2)),
        other => panic!("expected replay, got {other:?}"),
    }
    assert_eq!(
        l.ledger_stats(),
        LedgerStats {
            live: 6,
            terminal: 2,
            capacity: 8,
            evictions: 0
        }
    );
}

#[test]
fn l2_idempotency_outcomes() {
    let dir = TempDir::new("l2");
    let (mut l, _) = open(&dir.0, 0);
    // In progress.
    l.admit(req(&l, "a", 1, Some("k1"), false), 0);
    let err = l.accept(req(&l, "a", 2, Some("k1"), false), 1).unwrap_err();
    assert_eq!(err, LedgerError::RequestInProgress { rid: rid(1) });
    assert_eq!(err.code(), "request_in_progress");
    // Completed unary with retained body → replay; nothing new recorded.
    run(&mut l, 1, 1);
    let completion = Completion {
        retained: Some(retained("first-body")),
        session_commit: None,
    };
    l.complete(&rid(1), completion, 2).unwrap();
    match l.accept(req(&l, "a", 2, Some("k1"), false), 3).unwrap() {
        AcceptOutcome::Replay { rid: r, response } => {
            assert_eq!((r, response.body.as_slice()), (rid(1), &b"first-body"[..]));
        }
        other => panic!("expected replay, got {other:?}"),
    }
    assert!(l.status("a", &rid(2)).is_none());
    // Different digest under the same key.
    let mut changed = req(&l, "a", 2, Some("k1"), false);
    changed.digest = digest(&l, "a", &body("other"));
    let err = l.accept(changed, 3).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (LedgerError::IdempotencyConflict, "idempotency_conflict")
    );
    // The same key in another scope is independent.
    l.admit(req(&l, "b", 3, Some("k1"), false), 3);
    // Streamed original.
    l.admit(req(&l, "a", 4, Some("k2"), true), 3);
    run(&mut l, 4, 3);
    l.complete(
        &rid(4),
        Completion {
            retained: Some(retained("x")),
            session_commit: None,
        },
        4,
    )
    .unwrap();
    let err = l.accept(req(&l, "a", 5, Some("k2"), true), 5).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (
            LedgerError::StreamNotReplayable { rid: rid(4) },
            "stream_not_replayable"
        )
    );
    // Unary without retained body (failed without body after dispatch, and a body over the budget). A request
    // that ends before dispatch releases its key instead (tests/core_ledger_rebind.rs).
    l.admit(req(&l, "a", 5, Some("k3"), false), 5);
    run(&mut l, 5, 5);
    l.fail(&rid(5), "backend_error", None, 6).unwrap();
    let err = l.accept(req(&l, "a", 6, Some("k3"), false), 6).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (
            LedgerError::ResponseNotRetained { rid: rid(5) },
            "response_not_retained"
        )
    );
    l.admit(req(&l, "a", 6, Some("k4"), false), 6);
    run(&mut l, 6, 6);
    let big = "y".repeat(65);
    l.complete(
        &rid(6),
        Completion {
            retained: Some(retained(&big)),
            session_commit: None,
        },
        7,
    )
    .unwrap();
    assert_eq!(
        l.accept(req(&l, "a", 7, Some("k4"), false), 7),
        Err(LedgerError::ResponseNotRetained { rid: rid(6) })
    );
    // The retained-byte budget evicts the oldest body first.
    l.admit(req(&l, "a", 7, Some("k5"), false), 7);
    run(&mut l, 7, 7);
    l.complete(
        &rid(7),
        Completion {
            retained: Some(retained(&"z".repeat(60))),
            session_commit: None,
        },
        8,
    )
    .unwrap();
    assert_eq!(
        l.accept(req(&l, "a", 8, Some("k1"), false), 8),
        Err(LedgerError::ResponseNotRetained { rid: rid(1) })
    );
    assert!(matches!(
        l.accept(req(&l, "a", 8, Some("k5"), false), 8),
        Ok(AcceptOutcome::Replay { .. })
    ));
    // After retention: expired, never regenerated — also across a restart and its compaction.
    let err = l
        .accept(req(&l, "a", 8, Some("k1"), false), 1_002)
        .unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (
            LedgerError::IdempotencyKeyExpired,
            "idempotency_key_expired"
        )
    );
    drop(l);
    let (mut l, _) = open(&dir.0, 1_003);
    assert_eq!(
        l.accept(req(&l, "a", 8, Some("k1"), false), 1_003),
        Err(LedgerError::IdempotencyKeyExpired)
    );
    l.request_cancel("b", &rid(3), "client_cancel", 1_004)
        .unwrap();
    assert!(journal_records(&dir.0)
        .iter()
        .any(|r| matches!(r, Record::KeyExpired { key, .. } if key == "k1")));
    drop(l);
    let (mut l, _) = open(&dir.0, 1_005);
    assert_eq!(
        l.accept(req(&l, "a", 8, Some("k1"), false), 1_005),
        Err(LedgerError::IdempotencyKeyExpired)
    );
}

#[test]
fn l3_crash_replay_round_trip() {
    let dir = TempDir::new("l3-crash");
    let (mut a, rec) = open(&dir.0, 1_000);
    assert_eq!((rec.epoch, rec.orphans.len()), (1, 0));
    let keys = a.digest_keys().clone();
    // A session turn that commits revision 1.
    let session = a.create_session("a", 1_000).unwrap();
    let key = keys.scope_key("a");
    let user = vec![json!({"role": "user", "content": SENTINEL_PROMPT})];
    let claim = SessionClaim {
        session_id: session.session_id.clone(),
        model: "m".to_owned(),
        expected_revision: 0,
        prefix_digest: key.prefix_digest(&session.salt, &[]),
    };
    let mut r1 = req(&a, "a", 1, Some("k1"), false);
    r1.digest = digest(&a, "a", &body(SENTINEL_PROMPT));
    r1.session = Some(claim);
    a.admit(r1, 1_001);
    run(&mut a, 1, 1_002);
    // The grant pins the session (Q8): its `dispatched` record carries the owner (M4 Q3), so the restart below
    // keeps it.
    a.dispatched(&rid(1), "gpu0", 1, &local(URL0), 1_002)
        .unwrap();
    let mut committed = user.clone();
    committed.push(json!({"role": "assistant", "content": SENTINEL_OUTPUT}));
    let commit = Revision {
        message_count: 2,
        digest: key.prefix_digest(&session.salt, &committed),
    };
    a.complete(
        &rid(1),
        Completion {
            retained: Some(retained(SENTINEL_OUTPUT)),
            session_commit: Some(commit),
        },
        1_003,
    )
    .unwrap();
    // A cancelled request, a request in flight mid-stream, and a queued one.
    a.admit(req(&a, "a", 2, Some("k2"), false), 1_004);
    a.request_cancel("a", &rid(2), "client_cancel", 1_005)
        .unwrap();
    let mut r3 = req(&a, "b", 3, Some("k3"), true);
    r3.digest = digest(&a, "b", &body(SENTINEL_PROMPT));
    a.admit(r3, 1_006);
    a.mark_queued(&rid(3), 1_006).unwrap();
    a.mark_running(&rid(3), 1_007).unwrap();
    a.admit(req(&a, "a", 4, None, false), 1_008);
    a.mark_queued(&rid(4), 1_008).unwrap();
    let before = [a.status("a", &rid(1)), a.status("a", &rid(2))];
    let session_before = a.session("a", &session.session_id, 1_009).unwrap();
    assert_eq!(session_before.revision, 1);
    assert_eq!(
        (
            session_before.pinned_worker.as_deref(),
            session_before.owner_generation
        ),
        (Some("gpu0"), 1)
    );

    // Crash: no clean close (dropping a ledger writes nothing; it only closes the files and releases the
    // state-dir lock, as process death does), and the terminal of rid 3 was torn mid-write.
    drop(a);
    let torn = format!(r#"{{"type":"terminal","rid":"{}","state":"compl"#, rid(3));
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.0.join(JOURNAL_FILE))
            .unwrap();
        f.write_all(torn.as_bytes()).unwrap();
    }

    let (mut b, rec) = open(&dir.0, 2_000);
    assert_eq!(rec.epoch, 2);
    assert_eq!(rec.truncated_tail_bytes, torn.len() as u64);
    assert_eq!(
        rec.orphans,
        vec![
            Orphan {
                scope: "b".into(),
                rid: rid(3),
                epoch: 1,
                admitted_ms: 1_006,
                dispatched: None,
                worker: OrphanWorker::Legacy,
            },
            Orphan {
                scope: "a".into(),
                rid: rid(4),
                epoch: 1,
                admitted_ms: 1_008,
                dispatched: None,
                worker: OrphanWorker::Legacy,
            },
        ]
    );
    // Identical reconstructed state for everything that was terminal or durable.
    assert_eq!([b.status("a", &rid(1)), b.status("a", &rid(2))], before);
    let session_after = b.session("a", &session.session_id, 2_000).unwrap();
    assert_eq!(
        (
            session_after.revision,
            session_after.current,
            session_after.salt,
            &session_after.parent
        ),
        (
            session_before.revision,
            session_before.current,
            session_before.salt,
            &session_before.parent
        )
    );
    assert_eq!(
        (
            session_after.pinned_worker.as_deref(),
            session_after.owner_generation
        ),
        (Some("gpu0"), 1),
        "the session owner is durable (M4 Q3)"
    );
    // Prior-epoch non-terminals are failed worker_restart.
    for (scope, n) in [("b", 3), ("a", 4)] {
        let s = b.status(scope, &rid(n)).unwrap();
        assert_eq!(
            (s.state, s.reason.as_deref(), s.epoch),
            (RequestState::Failed, Some("worker_restart"), 1)
        );
    }
    // Same key after restart: no regeneration.
    let mut retry = req(&b, "b", 30, Some("k3"), true);
    retry.digest = digest(&b, "b", &body(SENTINEL_PROMPT));
    assert_eq!(
        b.accept(retry, 2_001),
        Err(LedgerError::StreamNotReplayable { rid: rid(3) })
    );
    // Retained bodies are memory-only.
    let mut again = req(&b, "a", 31, Some("k1"), false);
    again.digest = digest(&b, "a", &body(SENTINEL_PROMPT));
    again.session = None;
    assert_eq!(
        b.accept(again, 2_001),
        Err(LedgerError::ResponseNotRetained { rid: rid(1) })
    );

    // On disk: torn tail gone, epoch 2 fsync'd, worker_restart terminals journaled.
    let records = journal_records(&dir.0);
    assert!(records.contains(&Record::Epoch {
        epoch: 2,
        at_ms: 2_000
    }));
    for n in [3, 4] {
        assert!(records.iter().any(|r| matches!(r, Record::Terminal(t)
            if t.rid == rid(n) && t.reason.as_deref() == Some("worker_restart"))));
    }
    // A further restart finds nothing to recover.
    drop(b);
    let (_, rec) = open(&dir.0, 3_000);
    assert_eq!(
        (rec.epoch, rec.orphans.len(), rec.truncated_tail_bytes),
        (3, 0, 0)
    );

    // Privacy: modes, and no prompt/output text anywhere in the state directory.
    assert_eq!(std::fs::metadata(&dir.0).unwrap().mode() & 0o777, 0o700);
    for entry in std::fs::read_dir(&dir.0).unwrap() {
        let path = entry.unwrap().path();
        assert_eq!(
            std::fs::metadata(&path).unwrap().mode() & 0o777,
            0o600,
            "{}",
            path.display()
        );
        let bytes = std::fs::read(&path).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            !text.contains(SENTINEL_PROMPT) && !text.contains(SENTINEL_OUTPUT),
            "{}",
            path.display()
        );
    }
}

#[test]
fn l3_compaction_keeps_journal_bounded_and_state_identical() {
    let dir = TempDir::new("l3-compact");
    let lim = LedgerLimits {
        max_records: 1_000,
        compaction_slack: 16,
        ..limits()
    };
    let (mut l, _) = open_with(&dir.0, lim, 0);
    let s = l.create_session("a", 0).unwrap();
    for n in 1..=200 {
        l.admit(req(&l, "a", n, Some(&format!("k{n}")), false), n);
        // Dispatched, so the key stays bound (an undispatched cancel would release it, leaving no tombstone).
        run(&mut l, n, n);
        l.fail(&rid(n), "backend_error", None, n).unwrap();
        let lines = journal_records(&dir.0).len() as u64;
        let live = 2 + 2 * n; // epoch + session + (admitted, terminal) per request
        assert!(
            lines <= 2 * live + 16 + 2,
            "journal grew to {lines} lines for {live} live records"
        );
    }
    let statuses: Vec<_> = (1..=200).map(|n| l.status("a", &rid(n))).collect();
    // Retention passes for all of them; a later compaction (bounded by the growth rule) drops them while their
    // keys stay as tombstones.
    let mut dropped = false;
    for n in 201..=900 {
        l.admit(req(&l, "a", n, None, false), 5_000);
        l.request_cancel("a", &rid(n), "client_cancel", 5_000)
            .unwrap();
        let records = journal_records(&dir.0);
        if !records
            .iter()
            .any(|r| matches!(r, Record::Admitted(a) if a.rid == rid(1)))
        {
            assert!(records
                .iter()
                .any(|r| matches!(r, Record::KeyExpired { key, .. } if key == "k1")));
            dropped = true;
            break;
        }
    }
    assert!(dropped, "expired records were never compacted away");
    drop(l);
    let (l, rec) = open_with(&dir.0, lim, 5_000);
    assert!(rec.orphans.is_empty());
    assert_eq!(
        l.session("a", &s.session_id, 5_000).unwrap().current,
        s.current
    );
    assert!(l.status("a", &rid(1)).is_none());
    assert_eq!(
        l.status("a", &rid(260)).unwrap().state,
        RequestState::Cancelled
    );
    assert!(statuses
        .iter()
        .all(|s| s.as_ref().unwrap().state == RequestState::Failed));
    let mut l = l;
    assert_eq!(
        l.accept(req(&l, "a", 999, Some("k1"), false), 5_000),
        Err(LedgerError::IdempotencyKeyExpired)
    );
}

/// A legacy orphan (a ledger opened without configured workers journals no dispatch).
fn orphan(n: u64, admitted_ms: u64) -> Orphan {
    Orphan {
        scope: "a".into(),
        rid: rid(n),
        epoch: 1,
        admitted_ms,
        dispatched: None,
        worker: OrphanWorker::Legacy,
    }
}

/// Q7: a crash while an orphan's slot is held leaves a `worker_restart` terminal without `orphan_released`;
/// the next open re-charges it until its ORIGINAL `admitted_ms + terminal_retention_ms` (1 000 ms here), and
/// not at all once that bound has passed.
#[test]
fn q7_unreleased_orphan_is_recharged_until_its_original_bound() {
    let dir = TempDir::new("q7-recharge");
    let (mut l, _) = open(&dir.0, 0);
    l.admit(req(&l, "a", 1, None, true), 100);
    run(&mut l, 1, 100);
    drop(l);
    let (l, rec) = open(&dir.0, 200);
    assert_eq!(
        (rec.orphans.clone(), rec.recharged),
        (vec![orphan(1, 100)], vec![])
    );
    drop(l);
    // Second crash during the hold: nothing new to fail, the slot is charged again with the original bound.
    let (l, rec) = open(&dir.0, 500);
    assert_eq!((rec.orphans, rec.recharged), (vec![], vec![orphan(1, 100)]));
    assert_eq!(state(&l, 1), RequestState::Failed);
    drop(l);
    let (l, rec) = open(&dir.0, 1_099);
    assert_eq!(rec.recharged, vec![orphan(1, 100)]);
    drop(l);
    // The bound is never extended by a crash: at 100 + 1 000 the hold is over.
    let (_l, rec) = open(&dir.0, 1_100);
    assert_eq!((rec.orphans, rec.recharged), (vec![], vec![]));
}

/// Q7: `orphan_released {rid, at_ms, cause}` is journaled once, survives compaction, and a released orphan is
/// never re-charged; releasing an unknown or already released rid writes nothing.
#[test]
fn q7_released_orphan_is_journaled_and_never_recharged() {
    let dir = TempDir::new("q7-released");
    let (mut l, _) = open(&dir.0, 0);
    l.admit(req(&l, "a", 1, None, true), 100);
    l.admit(req(&l, "a", 2, None, true), 100);
    drop(l);
    let (mut l, rec) = open(&dir.0, 200);
    assert_eq!(rec.orphans, vec![orphan(1, 100), orphan(2, 100)]);
    l.release_orphan(&rid(1), OrphanReleaseCause::Load, 300)
        .unwrap();
    l.release_orphan(&rid(1), OrphanReleaseCause::Retention, 310)
        .unwrap();
    l.release_orphan(&rid(9), OrphanReleaseCause::Load, 320)
        .unwrap();
    let text = std::fs::read_to_string(dir.0.join(JOURNAL_FILE)).unwrap();
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("orphan_released"))
        .collect();
    assert_eq!(
        lines,
        [format!(
            r#"{{"type":"orphan_released","rid":"{}","at_ms":300,"cause":"load"}}"#,
            rid(1)
        )]
    );
    drop(l);
    // The open compacts the journal; the release record is part of the live state.
    let (mut l, rec) = open(&dir.0, 400);
    assert_eq!(rec.recharged, vec![orphan(2, 100)]);
    assert!(
        journal_records(&dir.0).contains(&Record::OrphanReleased(OrphanReleasedRecord {
            rid: rid(1),
            at_ms: 300,
            cause: OrphanReleaseCause::Load,
        }))
    );
    l.release_orphan(&rid(2), OrphanReleaseCause::Retention, 1_000)
        .unwrap();
    drop(l);
    let (_l, rec) = open(&dir.0, 1_050);
    assert_eq!((rec.orphans, rec.recharged), (vec![], vec![]));
    let records = journal_records(&dir.0);
    let released: Vec<_> = records
        .iter()
        .filter_map(|r| match r {
            Record::OrphanReleased(o) => Some((o.rid.clone(), o.cause)),
            _ => None,
        })
        .collect();
    assert_eq!(
        released,
        [
            (rid(1), OrphanReleaseCause::Load),
            (rid(2), OrphanReleaseCause::Retention)
        ]
    );
}

#[test]
fn s1_s2_s3_sessions_through_the_ledger() {
    let dir = TempDir::new("sessions");
    let (mut l, _) = open(&dir.0, 0);
    let keys = l.digest_keys().clone();
    let key = keys.scope_key("a");
    let s = l.create_session("a", 0).unwrap();
    assert_eq!(
        (s.revision, s.current.message_count, s.parent.clone()),
        (0, 0, None)
    );
    let claim = |rev: u64, prefix: Digest| SessionClaim {
        session_id: s.session_id.clone(),
        model: "m".to_owned(),
        expected_revision: rev,
        prefix_digest: prefix,
    };
    let with_claim = |l: &Ledger, n: u64, c: SessionClaim| AcceptRequest {
        session: Some(c),
        ..req(l, "a", n, None, false)
    };
    let genesis = key.prefix_digest(&s.salt, &[]);
    // Two writers at revision 0: the second is rejected before dispatch.
    l.admit(with_claim(&l, 1, claim(0, genesis)), 1);
    let err = l
        .accept(with_claim(&l, 2, claim(0, genesis)), 1)
        .unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (
            LedgerError::Session(SessionError::LeaseHeld),
            "revision_conflict"
        )
    );
    assert!(
        l.status("a", &rid(2)).is_none(),
        "rejected writer leaves no record"
    );
    // Commit advances to revision 1.
    run(&mut l, 1, 2);
    let turn1 = vec![
        json!({"role": "user", "content": "q1"}),
        json!({"role": "assistant", "content": "a1"}),
    ];
    let rev1 = Revision {
        message_count: 2,
        digest: key.prefix_digest(&s.salt, &turn1),
    };
    l.complete(
        &rid(1),
        Completion {
            retained: None,
            session_commit: Some(rev1),
        },
        3,
    )
    .unwrap();
    assert_eq!(l.session("a", &s.session_id, 3).unwrap().revision, 1);
    // Stale revision and edited history.
    assert_eq!(
        l.accept(with_claim(&l, 3, claim(0, genesis)), 4),
        Err(LedgerError::Session(SessionError::RevisionConflict))
    );
    let edited = vec![
        json!({"role": "user", "content": "EDITED"}),
        json!({"role": "assistant", "content": "a1"}),
    ];
    let bad = key.prefix_digest(&s.salt, &edited);
    assert_eq!(
        l.accept(with_claim(&l, 3, claim(1, bad)), 4),
        Err(LedgerError::Session(SessionError::RevisionConflict))
    );
    // A failed turn releases the lease without advancing.
    l.admit(with_claim(&l, 3, claim(1, rev1.digest)), 4);
    run(&mut l, 3, 4);
    l.fail(&rid(3), "backend_error", None, 5).unwrap();
    assert_eq!(l.session("a", &s.session_id, 5).unwrap().revision, 1);
    l.admit(with_claim(&l, 4, claim(1, rev1.digest)), 5);
    // A commit on a request without a session is refused.
    l.admit(req(&l, "a", 5, None, false), 5);
    run(&mut l, 5, 5);
    assert_eq!(
        l.complete(
            &rid(5),
            Completion {
                retained: None,
                session_commit: Some(rev1)
            },
            6
        ),
        Err(LedgerError::Session(SessionError::NotFound))
    );
    // Fork at revision 1: child revision 0 = parent revision 1, same salt lineage.
    let fork = l.fork_session("a", &s.session_id, 1, 6).unwrap();
    assert_eq!(
        (fork.parent_session_id.as_str(), fork.parent_revision),
        (s.session_id.as_str(), 1)
    );
    assert_eq!(
        (fork.child.revision, fork.child.current, fork.child.salt),
        (0, rev1, s.salt)
    );
    assert_eq!(
        l.fork_session("a", &s.session_id, 2, 6),
        Err(LedgerError::Session(SessionError::RevisionConflict))
    );
    // Session limit (2 per scope) and scope isolation.
    let err = l.create_session("a", 7).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (
            LedgerError::Session(SessionError::SessionLimit),
            "session_limit"
        )
    );
    assert_eq!(
        l.session("b", &s.session_id, 7),
        Err(LedgerError::Session(SessionError::NotFound))
    );
    l.create_session("b", 7).unwrap();
    // Fork survives a restart with its lineage.
    drop(l);
    let (l, rec) = open(&dir.0, 8);
    assert_eq!(rec.orphans.len(), 2);
    let child = l.session("a", &fork.child.session_id, 8).unwrap();
    assert_eq!(
        (child.revision, child.current, child.salt, child.parent),
        (0, rev1, s.salt, fork.child.parent)
    );
}

/// Review finding 3: a second open of a live state dir fails before it replays, recovers, compacts or appends,
/// so the first instance's journal stays the file a restart replays.
#[test]
fn l3_state_dir_lock_rejects_a_second_open() {
    let dir = TempDir::new("l3-lock");
    let (mut a, _) = open(&dir.0, 0);
    a.admit(req(&a, "a", 1, Some("k1"), true), 1);
    run(&mut a, 1, 2);
    let before = std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap();
    let keys = Arc::new(DigestKeys::open(&dir.0).unwrap());
    let err = Ledger::open(limits(), keys, &dir.0, &BTreeMap::new(), 10).unwrap_err();
    assert!(
        matches!(&err, LedgerError::Journal(m) if m.contains("locked by another")),
        "{err:?}"
    );
    assert_eq!(std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap(), before);
    assert_eq!(state(&a, 1), RequestState::Running);
    // The first instance's later records stay durable: the next incarnation recovers both requests.
    a.admit(req(&a, "a", 2, Some("k2"), false), 11);
    drop(a);
    let (b, rec) = open(&dir.0, 20);
    assert_eq!(rec.epoch, 2);
    let orphans: Vec<String> = rec.orphans.iter().map(|o| o.rid.clone()).collect();
    assert_eq!(orphans, [rid(1), rid(2)]);
    assert_eq!(
        b.status("a", &rid(2)).unwrap().reason.as_deref(),
        Some("worker_restart")
    );
}

/// Review finding 5: retention is measured from the terminal, also after a compacted journal was replayed
/// (compaction writes requests in acceptance order).
#[test]
fn l1_expiry_follows_terminal_time_after_compacted_replay() {
    let dir = TempDir::new("l1-expiry-order");
    let (mut l, _) = open(&dir.0, 0);
    l.admit(req(&l, "a", 1, Some("kA"), false), 0);
    l.admit(req(&l, "a", 2, Some("kB"), false), 10);
    run(&mut l, 1, 1);
    run(&mut l, 2, 11);
    l.complete(&rid(2), Completion::default(), 20).unwrap();
    l.complete(&rid(1), Completion::default(), 100).unwrap();
    drop(l);
    let (l, _) = open(&dir.0, 200);
    drop(l);
    let (mut l, _) = open(&dir.0, 300);
    // B ended at 20 → expired at 1_020; A (accepted first, ended at 100) is retained until 1_100.
    l.admit(req(&l, "a", 3, None, false), 1_050);
    assert!(l.status("a", &rid(2)).is_none(), "B outlived its retention");
    assert_eq!(
        l.accept(req(&l, "a", 4, Some("kB"), false), 1_050),
        Err(LedgerError::IdempotencyKeyExpired)
    );
    assert_eq!(state(&l, 1), RequestState::Completed);
}

/// Review finding 10: a terminal whose journal write fails leaves the request terminal in memory (failed
/// `journal_error`, no session advance, no retained body) and poisons the ledger: no new work is admitted.
/// The journal still holds `admitted` without a terminal, so the next incarnation fails it `worker_restart`.
#[test]
fn l1_terminal_journal_failure_fails_closed() {
    let dir = TempDir::new("l1-journal-fail");
    // Slack 1: the fourth journal line makes the next write compact first.
    let lim = LedgerLimits {
        compaction_slack: 1,
        ..limits()
    };
    let (mut l, _) = open_with(&dir.0, lim, 0);
    let s = l.create_session("a", 0).unwrap();
    let key = l.digest_keys().scope_key("a");
    let mut r1 = req(&l, "a", 1, Some("k1"), false);
    r1.session = Some(SessionClaim {
        session_id: s.session_id.clone(),
        model: "m".to_owned(),
        expected_revision: 0,
        prefix_digest: key.prefix_digest(&s.salt, &[]),
    });
    l.admit(r1, 1);
    run(&mut l, 1, 2);
    l.admit(req(&l, "a", 2, None, true), 3);
    run(&mut l, 2, 4);
    // The compaction temp path is a directory: the write fails (also for root).
    let blocker = dir.0.join("journal.jsonl.compact.tmp");
    std::fs::create_dir(&blocker).unwrap();
    let messages = vec![
        json!({"role": "user", "content": "hello"}),
        json!({"role": "assistant", "content": "hi"}),
    ];
    let result = l.complete(
        &rid(1),
        Completion {
            retained: Some(retained("hi")),
            session_commit: Some(Revision {
                message_count: 2,
                digest: key.prefix_digest(&s.salt, &messages),
            }),
        },
        5,
    );
    assert!(matches!(result, Err(LedgerError::Journal(_))), "{result:?}");
    let st = l.status("a", &rid(1)).unwrap();
    assert_eq!(
        (st.state, st.reason.as_deref()),
        (RequestState::Failed, Some("journal_error"))
    );
    assert_eq!(l.session("a", &s.session_id, 5).unwrap().revision, 0);
    // Poisoned even once the cause is gone: later terminals settle in memory, admissions are refused.
    std::fs::remove_dir(&blocker).unwrap();
    assert!(l.is_journal_poisoned());
    assert!(matches!(
        l.fail(&rid(2), "backend_error", None, 6),
        Err(LedgerError::Journal(_))
    ));
    assert_eq!(state(&l, 2), RequestState::Failed);
    assert!(l.in_flight().is_empty());
    assert!(matches!(
        l.accept(req(&l, "a", 3, None, false), 7),
        Err(LedgerError::Journal(_))
    ));
    assert!(matches!(
        l.create_session("a", 7),
        Err(LedgerError::Journal(_))
    ));
    drop(l);
    let (mut l, rec) = open_with(&dir.0, lim, 10);
    let orphans: Vec<String> = rec.orphans.iter().map(|o| o.rid.clone()).collect();
    assert_eq!(orphans, [rid(1), rid(2)]);
    assert_eq!(l.session("a", &s.session_id, 10).unwrap().revision, 0);
    assert_eq!(
        l.accept(req(&l, "a", 4, Some("k1"), false), 10),
        Err(LedgerError::ResponseNotRetained { rid: rid(1) })
    );
}

fn usage(prompt: u64, cached: Option<u64>) -> RequestUsage {
    RequestUsage {
        prompt_tokens: Some(prompt),
        completion_tokens: Some(4),
        cached_tokens: cached,
        cache_report: if cached.is_some() {
            CacheReport::Reported
        } else {
            CacheReport::Unavailable
        },
    }
}

/// H2/Q5: the recorded usage is journaled with the terminal, survives a restart, and a journal written before
/// usage existed (Terminal records without `usage`) still replays, as unavailable.
#[test]
fn h2_usage_is_journaled_and_pre_change_journals_replay() {
    let dir = TempDir::new("h2-usage");
    let (mut a, _) = open(&dir.0, 0);
    a.admit(req(&a, "a", 1, None, false), 1);
    run(&mut a, 1, 2);
    // Only a dispatched request takes a usage.
    a.admit(req(&a, "a", 2, None, false), 3);
    assert!(a.record_usage(&rid(2), usage(1, None)).is_err());
    a.record_usage(&rid(1), usage(7577, Some(7552))).unwrap();
    a.complete(&rid(1), Completion::default(), 4).unwrap();
    a.cancel(&rid(2), "client_cancel", 5).unwrap();
    assert_eq!(
        a.status("a", &rid(1)).unwrap().usage,
        usage(7577, Some(7552))
    );
    assert_eq!(
        a.status("a", &rid(2)).unwrap().usage,
        RequestUsage::UNAVAILABLE
    );
    let terminals: Vec<Value> = std::fs::read_to_string(dir.0.join(JOURNAL_FILE))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|r| r["type"] == "terminal")
        .collect();
    assert_eq!(
        terminals[0]["usage"],
        json!({"prompt_tokens": 7577, "completion_tokens": 4, "cached_tokens": 7552, "cache_report": "reported"})
    );
    assert_eq!(terminals[1]["usage"]["cache_report"], "unavailable");
    drop(a);

    let (b, _) = open(&dir.0, 10);
    assert_eq!(
        b.status("a", &rid(1)).unwrap().usage,
        usage(7577, Some(7552))
    );
    drop(b);

    // Rewrite every Terminal record in the M1.3 format (no `usage` key).
    let path = dir.0.join(JOURNAL_FILE);
    let mut old = String::new();
    for line in std::fs::read_to_string(&path).unwrap().lines() {
        let mut record: Value = serde_json::from_str(line).unwrap();
        if let Some(r) = record.as_object_mut() {
            r.remove("usage");
        }
        old.push_str(&record.to_string());
        old.push('\n');
    }
    assert!(!old.contains("usage"));
    std::fs::write(&path, old).unwrap();
    let (c, _) = open(&dir.0, 20);
    let s = c.status("a", &rid(1)).unwrap();
    assert_eq!(s.state, RequestState::Completed);
    assert_eq!(s.usage, RequestUsage::UNAVAILABLE);
}

/// Q5: per-scope counters of this incarnation's terminals; replayed records are not counted.
#[test]
fn h2_stats_count_this_incarnation_per_scope() {
    let dir = TempDir::new("h2-stats");
    let (mut a, _) = open(&dir.0, 1_000);
    assert_eq!(a.opened_ms(), 1_000);
    let zero = a.stats("a");
    assert_eq!(
        serde_json::to_value(&zero).unwrap(),
        json!({"requests": {"cancelled": 0, "completed": 0, "failed": 0}, "prompt_tokens_total": 0,
               "cached_tokens_total": 0, "cache_reported_requests": 0, "cache_unavailable_requests": 0})
    );
    for (n, scope, usage) in [
        (1, "a", usage(100, Some(64))),
        (2, "a", usage(50, Some(0))),
        (3, "b", usage(30, None)),
    ] {
        a.admit(req(&a, scope, n, None, false), 1_001);
        run(&mut a, n, 1_002);
        a.record_usage(&rid(n), usage).unwrap();
        a.complete(&rid(n), Completion::default(), 1_003).unwrap();
    }
    a.admit(req(&a, "a", 4, None, false), 1_004);
    a.fail(&rid(4), "queue_full", None, 1_005).unwrap();
    assert_eq!(
        serde_json::to_value(a.stats("a")).unwrap(),
        json!({"requests": {"cancelled": 0, "completed": 2, "failed": 1}, "prompt_tokens_total": 150,
               "cached_tokens_total": 64, "cache_reported_requests": 2, "cache_unavailable_requests": 1})
    );
    assert_eq!(
        serde_json::to_value(a.stats("b")).unwrap(),
        json!({"requests": {"cancelled": 0, "completed": 1, "failed": 0}, "prompt_tokens_total": 30,
               "cached_tokens_total": 0, "cache_reported_requests": 0, "cache_unavailable_requests": 1})
    );
    drop(a);
    let (b, _) = open(&dir.0, 2_000);
    assert_eq!(b.stats("a"), zero);
    assert_eq!(b.opened_ms(), 2_000);
}

// M3: worker epochs (Q4), `dispatched` records and orphan classes (Q9, Q10), `Entry.worker` (Q11).

const URL0: &str = "http://127.0.0.1:30000";
const URL1: &str = "http://127.0.0.1:30001";
const URL2: &str = "http://127.0.0.1:30002";

fn local(url: &str) -> WorkerTarget {
    WorkerTarget::Local {
        base_url: url.into(),
    }
}

fn workers(list: &[(&str, &str)]) -> BTreeMap<String, WorkerTarget> {
    list.iter()
        .map(|&(id, url)| (id.to_owned(), local(url)))
        .collect()
}

fn two() -> BTreeMap<String, WorkerTarget> {
    workers(&[("gpu0", URL0), ("gpu1", URL1)])
}

fn epochs(list: &[(&str, u64)]) -> BTreeMap<String, u64> {
    list.iter().map(|&(id, e)| (id.to_owned(), e)).collect()
}

fn rid_at(epoch: u64, n: u64) -> String {
    format!("cere-{epoch:020}-{n:020}-0123456789abcdef")
}

/// Admits rid `n` (scope `a`, streamed), runs it and journals its dispatch to `worker` at its current epoch.
fn dispatch(l: &mut Ledger, n: u64, worker: &str, url: &str, now: u64) {
    let r = req(l, "a", n, None, true);
    l.admit(r, now);
    run(l, n, now);
    let epoch = l.worker_epochs()[worker];
    l.dispatched(&rid(n), worker, epoch, &local(url), now)
        .unwrap();
}

fn known(id: &str, url: &str) -> OrphanWorker {
    OrphanWorker::Known {
        id: id.into(),
        base_url: url.into(),
    }
}

fn m3_orphan(
    n: u64,
    admitted_ms: u64,
    dispatched: Option<(&str, u64)>,
    worker: OrphanWorker,
) -> Orphan {
    Orphan {
        scope: "a".into(),
        rid: rid(n),
        epoch: 1,
        admitted_ms,
        dispatched: dispatched.map(|(id, e)| (id.to_owned(), e)),
        worker,
    }
}

fn worker_of(l: &Ledger, n: u64) -> (Option<String>, Option<u64>) {
    let s = l.status("a", &rid(n)).unwrap();
    (s.worker_id, s.worker_epoch)
}

fn journal_lines(dir: &Path) -> Vec<String> {
    std::fs::read_to_string(dir.join(JOURNAL_FILE))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

fn worker_epoch(worker: &str, epoch: u64, at_ms: u64) -> Record {
    Record::WorkerEpoch(WorkerEpochRecord {
        worker: worker.into(),
        epoch,
        at_ms,
    })
}

/// C2/Q4: each configured worker registers at its highest journaled epoch + 1 (default 1) on every open, durably
/// before the open returns; a bump journals +1 at once; a worker that is not configured keeps its highest epoch,
/// so no epoch is ever reused.
#[test]
fn c2_worker_epochs_are_last_plus_one_per_worker() {
    let dir = TempDir::new("c2-epochs");
    let (mut l, _) = open_full(&dir.0, limits(), &two(), 0);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 1), ("gpu1", 1)]));
    assert_eq!(
        journal_records(&dir.0),
        [
            Record::Epoch { epoch: 1, at_ms: 0 },
            worker_epoch("gpu0", 1, 0),
            worker_epoch("gpu1", 1, 0),
        ]
    );
    assert_eq!(l.bump_worker_epoch("gpu1", 5), Ok(2));
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 1), ("gpu1", 2)]));
    assert_eq!(
        journal_lines(&dir.0).last().unwrap(),
        r#"{"type":"worker_epoch","worker":"gpu1","epoch":2,"at_ms":5}"#
    );
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &two(), 10);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 2), ("gpu1", 3)]));
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &workers(&[("gpu0", URL0)]), 20);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 3), ("gpu1", 3)]));
    drop(l);
    let (mut l, _) = open_full(&dir.0, limits(), &workers(&[("gpu1", URL1)]), 30);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 3), ("gpu1", 4)]));
    // A worker the journal never saw starts at 1.
    assert_eq!(l.bump_worker_epoch("gpu2", 31), Ok(1));
    drop(l);
    // Without configured workers nothing registers, and every epoch survives the open's compaction.
    let (l, _) = open(&dir.0, 40);
    assert_eq!(
        l.worker_epochs(),
        epochs(&[("gpu0", 3), ("gpu1", 4), ("gpu2", 1)])
    );
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &two(), 50);
    assert_eq!(
        l.worker_epochs(),
        epochs(&[("gpu0", 4), ("gpu1", 5), ("gpu2", 1)])
    );
}

/// C2/C7 (review M3-02): compaction mid-run, a crash and two restarts: epochs strictly increase per worker and
/// every orphan keeps its worker and its class, because compaction re-emits each worker's highest epoch right
/// after the epoch record and each request's last `dispatched` right after its `admitted`.
#[test]
fn c2_c7_compaction_mid_run_then_two_restarts_keeps_epochs_and_orphan_workers() {
    let dir = TempDir::new("c7-compact");
    let lim = LedgerLimits {
        compaction_slack: 1,
        terminal_retention_ms: 10_000,
        ..limits()
    };
    let (mut l, _) = open_full(&dir.0, lim, &two(), 0);
    dispatch(&mut l, 1, "gpu0", URL0, 10);
    dispatch(&mut l, 2, "gpu1", URL1, 11);
    assert_eq!(l.bump_worker_epoch("gpu1", 12), Ok(2));
    dispatch(&mut l, 3, "gpu1", URL1, 13);
    // Admitted and running; the crash came before its backend call.
    l.admit(req(&l, "a", 4, None, true), 14);
    run(&mut l, 4, 14);
    // Superseded on gpu0 (Q10); its slot is still held at the crash.
    dispatch(&mut l, 5, "gpu0", URL0, 15);
    l.fail(&rid(5), REASON_WORKER_EPOCH_SUPERSEDED, None, 16)
        .unwrap();
    // Dispatched twice (a re-queue): the last record wins.
    dispatch(&mut l, 6, "gpu0", URL0, 17);
    l.dispatched(&rid(6), "gpu1", 2, &local(URL1), 18).unwrap();
    // Slack 1: the journal was compacted after the bump, which it folded into gpu1's single record.
    assert_eq!(
        journal_records(&dir.0)[..3],
        [
            Record::Epoch { epoch: 1, at_ms: 0 },
            worker_epoch("gpu0", 1, 0),
            worker_epoch("gpu1", 2, 12),
        ]
    );
    drop(l);

    let (l, rec) = open_full(&dir.0, lim, &two(), 100);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 2), ("gpu1", 3)]));
    let orphans = vec![
        m3_orphan(1, 10, Some(("gpu0", 1)), known("gpu0", URL0)),
        m3_orphan(2, 11, Some(("gpu1", 1)), known("gpu1", URL1)),
        m3_orphan(3, 13, Some(("gpu1", 2)), known("gpu1", URL1)),
        m3_orphan(4, 14, None, OrphanWorker::NotDispatched),
        m3_orphan(6, 17, Some(("gpu1", 2)), known("gpu1", URL1)),
    ];
    let superseded = m3_orphan(5, 15, Some(("gpu0", 1)), known("gpu0", URL0));
    assert_eq!(rec.orphans, orphans);
    assert_eq!(rec.recharged, vec![superseded.clone()]);
    assert_eq!(
        (
            orphans[4].worker.charged_to(&two()),
            orphans[4].worker.abort_targets(&two())
        ),
        (OrphanCharge::Worker("gpu1".into()), vec!["gpu1".to_owned()])
    );
    assert_eq!(worker_of(&l, 6), (Some("gpu1".into()), Some(2)));
    assert_eq!(worker_of(&l, 4), (None, None));
    // The open's compaction: each request's last `dispatched` right after its `admitted`, and only that one.
    let records = journal_records(&dir.0);
    assert_eq!(
        records[..3],
        [
            Record::Epoch {
                epoch: 2,
                at_ms: 100
            },
            worker_epoch("gpu0", 2, 100),
            worker_epoch("gpu1", 3, 100),
        ]
    );
    let after_admitted = |n: u64| {
        let at = records
            .iter()
            .position(|r| matches!(r, Record::Admitted(a) if a.rid == rid(n)))
            .unwrap();
        records[at + 1].clone()
    };
    for (n, worker, worker_epoch, base_url, at_ms) in [
        (1, "gpu0", 1, URL0, 10),
        (2, "gpu1", 1, URL1, 11),
        (3, "gpu1", 2, URL1, 13),
        (5, "gpu0", 1, URL0, 15),
        (6, "gpu1", 2, URL1, 18),
    ] {
        assert_eq!(
            after_admitted(n),
            Record::Dispatched(DispatchedRecord {
                rid: rid(n),
                worker: worker.into(),
                worker_epoch,
                node: Node::Local,
                base_url: Some(base_url.into()),
                endpoint: None,
                session_owner: None,
                at_ms,
            })
        );
    }
    assert!(matches!(after_admitted(4), Record::Terminal(_)));
    assert_eq!(
        records
            .iter()
            .filter(|r| matches!(r, Record::Dispatched(_)))
            .count(),
        5
    );
    drop(l);

    // Second restart: every held slot is charged again on its own worker; rid 4 was released at once.
    let (l, rec) = open_full(&dir.0, lim, &two(), 200);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 3), ("gpu1", 4)]));
    assert_eq!(rec.orphans, vec![]);
    assert_eq!(
        rec.recharged,
        vec![
            orphans[0].clone(),
            orphans[1].clone(),
            orphans[2].clone(),
            superseded,
            orphans[4].clone(),
        ]
    );
}

/// C7/Q9: a `dispatched` record that cannot be journaled returns the error and changes nothing, so the caller
/// fails the request before any backend sees it; after a crash that request is an M3 orphan never dispatched.
#[test]
fn c7_dispatched_journal_failure_returns_err() {
    let dir = TempDir::new("c7-dispatch-fail");
    // Slack 1: the open leaves 2 lines (epoch, gpu0's worker_epoch); the write after the 5th line compacts first.
    let lim = LedgerLimits {
        compaction_slack: 1,
        ..limits()
    };
    let one = workers(&[("gpu0", URL0)]);
    let (mut l, _) = open_full(&dir.0, lim, &one, 0);
    for n in 1..=4 {
        l.admit(req(&l, "a", n, None, true), n);
    }
    run(&mut l, 1, 5);
    // The compaction temp path is a directory: the write fails (also for root).
    let blocker = dir.0.join("journal.jsonl.compact.tmp");
    std::fs::create_dir(&blocker).unwrap();
    let before = std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap();
    let result = l.dispatched(&rid(1), "gpu0", 1, &local(URL0), 6);
    assert!(matches!(result, Err(LedgerError::Journal(_))), "{result:?}");
    assert_eq!(std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap(), before);
    assert_eq!(worker_of(&l, 1), (None, None));
    assert_eq!(state(&l, 1), RequestState::Running);
    assert!(!l.is_journal_poisoned());
    std::fs::remove_dir(&blocker).unwrap();
    run(&mut l, 2, 7);
    l.dispatched(&rid(2), "gpu0", 1, &local(URL0), 7).unwrap();
    assert_eq!(worker_of(&l, 2), (Some("gpu0".into()), Some(1)));
    drop(l);
    let (_l, rec) = open_full(&dir.0, lim, &one, 10);
    let classes: Vec<(String, OrphanWorker)> = rec
        .orphans
        .iter()
        .map(|o| (o.rid.clone(), o.worker.clone()))
        .collect();
    assert_eq!(
        classes,
        [
            (rid(1), OrphanWorker::NotDispatched),
            (rid(2), known("gpu0", URL0)),
            (rid(3), OrphanWorker::NotDispatched),
            (rid(4), OrphanWorker::NotDispatched),
        ]
    );
}

/// C7: `tests/fixtures/journal_pre_m3.jsonl` holds bytes written by the pre-M3 ledger (b7b310f, the code of
/// main 97222b4), made once by a throwaway test driving `Ledger` (open at 1 000; a session turn rid 1-1 of
/// scope a with key k1 completed; rid 1-2 of scope b left running; reopen at 2 000, which failed rid 1-2
/// `worker_restart` and never released it; rid 2-1 of scope a left running and rid 2-2 of scope b accepted).
/// It recovers with the legacy rule: against the single replica when one is configured, held globally with
/// the abort broadcast when two are; a second restart keeps the class.
#[test]
fn c7_pre_m3_journal_recovers_with_the_legacy_rule() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/journal_pre_m3.jsonl");
    let lim = LedgerLimits {
        terminal_retention_ms: 10_000,
        ..limits()
    };
    for (configured, charge, aborts) in [
        (
            workers(&[("gpu0", URL0)]),
            OrphanCharge::Worker("gpu0".into()),
            vec!["gpu0"],
        ),
        (two(), OrphanCharge::Global, vec!["gpu0", "gpu1"]),
    ] {
        let dir = TempDir::new(&format!("c7-pre-m3-{}", configured.len()));
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::copy(&fixture, dir.0.join(JOURNAL_FILE)).unwrap();
        let (l, rec) = open_full(&dir.0, lim, &configured, 3_000);
        assert_eq!(rec.epoch, 3);
        let legacy = |scope: &str, epoch: u64, n: u64, admitted_ms: u64| Orphan {
            scope: scope.into(),
            rid: rid_at(epoch, n),
            epoch,
            admitted_ms,
            dispatched: None,
            worker: OrphanWorker::Legacy,
        };
        let orphans = vec![legacy("a", 2, 1, 2_001), legacy("b", 2, 2, 2_003)];
        let recharged = vec![legacy("b", 1, 2, 1_004)];
        assert_eq!((&rec.orphans, &rec.recharged), (&orphans, &recharged));
        for o in orphans.iter().chain(&recharged) {
            assert_eq!(o.worker.charged_to(&configured), charge);
            assert_eq!(o.worker.abort_targets(&configured), aborts);
        }
        // Pre-M3 entries report no worker; the session turn and the epochs of the configured workers stand.
        let done = l.status("a", &rid_at(1, 1)).unwrap();
        assert_eq!(
            (done.state, done.worker_id, done.worker_epoch),
            (RequestState::Completed, None, None)
        );
        assert_eq!(
            l.session("a", "sess_c873ed3bb7296b43dc08a5355d751db0", 3_000)
                .unwrap()
                .revision,
            1
        );
        let first: BTreeMap<String, u64> = configured.keys().map(|id| (id.clone(), 1)).collect();
        assert_eq!(l.worker_epochs(), first);
        drop(l);
        let (_l, rec) = open_full(&dir.0, lim, &configured, 4_000);
        assert_eq!(rec.orphans, vec![]);
        assert_eq!(
            rec.recharged,
            vec![recharged[0].clone(), orphans[0].clone(), orphans[1].clone()]
        );
    }
}

/// C7/Q9 (review M3-13): an orphan whose journaled worker is not configured, or whose configured base_url moved,
/// is charged to no worker and aborted only on replicas at the same normalised base_url.
#[test]
fn c7_unconfigured_worker_orphan() {
    let dir = TempDir::new("c7-unconfigured");
    let lim = LedgerLimits {
        terminal_retention_ms: 10_000,
        ..limits()
    };
    let (mut l, _) = open_full(&dir.0, lim, &two(), 0);
    dispatch(&mut l, 1, "gpu1", URL1, 10);
    dispatch(&mut l, 2, "gpu0", URL0, 11);
    drop(l);
    // gpu1 is gone and a new id serves its old URL.
    let configured = workers(&[("gpu0", URL0), ("gpu2", URL1)]);
    let (l, rec) = open_full(&dir.0, lim, &configured, 100);
    let unconfigured = m3_orphan(
        1,
        10,
        Some(("gpu1", 1)),
        OrphanWorker::Unconfigured {
            base_url: URL1.into(),
        },
    );
    let on_gpu0 = m3_orphan(2, 11, Some(("gpu0", 1)), known("gpu0", URL0));
    assert_eq!(rec.orphans, vec![unconfigured.clone(), on_gpu0.clone()]);
    assert_eq!(
        unconfigured.worker.charged_to(&configured),
        OrphanCharge::Global
    );
    assert_eq!(unconfigured.worker.abort_targets(&configured), ["gpu2"]);
    assert_eq!(
        on_gpu0.worker.charged_to(&configured),
        OrphanCharge::Worker("gpu0".into())
    );
    assert_eq!(on_gpu0.worker.abort_targets(&configured), ["gpu0"]);
    // The unconfigured worker keeps its epoch; the new id registers at 1.
    assert_eq!(
        l.worker_epochs(),
        epochs(&[("gpu0", 2), ("gpu1", 1), ("gpu2", 1)])
    );
    drop(l);
    // gpu0 moved to another port: its orphan is no longer Known, and nothing serves either journaled URL.
    let moved = workers(&[("gpu0", URL2)]);
    let (_l, rec) = open_full(&dir.0, lim, &moved, 200);
    let gone = |o: &Orphan, url: &str| Orphan {
        worker: OrphanWorker::Unconfigured {
            base_url: url.into(),
        },
        ..o.clone()
    };
    assert_eq!(
        rec.recharged,
        vec![gone(&unconfigured, URL1), gone(&on_gpu0, URL0)]
    );
    for o in &rec.recharged {
        assert_eq!(
            (o.worker.charged_to(&moved), o.worker.abort_targets(&moved)),
            (OrphanCharge::Global, vec![])
        );
    }
}

/// C7/Q9: an M3 admission (`dispatch_journaled`) without a `dispatched` record never reached a backend: the
/// restart fails it `worker_restart` and journals `orphan_released` (`not_dispatched`) at once; nothing is held
/// or aborted, and a later restart never charges it, even when that release was lost (a crash inside the open).
#[test]
fn c7_m3_orphan_without_dispatched_is_released_at_once() {
    let dir = TempDir::new("c7-not-dispatched");
    let one = workers(&[("gpu0", URL0)]);
    let (mut l, _) = open_full(&dir.0, limits(), &one, 0);
    l.admit(req(&l, "a", 1, None, true), 100);
    run(&mut l, 1, 100);
    l.admit(req(&l, "a", 2, None, false), 101);
    assert!(journal_lines(&dir.0)
        .iter()
        .any(|line| line.contains(&rid(2)) && line.contains(r#""dispatch_journaled":true"#)));
    drop(l);
    let (l, rec) = open_full(&dir.0, limits(), &one, 200);
    assert_eq!(
        rec.orphans,
        vec![
            m3_orphan(1, 100, None, OrphanWorker::NotDispatched),
            m3_orphan(2, 101, None, OrphanWorker::NotDispatched),
        ]
    );
    assert_eq!(rec.recharged, vec![]);
    for o in &rec.orphans {
        assert_eq!(o.worker.charged_to(&one), OrphanCharge::NotHeld);
        assert!(o.worker.abort_targets(&one).is_empty());
        let s = l.status("a", &o.rid).unwrap();
        assert_eq!(
            (s.state, s.reason.as_deref()),
            (RequestState::Failed, Some("worker_restart"))
        );
    }
    let released: Vec<String> = journal_lines(&dir.0)
        .into_iter()
        .filter(|line| line.contains("orphan_released"))
        .collect();
    assert_eq!(
        released,
        [rid(1), rid(2)].map(|r| format!(
            r#"{{"type":"orphan_released","rid":"{r}","at_ms":200,"cause":"not_dispatched"}}"#
        ))
    );
    drop(l);
    let (l, rec) = open_full(&dir.0, limits(), &one, 300);
    assert_eq!((rec.orphans, rec.recharged), (vec![], vec![]));
    drop(l);
    // The releases never reached the disk: the next open releases the terminals instead of charging them.
    let kept: Vec<String> = journal_lines(&dir.0)
        .into_iter()
        .filter(|line| !line.contains("orphan_released"))
        .map(|line| line + "\n")
        .collect();
    std::fs::write(dir.0.join(JOURNAL_FILE), kept.concat()).unwrap();
    let (_l, rec) = open_full(&dir.0, limits(), &one, 400);
    assert_eq!((rec.orphans, rec.recharged), (vec![], vec![]));
    let released: Vec<String> = journal_lines(&dir.0)
        .into_iter()
        .filter(|line| line.contains("orphan_released"))
        .collect();
    assert_eq!(
        released,
        [rid(1), rid(2)].map(|r| format!(
            r#"{{"type":"orphan_released","rid":"{r}","at_ms":400,"cause":"not_dispatched"}}"#
        ))
    );
}

/// C2/C7: worker records that contradict the journal fail the open (fail closed, like a second terminal): a worker
/// epoch that does not increase, a `dispatched` for an unknown rid, a `dispatched` after the rid's terminal.
#[test]
fn c2_c7_inconsistent_worker_records_fail_the_open() {
    let dir = TempDir::new("c7-inconsistent");
    let (mut l, _) = open_full(&dir.0, limits(), &two(), 0);
    dispatch(&mut l, 1, "gpu0", URL0, 10);
    l.complete(&rid(1), Completion::default(), 11).unwrap();
    drop(l);
    let good = std::fs::read_to_string(dir.0.join(JOURNAL_FILE)).unwrap();
    let dispatched = |r: &str| {
        format!(
            r#"{{"type":"dispatched","rid":"{r}","worker":"gpu0","worker_epoch":1,"base_url":"{URL0}","at_ms":12}}"#
        )
    };
    for (extra, why) in [
        (
            r#"{"type":"worker_epoch","worker":"gpu0","epoch":1,"at_ms":12}"#.to_owned(),
            "worker gpu0 epoch 1 not increasing".to_owned(),
        ),
        (
            dispatched(&rid(9)),
            format!("dispatched for unknown rid {}", rid(9)),
        ),
        (
            dispatched(&rid(1)),
            format!("dispatched after the terminal of rid {}", rid(1)),
        ),
    ] {
        std::fs::write(dir.0.join(JOURNAL_FILE), format!("{good}{extra}\n")).unwrap();
        let keys = Arc::new(DigestKeys::open(&dir.0).unwrap());
        let err = Ledger::open(limits(), keys, &dir.0, &two(), 20).unwrap_err();
        assert!(
            matches!(&err, LedgerError::Journal(m) if m.ends_with(&why)),
            "{err:?}"
        );
    }
}

/// Q10 (review M3-06): an unreleased `worker_epoch_superseded` terminal is charged again on its own worker by a
/// restart until its original bound, like `worker_restart`; a released one (here `relaunch`) never is, and
/// other failures never are.
#[test]
fn q10_superseded_terminal_is_recharged_until_released() {
    let dir = TempDir::new("q10-superseded");
    let one = workers(&[("gpu0", URL0)]);
    let (mut l, _) = open_full(&dir.0, limits(), &one, 0);
    for n in 1..=4 {
        dispatch(&mut l, n, "gpu0", URL0, 100);
    }
    for n in 1..=3 {
        l.fail(&rid(n), REASON_WORKER_EPOCH_SUPERSEDED, None, 150)
            .unwrap();
    }
    l.fail(&rid(4), "backend_unavailable", None, 150).unwrap();
    l.release_orphan(&rid(2), OrphanReleaseCause::Relaunch, 160)
        .unwrap();
    assert_eq!(
        journal_lines(&dir.0).last().unwrap(),
        &format!(
            r#"{{"type":"orphan_released","rid":"{}","at_ms":160,"cause":"relaunch"}}"#,
            rid(2)
        )
    );
    drop(l);
    let held = |n: u64| m3_orphan(n, 100, Some(("gpu0", 1)), known("gpu0", URL0));
    let (mut l, rec) = open_full(&dir.0, limits(), &one, 200);
    assert_eq!(
        (rec.orphans, rec.recharged),
        (vec![], vec![held(1), held(3)])
    );
    assert_eq!(
        held(1).worker.charged_to(&one),
        OrphanCharge::Worker("gpu0".into())
    );
    l.release_orphan(&rid(1), OrphanReleaseCause::Load, 300)
        .unwrap();
    drop(l);
    let (l, rec) = open_full(&dir.0, limits(), &one, 1_099);
    assert_eq!(rec.recharged, vec![held(3)]);
    drop(l);
    // The original bound (admitted 100 + retention 1 000) is never extended.
    let (_l, rec) = open_full(&dir.0, limits(), &one, 1_100);
    assert_eq!(rec.recharged, vec![]);
}

/// Review J2: a request that ended at dispatch before any backend call never reached a backend, so it releases
/// its Idempotency-Key like one that ended while queued and the retry runs as new: its worker's epoch moved past
/// the grant (`worker_epoch_superseded` from `dispatched`), or a cancel raced the dispatch. A restart replays
/// the release.
#[test]
fn j2_a_request_ended_before_its_backend_call_releases_its_key() {
    let dir = TempDir::new("j2-dispatch-key");
    let one = workers(&[("gpu0", URL0)]);
    let (mut l, _) = open_full(&dir.0, limits(), &one, 0);
    for (n, key) in [(1, "k1"), (2, "k2"), (3, "k3")] {
        l.admit(req(&l, "a", n, Some(key), false), n);
        run(&mut l, n, n);
    }
    assert_eq!(l.bump_worker_epoch("gpu0", 4), Ok(2));
    for n in [1, 2] {
        assert_eq!(
            l.dispatched(&rid(n), "gpu0", 1, &local(URL0), 5),
            Err(LedgerError::WorkerEpochSuperseded)
        );
    }
    assert_eq!(
        l.request_cancel("a", &rid(3), "cancel_requested", 6),
        Ok(CancelOutcome::AbortRequired)
    );
    l.cancel(&rid(3), "cancel_requested", 7).unwrap();
    for (n, key) in [(4, "k1"), (5, "k3")] {
        assert_eq!(
            l.accept(req(&l, "a", n, Some(key), false), 8),
            Ok(AcceptOutcome::Accepted),
            "{key}"
        );
    }
    drop(l);
    let (mut l, _) = open_full(&dir.0, limits(), &one, 10);
    assert_eq!(
        l.accept(req(&l, "a", 6, Some("k2"), false), 11),
        Ok(AcceptOutcome::Accepted)
    );
}

/// Review J2: a request whose `dispatched` record was never written (the write failed, so the server failed it
/// `journal_error`, or the process died first) never reached a backend. The restart fails it `worker_restart`
/// and releases its key with that terminal, so the retry runs as new, also after a later restart; a dispatched
/// request keeps its key (`response_not_retained`).
#[test]
fn j2_a_never_dispatched_request_releases_its_key_on_restart() {
    let dir = TempDir::new("j2-restart-key");
    let one = workers(&[("gpu0", URL0)]);
    let (mut l, _) = open_full(&dir.0, limits(), &one, 0);
    for (n, key) in [(1, "k1"), (2, "k2"), (3, "k3")] {
        l.admit(req(&l, "a", n, Some(key), false), n);
        run(&mut l, n, n);
    }
    l.dispatched(&rid(2), "gpu0", 1, &local(URL0), 4).unwrap();
    drop(l);
    let (mut l, _) = open_full(&dir.0, limits(), &one, 10);
    assert_eq!(
        l.accept(req(&l, "a", 4, Some("k1"), false), 11),
        Ok(AcceptOutcome::Accepted)
    );
    assert_eq!(
        l.accept(req(&l, "a", 5, Some("k2"), false), 12),
        Err(LedgerError::ResponseNotRetained { rid: rid(2) })
    );
    drop(l);
    // rid 3's release was journaled with the first restart's terminal; this one replays it.
    let (mut l, _) = open_full(&dir.0, limits(), &one, 20);
    assert_eq!(
        l.accept(req(&l, "a", 6, Some("k3"), false), 21),
        Ok(AcceptOutcome::Accepted)
    );
    assert_eq!(
        l.accept(req(&l, "a", 7, Some("k2"), false), 22),
        Err(LedgerError::ResponseNotRetained { rid: rid(2) })
    );
}

/// Q11: `Entry.worker` (status `worker_id`/`worker_epoch`) is null before dispatch, set by `dispatched` and
/// restored on replay; `dispatched` needs a running request of this ledger.
#[test]
fn q11_entry_worker_is_null_set_and_replayed() {
    let dir = TempDir::new("q11-worker");
    let (mut l, _) = open_full(&dir.0, limits(), &two(), 0);
    l.admit(req(&l, "a", 1, None, false), 1);
    assert_eq!(worker_of(&l, 1), (None, None));
    assert_eq!(
        l.dispatched(&rid(1), "gpu0", 1, &local(URL0), 2),
        Err(LedgerError::InvalidTransition {
            from: RequestState::Accepted,
            to: RequestState::Accepted
        })
    );
    assert_eq!(
        l.dispatched(&rid(9), "gpu0", 1, &local(URL0), 2),
        Err(LedgerError::NotFound)
    );
    run(&mut l, 1, 2);
    l.dispatched(&rid(1), "gpu1", 1, &local(URL1), 3).unwrap();
    assert_eq!(worker_of(&l, 1), (Some("gpu1".into()), Some(1)));
    assert_eq!(
        journal_lines(&dir.0).last().unwrap(),
        &format!(
            r#"{{"type":"dispatched","rid":"{}","worker":"gpu1","worker_epoch":1,"base_url":"{URL1}","at_ms":3}}"#,
            rid(1)
        )
    );
    l.complete(&rid(1), Completion::default(), 4).unwrap();
    // Ended before dispatch (queue_full): no worker.
    l.admit(req(&l, "a", 2, None, false), 5);
    l.fail(&rid(2), "queue_full", None, 6).unwrap();
    assert_eq!(worker_of(&l, 2), (None, None));
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &two(), 10);
    assert_eq!(worker_of(&l, 1), (Some("gpu1".into()), Some(1)));
    assert_eq!(worker_of(&l, 2), (None, None));
    drop(l);
    // Replayed again from the compacted journal.
    let (l, _) = open(&dir.0, 20);
    assert_eq!(worker_of(&l, 1), (Some("gpu1".into()), Some(1)));
}

// M3.5: remote workers (Q4 epochs, Q11 `dispatched`, Q15 orphans) and journaled drain intent (Q13).

const ENDPOINT: &str = "100.64.0.2:50051";
const REMOTE: &str = "worker-a";

fn remote(endpoint: &str) -> WorkerTarget {
    WorkerTarget::Remote {
        endpoint: endpoint.into(),
    }
}

/// Local `gpu0` at URL0 and remote `worker-a` at ENDPOINT.
fn mixed() -> BTreeMap<String, WorkerTarget> {
    BTreeMap::from([
        ("gpu0".to_owned(), local(URL0)),
        (REMOTE.to_owned(), remote(ENDPOINT)),
    ])
}

fn remote_orphan(n: u64, admitted_ms: u64, epoch: u64) -> Orphan {
    m3_orphan(
        n,
        admitted_ms,
        Some((REMOTE, epoch)),
        OrphanWorker::Remote {
            id: REMOTE.into(),
            endpoint: ENDPOINT.into(),
        },
    )
}

/// P6/Q4: only local ids get the open-time epoch bump; a remote worker's epoch is the one its agent registered,
/// journaled by `accept_worker_epoch` only when it is above the journaled one, and never bumped by a restart.
#[test]
fn p6_remote_ids_are_not_bumped_at_open() {
    let dir = TempDir::new("p6-remote-epochs");
    let (mut l, _) = open_full(&dir.0, limits(), &mixed(), 0);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 1)]));
    assert_eq!(l.worker_epoch(REMOTE), None);
    assert_eq!(l.worker_epoch("gpu0"), Some(1));
    assert_eq!(
        journal_records(&dir.0),
        [
            Record::Epoch { epoch: 1, at_ms: 0 },
            worker_epoch("gpu0", 1, 0)
        ]
    );
    assert_eq!(l.accept_worker_epoch(REMOTE, 7, 5), Ok(()));
    assert_eq!(l.worker_epoch(REMOTE), Some(7));
    assert_eq!(
        journal_lines(&dir.0).last().unwrap(),
        r#"{"type":"worker_epoch","worker":"worker-a","epoch":7,"at_ms":5}"#
    );
    for stale in [7, 6] {
        assert_eq!(
            l.accept_worker_epoch(REMOTE, stale, 6),
            Err(LedgerError::StaleWorkerEpoch { journaled: 7 })
        );
    }
    assert_eq!(
        LedgerError::StaleWorkerEpoch { journaled: 7 }.code(),
        "stale_epoch"
    );
    assert_eq!(l.accept_worker_epoch(REMOTE, 9, 8), Ok(()));
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &mixed(), 10);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 2), (REMOTE, 9)]));
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &mixed(), 20);
    assert_eq!(l.worker_epochs(), epochs(&[("gpu0", 3), (REMOTE, 9)]));
}

/// P7/Q11: an M3-era journal (`dispatched` without `node`, with `base_url`) replays unchanged: its records mean
/// local workers, its orphan is Known on its worker, and compaction re-emits the very same lines.
#[test]
fn p7_m3_era_dispatched_lines_replay_as_local() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/journal_m3.jsonl");
    let text = std::fs::read_to_string(&fixture).unwrap();
    let m3_lines: Vec<&str> = text
        .lines()
        .filter(|l| l.contains(r#""type":"dispatched""#))
        .collect();
    assert_eq!(m3_lines.len(), 2);
    assert!(m3_lines.iter().all(|l| !l.contains("node")));
    assert_eq!(
        serde_json::from_str::<Record>(m3_lines[1]).unwrap(),
        Record::Dispatched(DispatchedRecord {
            rid: rid(2),
            worker: "gpu1".into(),
            worker_epoch: 1,
            node: Node::Local,
            base_url: Some(URL1.into()),
            endpoint: None,
            session_owner: None,
            at_ms: 1_002,
        })
    );
    let dir = TempDir::new("p7-m3-era");
    std::fs::create_dir_all(&dir.0).unwrap();
    std::fs::copy(&fixture, dir.0.join(JOURNAL_FILE)).unwrap();
    let (l, rec) = open_full(&dir.0, limits(), &two(), 1_500);
    assert_eq!(rec.epoch, 2);
    let orphan = m3_orphan(2, 1_002, Some(("gpu1", 1)), known("gpu1", URL1));
    assert_eq!(rec.orphans, vec![orphan.clone()]);
    assert_eq!(
        orphan.worker.charged_to(&two()),
        OrphanCharge::Worker("gpu1".into())
    );
    assert_eq!(worker_of(&l, 1), (Some("gpu0".into()), Some(1)));
    let lines = journal_lines(&dir.0);
    for m3 in &m3_lines {
        assert!(lines.iter().any(|l| l == m3), "{m3} not re-emitted");
    }
}

/// P7/Q11: a remote `dispatched` record carries `node: remote` and the endpoint (no base_url) and round-trips
/// through replay and compaction; a record whose node contradicts its address fields fails the open.
#[test]
fn p7_remote_dispatched_record_round_trips() {
    let dir = TempDir::new("p7-remote-dispatched");
    let (mut l, _) = open_full(&dir.0, limits(), &mixed(), 0);
    l.accept_worker_epoch(REMOTE, 3, 1).unwrap();
    l.admit(req(&l, "a", 1, None, true), 2);
    run(&mut l, 1, 2);
    l.dispatched(&rid(1), REMOTE, 3, &remote(ENDPOINT), 2)
        .unwrap();
    let line = format!(
        r#"{{"type":"dispatched","rid":"{}","worker":"worker-a","worker_epoch":3,"node":"remote","endpoint":"{ENDPOINT}","at_ms":2}}"#,
        rid(1)
    );
    assert_eq!(journal_lines(&dir.0).last().unwrap(), &line);
    assert_eq!(worker_of(&l, 1), (Some(REMOTE.into()), Some(3)));
    // The Q10 supersede rule holds for remote epochs too.
    l.admit(req(&l, "a", 2, None, true), 3);
    run(&mut l, 2, 3);
    assert_eq!(
        l.dispatched(&rid(2), REMOTE, 2, &remote(ENDPOINT), 3),
        Err(LedgerError::WorkerEpochSuperseded)
    );
    drop(l);
    let good = std::fs::read_to_string(dir.0.join(JOURNAL_FILE)).unwrap();
    let (l, _) = open_full(&dir.0, limits(), &mixed(), 10);
    assert_eq!(worker_of(&l, 1), (Some(REMOTE.into()), Some(3)));
    assert!(journal_lines(&dir.0).contains(&line));
    drop(l);
    let bad = |fields: &str| {
        format!(
            r#"{{"type":"dispatched","rid":"{}","worker":"worker-a","worker_epoch":3,{fields},"at_ms":4}}"#,
            rid(1)
        )
    };
    let local_why = format!(
        "dispatched rid {}: node local needs base_url and no endpoint",
        rid(1)
    );
    let remote_why = format!(
        "dispatched rid {}: node remote needs endpoint and no base_url",
        rid(1)
    );
    for (extra, why) in [
        (bad(&format!(r#""endpoint":"{ENDPOINT}""#)), &local_why),
        (
            bad(&format!(r#""base_url":"{URL0}","endpoint":"{ENDPOINT}""#)),
            &local_why,
        ),
        (bad(r#""node":"remote""#), &remote_why),
        (
            bad(&format!(r#""node":"remote","base_url":"{URL0}""#)),
            &remote_why,
        ),
    ] {
        std::fs::write(dir.0.join(JOURNAL_FILE), format!("{good}{extra}\n")).unwrap();
        let keys = Arc::new(DigestKeys::open(&dir.0).unwrap());
        let err = Ledger::open(limits(), keys, &dir.0, &mixed(), 20).unwrap_err();
        assert!(
            matches!(&err, LedgerError::Journal(m) if m.ends_with(why.as_str())),
            "{err:?}"
        );
    }
}

/// Q15: a coordinator restart fails a prior-term remote request `worker_restart` and holds it as an orphan
/// charged to its remote worker (when that id is still configured at the journaled endpoint) until the agent's
/// `cleanup_acknowledged` releases it (journaled), or until its retention bound.
#[test]
fn q15_remote_orphan_is_held_until_cleanup_acknowledged_or_retention() {
    let dir = TempDir::new("q15-remote-orphan");
    let lim = LedgerLimits {
        terminal_retention_ms: 10_000,
        ..limits()
    };
    let (mut l, _) = open_full(&dir.0, lim, &mixed(), 0);
    l.accept_worker_epoch(REMOTE, 1, 1).unwrap();
    for (n, at) in [(1, 10), (2, 11)] {
        l.admit(req(&l, "a", n, None, true), at);
        run(&mut l, n, at);
        l.dispatched(&rid(n), REMOTE, 1, &remote(ENDPOINT), at)
            .unwrap();
    }
    dispatch(&mut l, 3, "gpu0", URL0, 12);
    drop(l);

    let (mut l, rec) = open_full(&dir.0, lim, &mixed(), 100);
    let on_gpu0 = m3_orphan(3, 12, Some(("gpu0", 1)), known("gpu0", URL0));
    let orphans = vec![remote_orphan(1, 10, 1), remote_orphan(2, 11, 1), on_gpu0];
    assert_eq!(rec.orphans, orphans);
    assert_eq!(rec.recharged, vec![]);
    let s = l.status("a", &rid(1)).unwrap();
    assert_eq!(
        (s.state, s.reason.as_deref()),
        (RequestState::Failed, Some("worker_restart"))
    );
    let w = &orphans[0].worker;
    assert_eq!(
        (w.charged_to(&mixed()), w.abort_targets(&mixed())),
        (OrphanCharge::Worker(REMOTE.into()), vec![REMOTE.to_owned()])
    );
    // Not configured, configured at another endpoint, or now a local id: no worker holds it.
    for configured in [
        workers(&[("gpu0", URL0)]),
        BTreeMap::from([(REMOTE.to_owned(), remote("100.64.0.9:50051"))]),
        workers(&[(REMOTE, URL1)]),
    ] {
        assert_eq!(
            (w.charged_to(&configured), w.abort_targets(&configured)),
            (OrphanCharge::Global, vec![])
        );
    }
    // Only the remote worker it was dispatched to releases it, once.
    assert_eq!(l.release_remote_orphan(&rid(1), "gpu0", 150), Ok(false));
    assert_eq!(l.release_remote_orphan(&rid(3), "gpu0", 150), Ok(false));
    assert_eq!(l.release_remote_orphan(&rid(9), REMOTE, 150), Ok(false));
    assert_eq!(l.release_remote_orphan(&rid(1), REMOTE, 150), Ok(true));
    assert_eq!(
        journal_records(&dir.0).last().unwrap(),
        &Record::OrphanReleased(OrphanReleasedRecord {
            rid: rid(1),
            at_ms: 150,
            cause: OrphanReleaseCause::CleanupAcknowledged,
        })
    );
    assert_eq!(l.release_remote_orphan(&rid(1), REMOTE, 151), Ok(false));
    drop(l);

    // Unreleased remote orphans are charged again until their original bound (admitted + retention).
    let (_l, rec) = open_full(&dir.0, lim, &mixed(), 200);
    assert_eq!(rec.recharged, orphans[1..].to_vec());
    drop(_l);
    let (_l, rec) = open_full(&dir.0, lim, &mixed(), 10_011);
    assert_eq!(rec.recharged, orphans[2..].to_vec());
}

/// Q15 / H2: a held remote orphan (an accepted remote call that did not end naturally, journaled `orphan_held`
/// when its hold started) that has no `orphan_released` is charged again by the next open until its original
/// bound (admitted + retention), whatever its terminal (`worker_lost`, `client_disconnected`), also from a
/// compacted journal. The same terminal never held, one held and then released, and one past its bound are not.
#[test]
fn q15_a_held_remote_orphan_is_charged_again_after_a_restart() {
    let dir = TempDir::new("q15-held-recharge");
    let lim = LedgerLimits {
        terminal_retention_ms: 10_000,
        ..limits()
    };
    let (mut l, _) = open_full(&dir.0, lim, &mixed(), 0);
    l.accept_worker_epoch(REMOTE, 1, 1).unwrap();
    for (n, at) in [(4, 10), (1, 5_000), (2, 5_001), (3, 5_002), (5, 5_003)] {
        l.admit(req(&l, "a", n, None, true), at);
        run(&mut l, n, at);
        l.dispatched(&rid(n), REMOTE, 1, &remote(ENDPOINT), at)
            .unwrap();
    }
    // Held past its bound (admitted 10) by the reopen at 10_010; its terminal is still retained.
    l.fail(&rid(4), "deadline_exceeded", None, 20).unwrap();
    l.fail(&rid(1), "worker_lost", None, 5_010).unwrap();
    // The same terminal, never held (e.g. ended before its receipt).
    l.fail(&rid(2), "worker_lost", None, 5_010).unwrap();
    l.fail(&rid(3), "deadline_exceeded", None, 5_010).unwrap();
    l.request_cancel("a", &rid(5), "client_disconnected", 5_010)
        .unwrap();
    l.cancel(&rid(5), "client_disconnected", 5_010).unwrap();
    drop(l);
    let held = |n: u64, at_ms: u64| {
        format!(
            r#"{{"type":"orphan_held","rid":"{}","worker":"worker-a","at_ms":{at_ms}}}"#,
            rid(n)
        )
    };
    // The previous incarnation's holds; rid 3's hold was released (cleanup acknowledged) before it stopped.
    let released = format!(
        r#"{{"type":"orphan_released","rid":"{}","at_ms":5020,"cause":"cleanup_acknowledged"}}"#,
        rid(3)
    );
    let mut text = std::fs::read_to_string(dir.0.join(JOURNAL_FILE)).unwrap();
    for line in [
        held(4, 20),
        held(1, 5_010),
        held(3, 5_010),
        released,
        held(5, 5_010),
    ] {
        text.push_str(&line);
        text.push('\n');
    }
    std::fs::write(dir.0.join(JOURNAL_FILE), text).unwrap();

    let expected = vec![remote_orphan(1, 5_000, 1), remote_orphan(5, 5_003, 1)];
    let (l, rec) = open_full(&dir.0, lim, &mixed(), 10_010);
    assert_eq!(rec.orphans, vec![]);
    assert_eq!(rec.recharged, expected);
    let s = l.status("a", &rid(5)).unwrap();
    assert_eq!(
        (s.state, s.reason.as_deref()),
        (RequestState::Cancelled, Some("client_disconnected"))
    );
    drop(l);
    // The open compacted the journal: the unreleased holds were re-emitted, so the next open charges them again.
    let lines = journal_lines(&dir.0);
    for n in [1, 5] {
        assert!(lines.contains(&held(n, 5_010)), "{n} not re-emitted");
    }
    let rid3_held = r#""type":"orphan_held","rid":""#.to_owned() + &rid(3);
    assert!(!lines.iter().any(|line| line.contains(&rid3_held)));
    let (l, rec) = open_full(&dir.0, lim, &mixed(), 10_011);
    assert_eq!(rec.recharged, expected);
    drop(l);
    // Until their original bound only.
    let (_l, rec) = open_full(&dir.0, lim, &mixed(), 15_003);
    assert_eq!(rec.recharged, vec![]);
}

/// Q15 / H2: `hold_remote_orphan` journals one `orphan_held` per rid, only for the remote worker of its last
/// `dispatched` record; a hold journaled before the request's terminal (the executor drops the slot first on some
/// paths) still makes the terminal request charged again by the next open.
#[test]
fn q15_hold_remote_orphan_journals_once_for_its_remote_worker() {
    let dir = TempDir::new("q15-hold-api");
    let lim = LedgerLimits {
        terminal_retention_ms: 10_000,
        ..limits()
    };
    let (mut l, _) = open_full(&dir.0, lim, &mixed(), 0);
    l.accept_worker_epoch(REMOTE, 1, 1).unwrap();
    l.admit(req(&l, "a", 1, None, true), 5);
    run(&mut l, 1, 5);
    l.dispatched(&rid(1), REMOTE, 1, &remote(ENDPOINT), 5)
        .unwrap();
    dispatch(&mut l, 2, "gpu0", URL0, 6);
    l.admit(req(&l, "a", 3, None, true), 7);
    run(&mut l, 3, 7);
    // Before its terminal, and only once.
    assert_eq!(l.hold_remote_orphan(&rid(1), REMOTE, 8), Ok(true));
    assert_eq!(
        journal_lines(&dir.0).last().unwrap(),
        &format!(
            r#"{{"type":"orphan_held","rid":"{}","worker":"worker-a","at_ms":8}}"#,
            rid(1)
        )
    );
    assert_eq!(l.hold_remote_orphan(&rid(1), REMOTE, 9), Ok(false));
    // Another worker, a local dispatch, a request never dispatched, an unknown rid.
    assert_eq!(l.hold_remote_orphan(&rid(1), "gpu0", 9), Ok(false));
    assert_eq!(l.hold_remote_orphan(&rid(2), "gpu0", 9), Ok(false));
    assert_eq!(l.hold_remote_orphan(&rid(3), REMOTE, 9), Ok(false));
    assert_eq!(l.hold_remote_orphan(&rid(9), REMOTE, 9), Ok(false));
    l.fail(&rid(1), "worker_lost", None, 10).unwrap();
    l.complete(&rid(2), Completion::default(), 10).unwrap();
    l.fail(&rid(3), "deadline_exceeded", None, 10).unwrap();
    let held = journal_lines(&dir.0)
        .iter()
        .filter(|line| line.contains(r#""type":"orphan_held""#))
        .count();
    assert_eq!(held, 1);
    drop(l);
    let (_l, rec) = open_full(&dir.0, lim, &mixed(), 20);
    assert_eq!(rec.recharged, vec![remote_orphan(1, 5, 1)]);
}

/// P8/Q13: drain intent is journaled for any worker id (also one never registered), re-emitted by every
/// compaction (latest per id) and restored on reopen; undrain clears it.
#[test]
fn p8_drain_intent_survives_two_compactions() {
    let dir = TempDir::new("p8-drain");
    let (mut l, _) = open_full(&dir.0, limits(), &mixed(), 0);
    assert!(l.drain_intents().is_empty());
    l.set_drain(REMOTE, true, 1).unwrap();
    l.set_drain("gpu0", true, 2).unwrap();
    assert_eq!(
        l.drain_intents(),
        BTreeSet::from(["gpu0".to_owned(), REMOTE.to_owned()])
    );
    l.set_drain("gpu0", false, 3).unwrap();
    assert_eq!(
        journal_lines(&dir.0).last().unwrap(),
        r#"{"type":"worker_drain","worker":"gpu0","draining":false,"at_ms":3}"#
    );
    let kept = BTreeSet::from([REMOTE.to_owned()]);
    assert_eq!(l.drain_intents(), kept);
    drop(l);
    for now in [10, 20] {
        let (l, _) = open_full(&dir.0, limits(), &mixed(), now);
        assert_eq!(l.drain_intents(), kept);
        let drains: Vec<Record> = journal_records(&dir.0)
            .into_iter()
            .filter(|r| matches!(r, Record::WorkerDrain { .. }))
            .collect();
        assert_eq!(
            drains,
            [Record::WorkerDrain {
                worker: REMOTE.into(),
                draining: true,
                at_ms: 1
            }]
        );
    }
    let (mut l, _) = open_full(&dir.0, limits(), &mixed(), 30);
    l.set_drain(REMOTE, false, 31).unwrap();
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &mixed(), 40);
    assert!(l.drain_intents().is_empty());
}

/// P5 (m36 design note Q8): each `set_residency` is a new intent (`intent_seq` + 1 per worker, 1 for the first),
/// journaled as `worker_residency` before it returns; replay keeps the last record per worker, every compaction
/// re-emits exactly that one, and the sequence continues after a reopen.
#[test]
fn p5_residency_intents_increment_and_survive_compaction_and_reopen() {
    let dir = TempDir::new("p5-residency");
    let (mut l, _) = open_full(&dir.0, limits(), &mixed(), 0);
    assert!(l.residency_intents().is_empty());
    assert_eq!(l.set_residency(REMOTE, Residency::Resident, 1), Ok(1));
    assert_eq!(
        journal_lines(&dir.0).last().unwrap(),
        r#"{"type":"worker_residency","worker":"worker-a","desired":"resident","intent_seq":1,"at_ms":1}"#
    );
    // The same desire again is still a new intent.
    assert_eq!(l.set_residency(REMOTE, Residency::Resident, 2), Ok(2));
    assert_eq!(l.set_residency(REMOTE, Residency::Unloaded, 3), Ok(3));
    assert_eq!(l.set_residency("worker-b", Residency::Unloaded, 4), Ok(1));
    let latest = BTreeMap::from([
        (REMOTE.to_owned(), (Residency::Unloaded, 3)),
        ("worker-b".to_owned(), (Residency::Unloaded, 1)),
    ]);
    assert_eq!(l.residency_intents(), latest);
    let written = journal_lines(&dir.0)
        .iter()
        .filter(|line| line.contains(r#""type":"worker_residency""#))
        .count();
    assert_eq!(written, 4, "one fsynced record per call");
    drop(l);
    // Two reopens: two compactions, each keeping only the latest record per worker.
    for now in [10, 20] {
        let (l, _) = open_full(&dir.0, limits(), &mixed(), now);
        assert_eq!(l.residency_intents(), latest);
        let kept: Vec<Record> = journal_records(&dir.0)
            .into_iter()
            .filter(|r| matches!(r, Record::WorkerResidency { .. }))
            .collect();
        assert_eq!(
            kept,
            [
                Record::WorkerResidency {
                    worker: REMOTE.into(),
                    desired: Residency::Unloaded,
                    intent_seq: 3,
                    at_ms: 3
                },
                Record::WorkerResidency {
                    worker: "worker-b".into(),
                    desired: Residency::Unloaded,
                    intent_seq: 1,
                    at_ms: 4
                }
            ]
        );
    }
    let (mut l, _) = open_full(&dir.0, limits(), &mixed(), 30);
    assert_eq!(l.set_residency(REMOTE, Residency::Resident, 31), Ok(4));
    drop(l);
    let (l, _) = open_full(&dir.0, limits(), &mixed(), 40);
    assert_eq!(
        l.residency_intents()[REMOTE],
        (Residency::Resident, 4),
        "replay keeps the last record"
    );
}

/// P5 (m36 design note P5, review M36-16): the journal is forward-only. A reader whose `Record` lacks the
/// `worker_residency` variant (a pre-M3.6 `cered`; mirrored here with the same serde shape) refuses the line, the
/// way `Journal::open` turns any unparsable complete line into `InvalidData`.
#[test]
fn p5_a_reader_without_the_residency_variant_refuses_the_line() {
    #[allow(dead_code)]
    #[derive(Debug, serde::Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum PreM36Record {
        Epoch {
            epoch: u64,
            at_ms: u64,
        },
        WorkerDrain {
            worker: String,
            draining: bool,
            at_ms: u64,
        },
    }
    fn parse(line: &str) -> std::io::Result<PreM36Record> {
        serde_json::from_str(line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    let dir = TempDir::new("p5-forward-only");
    let (mut l, _) = open_full(&dir.0, limits(), &mixed(), 0);
    l.set_drain(REMOTE, true, 1).unwrap();
    l.set_residency(REMOTE, Residency::Resident, 2).unwrap();
    drop(l);
    let lines = journal_lines(&dir.0);
    let residency = lines
        .iter()
        .find(|line| line.contains(r#""type":"worker_residency""#))
        .expect("a worker_residency line");
    let drain = lines
        .iter()
        .find(|line| line.contains(r#""type":"worker_drain""#))
        .expect("a worker_drain line");
    assert!(parse(drain).is_ok(), "the mirror reads the older records");
    let err = parse(residency).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("worker_residency"), "{err}");
}

// ---------------------------------------------------------------------------------------------------------
// M5 D7: session model binding

/// A turn of session `s` (salt `salt`) on `model` after `history` (its expected revision is `history.len() / 2`).
fn model_turn(
    l: &Ledger,
    n: u64,
    s: &str,
    salt: &cere_runtime::digest::Salt,
    model: &str,
    history: &[Value],
) -> AcceptRequest {
    let prefix = l.digest_keys().scope_key("a").prefix_digest(salt, history);
    AcceptRequest {
        session: Some(SessionClaim {
            session_id: s.to_owned(),
            model: model.to_owned(),
            expected_revision: history.len() as u64 / 2,
            prefix_digest: prefix,
        }),
        ..req(l, "a", n, None, false)
    }
}

/// Admits and commits one turn on `model`; returns the history after it.
fn commit_turn(
    l: &mut Ledger,
    n: u64,
    s: &str,
    salt: &cere_runtime::digest::Salt,
    model: &str,
    history: &[Value],
    now: u64,
) -> Vec<Value> {
    l.admit(model_turn(l, n, s, salt, model, history), now);
    run(l, n, now);
    let mut committed = history.to_vec();
    committed.push(json!({"role": "user", "content": format!("q{n}")}));
    committed.push(json!({"role": "assistant", "content": format!("a{n}")}));
    let digest = l
        .digest_keys()
        .scope_key("a")
        .prefix_digest(salt, &committed);
    l.complete(
        &rid(n),
        Completion {
            retained: None,
            session_commit: Some(Revision {
                message_count: committed.len() as u64,
                digest,
            }),
        },
        now,
    )
    .unwrap();
    committed
}

fn bound_model(l: &Ledger, s: &str, now: u64) -> Option<String> {
    l.session("a", s, now).unwrap().model
}

/// The claim `turn` builds is refused `model_mismatch` with nothing journaled and no record left.
#[track_caller]
fn assert_mismatch(
    l: &mut Ledger,
    dir: &Path,
    turn: impl FnOnce(&Ledger) -> AcceptRequest,
    now: u64,
) {
    let turn = turn(l);
    let rid = turn.rid.clone();
    let before = journal_records(dir).len();
    let err = l.accept(turn, now).unwrap_err();
    assert_eq!(
        (err.clone(), err.code()),
        (
            LedgerError::Session(SessionError::ModelMismatch),
            "model_mismatch"
        )
    );
    assert_eq!(journal_records(dir).len(), before, "nothing journaled");
    assert!(l.status("a", &rid).is_none(), "no record left");
}

fn session_advances(dir: &Path) -> Vec<Option<String>> {
    journal_records(dir)
        .into_iter()
        .filter_map(|r| match r {
            Record::Terminal(t) => t.session.map(|adv| adv.model),
            _ => None,
        })
        .collect()
}

fn session_snapshots(dir: &Path) -> Vec<Option<String>> {
    journal_records(dir)
        .into_iter()
        .filter_map(|r| match r {
            Record::Session(s) => Some(s.model),
            _ => None,
        })
        .collect()
}

/// D7: a session is unbound until its first commit, which binds it to that turn's model (journaled on the
/// `session_advance`); a turn on another model is refused `model_mismatch` before anything is journaled, while a
/// key reused with another model stays `idempotency_conflict` (the lookup runs first). The binding survives a
/// reopen that replays the advance and the compaction that reopen runs (the `session` snapshot carries it), and
/// a fork inherits it.
#[test]
fn m5_session_binds_at_first_commit_and_survives_replay_and_compaction() {
    let dir = TempDir::new("m5-bind");
    let (mut l, _) = open(&dir.0, 1_000);
    let s = l.create_session("a", 1_000).unwrap();
    let (id, salt) = (s.session_id.clone(), s.salt);
    assert_eq!(bound_model(&l, &id, 1_000), None, "unbound until a commit");
    // A refused turn before the first commit does not bind either.
    l.admit(model_turn(&l, 1, &id, &salt, "x", &[]), 1_001);
    run(&mut l, 1, 1_001);
    l.fail(&rid(1), "backend_error", None, 1_002).unwrap();
    assert_eq!(bound_model(&l, &id, 1_002), None);
    let history = commit_turn(&mut l, 2, &id, &salt, "m", &[], 1_003);
    assert_eq!(bound_model(&l, &id, 1_003).as_deref(), Some("m"));
    assert_eq!(session_advances(&dir.0), vec![Some("m".to_owned())]);
    assert_mismatch(
        &mut l,
        &dir.0,
        |l| model_turn(l, 3, &id, &salt, "x", &history),
        1_004,
    );
    // The idempotency lookup precedes the binding check.
    let mut keyed = model_turn(&l, 4, &id, &salt, "m", &history);
    keyed.key = Some("k".to_owned());
    l.admit(keyed, 1_005);
    let mut reused = model_turn(&l, 5, &id, &salt, "x", &history);
    reused.key = Some("k".to_owned());
    reused.digest = digest(&l, "a", &body("another"));
    assert_eq!(
        l.accept(reused, 1_006),
        Err(LedgerError::IdempotencyConflict)
    );
    run(&mut l, 4, 1_007);
    l.fail(&rid(4), "backend_error", None, 1_007).unwrap();
    let fork = l.fork_session("a", &id, 1, 1_008).unwrap();
    assert_eq!(fork.child.model.as_deref(), Some("m"), "a fork inherits");
    drop(l);

    // Reopen 1 replays the advance, then compacts it into the session snapshot.
    let (mut l, _) = open(&dir.0, 1_100);
    assert_eq!(bound_model(&l, &id, 1_100).as_deref(), Some("m"));
    assert_eq!(
        bound_model(&l, &fork.child.session_id, 1_100).as_deref(),
        Some("m")
    );
    assert!(session_advances(&dir.0).is_empty(), "the reopen compacted");
    assert_eq!(
        session_snapshots(&dir.0),
        vec![Some("m".to_owned()), Some("m".to_owned())]
    );
    assert_mismatch(
        &mut l,
        &dir.0,
        |l| model_turn(l, 6, &id, &salt, "x", &history),
        1_101,
    );
    drop(l);

    // Reopen 2 reads the compacted snapshot.
    let (mut l, _) = open(&dir.0, 1_200);
    assert_eq!(bound_model(&l, &id, 1_200).as_deref(), Some("m"));
    assert_mismatch(
        &mut l,
        &dir.0,
        |l| model_turn(l, 7, &id, &salt, "x", &history),
        1_201,
    );
    commit_turn(&mut l, 8, &id, &salt, "m", &history, 1_202);
    assert_eq!(l.session("a", &id, 1_202).unwrap().revision, 2);
}

/// D7: a pre-M5 journal (no `model` on `session_advance` nor on the `session` snapshot) reads back unbound, and
/// the next commit binds the session to its model.
fn pre_m5_session_binds_at_next_commit(compacted: bool) {
    let dir = TempDir::new(if compacted {
        "m5-pre-compacted"
    } else {
        "m5-pre-advance"
    });
    let (mut l, _) = open(&dir.0, 1_000);
    let s = l.create_session("a", 1_000).unwrap();
    let (id, salt) = (s.session_id.clone(), s.salt);
    let history = commit_turn(&mut l, 1, &id, &salt, "m", &[], 1_001);
    drop(l);
    if compacted {
        drop(open(&dir.0, 1_050));
        assert_eq!(session_snapshots(&dir.0), vec![Some("m".to_owned())]);
    } else {
        assert_eq!(session_advances(&dir.0), vec![Some("m".to_owned())]);
    }
    // Rewrite the journal as a pre-M5 build wrote it.
    let path = dir.0.join(JOURNAL_FILE);
    let text = std::fs::read_to_string(&path).unwrap();
    let old = text.replace(",\"model\":\"m\"", "");
    assert_ne!(old, text);
    std::fs::write(&path, old).unwrap();
    if compacted {
        assert_eq!(session_snapshots(&dir.0), vec![None]);
    } else {
        assert_eq!(session_advances(&dir.0), vec![None]);
    }

    let (mut l, _) = open(&dir.0, 1_100);
    assert_eq!(bound_model(&l, &id, 1_100), None, "reads back unbound");
    assert_eq!(l.session("a", &id, 1_100).unwrap().revision, 1);
    let history = commit_turn(&mut l, 2, &id, &salt, "x", &history, 1_101);
    assert_eq!(bound_model(&l, &id, 1_101).as_deref(), Some("x"));
    assert_mismatch(
        &mut l,
        &dir.0,
        |l| model_turn(l, 3, &id, &salt, "m", &history),
        1_102,
    );
    drop(l);
    let (l, _) = open(&dir.0, 1_200);
    assert_eq!(bound_model(&l, &id, 1_200).as_deref(), Some("x"));
}

#[test]
fn m5_pre_m5_session_advance_reads_unbound_and_binds_at_next_commit() {
    pre_m5_session_binds_at_next_commit(false);
}

#[test]
fn m5_pre_m5_session_snapshot_reads_unbound_and_binds_at_next_commit() {
    pre_m5_session_binds_at_next_commit(true);
}
