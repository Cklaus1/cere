//! Review finding 2: a request that ends before it was ever dispatched to the backend (queue full, queued
//! deadline, shutdown, worker not ready) releases its Idempotency-Key, so a retry under the same key runs as a
//! new request, in memory, after journal replay and after compaction. A dispatched request, also as a prior-epoch
//! orphan (it may have reached the backend), keeps the key bound. The ledger runs with one configured worker, so
//! admissions are M3 (`dispatch_journaled`) and compaction re-emits worker epochs and `dispatched` records.

use cere_runtime::digest::{DigestKeys, RequestDigestInput};
use cere_runtime::ledger::{
    AcceptOutcome, AcceptRequest, Ledger, LedgerError, LedgerLimits, WorkerTarget,
};
use cere_runtime::session::SessionLimits;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const WORKER: &str = "gpu0";
const WORKER_URL: &str = "http://127.0.0.1:30000";

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cere-rebind-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir.join("state"))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
    }
}

fn limits(compaction_slack: u64) -> LedgerLimits {
    LedgerLimits {
        max_records: 16,
        terminal_retention_ms: 1_000,
        max_retained_bytes: 64,
        compaction_slack,
        sessions: SessionLimits {
            max_sessions_per_scope: 2,
            session_ttl_ms: 100_000,
            max_revisions_per_session: 8,
        },
        evict_terminal_when_full: false,
    }
}

fn worker() -> WorkerTarget {
    WorkerTarget::Local {
        base_url: WORKER_URL.to_owned(),
    }
}

fn open(dir: &Path, slack: u64, now: u64) -> Ledger {
    let keys = Arc::new(DigestKeys::open(dir).unwrap());
    let workers = BTreeMap::from([(WORKER.to_owned(), worker())]);
    Ledger::open(limits(slack), keys, dir, &workers, now)
        .unwrap()
        .0
}

fn rid(epoch: u64, n: u64) -> String {
    format!("cere-{epoch:020}-{n:020}-0123456789abcdef")
}

fn req(ledger: &Ledger, rid: String) -> AcceptRequest {
    let body =
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8});
    let digest = ledger
        .digest_keys()
        .scope_key("a")
        .request_digest(&RequestDigestInput {
            request: &body,
            session_id: None,
            expected_revision: None,
            served_model: "m",
            deadline_ms: 1_000,
            config_hash: "cfg",
            surface: None,
        });
    AcceptRequest {
        scope: "a".into(),
        rid,
        key: Some("k1".into()),
        digest,
        stream: false,
        surface: None,
        session: None,
    }
}

#[test]
fn l2_undispatched_rejections_release_the_key() {
    // queue_full (accepted), queued deadline (queued), worker_not_ready / shutdown (queued) all end pre-dispatch.
    let dir = TempDir::new("undispatched");
    let mut l = open(&dir.0, 1_024, 0);
    let mut n = 0;
    for (queued, reason) in [
        (false, "queue_full"),
        (true, "deadline_exceeded"),
        (true, "worker_not_ready"),
    ] {
        n += 1;
        assert_eq!(l.accept(req(&l, rid(1, n)), n), Ok(AcceptOutcome::Accepted));
        if queued {
            l.mark_queued(&rid(1, n), n).unwrap();
        }
        l.fail(&rid(1, n), reason, None, n).unwrap();
    }
    n += 1;
    assert_eq!(l.accept(req(&l, rid(1, n)), n), Ok(AcceptOutcome::Accepted));
    l.mark_queued(&rid(1, n), n).unwrap();
    l.cancel(&rid(1, n), "shutdown", n).unwrap();
    // The retry under the same key is a new request; the old rids keep their status.
    assert_eq!(
        l.accept(req(&l, rid(1, 9)), 10),
        Ok(AcceptOutcome::Accepted)
    );
    assert!(l.status("a", &rid(1, 1)).is_some());
}

#[test]
fn l2_released_key_stays_released_after_replay_and_compaction() {
    let dir = TempDir::new("replay");
    {
        let mut l = open(&dir.0, 1_024, 0);
        assert_eq!(l.accept(req(&l, rid(1, 1)), 1), Ok(AcceptOutcome::Accepted));
        l.fail(&rid(1, 1), "queue_full", None, 2).unwrap();
    }
    // Replay of the raw journal (this open then compacts to live state).
    {
        let mut l = open(&dir.0, 1_024, 3);
        assert_eq!(l.accept(req(&l, rid(2, 1)), 4), Ok(AcceptOutcome::Accepted));
        l.fail(&rid(2, 1), "queue_full", None, 5).unwrap();
    }
    // Replay of a compacted journal, then a slack-0 ledger that compacts on every write.
    let mut l = open(&dir.0, 0, 6);
    assert_eq!(l.accept(req(&l, rid(3, 1)), 7), Ok(AcceptOutcome::Accepted));
    l.fail(&rid(3, 1), "queue_full", None, 8).unwrap();
    assert_eq!(l.accept(req(&l, rid(3, 2)), 9), Ok(AcceptOutcome::Accepted));
    // Within retention a dispatched request's key is bound and never answers with a rerun.
    l.mark_queued(&rid(3, 2), 10).unwrap();
    l.mark_running(&rid(3, 2), 10).unwrap();
    let epoch = l.worker_epochs()[WORKER];
    l.dispatched(&rid(3, 2), WORKER, epoch, &worker(), 10)
        .unwrap();
    l.fail(&rid(3, 2), "backend_error", None, 11).unwrap();
    assert!(matches!(
        l.accept(req(&l, rid(3, 3)), 12),
        Err(LedgerError::ResponseNotRetained { .. })
    ));
    drop(l);
    let l = open(&dir.0, 1_024, 13);
    assert!(matches!(
        l.status("a", &rid(3, 1)).map(|s| s.reason),
        Some(Some(r)) if r == "queue_full"
    ));
    let mut l = l;
    assert!(matches!(
        l.accept(req(&l, rid(4, 1)), 14),
        Err(LedgerError::ResponseNotRetained { .. })
    ));
}

#[test]
fn l2_recovered_dispatched_orphan_keeps_the_key() {
    // A prior-epoch request without a terminal that was dispatched may have reached the backend: its key stays
    // bound (one never dispatched releases it: core_ledger.rs J2).
    let dir = TempDir::new("orphan");
    {
        let mut l = open(&dir.0, 1_024, 0);
        assert_eq!(l.accept(req(&l, rid(1, 1)), 1), Ok(AcceptOutcome::Accepted));
        l.mark_queued(&rid(1, 1), 1).unwrap();
        l.mark_running(&rid(1, 1), 1).unwrap();
        let epoch = l.worker_epochs()[WORKER];
        l.dispatched(&rid(1, 1), WORKER, epoch, &worker(), 1)
            .unwrap();
    }
    let mut l = open(&dir.0, 1_024, 2);
    assert!(matches!(
        l.accept(req(&l, rid(2, 1)), 3),
        Err(LedgerError::ResponseNotRetained { .. })
    ));
}
