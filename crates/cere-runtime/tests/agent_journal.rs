//! OPJ track (m35 design note Q12, row P4): the agent's durable operation journal.
//!
//! Two tests re-execute this test binary as a child process (`std::env::current_exe()`, one named test,
//! `--exact`): the crash child (`CERE_JOURNAL_CRASH_AT=<n>`) and the poison child. Both make the journal's real
//! append path fail through the kernel instead of a production failpoint: `RLIMIT_FSIZE` caps the journal file a
//! few bytes past its current end, so the next append is a genuine short write. The crash child turns the
//! following `SIGXFSZ` into `SIGKILL` (a torn record on disk, no rollback ran); the poison child ignores `SIGXFSZ`
//! so the append sees `EFBIG` and must roll back and poison. The rlimit is process-wide, hence the subprocess.
//! The child entry points are inert (return at once) unless their env var is set.

use cere_runtime::agent::journal::{
    OpError, OpIdentity, OpJournal, OpKey, OpLimits, OpView, OPERATIONS_FILE, SECRET_FILE,
};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Command;

const SUBMIT: &str = "SubmitWork";
const CANCEL: &str = "CancelWork";
const CRASH_AT_ENV: &str = "CERE_JOURNAL_CRASH_AT";
const CRASH_DIR_ENV: &str = "CERE_JOURNAL_CRASH_DIR";
const POISON_DIR_ENV: &str = "CERE_JOURNAL_POISON_DIR";
const WORK_CAP_MS: u64 = 600_000;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cere-opj-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir.join("state"))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
    }
}

fn limits() -> OpLimits {
    OpLimits {
        max_operations: 8,
        retry_horizon_ms: WORK_CAP_MS,
        rpc_timeout_ms: 5_000,
    }
}

fn key(scope: &str, op: &str) -> OpKey {
    OpKey {
        scope_id: scope.to_owned(),
        operation_id: op.to_owned(),
    }
}

fn payload(tag: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = tag
            .wrapping_mul(31)
            .wrapping_add(i as u8)
            .wrapping_add(0x5a);
    }
    out
}

fn identity(tag: u8) -> OpIdentity {
    OpIdentity {
        method: SUBMIT,
        payload_sha256: payload(tag),
        worker_epoch: 1,
        model_generation: 1,
        coordinator_term: 1,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, b| {
        out.push_str(&format!("{b:02x}"));
        out
    })
}

fn open(dir: &Path, now: u64) -> OpJournal {
    OpJournal::open(dir, limits(), now).unwrap().0
}

fn file_bytes(dir: &Path) -> Vec<u8> {
    std::fs::read(dir.join(OPERATIONS_FILE)).unwrap()
}

fn file_lines(dir: &Path) -> usize {
    file_bytes(dir).iter().filter(|&&b| b == b'\n').count()
}

fn all(journal: &OpJournal) -> Vec<OpView> {
    journal.list(None, usize::MAX).0
}

#[test]
fn fresh_accept_mints_a_ticket_and_journals_private_files() {
    let dir = TempDir::new("fresh");
    let (mut j, replay) = OpJournal::open(&dir.0, limits(), 1_000).unwrap();
    assert!(!replay.torn_tail);
    assert_eq!(replay.records, 0);
    let a = j
        .accept(key("s", "op-1"), identity(1), "rid-1", 1, 30_000, 1_000)
        .unwrap();
    assert!(a.fresh);
    assert_eq!(a.ticket.len(), 32);
    assert!(a
        .ticket
        .bytes()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(a.deadline_ms, 31_000);
    let b = j
        .accept(key("s", "op-2"), identity(2), "rid-2", 1, 30_000, 1_000)
        .unwrap();
    assert_ne!(a.ticket, b.ticket, "tickets are random per operation");

    let view = j.get(&key("s", "op-1")).unwrap();
    assert_eq!(view.method, SUBMIT);
    assert_eq!(view.request_id, "rid-1");
    assert_eq!(view.attempt, 1);
    assert_eq!(view.ticket, a.ticket);
    assert_eq!(
        (
            view.worker_epoch,
            view.model_generation,
            view.coordinator_term
        ),
        (1, 1, 1)
    );
    assert!(!view.terminal);
    assert_eq!(view.reason, "");
    assert!(!view.cleanup_acknowledged);
    assert_eq!(view.deadline_ms, 31_000);
    assert!(j.get(&key("s", "op-3")).is_none());

    let mode = |p: PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(dir.0.clone()), 0o700);
    assert_eq!(mode(dir.0.join(OPERATIONS_FILE)), 0o600);
    assert_eq!(mode(dir.0.join(SECRET_FILE)), 0o600);
}

#[test]
fn duplicate_returns_the_original_receipt_and_survives_reopen() {
    let dir = TempDir::new("dedupe");
    let mut j = open(&dir.0, 100);
    let first = j
        .accept(key("s", "op"), identity(7), "rid", 1, 10_000, 100)
        .unwrap();
    assert!(
        first.fresh,
        "the first accept is the one backend side effect"
    );
    let again = j
        .accept(key("s", "op"), identity(7), "rid", 1, 10_000, 200)
        .unwrap();
    assert!(
        !again.fresh,
        "a duplicate must not cause a second backend call"
    );
    assert_eq!(again.ticket, first.ticket);
    assert_eq!(again.deadline_ms, first.deadline_ms);
    assert_eq!(all(&j).len(), 1);
    drop(j);

    let mut j = open(&dir.0, 300);
    let replayed = j
        .accept(key("s", "op"), identity(7), "rid", 1, 10_000, 300)
        .unwrap();
    assert!(
        !replayed.fresh,
        "the agent secret is reused, so the digest re-verifies"
    );
    assert_eq!(replayed.ticket, first.ticket);
    assert_eq!(replayed.deadline_ms, first.deadline_ms);
}

#[test]
fn every_identity_field_change_is_a_conflict() {
    let dir = TempDir::new("conflict");
    let mut j = open(&dir.0, 0);
    let base = identity(3);
    let original = j.accept(key("s", "op"), base, "rid", 1, 10_000, 0).unwrap();
    let variants = [
        OpIdentity {
            method: CANCEL,
            ..base
        },
        OpIdentity {
            payload_sha256: payload(4),
            ..base
        },
        OpIdentity {
            worker_epoch: 2,
            ..base
        },
        OpIdentity {
            model_generation: 2,
            ..base
        },
        OpIdentity {
            coordinator_term: 2,
            ..base
        },
    ];
    for changed in variants {
        match j.accept(key("s", "op"), changed, "rid", 1, 10_000, 10) {
            Err(OpError::Conflict) => {}
            other => panic!("{changed:?}: expected Conflict, got {other:?}"),
        }
    }
    let view = j.get(&key("s", "op")).unwrap();
    assert_eq!(
        view.ticket, original.ticket,
        "a conflict never replaces the op"
    );
    assert_eq!(all(&j).len(), 1);
    assert!(!j.poisoned());
}

#[test]
fn smaller_budget_retry_is_not_a_conflict_and_never_renews_the_deadline() {
    let dir = TempDir::new("deadline");
    let mut j = open(&dir.0, 0);
    let first = j
        .accept(key("s", "op"), identity(1), "rid", 1, 1_000, 100)
        .unwrap();
    assert_eq!(first.deadline_ms, 1_100);
    let smaller = j
        .accept(key("s", "op"), identity(1), "rid", 1, 300, 500)
        .unwrap();
    assert!(!smaller.fresh);
    assert_eq!(smaller.deadline_ms, 1_100);
    let larger = j
        .accept(key("s", "op"), identity(1), "rid", 1, 5_000, 900)
        .unwrap();
    assert!(!larger.fresh);
    assert_eq!(
        larger.deadline_ms, 1_100,
        "a retry never renews the deadline"
    );
    assert_eq!(j.get(&key("s", "op")).unwrap().deadline_ms, 1_100);

    let capped = j
        .accept(
            key("s", "long"),
            identity(2),
            "rid-2",
            1,
            10 * WORK_CAP_MS,
            7,
        )
        .unwrap();
    assert_eq!(
        capped.deadline_ms,
        7 + WORK_CAP_MS,
        "deadline capped at the work cap"
    );
}

#[test]
fn cancel_is_keyed_separately_from_its_submit() {
    let dir = TempDir::new("cancel");
    let mut j = open(&dir.0, 0);
    let submit = j
        .accept(key("s", "op-submit"), identity(1), "rid", 1, 10_000, 0)
        .unwrap();
    let cancel_identity = OpIdentity {
        method: CANCEL,
        ..identity(9)
    };
    let cancel = j
        .accept(key("s", "op-cancel"), cancel_identity, "rid", 1, 10_000, 5)
        .unwrap();
    assert!(cancel.fresh);
    assert_ne!(cancel.ticket, submit.ticket);
    assert_eq!(j.get(&key("s", "op-submit")).unwrap().method, SUBMIT);
    assert_eq!(j.get(&key("s", "op-cancel")).unwrap().method, CANCEL);
    let dup = j
        .accept(key("s", "op-cancel"), cancel_identity, "rid", 1, 10_000, 6)
        .unwrap();
    assert!(!dup.fresh);
    assert_eq!(dup.ticket, cancel.ticket);
    // The same operation id in another scope is another operation.
    let other_scope = j
        .accept(key("t", "op-submit"), identity(1), "rid", 1, 10_000, 7)
        .unwrap();
    assert!(other_scope.fresh);
    assert_ne!(other_scope.ticket, submit.ticket);
}

/// m36 design note Q6/P1: the three lifecycle methods are journaled like SubmitWork/CancelWork and replay after a
/// reopen; a method outside the journaled set stays refused.
#[test]
fn lifecycle_methods_are_journaled_and_replay() {
    let dir = TempDir::new("lifecycle");
    let methods = ["LoadModel", "DrainModel", "UnloadModel"];
    let mut j = open(&dir.0, 0);
    for (i, method) in methods.into_iter().enumerate() {
        let op = format!("residency-1-{i}-1-x");
        let identity = OpIdentity {
            method,
            ..identity(20 + i as u8)
        };
        assert!(
            j.accept(key("_cere_control", &op), identity, &op, 1, 10_000, 1)
                .unwrap()
                .fresh
        );
    }
    let read = OpIdentity {
        method: "GetOperation",
        ..identity(30)
    };
    assert!(j
        .accept(key("_cere_control", "read"), read, "read", 1, 10_000, 1)
        .is_err());
    drop(j);
    let j = open(&dir.0, 2);
    for (i, method) in methods.into_iter().enumerate() {
        let view = j
            .get(&key("_cere_control", &format!("residency-1-{i}-1-x")))
            .unwrap();
        assert_eq!(view.method, method);
    }
}

#[test]
fn full_journal_refuses_and_keeps_replay_protection() {
    let dir = TempDir::new("full");
    let mut j = open(&dir.0, 0);
    let mut tickets = Vec::new();
    for i in 0..8u8 {
        let op = format!("op-{i}");
        tickets.push(
            j.accept(key("s", &op), identity(i), &op, 1, 1_000, 0)
                .unwrap()
                .ticket,
        );
    }
    // A terminal, cleaned-up tombstone inside its retry horizon still counts and is never evicted.
    j.mark_terminal(&key("s", "op-0"), "completed", 10).unwrap();
    j.mark_cleanup(&key("s", "op-0"), 10).unwrap();
    match j.accept(key("s", "op-new"), identity(99), "rid", 1, 1_000, 20) {
        Err(OpError::Full) => {}
        other => panic!("expected Full, got {other:?}"),
    }
    for i in 0..8u8 {
        let op = format!("op-{i}");
        let replay = j
            .accept(key("s", &op), identity(i), &op, 1, 1_000, 20)
            .unwrap();
        assert!(!replay.fresh);
        assert_eq!(replay.ticket, tickets[i as usize]);
    }
    match j.accept(key("s", "op-0"), identity(42), "op-0", 1, 1_000, 20) {
        Err(OpError::Conflict) => {}
        other => panic!("expected Conflict on a retained tombstone, got {other:?}"),
    }
    assert!(!j.poisoned(), "Full is a refusal, not a write failure");
    assert_eq!(all(&j).len(), 8);
    assert!(j.get(&key("s", "op-new")).is_none());
}

#[test]
fn tombstone_drop_rule_boundaries() {
    let dir = TempDir::new("tombstone");
    let lim = limits();
    let mut j = open(&dir.0, 0);
    // A: deadline-bound: terminal early, so deadline + rpc_timeout is the later bound.
    j.accept(key("s", "a"), identity(1), "a", 1, WORK_CAP_MS, 1_000)
        .unwrap();
    j.mark_terminal(&key("s", "a"), "completed", 2_000).unwrap();
    j.mark_cleanup(&key("s", "a"), 2_000).unwrap();
    let a_drop = (1_000 + WORK_CAP_MS + lim.rpc_timeout_ms).max(2_000 + lim.retry_horizon_ms);
    assert_eq!(a_drop, 606_000);
    // B: horizon-bound: terminal late, so terminal + retry_horizon is the later bound.
    j.accept(key("s", "b"), identity(2), "b", 1, 1_000, 1_000)
        .unwrap();
    j.mark_terminal(&key("s", "b"), "completed", 50_000)
        .unwrap();
    j.mark_cleanup(&key("s", "b"), 50_000).unwrap();
    let b_drop = (2_000 + lim.rpc_timeout_ms).max(50_000 + lim.retry_horizon_ms);
    assert_eq!(b_drop, 650_000);
    // C: terminal but never cleanup-acknowledged: kept (it still counts toward the live-op bound).
    j.accept(key("s", "c"), identity(3), "c", 1, 1_000, 1_000)
        .unwrap();
    j.mark_terminal(&key("s", "c"), "cancelled", 1_500).unwrap();
    // D: nonterminal: never dropped.
    j.accept(key("s", "d"), identity(4), "d", 1, 1_000, 1_000)
        .unwrap();

    j.expire(a_drop - 1);
    assert!(
        j.get(&key("s", "a")).is_some(),
        "kept until the drop instant"
    );
    j.expire(a_drop);
    assert!(
        j.get(&key("s", "a")).is_none(),
        "dropped at the drop instant"
    );
    j.expire(b_drop - 1);
    assert!(j.get(&key("s", "b")).is_some());
    j.expire(b_drop);
    assert!(j.get(&key("s", "b")).is_none());
    j.expire(u64::MAX / 2);
    assert!(j.get(&key("s", "c")).is_some());
    assert!(j.get(&key("s", "d")).is_some());

    // After its tombstone is gone the key is unknown: a new accept is new work with a new ticket.
    let renewed = j
        .accept(key("s", "a"), identity(1), "a", 1, 1_000, u64::MAX / 2)
        .unwrap();
    assert!(renewed.fresh);
    drop(j);
    // Replay accepts an op re-accepted after its tombstone's drop instant.
    let j = open(&dir.0, u64::MAX / 2);
    assert_eq!(j.get(&key("s", "a")).unwrap().ticket, renewed.ticket);
    assert!(j.get(&key("s", "b")).is_none());
    assert!(j.get(&key("s", "c")).is_some());
}

#[test]
fn reopen_drops_expired_tombstones() {
    let dir = TempDir::new("reopen-drop");
    let mut j = open(&dir.0, 0);
    j.accept(key("s", "a"), identity(1), "a", 1, 1_000, 0)
        .unwrap();
    j.mark_terminal(&key("s", "a"), "completed", 10).unwrap();
    j.mark_cleanup(&key("s", "a"), 10).unwrap();
    drop(j);
    let j = open(&dir.0, 10 + WORK_CAP_MS - 1);
    assert!(j.get(&key("s", "a")).is_some());
    drop(j);
    let j = open(&dir.0, 10 + WORK_CAP_MS);
    assert!(j.get(&key("s", "a")).is_none());
    assert_eq!(
        file_lines(&dir.0),
        0,
        "the reopen compaction removed it from disk"
    );
}

#[test]
fn restart_marks_previous_epoch_nonterminal_ops_worker_restart() {
    let dir = TempDir::new("restart");
    let mut j = open(&dir.0, 0);
    let epoch = |e: u64, tag: u8| OpIdentity {
        worker_epoch: e,
        ..identity(tag)
    };
    j.accept(key("s", "old-1"), epoch(1, 1), "r1", 1, 1_000, 0)
        .unwrap();
    j.accept(key("s", "old-2"), epoch(1, 2), "r2", 1, 1_000, 0)
        .unwrap();
    j.accept(key("s", "old-done"), epoch(1, 3), "r3", 1, 1_000, 0)
        .unwrap();
    j.mark_terminal(&key("s", "old-done"), "completed", 5)
        .unwrap();
    j.accept(key("s", "current"), epoch(2, 4), "r4", 1, 1_000, 0)
        .unwrap();

    let aborted = j.restart(2, 50).unwrap();
    let ids: Vec<(&str, &str)> = aborted
        .iter()
        .map(|v| (v.key.operation_id.as_str(), v.request_id.as_str()))
        .collect();
    assert_eq!(ids, vec![("old-1", "r1"), ("old-2", "r2")]);
    for view in &aborted {
        assert!(view.terminal);
        assert_eq!(view.reason, "worker_restart");
        assert!(!view.cleanup_acknowledged, "aborting is not cleanup");
    }
    assert_eq!(j.get(&key("s", "old-done")).unwrap().reason, "completed");
    assert!(!j.get(&key("s", "current")).unwrap().terminal);
    assert_eq!(j.live_ops(), 4, "no op is cleanup-acknowledged yet");
    assert!(j.restart(2, 60).unwrap().is_empty(), "idempotent");
    drop(j);

    let mut j = open(&dir.0, 100);
    let old = j.get(&key("s", "old-1")).unwrap();
    assert!(old.terminal);
    assert_eq!(old.reason, "worker_restart");
    // The incarnation-jump path: a later epoch fences the epoch-2 op too.
    let jumped = j.restart(5, 200).unwrap();
    assert_eq!(jumped.len(), 1);
    assert_eq!(jumped[0].key, key("s", "current"));
}

#[test]
fn terminal_and_cleanup_acknowledged_are_separate() {
    let dir = TempDir::new("cleanup");
    let mut j = open(&dir.0, 0);
    j.accept(key("s", "a"), identity(1), "a", 1, 1_000, 0)
        .unwrap();
    j.accept(key("s", "b"), identity(2), "b", 1, 1_000, 0)
        .unwrap();
    assert_eq!(j.live_ops(), 2);
    j.mark_terminal(&key("s", "a"), "cancelled", 10).unwrap();
    let a = j.get(&key("s", "a")).unwrap();
    assert!(a.terminal && !a.cleanup_acknowledged);
    assert_eq!(j.live_ops(), 2, "terminal alone still holds a live-op slot");
    j.mark_cleanup(&key("s", "a"), 20).unwrap();
    let a = j.get(&key("s", "a")).unwrap();
    assert!(a.terminal && a.cleanup_acknowledged);
    assert_eq!(a.reason, "cancelled");
    assert_eq!(j.live_ops(), 1);

    // A second terminal keeps the first reason; repeated cleanup is a no-op.
    j.mark_terminal(&key("s", "a"), "completed", 30).unwrap();
    j.mark_cleanup(&key("s", "a"), 30).unwrap();
    assert_eq!(j.get(&key("s", "a")).unwrap().reason, "cancelled");
    match j.mark_terminal(&key("s", "nope"), "completed", 30) {
        Err(OpError::Unknown) => {}
        other => panic!("expected Unknown, got {other:?}"),
    }
    match j.mark_cleanup(&key("s", "nope"), 30) {
        Err(OpError::Unknown) => {}
        other => panic!("expected Unknown, got {other:?}"),
    }
    drop(j);
    let j = open(&dir.0, 40);
    let a = j.get(&key("s", "a")).unwrap();
    assert!(a.terminal && a.cleanup_acknowledged);
    assert_eq!(a.reason, "cancelled");
    assert_eq!(j.live_ops(), 1);
}

/// m36 design note Q4: only SubmitWork ops count toward the live-op bound; a lifecycle or `CancelWork` op,
/// nonterminal or terminal without cleanup, never does (also after replay).
#[test]
fn only_submit_work_ops_count_as_live() {
    let dir = TempDir::new("live-methods");
    let mut j = open(&dir.0, 0);
    let methods = [
        SUBMIT,
        CANCEL,
        "LoadModel",
        "DrainModel",
        "UnloadModel",
        SUBMIT,
    ];
    for (tag, method) in (1u8..).zip(methods) {
        let op = format!("op-{tag}");
        let identity = OpIdentity {
            method,
            ..identity(tag)
        };
        j.accept(key("s", &op), identity, &op, 1, 1_000, 0).unwrap();
    }
    assert_eq!(j.live_ops(), 2, "nonterminal lifecycle and cancel ops");
    for tag in 2..=5 {
        j.mark_terminal(&key("s", &format!("op-{tag}")), "cancelled", 10)
            .unwrap();
    }
    assert_eq!(j.live_ops(), 2, "terminal, not cleanup-acknowledged");
    drop(j);
    let j = open(&dir.0, 20);
    assert_eq!(j.live_ops(), 2, "after replay");
}

#[test]
fn list_pages_in_scope_then_operation_order() {
    let dir = TempDir::new("list");
    let mut j = open(&dir.0, 0);
    let keys = [("b", "1"), ("a", "2"), ("a", "10"), ("c", "0"), ("a", "1")];
    for (i, (s, o)) in keys.iter().enumerate() {
        j.accept(key(s, o), identity(i as u8), "rid", 1, 1_000, 0)
            .unwrap();
    }
    let order = |views: &[OpView]| -> Vec<(String, String)> {
        views
            .iter()
            .map(|v| (v.key.scope_id.clone(), v.key.operation_id.clone()))
            .collect()
    };
    let (page, more) = j.list(None, 2);
    assert!(more);
    assert_eq!(
        order(&page),
        vec![("a".into(), "1".into()), ("a".into(), "10".into())]
    );
    let (page, more) = j.list(Some(&page[1].key), 2);
    assert!(more);
    assert_eq!(
        order(&page),
        vec![("a".into(), "2".into()), ("b".into(), "1".into())]
    );
    let (page, more) = j.list(Some(&page[1].key), 2);
    assert!(!more);
    assert_eq!(order(&page), vec![("c".into(), "0".into())]);
    let (page, more) = j.list(Some(&key("c", "0")), 2);
    assert!(page.is_empty() && !more);
    // `after` need not be a journaled key.
    let (page, _) = j.list(Some(&key("a", "3")), 1);
    assert_eq!(order(&page), vec![("b".into(), "1".into())]);
}

#[test]
fn torn_last_line_is_cut_and_reported() {
    let dir = TempDir::new("torn");
    let mut j = open(&dir.0, 0);
    j.accept(key("s", "a"), identity(1), "a", 1, 1_000, 0)
        .unwrap();
    drop(j);
    let good = file_bytes(&dir.0);
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(dir.0.join(OPERATIONS_FILE))
        .unwrap();
    f.write_all(b"{\"type\":\"accepted\",\"scope_id\":\"s\",\"oper")
        .unwrap();
    drop(f);
    let (j, replay) = OpJournal::open(&dir.0, limits(), 10).unwrap();
    assert!(replay.torn_tail);
    assert_eq!(replay.records, 1);
    assert!(j.get(&key("s", "a")).is_some());
    drop(j);
    assert_eq!(file_bytes(&dir.0), good);
    let (_, replay) = OpJournal::open(&dir.0, limits(), 20).unwrap();
    assert!(!replay.torn_tail);
}

#[test]
fn corrupt_complete_line_or_unknown_field_fails_closed() {
    let dir = TempDir::new("corrupt");
    let mut j = open(&dir.0, 0);
    j.accept(key("s", "a"), identity(1), "a", 1, 1_000, 0)
        .unwrap();
    drop(j);
    let good = String::from_utf8(file_bytes(&dir.0)).unwrap();
    let extra = good.replacen(
        "{\"type\":\"accepted\",",
        "{\"type\":\"accepted\",\"x\":1,",
        1,
    );
    assert_ne!(extra, good);
    std::fs::write(dir.0.join(OPERATIONS_FILE), &extra).unwrap();
    let err = OpJournal::open(&dir.0, limits(), 10).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    std::fs::write(dir.0.join(OPERATIONS_FILE), format!("{good}not json\n")).unwrap();
    let err = OpJournal::open(&dir.0, limits(), 10).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn secret_is_never_silently_replaced() {
    let dir = TempDir::new("secret");
    drop(open(&dir.0, 0));
    let secret = std::fs::read(dir.0.join(SECRET_FILE)).unwrap();
    drop(open(&dir.0, 0));
    assert_eq!(std::fs::read(dir.0.join(SECRET_FILE)).unwrap(), secret);
    std::fs::write(dir.0.join(SECRET_FILE), b"not-a-secret").unwrap();
    let err = OpJournal::open(&dir.0, limits(), 0).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(
        std::fs::read(dir.0.join(SECRET_FILE)).unwrap(),
        b"not-a-secret"
    );
}

#[test]
fn compaction_keeps_live_and_retained_records_and_replays_identically() {
    let dir = TempDir::new("compact");
    let mut j = open(&dir.0, 0);
    for i in 0..6u8 {
        let op = format!("op-{i}");
        j.accept(key("s", &op), identity(i), &op, u64::from(i), 1_000, 0)
            .unwrap();
    }
    j.mark_terminal(&key("s", "op-0"), "completed", 10).unwrap();
    j.mark_cleanup(&key("s", "op-0"), 10).unwrap();
    j.mark_terminal(&key("s", "op-1"), "cancelled", 20).unwrap();
    j.mark_cleanup(&key("s", "op-2"), 30).unwrap();
    j.mark_terminal(&key("s", "op-3"), "failed", 2 * WORK_CAP_MS)
        .unwrap();
    j.mark_cleanup(&key("s", "op-3"), 2 * WORK_CAP_MS).unwrap();
    let before = all(&j);
    let live = j.live_ops();
    drop(j);

    let j = open(&dir.0, 100);
    assert_eq!(all(&j), before, "replay after compaction is identical");
    assert_eq!(j.live_ops(), live);
    drop(j);
    let j = open(&dir.0, 200);
    assert_eq!(all(&j), before, "a second compaction changes nothing");
    drop(j);

    // op-0's tombstone is due at 10 + horizon: the reopen compaction drops exactly it.
    let j = open(&dir.0, 10 + WORK_CAP_MS);
    let expected: Vec<OpView> = before
        .iter()
        .filter(|v| v.key.operation_id != "op-0")
        .cloned()
        .collect();
    assert_eq!(all(&j), expected);
    // 5 accepted + op-1 terminal + op-2 cleanup + op-3 terminal + op-3 cleanup.
    assert_eq!(file_lines(&dir.0), 9);
}

#[test]
fn expire_compacts_a_grown_file() {
    let dir = TempDir::new("expire-compact");
    let lim = OpLimits {
        max_operations: 4,
        ..limits()
    };
    let (mut j, _) = OpJournal::open(&dir.0, lim, 0).unwrap();
    j.accept(key("s", "keep"), identity(0), "keep", 1, 1_000, 0)
        .unwrap();
    let mut now = 0;
    for i in 1..=4u8 {
        let op = format!("op-{i}");
        j.accept(key("s", &op), identity(i), &op, 1, 1_000, now)
            .unwrap();
        j.mark_terminal(&key("s", &op), "completed", now).unwrap();
        j.mark_cleanup(&key("s", &op), now).unwrap();
        now += WORK_CAP_MS;
        j.expire(now);
        assert!(j.get(&key("s", &op)).is_none());
    }
    // 1 + 4 x 3 = 13 lines appended against a bound of 2 x 0 + 4: expire rewrote the file to the one live op.
    assert!(file_lines(&dir.0) < 13, "lines = {}", file_lines(&dir.0));
    let survivors = all(&j);
    drop(j);
    let j = OpJournal::open(&dir.0, lim, now).unwrap().0;
    assert_eq!(all(&j), survivors);
    assert_eq!(survivors.len(), 1);
}

#[test]
fn file_never_contains_the_raw_payload_digest() {
    let dir = TempDir::new("no-raw");
    let mut j = open(&dir.0, 0);
    let mut digests = Vec::new();
    for i in 0..4u8 {
        let op = format!("op-{i}");
        let id = OpIdentity {
            method: if i % 2 == 0 { SUBMIT } else { CANCEL },
            ..identity(i)
        };
        digests.push(id.payload_sha256);
        j.accept(key("scope", &op), id, &op, 1, 1_000, 0).unwrap();
    }
    j.mark_terminal(&key("scope", "op-0"), "completed", 5)
        .unwrap();
    drop(j);
    drop(open(&dir.0, 10)); // compaction output too
    let bytes = file_bytes(&dir.0);
    let text = String::from_utf8(bytes.clone())
        .unwrap()
        .to_ascii_lowercase();
    for d in &digests {
        assert!(!text.contains(&hex(d)), "raw payload_sha256 hex on disk");
        assert!(
            !bytes.windows(32).any(|w| w == d),
            "raw payload_sha256 bytes on disk"
        );
        // Nor a long fragment of it.
        assert!(!text.contains(&hex(&d[..8])));
    }
}

#[test]
fn write_failure_poisons() {
    if std::env::var_os(POISON_DIR_ENV).is_some() {
        return;
    }
    let dir = TempDir::new("poison");
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["poison_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(POISON_DIR_ENV, &dir.0)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed: {:?}\n{stdout}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("POISON-CHILD-OK"), "{stdout}");
    // The failed append was rolled back: no torn tail, the refused op is absent, the rest replays.
    let (j, replay) = OpJournal::open(&dir.0, limits(), 100).unwrap();
    assert!(!replay.torn_tail);
    assert!(j.get(&key("s", "a")).unwrap().terminal);
    assert!(j.get(&key("s", "b")).is_some());
    assert!(j.get(&key("s", "c")).is_none());
    assert!(
        !j.poisoned(),
        "poison is per process; the reopened journal is sound"
    );
}

/// Child of [`write_failure_poisons`]; inert unless `CERE_JOURNAL_POISON_DIR` is set.
#[test]
fn poison_child() {
    let Some(dir) = std::env::var_os(POISON_DIR_ENV).map(PathBuf::from) else {
        return;
    };
    let mut j = open(&dir, 0);
    j.accept(key("s", "a"), identity(1), "a", 1, 1_000, 0)
        .unwrap();
    j.accept(key("s", "b"), identity(2), "b", 1, 1_000, 0)
        .unwrap();
    j.mark_terminal(&key("s", "a"), "completed", 1).unwrap();
    // SIGXFSZ ignored: the append past the limit gets a short write, then EFBIG.
    limit_file_size(&dir, 10, libc::SIG_IGN);
    match j.accept(key("s", "c"), identity(3), "c", 1, 1_000, 2) {
        Err(OpError::Io(_)) => {}
        other => panic!("expected Io, got {other:?}"),
    }
    assert!(j.poisoned());
    assert!(j.get(&key("s", "c")).is_none(), "state follows the disk");
    for result in [
        j.accept(key("s", "a"), identity(1), "a", 1, 1_000, 3)
            .map(|_| ()),
        j.accept(key("s", "d"), identity(4), "d", 1, 1_000, 3)
            .map(|_| ()),
        j.mark_terminal(&key("s", "b"), "completed", 3),
        j.mark_cleanup(&key("s", "a"), 3),
        j.restart(9, 3).map(|_| ()),
    ] {
        match result {
            Err(OpError::Poisoned) => {}
            other => panic!("expected Poisoned, got {other:?}"),
        }
    }
    assert!(j.get(&key("s", "a")).is_some(), "reads still served");
    let mut stdout = std::io::stdout();
    stdout.write_all(b"POISON-CHILD-OK\n").unwrap();
    stdout.flush().unwrap();
}

#[test]
fn crash_mid_append_cuts_the_torn_tail_and_keeps_every_acknowledged_receipt() {
    if std::env::var_os(CRASH_AT_ENV).is_some() {
        return;
    }
    let dir = TempDir::new("crash");
    let crash_at = 5;
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["crash_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(CRASH_AT_ENV, crash_at.to_string())
        .env(CRASH_DIR_ENV, &dir.0)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.signal(),
        Some(libc::SIGKILL),
        "child must die by SIGKILL mid-append: {:?}\n{stdout}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let acked: Vec<(String, String)> = stdout
        .lines()
        // libtest prints `test crash_child ... ` without a newline before the first ACK.
        .filter_map(|l| l.split_once("ACK ").map(|(_, rest)| rest))
        .map(|l| {
            let (op, ticket) = l.split_once(' ').unwrap();
            (op.to_owned(), ticket.to_owned())
        })
        .collect();
    assert_eq!(acked.len(), crash_at, "{stdout}");
    let raw = file_bytes(&dir.0);
    assert_ne!(raw.last(), Some(&b'\n'), "the kill left a torn record");

    let (j, replay) = OpJournal::open(&dir.0, limits(), 1_000).unwrap();
    assert!(replay.torn_tail);
    for (op, ticket) in &acked {
        let view = j.get(&key("s", op)).unwrap();
        assert_eq!(&view.ticket, ticket);
    }
    assert!(j.get(&key("s", "op-0")).unwrap().terminal);
    assert!(j.get(&key("s", &format!("op-{crash_at}"))).is_none());
    assert_eq!(all(&j).len(), crash_at);
    drop(j);
    assert_eq!(file_bytes(&dir.0).last(), Some(&b'\n'));
    let (_, replay) = OpJournal::open(&dir.0, limits(), 2_000).unwrap();
    assert!(!replay.torn_tail);
}

/// Child of the crash test; inert unless `CERE_JOURNAL_CRASH_AT` is set. Acknowledges `n` operations (one
/// `ACK <op> <ticket>` line each, flushed after the append returned), then crashes inside append `n + 1`.
#[test]
fn crash_child() {
    let Some(n) = std::env::var(CRASH_AT_ENV).ok() else {
        return;
    };
    let n: usize = n.parse().unwrap();
    let dir = PathBuf::from(std::env::var_os(CRASH_DIR_ENV).unwrap());
    let mut j = open(&dir, 0);
    let mut stdout = std::io::stdout();
    for i in 0..n {
        let op = format!("op-{i}");
        let a = j
            .accept(key("s", &op), identity(i as u8), &op, 1, 1_000, 0)
            .unwrap();
        stdout
            .write_all(format!("ACK {op} {}\n", a.ticket).as_bytes())
            .unwrap();
        stdout.flush().unwrap();
    }
    j.mark_terminal(&key("s", "op-0"), "completed", 1).unwrap();
    limit_file_size(&dir, 10, sigkill_self as libc::sighandler_t);
    let op = format!("op-{n}");
    let _ = j.accept(key("s", &op), identity(n as u8), &op, 1, 1_000, 2);
    panic!("the append past the file-size limit must have killed this process");
}

extern "C" fn sigkill_self(_: libc::c_int) {
    // SAFETY: `raise` is async-signal-safe.
    unsafe {
        libc::raise(libc::SIGKILL);
    }
}

/// Caps this process's file size `extra` bytes past the journal's current end and installs `on_xfsz` for
/// `SIGXFSZ`, so the next append is a short write of `extra` bytes followed by `SIGXFSZ` / `EFBIG`.
fn limit_file_size(dir: &Path, extra: u64, on_xfsz: libc::sighandler_t) {
    let len = std::fs::metadata(dir.join(OPERATIONS_FILE)).unwrap().len();
    let limit = libc::rlimit {
        rlim_cur: len + extra,
        rlim_max: libc::RLIM_INFINITY,
    };
    // SAFETY: plain syscalls on a valid struct and a valid handler; this child process is single-threaded
    // with respect to the journal.
    unsafe {
        assert_ne!(libc::signal(libc::SIGXFSZ, on_xfsz), libc::SIG_ERR);
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &limit), 0);
    }
}
