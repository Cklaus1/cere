//! M4 P1 (m4 design note Q2, Q3, Q5-Q7; design-m4 design note §2): the durable session owner. The first grant of a
//! session's turn pins the session at owner generation 1, journaled in that grant's `dispatched` record; every owner
//! change (a grant-time re-pin, an operator migration) bumps the generation by 1 and is journaled before it applies,
//! so worker and generation survive restarts and compactions and the generation never decreases. A turn commits
//! only at the owner generation it holds (the stale-owner fence). The ledger runs with two configured local
//! workers, so admissions are M3 (`dispatch_journaled`).

use cere_runtime::digest::{Digest, DigestKeys, RequestDigestInput};
use cere_runtime::journal::{OwnerRecord, Record, SessionOwnerRecord, JOURNAL_FILE};
use cere_runtime::ledger::{
    AcceptOutcome, AcceptRequest, Completion, HeldOwner, Ledger, LedgerError, LedgerLimits,
    MigrateError, MigrateRequest, Migrated, OwnerView, Recovery, RequestState, SessionClaim,
    WorkerCheck, WorkerTarget, REASON_SESSION_OWNER_SUPERSEDED,
};
use cere_runtime::session::{Revision, SessionError, SessionLimits};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const URL0: &str = "http://127.0.0.1:30000";
const URL1: &str = "http://127.0.0.1:30001";

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cere-owner-{tag}-{}", std::process::id()));
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
        max_records: 64,
        terminal_retention_ms: 1_000,
        max_retained_bytes: 64,
        compaction_slack: 1_024,
        sessions: SessionLimits {
            max_sessions_per_scope: 4,
            session_ttl_ms: 100_000,
            max_revisions_per_session: 8,
        },
        evict_terminal_when_full: false,
    }
}

/// Slack 1: a write compacts first once the journal holds more than `2 × compacted + 1` lines.
fn slack1() -> LedgerLimits {
    LedgerLimits {
        compaction_slack: 1,
        ..limits()
    }
}

fn local(url: &str) -> WorkerTarget {
    WorkerTarget::Local {
        base_url: url.into(),
    }
}

fn url(worker: &str) -> &'static str {
    match worker {
        "gpu0" => URL0,
        "gpu1" => URL1,
        other => panic!("no URL for {other}"),
    }
}

fn two() -> BTreeMap<String, WorkerTarget> {
    ["gpu0", "gpu1"]
        .into_iter()
        .map(|id| (id.to_owned(), local(url(id))))
        .collect()
}

fn try_open(dir: &Path, limits: LedgerLimits, now: u64) -> Result<(Ledger, Recovery), LedgerError> {
    let keys = Arc::new(DigestKeys::open(dir).unwrap());
    Ledger::open(limits, keys, dir, &two(), now)
}

fn open(dir: &Path, limits: LedgerLimits, now: u64) -> Ledger {
    try_open(dir, limits, now).unwrap().0
}

fn rid(n: u64) -> String {
    format!("cere-{:020}-{:020}-0123456789abcdef", 1, n)
}

/// A session-less streamed request `n` of scope `a`.
fn req(ledger: &Ledger, n: u64) -> AcceptRequest {
    let body =
        json!({"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8});
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
        rid: rid(n),
        key: None,
        digest,
        stream: true,
        surface: None,
        session: None,
    }
}

fn admit(l: &mut Ledger, r: AcceptRequest, now: u64) {
    assert_eq!(l.accept(r, now), Ok(AcceptOutcome::Accepted));
}

fn run(l: &mut Ledger, n: u64, now: u64) {
    l.mark_queued(&rid(n), now).unwrap();
    l.mark_running(&rid(n), now).unwrap();
}

/// A session history of `count` messages (revision `r` covers `2r` of them).
fn msgs(count: u64) -> Vec<Value> {
    (0..count)
        .map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": format!("m{i}")}))
        .collect()
}

/// Admits rid `n` as a turn of `session` at its current revision and runs it.
fn begin_turn(l: &mut Ledger, n: u64, session: &str, now: u64) {
    let view = l.session("a", session, now).unwrap();
    let prefix = l
        .digest_keys()
        .scope_key("a")
        .prefix_digest(&view.salt, &msgs(view.current.message_count));
    let mut r = req(l, n);
    r.session = Some(SessionClaim {
        session_id: session.into(),
        model: "m".to_owned(),
        expected_revision: view.revision,
        prefix_digest: prefix,
    });
    admit(l, r, now);
    run(l, n, now);
}

/// Journals rid `n`'s dispatch to `worker` at its current epoch.
fn dispatch(l: &mut Ledger, n: u64, worker: &str, now: u64) -> Result<(), LedgerError> {
    let epoch = l.worker_epochs()[worker];
    l.dispatched(&rid(n), worker, epoch, &local(url(worker)), now)
}

/// Completes rid `n`, committing the next revision of `session` (two more messages).
fn commit(l: &mut Ledger, n: u64, session: &str, now: u64) -> Result<(), LedgerError> {
    let view = l.session("a", session, now).unwrap();
    let count = view.current.message_count + 2;
    let entry = Revision {
        message_count: count,
        digest: l
            .digest_keys()
            .scope_key("a")
            .prefix_digest(&view.salt, &msgs(count)),
    };
    let completion = Completion {
        retained: None,
        session_commit: Some(entry),
    };
    l.complete(&rid(n), completion, now)
}

/// One whole turn of `session` granted on `worker`.
fn turn_on(l: &mut Ledger, n: u64, session: &str, worker: &str, now: u64) {
    begin_turn(l, n, session, now);
    dispatch(l, n, worker, now).unwrap();
    commit(l, n, session, now).unwrap();
}

fn owner(l: &Ledger, session: &str, now: u64) -> (Option<String>, u64) {
    let o = l.session_owner(session, now).unwrap();
    (o.worker, o.owner_generation)
}

fn owned(worker: &str, owner_generation: u64) -> (Option<String>, u64) {
    (Some(worker.to_owned()), owner_generation)
}

fn held(worker: Option<&str>, owner_generation: u64) -> Option<HeldOwner> {
    Some(HeldOwner {
        worker: worker.map(str::to_owned),
        owner_generation,
    })
}

fn owner_view(
    session: &str,
    worker: Option<&str>,
    owner_generation: u64,
    revision: u64,
) -> OwnerView {
    OwnerView {
        session_id: session.to_owned(),
        worker: worker.map(str::to_owned),
        owner_generation,
        revision,
    }
}

fn journal_records(dir: &Path) -> Vec<Record> {
    journal_lines(dir)
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn journal_lines(dir: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(dir.join(JOURNAL_FILE)).unwrap();
    assert!(text.ends_with('\n'));
    text.lines().map(str::to_owned).collect()
}

fn last_line(dir: &Path) -> String {
    journal_lines(dir).pop().unwrap()
}

/// A migration's worker check.
type Check<'a> = &'a dyn Fn(&str) -> WorkerCheck;

/// The workers a migration may see: `eligible` ones (Ready), `ineligible` ones (configured, not Ready); any other
/// id is unknown.
fn checker(
    eligible: &'static [&'static str],
    ineligible: &'static [&'static str],
) -> impl Fn(&str) -> WorkerCheck {
    move |w: &str| {
        if eligible.contains(&w) {
            WorkerCheck::Eligible(0)
        } else if ineligible.contains(&w) {
            WorkerCheck::Ineligible(0)
        } else {
            WorkerCheck::Unknown
        }
    }
}

fn migrate(
    l: &mut Ledger,
    session: &str,
    to: &str,
    expected: u64,
    force: bool,
    check: Check<'_>,
    now: u64,
) -> Result<Migrated, MigrateError> {
    let request = MigrateRequest {
        session_id: session,
        to_worker: to,
        expected_owner_generation: expected,
        force,
    };
    l.migrate_session(request, check, &|_: &str| Some(0), now)
}

fn migrated(
    session: &str,
    worker: &str,
    owner_generation: u64,
    revision: u64,
    previous: Option<&str>,
    superseded: Option<String>,
) -> Migrated {
    Migrated {
        session_id: session.to_owned(),
        scope: "a".into(),
        worker: worker.into(),
        owner_generation,
        revision,
        previous: previous.map(str::to_owned),
        superseded,
    }
}

/// Admits session-less requests from rid `next` until the journal's next write compacts first (`compacted`: its
/// line count right after its last compaction, e.g. an open; slack 1). Returns the next free rid number.
fn fill_until_compaction_due(
    l: &mut Ledger,
    dir: &Path,
    compacted: usize,
    mut next: u64,
    now: u64,
) -> u64 {
    while journal_lines(dir).len() <= 2 * compacted + 1 {
        let r = req(l, next);
        admit(l, r, now);
        next += 1;
    }
    next
}

/// Q2/Q3(a): the first grant of a session's turn pins it at generation 1, journaled in that `dispatched` record
/// (`session_owner`; the worker is the record's own); a later grant on the same worker changes nothing and
/// journals no `session_owner`. The entry holds the generation it was granted at.
#[test]
fn p1_first_pin_is_generation_one_journaled_in_dispatched() {
    let dir = TempDir::new("first-pin");
    let mut l = open(&dir.0, limits(), 0);
    let s = l.create_session("a", 1).unwrap();
    let sid = s.session_id.clone();
    assert_eq!((s.pinned_worker, s.owner_generation), (None, 0));
    assert_eq!(l.session_owner(&sid, 1), Ok(owner_view(&sid, None, 0, 0)));
    begin_turn(&mut l, 1, &sid, 2);
    assert_eq!(l.held_owner(&rid(1)), held(None, 0));
    dispatch(&mut l, 1, "gpu0", 3).unwrap();
    assert_eq!(
        last_line(&dir.0),
        format!(
            r#"{{"type":"dispatched","rid":"{}","worker":"gpu0","worker_epoch":1,"base_url":"{URL0}","session_owner":{{"session_id":"{sid}","owner_generation":1}},"at_ms":3}}"#,
            rid(1)
        )
    );
    assert_eq!(l.held_owner(&rid(1)), held(Some("gpu0"), 1));
    let view = l.session("a", &sid, 3).unwrap();
    assert_eq!(
        (view.pinned_worker.as_deref(), view.owner_generation),
        (Some("gpu0"), 1)
    );
    commit(&mut l, 1, &sid, 4).unwrap();

    // The same worker again: no bump, no `session_owner`.
    begin_turn(&mut l, 2, &sid, 5);
    assert_eq!(l.held_owner(&rid(2)), held(Some("gpu0"), 1));
    dispatch(&mut l, 2, "gpu0", 6).unwrap();
    let line = last_line(&dir.0);
    assert!(
        line.starts_with(r#"{"type":"dispatched""#) && !line.contains("session_owner"),
        "{line}"
    );
    assert_eq!(l.held_owner(&rid(2)), held(Some("gpu0"), 1));
    commit(&mut l, 2, &sid, 7).unwrap();
    assert_eq!(
        l.session_owner(&sid, 7),
        Ok(owner_view(&sid, Some("gpu0"), 1, 2))
    );

    // A request without a session holds no owner; an unknown rid neither.
    let r = req(&l, 3);
    admit(&mut l, r, 8);
    run(&mut l, 3, 8);
    dispatch(&mut l, 3, "gpu1", 8).unwrap();
    assert!(!last_line(&dir.0).contains("session_owner"));
    assert_eq!(l.held_owner(&rid(3)), None);
    assert_eq!(l.held_owner(&rid(99)), None);
    assert_eq!(
        l.session_owner("sess_unknown", 8),
        Err(LedgerError::Session(SessionError::NotFound))
    );
}

/// Q2/Q3: a grant on another worker (the pinned one not eligible, m3 Q8) re-pins the session at the next
/// generation, journaled with that grant, so the move survives a restart whether the turn then commits, fails or
/// is cut off by a crash.
#[test]
fn p1_grant_time_re_pin_survives_restart_whether_the_turn_commits_or_fails() {
    for end in ["commit", "fail", "crash"] {
        let dir = TempDir::new(&format!("re-pin-{end}"));
        let mut l = open(&dir.0, limits(), 0);
        let sid = l.create_session("a", 1).unwrap().session_id;
        turn_on(&mut l, 1, &sid, "gpu0", 2);
        begin_turn(&mut l, 2, &sid, 3);
        assert_eq!(l.held_owner(&rid(2)), held(Some("gpu0"), 1));
        dispatch(&mut l, 2, "gpu1", 4).unwrap();
        assert!(last_line(&dir.0).contains(&format!(
            r#""session_owner":{{"session_id":"{sid}","owner_generation":2}}"#
        )));
        assert_eq!(l.held_owner(&rid(2)), held(Some("gpu1"), 2));
        assert_eq!(owner(&l, &sid, 4), owned("gpu1", 2));
        let revision = match end {
            "commit" => {
                commit(&mut l, 2, &sid, 5).unwrap();
                2
            }
            "fail" => {
                l.fail(&rid(2), "backend_error", None, 5).unwrap();
                1
            }
            _ => 1,
        };
        drop(l);
        let l = open(&dir.0, limits(), 10);
        assert_eq!(
            l.session_owner(&sid, 10),
            Ok(owner_view(&sid, Some("gpu1"), 2, revision)),
            "{end}"
        );
    }
}

/// Q3(c): a compaction between a re-pin's grant and its turn's terminal re-emits that `dispatched` record without
/// `session_owner` and the session snapshot with the owner, so the move survives a restart either way.
#[test]
fn p1_re_pin_survives_a_compaction_between_grant_and_terminal() {
    for end in ["commit", "fail"] {
        let dir = TempDir::new(&format!("re-pin-compact-{end}"));
        let mut l = open(&dir.0, limits(), 0);
        let sid = l.create_session("a", 1).unwrap().session_id;
        turn_on(&mut l, 1, &sid, "gpu0", 2);
        drop(l);
        let mut l = open(&dir.0, slack1(), 10);
        let compacted = journal_lines(&dir.0).len();
        begin_turn(&mut l, 2, &sid, 11);
        dispatch(&mut l, 2, "gpu1", 12).unwrap();
        assert!(last_line(&dir.0).contains("session_owner"));
        let next = fill_until_compaction_due(&mut l, &dir.0, compacted, 10, 13);
        let r = req(&l, next);
        admit(&mut l, r, 14);
        let records = journal_records(&dir.0);
        assert!(
            records
                .iter()
                .all(|r| !matches!(r, Record::Dispatched(d) if d.session_owner.is_some())),
            "compaction re-emits `dispatched` without `session_owner`"
        );
        assert!(records.iter().any(|r| matches!(r, Record::Session(s)
            if s.session_id == sid
                && s.owner == Some(OwnerRecord { worker: "gpu1".into(), owner_generation: 2 }))));
        let revision = if end == "commit" {
            commit(&mut l, 2, &sid, 15).unwrap();
            2
        } else {
            l.fail(&rid(2), "backend_error", None, 15).unwrap();
            1
        };
        drop(l);
        let l = open(&dir.0, limits(), 20);
        assert_eq!(
            l.session_owner(&sid, 20),
            Ok(owner_view(&sid, Some("gpu1"), 2, revision)),
            "{end}"
        );
    }
}

/// Q3(b)/Q5: a migration journals `session_owner{worker, generation + 1, revision, digest, at_ms}` before it
/// applies; worker and generation survive restarts, and the generation never decreases over pins, migrations,
/// compactions and reopens.
#[test]
fn p1_migrate_then_restart_keeps_worker_and_generation() {
    let dir = TempDir::new("migrate-restart");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    turn_on(&mut l, 1, &sid, "gpu0", 2);
    let current = l.session("a", &sid, 3).unwrap().current;
    let both = checker(&["gpu0", "gpu1"], &[]);
    assert_eq!(
        migrate(&mut l, &sid, "gpu1", 1, false, &both, 5),
        Ok(migrated(&sid, "gpu1", 2, 1, Some("gpu0"), None))
    );
    assert_eq!(
        journal_records(&dir.0).last(),
        Some(&Record::SessionOwner(SessionOwnerRecord {
            session_id: sid.clone(),
            worker: "gpu1".into(),
            owner_generation: 2,
            revision: 1,
            digest: current.digest,
            at_ms: 5,
        }))
    );
    assert_eq!(owner(&l, &sid, 5), owned("gpu1", 2));
    drop(l);
    let mut generations = vec![1, 2];
    for now in [10, 20] {
        let l = open(&dir.0, limits(), now);
        assert_eq!(
            l.session_owner(&sid, now),
            Ok(owner_view(&sid, Some("gpu1"), 2, 1))
        );
    }
    // The next turn holds the migrated owner and runs there without a bump.
    let mut l = open(&dir.0, limits(), 30);
    begin_turn(&mut l, 2, &sid, 31);
    assert_eq!(l.held_owner(&rid(2)), held(Some("gpu1"), 2));
    dispatch(&mut l, 2, "gpu1", 32).unwrap();
    assert!(!last_line(&dir.0).contains("session_owner"));
    commit(&mut l, 2, &sid, 33).unwrap();
    // A second move (3) and a grant-time re-pin back (4), then compactions (slack 1) and two reopens.
    migrate(&mut l, &sid, "gpu0", 2, false, &both, 34).unwrap();
    generations.push(owner(&l, &sid, 34).1);
    turn_on(&mut l, 3, &sid, "gpu1", 35);
    generations.push(owner(&l, &sid, 35).1);
    drop(l);
    let mut l = open(&dir.0, slack1(), 40);
    let compacted = journal_lines(&dir.0).len();
    let next = fill_until_compaction_due(&mut l, &dir.0, compacted, 10, 41);
    let r = req(&l, next);
    admit(&mut l, r, 42);
    generations.push(owner(&l, &sid, 42).1);
    drop(l);
    for now in [50, 60] {
        let l = open(&dir.0, slack1(), now);
        assert_eq!(owner(&l, &sid, now), owned("gpu1", 4));
        generations.push(owner(&l, &sid, now).1);
    }
    assert_eq!(generations, [1, 2, 3, 4, 4, 4, 4]);
}

/// M4-00: `accept` takes the owner under its own lock: a migration committed between a turn's session read and its
/// accept is the owner the turn holds (and targets), never the read's older one.
#[test]
fn p1_accept_holds_the_owner_read_under_its_own_lock() {
    let dir = TempDir::new("accept-owner");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    turn_on(&mut l, 1, &sid, "gpu0", 2);
    let read = l.session("a", &sid, 3).unwrap();
    assert_eq!(
        (read.pinned_worker.as_deref(), read.owner_generation),
        (Some("gpu0"), 1)
    );
    migrate(
        &mut l,
        &sid,
        "gpu1",
        1,
        false,
        &checker(&["gpu0", "gpu1"], &[]),
        4,
    )
    .unwrap();
    begin_turn(&mut l, 2, &sid, 5);
    assert_eq!(l.held_owner(&rid(2)), held(Some("gpu1"), 2));
    dispatch(&mut l, 2, "gpu1", 6).unwrap();
    assert!(!last_line(&dir.0).contains("session_owner"));
    commit(&mut l, 2, &sid, 7).unwrap();
    assert_eq!(
        l.session_owner(&sid, 7),
        Ok(owner_view(&sid, Some("gpu1"), 2, 2))
    );
}

/// Q3(a): a `dispatched` record that cannot be journaled (the compaction before it fails: its temp path is a
/// directory) changes neither the session owner nor the generation the turn holds; the retried grant re-pins.
#[test]
fn p1_dispatched_journal_failure_leaves_owner_and_held_generation() {
    let dir = TempDir::new("dispatch-fail");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    turn_on(&mut l, 1, &sid, "gpu0", 2);
    drop(l);
    let mut l = open(&dir.0, slack1(), 10);
    let compacted = journal_lines(&dir.0).len();
    begin_turn(&mut l, 2, &sid, 11);
    fill_until_compaction_due(&mut l, &dir.0, compacted, 10, 12);
    let blocker = dir.0.join("journal.jsonl.compact.tmp");
    std::fs::create_dir(&blocker).unwrap();
    let before = std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap();
    let result = dispatch(&mut l, 2, "gpu1", 13);
    assert!(matches!(result, Err(LedgerError::Journal(_))), "{result:?}");
    assert_eq!(std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap(), before);
    assert_eq!(l.held_owner(&rid(2)), held(Some("gpu0"), 1));
    assert_eq!(owner(&l, &sid, 13), owned("gpu0", 1));
    assert_eq!(l.status("a", &rid(2)).unwrap().state, RequestState::Running);
    assert!(!l.is_journal_poisoned());
    std::fs::remove_dir(&blocker).unwrap();
    dispatch(&mut l, 2, "gpu1", 14).unwrap();
    assert_eq!(l.held_owner(&rid(2)), held(Some("gpu1"), 2));
    commit(&mut l, 2, &sid, 15).unwrap();
    assert_eq!(owner(&l, &sid, 15), owned("gpu1", 2));
}

/// Q5 steps (2)-(5) (design §2.3 steps 1-4): each refusal comes from the first failing step and changes nothing
/// (owner, generation, the held lease and generation, the journal).
#[test]
fn p1_migrate_refusals_in_order_change_nothing() {
    let dir = TempDir::new("migrate-refusals");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    turn_on(&mut l, 1, &sid, "gpu0", 2);
    // rid 2 holds the lease.
    begin_turn(&mut l, 2, &sid, 3);
    let snapshot = |l: &Ledger| {
        (
            l.session_owner(&sid, 4),
            l.held_owner(&rid(2)),
            journal_lines(&dir.0),
        )
    };
    let before = snapshot(&l);
    let healthy = checker(&["gpu0", "gpu1"], &[]);
    let gpu1_out = checker(&["gpu0"], &["gpu1"]);
    let gpu0_out = checker(&["gpu1"], &["gpu0"]);
    let cases: [(&str, &str, u64, bool, Check<'_>, MigrateError); 9] = [
        // (1) an unknown session, whatever else is wrong.
        (
            "sess_unknown",
            "gpu9",
            7,
            true,
            &healthy,
            MigrateError::NotFound,
        ),
        // (2) a stale generation, before any worker check.
        (
            &sid,
            "gpu9",
            0,
            true,
            &gpu1_out,
            MigrateError::Conflict {
                worker: Some("gpu0".into()),
                owner_generation: 1,
            },
        ),
        // (3) an unknown, an ineligible, the current owner.
        (
            &sid,
            "gpu9",
            1,
            false,
            &healthy,
            MigrateError::WorkerNotFound,
        ),
        (
            &sid,
            "gpu1",
            1,
            false,
            &gpu1_out,
            MigrateError::WorkerNotEligible,
        ),
        (
            &sid,
            "gpu1",
            1,
            true,
            &gpu1_out,
            MigrateError::WorkerNotEligible,
        ),
        (&sid, "gpu0", 1, true, &gpu1_out, MigrateError::AlreadyOwner),
        // (4) force against an eligible owner (lease held or not); a held lease without force.
        (&sid, "gpu1", 1, true, &healthy, MigrateError::OwnerHealthy),
        (&sid, "gpu1", 1, false, &healthy, MigrateError::SessionBusy),
        (&sid, "gpu1", 1, false, &gpu0_out, MigrateError::SessionBusy),
    ];
    for (i, (session, to, expected, force, check, error)) in cases.into_iter().enumerate() {
        assert_eq!(
            migrate(&mut l, session, to, expected, force, check, 4),
            Err(error),
            "case {i}"
        );
        assert_eq!(snapshot(&l), before, "case {i}");
    }
    // A free lease: force against an eligible owner is still refused.
    l.fail(&rid(2), "backend_error", None, 5).unwrap();
    let before = snapshot(&l);
    assert_eq!(
        migrate(&mut l, &sid, "gpu1", 1, true, &healthy, 6),
        Err(MigrateError::OwnerHealthy)
    );
    assert_eq!(snapshot(&l), before);
    // No owner counts as not eligible: force moves an unowned session.
    let q = l.create_session("a", 7).unwrap().session_id;
    assert_eq!(
        migrate(&mut l, &q, "gpu1", 0, true, &healthy, 8),
        Ok(migrated(&q, "gpu1", 1, 0, None, None))
    );
    // An expired session is not found (also for the owner read).
    let later = 2 + limits().sessions.session_ttl_ms;
    assert_eq!(
        migrate(&mut l, &sid, "gpu1", 1, false, &healthy, later),
        Err(MigrateError::NotFound)
    );
    assert_eq!(
        l.session_owner(&sid, later),
        Err(LedgerError::Session(SessionError::NotFound))
    );
}

/// Q6/Q7: a forced takeover from a not eligible owner breaks the held lease and names its holder; that turn's late
/// commit is refused `session_owner_superseded` before any other check, the revision unchanged, and a re-dispatch
/// of it never moves the owner back. Force against an eligible owner is `OwnerHealthy`.
#[test]
fn p1_force_breaks_the_lease_and_fences_the_held_turn() {
    let dir = TempDir::new("force");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    turn_on(&mut l, 1, &sid, "gpu0", 2);
    begin_turn(&mut l, 2, &sid, 3);
    dispatch(&mut l, 2, "gpu0", 4).unwrap();
    assert_eq!(
        migrate(
            &mut l,
            &sid,
            "gpu1",
            1,
            true,
            &checker(&["gpu0", "gpu1"], &[]),
            5
        ),
        Err(MigrateError::OwnerHealthy)
    );
    let gpu0_out = checker(&["gpu1"], &["gpu0"]);
    assert_eq!(
        migrate(&mut l, &sid, "gpu1", 1, true, &gpu0_out, 6),
        Ok(migrated(&sid, "gpu1", 2, 1, Some("gpu0"), Some(rid(2))))
    );
    // The lease is free: the next turn takes it at the same revision.
    begin_turn(&mut l, 3, &sid, 7);
    assert_eq!(l.held_owner(&rid(3)), held(Some("gpu1"), 2));
    // The superseded turn's late completion is fenced first (another writer holds the lease now).
    let refused = commit(&mut l, 2, &sid, 8).unwrap_err();
    assert_eq!(refused, LedgerError::Session(SessionError::OwnerSuperseded));
    assert_eq!(refused.code(), REASON_SESSION_OWNER_SUPERSEDED);
    assert_eq!(REASON_SESSION_OWNER_SUPERSEDED, "session_owner_superseded");
    assert_eq!(
        SessionError::OwnerSuperseded.code(),
        REASON_SESSION_OWNER_SUPERSEDED
    );
    assert_eq!(l.status("a", &rid(2)).unwrap().state, RequestState::Running);
    assert_eq!(l.session("a", &sid, 8).unwrap().revision, 1);
    // A re-dispatch of the superseded turn (a re-queue) does not hold the lease, so it never moves the owner.
    dispatch(&mut l, 2, "gpu0", 9).unwrap();
    assert!(!last_line(&dir.0).contains("session_owner"));
    assert_eq!(owner(&l, &sid, 9), owned("gpu1", 2));
    assert_eq!(l.held_owner(&rid(2)), held(None, 1));
    l.fail(&rid(2), REASON_SESSION_OWNER_SUPERSEDED, None, 10)
        .unwrap();
    let status = l.status("a", &rid(2)).unwrap();
    assert_eq!(
        (status.state, status.reason.as_deref()),
        (RequestState::Failed, Some(REASON_SESSION_OWNER_SUPERSEDED))
    );
    // The next turn runs on the new owner at the next revision.
    dispatch(&mut l, 3, "gpu1", 11).unwrap();
    commit(&mut l, 3, &sid, 12).unwrap();
    drop(l);
    let l = open(&dir.0, limits(), 20);
    assert_eq!(
        l.session_owner(&sid, 20),
        Ok(owner_view(&sid, Some("gpu1"), 2, 2))
    );
}

/// Q5 (7), F9: a `session_owner` record that cannot be written answers `Journal`, leaves the owner unchanged and
/// does not poison the ledger: the next accept works and the retried migration applies.
#[test]
fn p1_migrate_journal_failure_changes_nothing_and_does_not_poison() {
    let dir = TempDir::new("migrate-fail");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    turn_on(&mut l, 1, &sid, "gpu0", 2);
    drop(l);
    let mut l = open(&dir.0, slack1(), 10);
    let compacted = journal_lines(&dir.0).len();
    let next = fill_until_compaction_due(&mut l, &dir.0, compacted, 10, 11);
    let blocker = dir.0.join("journal.jsonl.compact.tmp");
    std::fs::create_dir(&blocker).unwrap();
    let before = std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap();
    let both = checker(&["gpu0", "gpu1"], &[]);
    let refused = migrate(&mut l, &sid, "gpu1", 1, false, &both, 12).unwrap_err();
    assert!(
        matches!(refused, MigrateError::Journal(LedgerError::Journal(_))),
        "{refused:?}"
    );
    assert_eq!(std::fs::read(dir.0.join(JOURNAL_FILE)).unwrap(), before);
    assert_eq!(owner(&l, &sid, 12), owned("gpu0", 1));
    assert!(!l.is_journal_poisoned());
    std::fs::remove_dir(&blocker).unwrap();
    let r = req(&l, next);
    admit(&mut l, r, 13);
    assert_eq!(
        migrate(&mut l, &sid, "gpu1", 1, false, &both, 14),
        Ok(migrated(&sid, "gpu1", 2, 1, Some("gpu0"), None))
    );
    drop(l);
    let l = open(&dir.0, limits(), 20);
    assert_eq!(owner(&l, &sid, 20), owned("gpu1", 2));
}

/// Q3 replay: a `session_owner` record for a session the journal does not hold, or whose revision or digest
/// differ from the replayed session's, fails the open (like a session advance of an unknown session); a matching
/// one applies.
#[test]
fn p1_replay_refuses_a_session_owner_of_an_unknown_session_or_another_revision() {
    let dir = TempDir::new("replay-owner");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    let genesis = l.session("a", &sid, 1).unwrap().current.digest;
    turn_on(&mut l, 1, &sid, "gpu0", 2);
    let current = l.session("a", &sid, 3).unwrap().current.digest;
    drop(l);
    let journal = std::fs::read_to_string(dir.0.join(JOURNAL_FILE)).unwrap();
    let line = |session: &str, revision: u64, digest: Digest| {
        serde_json::to_string(&Record::SessionOwner(SessionOwnerRecord {
            session_id: session.into(),
            worker: "gpu1".into(),
            owner_generation: 2,
            revision,
            digest,
            at_ms: 5,
        }))
        .unwrap()
    };
    let mismatch = format!("session owner {sid}: revision mismatch");
    for (bad, why) in [
        (
            line("sess_unknown", 1, current),
            "session owner sess_unknown: session_not_found",
        ),
        (line(&sid, 0, current), mismatch.as_str()),
        (line(&sid, 1, genesis), mismatch.as_str()),
    ] {
        std::fs::write(dir.0.join(JOURNAL_FILE), format!("{journal}{bad}\n")).unwrap();
        let err = try_open(&dir.0, limits(), 10).unwrap_err();
        assert!(
            matches!(&err, LedgerError::Journal(m) if m.contains(why)),
            "{err:?}"
        );
    }
    std::fs::write(
        dir.0.join(JOURNAL_FILE),
        format!("{journal}{}\n", line(&sid, 1, current)),
    )
    .unwrap();
    let l = open(&dir.0, limits(), 10);
    assert_eq!(
        l.session_owner(&sid, 10),
        Ok(owner_view(&sid, Some("gpu1"), 2, 1))
    );
}

/// Q3 replay: a `dispatched` record's `session_owner` for a session the journal no longer holds (compaction keeps
/// the request records of a purged session) is ignored; the record still names the request's worker.
#[test]
fn p1_replay_ignores_a_dispatched_owner_of_an_unknown_session() {
    let dir = TempDir::new("replay-purged");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    begin_turn(&mut l, 1, &sid, 2);
    dispatch(&mut l, 1, "gpu0", 3).unwrap();
    l.fail(&rid(1), "backend_error", None, 4).unwrap();
    drop(l);
    let text = std::fs::read_to_string(dir.0.join(JOURNAL_FILE)).unwrap();
    assert!(text.lines().any(|line| line.contains("session_owner")));
    let kept: String = text
        .lines()
        .filter(|line| !line.contains(r#""type":"session""#))
        .flat_map(|line| [line, "\n"])
        .collect();
    assert_ne!(kept, text);
    std::fs::write(dir.0.join(JOURNAL_FILE), kept).unwrap();
    let l = open(&dir.0, limits(), 10);
    assert_eq!(
        l.session_owner(&sid, 10),
        Err(LedgerError::Session(SessionError::NotFound))
    );
    let status = l.status("a", &rid(1)).unwrap();
    assert_eq!(
        (status.worker_id.as_deref(), status.state),
        (Some("gpu0"), RequestState::Failed)
    );
}

/// Q10, Q2: a fork inherits its parent's owner worker at generation 1 (no owner: generation 0), its creation
/// record carries that owner across a reopen, and a later move of the parent leaves the fork where it is.
#[test]
fn p1_fork_inherits_the_owner_at_generation_one_across_reopen() {
    let dir = TempDir::new("fork");
    let mut l = open(&dir.0, limits(), 0);
    let both = checker(&["gpu0", "gpu1"], &[]);
    let p = l.create_session("a", 1).unwrap().session_id;
    turn_on(&mut l, 1, &p, "gpu0", 2);
    migrate(&mut l, &p, "gpu1", 1, false, &both, 3).unwrap();
    let q = l.create_session("a", 4).unwrap().session_id;
    let pc = l.fork_session("a", &p, 1, 5).unwrap().child;
    assert_eq!(
        (pc.pinned_worker.as_deref(), pc.owner_generation),
        (Some("gpu1"), 1)
    );
    assert!(
        matches!(journal_records(&dir.0).last(), Some(Record::Session(s))
        if s.session_id == pc.session_id
            && s.owner == Some(OwnerRecord { worker: "gpu1".into(), owner_generation: 1 }))
    );
    let qc = l.fork_session("a", &q, 0, 6).unwrap().child;
    assert_eq!(
        (qc.pinned_worker.as_deref(), qc.owner_generation),
        (None, 0)
    );
    let line = last_line(&dir.0);
    assert!(
        line.starts_with(r#"{"type":"session""#) && !line.contains("owner"),
        "{line}"
    );
    // Moving the parent does not move its fork.
    migrate(&mut l, &p, "gpu0", 2, false, &both, 7).unwrap();
    assert_eq!(owner(&l, &pc.session_id, 7), owned("gpu1", 1));
    drop(l);
    let mut l = open(&dir.0, limits(), 10);
    assert_eq!(owner(&l, &p, 10), owned("gpu0", 3));
    assert_eq!(owner(&l, &pc.session_id, 10), owned("gpu1", 1));
    assert_eq!(owner(&l, &q, 10), (None, 0));
    assert_eq!(owner(&l, &qc.session_id, 10), (None, 0));
    // The fork's turn on its inherited owner keeps generation 1.
    begin_turn(&mut l, 2, &pc.session_id, 11);
    assert_eq!(l.held_owner(&rid(2)), held(Some("gpu1"), 1));
    dispatch(&mut l, 2, "gpu1", 12).unwrap();
    assert!(!last_line(&dir.0).contains("session_owner"));
    commit(&mut l, 2, &pc.session_id, 13).unwrap();
    assert_eq!(owner(&l, &pc.session_id, 13), owned("gpu1", 1));
}

/// Q3 compatibility (forward-only journal): a reader without the M4 fields (a pre-M4 `cered`; mirrored here with
/// the same serde shape, `deny_unknown_fields`) reads `session` and `dispatched` lines that hold no owner data and
/// refuses every line that does: a session with `owner`, a dispatch with `session_owner`, and `session_owner`.
#[test]
fn p1_a_reader_without_the_owner_fields_refuses_the_lines_that_have_them() {
    #[allow(dead_code)]
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PreM4Dispatched {
        rid: String,
        worker: String,
        worker_epoch: u64,
        #[serde(default)]
        node: Option<String>,
        #[serde(default)]
        base_url: Option<String>,
        #[serde(default)]
        endpoint: Option<String>,
        at_ms: u64,
    }
    #[allow(dead_code)]
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PreM4Session {
        scope: String,
        session_id: String,
        salt: String,
        parent: Option<Value>,
        revisions: Vec<Value>,
        updated_ms: u64,
    }
    #[allow(dead_code)]
    #[derive(Debug, serde::Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum PreM4Record {
        Dispatched(PreM4Dispatched),
        Session(PreM4Session),
    }
    fn parse(line: &str) -> std::io::Result<PreM4Record> {
        serde_json::from_str(line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    let dir = TempDir::new("forward-only");
    let mut l = open(&dir.0, limits(), 0);
    let sid = l.create_session("a", 1).unwrap().session_id;
    let created = last_line(&dir.0);
    begin_turn(&mut l, 1, &sid, 2);
    dispatch(&mut l, 1, "gpu0", 3).unwrap();
    let pinned = last_line(&dir.0);
    commit(&mut l, 1, &sid, 4).unwrap();
    begin_turn(&mut l, 2, &sid, 5);
    dispatch(&mut l, 2, "gpu0", 6).unwrap();
    let unchanged = last_line(&dir.0);
    commit(&mut l, 2, &sid, 7).unwrap();
    let forked = l.fork_session("a", &sid, 1, 8).unwrap();
    let fork_line = last_line(&dir.0);
    assert!(fork_line.contains(&forked.child.session_id));
    migrate(
        &mut l,
        &sid,
        "gpu1",
        1,
        false,
        &checker(&["gpu0", "gpu1"], &[]),
        9,
    )
    .unwrap();
    let moved = last_line(&dir.0);
    drop(l);
    for older in [&created, &unchanged] {
        assert!(parse(older).is_ok(), "the mirror reads {older}");
    }
    for (newer, field) in [
        (&pinned, "session_owner"),
        (&fork_line, "owner"),
        (&moved, "session_owner"),
    ] {
        let err = parse(newer).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains(field), "{err}");
    }
}

/// M5 D8 over two deployments: `gpu0` serves model `m` (deployment 0), `gpu1` serves `n` (deployment 1).
fn by_deployment(eligible: bool) -> impl Fn(&str) -> WorkerCheck {
    move |w: &str| {
        let deployment = match w {
            "gpu0" => 0,
            "gpu1" => 1,
            _ => return WorkerCheck::Unknown,
        };
        if eligible {
            WorkerCheck::Eligible(deployment)
        } else {
            WorkerCheck::Ineligible(deployment)
        }
    }
}

fn deployment_of_model(model: &str) -> Option<usize> {
    match model {
        "m" => Some(0),
        "n" => Some(1),
        _ => None,
    }
}

fn migrate_m5(
    l: &mut Ledger,
    session: &str,
    to: &str,
    expected: u64,
    check: Check<'_>,
    now: u64,
) -> Result<Migrated, MigrateError> {
    let request = MigrateRequest {
        session_id: session,
        to_worker: to,
        expected_owner_generation: expected,
        force: false,
    };
    l.migrate_session(request, check, &deployment_of_model, now)
}

/// D8 order: `session_not_found`, `owner_generation_conflict`, `worker_not_found`, then `model_mismatch` (the
/// destination's deployment is not the bound model's, or for an unbound session not its owner's), before
/// `worker_not_eligible`; an unbound ownerless session skips the model check. Every refusal journals nothing.
#[test]
fn m5_migrate_refuses_another_deployment_before_eligibility() {
    let dir = TempDir::new("m5-migrate");
    let mut l = open(&dir.0, limits(), 1_000);
    let eligible = by_deployment(true);
    let ineligible = by_deployment(false);
    // A session bound to `m` and owned by gpu0.
    let bound = l.create_session("a", 1_000).unwrap().session_id;
    turn_on(&mut l, 1, &bound, "gpu0", 1_001);
    assert_eq!(
        l.session("a", &bound, 1_001).unwrap().model.as_deref(),
        Some("m")
    );
    let lines = journal_lines(&dir.0).len();
    assert_eq!(
        migrate_m5(&mut l, "nope", "gpu1", 1, &eligible, 1_002),
        Err(MigrateError::NotFound)
    );
    assert!(matches!(
        migrate_m5(&mut l, &bound, "gpu1", 7, &eligible, 1_002),
        Err(MigrateError::Conflict { .. })
    ));
    assert_eq!(
        migrate_m5(&mut l, &bound, "gpu9", 1, &eligible, 1_002),
        Err(MigrateError::WorkerNotFound)
    );
    for check in [&eligible as Check<'_>, &ineligible] {
        assert_eq!(
            migrate_m5(&mut l, &bound, "gpu1", 1, check, 1_002),
            Err(MigrateError::ModelMismatch)
        );
    }
    assert_eq!(
        journal_lines(&dir.0).len(),
        lines,
        "refusals journal nothing"
    );
    assert_eq!(owner(&l, &bound, 1_002), owned("gpu0", 1));

    // An unbound session owned by gpu0 (its first turn failed after the grant): gpu0's deployment decides.
    let unbound = l.create_session("a", 1_003).unwrap().session_id;
    begin_turn(&mut l, 2, &unbound, 1_003);
    dispatch(&mut l, 2, "gpu0", 1_003).unwrap();
    l.fail(&rid(2), "backend_error", None, 1_004).unwrap();
    assert_eq!(l.session("a", &unbound, 1_004).unwrap().model, None);
    assert_eq!(owner(&l, &unbound, 1_004), owned("gpu0", 1));
    assert_eq!(
        migrate_m5(&mut l, &unbound, "gpu1", 1, &ineligible, 1_005),
        Err(MigrateError::ModelMismatch)
    );

    // An unbound ownerless session skips the check: only eligibility decides.
    let fresh = l.create_session("a", 1_006).unwrap().session_id;
    assert_eq!(
        migrate_m5(&mut l, &fresh, "gpu1", 0, &ineligible, 1_006),
        Err(MigrateError::WorkerNotEligible)
    );
    assert_eq!(
        migrate_m5(&mut l, &fresh, "gpu1", 0, &eligible, 1_006).map(|m| m.worker),
        Ok("gpu1".to_owned())
    );
}

/// D8: a session bound to a model no deployment serves any more cannot move anywhere (`model_mismatch`).
#[test]
fn m5_migrate_of_a_session_bound_to_an_unserved_model_is_refused() {
    let dir = TempDir::new("m5-migrate-unserved");
    let mut l = open(&dir.0, limits(), 1_000);
    let s = l.create_session("a", 1_000).unwrap().session_id;
    turn_on(&mut l, 1, &s, "gpu0", 1_001);
    let request = MigrateRequest {
        session_id: &s,
        to_worker: "gpu1",
        expected_owner_generation: 1,
        force: false,
    };
    assert_eq!(
        l.migrate_session(request, &by_deployment(true), &|_: &str| None, 1_002),
        Err(MigrateError::ModelMismatch)
    );
}
