//! Session records (m1 design note S1–S3, Q4, Q5): in-memory decisions only; durability goes through the
//! ledger's journal ([`crate::ledger::Ledger`] owns a [`SessionStore`] and journals every change).
//!
//! A session stores its scope, salt lineage and `revisions[]` of `(message_count, keyed prefix digest)`; never
//! message content. Revision `r` is `revisions[r]`; the current revision is the last entry. A continuation at
//! revision `r` needs the single-writer lease (Q4), `r` current, and the digest of `messages[..count]` equal to
//! `revisions[r].digest`. The commit re-checks lease and revision, appends one entry and releases the lease.
//! A fork at `r` creates a child whose revision 0 is the parent's `revisions[r]` under the same salt (Q5), and
//! inherits the parent's owner worker (m3 design note Q8) at owner generation 1. A turn granted on another
//! worker re-pins the session.
//!
//! Owner (m4 design note Q2, Q3, Q7): `pinned_worker` is the session's durable owner and `owner_generation`
//! counts its changes (0 iff unowned). Every change (first pin, grant-time re-pin, operator migration) bumps the
//! generation by 1 and is journaled by the ledger before [`SessionStore::set_owner`] applies it. A turn commits
//! only at the owner generation it was granted at (the stale-owner fence, checked first in
//! [`SessionStore::check_commit`]).
//!
//! Limits: `max_sessions_per_scope` live sessions; a session expires `session_ttl_ms` after its last commit (or
//! creation) unless a writer holds its lease; the current revision never exceeds `max_revisions_per_session`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::LedgerConfig;
use crate::digest::{Digest, Salt};
use crate::journal::{OwnerRecord, SessionRecord};
use crate::ledger::REASON_SESSION_OWNER_SUPERSEDED;

/// One session revision: the message count it covers and the keyed digest of exactly those messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Revision {
    pub message_count: u64,
    pub digest: Digest,
}

/// Fork lineage of a child session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForkParent {
    pub session_id: String,
    pub revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionLimits {
    pub max_sessions_per_scope: u64,
    pub session_ttl_ms: u64,
    pub max_revisions_per_session: u64,
}

impl SessionLimits {
    pub fn from_config(ledger: &LedgerConfig) -> Self {
        Self {
            max_sessions_per_scope: ledger.max_sessions_per_scope,
            session_ttl_ms: ledger.session_ttl_ms,
            max_revisions_per_session: ledger.max_revisions_per_session,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionError {
    /// Unknown, expired, or owned by another scope (callers answer 404 with the unknown-session body).
    NotFound,
    /// `max_sessions_per_scope` live sessions already exist (429 `session_limit`).
    SessionLimit,
    /// The next revision would exceed `max_revisions_per_session` (409 `session_revision_limit`).
    RevisionLimit,
    /// Stale/unknown expected revision or edited message prefix (409 `revision_conflict`).
    RevisionConflict,
    /// Another writer holds the single-writer lease (Q4; 409 `revision_conflict`, before dispatch).
    LeaseHeld,
    /// The committing turn holds an owner generation the session no longer has: a migration moved the owner
    /// after the turn's grant (m4 design note Q7; 409 `session_owner_superseded`).
    OwnerSuperseded,
    /// The session is bound to another model than the turn's (M5 D7; 409 `model_mismatch`, before admission).
    ModelMismatch,
}

impl SessionError {
    pub fn code(self) -> &'static str {
        match self {
            Self::NotFound => "session_not_found",
            Self::SessionLimit => "session_limit",
            Self::RevisionLimit => "session_revision_limit",
            Self::RevisionConflict | Self::LeaseHeld => "revision_conflict",
            Self::OwnerSuperseded => REASON_SESSION_OWNER_SUPERSEDED,
            Self::ModelMismatch => "model_mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub scope: String,
    pub session_id: String,
    pub salt: Salt,
    pub parent: Option<ForkParent>,
    pub revisions: Vec<Revision>,
    pub updated_ms: u64,
    /// Request id (rid) of the writer holding the lease. Memory-only: a restart fails every in-flight writer.
    pub lease: Option<String>,
    /// Durable owner (m4 design note Q3; the sticky worker of S4 and m3 design note Q8): the worker of the
    /// latest owner change, inherited by forks. Journaled with every change.
    pub pinned_worker: Option<String>,
    /// Owner changes so far (m4 design note Q2): 0 iff `pinned_worker` is `None`; never decreases.
    pub owner_generation: u64,
    /// The model the session is bound to (M5 D7): the model of its first committed turn, inherited by forks.
    /// `None` until then, and for a session journaled before M5 until its next commit.
    pub model: Option<String>,
}

impl Session {
    pub fn revision(&self) -> u64 {
        self.revisions.len() as u64 - 1
    }

    pub fn current(&self) -> Revision {
        *self
            .revisions
            .last()
            .expect("a session always has revision 0")
    }

    pub fn record(&self) -> SessionRecord {
        SessionRecord {
            scope: self.scope.clone(),
            session_id: self.session_id.clone(),
            salt: self.salt,
            parent: self.parent.clone(),
            revisions: self.revisions.clone(),
            updated_ms: self.updated_ms,
            owner: self.pinned_worker.as_ref().map(|worker| OwnerRecord {
                worker: worker.clone(),
                owner_generation: self.owner_generation,
            }),
            model: self.model.clone(),
        }
    }

    fn from_record(record: SessionRecord) -> Self {
        let (pinned_worker, owner_generation) = match record.owner {
            Some(o) => (Some(o.worker), o.owner_generation),
            None => (None, 0),
        };
        Self {
            scope: record.scope,
            session_id: record.session_id,
            salt: record.salt,
            parent: record.parent,
            revisions: record.revisions,
            updated_ms: record.updated_ms,
            lease: None,
            pinned_worker,
            owner_generation,
            model: record.model,
        }
    }
}

#[derive(Debug)]
pub struct SessionStore {
    limits: SessionLimits,
    sessions: HashMap<String, Session>,
    /// Team pools (net-teleport design note 4.4, C18): `[[auth.scopes]] max_sessions_per_scope` overrides.
    scope_caps: HashMap<String, u64>,
}

impl SessionStore {
    pub fn new(limits: SessionLimits) -> Self {
        Self {
            limits,
            sessions: HashMap::new(),
            scope_caps: HashMap::new(),
        }
    }

    /// Sets each listed scope's live-session cap in place of `max_sessions_per_scope`.
    pub fn set_scope_caps(&mut self, caps: HashMap<String, u64>) {
        self.scope_caps = caps;
    }

    fn expired(&self, s: &Session, now_ms: u64) -> bool {
        s.lease.is_none() && s.updated_ms.saturating_add(self.limits.session_ttl_ms) <= now_ms
    }

    /// Drops every expired, unleased session.
    pub fn purge_expired(&mut self, now_ms: u64) {
        let ttl = self.limits.session_ttl_ms;
        self.sessions
            .retain(|_, s| s.lease.is_some() || s.updated_ms.saturating_add(ttl) > now_ms);
    }

    /// A live session of `scope`.
    pub fn get(
        &self,
        scope: &str,
        session_id: &str,
        now_ms: u64,
    ) -> Result<&Session, SessionError> {
        match self.get_any(session_id, now_ms) {
            Ok(s) if s.scope == scope => Ok(s),
            _ => Err(SessionError::NotFound),
        }
    }

    /// A live session of any scope (operator routes carry no scope, m4 design note Q4).
    pub fn get_any(&self, session_id: &str, now_ms: u64) -> Result<&Session, SessionError> {
        match self.sessions.get(session_id) {
            Some(s) if !self.expired(s, now_ms) => Ok(s),
            _ => Err(SessionError::NotFound),
        }
    }

    /// A stored session whatever its scope or expiry: journal replay (sessions are purged only after it) and a
    /// request's own session (a leased session never expires).
    pub fn stored(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Fails with `SessionLimit` when `scope` already holds its cap of live sessions: its override
    /// ([`Self::set_scope_caps`]), else `max_sessions_per_scope`.
    pub fn check_capacity(&mut self, scope: &str, now_ms: u64) -> Result<(), SessionError> {
        self.purge_expired(now_ms);
        let live = self.sessions.values().filter(|s| s.scope == scope).count() as u64;
        let cap = self
            .scope_caps
            .get(scope)
            .copied()
            .unwrap_or(self.limits.max_sessions_per_scope);
        if live >= cap {
            return Err(SessionError::SessionLimit);
        }
        Ok(())
    }

    /// A new root session at revision 0 (not inserted; journal [`Session::record`] first, then [`Self::insert`]).
    pub fn new_root(
        &mut self,
        scope: &str,
        session_id: String,
        salt: Salt,
        genesis: Digest,
        now_ms: u64,
    ) -> Result<Session, SessionError> {
        self.check_capacity(scope, now_ms)?;
        Ok(Session {
            scope: scope.to_owned(),
            session_id,
            salt,
            parent: None,
            revisions: vec![Revision {
                message_count: 0,
                digest: genesis,
            }],
            updated_ms: now_ms,
            lease: None,
            pinned_worker: None,
            owner_generation: 0,
            model: None,
        })
    }

    /// A fork child of `parent_id` at `revision` (Q5; not inserted). `revision` must exist in the parent.
    pub fn new_fork(
        &mut self,
        scope: &str,
        parent_id: &str,
        revision: u64,
        child_id: String,
        now_ms: u64,
    ) -> Result<Session, SessionError> {
        let parent = self.get(scope, parent_id, now_ms)?;
        // m4 design note Q2/Q10: the child starts its own owner history at the parent's worker.
        let pinned_worker = parent.pinned_worker.clone();
        let owner_generation = u64::from(pinned_worker.is_some());
        // M5 D7: the child is bound to the parent's model (unbound with it).
        let model = parent.model.clone();
        let entry = *parent
            .revisions
            .get(usize::try_from(revision).map_err(|_| SessionError::RevisionConflict)?)
            .ok_or(SessionError::RevisionConflict)?;
        let salt = parent.salt;
        self.check_capacity(scope, now_ms)?;
        Ok(Session {
            scope: scope.to_owned(),
            session_id: child_id,
            salt,
            parent: Some(ForkParent {
                session_id: parent_id.to_owned(),
                revision,
            }),
            revisions: vec![entry],
            updated_ms: now_ms,
            lease: None,
            pinned_worker,
            owner_generation,
            model,
        })
    }

    pub fn insert(&mut self, session: Session) {
        self.sessions.insert(session.session_id.clone(), session);
    }

    /// Replay: a journaled session record replaces any earlier state of the same id.
    pub fn insert_record(&mut self, record: SessionRecord) {
        self.insert(Session::from_record(record));
    }

    /// S2 admission check for a writer of `model` at `expected_revision` whose `messages[..count]` digest is
    /// `prefix`. A session bound to another model refuses the turn first (M5 D7).
    pub fn check_claim(
        &self,
        scope: &str,
        session_id: &str,
        model: &str,
        expected_revision: u64,
        prefix: &Digest,
        now_ms: u64,
    ) -> Result<(), SessionError> {
        let s = self.get(scope, session_id, now_ms)?;
        if s.model.as_deref().is_some_and(|bound| bound != model) {
            return Err(SessionError::ModelMismatch);
        }
        if expected_revision != s.revision() || *prefix != s.current().digest {
            return Err(SessionError::RevisionConflict);
        }
        if s.revision() >= self.limits.max_revisions_per_session {
            return Err(SessionError::RevisionLimit);
        }
        if s.lease.is_some() {
            return Err(SessionError::LeaseHeld);
        }
        Ok(())
    }

    /// Takes the lease after a successful [`Self::check_claim`]. The owner changes at the grant
    /// ([`Self::owner_change`], Q8), not here.
    pub fn acquire(&mut self, session_id: &str, holder: &str) {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.lease = Some(holder.to_owned());
        }
    }

    /// The owner generation a grant of `holder`'s turn on `worker` moves the session to (m4 design note Q2):
    /// `Some(generation + 1)` iff the session exists, `holder` holds its lease (a stale turn never moves the owner)
    /// and `worker` is not already the owner. Changes nothing: the caller journals the change, then
    /// [`Self::set_owner`].
    pub fn owner_change(&self, session_id: &str, holder: &str, worker: &str) -> Option<u64> {
        let s = self.sessions.get(session_id)?;
        (s.lease.as_deref() == Some(holder) && s.pinned_worker.as_deref() != Some(worker))
            .then_some(s.owner_generation + 1)
    }

    /// Sets the session's owner and generation, unchecked: callers journal the change first (Q3).
    pub fn set_owner(&mut self, session_id: &str, worker: &str, owner_generation: u64) {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.pinned_worker = Some(worker.to_owned());
            s.owner_generation = owner_generation;
        }
    }

    /// Refreshes the session's TTL (a migration, m4 design note Q5 step 8).
    pub fn touch(&mut self, session_id: &str, now_ms: u64) {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.updated_ms = now_ms;
        }
    }

    /// Commit re-check under the lease (Q4): first the stale-owner fence (m4 design note Q7: the turn's
    /// `held_generation` is still the session's owner generation, else `OwnerSuperseded`), then `holder` holds the
    /// lease, the revision is still `expected_revision`, and the new entry covers more messages. Returns the new
    /// revision number.
    pub fn check_commit(
        &self,
        session_id: &str,
        holder: &str,
        expected_revision: u64,
        entry: &Revision,
        held_generation: u64,
    ) -> Result<u64, SessionError> {
        let s = self
            .sessions
            .get(session_id)
            .ok_or(SessionError::NotFound)?;
        if held_generation != s.owner_generation {
            return Err(SessionError::OwnerSuperseded);
        }
        if s.lease.as_deref() != Some(holder)
            || s.revision() != expected_revision
            || entry.message_count <= s.current().message_count
        {
            return Err(SessionError::RevisionConflict);
        }
        Ok(expected_revision + 1)
    }

    /// Appends `entry` as `revision` (live commit or replay), binds the session to `model` when given (M5 D7: a
    /// live commit always names its turn's model; a pre-M5 advance names none and leaves the binding), refreshes
    /// the TTL and releases the lease. Fails if `revision` is not exactly the next one.
    pub fn apply_advance(
        &mut self,
        session_id: &str,
        revision: u64,
        entry: Revision,
        model: Option<&str>,
        now_ms: u64,
    ) -> Result<(), SessionError> {
        let s = self
            .sessions
            .get_mut(session_id)
            .ok_or(SessionError::NotFound)?;
        if revision != s.revision() + 1 {
            return Err(SessionError::RevisionConflict);
        }
        s.revisions.push(entry);
        if let Some(model) = model {
            s.model = Some(model.to_owned());
        }
        s.updated_ms = now_ms;
        s.lease = None;
        Ok(())
    }

    /// Releases the lease without advancing (failed/cancelled turn).
    pub fn release(&mut self, session_id: &str, holder: &str) {
        if let Some(s) = self.sessions.get_mut(session_id) {
            if s.lease.as_deref() == Some(holder) {
                s.lease = None;
            }
        }
    }

    /// Records of every live session, for journal compaction.
    pub fn records(&self, now_ms: u64) -> Vec<SessionRecord> {
        let mut out: Vec<SessionRecord> = self
            .sessions
            .values()
            .filter(|s| !self.expired(s, now_ms))
            .map(Session::record)
            .collect();
        out.sort_by(|a, b| a.session_id.cmp(&b.session_id));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::DigestKeys;
    use serde_json::{json, Value};

    const LIMITS: SessionLimits = SessionLimits {
        max_sessions_per_scope: 2,
        session_ttl_ms: 1000,
        max_revisions_per_session: 2,
    };

    struct Fixture {
        dir: std::path::PathBuf,
        keys: DigestKeys,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "cere-session-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            let keys = DigestKeys::open(&dir).unwrap();
            Self { dir, keys }
        }

        fn digest(&self, scope: &str, salt: &Salt, messages: &[Value]) -> Digest {
            self.keys.scope_key(scope).prefix_digest(salt, messages)
        }

        fn root(&self, store: &mut SessionStore, scope: &str, id: &str, now: u64) -> Salt {
            let salt = Salt::random().unwrap();
            let s = store
                .new_root(scope, id.into(), salt, self.digest(scope, &salt, &[]), now)
                .unwrap();
            store.insert(s);
            salt
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn turn(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": format!("m{i}")}))
            .collect()
    }

    #[test]
    fn create_starts_at_revision_zero_and_session_limit_applies_per_scope() {
        let f = Fixture::new("limit");
        let mut store = SessionStore::new(LIMITS);
        f.root(&mut store, "a", "s1", 0);
        f.root(&mut store, "a", "s2", 0);
        assert_eq!(store.get("a", "s1", 0).unwrap().revision(), 0);
        assert_eq!(store.get("a", "s1", 0).unwrap().current().message_count, 0);
        let salt = Salt::random().unwrap();
        let err = store.new_root("a", "s3".into(), salt, f.digest("a", &salt, &[]), 0);
        assert_eq!(err, Err(SessionError::SessionLimit));
        assert_eq!(SessionError::SessionLimit.code(), "session_limit");
        // Another scope has its own budget; scope isolation hides sessions.
        f.root(&mut store, "b", "s4", 0);
        assert_eq!(store.get("b", "s1", 0), Err(SessionError::NotFound));
    }

    #[test]
    fn ttl_expiry_frees_capacity_but_not_while_leased() {
        let f = Fixture::new("ttl");
        let mut store = SessionStore::new(LIMITS);
        f.root(&mut store, "a", "s1", 0);
        f.root(&mut store, "a", "s2", 500);
        store.acquire("s1", "rid1");
        // s1 is leased: not expired at t=1000 even though its TTL passed; s2 is still young.
        assert_eq!(
            store.check_capacity("a", 1000),
            Err(SessionError::SessionLimit)
        );
        store.release("s1", "rid1");
        assert_eq!(store.get("a", "s1", 1000), Err(SessionError::NotFound));
        store.check_capacity("a", 1000).unwrap();
        assert_eq!(store.records(1000).len(), 1);
    }

    #[test]
    fn two_writers_at_same_revision_second_rejected_and_commit_advances() {
        let f = Fixture::new("lease");
        let mut store = SessionStore::new(LIMITS);
        let salt = f.root(&mut store, "a", "s", 0);
        let genesis = f.digest("a", &salt, &[]);
        store.check_claim("a", "s", "m", 0, &genesis, 1).unwrap();
        store.acquire("s", "w1");
        assert_eq!(store.owner_change("s", "w1", "worker-0"), Some(1));
        store.set_owner("s", "worker-0", 1);
        assert_eq!(
            store.check_claim("a", "s", "m", 0, &genesis, 1),
            Err(SessionError::LeaseHeld)
        );
        assert_eq!(SessionError::LeaseHeld.code(), "revision_conflict");
        // Commit: the second writer cannot commit; the holder can, at the owner generation it holds.
        let entry = Revision {
            message_count: 2,
            digest: f.digest("a", &salt, &turn(2)),
        };
        assert_eq!(
            store.check_commit("s", "w2", 0, &entry, 1),
            Err(SessionError::RevisionConflict)
        );
        assert_eq!(store.check_commit("s", "w1", 0, &entry, 1), Ok(1));
        store.apply_advance("s", 1, entry, Some("m"), 5).unwrap();
        let s = store.get("a", "s", 5).unwrap();
        assert_eq!((s.revision(), s.lease.clone(), s.updated_ms), (1, None, 5));
        assert_eq!(
            (s.pinned_worker.as_deref(), s.owner_generation),
            (Some("worker-0"), 1)
        );
    }

    /// M4 Q7: a commit at an owner generation other than the session's is `OwnerSuperseded`, checked before the
    /// lease, the revision and the growth, so the reason is exact.
    #[test]
    fn a_stale_owner_generation_is_refused_before_every_other_commit_check() {
        let f = Fixture::new("fence");
        let mut store = SessionStore::new(LIMITS);
        let salt = f.root(&mut store, "a", "s", 0);
        store.acquire("s", "w1");
        assert_eq!(store.owner_change("s", "w1", "gpu0"), Some(1));
        store.set_owner("s", "gpu0", 1);
        let entry = Revision {
            message_count: 2,
            digest: f.digest("a", &salt, &turn(2)),
        };
        // A migration moved the owner to generation 2 and broke the lease.
        store.set_owner("s", "gpu1", 2);
        store.release("s", "w1");
        assert_eq!(
            store.check_commit("s", "w1", 0, &entry, 1),
            Err(SessionError::OwnerSuperseded)
        );
        // Not the lease holder, a stale revision and no growth: still the owner fence first.
        store.acquire("s", "w2");
        let stale = Revision {
            message_count: 0,
            ..entry
        };
        assert_eq!(
            store.check_commit("s", "w1", 7, &stale, 1),
            Err(SessionError::OwnerSuperseded)
        );
        assert_eq!(
            SessionError::OwnerSuperseded.code(),
            "session_owner_superseded"
        );
        assert_eq!(store.check_commit("s", "w2", 0, &entry, 2), Ok(1));
        assert_eq!(
            store.check_commit("s", "w2", 0, &entry, 1),
            Err(SessionError::OwnerSuperseded)
        );
    }

    /// M4 Q2: only the lease holder's grant on another worker changes the owner, one generation up.
    #[test]
    fn owner_change_needs_the_lease_and_another_worker() {
        let f = Fixture::new("owner-change");
        let mut store = SessionStore::new(LIMITS);
        f.root(&mut store, "a", "s", 0);
        assert_eq!(store.owner_change("s", "w1", "gpu0"), None);
        assert_eq!(store.owner_change("missing", "w1", "gpu0"), None);
        store.acquire("s", "w1");
        assert_eq!(store.owner_change("s", "w2", "gpu0"), None);
        assert_eq!(store.owner_change("s", "w1", "gpu0"), Some(1));
        // `owner_change` only proposes; nothing moved until `set_owner`.
        let s = store.get("a", "s", 0).unwrap();
        assert_eq!((s.pinned_worker.clone(), s.owner_generation), (None, 0));
        store.set_owner("s", "gpu0", 1);
        assert_eq!(store.owner_change("s", "w1", "gpu0"), None);
        assert_eq!(store.owner_change("s", "w1", "gpu1"), Some(2));
    }

    /// M4 Q3(c): the session record carries the owner (absent without one) and restores it; operator reads
    /// (`get_any`) see every scope but never an expired session; `touch` refreshes the TTL.
    #[test]
    fn the_record_carries_the_owner_and_get_any_ignores_the_scope() {
        let f = Fixture::new("owner-record");
        let mut store = SessionStore::new(LIMITS);
        f.root(&mut store, "a", "s", 0);
        let bare = store.get("a", "s", 0).unwrap().record();
        assert_eq!(bare.owner, None);
        let line = serde_json::to_string(&bare).unwrap();
        assert!(!line.contains("owner"), "{line}");
        store.set_owner("s", "gpu1", 3);
        let record = store.get("a", "s", 0).unwrap().record();
        assert_eq!(
            record.owner,
            Some(OwnerRecord {
                worker: "gpu1".into(),
                owner_generation: 3
            })
        );
        let mut other = SessionStore::new(LIMITS);
        other.insert_record(record);
        let s = other.get_any("s", 0).unwrap();
        assert_eq!(
            (
                s.scope.as_str(),
                s.pinned_worker.as_deref(),
                s.owner_generation
            ),
            ("a", Some("gpu1"), 3)
        );
        assert_eq!(other.get_any("t", 0), Err(SessionError::NotFound));
        other.touch("s", 900);
        assert!(other.get_any("s", 1000).is_ok());
        assert_eq!(other.get_any("s", 1900), Err(SessionError::NotFound));
    }

    #[test]
    fn stale_revision_and_edited_prefix_conflict() {
        let f = Fixture::new("stale");
        let mut store = SessionStore::new(LIMITS);
        let salt = f.root(&mut store, "a", "s", 0);
        store.acquire("s", "w1");
        let entry = Revision {
            message_count: 2,
            digest: f.digest("a", &salt, &turn(2)),
        };
        store.apply_advance("s", 1, entry, Some("m"), 1).unwrap();
        // Stale revision 0.
        let genesis = f.digest("a", &salt, &[]);
        assert_eq!(
            store.check_claim("a", "s", "m", 0, &genesis, 2),
            Err(SessionError::RevisionConflict)
        );
        // Current revision 1 but edited history.
        let mut edited = turn(2);
        edited[0] = json!({"role": "user", "content": "EDITED"});
        let bad = f.digest("a", &salt, &edited);
        assert_eq!(
            store.check_claim("a", "s", "m", 1, &bad, 2),
            Err(SessionError::RevisionConflict)
        );
        store
            .check_claim("a", "s", "m", 1, &entry.digest, 2)
            .unwrap();
        // Replay of a non-consecutive revision is rejected.
        assert_eq!(
            store.apply_advance("s", 3, entry, Some("m"), 2),
            Err(SessionError::RevisionConflict)
        );
    }

    #[test]
    fn revision_cap_rejects_before_dispatch() {
        let f = Fixture::new("cap");
        let mut store = SessionStore::new(LIMITS);
        let salt = f.root(&mut store, "a", "s", 0);
        for r in 1..=2u64 {
            let entry = Revision {
                message_count: 2 * r,
                digest: f.digest("a", &salt, &turn(2 * r as usize)),
            };
            store.acquire("s", "w");
            store.apply_advance("s", r, entry, Some("m"), 1).unwrap();
        }
        let current = store.get("a", "s", 1).unwrap().current().digest;
        assert_eq!(
            store.check_claim("a", "s", "m", 2, &current, 1),
            Err(SessionError::RevisionLimit)
        );
        assert_eq!(SessionError::RevisionLimit.code(), "session_revision_limit");
    }

    #[test]
    fn fork_inherits_salt_lineage_and_parent_revision() {
        let f = Fixture::new("fork");
        let mut store = SessionStore::new(SessionLimits {
            max_sessions_per_scope: 4,
            ..LIMITS
        });
        let salt = f.root(&mut store, "a", "p", 0);
        let r1 = Revision {
            message_count: 2,
            digest: f.digest("a", &salt, &turn(2)),
        };
        store.acquire("p", "w");
        store.apply_advance("p", 1, r1, Some("m"), 1).unwrap();
        assert_eq!(
            store.new_fork("a", "p", 2, "c".into(), 2),
            Err(SessionError::RevisionConflict)
        );
        assert_eq!(
            store.new_fork("b", "p", 1, "c".into(), 2),
            Err(SessionError::NotFound)
        );
        let child = store.new_fork("a", "p", 1, "c".into(), 2).unwrap();
        assert_eq!(child.salt, salt);
        assert_eq!(child.revision(), 0);
        assert_eq!(child.current(), r1);
        assert_eq!(
            child.parent,
            Some(ForkParent {
                session_id: "p".into(),
                revision: 1
            })
        );
        store.insert(child);
        // The child continues from the parent's history: same messages verify under the inherited salt.
        store
            .check_claim("a", "c", "m", 0, &f.digest("a", &salt, &turn(2)), 2)
            .unwrap();
        // Fork at an older revision.
        let older = store.new_fork("a", "p", 0, "c0".into(), 2).unwrap();
        assert_eq!(older.current().message_count, 0);
    }

    /// m3 Q8, M4 Q2/Q10: a fork inherits its parent's owner worker (prefix locality) at generation 1, whatever
    /// the parent's generation; a root, and the fork of an unowned session, start unowned at generation 0.
    #[test]
    fn fork_inherits_the_parent_owner_at_generation_one() {
        let f = Fixture::new("fork-pin");
        let mut store = SessionStore::new(SessionLimits {
            max_sessions_per_scope: 4,
            ..LIMITS
        });
        f.root(&mut store, "a", "p", 0);
        let p = store.get("a", "p", 0).unwrap();
        assert_eq!((p.pinned_worker.clone(), p.owner_generation), (None, 0));
        let unowned = store.new_fork("a", "p", 0, "u".into(), 1).unwrap();
        assert_eq!((unowned.pinned_worker, unowned.owner_generation), (None, 0));
        // Owned by gpu1 after three owner changes.
        store.set_owner("p", "gpu1", 3);
        let child = store.new_fork("a", "p", 0, "c".into(), 1).unwrap();
        assert_eq!(
            (child.pinned_worker.as_deref(), child.owner_generation),
            (Some("gpu1"), 1)
        );
        store.insert(child);
        // The child's own later re-pin leaves the parent's alone.
        store.acquire("c", "w2");
        assert_eq!(store.owner_change("c", "w2", "gpu0"), Some(2));
        store.set_owner("c", "gpu0", 2);
        let c = store.get("a", "c", 1).unwrap();
        assert_eq!(
            (c.pinned_worker.as_deref(), c.owner_generation),
            (Some("gpu0"), 2)
        );
        let p = store.get("a", "p", 1).unwrap();
        assert_eq!(
            (p.pinned_worker.as_deref(), p.owner_generation),
            (Some("gpu1"), 3)
        );
    }

    /// m3 Q8: a turn whose pinned worker is not eligible is granted elsewhere and the grant re-pins the session
    /// one owner generation up (M4 Q2); only the lease holder's grant pins.
    #[tokio::test]
    async fn grant_on_another_worker_re_pins_the_session() {
        use crate::admission::{Admission, AdmissionLimits, Priority, Target, WorkerCap};
        use crate::worker::{Aging, WorkerHealth, WorkerView};
        let f = Fixture::new("re-pin");
        let mut store = SessionStore::new(LIMITS);
        f.root(&mut store, "a", "s", 0);
        store.acquire("s", "turn1");
        assert_eq!(store.owner_change("s", "turn1", "gpu0"), Some(1));
        store.set_owner("s", "gpu0", 1);
        store.release("s", "turn1");
        let adm = Admission::new(AdmissionLimits {
            max_inflight: 2,
            per_scope_inflight: 2,
            queue_depth: 2,
        });
        let aging = Aging {
            suspect_after_ms: 60_000,
            lost_after_ms: 60_000,
        };
        adm.set_workers(vec![
            WorkerCap {
                id: "gpu0".into(),
                cap: 1,
                aging,
                deployment: 0,
            },
            WorkerCap {
                id: "gpu1".into(),
                cap: 1,
                aging,
                deployment: 0,
            },
        ]);
        let now = crate::app::unix_ms();
        for (id, health) in [
            ("gpu0", WorkerHealth::Suspect),
            ("gpu1", WorkerHealth::Ready),
        ] {
            let view = WorkerView {
                health,
                last_seen_ms: now,
                epoch: 1,
                draining: false,
                lease_until_ms: None,
            };
            adm.update_worker(id, view);
        }
        store.acquire("s", "turn2");
        let pinned = store
            .get("a", "s", 0)
            .unwrap()
            .pinned_worker
            .clone()
            .unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let permit = adm
            .acquire(
                "a",
                Priority::Normal,
                deadline,
                Target::Pinned(pinned.into(), 0),
                1,
            )
            .await
            .unwrap();
        assert_eq!(permit.worker(), "gpu1");
        // A stale turn cannot move the pin; the lease holder's grant does.
        assert_eq!(store.owner_change("s", "turn1", permit.worker()), None);
        assert_eq!(store.owner_change("s", "turn2", permit.worker()), Some(2));
        store.set_owner("s", permit.worker(), 2);
        let s = store.get("a", "s", 0).unwrap();
        assert_eq!(
            (s.pinned_worker.as_deref(), s.owner_generation),
            (Some("gpu1"), 2)
        );
    }
}
