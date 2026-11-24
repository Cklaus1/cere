//! The agent's durable operation journal (m35 design note Q12).
//!
//! `<state_dir>/operations.jsonl`: one JSON record per line (`accepted`, `terminal`, `cleanup`), each append
//! `fsync`'d before the call returns, so a receipt the agent hands out is on disk. The directory is 0700 and the
//! file 0600. Journaled methods are `SubmitWork`, `CancelWork` and the M3.6 lifecycle methods `LoadModel`,
//! `DrainModel`, `UnloadModel` (m36 design note Q6); reads are not journaled. A journal holding lifecycle
//! records does not replay in a pre-M3.6 agent (forward-only).
//!
//! Operations are keyed by `(scope_id, operation_id)`. Their identity is the method, a keyed digest, the
//! worker epoch, the model generation and the coordinator term. The journal never stores the caller's
//! `payload_sha256`: it stores `HMAC-SHA256(agent_secret, len(scope_id) ‖ scope_id ‖ payload_sha256)` (the
//! scope id length-prefixed as u64 big-endian, so no two inputs share an encoding). `agent_secret` is a 32-byte
//! hex secret in `<state_dir>/agent_secret` (0600), created once on first open like `DigestKeys::open` creates the
//! server secret, and never silently replaced: a malformed secret fails the open.
//!
//! A key with the same identity returns the original receipt (`fresh == false`: the caller makes no second
//! backend call); a different identity is a conflict. The completion ticket is 128 random bits (hex), minted at
//! acceptance. The worker-local deadline is the receipt instant + min(`remaining_budget_ms`, the 600 000 ms work
//! cap); a retry never renews it.
//!
//! Bounded: `max_operations` live + retained operations; a new key beyond that is refused and nothing is evicted.
//! A terminal, cleanup-acknowledged operation (a tombstone) is dropped at max(terminal + `retry_horizon_ms`,
//! deadline + `rpc_timeout_ms`). An operation not yet cleanup-acknowledged is never dropped: it still counts
//! toward the live-op bound ([`OpJournal::live_ops`]). After a drop the key is unknown, and a later accept of it
//! is new work; replay accepts such a re-accept only when its receipt instant is at or past the old tombstone's
//! drop instant.
//!
//! Crash tolerance and compaction follow the ledger journal (`crate::journal`): an append is one `write` of
//! `line + "\n"`, so a crash can leave only a torn final line without its newline; open cuts it and reports it
//! ([`OpReplay::torn_tail`]). A complete line that does not parse, or has an unknown field, fails the open. Open
//! then drops due tombstones and compacts (temp file, fsync, rename, directory fsync); [`OpJournal::expire`]
//! compacts once the file has grown past twice its last compacted size plus `max_operations` lines.
//!
//! A failed write rolls the file back to its last good length and poisons the journal: every later mutation
//! (including a duplicate accept) is refused with [`OpError::Poisoned`] for the life of this value. Reads keep
//! working.
//!
//! Records hold scope ids, operation ids, request ids, attempts, tickets, keyed digests, epochs, generations,
//! terms, reason codes and timestamps. Never prompt or output text.

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{File, OpenOptions, Permissions};
use std::io::{self, Read, Write};
use std::ops::Bound;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use crate::digest::{hex, open_secret, random_bytes, unhex};
use crate::fed::wire;
use crate::journal::{ensure_private_dir, sync_dir};

/// The journal file inside the agent `state_dir`.
pub const OPERATIONS_FILE: &str = "operations.jsonl";
/// The agent HMAC secret inside the agent `state_dir`.
pub const SECRET_FILE: &str = "agent_secret";
const COMPACT_TMP_FILE: &str = "operations.jsonl.compact.tmp";
/// The journaled methods (Q12, M3.6 Q6); an [`OpIdentity::method`] is one of these.
const METHODS: [&str; 5] = [
    wire::METHOD_SUBMIT,
    wire::METHOD_CANCEL,
    wire::METHOD_LOAD,
    wire::METHOD_DRAIN,
    wire::METHOD_UNLOAD,
];

type HmacSha256 = Hmac<Sha256>;

/// Journal bounds (agent config `[journal]` and `[limits] rpc_timeout_ms`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpLimits {
    /// Live + retained operations.
    pub max_operations: usize,
    pub retry_horizon_ms: u64,
    pub rpc_timeout_ms: u64,
}

/// What [`OpJournal::open`] found on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpReplay {
    /// A torn final line (bytes after the last newline) was cut off.
    pub torn_tail: bool,
    /// Complete records replayed.
    pub records: usize,
}

/// An operation's key. Ordered by `scope_id`, then `operation_id` (the [`OpJournal::list`] order).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpKey {
    pub scope_id: String,
    pub operation_id: String,
}

/// What a duplicate must match. `payload_sha256` is the Q12 digest; it is only fed to the keyed HMAC.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OpIdentity {
    pub method: &'static str,
    pub payload_sha256: [u8; 32],
    pub worker_epoch: u64,
    pub model_generation: u64,
    pub coordinator_term: u64,
}

impl fmt::Debug for OpIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpIdentity")
            .field("method", &self.method)
            .field("payload_sha256", &"<redacted>")
            .field("worker_epoch", &self.worker_epoch)
            .field("model_generation", &self.model_generation)
            .field("coordinator_term", &self.coordinator_term)
            .finish()
    }
}

/// The receipt of an accepted operation; a duplicate gets the original's ticket and deadline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub ticket: String,
    /// True only for the first accept of a key: the caller performs the backend side effect only then.
    pub fresh: bool,
    pub deadline_ms: u64,
}

/// An operation's journaled state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpView {
    pub key: OpKey,
    pub method: &'static str,
    pub request_id: String,
    pub attempt: u64,
    pub ticket: String,
    pub worker_epoch: u64,
    pub model_generation: u64,
    pub coordinator_term: u64,
    pub terminal: bool,
    /// The terminal reason; empty while nonterminal.
    pub reason: String,
    pub cleanup_acknowledged: bool,
    pub deadline_ms: u64,
}

#[derive(Debug)]
pub enum OpError {
    /// Same key, different identity (`ALREADY_EXISTS operation_conflict`).
    Conflict,
    /// A new key while `max_operations` are live or retained (`RESOURCE_EXHAUSTED journal_full`).
    Full,
    /// An earlier write failed (`UNAVAILABLE poisoned`).
    Poisoned,
    /// No operation under this key (`NOT_FOUND unknown_operation`).
    Unknown,
    /// This call's write failed (the journal is now poisoned), or the request was invalid before any write.
    Io(io::Error),
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conflict => f.write_str("operation conflict"),
            Self::Full => f.write_str("operation journal full"),
            Self::Poisoned => f.write_str("operation journal poisoned by an earlier write failure"),
            Self::Unknown => f.write_str("unknown operation"),
            Self::Io(e) => write!(f, "operation journal: {e}"),
        }
    }
}

impl std::error::Error for OpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpRecord {
    Accepted(AcceptedRecord),
    Terminal(TerminalRecord),
    Cleanup(CleanupRecord),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptedRecord {
    scope_id: String,
    operation_id: String,
    method: String,
    /// Hex `HMAC-SHA256(agent_secret, len(scope_id) ‖ scope_id ‖ payload_sha256)`.
    digest: String,
    request_id: String,
    attempt: u64,
    ticket: String,
    worker_epoch: u64,
    model_generation: u64,
    coordinator_term: u64,
    accepted_ms: u64,
    deadline_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TerminalRecord {
    scope_id: String,
    operation_id: String,
    reason: String,
    at_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CleanupRecord {
    scope_id: String,
    operation_id: String,
    at_ms: u64,
}

#[derive(Debug, Clone)]
struct Op {
    method: &'static str,
    digest: [u8; 32],
    request_id: String,
    attempt: u64,
    ticket: String,
    worker_epoch: u64,
    model_generation: u64,
    coordinator_term: u64,
    accepted_ms: u64,
    deadline_ms: u64,
    terminal: Option<(String, u64)>,
    cleanup_ms: Option<u64>,
}

impl Op {
    /// When this op's tombstone is dropped; `None` while it must be kept (nonterminal or not cleaned up).
    fn drop_at(&self, limits: &OpLimits) -> Option<u64> {
        let (_, terminal_ms) = self.terminal.as_ref()?;
        self.cleanup_ms?;
        Some(
            terminal_ms
                .saturating_add(limits.retry_horizon_ms)
                .max(self.deadline_ms.saturating_add(limits.rpc_timeout_ms)),
        )
    }

    fn view(&self, key: &OpKey) -> OpView {
        OpView {
            key: key.clone(),
            method: self.method,
            request_id: self.request_id.clone(),
            attempt: self.attempt,
            ticket: self.ticket.clone(),
            worker_epoch: self.worker_epoch,
            model_generation: self.model_generation,
            coordinator_term: self.coordinator_term,
            terminal: self.terminal.is_some(),
            reason: self
                .terminal
                .as_ref()
                .map_or_else(String::new, |(reason, _)| reason.clone()),
            cleanup_acknowledged: self.cleanup_ms.is_some(),
            deadline_ms: self.deadline_ms,
        }
    }

    fn records(&self, key: &OpKey) -> Vec<OpRecord> {
        let mut out = vec![OpRecord::Accepted(AcceptedRecord {
            scope_id: key.scope_id.clone(),
            operation_id: key.operation_id.clone(),
            method: self.method.to_owned(),
            digest: hex(&self.digest),
            request_id: self.request_id.clone(),
            attempt: self.attempt,
            ticket: self.ticket.clone(),
            worker_epoch: self.worker_epoch,
            model_generation: self.model_generation,
            coordinator_term: self.coordinator_term,
            accepted_ms: self.accepted_ms,
            deadline_ms: self.deadline_ms,
        })];
        if let Some((reason, at_ms)) = &self.terminal {
            out.push(OpRecord::Terminal(TerminalRecord {
                scope_id: key.scope_id.clone(),
                operation_id: key.operation_id.clone(),
                reason: reason.clone(),
                at_ms: *at_ms,
            }));
        }
        if let Some(at_ms) = self.cleanup_ms {
            out.push(OpRecord::Cleanup(CleanupRecord {
                scope_id: key.scope_id.clone(),
                operation_id: key.operation_id.clone(),
                at_ms,
            }));
        }
        out
    }
}

/// The agent operation journal; see the module docs.
pub struct OpJournal {
    dir: PathBuf,
    file: File,
    len: u64,
    lines: u64,
    compacted_lines: u64,
    limits: OpLimits,
    secret: [u8; 32],
    ops: BTreeMap<OpKey, Op>,
    poisoned: bool,
}

impl fmt::Debug for OpJournal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpJournal")
            .field("dir", &self.dir)
            .field("ops", &self.ops.len())
            .field("lines", &self.lines)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl OpJournal {
    /// Opens (creating if needed) the journal and the agent secret under `state_dir`, replays the journal
    /// (cutting a torn final line), drops tombstones due at `now_ms` and compacts.
    pub fn open(
        state_dir: &Path,
        limits: OpLimits,
        now_ms: u64,
    ) -> io::Result<(OpJournal, OpReplay)> {
        ensure_private_dir(state_dir)?;
        let secret = open_secret(state_dir, SECRET_FILE)?;
        remove_if_exists(&state_dir.join(COMPACT_TMP_FILE))?;
        let path = state_dir.join(OPERATIONS_FILE);
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .mode(0o600)
            .open(&path)?;
        std::fs::set_permissions(&path, Permissions::from_mode(0o600))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let good_len = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        let torn_tail = good_len < bytes.len();
        if torn_tail {
            file.set_len(good_len as u64)?;
            file.sync_all()?;
        }
        let mut ops = BTreeMap::new();
        let mut records = 0usize;
        for (index, line) in bytes[..good_len].split(|&b| b == b'\n').enumerate() {
            if line.is_empty() {
                continue;
            }
            let corrupt = |what: &dyn fmt::Display| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} line {}: {what}", path.display(), index + 1),
                )
            };
            let record: OpRecord = serde_json::from_slice(line).map_err(|e| corrupt(&e))?;
            replay_record(&mut ops, record, &limits).map_err(|e| corrupt(&e))?;
            records += 1;
        }
        let mut journal = OpJournal {
            dir: state_dir.to_owned(),
            file,
            len: good_len as u64,
            lines: records as u64,
            compacted_lines: records as u64,
            limits,
            secret,
            ops,
            poisoned: false,
        };
        journal.drop_due(now_ms);
        journal.compact()?;
        Ok((journal, OpReplay { torn_tail, records }))
    }

    /// Accepts `key` or returns its original receipt. A new key is journaled (fsync'd) before this returns.
    pub fn accept(
        &mut self,
        key: OpKey,
        identity: OpIdentity,
        request_id: &str,
        attempt: u64,
        remaining_budget_ms: u64,
        now_ms: u64,
    ) -> Result<Accepted, OpError> {
        self.check_poison()?;
        let digest = self.keyed_digest(&key.scope_id, &identity.payload_sha256);
        if let Some(op) = self.ops.get(&key) {
            let same = op.method == identity.method
                && ct_eq(&op.digest, &digest)
                && op.worker_epoch == identity.worker_epoch
                && op.model_generation == identity.model_generation
                && op.coordinator_term == identity.coordinator_term;
            if !same {
                return Err(OpError::Conflict);
            }
            return Ok(Accepted {
                ticket: op.ticket.clone(),
                fresh: false,
                deadline_ms: op.deadline_ms,
            });
        }
        let method = journaled_method(identity.method).ok_or_else(|| {
            OpError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} is not a journaled method", identity.method),
            ))
        })?;
        self.drop_due(now_ms);
        if self.ops.len() >= self.limits.max_operations {
            return Err(OpError::Full);
        }
        let ticket = hex(&random_bytes::<16>().map_err(OpError::Io)?);
        let op = Op {
            method,
            digest,
            request_id: request_id.to_owned(),
            attempt,
            ticket,
            worker_epoch: identity.worker_epoch,
            model_generation: identity.model_generation,
            coordinator_term: identity.coordinator_term,
            accepted_ms: now_ms,
            deadline_ms: now_ms.saturating_add(remaining_budget_ms.min(wire::WORK_CAP_MS)),
            terminal: None,
            cleanup_ms: None,
        };
        let records = op.records(&key);
        self.append(&records[0])?;
        let accepted = Accepted {
            ticket: op.ticket.clone(),
            fresh: true,
            deadline_ms: op.deadline_ms,
        };
        self.ops.insert(key, op);
        Ok(accepted)
    }

    /// Makes `key` terminal with `reason`. An already terminal op keeps its first reason (no write).
    pub fn mark_terminal(&mut self, key: &OpKey, reason: &str, now_ms: u64) -> Result<(), OpError> {
        self.check_poison()?;
        let op = self.ops.get(key).ok_or(OpError::Unknown)?;
        if op.terminal.is_some() {
            return Ok(());
        }
        self.append(&OpRecord::Terminal(TerminalRecord {
            scope_id: key.scope_id.clone(),
            operation_id: key.operation_id.clone(),
            reason: reason.to_owned(),
            at_ms: now_ms,
        }))?;
        if let Some(op) = self.ops.get_mut(key) {
            op.terminal = Some((reason.to_owned(), now_ms));
        }
        Ok(())
    }

    /// Records that the backend no longer holds `key`'s work (independent of its terminal state).
    pub fn mark_cleanup(&mut self, key: &OpKey, now_ms: u64) -> Result<(), OpError> {
        self.check_poison()?;
        let op = self.ops.get(key).ok_or(OpError::Unknown)?;
        if op.cleanup_ms.is_some() {
            return Ok(());
        }
        self.append(&OpRecord::Cleanup(CleanupRecord {
            scope_id: key.scope_id.clone(),
            operation_id: key.operation_id.clone(),
            at_ms: now_ms,
        }))?;
        if let Some(op) = self.ops.get_mut(key) {
            op.cleanup_ms = Some(now_ms);
        }
        Ok(())
    }

    pub fn get(&self, key: &OpKey) -> Option<OpView> {
        self.ops.get(key).map(|op| op.view(key))
    }

    /// Up to `limit` operations strictly after `after`, in `(scope_id, operation_id)` order, and whether more
    /// follow.
    pub fn list(&self, after: Option<&OpKey>, limit: usize) -> (Vec<OpView>, bool) {
        let lower = after.map_or(Bound::Unbounded, Bound::Excluded);
        let mut range = self.ops.range::<OpKey, _>((lower, Bound::Unbounded));
        let page: Vec<OpView> = range
            .by_ref()
            .take(limit)
            .map(|(key, op)| op.view(key))
            .collect();
        (page, range.next().is_some())
    }

    /// Makes every nonterminal op of an epoch below `worker_epoch` terminal `worker_restart` and returns them
    /// (the caller aborts each request id on the backend). Their tombstones are kept.
    pub fn restart(&mut self, worker_epoch: u64, now_ms: u64) -> Result<Vec<OpView>, OpError> {
        self.check_poison()?;
        let fenced: Vec<OpKey> = self
            .ops
            .iter()
            .filter(|(_, op)| op.terminal.is_none() && op.worker_epoch < worker_epoch)
            .map(|(key, _)| key.clone())
            .collect();
        let mut out = Vec::with_capacity(fenced.len());
        for key in fenced {
            self.mark_terminal(&key, wire::WORKER_RESTART, now_ms)?;
            if let Some(view) = self.get(&key) {
                out.push(view);
            }
        }
        Ok(out)
    }

    /// `SubmitWork` operations not cleanup-acknowledged: the agent's live-op bound (Q14) and the drain quiescence
    /// count (m36 design note Q4). Lifecycle and `CancelWork` ops never count.
    pub fn live_ops(&self) -> usize {
        self.ops
            .values()
            .filter(|op| op.method == wire::METHOD_SUBMIT && op.cleanup_ms.is_none())
            .count()
    }

    /// Drops tombstones due at `now_ms` and compacts a grown file. A failed compaction poisons the journal.
    pub fn expire(&mut self, now_ms: u64) {
        self.drop_due(now_ms);
        let bound = self
            .compacted_lines
            .saturating_mul(2)
            .saturating_add(self.limits.max_operations as u64);
        if !self.poisoned && self.lines > bound && self.compact().is_err() {
            self.poisoned = true;
        }
    }

    /// True once a write failed; every later mutation is refused.
    pub fn poisoned(&self) -> bool {
        self.poisoned
    }

    fn check_poison(&self) -> Result<(), OpError> {
        if self.poisoned {
            Err(OpError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn drop_due(&mut self, now_ms: u64) {
        let limits = self.limits;
        self.ops
            .retain(|_, op| op.drop_at(&limits).is_none_or(|at| now_ms < at));
    }

    fn keyed_digest(&self, scope_id: &str, payload_sha256: &[u8; 32]) -> [u8; 32] {
        let mut mac =
            HmacSha256::new_from_slice(&self.secret).expect("HMAC accepts any key length");
        mac.update(&(scope_id.len() as u64).to_be_bytes());
        mac.update(scope_id.as_bytes());
        mac.update(payload_sha256);
        mac.finalize().into_bytes().into()
    }

    /// Appends one record and fsyncs it. On failure the file is cut back to its last good length (best
    /// effort) and the journal is poisoned.
    fn append(&mut self, record: &OpRecord) -> Result<(), OpError> {
        let mut line = serde_json::to_vec(record).map_err(|e| OpError::Io(io::Error::other(e)))?;
        line.push(b'\n');
        match self
            .file
            .write_all(&line)
            .and_then(|()| self.file.sync_data())
        {
            Ok(()) => {
                self.len += line.len() as u64;
                self.lines += 1;
                Ok(())
            }
            Err(e) => {
                self.poisoned = true;
                // Best effort: a torn tail left behind is cut by the next open anyway.
                let _ = self
                    .file
                    .set_len(self.len)
                    .and_then(|()| self.file.sync_all());
                Err(OpError::Io(e))
            }
        }
    }

    /// Atomically replaces the file with the records of every kept operation (temp file, fsync, rename,
    /// directory fsync).
    fn compact(&mut self) -> io::Result<()> {
        let tmp = self.dir.join(COMPACT_TMP_FILE);
        let path = self.dir.join(OPERATIONS_FILE);
        let mut out = Vec::new();
        let mut lines = 0u64;
        for (key, op) in &self.ops {
            for record in op.records(key) {
                serde_json::to_writer(&mut out, &record).map_err(io::Error::other)?;
                out.push(b'\n');
                lines += 1;
            }
        }
        // `mode` only applies at creation: never reuse a stale temp file with looser permissions.
        remove_if_exists(&tmp)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&out)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, &path)?;
        sync_dir(&self.dir)?;
        self.file = OpenOptions::new().read(true).append(true).open(&path)?;
        self.len = out.len() as u64;
        self.lines = lines;
        self.compacted_lines = lines;
        Ok(())
    }
}

/// Applies one replayed record. Order rules: `accepted` first per key (or again only once the previous
/// tombstone was due at its receipt instant), `terminal` and `cleanup` at most once each after it.
fn replay_record(
    ops: &mut BTreeMap<OpKey, Op>,
    record: OpRecord,
    limits: &OpLimits,
) -> Result<(), String> {
    match record {
        OpRecord::Accepted(r) => {
            let key = OpKey {
                scope_id: r.scope_id,
                operation_id: r.operation_id,
            };
            if let Some(old) = ops.get(&key) {
                if old.drop_at(limits).is_none_or(|at| r.accepted_ms < at) {
                    return Err("operation accepted twice".to_owned());
                }
            }
            let method = journaled_method(&r.method)
                .ok_or_else(|| format!("unknown method {:?}", r.method))?;
            let digest = unhex(&r.digest)
                .and_then(|b| <[u8; 32]>::try_from(b).ok())
                .ok_or("digest must be 64 hex chars")?;
            ops.insert(
                key,
                Op {
                    method,
                    digest,
                    request_id: r.request_id,
                    attempt: r.attempt,
                    ticket: r.ticket,
                    worker_epoch: r.worker_epoch,
                    model_generation: r.model_generation,
                    coordinator_term: r.coordinator_term,
                    accepted_ms: r.accepted_ms,
                    deadline_ms: r.deadline_ms,
                    terminal: None,
                    cleanup_ms: None,
                },
            );
        }
        OpRecord::Terminal(r) => {
            let op = replayed_op(ops, r.scope_id, r.operation_id)?;
            if op.terminal.is_some() {
                return Err("operation terminal twice".to_owned());
            }
            op.terminal = Some((r.reason, r.at_ms));
        }
        OpRecord::Cleanup(r) => {
            let op = replayed_op(ops, r.scope_id, r.operation_id)?;
            if op.cleanup_ms.is_some() {
                return Err("operation cleaned up twice".to_owned());
            }
            op.cleanup_ms = Some(r.at_ms);
        }
    }
    Ok(())
}

fn replayed_op(
    ops: &mut BTreeMap<OpKey, Op>,
    scope_id: String,
    operation_id: String,
) -> Result<&mut Op, String> {
    ops.get_mut(&OpKey {
        scope_id,
        operation_id,
    })
    .ok_or_else(|| "record for an operation never accepted".to_owned())
}

fn journaled_method(method: &str) -> Option<&'static str> {
    METHODS.into_iter().find(|m| *m == method)
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Constant-time equality of two keyed digests.
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}
