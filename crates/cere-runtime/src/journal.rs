//! Durable journal and epoch store (m1 design note L3).
//!
//! `<ledger.state_dir>/journal.jsonl`: one JSON [`Record`] per line, each append `fsync`'d before it returns.
//! The directory is 0700 and the file 0600. [`Journal::open`] first takes an exclusive, non-blocking `flock` on
//! `<state_dir>/lock` (0600), held until the journal is dropped (the process lifetime for `cered`): a second
//! instance on the same directory fails before it replays, recovers, aborts or compacts anything. It then
//! replays the file, appends the next epoch record and fsyncs it before returning, so a server never serves
//! under an epoch that is not on disk.
//!
//! Crash tolerance: an append is a single `write` of `line + "\n"`; a crash can leave a torn final line without
//! its newline. Bytes after the last newline are therefore never acknowledged and are cut off on open. Any
//! complete line that does not parse is corruption and fails the open (fail closed).
//!
//! Compaction ([`Journal::compact`]) writes the current epoch record plus the caller's live records to a
//! temporary file, fsyncs it, renames it over the journal and fsyncs the directory: a crash leaves either the
//! old or the new journal. [`Journal::needs_compaction`] bounds the file to about twice the last compacted
//! size plus a slack, in lines and in bytes (a session commit re-journals the session's whole revision list, so a
//! long session grows the file far faster in bytes than in lines). The new file is opened before the rename and
//! becomes the journal's handle the moment the rename succeeds; a failure after the rename poisons the journal.
//!
//! M3 (m3 design note Q4, Q9) adds `worker_epoch` (a worker's epoch, monotonic per worker id across restarts)
//! and `dispatched` (the worker a request was sent to, journaled immediately before the backend call). Records
//! written before M3 parse unchanged: a missing `dispatch_journaled` reads as `false`.
//!
//! M4 (m4 design note Q3) makes the session owner durable: `dispatched.session_owner` (a grant that changed
//! the owner), `session.owner` (creation, fork and compaction snapshots) and `session_owner` (an operator
//! migration). The fields are absent when empty, so older journals replay unchanged and a journal holding no owner
//! data stays readable by an older `cered`; one holding owner data is refused by it (forward-only).
//!
//! OH-03 (ops-hardening design note, `ledger.evict_terminal_when_full`) adds no record type, so a release before it
//! still opens the journal (rollback): a keyed terminal record a full ledger evicts is journaled as `key_expired`, and
//! an unkeyed one is evicted again when replay leaves the ledger over its bound.
//!
//! Records hold scope ids, idempotency keys, request ids, keyed digests, salts, states, reason codes, timings,
//! reported token counts, worker ids and backend base URLs. Never prompt or output text.

use serde::{Deserialize, Serialize};
use std::fs::{DirBuilder, File, OpenOptions, Permissions};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use crate::api_types::RequestUsage;
use crate::digest::{Digest, Salt};
use crate::ledger::{TerminalState, Timings};
use crate::session::{ForkParent, Revision};

pub const JOURNAL_FILE: &str = "journal.jsonl";
/// Exclusive-lock file of a state directory (see [`Journal::open`]).
pub const LOCK_FILE: &str = "lock";
const COMPACT_TMP_FILE: &str = "journal.jsonl.compact.tmp";
/// Byte slack of the size trigger in [`Journal::needs_compaction`]: the file may grow to twice the last compacted
/// size plus this many bytes before the next write compacts it.
const COMPACTION_SLACK_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Record {
    /// A new process incarnation; the highest epoch in the file is the previous incarnation's.
    Epoch { epoch: u64, at_ms: u64 },
    /// A worker's new epoch (Q4); the highest per worker id is current. Compaction writes one per worker id ever
    /// seen, right after the epoch record, so an epoch is never reused.
    WorkerEpoch(WorkerEpochRecord),
    /// Written before the request can reach the backend.
    Admitted(AdmittedRecord),
    /// The request is sent to a worker next (Q9). A rid's last `dispatched` record names the worker that may
    /// hold its backend work.
    Dispatched(DispatchedRecord),
    /// The one terminal of a request, optionally with the session revision it commits.
    Terminal(TerminalRecord),
    /// A created/forked session, or a session snapshot written by compaction (replaces earlier state), with its
    /// owner (M4 Q3c).
    Session(SessionRecord),
    /// An operator migration moved the session's owner (m4 design note Q3b, Q5): fsynced before it applies and
    /// before the route answers; replay applies it only if its generation is above the session's.
    SessionOwner(SessionOwnerRecord),
    /// An idempotency key whose record outlived its retention (written by compaction), or whose terminal record a full
    /// ledger evicted (OH-03; replay then drops that record).
    KeyExpired {
        scope: String,
        key: String,
        at_ms: u64,
    },
    /// A recovered orphan's admission slot was released (m1close Q7): never charged again after a restart.
    OrphanReleased(OrphanReleasedRecord),
    /// An H2 hold of a remote request's slot started (M3.5 Q15): its agent may still hold the work. Until an
    /// `orphan_released` record follows, a restart charges the slot to that worker again until the request's
    /// original bound; compaction re-emits it for every live request held and not released.
    OrphanHeld(OrphanHeldRecord),
    /// Operator drain intent of a worker id (M3.5 Q13), journaled on every change; compaction re-emits one per
    /// draining id, so the intent survives every compaction until an undrain clears it.
    WorkerDrain {
        worker: String,
        draining: bool,
        at_ms: u64,
    },
    /// Operator residency intent of a remote worker (m36 design note Q8): every `load`/`unload` call is a new
    /// intent with `intent_seq` one above the worker's previous one, fsynced before the route answers.
    /// Compaction re-emits the latest record per worker; replay keeps the last one per worker.
    WorkerResidency {
        worker: String,
        desired: Residency,
        intent_seq: u64,
        at_ms: u64,
    },
}

/// The residency an operator asked for on a remote worker (m36 design note Q8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    Resident,
    Unloaded,
}

/// Why a recovered orphan's slot was released.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrphanReleaseCause {
    /// The backend load showed no work beyond `cered`'s own confirmed streams (Q6).
    Load,
    /// `admitted_ms + terminal_retention_ms` passed.
    Retention,
    /// The worker's backend relaunched (a new `instance_id`): its work is gone (M3 Q10).
    Relaunch,
    /// An M3 request that was never dispatched (no `dispatched` record): no backend holds work for it (M3 Q9).
    NotDispatched,
    /// The remote worker's agent reported the operation `cleanup_acknowledged` (M3.5 Q15).
    CleanupAcknowledged,
}

impl OrphanReleaseCause {
    /// The journal and log spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Retention => "retention",
            Self::Relaunch => "relaunch",
            Self::NotDispatched => "not_dispatched",
            Self::CleanupAcknowledged => "cleanup_acknowledged",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerEpochRecord {
    pub worker: String,
    pub epoch: u64,
    pub at_ms: u64,
}

/// Where a dispatched worker runs (M3.5 Q11): `cered`'s own host or a federated agent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Node {
    #[default]
    Local,
    Remote,
}

impl Node {
    fn is_local(&self) -> bool {
        *self == Self::Local
    }
}

/// `base_url` is present iff `node` is local, `endpoint` iff remote ([`crate::ledger::Ledger`] checks it on replay).
/// A local record omits `node` (absent = local), so local lines are byte-identical to M3 ones and an M3 journal
/// replays unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchedRecord {
    pub rid: String,
    pub worker: String,
    pub worker_epoch: u64,
    #[serde(default, skip_serializing_if = "Node::is_local")]
    pub node: Node,
    /// A local worker's configured `base_url` at dispatch: recovery aborts only where this URL is served.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// A remote worker's configured agent endpoint at dispatch (Q15).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// M4 Q3a: this grant changed the session's owner to `worker` at this generation (absent otherwise, and in
    /// compaction's re-emitted records: the session snapshot carries the owner).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_owner: Option<OwnerRef>,
    pub at_ms: u64,
}

/// The session owner a `dispatched` record publishes; the worker is the record's own (M4 Q3a).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerRef {
    pub session_id: String,
    pub owner_generation: u64,
}

/// A session's owner worker and generation, as its [`SessionRecord`] carries it (M4 Q3c).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerRecord {
    pub worker: String,
    pub owner_generation: u64,
}

/// An operator migration (M4 Q5 step 7): the new owner at `owner_generation`, and the revision and digest the
/// session was frozen at, which replay checks against the replayed session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionOwnerRecord {
    pub session_id: String,
    pub worker: String,
    pub owner_generation: u64,
    pub revision: u64,
    pub digest: Digest,
    pub at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrphanReleasedRecord {
    pub rid: String,
    pub at_ms: u64,
    pub cause: OrphanReleaseCause,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrphanHeldRecord {
    pub rid: String,
    /// The remote worker id of the rid's last `dispatched` record, whose agent may hold the work.
    pub worker: String,
    pub at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRef {
    pub session_id: String,
    pub expected_revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedRecord {
    pub scope: String,
    pub rid: String,
    pub key: Option<String>,
    pub digest: Digest,
    pub epoch: u64,
    pub stream: bool,
    pub session: Option<SessionRef>,
    pub at_ms: u64,
    /// Written by an M3 ledger, which journals `dispatched` before any backend call: without a `dispatched`
    /// record the request never reached a backend. Absent (false) in older records.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dispatch_journaled: bool,
    /// M2.14: the API surface (`anthropic`, `responses`); absent for chat, so chat records keep their bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionAdvance {
    pub session_id: String,
    pub revision: u64,
    pub entry: Revision,
    /// The committed turn's model (M5 D7); absent in records written before M5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalRecord {
    pub rid: String,
    pub state: TerminalState,
    pub reason: Option<String>,
    pub timings: Timings,
    pub session: Option<SessionAdvance>,
    pub at_ms: u64,
    /// The request ended before it was dispatched to the backend in this incarnation, so its Idempotency-Key
    /// was released (review finding 2): replay drops the binding. Absent (false) in older records.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub key_released: bool,
    /// What the backend reported for the request (Q1/Q5). Absent in records written before M1.4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<RequestUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRecord {
    pub scope: String,
    pub session_id: String,
    pub salt: Salt,
    pub parent: Option<ForkParent>,
    pub revisions: Vec<Revision>,
    pub updated_ms: u64,
    /// The durable owner (M4 Q3c); absent for an unowned session and in records written before M4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<OwnerRecord>,
    /// The bound model (M5 D7); absent for an unbound session and in records written before M5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// What [`Journal::open`] found on disk.
#[derive(Debug)]
pub struct Replay {
    /// Records in file order, excluding the epoch record appended by this open.
    pub records: Vec<Record>,
    /// Bytes of a torn final line that were cut off (0 for a clean file).
    pub truncated_tail_bytes: u64,
}

#[derive(Debug)]
pub struct Journal {
    dir: PathBuf,
    /// Holds the exclusive `flock` on `<dir>/lock`; released when the journal (the open file) is dropped.
    _lock: File,
    file: File,
    len: u64,
    epoch: u64,
    epoch_at_ms: u64,
    lines: u64,
    compacted_lines: u64,
    compacted_len: u64,
    poisoned: bool,
}

impl Journal {
    /// Locks `state_dir` (fails with `WouldBlock` while another journal holds it), opens (creating if needed)
    /// the journal under it, replays it, and durably records epoch `previous + 1` (1 for a fresh directory)
    /// before returning.
    pub fn open(state_dir: &Path, now_ms: u64) -> io::Result<(Self, Replay)> {
        ensure_private_dir(state_dir)?;
        let lock = lock_dir(state_dir, "cered")?;
        remove_if_exists(&state_dir.join(COMPACT_TMP_FILE))?;
        let path = state_dir.join(JOURNAL_FILE);
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
        let truncated_tail_bytes = (bytes.len() - good_len) as u64;
        if truncated_tail_bytes > 0 {
            file.set_len(good_len as u64)?;
            file.sync_all()?;
        }
        let mut records = Vec::new();
        let mut previous = 0u64;
        for (index, line) in bytes[..good_len].split(|&b| b == b'\n').enumerate() {
            if line.is_empty() {
                continue;
            }
            let record: Record = serde_json::from_slice(line).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} line {}: {e}", path.display(), index + 1),
                )
            })?;
            if let Record::Epoch { epoch, .. } = record {
                if epoch <= previous {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{} line {}: epoch {epoch} not increasing",
                            path.display(),
                            index + 1
                        ),
                    ));
                }
                previous = epoch;
            }
            records.push(record);
        }
        let epoch = previous
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "epoch overflow"))?;
        let lines = records.len() as u64;
        let mut journal = Self {
            dir: state_dir.to_owned(),
            _lock: lock,
            file,
            len: good_len as u64,
            epoch,
            epoch_at_ms: now_ms,
            lines,
            compacted_lines: lines,
            compacted_len: good_len as u64,
            poisoned: false,
        };
        journal.append(&Record::Epoch {
            epoch,
            at_ms: now_ms,
        })?;
        sync_dir(state_dir)?;
        Ok((
            journal,
            Replay {
                records,
                truncated_tail_bytes,
            },
        ))
    }

    /// The epoch of this process incarnation.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Appends one record and fsyncs it. A failed write is rolled back to the previous length; if that fails
    /// too, the journal refuses every further append.
    pub fn append(&mut self, record: &Record) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal poisoned by an earlier write failure",
            ));
        }
        let mut line = serde_json::to_vec(record).map_err(io::Error::other)?;
        line.push(b'\n');
        let result = self
            .file
            .write_all(&line)
            .and_then(|()| self.file.sync_data());
        match result {
            Ok(()) => {
                self.len += line.len() as u64;
                self.lines += 1;
                Ok(())
            }
            Err(e) => {
                if self
                    .file
                    .set_len(self.len)
                    .and_then(|()| self.file.sync_all())
                    .is_err()
                {
                    self.poisoned = true;
                }
                Err(e)
            }
        }
    }

    /// True when the file has grown past `2 × last compacted size + slack` lines, or past `2 × last compacted
    /// size + COMPACTION_SLACK_BYTES` bytes.
    pub fn needs_compaction(&self, slack: u64) -> bool {
        self.needs_compaction_with(slack, COMPACTION_SLACK_BYTES)
    }

    fn needs_compaction_with(&self, slack_lines: u64, slack_bytes: u64) -> bool {
        self.lines
            > self
                .compacted_lines
                .saturating_mul(2)
                .saturating_add(slack_lines)
            || self.len
                > self
                    .compacted_len
                    .saturating_mul(2)
                    .saturating_add(slack_bytes)
    }

    /// Lines currently in the journal file.
    pub fn lines(&self) -> u64 {
        self.lines
    }

    /// True once a write failure left the file or the caller's state unrecorded; every further write fails.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Refuses every further write: the caller's state no longer matches the file.
    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Atomically replaces the journal with this epoch's record followed by `live` (temp file, fsync, rename,
    /// directory fsync). The new file is opened before the rename, so a failure up to and including the rename
    /// leaves the old file and every counter untouched; once the rename succeeds the new file is the journal's
    /// handle, and a failed directory fsync (the rename may not be durable) poisons the journal.
    pub fn compact(&mut self, live: &[Record]) -> io::Result<()> {
        self.compact_with(live, sync_dir)
    }

    fn compact_with(
        &mut self,
        live: &[Record],
        sync: impl FnOnce(&Path) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal poisoned by an earlier write failure",
            ));
        }
        let tmp = self.dir.join(COMPACT_TMP_FILE);
        let path = self.dir.join(JOURNAL_FILE);
        let mut out = Vec::new();
        let epoch = Record::Epoch {
            epoch: self.epoch,
            at_ms: self.epoch_at_ms,
        };
        for record in std::iter::once(&epoch).chain(live) {
            serde_json::to_writer(&mut out, record).map_err(io::Error::other)?;
            out.push(b'\n');
        }
        // `mode` only applies at creation: never reuse a stale temp file with looser permissions.
        remove_if_exists(&tmp)?;
        // Read and append like the handle `open` returns; the descriptor follows the inode through the rename.
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&out)?;
        file.sync_all()?;
        std::fs::rename(&tmp, &path)?;
        self.file = file;
        self.len = out.len() as u64;
        self.lines = 1 + live.len() as u64;
        self.compacted_lines = self.lines;
        self.compacted_len = self.len;
        if let Err(e) = sync(&self.dir) {
            self.poisoned = true;
            return Err(e);
        }
        Ok(())
    }
}

/// Creates `dir` (and missing parents) with mode 0700 and tightens an existing `dir` to 0700.
pub(crate) fn ensure_private_dir(dir: &Path) -> io::Result<()> {
    DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    std::fs::set_permissions(dir, Permissions::from_mode(0o700))
}

/// How long `lock_dir` retries a held lock. A descriptor of the lock can outlive its holder for a moment: another
/// thread of the same process that forks a child duplicates it into the child until `exec` closes it (O_CLOEXEC).
/// A second `cered`/`cere-worker` on the same state dir holds the lock far longer and is still refused.
const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// Opens `<dir>/lock` (0600) and takes an exclusive `flock` on it, held while the returned file lives; a held lock is
/// retried for up to [`LOCK_WAIT`]. `owner` names the binary in the refusal (`cered`, `cere-worker`).
pub(crate) fn lock_dir(dir: &Path, owner: &str) -> io::Result<File> {
    let path = dir.join(LOCK_FILE);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(&path)?;
    std::fs::set_permissions(&path, Permissions::from_mode(0o600))?;
    let deadline = std::time::Instant::now() + LOCK_WAIT;
    loop {
        // SAFETY: `file` owns a valid open descriptor for the duration of the call; `flock` touches no memory.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(file);
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::WouldBlock {
            return Err(e);
        }
        if std::time::Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "{} is locked by another {owner} process using this state dir",
                    path.display()
                ),
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Makes a directory entry change (create/rename) durable.
pub(crate) fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cere-journal-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn key_expired(i: u64) -> Record {
        Record::KeyExpired {
            scope: "s".into(),
            key: format!("k{i}"),
            at_ms: i,
        }
    }

    fn file_lines(dir: &Path) -> Vec<Record> {
        std::fs::read_to_string(dir.join(JOURNAL_FILE))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// A lock held only briefly (e.g. a descriptor duplicated into a child between `fork` and `exec` by another
    /// thread of this process) does not refuse the open: `lock_dir` waits up to `LOCK_WAIT` for it.
    #[test]
    fn a_briefly_held_lock_is_waited_for() {
        let dir = temp_dir("lock-brief");
        ensure_private_dir(&dir).unwrap();
        let held = lock_dir(&dir, "cered").unwrap();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            drop(held);
        });
        let second = lock_dir(&dir, "cered");
        releaser.join().unwrap();
        assert!(second.is_ok(), "{second:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_lock_held_past_the_wait_is_refused_with_the_owner_named() {
        let dir = temp_dir("lock-held");
        ensure_private_dir(&dir).unwrap();
        let _held = lock_dir(&dir, "cered").unwrap();
        let e = lock_dir(&dir, "cere-worker").unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::WouldBlock);
        assert!(
            e.to_string()
                .contains("locked by another cere-worker process"),
            "{e}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn release_cause_log_spelling_is_the_journal_spelling() {
        for cause in [
            OrphanReleaseCause::Load,
            OrphanReleaseCause::Retention,
            OrphanReleaseCause::Relaunch,
            OrphanReleaseCause::NotDispatched,
            OrphanReleaseCause::CleanupAcknowledged,
        ] {
            assert_eq!(serde_json::to_value(cause).unwrap(), cause.as_str());
        }
        assert_eq!(
            OrphanReleaseCause::CleanupAcknowledged.as_str(),
            "cleanup_acknowledged"
        );
    }

    #[test]
    fn epoch_increments_per_open_and_is_on_disk() {
        let dir = temp_dir("epoch").join("nested");
        let (j, replay) = Journal::open(&dir, 10).unwrap();
        assert_eq!((j.epoch(), replay.records.len()), (1, 0));
        drop(j);
        let (j, replay) = Journal::open(&dir, 20).unwrap();
        assert_eq!(j.epoch(), 2);
        assert_eq!(
            replay.records,
            vec![Record::Epoch {
                epoch: 1,
                at_ms: 10
            }]
        );
        assert_eq!(
            file_lines(&dir).last(),
            Some(&Record::Epoch {
                epoch: 2,
                at_ms: 20
            })
        );
        assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o777, 0o700);
        assert_eq!(
            std::fs::metadata(dir.join(JOURNAL_FILE)).unwrap().mode() & 0o777,
            0o600
        );
        std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }

    #[test]
    fn torn_tail_is_cut_and_corrupt_line_fails_closed() {
        let dir = temp_dir("torn");
        let (mut j, _) = Journal::open(&dir, 0).unwrap();
        j.append(&key_expired(1)).unwrap();
        drop(j);
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.join(JOURNAL_FILE))
            .unwrap();
        f.write_all(br#"{"type":"key_expired","scope":"s","ke"#)
            .unwrap();
        drop(f);
        let (mut j, replay) = Journal::open(&dir, 1).unwrap();
        assert_eq!(replay.truncated_tail_bytes, 37);
        assert_eq!(replay.records.len(), 2);
        // The next append starts on a clean line.
        j.append(&key_expired(2)).unwrap();
        drop(j);
        assert_eq!(file_lines(&dir).len(), 4);
        // A complete but unparsable line in the middle is corruption.
        let text = std::fs::read_to_string(dir.join(JOURNAL_FILE)).unwrap();
        std::fs::write(dir.join(JOURNAL_FILE), format!("garbage\n{text}")).unwrap();
        let err = Journal::open(&dir, 2).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn compaction_is_atomic_bounded_and_private() {
        let dir = temp_dir("compact");
        let (mut j, _) = Journal::open(&dir, 0).unwrap();
        for i in 0..10 {
            j.append(&key_expired(i)).unwrap();
        }
        assert!(j.needs_compaction(4));
        assert!(!j.needs_compaction(20));
        // A stale temp file from a crashed compaction is ignored and replaced.
        std::fs::write(dir.join(COMPACT_TMP_FILE), "junk").unwrap();
        j.compact(&[key_expired(9)]).unwrap();
        assert!(!dir.join(COMPACT_TMP_FILE).exists());
        assert_eq!(j.lines(), 2);
        assert!(!j.needs_compaction(4));
        j.append(&key_expired(10)).unwrap();
        assert_eq!(
            file_lines(&dir),
            vec![
                Record::Epoch { epoch: 1, at_ms: 0 },
                key_expired(9),
                key_expired(10)
            ]
        );
        assert_eq!(
            std::fs::metadata(dir.join(JOURNAL_FILE)).unwrap().mode() & 0o777,
            0o600
        );
        drop(j);
        std::fs::write(dir.join(COMPACT_TMP_FILE), "junk").unwrap();
        let (j, replay) = Journal::open(&dir, 5).unwrap();
        assert_eq!(j.epoch(), 2);
        assert_eq!(replay.records.len(), 3);
        assert!(!dir.join(COMPACT_TMP_FILE).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// F2: a rename that fails leaves the old file, the counters and the journal usable.
    #[test]
    fn a_failed_compaction_before_the_rename_leaves_the_journal_unchanged() {
        let dir = temp_dir("compact-fail");
        let (mut j, _) = Journal::open(&dir, 0).unwrap();
        j.append(&key_expired(1)).unwrap();
        // `remove_file` refuses a directory in the temp file's place, so the compaction fails before it touches
        // the journal.
        std::fs::create_dir(dir.join(COMPACT_TMP_FILE)).unwrap();
        assert!(j.compact(&[key_expired(9)]).is_err());
        assert!(!j.is_poisoned());
        assert_eq!(j.lines(), 2);
        std::fs::remove_dir(dir.join(COMPACT_TMP_FILE)).unwrap();
        j.append(&key_expired(2)).unwrap();
        assert_eq!(file_lines(&dir).len(), 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// F2: after the rename the new file is the journal's handle, and a directory fsync that fails poisons the
    /// journal, so nothing is acknowledged against a file whose replacement may not be durable.
    #[test]
    fn a_failed_directory_sync_after_the_rename_poisons_the_journal() {
        let dir = temp_dir("compact-sync-fail");
        let (mut j, _) = Journal::open(&dir, 0).unwrap();
        for i in 0..4 {
            j.append(&key_expired(i)).unwrap();
        }
        let err = j
            .compact_with(&[key_expired(9)], |_| Err(io::Error::other("EIO")))
            .unwrap_err();
        assert_eq!(err.to_string(), "EIO");
        assert!(j.is_poisoned());
        assert!(j.append(&key_expired(10)).is_err());
        assert!(j.compact(&[]).is_err());
        // The file on disk is the compacted one, and nothing was appended to any other inode.
        assert_eq!(
            file_lines(&dir),
            vec![Record::Epoch { epoch: 1, at_ms: 0 }, key_expired(9)]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// F2: the handle opened before the rename is the one that receives the next append.
    #[test]
    fn an_append_after_a_compaction_lands_in_the_new_file() {
        let dir = temp_dir("compact-handle");
        let (mut j, _) = Journal::open(&dir, 0).unwrap();
        j.append(&key_expired(1)).unwrap();
        j.compact(&[key_expired(1)]).unwrap();
        j.append(&key_expired(2)).unwrap();
        let on_disk = std::fs::metadata(dir.join(JOURNAL_FILE)).unwrap();
        assert_eq!(on_disk.len(), j.len);
        assert_eq!(file_lines(&dir).last(), Some(&key_expired(2)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// F3: the byte trigger fires on a few long lines that the line-count trigger would let grow, and both
    /// triggers clear after a compaction.
    #[test]
    fn the_byte_budget_triggers_compaction_independently_of_the_line_count() {
        let dir = temp_dir("compact-bytes");
        let (mut j, _) = Journal::open(&dir, 0).unwrap();
        j.compact(&[key_expired(0)]).unwrap();
        let slack_lines = 1_000;
        let slack_bytes = 4_096;
        assert!(!j.needs_compaction_with(slack_lines, slack_bytes));
        // Three lines of ~2 KiB: far under the line budget, over `2 × compacted + 4 KiB` bytes.
        let long = |i: u64| Record::KeyExpired {
            scope: "s".into(),
            key: "k".repeat(2_000) + &i.to_string(),
            at_ms: i,
        };
        j.append(&long(1)).unwrap();
        j.append(&long(2)).unwrap();
        assert!(!j.needs_compaction_with(slack_lines, slack_bytes));
        j.append(&long(3)).unwrap();
        assert!(j.lines() < 10);
        assert!(j.needs_compaction_with(slack_lines, slack_bytes));
        // The shipped slack is far larger: the same file does not trigger.
        assert!(!j.needs_compaction(slack_lines));
        j.compact(&[long(3)]).unwrap();
        assert!(!j.needs_compaction_with(slack_lines, slack_bytes));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
