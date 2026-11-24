//! The managed backend's supervisor (m36 design note Q3, row P3). Linux only (`/proc`, `prctl`).
//!
//! One dedicated `std::thread` per agent spawns the backend, watches it and stops it. `PR_SET_PDEATHSIG` follows
//! the thread that forked, so the spawner is never a tokio worker or `spawn_blocking` thread, whose exit would kill
//! a healthy backend (review M36-07); async callers reach the thread over a channel and get their answer through a
//! oneshot. Between fork and exec the child runs `setpgid(0, 0)`, the privilege drop (`setgroups([])`, `setgid`,
//! `setuid`) when the spec sets `uid`/`gid`, then `prctl(PR_SET_PDEATHSIG, SIGKILL)` and `getppid() == parent`.
//! The child's environment is the spec env plus `PATH=/usr/bin:/bin` only; stdin is null, stdout/stderr are the
//! agent's.
//!
//! `<state_dir>/backend_child.json` holds `{pid, pgid, start_time_ticks}` (`/proc/<pid>/stat` field 22) of the
//! group leader. It is written atomically (temp file 0600, fsync, rename, directory fsync) before
//! [`Supervisor::spawn`] returns, and deleted only once the whole group is gone: [`Supervisor::stop`] sends TERM
//! to the group, waits up to `stop_timeout` for the leader, sends KILL, reaps the leader and waits up to
//! `stop_timeout` more for `kill(-pgid, 0)` to fail with `ESRCH` (review M36-08). [`reconcile_at_start`] kills a
//! group the previous agent left behind.
//!
//! Only the recorded group is ever signalled. The leader stays this thread's unreaped child until `stop`, so its
//! pid keeps the pgid allocated and `kill(-pgid, ..)` cannot reach a later group that reused the number: a leader
//! that exits on its own is observed with `waitid(WNOWAIT)` (reported once as a [`ChildExit`]) and reaped by the
//! next `stop`, after the group KILL. Once the leader is reaped, only the existence probe `kill(-pgid, 0)` goes to
//! the group. Start reconciliation signals single pids it has just matched by pgrp and starttime, and nothing at
//! all when the recorded leader pid belongs to a process with another starttime.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::journal::sync_dir;

/// The group record inside the agent `state_dir`.
pub const CHILD_FILE: &str = "backend_child.json";
const CHILD_TMP_FILE: &str = "backend_child.json.tmp";
/// The child's fixed `PATH`; nothing else of the agent's environment reaches it (Q3).
const CHILD_PATH: &str = "/usr/bin:/bin";
/// How often the idle thread looks for the leader's exit.
const EXIT_POLL: Duration = Duration::from_millis(100);
/// Poll period while stopping or reconciling.
const STEP: Duration = Duration::from_millis(10);

/// What the supervisor launches; the agent builds it from `[backend.launch]` (Q3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchSpec {
    /// `argv[0]` is the absolute program path.
    pub argv: Vec<String>,
    /// The child's whole environment besides `PATH=/usr/bin:/bin`.
    pub env: BTreeMap<String, String>,
    /// Each of the two waits of a stop (for the leader after TERM, for the group after KILL).
    pub stop_timeout: Duration,
    /// Both or neither: the child drops to them before exec.
    pub uid: Option<u32>,
    pub gid: Option<u32>,
}

/// `backend_child.json`: the group leader's pid, its pgid (= pid) and its starttime in clock ticks after boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChildRecord {
    pub pid: i32,
    pub pgid: i32,
    pub start_time_ticks: u64,
}

/// The leader exited on its own (not during a stop): its exit code, or the signal that ended it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChildExit {
    pub record: ChildRecord,
    pub code: Option<i32>,
    pub signal: Option<i32>,
}

#[derive(Debug)]
pub enum SpawnError {
    /// A group is still recorded (running, exited, or not confirmed gone): stop it first.
    Recorded(ChildRecord),
    /// The program did not start: `argv[0]` missing or not executable, an invalid spec, or a pre-exec step failed.
    Spawn(io::Error),
    /// The group started but its record could not be read or written; it was stopped again (an unconfirmed stop
    /// keeps it recorded for the next [`Supervisor::stop`]).
    Record(io::Error),
    /// The supervisor thread is gone.
    Closed,
}

impl fmt::Display for SpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recorded(r) => write!(f, "backend group {} is still recorded", r.pgid),
            Self::Spawn(e) => write!(f, "cannot start the backend: {e}"),
            Self::Record(e) => write!(f, "cannot record the backend group: {e}"),
            Self::Closed => f.write_str("backend supervisor stopped"),
        }
    }
}

impl std::error::Error for SpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(e) | Self::Record(e) => Some(e),
            Self::Recorded(_) | Self::Closed => None,
        }
    }
}

#[derive(Debug)]
pub enum StopError {
    /// The group (or its unreaped leader) outlived `stop_timeout` after SIGKILL. It stays recorded, on disk and in
    /// the supervisor; a later stop retries.
    Survivors(ChildRecord),
    /// The group is gone but `backend_child.json` could not be removed.
    Io(io::Error),
    /// The supervisor thread is gone.
    Closed,
}

impl fmt::Display for StopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Survivors(r) => write!(f, "backend group {} not gone after SIGKILL", r.pgid),
            Self::Io(e) => write!(f, "backend group gone, but {CHILD_FILE} not removed: {e}"),
            Self::Closed => f.write_str("backend supervisor stopped"),
        }
    }
}

impl std::error::Error for StopError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Survivors(_) | Self::Closed => None,
        }
    }
}

/// What [`reconcile_at_start`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reconciled {
    /// No `backend_child.json`.
    NoRecord,
    /// The record named a group; `processes` of it were SIGKILLed (0: nothing of it was left) and the file removed.
    Killed { processes: usize },
}

#[derive(Debug)]
pub enum ReconcileError {
    /// `backend_child.json` cannot be read, parsed or removed, or `/proc` cannot be listed.
    Io(io::Error),
    /// Processes of the recorded group were still there `stop_timeout` after SIGKILL; the file stays.
    Survivors { record: ChildRecord, pids: Vec<i32> },
}

impl fmt::Display for ReconcileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{CHILD_FILE}: {e}"),
            Self::Survivors { record, pids } => write!(
                f,
                "recorded backend group {} survived SIGKILL: pids {pids:?}",
                record.pgid
            ),
        }
    }
}

impl std::error::Error for ReconcileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Survivors { .. } => None,
        }
    }
}

enum Request {
    Spawn(oneshot::Sender<Result<ChildRecord, SpawnError>>),
    Stop(oneshot::Sender<Result<(), StopError>>),
    Shutdown(oneshot::Sender<()>),
}

/// The handle of the supervisor thread; one per managed agent. Dropping it stops the group like
/// [`Supervisor::shutdown`], without waiting.
pub struct Supervisor {
    requests: std_mpsc::Sender<Request>,
}

impl Supervisor {
    /// Starts the dedicated spawn/reap thread (a `std::thread`, never tokio); it lives until `shutdown`. Refused
    /// while `backend_child.json` exists: [`reconcile_at_start`] removes it first, so no record is overwritten.
    pub fn start(
        state_dir: PathBuf,
        spec: LaunchSpec,
    ) -> io::Result<(Supervisor, mpsc::UnboundedReceiver<ChildExit>)> {
        let file = state_dir.join(CHILD_FILE);
        if file.try_exists()? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("{} exists: reconcile it first", file.display()),
            ));
        }
        let (exits, exit_rx) = mpsc::unbounded_channel();
        let (requests, inbox) = std_mpsc::channel();
        let worker = Worker {
            state_dir,
            spec,
            group: None,
            exits,
        };
        std::thread::Builder::new()
            .name("cere-backend-supervisor".to_owned())
            .spawn(move || worker.run(&inbox))?;
        Ok((Supervisor { requests }, exit_rx))
    }

    /// Spawns the group from the thread; returns only after `backend_child.json` is written and fsynced.
    pub async fn spawn(&self) -> Result<ChildRecord, SpawnError> {
        let (reply, answer) = oneshot::channel();
        self.requests
            .send(Request::Spawn(reply))
            .map_err(|_| SpawnError::Closed)?;
        answer.await.map_err(|_| SpawnError::Closed)?
    }

    /// TERM group, wait <= stop_timeout, KILL group, reap leader, poll kill(-pgid,0) until ESRCH (<= stop_timeout),
    /// then delete the file. Ok only when the group is gone. Idempotent when nothing runs.
    pub async fn stop(&self) -> Result<(), StopError> {
        let (reply, answer) = oneshot::channel();
        self.requests
            .send(Request::Stop(reply))
            .map_err(|_| StopError::Closed)?;
        answer.await.map_err(|_| StopError::Closed)?
    }

    /// pid alive with the recorded starttime (`/proc/<pid>/stat` field 22), not a zombie and not exiting.
    pub fn alive(record: &ChildRecord) -> bool {
        proc_stat(record.pid).is_ok_and(|stat| stat.running(record))
    }

    /// Stops any running group, then ends the thread.
    pub async fn shutdown(self) {
        let (reply, answer) = oneshot::channel();
        if self.requests.send(Request::Shutdown(reply)).is_ok() {
            let _ = answer.await;
        }
    }
}

/// The recorded group as the thread holds it.
enum Group {
    /// The leader is the thread's unreaped child (running, or exited and kept as a zombie): its pid holds the
    /// pgid, so signals to the group reach no other group. `report_exit` until its exit has been seen.
    Held {
        leader: Child,
        record: ChildRecord,
        report_exit: bool,
    },
    /// A stop reaped the leader, then timed out waiting for the rest of the group: only `kill(-pgid, 0)` is sent.
    Reaped(ChildRecord),
}

impl Group {
    fn record(&self) -> ChildRecord {
        match self {
            Self::Held { record, .. } | Self::Reaped(record) => *record,
        }
    }
}

struct Worker {
    state_dir: PathBuf,
    spec: LaunchSpec,
    group: Option<Group>,
    exits: mpsc::UnboundedSender<ChildExit>,
}

impl Worker {
    fn run(mut self, inbox: &std_mpsc::Receiver<Request>) {
        loop {
            match inbox.recv_timeout(EXIT_POLL) {
                Ok(Request::Spawn(reply)) => {
                    let _ = reply.send(self.spawn());
                }
                Ok(Request::Stop(reply)) => {
                    let _ = reply.send(self.stop());
                }
                Ok(Request::Shutdown(reply)) => {
                    self.stop_at_exit();
                    let _ = reply.send(());
                    return;
                }
                Err(std_mpsc::RecvTimeoutError::Timeout) => {}
                Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                    self.stop_at_exit();
                    return;
                }
            }
            self.poll_exit();
        }
    }

    /// Reports the leader's exit once, leaving it unreaped (it keeps holding the pgid until `stop`).
    fn poll_exit(&mut self) {
        if let Some(Group::Held {
            record,
            report_exit: report_exit @ true,
            ..
        }) = &mut self.group
        {
            if let Some(exit) = leader_exit(record) {
                *report_exit = false;
                let _ = self.exits.send(exit);
            }
        }
    }

    fn spawn(&mut self) -> Result<ChildRecord, SpawnError> {
        if let Some(group) = &self.group {
            return Err(SpawnError::Recorded(group.record()));
        }
        let leader = launch(&self.spec).map_err(SpawnError::Spawn)?;
        let pid = leader.id() as i32;
        let mut record = ChildRecord {
            pid,
            pgid: pid,
            start_time_ticks: 0,
        };
        // The leader is unreaped, so its stat stays readable even if it has already exited.
        let recorded = proc_stat(pid).and_then(|stat| {
            record.start_time_ticks = stat.start_time_ticks;
            write_record(&self.state_dir, &record)
        });
        self.group = Some(Group::Held {
            leader,
            record,
            report_exit: true,
        });
        match recorded {
            Ok(()) => Ok(record),
            Err(e) => {
                // Never leave a group running without its record on disk.
                let _ = self.stop();
                Err(SpawnError::Record(e))
            }
        }
    }

    fn stop(&mut self) -> Result<(), StopError> {
        let timeout = self.spec.stop_timeout;
        let (record, deadline) = match self.group.take() {
            None => return Ok(()),
            Some(Group::Held {
                mut leader, record, ..
            }) => {
                // The unreaped leader holds the pgid while both group signals go out.
                signal_group(record.pgid, libc::SIGTERM);
                let deadline = Instant::now() + timeout;
                while leader_exit(&record).is_none() && Instant::now() < deadline {
                    std::thread::sleep(STEP);
                }
                signal_group(record.pgid, libc::SIGKILL);
                let deadline = Instant::now() + timeout;
                loop {
                    match leader.try_wait() {
                        Ok(Some(_)) => break,
                        Ok(None) if Instant::now() < deadline => std::thread::sleep(STEP),
                        _ => {
                            self.group = Some(Group::Held {
                                leader,
                                record,
                                report_exit: false,
                            });
                            return Err(StopError::Survivors(record));
                        }
                    }
                }
                (record, deadline)
            }
            Some(Group::Reaped(record)) => (record, Instant::now() + timeout),
        };
        if !wait_group_gone(record.pgid, deadline) {
            self.group = Some(Group::Reaped(record));
            return Err(StopError::Survivors(record));
        }
        remove_record(&self.state_dir).map_err(StopError::Io)
    }

    /// Shutdown, or the [`Supervisor`] was dropped. An unconfirmed stop leaves the file to the next start.
    fn stop_at_exit(&mut self) {
        if let Err(e) = self.stop() {
            eprintln!("cere-worker: backend stop at shutdown not confirmed: {e}");
        }
    }
}

fn invalid_input(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, what.to_owned())
}

/// Forks and execs the leader from the calling (supervisor) thread.
fn launch(spec: &LaunchSpec) -> io::Result<Child> {
    let Some((program, args)) = spec.argv.split_first() else {
        return Err(invalid_input("empty argv"));
    };
    let ids: Option<(libc::uid_t, libc::gid_t)> = match (spec.uid, spec.gid) {
        (Some(uid), Some(gid)) => Some((uid, gid)),
        (None, None) => None,
        _ => return Err(invalid_input("uid and gid are set together")),
    };
    // SAFETY: getpid has no preconditions.
    let parent = unsafe { libc::getpid() };
    let mut command = Command::new(program);
    command
        .args(args)
        .env_clear()
        .envs(&spec.env)
        .env("PATH", CHILD_PATH)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    // SAFETY: the closure runs in the child between fork and exec and calls only async-signal-safe functions
    // (setpgid, setgroups, setgid, setuid, prctl, getppid, _exit); it allocates nothing.
    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            if let Some((uid, gid)) = ids {
                if libc::setgroups(0, std::ptr::null()) != 0
                    || libc::setgid(gid) != 0
                    || libc::setuid(uid) != 0
                {
                    return Err(io::Error::last_os_error());
                }
            }
            // After the drop: a uid/gid change clears the parent-death signal (prctl(2)).
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong) != 0 {
                return Err(io::Error::last_os_error());
            }
            // The agent died before the prctl: no parent-death signal will come.
            if libc::getppid() != parent {
                libc::_exit(1);
            }
            Ok(())
        });
    }
    command.spawn()
}

/// The leader's exit, if it has exited, observed with `WNOWAIT`: it stays an unreaped zombie.
fn leader_exit(record: &ChildRecord) -> Option<ChildExit> {
    // SAFETY: an all-zero siginfo_t is a valid value.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid, writable siginfo_t for the duration of the call.
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            record.pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    // SAFETY: after a successful waitid the SIGCHLD fields are set (si_pid stays 0 while the child runs).
    let (pid, status) = unsafe { (info.si_pid(), info.si_status()) };
    if rc != 0 || pid == 0 {
        return None;
    }
    let exited = info.si_code == libc::CLD_EXITED;
    Some(ChildExit {
        record: *record,
        code: exited.then_some(status),
        signal: (!exited).then_some(status),
    })
}

fn signal_group(pgid: i32, signal: libc::c_int) {
    // SAFETY: kill(2) has no memory preconditions; the caller holds the pgid through the unreaped leader.
    unsafe {
        libc::kill(-pgid, signal);
    }
}

/// Polls `kill(-pgid, 0)` until `ESRCH` (no process of the group is left, zombies included) or `deadline`.
fn wait_group_gone(pgid: i32, deadline: Instant) -> bool {
    loop {
        // SAFETY: signal 0 delivers nothing; it only checks whether a member exists.
        let rc = unsafe { libc::kill(-pgid, 0) };
        if rc == -1 && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(STEP);
    }
}

/// Agent start, before the listener: kill the recorded group (pgrp == pgid && starttime >= recorded), wait for ESRCH,
/// delete the file. Err(Survivors) after `stop_timeout` -> the agent exits 1.
///
/// A process now under the recorded leader pid with another starttime means the group is gone (a pgid stays
/// allocated while any member lives, so the leader pid could only be reused after the last member died): nothing
/// is signalled and the file is removed. Processes that started before the recorded leader are never members.
pub fn reconcile_at_start(
    state_dir: &Path,
    stop_timeout: Duration,
) -> Result<Reconciled, ReconcileError> {
    let Some(record) = read_record(state_dir).map_err(ReconcileError::Io)? else {
        return Ok(Reconciled::NoRecord);
    };
    let deadline = Instant::now() + stop_timeout;
    let mut killed = HashSet::new();
    loop {
        let members = group_members(&record).map_err(ReconcileError::Io)?;
        if members.is_empty() {
            break;
        }
        for &(pid, start_time_ticks) in &members {
            if killed.insert((pid, start_time_ticks)) {
                // SAFETY: kill(2) has no memory preconditions; `pid` was just matched by pgrp and starttime.
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }
        if Instant::now() >= deadline {
            return Err(ReconcileError::Survivors {
                record,
                pids: members.into_iter().map(|(pid, _)| pid).collect(),
            });
        }
        std::thread::sleep(STEP);
    }
    remove_record(state_dir).map_err(ReconcileError::Io)?;
    Ok(Reconciled::Killed {
        processes: killed.len(),
    })
}

/// `(pid, starttime)` of every process in the recorded group except this one; empty once the recorded leader pid
/// belongs to another process.
fn group_members(record: &ChildRecord) -> io::Result<Vec<(i32, u64)>> {
    if proc_stat(record.pid).is_ok_and(|stat| stat.start_time_ticks != record.start_time_ticks) {
        return Ok(Vec::new());
    }
    let me = std::process::id() as i32;
    let mut members = Vec::new();
    for entry in std::fs::read_dir("/proc")? {
        let Some(pid) = entry?
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<i32>().ok())
        else {
            continue;
        };
        // A process that ends between the listing and the read is no member any more.
        let Ok(stat) = proc_stat(pid) else {
            continue;
        };
        if pid != me && stat.pgrp == record.pgid && stat.start_time_ticks >= record.start_time_ticks
        {
            members.push((pid, stat.start_time_ticks));
        }
    }
    Ok(members)
}

fn read_record(state_dir: &Path) -> io::Result<Option<ChildRecord>> {
    let path = state_dir.join(CHILD_FILE);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let invalid = |why: String| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {why}", path.display()),
        )
    };
    let record: ChildRecord = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
    // Only a group leader this supervisor created (pgid = pid, never init) is ever matched.
    if record.pid <= 1 || record.pgid != record.pid {
        return Err(invalid(format!(
            "not a supervised group leader: {record:?}"
        )));
    }
    Ok(Some(record))
}

fn write_record(state_dir: &Path, record: &ChildRecord) -> io::Result<()> {
    let tmp = state_dir.join(CHILD_TMP_FILE);
    let mut bytes = serde_json::to_vec(record).map_err(io::Error::other)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, state_dir.join(CHILD_FILE))?;
    sync_dir(state_dir)
}

fn remove_record(state_dir: &Path) -> io::Result<()> {
    match std::fs::remove_file(state_dir.join(CHILD_FILE)) {
        Ok(()) => sync_dir(state_dir),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// `PF_EXITING` (`include/linux/sched.h`): set by `do_exit` (`exit_signals`) before `exit_files` closes the task's
/// descriptors, so a leader whose sockets were closed by its exit shows it until it becomes a zombie.
const PF_EXITING: u32 = 0x0000_0004;

struct ProcStat {
    state: u8,
    pgrp: i32,
    flags: u32,
    start_time_ticks: u64,
}

impl ProcStat {
    /// `record`'s process (same starttime), neither a zombie nor exiting: a peer that saw the leader's sockets
    /// close because it is dying never reads it as running (O25: a Drain must then end `backend_exited`).
    fn running(&self, record: &ChildRecord) -> bool {
        self.start_time_ticks == record.start_time_ticks
            && !matches!(self.state, b'Z' | b'X')
            && self.flags & PF_EXITING == 0
    }
}

fn proc_stat(pid: i32) -> io::Result<ProcStat> {
    let path = format!("/proc/{pid}/stat");
    let raw = std::fs::read(&path)?;
    parse_stat(&raw).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: unexpected format"),
        )
    })
}

/// Fields 3 (state), 5 (pgrp), 9 (flags) and 22 (starttime) of a `/proc/<pid>/stat` line, counted after the last
/// `)`: field 2, the command name, may hold spaces and parens.
fn parse_stat(raw: &[u8]) -> Option<ProcStat> {
    let close = raw.iter().rposition(|&b| b == b')')?;
    let mut fields = std::str::from_utf8(&raw[close + 1..])
        .ok()?
        .split_ascii_whitespace();
    let state = *fields.next()?.as_bytes().first()?;
    let pgrp = fields.nth(1)?.parse().ok()?;
    let flags = fields.nth(3)?.parse().ok()?;
    let start_time_ticks = fields.nth(12)?.parse().ok()?;
    Some(ProcStat {
        state,
        pgrp,
        flags,
        start_time_ticks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_fields_are_counted_after_the_last_paren() {
        let line =
            b"4242 (a) b (c) S 1 4240 4240 0 -1 4194304 1 2 3 4 5 6 7 8 20 0 1 0 987654 1 2\n";
        let stat = parse_stat(line).unwrap();
        assert_eq!(
            (stat.state, stat.pgrp, stat.flags, stat.start_time_ticks),
            (b'S', 4240, 4194304, 987654)
        );
        assert!(parse_stat(b"4242 (cut").is_none());
        assert!(parse_stat(b"4242 (x) S 1 4240").is_none());
    }

    /// O25: a SIGKILLed leader closes its sockets in `do_exit` before it becomes a zombie; in between its stat shows
    /// state `R`/`S`/`D` with `PF_EXITING` in the flags, and it is not running.
    #[test]
    fn an_exiting_leader_is_not_running() {
        let record = ChildRecord {
            pid: 4242,
            pgid: 4242,
            start_time_ticks: 987654,
        };
        let stat = |state: &str, flags: u32| {
            let line = format!(
                "4242 (mock) {state} 1 4242 4242 0 -1 {flags} 1 2 3 4 5 6 7 8 20 0 1 0 987654 1 2\n"
            );
            parse_stat(line.as_bytes()).unwrap()
        };
        assert!(stat("S", 4194304).running(&record));
        for state in ["R", "S", "D"] {
            assert!(
                !stat(state, 4194304 | PF_EXITING).running(&record),
                "{state}"
            );
        }
        assert!(!stat("Z", 4194304).running(&record));
        let other = ChildRecord {
            start_time_ticks: 1,
            ..record
        };
        assert!(!stat("S", 4194304).running(&other));
    }
}
