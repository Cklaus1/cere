//! SUP track (m36 design note Q3, row P3): the backend supervisor, driven with `/bin/sh` scripts written to a
//! temp dir. A test signals only processes it started: its supervised groups, processes it spawns itself and the
//! re-executed agent.
//!
//! Two tests re-execute this test binary as the agent (`std::env::current_exe()`, test `agent_child`, `--exact`, as
//! `agent_journal.rs` does): the child starts a supervisor on `CERE_SUP_AGENT_DIR`, spawns the script named by
//! `CERE_SUP_AGENT_SCRIPT` (as `CERE_SUP_AGENT_UID` when set) and blocks until the parent SIGKILLs it. `agent_child`
//! is inert unless `CERE_SUP_AGENT_DIR` is set.

#[allow(dead_code)]
#[path = "support/exec_file.rs"]
mod exec_file;

use cere_runtime::agent::supervisor::{
    reconcile_at_start, ChildRecord, LaunchSpec, ReconcileError, Reconciled, SpawnError, StopError,
    Supervisor, CHILD_FILE,
};
use std::collections::BTreeSet;
use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const AGENT_DIR_ENV: &str = "CERE_SUP_AGENT_DIR";
const AGENT_SCRIPT_ENV: &str = "CERE_SUP_AGENT_SCRIPT";
const AGENT_UID_ENV: &str = "CERE_SUP_AGENT_UID";
/// uid and gid `nobody`, the unprivileged identity of the live deploy (m36 design note Q16).
const NOBODY: u32 = 65534;
/// Every script's long sleep; an unusual duration, so a leftover is easy to find (`pgrep -f 'sleep 597'`).
const SLEEP: &str = "sleep 597";
const STOP: Duration = Duration::from_millis(500);

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cere-sup-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("state")).unwrap();
        // Traversable by `nobody`, which runs the privilege-dropped scripts from here.
        std::fs::set_permissions(&dir, Permissions::from_mode(0o755)).unwrap();
        Self(dir)
    }

    fn state(&self) -> PathBuf {
        self.0.join("state")
    }

    fn child_file(&self) -> PathBuf {
        self.state().join(CHILD_FILE)
    }

    /// An executable `/bin/sh` script `name` with `body`. Its output goes to /dev/null: a process leaked by a failed
    /// test must not hold the test runner's stdout/stderr pipes open. Written by a child process: a sibling test's
    /// fork never holds it open for writing, which would make executing it fail ETXTBSY (O30).
    fn script(&self, name: &str, body: &str) -> PathBuf {
        let path = self.0.join(name);
        let text = format!("#!/bin/sh\nexec >/dev/null 2>&1\n{body}\n");
        exec_file::write_file(&path, text.as_bytes(), 0o755);
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A process this test started directly (a re-executed agent or a foreign process); SIGKILLed and reaped on drop.
struct Owned(Child);

impl Owned {
    fn pid(&self) -> i32 {
        self.0.id() as i32
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spec(script: &Path, env: &[(&str, &str)]) -> LaunchSpec {
    LaunchSpec {
        argv: vec![script.to_str().unwrap().to_owned()],
        env: env
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
        stop_timeout: STOP,
        uid: None,
        gid: None,
    }
}

/// `/proc/<pid>/stat` state, pgrp and starttime (fields 3, 5, 22), split after the last `)`: the command name may
/// hold spaces and parens.
fn stat(pid: i32) -> Option<(char, i32, u64)> {
    let raw = std::fs::read(format!("/proc/{pid}/stat")).ok()?;
    let close = raw.iter().rposition(|&b| b == b')')?;
    let fields: Vec<&str> = std::str::from_utf8(&raw[close + 1..])
        .ok()?
        .split_ascii_whitespace()
        .collect();
    Some((
        fields.first()?.chars().next()?,
        fields.get(2)?.parse().ok()?,
        fields.get(19)?.parse().ok()?,
    ))
}

/// The live process `pid` as a record (its own pgrp and starttime).
fn record_of(pid: i32) -> ChildRecord {
    let (_, pgid, start_time_ticks) = stat(pid).unwrap();
    ChildRecord {
        pid,
        pgid,
        start_time_ticks,
    }
}

/// No process (not even a zombie) is left under the record's pid and starttime.
fn gone(record: &ChildRecord) -> bool {
    stat(record.pid).is_none_or(|(_, _, start)| start != record.start_time_ticks)
}

fn group_gone(pgid: i32) -> bool {
    // SAFETY: signal 0 delivers nothing; it only checks whether any member of the group exists.
    let rc = unsafe { libc::kill(-pgid, 0) };
    rc == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

fn wait_until(what: &str, timeout: Duration, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The pid a script wrote to `path` (one line).
fn wait_pid_file(path: &Path) -> i32 {
    let mut pid = None;
    wait_until(
        &format!("{}", path.display()),
        Duration::from_secs(10),
        || {
            pid = std::fs::read_to_string(path)
                .ok()
                .filter(|text| text.ends_with('\n'))
                .and_then(|text| text.trim().parse().ok());
            pid.is_some()
        },
    );
    pid.unwrap()
}

fn write_record(dir: &TempDir, record: &ChildRecord) {
    std::fs::write(dir.child_file(), serde_json::to_vec(record).unwrap()).unwrap();
}

fn euid() -> u32 {
    // SAFETY: geteuid has no preconditions.
    unsafe { libc::geteuid() }
}

/// Re-executes this binary as the agent (`agent_child`) on `dir`'s state dir.
fn run_agent(dir: &TempDir, script: &Path, uid: Option<u32>) -> Owned {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["agent_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(AGENT_DIR_ENV, dir.state())
        .env(AGENT_SCRIPT_ENV, script)
        .stdin(Stdio::null())
        .stdout(Stdio::null());
    if let Some(uid) = uid {
        cmd.env(AGENT_UID_ENV, uid.to_string());
    }
    Owned(cmd.spawn().unwrap())
}

/// The record the agent child wrote, once `backend_child.json` exists.
fn wait_record(dir: &TempDir, agent: &mut Owned) -> ChildRecord {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(bytes) = std::fs::read(dir.child_file()) {
            return serde_json::from_slice(&bytes).unwrap();
        }
        if let Some(status) = agent.0.try_wait().unwrap() {
            panic!("the agent exited before its spawn returned: {status}");
        }
        assert!(Instant::now() < deadline, "the agent wrote no {CHILD_FILE}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn kill_agent(agent: &mut Owned) {
    agent.0.kill().unwrap();
    agent.0.wait().unwrap();
}

/// Waits up to `timeout` for the record's process to be gone or a zombie; false if it is still alive. Does not
/// panic, so a subprocess test can reconcile (kill what it started) before it asserts.
fn dies_within(record: &ChildRecord, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Supervisor::alive(record) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    !Supervisor::alive(record)
}

/// The `Uid:`/`Gid:`/`Groups:` values of `/proc/<pid>/status`.
fn credentials(pid: i32, field: &str) -> Vec<String> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
    status
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .unwrap()
        .split_ascii_whitespace()
        .map(str::to_owned)
        .collect()
}

/// O30: a script written while sibling tests spawn processes executes. A sibling thread forks a child every 2 ms
/// that stays forked but not exec'd for 100 ms (`pre_exec` sleeps), as a parallel test's spawn is under load; a
/// child forked while the test process held the script open for writing would keep it open, and executing the
/// script would fail ETXTBSY until that child execs.
#[test]
fn a_script_written_while_siblings_fork_executes() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let dir = TempDir::new("txtbsy");
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut forked = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let mut command = Command::new("/bin/true");
                // SAFETY: the hook only sleeps, which is async-signal-safe in the forked child.
                unsafe {
                    command.pre_exec(|| {
                        std::thread::sleep(Duration::from_millis(100));
                        Ok(())
                    });
                }
                forked.push(command.spawn().unwrap());
                std::thread::sleep(Duration::from_millis(2));
            }
            for mut child in forked {
                child.wait().unwrap();
            }
        });
        let runs = std::panic::catch_unwind(|| {
            (0..20)
                .map(|i| {
                    let script = dir.script(&format!("run{i}.sh"), "exit 0");
                    Command::new(&script).status().map(|s| s.success())
                })
                .collect::<Vec<_>>()
        });
        stop.store(true, Ordering::Relaxed);
        let runs = runs.unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        assert!(runs.iter().all(|r| matches!(r, Ok(true))), "{runs:?}");
    });
}

#[tokio::test]
async fn the_child_leads_its_own_group_with_only_the_spec_env() {
    let dir = TempDir::new("env");
    let pid_file = dir.0.join("sleep.pid");
    // The command name holds spaces and parens: `/proc/<pid>/stat` must be split after the last `)`.
    let script = dir.script(
        "a (b) c.sh",
        &format!("{SLEEP} &\necho $! > \"$SUP_PID_FILE\"\nwait"),
    );
    let pid_path = pid_file.to_str().unwrap();
    let launch = spec(
        &script,
        &[
            ("SUP_PID_FILE", pid_path),
            ("FOO", "bar baz"),
            ("PATH", "/elsewhere"),
        ],
    );
    let (sup, _exits) = Supervisor::start(dir.state(), launch).unwrap();
    let record = sup.spawn().await.unwrap();
    assert_eq!(record.pgid, record.pid, "the leader's pgid is its pid");
    let (_, pgrp, start) = stat(record.pid).unwrap();
    assert_eq!((pgrp, start), (record.pid, record.start_time_ticks));
    assert!(Supervisor::alive(&record));

    let environ = std::fs::read(format!("/proc/{}/environ", record.pid)).unwrap();
    let env: BTreeSet<String> = environ
        .split(|&b| b == 0)
        .filter(|entry| !entry.is_empty())
        .map(|entry| String::from_utf8(entry.to_vec()).unwrap())
        .collect();
    let want: BTreeSet<String> = [
        format!("SUP_PID_FILE={pid_path}"),
        "FOO=bar baz".to_owned(),
        "PATH=/usr/bin:/bin".to_owned(),
    ]
    .into();
    assert_eq!(env, want, "only the spec env plus the fixed PATH");
    assert_eq!(
        std::fs::read_link(format!("/proc/{}/fd/0", record.pid)).unwrap(),
        Path::new("/dev/null"),
        "stdin is null"
    );
    let sleeper = record_of(wait_pid_file(&pid_file));
    assert_eq!(
        sleeper.pgid, record.pgid,
        "the leader's children share its group"
    );

    sup.stop().await.unwrap();
    assert!(gone(&record) && gone(&sleeper));
    assert!(group_gone(record.pgid));
}

#[tokio::test]
async fn the_child_file_exists_when_spawn_returns() {
    let dir = TempDir::new("file");
    let script = dir.script("file.sh", &format!("exec {SLEEP}"));
    let (sup, _exits) = Supervisor::start(dir.state(), spec(&script, &[])).unwrap();
    let record = sup.spawn().await.unwrap();
    let on_disk: ChildRecord =
        serde_json::from_slice(&std::fs::read(dir.child_file()).unwrap()).unwrap();
    assert_eq!(on_disk, record);
    assert_eq!(
        std::fs::metadata(dir.child_file())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert_eq!(stat(record.pid).unwrap().2, record.start_time_ticks);
    let names: Vec<String> = std::fs::read_dir(dir.state())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, [CHILD_FILE], "no temp file is left behind");

    sup.stop().await.unwrap();
    assert!(!dir.child_file().exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_child_outlives_an_idle_runtime() {
    let dir = TempDir::new("idle");
    let script = dir.script("idle.sh", &format!("exec {SLEEP}"));
    let (sup, mut exits) = Supervisor::start(dir.state(), spec(&script, &[])).unwrap();
    let record = sup.spawn().await.unwrap();
    // Longer than tokio's 10 s blocking-thread keep-alive: PR_SET_PDEATHSIG follows the forking thread (M36-07).
    tokio::time::sleep(Duration::from_secs(15)).await;
    assert!(
        Supervisor::alive(&record),
        "the child died with an idle runtime thread"
    );
    assert!(exits.try_recv().is_err(), "no exit was reported");

    sup.shutdown().await;
    assert!(gone(&record));
    assert!(group_gone(record.pgid));
    assert!(!dir.child_file().exists());
}

#[test]
fn sigkill_of_the_agent_kills_the_child() {
    agent_kill_case("kill", None);
    if euid() == 0 {
        agent_kill_case("kill-nobody", Some(NOBODY));
    } else {
        println!(
            "privilege-dropped variant skipped: it needs euid 0 (running as euid {})",
            euid()
        );
    }
}

fn agent_kill_case(tag: &str, uid: Option<u32>) {
    let dir = TempDir::new(tag);
    let script = dir.script(&format!("{tag}.sh"), &format!("exec {SLEEP}"));
    let mut agent = run_agent(&dir, &script, uid);
    let record = wait_record(&dir, &mut agent);
    assert!(Supervisor::alive(&record));
    if let Some(uid) = uid {
        assert_eq!(credentials(record.pid, "Uid:")[0], uid.to_string());
    }
    kill_agent(&mut agent);
    let died = dies_within(&record, Duration::from_secs(5));
    // The record outlives the agent for the next start's reconciliation, which also removes the group of a
    // failed case before the assertions below.
    assert!(dir.child_file().exists());
    let reconciled = reconcile_at_start(&dir.state(), STOP);
    assert!(died, "the child did not die with its agent");
    assert!(matches!(reconciled, Ok(Reconciled::Killed { .. })));
    assert!(group_gone(record.pgid));
}

/// The re-executed agent of the subprocess tests; inert unless `CERE_SUP_AGENT_DIR` is set.
#[test]
fn agent_child() {
    let Some(state) = std::env::var_os(AGENT_DIR_ENV).map(PathBuf::from) else {
        return;
    };
    let script = PathBuf::from(std::env::var_os(AGENT_SCRIPT_ENV).unwrap());
    let mut launch = spec(&script, &[]);
    if let Ok(uid) = std::env::var(AGENT_UID_ENV) {
        let id = uid.parse().unwrap();
        launch.uid = Some(id);
        launch.gid = Some(id);
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (sup, _exits) = Supervisor::start(state, launch).unwrap();
    runtime.block_on(sup.spawn()).unwrap();
    // The parent SIGKILLs this process; if it never does, the exit ends the child through PR_SET_PDEATHSIG.
    std::thread::sleep(Duration::from_secs(60));
    panic!("the parent did not kill the agent");
}

#[tokio::test]
async fn stop_terms_waits_then_kills_a_term_ignoring_leader() {
    let dir = TempDir::new("stop");
    let log = dir.0.join("term.log");
    // The trap runs once the foreground sleep (default TERM) has died; the loop then goes on until SIGKILL.
    let script = dir.script(
        "stop.sh",
        "trap 'echo term >> \"$SUP_LOG\"' TERM\necho ready > \"$SUP_LOG\"\nwhile :; do sleep 0.05; done",
    );
    let launch = spec(&script, &[("SUP_LOG", log.to_str().unwrap())]);
    let (sup, mut exits) = Supervisor::start(dir.state(), launch).unwrap();
    let record = sup.spawn().await.unwrap();
    wait_until("the trap", Duration::from_secs(5), || {
        std::fs::read_to_string(&log).is_ok_and(|text| text == "ready\n")
    });

    let started = Instant::now();
    sup.stop().await.unwrap();
    let took = started.elapsed();
    assert!(took >= STOP, "{took:?}: KILL only after stop_timeout");
    assert!(took < STOP * 2 + Duration::from_secs(2), "{took:?}");
    assert_eq!(
        std::fs::read_to_string(&log).unwrap(),
        "ready\nterm\n",
        "TERM reached the leader first"
    );
    assert!(gone(&record), "the leader is killed and reaped");
    assert!(group_gone(record.pgid));
    assert!(!dir.child_file().exists());
    assert!(
        exits.try_recv().is_err(),
        "an exit during stop is not reported"
    );
    sup.stop().await.unwrap();
}

#[tokio::test]
async fn a_grandchild_ignoring_term_delays_stop_until_it_is_gone() {
    let dir = TempDir::new("grandchild");
    let pid_file = dir.0.join("grandchild.pid");
    // The leader dies of TERM at once; the grandchild ignores it and is left to the group KILL.
    let script = dir.script(
        "grandchild.sh",
        &format!(
            "/bin/sh -c 'trap \"\" TERM; echo $$ > \"$1.tmp\"; mv \"$1.tmp\" \"$1\"; exec {SLEEP}' grandchild \"$SUP_PID_FILE\" &\nwait"
        ),
    );
    let launch = spec(&script, &[("SUP_PID_FILE", pid_file.to_str().unwrap())]);
    let (sup, _exits) = Supervisor::start(dir.state(), launch).unwrap();
    let record = sup.spawn().await.unwrap();
    let grandchild = record_of(wait_pid_file(&pid_file));
    assert_eq!(grandchild.pgid, record.pgid);

    sup.stop().await.unwrap();
    assert!(
        gone(&grandchild),
        "stop is confirmed only once the TERM-ignoring grandchild is gone"
    );
    assert!(gone(&record));
    assert!(group_gone(record.pgid));
    assert!(!dir.child_file().exists());
}

#[tokio::test]
async fn stop_is_not_confirmed_while_a_group_member_remains() {
    let dir = TempDir::new("member");
    let script = dir.script("member.sh", &format!("exec {SLEEP}"));
    let (sup, _exits) = Supervisor::start(dir.state(), spec(&script, &[])).unwrap();
    let record = sup.spawn().await.unwrap();
    // A process of this test joins the group and exits; unreaped, it stays a member no signal removes.
    let mut member = Owned(
        Command::new("/bin/sh")
            .args(["-c", "exit 0"])
            .process_group(record.pgid)
            .spawn()
            .unwrap(),
    );
    let member_pid = member.pid();
    wait_until("the member to exit", Duration::from_secs(5), || {
        stat(member_pid).is_some_and(|(state, pgrp, _)| state == 'Z' && pgrp == record.pgid)
    });

    let started = Instant::now();
    match sup.stop().await {
        Err(StopError::Survivors(survivor)) => assert_eq!(survivor, record),
        other => panic!("expected Survivors, got {other:?}"),
    }
    assert!(started.elapsed() >= STOP);
    assert!(gone(&record), "the leader itself was killed and reaped");
    assert!(
        dir.child_file().exists(),
        "the record stays while the group lives"
    );
    match sup.spawn().await {
        Err(SpawnError::Recorded(recorded)) => assert_eq!(recorded, record),
        other => panic!("expected Recorded, got {other:?}"),
    }

    member.0.wait().unwrap();
    sup.stop().await.unwrap();
    assert!(group_gone(record.pgid));
    assert!(!dir.child_file().exists());
}

#[test]
fn reconcile_kills_a_recorded_group_with_a_dead_leader_and_a_live_grandchild() {
    let dir = TempDir::new("reconcile");
    let pid_file = dir.0.join("grandchild.pid");
    let pid_path = pid_file.to_str().unwrap();
    let script = dir.script(
        "reconcile.sh",
        &format!(
            "/bin/sh -c 'echo $$ > \"$1.tmp\"; mv \"$1.tmp\" \"$1\"; exec {SLEEP}' grandchild '{pid_path}' &\nexec {SLEEP}"
        ),
    );
    let mut agent = run_agent(&dir, &script, None);
    let record = wait_record(&dir, &mut agent);
    let grandchild = record_of(wait_pid_file(&pid_file));
    assert_eq!(grandchild.pgid, record.pgid);
    kill_agent(&mut agent);
    let leader_died = dies_within(&record, Duration::from_secs(5));
    let grandchild_lived = Supervisor::alive(&grandchild);
    let reconciled = reconcile_at_start(&dir.state(), STOP);
    assert!(leader_died, "the leader did not die with its agent");
    assert!(
        grandchild_lived,
        "no parent-death signal reaches the grandchild"
    );
    match reconciled {
        // The leader counts when its zombie has not been reaped by init yet.
        Ok(Reconciled::Killed { processes }) => {
            assert!((1..=2).contains(&processes), "{processes}")
        }
        other => panic!("expected Killed, got {other:?}"),
    }
    assert!(gone(&grandchild));
    assert!(group_gone(record.pgid));
    assert!(!dir.child_file().exists());
}

#[test]
fn reconcile_ignores_a_recycled_pid() {
    let dir = TempDir::new("recycled");
    let foreign = Owned(
        Command::new("/bin/sh")
            .args(["-c", &format!("exec {SLEEP}")])
            .process_group(0)
            .spawn()
            .unwrap(),
    );
    let pid = foreign.pid();
    wait_until("the foreign group", Duration::from_secs(5), || {
        stat(pid).is_some_and(|(_, pgrp, _)| pgrp == pid)
    });
    let current = record_of(pid);
    // The recorded leader started before (pid reused later) or after (another boot) the process now under its pid.
    for start_time_ticks in [current.start_time_ticks - 1, current.start_time_ticks + 1] {
        write_record(
            &dir,
            &ChildRecord {
                start_time_ticks,
                ..current
            },
        );
        match reconcile_at_start(&dir.state(), STOP) {
            Ok(Reconciled::Killed { processes: 0 }) => {}
            other => panic!("expected Killed {{ processes: 0 }}, got {other:?}"),
        }
        assert!(
            Supervisor::alive(&current),
            "a process that only reuses the recorded pid is never signalled"
        );
        assert!(!dir.child_file().exists());
    }
}

#[test]
fn reconcile_without_a_record_is_no_record() {
    let dir = TempDir::new("norecord");
    assert!(matches!(
        reconcile_at_start(&dir.state(), STOP),
        Ok(Reconciled::NoRecord)
    ));
}

#[test]
fn reconcile_reports_survivors_and_keeps_the_record() {
    let dir = TempDir::new("survivors");
    // Unreaped by this test, the exited process stays in its group: SIGKILL cannot remove it.
    let mut zombie = Owned(
        Command::new("/bin/sh")
            .args(["-c", "exit 0"])
            .process_group(0)
            .spawn()
            .unwrap(),
    );
    let pid = zombie.pid();
    wait_until("the zombie", Duration::from_secs(5), || {
        stat(pid).is_some_and(|(state, _, _)| state == 'Z')
    });
    let record = record_of(pid);
    write_record(&dir, &record);

    let started = Instant::now();
    match reconcile_at_start(&dir.state(), STOP) {
        Err(ReconcileError::Survivors {
            record: survivor,
            pids,
        }) => {
            assert_eq!(survivor, record);
            assert_eq!(pids, [pid]);
        }
        other => panic!("expected Survivors, got {other:?}"),
    }
    assert!(started.elapsed() >= STOP);
    assert!(dir.child_file().exists(), "the next start reconciles again");

    zombie.0.wait().unwrap();
    assert!(matches!(
        reconcile_at_start(&dir.state(), STOP),
        Ok(Reconciled::Killed { processes: 0 })
    ));
    assert!(!dir.child_file().exists());
}

#[tokio::test]
async fn child_exit_is_observed_with_its_code_and_signal() {
    for (tag, body, code, signal) in [
        ("exit-code", "exit 3", Some(3), None),
        (
            "exit-signal",
            "kill -USR1 $$\nsleep 5",
            None,
            Some(libc::SIGUSR1),
        ),
    ] {
        let dir = TempDir::new(tag);
        let script = dir.script(&format!("{tag}.sh"), body);
        let (sup, mut exits) = Supervisor::start(dir.state(), spec(&script, &[])).unwrap();
        let record = sup.spawn().await.unwrap();
        let exit = tokio::time::timeout(Duration::from_secs(5), exits.recv())
            .await
            .expect("no exit reported")
            .unwrap();
        assert_eq!(
            (exit.record, exit.code, exit.signal),
            (record, code, signal)
        );
        assert!(!Supervisor::alive(&record));
        assert!(
            dir.child_file().exists(),
            "the group stays recorded until stop"
        );

        sup.stop().await.unwrap();
        assert!(gone(&record));
        assert!(group_gone(record.pgid));
        assert!(!dir.child_file().exists());
        assert!(exits.try_recv().is_err(), "one exit per leader");
    }
}

#[tokio::test]
async fn the_child_drops_to_the_configured_uid_and_gid_when_root() {
    if euid() != 0 {
        println!(
            "skipped: the privilege drop needs euid 0 (running as euid {})",
            euid()
        );
        return;
    }
    let dir = TempDir::new("nobody");
    let script = dir.script("nobody.sh", &format!("exec {SLEEP}"));
    let mut launch = spec(&script, &[]);
    launch.uid = Some(NOBODY);
    launch.gid = Some(NOBODY);
    let (sup, _exits) = Supervisor::start(dir.state(), launch).unwrap();
    let record = sup.spawn().await.unwrap();
    let nobody = NOBODY.to_string();
    assert_eq!(credentials(record.pid, "Uid:"), [nobody.as_str(); 4]);
    assert_eq!(credentials(record.pid, "Gid:"), [nobody.as_str(); 4]);
    assert!(
        credentials(record.pid, "Groups:").is_empty(),
        "no supplementary groups"
    );

    sup.stop().await.unwrap();
    assert!(gone(&record));
    assert!(!dir.child_file().exists());
}
