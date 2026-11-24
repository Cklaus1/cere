//! The built `cere-mock-sglang` binary for tests whose `cere-worker` launches its backend (m36 design note Q3,
//! Q14). The agent drops a root child to uid/gid 65534 (Q3), which cannot reach a binary under a 0750 home, so a
//! [`MockBin`] places the binary in its own 0755 directory under the temp dir; the placed path also tells this
//! test's launched backends apart from every other process (`/proc/<pid>/exe` names the path that was executed).
//!
//! `make ci` builds every workspace binary before the tests; a bare `cargo test -p cere-runtime` may not, so run
//! `cargo build -p cere-mock-sglang` first.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use super::StateDir;

/// The uid and gid the launched backend runs as: 65534 (`nobody`) when this process is root (required by Q3),
/// none otherwise (a non-root `setgroups` fails).
pub fn launch_ids() -> Option<u32> {
    // SAFETY: geteuid has no preconditions.
    (unsafe { libc::geteuid() } == 0).then_some(65534)
}

/// Puts the built mock at `dst`, executable by any uid: a hard link (no disk: a CI runner has no room for a
/// 41 MB debug copy per fixture), or a 0755 copy when the temp dir is another filesystem (a tmpfs `/tmp`), links
/// are refused, or the built file is not world-executable. A child process makes the copy (O30): a sibling test's
/// fork never holds it open for writing, which would make executing it fail ETXTBSY.
pub fn place(dst: &Path) {
    let src = built();
    if std::fs::hard_link(&src, dst).is_ok() {
        if std::fs::metadata(dst).unwrap().permissions().mode() & 0o005 == 0o005 {
            return;
        }
        std::fs::remove_file(dst).unwrap();
    }
    super::exec_file::copy_file(&src, dst, 0o755);
}

/// `<target>/debug/cere-mock-sglang`: this test binary is `<target>/debug/deps/<test>`.
pub fn built() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let path = exe
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .join("cere-mock-sglang");
    assert!(
        path.is_file(),
        "{} is missing: run `cargo build -p cere-mock-sglang` first",
        path.display()
    );
    path
}

/// A 0755 directory holding the placed mock ([`place`]); files written into it are readable by the backend's
/// uid. On drop it SIGKILLs every process still running the placed binary, then removes the directory.
pub struct MockBin {
    dir: StateDir,
    /// The placed binary: launch argv points here.
    pub binary: PathBuf,
}

impl MockBin {
    pub fn new() -> Self {
        let dir = StateDir::new();
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        let binary = dir.path().join("cere-mock-sglang");
        place(&binary);
        Self { dir, binary }
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Writes `text` to `name` in the directory with `mode`, atomically (a reader sees the old or the new text). A
    /// child process writes it (O30: a sibling test's fork never holds an executable one open for writing).
    pub fn write(&self, name: &str, text: &str, mode: u32) -> PathBuf {
        let path = self.path().join(name);
        let tmp = self.path().join(format!(".{name}.tmp"));
        super::exec_file::write_file(&tmp, text.as_bytes(), mode);
        std::fs::rename(&tmp, &path).unwrap();
        path
    }

    /// Pids of the live processes executing the copy ([`running`]).
    pub fn running(&self) -> Vec<i32> {
        running(&self.binary)
    }
}

/// Pids of the live processes executing `binary`, by `/proc/<pid>/exe` (a zombie has no `exe` link and is not
/// counted). The path of a placed copy names one test's launches only; an argv match does not (O48: a process in
/// another network namespace, such as a ci-docker container's mock, can name the same `--port`).
pub fn running(binary: &Path) -> Vec<i32> {
    let mut pids = Vec::new();
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        if std::fs::read_link(entry.path().join("exe")).is_ok_and(|exe| exe == binary) {
            pids.push(pid);
        }
    }
    pids
}

impl Default for MockBin {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MockBin {
    fn drop(&mut self) {
        for pid in self.running() {
            // SAFETY: plain kill(2) of a process running this test's own copy of the mock.
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
}

/// Pids of the live (not zombie) processes in process group `pgid` (`/proc/<pid>/stat` field 5).
pub fn group(pgid: i32) -> Vec<i32> {
    let mut pids = Vec::new();
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // Fields after the command name, which may contain spaces: state, ppid, pgrp, ...
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        let fields: Vec<&str> = rest.split_whitespace().collect();
        if fields
            .first()
            .is_some_and(|state| *state != "Z" && *state != "X")
            && fields.get(2).and_then(|g| g.parse::<i32>().ok()) == Some(pgid)
        {
            pids.push(pid);
        }
    }
    pids
}
