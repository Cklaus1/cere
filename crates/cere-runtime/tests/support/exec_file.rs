//! Files a test later executes, written by a child process (O30). A process forked while a file is open for writing
//! holds that descriptor until it execs, and executing the file meanwhile fails ETXTBSY ("Text file busy"). Test
//! threads spawn processes in parallel, so a file the test process itself wrote could be held open by a sibling
//! test's child that has forked but not yet exec'd; written here, the test process never holds a writable
//! descriptor of it (a sibling's child inherits at most the stdin pipe, which only delays the writer's EOF).
//!
//! Plain std, no crate imports: `agent_supervisor.rs` includes this file by path.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Writes `bytes` to `path` with permission bits `mode` (replacing any file there).
pub fn write_file(path: &Path, bytes: &[u8], mode: u32) {
    let mut child = Command::new("/bin/sh")
        .args(["-c", "cat > \"$1\" && chmod \"$2\" \"$1\"", "sh"])
        .arg(path)
        .arg(format!("{mode:o}"))
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let status = child.wait().unwrap();
    assert!(status.success(), "writing {}: {status}", path.display());
}

/// Copies `src` to `dst` with permission bits `mode`.
pub fn copy_file(src: &Path, dst: &Path, mode: u32) {
    let status = Command::new("/bin/sh")
        .args(["-c", "cp -- \"$1\" \"$2\" && chmod \"$3\" \"$2\"", "sh"])
        .arg(src)
        .arg(dst)
        .arg(format!("{mode:o}"))
        .status()
        .unwrap();
    assert!(
        status.success(),
        "copying {} to {}: {status}",
        src.display(),
        dst.display()
    );
}
