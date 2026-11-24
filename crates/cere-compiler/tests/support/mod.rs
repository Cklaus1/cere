//! Shared helpers for `cerec` integration tests: repo paths, temp directories, fixture copies, process runs.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

pub fn fixture(rel: &str) -> PathBuf {
    root().join("tests/fixtures").join(rel)
}

pub fn model_src() -> PathBuf {
    fixture("model_src")
}

/// A temp directory removed on drop (read-only modes are reset first). `Tmp::new` puts it under Cargo's
/// `CARGO_TARGET_TMPDIR`, which follows the target dir: usually a disk file system, but tmpfs when the target is on a
/// tmpfs `/tmp` (the gate's uid 65534 pass, O314), where page-cache eviction and `mincore` mean nothing (`fs_kind`).
pub struct Tmp(pub PathBuf);

static SEQ: AtomicUsize = AtomicUsize::new(0);

impl Tmp {
    pub fn new(tag: &str) -> Tmp {
        Tmp::under(Path::new(env!("CARGO_TARGET_TMPDIR")), tag)
    }

    /// A fresh directory in `base`.
    pub fn under(base: &Path, tag: &str) -> Tmp {
        let p = base.join(format!(
            "cerec-it-{tag}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = remove_tree(&p);
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = remove_tree(&self.0);
    }
}

/// ramfs's `statfs` `f_type` (`linux/magic.h`; libc 0.2.190 does not define it).
const RAMFS_MAGIC: libc::c_long = 0x858458f6;

/// A file system as `statfs` names it, and whether `cerec evict` (`fsync` + `POSIX_FADV_DONTNEED`) can drop its pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsKind {
    pub name: &'static str,
    pub magic: libc::c_long,
    pub evictable: bool,
}

impl std::fmt::Display for FsKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (f_type {:#x})", self.name, self.magic)
    }
}

impl FsKind {
    /// tmpfs and ramfs keep a file's only copy in the page cache, so nothing can evict it: those are the two that are
    /// not evictable. Every other file system, named or `unknown`, counts as evictable, so an eviction test there runs
    /// its assertion (and fails loudly if the file system keeps its pages) instead of being skipped.
    pub fn of_magic(magic: libc::c_long) -> FsKind {
        let (name, evictable) = match magic {
            libc::TMPFS_MAGIC => ("tmpfs", false),
            RAMFS_MAGIC => ("ramfs", false),
            libc::EXT4_SUPER_MAGIC => ("ext4", true),
            libc::XFS_SUPER_MAGIC => ("xfs", true),
            libc::BTRFS_SUPER_MAGIC => ("btrfs", true),
            libc::OVERLAYFS_SUPER_MAGIC => ("overlayfs", true),
            _ => ("unknown", true),
        };
        FsKind {
            name,
            magic,
            evictable,
        }
    }
}

/// The file system holding `dir` (`fstatfs` of the directory).
pub fn fs_kind(dir: &Path) -> FsKind {
    use std::os::fd::AsRawFd;
    let d = std::fs::File::open(dir).unwrap_or_else(|e| panic!("open {}: {e}", dir.display()));
    let mut buf = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `d` is an open descriptor and `buf` points to a writable `statfs`.
    let rc = unsafe { libc::fstatfs(d.as_raw_fd(), buf.as_mut_ptr()) };
    assert_eq!(
        rc,
        0,
        "fstatfs {}: {}",
        dir.display(),
        std::io::Error::last_os_error()
    );
    // SAFETY: `fstatfs` returned 0, so it filled `buf`.
    FsKind::of_magic(unsafe { buf.assume_init() }.f_type)
}

/// `rm -rf` that first makes 0555 directories writable.
pub fn remove_tree(p: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::symlink_metadata(p) {
        if meta.is_dir() {
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755));
            for e in std::fs::read_dir(p)? {
                remove_tree(&e?.path())?;
            }
            return std::fs::remove_dir(p);
        }
        return std::fs::remove_file(p);
    }
    Ok(())
}

/// Recursive copy of regular files and directories (mtimes are not preserved).
pub fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let ty = e.file_type().unwrap();
        let dest = to.join(e.file_name());
        if ty.is_dir() {
            copy_tree(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), &dest).unwrap();
        }
    }
}

pub fn cerec() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_cerec"));
    c.current_dir(root());
    c
}

/// A `CEREC_TEST_MEMINFO` file in `t` reporting `bytes` of `MemAvailable` (rounded down to KiB).
pub fn meminfo(t: &Tmp, bytes: u64) -> PathBuf {
    let p = t.join("meminfo");
    std::fs::write(&p, format!("MemAvailable: {} kB\n", bytes / 1024)).unwrap();
    p
}

pub fn python(script: &str) -> Command {
    let mut c = Command::new("python3");
    c.current_dir(root())
        .arg(root().join("scripts").join(script));
    c
}

pub fn run(c: &mut Command) -> Output {
    c.output().unwrap_or_else(|e| panic!("running {c:?}: {e}"))
}

/// The error class of a rejection: exit 2 and a first stderr line `<class>: <message>`.
pub fn class_of(out: &Output) -> String {
    let err = String::from_utf8_lossy(&out.stderr);
    let first = err.lines().next().unwrap_or("");
    first
        .split_once(':')
        .map_or(first, |(c, _)| c)
        .trim()
        .to_string()
}

pub fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).unwrap()
}

pub fn ok(out: Output) -> Output {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

pub fn json(out: &Output) -> serde_json::Value {
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stdout)))
}
