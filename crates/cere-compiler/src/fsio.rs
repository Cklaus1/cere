//! Descriptor-based file access (m21 design note ground rules, Q7-Q9).
//!
//! A directory is opened once (`O_RDONLY|O_DIRECTORY`); every file below it is reached with
//! `openat(..., O_NOFOLLOW|O_CLOEXEC)` one path component at a time, must be a regular file, and is read with
//! `pread` through the held descriptor. Also: page-cache eviction/residency (`posix_fadvise`, `mincore`),
//! `flock`, and the `MemAvailable` reader (`CEREC_TEST_MEMINFO=<file>` overrides `/proc/meminfo`).

use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Why a file could not be opened safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsErrorKind {
    /// A component is a symlink (`O_NOFOLLOW` → `ELOOP`/`ENOTDIR`).
    Symlink,
    /// Not a regular file (FIFO, device, socket, directory).
    NotRegular,
    /// Absolute, empty, `.`/`..` or empty components, or NUL.
    BadPath,
    Missing,
    Io,
}

#[derive(Debug)]
pub struct FsError {
    pub kind: FsErrorKind,
    pub path: String,
    pub source: Option<io::Error>,
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.path)?;
        if let Some(e) = &self.source {
            write!(f, ": {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for FsError {}

fn fs_err(kind: FsErrorKind, path: &str, source: Option<io::Error>) -> FsError {
    FsError {
        kind,
        path: path.to_string(),
        source,
    }
}

/// The `fstat` fields a stamp records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stat {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
    pub mode: u32,
}

impl Stat {
    pub fn of(file: &File) -> io::Result<Stat> {
        let m = file.metadata()?;
        Ok(Stat {
            dev: m.dev(),
            ino: m.ino(),
            size: m.size(),
            mtime_ns: i128::from(m.mtime()) * 1_000_000_000 + i128::from(m.mtime_nsec()),
            ctime_ns: i128::from(m.ctime()) * 1_000_000_000 + i128::from(m.ctime_nsec()),
            mode: m.mode(),
        })
    }
}

/// A regular file opened through a directory descriptor, with its `fstat` at open.
#[derive(Debug)]
pub struct OpenFile {
    pub file: File,
    pub stat: Stat,
}

impl OpenFile {
    pub fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        self.file.read_exact_at(buf, offset)
    }

    /// Re-`fstat`s the held descriptor.
    pub fn restat(&self) -> io::Result<Stat> {
        Stat::of(&self.file)
    }

    /// A second descriptor of the held file's inode opened `O_RDONLY|O_DIRECT` (through `/proc/self/fd`, so no path
    /// is resolved again), for page-cache-bypassing reads; `None` when the file system refuses `O_DIRECT`
    /// (`EINVAL`). A descriptor of another inode is an error.
    pub fn reopen_direct(&self) -> io::Result<Option<File>> {
        let path = CString::new(format!("/proc/self/fd/{}", self.file.as_raw_fd()))
            .expect("no NUL in a fd path");
        // SAFETY: a NUL-terminated path; the returned descriptor is owned below.
        let fd = unsafe {
            libc::open(
                path.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_DIRECT,
            )
        };
        if fd < 0 {
            let e = io::Error::last_os_error();
            return match e.raw_os_error() {
                Some(libc::EINVAL) => Ok(None),
                _ => Err(e),
            };
        }
        // SAFETY: `fd` is a fresh descriptor nothing else owns.
        let file = unsafe { File::from_raw_fd(fd) };
        let stat = Stat::of(&file)?;
        if (stat.dev, stat.ino) != (self.stat.dev, self.stat.ino) {
            return Err(io::Error::other("the O_DIRECT descriptor is another inode"));
        }
        Ok(Some(file))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntryKind {
    File,
    Dir,
    Symlink,
    Other,
}

/// An open directory descriptor.
#[derive(Debug)]
pub struct Dir {
    fd: OwnedFd,
}

fn cstr(s: &[u8]) -> io::Result<CString> {
    CString::new(s).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))
}

fn openat(
    dirfd: RawFd,
    name: &CStr,
    flags: libc::c_int,
    mode: libc::c_uint,
) -> io::Result<OwnedFd> {
    // SAFETY: `name` is NUL-terminated; the returned descriptor is owned exactly once.
    let fd = unsafe { libc::openat(dirfd, name.as_ptr(), flags | libc::O_CLOEXEC, mode) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` is a fresh descriptor we own.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Splits an artifact-relative path into safe components.
pub fn components(rel: &str) -> Option<Vec<&str>> {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\0') {
        return None;
    }
    let parts: Vec<&str> = rel.split('/').collect();
    if parts
        .iter()
        .any(|p| p.is_empty() || *p == "." || *p == "..")
    {
        return None;
    }
    Some(parts)
}

impl Dir {
    /// Opens `path` as a directory (the final component may not be a symlink).
    pub fn open(path: &Path) -> io::Result<Dir> {
        let c = cstr(path.as_os_str().as_bytes())?;
        let fd = openat(
            libc::AT_FDCWD,
            &c,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW,
            0,
        )?;
        Ok(Dir { fd })
    }

    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    fn classify(e: io::Error, path: &str) -> FsError {
        match e.raw_os_error() {
            Some(libc::ELOOP) | Some(libc::ENOTDIR) => fs_err(FsErrorKind::Symlink, path, Some(e)),
            Some(libc::ENOENT) => fs_err(FsErrorKind::Missing, path, Some(e)),
            _ => fs_err(FsErrorKind::Io, path, Some(e)),
        }
    }

    /// The parent of this directory (`openat(fd, "..")`): the directory that held it when it was opened, not a
    /// re-resolved path.
    pub fn parent(&self) -> io::Result<Dir> {
        let fd = openat(
            self.fd.as_raw_fd(),
            c"..",
            libc::O_RDONLY | libc::O_DIRECTORY,
            0,
        )?;
        Ok(Dir { fd })
    }

    /// Opens the subdirectory `rel` (relative, no symlink components).
    pub fn open_dir(&self, rel: &str) -> Result<Dir, FsError> {
        let parts = components(rel).ok_or_else(|| fs_err(FsErrorKind::BadPath, rel, None))?;
        let mut current: Option<OwnedFd> = None;
        for part in parts {
            let base = current
                .as_ref()
                .map_or(self.fd.as_raw_fd(), |f| f.as_raw_fd());
            let c = cstr(part.as_bytes()).map_err(|_| fs_err(FsErrorKind::BadPath, rel, None))?;
            let fd = openat(
                base,
                &c,
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW,
                0,
            )
            .map_err(|e| Self::classify(e, rel))?;
            current = Some(fd);
        }
        Ok(Dir {
            fd: current.expect("components() is non-empty"),
        })
    }

    /// Opens the regular file `rel` read-only. `O_NONBLOCK` keeps a FIFO from blocking the open.
    pub fn open_file(&self, rel: &str) -> Result<OpenFile, FsError> {
        let parts = components(rel).ok_or_else(|| fs_err(FsErrorKind::BadPath, rel, None))?;
        let (name, dirs) = parts.split_last().expect("components() is non-empty");
        let sub;
        let base = if dirs.is_empty() {
            self.fd.as_raw_fd()
        } else {
            sub = self.open_dir(&dirs.join("/"))?;
            sub.fd.as_raw_fd()
        };
        let c = cstr(name.as_bytes()).map_err(|_| fs_err(FsErrorKind::BadPath, rel, None))?;
        let fd = openat(
            base,
            &c,
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            0,
        )
        .map_err(|e| Self::classify(e, rel))?;
        let file = File::from(fd);
        let stat = Stat::of(&file).map_err(|e| fs_err(FsErrorKind::Io, rel, Some(e)))?;
        if stat.mode & libc::S_IFMT != libc::S_IFREG {
            return Err(fs_err(FsErrorKind::NotRegular, rel, None));
        }
        // Clear O_NONBLOCK again; it has no effect on regular files but keep the descriptor plain.
        // SAFETY: plain fcntl on a descriptor we own.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, 0) };
        Ok(OpenFile { file, stat })
    }

    /// Creates (exclusively) and opens `name` for writing in this directory, mode `0600`.
    pub fn create_file(&self, name: &str) -> io::Result<File> {
        let c = cstr(name.as_bytes())?;
        let fd = openat(
            self.fd.as_raw_fd(),
            &c,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW,
            0o600,
        )?;
        Ok(File::from(fd))
    }

    /// Directory entries (name, kind) without `.`/`..`, in readdir order.
    pub fn entries(&self) -> io::Result<Vec<(String, EntryKind)>> {
        // SAFETY: dup gives fdopendir its own descriptor, which closedir releases.
        let dup = unsafe { libc::fcntl(self.fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
        if dup < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `dup` is a valid directory descriptor; ownership moves to the DIR stream.
        let dirp = unsafe { libc::fdopendir(dup) };
        if dirp.is_null() {
            let e = io::Error::last_os_error();
            // SAFETY: fdopendir failed, so `dup` is still ours.
            unsafe { libc::close(dup) };
            return Err(e);
        }
        // SAFETY: the stream is fresh; rewind so a reused offset never hides entries.
        unsafe { libc::rewinddir(dirp) };
        let mut out = Vec::new();
        loop {
            // SAFETY: `dirp` is a valid stream; entries are copied before the next readdir.
            let ent = unsafe { libc::readdir(dirp) };
            if ent.is_null() {
                break;
            }
            // SAFETY: d_name is NUL-terminated inside the dirent readdir returned.
            let (name, d_type) = unsafe { (CStr::from_ptr((*ent).d_name.as_ptr()), (*ent).d_type) };
            let name = name.to_bytes();
            if name == b"." || name == b".." {
                continue;
            }
            let name = String::from_utf8_lossy(name).into_owned();
            let kind = match d_type {
                libc::DT_REG => EntryKind::File,
                libc::DT_DIR => EntryKind::Dir,
                libc::DT_LNK => EntryKind::Symlink,
                libc::DT_UNKNOWN => self.lstat_kind(&name)?,
                _ => EntryKind::Other,
            };
            out.push((name, kind));
        }
        // SAFETY: closes the stream and its descriptor exactly once.
        unsafe { libc::closedir(dirp) };
        Ok(out)
    }

    fn lstat_kind(&self, name: &str) -> io::Result<EntryKind> {
        let c = cstr(name.as_bytes())?;
        // SAFETY: zeroed `stat` is a valid out-parameter.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: valid dirfd, NUL-terminated name, valid out-pointer.
        if unsafe {
            libc::fstatat(
                self.fd.as_raw_fd(),
                c.as_ptr(),
                &mut st,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(match st.st_mode & libc::S_IFMT {
            libc::S_IFREG => EntryKind::File,
            libc::S_IFDIR => EntryKind::Dir,
            libc::S_IFLNK => EntryKind::Symlink,
            _ => EntryKind::Other,
        })
    }

    /// `fsync` of the directory itself.
    pub fn sync(&self) -> io::Result<()> {
        // SAFETY: plain fsync on a descriptor we own.
        if unsafe { libc::fsync(self.fd.as_raw_fd()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// `mkdirat(name, mode)` and opens the new directory.
    pub fn create_dir(&self, name: &str, mode: u32) -> io::Result<Dir> {
        let c = cstr(name.as_bytes())?;
        // SAFETY: valid dirfd and NUL-terminated name.
        if unsafe { libc::mkdirat(self.fd.as_raw_fd(), c.as_ptr(), mode as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error());
        }
        self.open_dir(name)
            .map_err(|e| e.source.unwrap_or_else(|| io::Error::other(e.path)))
    }

    /// Hardlinks `self/rel` to `dest/rel` (`linkat` without following a final symlink).
    pub fn link_into(&self, rel: &str, dest: &Dir) -> io::Result<()> {
        let c = cstr(rel.as_bytes())?;
        // SAFETY: valid dirfds and NUL-terminated names.
        if unsafe {
            libc::linkat(
                self.fd.as_raw_fd(),
                c.as_ptr(),
                dest.fd.as_raw_fd(),
                c.as_ptr(),
                0,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// `renameat2(from, to, RENAME_NOREPLACE)` within this directory; an existing `to` is `AlreadyExists`.
    pub fn rename_noreplace(&self, from: &str, to: &str) -> io::Result<()> {
        let (f, t) = (cstr(from.as_bytes())?, cstr(to.as_bytes())?);
        let fd = self.fd.as_raw_fd();
        // SAFETY: valid dirfd and NUL-terminated names.
        if unsafe { libc::renameat2(fd, f.as_ptr(), fd, t.as_ptr(), libc::RENAME_NOREPLACE) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// `renameat(from, to)` within this directory, replacing `to` (atomic file publish).
    pub fn rename_replace(&self, from: &str, to: &str) -> io::Result<()> {
        let (f, t) = (cstr(from.as_bytes())?, cstr(to.as_bytes())?);
        let fd = self.fd.as_raw_fd();
        // SAFETY: valid dirfd and NUL-terminated names.
        if unsafe { libc::renameat(fd, f.as_ptr(), fd, t.as_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// `fchmod` of the directory itself.
    pub fn chmod(&self, mode: u32) -> io::Result<()> {
        // SAFETY: plain fchmod on a descriptor we own.
        if unsafe { libc::fchmod(self.fd.as_raw_fd(), mode as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// `posix_fadvise` on `[offset, offset + len)` (`len` 0 = to the end).
pub fn fadvise(file: &File, offset: u64, len: u64, advice: libc::c_int) -> io::Result<()> {
    // SAFETY: plain syscall on an owned descriptor; returns an errno value rather than setting errno.
    let rc = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            advice,
        )
    };
    if rc != 0 {
        return Err(io::Error::from_raw_os_error(rc));
    }
    Ok(())
}

/// Source reads: sequential, no reuse expected.
pub fn advise_source(file: &File) -> io::Result<()> {
    fadvise(file, 0, 0, libc::POSIX_FADV_SEQUENTIAL)?;
    fadvise(file, 0, 0, libc::POSIX_FADV_NOREUSE)
}

/// Writes back `[offset, offset + len)` and waits, then drops those pages from the page cache (Q8).
pub fn flush_and_evict_range(file: &File, offset: u64, len: u64) -> io::Result<()> {
    let flags = libc::SYNC_FILE_RANGE_WAIT_BEFORE
        | libc::SYNC_FILE_RANGE_WRITE
        | libc::SYNC_FILE_RANGE_WAIT_AFTER;
    // SAFETY: plain syscall on an owned descriptor.
    if unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            offset as libc::off64_t,
            len as libc::off64_t,
            flags,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    fadvise(file, offset, len, libc::POSIX_FADV_DONTNEED)
}

/// `fsync` then `POSIX_FADV_DONTNEED` over the whole file (artifact files only).
pub fn evict(file: &File) -> io::Result<()> {
    // fsync needs no write access; on a read-only descriptor it still flushes the inode's dirty pages.
    // SAFETY: plain syscall on an owned descriptor.
    if unsafe { libc::fsync(file.as_raw_fd()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    fadvise(file, 0, 0, libc::POSIX_FADV_DONTNEED)
}

/// Page-cache residency of one file or a sum of files.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Residency {
    pub resident_pages: u64,
    pub total_pages: u64,
}

impl Residency {
    pub fn add(&mut self, other: Residency) {
        self.resident_pages += other.resident_pages;
        self.total_pages += other.total_pages;
    }

    /// Resident share in basis points (0..=10000); an empty set counts as 0.
    pub fn basis_points(&self) -> u64 {
        if self.total_pages == 0 {
            0
        } else {
            self.resident_pages * 10_000 / self.total_pages
        }
    }
}

pub fn page_size() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 {
        ps as u64
    } else {
        4096
    }
}

/// One `mincore` flag per page of `size` bytes of `file` (true = resident), from a read-only shared mapping that is
/// never touched.
pub fn resident_pages(file: &File, size: u64) -> io::Result<Vec<bool>> {
    if size == 0 {
        return Ok(Vec::new());
    }
    let pages = size.div_ceil(page_size());
    let len = usize::try_from(size).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: a fresh PROT_READ shared mapping of an open regular file; unmapped below on every path.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let mut vec = vec![0u8; pages as usize];
    // SAFETY: `addr`/`len` describe the mapping above; `vec` has one byte per page.
    let rc = unsafe { libc::mincore(addr, len, vec.as_mut_ptr()) };
    let err = io::Error::last_os_error();
    // SAFETY: unmaps exactly the mapping created above.
    unsafe { libc::munmap(addr, len) };
    if rc != 0 {
        return Err(err);
    }
    Ok(vec.into_iter().map(|b| b & 1 == 1).collect())
}

/// `mincore` residency of a whole file.
pub fn residency(file: &File, size: u64) -> io::Result<Residency> {
    let pages = resident_pages(file, size)?;
    Ok(Residency {
        resident_pages: pages.iter().filter(|r| **r).count() as u64,
        total_pages: pages.len() as u64,
    })
}

/// `mincore` residency of the pages touched by `ranges` (`(offset, len)`, ascending, non-overlapping) of a file of
/// `size` bytes; a page shared by two ranges counts once.
pub fn residency_of_ranges(
    file: &File,
    size: u64,
    ranges: impl IntoIterator<Item = (u64, u64)>,
) -> io::Result<Residency> {
    let pages = resident_pages(file, size)?;
    let ps = page_size();
    let mut out = Residency::default();
    let mut next_page = 0u64;
    for (offset, len) in ranges {
        let end = offset.saturating_add(len).min(size);
        if end <= offset {
            continue;
        }
        let first = (offset / ps).max(next_page);
        let last = (end - 1) / ps;
        for p in first..=last {
            out.total_pages += 1;
            out.resident_pages += u64::from(pages[p as usize]);
        }
        next_page = next_page.max(last + 1);
    }
    Ok(out)
}

/// Why an exclusive lock was not taken.
#[derive(Debug)]
pub enum LockError {
    /// Held by someone else and `no_wait` was set.
    Busy,
    Io(io::Error),
}

/// An exclusive `flock` on a lock file, released on drop.
#[derive(Debug)]
pub struct FileLock {
    _file: File,
}

impl FileLock {
    pub fn acquire(path: &Path, no_wait: bool) -> Result<FileLock, LockError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(LockError::Io)?;
        let op = libc::LOCK_EX | if no_wait { libc::LOCK_NB } else { 0 };
        // SAFETY: plain flock on an owned descriptor.
        if unsafe { libc::flock(file.as_raw_fd(), op) } != 0 {
            let e = io::Error::last_os_error();
            return Err(if e.raw_os_error() == Some(libc::EWOULDBLOCK) {
                LockError::Busy
            } else {
                LockError::Io(e)
            });
        }
        Ok(FileLock { _file: file })
    }
}

/// Environment variable that replaces `/proc/meminfo` for tests.
pub const TEST_MEMINFO_ENV: &str = "CEREC_TEST_MEMINFO";

/// `MemAvailable` in bytes from meminfo text. A test file may hold several `MemAvailable:` lines: reading number
/// `index` of a process uses line `index` (the last one once exhausted). `/proc/meminfo` has exactly one.
pub fn mem_available_from(text: &str, index: usize) -> Option<u64> {
    let values: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("MemAvailable:"))
        .collect();
    let line = values.get(index.min(values.len().checked_sub(1)?))?;
    let kib = line.trim().strip_suffix("kB")?.trim().parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

static MEMINFO_READS: AtomicUsize = AtomicUsize::new(0);

/// Current `MemAvailable` in bytes, from `CEREC_TEST_MEMINFO` if set, else `/proc/meminfo`.
pub fn mem_available() -> io::Result<u64> {
    let path = std::env::var_os(TEST_MEMINFO_ENV)
        .map_or_else(|| PathBuf::from("/proc/meminfo"), PathBuf::from);
    let text = std::fs::read_to_string(&path)?;
    let index = MEMINFO_READS.fetch_add(1, Ordering::Relaxed);
    mem_available_from(&text, index).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no MemAvailable in {}", path.display()),
        )
    })
}

/// `VmHWM` (peak RSS) of this process in bytes.
pub fn peak_rss() -> io::Result<u64> {
    let text = std::fs::read_to_string("/proc/self/status")?;
    text.lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().strip_suffix("kB")?.trim().parse::<u64>().ok())
        .map(|kib| kib * 1024)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no VmHWM in /proc/self/status"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    struct Tmp(PathBuf);
    impl Tmp {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!("cerec-fsio-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Tmp(p)
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn opens_regular_files_and_preads() {
        let t = Tmp::new("reg");
        std::fs::write(t.0.join("a.bin"), b"0123456789").unwrap();
        std::fs::create_dir(t.0.join("sub")).unwrap();
        std::fs::write(t.0.join("sub/b"), b"xy").unwrap();
        let dir = Dir::open(&t.0).unwrap();
        let f = dir.open_file("a.bin").unwrap();
        assert_eq!(f.stat.size, 10);
        let mut buf = [0u8; 4];
        f.read_exact_at(&mut buf, 3).unwrap();
        assert_eq!(&buf, b"3456");
        assert_eq!(dir.open_file("sub/b").unwrap().stat.size, 2);
        let mut names = dir.entries().unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![
                ("a.bin".to_string(), EntryKind::File),
                ("sub".to_string(), EntryKind::Dir)
            ]
        );
    }

    #[test]
    fn rejects_symlinks_specials_and_unsafe_names() {
        let t = Tmp::new("bad");
        std::fs::write(t.0.join("real"), b"x").unwrap();
        symlink(t.0.join("real"), t.0.join("link")).unwrap();
        std::fs::create_dir(t.0.join("d")).unwrap();
        symlink(t.0.join("d"), t.0.join("dlink")).unwrap();
        std::fs::write(t.0.join("d/f"), b"x").unwrap();
        let fifo = std::ffi::CString::new(t.0.join("fifo").as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        let dir = Dir::open(&t.0).unwrap();
        assert_eq!(
            dir.open_file("link").unwrap_err().kind,
            FsErrorKind::Symlink
        );
        assert_eq!(
            dir.open_file("dlink/f").unwrap_err().kind,
            FsErrorKind::Symlink
        );
        assert_eq!(
            dir.open_file("fifo").unwrap_err().kind,
            FsErrorKind::NotRegular
        );
        assert_eq!(
            dir.open_file("d").unwrap_err().kind,
            FsErrorKind::NotRegular
        );
        assert_eq!(
            dir.open_file("../real").unwrap_err().kind,
            FsErrorKind::BadPath
        );
        assert_eq!(
            dir.open_file("/etc/passwd").unwrap_err().kind,
            FsErrorKind::BadPath
        );
        assert_eq!(
            dir.open_file("a//b").unwrap_err().kind,
            FsErrorKind::BadPath
        );
        assert_eq!(
            dir.open_file("missing").unwrap_err().kind,
            FsErrorKind::Missing
        );
        assert!(Dir::open(&t.0.join("dlink")).is_err());
    }

    #[test]
    fn mem_available_parses_kib_and_steps_through_test_lines() {
        let text = "MemTotal: 100 kB\nMemAvailable:   36112732 kB\nCached: 1 kB\n";
        assert_eq!(mem_available_from(text, 0), Some(36112732 * 1024));
        assert_eq!(mem_available_from(text, 5), Some(36112732 * 1024));
        let seq = "MemAvailable: 10 kB\nMemAvailable: 2 kB\n";
        assert_eq!(mem_available_from(seq, 0), Some(10240));
        assert_eq!(mem_available_from(seq, 1), Some(2048));
        assert_eq!(mem_available_from(seq, 9), Some(2048));
        assert_eq!(mem_available_from("MemFree: 1 kB\n", 0), None);
        assert_eq!(mem_available_from("MemAvailable: x kB\n", 0), None);
    }

    #[test]
    fn residency_of_an_empty_file_is_zero_pages() {
        let t = Tmp::new("res");
        std::fs::write(t.0.join("e"), b"").unwrap();
        let f = Dir::open(&t.0).unwrap().open_file("e").unwrap();
        assert_eq!(
            residency(&f.file, 0).unwrap(),
            Residency {
                resident_pages: 0,
                total_pages: 0
            }
        );
    }

    #[test]
    fn exclusive_lock_is_exclusive() {
        let t = Tmp::new("lock");
        let a = FileLock::acquire(&t.0.join(".l"), true).unwrap();
        assert!(matches!(
            FileLock::acquire(&t.0.join(".l"), true),
            Err(LockError::Busy)
        ));
        drop(a);
        assert!(FileLock::acquire(&t.0.join(".l"), true).is_ok());
    }
}
