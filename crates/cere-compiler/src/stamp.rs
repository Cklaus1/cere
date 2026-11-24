//! The verification stamp (m21 design note Q7): `ROOT/<artifact_id>.verified.json`, a sidecar outside the artifact
//! directory. It caches a verification result against the `fstat` identity of the artifact's files; it is not an
//! authenticity proof (ARTIFACT_FORMAT.md:9).
//!
//! Format 2 (owner 2026-10-15) names each file's filesystem by its UUID-derived `fsid` (`fstatfs` `f_fsid`, which
//! ext2/3/4 fold from the superblock UUID) instead of the volatile `st_dev`, so a reboot that renumbers the block device
//! keeps the stamp. A file on a filesystem without a UUID-derived id is named by `dev`, as format 1 (no `format` key)
//! named every file.
use crate::fsio::{Dir, OpenFile, Stat};
use crate::{canon_json, Error};
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// Written by `cerec prepare` from the hashes computed while copying; only prepare's reuse decision accepts it.
pub const PREPARE_HASH: &str = "prepare_hash";
/// Written by `cerec verify --mode full`; the only source stamp mode accepts.
pub const FULL_VERIFY: &str = "full_verify";
pub const SUFFIX: &str = ".verified.json";
/// The format written: files named by `fsid` where it is UUID-derived, else by `dev`.
pub const FORMAT: u64 = 2;
/// Stamps without a `format` key: every file named by `dev`.
pub const LEGACY_FORMAT: u64 = 1;
const MAX_STAMP_BYTES: u64 = 16 << 20;

/// How a stamp names the filesystem holding a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fs {
    /// `f_fsid` of a filesystem whose driver derives it from the on-disk UUID: the same after a reboot.
    Fsid(u64),
    /// `st_dev`: format 1, and format 2 for a filesystem without a UUID-derived id. A reboot can renumber it.
    Dev(u64),
}

/// One file's record in the stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileId {
    pub fs: Fs,
    pub ino: u64,
    pub size: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
    pub mode: u32,
}

/// The filesystem of a held file as the kernel reports it now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsView {
    /// `st_dev`.
    pub dev: u64,
    /// `f_fsid` as one integer, `val[0] | val[1] << 32` (what glibc `fstatvfs` and Python `os.fstatvfs` report).
    pub fsid: u64,
    /// Whether the filesystem type derives `f_fsid` from its UUID (ext2/3/4, magic 0xEF53).
    pub uuid_derived: bool,
}

/// Why a held file is not the file a stamp record describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mismatch {
    /// ino, size, mtime, ctime, mode or the recorded filesystem differ.
    Changed,
    /// A format-1 record that matches except `dev`: format 1 cannot tell a renumbered device from another file.
    PredatesFsUuid,
}

type ViewFn = Box<dyn Fn(u64) -> FsView>;

thread_local! {
    static FS_VIEW: RefCell<Option<ViewFn>> = const { RefCell::new(None) };
}

/// Test seam: while `f` runs on this thread, `view(st_dev)` stands in for `fstatfs` of every held file (a reboot that
/// renumbers the device, another filesystem, a filesystem without a UUID).
pub fn with_fs_view<R>(view: impl Fn(u64) -> FsView + 'static, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<ViewFn>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            FS_VIEW.with(|v| *v.borrow_mut() = prev);
        }
    }
    let _restore = Restore(FS_VIEW.with(|v| v.borrow_mut().replace(Box::new(view))));
    f()
}

/// The filesystem of `file`, whose `fstat` is `st`.
pub fn fs_view(file: &File, st: &Stat) -> io::Result<FsView> {
    if let Some(v) = FS_VIEW.with(|v| v.borrow().as_ref().map(|view| view(st.dev))) {
        return Ok(v);
    }
    let mut buf = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `file` is an open descriptor and `buf` points to a writable `statfs`.
    if unsafe { libc::fstatfs(file.as_raw_fd(), buf.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fstatfs` returned 0, so it filled `buf`.
    let s = unsafe { buf.assume_init() };
    // SAFETY: `fsid_t` is `struct { int __val[2]; }`; libc keeps the field private.
    let [lo, hi]: [libc::c_int; 2] = unsafe { std::mem::transmute(s.f_fsid) };
    Ok(FsView {
        dev: st.dev,
        fsid: u64::from(lo as u32) | (u64::from(hi as u32) << 32),
        uuid_derived: s.f_type == libc::EXT4_SUPER_MAGIC,
    })
}

impl FileId {
    /// The record of a file with `fstat` `st` on `fs`: by `fsid` when it is UUID-derived, else by `dev`.
    pub fn new(st: &Stat, fs: &FsView) -> FileId {
        FileId {
            fs: if fs.uuid_derived {
                Fs::Fsid(fs.fsid)
            } else {
                Fs::Dev(fs.dev)
            },
            ino: st.ino,
            size: st.size,
            mtime_ns: st.mtime_ns,
            ctime_ns: st.ctime_ns,
            mode: st.mode,
        }
    }

    /// The record of a held file.
    pub fn of(f: &OpenFile) -> io::Result<FileId> {
        Ok(FileId::new(&f.stat, &fs_view(&f.file, &f.stat)?))
    }

    /// Whether the held file (`st` on `fs`) is the file this record of a stamp in `format` describes. ctime catches
    /// in-place writes and metadata changes even when mtime is put back; ino catches a replacement by rename.
    pub fn check(&self, format: u64, st: &Stat, fs: &FsView) -> Result<(), Mismatch> {
        if (self.ino, self.size, self.mtime_ns, self.ctime_ns, self.mode)
            != (st.ino, st.size, st.mtime_ns, st.ctime_ns, st.mode)
        {
            return Err(Mismatch::Changed);
        }
        match self.fs {
            Fs::Fsid(id) if id == fs.fsid => Ok(()),
            Fs::Dev(d) if d == fs.dev => Ok(()),
            Fs::Dev(_) if format == LEGACY_FORMAT => Err(Mismatch::PredatesFsUuid),
            _ => Err(Mismatch::Changed),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    /// [`FORMAT`], or [`LEGACY_FORMAT`] for a stamp read from before it.
    pub format: u64,
    pub manifest_sha256: String,
    pub stamp_source: String,
    pub source_fingerprint: String,
    pub files: BTreeMap<String, FileId>,
    pub verified_utc: String,
}

/// The stamp path of an artifact directory: its parent joined with `<basename>.verified.json`.
pub fn path_for(artifact_dir: &Path) -> Option<PathBuf> {
    let name = artifact_dir.file_name()?.to_str()?;
    Some(artifact_dir.parent()?.join(format!("{name}{SUFFIX}")))
}

fn id_json(f: &FileId) -> Value {
    let mut v = json!({
        "ino": f.ino, "size": f.size, "mode": f.mode,
        "mtime_ns": i64::try_from(f.mtime_ns).unwrap_or(i64::MAX),
        "ctime_ns": i64::try_from(f.ctime_ns).unwrap_or(i64::MAX),
    });
    match f.fs {
        Fs::Fsid(id) => v["fsid"] = json!(format!("{id:016x}")),
        Fs::Dev(d) => v["dev"] = json!(d),
    }
    v
}

fn id_from(v: &Value, format: u64) -> Option<FileId> {
    let o = v.as_object()?;
    if o.len() != 6 {
        return None;
    }
    let fs = match (o.get("fsid"), o.get("dev")) {
        (Some(id), None) if format == FORMAT => {
            let hex = id.as_str()?;
            if hex.len() != 16 || !hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
                return None;
            }
            Fs::Fsid(u64::from_str_radix(hex, 16).ok()?)
        }
        (None, Some(d)) => Fs::Dev(d.as_u64()?),
        _ => return None,
    };
    Some(FileId {
        fs,
        ino: o.get("ino")?.as_u64()?,
        size: o.get("size")?.as_u64()?,
        mtime_ns: i128::from(o.get("mtime_ns")?.as_i64()?),
        ctime_ns: i128::from(o.get("ctime_ns")?.as_i64()?),
        mode: u32::try_from(o.get("mode")?.as_u64()?).ok()?,
    })
}

impl Stamp {
    pub fn to_bytes(&self) -> Vec<u8> {
        let files: Map<String, Value> = self
            .files
            .iter()
            .map(|(k, s)| (k.clone(), id_json(s)))
            .collect();
        let mut v = json!({
            "manifest_sha256": self.manifest_sha256,
            "stamp_source": self.stamp_source,
            "source_fingerprint": self.source_fingerprint,
            "files": files,
            "verified_utc": self.verified_utc,
        });
        if self.format != LEGACY_FORMAT {
            v["format"] = json!(self.format);
        }
        canon_json::to_vec(&v).expect("stamp holds integers only")
    }

    pub fn parse(bytes: &[u8]) -> Option<Stamp> {
        let v = canon_json::parse_strict(bytes).ok()?;
        let o = v.as_object()?;
        let format = match o.get("format") {
            None => LEGACY_FORMAT,
            Some(f) => f.as_u64().filter(|&f| f == FORMAT)?,
        };
        if o.len() != 5 + usize::from(format != LEGACY_FORMAT) {
            return None;
        }
        let s = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_string);
        let stamp_source = s("stamp_source")?;
        if stamp_source != PREPARE_HASH && stamp_source != FULL_VERIFY {
            return None;
        }
        let files = o
            .get("files")?
            .as_object()?
            .iter()
            .map(|(k, v)| Some((k.clone(), id_from(v, format)?)))
            .collect::<Option<BTreeMap<_, _>>>()?;
        Some(Stamp {
            format,
            manifest_sha256: s("manifest_sha256")?,
            stamp_source,
            source_fingerprint: s("source_fingerprint")?,
            files,
            verified_utc: s("verified_utc")?,
        })
    }

    /// Reads the stamp at `path` (plain `None` when absent or malformed). Returns the stamp and the bytes read.
    pub fn read(path: &Path) -> Option<(Stamp, u64)> {
        let meta = std::fs::symlink_metadata(path).ok()?;
        if !meta.is_file() || meta.len() > MAX_STAMP_BYTES {
            return None;
        }
        let bytes = std::fs::read(path).ok()?;
        Some((Stamp::parse(&bytes)?, bytes.len() as u64))
    }

    /// Whether the stamp is for `manifest_sha256` and its records describe exactly `held` (manifest paths → the
    /// held descriptors now).
    pub fn matches(&self, manifest_sha256: &str, held: &BTreeMap<String, Held>) -> bool {
        self.manifest_sha256 == manifest_sha256
            && self.files.len() == held.len()
            && held.iter().all(|(p, h)| {
                self.files
                    .get(p)
                    .is_some_and(|f| f.check(self.format, &h.stat, &h.fs).is_ok())
            })
    }

    /// What this stamp does not anchor on a filesystem UUID, to be said aloud: a format-1 stamp (every file by
    /// `st_dev`), or the files of format 2 on a filesystem without a UUID-derived id.
    pub fn dev_note(&self) -> Option<String> {
        if self.format == LEGACY_FORMAT {
            return Some(
                "the stamp predates fs-uuid format and names files by st_dev (every st_dev matches); the next full \
                 verify writes format 2"
                    .into(),
            );
        }
        let n = self
            .files
            .values()
            .filter(|f| matches!(f.fs, Fs::Dev(_)))
            .count();
        (n > 0).then(|| {
            format!(
                "{n} of {} files are on a filesystem without a UUID-derived id: the stamp names them by st_dev, so a \
                 reboot that renumbers the device requires a full verify",
                self.files.len()
            )
        })
    }

    /// Atomically writes the stamp to `ROOT/<artifact_id>.verified.json`; returns the bytes written.
    pub fn write(&self, root: &Dir, artifact_id: &str) -> Result<u64, Error> {
        let bytes = self.to_bytes();
        let tmp = format!("{}.stamp", tmp_name());
        let name = format!("{artifact_id}{SUFFIX}");
        let io = |e: std::io::Error| Error::io(format!("writing stamp {name}"), e);
        let mut f = root.create_file(&tmp).map_err(io)?;
        f.write_all(&bytes).map_err(io)?;
        f.sync_all().map_err(io)?;
        drop(f);
        root.rename_replace(&tmp, &name).map_err(io)?;
        root.sync().map_err(io)?;
        Ok(bytes.len() as u64)
    }
}

/// A manifest path's file as held now: its `fstat` and filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Held {
    pub stat: Stat,
    pub fs: FsView,
}

impl Held {
    /// Its stamp record.
    pub fn id(&self) -> FileId {
        FileId::new(&self.stat, &self.fs)
    }
}

/// `fstat` and `fstatfs` of every manifest path through `dir` (no symlinks, regular files only).
pub fn stat_files<'a>(
    dir: &Dir,
    paths: impl Iterator<Item = &'a str>,
) -> Result<BTreeMap<String, Held>, Error> {
    paths
        .map(|p| {
            let f = dir.open_file(p)?;
            let fs = fs_view(&f.file, &f.stat).map_err(|e| Error::io(format!("fstatfs {p}"), e))?;
            Ok((p.to_string(), Held { stat: f.stat, fs }))
        })
        .collect()
}

/// A fresh `.tmp-<hex>` name (pid, time and a counter); stale ones are removed by `cerec prepare` under its lock.
pub fn tmp_name() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!(
        ".tmp-{:x}-{nanos:x}-{:x}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    utc_from_unix(secs)
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
pub fn utc_from_unix(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_utc() {
        assert_eq!(utc_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_from_unix(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_from_unix(1_791_291_071), "2026-10-06T12:51:11Z");
    }

    #[test]
    fn round_trips_both_formats_and_rejects_unknown_sources() {
        let id = |fs| FileId {
            fs,
            ino: 2,
            size: 3,
            mtime_ns: 4,
            ctime_ns: 5,
            mode: 0o100444,
        };
        let s = Stamp {
            format: FORMAT,
            manifest_sha256: "ab".into(),
            stamp_source: FULL_VERIFY.into(),
            source_fingerprint: "cd".into(),
            files: [
                ("a".to_string(), id(Fs::Fsid(0x715f_00ac_f4a6_680f))),
                ("b".to_string(), id(Fs::Dev(66306))),
            ]
            .into(),
            verified_utc: utc_from_unix(0),
        };
        let text = String::from_utf8(s.to_bytes()).unwrap();
        assert!(text.contains("\"format\": 2") && text.contains("\"fsid\": \"715f00acf4a6680f\""));
        assert_eq!(Stamp::parse(text.as_bytes()), Some(s.clone()));
        let bad = text.replace(FULL_VERIFY, "trust_me");
        assert_eq!(Stamp::parse(bad.as_bytes()), None);
        // Format 1 has no `format` key and names every file by `dev`; an `fsid` there, or format 3, is malformed.
        let legacy = Stamp {
            format: LEGACY_FORMAT,
            files: [("b".to_string(), id(Fs::Dev(66306)))].into(),
            ..s.clone()
        };
        let text1 = String::from_utf8(legacy.to_bytes()).unwrap();
        assert!(!text1.contains("format"));
        assert_eq!(Stamp::parse(text1.as_bytes()), Some(legacy));
        let fsid_in_legacy = Stamp {
            format: LEGACY_FORMAT,
            ..s.clone()
        };
        assert_eq!(Stamp::parse(&fsid_in_legacy.to_bytes()), None);
        assert_eq!(
            Stamp::parse(text.replace("\"format\": 2", "\"format\": 3").as_bytes()),
            None
        );
    }

    #[test]
    fn checks_every_field_and_tells_an_old_device_number_apart() {
        let st = Stat {
            dev: 66306,
            ino: 2,
            size: 3,
            mtime_ns: 4,
            ctime_ns: 5,
            mode: 0o100444,
        };
        let ext4 = FsView {
            dev: 66306,
            fsid: 7,
            uuid_derived: true,
        };
        let rebooted = FsView { dev: 66311, ..ext4 };
        let rec = FileId::new(&st, &ext4);
        assert_eq!(rec.fs, Fs::Fsid(7));
        assert_eq!(rec.check(FORMAT, &st, &rebooted), Ok(()));
        assert_eq!(
            rec.check(FORMAT, &st, &FsView { fsid: 8, ..ext4 }),
            Err(Mismatch::Changed)
        );
        let by_dev = FileId::new(
            &st,
            &FsView {
                uuid_derived: false,
                ..ext4
            },
        );
        assert_eq!(by_dev.fs, Fs::Dev(66306));
        assert_eq!(by_dev.check(FORMAT, &st, &ext4), Ok(()));
        assert_eq!(by_dev.check(FORMAT, &st, &rebooted), Err(Mismatch::Changed));
        assert_eq!(by_dev.check(LEGACY_FORMAT, &st, &ext4), Ok(()));
        assert_eq!(
            by_dev.check(LEGACY_FORMAT, &st, &rebooted),
            Err(Mismatch::PredatesFsUuid)
        );
        let touched = Stat { ctime_ns: 6, ..st };
        assert_eq!(rec.check(FORMAT, &touched, &ext4), Err(Mismatch::Changed));
        assert_eq!(
            by_dev.check(LEGACY_FORMAT, &touched, &rebooted),
            Err(Mismatch::Changed)
        );
    }

    /// `fs_view` reads `f_fsid` as glibc `fstatvfs` (and so Python's `os.fstatvfs`) reports it, and tells ext4.
    #[test]
    fn fs_view_matches_fstatvfs() {
        let f = File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")).unwrap();
        let st = Stat::of(&f).unwrap();
        let v = fs_view(&f, &st).unwrap();
        let mut buf = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: `f` is open and `buf` points to a writable `statvfs`.
        assert_eq!(
            unsafe { libc::fstatvfs(f.as_raw_fd(), buf.as_mut_ptr()) },
            0
        );
        // SAFETY: `fstatvfs` returned 0.
        let vfs = unsafe { buf.assume_init() };
        assert_eq!(v.fsid, vfs.f_fsid);
        assert_eq!(v.dev, st.dev);
        let fstype = std::process::Command::new("stat")
            .args(["-f", "-c", "%t", env!("CARGO_MANIFEST_DIR")])
            .output()
            .unwrap();
        assert_eq!(fstype.stdout == b"ef53\n", v.uuid_derived);
    }
}
