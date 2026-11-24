//! The prepared-artifact loader (m21 design note "Artifact contract for M2.2", Q7, Q8, Q16).
//!
//! [`open`] opens the artifact directory once, opens every listed file with `openat(O_NOFOLLOW)` and holds those
//! descriptors for the [`Artifact`]'s lifetime; every later read is a `pread` through them. The checks run in the
//! contract order shared with `scripts/verify_artifact.py` (`tests/fixtures/artifact_invalid.json`):
//! manifest size, digest, strict parse, canonical bytes, structure, rules; the directory walk; file opens and sizes;
//! shard headers (bounds, pinned hash, parse, tensor totals); families, expert metadata and groups; per-file tensor
//! counts and family counters; then the stamp ([`Mode::Stamp`]) or the sha256 of every file ([`Mode::Full`]).
//! Any rejection is a typed [`Error`]; there is no partial artifact.
//!
//! [`Artifact::stage`] streams the tensors of a placement [`Class`] into caller-provided buffers and hands every
//! piece to a [`Sink`].
use crate::families::{self, Role, Rules};
use crate::fsio::{self, Dir, EntryKind, FileLock, OpenFile, Residency};
use crate::manifest::{self, FamilyRow, Manifest, Profile, DIGEST_FILE, MANIFEST_FILE};
use crate::prepare::{Floor, LOCK_FILE};
use crate::safetensors::{self, Dtype, Header, HeaderError};
use crate::source::{role, SMALL_FILE_MAX};
use crate::stamp::{self, Mismatch, Stamp};
use crate::{hex, sha256_hex, Error, Limits, EXIT_IO};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Exit status of a stamp that does not cover the artifact as it is now ("full verify required").
pub const EXIT_STAMP: i32 = 4;
/// Smallest staging buffer [`Artifact::stage`] accepts.
pub const MIN_BUFFER_BYTES: usize = 64 << 10;
/// Alignment (and size granularity) of staging buffers.
pub const BUFFER_ALIGN: usize = 4096;
/// Read size and thread count of the full-mode hash pass.
pub const HASH_CHUNK_BYTES: usize = 8 << 20;
pub const HASH_THREADS: usize = 8;
/// Threads of the shard header pass at open (read, hash, parse).
pub const HEADER_THREADS: usize = 8;
const MAX_STAMP_BYTES: u64 = 16 << 20;

/// How much of the artifact [`open`] establishes beyond the headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Structure, headers and families; reads no payload byte.
    Headers,
    /// Headers plus a `full_verify` stamp matching `fstat` of the held descriptors; reads no payload byte.
    Stamp,
    /// Headers plus the sha256 of every listed file, read through the held descriptors.
    Full,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "headers" => Some(Mode::Headers),
            "stamp" => Some(Mode::Stamp),
            "full" => Some(Mode::Full),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Headers => "headers",
            Mode::Stamp => "stamp",
            Mode::Full => "full",
        }
    }
}

/// A placement class (Q8): `device`, `host` (`host_pinned`) or `all` tensors (unused ones included).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Device,
    Host,
    All,
}

impl Class {
    pub fn parse(s: &str) -> Option<Class> {
        match s {
            "device" => Some(Class::Device),
            "host" => Some(Class::Host),
            "all" => Some(Class::All),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Class::Device => "device",
            Class::Host => "host",
            Class::All => "all",
        }
    }

    pub fn admits(self, placement: &str) -> bool {
        match self {
            Class::Device => placement == "device",
            Class::Host => placement == "host_pinned",
            Class::All => true,
        }
    }
}

/// The cache state a run asks for (Q8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cache {
    /// The class's files are evicted first; the run is `cold` only if at most 1 % of its pages stay resident.
    Cold,
    /// An immediate repeat; `warm` only if at least 95 % of its pages are resident.
    Warm,
}

impl Cache {
    pub fn parse(s: &str) -> Option<Cache> {
        match s {
            "cold" => Some(Cache::Cold),
            "warm" => Some(Cache::Warm),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Cache::Cold => "cold",
            Cache::Warm => "warm",
        }
    }
}

/// The Q8 label of a run from the residency measured before it: `cold` (≤ 1 %) or `cold_unverified`; `warm`
/// (≥ 95 %) or `partial_warm`. An empty set is trivially cold and never warm.
pub fn cache_label(cache: Cache, before: Residency) -> &'static str {
    let (resident, total) = (
        u128::from(before.resident_pages),
        u128::from(before.total_pages),
    );
    match cache {
        Cache::Cold if resident * 100 <= total => "cold",
        Cache::Cold => "cold_unverified",
        Cache::Warm if total > 0 && resident * 100 >= total * 95 => "warm",
        Cache::Warm => "partial_warm",
    }
}

/// A listed file and its held descriptor.
#[derive(Debug)]
pub struct ArtifactFile {
    pub path: String,
    pub role: String,
    pub bytes: u64,
    pub sha256: String,
    open: OpenFile,
    shard: Option<usize>,
}

/// A weights file: its parsed header, the rule of every tensor, and the sha256 state after the header prefix (so
/// hashing the file never re-reads the prefix).
#[derive(Debug)]
struct Shard {
    file: usize,
    header: Header,
    rule: Vec<usize>,
    prefix_hash: Sha256,
}

/// One tensor as the loader sees it (Q16: `offset` may be element-misaligned; copy, never view).
#[derive(Debug, Clone, Copy)]
pub struct TensorRef<'a> {
    pub path: &'a str,
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [u64],
    /// Absolute file offset.
    pub offset: u64,
    pub bytes: u64,
    pub element_aligned: bool,
    pub family: &'a str,
    pub role: Role,
    pub placement: &'a str,
}

/// The Q7 counters of one artifact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteCounts {
    pub payload_bytes_read: u64,
    pub metadata_bytes_read: u64,
}

#[derive(Debug, Default)]
struct Counters {
    payload: AtomicU64,
    metadata: AtomicU64,
}

impl Counters {
    fn payload(&self, n: u64) {
        self.payload.fetch_add(n, Ordering::Relaxed);
    }
    fn metadata(&self, n: u64) {
        self.metadata.fetch_add(n, Ordering::Relaxed);
    }
}

/// Stage timings of [`open`] and the hash pass (Q8). DISCOVER is the root open, the manifest read and its digest;
/// VALIDATE is everything after it up to the stamp check (absent for the unvalidated baseline).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timings {
    pub discover_ms: u64,
    pub validate_ms: Option<u64>,
    pub hash_ms: Option<u64>,
}

/// An opened artifact. Its descriptors stay open until it is dropped.
#[derive(Debug)]
pub struct Artifact {
    pub manifest_sha256: String,
    pub manifest: Manifest,
    pub rules: Rules,
    pub profile: Profile,
    pub files: Vec<ArtifactFile>,
    shards: Vec<Shard>,
    /// Placement of every rule, from the embedded profile.
    placements: Vec<String>,
    root: Dir,
    root_path: PathBuf,
    artifact_id: String,
    counters: Counters,
    timings: Timings,
    hashed: bool,
    /// Filled by the full hash pass or the first [`Artifact::token_map`] call.
    token_map: OnceLock<Vec<u32>>,
    /// [`Stamp::dev_note`] of the stamp checked or written, if any.
    stamp_note: OnceLock<String>,
}

fn ms(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

fn stamp_err(msg: impl Into<String>) -> Error {
    Error {
        class: "stamp",
        exit: EXIT_STAMP,
        message: format!("full verify required: {}", msg.into()),
    }
}

fn buffer_err(msg: impl Into<String>) -> Error {
    Error::reject("buffer", msg)
}

fn decode_u32le(raw: &[u8]) -> Vec<u32> {
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect()
}

/// `pread` through a held descriptor; a file that ends early is a `file_size` rejection.
fn read_at(f: &ArtifactFile, buf: &mut [u8], offset: u64) -> Result<(), Error> {
    f.open.read_exact_at(buf, offset).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::reject("file_size", format!("{} shrank while reading", f.path))
        } else {
            Error::io(&f.path, e)
        }
    })
}

/// An `O_DIRECT` read of `[start, start + len)` of `f` through `direct`: the [`DIRECT_ALIGN`]-aligned span around
/// it into `buf` (aligned, at least that long); returns where `start` is in `buf`. A file that ends before
/// `start + len` is a `file_size` rejection.
fn read_direct(
    f: &ArtifactFile,
    direct: &std::fs::File,
    buf: &mut [u8],
    start: u64,
    len: u64,
) -> Result<usize, Error> {
    use std::os::unix::fs::FileExt;
    let from = start / DIRECT_ALIGN * DIRECT_ALIGN;
    let end = start + len;
    let want = (end - from) as usize;
    let buf = &mut buf[..(end.div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN - from) as usize];
    let mut got = 0;
    while got < want {
        match direct.read_at(&mut buf[got..], from + got as u64) {
            // Only the end of the file leaves a direct read short at an unaligned length (and the next read
            // there would be EINVAL).
            Ok(n) if n == 0 || (got + n < want && (got + n) as u64 % DIRECT_ALIGN != 0) => {
                return Err(Error::reject(
                    "file_size",
                    format!("{} shrank while reading", f.path),
                ))
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::io(&f.path, e)),
        }
    }
    Ok((start - from) as usize)
}

/// Hands one piece to a sink; its refusal is class `sink`.
fn accept<S: Sink + ?Sized>(
    sink: &S,
    t: &TensorRef<'_>,
    off: u64,
    piece: &[u8],
    last: bool,
) -> Result<(), Error> {
    sink.accept(t, off, piece, last).map_err(|e| Error {
        class: "sink",
        exit: EXIT_IO,
        message: format!("{}:{}: {e}", t.path, t.name),
    })
}

/// Samples `(ms since t, reads in flight, bytes read)` every [`SAMPLE_MS`] ms until `stop` holds true, then once more.
fn sample<S: ?Sized>(
    shared: &Shared<'_, S>,
    t: Instant,
    stop: &(Mutex<bool>, Condvar),
) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    let mut done = stop.0.lock().expect("sampler");
    loop {
        out.push((
            ms(t),
            shared.reads_now.load(Ordering::Relaxed),
            shared.bytes.load(Ordering::Relaxed),
        ));
        if *done {
            return out;
        }
        done = stop
            .1
            .wait_timeout(done, Duration::from_millis(SAMPLE_MS))
            .expect("sampler")
            .0;
    }
}

/// The mean reads in flight over `samples` and, per second, their mean and the bytes read in it; means in hundredths
/// (reports are canonical JSON: integers only).
fn timeline(samples: &[(u64, u64, u64)]) -> (u64, Vec<Second>) {
    let centi = |sum: u64, n: u64| if n == 0 { 0 } else { (sum * 100 + n / 2) / n };
    let mean = centi(samples.iter().map(|s| s.1).sum(), samples.len() as u64);
    let mut out: Vec<Second> = Vec::new();
    let (mut sum, mut count, mut before) = (0u64, 0u64, 0u64);
    for &(t_ms, reads, bytes) in samples {
        let t_s = t_ms / 1000;
        match out.last_mut() {
            Some(s) if s.t_s == t_s => {
                sum += reads;
                count += 1;
                s.bytes = bytes - before;
            }
            _ => {
                if let Some(s) = out.last_mut() {
                    s.reads_in_flight_x100 = centi(sum, count);
                    before += s.bytes;
                }
                out.push(Second {
                    t_s,
                    reads_in_flight_x100: 0,
                    bytes: bytes - before,
                });
                (sum, count) = (reads, 1);
            }
        }
    }
    if let Some(s) = out.last_mut() {
        s.reads_in_flight_x100 = centi(sum, count);
    }
    (mean, out)
}

/// Opens the artifact root once (`O_RDONLY|O_DIRECTORY|O_NOFOLLOW`); returns it with its real path and name.
fn open_root(dir: &Path) -> Result<(Dir, PathBuf, String), Error> {
    if std::fs::symlink_metadata(dir).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(Error::reject("symlink", "artifact root is a symlink"));
    }
    let root = Dir::open(dir).map_err(|e| match e.raw_os_error() {
        Some(libc::ELOOP) => Error::reject("symlink", "artifact root is a symlink"),
        Some(libc::ENOTDIR) => Error::reject(
            "special_file",
            format!("{} is not a directory", dir.display()),
        ),
        Some(libc::ENOENT) => Error::reject("missing_file", format!("{}: {e}", dir.display())),
        _ => Error::io(dir.display(), e),
    })?;
    let real = std::fs::canonicalize(dir).map_err(|e| Error::io(dir.display(), e))?;
    let name = real
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::reject("unsafe_path", format!("{} has no name", real.display())))?
        .to_string();
    Ok((root, real, name))
}

/// Steps 1-2: the manifest's size bound and its digest; returns the verified bytes.
fn read_manifest(
    root: &Dir,
    expected_sha256: Option<&str>,
    limits: &Limits,
    counters: &Counters,
) -> Result<Vec<u8>, Error> {
    let f = root.open_file(MANIFEST_FILE)?;
    if f.stat.size > limits.max_manifest_bytes {
        return Err(Error::reject(
            "manifest_size",
            format!(
                "manifest is {} bytes > {}",
                f.stat.size, limits.max_manifest_bytes
            ),
        ));
    }
    let mut raw = vec![0u8; f.stat.size as usize];
    f.read_exact_at(&mut raw, 0)
        .map_err(|e| Error::io(MANIFEST_FILE, e))?;
    counters.metadata(raw.len() as u64);
    let actual = sha256_hex(&raw);
    if let Some(expected) = expected_sha256.filter(|e| *e != actual) {
        return Err(Error::reject(
            "manifest_digest",
            format!("manifest sha256 {actual} != expected {expected}"),
        ));
    }
    Ok(raw)
}

/// Step 7: everything under the root is a listed file, the manifest pair, or an ancestor directory of a listed file.
fn walk(
    dir: &Dir,
    prefix: &str,
    listed: &BTreeSet<&str>,
    dirs: &BTreeSet<String>,
) -> Result<(), Error> {
    let mut entries = dir
        .entries()
        .map_err(|e| Error::io(format!("listing {prefix:?}"), e))?;
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    for (name, kind) in entries {
        let rel = format!("{prefix}{name}");
        match kind {
            EntryKind::Symlink => {
                return Err(Error::reject("symlink", format!("{rel} is a symlink")))
            }
            EntryKind::Dir => {
                if !dirs.contains(&rel) {
                    return Err(Error::reject(
                        "unlisted_file",
                        format!("{rel}/ is not listed"),
                    ));
                }
                walk(&dir.open_dir(&name)?, &format!("{rel}/"), listed, dirs)?;
            }
            EntryKind::File => {
                if !listed.contains(rel.as_str()) && rel != MANIFEST_FILE && rel != DIGEST_FILE {
                    return Err(Error::reject(
                        "unlisted_file",
                        format!("{rel} is not listed"),
                    ));
                }
            }
            EntryKind::Other => {
                return Err(Error::reject(
                    "special_file",
                    format!("{rel} is not a regular file or directory"),
                ))
            }
        }
    }
    Ok(())
}

/// Opens every listed file through `root` (no symlinks, regular files only); `check_size` compares with the manifest.
fn open_files(root: &Dir, m: &Manifest, check_size: bool) -> Result<Vec<ArtifactFile>, Error> {
    let mut out = Vec::with_capacity(m.files.len());
    for f in &m.files {
        let open = root.open_file(&f.path)?;
        if check_size && open.stat.size != f.bytes {
            return Err(Error::reject(
                "file_size",
                format!(
                    "{} is {} bytes, manifest says {}",
                    f.path, open.stat.size, f.bytes
                ),
            ));
        }
        out.push(ArtifactFile {
            path: f.path.clone(),
            role: f.role.clone(),
            bytes: f.bytes,
            sha256: f.sha256.clone(),
            open,
            shard: None,
        });
    }
    Ok(out)
}

/// Reads a shard's header prefix `[0, 8 + n)` through its descriptor (bounds first).
fn read_prefix(f: &ArtifactFile, limits: &Limits) -> Result<Vec<u8>, Error> {
    let size = f.open.stat.size;
    if size < 8 {
        return Err(HeaderError::Bounds.into());
    }
    let mut first8 = [0u8; 8];
    read_at(f, &mut first8, 0)?;
    let n = safetensors::header_len(&first8, size, limits)?;
    let mut prefix = vec![0u8; 8 + n as usize];
    read_at(f, &mut prefix, 0)?;
    Ok(prefix)
}

/// One weights shard's parsed header and its prefix's running hash, or why it was rejected.
type ShardHeader = Result<(Header, Sha256), Error>;

/// Steps 9-10 for every weights file in manifest order; `pinned` compares each prefix with the manifest's
/// `header_bytes`/`header_sha256` and enforces the total tensor bound. The shards are read, hashed and parsed on up to
/// [`HEADER_THREADS`] threads (BeatSglang L1: ~40 MB of header JSON in 21 shards was one serial second of every
/// load); the results are then judged in manifest order, so the first failing shard's error is the one returned, as
/// when they ran one after another.
fn read_shards(
    files: &mut [ArtifactFile],
    m: &Manifest,
    limits: &Limits,
    counters: &Counters,
    pinned: bool,
) -> Result<Vec<(Header, Sha256)>, Error> {
    let weights: Vec<usize> = (0..files.len().min(m.files.len()))
        .filter(|&i| m.files[i].role == role::WEIGHTS)
        .collect();
    let shared: &[ArtifactFile] = files;
    let shard = |i: usize| -> ShardHeader {
        let (f, entry) = (&shared[i], &m.files[i]);
        let prefix = read_prefix(f, limits)?;
        counters.metadata(prefix.len() as u64);
        let mut h = Sha256::new();
        h.update(&prefix);
        if pinned {
            let info = entry.safetensors.as_ref().ok_or_else(|| {
                Error::reject(
                    "manifest_rules",
                    format!("{} has no safetensors info", f.path),
                )
            })?;
            if prefix.len() as u64 != info.header_bytes
                || hex(&h.clone().finalize()) != info.header_sha256
            {
                return Err(Error::reject(
                    "header_hash",
                    format!("{}: header differs from the manifest", f.path),
                ));
            }
        }
        Ok((safetensors::parse(&prefix, f.open.stat.size, limits)?, h))
    };
    let slots: Vec<Mutex<Option<ShardHeader>>> = weights.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let work = || loop {
        let k = next.fetch_add(1, Ordering::Relaxed);
        let Some(&i) = weights.get(k) else { break };
        let r = shard(i);
        *slots[k].lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(HEADER_THREADS)
        .min(weights.len().max(1));
    std::thread::scope(|scope| {
        for _ in 1..threads {
            // A thread that cannot start leaves its shards to the others and the calling thread.
            let _ = std::thread::Builder::new().spawn_scoped(scope, work);
        }
        work();
    });
    let mut out = Vec::with_capacity(weights.len());
    let mut total = 0u64;
    for (&i, slot) in weights.iter().zip(slots) {
        let (header, h) = slot
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .expect("every shard read")?;
        total += header.tensors.len() as u64;
        if pinned && total > limits.max_total_tensors {
            return Err(Error::reject(
                "too_many_tensors",
                format!("more than {} tensors", limits.max_total_tensors),
            ));
        }
        files[i].shard = Some(out.len());
        out.push((header, h));
    }
    Ok(out)
}

/// The placement of every rule from the embedded profile.
fn rule_placements(rules: &Rules, profile: &Profile) -> Result<Vec<String>, Error> {
    rules
        .rules
        .iter()
        .map(|r| {
            profile
                .placement_of(r.role)
                .map(str::to_string)
                .ok_or_else(|| {
                    Error::reject(
                        "profile",
                        format!("profile has no placement for role {}", r.role.name()),
                    )
                })
        })
        .collect()
}

/// Opens and validates the artifact at `dir` against the pinned manifest digest, with the default limits.
pub fn open(dir: &Path, expected_manifest_sha256: &str, mode: Mode) -> Result<Artifact, Error> {
    open_with(dir, expected_manifest_sha256, mode, &Limits::default())
}

/// [`open`] with explicit resource limits (Q15).
pub fn open_with(
    dir: &Path,
    expected_manifest_sha256: &str,
    mode: Mode,
    limits: &Limits,
) -> Result<Artifact, Error> {
    let t0 = Instant::now();
    let counters = Counters::default();
    let (root, root_path, artifact_id) = open_root(dir)?;
    let raw = read_manifest(&root, Some(expected_manifest_sha256), limits, &counters)?;
    let discover_ms = ms(t0);
    let t1 = Instant::now();
    let (m, rules, profile) = manifest::parse(&raw)?;
    manifest::check_rules(&m, &rules, &profile, limits)?;
    let listed: BTreeSet<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
    let dirs: BTreeSet<String> = m
        .files
        .iter()
        .flat_map(|f| {
            let parts: Vec<&str> = f.path.split('/').collect();
            (1..parts.len()).map(move |i| parts[..i].join("/"))
        })
        .collect();
    walk(&root, "", &listed, &dirs)?;
    let mut files = open_files(&root, &m, true)?;
    let parsed = read_shards(&mut files, &m, limits, &counters, true)?;
    let headers: Vec<&Header> = parsed.iter().map(|(h, _)| h).collect();
    let classified = families::classify(&rules, &headers)?;
    for (entry, (h, _)) in m
        .files
        .iter()
        .filter(|f| f.role == role::WEIGHTS)
        .zip(&parsed)
    {
        if Some(h.tensors.len() as u64) != entry.safetensors.as_ref().map(|s| s.tensors) {
            return Err(Error::reject(
                "family_totals",
                format!("{}: tensor count differs from the manifest", entry.path),
            ));
        }
    }
    let (rows, _) = manifest::families_and_totals(&rules, &profile, &classified.tallies)?;
    if rows != m.families {
        let bad = rows
            .iter()
            .zip(&m.families)
            .find(|(a, b)| a != b)
            .map_or("?", |(a, _)| a.family.as_str());
        return Err(Error::reject(
            "family_totals",
            format!("family {bad}: headers differ from the manifest counters"),
        ));
    }
    let placements = rule_placements(&rules, &profile)?;
    let shards = parsed
        .into_iter()
        .zip(classified.per_shard)
        .enumerate()
        .map(|(si, ((header, prefix_hash), rule))| Shard {
            file: files
                .iter()
                .position(|f| f.shard == Some(si))
                .expect("shard file"),
            header,
            rule,
            prefix_hash,
        })
        .collect();
    let mut art = Artifact {
        manifest_sha256: expected_manifest_sha256.to_string(),
        manifest: m,
        rules,
        profile,
        files,
        shards,
        placements,
        root,
        root_path,
        artifact_id,
        counters,
        timings: Timings {
            discover_ms,
            validate_ms: None,
            hash_ms: None,
        },
        hashed: false,
        token_map: OnceLock::new(),
        stamp_note: OnceLock::new(),
    };
    art.timings.validate_ms = Some(ms(t1));
    match mode {
        Mode::Headers => {}
        Mode::Stamp => art.check_stamp()?,
        Mode::Full => art.hash_files(None)?,
    }
    Ok(art)
}

/// The staged-copy baseline (Q8): the same files and tensor byte ranges as [`open`], with no validation beyond what
/// locating the ranges needs. The manifest digest is still compared, the manifest parsed and the shard headers
/// parsed for their offsets; there is no walk, size or header-hash check, no family/group check and no stamp.
/// Each tensor takes the placement of the first rule matching its name.
pub fn open_baseline(
    dir: &Path,
    expected_manifest_sha256: &str,
    limits: &Limits,
) -> Result<Artifact, Error> {
    let t0 = Instant::now();
    let counters = Counters::default();
    let (root, root_path, artifact_id) = open_root(dir)?;
    let raw = read_manifest(&root, Some(expected_manifest_sha256), limits, &counters)?;
    let (m, rules, profile) = manifest::parse(&raw)?;
    let mut files = open_files(&root, &m, false)?;
    let parsed = read_shards(&mut files, &m, limits, &counters, false)?;
    let placements = rule_placements(&rules, &profile)?;
    let mut shards = Vec::with_capacity(parsed.len());
    for (si, (header, prefix_hash)) in parsed.into_iter().enumerate() {
        let rule = header
            .tensors
            .iter()
            .map(|t| {
                rules
                    .rules
                    .iter()
                    .position(|r| r.pattern.full_match(&t.name))
                    .ok_or_else(|| families::FamilyError::Unmatched(t.name.clone()).into())
            })
            .collect::<Result<Vec<usize>, Error>>()?;
        shards.push(Shard {
            file: files
                .iter()
                .position(|f| f.shard == Some(si))
                .expect("shard file"),
            header,
            rule,
            prefix_hash,
        });
    }
    Ok(Artifact {
        manifest_sha256: expected_manifest_sha256.to_string(),
        manifest: m,
        rules,
        profile,
        files,
        shards,
        placements,
        root,
        root_path,
        artifact_id,
        counters,
        timings: Timings {
            discover_ms: ms(t0),
            validate_ms: None,
            hash_ms: None,
        },
        hashed: false,
        token_map: OnceLock::new(),
        stamp_note: OnceLock::new(),
    })
}

/// `fsync` + `POSIX_FADV_DONTNEED` on every artifact file (the listed files and the manifest pair) of a structurally
/// valid manifest, then their `mincore` residency. With `expected_manifest_sha256` the digest is checked first (the
/// cold pre-pass of `verify`/`probe-load`); `cerec evict DIR` pins none, so nothing beyond the file list is trusted.
/// Returns the evicted file count and the residency afterwards. The descriptors are closed on return.
pub fn evict_dir(
    dir: &Path,
    expected_manifest_sha256: Option<&str>,
    limits: &Limits,
) -> Result<(usize, Residency), Error> {
    let (root, _, _) = open_root(dir)?;
    let raw = read_manifest(
        &root,
        expected_manifest_sha256,
        limits,
        &Counters::default(),
    )?;
    let (m, rules, profile) = manifest::parse(&raw)?;
    manifest::check_rules(&m, &rules, &profile, limits)?;
    let mut files: Vec<(String, OpenFile)> = open_files(&root, &m, false)?
        .into_iter()
        .map(|f| (f.path, f.open))
        .collect();
    for name in [MANIFEST_FILE, DIGEST_FILE] {
        match root.open_file(name) {
            Ok(f) => files.push((name.to_string(), f)),
            Err(e) if name == DIGEST_FILE && e.kind == fsio::FsErrorKind::Missing => {}
            Err(e) => return Err(e.into()),
        }
    }
    let mut after = Residency::default();
    for (path, f) in &files {
        fsio::evict(&f.file).map_err(|e| Error::io(format!("evicting {path}"), e))?;
    }
    for (path, f) in &files {
        after.add(
            fsio::residency(&f.file, f.stat.size)
                .map_err(|e| Error::io(format!("mincore {path}"), e))?,
        );
    }
    Ok((files.len(), after))
}

/// Receives staged pieces (contract "Read", m216 design note L7). The pieces of one tensor tile it exactly, the
/// one that ends it with `is_last`; they arrive from any staging thread in any order, concurrently with each other
/// and with other tensors' pieces. The slice must not be retained after `accept` returns (the buffer is reused once
/// every sink has returned); an `Err` aborts staging.
pub trait Sink: Sync {
    /// Whether staging delivers `tensor` at all (default: every tensor of the class). A declined tensor's bytes are
    /// never read, unless its file is hashed ([`StageOptions::hash`]): then they are read and hashed but not
    /// delivered (a direct read, [`ReadMode::Direct`], may also cover less than [`DIRECT_ALIGN`] bytes of it at the
    /// edge of an aligned span; those are neither counted as payload nor delivered).
    fn wants(&self, _tensor: &TensorRef<'_>) -> bool {
        true
    }

    /// Whether `tensor` may be read after every tensor that is not late (default: no); an order hint only
    /// (load-soft design note): the bytes delivered are the same.
    fn late(&self, _tensor: &TensorRef<'_>) -> bool {
        false
    }

    fn accept(
        &self,
        tensor: &TensorRef<'_>,
        offset_in_tensor: u64,
        data: &[u8],
        is_last: bool,
    ) -> Result<(), String>;
}

/// A 4096-aligned staging buffer (contract "Read"). Callers may also pass their own memory as `&mut [u8]`.
#[derive(Debug)]
pub struct Buffer {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: `Buffer` exclusively owns its allocation; access goes through `&mut self` or `&self`.
unsafe impl Send for Buffer {}
// SAFETY: shared references only expose the length.
unsafe impl Sync for Buffer {}

fn check_buffer_len(len: usize) -> Result<(), Error> {
    if len < MIN_BUFFER_BYTES || len % BUFFER_ALIGN != 0 {
        return Err(buffer_err(format!(
            "staging buffers must be >= {MIN_BUFFER_BYTES} bytes and a multiple of {BUFFER_ALIGN}, got {len}"
        )));
    }
    Ok(())
}

impl Buffer {
    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len, BUFFER_ALIGN).expect("checked buffer layout")
    }

    /// A zeroed buffer of `len` bytes (≥ [`MIN_BUFFER_BYTES`], a multiple of [`BUFFER_ALIGN`]).
    pub fn new(len: usize) -> Result<Buffer, Error> {
        check_buffer_len(len)?;
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(Self::layout(len)) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::io("allocating a staging buffer", "out of memory"))?;
        Ok(Buffer { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` points to `len` initialized (zeroed) bytes owned by `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl AsMut<[u8]> for Buffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), Self::layout(self.len)) };
    }
}

/// `count` staging buffers of `bytes` each.
pub fn buffers(count: usize, bytes: usize) -> Result<Vec<Buffer>, Error> {
    if count == 0 {
        return Err(buffer_err("at least one staging buffer is needed"));
    }
    (0..count).map(|_| Buffer::new(bytes)).collect()
}

/// How [`Artifact::stage`] reads tensor data (m216 design note L7).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadMode {
    /// `pread` through the held descriptors (page cache).
    #[default]
    Buffered,
    /// `O_DIRECT` reads of [`DIRECT_ALIGN`]-aligned spans through a second descriptor of the same inode (bypassing
    /// the page cache); a file system that refuses `O_DIRECT` reads that file buffered.
    Direct,
    /// Per file: direct when less than half of its pages are resident (`mincore`), else buffered.
    Auto,
}

impl ReadMode {
    pub fn parse(s: &str) -> Option<ReadMode> {
        match s {
            "buffered" => Some(ReadMode::Buffered),
            "direct" => Some(ReadMode::Direct),
            "auto" => Some(ReadMode::Auto),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ReadMode::Buffered => "buffered",
            ReadMode::Direct => "direct",
            ReadMode::Auto => "auto",
        }
    }
}

/// Offset, length and buffer alignment of a direct read; a direct pass reads spans of at most the buffer length
/// minus `2 * DIRECT_ALIGN`, so the aligned span still fits.
pub const DIRECT_ALIGN: u64 = 4096;
/// Delivery threads per sink when [`StageOptions::consumers`] is 0: one per buffer, at most this many.
pub const MAX_DEFAULT_CONSUMERS: usize = 16;
/// Most sinks one [`Artifact::stage_to`] call feeds.
pub const MAX_SINKS: usize = 64;

/// Options of [`Artifact::stage`]; `None` means no memory floor, no hashing, buffered reads, one reader per buffer.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageOptions<'a> {
    /// Rechecked after every chunk (1 GiB) of payload read.
    pub floor: Option<&'a Floor>,
    /// Hash every file whose tensors are all in the class (header prefix + staged pieces) and compare it with the
    /// manifest sha256 (`file_hash` on a difference). Files the class covers only partly are not hashed.
    pub hash: bool,
    pub read: ReadMode,
    /// Reader threads, each with at most one read in flight; 0 = one per buffer.
    pub readers: usize,
    /// Delivery threads per sink; 0 = one per buffer, at most [`MAX_DEFAULT_CONSUMERS`].
    pub consumers: usize,
}

/// One second of a stage call: the mean of the reads in flight sampled in it (hundredths) and the bytes read in it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Second {
    pub t_s: u64,
    pub reads_in_flight_x100: u64,
    pub bytes: u64,
}

/// What one [`Artifact::stage`] call did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageReport {
    pub tensors: u64,
    pub bytes: u64,
    /// Most staging bytes holding data at once (read, not yet released by every sink).
    pub high_water_bytes: u64,
    /// Reader threads that ran (≤ the readers asked for, ≤ units of the plan).
    pub threads: usize,
    /// Delivery threads per sink, and the sinks.
    pub consumers: usize,
    pub sinks: usize,
    /// Tensor-data reads (a run of contiguous tensors is one; so is each buffer-sized piece of a larger tensor).
    pub reads: u64,
    /// Files read with `O_DIRECT` ([`ReadMode`]).
    pub direct_files: u64,
    pub read_ms: u64,
    /// Summed thread time inside the tensor-data reads (page cache or disk into the staging buffers).
    pub pread_ms: u64,
    /// Summed reader time waiting for a free buffer (the sinks are behind) and delivery time waiting for a filled
    /// one (the reads are behind).
    pub wait_free_ms: u64,
    pub wait_fill_ms: u64,
    /// Summed time inside the sinks' `accept`.
    pub deliver_ms: u64,
    /// Summed thread time spent hashing (0 without `hash`).
    pub hash_ms: u64,
    pub hashed_files: u64,
    /// Reads in flight, sampled every [`SAMPLE_MS`] ms: the mean (hundredths) and the most seen; and per second.
    pub reads_in_flight_mean_x100: u64,
    pub reads_in_flight_max: u64,
    pub timeline: Vec<Second>,
}

/// Sampling period of [`StageReport::timeline`].
pub const SAMPLE_MS: u64 = 50;

/// State shared by the staging threads.
struct Shared<'a, S: ?Sized> {
    class: Class,
    sinks: &'a [&'a S],
    opts: StageOptions<'a>,
    stop: AtomicBool,
    in_flight: AtomicU64,
    high_water: AtomicU64,
    hash_ns: AtomicU64,
    pread_ns: AtomicU64,
    wait_free_ns: AtomicU64,
    wait_fill_ns: AtomicU64,
    deliver_ns: AtomicU64,
    tensors: AtomicU64,
    bytes: AtomicU64,
    hashed_files: AtomicU64,
    reads: AtomicU64,
    reads_now: AtomicU64,
    reads_max: AtomicU64,
    first_err: Mutex<Option<Error>>,
}

impl<S: ?Sized> Shared<'_, S> {
    fn fail(&self, e: Error) {
        self.stop.store(true, Ordering::Relaxed);
        self.first_err.lock().expect("error slot").get_or_insert(e);
    }
}

/// One work item ([`Artifact::plan`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    /// Every class tensor of a hashed shard, in file-offset order, read and delivered by one reader.
    Shard(usize),
    /// `Plan::order[lo..hi]`, contiguous tensors of `shard` that together fit one buffer: one read.
    Run { shard: usize, lo: usize, hi: usize },
    /// `[off, off + len)` of `Plan::order[at]`, a tensor of `shard` larger than a buffer: one read.
    Piece {
        shard: usize,
        at: usize,
        off: u64,
        len: u64,
    },
}

/// The units of one [`Artifact::stage`] call; runs and pieces index the read tensors in `order`, and `sinks[j]` is
/// the bit set of the sinks that want `order[j]`.
#[derive(Debug, Default)]
struct Plan {
    order: Vec<usize>,
    sinks: Vec<u64>,
    units: Vec<Unit>,
}

/// The staging buffers as raw slots; a slot is owned by one reader (taken from the free list) until it is filled,
/// then shared read-only by the delivery threads of every sink, and free again when the last of them releases it.
struct Slots(Vec<(*mut u8, usize)>);

// SAFETY: the ring protocol above gives each slot one writer at a time, and readers only after its fill is published
// under the queue mutex; the buffers outlive the scope that uses the slots.
unsafe impl Send for Slots {}
// SAFETY: as above.
unsafe impl Sync for Slots {}

/// Where a filled slot's unit is: `[at, at + len)` of the slot.
#[derive(Debug, Clone, Copy, Default)]
struct Fill {
    unit: usize,
    at: usize,
    len: usize,
}

/// The filled slots waiting for one sink's delivery threads.
#[derive(Default)]
struct Queue {
    slots: Mutex<std::collections::VecDeque<usize>>,
    ready: Condvar,
}

/// The staging ring of one stage call: free slots, one queue per sink, and each filled slot's fill and the sinks
/// that still hold it.
struct Ring {
    free: Mutex<Vec<usize>>,
    freed: Condvar,
    queues: Vec<Queue>,
    fills: Vec<Mutex<Fill>>,
    holders: Vec<AtomicUsize>,
    readers_left: AtomicUsize,
}

impl Ring {
    fn new(slots: usize, sinks: usize, readers: usize) -> Ring {
        Ring {
            free: Mutex::new((0..slots).rev().collect()),
            freed: Condvar::new(),
            queues: (0..sinks).map(|_| Queue::default()).collect(),
            fills: (0..slots).map(|_| Mutex::new(Fill::default())).collect(),
            holders: (0..slots).map(|_| AtomicUsize::new(0)).collect(),
            readers_left: AtomicUsize::new(readers),
        }
    }

    /// A free slot, waiting while there is none; `None` once staging stops.
    fn acquire<S: ?Sized>(&self, shared: &Shared<'_, S>) -> Option<usize> {
        let mut free = self.free.lock().expect("ring");
        let t = Instant::now();
        let slot = loop {
            if shared.stop.load(Ordering::Relaxed) {
                break None;
            }
            if let Some(s) = free.pop() {
                break Some(s);
            }
            free = self.freed.wait(free).expect("ring");
        };
        shared
            .wait_free_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        slot
    }

    fn release(&self, slot: usize) {
        self.free.lock().expect("ring").push(slot);
        self.freed.notify_one();
    }

    /// Publishes a filled slot to every sink; it is free again once each of them has released it.
    fn publish(&self, slot: usize, fill: Fill) {
        *self.fills[slot].lock().expect("ring") = fill;
        self.holders[slot].store(self.queues.len(), Ordering::Release);
        for q in &self.queues {
            q.slots.lock().expect("ring").push_back(slot);
            q.ready.notify_one();
        }
    }

    /// The next filled slot for sink `k`, waiting while readers are still running; `None` when they are done and
    /// the queue is empty.
    fn next<S: ?Sized>(&self, k: usize, shared: &Shared<'_, S>) -> Option<usize> {
        let q = &self.queues[k];
        let mut slots = q.slots.lock().expect("ring");
        let t = Instant::now();
        let slot = loop {
            if let Some(s) = slots.pop_front() {
                break Some(s);
            }
            if self.readers_left.load(Ordering::Acquire) == 0 {
                break None;
            }
            slots = q.ready.wait(slots).expect("ring");
        };
        shared
            .wait_fill_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        slot
    }

    /// Sink `k` is done with `slot`; the last holder frees it (and its bytes leave the in-flight count).
    fn done<S: ?Sized>(&self, slot: usize, shared: &Shared<'_, S>) {
        if self.holders[slot].fetch_sub(1, Ordering::AcqRel) == 1 {
            let len = self.fills[slot].lock().expect("ring").len as u64;
            shared.in_flight.fetch_sub(len, Ordering::Relaxed);
            self.release(slot);
        }
    }

    /// A reader exits; the last one wakes every delivery thread so they drain their queues and stop.
    fn reader_done(&self) {
        if self.readers_left.fetch_sub(1, Ordering::AcqRel) == 1 {
            for q in &self.queues {
                let _held = q.slots.lock().expect("ring");
                q.ready.notify_all();
            }
        }
    }

    /// Staging stopped: wakes every reader waiting for a free slot.
    fn wake_readers(&self) {
        let _held = self.free.lock().expect("ring");
        self.freed.notify_all();
    }
}

/// Held by every reader and delivery thread of a stage call. Unwinding out of one stops the staging and wakes the
/// waiting readers, so the panic reaches `stage_to`'s join instead of a reader waiting forever for a buffer the
/// panicked thread held; a reader's guard also counts the reader out (on any exit), so delivery threads drain.
struct Exit<'r, 'a, S: ?Sized> {
    ring: &'r Ring,
    shared: &'r Shared<'a, S>,
    reader: bool,
}

impl<S: ?Sized> Drop for Exit<'_, '_, S> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.shared.stop.store(true, Ordering::Relaxed);
            self.ring.wake_readers();
        }
        if self.reader {
            self.ring.reader_done();
        }
    }
}

impl Artifact {
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn counters(&self) -> ByteCounts {
        ByteCounts {
            payload_bytes_read: self.counters.payload.load(Ordering::Relaxed),
            metadata_bytes_read: self.counters.metadata.load(Ordering::Relaxed),
        }
    }

    pub fn timings(&self) -> Timings {
        self.timings
    }

    /// The family rows (placement, dtype, count, bytes; group membership is the row's role).
    pub fn families(&self) -> &[FamilyRow] {
        &self.manifest.families
    }

    /// The parsed header of the weights file `path`.
    pub fn header(&self, path: &str) -> Option<&Header> {
        self.shards
            .iter()
            .find(|s| self.files[s.file].path == path)
            .map(|s| &s.header)
    }

    fn tensor_ref<'a>(&'a self, s: &'a Shard, i: usize) -> TensorRef<'a> {
        let t = &s.header.tensors[i];
        let r = &self.rules.rules[s.rule[i]];
        TensorRef {
            path: &self.files[s.file].path,
            name: &t.name,
            dtype: t.dtype,
            shape: &t.shape,
            offset: t.offset,
            bytes: t.bytes,
            element_aligned: t.element_aligned,
            family: &r.family,
            role: r.role,
            placement: &self.placements[s.rule[i]],
        }
    }

    /// Every tensor, shards in manifest order, tensors in file-offset order.
    pub fn tensors(&self) -> impl Iterator<Item = TensorRef<'_>> + '_ {
        self.shards
            .iter()
            .flat_map(move |s| (0..s.header.tensors.len()).map(move |i| self.tensor_ref(s, i)))
    }

    /// Checks the weights file `path` against its manifest sha256 without reading it (m216 design note L4):
    /// `views` are the bytes of its tensors in file-offset order, held elsewhere (an attached shared segment), and
    /// sha256(header prefix ‖ views) must equal the manifest's, as the tensors tile the data region (the header
    /// parse guarantees it). A view count or length that differs from the header is class `resident`; a digest
    /// difference is `file_hash`.
    pub fn check_file_views(&self, path: &str, views: &[&[u8]]) -> Result<(), Error> {
        let s = self
            .shards
            .iter()
            .find(|s| self.files[s.file].path == path)
            .ok_or_else(|| Error::reject("resident", format!("{path}: not a weights file")))?;
        let tensors = &s.header.tensors;
        if views.len() != tensors.len()
            || views
                .iter()
                .zip(tensors)
                .any(|(v, t)| v.len() as u64 != t.bytes)
        {
            return Err(Error::reject(
                "resident",
                format!("{path}: the resident views do not match the header's tensors"),
            ));
        }
        let mut hasher = s.prefix_hash.clone();
        for v in views {
            hasher.update(v);
        }
        let f = &self.files[s.file];
        if hex(&hasher.finalize()) != f.sha256 {
            return Err(Error::reject(
                "file_hash",
                format!("{}: sha256 differs from the manifest", f.path),
            ));
        }
        Ok(())
    }

    /// Indices (file-offset order) of the tensors of `s` in `class`.
    fn admitted<'a>(&'a self, s: &'a Shard, class: Class) -> impl Iterator<Item = usize> + 'a {
        s.rule
            .iter()
            .enumerate()
            .filter(move |(_, &r)| class.admits(&self.placements[r]))
            .map(|(i, _)| i)
    }

    /// Reads a small payload file whole through its descriptor (payload) and checks its manifest sha256.
    fn read_checked(&self, f: &ArtifactFile) -> Result<Vec<u8>, Error> {
        let mut raw = vec![0u8; f.open.stat.size as usize];
        read_at(f, &mut raw, 0)?;
        self.counters.payload(raw.len() as u64);
        if sha256_hex(&raw) != f.sha256 {
            return Err(Error::reject(
                "file_hash",
                format!("{}: sha256 differs from the manifest", f.path),
            ));
        }
        Ok(raw)
    }

    /// The draft token map (`draft/token_map.u32le`; empty when the artifact has none). [`Mode::Full`] keeps it from
    /// the hash pass; otherwise the first call reads it whole (payload) and checks its manifest sha256.
    pub fn token_map(&self) -> Result<&[u32], Error> {
        if let Some(m) = self.token_map.get() {
            return Ok(m);
        }
        let raw = match self.files.iter().find(|f| f.role == role::DRAFT_TOKEN_MAP) {
            Some(f) => self.read_checked(f)?,
            None => Vec::new(),
        };
        Ok(self.token_map.get_or_init(|| decode_u32le(&raw)))
    }

    /// Step 13 (stamp mode): the sidecar `<parent>/<artifact_id>.verified.json`, opened through the parent of the
    /// held root, must be `full_verify`, for this manifest, and record exactly `fstat` of the held descriptors.
    pub fn check_stamp(&mut self) -> Result<(), Error> {
        let t = Instant::now();
        let parent = self
            .root
            .parent()
            .map_err(|e| stamp_err(format!("cannot open the stamp directory: {e}")))?;
        let name = format!("{}{}", self.artifact_id, stamp::SUFFIX);
        let f = parent
            .open_file(&name)
            .map_err(|e| stamp_err(format!("no readable stamp {name}: {e}")))?;
        if f.stat.size > MAX_STAMP_BYTES {
            return Err(stamp_err(format!("stamp {name} is too large")));
        }
        let mut raw = vec![0u8; f.stat.size as usize];
        f.read_exact_at(&mut raw, 0)
            .map_err(|e| stamp_err(format!("reading {name}: {e}")))?;
        self.counters.metadata(raw.len() as u64);
        let s = Stamp::parse(&raw).ok_or_else(|| stamp_err(format!("{name} is malformed")))?;
        if s.stamp_source != stamp::FULL_VERIFY {
            return Err(stamp_err(format!(
                "stamp_source is {}, not {}",
                s.stamp_source,
                stamp::FULL_VERIFY
            )));
        }
        if s.manifest_sha256 != self.manifest_sha256 {
            return Err(stamp_err("the stamp is for another manifest"));
        }
        if s.files.len() != self.files.len()
            || self.files.iter().any(|f| !s.files.contains_key(&f.path))
        {
            return Err(stamp_err("the stamp's file set differs from the manifest"));
        }
        for f in &self.files {
            let fs = stamp::fs_view(&f.open.file, &f.open.stat)
                .map_err(|e| Error::io(format!("fstatfs {}", f.path), e))?;
            match s.files[&f.path].check(s.format, &f.open.stat, &fs) {
                Ok(()) => {}
                Err(Mismatch::Changed) => {
                    return Err(stamp_err(format!("{} changed since the stamp", f.path)));
                }
                Err(Mismatch::PredatesFsUuid) => {
                    return Err(Error {
                        class: "stamp",
                        exit: EXIT_STAMP,
                        message: format!(
                            "{}: st_dev differs and the stamp predates fs-uuid format; full verify required",
                            f.path
                        ),
                    });
                }
            }
        }
        if let Some(note) = s.dev_note() {
            let _ = self.stamp_note.set(note);
        }
        self.timings.validate_ms = Some(self.timings.validate_ms.unwrap_or(0) + ms(t));
        Ok(())
    }

    /// Step 13 (full mode): the sha256 of every listed file through its held descriptor (the header prefix from
    /// memory), on up to [`HASH_THREADS`] threads, with `floor` rechecked every chunk. The reported rejection is the
    /// first failing file in manifest order, as in `verify_artifact.py`: a failure stops only later files.
    pub fn hash_files(&mut self, floor: Option<&Floor>) -> Result<(), Error> {
        let t = Instant::now();
        let next = AtomicUsize::new(0);
        // Index of the earliest failed file so far (usize::MAX: none).
        let stop = AtomicUsize::new(usize::MAX);
        let first_err: Mutex<Option<(usize, Error)>> = Mutex::new(None);
        let threads = HASH_THREADS.min(self.files.len()).max(1);
        let this = &*self;
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    let mut buf = vec![0u8; HASH_CHUNK_BYTES];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= this.files.len() || i > stop.load(Ordering::Relaxed) {
                            return;
                        }
                        if let Err(e) = this.hash_one(i, &mut buf, floor, &stop) {
                            stop.fetch_min(i, Ordering::Relaxed);
                            let mut slot = first_err.lock().expect("error slot");
                            if slot.as_ref().is_none_or(|(j, _)| i < *j) {
                                *slot = Some((i, e));
                            }
                            return;
                        }
                    }
                });
            }
        });
        if let Some((_, e)) = first_err.into_inner().expect("error slot") {
            return Err(e);
        }
        self.hashed = true;
        self.timings.hash_ms = Some(ms(t));
        Ok(())
    }

    fn hash_one(
        &self,
        i: usize,
        buf: &mut [u8],
        floor: Option<&Floor>,
        stop: &AtomicUsize,
    ) -> Result<(), Error> {
        let f = &self.files[i];
        if f.role == role::DRAFT_TOKEN_MAP && f.open.stat.size <= SMALL_FILE_MAX {
            let raw = self.read_checked(f)?;
            if let Some(fl) = floor {
                fl.progress(raw.len() as u64)?;
            }
            let _ = self.token_map.set(decode_u32le(&raw));
            return Ok(());
        }
        let (mut h, mut pos) = match f.shard {
            Some(s) => (
                self.shards[s].prefix_hash.clone(),
                self.shards[s].header.header_bytes,
            ),
            None => (Sha256::new(), 0),
        };
        let size = f.open.stat.size;
        while pos < size {
            if stop.load(Ordering::Relaxed) < i {
                return Ok(());
            }
            let n = (size - pos).min(buf.len() as u64) as usize;
            read_at(f, &mut buf[..n], pos)?;
            self.counters.payload(n as u64);
            h.update(&buf[..n]);
            if let Some(fl) = floor {
                fl.progress(n as u64)?;
            }
            pos += n as u64;
        }
        if hex(&h.finalize()) != f.sha256 {
            return Err(Error::reject(
                "file_hash",
                format!("{}: sha256 differs from the manifest", f.path),
            ));
        }
        Ok(())
    }

    /// After [`Artifact::hash_files`]: writes `<parent>/<artifact_id>.verified.json` as `full_verify` from `fstat`
    /// of the held descriptors, under the root's `.cerec.lock`. A previous stamp's source fingerprint (same manifest)
    /// is kept. Returns the bytes written.
    pub fn write_full_verify_stamp(&self) -> Result<u64, Error> {
        if !self.hashed {
            return Err(stamp_err("no full hash pass ran"));
        }
        let parent_path = self
            .root_path
            .parent()
            .ok_or_else(|| Error::reject("unsafe_path", "the artifact root has no parent"))?;
        let _lock = FileLock::acquire(&parent_path.join(LOCK_FILE), false).map_err(|e| {
            Error::io(
                format!("locking {}", parent_path.join(LOCK_FILE).display()),
                format!("{e:?}"),
            )
        })?;
        let parent = self
            .root
            .parent()
            .map_err(|e| Error::io("opening the stamp directory", e))?;
        let name = format!("{}{}", self.artifact_id, stamp::SUFFIX);
        let fingerprint = parent
            .open_file(&name)
            .ok()
            .filter(|f| f.stat.size <= MAX_STAMP_BYTES)
            .and_then(|f| {
                let mut raw = vec![0u8; f.stat.size as usize];
                f.read_exact_at(&mut raw, 0).ok()?;
                self.counters.metadata(raw.len() as u64);
                Stamp::parse(&raw)
            })
            .filter(|s| s.manifest_sha256 == self.manifest_sha256)
            .map(|s| s.source_fingerprint)
            .unwrap_or_default();
        // The stamp records the identities that were hashed: a file touched during the pass gets no stamp.
        for f in &self.files {
            let now = f.open.restat().map_err(|e| Error::io(&f.path, e))?;
            if now != f.open.stat {
                return Err(stamp_err(format!(
                    "{} changed during the full verify",
                    f.path
                )));
            }
        }
        let files = self
            .files
            .iter()
            .map(|f| {
                stamp::FileId::of(&f.open)
                    .map(|id| (f.path.clone(), id))
                    .map_err(|e| Error::io(format!("fstatfs {}", f.path), e))
            })
            .collect::<Result<_, Error>>()?;
        let s = Stamp {
            format: stamp::FORMAT,
            manifest_sha256: self.manifest_sha256.clone(),
            stamp_source: stamp::FULL_VERIFY.to_string(),
            source_fingerprint: fingerprint,
            files,
            verified_utc: stamp::utc_now(),
        };
        let n = s.write(&parent, &self.artifact_id)?;
        if let Some(note) = s.dev_note() {
            let _ = self.stamp_note.set(note);
        }
        Ok(n)
    }

    /// What the stamp checked or written could not anchor on a filesystem UUID (files named by `st_dev`, or a format-1
    /// stamp accepted on a device match); `cerec verify` prints it.
    pub fn stamp_note(&self) -> Option<&str> {
        self.stamp_note.get().map(String::as_str)
    }

    /// `fsync` + `POSIX_FADV_DONTNEED` per file (Q8): every listed file, or those holding tensors of `class`.
    pub fn evict(&self, class: Option<Class>) -> Result<usize, Error> {
        let mut n = 0;
        for f in &self.files {
            let wanted = match (class, f.shard) {
                (None, _) => true,
                (Some(c), Some(s)) => self.admitted(&self.shards[s], c).next().is_some(),
                (Some(_), None) => false,
            };
            if wanted {
                fsio::evict(&f.open.file)
                    .map_err(|e| Error::io(format!("evicting {}", f.path), e))?;
                n += 1;
            }
        }
        Ok(n)
    }

    /// `mincore` residency of every listed file, or of the pages holding tensors of `class`.
    pub fn residency(&self, class: Option<Class>) -> Result<Residency, Error> {
        let mut out = Residency::default();
        let io = |f: &ArtifactFile, e| Error::io(format!("mincore {}", f.path), e);
        match class {
            None => {
                for f in &self.files {
                    out.add(fsio::residency(&f.open.file, f.open.stat.size).map_err(|e| io(f, e))?);
                }
            }
            Some(c) => {
                for s in &self.shards {
                    let f = &self.files[s.file];
                    let ranges: Vec<(u64, u64)> = self
                        .admitted(s, c)
                        .map(|i| (s.header.tensors[i].offset, s.header.tensors[i].bytes))
                        .collect();
                    if ranges.is_empty() {
                        continue;
                    }
                    out.add(
                        fsio::residency_of_ranges(&f.open.file, f.open.stat.size, ranges)
                            .map_err(|e| io(f, e))?,
                    );
                }
            }
        }
        Ok(out)
    }

    /// The staging plan of `class` for `sinks` with buffers of `step` bytes (contract "Read", m216 design note L2,
    /// L7): the tensors some sink wants, in units. A file staged with `hash` whose tensors are all in the class is
    /// one [`Unit::Shard`] (the hash needs it whole and in order); those come first, longest first. Elsewhere, in
    /// file order, consecutive wanted tensors that are contiguous in the file and together fit one buffer form one
    /// [`Unit::Run`], and a tensor larger than a buffer is cut into buffer-sized [`Unit::Piece`]s, so every read is
    /// at most one buffer and all readers share one queue of them across files. The runs and pieces whose tensors
    /// are all [`Sink::late`] to their sinks come after the others.
    fn plan<S: Sink + ?Sized>(&self, class: Class, sinks: &[&S], hash: bool, step: u64) -> Plan {
        let mut plan = Plan::default();
        let mut whole: Vec<(u64, Unit)> = Vec::new();
        let mut units = Vec::new();
        for (si, s) in self.shards.iter().enumerate() {
            let admitted: Vec<usize> = self.admitted(s, class).collect();
            if admitted.is_empty() {
                continue;
            }
            if hash && admitted.len() == s.header.tensors.len() {
                whole.push((self.files[s.file].bytes, Unit::Shard(si)));
                continue;
            }
            // The open run: its first index in `order` and its file span.
            let mut open: Option<(usize, u64, u64)> = None;
            for i in admitted {
                let t = self.tensor_ref(s, i);
                let mask = sinks
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| k.wants(&t))
                    .fold(0u64, |m, (k, _)| m | (1 << k));
                if mask == 0 {
                    continue;
                }
                let end = t.offset + t.bytes;
                match open {
                    Some((lo, start, at)) if t.offset == at && end - start <= step => {
                        open = Some((lo, start, end));
                    }
                    _ => {
                        if let Some((lo, ..)) = open.take() {
                            units.push(Unit::Run {
                                shard: si,
                                lo,
                                hi: plan.order.len(),
                            });
                        }
                        if t.bytes > step {
                            let at = plan.order.len();
                            let mut off = 0;
                            while off < t.bytes {
                                let len = (t.bytes - off).min(step);
                                units.push(Unit::Piece {
                                    shard: si,
                                    at,
                                    off,
                                    len,
                                });
                                off += len;
                            }
                        } else {
                            open = Some((plan.order.len(), t.offset, end));
                        }
                    }
                }
                plan.order.push(i);
                plan.sinks.push(mask);
            }
            if let Some((lo, ..)) = open {
                units.push(Unit::Run {
                    shard: si,
                    lo,
                    hi: plan.order.len(),
                });
            }
        }
        // load-soft design note: a unit whose every wanted tensor is late to every sink that wants it is read after
        // all the others (in the same order among themselves).
        let is_late = |u: &Unit| -> bool {
            let (shard, js) = match *u {
                Unit::Run { shard, lo, hi } => (shard, lo..hi),
                Unit::Piece { shard, at, .. } => (shard, at..at + 1),
                Unit::Shard(_) => return false,
            };
            let s = &self.shards[shard];
            js.into_iter().all(|j| {
                let t = self.tensor_ref(s, plan.order[j]);
                sinks
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| plan.sinks[j] & (1 << k) != 0)
                    .all(|(_, k)| k.late(&t))
            })
        };
        let (early, late): (Vec<Unit>, Vec<Unit>) = units.into_iter().partition(|u| !is_late(u));
        whole.sort_by(|a, b| b.0.cmp(&a.0));
        plan.units = whole
            .into_iter()
            .map(|(_, u)| u)
            .chain(early)
            .chain(late)
            .collect();
        plan
    }

    /// The shard of a run or piece and the file span `[start, start + len)` it reads.
    fn span(&self, plan: &Plan, unit: Unit) -> (usize, u64, u64) {
        match unit {
            Unit::Run { shard, lo, hi } => {
                let s = &self.shards[shard];
                let first = &s.header.tensors[plan.order[lo]];
                let last = &s.header.tensors[plan.order[hi - 1]];
                (shard, first.offset, last.offset + last.bytes - first.offset)
            }
            Unit::Piece {
                shard,
                at,
                off,
                len,
            } => (
                shard,
                self.shards[shard].header.tensors[plan.order[at]].offset + off,
                len,
            ),
            Unit::Shard(_) => unreachable!("a shard unit has no single span"),
        }
    }

    /// The `O_DIRECT` descriptor, by file index, of every file a run or piece of `plan` reads that `mode` reads
    /// direct; `None` for the rest (and for a file whose file system refuses `O_DIRECT`).
    fn direct_files(
        &self,
        plan: &Plan,
        mode: ReadMode,
    ) -> Result<Vec<Option<std::fs::File>>, Error> {
        let mut out: Vec<Option<std::fs::File>> = (0..self.files.len()).map(|_| None).collect();
        if mode == ReadMode::Buffered {
            return Ok(out);
        }
        let mut read = vec![false; self.files.len()];
        for u in &plan.units {
            if let Unit::Run { shard, .. } | Unit::Piece { shard, .. } = *u {
                read[self.shards[shard].file] = true;
            }
        }
        for (i, f) in self.files.iter().enumerate() {
            if !read[i] {
                continue;
            }
            if mode == ReadMode::Auto {
                let r = fsio::residency(&f.open.file, f.open.stat.size)
                    .map_err(|e| Error::io(format!("mincore {}", f.path), e))?;
                if r.resident_pages * 2 >= r.total_pages {
                    continue;
                }
            }
            out[i] = f
                .open
                .reopen_direct()
                .map_err(|e| Error::io(format!("O_DIRECT open {}", f.path), e))?;
        }
        Ok(out)
    }

    /// Streams every tensor of `class` that `sink` wants exactly once through `buffers` into `sink` (contract
    /// "Read"): [`Artifact::stage_to`] with one sink.
    pub fn stage<B, S>(
        &self,
        class: Class,
        buffers: &mut [B],
        sink: &S,
        options: Option<&StageOptions<'_>>,
    ) -> Result<StageReport, Error>
    where
        B: AsMut<[u8]> + Send,
        S: Sink + ?Sized,
    {
        self.stage_to(class, buffers, &[sink], options)
    }

    /// Streams every tensor of `class` that some sink wants exactly once through `buffers` (contract "Read",
    /// m216 design note L7): reader threads take a free buffer, then the next unit of [`Artifact::plan`] from one
    /// queue shared across files, read it ([`ReadMode`]) and publish the buffer to every sink; each sink's delivery
    /// threads hand the pieces of the tensors it wants to it, and the buffer is free again only after every sink has
    /// released it (its `accept` returned). A tensor arrives as pieces of at most one buffer covering
    /// `[k·len, (k+1)·len)` of it (`len` the plan's read size), from any delivery thread in any order, the one that
    /// ends the tensor with `is_last` (a 0-byte tensor is one empty piece); the tensors of one run share one read.
    /// Reads go only through the held descriptors (a direct read through a second descriptor of the same inode).
    pub fn stage_to<B, S>(
        &self,
        class: Class,
        buffers: &mut [B],
        sinks: &[&S],
        options: Option<&StageOptions<'_>>,
    ) -> Result<StageReport, Error>
    where
        B: AsMut<[u8]> + Send,
        S: Sink + ?Sized,
    {
        let t = Instant::now();
        if buffers.is_empty() {
            return Err(buffer_err("at least one staging buffer is needed"));
        }
        if sinks.is_empty() || sinks.len() > MAX_SINKS {
            return Err(buffer_err(format!(
                "staging feeds 1 to {MAX_SINKS} sinks, got {}",
                sinks.len()
            )));
        }
        let mut step = usize::MAX;
        let mut slots = Vec::with_capacity(buffers.len());
        for b in buffers.iter_mut() {
            let s = b.as_mut();
            if s.as_ptr() as usize % BUFFER_ALIGN != 0 {
                return Err(buffer_err(format!(
                    "staging buffers must be {BUFFER_ALIGN}-byte aligned"
                )));
            }
            check_buffer_len(s.len())?;
            step = step.min(s.len());
            slots.push((s.as_mut_ptr(), s.len()));
        }
        let opts = options.copied().unwrap_or_default();
        let read_step = match opts.read {
            ReadMode::Buffered => step as u64,
            ReadMode::Direct | ReadMode::Auto => step as u64 - 2 * DIRECT_ALIGN,
        };
        let plan = self.plan(class, sinks, opts.hash, read_step);
        let direct = self.direct_files(&plan, opts.read)?;
        let asked = if opts.readers == 0 {
            slots.len()
        } else {
            opts.readers
        };
        let readers = asked.min(plan.units.len());
        let consumers = match (readers, opts.consumers) {
            (0, _) => 0,
            (_, 0) => slots.len().min(MAX_DEFAULT_CONSUMERS),
            (_, c) => c,
        };
        let shared = Shared {
            class,
            sinks,
            opts,
            stop: AtomicBool::new(false),
            in_flight: AtomicU64::new(0),
            high_water: AtomicU64::new(0),
            hash_ns: AtomicU64::new(0),
            pread_ns: AtomicU64::new(0),
            wait_free_ns: AtomicU64::new(0),
            wait_fill_ns: AtomicU64::new(0),
            deliver_ns: AtomicU64::new(0),
            tensors: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            hashed_files: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            reads_now: AtomicU64::new(0),
            reads_max: AtomicU64::new(0),
            first_err: Mutex::new(None),
        };
        let ring = Ring::new(slots.len(), sinks.len(), readers);
        let slots = Slots(slots);
        let next = AtomicUsize::new(0);
        let sampling = (Mutex::new(false), Condvar::new());
        let samples = std::thread::scope(|sc| {
            let (shared, ring, slots, plan, next, direct) =
                (&shared, &ring, &slots, &plan, &next, &direct);
            let sampler = sc.spawn(|| sample(shared, t, &sampling));
            let mut threads = Vec::with_capacity(readers + consumers * sinks.len());
            for _ in 0..readers {
                threads.push(
                    sc.spawn(move || self.reader(plan, slots, ring, next, direct, step, shared)),
                );
            }
            for k in 0..sinks.len() {
                for _ in 0..consumers {
                    threads.push(sc.spawn(move || self.consumer(k, plan, slots, ring, shared)));
                }
            }
            let panicked = threads.into_iter().filter_map(|h| h.join().err()).next();
            *sampling.0.lock().expect("sampler") = true;
            sampling.1.notify_all();
            let samples = sampler.join().expect("the sampler never panics");
            if let Some(p) = panicked {
                std::panic::resume_unwind(p);
            }
            samples
        });
        if let Some(e) = shared.first_err.into_inner().expect("error slot") {
            return Err(e);
        }
        let (reads_in_flight_mean_x100, timeline) = timeline(&samples);
        let ns_ms = |a: AtomicU64| a.into_inner() / 1_000_000;
        Ok(StageReport {
            tensors: shared.tensors.into_inner(),
            bytes: shared.bytes.into_inner(),
            high_water_bytes: shared.high_water.into_inner(),
            threads: readers,
            consumers,
            sinks: sinks.len(),
            reads: shared.reads.into_inner(),
            direct_files: direct.iter().filter(|d| d.is_some()).count() as u64,
            read_ms: ms(t),
            pread_ms: ns_ms(shared.pread_ns),
            wait_free_ms: ns_ms(shared.wait_free_ns),
            wait_fill_ms: ns_ms(shared.wait_fill_ns),
            deliver_ms: ns_ms(shared.deliver_ns),
            hash_ms: ns_ms(shared.hash_ns),
            hashed_files: shared.hashed_files.into_inner(),
            reads_in_flight_mean_x100,
            reads_in_flight_max: shared.reads_max.into_inner(),
            timeline,
        })
    }

    /// One reader thread: free buffer, next unit, read, publish (a shard unit is read, hashed and delivered here).
    #[allow(clippy::too_many_arguments)]
    fn reader<S: Sink + ?Sized>(
        &self,
        plan: &Plan,
        slots: &Slots,
        ring: &Ring,
        next: &AtomicUsize,
        direct: &[Option<std::fs::File>],
        step: usize,
        shared: &Shared<'_, S>,
    ) {
        let _exit = Exit {
            ring,
            shared,
            reader: true,
        };
        while let Some(slot) = ring.acquire(shared) {
            let u = next.fetch_add(1, Ordering::Relaxed);
            let Some(&unit) = plan.units.get(u) else {
                ring.release(slot);
                break;
            };
            let (ptr, len) = slots.0[slot];
            // SAFETY: a slot taken from the free list is this reader's alone until it publishes or releases it.
            let buf = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            let done = match unit {
                Unit::Shard(si) => {
                    let r = self.stage_shard(&self.shards[si], &mut buf[..step], shared);
                    ring.release(slot);
                    r
                }
                _ => match self.read_unit(plan, unit, buf, direct, shared) {
                    Ok((at, len)) => {
                        ring.publish(slot, Fill { unit: u, at, len });
                        Ok(())
                    }
                    Err(e) => {
                        ring.release(slot);
                        Err(e)
                    }
                },
            };
            if let Err(e) = done {
                shared.fail(e);
                ring.wake_readers();
                break;
            }
        }
    }

    /// One delivery thread of sink `k`: every filled buffer published to it, delivered (unless staging stopped) and
    /// released.
    fn consumer<S: Sink + ?Sized>(
        &self,
        k: usize,
        plan: &Plan,
        slots: &Slots,
        ring: &Ring,
        shared: &Shared<'_, S>,
    ) {
        let _exit = Exit {
            ring,
            shared,
            reader: false,
        };
        while let Some(slot) = ring.next(k, shared) {
            if !shared.stop.load(Ordering::Relaxed) {
                let fill = *ring.fills[slot].lock().expect("ring");
                let (ptr, _) = slots.0[slot];
                // SAFETY: the fill was published under the queue mutex; the slot stays read-only until every sink
                // has released it.
                let data = unsafe { std::slice::from_raw_parts(ptr.add(fill.at), fill.len) };
                let t = Instant::now();
                let r = self.deliver_unit(plan, plan.units[fill.unit], data, k, shared.sinks[k]);
                shared
                    .deliver_ns
                    .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if let Err(e) = r {
                    shared.fail(e);
                    ring.wake_readers();
                }
            }
            ring.done(slot, shared);
        }
    }

    /// Hands the pieces of a run or piece in `data` (its span) that sink `k` wants to it.
    fn deliver_unit<S: Sink + ?Sized>(
        &self,
        plan: &Plan,
        unit: Unit,
        data: &[u8],
        k: usize,
        sink: &S,
    ) -> Result<(), Error> {
        let bit = 1u64 << k;
        match unit {
            Unit::Run { shard, lo, hi } => {
                let s = &self.shards[shard];
                let start = s.header.tensors[plan.order[lo]].offset;
                for j in lo..hi {
                    if plan.sinks[j] & bit == 0 {
                        continue;
                    }
                    let t = self.tensor_ref(s, plan.order[j]);
                    let at = (t.offset - start) as usize;
                    accept(sink, &t, 0, &data[at..at + t.bytes as usize], true)?;
                }
            }
            Unit::Piece {
                shard,
                at,
                off,
                len,
            } => {
                if plan.sinks[at] & bit != 0 {
                    let t = self.tensor_ref(&self.shards[shard], plan.order[at]);
                    accept(sink, &t, off, data, off + len == t.bytes)?;
                }
            }
            Unit::Shard(_) => unreachable!("a shard unit is delivered by its reader"),
        }
        Ok(())
    }

    /// One counted tensor-data read of `len` bytes (`read` returns where they start in the buffer): the reads in
    /// flight, the read time, payload and staged bytes, and the floor.
    fn timed_read<S: Sink + ?Sized>(
        &self,
        len: u64,
        shared: &Shared<'_, S>,
        read: impl FnOnce() -> Result<usize, Error>,
    ) -> Result<usize, Error> {
        let now = shared.reads_now.fetch_add(1, Ordering::Relaxed) + 1;
        shared.reads_max.fetch_max(now, Ordering::Relaxed);
        let t = Instant::now();
        let r = read();
        shared
            .pread_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        shared.reads_now.fetch_sub(1, Ordering::Relaxed);
        let at = r?;
        self.counters.payload(len);
        shared.reads.fetch_add(1, Ordering::Relaxed);
        shared.bytes.fetch_add(len, Ordering::Relaxed);
        if let Some(fl) = shared.opts.floor {
            fl.progress(len)?;
        }
        Ok(at)
    }

    /// Reads a run or piece into `buf` (direct or buffered); returns where its span starts in `buf` and its length.
    /// The span's bytes count as in flight until the last sink releases the buffer.
    fn read_unit<S: Sink + ?Sized>(
        &self,
        plan: &Plan,
        unit: Unit,
        buf: &mut [u8],
        direct: &[Option<std::fs::File>],
        shared: &Shared<'_, S>,
    ) -> Result<(usize, usize), Error> {
        let (shard, start, len) = self.span(plan, unit);
        let file = self.shards[shard].file;
        let f = &self.files[file];
        let at = self.timed_read(len, shared, || match &direct[file] {
            Some(d) => read_direct(f, d, buf, start, len),
            None => read_at(f, &mut buf[..len as usize], start).map(|()| 0),
        })?;
        let tensors = match unit {
            Unit::Run { lo, hi, .. } => (hi - lo) as u64,
            Unit::Piece { off, .. } => u64::from(off == 0),
            Unit::Shard(_) => 0,
        };
        shared.tensors.fetch_add(tensors, Ordering::Relaxed);
        let held = shared.in_flight.fetch_add(len, Ordering::Relaxed) + len;
        shared.high_water.fetch_max(held, Ordering::Relaxed);
        Ok((at, len as usize))
    }

    /// A [`Unit::Shard`]: every tensor of a hashed file in file-offset order, read whole or in buffer-sized pieces,
    /// hashed, and delivered to each sink that wants it; the pieces of tensors no sink wants are hashed only.
    fn stage_shard<S: Sink + ?Sized>(
        &self,
        s: &Shard,
        buf: &mut [u8],
        shared: &Shared<'_, S>,
    ) -> Result<(), Error> {
        let f = &self.files[s.file];
        let mut hasher = s.prefix_hash.clone();
        let step = buf.len() as u64;
        for i in self.admitted(s, shared.class) {
            let t = self.tensor_ref(s, i);
            let wanted: Vec<&S> = shared
                .sinks
                .iter()
                .copied()
                .filter(|k| k.wants(&t))
                .collect();
            let mut off = 0u64;
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let piece = &mut buf[..(t.bytes - off).min(step) as usize];
                let n = piece.len() as u64;
                self.timed_read(n, shared, || read_at(f, piece, t.offset + off).map(|()| 0))?;
                let piece = &*piece;
                let th = Instant::now();
                hasher.update(piece);
                shared
                    .hash_ns
                    .fetch_add(th.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let last = off + n == t.bytes;
                let held = shared.in_flight.fetch_add(n, Ordering::Relaxed) + n;
                shared.high_water.fetch_max(held, Ordering::Relaxed);
                let td = Instant::now();
                let delivered = wanted
                    .iter()
                    .try_for_each(|k| accept(*k, &t, off, piece, last));
                shared
                    .deliver_ns
                    .fetch_add(td.elapsed().as_nanos() as u64, Ordering::Relaxed);
                shared.in_flight.fetch_sub(n, Ordering::Relaxed);
                delivered?;
                off += n;
                if last {
                    break;
                }
            }
            if !wanted.is_empty() {
                shared.tensors.fetch_add(1, Ordering::Relaxed);
            }
        }
        if hex(&hasher.finalize()) != f.sha256 {
            return Err(Error::reject(
                "file_hash",
                format!("{}: sha256 differs from the manifest", f.path),
            ));
        }
        shared.hashed_files.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(resident_pages: u64, total_pages: u64) -> Residency {
        Residency {
            resident_pages,
            total_pages,
        }
    }

    #[test]
    fn cold_label_needs_at_most_one_percent_resident() {
        assert_eq!(cache_label(Cache::Cold, res(0, 1000)), "cold");
        assert_eq!(cache_label(Cache::Cold, res(10, 1000)), "cold");
        assert_eq!(cache_label(Cache::Cold, res(11, 1000)), "cold_unverified");
        assert_eq!(cache_label(Cache::Cold, res(1000, 1000)), "cold_unverified");
        assert_eq!(cache_label(Cache::Cold, res(0, 0)), "cold");
    }

    #[test]
    fn warm_label_needs_at_least_95_percent_resident() {
        assert_eq!(cache_label(Cache::Warm, res(1000, 1000)), "warm");
        assert_eq!(cache_label(Cache::Warm, res(950, 1000)), "warm");
        assert_eq!(cache_label(Cache::Warm, res(949, 1000)), "partial_warm");
        assert_eq!(cache_label(Cache::Warm, res(0, 1000)), "partial_warm");
        assert_eq!(cache_label(Cache::Warm, res(0, 0)), "partial_warm");
        // No float rounding: 94.999 % is not warm.
        assert_eq!(
            cache_label(Cache::Warm, res(94_999, 100_000)),
            "partial_warm"
        );
    }

    #[test]
    fn buffers_are_aligned_and_sized() {
        let mut b = buffers(2, MIN_BUFFER_BYTES).unwrap();
        for x in &mut b {
            assert_eq!(x.as_mut_slice().as_ptr() as usize % BUFFER_ALIGN, 0);
            assert_eq!(x.len(), MIN_BUFFER_BYTES);
            assert!(x.as_mut_slice().iter().all(|&v| v == 0));
        }
        assert_eq!(buffers(0, MIN_BUFFER_BYTES).unwrap_err().class, "buffer");
        assert_eq!(
            buffers(1, MIN_BUFFER_BYTES - BUFFER_ALIGN)
                .unwrap_err()
                .class,
            "buffer"
        );
        assert_eq!(
            buffers(1, MIN_BUFFER_BYTES + 1).unwrap_err().class,
            "buffer"
        );
    }
}
