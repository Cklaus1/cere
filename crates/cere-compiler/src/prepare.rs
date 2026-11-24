//! `cerec prepare` (m21 design note R3; Q1-Q3, Q7-Q10): a layout_only copy of the source payload plus the converted
//! token map, indexed by a canonical v2 manifest, published atomically under `ROOT/<NAME>-<manifest_sha256[:16]>`.
//!
//! Under the flock `ROOT/.cerec.lock` the Q7 decision runs first: a stamp whose source fingerprint matches gives the
//! file hashes without reading payload (`reused` / `relabelled`); an artifact with the same `(path, bytes)` file set
//! gets a hash-only pass over the source (`reused_after_rehash`); otherwise one copy+hash pass (`prepared`).
use crate::families::{classify, Rules, Tally};
use crate::fsio::{self, Dir, FileLock, LockError, OpenFile};
use crate::inventory::load_rules;
use crate::manifest::{
    self, families_and_totals, identity_hashes, FileEntry, GitUpstream, LfsUpstream, Manifest,
    Profile, ShardInfo, SourceRef, Transform, Upstream,
};
use crate::safetensors::{self, Header};
use crate::source::{self, role, Select, Source, SourceFile, SMALL_FILE_MAX};
use crate::stamp::{self, Stamp};
use crate::{canon_json, hex, sha256_hex, token_map, Error, Limits};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Exit status when `--no-wait` finds the lock held.
pub const EXIT_BUSY: i32 = 3;
/// Exit status when `MemAvailable` is below the floor.
pub const EXIT_MEMORY: i32 = 5;
/// Staging buffer per thread (Q9).
pub const BUFFER_BYTES: u64 = 64 << 20;
/// Write-back, eviction and memory-floor interval (Q8, Q9); `CEREC_TEST_CHUNK_BYTES` overrides it in tests.
pub const CHUNK_BYTES: u64 = 1 << 30;
pub const LOCK_FILE: &str = ".cerec.lock";
const GIB: u64 = 1 << 30;

#[derive(Debug, Clone)]
pub struct Options {
    pub src: PathBuf,
    /// The draft token map: required for a `speculative: "nextn"` profile, refused for `"none"` (SelfTune ST4 S3b).
    pub token_map: Option<PathBuf>,
    pub name: String,
    pub lock: Option<Vec<u8>>,
    pub families: Vec<u8>,
    pub profile: Vec<u8>,
    pub out: PathBuf,
    pub no_wait: bool,
    pub threads: usize,
    pub min_mem_available_gib: u64,
    pub limits: Limits,
}

/// The byte counters of Q7, shared by worker threads.
#[derive(Debug, Default)]
pub struct Counters {
    pub payload_bytes_read: AtomicU64,
    pub metadata_bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,
}

impl Counters {
    fn payload(&self, n: u64) {
        self.payload_bytes_read.fetch_add(n, Ordering::Relaxed);
    }
    fn metadata(&self, n: u64) {
        self.metadata_bytes_read.fetch_add(n, Ordering::Relaxed);
    }
    fn written(&self, n: u64) {
        self.bytes_written.fetch_add(n, Ordering::Relaxed);
    }
}

/// The memory floor `MemAvailable ≥ base + staging` (ground rules), checked at start and every chunk of I/O.
#[derive(Debug)]
pub struct Floor {
    pub bytes: u64,
    chunk: u64,
    done: AtomicU64,
}

impl Floor {
    pub fn new(base_gib: u64, staging: u64) -> Floor {
        let chunk = std::env::var("CEREC_TEST_CHUNK_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&c: &u64| c > 0)
            .unwrap_or(CHUNK_BYTES);
        Floor {
            bytes: base_gib.saturating_mul(GIB).saturating_add(staging),
            chunk,
            done: AtomicU64::new(0),
        }
    }

    pub fn chunk(&self) -> u64 {
        self.chunk
    }

    /// Checks `MemAvailable` now; the available bytes or exit 5.
    pub fn check(&self) -> Result<u64, Error> {
        let avail = fsio::mem_available().map_err(|e| Error::io("reading MemAvailable", e))?;
        if avail < self.bytes {
            return Err(Error {
                class: "memory_floor",
                exit: EXIT_MEMORY,
                message: format!("MemAvailable {avail} B is below the floor {} B", self.bytes),
            });
        }
        Ok(avail)
    }

    /// Records `n` bytes of payload I/O and rechecks whenever a chunk boundary is crossed.
    pub fn progress(&self, n: u64) -> Result<(), Error> {
        let before = self.done.fetch_add(n, Ordering::Relaxed);
        if (before + n) / self.chunk > before / self.chunk {
            self.check()?;
        }
        Ok(())
    }
}

/// The test hook `CEREC_TEST_ABORT_AFTER_FILES=N`: the process dies after N copied files, leaving its temp directory.
fn abort_after_files() -> Option<usize> {
    std::env::var("CEREC_TEST_ABORT_AFTER_FILES")
        .ok()?
        .parse()
        .ok()
}

fn reject(class: &'static str, msg: impl Into<String>) -> Error {
    Error::reject(class, msg)
}

/// `realpath` of a path that may not exist yet: the longest existing ancestor resolved, the rest appended.
fn resolve_lenient(p: &Path) -> Result<PathBuf, Error> {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| Error::io("cwd", e))?
            .join(p)
    };
    let mut rest = Vec::new();
    let mut cur = abs.as_path();
    loop {
        match std::fs::canonicalize(cur) {
            Ok(real) => {
                let mut out = real;
                for c in rest.iter().rev() {
                    out.push(c);
                }
                return Ok(out);
            }
            Err(_) => {
                let name = cur.file_name().ok_or_else(|| {
                    reject("unsafe_out", format!("cannot resolve {}", p.display()))
                })?;
                if name == ".." {
                    return Err(reject(
                        "unsafe_out",
                        format!("{}: '..' below a missing directory", p.display()),
                    ));
                }
                rest.push(name.to_owned());
                cur = cur.parent().ok_or_else(|| {
                    reject("unsafe_out", format!("cannot resolve {}", p.display()))
                })?;
            }
        }
    }
}

/// `--out` equal to, inside or containing SRC exits 2 (compared after `realpath`).
pub fn check_out(src_real: &Path, out_real: &Path) -> Result<(), Error> {
    if out_real.starts_with(src_real) || src_real.starts_with(out_real) {
        return Err(reject(
            "unsafe_out",
            format!(
                "--out {} overlaps the source {}",
                out_real.display(),
                src_real.display()
            ),
        ));
    }
    Ok(())
}

fn check_name(name: &str) -> Result<(), Error> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));
    if !ok {
        return Err(reject(
            "usage",
            format!("--name {name:?} must be [A-Za-z0-9._-]+ not starting with '.'"),
        ));
    }
    Ok(())
}

/// The lock-definition hashes a `--lock` file pins (docs/CONFIGURATION.md:196-205).
fn lock_hashes(bytes: &[u8]) -> Result<(String, String, String), Error> {
    let v = canon_json::parse_strict(bytes).map_err(|e| reject("lock", e.to_string()))?;
    let get = |k: &str| {
        v.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| reject("lock", format!("lock has no string {k}")))
    };
    Ok((
        get("model_content_sha256")?,
        get("tokenizer_sha256")?,
        get("template_sha256")?,
    ))
}

/// The token map source: path, held descriptor, and its converted output once read.
struct TokenSource {
    path: PathBuf,
    open: OpenFile,
}

fn open_token_map(path: &Path) -> Result<TokenSource, Error> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| reject("token_map", format!("bad --token-map {}", path.display())))?;
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let parent = std::fs::canonicalize(parent)
        .map_err(|e| reject("token_map", format!("{}: {e}", path.display())))?;
    let dir = Dir::open(&parent)
        .map_err(|e| reject("token_map", format!("{}: {e}", parent.display())))?;
    let open = dir.open_file(name)?;
    Ok(TokenSource {
        path: parent.join(name),
        open,
    })
}

/// The converted token map as recorded in a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenOut {
    source_sha256: String,
    source_bytes: u64,
    out_sha256: String,
    out_bytes: u64,
}

impl TokenOut {
    fn from_manifest(m: &Manifest) -> Option<TokenOut> {
        let [t] = m.transforms.as_slice() else {
            return None;
        };
        let f = m.file(&t.output)?;
        Some(TokenOut {
            source_sha256: t.source_sha256.clone(),
            source_bytes: t.source_bytes,
            out_sha256: f.sha256.clone(),
            out_bytes: f.bytes,
        })
    }
}

/// Q7 source fingerprint: lines `path size mtime_ns header_sha256` (bytewise path order) for the selected source files,
/// then, when there is a token map, its line with its sha256. Stat-only: the token map sha comes from the candidate
/// manifest.
struct Fingerprint {
    lines: String,
    token_line: Option<String>,
}

impl Fingerprint {
    fn new(
        src: &Source,
        headers: &BTreeMap<String, Header>,
        token: Option<&TokenSource>,
    ) -> Fingerprint {
        let mut lines = String::new();
        for f in &src.files {
            let h = headers
                .get(&f.name)
                .map_or("-", |h| h.header_sha256.as_str());
            let path = src.path.join(&f.name);
            lines.push_str(&format!(
                "{} {} {} {h}\n",
                path.display(),
                f.open.stat.size,
                f.open.stat.mtime_ns
            ));
        }
        let token_line = token.map(|t| {
            format!(
                "{} {} {}",
                t.path.display(),
                t.open.stat.size,
                t.open.stat.mtime_ns
            )
        });
        Fingerprint { lines, token_line }
    }

    /// The fingerprint of a manifest whose token map transform (if any) has this source sha256; `None` when the
    /// token map presence differs from this run's.
    fn with_token(&self, token: Option<&TokenOut>) -> Option<String> {
        match (&self.token_line, token) {
            (Some(line), Some(t)) => Some(sha256_hex(
                format!("{}{line} {}\n", self.lines, t.source_sha256).as_bytes(),
            )),
            (None, None) => Some(sha256_hex(self.lines.as_bytes())),
            _ => None,
        }
    }
}

/// An existing artifact under ROOT that is a reuse candidate.
struct Candidate {
    id: String,
    dir: Dir,
    manifest: Manifest,
    /// The stamp, if present and its file identities match the held descriptors.
    stamp: Option<Stamp>,
}

fn read_candidates(
    root: &Dir,
    root_path: &Path,
    name: &str,
    limits: &Limits,
    counters: &Counters,
) -> Vec<Candidate> {
    let Ok(mut entries) = root.entries() else {
        return Vec::new();
    };
    entries.sort();
    let mut out = Vec::new();
    for (entry, kind) in entries {
        let is_id = entry
            .strip_prefix(name)
            .and_then(|r| r.strip_prefix('-'))
            .is_some_and(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()));
        if kind != fsio::EntryKind::Dir || !is_id {
            continue;
        }
        let Ok(dir) = root.open_dir(&entry) else {
            continue;
        };
        let Ok(mf) = dir.open_file(manifest::MANIFEST_FILE) else {
            continue;
        };
        if mf.stat.size > limits.max_manifest_bytes {
            continue;
        }
        let Ok(bytes) = source::read_all(&mf, manifest::MANIFEST_FILE) else {
            continue;
        };
        counters.metadata(bytes.len() as u64);
        let Ok((m, _, _)) = manifest::parse(&bytes) else {
            continue;
        };
        let sha = sha256_hex(&bytes);
        let stamp =
            Stamp::read(&root_path.join(format!("{entry}{}", stamp::SUFFIX))).and_then(|(s, n)| {
                counters.metadata(n);
                let stats =
                    stamp::stat_files(&dir, m.files.iter().map(|f| f.path.as_str())).ok()?;
                s.matches(&sha, &stats).then_some(s)
            });
        out.push(Candidate {
            id: entry,
            dir,
            manifest: m,
            stamp,
        });
    }
    out
}

/// Upstream provenance of one file (Q3) and the local-modification report entry, if any.
fn upstream_of(
    src: &Source,
    name: &str,
    bytes: u64,
    sha: &str,
    up: &mut UpstreamReport,
) -> Result<Option<Upstream>, Error> {
    let Some(entry) = src.tree.as_ref().and_then(|t| t.files.get(name)) else {
        up.not_in_tree += 1;
        return Ok(None);
    };
    if let Some(lfs) = &entry.lfs_sha256 {
        up.lfs_files += 1;
        if lfs != sha || entry.size != bytes {
            return Err(reject(
                "upstream_lfs",
                format!("{name}: sha256 {sha} differs from the upstream LFS sha256 {lfs}"),
            ));
        }
        up.lfs_match += 1;
        return Ok(Some(Upstream::Lfs(LfsUpstream {
            lfs_sha256: lfs.clone(),
        })));
    }
    up.non_lfs_files += 1;
    let modified = entry.size != bytes;
    if modified {
        up.local_modification.push(json!({
            "path": name, "upstream_size": entry.size, "upstream_blob_id": entry.blob_id, "local_sha256": sha,
        }));
    }
    Ok(Some(Upstream::Git(GitUpstream {
        size: entry.size,
        blob_id: entry.blob_id.clone(),
        local_modification: modified,
    })))
}

#[derive(Debug, Default)]
struct UpstreamReport {
    lfs_files: u64,
    lfs_match: u64,
    non_lfs_files: u64,
    not_in_tree: u64,
    local_modification: Vec<Value>,
}

/// Everything the manifest needs besides file hashes.
struct Inputs<'a> {
    src: &'a Source,
    headers: &'a BTreeMap<String, Header>,
    rules: &'a Rules,
    profile: &'a Profile,
    tallies: &'a [Tally],
}

/// Builds the manifest from known hashes (no payload read) and runs the upstream and `--lock` checks.
fn build_manifest(
    inp: &Inputs,
    hashes: &BTreeMap<String, String>,
    token: Option<&TokenOut>,
    lock: Option<&(String, String, String)>,
) -> Result<(Manifest, UpstreamReport), Error> {
    let tree = inp
        .src
        .tree
        .as_ref()
        .ok_or_else(|| reject("source_tree", "no .cache/huggingface/trees/*.json"))?;
    let mut up = UpstreamReport::default();
    let mut files = Vec::with_capacity(inp.src.files.len() + 1);
    for f in &inp.src.files {
        let sha = hashes
            .get(&f.name)
            .ok_or_else(|| Error::io(&f.name, "no hash computed"))?;
        let bytes = f.open.stat.size;
        files.push(FileEntry {
            path: f.name.clone(),
            bytes,
            sha256: sha.clone(),
            role: f.role.to_string(),
            upstream: upstream_of(inp.src, &f.name, bytes, sha, &mut up)?,
            safetensors: inp.headers.get(&f.name).map(|h| ShardInfo {
                header_bytes: h.header_bytes,
                header_sha256: h.header_sha256.clone(),
                tensors: h.tensors.len() as u64,
            }),
        });
    }
    if let Some(token) = token {
        files.push(FileEntry {
            path: token_map::OUTPUT_PATH.to_string(),
            bytes: token.out_bytes,
            sha256: token.out_sha256.clone(),
            role: role::DRAFT_TOKEN_MAP.to_string(),
            upstream: None,
            safetensors: None,
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let (model_hash, tokenizer_hash, template_hash) = identity_hashes(&files);
    if let Some((m, t, p)) = lock {
        for (what, ours, theirs) in [
            ("model_content_sha256", &model_hash, m),
            ("tokenizer_sha256", &tokenizer_hash, t),
            ("template_sha256", &template_hash, p),
        ] {
            if ours != theirs {
                return Err(reject(
                    "lock_mismatch",
                    format!("{what}: artifact {ours}, lock {theirs}"),
                ));
            }
        }
    }
    let (families, totals) = families_and_totals(inp.rules, inp.profile, inp.tallies)?;
    let m = Manifest {
        schema_version: manifest::SCHEMA_VERSION.to_string(),
        tensor_layout_abi: manifest::TENSOR_LAYOUT_ABI,
        quality_mode: manifest::QUALITY_MODE.to_string(),
        source: SourceRef {
            revision: tree.revision.clone(),
            tree_sha256: tree.sha256.clone(),
        },
        profile_id: inp.profile.profile_id.clone(),
        profile: serde_json::to_value(inp.profile).expect("profile serializes"),
        family_rules: inp.rules.value.clone(),
        model_hash,
        tokenizer_hash,
        template_hash,
        transforms: token
            .map(|t| Transform {
                name: token_map::TRANSFORM.to_string(),
                output: token_map::OUTPUT_PATH.to_string(),
                source_sha256: t.source_sha256.clone(),
                source_bytes: t.source_bytes,
            })
            .into_iter()
            .collect(),
        files,
        families,
        totals,
    };
    Ok((m, up))
}

/// Content identity: everything except the rules- and profile-derived fields (a relabel changes only those).
fn same_content(a: &Manifest, b: &Manifest) -> bool {
    (
        &a.source,
        &a.files,
        &a.transforms,
        &a.model_hash,
        &a.tokenizer_hash,
        &a.template_hash,
    ) == (
        &b.source,
        &b.files,
        &b.transforms,
        &b.model_hash,
        &b.tokenizer_hash,
        &b.template_hash,
    )
}

/// One worker pass over the source files: hash (and optionally copy) each, with chunked write-back and floor checks.
struct Pass<'a> {
    src: &'a Source,
    prefixes: &'a BTreeMap<String, Vec<u8>>,
    /// `Some(temp dir)` copies; `None` only hashes.
    dest: Option<&'a Dir>,
    counters: &'a Counters,
    floor: &'a Floor,
    threads: usize,
    abort_after: Option<usize>,
}

struct PassOut {
    hashes: BTreeMap<String, String>,
    /// Whole small files kept in memory (`config.json` gives the token map's vocabulary).
    small: BTreeMap<String, Vec<u8>>,
    staging_bytes: u64,
}

/// One file's sha256 and, for small files, its bytes.
type FileResult = (String, Option<Vec<u8>>);

impl Pass<'_> {
    fn run(&self) -> Result<PassOut, Error> {
        let next = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let first_err: Mutex<Option<Error>> = Mutex::new(None);
        let results: Mutex<BTreeMap<String, FileResult>> = Mutex::new(BTreeMap::new());
        let largest = self
            .src
            .files
            .iter()
            .map(|f| f.open.stat.size)
            .max()
            .unwrap_or(0);
        let buf_len = BUFFER_BYTES.min(largest.max(1)) as usize;
        let threads = self.threads.clamp(1, self.src.files.len().max(1));
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut buf = vec![0u8; buf_len];
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(f) = self.src.files.get(i) else {
                            return;
                        };
                        match self.one(f, &mut buf) {
                            Ok(r) => {
                                results.lock().expect("results").insert(f.name.clone(), r);
                                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                                if self.dest.is_some() && self.abort_after == Some(n) {
                                    eprintln!("io: CEREC_TEST_ABORT_AFTER_FILES={n} reached");
                                    std::process::exit(1);
                                }
                            }
                            Err(e) => {
                                stop.store(true, Ordering::Relaxed);
                                first_err.lock().expect("error slot").get_or_insert(e);
                                return;
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = first_err.into_inner().expect("error slot") {
            return Err(e);
        }
        let mut hashes = BTreeMap::new();
        let mut small = BTreeMap::new();
        for (name, (sha, kept)) in results.into_inner().expect("results") {
            if let Some(bytes) = kept {
                small.insert(name.clone(), bytes);
            }
            hashes.insert(name, sha);
        }
        Ok(PassOut {
            hashes,
            small,
            staging_bytes: (threads * buf_len) as u64,
        })
    }

    /// Hashes (and copies) one file; small non-shard files are read once whole and kept.
    fn one(&self, f: &SourceFile, buf: &mut [u8]) -> Result<(String, Option<Vec<u8>>), Error> {
        let io = |e: std::io::Error| Error::io(&f.name, e);
        let size = f.open.stat.size;
        let _ = fsio::advise_source(&f.open.file);
        let mut out = match self.dest {
            Some(d) => Some(d.create_file(&f.name).map_err(io)?),
            None => None,
        };
        let mut hasher = Sha256::new();
        let mut written = 0u64;
        let mut flushed = 0u64;
        let mut sink = |bytes: &[u8],
                        hasher: &mut Sha256,
                        written: &mut u64,
                        flushed: &mut u64|
         -> Result<(), Error> {
            hasher.update(bytes);
            if let Some(o) = out.as_mut() {
                o.write_all(bytes).map_err(io)?;
                *written += bytes.len() as u64;
                self.counters.written(bytes.len() as u64);
                if *written - *flushed >= self.floor.chunk() {
                    fsio::flush_and_evict_range(o, *flushed, *written - *flushed).map_err(io)?;
                    *flushed = *written;
                }
            }
            Ok(())
        };
        let mut kept = None;
        if let Some(prefix) = self.prefixes.get(&f.name) {
            // The header region was read (as metadata) and parsed before; copy it from memory.
            sink(prefix, &mut hasher, &mut written, &mut flushed)?;
            let mut off = prefix.len() as u64;
            while off < size {
                let n = (size - off).min(buf.len() as u64) as usize;
                f.open.read_exact_at(&mut buf[..n], off).map_err(io)?;
                self.counters.payload(n as u64);
                sink(&buf[..n], &mut hasher, &mut written, &mut flushed)?;
                self.floor.progress(n as u64)?;
                off += n as u64;
            }
        } else if size < SMALL_FILE_MAX {
            let bytes = source::read_all(&f.open, &f.name)?;
            self.counters.payload(size);
            sink(&bytes, &mut hasher, &mut written, &mut flushed)?;
            self.floor.progress(size)?;
            kept = Some(bytes);
        } else {
            let mut off = 0u64;
            while off < size {
                let n = (size - off).min(buf.len() as u64) as usize;
                f.open.read_exact_at(&mut buf[..n], off).map_err(io)?;
                self.counters.payload(n as u64);
                sink(&buf[..n], &mut hasher, &mut written, &mut flushed)?;
                self.floor.progress(n as u64)?;
                off += n as u64;
            }
        }
        source::check_unchanged(&f.open, &f.name)?;
        if let Some(o) = out.as_mut() {
            if written > flushed {
                fsio::flush_and_evict_range(o, flushed, written - flushed).map_err(io)?;
            }
            o.sync_all().map_err(io)?;
        }
        Ok((hex(&hasher.finalize()), kept))
    }
}

/// Reads the token map source whole (payload), converts it with the vocabulary of `config.json`.
fn convert_token_map(
    token: &TokenSource,
    config: &[u8],
    limits: &Limits,
    counters: &Counters,
) -> Result<(Vec<u8>, TokenOut), Error> {
    // Bounded before the whole-file read: the Q9 floor never sees this allocation.
    let max = token_map::max_file_bytes(limits);
    if token.open.stat.size > max {
        return Err(Error::reject(
            token_map::TokenMapError::ZipEntry.class(),
            format!(
                "token map {} is {} bytes > {max}",
                token.path.display(),
                token.open.stat.size
            ),
        ));
    }
    let raw = source::read_all(&token.open, &token.path.display().to_string())?;
    counters.payload(raw.len() as u64);
    let vocab = token_map::vocab_size_from_config(config)
        .map_err(|e| reject("token_map", e.to_string()))?;
    let ids = token_map::convert(&raw, vocab, limits)?;
    let out = token_map::to_u32le(&ids);
    let t = TokenOut {
        source_sha256: sha256_hex(&raw),
        source_bytes: raw.len() as u64,
        out_sha256: sha256_hex(&out),
        out_bytes: out.len() as u64,
    };
    Ok((out, t))
}

fn write_new(
    dir: &Dir,
    name: &str,
    bytes: &[u8],
    counters: &Counters,
) -> Result<std::fs::File, Error> {
    let io = |e: std::io::Error| Error::io(name, e);
    let mut f = dir.create_file(name).map_err(io)?;
    f.write_all(bytes).map_err(io)?;
    counters.written(bytes.len() as u64);
    fsio::flush_and_evict_range(&f, 0, bytes.len() as u64).map_err(io)?;
    f.sync_all().map_err(io)?;
    Ok(f)
}

/// Writes `manifest.json` and `manifest.sha256`, makes everything read-only, syncs, and publishes `tmp` as `id`.
fn seal_and_publish(
    root: &Dir,
    tmp: &str,
    tmp_dir: &Dir,
    m: &Manifest,
    id: &str,
    counters: &Counters,
) -> Result<(), Error> {
    let io = |what: &str, e: std::io::Error| Error::io(what, e);
    let bytes = m.to_bytes();
    let sha = sha256_hex(&bytes);
    for (name, content) in [
        (manifest::MANIFEST_FILE, bytes),
        (manifest::DIGEST_FILE, format!("{sha}\n").into_bytes()),
    ] {
        write_new(tmp_dir, name, &content, counters)?;
    }
    let mut paths: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
    paths.extend([manifest::MANIFEST_FILE, manifest::DIGEST_FILE]);
    for p in paths {
        let f = tmp_dir.open_file(p)?;
        f.file
            .set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o444))
            .map_err(|e| io(p, e))?;
    }
    if m.file(token_map::OUTPUT_PATH).is_some() {
        let draft = tmp_dir.open_dir("draft")?;
        draft.sync().map_err(|e| io("draft", e))?;
        draft.chmod(0o555).map_err(|e| io("draft", e))?;
    }
    tmp_dir.sync().map_err(|e| io(tmp, e))?;
    tmp_dir.chmod(0o555).map_err(|e| io(tmp, e))?;
    root.rename_noreplace(tmp, id).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            reject(
                "artifact_exists",
                format!("{id} exists without a valid stamp; verify or remove it"),
            )
        } else {
            io(id, e)
        }
    })?;
    root.sync().map_err(|e| io("ROOT", e))
}

/// Removes a temp directory (making read-only directories writable first).
fn remove_tree(p: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::symlink_metadata(p) else {
        return;
    };
    if meta.is_dir() {
        let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700));
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                remove_tree(&e.path());
            }
        }
        let _ = std::fs::remove_dir(p);
    } else {
        let _ = std::fs::remove_file(p);
    }
}

fn remove_stale_tmp(root: &Dir, root_path: &Path) -> Result<Vec<String>, Error> {
    let entries = root
        .entries()
        .map_err(|e| Error::io(root_path.display(), e))?;
    let mut removed = Vec::new();
    for (name, _) in entries {
        if name.starts_with(".tmp-") {
            remove_tree(&root_path.join(&name));
            removed.push(name);
        }
    }
    removed.sort();
    Ok(removed)
}

fn ms(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

/// The stamp of a published artifact, from `fstat` and `fstatfs` of its files.
fn new_stamp(
    held: &BTreeMap<String, stamp::Held>,
    sha: &str,
    source: &str,
    fingerprint: String,
) -> Stamp {
    Stamp {
        format: stamp::FORMAT,
        manifest_sha256: sha.to_string(),
        stamp_source: source.to_string(),
        source_fingerprint: fingerprint,
        files: held.iter().map(|(p, h)| (p.clone(), h.id())).collect(),
        verified_utc: stamp::utc_now(),
    }
}

/// Runs `cerec prepare`; the report JSON on success.
pub fn run(o: &Options) -> Result<Value, Error> {
    let t_start = Instant::now();
    check_name(&o.name)?;
    let threads = o.threads.max(1);
    let floor = Floor::new(o.min_mem_available_gib, threads as u64 * BUFFER_BYTES);
    let mem_before = floor.check()?;
    let counters = Counters::default();
    counters.metadata(
        (o.families.len() + o.profile.len() + o.lock.as_ref().map_or(0, Vec::len)) as u64,
    );
    let rules = load_rules(&o.families)?;
    let profile_value =
        canon_json::parse_strict(&o.profile).map_err(|e| reject("profile", e.to_string()))?;
    let profile = Profile::from_value(&profile_value).map_err(|e| reject("profile", e))?;
    let lock = o.lock.as_deref().map(lock_hashes).transpose()?;

    let src_real = std::fs::canonicalize(&o.src)
        .map_err(|e| reject("source", format!("{}: {e}", o.src.display())))?;
    let out_real = resolve_lenient(&o.out)?;
    check_out(&src_real, &out_real)?;
    std::fs::create_dir_all(&out_real).map_err(|e| Error::io(out_real.display(), e))?;
    let root = Dir::open(&out_real)
        .map_err(|e| reject("unsafe_out", format!("{}: {e}", out_real.display())))?;
    let _lock = FileLock::acquire(&out_real.join(LOCK_FILE), o.no_wait).map_err(|e| match e {
        LockError::Busy => Error {
            class: "busy",
            exit: EXIT_BUSY,
            message: format!(
                "{} is held by another cerec",
                out_real.join(LOCK_FILE).display()
            ),
        },
        LockError::Io(e) => Error::io(LOCK_FILE, e),
    })?;
    let stale_removed = remove_stale_tmp(&root, &out_real)?;

    // Open: source descriptors, tree, headers, families (metadata reads only).
    let t_open = Instant::now();
    let src = source::open(&src_real, Select::Payload)?;
    counters.metadata(src.metadata_bytes_read);
    if src.tree.is_none() {
        return Err(reject(
            "source_tree",
            "no .cache/huggingface/trees/*.json in SRC",
        ));
    }
    if src.files.len() as u64 + 1 > o.limits.max_files {
        return Err(reject(
            "too_many_files",
            format!("{} payload files", src.files.len()),
        ));
    }
    let mut headers = BTreeMap::new();
    let mut prefixes = BTreeMap::new();
    let mut total_tensors = 0u64;
    for f in src.files.iter().filter(|f| f.role == role::WEIGHTS) {
        let mut first8 = [0u8; 8];
        if f.open.stat.size < 8 {
            return Err(safetensors::HeaderError::Bounds.into());
        }
        f.open
            .read_exact_at(&mut first8, 0)
            .map_err(|e| Error::io(&f.name, e))?;
        let n = safetensors::header_len(&first8, f.open.stat.size, &o.limits)?;
        let mut prefix = vec![0u8; 8 + n as usize];
        f.open
            .read_exact_at(&mut prefix, 0)
            .map_err(|e| Error::io(&f.name, e))?;
        counters.metadata(prefix.len() as u64);
        let h = safetensors::parse(&prefix, f.open.stat.size, &o.limits)?;
        total_tensors += h.tensors.len() as u64;
        if total_tensors > o.limits.max_total_tensors {
            return Err(reject(
                "too_many_tensors",
                format!("more than {} tensors", o.limits.max_total_tensors),
            ));
        }
        headers.insert(f.name.clone(), h);
        prefixes.insert(f.name.clone(), prefix);
    }
    let ordered: Vec<&Header> = src
        .files
        .iter()
        .filter_map(|f| headers.get(&f.name))
        .collect();
    let classified = classify(&rules, &ordered)?;
    let token = match (profile.speculative.as_str(), &o.token_map) {
        ("none", None) => None,
        ("none", Some(p)) => {
            return Err(reject(
                "token_map",
                format!(
                    "--token-map {} given for profile {} (speculative none)",
                    p.display(),
                    profile.profile_id
                ),
            ))
        }
        (_, Some(p)) => Some(open_token_map(p)?),
        (spec, None) => {
            return Err(reject(
                "token_map",
                format!(
                    "profile {} (speculative {spec}) needs --token-map",
                    profile.profile_id
                ),
            ))
        }
    };
    let fp = Fingerprint::new(&src, &headers, token.as_ref());
    let inputs = Inputs {
        src: &src,
        headers: &headers,
        rules: &rules,
        profile: &profile,
        tallies: &classified.tallies,
    };
    let open_ms = ms(t_open);

    // Decide (Q7).
    let t_decide = Instant::now();
    let candidates = read_candidates(&root, &out_real, &o.name, &o.limits, &counters);
    let mut stages = BTreeMap::from([("open".to_string(), open_ms)]);
    let mut outcome = "prepared";
    let mut rehash_pass = false;
    let mut staging_bytes = 0u64;
    let mut chosen: Option<(Manifest, UpstreamReport, &Candidate)> = None;
    // (1a) a valid stamp with this fingerprint: hashes known without reading payload.
    for c in candidates.iter().filter(|c| c.stamp.is_some()) {
        let tok = TokenOut::from_manifest(&c.manifest);
        let Some(want) = fp.with_token(tok.as_ref()) else {
            continue;
        };
        if c.stamp.as_ref().map(|s| s.source_fingerprint.as_str()) != Some(want.as_str()) {
            continue;
        }
        let known: BTreeMap<String, String> = c
            .manifest
            .files
            .iter()
            .map(|f| (f.path.clone(), f.sha256.clone()))
            .collect();
        if src.files.iter().any(|f| {
            c.manifest
                .file(&f.name)
                .is_none_or(|e| e.bytes != f.open.stat.size)
        }) {
            continue;
        }
        let (m, up) = build_manifest(&inputs, &known, tok.as_ref(), lock.as_ref())?;
        if same_content(&m, &c.manifest) {
            let exact = candidates
                .iter()
                .find(|x| x.stamp.is_some() && x.manifest == m);
            outcome = if exact.is_some() {
                "reused"
            } else {
                "relabelled"
            };
            chosen = Some((m, up, exact.unwrap_or(c)));
            break;
        }
    }
    // (1b) same (path, bytes) file set: a hash-only pass over the source, no write. Changed content then costs the
    // one documented double read (`rehash_pass` with outcome `prepared`).
    if chosen.is_none() {
        let set_matches = |c: &Candidate| {
            let tok = TokenOut::from_manifest(&c.manifest);
            c.manifest.files.len() == src.files.len() + usize::from(token.is_some())
                && match (&tok, &token) {
                    (Some(t), Some(s)) => t.source_bytes == s.open.stat.size,
                    (None, None) => true,
                    _ => false,
                }
                && src.files.iter().all(|f| {
                    c.manifest
                        .file(&f.name)
                        .is_some_and(|e| e.bytes == f.open.stat.size)
                })
        };
        if let Some(c) = candidates
            .iter()
            .find(|c| c.stamp.is_some() && set_matches(c))
        {
            let t = Instant::now();
            rehash_pass = true;
            let pass = Pass {
                src: &src,
                prefixes: &prefixes,
                dest: None,
                counters: &counters,
                floor: &floor,
                threads,
                abort_after: None,
            };
            let out = pass.run()?;
            staging_bytes = staging_bytes.max(out.staging_bytes);
            let config = out
                .small
                .get("config.json")
                .ok_or_else(|| reject("source", "no config.json"))?;
            let tok = token
                .as_ref()
                .map(|t| convert_token_map(t, config, &o.limits, &counters).map(|(_, tok)| tok))
                .transpose()?;
            let (m, up) = build_manifest(&inputs, &out.hashes, tok.as_ref(), lock.as_ref())?;
            stages.insert("rehash".into(), ms(t));
            if same_content(&m, &c.manifest) {
                let exact = candidates
                    .iter()
                    .find(|x| x.stamp.is_some() && x.manifest == m);
                outcome = if exact.is_some() {
                    "reused_after_rehash"
                } else {
                    "relabelled"
                };
                chosen = Some((m, up, exact.unwrap_or(c)));
            }
        }
    }
    stages.insert("decide".into(), ms(t_decide));

    let fingerprint = |m: &Manifest| {
        fp.with_token(TokenOut::from_manifest(m).as_ref())
            .expect("the chosen manifest has this run's token map presence")
    };
    let t_publish = Instant::now();
    let (manifest, upstream, artifact_id) = match chosen {
        Some((m, up, c)) if outcome == "reused" || outcome == "reused_after_rehash" => {
            let mut s = c.stamp.clone().expect("candidate has a stamp");
            let mut stale = false;
            let want = fingerprint(&m);
            if s.source_fingerprint != want {
                s.source_fingerprint = want;
                stale = true;
            }
            // A format-1 stamp that still matches is restated as format 2 from the files held now (Q7), so a reboot
            // that renumbers `st_dev` keeps the reuse. One that no longer matches is left alone.
            if s.format == stamp::LEGACY_FORMAT {
                if let Ok(held) = stamp::stat_files(&c.dir, s.files.keys().map(String::as_str)) {
                    if s.matches(&s.manifest_sha256, &held) {
                        s.format = stamp::FORMAT;
                        s.files = held.iter().map(|(p, h)| (p.clone(), h.id())).collect();
                        stale = true;
                    }
                }
            }
            if stale {
                counters.written(s.write(&root, &c.id)?);
            }
            (m, up, c.id.clone())
        }
        Some((m, up, c)) => {
            // Relabel: hardlink the candidate's payload into a new artifact; only manifest and stamps are written.
            let sha = sha256_hex(&m.to_bytes());
            let id = format!("{}-{}", o.name, &sha[..16]);
            let tmp = stamp::tmp_name();
            let tmp_dir = root
                .create_dir(&tmp, 0o700)
                .map_err(|e| Error::io(&tmp, e))?;
            let res = (|| {
                if m.file(token_map::OUTPUT_PATH).is_some() {
                    tmp_dir
                        .create_dir("draft", 0o700)
                        .map_err(|e| Error::io("draft", e))?;
                }
                for f in &m.files {
                    c.dir
                        .link_into(&f.path, &tmp_dir)
                        .map_err(|e| Error::io(&f.path, e))?;
                }
                seal_and_publish(&root, &tmp, &tmp_dir, &m, &id, &counters)
            })();
            if let Err(e) = res {
                remove_tree(&out_real.join(&tmp));
                return Err(e);
            }
            let dir = root.open_dir(&id)?;
            let held = stamp::stat_files(&dir, m.files.iter().map(|f| f.path.as_str()))?;
            let st = new_stamp(&held, &sha, stamp::PREPARE_HASH, fingerprint(&m));
            // Linking changed the ctime of the shared inodes, which every earlier relabel of the same payload also
            // holds: refresh each valid stamp whose files changed only in the ctime of a linked inode (R3(d)).
            let linked: BTreeSet<(u64, u64)> =
                held.values().map(|h| (h.stat.dev, h.stat.ino)).collect();
            for other in &candidates {
                let Some(old) = &other.stamp else {
                    continue;
                };
                let Ok(now) = stamp::stat_files(&other.dir, old.files.keys().map(String::as_str))
                else {
                    continue;
                };
                let only_link_ctime = old.files.iter().all(|(p, s)| {
                    now.get(p).is_some_and(|n| {
                        let mut at = n.stat;
                        if linked.contains(&(at.dev, at.ino)) {
                            at.ctime_ns = s.ctime_ns;
                        }
                        s.check(old.format, &at, &n.fs).is_ok()
                    })
                });
                if only_link_ctime
                    && old
                        .files
                        .iter()
                        .any(|(p, s)| now[p].stat.ctime_ns != s.ctime_ns)
                {
                    let mut refreshed = old.clone();
                    refreshed.format = stamp::FORMAT;
                    refreshed.files = now.iter().map(|(p, n)| (p.clone(), n.id())).collect();
                    counters.written(refreshed.write(&root, &other.id)?);
                }
            }
            counters.written(st.write(&root, &id)?);
            (m, up, id)
        }
        None => {
            // (3) Full prepare: one copy+hash pass into ROOT/.tmp-*, then publish.
            let t = Instant::now();
            let tmp = stamp::tmp_name();
            let tmp_dir = root
                .create_dir(&tmp, 0o700)
                .map_err(|e| Error::io(&tmp, e))?;
            let res = (|| -> Result<(Manifest, UpstreamReport, String), Error> {
                let pass = Pass {
                    src: &src,
                    prefixes: &prefixes,
                    dest: Some(&tmp_dir),
                    counters: &counters,
                    floor: &floor,
                    threads,
                    abort_after: abort_after_files(),
                };
                let out = pass.run()?;
                staging_bytes = staging_bytes.max(out.staging_bytes);
                let config = out
                    .small
                    .get("config.json")
                    .ok_or_else(|| reject("source", "no config.json"))?;
                let tok = match &token {
                    Some(t) => {
                        let (bytes, tok) = convert_token_map(t, config, &o.limits, &counters)?;
                        let draft = tmp_dir
                            .create_dir("draft", 0o700)
                            .map_err(|e| Error::io("draft", e))?;
                        write_new(&draft, "token_map.u32le", &bytes, &counters)?;
                        Some(tok)
                    }
                    None => None,
                };
                let (m, up) = build_manifest(&inputs, &out.hashes, tok.as_ref(), lock.as_ref())?;
                stages.insert("copy".into(), ms(t));
                let sha = sha256_hex(&m.to_bytes());
                let id = format!("{}-{}", o.name, &sha[..16]);
                seal_and_publish(&root, &tmp, &tmp_dir, &m, &id, &counters)?;
                Ok((m, up, id))
            })();
            let (m, up, id) = match res {
                Ok(v) => v,
                Err(e) => {
                    remove_tree(&out_real.join(&tmp));
                    return Err(e);
                }
            };
            let dir = root.open_dir(&id)?;
            let sha = sha256_hex(&m.to_bytes());
            let held = stamp::stat_files(&dir, m.files.iter().map(|f| f.path.as_str()))?;
            let st = new_stamp(&held, &sha, stamp::PREPARE_HASH, fingerprint(&m));
            counters.written(st.write(&root, &id)?);
            (m, up, id)
        }
    };
    stages.insert("publish".into(), ms(t_publish));
    stages.insert("total".into(), ms(t_start));
    let mem_after = fsio::mem_available().unwrap_or(0);
    let manifest_sha256 = sha256_hex(&manifest.to_bytes());
    Ok(json!({
        "outcome": outcome,
        "artifact_id": artifact_id,
        "artifact_dir": out_real.join(&artifact_id).display().to_string(),
        "manifest_sha256": manifest_sha256,
        "payload_bytes_read": counters.payload_bytes_read.load(Ordering::Relaxed),
        "metadata_bytes_read": counters.metadata_bytes_read.load(Ordering::Relaxed),
        "bytes_written": counters.bytes_written.load(Ordering::Relaxed),
        "rehash_pass": rehash_pass,
        "stages_ms": stages,
        "upstream": {
            "lfs_files": upstream.lfs_files, "lfs_match": upstream.lfs_match,
            "non_lfs_files": upstream.non_lfs_files, "not_in_tree": upstream.not_in_tree,
            "local_modification": upstream.local_modification,
        },
        "model_hash": manifest.model_hash,
        "tokenizer_hash": manifest.tokenizer_hash,
        "template_hash": manifest.template_hash,
        "lock": if lock.is_some() { "matched" } else { "not_given" },
        "excluded": src.excluded,
        "stale_tmp_removed": stale_removed,
        "threads": threads,
        "staging_bytes": staging_bytes,
        "mem_available_before": mem_before,
        "mem_available_after": mem_after,
        "memory_floor": floor.bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_may_not_overlap_the_source() {
        let src = Path::new("/m/src");
        assert!(check_out(src, Path::new("/m/src")).is_err());
        assert!(check_out(src, Path::new("/m/src/out")).is_err());
        assert!(check_out(src, Path::new("/m")).is_err());
        assert!(check_out(src, Path::new("/m/src2")).is_ok());
        assert!(check_out(src, Path::new("/a/out")).is_ok());
    }

    #[test]
    fn names_are_plain() {
        assert!(check_name("fixture-model").is_ok());
        for bad in ["", ".x", "a/b", "a b", ".tmp-1"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
    }
}
