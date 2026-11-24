//! NetTeleport S3 (net-teleport design note 4.2 L1, C7, D-T3, D-T5): warm restart on one install.
//!
//! With `[state] park_on_stop`, a [`Recorder`] keeps the prompt ids of the most recent text-only namespaced requests
//! in memory (never on disk). At a graceful stop, after the drain, [`Park::park`] exports the deepest cached prefix
//! state of each recorded prompt, most recent first, until `state.park_deadline_ms` or `state.park_max_bytes`, and
//! writes each one sealed into `<ledger.state_dir>/parked/` (0700 directory, 0600 files, fsync'd). A parked file is
//! `CEREPRK1`, the 16-byte namespace, then S1's `CERESTB1` blob sealed under a key HMAC-derived from the install's
//! `server_secret` and that namespace, so a file moved to another namespace (or another install) fails its key check.
//! At the next start, after the model is loaded and before the backend serves, [`Park::unpark`] imports every file
//! whose library, profile, layout, compat id and tier match strictly (L1 keeps the strict profile check, D-T3) and
//! deletes every file, imported or refused. A refused image costs only a replay: the client re-issues its turn and
//! the prompt is prefilled cold (D-T1). Log lines carry counts, bytes, refusal codes and timings; never tokens.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use super::export::{LockedBuf, StateIdentity};
use super::teleport::{
    admit, BlobHeader, HeaderReader, Local, Opener, Refusal, SealMeta, Sealer, StateKey,
};
use crate::app::NativeBackend;
use crate::config::RuntimeConfig;
use crate::digest::{open_secret, SECRET_FILE};
use crate::journal::{ensure_private_dir, sync_dir};
use crate::tier::Tier;

/// `state.park_max_sessions` when absent.
pub const DEFAULT_MAX_SESSIONS: u64 = 8;
/// `state.park_max_bytes` when absent: 16 GiB of images.
pub const DEFAULT_MAX_BYTES: u64 = 16 << 30;
/// `state.park_deadline_ms` when absent.
pub const DEFAULT_DEADLINE_MS: u64 = 10_000;
/// The parked images' directory under `ledger.state_dir`.
pub const DIR: &str = "parked";
/// Parked file magic (format 1).
pub const MAGIC: &[u8; 8] = b"CEREPRK1";
/// Magic plus namespace.
pub const PREFIX_BYTES: usize = 8 + 16;
const SUFFIX: &str = ".stb";
const KEY_INFO: &[u8] = b"cere-park-key-v1";
const FILE_KEY_INFO: &[u8] = b"cere-park-file-v1";
/// Read size while opening a parked file.
const READ_BYTES: usize = 1 << 20;

type HmacSha256 = Hmac<Sha256>;

/// One recorded prompt: its prefix-cache namespace and token ids.
pub type Recorded = ([u8; 16], Vec<u32>);

/// The recent prompts a stop parks, most recent first. Cloned handles share one list.
#[derive(Clone)]
pub struct Recorder {
    inner: Arc<Mutex<VecDeque<Recorded>>>,
    max: usize,
}

impl Recorder {
    pub fn new(max: usize) -> Recorder {
        Recorder {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(max))),
            max,
        }
    }

    /// Records `ids` under `namespace` as the most recent prompt. A session's next turn extends its previous prompt,
    /// so the new prompt replaces every recorded one of the namespace it extends (or that extends it, a re-issue).
    pub fn record(&self, namespace: [u8; 16], ids: &[u32]) {
        if ids.is_empty() || self.max == 0 {
            return;
        }
        let mut list = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        list.retain(|(ns, old)| {
            *ns != namespace || !(ids.starts_with(old) || old.starts_with(ids))
        });
        list.push_front((namespace, ids.to_vec()));
        list.truncate(self.max);
    }

    /// Takes the recorded prompts, most recent first, leaving the list empty.
    pub fn take(&self) -> Vec<Recorded> {
        let mut list = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        std::mem::take(&mut *list).into()
    }
}

/// The live limits of one stop's park.
#[derive(Clone, Copy)]
pub struct Limits {
    /// `state.park_max_bytes`: image bytes of the whole park.
    pub max_bytes: u64,
    /// `state.max_blob_bytes`: image bytes of one parked state.
    pub max_blob_bytes: u64,
    /// The drain's end plus `state.park_deadline_ms`: no export starts after it.
    pub deadline: Instant,
    /// `server.tier`: the source tier and the one allowed tier of every parked image.
    pub tier: Tier,
}

/// What one park did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Parked {
    pub parked: u64,
    pub bytes: u64,
    /// Recorded prompts whose deepest cached state another recorded prompt already parked.
    pub duplicate: u64,
    /// Recorded prompts left when the deadline or the byte budget ended the park.
    pub left: u64,
    /// Refusal (or `io`) code -> count.
    pub refused: BTreeMap<&'static str, u64>,
}

/// What one start's unpark did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Unparked {
    pub imported: u64,
    pub tokens: u64,
    pub bytes: u64,
    /// Refusal (`state_profile`, `io` or a [`Refusal`] code) -> count.
    pub refused: BTreeMap<&'static str, u64>,
    /// Files in the directory that were not parked images (an interrupted write's temp file).
    pub stale: u64,
}

/// Why a parked file was not imported.
#[derive(Debug)]
pub enum Skip {
    /// L1's strict check (D-T3): the image's capability profile is not this backend's.
    Profile,
    Refused(Refusal),
    Io(io::Error),
}

impl Skip {
    pub fn code(&self) -> &'static str {
        match self {
            Skip::Profile => "state_profile_mismatch",
            Skip::Refused(r) => r.code(),
            Skip::Io(_) => "io",
        }
    }
}

impl From<Refusal> for Skip {
    fn from(r: Refusal) -> Skip {
        Skip::Refused(r)
    }
}

impl From<io::Error> for Skip {
    fn from(e: io::Error) -> Skip {
        Skip::Io(e)
    }
}

/// The parked-state store of one install: its directory, its key and the recorder its backends share.
pub struct Park {
    dir: PathBuf,
    /// HMAC-SHA256(server_secret, "cere-park-key-v1"); wiped on drop.
    key: [u8; 32],
    recorder: Recorder,
    /// The start's `server.tier`, `state.max_blob_bytes` and `state.busy_wait_ms`, for the unpark.
    tier: Tier,
    max_blob_bytes: u64,
    busy_wait: Duration,
}

impl Drop for Park {
    fn drop(&mut self) {
        wipe(&mut self.key);
    }
}

impl Park {
    /// `None` without `[state] park_on_stop`; else the store under `<ledger.state_dir>/parked/` (created 0700) keyed
    /// from the install's `server_secret` (which `DigestKeys::open` created).
    pub fn open(config: &RuntimeConfig) -> io::Result<Option<Park>> {
        if !config.state.park_on_stop() {
            return Ok(None);
        }
        let state_dir = &config.ledger.state_dir;
        let mut secret = open_secret(state_dir, SECRET_FILE)?;
        let key = mac(&secret, &[KEY_INFO]);
        wipe(&mut secret);
        let dir = state_dir.join(DIR);
        ensure_private_dir(&dir)?;
        let max = usize::try_from(config.state.park_max_sessions()).unwrap_or(usize::MAX);
        Ok(Some(Park {
            dir,
            key,
            recorder: Recorder::new(max),
            tier: config.server.tier(),
            max_blob_bytes: config.state.max_blob_bytes(),
            busy_wait: Duration::from_millis(config.state.busy_wait_ms()),
        }))
    }

    /// A handle on the shared recorder (every load of the backend records into the one list a stop takes).
    pub fn recorder(&self) -> Recorder {
        self.recorder.clone()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The sealing key of a parked file under `namespace`.
    fn file_key(&self, namespace: [u8; 16]) -> StateKey {
        StateKey::new(mac(&self.key, &[FILE_KEY_INFO, &namespace[..]]))
    }

    /// Exports and parks `recorded` (most recent first) under `limits`; logs and returns the counts. Blocking.
    pub fn park(
        &self,
        backend: &NativeBackend,
        recorded: Vec<Recorded>,
        limits: &Limits,
    ) -> Parked {
        let started = Instant::now();
        let created = unix_ms();
        let identity = &backend.identity;
        let compat_id = backend.state_compat_id();
        let mut out = Parked::default();
        let mut seen = HashSet::new();
        let total = recorded.len();
        for (rank, (namespace, ids)) in recorded.into_iter().enumerate() {
            let room = limits.max_bytes.saturating_sub(out.bytes);
            if Instant::now() >= limits.deadline || room == 0 {
                out.left = (total - rank) as u64;
                break;
            }
            let cap = limits.max_blob_bytes.min(room);
            let (image, tokens) =
                match backend.export_prefix_state(namespace, &ids, cap, limits.deadline) {
                    Ok(exported) => exported,
                    Err(r) => {
                        *out.refused.entry(r.code()).or_default() += 1;
                        continue;
                    }
                };
            let prefix = usize::try_from(tokens)
                .ok()
                .and_then(|n| ids.get(..n))
                .unwrap_or(&ids);
            if !seen.insert(prefix_digest(namespace, prefix)) {
                out.duplicate += 1;
                continue;
            }
            let bytes = image.len() as u64;
            let meta = seal_meta(identity, compat_id, tokens, limits.tier, created);
            match self.write(&file_name(created, rank), namespace, meta, image) {
                Ok(()) => {
                    out.parked += 1;
                    out.bytes += bytes;
                }
                Err(skip) => *out.refused.entry(skip.code()).or_default() += 1,
            }
        }
        eprintln!(
            "cered state park: recorded={total} parked={} image_bytes={} duplicate={} left={} refused={} ms={}",
            out.parked,
            out.bytes,
            out.duplicate,
            out.left,
            codes(&out.refused),
            started.elapsed().as_millis()
        );
        out
    }

    /// Imports every parked image `backend` admits (oldest stop first, its least recent session first, so the most
    /// recent ends most recently used), deletes every file, logs and returns the counts. Blocking; the executor is
    /// loaded and idle (the slot is not filled yet).
    pub fn unpark(&self, backend: &NativeBackend) -> Unparked {
        let started = Instant::now();
        let mut out = Unparked::default();
        let mut files = Vec::new();
        match fs::read_dir(&self.dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    match entry.file_name().to_str().and_then(parse_name) {
                        Some(order) => files.push((order, path)),
                        None => {
                            out.stale += 1;
                            remove(&path);
                        }
                    }
                }
            }
            Err(_) => *out.refused.entry("io").or_default() += 1,
        }
        files.sort_by_key(|&((created, rank), _)| (created, std::cmp::Reverse(rank)));
        let local = Local {
            identity: &backend.identity,
            compat_id: backend.state_compat_id(),
            tier: self.tier,
            max_blob_bytes: self.max_blob_bytes,
        };
        for (_, path) in &files {
            let imported =
                self.read(path, |header| strict(header, &local))
                    .and_then(|(namespace, image)| {
                        let deadline = Instant::now() + self.busy_wait;
                        let tokens = backend.import_prefix_state(namespace, &image, deadline)?;
                        Ok((tokens, image.len() as u64))
                    });
            match imported {
                Ok((tokens, bytes)) => {
                    out.imported += 1;
                    out.tokens += tokens;
                    out.bytes += bytes;
                }
                Err(skip) => *out.refused.entry(skip.code()).or_default() += 1,
            }
            remove(path);
        }
        if !files.is_empty() || out.stale > 0 {
            let _ = sync_dir(&self.dir);
        }
        eprintln!(
            "cered state unpark: files={} imported={} tokens={} image_bytes={} refused={} stale={} ms={}",
            files.len(),
            out.imported,
            out.tokens,
            out.bytes,
            codes(&out.refused),
            out.stale,
            started.elapsed().as_millis()
        );
        out
    }

    /// Seals `image` under `namespace`'s file key into `<dir>/<name>`: a 0600 temp file written and fsync'd, renamed
    /// into place, the directory fsync'd. The temp file is removed on failure.
    pub fn write(
        &self,
        name: &str,
        namespace: [u8; 16],
        meta: SealMeta,
        image: LockedBuf,
    ) -> Result<(), Skip> {
        let mut sealer = Sealer::new(&self.file_key(namespace), meta, image)?;
        let tmp = self.dir.join(format!(".{name}.tmp"));
        let written = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?;
            file.write_all(MAGIC)?;
            file.write_all(&namespace)?;
            while let Some(chunk) = sealer.next_chunk() {
                file.write_all(&chunk)?;
            }
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp, self.dir.join(name))?;
            sync_dir(&self.dir)
        })();
        if written.is_err() {
            remove(&tmp);
        }
        Ok(written?)
    }

    /// Opens the parked file at `path`: its namespace and image, once `check` admitted its header (before any key
    /// use or allocation) and every frame opened under the namespace's file key.
    pub fn read(
        &self,
        path: &Path,
        check: impl FnOnce(&BlobHeader) -> Result<(), Skip>,
    ) -> Result<([u8; 16], LockedBuf), Skip> {
        let mut file = File::open(path)?;
        let mut prefix = [0u8; PREFIX_BYTES];
        file.read_exact(&mut prefix)
            .map_err(|_| Refusal::Malformed("short parked file".to_owned()))?;
        if &prefix[..8] != MAGIC {
            return Err(Refusal::Malformed("bad parked magic".to_owned()).into());
        }
        let namespace: [u8; 16] = prefix[8..].try_into().expect("16 bytes");
        let mut buf = vec![0u8; READ_BYTES];
        let mut reader = HeaderReader::default();
        let parsed = loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                return Err(Refusal::Malformed("truncated header".to_owned()).into());
            }
            if let Some(parsed) = reader.push(&buf[..n])? {
                break parsed;
            }
        };
        check(&parsed.header)?;
        let (mut opener, rest) = Opener::new(&self.file_key(namespace), parsed)?;
        opener.push(&rest)?;
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            opener.push(&buf[..n])?;
        }
        Ok((namespace, opener.finish()?))
    }
}

/// L1's admission (D-T3): the strict profile check, then S1's size, library, layout, compat id and tier checks.
pub fn strict(header: &BlobHeader, local: &Local) -> Result<(), Skip> {
    if header.profile_id != local.identity.profile_id {
        return Err(Skip::Profile);
    }
    Ok(admit(header, local)?)
}

/// The header fields of a state this install parks for itself.
pub fn seal_meta(
    identity: &StateIdentity,
    compat_id: [u8; 32],
    tokens: u64,
    tier: Tier,
    created_unix_ms: u64,
) -> SealMeta {
    SealMeta {
        library_sha256: identity.library_sha256.clone(),
        profile_id: identity.profile_id.clone(),
        layout_id: identity.layout_id,
        compat_id,
        tokens,
        source_tier: tier,
        allowed_tiers: vec![tier],
        created_unix_ms,
    }
}

/// `<created ms>-<rank>.stb`: rank 0 is the stop's most recent session.
pub fn file_name(created_unix_ms: u64, rank: usize) -> String {
    format!("{created_unix_ms:016}-{rank:03}{SUFFIX}")
}

fn parse_name(name: &str) -> Option<(u64, usize)> {
    let (created, rank) = name.strip_suffix(SUFFIX)?.split_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(created) || !digits(rank) {
        return None;
    }
    Some((created.parse().ok()?, rank.parse().ok()?))
}

fn prefix_digest(namespace: [u8; 16], ids: &[u32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(namespace);
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    hash.finalize().into()
}

fn mac(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    for part in parts {
        mac.update(part);
    }
    mac.finalize().into_bytes().into()
}

fn codes(refused: &BTreeMap<&'static str, u64>) -> String {
    if refused.is_empty() {
        return "0".to_owned();
    }
    refused
        .iter()
        .map(|(code, n)| format!("{code}:{n}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn remove(path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        if e.kind() != io::ErrorKind::NotFound {
            eprintln!("cered state unpark: cannot remove a parked file: {e}");
        }
    }
}

fn wipe(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        // SAFETY: `b` is a valid, aligned byte; volatile so the wipe is not elided.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> StateIdentity {
        StateIdentity {
            library_sha256: "ab".repeat(32),
            profile_id: "profile-a".to_owned(),
            layout_id: [7; 32],
        }
    }

    fn park(dir: &Path) -> Park {
        keyed(dir, [3; 32])
    }

    fn keyed(dir: &Path, key: [u8; 32]) -> Park {
        ensure_private_dir(dir).unwrap();
        Park {
            dir: dir.to_owned(),
            key,
            recorder: Recorder::new(4),
            tier: Tier::Own,
            max_blob_bytes: 1 << 30,
            busy_wait: Duration::from_millis(10),
        }
    }

    fn image(len: usize) -> LockedBuf {
        let mut image = LockedBuf::new(len).unwrap();
        for (i, b) in image.as_mut_slice().iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        image
    }

    fn local(identity: &StateIdentity) -> Local<'_> {
        Local {
            identity,
            compat_id: [9; 32],
            tier: Tier::Own,
            max_blob_bytes: 1 << 30,
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cere-park-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn recorder_keeps_the_latest_turn_of_a_session_most_recent_first() {
        let r = Recorder::new(3);
        r.record([1; 16], &[1, 2, 3]);
        r.record([1; 16], &[1, 2, 3, 4, 5]); // the next turn replaces its previous one
        r.record([2; 16], &[1, 2, 3]); // the same tokens in another namespace stay apart
        r.record([1; 16], &[9, 9]);
        r.record([1; 16], &[1, 2, 3, 4, 5]); // a re-issue moves to the front
        r.record([1; 16], &[]);
        assert_eq!(
            r.take(),
            vec![
                ([1; 16], vec![1, 2, 3, 4, 5]),
                ([1; 16], vec![9, 9]),
                ([2; 16], vec![1, 2, 3]),
            ]
        );
        assert!(r.take().is_empty());
        r.record([4; 16], &[1]);
        r.record([5; 16], &[1]);
        r.record([6; 16], &[1]);
        r.record([7; 16], &[1]);
        assert_eq!(r.take().len(), 3, "bounded by park_max_sessions");
    }

    #[test]
    fn a_parked_file_is_private_sealed_and_opens_only_under_its_namespace() {
        let dir = tmp("seal");
        let p = park(&dir);
        let id = identity();
        // A few KiB: the test process may lock only RLIMIT_MEMLOCK (64 KiB in some CI containers); S1's teleport
        // tests cover multi-frame sealing.
        let len = 8_209;
        let name = file_name(1_700_000_000_000, 0);
        p.write(
            &name,
            [5; 16],
            seal_meta(&id, [9; 32], 42, Tier::Own, 1),
            image(len),
        )
        .unwrap();
        let path = dir.join(&name);
        let mode = fs::metadata(&path).unwrap().permissions();
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(&mode) & 0o777,
            0o600
        );
        let dir_mode = fs::metadata(&dir).unwrap().permissions();
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(&dir_mode) & 0o777,
            0o700
        );
        let bytes = fs::read(&path).unwrap();
        let plain = image(len);
        assert!(
            !bytes
                .windows(64)
                .any(|w| w == &plain.as_slice()[1000..1064]),
            "no plaintext run of the image on disk"
        );
        let (namespace, opened) = p.read(&path, |h| strict(h, &local(&id))).unwrap();
        assert_eq!(namespace, [5; 16]);
        assert_eq!(opened.as_slice(), plain.as_slice());

        // The namespace is bound by the file key: a file moved to another namespace fails its key check.
        let mut moved = bytes.clone();
        moved[8..24].copy_from_slice(&[6; 16]);
        fs::write(&path, &moved).unwrap();
        let err = p.read(&path, |h| strict(h, &local(&id))).unwrap_err();
        assert_eq!(err.code(), Refusal::WrongKey.code());

        // Another install's secret: the same.
        fs::write(&path, &bytes).unwrap();
        let other = keyed(&dir, [4; 32]);
        let err = other.read(&path, |h| strict(h, &local(&id))).unwrap_err();
        assert_eq!(err.code(), Refusal::WrongKey.code());

        // A flipped ciphertext byte.
        let mut flipped = bytes.clone();
        let at = bytes.len() - 100;
        flipped[at] ^= 1;
        fs::write(&path, &flipped).unwrap();
        let err = p.read(&path, |h| strict(h, &local(&id))).unwrap_err();
        assert_eq!(err.code(), Refusal::Tampered.code());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_strict_check_refuses_another_profile_library_layout_or_compat_id() {
        let dir = tmp("strict");
        let p = park(&dir);
        let id = identity();
        let name = file_name(1, 0);
        p.write(
            &name,
            [5; 16],
            seal_meta(&id, [9; 32], 1, Tier::Own, 1),
            image(4096),
        )
        .unwrap();
        let path = dir.join(&name);
        let refused = |local: &StateIdentity, compat: [u8; 32]| {
            let l = Local {
                compat_id: compat,
                ..local_of(local)
            };
            p.read(&path, |h| strict(h, &l)).unwrap_err().code()
        };
        fn local_of(identity: &StateIdentity) -> Local<'_> {
            Local {
                identity,
                compat_id: [9; 32],
                tier: Tier::Own,
                max_blob_bytes: 1 << 30,
            }
        }
        let profile = StateIdentity {
            profile_id: "profile-b".to_owned(),
            ..identity()
        };
        assert_eq!(refused(&profile, [9; 32]), "state_profile_mismatch");
        let library = StateIdentity {
            library_sha256: "cd".repeat(32),
            ..identity()
        };
        assert_eq!(refused(&library, [9; 32]), Refusal::Library.code());
        let layout = StateIdentity {
            layout_id: [8; 32],
            ..identity()
        };
        assert_eq!(refused(&layout, [9; 32]), Refusal::Layout.code());
        assert_eq!(refused(&id, [1; 32]), Refusal::Compat.code());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_names_order_stops_then_ranks() {
        assert_eq!(parse_name(&file_name(12, 3)), Some((12, 3)));
        assert_eq!(parse_name(".0000000000000012-003.stb.tmp"), None);
        assert_eq!(parse_name("x-1.stb"), None);
        assert_eq!(parse_name("1-.stb"), None);
    }
}
