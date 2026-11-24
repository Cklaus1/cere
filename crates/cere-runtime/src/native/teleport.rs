//! NetTeleport S1 (net-teleport design note 4.2 L2, 4.3, D-T3, D-T5): the sealed blob a prefix-cache state travels
//! in between installs, and the compat id that admits it.
//!
//! A blob is `CERESTB1`, a little-endian u32 header length, the JSON [`BlobHeader`] (the AAD of 4.3: the F9 envelope,
//! `state_compat_id`, the source tier, the allowed tiers and the creation time), then the `CERESNP1` image in
//! [`FRAME_BYTES`] frames, each AES-256-GCM ciphertext followed by its 16-byte tag. The frame key is HKDF-SHA256 of the
//! client's 32-byte key (`X-Cere-State-Key`, never stored or logged) under the header's random salt; frame `i` has the
//! nonce `0^4 || be64(i)` and the AAD `prefix || header || le64(i) || last`, so a reordered, truncated or edited blob
//! fails to open. `key_check` (HKDF of the same key and salt, other info) tells a wrong key from a tampered blob.
//! Opening decrypts frame by frame into one [`LockedBuf`]; the plaintext never lives in pageable memory.

use std::fmt;
use std::time::{Duration, Instant};

use ring::aead::{Aad, LessSafeKey, Nonce, Tag, UnboundKey, AES_256_GCM, NONCE_LEN};
use ring::hkdf;
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::driver::{Driver, DriverError, Model};
use super::export::{LockedBuf, StateIdentity, AT_LAYOUT};
use super::ffi::StatusError;
use crate::digest::{hex, unhex};
use crate::levers::{LeverSet, Numerics, REGISTRY};
use crate::tier::{allows, Tier};

/// Blob magic (format 1).
pub const MAGIC: &[u8; 8] = b"CERESTB1";
pub const FORMAT: u32 = 1;
/// Plaintext bytes per frame (4 MiB, the 4.2 L3 frame size).
#[cfg(not(test))]
pub const FRAME_BYTES: usize = 4 << 20;
/// The unit tests' frame: 1 KiB, so multi-frame images stay a few KiB and every test's `mlock`ed buffers together fit
/// a 64 KiB RLIMIT_MEMLOCK (a CI container's limit may be that low). The framing code is size-generic; the integration
/// tests (`tests/state_teleport.rs`) run the 4 MiB frame.
#[cfg(test)]
pub const FRAME_BYTES: usize = 1 << 10;
pub const TAG_BYTES: usize = 16;
/// Magic plus the u32 header length.
pub const PREFIX_BYTES: usize = 12;
/// The longest header a blob may carry.
pub const MAX_HEADER_BYTES: usize = 4096;
/// The request header carrying the client's key: 64 hex digits (32 bytes).
pub const KEY_HEADER: &str = "x-cere-state-key";
/// `state.max_blob_bytes` when absent: 4 GiB of image (4.3).
pub const DEFAULT_MAX_BLOB_BYTES: u64 = 4 << 30;
/// `state.busy_wait_ms` when absent: how long an export or import waits for an idle executor (D-T1, R-T5).
pub const DEFAULT_BUSY_WAIT_MS: u64 = 30_000;
const AEAD_INFO: &[u8] = b"cere state teleport v1 aead";
const CHECK_INFO: &[u8] = b"cere state teleport v1 key check";
const SALT_BYTES: usize = 32;
const KEY_CHECK_BYTES: usize = 16;

/// Why a teleport export or import refused. Each refusal imports nothing; the client replays (D-T1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// `X-Cere-State-Key` absent or not 64 hex digits.
    Key,
    /// Not a format-1 blob (magic, header, frame size).
    Malformed(String),
    /// The blob's library build is not this backend's.
    Library,
    /// The blob's (or its image's) state layout is not this backend's.
    Layout,
    /// Same library and layout, other numerics-relevant levers (D-T3).
    Compat,
    /// The tier policy refuses: the destination tier is not in the scope's allowed tiers (export), or this
    /// install's tier is not in the blob's (import).
    Tier(String),
    /// The image exceeds `state.max_blob_bytes`.
    Oversize { bytes: u64, max: u64 },
    /// The key does not open this blob.
    WrongKey,
    /// The blob fails authentication (edited, reordered, truncated or extended).
    Tampered,
    /// The image buffer could not be `mlock`ed (no pageable fallback).
    Memlock,
    /// The executor stayed busy (a ticket in flight) until the deadline (D-T1, R-T5).
    Busy,
    /// The executor has no free state slot, pages or arena room for the import (R-T4).
    Capacity,
    /// No cached prefix of the given tokens in the scope's namespace.
    NoPrefix,
    /// This deployment cannot export or import prefix state (no prefix cache or arena, multi-replica).
    Unsupported(String),
    /// The executor failed otherwise.
    Internal(String),
}

impl Refusal {
    /// HTTP status, error type and the named code of the refusal (docs/API.md "State teleport").
    pub fn parts(&self) -> (u16, &'static str, &'static str) {
        use crate::api_types::{INVALID_REQUEST, SERVER_ERROR};
        match self {
            Refusal::Key => (400, INVALID_REQUEST, "state_key_invalid"),
            Refusal::Malformed(_) => (400, INVALID_REQUEST, "state_malformed"),
            Refusal::Library => (409, INVALID_REQUEST, "state_library_mismatch"),
            Refusal::Layout => (409, INVALID_REQUEST, "state_layout_mismatch"),
            Refusal::Compat => (409, INVALID_REQUEST, "state_compat_mismatch"),
            Refusal::Tier(_) => (403, INVALID_REQUEST, "state_tier_not_allowed"),
            Refusal::Oversize { .. } => (413, INVALID_REQUEST, "state_oversize"),
            Refusal::WrongKey => (400, INVALID_REQUEST, "state_wrong_key"),
            Refusal::Tampered => (400, INVALID_REQUEST, "state_tampered"),
            Refusal::Memlock => (503, SERVER_ERROR, "state_memlock"),
            Refusal::Busy => (409, SERVER_ERROR, "state_busy"),
            Refusal::Capacity => (409, SERVER_ERROR, "state_capacity"),
            Refusal::NoPrefix => (404, INVALID_REQUEST, "state_no_prefix"),
            Refusal::Unsupported(_) => (409, INVALID_REQUEST, "state_unsupported"),
            Refusal::Internal(_) => (500, SERVER_ERROR, "state_internal"),
        }
    }

    pub fn code(&self) -> &'static str {
        self.parts().2
    }

    pub fn api_error(&self) -> crate::api_types::ApiError {
        let (status, kind, code) = self.parts();
        crate::api_types::ApiError::new(status, kind, code, self.to_string(), None)
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Key => write!(f, "{KEY_HEADER} must be 64 hex digits (a 32-byte key)"),
            Refusal::Malformed(why) => write!(f, "not a state blob: {why}"),
            Refusal::Library => f.write_str("state from another library build; replay"),
            Refusal::Layout => f.write_str("state layout differs from this backend's; replay"),
            Refusal::Compat => {
                f.write_str("state_compat_id differs (numerics-relevant levers); replay")
            }
            Refusal::Tier(why) => write!(f, "tier policy refuses: {why}"),
            Refusal::Oversize { bytes, max } => {
                write!(
                    f,
                    "state image of {bytes} bytes exceeds state.max_blob_bytes {max}"
                )
            }
            Refusal::WrongKey => f.write_str("the key does not open this state blob"),
            Refusal::Tampered => f.write_str("state blob failed authentication"),
            Refusal::Memlock => f.write_str("the state buffer could not be locked in memory"),
            Refusal::Busy => {
                f.write_str("the native executor stayed busy until state.busy_wait_ms; replay")
            }
            Refusal::Capacity => {
                f.write_str("no free prefix state slot, pages or arena room; replay")
            }
            Refusal::NoPrefix => f.write_str("no cached prefix of these tokens in this scope"),
            Refusal::Unsupported(why) => write!(f, "state teleport unsupported here: {why}"),
            Refusal::Internal(why) => write!(f, "native state call failed: {why}"),
        }
    }
}

impl std::error::Error for Refusal {}

/// The levers whose active implementation changes numerics (`Numerics::Changing`), in registry order.
pub fn numerics_levers(set: &LeverSet) -> Vec<(&'static str, &'static str)> {
    REGISTRY
        .iter()
        .filter_map(|l| {
            let id = set.get(l.name)?;
            l.impls
                .iter()
                .any(|i| i.id == id && i.numerics == Numerics::Changing)
                .then_some((l.name, id))
        })
        .collect()
}

/// D-T3 `state_compat_id`: SHA-256 over the library sha, the state layout and the numerics-relevant lever choices.
/// Teleport imports compare it instead of `profile_id` (preemption keeps the strict envelope check).
pub fn compat_id(library_sha256: &str, layout_id: &[u8; 32], levers: &LeverSet) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"cere state_compat_id v1\0");
    h.update(library_sha256.as_bytes());
    h.update([0]);
    h.update(layout_id);
    for (name, id) in numerics_levers(levers) {
        h.update(name.as_bytes());
        h.update([b'=']);
        h.update(id.as_bytes());
        h.update([0]);
    }
    h.finalize().into()
}

/// The client's 32-byte key, zeroed on drop; never printed.
pub struct StateKey([u8; 32]);

impl StateKey {
    pub fn new(bytes: [u8; 32]) -> StateKey {
        StateKey(bytes)
    }

    /// The key from the `X-Cere-State-Key` value: exactly 64 hex digits (either case).
    pub fn from_header(value: Option<&[u8]>) -> Result<StateKey, Refusal> {
        fn nibble(c: u8) -> Option<u8> {
            match c {
                b'0'..=b'9' => Some(c - b'0'),
                b'a'..=b'f' => Some(c - b'a' + 10),
                b'A'..=b'F' => Some(c - b'A' + 10),
                _ => None,
            }
        }
        let value = value.ok_or(Refusal::Key)?;
        if value.len() != 64 {
            return Err(Refusal::Key);
        }
        let mut key = StateKey([0; 32]);
        for (i, pair) in value.chunks_exact(2).enumerate() {
            key.0[i] = (nibble(pair[0]).ok_or(Refusal::Key)? << 4)
                | nibble(pair[1]).ok_or(Refusal::Key)?;
        }
        Ok(key)
    }

    /// The frame key and the key check under `salt`.
    fn derive(&self, salt: &[u8]) -> (LessSafeKey, [u8; KEY_CHECK_BYTES]) {
        struct Len(usize);
        impl hkdf::KeyType for Len {
            fn len(&self) -> usize {
                self.0
            }
        }
        let prk = hkdf::Salt::new(hkdf::HKDF_SHA256, salt).extract(&self.0);
        let mut aead = [0u8; 32];
        let mut check = [0u8; KEY_CHECK_BYTES];
        // HKDF-Expand fails only for an output longer than 255 hash lengths.
        let aead_info = [AEAD_INFO];
        let check_info = [CHECK_INFO];
        prk.expand(&aead_info, Len(32))
            .and_then(|okm| okm.fill(&mut aead))
            .expect("32-byte HKDF output");
        prk.expand(&check_info, Len(KEY_CHECK_BYTES))
            .and_then(|okm| okm.fill(&mut check))
            .expect("16-byte HKDF output");
        let key = LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &aead).expect("32-byte AES key"));
        wipe(&mut aead);
        (key, check)
    }
}

impl Drop for StateKey {
    fn drop(&mut self) {
        wipe(&mut self.0);
    }
}

impl fmt::Debug for StateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StateKey(..)")
    }
}

fn wipe(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        // SAFETY: `b` is a valid, aligned byte; volatile so the wipe is not elided.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// The blob header: the AAD of every frame (4.3). Digests are lowercase hex.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobHeader {
    pub format: u32,
    pub library_sha256: String,
    /// The exporter's profile; informational (D-T3: the compat id admits).
    pub profile_id: String,
    pub layout_id: String,
    pub state_compat_id: String,
    pub image_sha256: String,
    pub image_bytes: u64,
    /// The cached prefix depth the image holds.
    pub tokens: u64,
    pub source_tier: Tier,
    pub allowed_tiers: Vec<Tier>,
    pub created_unix_ms: u64,
    pub frame_bytes: u64,
    pub salt: String,
    pub key_check: String,
}

fn frames(image_bytes: u64) -> u64 {
    image_bytes.div_ceil(FRAME_BYTES as u64)
}

impl BlobHeader {
    /// Bytes of the whole blob: prefix, header, image and one tag per frame.
    pub fn blob_bytes(&self, header_len: usize) -> u64 {
        (PREFIX_BYTES + header_len) as u64
            + self.image_bytes
            + frames(self.image_bytes) * TAG_BYTES as u64
    }

    fn check_shape(&self) -> Result<(), Refusal> {
        let digest = |field: &str, value: &str, len: usize| match unhex(value) {
            Some(bytes) if bytes.len() == len => Ok(()),
            _ => Err(Refusal::Malformed(format!(
                "{field} is not {len} hex bytes"
            ))),
        };
        if self.format != FORMAT {
            return Err(Refusal::Malformed(format!("format {}", self.format)));
        }
        if self.frame_bytes != FRAME_BYTES as u64 {
            return Err(Refusal::Malformed(format!(
                "frame_bytes {}",
                self.frame_bytes
            )));
        }
        if self.image_bytes == 0 || self.allowed_tiers.is_empty() {
            return Err(Refusal::Malformed(
                "empty image or allowed_tiers".to_owned(),
            ));
        }
        digest("library_sha256", &self.library_sha256, 32)?;
        digest("layout_id", &self.layout_id, 32)?;
        digest("state_compat_id", &self.state_compat_id, 32)?;
        digest("image_sha256", &self.image_sha256, 32)?;
        digest("salt", &self.salt, SALT_BYTES)?;
        digest("key_check", &self.key_check, KEY_CHECK_BYTES)
    }
}

/// What the importing backend admits a blob against.
pub struct Local<'a> {
    pub identity: &'a StateIdentity,
    pub compat_id: [u8; 32],
    pub tier: Tier,
    pub max_blob_bytes: u64,
}

/// The policy checks on a parsed header, before any key use or allocation: size, library, layout, compat id, tier.
pub fn admit(header: &BlobHeader, local: &Local) -> Result<(), Refusal> {
    if header.image_bytes > local.max_blob_bytes {
        return Err(Refusal::Oversize {
            bytes: header.image_bytes,
            max: local.max_blob_bytes,
        });
    }
    if header.library_sha256 != local.identity.library_sha256 {
        return Err(Refusal::Library);
    }
    if header.layout_id != hex(&local.identity.layout_id) {
        return Err(Refusal::Layout);
    }
    if header.state_compat_id != hex(&local.compat_id) {
        return Err(Refusal::Compat);
    }
    if !allows(&header.allowed_tiers, local.tier) {
        return Err(Refusal::Tier(format!(
            "this install is tier {}, the blob allows {}",
            local.tier.as_str(),
            tier_list(&header.allowed_tiers)
        )));
    }
    Ok(())
}

pub fn tier_list(tiers: &[Tier]) -> String {
    tiers
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

/// The exporter's side of a header.
pub struct SealMeta {
    pub library_sha256: String,
    pub profile_id: String,
    pub layout_id: [u8; 32],
    pub compat_id: [u8; 32],
    pub tokens: u64,
    pub source_tier: Tier,
    pub allowed_tiers: Vec<Tier>,
    pub created_unix_ms: u64,
}

fn nonce(index: u64) -> Nonce {
    let mut n = [0u8; NONCE_LEN];
    n[4..].copy_from_slice(&index.to_be_bytes());
    Nonce::assume_unique_for_key(n)
}

fn frame_aad(prefix: &[u8], index: u64, last: bool) -> Vec<u8> {
    let mut aad = Vec::with_capacity(prefix.len() + 9);
    aad.extend_from_slice(prefix);
    aad.extend_from_slice(&index.to_le_bytes());
    aad.push(u8::from(last));
    aad
}

/// Seals an image frame by frame, in place in its [`LockedBuf`]: [`Sealer::next_chunk`] yields the prefix and header,
/// then each frame's ciphertext and tag.
pub struct Sealer {
    key: LessSafeKey,
    /// Magic, header length and header bytes: the first chunk and every frame's AAD.
    prefix: Vec<u8>,
    header: BlobHeader,
    image: LockedBuf,
    /// The next frame; `None` before the prefix went out.
    next: Option<u64>,
}

impl Sealer {
    pub fn new(key: &StateKey, meta: SealMeta, image: LockedBuf) -> Result<Sealer, Refusal> {
        if image.is_empty() {
            return Err(Refusal::Internal("empty state image".to_owned()));
        }
        let mut salt = [0u8; SALT_BYTES];
        SystemRandom::new()
            .fill(&mut salt)
            .map_err(|_| Refusal::Internal("no system randomness".to_owned()))?;
        let (aead, check) = key.derive(&salt);
        let header = BlobHeader {
            format: FORMAT,
            library_sha256: meta.library_sha256,
            profile_id: meta.profile_id,
            layout_id: hex(&meta.layout_id),
            state_compat_id: hex(&meta.compat_id),
            image_sha256: hex(&Sha256::digest(image.as_slice())),
            image_bytes: image.len() as u64,
            tokens: meta.tokens,
            source_tier: meta.source_tier,
            allowed_tiers: meta.allowed_tiers,
            created_unix_ms: meta.created_unix_ms,
            frame_bytes: FRAME_BYTES as u64,
            salt: hex(&salt),
            key_check: hex(&check),
        };
        let json = serde_json::to_vec(&header).map_err(|e| Refusal::Internal(e.to_string()))?;
        let mut prefix = Vec::with_capacity(PREFIX_BYTES + json.len());
        prefix.extend_from_slice(MAGIC);
        prefix.extend_from_slice(&(json.len() as u32).to_le_bytes());
        prefix.extend_from_slice(&json);
        Ok(Sealer {
            key: aead,
            prefix,
            header,
            image,
            next: None,
        })
    }

    pub fn header(&self) -> &BlobHeader {
        &self.header
    }

    pub fn blob_bytes(&self) -> u64 {
        self.header.blob_bytes(self.prefix.len() - PREFIX_BYTES)
    }

    /// The next piece of the blob; `None` after the last frame.
    pub fn next_chunk(&mut self) -> Option<Vec<u8>> {
        let Some(index) = self.next else {
            self.next = Some(0);
            return Some(self.prefix.clone());
        };
        let total = frames(self.header.image_bytes);
        if index >= total {
            return None;
        }
        let start = index as usize * FRAME_BYTES;
        let end = (start + FRAME_BYTES).min(self.image.len());
        let aad = frame_aad(&self.prefix, index, index + 1 == total);
        let frame = &mut self.image.as_mut_slice()[start..end];
        let tag = self
            .key
            .seal_in_place_separate_tag(nonce(index), Aad::from(&aad), frame)
            .expect("an AES-GCM frame far below the per-nonce limit");
        let mut chunk = Vec::with_capacity(frame.len() + TAG_BYTES);
        chunk.extend_from_slice(frame);
        chunk.extend_from_slice(tag.as_ref());
        self.next = Some(index + 1);
        Some(chunk)
    }
}

/// Reads a blob's prefix and header from the first bytes of a stream.
#[derive(Default)]
pub struct HeaderReader {
    buf: Vec<u8>,
}

/// A parsed header, its raw prefix (the frames' AAD) and the stream bytes after it.
pub struct Parsed {
    pub header: BlobHeader,
    pub prefix: Vec<u8>,
    pub rest: Vec<u8>,
}

impl HeaderReader {
    /// Feeds `chunk`; `Some` once the header is complete.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Option<Parsed>, Refusal> {
        self.buf.extend_from_slice(chunk);
        if self.buf.len() < PREFIX_BYTES {
            return Ok(None);
        }
        if &self.buf[..8] != MAGIC {
            return Err(Refusal::Malformed("bad magic".to_owned()));
        }
        let len = u32::from_le_bytes(self.buf[8..12].try_into().expect("4 bytes")) as usize;
        if len == 0 || len > MAX_HEADER_BYTES {
            return Err(Refusal::Malformed(format!("header length {len}")));
        }
        if self.buf.len() < PREFIX_BYTES + len {
            return Ok(None);
        }
        let header: BlobHeader =
            serde_json::from_slice(&self.buf[PREFIX_BYTES..PREFIX_BYTES + len])
                .map_err(|e| Refusal::Malformed(format!("header: {e}")))?;
        header.check_shape()?;
        let rest = self.buf.split_off(PREFIX_BYTES + len);
        Ok(Some(Parsed {
            header,
            prefix: std::mem::take(&mut self.buf),
            rest,
        }))
    }
}

/// Opens an admitted blob frame by frame into one [`LockedBuf`].
pub struct Opener {
    key: LessSafeKey,
    prefix: Vec<u8>,
    image: LockedBuf,
    image_sha256: String,
    frames: u64,
    /// The frame being received and its bytes received so far (ciphertext into `image`, then the tag).
    index: u64,
    got: usize,
    tag: [u8; TAG_BYTES],
}

impl Opener {
    /// Checks the key against the header (wrong key) and locks the image buffer (memlock).
    pub fn new(key: &StateKey, parsed: Parsed) -> Result<(Opener, Vec<u8>), Refusal> {
        let Parsed {
            header,
            prefix,
            rest,
        } = parsed;
        let salt = unhex(&header.salt).ok_or(Refusal::Tampered)?;
        let (aead, check) = key.derive(&salt);
        // Not secret-dependent timing worth hiding: the check value is public in the header.
        if hex(&check) != header.key_check {
            return Err(Refusal::WrongKey);
        }
        let len = usize::try_from(header.image_bytes).map_err(|_| Refusal::Memlock)?;
        let image = LockedBuf::new(len).map_err(|_| Refusal::Memlock)?;
        Ok((
            Opener {
                key: aead,
                prefix,
                image,
                image_sha256: header.image_sha256,
                frames: frames(header.image_bytes),
                index: 0,
                got: 0,
                tag: [0; TAG_BYTES],
            },
            rest,
        ))
    }

    fn frame_len(&self) -> usize {
        let start = self.index as usize * FRAME_BYTES;
        (self.image.len() - start).min(FRAME_BYTES)
    }

    /// Feeds the next stream bytes; a frame that fails to open, or bytes past the last frame, refuse.
    pub fn push(&mut self, mut chunk: &[u8]) -> Result<(), Refusal> {
        while !chunk.is_empty() {
            if self.index >= self.frames {
                return Err(Refusal::Tampered);
            }
            let len = self.frame_len();
            let start = self.index as usize * FRAME_BYTES;
            if self.got < len {
                let n = (len - self.got).min(chunk.len());
                self.image.as_mut_slice()[start + self.got..start + self.got + n]
                    .copy_from_slice(&chunk[..n]);
                self.got += n;
                chunk = &chunk[n..];
                continue;
            }
            let at = self.got - len;
            let n = (TAG_BYTES - at).min(chunk.len());
            self.tag[at..at + n].copy_from_slice(&chunk[..n]);
            self.got += n;
            chunk = &chunk[n..];
            if self.got == len + TAG_BYTES {
                let aad = frame_aad(&self.prefix, self.index, self.index + 1 == self.frames);
                let frame = &mut self.image.as_mut_slice()[start..start + len];
                self.key
                    .open_in_place_separate_tag(
                        nonce(self.index),
                        Aad::from(&aad),
                        Tag::from(self.tag),
                        frame,
                        0..,
                    )
                    .map_err(|_| Refusal::Tampered)?;
                self.index += 1;
                self.got = 0;
            }
        }
        Ok(())
    }

    /// The opened image once every frame arrived and its SHA-256 equals the header's.
    pub fn finish(self) -> Result<LockedBuf, Refusal> {
        if self.index != self.frames || self.got != 0 {
            return Err(Refusal::Tampered);
        }
        if hex(&Sha256::digest(self.image.as_slice())) != self.image_sha256 {
            return Err(Refusal::Tampered);
        }
        Ok(self.image)
    }
}

/// Pause between two BUSY attempts of a prefix-cache call.
const BUSY_RETRY: Duration = Duration::from_millis(20);

fn driver_refusal(e: DriverError) -> Refusal {
    match (e.status(), &e) {
        (Some(StatusError::Busy), _) => Refusal::Busy,
        (Some(StatusError::Capacity), _) => Refusal::Capacity,
        (Some(StatusError::Unsupported), _) => Refusal::Unsupported(e.to_string()),
        (_, DriverError::Missing(call)) => {
            Refusal::Unsupported(format!("the library predates {call}"))
        }
        _ => Refusal::Internal(e.to_string()),
    }
}

/// Runs `call` until it is not BUSY (the library refuses prefix-cache export/import while any ticket is in flight)
/// or `deadline` passes; the last BUSY is returned then.
fn until_idle<T>(
    deadline: Instant,
    mut call: impl FnMut() -> Result<T, DriverError>,
) -> Result<T, DriverError> {
    loop {
        match call() {
            Err(e) if e.status() == Some(StatusError::Busy) => {
                let now = Instant::now();
                if now >= deadline {
                    return Err(e);
                }
                std::thread::sleep(BUSY_RETRY.min(deadline - now));
            }
            other => return other,
        }
    }
}

/// `cere_prefix_cache_export`: the deepest cached state of `tokens` under `namespace` copied into a locked buffer,
/// with its token count. The arena snapshot is released on every path. Blocking (driver calls).
pub fn export_image(
    driver: &Driver,
    model: &Model,
    identity: &StateIdentity,
    namespace: [u8; 16],
    tokens: &[u32],
    max_blob_bytes: u64,
    deadline: Instant,
) -> Result<(LockedBuf, u64), Refusal> {
    let snapshot = until_idle(deadline, || driver.export_prefix(model, namespace, tokens))
        .map_err(|e| match e.status() {
            Some(StatusError::Argument) => Refusal::NoPrefix,
            _ => driver_refusal(e),
        })?;
    let copied = (|| {
        let info = driver.snapshot_info(&snapshot).map_err(driver_refusal)?;
        if info.layout_id != identity.layout_id {
            return Err(Refusal::Layout);
        }
        if info.bytes > max_blob_bytes {
            return Err(Refusal::Oversize {
                bytes: info.bytes,
                max: max_blob_bytes,
            });
        }
        let len = usize::try_from(info.bytes).map_err(|_| Refusal::Memlock)?;
        let mut image = LockedBuf::new(len).map_err(|_| Refusal::Memlock)?;
        driver
            .read_snapshot(&snapshot, 0, image.as_mut_slice())
            .map_err(driver_refusal)?;
        Ok((image, info.token_count))
    })();
    let released = driver.release_snapshot(snapshot);
    let out = copied?;
    released.map_err(driver_refusal)?;
    Ok(out)
}

/// `cere_snapshot_import` then `cere_prefix_cache_import` of an opened image under `namespace`; the arena snapshot
/// is released on every path. Returns the imported prefix's token count. Blocking (driver calls).
pub fn import_image(
    driver: &Driver,
    model: &Model,
    identity: &StateIdentity,
    namespace: [u8; 16],
    image: &LockedBuf,
    deadline: Instant,
) -> Result<u64, Refusal> {
    if image.as_slice().get(AT_LAYOUT..AT_LAYOUT + 32) != Some(&identity.layout_id[..]) {
        return Err(Refusal::Layout);
    }
    let snapshot = until_idle(deadline, || driver.import_snapshot(model, image.as_slice()))
        .map_err(|e| match e.status() {
            // The library validates the image against the model (D14) after our layout check passed.
            Some(StatusError::Argument) => Refusal::Layout,
            _ => driver_refusal(e),
        })?;
    let imported = (|| {
        let info = driver.snapshot_info(&snapshot).map_err(driver_refusal)?;
        until_idle(deadline, || {
            driver.import_prefix(model, namespace, &snapshot)
        })
        .map_err(driver_refusal)?;
        Ok(info.token_count)
    })();
    let released = driver.release_snapshot(snapshot);
    let tokens = imported?;
    released.map_err(driver_refusal)?;
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levers::LeverSet;

    fn identity() -> StateIdentity {
        StateIdentity {
            library_sha256: "ab".repeat(32),
            profile_id: "p".to_owned(),
            layout_id: [7; 32],
        }
    }

    fn image(len: usize) -> LockedBuf {
        let mut b = LockedBuf::new(len).unwrap();
        for (i, x) in b.as_mut_slice().iter_mut().enumerate() {
            *x = (i * 31 % 251) as u8;
        }
        b
    }

    fn meta(compat: [u8; 32]) -> SealMeta {
        let id = identity();
        SealMeta {
            library_sha256: id.library_sha256,
            profile_id: id.profile_id,
            layout_id: id.layout_id,
            compat_id: compat,
            tokens: 64,
            source_tier: Tier::Own,
            allowed_tiers: vec![Tier::Own],
            created_unix_ms: 1,
        }
    }

    fn seal(key: &StateKey, len: usize) -> Vec<u8> {
        let mut s = Sealer::new(key, meta([9; 32]), image(len)).unwrap();
        let mut blob = Vec::new();
        while let Some(c) = s.next_chunk() {
            blob.extend_from_slice(&c);
        }
        assert_eq!(blob.len() as u64, s.blob_bytes());
        blob
    }

    fn open(key: &StateKey, blob: &[u8], step: usize) -> Result<LockedBuf, Refusal> {
        let mut reader = HeaderReader::default();
        let mut chunks = blob.chunks(step);
        let parsed = loop {
            let c = chunks.next().ok_or(Refusal::Tampered)?;
            if let Some(p) = reader.push(c)? {
                break p;
            }
        };
        let (mut opener, rest) = Opener::new(key, parsed)?;
        opener.push(&rest)?;
        for c in chunks {
            opener.push(c)?;
        }
        opener.finish()
    }

    #[test]
    fn round_trip_across_frames_and_chunkings() {
        let key = StateKey::new([3; 32]);
        for len in [1, FRAME_BYTES - 1, FRAME_BYTES, 2 * FRAME_BYTES + 17] {
            let blob = seal(&key, len);
            for step in [1, 333, FRAME_BYTES + TAG_BYTES, blob.len()] {
                let got = open(&key, &blob, step).unwrap();
                assert_eq!(
                    got.as_slice(),
                    image(len).as_slice(),
                    "len {len} step {step}"
                );
            }
        }
    }

    #[test]
    fn wrong_key_tamper_truncation_and_extension_refuse() {
        let key = StateKey::new([3; 32]);
        let blob = seal(&key, FRAME_BYTES + 100);
        let other = StateKey::new([4; 32]);
        assert_eq!(open(&other, &blob, 1 << 16).err(), Some(Refusal::WrongKey));
        let mut flipped = blob.clone();
        let last = flipped.len() - 40;
        flipped[last] ^= 1;
        assert_eq!(open(&key, &flipped, 1 << 16).err(), Some(Refusal::Tampered));
        let truncated = &blob[..blob.len() - 1];
        assert_eq!(
            open(&key, truncated, 1 << 16).err(),
            Some(Refusal::Tampered)
        );
        let mut extended = blob.clone();
        extended.push(0);
        assert_eq!(
            open(&key, &extended, 1 << 16).err(),
            Some(Refusal::Tampered)
        );
        // An edited header (the AAD) fails every frame though it still parses and the key check still passes.
        let mut retiered = blob.clone();
        let at = blob
            .windows(b"\"tokens\":64".len())
            .position(|w| w == b"\"tokens\":64")
            .unwrap();
        retiered[at + 9] = b'5';
        assert_eq!(
            open(&key, &retiered, 1 << 16).err(),
            Some(Refusal::Tampered)
        );
    }

    #[test]
    fn admit_refuses_by_name() {
        let id = identity();
        let levers = LeverSet::default();
        let compat = compat_id(&id.library_sha256, &id.layout_id, &levers);
        let local = Local {
            identity: &id,
            compat_id: compat,
            tier: Tier::Own,
            max_blob_bytes: 1 << 20,
        };
        let key = StateKey::new([1; 32]);
        let header = |edit: &dyn Fn(&mut SealMeta)| {
            let mut m = meta(compat);
            edit(&mut m);
            Sealer::new(&key, m, image(1000)).unwrap().header().clone()
        };
        assert_eq!(admit(&header(&|_| {}), &local), Ok(()));
        assert_eq!(
            admit(&header(&|m| m.library_sha256 = "cd".repeat(32)), &local),
            Err(Refusal::Library)
        );
        assert_eq!(
            admit(&header(&|m| m.layout_id = [8; 32]), &local),
            Err(Refusal::Layout)
        );
        assert_eq!(
            admit(&header(&|m| m.compat_id = [0; 32]), &local),
            Err(Refusal::Compat)
        );
        assert_eq!(
            admit(&header(&|m| m.allowed_tiers = vec![Tier::Team]), &local).map_err(|e| e.code()),
            Err("state_tier_not_allowed")
        );
        let small = Local {
            max_blob_bytes: 999,
            ..local
        };
        assert_eq!(
            admit(&header(&|_| {}), &small),
            Err(Refusal::Oversize {
                bytes: 1000,
                max: 999
            })
        );
    }

    #[test]
    fn compat_id_ignores_profile_and_bit_identical_levers_only() {
        let id = identity();
        let base = LeverSet::default();
        let a = compat_id(&id.library_sha256, &id.layout_id, &base);
        assert_eq!(a, compat_id(&id.library_sha256, &id.layout_id, &base));
        assert_ne!(a, compat_id(&"cd".repeat(32), &id.layout_id, &base));
        assert_ne!(a, compat_id(&id.library_sha256, &[8; 32], &base));
        assert!(
            numerics_levers(&base).is_empty(),
            "the defaults are bit-identical"
        );
        for lever in REGISTRY {
            for imp in lever.impls {
                let table = [(lever.name.to_owned(), imp.id.to_owned())].into();
                let Ok(set) = LeverSet::from_table(&table, 3) else {
                    continue;
                };
                let same = compat_id(&id.library_sha256, &id.layout_id, &set) == a;
                assert_eq!(
                    same,
                    imp.numerics != Numerics::Changing,
                    "{}={}",
                    lever.name,
                    imp.id
                );
            }
        }
    }

    #[test]
    fn key_header_takes_64_hex_digits() {
        assert!(StateKey::from_header(Some("0aF9".repeat(16).as_bytes())).is_ok());
        for bad in [None, Some(&b"00"[..]), Some("g".repeat(64).as_bytes())] {
            assert_eq!(StateKey::from_header(bad).err(), Some(Refusal::Key));
        }
        assert_eq!(format!("{:?}", StateKey::new([5; 32])), "StateKey(..)");
    }
}
