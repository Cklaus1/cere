//! U3 (m211 design note D3, D9): a session's state as bytes that leave one executor and enter another (the M4
//! hand-off). An export is a `CERESNP1` image in an [`Envelope`] {library_sha256, profile_id, layout_id,
//! sha256(bytes)}; import refuses any mismatch before the library sees the bytes. The image lives in a
//! [`LockedBuf`]: `mlock`ed (never pageable, so session state cannot reach swap) and zeroed on drop.
use std::fmt;

use sha2::{Digest, Sha256};

use super::driver::{Driver, DriverError, Model, ModelConfig, Session, Snapshot};
use super::ffi::DTYPE_F8_E4M3;

/// `CERESNP1` header grammar version (`snapshot_format.h` `kFormat`).
const FORMAT: u32 = 1;
/// Section kinds this build writes, ascending (`snapshot_format.h` `kWritableKinds`: 0-8, kind 8 the sampler counts
/// (CR13), and 11, the rope delta (M2.13 CR14)).
const WRITABLE_KINDS: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11];

/// `layout_id` of a model config (D3): SHA-256 over `format`, the config struct bytes (`cere_qwen4_exp_config`; on
/// family sdar `cere_sdar_config`, whose snapshots are refused, ST4 S3), the KV dtype FP8 E4M3 and the sorted kinds
/// this build writes, all integers little-endian u32 (`snapshot_format.cpp` `layout_id`). Equal to
/// `cere_snapshot_info.layout_id` of every snapshot of a model with this config.
pub fn layout_id(config: &ModelConfig) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(FORMAT.to_le_bytes());
    h.update(config.bytes());
    h.update(DTYPE_F8_E4M3.to_le_bytes());
    h.update((WRITABLE_KINDS.len() as u32).to_le_bytes());
    for kind in WRITABLE_KINDS {
        h.update(kind.to_le_bytes());
    }
    h.finalize().into()
}

/// Who may import an export: the library build, the capability profile and the state layout of one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateIdentity {
    /// SHA-256 (hex) of `libcere_executor.so` (`deployments.native.library_sha256`).
    pub library_sha256: String,
    /// The capability profile id the backend serves.
    pub profile_id: String,
    /// [`layout_id`] of the backend's model config.
    pub layout_id: [u8; 32],
}

/// What travels with an exported image (D3): the exporter's identity and the image's SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Envelope {
    pub library_sha256: String,
    pub profile_id: String,
    pub layout_id: [u8; 32],
    pub sha256: [u8; 32],
}

/// An exported session state: the envelope and the `CERESNP1` image.
pub struct Exported {
    pub envelope: Envelope,
    pub bytes: LockedBuf,
}

/// Why an export or import failed. The envelope errors are raised before any library call.
#[derive(Debug)]
pub enum ExportError {
    /// A driver call failed.
    Driver(DriverError),
    /// `mlock` of the byte buffer failed; there is no pageable fallback (D9).
    Lock(std::io::Error),
    /// The image's SHA-256 differs from the envelope's.
    Checksum,
    /// The envelope names another library build.
    Library { expected: String, found: String },
    /// The envelope names another capability profile.
    Profile { expected: String, found: String },
    /// The envelope (or the image header) names another state layout.
    Layout,
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportError::Driver(e) => write!(f, "{e}"),
            ExportError::Lock(e) => write!(f, "mlock of the state buffer failed: {e}"),
            ExportError::Checksum => f.write_str("state image sha256 differs from its envelope"),
            ExportError::Library { expected, found } => {
                write!(
                    f,
                    "state from library {found}, this backend runs {expected}"
                )
            }
            ExportError::Profile { expected, found } => {
                write!(
                    f,
                    "state from profile {found}, this backend serves {expected}"
                )
            }
            ExportError::Layout => f.write_str("state layout_id differs from this backend's"),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<DriverError> for ExportError {
    fn from(e: DriverError) -> Self {
        ExportError::Driver(e)
    }
}

/// A zero-initialized byte buffer locked in RAM (`mlock`) for its whole life and zeroed before it is unlocked
/// and freed (D9).
pub struct LockedBuf {
    bytes: Box<[u8]>,
}

impl LockedBuf {
    /// `len` zero bytes, locked; `mlock` failure is an error, never a pageable buffer.
    pub fn new(len: usize) -> Result<LockedBuf, std::io::Error> {
        let bytes = vec![0u8; len].into_boxed_slice();
        if len > 0 {
            // SAFETY: the range is the live allocation `bytes`.
            if unsafe { libc::mlock(bytes.as_ptr().cast(), len) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(LockedBuf { bytes })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl Drop for LockedBuf {
    fn drop(&mut self) {
        for b in self.bytes.iter_mut() {
            // SAFETY: `b` is a valid, aligned byte of the buffer; volatile so the wipe is not elided.
            unsafe { std::ptr::write_volatile(b, 0) };
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
        if !self.bytes.is_empty() {
            // SAFETY: the range was locked by `new` and is still allocated.
            unsafe { libc::munlock(self.bytes.as_ptr().cast(), self.bytes.len()) };
        }
    }
}

impl fmt::Debug for LockedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LockedBuf({} bytes)", self.bytes.len())
    }
}

/// Exports the idle `session`: snapshot, copy the image into a [`LockedBuf`], release the snapshot, wrap it in
/// `identity`'s envelope. The image's layout must be `identity`'s.
pub fn export(
    driver: &Driver,
    session: &Session,
    identity: &StateIdentity,
) -> Result<Exported, ExportError> {
    let snapshot = driver.snapshot(session)?;
    let copied = copy_out(driver, &snapshot, identity);
    let released = driver.release_snapshot(snapshot);
    let bytes = copied?;
    released?;
    Ok(Exported {
        envelope: Envelope {
            library_sha256: identity.library_sha256.clone(),
            profile_id: identity.profile_id.clone(),
            layout_id: identity.layout_id,
            sha256: Sha256::digest(bytes.as_slice()).into(),
        },
        bytes,
    })
}

fn copy_out(
    driver: &Driver,
    snapshot: &Snapshot,
    identity: &StateIdentity,
) -> Result<LockedBuf, ExportError> {
    let info = driver.snapshot_info(snapshot)?;
    if info.layout_id != identity.layout_id {
        return Err(ExportError::Layout);
    }
    let len = usize::try_from(info.bytes)
        .map_err(|_| ExportError::Lock(std::io::Error::from(std::io::ErrorKind::OutOfMemory)))?;
    let mut bytes = LockedBuf::new(len).map_err(ExportError::Lock)?;
    driver.read_snapshot(snapshot, 0, bytes.as_mut_slice())?;
    Ok(bytes)
}

/// `CERESNP1` header offset of `layout_id` (`snapshot_format.cpp` `kAtLayout`).
pub(crate) const AT_LAYOUT: usize = 24;

/// Checks `exported` against `identity` (checksum, library, profile, layout in the envelope and in the image
/// header) and only then imports the image into `model`'s arena, restores it into a fresh session of
/// `max_context_tokens` and releases the snapshot. The returned session continues where the exporter's stopped.
pub fn import(
    driver: &Driver,
    model: &Model,
    max_context_tokens: u64,
    identity: &StateIdentity,
    exported: &Exported,
) -> Result<Session, ExportError> {
    let env = &exported.envelope;
    let bytes = exported.bytes.as_slice();
    if <[u8; 32]>::from(Sha256::digest(bytes)) != env.sha256 {
        return Err(ExportError::Checksum);
    }
    if env.library_sha256 != identity.library_sha256 {
        return Err(ExportError::Library {
            expected: identity.library_sha256.clone(),
            found: env.library_sha256.clone(),
        });
    }
    if env.profile_id != identity.profile_id {
        return Err(ExportError::Profile {
            expected: identity.profile_id.clone(),
            found: env.profile_id.clone(),
        });
    }
    if env.layout_id != identity.layout_id
        || bytes.get(AT_LAYOUT..AT_LAYOUT + 32) != Some(&identity.layout_id[..])
    {
        return Err(ExportError::Layout);
    }
    let snapshot = driver.import_snapshot(model, bytes)?;
    let restored =
        driver
            .create_session(model, max_context_tokens)
            .and_then(|session| match driver.restore(&session, &snapshot) {
                Ok(()) => Ok(session),
                Err(e) => {
                    let _ = driver.release_session(session);
                    Err(e)
                }
            });
    let released = driver.release_snapshot(snapshot);
    let session = restored?;
    if let Err(e) = released {
        let _ = driver.release_session(session);
        return Err(e.into());
    }
    Ok(session)
}
