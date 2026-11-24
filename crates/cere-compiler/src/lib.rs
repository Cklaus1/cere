//! `cerec` library: the M2.1 prepared-artifact path (m21 design note).
//!
//! Source headers are parsed by [`safetensors`], classified by [`families`] rules, and indexed by a canonical
//! ([`canon_json`]) v2 [`manifest`]. All file access goes through held descriptors ([`fsio`]).
pub mod artifact;
pub mod canon_json;
pub mod convert;
pub mod families;
pub mod fsio;
pub mod inventory;
pub mod manifest;
pub mod prepare;
pub mod probe;
pub mod safetensors;
pub mod source;
pub mod stamp;
pub mod token_map;
pub mod verify;

use sha2::{Digest, Sha256};
use std::fmt;

/// Exit status of a rejected input (any error class of the shared table, usage, unsafe paths).
pub const EXIT_REJECT: i32 = 2;
/// I/O failure that is not a rejection of the input.
pub const EXIT_IO: i32 = 1;

/// A `cerec` failure: an error class (the shared table's names where one applies), the process exit status and a
/// message. The CLI prints `<class>: <message>` as the first stderr line, as the Python reference tools do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub class: &'static str,
    pub exit: i32,
    pub message: String,
}

impl Error {
    pub fn reject(class: &'static str, message: impl Into<String>) -> Error {
        Error {
            class,
            exit: EXIT_REJECT,
            message: message.into(),
        }
    }

    pub fn io(what: impl fmt::Display, e: impl fmt::Display) -> Error {
        Error {
            class: "io",
            exit: EXIT_IO,
            message: format!("{what}: {e}"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class, self.message)
    }
}

impl std::error::Error for Error {}

impl From<fsio::FsError> for Error {
    fn from(e: fsio::FsError) -> Error {
        match e.kind {
            fsio::FsErrorKind::Symlink => Error::reject("symlink", e.to_string()),
            fsio::FsErrorKind::NotRegular => Error::reject("special_file", e.to_string()),
            fsio::FsErrorKind::BadPath => Error::reject("unsafe_path", e.to_string()),
            fsio::FsErrorKind::Missing => Error::reject("missing_file", e.to_string()),
            fsio::FsErrorKind::Io => Error::io(
                &e.path,
                e.source.as_ref().map_or(String::new(), |s| s.to_string()),
            ),
        }
    }
}

impl From<safetensors::HeaderError> for Error {
    fn from(e: safetensors::HeaderError) -> Error {
        Error::reject(e.class(), e.to_string())
    }
}

impl From<safetensors::ReadError> for Error {
    fn from(e: safetensors::ReadError) -> Error {
        match e {
            safetensors::ReadError::Io(io) => Error::io("reading header", io),
            safetensors::ReadError::Header(h) => h.into(),
        }
    }
}

impl From<families::FamilyError> for Error {
    fn from(e: families::FamilyError) -> Error {
        Error::reject(e.class(), e.to_string())
    }
}

impl From<token_map::TokenMapError> for Error {
    fn from(e: token_map::TokenMapError) -> Error {
        Error::reject(e.class(), e.to_string())
    }
}

/// Resource bounds (Q15). Defaults are the spec's; the shared invalid-case table overrides them per case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Largest header JSON length `n` (80× the largest real header).
    pub max_header_bytes: u64,
    pub max_tensors_per_file: u64,
    pub max_total_tensors: u64,
    pub max_files: u64,
    pub max_manifest_bytes: u64,
    pub max_token_map_pkl_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_header_bytes: 16 << 20,
            max_tensors_per_file: 1_000_000,
            max_total_tensors: 4_000_000,
            max_files: 16_384,
            max_manifest_bytes: 4 << 20,
            max_token_map_pkl_bytes: 16 << 20,
        }
    }
}

impl Limits {
    /// The defaults with the overrides of a JSON object `{name: integer}` (a table case's `limits`).
    pub fn with_overrides(overrides: &serde_json::Value) -> Result<Limits, String> {
        let mut l = Limits::default();
        let obj = overrides.as_object().ok_or("limits must be an object")?;
        for (k, v) in obj {
            let n = v
                .as_u64()
                .ok_or_else(|| format!("limit {k} must be a non-negative integer"))?;
            let slot = match k.as_str() {
                "max_header_bytes" => &mut l.max_header_bytes,
                "max_tensors_per_file" => &mut l.max_tensors_per_file,
                "max_total_tensors" => &mut l.max_total_tensors,
                "max_files" => &mut l.max_files,
                "max_manifest_bytes" => &mut l.max_manifest_bytes,
                "max_token_map_pkl_bytes" => &mut l.max_token_map_pkl_bytes,
                _ => return Err(format!("unknown limit {k}")),
            };
            *slot = n;
        }
        Ok(l)
    }

    pub fn from_json(text: &[u8]) -> Result<Limits, String> {
        let v = canon_json::parse_strict(text).map_err(|e| e.to_string())?;
        Limits::with_overrides(&v)
    }
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 15)] as char);
    }
    out
}

/// Hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// SHA-256 over `sha256sum` lines `<hex>  <name>\n` in the given order (callers pass bytewise name order), the
/// definition of the lock digests (docs/CONFIGURATION.md:196-205) and of the Q12 header digest.
pub fn sha256sum_lines_digest<'a>(lines: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut h = Sha256::new();
    for (hex_sha, name) in lines {
        h.update(hex_sha.as_bytes());
        h.update(b"  ");
        h.update(name.as_bytes());
        h.update(b"\n");
    }
    hex(&h.finalize())
}
