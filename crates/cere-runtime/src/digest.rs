//! Keyed digests (m1 design note Q3).
//!
//! A 32-byte server secret is persisted once as `<state_dir>/server_secret` (hex, mode 0600, directory 0700) and
//! never rotated by the runtime: idempotency digests and session prefix digests must re-verify after a restart.
//! Every scope gets its own HMAC-SHA256 key derived from that secret, so equal content in two scopes yields
//! unrelated digests. Digests are sensitive (SPEC §12): `Debug` prints only a short prefix.
//!
//! Inputs are framed (domain tag, then length-prefixed fields) so no two distinct inputs share an encoding.
//! JSON is canonicalised (object keys sorted, no whitespace) independently of serde_json's map ordering.

use hmac::{Hmac, Mac};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha2::Sha256;
use std::fmt;
use std::fs::{File, OpenOptions, Permissions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

use crate::journal::{ensure_private_dir, sync_dir};

type HmacSha256 = Hmac<Sha256>;

/// File name of the persisted server secret inside `ledger.state_dir`.
pub const SECRET_FILE: &str = "server_secret";

/// A 32-byte keyed digest. Equality is constant-time; serialized as 64 lowercase hex characters.
#[derive(Clone, Copy, Eq)]
pub struct Digest([u8; 32]);

impl Digest {
    pub fn to_hex(&self) -> String {
        hex(&self.0)
    }

    pub fn from_hex(text: &str) -> Option<Self> {
        let bytes = unhex(text)?;
        Some(Self(bytes.try_into().ok()?))
    }
}

impl PartialEq for Digest {
    fn eq(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest({}…)", &self.to_hex()[..8])
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Digest::from_hex(&text)
            .ok_or_else(|| serde::de::Error::custom("digest must be 64 hex chars"))
    }
}

/// The persisted server secret. `Debug` never prints it.
pub struct DigestKeys {
    secret: [u8; 32],
}

impl fmt::Debug for DigestKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DigestKeys(<redacted>)")
    }
}

impl DigestKeys {
    /// Loads `<state_dir>/server_secret`, creating it on first start. Creation never exposes a partial file
    /// (review finding 11): the hex is written to a unique 0600 temp file and fsync'd, then hard-linked into
    /// place (which fails if the secret already exists, so a concurrent first start adopts the winner's secret),
    /// the temp name is removed and the directory fsync'd. A crash mid-creation can leave only an unused temp
    /// file, and only before the first secret exists. A secret file that is not exactly 64 hex characters is an
    /// error, never silently replaced.
    pub fn open(state_dir: &Path) -> io::Result<Self> {
        ensure_private_dir(state_dir)?;
        let secret = open_secret(state_dir, SECRET_FILE)?;
        Ok(Self { secret })
    }

    /// The HMAC key of one auth scope.
    pub fn scope_key(&self, scope_id: &str) -> ScopeKey {
        let mut mac = new_mac(&self.secret);
        frame(&mut mac, b"cere-scope-key-v1");
        frame(&mut mac, scope_id.as_bytes());
        ScopeKey {
            key: mac.finalize().into_bytes().into(),
        }
    }

    /// Q2: the SGLang `cache_salt` of one auth scope, sent on its backend chat calls under
    /// `prefix_cache_isolation = "cache_salt"`: the first 32 hex characters of
    /// HMAC-SHA256(secret, `"cere-cache-salt\0"` ‖ scope_id). Stable across restarts (the secret is), distinct
    /// per scope, and not derivable from the scope id without the secret.
    pub fn cache_salt(&self, scope_id: &str) -> String {
        let mut mac = new_mac(&self.secret);
        mac.update(b"cere-cache-salt\0");
        mac.update(scope_id.as_bytes());
        // 32 hex characters = the first 16 bytes of the MAC.
        hex(&mac.finalize().into_bytes()[..16])
    }
}

/// Loads the 32-byte hex secret `state_dir/file_name`, creating it on first use as [`DigestKeys::open`] describes
/// (also the agent operation journal's secret), then fsyncs the directory. `state_dir` must already be private.
pub(crate) fn open_secret(state_dir: &Path, file_name: &str) -> io::Result<[u8; 32]> {
    let path = state_dir.join(file_name);
    let secret = match read_secret(&path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            create_secret(state_dir, file_name, &path)?
        }
        other => other?,
    };
    sync_dir(state_dir)?;
    Ok(secret)
}

/// Reads an existing secret file (tightening it to 0600).
fn read_secret(path: &Path) -> io::Result<[u8; 32]> {
    std::fs::set_permissions(path, Permissions::from_mode(0o600))?;
    let mut text = String::new();
    File::open(path)?.read_to_string(&mut text)?;
    unhex(text.trim_end())
        .and_then(|b| <[u8; 32]>::try_from(b).ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} is not a 32-byte hex secret", path.display()),
            )
        })
}

/// Publishes a new secret at `path` atomically; if another process published first, returns its secret.
fn create_secret(state_dir: &Path, file_name: &str, path: &Path) -> io::Result<[u8; 32]> {
    let secret = random_bytes::<32>()?;
    let tmp = state_dir.join(format!(
        "{file_name}.tmp.{}.{}",
        std::process::id(),
        hex(&random_bytes::<8>()?)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp)?;
    let published = file
        .write_all(hex(&secret).as_bytes())
        .and_then(|()| file.sync_all())
        .and_then(|()| std::fs::hard_link(&tmp, path));
    drop(file);
    std::fs::remove_file(&tmp)?;
    match published {
        Ok(()) => Ok(secret),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => read_secret(path),
        Err(e) => Err(e),
    }
}

/// A per-scope HMAC-SHA256 key. `Debug` never prints it.
#[derive(Clone)]
pub struct ScopeKey {
    key: [u8; 32],
}

impl fmt::Debug for ScopeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ScopeKey(<redacted>)")
    }
}

/// Everything Q3 binds into a request digest. `request` is the normalized request body (explicit output budget,
/// approved fields only); it is only fed to the HMAC and never stored.
#[derive(Debug, Clone, Copy)]
pub struct RequestDigestInput<'a> {
    pub request: &'a Value,
    pub session_id: Option<&'a str>,
    pub expected_revision: Option<u64>,
    pub served_model: &'a str,
    /// The effective (clamped) deadline budget in milliseconds.
    pub deadline_ms: u64,
    pub config_hash: &'a str,
    /// The API surface (M2.14): `None` for chat, so chat digests keep their pre-M2.14 bytes; `Some("anthropic")`
    /// or `Some("responses")` otherwise, so a key reused across surfaces is `idempotency_conflict`.
    pub surface: Option<&'a str>,
}

impl ScopeKey {
    /// Idempotency digest of a request (Q3): equal only when every semantically relevant input is equal.
    pub fn request_digest(&self, input: &RequestDigestInput<'_>) -> Digest {
        let mut mac = new_mac(&self.key);
        frame(&mut mac, b"cere-request-v1");
        frame(&mut mac, canonical_json(input.request).as_bytes());
        frame_opt(&mut mac, input.session_id.map(str::as_bytes));
        frame_opt(
            &mut mac,
            input
                .expected_revision
                .map(u64::to_be_bytes)
                .as_ref()
                .map(|b| &b[..]),
        );
        frame(&mut mac, input.served_model.as_bytes());
        frame(&mut mac, &input.deadline_ms.to_be_bytes());
        frame(&mut mac, input.config_hash.as_bytes());
        if let Some(surface) = input.surface {
            frame(&mut mac, b"surface");
            frame(&mut mac, surface.as_bytes());
        }
        Digest(mac.finalize().into_bytes().into())
    }

    /// Digest of a session's message prefix under its salt lineage (S1/S2/Q5). `messages` is the exact ordered
    /// message list (`[]` for revision 0).
    pub fn prefix_digest(&self, salt: &Salt, messages: &[Value]) -> Digest {
        let mut mac = new_mac(&self.key);
        frame(&mut mac, b"cere-session-prefix-v1");
        frame(&mut mac, &salt.0);
        frame(&mut mac, &(messages.len() as u64).to_be_bytes());
        for message in messages {
            frame(&mut mac, canonical_json(message).as_bytes());
        }
        Digest(mac.finalize().into_bytes().into())
    }
}

/// A session salt lineage: created with a root session, inherited unchanged by forks (Q5).
/// Serialized as 32 hex characters; internal state only, never returned to clients.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Salt([u8; 16]);

impl Salt {
    pub fn random() -> io::Result<Self> {
        Ok(Self(random_bytes::<16>()?))
    }
}

impl fmt::Debug for Salt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Salt(<redacted>)")
    }
}

impl Serialize for Salt {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex(&self.0))
    }
}

impl<'de> Deserialize<'de> for Salt {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        unhex(&text)
            .and_then(|b| <[u8; 16]>::try_from(b).ok())
            .map(Salt)
            .ok_or_else(|| serde::de::Error::custom("salt must be 32 hex chars"))
    }
}

/// `N` bytes from the kernel CSPRNG (`/dev/urandom`).
pub fn random_bytes<const N: usize>() -> io::Result<[u8; N]> {
    let mut out = [0u8; N];
    File::open("/dev/urandom")?.read_exact(&mut out)?;
    Ok(out)
}

/// Canonical JSON: object keys sorted by code point, no insignificant whitespace.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical(item, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn new_mac(key: &[u8]) -> HmacSha256 {
    HmacSha256::new_from_slice(key).expect("HMAC accepts any key length")
}

fn frame(mac: &mut HmacSha256, bytes: &[u8]) {
    mac.update(&(bytes.len() as u64).to_be_bytes());
    mac.update(bytes);
}

fn frame_opt(mac: &mut HmacSha256, bytes: Option<&[u8]>) {
    match bytes {
        None => mac.update(&[0]),
        Some(b) => {
            mac.update(&[1]);
            frame(mac, b);
        }
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0xf) as usize] as char);
    }
    out
}

pub(crate) fn unhex(text: &str) -> Option<Vec<u8>> {
    fn nibble(c: u8) -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        }
    }
    let bytes = text.as_bytes();
    if bytes.len() % 2 != 0 {
        return None;
    }
    bytes
        .chunks(2)
        .map(|pair| Some((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::MetadataExt;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cere-digest-{tag}-{}-{}",
            std::process::id(),
            hex(&random_bytes::<4>().unwrap())
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn input(request: &Value) -> RequestDigestInput<'_> {
        RequestDigestInput {
            request,
            session_id: None,
            expected_revision: None,
            served_model: "m",
            deadline_ms: 1000,
            config_hash: "c",
            surface: None,
        }
    }

    #[test]
    fn secret_is_persisted_private_and_stable() {
        let dir = temp_dir("secret");
        let a = DigestKeys::open(&dir).unwrap();
        let meta = std::fs::metadata(dir.join(SECRET_FILE)).unwrap();
        assert_eq!(meta.mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o777, 0o700);
        let b = DigestKeys::open(&dir).unwrap();
        let req = json!({"a": 1});
        assert_eq!(
            a.scope_key("s").request_digest(&input(&req)),
            b.scope_key("s").request_digest(&input(&req))
        );
        std::fs::write(dir.join(SECRET_FILE), "short").unwrap();
        assert!(DigestKeys::open(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Contract 2 (an internal design note H3): hex(HMAC-SHA256(secret, "cere-cache-salt\0" ‖ scope))[..32], vectors computed
    /// with Python's `hmac`; the salt survives a restart and never equals the scope id.
    #[test]
    fn cache_salt_matches_the_contract() {
        let keys = DigestKeys {
            secret: std::array::from_fn(|i| i as u8),
        };
        assert_eq!(keys.cache_salt("a"), "972bae366f5a92641531c30be7b44530");
        assert_eq!(keys.cache_salt("b"), "37967bd336f1ec958763cbaaa1c882fa");
        let dir = temp_dir("salt");
        let first = DigestKeys::open(&dir).unwrap().cache_salt("a");
        assert_eq!(DigestKeys::open(&dir).unwrap().cache_salt("a"), first);
        assert_ne!(first, "a");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Review finding 11: concurrent first starts never observe an empty or partial secret, and all agree.
    #[test]
    fn concurrent_first_open_never_sees_a_partial_secret() {
        for _ in 0..200 {
            let dir = temp_dir("race");
            let barrier = std::sync::Barrier::new(8);
            let digests: Vec<Digest> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..8)
                    .map(|_| {
                        s.spawn(|| {
                            barrier.wait();
                            DigestKeys::open(&dir)
                                .map(|k| k.scope_key("s").request_digest(&input(&json!({}))))
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap().expect("open during a concurrent create"))
                    .collect()
            });
            assert!(digests.windows(2).all(|w| w[0] == w[1]));
            let names: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .collect();
            assert_eq!(names, [SECRET_FILE], "no temp files left behind");
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn request_digest_binds_every_input_and_scope() {
        let dir = temp_dir("bind");
        let keys = DigestKeys::open(&dir).unwrap();
        let key = keys.scope_key("s");
        let req =
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5});
        let reordered: Value = serde_json::from_str(
            r#"{"max_tokens":5,"messages":[{"content":"hi","role":"user"}],"model":"m"}"#,
        )
        .unwrap();
        let base = key.request_digest(&input(&req));
        assert_eq!(base, key.request_digest(&input(&reordered)));
        let other_req =
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 6});
        let variants = [
            key.request_digest(&input(&other_req)),
            key.request_digest(&RequestDigestInput {
                session_id: Some("x"),
                ..input(&req)
            }),
            key.request_digest(&RequestDigestInput {
                expected_revision: Some(0),
                ..input(&req)
            }),
            key.request_digest(&RequestDigestInput {
                served_model: "n",
                ..input(&req)
            }),
            key.request_digest(&RequestDigestInput {
                deadline_ms: 1001,
                ..input(&req)
            }),
            key.request_digest(&RequestDigestInput {
                config_hash: "d",
                ..input(&req)
            }),
            key.request_digest(&RequestDigestInput {
                surface: Some("anthropic"),
                ..input(&req)
            }),
            key.request_digest(&RequestDigestInput {
                surface: Some("responses"),
                ..input(&req)
            }),
            keys.scope_key("t").request_digest(&input(&req)),
        ];
        let anthropic = key.request_digest(&RequestDigestInput {
            surface: Some("anthropic"),
            ..input(&req)
        });
        let responses = key.request_digest(&RequestDigestInput {
            surface: Some("responses"),
            ..input(&req)
        });
        assert_ne!(anthropic, responses);
        for v in variants {
            assert_ne!(base, v);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn prefix_digest_depends_on_salt_and_exact_messages() {
        let dir = temp_dir("prefix");
        let key = DigestKeys::open(&dir).unwrap().scope_key("s");
        let salt = Salt::random().unwrap();
        let m = vec![json!({"role": "user", "content": "a"})];
        let edited = vec![json!({"role": "user", "content": "b"})];
        assert_eq!(key.prefix_digest(&salt, &m), key.prefix_digest(&salt, &m));
        assert_ne!(
            key.prefix_digest(&salt, &m),
            key.prefix_digest(&salt, &edited)
        );
        assert_ne!(
            key.prefix_digest(&salt, &m),
            key.prefix_digest(&Salt::random().unwrap(), &m)
        );
        assert_ne!(key.prefix_digest(&salt, &[]), key.prefix_digest(&salt, &m));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn digest_and_salt_round_trip_as_hex() {
        let d = Digest([7; 32]);
        let text = serde_json::to_string(&d).unwrap();
        assert_eq!(serde_json::from_str::<Digest>(&text).unwrap(), d);
        assert!(serde_json::from_str::<Digest>("\"zz\"").is_err());
        let s = Salt::random().unwrap();
        assert_eq!(
            serde_json::from_str::<Salt>(&serde_json::to_string(&s).unwrap()).unwrap(),
            s
        );
        assert_eq!(
            canonical_json(&json!({"b": [1, {"d": null, "c": "x"}], "a": true})),
            r#"{"a":true,"b":[1,{"c":"x","d":null}]}"#
        );
    }
}
