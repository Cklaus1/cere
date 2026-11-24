//! mTLS identities and the authorization gate of the worker RPC (m35 design note Q2).
//!
//! One private cluster CA per deployment. The server requires a client certificate under that CA; the client
//! trusts only that CA and checks the server against the expected peer principal (`domain_name`). A principal is
//! the only SAN entry of the CA-verified leaf and must be a dNSName; zero or several SANs, or an IP/URI/email (or
//! any other) SAN, are refused. Caller-supplied ids are never identity: `Authz` binds principals to roles before
//! a handler touches any state.

use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use rustls_pki_types::pem::PemObject;
use rustls_pki_types::PrivateKeyDer;
use tonic::transport::{Certificate, CertificateDer, ClientTlsConfig, Identity, ServerTlsConfig};
use tonic::{Code, Request, Status};

use crate::fed::wire;

#[derive(Debug)]
pub enum FedError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    /// A PEM file without the expected block (or with a malformed one).
    Pem {
        path: PathBuf,
        reason: &'static str,
    },
    /// A certificate that is not parseable or breaks the Q2 SAN rule.
    Principal(&'static str),
    /// The own leaf names another principal than the configured one.
    PrincipalMismatch {
        expected: String,
        found: String,
    },
    /// A worker channel that cannot be built from the configured endpoint and TLS settings.
    Channel {
        endpoint: String,
        reason: String,
    },
}

impl fmt::Display for FedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FedError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            FedError::Pem { path, reason } => write!(f, "{}: {reason}", path.display()),
            FedError::Principal(reason) => write!(f, "certificate principal: {reason}"),
            FedError::PrincipalMismatch { expected, found } => write!(
                f,
                "certificate names principal {found:?}, configured principal is {expected:?}"
            ),
            FedError::Channel { endpoint, reason } => write!(f, "channel to {endpoint}: {reason}"),
        }
    }
}

impl std::error::Error for FedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FedError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The node's TLS material: cluster CA, own leaf (chain) and key, all PEM.
#[derive(Debug, Clone)]
pub struct Pki {
    ca_pem: Vec<u8>,
    cert_pem: Vec<u8>,
    key_pem: Vec<u8>,
    principal: String,
}

fn read(path: &Path) -> Result<Vec<u8>, FedError> {
    std::fs::read(path).map_err(|source| FedError::Io {
        path: path.to_owned(),
        source,
    })
}

fn pem_error(path: &Path, reason: &'static str) -> FedError {
    FedError::Pem {
        path: path.to_owned(),
        reason,
    }
}

/// The first certificate of a PEM file (`None` when it has none); a malformed PEM is an error.
fn first_certificate(path: &Path, pem: &[u8]) -> Result<Option<CertificateDer<'static>>, FedError> {
    CertificateDer::pem_slice_iter(pem)
        .next()
        .transpose()
        .map_err(|_| pem_error(path, "malformed PEM"))
}

impl Pki {
    /// Reads the PEM files and checks that the own leaf (the first certificate of `cert`) obeys the Q2 SAN rule
    /// and names `own_principal`.
    pub fn load(ca: &Path, cert: &Path, key: &Path, own_principal: &str) -> Result<Pki, FedError> {
        let ca_pem = read(ca)?;
        if first_certificate(ca, &ca_pem)?.is_none() {
            return Err(pem_error(ca, "no CERTIFICATE block"));
        }

        let cert_pem = read(cert)?;
        let leaf = first_certificate(cert, &cert_pem)?
            .ok_or_else(|| pem_error(cert, "no CERTIFICATE block"))?;
        let principal = principal_from_der(&leaf)?;
        if principal != own_principal {
            return Err(FedError::PrincipalMismatch {
                expected: own_principal.to_owned(),
                found: principal,
            });
        }

        let key_pem = read(key)?;
        // PKCS#8, PKCS#1 and SEC1 sections only: an `ENCRYPTED PRIVATE KEY` is not a key here.
        let mut private_keys = 0usize;
        for k in PrivateKeyDer::pem_slice_iter(&key_pem) {
            k.map_err(|_| pem_error(key, "malformed PEM"))?;
            private_keys += 1;
        }
        if private_keys != 1 {
            return Err(pem_error(
                key,
                "expected exactly one unencrypted PRIVATE KEY block",
            ));
        }

        Ok(Pki {
            ca_pem,
            cert_pem,
            key_pem,
            principal,
        })
    }

    /// The own principal (the leaf's single DNS SAN).
    pub fn principal(&self) -> &str {
        &self.principal
    }

    fn identity(&self) -> Identity {
        Identity::from_pem(&self.cert_pem, &self.key_pem)
    }

    /// Server side: own identity; a client certificate under the cluster CA is required.
    pub fn server_tls(&self) -> ServerTlsConfig {
        ServerTlsConfig::new()
            .identity(self.identity())
            .client_ca_root(Certificate::from_pem(&self.ca_pem))
    }

    /// Client side: trusts only the cluster CA, presents the own identity, and requires the server certificate to
    /// name `peer_principal` (never an IP).
    pub fn client_tls(&self, peer_principal: &str) -> ClientTlsConfig {
        ClientTlsConfig::new()
            .ca_certificate(Certificate::from_pem(&self.ca_pem))
            .identity(self.identity())
            .domain_name(peer_principal)
    }
}

/// The principal of a DER certificate under the Q2 SAN rule: exactly one SAN, a valid (non-wildcard) dNSName,
/// confirmed with `webpki::EndEntityCert::verify_is_valid_for_subject_name`. Chain validity is the TLS layer's job.
pub fn principal_from_der(der: &[u8]) -> Result<String, FedError> {
    let cert_der = CertificateDer::from(der);
    let cert = webpki::EndEntityCert::try_from(&cert_der)
        .map_err(|_| FedError::Principal("not a parseable X.509 certificate"))?;
    // webpki parsed the DER and refused duplicate extensions; it exposes no iterator over every SAN kind, so
    // the SAN entries are counted from the extension itself.
    let sans = san_entries(der).ok_or(FedError::Principal("malformed certificate"))?;
    let name = match sans.as_slice() {
        [] => return Err(FedError::Principal("no subjectAltName")),
        [(GENERAL_NAME_DNS, name)] => *name,
        [_] => return Err(FedError::Principal("the subjectAltName is not a dNSName")),
        _ => return Err(FedError::Principal("more than one subjectAltName")),
    };
    let name =
        std::str::from_utf8(name).map_err(|_| FedError::Principal("the dNSName is not ASCII"))?;
    cert.verify_is_valid_for_subject_name(
        &name
            .try_into()
            .map_err(|_| FedError::Principal("the dNSName is not a valid DNS name"))?,
    )
    .map_err(|_| FedError::Principal("the certificate is not valid for its dNSName"))?;
    Ok(name.to_owned())
}

/// The principal of the CA-verified client leaf (tonic lists the peer chain leaf first).
/// `PERMISSION_DENIED unauthenticated_peer` when there is no certificate or it breaks the SAN rule.
pub fn peer_principal<T>(req: &Request<T>) -> Result<String, Status> {
    let unauthenticated = || wire::status(Code::PermissionDenied, wire::UNAUTHENTICATED_PEER);
    let certs = req.peer_certs().ok_or_else(unauthenticated)?;
    let leaf = certs.first().ok_or_else(unauthenticated)?;
    principal_from_der(leaf.as_ref()).map_err(|_| unauthenticated())
}

#[derive(Debug, Clone)]
enum Role {
    /// `cered`: serves the registry to bound worker principals.
    Coordinator { bindings: BTreeMap<String, String> },
    /// `cere-worker`: serves WorkerService/WorkerInferenceService to the coordinator principal only.
    Agent { coordinator_principal: String },
}

/// The Q2 role table. Every refusal is `PERMISSION_DENIED permission_denied`; handlers call it first.
#[derive(Debug, Clone)]
pub struct Authz {
    role: Role,
}

fn denied() -> Status {
    wire::status(Code::PermissionDenied, wire::PERMISSION_DENIED_REASON)
}

impl Authz {
    /// `bindings`: worker principal -> the only worker_id it may register and heartbeat for.
    pub fn coordinator_side(bindings: BTreeMap<String, String>) -> Authz {
        Authz {
            role: Role::Coordinator { bindings },
        }
    }

    pub fn agent_side(coordinator_principal: String) -> Authz {
        Authz {
            role: Role::Agent {
                coordinator_principal,
            },
        }
    }

    /// Register/Heartbeat: the caller must be a worker principal bound to exactly `claimed_worker_id`.
    pub fn registry_caller<T>(
        &self,
        req: &Request<T>,
        claimed_worker_id: &str,
    ) -> Result<String, Status> {
        let principal = peer_principal(req).map_err(|_| denied())?;
        match &self.role {
            Role::Coordinator { bindings }
                if bindings.get(&principal).map(String::as_str) == Some(claimed_worker_id) =>
            {
                Ok(principal)
            }
            _ => Err(denied()),
        }
    }

    /// WorkerService/WorkerInferenceService: the caller must be the bound coordinator principal.
    pub fn coordinator_caller<T>(&self, req: &Request<T>) -> Result<String, Status> {
        let principal = peer_principal(req).map_err(|_| denied())?;
        match &self.role {
            Role::Agent {
                coordinator_principal,
            } if principal == *coordinator_principal => Ok(principal),
            _ => Err(denied()),
        }
    }
}

/// GeneralName tag of a dNSName (`[2] IMPLICIT IA5String`).
const GENERAL_NAME_DNS: u8 = 0x82;
/// OID 2.5.29.17 (subjectAltName), contents octets.
const OID_SUBJECT_ALT_NAME: &[u8] = &[0x55, 0x1d, 0x11];
const TAG_SEQUENCE: u8 = 0x30;
const TAG_EXTENSIONS: u8 = 0xa3;
const TAG_OID: u8 = 0x06;
const TAG_BOOLEAN: u8 = 0x01;
const TAG_OCTET_STRING: u8 = 0x04;

/// One DER element: (tag, contents, rest). Low tag numbers and definite lengths only, as X.509 uses.
fn tlv(input: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, rest) = input.split_first()?;
    if tag & 0x1f == 0x1f {
        return None;
    }
    let (&first, rest) = rest.split_first()?;
    let (len, rest) = if first < 0x80 {
        (usize::from(first), rest)
    } else {
        let n = usize::from(first & 0x7f);
        if n == 0 || n > 4 || rest.len() < n {
            return None;
        }
        let (bytes, rest) = rest.split_at(n);
        let len = bytes
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        (len, rest)
    };
    if rest.len() < len {
        return None;
    }
    let (contents, rest) = rest.split_at(len);
    Some((tag, contents, rest))
}

/// Every GeneralName (tag, contents) of the subjectAltName extension; empty when the extension is absent.
fn san_entries(der: &[u8]) -> Option<Vec<(u8, &[u8])>> {
    let (TAG_SEQUENCE, cert, _) = tlv(der)? else {
        return None;
    };
    let (TAG_SEQUENCE, mut tbs, _) = tlv(cert)? else {
        return None;
    };
    while !tbs.is_empty() {
        let (tag, contents, rest) = tlv(tbs)?;
        tbs = rest;
        if tag != TAG_EXTENSIONS {
            continue;
        }
        let (TAG_SEQUENCE, mut extensions, _) = tlv(contents)? else {
            return None;
        };
        while !extensions.is_empty() {
            let (TAG_SEQUENCE, extension, rest) = tlv(extensions)? else {
                return None;
            };
            extensions = rest;
            let (TAG_OID, oid, fields) = tlv(extension)? else {
                return None;
            };
            if oid != OID_SUBJECT_ALT_NAME {
                continue;
            }
            let (mut tag, mut value, mut trailing) = tlv(fields)?;
            if tag == TAG_BOOLEAN {
                (tag, value, trailing) = tlv(trailing)?;
            }
            if tag != TAG_OCTET_STRING || !trailing.is_empty() {
                return None;
            }
            let (TAG_SEQUENCE, mut names, []) = tlv(value)? else {
                return None;
            };
            let mut entries = Vec::new();
            while !names.is_empty() {
                let (tag, name, rest) = tlv(names)?;
                entries.push((tag, name));
                names = rest;
            }
            return Some(entries);
        }
    }
    Some(Vec::new())
}
