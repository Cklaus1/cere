//! Ephemeral cluster PKI for the federation tests (m35 design note Q2): an rcgen CA and leaves minted with EKU
//! serverAuth + clientAuth, written as PEM files into a test temp dir. A second `TestPki` is a foreign CA.

use std::fs::OpenOptions;
use std::io::Write;
use std::net::IpAddr;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose, SanType,
};

/// One subject alternative name of a test leaf.
pub enum San {
    Dns(String),
    Ip(IpAddr),
    Uri(String),
    Email(String),
}

pub struct TestPki {
    ca_pem: String,
    issuer: Issuer<'static, KeyPair>,
}

pub struct Leaf {
    cert_pem: String,
    key_pem: String,
    der: Vec<u8>,
}

fn next_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn write_file(path: &Path, contents: &str, mode: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(path)
        .unwrap();
    file.write_all(contents.as_bytes()).unwrap();
}

impl TestPki {
    pub fn new() -> TestPki {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::default();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        params
            .distinguished_name
            .push(DnType::CommonName, "cere test cluster CA");
        let ca = params.self_signed(&key).unwrap();
        TestPki {
            ca_pem: ca.pem(),
            issuer: Issuer::new(params, key),
        }
    }

    /// A leaf whose only SAN is the dNSName `principal` (the Q2 rule).
    pub fn leaf(&self, principal: &str) -> Leaf {
        self.leaf_with_sans(&[San::Dns(principal.to_owned())])
    }

    /// A leaf with exactly `sans` (possibly none), for the SAN-rule refusals.
    pub fn leaf_with_sans(&self, sans: &[San]) -> Leaf {
        let mut params = CertificateParams::default();
        params.subject_alt_names = sans
            .iter()
            .map(|san| match san {
                San::Dns(name) => SanType::DnsName(name.clone().try_into().unwrap()),
                San::Ip(ip) => SanType::IpAddress(*ip),
                San::Uri(uri) => SanType::URI(uri.clone().try_into().unwrap()),
                San::Email(email) => SanType::Rfc822Name(email.clone().try_into().unwrap()),
            })
            .collect();
        params
            .distinguished_name
            .push(DnType::CommonName, "cere test leaf");
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ServerAuth,
            ExtendedKeyUsagePurpose::ClientAuth,
        ];
        let key = KeyPair::generate().unwrap();
        let cert = params.signed_by(&key, &self.issuer).unwrap();
        Leaf {
            cert_pem: cert.pem(),
            key_pem: key.serialize_pem(),
            der: cert.der().to_vec(),
        }
    }

    pub fn ca_pem(&self) -> &str {
        &self.ca_pem
    }

    /// Writes the CA certificate (never its key) into `dir`.
    pub fn write_ca(&self, dir: &Path) -> PathBuf {
        let path = dir.join(format!("ca-{}.pem", next_id()));
        write_file(&path, &self.ca_pem, 0o644);
        path
    }
}

impl Leaf {
    pub fn cert_pem(&self) -> &str {
        &self.cert_pem
    }

    pub fn key_pem(&self) -> &str {
        &self.key_pem
    }

    pub fn der(&self) -> &[u8] {
        &self.der
    }

    /// Writes the certificate and its private key (mode 0600) into `dir`.
    pub fn write(&self, dir: &Path) -> (PathBuf, PathBuf) {
        let id = next_id();
        let cert = dir.join(format!("leaf-{id}.pem"));
        let key = dir.join(format!("leaf-{id}.key"));
        write_file(&cert, &self.cert_pem, 0o644);
        write_file(&key, &self.key_pem, 0o600);
        (cert, key)
    }
}
