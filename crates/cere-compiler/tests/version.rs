//! `cerec --version` reports the repo's PACKAGE_VERSION (spec m1close Q11).

use std::path::Path;
use std::process::Command;

#[test]
fn cerec_version_reports_package_version() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let version = std::fs::read_to_string(root.join("PACKAGE_VERSION")).expect("PACKAGE_VERSION");
    let out = Command::new(env!("CARGO_BIN_EXE_cerec"))
        .arg("--version")
        .output()
        .expect("cerec --version");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    assert!(
        stdout.contains(&format!("package v{}", version.trim())),
        "{stdout}"
    );
}
