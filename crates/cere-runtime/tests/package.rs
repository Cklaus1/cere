//! Packaging invariants (spec m1close Q11): no crate is publishable, and `cered --version` reports the repo's
//! PACKAGE_VERSION. The publish check reads the manifests itself instead of running `cargo metadata`: a test binary
//! runs as users who cannot run the toolchain that built it (the gate's uid 65534 pass cannot enter root's `~/.cargo`,
//! O314, rust-nonroot design note R1).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn package_version() -> String {
    std::fs::read_to_string(repo_root().join("PACKAGE_VERSION"))
        .expect("PACKAGE_VERSION")
        .trim()
        .to_string()
}

fn read_toml(path: &Path) -> toml::Table {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.parse()
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Each package of the workspace whose root manifest is `root/Cargo.toml` (the root's own `[package]`, then each
/// member's), with whether cargo would let it be published: its `[package] publish`, or `[workspace.package] publish`
/// when it sets `publish.workspace = true`. `false` and `[]` (both `[]` in `cargo metadata`) refuse publishing; `true`,
/// a registry list and no key allow it. A member glob or a `publish` of another shape panics, naming it, instead of
/// being skipped.
fn packages_publishable(root: &Path) -> Vec<(String, bool)> {
    let top = read_toml(&root.join("Cargo.toml"));
    let ws = top
        .get("workspace")
        .and_then(toml::Value::as_table)
        .expect("[workspace]");
    let inherited = ws.get("package").and_then(|p| p.get("publish"));
    let members = ws
        .get("members")
        .and_then(toml::Value::as_array)
        .expect("workspace.members");
    let mut manifests = vec![(".".to_string(), top.clone())];
    for m in members {
        let rel = m.as_str().expect("a workspace member is a path");
        assert!(
            !rel.contains(['*', '?', '[']),
            "workspace member glob {rel:?}: list the members by path"
        );
        manifests.push((
            rel.to_string(),
            read_toml(&root.join(rel).join("Cargo.toml")),
        ));
    }
    let mut out = Vec::new();
    for (rel, manifest) in &manifests {
        let Some(package) = manifest.get("package").and_then(toml::Value::as_table) else {
            assert_eq!(rel, ".", "{rel}/Cargo.toml has no [package]");
            continue;
        };
        let name = package
            .get("name")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("{rel}: package.name"));
        let publish = match package.get("publish") {
            Some(toml::Value::Table(t))
                if t.get("workspace") == Some(&toml::Value::Boolean(true)) =>
            {
                Some(inherited.unwrap_or_else(|| {
                    panic!("{name}: publish.workspace = true without workspace.package.publish")
                }))
            }
            other => other,
        };
        let publishable = match publish {
            None => true,
            Some(toml::Value::Boolean(b)) => *b,
            Some(toml::Value::Array(registries)) => !registries.is_empty(),
            Some(other) => panic!("{name}: publish = {other}"),
        };
        out.push((name.to_string(), publishable));
    }
    out
}

#[test]
fn no_workspace_package_is_publishable() {
    let packages = packages_publishable(&repo_root());
    assert!(packages.len() >= 4, "workspace packages: {packages:?}");
    for (name, publishable) in &packages {
        assert!(!publishable, "{name} must set publish = false");
    }
}

/// Writes a workspace at a fresh dir of `CARGO_TARGET_TMPDIR`: a root manifest with `members` (as written, so a glob
/// can be one) and, when `ws_publish` is set, `[workspace.package] publish = <ws_publish>`; then one library package
/// per `(name, publish line)` in `packages`.
fn write_workspace(
    tag: &str,
    members: &[&str],
    ws_publish: Option<&str>,
    packages: &[(&str, &str)],
) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("package-publish-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let members: Vec<String> = members.iter().map(|m| format!("{m:?}")).collect();
    let mut top = format!(
        "[workspace]\nmembers = [{}]\nresolver = \"3\"\n",
        members.join(", ")
    );
    if let Some(p) = ws_publish {
        top.push_str(&format!("\n[workspace.package]\npublish = {p}\n"));
    }
    std::fs::write(root.join("Cargo.toml"), top).unwrap();
    for (name, publish) in packages {
        let dir = root.join(name);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "").unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            format!("[package]\nname = \"{name}\"\nversion = \"0.0.1\"\nedition = \"2021\"\n{publish}\n"),
        )
        .unwrap();
    }
    root
}

fn publishable_in(
    tag: &str,
    ws_publish: Option<&str>,
    packages: &[(&str, &str)],
) -> Vec<(String, bool)> {
    let names: Vec<&str> = packages.iter().map(|(n, _)| *n).collect();
    let root = write_workspace(tag, &names, ws_publish, packages);
    let got = packages_publishable(&root);
    std::fs::remove_dir_all(&root).unwrap();
    got
}

fn owned(want: &[(&str, bool)]) -> Vec<(String, bool)> {
    want.iter().map(|(n, p)| (n.to_string(), *p)).collect()
}

/// The check above reads `publish` as cargo does. Under `[workspace.package] publish = false`: no key, `false`, `[]`,
/// `true`, a registry list and `publish.workspace = true` (`cargo metadata --no-deps` on the same files: `null`, `[]`,
/// `[]`, `null`, `["internal"]`, `[]`). A member with no key does not inherit: only `publish.workspace = true` does,
/// so under `[workspace.package] publish = ["internal"]` and `= true` an inheriting member is publishable too (cargo:
/// `["internal"]`, `null`) and one with no key is publishable (`null`) under `= false` as well.
#[test]
fn the_publish_check_reads_publish_as_cargo_does() {
    let shapes = [
        ("absent", ""),
        ("no", "publish = false"),
        ("empty", "publish = []"),
        ("yes", "publish = true"),
        ("registry", "publish = [\"internal\"]"),
        ("inherits", "publish.workspace = true"),
    ];
    assert_eq!(
        publishable_in("shapes", Some("false"), &shapes),
        owned(&[
            ("absent", true),
            ("no", false),
            ("empty", false),
            ("yes", true),
            ("registry", true),
            ("inherits", false),
        ])
    );
    let inherit = [("absent", ""), ("inherits", "publish.workspace = true")];
    assert_eq!(
        publishable_in("registry", Some("[\"internal\"]"), &inherit),
        owned(&[("absent", true), ("inherits", true)])
    );
    assert_eq!(
        publishable_in("true", Some("true"), &inherit),
        owned(&[("absent", true), ("inherits", true)])
    );
}

/// (tag, members, `[workspace.package] publish`, packages, expected panic text).
type Refusal<'a> = (
    &'a str,
    &'a [&'a str],
    Option<&'a str>,
    &'a [(&'a str, &'a str)],
    &'a str,
);

/// A manifest shape the check does not read is a failure naming it, never a guess: a `publish` string or a
/// `publish = { workspace = false }` table (cargo refuses both), `publish.workspace = true` with no
/// `[workspace.package] publish` (cargo refuses it), and a member glob (cargo expands it; the check does not).
#[test]
fn the_publish_check_refuses_shapes_it_does_not_read() {
    let cases: [Refusal; 4] = [
        (
            "string",
            &["s"],
            None,
            &[("s", "publish = \"internal\"")],
            "s: publish = \"internal\"",
        ),
        (
            "table",
            &["t"],
            None,
            &[("t", "publish = { workspace = false }")],
            "t: publish = ",
        ),
        (
            "orphan",
            &["o"],
            None,
            &[("o", "publish.workspace = true")],
            "o: publish.workspace = true without workspace.package.publish",
        ),
        (
            "glob",
            &["g*"],
            Some("false"),
            &[("g", "publish = false")],
            "workspace member glob \"g*\"",
        ),
    ];
    for (tag, members, ws_publish, packages, want) in cases {
        let root = write_workspace(tag, members, ws_publish, packages);
        let got = std::panic::catch_unwind(|| packages_publishable(&root));
        std::fs::remove_dir_all(&root).unwrap();
        let err = got.expect_err(tag);
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(msg.contains(want), "{tag}: {msg:?} lacks {want:?}");
    }
}

#[test]
fn cered_version_reports_package_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_cered"))
        .arg("--version")
        .output()
        .expect("cered --version");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    assert!(
        stdout.contains(&format!("package v{}", package_version())),
        "{stdout}"
    );
}
