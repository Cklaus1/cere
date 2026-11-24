//! Shared helpers of the `native_*` integration tests (spec m22 R1-R3): the toy executor libraries, a verified copy
//! of the M2.1 fixture artifact (`examples/artifact`, prepared from `tests/fixtures/model_src`), temp directories and
//! the toy model's token rule.
#![allow(dead_code)]

use cere_compiler::artifact::{self, Mode};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// The library named by `var` (exported by `make test-rust`). Absent variable or file is a failure, never a skip.
fn toy(var: &str) -> PathBuf {
    let path = std::env::var_os(var).unwrap_or_else(|| {
        panic!("{var} is not set: run `make test-executor-host` (make test-rust exports it)")
    });
    let path = PathBuf::from(path);
    assert!(
        path.is_file(),
        "{var}={} is not a file: run `make test-executor-host`",
        path.display()
    );
    path
}

/// `$CERE_TOY_EXECUTOR`: the CPU toy executor with its forward pass.
pub fn toy_library() -> PathBuf {
    toy("CERE_TOY_EXECUTOR")
}

/// `$CERE_TOY_EXECUTOR_NOFORWARD`: the toy built with `CERE_TOY_NO_FORWARD`.
pub fn toy_library_noforward() -> PathBuf {
    toy("CERE_TOY_EXECUTOR_NOFORWARD")
}

/// `$CERE_TOY_EXECUTOR_PRESTAGING`: the toy built with `CERE_TOY_PRE_STAGING`, which lacks the two put staging
/// exports (m216 design note M216S-11), `cere_executor_set_decode_rounds` (O79, m210 design note PF1),
/// `cere_executor_set_prefill_slice` (MixedPD, mixed-pd design note), the constrained-draft calls
/// `cere_executor_round_drafts` / `cere_executor_constrain_drafts` (GrammarSpec GS-01) and
/// `cere_executor_set_decode_share` (S1 G4, beat-sglang design note) as a library from before them does.
pub fn toy_library_prestaging() -> PathBuf {
    toy("CERE_TOY_EXECUTOR_PRESTAGING")
}

/// The toy model's next token: `(31·last + position) mod vocab`, `position` = the sequence length before it.
pub fn toy_next(last: u32, position: u64, vocab: u32) -> u32 {
    ((31 * u64::from(last) + position) % u64::from(vocab)) as u32
}

/// `n` toy tokens continuing a sequence of length `len` that ends with `last`.
pub fn toy_tokens(mut last: u32, mut len: u64, n: usize, vocab: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        last = toy_next(last, len, vocab);
        len += 1;
        out.push(last);
    }
    out
}

/// The fixture model's vocabulary (`tests/fixtures/model_src/config.json`).
pub const FIXTURE_VOCAB: u32 = 70_000;

/// A temp directory under Cargo's `CARGO_TARGET_TMPDIR` (a real filesystem, so eviction and `mincore` mean
/// something), removed on drop.
pub struct Tmp(pub PathBuf);

static SEQ: AtomicUsize = AtomicUsize::new(0);

impl Tmp {
    pub fn new(tag: &str) -> Tmp {
        let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "native-it-{tag}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), &dest).unwrap();
        }
    }
}

/// The prepared fixture artifact in the repo (read-only use: `Mode::Headers`).
pub fn fixture_artifact() -> (PathBuf, String) {
    let dir = root().join("examples/artifact");
    let sha = std::fs::read_to_string(dir.join("manifest.sha256"))
        .unwrap()
        .trim()
        .to_string();
    (dir, sha)
}

/// A copy of the fixture artifact in `t` with a `full_verify` stamp (what `Mode::Stamp` needs); its path and
/// manifest digest.
pub fn verified_artifact(t: &Tmp) -> (PathBuf, String) {
    let (src, sha) = fixture_artifact();
    let dir = t.join("fixture-artifact");
    copy_tree(&src, &dir);
    let art = artifact::open(&dir, &sha, Mode::Full).unwrap();
    art.write_full_verify_stamp().unwrap();
    (dir, sha)
}

/// [`verified_artifact`] with `tokenizer.json` replaced by `tokenizer` (its manifest entry and the composite
/// `tokenizer_hash` recomputed, the manifest re-encoded canonically): its path and the new manifest digest.
pub fn verified_artifact_with_tokenizer(t: &Tmp, tokenizer: &[u8]) -> (PathBuf, String) {
    use sha2::{Digest, Sha256};
    let hex = |bytes: &[u8]| -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .concat()
    };
    let (src, _) = fixture_artifact();
    let dir = t.join("fixture-artifact");
    copy_tree(&src, &dir);
    std::fs::write(dir.join("tokenizer.json"), tokenizer).unwrap();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let entry = manifest["files"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|f| f["path"] == "tokenizer.json")
        .unwrap();
    let sha = hex(tokenizer);
    entry["bytes"] = serde_json::json!(tokenizer.len());
    entry["sha256"] = serde_json::json!(sha);
    entry["upstream"]["lfs_sha256"] = serde_json::json!(sha);
    let files: Vec<cere_compiler::manifest::FileEntry> =
        serde_json::from_value(manifest["files"].clone()).unwrap();
    let (_, tokenizer_hash, _) = cere_compiler::manifest::identity_hashes(&files);
    manifest["tokenizer_hash"] = serde_json::json!(tokenizer_hash);
    let bytes = cere_compiler::canon_json::to_vec(&manifest).unwrap();
    std::fs::write(dir.join("manifest.json"), &bytes).unwrap();
    let sha = hex(&bytes);
    std::fs::write(dir.join("manifest.sha256"), format!("{sha}\n")).unwrap();
    let art = artifact::open(&dir, &sha, Mode::Full).unwrap();
    art.write_full_verify_stamp().unwrap();
    (dir, sha)
}

/// Per-placement byte totals of the fixture inventory (`tests/fixtures/model_src.inventory.json`): device tensors
/// outside the MTP draft, the MTP draft, and the pinned-host tensors.
pub struct InventoryTotals {
    pub device: u64,
    pub draft: u64,
    pub host: u64,
}

pub fn inventory_totals() -> InventoryTotals {
    let raw = std::fs::read(root().join("tests/fixtures/model_src.inventory.json")).unwrap();
    let inv: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let mut t = InventoryTotals {
        device: 0,
        draft: 0,
        host: 0,
    };
    for f in inv["families"].as_array().unwrap() {
        let bytes = f["bytes"].as_u64().unwrap();
        match (
            f["placement"].as_str().unwrap(),
            f["role"].as_str().unwrap(),
        ) {
            ("device", "draft") => t.draft += bytes,
            ("device", _) => t.device += bytes,
            ("host_pinned", _) => t.host += bytes,
            _ => {}
        }
    }
    assert_eq!(
        t.device + t.draft,
        inv["by_placement"]["device"]["bytes"].as_u64().unwrap()
    );
    assert_eq!(
        t.host,
        inv["by_placement"]["host_pinned"]["bytes"]
            .as_u64()
            .unwrap()
    );
    t
}
