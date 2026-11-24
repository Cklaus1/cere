//! R3 (m21 design note Q1-Q3, Q7, Q8, Q10): `cerec prepare` on a copy of the fixture, ROOT in a tempdir. The
//! letters follow the R3 acceptance cell.
mod support;

use cere_compiler::fsio::FileLock;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, SystemTime};
use support::*;

const TOKEN_OUT: &str = "draft/token_map.u32le";
/// 1 TiB available: far above the default floor.
const PLENTY: &str = "MemAvailable: 1073741824 kB\n";

struct Env {
    tmp: Tmp,
    src: PathBuf,
    root: PathBuf,
    meminfo: PathBuf,
    token_map: PathBuf,
    /// Pass `--token-map` (false: the `speculative: "none"` profile runs, SelfTune ST4 S3b).
    with_token_map: bool,
    profile: PathBuf,
}

impl Env {
    fn new(tag: &str) -> Env {
        let tmp = Tmp::new(&format!("prep-{tag}"));
        let src = tmp.join("src");
        copy_tree(&model_src(), &src);
        let root = tmp.join("root");
        std::fs::create_dir(&root).unwrap();
        let meminfo = tmp.join("meminfo");
        std::fs::write(&meminfo, PLENTY).unwrap();
        Env {
            token_map: src.join("token_map.pt"),
            with_token_map: true,
            profile: fixture("model_src.profile.json"),
            tmp,
            src,
            root,
            meminfo,
        }
    }

    fn cmd(&self, extra: &[&str]) -> Command {
        let mut c = cerec();
        c.arg("prepare").arg(&self.src);
        if self.with_token_map {
            c.arg("--token-map").arg(&self.token_map);
        }
        c.args(["--name", "fixture-model", "--families"])
            .arg(fixture("model_src.families.json"))
            .arg("--profile")
            .arg(&self.profile)
            .arg("--out")
            .arg(&self.root)
            .args(extra)
            .env("CEREC_TEST_MEMINFO", &self.meminfo);
        c
    }

    fn prepare(&self, extra: &[&str]) -> Output {
        run(&mut self.cmd(extra))
    }

    fn ok(&self, extra: &[&str]) -> Value {
        json(&ok(self.prepare(extra)))
    }

    /// Published artifact directories under ROOT.
    fn artifacts(&self) -> Vec<String> {
        self.root_entries()
            .into_iter()
            .filter(|n| !n.starts_with('.') && !n.ends_with(".verified.json"))
            .collect()
    }

    fn temp_dirs(&self) -> Vec<String> {
        self.root_entries()
            .into_iter()
            .filter(|n| n.starts_with(".tmp-"))
            .collect()
    }

    fn root_entries(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(&self.root)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        v.sort();
        v
    }

    fn stamp_path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.verified.json"))
    }
}

fn lock_arg() -> String {
    fixture("model_src.lock.json").display().to_string()
}

fn dir_of(report: &Value) -> PathBuf {
    PathBuf::from(report["artifact_dir"].as_str().unwrap())
}

fn manifest_of(report: &Value) -> Value {
    serde_json::from_slice(&std::fs::read(dir_of(report).join("manifest.json")).unwrap()).unwrap()
}

/// Q7's `payload_bytes_read` of a full source read: shard data regions, whole other payload files, the token map.
fn source_payload(m: &Value) -> u64 {
    let files: u64 = m["files"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["path"] != TOKEN_OUT)
        .map(|f| {
            f["bytes"].as_u64().unwrap() - f["safetensors"]["header_bytes"].as_u64().unwrap_or(0)
        })
        .sum();
    files + m["transforms"][0]["source_bytes"].as_u64().unwrap()
}

/// Relative paths of the regular files below `dir`, sorted.
fn list_files(dir: &Path) -> Vec<String> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(base, &p, out);
            } else {
                out.push(p.strip_prefix(base).unwrap().to_str().unwrap().to_string());
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

fn mtimes(dir: &Path) -> BTreeMap<String, (i64, i64)> {
    list_files(dir)
        .into_iter()
        .map(|p| {
            let m = std::fs::metadata(dir.join(&p)).unwrap();
            (p, (m.mtime(), m.mtime_nsec()))
        })
        .collect()
}

fn set_mtime(p: &Path, secs: u64) {
    let f = std::fs::File::options().write(true).open(p).unwrap();
    f.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
        .unwrap();
}

fn sha_hex(bytes: &[u8]) -> String {
    cere_compiler::hex(&Sha256::digest(bytes))
}

/// Rewrites one `lfs_sha256` of the fixture copy's HF tree file.
fn set_tree_lfs(src: &Path, name: &str, lfs: &str) {
    let trees = src.join(".cache/huggingface/trees");
    let file = std::fs::read_dir(&trees)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut v: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    v["files"][name]["lfs_sha256"] = Value::String(lfs.to_string());
    std::fs::write(&file, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
}

/// The fixture token map with one byte of its pickle replaced (sizes unchanged; no CRC is checked, Q10).
fn mutated_token_map(dest: &Path, from: &[u8], to: &[u8]) {
    let mut bytes = std::fs::read(model_src().join("token_map.pt")).unwrap();
    let at = bytes
        .windows(from.len())
        .position(|w| w == from)
        .expect("pattern in data.pkl");
    bytes[at..at + to.len()].copy_from_slice(to);
    std::fs::write(dest, bytes).unwrap();
}

#[test]
fn a_fixture_prepare_equals_examples_artifact_and_python_full_verify_accepts() {
    let e = Env::new("a");
    for (i, p) in list_files(&e.src).iter().enumerate() {
        set_mtime(&e.src.join(p), 1_500_000_000 + i as u64);
    }
    let r = e.ok(&["--lock", &lock_arg()]);
    assert_eq!(r["outcome"], "prepared");
    assert_eq!(r["lock"], "matched");
    assert_eq!(r["rehash_pass"], false);
    let dir = dir_of(&r);
    let sha = r["manifest_sha256"].as_str().unwrap();
    assert_eq!(r["artifact_id"], format!("fixture-model-{}", &sha[..16]));

    let example = root().join("examples/artifact");
    assert_eq!(list_files(&dir), list_files(&example));
    for p in list_files(&dir) {
        assert!(
            std::fs::read(dir.join(&p)).unwrap() == std::fs::read(example.join(&p)).unwrap(),
            "{p} differs"
        );
    }
    assert_eq!(
        std::fs::read_to_string(example.join("manifest.sha256")).unwrap(),
        format!("{sha}\n")
    );

    let m = manifest_of(&r);
    assert_eq!(r["payload_bytes_read"], source_payload(&m));
    assert_eq!(
        r["excluded"],
        serde_json::json!([
            ".cache",
            "config.json.orig",
            "layer-00000-experts-0000-0001.complete.json",
            "qualification-notes.txt",
            "token_map.pt"
        ])
    );
    // (j) the fixture's config.json differs from its upstream blob: reported, not rejected.
    let up = &r["upstream"];
    assert_eq!(up["lfs_match"], up["lfs_files"]);
    let lm = up["local_modification"].as_array().unwrap();
    assert_eq!(lm.len(), 1);
    assert_eq!(lm[0]["path"], "config.json");
    assert_eq!(lm[0]["upstream_size"], 806);
    assert!(lm[0]["upstream_blob_id"]
        .as_str()
        .is_some_and(|s| s.len() == 40));

    ok(run(python("verify_artifact.py")
        .args(["--mode", "full", "--expected-manifest-sha256", sha])
        .arg(&dir)));
    // Directories 0555, files 0444 after publish (Q2).
    assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o777, 0o555);
    assert_eq!(
        std::fs::metadata(dir.join(TOKEN_OUT)).unwrap().mode() & 0o777,
        0o444
    );
}

#[test]
fn b_second_prepare_is_reused_without_payload_reads() {
    let e = Env::new("b");
    let first = e.ok(&[]);
    let before = mtimes(&dir_of(&first));
    let second = e.ok(&[]);
    assert_eq!(second["outcome"], "reused");
    assert_eq!(second["payload_bytes_read"], 0);
    assert_eq!(second["bytes_written"], 0);
    assert_eq!(second["artifact_id"], first["artifact_id"]);
    assert_eq!(mtimes(&dir_of(&second)), before);
    assert_eq!(e.artifacts().len(), 1);
}

/// A reused artifact whose stamp predates the fs-uuid format (format 1: every file by `st_dev`) and still matches gets
/// its stamp restated as format 2, so a later reboot that renumbers `st_dev` does not force a full copy (Q7).
#[test]
fn prepare_reuse_upgrades_a_format_1_stamp() {
    use cere_compiler::stamp::{self, Fs, Stamp};
    let e = Env::new("legacy");
    let first = e.ok(&[]);
    let id = first["artifact_id"].as_str().unwrap();
    let path = e.stamp_path(id);
    let (written, _) = Stamp::read(&path).unwrap();
    let mut legacy = written.clone();
    legacy.format = stamp::LEGACY_FORMAT;
    for (p, f) in &mut legacy.files {
        f.fs = Fs::Dev(std::fs::metadata(dir_of(&first).join(p)).unwrap().dev());
    }
    std::fs::write(&path, legacy.to_bytes()).unwrap();
    let second = e.ok(&[]);
    assert_eq!(second["outcome"], "reused");
    assert_eq!(second["payload_bytes_read"], 0);
    assert_eq!(second["artifact_id"], first["artifact_id"]);
    let (upgraded, n) = Stamp::read(&path).unwrap();
    assert_eq!(second["bytes_written"], n);
    assert_eq!(upgraded.format, stamp::FORMAT);
    assert_eq!(upgraded.files, written.files);
    assert_eq!(upgraded.stamp_source, stamp::PREPARE_HASH);
    assert_eq!(upgraded.manifest_sha256, written.manifest_sha256);
    assert_eq!(upgraded.source_fingerprint, written.source_fingerprint);
    // Nothing more to upgrade: the next reuse writes nothing.
    let third = e.ok(&[]);
    assert_eq!(third["outcome"], "reused");
    assert_eq!(third["bytes_written"], 0);
}

#[test]
fn c_touched_shard_is_reused_after_rehash() {
    let e = Env::new("c");
    let first = e.ok(&[]);
    set_mtime(&e.src.join("model-plefp8-00000.safetensors"), 1_600_000_000);
    let r = e.ok(&[]);
    assert_eq!(r["outcome"], "reused_after_rehash");
    assert_eq!(r["rehash_pass"], true);
    assert_eq!(
        r["payload_bytes_read"],
        source_payload(&manifest_of(&first))
    );
    let id = first["artifact_id"].as_str().unwrap();
    assert_eq!(r["artifact_id"], id);
    assert_eq!(
        r["bytes_written"],
        std::fs::metadata(e.stamp_path(id)).unwrap().len()
    );
    assert_eq!(e.artifacts(), vec![id.to_string()]);
    // The refreshed fingerprint makes the next run a plain reuse.
    assert_eq!(e.ok(&[])["outcome"], "reused");
}

#[test]
fn d_changed_profile_is_relabelled_by_hardlinks() {
    let mut e = Env::new("d");
    let first = e.ok(&[]);
    let old_id = first["artifact_id"].as_str().unwrap().to_string();
    let old_dir = dir_of(&first);
    let old_stamp_bytes = std::fs::metadata(e.stamp_path(&old_id)).unwrap().len();

    let mut profile: Value = serde_json::from_slice(&std::fs::read(&e.profile).unwrap()).unwrap();
    profile["device_model"] = Value::String("fixture-cpu-relabelled".into());
    e.profile = e.tmp.join("profile.json");
    std::fs::write(
        &e.profile,
        cere_compiler::canon_json::to_vec(&profile).unwrap(),
    )
    .unwrap();

    let r = e.ok(&[]);
    assert_eq!(r["outcome"], "relabelled");
    assert_eq!(r["payload_bytes_read"], 0);
    let new_id = r["artifact_id"].as_str().unwrap();
    assert_ne!(new_id, old_id);
    let new_dir = dir_of(&r);
    let size = |p: PathBuf| std::fs::metadata(p).unwrap().len();
    // The new manifest and its digest file, the new stamp, and the old stamp rewritten with the new ctimes (Q7).
    let written = size(new_dir.join("manifest.json"))
        + size(new_dir.join("manifest.sha256"))
        + size(e.stamp_path(new_id))
        + std::fs::metadata(e.stamp_path(&old_id)).unwrap().len();
    assert_eq!(
        std::fs::metadata(e.stamp_path(&old_id)).unwrap().len(),
        old_stamp_bytes
    );
    assert_eq!(r["bytes_written"], written);
    assert_eq!(
        manifest_of(&r)["profile"]["device_model"],
        "fixture-cpu-relabelled"
    );

    for f in manifest_of(&first)["files"].as_array().unwrap() {
        let p = f["path"].as_str().unwrap();
        assert_eq!(
            std::fs::metadata(old_dir.join(p)).unwrap().ino(),
            std::fs::metadata(new_dir.join(p)).unwrap().ino(),
            "{p} is not a hardlink"
        );
    }
    // The old stamp still matches every old file's fstat identity, ctime included.
    let stamp: Value =
        serde_json::from_slice(&std::fs::read(e.stamp_path(&old_id)).unwrap()).unwrap();
    for (p, s) in stamp["files"].as_object().unwrap() {
        let m = std::fs::metadata(old_dir.join(p)).unwrap();
        let ctime_ns = m.ctime() as i128 * 1_000_000_000 + m.ctime_nsec() as i128;
        let mtime_ns = m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128;
        assert_eq!(s["ino"], m.ino(), "{p}");
        assert_eq!(s["ctime_ns"].as_i64().unwrap() as i128, ctime_ns, "{p}");
        assert_eq!(s["mtime_ns"].as_i64().unwrap() as i128, mtime_ns, "{p}");
    }
    let old_sha = first["manifest_sha256"].as_str().unwrap();
    ok(run(python("verify_artifact.py")
        .args(["--mode", "full", "--expected-manifest-sha256", old_sha])
        .arg(&old_dir)));
}

/// Points `e.profile` at a copy of the fixture profile with another `device_model` (a relabel, Q7).
fn relabel_profile(e: &mut Env, device_model: &str) {
    let mut profile: Value =
        serde_json::from_slice(&std::fs::read(fixture("model_src.profile.json")).unwrap()).unwrap();
    profile["device_model"] = Value::String(device_model.into());
    e.profile = e.tmp.join(&format!("profile-{device_model}.json"));
    std::fs::write(
        &e.profile,
        cere_compiler::canon_json::to_vec(&profile).unwrap(),
    )
    .unwrap();
}

fn cerec_verify(e: &Env, r: &Value, mode: &str) -> Output {
    run(cerec()
        .arg("verify")
        .arg(dir_of(r))
        .args([
            "--expected-manifest-sha256",
            r["manifest_sha256"].as_str().unwrap(),
        ])
        .args(["--mode", mode])
        .env("CEREC_TEST_MEMINFO", &e.meminfo))
}

/// R3(d) across several relabels: a later relabel links the shared inodes again (new ctime), and every earlier
/// artifact sharing them keeps a stamp that passes stamp-mode verify, not only the link source.
#[test]
fn d2_every_sibling_sharing_the_inodes_keeps_its_stamp_after_a_later_relabel() {
    let mut e = Env::new("d2");
    let first = e.ok(&[]);
    relabel_profile(&mut e, "fixture-cpu-r2");
    let second = e.ok(&[]);
    assert_eq!(second["outcome"], "relabelled");
    for r in [&first, &second] {
        ok(cerec_verify(&e, r, "full"));
    }
    relabel_profile(&mut e, "fixture-cpu-r3");
    let third = e.ok(&[]);
    assert_eq!(third["outcome"], "relabelled");
    // The new artifact's stamp is prepare_hash (stamp mode needs full_verify); the earlier two must still pass.
    for r in [&first, &second] {
        let out = cerec_verify(&e, r, "stamp");
        assert!(
            out.status.success(),
            "{}: {}",
            r["artifact_id"],
            String::from_utf8_lossy(&out.stderr)
        );
        ok(run(python("verify_artifact.py")
            .args(["--mode", "stamp", "--expected-manifest-sha256"])
            .arg(r["manifest_sha256"].as_str().unwrap())
            .arg(dir_of(r))));
    }
}

#[test]
fn e_changed_tensor_byte_gives_a_new_artifact_and_keeps_the_old() {
    let e = Env::new("e");
    let first = e.ok(&[]);
    let shard = e.src.join("model-bf16-00001.safetensors");
    let mut bytes = std::fs::read(&shard).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    std::fs::write(&shard, &bytes).unwrap();
    set_tree_lfs(&e.src, "model-bf16-00001.safetensors", &sha_hex(&bytes));

    let r = e.ok(&[]);
    assert_eq!(r["outcome"], "prepared");
    // Same (path, bytes) set: the documented double read, reported (Q7 step 3).
    assert_eq!(r["rehash_pass"], true);
    assert_ne!(r["artifact_id"], first["artifact_id"]);
    assert_eq!(e.artifacts().len(), 2);
    let old_sha = first["manifest_sha256"].as_str().unwrap();
    ok(run(python("verify_artifact.py")
        .args(["--mode", "full", "--expected-manifest-sha256", old_sha])
        .arg(dir_of(&first))));
}

#[test]
fn f_aborted_prepare_publishes_nothing_and_the_next_run_cleans_up() {
    let e = Env::new("f");
    let out = run(e.cmd(&[]).env("CEREC_TEST_ABORT_AFTER_FILES", "1"));
    assert!(!out.status.success());
    assert!(e.artifacts().is_empty());
    let stale = e.temp_dirs();
    assert_eq!(stale.len(), 1);
    let r = e.ok(&[]);
    assert_eq!(r["outcome"], "prepared");
    assert_eq!(r["stale_tmp_removed"], serde_json::json!(stale));
    assert!(e.temp_dirs().is_empty());
}

#[test]
fn g_concurrent_no_wait_exits_3() {
    let e = Env::new("g");
    let _held = FileLock::acquire(&e.root.join(".cerec.lock"), true).unwrap();
    let out = e.prepare(&["--no-wait"]);
    assert_eq!(
        out.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.stdout.is_empty());
    assert!(e.artifacts().is_empty());
}

#[test]
fn h_out_inside_src_directly_or_through_a_symlink_exits_2() {
    let mut e = Env::new("h");
    std::fs::create_dir(e.src.join("out")).unwrap();
    let via = e.tmp.join("via");
    std::os::unix::fs::symlink(&e.src, &via).unwrap();
    for out in [
        e.src.join("out"),
        e.src.clone(),
        via.clone(),
        via.join("out"),
    ] {
        e.root = out.clone();
        let o = e.prepare(&[]);
        assert_eq!(o.status.code(), Some(2), "{}", out.display());
        assert_eq!(class_of(&o), "unsafe_out", "{}", out.display());
    }
    assert_eq!(std::fs::read_dir(e.src.join("out")).unwrap().count(), 0);
}

#[test]
fn i_symlinked_shard_and_fifo_payload_exit_2() {
    let e = Env::new("i");
    let shard = e.src.join("model-plefp8-00000.safetensors");
    let outside = e.tmp.join("outside.safetensors");
    std::fs::rename(&shard, &outside).unwrap();
    std::os::unix::fs::symlink(&outside, &shard).unwrap();
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "symlink")
    );

    let e = Env::new("i2");
    let vocab = e.src.join("vocab.json");
    std::fs::remove_file(&vocab).unwrap();
    let c = std::ffi::CString::new(vocab.to_str().unwrap()).unwrap();
    // SAFETY: NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o644) }, 0);
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "special_file")
    );
    assert!(e.artifacts().is_empty() && e.temp_dirs().is_empty());
}

#[test]
fn j_lfs_sha_mismatch_on_a_weights_file_exits_2() {
    let e = Env::new("j");
    set_tree_lfs(
        &e.src,
        "layer-00000-experts-0000-0001.safetensors",
        &"0".repeat(64),
    );
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "upstream_lfs")
    );
    assert!(e.artifacts().is_empty() && e.temp_dirs().is_empty());
}

/// docs/CONFIGURATION.md:196-205: SHA-256 of the `sha256sum` lines of the listed files, sorted bytewise by name.
fn lock_digest(src: &Path, names: &[String]) -> String {
    let mut names = names.to_vec();
    names.sort();
    let lines: String = names
        .iter()
        .map(|n| format!("{}  {n}\n", sha_hex(&std::fs::read(src.join(n)).unwrap())))
        .collect::<Vec<_>>()
        .concat();
    sha_hex(lines.as_bytes())
}

#[test]
fn k_identity_hashes_follow_the_lock_definitions_and_a_wrong_lock_publishes_nothing() {
    let e = Env::new("k");
    let mut model: Vec<String> = list_files(&e.src)
        .into_iter()
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    model.extend(
        [
            "model.safetensors.index.json",
            "config.json",
            "hf_quant_config.json",
        ]
        .map(String::from),
    );
    let tokenizer = [
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    .map(String::from);
    let template = sha_hex(&std::fs::read(e.src.join("chat_template.jinja")).unwrap());

    let mut lock: Value =
        serde_json::from_slice(&std::fs::read(fixture("model_src.lock.json")).unwrap()).unwrap();
    lock["model_content_sha256"] = Value::String("1".repeat(64));
    let bad = e.tmp.join("bad.lock.json");
    std::fs::write(&bad, serde_json::to_vec_pretty(&lock).unwrap()).unwrap();
    let o = e.prepare(&["--lock", bad.to_str().unwrap()]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "lock_mismatch")
    );
    assert!(o.stdout.is_empty());
    assert!(e.artifacts().is_empty() && e.temp_dirs().is_empty());

    let r = e.ok(&[]);
    assert_eq!(r["lock"], "not_given");
    assert_eq!(r["model_hash"], lock_digest(&e.src, &model));
    assert_eq!(r["tokenizer_hash"], lock_digest(&e.src, &tokenizer));
    assert_eq!(r["template_hash"], template);
}

#[test]
fn l_global_opcode_token_map_exits_2_and_a_changed_map_is_not_reused() {
    let mut e = Env::new("l");
    let first = e.ok(&[]);
    // data.pkl starts `PROTO 2, EMPTY_LIST, BINPUT 0, MARK, BININT1 0, BININT1 2, BININT1 4, ...`.
    let ids = b"(K\x00K\x02K\x04";
    e.token_map = e.tmp.join("global.pt");
    mutated_token_map(&e.token_map, ids, b"(c");
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "pickle_grammar")
    );

    e.token_map = e.tmp.join("changed.pt");
    mutated_token_map(&e.token_map, ids, b"(K\x00K\x03");
    let r = e.ok(&[]);
    assert_ne!(r["outcome"], "reused");
    assert_ne!(r["artifact_id"], first["artifact_id"]);
    let map = std::fs::read(dir_of(&r).join(TOKEN_OUT)).unwrap();
    assert_eq!(&map[..12], &[0, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);
}

#[test]
fn m_memory_floor_at_start_and_after_the_first_chunk_exits_5() {
    let e = Env::new("m");
    std::fs::write(&e.meminfo, "MemAvailable: 1024 kB\n").unwrap();
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(5), "memory_floor")
    );
    assert!(e.root_entries().iter().all(|n| n == ".cerec.lock"));

    // First read (start) passes, every later read (after each 4 KiB chunk) is below the floor.
    std::fs::write(&e.meminfo, format!("{PLENTY}MemAvailable: 1024 kB\n")).unwrap();
    let o = run(e.cmd(&[]).env("CEREC_TEST_CHUNK_BYTES", "4096"));
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(5), "memory_floor")
    );
    assert!(e.artifacts().is_empty() && e.temp_dirs().is_empty());
}

#[test]
fn n_oversized_token_map_is_rejected_before_it_is_read() {
    let mut e = Env::new("n");
    // Sparse, one byte over the default max_token_map_pkl_bytes (16 MiB) + 1 MiB of zip structure.
    e.token_map = e.tmp.join("huge.pt");
    let f = std::fs::File::create(&e.token_map).unwrap();
    f.set_len((17 << 20) + 1).unwrap();
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "zip_entry")
    );
    let first = String::from_utf8_lossy(&o.stderr);
    assert!(first.contains("bytes > 17825792"), "{first}");
    assert!(e.artifacts().is_empty() && e.temp_dirs().is_empty());
}

/// SelfTune ST4 S3b: `--token-map` is required for a `speculative: "nextn"` profile and refused for `"none"`; a
/// `"none"` artifact has no draft directory and no transform, passes full verify and is reused like any other.
#[test]
fn o_token_map_follows_the_profile_speculative_field() {
    let mut e = Env::new("o");
    e.with_token_map = false;
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "token_map")
    );
    assert!(String::from_utf8_lossy(&o.stderr).contains("needs --token-map"));
    assert!(e.artifacts().is_empty() && e.temp_dirs().is_empty());

    let mut profile: Value =
        serde_json::from_slice(&std::fs::read(fixture("model_src.profile.json")).unwrap()).unwrap();
    profile["speculative"] = Value::String("none".into());
    profile["placement"]
        .as_object_mut()
        .unwrap()
        .insert("draft".into(), Value::String("unused".into()));
    e.profile = e.tmp.join("profile-none.json");
    std::fs::write(
        &e.profile,
        cere_compiler::canon_json::to_vec(&profile).unwrap(),
    )
    .unwrap();
    e.with_token_map = true;
    let o = e.prepare(&[]);
    assert_eq!(
        (o.status.code(), class_of(&o).as_str()),
        (Some(2), "token_map")
    );
    assert!(String::from_utf8_lossy(&o.stderr).contains("speculative none"));
    assert!(e.artifacts().is_empty() && e.temp_dirs().is_empty());

    e.with_token_map = false;
    let r = e.ok(&[]);
    assert_eq!(r["outcome"], "prepared");
    let dir = dir_of(&r);
    assert!(!dir.join("draft").exists());
    let m = manifest_of(&r);
    assert_eq!(m["transforms"], serde_json::json!([]));
    assert!(m["files"]
        .as_array()
        .unwrap()
        .iter()
        .all(|f| f["role"] != "draft_token_map"));
    ok(cerec_verify(&e, &r, "full"));
    ok(run(python("verify_artifact.py")
        .args(["--mode", "full", "--expected-manifest-sha256"])
        .arg(r["manifest_sha256"].as_str().unwrap())
        .arg(&dir)));
    let again = e.ok(&[]);
    assert_eq!(again["outcome"], "reused");
    assert_eq!(again["payload_bytes_read"], 0);
    assert_eq!(again["artifact_id"], r["artifact_id"]);
}
