//! R4 (m21 design note Q7, artifact contract): `cerec verify` and `artifact::open` on copies of the v2
//! `examples/artifact` (written by `cerec prepare`), in temp directories next to their stamp sidecar.
mod support;

use cere_compiler::artifact::{self, Class, Mode, Sink, TensorRef};
use cere_compiler::stamp::{self, FileId, Fs, FsView, Stamp};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Mutex;
use support::*;

/// A writable copy of the example as `<tmp>/<name>`; returns its path and manifest digest.
fn example_copy(t: &Tmp, name: &str) -> (PathBuf, String) {
    let dir = t.join(name);
    copy_tree(&root().join("examples/artifact"), &dir);
    let sha = std::fs::read_to_string(dir.join("manifest.sha256"))
        .unwrap()
        .trim()
        .to_string();
    (dir, sha)
}

/// `cerec verify` with 1 TiB `MemAvailable` injected (the Q9 floor is never the reason a run fails here).
fn verify(t: &Tmp, dir: &Path, sha: &str, mode: &str, extra: &[&str]) -> Output {
    run(cerec()
        .arg("verify")
        .arg(dir)
        .args(["--expected-manifest-sha256", sha, "--mode", mode])
        .args(extra)
        .env("CEREC_TEST_MEMINFO", meminfo(t, 1 << 40)))
}

fn table() -> Value {
    serde_json::from_slice(&std::fs::read(fixture("artifact_invalid.json")).unwrap()).unwrap()
}

/// Applies a table case to a fresh copy of the example with the Python reference applier
/// (`tests/artifact_helpers.py`), so both languages see the same bytes. Returns the digest to verify against.
fn prepare_case(dest: &Path, case: &Value) -> String {
    let code = "import json, sys; from pathlib import Path; sys.path.insert(0, 'tests'); \
                from artifact_helpers import prepare_case; \
                print(prepare_case(Path(sys.argv[2]), json.loads(sys.argv[1])))";
    let out = ok(run(Command::new("python3")
        .current_dir(root())
        .args(["-c", code, &case.to_string()])
        .arg(dest)));
    stdout(&out).trim().to_string()
}

#[test]
fn every_artifact_case_is_rejected_like_python_with_its_class() {
    let t = table();
    let cases: Vec<&Value> = t["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["tag"] == "artifact")
        .collect();
    assert!(cases.len() >= 40, "only {} artifact cases", cases.len());
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let tmp = Tmp::new("vcase");
        let dir = tmp.join("artifact");
        let sha = prepare_case(&dir, case);
        let limits = case.get("limits").map(Value::to_string);
        let extra: Vec<&str> = limits.as_deref().map_or(vec![], |l| vec!["--limits", l]);
        let rust = verify(&tmp, &dir, &sha, "full", &extra);
        let py = run(python("verify_artifact.py")
            .arg(&dir)
            .args(["--expected-manifest-sha256", &sha, "--mode", "full"])
            .args(&extra));
        let want = case["error"].as_str().unwrap();
        assert_eq!(
            (rust.status.code(), class_of(&rust).as_str()),
            (Some(2), want),
            "{id}: cerec stderr {}",
            String::from_utf8_lossy(&rust.stderr)
        );
        assert!(rust.stdout.is_empty(), "{id}: stdout on rejection");
        assert_eq!(
            (py.status.code(), class_of(&py)),
            (rust.status.code(), class_of(&rust)),
            "{id}: python and cerec disagree"
        );
    }
}

#[test]
fn full_mode_accepts_the_example_like_python_and_writes_a_full_verify_stamp() {
    let t = Tmp::new("vfull");
    let (dir, sha) = example_copy(&t, "fixture-model");
    let r = json(&ok(verify(&t, &dir, &sha, "full", &[])));
    let py = json(&ok(run(python("verify_artifact.py").arg(&dir).args([
        "--expected-manifest-sha256",
        &sha,
        "--mode",
        "full",
    ]))));
    assert_eq!(r["outcome"], "verified");
    assert_eq!(r["mode"], "full");
    assert_eq!(r["stamp_written"], "full_verify");
    for key in [
        "manifest_sha256",
        "families",
        "by_dtype",
        "by_placement",
        "files",
        "shards",
        "tensors",
        "misaligned",
        "payload_bytes_read",
        "metadata_bytes_read",
    ] {
        assert_eq!(r[key], py[key], "{key} differs from verify_artifact.py");
    }
    assert!(r["stages_ms"]["total"].is_u64());
    let (s, _) = Stamp::read(&stamp::path_for(&dir).unwrap()).unwrap();
    assert_eq!(s.stamp_source, stamp::FULL_VERIFY);
    assert_eq!(s.manifest_sha256, sha);
    // Python's stamp mode accepts the stamp cerec wrote.
    ok(run(python("verify_artifact.py").arg(&dir).args([
        "--expected-manifest-sha256",
        &sha,
        "--mode",
        "stamp",
    ])));
}

#[test]
fn stamp_mode_reads_no_payload() {
    let t = Tmp::new("vstamp");
    let (dir, sha) = example_copy(&t, "fixture-model");
    ok(verify(&t, &dir, &sha, "full", &[]));
    let r = json(&ok(verify(&t, &dir, &sha, "stamp", &[])));
    assert_eq!(r["mode"], "stamp");
    assert_eq!(r["payload_bytes_read"], 0);
    assert!(r["metadata_bytes_read"].as_u64().unwrap() > 0);
    assert_eq!(r["stamp_written"], Value::Null);
}

#[test]
fn a_prepare_hash_stamp_or_a_touched_file_needs_a_full_verify() {
    let t = Tmp::new("vprep");
    let root_dir = t.join("root");
    std::fs::create_dir(&root_dir).unwrap();
    let out = json(&ok(run(cerec()
        .arg("prepare")
        .arg(model_src())
        .args(["--token-map"])
        .arg(model_src().join("token_map.pt"))
        .args(["--name", "fixture-model", "--families"])
        .arg(fixture("model_src.families.json"))
        .arg("--profile")
        .arg(fixture("model_src.profile.json"))
        .arg("--out")
        .arg(&root_dir)
        .env("CEREC_TEST_MEMINFO", meminfo(&t, 1 << 40)))));
    let dir = PathBuf::from(out["artifact_dir"].as_str().unwrap());
    let sha = out["manifest_sha256"].as_str().unwrap().to_string();
    let r = verify(&t, &dir, &sha, "stamp", &[]);
    assert_eq!((r.status.code(), class_of(&r).as_str()), (Some(4), "stamp"));
    assert!(String::from_utf8_lossy(&r.stderr).contains("full verify required"));
    ok(verify(&t, &dir, &sha, "full", &[]));
    ok(verify(&t, &dir, &sha, "stamp", &[]));
    // `touch` changes mtime/ctime of one held file: the stamp no longer describes it.
    std::thread::sleep(std::time::Duration::from_millis(20));
    ok(run(Command::new("touch").arg(dir.join("config.json"))));
    let r = verify(&t, &dir, &sha, "stamp", &[]);
    assert_eq!((r.status.code(), class_of(&r).as_str()), (Some(4), "stamp"));
    assert!(String::from_utf8_lossy(&r.stderr).contains("full verify required"));
}

/// Q7: every recorded `fstat` field invalidates the stamp on its own. The stamp is rewritten with exactly one field
/// of one file changed; ctime and the other fields stay as they are, so no field can hide behind another.
#[test]
fn each_stamp_identity_field_alone_needs_a_full_verify() {
    let t = Tmp::new("vfield");
    let (dir, sha) = example_copy(&t, "fixture-model");
    ok(verify(&t, &dir, &sha, "full", &[]));
    let path = stamp::path_for(&dir).unwrap();
    let good_bytes = std::fs::read(&path).unwrap();
    let good = Stamp::parse(&good_bytes).unwrap();
    let file = good.files.keys().next().unwrap().clone();
    type Edit = fn(&mut FileId);
    let edits: [(&str, Edit); 6] = [
        ("fs", |s| {
            s.fs = match s.fs {
                Fs::Fsid(x) => Fs::Fsid(x ^ 1),
                Fs::Dev(d) => Fs::Dev(d + 1),
            }
        }),
        ("ino", |s| s.ino += 1),
        ("size", |s| s.size += 1),
        ("mtime_ns", |s| s.mtime_ns += 1),
        ("ctime_ns", |s| s.ctime_ns += 1),
        ("mode", |s| s.mode ^= 0o200),
    ];
    for (field, edit) in edits {
        let mut s = good.clone();
        edit(s.files.get_mut(&file).unwrap());
        std::fs::write(&path, s.to_bytes()).unwrap();
        let r = verify(&t, &dir, &sha, "stamp", &[]);
        assert_eq!(
            (r.status.code(), class_of(&r).as_str()),
            (Some(4), "stamp"),
            "{field}"
        );
        assert!(
            String::from_utf8_lossy(&r.stderr).contains("full verify required"),
            "{field}"
        );
    }
    std::fs::write(&path, &good_bytes).unwrap();
    ok(verify(&t, &dir, &sha, "stamp", &[]));
}

/// The filesystem as `fstatfs` would report it after a reboot: `dev` is the (volatile) device number, `fsid` the
/// UUID-derived filesystem id.
fn fs_view(dev: u64, fsid: u64, uuid_derived: bool) -> impl Fn(u64) -> FsView {
    move |_| FsView {
        dev,
        fsid,
        uuid_derived,
    }
}

/// Stamp mode in process under an injected filesystem view.
fn stamp_under(view: impl Fn(u64) -> FsView + 'static, dir: &Path, sha: &str) -> Result<(), i32> {
    stamp::with_fs_view(view, || {
        artifact::open(dir, sha, Mode::Stamp)
            .map(drop)
            .map_err(|e| {
                assert_eq!(e.class, "stamp", "{}", e.message);
                assert!(e.message.contains("full verify required"), "{}", e.message);
                e.exit
            })
    })
}

/// A full verify in process under an injected filesystem view (it writes the stamp).
fn full_under(view: impl Fn(u64) -> FsView + 'static, dir: &Path, sha: &str) {
    stamp::with_fs_view(view, || {
        let art = artifact::open(dir, sha, Mode::Full).unwrap();
        art.write_full_verify_stamp().unwrap();
    });
}

/// Owner 2026-10-15: a reboot that renumbers the block device (`st_dev` 66311 → 66306, as on 2026-10-14) leaves the
/// same file, inode and filesystem UUID: the stamp still covers it, because it names the filesystem by `fsid`.
#[test]
fn a_reboot_that_renumbers_the_device_keeps_the_stamp() {
    let t = Tmp::new("vreboot");
    let (dir, sha) = example_copy(&t, "fixture-model");
    full_under(fs_view(66311, 0x715f_00ac_f4a6_680f, true), &dir, &sha);
    let (s, _) = Stamp::read(&stamp::path_for(&dir).unwrap()).unwrap();
    assert_eq!(s.format, stamp::FORMAT);
    assert!(s
        .files
        .values()
        .all(|f| f.fs == Fs::Fsid(0x715f_00ac_f4a6_680f)));
    assert_eq!(
        stamp_under(fs_view(66306, 0x715f_00ac_f4a6_680f, true), &dir, &sha),
        Ok(())
    );
}

/// Another filesystem UUID under the same path, inode and device number is another file.
#[test]
fn a_different_filesystem_uuid_needs_a_full_verify() {
    let t = Tmp::new("vuuid");
    let (dir, sha) = example_copy(&t, "fixture-model");
    full_under(fs_view(66306, 0x715f_00ac_f4a6_680f, true), &dir, &sha);
    assert_eq!(
        stamp_under(fs_view(66306, 0x1111_2222_3333_4444, true), &dir, &sha),
        Err(artifact::EXIT_STAMP)
    );
}

/// Without a UUID-derived id (tmpfs, overlay, xfs, ...) the stamp names the file by `dev`, says so, and a renumbered
/// device still needs a full verify: nothing passes on a guess.
#[test]
fn without_a_filesystem_uuid_the_stamp_keeps_the_device_number() {
    let t = Tmp::new("vnouuid");
    let (dir, sha) = example_copy(&t, "fixture-model");
    full_under(fs_view(66311, 0x715f_00ac_f4a6_680f, false), &dir, &sha);
    let (s, _) = Stamp::read(&stamp::path_for(&dir).unwrap()).unwrap();
    assert!(s.files.values().all(|f| f.fs == Fs::Dev(66311)));
    let raw = std::fs::read_to_string(stamp::path_for(&dir).unwrap()).unwrap();
    assert!(raw.contains("\"dev\": 66311") && !raw.contains("fsid"));
    assert_eq!(
        stamp_under(fs_view(66311, 0x715f_00ac_f4a6_680f, false), &dir, &sha),
        Ok(())
    );
    assert_eq!(
        stamp_under(fs_view(66306, 0x715f_00ac_f4a6_680f, false), &dir, &sha),
        Err(artifact::EXIT_STAMP)
    );
}

/// Makes one artifact file writable before the full verify (root could write it anyway).
fn writable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644)).unwrap();
}

/// Flips the first byte of `path` in place (same inode): mtime and ctime change.
fn flip_byte_in_place(path: &Path) {
    use std::os::unix::fs::FileExt;
    std::thread::sleep(std::time::Duration::from_millis(20));
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, 0).unwrap();
    f.write_all_at(&[b[0] ^ 0xff], 0).unwrap();
}

fn assert_full_verify_required(r: &Output) {
    assert_eq!((r.status.code(), class_of(r).as_str()), (Some(4), "stamp"));
    assert!(String::from_utf8_lossy(&r.stderr).contains("full verify required"));
}

#[test]
fn an_in_place_byte_write_needs_a_full_verify() {
    let t = Tmp::new("vwrite");
    let (dir, sha) = example_copy(&t, "fixture-model");
    writable(&dir.join("config.json"));
    ok(verify(&t, &dir, &sha, "full", &[]));
    flip_byte_in_place(&dir.join("config.json"));
    assert_full_verify_required(&verify(&t, &dir, &sha, "stamp", &[]));
}

/// `touch -r` puts the old mtime back after the write; the ctime it cannot restore still refuses the stamp.
#[test]
fn restoring_the_mtime_after_a_write_still_needs_a_full_verify() {
    use std::os::unix::fs::MetadataExt;
    let t = Tmp::new("vtouchr");
    let (dir, sha) = example_copy(&t, "fixture-model");
    let file = dir.join("config.json");
    writable(&file);
    ok(verify(&t, &dir, &sha, "full", &[]));
    let reference = t.join("mtime.ref");
    std::fs::write(&reference, b"").unwrap();
    ok(run(Command::new("touch")
        .arg("-r")
        .arg(&file)
        .arg(&reference)));
    let before = std::fs::metadata(&file).unwrap();
    flip_byte_in_place(&file);
    ok(run(Command::new("touch")
        .arg("-r")
        .arg(&reference)
        .arg(&file)));
    let after = std::fs::metadata(&file).unwrap();
    assert_eq!(
        (after.ino(), after.size(), after.mtime(), after.mtime_nsec()),
        (
            before.ino(),
            before.size(),
            before.mtime(),
            before.mtime_nsec()
        )
    );
    assert_ne!(
        (after.ctime(), after.ctime_nsec()),
        (before.ctime(), before.ctime_nsec())
    );
    assert_full_verify_required(&verify(&t, &dir, &sha, "stamp", &[]));
}

/// An identical-content copy renamed over the file, with its mtime and mode copied too: only the inode (and ctime)
/// tell it apart, and that is enough.
#[test]
fn an_identical_copy_renamed_over_a_file_needs_a_full_verify() {
    let t = Tmp::new("vrename");
    let (dir, sha) = example_copy(&t, "fixture-model");
    ok(verify(&t, &dir, &sha, "full", &[]));
    let file = dir.join("config.json");
    let copy = dir.join("config.json.copy");
    std::fs::copy(&file, &copy).unwrap();
    ok(run(Command::new("touch").arg("-r").arg(&file).arg(&copy)));
    std::fs::rename(&copy, &file).unwrap();
    assert_full_verify_required(&verify(&t, &dir, &sha, "stamp", &[]));
}

/// A stamp written before the fs-uuid format (format 1: `dev` per file, no `format` key) is still accepted while
/// every `st_dev` matches; after a renumbering it is refused as predating the format, never accepted.
#[test]
fn an_old_format_stamp_is_accepted_only_on_a_device_match() {
    use std::os::unix::fs::MetadataExt;
    let t = Tmp::new("vlegacy");
    let (dir, sha) = example_copy(&t, "fixture-model");
    ok(verify(&t, &dir, &sha, "full", &[]));
    let path = stamp::path_for(&dir).unwrap();
    let mut legacy = Stamp::parse(&std::fs::read(&path).unwrap()).unwrap();
    legacy.format = stamp::LEGACY_FORMAT;
    for (p, f) in &mut legacy.files {
        f.fs = Fs::Dev(std::fs::metadata(dir.join(p)).unwrap().dev());
    }
    let bytes = legacy.to_bytes();
    let raw: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(raw.get("format").is_none());
    assert!(raw["files"]
        .as_object()
        .unwrap()
        .values()
        .all(|f| f["dev"].is_u64() && f.get("fsid").is_none()));
    std::fs::write(&path, &bytes).unwrap();
    let python_stamp = || {
        run(python("verify_artifact.py").arg(&dir).args([
            "--expected-manifest-sha256",
            &sha,
            "--mode",
            "stamp",
        ]))
    };
    ok(verify(&t, &dir, &sha, "stamp", &[]));
    ok(python_stamp());
    let first = legacy.files.keys().next().unwrap().clone();
    if let Fs::Dev(d) = legacy.files[&first].fs {
        legacy.files.get_mut(&first).unwrap().fs = Fs::Dev(d + 1);
    }
    std::fs::write(&path, legacy.to_bytes()).unwrap();
    let r = verify(&t, &dir, &sha, "stamp", &[]);
    assert_full_verify_required(&r);
    assert!(String::from_utf8_lossy(&r.stderr)
        .contains("stamp predates fs-uuid format; full verify required"));
    let py = python_stamp();
    assert_ne!(py.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&py.stderr)
        .contains("stamp predates fs-uuid format; full verify required"));
    // A new full verify writes the new format.
    ok(verify(&t, &dir, &sha, "full", &[]));
    let (s, _) = Stamp::read(&path).unwrap();
    assert_eq!(s.format, stamp::FORMAT);
    ok(verify(&t, &dir, &sha, "stamp", &[]));
}

/// Collects every staged piece, checking the sink contract on the way.
#[derive(Default)]
struct Collect(Mutex<BTreeMap<String, Vec<u8>>>);

impl Sink for Collect {
    fn accept(
        &self,
        t: &TensorRef<'_>,
        offset: u64,
        data: &[u8],
        is_last: bool,
    ) -> Result<(), String> {
        // Pieces of one tensor may arrive in any order (m216 design note L7); each lands at its offset.
        let mut m = self.0.lock().unwrap();
        let v = m
            .entry(format!("{}:{}", t.path, t.name))
            .or_insert_with(|| vec![0; t.bytes as usize]);
        let at = offset as usize;
        v[at..at + data.len()].copy_from_slice(data);
        assert_eq!(is_last, offset + data.len() as u64 == t.bytes);
        Ok(())
    }
}

#[test]
fn a_file_swapped_by_rename_after_open_is_not_read() {
    let t = Tmp::new("vswap");
    let (dir, sha) = example_copy(&t, "fixture-model");
    ok(verify(&t, &dir, &sha, "full", &[]));
    let shard = "model-bf16-00001.safetensors";
    let original = std::fs::read(dir.join(shard)).unwrap();
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    // Replace the shard under the same name: a new inode with every data byte flipped.
    let mut swapped = original.clone();
    let start = 8 + u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    for b in &mut swapped[start..] {
        *b ^= 0xff;
    }
    std::fs::write(dir.join("swap.tmp"), &swapped).unwrap();
    std::fs::rename(dir.join("swap.tmp"), dir.join(shard)).unwrap();
    let sink = Collect::default();
    let mut bufs = artifact::buffers(2, artifact::MIN_BUFFER_BYTES).unwrap();
    art.stage(Class::All, &mut bufs, &sink, None).unwrap();
    let staged = sink.0.into_inner().unwrap();
    let header = art.header(shard).unwrap();
    for ti in &header.tensors {
        let begin = ti.offset as usize;
        assert_eq!(
            staged[&format!("{shard}:{}", ti.name)],
            original[begin..begin + ti.bytes as usize],
            "{} was read from the swapped file",
            ti.name
        );
    }
}

#[test]
fn a_symlinked_artifact_file_is_rejected_in_stamp_mode() {
    let t = Tmp::new("vlink");
    let (dir, sha) = example_copy(&t, "fixture-model");
    ok(verify(&t, &dir, &sha, "full", &[]));
    std::fs::rename(dir.join("vocab.json"), t.join("vocab.json")).unwrap();
    std::os::unix::fs::symlink(t.join("vocab.json"), dir.join("vocab.json")).unwrap();
    let r = verify(&t, &dir, &sha, "stamp", &[]);
    assert_eq!(
        (r.status.code(), class_of(&r).as_str()),
        (Some(2), "symlink")
    );
    match artifact::open(&dir, &sha, Mode::Stamp) {
        Err(e) => assert_eq!(e.class, "symlink"),
        Ok(_) => panic!("a symlinked file was accepted"),
    }
}

#[test]
fn cold_cache_output_carries_a_residency_label() {
    let t = Tmp::new("vcold");
    let (dir, sha) = example_copy(&t, "fixture-model");
    let r = json(&ok(verify(&t, &dir, &sha, "full", &["--cache", "cold"])));
    let cache = &r["cache"];
    assert!(["cold", "cold_unverified"].contains(&cache["label"].as_str().unwrap()));
    let res = &cache["residency_before"];
    assert!(res["resident_pages"].is_u64() && res["total_pages"].as_u64().unwrap() > 0);
    assert!(cache["evicted_files"].as_u64().unwrap() >= 15);
}

/// Q9: the one floor `MemAvailable ≥ G GiB + staging` (staging: 8 hash threads × 8 MiB) is checked at start and
/// after every chunk of payload read; below it verify exits 5 `memory_floor` and writes no stamp.
#[test]
fn full_verify_below_the_memory_floor_exits_5_without_a_stamp() {
    let t = Tmp::new("vfloor");
    let (dir, sha) = example_copy(&t, "fixture-model");
    let stamp = t.join("fixture-model.verified.json");
    let floor = (16u64 << 30) + (64 << 20);
    let low = t.join("low");
    std::fs::write(&low, "MemAvailable: 1024 kB\n").unwrap();
    // Second read (after the first 4 KiB chunk) is below the floor; the start check passes.
    let late = t.join("late");
    std::fs::write(
        &late,
        "MemAvailable: 1073741824 kB\nMemAvailable: 1024 kB\n",
    )
    .unwrap();
    for (meminfo, chunk) in [(&low, None), (&late, Some("4096"))] {
        let mut c = cerec();
        c.arg("verify")
            .arg(&dir)
            .args(["--expected-manifest-sha256", &sha, "--mode", "full"])
            .env("CEREC_TEST_MEMINFO", meminfo);
        if let Some(n) = chunk {
            c.env("CEREC_TEST_CHUNK_BYTES", n);
        }
        let out = run(&mut c);
        assert_eq!(
            (out.status.code(), class_of(&out).as_str()),
            (Some(5), "memory_floor"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.stdout.is_empty());
        assert!(!stamp.exists());
    }
    let r = json(&ok(verify(&t, &dir, &sha, "full", &[])));
    assert_eq!(r["memory_floor"], floor);
    assert_eq!(r["mem_available_before"], 1u64 << 40);
    assert!(stamp.exists());
}

#[test]
fn library_open_exposes_the_artifact_contract() {
    let t = Tmp::new("vlib");
    let (dir, sha) = example_copy(&t, "fixture-model");
    let art = artifact::open(&dir, &sha, Mode::Full).unwrap();
    assert_eq!(art.manifest_sha256, sha);
    assert_eq!(art.manifest.profile_id, art.profile.profile_id);
    assert_eq!(art.files.len(), art.manifest.files.len());
    let tensors: u64 = art.tensors().count() as u64;
    let family_count: u64 = art.manifest.families.iter().map(|f| f.count).sum();
    assert_eq!(tensors, family_count);
    let map = art.token_map().unwrap();
    assert!(!map.is_empty() && map.windows(2).all(|w| w[0] < w[1]));
    // Full open hashes every listed byte once (the header prefixes count as metadata); the token map is kept from
    // that pass, not read a second time (Q7: small payload files are read once).
    let listed: u64 = art.files.iter().map(|f| f.bytes).sum();
    let headers: u64 = art
        .manifest
        .files
        .iter()
        .filter_map(|f| f.safetensors.as_ref())
        .map(|s| s.header_bytes)
        .sum();
    assert_eq!(art.counters().payload_bytes_read, listed - headers);
}
