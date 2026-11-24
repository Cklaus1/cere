//! R5 (m21 design note Q8, Q9, artifact contract "Read"): `cerec probe-load`, `cerec evict` and the
//! `Artifact::stage` sink contract, on a copy of the v2 `examples/artifact` in a temp directory. L7
//! (m216 design note): one shared read queue, pieces in any order, N sinks fed from one read, a buffer reused only
//! after every sink released it, reads overlapping the puts, direct reads, and the read error paths.
mod support;

use cere_compiler::artifact::{self, Class, Mode, ReadMode, Sink, StageOptions, TensorRef};
use cere_compiler::families::Role;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Output;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};
use support::*;

const GIB: u64 = 1 << 30;

/// A verified (full_verify stamp) copy of the example; returns its path and manifest digest.
fn verified_copy(t: &Tmp) -> (PathBuf, String) {
    let dir = t.join("fixture-model");
    copy_tree(&root().join("examples/artifact"), &dir);
    let sha = std::fs::read_to_string(dir.join("manifest.sha256"))
        .unwrap()
        .trim()
        .to_string();
    ok(run(cerec()
        .arg("verify")
        .arg(&dir)
        .args(["--expected-manifest-sha256", &sha, "--mode", "full"])
        .env("CEREC_TEST_MEMINFO", meminfo(t, 64 * GIB))));
    (dir, sha)
}

fn probe(t: &Tmp, dir: &Path, sha: &str, args: &[&str]) -> Output {
    run(cerec()
        .arg("probe-load")
        .arg(dir)
        .args(["--expected-manifest-sha256", sha])
        .args(args)
        .env("CEREC_TEST_MEMINFO", meminfo(t, 64 * GIB)))
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap()
}

fn family_bytes(m: &Value, keep: impl Fn(&Value) -> bool) -> u64 {
    m["families"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| keep(f))
        .map(|f| f["bytes"].as_u64().unwrap())
        .sum()
}

#[test]
fn host_class_reads_exactly_the_ple_tables_and_all_reads_every_tensor_once() {
    let t = Tmp::new("pclass");
    let (dir, sha) = verified_copy(&t);
    let m = manifest(&dir);
    let host = family_bytes(&m, |f| f["placement"] == "host_pinned");
    let ple = family_bytes(&m, |f| {
        ["ple_table", "ple_scale", "ple_index"].contains(&f["role"].as_str().unwrap())
    });
    assert_eq!(
        host, ple,
        "the fixture profile pins exactly the PLE table/scale/index"
    );
    let all = family_bytes(&m, |_| true);
    for (class, want) in [("host", host), ("all", all)] {
        let r = json(&ok(probe(
            &t,
            &dir,
            &sha,
            &["--class", class, "--cache", "warm", "--hash", "off"],
        )));
        assert_eq!(r["payload_bytes_read"], want, "class {class}");
        assert_eq!(r["class"], class);
        assert_eq!(r["outcome"], "probed");
        assert_eq!(r["hash"], false);
        for key in ["discover", "validate", "read", "total"] {
            assert!(r["stages_ms"][key].is_u64(), "stages_ms.{key}");
        }
        assert!(r["staging_high_water_bytes"].as_u64().unwrap() > 0);
        assert!(r["peak_rss_bytes"].as_u64().unwrap() > 0);
        assert_eq!(r["mem_available_before"], 64 * GIB);
        for key in ["residency_before", "residency_after"] {
            assert!(r[key]["resident_pages"].is_u64() && r[key]["total_pages"].is_u64());
        }
    }
    let device = json(&ok(probe(
        &t,
        &dir,
        &sha,
        &["--class", "device", "--cache", "cold", "--hash", "on"],
    )));
    assert_eq!(device["hash"], true);
    assert!(device["stages_ms"]["hash"].is_u64());
    assert!(["cold", "cold_unverified"].contains(&device["label"].as_str().unwrap()));
}

#[test]
fn baseline_reads_the_same_bytes_as_the_matching_run() {
    let t = Tmp::new("pbase");
    let (dir, sha) = verified_copy(&t);
    for class in ["device", "host", "all"] {
        let args = ["--class", class, "--cache", "warm", "--hash", "off"];
        let normal = json(&ok(probe(&t, &dir, &sha, &args)));
        let mut with = args.to_vec();
        with.push("--baseline");
        let base = json(&ok(probe(&t, &dir, &sha, &with)));
        assert_eq!(base["baseline"], true);
        assert_eq!(
            base["payload_bytes_read"], normal["payload_bytes_read"],
            "class {class}"
        );
        assert_eq!(base["tensors"], normal["tensors"], "class {class}");
        assert!(
            base["stages_ms"].get("validate").is_none(),
            "baseline has no VALIDATE stage"
        );
    }
}

#[test]
fn memory_floor_refuses_with_injected_mem_available() {
    let t = Tmp::new("pfloor");
    let (dir, sha) = verified_copy(&t);
    let r = run(cerec()
        .arg("probe-load")
        .arg(&dir)
        .args([
            "--expected-manifest-sha256",
            &sha,
            "--class",
            "all",
            "--cache",
            "warm",
        ])
        .args(["--hash", "off"])
        .env("CEREC_TEST_MEMINFO", meminfo(&t, 16 * GIB)));
    assert_eq!(
        (r.status.code(), class_of(&r).as_str()),
        (Some(5), "memory_floor")
    );
    assert!(r.stdout.is_empty());
}

/// Warms the verified copy at `dir` (reads every tensor), runs `cerec evict` on it and returns its report and
/// `residency_after` as (resident, total) pages.
fn warm_then_evict(t: &Tmp, dir: &Path, sha: &str) -> (Value, u64, u64) {
    ok(probe(
        t,
        dir,
        sha,
        &["--class", "all", "--cache", "warm", "--hash", "off"],
    ));
    let r = json(&ok(run(cerec().arg("evict").arg(dir))));
    let res = &r["residency_after"];
    let (resident, total) = (
        res["resident_pages"].as_u64().unwrap(),
        res["total_pages"].as_u64().unwrap(),
    );
    (r, resident, total)
}

/// On a file system whose pages cannot be evicted (tmpfs, ramfs: `fs_kind`), the eviction bound does not apply. The
/// test says so on the process's real stderr (libtest captures only `print!`/`eprint!`, so the line reaches the log
/// of a passing run) and checks nothing else; everywhere else the bound is checked unchanged (O314).
#[test]
fn evict_leaves_the_artifact_at_most_one_percent_resident() {
    let t = Tmp::new("pevict");
    let kind = fs_kind(t.path());
    if !kind.evictable {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "NOT APPLICABLE: evict_leaves_the_artifact_at_most_one_percent_resident: {} is on {kind}, whose pages \
             cannot be evicted; the eviction bound is not checked there",
            t.path().display()
        );
        return;
    }
    let (dir, sha) = verified_copy(&t);
    let (r, resident, total) = warm_then_evict(&t, &dir, &sha);
    assert!(
        total > 0 && resident * 100 <= total,
        "{resident}/{total} pages resident"
    );
    assert!(r["files"].as_u64().unwrap() >= 15);
}

/// `fs_kind` names tmpfs and ramfs as not evictable and every other file system (named or not) as evictable.
#[test]
fn fs_kind_names_only_tmpfs_and_ramfs_unevictable() {
    let got: Vec<(&str, bool)> = [
        libc::TMPFS_MAGIC,
        0x858458f6,
        libc::EXT4_SUPER_MAGIC,
        libc::XFS_SUPER_MAGIC,
        libc::BTRFS_SUPER_MAGIC,
        libc::OVERLAYFS_SUPER_MAGIC,
        0x1234,
    ]
    .into_iter()
    .map(|m| {
        let k = FsKind::of_magic(m);
        (k.name, k.evictable)
    })
    .collect();
    assert_eq!(
        got,
        [
            ("tmpfs", false),
            ("ramfs", false),
            ("ext4", true),
            ("xfs", true),
            ("btrfs", true),
            ("overlayfs", true),
            ("unknown", true)
        ]
    );
}

/// The detection agrees with a real eviction on both kinds of file system: `/dev/shm` is tmpfs and `cerec evict`
/// leaves the artifact fully resident there; in `CARGO_TARGET_TMPDIR` (a disk file system unless the target is on a
/// tmpfs) eviction meets the 1 % bound exactly when `fs_kind` says the file system is evictable.
#[test]
fn the_evictability_check_matches_cerec_evict_on_tmpfs_and_the_target_dir() {
    let shm = fs_kind(Path::new("/dev/shm"));
    assert_eq!((shm.name, shm.evictable), ("tmpfs", false), "{shm}");
    for base in [
        Path::new("/dev/shm"),
        Path::new(env!("CARGO_TARGET_TMPDIR")),
    ] {
        let t = Tmp::under(base, "pfskind");
        let kind = fs_kind(t.path());
        let (dir, sha) = verified_copy(&t);
        let (_, resident, total) = warm_then_evict(&t, &dir, &sha);
        assert!(total > 0, "{}", base.display());
        assert_eq!(
            resident * 100 <= total,
            kind.evictable,
            "{} on {kind}: {resident}/{total} pages resident after evict",
            base.display()
        );
        if !kind.evictable {
            assert_eq!(resident, total, "{} on {kind}", base.display());
        }
    }
}

/// One staged piece: (tensor, offset, len, is_last, buffer address).
type Piece = (String, u64, u64, bool, usize);

/// Records every piece.
#[derive(Default)]
struct Record(Mutex<Vec<Piece>>);

impl Sink for Record {
    fn accept(
        &self,
        t: &TensorRef<'_>,
        offset: u64,
        data: &[u8],
        is_last: bool,
    ) -> Result<(), String> {
        self.0.lock().unwrap().push((
            format!("{}:{}", t.path, t.name),
            offset,
            data.len() as u64,
            is_last,
            data.as_ptr() as usize,
        ));
        Ok(())
    }
}

/// L7: a tensor larger than a buffer is cut into buffer-sized pieces, each its own read from the shared queue; the
/// pieces tile it exactly once (any order), the one that ends it carries `is_last`, every buffer is aligned.
#[test]
fn pieces_of_a_large_tensor_tile_it_exactly_once_in_any_order() {
    let t = Tmp::new("ppieces");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let mut bufs = artifact::buffers(3, artifact::MIN_BUFFER_BYTES).unwrap();
    for b in &mut bufs {
        assert_eq!(b.as_mut_slice().as_ptr() as usize % 4096, 0);
        assert_eq!(b.as_mut_slice().len(), 64 << 10);
    }
    let sink = Record::default();
    let report = art.stage(Class::All, &mut bufs, &sink, None).unwrap();
    let pieces = sink.0.into_inner().unwrap();
    assert_eq!(report.tensors as usize, art.tensors().count());
    let big = art.tensors().max_by_key(|t| t.bytes).expect("tensors");
    assert!(
        big.bytes > 2 * (64 << 10),
        "{} spans fewer than 3 buffers",
        big.name
    );
    let key = format!("{}:{}", big.path, big.name);
    let mut mine: Vec<_> = pieces.iter().filter(|p| p.0 == key).collect();
    assert_eq!(mine.len() as u64, big.bytes.div_ceil(64 << 10));
    mine.sort_by_key(|p| p.1);
    let mut next = 0;
    for p in &mine {
        assert_eq!(p.1, next, "the pieces tile the tensor exactly once");
        assert_eq!(p.1 % (64 << 10), 0, "pieces start at buffer multiples");
        assert_eq!(p.4 % 4096, 0, "piece buffers are 4096-aligned");
        next += p.2;
    }
    assert_eq!(next, big.bytes);
    assert_eq!(mine.iter().filter(|p| p.3).count(), 1);
    assert!(mine.last().unwrap().3, "is_last on the piece that ends it");
    // Every tensor: exactly one is_last, total bytes = its size.
    for t in art.tensors() {
        let k = format!("{}:{}", t.path, t.name);
        let ps: Vec<_> = pieces.iter().filter(|p| p.0 == k).collect();
        assert_eq!(ps.iter().map(|p| p.2).sum::<u64>(), t.bytes, "{k}");
        assert_eq!(ps.iter().filter(|p| p.3).count(), 1, "{k}");
    }
}

#[test]
fn a_sink_error_aborts_staging() {
    struct Fail;
    impl Sink for Fail {
        fn accept(&self, _: &TensorRef<'_>, _: u64, _: &[u8], _: bool) -> Result<(), String> {
            Err("device full".into())
        }
    }
    let t = Tmp::new("psinkerr");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let mut bufs = artifact::buffers(2, artifact::MIN_BUFFER_BYTES).unwrap();
    let e = art.stage(Class::All, &mut bufs, &Fail, None).unwrap_err();
    assert_eq!(e.class, "sink");
    assert!(e.message.contains("device full"));
}

/// A sink that panics stops staging and the panic reaches the caller (no reader left waiting for a buffer that
/// the panicked delivery thread still holds).
#[test]
fn a_panicking_sink_propagates_instead_of_hanging() {
    struct Boom;
    impl Sink for Boom {
        fn accept(&self, _: &TensorRef<'_>, _: u64, _: &[u8], _: bool) -> Result<(), String> {
            panic!("sink bug");
        }
    }
    let t = Tmp::new("psinkpanic");
    let (dir, sha) = verified_copy(&t);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
        let mut bufs = artifact::buffers(2, artifact::MIN_BUFFER_BYTES).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = art.stage(Class::All, &mut bufs, &Boom, None);
        }));
        let _ = tx.send(r.is_err());
    });
    let panicked = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("staging hung after a sink panic");
    assert!(panicked, "the sink's panic did not reach the caller");
}

#[test]
fn buffers_below_64_kib_or_unaligned_sizes_are_refused() {
    assert!(artifact::buffers(1, 4096).is_err());
    assert!(artifact::buffers(1, (64 << 10) + 1).is_err());
    assert!(artifact::buffers(0, 64 << 10).is_err());
}

/// A tensor's delivered pieces: offset -> (bytes, is_last).
type Pieces = BTreeMap<u64, (Vec<u8>, bool)>;

/// Keeps every delivered piece per tensor (the fixture is small) and declines the tensors of `decline`; a piece
/// delivered twice is an error.
struct Keep {
    decline: Option<Role>,
    got: Mutex<BTreeMap<String, Pieces>>,
}

impl Keep {
    fn new(decline: Option<Role>) -> Keep {
        Keep {
            decline,
            got: Mutex::default(),
        }
    }

    /// Each tensor's bytes, its pieces in offset order; panics unless they tile it with one `is_last` on the end.
    fn into_bytes(self) -> BTreeMap<String, Vec<u8>> {
        let mut out = BTreeMap::new();
        for (key, pieces) in self.got.into_inner().unwrap() {
            let mut v = Vec::new();
            let lasts: Vec<u64> = pieces
                .iter()
                .filter(|(_, (_, last))| *last)
                .map(|(&off, _)| off)
                .collect();
            let end = *pieces.keys().next_back().unwrap();
            assert_eq!(lasts, vec![end], "{key}: is_last only on the final piece");
            for (off, (data, _)) in pieces {
                assert_eq!(off, v.len() as u64, "{key}: a gap or an overlap");
                v.extend_from_slice(&data);
            }
            out.insert(key, v);
        }
        out
    }
}

impl Sink for Keep {
    fn wants(&self, t: &TensorRef<'_>) -> bool {
        Some(t.role) != self.decline
    }

    fn accept(&self, t: &TensorRef<'_>, off: u64, data: &[u8], last: bool) -> Result<(), String> {
        let key = format!("{}:{}", t.path, t.name);
        if self.decline == Some(t.role) {
            return Err(format!("{key} was declined but delivered"));
        }
        let mut got = self.got.lock().unwrap();
        if got
            .entry(key.clone())
            .or_default()
            .insert(off, (data.to_vec(), last))
            .is_some()
        {
            return Err(format!("{key}: the piece at {off} arrived twice"));
        }
        Ok(())
    }
}

/// Each tensor's bytes read straight from its file.
fn direct(dir: &Path, t: &TensorRef<'_>) -> Vec<u8> {
    let raw = std::fs::read(dir.join(t.path)).unwrap();
    raw[t.offset as usize..(t.offset + t.bytes) as usize].to_vec()
}

/// L1 (m216 design note C1): a tensor the sink declines is never read.
#[test]
fn a_declined_tensor_is_never_read() {
    let t = Tmp::new("pdecline");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let draft: u64 = art
        .tensors()
        .filter(|t| t.role == Role::Draft)
        .map(|t| t.bytes)
        .sum();
    assert!(draft > 0, "the fixture has a draft family");
    let all: u64 = art.tensors().map(|t| t.bytes).sum();
    let mut bufs = artifact::buffers(2, artifact::MIN_BUFFER_BYTES).unwrap();
    let sink = Keep::new(Some(Role::Draft));
    let before = art.counters().payload_bytes_read;
    let r = art.stage(Class::All, &mut bufs, &sink, None).unwrap();
    assert_eq!(art.counters().payload_bytes_read - before, all - draft);
    assert_eq!(r.bytes, all - draft);
    let got = sink.into_bytes();
    for t in art.tensors() {
        let key = format!("{}:{}", t.path, t.name);
        if t.role == Role::Draft {
            assert!(!got.contains_key(&key), "{key} delivered");
        } else if t.bytes > 0 {
            assert_eq!(got[&key], direct(&dir, &t), "{key}");
        }
    }
    assert_eq!(
        r.tensors,
        art.tensors().filter(|t| t.role != Role::Draft).count() as u64
    );
}

/// load-soft design note LSF-01: the tensors a sink calls late are read after every other one (one reader, one
/// buffer, one consumer: delivery in plan order), and every tensor's bytes are delivered as without the hint.
#[test]
fn late_tensors_are_read_after_the_others_with_the_same_bytes() {
    struct Late {
        path: String,
        keep: Keep,
        order: Mutex<Vec<bool>>,
    }
    impl Sink for Late {
        fn late(&self, t: &TensorRef<'_>) -> bool {
            t.path == self.path
        }
        fn accept(
            &self,
            t: &TensorRef<'_>,
            off: u64,
            data: &[u8],
            last: bool,
        ) -> Result<(), String> {
            self.order.lock().unwrap().push(t.path == self.path);
            self.keep.accept(t, off, data, last)
        }
    }
    let t = Tmp::new("plate");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    // The first file in path order, so the hint moves its tensors behind the other files'.
    let path = art
        .tensors()
        .filter(|t| t.bytes > 0)
        .map(|t| t.path)
        .min()
        .unwrap()
        .to_owned();
    let sink = Late {
        path,
        keep: Keep::new(None),
        order: Mutex::default(),
    };
    let mut bufs = artifact::buffers(1, artifact::MIN_BUFFER_BYTES).unwrap();
    let opts = read_opts(ReadMode::Buffered, 1, 1);
    art.stage(Class::All, &mut bufs, &sink, Some(&opts))
        .unwrap();
    let order = sink.order.into_inner().unwrap();
    let first_late = order.iter().position(|&late| late).expect("a late tensor");
    assert!(first_late > 0, "an early tensor comes first");
    assert!(
        order[first_late..].iter().all(|&late| late),
        "an early tensor after a late one: {order:?}"
    );
    let got = sink.keep.into_bytes();
    for t in art.tensors().filter(|t| t.bytes > 0) {
        let key = format!("{}:{}", t.path, t.name);
        assert_eq!(got[&key], direct(&dir, &t), "{key}");
    }
}

/// L1 with `hash`: the declined tensors of a hashed file are read and hashed but not delivered, so a changed byte
/// inside one still fails `file_hash`.
#[test]
fn a_hashed_file_still_hashes_its_declined_tensors() {
    let t = Tmp::new("pdeclhash");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let all: u64 = art.tensors().map(|t| t.bytes).sum();
    let hash = StageOptions {
        hash: true,
        ..Default::default()
    };
    let mut bufs = artifact::buffers(2, artifact::MIN_BUFFER_BYTES).unwrap();
    let sink = Keep::new(Some(Role::Draft));
    let before = art.counters().payload_bytes_read;
    let r = art
        .stage(Class::All, &mut bufs, &sink, Some(&hash))
        .unwrap();
    assert_eq!(art.counters().payload_bytes_read - before, all);
    let weights = manifest(&dir)["files"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["role"] == "weights")
        .count() as u64;
    assert_eq!(r.hashed_files, weights);
    // Flip one byte inside a draft tensor behind the open artifact (its descriptors read the new byte).
    let d = art
        .tensors()
        .find(|t| t.role == Role::Draft && t.bytes > 0)
        .unwrap();
    let path = dir.join(d.path);
    let mut raw = std::fs::read(&path).unwrap();
    raw[d.offset as usize] ^= 0x01;
    std::fs::write(&path, &raw).unwrap();
    let e = art
        .stage(
            Class::All,
            &mut bufs,
            &Keep::new(Some(Role::Draft)),
            Some(&hash),
        )
        .unwrap_err();
    assert_eq!(e.class, "file_hash", "{e:?}");
}

/// L2 (m216 design note C2): contiguous small tensors share one read, every tensor still arrives whole (tiled once,
/// one `is_last`), bit-identical to its file bytes, for 1 to 32 buffers.
#[test]
fn runs_of_small_tensors_share_reads_and_every_buffer_count_delivers_the_same_bytes() {
    let t = Tmp::new("pruns");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let tensors = art.tensors().count() as u64;
    let big = art.tensors().filter(|t| t.bytes > 64 << 10).count() as u64;
    for n in [1, 2, 4, 32] {
        let mut bufs = artifact::buffers(n, artifact::MIN_BUFFER_BYTES).unwrap();
        let sink = Keep::new(None);
        let r = art.stage(Class::All, &mut bufs, &sink, None).unwrap();
        assert_eq!(r.tensors, tensors, "{n} buffers");
        assert!(
            r.threads >= 1 && r.threads <= n,
            "{n} buffers: {} threads",
            r.threads
        );
        assert!(
            r.reads < tensors,
            "{n} buffers: {} reads for {tensors} tensors",
            r.reads
        );
        assert!(r.high_water_bytes <= (n * artifact::MIN_BUFFER_BYTES) as u64);
        let got = sink.into_bytes();
        for t in art.tensors().filter(|t| t.bytes > 0) {
            let key = format!("{}:{}", t.path, t.name);
            assert_eq!(got[&key], direct(&dir, &t), "{n} buffers: {key}");
        }
        assert!(big >= 1, "the fixture has a tensor larger than one buffer");
    }
}

/// Whether the file system of `dir` opens files `O_DIRECT` (tmpfs before Linux 6.6 does not).
fn direct_supported(dir: &Path) -> bool {
    use std::os::unix::fs::OpenOptionsExt;
    let probe = dir.join("manifest.json");
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(probe)
        .is_ok()
}

fn read_opts(read: ReadMode, readers: usize, consumers: usize) -> StageOptions<'static> {
    StageOptions {
        read,
        readers,
        consumers,
        ..StageOptions::default()
    }
}

/// L7: `O_DIRECT` reads (aligned spans through a second descriptor of the same inode) deliver exactly the buffered
/// bytes and count only the tensor bytes as payload; `auto` reads the resident files buffered and, after an
/// eviction, every read file direct.
#[test]
fn direct_and_auto_reads_deliver_the_buffered_bytes() {
    let t = Tmp::new("pdirect");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let all: u64 = art.tensors().map(|t| t.bytes).sum();
    let files = art.tensors().map(|t| t.path).collect::<BTreeSet<_>>().len() as u64;
    let expect_direct = if direct_supported(&dir) { files } else { 0 };
    for n in [1, 3, 8] {
        let mut bufs = artifact::buffers(n, artifact::MIN_BUFFER_BYTES).unwrap();
        let sink = Keep::new(None);
        let before = art.counters().payload_bytes_read;
        let r = art
            .stage(
                Class::All,
                &mut bufs,
                &sink,
                Some(&read_opts(ReadMode::Direct, 0, 0)),
            )
            .unwrap();
        assert_eq!(r.direct_files, expect_direct, "{n} buffers");
        assert_eq!(art.counters().payload_bytes_read - before, all);
        assert_eq!(r.bytes, all);
        let got = sink.into_bytes();
        for t in art.tensors().filter(|t| t.bytes > 0) {
            let key = format!("{}:{}", t.path, t.name);
            assert_eq!(got[&key], direct(&dir, &t), "direct, {n} buffers: {key}");
        }
    }
    let mut bufs = artifact::buffers(4, artifact::MIN_BUFFER_BYTES).unwrap();
    // Just read buffered: every file is resident, so auto reads none direct.
    let auto = read_opts(ReadMode::Auto, 0, 0);
    art.stage(Class::All, &mut bufs, &Keep::new(None), None)
        .unwrap();
    let r = art
        .stage(Class::All, &mut bufs, &Keep::new(None), Some(&auto))
        .unwrap();
    assert_eq!(r.direct_files, 0);
    let (_, after) = artifact::evict_dir(&dir, Some(&sha), &Default::default()).unwrap();
    let sink = Keep::new(None);
    let r = art
        .stage(Class::All, &mut bufs, &sink, Some(&auto))
        .unwrap();
    // A file system that keeps evicted pages (tmpfs) leaves them resident: auto then reads them buffered.
    let cold = after.basis_points() <= 100;
    assert_eq!(r.direct_files, if cold { expect_direct } else { 0 });
    let got = sink.into_bytes();
    for t in art.tensors().filter(|t| t.bytes > 0) {
        let key = format!("{}:{}", t.path, t.name);
        assert_eq!(got[&key], direct(&dir, &t), "auto after evict: {key}");
    }
}

/// Bitwise CRC-32 (IEEE, reflected 0xEDB88320), the reference the probe's table + combine must equal.
fn crc32_ref(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// The `tensor_digests` summary computed from the files alone: one line `path\toffset\tbytes\tcrc32` per non-empty
/// tensor in (path, offset) order, then SHA-256 over the lines.
fn digest_ref(dir: &Path, art: &artifact::Artifact) -> (u64, u64, String) {
    let mut ts: Vec<_> = art.tensors().filter(|t| t.bytes > 0).collect();
    ts.sort_by(|a, b| (a.path, a.offset).cmp(&(b.path, b.offset)));
    let mut lines = String::new();
    for t in &ts {
        let crc = crc32_ref(&direct(dir, t));
        lines.push_str(&format!(
            "{}\t{}\t{}\t{crc:08x}\n",
            t.path, t.offset, t.bytes
        ));
    }
    let bytes = ts.iter().map(|t| t.bytes).sum();
    (
        ts.len() as u64,
        bytes,
        cere_compiler::sha256_hex(lines.as_bytes()),
    )
}

/// L7 (m216 design note, O100 "per-tensor digests identical"): the probe's per-tensor CRC-32 digests, combined
/// from pieces that arrive in any order, equal the files' bytes for buffered, direct and auto reads and any buffer
/// count (different piece boundaries); `cerec probe-load --tensor-digests on` reports the summary, the default
/// report has no such key.
#[test]
fn tensor_digests_equal_the_file_bytes_for_every_read_mode() {
    use cere_compiler::probe::TensorDigests;
    let t = Tmp::new("pdigests");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let (tensors, bytes, want) = digest_ref(&dir, &art);
    assert!(
        art.tensors().any(|t| t.bytes > 2 * (64 << 10)),
        "a multi-piece tensor"
    );
    for read in [ReadMode::Buffered, ReadMode::Direct, ReadMode::Auto] {
        for n in [1, 3, 8] {
            let mut bufs = artifact::buffers(n, artifact::MIN_BUFFER_BYTES).unwrap();
            let sink = TensorDigests::default();
            art.stage(Class::All, &mut bufs, &sink, Some(&read_opts(read, 0, 0)))
                .unwrap();
            let s = sink.finish().unwrap();
            assert_eq!(
                (s.tensors, s.bytes),
                (tensors, bytes),
                "{read:?}, {n} buffers"
            );
            assert_eq!(s.sha256, want, "{read:?}, {n} buffers");
        }
    }
    for read in ["buffered", "direct", "auto"] {
        let r = json(&ok(probe(
            &t,
            &dir,
            &sha,
            &[
                "--class",
                "all",
                "--cache",
                "warm",
                "--hash",
                "off",
                "--read",
                read,
                "--tensor-digests",
                "on",
            ],
        )));
        assert_eq!(r["tensor_digests"]["tensors"], tensors, "{read}");
        assert_eq!(r["tensor_digests"]["bytes"], bytes, "{read}");
        assert_eq!(r["tensor_digests"]["sha256"], want.as_str(), "{read}");
    }
    let plain = json(&ok(probe(
        &t,
        &dir,
        &sha,
        &["--class", "all", "--cache", "warm", "--hash", "off"],
    )));
    assert!(plain.get("tensor_digests").is_none());
}

/// A piece set that does not tile its tensor (a gap or an overlap) is an error, never a digest.
#[test]
fn tensor_digests_refuse_pieces_that_do_not_tile_the_tensor() {
    use cere_compiler::probe::TensorDigests;
    let t = Tmp::new("pdtile");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let big = art.tensors().max_by_key(|t| t.bytes).unwrap();
    let data = direct(&dir, &big);
    let half = data.len() / 2;
    let gap = TensorDigests::default();
    gap.accept(&big, 0, &data[..half], false).unwrap();
    assert!(gap.finish().is_err(), "a gap");
    let overlap = TensorDigests::default();
    overlap.accept(&big, 0, &data[..half + 1], false).unwrap();
    overlap
        .accept(&big, half as u64, &data[half..], true)
        .unwrap();
    assert!(overlap.finish().is_err(), "an overlap");
    let whole = TensorDigests::default();
    whole
        .accept(&big, half as u64, &data[half..], true)
        .unwrap();
    whole.accept(&big, 0, &data[..half], false).unwrap();
    let s = whole.finish().unwrap();
    let line = format!(
        "{}\t{}\t{}\t{:08x}\n",
        big.path,
        big.offset,
        big.bytes,
        crc32_ref(&data)
    );
    assert_eq!(s.sha256, cere_compiler::sha256_hex(line.as_bytes()));
}

/// Keeps every piece like [`Keep`] and, before returning from `accept`, waits `hold` and checks that the piece's
/// bytes did not change meanwhile (a buffer reused before this sink released it would be overwritten).
struct Slow {
    keep: Keep,
    hold: Duration,
    accepts: AtomicU64,
}

impl Sink for Slow {
    fn accept(&self, t: &TensorRef<'_>, off: u64, data: &[u8], last: bool) -> Result<(), String> {
        let copy = data.to_vec();
        std::thread::sleep(self.hold);
        if data != copy.as_slice() {
            return Err(format!(
                "{}:{} at {off}: the buffer was reused before this sink released it",
                t.path, t.name
            ));
        }
        self.accepts.fetch_add(1, Ordering::Relaxed);
        self.keep.accept(t, off, data, last)
    }
}

/// L7 (one producer, N consumers; M3.2N): one read feeds every sink, each sink gets exactly its wanted bytes, and a
/// buffer is read into again only after every sink has released it: a fast sink and a slow one that checks its
/// piece's bytes after a hold, on 2 buffers with 2 readers (the fast sink's release alone would free the buffer).
#[test]
fn every_sink_gets_its_bytes_and_a_buffer_is_reused_only_after_every_sink_released_it() {
    let t = Tmp::new("psinks");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let all: u64 = art.tensors().map(|t| t.bytes).sum();
    for read in [ReadMode::Buffered, ReadMode::Direct] {
        let fast = Keep::new(Some(Role::Draft));
        let slow = Slow {
            keep: Keep::new(None),
            hold: Duration::from_millis(2),
            accepts: AtomicU64::new(0),
        };
        let sinks: [&dyn Sink; 2] = [&fast, &slow];
        let mut bufs = artifact::buffers(2, artifact::MIN_BUFFER_BYTES).unwrap();
        let before = art.counters().payload_bytes_read;
        let r = art
            .stage_to(Class::All, &mut bufs, &sinks, Some(&read_opts(read, 2, 1)))
            .unwrap();
        assert_eq!(r.sinks, 2);
        // Each byte is read once for both sinks: the draft tensors only the slow sink wants are read too.
        assert_eq!(art.counters().payload_bytes_read - before, all, "{read:?}");
        assert!(slow.accepts.load(Ordering::Relaxed) > 2, "{read:?}");
        assert!(r.high_water_bytes <= 2 * artifact::MIN_BUFFER_BYTES as u64);
        let (fast, slow) = (fast.into_bytes(), slow.keep.into_bytes());
        for t in art.tensors().filter(|t| t.bytes > 0) {
            let key = format!("{}:{}", t.path, t.name);
            let file = direct(&dir, &t);
            assert_eq!(slow[&key], file, "{read:?} slow: {key}");
            if t.role == Role::Draft {
                assert!(!fast.contains_key(&key), "{read:?} fast got declined {key}");
            } else {
                assert_eq!(fast[&key], file, "{read:?} fast: {key}");
            }
        }
    }
    let mut bufs = artifact::buffers(1, artifact::MIN_BUFFER_BYTES).unwrap();
    let none: [&Keep; 0] = [];
    let e = art
        .stage_to(Class::All, &mut bufs, &none, None)
        .unwrap_err();
    assert_eq!(e.class, "buffer");
}

/// Holds the first piece's `accept` until a piece of another buffer (`buffers`: their address ranges) arrives on
/// another delivery thread (or 10 s pass: then the first `accept` fails).
struct Overlap {
    buffers: Vec<(usize, usize)>,
    first: Mutex<Option<usize>>,
    other: Condvar,
    seen_other: AtomicUsize,
}

impl Sink for Overlap {
    fn accept(&self, _: &TensorRef<'_>, _: u64, data: &[u8], _: bool) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let p = data.as_ptr() as usize;
        let Some(buffer) = self.buffers.iter().position(|&(lo, hi)| lo <= p && p < hi) else {
            return Err(format!("a piece at {p:#x} is in no staging buffer"));
        };
        let mut first = self.first.lock().unwrap();
        match *first {
            None => {
                *first = Some(buffer);
                let deadline = Instant::now() + Duration::from_secs(10);
                while self.seen_other.load(Ordering::Relaxed) == 0 {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(
                            "no other buffer was read while the first put was in flight".into()
                        );
                    }
                    first = self.other.wait_timeout(first, left).unwrap().0;
                }
            }
            Some(b) if b != buffer => {
                self.seen_other.fetch_add(1, Ordering::Relaxed);
                self.other.notify_all();
            }
            Some(_) => {}
        }
        Ok(())
    }
}

/// L7: reads overlap the puts: with ONE reader and two delivery threads, a second buffer is read and delivered
/// while the first piece's `accept` (a put, a DMA) is still in flight.
#[test]
fn one_reader_reads_the_next_buffer_while_a_put_is_in_flight() {
    let t = Tmp::new("poverlap");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let mut bufs = artifact::buffers(3, artifact::MIN_BUFFER_BYTES).unwrap();
    let sink = Overlap {
        buffers: bufs
            .iter_mut()
            .map(|b| {
                let s = b.as_mut_slice();
                (s.as_ptr() as usize, s.as_ptr() as usize + s.len())
            })
            .collect(),
        first: Mutex::new(None),
        other: Condvar::new(),
        seen_other: AtomicUsize::new(0),
    };
    let r = art
        .stage(
            Class::All,
            &mut bufs,
            &sink,
            Some(&read_opts(ReadMode::Buffered, 1, 2)),
        )
        .unwrap();
    assert_eq!(r.threads, 1);
    assert_eq!(r.consumers, 2);
    assert!(sink.seen_other.load(Ordering::Relaxed) >= 1);
    assert_eq!(r.tensors as usize, art.tensors().count());
}

/// L7 error paths: a weights file that shrinks behind the open artifact fails the pass with `file_size` (buffered
/// and direct reads, several readers and delivery threads: the other readers stop, nothing hangs).
#[test]
fn a_file_that_shrinks_while_staging_is_file_size_for_buffered_and_direct_reads() {
    let t = Tmp::new("pshrink");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let big = art.tensors().max_by_key(|t| t.bytes).unwrap();
    let path = dir.join(big.path);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(big.offset + big.bytes / 2)
        .unwrap();
    for read in [ReadMode::Buffered, ReadMode::Direct] {
        let mut bufs = artifact::buffers(4, artifact::MIN_BUFFER_BYTES).unwrap();
        let e = art
            .stage(
                Class::All,
                &mut bufs,
                &Keep::new(None),
                Some(&read_opts(read, 4, 2)),
            )
            .unwrap_err();
        assert_eq!(e.class, "file_size", "{read:?}: {e:?}");
        assert!(e.message.contains(big.path), "{read:?}: {}", e.message);
    }
}

/// L4 (m216 design note C3): a weights file is checked from views of its tensors without a read; a changed byte is
/// `file_hash`, a missing or short view is `resident`.
#[test]
fn file_views_are_checked_against_the_manifest_without_a_read() {
    let t = Tmp::new("pviews");
    let (dir, sha) = verified_copy(&t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    let path = art
        .tensors()
        .find(|t| t.placement == "host_pinned")
        .map(|t| t.path.to_string())
        .expect("the fixture has a host_pinned file");
    let mut owned: Vec<Vec<u8>> = art
        .tensors()
        .filter(|t| t.path == path)
        .map(|t| direct(&dir, &t))
        .collect();
    let before = art.counters().payload_bytes_read;
    art.check_file_views(&path, &views_of(&owned)).unwrap();
    assert_eq!(art.counters().payload_bytes_read, before, "nothing is read");
    let short = views_of(&owned);
    let e = art
        .check_file_views(&path, &short[..short.len() - 1])
        .unwrap_err();
    assert_eq!(e.class, "resident");
    let i = owned.iter().position(|v| !v.is_empty()).unwrap();
    owned[i][0] ^= 1;
    let e = art.check_file_views(&path, &views_of(&owned)).unwrap_err();
    assert_eq!(e.class, "file_hash");
    let e = art.check_file_views("config.json", &[]).unwrap_err();
    assert_eq!(e.class, "resident");
}

fn views_of(owned: &[Vec<u8>]) -> Vec<&[u8]> {
    owned.iter().map(Vec::as_slice).collect()
}
