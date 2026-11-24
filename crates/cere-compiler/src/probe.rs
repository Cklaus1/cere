//! `cerec probe-load` (m21 design note R5, Q8, Q9): a read-only staging run over [`Artifact::stage`] into
//! discarded buffers, timed per stage, with residency, labels, byte counters and the memory floor.
use crate::artifact::{
    self, Artifact, Cache, Class, Mode, ReadMode, Sink, StageOptions, TensorRef,
};
use crate::fsio;
use crate::prepare::Floor;
use crate::{Error, Limits};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Options {
    pub dir: PathBuf,
    pub expected_manifest_sha256: String,
    pub class: Class,
    pub cache: Cache,
    pub hash: bool,
    /// The staged-copy baseline: same files, ranges and buffers; no validation and no hashing.
    pub baseline: bool,
    pub threads: usize,
    pub buffer_bytes: usize,
    pub read: ReadMode,
    pub min_mem_available_gib: u64,
    /// `--tensor-digests on` (m216 design note L7): a per-tensor CRC-32 of the delivered bytes, reported as
    /// [`DigestSummary`] (`tensor_digests`), so read modes can be compared byte for byte without keeping the bytes.
    pub tensor_digests: bool,
}

/// Accepts every piece and keeps nothing: the probe measures reading into the staging buffers.
struct Discard;

impl Sink for Discard {
    fn accept(&self, _: &TensorRef<'_>, _: u64, _: &[u8], _: bool) -> Result<(), String> {
        Ok(())
    }
}

/// CRC-32 (IEEE, reflected polynomial 0xEDB88320) lookup table.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(!0u32, |c, &b| {
        CRC_TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8)
    })
}

fn gf2_times(mat: &[u32; 32], mut vec: u32) -> u32 {
    let mut sum = 0;
    let mut i = 0;
    while vec != 0 {
        if vec & 1 != 0 {
            sum ^= mat[i];
        }
        vec >>= 1;
        i += 1;
    }
    sum
}

fn gf2_square(mat: &[u32; 32]) -> [u32; 32] {
    std::array::from_fn(|n| gf2_times(mat, mat[n]))
}

/// The CRC-32 of A followed by B from crc(A), crc(B) and len(B) (zlib's `crc32_combine`).
fn crc32_combine(mut crc1: u32, crc2: u32, mut len2: u64) -> u32 {
    if len2 == 0 {
        return crc1;
    }
    // The operator for one zero bit, then squared to two and four bits; each loop step applies 2^k zero bytes.
    let mut odd = [0u32; 32];
    odd[0] = 0xEDB8_8320;
    for (n, v) in odd.iter_mut().enumerate().skip(1) {
        *v = 1 << (n - 1);
    }
    let mut even = gf2_square(&odd);
    odd = gf2_square(&even);
    loop {
        even = gf2_square(&odd);
        if len2 & 1 != 0 {
            crc1 = gf2_times(&even, crc1);
        }
        len2 >>= 1;
        if len2 == 0 {
            break;
        }
        odd = gf2_square(&even);
        if len2 & 1 != 0 {
            crc1 = gf2_times(&odd, crc1);
        }
        len2 >>= 1;
        if len2 == 0 {
            break;
        }
    }
    crc1 ^ crc2
}

/// The pieces of one tensor seen so far: (offset in tensor, length, CRC-32), any order.
struct Pieces {
    bytes: u64,
    pieces: Vec<(u64, u64, u32)>,
}

/// A sink that keeps a CRC-32 per delivered piece and, at [`TensorDigests::finish`], combines each tensor's pieces
/// in offset order into the tensor's CRC-32 (independent of where the pieces were cut). Empty pieces are ignored.
#[derive(Default)]
pub struct TensorDigests(Mutex<BTreeMap<(String, u64), Pieces>>);

/// The per-tensor digests of one pass: `tensors` non-empty tensors, `bytes` their bytes, and `sha256` over one
/// line `path\toffset\tbytes\tcrc32 (8 hex)\n` per tensor in (path, file offset) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestSummary {
    pub tensors: u64,
    pub bytes: u64,
    pub sha256: String,
}

impl Sink for TensorDigests {
    fn accept(&self, t: &TensorRef<'_>, off: u64, data: &[u8], _: bool) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let crc = crc32(data);
        let mut map = self
            .0
            .lock()
            .map_err(|_| "tensor digests poisoned".to_string())?;
        map.entry((t.path.to_string(), t.offset))
            .or_insert_with(|| Pieces {
                bytes: t.bytes,
                pieces: Vec::new(),
            })
            .pieces
            .push((off, data.len() as u64, crc));
        Ok(())
    }
}

impl TensorDigests {
    /// The summary; an error names the first tensor whose pieces do not tile it exactly once.
    pub fn finish(self) -> Result<DigestSummary, String> {
        let map = self
            .0
            .into_inner()
            .map_err(|_| "tensor digests poisoned".to_string())?;
        let mut lines = String::new();
        let mut bytes = 0u64;
        for ((path, offset), mut p) in map {
            p.pieces.sort_unstable_by_key(|x| x.0);
            let mut next = 0u64;
            let mut crc = 0u32;
            for &(off, len, c) in &p.pieces {
                if off != next {
                    return Err(format!("{path}@{offset}: piece at {off}, expected {next}"));
                }
                crc = if next == 0 {
                    c
                } else {
                    crc32_combine(crc, c, len)
                };
                next += len;
            }
            if next != p.bytes {
                return Err(format!(
                    "{path}@{offset}: pieces cover {next} of {} bytes",
                    p.bytes
                ));
            }
            bytes += p.bytes;
            lines.push_str(&format!("{path}\t{offset}\t{}\t{crc:08x}\n", p.bytes));
        }
        Ok(DigestSummary {
            tensors: lines.lines().count() as u64,
            bytes,
            sha256: crate::sha256_hex(lines.as_bytes()),
        })
    }
}

fn open(o: &Options, limits: &Limits) -> Result<Artifact, Error> {
    if o.baseline {
        artifact::open_baseline(&o.dir, &o.expected_manifest_sha256, limits)
    } else {
        artifact::open_with(&o.dir, &o.expected_manifest_sha256, Mode::Stamp, limits)
    }
}

/// Bytes over milliseconds as MB/s, an integer (canonical JSON reports carry no floats; 0 for an empty or instant
/// pass).
pub fn mb_per_s(bytes: u64, ms: u64) -> u64 {
    bytes.checked_div(ms).map_or(0, |b_per_ms| b_per_ms / 1000)
}

/// Runs `cerec probe-load`; the report on success. A non-baseline run needs a `full_verify` stamp (stamp mode).
pub fn run(o: &Options) -> Result<Value, Error> {
    if o.baseline && o.hash {
        return Err(Error::reject(
            "usage",
            "--baseline does no hashing: use --hash off",
        ));
    }
    let limits = Limits::default();
    let floor = Floor::new(
        o.min_mem_available_gib,
        (o.threads as u64).saturating_mul(o.buffer_bytes as u64),
    );
    let mem_before = floor.check()?;
    let evicted = match o.cache {
        Cache::Cold => artifact::evict_dir(&o.dir, Some(&o.expected_manifest_sha256), &limits)?.0,
        Cache::Warm => 0,
    };
    let t = Instant::now();
    let art = open(o, &limits)?;
    // Residency of the class's tensor pages, after DISCOVER/VALIDATE and before READ.
    let before = art.residency(Some(o.class))?;
    let mut buffers = artifact::buffers(o.threads, o.buffer_bytes)?;
    let digests = TensorDigests::default();
    let sink: &dyn Sink = if o.tensor_digests { &digests } else { &Discard };
    let report = art.stage(
        o.class,
        &mut buffers,
        sink,
        Some(&StageOptions {
            floor: Some(&floor),
            hash: o.hash,
            read: o.read,
            ..StageOptions::default()
        }),
    )?;
    let total_ms = t.elapsed().as_millis() as u64;
    let after = art.residency(Some(o.class))?;
    let mem_after = fsio::mem_available().map_err(|e| Error::io("reading MemAvailable", e))?;
    let peak_rss = fsio::peak_rss().map_err(|e| Error::io("reading VmHWM", e))?;
    let timings = art.timings();
    let mut stages = json!({
        "discover": timings.discover_ms,
        "read": report.read_ms,
        "total": total_ms,
    });
    if let Some(v) = timings.validate_ms {
        stages["validate"] = json!(v);
    }
    if o.hash {
        stages["hash"] = json!(report.hash_ms);
    }
    let counters = art.counters();
    let mut out = json!({
        "outcome": "probed",
        "manifest_sha256": art.manifest_sha256,
        "class": o.class.name(),
        "cache": o.cache.name(),
        "hash": o.hash,
        "baseline": o.baseline,
        "label": artifact::cache_label(o.cache, before),
        "residency_before": before,
        "residency_after": after,
        "evicted_files": evicted,
        "tensors": report.tensors,
        "hashed_files": report.hashed_files,
        "payload_bytes_read": counters.payload_bytes_read,
        "metadata_bytes_read": counters.metadata_bytes_read,
        "staging_high_water_bytes": report.high_water_bytes,
        "read_mode": o.read,
        "reads": report.reads,
        "direct_files": report.direct_files,
        "read_mb_per_s": mb_per_s(report.bytes, report.read_ms),
        "pread_ms": report.pread_ms,
        "wait_free_ms": report.wait_free_ms,
        "wait_fill_ms": report.wait_fill_ms,
        "reads_in_flight_mean_x100": report.reads_in_flight_mean_x100,
        "reads_in_flight_max": report.reads_in_flight_max,
        "timeline": report.timeline,
        "buffer_bytes": o.buffer_bytes,
        "threads": o.threads,
        "peak_rss_bytes": peak_rss,
        "mem_available_before": mem_before,
        "mem_available_after": mem_after,
        "memory_floor": floor.bytes,
        "stages_ms": stages,
    });
    if o.tensor_digests {
        let s = digests
            .finish()
            .map_err(|e| Error::reject("tensor_digests", e))?;
        out["tensor_digests"] = json!({"tensors": s.tensors, "bytes": s.bytes, "sha256": s.sha256});
    }
    Ok(out)
}
