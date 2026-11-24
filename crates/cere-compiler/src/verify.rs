//! `cerec verify` and `cerec evict` (m21 design note R4, R5, Q7, Q8, Q9): JSON reports over [`artifact`].
use crate::artifact::{self, Cache, Mode, HASH_CHUNK_BYTES, HASH_THREADS};
use crate::manifest::FamilyRow;
use crate::prepare::Floor;
use crate::source::role;
use crate::{fsio, stamp, Error, Limits};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Options {
    pub dir: PathBuf,
    pub expected_manifest_sha256: String,
    /// [`Mode::Full`] or [`Mode::Stamp`].
    pub mode: Mode,
    /// `--cache cold`: evict the artifact's files and record their residency before the run.
    pub cold: bool,
    pub limits: Limits,
    /// The Q9 floor `MemAvailable ≥ G GiB + staging` (staging: the hash buffers), checked at start and, in full
    /// mode, after every chunk of payload read (exit 5 `memory_floor`).
    pub min_mem_available_gib: u64,
}

/// `by_dtype` of the report, as `verify_artifact.py` prints it: the manifest totals plus misaligned counts.
fn by_dtype(families: &[FamilyRow]) -> Value {
    let mut out: BTreeMap<&str, (u64, u64, u64)> = BTreeMap::new();
    for f in families.iter().filter(|f| f.count > 0) {
        let e = out.entry(f.dtype.as_str()).or_default();
        e.0 += f.count;
        e.1 += f.bytes;
        e.2 += f.misaligned;
    }
    out.into_iter()
        .map(|(d, (count, bytes, misaligned))| {
            (
                d.to_string(),
                json!({"count": count, "bytes": bytes, "misaligned": misaligned}),
            )
        })
        .collect()
}

/// Runs `cerec verify`; the report on success. Full mode rewrites the stamp as `full_verify`.
pub fn run(o: &Options) -> Result<Value, Error> {
    let floor = Floor::new(
        o.min_mem_available_gib,
        (HASH_THREADS * HASH_CHUNK_BYTES) as u64,
    );
    let mem_before = floor.check()?;
    let cache = if o.cold {
        let (evicted, before) =
            artifact::evict_dir(&o.dir, Some(&o.expected_manifest_sha256), &o.limits)?;
        Some(json!({
            "label": artifact::cache_label(Cache::Cold, before),
            "residency_before": before,
            "evicted_files": evicted,
        }))
    } else {
        None
    };
    // `total` times the verification itself, after the cold pre-pass.
    let t = Instant::now();
    // Full mode is the headers pass plus the hash pass, which here runs under the floor.
    let open_mode = if o.mode == Mode::Full {
        Mode::Headers
    } else {
        o.mode
    };
    let mut art = artifact::open_with(&o.dir, &o.expected_manifest_sha256, open_mode, &o.limits)?;
    if o.mode == Mode::Full {
        art.hash_files(Some(&floor))?;
    }
    let stamp_written = match o.mode {
        Mode::Full => {
            art.write_full_verify_stamp()?;
            Some(stamp::FULL_VERIFY)
        }
        _ => None,
    };
    // Never silent: what the stamp could not anchor on a filesystem UUID goes to stderr (the report stays JSON).
    if let Some(note) = art.stamp_note() {
        eprintln!("note: {note}");
    }
    let timings = art.timings();
    let mut stages = json!({
        "discover": timings.discover_ms,
        "validate": timings.validate_ms.unwrap_or(0),
        "total": t.elapsed().as_millis() as u64,
    });
    if let Some(h) = timings.hash_ms {
        stages["hash"] = json!(h);
    }
    let m = &art.manifest;
    let counters = art.counters();
    let mem_after = fsio::mem_available().map_err(|e| Error::io("reading MemAvailable", e))?;
    let mut out = json!({
        "outcome": "verified",
        "mode": o.mode.name(),
        "manifest_sha256": art.manifest_sha256,
        "profile_id": m.profile_id,
        "files": m.files.len(),
        "shards": m.files.iter().filter(|f| f.role == role::WEIGHTS).count(),
        "tensors": art.tensors().count(),
        "families": m.families,
        "by_dtype": by_dtype(&m.families),
        "by_placement": m.totals.by_placement,
        "misaligned": m.families.iter().map(|f| f.misaligned).sum::<u64>(),
        "payload_bytes_read": counters.payload_bytes_read,
        "metadata_bytes_read": counters.metadata_bytes_read,
        "stamp_written": stamp_written,
        "stages_ms": stages,
        "mem_available_before": mem_before,
        "mem_available_after": mem_after,
        "memory_floor": floor.bytes,
    });
    if let Some(c) = cache {
        out["cache"] = c;
    }
    Ok(out)
}

/// Runs `cerec evict DIR`: every artifact file evicted; the count and the residency afterwards.
pub fn evict(dir: &std::path::Path, limits: &Limits) -> Result<Value, Error> {
    let (files, after) = artifact::evict_dir(dir, None, limits)?;
    Ok(json!({"files": files, "residency_after": after}))
}
