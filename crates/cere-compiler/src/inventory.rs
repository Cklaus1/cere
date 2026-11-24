//! `cerec inventory` (m21 design note Q12): committed per-shard header metadata and family tables of a source
//! model, in canonical JSON (Q17). Byte-identical to `scripts/model_inventory.py`. Reads headers only.
use crate::families::{classify, Rules, Tally};
use crate::safetensors::{self, Header};
use crate::source::{self, Select, Tree};
use crate::{canon_json, sha256_hex, sha256sum_lines_digest, Error, Limits};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// A source shard with its parsed header.
#[derive(Debug)]
pub struct Shard {
    pub name: String,
    pub bytes: u64,
    pub header: Header,
}

/// Parses a family rules file (strict JSON, structural checks).
pub fn load_rules(bytes: &[u8]) -> Result<Rules, Error> {
    let value = canon_json::parse_strict(bytes)
        .map_err(|e| Error::reject("family_rules", e.to_string()))?;
    Rules::from_value(value).map_err(|e| Error::reject("family_rules", e.to_string()))
}

/// Reads and parses every shard header of `files` (already opened), enforcing the file and tensor caps.
pub fn read_headers(
    files: &[&source::SourceFile],
    limits: &Limits,
) -> Result<(Vec<Shard>, u64), Error> {
    if files.len() as u64 > limits.max_files {
        return Err(Error::reject(
            "too_many_files",
            format!("{} shards", files.len()),
        ));
    }
    let mut out = Vec::with_capacity(files.len());
    let mut total: u64 = 0;
    let mut read: u64 = 0;
    for f in files {
        let (header, n) = safetensors::read(&f.open, f.open.stat.size, limits)?;
        source::check_unchanged(&f.open, &f.name)?;
        read += n;
        total += header.tensors.len() as u64;
        if total > limits.max_total_tensors {
            return Err(Error::reject(
                "too_many_tensors",
                format!("more than {} tensors", limits.max_total_tensors),
            ));
        }
        out.push(Shard {
            name: f.name.clone(),
            bytes: f.open.stat.size,
            header,
        });
    }
    Ok((out, read))
}

/// The Q12 header digest over shards in bytewise name order.
pub fn header_digest(shards: &[Shard]) -> String {
    let mut sorted: Vec<&Shard> = shards.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    sha256sum_lines_digest(
        sorted
            .iter()
            .map(|s| (s.header.header_sha256.as_str(), s.name.as_str())),
    )
}

fn tally_json(t: &Tally) -> Value {
    json!({"count": t.count, "bytes": t.bytes, "misaligned": t.misaligned})
}

/// Per-rule family rows with the given placement per rule.
pub fn family_rows(
    rules: &Rules,
    tallies: &[Tally],
    placement: impl Fn(usize) -> String,
) -> Vec<Value> {
    rules
        .rules
        .iter()
        .zip(tallies)
        .enumerate()
        .map(|(i, (r, t))| {
            json!({
                "family": r.family, "component": r.component, "role": r.role.name(), "dtype": r.dtype.name(),
                "placement": placement(i), "count": t.count, "bytes": t.bytes, "misaligned": t.misaligned,
            })
        })
        .collect()
}

/// Sums of family tallies by dtype (with misaligned) and by placement (count and bytes), only non-empty keys.
pub fn by_dtype_and_placement(
    rules: &Rules,
    tallies: &[Tally],
    placement: impl Fn(usize) -> String,
) -> (Value, Value) {
    let mut by_dtype: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut by_placement: BTreeMap<String, Tally> = BTreeMap::new();
    for (i, (r, t)) in rules.rules.iter().zip(tallies).enumerate() {
        if t.count == 0 {
            continue;
        }
        for acc in [
            by_dtype.entry(r.dtype.name()).or_default(),
            by_placement.entry(placement(i)).or_default(),
        ] {
            acc.count += t.count;
            acc.bytes += t.bytes;
            acc.misaligned += t.misaligned;
        }
    }
    let d: Map<String, Value> = by_dtype
        .iter()
        .map(|(k, t)| (k.to_string(), tally_json(t)))
        .collect();
    let p: Map<String, Value> = by_placement
        .into_iter()
        .map(|(k, t)| (k, json!({"count": t.count, "bytes": t.bytes})))
        .collect();
    (Value::Object(d), Value::Object(p))
}

/// Builds the inventory value of `src`.
pub fn build(src: &Path, families_bytes: &[u8], limits: &Limits) -> Result<Value, Error> {
    let rules = load_rules(families_bytes)?;
    let source = source::open(src, Select::Shards)?;
    let files: Vec<&source::SourceFile> = source.files.iter().collect();
    let (shards, _) = read_headers(&files, limits)?;
    let headers: Vec<&Header> = shards.iter().map(|s| &s.header).collect();
    let classified = classify(&rules, &headers)?;
    let placement = |i: usize| rules.rules[i].placement.clone();
    let (by_dtype, by_placement) = by_dtype_and_placement(&rules, &classified.tallies, placement);
    let tree: Option<&Tree> = source.tree.as_ref();
    let (mut lfs_files, mut lfs_size_match, mut non_lfs_files, mut not_in_tree) =
        (0u64, 0u64, 0u64, 0u64);
    let mut file_rows = Vec::with_capacity(shards.len());
    for s in &shards {
        let upstream = match tree.and_then(|t| t.files.get(&s.name)) {
            None => {
                not_in_tree += 1;
                Value::Null
            }
            Some(e) => match &e.lfs_sha256 {
                Some(lfs) => {
                    lfs_files += 1;
                    lfs_size_match += u64::from(e.size == s.bytes);
                    json!({"lfs_sha256": lfs, "size": e.size})
                }
                None => {
                    non_lfs_files += 1;
                    json!({"blob_id": e.blob_id, "size": e.size})
                }
            },
        };
        file_rows.push(json!({
            "path": s.name, "bytes": s.bytes, "header_bytes": s.header.header_bytes,
            "header_sha256": s.header.header_sha256, "tensors": s.header.tensors.len() as u64, "upstream": upstream,
        }));
    }
    let sum = |f: &dyn Fn(&Shard) -> u64| shards.iter().map(f).sum::<u64>();
    let tensors: u64 = sum(&|s| s.header.tensors.len() as u64);
    let misaligned: u64 = sum(&|s| {
        s.header
            .tensors
            .iter()
            .filter(|t| !t.element_aligned)
            .count() as u64
    });
    Ok(json!({
        "schema_version": "1",
        "family_rules": {"rules_id": rules.rules_id, "sha256": sha256_hex(families_bytes)},
        "source": tree.map_or(Value::Null, |t| json!({"revision": t.revision, "tree_sha256": t.sha256})),
        "header_digest": header_digest(&shards),
        "files": file_rows,
        "families": family_rows(&rules, &classified.tallies, placement),
        "by_dtype": by_dtype,
        "by_placement": by_placement,
        "totals": {
            "files": shards.len() as u64, "tensors": tensors, "file_bytes": sum(&|s| s.bytes),
            "header_bytes": sum(&|s| s.header.header_bytes), "data_bytes": sum(&|s| s.header.data_bytes()),
            "misaligned": misaligned,
        },
        "upstream": {
            "lfs_files": lfs_files, "lfs_size_match": lfs_size_match, "non_lfs_files": non_lfs_files,
            "not_in_tree": not_in_tree,
        },
    }))
}

/// The inventory of `src` as canonical JSON bytes.
pub fn run(src: &Path, families_bytes: &[u8], limits: &Limits) -> Result<Vec<u8>, Error> {
    let v = build(src, families_bytes, limits)?;
    canon_json::to_vec(&v).map_err(|e| Error::io("encoding inventory", e))
}
