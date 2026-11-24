//! The v2 manifest (m21 design note Q4, Q17): a pure function of content, rules, profile and transforms, encoded
//! as canonical JSON. `cerec prepare` is its only producer; verifiers parse it, require canonical bytes, check its
//! structure (`manifest_schema`) and its semantic rules (`unsafe_path`, `too_many_files`, `manifest_rules`).
use crate::families::{Role, Rules, Tally, COMPONENTS, PLACEMENTS};
use crate::safetensors::Dtype;
use crate::source::{payload_role, role};
use crate::{canon_json, fsio, sha256sum_lines_digest, token_map, Error, Limits};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: &str = "2";
pub const TENSOR_LAYOUT_ABI: u64 = 1;
pub const QUALITY_MODE: &str = "layout_only";
pub const MANIFEST_FILE: &str = "manifest.json";
pub const DIGEST_FILE: &str = "manifest.sha256";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRef {
    pub revision: String,
    pub tree_sha256: String,
}

/// `{lfs_sha256}` for LFS files, `{size, blob_id, local_modification}` otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Upstream {
    Lfs(LfsUpstream),
    Git(GitUpstream),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LfsUpstream {
    pub lfs_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitUpstream {
    pub size: u64,
    pub blob_id: String,
    pub local_modification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardInfo {
    pub header_bytes: u64,
    pub header_sha256: String,
    pub tensors: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream: Option<Upstream>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safetensors: Option<ShardInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub name: String,
    pub output: String,
    pub source_sha256: String,
    pub source_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyRow {
    pub family: String,
    pub component: String,
    pub role: String,
    pub dtype: String,
    pub placement: String,
    pub count: u64,
    pub bytes: u64,
    pub misaligned: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountBytes {
    pub count: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Totals {
    pub by_dtype: BTreeMap<String, CountBytes>,
    pub by_placement: BTreeMap<String, CountBytes>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema_version: String,
    pub profile_id: String,
    pub device_model: String,
    pub device_memory_mib: u64,
    pub compute_capability: String,
    pub compute_capability_source: String,
    pub modalities: Vec<String>,
    pub speculative: String,
    pub placement: BTreeMap<String, String>,
}

impl Profile {
    /// Structural checks of a capability profile (`schemas/artifact_profile.schema.json`).
    pub fn from_value(v: &Value) -> Result<Profile, String> {
        let p: Profile = serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
        let cc_ok = p.compute_capability.split_once('.').is_some_and(|(a, b)| {
            [a, b]
                .iter()
                .all(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
        });
        if p.schema_version != "1"
            || p.profile_id.is_empty()
            || p.device_model.is_empty()
            || !cc_ok
            || !["inferred", "queried"].contains(&p.compute_capability_source.as_str())
            || p.modalities.is_empty()
            || !["nextn", "none"].contains(&p.speculative.as_str())
        {
            return Err("profile fields out of range".into());
        }
        for (r, place) in &p.placement {
            if Role::parse(r).is_none() || !PLACEMENTS.contains(&place.as_str()) {
                return Err(format!("profile placement {r}: {place}"));
            }
        }
        Ok(p)
    }

    pub fn placement_of(&self, role: Role) -> Option<&str> {
        self.placement.get(role.name()).map(String::as_str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: String,
    pub tensor_layout_abi: u64,
    pub quality_mode: String,
    pub source: SourceRef,
    pub profile_id: String,
    pub profile: Value,
    pub family_rules: Value,
    pub model_hash: String,
    pub tokenizer_hash: String,
    pub template_hash: String,
    pub transforms: Vec<Transform>,
    pub files: Vec<FileEntry>,
    pub families: Vec<FamilyRow>,
    pub totals: Totals,
}

/// The lock-definition hashes (docs/CONFIGURATION.md:196-205) over `files`: model content (weights, index, config,
/// quant config), tokenizer, and the template file's own sha256.
pub fn identity_hashes(files: &[FileEntry]) -> (String, String, String) {
    let mut sorted: Vec<&FileEntry> = files.iter().collect();
    sorted.sort_by(|a, b| a.path.cmp(&b.path));
    let digest = |roles: &[&str]| {
        sha256sum_lines_digest(
            sorted
                .iter()
                .filter(|f| roles.contains(&f.role.as_str()))
                .map(|f| (f.sha256.as_str(), f.path.as_str())),
        )
    };
    let model = digest(&[
        role::WEIGHTS,
        role::WEIGHT_INDEX,
        role::CONFIG,
        role::QUANT_CONFIG,
    ]);
    let tokenizer = digest(&[role::TOKENIZER]);
    let template = sorted
        .iter()
        .find(|f| f.role == role::TEMPLATE)
        .map_or_else(String::new, |f| f.sha256.clone());
    (model, tokenizer, template)
}

/// Family rows (placement from the profile) and totals of classified tallies.
pub fn families_and_totals(
    rules: &Rules,
    profile: &Profile,
    tallies: &[Tally],
) -> Result<(Vec<FamilyRow>, Totals), Error> {
    let mut rows = Vec::with_capacity(rules.rules.len());
    let mut by_dtype: BTreeMap<String, CountBytes> = BTreeMap::new();
    let mut by_placement: BTreeMap<String, CountBytes> = BTreeMap::new();
    for (r, t) in rules.rules.iter().zip(tallies) {
        let placement = profile.placement_of(r.role).ok_or_else(|| {
            Error::reject(
                "profile",
                format!("profile has no placement for role {}", r.role.name()),
            )
        })?;
        rows.push(FamilyRow {
            family: r.family.clone(),
            component: r.component.clone(),
            role: r.role.name().to_string(),
            dtype: r.dtype.name().to_string(),
            placement: placement.to_string(),
            count: t.count,
            bytes: t.bytes,
            misaligned: t.misaligned,
        });
        if t.count > 0 {
            for acc in [
                by_dtype.entry(r.dtype.name().to_string()).or_default(),
                by_placement.entry(placement.to_string()).or_default(),
            ] {
                acc.count += t.count;
                acc.bytes += t.bytes;
            }
        }
    }
    Ok((
        rows,
        Totals {
            by_dtype,
            by_placement,
        },
    ))
}

impl Manifest {
    /// Canonical bytes (Q17).
    pub fn to_bytes(&self) -> Vec<u8> {
        let v = serde_json::to_value(self).expect("manifest serializes");
        canon_json::to_vec(&v).expect("manifest holds integers only")
    }

    pub fn file(&self, path: &str) -> Option<&FileEntry> {
        self.files.iter().find(|f| f.path == path)
    }
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn schema_err(msg: impl Into<String>) -> Error {
    Error::reject("manifest_schema", msg)
}

fn rules_err(msg: impl Into<String>) -> Error {
    Error::reject("manifest_rules", msg)
}

/// Verification steps 3-5: strict parse (duplicate keys), canonical bytes, structure. Returns the manifest and its
/// parsed rules and profile.
pub fn parse(bytes: &[u8]) -> Result<(Manifest, Rules, Profile), Error> {
    let value = canon_json::parse_strict(bytes).map_err(|e| match e {
        canon_json::JsonError::DuplicateKey(_) => Error::reject("duplicate_key", e.to_string()),
        _ => schema_err(e.to_string()),
    })?;
    // A value outside the canonical set (a float, `-0`, `1e3`) has no canonical encoding: non-canonical bytes, as
    // verify_artifact.py classifies it.
    match canon_json::to_vec(&value) {
        Ok(canon) if canon == bytes => {}
        Ok(_) => {
            return Err(Error::reject(
                "manifest_noncanonical",
                "manifest bytes are not canonical JSON",
            ))
        }
        Err(e) => return Err(Error::reject("manifest_noncanonical", e.to_string())),
    }
    check_raw_structure(&value)?;
    let m: Manifest = serde_json::from_value(value).map_err(|e| schema_err(e.to_string()))?;
    if m.schema_version != SCHEMA_VERSION
        || m.tensor_layout_abi != TENSOR_LAYOUT_ABI
        || m.quality_mode != QUALITY_MODE
    {
        return Err(schema_err(
            "schema_version, tensor_layout_abi or quality_mode",
        ));
    }
    let hexes = [
        (&m.source.revision, 40),
        (&m.source.tree_sha256, 64),
        (&m.model_hash, 64),
        (&m.tokenizer_hash, 64),
        (&m.template_hash, 64),
    ];
    if hexes.iter().any(|(s, n)| !is_hex(s, *n)) {
        return Err(schema_err("hash fields must be lowercase hex"));
    }
    for f in &m.files {
        let upstream_ok = match &f.upstream {
            None => true,
            Some(Upstream::Lfs(u)) => is_hex(&u.lfs_sha256, 64),
            Some(Upstream::Git(u)) => is_hex(&u.blob_id, 40),
        };
        let shard_ok = f
            .safetensors
            .as_ref()
            .is_none_or(|s| is_hex(&s.header_sha256, 64) && s.header_bytes >= 10);
        if f.path.is_empty()
            || !is_hex(&f.sha256, 64)
            || !upstream_ok
            || !shard_ok
            || !ROLES.contains(&f.role.as_str())
        {
            return Err(schema_err(format!("file entry {}", f.path)));
        }
    }
    for t in &m.transforms {
        if t.name != token_map::TRANSFORM || !is_hex(&t.source_sha256, 64) || t.output.is_empty() {
            return Err(schema_err("transform"));
        }
    }
    for row in &m.families {
        if Role::parse(&row.role).is_none()
            || Dtype::parse(&row.dtype).is_none()
            || !COMPONENTS.contains(&row.component.as_str())
            || !PLACEMENTS.contains(&row.placement.as_str())
        {
            return Err(schema_err(format!("family {}", row.family)));
        }
    }
    if m.totals.by_dtype.keys().any(|d| Dtype::parse(d).is_none())
        || m.totals
            .by_placement
            .keys()
            .any(|p| !PLACEMENTS.contains(&p.as_str()))
    {
        return Err(schema_err("totals keys"));
    }
    let rules = Rules::from_value(m.family_rules.clone()).map_err(|e| schema_err(e.to_string()))?;
    let profile =
        Profile::from_value(&m.profile).map_err(|e| schema_err(format!("profile: {e}")))?;
    Ok((m, rules, profile))
}

pub const ROLES: [&str; 9] = [
    role::WEIGHTS,
    role::WEIGHT_INDEX,
    role::CONFIG,
    role::QUANT_CONFIG,
    role::GENERATION_CONFIG,
    role::TOKENIZER,
    role::TEMPLATE,
    role::PREPROCESSOR,
    role::DRAFT_TOKEN_MAP,
];

/// Structure the typed parse cannot see: explicit `null`s for optional members.
fn check_raw_structure(v: &Value) -> Result<(), Error> {
    let files = v
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| schema_err("files must be a list"))?;
    for f in files {
        for key in ["upstream", "safetensors"] {
            if f.get(key).is_some_and(Value::is_null) {
                return Err(schema_err(format!("{key} may not be null")));
            }
        }
    }
    Ok(())
}

/// The role a manifest path must carry.
pub fn role_for_path(path: &str) -> Option<&'static str> {
    if path == token_map::OUTPUT_PATH {
        return Some(role::DRAFT_TOKEN_MAP);
    }
    if path.contains('/') {
        return None;
    }
    payload_role(path)
}

/// Verification step 6: path safety, file count, then every semantic rule.
pub fn check_rules(
    m: &Manifest,
    rules: &Rules,
    profile: &Profile,
    limits: &Limits,
) -> Result<(), Error> {
    for f in &m.files {
        let safe = fsio::components(&f.path).is_some_and(|parts| {
            !parts.iter().any(|p| p.starts_with(".tmp-"))
                && f.path != MANIFEST_FILE
                && f.path != DIGEST_FILE
        });
        if !safe {
            return Err(Error::reject(
                "unsafe_path",
                format!("unsafe path {:?}", f.path),
            ));
        }
    }
    if m.files.len() as u64 > limits.max_files {
        return Err(Error::reject(
            "too_many_files",
            format!("{} files", m.files.len()),
        ));
    }
    if m.files.windows(2).any(|w| w[0].path >= w[1].path) {
        return Err(rules_err("files must be strictly sorted by path"));
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for f in &m.files {
        if role_for_path(&f.path) != Some(f.role.as_str()) {
            return Err(rules_err(format!("{} cannot have role {}", f.path, f.role)));
        }
        if f.safetensors.is_some() != (f.role == role::WEIGHTS) {
            return Err(rules_err(format!(
                "{}: safetensors info iff weights",
                f.path
            )));
        }
        *counts.entry(f.role.as_str()).or_default() += 1;
    }
    let n = |r: &str| counts.get(r).copied().unwrap_or(0);
    let cardinality_ok = n(role::WEIGHTS) >= 1
        && n(role::WEIGHT_INDEX) == 1
        && n(role::CONFIG) == 1
        && n(role::TOKENIZER) >= 1
        && n(role::TEMPLATE) == 1
        && n(role::QUANT_CONFIG) <= 1
        && n(role::GENERATION_CONFIG) <= 1
        && n(role::DRAFT_TOKEN_MAP) <= 1;
    if !cardinality_ok {
        return Err(rules_err("role cardinalities"));
    }
    if (
        m.model_hash.clone(),
        m.tokenizer_hash.clone(),
        m.template_hash.clone(),
    ) != identity_hashes(&m.files)
    {
        return Err(rules_err(
            "model/tokenizer/template hash disagrees with files",
        ));
    }
    if m.profile_id != profile.profile_id {
        return Err(rules_err("profile_id differs from the embedded profile"));
    }
    let map = m.files.iter().find(|f| f.role == role::DRAFT_TOKEN_MAP);
    let transform_ok = match (map, m.transforms.as_slice()) {
        (None, []) => true,
        (Some(f), [t]) => t.output == f.path && f.bytes % 4 == 0 && f.bytes > 0,
        _ => false,
    };
    if !transform_ok {
        return Err(rules_err("draft token map and transforms disagree"));
    }
    if m.families.len() != rules.rules.len() {
        return Err(rules_err("families differ from the embedded rules"));
    }
    let tallies: Vec<Tally> = m
        .families
        .iter()
        .map(|f| Tally {
            count: f.count,
            bytes: f.bytes,
            misaligned: f.misaligned,
        })
        .collect();
    let (rows, totals) =
        families_and_totals(rules, profile, &tallies).map_err(|e| rules_err(e.message))?;
    if rows != m.families {
        return Err(rules_err(
            "families differ from the embedded rules and profile",
        ));
    }
    if totals != m.totals {
        return Err(rules_err("totals are not the sums of the families"));
    }
    Ok(())
}
