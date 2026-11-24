//! Tensor family rules (m21 design note Q5): every tensor fullmatches exactly one rule, then the expert-shard
//! metadata (Q5e) and the group checks (Q5a-c) run. Error variants are the family classes of
//! `tests/fixtures/artifact_invalid.json`; the check order is the shared verification order (steps 10-12).
pub mod pattern;

use crate::safetensors::{Dtype, Header, TensorInfo, MAX_DIMS};
use pattern::Pattern;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Tensor roles (closed list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    Nvfp4Weight,
    Nvfp4BlockScale,
    Nvfp4GlobalScale,
    Nvfp4InputScale,
    Dense,
    PleModule,
    PleTable,
    PleScale,
    PleIndex,
    Draft,
    Vision,
}

impl Role {
    pub const ALL: [Role; 11] = [
        Role::Nvfp4Weight,
        Role::Nvfp4BlockScale,
        Role::Nvfp4GlobalScale,
        Role::Nvfp4InputScale,
        Role::Dense,
        Role::PleModule,
        Role::PleTable,
        Role::PleScale,
        Role::PleIndex,
        Role::Draft,
        Role::Vision,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Role::Nvfp4Weight => "nvfp4_weight",
            Role::Nvfp4BlockScale => "nvfp4_block_scale",
            Role::Nvfp4GlobalScale => "nvfp4_global_scale",
            Role::Nvfp4InputScale => "nvfp4_input_scale",
            Role::Dense => "dense",
            Role::PleModule => "ple_module",
            Role::PleTable => "ple_table",
            Role::PleScale => "ple_scale",
            Role::PleIndex => "ple_index",
            Role::Draft => "draft",
            Role::Vision => "vision",
        }
    }

    pub fn parse(name: &str) -> Option<Role> {
        Role::ALL.into_iter().find(|r| r.name() == name)
    }

    /// The name suffix of an NVFP4 quad member.
    fn nvfp4_suffix(self) -> Option<&'static str> {
        match self {
            Role::Nvfp4Weight => Some(".weight"),
            Role::Nvfp4BlockScale => Some(".weight_scale"),
            Role::Nvfp4GlobalScale => Some(".weight_scale_2"),
            Role::Nvfp4InputScale => Some(".input_scale"),
            _ => None,
        }
    }
}

pub const COMPONENTS: [&str; 3] = ["target", "draft", "vision"];
pub const PLACEMENTS: [&str; 3] = ["device", "host_pinned", "unused"];

/// `any` (rank 0..=16) or an exact rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeRule {
    Any,
    Rank(usize),
}

impl ShapeRule {
    fn parse(s: &str) -> Option<ShapeRule> {
        if s == "any" {
            return Some(ShapeRule::Any);
        }
        let digits = s.strip_prefix("rank:")?;
        if digits.is_empty()
            || !digits.bytes().all(|b| b.is_ascii_digit())
            || (digits.len() > 1 && digits.starts_with('0'))
        {
            return None;
        }
        digits
            .parse()
            .ok()
            .filter(|&r| r <= MAX_DIMS)
            .map(ShapeRule::Rank)
    }

    fn admits(self, rank: usize) -> bool {
        match self {
            ShapeRule::Any => true,
            ShapeRule::Rank(r) => r == rank,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub family: String,
    pub pattern: Pattern,
    pub component: String,
    pub role: Role,
    pub dtype: Dtype,
    pub shape_rule: ShapeRule,
    /// The inventory placement; a manifest uses the profile's placement for `role` instead.
    pub placement: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Groups {
    pub nvfp4_group_size: u64,
    pub ple_table_shards: u64,
    pub ple_module_members: Vec<String>,
    pub ple_index_members: Vec<String>,
}

/// A parsed family rules file.
#[derive(Debug, Clone)]
pub struct Rules {
    pub rules_id: String,
    pub groups: Groups,
    pub rules: Vec<Rule>,
    /// The file as parsed, embedded verbatim in the manifest.
    pub value: Value,
}

/// Why a rules file is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RulesError(pub String);

impl fmt::Display for RulesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "family rules: {}", self.0)
    }
}

impl std::error::Error for RulesError {}

fn bad(what: impl Into<String>) -> RulesError {
    RulesError(what.into())
}

fn object<'a>(
    v: &'a Value,
    what: &str,
    keys: &[&str],
) -> Result<&'a Map<String, Value>, RulesError> {
    let obj = v
        .as_object()
        .ok_or_else(|| bad(format!("{what} is not an object")))?;
    let mut expected: Vec<&str> = keys.to_vec();
    expected.sort_unstable();
    let actual: Vec<&str> = obj.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(bad(format!("{what} keys must be exactly {keys:?}")));
    }
    Ok(obj)
}

fn string<'a>(obj: &'a Map<String, Value>, key: &str) -> Result<&'a str, RulesError> {
    obj[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad(format!("{key} must be a non-empty string")))
}

/// The schema's identifier patterns: 1-128 ASCII bytes, a first byte from `[A-Za-z0-9]` (plus `_` when
/// `first_underscore`), the rest from `[A-Za-z0-9]` and `rest`.
fn is_ident(s: &str, first_underscore: bool, rest: &[u8]) -> bool {
    let b = s.as_bytes();
    (1..=128).contains(&b.len())
        && (b[0].is_ascii_alphanumeric() || (first_underscore && b[0] == b'_'))
        && b[1..]
            .iter()
            .all(|c| c.is_ascii_alphanumeric() || rest.contains(c))
}

/// `rules_id` and `family` (`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`).
fn name(obj: &Map<String, Value>, key: &str) -> Result<String, RulesError> {
    let s = string(obj, key)?;
    if !is_ident(s, false, b"_.:-") {
        return Err(bad(format!(
            "{key} {s:?} does not match the schema pattern"
        )));
    }
    Ok(s.to_string())
}

/// Group member names: 1..=64 unique strings matching `^[A-Za-z0-9_][A-Za-z0-9_.]{0,127}$`.
fn member_list(v: &Value, key: &str) -> Result<Vec<String>, RulesError> {
    let items = v
        .as_array()
        .filter(|a| (1..=MAX_MEMBERS).contains(&a.len()))
        .ok_or_else(|| bad(format!("{key} must be a list of 1 to {MAX_MEMBERS} names")))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let s = item
            .as_str()
            .filter(|s| is_ident(s, true, b"_."))
            .ok_or_else(|| bad(format!("{key} entries must match the schema pattern")))?;
        if out.iter().any(|o| o == s) {
            return Err(bad(format!("{key} has a duplicate")));
        }
        out.push(s.to_string());
    }
    Ok(out)
}

/// Schema bounds of `schemas/artifact_families.schema.json`.
const MAX_MEMBERS: usize = 64;
const MAX_RULES: usize = 1024;
const MAX_PATTERN_CHARS: usize = 4096;
const MAX_NVFP4_GROUP_SIZE: u64 = 4096;
const MAX_PLE_TABLE_SHARDS: u64 = 65536;

impl Rules {
    /// Validates the structure of a parsed rules file.
    pub fn from_value(value: Value) -> Result<Rules, RulesError> {
        let top = object(
            &value,
            "rules file",
            &["schema_version", "rules_id", "groups", "rules"],
        )?;
        if top["schema_version"] != "1" {
            return Err(bad("schema_version must be \"1\""));
        }
        let rules_id = name(top, "rules_id")?;
        let g = object(
            &top["groups"],
            "groups",
            &[
                "nvfp4_group_size",
                "ple_table_shards",
                "ple_module_members",
                "ple_index_members",
            ],
        )?;
        let bounded = |key: &str, max: u64| {
            g[key]
                .as_u64()
                .filter(|n| (1..=max).contains(n))
                .ok_or_else(|| bad(format!("{key} must be an integer in 1..={max}")))
        };
        let groups = Groups {
            nvfp4_group_size: bounded("nvfp4_group_size", MAX_NVFP4_GROUP_SIZE)?,
            ple_table_shards: bounded("ple_table_shards", MAX_PLE_TABLE_SHARDS)?,
            ple_module_members: member_list(&g["ple_module_members"], "ple_module_members")?,
            ple_index_members: member_list(&g["ple_index_members"], "ple_index_members")?,
        };
        let list = top["rules"]
            .as_array()
            .filter(|a| (1..=MAX_RULES).contains(&a.len()))
            .ok_or_else(|| bad(format!("rules must be a list of 1 to {MAX_RULES} rules")))?;
        let mut rules: Vec<Rule> = Vec::with_capacity(list.len());
        for r in list {
            let o = object(
                r,
                "rule",
                &[
                    "family",
                    "pattern",
                    "component",
                    "role",
                    "dtype",
                    "shape_rule",
                    "placement",
                ],
            )?;
            let family = name(o, "family")?;
            if rules.iter().any(|x| x.family == family) {
                return Err(bad(format!("duplicate family {family}")));
            }
            let pattern = string(o, "pattern")?;
            if pattern.chars().count() > MAX_PATTERN_CHARS {
                return Err(bad(format!(
                    "pattern longer than {MAX_PATTERN_CHARS} characters"
                )));
            }
            let pattern = Pattern::new(pattern).map_err(|e| bad(e.to_string()))?;
            let component = string(o, "component")?;
            if !COMPONENTS.contains(&component) {
                return Err(bad(format!("unknown component {component}")));
            }
            let role = Role::parse(string(o, "role")?).ok_or_else(|| bad("unknown role"))?;
            let dtype = Dtype::parse(string(o, "dtype")?).ok_or_else(|| bad("unknown dtype"))?;
            let shape_rule =
                ShapeRule::parse(string(o, "shape_rule")?).ok_or_else(|| bad("bad shape_rule"))?;
            let placement = string(o, "placement")?;
            if !PLACEMENTS.contains(&placement) {
                return Err(bad(format!("unknown placement {placement}")));
            }
            rules.push(Rule {
                family,
                pattern,
                component: component.to_string(),
                role,
                dtype,
                shape_rule,
                placement: placement.to_string(),
            });
        }
        Ok(Rules {
            rules_id,
            groups,
            rules,
            value,
        })
    }
}

/// Why a tensor set fails the rules (artifact classes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FamilyError {
    Unmatched(String),
    Ambiguous(String),
    Dtype(String),
    Shape(String),
    ExpertMetadata(String),
    GroupNvfp4(String),
    GroupPleTable(String),
    GroupPleModule(String),
}

impl FamilyError {
    pub fn class(&self) -> &'static str {
        match self {
            FamilyError::Unmatched(_) => "family_unmatched",
            FamilyError::Ambiguous(_) => "family_ambiguous",
            FamilyError::Dtype(_) => "family_dtype",
            FamilyError::Shape(_) => "family_shape",
            FamilyError::ExpertMetadata(_) => "expert_metadata",
            FamilyError::GroupNvfp4(_) => "group_nvfp4",
            FamilyError::GroupPleTable(_) => "group_ple_table",
            FamilyError::GroupPleModule(_) => "group_ple_module",
        }
    }

    fn detail(&self) -> &str {
        match self {
            FamilyError::Unmatched(d)
            | FamilyError::Ambiguous(d)
            | FamilyError::Dtype(d)
            | FamilyError::Shape(d)
            | FamilyError::ExpertMetadata(d)
            | FamilyError::GroupNvfp4(d)
            | FamilyError::GroupPleTable(d)
            | FamilyError::GroupPleModule(d) => d,
        }
    }
}

impl fmt::Display for FamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class(), self.detail())
    }
}

impl std::error::Error for FamilyError {}

/// Count, bytes and element-misaligned count of a tensor set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    pub count: u64,
    pub bytes: u64,
    pub misaligned: u64,
}

impl Tally {
    pub fn add(&mut self, t: &TensorInfo) {
        self.count += 1;
        self.bytes += t.bytes;
        self.misaligned += u64::from(!t.element_aligned);
    }
}

/// The rule index of every tensor, per shard, in data-offset order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classified {
    pub per_shard: Vec<Vec<usize>>,
    /// Per rule, in rules order.
    pub tallies: Vec<Tally>,
}

/// Shards classified on parallel threads (m216 design note L5): the matcher is the VALIDATE stage's cost.
const CLASSIFY_THREADS: usize = 16;

/// Step 10 for one shard: the rule index of each tensor, or the shard's first unmatched, ambiguous, dtype or shape
/// error in tensor order.
fn classify_shard(rules: &Rules, header: &Header) -> Result<Vec<usize>, FamilyError> {
    let mut idx = Vec::with_capacity(header.tensors.len());
    for t in &header.tensors {
        let mut hits = rules
            .rules
            .iter()
            .enumerate()
            .filter(|(_, r)| r.pattern.full_match(&t.name));
        let (i, rule) = hits
            .next()
            .ok_or_else(|| FamilyError::Unmatched(t.name.clone()))?;
        if hits.next().is_some() {
            return Err(FamilyError::Ambiguous(t.name.clone()));
        }
        if t.dtype != rule.dtype {
            return Err(FamilyError::Dtype(format!(
                "{} is {}, rule {} wants {}",
                t.name, t.dtype, rule.family, rule.dtype
            )));
        }
        if !rule.shape_rule.admits(t.shape.len()) {
            return Err(FamilyError::Shape(format!(
                "{} has rank {}",
                t.name,
                t.shape.len()
            )));
        }
        idx.push(i);
    }
    Ok(idx)
}

/// Runs steps 10-12 over `shards` (in manifest order): classification, expert metadata, groups. Steps 10 and 11 run
/// per shard on up to [`CLASSIFY_THREADS`] threads; results are merged in shard order, so the tallies, the role
/// lists and the first error (step 10 over every shard before step 11) are those of a serial pass.
pub fn classify(rules: &Rules, shards: &[&Header]) -> Result<Classified, FamilyError> {
    type ShardResult = (Result<Vec<usize>, FamilyError>, Result<(), FamilyError>);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(CLASSIFY_THREADS)
        .min(shards.len())
        .max(1);
    let mut done: Vec<Option<ShardResult>> = (0..shards.len()).map(|_| None).collect();
    std::thread::scope(|sc| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let si = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(header) = shards.get(si) else {
                            return out;
                        };
                        out.push((
                            si,
                            (classify_shard(rules, header), check_expert_metadata(header)),
                        ));
                    }
                })
            })
            .collect();
        for w in workers {
            for (si, r) in w.join().expect("classify thread") {
                done[si] = Some(r);
            }
        }
    });
    let mut per_shard = Vec::with_capacity(shards.len());
    let mut metadata = Vec::with_capacity(shards.len());
    let mut tallies = vec![Tally::default(); rules.rules.len()];
    let mut by_role: BTreeMap<Role, Vec<&TensorInfo>> = BTreeMap::new();
    for (header, r) in shards.iter().zip(done) {
        let (idx, meta) = r.expect("every shard classified");
        let idx = idx?;
        for (t, &i) in header.tensors.iter().zip(&idx) {
            tallies[i].add(t);
            by_role.entry(rules.rules[i].role).or_default().push(t);
        }
        per_shard.push(idx);
        metadata.push(meta);
    }
    for meta in metadata {
        meta?;
    }
    let get = |r: Role| by_role.get(&r).map_or(&[][..], Vec::as_slice);
    check_nvfp4(
        rules.groups.nvfp4_group_size,
        [
            (Role::Nvfp4Weight, get(Role::Nvfp4Weight)),
            (Role::Nvfp4BlockScale, get(Role::Nvfp4BlockScale)),
            (Role::Nvfp4GlobalScale, get(Role::Nvfp4GlobalScale)),
            (Role::Nvfp4InputScale, get(Role::Nvfp4InputScale)),
        ],
    )?;
    check_ple_table(
        &rules.groups,
        get(Role::PleTable),
        get(Role::PleScale),
        get(Role::PleIndex),
    )?;
    check_ple_module(&rules.groups, get(Role::PleModule))?;
    Ok(Classified { per_shard, tallies })
}

fn ascii_decimal(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// The first `(^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.` in `name` (Python `re.search` semantics).
pub fn layer_expert(name: &str) -> Option<(u64, u64)> {
    let b = name.as_bytes();
    let digits = |from: usize| b[from..].iter().take_while(|c| c.is_ascii_digit()).count();
    (0..b.len()).find_map(|i| {
        if i > 0 && b[i - 1] != b'.' {
            return None;
        }
        let rest = name.get(i..)?.strip_prefix("layers.")?;
        let at = name.len() - rest.len();
        let nl = digits(at);
        let after = name[at + nl..].strip_prefix(".mlp.experts.")?;
        let at2 = name.len() - after.len();
        let ne = digits(at2);
        if nl == 0 || ne == 0 || b.get(at2 + ne) != Some(&b'.') {
            return None;
        }
        Some((
            ascii_decimal(&name[at..at + nl])?,
            ascii_decimal(&name[at2..at2 + ne])?,
        ))
    })
}

/// Q5e: a shard declaring `layer`/`expert_start`/`expert_end` holds exactly those experts of that layer.
fn check_expert_metadata(header: &Header) -> Result<(), FamilyError> {
    let Some(meta) = &header.metadata else {
        return Ok(());
    };
    let keys = ["layer", "expert_start", "expert_end"];
    if !keys.iter().any(|k| meta.contains_key(*k)) {
        return Ok(());
    }
    let err = |d: String| FamilyError::ExpertMetadata(d);
    let num = |k: &str| {
        meta.get(k)
            .and_then(|v| ascii_decimal(v))
            .ok_or_else(|| err(format!("{k} is not a decimal string")))
    };
    let (layer, start, end) = (num("layer")?, num("expert_start")?, num("expert_end")?);
    let mut seen = BTreeSet::new();
    for t in &header.tensors {
        match layer_expert(&t.name) {
            Some((l, e)) if l == layer && start <= e && e < end => {
                seen.insert(e);
            }
            _ => {
                return Err(err(format!(
                    "{} is outside layer {layer} experts [{start}, {end})",
                    t.name
                )))
            }
        }
    }
    if !seen.iter().copied().eq(start..end.max(start)) {
        return Err(err(format!("experts present differ from [{start}, {end})")));
    }
    Ok(())
}

/// Q5a: every NVFP4 prefix has weight `[o, i/2]`, block scale `[o, i/G]`, and `[]` global and input scales.
fn check_nvfp4(group: u64, members: [(Role, &[&TensorInfo]); 4]) -> Result<(), FamilyError> {
    let err = |d: String| FamilyError::GroupNvfp4(d);
    let mut quads: BTreeMap<&str, [Option<&TensorInfo>; 4]> = BTreeMap::new();
    for (slot, (role, tensors)) in members.iter().enumerate() {
        let suffix = role.nvfp4_suffix().expect("nvfp4 role");
        for t in *tensors {
            let prefix = t
                .name
                .strip_suffix(suffix)
                .ok_or_else(|| err(format!("{} lacks suffix {suffix}", t.name)))?;
            let quad = quads.entry(prefix).or_default();
            if quad[slot].replace(t).is_some() {
                return Err(err(format!("{} repeated", t.name)));
            }
        }
    }
    for (prefix, quad) in quads {
        let [Some(w), Some(s), Some(g), Some(x)] = quad else {
            return Err(err(format!("{prefix} quad is incomplete")));
        };
        let (&[o, half], &[so, blocks]) = (w.shape.as_slice(), s.shape.as_slice()) else {
            return Err(err(format!("{prefix} weight/scale must be rank 2")));
        };
        let i = half
            .checked_mul(2)
            .ok_or_else(|| err(format!("{prefix} input dim overflows")))?;
        if so != o
            || i % group != 0
            || blocks != i / group
            || !g.shape.is_empty()
            || !x.shape.is_empty()
        {
            return Err(err(format!("{prefix} quad shapes disagree")));
        }
    }
    Ok(())
}

/// The decimal `k` of a name ending `shard_<k>.weight`.
fn shard_index(name: &str) -> Option<u64> {
    let head = name.strip_suffix(".weight")?;
    let digits = head.len() - head.bytes().rev().take_while(u8::is_ascii_digit).count();
    let k = &head[digits..];
    head[..digits].strip_suffix("shard_")?;
    ascii_decimal(k)
}

/// Exactly the `members`, each once, matched by name suffix `.<member>`.
fn exact_members<'a>(
    tensors: &[&'a TensorInfo],
    members: &[String],
) -> Option<Vec<&'a TensorInfo>> {
    let mut found: Vec<Option<&TensorInfo>> = vec![None; members.len()];
    for t in tensors {
        let slot = members.iter().position(|m| {
            t.name
                .strip_suffix(m.as_str())
                .is_some_and(|p| p.ends_with('.'))
        })?;
        if found[slot].replace(t).is_some() {
            return None;
        }
    }
    found.into_iter().collect()
}

/// Q5b: table shards `0..N` with identical rank-2 shapes, one `[1]` scale, and exactly the index members. A U8 table
/// is the NVFP4 PLE layout (fp4local design note D8): its scales are checked by [`check_ple_nvfp4_scales`].
fn check_ple_table(
    groups: &Groups,
    table: &[&TensorInfo],
    scale: &[&TensorInfo],
    index: &[&TensorInfo],
) -> Result<(), FamilyError> {
    if table.is_empty() && scale.is_empty() && index.is_empty() {
        return Ok(());
    }
    let err = |d: &str| FamilyError::GroupPleTable(d.to_string());
    let mut ks = BTreeSet::new();
    for t in table {
        let k = shard_index(&t.name).ok_or_else(|| err("table name lacks shard_<k>.weight"))?;
        if !ks.insert(k) {
            return Err(err("table shard repeated"));
        }
        if t.shape.len() != 2 || t.shape != table[0].shape {
            return Err(err("table shards differ in shape"));
        }
    }
    if !ks.iter().copied().eq(0..groups.ple_table_shards) {
        return Err(err("table shards incomplete"));
    }
    if table.first().is_some_and(|t| t.dtype == Dtype::U8) {
        check_ple_nvfp4_scales(groups, &table[0].shape, &ks, scale)?;
    } else if scale.len() != 1 || scale[0].shape != [1] {
        return Err(err("need exactly one [1] scale"));
    }
    exact_members(index, &groups.ple_index_members).ok_or_else(|| err("index members differ"))?;
    Ok(())
}

/// D8 NVFP4 PLE scales: per table shard `shard_<k>.weight_scale` F8_E4M3 `[rows, 2 * cols / nvfp4_group_size]` (the
/// table is U8 `[rows, cols]`, two E2M1 values per byte), and exactly one F32 `[1]` global `weight_scale_2`.
fn check_ple_nvfp4_scales(
    groups: &Groups,
    table_shape: &[u64],
    shards: &BTreeSet<u64>,
    scale: &[&TensorInfo],
) -> Result<(), FamilyError> {
    let err = |d: &str| FamilyError::GroupPleTable(d.to_string());
    let (rows, cols) = (table_shape[0], table_shape[1]);
    let group = groups.nvfp4_group_size;
    if group == 0 || (2 * cols) % group != 0 {
        return Err(err("NVFP4 table width is not a whole number of groups"));
    }
    let want = [rows, 2 * cols / group];
    let mut seen = BTreeSet::new();
    let mut globals = 0;
    for s in scale {
        if let Some(k) = s.name.strip_suffix("_scale").and_then(shard_index) {
            if s.dtype != Dtype::F8E4m3 || s.shape != want || !seen.insert(k) {
                return Err(err(
                    "NVFP4 shard scale repeated or not F8_E4M3 [rows, 2*cols/group]",
                ));
            }
        } else if s.name.ends_with(".weight_scale_2") && s.dtype == Dtype::F32 && s.shape == [1] {
            globals += 1;
        } else {
            return Err(err(
                "NVFP4 scale is neither a shard scale nor an F32 [1] weight_scale_2",
            ));
        }
    }
    if seen != *shards || globals != 1 {
        return Err(err(
            "need one scale per NVFP4 shard and exactly one weight_scale_2",
        ));
    }
    Ok(())
}

/// Q5c: exactly the module members, all BF16.
fn check_ple_module(groups: &Groups, module: &[&TensorInfo]) -> Result<(), FamilyError> {
    if module.is_empty() {
        return Ok(());
    }
    let err = |d: &str| FamilyError::GroupPleModule(d.to_string());
    let found = exact_members(module, &groups.ple_module_members)
        .ok_or_else(|| err("module members differ"))?;
    if found.iter().any(|t| t.dtype != Dtype::Bf16) {
        return Err(err("module weight not BF16"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
