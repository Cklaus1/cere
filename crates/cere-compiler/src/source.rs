//! The read-only source model directory (m21 design note ground rules): opened once as `O_DIRECTORY`, every
//! file reached by `openat(..., O_NOFOLLOW)`, required to be regular, and size-checked against `fstat`.
use crate::fsio::{Dir, EntryKind, OpenFile};
use crate::{canon_json, sha256_hex, Error};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Q4 file roles.
pub mod role {
    pub const WEIGHTS: &str = "weights";
    pub const WEIGHT_INDEX: &str = "weight_index";
    pub const CONFIG: &str = "config";
    pub const QUANT_CONFIG: &str = "quant_config";
    pub const GENERATION_CONFIG: &str = "generation_config";
    pub const TOKENIZER: &str = "tokenizer";
    pub const TEMPLATE: &str = "template";
    pub const PREPROCESSOR: &str = "preprocessor";
    pub const DRAFT_TOKEN_MAP: &str = "draft_token_map";
}

/// The role a top-level source file name carries, or `None` for a non-payload file.
pub fn payload_role(name: &str) -> Option<&'static str> {
    Some(match name {
        "model.safetensors.index.json" => role::WEIGHT_INDEX,
        "config.json" => role::CONFIG,
        "hf_quant_config.json" => role::QUANT_CONFIG,
        "generation_config.json" => role::GENERATION_CONFIG,
        "tokenizer.json" | "tokenizer_config.json" | "vocab.json" | "merges.txt" => role::TOKENIZER,
        "chat_template.jinja" => role::TEMPLATE,
        // `processor_config.json`: Inkling 2 Mini's image processor (inkling design note S6g; no other source has it).
        "preprocessor_config.json" | "video_preprocessor_config.json" | "processor_config.json" => {
            role::PREPROCESSOR
        }
        _ if is_shard(name) => role::WEIGHTS,
        _ => return None,
    })
}

pub fn is_shard(name: &str) -> bool {
    name.len() > ".safetensors".len() && name.ends_with(".safetensors")
}

/// One file of the HF tree metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeEntry {
    pub size: u64,
    pub blob_id: String,
    pub lfs_sha256: Option<String>,
}

/// `.cache/huggingface/trees/<revision>.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    pub revision: String,
    pub sha256: String,
    pub bytes: u64,
    pub files: BTreeMap<String, TreeEntry>,
}

/// A selected source file, opened and `fstat`ed.
#[derive(Debug)]
pub struct SourceFile {
    pub name: String,
    pub role: &'static str,
    pub open: OpenFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Select {
    /// Top-level `*.safetensors` only (inventory).
    Shards,
    /// Every Q4 payload role (prepare).
    Payload,
}

#[derive(Debug)]
pub struct Source {
    /// `realpath` of SRC.
    pub path: PathBuf,
    pub dir: Dir,
    /// Selected files in bytewise name order.
    pub files: Vec<SourceFile>,
    /// Other top-level entries, bytewise sorted (prepare reports them).
    pub excluded: Vec<String>,
    pub tree: Option<Tree>,
    /// Bytes of the tree file read.
    pub metadata_bytes_read: u64,
}

/// Largest small file read whole into memory (Q7: < 16 MiB).
pub const SMALL_FILE_MAX: u64 = 16 << 20;

/// Reads a whole file through its held descriptor; the size must still equal the `fstat` at open.
pub fn read_all(open: &OpenFile, name: &str) -> Result<Vec<u8>, Error> {
    let size = usize::try_from(open.stat.size)
        .map_err(|_| Error::reject("file_size", format!("{name} too large")))?;
    let mut buf = vec![0u8; size];
    open.read_exact_at(&mut buf, 0)
        .map_err(|e| Error::io(name, e))?;
    check_unchanged(open, name)?;
    Ok(buf)
}

/// Exits 2 when the file's size changed since it was opened.
pub fn check_unchanged(open: &OpenFile, name: &str) -> Result<(), Error> {
    let now = open.restat().map_err(|e| Error::io(name, e))?;
    if now.size != open.stat.size {
        return Err(Error::reject(
            "source_changed",
            format!("{name} changed size during the read"),
        ));
    }
    Ok(())
}

/// Opens SRC and the selected files.
pub fn open(path: &Path, select: Select) -> Result<Source, Error> {
    let real = std::fs::canonicalize(path)
        .map_err(|e| Error::reject("source", format!("{}: {e}", path.display())))?;
    let dir = Dir::open(&real)
        .map_err(|e| Error::reject("source", format!("{}: {e}", real.display())))?;
    let mut entries = dir.entries().map_err(|e| Error::io(real.display(), e))?;
    entries.sort();
    let mut files = Vec::new();
    let mut excluded = Vec::new();
    for (name, kind) in entries {
        let role = match select {
            Select::Shards => is_shard(&name).then_some(role::WEIGHTS),
            Select::Payload => payload_role(&name),
        };
        let Some(role) = role else {
            excluded.push(name);
            continue;
        };
        match kind {
            EntryKind::Symlink => {
                return Err(Error::reject(
                    "symlink",
                    format!("source {name} is a symlink"),
                ))
            }
            EntryKind::Dir | EntryKind::Other => {
                return Err(Error::reject(
                    "special_file",
                    format!("source {name} is not a regular file"),
                ))
            }
            EntryKind::File => {}
        }
        let open = dir.open_file(&name)?;
        files.push(SourceFile { name, role, open });
    }
    let (tree, metadata_bytes_read) = read_tree(&dir)?;
    Ok(Source {
        path: real,
        dir,
        files,
        excluded,
        tree,
        metadata_bytes_read,
    })
}

const TREES_DIR: &str = ".cache/huggingface/trees";

fn tree_err(msg: impl Into<String>) -> Error {
    Error::reject("source_tree", msg)
}

/// The single HF tree file, if any (none → `None`; several → error).
fn read_tree(dir: &Dir) -> Result<(Option<Tree>, u64), Error> {
    let trees = match dir.open_dir(TREES_DIR) {
        Ok(d) => d,
        Err(e) if e.kind == crate::fsio::FsErrorKind::Missing => return Ok((None, 0)),
        Err(e) => return Err(e.into()),
    };
    let mut names: Vec<String> = trees
        .entries()
        .map_err(|e| Error::io(TREES_DIR, e))?
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.ends_with(".json"))
        .collect();
    names.sort();
    let name = match names.as_slice() {
        [] => return Ok((None, 0)),
        [one] => one.clone(),
        _ => return Err(tree_err(format!("several tree files in {TREES_DIR}"))),
    };
    let open = trees.open_file(&name)?;
    if open.stat.size > SMALL_FILE_MAX {
        return Err(tree_err("tree file too large"));
    }
    let bytes = read_all(&open, &name)?;
    let tree = parse_tree(&name, &bytes)?;
    Ok((Some(tree), bytes.len() as u64))
}

fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Parses the HF tree file `<revision>.json`: the revision is 40 lowercase hex, the body
/// `{"files": {name: {size, blob_id, lfs_sha256?}}}` with u64 sizes, 40-hex blob ids and 64-hex lfs digests (what
/// the inventory and manifest schemas need). Anything else is `source_tree`, as `model_inventory.py parse_tree`
/// (shared cases: the `inventory` tag of `tests/fixtures/artifact_invalid.json`).
fn parse_tree(name: &str, bytes: &[u8]) -> Result<Tree, Error> {
    let unparsable = || tree_err(format!("unparsable HF tree file {name}"));
    let revision = name.strip_suffix(".json").ok_or_else(unparsable)?;
    if !is_lower_hex(revision, 40) {
        return Err(unparsable());
    }
    let v = canon_json::parse_strict(bytes).map_err(|_| unparsable())?;
    let files = v
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(unparsable)?;
    let mut out = BTreeMap::new();
    for (file, e) in files {
        let size = e
            .get("size")
            .and_then(Value::as_u64)
            .ok_or_else(unparsable)?;
        let blob_id = e
            .get("blob_id")
            .and_then(Value::as_str)
            .filter(|s| is_lower_hex(s, 40))
            .ok_or_else(unparsable)?;
        let lfs_sha256 = match e.get("lfs_sha256") {
            None => None,
            Some(Value::String(s)) if is_lower_hex(s, 64) => Some(s.clone()),
            Some(_) => return Err(unparsable()),
        };
        out.insert(
            file.clone(),
            TreeEntry {
                size,
                blob_id: blob_id.to_string(),
                lfs_sha256,
            },
        );
    }
    Ok(Tree {
        revision: revision.to_string(),
        sha256: sha256_hex(bytes),
        bytes: bytes.len() as u64,
        files: out,
    })
}
