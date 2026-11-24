//! `cerec convert`: the one-time offline FP4-Local -> CERE-layout converter (fp4local design note D2-D10, FL-10).
//!
//! Reads a modelopt `MIXED_PRECISION` checkpoint through its `model.safetensors.index.json` (shards may live in a
//! subdirectory) and writes a flat directory whose tensors carry the shipped NVFP4 checkpoint's names, dtypes and
//! shapes, except the PLE table, which stays NVFP4 (D8). Per-tensor rules, decided by name and dtype:
//!
//! - vision (`model.visual.*`): never written (D10);
//! - MXFP8 (`X.weight` F8_E4M3 [o, i] + `X.weight_scale` U8 [o, i/32], UE8M0): exact upcast to `X.weight` BF16; any
//!   value that is not exactly a BF16 value is refused (D4);
//! - NVFP4-CSF routed scales (`X.nvfp4_csf_fixed` U8 + `X.nvfp4_csf_exceptions` U32): expanded byte-exactly to `X`
//!   F8_E4M3 [rows, cols]; streams that are not in canonical form are refused (D5);
//! - MTP routed experts (`mtp.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2}`):
//!   dequantized in F32 (E2M1 x E4M3 exact, x global, one F32 rounding) and rounded to BF16 (RNE), fused into
//!   `mtp.layers.L.mlp.experts.gate_up_proj` [E, 2I, H] (`[gate; up]` along dim 1) and `down_proj` [E, H, I] (D7);
//! - F32 norm weights (`*norm*.weight`): RNE to BF16; values that change are counted (D6);
//! - everything else (routed NVFP4 weights and scalars, BF16, I64, the NVFP4 PLE table): byte copy.
//!
//! `--format s3` (fp4local design note S3, FL-30) changes two rules, the rest is as above (the default, `s2`):
//!
//! - MXFP8: `X.weight` F8_E4M3 and `X.weight_scale` U8 kept byte for byte, canonical only (i % 32 == 0, every
//!   scale in [`MXFP8_MIN_SCALE`, `MXFP8_MAX_SCALE`] so every product is a normal BF16, no E4M3 NaN byte);
//! - MTP experts: kept NVFP4 W4A16, fused like S2: `gate_up_proj` U8 [E, 2I, H/2], `gate_up_proj_scale` F8_E4M3
//!   [E, 2I, H/16], `gate_up_proj_scale_2` F32 [E, 2] (gate, up), `down_proj` U8 [E, H, I/2], `down_proj_scale`
//!   F8_E4M3 [E, H, I/16], `down_proj_scale_2` F32 [E]; every piece is checked to dequantize under the S2 rule.
//!
//! Small text payload files are byte-copied (config.json keeps the source `quantization_config`; CERE reads only
//! `text_config`). The source must hold the HF tree `cerec prepare` reads (`.cache/huggingface/trees/<rev>.json`);
//! the output gets a copy without the entries of rewritten files (the index), so `prepare` checks the byte-copied
//! files against their upstream digests. `CONVERSION.json` records the source revision and tree digest, the converter
//! git sha, the rule counts, the sha256 of every written file and, for `s3` only, the format (an `s2` file is
//! unchanged); it is deterministic.
use crate::fsio::{Dir, OpenFile};
use crate::safetensors::{self, Dtype, Header, TensorInfo};
use crate::{hex, Error, Limits};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

/// Text payload files copied byte for byte when present (`config.json` is required). Vision preprocessor configs
/// are not copied (D10, text only).
pub const COPIED_FILES: [&str; 7] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
];
pub const CONVERSION_FILE: &str = "CONVERSION.json";
pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const MTP_FILE: &str = "cere-mtp-experts.safetensors";
const FIXED: &str = ".nvfp4_csf_fixed";
const EXCEPTIONS: &str = ".nvfp4_csf_exceptions";
/// Largest canonical CSF row base (codec "byte-window4-fixed-stream-u24-exceptions/1").
pub const CSF_MAX_BASE: u8 = 240;
const COPY_CHUNK: usize = 16 << 20;
/// Canonical S3 UE8M0 scale bounds (`k2m::kMinScale`/`kMaxScale`): 2^-9 x 2^(10-127) = 2^-126 is the smallest
/// normal BF16 and 448 x 2^(246-127) < BF16 max, so every E4M3 x scale product is a normal BF16 value.
pub const MXFP8_MIN_SCALE: u8 = 10;
pub const MXFP8_MAX_SCALE: u8 = 246;

/// The output layout (fp4local design note S2/S3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Today's executor layout: MXFP8 upcast, MTP experts BF16 (the default).
    S2,
    /// MXFP8 dense and NVFP4 MTP experts kept for the native kernels.
    S3,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s {
            "s2" => Some(Format::S2),
            "s3" => Some(Format::S3),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::S2 => "s2",
            Format::S3 => "s3",
        }
    }
}

fn reject(class: &'static str, msg: impl Into<String>) -> Error {
    Error::reject(class, msg)
}

// ---------------------------------------------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------------------------------------------

/// The value of an E4M3 (FN) byte; `None` for NaN (`0x7F`/`0xFF`).
pub fn e4m3_value(b: u8) -> Option<f32> {
    let e = (b >> 3) & 0xF;
    let m = b & 7;
    if e == 0xF && m == 7 {
        return None;
    }
    let mag = if e == 0 {
        f32::from(m) * 2f32.powi(-9)
    } else {
        f32::from(8 + m) * 2f32.powi(i32::from(e) - 10)
    };
    Some(if b & 0x80 != 0 { -mag } else { mag })
}

/// E2M1 code -> value (bit 3 is the sign).
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// F32 -> BF16 bits, round to nearest even; NaN stays a (quiet) NaN.
pub fn bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    // Non-NaN bits are <= 0xFF80_0000, so the addition cannot overflow.
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// The exact BF16 bits of `e4m3(w) x 2^(s-127)`, or `None` when that is NaN or not a BF16 value.
fn mxfp8_exact(w: u8, s: u8) -> Option<u16> {
    if s == 0xFF {
        return None;
    }
    let v = f64::from(e4m3_value(w)?) * 2f64.powi(i32::from(s) - 127);
    let f = v as f32;
    if f64::from(f) != v || f.is_infinite() {
        return None;
    }
    let bits = f.to_bits();
    (bits & 0xFFFF == 0).then_some((bits >> 16) as u16)
}

/// `[scale * 256 + byte]` -> BF16 bits, bit 16 set when not exact.
static MXFP8_TABLE: LazyLock<Vec<u32>> = LazyLock::new(|| {
    (0..=255u8)
        .flat_map(|s| (0..=255u8).map(move |w| mxfp8_exact(w, s).map_or(1 << 16, u32::from)))
        .collect()
});

/// MXFP8 [rows, cols] E4M3 bytes with UE8M0 scales [rows, cols/32] -> BF16 bytes (little-endian), exact.
pub fn mxfp8_to_bf16(
    weight: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, Error> {
    let err = |m: String| reject("convert_mxfp8", m);
    if cols % 32 != 0 || weight.len() != rows * cols || scales.len() != rows * (cols / 32) {
        return Err(err(format!(
            "weight {} B and scales {} B do not form a [{rows}, {cols}] MXFP8 tensor with 32-wide groups",
            weight.len(),
            scales.len()
        )));
    }
    let table: &[u32] = &MXFP8_TABLE;
    let mut out = Vec::with_capacity(weight.len() * 2);
    for (i, (group, &s)) in weight.chunks_exact(32).zip(scales).enumerate() {
        let row = &table[usize::from(s) * 256..usize::from(s) * 256 + 256];
        for (j, &w) in group.iter().enumerate() {
            let e = row[usize::from(w)];
            if e >> 16 != 0 {
                return Err(err(format!(
                    "element {} (byte {w:#04x}, scale {s}) is NaN or not exactly a BF16 value",
                    i * 32 + j
                )));
            }
            out.extend_from_slice(&(e as u16).to_le_bytes());
        }
    }
    Ok(out)
}

/// S3 canonical MXFP8 weight bytes: no E4M3 NaN byte (`0x7F`/`0xFF`). With canonical scales
/// ([`check_mxfp8_scales`]) and i % 32 == 0 (checked by the plan) every upcast value is a normal BF16 value.
pub fn check_mxfp8_weight(weight: &[u8]) -> Result<(), Error> {
    match weight.iter().position(|&b| b & 0x7F == 0x7F) {
        Some(i) => Err(reject(
            "convert_mxfp8",
            format!("element {i} is the E4M3 NaN byte {:#04x}", weight[i]),
        )),
        None => Ok(()),
    }
}

/// S3 canonical UE8M0 scale bytes: every one in [`MXFP8_MIN_SCALE`, `MXFP8_MAX_SCALE`].
pub fn check_mxfp8_scales(scales: &[u8]) -> Result<(), Error> {
    match scales
        .iter()
        .position(|s| !(MXFP8_MIN_SCALE..=MXFP8_MAX_SCALE).contains(s))
    {
        Some(i) => Err(reject(
            "convert_mxfp8",
            format!(
                "scale {i} is {}, outside the canonical [{MXFP8_MIN_SCALE}, {MXFP8_MAX_SCALE}]",
                scales[i]
            ),
        )),
        None => Ok(()),
    }
}

/// `(rows, cols)` of the plane of a `.nvfp4_csf_fixed` stream of shape `[rows/16, 16 * (1 + cols/2)]`.
pub fn csf_geometry(shape: &[u64]) -> Result<(usize, usize), Error> {
    match *shape {
        [slabs, width] if slabs > 0 && width > 16 && width % 16 == 0 => {
            let (rows, cols) = (slabs as usize * 16, (width as usize / 16 - 1) * 2);
            if rows.checked_mul(cols).is_none_or(|n| n > 1 << 24) {
                return Err(reject(
                    "convert_csf",
                    "plane exceeds the 24-bit exception positions",
                ));
            }
            Ok((rows, cols))
        }
        _ => Err(reject(
            "convert_csf",
            format!("fixed stream shape {shape:?} is not [rows/16, 16*(1+cols/2)]"),
        )),
    }
}

/// The plain E4M3 plane of one NVFP4-CSF stream pair, refusing anything but the canonical form: row bases <= 240,
/// exception positions strictly increasing and inside the plane, each over a zero offset nibble with a byte outside
/// its row's `[base, base + 15]` window (the streams are then the unique encoding of the plane).
pub fn csf_expand(fixed: &[u8], words: &[u32], rows: usize, cols: usize) -> Result<Vec<u8>, Error> {
    let err = |m: String| reject("convert_csf", m);
    let half = cols / 2;
    let width = 16 * (1 + half);
    if rows % 16 != 0 || cols % 2 != 0 || fixed.len() != rows / 16 * width {
        return Err(err(format!(
            "fixed stream of {} B does not hold a [{rows}, {cols}] plane",
            fixed.len()
        )));
    }
    let base_of = |r: usize| fixed[r / 16 * width + r % 16];
    let codes_of = |r: usize| {
        let start = r / 16 * width + 16 + r % 16 * half;
        &fixed[start..start + half]
    };
    let mut plane = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let base = base_of(r);
        if base > CSF_MAX_BASE {
            return Err(err(format!("row {r} base {base} above {CSF_MAX_BASE}")));
        }
        for &code in codes_of(r) {
            plane.push(base + (code & 15));
            plane.push(base + (code >> 4));
        }
    }
    let mut prev: Option<usize> = None;
    for &w in words {
        let p = (w & 0xFF_FFFF) as usize;
        let byte = (w >> 24) as u8;
        if prev.is_some_and(|q| p <= q) {
            return Err(err("exception positions are not strictly increasing".into()));
        }
        if p >= rows * cols {
            return Err(err(format!(
                "exception position {p} outside the [{rows}, {cols}] plane"
            )));
        }
        prev = Some(p);
        let (r, c) = (p / cols, p % cols);
        let code = codes_of(r)[c / 2];
        if (if c % 2 == 1 { code >> 4 } else { code & 15 }) != 0 {
            return Err(err(format!(
                "exception at {p} over a non-zero offset nibble"
            )));
        }
        if byte.wrapping_sub(base_of(r)) <= 15 && byte >= base_of(r) {
            return Err(err(format!("exception at {p} inside its row's window")));
        }
        plane[p] = byte;
    }
    Ok(plane)
}

/// F32 (little-endian) -> BF16 RNE bytes, and the number of values that changed (not BF16-representable).
pub fn f32_to_bf16_rne(src: &[u8]) -> (Vec<u8>, u64) {
    let mut out = Vec::with_capacity(src.len() / 2);
    let mut inexact = 0;
    for v in src.chunks_exact(4) {
        let x = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
        inexact += u64::from(!x.is_nan() && x.to_bits() & 0xFFFF != 0);
        out.extend_from_slice(&bf16_rne(x).to_le_bytes());
    }
    (out, inexact)
}

/// NVFP4 [rows, cols] (packed [rows, cols/2], even column in the low nibble; E4M3 scales [rows, cols/16]; F32
/// global) -> BF16 bytes, appended to `out`: `bf16_rne((e2m1 x scale) x global)` in F32. Returns the number of
/// values whose F32 product is not a BF16 value (rounded by the RNE).
pub fn nvfp4_to_bf16(
    packed: &[u8],
    scales: &[u8],
    global: f32,
    rows: usize,
    cols: usize,
    out: &mut Vec<u8>,
) -> Result<u64, Error> {
    let err = |m: String| reject("convert_nvfp4", m);
    if cols % 16 != 0 || packed.len() != rows * cols / 2 || scales.len() != rows * (cols / 16) {
        return Err(err(format!(
            "packed {} B and scales {} B do not form a [{rows}, {cols}] NVFP4 tensor",
            packed.len(),
            scales.len()
        )));
    }
    if !global.is_finite() {
        return Err(err(format!("global scale {global} is not finite")));
    }
    let mut inexact = 0;
    for (block, &sb) in packed.chunks_exact(8).zip(scales) {
        let s = e4m3_value(sb).ok_or_else(|| err(format!("block scale {sb:#04x} is NaN")))?;
        for &byte in block {
            for code in [byte & 15, byte >> 4] {
                let v = (E2M1[usize::from(code)] * s) * global;
                inexact += u64::from(v.to_bits() & 0xFFFF != 0);
                out.extend_from_slice(&bf16_rne(v).to_le_bytes());
            }
        }
    }
    Ok(inexact)
}

// ---------------------------------------------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------------------------------------------

/// One source tensor: its file (index into the plan's source files) and header entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Src {
    pub file: usize,
    pub info: TensorInfo,
}

/// An NVFP4 W4A16 expert projection: packed weight, E4M3 block scales, F32 global.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nvfp4Src {
    pub weight: Src,
    pub scale: Src,
    pub global: Src,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Copy(Src),
    Mxfp8 {
        weight: Src,
        scale: Src,
    },
    /// S3: the E4M3 weight bytes kept (canonical check).
    Mxfp8Weight(Src),
    /// S3: the UE8M0 scale bytes kept (canonical check).
    Mxfp8Scale(Src),
    Csf {
        fixed: Src,
        exceptions: Src,
    },
    NormRne(Src),
    /// Per expert `[gate; up]`.
    MtpGateUp(Vec<(Nvfp4Src, Nvfp4Src)>),
    MtpDown(Vec<Nvfp4Src>),
    /// S3: one NVFP4 piece of every expert's projections in order (`[gate, up]` or `[down]` per expert).
    MtpNvfp4 {
        part: MtpPart,
        experts: Vec<Vec<Nvfp4Src>>,
    },
}

/// Which NVFP4 piece an S3 fused MTP tensor holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpPart {
    /// Packed E2M1 bytes (checked to dequantize under the S2 rule).
    Weight,
    /// E4M3 block scales.
    Scale,
    /// F32 global scales.
    Global,
}

/// The rule names reported per op.
impl Op {
    pub fn rule(&self) -> &'static str {
        match self {
            Op::Copy(_) => "copy",
            Op::Mxfp8 { .. } => "mxfp8_to_bf16",
            Op::Mxfp8Weight(_) => "mxfp8_e4m3_kept",
            Op::Mxfp8Scale(_) => "mxfp8_scale_kept",
            Op::Csf { .. } => "csf_to_e4m3",
            Op::NormRne(_) => "f32_norm_to_bf16",
            Op::MtpGateUp(_) | Op::MtpDown(_) => "mtp_nvfp4_to_bf16_fused",
            Op::MtpNvfp4 { .. } => "mtp_nvfp4_fused",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub op: Op,
}

impl OutTensor {
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.dtype.itemsize()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutFile {
    pub name: String,
    pub tensors: Vec<OutTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub files: Vec<OutFile>,
    /// Source tensors under `model.visual.` (never written).
    pub dropped_vision: u64,
}

fn mtp_expert(name: &str) -> Option<(u64, u64, &str, &str)> {
    let rest = name.strip_prefix("mtp.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (expert, rest) = rest.split_once('.')?;
    let (proj, part) = rest.split_once('.')?;
    let proj = proj.strip_suffix("_proj")?;
    if !matches!(proj, "gate" | "up" | "down")
        || !matches!(part, "weight" | "weight_scale" | "weight_scale_2")
    {
        return None;
    }
    let num = |s: &str| {
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok())?
    };
    Some((num(layer)?, num(expert)?, proj, part))
}

fn is_norm_weight(name: &str) -> bool {
    name.strip_suffix(".weight")
        .and_then(|m| m.rsplit('.').next())
        .is_some_and(|last| last.contains("norm"))
}

fn want(t: &TensorInfo, dtype: Dtype, rank: usize, what: &str) -> Result<(), Error> {
    if t.dtype != dtype || t.shape.len() != rank {
        return Err(reject(
            "convert_plan",
            format!(
                "{}: {what} must be {dtype} rank {rank}, got {} {:?}",
                t.name, t.dtype, t.shape
            ),
        ));
    }
    Ok(())
}

/// Output shape `[rows, cols]` of an NVFP4 projection (packed [rows, cols/2], scales [rows, cols/16], global []).
fn nvfp4_shape(p: &Nvfp4Src) -> Result<(u64, u64), Error> {
    want(&p.weight.info, Dtype::U8, 2, "NVFP4 weight")?;
    want(&p.scale.info, Dtype::F8E4m3, 2, "NVFP4 block scale")?;
    want(&p.global.info, Dtype::F32, 0, "NVFP4 global scale")?;
    let (rows, cols) = (p.weight.info.shape[0], p.weight.info.shape[1] * 2);
    if cols % 16 != 0 || p.scale.info.shape != [rows, cols / 16] {
        return Err(reject(
            "convert_plan",
            format!("{}: NVFP4 weight/scale shapes disagree", p.weight.info.name),
        ));
    }
    Ok((rows, cols))
}

/// Builds the plan from the source shards (`(output-relevant file name, header)` in index order). Output files: one
/// `cere-<source file name>` per source shard that keeps a tensor, plus [`MTP_FILE`] for the fused MTP experts.
pub fn plan(sources: &[(String, Header)], format: Format) -> Result<Plan, Error> {
    let mut by_name: BTreeMap<&str, Src> = BTreeMap::new();
    for (file, (fname, h)) in sources.iter().enumerate() {
        for t in &h.tensors {
            if by_name
                .insert(
                    &t.name,
                    Src {
                        file,
                        info: t.clone(),
                    },
                )
                .is_some()
            {
                return Err(reject(
                    "convert_plan",
                    format!("tensor {} in two shards ({fname})", t.name),
                ));
            }
        }
    }
    let mut consumed: BTreeSet<&str> = BTreeSet::new();
    let mut per_file: Vec<Vec<OutTensor>> = vec![Vec::new(); sources.len()];
    // MTP experts: layer -> expert -> proj -> part.
    type Parts = BTreeMap<String, Src>;
    let mut mtp: BTreeMap<u64, BTreeMap<u64, BTreeMap<String, Parts>>> = BTreeMap::new();
    let mut dropped_vision = 0;
    for (fi, (_, h)) in sources.iter().enumerate() {
        for t in &h.tensors {
            let name = t.name.as_str();
            if consumed.contains(name) {
                continue;
            }
            let src = || Src {
                file: fi,
                info: t.clone(),
            };
            if name.starts_with("model.visual.") {
                dropped_vision += 1;
                continue;
            }
            if let Some((layer, expert, proj, part)) = mtp_expert(name) {
                mtp.entry(layer)
                    .or_default()
                    .entry(expert)
                    .or_default()
                    .entry(proj.to_string())
                    .or_default()
                    .insert(part.to_string(), src());
                continue;
            }
            if let Some(base) = name.strip_suffix(FIXED) {
                let ex_name = format!("{base}{EXCEPTIONS}");
                let ex = by_name.get(ex_name.as_str()).ok_or_else(|| {
                    reject("convert_csf", format!("{name}: no {EXCEPTIONS} stream"))
                })?;
                want(t, Dtype::U8, 2, "CSF fixed stream")?;
                want(&ex.info, Dtype::U32, 1, "CSF exception stream")?;
                let (rows, cols) = csf_geometry(&t.shape)?;
                consumed.insert(by_name.get_key_value(ex_name.as_str()).unwrap().0);
                per_file[fi].push(OutTensor {
                    name: base.to_string(),
                    dtype: Dtype::F8E4m3,
                    shape: vec![rows as u64, cols as u64],
                    op: Op::Csf {
                        fixed: src(),
                        exceptions: ex.clone(),
                    },
                });
                continue;
            }
            if let Some(base) = name.strip_suffix(EXCEPTIONS) {
                if !by_name.contains_key(format!("{base}{FIXED}").as_str()) {
                    return Err(reject("convert_csf", format!("{name}: no {FIXED} stream")));
                }
                continue; // emitted with its fixed stream
            }
            if t.dtype == Dtype::F8E4m3 && name.ends_with(".weight") {
                let scale_name = format!("{name}_scale");
                if let Some(scale) = by_name
                    .get(scale_name.as_str())
                    .filter(|s| s.info.dtype == Dtype::U8)
                {
                    want(t, Dtype::F8E4m3, 2, "MXFP8 weight")?;
                    let (o, i) = (t.shape[0], t.shape[1]);
                    if i % 32 != 0 {
                        return Err(reject(
                            "convert_mxfp8",
                            format!("{name}: in-features {i} is not a multiple of 32"),
                        ));
                    }
                    if scale.info.shape != [o, i / 32] {
                        return Err(reject(
                            "convert_mxfp8",
                            format!(
                                "{name}: scale shape {:?} is not [{o}, {i}/32]",
                                scale.info.shape
                            ),
                        ));
                    }
                    consumed.insert(by_name.get_key_value(scale_name.as_str()).unwrap().0);
                    match format {
                        Format::S2 => per_file[fi].push(OutTensor {
                            name: name.to_string(),
                            dtype: Dtype::Bf16,
                            shape: t.shape.clone(),
                            op: Op::Mxfp8 {
                                weight: src(),
                                scale: scale.clone(),
                            },
                        }),
                        Format::S3 => {
                            per_file[fi].push(OutTensor {
                                name: name.to_string(),
                                dtype: Dtype::F8E4m3,
                                shape: t.shape.clone(),
                                op: Op::Mxfp8Weight(src()),
                            });
                            per_file[fi].push(OutTensor {
                                name: scale_name,
                                dtype: Dtype::U8,
                                shape: scale.info.shape.clone(),
                                op: Op::Mxfp8Scale(scale.clone()),
                            });
                        }
                    }
                    continue;
                }
            }
            if t.dtype == Dtype::U8 && name.ends_with(".weight_scale") {
                let weight = &name[..name.len() - "_scale".len()];
                if by_name
                    .get(weight)
                    .is_some_and(|w| w.info.dtype == Dtype::F8E4m3)
                {
                    continue; // MXFP8 scale, emitted with its weight
                }
                return Err(reject(
                    "convert_mxfp8",
                    format!("{name}: U8 scale without an F8_E4M3 weight"),
                ));
            }
            if t.dtype == Dtype::F32 && is_norm_weight(name) {
                per_file[fi].push(OutTensor {
                    name: name.to_string(),
                    dtype: Dtype::Bf16,
                    shape: t.shape.clone(),
                    op: Op::NormRne(src()),
                });
                continue;
            }
            per_file[fi].push(OutTensor {
                name: name.to_string(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                op: Op::Copy(src()),
            });
        }
    }
    let mut files: Vec<OutFile> = sources
        .iter()
        .zip(per_file)
        .filter(|(_, t)| !t.is_empty())
        .map(|((name, _), tensors)| OutFile {
            name: format!("cere-{name}"),
            tensors,
        })
        .collect();
    let mut mtp_tensors = Vec::new();
    for (layer, experts) in mtp {
        let n = experts.len() as u64;
        if !experts.keys().copied().eq(0..n) {
            return Err(reject(
                "convert_plan",
                format!("mtp layer {layer}: experts are not 0..{n}"),
            ));
        }
        let mut gate_up = Vec::new();
        let mut down = Vec::new();
        let mut shapes: Option<[(u64, u64); 3]> = None;
        for (e, mut projs) in experts {
            let mut take = |proj: &str| -> Result<Nvfp4Src, Error> {
                let mut parts = projs.remove(proj).unwrap_or_default();
                let mut part = |p: &str| {
                    parts.remove(p).ok_or_else(|| {
                        reject(
                            "convert_plan",
                            format!("mtp layer {layer} expert {e}: no {proj}_proj.{p}"),
                        )
                    })
                };
                Ok(Nvfp4Src {
                    weight: part("weight")?,
                    scale: part("weight_scale")?,
                    global: part("weight_scale_2")?,
                })
            };
            let (g, u, d) = (take("gate")?, take("up")?, take("down")?);
            let s = [nvfp4_shape(&g)?, nvfp4_shape(&u)?, nvfp4_shape(&d)?];
            if s[0] != s[1] || s[2] != (s[0].1, s[0].0) || shapes.is_some_and(|p| p != s) {
                return Err(reject(
                    "convert_plan",
                    format!("mtp layer {layer} expert {e}: projection shapes {s:?} disagree"),
                ));
            }
            shapes = Some(s);
            gate_up.push((g, u));
            down.push(d);
        }
        let [(i, h), _, _] = shapes.expect("at least one expert");
        let prefix = format!("mtp.layers.{layer}.mlp.experts");
        match format {
            Format::S2 => {
                mtp_tensors.push(OutTensor {
                    name: format!("{prefix}.gate_up_proj"),
                    dtype: Dtype::Bf16,
                    shape: vec![n, 2 * i, h],
                    op: Op::MtpGateUp(gate_up),
                });
                mtp_tensors.push(OutTensor {
                    name: format!("{prefix}.down_proj"),
                    dtype: Dtype::Bf16,
                    shape: vec![n, h, i],
                    op: Op::MtpDown(down),
                });
            }
            Format::S3 => {
                let gate_up: Vec<Vec<Nvfp4Src>> =
                    gate_up.into_iter().map(|(g, u)| vec![g, u]).collect();
                let down: Vec<Vec<Nvfp4Src>> = down.into_iter().map(|d| vec![d]).collect();
                let pieces = [
                    (
                        "gate_up_proj",
                        &gate_up,
                        MtpPart::Weight,
                        Dtype::U8,
                        vec![n, 2 * i, h / 2],
                    ),
                    (
                        "gate_up_proj_scale",
                        &gate_up,
                        MtpPart::Scale,
                        Dtype::F8E4m3,
                        vec![n, 2 * i, h / 16],
                    ),
                    (
                        "gate_up_proj_scale_2",
                        &gate_up,
                        MtpPart::Global,
                        Dtype::F32,
                        vec![n, 2],
                    ),
                    (
                        "down_proj",
                        &down,
                        MtpPart::Weight,
                        Dtype::U8,
                        vec![n, h, i / 2],
                    ),
                    (
                        "down_proj_scale",
                        &down,
                        MtpPart::Scale,
                        Dtype::F8E4m3,
                        vec![n, h, i / 16],
                    ),
                    (
                        "down_proj_scale_2",
                        &down,
                        MtpPart::Global,
                        Dtype::F32,
                        vec![n],
                    ),
                ];
                for (suffix, experts, part, dtype, shape) in pieces {
                    mtp_tensors.push(OutTensor {
                        name: format!("{prefix}.{suffix}"),
                        dtype,
                        shape,
                        op: Op::MtpNvfp4 {
                            part,
                            experts: experts.clone(),
                        },
                    });
                }
            }
        }
    }
    if !mtp_tensors.is_empty() {
        if files.iter().any(|f| f.name == MTP_FILE) {
            return Err(reject(
                "convert_plan",
                format!("source shard name collides with {MTP_FILE}"),
            ));
        }
        files.push(OutFile {
            name: MTP_FILE.to_string(),
            tensors: mtp_tensors,
        });
    }
    Ok(Plan {
        files,
        dropped_vision,
    })
}

// ---------------------------------------------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------------------------------------------

/// The safetensors header bytes (length prefix, JSON padded with spaces to a multiple of 8) for these tensors in order.
pub fn header_bytes(tensors: &[OutTensor]) -> Vec<u8> {
    let mut obj = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut off = 0u64;
    for t in tensors {
        let end = off + t.bytes();
        obj.push(',');
        obj.push_str(&serde_json::to_string(&t.name).expect("string"));
        obj.push_str(&format!(
            ":{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{off},{end}]}}",
            t.dtype.name(),
            serde_json::to_string(&t.shape).expect("shape")
        ));
        off = end;
    }
    obj.push('}');
    while (8 + obj.len()) % 8 != 0 {
        obj.push(' ');
    }
    let mut out = (obj.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(obj.as_bytes());
    out
}

#[derive(Debug, Default, Clone)]
pub struct Counts {
    pub tensors: BTreeMap<&'static str, u64>,
    pub mxfp8_values: u64,
    pub csf_scales: u64,
    pub csf_exceptions: u64,
    pub norm_values: u64,
    pub norm_rne_inexact: u64,
    pub mtp_values: u64,
    pub mtp_rne_inexact: u64,
    pub copied_bytes: u64,
}

impl Counts {
    fn add(&mut self, o: &Counts) {
        for (k, v) in &o.tensors {
            *self.tensors.entry(k).or_default() += v;
        }
        self.mxfp8_values += o.mxfp8_values;
        self.csf_scales += o.csf_scales;
        self.csf_exceptions += o.csf_exceptions;
        self.norm_values += o.norm_values;
        self.norm_rne_inexact += o.norm_rne_inexact;
        self.mtp_values += o.mtp_values;
        self.mtp_rne_inexact += o.mtp_rne_inexact;
        self.copied_bytes += o.copied_bytes;
    }

    fn to_json(&self) -> Value {
        json!({
            "tensors_by_rule": self.tensors,
            "mxfp8_values": self.mxfp8_values,
            "csf_scales": self.csf_scales,
            "csf_exceptions": self.csf_exceptions,
            "norm_values": self.norm_values,
            "norm_rne_inexact": self.norm_rne_inexact,
            "mtp_values": self.mtp_values,
            "mtp_rne_inexact": self.mtp_rne_inexact,
            "copied_bytes": self.copied_bytes,
        })
    }
}

struct Sink {
    file: std::io::BufWriter<File>,
    hash: Sha256,
    bytes: u64,
    name: String,
}

impl Sink {
    fn put(&mut self, b: &[u8]) -> Result<(), Error> {
        self.file
            .write_all(b)
            .map_err(|e| Error::io(&self.name, e))?;
        self.hash.update(b);
        self.bytes += b.len() as u64;
        Ok(())
    }
}

fn read_src(files: &[OpenFile], s: &Src) -> Result<Vec<u8>, Error> {
    let mut buf = vec![0u8; s.info.bytes as usize];
    files[s.file]
        .read_exact_at(&mut buf, s.info.offset)
        .map_err(|e| Error::io(&s.info.name, e))?;
    Ok(buf)
}

/// One NVFP4 projection: (packed weight bytes, its S2 BF16 bytes); the dequant is also S3's validity check.
fn nvfp4_bf16(
    files: &[OpenFile],
    p: &Nvfp4Src,
    c: &mut Counts,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let (rows, cols) = nvfp4_shape(p)?;
    let g = read_src(files, &p.global)?;
    let global = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
    let packed = read_src(files, &p.weight)?;
    let mut out = Vec::with_capacity((rows * cols * 2) as usize);
    c.mtp_rne_inexact += nvfp4_to_bf16(
        &packed,
        &read_src(files, &p.scale)?,
        global,
        rows as usize,
        cols as usize,
        &mut out,
    )
    .map_err(|e| reject(e.class, format!("{}: {}", p.weight.info.name, e.message)))?;
    c.mtp_values += rows * cols;
    Ok((packed, out))
}

fn write_nvfp4(
    files: &[OpenFile],
    p: &Nvfp4Src,
    sink: &mut Sink,
    c: &mut Counts,
) -> Result<(), Error> {
    sink.put(&nvfp4_bf16(files, p, c)?.1)
}

fn write_tensor(
    files: &[OpenFile],
    t: &OutTensor,
    sink: &mut Sink,
    c: &mut Counts,
) -> Result<(), Error> {
    let start = sink.bytes;
    match &t.op {
        Op::Copy(s) => {
            let src = &files[s.file];
            let mut buf = vec![0u8; COPY_CHUNK.min(s.info.bytes as usize)];
            let mut done = 0u64;
            while done < s.info.bytes {
                let n = (s.info.bytes - done).min(buf.len() as u64) as usize;
                src.read_exact_at(&mut buf[..n], s.info.offset + done)
                    .map_err(|e| Error::io(&s.info.name, e))?;
                sink.put(&buf[..n])?;
                done += n as u64;
            }
            c.copied_bytes += s.info.bytes;
        }
        Op::Mxfp8Weight(s) => {
            let w = read_src(files, s)?;
            check_mxfp8_weight(&w)
                .map_err(|e| reject(e.class, format!("{}: {}", t.name, e.message)))?;
            c.mxfp8_values += w.len() as u64;
            sink.put(&w)?;
        }
        Op::Mxfp8Scale(s) => {
            let b = read_src(files, s)?;
            check_mxfp8_scales(&b)
                .map_err(|e| reject(e.class, format!("{}: {}", t.name, e.message)))?;
            sink.put(&b)?;
        }
        Op::MtpNvfp4 { part, experts } => {
            for p in experts.iter().flatten() {
                match part {
                    MtpPart::Weight => sink.put(&nvfp4_bf16(files, p, c)?.0)?,
                    MtpPart::Scale => sink.put(&read_src(files, &p.scale)?)?,
                    MtpPart::Global => sink.put(&read_src(files, &p.global)?)?,
                }
            }
        }
        Op::Mxfp8 { weight, scale } => {
            let (o, i) = (t.shape[0] as usize, t.shape[1] as usize);
            let out = mxfp8_to_bf16(&read_src(files, weight)?, &read_src(files, scale)?, o, i)
                .map_err(|e| reject(e.class, format!("{}: {}", t.name, e.message)))?;
            c.mxfp8_values += (o * i) as u64;
            sink.put(&out)?;
        }
        Op::Csf { fixed, exceptions } => {
            let (rows, cols) = csf_geometry(&fixed.info.shape)?;
            let words: Vec<u32> = read_src(files, exceptions)?
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let plane = csf_expand(&read_src(files, fixed)?, &words, rows, cols)
                .map_err(|e| reject(e.class, format!("{}: {}", t.name, e.message)))?;
            c.csf_scales += plane.len() as u64;
            c.csf_exceptions += words.len() as u64;
            sink.put(&plane)?;
        }
        Op::NormRne(s) => {
            let (out, inexact) = f32_to_bf16_rne(&read_src(files, s)?);
            c.norm_values += s.info.bytes / 4;
            c.norm_rne_inexact += inexact;
            sink.put(&out)?;
        }
        Op::MtpGateUp(experts) => {
            for (g, u) in experts {
                write_nvfp4(files, g, sink, c)?;
                write_nvfp4(files, u, sink, c)?;
            }
        }
        Op::MtpDown(experts) => {
            for d in experts {
                write_nvfp4(files, d, sink, c)?;
            }
        }
    }
    if sink.bytes - start != t.bytes() {
        return Err(Error::io(
            &t.name,
            "wrote a different byte count than the header declares",
        ));
    }
    *c.tensors.entry(t.op.rule()).or_default() += 1;
    Ok(())
}

/// Writes one output file; returns (bytes, sha256).
fn write_file(
    files: &[OpenFile],
    out_dir: &Path,
    f: &OutFile,
    c: &mut Counts,
) -> Result<(u64, String), Error> {
    let path = out_dir.join(&f.name);
    let file = File::options()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| Error::io(path.display(), e))?;
    let mut sink = Sink {
        file: std::io::BufWriter::with_capacity(8 << 20, file),
        hash: Sha256::new(),
        bytes: 0,
        name: f.name.clone(),
    };
    sink.put(&header_bytes(&f.tensors))?;
    for t in &f.tensors {
        write_tensor(files, t, &mut sink, c)?;
    }
    let file = sink
        .file
        .into_inner()
        .map_err(|e| Error::io(&f.name, e.error()))?;
    file.sync_all().map_err(|e| Error::io(&f.name, e))?;
    Ok((sink.bytes, hex(&sink.hash.finalize())))
}

pub struct Options {
    pub src: PathBuf,
    pub out: PathBuf,
    pub converter_git_sha: String,
    pub threads: usize,
    pub limits: Limits,
    pub format: Format,
}

/// The HF tree directory `cerec prepare` reads (`source.rs`).
pub const TREES_DIR: &str = ".cache/huggingface/trees";

fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut f = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| Error::io(path.display(), e))?;
    f.write_all(bytes)
        .map_err(|e| Error::io(path.display(), e))?;
    f.sync_all().map_err(|e| Error::io(path.display(), e))
}

fn read_small(dir: &Dir, rel: &str) -> Result<Vec<u8>, Error> {
    let open = dir.open_file(rel)?;
    crate::source::read_all(&open, rel)
}

/// Runs the conversion; returns the report (also the basis of `CONVERSION.json`).
pub fn run(opts: &Options) -> Result<Value, Error> {
    let started = Instant::now();
    if !is_lower_hex(&opts.converter_git_sha, 40) {
        return Err(reject(
            "usage",
            "--converter-git-sha must be 40 lowercase hex",
        ));
    }
    let src_path = std::fs::canonicalize(&opts.src)
        .map_err(|e| reject("source", format!("{}: {e}", opts.src.display())))?;
    let dir = Dir::open(&src_path)
        .map_err(|e| reject("source", format!("{}: {e}", src_path.display())))?;
    // The source revision is the single HF tree file's name, as `cerec prepare` reads it.
    let trees = dir.open_dir(TREES_DIR)?;
    let tree_names: Vec<String> = trees
        .entries()
        .map_err(|e| Error::io(TREES_DIR, e))?
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.ends_with(".json"))
        .collect();
    let [tree_name] = tree_names.as_slice() else {
        return Err(reject(
            "source_tree",
            format!("need exactly one {TREES_DIR}/<revision>.json"),
        ));
    };
    let revision = tree_name.trim_end_matches(".json").to_string();
    if !is_lower_hex(&revision, 40) {
        return Err(reject(
            "source_tree",
            format!("{tree_name}: not a 40-hex revision"),
        ));
    }
    let tree_rel = format!("{TREES_DIR}/{tree_name}");
    let src_tree_bytes = read_small(&dir, &tree_rel)?;
    let src_tree: Value = serde_json::from_slice(&src_tree_bytes)
        .map_err(|e| reject("source_tree", format!("{tree_rel}: {e}")))?;
    match read_small(&dir, "DOWNLOAD_PIN.json") {
        Ok(pin) => {
            let v: Value = serde_json::from_slice(&pin)
                .map_err(|e| reject("source", format!("DOWNLOAD_PIN.json: {e}")))?;
            if v.get("revision").and_then(Value::as_str) != Some(revision.as_str()) {
                return Err(reject(
                    "source",
                    "DOWNLOAD_PIN.json revision differs from the HF tree revision",
                ));
            }
            if v.get("bad")
                .and_then(Value::as_array)
                .is_none_or(|b| !b.is_empty())
            {
                return Err(reject("source", "DOWNLOAD_PIN.json lists bad files"));
            }
        }
        Err(e) if e.class == "missing_file" => {}
        Err(e) => return Err(e),
    }
    let index_bytes = read_small(&dir, INDEX_FILE)?;
    let index: Value = serde_json::from_slice(&index_bytes)
        .map_err(|e| reject("source", format!("{INDEX_FILE}: {e}")))?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| reject("source", format!("{INDEX_FILE}: no weight_map")))?;
    let shard_paths: BTreeSet<&str> = weight_map
        .values()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| reject("source", "weight_map value is not a string"))
        })
        .collect::<Result<_, _>>()?;
    let mut opened = Vec::new();
    let mut sources = Vec::new();
    let mut basenames = BTreeSet::new();
    for rel in &shard_paths {
        let open = dir.open_file(rel)?;
        let (header, _) = safetensors::read(&open, open.stat.size, &opts.limits)
            .map_err(Error::from)
            .map_err(|e| reject(e.class, format!("{rel}: {}", e.message)))?;
        let base = rel.rsplit('/').next().unwrap_or(rel).to_string();
        if !basenames.insert(base.clone()) {
            return Err(reject("source", format!("two shards named {base}")));
        }
        for t in &header.tensors {
            if weight_map.get(&t.name).and_then(Value::as_str) != Some(*rel) {
                return Err(reject(
                    "source",
                    format!("{rel}: tensor {} is not mapped to it by the index", t.name),
                ));
            }
        }
        opened.push(open);
        sources.push((base, header));
    }
    let total: usize = sources.iter().map(|(_, h)| h.tensors.len()).sum();
    if total != weight_map.len() {
        return Err(reject(
            "source",
            format!(
                "index maps {} tensors, shards hold {total}",
                weight_map.len()
            ),
        ));
    }
    let plan = plan(&sources, opts.format)?;

    std::fs::create_dir(&opts.out).map_err(|e| Error::io(opts.out.display(), e))?;
    let out = std::fs::canonicalize(&opts.out).map_err(|e| Error::io(opts.out.display(), e))?;
    if out.starts_with(&src_path) || src_path.starts_with(&out) {
        return Err(reject("unsafe_path", "--out overlaps SRC"));
    }

    // Largest files first; each worker takes the next file.
    let mut order: Vec<usize> = (0..plan.files.len()).collect();
    order.sort_by_key(|&i| {
        std::cmp::Reverse(
            plan.files[i]
                .tensors
                .iter()
                .map(OutTensor::bytes)
                .sum::<u64>(),
        )
    });
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    type Done = (usize, Result<((u64, String), Counts), Error>);
    let done: Vec<Done> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..opts.threads.max(1))
            .map(|_| {
                s.spawn(|| {
                    let mut mine: Vec<Done> = Vec::new();
                    while !stop.load(Ordering::Relaxed) {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&i) = order.get(k) else { break };
                        let mut c = Counts::default();
                        let r = write_file(&opened, &out, &plan.files[i], &mut c).map(|r| (r, c));
                        if r.is_err() {
                            stop.store(true, Ordering::Relaxed);
                        }
                        mine.push((i, r));
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("converter worker panicked"))
            .collect()
    });
    let mut counts = Counts::default();
    let mut results: Vec<Option<(u64, String)>> = vec![None; plan.files.len()];
    for (i, r) in done {
        let (r, c) = r?;
        counts.add(&c);
        results[i] = Some(r);
    }

    let mut written: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut weight_map_out = Map::new();
    let mut total_size = 0u64;
    for (f, r) in plan.files.iter().zip(results) {
        written.insert(f.name.clone(), r.expect("every file written"));
        for t in &f.tensors {
            weight_map_out.insert(t.name.clone(), Value::String(f.name.clone()));
            total_size += t.bytes();
        }
    }
    let index_out = serde_json::to_vec_pretty(&json!({
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }))
    .map_err(|e| Error::io(INDEX_FILE, e))?;
    write_new(&out.join(INDEX_FILE), &index_out)?;
    written.insert(
        INDEX_FILE.into(),
        (index_out.len() as u64, crate::sha256_hex(&index_out)),
    );
    for name in COPIED_FILES {
        let bytes = match read_small(&dir, name) {
            Ok(b) => b,
            Err(e) if e.class == "missing_file" && name != "config.json" => continue,
            Err(e) => return Err(e),
        };
        write_new(&out.join(name), &bytes)?;
        written.insert(name.into(), (bytes.len() as u64, crate::sha256_hex(&bytes)));
    }

    // The source's HF tree, minus the entries of files this converter rewrote (their upstream digests no longer
    // apply; prepare then counts them `not_in_tree`). Byte-copied files keep their upstream entries.
    let mut tree = src_tree;
    let tree_files = tree
        .get_mut("files")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| reject("source_tree", format!("{tree_rel}: no files object")))?;
    let rewritten: Vec<String> = written
        .keys()
        .filter(|n| !COPIED_FILES.contains(&n.as_str()) && tree_files.contains_key(n.as_str()))
        .cloned()
        .collect();
    for n in &rewritten {
        tree_files.remove(n);
    }
    let tree_bytes = serde_json::to_vec_pretty(&tree).map_err(|e| Error::io(&tree_rel, e))?;
    std::fs::create_dir_all(out.join(TREES_DIR)).map_err(|e| Error::io(TREES_DIR, e))?;
    write_new(&out.join(&tree_rel), &tree_bytes)?;

    let files_json: Map<String, Value> = written
        .iter()
        .map(|(n, (b, s))| (n.clone(), json!({"bytes": b, "sha256": s})))
        .collect();
    let mut conversion = json!({
        "schema": "cere-conversion/1",
        "spec": "tasks/fp4local-spec.md D2-D10",
        "source": {
            "path": src_path.display().to_string(),
            "revision": revision,
            "tree_sha256": crate::sha256_hex(&src_tree_bytes),
            "index_sha256": crate::sha256_hex(&index_bytes),
            "tensors": weight_map.len(),
        },
        "converter": {
            "tool": "cerec convert",
            "git_sha": opts.converter_git_sha,
            "version": env!("CARGO_PKG_VERSION"),
        },
        "rules": {
            "vision": "dropped (D10)",
            "mxfp8": "F8_E4M3 x 2^(UE8M0-127) -> BF16, exact (D4)",
            "csf": "NVFP4-CSF -> plain E4M3, canonical streams only, byte-exact (D5)",
            "mtp_experts": "NVFP4 -> BF16: (E2M1 x E4M3) x global in F32, RNE; [gate; up] fused on dim 1 (D7)",
            "f32_norms": "RNE to BF16 (D6)",
            "ple": "NVFP4 table, per-shard E4M3 scales and F32 global copied (D8)",
            "other": "byte copy; config.json keeps the source quantization_config",
            "tree": "the source HF tree without the entries of rewritten files",
        },
        "counts": counts.to_json(),
        "dropped_vision_tensors": plan.dropped_vision,
        "output_tensors": weight_map_out.len(),
        "tree": {"path": tree_rel, "sha256": crate::sha256_hex(&tree_bytes), "dropped_entries": rewritten},
        "files": files_json,
    });
    if opts.format == Format::S3 {
        conversion["format"] = json!("s3");
        conversion["spec"] = json!("tasks/fp4local-spec.md D2-D10, S3 (FL-30)");
        conversion["rules"]["mxfp8"] = json!(
            "F8_E4M3 weight + UE8M0 U8 scale kept byte for byte; canonical only: i % 32 == 0, scales in [10, 246], \
             no E4M3 NaN byte (S3)"
        );
        conversion["rules"]["mtp_experts"] = json!(
            "NVFP4 W4A16 kept: packed E2M1, E4M3 block scales and F32 globals fused [gate; up] on dim 1, globals \
             [E, 2] / [E]; each piece checked to dequantize under the S2 rule (S3)"
        );
    }
    let conv_bytes =
        serde_json::to_vec_pretty(&conversion).map_err(|e| Error::io(CONVERSION_FILE, e))?;
    write_new(&out.join(CONVERSION_FILE), &conv_bytes)?;
    let conv_sha = crate::sha256_hex(&conv_bytes);

    let bytes: u64 = written.values().map(|(b, _)| b).sum::<u64>() + conv_bytes.len() as u64;
    Ok(json!({
        "format": opts.format.name(),
        "out": out.display().to_string(),
        "conversion_sha256": conv_sha,
        "tree_sha256": crate::sha256_hex(&tree_bytes),
        "files": written.len() + 1,
        "bytes": bytes,
        "output_tensors": weight_map_out.len(),
        "dropped_vision_tensors": plan.dropped_vision,
        "counts": counts.to_json(),
        "wall_ms": u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    }))
}
