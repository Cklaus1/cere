//! Bounded safetensors header parser (m21 design note Q15, Q16).
//!
//! A file is `u64 n (LE)`, `n` bytes of a JSON object, then the data region. Every bound is checked before any
//! allocation sized by the file, arithmetic is checked, and the JSON parser rejects duplicate keys.

use crate::canon_json::{self, JsonError};
use crate::Limits;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt;

/// Most dims a shape may have (Q4).
pub const MAX_DIMS: usize = 16;

/// Storage dtypes, by their safetensors names (Q4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F8E4m3,
    F8E5m2,
    Bool,
}

impl Dtype {
    const ALL: [(Dtype, &'static str, u64); 15] = [
        (Dtype::Bf16, "BF16", 2),
        (Dtype::F16, "F16", 2),
        (Dtype::F32, "F32", 4),
        (Dtype::F64, "F64", 8),
        (Dtype::I8, "I8", 1),
        (Dtype::I16, "I16", 2),
        (Dtype::I32, "I32", 4),
        (Dtype::I64, "I64", 8),
        (Dtype::U8, "U8", 1),
        (Dtype::U16, "U16", 2),
        (Dtype::U32, "U32", 4),
        (Dtype::U64, "U64", 8),
        (Dtype::F8E4m3, "F8_E4M3", 1),
        (Dtype::F8E5m2, "F8_E5M2", 1),
        (Dtype::Bool, "BOOL", 1),
    ];

    pub fn parse(name: &str) -> Option<Dtype> {
        Self::ALL
            .iter()
            .find(|(_, n, _)| *n == name)
            .map(|(d, _, _)| *d)
    }

    pub fn name(self) -> &'static str {
        Self::ALL[self as usize].1
    }

    pub fn itemsize(self) -> u64 {
        Self::ALL[self as usize].2
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why a header was rejected. Variants are the `header` error classes of `tests/fixtures/artifact_invalid.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    /// `n` < 2, `n` > `max_header_bytes`, or `8 + n` beyond the file.
    Bounds,
    Utf8,
    /// Invalid JSON, or JSON that is not an object.
    Json(String),
    DuplicateKey,
    /// `__metadata__` other than a string→string object.
    Metadata,
    /// An entry that is not an object with exactly `dtype`/`shape`/`data_offsets`, or malformed `data_offsets`.
    TensorFields,
    Dtype,
    /// Not a list of 0..=16 integers, each ≥ 1.
    Shape,
    TooManyTensors,
    /// `prod(shape) × itemsize` overflows or differs from `end - begin`.
    ByteSize,
    /// `end < begin`, overlap, gap, a start ≠ 0, or a last end ≠ the data region length.
    Offsets,
}

impl HeaderError {
    /// The shared error class name.
    pub fn class(&self) -> &'static str {
        match self {
            HeaderError::Bounds => "header_bounds",
            HeaderError::Utf8 => "header_utf8",
            HeaderError::Json(_) => "header_json",
            HeaderError::DuplicateKey => "duplicate_key",
            HeaderError::Metadata => "metadata",
            HeaderError::TensorFields => "tensor_fields",
            HeaderError::Dtype => "dtype",
            HeaderError::Shape => "shape",
            HeaderError::TooManyTensors => "too_many_tensors",
            HeaderError::ByteSize => "byte_size",
            HeaderError::Offsets => "offsets",
        }
    }
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderError::Json(e) => write!(f, "safetensors header: {}: {e}", self.class()),
            other => write!(f, "safetensors header: {}", other.class()),
        }
    }
}

impl std::error::Error for HeaderError {}

/// One tensor of a parsed header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// `data_offsets[0]`, relative to the data region.
    pub data_begin: u64,
    /// Absolute file offset (`8 + n + data_begin`).
    pub offset: u64,
    pub bytes: u64,
    /// `offset % itemsize == 0` (Q16: recorded, never rejected).
    pub element_aligned: bool,
}

/// A validated header. `tensors` are in file-offset order and tile the data region exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// `8 + n`.
    pub header_bytes: u64,
    /// Hex SHA-256 of bytes `[0, 8 + n)`.
    pub header_sha256: String,
    pub file_bytes: u64,
    pub metadata: Option<BTreeMap<String, String>>,
    pub tensors: Vec<TensorInfo>,
}

impl Header {
    pub fn data_bytes(&self) -> u64 {
        self.file_bytes - self.header_bytes
    }
}

/// Validates the length prefix against the file size and returns `n`.
pub fn header_len(first8: &[u8], file_size: u64, limits: &Limits) -> Result<u64, HeaderError> {
    let prefix: [u8; 8] = first8
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .ok_or(HeaderError::Bounds)?;
    let n = u64::from_le_bytes(prefix);
    if n < 2 || n > limits.max_header_bytes || n.checked_add(8).is_none_or(|end| end > file_size) {
        return Err(HeaderError::Bounds);
    }
    Ok(n)
}

/// Parses a whole in-memory file.
pub fn parse_file_bytes(bytes: &[u8], limits: &Limits) -> Result<Header, HeaderError> {
    let n = header_len(bytes, bytes.len() as u64, limits)?;
    parse(&bytes[..8 + n as usize], bytes.len() as u64, limits)
}

/// Reads and parses the header of an open file of `file_size` bytes; returns the header and the bytes read.
pub fn read(
    file: &crate::fsio::OpenFile,
    file_size: u64,
    limits: &Limits,
) -> Result<(Header, u64), ReadError> {
    let mut first8 = [0u8; 8];
    if file_size < 8 {
        return Err(ReadError::Header(HeaderError::Bounds));
    }
    file.read_exact_at(&mut first8, 0).map_err(ReadError::Io)?;
    let n = header_len(&first8, file_size, limits).map_err(ReadError::Header)?;
    let mut prefix = vec![0u8; 8 + n as usize];
    file.read_exact_at(&mut prefix, 0).map_err(ReadError::Io)?;
    let header = parse(&prefix, file_size, limits).map_err(ReadError::Header)?;
    Ok((header, 8 + n))
}

/// A header read failure: I/O or a rejected header.
#[derive(Debug)]
pub enum ReadError {
    Io(std::io::Error),
    Header(HeaderError),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Io(e) => write!(f, "reading safetensors header: {e}"),
            ReadError::Header(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for ReadError {}

struct Entry<'a> {
    name: &'a str,
    dtype: Dtype,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}

/// Parses `prefix` = bytes `[0, 8 + n)` of a file of `file_size` bytes, in the shared class order.
pub fn parse(prefix: &[u8], file_size: u64, limits: &Limits) -> Result<Header, HeaderError> {
    let n = header_len(prefix, file_size, limits)?;
    if prefix.len() as u64 != 8 + n {
        return Err(HeaderError::Bounds);
    }
    let header_bytes = 8 + n;
    let data_len = file_size - header_bytes;
    let root = match canon_json::parse_strict_ordered(&prefix[8..]) {
        Ok(Some(entries)) => entries,
        Ok(None) => return Err(HeaderError::Json("not an object".into())),
        Err(JsonError::NotUtf8) => return Err(HeaderError::Utf8),
        Err(JsonError::DuplicateKey(_)) => return Err(HeaderError::DuplicateKey),
        Err(e) => return Err(HeaderError::Json(e.to_string())),
    };
    let mut metadata = None;
    if let Some((_, meta)) = root.iter().find(|(k, _)| k == "__metadata__") {
        metadata = Some(parse_metadata(meta)?);
    }
    let mut entries = Vec::with_capacity(root.len());
    for (name, entry) in root.iter().filter(|(k, _)| k != "__metadata__") {
        entries.push(parse_entry(name, entry)?);
    }
    if entries.len() as u64 > limits.max_tensors_per_file {
        return Err(HeaderError::TooManyTensors);
    }
    for e in &entries {
        if e.end < e.begin {
            return Err(HeaderError::Offsets);
        }
        let expected = e
            .shape
            .iter()
            .try_fold(e.dtype.itemsize(), |acc, &d| acc.checked_mul(d));
        if expected != Some(e.end - e.begin) {
            return Err(HeaderError::ByteSize);
        }
    }
    entries.sort_by_key(|e| (e.begin, e.end));
    let mut cursor = 0u64;
    for e in &entries {
        if e.begin != cursor {
            return Err(HeaderError::Offsets);
        }
        cursor = e.end;
    }
    if cursor != data_len {
        return Err(HeaderError::Offsets);
    }
    let tensors = entries
        .into_iter()
        .map(|e| {
            let offset = header_bytes + e.begin;
            TensorInfo {
                name: e.name.to_string(),
                dtype: e.dtype,
                shape: e.shape,
                data_begin: e.begin,
                offset,
                bytes: e.end - e.begin,
                element_aligned: offset % e.dtype.itemsize() == 0,
            }
        })
        .collect();
    Ok(Header {
        header_bytes,
        header_sha256: crate::sha256_hex(prefix),
        file_bytes: file_size,
        metadata,
        tensors,
    })
}

fn parse_metadata(entry: &Value) -> Result<BTreeMap<String, String>, HeaderError> {
    let map = entry.as_object().ok_or(HeaderError::Metadata)?;
    map.iter()
        .map(|(k, v)| match v {
            Value::String(s) => Ok((k.clone(), s.clone())),
            _ => Err(HeaderError::Metadata),
        })
        .collect()
}

fn parse_entry<'a>(name: &'a str, entry: &Value) -> Result<Entry<'a>, HeaderError> {
    let fields: &Map<String, Value> = entry.as_object().ok_or(HeaderError::TensorFields)?;
    let (Some(dtype), Some(shape), Some(offsets), 3) = (
        fields.get("dtype"),
        fields.get("shape"),
        fields.get("data_offsets"),
        fields.len(),
    ) else {
        return Err(HeaderError::TensorFields);
    };
    let [begin, end] = offsets
        .as_array()
        .and_then(|a| <&[Value; 2]>::try_from(a.as_slice()).ok())
        .and_then(|[b, e]| Some([b.as_u64()?, e.as_u64()?]))
        .ok_or(HeaderError::TensorFields)?;
    let dtype = dtype
        .as_str()
        .and_then(Dtype::parse)
        .ok_or(HeaderError::Dtype)?;
    let dims = shape.as_array().ok_or(HeaderError::Shape)?;
    if dims.len() > MAX_DIMS {
        return Err(HeaderError::Shape);
    }
    let shape = dims
        .iter()
        .map(|d| d.as_u64().filter(|&d| d >= 1).ok_or(HeaderError::Shape))
        .collect::<Result<Vec<u64>, _>>()?;
    Ok(Entry {
        name,
        dtype,
        shape,
        begin,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data_len: usize) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + data_len, 0xab);
        out
    }

    fn err(bytes: &[u8]) -> &'static str {
        parse_file_bytes(bytes, &Limits::default())
            .unwrap_err()
            .class()
    }

    #[test]
    fn parses_contiguous_tensors_in_offset_order_with_alignment() {
        let h = r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"BF16","shape":[2],"data_offsets":[1,5]},
                    "a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"s":{"dtype":"F32","shape":[],"data_offsets":[5,9]}}  "#;
        let bytes = file(h, 9);
        let hdr = parse_file_bytes(&bytes, &Limits::default()).unwrap();
        assert_eq!(hdr.header_bytes, 8 + h.len() as u64);
        let names: Vec<_> = hdr.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "s"]);
        let b = &hdr.tensors[1];
        assert_eq!(
            (b.dtype, b.bytes, b.offset),
            (Dtype::Bf16, 4, hdr.header_bytes + 1)
        );
        assert_eq!(b.element_aligned, (hdr.header_bytes + 1) % 2 == 0);
        assert_eq!(hdr.tensors[2].shape, Vec::<u64>::new());
        assert_eq!(hdr.metadata.as_ref().unwrap()["format"], "pt");
        assert_eq!(
            hdr.header_sha256,
            crate::sha256_hex(&bytes[..hdr.header_bytes as usize])
        );
        assert_eq!(hdr.data_bytes(), 9);
    }

    #[test]
    fn rejects_length_prefix_bounds_before_reading() {
        assert_eq!(err(&[1, 0, 0, 0, 0, 0, 0, 0, b'{']), "header_bounds");
        assert_eq!(err(&[0, 0, 0, 0, 0, 1, 0, 0]), "header_bounds");
        assert_eq!(err(&[10, 0, 0, 0, 0, 0, 0, 0, b'{', b'}']), "header_bounds");
        assert_eq!(err(&[1, 2, 3]), "header_bounds");
        assert_eq!(err(&u64::MAX.to_le_bytes()), "header_bounds");
        let limits = Limits::default();
        assert_eq!(
            header_len(&((16u64 << 20) + 1).to_le_bytes(), u64::MAX, &limits),
            Err(HeaderError::Bounds)
        );
        assert_eq!(
            header_len(&(16u64 << 20).to_le_bytes(), (16 << 20) + 8, &limits),
            Ok(16 << 20)
        );
    }

    #[test]
    fn rejects_bad_json_shapes() {
        assert_eq!(err(&file("[1, 2]", 0)), "header_json");
        let mut raw = file("{\"a\":1}", 0);
        raw[10] = 0xff;
        assert_eq!(err(&raw), "header_utf8");
        assert_eq!(err(&file("{\"a\":", 0)), "header_json");
        let dup = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        assert_eq!(err(&file(dup, 2)), "duplicate_key");
    }

    #[test]
    fn rejects_bad_tensor_entries() {
        let one = |entry: &str, data| err(&file(&format!("{{\"t\":{entry}}}"), data));
        assert_eq!(
            one(
                r#"{"dtype":"U8","shape":[1],"data_offsets":[0,1],"x":1}"#,
                1
            ),
            "tensor_fields"
        );
        assert_eq!(one(r#"{"dtype":"U8","shape":[1]}"#, 1), "tensor_fields");
        assert_eq!(
            one(r#"{"dtype":"U8","shape":[1],"data_offsets":[0,1,2]}"#, 1),
            "tensor_fields"
        );
        assert_eq!(
            one(r#"{"dtype":"U8","shape":[1],"data_offsets":[-1,1]}"#, 1),
            "tensor_fields"
        );
        assert_eq!(
            one(r#"{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}"#, 1),
            "dtype"
        );
        assert_eq!(
            one(r#"{"dtype":"U8","shape":[0],"data_offsets":[0,0]}"#, 0),
            "shape"
        );
        assert_eq!(
            one(r#"{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}"#, 1),
            "shape"
        );
        let deep = format!(
            r#"{{"dtype":"U8","shape":[{}],"data_offsets":[0,1]}}"#,
            ["1"; 17].join(",")
        );
        assert_eq!(one(&deep, 1), "shape");
        assert_eq!(
            one(
                r#"{"dtype":"F64","shape":[4294967296,4294967296],"data_offsets":[0,1]}"#,
                1
            ),
            "byte_size"
        );
        assert_eq!(
            one(r#"{"dtype":"F32","shape":[1],"data_offsets":[0,2]}"#, 2),
            "byte_size"
        );
        assert_eq!(
            one(r#"{"dtype":"U8","shape":[1],"data_offsets":[1,0]}"#, 1),
            "offsets"
        );
        assert_eq!(
            one(r#"{"dtype":"U8","shape":[2],"data_offsets":[0,2]}"#, 1),
            "offsets"
        );
        assert_eq!(err(&file("{\"__metadata__\":{\"a\":1}}", 0)), "metadata");
        assert_eq!(err(&file("{\"__metadata__\":null}", 0)), "metadata");
    }

    #[test]
    fn rejects_gaps_overlaps_and_short_regions() {
        let two = |a: &str, b: &str, data| {
            err(&file(
                &format!(
                    r#"{{"a":{{"dtype":"U8","shape":[2],"data_offsets":{a}}},"b":{{"dtype":"U8","shape":[2],"data_offsets":{b}}}}}"#
                ),
                data,
            ))
        };
        assert_eq!(two("[0,2]", "[3,5]", 5), "offsets");
        assert_eq!(two("[1,3]", "[3,5]", 5), "offsets");
        assert_eq!(two("[0,2]", "[1,3]", 3), "offsets");
        assert_eq!(two("[0,2]", "[2,4]", 5), "offsets");
        let ok = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#;
        assert!(parse_file_bytes(&file(ok, 4), &Limits::default()).is_ok());
    }

    #[test]
    fn caps_tensor_count_per_file_after_entry_checks() {
        let limits = Limits {
            max_tensors_per_file: 1,
            ..Limits::default()
        };
        let two = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#;
        assert_eq!(
            parse_file_bytes(&file(two, 4), &limits).unwrap_err(),
            HeaderError::TooManyTensors
        );
        let bad_dtype = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"X","shape":[2],"data_offsets":[2,4]}}"#;
        assert_eq!(
            parse_file_bytes(&file(bad_dtype, 4), &limits).unwrap_err(),
            HeaderError::Dtype
        );
    }
}
