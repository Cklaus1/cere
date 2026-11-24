//! WeightLab vectors (weightlab design note "Experiment API", WL-14, track WL-B): the host store behind
//! `POST /v1/cere/research/vectors?name=<name>` and `DELETE /v1/cere/research/vectors/{id}`.
//!
//! The upload body is a safetensors file with exactly one BF16 tensor, shape `[cols]` or `[rows, cols]` (rows in
//! `[1, 4096]`, the replace kind's source rows; a `[cols]` tensor is one row), every value finite and at most
//! [`ffi::RESEARCH_MAX_PLAN_BYTES`] of data (a larger vector could never be part of a loaded plan). The answer is
//! `{vector_id, sha256, rows, cols}`, `sha256` the hex SHA-256 of the whole body (the experiment record's
//! `vector_sha256`). Refusals change nothing: 400 `invalid_research_vector` (body, name, dtype, shape, size or a
//! non-finite value), 409 `research_vector_exists` (the name is taken), 507 `research_vectors_full` (`max_vectors`
//! stored). A plan holds its vectors' [`Arc`] while it is loaded (the executor copies the bytes at plan_load, the
//! store keeps them for the plan's record and reloads), so a delete is 409 `research_vector_busy` while any plan
//! holds one and 404 `research_vector_not_found` for an unknown id. Ids are `vec_<n>`, `n` counting uploads from 1
//! within one `cered` process; vectors are not persisted across restarts.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use axum::http::StatusCode;
use axum::response::Response;
use sha2::{Digest, Sha256};

use crate::api_types::{ApiError, INVALID_REQUEST};
use crate::digest::hex;
use crate::native::ffi;
use crate::server::{api_error, json_response};

/// `POST /v1/cere/research/vectors?name=<name>` after the operator check: 200 `{vector_id, sha256, rows, cols}`.
pub fn post_vector_response(store: &VectorStore, name: Option<&str>, body: &[u8]) -> Response {
    match store.insert(name.unwrap_or(""), body) {
        Ok(info) => json_response(
            StatusCode::OK,
            &serde_json::to_value(info).expect("plain struct"),
        ),
        Err(e) => vector_error(&e),
    }
}

/// `DELETE /v1/cere/research/vectors/{id}` after the operator check: 200 `{vector_id, deleted: true}`.
pub fn delete_vector_response(store: &VectorStore, id: &str) -> Response {
    match store.remove(id) {
        Ok(()) => json_response(
            StatusCode::OK,
            &serde_json::json!({"vector_id": id, "deleted": true}),
        ),
        Err(e) => vector_error(&e),
    }
}

fn vector_error(e: &VectorError) -> Response {
    let (status, code) = e.status_code();
    let param = matches!(e, VectorError::Exists(_)).then(|| "name".to_owned());
    api_error(&ApiError::new(
        status,
        INVALID_REQUEST,
        code,
        e.to_string(),
        param,
    ))
}

/// Most rows a vector may have (the replace kind's source rows, research/patch.h kMaxVectorRows).
pub const MAX_VECTOR_ROWS: u64 = 4096;
/// Longest vector name in bytes.
pub const MAX_NAME_BYTES: usize = 128;

/// One stored vector: BF16 little-endian `[rows, cols]`.
#[derive(Debug)]
pub struct Vector {
    pub id: String,
    pub name: String,
    pub sha256: String,
    pub rows: u32,
    pub cols: u32,
    pub data: Vec<u8>,
}

/// What an upload answers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct VectorInfo {
    pub vector_id: String,
    pub sha256: String,
    pub rows: u32,
    pub cols: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum VectorError {
    Invalid(String),
    Exists(String),
    Full(u32),
    NotFound(String),
    Busy(String),
}

impl VectorError {
    /// HTTP status and error code.
    pub fn status_code(&self) -> (u16, &'static str) {
        match self {
            Self::Invalid(_) => (400, "invalid_research_vector"),
            Self::Exists(_) => (409, "research_vector_exists"),
            Self::Full(_) => (507, "research_vectors_full"),
            Self::NotFound(_) => (404, "research_vector_not_found"),
            Self::Busy(_) => (409, "research_vector_busy"),
        }
    }
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(m) => write!(f, "{m}"),
            Self::Exists(name) => write!(f, "a vector named {name:?} exists"),
            Self::Full(max) => write!(
                f,
                "{max} vectors are stored (max_vectors); delete one first"
            ),
            Self::NotFound(id) => write!(f, "no vector {id:?}"),
            Self::Busy(id) => write!(f, "vector {id:?} is used by a loaded plan"),
        }
    }
}

impl std::error::Error for VectorError {}

#[derive(Default)]
struct Inner {
    by_id: BTreeMap<String, Arc<Vector>>,
    next: u64,
}

/// The vectors of one `cered` process, at most `max_vectors` at a time.
pub struct VectorStore {
    max_vectors: u32,
    inner: Mutex<Inner>,
}

impl VectorStore {
    pub fn new(max_vectors: u32) -> Self {
        Self {
            max_vectors,
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Validates `body` (see the module docs) and stores it under `name`.
    pub fn insert(&self, name: &str, body: &[u8]) -> Result<VectorInfo, VectorError> {
        check_name(name)?;
        let (rows, cols, data) = parse(body)?;
        let sha256 = hex(&Sha256::digest(body));
        let mut inner = self.lock();
        if inner.by_id.values().any(|v| v.name == name) {
            return Err(VectorError::Exists(name.to_owned()));
        }
        if inner.by_id.len() >= self.max_vectors as usize {
            return Err(VectorError::Full(self.max_vectors));
        }
        inner.next += 1;
        let id = format!("vec_{}", inner.next);
        let info = VectorInfo {
            vector_id: id.clone(),
            sha256: sha256.clone(),
            rows,
            cols,
        };
        inner.by_id.insert(
            id.clone(),
            Arc::new(Vector {
                id,
                name: name.to_owned(),
                sha256,
                rows,
                cols,
                data,
            }),
        );
        Ok(info)
    }

    /// The vector `id`; a plan keeps the returned `Arc` while it is loaded, which makes `remove` refuse.
    pub fn get(&self, id: &str) -> Option<Arc<Vector>> {
        self.lock().by_id.get(id).cloned()
    }

    /// Deletes `id`: `NotFound` for an unknown id, `Busy` while a plan holds it (nothing changes).
    pub fn remove(&self, id: &str) -> Result<(), VectorError> {
        let mut inner = self.lock();
        let held = match inner.by_id.get(id) {
            None => return Err(VectorError::NotFound(id.to_owned())),
            Some(v) => Arc::strong_count(v) > 1,
        };
        if held {
            return Err(VectorError::Busy(id.to_owned()));
        }
        inner.by_id.remove(id);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.lock().by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn check_name(name: &str) -> Result<(), VectorError> {
    let ok = !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        Err(VectorError::Invalid(format!(
            "name must be 1..={MAX_NAME_BYTES} bytes of [A-Za-z0-9._-]"
        )))
    }
}

/// `(rows, cols, BF16 bytes)` of an upload body.
fn parse(body: &[u8]) -> Result<(u32, u32, Vec<u8>), VectorError> {
    use cere_compiler::safetensors::Dtype;
    let invalid = |m: String| VectorError::Invalid(m);
    let header =
        cere_compiler::safetensors::parse_file_bytes(body, &cere_compiler::Limits::default())
            .map_err(|e| invalid(format!("not a safetensors file: {e}")))?;
    let [t] = header.tensors.as_slice() else {
        return Err(invalid(format!(
            "expected exactly one tensor, found {}",
            header.tensors.len()
        )));
    };
    if t.dtype != Dtype::Bf16 {
        return Err(invalid(format!(
            "tensor {} is {:?}; expected BF16",
            t.name, t.dtype
        )));
    }
    let (rows, cols) = match t.shape.as_slice() {
        [cols] => (1, *cols),
        [rows, cols] => (*rows, *cols),
        other => {
            return Err(invalid(format!(
                "tensor {} has shape {other:?}; expected [cols] or [rows, cols]",
                t.name
            )))
        }
    };
    if !(1..=MAX_VECTOR_ROWS).contains(&rows) || cols == 0 {
        return Err(invalid(format!(
            "tensor {} has shape {:?}; rows must be in [1, {MAX_VECTOR_ROWS}] and cols >= 1",
            t.name, t.shape
        )));
    }
    if t.bytes > ffi::RESEARCH_MAX_PLAN_BYTES {
        return Err(invalid(format!(
            "tensor {} holds {} bytes; a plan holds at most {}",
            t.name,
            t.bytes,
            ffi::RESEARCH_MAX_PLAN_BYTES
        )));
    }
    let start = t.offset as usize;
    let data = &body[start..start + t.bytes as usize];
    if data
        .chunks_exact(2)
        .any(|c| u16::from_le_bytes([c[0], c[1]]) & 0x7f80 == 0x7f80)
    {
        return Err(invalid(format!("tensor {} has a non-finite value", t.name)));
    }
    // rows <= 4096 and rows * cols * 2 <= RESEARCH_MAX_PLAN_BYTES, so both fit u32.
    Ok((rows as u32, cols as u32, data.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(tensors: &[(&str, &str, &[u64], Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({"dtype": dtype, "shape": shape,
                                   "data_offsets": [offset, offset + bytes.len()]}),
            );
            offset += bytes.len();
            data.extend_from_slice(bytes);
        }
        let mut raw = serde_json::to_vec(&header).unwrap();
        while raw.len() % 8 != 0 {
            raw.push(b' ');
        }
        let mut out = (raw.len() as u64).to_le_bytes().to_vec();
        out.extend(raw);
        out.extend(data);
        out
    }

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn vector(cols: u64) -> Vec<u8> {
        let values: Vec<f32> = (0..cols).map(|i| i as f32 * 0.25 - 1.0).collect();
        file(&[("v", "BF16", &[cols], bf16(&values))])
    }

    #[test]
    fn insert_get_remove() {
        let store = VectorStore::new(4);
        let body = vector(2560);
        let info = store.insert("steer.a", &body).unwrap();
        assert_eq!(info.vector_id, "vec_1");
        assert_eq!((info.rows, info.cols), (1, 2560));
        assert_eq!(info.sha256, hex(&Sha256::digest(&body)));
        let v = store.get("vec_1").unwrap();
        assert_eq!(v.data.len(), 2 * 2560);
        assert_eq!(&v.data[..], &body[body.len() - 2 * 2560..]);
        assert_eq!(v.name, "steer.a");
        drop(v);
        store.remove("vec_1").unwrap();
        assert!(store.get("vec_1").is_none());
        assert!(store.is_empty());
        // Ids are never reused within a process.
        assert_eq!(store.insert("steer.a", &body).unwrap().vector_id, "vec_2");
    }

    #[test]
    fn two_dimensional_vectors_for_replace() {
        let store = VectorStore::new(1);
        let values: Vec<f32> = (0..3 * 512).map(|i| i as f32).collect();
        let body = file(&[("r", "BF16", &[3, 512], bf16(&values))]);
        let info = store.insert("rep", &body).unwrap();
        assert_eq!((info.rows, info.cols), (3, 512));
    }

    #[test]
    fn busy_while_a_plan_holds_it() {
        let store = VectorStore::new(2);
        store.insert("a", &vector(8)).unwrap();
        let held = store.get("vec_1").unwrap();
        let err = store.remove("vec_1").unwrap_err();
        assert_eq!(err.status_code(), (409, "research_vector_busy"));
        assert!(store.get("vec_1").is_some());
        drop(held);
        store.remove("vec_1").unwrap();
        assert_eq!(
            store.remove("vec_1").unwrap_err().status_code(),
            (404, "research_vector_not_found")
        );
    }

    #[test]
    fn responders() {
        let store = VectorStore::new(1);
        let ok = post_vector_response(&store, Some("a"), &vector(8));
        assert_eq!(ok.status(), StatusCode::OK);
        assert_eq!(
            post_vector_response(&store, None, &vector(8)).status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            post_vector_response(&store, Some("a"), &vector(8)).status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            post_vector_response(&store, Some("b"), &vector(8)).status(),
            StatusCode::INSUFFICIENT_STORAGE
        );
        assert_eq!(
            delete_vector_response(&store, "vec_9").status(),
            StatusCode::NOT_FOUND
        );
        let held = store.get("vec_1").unwrap();
        assert_eq!(
            delete_vector_response(&store, "vec_1").status(),
            StatusCode::CONFLICT
        );
        drop(held);
        assert_eq!(
            delete_vector_response(&store, "vec_1").status(),
            StatusCode::OK
        );
    }

    #[test]
    fn exists_and_full() {
        let store = VectorStore::new(2);
        store.insert("a", &vector(8)).unwrap();
        let err = store.insert("a", &vector(16)).unwrap_err();
        assert_eq!(err.status_code(), (409, "research_vector_exists"));
        store.insert("b", &vector(8)).unwrap();
        let err = store.insert("c", &vector(8)).unwrap_err();
        assert_eq!(err.status_code(), (507, "research_vectors_full"));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn refusals_change_nothing() {
        let store = VectorStore::new(8);
        let nan = {
            let mut b = bf16(&[1.0, 2.0, 3.0, 4.0]);
            b[2..4].copy_from_slice(&0x7fc0u16.to_le_bytes());
            b
        };
        let inf = bf16(&[1.0, f32::INFINITY]);
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let too_many_rows = bf16(&vec![0.0; 4097]);
        let too_big_cols = (ffi::RESEARCH_MAX_PLAN_BYTES / 2) + 1;
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("garbage", b"not a safetensors file".to_vec()),
            ("nan", file(&[("v", "BF16", &[4], nan)])),
            ("inf", file(&[("v", "BF16", &[2], inf)])),
            ("f32", file(&[("v", "F32", &[2], f32_bytes)])),
            (
                "two",
                file(&[
                    ("a", "BF16", &[1], bf16(&[1.0])),
                    ("b", "BF16", &[1], bf16(&[1.0])),
                ]),
            ),
            (
                "rank3",
                file(&[("v", "BF16", &[1, 1, 2], bf16(&[1.0, 2.0]))]),
            ),
            ("rows", file(&[("v", "BF16", &[4097, 1], too_many_rows)])),
            ("zero_rows", file(&[("v", "BF16", &[0, 4], Vec::new())])),
            ("zero_cols", file(&[("v", "BF16", &[0], Vec::new())])),
            (
                "too_big",
                file(&[(
                    "v",
                    "BF16",
                    &[too_big_cols],
                    vec![0u8; 2 * too_big_cols as usize],
                )]),
            ),
        ];
        for (name, body) in cases {
            let err = store.insert(name, &body).unwrap_err();
            assert_eq!(
                err.status_code(),
                (400, "invalid_research_vector"),
                "{name}: {err}"
            );
        }
        for bad in ["", "a b", "a/b", &"x".repeat(MAX_NAME_BYTES + 1)] {
            let err = store.insert(bad, &vector(4)).unwrap_err();
            assert_eq!(
                err.status_code(),
                (400, "invalid_research_vector"),
                "{bad:?}"
            );
        }
        assert!(store.is_empty());
        // The largest legal vector loads.
        let cols = ffi::RESEARCH_MAX_PLAN_BYTES / 2;
        let body = file(&[("v", "BF16", &[cols], vec![0u8; 2 * cols as usize])]);
        assert_eq!(store.insert("max", &body).unwrap().cols as u64, cols);
    }
}
