//! R1 (m21 design note Q15, Q16): every `header` case of the shared invalid table is rejected with its class, and
//! the fixture shards parse to the same tensor list as the Python reference (both dumped through canon_json).
mod support;

use cere_compiler::safetensors::{parse_file_bytes, Header};
use cere_compiler::{canon_json, Limits};
use serde_json::{json, Value};
use support::*;

fn table() -> Value {
    serde_json::from_slice(&std::fs::read(fixture("artifact_invalid.json")).unwrap()).unwrap()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// `u64le(n) ‖ header ‖ zeros(data_len)` of a `header` case.
fn case_file(case: &Value) -> Vec<u8> {
    let header = match (case.get("header"), case.get("header_hex")) {
        (Some(Value::String(s)), None) => s.as_bytes().to_vec(),
        (None, Some(Value::String(h))) => unhex(h),
        _ => panic!("case {} has neither header nor header_hex", case["id"]),
    };
    let n = case
        .get("n")
        .map_or(header.len() as u64, |v| v.as_u64().unwrap());
    let data_len = case.get("data_len").map_or(0, |v| v.as_u64().unwrap()) as usize;
    let mut out = n.to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.resize(out.len() + data_len, 0);
    out
}

fn case_limits(case: &Value) -> Limits {
    case.get("limits")
        .map_or_else(Limits::default, |l| Limits::with_overrides(l).unwrap())
}

#[test]
fn every_header_case_is_rejected_with_its_class() {
    let t = table();
    let classes: Vec<&str> = t["error_classes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    let cases: Vec<&Value> = t["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["tag"] == "header")
        .collect();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for case in cases {
        let want = case["error"].as_str().unwrap();
        assert!(
            classes.contains(&want),
            "{}: unknown class {want}",
            case["id"]
        );
        match parse_file_bytes(&case_file(case), &case_limits(case)) {
            Ok(_) => failures.push(format!("{}: accepted, want {want}", case["id"])),
            Err(e) if e.class() != want => {
                failures.push(format!("{}: got {}, want {want}", case["id"], e.class()))
            }
            Err(_) => {}
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The cross-language dump: `{header_bytes, header_sha256, metadata, tensors:[{name, dtype, shape, offset, bytes,
/// element_aligned}]}`, tensors in data-offset order.
fn dump(h: &Header) -> Vec<u8> {
    let tensors: Vec<Value> = h
        .tensors
        .iter()
        .map(|t| {
            json!({"name": t.name, "dtype": t.dtype.name(), "shape": t.shape, "offset": t.offset, "bytes": t.bytes,
                   "element_aligned": t.element_aligned})
        })
        .collect();
    canon_json::to_vec(&json!({
        "header_bytes": h.header_bytes, "header_sha256": h.header_sha256, "metadata": h.metadata, "tensors": tensors,
    }))
    .unwrap()
}

#[test]
fn fixture_shards_parse_like_python() {
    let mut shards: Vec<_> = std::fs::read_dir(model_src())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    assert_eq!(shards.len(), 3);
    for shard in shards {
        let h = parse_file_bytes(&std::fs::read(&shard).unwrap(), &Limits::default()).unwrap();
        let py = ok(run(python("model_inventory.py")
            .arg("--dump-header")
            .arg(&shard)))
        .stdout;
        assert_eq!(
            String::from_utf8_lossy(&dump(&h)),
            String::from_utf8_lossy(&py),
            "{}",
            shard.display()
        );
    }
}
