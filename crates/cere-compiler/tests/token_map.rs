//! R1 (m21 design note Q10): every `token_map` case of the shared invalid table is rejected with its class, and the
//! fixture map converts to the same bytes as `scripts/token_map.py`.
mod support;

use cere_compiler::token_map::{convert, to_u32le, vocab_size_from_config};
use cere_compiler::Limits;
use serde_json::Value;
use support::*;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn every_token_map_case_is_rejected_with_its_class() {
    let t: Value =
        serde_json::from_slice(&std::fs::read(fixture("artifact_invalid.json")).unwrap()).unwrap();
    let cases: Vec<&Value> = t["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["tag"] == "token_map")
        .collect();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for case in cases {
        let want = case["error"].as_str().unwrap();
        let limits = case
            .get("limits")
            .map_or_else(Limits::default, |l| Limits::with_overrides(l).unwrap());
        let zip = unhex(case["file_hex"].as_str().unwrap());
        match convert(&zip, case["vocab_size"].as_u64().unwrap(), &limits) {
            Ok(_) => failures.push(format!("{}: accepted, want {want}", case["id"])),
            Err(e) if e.class() != want => {
                failures.push(format!("{}: got {}, want {want}", case["id"], e.class()))
            }
            Err(_) => {}
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn fixture_map_converts_like_python() {
    let pt = model_src().join("token_map.pt");
    let config = model_src().join("config.json");
    let vocab = vocab_size_from_config(&std::fs::read(&config).unwrap()).unwrap();
    let ids = convert(&std::fs::read(&pt).unwrap(), vocab, &Limits::default()).unwrap();
    let rust = to_u32le(&ids);
    let py = ok(run(python("token_map.py")
        .arg(&pt)
        .arg("--config")
        .arg(&config)))
    .stdout;
    assert_eq!(rust.len(), py.len());
    assert!(
        rust == py,
        "converted bytes differ from scripts/token_map.py"
    );
}

#[test]
fn valid_token_maps_convert_to_the_table_bytes() {
    let t: Value =
        serde_json::from_slice(&std::fs::read(fixture("artifact_invalid.json")).unwrap()).unwrap();
    let cases = t["valid_token_maps"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let zip = unhex(case["file_hex"].as_str().unwrap());
        let ids = convert(
            &zip,
            case["vocab_size"].as_u64().unwrap(),
            &Limits::default(),
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case["id"]));
        assert_eq!(
            to_u32le(&ids),
            unhex(case["u32le_hex"].as_str().unwrap()),
            "{}",
            case["id"]
        );
    }
}

/// The real NEXTN map (read-only, metadata-sized); skipped when the model is not on this host.
#[test]
fn real_token_map_converts_like_python() {
    let pt = std::path::Path::new("/var/lib/cere/models/qwen38-flash-next/flash-next-64k.pt");
    if !pt.is_file() {
        eprintln!("skip: {} absent", pt.display());
        return;
    }
    const VOCAB: u64 = 248_320; // m21 design note: text_config.vocab_size
    let ids = convert(&std::fs::read(pt).unwrap(), VOCAB, &Limits::default()).unwrap();
    assert_eq!(ids.len(), 65_536);
    assert!(ids.windows(2).all(|w| w[0] < w[1]));
    assert!(u64::from(*ids.last().unwrap()) < VOCAB);
    let tmp = Tmp::new("real-token-map");
    let config = tmp.join("config.json");
    std::fs::write(&config, br#"{"text_config": {"vocab_size": 248320}}"#).unwrap();
    let py = ok(run(python("token_map.py")
        .arg(pt)
        .arg("--config")
        .arg(&config)))
    .stdout;
    assert!(
        to_u32le(&ids) == py,
        "converted bytes differ from scripts/token_map.py"
    );
}
