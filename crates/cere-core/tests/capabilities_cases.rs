use cere_core::capabilities::ExecutionCapabilities;
use serde_json::Value;

/// Same table a Python parity test consumes; verdicts must agree across languages.
#[test]
fn shared_case_table_verdicts() {
    let table: Value =
        serde_json::from_str(include_str!("fixtures/capabilities_cases.json")).unwrap();
    assert_eq!(table["base"], "examples/execution_capabilities.json");
    let base: Value = serde_json::from_str(include_str!(
        "../../../examples/execution_capabilities.json"
    ))
    .unwrap();
    let cases = table["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let mut mismatches = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mut doc = base.clone();
        for (key, value) in case["set"].as_object().unwrap() {
            assert!(doc.get(key).is_some(), "{name}: unknown key {key}");
            doc[key] = value.clone();
        }
        let caps: ExecutionCapabilities = serde_json::from_value(doc)
            .unwrap_or_else(|e| panic!("{name}: case must deserialize: {e}"));
        let expected = case["valid"].as_bool().unwrap();
        let verdict = caps.validate();
        if verdict.is_ok() != expected {
            mismatches.push(format!(
                "{name}: expected valid={expected}, got {verdict:?}"
            ));
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// M2.6 C6 (D10), raised by m23 design note P8: the shipped native profile validates with decode `row_counts`
/// [1..16]; [17] exceeds its rows.
#[test]
fn native_profile_decode_rows_one_to_sixteen() {
    let mut caps: ExecutionCapabilities = serde_json::from_str(include_str!(
        "../../../configs/deployments/qwen38-flash-next-native.capabilities.json"
    ))
    .unwrap();
    assert_eq!((caps.decode_batch_min, caps.decode_batch_max), (1, 16));
    assert!(caps.native_decode_batch);
    let decode_rows: Vec<u64> = (1..=16).collect();
    for c in &caps.validated_combinations {
        let rows: &[u64] = if c.operation == "decode" {
            &decode_rows
        } else {
            &[1]
        };
        assert_eq!(c.row_counts, rows, "{}", c.operation);
    }
    assert_eq!(caps.validate(), Ok(()));
    for c in caps.validated_combinations.iter_mut() {
        if c.operation == "decode" {
            c.row_counts = vec![17];
        }
    }
    assert!(caps.validate().is_err());
}
