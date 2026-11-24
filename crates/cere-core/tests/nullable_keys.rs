use cere_core::capabilities::ExecutionCapabilities;
use cere_core::resource_graph::ResourceGraph;
use cere_core::work_unit::WorkUnit;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// `key` in the object at JSON pointer `object` is schema-required but nullable:
/// omission must fail, explicit null must pass.
fn check<T: DeserializeOwned>(example: &str, object: &str, key: &str) {
    let base: Value = serde_json::from_str(example).unwrap();

    let mut omitted = base.clone();
    omitted
        .pointer_mut(object)
        .and_then(Value::as_object_mut)
        .unwrap()
        .remove(key)
        .unwrap();
    assert!(
        serde_json::from_value::<T>(omitted).is_err(),
        "{object}/{key}: omission accepted"
    );

    let mut null = base;
    null.pointer_mut(object).unwrap()[key] = Value::Null;
    if let Err(e) = serde_json::from_value::<T>(null) {
        panic!("{object}/{key}: explicit null rejected: {e}");
    }
}

#[test]
fn work_unit_required_nullable_keys() {
    let example = include_str!("../../../examples/work_unit.json");
    for key in ["session_id", "deadline_mono_ns", "deadline_clock_domain"] {
        check::<WorkUnit>(example, "", key);
    }
}

#[test]
fn execution_capabilities_required_nullable_keys() {
    let example = include_str!("../../../examples/execution_capabilities.json");
    check::<ExecutionCapabilities>(example, "", "json_schema_dialect");
}

#[test]
fn resource_graph_required_nullable_keys() {
    let example = include_str!("../../../examples/resource_graph.json");
    check::<ResourceGraph>(example, "/nodes/0", "capacity_bytes");
    for key in ["bandwidth_bytes_s", "latency_ns", "peer_access"] {
        check::<ResourceGraph>(example, "/links/0", key);
    }
}
