use cere_core::capabilities::ExecutionCapabilities;
use cere_core::resource_graph::ResourceGraph;
use cere_core::state::StateObject;
use cere_core::work_unit::WorkUnit;
use cere_core::worker::WorkerNode;
use serde::{de::DeserializeOwned, Serialize};

fn roundtrip<T: DeserializeOwned + Serialize>(text: &str) {
    let value: serde_json::Value = serde_json::from_str(text).unwrap();
    let typed: T = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(typed).unwrap(), value);
}

#[test]
fn schema_fixture_roundtrips() {
    roundtrip::<ResourceGraph>(include_str!("../../../examples/resource_graph.json"));
    roundtrip::<ExecutionCapabilities>(include_str!(
        "../../../examples/execution_capabilities.json"
    ));
    roundtrip::<StateObject>(include_str!("../../../examples/state_object.json"));
    roundtrip::<WorkUnit>(include_str!("../../../examples/work_unit.json"));
    roundtrip::<WorkerNode>(include_str!("../../../examples/worker_node.json"));
}

#[test]
fn example_semantics() {
    let graph: ResourceGraph =
        serde_json::from_str(include_str!("../../../examples/resource_graph.json")).unwrap();
    graph.validate().unwrap();
    let caps: ExecutionCapabilities = serde_json::from_str(include_str!(
        "../../../examples/execution_capabilities.json"
    ))
    .unwrap();
    caps.validate().unwrap();
}

#[test]
fn unknown_fields_rejected() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../../../examples/work_unit.json")).unwrap();
    value["typo"] = serde_json::json!(true);
    assert!(serde_json::from_value::<WorkUnit>(value).is_err());
}
