use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLifecycle {
    Joining,
    Ready,
    Draining,
    Suspect,
    Lost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeartbeatInfo {
    pub last_seen_ns: u64,
    pub interval_ms: u64,
    pub clock_domain: String,
    pub sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerNode {
    pub schema_version: String,
    pub worker_id: String,
    pub worker_epoch: u64,
    pub node_name: String,
    pub lifecycle_state: WorkerLifecycle,
    pub protocol_version: String,
    pub transport: String,
    pub resource_graph_name: String,
    pub resource_ids: Vec<String>,
    pub backend_profile_ids: Vec<String>,
    pub artifact_ids: Vec<String>,
    pub model_instance_ids: Vec<String>,
    pub heartbeat: HeartbeatInfo,
    pub labels: BTreeMap<String, String>,
    pub coordinator_term: u64,
    pub desired_state: String,
    pub reconciled: bool,
}
