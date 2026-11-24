use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkType {
    Prefill,
    Decode,
    Verify,
    Draft,
    StateMove,
    StateRestore,
    ToolWait,
    Load,
    Infill,
    Embed,
    Rerank,
    Classify,
    Score,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkUnit {
    pub schema_version: String,
    pub id: String,
    pub work_type: WorkType,
    pub scope_id: String,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub session_id: Option<String>,
    pub request_id: String,
    pub payload_sha256: String,
    pub artifact_sha256: String,
    pub expected_session_revision: u64,
    pub attempt: u64,
    pub worker_id: String,
    pub worker_epoch: u64,
    pub coordinator_term: u64,
    pub model_generation: u64,
    pub priority: u64,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub deadline_mono_ns: Option<u64>,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub deadline_clock_domain: Option<String>,
    pub input_tokens: u64,
    pub max_new_tokens: u64,
    pub estimated_peak_bytes: u64,
    pub preemptible: bool,
    pub migratable: bool,
    pub speculative: bool,
    pub state_dependencies: Vec<String>,
}
