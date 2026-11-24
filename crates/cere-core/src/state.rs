use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTier {
    DeviceMemory,
    HostRam,
    Nvme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateObject {
    pub schema_version: String,
    pub id: String,
    pub state_type: String,
    pub scope_id: String,
    pub model_fingerprint: String,
    pub input_fingerprint: String,
    pub execution_fingerprint: String,
    pub size_bytes: u64,
    pub tier: StateTier,
    pub current_location: String,
    pub generation: u64,
    pub token_boundary: u64,
    pub immutable: bool,
    pub sharing_count: u64,
    pub lease_count: u64,
    pub reuse_probability: f64,
}
