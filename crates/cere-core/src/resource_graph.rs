use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Machine,
    Gpu,
    Cpu,
    DeviceMemory,
    HostMemory,
    Nvme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceNode {
    pub id: String,
    pub kind: ResourceKind,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub capacity_bytes: Option<u64>,
    pub attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceLink {
    pub from: String,
    pub to: String,
    pub kind: String,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub bandwidth_bytes_s: Option<f64>,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub latency_ns: Option<f64>,
    #[serde(deserialize_with = "crate::required_nullable")]
    pub peer_access: Option<bool>,
    pub contention_domain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceGraph {
    pub schema_version: String,
    pub name: String,
    pub evidence: String,
    pub nodes: Vec<ResourceNode>,
    pub links: Vec<ResourceLink>,
}

impl ResourceGraph {
    /// Additional graph semantics after canonical JSON Schema validation.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != "1" || self.nodes.is_empty() {
            return Err("invalid version or empty graph");
        }
        let mut ids = HashSet::new();
        for node in &self.nodes {
            if node.id.is_empty() || !ids.insert(node.id.as_str()) {
                return Err("empty or duplicate node");
            }
        }
        for link in &self.links {
            if !ids.contains(link.from.as_str()) || !ids.contains(link.to.as_str()) {
                return Err("unknown endpoint");
            }
            if link.from == link.to {
                return Err("self link");
            }
            if let Some(bandwidth) = link.bandwidth_bytes_s {
                if !bandwidth.is_finite() || bandwidth <= 0.0 {
                    return Err("invalid bandwidth");
                }
            }
            if let Some(latency) = link.latency_ns {
                if !latency.is_finite() || latency < 0.0 {
                    return Err("invalid latency");
                }
            }
        }
        Ok(())
    }
}
