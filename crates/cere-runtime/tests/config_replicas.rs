//! M3 C1: `[[deployments.replicas]]` (Q2, Q3, Q6, Q13) and `scheduler.worker_lost_after_ms` (Q10). Every refused
//! case is also a row of the shared table `tests/fixtures/runtime_config_invalid.json` (run by
//! `config_rev2.rs` and tests/test_cere_contracts.py); here each is checked for its specific error.
mod config_pins;

use std::path::{Path, PathBuf};
use std::process::Command;

use cere_runtime::config::{NativeReplicaConfig, ReplicaConfig, RuntimeConfig};
use serde_json::{json, Value};

const LIVE: &str = "configs/runtime/sglang-t4-qwen38.toml";
const NATIVE: &str = "configs/runtime/native-g0.toml";
/// m32n design note D1/D2: one `native_cuda` deployment with two in-process native replicas (gpu0, gpu1).
const NATIVE_2GPU: &str = "configs/runtime/native-2gpu.toml";
const GPU0_UUID: &str = "GPU-aaaaaaaa-0000-4000-8000-000000000000";
const GPU1_UUID: &str = "GPU-bbbbbbbb-1111-4111-8111-111111111111";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// RFC 7396 JSON merge patch (arrays are replaced whole; `null` removes a key).
fn merge_patch(target: Value, patch: &Value) -> Value {
    match patch {
        Value::Object(patch) => {
            let mut out = match target {
                Value::Object(map) => map,
                _ => serde_json::Map::new(),
            };
            for (key, value) in patch {
                if value.is_null() {
                    out.remove(key);
                } else {
                    let current = out.remove(key).unwrap_or(Value::Null);
                    out.insert(key.clone(), merge_patch(current, value));
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// `base` with `patch` merged, then `patch["entry"]` merged into `deployments[0]` (the fixture table's
/// `entry_patch`, M5 D4).
fn load_patched(base: &str, patch: &Value) -> Result<RuntimeConfig, String> {
    let text = std::fs::read_to_string(root().join(base)).unwrap();
    let table: toml::Table = toml::from_str(&text).unwrap();
    let mut patch = patch.clone();
    let entry = patch.as_object_mut().unwrap().remove("entry");
    let mut value = merge_patch(serde_json::to_value(table).unwrap(), &patch);
    if let Some(entry) = entry {
        let first = &mut value["deployments"][0];
        *first = merge_patch(first.take(), &entry);
    }
    RuntimeConfig::from_toml_str(&toml::to_string(&value).unwrap(), &root())
        .map_err(|e| e.message().to_owned())
}

/// Two replicas of the T4 deployment: per-worker cap 4 (`expect.max_running_requests`), global cap 8.
fn two_replicas() -> Value {
    json!({
        "entry": {
            "base_url": null,
            "replicas": [
                {"device": "gpu0", "base_url": "http://127.0.0.1:18080", "load_endpoint": "/get_load"},
                {"device": "gpu1", "base_url": "http://127.0.0.1:18317"}
            ]
        },
        "scheduler": {"devices": ["gpu0", "gpu1"]},
        "admission": {"max_backend_inflight": 8, "per_scope_inflight": 8}
    })
}

fn with(patch: Value) -> Value {
    merge_patch(two_replicas(), &patch)
}

fn refused(base: &str, patch: &Value, needle: &str) {
    match load_patched(base, patch) {
        Ok(_) => panic!("accepted, expected {needle:?}: {patch}"),
        Err(e) => assert!(e.contains(needle), "error {e:?} lacks {needle:?}"),
    }
}

#[test]
fn two_replica_config_loads_and_lists_both() {
    let config = load_patched(LIVE, &two_replicas()).unwrap();
    let list = config.deployments[0].replica_list(&config.scheduler.devices);
    assert_eq!(
        list,
        vec![
            ReplicaConfig {
                device: "gpu0".into(),
                base_url: "http://127.0.0.1:18080".into(),
                load_endpoint: Some("/get_load".into()),
            },
            ReplicaConfig {
                device: "gpu1".into(),
                base_url: "http://127.0.0.1:18317".into(),
                load_endpoint: None,
            },
        ]
    );
    assert_eq!(config.admission.max_backend_inflight, 8);
}

#[test]
fn single_form_is_one_replica_on_the_first_device() {
    let config = load_patched(LIVE, &json!({"entry": {"load_endpoint": "/get_load"}})).unwrap();
    assert_eq!(
        config.deployments[0].replica_list(&config.scheduler.devices),
        vec![ReplicaConfig {
            device: "gpu1".into(),
            base_url: "http://127.0.0.1:18080".into(),
            load_endpoint: Some("/get_load".into()),
        }]
    );
}

#[test]
fn worker_lost_after_defaults_and_stays_out_of_the_hash() {
    let absent = load_patched(LIVE, &json!({})).unwrap();
    assert_eq!(absent.scheduler.worker_lost_after_ms, None);
    assert_eq!(absent.scheduler.worker_lost_after().as_millis(), 60_000);
    let set = load_patched(LIVE, &json!({"scheduler": {"worker_lost_after_ms": 5000}})).unwrap();
    assert_eq!(set.scheduler.worker_lost_after().as_millis(), 5000);
    assert_ne!(absent.config_hash(), set.config_hash());
    refused(
        LIVE,
        &json!({"scheduler": {"worker_lost_after_ms": 999}}),
        "scheduler.worker_lost_after_ms",
    );
}

#[test]
fn every_refused_replica_case_has_its_error() {
    let cases: [(&str, Value, &str); 11] = [
        (
            LIVE,
            with(json!({"entry": {"base_url": "http://127.0.0.1:18080"}})),
            "exactly one of deployments[0].base_url and deployments[0].replicas",
        ),
        (
            LIVE,
            json!({"entry": {"base_url": null}}),
            "exactly one of deployments[0].base_url and deployments[0].replicas",
        ),
        (
            LIVE,
            with(json!({"entry": {"replicas": [
                {"device": "gpu0", "base_url": "http://127.0.0.1:18080"},
                {"device": "gpu0", "base_url": "http://127.0.0.1:18317"}
            ]}})),
            "duplicate replica device",
        ),
        (
            LIVE,
            with(json!({"entry": {"replicas": [
                {"device": "gpu0", "base_url": "http://127.0.0.1:18080"},
                {"device": "gpu1", "base_url": "http://127.0.0.1:18080"}
            ]}})),
            "duplicate replica base_url",
        ),
        (
            LIVE,
            with(json!({"entry": {"replicas": [
                {"device": "gpu0", "base_url": "http://127.0.0.1:18080"},
                {"device": "gpu2", "base_url": "http://127.0.0.1:18317"}
            ]}})),
            "replica device is not in scheduler.devices",
        ),
        (
            LIVE,
            with(json!({"scheduler": {"devices": ["gpu0", "gpu1", "gpu2"]}})),
            "scheduler.devices must list exactly the replica devices",
        ),
        (
            LIVE,
            with(json!({"entry": {"load_endpoint": "/get_load"}})),
            "deployments[0].load_endpoint is refused with deployments[0].replicas",
        ),
        (
            NATIVE,
            json!({"entry": {"replicas": [
                {"device": "gpu0", "base_url": "http://127.0.0.1:18080"}
            ]}}),
            "native_cuda refuses deployments[0].replicas",
        ),
        (
            LIVE,
            with(json!({"admission": {"max_backend_inflight": 9}})),
            "admission max_backend_inflight exceeds replicas x expect.max_running_requests",
        ),
        (
            LIVE,
            with(json!({"entry": {"expect": {"max_running_requests": 5}}})),
            "expect.max_running_requests exceeds the profile's validated rows",
        ),
        (
            LIVE,
            json!({"scheduler": {"devices": ["gpu0", "gpu1"]}}),
            "single backend form needs exactly one scheduler device",
        ),
    ];
    for (base, patch, needle) in &cases {
        refused(base, patch, needle);
    }
}

/// m32n design note D2: the two-card config lists gpu0 then gpu1 with their UUIDs; the deployment-level
/// `device_uuid` is absent.
#[test]
fn native_replicas_config_loads_two_workers() {
    let config = RuntimeConfig::load(root().join(NATIVE_2GPU)).unwrap();
    let native = config.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.device_uuid, None);
    assert_eq!(config.scheduler.devices, ["gpu0", "gpu1"]);
    assert_eq!(
        native.replica_list(&config.scheduler.devices),
        vec![
            NativeReplicaConfig {
                device: "gpu0".into(),
                device_uuid: GPU0_UUID.into(),
            },
            NativeReplicaConfig {
                device: "gpu1".into(),
                device_uuid: GPU1_UUID.into(),
            },
        ]
    );
    assert_eq!(config.admission.max_backend_inflight, 8);
}

/// The single form is one native replica on `scheduler.devices[0]` with the deployment's `device_uuid`.
#[test]
fn native_single_form_is_one_replica_on_the_first_device() {
    for rel in config_pins::config_hashes(&root()).into_keys() {
        let config = RuntimeConfig::load(root().join(&rel)).unwrap();
        let Some(native) = config.deployments[0].native.as_ref() else {
            continue;
        };
        if rel == NATIVE_2GPU {
            continue;
        }
        let uuid = native
            .device_uuid
            .clone()
            .unwrap_or_else(|| panic!("{rel}"));
        assert_eq!(
            native.replica_list(&config.scheduler.devices),
            vec![NativeReplicaConfig {
                device: config.scheduler.devices[0].clone(),
                device_uuid: uuid,
            }],
            "{rel}"
        );
    }
}

/// amd-amd design note PAR-6: an AMD GPU is bound as `hip:<ordinal>` (HIP_VISIBLE_DEVICES), the single form only;
/// it lists as one replica on `scheduler.devices[0]` like a GPU uuid. SelfTune D3: the HIP library is the k7 MoE
/// build, so the binding selects `moe_gemm = "k7"`; the default `p3b` is not proven on gfx1151 and is refused.
#[test]
fn native_hip_binding_is_one_replica_on_the_first_device() {
    let config = load_patched(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "hip:0", "levers": {"moe_gemm": "k7"}}}}),
    )
    .unwrap();
    let entry = &config.deployments[0];
    let native = entry.native.as_ref().unwrap();
    assert_eq!(native.device_uuid.as_deref(), Some("hip:0"));
    assert_eq!(
        native.replica_list(&config.scheduler.devices),
        vec![NativeReplicaConfig {
            device: config.scheduler.devices[0].clone(),
            device_uuid: "hip:0".into()
        }]
    );
    assert_eq!(cere_runtime::config::hip_ordinal("hip:0"), Some("0"));
    assert_eq!(cere_runtime::config::hip_ordinal("hip:12"), Some("12"));
    for bad in [
        "hip:00", "hip:x", "hip:", "hip:1234", "hip:-1", "HIP:0", GPU0_UUID,
    ] {
        assert_eq!(cere_runtime::config::hip_ordinal(bad), None, "{bad}");
    }
    refused(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "hip:01"}}}),
        "deployments[0].native.device_uuid must be GPU-<8-4-4-4-12 lowercase hex>, hip:<ordinal> or vk:<ordinal>",
    );
    refused(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "hip:0"}}}),
        "deployments[0].native.levers.moe_gemm: \"p3b\" is not proven on gfx1151",
    );
}

/// vulkan-amd design note D2: the Vulkan executor is bound as `vk:<ordinal>` (the executor device id), the single
/// form only; it lists as one replica on `scheduler.devices[0]`. The Vulkan arch is proven for the k7 MoE build only:
/// the default `p3b` is refused, and a replicas table takes GPU uuids only.
#[test]
fn native_vulkan_binding_is_one_replica_on_the_first_device() {
    let config = load_patched(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "vk:0", "levers": {"moe_gemm": "k7"}}}}),
    )
    .unwrap();
    let entry = &config.deployments[0];
    let native = entry.native.as_ref().unwrap();
    assert_eq!(native.device_uuid.as_deref(), Some("vk:0"));
    assert_eq!(
        native.replica_list(&config.scheduler.devices),
        vec![NativeReplicaConfig {
            device: config.scheduler.devices[0].clone(),
            device_uuid: "vk:0".into()
        }]
    );
    assert_eq!(cere_runtime::config::vk_ordinal("vk:0"), Some("0"));
    assert_eq!(cere_runtime::config::vk_ordinal("vk:12"), Some("12"));
    assert_eq!(cere_runtime::config::vk_ordinal("vk:999"), Some("999"));
    for bad in [
        "vk:00", "vk:01", "vk:x", "vk:", "vk:1234", "vk:-1", "vk:+1", "VK:0", "hip:0", "vk:0 ",
        GPU0_UUID,
    ] {
        assert_eq!(cere_runtime::config::vk_ordinal(bad), None, "{bad}");
    }
    assert_eq!(cere_runtime::config::hip_ordinal("vk:0"), None);
    for bad in ["vk:01", "vk:", "vk:x", "vk:-1"] {
        refused(
            NATIVE,
            &json!({"entry": {"native": {"device_uuid": bad}}}),
            "deployments[0].native.device_uuid must be GPU-<8-4-4-4-12 lowercase hex>, hip:<ordinal> or vk:<ordinal>",
        );
    }
    refused(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "vk:0"}}}),
        "deployments[0].native.levers.moe_gemm: \"p3b\" is not proven on vulkan",
    );
    refused(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": null, "replicas": [
            {"device": "gpu0", "device_uuid": "vk:0"},
            {"device": "gpu1", "device_uuid": GPU1_UUID}
        ]}}}),
        "deployments[0].native.replicas.device_uuid must be GPU-<8-4-4-4-12 lowercase hex>",
    );
    // An sdar library is built on the CUDA arch only (check_family).
    refused(
        "configs/runtime/native-g0-dice8b.toml",
        &json!({"entry": {"native": {"device_uuid": "vk:0"}}}),
        "native.family = \"sdar\"",
    );
    // `p4` is the Vulkan FP16 cooperative-matrix prefill attention: accepted on vulkan, refused on the HIP arch.
    let p4 = json!({"moe_gemm": "k7", "prefill_attention": "p4"});
    load_patched(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "vk:0", "levers": p4}}}),
    )
    .unwrap();
    refused(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "hip:0", "levers": p4}}}),
        "levers.prefill_attention: \"p4\" is not proven on gfx1151",
    );
    // The levers the Vulkan library refuses at first light are refused at config validation, lever name first.
    for (levers, want) in [
        (
            json!({"k2_gemm": "small_m"}),
            "levers.k2_gemm: \"small_m\" is not proven on vulkan",
        ),
        (
            json!({"k2_gemm": "small_m_a8"}),
            "levers.k2_gemm: \"small_m_a8\" is not proven on vulkan",
        ),
        (
            json!({"decode_step": "graph"}),
            "levers.decode_step: \"graph\" is not proven on vulkan",
        ),
        (
            json!({"verify_attention": "batched"}),
            "levers.verify_attention: \"batched\" is not proven on vulkan",
        ),
        (
            json!({"k2_weights": "fp8"}),
            "levers.k2_weights: \"fp8\" is not proven on vulkan",
        ),
    ] {
        let mut levers = levers;
        levers["moe_gemm"] = json!("k7");
        refused(
            NATIVE,
            &json!({"entry": {"native": {"device_uuid": "vk:0", "levers": levers}}}),
            want,
        );
    }
}

fn native_replicas(replicas: Value) -> Value {
    json!({"entry": {"native": {"replicas": replicas}}})
}

/// m32n design note D2/D16: each native replica rule refuses with its own error.
#[test]
fn every_refused_native_replica_case_has_its_error() {
    let three = json!([
        {"device": "gpu0", "device_uuid": GPU0_UUID},
        {"device": "gpu1", "device_uuid": GPU1_UUID},
        {"device": "gpu2", "device_uuid": "GPU-11111111-2222-3333-4444-555555555555"}
    ]);
    let cases: [(Value, String); 14] = [
        (
            native_replicas(json!([
                {"device": "gpu0", "device_uuid": GPU0_UUID, "base_url": "http://127.0.0.1:18308"},
                {"device": "gpu1", "device_uuid": GPU1_UUID}
            ])),
            "unknown field `base_url`".into(),
        ),
        (
            native_replicas(json!([
                {"device": "gpu0", "device_uuid": GPU0_UUID},
                {"device": "gpu0", "device_uuid": GPU1_UUID}
            ])),
            "duplicate native replica device gpu0".into(),
        ),
        (
            native_replicas(json!([
                {"device": "gpu0", "device_uuid": GPU0_UUID},
                {"device": "gpu1", "device_uuid": GPU0_UUID}
            ])),
            format!("duplicate native replica device_uuid {GPU0_UUID}"),
        ),
        (
            json!({"entry": {"native": {"device_uuid": GPU0_UUID}}}),
            "deployments[0].native.device_uuid is refused with deployments[0].native.replicas; set it per replica"
                .into(),
        ),
        (
            json!({"entry": {"native": {"replicas": null}}}),
            "deployments[0].native requires exactly one of device_uuid and replicas".into(),
        ),
        (
            native_replicas(json!([
                {"device": "gpu0", "device_uuid": GPU0_UUID},
                {"device": "gpu2", "device_uuid": GPU1_UUID}
            ])),
            "native replica device is not in scheduler.devices: gpu2".into(),
        ),
        (
            json!({"scheduler": {"devices": ["gpu0", "gpu1", "gpu2"]}}),
            "scheduler.devices must list exactly the native replica devices".into(),
        ),
        (
            native_replicas(json!([{"device": "gpu0", "device_uuid": GPU0_UUID}])),
            "deployments[0].native.replicas must list exactly 2 replicas".into(),
        ),
        (
            merge_patch(
                native_replicas(three),
                &json!({"scheduler": {"devices": ["gpu0", "gpu1", "gpu2"]}}),
            ),
            "deployments[0].native.replicas must list exactly 2 replicas".into(),
        ),
        (
            native_replicas(json!([
                {"device": "gpu0", "device_uuid": GPU0_UUID},
                {"device": "gpu1", "device_uuid": "GPU-BBBBBBBB-1111-4111-8111-111111111111"}
            ])),
            "deployments[0].native.replicas.device_uuid must be GPU-<8-4-4-4-12 lowercase hex>".into(),
        ),
        (
            native_replicas(json!([
                {"device": "gpu 0", "device_uuid": GPU0_UUID},
                {"device": "gpu1", "device_uuid": GPU1_UUID}
            ])),
            "deployments[0].native.replicas.device".into(),
        ),
        (
            json!({"admission": {"max_backend_inflight": 9}}),
            "admission max_backend_inflight exceeds native replicas x the profile's validated rows (+ max_parked_sessions with preempt)"
                .into(),
        ),
        (
            json!({"entry": {"native": {"lever_hot_swap": true}}}),
            "deployments[0].native.lever_hot_swap is refused with deployments[0].native.replicas".into(),
        ),
        (
            json!({"entry": {"native": {"lora": {"max_adapters": 2, "max_rank": 16, "adapter_root": "/var/lib/cere/lora"}}}}),
            "deployments[0].native.lora is refused with deployments[0].native.replicas".into(),
        ),
    ];
    for (patch, needle) in &cases {
        refused(NATIVE_2GPU, patch, needle);
    }
    // The global cap is exactly replicas x rows (D16): 8 loads.
    load_patched(
        NATIVE_2GPU,
        &json!({"admission": {"max_backend_inflight": 8}}),
    )
    .unwrap();
}

/// Every committed runtime config's `config_hash` equals its pin (tests/fixtures/config_pins.json, generated from the
/// Python `canonical_digest`; a relock repins it with `scripts/pin_configs.py --write`).
#[test]
fn committed_config_hashes_are_unchanged() {
    let pins = config_pins::config_hashes(&root());
    assert!(pins.len() > 40, "{}", pins.len());
    for (rel, hash) in pins {
        let config =
            RuntimeConfig::load(root().join(&rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert_eq!(config.config_hash(), hash, "{rel}");
    }
}

#[test]
fn cered_check_config_accepts_every_committed_toml() {
    for rel in config_pins::config_hashes(&root()).into_keys() {
        let status = Command::new(env!("CARGO_BIN_EXE_cered"))
            .arg("--check-config")
            .arg(root().join(&rel))
            .status()
            .unwrap();
        assert!(status.success(), "{rel}");
    }
}
