//! Runtime config revision 2: shipped TOMLs load, the shared invalid table (also run by
//! tests/test_contracts.py) is rejected, auth resolution and deployment identity.
mod config_pins;

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use cere_runtime::config::{
    parse_loopback_base_url, BackendKind, PrefixCacheIsolation, RuntimeConfig, SharedBackendTenancy,
};
use cere_runtime::ledger::LedgerLimits;
use serde_json::Value;
use sha2::{Digest, Sha256};

const LIVE: &str = "configs/runtime/sglang-t4-qwen38.toml";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap()
}

fn live() -> RuntimeConfig {
    RuntimeConfig::load(root().join(LIVE)).unwrap()
}

/// RFC 7396 JSON merge patch; `${REPO_ROOT}` in strings becomes the repository root.
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
        Value::String(s) => Value::String(s.replace("${REPO_ROOT}", root().to_str().unwrap())),
        other => other.clone(),
    }
}

/// `base` with `patch` merged, then `entry` merged into `deployments[0]` (the fixture table's `entry_patch`, M5 D4).
fn patch_document(base: &str, patch: &Value, entry: Option<&Value>) -> String {
    let table: toml::Table = toml::from_str(&read(base)).unwrap();
    let mut value = merge_patch(serde_json::to_value(table).unwrap(), patch);
    if let Some(entry) = entry {
        let first = &mut value["deployments"][0];
        *first = merge_patch(first.take(), entry);
    }
    toml::to_string(&value).expect("patched case must still be expressible as TOML")
}

/// [`patch_document`] with `patch["entry"]` as the entry patch.
fn patched_toml(base: &str, patch: &Value) -> String {
    let mut patch = patch.clone();
    let entry = patch.as_object_mut().unwrap().remove("entry");
    patch_document(base, &patch, entry.as_ref())
}

/// The TOML text of one shared-table case: `patch`, then `entry_patch` on `deployments[0]`. A top-level
/// `backend` key in `patch` is refused (M5 D4): it would only add an unknown table and pass for the wrong reason.
fn case_toml(case: &Value) -> Result<String, String> {
    let name = case["name"].as_str().unwrap_or("?");
    if case["patch"].get("backend").is_some() {
        return Err(format!(
            "case {name}: patch.backend is the pre-M5 shape; use entry_patch"
        ));
    }
    Ok(patch_document(
        case["base"].as_str().unwrap(),
        &case["patch"],
        case.get("entry_patch"),
    ))
}

/// A scratch root holding `configs/deployments/` with each part of the deployment `config` (TOML text) names in
/// `deployments[0].deployment` merge-patched and the lock re-pinned to the patched recipe bytes (unless the lock patch
/// sets `recipe_sha256`), so only the patched constraint is under test. tests/test_contracts.py builds the same tree.
struct DeploymentRoot(PathBuf);

impl DeploymentRoot {
    fn new(tag: &str, config: &str, patches: &Value) -> Self {
        let dir =
            std::env::temp_dir().join(format!("cere-config-rev2-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("configs/deployments")).unwrap();
        let config: toml::Table = toml::from_str(config).unwrap();
        let deployment = config["deployments"][0]["deployment"].as_str().unwrap();
        let mut recipe_sha = String::new();
        for part in ["capabilities", "recipe", "lock"] {
            let rel = format!("{deployment}.{part}.json");
            let base: Value = serde_json::from_str(&read(&rel)).unwrap();
            let patch = patches
                .get(part)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let mut doc = merge_patch(base, &patch);
            if part == "lock" && patch.get("recipe_sha256").is_none() {
                doc["recipe_sha256"] = Value::String(recipe_sha.clone());
            }
            let bytes = serde_json::to_vec_pretty(&doc).unwrap();
            if part == "recipe" {
                recipe_sha = sha256_hex(&bytes);
            }
            std::fs::write(dir.join(&rel), bytes).unwrap();
        }
        Self(dir)
    }
}

impl Drop for DeploymentRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Every shipped runtime TOML loads, and its `config_hash` is the SHA-256 of the canonical JSON of the TOML
/// document (`canonical_digest` in scripts/cere_contracts.py), so Python and Rust agree on every one (Q3),
/// including the two-entry `multimodel-*.toml` (M5 C2).
#[test]
fn every_shipped_runtime_toml_loads() {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(root().join("configs/runtime")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "toml") {
            let config = RuntimeConfig::load(&path)
                .unwrap_or_else(|e| panic!("{} rejected: {e}", path.display()));
            let table: toml::Table =
                toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let document = serde_json::to_string(&serde_json::to_value(&table).unwrap()).unwrap();
            assert_eq!(
                config.config_hash(),
                sha256_hex(document.as_bytes()),
                "{}",
                path.display()
            );
            names.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    names.sort();
    assert_eq!(
        names,
        [
            "amd-flashnext.toml",
            "amd-serve-spec.toml",
            "amd-serve.toml",
            "dev-single-gpu.toml",
            "dev-two-replica.toml",
            "federation-dev.toml",
            "federation-flashnext-amd.toml",
            "federation-gemma-amd.toml",
            "federation-gemma.toml",
            "multimodel-dev.toml",
            "multimodel-g0g1.toml",
            "native-2gpu.toml",
            "native-amd-inkling-vk.toml",
            "native-amd-s3-spec.toml",
            "native-amd-s3-vk-spec.toml",
            "native-amd-s3-vk.toml",
            "native-amd-s3.toml",
            "native-g0-dice8b-gemm-changing.toml",
            "native-g0-dice8b.toml",
            "native-g0-inkling-batch.toml",
            "native-g0-inkling-plain.toml",
            "native-g0-inkling-s13.toml",
            "native-g0-inkling.toml",
            "native-g0-k2s.toml",
            "native-g0-p4.toml",
            "native-g0-prefix.toml",
            "native-g0-rows16-v58.toml",
            "native-g0-sessions-v53.toml",
            "native-g0-sessions-v59.toml",
            "native-g0-sessions.toml",
            "native-g0-spec-262k.toml",
            "native-g0-spec-chains.toml",
            "native-g0-spec-eval.toml",
            "native-g0-spec-fp4local-eval.toml",
            "native-g0-spec-fp4local-s3-eval.toml",
            "native-g0-spec-fp4local-s3.toml",
            "native-g0-spec-fp4local.toml",
            "native-g0-spec-fp8w.toml",
            "native-g0-spec-grammar.toml",
            "native-g0-spec-k2s.toml",
            "native-g0-spec-lookup.toml",
            "native-g0-spec-lora.toml",
            "native-g0-spec-nvfp4.toml",
            "native-g0-spec-research.toml",
            "native-g0-spec-tree.toml",
            "native-g0-spec-v22.toml",
            "native-g0-spec-v47.toml",
            "native-g0-spec-v51.toml",
            "native-g0-spec-v52.toml",
            "native-g0-spec-v56.toml",
            "native-g0-spec-vision-v54.toml",
            "native-g0-spec-vision-v55.toml",
            "native-g0-spec-vision.toml",
            "native-g0-spec.toml",
            "native-g0-vision.toml",
            "native-g0.toml",
            "native-g1-prefix.toml",
            "native-g1.toml",
            "prod-t4.toml",
            "sglang-bench-2gpu.toml",
            "sglang-bench-g0-replicas.toml",
            "sglang-bench-g0.toml",
            "sglang-t4-qwen38.toml"
        ]
    );
}

/// Every shared-table row gets its verdict: an `expect: "invalid"` row is refused and an `expect: "valid"` row (a
/// boundary case only the specified rule admits, e.g. the D2(j) per-entry sum) loads.
#[test]
fn shared_invalid_table_is_rejected() {
    let cases: Vec<Value> =
        serde_json::from_str(&read("tests/fixtures/runtime_config_invalid.json")).unwrap();
    assert!(cases.len() >= 40);
    let mut wrong = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let valid = match case["expect"].as_str() {
            Some("invalid") => false,
            Some("valid") => true,
            other => panic!("{name}: expect {other:?}"),
        };
        let text = case_toml(case).unwrap_or_else(|e| panic!("{e}"));
        let verdict = match case.get("deployment_patch") {
            None => RuntimeConfig::from_toml_str(&text, &root()),
            Some(patches) => {
                let scratch = DeploymentRoot::new(name, &text, patches);
                RuntimeConfig::from_toml_str(&text, &scratch.0)
            }
        };
        if verdict.is_ok() != valid {
            wrong.push(format!("{name}: {:?}", verdict.err()));
        }
    }
    assert!(wrong.is_empty(), "wrong verdicts: {wrong:?}");
}

/// Shared-table M5 multi-deployment rows (D2 (a)-(j)) -> the rule that refuses each; `tests/test_cere_contracts.py`
/// pins the same rows (schema rows there name `runtime_config`).
const M5_CASES: [(&str, &str); 24] = [
    ("mm_zero_deployments", "deployments item count out of range"),
    ("mm_seventeen_deployments", "deployments item count out of range"),
    ("mm_base_url_with_two", "deployments[1].base_url is refused with more than one deployment; use deployments[1].replicas"),
    ("mm_load_endpoint_with_two", "deployments[1].load_endpoint is refused with more than one deployment; set it per replica"),
    ("mm_no_local_replica", "at least one deployment needs a local replica"),
    ("mm_duplicate_served_model_name", "duplicate deployments served_model_name Qwen3.8-Flash-Next"),
    ("mm_duplicate_deployment", "duplicate deployments deployment configs/deployments/qwen38-flash-next"),
    ("mm_replica_device_in_two_entries", "replica device gpu0 belongs to deployments[0] and deployments[1]"),
    ("mm_devices_not_the_replica_union", "scheduler.devices must list exactly the replica devices"),
    ("mm_worker_matches_no_deployment", "federation worker worker-a expect.served_model_name other-model matches no deployments served_model_name"),
    ("mm_entry_without_worker", "deployments[1] has no worker (no local replica and no remote worker)"),
    ("mm_tenancy_differs", "deployments[1] shared_backend_tenancy, allow_cross_scope_prefix_cache and prefix_cache_isolation must equal deployments[0]'s"),
    ("mm_cross_scope_cache_differs", "deployments[1] shared_backend_tenancy, allow_cross_scope_prefix_cache and prefix_cache_isolation must equal deployments[0]'s"),
    ("mm_isolation_differs", "deployments[1] shared_backend_tenancy, allow_cross_scope_prefix_cache and prefix_cache_isolation must equal deployments[0]'s"),
    ("mm_two_native_entries", "more than one deployment requires every entry to be sglang_http (native_cuda, unconfigured and cpu_reference allow exactly one)"),
    ("mm_two_planning_entries", "more than one deployment requires every entry to be sglang_http (native_cuda, unconfigured and cpu_reference allow exactly one)"),
    ("mm_entry_cap_exceeds_rows", "expect.max_running_requests exceeds the profile's validated rows"),
    ("mm_entry_context_differs", "deployment context differs from expect.context_length"),
    ("mm_two_scopes_shared_cache", "more than one auth scope on a shared radix cache requires allow_cross_scope_prefix_cache or prefix_cache_isolation"),
    ("mm_global_cap_exceeded", "admission max_backend_inflight exceeds local + remote workers x min(expect.max_running_requests, every max_inflight), summed over deployments"),
    ("mm_global_cap_uses_entry_min_max_inflight", "admission max_backend_inflight exceeds local + remote workers x min(expect.max_running_requests, every max_inflight), summed over deployments"),
    ("mm_second_entry_without_expect", "sglang_http requires served_model_name, deployment and expect"),
    ("mm_second_entry_without_served_model_name", "sglang_http requires served_model_name, deployment and expect"),
    ("mm_second_entry_without_deployment", "sglang_http requires served_model_name, deployment and expect"),
];

/// Valid M5 rows of the shared table: boundary cases only the specified rule admits.
const M5_VALID_CASES: [&str; 1] = ["mm_global_cap_sums_entry_mins"];

#[test]
fn multi_deployment_cases_hit_their_rule() {
    let cases: Vec<Value> =
        serde_json::from_str(&read("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let named = |expect: &str| -> Vec<&str> {
        cases
            .iter()
            .filter(|c| c["expect"] == expect)
            .filter_map(|c| c["name"].as_str().filter(|n| n.starts_with("mm_")))
            .collect()
    };
    assert_eq!(named("invalid"), M5_CASES.map(|(n, _)| n));
    assert_eq!(named("valid"), M5_VALID_CASES);
    for (name, needle) in M5_CASES {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        let text = case_toml(case).unwrap_or_else(|e| panic!("{e}"));
        let err = RuntimeConfig::from_toml_str(&text, &root()).unwrap_err();
        assert!(err.message().contains(needle), "{name}: {err}");
    }
    for name in M5_VALID_CASES {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        let text = case_toml(case).unwrap_or_else(|e| panic!("{e}"));
        if let Err(err) = RuntimeConfig::from_toml_str(&text, &root()) {
            panic!("{name}: {err}");
        }
    }
}

/// M2.4 C9 rows -> the rule that refuses each (the D13 twin rule, the D12 range), the M2.7 C9 rows (D11
/// `sampling_params` names, a native sampled row without `sampling_params`) and the M2.13 C10 rows (D5);
/// `tests/test_contracts.py` pins the same rows with the schema's wording.
const C9_CASES: [(&str, &str); 6] = [
    (
        "capabilities_tool_calls_without_text",
        "tool_calls decode combination lacks the matching text decode combination",
    ),
    (
        "capabilities_max_top_logprobs_range",
        "capabilities.max_top_logprobs must be 0..20",
    ),
    (
        "capabilities_sampling_params_unknown",
        "capabilities.sampling_params must be one of",
    ),
    (
        "capabilities_sampled_without_sampler",
        "native sampled combination lacks sampling_params",
    ),
    // M2.13 C10 (m213 design note D5): the image twin rule and `modalities` consistency.
    (
        "capabilities_image_without_text",
        "image decode combination lacks the matching text decode combination",
    ),
    (
        "capabilities_image_modality_without_combination",
        "modalities lists image without an image decode combination",
    ),
];

#[test]
fn c9_capability_cases_hit_their_rule() {
    let cases: Vec<Value> =
        serde_json::from_str(&read("tests/fixtures/runtime_config_invalid.json")).unwrap();
    for (name, needle) in C9_CASES {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        let text = case_toml(case).unwrap_or_else(|e| panic!("{e}"));
        let scratch = DeploymentRoot::new(name, &text, &case["deployment_patch"]);
        let err = RuntimeConfig::from_toml_str(&text, &scratch.0).unwrap_err();
        assert!(err.message().contains(needle), "{name}: {err}");
    }
}

/// M2.11 C6 rows (m211 design note D8, D13; ledger C 56, LR5) and SE-09's (sessions design note) -> the rule that
/// refuses each, and the valid boundaries; `tests/test_cere_contracts.py` pins the same rows.
const SNAPSHOT_CASES: [(&str, &str); 15] = [
    (
        "snapshot_parked_exceed_arena",
        "deployments[0].native.max_parked_sessions x the snapshot bytes at max_context_tokens exceed snapshot_host_bytes",
    ),
    (
        "snapshot_preempt_without_arena",
        "deployments[0].native.preempt requires snapshot_host_bytes > 0",
    ),
    (
        "snapshot_preempt_without_host_spill",
        "deployments[0].native.preempt requires [state] host_spill_enabled",
    ),
    (
        "snapshot_inflight_above_rows_plus_parked",
        "admission max_backend_inflight exceeds the profile's validated rows (+ max_parked_sessions with preempt)",
    ),
    (
        "snapshot_parked_without_preempt",
        "admission max_backend_inflight exceeds the profile's validated rows (+ max_parked_sessions with preempt)",
    ),
    (
        "snapshot_arena_on_shipped_profile",
        "deployments[0].native.snapshot_host_bytes > 0 must equal the profile's native_state_snapshot",
    ),
    (
        "snapshot_profile_without_arena",
        "deployments[0].native.snapshot_host_bytes > 0 must equal the profile's native_state_snapshot",
    ),
    (
        "snapshot_preempt_every_shipped_profile",
        "deployments[0].native.preempt_every requires the dev/live <profile id>-snapshot profile",
    ),
    (
        "snapshot_preempt_every_without_suffix",
        "deployments[0].native.preempt_every requires the dev/live <profile id>-snapshot profile",
    ),
    (
        "snapshot_reserve_without_preempt",
        "deployments[0].native.reserve_output_tokens requires preempt = true",
    ),
    (
        "snapshot_reserve_not_page_multiple",
        "deployments[0].native.reserve_output_tokens must be a multiple of 64",
    ),
    (
        "snapshot_reserve_parked_below_inflight",
        "deployments[0].native.reserve_output_tokens requires max_parked_sessions >= admission.max_backend_inflight",
    ),
    (
        "snapshot_preempt_every_without_preempt",
        "deployments[0].native.preempt_every requires preempt = true",
    ),
    (
        "snapshot_preempt_every_without_parked",
        "deployments[0].native.preempt_every requires max_parked_sessions >= 1",
    ),
    (
        "snapshot_preempt_min_tokens_zero",
        "deployments[0].native.preempt_min_tokens must be 1..16777216",
    ),
];
const SNAPSHOT_VALID_CASES: [&str; 4] = [
    "snapshot_dev_profile_preempt_valid",
    "snapshot_parked_arena_boundary_valid",
    "snapshot_preempt_with_rows_valid",
    "snapshot_reserve_valid",
];

#[test]
fn snapshot_cases_hit_their_rule() {
    let cases: Vec<Value> =
        serde_json::from_str(&read("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let named = |expect: &str| -> Vec<&str> {
        cases
            .iter()
            .filter(|c| c["expect"] == expect)
            .filter_map(|c| c["name"].as_str().filter(|n| n.starts_with("snapshot_")))
            .collect()
    };
    let mut invalid = named("invalid");
    invalid.sort_unstable();
    let mut expected: Vec<&str> = SNAPSHOT_CASES.iter().map(|(n, _)| *n).collect();
    expected.sort_unstable();
    assert_eq!(invalid, expected);
    assert_eq!(named("valid"), SNAPSHOT_VALID_CASES);
    let verdict = |name: &str| {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        let text = case_toml(case).unwrap_or_else(|e| panic!("{e}"));
        match case.get("deployment_patch") {
            None => RuntimeConfig::from_toml_str(&text, &root()),
            Some(patches) => {
                let scratch = DeploymentRoot::new(name, &text, patches);
                RuntimeConfig::from_toml_str(&text, &scratch.0)
            }
        }
    };
    for (name, needle) in SNAPSHOT_CASES {
        let err = verdict(name).unwrap_err();
        assert!(err.message().contains(needle), "{name}: {err}");
    }
    for name in SNAPSHOT_VALID_CASES {
        if let Err(err) = verdict(name) {
            panic!("{name}: {err}");
        }
    }
}

/// M2.12 C10 rows -> the rule that refuses each (D2: `token_controls` items, `thinking_toggle` type, `vocab_size`
/// >= 1 and integer); `tests/test_contracts.py` pins the same rows with the schema's wording.
const M212_C10_CASES: [(&str, &str); 5] = [
    (
        "capabilities_token_controls_unknown",
        "capabilities.token_controls must be one of",
    ),
    (
        "capabilities_thinking_toggle_non_bool",
        "expected a boolean",
    ),
    (
        "capabilities_vocab_size_zero",
        "capabilities.vocab_size must be 1..",
    ),
    ("capabilities_vocab_size_non_integer", "expected u32"),
    // BC1 leftover (batch-concurrency design note "Prefill selection signature").
    (
        "capabilities_prefill_signature_not_hex",
        "capabilities.prefill_signature must be 64 lowercase hex",
    ),
];

#[test]
fn m212_c10_capability_cases_hit_their_rule() {
    let cases: Vec<Value> =
        serde_json::from_str(&read("tests/fixtures/runtime_config_invalid.json")).unwrap();
    for (name, needle) in M212_C10_CASES {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        let text = case_toml(case).unwrap_or_else(|e| panic!("{e}"));
        let scratch = DeploymentRoot::new(name, &text, &case["deployment_patch"]);
        let err = RuntimeConfig::from_toml_str(&text, &scratch.0).unwrap_err();
        assert!(err.message().contains(needle), "{name}: {err}");
    }
}

/// g9-prod design note F2: `[auth] keyless_scope` must name a configured scope; the shared rows pin the rule.
#[test]
fn keyless_scope_cases_hit_their_rule() {
    let cases: Vec<Value> =
        serde_json::from_str(&read("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let case = |name: &str| cases.iter().find(|c| c["name"] == name).unwrap();
    for name in ["keyless_scope_unknown", "keyless_scope_pattern"] {
        let err =
            RuntimeConfig::from_toml_str(&case_toml(case(name)).unwrap(), &root()).unwrap_err();
        assert!(
            err.to_string()
                .contains("auth.keyless_scope must name a configured auth scope"),
            "{name}: {err}"
        );
    }
    let config = RuntimeConfig::from_toml_str(
        &case_toml(case("keyless_scope_configured")).unwrap(),
        &root(),
    )
    .unwrap();
    assert_eq!(config.auth.keyless_scope.as_deref(), Some("dev"));
    assert!(live().auth.keyless_scope.is_none());
}

/// M5 D4: every reader refuses a case that patches the pre-M5 top-level `backend` table.
#[test]
fn fixture_case_with_patch_backend_is_refused() {
    let case = serde_json::json!({
        "name": "old_shape", "base": LIVE, "expect": "invalid", "reason": "r",
        "patch": {"backend": {"base_url": null}},
    });
    let err = case_toml(&case).unwrap_err();
    assert!(err.contains("patch.backend"), "{err}");
    let ok = serde_json::json!({
        "name": "new_shape", "base": LIVE, "expect": "invalid", "reason": "r",
        "patch": {}, "entry_patch": {"base_url": null},
    });
    let doc: toml::Table = toml::from_str(&case_toml(&ok).unwrap()).unwrap();
    assert!(doc["deployments"][0].get("base_url").is_none());
    assert!(doc["deployments"][0].get("served_model_name").is_some());
}

/// M5 D1: the pre-M5 `[backend]` table is an unknown field; there is no alias.
#[test]
fn backend_table_is_an_unknown_field() {
    let text = read(LIVE)
        .replace("[[deployments]]", "[backend]")
        .replace("[deployments.expect]", "[backend.expect]");
    assert!(
        text.contains("[backend]")
            && !text.contains("[deployments")
            && !text.contains("[[deployments")
    );
    let err = RuntimeConfig::from_toml_str(&text, &root()).unwrap_err();
    assert!(err.message().contains("unknown field `backend`"), "{err}");
}

/// M5 C2: the valid two-entry mock profile loads both entries in config order and hashes its raw document.
#[test]
fn multimodel_dev_round_trips_its_canonical_hash() {
    let rel = "configs/runtime/multimodel-dev.toml";
    let c = RuntimeConfig::load(root().join(rel)).unwrap();
    let names: Vec<_> = c
        .deployments
        .iter()
        .map(|d| d.served_model_name.as_deref().unwrap())
        .collect();
    assert_eq!(names, ["Qwen3.8-Flash-Next", "gemma-4-e4b-it"]);
    assert!(c.deployments.iter().all(|d| d.lock.is_some()));
    let table: toml::Table = toml::from_str(&read(rel)).unwrap();
    let document = serde_json::to_string(&serde_json::to_value(&table).unwrap()).unwrap();
    assert_eq!(c.config_hash(), sha256_hex(document.as_bytes()));
    assert_eq!(c.config_hash(), config_pins::config_hashes(&root())[rel]);
}

#[test]
fn unpatched_bases_round_trip_through_the_harness() {
    for base in [LIVE, "configs/runtime/dev-single-gpu.toml"] {
        let text = patched_toml(base, &serde_json::json!({}));
        RuntimeConfig::from_toml_str(&text, &root()).unwrap();
    }
    let text = patched_toml(LIVE, &serde_json::json!({}));
    let scratch = DeploymentRoot::new("unpatched", &text, &serde_json::json!({}));
    RuntimeConfig::from_toml_str(&text, &scratch.0).unwrap();
}

#[test]
fn live_config_pins() {
    let c = live();
    assert!(!c.planning_only);
    let b = &c.deployments[0];
    assert_eq!(b.kind, BackendKind::SglangHttp);
    assert_eq!(b.base_url.as_deref(), Some("http://127.0.0.1:18080"));
    assert_eq!(b.served_model_name.as_deref(), Some("Qwen3.8-Flash-Next"));
    assert_eq!(
        b.shared_backend_tenancy,
        SharedBackendTenancy::SharedAcknowledged
    );
    assert_eq!(c.admission.max_backend_inflight, 2);
    let expect = b.expect.as_ref().unwrap();
    assert_eq!(expect.max_running_requests, 4);
    assert_eq!(expect.context_length, 200_000);
    assert_eq!(expect.chat_template, None);
    assert_eq!(c.auth.scopes.len(), 1);
    assert_eq!(c.auth.scopes[0].token_env, "CERE_TOKEN_DEV");
    assert_eq!(c.ledger.state_dir, PathBuf::from("/var/lib/cere/sglang-t4"));
}

#[test]
fn live_config_exposes_deployment_identity() {
    let c = live();
    let d = c.deployments[0].lock.as_ref().unwrap();
    assert_eq!(d.recipe.recipe_id, "qwen38-flash-next");
    assert_eq!(d.profile.profile_id, c.deployments[0].profile_id);
    assert_eq!(d.profile.execution_mode, "backend_managed");
    let lock =
        std::fs::read(root().join("configs/deployments/qwen38-flash-next.lock.json")).unwrap();
    assert_eq!(d.lock_sha256, sha256_hex(&lock));
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .concat()
}

/// The five nullable expect keys besides `chat_template` may be absent (pins null); the config hash stays the
/// canonical JSON of the raw TOML document.
#[test]
fn expect_without_nullable_pins_loads_and_hashes_raw() {
    let patch = serde_json::json!({"entry": {"expect": {
        "quantization": null,
        "speculative_algorithm": null,
        "speculative_num_steps": null,
        "speculative_eagle_topk": null,
        "speculative_num_draft_tokens": null,
    }}});
    let text = patched_toml(LIVE, &patch);
    let config = RuntimeConfig::from_toml_str(&text, &root()).unwrap();
    let expect = config.deployments[0].expect.as_ref().unwrap();
    assert_eq!(expect.quantization, None);
    assert_eq!(expect.speculative_num_steps, None);
    let table: toml::Table = toml::from_str(&text).unwrap();
    let document = serde_json::to_string(&serde_json::to_value(&table).unwrap()).unwrap();
    assert_eq!(config.config_hash(), sha256_hex(document.as_bytes()));
}

#[test]
fn partial_speculative_pins_name_the_missing_keys() {
    let patch = serde_json::json!({"entry": {"expect": {
        "speculative_eagle_topk": null,
        "speculative_num_draft_tokens": null,
    }}});
    let err = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &patch), &root())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("speculative_eagle_topk") && err.contains("speculative_num_draft_tokens"),
        "{err}"
    );
    assert!(!err.contains("speculative_num_steps"), "{err}");
}

#[test]
fn dev_config_has_no_deployment() {
    let dev = RuntimeConfig::load(root().join("configs/runtime/dev-single-gpu.toml")).unwrap();
    assert!(dev.deployments[0].lock.is_none());
}

#[test]
fn config_hash_is_canonical() {
    let c = live();
    // Key order and comments do not matter; values do.
    let mut table: toml::Table = toml::from_str(&read(LIVE)).unwrap();
    let reordered =
        RuntimeConfig::from_toml_str(&toml::to_string(&table).unwrap(), &root()).unwrap();
    assert_eq!(c.config_hash(), reordered.config_hash());
    // The effective config serializes to exactly the TOML document, so `canonical_digest` over the
    // parsed TOML in scripts/cere_contracts.py yields the same hash.
    let document = serde_json::to_string(&serde_json::to_value(&table).unwrap()).unwrap();
    assert_eq!(c.config_hash(), sha256_hex(document.as_bytes()));
    table["api"]["max_messages"] = toml::Value::Integer(511);
    let changed = RuntimeConfig::from_toml_str(&toml::to_string(&table).unwrap(), &root()).unwrap();
    assert_ne!(c.config_hash(), changed.config_hash());
}

#[test]
fn two_scopes_need_and_accept_cross_scope_acknowledgement() {
    let patch = serde_json::json!({
        "entry": {"allow_cross_scope_prefix_cache": true},
        "auth": {"scopes": [
            {"scope_id": "dev", "token_env": "CERE_TOKEN_DEV"},
            {"scope_id": "ci", "token_env": "CERE_TOKEN_CI"}
        ]}
    });
    RuntimeConfig::from_toml_str(&patched_toml(LIVE, &patch), &root()).unwrap();
}

/// Q2/Q3: the per-scope cache salt is an alternative to acknowledging a shared radix cache.
#[test]
fn two_scopes_with_cache_salt_isolation_need_no_cross_scope_acknowledgement() {
    let two_scopes = serde_json::json!([
        {"scope_id": "dev", "token_env": "CERE_TOKEN_DEV"},
        {"scope_id": "ci", "token_env": "CERE_TOKEN_CI"}
    ]);
    let with = |isolation: Value| {
        let patch = serde_json::json!({
            "entry": {"allow_cross_scope_prefix_cache": false, "prefix_cache_isolation": isolation},
            "auth": {"scopes": two_scopes},
        });
        RuntimeConfig::from_toml_str(&patched_toml(LIVE, &patch), &root())
    };
    let salted = with(serde_json::json!("cache_salt")).unwrap();
    assert_eq!(
        salted.deployments[0].prefix_cache_isolation(),
        PrefixCacheIsolation::CacheSalt
    );
    let none = with(serde_json::json!("none")).unwrap_err();
    assert!(
        none.to_string().contains("prefix_cache_isolation"),
        "{none}"
    );
    // `null` removes the key in the merge patch: absent means `none`.
    with(Value::Null).unwrap_err();
    with(serde_json::json!("per_scope")).unwrap_err();
    // One scope: `none` is valid and explicit.
    let one = serde_json::json!({"entry": {"prefix_cache_isolation": "none"}});
    let explicit = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &one), &root()).unwrap();
    assert_eq!(
        explicit.deployments[0].prefix_cache_isolation,
        Some(PrefixCacheIsolation::None)
    );
}

/// Q3: the optional key is skipped when absent, so the t4 hash is the canonical JSON of its raw document (its pin in
/// tests/fixtures/config_pins.json), and stating `none` explicitly is a different config.
#[test]
fn t4_config_hash_is_unchanged_by_the_optional_isolation_key() {
    let live = live();
    assert_eq!(live.deployments[0].prefix_cache_isolation, None);
    assert_eq!(
        live.config_hash(),
        config_pins::config_hashes(&root())[LIVE]
    );
    let explicit = serde_json::json!({"entry": {"prefix_cache_isolation": "none"}});
    let explicit = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &explicit), &root()).unwrap();
    assert_ne!(explicit.config_hash(), live.config_hash());
}

/// m1close R1/R2: scope weight, aging and the resume bounds are optional; absent keys mean the documented
/// defaults and leave the `config_hash` unchanged (the t4 hash above), set keys load at their range bounds.
#[test]
fn scheduling_and_resume_keys_are_optional() {
    let live = live();
    assert_eq!(live.auth.scopes[0].weight, None);
    assert_eq!(live.auth.scopes[0].weight(), 1);
    assert_eq!(live.admission.aging_ms(), 30_000);
    assert_eq!(live.api.resume_buffer_bytes(), 4_194_304);
    assert_eq!(live.api.resume_buffer_total_bytes(), 67_108_864);
    assert_eq!(live.api.resume_retention_ms(), 60_000);
    let patch = serde_json::json!({
        "api": {
            "resume_buffer_bytes": 4096,
            "resume_buffer_total_bytes": 4_294_967_296u64,
            "resume_retention_ms": 3_600_000
        },
        "admission": {"aging_ms": 1000},
        "auth": {"scopes": [{"scope_id": "dev", "token_env": "CERE_TOKEN_DEV", "weight": 100}]}
    });
    let set = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &patch), &root()).unwrap();
    assert_eq!(set.auth.scopes[0].weight(), 100);
    assert_eq!(set.admission.aging_ms(), 1000);
    assert_eq!(set.api.resume_buffer_bytes(), 4096);
    assert_eq!(set.api.resume_buffer_total_bytes(), 4_294_967_296);
    assert_eq!(set.api.resume_retention_ms(), 3_600_000);
    assert_ne!(set.config_hash(), live.config_hash());
}

/// The resume total holds a full per-request buffer for every backend slot (`resume_buffer_total_bytes >=
/// resume_buffer_bytes x admission.max_backend_inflight`, effective defaults included), so live streams alone
/// never overflow it. The t4 profile runs 2 in flight.
#[test]
fn resume_total_covers_a_full_buffer_per_backend_slot() {
    let with = |api: Value| {
        RuntimeConfig::from_toml_str(
            &patched_toml(LIVE, &serde_json::json!({ "api": api })),
            &root(),
        )
    };
    let mib = 1_048_576u64;
    with(
        serde_json::json!({"resume_buffer_bytes": 32 * mib, "resume_buffer_total_bytes": 64 * mib}),
    )
    .unwrap();
    for api in [
        serde_json::json!({"resume_buffer_bytes": 32 * mib, "resume_buffer_total_bytes": 64 * mib - 1}),
        // the 4 MiB default per-request buffer
        serde_json::json!({"resume_buffer_total_bytes": 8 * mib - 1}),
        // the 64 MiB default total
        serde_json::json!({"resume_buffer_bytes": 32 * mib + 1}),
    ] {
        let err = with(api.clone()).unwrap_err();
        assert!(
            err.message().contains("max_backend_inflight"),
            "{api}: {err}"
        );
    }
    with(serde_json::json!({"resume_buffer_total_bytes": 8 * mib})).unwrap();
}

/// m1close R3 (Q6): the orphan load release is optional. Absent keys mean retention-only holds and a 5 s poll
/// period and leave the t4 `config_hash` unchanged; the bench profile alone sets `load_endpoint`.
#[test]
fn orphan_load_release_keys_are_optional() {
    let live = live();
    assert_eq!(live.deployments[0].load_endpoint, None);
    assert_eq!(live.ledger.orphan_load_poll_ms, None);
    assert_eq!(live.ledger.orphan_load_poll_ms(), 5000);
    let patch = serde_json::json!({
        "entry": {"load_endpoint": "/get_load"},
        "ledger": {"orphan_load_poll_ms": 1000}
    });
    let set = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &patch), &root()).unwrap();
    assert_eq!(
        set.deployments[0].load_endpoint.as_deref(),
        Some("/get_load")
    );
    assert_eq!(set.ledger.orphan_load_poll_ms(), 1000);
    assert_ne!(set.config_hash(), live.config_hash());
    let top = serde_json::json!({"ledger": {"orphan_load_poll_ms": 600_000}});
    let top = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &top), &root()).unwrap();
    assert_eq!(top.ledger.orphan_load_poll_ms(), 600_000);
    let bench = RuntimeConfig::load(root().join("configs/runtime/sglang-bench-g0.toml")).unwrap();
    assert_eq!(
        bench.deployments[0].load_endpoint.as_deref(),
        Some("/get_load")
    );
    assert_eq!(bench.ledger.orphan_load_poll_ms, None);
}

/// OH-03: `[ledger] evict_terminal_when_full` is optional. Absent means false and leaves the `config_hash`
/// unchanged; `prod-t4.toml` leaves it out (a rollback to a release without eviction cannot restore admission on a
/// full ledger), and the ledger limits carry it.
#[test]
fn ledger_eviction_flag_is_optional_and_reaches_the_ledger_limits() {
    let live = live();
    assert_eq!(live.ledger.evict_terminal_when_full, None);
    assert!(!LedgerLimits::from_config(&live).evict_terminal_when_full);
    let patch = serde_json::json!({"ledger": {"evict_terminal_when_full": true}});
    let set = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &patch), &root()).unwrap();
    assert!(LedgerLimits::from_config(&set).evict_terminal_when_full);
    assert_ne!(set.config_hash(), live.config_hash());
    let off = serde_json::json!({"ledger": {"evict_terminal_when_full": false}});
    let off = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &off), &root()).unwrap();
    assert!(!LedgerLimits::from_config(&off).evict_terminal_when_full);
    let prod = RuntimeConfig::load(root().join("configs/runtime/prod-t4.toml")).unwrap();
    assert_eq!(prod.ledger.evict_terminal_when_full, None);
}

#[test]
fn base_url_parse() {
    assert_eq!(
        parse_loopback_base_url("http://127.0.0.1:18080").unwrap(),
        ("127.0.0.1".to_owned(), 18080)
    );
    assert_eq!(
        parse_loopback_base_url("http://127.255.0.9:1").unwrap(),
        ("127.255.0.9".to_owned(), 1)
    );
    assert_eq!(
        parse_loopback_base_url("http://[::1]:65535").unwrap(),
        ("::1".to_owned(), 65535)
    );
    for bad in [
        "http://127.0.0.1:65536",
        "http://127.0.0.1:018080",
        "http://127.1:80",
        "http://[::1]",
        "http://[0:0:0:0:0:0:0:1]:80",
        "http://127.0.0.1:80\n",
        "http://127.0.0.1:８０",
    ] {
        assert!(parse_loopback_base_url(bad).is_err(), "{bad:?}");
    }
}

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
    let map: HashMap<String, OsString> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), OsString::from(v)))
        .collect();
    move |name| map.get(name).cloned()
}

#[test]
fn resolve_auth_reads_env_and_fails_closed() {
    let c = live();
    let creds = c
        .resolve_auth_with(env(&[("CERE_TOKEN_DEV", "secret-dev")]))
        .unwrap();
    assert_eq!(creds.len(), 1);
    assert_eq!(creds[0].scope_id, "dev");
    assert_eq!(creds[0].token.expose(), "secret-dev");
    assert!(!format!("{creds:?}").contains("secret-dev"));
    assert!(c.resolve_auth_with(env(&[])).is_err());
    assert!(c.resolve_auth_with(env(&[("CERE_TOKEN_DEV", "")])).is_err());
}

#[test]
fn resolve_auth_rejects_shared_token_values() {
    let patch = serde_json::json!({
        "entry": {"allow_cross_scope_prefix_cache": true},
        "auth": {"scopes": [
            {"scope_id": "dev", "token_env": "CERE_TOKEN_DEV"},
            {"scope_id": "ci", "token_env": "CERE_TOKEN_CI"}
        ]}
    });
    let c = RuntimeConfig::from_toml_str(&patched_toml(LIVE, &patch), &root()).unwrap();
    assert!(c
        .resolve_auth_with(env(&[
            ("CERE_TOKEN_DEV", "same"),
            ("CERE_TOKEN_CI", "same")
        ]))
        .is_err());
    assert_eq!(
        c.resolve_auth_with(env(&[("CERE_TOKEN_DEV", "a"), ("CERE_TOKEN_CI", "b")]))
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn load_outside_a_checkout_fails() {
    let dir = std::env::temp_dir().join(format!("cere-config-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("x.toml");
    std::fs::write(&path, read("configs/runtime/dev-single-gpu.toml")).unwrap();
    let err = RuntimeConfig::load(&path).unwrap_err();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(err.message().contains("repository"), "{err}");
}

#[test]
fn cered_check_config() {
    let ok = Command::new(env!("CARGO_BIN_EXE_cered"))
        .args(["--check-config", root().join(LIVE).to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(ok.status.code(), Some(0));
    let stdout = String::from_utf8(ok.stdout).unwrap();
    assert_eq!(
        stdout.lines().next().unwrap(),
        format!("config ok {}", live().config_hash())
    );
    assert_eq!(
        stdout.lines().nth(1).unwrap(),
        format!(
            "deployment served_model_name=Qwen3.8-Flash-Next recipe_id=qwen38-flash-next lock_sha256={} profile_id=sglang-t4-qwen38-flash-next-v1",
            live().deployments[0].lock.as_ref().unwrap().lock_sha256
        )
    );
    assert_eq!(stdout.lines().count(), 2);
    // M5 D10: one full `deployment` line per entry, in config order, each with its own entry's recipe, lock and
    // profile.
    let multi_path = root().join("configs/runtime/multimodel-g0g1.toml");
    let multi = Command::new(env!("CARGO_BIN_EXE_cered"))
        .args(["--check-config", multi_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(multi.status.code(), Some(0));
    let loaded = RuntimeConfig::load(&multi_path).unwrap();
    let expected: Vec<String> = [
        (
            "Qwen3.8-Flash-Next",
            "qwen38-flash-next",
            "sglang-t4-qwen38-flash-next-v1",
        ),
        (
            "gemma-4-e4b-it",
            "gemma-4-e4b-it",
            "sglang-gemma-4-e4b-it-v1",
        ),
    ]
    .iter()
    .zip(&loaded.deployments)
    .map(|((name, recipe, profile), entry)| {
        let d = entry.lock.as_ref().unwrap();
        assert_eq!(entry.served_model_name.as_deref(), Some(*name));
        assert_eq!(d.recipe.recipe_id, *recipe);
        assert_eq!(d.profile.profile_id, *profile);
        format!(
            "deployment served_model_name={name} recipe_id={recipe} lock_sha256={} profile_id={profile}",
            d.lock_sha256
        )
    })
    .collect();
    assert_ne!(
        loaded.deployments[0].lock.as_ref().unwrap().lock_sha256,
        loaded.deployments[1].lock.as_ref().unwrap().lock_sha256
    );
    let stdout = String::from_utf8(multi.stdout).unwrap();
    assert_eq!(stdout.lines().skip(1).collect::<Vec<_>>(), expected);
    let bad = Command::new(env!("CARGO_BIN_EXE_cered"))
        .args(["--check-config", root().join("README.md").to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
    assert!(!bad.stderr.is_empty());
}
