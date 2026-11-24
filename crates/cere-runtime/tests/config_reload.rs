//! The live-reload key classification (`cere_runtime::reload::KEYS`) against the schema and the shipped configs:
//! every key has one row, every change of a shipped config is refused exactly when its row is not `live`, a
//! shipped profile reloaded without change is a no-op, and `config_hash` of every shipped config stays the digest
//! of its TOML (the reload source never enters it).
use std::path::{Path, PathBuf};

use cere_runtime::config::RuntimeConfig;
use cere_runtime::reload::{self, KeyClass, KEYS};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Rows that are not schema leaves: the pinned deployment files behind `deployments[].deployment`, and a native
/// key the Rust config accepts that the schema does not list.
const NOT_IN_SCHEMA: [&str; 2] = ["deployments[].lock", "deployments[].native.keep_host_share"];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Every `configs/runtime/*.toml`, by file name.
fn shipped() -> Vec<(String, PathBuf)> {
    let mut paths: Vec<(String, PathBuf)> = std::fs::read_dir(root().join("configs/runtime"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p))
        .collect();
    paths.sort();
    assert!(paths.len() >= 10, "shipped configs: {paths:?}");
    paths
}

fn load(name: &str) -> RuntimeConfig {
    RuntimeConfig::load(root().join("configs/runtime").join(name))
        .unwrap_or_else(|e| panic!("{name} rejected: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .concat()
}

/// The schema's leaf paths (`[]` for an array of objects), as `crate::reload` names them.
fn schema_leaves() -> Vec<String> {
    fn walk(node: &Value, path: &str, out: &mut Vec<String>) {
        if let Some(props) = node.get("properties").and_then(Value::as_object) {
            for (key, child) in props {
                let next = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                walk(child, &next, out);
            }
        } else if node["type"] == "array" && node["items"].get("properties").is_some() {
            walk(&node["items"], &format!("{path}[]"), out);
        } else {
            out.push(path.to_owned());
        }
    }
    let text = std::fs::read_to_string(root().join("schemas/runtime_config.schema.json")).unwrap();
    let schema: Value = serde_json::from_str(&text).unwrap();
    let mut out = Vec::new();
    walk(&schema, "", &mut out);
    out
}

/// Concrete leaf paths of a serialized config with their JSON pointers (objects by key, arrays of objects by
/// index, anything else one leaf), the way `crate::reload` diffs.
fn leaves(value: &Value, path: &str, pointer: &str, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let next = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                leaves(child, &next, &format!("{pointer}/{key}"), out);
            }
        }
        Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_object) => {
            for (i, child) in items.iter().enumerate() {
                leaves(
                    child,
                    &format!("{path}[{i}]"),
                    &format!("{pointer}/{i}"),
                    out,
                );
            }
        }
        _ => out.push((path.to_owned(), pointer.to_owned())),
    }
}

/// Another value of the same JSON type; `None` for a null or an empty array.
fn mutate(value: &Value) -> Option<Value> {
    Some(match value {
        Value::Bool(b) => Value::Bool(!b),
        Value::Number(n) => Value::from(n.as_u64()? + 1),
        Value::String(s) => Value::String(format!("{s}x")),
        Value::Array(items) => {
            let mut items = items.clone();
            items.push(mutate(items.first()?)?);
            Value::Array(items)
        }
        _ => return None,
    })
}

#[test]
fn every_schema_key_has_exactly_one_row() {
    let schema = schema_leaves();
    for path in &schema {
        assert!(
            reload::rule(path).is_some(),
            "{path} has no reload::KEYS row"
        );
    }
    // m23 design note PI-02: the schema's `levers` is a map (names come from the lever registry), so its per-lever
    // rows are extras exactly when they name a registry lever; ab-split design note AB-05: an A/B arm's `levers`
    // likewise, for a runtime-layer registry lever.
    let lever_row = |path: &str| {
        let registered = |name: &str, runtime_only: bool| {
            cere_runtime::levers::REGISTRY.iter().any(|l| {
                l.name == name && (!runtime_only || l.layer == cere_runtime::levers::Layer::Runtime)
            })
        };
        path.strip_prefix("deployments[].native.levers.")
            .is_some_and(|name| registered(name, false))
            || path
                .strip_prefix("deployments[].native.ab.arms[].levers.")
                .is_some_and(|name| registered(name, true))
    };
    for row in KEYS {
        assert!(
            schema.iter().any(|p| p == row.path)
                || NOT_IN_SCHEMA.contains(&row.path)
                || lever_row(row.path),
            "{} is neither a schema key nor a listed extra",
            row.path
        );
    }
    let mut paths: Vec<&str> = KEYS.iter().map(|r| r.path).collect();
    paths.sort_unstable();
    paths.dedup();
    assert_eq!(paths.len(), KEYS.len(), "a KEYS path is listed twice");
}

#[test]
fn memory_identity_and_pin_keys_are_restart() {
    for path in [
        "server.bind",
        "deployments[].profile_id",
        "deployments[].deployment",
        "deployments[].lock",
        "deployments[].native.library",
        "deployments[].native.library_sha256",
        "deployments[].native.artifact_dir",
        "deployments[].native.kv_pool_tokens",
        "deployments[].native.device_uuid",
        "deployments[].native.replicas[].device",
        "deployments[].native.replicas[].device_uuid",
        "deployments[].native.speculative_draft_tokens",
        "deployments[].native.max_parked_sessions",
        "deployments[].native.snapshot_host_bytes",
        "deployments[].native.prefix_state_slots",
        "deployments[].native.decode_rounds_per_prefill_chunk",
        "deployments[].native.prefill_slice_ms",
        "deployments[].native.decode_share_pct",
        "deployments[].native.prefill_age_calls",
        "scheduler.devices",
        "api.resume_buffer_total_bytes",
        "api.max_request_body_bytes",
        "ledger.max_records",
        "auth.scopes[].token_env",
    ] {
        assert_eq!(
            reload::rule(path).unwrap().class,
            KeyClass::Restart,
            "{path}"
        );
    }
}

#[test]
fn shipped_profile_reload_without_change_is_a_noop() {
    for (name, path) in shipped() {
        let running = load(&name);
        let again = RuntimeConfig::load(&path).unwrap();
        assert_eq!(
            reload::diff(&running, &again),
            Vec::<String>::new(),
            "{name}"
        );
        assert_eq!(running.config_hash(), again.config_hash(), "{name}");
    }
}

#[test]
fn shipped_config_hash_is_the_digest_of_its_toml() {
    // `config_hash` pins (perf_smoke's one-hash-per-run, the `cere` identity) see the TOML document only: the
    // reload source is never serialized.
    for (name, path) in shipped() {
        let config = load(&name);
        assert!(config.source.is_some(), "{name}: load records its source");
        let table: toml::Table = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let document = serde_json::to_string(&serde_json::to_value(&table).unwrap()).unwrap();
        assert_eq!(
            config.config_hash(),
            sha256_hex(document.as_bytes()),
            "{name}"
        );
    }
}

#[test]
fn every_leaf_change_of_a_shipped_config_follows_its_row() {
    let mut checked = 0;
    for name in [
        "native-g0-spec.toml",
        "sglang-t4-qwen38.toml",
        "federation-dev.toml",
        "multimodel-dev.toml",
    ] {
        let base_value = serde_json::to_value(load(name)).unwrap();
        let base: RuntimeConfig = serde_json::from_value(base_value.clone()).unwrap();
        let mut found = Vec::new();
        leaves(&base_value, "", "", &mut found);
        for (path, pointer) in found {
            let Some(changed) = base_value.pointer(&pointer).and_then(mutate) else {
                continue;
            };
            let mut value = base_value.clone();
            *value.pointer_mut(&pointer).unwrap() = changed;
            // A value outside the key's type (an enum's unknown variant) is not a config.
            let Ok(candidate) = serde_json::from_value::<RuntimeConfig>(value) else {
                continue;
            };
            assert_eq!(
                reload::diff(&base, &candidate),
                vec![path.clone()],
                "{name}"
            );
            let refused = reload::refused(std::slice::from_ref(&path));
            let class = reload::rule(&path)
                .unwrap_or_else(|| panic!("{name}: {path} has no row"))
                .class;
            assert_eq!(
                refused.is_empty(),
                class == KeyClass::Live,
                "{name}: {path} ({}) refused as {refused:?}",
                class.as_str()
            );
            checked += 1;
        }
    }
    assert!(checked > 150, "only {checked} leaf changes checked");
}

/// `name`'s config with one edit: (diff, refused).
fn change(name: &str, edit: impl FnOnce(&mut RuntimeConfig)) -> (Vec<String>, Vec<String>) {
    let running = load(name);
    let mut candidate = running.clone();
    edit(&mut candidate);
    let changed = reload::diff(&running, &candidate);
    let refused = reload::refused(&changed);
    (changed, refused)
}

#[test]
fn native_memory_and_pin_changes_are_refused() {
    type Edit = fn(&mut RuntimeConfig);
    let cases: [(&str, Edit); 16] = [
        ("deployments[0].native.kv_pool_tokens", |c| {
            let n = c.deployments[0].native.as_mut().unwrap();
            n.kv_pool_tokens = Some(n.kv_pool_tokens.unwrap_or(0) + 1);
        }),
        ("deployments[0].native.library", |c| {
            c.deployments[0].native.as_mut().unwrap().library = "/opt/cere/other.so".into();
        }),
        ("deployments[0].native.library_sha256", |c| {
            c.deployments[0].native.as_mut().unwrap().library_sha256 = "0".repeat(64);
        }),
        ("deployments[0].native.artifact_dir", |c| {
            c.deployments[0].native.as_mut().unwrap().artifact_dir = "/var/lib/other".into();
        }),
        ("deployments[0].native.device_uuid", |c| {
            c.deployments[0].native.as_mut().unwrap().device_uuid = Some("GPU-other".into());
        }),
        ("deployments[0].native.max_context_tokens", |c| {
            c.deployments[0].native.as_mut().unwrap().max_context_tokens -= 1;
        }),
        ("deployments[0].native.speculative_draft_tokens", |c| {
            c.deployments[0]
                .native
                .as_mut()
                .unwrap()
                .speculative_draft_tokens = 0;
        }),
        ("deployments[0].native.levers.drafter", |c| {
            let n = c.deployments[0].native.as_mut().unwrap();
            n.levers.insert("drafter".into(), "lookup".into());
        }),
        ("deployments[0].native.executor_memory_limit_bytes", |c| {
            c.deployments[0]
                .native
                .as_mut()
                .unwrap()
                .executor_memory_limit_bytes = Some(1 << 30);
        }),
        (
            "deployments[0].native.decode_rounds_per_prefill_chunk",
            |c| {
                c.deployments[0]
                    .native
                    .as_mut()
                    .unwrap()
                    .decode_rounds_per_prefill_chunk = Some(4);
            },
        ),
        ("deployments[0].native.prefill_slice_ms", |c| {
            let n = c.deployments[0].native.as_mut().unwrap();
            n.prefill_slice_ms = Some(n.prefill_slice_ms.unwrap_or(0) + 50);
        }),
        ("deployments[0].native.decode_share_pct", |c| {
            c.deployments[0].native.as_mut().unwrap().decode_share_pct = Some(100);
        }),
        ("deployments[0].native.prefill_age_calls", |c| {
            c.deployments[0].native.as_mut().unwrap().prefill_age_calls = Some(8);
        }),
        ("deployments[0].profile_id", |c| {
            c.deployments[0].profile_id = "other-profile".into();
        }),
        ("deployments[0].lock", |c| {
            c.deployments[0].lock.as_mut().unwrap().lock_sha256 = "0".repeat(64);
        }),
        ("server.bind", |c| c.server.bind = "127.0.0.1:1".into()),
    ];
    for (key, edit) in cases {
        let (changed, refused) = change("native-g0-spec.toml", edit);
        assert_eq!(changed, vec![key.to_owned()]);
        assert_eq!(refused, vec![key.to_owned()]);
    }
}

#[test]
fn native_live_changes_are_accepted() {
    let (changed, refused) = change("native-g0-spec.toml", |c| {
        c.admission.queue_depth += 1;
        c.api.stall_timeout_ms += 1;
        c.auth.scopes[0].weight = Some(3);
    });
    assert_eq!(
        changed,
        [
            "admission.queue_depth",
            "api.stall_timeout_ms",
            "auth.scopes[0].weight",
        ]
    );
    assert_eq!(refused, Vec::<String>::new());
}

#[test]
#[ignore = "reads the operator runbook, which is not published"]
fn runbook_table_is_keys() {
    let text = std::fs::read_to_string(root().join("docs/RUNBOOK.md")).unwrap();
    let start = text
        .find("<!-- reload-keys:begin -->")
        .expect("RUNBOOK reload table");
    let end = text
        .find("<!-- reload-keys:end -->")
        .expect("RUNBOOK reload table end");
    let rows: Vec<(String, String, String)> = text[start..end]
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(|l| {
            let cells: Vec<&str> = l.trim_matches('|').split(" | ").map(str::trim).collect();
            assert_eq!(cells.len(), 3, "{l}");
            (
                cells[0].trim_matches('`').to_owned(),
                cells[1].to_owned(),
                cells[2].to_owned(),
            )
        })
        .collect();
    let keys: Vec<(String, String, String)> = KEYS
        .iter()
        .map(|r| {
            (
                r.path.to_owned(),
                r.class.as_str().to_owned(),
                r.why.to_owned(),
            )
        })
        .collect();
    assert_eq!(
        rows, keys,
        "docs/RUNBOOK.md \"Live config reload\" table != reload::KEYS"
    );
}

/// m23 design note PI-02: every registry lever is a config key, so each has a KEYS row (restart: the executor takes
/// the selection at load; the hot ones swap through the dev lever route, never a reload).
#[test]
fn every_registry_lever_has_a_restart_row() {
    for lever in cere_runtime::levers::REGISTRY {
        let path = format!("deployments[0].native.levers.{}", lever.name);
        let row = reload::rule(&path).unwrap_or_else(|| panic!("no KEYS row for {path}"));
        assert_eq!(row.class, KeyClass::Restart, "{path}");
    }
}

/// ab-split design note AB-05: every runtime-layer lever may be an A/B arm's choice, and an arm's choices are
/// live (a request takes its arm from the settings it was accepted under; only new sessions see a change); no
/// executor-layer lever has an arm row (it is process-global).
#[test]
fn every_runtime_lever_has_a_live_arm_row() {
    for lever in cere_runtime::levers::REGISTRY {
        let path = format!("deployments[0].native.ab.arms[1].levers.{}", lever.name);
        match lever.layer {
            cere_runtime::levers::Layer::Runtime => {
                let row = reload::rule(&path).unwrap_or_else(|| panic!("no KEYS row for {path}"));
                assert_eq!(row.class, KeyClass::Live, "{path}");
            }
            cere_runtime::levers::Layer::Executor => {
                assert!(reload::rule(&path).is_none(), "{path}")
            }
        }
    }
    for key in ["salt", "arms[0].id", "arms[2].weight", "arms[0].levers"] {
        let path = format!("deployments[3].native.ab.{key}");
        assert_eq!(
            reload::rule(&path).map(|r| r.class),
            Some(KeyClass::Live),
            "{path}"
        );
    }
}
