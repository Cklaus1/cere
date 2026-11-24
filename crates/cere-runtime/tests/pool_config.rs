//! Team pools (net-teleport design note 4.4, S6): the `[[federation.workers]]` owner/tier/owner_only keys, the
//! `[[auth.scopes]]` token_envs/class/quota keys and `[admission]` batch_floor/owner_wait_cap_ms. Each `pool_` row of
//! `tests/fixtures/runtime_config_invalid.json` is checked here for its Rust verdict and message;
//! `tests/test_contracts.py` checks the same rows against the Python mirror.
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use cere_runtime::auth::Authenticator;
use cere_runtime::config::RuntimeConfig;
use cere_runtime::pool::{self, Class};
use cere_runtime::tier::Tier;
use serde_json::Value;

const FEDERATION: &str = "configs/runtime/federation-dev.toml";

/// Every `pool_` row: its name and, for an `invalid` one, a needle of the Rust message (`None` for `valid`).
const POOL_CASES: [(&str, Option<&str>); 21] = [
    (
        "pool_worker_owner_unknown",
        Some("federation worker worker-a owner must name a configured auth scope"),
    ),
    (
        "pool_worker_owner_only_without_owner",
        Some("federation worker worker-a owner_only needs an owner"),
    ),
    ("pool_worker_owner_only_unknown_zone", Some("Mars/Olympus")),
    (
        "pool_worker_owner_only_overnight",
        Some("the start must be before the end (an overnight window is two entries)"),
    ),
    (
        "pool_worker_owner_only_bad_day",
        Some("days are Mon..Sun, ranges a-b or comma lists"),
    ),
    ("pool_worker_tier_unknown", Some("unknown variant `public`")),
    ("pool_worker_owned_team_window_valid", None),
    (
        "pool_worker_owner_with_smoke_scope",
        Some("auth scope cere-smoke is reserved for the remote readiness smoke"),
    ),
    (
        "pool_batch_floor_above_half",
        Some("admission.batch_floor must be within 0.0..=0.5"),
    ),
    (
        "pool_batch_floor_negative",
        Some("admission.batch_floor must be within 0.0..=0.5"),
    ),
    (
        "pool_owner_wait_cap_below_1s",
        Some("admission.owner_wait_cap_ms must be 1000..600000"),
    ),
    (
        "pool_owner_wait_cap_above_work_cap",
        Some("admission.owner_wait_cap_ms must be 1000..600000"),
    ),
    ("pool_admission_keys_valid", None),
    (
        "pool_token_envs_repeat_token_env",
        Some("one token_env cannot authenticate two scopes"),
    ),
    (
        "pool_token_envs_operator",
        Some("auth.operator_token_env must differ from every scope token_env"),
    ),
    (
        "pool_token_envs_empty",
        Some("auth.scopes.token_envs item count out of range"),
    ),
    (
        "pool_token_envs_lowercase",
        Some("auth.scopes.token_env does not match its pattern"),
    ),
    ("pool_class_unknown", Some("unknown variant `bulk`")),
    (
        "pool_per_scope_inflight_above_global",
        Some("per-scope in-flight cap exceeds the global backend cap"),
    ),
    (
        "pool_max_batch_inflight_zero",
        Some("auth.scopes.max_batch_inflight must be 1..4096"),
    ),
    ("pool_scope_keys_valid", None),
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// RFC 7396 JSON merge patch.
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

fn case_config(case: &Value) -> Result<RuntimeConfig, String> {
    let text = std::fs::read_to_string(root().join(case["base"].as_str().unwrap())).unwrap();
    let table: toml::Table = toml::from_str(&text).unwrap();
    let value = merge_patch(serde_json::to_value(table).unwrap(), &case["patch"]);
    RuntimeConfig::from_toml_str(&toml::to_string(&value).unwrap(), &root())
        .map_err(|e| e.message().to_owned())
}

fn pool_rows() -> Vec<Value> {
    let text =
        std::fs::read_to_string(root().join("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let rows: Vec<Value> = serde_json::from_str(&text).unwrap();
    rows.into_iter()
        .filter(|c| c["name"].as_str().unwrap().starts_with("pool_"))
        .collect()
}

#[test]
fn pool_rows_hit_their_rule() {
    let rows = pool_rows();
    let named: Vec<&str> = rows.iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(named, POOL_CASES.map(|(n, _)| n));
    for (name, needle) in POOL_CASES {
        let case = rows.iter().find(|c| c["name"] == name).unwrap();
        let verdict = case_config(case);
        match needle {
            None => {
                assert_eq!(case["expect"], "valid", "{name}");
                verdict.unwrap_or_else(|e| panic!("{name} refused: {e}"));
            }
            Some(needle) => {
                assert_eq!(case["expect"], "invalid", "{name}");
                let err = verdict.err().unwrap_or_else(|| panic!("{name} accepted"));
                assert!(err.contains(needle), "{name}: {err}");
            }
        }
    }
}

/// The valid rows resolve to the documented values; an absent key keeps its default.
#[test]
fn valid_rows_resolve() {
    let rows = pool_rows();
    let row = |name: &str| case_config(rows.iter().find(|c| c["name"] == name).unwrap()).unwrap();
    let owned = row("pool_worker_owned_team_window_valid");
    let worker = &owned.federation.as_ref().unwrap().workers[0];
    assert_eq!(worker.owner.as_deref(), Some("dev"));
    assert_eq!(worker.tier(), Tier::Team);
    assert_eq!(worker.pool().windows.len(), 2);

    let admission = row("pool_admission_keys_valid").admission;
    assert_ne!(admission.batch_floor(), pool::DEFAULT_BATCH_FLOOR);
    assert_ne!(
        admission.owner_wait_cap_ms(),
        pool::DEFAULT_OWNER_WAIT_CAP_MS
    );

    let scoped = row("pool_scope_keys_valid");
    let scope = &scoped.auth.scopes[0];
    assert_eq!(scope.class(), Class::Batch);
    assert_eq!(pool::class_of(&scoped, "dev", None), Class::Batch);
    assert_eq!(
        pool::class_of(&scoped, "dev", Some(Class::Interactive)),
        Class::Interactive
    );
    assert_eq!(pool::class_of(&scoped, "nobody", None), pool::DEFAULT_CLASS);
    assert_eq!(pool::session_caps(&scoped).get("dev").copied(), Some(8));

    let base = RuntimeConfig::load(root().join(FEDERATION)).unwrap();
    assert_eq!(base.admission.batch_floor(), pool::DEFAULT_BATCH_FLOOR);
    assert_eq!(
        base.admission.owner_wait_cap_ms(),
        pool::DEFAULT_OWNER_WAIT_CAP_MS
    );
    assert_eq!(base.auth.scopes[0].class(), pool::DEFAULT_CLASS);
    let w = &base.federation.as_ref().unwrap().workers[0];
    assert_eq!((w.owner.as_deref(), w.tier()), (None, Tier::Own));
    assert!(w.pool().windows.is_empty());
}

/// C14: `token_envs` adds device tokens that all authenticate the same person scope; `token_env` keeps working.
#[test]
fn token_envs_authenticate_one_scope() {
    let rows = pool_rows();
    let config = case_config(
        rows.iter()
            .find(|c| c["name"] == "pool_scope_keys_valid")
            .unwrap(),
    )
    .unwrap();
    let envs: Vec<&str> = config.auth.scopes[0].token_env_names().collect();
    assert_eq!(envs.len(), 3, "token_env, then token_envs");
    let lookup = |name: &str| {
        envs.iter()
            .position(|e| *e == name)
            .map(|i| OsString::from(format!("token-{i}")))
    };
    let credentials = config.resolve_auth_with(lookup).unwrap();
    assert_eq!(credentials.len(), 3);
    let auth = Authenticator::new(&credentials, None, None);
    for i in 0..3 {
        let header = format!("Bearer token-{i}");
        assert_eq!(auth.authenticate(Some(header.as_bytes())), Some("dev"));
    }
    assert_eq!(auth.authenticate(Some(b"Bearer token-3")), None);
    // A missing device token is a startup failure, naming the env var.
    let partial = |name: &str| (name == envs[0]).then(|| OsString::from("token-0"));
    let err = config.resolve_auth_with(partial).unwrap_err();
    assert!(err.message().contains(envs[1]), "{}", err.message());
}
