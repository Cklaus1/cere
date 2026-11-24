//! The native backend kind (m22 design note R4): `native_cuda` with mode `native` and a `[deployments.native]`
//! table. The shared invalid table (`tests/fixtures/runtime_config_invalid.json`, run by `config_rev2.rs` and
//! tests/test_contracts.py) carries every refused native shape; this file covers the shipped profile, the
//! unchanged hashes of the existing configs and the refusals the spec names.
mod config_pins;

use std::path::{Path, PathBuf};

use cere_runtime::config::{
    production_marker_uuid, BackendKind, BackendMode, LoraConfig, PrefixCacheIsolation,
    ResearchConfig, RuntimeConfig,
};
use cere_runtime::levers::{
    DraftPolicy, Drafter, GrammarDrafts, K2Gemm, K2Weights, LeverSet, MoeGemm, PrefillAttention,
    SdarGemm,
};
use serde_json::{json, Value};

const NATIVE: &str = "configs/runtime/native-g0.toml";
/// The opt-in prefix-cache profile (m210 design note D9C): native-g0 plus the cache, profile `-v11`.
const NATIVE_PREFIX: &str = "configs/runtime/native-g0-prefix.toml";
/// m29 design note D11/D12 (M29S2-16): the speculation profile `-v19` (4 MTP draft tokens; owner decision
/// 2026-10-15: its speculation levers incl. k2_gemm small_m and drafter mtp+lookup, and the prefix cache), the NVFP4
/// opt-in since the FP4-Local flip (fp4local design note FL-31).
const SPEC: &str = "configs/runtime/native-g0-spec-nvfp4.toml";
/// m29l design note D7: the opt-in `-v15` profile, `SPEC` with the default `drafter` lever "mtp" (M1 ship-mtp).
const SPEC_LOOKUP: &str = "configs/runtime/native-g0-spec-lookup.toml";
/// Fp8W: the opt-in `-v20` profile, `SPEC` plus the numerics-changing `k2_weights = "fp8"`.
const SPEC_FP8W: &str = "configs/runtime/native-g0-spec-fp8w.toml";
/// m32n design note D1: the two-card deployment (`-v19`'s native table, two `[[deployments.native.replicas]]`).
const NATIVE_2GPU: &str = "configs/runtime/native-2gpu.toml";
/// fp4local design note FL-13 (D12, D16): the S2 dev profile `-v19-fp4local`, `SPEC` on the converted FP4-Local
/// artifact (artifact profile `rtxpro6000bw-plehost-nvfp4-text-v1`, `configs/artifacts/qwen38-flash-next-fp4local-cere.pin.json`).
const SPEC_FP4LOCAL: &str = "configs/runtime/native-g0-spec-fp4local.toml";
/// fp4local design note S3 (D12, FL-30): the opt-in S3 profile `-v21` (id tentative), `SPEC_FP4LOCAL` on the S3
/// artifact (artifact profile `rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1`,
/// `configs/artifacts/qwen38-flash-next-fp4local-cere-s3.pin.json`).
const SPEC_FP4LOCAL_S3: &str = "configs/runtime/native-g0-spec-fp4local-s3.toml";
/// Headroom's `-v22` (m210 design note D15⁗′ O112; the default from the FP4-Local flip, fp4local design note
/// FL-31, to the DraftTrain flip): `SPEC_FP4LOCAL_S3` with the GPU0 memory S3 frees spent on 32 prefix state slots,
/// prefix end states and a 300,032-token KV pool; opt-in since the DraftTrain flip.
const SPEC_V22: &str = "configs/runtime/native-g0-spec-v22.toml";
/// The default speculation profile since the DraftTrain flip (v29default design note VD-03): `-v29`, `SPEC_V22` on
/// the trained-draft artifact variant.
const SPEC_DEFAULT: &str = "configs/runtime/native-g0-spec.toml";
/// TreeSpec (tree-spec design note TS-03): the opt-in `-v28`, `SPEC_V22` with levers.drafter "mtp+lookup+long".
const SPEC_TREE: &str = "configs/runtime/native-g0-spec-tree.toml";
/// Chains (chains design note): the opt-in `-v36`, `SPEC_TREE` with levers.drafter "mtp+lookup+chain".
const SPEC_CHAINS: &str = "configs/runtime/native-g0-spec-chains.toml";
/// Sessions (sessions design note O211): the opt-in `-v45`, `SPEC_DEFAULT` plus the M2.10 prefix-cache host tier.
const SESSIONS: &str = "configs/runtime/native-g0-sessions.toml";
/// Ctx262 (ctx262 design note CX-02): the opt-in `-v44`, `SPEC_DEFAULT` at a 262,144-token context with a
/// 524,288-token KV pool.
const SPEC_262K: &str = "configs/runtime/native-g0-spec-262k.toml";
/// PrefixHit (m210 design note D17): the opt-in `-v51`, `SPEC_DEFAULT` with `prefix_alt_edges = true`.
const SPEC_V51: &str = "configs/runtime/native-g0-spec-v51.toml";
/// Track LoRA: the opt-in `-v50`, `SPEC_DEFAULT` with `[deployments.native.lora]`.
const SPEC_LORA: &str = "configs/runtime/native-g0-spec-lora.toml";
/// WeightLab (weightlab design note WL-01): the dev-only `-v57`, `SPEC_LORA` with `[deployments.native.research]`.
const SPEC_RESEARCH: &str = "configs/runtime/native-g0-spec-research.toml";
/// Ctx262 CX-09 (ctx262 design note): `-v52`, `SPEC_262K` with `prefix_alt_edges = true` (production's base).
const SPEC_V52: &str = "configs/runtime/native-g0-spec-v52.toml";
/// MixedPD (mixed-pd design note): the opt-in `-v47`, `SPEC_DEFAULT` with prefill_slice_ms 200.
const SPEC_V47: &str = "configs/runtime/native-g0-spec-v47.toml";
/// BeatSglang S7 (beat-sglang design note S7): the opt-in `-v56`, `SPEC_DEFAULT` with levers k2_gemm "small_m_a8"
/// and prefill_attention "p4" (numerics-changing prefill levers).
const SPEC_V56: &str = "configs/runtime/native-g0-spec-v56.toml";
/// Sessions SE-11 (sessions design note): the opt-in `-v53`, `SPEC_V52` plus the `-v45` host tier.
const SESSIONS_V53: &str = "configs/runtime/native-g0-sessions-v53.toml";
/// BatchConcurrency BC4 (batch-concurrency design note BC4): the opt-in `-v59`, `SESSIONS_V53` with MixedPD's
/// prefill_slice_ms 50.
const SESSIONS_V59: &str = "configs/runtime/native-g0-sessions-v59.toml";
/// BatchConc BC2 (batch-concurrency design note BC2): the dev `-v58`, production's composition (`prod-t4.toml`,
/// `-v53`) on the GPU0 lane at 16 decode rows (owner Q1 (b), Q8) with a smaller KV pool (Q2 (a)).
const ROWS16_V58: &str = "configs/runtime/native-g0-rows16-v58.toml";
/// VisionLong (m213l design note): the opt-in `-v43`, `SPEC_DEFAULT` on the vision artifact with sidecar feature
/// `image`, and its dev-only diagnostic `-v54`, `SPEC_VISION` with `prefix_alt_edges = true` (Amendment A2).
const SPEC_VISION: &str = "configs/runtime/native-g0-spec-vision.toml";
const SPEC_VISION_V54: &str = "configs/runtime/native-g0-spec-vision-v54.toml";
/// Ctx262 CX-R-vision (ctx262 design note CX-R18): the dev-only diagnostic `-v55`, `SPEC_VISION_V54` at a
/// 262,144-token context with `-v44`'s 524,288-token KV pool and a raised executor memory limit.
const SPEC_VISION_V55: &str = "configs/runtime/native-g0-spec-vision-v55.toml";
/// GrammarSpec (grammar-spec design note): the opt-in `-v49`, `SPEC_DEFAULT` with levers.grammar_drafts "on".
const SPEC_GRAMMAR: &str = "configs/runtime/native-g0-spec-grammar.toml";
/// `SPEC_DEFAULT` `[deployments.native]` keys BeatSglang flipped on the default after the opt-in profiles were derived
/// from it (beat-sglang design note S5, OQ2/OQ3; S1 C4Sched G4): (key, value before (`null` = absent), value now), as
/// JSON.
const SPEC_DEFAULT_FLIPS: &[(&str, &str, &str)] = &[
    ("prefix_state_slots", "32", "48"),
    ("prefill_slice_ms", "null", "200"),
    ("decode_share_pct", "null", "200"),
];

/// `SPEC_DEFAULT` with `SPEC_DEFAULT_FLIPS` reverted, after asserting each holds its new value: the default the derived
/// opt-in profiles still equal.
fn spec_default_before_flips() -> RuntimeConfig {
    let now = load(SPEC_DEFAULT);
    let native = serde_json::to_value(now.deployments[0].native.as_ref().unwrap()).unwrap();
    let mut revert = serde_json::Map::new();
    for (key, before, flipped) in SPEC_DEFAULT_FLIPS {
        let flipped: Value = serde_json::from_str(flipped).unwrap();
        assert_eq!(native[*key], flipped, "{key}");
        revert.insert(key.to_string(), serde_json::from_str(before).unwrap());
    }
    patched(SPEC_DEFAULT, &json!({"entry": {"native": revert}}))
        .unwrap_or_else(|e| panic!("{SPEC_DEFAULT} before its flips rejected: {e}"))
}

/// The `[deployments.native.levers]` table of `SPEC` (`-v19`, owner decision 2026-10-15).
fn spec_levers() -> Value {
    json!({
        "decode_step": "spec_graph",
        "verify_attention": "batched",
        "draft_policy": "adaptive",
        "k2_gemm": "small_m",
        "drafter": "mtp+lookup"
    })
}

/// A `patched` levers patch on `SPEC` that first drops its own levers table (the lever-free `-v19` shape, the old
/// `-v10` one but for the prefix-cache keys), so `value` is the whole resulting table.
fn bare_spec_levers(value: Value) -> Value {
    let mut table = json!({
        "decode_step": null,
        "verify_attention": null,
        "draft_policy": null,
        "k2_gemm": null,
        "drafter": null
    });
    if let (Some(t), Some(v)) = (table.as_object_mut(), value.as_object()) {
        t.extend(v.clone());
    }
    json!({"entry": {"native": {"levers": table}}})
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn load(rel: &str) -> RuntimeConfig {
    RuntimeConfig::load(root().join(rel)).unwrap_or_else(|e| panic!("{rel} rejected: {e}"))
}

/// `base` with `patch` deep-merged (objects merge, `null` deletes, anything else replaces), then `patch["entry"]`
/// deep-merged into `deployments[0]` (the fixture table's `entry_patch`, M5 D4).
fn patched(base: &str, patch: &Value) -> Result<RuntimeConfig, String> {
    fn merge(target: &mut Value, patch: &Value) {
        match (target, patch) {
            (Value::Object(t), Value::Object(p)) => {
                for (k, v) in p {
                    if v.is_null() {
                        t.remove(k);
                    } else {
                        merge(t.entry(k.clone()).or_insert(Value::Null), v);
                    }
                }
            }
            (t, p) => *t = p.clone(),
        }
    }
    let text = std::fs::read_to_string(root().join(base)).unwrap();
    let mut doc = serde_json::to_value(toml::from_str::<toml::Table>(&text).unwrap()).unwrap();
    let mut patch = patch.clone();
    let entry = patch.as_object_mut().unwrap().remove("entry");
    merge(&mut doc, &patch);
    if let Some(entry) = entry {
        merge(&mut doc["deployments"][0], &entry);
    }
    RuntimeConfig::from_toml_str(&toml::to_string(&doc).unwrap(), &root())
        .map_err(|e| e.to_string())
}

#[test]
fn native_profile_loads_with_its_deployment() {
    let config = load(NATIVE);
    let b = &config.deployments[0];
    assert_eq!(
        (b.mode, b.kind),
        (BackendMode::Native, BackendKind::NativeCuda)
    );
    assert!(b.base_url.is_none() && b.expect.is_none() && b.load_endpoint.is_none());
    let native = b.native.as_ref().expect("[deployments.native]");
    let d = b.lock.as_ref().expect("deployment");
    assert_eq!(d.lock.backend_build_sha256, native.library_sha256);
    assert_eq!(d.lock.artifact_sha256, native.expected_manifest_sha256);
    assert_eq!(d.profile.max_context_tokens, native.max_context_tokens);
    assert_eq!(d.profile.execution_mode, "native");
    assert_eq!(native.max_context_tokens, 200_000);
    assert_eq!(native.kv_pool_tokens, Some(200_000));
    assert_eq!(config.server.bind.to_string(), "127.0.0.1:18309");
}

/// m1close Q3: the optional `native` key is skipped when absent, so adding a kind never moves an existing
/// `config_hash`: each equals its pin (tests/fixtures/config_pins.json, the Python `canonical_digest` of the raw TOML).
#[test]
fn existing_configs_keep_their_config_hash() {
    let pins = config_pins::config_hashes(&root());
    for rel in [
        "configs/runtime/dev-single-gpu.toml",
        "configs/runtime/dev-two-replica.toml",
        "configs/runtime/sglang-bench-g0.toml",
        "configs/runtime/sglang-t4-qwen38.toml",
    ] {
        assert_eq!(load(rel).config_hash(), pins[rel], "{rel}");
    }
}

/// m210 design note D15⁗′ stage 2a: `prefix_end_state` is skipped when false, so the `-v19` speculation profile and
/// the cache-off profiles keep a `config_hash` without the key: the hash is the canonical JSON of the raw TOML document
/// (`scripts/cere_contracts.py` `canonical_digest`), which has no such key, and an explicit `false` hashes as absent.
/// Derived from the committed files rather than pinned: a relock moves these hashes with `library_sha256`, and
/// tests/fixtures/config_pins.json (`scripts/pin_configs.py`) is the deliberate pin of each value.
#[test]
fn prefix_end_state_off_keeps_mains_config_hash() {
    for rel in [
        "configs/runtime/native-g0-spec-nvfp4.toml",
        "configs/runtime/native-g0.toml",
        "configs/runtime/native-g1.toml",
    ] {
        let config = load(rel);
        assert!(config
            .deployments
            .iter()
            .all(|d| d.native.as_ref().is_none_or(|n| !n.prefix_end_state)));
        let text = std::fs::read_to_string(root().join(rel)).unwrap();
        assert!(!text.contains("prefix_end_state"), "{rel}");
        let table: toml::Table = toml::from_str(&text).unwrap();
        let document = serde_json::to_string(&serde_json::to_value(&table).unwrap()).unwrap();
        assert_eq!(
            config.config_hash(),
            sha256_hex(document.as_bytes()),
            "{rel}"
        );
        let explicit = patched(
            rel,
            &json!({"entry": {"native": {"prefix_end_state": false}}}),
        )
        .unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert_eq!(explicit.config_hash(), config.config_hash(), "{rel}");
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .concat()
}

/// D15⁗′ stage 2a: `prefix_end_state = true` is refused without the prefix cache and loads on a prefix profile, where
/// it is part of the `config_hash`.
#[test]
fn prefix_end_state_needs_the_prefix_cache() {
    let err = patched(
        NATIVE,
        &json!({"entry": {"native": {"prefix_end_state": true}}}),
    )
    .unwrap_err();
    assert!(
        err.contains("native.prefix_end_state requires prefix_state_slots > 0"),
        "{err}"
    );
    let off = patched(
        NATIVE_PREFIX,
        &json!({"entry": {"native": {"prefix_end_state": null}}}),
    )
    .unwrap();
    let on = patched(
        NATIVE_PREFIX,
        &json!({"entry": {"native": {"prefix_end_state": true}}}),
    )
    .unwrap();
    let native = |c: &RuntimeConfig| c.deployments[0].native.as_ref().unwrap().prefix_end_state;
    assert!(!native(&off) && native(&on));
    assert_ne!(off.config_hash(), on.config_hash());
}

/// m210 design note D17: `prefix_alt_edges = true` is refused without the prefix cache and loads on a prefix profile,
/// where it is part of the `config_hash`; absent, the key is skipped (every existing config keeps its hash).
#[test]
fn prefix_alt_edges_needs_the_prefix_cache() {
    let err = patched(
        NATIVE,
        &json!({"entry": {"native": {"prefix_alt_edges": true}}}),
    )
    .unwrap_err();
    assert!(
        err.contains("native.prefix_alt_edges requires prefix_state_slots > 0"),
        "{err}"
    );
    let off = patched(
        NATIVE_PREFIX,
        &json!({"entry": {"native": {"prefix_alt_edges": null}}}),
    )
    .unwrap();
    let on = patched(
        NATIVE_PREFIX,
        &json!({"entry": {"native": {"prefix_alt_edges": true}}}),
    )
    .unwrap();
    let native = |c: &RuntimeConfig| c.deployments[0].native.as_ref().unwrap().prefix_alt_edges;
    assert!(!native(&off) && native(&on));
    assert_ne!(off.config_hash(), on.config_hash());
    assert_eq!(
        off.config_hash(),
        patched(
            NATIVE_PREFIX,
            &json!({"entry": {"native": {"prefix_alt_edges": false}}})
        )
        .unwrap()
        .config_hash()
    );
}

/// Audit N1: `sidecar_deadline_ms` is 0 (the default) or 1 s to 1 h; set, it is part of the `config_hash`, absent or
/// 0 it is skipped (every existing config keeps its hash).
#[test]
fn sidecar_deadline_is_bounded_and_skipped_when_unset() {
    let deadline = |c: &RuntimeConfig| {
        c.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .sidecar_deadline_ms
    };
    let unset = patched(
        NATIVE,
        &json!({"entry": {"native": {"sidecar_deadline_ms": null}}}),
    )
    .unwrap();
    let zero = patched(
        NATIVE,
        &json!({"entry": {"native": {"sidecar_deadline_ms": 0}}}),
    )
    .unwrap();
    let set = patched(
        NATIVE,
        &json!({"entry": {"native": {"sidecar_deadline_ms": 30000}}}),
    )
    .unwrap();
    assert_eq!(
        (deadline(&unset), deadline(&zero), deadline(&set)),
        (0, 0, 30_000)
    );
    assert_eq!(unset.config_hash(), zero.config_hash());
    assert_ne!(unset.config_hash(), set.config_hash());
    for edge in [1_000, 3_600_000] {
        patched(
            NATIVE,
            &json!({"entry": {"native": {"sidecar_deadline_ms": edge}}}),
        )
        .unwrap();
    }
    for bad in [999, 3_600_001] {
        let err = patched(
            NATIVE,
            &json!({"entry": {"native": {"sidecar_deadline_ms": bad}}}),
        )
        .unwrap_err();
        assert!(err.contains("native.sidecar_deadline_ms"), "{bad}: {err}");
    }
}

#[test]
fn native_refuses_base_url_and_expect() {
    let err = patched(
        NATIVE,
        &json!({"entry": {"base_url": "http://127.0.0.1:18308"}}),
    )
    .unwrap_err();
    assert!(err.contains("native_cuda refuses base_url"), "{err}");
    let expect = load("configs/runtime/sglang-bench-g0.toml").deployments[0]
        .expect
        .clone();
    let err = patched(NATIVE, &json!({"entry": {"expect": expect}})).unwrap_err();
    assert!(
        err.contains("native_cuda refuses base_url, expect"),
        "{err}"
    );
}

#[test]
fn native_mode_and_native_kind_imply_each_other() {
    let err = patched(NATIVE, &json!({"entry": {"mode": "backend_managed"}})).unwrap_err();
    assert!(
        err.contains("mode native requires kind native_cuda"),
        "{err}"
    );
    for base in [
        "configs/runtime/sglang-t4-qwen38.toml",
        "configs/runtime/dev-single-gpu.toml",
    ] {
        let err = patched(base, &json!({"entry": {"mode": "native"}})).unwrap_err();
        assert!(
            err.contains("mode native requires kind native_cuda"),
            "{base}: {err}"
        );
    }
}

#[test]
fn native_table_is_native_cuda_only() {
    let native = serde_json::to_value(&load(NATIVE).deployments[0].native).unwrap();
    let err = patched(
        "configs/runtime/sglang-t4-qwen38.toml",
        &json!({"entry": {"native": native}}),
    )
    .unwrap_err();
    assert!(
        err.contains("deployments[0].native is native_cuda-only"),
        "{err}"
    );
}

/// The lock pins the library and the manifest; a config that disagrees with it never loads.
#[test]
fn native_identity_must_match_the_lock() {
    for (field, needle) in [
        ("library_sha256", "backend_build_sha256 differs"),
        ("expected_manifest_sha256", "artifact_sha256 differs"),
    ] {
        let err = patched(
            NATIVE,
            &json!({"entry": {"native": {field: "0".repeat(64)}}}),
        )
        .unwrap_err();
        assert!(err.contains(needle), "{field}: {err}");
    }
    let err = patched(
        NATIVE,
        &json!({"entry": {"native": {"max_context_tokens": 4096}}}),
    )
    .unwrap_err();
    assert!(err.contains("deployment context differs"), "{err}");
}

/// M2.11 C6 (m211 design note D8; ledger C 56, LR5): the shipped profile has no host-RAM arena and no preemption,
/// and declares `native_state_snapshot: false`.
#[test]
fn shipped_native_profile_has_no_snapshot_arena() {
    let config = load(NATIVE);
    let b = &config.deployments[0];
    let native = b.native.as_ref().expect("[deployments.native]");
    assert_eq!(
        (
            native.snapshot_host_bytes,
            native.preempt,
            native.preempt_min_tokens,
            native.max_parked_sessions,
            native.preempt_every,
        ),
        (0, false, 512, 0, 0)
    );
    assert!(!b.lock.as_ref().unwrap().profile.native_state_snapshot);
}

/// The D8 config bound's snapshot size (kinds 0-8 at `position` = `token_count` = the context; kind 10 too on a model
/// with drafts), pinned at the same positions as the C side (`executor/tests/host/test_state.cpp` C3 case).
#[test]
fn max_snapshot_bytes_matches_the_executor_plan() {
    use cere_runtime::native::snapshot::max_snapshot_bytes;
    for (tokens, bytes, drafted) in [
        (1, 60_028_928, 60_030_224),
        (5, 60_081_152, 60_086_800),
        (73_728, 1_022_900_992, 1_103_117_072),
    ] {
        assert_eq!(max_snapshot_bytes(tokens, false), Some(bytes), "{tokens}");
        assert_eq!(max_snapshot_bytes(tokens, true), Some(drafted), "{tokens}");
    }
    assert_eq!(max_snapshot_bytes(u64::MAX / 4, false), None);
    assert_eq!(max_snapshot_bytes(u64::MAX / 4, true), None);
}

/// M2.6 C6 (D9/D10), raised by m23 design note P8: native-g0 runs 16 rows, so 16 requests in flight are valid and
/// 17 exceed the profile's rows; a scope's cap above the global 16 is refused too (batch-concurrency Q1/Q8).
#[test]
fn native_rows_profile_admits_sixteen_in_flight() {
    let config = load(NATIVE);
    let d = &config.deployments[0];
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v8");
    let a = &config.admission;
    assert_eq!(
        (a.max_backend_inflight, a.per_scope_inflight, a.queue_depth),
        (16, 16, 16)
    );
    let profile = &d.lock.as_ref().unwrap().profile;
    assert_eq!(
        (profile.decode_batch_max, profile.native_decode_batch),
        (16, true)
    );
    // The resume buffers' default total (64 MiB) holds 16 per-request buffers (4 MiB), so 17 raises it too.
    let seventeen = json!({
        "admission": {"max_backend_inflight": 17},
        "api": {"resume_buffer_total_bytes": 17u64 << 22}
    });
    let err = patched(NATIVE, &seventeen).unwrap_err();
    assert!(
        err.contains("admission max_backend_inflight exceeds the profile's validated rows"),
        "{err}"
    );
    let per_scope = json!({"admission": {"per_scope_inflight": 17}});
    let err = patched(NATIVE, &per_scope).unwrap_err();
    assert!(
        err.contains("per-scope in-flight cap exceeds the global backend cap"),
        "{err}"
    );
}

/// batch-concurrency design note BC-D5: the rows cap is the profile's `decode_batch_max` alone, so the deleted
/// `[scheduler]` keys are unknown fields; no shipped config carries them and native-g0 loads without them.
#[test]
fn deleted_scheduler_row_keys_are_unknown_fields() {
    for (key, value) in [
        ("max_active_decode_rows_per_device", 4),
        ("prefill_quantum_tokens", 128),
    ] {
        let err = patched(NATIVE, &json!({"scheduler": {key: value}})).unwrap_err();
        assert!(
            err.contains("unknown field") && err.contains(key),
            "{key}: {err}"
        );
    }
    for entry in std::fs::read_dir(root().join("configs/runtime")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "toml") {
            let text = std::fs::read_to_string(&path).unwrap();
            let doc: toml::Table = toml::from_str(&text).unwrap();
            let scheduler = doc["scheduler"].as_table().unwrap();
            assert!(
                [
                    "max_active_decode_rows_per_device",
                    "prefill_quantum_tokens"
                ]
                .iter()
                .all(|k| !scheduler.contains_key(*k)),
                "{}",
                path.display()
            );
        }
    }
    load(NATIVE);
}

/// m25 design note C3 (D4; ledger D 72): `kv_pool_tokens` is optional; absent or 0 is the derived pool, a non-zero
/// value is a multiple of 64 and at least `max_context_tokens` (200,000 in native-g0). The executor bounds it above.
#[test]
fn kv_pool_tokens_is_optional_and_validated() {
    let pool = |value: Value| json!({"entry": {"native": {"kv_pool_tokens": value}}});
    let absent = patched(NATIVE, &pool(Value::Null)).unwrap();
    let native = absent.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.kv_pool_tokens, None);
    // Absent is not serialized, so configs without the key keep their config_hash.
    let value = serde_json::to_value(native).unwrap();
    assert!(value.get("kv_pool_tokens").is_none(), "{value}");
    for ok in [0u64, 200_000, 200_064] {
        let config = patched(NATIVE, &pool(json!(ok))).unwrap_or_else(|e| panic!("{ok}: {e}"));
        assert_eq!(
            config.deployments[0]
                .native
                .as_ref()
                .unwrap()
                .kv_pool_tokens,
            Some(ok)
        );
    }
    for (bad, needle) in [
        (
            199_936u64,
            "at least deployments[0].native.max_context_tokens",
        ),
        (200_001, "multiple of 64"),
    ] {
        let err = patched(NATIVE, &pool(json!(bad))).unwrap_err();
        assert!(
            err.contains("deployments[0].native.kv_pool_tokens") && err.contains(needle),
            "{bad}: {err}"
        );
    }
}

/// m210 design note C4 (D2; D9C): native accepts `prefix_cache_isolation = "cache_salt"`. The shipped native-g0
/// runs the cache off (`prefix_state_slots` absent = 0, no isolation key) with the operator token for the flush route
/// (D11, N2 (a)); the opt-in native-g0-prefix runs it (16 slots) under `cache_salt` on profile `-v11` (the D9 batch
/// kept it opt-in: its cold-request capture regressed perf smoke's cold TTFT, perf_smoke-20261014T191705Z).
#[test]
fn shipped_native_profile_runs_the_cache_off_and_the_prefix_profile_under_cache_salt() {
    let shipped = load(NATIVE);
    let b = &shipped.deployments[0];
    assert_eq!(b.prefix_cache_isolation(), PrefixCacheIsolation::None);
    assert_eq!(b.native.as_ref().unwrap().prefix_state_slots, 0);
    assert_eq!(
        shipped.auth.operator_token_env.as_deref(),
        Some("CERE_OPERATOR_TOKEN")
    );
    let config = load(NATIVE_PREFIX);
    let b = &config.deployments[0];
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v11");
    assert_eq!(b.prefix_cache_isolation(), PrefixCacheIsolation::CacheSalt);
    assert!(!b.allow_cross_scope_prefix_cache);
    assert_eq!(b.native.as_ref().unwrap().prefix_state_slots, 16);
    assert_eq!(config.auth.scopes.len(), 2);
    assert_eq!(
        config.auth.operator_token_env.as_deref(),
        Some("CERE_OPERATOR_TOKEN")
    );
}

/// m210 design note C4 (D2): two scopes without salt or cross-scope sharing are refused through the shared
/// cross-scope helper (the `sglang_http` message), only while the cache is on (`prefix_state_slots > 0`).
#[test]
fn native_cache_scopes_follow_the_shared_cross_scope_rule() {
    let needle = "more than one auth scope on a shared radix cache requires allow_cross_scope_prefix_cache or prefix_cache_isolation = \"cache_salt\"";
    let err = patched(
        NATIVE_PREFIX,
        &json!({"entry": {"prefix_cache_isolation": null}}),
    )
    .unwrap_err();
    assert!(err.contains(needle), "{err}");
    let err = patched(
        NATIVE_PREFIX,
        &json!({"entry": {"prefix_cache_isolation": "none"}}),
    )
    .unwrap_err();
    assert!(err.contains(needle), "{err}");
    for ok in [
        json!({"entry": {"prefix_cache_isolation": null, "allow_cross_scope_prefix_cache": true}}),
        json!({"entry": {"prefix_cache_isolation": null, "native": {"prefix_state_slots": 0}}}),
    ] {
        patched(NATIVE_PREFIX, &ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    let err = patched(
        "configs/runtime/sglang-t4-qwen38.toml",
        &json!({"auth": {"scopes": [
            {"scope_id": "a", "token_env": "CERE_TOKEN_A"},
            {"scope_id": "b", "token_env": "CERE_TOKEN_B"}
        ]}, "entry": {"allow_cross_scope_prefix_cache": false, "prefix_cache_isolation": null}}),
    )
    .unwrap_err();
    assert!(err.contains(needle), "{err}");
}

/// m210 design note C4 (D9, M210S-15): with the cache on and `[state] host_spill_enabled = false` cered passes
/// `snapshot_host_bytes` 0 in the desc (no host tier), so a configured arena then neither reaches the executor nor
/// needs a snapshot profile; with host spill, or with the cache off, the arena is passed as configured.
#[test]
fn host_spill_off_passes_no_snapshot_arena_with_the_cache_on() {
    let arena = 4u64 << 30;
    let config = patched(
        NATIVE_PREFIX,
        &json!({"entry": {"native": {"snapshot_host_bytes": arena}}}),
    )
    .unwrap();
    assert!(!config.state.host_spill_enabled);
    let native = config.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.desc_snapshot_host_bytes(false), 0);
    assert_eq!(native.desc_snapshot_host_bytes(true), arena);
    let mut off = native.clone();
    off.prefix_state_slots = 0;
    assert_eq!(off.desc_snapshot_host_bytes(false), arena);
    // Passed as configured, the arena needs a profile with `native_state_snapshot` (the shipped one has none).
    for patch in [
        json!({"state": {"host_spill_enabled": true}, "entry": {"native": {"snapshot_host_bytes": arena}}}),
        json!({"entry": {"native": {"snapshot_host_bytes": arena, "prefix_state_slots": 0}}}),
    ] {
        let err = patched(NATIVE_PREFIX, &patch).unwrap_err();
        assert!(
            err.contains("must equal the profile's native_state_snapshot"),
            "{patch}: {err}"
        );
    }
}

/// m210 design note C4 (D6, ledger D 104): `prefix_state_slots` is 0..=1024 (`CERE_MAX_PREFIX_STATE_SLOTS`); 0
/// is not serialized, so configs without the key keep their `config_hash`.
#[test]
fn prefix_state_slots_bounds() {
    let slots = |value: Value| json!({"entry": {"native": {"prefix_state_slots": value}}});
    for ok in [0u64, 1, 1024] {
        let config =
            patched(NATIVE_PREFIX, &slots(json!(ok))).unwrap_or_else(|e| panic!("{ok}: {e}"));
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(native.prefix_state_slots, ok);
        let value = serde_json::to_value(native).unwrap();
        assert_eq!(value.get("prefix_state_slots").is_some(), ok > 0, "{value}");
    }
    let absent = patched(NATIVE_PREFIX, &slots(Value::Null)).unwrap();
    assert_eq!(
        absent.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .prefix_state_slots,
        0
    );
    let err = patched(NATIVE_PREFIX, &slots(json!(1025))).unwrap_err();
    assert!(
        err.contains("deployments[0].native.prefix_state_slots"),
        "{err}"
    );
    assert!(patched(NATIVE_PREFIX, &slots(json!(-1))).is_err());
}

/// O79 (m210 design note PF1): `decode_rounds_per_prefill_chunk` is optional (absent = the executor default 1,
/// not serialized, so committed configs keep their `config_hash`) and 1..=64 (`CERE_MAX_DECODE_ROUNDS_PER_CHUNK`).
#[test]
fn decode_rounds_per_prefill_chunk_bounds() {
    let rounds =
        |value: Value| json!({"entry": {"native": {"decode_rounds_per_prefill_chunk": value}}});
    let committed = load(NATIVE);
    let absent = patched(NATIVE, &rounds(Value::Null)).unwrap();
    let native = absent.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.decode_rounds_per_prefill_chunk, None);
    let value = serde_json::to_value(native).unwrap();
    assert!(
        value.get("decode_rounds_per_prefill_chunk").is_none(),
        "{value}"
    );
    assert_eq!(absent.config_hash(), committed.config_hash());
    for ok in [1u64, 4, 64] {
        let config = patched(NATIVE, &rounds(json!(ok))).unwrap_or_else(|e| panic!("{ok}: {e}"));
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(native.decode_rounds_per_prefill_chunk, Some(ok));
        assert_ne!(config.config_hash(), committed.config_hash(), "{ok}");
    }
    for bad in [0u64, 65] {
        assert_eq!(
            patched(NATIVE, &rounds(json!(bad))).unwrap_err(),
            "invalid runtime config: deployments[0].native.decode_rounds_per_prefill_chunk must be 1..64",
            "{bad}"
        );
    }
}

/// MixedPD (mixed-pd design note): `prefill_slice_ms` is optional (absent = every prefill call runs whole, not
/// serialized, so committed configs keep their `config_hash`) and 1..=10000 (`CERE_MAX_PREFILL_SLICE_MS`).
#[test]
fn prefill_slice_ms_bounds() {
    let slice = |value: Value| json!({"entry": {"native": {"prefill_slice_ms": value}}});
    let committed = load(NATIVE);
    let absent = patched(NATIVE, &slice(Value::Null)).unwrap();
    let native = absent.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.prefill_slice_ms, None);
    let value = serde_json::to_value(native).unwrap();
    assert!(value.get("prefill_slice_ms").is_none(), "{value}");
    assert_eq!(absent.config_hash(), committed.config_hash());
    for ok in [1u64, 200, 10_000] {
        let config = patched(NATIVE, &slice(json!(ok))).unwrap_or_else(|e| panic!("{ok}: {e}"));
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(native.prefill_slice_ms, Some(ok));
        assert_ne!(config.config_hash(), committed.config_hash(), "{ok}");
    }
    for bad in [0u64, 10_001] {
        assert_eq!(
            patched(NATIVE, &slice(json!(bad))).unwrap_err(),
            "invalid runtime config: deployments[0].native.prefill_slice_ms must be 1..10000",
            "{bad}"
        );
    }
}

/// S1 G4 (beat-sglang design note): `decode_share_pct` is optional (absent = the decode rounds per chunk stay, not
/// serialized, so committed configs keep their `config_hash`) and 1..=1000 (`CERE_MAX_DECODE_SHARE_PCT`).
#[test]
fn decode_share_pct_bounds() {
    let share = |value: Value| json!({"entry": {"native": {"decode_share_pct": value}}});
    let committed = load(NATIVE);
    let absent = patched(NATIVE, &share(Value::Null)).unwrap();
    let native = absent.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.decode_share_pct, None);
    let value = serde_json::to_value(native).unwrap();
    assert!(value.get("decode_share_pct").is_none(), "{value}");
    assert_eq!(absent.config_hash(), committed.config_hash());
    for ok in [1u64, 100, 1_000] {
        let config = patched(NATIVE, &share(json!(ok))).unwrap_or_else(|e| panic!("{ok}: {e}"));
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(native.decode_share_pct, Some(ok));
        assert_ne!(config.config_hash(), committed.config_hash(), "{ok}");
    }
    for bad in [0u64, 1_001] {
        assert_eq!(
            patched(NATIVE, &share(json!(bad))).unwrap_err(),
            "invalid runtime config: deployments[0].native.decode_share_pct must be 1..1000",
            "{bad}"
        );
    }
}

/// BC4 (b) (batch-concurrency design note): `prefill_age_calls` is optional (absent = the call is not made and the
/// executor default 8 stays, not serialized, so committed configs keep their `config_hash`) and 1..=1024
/// (`CERE_MAX_PREFILL_AGE_CALLS`).
#[test]
fn prefill_age_calls_bounds() {
    let age = |value: Value| json!({"entry": {"native": {"prefill_age_calls": value}}});
    let committed = load(NATIVE);
    let absent = patched(NATIVE, &age(Value::Null)).unwrap();
    let native = absent.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.prefill_age_calls, None);
    let value = serde_json::to_value(native).unwrap();
    assert!(value.get("prefill_age_calls").is_none(), "{value}");
    assert_eq!(absent.config_hash(), committed.config_hash());
    for ok in [1u64, 8, 1_024] {
        let config = patched(NATIVE, &age(json!(ok))).unwrap_or_else(|e| panic!("{ok}: {e}"));
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(native.prefill_age_calls, Some(ok));
        assert_ne!(config.config_hash(), committed.config_hash(), "{ok}");
    }
    for bad in [0u64, 1_025] {
        assert_eq!(
            patched(NATIVE, &age(json!(bad))).unwrap_err(),
            "invalid runtime config: deployments[0].native.prefill_age_calls must be 1..1024",
            "{bad}"
        );
    }
}

/// Opportunity O47 (build-loop-ple design note): `host_share_dir` is optional (absent = the private HOST_PINNED block,
/// not serialized, so configs without it keep their `config_hash`); a present value is an absolute path without
/// commas (the executor's device-id options are comma-separated).
#[test]
fn host_share_dir_is_optional_and_validated() {
    let share = |value: Value| json!({"entry": {"native": {"host_share_dir": value}}});
    let absent = patched(NATIVE, &share(Value::Null)).unwrap();
    let native = absent.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.host_share_dir, None);
    let value = serde_json::to_value(native).unwrap();
    assert!(value.get("host_share_dir").is_none(), "{value}");
    let ok = patched(NATIVE, &share(json!("/dev/shm/cere-host"))).unwrap();
    assert_eq!(
        ok.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .host_share_dir
            .as_deref(),
        Some("/dev/shm/cere-host")
    );
    for bad in ["dev/shm/cere-host", "/dev/shm/a,b", "", "/dev/shm/x\n"] {
        let err = patched(NATIVE, &share(json!(bad))).unwrap_err();
        assert!(
            err.contains("deployments[0].native.host_share_dir"),
            "{bad:?}: {err}"
        );
    }
}

/// m29 design note C7 (D11): `speculative_draft_tokens` is 0 (absent, not serialized: every config keeps its
/// `config_hash`) or 4; any other value is refused at load. It must agree with the profile's
/// `native_speculative_batch` (the executor reports it iff it has a draft, so a mismatch could never pass readiness).
#[test]
fn speculative_draft_tokens_is_0_or_4_and_follows_the_profile() {
    let drafts = |value: Value| json!({"entry": {"native": {"speculative_draft_tokens": value}}});
    let shipped = load(NATIVE);
    let native = shipped.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.speculative_draft_tokens, 0);
    let value = serde_json::to_value(native).unwrap();
    assert!(value.get("speculative_draft_tokens").is_none(), "{value}");
    patched(NATIVE, &drafts(json!(0))).unwrap();
    let spec = load(SPEC);
    let d = &spec.deployments[0];
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v19");
    assert_eq!(d.native.as_ref().unwrap().speculative_draft_tokens, 4);
    assert!(d.lock.as_ref().unwrap().profile.native_speculative_batch);
    for bad in [1u64, 2, 3, 5, 8] {
        for base in [NATIVE, SPEC] {
            let err = patched(base, &drafts(json!(bad))).unwrap_err();
            assert!(
                err.contains("deployments[0].native.speculative_draft_tokens must be 0 or 4"),
                "{base} {bad}: {err}"
            );
        }
    }
    let err = patched(NATIVE, &drafts(json!(4))).unwrap_err();
    assert!(
        err.contains("must equal the profile's native_speculative_batch"),
        "{err}"
    );
    // Without its levers table (its "adaptive" draft_policy alone needs the drafts), the profile rule refuses.
    let err = patched(
        SPEC,
        &json!({"entry": {"native": {"speculative_draft_tokens": 0, "levers": null}}}),
    )
    .unwrap_err();
    assert!(
        err.contains("must equal the profile's native_speculative_batch"),
        "{err}"
    );
}

/// m29l design note D7 and ship rule: the `-v15` profile loads with 4 draft tokens and MTP drafts only (M1 ship-mtp,
/// an internal benchmark report); lookup stays opt-in per config (the `drafter` lever, m23 PI-02).
#[test]
fn spec_lookup_profile_loads_with_mtp_drafts() {
    let config = load(SPEC_LOOKUP);
    let d = &config.deployments[0];
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v15");
    let native = d.native.as_ref().unwrap();
    assert_eq!(native.speculative_draft_tokens, 4);
    assert_eq!(native.lever_set().unwrap().drafter, Drafter::Mtp);
    assert!(d.lock.as_ref().unwrap().profile.native_speculative_batch);
}

/// Fp8W: the opt-in `-v20` profile is `-v19` plus `k2_weights = "fp8"` (numerics-changing, fixed at model create);
/// the shipped `-v19` keeps the BF16 weights.
#[test]
fn spec_fp8w_profile_loads_with_fp8_weights() {
    let config = load(SPEC_FP8W);
    let d = &config.deployments[0];
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v20");
    let native = d.native.as_ref().unwrap();
    let mut want = spec_levers();
    want["k2_weights"] = json!("fp8");
    assert_eq!(serde_json::to_value(native).unwrap()["levers"], want);
    let set = native.lever_set().unwrap();
    assert_eq!(set.k2_weights, K2Weights::Fp8);
    assert_eq!(
        LeverSet {
            k2_weights: K2Weights::Bf16,
            ..set
        },
        load(SPEC).deployments[0]
            .native
            .as_ref()
            .unwrap()
            .lever_set()
            .unwrap()
    );
    assert!(d.lock.as_ref().unwrap().profile.native_speculative_batch);
}

/// fp4local design note FL-13: the S2 dev profile `-v19-fp4local` is `-v19` (same levers, drafts, prefix cache,
/// memory limit and library) with only its profile id, deployment and artifact changed: the converted FP4-Local
/// artifact its lock and pin name.
#[test]
fn spec_fp4local_profile_is_v19_on_the_converted_artifact() {
    let pin: Value = serde_json::from_str(
        &std::fs::read_to_string(
            root().join("configs/artifacts/qwen38-flash-next-fp4local-cere.pin.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let config = load(SPEC_FP4LOCAL);
    let spec = load(SPEC);
    let (d, base) = (&config.deployments[0], &spec.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v19-fp4local");
    let (native, v19) = (d.native.as_ref().unwrap(), base.native.as_ref().unwrap());
    assert_eq!(native.artifact_dir, pin["path"].as_str().unwrap());
    assert_eq!(
        native.expected_manifest_sha256,
        pin["manifest_sha256"].as_str().unwrap()
    );
    assert_eq!(native.lever_set().unwrap(), v19.lever_set().unwrap());
    assert_eq!(
        native.speculative_draft_tokens,
        v19.speculative_draft_tokens
    );
    assert_eq!(native.library_sha256, v19.library_sha256);
    assert_eq!(native.executor_memory_limit(), v19.executor_memory_limit());
    let lock = d.lock.as_ref().unwrap();
    assert_eq!(lock.lock.lock_id, pin["lock_id"].as_str().unwrap());
    assert_eq!(
        lock.lock.model_content_sha256,
        pin["model_hash"].as_str().unwrap()
    );
    assert_ne!(native.artifact_dir, v19.artifact_dir);
}

/// fp4local design note FL-30: the opt-in S3 profile `-v21` is the S2 dev profile (so `-v19`'s levers, drafts,
/// prefix cache, memory limit and library) with only its profile id, deployment and artifact changed: the S3
/// artifact (MXFP8 dense weights and NVFP4 MTP experts kept) its lock and pin name.
#[test]
fn spec_fp4local_s3_profile_is_s2_on_the_s3_artifact() {
    let pin: Value = serde_json::from_str(
        &std::fs::read_to_string(
            root().join("configs/artifacts/qwen38-flash-next-fp4local-cere-s3.pin.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        pin["profile_id"],
        "rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1"
    );
    let config = load(SPEC_FP4LOCAL_S3);
    let s2 = load(SPEC_FP4LOCAL);
    let (d, base) = (&config.deployments[0], &s2.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v21");
    let (native, s2n) = (d.native.as_ref().unwrap(), base.native.as_ref().unwrap());
    assert_eq!(native.artifact_dir, pin["path"].as_str().unwrap());
    assert_eq!(
        native.expected_manifest_sha256,
        pin["manifest_sha256"].as_str().unwrap()
    );
    assert_eq!(native.lever_set().unwrap(), s2n.lever_set().unwrap());
    assert_eq!(
        native.speculative_draft_tokens,
        s2n.speculative_draft_tokens
    );
    assert_eq!(native.library_sha256, s2n.library_sha256);
    assert_eq!(native.executor_memory_limit(), s2n.executor_memory_limit());
    let lock = d.lock.as_ref().unwrap();
    assert_eq!(lock.lock.lock_id, pin["lock_id"].as_str().unwrap());
    assert_eq!(
        lock.lock.model_content_sha256,
        pin["model_hash"].as_str().unwrap()
    );
    assert_ne!(native.artifact_dir, s2n.artifact_dir);
}

/// fp4local design note FL-31 (Headroom's `-v22`, m210 design note O112): `-v22` is `-v21` (artifact, levers,
/// drafts, library, memory limit, context, rows) with only its profile id, deployment and the three headroom keys
/// changed.
#[test]
fn v22_spec_profile_is_v21_plus_headroom() {
    let config = load(SPEC_V22);
    let s3 = load(SPEC_FP4LOCAL_S3);
    let (d, base) = (&config.deployments[0], &s3.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v22");
    assert_eq!(base.profile_id, "native-g0-qwen38-flash-next-v21");
    let (native, s3n) = (d.native.as_ref().unwrap(), base.native.as_ref().unwrap());
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (32, true, Some(300_032))
    );
    assert_eq!(
        (
            s3n.prefix_state_slots,
            s3n.prefix_end_state,
            s3n.kv_pool_tokens
        ),
        (16, false, Some(200_000))
    );
    assert_eq!(native.artifact_dir, s3n.artifact_dir);
    assert_eq!(native.lever_set().unwrap(), s3n.lever_set().unwrap());
    assert_eq!(
        native.speculative_draft_tokens,
        s3n.speculative_draft_tokens
    );
    assert_eq!(native.library_sha256, s3n.library_sha256);
    assert_eq!(native.executor_memory_limit(), s3n.executor_memory_limit());
    assert_eq!(native.max_context_tokens, s3n.max_context_tokens);
    assert_eq!(
        config.admission.max_backend_inflight,
        s3.admission.max_backend_inflight
    );
    assert_ne!(config.config_hash(), s3.config_hash());
}

/// TreeSpec (tree-spec design note TS-03): `-v28` is `-v22` (artifact, drafts, library, memory limit, context,
/// prefix and pool keys) with only its profile id, deployment and levers.drafter "mtp+lookup+long" changed.
#[test]
fn tree_spec_profile_is_v22_plus_the_long_drafter() {
    let config = load(SPEC_TREE);
    let base = load(SPEC_V22);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v28");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v22");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    let (levers, base_levers) = (native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(
        (levers.drafter, base_levers.drafter),
        (Drafter::MtpLookupLong, Drafter::MtpLookup)
    );
    assert_eq!(
        LeverSet {
            drafter: Drafter::MtpLookup,
            ..levers
        },
        base_levers
    );
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens
        )
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// Chains (chains design note): `-v36` is the TreeSpec `-v28` (artifact, drafts, library, memory limit, context,
/// prefix and pool keys) with only its profile id, deployment and levers.drafter "mtp+lookup+chain" changed.
#[test]
fn chains_profile_is_v28_plus_the_chain_drafter() {
    let config = load(SPEC_CHAINS);
    let base = load(SPEC_TREE);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v36");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v28");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    let (levers, base_levers) = (native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(
        (levers.drafter, base_levers.drafter),
        (Drafter::MtpLookupChain, Drafter::MtpLookupLong)
    );
    assert_eq!(
        LeverSet {
            drafter: Drafter::MtpLookupLong,
            ..levers
        },
        base_levers
    );
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens
        )
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// GrammarSpec (grammar-spec design note GS-04): `-v49` is the default `-v29` (artifact, drafts, library, memory
/// limit, context, prefix and pool keys) with only its profile id, deployment and levers.grammar_drafts "on" changed.
#[test]
fn grammar_profile_is_v29_plus_grammar_drafts() {
    let config = load(SPEC_GRAMMAR);
    let base = spec_default_before_flips();
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v49");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v29");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    let (levers, base_levers) = (native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(
        (levers.grammar_drafts, base_levers.grammar_drafts),
        (GrammarDrafts::On, GrammarDrafts::Off)
    );
    assert_eq!(
        LeverSet {
            grammar_drafts: GrammarDrafts::Off,
            ..levers
        },
        base_levers
    );
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens
        )
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// MixedPD (mixed-pd design note): `-v47` is the default `-v29` with only its profile id, deployment and
/// `prefill_slice_ms` 200 (absent on `-v29`) changed.
#[test]
fn mixed_pd_profile_is_v29_plus_the_prefill_slice() {
    let config = load(SPEC_V47);
    let base = spec_default_before_flips();
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v47");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v29");
    assert_eq!(
        d.deployment.as_deref(),
        Some("configs/deployments/qwen38-flash-next-native-spec-v47")
    );
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        (native.prefill_slice_ms, bn.prefill_slice_ms),
        (Some(200), None)
    );
    let mut value = serde_json::to_value(native).unwrap();
    value.as_object_mut().unwrap().remove("prefill_slice_ms");
    assert_eq!(value, serde_json::to_value(bn).unwrap());
    assert_ne!(config.config_hash(), base.config_hash());
}

/// Sessions (sessions design note O211): `-v45` is the default `-v29` (artifact, levers, drafts, library, memory
/// limit, context, rows, prefix and pool keys) with only its profile id, deployment and the M2.10 host tier changed:
/// a 16 GiB pinned arena and `[state] host_spill_enabled`, so cered passes the arena to the executor.
#[test]
fn sessions_profile_is_v29_plus_the_host_tier() {
    let config = load(SESSIONS);
    let base = spec_default_before_flips();
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v45");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v29");
    assert!(config.state.host_spill_enabled);
    assert!(!base.state.host_spill_enabled);
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(native.snapshot_host_bytes, 17_179_869_184);
    assert_eq!(bn.snapshot_host_bytes, 0);
    assert_eq!(
        native.desc_snapshot_host_bytes(config.state.host_spill_enabled),
        17_179_869_184
    );
    assert!(d.lock.as_ref().unwrap().profile.native_state_snapshot);
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens
        )
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// sessions design note SE-09: `-v45` with U1 at its four rows and the optimistic reserve. The arena bound counts
/// kind 10 on this drafting model (5 parked at 200,000 tokens fit 16 GiB, 6 do only without kind 10), and every
/// generation in flight (4) needs a park slot to grow.
#[test]
fn sessions_profile_admits_preemption_with_the_reserve() {
    use cere_runtime::native::snapshot::arena_slot_bytes;
    let reserve = |parked: u64| {
        json!({"entry": {"native": {
            "preempt": true, "max_parked_sessions": parked, "reserve_output_tokens": 4096
        }}})
    };
    let config = patched(SESSIONS, &reserve(5)).unwrap();
    let native = config.deployments[0].native.as_ref().unwrap();
    assert_eq!(
        (
            native.preempt,
            native.max_parked_sessions,
            native.reserve_output_tokens
        ),
        (true, 5, 4096)
    );
    assert_eq!(
        config.deployments[0]
            .lock
            .as_ref()
            .unwrap()
            .profile
            .decode_batch_max,
        4
    );
    let (plain, drafted) = (
        arena_slot_bytes(200_000, false).unwrap(),
        arena_slot_bytes(200_000, true).unwrap(),
    );
    assert!(6 * plain <= 17_179_869_184 && 6 * drafted > 17_179_869_184);
    let err = patched(SESSIONS, &reserve(6)).unwrap_err();
    assert!(
        err.contains("max_parked_sessions x the snapshot bytes"),
        "{err}"
    );
    let err = patched(SESSIONS, &reserve(3)).unwrap_err();
    assert!(
        err.contains(
            "reserve_output_tokens requires max_parked_sessions >= admission.max_backend_inflight"
        ),
        "{err}"
    );
}

/// DraftTrain (drafttrain design note D7; default since v29default design note VD-03): `-v29` is `-v22` (levers,
/// drafts, library, memory limit, context, prefix and pool keys) with only its profile id, deployment and artifact
/// changed: the S3 artifact variant whose BF16 `mtp.*` draft weights were self-distilled and whose draft token map was
/// re-derived, its pin and
/// the default lock (tokenizer and template are S3's).
#[test]
fn default_spec_profile_is_v22_on_the_trained_draft_artifact() {
    let pin: Value = serde_json::from_str(
        &std::fs::read_to_string(
            root().join("configs/artifacts/qwen38-flash-next-fp4local-cere-s3-draft.pin.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let s3: Value = serde_json::from_str(
        &std::fs::read_to_string(
            root().join("configs/artifacts/qwen38-flash-next-fp4local-cere-s3.pin.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for key in [
        "profile_id",
        "tokenizer_hash",
        "template_hash",
        "source_revision",
    ] {
        assert_eq!(pin[key], s3[key], "{key}");
    }
    let config = spec_default_before_flips();
    let base = load(SPEC_V22);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v29");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v22");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(native.artifact_dir, pin["path"].as_str().unwrap());
    assert_eq!(
        native.expected_manifest_sha256,
        pin["manifest_sha256"].as_str().unwrap()
    );
    assert_ne!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(
        native.sidecar.last().map(String::as_str),
        Some(native.artifact_dir.as_str())
    );
    let lock = d.lock.as_ref().unwrap();
    assert_eq!(lock.lock.lock_id, pin["lock_id"].as_str().unwrap());
    assert_eq!(
        lock.lock.model_content_sha256,
        pin["model_hash"].as_str().unwrap()
    );
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens
        )
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// Ctx262 (ctx262 design note CX-02): `-v44` is the default `-v29` (artifact, levers, drafts, library, memory limit,
/// prefix keys) with only its profile id, deployment, max_context_tokens 262,144 (the model's max_position_embeddings)
/// and kv_pool_tokens 524,288 (within the CX-01 A4 support bound 1,048,576) changed.
#[test]
fn spec_262k_profile_is_v29_plus_262k_context() {
    let config = load(SPEC_262K);
    let base = spec_default_before_flips();
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v44");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v29");
    assert_eq!(
        d.deployment.as_deref(),
        Some("configs/deployments/qwen38-flash-next-native-spec-262k")
    );
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        (native.max_context_tokens, bn.max_context_tokens),
        (262_144, 200_000)
    );
    assert_eq!(
        (native.kv_pool_tokens, bn.kv_pool_tokens),
        (Some(524_288), Some(300_032))
    );
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.expected_manifest_sha256, bn.expected_manifest_sha256);
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(
        (native.prefix_state_slots, native.prefix_end_state),
        (bn.prefix_state_slots, bn.prefix_end_state)
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// PrefixHit (m210 design note D17): `-v51` is the default `-v29` (artifact, levers, drafts, library, memory limit,
/// context, prefix and pool keys) with only its profile id, deployment and `prefix_alt_edges = true` changed.
#[test]
fn prefix_hit_profile_is_v29_plus_alt_edges() {
    let config = load(SPEC_V51);
    let base = spec_default_before_flips();
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v51");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v29");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        (native.prefix_alt_edges, bn.prefix_alt_edges),
        (true, false)
    );
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.expected_manifest_sha256, bn.expected_manifest_sha256);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens
        )
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// BeatSglang S7 (beat-sglang design note S7): `-v56` is the default (artifact, drafts, library, memory limit,
/// context, prefix and pool keys) with only its profile id, deployment and two prefill levers changed: k2_gemm
/// `small_m_a8` (W8A8 MXFP8 dense GEMMs at m > 16) and prefill_attention `p4`.
#[test]
fn opt_in_prefill_profile_is_default_plus_prefill_levers() {
    let config = load(SPEC_V56);
    let base = load(SPEC_DEFAULT);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v56");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v29");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    let (levers, bl) = (native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(
        (levers.k2_gemm, levers.prefill_attention),
        (K2Gemm::SmallMA8, PrefillAttention::P4)
    );
    assert_eq!(
        (bl.k2_gemm, bl.prefill_attention),
        (K2Gemm::SmallM, PrefillAttention::CudaCore)
    );
    assert_eq!(
        LeverSet {
            k2_gemm: bl.k2_gemm,
            prefill_attention: bl.prefill_attention,
            ..levers
        },
        bl
    );
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.expected_manifest_sha256, bn.expected_manifest_sha256);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens,
            native.prefix_alt_edges
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens,
            bn.prefix_alt_edges
        )
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// VisionLong (m213l design note Amendment A2): the dev-only diagnostic `-v54` is the vision `-v43` (artifact, sidecar,
/// levers, drafts, library, memory limit, context, prefix and pool keys, ledger) with only its profile id, deployment
/// and `prefix_alt_edges = true` changed.
#[test]
fn vision_diagnostic_profile_is_v43_plus_alt_edges() {
    let config = load(SPEC_VISION_V54);
    let base = load(SPEC_VISION);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v54");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v43");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        (native.prefix_alt_edges, bn.prefix_alt_edges),
        (true, false)
    );
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.expected_manifest_sha256, bn.expected_manifest_sha256);
    assert_eq!(native.sidecar, bn.sidecar);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.kv_pool_tokens
        ),
        (
            bn.prefix_state_slots,
            bn.prefix_end_state,
            bn.kv_pool_tokens
        )
    );
    assert_eq!(config.ledger.state_dir, base.ledger.state_dir);
    assert_ne!(config.config_hash(), base.config_hash());
}

/// Ctx262 CX-R-vision (ctx262 design note CX-R18): the dev-only diagnostic `-v55` is VisionLong's `-v54` (artifact,
/// sidecar, levers, drafts, library, prefix keys, ledger) with only its profile id, deployment, context 262,144, a
/// 479,168-token pool (`-v44`'s 524,288 cut by the shrink rule, revision CX-R18-limit) and the executor memory limit
/// at the schema bound (86,432 MiB) changed.
#[test]
fn vision_context_diagnostic_profile_is_v54_at_262k() {
    let config = load(SPEC_VISION_V55);
    let base = load(SPEC_VISION_V54);
    let ctx = load(SPEC_262K);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v55");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v54");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    let cn = ctx.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.max_context_tokens, cn.max_context_tokens);
    assert_eq!(
        (
            native.max_context_tokens,
            native.kv_pool_tokens,
            cn.kv_pool_tokens
        ),
        (262_144, Some(479_168), Some(524_288))
    );
    assert_eq!(
        (native.executor_memory_limit(), bn.executor_memory_limit()),
        (86_432 << 20, 86_016 << 20)
    );
    assert_eq!(
        (native.prefix_alt_edges, native.lever_set().unwrap()),
        (bn.prefix_alt_edges, bn.lever_set().unwrap())
    );
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.expected_manifest_sha256, bn.expected_manifest_sha256);
    assert_eq!(native.sidecar, bn.sidecar);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(
        (native.prefix_state_slots, native.prefix_end_state),
        (bn.prefix_state_slots, bn.prefix_end_state)
    );
    assert_eq!(config.ledger.state_dir, base.ledger.state_dir);
    assert_ne!(config.config_hash(), base.config_hash());
}

/// Ctx262 CX-09 (ctx262 design note): `-v52` is `-v44` (artifact, levers, drafts, library, memory limit, context
/// 262,144, prefix and pool keys) with only its profile id, deployment and `prefix_alt_edges = true` changed.
#[test]
fn ctx262_prefix_hit_profile_is_v44_plus_alt_edges() {
    let config = load(SPEC_V52);
    let base = load(SPEC_262K);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v52");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v44");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        (native.prefix_alt_edges, bn.prefix_alt_edges),
        (true, false)
    );
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(
        (native.max_context_tokens, native.kv_pool_tokens),
        (262_144, Some(524_288))
    );
    assert_eq!(
        (native.max_context_tokens, native.kv_pool_tokens),
        (bn.max_context_tokens, bn.kv_pool_tokens)
    );
    assert_eq!(
        (native.prefix_state_slots, native.prefix_end_state),
        (bn.prefix_state_slots, bn.prefix_end_state)
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// Sessions SE-11 (sessions design note): `-v53` is `-v52` (artifact, levers, drafts, library, memory limit,
/// context 262,144, pool, prefix keys and alternative edges) with only its profile id, deployment and the `-v45` host
/// tier changed: a 16 GiB pinned arena and `[state] host_spill_enabled`.
#[test]
fn sessions_v53_profile_is_v52_plus_the_host_tier() {
    let config = load(SESSIONS_V53);
    let base = load(SPEC_V52);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v53");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v52");
    assert!(config.state.host_spill_enabled);
    assert!(!base.state.host_spill_enabled);
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        (native.snapshot_host_bytes, bn.snapshot_host_bytes),
        (17_179_869_184, 0)
    );
    assert_eq!(
        native.desc_snapshot_host_bytes(config.state.host_spill_enabled),
        17_179_869_184
    );
    assert!(d.lock.as_ref().unwrap().profile.native_state_snapshot);
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(
        (native.max_context_tokens, native.kv_pool_tokens),
        (bn.max_context_tokens, bn.kv_pool_tokens)
    );
    assert_eq!(
        (
            native.prefix_state_slots,
            native.prefix_end_state,
            native.prefix_alt_edges
        ),
        (bn.prefix_state_slots, bn.prefix_end_state, true)
    );
    assert_ne!(config.config_hash(), base.config_hash());
}

/// BatchConcurrency BC4 (batch-concurrency design note BC4): `-v59` is the production Sessions `-v53` with only its
/// profile id, deployment and MixedPD's `prefill_slice_ms` 50 (absent on `-v53`) changed.
#[test]
fn opt_in_prefill_slice_profile_is_v53_plus_prefill_slice_50() {
    let config = load(SESSIONS_V59);
    let base = load(SESSIONS_V53);
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v59");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v53");
    assert_eq!(
        d.deployment.as_deref(),
        Some("configs/deployments/qwen38-flash-next-native-sessions-v59")
    );
    assert_eq!(
        config.state.host_spill_enabled,
        base.state.host_spill_enabled
    );
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        (native.prefill_slice_ms, bn.prefill_slice_ms),
        (Some(50), None)
    );
    let mut value = serde_json::to_value(native).unwrap();
    value.as_object_mut().unwrap().remove("prefill_slice_ms");
    assert_eq!(value, serde_json::to_value(bn).unwrap());
    assert_ne!(config.config_hash(), base.config_hash());
}

/// BatchConc BC2 (batch-concurrency design note BC2, owner Q1 (b), Q8): `-v58` admits 16 in flight and 16 queued on
/// `-v53`'s artifact, library, drafts and context at `decode_batch_max` 16, with production's levers; production
/// (`prod-t4.toml`) serves `-v58` at the same admission since window 11 (BC7).
#[test]
fn rows16_v58_profile_admits_sixteen_in_flight() {
    let config = load(ROWS16_V58);
    let base = load(SESSIONS_V53);
    let prod = load("configs/runtime/prod-t4.toml");
    let (d, b, p) = (
        &config.deployments[0],
        &base.deployments[0],
        &prod.deployments[0],
    );
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v58");
    assert_eq!(p.profile_id, d.profile_id);
    assert_eq!(
        (
            prod.admission.max_backend_inflight,
            prod.admission.per_scope_inflight,
            prod.admission.queue_depth
        ),
        (16, 16, 16)
    );
    let (lock, base_lock) = (d.lock.as_ref().unwrap(), b.lock.as_ref().unwrap());
    assert_eq!(
        (
            lock.profile.decode_batch_max,
            base_lock.profile.decode_batch_max
        ),
        (16, 4)
    );
    assert_eq!(
        (
            config.admission.max_backend_inflight,
            config.admission.per_scope_inflight,
            config.admission.queue_depth
        ),
        (16, 16, 16)
    );
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert!(!native.preempt);
    assert_eq!(
        native.lever_set().unwrap(),
        p.native.as_ref().unwrap().lever_set().unwrap()
    );
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.max_context_tokens, 262_144);
    // Q2 (a): the pool shrinks for R16's plan but still holds one session at the full context.
    assert!(native.kv_pool_tokens.unwrap() >= native.max_context_tokens);
    assert!(native.kv_pool_tokens < bn.kv_pool_tokens);
}

/// BatchConc BC2 (fail-first): `max_backend_inflight` above workers x (`decode_batch_max` + parked) is refused: 17
/// on `-v58`'s 16 rows, and 16 on `-v53`'s 4 rows (the deployment without the rows), which loads at 4. The resume
/// buffers' total grows with the cap (one 4 MiB buffer per backend slot), so only the rows bound refuses.
#[test]
fn inflight_above_the_rows_plus_parked_is_refused() {
    let inflight = |n: u64| {
        json!({
            "admission": {"max_backend_inflight": n, "per_scope_inflight": n},
            "api": {"resume_buffer_total_bytes": n * 4_194_304}
        })
    };
    let refused = "admission max_backend_inflight exceeds the profile's validated rows";
    assert_eq!(
        patched(ROWS16_V58, &inflight(16))
            .unwrap()
            .admission
            .max_backend_inflight,
        16
    );
    let err = patched(ROWS16_V58, &inflight(17)).unwrap_err();
    assert!(err.contains(refused), "{err}");
    let mut four_rows = inflight(16);
    four_rows["entry"] = json!({
        "profile_id": "native-g0-qwen38-flash-next-v53",
        "deployment": "configs/deployments/qwen38-flash-next-native-sessions-v53"
    });
    let err = patched(ROWS16_V58, &four_rows).unwrap_err();
    assert!(err.contains(refused), "{err}");
    four_rows["admission"] = inflight(4)["admission"].clone();
    four_rows["api"] = inflight(4)["api"].clone();
    patched(ROWS16_V58, &four_rows).unwrap();
}

/// m23 design note PI-02 (the cutover of m29l design note C5L D6): `[deployments.native.levers]` is optional
/// (absent, not serialized: every config keeps its `config_hash`); `drafter` is "mtp", "lookup" or "mtp+lookup" and
/// the lookup values need `speculative_draft_tokens = 4`; anything outside the registry is refused at load.
#[test]
fn levers_are_optional_and_validated_against_the_registry() {
    let levers = |value: Value| json!({"entry": {"native": {"levers": value}}});
    let shipped = load(NATIVE);
    let native = shipped.deployments[0].native.as_ref().unwrap();
    assert!(native.levers.is_empty() && !native.lever_hot_swap);
    assert_eq!(native.lever_set().unwrap(), LeverSet::default());
    let value = serde_json::to_value(native).unwrap();
    assert!(
        value.get("levers").is_none() && value.get("lever_hot_swap").is_none(),
        "{value}"
    );
    // Owner decision 2026-10-15: `-v19` carries its speculation levers (was -v18's bit-identical three).
    let spec = load(SPEC);
    let native = spec.deployments[0].native.as_ref().unwrap();
    assert!(!native.lever_hot_swap);
    assert_eq!(
        serde_json::to_value(native).unwrap()["levers"],
        spec_levers()
    );
    for (raw, want) in [
        ("mtp", Drafter::Mtp),
        ("lookup", Drafter::Lookup),
        ("mtp+lookup", Drafter::MtpLookup),
        ("mtp+lookup+tail", Drafter::MtpLookupTail),
        ("mtp+lookup+long", Drafter::MtpLookupLong),
        ("mtp+lookup+chain", Drafter::MtpLookupChain),
    ] {
        let config = patched(SPEC, &bare_spec_levers(json!({ "drafter": raw }))).unwrap();
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(native.lever_set().unwrap().drafter, want, "{raw}");
        let value = serde_json::to_value(native).unwrap();
        assert_eq!(value["levers"], json!({ "drafter": raw }));
    }
    let kernels = patched(
        NATIVE,
        &levers(json!({"k2_gemm": "small_m", "prefill_attention": "p4", "moe_gemm": "k7"})),
    )
    .unwrap();
    let set = kernels.deployments[0]
        .native
        .as_ref()
        .unwrap()
        .lever_set()
        .unwrap();
    assert_eq!(
        (set.k2_gemm, set.prefill_attention, set.moe_gemm),
        (K2Gemm::SmallM, PrefillAttention::P4, MoeGemm::K7)
    );
    patched(
        NATIVE,
        &levers(json!({"drafter": "mtp", "draft_tokens": "max"})),
    )
    .unwrap();
    for (bad, needle) in [
        (
            json!({"drafter": "lookup"}),
            "deployments[0].native.levers.drafter: \"lookup\" needs speculative_draft_tokens = 4",
        ),
        (
            json!({"drafter": "mtp+lookup"}),
            "deployments[0].native.levers.drafter: \"mtp+lookup\" needs speculative_draft_tokens = 4",
        ),
        (
            json!({"draft_tokens": "2"}),
            "deployments[0].native.levers.draft_tokens: \"2\" needs speculative_draft_tokens = 4",
        ),
        (
            json!({"grammar_drafts": "on"}),
            "deployments[0].native.levers.grammar_drafts: \"on\" needs speculative_draft_tokens = 4",
        ),
    ] {
        let err = patched(NATIVE, &levers(bad.clone())).unwrap_err();
        assert!(err.contains(needle), "{bad}: {err}");
    }
    for (bad, needle) in [
        (
            json!({"drafter": "ngram"}),
            "levers.drafter: unknown implementation \"ngram\"",
        ),
        (
            json!({"drafter": "MTP"}),
            "levers.drafter: unknown implementation \"MTP\"",
        ),
        (
            json!({"drafter": ""}),
            "levers.drafter: unknown implementation \"\"",
        ),
        (json!({"k2": "small_m"}), "levers.k2: unknown lever"),
        (
            json!({"grammar_drafts": "auto"}),
            "levers.grammar_drafts: unknown implementation \"auto\"",
        ),
        (
            json!({"drafter": 1}),
            "invalid type: integer `1`, expected a string",
        ),
    ] {
        let err = patched(SPEC, &levers(bad.clone())).unwrap_err();
        assert!(err.contains(needle), "{bad}: {err}");
    }
}

/// m23 design note PI-02: the lever rows of the shared table hit their rule (Python mirrors the needles). SelfTune
/// D3: a selected impl must be proven on the device's arch.
const LEVER_CASES: [(&str, &str); 16] = [
    (
        "native_levers_moe_p3b_on_hip",
        "deployments[0].native.levers.moe_gemm: \"p3b\" is not proven on gfx1151",
    ),
    (
        "native_levers_moe_p3b_on_vk",
        "deployments[0].native.levers.moe_gemm: \"p3b\" is not proven on vulkan",
    ),
    (
        "native_levers_k2_small_m_on_vk",
        "deployments[0].native.levers.k2_gemm: \"small_m\" is not proven on vulkan",
    ),
    (
        "native_levers_prefill_p4_on_hip",
        "deployments[0].native.levers.prefill_attention: \"p4\" is not proven on gfx1151",
    ),
    (
        "native_levers_decode_spec_graph_on_vk",
        "deployments[0].native.levers.decode_step: \"spec_graph\" is not proven on vulkan",
    ),
    (
        "native_levers_verify_batched_on_vk",
        "deployments[0].native.levers.verify_attention: \"batched\" is not proven on vulkan",
    ),
    (
        "native_levers_k2_weights_fp8_on_vk",
        "deployments[0].native.levers.k2_weights: \"fp8\" is not proven on vulkan",
    ),
    (
        "native_levers_drafter_unknown",
        "deployments[0].native.levers.drafter: unknown implementation \"ngram\" (registered: mtp, lookup, mtp+lookup, mtp+lookup+tail, mtp+lookup+long, mtp+lookup+chain)",
    ),
    (
        "native_levers_drafter_lookup_without_drafts",
        "deployments[0].native.levers.drafter: \"mtp+lookup\" needs speculative_draft_tokens = 4",
    ),
    (
        "native_levers_unknown_lever",
        "deployments[0].native.levers.k2: unknown lever",
    ),
    (
        "native_levers_unknown_impl",
        "deployments[0].native.levers.k2_gemm: unknown implementation \"split_k9\" (registered: cublaslt, small_m, small_m_a8, tail_split)",
    ),
    (
        "native_levers_draft_tokens_without_drafts",
        "deployments[0].native.levers.draft_tokens: \"3\" needs speculative_draft_tokens = 4",
    ),
    (
        "native_levers_non_string",
        "invalid type: integer `1`, expected a string",
    ),
    (
        "native_levers_draft_policy_unknown",
        "deployments[0].native.levers.draft_policy: unknown implementation \"greedy\" (registered: fixed, adaptive)",
    ),
    (
        "native_levers_draft_policy_adaptive_without_drafts",
        "deployments[0].native.levers.draft_policy: \"adaptive\" needs speculative_draft_tokens = 4",
    ),
    (
        "native_levers_grammar_drafts_on_without_drafts",
        "deployments[0].native.levers.grammar_drafts: \"on\" needs speculative_draft_tokens = 4",
    ),
];

/// Valid lever rows of the shared table.
const LEVER_VALID_CASES: [&str; 4] = [
    "native_levers_drafter_lookup_valid",
    "native_levers_kernels_valid",
    "native_levers_prefill_p4_on_vk_valid",
    "native_levers_draft_policy_adaptive_valid",
];

/// The rows' patches are all `{}`, so each row is its base with `entry_patch` merged into `deployments[0]`.
#[test]
fn lever_cases_hit_their_rule() {
    let raw =
        std::fs::read_to_string(root().join("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&raw).unwrap();
    let named = |expect: &str| -> Vec<&str> {
        cases
            .iter()
            .filter(|c| c["expect"] == expect)
            .filter_map(|c| c["name"].as_str())
            .filter(|n| n.starts_with("native_levers_"))
            .collect()
    };
    assert_eq!(named("invalid"), LEVER_CASES.map(|(n, _)| n));
    assert_eq!(named("valid"), LEVER_VALID_CASES);
    let run = |name: &str| {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        assert_eq!(case["patch"], json!({}), "{name}");
        patched(
            case["base"].as_str().unwrap(),
            &json!({"entry": case["entry_patch"]}),
        )
    };
    for (name, needle) in LEVER_CASES {
        let err = run(name).err().unwrap_or_else(|| panic!("{name} accepted"));
        assert!(err.contains(needle), "{name}: {err}");
    }
    for name in LEVER_VALID_CASES {
        run(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// m23 design note PI-04: the dev hot swap (`cere_model_set_selection`) does not flush the prefix cache, so a config
/// with both is refused (a NO_SPECULATION request after a swap could hit states of the previous levers); the shared
/// table's row (Python mirrors the needle in tests/test_cere_contracts.py `SPEC_CASES`) is `-v19` plus the swap.
/// The swap cache-off and the cache without the swap both load.
#[test]
fn lever_hot_swap_is_refused_with_the_prefix_cache() {
    let needle = "deployments[0].native.lever_hot_swap with prefix_state_slots > 0: the hot swap leaves prefix \
                  states computed under the previous levers; serve the hot swap cache-off";
    let raw =
        std::fs::read_to_string(root().join("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&raw).unwrap();
    let case = cases
        .iter()
        .find(|c| c["name"] == "native_lever_hot_swap_with_prefix_cache")
        .unwrap();
    assert_eq!(
        (
            case["expect"].as_str(),
            case["base"].as_str(),
            &case["patch"]
        ),
        (Some("invalid"), Some(SPEC), &json!({}))
    );
    let err = patched(SPEC, &json!({"entry": case["entry_patch"]})).unwrap_err();
    assert!(err.contains(needle), "{err}");
    assert_eq!(
        load(SPEC).deployments[0]
            .native
            .as_ref()
            .unwrap()
            .prefix_state_slots,
        16
    );
    let off = patched(
        SPEC,
        &json!({"entry": {"prefix_cache_isolation": null,
                          "native": {"lever_hot_swap": true, "prefix_state_slots": null}}}),
    )
    .unwrap();
    let native = off.deployments[0].native.as_ref().unwrap();
    assert!(native.lever_hot_swap && native.prefix_state_slots == 0);
}

/// ab-split design note AB-02: the `[deployments.native.ab]` rows of the shared table hit their rule (Python
/// mirrors the needles in tests/test_cere_contracts.py `AB_CASES`).
const AB_CASES: [(&str, &str); 6] = [
    (
        "native_ab_unknown_lever",
        "deployments[0].native.ab.arms[0].levers.draft: unknown lever",
    ),
    (
        "native_ab_unknown_impl",
        "deployments[0].native.ab.arms[0].levers.draft_policy: unknown implementation \"greedy\" (registered: fixed, adaptive)",
    ),
    (
        "native_ab_executor_lever",
        "deployments[0].native.ab.arms[0].levers.k2_gemm: an executor-layer lever is process-global; an arm takes runtime-layer levers only",
    ),
    (
        "native_ab_zero_weights",
        "deployments[0].native.ab.arms weights must sum above 0",
    ),
    (
        "native_ab_duplicate_ids",
        "deployments[0].native.ab.arms[1].id \"a\" is a duplicate arm id",
    ),
    (
        "native_ab_adaptive_without_drafts",
        "deployments[0].native.ab.arms[0].levers.draft_policy: \"adaptive\" needs speculative_draft_tokens = 4",
    ),
];

#[test]
fn ab_cases_hit_their_rule() {
    let raw =
        std::fs::read_to_string(root().join("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&raw).unwrap();
    let named = |expect: &str| -> Vec<&str> {
        cases
            .iter()
            .filter(|c| c["expect"] == expect)
            .filter_map(|c| c["name"].as_str())
            .filter(|n| n.starts_with("native_ab_"))
            .collect()
    };
    assert_eq!(named("invalid"), AB_CASES.map(|(n, _)| n));
    assert_eq!(named("valid"), ["native_ab_fixed_vs_adaptive_valid"]);
    let run = |name: &str| {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        patched(
            case["base"].as_str().unwrap(),
            &json!({"entry": case["entry_patch"]}),
        )
    };
    for (name, needle) in AB_CASES {
        let err = run(name).err().unwrap_or_else(|| panic!("{name} accepted"));
        assert!(err.contains(needle), "{name}: {err}");
    }
    let config = run("native_ab_fixed_vs_adaptive_valid").unwrap();
    let plan = cere_runtime::ab::Plan::from_native(config.deployments[0].native.as_ref().unwrap())
        .unwrap()
        .expect("a plan");
    let ids: Vec<&str> = plan.arms().iter().map(|a| &*a.id).collect();
    assert_eq!(ids, ["fixed", "adaptive"]);
}

/// AB-02: the remaining bounds: a salt and arm ids are identifiers, 1..=16 arms, weights at most 1_000_000, and
/// no unknown key in the section or an arm.
#[test]
fn ab_section_bounds_are_refused() {
    let ab = |value: Value| json!({"entry": {"native": {"ab": value}}});
    let arm = |id: &str| json!({"id": id, "weight": 1});
    for (bad, needle) in [
        (
            json!({"salt": "", "arms": [arm("a")]}),
            "deployments[0].native.ab.salt",
        ),
        (
            json!({"salt": "s", "arms": []}),
            "deployments[0].native.ab.arms item count out of range",
        ),
        (
            json!({"salt": "s", "arms": (0..17).map(|i| arm(&format!("a{i}"))).collect::<Vec<_>>()}),
            "deployments[0].native.ab.arms item count out of range",
        ),
        (
            json!({"salt": "s", "arms": [{"id": "a b", "weight": 1}]}),
            "deployments[0].native.ab.arms[0].id",
        ),
        (
            json!({"salt": "s", "arms": [{"id": "a", "weight": 1_000_001}]}),
            "deployments[0].native.ab.arms[0].weight must be 0..1000000",
        ),
        (
            json!({"salt": "s", "arms": [{"id": "a", "weight": 1, "share": 2}]}),
            "unknown field `share`",
        ),
        (
            json!({"salt": "s", "arms": [arm("a")], "sticky": true}),
            "unknown field `sticky`",
        ),
    ] {
        let err = patched(SPEC, &ab(bad.clone())).unwrap_err();
        assert!(err.contains(needle), "{bad}: {err}");
    }
}

/// AB-01/AB-06: no shipped profile has `[deployments.native.ab]`, so it is not serialized (the serialized config and
/// every reload leaf are as before), and a config that adds it has another `config_hash` (the hash names the
/// experiment).
#[test]
fn ab_is_absent_from_shipped_profiles_and_moves_the_hash_when_present() {
    for entry in std::fs::read_dir(root().join("configs/runtime")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") {
            continue;
        }
        let config = RuntimeConfig::load(&path).unwrap();
        for (i, d) in config.deployments.iter().enumerate() {
            if let Some(native) = &d.native {
                assert!(native.ab.is_none(), "{}: deployments[{i}]", path.display());
            }
        }
        let value = serde_json::to_value(&config).unwrap();
        for d in value["deployments"].as_array().unwrap() {
            assert!(d["native"].get("ab").is_none(), "{}", path.display());
        }
    }
    let base = load(SPEC);
    let with = patched(
        SPEC,
        &json!({"entry": {"native": {"ab": {"salt": "exp1", "arms": [
            {"id": "fixed", "weight": 1, "levers": {"draft_policy": "fixed"}},
            {"id": "adaptive", "weight": 1, "levers": {"draft_policy": "adaptive"}},
        ]}}}}),
    )
    .unwrap();
    assert_ne!(base.config_hash(), with.config_hash());
    assert!(with.deployments[0].native.as_ref().unwrap().ab.is_some());
}

/// loop-guard design note LG-03: the `[deployments.native.loop_guard]` rows of the shared table hit their rule
/// (Python mirrors the names in tests/test_cere_contracts.py `LOOP_GUARD_CASES`).
const LOOP_GUARD_CASES: [(&str, &str); 8] = [
    (
        "native_loop_guard_both_off",
        "deployments[0].native.loop_guard turns both detectors off (min_repeats = 0, tool_repeat_run = 0); remove the table instead",
    ),
    (
        "native_loop_guard_period_inverted",
        "deployments[0].native.loop_guard.min_period_bytes must be at most max_period_bytes",
    ),
    (
        "native_loop_guard_one_repeat",
        "deployments[0].native.loop_guard.min_repeats must be 0 or 2..64",
    ),
    (
        "native_loop_guard_window_too_big",
        "deployments[0].native.loop_guard: min_repeats x max_period_bytes must be at most 65536",
    ),
    (
        "native_loop_guard_period_too_long",
        "deployments[0].native.loop_guard.max_period_bytes must be 1..4096",
    ),
    (
        "native_loop_guard_one_tool_turn",
        "deployments[0].native.loop_guard.tool_repeat_run must be 0 or 2..10000",
    ),
    (
        "native_loop_guard_empty_ignore_key",
        "deployments[0].native.loop_guard.ignore_arg_keys[] must be 1..64",
    ),
    ("native_loop_guard_unknown_key", "unknown field `window_bytes`"),
];

#[test]
fn loop_guard_cases_hit_their_rule() {
    let raw =
        std::fs::read_to_string(root().join("tests/fixtures/runtime_config_invalid.json")).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&raw).unwrap();
    let named = |expect: &str| -> Vec<&str> {
        cases
            .iter()
            .filter(|c| c["expect"] == expect)
            .filter_map(|c| c["name"].as_str())
            .filter(|n| n.starts_with("native_loop_guard_"))
            .collect()
    };
    assert_eq!(named("invalid"), LOOP_GUARD_CASES.map(|(n, _)| n));
    assert_eq!(
        named("valid"),
        [
            "native_loop_guard_defaults_valid",
            "native_loop_guard_tools_only_valid"
        ]
    );
    let run = |name: &str| {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        patched(
            case["base"].as_str().unwrap(),
            &json!({"entry": case["entry_patch"]}),
        )
    };
    for (name, needle) in LOOP_GUARD_CASES {
        let err = run(name).err().unwrap_or_else(|| panic!("{name} accepted"));
        assert!(err.contains(needle), "{name}: {err}");
    }
    let config = run("native_loop_guard_defaults_valid").unwrap();
    let guard = config.deployments[0]
        .native
        .as_ref()
        .unwrap()
        .loop_guard
        .as_ref();
    assert_eq!(
        guard,
        Some(&cere_runtime::config::LoopGuardConfig::default())
    );
    let defaults = cere_runtime::config::LoopGuardConfig::default();
    assert_eq!(
        (
            defaults.min_period_bytes,
            defaults.max_period_bytes,
            defaults.min_repeats,
            defaults.min_span_bytes,
            defaults.tool_repeat_run,
            defaults.ignore_arg_keys.as_slice(),
        ),
        (16, 1024, 4, 512, 10, ["description".to_owned()].as_slice())
    );
    let config = run("native_loop_guard_tools_only_valid").unwrap();
    let guard = config.deployments[0]
        .native
        .as_ref()
        .unwrap()
        .loop_guard
        .as_ref()
        .unwrap();
    assert_eq!((guard.min_repeats, guard.tool_repeat_run), (0, 10));
}

/// LG-03: no shipped profile but `prod-t4.toml` (production window 5, owner 2026-10-25: the table at its defaults)
/// and its 16-row dev twin `-v58` (batch-concurrency design note BC2) has `[deployments.native.loop_guard]`, so it
/// is not serialized (`config_hash` and every reload leaf as before); a config that adds it has another `config_hash`.
#[test]
fn loop_guard_is_absent_from_shipped_profiles_and_moves_the_hash_when_present() {
    for entry in std::fs::read_dir(root().join("configs/runtime")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") {
            continue;
        }
        let prod = path
            .file_name()
            .is_some_and(|n| n == "prod-t4.toml" || n == "native-g0-rows16-v58.toml");
        let config = RuntimeConfig::load(&path).unwrap();
        let value = serde_json::to_value(&config).unwrap();
        for (i, d) in config.deployments.iter().enumerate() {
            if let Some(native) = &d.native {
                if prod {
                    assert_eq!(
                        native.loop_guard,
                        Some(cere_runtime::config::LoopGuardConfig::default())
                    );
                    continue;
                }
                assert!(
                    native.loop_guard.is_none(),
                    "{}: deployments[{i}]",
                    path.display()
                );
                assert!(
                    value["deployments"][i]["native"]
                        .get("loop_guard")
                        .is_none(),
                    "{}",
                    path.display()
                );
            }
        }
    }
    let base = load(NATIVE);
    let with = patched(NATIVE, &json!({"entry": {"native": {"loop_guard": {}}}})).unwrap();
    assert_ne!(base.config_hash(), with.config_hash());
}

/// m29a design note A1, A4 (since m23 design note PI-02 the registry's `draft_policy` lever): the shipped plain
/// and lookup configs leave it absent (not serialized, so `config_hash` is unchanged) and run "fixed", the default;
/// `-v19` sets "adaptive" in its levers table (as -v18 did); "adaptive" needs `speculative_draft_tokens = 4`; values
/// outside the registry are refused at load.
#[test]
fn levers_draft_policy_defaults_to_fixed() {
    let levers = |value: Value| json!({"entry": {"native": {"levers": value}}});
    let spec = load(SPEC);
    assert_eq!(
        spec.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .lever_set()
            .unwrap()
            .draft_policy,
        DraftPolicy::Adaptive
    );
    for rel in [NATIVE, SPEC_LOOKUP] {
        let config = load(rel);
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(
            native.lever_set().unwrap().draft_policy,
            DraftPolicy::Fixed,
            "{rel}"
        );
        let value = serde_json::to_value(native).unwrap();
        assert!(value.get("levers").is_none(), "{rel}: {value}");
    }
    for (raw, want) in [
        ("fixed", DraftPolicy::Fixed),
        ("adaptive", DraftPolicy::Adaptive),
    ] {
        let config = patched(SPEC, &bare_spec_levers(json!({"draft_policy": raw}))).unwrap();
        let native = config.deployments[0].native.as_ref().unwrap();
        assert_eq!(native.lever_set().unwrap().draft_policy, want, "{raw}");
        let value = serde_json::to_value(native).unwrap();
        assert_eq!(value["levers"], json!({ "draft_policy": raw }));
    }
    let empty = patched(SPEC, &bare_spec_levers(json!({}))).unwrap();
    assert_eq!(
        empty.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .lever_set()
            .unwrap(),
        LeverSet::default()
    );
    patched(NATIVE, &levers(json!({"draft_policy": "fixed"}))).unwrap();
    let err = patched(NATIVE, &levers(json!({"draft_policy": "adaptive"}))).unwrap_err();
    assert!(
        err.contains(
            "deployments[0].native.levers.draft_policy: \"adaptive\" needs speculative_draft_tokens = 4"
        ),
        "{err}"
    );
    for (bad, needle) in [
        (
            json!({"draft_policy": "greedy"}),
            "levers.draft_policy: unknown implementation \"greedy\"",
        ),
        (
            json!({"draft_policy": 1}),
            "invalid type: integer `1`, expected a string",
        ),
        (
            json!({"draft_count": "2"}),
            "levers.draft_count: unknown lever",
        ),
    ] {
        let err = patched(SPEC, &levers(bad.clone())).unwrap_err();
        assert!(err.contains(needle), "{bad}: {err}");
    }
}

/// m29 design note M29S2-02 (ledger M1): the executor memory limit is per config. Absent = the 80 GiB limit of
/// every profile today (not serialized, so configs keep their `config_hash`); a present value is a whole MiB at
/// least 80 GiB. The speculation profile declares its own.
#[test]
fn executor_memory_limit_bytes_is_optional_and_validated() {
    let limit = |value: Value| json!({"entry": {"native": {"executor_memory_limit_bytes": value}}});
    let shipped = load(NATIVE);
    let native = shipped.deployments[0].native.as_ref().unwrap();
    assert_eq!(native.executor_memory_limit_bytes, None);
    assert_eq!(native.executor_memory_limit(), 80 << 30);
    let value = serde_json::to_value(native).unwrap();
    assert!(
        value.get("executor_memory_limit_bytes").is_none(),
        "{value}"
    );
    let ok = patched(NATIVE, &limit(json!(86_016u64 << 20))).unwrap();
    assert_eq!(
        ok.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .executor_memory_limit(),
        86_016 << 20
    );
    for bad in [(80u64 << 30) - (1 << 20), (84u64 << 30) + 1, 0] {
        let err = patched(NATIVE, &limit(json!(bad))).unwrap_err();
        assert!(
            err.contains("deployments[0].native.executor_memory_limit_bytes"),
            "{bad}: {err}"
        );
    }
    let spec = load(SPEC);
    let spec_limit = spec.deployments[0]
        .native
        .as_ref()
        .unwrap()
        .executor_memory_limit();
    assert!(spec_limit > 80 << 30, "{spec_limit}");
}

/// m23 design note P8 (ledger M1, N7): readiness refuses an executor memory limit above the GPU0 ceiling (GPU0
/// total 97,887 MiB - desktop baseline L0 1,215 MiB - the invariant 10,240 MiB free floor = 86,432 MiB), since such
/// a budget lets the executor cross the floor; every shipped profile's limit is at or below it.
#[test]
fn executor_memory_limit_above_the_floor_ceiling_is_refused() {
    let limit = |value: u64| json!({"entry": {"native": {"executor_memory_limit_bytes": value}}});
    let ceiling = (97_887u64 - 1_215 - 10_240) << 20;
    assert_eq!(
        patched(NATIVE, &limit(ceiling)).unwrap().deployments[0]
            .native
            .as_ref()
            .unwrap()
            .executor_memory_limit(),
        ceiling
    );
    for bad in [ceiling + (1 << 20), 96 << 30] {
        let err = patched(NATIVE, &limit(bad)).unwrap_err();
        assert!(
            err.contains("deployments[0].native.executor_memory_limit_bytes")
                && err.contains("10,240 MiB"),
            "{bad}: {err}"
        );
    }
    for config in [
        NATIVE,
        SPEC,
        SPEC_LOOKUP,
        SPEC_FP8W,
        SPEC_FP4LOCAL,
        SPEC_FP4LOCAL_S3,
        SPEC_V22,
        SPEC_DEFAULT,
        SPEC_GRAMMAR,
        NATIVE_2GPU,
    ] {
        let shipped = load(config).deployments[0]
            .native
            .as_ref()
            .unwrap()
            .executor_memory_limit();
        assert!(shipped <= ceiling, "{config}: {shipped}");
    }
}

/// M2.16 L4 (m216 design note): `keep_host_share` is optional and default false (not serialized when false, so
/// configs keep their `config_hash`), and true requires `host_share_dir`.
#[test]
fn keep_host_share_requires_host_share_dir() {
    let absent = patched(NATIVE, &json!({"entry": {"native": {}}})).unwrap();
    let native = absent.deployments[0].native.as_ref().unwrap();
    assert!(!native.keep_host_share);
    let value = serde_json::to_value(native).unwrap();
    assert!(value.get("keep_host_share").is_none(), "{value}");
    let kept = json!({"entry": {"native": {"host_share_dir": "/dev/shm/cere-host", "keep_host_share": true}}});
    assert!(
        patched(NATIVE, &kept).unwrap().deployments[0]
            .native
            .as_ref()
            .unwrap()
            .keep_host_share
    );
    let err = patched(
        NATIVE,
        &json!({"entry": {"native": {"keep_host_share": true}}}),
    )
    .unwrap_err();
    assert!(
        err.contains("deployments[0].native.keep_host_share") && err.contains("host_share_dir"),
        "{err}"
    );
}

/// Track LoRA: the opt-in `-v50` is `SPEC_DEFAULT` (`-v29`) plus `[deployments.native.lora]` with only its profile id
/// and deployment changed; the shipped profiles carry no lora table.
#[test]
fn lora_profile_is_v29_plus_the_lora_table() {
    let config = load(SPEC_LORA);
    let base = spec_default_before_flips();
    let (d, b) = (&config.deployments[0], &base.deployments[0]);
    assert_eq!(d.profile_id, "native-g0-qwen38-flash-next-v50");
    assert_eq!(b.profile_id, "native-g0-qwen38-flash-next-v29");
    let (native, bn) = (d.native.as_ref().unwrap(), b.native.as_ref().unwrap());
    assert_eq!(
        native.lora,
        Some(LoraConfig {
            max_adapters: 4,
            max_rank: 64,
            adapter_root: "/var/lib/cere/lora".to_owned(),
            preload: Vec::new(),
        })
    );
    assert_eq!(bn.lora, None);
    assert_eq!(native.lever_set().unwrap(), bn.lever_set().unwrap());
    assert_eq!(native.artifact_dir, bn.artifact_dir);
    assert_eq!(native.speculative_draft_tokens, bn.speculative_draft_tokens);
    assert_eq!(native.library_sha256, bn.library_sha256);
    assert_eq!(native.executor_memory_limit(), bn.executor_memory_limit());
    assert_eq!(native.max_context_tokens, bn.max_context_tokens);
    for rel in [NATIVE, NATIVE_PREFIX, SPEC, SPEC_DEFAULT, SPEC_CHAINS] {
        assert_eq!(
            load(rel).deployments[0].native.as_ref().unwrap().lora,
            None,
            "{rel}"
        );
    }
}

/// Track LoRA: `[deployments.native.lora]` bounds and paths, each refusal with its field.
#[test]
fn lora_table_bounds() {
    let lora = |patch: Value| {
        let mut table =
            json!({"max_adapters": 2, "max_rank": 16, "adapter_root": "/var/lib/cere/lora"});
        if let (Some(t), Some(p)) = (table.as_object_mut(), patch.as_object()) {
            t.extend(p.clone());
        }
        patched(SPEC_DEFAULT, &json!({"entry": {"native": {"lora": table}}}))
    };
    let ok = lora(json!({"preload": [{"name": "a", "path": "/var/lib/cere/lora/a"}]})).unwrap();
    assert_eq!(
        ok.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .lora
            .as_ref()
            .unwrap()
            .preload[0]
            .name,
        "a"
    );
    for (patch, needle) in [
        (json!({"max_adapters": 0}), "native.lora.max_adapters"),
        (json!({"max_adapters": 65}), "native.lora.max_adapters"),
        (json!({"max_rank": 0}), "native.lora.max_rank"),
        (json!({"max_rank": 257}), "native.lora.max_rank"),
        (json!({"adapter_root": "lora"}), "native.lora.adapter_root"),
        (
            json!({"adapter_root": "/var/lib/cere lora"}),
            "native.lora.adapter_root",
        ),
        (
            json!({"max_adapters": 1, "preload": [{"name": "a", "path": "/a"}, {"name": "b", "path": "/b"}]}),
            "native.lora.preload",
        ),
        (
            json!({"preload": [{"name": "a", "path": "/a"}, {"name": "a", "path": "/b"}]}),
            "duplicate deployments[0].native.lora.preload name a",
        ),
        (
            json!({"preload": [{"name": "a b", "path": "/a"}]}),
            "native.lora.preload.name",
        ),
        (
            json!({"preload": [{"name": "a", "path": "a"}]}),
            "native.lora.preload.path",
        ),
        (json!({"rank": 8}), "unknown field `rank`"),
    ] {
        let err = lora(patch.clone()).unwrap_err();
        assert!(err.contains(needle), "{patch}: {err}");
    }
}

/// WeightLab WL-01: `-v57` is `SPEC_LORA` (`-v50`) plus `[deployments.native.research]` with only its profile id,
/// deployment, `api_compat_anthropic`, production's default deadline and the three arm scopes changed; no other
/// profile carries the research table (so every other config keeps its `config_hash`).
#[test]
fn research_profile_is_v50_plus_the_research_table() {
    let config = load(SPEC_RESEARCH);
    let native = config.deployments[0].native.as_ref().unwrap();
    assert_eq!(
        native.research,
        Some(ResearchConfig {
            device_staging_bytes: 67_108_864,
            host_buffer_bytes: 1_073_741_824,
            max_capture_bytes_per_request: 268_435_456,
            max_plans: 8,
            max_vectors: 64,
        })
    );
    assert_eq!(config.server.bind, "127.0.0.1:18309");
    assert!(config.auth.operator_token_env.is_some());
    let mut got = serde_json::to_value(&config).unwrap();
    let base = serde_json::to_value(load(SPEC_LORA)).unwrap();
    let d = &mut got["deployments"][0];
    assert_eq!(d["profile_id"], "native-g0-qwen38-flash-next-v57");
    assert_eq!(d["api_compat_anthropic"], true);
    d["native"].as_object_mut().unwrap().remove("research");
    for key in ["profile_id", "deployment", "api_compat_anthropic"] {
        match base["deployments"][0].get(key) {
            Some(v) => d[key] = v.clone(),
            None => {
                d.as_object_mut().unwrap().remove(key);
            }
        }
    }
    assert_eq!(got["api"]["default_deadline_ms"], 900_000);
    got["api"]["default_deadline_ms"] = base["api"]["default_deadline_ms"].clone();
    let scopes = got["auth"]["scopes"].as_array_mut().unwrap();
    let arms: Vec<Value> = scopes.split_off(scopes.len() - 3);
    assert_eq!(
        arms.iter()
            .map(|s| s["scope_id"].clone())
            .collect::<Vec<_>>(),
        [json!("wl-a"), json!("wl-b"), json!("wl-p")]
    );
    assert_eq!(got, base);
    for rel in [
        NATIVE,
        NATIVE_PREFIX,
        SPEC,
        SPEC_DEFAULT,
        SPEC_LORA,
        SPEC_V52,
        NATIVE_2GPU,
    ] {
        let c = load(rel);
        assert_eq!(
            c.deployments[0].native.as_ref().unwrap().research,
            None,
            "{rel}"
        );
        let v = serde_json::to_value(c.deployments[0].native.as_ref().unwrap()).unwrap();
        assert!(v.get("research").is_none(), "{rel}: {v}");
    }
}

/// A `[deployments.native.research]` table patched onto `SPEC_LORA` (the `-v57` values, then `patch`).
fn research(patch: Value) -> Result<RuntimeConfig, String> {
    let mut table = json!({
        "device_staging_bytes": 67108864u64,
        "host_buffer_bytes": 1073741824u64,
        "max_capture_bytes_per_request": 268435456u64,
        "max_plans": 8,
        "max_vectors": 64
    });
    if let (Some(t), Some(p)) = (table.as_object_mut(), patch.as_object()) {
        for (k, v) in p {
            if v.is_null() {
                t.remove(k);
            } else {
                t.insert(k.clone(), v.clone());
            }
        }
    }
    patched(
        SPEC_LORA,
        &json!({"entry": {"native": {"research": table}}}),
    )
}

/// WeightLab WL-01: `[deployments.native.research]` bounds, each refusal with its field; every key required, no
/// other key.
#[test]
fn research_table_bounds() {
    assert!(research(json!({})).is_ok());
    let edges = json!({"device_staging_bytes": 1u64 << 20, "host_buffer_bytes": 16u64 << 30,
                       "max_capture_bytes_per_request": 16u64 << 30, "max_plans": 64, "max_vectors": 1024});
    assert!(research(edges).is_ok());
    let low = json!({"device_staging_bytes": 1u64 << 30, "host_buffer_bytes": 1u64 << 20,
                     "max_capture_bytes_per_request": 4096, "max_plans": 1, "max_vectors": 1});
    assert!(research(low).is_ok());
    for (patch, needle) in [
        (
            json!({"device_staging_bytes": (1u64 << 20) - 1}),
            "native.research.device_staging_bytes must be",
        ),
        (
            json!({"device_staging_bytes": (1u64 << 30) + 1}),
            "native.research.device_staging_bytes must be",
        ),
        (
            json!({"host_buffer_bytes": (1u64 << 20) - 1}),
            "native.research.host_buffer_bytes must be",
        ),
        (
            json!({"host_buffer_bytes": (16u64 << 30) + 1}),
            "native.research.host_buffer_bytes must be",
        ),
        (
            json!({"max_capture_bytes_per_request": 4095}),
            "native.research.max_capture_bytes_per_request must be 4096..1073741824",
        ),
        (
            json!({"max_capture_bytes_per_request": 1073741825u64}),
            "native.research.max_capture_bytes_per_request must be 4096..1073741824",
        ),
        (
            json!({"max_plans": 0}),
            "native.research.max_plans must be 1..64",
        ),
        (
            json!({"max_plans": 65}),
            "native.research.max_plans must be 1..64",
        ),
        (
            json!({"max_vectors": 0}),
            "native.research.max_vectors must be 1..1024",
        ),
        (
            json!({"max_vectors": 1025}),
            "native.research.max_vectors must be 1..1024",
        ),
        (json!({"max_plans": null}), "missing field `max_plans`"),
        (
            json!({"device_staging_bytes": null}),
            "missing field `device_staging_bytes`",
        ),
        (json!({"slots": 2}), "unknown field `slots`"),
    ] {
        let err = research(patch.clone()).unwrap_err();
        assert!(err.contains(needle), "{patch}: {err}");
    }
}

/// WeightLab WL-18: a research table on the production bind port 18080 is a config error.
#[test]
fn research_refuses_the_production_port() {
    let err = patched(
        SPEC_RESEARCH,
        &json!({"server": {"bind": "127.0.0.1:18080"}}),
    )
    .unwrap_err();
    assert!(
        err.contains("deployments[0].native.research is refused on the production bind port 18080"),
        "{err}"
    );
    // Without the table the port is a plain dev config (production itself binds it).
    patched(SPEC_LORA, &json!({"server": {"bind": "127.0.0.1:18080"}})).unwrap();
}

/// WeightLab WL-18: a research table without `[auth] operator_token_env` is a config error.
#[test]
fn research_requires_the_operator_token() {
    let err = patched(
        SPEC_RESEARCH,
        &json!({"auth": {"operator_token_env": null}}),
    )
    .unwrap_err();
    assert!(
        err.contains("deployments[0].native.research requires [auth] operator_token_env"),
        "{err}"
    );
    patched(SPEC_LORA, &json!({"auth": {"operator_token_env": null}})).unwrap();
}

/// WeightLab WL-18: a research table whose `device_uuid` equals the production marker's `uuid` is refused; another
/// uuid, or no marker, passes.
#[test]
fn research_refuses_the_production_marker_gpu() {
    let config = load(SPEC_RESEARCH);
    let uuid = config.deployments[0]
        .native
        .as_ref()
        .unwrap()
        .device_uuid
        .clone()
        .unwrap();
    let err = config
        .validate_research_isolation(Some(&uuid))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(&format!(
            "deployments[0].native.research is refused on the production GPU: device_uuid {uuid} equals the \
             production marker's uuid"
        )),
        "{err}"
    );
    config
        .validate_research_isolation(Some("GPU-00000000-0000-0000-0000-000000000000"))
        .unwrap();
    config.validate_research_isolation(None).unwrap();
    // Without the table nothing is checked.
    load(SPEC_LORA)
        .validate_research_isolation(Some(&uuid))
        .unwrap();
}

/// WeightLab WL-18: the uuid check covers every native replica's `device_uuid`.
#[test]
fn research_refuses_the_production_marker_gpu_on_a_replica() {
    let mut config = load(NATIVE_2GPU);
    let research = load(SPEC_RESEARCH).deployments[0]
        .native
        .as_ref()
        .unwrap()
        .research
        .clone();
    let native = config.deployments[0].native.as_mut().unwrap();
    native.research = research;
    let uuid = native.replicas.as_ref().unwrap()[1].device_uuid.clone();
    config.auth.operator_token_env = Some("CERE_OPERATOR_TOKEN".to_owned());
    let err = config
        .validate_research_isolation(Some(&uuid))
        .unwrap_err()
        .to_string();
    assert!(err.contains(&format!("device_uuid {uuid} equals")), "{err}");
    // The research table itself is refused with replicas (plans live in one executor); fake replica uuids keep the
    // host's own production marker out of the verdict.
    let replicas: Vec<Value> = config.deployments[0].native.as_ref().unwrap().replicas.as_ref().unwrap()
        .iter()
        .enumerate()
        .map(|(i, r)| json!({"device": r.device, "device_uuid": format!("GPU-0000000{i}-0000-0000-0000-000000000000")}))
        .collect();
    let err = patched(
        NATIVE_2GPU,
        &json!({"auth": {"operator_token_env": "CERE_OPERATOR_TOKEN"},
                "entry": {"native": {"replicas": replicas, "research": {"device_staging_bytes": 67108864u64,
                 "host_buffer_bytes": 1073741824u64, "max_capture_bytes_per_request": 268435456u64,
                 "max_plans": 8, "max_vectors": 64}}}}),
    )
    .unwrap_err();
    assert!(
        err.contains(
            "deployments[0].native.research is refused with deployments[0].native.replicas"
        ),
        "{err}"
    );
}

/// WeightLab WL-18: the marker reads as `scripts/live/_common.py` `gpu1_production`: a missing file is no marker,
/// an unreadable or malformed one (uuid "unknown" there) matches no device.
#[test]
fn production_marker_uuid_reads_like_common_py() {
    let dir = std::env::temp_dir().join(format!("cere-marker-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let marker = dir.join("gpu1.production");
    assert_eq!(production_marker_uuid(&marker), None);
    std::fs::write(
        &marker,
        r#"{"service": "cere-t4.service", "since": "2026-10-29T13:00:22Z", "uuid": "GPU-bbbbbbbb-1111-4111-8111-111111111111", "container": "prod-cere-t4"}"#,
    )
    .unwrap();
    assert_eq!(
        production_marker_uuid(&marker).as_deref(),
        Some("GPU-bbbbbbbb-1111-4111-8111-111111111111")
    );
    for bad in [
        "",
        "not json",
        "[1]",
        r#"{"service": "x"}"#,
        r#"{"uuid": 5}"#,
    ] {
        std::fs::write(&marker, bad).unwrap();
        assert_eq!(production_marker_uuid(&marker), None, "{bad}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// SelfTune ST4 S3a (fail-first): `native.family = "sdar"` (DICE-8B) refuses, by key name, every feature an SDAR
/// library does not have; absent and `"qwen4_exp"` are the same config (the key is not serialized).
#[test]
fn family_sdar_refuses_qwen_features_by_name() {
    let sdar = |base: &str, native: Value| {
        let mut native = native;
        native["family"] = json!("sdar");
        patched(base, &json!({ "entry": { "native": native } })).unwrap_err()
    };
    let research = json!({"device_staging_bytes": 67108864u64, "host_buffer_bytes": 1073741824u64,
                          "max_capture_bytes_per_request": 268435456u64, "max_plans": 8, "max_vectors": 64});
    for (base, native, key) in [
        (SPEC_DEFAULT, json!({}), "native.speculative_draft_tokens"),
        (NATIVE_PREFIX, json!({}), "native.prefix_state_slots"),
        (
            NATIVE,
            json!({"snapshot_host_bytes": 1073741824u64}),
            "native.snapshot_host_bytes",
        ),
        (
            NATIVE,
            json!({"lever_hot_swap": true}),
            "native.lever_hot_swap",
        ),
        (
            NATIVE,
            json!({"lora": {"max_adapters": 4, "max_rank": 64, "adapter_root": "/var/lib/cere/lora"}}),
            "native.lora",
        ),
        (NATIVE, json!({ "research": research }), "native.research"),
        (NATIVE, json!({}), "profile decode_batch_max > 1"),
    ] {
        let err = sdar(base, native);
        assert!(
            err.contains(&format!(
                "{key} is refused with deployments[0].native.family = \"sdar\""
            )),
            "{base} {key}: {err}"
        );
    }
    let absent = load(NATIVE);
    let qwen = patched(
        NATIVE,
        &json!({"entry": {"native": {"family": "qwen4_exp"}}}),
    )
    .unwrap();
    assert_eq!(absent.config_hash(), qwen.config_hash());
    let err = patched(NATIVE, &json!({"entry": {"native": {"family": "llama"}}})).unwrap_err();
    assert!(err.contains("family"), "{err}");
}

/// Inkling 2 Mini S0 (fail-first, inkling design note D-I5): `native.family = "inkling2_loop"` refuses, by key
/// name, the prefix cache, snapshots, LoRA, research, the lever hot swap, a hip device and a profile
/// advertising a grammar output (the Qwen profile does); decode rows > 1 stay admitted (a text-only profile with no
/// refused key validates under the family). The drafter's window is `inkling_mtp_draft_window_is_2_to_8`'s.
#[test]
fn family_inkling2_loop_refuses_unported_features_by_name() {
    let inkling = |base: &str, native: Value| {
        let mut native = native;
        native["family"] = json!("inkling2_loop");
        patched(base, &json!({ "entry": { "native": native } }))
    };
    let research = json!({"device_staging_bytes": 67108864u64, "host_buffer_bytes": 1073741824u64,
                          "max_capture_bytes_per_request": 268435456u64, "max_plans": 8, "max_vectors": 64});
    for (base, native, key) in [
        (NATIVE_PREFIX, json!({}), "native.prefix_state_slots"),
        (
            NATIVE,
            json!({"snapshot_host_bytes": 1073741824u64}),
            "native.snapshot_host_bytes",
        ),
        (
            NATIVE,
            json!({"lever_hot_swap": true}),
            "native.lever_hot_swap",
        ),
        (
            NATIVE,
            json!({"lora": {"max_adapters": 4, "max_rank": 64, "adapter_root": "/var/lib/cere/lora"}}),
            "native.lora",
        ),
        (NATIVE, json!({ "research": research }), "native.research"),
        (
            NATIVE,
            json!({"device_uuid": "hip:0"}),
            "native.device_uuid on the hip arch",
        ),
    ] {
        let err = inkling(base, native).unwrap_err();
        assert!(
            err.contains(&format!("deployments[0].{key}"))
                && err.contains("is refused with deployments[0].native.family = \"inkling2_loop\""),
            "{base} {key}: {err}"
        );
    }
    let err = inkling(NATIVE, json!({})).unwrap_err();
    assert!(
        err.contains("profile structured outputs / tool_calls_constrained is refused with"),
        "{err}"
    );
    inkling("configs/runtime/native-g0-dice8b.toml", json!({})).unwrap();
}

/// inkling design note D-I-S9B: the default Inkling lane (`configs/runtime/native-g0-inkling.toml`, MTP drafter on)
/// validates with its priced `speculative_draft_tokens = 4` and with every window 2..8 (1..7 drafts); on family
/// inkling2_loop 1 and 9 are refused, on every other family the values stay 0 or 4 (6 refused); Inkling drafts are
/// CUDA only (a Vulkan device refuses them by name).
#[test]
fn inkling_mtp_draft_window_is_2_to_8() {
    const MTP: &str = "configs/runtime/native-g0-inkling.toml";
    let config = patched(MTP, &json!({})).unwrap();
    assert_eq!(
        config.deployments[0]
            .native
            .as_ref()
            .unwrap()
            .speculative_draft_tokens,
        4
    );
    for window in 2..=8u32 {
        let config = patched(
            MTP,
            &json!({"entry": {"native": {"speculative_draft_tokens": window}}}),
        )
        .unwrap();
        assert_eq!(
            config.deployments[0]
                .native
                .as_ref()
                .unwrap()
                .speculative_draft_tokens,
            window
        );
    }
    for bad in [1u32, 9, 16] {
        let err = patched(
            MTP,
            &json!({"entry": {"native": {"speculative_draft_tokens": bad}}}),
        )
        .unwrap_err();
        assert!(err.contains("speculative_draft_tokens"), "{bad}: {err}");
    }
    let err = patched(
        MTP,
        &json!({"entry": {"native": {"speculative_draft_tokens": 1}}}),
    )
    .unwrap_err();
    assert!(
        err.contains("deployments[0].native.speculative_draft_tokens must be 0 or 2..8 with family inkling2_loop"),
        "{err}"
    );
    let err = patched(MTP, &json!({"entry": {"native": {"device_uuid": "vk:0"}}})).unwrap_err();
    assert!(
        err.contains(
            "deployments[0].native.speculative_draft_tokens on a non-CUDA device is refused"
        ),
        "{err}"
    );
    let err = patched(
        SPEC_DEFAULT,
        &json!({"entry": {"native": {"speculative_draft_tokens": 6}}}),
    )
    .unwrap_err();
    assert!(
        err.contains("deployments[0].native.speculative_draft_tokens must be 0 or 4"),
        "{err}"
    );
    let err = patched(MTP, &json!({"entry": {"native": {"family": "sdar"}}})).unwrap_err();
    assert!(err.contains("speculative_draft_tokens"), "{err}");
}

/// SdarOptIn (fail-first): `sdar_gemm = "changing"` validates on the DICE-8B (family sdar) profile and is refused by
/// name on a qwen4_exp one; a qwen4_exp HIP deployment keeps `sdar_gemm`'s CUDA-only default (never sent).
#[test]
fn sdar_gemm_changing_validates_on_family_sdar_only() {
    let changing = json!({"entry": {"native": {"levers": {"sdar_gemm": "changing"}}}});
    let sdar = patched("configs/runtime/native-g0-dice8b.toml", &changing).unwrap();
    let levers = sdar.deployments[0]
        .native
        .as_ref()
        .unwrap()
        .lever_set()
        .unwrap();
    assert_eq!(levers.sdar_gemm, SdarGemm::Changing);
    let err = patched(NATIVE, &changing).unwrap_err();
    assert!(
        err.contains(
            "deployments[0].native.levers.sdar_gemm: \"changing\" is not built for family qwen4_exp"
        ),
        "{err}"
    );
    patched(
        NATIVE,
        &json!({"entry": {"native": {"device_uuid": "hip:0", "levers": {"moe_gemm": "k7"}}}}),
    )
    .unwrap();
}
