//! P2 wire helpers (m35 design note Q10a inventory, Q12 payload digest, Q14 limits and per-principal quota) and
//! the M3.6 P1 additions (m36 design note Q6 lifecycle digest, Q7 inventory v2, Q12 reasons).

use cere_runtime::fed::quota::PrincipalQuota;
use cere_runtime::fed::wire::{
    self, AdmissionState, Inventory, InventoryError, InventoryModel, ModelState,
};
use tonic::Code;

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        write!(out, "{b:02x}").unwrap();
        out
    })
}

const WORK_UNIT: &[u8] = br#"{"kind":"chat","stream":false}"#;
const INPUT: &[u8] = br#"{"rid":"cere-1-2-abc"}"#;

#[test]
fn digest_matches_independent_vectors() {
    // Expected values computed outside Rust (Python hashlib over u64-BE length prefixes), so the vector pins the
    // Q12 encoding itself, not this implementation.
    let submit = wire::payload_sha256(
        wire::METHOD_SUBMIT,
        "a",
        "cere-1-2-abc",
        1,
        WORK_UNIT,
        INPUT,
    );
    assert_eq!(
        hex(&submit),
        "74f57cfdc65c30422fad13538f17937ffc710a76cbd08b268d2aa21272c6fd06"
    );
    let cancel = wire::payload_sha256(wire::METHOD_CANCEL, "a", "cere-1-2-abc", 12, b"", b"tkt-1");
    assert_eq!(
        hex(&cancel),
        "c8eb85b4a911c63f7bd87170b55dddd46bfe1af278265054a1bfc0e1282deb1f"
    );
}

#[test]
fn digest_changes_with_every_included_field() {
    let base = wire::payload_sha256(wire::METHOD_SUBMIT, "a", "r", 1, WORK_UNIT, INPUT);
    let variants = [
        wire::payload_sha256(wire::METHOD_CANCEL, "a", "r", 1, WORK_UNIT, INPUT),
        wire::payload_sha256(wire::METHOD_SUBMIT, "b", "r", 1, WORK_UNIT, INPUT),
        wire::payload_sha256(wire::METHOD_SUBMIT, "a", "s", 1, WORK_UNIT, INPUT),
        wire::payload_sha256(wire::METHOD_SUBMIT, "a", "r", 2, WORK_UNIT, INPUT),
        wire::payload_sha256(wire::METHOD_SUBMIT, "a", "r", 1, b"{}", INPUT),
        wire::payload_sha256(wire::METHOD_SUBMIT, "a", "r", 1, WORK_UNIT, b"{}"),
    ];
    for (i, variant) in variants.iter().enumerate() {
        assert_ne!(*variant, base, "field {i} must change the digest");
    }
    // Deterministic: the same inputs give the same digest. The remaining budget, term, epoch and generation are
    // not parameters at all, so a retry with a smaller budget cannot change it.
    assert_eq!(
        wire::payload_sha256(wire::METHOD_SUBMIT, "a", "r", 1, WORK_UNIT, INPUT),
        base
    );
}

#[test]
fn digest_length_prefixing_removes_concatenation_ambiguity() {
    let m = wire::METHOD_SUBMIT;
    assert_ne!(
        wire::payload_sha256(m, "ab", "c", 1, b"", b""),
        wire::payload_sha256(m, "a", "bc", 1, b"", b"")
    );
    assert_ne!(
        wire::payload_sha256(m, "a", "r", 1, b"xy", b"z"),
        wire::payload_sha256(m, "a", "r", 1, b"x", b"yz")
    );
    // attempt is decimal ASCII between request_id and work_unit_json: ("r1", 1) != ("r", 11).
    assert_ne!(
        wire::payload_sha256(m, "a", "r1", 1, b"", b""),
        wire::payload_sha256(m, "a", "r", 11, b"", b"")
    );
}

#[test]
fn lifecycle_digest_matches_independent_vectors() {
    // Python hashlib over u64-BE length prefixes of method, scope_id, the 32 raw lock bytes, instance_id and then
    // each device_id (m36 design note Q6).
    let lock: Vec<u8> = (0u8..32).collect();
    let load = wire::lifecycle_sha256(wire::METHOD_LOAD, "_cere_control", &lock, "", &[]);
    assert_eq!(
        hex(&load),
        "a4a80b6b2511d694754c230f506097f3f8d262b7b76c67031c549f4875bf831f"
    );
    let devices = ["gpu0".to_owned(), "gpu1".to_owned()];
    let drain = wire::lifecycle_sha256(
        wire::METHOD_DRAIN,
        "_cere_control",
        &lock,
        "inst-1",
        &devices,
    );
    assert_eq!(
        hex(&drain),
        "b7a67d115f40efd986d78eec10eb77f26fe7c83cad67aa8e65ab257b50ce579c"
    );
    // Each device id is its own length-prefixed part: ["ab"] and ["a", "b"] differ.
    assert_ne!(
        wire::lifecycle_sha256(wire::METHOD_UNLOAD, "s", &lock, "", &["ab".to_owned()]),
        wire::lifecycle_sha256(
            wire::METHOD_UNLOAD,
            "s",
            &lock,
            "",
            &["a".to_owned(), "b".to_owned()]
        )
    );
}

#[test]
fn lifecycle_methods_and_reasons_are_the_spec_strings() {
    assert_eq!(
        [wire::METHOD_LOAD, wire::METHOD_DRAIN, wire::METHOD_UNLOAD],
        ["LoadModel", "DrainModel", "UnloadModel"]
    );
    assert_eq!(
        [
            wire::LOCK_NOT_APPROVED,
            wire::MODEL_UNMANAGED,
            wire::MODEL_BUSY,
            wire::UNSUPPORTED_PLACEMENT,
            wire::MODEL_DRAINING,
            wire::MODEL_LOAD_FAILED,
            wire::BACKEND_EXITED,
            wire::BACKEND_PORT_IN_USE,
        ],
        [
            "lock_not_approved",
            "model_unmanaged",
            "model_busy",
            "unsupported_placement",
            "model_draining",
            "model_load_failed",
            "backend_exited",
            "backend_port_in_use",
        ]
    );
}

const LOCK: &str = "4f1c0b8e2a7d9e3f5a6b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f";

/// An attached-mode (M3.5 agent without `[backend.launch]`) inventory.
fn inventory() -> Inventory {
    Inventory {
        schema_version: 2,
        worker_id: "worker-a".into(),
        served_model_name: "Qwen3.8-Flash-Next".into(),
        tp_size: Some(1),
        kv_cache_dtype: Some("fp8_e4m3".into()),
        max_running_requests: Some(8),
        instance_id: Some("inst-1".into()),
        model_generation: 3,
        admission_state: AdmissionState::Fenced,
        gpus: vec![],
        models: vec![InventoryModel {
            served_model_name: "Qwen3.8-Flash-Next".into(),
            lock_sha256: None,
            state: ModelState::Attached,
            instance_id: Some("inst-1".into()),
            model_generation: 3,
        }],
        artifacts: vec![],
    }
}

/// A managed inventory in `state`, with the backend identity confirmed (`confirmed`) or not.
fn managed(state: ModelState, confirmed: bool) -> Inventory {
    let mut inv = inventory();
    if !confirmed {
        inv.tp_size = None;
        inv.kv_cache_dtype = None;
        inv.max_running_requests = None;
        inv.instance_id = None;
    }
    inv.models[0].lock_sha256 = Some(LOCK.into());
    inv.models[0].state = state;
    inv.models[0].instance_id = inv.instance_id.clone();
    inv
}

#[test]
fn inventory_round_trips_and_hash_is_over_exact_bytes() {
    let inv = inventory();
    let bytes = inv.to_json_bytes().unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains(r#""admission_state":"fenced""#), "{text}");
    assert!(text.contains(r#""gpus":[]"#), "{text}");
    assert!(
        text.contains(r#""lock_sha256":null,"state":"attached""#),
        "{text}"
    );
    assert_eq!(Inventory::from_json_bytes(&bytes).unwrap(), inv);
    assert_eq!(
        inv.to_json_bytes().unwrap(),
        bytes,
        "serialization is stable"
    );
    assert_eq!(Inventory::sha256(&bytes), Inventory::sha256(&bytes));
    // The hash is over the exact bytes: a semantically equal but reformatted document hashes differently.
    let mut spaced = bytes.clone();
    spaced.push(b' ');
    assert_eq!(Inventory::from_json_bytes(&spaced).unwrap(), inv);
    assert_ne!(Inventory::sha256(&spaced), Inventory::sha256(&bytes));
    // A content change (here the admission state) changes the hash.
    let mut open = inv.clone();
    open.admission_state = AdmissionState::Open;
    assert_ne!(
        Inventory::sha256(&open.to_json_bytes().unwrap()),
        Inventory::sha256(&bytes)
    );
    // The model state is hashed too (Q7: every state change sends a Ready worker back to Joining).
    let ready = managed(ModelState::Ready, true).to_json_bytes().unwrap();
    let draining = managed(ModelState::Draining, true).to_json_bytes().unwrap();
    assert_ne!(Inventory::sha256(&ready), Inventory::sha256(&draining));
}

#[test]
fn inventory_v2_accepts_every_model_state_with_its_identity_rule() {
    // Q7: the identity fields are set in attached mode and in `ready`; null in `unloaded`/`loading`/`failed`;
    // either in `draining`/`unloading` (the probe may lose the backend there).
    let accepted = [
        managed(ModelState::Unloaded, false),
        managed(ModelState::Loading, false),
        managed(ModelState::Failed, false),
        managed(ModelState::Ready, true),
        managed(ModelState::Draining, true),
        managed(ModelState::Draining, false),
        managed(ModelState::Unloading, true),
        managed(ModelState::Unloading, false),
        inventory(),
    ];
    for inv in accepted {
        let bytes = inv
            .to_json_bytes()
            .unwrap_or_else(|e| panic!("{:?}: {e}", inv.models[0].state));
        let back = Inventory::from_json_bytes(&bytes).unwrap();
        assert_eq!(back.model().unwrap().state, inv.models[0].state);
        assert_eq!(back, inv);
    }
    let refused = [
        managed(ModelState::Ready, false),
        managed(ModelState::Unloaded, true),
        managed(ModelState::Loading, true),
        managed(ModelState::Failed, true),
    ];
    for inv in refused {
        assert!(
            Inventory::from_json_bytes(&serde_json::to_vec(&inv).unwrap()).is_err(),
            "{:?}",
            inv.models[0].state
        );
        assert!(inv.to_json_bytes().is_err());
    }
    // Attached mode is only described once confirmed.
    let mut unconfirmed = inventory();
    unconfirmed.tp_size = None;
    unconfirmed.kv_cache_dtype = None;
    unconfirmed.max_running_requests = None;
    unconfirmed.instance_id = None;
    unconfirmed.models[0].instance_id = None;
    assert!(unconfirmed.to_json_bytes().is_err());
}

#[test]
fn model_state_strings() {
    for (state, text) in [
        (ModelState::Attached, "attached"),
        (ModelState::Unloaded, "unloaded"),
        (ModelState::Loading, "loading"),
        (ModelState::Ready, "ready"),
        (ModelState::Draining, "draining"),
        (ModelState::Unloading, "unloading"),
        (ModelState::Failed, "failed"),
    ] {
        assert_eq!(state.as_str(), text);
        assert_eq!(ModelState::parse(text), Some(state));
        assert_eq!(
            serde_json::to_value(state).unwrap(),
            serde_json::Value::from(text)
        );
    }
    assert_eq!(ModelState::parse("preparing"), None);
}

#[test]
fn inventory_refuses_unknown_fields_and_bad_values() {
    let value: serde_json::Value =
        serde_json::from_slice(&inventory().to_json_bytes().unwrap()).unwrap();
    let unloaded: serde_json::Value = serde_json::from_slice(
        &managed(ModelState::Unloaded, false)
            .to_json_bytes()
            .unwrap(),
    )
    .unwrap();
    let refused_from = |base: &serde_json::Value, edit: &dyn Fn(&mut serde_json::Value)| {
        let mut v = base.clone();
        edit(&mut v);
        Inventory::from_json_bytes(&serde_json::to_vec(&v).unwrap()).is_err()
    };
    let refused = |edit: &dyn Fn(&mut serde_json::Value)| refused_from(&value, edit);
    assert!(refused(&|v| v["extra"] = 1.into()));
    assert!(refused(&|v| v["models"][0]["extra"] = 1.into()));
    assert!(refused(&|v| v["schema_version"] = 1.into()));
    assert!(refused(&|v| v["schema_version"] = 3.into()));
    assert!(refused(&|v| v["admission_state"] = "ready".into()));
    assert!(refused(&|v| v["gpus"] = serde_json::json!([{"index": 0}])));
    assert!(refused(&|v| v["artifacts"] = serde_json::json!(["x"])));
    assert!(refused(&|v| v["tp_size"] = 0.into()));
    assert!(refused(&|v| v["max_running_requests"] = 0.into()));
    assert!(refused(&|v| v["worker_id"] = "".into()));
    assert!(refused(&|v| {
        v["instance_id"] = "x".repeat(257).into();
        v["models"][0]["instance_id"] = "x".repeat(257).into();
    }));
    assert!(refused(&|v| v["models"][0]["served_model_name"] = "".into()));
    assert!(refused(&|v| {
        v.as_object_mut().unwrap().remove("models");
    }));
    // Exactly one model, and it repeats the top-level name, instance and generation.
    assert!(refused(&|v| v["models"] = serde_json::json!([])));
    assert!(refused(&|v| {
        let model = v["models"][0].clone();
        v["models"].as_array_mut().unwrap().push(model);
    }));
    assert!(refused(
        &|v| v["models"][0]["served_model_name"] = "other".into()
    ));
    assert!(refused(&|v| v["models"][0]["instance_id"] = "inst-2".into()));
    assert!(refused(&|v| v["models"][0]["model_generation"] = 4.into()));
    // Nullable is not optional: the key must be present.
    assert!(refused(&|v| {
        v.as_object_mut().unwrap().remove("tp_size");
    }));
    assert!(refused(&|v| {
        v["models"][0]
            .as_object_mut()
            .unwrap()
            .remove("lock_sha256");
    }));
    // The identity fields are all set or all null.
    assert!(refused(&|v| v["kv_cache_dtype"] = serde_json::Value::Null));
    // Attached mode has no lock; managed mode has a 64-lowercase-hex lock.
    assert!(refused(&|v| v["models"][0]["lock_sha256"] = LOCK.into()));
    assert!(refused_from(&unloaded, &|v| v["models"][0]
        ["lock_sha256"] =
        serde_json::Value::Null));
    assert!(refused_from(&unloaded, &|v| v["models"][0]
        ["lock_sha256"] =
        LOCK.to_uppercase().into()));
    assert!(refused_from(&unloaded, &|v| v["models"][0]
        ["lock_sha256"] =
        LOCK[..63].into()));
    assert!(refused_from(&unloaded, &|v| v["models"][0]["state"] = "preparing".into()));
    assert!(!refused(&|_| {}));
    assert!(!refused_from(&unloaded, &|_| {}));
}

#[test]
fn inventory_over_64_kib_is_refused_before_parsing() {
    // Exactly at the bound is accepted by the parser (whitespace padding keeps the content valid).
    let mut small = inventory().to_json_bytes().unwrap();
    small.resize(wire::INVENTORY_MAX, b' ');
    assert!(Inventory::from_json_bytes(&small).is_ok());
    small.push(b' ');
    assert!(matches!(
        Inventory::from_json_bytes(&small),
        Err(InventoryError::TooLarge(len)) if len == wire::INVENTORY_MAX + 1
    ));
}

#[test]
fn admission_state_strings() {
    for (state, text) in [
        (AdmissionState::Open, "open"),
        (AdmissionState::Fenced, "fenced"),
        (AdmissionState::Poisoned, "poisoned"),
    ] {
        assert_eq!(state.as_str(), text);
        assert_eq!(AdmissionState::parse(text), Some(state));
    }
    assert_eq!(AdmissionState::parse("Open"), None);
    assert_eq!(AdmissionState::parse(""), None);
}

#[test]
fn status_message_is_exactly_the_reason_code() {
    let status = wire::status(Code::FailedPrecondition, wire::STALE_TERM);
    assert_eq!(status.code(), Code::FailedPrecondition);
    assert_eq!(status.message(), "stale_term");
    assert_eq!(wire::reason(&status), wire::STALE_TERM);
    assert_eq!(wire::PERMISSION_DENIED_REASON, "permission_denied");
    assert_eq!(wire::QUOTA_EXHAUSTED, "quota_exhausted");
}

#[test]
fn limit_constants_match_q14() {
    assert_eq!(wire::CONTROL_MESSAGE_MAX, 1_048_576);
    assert_eq!(wire::INFERENCE_MESSAGE_MAX, 8_388_608);
    assert_eq!(wire::LIST_PAGE_MAX, 256);
    assert_eq!(wire::WORK_EVENT_MAX, 65_536);
    assert_eq!(wire::INVENTORY_MAX, 65_536);
    assert_eq!(wire::WORK_CAP_MS, 600_000);
    assert_eq!(wire::METHOD_SUBMIT, "SubmitWork");
    assert_eq!(wire::METHOD_CANCEL, "CancelWork");
    assert_eq!(wire::DESIRED_DRAINING, "DRAINING");
}

#[test]
fn quota_is_per_principal_and_released_on_drop() {
    let quota = PrincipalQuota::new(64);
    let mut held: Vec<_> = (0..64)
        .map(|_| quota.try_acquire("cere-worker-a").unwrap())
        .collect();
    let refused = quota.try_acquire("cere-worker-a").err().unwrap();
    assert_eq!(refused.code(), Code::ResourceExhausted);
    assert_eq!(wire::reason(&refused), wire::QUOTA_EXHAUSTED);
    // Another principal has its own pool.
    let other = quota.try_acquire("cere-worker-b").unwrap();
    drop(held.pop());
    let again = quota.try_acquire("cere-worker-a").unwrap();
    assert!(quota.try_acquire("cere-worker-a").is_err());
    drop(again);
    drop(held);
    drop(other);
    let all: Vec<_> = (0..64)
        .map(|_| quota.try_acquire("cere-worker-a").unwrap())
        .collect();
    assert_eq!(all.len(), 64);
}

#[test]
fn quota_guard_outlives_a_moved_quota_handle() {
    // Guards are moved into response streams: they must be 'static + Send and still release.
    fn assert_static_send<T: Send + 'static>(_: &T) {}
    let quota = PrincipalQuota::new(1);
    let guard = quota.try_acquire("p").unwrap();
    assert_static_send(&guard);
    let worker = std::thread::spawn(move || drop(guard));
    worker.join().unwrap();
    assert!(quota.try_acquire("p").is_ok());
}
