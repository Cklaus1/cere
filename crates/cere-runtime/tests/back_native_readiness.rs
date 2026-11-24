//! Native readiness (m22 design note R4, "Native path contract"): the real `cered` binary with a `native_cuda`
//! config over the CPU toy executor, the fixture artifact and the stub tokenizer sidecar. `/readyz` is ready only
//! once every identity check and the 1-token SMOKE passed; each identity mismatch leaves it not ready with a
//! problem naming the field. Every case is its own process, so `CUDA_VISIBLE_DEVICES` is per case.
mod native_server;
mod native_support;
mod support;

use std::time::{Duration, Instant};

use native_server::{file_sha256, worker, Case, Native, OTHER_UUID, ZERO_SHA};
use native_support::{toy_library_noforward, Tmp};
use serde_json::Value;

/// The problem fields of the native worker's Q11 `/readyz` entry.
fn fields(body: &Value) -> Vec<&str> {
    worker(body)["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["field"].as_str().unwrap())
        .collect()
}

async fn not_ready_naming(case: Case, expected: &[&str]) -> Value {
    let native = Native::start(case).await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["ready"], false, "{body}");
    assert_eq!(worker(&body)["ready"], false, "{body}");
    assert_eq!(body["problems"], serde_json::json!([]), "{body}");
    assert_eq!(fields(&body), expected, "{body}");
    body
}

#[tokio::test]
async fn ready_after_every_identity_check_and_the_smoke() {
    let native = Native::start(Case::matching()).await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(fields(&body), Vec::<&str>::new());
    let entry = worker(&body);
    assert_eq!(entry["state"], "ready", "{body}");
    assert_eq!(entry["ready"], true, "{body}");
    assert_eq!(entry["smoke"]["state"], "passed", "{body}");
    let instance = entry["instance_id"].as_str().unwrap();
    assert!(instance.starts_with("native-"), "{instance}");
    assert_eq!(entry["smoke"]["instance_id"], instance);
    assert_eq!(entry["served_model_name"], "Qwen3.8-Flash-Next");
}

#[tokio::test]
async fn library_hash_mismatch_names_library_sha256() {
    let case = Case {
        library_pin: ZERO_SHA.to_owned(),
        ..Case::matching()
    };
    not_ready_naming(case, &["library_sha256"]).await;
}

#[tokio::test]
async fn manifest_mismatch_names_expected_manifest_sha256() {
    let case = Case {
        manifest_pin: Some(ZERO_SHA.to_owned()),
        ..Case::matching()
    };
    not_ready_naming(case, &["expected_manifest_sha256"]).await;
}

#[tokio::test]
async fn tokenizer_and_template_mismatches_name_their_fields() {
    let case = Case {
        tokenizer_sha256: ZERO_SHA.to_owned(),
        ..Case::matching()
    };
    not_ready_naming(case, &["tokenizer_sha256"]).await;
    let case = Case {
        template_sha256: ZERO_SHA.to_owned(),
        ..Case::matching()
    };
    not_ready_naming(case, &["template_sha256"]).await;
}

/// A library without a forward pass reports `unsupported` and no operations; the profile pins `native` and
/// prefill/decode/cancel.
#[tokio::test]
async fn capability_mismatch_names_each_capabilities_field() {
    let library = toy_library_noforward();
    let case = Case {
        library_pin: file_sha256(&library),
        library,
        ..Case::matching()
    };
    // The no-forward toy reports `native_decode_batch` 0 at any rows; the profile's rows need it (m26 D3, D10).
    // It has no sampler, so no vocab mask either (A3), which the `-v8` grammar outputs need (M2.8 D10).
    let body = not_ready_naming(
        case,
        &[
            "capabilities.execution_mode",
            "capabilities.operations",
            "capabilities.native_decode_batch",
            "capabilities.native_token_mask",
            "capabilities.native_sampler",
        ],
    )
    .await;
    assert!(worker(&body)["problems"][0]["detail"]
        .as_str()
        .unwrap()
        .contains("\"unsupported\""));
}

/// M2.7 D11: the shipped profile has sampled decode rows, so the library must report `native_sampler` 1; the toy
/// without its sampler (`sampler=0`, everything else matching) is not ready, naming that field and the vocab mask
/// (A3: the library reports the mask with the sampler; the `-v8` grammar outputs need it, M2.8 D10).
#[tokio::test]
async fn sampled_rows_without_native_sampler_name_capabilities_native_sampler() {
    let case = Case {
        device_id: "toy,sampler=0".to_owned(),
        ..Case::matching()
    };
    let body = not_ready_naming(
        case,
        &[
            "capabilities.native_token_mask",
            "capabilities.native_sampler",
        ],
    )
    .await;
    assert!(worker(&body)["problems"][1]["detail"]
        .as_str()
        .unwrap()
        .contains("native_sampler"));
}

#[tokio::test]
async fn device_binding_mismatch_names_device_uuid() {
    let case = Case {
        cuda_visible_devices: OTHER_UUID,
        ..Case::matching()
    };
    not_ready_naming(case, &["device_uuid"]).await;
}

/// m25 design note D4, C3: the configured `kv_pool_tokens` reaches `cere_model_create`; a pool over the executor's
/// support (1,048,576 on the toy, ctx262 design note CX-01) passes config validation and is refused there
/// (UNSUPPORTED): a `load` problem.
#[tokio::test]
async fn a_pool_over_the_executor_support_is_a_load_problem() {
    let case = Case {
        kv_pool_tokens: Some(1_048_640),
        ..Case::matching()
    };
    let body = not_ready_naming(case, &["load"]).await;
    assert_eq!(
        worker(&body)["problems"][0]["detail"],
        "cere_model_create: unsupported",
        "{body}"
    );
}

/// m25 design note D4: `kv_pool_tokens = 0` is the derived pool (one session x round64(200,000) = 200,000), which the
/// executor reports and readiness accepts.
#[tokio::test]
async fn the_derived_pool_is_ready() {
    let case = Case {
        kv_pool_tokens: Some(0),
        ..Case::matching()
    };
    let native = Native::start(case).await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(fields(&body), Vec::<&str>::new());
}

/// The sidecar answers `info` and dies before the SMOKE's `encode_chat`: SMOKE failed, never ready.
#[tokio::test]
async fn failed_smoke_is_not_ready() {
    let case = Case {
        sidecar_exit_after: Some(1),
        ..Case::matching()
    };
    let body = not_ready_naming(case, &["smoke"]).await;
    assert_eq!(worker(&body)["smoke"]["state"], "failed", "{body}");
}

/// M2.8 C1 (native half): the shipped profile (`-v8`) pins grammar outputs, which need `native_token_mask` 1; the
/// toy without its sampler reports 0, and with the sampled rows dropped from the profile only the mask is named.
#[tokio::test]
async fn grammar_output_without_native_token_mask_names_capabilities_native_token_mask() {
    let path =
        support::repo_root().join(format!("{}.capabilities.json", native_server::DEPLOYMENT));
    let shipped: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let greedy: Vec<Value> = shipped["validated_combinations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["sampler_mode"] != "sampled")
        .cloned()
        .collect();
    let case = Case {
        device_id: "toy,sampler=0".to_owned(),
        capabilities_patch: serde_json::json!({ "validated_combinations": greedy }),
        ..Case::matching()
    };
    let body = not_ready_naming(case, &["capabilities.native_token_mask"]).await;
    assert!(worker(&body)["problems"][0]["detail"]
        .as_str()
        .unwrap()
        .contains("native_token_mask 1"));
}

/// M2.8 C4: the grammar process must report the dialect's xgrammar version (`json-schema-2020-12/xgrammar-0.2.1`,
/// the shipped `-v8` profile's).
#[tokio::test]
async fn grammar_sidecar_xgrammar_mismatch_names_grammar() {
    let case = Case {
        sidecar_args: vec!["--xgrammar".to_owned(), "0.2.0".to_owned()],
        ..Case::matching()
    };
    let body = not_ready_naming(case, &["grammar"]).await;
    let detail = worker(&body)["problems"][0]["detail"].as_str().unwrap();
    assert!(
        detail.contains("0.2.0") && detail.contains("0.2.1"),
        "{detail}"
    );
}

/// M2.8 C4: a grammar process without the `grammar` feature is refused.
#[tokio::test]
async fn grammar_sidecar_without_the_grammar_feature_names_grammar() {
    let case = Case {
        sidecar_args: vec!["--no-grammar-feature".to_owned()],
        ..Case::matching()
    };
    let body = not_ready_naming(case, &["grammar"]).await;
    let detail = worker(&body)["problems"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("feature grammar"), "{detail}");
}

/// M2.12 C3 (D4, ledger S): a native profile declaring `thinking_toggle` needs a sidecar serving the feature
/// `thinking`; without it the deployment is not ready, naming the sidecar.
#[tokio::test]
async fn thinking_profile_needs_the_sidecar_feature() {
    let case = Case {
        sidecar_features: Some("grammar"),
        capabilities_patch: serde_json::json!({"thinking_toggle": true}),
        ..Case::matching()
    };
    let body = not_ready_naming(case, &["sidecar"]).await;
    assert!(body.to_string().contains("thinking"), "{body}");
}

/// m29 design note C6 (D11/D12, I3): the speculation profile (`verify`, `native_speculative_batch` true, 4 draft
/// tokens) is never ready on a library without speculation: the no-forward toy reports none (or refuses the drafts
/// desc at ALLOCATE), so the load is refused before the SMOKE and no instance is reported.
#[tokio::test]
async fn speculation_profile_without_executor_speculation_is_not_ready() {
    let library = toy_library_noforward();
    let case = Case {
        library_pin: file_sha256(&library),
        library,
        speculation: true,
        ..Case::matching()
    };
    let native = Native::start(case).await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 503, "{body}");
    assert_eq!(worker(&body)["ready"], false, "{body}");
    assert_eq!(worker(&body)["instance_id"], Value::Null, "{body}");
    let fields = fields(&body);
    assert!(
        fields == ["load"] || fields.contains(&"capabilities.native_speculative_batch"),
        "{body}"
    );
}

/// m29 design note C6 (D11/D12, I3): over the HOST track's toy speculation the speculation profile is ready (the
/// toy reports `verify`, `native_speculative_batch` 1 and 4 draft tokens, matching the profile and config). Until
/// the toy accepts the drafts desc, its ALLOCATE refusal (UNSUPPORTED) is reported on stderr instead.
#[tokio::test]
async fn speculation_profile_is_ready_over_the_speculating_toy() {
    let native = Native::start(Case {
        speculation: true,
        ..Case::matching()
    })
    .await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(fields(&body), Vec::<&str>::new());
    assert_eq!(worker(&body)["smoke"]["state"], "passed", "{body}");
}

/// load-speed design note LS-04: a [`Case::matching`] whose tokenizer and grammar stub processes (both get
/// `extra` arguments) hold their start (no `info` answer) while a file exists. The toy writes its tensor digests at
/// INSTANTIATE, so that file appearing while the hold is in place shows the model loaded beside the startup; then
/// the hold is released and the load's outcome awaited. Fails on main 76071574: the load waited for the sidecar.
async fn settled_after_a_held_start(tag: &str, extra: &[&str]) -> (Native, u16, Value) {
    let t = Tmp::new(tag);
    let hold = t.join("hold");
    let digests = t.join("digests");
    std::fs::write(&hold, b"").unwrap();
    let mut sidecar_args = vec!["--hold-start-while".to_owned(), hold.display().to_string()];
    sidecar_args.extend(extra.iter().map(|a| (*a).to_owned()));
    let native = Native::start(Case {
        device_id: format!("toy,tensor_digests={}", digests.display()),
        sidecar_args,
        ..Case::matching()
    })
    .await;
    let until = Instant::now() + Duration::from_secs(60);
    while !digests.exists() {
        assert!(
            Instant::now() < until,
            "no INSTANTIATE while the processes held their start: {}",
            native.cered.stderr_so_far()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    std::fs::remove_file(&hold).unwrap();
    let (status, body) = native.settled().await;
    (native, status, body)
}

/// LS-04: the model loads while the tokenizer and grammar processes start up; once they answer the worker is
/// ready and the load's stages line names each step.
#[tokio::test]
async fn the_model_loads_while_the_sidecar_starts() {
    let (native, status, body) = settled_after_a_held_start("ls04-overlap", &[]).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(fields(&body), Vec::<&str>::new());
    let log = native.cered.stderr_so_far();
    let stages = log
        .lines()
        .find(|l| l.starts_with("cered readiness: native load stages "))
        .unwrap_or_else(|| panic!("no stages line: {log}"));
    for step in [
        "upload_ms=",
        "sidecar_ms=",
        "grammar_ms=",
        "smoke_ms=",
        "total_ms=",
    ] {
        assert!(stages.contains(step), "{stages}");
    }
}

/// LS-04: a tokenizer sidecar that dies at startup while the model loads beside it leaves the worker not ready,
/// naming `sidecar`; the loaded model is released and no backend is reported ready.
#[tokio::test]
async fn a_sidecar_dying_during_the_load_is_not_ready() {
    let (native, status, body) =
        settled_after_a_held_start("ls04-sidecar", &["--die-at-start-role", "tokenizer"]).await;
    assert_eq!(status, 503, "{body}");
    assert_eq!(worker(&body)["ready"], false, "{body}");
    assert_eq!(fields(&body), vec!["sidecar"], "{body}");
    let log = native.cered.stderr_so_far();
    assert!(!log.contains("native backend ready"), "{log}");
}

/// LS-04: a grammar process that dies at startup while the model loads beside it leaves the worker not ready,
/// naming `grammar`; no backend is reported ready.
#[tokio::test]
async fn a_grammar_process_dying_during_the_load_is_not_ready() {
    let (native, status, body) =
        settled_after_a_held_start("ls04-grammar", &["--die-at-start-role", "grammar"]).await;
    assert_eq!(status, 503, "{body}");
    assert_eq!(worker(&body)["ready"], false, "{body}");
    assert_eq!(fields(&body), vec!["grammar"], "{body}");
    let log = native.cered.stderr_so_far();
    assert!(!log.contains("native backend ready"), "{log}");
}

/// LS-04: a process that died at its startup beside the load (as a startup deadline missed on a loaded host does)
/// starts once more alone after the load; when that start answers the worker is ready, and the stages line has the
/// second start's milliseconds. Fails before the retry: the worker stayed not ready naming the process.
async fn ready_after_a_first_start_died(tag: &str, role: &str, field: &str, lap: &str) {
    let t = Tmp::new(tag);
    let first = t.join("first");
    let first = first.display().to_string();
    let (native, status, body) =
        settled_after_a_held_start(tag, &["--die-first-start-role", role, &first]).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(fields(&body), Vec::<&str>::new());
    let log = native.cered.stderr_so_far();
    assert!(
        log.contains(&format!("the {field} did not start beside the load")),
        "{log}"
    );
    let stages = log
        .lines()
        .find(|l| l.starts_with("cered readiness: native load stages "))
        .unwrap_or_else(|| panic!("no stages line: {log}"));
    assert!(stages.contains(lap), "{stages}");
}

#[tokio::test]
async fn a_sidecar_that_missed_its_start_beside_the_load_starts_again_alone() {
    ready_after_a_first_start_died(
        "ls04-sidecar-again",
        "tokenizer",
        "tokenizer sidecar",
        "sidecar_again_ms=",
    )
    .await;
}

#[tokio::test]
async fn a_grammar_process_that_missed_its_start_beside_the_load_starts_again_alone() {
    ready_after_a_first_start_died(
        "ls04-grammar-again",
        "grammar",
        "grammar process",
        "grammar_again_ms=",
    )
    .await;
}
