//! Shared harness of the tests that run the real `cered` with a `native_cuda` config (spec m22 R4, R5): a scratch
//! repository holding the native deployment re-pinned to a [`Case`], the fixture artifact, the stub tokenizer
//! sidecar (`native_support/stub_sidecar.py`) and the CPU toy executor. Every case is its own process, so
//! `CUDA_VISIBLE_DEVICES` and the toy device options are per case.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cere_runtime::native::snapshot::arena_slot_bytes;
use cere_runtime::readiness::NATIVE_DEVICE_ID_ENV;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::native_support::{
    self, toy_library, verified_artifact, verified_artifact_with_tokenizer, Tmp, FIXTURE_VOCAB,
};
use crate::support::{self, http, Cered, StateDir};

pub const DEPLOYMENT: &str = "configs/deployments/qwen38-flash-next-native";
/// The M2.9 speculation deployment (`-v19`, m29 design note D12).
pub const SPEC_DEPLOYMENT: &str = "configs/deployments/qwen38-flash-next-native-spec";
pub const DEVICE_UUID: &str = "GPU-aaaaaaaa-0000-4000-8000-000000000000";
pub const OTHER_UUID: &str = "GPU-00000000-0000-0000-0000-000000000000";
pub const ZERO_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// The bearer token of scope `native-a` (`CERE_TOKEN_NATIVE_A`).
pub const TOKEN: &str = "secret-a";
/// The operator token (`[auth] operator_token_env = "CERE_OPERATOR_TOKEN"`, M2.10 N2 (a)).
pub const OPERATOR_TOKEN: &str = "secret-operator";
/// The native worker: `scheduler.devices[0]` of `configs/runtime/native-g0.toml` (M3 Q3, C3).
pub const WORKER: &str = "gpu0";
/// m32n design note D2: the second replica of a [`Case::replicas`] deployment and its card.
pub const SECOND_WORKER: &str = "gpu1";
pub const SECOND_UUID: &str = "GPU-bbbbbbbb-1111-4111-8111-111111111111";
/// `CUDA_VISIBLE_DEVICES` of a [`Case::replicas`] deployment: both cards in replica order (D3).
pub const BOTH_UUIDS: &str =
    "GPU-aaaaaaaa-0000-4000-8000-000000000000,GPU-bbbbbbbb-1111-4111-8111-111111111111";

/// The Q11 `/readyz` entry of [`WORKER`].
pub fn worker(body: &Value) -> &Value {
    &body["workers"][WORKER]
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .concat()
}

pub fn file_sha256(path: &Path) -> String {
    sha256_hex(&std::fs::read(path).unwrap())
}

/// The committed deployment lock.
pub fn shipped_lock() -> Value {
    let path = support::repo_root().join(format!("{DEPLOYMENT}.lock.json"));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// M2.13 image switches of a [`Case`]; all off is the text-only deployment.
#[derive(Clone, Copy, Debug, Default)]
pub struct Images {
    /// The artifact carries the A4 text config, its `vision_config` and device-placed `model.visual.*` tensors, so
    /// the toy loads it against its `a4_vision_support` entry (`native_image_input` 1).
    pub artifact: bool,
    /// The profile gains the greedy image decode combination (rows [1], [`IMAGE_CONTEXT`]) and its text twin.
    pub combo: bool,
    /// The stub sidecar serves feature `image` (reporting the artifact's `preprocessor_config.json` digest).
    pub sidecar: bool,
    /// The stub reports every image's `token_offset` one row late.
    pub bad_offset: bool,
}

/// The context of the image decode combination: the A4 `indexer_budget` (D5, D11).
pub const IMAGE_CONTEXT: u64 = 2048;

/// The fixture's `config.json` with the A4 `text_config` (the keys the desc reads, `$ART/config.json` values)
/// and the A4 vision tower: equal to the toy's `a4_vision_support` entry. The fixture's `eos_token_id` stays.
fn a4_vision_config(fixture: &Value) -> Value {
    let layer_types: Vec<&str> = (0..48)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let mut config = fixture.clone();
    config["text_config"] = json!({
        "model_type": "qwen4_exp_text",
        "eos_token_id": fixture["text_config"]["eos_token_id"],
        "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
        "layer_types": layer_types, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
        "output_gate_type": "sigmoid",
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                            "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_compress_ratio": 4,
        "indexer_budget": IMAGE_CONTEXT, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32", "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640, "hc_count": 4, "hc_lowrank": 320, "ple_layer_ids": [2],
        "ple_embed_dim": 2560, "heads_per_ngram": 8, "ngram_size": 3, "ple_conv_kernel_size": 4,
        "ngram_vocab_size_base": 20000000, "split_ngram_parts": 128, "make_ngram_vocab_size_divisible_by": 128,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": false, "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262144,
    });
    config["vision_config"] = json!({
        "deepstack_visual_indexes": [], "depth": 27, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152,
        "in_channels": 3, "intermediate_size": 4304, "model_type": "qwen4_exp", "num_heads": 16,
        "num_position_embeddings": 2304, "out_hidden_size": 2560, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    });
    config["image_token_id"] = json!(248056);
    config["video_token_id"] = json!(248057);
    config["vision_start_token_id"] = json!(248053);
    config["vision_end_token_id"] = json!(248054);
    config
}

/// [`verified_artifact`] turned into a vision artifact: `config.json` from [`a4_vision_config`] and the `vision`
/// role placed on the device (profile, family rows, totals), the manifest re-hashed and re-encoded canonically.
fn vision_artifact(t: &Tmp) -> (PathBuf, String) {
    let (dir, _) = verified_artifact(t);
    let fixture: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let config = serde_json::to_vec_pretty(&a4_vision_config(&fixture)).unwrap();
    std::fs::write(dir.join("config.json"), &config).unwrap();
    let mut m: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let entry = m["files"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|f| f["path"] == "config.json")
        .unwrap();
    entry["bytes"] = json!(config.len());
    entry["sha256"] = json!(sha256_hex(&config));
    m["profile"]["placement"]["vision"] = json!("device");
    let (mut count, mut bytes) = (0, 0);
    for row in m["families"].as_array_mut().unwrap() {
        if row["role"] == "vision" {
            row["placement"] = json!("device");
            count += row["count"].as_u64().unwrap();
            bytes += row["bytes"].as_u64().unwrap();
        }
    }
    let totals = m["totals"]["by_placement"].as_object_mut().unwrap();
    for (place, sign) in [("unused", -1i64), ("device", 1)] {
        let slot = totals
            .entry(place)
            .or_insert(json!({"bytes": 0, "count": 0}));
        for (key, n) in [("bytes", bytes), ("count", count)] {
            slot[key] = json!(slot[key].as_i64().unwrap() + sign * n as i64);
        }
    }
    if totals["unused"]["count"] == 0 {
        totals.remove("unused");
    }
    let files: Vec<cere_compiler::manifest::FileEntry> =
        serde_json::from_value(m["files"].clone()).unwrap();
    let (model_hash, tokenizer_hash, template_hash) =
        cere_compiler::manifest::identity_hashes(&files);
    m["model_hash"] = json!(model_hash);
    m["tokenizer_hash"] = json!(tokenizer_hash);
    m["template_hash"] = json!(template_hash);
    let bytes = cere_compiler::canon_json::to_vec(&m).unwrap();
    std::fs::write(dir.join("manifest.json"), &bytes).unwrap();
    let sha = sha256_hex(&bytes);
    std::fs::write(dir.join("manifest.sha256"), format!("{sha}\n")).unwrap();
    let art =
        cere_compiler::artifact::open(&dir, &sha, cere_compiler::artifact::Mode::Full).unwrap();
    art.write_full_verify_stamp().unwrap();
    (dir, sha)
}

/// The profile's greedy image decode combination and its text twin (same sampler, rows and context, D5).
fn add_image_combo(caps: &mut Value) {
    caps["modalities"]
        .as_array_mut()
        .unwrap()
        .push(json!("image"));
    let combos = caps["validated_combinations"].as_array_mut().unwrap();
    for modality in ["text", "image"] {
        combos.push(json!({
            "operation": "decode", "modality": modality, "output": "text", "numerical_mode": "optimized",
            "sampler_mode": "greedy", "row_counts": [1], "max_context_tokens": IMAGE_CONTEXT,
        }));
    }
}

/// One native deployment under test: what the config and lock pin, and what the process sees.
pub struct Case {
    pub library: PathBuf,
    /// `library_sha256` in the config and `backend_build_sha256` in the lock.
    pub library_pin: String,
    /// `expected_manifest_sha256`; the lock's and profile's `artifact_sha256`. `None` = the artifact's digest.
    pub manifest_pin: Option<String>,
    /// What the stub sidecar reports; the lock pins the shipped values.
    pub tokenizer_sha256: String,
    pub template_sha256: String,
    /// The stub exits after this many replies.
    pub sidecar_exit_after: Option<u32>,
    /// The stub's `info.features` (comma-separated); `None` = the stub's default (`thinking`).
    pub sidecar_features: Option<&'static str>,
    pub cuda_visible_devices: &'static str,
    /// `CERE_NATIVE_DEVICE_ID`: the toy with its options (`toy,hold_after=N`).
    pub device_id: String,
    /// The artifact's `tokenizer.json` (the manifest re-hashed); `None` = the fixture's.
    pub tokenizer: Option<Vec<u8>>,
    /// M2.11: serve on the dev `<profile id>-snapshot` profile with a snapshot arena (D5, D8).
    pub snapshot: Option<SnapshotProfile>,
    /// Merged into the shipped native profile (`capabilities.json`), e.g. M2.12 `token_controls`; a `null` value removes the key.
    pub capabilities_patch: Value,
    /// `api.stream_buffer_events`; `None` = the shipped config's.
    pub stream_buffer_events: Option<i64>,
    /// `api.max_output_tokens`; `None` = the shipped config's.
    pub max_output_tokens: Option<u64>,
    /// `deployments.native.kv_pool_tokens`; `None` = the shipped config's.
    pub kv_pool_tokens: Option<u64>,
    /// M2.14 compat capabilities set on the deployment (`api_compat_anthropic`, ...); none by default (N6).
    pub compat: &'static [&'static str],
    /// Extra stub sidecar arguments (`--xgrammar V`, `--no-grammar-feature`); both roles receive them.
    pub sidecar_args: Vec<String>,
    /// M2.13 image switches; all off by default.
    pub images: Images,
    /// `deployments.native.prefix_state_slots` (M2.10 D6); `None` = the shipped config's (absent: 0, the cache off, so
    /// a repeated prompt prefills in full). The C5 cases turn it on with native-g0-prefix.toml's `cache_salt` (D9C).
    pub prefix_state_slots: Option<u64>,
    /// `deployments.native.prefix_alt_edges` (m210 design note D17); needs `prefix_state_slots`.
    pub prefix_alt_edges: bool,
    /// `deployments.native.decode_rounds_per_prefill_chunk` (O79, m210 design note PF1); `None` = absent.
    pub decode_rounds_per_prefill_chunk: Option<u64>,
    /// `deployments.native.prefill_slice_ms` (MixedPD, mixed-pd design note); `None` = absent.
    pub prefill_slice_ms: Option<u64>,
    /// `deployments.native.decode_share_pct` (S1 G4, beat-sglang design note); `None` = absent.
    pub decode_share_pct: Option<u64>,
    /// `deployments.native.prefill_age_calls` (BC4 (b), batch-concurrency design note); `None` = absent.
    pub prefill_age_calls: Option<u64>,
    /// m29 design note C8 (D11/D12): serve with MTP speculation, the `-v19` profile's speculation fields
    /// (`operations` with `verify`, `native_speculative_batch`, the verify combinations) on the shipped profile and
    /// `deployments.native.speculative_draft_tokens` 4.
    pub speculation: bool,
    /// m23 design note PI-02 (the drafter of m29l design note C5L D6): `[deployments.native.levers]`, lever
    /// name -> impl id; empty = absent (every lever at its registry default).
    pub levers: &'static [(&'static str, &'static str)],
    /// PI-04: `deployments.native.lever_hot_swap` (the dev hot swap route).
    pub lever_hot_swap: bool,
    /// ab-split design note AB-01: `[deployments.native.ab]` as JSON; `Null` = absent (every shipped profile).
    pub ab: Value,
    /// loop-guard design note: `[deployments.native.loop_guard]` as JSON; `Null` = absent (every shipped profile).
    pub loop_guard: Value,
    /// m32n design note D2: `[[deployments.native.replicas]]` `gpu0` ([`DEVICE_UUID`]) and `gpu1`
    /// ([`SECOND_UUID`]) instead of `device_uuid`; `cuda_visible_devices` must then be [`BOTH_UUIDS`]. A
    /// `device_id` of `a;b` gives replica 0 `a` and replica 1 `b`.
    pub replicas: bool,
    /// O47 / m32n design note D21: `deployments.native.host_share_dir` = `<root>/shm` (created): the toy puts the
    /// HOST_PINNED tensors in the deployment's one shared segment, which the replicas of [`Case::replicas`] share.
    pub host_share: bool,
    /// `deployments.native.sidecar_deadline_ms` (audit N1, ops-hardening design note OH-01); `None` = absent.
    pub sidecar_deadline_ms: Option<u64>,
    /// Set in `[ledger]` (e.g. `max_records`, `evict_terminal_when_full`); empty = the shipped config's.
    pub ledger: Vec<(&'static str, toml::Value)>,
    /// Track LoRA: `[deployments.native.lora]` as JSON; `Null` = absent (every shipped profile but `-v50`).
    pub lora: Value,
    /// WeightLab: `[deployments.native.research]` as JSON; `Null` = absent (every shipped profile but `-v57`).
    pub research: Value,
}

/// The dev snapshot profile of a case (M2.11 D8): an arena for `parked` parked sessions at the served context,
/// `[state] host_spill_enabled`, and `preempt` with its admission headroom.
#[derive(Clone, Copy)]
pub struct SnapshotProfile {
    pub preempt: bool,
    pub preempt_min_tokens: u64,
    /// `admission.max_backend_inflight` and `per_scope_inflight` (rows + parked with `preempt`).
    pub inflight: u64,
    /// `preempt_every` (D13 dev knob; 0 = absent).
    pub preempt_every: u32,
    /// The decode rows a preempting case serves: 1 = the dev/live `-snapshot` profile's one row (decode_batch_max 1),
    /// more = the shipped profile's rows, its capabilities' `decode_batch_max` (the one rows cap, BC-D5; `inflight`
    /// bounds how many run; sessions design note SE-09).
    pub rows: u64,
    /// `max_parked_sessions` with `preempt`, and the arena's snapshots.
    pub parked: u64,
    /// SE-09 `reserve_output_tokens` (0 = absent).
    pub reserve: u64,
}

impl Case {
    /// Every identity matches: the toy library pinned by its hash, the sidecar reporting the lock's hashes.
    pub fn matching() -> Case {
        let lock = shipped_lock();
        let library = toy_library();
        Case {
            library_pin: file_sha256(&library),
            library,
            manifest_pin: None,
            tokenizer_sha256: lock["tokenizer_sha256"].as_str().unwrap().to_owned(),
            template_sha256: lock["template_sha256"].as_str().unwrap().to_owned(),
            sidecar_exit_after: None,
            sidecar_features: None,
            cuda_visible_devices: DEVICE_UUID,
            device_id: "toy".to_owned(),
            tokenizer: None,
            snapshot: None,
            capabilities_patch: Value::Null,
            stream_buffer_events: None,
            max_output_tokens: None,
            kv_pool_tokens: None,
            compat: &[],
            sidecar_args: Vec::new(),
            images: Images::default(),
            prefix_state_slots: None,
            prefix_alt_edges: false,
            decode_rounds_per_prefill_chunk: None,
            prefill_slice_ms: None,
            decode_share_pct: None,
            prefill_age_calls: None,
            speculation: false,
            levers: &[],
            lever_hot_swap: false,
            ab: Value::Null,
            loop_guard: Value::Null,
            replicas: false,
            host_share: false,
            sidecar_deadline_ms: None,
            ledger: Vec::new(),
            lora: Value::Null,
            research: Value::Null,
        }
    }

    /// m32n design note D2: [`Case::matching`] served by two toy replicas (`gpu0`, `gpu1`) in one `cered`.
    pub fn replicas() -> Case {
        Case {
            replicas: true,
            cuda_visible_devices: BOTH_UUIDS,
            ..Case::matching()
        }
    }
}

/// A scratch repository (PACKAGE_VERSION, the native deployment re-pinned to the case, the fixture artifact)
/// and a running `cered` on it.
pub struct Native {
    pub cered: Cered,
    /// The prepared artifact `cered` loads.
    pub artifact: PathBuf,
    pub state: StateDir,
    /// The stub sidecar's `--log`: its startup line, then one op name per request.
    pub sidecar_log_path: PathBuf,
    /// The config file `cered` started from (and re-reads on `POST /v1/cere/config/reload`).
    pub config_path: PathBuf,
    cuda_visible_devices: &'static str,
    _root: Tmp,
}

/// `binary --config config` with the native cases' tokens, card and toy `device_id`.
async fn spawn_cered(
    binary: &Path,
    config: &Path,
    cuda_visible_devices: &str,
    device_id: &str,
) -> Cered {
    Cered::spawn_binary(
        binary,
        config,
        &[
            ("CERE_TOKEN_NATIVE_A", Some(TOKEN)),
            ("CERE_TOKEN_NATIVE_B", Some("secret-b")),
            ("CERE_OPERATOR_TOKEN", Some(OPERATOR_TOKEN)),
            ("CUDA_VISIBLE_DEVICES", Some(cuda_visible_devices)),
            // WeightLab WL-18: no host GPU1 production marker reaches a research config's isolation check.
            (
                "CERE_GPU1_PRODUCTION_MARKER",
                Some("/nonexistent/cere-test-absent-gpu1.production"),
            ),
            (NATIVE_DEVICE_ID_ENV, Some(device_id)),
        ],
    )
    .await
    .unwrap_or_else(|(code, stderr)| panic!("cered exited {code:?}: {stderr}"))
}

impl Native {
    pub async fn start(case: Case) -> Native {
        let repo = support::repo_root();
        let root = Tmp::new("native-cered");
        std::fs::copy(repo.join("PACKAGE_VERSION"), root.join("PACKAGE_VERSION")).unwrap();
        let (artifact, manifest) = match &case.tokenizer {
            Some(tokenizer) => verified_artifact_with_tokenizer(&root, tokenizer),
            None if case.images.artifact => vision_artifact(&root),
            None => verified_artifact(&root),
        };
        let manifest_pin = case.manifest_pin.unwrap_or(manifest);
        std::fs::create_dir_all(root.join("configs/deployments")).unwrap();
        std::fs::create_dir_all(root.join("configs/runtime")).unwrap();
        let mut lock = shipped_lock();
        let mut caps: Value = serde_json::from_slice(
            &std::fs::read(repo.join(format!("{DEPLOYMENT}.capabilities.json"))).unwrap(),
        )
        .unwrap();
        let mut recipe = std::fs::read(repo.join(format!("{DEPLOYMENT}.recipe.json"))).unwrap();
        let profile_id = format!("{}-snapshot", caps["profile_id"].as_str().unwrap());
        if case.snapshot.is_some() {
            caps["profile_id"] = Value::from(profile_id.as_str());
            caps["native_state_snapshot"] = Value::from(true);
            caps["operations"] =
                serde_json::json!(["prefill", "decode", "snapshot", "restore", "cancel"]);
            // As the dev/live `-snapshot` profile: a one-row preempting case serves one row of the shipped four
            // (U1 at rows > 1 is SE-09's, sessions design note).
            if case
                .snapshot
                .as_ref()
                .is_some_and(|s| s.preempt && s.rows == 1)
            {
                caps["decode_batch_max"] = Value::from(1);
                caps["native_decode_batch"] = Value::from(false);
                for combination in caps["validated_combinations"].as_array_mut().unwrap() {
                    combination["row_counts"] = serde_json::json!([1]);
                }
            }
            let mut patched: Value = serde_json::from_slice(&recipe).unwrap();
            patched["backend_profile_id"] = Value::from(profile_id.as_str());
            recipe = serde_json::to_vec_pretty(&patched).unwrap();
            lock["backend_profile_id"] = Value::from(profile_id.as_str());
        }
        std::fs::write(root.join(&format!("{DEPLOYMENT}.recipe.json")), &recipe).unwrap();
        lock["recipe_sha256"] = Value::from(sha256_hex(&recipe));
        lock["backend_build_sha256"] = Value::from(case.library_pin.as_str());
        lock["artifact_sha256"] = Value::from(manifest_pin.as_str());
        std::fs::write(
            root.join(&format!("{DEPLOYMENT}.lock.json")),
            serde_json::to_vec_pretty(&lock).unwrap(),
        )
        .unwrap();
        caps["artifact_sha256"] = Value::from(manifest_pin.as_str());
        if case.images.combo {
            add_image_combo(&mut caps);
        }
        if case.speculation {
            let spec: Value = serde_json::from_slice(
                &std::fs::read(repo.join(format!("{SPEC_DEPLOYMENT}.capabilities.json"))).unwrap(),
            )
            .unwrap();
            for key in [
                "operations",
                "native_speculative_batch",
                "validated_combinations",
            ] {
                caps[key] = spec[key].clone();
            }
            // A snapshot case keeps its operations (the `-v45` shape: speculation with U1) and its one-row
            // combinations.
            if let Some(s) = case.snapshot {
                let ops = caps["operations"].as_array_mut().unwrap();
                let at = ops.len() - 1;
                ops.splice(at..at, [Value::from("snapshot"), Value::from("restore")]);
                if s.preempt && s.rows == 1 {
                    for combination in caps["validated_combinations"].as_array_mut().unwrap() {
                        combination["row_counts"] = serde_json::json!([1]);
                    }
                }
            }
        }
        if let Some(patch) = case.capabilities_patch.as_object() {
            for (key, value) in patch {
                if value.is_null() {
                    caps.as_object_mut().unwrap().remove(key);
                } else {
                    caps[key] = value.clone();
                }
            }
        }
        std::fs::write(
            root.join(&format!("{DEPLOYMENT}.capabilities.json")),
            serde_json::to_vec_pretty(&caps).unwrap(),
        )
        .unwrap();

        let state = StateDir::new();
        let text = std::fs::read_to_string(repo.join("configs/runtime/native-g0.toml")).unwrap();
        let mut config: toml::Table = toml::from_str(&text).unwrap();
        config["server"]["bind"] = format!("127.0.0.1:{}", support::free_port()).into();
        config["ledger"]["state_dir"] = state.path().display().to_string().into();
        let ledger = config["ledger"].as_table_mut().unwrap();
        for (key, value) in &case.ledger {
            ledger.insert((*key).into(), value.clone());
        }
        if let Some(events) = case.stream_buffer_events {
            config["api"]["stream_buffer_events"] = events.into();
        }
        if let Some(tokens) = case.max_output_tokens {
            config["api"]["max_output_tokens"] = (tokens as i64).into();
        }
        let mut sidecar = vec![
            "python3".to_owned(),
            native_support::root()
                .join("crates/cere-runtime/tests/native_support/stub_sidecar.py")
                .display()
                .to_string(),
            "--tokenizer-sha256".to_owned(),
            case.tokenizer_sha256,
            "--template-sha256".to_owned(),
            case.template_sha256,
            // The grammar role's matchers as the toy fixture model's (spec m28 C7 startup check): its vocabulary
            // (70,000 ids, 2,188 mask words; the vision artifact's config is A4's, 248,320) and
            // `generation_config.json` EOS. A case's `sidecar_args` override.
            "--vocab-size".to_owned(),
            if case.images.artifact {
                "248320".to_owned()
            } else {
                FIXTURE_VOCAB.to_string()
            },
            "--stop-token-ids".to_owned(),
            "7:6".to_owned(),
        ];
        let sidecar_log = root.join("sidecar.log");
        sidecar.extend(["--log".to_owned(), sidecar_log.display().to_string()]);
        if let Some(n) = case.sidecar_exit_after {
            sidecar.extend(["--exit-after".to_owned(), n.to_string()]);
        }
        if case.images.sidecar {
            let preprocessor = file_sha256(&artifact.join("preprocessor_config.json"));
            sidecar.extend([
                "--feature".to_owned(),
                "image".to_owned(),
                "--preprocessor-sha256".to_owned(),
                preprocessor,
            ]);
        }
        if case.images.bad_offset {
            sidecar.push("--bad-image-offset".to_owned());
        }
        if let Some(features) = case.sidecar_features {
            sidecar.extend(["--features".to_owned(), features.to_owned()]);
        }
        sidecar.extend(case.sidecar_args);
        let deployment = config["deployments"][0].as_table_mut().unwrap();
        for key in case.compat {
            deployment.insert((*key).into(), true.into());
        }
        // A cache-on case serves native-g0-prefix.toml's isolation (D9C); the shipped native-g0 runs the cache off.
        if case.prefix_state_slots.is_some_and(|slots| slots > 0) {
            deployment.insert("prefix_cache_isolation".into(), "cache_salt".into());
        }
        let native = config["deployments"][0]["native"].as_table_mut().unwrap();
        native.insert("library".into(), case.library.display().to_string().into());
        native.insert("library_sha256".into(), case.library_pin.into());
        native.insert("artifact_dir".into(), artifact.display().to_string().into());
        native.insert("expected_manifest_sha256".into(), manifest_pin.into());
        if let Some(pool) = case.kv_pool_tokens {
            native.insert("kv_pool_tokens".into(), (pool as i64).into());
        }
        if let Some(slots) = case.prefix_state_slots {
            native.insert("prefix_state_slots".into(), (slots as i64).into());
        }
        if case.prefix_alt_edges {
            native.insert("prefix_alt_edges".into(), true.into());
        }
        if let Some(deadline) = case.sidecar_deadline_ms {
            native.insert("sidecar_deadline_ms".into(), (deadline as i64).into());
        }
        if let Some(rounds) = case.decode_rounds_per_prefill_chunk {
            native.insert(
                "decode_rounds_per_prefill_chunk".into(),
                (rounds as i64).into(),
            );
        }
        if let Some(slice_ms) = case.prefill_slice_ms {
            native.insert("prefill_slice_ms".into(), (slice_ms as i64).into());
        }
        if let Some(share_pct) = case.decode_share_pct {
            native.insert("decode_share_pct".into(), (share_pct as i64).into());
        }
        if let Some(calls) = case.prefill_age_calls {
            native.insert("prefill_age_calls".into(), (calls as i64).into());
        }
        if case.speculation {
            native.insert("speculative_draft_tokens".into(), 4.into());
        }
        if !case.levers.is_empty() {
            let levers: toml::Table = case
                .levers
                .iter()
                .map(|(name, id)| ((*name).to_owned(), toml::Value::from(*id)))
                .collect();
            native.insert("levers".into(), levers.into());
        }
        if case.lever_hot_swap {
            native.insert("lever_hot_swap".into(), true.into());
        }
        if !case.ab.is_null() {
            native.insert("ab".into(), toml::Value::try_from(&case.ab).unwrap());
        }
        if !case.lora.is_null() {
            native.insert("lora".into(), toml::Value::try_from(&case.lora).unwrap());
        }
        if !case.research.is_null() {
            native.insert(
                "research".into(),
                toml::Value::try_from(&case.research).unwrap(),
            );
        }
        if !case.loop_guard.is_null() {
            native.insert(
                "loop_guard".into(),
                toml::Value::try_from(&case.loop_guard).unwrap(),
            );
        }
        native.insert(
            "sidecar".into(),
            toml::Value::Array(sidecar.into_iter().map(toml::Value::from).collect()),
        );
        if let Some(s) = case.snapshot {
            let context = native["max_context_tokens"].as_integer().unwrap() as u64;
            let arena = arena_slot_bytes(context, case.speculation).unwrap() * s.parked.max(1);
            native.insert("snapshot_host_bytes".into(), (arena as i64).into());
            if s.preempt {
                native.insert("preempt".into(), true.into());
                native.insert("max_parked_sessions".into(), (s.parked as i64).into());
                if s.preempt_every > 0 {
                    native.insert("preempt_every".into(), (s.preempt_every as i64).into());
                }
                if s.reserve > 0 {
                    native.insert("reserve_output_tokens".into(), (s.reserve as i64).into());
                }
                native.insert(
                    "preempt_min_tokens".into(),
                    (s.preempt_min_tokens as i64).into(),
                );
            }
            config["deployments"][0]["profile_id"] = profile_id.into();
            config["state"]["host_spill_enabled"] = true.into();
            config["admission"]["max_backend_inflight"] = (s.inflight as i64).into();
            config["admission"]["per_scope_inflight"] = (s.inflight as i64).into();
        }
        if case.host_share {
            let dir = root.join("shm");
            std::fs::create_dir_all(&dir).unwrap();
            let native = config["deployments"][0]["native"].as_table_mut().unwrap();
            native.insert("host_share_dir".into(), dir.display().to_string().into());
        }
        // m32n design note D2/D3: two toy replicas, `gpu0` and `gpu1`, both cards visible in replica order.
        if case.replicas {
            let native = config["deployments"][0]["native"].as_table_mut().unwrap();
            native.remove("device_uuid");
            let replicas = [(WORKER, DEVICE_UUID), (SECOND_WORKER, SECOND_UUID)]
                .into_iter()
                .map(|(device, uuid)| {
                    let mut replica = toml::Table::new();
                    replica.insert("device".into(), device.into());
                    replica.insert("device_uuid".into(), uuid.into());
                    toml::Value::Table(replica)
                })
                .collect();
            native.insert("replicas".into(), toml::Value::Array(replicas));
            config["scheduler"]["devices"] =
                toml::Value::Array(vec![WORKER.into(), SECOND_WORKER.into()]);
        }
        let path = root.join("configs/runtime/native.toml");
        std::fs::write(&path, toml::to_string(&config).unwrap()).unwrap();

        let cered = spawn_cered(
            Path::new(env!("CARGO_BIN_EXE_cered")),
            &path,
            case.cuda_visible_devices,
            &case.device_id,
        )
        .await;
        Native {
            cered,
            artifact,
            state,
            sidecar_log_path: sidecar_log,
            config_path: path,
            cuda_visible_devices: case.cuda_visible_devices,
            _root: root,
        }
    }

    /// Serves the scratch repository with `binary` (e.g. an earlier release) from `config` and the toy `device_id`,
    /// in place of [`Native::cered`], which must have exited (the state directory's lock).
    pub async fn respawn(&mut self, binary: &Path, config: &Path, device_id: &str) {
        self.cered = spawn_cered(binary, config, self.cuda_visible_devices, device_id).await;
    }

    /// The stub sidecar's log lines so far (startup line, then op names).
    pub fn sidecar_log(&self) -> Vec<String> {
        std::fs::read_to_string(&self.sidecar_log_path)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// `/readyz` once the native load of every worker (one, or both replicas of [`Case::replicas`]) reached its
    /// final outcome (the first probe owns the load).
    pub async fn settled(&self) -> (u16, Value) {
        let http = http();
        for _ in 0..1500 {
            let response = http.get(self.cered.url("/readyz")).send().await.unwrap();
            let status = response.status().as_u16();
            let body: Value = response.json().await.unwrap();
            let mut workers = body["workers"]
                .as_object()
                .into_iter()
                .flat_map(|w| w.values());
            if !worker(&body)["probed_at_unix_ms"].is_null()
                && workers.all(|w| !w["probed_at_unix_ms"].is_null())
            {
                return (status, body);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the native load never finished");
    }

    /// Starts `case` and asserts the native load reached ready.
    pub async fn ready(case: Case) -> Native {
        let native = Native::start(case).await;
        let (status, body) = native.settled().await;
        assert_eq!(status, 200, "{body}");
        native
    }

    pub fn chat(&self, body: &Value) -> reqwest::RequestBuilder {
        http()
            .post(self.cered.url("/v1/chat/completions"))
            .bearer_auth(TOKEN)
            .json(body)
    }

    pub async fn get(&self, path: &str) -> (u16, Value) {
        let response = http()
            .get(self.cered.url(path))
            .bearer_auth(TOKEN)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap())
    }

    /// `levers` of `GET /v1/cere/stats` (m23 design note PI-05): every registry lever's active impl.
    pub async fn stats_levers(&self) -> Value {
        let (status, body) = self.get("/v1/cere/stats").await;
        assert_eq!(status, 200, "{body}");
        body["levers"].clone()
    }

    /// `GET /v1/cere/levers` with the operator token (PI-04).
    pub async fn levers(&self) -> (u16, Value) {
        let response = http()
            .get(self.cered.url("/v1/cere/levers"))
            .bearer_auth(OPERATOR_TOKEN)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap())
    }

    /// `POST /v1/cere/levers` with `body` and the bearer `token` (PI-04).
    pub async fn set_levers(&self, token: &str, body: &Value) -> (u16, Value) {
        let response = http()
            .post(self.cered.url("/v1/cere/levers"))
            .bearer_auth(token)
            .json(body)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap())
    }

    /// AB-05: rewrites the config's `[deployments.native.ab]` (`Null` removes it) and posts
    /// `POST /v1/cere/config/reload` with the operator token.
    pub async fn reload_ab(&self, ab: &Value) -> (u16, Value) {
        let text = std::fs::read_to_string(&self.config_path).unwrap();
        let mut config: toml::Table = toml::from_str(&text).unwrap();
        let native = config["deployments"].as_array_mut().unwrap()[0]["native"]
            .as_table_mut()
            .unwrap();
        if ab.is_null() {
            native.remove("ab");
        } else {
            native.insert("ab".into(), toml::Value::try_from(ab).unwrap());
        }
        std::fs::write(&self.config_path, toml::to_string(&config).unwrap()).unwrap();
        let response = http()
            .post(self.cered.url("/v1/cere/config/reload"))
            .bearer_auth(OPERATOR_TOKEN)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap())
    }

    /// `ab` of `GET /v1/cere/stats` (AB-04): per served model, per arm; `Null` without arms.
    pub async fn stats_ab(&self) -> Value {
        let (status, body) = self.get("/v1/cere/stats").await;
        assert_eq!(status, 200, "{body}");
        body["ab"].clone()
    }

    /// `native_preemptions` of `GET /v1/cere/stats` (M2.11 D8).
    pub async fn preemptions(&self) -> u64 {
        let (status, body) = self.get("/v1/cere/stats").await;
        assert_eq!(status, 200, "{body}");
        body["native_preemptions"]
            .as_u64()
            .unwrap_or_else(|| panic!("native_preemptions is not an integer: {body}"))
    }

    /// `native_grows` of `GET /v1/cere/stats` (sessions design note SE-09).
    pub async fn grows(&self) -> u64 {
        let (status, body) = self.get("/v1/cere/stats").await;
        assert_eq!(status, 200, "{body}");
        body["native_grows"]
            .as_u64()
            .unwrap_or_else(|| panic!("native_grows is not an integer: {body}"))
    }

    /// `native_submissions` of `GET /v1/cere/stats`.
    pub async fn submissions(&self) -> u64 {
        let (status, body) = self.get("/v1/cere/stats").await;
        assert_eq!(status, 200, "{body}");
        body["native_submissions"]
            .as_u64()
            .unwrap_or_else(|| panic!("native_submissions is not an integer: {body}"))
    }

    /// `(native_spec_verify_steps, native_spec_accepted_drafts)` of `GET /v1/cere/stats` (m29 design note D11).
    pub async fn spec_counters(&self) -> (u64, u64) {
        let (status, body) = self.get("/v1/cere/stats").await;
        assert_eq!(status, 200, "{body}");
        let counter = |key: &str| {
            body[key]
                .as_u64()
                .unwrap_or_else(|| panic!("{key} is not an integer: {body}"))
        };
        (
            counter("native_spec_verify_steps"),
            counter("native_spec_accepted_drafts"),
        )
    }

    /// Polls the request status until it is terminal.
    pub async fn terminal(&self, rid: &str) -> Value {
        for _ in 0..400 {
            let (status, body) = self.get(&format!("/v1/cere/requests/{rid}")).await;
            assert_eq!(status, 200, "{body}");
            if ["completed", "failed", "cancelled"].contains(&body["state"].as_str().unwrap()) {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("request {rid} did not reach a terminal state");
    }

    /// Polls `GET /v1/cere/stats` until no worker holds an admission slot of this scope (`workers.*.inflight` 0),
    /// for at most 30 s. A terminal state is visible before admission releases the request's slot (O368), so a
    /// step that needs an idle server (a lever hot swap: 409 `requests_in_flight`) waits here, not on `terminal`.
    pub async fn idle(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let (status, body) = self.get("/v1/cere/stats").await;
            assert_eq!(status, 200, "{body}");
            let busy = body["workers"]
                .as_object()
                .unwrap_or_else(|| panic!("stats without workers: {body}"))
                .values()
                .any(|w| w["inflight"].as_u64() != Some(0));
            if !busy {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "requests stayed in flight: {body}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
