//! R2 (spec m22 Q13): the upload sink stages the M2.1 fixture artifact (prepared from `tests/fixtures/model_src`)
//! into the CPU toy executor (`$CERE_TOY_EXECUTOR`): byte counters equal the fixture inventory with 1 and 4 staging
//! buffers, a changed config value is refused as UNSUPPORTED at ALLOCATE (before any put), a sink error aborts
//! staging with nothing sealed, and MTP-draft tensors reach the executor only as DISCARD.
//!
//! M2.5 (m25 design note D4, C3): `PlanOptions.kv_pool_tokens` is the desc's pool; the toy's capabilities report the
//! pool it allocated, and an invalid pool is refused at ALLOCATE with its typed status.
//!
//! M216S-11 (m216 design note): the pass registers its staging buffers as put staging when the library has the two
//! exports (DEVICE puts take the direct path), and a library from before them (`$CERE_TOY_EXECUTOR_PRESTAGING`) loads
//! on the bounce ring.
mod native_support;

use cere_compiler::artifact::{self, Artifact, Mode, ReadMode, MIN_BUFFER_BYTES};
use cere_compiler::families::Role;
use cere_compiler::safetensors::Dtype;
use cere_runtime::native::driver::{
    allowed_cpus, AllocateThread, Driver, DriverError, ExecutorConfig, ModelConfig, ModelSpec,
    WorkRequest,
};
use cere_runtime::native::ffi::{self, Library, StatusError};
use cere_runtime::native::sampling::GenerationDefaults;
use cere_runtime::native::upload::{
    inkling_vision_config, model_config, model_spec, placement, ple_table_format, qwen4_exp_config,
    s3_quantized_tensors, stage_all, stage_all_to, staging_block, upload, PlanOptions,
    PleTableFormat, Target, UploadError, UploadOptions, BACKGROUND_ALLOCATE_CPUS,
    BACKGROUND_COPIES_IN_FLIGHT, BACKGROUND_NICE, PLE_NVFP4_PROFILE_ID, S3_MXFP8_PROFILE_ID,
    S3_MXFP8_VISION_PROFILE_ID,
};
use native_support::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PLAN: PlanOptions = PlanOptions {
    max_sessions: 1,
    max_context_tokens: 4096,
    snapshot_host_bytes: 0,
    kv_pool_tokens: 0,
    prefix_state_slots: 0,
    speculative_draft_tokens: 0,
};

/// m29 design note D11: the speculation-on plan (4 MTP draft tokens).
const PLAN_DRAFTS: PlanOptions = PlanOptions {
    speculative_draft_tokens: 4,
    ..PLAN
};

fn toy_driver() -> Driver {
    let lib = Arc::new(Library::open(&toy_library(), None).unwrap());
    Driver::start(
        lib,
        &ExecutorConfig {
            device_id: "toy".to_string(),
            worker_epoch: 1,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap()
}

/// The stamped copy of the fixture artifact, opened as the native path opens it (`Mode::Stamp`).
fn opened(t: &Tmp) -> (PathBuf, Artifact) {
    let (dir, sha) = verified_artifact(t);
    let art = artifact::open(&dir, &sha, Mode::Stamp).unwrap();
    (dir, art)
}

fn options(buffers: usize) -> UploadOptions {
    read_options(buffers, ReadMode::Buffered)
}

/// `buffers` minimum-size staging buffers, the engine's default readers and put threads, and `read`.
fn read_options(buffers: usize, read: ReadMode) -> UploadOptions {
    UploadOptions {
        buffers,
        buffer_bytes: MIN_BUFFER_BYTES,
        readers: 0,
        consumers: 0,
        read,
        background: false,
    }
}

/// `(count, bytes)` of the fixture inventory's MTP draft families.
fn inventory_draft() -> (u64, u64) {
    let raw = std::fs::read(root().join("tests/fixtures/model_src.inventory.json")).unwrap();
    let inv: Value = serde_json::from_slice(&raw).unwrap();
    let mut out = (0, 0);
    for f in inv["families"].as_array().unwrap() {
        if f["role"] == "draft" {
            out.0 += f["count"].as_u64().unwrap();
            out.1 += f["bytes"].as_u64().unwrap();
        }
    }
    out
}

#[test]
fn model_spec_reads_the_fixture_config_tensor_table_and_stop_tokens() {
    let t = Tmp::new("spec");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let c = spec
        .config
        .qwen4_exp()
        .expect("the fixture is family qwen4_exp");
    assert_eq!(
        c.struct_size as usize,
        std::mem::size_of::<ffi::Qwen4ExpConfig>()
    );
    assert_eq!(c.abi_version, ffi::ABI_VERSION);
    assert_eq!(c.vocab_size, FIXTURE_VOCAB);
    assert_eq!(c.hidden_size, 32);
    assert_eq!(c.num_hidden_layers, 1);
    assert_eq!(c.full_attention_interval, 4);
    assert_eq!(c.num_experts, 2);
    assert_eq!(c.num_experts_per_tok, 1);
    assert_eq!(c.moe_intermediate_size, 16);
    assert_eq!(c.norm_topk_prob, 1, "absent norm_topk_prob is 1");
    assert_eq!(c.ple_layer_ids, 1 << 2);
    assert_eq!(c.ngram_size, 3);
    assert_eq!(c.split_ngram_parts, 4);
    assert_eq!(c.mtp_num_hidden_layers, 1);
    assert_eq!(c.mamba_ssm_dtype, ffi::DTYPE_F32);
    assert_eq!(c.num_attention_heads, 0, "absent keys are 0");
    assert_eq!(c.rope_theta, 0.0);
    assert_eq!(spec.stop_token_ids, vec![7, 6]);
    // C8 (m27 D10): the sampling defaults of `generation_config.json`, filled into a native request's absent fields.
    assert_eq!(
        spec.generation_defaults,
        GenerationDefaults {
            temperature: Some(1.0),
            top_k: Some(20),
            top_p: Some(0.95),
        }
    );
    // Every tensor but the `unused` ones, in the artifact's order.
    let staged: Vec<_> = art
        .tensors()
        .filter(|t| t.placement != "unused")
        .map(|t| t.name.to_string())
        .collect();
    let names: Vec<_> = spec.tensors.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, staged);
    let inv = inventory_totals();
    let sum = |p: u32| -> u64 {
        spec.tensors
            .iter()
            .filter(|t| t.placement == p)
            .map(|t| t.bytes)
            .sum()
    };
    assert_eq!(sum(ffi::PLACEMENT_DEVICE), inv.device);
    assert_eq!(sum(ffi::PLACEMENT_HOST_PINNED), inv.host);
    assert_eq!(sum(ffi::PLACEMENT_DISCARD), inv.draft);
}

/// FL-13 (fp4local design note D8, D14): the K4 NVFP4 variant is reachable only from the FP4-Local artifact
/// profile. Its U8 table selects NVFP4; the shipped profile (and any other, e.g. the fixture's) keeps the E4M3
/// table, refuses a U8 one, and the NVFP4 profile refuses an E4M3 table, a mix or another dtype (class `ple_table`).
#[test]
fn the_nvfp4_ple_table_is_admitted_only_under_its_artifact_profile() {
    const CUR: &str = "rtxpro6000bw-plehost-text-v1";
    assert_eq!(PLE_NVFP4_PROFILE_ID, "rtxpro6000bw-plehost-nvfp4-text-v1");
    let shards = |d: Dtype| vec![d; 128];
    assert_eq!(
        ple_table_format(PLE_NVFP4_PROFILE_ID, shards(Dtype::U8)).unwrap(),
        PleTableFormat::Nvfp4
    );
    for profile in [
        CUR,
        "fixture-plehost-text-v1",
        "rtxpro6000bw-plehost-vision-v1",
    ] {
        assert_eq!(
            ple_table_format(profile, shards(Dtype::F8E4m3)).unwrap(),
            PleTableFormat::Fp8
        );
        let err = ple_table_format(profile, shards(Dtype::U8)).unwrap_err();
        assert_eq!(err.class(), "ple_table", "{profile}: {err}");
    }
    let mut mixed = shards(Dtype::U8);
    mixed[127] = Dtype::F8E4m3;
    for (profile, dtypes) in [
        (PLE_NVFP4_PROFILE_ID, shards(Dtype::F8E4m3)),
        (PLE_NVFP4_PROFILE_ID, mixed.clone()),
        (CUR, mixed),
        (PLE_NVFP4_PROFILE_ID, shards(Dtype::Bf16)),
    ] {
        let err = ple_table_format(profile, dtypes).unwrap_err();
        assert_eq!(err.class(), "ple_table", "{profile}: {err}");
    }
    // The fixture artifact (E4M3 table, its own profile id) still builds its desc.
    let t = Tmp::new("ple_format");
    let (dir, art) = opened(&t);
    assert_eq!(art.profile.profile_id, "fixture-plehost-text-v1");
    model_spec(&art, &dir, &PLAN).unwrap();
}

/// fp4local design note S3 (FL-30): the MXFP8 dense pairs and the fused NVFP4 MTP experts of `cerec convert --format
/// s3` are admitted only under the S3 artifact profile (classes `mxfp8` / `mtp_nvfp4` under any other); under it, a
/// non-BF16 dense/draft tensor of another form is still refused, the PLE table is NVFP4, and the MTP experts follow the
/// draft placement (DISCARD when the plan does not speculate), as under S2.
#[test]
fn s3_quantized_tensors_are_admitted_only_under_the_s3_profile() {
    assert_eq!(
        S3_MXFP8_PROFILE_ID,
        "rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1"
    );
    let l = "model.language_model.layers.3.";
    let e = "mtp.layers.0.mlp.experts.";
    let mxfp8 = [
        (
            format!("{l}self_attn.q_proj.weight"),
            Role::Dense,
            Dtype::F8E4m3,
        ),
        (
            format!("{l}self_attn.q_proj.weight_scale"),
            Role::Dense,
            Dtype::U8,
        ),
        (
            format!("{l}linear_attn.in_proj_qkv.weight"),
            Role::Dense,
            Dtype::F8E4m3,
        ),
        (
            format!("{l}mlp.shared_expert.down_proj.weight_scale"),
            Role::Dense,
            Dtype::U8,
        ),
    ];
    let mtp = [
        (format!("{e}gate_up_proj"), Role::Draft, Dtype::U8),
        (format!("{e}gate_up_proj_scale"), Role::Draft, Dtype::F8E4m3),
        (format!("{e}gate_up_proj_scale_2"), Role::Draft, Dtype::F32),
        (format!("{e}down_proj"), Role::Draft, Dtype::U8),
        (format!("{e}down_proj_scale"), Role::Draft, Dtype::F8E4m3),
        (format!("{e}down_proj_scale_2"), Role::Draft, Dtype::F32),
    ];
    let bf16 = [
        (format!("{l}linear_attn.A_log"), Role::Dense, Dtype::Bf16),
        ("mtp.fc_hidden.weight".to_string(), Role::Draft, Dtype::Bf16),
    ];
    let gate = |profile: &str, ts: &[(String, Role, Dtype)]| {
        s3_quantized_tensors(profile, ts.iter().map(|(n, r, d)| (n.as_str(), *r, *d)))
    };
    let all: Vec<_> = mxfp8.iter().chain(&mtp).chain(&bf16).cloned().collect();
    gate(S3_MXFP8_PROFILE_ID, &all).unwrap();
    assert_eq!(
        ple_table_format(S3_MXFP8_PROFILE_ID, vec![Dtype::U8; 128]).unwrap(),
        PleTableFormat::Nvfp4
    );
    assert_eq!(
        ple_table_format(S3_MXFP8_PROFILE_ID, vec![Dtype::F8E4m3; 128])
            .unwrap_err()
            .class(),
        "ple_table"
    );
    for profile in [
        PLE_NVFP4_PROFILE_ID,
        "rtxpro6000bw-plehost-text-v1",
        "rtxpro6000bw-plehost-vision-v1",
        "fixture-plehost-text-v1",
    ] {
        gate(profile, &bf16).unwrap();
        for (ts, class) in [(&mxfp8[..], "mxfp8"), (&mtp[..], "mtp_nvfp4")] {
            for t in ts {
                let err = gate(profile, std::slice::from_ref(t)).unwrap_err();
                assert_eq!(err.class(), class, "{profile} {}: {err}", t.0);
                assert!(err.to_string().contains(S3_MXFP8_PROFILE_ID), "{err}");
            }
        }
    }
    for (t, class) in [
        (
            (
                format!("{l}self_attn.q_proj.weight"),
                Role::Dense,
                Dtype::U8,
            ),
            "mxfp8",
        ),
        (
            (
                format!("{l}self_attn.q_proj.weight_scale"),
                Role::Dense,
                Dtype::F8E4m3,
            ),
            "mxfp8",
        ),
        (
            (
                format!("{l}self_attn.q_proj.weight"),
                Role::Dense,
                Dtype::F32,
            ),
            "mxfp8",
        ),
        (
            (format!("{e}gate_up_proj"), Role::Draft, Dtype::F16),
            "mtp_nvfp4",
        ),
        (
            (
                "mtp.fc_hidden.weight".to_string(),
                Role::Draft,
                Dtype::F8E4m3,
            ),
            "mtp_nvfp4",
        ),
    ] {
        let err = gate(S3_MXFP8_PROFILE_ID, std::slice::from_ref(&t)).unwrap_err();
        assert_eq!(err.class(), class, "{}: {err}", t.0);
    }
    for t in &mtp {
        assert_eq!(placement("device", t.1, 0), Some(ffi::PLACEMENT_DISCARD));
        assert_eq!(placement("device", t.1, 3), Some(ffi::PLACEMENT_DEVICE));
    }
    for t in &mxfp8 {
        assert_eq!(placement("device", t.1, 0), Some(ffi::PLACEMENT_DEVICE));
    }
}

/// m213l design note D-L1 (VL-L9): the S3 vision artifact profile (the S3 text profile plus the vision tower on the
/// device) admits exactly what the S3 profile admits: the NVFP4 PLE table, the MXFP8 dense pairs and the NVFP4 MTP
/// experts, still refusing an E4M3 table or another non-BF16 form.
#[test]
fn the_s3_vision_profile_admits_the_s3_tensor_forms() {
    assert_eq!(
        S3_MXFP8_VISION_PROFILE_ID,
        "rtxpro6000bw-plehost-nvfp4-mxfp8-vision-v1"
    );
    let p = S3_MXFP8_VISION_PROFILE_ID;
    assert_eq!(
        ple_table_format(p, vec![Dtype::U8; 128]).unwrap(),
        PleTableFormat::Nvfp4
    );
    let err = ple_table_format(p, vec![Dtype::F8E4m3; 128]).unwrap_err();
    assert_eq!(err.class(), "ple_table");
    assert!(err.to_string().contains(p), "{err}");
    let ts = [
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            Role::Dense,
            Dtype::F8E4m3,
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight_scale",
            Role::Dense,
            Dtype::U8,
        ),
        ("mtp.layers.0.mlp.experts.down_proj", Role::Draft, Dtype::U8),
        (
            "mtp.layers.0.mlp.experts.down_proj_scale_2",
            Role::Draft,
            Dtype::F32,
        ),
        (
            "model.visual.blocks.0.attn.qkv.weight",
            Role::Vision,
            Dtype::Bf16,
        ),
    ];
    s3_quantized_tensors(p, ts).unwrap();
    let bad = [(
        "model.language_model.layers.3.self_attn.q_proj.weight",
        Role::Dense,
        Dtype::F32,
    )];
    assert_eq!(s3_quantized_tensors(p, bad).unwrap_err().class(), "mxfp8");
    let err = s3_quantized_tensors("rtxpro6000bw-plehost-vision-v1", [ts[0]]).unwrap_err();
    assert!(err.to_string().contains(p), "{err}");
}

#[test]
fn counters_equal_the_inventory_totals_with_1_4_and_32_staging_buffers() {
    let t = Tmp::new("counters");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let inv = inventory_totals();
    for buffers in [1, 4, 32] {
        let d = toy_driver();
        let read_before = art.counters().payload_bytes_read;
        let up = upload(&d, &art, &spec, &options(buffers)).unwrap();
        let c = up.staging.counters;
        assert_eq!(c.uploaded_device_bytes, inv.device, "{buffers} buffers");
        assert_eq!(c.uploaded_host_bytes, inv.host, "{buffers} buffers");
        assert_eq!(c.discarded_bytes, inv.draft, "{buffers} buffers");
        // One `stage(All)` pass (m216 design note L2): every desc tensor but the skipped DISCARD ones, read in
        // fewer `pread`s than tensors.
        let p = up.staging.pass;
        assert!(p.threads >= 1 && p.threads <= buffers);
        assert!(p.high_water_bytes <= (buffers * MIN_BUFFER_BYTES) as u64);
        assert_eq!(p.bytes, inv.device + inv.host);
        let discard = spec
            .tensors
            .iter()
            .filter(|t| t.placement == ffi::PLACEMENT_DISCARD)
            .count() as u64;
        assert_eq!(p.tensors, spec.tensors.len() as u64 - discard);
        assert!(p.reads >= 1 && p.reads < p.tensors, "{p:?}");
        assert_eq!(
            art.counters().payload_bytes_read - read_before,
            inv.device + inv.host
        );
        assert_eq!(up.capabilities.max_context_tokens, PLAN.max_context_tokens);
        assert_eq!(up.capabilities.worker_epoch, 1);
        // The sealed model serves: one greedy toy step.
        let s = d.create_session(&up.model, 64).unwrap();
        let r = d
            .submit(&s, WorkRequest::prefill(vec![1, 2, 3], 1))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(r.tokens, toy_tokens(3, 3, 1, FIXTURE_VOCAB));
        d.release_session(s).unwrap();
        d.release_model(up.model).unwrap();
        d.shutdown().unwrap();
    }
}

/// The calling thread's nice value (Linux: a per-thread attribute).
fn own_nice() -> i32 {
    // SAFETY: reads the calling thread's own nice value.
    unsafe { libc::getpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t) }
}

/// m32n design note D22 (3): a background upload (a replica's load while the other card serves) stages through
/// `BACKGROUND_COPIES_IN_FLIGHT` staging buffers with at most as many readers and put threads, runs its threads at
/// nice `BACKGROUND_NICE`, runs its ALLOCATE on a thread at nice `BACKGROUND_NICE` confined to
/// `BACKGROUND_ALLOCATE_CPUS` CPUs, and leaves the caller's priority and CPUs alone; the same upload in the foreground
/// runs at the caller's (its ALLOCATE on the driver thread, which inherits them). Both stage every byte.
#[test]
fn a_background_upload_runs_at_the_lowest_priority_with_2_copies_in_flight() {
    let t = Tmp::new("background");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let inv = inventory_totals();
    let caller = own_nice();
    let cpus = allowed_cpus().unwrap().len();
    let background = UploadOptions {
        buffer_bytes: MIN_BUFFER_BYTES,
        ..UploadOptions::background()
    };
    assert_eq!(background.buffers, BACKGROUND_COPIES_IN_FLIGHT);
    assert_eq!(BACKGROUND_COPIES_IN_FLIGHT, 2);
    for (opts, nice, allocate) in [
        (
            UploadOptions {
                background: false,
                ..background
            },
            caller,
            AllocateThread { nice: caller, cpus },
        ),
        (
            background,
            BACKGROUND_NICE,
            AllocateThread {
                nice: BACKGROUND_NICE,
                cpus: cpus.min(BACKGROUND_ALLOCATE_CPUS),
            },
        ),
    ] {
        let d = toy_driver();
        let up = upload(&d, &art, &spec, &opts).unwrap();
        let s = &up.staging;
        assert_eq!(s.nice, nice, "{opts:?}");
        assert_eq!(up.allocate, allocate, "{opts:?}");
        assert!(s.pass.threads >= 1 && s.pass.threads <= 2, "{s:?}");
        assert!(s.pass.consumers >= 1 && s.pass.consumers <= 2, "{s:?}");
        assert!((1..=2).contains(&s.copies_in_flight_max), "{s:?}");
        assert_eq!(s.counters.uploaded_device_bytes, inv.device);
        assert_eq!(s.counters.uploaded_host_bytes, inv.host);
        assert_eq!(own_nice(), caller, "the caller's priority changed");
        assert_eq!(
            allowed_cpus().unwrap().len(),
            cpus,
            "the caller's CPUs changed"
        );
        d.release_model(up.model).unwrap();
        d.shutdown().unwrap();
    }
    assert_eq!(BACKGROUND_NICE, 19);
    assert_eq!(BACKGROUND_ALLOCATE_CPUS, 2);
}

/// The fixture `config.json` with `text_config.<key>` set to `value`.
fn changed_config(key: &str, value: Value) -> ModelConfig {
    let raw = std::fs::read(fixture_artifact().0.join("config.json")).unwrap();
    let mut config: Value = serde_json::from_slice(&raw).unwrap();
    config["text_config"][key] = value;
    ModelConfig::Qwen4Exp(qwen4_exp_config(&config).unwrap())
}

/// SelfTune ST4 S3b: a flat `config.json` of `model_type` `sdar` (DICE-8B) fills `cere_sdar_config`; the same flat
/// form under any other `model_type` is the Qwen form without its `text_config`, refused by name.
#[test]
fn the_model_type_selects_the_config_family() {
    let path = root().join("tests/fixtures/model_configs/dice-8b/config.json");
    let mut dice: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let config = model_config(&dice, None).unwrap();
    assert_eq!(config.family(), ffi::FAMILY_SDAR);
    assert_eq!(config.model_kind(), ffi::MODEL_SDAR_BF16);
    assert!(config.qwen4_exp().is_none());
    let ModelConfig::Sdar(c) = config else {
        panic!("{config:?}")
    };
    assert_eq!(
        c,
        ffi::SdarConfig {
            struct_size: std::mem::size_of::<ffi::SdarConfig>() as u32,
            abi_version: ffi::ABI_VERSION,
            vocab_size: 151_936,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            block_size: 4,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
            mask_token_id: 151_669,
            tie_word_embeddings: 0,
            max_position_embeddings: 32_768,
        }
    );
    assert_eq!(config.vocab_size(), 151_936);
    assert_eq!(config.bytes().len(), std::mem::size_of::<ffi::SdarConfig>());

    dice["block_size"] = Value::String("4".into());
    let err = model_config(&dice, None).unwrap_err();
    assert!(
        err.to_string().contains("config.block_size is not"),
        "{err}"
    );

    dice["model_type"] = Value::String("qwen4_exp".into());
    let err = model_config(&dice, None).unwrap_err();
    assert!(err.to_string().contains("no text_config"), "{err}");
    dice["model_type"] = 7.into();
    let err = model_config(&dice, None).unwrap_err();
    assert!(
        err.to_string().contains("model_type is not a string"),
        "{err}"
    );

    let raw = std::fs::read(fixture_artifact().0.join("config.json")).unwrap();
    let qwen = model_config(&serde_json::from_slice(&raw).unwrap(), None).unwrap();
    assert_eq!(
        (qwen.family(), qwen.model_kind()),
        (ffi::FAMILY_QWEN4_EXP, ffi::MODEL_QWEN4_EXP_NVFP4)
    );
}

/// Inkling 2 Mini (inkling design note S5): `model_type` `inkling2_loop` fills `cere_inkling2_config` from
/// `text_config`, its `loop_config`, the top-level `eos_token_id` and the expert formats of `hf_quant_config.json`
/// (MXFP8 on layers 20-23); without the quant config it is refused.
#[test]
fn an_inkling2_loop_config_reads_its_quant_config() {
    let dir = root().join("tests/fixtures/model_configs/inkling-2-mini");
    let read = |f: &str| -> Value {
        serde_json::from_slice(&std::fs::read(dir.join(f)).unwrap()).unwrap()
    };
    let (config, quant) = (read("config.json"), read("hf_quant_config.json"));
    let err = model_config(&config, None).unwrap_err();
    assert!(err.to_string().contains("hf_quant_config.json"), "{err}");
    let got = model_config(&config, Some(&quant)).unwrap();
    assert_eq!(got.family(), ffi::FAMILY_INKLING2);
    assert_eq!(got.model_kind(), ffi::MODEL_INKLING2_NVFP4);
    assert!(got.qwen4_exp().is_none());
    let ModelConfig::Inkling2(c) = got else {
        panic!("{got:?}")
    };
    let local = [
        0u32, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
    ];
    assert_eq!(
        c,
        ffi::Inkling2Config {
            struct_size: std::mem::size_of::<ffi::Inkling2Config>() as u32,
            abi_version: ffi::ABI_VERSION,
            vocab_size: 201_024,
            unpadded_vocab_size: 200_058,
            hidden_size: 2304,
            num_hidden_layers: 24,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            head_dim: 128,
            moe_intermediate_size: 1152,
            num_experts: 128,
            num_experts_per_tok: 8,
            sliding_window_size: 1024,
            rel_extent: 1024,
            d_rel: 16,
            loop_core_start: 6,
            loop_core_end: 18,
            loop_num_passes: 3,
            loop_gate_rank: 64,
            n_layers_per_attn_res_block: 6,
            conv_kernel_size: 4,
            eos_token_id: 200_006,
            local_layer_mask: local.iter().fold(0, |m, i| m | (1 << i)),
            mxfp8_layer_mask: 0xF0_0000,
            rms_norm_eps: 1e-6,
            final_logit_softcapping: 30.0,
            route_scale: 8.0,
            max_position_embeddings: 1 << 20,
        }
    );
    assert_eq!(got.vocab_size(), 201_024);
    assert_eq!(
        got.bytes().len(),
        std::mem::size_of::<ffi::Inkling2Config>()
    );
}

/// Inkling 2 Mini (inkling design note S6g): the checkpoint's `vision_config` maps to the tower the library runs
/// (one frame per image, 3 x 20 x 20 values per patch; the template's image frame; the erf GELU), defaults where it
/// is silent; another activation and a fractional `mlp_ratio` size are refused; no `vision_config`, no tower.
#[test]
fn an_inkling_vision_config_maps_to_the_tower() {
    let dir = root().join("tests/fixtures/model_configs/inkling-2-mini");
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let want = ffi::Qwen4ExpVisionConfig {
        struct_size: std::mem::size_of::<ffi::Qwen4ExpVisionConfig>() as u32,
        abi_version: ffi::ABI_VERSION,
        depth: 23,
        hidden_size: 1024,
        intermediate_size: 4096,
        num_heads: 16,
        patch_size: 20,
        temporal_patch_size: 1,
        spatial_merge_size: 2,
        in_channels: 3,
        out_hidden_size: 2304,
        num_position_embeddings: 1024,
        deepstack_count: 0,
        hidden_act: ffi::ACT_GELU_ERF,
        image_token_id: 200_054,
        video_token_id: 0,
        vision_start_token_id: 200_005,
        vision_end_token_id: 200_010,
        layer_norm_eps: 1e-5,
    };
    assert_eq!(inkling_vision_config(&config).unwrap(), Some(want));
    let mut silent = config.clone();
    silent["vision_config"] = serde_json::json!({"text_hidden_size": 2304});
    assert_eq!(inkling_vision_config(&silent).unwrap(), Some(want));
    for (key, value) in [
        ("hidden_act", Value::from("gelu_pytorch_tanh")),
        ("mlp_ratio", Value::from(4.0001)),
    ] {
        let mut bad = config.clone();
        bad["vision_config"][key] = value;
        let err = inkling_vision_config(&bad).unwrap_err();
        assert!(err.to_string().contains(key), "{err}");
    }
    let mut text = config.clone();
    text.as_object_mut().unwrap().remove("vision_config");
    assert_eq!(inkling_vision_config(&text).unwrap(), None);
}

#[test]
fn a_changed_config_value_is_unsupported_before_any_put() {
    let t = Tmp::new("config");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    // The unchanged file and an explicit default round-trip to the same struct.
    assert_eq!(changed_config("hidden_size", 32.into()), spec.config);
    assert_eq!(changed_config("norm_topk_prob", true.into()), spec.config);
    let changes: [(&str, Value); 8] = [
        ("vocab_size", 70_001.into()),
        ("hidden_size", 64.into()),
        ("num_hidden_layers", 2.into()),
        ("num_experts", 4.into()),
        ("moe_intermediate_size", 32.into()),
        ("norm_topk_prob", false.into()),
        ("ple_layer_ids", serde_json::json!([3])),
        ("head_dim", 256.into()),
    ];
    let d = toy_driver();
    for (key, value) in changes {
        let changed = ModelSpec {
            config: changed_config(key, value),
            ..spec.clone()
        };
        assert_ne!(changed.config, spec.config, "{key}");
        let err = upload(&d, &art, &changed, &options(4)).unwrap_err();
        // Refused by `cere_model_create` (ALLOCATE): no model exists, so no put was made.
        assert!(
            matches!(
                err,
                UploadError::Driver(DriverError::Status {
                    call: "cere_model_create",
                    status: StatusError::Unsupported,
                })
            ),
            "{key}: {err:?}"
        );
    }
    // The executor is unharmed: the unchanged desc still uploads.
    let up = upload(&d, &art, &spec, &options(4)).unwrap();
    d.release_model(up.model).unwrap();
    d.shutdown().unwrap();
}

/// `spec` with the first device tensor of more than one row declared one row shorter: every put of its last
/// row is out of range, so the executor refuses it inside the staging thread.
fn one_row_short(spec: &ModelSpec) -> (ModelSpec, String) {
    let mut short = spec.clone();
    let t = short
        .tensors
        .iter_mut()
        .find(|t| t.placement == ffi::PLACEMENT_DEVICE && t.shape.len() >= 2 && t.shape[0] > 1)
        .unwrap();
    let row = t.bytes / t.shape[0];
    t.shape[0] -= 1;
    t.bytes -= row;
    let name = t.name.clone();
    (short, name)
}

#[test]
fn a_sink_error_aborts_staging_with_nothing_sealed() {
    let t = Tmp::new("sink");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let (short, name) = one_row_short(&spec);
    let d = toy_driver();

    // The whole upload: staging aborts with class `sink`, naming the refused tensor; the unsealed model is released.
    let err = upload(&d, &art, &short, &options(4)).unwrap_err();
    match &err {
        UploadError::Stage(e) => {
            assert_eq!(e.class, "sink", "{err:?}");
            assert!(e.message.contains(&name), "{err:?}");
            assert!(e.message.contains("cere_model_put: argument"), "{err:?}");
        }
        other => panic!("expected a staging error, got {other:?}"),
    }

    // The same staging on a model kept by the test: after the abort the model cannot be sealed. With 4 buffers the
    // caller makes the staging block beforehand (LSF-03); a block of another length is refused before any read.
    for buffers in [1, 4] {
        let mut model = d.create_model(&short).unwrap();
        if buffers == 4 {
            let short_block = staging_block(&options(1)).unwrap();
            let err =
                stage_all(&d, &model, &art, &short, &options(4), Some(short_block)).unwrap_err();
            assert_eq!(err.class(), "buffer", "{err:?}");
        }
        let block = (buffers == 4).then(|| staging_block(&options(buffers)).unwrap());
        let err = stage_all(&d, &model, &art, &short, &options(buffers), block).unwrap_err();
        assert_eq!(err.class(), "sink", "{err:?}");
        let sealed = d.seal(&mut model).unwrap_err();
        assert_eq!(sealed.status(), Some(StatusError::Argument), "{sealed:?}");
        d.release_model(model).unwrap();
    }
    d.shutdown().unwrap();
}

#[test]
fn mtp_draft_tensors_reach_the_executor_only_as_discard() {
    let t = Tmp::new("mtp");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let draft: Vec<String> = art
        .tensors()
        .filter(|t| t.role == cere_compiler::families::Role::Draft)
        .map(|t| t.name.to_string())
        .collect();
    let (count, bytes) = inventory_draft();
    assert_eq!(draft.len() as u64, count);
    for t in &spec.tensors {
        let is_draft = draft.contains(&t.name);
        assert_eq!(
            t.placement == ffi::PLACEMENT_DISCARD,
            is_draft,
            "{}: placement {}",
            t.name,
            t.placement
        );
        if is_draft {
            assert!(t.name.starts_with("mtp."), "{}", t.name);
        }
    }
    let d = toy_driver();
    let up = upload(&d, &art, &spec, &options(4)).unwrap();
    assert_eq!(up.staging.counters.discarded_bytes, bytes);
    assert_eq!(
        up.staging.counters.uploaded_device_bytes + up.staging.counters.uploaded_host_bytes,
        inventory_totals().device + inventory_totals().host
    );
    d.release_model(up.model).unwrap();
    d.shutdown().unwrap();
}

/// m29 design note C6 (D11, M29S2-15): with `speculative_draft_tokens` 4 the MTP draft tensors are DEVICE (the
/// rest unchanged: the shared-PLE HOST_PINNED segment keeps its placement) and the desc carries the artifact's token
/// map (strictly increasing u32 ids, `Artifact::token_map`); with 0 they stay DISCARD and the desc has no map.
#[test]
fn drafts_are_device_with_4_and_discard_with_0_and_the_map_rides_with_4() {
    let t = Tmp::new("drafts");
    let (dir, art) = opened(&t);
    let off = model_spec(&art, &dir, &PLAN).unwrap();
    let on = model_spec(&art, &dir, &PLAN_DRAFTS).unwrap();
    assert_eq!(off.speculative_draft_tokens, 0);
    assert!(off.draft_token_map.is_empty());
    assert_eq!(on.speculative_draft_tokens, 4);
    let map = art.token_map().unwrap();
    assert!(!map.is_empty());
    assert_eq!(on.draft_token_map, map);
    assert!(on.draft_token_map.windows(2).all(|w| w[0] < w[1]));
    assert!(on
        .draft_token_map
        .iter()
        .all(|&id| u64::from(id) < u64::from(FIXTURE_VOCAB)));
    let draft: Vec<String> = art
        .tensors()
        .filter(|t| t.role == cere_compiler::families::Role::Draft)
        .map(|t| t.name.to_string())
        .collect();
    assert!(!draft.is_empty());
    assert_eq!(off.tensors.len(), on.tensors.len());
    for (a, b) in off.tensors.iter().zip(&on.tensors) {
        assert_eq!(a.name, b.name);
        if draft.contains(&a.name) {
            assert_eq!(a.placement, ffi::PLACEMENT_DISCARD, "{}", a.name);
            assert_eq!(b.placement, ffi::PLACEMENT_DEVICE, "{}", b.name);
        } else {
            assert_eq!(a, b);
        }
    }
    let inv = inventory_totals();
    let sum = |s: &ModelSpec, p: u32| -> u64 {
        s.tensors
            .iter()
            .filter(|t| t.placement == p)
            .map(|t| t.bytes)
            .sum()
    };
    assert_eq!(sum(&on, ffi::PLACEMENT_DEVICE), inv.device + inv.draft);
    assert_eq!(sum(&on, ffi::PLACEMENT_HOST_PINNED), inv.host);
    assert_eq!(sum(&on, ffi::PLACEMENT_DISCARD), 0);
}

/// m29 design note C6 (M29S2-15): on the shipped artifact's inventory the drafts-off path DISCARDs exactly the
/// 31 MTP tensors' 5,214,301,696 bytes (as today, `discarded_bytes` of every live probe); drafts on moves them to
/// DEVICE and leaves the HOST_PINNED bytes (the shared PLE segment) untouched.
#[test]
fn shipped_inventory_discard_path_is_unchanged_and_drafts_move_to_device() {
    let raw =
        std::fs::read(root().join("configs/artifacts/qwen38-flash-next-nvfp4.inventory.json"))
            .unwrap();
    let inv: Value = serde_json::from_slice(&raw).unwrap();
    let totals = |drafts: u32| {
        let mut out = std::collections::BTreeMap::<u32, u64>::new();
        for f in inv["families"].as_array().unwrap() {
            let role = cere_compiler::families::Role::parse(f["role"].as_str().unwrap()).unwrap();
            if let Some(p) = placement(f["placement"].as_str().unwrap(), role, drafts) {
                *out.entry(p).or_default() += f["bytes"].as_u64().unwrap();
            }
        }
        out
    };
    let off = totals(0);
    assert_eq!(off[&ffi::PLACEMENT_DISCARD], 5_214_301_696);
    assert_eq!(off[&ffi::PLACEMENT_DEVICE], 77_843_711_744);
    assert_eq!(off[&ffi::PLACEMENT_HOST_PINNED], 51_200_246_042);
    let on = totals(4);
    assert_eq!(on.get(&ffi::PLACEMENT_DISCARD), None);
    assert_eq!(on[&ffi::PLACEMENT_DEVICE], 77_843_711_744 + 5_214_301_696);
    assert_eq!(on[&ffi::PLACEMENT_HOST_PINNED], 51_200_246_042);
}

/// m29 design note C6 over the toy: a drafts-on upload puts the MTP bytes as DEVICE (nothing discarded) and the
/// sealed model reports `native_speculative_batch` 1, VERIFY and 4 draft tokens.
#[test]
fn drafts_on_upload_puts_the_mtp_bytes_on_device() {
    let t = Tmp::new("drafts-up");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN_DRAFTS).unwrap();
    let d = toy_driver();
    let up = upload(&d, &art, &spec, &options(4)).unwrap();
    let inv = inventory_totals();
    assert_eq!(up.staging.counters.discarded_bytes, 0);
    assert_eq!(
        up.staging.counters.uploaded_device_bytes,
        inv.device + inv.draft
    );
    assert_eq!(up.staging.counters.uploaded_host_bytes, inv.host);
    assert_eq!(up.capabilities.native_speculative_batch, 1);
    assert_eq!(up.capabilities.speculative_draft_tokens, 4);
    assert_ne!(up.capabilities.operations & ffi::OPS_VERIFY, 0);
    d.release_model(up.model).unwrap();
    d.shutdown().unwrap();
}

fn staging_driver(library: &std::path::Path, device_id: &str) -> Driver {
    let lib = Arc::new(Library::open(library, None).unwrap());
    Driver::start(
        lib,
        &ExecutorConfig {
            device_id: device_id.to_string(),
            worker_epoch: 1,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap()
}

/// m216 design note M216S-11: with the put staging exports, the pass registers every staging buffer and its DEVICE
/// puts take the direct path: the toy holds each staged put while `staging_gate` exists, so the upload cannot finish
/// before the test removes the gate (300 ms after the start). The counters are the inventory's, the model serves, and
/// every buffer is unregistered by the end (cere_executor_destroy is BUSY while a range is registered).
#[test]
fn the_staging_buffers_are_registered_and_device_puts_take_the_direct_path() {
    let t = Tmp::new("staged");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let inv = inventory_totals();
    let gate = t.path().join("staging-gate");
    std::fs::write(&gate, b"").unwrap();
    let d = staging_driver(
        &toy_library(),
        &format!("toy,hold_ms=30000,staging_gate={}", gate.display()),
    );
    assert!(d.library().api().executor_register_staging.is_some());
    // Timed inside the scope: its end also waits for the gate thread.
    let (up, held) = std::thread::scope(|scope| {
        let start = std::time::Instant::now();
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(300));
            std::fs::remove_file(&gate).unwrap();
        });
        let up = upload(&d, &art, &spec, &options(4)).unwrap();
        (up, start.elapsed())
    });
    assert!(
        held >= std::time::Duration::from_millis(300),
        "no put waited on the staging gate: {held:?}"
    );
    assert_eq!(up.staging.registered_bytes, (4 * MIN_BUFFER_BYTES) as u64);
    let c = up.staging.counters;
    assert_eq!(c.uploaded_device_bytes, inv.device);
    assert_eq!(c.uploaded_host_bytes, inv.host);
    assert_eq!(c.discarded_bytes, inv.draft);
    let s = d.create_session(&up.model, 64).unwrap();
    let r = d
        .submit(&s, WorkRequest::prefill(vec![1, 2, 3], 1))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.tokens, toy_tokens(3, 3, 1, FIXTURE_VOCAB));
    d.release_session(s).unwrap();
    d.release_model(up.model).unwrap();
    d.shutdown().unwrap();
}

/// m216 design note M216S-11: a library from before the put staging exports (the toy built with
/// CERE_TOY_PRE_STAGING) still loads, nothing is registered, every put takes the bounce ring, the counters are the
/// inventory's and the sealed model serves the same token.
#[test]
fn a_library_without_put_staging_loads_on_the_bounce_ring() {
    let t = Tmp::new("prestaging");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let inv = inventory_totals();
    let d = staging_driver(&toy_library_prestaging(), "toy");
    let api = d.library().api();
    assert!(api.executor_register_staging.is_none() && api.executor_unregister_staging.is_none());
    let up = upload(&d, &art, &spec, &options(4)).unwrap();
    assert_eq!(up.staging.registered_bytes, 0);
    let c = up.staging.counters;
    assert_eq!(c.uploaded_device_bytes, inv.device);
    assert_eq!(c.uploaded_host_bytes, inv.host);
    assert_eq!(c.discarded_bytes, inv.draft);
    let s = d.create_session(&up.model, 64).unwrap();
    let r = d
        .submit(&s, WorkRequest::prefill(vec![1, 2, 3], 1))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.tokens, toy_tokens(3, 3, 1, FIXTURE_VOCAB));
    d.release_session(s).unwrap();
    d.release_model(up.model).unwrap();
    d.shutdown().unwrap();
}

/// m25 design note D4: the plan's pool flows through `ModelSpec` into `cere_model_desc.kv_pool_tokens`; the sealed
/// model's capabilities report the allocated pool (0 -> max_sessions x round64(context)) and the 64-token page; a pool
/// that is not a multiple of 64 or under the context is ARGUMENT and one over the support bound UNSUPPORTED, both
/// at `cere_model_create`.
#[test]
fn the_plan_pool_reaches_the_executor_and_its_capabilities() {
    let t = Tmp::new("pool");
    let (dir, art) = opened(&t);
    let plan = |max_sessions, kv_pool_tokens| PlanOptions {
        max_sessions,
        kv_pool_tokens,
        ..PLAN
    };
    let d = toy_driver();
    for (sessions, requested, allocated) in [(1, 0, 4096), (2, 0, 8192), (1, 12_288, 12_288)] {
        let spec = model_spec(&art, &dir, &plan(sessions, requested)).unwrap();
        assert_eq!(spec.kv_pool_tokens, requested);
        let up = upload(&d, &art, &spec, &options(4)).unwrap();
        assert_eq!(
            (
                up.capabilities.kv_pool_tokens,
                up.capabilities.kv_page_tokens
            ),
            (allocated, 64),
            "{sessions} sessions, pool {requested}"
        );
        d.release_model(up.model).unwrap();
    }
    for (requested, status) in [
        (4100, StatusError::Argument),
        (2048, StatusError::Argument),
        (1_048_640, StatusError::Unsupported),
    ] {
        let spec = model_spec(&art, &dir, &plan(1, requested)).unwrap();
        let err = upload(&d, &art, &spec, &options(4)).unwrap_err();
        assert!(
            matches!(
                &err,
                UploadError::Driver(DriverError::Status {
                    call: "cere_model_create",
                    status: s,
                }) if *s == status
            ),
            "pool {requested}: {err:?}"
        );
    }
    d.shutdown().unwrap();
}

/// FNV-1a 64 of `bytes`, as the toy's `tensor_digests` lines print it.
fn fnv(bytes: &[u8]) -> String {
    let h = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    format!("{h:016x}")
}

/// The digest of every desc tensor that is not DISCARD, from the artifact's files: what the toy must hold.
fn expected_digests(dir: &Path, art: &Artifact, spec: &ModelSpec) -> BTreeMap<u64, String> {
    let mut files: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    let mut out = BTreeMap::new();
    for (i, t) in spec.tensors.iter().enumerate() {
        if t.placement == ffi::PLACEMENT_DISCARD {
            continue;
        }
        let r = art.tensors().find(|r| r.name == t.name).unwrap();
        let raw = files
            .entry(r.path)
            .or_insert_with(|| std::fs::read(dir.join(r.path)).unwrap());
        out.insert(
            i as u64,
            fnv(&raw[r.offset as usize..(r.offset + r.bytes) as usize]),
        );
    }
    out
}

/// The toy's `tensor_digests` file: index -> digest.
fn toy_digests(path: &Path) -> BTreeMap<u64, String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| {
            let (i, h) = l.split_once(' ').unwrap();
            (i.parse().unwrap(), h.to_string())
        })
        .collect()
}

/// load-speed design note LS-05: a load with the default read mode reads a cold artifact with `O_DIRECT` (on a
/// file system that takes it and drops evicted pages) and a resident one (every file just read through the page
/// cache) buffered, and in both cases stores every tensor's file bytes. Fails on main 76071574 (the default was
/// buffered).
#[test]
fn the_default_read_is_direct_when_cold_and_buffered_when_resident() {
    use std::os::unix::fs::OpenOptionsExt;
    let t = Tmp::new("defread");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let want = expected_digests(&dir, &art, &spec);
    let opts = UploadOptions {
        buffers: 4,
        buffer_bytes: MIN_BUFFER_BYTES,
        readers: 0,
        consumers: 0,
        ..UploadOptions::default()
    };
    let (_, after) =
        artifact::evict_dir(&dir, Some(&art.manifest_sha256), &Default::default()).unwrap();
    let o_direct = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(dir.join("manifest.json"))
        .is_ok();
    // A file system that keeps evicted pages (tmpfs) leaves them resident: auto then reads them buffered.
    let cold = o_direct && after.basis_points() <= 100;
    for (k, expect_direct) in [(0, cold), (1, false)] {
        if k == 1 {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_file() {
                    std::fs::read(path).unwrap();
                }
            }
        }
        let digests = t.path().join(format!("digests-{k}"));
        let d = staging_driver(
            &toy_library(),
            &format!("toy,tensor_digests={}", digests.display()),
        );
        let up = upload(&d, &art, &spec, &opts).unwrap();
        assert_eq!(up.staging.pass.read, ReadMode::Auto, "load {k}");
        assert_eq!(up.staging.pass.direct_files > 0, expect_direct, "load {k}");
        assert_eq!(toy_digests(&digests), want, "load {k}");
        d.release_model(up.model).unwrap();
        d.shutdown().unwrap();
    }
}

/// m216 design note L7: the bytes every stored tensor holds after the upload equal its file bytes (FNV-1a 64 per
/// tensor, from the toy's `tensor_digests`), for buffered and direct reads, 1 to 4 buffers, the speculation plan
/// (draft tensors on DEVICE) and the bounce-ring fallback of a library without put staging.
#[test]
fn every_uploaded_tensor_is_byte_exact_for_buffered_and_direct_reads_and_the_bounce_ring() {
    let t = Tmp::new("digests");
    let (dir, art) = opened(&t);
    let rows = [
        (toy_library(), ReadMode::Buffered, 1, PLAN),
        (toy_library(), ReadMode::Buffered, 4, PLAN_DRAFTS),
        (toy_library(), ReadMode::Direct, 4, PLAN),
        (toy_library(), ReadMode::Auto, 3, PLAN_DRAFTS),
        (toy_library_prestaging(), ReadMode::Buffered, 4, PLAN),
        (toy_library_prestaging(), ReadMode::Direct, 2, PLAN_DRAFTS),
    ];
    for (k, (library, read, buffers, plan)) in rows.into_iter().enumerate() {
        let spec = model_spec(&art, &dir, &plan).unwrap();
        let digests = t.path().join(format!("digests-{k}"));
        let d = staging_driver(
            &library,
            &format!("toy,tensor_digests={}", digests.display()),
        );
        let up = upload(&d, &art, &spec, &read_options(buffers, read)).unwrap();
        assert_eq!(up.staging.pass.read, read, "row {k}");
        let want = expected_digests(&dir, &art, &spec);
        assert!(want.len() > 1, "row {k}");
        assert_eq!(
            toy_digests(&digests),
            want,
            "row {k}: {read:?}, {buffers} buffers"
        );
        d.release_model(up.model).unwrap();
        d.shutdown().unwrap();
    }
}

/// m216 design note L7 (one producer, N consumers; M3.2N): one pass uploads into two models of two executors. The
/// first target's staged puts wait on its `staging_gate` for 300 ms while the second's run free; with 2 buffers the
/// reads go on only as each buffer is released by BOTH targets, so each byte is read once, both models hold every
/// byte exactly (a buffer reused before the held put returned would corrupt the first model), and both serve.
#[test]
fn two_targets_share_one_read_and_each_buffer_is_reused_only_after_both_put_it() {
    let t = Tmp::new("two");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let inv = inventory_totals();
    let gate = t.path().join("gate");
    std::fs::write(&gate, b"").unwrap();
    let (da, db) = (t.path().join("digests-a"), t.path().join("digests-b"));
    let a = staging_driver(
        &toy_library(),
        &format!(
            "toy,hold_ms=30000,staging_gate={},tensor_digests={}",
            gate.display(),
            da.display()
        ),
    );
    let b = staging_driver(
        &toy_library(),
        &format!("toy,tensor_digests={}", db.display()),
    );
    let (mut ma, mut mb) = (
        a.create_model(&spec).unwrap(),
        b.create_model(&spec).unwrap(),
    );
    let opts = UploadOptions {
        readers: 2,
        consumers: 1,
        ..options(2)
    };
    let read_before = art.counters().payload_bytes_read;
    let (reports, held) = std::thread::scope(|scope| {
        let start = std::time::Instant::now();
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(300));
            std::fs::remove_file(&gate).unwrap();
        });
        let targets = [
            Target {
                driver: &a,
                model: &ma,
                spec: &spec,
            },
            Target {
                driver: &b,
                model: &mb,
                spec: &spec,
            },
        ];
        let r = stage_all_to(&art, &targets, &opts, None).unwrap();
        (r, start.elapsed())
    });
    assert!(held >= std::time::Duration::from_millis(300), "{held:?}");
    assert_eq!(
        art.counters().payload_bytes_read - read_before,
        inv.device + inv.host
    );
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert_eq!(r.counters.uploaded_device_bytes, inv.device);
        assert_eq!(r.counters.uploaded_host_bytes, inv.host);
        assert_eq!(r.registered_bytes, (2 * MIN_BUFFER_BYTES) as u64);
        assert_eq!(r.pass.consumers, 1);
        assert!(r.pass.high_water_bytes <= (2 * MIN_BUFFER_BYTES) as u64);
    }
    a.seal(&mut ma).unwrap();
    b.seal(&mut mb).unwrap();
    let want = expected_digests(&dir, &art, &spec);
    assert_eq!(toy_digests(&da), want, "the held target");
    assert_eq!(toy_digests(&db), want, "the free target");
    for (d, m) in [(&a, &ma), (&b, &mb)] {
        let s = d.create_session(m, 64).unwrap();
        let r = d
            .submit(&s, WorkRequest::prefill(vec![1, 2, 3], 1))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(r.tokens, toy_tokens(3, 3, 1, FIXTURE_VOCAB));
        d.release_session(s).unwrap();
    }
    a.release_model(ma).unwrap();
    b.release_model(mb).unwrap();
    a.shutdown().unwrap();
    b.shutdown().unwrap();
}

/// m216 design note L7 error path: a weights file that shrinks behind the open artifact fails the upload with
/// `file_size` naming it, for buffered and direct reads; the unsealed model is released and the driver shuts down
/// (every staging range unregistered).
#[test]
fn a_short_read_fails_the_upload_with_file_size_and_nothing_sealed() {
    let t = Tmp::new("short");
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let big = spec
        .tensors
        .iter()
        .filter(|t| t.placement == ffi::PLACEMENT_DEVICE)
        .max_by_key(|t| t.bytes)
        .unwrap();
    let r = art.tensors().find(|r| r.name == big.name).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(dir.join(r.path))
        .unwrap()
        .set_len(r.offset + r.bytes / 2)
        .unwrap();
    for read in [ReadMode::Buffered, ReadMode::Direct] {
        let d = toy_driver();
        let err = upload(&d, &art, &spec, &read_options(4, read)).unwrap_err();
        match &err {
            UploadError::Stage(e) => {
                assert_eq!(e.class, "file_size", "{read:?}: {err:?}");
                assert!(e.message.contains(r.path), "{read:?}: {err:?}");
            }
            other => panic!("{read:?}: expected a staging error, got {other:?}"),
        }
        d.shutdown().unwrap();
    }
}

fn share_driver(share: &std::path::Path, artifact_sha: &str, keep: bool) -> Driver {
    let lib = Arc::new(Library::open(&toy_library(), None).unwrap());
    let keep = if keep { ",host_share_keep=1" } else { "" };
    Driver::start(
        lib,
        &ExecutorConfig {
            device_id: format!(
                "toy,host_share={},artifact={artifact_sha}{keep}",
                share.display()
            ),
            worker_epoch: 1,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap()
}

/// L4 (m216 design note C3): a restart attaches the kept segment, so the all-HOST_PINNED files are checked from
/// the segment against their manifest sha256 instead of read or put; every other byte is staged as before.
#[test]
fn an_attached_segment_skips_reading_the_pinned_files() {
    let t = Tmp::new("resident");
    let share = t.path().join("share");
    std::fs::create_dir(&share).unwrap();
    let (dir, art) = opened(&t);
    let spec = model_spec(&art, &dir, &PLAN).unwrap();
    let inv = inventory_totals();
    // The creator keeps the published segment across its stop.
    let d = share_driver(&share, &art.manifest_sha256, true);
    let up = upload(&d, &art, &spec, &options(4)).unwrap();
    assert_eq!(up.staging.resident_files, 0);
    assert_eq!(up.staging.counters.uploaded_host_bytes, inv.host);
    d.release_model(up.model).unwrap();
    d.shutdown().unwrap();
    // The restart attaches: the pinned files are not read.
    let art = artifact::open(&dir, &art.manifest_sha256, Mode::Stamp).unwrap();
    let before = art.counters().payload_bytes_read;
    let d = share_driver(&share, &art.manifest_sha256, false);
    let up = upload(&d, &art, &spec, &options(4)).unwrap();
    let c = up.staging.counters;
    assert!(up.staging.resident_files >= 1);
    assert!(c.resident_host_bytes > 0);
    assert_eq!(c.uploaded_device_bytes, inv.device);
    assert_eq!(c.uploaded_host_bytes + c.resident_host_bytes, inv.host);
    assert_eq!(
        art.counters().payload_bytes_read - before,
        inv.device + c.uploaded_host_bytes
    );
    d.release_model(up.model).unwrap();
    d.shutdown().unwrap();
    // The attacher without keep was the last mapper: the segment is gone.
    let left: Vec<_> = std::fs::read_dir(&share)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| !e.file_name().to_string_lossy().ends_with(".lock"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
}
