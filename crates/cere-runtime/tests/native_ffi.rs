//! R1 (spec m22, "ABI v3 contract"; ABI v4 since m23 design note PI-03): the Rust projection equals the C layout
//! fixture, the loader refuses a library by hash before `dlopen` and by ABI version or missing symbol after it, and
//! the driver thread runs the create/model/session/submit/poll/cancel round trip against the CPU toy executor
//! (`$CERE_TOY_EXECUTOR`).
mod native_support;

use cere_compiler::artifact::{self, Mode};
use cere_runtime::levers::{
    DraftTokens, Drafter, Family, K2Gemm, K2Weights, LeverSet, MoeGemm, PrefillAttention,
};
use cere_runtime::native::driver::{
    Committed, Driver, DriverError, ExecutorConfig, Finish, ModelSpec, Sampling, Session, Snapshot,
    WorkRequest,
};
use cere_runtime::native::export::{
    self, Envelope, ExportError, Exported, LockedBuf, StateIdentity,
};
use cere_runtime::native::ffi::{self, Library, LoadError, StatusError};
use cere_runtime::native::upload::{model_spec, PlanOptions};
use native_support::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::mem::{align_of, offset_of, size_of};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

fn layout_fixture() -> Value {
    let raw = std::fs::read(root().join("tests/fixtures/executor_abi_v5_layout.json")).unwrap();
    serde_json::from_slice(&raw).unwrap()
}

fn field_size<T, F>(_: fn(&T) -> &F) -> usize {
    size_of::<F>()
}

/// (size, align, [(field, offset, size)]) of a Rust projection struct.
macro_rules! layout {
    ($t:ty { $($f:ident),* $(,)? }) => {
        (
            size_of::<$t>(),
            align_of::<$t>(),
            vec![$((stringify!($f), offset_of!($t, $f), field_size::<$t, _>(|s: &$t| &s.$f))),*],
        )
    };
}

type Layout = (usize, usize, Vec<(&'static str, usize, usize)>);

fn rust_layouts() -> BTreeMap<&'static str, Layout> {
    use ffi::*;
    BTreeMap::from([
        (
            "cere_executor_options",
            layout!(ExecutorOptions {
                struct_size,
                abi_version,
                device_id,
                worker_epoch,
                memory_budget_bytes
            }),
        ),
        (
            "cere_execution_capabilities",
            layout!(ExecutionCapabilities {
                struct_size,
                abi_version,
                profile_generation,
                worker_epoch,
                decode_batch_min,
                decode_batch_max,
                max_context_tokens,
                execution_mode,
                native_decode_batch,
                native_mixed_prefill_decode,
                native_speculative_batch,
                native_moe_grouping,
                native_state_snapshot,
                native_sampler,
                native_token_mask,
                speculative_draft_tokens,
                native_prefix_cache,
                native_image_input,
                reserved,
                operations,
                kv_pool_tokens,
                kv_page_tokens,
                reserved2
            }),
        ),
        (
            "cere_completion_ticket",
            layout!(CompletionTicket {
                worker_epoch,
                model_generation,
                submission_id
            }),
        ),
        (
            "cere_qwen4_exp_config",
            layout!(Qwen4ExpConfig {
                struct_size,
                abi_version,
                vocab_size,
                hidden_size,
                num_hidden_layers,
                full_attention_interval,
                full_attention_layers,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                output_gate_type,
                rope_theta,
                partial_rotary_factor,
                mrope_interleaved,
                mrope_section,
                indexer_n_heads,
                indexer_head_dim,
                indexer_kv_heads,
                indexer_compress_ratio,
                indexer_budget,
                linear_num_key_heads,
                linear_num_value_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
                mamba_ssm_dtype,
                num_experts,
                num_experts_per_tok,
                moe_intermediate_size,
                shared_expert_intermediate_size,
                norm_topk_prob,
                hc_count,
                hc_lowrank,
                ple_layer_ids,
                ple_embed_dim,
                heads_per_ngram,
                ngram_size,
                ple_conv_kernel_size,
                ngram_vocab_size_base,
                split_ngram_parts,
                make_ngram_vocab_size_divisible_by,
                rms_norm_eps,
                tie_word_embeddings,
                mtp_num_hidden_layers,
                max_position_embeddings
            }),
        ),
        (
            "cere_sdar_config",
            layout!(SdarConfig {
                struct_size,
                abi_version,
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                block_size,
                rope_theta,
                rms_norm_eps,
                mask_token_id,
                tie_word_embeddings,
                max_position_embeddings
            }),
        ),
        (
            "cere_inkling2_config",
            layout!(Inkling2Config {
                struct_size,
                abi_version,
                vocab_size,
                unpadded_vocab_size,
                hidden_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                moe_intermediate_size,
                num_experts,
                num_experts_per_tok,
                sliding_window_size,
                rel_extent,
                d_rel,
                loop_core_start,
                loop_core_end,
                loop_num_passes,
                loop_gate_rank,
                n_layers_per_attn_res_block,
                conv_kernel_size,
                eos_token_id,
                local_layer_mask,
                mxfp8_layer_mask,
                rms_norm_eps,
                final_logit_softcapping,
                route_scale,
                max_position_embeddings
            }),
        ),
        (
            "cere_qwen4_exp_vision_config",
            layout!(Qwen4ExpVisionConfig {
                struct_size,
                abi_version,
                depth,
                hidden_size,
                intermediate_size,
                num_heads,
                patch_size,
                temporal_patch_size,
                spatial_merge_size,
                in_channels,
                out_hidden_size,
                num_position_embeddings,
                deepstack_count,
                hidden_act,
                image_token_id,
                video_token_id,
                vision_start_token_id,
                vision_end_token_id,
                layer_norm_eps
            }),
        ),
        (
            "cere_tensor_desc",
            layout!(TensorDesc {
                struct_size,
                abi_version,
                name,
                dtype,
                placement,
                ndim,
                reserved,
                shape,
                bytes
            }),
        ),
        (
            "cere_slot_choice",
            layout!(SlotChoice {
                slot,
                r#impl,
                params
            }),
        ),
        (
            "cere_slot_selection",
            layout!(SlotSelection {
                struct_size,
                abi_version,
                block_version,
                count,
                choices
            }),
        ),
        (
            "cere_model_desc",
            layout!(ModelDesc {
                struct_size,
                abi_version,
                model_kind,
                speculative_draft_tokens,
                config,
                tensors,
                tensor_count,
                max_sessions,
                max_context_tokens,
                stop_token_ids,
                stop_token_count,
                kv_pool_tokens,
                draft_token_map,
                draft_token_map_count,
                snapshot_host_bytes,
                prefix_state_slots,
                vision_config,
                selection,
                family,
                calibrate_windows
            }),
        ),
        (
            "cere_token_score",
            layout!(TokenScore {
                id,
                runner_up,
                logit,
                runner_up_logit
            }),
        ),
        ("cere_logit_bias", layout!(LogitBias { id, bias })),
        (
            "cere_sampling_params",
            layout!(SamplingParams {
                struct_size,
                abi_version,
                temperature,
                top_p,
                min_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                repetition_penalty,
                flags,
                seed,
                logit_bias,
                logit_bias_count,
                suppress_token_ids,
                suppress_token_count,
                mask_until_committed
            }),
        ),
        ("cere_logprob_entry", layout!(LogprobEntry { id, logprob })),
        (
            "cere_token_logprobs",
            layout!(TokenLogprobs {
                id,
                logprob,
                top_count,
                reserved,
                top
            }),
        ),
        (
            "cere_image_input",
            layout!(ImageInput {
                struct_size,
                abi_version,
                grid_t,
                grid_h,
                grid_w,
                pad_id,
                token_offset,
                pixel_patches,
                pixel_bytes,
                pixel_sha256
            }),
        ),
        (
            "cere_work",
            layout!(Work {
                struct_size,
                abi_version,
                operation,
                flags,
                expected_worker_epoch,
                expected_session_revision,
                session,
                tokens,
                token_count,
                max_new_tokens,
                reserved_peak_bytes,
                out_tokens,
                out_capacity,
                out_scores,
                stop_token_ids,
                stop_token_count,
                sampling,
                out_logprobs,
                top_logprobs,
                capture_position,
                vocab_mask,
                vocab_mask_words,
                images,
                image_count
            }),
        ),
        (
            "cere_batch",
            layout!(Batch {
                struct_size,
                abi_version,
                expected_worker_epoch,
                capability_profile_generation,
                items,
                item_count,
                reserved_peak_bytes
            }),
        ),
        (
            "cere_completion",
            layout!(Completion {
                struct_size,
                abi_version,
                status,
                terminal,
                committed_tokens,
                session_revision,
                released_bytes,
                finish,
                cached_tokens,
                verify_steps,
                accepted_drafts
            }),
        ),
        (
            "cere_snapshot_info",
            layout!(SnapshotInfo {
                struct_size,
                abi_version,
                bytes,
                position,
                token_count,
                fed,
                layout_id
            }),
        ),
        (
            "cere_prefix_stats",
            layout!(PrefixStats {
                struct_size,
                abi_version,
                alt_edges_on,
                reserved,
                hits,
                captures,
                capture_skipped,
                capture_page_mismatch,
                inserts,
                state_evictions,
                leaf_evictions,
                host_spills,
                host_loads,
                host_drops,
                alt_edges
            }),
        ),
        (
            "cere_prefix_capture_stats",
            layout!(PrefixCaptureStats {
                struct_size,
                abi_version,
                skipped_prefill_slot,
                skipped_decode_track,
                skipped_end_pair,
                hit_shortfall_tokens,
                hit_shortfall_lookups
            }),
        ),
        ("cere_research_range", layout!(ResearchRange { first, end })),
        (
            "cere_research_op",
            layout!(ResearchOp {
                kind,
                top_k,
                point,
                layers,
                prompt,
                prompt_count,
                generated,
                generated_max,
                alpha,
                col_begin,
                col_end,
                vector_rows,
                vector,
                vector_cols,
                expert_count,
                experts,
                weights
            }),
        ),
        (
            "cere_research_plan",
            layout!(ResearchPlan {
                ops,
                op_count,
                reserved,
                sha256
            }),
        ),
        (
            "cere_research_tensor",
            layout!(ResearchTensor {
                name,
                op,
                layer,
                dtype,
                cols,
                rows,
                positions,
                data
            }),
        ),
        (
            "cere_research_capture",
            layout!(ResearchCapture {
                tensors,
                tensor_count,
                bytes,
                prefix_lookup_skipped,
                reserved,
                plan_sha256
            }),
        ),
    ])
}

#[test]
fn struct_layouts_equal_the_c_fixture() {
    let fixture = layout_fixture();
    let rust = rust_layouts();
    let structs = fixture["structs"].as_array().unwrap();
    assert_eq!(structs.len(), rust.len(), "struct count");
    for s in structs {
        let name = s["name"].as_str().unwrap();
        let (size, align, fields) = rust
            .get(name)
            .unwrap_or_else(|| panic!("{name} has no Rust projection"));
        assert_eq!(*size as u64, s["size"].as_u64().unwrap(), "{name} size");
        assert_eq!(*align as u64, s["align"].as_u64().unwrap(), "{name} align");
        let want: Vec<(String, u64, u64)> = s["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (
                    f["name"].as_str().unwrap().to_string(),
                    f["offset"].as_u64().unwrap(),
                    f["size"].as_u64().unwrap(),
                )
            })
            .collect();
        let got: Vec<(String, u64, u64)> = fields
            .iter()
            .map(|(f, o, z)| (f.trim_start_matches("r#").to_string(), *o as u64, *z as u64))
            .collect();
        assert_eq!(got, want, "{name} fields");
    }
}

#[test]
fn constants_and_symbols_equal_the_c_fixture() {
    let fixture = layout_fixture();
    assert_eq!(fixture["abi_version"], ffi::ABI_VERSION);
    let rust: BTreeMap<&str, i64> = ffi::CONSTANTS.iter().copied().collect();
    let constants = fixture["constants"].as_array().unwrap();
    assert_eq!(constants.len(), rust.len(), "constant count");
    for c in constants {
        let name = c["name"].as_str().unwrap();
        assert_eq!(
            rust.get(name).copied(),
            c["value"].as_i64(),
            "constant {name}"
        );
    }
    let symbols: Vec<&str> = fixture["functions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap())
        .collect();
    assert_eq!(ffi::SYMBOLS.to_vec(), symbols);
}

#[test]
fn library_hash_mismatch_is_refused_before_dlopen() {
    let t = Tmp::new("hash");
    // Not an ELF object: dlopen would fail, so a hash error proves the hash was checked first.
    let bogus = t.join("libbogus.so");
    std::fs::write(&bogus, b"not a shared object").unwrap();
    let actual = cere_compiler::sha256_hex(b"not a shared object");
    let wrong = "0".repeat(64);
    match Library::open(&bogus, Some(&wrong)) {
        Err(LoadError::HashMismatch {
            expected,
            actual: a,
        }) => {
            assert_eq!(expected, wrong);
            assert_eq!(a, actual);
        }
        other => panic!("expected a hash mismatch, got {other:?}"),
    }
    assert_eq!(
        Library::open(&bogus, Some(&wrong)).unwrap_err().class(),
        "library_hash"
    );
    // With the right hash the loader goes on to dlopen, which refuses the file.
    let err = Library::open(&bogus, Some(&actual)).unwrap_err();
    assert!(matches!(err, LoadError::Open(_)), "{err:?}");
    let missing = Library::open(&t.join("absent.so"), None).unwrap_err();
    assert!(matches!(missing, LoadError::Io { .. }), "{missing:?}");
}

/// Builds `source` into a shared library in `t` with the host C compiler (`$CC` or `cc`, as `make test-c`).
fn stub_library(t: &Tmp, name: &str, source: &str) -> PathBuf {
    let c = t.join(&format!("{name}.c"));
    std::fs::write(&c, source).unwrap();
    let out = t.join(&format!("lib{name}.so"));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let st = Command::new(&cc)
        .args(["-shared", "-fPIC", "-o"])
        .arg(&out)
        .arg(&c)
        .status()
        .unwrap_or_else(|e| panic!("running {cc}: {e}"));
    assert!(st.success(), "{cc} failed for {name}");
    out
}

#[test]
fn wrong_abi_version_or_missing_symbol_is_a_typed_error() {
    let t = Tmp::new("abi");
    // A library loaded (and kept, RTLD_NODELETE) first must not shadow the files loaded after it.
    let toy = Library::open(&toy_library(), None).unwrap();
    assert_eq!(toy.abi_version(), ffi::ABI_VERSION);
    drop(toy);
    let v2 = stub_library(
        &t,
        "abi_v2",
        "unsigned int cere_executor_abi_version(void) { return 2; }\n",
    );
    let err = Library::open(&v2, None).unwrap_err();
    assert!(matches!(err, LoadError::AbiVersion { found: 2 }), "{err:?}");
    assert_eq!(err.class(), "abi_version");
    // m23 design note PI-03: a v3 library (cere_model_desc without `levers`) is refused before any symbol.
    let v3 = stub_library(
        &t,
        "abi_v3",
        "unsigned int cere_executor_abi_version(void) { return 3; }\n",
    );
    let err = Library::open(&v3, None).unwrap_err();
    assert!(matches!(err, LoadError::AbiVersion { found: 3 }), "{err:?}");
    // selftune design note ST2: a v4 library (`cere_levers` instead of the slot-selection block) is refused too.
    let v4 = stub_library(
        &t,
        "abi_v4",
        "unsigned int cere_executor_abi_version(void) { return 4; }\n",
    );
    let err = Library::open(&v4, None).unwrap_err();
    assert!(matches!(err, LoadError::AbiVersion { found: 4 }), "{err:?}");
    let partial = stub_library(
        &t,
        "abi_v5_partial",
        "unsigned int cere_executor_abi_version(void) { return 5; }\n",
    );
    let err = Library::open(&partial, None).unwrap_err();
    assert!(
        matches!(err, LoadError::MissingSymbol("cere_executor_create")),
        "{err:?}"
    );
    assert_eq!(err.class(), "library_symbol");
    let none = stub_library(&t, "no_symbols", "int unrelated(void) { return 0; }\n");
    let err = Library::open(&none, None).unwrap_err();
    assert!(
        matches!(err, LoadError::MissingSymbol("cere_executor_abi_version")),
        "{err:?}"
    );
}

#[test]
fn toy_refuses_a_struct_with_the_wrong_abi_version_as_unsupported() {
    let lib = Library::open(&toy_library(), None).unwrap();
    assert_eq!(lib.abi_version(), ffi::ABI_VERSION);
    let device = std::ffi::CString::new("toy").unwrap();
    let mut options = ffi::ExecutorOptions {
        struct_size: size_of::<ffi::ExecutorOptions>() as u32,
        abi_version: 2,
        device_id: device.as_ptr(),
        worker_epoch: 1,
        memory_budget_bytes: 1 << 30,
    };
    let mut out: *mut ffi::RawExecutor = std::ptr::null_mut();
    // SAFETY: valid options and out pointer; the toy follows the header contract.
    let rc = unsafe { (lib.api().executor_create)(&mut out, &options) };
    assert_eq!(StatusError::check(rc), Err(StatusError::Unsupported));
    assert!(out.is_null(), "outputs stay null on error");
    options.abi_version = ffi::ABI_VERSION;
    options.struct_size -= 8;
    // SAFETY: as above.
    let rc = unsafe { (lib.api().executor_create)(&mut out, &options) };
    assert_eq!(StatusError::check(rc), Err(StatusError::Argument));
    assert!(out.is_null());
}

fn toy_driver(budget: u64) -> Driver {
    toy_driver_on("toy", budget)
}

fn toy_driver_on(device_id: &str, budget: u64) -> Driver {
    let lib = Arc::new(Library::open(&toy_library(), None).unwrap());
    Driver::start(
        lib,
        &ExecutorConfig {
            device_id: device_id.to_string(),
            worker_epoch: 7,
            memory_budget_bytes: budget,
        },
    )
    .unwrap()
}

/// O79 (m210 design note PF1): `cere_executor_set_decode_rounds` takes 1..=64 and refuses anything else as
/// ARGUMENT; a library predating it (the CERE_TOY_PRE_STAGING toy) has no symbol and the driver says so.
#[test]
fn set_decode_rounds_is_bounded_and_optional() {
    let d = toy_driver(1 << 30);
    assert!(d.library().api().executor_set_decode_rounds.is_some());
    for ok in [1, 4, ffi::MAX_DECODE_ROUNDS_PER_CHUNK] {
        d.set_decode_rounds(ok)
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for bad in [0, ffi::MAX_DECODE_ROUNDS_PER_CHUNK + 1] {
        assert_eq!(
            d.set_decode_rounds(bad),
            Err(DriverError::Status {
                call: "cere_executor_set_decode_rounds",
                status: StatusError::Argument,
            }),
            "{bad}"
        );
    }
    d.shutdown().unwrap();

    let old = Driver::start(
        Arc::new(Library::open(&toy_library_prestaging(), None).unwrap()),
        &ExecutorConfig {
            device_id: "toy".into(),
            worker_epoch: 7,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap();
    assert!(old.library().api().executor_set_decode_rounds.is_none());
    let err = old.set_decode_rounds(4).unwrap_err();
    assert_eq!(err, DriverError::Missing("cere_executor_set_decode_rounds"));
    assert_eq!(
        err.to_string(),
        "the library predates cere_executor_set_decode_rounds"
    );
    old.shutdown().unwrap();
}

/// `cere_model_prefix_stats` (appended to ABI v4): a fresh model with prefix state slots reads all counters 0 with the
/// header the driver set and `alt_edges_on` following `prefix_alt_edges`; a model without a prefix cache is
/// UNSUPPORTED (an error the runtime shows as "no counters"); a library from before the export is `Missing`.
#[test]
fn prefix_stats_reads_zero_counters_and_is_optional() {
    let d = toy_driver(1 << 30);
    assert!(d.library().api().model_prefix_stats.is_some());
    for alt in [false, true] {
        let spec = ModelSpec {
            prefix_state_slots: 4,
            prefix_alt_edges: alt,
            ..fixture_spec(4096)
        };
        let (model, _) = sealed_fixture_model(&d, &spec);
        let stats = d.prefix_stats(&model).unwrap();
        assert_eq!(
            stats,
            ffi::PrefixStats {
                struct_size: size_of::<ffi::PrefixStats>() as u32,
                abi_version: ffi::ABI_VERSION,
                alt_edges_on: u32::from(alt),
                ..ffi::PrefixStats::default()
            }
        );
        assert_eq!(
            d.prefix_stats(&model).unwrap(),
            stats,
            "a read changes nothing"
        );
        d.release_model(model).unwrap();
    }
    let (off, _) = sealed_fixture_model(&d, &fixture_spec(4096));
    assert_eq!(
        d.prefix_stats(&off),
        Err(DriverError::Status {
            call: "cere_model_prefix_stats",
            status: StatusError::Unsupported,
        })
    );
    d.release_model(off).unwrap();
    d.shutdown().unwrap();

    let old = Driver::start(
        Arc::new(Library::open(&toy_library_prestaging(), None).unwrap()),
        &ExecutorConfig {
            device_id: "toy".into(),
            worker_epoch: 7,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap();
    assert!(old.library().api().model_prefix_stats.is_none());
    let spec = ModelSpec {
        prefix_state_slots: 4,
        ..fixture_spec(4096)
    };
    let (model, _) = sealed_fixture_model(&old, &spec);
    let err = old.prefix_stats(&model).unwrap_err();
    assert_eq!(err, DriverError::Missing("cere_model_prefix_stats"));
    assert_eq!(
        err.to_string(),
        "the library predates cere_model_prefix_stats"
    );
    old.release_model(model).unwrap();
    old.shutdown().unwrap();
}

/// `cere_model_prefix_capture_stats` (prefix-token design note PT0a, appended after `cere_model_prefix_stats`): a
/// fresh model with prefix state slots reads every site and shortfall counter 0 with the header the driver set; a
/// model without a prefix cache is UNSUPPORTED; a library from before the export is `Missing`.
#[test]
fn prefix_capture_stats_reads_zero_counters_and_is_optional() {
    let d = toy_driver(1 << 30);
    assert!(d.library().api().model_prefix_capture_stats.is_some());
    let spec = ModelSpec {
        prefix_state_slots: 4,
        ..fixture_spec(4096)
    };
    let (model, _) = sealed_fixture_model(&d, &spec);
    let sites = d.prefix_capture_stats(&model).unwrap();
    assert_eq!(
        sites,
        ffi::PrefixCaptureStats {
            struct_size: size_of::<ffi::PrefixCaptureStats>() as u32,
            abi_version: ffi::ABI_VERSION,
            ..ffi::PrefixCaptureStats::default()
        }
    );
    assert_eq!(
        d.prefix_capture_stats(&model).unwrap(),
        sites,
        "a read changes nothing"
    );
    d.release_model(model).unwrap();
    let (off, _) = sealed_fixture_model(&d, &fixture_spec(4096));
    assert_eq!(
        d.prefix_capture_stats(&off),
        Err(DriverError::Status {
            call: "cere_model_prefix_capture_stats",
            status: StatusError::Unsupported,
        })
    );
    d.release_model(off).unwrap();
    d.shutdown().unwrap();

    let old = Driver::start(
        Arc::new(Library::open(&toy_library_prestaging(), None).unwrap()),
        &ExecutorConfig {
            device_id: "toy".into(),
            worker_epoch: 7,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap();
    assert!(old.library().api().model_prefix_capture_stats.is_none());
    let (model, _) = sealed_fixture_model(&old, &spec);
    let err = old.prefix_capture_stats(&model).unwrap_err();
    assert_eq!(err, DriverError::Missing("cere_model_prefix_capture_stats"));
    assert_eq!(
        err.to_string(),
        "the library predates cere_model_prefix_capture_stats"
    );
    old.release_model(model).unwrap();
    old.shutdown().unwrap();
}

/// MixedPD (mixed-pd design note): `cere_executor_set_prefill_slice` takes 0..=10000 ms (0 = off) and refuses
/// anything above as ARGUMENT; a library predating it (the CERE_TOY_PRE_STAGING toy) has no symbol.
#[test]
fn set_prefill_slice_is_bounded_and_optional() {
    let d = toy_driver(1 << 30);
    assert!(d.library().api().executor_set_prefill_slice.is_some());
    for ok in [0, 1, 200, ffi::MAX_PREFILL_SLICE_MS] {
        d.set_prefill_slice(ok)
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for bad in [ffi::MAX_PREFILL_SLICE_MS + 1, u32::MAX] {
        assert_eq!(
            d.set_prefill_slice(bad),
            Err(DriverError::Status {
                call: "cere_executor_set_prefill_slice",
                status: StatusError::Argument,
            }),
            "{bad}"
        );
    }
    d.shutdown().unwrap();

    let old = Driver::start(
        Arc::new(Library::open(&toy_library_prestaging(), None).unwrap()),
        &ExecutorConfig {
            device_id: "toy".into(),
            worker_epoch: 7,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap();
    assert!(old.library().api().executor_set_prefill_slice.is_none());
    let err = old.set_prefill_slice(200).unwrap_err();
    assert_eq!(err, DriverError::Missing("cere_executor_set_prefill_slice"));
    assert_eq!(
        err.to_string(),
        "the library predates cere_executor_set_prefill_slice"
    );
    old.shutdown().unwrap();
}

/// S1 G4 (beat-sglang design note): `cere_executor_set_decode_share` takes 0..=1000 percent (0 = off) and refuses
/// anything above as ARGUMENT; a library predating it (the CERE_TOY_PRE_STAGING toy) has no symbol.
#[test]
fn set_decode_share_is_bounded_and_optional() {
    let d = toy_driver(1 << 30);
    assert!(d.library().api().executor_set_decode_share.is_some());
    for ok in [0, 1, 100, ffi::MAX_DECODE_SHARE_PCT] {
        d.set_decode_share(ok)
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for bad in [ffi::MAX_DECODE_SHARE_PCT + 1, u32::MAX] {
        assert_eq!(
            d.set_decode_share(bad),
            Err(DriverError::Status {
                call: "cere_executor_set_decode_share",
                status: StatusError::Argument,
            }),
            "{bad}"
        );
    }
    d.shutdown().unwrap();

    let old = Driver::start(
        Arc::new(Library::open(&toy_library_prestaging(), None).unwrap()),
        &ExecutorConfig {
            device_id: "toy".into(),
            worker_epoch: 7,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap();
    assert!(old.library().api().executor_set_decode_share.is_none());
    let err = old.set_decode_share(100).unwrap_err();
    assert_eq!(err, DriverError::Missing("cere_executor_set_decode_share"));
    assert_eq!(
        err.to_string(),
        "the library predates cere_executor_set_decode_share"
    );
    old.shutdown().unwrap();
}

/// BC4 (b) (batch-concurrency design note): `cere_executor_set_prefill_age` takes 1..=1024 overtaking prefill calls
/// and refuses 0 or anything above as ARGUMENT; every toy, the CERE_TOY_PRE_STAGING one included, exports it.
#[test]
fn set_prefill_age_is_bounded() {
    let d = toy_driver(1 << 30);
    assert!(d.library().api().executor_set_prefill_age.is_some());
    for ok in [
        1,
        ffi::DEFAULT_PREFILL_AGE_CALLS,
        ffi::MAX_PREFILL_AGE_CALLS,
    ] {
        d.set_prefill_age(ok)
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for bad in [0, ffi::MAX_PREFILL_AGE_CALLS + 1, u32::MAX] {
        assert_eq!(
            d.set_prefill_age(bad),
            Err(DriverError::Status {
                call: "cere_executor_set_prefill_age",
                status: StatusError::Argument,
            }),
            "{bad}"
        );
    }
    d.shutdown().unwrap();

    let old = Library::open(&toy_library_prestaging(), None).unwrap();
    assert!(old.api().executor_set_prefill_age.is_some());
}

/// BC1 leftover (batch-concurrency design note "Prefill selection signature"): the sealed toy model's
/// `cere_model_prefill_signature` is the SHA-256 of the format v1 text; the toy's `prefill_sig=2` (one changed
/// "pinned bucket") gives another digest, so a pinned digest stops matching.
#[test]
fn prefill_signature_is_the_digest_of_the_selection_text() {
    for (device, want) in [
        (
            "toy",
            "63c07240fed82246042181ae7d6429d75511153a0a8fd50089fa704873ad7a7f",
        ),
        (
            "toy,prefill_sig=2",
            "dbdbf2679106c5b7f45bb9e298d02a4c3224fedd491e2dd35ed7fe3f557ff086",
        ),
    ] {
        let d = toy_driver_on(device, 1 << 30);
        assert!(d.library().api().model_prefill_signature.is_some());
        let (model, _) = sealed_fixture_model(&d, &fixture_spec(4096));
        let signature = d
            .prefill_signature(&model)
            .unwrap()
            .map(|s| cere_compiler::hex(&s));
        assert_eq!(signature.as_deref(), Some(want), "{device}");
        d.release_model(model).unwrap();
        d.shutdown().unwrap();
    }
}

fn fixture_spec(max_context_tokens: u64) -> ModelSpec {
    fixture_spec_with_drafts(max_context_tokens, 0)
}

/// The fixture's `cere_qwen4_exp_config` (the fixture is family qwen4_exp).
fn fixture_qwen_config() -> ffi::Qwen4ExpConfig {
    *fixture_spec(64)
        .config
        .qwen4_exp()
        .expect("the fixture is family qwen4_exp")
}

/// The fixture model's desc with `speculative_draft_tokens` (m29 design note D11: 4 = MTP drafts on DEVICE plus
/// the artifact's token map).
fn fixture_spec_with_drafts(max_context_tokens: u64, speculative_draft_tokens: u32) -> ModelSpec {
    let (dir, sha) = fixture_artifact();
    let art = artifact::open(&dir, &sha, Mode::Headers).unwrap();
    model_spec(
        &art,
        &dir,
        &PlanOptions {
            max_sessions: 1,
            max_context_tokens,
            snapshot_host_bytes: 0,
            kv_pool_tokens: 0,
            prefix_state_slots: 0,
            speculative_draft_tokens,
        },
    )
    .unwrap()
}

/// Creates, fills (zeros, through direct puts) and seals the fixture model.
fn sealed_fixture_model(
    d: &Driver,
    spec: &ModelSpec,
) -> (
    cere_runtime::native::driver::Model,
    ffi::ExecutionCapabilities,
) {
    let mut model = d.create_model(spec).unwrap();
    {
        let putter = d.putter(&model);
        for (i, t) in spec.tensors.iter().enumerate() {
            if t.placement != ffi::PLACEMENT_DISCARD {
                putter
                    .put(i as u64, 0, &vec![0u8; t.bytes as usize])
                    .unwrap();
            }
        }
    }
    let caps = d.seal(&mut model).unwrap();
    (model, caps)
}

#[test]
fn toy_round_trip_create_model_session_submit_poll_cancel() {
    let d = toy_driver(1 << 30);
    let spec = fixture_spec(4096);
    let (model, caps) = sealed_fixture_model(&d, &spec);
    assert_eq!(caps.execution_mode, ffi::EXECUTION_MODE_NATIVE);
    assert_eq!(caps.worker_epoch, 7);
    let ops = ffi::OPS_PREFILL | ffi::OPS_DECODE;
    assert_eq!(
        caps.operations & ops,
        ops,
        "operations {:#x}",
        caps.operations
    );
    assert_eq!(d.capabilities(&model).unwrap(), caps);

    // PREFILL then DECODE on one session: the toy sequence continues across items.
    let session = d.create_session(&model, 4096).unwrap();
    let prompt = vec![1u32, 2, 3];
    let r = d
        .submit(
            &session,
            WorkRequest::prefill(prompt.clone(), 5).with_scores(),
        )
        .unwrap()
        .wait()
        .unwrap();
    let want = toy_tokens(3, 3, 8, FIXTURE_VOCAB);
    assert_eq!(r.tokens, want[..5]);
    assert_eq!(r.finish, Finish::Length);
    assert_eq!(r.status, Ok(()));
    assert_eq!(r.committed_tokens, 5);
    assert_eq!(r.scores.len(), 5);
    assert!(r.scores.iter().zip(&r.tokens).all(|(s, t)| s.id == *t));
    let r = d
        .submit(&session, WorkRequest::decode(3))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.tokens, want[5..]);
    assert!(r.scores.is_empty(), "scores only when asked for");
    assert!(r.session_revision > 0);
    d.release_session(session).unwrap();

    // A per-item stop token is committed, counted and ends the work.
    let session = d.create_session(&model, 4096).unwrap();
    let stop = want[1];
    let r = d
        .submit(
            &session,
            WorkRequest::prefill(prompt.clone(), 5).with_stop_tokens(vec![stop]),
        )
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.tokens, want[..2]);
    assert_eq!(r.finish, Finish::StopToken);
    d.release_session(session).unwrap();

    // The terminal poll released the ticket: a cancel through the driver names an unknown ticket. (The CANCELLED
    // fence itself is `toy_cancel_is_a_token_boundary_fence`: the driver thread polls continuously, so a cancel
    // sent after submit can lose the race against a job that finishes first.)
    assert!(matches!(
        d.cancel(r.ticket),
        Err(DriverError::Status {
            call: "cere_executor_cancel",
            status: StatusError::Argument,
        })
    ));
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// m29 design note C6 (I3, I5): a model without a draft reports no speculation (`native_speculative_batch` 0, no
/// VERIFY, 0 draft tokens) and every completion reads `verify_steps` and `accepted_drafts` 0.
#[test]
fn spec_off_model_reports_no_speculation_and_zero_counters() {
    let d = toy_driver(1 << 30);
    let spec = fixture_spec(4096);
    assert_eq!(spec.speculative_draft_tokens, 0);
    assert!(spec.draft_token_map.is_empty());
    let (model, caps) = sealed_fixture_model(&d, &spec);
    assert_eq!(
        (
            caps.native_speculative_batch,
            caps.speculative_draft_tokens,
            caps.operations & ffi::OPS_VERIFY
        ),
        (0, 0, 0)
    );
    let session = d.create_session(&model, 4096).unwrap();
    for work in [
        WorkRequest::prefill(vec![1, 2, 3], 5),
        WorkRequest::decode(3),
    ] {
        let r = d.submit(&session, work).unwrap().wait().unwrap();
        assert_eq!((r.verify_steps, r.accepted_drafts), (0, 0));
    }
    d.release_session(session).unwrap();
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// m29 design note C6 (I3, I5) over the HOST track's toy speculation (`toy` drafts with 4 and keeps its target
/// rule, so tokens equal the spec-off run; `,draft_miss=K` commits K tokens per round, absent = 4): the sealed model
/// reports `native_speculative_batch` 1, VERIFY and 4 draft tokens, and the completion counters are read through
/// `JobResult`: a PREFILL of N tokens runs ceil((N-1)/K) rounds (its first token is not a round), a DECODE of N
/// ceil(N/K), accepting N' - rounds drafts. Pending the HOST merge: a toy without draft support refuses the drafts-on
/// desc with UNSUPPORTED, reported on stderr.
#[test]
fn drafts_on_model_reports_speculation_and_reads_the_completion_counters() {
    let plain = {
        let d = toy_driver(1 << 30);
        let (model, _) = sealed_fixture_model(&d, &fixture_spec(4096));
        let session = d.create_session(&model, 4096).unwrap();
        let mut tokens = d
            .submit(&session, WorkRequest::prefill(vec![1, 2, 3], 9))
            .unwrap()
            .wait()
            .unwrap()
            .tokens;
        tokens.extend(
            d.submit(&session, WorkRequest::decode(7))
                .unwrap()
                .wait()
                .unwrap()
                .tokens,
        );
        d.release_session(session).unwrap();
        d.release_model(model).unwrap();
        d.shutdown().unwrap();
        tokens
    };
    let spec = fixture_spec_with_drafts(4096, 4);
    assert_eq!(spec.speculative_draft_tokens, 4);
    assert!(!spec.draft_token_map.is_empty());
    for (device_id, k) in [
        ("toy", 4u64),
        ("toy,draft_miss=1", 1),
        ("toy,draft_miss=3", 3),
    ] {
        let d = toy_driver_on(device_id, 1 << 30);
        let model = d
            .create_model(&spec)
            .unwrap_or_else(|e| panic!("{device_id}: {e}"));
        d.release_model(model).unwrap();
        let (model, caps) = sealed_fixture_model(&d, &spec);
        assert_eq!(caps.native_speculative_batch, 1, "{device_id}");
        assert_eq!(caps.speculative_draft_tokens, 4, "{device_id}");
        assert_ne!(caps.operations & ffi::OPS_VERIFY, 0, "{device_id}");
        let session = d.create_session(&model, 4096).unwrap();
        let prefill = d
            .submit(&session, WorkRequest::prefill(vec![1, 2, 3], 9))
            .unwrap()
            .wait()
            .unwrap();
        let rounds = 8u64.div_ceil(k);
        assert_eq!(
            (prefill.verify_steps, prefill.accepted_drafts),
            (rounds, 8 - rounds),
            "{device_id}"
        );
        let decode = d
            .submit(&session, WorkRequest::decode(7))
            .unwrap()
            .wait()
            .unwrap();
        let rounds = 7u64.div_ceil(k);
        assert_eq!(
            (decode.verify_steps, decode.accepted_drafts),
            (rounds, 7 - rounds),
            "{device_id}"
        );
        let mut tokens = prefill.tokens;
        tokens.extend(decode.tokens);
        assert_eq!(tokens, plain, "{device_id}");
        d.release_session(session).unwrap();
        d.release_model(model).unwrap();
        d.shutdown().unwrap();
    }
}

/// I11 (m27): `WorkRequest::sampling` reaches the executor (its `logit_bias` included) and
/// `WorkRequest::top_logprobs` leases `out_logprobs`, returned one entry per committed token in
/// `JobResult::logprobs`; the toy reports `native_sampler` 1.
#[test]
fn toy_sampling_and_logprobs_round_trip() {
    let d = toy_driver(1 << 30);
    let spec = fixture_spec(4096);
    let (model, caps) = sealed_fixture_model(&d, &spec);
    assert_eq!(caps.native_sampler, 1);
    let prompt = vec![1u32, 2, 3];
    let want = toy_tokens(3, 3, 6, FIXTURE_VOCAB);
    let run = |work: WorkRequest| {
        let session = d.create_session(&model, 4096).unwrap();
        let r = d.submit(&session, work).unwrap().wait().unwrap();
        d.release_session(session).unwrap();
        assert_eq!(r.status, Ok(()));
        r
    };

    // Greedy with logprobs (null sampling): the ids are the greedy ones; the toy's logits are 8 - j at
    // next + j (j < 8), so the committed token's logprob is -ln(sum e^-j).
    let r = run(WorkRequest::prefill(prompt.clone(), 6).with_logprobs(3));
    assert_eq!(r.tokens, want);
    assert_eq!(r.logprobs.len(), 6);
    let lse = (0..8).map(|j| (-(j as f64)).exp()).sum::<f64>().ln();
    for (lp, &id) in r.logprobs.iter().zip(&r.tokens) {
        assert_eq!((lp.id, lp.top_count, lp.top[0].id), (id, 3, id));
        assert!((f64::from(lp.logprob) + lse).abs() < 1e-5, "{lp:?}");
        assert_eq!(lp.top[1].id, (id + 1) % FIXTURE_VOCAB);
    }
    assert!(run(WorkRequest::prefill(prompt.clone(), 2))
        .logprobs
        .is_empty());

    // Progress carries each committed token's logprobs with it, the terminal poll's included (C5: SSE).
    let session = d.create_session(&model, 4096).unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let work = WorkRequest::prefill(prompt.clone(), 6).with_logprobs(3);
    let r = d
        .submit_with_progress(&session, work, tx)
        .unwrap()
        .wait()
        .unwrap();
    d.release_session(session).unwrap();
    let mut streamed = Committed::default();
    while let Some(more) = rx.blocking_recv() {
        assert_eq!(more.logprobs.len(), more.tokens.len());
        streamed.tokens.extend(more.tokens);
        streamed.logprobs.extend(more.logprobs);
    }
    assert_eq!((streamed.tokens, streamed.logprobs), (r.tokens, r.logprobs));

    // Sampled: a +100 `logit_bias` on the toy's least likely candidate makes it the first token.
    let sampled = |seed: u64, bias: Vec<ffi::LogitBias>| Sampling {
        temperature: 1.0,
        top_p: 1.0,
        min_p: 0.0,
        top_k: FIXTURE_VOCAB,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        seed,
        logit_bias: bias,
    };
    let biased = (want[0] + 7) % FIXTURE_VOCAB;
    let r = run(WorkRequest::prefill(prompt.clone(), 3)
        .with_sampling(sampled(
            42,
            vec![ffi::LogitBias {
                id: biased,
                bias: 100.0,
            }],
        ))
        .with_logprobs(0));
    assert_eq!(r.tokens[0], biased);
    assert_eq!((r.logprobs[0].id, r.logprobs[0].top_count), (biased, 0));
    assert!(r.logprobs[0].logprob > -1e-6);
    // The seed decides the draw: equal seeds give equal tokens on fresh sessions, another seed differs.
    let tokens = |seed| {
        run(WorkRequest::prefill(prompt.clone(), 12).with_sampling(sampled(seed, Vec::new())))
            .tokens
    };
    assert_eq!(tokens(42), tokens(42));
    assert_ne!(tokens(42), tokens(43));
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// The toy commits one token per `cere_executor_poll`, so a caller that polls by itself (no driver thread) controls
/// exactly how far the work gets before the cancel.
struct RawToy {
    lib: Library,
    executor: *mut ffi::RawExecutor,
}

impl RawToy {
    const EPOCH: u64 = 3;

    fn new() -> RawToy {
        let lib = Library::open(&toy_library(), None).unwrap();
        let device = std::ffi::CString::new("toy").unwrap();
        let options = ffi::ExecutorOptions {
            struct_size: size_of::<ffi::ExecutorOptions>() as u32,
            abi_version: ffi::ABI_VERSION,
            device_id: device.as_ptr(),
            worker_epoch: Self::EPOCH,
            memory_budget_bytes: 1 << 20,
        };
        let mut executor = std::ptr::null_mut();
        // SAFETY: valid options (the device id is copied) and out pointer.
        let rc = unsafe { (lib.api().executor_create)(&mut executor, &options) };
        assert_eq!(StatusError::check(rc), Ok(()));
        RawToy { lib, executor }
    }

    fn api(&self) -> &ffi::Api {
        self.lib.api()
    }

    /// A sealed one-tensor model with the fixture config (the toy's smallest accepted desc); its profile generation.
    fn sealed_model(&self, config: &ffi::Qwen4ExpConfig) -> (*mut ffi::RawModel, u64) {
        let (model, caps) = self.pool_model(config, 1, 0).unwrap();
        (model, caps.profile_generation)
    }

    /// `sealed_model` with `max_sessions` and a desc `kv_pool_tokens` (m25 design note D4): the sealed model and
    /// its capabilities, or the `cere_model_create` status (nothing created).
    fn pool_model(
        &self,
        config: &ffi::Qwen4ExpConfig,
        max_sessions: u64,
        kv_pool_tokens: u64,
    ) -> Result<(*mut ffi::RawModel, ffi::ExecutionCapabilities), ffi::Status> {
        let name = std::ffi::CString::new("model.embed_tokens.weight").unwrap();
        let mut shape = [0u64; ffi::TENSOR_MAX_DIMS];
        shape[..2].copy_from_slice(&[4, 8]);
        let tensor = ffi::TensorDesc {
            struct_size: size_of::<ffi::TensorDesc>() as u32,
            abi_version: ffi::ABI_VERSION,
            name: name.as_ptr(),
            dtype: ffi::DTYPE_BF16,
            placement: ffi::PLACEMENT_DEVICE,
            ndim: 2,
            reserved: 0,
            shape,
            bytes: 64,
        };
        let desc = ffi::ModelDesc {
            struct_size: size_of::<ffi::ModelDesc>() as u32,
            abi_version: ffi::ABI_VERSION,
            model_kind: ffi::MODEL_QWEN4_EXP_NVFP4,
            speculative_draft_tokens: 0,
            config,
            tensors: &tensor,
            tensor_count: 1,
            max_sessions,
            max_context_tokens: 64,
            stop_token_ids: std::ptr::null(),
            stop_token_count: 0,
            kv_pool_tokens,
            draft_token_map: std::ptr::null(),
            draft_token_map_count: 0,
            snapshot_host_bytes: 0,
            prefix_state_slots: 0,
            vision_config: std::ptr::null(),
            selection: std::ptr::null(),
            family: ffi::FAMILY_QWEN4_EXP,
            calibrate_windows: ffi::CALIBRATE_ALL,
        };
        let api = self.api();
        let mut model = std::ptr::null_mut();
        let bytes = [0u8; 64];
        let mut caps = ffi::ExecutionCapabilities {
            struct_size: size_of::<ffi::ExecutionCapabilities>() as u32,
            abi_version: ffi::ABI_VERSION,
            ..Default::default()
        };
        // SAFETY: the desc and everything it points to outlive the calls; `bytes` is valid for 64 bytes; the model
        // and capabilities pointers come from / point to live objects.
        unsafe {
            let rc = (api.model_create)(self.executor, &desc, &mut model);
            if rc != ffi::OK {
                assert!(model.is_null());
                return Err(rc);
            }
            assert_eq!(
                (api.model_put)(self.executor, model, 0, 0, bytes.as_ptr().cast(), 64),
                ffi::OK
            );
            assert_eq!((api.model_seal)(self.executor, model), ffi::OK);
            assert_eq!(
                (api.executor_capabilities)(self.executor, model, &mut caps),
                ffi::OK
            );
        }
        Ok((model, caps))
    }

    fn poll(&self, ticket: ffi::CompletionTicket) -> (ffi::Status, ffi::Completion) {
        let mut c = ffi::Completion {
            struct_size: size_of::<ffi::Completion>() as u32,
            abi_version: ffi::ABI_VERSION,
            ..Default::default()
        };
        // SAFETY: a live executor and a completion with its header set.
        let rc = unsafe { (self.api().executor_poll)(self.executor, ticket, &mut c) };
        (rc, c)
    }

    fn cancel(&self, ticket: ffi::CompletionTicket) -> ffi::Status {
        // SAFETY: a live executor; any ticket value is allowed.
        unsafe { (self.api().executor_cancel)(self.executor, ticket) }
    }
}

impl Drop for RawToy {
    fn drop(&mut self) {
        // SAFETY: the executor came from `executor_create`; the test released its models and sessions first.
        let rc = unsafe { (self.api().executor_destroy)(self.executor) };
        if !std::thread::panicking() {
            assert_eq!(rc, ffi::OK);
        }
    }
}

#[test]
fn toy_cancel_is_a_token_boundary_fence() {
    let toy = RawToy::new();
    let api = *toy.api();
    let (model, generation) = toy.sealed_model(&fixture_qwen_config());
    let mut session = std::ptr::null_mut();
    // SAFETY: a live executor and sealed model of it.
    let rc = unsafe { (api.session_create)(toy.executor, model, 64, &mut session) };
    assert_eq!(rc, ffi::OK);

    const MAX_NEW: u64 = 20;
    let prompt = [5u32];
    let mut out = vec![0u32; MAX_NEW as usize];
    let item = ffi::Work {
        struct_size: size_of::<ffi::Work>() as u32,
        abi_version: ffi::ABI_VERSION,
        operation: ffi::OP_PREFILL,
        flags: 0,
        expected_worker_epoch: RawToy::EPOCH,
        expected_session_revision: 0,
        session,
        tokens: prompt.as_ptr(),
        token_count: 1,
        max_new_tokens: MAX_NEW,
        reserved_peak_bytes: 0,
        out_tokens: out.as_mut_ptr(),
        out_capacity: MAX_NEW,
        out_scores: std::ptr::null_mut(),
        stop_token_ids: std::ptr::null(),
        stop_token_count: 0,
        sampling: std::ptr::null(),
        out_logprobs: std::ptr::null_mut(),
        top_logprobs: 0,
        capture_position: 0,
        vocab_mask: std::ptr::null(),
        vocab_mask_words: 0,
        images: std::ptr::null(),
        image_count: 0,
    };
    let batch = ffi::Batch {
        struct_size: size_of::<ffi::Batch>() as u32,
        abi_version: ffi::ABI_VERSION,
        expected_worker_epoch: RawToy::EPOCH,
        capability_profile_generation: generation,
        items: &item,
        item_count: 1,
        reserved_peak_bytes: 0,
    };
    let mut ticket = ffi::CompletionTicket::default();
    // SAFETY: the batch and its lists outlive the call; `out` stays in place until the terminal poll.
    let rc = unsafe { (api.executor_submit)(toy.executor, &batch, &mut ticket) };
    assert_eq!(rc, ffi::OK);

    // Three non-terminal polls commit three tokens.
    for i in 1..=3u64 {
        let (rc, c) = toy.poll(ticket);
        assert_eq!((rc, c.terminal, c.committed_tokens), (ffi::PENDING, 0, i));
    }
    // Cancel is intent, idempotent until the terminal poll.
    assert_eq!(toy.cancel(ticket), ffi::OK);
    assert_eq!(toy.cancel(ticket), ffi::OK);
    // The next poll is the terminal CANCELLED fence with the committed tokens kept.
    let (rc, c) = toy.poll(ticket);
    assert_eq!(rc, ffi::OK);
    assert_eq!(c.terminal, 1);
    assert_eq!(c.finish, ffi::FINISH_CANCELLED);
    assert_eq!(c.status, ffi::OK);
    assert_eq!(c.committed_tokens, 3);
    assert_eq!(c.session_revision, 1);
    assert_eq!(out[..3], toy_tokens(5, 1, 3, FIXTURE_VOCAB));
    // The terminal poll released the ticket.
    assert_eq!(toy.cancel(ticket), ffi::ERR_ARGUMENT);
    assert_eq!(toy.poll(ticket).0, ffi::ERR_ARGUMENT);

    // SAFETY: no ticket references the session or the model any more.
    unsafe {
        assert_eq!((api.session_release)(toy.executor, session), ffi::OK);
        assert_eq!((api.model_release)(toy.executor, model), ffi::OK);
    }
}

/// m25 design note C1 (D4; ledger D 72, C 72/80): a non-zero desc `kv_pool_tokens` is accepted (it was
/// UNSUPPORTED before M2.5) and bounds the sessions; 0 is the derived pool, max_sessions x round64(context); a pool
/// that is not a multiple of 64 or is below the context is ARGUMENT, one above the support bound UNSUPPORTED; the
/// capabilities report the pool and the 64-token page.
#[test]
fn kv_pool_tokens_is_accepted_reported_and_bounds_sessions() {
    let toy = RawToy::new();
    let api = *toy.api();
    let config = fixture_qwen_config();

    let (derived, caps) = toy.pool_model(&config, 2, 0).unwrap();
    assert_eq!((caps.kv_pool_tokens, caps.kv_page_tokens), (128, 64));
    for (pool, status) in [(100, ffi::ERR_ARGUMENT), (1_048_640, ffi::ERR_UNSUPPORTED)] {
        assert_eq!(
            toy.pool_model(&config, 2, pool).map(|_| ()),
            Err(status),
            "pool {pool}"
        );
    }

    // One page: the second session finds no free page until the first is released.
    let (model, caps) = toy.pool_model(&config, 2, 64).unwrap();
    assert_eq!((caps.kv_pool_tokens, caps.kv_page_tokens), (64, 64));
    let (mut first, mut second) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: a live executor, its sealed models and sessions of them.
    unsafe {
        assert_eq!(
            (api.session_create)(toy.executor, model, 64, &mut first),
            ffi::OK
        );
        assert_eq!(
            (api.session_create)(toy.executor, model, 64, &mut second),
            ffi::ERR_CAPACITY
        );
        assert!(second.is_null());
        assert_eq!((api.session_release)(toy.executor, first), ffi::OK);
        assert_eq!(
            (api.session_create)(toy.executor, model, 64, &mut second),
            ffi::OK
        );
        assert_eq!((api.session_release)(toy.executor, second), ffi::OK);
        assert_eq!((api.model_release)(toy.executor, model), ffi::OK);
        assert_eq!((api.model_release)(toy.executor, derived), ffi::OK);
    }
}

#[test]
fn shutdown_drains_in_flight_work_and_releases_everything() {
    let d = toy_driver(1 << 30);
    let spec = fixture_spec(4096);
    let (model, _) = sealed_fixture_model(&d, &spec);
    let session = d.create_session(&model, 4096).unwrap();
    let job = d
        .submit(&session, WorkRequest::prefill(vec![9], 4000))
        .unwrap();
    // Neither the session nor the model is released by the caller: shutdown cancels, drains and releases them.
    d.shutdown().unwrap();
    let r = job.wait().unwrap();
    assert!(
        matches!(r.finish, Finish::Cancelled | Finish::Length),
        "{:?}",
        r.finish
    );
}

/// R5's streaming path: a pending job reports its newly committed tokens in order, and a cancel sent while a DECODE
/// is pending ends it at a token boundary with `CANCELLED`; the terminal poll releases the session. The toy holds
/// the session after `HOLD` generated tokens (`hold_after`), so the cancel lands mid-DECODE on every run.
#[test]
fn progress_streams_committed_tokens_and_cancel_lands_mid_decode() {
    const CONTEXT: u64 = 4096;
    const DECODE: u64 = 100;
    const HOLD: usize = 5;
    let d = toy_driver_on(&format!("toy,hold_after={HOLD}"), 1 << 30);
    let spec = fixture_spec(CONTEXT);
    let (model, _) = sealed_fixture_model(&d, &spec);
    let session = d.create_session(&model, CONTEXT).unwrap();
    let first = d
        .submit(&session, WorkRequest::prefill(vec![1, 2, 3], 1))
        .unwrap()
        .wait()
        .unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let job = d
        .submit_with_progress(&session, WorkRequest::decode(DECODE), tx)
        .unwrap();
    let mut streamed = Vec::new();
    while streamed.len() < HOLD - 1 {
        let more = rx
            .blocking_recv()
            .expect("progress while the job is pending");
        assert!(!more.tokens.is_empty());
        assert!(more.logprobs.is_empty());
        streamed.extend(more.tokens);
    }
    d.cancel(job.ticket()).unwrap();
    let r = job.wait().unwrap();
    assert_eq!(r.finish, Finish::Cancelled);
    assert_eq!(r.status, Ok(()));
    assert_eq!(r.committed_tokens, HOLD as u64 - 1);
    // Progress ends with the job (the sender is dropped at the terminal poll) and is a prefix of its tokens.
    while let Some(more) = rx.blocking_recv() {
        streamed.extend(more.tokens);
    }
    assert!(r.tokens.starts_with(&streamed));
    let want = toy_tokens(3, 3, 1 + r.tokens.len(), FIXTURE_VOCAB);
    assert_eq!(first.tokens, want[..1]);
    assert_eq!(r.tokens, want[1..]);
    d.release_session(session).unwrap();
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// The impl and params `selection` lists for `slot` (every slot is listed once by `cere_model_selection`).
fn slot(selection: &ffi::SlotSelection, slot: u32) -> ffi::SlotChoice {
    let listed: Vec<&ffi::SlotChoice> = selection.choices[..selection.count as usize]
        .iter()
        .filter(|c| c.slot == slot)
        .collect();
    assert_eq!(listed.len(), 1, "slot {slot} in {selection:?}");
    *listed[0]
}

/// m23 design note PI-03/PI-05: `cere_model_desc.selection` reaches the toy and `cere_model_selection` reads it back; the
/// model-spec default is every lever's default (MoE `CERE_LEVER_MOE_P3B`, the toy's backend). A MoE lever other than
/// the library's backend, or a `draft_tokens` cap on a model without drafts, is refused at create (UNSUPPORTED).
#[test]
fn desc_levers_reach_the_executor_and_read_back() {
    let d = toy_driver(1 << 30);
    let mut spec = fixture_spec(4096);
    assert_eq!(spec.levers, LeverSet::default());
    let (model, _) = sealed_fixture_model(&d, &spec);
    let read = d.levers(&model).unwrap();
    assert_eq!(read, LeverSet::default().ffi(Family::Qwen4Exp));
    assert_eq!(slot(&read, ffi::SLOT_MOE_GEMM).r#impl, ffi::LEVER_MOE_P3B);
    d.release_model(model).unwrap();
    spec.levers.k2_gemm = K2Gemm::SmallM;
    spec.levers.prefill_attention = PrefillAttention::P4;
    let (model, _) = sealed_fixture_model(&d, &spec);
    let read = d.levers(&model).unwrap();
    assert_eq!(
        (
            slot(&read, ffi::SLOT_K2_GEMM).r#impl,
            slot(&read, ffi::SLOT_PREFILL_ATTENTION).r#impl
        ),
        (ffi::LEVER_K2_SMALL_M, ffi::LEVER_PREFILL_P4)
    );
    assert_eq!(
        LeverSet::from_ffi(&read, &spec.levers).unwrap(),
        spec.levers
    );
    d.release_model(model).unwrap();
    for (levers, what) in [
        (
            LeverSet {
                moe_gemm: MoeGemm::K7,
                ..LeverSet::default()
            },
            "k7 on a p3b library",
        ),
        (
            LeverSet {
                draft_tokens: DraftTokens::Three,
                ..LeverSet::default()
            },
            "draft_tokens without drafts",
        ),
    ] {
        let err = d
            .create_model(&ModelSpec {
                levers,
                ..spec.clone()
            })
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(
                err,
                DriverError::Status {
                    call: "cere_model_create",
                    status: StatusError::Unsupported
                }
            ),
            "{what}: {err:?}"
        );
    }
    d.shutdown().unwrap();
}

/// PI-04: `cere_model_set_selection` swaps the sealed model's levers between requests (read back, and the output of
/// the toy is unchanged: its tokens do not depend on the kernel levers); with a ticket in flight it is
/// `CERE_ERR_BUSY` and the active set is unchanged. A `draft_tokens` cap of 3 on the drafts model verifies at most
/// 2 drafts per round and commits the plain tokens (greedy speculation equals plain decoding).
#[test]
fn set_levers_swaps_between_requests_and_is_busy_in_flight() {
    const HOLD: usize = 5;
    let d = toy_driver_on(&format!("toy,hold_after={HOLD}"), 1 << 30);
    let spec = fixture_spec(4096);
    let (model, _) = sealed_fixture_model(&d, &spec);
    let swapped = LeverSet {
        k2_gemm: K2Gemm::SmallM,
        prefill_attention: PrefillAttention::P4,
        ..LeverSet::default()
    };
    d.set_levers(&model, &swapped).unwrap();
    assert_eq!(d.levers(&model).unwrap(), swapped.ffi(Family::Qwen4Exp));
    let session = d.create_session(&model, 4096).unwrap();
    let job = d
        .submit(&session, WorkRequest::prefill(vec![1, 2, 3], 100))
        .unwrap();
    let err = d.set_levers(&model, &LeverSet::default()).unwrap_err();
    assert_eq!(err.status(), Some(StatusError::Busy), "{err:?}");
    assert_eq!(d.levers(&model).unwrap(), swapped.ffi(Family::Qwen4Exp));
    d.cancel(job.ticket()).unwrap();
    let r = job.wait().unwrap();
    assert_eq!(r.finish, Finish::Cancelled);
    d.set_levers(&model, &LeverSet::default()).unwrap();
    assert_eq!(
        d.levers(&model).unwrap(),
        LeverSet::default().ffi(Family::Qwen4Exp)
    );
    d.release_session(session).unwrap();
    d.release_model(model).unwrap();
    d.shutdown().unwrap();

    let plain = toy_tokens(3, 3, 9, FIXTURE_VOCAB);
    let d = toy_driver(1 << 30);
    let spec = ModelSpec {
        levers: LeverSet {
            draft_tokens: DraftTokens::Three,
            ..LeverSet::default()
        },
        ..fixture_spec_with_drafts(4096, 4)
    };
    let (model, _) = sealed_fixture_model(&d, &spec);
    assert_eq!(
        slot(&d.levers(&model).unwrap(), ffi::SLOT_DRAFT_TOKENS).params,
        [3, 0, 0, 0]
    );
    let session = d.create_session(&model, 4096).unwrap();
    let r = d
        .submit(&session, WorkRequest::prefill(vec![1, 2, 3], 9))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.tokens, plain);
    // 8 speculated tokens, each round the committed token plus at most 2 drafts.
    let rounds = 8u64.div_ceil(3);
    assert_eq!((r.verify_steps, r.accepted_drafts), (rounds, 8 - rounds));
    d.release_session(session).unwrap();
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// Fp8W: `k2_weights` reaches the toy at create and reads back; it is fixed there (INSTANTIATE converts the weights),
/// so `cere_model_set_selection` with another value is `CERE_ERR_UNSUPPORTED` and changes nothing, while the hot levers
/// still swap at the model's own `k2_weights`.
#[test]
fn k2_weights_is_fixed_at_create() {
    let d = toy_driver(1 << 30);
    let spec = ModelSpec {
        levers: LeverSet {
            k2_weights: K2Weights::Fp8,
            ..LeverSet::default()
        },
        ..fixture_spec(4096)
    };
    let (model, _) = sealed_fixture_model(&d, &spec);
    let read = d.levers(&model).unwrap();
    assert_eq!(
        slot(&read, ffi::SLOT_K2_WEIGHTS).r#impl,
        ffi::LEVER_K2_WEIGHTS_FP8
    );
    assert_eq!(
        LeverSet::from_ffi(&read, &spec.levers).unwrap(),
        spec.levers
    );
    let err = d.set_levers(&model, &LeverSet::default()).unwrap_err();
    assert_eq!(err.status(), Some(StatusError::Unsupported), "{err:?}");
    assert_eq!(d.levers(&model).unwrap(), spec.levers.ffi(Family::Qwen4Exp));
    let swapped = LeverSet {
        k2_gemm: K2Gemm::SmallM,
        ..spec.levers
    };
    d.set_levers(&model, &swapped).unwrap();
    assert_eq!(d.levers(&model).unwrap(), swapped.ffi(Family::Qwen4Exp));
    d.release_model(model).unwrap();

    let (model, _) = sealed_fixture_model(&d, &fixture_spec(4096));
    let fp8 = LeverSet {
        k2_weights: K2Weights::Fp8,
        ..LeverSet::default()
    };
    let err = d.set_levers(&model, &fp8).unwrap_err();
    assert_eq!(err.status(), Some(StatusError::Unsupported), "{err:?}");
    assert_eq!(
        d.levers(&model).unwrap(),
        LeverSet::default().ffi(Family::Qwen4Exp)
    );
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// m210 design note D17: `prefix_alt_edges`, like `prefix_end_state`, is set right after `cere_model_create`. On a
/// cache-off model the setter's UNSUPPORTED fails the create (the driver releases the model); with prefix state slots
/// both setters succeed and the model seals.
#[test]
fn prefix_alt_edges_is_set_at_create() {
    let d = toy_driver(1 << 30);
    assert!(d.library().api().model_set_prefix_alt_edges.is_some());
    let err = d
        .create_model(&ModelSpec {
            prefix_alt_edges: true,
            ..fixture_spec(4096)
        })
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::Status {
                call: "cere_model_set_prefix_alt_edges",
                status: StatusError::Unsupported
            }
        ),
        "{err:?}"
    );
    let spec = ModelSpec {
        prefix_state_slots: 4,
        prefix_end_state: true,
        prefix_alt_edges: true,
        ..fixture_spec(4096)
    };
    let (model, _) = sealed_fixture_model(&d, &spec);
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// LoadCrit LC-01: `cere_model_desc.calibrate_windows` follows the `drafter` lever (MTP for the drafters whose tickets
/// never carry work bits 5-7, ALL for the tail, long and chain drafters); it is a create-time desc field, not a slot,
/// so a chain-drafter model's selection reads back as its set and the hot swap of the slots still applies.
#[test]
fn calibrate_windows_follows_the_drafter_and_is_not_a_slot() {
    for (drafter, want) in [
        (Drafter::Mtp, ffi::CALIBRATE_MTP),
        (Drafter::Lookup, ffi::CALIBRATE_MTP),
        (Drafter::MtpLookup, ffi::CALIBRATE_MTP),
        (Drafter::MtpLookupTail, ffi::CALIBRATE_ALL),
        (Drafter::MtpLookupLong, ffi::CALIBRATE_ALL),
        (Drafter::MtpLookupChain, ffi::CALIBRATE_ALL),
    ] {
        let levers = LeverSet {
            drafter,
            ..LeverSet::default()
        };
        assert_eq!(
            levers.calibrate_windows(Family::Qwen4Exp),
            want,
            "{drafter:?}"
        );
        assert_eq!(
            levers.ffi(Family::Qwen4Exp),
            LeverSet::default().ffi(Family::Qwen4Exp),
            "{drafter:?}"
        );
    }
    let d = toy_driver(1 << 30);
    let spec = ModelSpec {
        levers: LeverSet {
            drafter: Drafter::MtpLookupChain,
            ..LeverSet::default()
        },
        ..fixture_spec(4096)
    };
    let (model, _) = sealed_fixture_model(&d, &spec);
    let read = d.levers(&model).unwrap();
    assert_eq!(read, spec.levers.ffi(Family::Qwen4Exp));
    assert_eq!(
        LeverSet::from_ffi(&read, &spec.levers).unwrap(),
        spec.levers
    );
    let swapped = LeverSet {
        k2_gemm: K2Gemm::SmallM,
        ..LeverSet::default()
    };
    d.set_levers(&model, &swapped).unwrap();
    let read = d.levers(&model).unwrap();
    assert_eq!(read, swapped.ffi(Family::Qwen4Exp));
    assert_eq!(LeverSet::from_ffi(&read, &swapped).unwrap(), swapped);
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// `progress` carries every committed token exactly once, including those the terminal poll commits (the toy
/// commits a PREFILL's token and a DECODE's last token in the poll that ends the item): the concatenation equals
/// the terminal tokens, for a PREFILL alone and for PREFILL + DECODE.
#[test]
fn progress_carries_every_committed_token_once() {
    const CONTEXT: u64 = 4096;
    let d = toy_driver(1 << 30);
    let spec = fixture_spec(CONTEXT);
    let (model, _) = sealed_fixture_model(&d, &spec);
    for decode in [0u64, 1, 2, 5] {
        let session = d.create_session(&model, CONTEXT).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut items = vec![WorkRequest::prefill(vec![1, 2, 3], 1)];
        if decode > 0 {
            items.push(WorkRequest::decode(decode));
        }
        let mut committed = Vec::new();
        for work in items {
            let r = d
                .submit_with_progress(&session, work, tx.clone())
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(r.status, Ok(()));
            committed.extend(r.tokens);
        }
        drop(tx);
        let mut streamed = Vec::new();
        while let Some(more) = rx.blocking_recv() {
            streamed.extend(more.tokens);
        }
        assert_eq!(
            committed,
            toy_tokens(3, 3, 1 + decode as usize, FIXTURE_VOCAB)
        );
        assert_eq!(streamed, committed, "decode {decode}");
        d.release_session(session).unwrap();
    }
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// The toy fixture model with a snapshot arena (M2.11 D5): ample for its small sections.
const ARENA: u64 = 64 << 20;

fn snapshot_spec(max_context_tokens: u64) -> ModelSpec {
    ModelSpec {
        snapshot_host_bytes: ARENA,
        ..fixture_spec(max_context_tokens)
    }
}

/// The whole image of `snapshot` through `Driver::read_snapshot`.
fn read_all(d: &Driver, snapshot: &Snapshot) -> Vec<u8> {
    let info = d.snapshot_info(snapshot).unwrap();
    let mut bytes = vec![0u8; info.bytes as usize];
    d.read_snapshot(snapshot, 0, &mut bytes).unwrap();
    bytes
}

/// M2.11 C4 (spec D1, D6): the driver's snapshot commands. A snapshot of an idle session reports its position,
/// two snapshots of it are byte-identical, and a fresh session restored from it continues exactly as the
/// uninterrupted run; a restored (already submitted) session refuses a second restore.
#[test]
fn driver_snapshot_restore_continues_the_uninterrupted_run() {
    const CONTEXT: u64 = 4096;
    let d = toy_driver(1 << 30);
    let spec = snapshot_spec(CONTEXT);
    let (model, caps) = sealed_fixture_model(&d, &spec);
    let ops = ffi::OPS_SNAPSHOT | ffi::OPS_RESTORE;
    assert_eq!(
        caps.operations & ops,
        ops,
        "operations {:#x}",
        caps.operations
    );
    let want = toy_tokens(3, 3, 12, FIXTURE_VOCAB);

    let session = d.create_session(&model, CONTEXT).unwrap();
    let mut run = d
        .submit(&session, WorkRequest::prefill(vec![1, 2, 3], 1))
        .unwrap()
        .wait()
        .unwrap()
        .tokens;
    run.extend(
        d.submit(&session, WorkRequest::decode(4))
            .unwrap()
            .wait()
            .unwrap()
            .tokens,
    );
    assert_eq!(run, want[..5]);

    let snapshot = d.snapshot(&session).unwrap();
    let info = d.snapshot_info(&snapshot).unwrap();
    // 3 prompt + 5 committed tokens; the last committed token is not fed yet.
    assert_eq!((info.position, info.token_count, info.fed), (7, 8, 7));
    let image = read_all(&d, &snapshot);
    assert_eq!(image.len() as u64, info.bytes);
    assert_eq!(&image[..8], b"CERESNP1");
    let again = d.snapshot(&session).unwrap();
    assert_eq!(read_all(&d, &again), image, "two snapshots byte-identical");
    d.release_snapshot(again).unwrap();
    d.release_session(session).unwrap();

    let restored = d.create_session(&model, CONTEXT).unwrap();
    d.restore(&restored, &snapshot).unwrap();
    let rest = d
        .submit(&restored, WorkRequest::decode(7))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(rest.status, Ok(()));
    assert_eq!(
        rest.tokens,
        want[5..],
        "restore then decode = uninterrupted"
    );
    assert!(matches!(
        d.restore(&restored, &snapshot),
        Err(DriverError::Status {
            call: "cere_session_restore",
            status: StatusError::Argument,
        })
    ));
    d.release_session(restored).unwrap();
    d.release_snapshot(snapshot).unwrap();
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// M2.11 C4 shutdown order: the library refuses to release a model that still has snapshots (BUSY), so shutdown
/// releases snapshots (made or imported) before sessions and models.
#[test]
fn shutdown_releases_snapshots_before_sessions_and_models() {
    let d = toy_driver(1 << 30);
    let spec = snapshot_spec(4096);
    let (model, _) = sealed_fixture_model(&d, &spec);
    let session = d.create_session(&model, 4096).unwrap();
    d.submit(&session, WorkRequest::prefill(vec![5, 6], 3))
        .unwrap()
        .wait()
        .unwrap();
    let made = d.snapshot(&session).unwrap();
    let imported = d.import_snapshot(&model, &read_all(&d, &made)).unwrap();
    assert_eq!(
        d.snapshot_info(&imported).unwrap(),
        d.snapshot_info(&made).unwrap()
    );
    // Neither snapshot, the session nor the model is released by the caller.
    d.shutdown().unwrap();
}

#[test]
fn executor_create_errors_are_typed() {
    let lib = Arc::new(Library::open(&toy_library(), None).unwrap());
    let err = Driver::start(
        lib,
        &ExecutorConfig {
            device_id: "no-such-device".to_string(),
            worker_epoch: 1,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::Status {
                call: "cere_executor_create",
                status: StatusError::Unsupported
            }
        ),
        "{err:?}"
    );
}

#[test]
fn device_id_with_a_nul_byte_is_refused_before_the_library() {
    let lib = Arc::new(Library::open(&toy_library(), None).unwrap());
    let err = Driver::start(
        lib,
        &ExecutorConfig {
            device_id: "toy\0x".to_string(),
            worker_epoch: 1,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap_err();
    assert!(matches!(err, DriverError::Argument(_)), "{err:?}");
}

/// The identity a toy backend of `spec` exports under: the toy library's digest, a profile id, the spec's layout.
fn toy_identity(spec: &ModelSpec, profile_id: &str) -> StateIdentity {
    StateIdentity {
        library_sha256: cere_compiler::sha256_hex(&std::fs::read(toy_library()).unwrap()),
        profile_id: profile_id.to_owned(),
        layout_id: export::layout_id(&spec.config),
    }
}

/// A toy executor with the fixture model sealed under a snapshot arena, and an idle session of it that has run
/// PREFILL([1, 2, 3], 1) + DECODE(4): 5 of the 12 tokens of the uninterrupted run.
fn exporter(context: u64) -> (Driver, cere_runtime::native::driver::Model, Session) {
    let d = toy_driver(1 << 30);
    let (model, _) = sealed_fixture_model(&d, &snapshot_spec(context));
    let session = d.create_session(&model, context).unwrap();
    d.submit(&session, WorkRequest::prefill(vec![1, 2, 3], 1))
        .unwrap()
        .wait()
        .unwrap();
    d.submit(&session, WorkRequest::decode(4))
        .unwrap()
        .wait()
        .unwrap();
    (d, model, session)
}

/// M2.11 C5 (spec D3, D9, U3): export a session from executor A, import it into executor B (a second toy
/// executor), restore, continue: B's tokens are A's continuation. The envelope carries A's identity, its
/// `layout_id` is the one the library stamps (the Rust formula equals `snapshot_format.cpp`'s) and its
/// checksum is the image's SHA-256.
#[test]
fn u3_export_from_a_imports_into_b_and_continues_as_a() {
    const CONTEXT: u64 = 4096;
    let spec = snapshot_spec(CONTEXT);
    let identity = toy_identity(&spec, "toy-snapshot");
    let want = toy_tokens(3, 3, 12, FIXTURE_VOCAB);

    let (a, a_model, a_session) = exporter(CONTEXT);
    let snapshot = a.snapshot(&a_session).unwrap();
    let info = a.snapshot_info(&snapshot).unwrap();
    a.release_snapshot(snapshot).unwrap();
    assert_eq!(info.layout_id, identity.layout_id, "Rust layout_id = C");
    let exported = export::export(&a, &a_session, &identity).unwrap();
    assert_eq!(exported.bytes.len() as u64, info.bytes);
    assert_eq!(&exported.bytes.as_slice()[..8], b"CERESNP1");
    let env = &exported.envelope;
    assert_eq!(
        (&env.library_sha256, &env.profile_id, env.layout_id),
        (
            &identity.library_sha256,
            &identity.profile_id,
            identity.layout_id
        )
    );
    let digest: [u8; 32] = Sha256::digest(exported.bytes.as_slice()).into();
    assert_eq!(env.sha256, digest);
    // A's own continuation, for reference: the export left its session valid.
    let a_rest = a
        .submit(&a_session, WorkRequest::decode(7))
        .unwrap()
        .wait()
        .unwrap()
        .tokens;
    assert_eq!(a_rest, want[5..]);
    a.release_session(a_session).unwrap();
    a.release_model(a_model).unwrap();
    a.shutdown().unwrap();

    let b = toy_driver(1 << 30);
    let (b_model, _) = sealed_fixture_model(&b, &spec);
    let b_session = export::import(&b, &b_model, CONTEXT, &identity, &exported).unwrap();
    let b_rest = b
        .submit(&b_session, WorkRequest::decode(7))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(b_rest.status, Ok(()));
    assert_eq!(b_rest.tokens, a_rest, "B continues as A");
    b.release_session(b_session).unwrap();
    b.release_model(b_model).unwrap();
    b.shutdown().unwrap();
}

/// M2.11 C5 (spec D3, D9, F7): an envelope whose checksum, library sha, profile or layout does not match the
/// importer is refused before the library sees the bytes (typed `ExportError`, not a library status), and the
/// importer's arena stays usable: the untouched export imports afterwards.
#[test]
fn u3_envelope_mismatches_are_refused_before_import() {
    const CONTEXT: u64 = 4096;
    let spec = snapshot_spec(CONTEXT);
    let identity = toy_identity(&spec, "toy-snapshot");
    let (a, _, a_session) = exporter(CONTEXT);
    let exported = export::export(&a, &a_session, &identity).unwrap();
    a.shutdown().unwrap();
    let b = toy_driver(1 << 30);
    let (b_model, _) = sealed_fixture_model(&b, &spec);

    let tampered = |edit: &dyn Fn(&mut Envelope, &mut [u8])| {
        let mut bytes = LockedBuf::new(exported.bytes.len()).unwrap();
        bytes
            .as_mut_slice()
            .copy_from_slice(exported.bytes.as_slice());
        let mut envelope = exported.envelope.clone();
        edit(&mut envelope, bytes.as_mut_slice());
        Exported { envelope, bytes }
    };
    let resum = |env: &mut Envelope, bytes: &[u8]| env.sha256 = Sha256::digest(bytes).into();
    let cases: [(&str, Exported); 5] = [
        (
            "checksum",
            tampered(&|_, bytes| *bytes.last_mut().unwrap() ^= 1),
        ),
        (
            "library",
            tampered(&|env, _| env.library_sha256 = "0".repeat(64)),
        ),
        (
            "profile",
            tampered(&|env, _| env.profile_id = "other-profile".to_owned()),
        ),
        ("layout", tampered(&|env, _| env.layout_id[0] ^= 1)),
        // The image header's layout_id (offset 24) differs from the envelope's, the checksum recomputed.
        (
            "header layout",
            tampered(&|env, bytes| {
                bytes[24] ^= 1;
                resum(env, bytes);
            }),
        ),
    ];
    for (name, bad) in &cases {
        let err = export::import(&b, &b_model, CONTEXT, &identity, bad).unwrap_err();
        let ok = match *name {
            "checksum" => matches!(err, ExportError::Checksum),
            "library" => matches!(err, ExportError::Library { .. }),
            "profile" => matches!(err, ExportError::Profile { .. }),
            _ => matches!(err, ExportError::Layout),
        };
        assert!(ok, "{name}: {err:?}");
    }
    // Another profile's importer refuses the genuine export too.
    let other = toy_identity(&spec, "other-profile");
    assert!(matches!(
        export::import(&b, &b_model, CONTEXT, &other, &exported),
        Err(ExportError::Profile { .. })
    ));
    let session = export::import(&b, &b_model, CONTEXT, &identity, &exported).unwrap();
    b.release_session(session).unwrap();
    b.shutdown().unwrap();
}

// ---------------------------------------------------------------------------------------------------------
// M2.8 A7 (spec m28 A7, D8, D9, C10b): the driver's constrained jobs against the toy (`native_token_mask` 1).

/// `⌈FIXTURE_VOCAB / 32⌉`: the words of one mask row.
const MASK_WORDS: usize = FIXTURE_VOCAB.div_ceil(32) as usize;

/// A mask allowing exactly `id`.
fn only(id: u32) -> Vec<u32> {
    let mut mask = vec![0u32; MASK_WORDS];
    mask[(id >> 5) as usize] |= 1 << (id & 31);
    mask
}

/// Greedy with processors (temperature 0): a constrained item needs non-null sampling (A1).
fn masked_greedy() -> Sampling {
    Sampling {
        temperature: 0.0,
        top_p: 1.0,
        min_p: 0.0,
        top_k: FIXTURE_VOCAB,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        seed: 0,
        logit_bias: Vec::new(),
    }
}

fn constrained(work: WorkRequest, mask: &[u32]) -> WorkRequest {
    work.with_sampling(masked_greedy())
        .with_flags(ffi::WORK_NO_SPECULATION)
        .with_vocab_mask(mask)
}

/// The toy's `record=` lines: `key=value` pairs per scored step.
fn record_lines(path: &std::path::Path) -> Vec<BTreeMap<String, String>> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|line| {
            line.split(' ')
                .map(|kv| {
                    let (k, v) = kv.split_once('=').unwrap();
                    (k.to_owned(), v.to_owned())
                })
                .collect()
        })
        .collect()
}

/// A7/D8: a constrained PREFILL commits its position-0 mask's id; a constrained DECODE commits position 0 with its
/// submit mask, then is not stepped (no record line, no progress) until `Driver::constrain` supplies the next
/// position's mask through `Job::mask`; a Constrain for another position is ARGUMENT. Unconstrained submits carry
/// `vocab_mask_words` 0 (C10b: their record lines have no `mask=`), and a Constrain of one is refused.
#[test]
fn constrained_job_waits_for_each_mask_and_commits_the_allowed_ids() {
    const CONTEXT: u64 = 4096;
    let t = Tmp::new("a7-record");
    let record = t.join("steps.record");
    let d = toy_driver_on(&format!("toy,record={}", record.display()), 1 << 30);
    let spec = fixture_spec(CONTEXT);
    let (model, caps) = sealed_fixture_model(&d, &spec);
    assert_eq!(caps.native_token_mask, 1);

    // C10b: an unconstrained PREFILL + DECODE records no mask, and has no mask buffer.
    let session = d.create_session(&model, CONTEXT).unwrap();
    let r = d
        .submit(&session, WorkRequest::prefill(vec![1, 2, 3], 1))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.tokens, toy_tokens(3, 3, 1, FIXTURE_VOCAB));
    let job = d.submit(&session, WorkRequest::decode(2)).unwrap();
    assert!(job.mask().is_none());
    assert!(matches!(
        d.constrain(&job, 0),
        Err(DriverError::Argument(_)) | Ok(())
    ));
    assert_eq!(job.wait().unwrap().tokens.len(), 2);
    d.release_session(session).unwrap();
    let plain = record_lines(&record);
    assert_eq!(plain.len(), 3, "{plain:?}");
    assert!(plain.iter().all(|s| !s.contains_key("mask")), "{plain:?}");

    // The constrained PREFILL: its one token is the mask's only id, not the toy's `next`.
    let session = d.create_session(&model, CONTEXT).unwrap();
    let free = toy_tokens(3, 3, 1, FIXTURE_VOCAB)[0];
    let t0 = (free + 17) % FIXTURE_VOCAB;
    let r = d
        .submit(
            &session,
            constrained(WorkRequest::prefill(vec![1, 2, 3], 1), &only(t0)),
        )
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        (r.tokens.as_slice(), r.finish),
        ([t0].as_slice(), Finish::Length)
    );

    // The constrained DECODE: position 0 from the submit, then one Constrain per position.
    let picks = [101u32, 202, 303];
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let job = d
        .submit_with_progress(
            &session,
            constrained(WorkRequest::decode(3), &only(picks[0])),
            tx,
        )
        .unwrap();
    let mut streamed = Vec::new();
    for (position, &next) in picks.iter().enumerate().skip(1) {
        while streamed.len() < position {
            streamed.extend(rx.blocking_recv().expect("progress").tokens);
        }
        assert_eq!(streamed, picks[..position]);
        // Awaiting: no step is taken (no record line) and nothing more is committed.
        let recorded = record_lines(&record).len();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(
            record_lines(&record).len(),
            recorded,
            "stepped without a mask"
        );
        assert!(rx.try_recv().is_err());
        // Only the next position (= committed) takes a mask.
        assert!(matches!(
            d.constrain(&job, position as u64 + 1),
            Err(DriverError::Status {
                call: "cere_executor_constrain",
                status: StatusError::Argument,
            })
        ));
        job.mask().unwrap().copy_from_slice(&only(next));
        d.constrain(&job, position as u64).unwrap();
    }
    let r = job.wait().unwrap();
    assert_eq!(
        (r.tokens.as_slice(), r.finish),
        (picks.as_slice(), Finish::Length)
    );
    assert_eq!(r.status, Ok(()));
    d.release_session(session).unwrap();
    let masked = &record_lines(&record)[plain.len()..];
    assert_eq!(masked.len(), 4, "{masked:?}");
    for (step, want) in masked.iter().zip([t0, picks[0], picks[1], picks[2]]) {
        assert_eq!(step["mask"], "1", "{step:?}");
        assert_eq!(step["token"], want.to_string(), "{step:?}");
    }
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// A7/D8: a cancel of a constrained job awaiting its mask clears `awaiting_mask`, so the next poll reaches
/// `CERE_FINISH_CANCELLED` without a mask (no hang).
#[test]
fn cancel_while_awaiting_a_mask_finishes_cancelled() {
    const CONTEXT: u64 = 4096;
    let d = toy_driver(1 << 30);
    let spec = fixture_spec(CONTEXT);
    let (model, _) = sealed_fixture_model(&d, &spec);
    let session = d.create_session(&model, CONTEXT).unwrap();
    let all = vec![u32::MAX; MASK_WORDS];
    d.submit(
        &session,
        constrained(WorkRequest::prefill(vec![1, 2, 3], 1), &all),
    )
    .unwrap()
    .wait()
    .unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let job = d
        .submit_with_progress(&session, constrained(WorkRequest::decode(5), &all), tx)
        .unwrap();
    assert_eq!(rx.blocking_recv().expect("position 0").tokens.len(), 1);
    d.cancel(job.ticket()).unwrap();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || done_tx.send(job.wait()).unwrap());
    let r = done_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the cancelled job awaiting its mask hung")
        .unwrap();
    assert_eq!(r.finish, Finish::Cancelled);
    assert_eq!(r.status, Ok(()));
    assert_eq!(r.committed_tokens, 1);
    d.release_session(session).unwrap();
    d.release_model(model).unwrap();
    d.shutdown().unwrap();
}

/// GrammarSpec GS-02: a `grammar_drafts` DECODE on the drafting toy reports the frozen drafts with each report that
/// leaves it awaiting a mask (`cere_executor_round_drafts`); `Driver::constrain_drafts` with one mask per kept draft
/// plus the position's verifies them in one round, and a report with no drafts takes a plain `constrain`. The
/// tokens equal the same ticket steered without speculation. The driver refuses more than
/// CERE_MAX_CONSTRAINED_DRAFTS drafts and a buffer without one mask per kept draft (ARGUMENT, nothing sent), a
/// `grammar_drafts` item without a mask, and on a library predating the calls (the pre-staging toy) it is Missing.
#[test]
fn grammar_drafts_job_reports_frozen_drafts_and_verifies_them_under_their_masks() {
    const CONTEXT: u64 = 4096;
    let all = vec![u32::MAX; MASK_WORDS];
    let run = |speculate: bool| {
        let d = toy_driver(1 << 30);
        let (model, _) = sealed_fixture_model(&d, &fixture_spec_with_drafts(CONTEXT, 4));
        let session = d.create_session(&model, CONTEXT).unwrap();
        let flags = if speculate {
            0
        } else {
            ffi::WORK_NO_SPECULATION
        };
        let item = |work: WorkRequest| constrained(work, &all).with_flags(flags);
        let mut tokens = d
            .submit(&session, item(WorkRequest::prefill(vec![1, 2, 3], 1)))
            .unwrap()
            .wait()
            .unwrap()
            .tokens;
        let mut decode = item(WorkRequest::decode(6));
        if speculate {
            decode = decode.with_grammar_drafts();
        }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Committed>();
        let job = d.submit_with_progress(&session, decode, tx).unwrap();
        let mut committed = 0u64;
        let mut kept = Vec::new();
        while let Some(c) = rx.blocking_recv() {
            committed += c.tokens.len() as u64;
            if committed == 6 {
                assert_eq!(c.drafts, None, "the terminal report awaits nothing");
                continue;
            }
            let drafts = match c.drafts {
                None => {
                    assert!(
                        !speculate,
                        "a grammar_drafts job reports its round's drafts"
                    );
                    Vec::new()
                }
                Some(drafts) => drafts.unwrap().as_slice().to_vec(),
            };
            if drafts.is_empty() {
                d.constrain(&job, committed).unwrap();
                continue;
            }
            assert!(drafts.len() <= ffi::MAX_CONSTRAINED_DRAFTS as usize);
            assert_eq!(
                d.constrain_drafts(&job, committed, &[1, 2, 3, 4]),
                Err(DriverError::Argument(
                    "4 drafts exceed CERE_MAX_CONSTRAINED_DRAFTS (3)".to_owned()
                ))
            );
            assert!(matches!(
                d.constrain_drafts(&job, committed, &drafts),
                Err(DriverError::Argument(_))
            ));
            {
                let mut buffer = job.mask().unwrap();
                for _ in &drafts {
                    buffer.extend_from_slice(&all);
                }
            }
            d.constrain_drafts(&job, committed, &drafts).unwrap();
            job.mask().unwrap().truncate(MASK_WORDS);
            kept.extend(drafts);
        }
        let r = job.wait().unwrap();
        assert_eq!((r.finish, r.status), (Finish::Length, Ok(())));
        tokens.extend(r.tokens);
        d.release_session(session).unwrap();
        d.release_model(model).unwrap();
        d.shutdown().unwrap();
        (tokens, kept, r.accepted_drafts)
    };
    let (plain, none, accepted) = run(false);
    assert_eq!((none.len(), accepted), (0, 0));
    let (tokens, kept, accepted) = run(true);
    assert_eq!(tokens, plain);
    assert!(!kept.is_empty(), "the drafting toy froze no drafts");
    assert_eq!(
        accepted,
        kept.len() as u64,
        "every kept draft is the toy's own"
    );

    // A grammar_drafts item without a mask is refused before the library.
    let d = toy_driver(1 << 30);
    let (model, _) = sealed_fixture_model(&d, &fixture_spec_with_drafts(CONTEXT, 4));
    let session = d.create_session(&model, CONTEXT).unwrap();
    assert!(matches!(
        d.submit(
            &session,
            WorkRequest::prefill(vec![1, 2, 3], 1).with_grammar_drafts()
        ),
        Err(DriverError::Argument(_))
    ));
    d.release_session(session).unwrap();
    d.release_model(model).unwrap();
    d.shutdown().unwrap();

    let old = Driver::start(
        Arc::new(Library::open(&toy_library_prestaging(), None).unwrap()),
        &ExecutorConfig {
            device_id: "toy".into(),
            worker_epoch: 7,
            memory_budget_bytes: 1 << 30,
        },
    )
    .unwrap();
    let api = old.library().api();
    assert!(api.executor_round_drafts.is_none() && api.executor_constrain_drafts.is_none());
    let (model, _) = sealed_fixture_model(&old, &fixture_spec(CONTEXT));
    let session = old.create_session(&model, CONTEXT).unwrap();
    let work = constrained(WorkRequest::decode(2), &all).with_grammar_drafts();
    assert_eq!(
        old.submit(&session, work).unwrap_err(),
        DriverError::Missing("cere_executor_round_drafts")
    );
    old.release_session(session).unwrap();
    old.release_model(model).unwrap();
    old.shutdown().unwrap();
}
