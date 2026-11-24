//! R3 (spec m22): `cere-native probe` against the CPU toy executor (`$CERE_TOY_EXECUTOR`) and a stamped copy of the
//! M2.1 fixture artifact: the report has every field and the toy model's smoke token, `--cache cold` evicts and
//! labels per the M2.1 rules, a wrong manifest digest exits 2 before the library is touched, and the toy built
//! without its forward pass reports SMOKE `unsupported` (exit 0, `ready: false`).
//!
//! R5: `cere-native generate` and `score` take token IDs on stdin and write token IDs on stdout (greedy PREFILL +
//! DECODE on the toy), and `detok --check` counts the `{ids, text_sha256}` frames whose decode differs.
mod native_support;

use cere_compiler::artifact::{self, Cache};
use cere_compiler::fsio::Residency;
use native_support::*;
use serde_json::Value;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

/// The fixed SMOKE prompt of `cere-native probe`.
const SMOKE_PROMPT: [u32; 4] = [1, 2, 3, 4];
/// BC1 leftover: the toy's prefill selection signature, the SHA-256 of the format v1 text (cere_executor.h).
const TOY_PREFILL_SIGNATURE: &str =
    "63c07240fed82246042181ae7d6429d75511153a0a8fd50089fa704873ad7a7f";

fn probe(dir: &Path, sha: &str, library: &Path, extra: &[&str]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_cere-native"));
    c.arg("probe")
        .arg("--artifact")
        .arg(dir)
        .args(["--expected-manifest-sha256", sha])
        .arg("--library")
        .arg(library)
        .args(["--device-id", "toy"])
        .args(extra);
    c.output().unwrap_or_else(|e| panic!("running {c:?}: {e}"))
}

fn ok(out: Output) -> Output {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn json(out: &Output) -> Value {
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stdout)))
}

/// The class of a failure: the first stderr line `<class>: <message>`.
fn class_of(out: &Output) -> String {
    let err = String::from_utf8_lossy(&out.stderr);
    let first = err.lines().next().unwrap_or("");
    first.split_once(':').map_or(first, |(c, _)| c).to_string()
}

fn residency(v: &Value) -> Residency {
    Residency {
        resident_pages: v["resident_pages"].as_u64().unwrap(),
        total_pages: v["total_pages"].as_u64().unwrap(),
    }
}

const FIELDS: [&str; 34] = [
    "outcome",
    "artifact",
    "artifact_id",
    "manifest_sha256",
    "library",
    "device_id",
    "worker_epoch",
    "memory_budget_bytes",
    "max_context_tokens",
    "kv_pool_tokens",
    "rows",
    "speculative_draft_tokens",
    "cache",
    "label",
    "evicted_files",
    "residency_before",
    "residency_after",
    "stages_ms",
    "tensors",
    "uploaded_device_bytes",
    "uploaded_host_bytes",
    "discarded_bytes",
    "resident_host_bytes",
    "resident_files",
    "payload_bytes_read",
    "metadata_bytes_read",
    "model_bytes",
    "staging",
    "stop_token_ids",
    "capabilities",
    "smoke",
    "ready",
    "released",
    "peak_rss_bytes",
];

#[test]
fn probe_reports_every_field_and_the_toy_smoke_token() {
    let t = Tmp::new("cli");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let r = json(&ok(probe(&dir, &sha, &lib, &["--cache", "warm", "--json"])));
    let obj = r.as_object().unwrap();
    for k in FIELDS {
        assert!(obj.contains_key(k), "missing {k}: {r:#}");
    }
    assert_eq!(obj.len(), FIELDS.len(), "unexpected fields: {r:#}");
    assert_eq!(r["outcome"], "probed");
    assert_eq!(r["manifest_sha256"], sha.as_str());
    let lib_bytes = std::fs::read(&lib).unwrap();
    assert_eq!(
        r["library"]["sha256"],
        cere_compiler::sha256_hex(&lib_bytes).as_str()
    );
    assert_eq!(r["library"]["abi_version"], 5);
    assert_eq!(r["device_id"], "toy");
    for stage in [
        "discover",
        "validate",
        "allocate",
        "read_upload",
        "upload_put",
        "instantiate",
        "smoke",
        "executor",
        "ready",
        "release",
        "total",
    ] {
        assert!(r["stages_ms"][stage].is_u64(), "stages_ms.{stage}: {r:#}");
    }
    // M2.16: load-to-ready excludes the probe's eviction, residency passes and the release.
    let st = &r["stages_ms"];
    assert!(st["ready"].as_u64() <= st["total"].as_u64(), "{st:#}");
    assert!(
        st["ready"].as_u64().unwrap()
            >= st["allocate"].as_u64().unwrap() + st["read_upload"].as_u64().unwrap(),
        "{st:#}"
    );

    // The toy model's token after the fixed prompt.
    let want = toy_tokens(4, SMOKE_PROMPT.len() as u64, 1, FIXTURE_VOCAB)[0];
    assert_eq!(r["smoke"]["status"], "passed", "{r:#}");
    assert_eq!(r["smoke"]["prompt"], serde_json::json!(SMOKE_PROMPT));
    assert_eq!(r["smoke"]["token"], want);
    assert_eq!(r["smoke"]["finish"], "length");
    assert_eq!(r["ready"], true);
    assert_eq!(r["released"], true);

    // Counters and model bytes equal the fixture inventory.
    let inv = inventory_totals();
    assert_eq!(r["uploaded_device_bytes"], inv.device);
    assert_eq!(r["uploaded_host_bytes"], inv.host);
    assert_eq!(r["discarded_bytes"], inv.draft);
    assert_eq!(r["model_bytes"]["device"], inv.device);
    assert_eq!(r["model_bytes"]["host_pinned"], inv.host);
    assert_eq!(r["model_bytes"]["discard"], inv.draft);
    // L1 (m216 design note): the DISCARD draft bytes are skipped, never read; `discarded_bytes` counts them.
    assert_eq!(
        r["payload_bytes_read"].as_u64().unwrap(),
        inv.device + inv.host
    );
    assert!(
        r["staging"]["pass"]["reads"].as_u64().unwrap() >= 1,
        "{r:#}"
    );
    assert_eq!(r["stop_token_ids"], serde_json::json!([7, 6]));

    let caps = &r["capabilities"];
    assert_eq!(caps["execution_mode"], "native");
    assert_eq!(
        caps["operations"],
        serde_json::json!(["prefill", "decode", "cancel"])
    );
    assert_eq!(caps["max_context_tokens"], 2048);
    assert_eq!(caps["worker_epoch"], 1);
    assert_eq!(caps["prefill_signature"], TOY_PREFILL_SIGNATURE);

    // The text form carries the same verdict.
    let out = ok(probe(&dir, &sha, &lib, &["--cache", "warm"]));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.contains(&format!("smoke: passed (token {want})")),
        "{text}"
    );
    assert!(text.contains("ready: true"), "{text}");
}

/// M2.6 D9: `--rows N` plans N sessions (default 1); the executor reports them as its decode rows (D3).
#[test]
fn rows_plan_that_many_sessions() {
    let t = Tmp::new("cli-rows");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    for (extra, rows, batched) in [
        (&["--cache", "warm", "--json"][..], 1, false),
        (&["--cache", "warm", "--json", "--rows", "4"][..], 4, true),
    ] {
        let r = json(&ok(probe(&dir, &sha, &lib, extra)));
        let caps = &r["capabilities"];
        assert_eq!(caps["decode_batch_min"], 1, "{r:#}");
        assert_eq!(caps["decode_batch_max"], rows, "{r:#}");
        assert_eq!(caps["native_decode_batch"], batched, "{r:#}");
        assert_eq!(r["smoke"]["status"], "passed", "{r:#}");
    }
    let out = probe(&dir, &sha, &lib, &["--cache", "warm", "--rows", "0"]);
    assert_eq!(
        (out.status.code(), class_of(&out).as_str()),
        (Some(2), "usage")
    );
}

/// P8 (m23 design note): `probe --speculative-draft-tokens 4` plans the MTP draft too (the speculation profiles'
/// memory) and reports it; only 0 or 4, and only for `probe`.
#[test]
fn probe_plans_the_draft_when_asked() {
    let t = Tmp::new("cli-drafts");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let warm = ["--cache", "warm", "--json"];
    let r = json(&ok(probe(&dir, &sha, &lib, &warm)));
    assert_eq!(r["speculative_draft_tokens"], 0, "{r:#}");
    let r = json(&ok(probe(
        &dir,
        &sha,
        &lib,
        &[
            &warm[..],
            &["--speculative-draft-tokens", "4", "--rows", "6"],
        ]
        .concat(),
    )));
    assert_eq!(r["speculative_draft_tokens"], 4, "{r:#}");
    assert_eq!(r["rows"], 6, "{r:#}");
    assert_eq!(r["capabilities"]["decode_batch_max"], 6, "{r:#}");
    let out = probe(
        &dir,
        &sha,
        &lib,
        &["--cache", "warm", "--speculative-draft-tokens", "3"],
    );
    assert_eq!(
        (out.status.code(), class_of(&out).as_str()),
        (Some(2), "usage")
    );
    for cmd in ["generate", "score", "logprobs"] {
        let mut args = model_args(cmd, &dir, &sha, &lib);
        args.extend(["--speculative-draft-tokens", "4"]);
        let out = run_with_stdin(&args, b"{}");
        assert_eq!(out.status.code(), Some(2), "{cmd}: {out:?}");
        assert_eq!(class_of(&out), "usage", "{cmd}: {out:?}");
    }
}

#[test]
fn cold_evicts_and_labels_per_the_m21_rules() {
    let t = Tmp::new("clicold");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let cold = json(&ok(probe(&dir, &sha, &lib, &["--cache", "cold", "--json"])));
    assert_eq!(cold["cache"], "cold");
    assert!(cold["evicted_files"].as_u64().unwrap() > 0, "{cold:#}");
    let before = residency(&cold["residency_before"]);
    assert_eq!(cold["label"], artifact::cache_label(Cache::Cold, before));
    assert!(
        ["cold", "cold_unverified"].contains(&cold["label"].as_str().unwrap()),
        "{cold:#}"
    );
    // Staging read every tensor byte of the desc but the skipped DISCARD ones, counted by the reader (page-cache
    // residency after it is the kernel's: on a host under memory pressure reclaim takes staged pages back at once,
    // ci-docker `3f53be2`), and the residency after it measures the same pages as before.
    let inv = inventory_totals();
    assert_eq!(
        cold["payload_bytes_read"].as_u64().unwrap(),
        inv.device + inv.host,
        "{cold:#}"
    );
    let after = residency(&cold["residency_after"]);
    assert_eq!(after.total_pages, before.total_pages, "{cold:#}");

    // A warm repeat evicts nothing and labels by the residency it measured (the label rule itself:
    // `cere_compiler::artifact` tests `cold_label_needs_at_most_one_percent_resident`,
    // `warm_label_needs_at_least_95_percent_resident`).
    let warm = json(&ok(probe(&dir, &sha, &lib, &["--cache", "warm", "--json"])));
    assert_eq!(warm["evicted_files"], 0);
    let before = residency(&warm["residency_before"]);
    assert_eq!(warm["label"], artifact::cache_label(Cache::Warm, before));
    assert_eq!(before.total_pages, after.total_pages, "{warm:#}");
}

#[test]
fn a_wrong_manifest_digest_exits_2_without_loading_the_library() {
    let t = Tmp::new("clisha");
    let (dir, sha) = verified_artifact(&t);
    let wrong = format!(
        "{}{}",
        &sha[..63],
        if sha.ends_with('0') { "1" } else { "0" }
    );
    // A library path that does not exist: touching it at all would fail as `library_io`.
    let absent = t.join("no-such-library.so");
    for cache in ["cold", "warm"] {
        let out = probe(&dir, &wrong, &absent, &["--cache", cache, "--json"]);
        assert_eq!(out.status.code(), Some(2), "{cache}: {out:?}");
        assert_eq!(class_of(&out), "manifest_digest", "{cache}: {out:?}");
        assert!(out.stdout.is_empty(), "{cache}: {out:?}");
    }
    // With the right digest the same absent library is the failure.
    let out = probe(&dir, &sha, &absent, &["--cache", "warm"]);
    assert_eq!(
        (out.status.code(), class_of(&out).as_str()),
        (Some(1), "library_io")
    );
}

#[test]
fn a_library_hash_mismatch_exits_2_before_dlopen() {
    let t = Tmp::new("clilibsha");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let out = probe(
        &dir,
        &sha,
        &lib,
        &["--cache", "warm", "--library-sha256", &"0".repeat(64)],
    );
    assert_eq!(
        (out.status.code(), class_of(&out).as_str()),
        (Some(2), "library_hash"),
        "{out:?}"
    );
    assert!(out.stdout.is_empty());
    let pinned = cere_compiler::sha256_hex(&std::fs::read(&lib).unwrap());
    let r = json(&ok(probe(
        &dir,
        &sha,
        &lib,
        &["--cache", "warm", "--json", "--library-sha256", &pinned],
    )));
    assert_eq!(r["library"]["sha256"], pinned.as_str());
}

#[test]
fn a_library_without_a_forward_pass_reports_smoke_unsupported() {
    let t = Tmp::new("clinofwd");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library_noforward();
    let r = json(&ok(probe(&dir, &sha, &lib, &["--cache", "warm", "--json"])));
    assert_eq!(r["smoke"]["status"], "unsupported", "{r:#}");
    assert!(r["smoke"]["token"].is_null());
    assert!(r["smoke"]["reason"].is_string());
    assert_eq!(r["ready"], false);
    assert_eq!(r["capabilities"]["execution_mode"], "unsupported");
    assert_eq!(r["capabilities"]["operations"], serde_json::json!([]));
    // The upload itself is complete.
    let inv = inventory_totals();
    assert_eq!(r["uploaded_device_bytes"], inv.device);
    assert_eq!(r["uploaded_host_bytes"], inv.host);
    assert_eq!(r["discarded_bytes"], inv.draft);

    let out = ok(probe(&dir, &sha, &lib, &["--cache", "warm"]));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("smoke: unsupported"), "{text}");
    assert!(text.contains("ready: false"), "{text}");
}

#[test]
fn usage_errors_exit_2() {
    let bin = env!("CARGO_BIN_EXE_cere-native");
    for args in [
        vec![],
        vec!["load"],
        vec!["probe", "--artifact", "x"],
        vec![
            "probe",
            "--artifact",
            "x",
            "--expected-manifest-sha256",
            "y",
            "--library",
            "z",
            "--cache",
            "lukewarm",
        ],
    ] {
        let out = Command::new(bin).args(&args).output().unwrap();
        assert_eq!(
            (out.status.code(), class_of(&out).as_str()),
            (Some(2), "usage"),
            "{args:?}"
        );
    }
    let out = Command::new(bin).arg("--help").output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("cere-native probe"));
}

/// m25 design note C3/Interfaces "Rust": `--kv-pool-tokens N` (default 0 = the derived pool) is the desc's
/// `kv_pool_tokens`, reported beside `max_context_tokens`; a value that is not a non-negative integer is a usage error.
#[test]
fn kv_pool_tokens_flag_is_parsed_and_reported() {
    let t = Tmp::new("clipoolflag");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let r = json(&ok(probe(&dir, &sha, &lib, &["--cache", "warm", "--json"])));
    assert_eq!(r["kv_pool_tokens"], 0, "{r:#}");
    let r = json(&ok(probe(
        &dir,
        &sha,
        &lib,
        &["--cache", "warm", "--json", "--kv-pool-tokens", "0"],
    )));
    assert_eq!(r["kv_pool_tokens"], 0, "{r:#}");
    for bad in ["-64", "many", ""] {
        let out = probe(
            &dir,
            &sha,
            &lib,
            &["--cache", "warm", "--kv-pool-tokens", bad],
        );
        assert_eq!(
            (out.status.code(), class_of(&out).as_str()),
            (Some(2), "usage"),
            "{bad:?}: {out:?}"
        );
    }
}

/// m25 design note D4 through `cere-native`: the pool reaches `cere_model_create` and the capabilities report the
/// pool allocated (0 -> 1 session x round64(2,048)) and the 64-token page; a pool that is not a multiple of 64 is
/// ARGUMENT, one over the toy support's 200,000 UNSUPPORTED (both at ALLOCATE, exit 1).
#[test]
fn kv_pool_tokens_reaches_the_executor_and_its_capabilities() {
    let t = Tmp::new("clipool");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let r = json(&ok(probe(&dir, &sha, &lib, &["--cache", "warm", "--json"])));
    assert_eq!(r["capabilities"]["kv_pool_tokens"], 2048, "{r:#}");
    assert_eq!(r["capabilities"]["kv_page_tokens"], 64, "{r:#}");
    let r = json(&ok(probe(
        &dir,
        &sha,
        &lib,
        &["--cache", "warm", "--json", "--kv-pool-tokens", "8192"],
    )));
    assert_eq!(r["kv_pool_tokens"], 8192, "{r:#}");
    assert_eq!(r["capabilities"]["kv_pool_tokens"], 8192, "{r:#}");
    assert_eq!(r["capabilities"]["kv_page_tokens"], 64, "{r:#}");
    assert_eq!(r["smoke"]["status"], "passed", "{r:#}");
    for (pool, class) in [
        ("2100", "argument"),
        ("1024", "argument"),
        ("1048640", "unsupported"),
    ] {
        let out = probe(
            &dir,
            &sha,
            &lib,
            &["--cache", "warm", "--json", "--kv-pool-tokens", pool],
        );
        assert_eq!(
            (out.status.code(), class_of(&out).as_str()),
            (Some(1), class),
            "{pool}: {out:?}"
        );
        assert!(out.stdout.is_empty(), "{pool}: {out:?}");
    }
}

/// Runs `cere-native <cmd>` on the toy with `stdin` written to the child.
fn run_with_stdin(args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cere-native"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // A refusing child may exit before reading everything; the write error is then expected.
    let _ = child.stdin.take().unwrap().write_all(stdin);
    child.wait_with_output().unwrap()
}

fn model_args<'a>(cmd: &'a str, dir: &'a Path, sha: &'a str, lib: &'a Path) -> Vec<&'a str> {
    vec![
        cmd,
        "--artifact",
        dir.to_str().unwrap(),
        "--expected-manifest-sha256",
        sha,
        "--library",
        lib.to_str().unwrap(),
        "--device-id",
        "toy",
        "--cache",
        "warm",
    ]
}

/// The toy's greedy continuation of `prompt`, ended at the fixture model's first stop token (committed).
fn toy_greedy(prompt: &[u32], max_new: usize) -> (Vec<u32>, &'static str) {
    let last = *prompt.last().unwrap();
    let mut out = Vec::new();
    for id in toy_tokens(last, prompt.len() as u64, max_new, FIXTURE_VOCAB) {
        out.push(id);
        if [7, 6].contains(&id) {
            return (out, "stop_token");
        }
    }
    (out, "length")
}

#[test]
fn generate_writes_the_greedy_token_ids() {
    let t = Tmp::new("cligen");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let args = model_args("generate", &dir, &sha, &lib);
    let out = ok(run_with_stdin(
        &args,
        br#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 9}"#,
    ));
    let r = json(&out);
    let (ids, finish) = toy_greedy(&[1, 2, 3], 9);
    assert_eq!(
        r,
        serde_json::json!({"prompt_tokens": 3, "ids": ids, "finish": finish, "prefill_signature": TOY_PREFILL_SIGNATURE})
    );
    assert!(out.stderr.is_empty(), "{out:?}");

    // Prompt + max_new_tokens over the context (2,048 by default) is refused before any work.
    for doc in [
        r#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 2046}"#,
        r#"{"prompt_ids": [], "max_new_tokens": 1}"#,
        r#"{"prompt_ids": [1], "max_new_tokens": 0}"#,
        r#"{"prompt_ids": ["x"], "max_new_tokens": 1}"#,
    ] {
        let out = run_with_stdin(&args, doc.as_bytes());
        assert_eq!(out.status.code(), Some(2), "{doc}: {out:?}");
        assert!(out.stdout.is_empty(), "{doc}: {out:?}");
    }
    let out = run_with_stdin(
        &args,
        br#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 2046}"#,
    );
    assert_eq!(class_of(&out), "context_length_exceeded");
}

/// M2.11 R3/D13: `generate --preempt-every N` snapshots the session after every N committed tokens, releases it,
/// restores the snapshot into a fresh session and continues: the ids equal the unpreempted run's, and the
/// preemptions are counted. The option is `generate`'s only, and N is a positive integer.
#[test]
fn generate_preempt_every_restores_and_equals_the_unpreempted_run() {
    let t = Tmp::new("clipreempt");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let doc = br#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 9}"#;
    let plain = json(&ok(run_with_stdin(
        &model_args("generate", &dir, &sha, &lib),
        doc,
    )));
    let (ids, finish) = toy_greedy(&[1, 2, 3], 9);
    assert!(ids.len() > 2, "the toy run must outlast one preemption");
    for n in [1u64, 2, 3] {
        let mut args = model_args("generate", &dir, &sha, &lib);
        let every = n.to_string();
        args.extend(["--preempt-every", every.as_str()]);
        let out = ok(run_with_stdin(&args, doc));
        assert!(out.stderr.is_empty(), "{out:?}");
        let r = json(&out);
        assert_eq!(r["ids"], plain["ids"], "n = {n}");
        assert_eq!(r["finish"], finish, "n = {n}");
        assert_eq!(r["prompt_tokens"], 3);
        // A preemption after every n committed tokens while the generation goes on.
        assert_eq!(r["preemptions"], (ids.len() as u64 - 1) / n, "n = {n}");
    }
    // Without the option the report has no preemption count.
    assert!(plain.get("preemptions").is_none(), "{plain}");

    for (cmd, every) in [
        ("generate", "0"),
        ("generate", "x"),
        ("score", "2"),
        ("probe", "2"),
    ] {
        let mut args = model_args(cmd, &dir, &sha, &lib);
        args.extend(["--preempt-every", every]);
        let out = run_with_stdin(&args, doc);
        assert_eq!(out.status.code(), Some(2), "{cmd} {every}: {out:?}");
        assert_eq!(class_of(&out), "usage", "{cmd} {every}: {out:?}");
    }
}

/// m23 design note PI-02, D9: `generate --levers NAME=ID[,...]` creates the model with that lever table (the toy's
/// tokens do not depend on the levers, so the ids equal the default-lever run's). The table is checked as cered
/// checks a profile's levers for a model without drafts: an unknown lever or implementation, a speculation lever that
/// needs the drafts, a malformed item and a lever named twice are usage errors (exit 2) before anything loads; the
/// option is `generate`'s only.
#[test]
fn generate_levers_select_the_model_levers_and_are_generate_only() {
    let t = Tmp::new("clilevers");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let doc = br#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 9}"#;
    let plain = json(&ok(run_with_stdin(
        &model_args("generate", &dir, &sha, &lib),
        doc,
    )));
    let mut args = model_args("generate", &dir, &sha, &lib);
    args.extend([
        "--levers",
        "k2_gemm=small_m,decode_step=spec_graph,verify_attention=batched,k2_weights=fp8",
    ]);
    let out = ok(run_with_stdin(&args, doc));
    assert!(out.stderr.is_empty(), "{out:?}");
    assert_eq!(json(&out), plain);

    for (levers, needle) in [
        ("k3_gemm=small_m", "--levers k3_gemm: unknown lever"),
        (
            "k2_gemm=tiny",
            "--levers k2_gemm: unknown implementation \"tiny\"",
        ),
        (
            "k2_gemm=small_m,drafter=mtp+lookup",
            "--levers drafter: \"mtp+lookup\" needs speculative_draft_tokens = 4",
        ),
        (
            "draft_policy=adaptive",
            "--levers draft_policy: \"adaptive\" needs speculative_draft_tokens = 4",
        ),
        ("k2_gemm", "--levers item \"k2_gemm\" is not NAME=ID"),
        (
            "k2_gemm=small_m,k2_gemm=cublaslt",
            "--levers names k2_gemm twice",
        ),
    ] {
        let mut args = model_args("generate", &dir, &sha, &lib);
        args.extend(["--levers", levers]);
        let out = run_with_stdin(&args, doc);
        assert_eq!(out.status.code(), Some(2), "{levers}: {out:?}");
        assert_eq!(class_of(&out), "usage", "{levers}: {out:?}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(needle),
            "{levers}: {out:?}"
        );
        assert!(out.stdout.is_empty(), "{levers}: {out:?}");
    }
    for cmd in ["probe", "score", "logprobs"] {
        let mut args = model_args(cmd, &dir, &sha, &lib);
        args.extend(["--levers", "k2_gemm=small_m"]);
        let out = run_with_stdin(&args, doc);
        assert_eq!(out.status.code(), Some(2), "{cmd}: {out:?}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("--levers is a generate option"),
            "{cmd}: {out:?}"
        );
    }
}

/// O60: `generate --jsonl` answers one request per stdin line with one compact result per stdout line, in order,
/// each equal to the one-request form's document for that request (a fresh session per request). A malformed line
/// refuses the whole run before anything is generated, naming the line; `--jsonl` is `generate`'s only.
#[test]
fn generate_jsonl_answers_one_line_per_request() {
    let t = Tmp::new("clijsonl");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let single = model_args("generate", &dir, &sha, &lib);
    let mut args = single.clone();
    args.push("--jsonl");
    let prompts: [&[u32]; 4] = [&[1, 2, 3], &[5, 9, 11, 4], &[2], &[1, 2, 3]];
    let mut stdin = String::new();
    for p in prompts {
        stdin.push_str(&serde_json::json!({"prompt_ids": p, "max_new_tokens": 9}).to_string());
        stdin.push('\n');
    }
    let out = ok(run_with_stdin(&args, stdin.as_bytes()));
    assert!(out.stderr.is_empty(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), prompts.len(), "{text}");
    for (line, prompt) in lines.iter().zip(prompts) {
        let got: Value = serde_json::from_str(line).unwrap();
        let request = serde_json::json!({"prompt_ids": prompt, "max_new_tokens": 9}).to_string();
        let one = json(&ok(run_with_stdin(&single, request.as_bytes())));
        assert_eq!(got, one, "{prompt:?}");
        let (ids, finish) = toy_greedy(prompt, 9);
        assert_eq!(
            got,
            serde_json::json!({"prompt_tokens": prompt.len(), "ids": ids, "finish": finish, "prefill_signature": TOY_PREFILL_SIGNATURE})
        );
    }
    // The single-request form is unchanged: one pretty-printed document.
    let first = stdin.lines().next().unwrap();
    let one = ok(run_with_stdin(&single, first.as_bytes()));
    assert!(
        String::from_utf8_lossy(&one.stdout).lines().count() > 1,
        "{one:?}"
    );

    let good = r#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 9}"#;
    for (stdin, class, line) in [
        (format!("{good}\nnot json\n{good}\n"), "input", Some(2)),
        (format!("{good}\n\n{good}\n"), "input", Some(2)),
        (
            format!("{good}\n{good}\n{{\"prompt_ids\": [1, 2, 3], \"max_new_tokens\": 2046}}\n"),
            "context_length_exceeded",
            Some(3),
        ),
        (
            format!("{{\"prompt_ids\": [], \"max_new_tokens\": 1}}\n{good}\n"),
            "input",
            Some(1),
        ),
        (
            format!("{good}\n{{\"prompt_ids\": [1], \"max_new_tokens\": 0}}"),
            "input",
            Some(2),
        ),
        (String::new(), "input", None),
    ] {
        let out = run_with_stdin(&args, stdin.as_bytes());
        assert_eq!(out.status.code(), Some(2), "{stdin:?}: {out:?}");
        assert!(out.stdout.is_empty(), "{stdin:?}: {out:?}");
        assert_eq!(class_of(&out), class, "{stdin:?}: {out:?}");
        if let Some(n) = line {
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(
                err.lines()
                    .next()
                    .unwrap()
                    .starts_with(&format!("{class}: line {n}: ")),
                "{stdin:?}: {err}"
            );
        }
    }

    for cmd in ["probe", "score", "logprobs"] {
        let mut args = model_args(cmd, &dir, &sha, &lib);
        args.push("--jsonl");
        let out = run_with_stdin(&args, good.as_bytes());
        assert_eq!(out.status.code(), Some(2), "{cmd}: {out:?}");
        assert_eq!(class_of(&out), "usage", "{cmd}: {out:?}");
    }
}

/// O105: a `generate --jsonl` line's `"levers"` table runs its request with that lever set (the hot swap between
/// requests; a line without it runs the model's own set), so one load answers the plain and a profile's identity
/// lines; every result line equals the one-request form's for its request (the toy's tokens do not depend on the
/// levers). Each table is checked before the model loads as `--levers` is, and may not change a restart-class lever
/// from the model's: a bad one refuses the run naming the line, nothing generated. The one-request form refuses
/// the field.
#[test]
fn generate_jsonl_lines_carry_their_levers() {
    let t = Tmp::new("clijsonllevers");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let single = model_args("generate", &dir, &sha, &lib);
    let mut args = single.clone();
    args.push("--jsonl");
    let profile = serde_json::json!({"k2_gemm": "small_m", "decode_step": "spec_graph",
                                     "verify_attention": "batched"});
    let prompts: [&[u32]; 4] = [&[1, 2, 3], &[5, 9, 11, 4], &[1, 2, 3], &[2]];
    let tables = [
        None,
        Some(profile.clone()),
        Some(serde_json::json!({})),
        Some(profile),
    ];
    let mut stdin = String::new();
    for (p, levers) in prompts.iter().zip(&tables) {
        let mut line = serde_json::json!({"prompt_ids": p, "max_new_tokens": 9});
        if let Some(levers) = levers {
            line["levers"] = levers.clone();
        }
        stdin.push_str(&line.to_string());
        stdin.push('\n');
    }
    let out = ok(run_with_stdin(&args, stdin.as_bytes()));
    assert!(out.stderr.is_empty(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), prompts.len(), "{text}");
    for (line, prompt) in lines.iter().zip(prompts) {
        let got: Value = serde_json::from_str(line).unwrap();
        let request = serde_json::json!({"prompt_ids": prompt, "max_new_tokens": 9}).to_string();
        assert_eq!(
            got,
            json(&ok(run_with_stdin(&single, request.as_bytes()))),
            "{prompt:?}"
        );
    }
    // A model created with a restart-class lever keeps it: a line may restate it, not drop it.
    let mut fp8 = args.clone();
    fp8.extend(["--levers", "k2_weights=fp8"]);
    let keeps = r#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 9, "levers": {"k2_weights": "fp8", "k2_gemm": "small_m"}}"#;
    let out = ok(run_with_stdin(&fp8, format!("{keeps}\n").as_bytes()));
    assert_eq!(String::from_utf8(out.stdout).unwrap().lines().count(), 1);

    let good = r#"{"prompt_ids": [1, 2, 3], "max_new_tokens": 9}"#;
    let with = |levers: &str| {
        format!(r#"{{"prompt_ids": [1, 2, 3], "max_new_tokens": 9, "levers": {levers}}}"#)
    };
    for (argv, levers, needle) in [
        (
            &args,
            r#"{"k3_gemm": "small_m"}"#,
            "levers k3_gemm: unknown lever",
        ),
        (
            &args,
            r#"{"k2_gemm": "tiny"}"#,
            "levers k2_gemm: unknown implementation \"tiny\"",
        ),
        (
            &args,
            r#"{"drafter": "mtp+lookup"}"#,
            "levers drafter: \"mtp+lookup\" needs speculative_draft_tokens = 4",
        ),
        (&args, r#""k2_gemm=small_m""#, "levers must be an object"),
        (&args, r#"{"k2_gemm": 1}"#, "levers must be an object"),
        (
            &args,
            r#"{"k2_weights": "fp8"}"#,
            "levers k2_weights: \"fp8\" is not the model's \"bf16\"",
        ),
        (
            &args,
            r#"{"moe_gemm": "k7"}"#,
            "levers moe_gemm: \"k7\" is not the model's",
        ),
        (
            &fp8,
            r#"{}"#,
            "levers k2_weights: \"bf16\" is not the model's \"fp8\"",
        ),
    ] {
        let stdin = format!("{good}\n{}\n", with(levers));
        let out = run_with_stdin(argv, stdin.as_bytes());
        assert_eq!(out.status.code(), Some(2), "{levers}: {out:?}");
        assert!(out.stdout.is_empty(), "{levers}: {out:?}");
        assert_eq!(class_of(&out), "input", "{levers}: {out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        let first = err.lines().next().unwrap();
        assert!(first.starts_with("input: line 2: "), "{levers}: {err}");
        assert!(first.contains(needle), "{levers}: {err}");
    }
    let out = run_with_stdin(&single, with(r#"{"k2_gemm": "small_m"}"#).as_bytes());
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(out.stdout.is_empty(), "{out:?}");
    assert_eq!(class_of(&out), "input", "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("levers is a generate --jsonl line field"),
        "{out:?}"
    );
}

#[test]
fn score_reports_the_first_divergence_and_the_top2_margin() {
    let t = Tmp::new("cliscore");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let prompt = [5u32, 9, 11];
    let (same, _) = toy_greedy(&prompt, 6);
    assert_eq!(same.len(), 6, "the fixture prompt hits no stop token");
    let mut diverged = same.clone();
    diverged[2] = (diverged[2] + 3) % FIXTURE_VOCAB;
    let doc = serde_json::json!({"prompts": [
        {"id": "p01", "prompt_ids": prompt, "reference_ids": same},
        {"id": "p02", "prompt_ids": prompt, "reference_ids": diverged},
    ]});
    let out = ok(run_with_stdin(
        &model_args("score", &dir, &sha, &lib),
        doc.to_string().as_bytes(),
    ));
    // The toy scores its token with logit 1 and the next ID with logit 0.
    let runner_up = (same[2] + 1) % FIXTURE_VOCAB;
    assert_eq!(
        json(&out),
        serde_json::json!({"prompts": [
            {"id": "p01", "first_divergence": null, "native_top2": null, "margin": null},
            {"id": "p02", "first_divergence": 2, "native_top2": [[same[2], 1.0], [runner_up, 0.0]],
             "margin": 1.0},
        ]})
    );
}

/// M2.7 R3 (m27 design note): `cere-native logprobs` PREFILLs prompt + reference[..i] on a fresh session per
/// position and reports the logprob of reference[i] (null outside the top N) plus the top N, greedy (null
/// sampling). The toy's logits are 8 - j at next + j (j < 8), so next + j has logprob -j - ln(sum_k e^-k).
#[test]
fn logprobs_reports_the_reference_logprob_and_top_n_per_position() {
    let t = Tmp::new("clilp");
    let (dir, sha) = verified_artifact(&t);
    let lib = toy_library();
    let prompt = [5u32, 9, 11];
    let (greedy, _) = toy_greedy(&prompt, 3);
    assert_eq!(greedy.len(), 3, "the fixture prompt hits no stop token");
    // Position 1 references next + 2 (inside the top 3); position 2 a token outside the toy's support.
    let mut reference = greedy.clone();
    reference[1] = (greedy[1] + 2) % FIXTURE_VOCAB;
    let forced = toy_tokens(reference[1], (prompt.len() + 2) as u64, 1, FIXTURE_VOCAB)[0];
    reference[2] = (forced + 9) % FIXTURE_VOCAB;
    let doc = serde_json::json!({"top_logprobs": 3, "prompts": [
        {"id": "p01", "prompt_ids": prompt, "reference_ids": reference},
    ]});
    let args = model_args("logprobs", &dir, &sha, &lib);
    let out = ok(run_with_stdin(&args, doc.to_string().as_bytes()));
    let lse: f64 = (0..8).map(|k| (-f64::from(k)).exp()).sum::<f64>().ln();
    let r = json(&out);
    let positions = r["prompts"][0]["positions"].as_array().unwrap();
    assert_eq!(r["prompts"][0]["id"], "p01");
    assert_eq!(positions.len(), 3);
    let mut context = prompt.to_vec();
    for (i, pos) in positions.iter().enumerate() {
        // The native next token after prompt + reference[..i] (teacher-forced, not the native continuation).
        let next = toy_tokens(
            *context.last().unwrap(),
            context.len() as u64,
            1,
            FIXTURE_VOCAB,
        )[0];
        let top = pos["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 3, "{pos}");
        for (j, entry) in top.iter().enumerate() {
            assert_eq!(
                entry[0].as_u64(),
                Some(u64::from((next + j as u32) % FIXTURE_VOCAB)),
                "{pos}"
            );
            assert!(
                (entry[1].as_f64().unwrap() + j as f64 + lse).abs() < 1e-5,
                "{pos}"
            );
        }
        let want = match i {
            0 => Some(-lse),
            1 => Some(-2.0 - lse),
            _ => None,
        };
        match want {
            Some(w) => assert!(
                (pos["reference_logprob"].as_f64().unwrap() - w).abs() < 1e-5,
                "{pos}"
            ),
            None => assert!(pos["reference_logprob"].is_null(), "{pos}"),
        }
        context.push(reference[i]);
    }

    // top_logprobs outside 1..=20, a bad prompt, or prompt + reference over the context: refused before any work.
    for doc in [
        serde_json::json!({"top_logprobs": 0, "prompts": [{"id": "a", "prompt_ids": [1], "reference_ids": [2]}]}),
        serde_json::json!({"top_logprobs": 21, "prompts": [{"id": "a", "prompt_ids": [1], "reference_ids": [2]}]}),
        serde_json::json!({"top_logprobs": 3, "prompts": [{"id": "a", "prompt_ids": [], "reference_ids": [2]}]}),
        serde_json::json!({"top_logprobs": 3, "prompts": [{"id": "a", "prompt_ids": [1], "reference_ids": []}]}),
        serde_json::json!({"top_logprobs": 3, "prompts": [{"id": "a", "prompt_ids": [1, 2, 3],
                           "reference_ids": vec![1u32; 2046]}]}),
    ] {
        let out = run_with_stdin(&args, doc.to_string().as_bytes());
        assert_eq!(out.status.code(), Some(2), "{doc}: {out:?}");
        assert!(out.stdout.is_empty(), "{doc}: {out:?}");
    }
}

/// One `detok --check` frame: 4-byte little-endian length + JSON.
fn frame(v: &Value) -> Vec<u8> {
    let body = v.to_string().into_bytes();
    let mut out = (body.len() as u32).to_le_bytes().to_vec();
    out.extend(body);
    out
}

#[test]
fn detok_check_counts_mismatched_frames() {
    let fixtures = root().join("tests/fixtures/native");
    let cases: Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("detok_cases.json")).unwrap()).unwrap();
    let tokenizer = fixtures.join(cases["tokenizer"].as_str().unwrap());
    let all = cases["cases"].as_array().unwrap();
    let frames = |altered: Option<usize>| -> Vec<u8> {
        let mut out = Vec::new();
        for (i, case) in all.iter().enumerate() {
            let mut sha = case["text_sha256"].as_str().unwrap().to_string();
            if altered == Some(i) {
                sha = sha.chars().rev().collect();
            }
            out.extend(frame(
                &serde_json::json!({"ids": case["ids"], "text_sha256": sha}),
            ));
        }
        out
    };
    let args = [
        "detok",
        "--check",
        "--tokenizer",
        tokenizer.to_str().unwrap(),
    ];
    let r = json(&ok(run_with_stdin(&args, &frames(None))));
    assert_eq!(
        r,
        serde_json::json!({"frames": all.len(), "mismatches": 0, "mismatched_frames": []})
    );
    let r = json(&ok(run_with_stdin(&args, &frames(Some(7)))));
    assert_eq!(
        r,
        serde_json::json!({"frames": all.len(), "mismatches": 1, "mismatched_frames": [7]})
    );
    // A truncated frame is an input error, not a count.
    let mut torn = frames(None);
    torn.truncate(torn.len() - 1);
    let out = run_with_stdin(&args, &torn);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(out.stdout.is_empty());
}

/// `--keep-special` decodes the special tokens into the text (cered's Inkling text keeps them).
#[test]
fn detok_check_keep_special_keeps_the_special_tokens() {
    let fixtures = root().join("tests/fixtures/native");
    let tokenizer = fixtures.join("tokenizer_mini.json");
    let mut stdin = frame(&serde_json::json!({"ids": [248045],
        "text_sha256": "17a9ea43212df48ca4fca436c080e1e35cded043b9197b44a4bd53fbb04f3193"}));
    stdin.extend(frame(&serde_json::json!({"ids": [87, 248046],
        "text_sha256": "0a9c31b133489259ce0c7b0cd9f8c0ddfe978a4a727e098ae256059f040dd75d"})));
    let path = tokenizer.to_str().unwrap();
    let kept = json(&ok(run_with_stdin(
        &["detok", "--check", "--keep-special", "--tokenizer", path],
        &stdin,
    )));
    assert_eq!(
        kept,
        serde_json::json!({"frames": 2, "mismatches": 0, "mismatched_frames": []})
    );
    let skipped = json(&ok(run_with_stdin(
        &["detok", "--check", "--tokenizer", path],
        &stdin,
    )));
    assert_eq!(
        skipped,
        serde_json::json!({"frames": 2, "mismatches": 2, "mismatched_frames": [0, 1]})
    );
}
