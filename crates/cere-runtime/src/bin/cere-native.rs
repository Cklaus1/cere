//! `cere-native`: the native executor tool (spec m22 R3, Q10, Q13).
//!
//! `cere-native probe` opens a prepared artifact (stamp mode, after evicting it for `--cache cold`), loads the executor
//! library (SHA-256 of the file before `dlopen`), uploads and seals the model through the driver thread, runs a
//! 1-token greedy SMOKE on a fixed token sequence, releases everything and reports stage timings, the M2.1 residency
//! label, byte counters, model bytes per placement and the capabilities. A library whose forward pass is not built
//! reports SMOKE `unsupported` (exit 0, `ready: false`), never `passed`.
//!
//! `cere-native generate` and `score` (spec m22 R5, the X8/X9 diagnostics) load the model the same way and read
//! one JSON document of token IDs from stdin: `generate` runs greedy PREFILL + DECODE and writes the committed IDs,
//! `score` runs each prompt for as many tokens as its reference and writes the first index where the native greedy
//! token differs from the reference (until then greedy decoding *is* teacher forcing) with the native top-2 there.
//! `logprobs` (m27 R3) PREFILLs prompt + reference[..i] on a fresh session per position and writes reference[i]'s
//! logprob (null outside the top N) and the top N logprobs there.
//! `generate --preempt-every N` (M2.11 D13) preempts the session in-process after every N committed tokens:
//! snapshot into the executor's pinned arena (sized for one snapshot at `--max-context-tokens`), release, create,
//! restore, continue; it reports the preemption count beside the ids. Every `generate` result carries
//! `prefill_signature` (BC1 leftover, batch-concurrency design note: `cere_model_prefill_signature` of the loaded
//! model as lowercase hex, null when the library reports none), as `probe` does in `capabilities`.
//! `generate --jsonl` (O60) loads the model once and answers one request per stdin line with one compact result per
//! stdout line, in order, each written when its request ends. Every line is parsed and checked against the context
//! before the model loads; a malformed line refuses the run (`input: line N: ...`, exit 2, nothing generated). Each
//! request runs on a fresh session, created (the executor zeroes its slot: recurrent state, n-gram history,
//! position 0; fresh KV pages) and released, as in the one-request form.
//! `generate --levers NAME=ID[,NAME=ID...]` (m23 design note PI-02, D9) creates the model with that lever table
//! (absent levers take the registry default; absent option = the default set), checked as cered checks
//! `[deployments.native.levers]` for a model without drafts: an unknown lever or implementation, or a speculation
//! lever that needs the drafts, is a usage error before anything loads. The D9 token identity of a profile whose
//! levers change numerics (`native_parity.py --identity --native-config`) runs through it.
//! A `generate --jsonl` line may carry `"levers": {NAME: ID, ...}` (O105): its request runs with that lever table
//! (absent levers take the registry default; a line without it runs the model's own set, `--levers` or the defaults),
//! set between requests by the executor's hot swap (`cere_model_set_selection`) only when it differs from the active
//! set. Each line's table is checked as `--levers` is, before the model loads, and may not change a restart-class
//! lever (`moe_gemm`, `k2_weights`) from the model's: the hot swap cannot. One load then answers the plain and a
//! profile's identity lines (`native_parity.py --identity --with-plain`). The one-request form refuses the field.
//! `cere-native detok --check` decodes `{ids, text_sha256}` frames (Q9 framing) with a `tokenizer.json` and writes
//! the mismatch count; `--keep-special` decodes with the special tokens kept, as cered's Inkling text does
//! (`_patch_reasoning_skip_special_tokens`). No subcommand reads or writes text.
//!
//! Exit status: 0 for a report (SMOKE `passed` or `unsupported`); 2 for a rejected input (usage, manifest digest,
//! artifact contract, library hash or ABI, model desc, malformed stdin, a prompt over the context); 1 for an I/O or
//! executor failure, including a SMOKE that fails (its report is still printed). Every failure prints
//! `<class>: <message>` as the first stderr line.
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cere_compiler::artifact::{self, Artifact, Cache, Class, Mode, ReadMode};
use cere_compiler::{fsio, hex, sha256_hex, Error, Limits, EXIT_IO, EXIT_REJECT};
use cere_runtime::levers::{self, LeverClass, LeverSet};
use cere_runtime::native::detok::Vocab;
use cere_runtime::native::driver::{
    Driver, DriverError, ExecutorConfig, Finish, JobResult, Model, ModelSpec, Session, WorkRequest,
};
use cere_runtime::native::ffi::{self, ExecutionCapabilities, Library, LoadError, StatusError};
use cere_runtime::native::snapshot::arena_slot_bytes;
use cere_runtime::native::upload::{self, PlanOptions, UploadError, UploadOptions};
use serde_json::{json, Value};

const USAGE: &str = "\
cere-native probe --artifact DIR --expected-manifest-sha256 H --library L --cache cold|warm [--json]
                  [--library-sha256 H] [--device-id ID] [--memory-budget-bytes N] [--max-context-tokens N] [--rows N]
                  [--kv-pool-tokens N (0 = derived)] [--speculative-draft-tokens 0|4 (probe only: the MTP draft too)]
                  [--read buffered|direct|auto (how the upload reads tensor data; tasks/m216-spec.md L7)]
cere-native generate|score|logprobs <the probe options but --speculative-draft-tokens>
    generate  stdin {\"prompt_ids\": [..], \"max_new_tokens\": N}
              stdout {\"prompt_tokens\": N, \"ids\": [..], \"finish\": \"stop_token\"|\"length\"}
              [--preempt-every N]: snapshot the session after every N committed tokens, release it, restore the
              snapshot into a fresh session and continue (M2.11 D13); stdout then also has \"preemptions\": K
              [--jsonl]: stdin one request per line, stdout one compact result per line, one model load (O60);
              a line's optional \"levers\": {NAME: ID, ..} runs it with that lever table (hot swap, O105)
              [--levers NAME=ID[,NAME=ID...]]: create the model with these levers (tasks/m23-spec.md PI-02; no
              drafts, so drafter, draft_tokens and draft_policy keep their defaults)
    score     stdin {\"prompts\": [{\"id\": S, \"prompt_ids\": [..], \"reference_ids\": [..]}, ..]}
              stdout {\"prompts\": [{\"id\": S, \"first_divergence\": K|null,
                                    \"native_top2\": [[id, logit], [id, logit]]|null, \"margin\": X|null}, ..]}
    logprobs  stdin {\"top_logprobs\": N (1..=20), \"prompts\": [{\"id\": S, \"prompt_ids\": [..], \"reference_ids\": [..]}, ..]}
              stdout {\"prompts\": [{\"id\": S, \"positions\": [{\"reference_logprob\": X|null,
                                    \"top_logprobs\": [[id, logprob], ..]}, ..]}, ..]}
cere-native detok --check [--keep-special] --tokenizer FILE
    stdin {\"ids\": [..], \"text_sha256\": H} frames (4-byte little-endian length + JSON) until EOF
    stdout {\"frames\": N, \"mismatches\": M, \"mismatched_frames\": [..]}
cere-native --help | --version";

/// The SMOKE prompt: a fixed token sequence, one greedy token generated after it.
const SMOKE_PROMPT: [u32; 4] = [1, 2, 3, 4];
/// `cere_executor_options.device_id` default: the first device visible to the process (`--gpus device=<GPU0>`).
const DEFAULT_DEVICE_ID: &str = "0";
/// Q12: the executor's device bytes on GPU0.
const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 80 << 30;
/// R6: the context of the first native deployment.
const DEFAULT_MAX_CONTEXT_TOKENS: u64 = 2048;
const WORKER_EPOCH: u64 = 1;

fn usage(msg: impl Into<String>) -> Error {
    Error::reject("usage", format!("{}\n{USAGE}", msg.into()))
}

struct Options {
    artifact: PathBuf,
    expected_manifest_sha256: String,
    library: PathBuf,
    library_sha256: Option<String>,
    cache: Cache,
    json: bool,
    executor: ExecutorConfig,
    max_context_tokens: u64,
    /// `generate --preempt-every N` (M2.11 D13).
    preempt_every: Option<u64>,
    /// `generate --jsonl` (O60): one request per stdin line, one result per stdout line.
    jsonl: bool,
    /// `generate --levers` (m23 design note PI-02): the model's lever set; `None` = the registry defaults.
    levers: Option<LeverSet>,
    /// `--rows`: the plan's `max_sessions`, the executor's decode rows (m26 design note D9).
    rows: u64,
    /// `cere_model_desc.kv_pool_tokens`; 0 = the derived pool (m25 design note D4).
    kv_pool_tokens: u64,
    /// `probe --speculative-draft-tokens 4` (m23 design note P8): the plan with the MTP draft, so the probe measures
    /// a speculation profile's memory; 0 (default) = the target only (m29 design note D11).
    speculative_draft_tokens: u32,
    /// `--read`: how the upload's staging pass reads tensor data (m216 design note L7).
    read: ReadMode,
}

impl Options {
    fn parse(raw: &[String]) -> Result<Options, Error> {
        const VALUED: [&str; 14] = [
            "artifact",
            "expected-manifest-sha256",
            "library",
            "library-sha256",
            "cache",
            "device-id",
            "memory-budget-bytes",
            "max-context-tokens",
            "preempt-every",
            "rows",
            "kv-pool-tokens",
            "speculative-draft-tokens",
            "levers",
            "read",
        ];
        let mut values = BTreeMap::new();
        let mut json = false;
        let mut jsonl = false;
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            let Some(name) = a.strip_prefix("--") else {
                return Err(usage(format!("unexpected argument {a:?}")));
            };
            if name == "json" {
                json = true;
            } else if name == "jsonl" {
                jsonl = true;
            } else if VALUED.contains(&name) {
                let v = it
                    .next()
                    .ok_or_else(|| usage(format!("--{name} needs a value")))?;
                if values.insert(name, v.clone()).is_some() {
                    return Err(usage(format!("--{name} given twice")));
                }
            } else {
                return Err(usage(format!("unknown option --{name}")));
            }
        }
        let mut required = |name: &str| {
            values
                .remove(name)
                .ok_or_else(|| usage(format!("--{name} is required")))
        };
        let artifact = PathBuf::from(required("artifact")?);
        let expected_manifest_sha256 = required("expected-manifest-sha256")?;
        let library = PathBuf::from(required("library")?);
        let cache = required("cache")?;
        let cache = Cache::parse(&cache).ok_or_else(|| usage("--cache must be cold or warm"))?;
        let number = |name: &str, default: u64| -> Result<u64, Error> {
            match values.get(name) {
                None => Ok(default),
                Some(v) => v
                    .parse::<u64>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| usage(format!("--{name} must be a positive integer"))),
            }
        };
        let memory_budget_bytes = number("memory-budget-bytes", DEFAULT_MEMORY_BUDGET_BYTES)?;
        let max_context_tokens = number("max-context-tokens", DEFAULT_MAX_CONTEXT_TOKENS)?;
        let preempt_every = match values.contains_key("preempt-every") {
            true => Some(number("preempt-every", 0)?),
            false => None,
        };
        let rows = number("rows", 1)?;
        let kv_pool_tokens = match values.get("kv-pool-tokens") {
            None => 0,
            Some(v) => v
                .parse::<u64>()
                .map_err(|_| usage("--kv-pool-tokens must be a non-negative integer"))?,
        };
        let speculative_draft_tokens = match values.get("speculative-draft-tokens") {
            None => 0,
            Some(v) => v
                .parse::<u32>()
                .ok()
                .filter(|n| cere_runtime::config::SPECULATIVE_DRAFT_TOKENS.contains(n))
                .ok_or_else(|| usage("--speculative-draft-tokens must be 0 or 4"))?,
        };
        let library_sha256 = values.remove("library-sha256");
        if let Some(h) = &library_sha256 {
            if h.len() != 64
                || !h
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(usage("--library-sha256 must be 64 lowercase hex digits"));
            }
        }
        // `generate` loads the target only (`without_drafts`), so the table is checked against a model without drafts.
        let levers = match values.get("levers") {
            None => None,
            Some(v) => Some(
                LeverSet::from_table(&lever_table(v)?, 0)
                    .map_err(|e| usage(format!("--levers {e}")))?,
            ),
        };
        let device_id = values
            .remove("device-id")
            .unwrap_or_else(|| DEFAULT_DEVICE_ID.to_string());
        let read = match values.get("read") {
            None => UploadOptions::default().read,
            Some(v) => ReadMode::parse(v)
                .ok_or_else(|| usage("--read must be buffered, direct or auto"))?,
        };
        Ok(Options {
            artifact,
            expected_manifest_sha256,
            library,
            library_sha256,
            cache,
            json,
            executor: ExecutorConfig {
                device_id,
                worker_epoch: WORKER_EPOCH,
                memory_budget_bytes,
            },
            max_context_tokens,
            preempt_every,
            jsonl,
            levers,
            rows,
            kv_pool_tokens,
            speculative_draft_tokens,
            read,
        })
    }

    /// `probe`, `score` and `logprobs`: `--preempt-every`, `--jsonl` and `--levers` are `generate`'s only.
    fn parse_without_generate_options(raw: &[String]) -> Result<Options, Error> {
        let o = Options::parse(raw)?;
        if o.preempt_every.is_some() {
            return Err(usage("--preempt-every is a generate option"));
        }
        if o.jsonl {
            return Err(usage("--jsonl is a generate option"));
        }
        if o.levers.is_some() {
            return Err(usage("--levers is a generate option"));
        }
        Ok(o)
    }

    /// `generate`, `score` and `logprobs` load the target only: `--speculative-draft-tokens` is `probe`'s.
    fn without_drafts(self) -> Result<Options, Error> {
        if self.speculative_draft_tokens != 0 {
            return Err(usage("--speculative-draft-tokens is a probe option"));
        }
        Ok(self)
    }
}

/// `--levers NAME=ID[,NAME=ID...]` as a lever table; an item without `=`, an empty name or id, or a lever named twice
/// is a usage error (the names and ids are checked by `LeverSet::from_table`).
fn lever_table(text: &str) -> Result<BTreeMap<String, String>, Error> {
    let mut table = BTreeMap::new();
    for item in text.split(',') {
        let (name, id) = item
            .split_once('=')
            .filter(|(n, i)| !n.is_empty() && !i.is_empty())
            .ok_or_else(|| usage(format!("--levers item {item:?} is not NAME=ID")))?;
        if table.insert(name.to_owned(), id.to_owned()).is_some() {
            return Err(usage(format!("--levers names {name} twice")));
        }
    }
    Ok(table)
}

fn load_error(e: LoadError) -> Error {
    let exit = match e {
        LoadError::Io { .. } | LoadError::Open(_) => EXIT_IO,
        LoadError::HashMismatch { .. }
        | LoadError::MissingSymbol(_)
        | LoadError::AbiVersion { .. } => EXIT_REJECT,
    };
    Error {
        class: e.class(),
        exit,
        message: e.to_string(),
    }
}

fn driver_error(e: DriverError) -> Error {
    let class = match &e {
        DriverError::Status { status, .. } => status.name(),
        DriverError::Argument(_) => "argument",
        DriverError::Missing(_) => "missing_symbol",
        DriverError::Stopped => "driver_stopped",
    };
    Error {
        class,
        exit: EXIT_IO,
        message: e.to_string(),
    }
}

fn upload_error(e: UploadError) -> Error {
    match e {
        UploadError::Plan { class, message } => Error::reject(class, message),
        UploadError::Driver(e) => driver_error(e),
        UploadError::Stage(e) => e,
    }
}

/// `mincore` residency of the pages holding the staged tensors (device and host classes; `unused` excluded).
fn residency(art: &Artifact) -> Result<fsio::Residency, Error> {
    let mut r = art.residency(Some(Class::Device))?;
    r.add(art.residency(Some(Class::Host))?);
    Ok(r)
}

fn execution_mode(mode: u32) -> &'static str {
    match mode {
        ffi::EXECUTION_MODE_NATIVE => "native",
        ffi::EXECUTION_MODE_BACKEND_MANAGED => "backend_managed",
        ffi::EXECUTION_MODE_SERIAL => "serial",
        ffi::EXECUTION_MODE_UNSUPPORTED => "unsupported",
        _ => "unknown",
    }
}

/// The sealed model's capabilities and, BC1 leftover (batch-concurrency design note "Prefill selection
/// signature"), its prefill selection signature (`cere_model_prefill_signature`, lowercase hex; null when the
/// library reports none).
fn capabilities_json(c: &ExecutionCapabilities, prefill_signature: Option<&[u8; 32]>) -> Value {
    let operations: Vec<&str> = [
        (ffi::OPS_PREFILL, "prefill"),
        (ffi::OPS_DECODE, "decode"),
        (ffi::OPS_VERIFY, "verify"),
        (ffi::OPS_CANCEL, "cancel"),
    ]
    .into_iter()
    .filter(|(bit, _)| c.operations & bit != 0)
    .map(|(_, name)| name)
    .collect();
    json!({
        "execution_mode": execution_mode(c.execution_mode),
        "operations": operations,
        "profile_generation": c.profile_generation,
        "worker_epoch": c.worker_epoch,
        "decode_batch_min": c.decode_batch_min,
        "decode_batch_max": c.decode_batch_max,
        "max_context_tokens": c.max_context_tokens,
        "kv_pool_tokens": c.kv_pool_tokens,
        "kv_page_tokens": c.kv_page_tokens,
        "native_decode_batch": c.native_decode_batch != 0,
        "native_mixed_prefill_decode": c.native_mixed_prefill_decode != 0,
        "native_speculative_batch": c.native_speculative_batch != 0,
        "native_moe_grouping": c.native_moe_grouping != 0,
        "native_state_snapshot": c.native_state_snapshot != 0,
        "prefill_signature": prefill_signature.map(|d| hex(d)),
    })
}

/// Bytes the model desc places on the device, in pinned host memory and discards.
fn model_bytes(spec: &ModelSpec) -> Value {
    let sum = |p: u32| -> u64 {
        spec.tensors
            .iter()
            .filter(|t| t.placement == p)
            .map(|t| t.bytes)
            .sum()
    };
    json!({
        "device": sum(ffi::PLACEMENT_DEVICE),
        "host_pinned": sum(ffi::PLACEMENT_HOST_PINNED),
        "discard": sum(ffi::PLACEMENT_DISCARD),
    })
}

/// SMOKE outcome: `passed` with the token, `unsupported` (typed refusal of the forward pass) or `failed`.
struct Smoke {
    status: &'static str,
    token: Option<u32>,
    finish: Option<&'static str>,
    reason: Option<String>,
}

impl Smoke {
    fn unsupported(reason: String) -> Smoke {
        Smoke {
            status: "unsupported",
            token: None,
            finish: None,
            reason: Some(reason),
        }
    }

    fn failed(reason: String) -> Smoke {
        Smoke {
            status: "failed",
            token: None,
            finish: None,
            reason: Some(reason),
        }
    }
}

/// One greedy token after [`SMOKE_PROMPT`] on a fresh session, which is released afterwards.
fn smoke(driver: &Driver, up: &upload::Upload, max_context_tokens: u64) -> Result<Smoke, Error> {
    if up.capabilities.operations & ffi::OPS_PREFILL == 0 {
        return Ok(Smoke::unsupported(format!(
            "the library reports no PREFILL operation (execution_mode {})",
            execution_mode(up.capabilities.execution_mode)
        )));
    }
    let session = driver
        .create_session(&up.model, max_context_tokens)
        .map_err(driver_error)?;
    let outcome = driver
        .submit(&session, WorkRequest::prefill(SMOKE_PROMPT.to_vec(), 1))
        .and_then(|job| job.wait());
    let released = driver.release_session(session);
    let smoke = match outcome {
        Err(e) if e.status() == Some(StatusError::Unsupported) => Smoke::unsupported(e.to_string()),
        Err(e) => Smoke::failed(e.to_string()),
        Ok(r) => match (r.status, r.tokens.as_slice()) {
            (Err(StatusError::Unsupported), _) => {
                Smoke::unsupported("the forward pass ended with unsupported".to_string())
            }
            (Err(status), _) => {
                Smoke::failed(format!("the forward pass ended with {}", status.name()))
            }
            (Ok(()), [token]) => Smoke {
                status: "passed",
                token: Some(*token),
                finish: Some(r.finish.name()),
                reason: None,
            },
            (Ok(()), tokens) => {
                Smoke::failed(format!("{} tokens committed, 1 requested", tokens.len()))
            }
        },
    };
    released.map_err(driver_error)?;
    Ok(smoke)
}

/// DISCOVER/VALIDATE after the `--cache cold` eviction: a wrong manifest digest never loads the library. Returns the
/// evicted-file count and the time `open` took (the eviction is a measurement pre-pass, not part of the load).
fn open_artifact(o: &Options) -> Result<(Artifact, usize, Duration), Error> {
    let limits = Limits::default();
    let evicted = match o.cache {
        Cache::Cold => {
            artifact::evict_dir(&o.artifact, Some(&o.expected_manifest_sha256), &limits)?.0
        }
        Cache::Warm => 0,
    };
    let t = Instant::now();
    let art = artifact::open(&o.artifact, &o.expected_manifest_sha256, Mode::Stamp)?;
    Ok((art, evicted, t.elapsed()))
}

/// The artifact's model desc for `--rows` sessions of `--max-context-tokens`; with `--preempt-every`, a snapshot
/// arena that holds one snapshot of that session at its largest (M2.11 D5, the kinds this build writes: kind 10 too
/// with drafts).
fn spec_of(o: &Options, art: &Artifact) -> Result<ModelSpec, Error> {
    let snapshot_host_bytes = match o.preempt_every {
        None => 0,
        Some(_) => arena_slot_bytes(o.max_context_tokens, o.speculative_draft_tokens > 0)
            .ok_or_else(|| usage("--max-context-tokens is too large for a snapshot arena"))?,
    };
    let mut spec = upload::model_spec(
        art,
        &o.artifact,
        &PlanOptions {
            max_sessions: o.rows,
            max_context_tokens: o.max_context_tokens,
            snapshot_host_bytes,
            kv_pool_tokens: o.kv_pool_tokens,
            // The CLI's runs never namespace a session, so a cache would never be used (M2.10 D2).
            prefix_state_slots: 0,
            // The target only unless `probe --speculative-draft-tokens 4` (m29 design note D11 default; P8).
            speculative_draft_tokens: o.speculative_draft_tokens,
        },
    )
    .map_err(upload_error)?;
    // m23 design note PI-03: `generate --levers`, else the library defaults (as `model_spec` leaves them).
    if let Some(levers) = o.levers {
        spec.levers = levers;
    }
    Ok(spec)
}

/// Loads the library (hash-checked before `dlopen`) and starts the driver thread; the library's report entry too.
fn start_driver(o: &Options) -> Result<(Driver, Value), Error> {
    let lib = Library::open(&o.library, o.library_sha256.as_deref()).map_err(load_error)?;
    let library = json!({
        "path": lib.path(),
        "sha256": lib.sha256(),
        "abi_version": lib.abi_version(),
    });
    let driver = Driver::start(Arc::new(lib), &o.executor).map_err(driver_error)?;
    Ok((driver, library))
}

/// The report, and whether SMOKE failed. `stages_ms.ready` is load-to-ready (m216 design note): open, the model
/// spec, the executor start, the upload and the smoke, without the probe's own eviction and residency passes and
/// without the release after it (`release`).
fn probe(o: &Options) -> Result<(Value, bool), Error> {
    let total = Instant::now();
    let (art, evicted, open) = open_artifact(o)?;
    let before = residency(&art)?;
    let t = Instant::now();
    let spec = spec_of(o, &art)?;
    let spec_time = t.elapsed();
    let t = Instant::now();
    let (driver, library) = start_driver(o)?;
    let executor = t.elapsed();
    let staging = UploadOptions {
        read: o.read,
        ..UploadOptions::default()
    };
    let t = Instant::now();
    let up = upload::upload(&driver, &art, &spec, &staging).map_err(upload_error)?;
    let upload_time = t.elapsed();
    let after = residency(&art)?;

    let t = Instant::now();
    let smoke = smoke(&driver, &up, o.max_context_tokens);
    let smoke_time = t.elapsed();
    let smoke_ms = smoke_time.as_millis() as u64;
    let ready = open + spec_time + executor + upload_time + smoke_time;
    let t = Instant::now();
    let released = driver.release_model(up.model);
    let shutdown = driver.shutdown();
    let release_ms = t.elapsed().as_millis() as u64;
    let smoke = smoke?;
    released.map_err(driver_error)?;
    shutdown.map_err(driver_error)?;

    let timings = art.timings();
    let counters = art.counters();
    let s = &up.staging;
    let peak_rss = fsio::peak_rss().map_err(|e| Error::io("reading VmHWM", e))?;
    let report = json!({
        "outcome": "probed",
        "artifact": o.artifact,
        "artifact_id": art.artifact_id(),
        "manifest_sha256": art.manifest_sha256,
        "library": library,
        "device_id": o.executor.device_id,
        "worker_epoch": o.executor.worker_epoch,
        "memory_budget_bytes": o.executor.memory_budget_bytes,
        "max_context_tokens": o.max_context_tokens,
        "kv_pool_tokens": o.kv_pool_tokens,
        "rows": o.rows,
        "speculative_draft_tokens": o.speculative_draft_tokens,
        "cache": o.cache.name(),
        "label": artifact::cache_label(o.cache, before),
        "evicted_files": evicted,
        "residency_before": before,
        "residency_after": after,
        "stages_ms": {
            "discover": timings.discover_ms,
            "validate": timings.validate_ms,
            "allocate": up.allocate_ms,
            "read_upload": up.upload_ms,
            "upload_put": s.put_ms,
            "upload_device_put": s.device_put_ms,
            "resident_check": s.resident_ms,
            "instantiate": up.instantiate_ms,
            "smoke": smoke_ms,
            "executor": executor.as_millis() as u64,
            "ready": ready.as_millis() as u64,
            "release": release_ms,
            "total": total.elapsed().as_millis() as u64,
        },
        "tensors": spec.tensors.len(),
        "uploaded_device_bytes": s.counters.uploaded_device_bytes,
        "uploaded_host_bytes": s.counters.uploaded_host_bytes,
        "discarded_bytes": s.counters.discarded_bytes,
        "resident_host_bytes": s.counters.resident_host_bytes,
        "resident_files": s.resident_files,
        "payload_bytes_read": counters.payload_bytes_read,
        "metadata_bytes_read": counters.metadata_bytes_read,
        "model_bytes": model_bytes(&spec),
        "staging": {
            "buffers": staging.buffers,
            "buffer_bytes": staging.buffer_bytes,
            "readers": staging.readers,
            "consumers": staging.consumers,
            "registered_bytes": s.registered_bytes,
            "phases_ms": s.phases_ms,
            "pass": s.pass,
        },
        "stop_token_ids": spec.stop_token_ids,
        "capabilities": capabilities_json(&up.capabilities, up.prefill_signature.as_ref()),
        "smoke": {
            "status": smoke.status,
            "prompt": SMOKE_PROMPT,
            "token": smoke.token,
            "finish": smoke.finish,
            "reason": smoke.reason,
        },
        "ready": smoke.status == "passed",
        "released": true,
        "peak_rss_bytes": peak_rss,
    });
    Ok((report, smoke.status == "failed"))
}

/// The human-readable report.
fn text(r: &Value) -> String {
    let mut out = String::new();
    let mut line = |k: &str, v: String| out.push_str(&format!("{k}: {v}\n"));
    line("manifest_sha256", r["manifest_sha256"].to_string());
    line(
        "library",
        format!("{} sha256 {}", r["library"]["path"], r["library"]["sha256"]),
    );
    line(
        "cache",
        format!(
            "{} (label {}, evicted_files {})",
            r["cache"], r["label"], r["evicted_files"]
        ),
    );
    for k in [
        "uploaded_device_bytes",
        "uploaded_host_bytes",
        "discarded_bytes",
        "resident_host_bytes",
        "stages_ms",
        "capabilities",
    ] {
        line(k, r[k].to_string());
    }
    let smoke = &r["smoke"];
    let detail = match smoke["status"].as_str() {
        Some("passed") => format!("token {}", smoke["token"]),
        _ => smoke["reason"].to_string(),
    };
    line(
        "smoke",
        format!("{} ({detail})", smoke["status"].as_str().unwrap_or("?")),
    );
    line("ready", r["ready"].to_string());
    out
}

/// The largest stdin document `generate`/`score`/`logprobs` accept, and the largest `detok --check` frame.
const MAX_INPUT_BYTES: u64 = 64 << 20;
const MAX_FRAME_BYTES: u32 = 16 << 20;

fn input(message: impl Into<String>) -> Error {
    Error::reject("input", message)
}

/// Stdin, at most MAX_INPUT_BYTES.
fn read_stdin() -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::io("reading stdin", e))?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err(input(format!("stdin exceeds {MAX_INPUT_BYTES} bytes")));
    }
    Ok(bytes)
}

/// The one JSON document on stdin.
fn read_doc() -> Result<Value, Error> {
    serde_json::from_slice(&read_stdin()?)
        .map_err(|e| input(format!("stdin is not one JSON document: {e}")))
}

/// `e` with its message prefixed by the 1-based stdin line it belongs to (`generate --jsonl`).
fn at_line(line: usize, e: Error) -> Error {
    Error {
        message: format!("line {line}: {}", e.message),
        ..e
    }
}

/// `doc[key]` as a list of token IDs.
fn token_ids(doc: &Value, key: &str) -> Result<Vec<u32>, Error> {
    doc.get(key)
        .and_then(Value::as_array)
        .and_then(|ids| {
            ids.iter()
                .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                .collect::<Option<Vec<u32>>>()
        })
        .ok_or_else(|| input(format!("{key} must be a list of token IDs")))
}

/// Refuses an empty prompt, and a prompt whose output budget does not fit the context.
fn check_context(prompt: &[u32], max_new_tokens: u64, context: u64) -> Result<(), Error> {
    if prompt.is_empty() {
        return Err(input("the prompt has no tokens"));
    }
    if prompt.len() as u64 + max_new_tokens > context {
        return Err(Error::reject(
            "context_length_exceeded",
            format!(
                "{} prompt tokens + {max_new_tokens} new tokens exceed the context of {context}",
                prompt.len()
            ),
        ));
    }
    Ok(())
}

fn write_doc(doc: &Value) -> Result<(), Error> {
    std::io::stdout()
        .write_all(format!("{doc:#}\n").as_bytes())
        .map_err(|e| Error::io("writing stdout", e))
}

/// Uploads and seals the model, runs `f` on it and releases everything, whatever `f` returned.
fn with_model<T>(
    o: &Options,
    f: impl FnOnce(&Driver, &Model) -> Result<T, Error>,
) -> Result<T, Error> {
    let (art, _, _) = open_artifact(o)?;
    let spec = spec_of(o, &art)?;
    let (driver, _) = start_driver(o)?;
    let opts = UploadOptions {
        read: o.read,
        ..UploadOptions::default()
    };
    let up = upload::upload(&driver, &art, &spec, &opts).map_err(upload_error)?;
    let mut need = ffi::OPS_PREFILL | ffi::OPS_DECODE;
    if o.preempt_every.is_some() {
        need |= ffi::OPS_SNAPSHOT | ffi::OPS_RESTORE;
    }
    let out = if up.capabilities.operations & need == need {
        f(&driver, &up.model)
    } else {
        Err(Error {
            class: "unsupported",
            exit: EXIT_IO,
            message: format!(
                "the library reports operations {:#x}, {:#x} needed (execution_mode {})",
                up.capabilities.operations,
                need,
                execution_mode(up.capabilities.execution_mode)
            ),
        })
    };
    let released = driver.release_model(up.model);
    let shutdown = driver.shutdown();
    let out = out?;
    released.map_err(driver_error)?;
    shutdown.map_err(driver_error)?;
    Ok(out)
}

/// A terminal completion that ended without an error.
fn completed(r: Result<JobResult, DriverError>) -> Result<JobResult, Error> {
    let r = r.map_err(driver_error)?;
    match r.status {
        Ok(()) => Ok(r),
        Err(status) => Err(Error {
            class: status.name(),
            exit: EXIT_IO,
            message: format!("the forward pass ended with {}", status.name()),
        }),
    }
}

/// One preemption (M2.11 D13): snapshot the idle session, release it, restore the snapshot into a fresh session of
/// `context` tokens, release the snapshot. `slot` holds the live session afterwards (none when the release failed).
fn preempt(
    driver: &Driver,
    model: &Model,
    slot: &mut Option<Session>,
    context: u64,
) -> Result<(), Error> {
    let snapshot = driver
        .snapshot(slot.as_ref().expect("a live session"))
        .map_err(driver_error)?;
    let mut moved = || -> Result<(), Error> {
        driver
            .release_session(slot.take().expect("a live session"))
            .map_err(driver_error)?;
        let fresh = driver
            .create_session(model, context)
            .map_err(driver_error)?;
        let restored = driver.restore(&fresh, &snapshot).map_err(driver_error);
        *slot = Some(fresh);
        restored
    };
    let moved = moved();
    let released = driver.release_snapshot(snapshot).map_err(driver_error);
    moved?;
    released
}

/// Greedy decoding on a fresh session (released afterwards): PREFILL commits the first token, DECODE the rest. The
/// model's stop tokens end it; the result holds every committed token, with scores when asked for. With
/// `preempt_every` N, DECODE runs in steps that end at every multiple of N committed tokens, and the session is
/// preempted there ([`preempt`]) while the generation goes on; the second value counts the preemptions.
fn greedy(
    driver: &Driver,
    model: &Model,
    prompt: &[u32],
    max_new_tokens: u64,
    context: u64,
    scores: bool,
    preempt_every: Option<u64>,
) -> Result<(JobResult, u64), Error> {
    let mut slot = Some(
        driver
            .create_session(model, context)
            .map_err(driver_error)?,
    );
    let item = |w: WorkRequest| if scores { w.with_scores() } else { w };
    let mut preemptions = 0;
    let mut run = || -> Result<JobResult, Error> {
        let mut r = completed(
            driver
                .submit(
                    slot.as_ref().expect("a live session"),
                    item(WorkRequest::prefill(prompt.to_vec(), 1)),
                )
                .and_then(|job| job.wait()),
        )?;
        let mut done = r.tokens.len() as u64;
        while r.finish == Finish::Length && done < max_new_tokens {
            let mut step = max_new_tokens - done;
            if let Some(n) = preempt_every {
                if done % n == 0 {
                    preempt(driver, model, &mut slot, context)?;
                    preemptions += 1;
                }
                step = step.min(n - done % n);
            }
            let d = completed(
                driver
                    .submit(
                        slot.as_ref().expect("a live session"),
                        item(WorkRequest::decode(step)),
                    )
                    .and_then(|job| job.wait()),
            )?;
            let short = (d.tokens.len() as u64) < step;
            r.tokens.extend(d.tokens);
            r.scores.extend(d.scores);
            r.finish = d.finish;
            r.committed_tokens += d.committed_tokens;
            done = r.tokens.len() as u64;
            if short {
                break;
            }
        }
        Ok(r)
    };
    let out = run();
    let released = slot.map(|s| driver.release_session(s));
    let out = out?;
    if let Some(released) = released {
        released.map_err(driver_error)?;
    }
    Ok((out, preemptions))
}

/// One `generate` request: its prompt and `max_new_tokens`, refused unless it fits `context`.
fn generate_request(doc: &Value, context: u64) -> Result<(Vec<u32>, u64), Error> {
    let prompt = token_ids(doc, "prompt_ids")?;
    let max_new_tokens = doc
        .get("max_new_tokens")
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
        .ok_or_else(|| input("max_new_tokens must be a positive integer"))?;
    check_context(&prompt, max_new_tokens, context)?;
    Ok((prompt, max_new_tokens))
}

/// The result document of one request, greedy-decoded on a fresh session of the loaded model ([`greedy`]).
fn generate_one(
    o: &Options,
    driver: &Driver,
    model: &Model,
    prompt: &[u32],
    max_new_tokens: u64,
) -> Result<Value, Error> {
    let (r, preemptions) = greedy(
        driver,
        model,
        prompt,
        max_new_tokens,
        o.max_context_tokens,
        false,
        o.preempt_every,
    )?;
    // BC1 leftover (batch-concurrency design note "Prefill selection signature"): the loaded model's prefill
    // selection signature (null when the library reports none), so an identity report records the kernels it ran.
    let signature = driver.prefill_signature(model).map_err(driver_error)?;
    let mut doc = json!({
        "prompt_tokens": prompt.len(),
        "ids": r.tokens,
        "finish": r.finish.name(),
        "prefill_signature": signature.as_ref().map(|d| hex(d)),
    });
    if o.preempt_every.is_some() {
        doc["preemptions"] = json!(preemptions);
    }
    Ok(doc)
}

/// `generate --jsonl` (O105): the lever set of a request line's optional `levers` object (None: the model's own
/// set), checked as `--levers` is for a model without drafts; a restart-class lever must keep the model's
/// (`model`, the set it was created with): the hot swap cannot change it.
fn line_levers(doc: &Value, model: &LeverSet) -> Result<Option<LeverSet>, Error> {
    let Some(value) = doc.get("levers") else {
        return Ok(None);
    };
    let table = value
        .as_object()
        .and_then(|m| {
            m.iter()
                .map(|(name, id)| id.as_str().map(|id| (name.clone(), id.to_owned())))
                .collect::<Option<BTreeMap<String, String>>>()
        })
        .ok_or_else(|| input("levers must be an object of lever names to implementation ids"))?;
    let set = LeverSet::from_table(&table, 0).map_err(|e| input(format!("levers {e}")))?;
    for lever in levers::REGISTRY {
        if lever.class == LeverClass::Restart && set.get(lever.name) != model.get(lever.name) {
            return Err(input(format!(
                "levers {}: {:?} is not the model's {:?}, and a restart-class lever never changes between \
                 requests",
                lever.name,
                set.get(lever.name).unwrap_or_default(),
                model.get(lever.name).unwrap_or_default()
            )));
        }
    }
    Ok(Some(set))
}

/// `generate`: one stdin document, or with `--jsonl` one request per stdin line (every line checked before the
/// model loads) answered line by line on one model load, each with its line's levers (O105).
fn generate(o: &Options) -> Result<(), Error> {
    if !o.jsonl {
        let doc = read_doc()?;
        if doc.get("levers").is_some() {
            return Err(input(
                "levers is a generate --jsonl line field (the one-request form takes --levers)",
            ));
        }
        let (prompt, max_new_tokens) = generate_request(&doc, o.max_context_tokens)?;
        let doc = with_model(o, |driver, model| {
            generate_one(o, driver, model, &prompt, max_new_tokens)
        })?;
        return write_doc(&doc);
    }
    // The set the model is created with (spec_of): `--levers`, else the registry defaults.
    let created = o.levers.unwrap_or_default();
    let bytes = read_stdin()?;
    let text =
        std::str::from_utf8(&bytes).map_err(|e| input(format!("stdin is not UTF-8: {e}")))?;
    let requests = text
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str::<Value>(line)
                .map_err(|e| input(format!("not one JSON request: {e}")))
                .and_then(|doc| {
                    let (prompt, max_new_tokens) = generate_request(&doc, o.max_context_tokens)?;
                    let levers = line_levers(&doc, &created)?.unwrap_or(created);
                    Ok((prompt, max_new_tokens, levers))
                })
                .map_err(|e| at_line(i + 1, e))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if requests.is_empty() {
        return Err(input("stdin holds no request line"));
    }
    with_model(o, |driver, model| {
        let mut stdout = std::io::stdout().lock();
        let mut active = created;
        for (i, (prompt, max_new_tokens, levers)) in requests.iter().enumerate() {
            if *levers != active {
                driver
                    .set_levers(model, levers)
                    .map_err(|e| at_line(i + 1, driver_error(e)))?;
                active = *levers;
            }
            let doc = generate_one(o, driver, model, prompt, *max_new_tokens)
                .map_err(|e| at_line(i + 1, e))?;
            stdout
                .write_all(format!("{doc}\n").as_bytes())
                .and_then(|()| stdout.flush())
                .map_err(|e| Error::io("writing stdout", e))?;
        }
        Ok(())
    })
}

fn score(o: &Options) -> Result<Value, Error> {
    let doc = read_doc()?;
    let prompts = doc
        .get("prompts")
        .and_then(Value::as_array)
        .ok_or_else(|| input("prompts must be a list"))?;
    let mut jobs = Vec::with_capacity(prompts.len());
    for p in prompts {
        let id = p
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| input("every prompt needs a string id"))?;
        let prompt = token_ids(p, "prompt_ids")?;
        let reference = token_ids(p, "reference_ids")?;
        if reference.is_empty() {
            return Err(input(format!("{id}: reference_ids is empty")));
        }
        check_context(&prompt, reference.len() as u64, o.max_context_tokens)?;
        jobs.push((id, prompt, reference));
    }
    let rows = with_model(o, |driver, model| {
        jobs.iter()
            .map(|(id, prompt, reference)| {
                let (r, _) = greedy(
                    driver,
                    model,
                    prompt,
                    reference.len() as u64,
                    o.max_context_tokens,
                    true,
                    None,
                )?;
                // Until the first divergence the native greedy token is the reference token, so this run is the
                // teacher-forced one; a native run that ends early diverges where it ended (no scores there).
                let k = reference
                    .iter()
                    .zip(&r.tokens)
                    .position(|(want, got)| want != got)
                    .or_else(|| (r.tokens.len() < reference.len()).then_some(r.tokens.len()));
                let (top2, margin) = match k.and_then(|k| r.scores.get(k)) {
                    Some(s) => (
                        json!([[s.id, s.logit], [s.runner_up, s.runner_up_logit]]),
                        json!(s.logit - s.runner_up_logit),
                    ),
                    None => (Value::Null, Value::Null),
                };
                Ok(json!({
                    "id": id,
                    "first_divergence": k,
                    "native_top2": top2,
                    "margin": margin,
                }))
            })
            .collect::<Result<Vec<Value>, Error>>()
    })?;
    Ok(json!({ "prompts": rows }))
}

/// M2.7 R3 (m27 design note): teacher-forced logprobs. Per reference position i, a fresh session PREFILLs
/// prompt + reference[..i] and commits one greedy token with `top_logprobs` alternatives, so every position is
/// scored from prefill logits like SGLang's teacher-forced `/generate`. `reference_logprob` is reference[i]'s
/// logprob when it is the committed token or among the top N, else null (the executor reports no other ids).
fn logprobs(o: &Options) -> Result<Value, Error> {
    let doc = read_doc()?;
    let top = doc
        .get("top_logprobs")
        .and_then(Value::as_u64)
        .filter(|n| (1..=u64::from(ffi::MAX_TOP_LOGPROBS)).contains(n))
        .ok_or_else(|| {
            input(format!(
                "top_logprobs must be an integer in 1..={}",
                ffi::MAX_TOP_LOGPROBS
            ))
        })? as u32;
    let prompts = doc
        .get("prompts")
        .and_then(Value::as_array)
        .ok_or_else(|| input("prompts must be a list"))?;
    let mut jobs = Vec::with_capacity(prompts.len());
    for p in prompts {
        let id = p
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| input("every prompt needs a string id"))?;
        let prompt = token_ids(p, "prompt_ids")?;
        let reference = token_ids(p, "reference_ids")?;
        if reference.is_empty() {
            return Err(input(format!("{id}: reference_ids is empty")));
        }
        check_context(&prompt, reference.len() as u64, o.max_context_tokens)?;
        jobs.push((id, prompt, reference));
    }
    let rows = with_model(o, |driver, model| {
        jobs.iter()
            .map(|(id, prompt, reference)| {
                let mut context = prompt.clone();
                let mut positions = Vec::with_capacity(reference.len());
                for &want in reference {
                    let session = driver
                        .create_session(model, o.max_context_tokens)
                        .map_err(driver_error)?;
                    let r = completed(
                        driver
                            .submit(
                                &session,
                                WorkRequest::prefill(context.clone(), 1).with_logprobs(top),
                            )
                            .and_then(|job| job.wait()),
                    );
                    let released = driver.release_session(session);
                    let r = r?;
                    released.map_err(driver_error)?;
                    let lp = r.logprobs.first().ok_or_else(|| Error {
                        class: "internal",
                        exit: EXIT_IO,
                        message: format!("{id}: the PREFILL committed no token logprobs"),
                    })?;
                    let entries = &lp.top[..lp.top_count.min(ffi::MAX_TOP_LOGPROBS) as usize];
                    let reference_logprob = if lp.id == want {
                        Some(lp.logprob)
                    } else {
                        entries.iter().find(|e| e.id == want).map(|e| e.logprob)
                    };
                    positions.push(json!({
                        "reference_logprob": reference_logprob,
                        "top_logprobs": entries.iter().map(|e| json!([e.id, e.logprob])).collect::<Vec<_>>(),
                    }));
                    context.push(want);
                }
                Ok(json!({ "id": id, "positions": positions }))
            })
            .collect::<Result<Vec<Value>, Error>>()
    })?;
    Ok(json!({ "prompts": rows }))
}

/// `detok --check [--keep-special] --tokenizer FILE`.
fn detok_check(raw: &[String]) -> Result<Value, Error> {
    let (tokenizer, skip_special) = match raw {
        [check, flag, path] if check == "--check" && flag == "--tokenizer" => {
            (Path::new(path), true)
        }
        [check, keep, flag, path]
            if check == "--check" && keep == "--keep-special" && flag == "--tokenizer" =>
        {
            (Path::new(path), false)
        }
        _ => {
            return Err(usage(
                "detok needs --check [--keep-special] --tokenizer FILE",
            ))
        }
    };
    let bytes = std::fs::read(tokenizer)
        .map_err(|e| Error::io(format!("reading {}", tokenizer.display()), e))?;
    let vocab = Vocab::from_json(&bytes).map_err(|e| Error::reject("tokenizer", e.to_string()))?;
    let mut stdin = std::io::stdin().lock();
    let mut frames = 0u64;
    let mut mismatched = Vec::new();
    loop {
        let mut header = [0u8; 4];
        let got = read_full(&mut stdin, &mut header)?;
        if got == 0 {
            break;
        }
        if got < header.len() {
            return Err(input(format!("frame {frames}: truncated length")));
        }
        let len = u32::from_le_bytes(header);
        if len > MAX_FRAME_BYTES {
            return Err(input(format!(
                "frame {frames}: {len} bytes exceed {MAX_FRAME_BYTES}"
            )));
        }
        let mut body = vec![0u8; len as usize];
        if read_full(&mut stdin, &mut body)? < body.len() {
            return Err(input(format!("frame {frames}: truncated body")));
        }
        let frame: Value = serde_json::from_slice(&body)
            .map_err(|e| input(format!("frame {frames} is not JSON: {e}")))?;
        let ids = token_ids(&frame, "ids")?;
        let want = frame
            .get("text_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| input(format!("frame {frames}: text_sha256 must be a string")))?;
        let whole = vocab.decode_with(&ids, skip_special);
        let mut decoder = vocab.decoder_with(skip_special);
        let mut streamed: String = ids.iter().map(|&id| decoder.push(id)).collect();
        streamed.push_str(&decoder.finish());
        if sha256_hex(whole.as_bytes()) != want || streamed != whole {
            mismatched.push(frames);
        }
        frames += 1;
    }
    Ok(json!({
        "frames": frames,
        "mismatches": mismatched.len(),
        "mismatched_frames": mismatched,
    }))
}

/// Fills `buf` unless EOF comes first; the number of bytes read.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> Result<usize, Error> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::io("reading stdin", e)),
        }
    }
    Ok(got)
}

fn run(args: &[String]) -> Result<(), Error> {
    match args.split_first() {
        Some((cmd, rest)) if cmd == "probe" => {
            let o = Options::parse_without_generate_options(rest)?;
            let (report, failed) = probe(&o)?;
            let body = if o.json {
                format!("{report:#}\n")
            } else {
                text(&report)
            };
            std::io::stdout()
                .write_all(body.as_bytes())
                .map_err(|e| Error::io("writing the report", e))?;
            if failed {
                return Err(Error {
                    class: "smoke",
                    exit: EXIT_IO,
                    message: report["smoke"]["reason"]
                        .as_str()
                        .unwrap_or("failed")
                        .to_string(),
                });
            }
            Ok(())
        }
        Some((cmd, rest)) if cmd == "generate" => {
            generate(&Options::parse(rest)?.without_drafts()?)
        }
        Some((cmd, rest)) if cmd == "score" => write_doc(&score(
            &Options::parse_without_generate_options(rest)?.without_drafts()?,
        )?),
        Some((cmd, rest)) if cmd == "logprobs" => write_doc(&logprobs(
            &Options::parse_without_generate_options(rest)?.without_drafts()?,
        )?),
        Some((cmd, rest)) if cmd == "detok" => write_doc(&detok_check(rest)?),
        Some((cmd, _)) => Err(usage(format!("unknown command {cmd:?}"))),
        None => Err(usage("a command is required")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [a] if a == "--help" || a == "-h" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        [a] if a == "--version" => {
            println!(
                "cere-native {} (package v{})",
                env!("CARGO_PKG_VERSION"),
                include_str!("../../../../PACKAGE_VERSION").trim()
            );
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}: {}", e.class, e.message);
            ExitCode::from(u8::try_from(e.exit).unwrap_or(1))
        }
    }
}
