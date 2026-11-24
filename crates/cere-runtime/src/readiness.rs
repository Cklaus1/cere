//! Backend readiness, cached as a [`ReadinessSnapshot`] that `/readyz` serializes.
//!
//! `sglang_http` (m1 design note A2, Q8, Q9): a background probe of `GET /get_server_info` and `GET /v1/models`
//! every [`PROBE_INTERVAL`]. Ready means: both endpoints answered, every pinned `deployments.expect` field (except the
//! recorded-only `image_digest`) equals the server's value, the served model name is listed, and the startup
//! smoke (one 1-token greedy chat under a prefix-free rid) passed for the current backend identity. The smoke is
//! backend work: it waits for an admission slot of its own worker (internal scope [`SMOKE_SCOPE`],
//! `Target::Only(worker_id)`) like any request, and a smoke that does not complete is aborted by its rid. Every
//! failed check is reported as a [`Problem`] naming the field (or the endpoint, on connection loss). A probe that
//! cannot confirm the identity forgets it, so a backend that comes back is smoked again before it is ready.
//!
//! The identity also covers the backend launch (m1close design note Q10) when `/get_server_info` reports
//! both `random_seed` and `startup_time`. These are observations, never pins: they never produce a problem. Every
//! smoke is preceded by a not-ready (smoke pending) publish under the confirmed instance id, so a launch change is
//! published before the new launch is smoked, then ready under the new instance id. Each publish calls the
//! optional publish hook ([`Readiness::on_publish`]): `cered` feeds its worker registry and admission from it
//! (m3 design note Q4, Q7), before the smoke that follows a pending publish.
//!
//! `native_cuda` (m22 design note R4, "Native path contract"): the first probe loads the in-process executor
//! once (fail closed): the device bound by `CUDA_VISIBLE_DEVICES` is `deployments.native.device_uuid` (with
//! `[[deployments.native.replicas]]`, m32n design note D3: the visible list is every replica's uuid in order and
//! replica `i` runs on executor device `i`; the library file is opened and hash-checked once for all), the library
//! file hashes to `library_sha256` before `dlopen`, the artifact opens (stamp mode) with the pinned manifest
//! SHA-256, the tokenizer sidecar reports the lock's tokenizer/template hashes, the uploaded model's capabilities
//! equal the deployment capabilities file field by field (`capabilities.<field>`) and its `kv_pool_tokens` the
//! configured or derived KV pool (m25 design note D4), and the SMOKE (one greedy token:
//! sidecar encode, PREFILL on a fresh session) passed. Only then is the [`NativeBackend`] published in its
//! [`NativeSlot`]. The process cannot reload, so the outcome is final. Later probes report the tokenizer sidecar
//! (ops-hardening design note OH-01): `sidecar` advisory while a child lost after a served call is respawning
//! (requests wait for it), `sidecar` not ready while no child serves and none is coming before a backoff ends; every
//! change of its health wakes a probe at once, which publishes under the sidecar's health lock, so a stale state is
//! never published after a newer one. An executor crash ends the process.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, Weak};
use std::thread::{Scope, ScopedJoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cere_compiler::artifact::{self, Artifact, Mode};
use cere_core::capabilities::ExecutionCapabilities;
use cere_proto::worker::v2 as pb;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::admission::{self, Admission, Permit, Priority};
use crate::app::{LoadReport, NativeBackend, NativeSlot};
use crate::config::{
    NativeConfig, PrefixCacheIsolation, RemoteExpect, RemoteWorkerConfig, RuntimeConfig,
    SharedBackendTenancy,
};
use crate::digest::hex;
use crate::fed::client::{AuthorityBase, OpRef, Work, WorkerClient};
use crate::fed::registry::{Desired, RemoteError, RemoteRegistry, RemoteState};
use crate::fed::remote::is_transport;
use crate::fed::wire::{self, AdmissionState, Inventory, ModelState};
use crate::ledger::Ledger;
use crate::levers::{GrammarDrafts, LeverSet};
use crate::native;
use crate::native::detok::Vocab;
use crate::native::driver::{Driver, ExecutorConfig, Model, ModelSpec, WorkRequest};
use crate::native::export::{self, StateIdentity};
use crate::native::ffi::{self, Library, LoadError};
use crate::native::sidecar::{
    self, GrammarHealth, GrammarInfo, GrammarSidecar, Sidecar, SidecarHealth, SidecarInfo,
    TemplateKwargs,
};
use crate::native::upload::{self, PlanOptions, UploadOptions};
use crate::sglang::{RidGenerator, SglangHttp};
use crate::validate::{classify_response, ChatMessage, Finish, Role};
use crate::worker::WorkerState;

/// Probe period. Q8 forbids anything shorter outside tests.
pub const PROBE_INTERVAL: Duration = Duration::from_secs(5);

/// Admission scope of the readiness smoke. Not a valid `auth.scopes.scope_id`, so it never collides.
pub const SMOKE_SCOPE: &str = "#readiness";

/// One failed readiness check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Problem {
    /// A pinned `get_server_info` key, `served_model_name`, `smoke`, `probe`, or the unreachable endpoint path.
    pub field: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SmokeStatus {
    /// No backend identity confirmed yet (or it was lost).
    Pending,
    Passed {
        instance_id: String,
    },
    Failed {
        instance_id: String,
        detail: String,
    },
}

/// Q9 posture, reported verbatim from the config.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TenancyPosture {
    pub shared_backend_tenancy: SharedBackendTenancy,
    pub allow_cross_scope_prefix_cache: bool,
    /// Effective Q2 isolation (`none` when the config omits it).
    pub prefix_cache_isolation: PrefixCacheIsolation,
    pub scope_count: usize,
}

/// Cached result of the latest probe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReadinessSnapshot {
    pub ready: bool,
    /// The blocking problems, then the advisory ones: `ready` exactly when no blocking problem stands. The one
    /// advisory problem is the native `grammar` process respawning or down (spec m28 D7): constrained requests are
    /// refused while the tokenizer sidecar and unconstrained serving go on.
    pub problems: Vec<Problem>,
    pub probed_at_unix_ms: Option<u64>,
    pub served_model_name: String,
    /// `sglang-<16 hex>` derived from base URL, served name, the pinned server values and, when reported, the
    /// backend launch; set when every pin matched on the latest probe.
    pub instance_id: Option<String>,
    pub smoke: SmokeStatus,
    pub tenancy: TenancyPosture,
}

/// Shared handle; clones observe the same snapshot.
#[derive(Clone)]
pub struct Readiness {
    inner: Arc<Inner>,
    interval: Duration,
}

/// Called with every snapshot about to be published, before it is cached.
type PublishHook = Box<dyn Fn(&ReadinessSnapshot) + Send + Sync>;

struct Inner {
    served_model_name: String,
    target: Target,
    snapshot: Mutex<ReadinessSnapshot>,
    hook: OnceLock<PublishHook>,
}

/// What is probed: an SGLang server over HTTP, or the in-process native executor.
enum Target {
    Sglang(SglangProbe),
    Native(NativeProbe),
}

struct SglangProbe {
    /// [`SglangHttp::fresh_connections`]: a probe's verdict is the backend's state now, never the fate of a
    /// connection kept from an earlier probe (a lost backend always reads as refused, `/get_server_info` too).
    client: SglangHttp,
    /// The worker the smoke is admitted on (its device, m3 design note Q3).
    worker_id: Arc<str>,
    rids: Arc<RidGenerator>,
    admission: Admission,
    /// Pinned `get_server_info` key → expected value.
    expected: BTreeMap<String, Value>,
    smoke_deadline: Duration,
    /// Held for a whole probe, so probes never overlap.
    probe: tokio::sync::Mutex<ProbeState>,
}

/// Probe-serialised state: the smoke status and the identity the latest smoke ran against (kept through an
/// outage, so a relaunch is recognised when the backend returns).
struct ProbeState {
    smoke: SmokeStatus,
    smoked: Option<String>,
}

impl Readiness {
    /// `sglang_http` for worker `worker_id` of entry `deployment` (M5 D9: its own `expect` and
    /// `served_model_name`); the probe calls it over fresh connections. `rids` is the process rid generator; the
    /// smoke request waits for an `admission` slot on `worker_id` and runs under `smoke_deadline` (queue wait
    /// included). Nothing is probed until [`Readiness::probe`] or [`Readiness::spawn`].
    pub fn new(
        client: SglangHttp,
        worker_id: &str,
        config: &RuntimeConfig,
        deployment: usize,
        rids: Arc<RidGenerator>,
        admission: Admission,
        smoke_deadline: Duration,
    ) -> Result<Self, String> {
        let expect = config.deployments[deployment]
            .expect
            .as_ref()
            .ok_or_else(|| format!("readiness needs deployments[{deployment}].expect"))?;
        let Value::Object(pins) = serde_json::to_value(expect).map_err(|e| e.to_string())? else {
            return Err(format!(
                "deployments[{deployment}].expect does not serialize to an object"
            ));
        };
        let mut expected: BTreeMap<String, Value> = pins.into_iter().collect();
        // Recorded evidence only; SGLang does not report it.
        expected.remove("image_digest");
        // Absent nullable pins (chat_template = the model's bundled template, no quantization, no
        // speculative decoding) require the backend to report null.
        for key in [
            "chat_template",
            "quantization",
            "speculative_algorithm",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
        ] {
            expected.entry(key.to_owned()).or_insert(Value::Null);
        }
        Self::with_target(
            config,
            deployment,
            Target::Sglang(SglangProbe {
                client: client.fresh_connections().map_err(|e| e.to_string())?,
                worker_id: worker_id.into(),
                rids,
                admission,
                expected,
                smoke_deadline,
                probe: tokio::sync::Mutex::new(ProbeState {
                    smoke: SmokeStatus::Pending,
                    smoked: None,
                }),
            }),
        )
    }

    /// `native_cuda` replica `ordinal` of the deployment (m32n design note D1/D2: the single form is replica 0 on
    /// `device_uuid`). Needs `deployments.served_model_name`, `deployments.native` and the pinned deployment. The
    /// first probe loads the executor (executor `worker_epoch` = the worker's journaled epoch, m3 design note Q4)
    /// and fills `slot` once every identity check and the SMOKE passed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn native(
        config: &RuntimeConfig,
        slot: NativeSlot,
        worker_epoch: u64,
        ordinal: usize,
        library: SharedLibrary,
        lora: Option<Arc<crate::native::lora::Lora>>,
        research: Option<Arc<crate::native::research::Research>>,
        park: Option<Arc<crate::native::park::Park>>,
    ) -> Result<Self, String> {
        let native = config.deployments[0]
            .native
            .clone()
            .ok_or("readiness needs deployments[0].native")?;
        let deployment = config.deployments[0]
            .lock
            .as_ref()
            .ok_or("readiness needs the pinned deployment")?;
        let replicas = native.replica_list(&config.scheduler.devices);
        let device_uuid = replicas
            .get(ordinal)
            .map(|r| r.device_uuid.clone())
            .ok_or_else(|| format!("readiness needs native replica {ordinal}"))?;
        let visible = replicas
            .iter()
            .map(|r| r.device_uuid.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let replica = native.replicas.is_some();
        let load = NativeLoad {
            sidecar_argv: sidecar::argv(&native.sidecar, config.api.max_request_body_bytes),
            native,
            worker_epoch,
            tokenizer_sha256: deployment.lock.tokenizer_sha256.clone(),
            template_sha256: deployment.lock.template_sha256.clone(),
            profile: deployment.profile.clone(),
            host_spill_enabled: config.state.host_spill_enabled,
            device_uuid,
            visible,
            ordinal,
            library,
            loads: AtomicU64::new(0),
            lora,
            research,
            park,
        };
        Self::with_target(
            config,
            0,
            Target::Native(NativeProbe {
                load: Arc::new(load),
                slot,
                replica,
                state: tokio::sync::Mutex::new(None),
                changed: Arc::new(tokio::sync::Notify::new()),
            }),
        )
    }

    fn with_target(
        config: &RuntimeConfig,
        deployment: usize,
        target: Target,
    ) -> Result<Self, String> {
        let backend = &config.deployments[deployment];
        let served_model_name = backend.served_model_name.clone().ok_or_else(|| {
            format!("readiness needs deployments[{deployment}].served_model_name")
        })?;
        let snapshot = ReadinessSnapshot {
            ready: false,
            problems: vec![Problem {
                field: "probe".to_owned(),
                detail: "no probe has completed".to_owned(),
            }],
            probed_at_unix_ms: None,
            served_model_name: served_model_name.clone(),
            instance_id: None,
            smoke: SmokeStatus::Pending,
            tenancy: TenancyPosture {
                shared_backend_tenancy: backend.shared_backend_tenancy,
                allow_cross_scope_prefix_cache: backend.allow_cross_scope_prefix_cache,
                prefix_cache_isolation: backend.prefix_cache_isolation(),
                scope_count: config.auth.scopes.len(),
            },
        };
        Ok(Self {
            inner: Arc::new(Inner {
                served_model_name,
                target,
                snapshot: Mutex::new(snapshot),
                hook: OnceLock::new(),
            }),
            interval: PROBE_INTERVAL,
        })
    }

    /// Sets the publish hook, called with every snapshot this readiness publishes before `/readyz` can see it.
    /// Set once, before probing starts; a later call is ignored.
    pub(crate) fn on_publish(&self, hook: impl Fn(&ReadinessSnapshot) + Send + Sync + 'static) {
        let _ = self.inner.hook.set(Box::new(hook));
    }

    /// A different probe period (tests; production uses [`PROBE_INTERVAL`] through `Tuning::default`).
    pub fn with_probe_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Latest cached result; never touches the backend.
    pub fn snapshot(&self) -> ReadinessSnapshot {
        self.inner.lock_snapshot().clone()
    }

    /// Runs one probe now (waiting for any probe in progress, including the native load), caches and returns
    /// the result.
    pub async fn probe(&self) -> ReadinessSnapshot {
        match &self.inner.target {
            Target::Sglang(sglang) => sglang.probe(&self.inner).await,
            Target::Native(native) => native.probe(&self.inner).await,
        }
    }

    /// Probes immediately, then every interval, until the task is aborted. A native backend also probes at once
    /// whenever its tokenizer sidecar's health changes (OH-01).
    pub fn spawn(&self) -> tokio::task::JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(this.interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            let changed = match &this.inner.target {
                Target::Native(native) => Some(native.changed.clone()),
                Target::Sglang(_) => None,
            };
            loop {
                match &changed {
                    Some(changed) => tokio::select! {
                        _ = ticker.tick() => {}
                        () = changed.notified() => {}
                    },
                    None => {
                        ticker.tick().await;
                    }
                }
                this.probe().await;
            }
        })
    }
}

impl Inner {
    fn lock_snapshot(&self) -> MutexGuard<'_, ReadinessSnapshot> {
        self.snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Caches a probe result and returns it: `problems` block readiness, `advisory` ones are only reported after
    /// them. The hook runs first, so what it records (an epoch bump) is in place before the snapshot is visible.
    /// Publishes of one readiness never overlap (each runs under its probe lock).
    fn publish(
        &self,
        mut problems: Vec<Problem>,
        advisory: Vec<Problem>,
        instance_id: Option<String>,
        smoke: &SmokeStatus,
    ) -> ReadinessSnapshot {
        let mut next = self.lock_snapshot().clone();
        next.ready = problems.is_empty();
        problems.extend(advisory);
        next.problems = problems;
        next.probed_at_unix_ms = Some(unix_ms());
        next.instance_id = instance_id;
        next.smoke = smoke.clone();
        if let Some(hook) = self.hook.get() {
            hook(&next);
        }
        *self.lock_snapshot() = next.clone();
        next
    }
}

impl SglangProbe {
    async fn probe(&self, inner: &Inner) -> ReadinessSnapshot {
        let mut state = self.probe.lock().await;
        let (problems, instance_id) = self.check(inner, &mut state).await;
        inner.publish(problems, Vec::new(), instance_id, &state.smoke)
    }

    /// Problems found and the instance id when every pin matched. Before a smoke, the confirmed identity is
    /// published not ready (smoke pending); for a launch change (Q10) that closes the gate before the new launch
    /// is smoked.
    async fn check(&self, inner: &Inner, state: &mut ProbeState) -> (Vec<Problem>, Option<String>) {
        let served_model_name = &inner.served_model_name;
        let smoke = &mut state.smoke;
        let mut problems = Vec::new();
        let info = self
            .client
            .server_info()
            .await
            .map_err(|e| problems.push(problem("/get_server_info", e.to_string())))
            .ok();
        match self.client.models().await {
            Ok(ids) if ids.contains(served_model_name) => {}
            Ok(_) => problems.push(problem(
                "served_model_name",
                format!("{served_model_name:?} is not listed by /v1/models"),
            )),
            Err(e) => problems.push(problem("/v1/models", e.to_string())),
        }
        if let Some(info) = &info {
            for (key, expected) in &self.expected {
                match info.get(key) {
                    Some(actual) if actual == expected => {}
                    Some(actual) => problems.push(problem(
                        key,
                        format!("expected {expected}, backend reports {actual}"),
                    )),
                    None => problems.push(problem(
                        key,
                        format!("expected {expected}, absent from /get_server_info"),
                    )),
                }
            }
        }
        let Some(info) = info.filter(|_| problems.is_empty()) else {
            *smoke = SmokeStatus::Pending;
            return (problems, None);
        };
        let instance_id = self.instance_id(served_model_name, launch(&info).as_deref());
        // With the pins and the base URL fixed, a different id than the last smoked one means the backend was
        // relaunched, also when an outage (a real relaunch takes minutes) reset the smoke to pending in between.
        let relaunched = state.smoked.as_deref().is_some_and(|p| p != instance_id);
        if relaunched {
            eprintln!(
                "cered readiness: backend launch changed instance_id={} -> {instance_id}; smoking again",
                state.smoked.as_deref().unwrap_or_default()
            );
        }
        let smoked = !relaunched
            && matches!(&*smoke, SmokeStatus::Passed { instance_id: passed } if *passed == instance_id);
        if !smoked {
            *smoke = SmokeStatus::Pending;
            let detail = if relaunched {
                "pending: backend launch changed"
            } else {
                "pending: smoke in progress"
            };
            inner.publish(
                vec![problem("smoke", detail.to_owned())],
                Vec::new(),
                Some(instance_id.clone()),
                smoke,
            );
            *smoke = match self.smoke(served_model_name).await {
                Ok(()) => SmokeStatus::Passed {
                    instance_id: instance_id.clone(),
                },
                Err(detail) => {
                    problems.push(problem("smoke", detail.clone()));
                    SmokeStatus::Failed {
                        instance_id: instance_id.clone(),
                        detail,
                    }
                }
            };
            state.smoked = Some(instance_id.clone());
        }
        (problems, Some(instance_id))
    }

    /// Called only when every pin matched, so the pinned values are the observed ones. `launch` (see
    /// [`launch`]) joins the hashed identity under the key `"launch"` only when reported, so a backend without
    /// launch fields keeps the pre-Q10 id byte for byte.
    fn instance_id(&self, served_model_name: &str, launch: Option<&str>) -> String {
        let mut identity = json!({
            "base_url": self.client.base_url(),
            "served_model_name": served_model_name,
            "server_info": self.expected,
        });
        if let Some(launch) = launch {
            identity["launch"] = json!(launch);
        }
        prefixed_id("sglang-", &identity)
    }

    /// One greedy 1-token chat under an admission slot; a `stop` or full-budget `length` finish passes. A call
    /// that ends without a response (timeout, connection loss) sends `/abort_request` for the smoke's rid before
    /// the slot is released. That abort runs detached and keeps the slot (review A1 pass 3): the probe publishes
    /// its outcome, a sighting for a Suspect worker, without waiting up to the control timeout for it.
    async fn smoke(&self, served_model_name: &str) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + self.smoke_deadline;
        let permit = self
            .admission
            .acquire(
                SMOKE_SCOPE,
                Priority::Normal,
                deadline,
                admission::Target::Only(self.worker_id.clone()),
                1,
            )
            .await
            .map_err(|e| format!("no admission slot for the smoke: {}", e.code()))?;
        let rid = self.rids.next_rid();
        let body = smoke_body(rid.as_str(), served_model_name);
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let response = match self.client.chat(&body, remaining).await {
            Ok(response) => response,
            Err(e) => {
                let client = self.client.clone();
                tokio::spawn(async move {
                    if let Err(abort) = client.abort(&rid).await {
                        eprintln!(
                            "cered readiness: abort of smoke rid={} failed: {abort}",
                            rid.as_str()
                        );
                    }
                    drop(permit);
                });
                return Err(e.to_string());
            }
        };
        match classify_response(&response, 1, 1).map_err(|e| e.to_string())? {
            Finish::Stop | Finish::Length => Ok(()),
            other => Err(format!("smoke chat finished with {}", other.as_str())),
        }
    }
}

/// The user message of every smoke (both kinds).
const SMOKE_MESSAGE: &str = "ping";

/// The chat body of an SGLang smoke (local, and the remote smoke's inference input): greedy, one token, unary.
fn smoke_body(rid: &str, served_model_name: &str) -> Value {
    json!({
        "rid": rid,
        "model": served_model_name,
        "messages": [{ "role": "user", "content": SMOKE_MESSAGE }],
        "max_completion_tokens": 1,
        "temperature": 0,
        "stream": false,
    })
}

/// `cere_executor_options.device_id`: the replica's ordinal among the cards `CUDA_VISIBLE_DEVICES` leaves visible
/// (`0` for the single form; m32n design note D3), unless the process environment names another executor device
/// in [`NATIVE_DEVICE_ID_ENV`] (the CPU toy executor's `toy[,...]`).
pub const NATIVE_DEVICE_ID_ENV: &str = "CERE_NATIVE_DEVICE_ID";
/// Q12: the executor's device bytes on GPU0; the default of `[deployments.native] executor_memory_limit_bytes`
/// (m29 design note M29S2-02: a profile may raise it to its measured peak + 512 MiB, ledger M1).
pub(crate) const NATIVE_MEMORY_BUDGET_BYTES: u64 = 80 << 30;
/// m23 design note P8 (ledger M1, N7): the ceiling of any executor memory limit, GPU0 total (97,887 MiB) - the
/// desktop baseline L0 (1,215 MiB, `native_spec-G-20261013T125617Z` R3 `baseline_used_mib`) - the invariant 10,240
/// MiB free floor = 86,432 MiB. A limit above it would let the executor cross the floor, so the config is refused;
/// the live gates still check free at the measured peak.
pub(crate) const NATIVE_MEMORY_CEILING_BYTES: u64 = (97_887 - 1_215 - 10_240) << 20;

/// The executor device id: `env` (the [`NATIVE_DEVICE_ID_ENV`] value; with several replicas, a `;`-separated value
/// gives replica `ordinal` its entry, the last entry going to every later replica) or the replica `ordinal` (a
/// `vk:<ordinal>` binding: that ordinal), then,
/// with `host_share_dir`, the share options `,host_share=<dir>,artifact=<expected manifest sha256>` (O47,
/// build-loop-ple design note; the executor fails closed on any share failure), then for a replica of the
/// multi-replica form `,release_device=1` (m32n design note D10: an operator `unload` destroys the replica's
/// executor, the card's last, so the executor resets the card's primary context and the card returns to the
/// driver; the single form loads once for the life of the process and keeps its context).
/// The device-binding check of a load (fail closed, before any device work): an NVIDIA binding (the uuid list) needs
/// `CUDA_VISIBLE_DEVICES` equal to it; an AMD binding `hip:<ordinal>` (NativeConfig::device_uuid, the HIP executor)
/// needs `HIP_VISIBLE_DEVICES` equal to the ordinal and `CUDA_VISIBLE_DEVICES` unset (HIP honors it too, so a stray
/// value could select another card or none); a Vulkan binding `vk:<ordinal>` (the Vulkan executor) names its device
/// by the executor device id alone ([`native_device_id`]): neither variable is read by the Vulkan loader or RADV, so
/// neither is required or checked. `None` when the binding holds, else the problem's detail.
fn binding_problem(visible: &str, cuda: Option<String>, hip: Option<String>) -> Option<String> {
    let seen = |v: &Option<String>| {
        v.as_ref()
            .map_or_else(|| "not set".to_owned(), |v| format!("{v:?}"))
    };
    if crate::config::vk_ordinal(visible).is_some() {
        return None;
    }
    match crate::config::hip_ordinal(visible) {
        Some(ord) => {
            if cuda.is_some() {
                Some(format!(
                    "CUDA_VISIBLE_DEVICES must be unset with {visible} (HIP honors it), it is {}",
                    seen(&cuda)
                ))
            } else if hip.as_deref() != Some(ord) {
                Some(format!(
                    "expected HIP_VISIBLE_DEVICES {ord}, it is {}",
                    seen(&hip)
                ))
            } else {
                None
            }
        }
        None if cuda.as_deref() == Some(visible) => None,
        None => Some(format!(
            "expected CUDA_VISIBLE_DEVICES {visible}, it is {}",
            seen(&cuda)
        )),
    }
}

fn native_device_id(env: Option<String>, n: &NativeConfig, ordinal: usize) -> String {
    let mut id = match env {
        Some(env) => {
            let entries = env.split(';').collect::<Vec<_>>();
            entries[ordinal.min(entries.len() - 1)].to_owned()
        }
        // `vk:<ordinal>` (single form): the library selects Vulkan physical device `<ordinal>`; no environment
        // variable narrows the visible devices, so the replica index (0) would pick the wrong card.
        None => match n.device_uuid.as_deref().and_then(crate::config::vk_ordinal) {
            Some(vk) => vk.to_owned(),
            None => ordinal.to_string(),
        },
    };
    if let Some(dir) = &n.host_share_dir {
        id.push_str(&format!(
            ",host_share={dir},artifact={}",
            n.expected_manifest_sha256
        ));
        // M2.16 L4: the published segment outlives this process, so a restart attaches instead of re-pinning.
        if n.keep_host_share {
            id.push_str(",host_share_keep=1");
        }
    }
    if n.replicas.is_some() {
        id.push_str(",release_device=1");
    }
    id
}

struct NativeProbe {
    load: Arc<NativeLoad>,
    slot: NativeSlot,
    /// m32n design note D9/D10: a replica of a multi-replica deployment, which re-smokes its card after a fault,
    /// loads once more when that fails, and is unloaded and loaded by the operator. The single form loads once.
    replica: bool,
    /// `None` until the load ran; held for a whole probe, so the first probe owns the load.
    state: tokio::sync::Mutex<Option<Loaded>>,
    /// Notified by the loaded tokenizer sidecar at every change of its health: [`Readiness::spawn`] probes at once.
    changed: Arc<tokio::sync::Notify>,
}

/// The final outcome of the native load.
struct Loaded {
    problems: Vec<Problem>,
    instance_id: Option<String>,
    smoke: SmokeStatus,
}

impl Loaded {
    fn refused(problems: Vec<Problem>) -> Loaded {
        Loaded {
            problems,
            instance_id: None,
            smoke: SmokeStatus::Pending,
        }
    }
}

impl NativeProbe {
    async fn probe(&self, inner: &Inner) -> ReadinessSnapshot {
        let mut state = self.state.lock().await;
        if self.replica {
            self.lifecycle(inner, &mut state).await;
        }
        let loaded = match state.take() {
            Some(loaded) => loaded,
            None => self.run_load().await,
        };
        let loaded = state.insert(loaded);
        let mut problems = loaded.problems.clone();
        let mut advisory = Vec::new();
        let Some(backend) = self.slot.get() else {
            return inner.publish(
                problems,
                advisory,
                loaded.instance_id.clone(),
                &loaded.smoke,
            );
        };
        advisory.extend(
            backend
                .grammar
                .as_ref()
                .and_then(|grammar| grammar_problem(grammar.health())),
        );
        // Published under the sidecar's health lock: a change meanwhile waits for this publish and then wakes the
        // next probe, so this older state never lands after it.
        backend.sidecar.with_health(|health| {
            match health {
                SidecarHealth::Ready => {}
                SidecarHealth::Respawning => advisory.push(problem(
                    "sidecar",
                    "the tokenizer sidecar is respawning; requests wait for it".to_owned(),
                )),
                SidecarHealth::Down => problems.push(problem(
                    "sidecar",
                    "the tokenizer sidecar exited and did not respawn".to_owned(),
                )),
            }
            inner.publish(
                problems,
                advisory,
                loaded.instance_id.clone(),
                &loaded.smoke,
            )
        })
    }

    /// m32n design note D9 (b), D10, before the probe reads the load outcome: an operator load (requested since
    /// the last probe) loads again; an unloaded replica reports `unloaded` and loads nothing; a card whose request
    /// ended with a device-scoped error (`fault`) is re-smoked first (published not ready meanwhile, so it takes
    /// no new grant), and when the re-smoke fails its executor is released and the load runs once more: a success
    /// is a new `instance_id` (a relaunch, epoch + 1), a failure marks the slot faulted (N4: the replica stays
    /// not ready, Lost by aging, and an operator load is refused until a `cered` restart).
    async fn lifecycle(&self, inner: &Inner, state: &mut Option<Loaded>) {
        let ordinal = self.load.ordinal;
        if self.slot.take_reload() {
            eprintln!("cered readiness: native replica {ordinal}: load requested; loading");
            *state = None;
            return;
        }
        if self.slot.is_unloaded() {
            *state = Some(Loaded::refused(vec![problem(
                "unloaded",
                "the operator unloaded this native replica".to_owned(),
            )]));
            return;
        }
        let Some(backend) = self
            .slot
            .get()
            .filter(|b| b.fault.swap(false, Ordering::AcqRel))
        else {
            return;
        };
        let instance_id = state.as_ref().and_then(|l| l.instance_id.clone());
        inner.publish(
            vec![problem(
                "fault",
                "a request on this card ended with a device error; re-smoking the card".to_owned(),
            )],
            Vec::new(),
            instance_id,
            &SmokeStatus::Pending,
        );
        let b = backend.clone();
        let smoked = tokio::task::spawn_blocking(move || {
            smoke(
                &b.driver,
                &b.model,
                &b.sidecar,
                b.max_context_tokens,
                &b.stop_token_ids,
            )
        })
        .await
        .unwrap_or_else(|e| Err(format!("the re-smoke task ended without a result: {e}")));
        match smoked {
            Ok(_) => eprintln!("cered readiness: native replica {ordinal}: re-smoke passed"),
            Err(detail) => {
                eprintln!(
                    "cered readiness: native replica {ordinal}: re-smoke failed: {detail}; releasing the \
                     executor and loading once more"
                );
                drop(backend);
                drop(self.slot.release());
                let loaded = self.run_load().await;
                if self.slot.get().is_none() {
                    eprintln!(
                        "cered readiness: native replica {ordinal}: the reload after the fault failed; the \
                         replica stays not ready until a cered restart"
                    );
                    self.slot.set_faulted();
                }
                *state = Some(loaded);
            }
        }
    }

    /// The load runs on a blocking thread (it reads and uploads the whole artifact).
    async fn run_load(&self) -> Loaded {
        let load = self.load.clone();
        let (loaded, backend) = tokio::task::spawn_blocking(move || load.run())
            .await
            .unwrap_or_else(|e| {
                let detail = format!("the load task ended without a result: {e}");
                (Loaded::refused(vec![problem("load", detail)]), None)
            });
        for p in &loaded.problems {
            eprintln!(
                "cered readiness: native backend not ready: {}: {}",
                p.field, p.detail
            );
        }
        if let Some(backend) = backend {
            eprintln!(
                "cered readiness: native backend ready instance_id={}",
                backend.instance_id
            );
            // The probe reads the health itself, under the lock that orders it with every publish.
            let changed = self.changed.clone();
            backend.sidecar.watch(move |_| changed.notify_one());
            self.slot.fill(backend);
        }
        loaded
    }
}

/// The advisory `grammar` problem of a grammar process that is not ready (spec m28 D7).
fn grammar_problem(health: GrammarHealth) -> Option<Problem> {
    let detail = match health {
        GrammarHealth::Ready => return None,
        GrammarHealth::Respawning => {
            "the grammar process is respawning; constrained requests are refused"
        }
        GrammarHealth::Down => {
            "the grammar process did not respawn; constrained requests are refused"
        }
    };
    Some(problem("grammar", detail.to_owned()))
}

/// Spec m28 C7 startup check: the grammar process's matchers must stop on exactly the model's stop tokens (its
/// `generation_config` EOS; a request's `stop_token_ids` never enter the grammar's stop set, D15) and cover the
/// model's vocabulary (the mask length). `Some(detail)` refuses readiness.
fn grammar_tokenizer_problem(
    info: &GrammarInfo,
    model_stops: &[u32],
    vocab: u32,
) -> Option<String> {
    let grammar: BTreeSet<u32> = info.stop_token_ids.iter().copied().collect();
    let model: BTreeSet<u32> = model_stops.iter().copied().collect();
    if grammar != model {
        return Some(format!(
            "the grammar process stops on {grammar:?}, the model on {model:?}"
        ));
    }
    (info.vocab_size != vocab).then(|| {
        format!(
            "the grammar process's vocabulary is {}, the model's {vocab}",
            info.vocab_size
        )
    })
}

/// GrammarSpec GS-04: `grammar_drafts = "on"` needs a library exporting `cere_executor_round_drafts` and
/// `cere_executor_constrain_drafts` (`library`), a profile advertising a grammar output (`grammar`) and a grammar
/// process serving `grammar_drafts`; checked once at load (the lever is restart-class).
fn grammar_drafts_problem(
    lever: GrammarDrafts,
    library: bool,
    grammar: Option<&GrammarInfo>,
) -> Option<String> {
    if lever == GrammarDrafts::Off {
        return None;
    }
    if !library {
        return Some(
            "grammar_drafts \"on\" needs a library exporting cere_executor_round_drafts and \
             cere_executor_constrain_drafts"
                .to_owned(),
        );
    }
    match grammar {
        None => Some(
            "grammar_drafts \"on\" needs a profile that advertises a grammar output".to_owned(),
        ),
        Some(info) if !info.drafts => Some(format!(
            "grammar_drafts \"on\" needs a grammar process serving {}",
            sidecar::GRAMMAR_DRAFTS_FEATURE
        )),
        Some(_) => None,
    }
}

/// What [`NativeLoad::load`] hands the backend: driver, sealed model, capabilities, model spec, `layout_id`, how
/// the load ran and the prefill selection signature (lowercase hex, `None` when the library reports none).
type LoadedModel = (
    Driver,
    Model,
    ffi::ExecutionCapabilities,
    ModelSpec,
    [u8; 32],
    LoadReport,
    Option<String>,
);

/// The inputs of the native load, fixed at startup.
struct NativeLoad {
    native: NativeConfig,
    /// The sidecar argv `cered` starts (`native.sidecar` plus the m213 D18 request frame limit).
    sidecar_argv: Vec<String>,
    worker_epoch: u64,
    /// The deployment lock's tokenizer/template hashes.
    tokenizer_sha256: String,
    template_sha256: String,
    /// The deployment capabilities file.
    profile: ExecutionCapabilities,
    /// `[state] host_spill_enabled`: with the prefix cache on it gates the arena (M2.10 D9).
    host_spill_enabled: bool,
    /// m32n design note D1: this replica's card, the `CUDA_VISIBLE_DEVICES` the process must have (the single
    /// form's uuid, or the replicas' uuids comma-joined in replica order) and the executor device ordinal (CUDA
    /// enumerates the listed uuids in list order).
    device_uuid: String,
    visible: String,
    ordinal: usize,
    /// R3: the library is opened and hash-checked once for every replica of the process.
    library: SharedLibrary,
    /// m32n design note D10: loads started so far; a load after the first carries its number in the
    /// `instance_id` (a relaunch to the worker registry: epoch + 1). The first load's id is unchanged.
    loads: AtomicU64,
    /// Track LoRA: the deployment's adapter registry when `[deployments.native.lora]` is set; the model reserves
    /// its pool at create and the registry's adapters are loaded after the seal.
    lora: Option<Arc<crate::native::lora::Lora>>,
    /// WeightLab: the deployment's research registry when `[deployments.native.research]` is set; the model
    /// reserves its staging, ring and plan slots at create and the registered plans are loaded after the seal.
    research: Option<Arc<crate::native::research::Research>>,
    /// NetTeleport S3: the parked-state store under `[state] park_on_stop` (the single form only, by config): the
    /// load imports the parked images before the backend serves, and the backend records the prompts a stop parks.
    park: Option<Arc<crate::native::park::Park>>,
}

/// The one `dlopen`ed executor library of a native deployment (m32n design note D1, R3): the first replica's
/// load opens and hash-checks it; every later replica runs the same handle. A refused open is not kept, so a
/// later load re-checks the file.
///
/// With it, the lock on the deployment's one shared PLE segment (`host_share_dir` + the expected manifest, O47;
/// m32n design note D21): a replica holds it from ALLOCATE to INSTANTIATE, so the replicas create and attach
/// that segment one at a time. While a model of this process creates the segment and is not yet sealed, the library
/// answers another create of it `cere_model_create` BUSY (host_share.cpp, process-wide); serialized, a sibling
/// replica is never that creator, and a BUSY create still means a real busy segment, which fails the load at once.
#[derive(Clone, Default)]
pub(crate) struct SharedLibrary {
    library: Arc<Mutex<Option<Arc<Library>>>>,
    segment: Arc<Mutex<()>>,
}

impl SharedLibrary {
    fn open(&self, path: &Path, sha256: &str) -> Result<Arc<Library>, LoadError> {
        let mut held = self.library.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(library) = held.as_ref() {
            return Ok(library.clone());
        }
        let library = Arc::new(Library::open(path, Some(sha256))?);
        *held = Some(library.clone());
        Ok(library)
    }

    /// Waits until no other replica of this process is between ALLOCATE and INSTANTIATE on the shared segment
    /// (logged when it has to wait), then holds the segment until the guard drops.
    fn hold_segment(&self, ordinal: usize) -> MutexGuard<'_, ()> {
        match self.segment.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                eprintln!(
                    "cered readiness: native replica {ordinal}: waiting for the replica that is loading the \
                     shared PLE segment (D21: one ALLOCATE..INSTANTIATE on it at a time)"
                );
                self.segment.lock().unwrap_or_else(PoisonError::into_inner)
            }
        }
    }
}

impl NativeLoad {
    /// The tokenizer sidecar's I/O deadline: `native.sidecar_deadline_ms`, else the default.
    fn sidecar_deadline(&self) -> Duration {
        match self.native.sidecar_deadline_ms {
            0 => sidecar::DEFAULT_IO_DEADLINE,
            ms => Duration::from_millis(ms),
        }
    }

    /// Identity checks, load, capability comparison and SMOKE; the backend only when all passed. On any failure
    /// everything loaded so far is released (dropping the driver releases the model and the executor).
    fn run(&self) -> (Loaded, Option<NativeBackend>) {
        std::thread::scope(|scope| {
            let processes = Processes::start(
                scope,
                &self.sidecar_argv,
                self.sidecar_deadline(),
                &self.profile,
                &self.native.sidecar,
            );
            self.run_with(processes)
        })
    }

    /// [`NativeLoad::run`] with the tokenizer and grammar processes starting (load-speed design note LS-04).
    fn run_with(&self, processes: Processes<'_>) -> (Loaded, Option<NativeBackend>) {
        let n = &self.native;
        let load = self.loads.fetch_add(1, Ordering::Relaxed);
        let mut laps = Laps::start();
        let mut problems = Vec::new();
        // Before any load, so a wrong binding never touches a device.
        if let Some(detail) = binding_problem(
            &self.visible,
            std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            std::env::var("HIP_VISIBLE_DEVICES").ok(),
        ) {
            problems.push(problem("device_uuid", detail));
        }
        // load-soft design note LSF-02: the library's hash and dlopen, then the executor's start (CUDA context and
        // eager module load), run on one thread beside the artifact open and the vocab. The executor starts only on
        // the bound device (no problem so far) and is dropped unused when a problem refuses the load; the problems
        // keep their order (library, artifact, tokenizer). BeatSglang L1: the vocab (the detokenizer's table, from
        // the artifact's `tokenizer.json`, read once per load) loads on a thread of its own beside the artifact open,
        // so `vocab_ms` is its own duration, no longer a serial lap after the open; and the model spec (pure CPU work
        // on the opened artifact) is built right after the open, while the executor may still be starting.
        let device_bound = problems.is_empty();
        let (library, library_ms, driver, art, spec, vocab, vocab_ms) =
            std::thread::scope(|scope| {
                let started = scope.spawn(|| {
                    let t = Instant::now();
                    let library = self.library.open(Path::new(&n.library), &n.library_sha256);
                    let library_ms = t.elapsed().as_millis();
                    let driver = match &library {
                        Ok(library) if device_bound => {
                            let t = Instant::now();
                            Some((self.start_driver(library.clone()), t.elapsed().as_millis()))
                        }
                        _ => None,
                    };
                    (library, library_ms, driver)
                });
                let vocab = scope.spawn(|| {
                    let t = Instant::now();
                    (
                        Vocab::load(Path::new(&n.artifact_dir)),
                        t.elapsed().as_millis(),
                    )
                });
                let art = artifact::open(
                    Path::new(&n.artifact_dir),
                    &n.expected_manifest_sha256,
                    Mode::Stamp,
                );
                laps.lap("artifact_open_ms");
                let spec = match &art {
                    Ok(art) if device_bound => Some(self.model_spec(art)),
                    _ => None,
                };
                laps.lap("model_spec_ms");
                let (vocab, vocab_ms) = vocab
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                laps.lap("vocab_wait_ms");
                let (library, library_ms, driver) = started
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                (library, library_ms, driver, art, spec, vocab, vocab_ms)
            });
        laps.lap("library_wait_ms");
        laps.record("vocab_ms", vocab_ms);
        laps.record("library_open_ms", library_ms);
        let driver = driver.map(|(driver, ms)| {
            laps.record("executor_ms", ms);
            driver
        });
        let library_ok = library
            .map_err(|e| {
                let field = match e {
                    LoadError::HashMismatch { .. } => "library_sha256",
                    _ => "library",
                };
                problems.push(problem(field, e.to_string()));
            })
            .is_ok();
        let art = art
            .map_err(|e| {
                let field = match e.class {
                    "manifest_digest" => "expected_manifest_sha256",
                    _ => "artifact_dir",
                };
                problems.push(problem(field, format!("{}: {}", e.class, e.message)));
            })
            .ok();
        let vocab = vocab
            .map_err(|e| problems.push(problem("tokenizer", e.to_string())))
            .ok();
        // LS-04: the model loads while the processes start up; a problem found so far refuses before the load, a
        // process's identity problem after it (the loaded model is released). m32n design note D22 (3): a load
        // after the replica's first (an operator `load`, or the one D9 reload after a fault) runs while the other
        // card serves, so it runs in the background (`load > 0`).
        let loaded = match (driver, &art, spec, &vocab) {
            (Some(driver), Some(art), Some(spec), Some(_)) if library_ok && problems.is_empty() => {
                Some(self.load(driver, art, spec, load > 0, &mut laps))
            }
            _ => None,
        };
        let (mut sidecar, sidecar_ms) = processes
            .sidecar
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        let (mut grammar, grammar_ms) = processes
            .grammar
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        laps.lap("processes_wait_ms");
        laps.record("sidecar_ms", sidecar_ms);
        laps.record("grammar_ms", grammar_ms);
        // LS-04: a process that did not start beside a load that succeeded (a startup `info` deadline missed on a
        // loaded host, load-speed design note F8) starts once more alone, as it did before the load on main;
        // failing again refuses with that detail.
        if matches!(loaded, Some(Ok(_))) {
            if let Err(detail) = &sidecar {
                eprintln!("cered readiness: the tokenizer sidecar did not start beside the load ({detail}); starting it again alone");
                let (again, ms) = start_sidecar(&self.sidecar_argv, self.sidecar_deadline());
                sidecar = again;
                laps.record("sidecar_again_ms", ms);
            }
            if let Err(detail) = &grammar {
                eprintln!("cered readiness: the grammar process did not start beside the load ({detail}); starting it again alone");
                let (again, ms) = start_grammar(&self.profile, &self.native.sidecar);
                grammar = again;
                laps.record("grammar_again_ms", ms);
            }
        }
        let sidecar = self.sidecar(sidecar, art.as_ref(), &mut problems);
        // M2.8 D7: the grammar process, only for a profile that advertises a grammar output.
        let grammar = grammar
            .map_err(|detail| problems.push(problem("grammar", detail)))
            .ok()
            .flatten();
        let (Some(vocab), Some(sidecar), Some(loaded)) = (vocab, sidecar, loaded) else {
            return (Loaded::refused(problems), None);
        };
        if !problems.is_empty() {
            return (Loaded::refused(problems), None);
        }
        let (driver, model, capabilities, spec, layout_id, load_report, prefill_signature) =
            match loaded {
                Ok(loaded) => loaded,
                Err(detail) => return (Loaded::refused(vec![problem("load", detail)]), None),
            };
        let mut problems = capability_problems(
            &capabilities,
            &self.profile,
            expected_kv_pool_tokens(&self.native, self.profile.decode_batch_max),
            self.native.prefix_state_slots,
            self.native.speculative_draft_tokens,
        );
        problems.extend(prefill_signature_problem(
            prefill_signature.as_deref(),
            &self.profile,
        ));
        if !problems.is_empty() {
            return (Loaded::refused(problems), None);
        }
        if let Some(detail) = grammar.as_ref().and_then(|g| {
            grammar_tokenizer_problem(g.info(), &spec.stop_token_ids, spec.config.vocab_size())
        }) {
            return (Loaded::refused(vec![problem("grammar", detail)]), None);
        }
        let api = driver.library().api();
        if let Some(detail) = grammar_drafts_problem(
            spec.levers.grammar_drafts,
            api.executor_round_drafts.is_some() && api.executor_constrain_drafts.is_some(),
            grammar.as_ref().map(GrammarSidecar::info),
        ) {
            return (Loaded::refused(vec![problem("levers", detail)]), None);
        }
        // Track LoRA: the preload list on the first load, the registered adapters again on a later one.
        if let Some(lora) = &self.lora {
            let attached = match spec.config.qwen4_exp() {
                Some(config) => lora.attach(&driver, &model, config),
                None => Err("track LoRA adapters exist only for the qwen4_exp family".to_string()),
            };
            if let Err(detail) = attached {
                return (Loaded::refused(vec![problem("lora", detail)]), None);
            }
        }
        // WeightLab: the registered plans again into a reloaded model (none on the first load).
        if let Some(research) = &self.research {
            if let Err(detail) = research.attach(&driver, &model) {
                return (Loaded::refused(vec![problem("research", detail)]), None);
            }
        }
        let mut identity = json!({
            "library_sha256": n.library_sha256,
            "manifest_sha256": n.expected_manifest_sha256,
            "device_uuid": self.device_uuid,
            "tokenizer_sha256": self.tokenizer_sha256,
            "template_sha256": self.template_sha256,
            "profile_id": self.profile.profile_id,
        });
        if load > 0 {
            identity["load"] = json!(load);
        }
        let instance_id = prefixed_id("native-", &identity);
        if let Err(detail) = smoke(
            &driver,
            &model,
            &sidecar,
            n.max_context_tokens,
            &spec.stop_token_ids,
        ) {
            let failed = Loaded {
                problems: vec![problem("smoke", detail.clone())],
                instance_id: Some(instance_id.clone()),
                smoke: SmokeStatus::Failed {
                    instance_id,
                    detail,
                },
            };
            return (failed, None);
        }
        laps.lap("smoke_ms");
        // m23 design note PI-05: the levers the executor reports for the sealed model (not the config's echo); a
        // set other than the desc's means the library did not run what the profile pins.
        let levers = driver
            .levers(&model)
            .map_err(|e| e.to_string())
            .and_then(|raw| LeverSet::from_ffi(&raw, &spec.levers))
            .and_then(|levers| {
                if levers == spec.levers {
                    Ok(levers)
                } else {
                    Err(format!(
                        "the executor reports levers {:?}, the profile pins {:?}",
                        levers.to_table(),
                        spec.levers.to_table()
                    ))
                }
            });
        let levers = match levers {
            Ok(levers) => levers,
            Err(detail) => {
                let failed = Loaded {
                    problems: vec![problem("levers", detail.clone())],
                    instance_id: Some(instance_id.clone()),
                    smoke: SmokeStatus::Failed {
                        instance_id,
                        detail,
                    },
                };
                return (failed, None);
            }
        };
        laps.lap("levers_ms");
        eprintln!("cered readiness: native load stages {}", laps.line());
        let loaded = Loaded {
            problems: Vec::new(),
            instance_id: Some(instance_id.clone()),
            smoke: SmokeStatus::Passed {
                instance_id: instance_id.clone(),
            },
        };
        let backend = NativeBackend {
            driver,
            model,
            sidecar,
            grammar,
            vocab,
            capabilities,
            prefill_signature,
            stop_token_ids: spec.stop_token_ids,
            vocab_size: spec.config.vocab_size(),
            generation_defaults: spec.generation_defaults,
            max_context_tokens: n.max_context_tokens,
            instance_id,
            submissions: AtomicU64::new(0),
            spec_verify_steps: AtomicU64::new(0),
            spec_accepted_drafts: AtomicU64::new(0),
            levers: RwLock::new(levers),
            lever_hot_swap: n.lever_hot_swap,
            grammar_stats: Default::default(),
            preemption: n.preempt.then(|| {
                crate::native::serve::Preemption::new(crate::native::preempt::Policy {
                    min_tokens: n.preempt_min_tokens,
                    arena_bytes: n.snapshot_host_bytes,
                    max_parked: n.max_parked_sessions,
                    every: u64::from(n.preempt_every),
                    reserve: n.reserve_output_tokens,
                    draft: n.speculative_draft_tokens > 0,
                })
            }),
            identity: StateIdentity {
                library_sha256: n.library_sha256.clone(),
                profile_id: self.profile.profile_id.clone(),
                layout_id,
            },
            sessions: Default::default(),
            device_uuid: self.device_uuid.clone(),
            fault: Default::default(),
            affinity: Default::default(),
            committed: AtomicU64::new(0),
            prefix_hits: AtomicU64::new(0),
            rate_mark: Mutex::new((std::time::Instant::now(), 0)),
            load: load_report,
            loop_guard: n
                .loop_guard
                .clone()
                .map(crate::native::loop_guard::LoopGuard::new),
            parked: self.park.as_ref().map(|p| p.recorder()),
        };
        // NetTeleport S3 (L1): the parked images go into the loaded, idle executor before the slot is filled, so
        // no request runs beside the import; every file is deleted, imported or refused.
        if let Some(park) = &self.park {
            park.unpark(&backend);
        }
        (loaded, Some(backend))
    }

    /// Compares the started sidecar's hashes ([`Processes`]) with the lock's (and, with feature `image`, its
    /// preprocessor config with the artifact's, m213 D8); `None` when it did not start or cannot answer `info`.
    fn sidecar(
        &self,
        started: StartedSidecar,
        art: Option<&Artifact>,
        problems: &mut Vec<Problem>,
    ) -> Option<Sidecar> {
        let (sidecar, info) = started
            .map_err(|detail| problems.push(problem("sidecar", detail)))
            .ok()?;
        for (field, pinned, reported) in [
            (
                "tokenizer_sha256",
                &self.tokenizer_sha256,
                &info.tokenizer_sha256,
            ),
            (
                "template_sha256",
                &self.template_sha256,
                &info.template_sha256,
            ),
        ] {
            if pinned != reported {
                problems.push(problem(
                    field,
                    format!("the lock pins {pinned}, the sidecar reports {reported}"),
                ));
            }
        }
        // The sidecar hashes the image processor's config: Qwen's `preprocessor_config.json`, Inkling 2 Mini's
        // `processor_config.json` (inkling design note S6g); an artifact lists one of the two.
        if let (Some(reported), Some(art)) = (&info.preprocessor_sha256, art) {
            let listed = art
                .manifest
                .file("preprocessor_config.json")
                .or_else(|| art.manifest.file("processor_config.json"));
            match listed {
                Some(entry) if &entry.sha256 == reported => {}
                Some(entry) => problems.push(problem(
                    "preprocessor_sha256",
                    format!(
                        "the artifact lists {}, the sidecar reports {reported}",
                        entry.sha256
                    ),
                )),
                None => problems.push(problem(
                    "preprocessor_sha256",
                    "the artifact lists no preprocessor_config.json or processor_config.json"
                        .to_owned(),
                )),
            }
        }
        // M2.12 D4 (ledger S): a profile declaring `thinking_toggle` renders through the feature `thinking`.
        if self.profile.thinking_toggle == Some(true)
            && !info.features.iter().any(|f| f == "thinking")
        {
            problems.push(problem(
                "sidecar",
                "the profile declares thinking_toggle; the sidecar lacks the feature thinking"
                    .to_owned(),
            ));
        }
        Some(sidecar)
    }

    /// The executor on this replica's device ([`Driver::start`], then the O79 decode rounds when configured).
    fn start_driver(&self, library: Arc<Library>) -> Result<Driver, String> {
        let n = &self.native;
        let driver = Driver::start(
            library,
            &ExecutorConfig {
                device_id: native_device_id(
                    std::env::var(NATIVE_DEVICE_ID_ENV).ok(),
                    n,
                    self.ordinal,
                ),
                worker_epoch: self.worker_epoch,
                memory_budget_bytes: n.executor_memory_limit(),
            },
        )
        .map_err(|e| e.to_string())?;
        // O79 (m210 design note PF1): only when configured, so a library predating the call keeps loading.
        if let Some(rounds) = n.decode_rounds_per_prefill_chunk {
            let rounds = u32::try_from(rounds).unwrap_or(u32::MAX);
            driver
                .set_decode_rounds(rounds)
                .map_err(|e| e.to_string())?;
        }
        // S1 G4 (beat-sglang design note): takes effect at the next round; only when configured, so a library
        // predating the call keeps loading.
        if let Some(share_pct) = n.decode_share_pct {
            let share_pct = u32::try_from(share_pct).unwrap_or(u32::MAX);
            driver
                .set_decode_share(share_pct)
                .map_err(|e| e.to_string())?;
        }
        // BC4 (b) (batch-concurrency design note): takes effect at the next prefill round; only when configured, so
        // a library predating the call keeps loading (with the executor default).
        if let Some(calls) = n.prefill_age_calls {
            let calls = u32::try_from(calls).unwrap_or(u32::MAX);
            driver.set_prefill_age(calls).map_err(|e| e.to_string())?;
        }
        // MixedPD (mixed-pd design note): before the model is created (a model created while it is 0 never slices);
        // only when configured, so a library predating the call keeps loading.
        if let Some(slice_ms) = n.prefill_slice_ms {
            let slice_ms = u32::try_from(slice_ms).unwrap_or(u32::MAX);
            driver
                .set_prefill_slice(slice_ms)
                .map_err(|e| e.to_string())?;
        }
        Ok(driver)
    }

    /// The model spec of the opened `art` (stop tokens, vocabulary size, sampling defaults, the plan of the
    /// profile's `decode_batch_max` sessions, the rows (m26 D9)), with the configured levers, prefix settings and
    /// LoRA reserve: pure CPU work, built beside the executor's start ([`NativeLoad::run_with`], BeatSglang L1).
    fn model_spec(&self, art: &Artifact) -> Result<ModelSpec, String> {
        let n = &self.native;
        let mut spec = upload::model_spec(
            art,
            Path::new(&n.artifact_dir),
            &PlanOptions {
                max_sessions: self.profile.decode_batch_max,
                max_context_tokens: n.max_context_tokens,
                snapshot_host_bytes: n.desc_snapshot_host_bytes(self.host_spill_enabled),
                kv_pool_tokens: n.kv_pool_tokens.unwrap_or(0),
                prefix_state_slots: n.prefix_state_slots,
                speculative_draft_tokens: n.speculative_draft_tokens,
            },
        )
        .map_err(|e| e.to_string())?;
        spec.levers = n.lever_set()?;
        spec.prefix_end_state = n.prefix_end_state;
        spec.prefix_alt_edges = n.prefix_alt_edges;
        spec.lora = self.lora.as_ref().map(|l| l.reserve());
        spec.research = n
            .research
            .as_ref()
            .map(|r| crate::native::driver::ResearchReserve {
                device_staging_bytes: r.device_staging_bytes,
                host_buffer_bytes: r.host_buffer_bytes,
                max_plans: r.max_plans,
            });
        Ok(spec)
    }

    /// ALLOCATE, UPLOAD and INSTANTIATE (`upload::upload`) of `spec` ([`NativeLoad::model_spec`]) on the started
    /// `driver` ([`NativeLoad::start_driver`], whose error is reported after a model spec error, as when both ran
    /// here): the driver, the sealed model, its capabilities, the model spec, the state `layout_id` (M2.11 D3) and
    /// how the load ran. A `background` load (m32n design note D22 (3)) stages through
    /// [`UploadOptions::background`].
    fn load(
        &self,
        driver: Result<Driver, String>,
        art: &Artifact,
        spec: Result<ModelSpec, String>,
        background: bool,
        laps: &mut Laps,
    ) -> Result<LoadedModel, String> {
        let n = &self.native;
        let spec = spec?;
        let driver = driver?;
        // m32n design note D21: ALLOCATE..INSTANTIATE under the shared segment's lock, so a sibling replica
        // creating the segment never makes this create BUSY ([`SharedLibrary`]); the guard drops after the seal
        // or after `upload` released a failed model.
        let segment = n
            .host_share_dir
            .is_some()
            .then(|| self.library.hold_segment(self.ordinal));
        let opts = if background {
            UploadOptions::background()
        } else {
            UploadOptions::default()
        };
        let up = upload::upload(&driver, art, &spec, &opts);
        drop(segment);
        let up = up.map_err(|e| e.to_string())?;
        laps.lap("upload_ms");
        let report = LoadReport {
            background,
            nice: up.staging.nice,
            allocate_nice: up.allocate.nice,
            allocate_cpus: up.allocate.cpus,
            staging_slots: opts.buffers,
            copies_in_flight_max: up.staging.copies_in_flight_max,
        };
        // LoadCrit LC-02: stage_all's phases and the pass's first three seconds (MB read in each), so a cold row
        // splits read_upload into the pre-pass work, a slow pass start, the pass and the tail (O158).
        let phases = up.staging.phases_ms;
        let second_mb = |s: u64| {
            up.staging
                .pass
                .timeline
                .iter()
                .find(|x| x.t_s == s)
                .map_or(0, |x| x.bytes / 1_000_000)
        };
        eprintln!(
            "cered readiness: native model sealed allocate_ms={} read_upload_ms={} instantiate_ms={} \
             kv_pool_tokens={} background={} nice={} allocate_nice={} allocate_cpus={} staging_slots={} \
             copies_in_flight_max={} index_ms={} prepare_ms={} buffers_ms={} register_ms={} pass_ms={} \
             checks_after_pass_ms={} unregister_ms={} pass_mb_s0={} pass_mb_s1={} pass_mb_s2={}",
            up.allocate_ms,
            up.upload_ms,
            up.instantiate_ms,
            up.capabilities.kv_pool_tokens,
            report.background,
            report.nice,
            report.allocate_nice,
            report.allocate_cpus,
            report.staging_slots,
            report.copies_in_flight_max,
            phases.index,
            phases.prepare,
            phases.buffers,
            phases.register,
            phases.pass,
            phases.checks_after_pass,
            phases.unregister,
            second_mb(0),
            second_mb(1),
            second_mb(2)
        );
        let layout_id = export::layout_id(&spec.config);
        let prefill_signature = up.prefill_signature.as_ref().map(|s| hex(s));
        Ok((
            driver,
            up.model,
            up.capabilities,
            spec,
            layout_id,
            report,
            prefill_signature,
        ))
    }
}

/// A started tokenizer sidecar with its `info` answer, or why it did not start.
type StartedSidecar = Result<(Sidecar, SidecarInfo), String>;
/// A started grammar process (`None` for a profile without a grammar output), or why it did not start.
type StartedGrammar = Result<Option<GrammarSidecar>, String>;

/// The tokenizer sidecar (spawned and asked `info`) and the grammar process (M2.8 D7, only for a profile that
/// advertises a grammar output), each started on its own scoped thread with its wall milliseconds: seconds of
/// Python imports each, which overlap the model load (load-speed design note LS-04). The scope joins them on every
/// path, so no early return leaves a process starting.
struct Processes<'scope> {
    sidecar: ScopedJoinHandle<'scope, (StartedSidecar, u128)>,
    grammar: ScopedJoinHandle<'scope, (StartedGrammar, u128)>,
}

impl<'scope> Processes<'scope> {
    fn start<'env>(
        scope: &'scope Scope<'scope, 'env>,
        sidecar_argv: &'env [String],
        sidecar_deadline: Duration,
        profile: &'env ExecutionCapabilities,
        grammar_argv: &'env [String],
    ) -> Processes<'scope> {
        Processes {
            sidecar: scope.spawn(move || start_sidecar(sidecar_argv, sidecar_deadline)),
            grammar: scope.spawn(move || start_grammar(profile, grammar_argv)),
        }
    }
}

/// Spawns the tokenizer sidecar and asks `info`, with the wall milliseconds it took.
fn start_sidecar(argv: &[String], deadline: Duration) -> (StartedSidecar, u128) {
    let t = Instant::now();
    let started = Sidecar::spawn(argv, deadline)
        .and_then(|sidecar| sidecar.info().map(|info| (sidecar, info)))
        .map_err(|e| e.to_string());
    (started, t.elapsed().as_millis())
}

/// Starts the grammar process (M2.8 D7), with the wall milliseconds it took.
fn start_grammar(profile: &ExecutionCapabilities, argv: &[String]) -> (StartedGrammar, u128) {
    let t = Instant::now();
    let started = native::spawn_grammar(profile, argv);
    (started, t.elapsed().as_millis())
}

/// The wall milliseconds of each step of a native load, in order (load-speed design note LS-02), and of the
/// [`Processes`]: logged once as `cered readiness: native load stages <name>=<ms> ... total_ms=<ms>` when the
/// backend is ready.
struct Laps {
    start: Instant,
    mark: Instant,
    laps: Vec<(&'static str, u128)>,
}

impl Laps {
    fn start() -> Laps {
        let now = Instant::now();
        Laps {
            start: now,
            mark: now,
            laps: Vec::new(),
        }
    }

    /// Ends the step `name` (the time since the previous lap or the start).
    fn lap(&mut self, name: &'static str) {
        let now = Instant::now();
        self.laps.push((name, (now - self.mark).as_millis()));
        self.mark = now;
    }

    /// A step that ran beside the laps (its own `ms`); the mark stays.
    fn record(&mut self, name: &'static str, ms: u128) {
        self.laps.push((name, ms));
    }

    fn line(&self) -> String {
        let mut out = String::new();
        for (name, ms) in &self.laps {
            let _ = write!(out, "{name}={ms} ");
        }
        let _ = write!(out, "total_ms={}", (self.mark - self.start).as_millis());
        out
    }
}

/// SMOKE: one greedy token through the request path (sidecar encode, PREFILL on a fresh session that is released
/// afterwards); passes when exactly one token is committed without error.
fn smoke(
    driver: &Driver,
    model: &Model,
    sidecar: &Sidecar,
    max_context_tokens: u64,
    stop_token_ids: &[u32],
) -> Result<u32, String> {
    let prompt = sidecar
        .encode_chat(
            &[ChatMessage::text(Role::User, SMOKE_MESSAGE)],
            None,
            false,
            &[],
            &TemplateKwargs::default(),
        )
        .map_err(|e| e.to_string())?
        .ids;
    let session = driver
        .create_session(model, max_context_tokens)
        .map_err(|e| e.to_string())?;
    let work = WorkRequest::prefill(prompt, 1).with_stop_tokens(stop_token_ids.to_vec());
    let outcome = driver.submit(&session, work).and_then(|job| job.wait());
    let released = driver.release_session(session);
    let result = outcome.map_err(|e| e.to_string())?;
    released.map_err(|e| e.to_string())?;
    match (result.status, result.tokens.as_slice()) {
        (Err(status), _) => Err(format!("the forward pass ended with {}", status.name())),
        (Ok(()), [token]) => Ok(*token),
        (Ok(()), tokens) => Err(format!("{} tokens committed, 1 requested", tokens.len())),
    }
}

/// `ffi::EXECUTION_MODE_*` under the capabilities file's names.
fn execution_mode_name(mode: u32) -> String {
    match mode {
        ffi::EXECUTION_MODE_NATIVE => "native".to_owned(),
        ffi::EXECUTION_MODE_BACKEND_MANAGED => "backend_managed".to_owned(),
        ffi::EXECUTION_MODE_SERIAL => "serial".to_owned(),
        ffi::EXECUTION_MODE_UNSUPPORTED => "unsupported".to_owned(),
        other => format!("unknown({other})"),
    }
}

/// The KV pool the executor must report (m25 design note D4, ledger CR1): the configured non-zero
/// `kv_pool_tokens`, else the derived pool the executor builds for the model desc, `rows` (the plan's
/// `max_sessions`, the profile's `decode_batch_max`) x the context rounded up to 64-token pages.
fn expected_kv_pool_tokens(n: &NativeConfig, rows: u64) -> u64 {
    match n.kv_pool_tokens {
        Some(pool) if pool != 0 => pool,
        _ => rows * n.max_context_tokens.div_ceil(64) * 64,
    }
}

/// m32n design note D6: the KV pool admission tracks per native replica of a multi-replica deployment (the pool
/// readiness requires each replica's executor to report); `None` for the single form and every other kind, whose
/// pool admission leaves untracked.
pub(crate) fn replica_pool_tokens(config: &RuntimeConfig) -> Option<u64> {
    let deployment = config.deployments.first()?;
    let native = deployment.native.as_ref()?;
    native.replicas.as_ref()?;
    let profile = &deployment.lock.as_ref()?.profile;
    Some(expected_kv_pool_tokens(native, profile.decode_batch_max))
}

/// Every capability the executor reports, compared with the deployment capabilities file, its KV pool with
/// `kv_pool_tokens` (the configured or derived pool), `native_prefix_cache` with `prefix_state_slots > 0` (M2.10
/// D6, ledger C 60) and its draft tokens with the config's `speculative_draft_tokens` (m29 design note D11/D12,
/// I3); one `capabilities.<field>` problem per difference.
fn capability_problems(
    reported: &ffi::ExecutionCapabilities,
    profile: &ExecutionCapabilities,
    kv_pool_tokens: u64,
    prefix_state_slots: u64,
    speculative_draft_tokens: u32,
) -> Vec<Problem> {
    let mut operations: Vec<&str> = [
        (ffi::OPS_PREFILL, "prefill"),
        (ffi::OPS_DECODE, "decode"),
        (ffi::OPS_VERIFY, "verify"),
        (ffi::OPS_CANCEL, "cancel"),
        (ffi::OPS_SNAPSHOT, "snapshot"),
        (ffi::OPS_RESTORE, "restore"),
    ]
    .into_iter()
    .filter(|(bit, _)| reported.operations & bit != 0)
    .map(|(_, name)| name)
    .collect();
    operations.sort_unstable();
    let mut pinned_operations: Vec<&str> = profile.operations.iter().map(String::as_str).collect();
    pinned_operations.sort_unstable();
    let flag = |v: u8| json!(v != 0);
    let fields = [
        (
            "execution_mode",
            json!(profile.execution_mode),
            json!(execution_mode_name(reported.execution_mode)),
        ),
        ("operations", json!(pinned_operations), json!(operations)),
        (
            "decode_batch_min",
            json!(profile.decode_batch_min),
            json!(reported.decode_batch_min),
        ),
        (
            "decode_batch_max",
            json!(profile.decode_batch_max),
            json!(reported.decode_batch_max),
        ),
        (
            "max_context_tokens",
            json!(profile.max_context_tokens),
            json!(reported.max_context_tokens),
        ),
        (
            "kv_pool_tokens",
            json!(kv_pool_tokens),
            json!(reported.kv_pool_tokens),
        ),
        (
            "native_decode_batch",
            json!(profile.native_decode_batch),
            flag(reported.native_decode_batch),
        ),
        (
            "native_mixed_prefill_decode",
            json!(profile.native_mixed_prefill_decode),
            flag(reported.native_mixed_prefill_decode),
        ),
        (
            "native_speculative_batch",
            json!(profile.native_speculative_batch),
            flag(reported.native_speculative_batch),
        ),
        (
            "native_moe_grouping",
            json!(profile.native_moe_grouping),
            flag(reported.native_moe_grouping),
        ),
        (
            "native_state_snapshot",
            json!(profile.native_state_snapshot),
            flag(reported.native_state_snapshot),
        ),
        (
            "native_prefix_cache",
            json!(prefix_state_slots > 0),
            flag(reported.native_prefix_cache),
        ),
        (
            "speculative_draft_tokens",
            json!(speculative_draft_tokens),
            json!(reported.speculative_draft_tokens),
        ),
    ];
    let mut problems: Vec<Problem> = fields
        .into_iter()
        .filter(|(_, pinned, actual)| pinned != actual)
        .map(|(field, pinned, actual)| {
            problem(
                &format!("capabilities.{field}"),
                format!("the deployment pins {pinned}, the executor reports {actual}"),
            )
        })
        .collect();
    // M2.8 C1: grammar outputs are served by masking the executor's pick; a profile without one accepts either.
    if native::advertises_grammar(profile) && reported.native_token_mask != 1 {
        problems.push(problem(
            "capabilities.native_token_mask",
            format!(
                "the deployment pins grammar outputs, which need native_token_mask 1; the executor reports {}",
                reported.native_token_mask
            ),
        ));
    }
    // M2.7 D11: sampled rows are served by the library's device sampler; a greedy-only profile accepts either.
    let sampled = profile
        .validated_combinations
        .iter()
        .any(|c| c.sampler_mode == "sampled");
    if sampled && reported.native_sampler != 1 {
        problems.push(problem(
            "capabilities.native_sampler",
            format!(
                "the deployment pins sampled decode rows, which need native_sampler 1; the executor reports {}",
                reported.native_sampler
            ),
        ));
    }
    problems
}

/// BC1 leftover (batch-concurrency design note "Prefill selection signature"): a deployment capabilities file
/// that declares `prefill_signature` pins the loaded model's prefill kernel selection; the
/// `capabilities.prefill_signature` problem when the executor reports another digest or none (a library before
/// `cere_model_prefill_signature`, a device without a selection). Undeclared, any or no signature is accepted.
fn prefill_signature_problem(
    reported: Option<&str>,
    profile: &ExecutionCapabilities,
) -> Option<Problem> {
    let pinned = profile.prefill_signature.as_deref()?;
    (reported != Some(pinned)).then(|| {
        problem(
            "capabilities.prefill_signature",
            format!(
                "the deployment pins {pinned}, the executor reports {}",
                reported.unwrap_or("none")
            ),
        )
    })
}

/// `prefix` + the first 8 bytes (16 hex digits) of the SHA-256 of `identity`'s compact JSON.
fn prefixed_id(prefix: &str, identity: &Value) -> String {
    let digest = Sha256::digest(identity.to_string().as_bytes());
    let mut id = String::from(prefix);
    for byte in &digest[..8] {
        // Writing to a String cannot fail.
        let _ = write!(id, "{byte:02x}");
    }
    id
}

/// The backend launch (Q10): lowercase hex sha256 of the compact JSON array `[random_seed, startup_time]`, the
/// values as `/get_server_info` reports them (re-serialized by serde_json: object keys sorted, no whitespace).
/// `None` unless both are present and non-null.
fn launch(info: &Value) -> Option<String> {
    let seed = info.get("random_seed").filter(|v| !v.is_null())?;
    let startup = info.get("startup_time").filter(|v| !v.is_null())?;
    let digest = Sha256::digest(json!([seed, startup]).to_string().as_bytes());
    let mut hex = String::with_capacity(2 * digest.len());
    for byte in &digest {
        // Writing to a String cannot fail.
        let _ = write!(hex, "{byte:02x}");
    }
    Some(hex)
}

fn problem(field: &str, detail: String) -> Problem {
    Problem {
        field: field.to_owned(),
        detail,
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

// ---------------------------------------------------------------------------------------------------------
// Remote readiness (m35 design note Q7a, Q9, Q10, Q10a)

/// The scope id of the remote readiness smoke on the agent (build-loop-m35 design note "Wire behaviour").
pub const REMOTE_SMOKE_SCOPE: &str = "cere-smoke";

/// The `work_unit_json` of a unary chat over the RPC.
const UNARY_CHAT_UNIT: &[u8] = br#"{"kind":"chat","stream":false}"#;

/// The failed checks of each remote worker's Joining round, for `/readyz`; later passes of the round's worker
/// (Ready, Lost, Draining) refresh only the residency-owned problems (`residency`, `lock_mismatch`).
#[derive(Debug, Default)]
pub struct RemoteProblems(Mutex<BTreeMap<String, (u64, Vec<Problem>)>>);

/// The problem fields the Q8 residency step owns: it runs on every pass whatever the worker's state.
const RESIDENCY_FIELDS: [&str; 2] = ["residency", "lock_mismatch"];

impl RemoteProblems {
    /// The problems found in round `round` of `worker_id` (none for another round).
    pub fn get(&self, worker_id: &str, round: u64) -> Vec<Problem> {
        let problems = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        match problems.get(worker_id) {
            Some((at, problems)) if *at == round => problems.clone(),
            _ => Vec::new(),
        }
    }

    /// A Joining pass of round `round`: its problems replace the stored ones.
    fn set(&self, worker_id: &str, round: u64, problems: Vec<Problem>) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(worker_id.to_owned(), (round, problems));
    }

    /// A pass of a worker no longer Joining in round `round`: `residency` holds only residency-owned problems,
    /// which replace the stored ones (or remove them); the round's other problems stay.
    fn refresh(&self, worker_id: &str, round: u64, mut residency: Vec<Problem>) {
        let mut all = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        match all.get_mut(worker_id) {
            Some((at, stored)) if *at == round => {
                residency.extend(
                    stored
                        .drain(..)
                        .filter(|p| !RESIDENCY_FIELDS.contains(&p.field.as_str())),
                );
                *stored = residency;
            }
            _ => {
                all.insert(worker_id.to_owned(), (round, residency));
            }
        }
    }
}

/// Whether the driver checks again at once or waits for a round bump or the probe interval.
enum Next {
    Now,
    Wait,
}

/// The scope of the coordinator's residency calls (m36 design note Q8).
const CONTROL_SCOPE: &str = "_cere_control";

/// One residency call (m36 design note Q8; design-m36-coord design note §3.1): at most one per
/// `(intent_seq, epoch, term)`; a new intent or epoch replaces it.
struct Call {
    intent_seq: u64,
    epoch: u64,
    term: u64,
    method: &'static str,
    /// `residency-<term>-<intent_seq>-<epoch>-<load|unload>` in [`CONTROL_SCOPE`].
    op_id: String,
    /// Built on the first try and sent unchanged on every retry; dropped by a `stale_*` refusal (the agent
    /// journaled nothing, Q6 order) and rebuilt under the same op id on the next round.
    authority: Option<pb::Authority>,
    receipted: bool,
    /// The op ended, or the call was refused for good; a non-empty `reason` is `/readyz` problem `residency`
    /// until the next intent.
    done: bool,
    reason: String,
}

/// The driver's residency bookkeeping, owned by its task.
struct Residency {
    /// The coordinator's deployment lock as the inventory spells it (lowercase hex).
    lock_hex: String,
    call: Option<Call>,
    /// `(epoch, intent_seq)` when the model was first seen `failed` (cleared once it is not): `resident` +
    /// `failed` loads only under a later intent (no implicit retry, MODEL_LIFECYCLE:104).
    failed_under: Option<(u64, u64)>,
}

/// The Q8 table: the lifecycle call that `desired` asks of a model described in `state`; `resident` + `failed`
/// loads only for an intent issued after the failure was seen (`reset`, the operator's explicit reset).
fn residency_call(desired: Desired, state: ModelState, reset: bool) -> Option<&'static str> {
    match (desired, state) {
        (Desired::Resident, ModelState::Unloaded | ModelState::Draining) => Some(wire::METHOD_LOAD),
        (Desired::Resident, ModelState::Failed) if reset => Some(wire::METHOD_LOAD),
        (
            Desired::Unloaded,
            ModelState::Loading | ModelState::Ready | ModelState::Draining | ModelState::Failed,
        ) => Some(wire::METHOD_UNLOAD),
        _ => None,
    }
}

/// Drives one remote worker from Joining to Ready, one round at a time. Each pass reads the worker's state and
/// runs the first check that has not passed in its round: DescribeResources (the exact bytes must hash to the
/// agent-reported `inventory_sha256`, Q10a), the `[federation.workers.expect]` pins, reconciliation (Q9: cancel
/// every live operation no live ledger request on this worker incarnation owns, then `mark_reconciled`), then,
/// once the agent reports `open`, the smoke (one unary 1-token chat in scope [`REMOTE_SMOKE_SCOPE`]; a failed
/// smoke is cancelled, detached) and `mark_ready`. Every transition names the round it was computed in; the
/// registry refuses a stale one, and the pass starts over.
///
/// Residency (m36 design note Q8, Q10; design-m36-coord design note §3): every pass in any state, and again right
/// after a describe, the driver issues the Q8 table's `LoadModel`/`UnloadModel` for the operator's intent on a
/// managed worker's current inventory, and a managed worker whose model is not `ready` under the coordinator's
/// lock or not wanted skips the pins, reconciliation and the smoke and stays Joining.
pub(crate) struct RemoteDriver {
    pub worker: RemoteWorkerConfig,
    pub cluster_id: String,
    pub coordinator_id: String,
    pub registry: Arc<RemoteRegistry>,
    pub client: WorkerClient,
    /// Weak: the driver never keeps the ledger (and its state-dir lock) alive; it stops once it is gone.
    pub ledger: Weak<Mutex<Ledger>>,
    pub rids: Arc<RidGenerator>,
    /// The local smoke's deadline cap.
    pub smoke_deadline: Duration,
    pub interval: Duration,
    pub problems: Arc<RemoteProblems>,
    /// P7: the remote smoke waits for an admission slot of its worker, like the local smoke; a Ready transition
    /// is pushed to admission at once.
    pub admission: Admission,
    /// The lock of this worker's deployment (`config.deployment_of_worker`, M5 D8), the only one its residency
    /// calls name (Q8).
    pub deployment_lock: [u8; 32],
}

impl RemoteDriver {
    /// Runs until the ledger is gone (or the task is aborted), woken on every round bump and at the interval.
    pub(crate) fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut rounds = self.registry.subscribe();
            let mut residency = Residency {
                lock_hex: hex(&self.deployment_lock),
                call: None,
                failed_under: None,
            };
            while self.ledger.strong_count() > 0 {
                if let Next::Now = self.pass(&mut residency).await {
                    continue;
                }
                tokio::select! {
                    _ = rounds.changed() => {}
                    _ = tokio::time::sleep(self.interval) => {}
                }
            }
        })
    }

    fn id(&self) -> &str {
        &self.worker.worker_id
    }

    async fn pass(&self, residency: &mut Residency) -> Next {
        let Some(state) = self.registry.get(self.id(), unix_ms()) else {
            return Next::Wait;
        };
        let (Some(epoch), Some(term)) = (state.epoch, state.term) else {
            return Next::Wait;
        };
        let at = AuthorityBase {
            cluster_id: self.cluster_id.clone(),
            worker_id: self.worker.worker_id.clone(),
            worker_epoch: epoch,
            coordinator_id: self.coordinator_id.clone(),
            coordinator_term: term,
            model_generation: state.generation,
        };
        let mut problems = Vec::new();
        // Q8: on every probe round, whatever the worker's state.
        if self.residency(&state, &at, residency, &mut problems).await {
            return Next::Now;
        }
        if state.state != WorkerState::Joining {
            self.problems.refresh(self.id(), state.round, problems);
            return Next::Wait;
        }
        let next = self.checks(&state, &at, residency, &mut problems).await;
        self.problems.set(self.id(), state.round, problems);
        next
    }

    /// A registry refusal: a stale round starts over at once, anything else waits for the next pass.
    fn refused(&self, round: u64, error: RemoteError) -> Next {
        match error {
            RemoteError::StaleRound { current } => {
                eprintln!(
                    "cered remote worker={} event=stale_round round={round} current={current}",
                    self.id()
                );
                Next::Now
            }
            _ => Next::Wait,
        }
    }

    async fn checks(
        &self,
        state: &RemoteState,
        at: &AuthorityBase,
        residency: &mut Residency,
        problems: &mut Vec<Problem>,
    ) -> Next {
        let round = state.round;
        let inventory = match (&state.inventory, state.inventory_sha256) {
            (Some(inventory), Some(sha)) if Some(sha) == state.reported_inventory_sha256 => {
                inventory.clone()
            }
            _ => match self.describe(state, at).await {
                Ok((inventory, sha)) => {
                    let recorded =
                        self.registry
                            .record_inventory(self.id(), round, inventory.clone(), sha);
                    if let Err(e) = recorded {
                        return self.refused(round, e);
                    }
                    // Q8: reconcile the new description at once.
                    if let Some(described) = self.registry.get(self.id(), unix_ms()) {
                        if self.residency(&described, at, residency, problems).await {
                            return Next::Now;
                        }
                    }
                    inventory
                }
                Err(failed) => {
                    problems.push(failed);
                    return Next::Wait;
                }
            },
        };
        // Q10 (review M36-03): a managed model is pin-checked, reconciled and smoked only while it is `ready` under
        // the coordinator's lock and wanted; otherwise the worker stays Joining without pin or smoke problems.
        if let Some(model) = inventory.model() {
            if model.state != ModelState::Attached
                && (model.state != ModelState::Ready
                    || model.lock_sha256.as_deref() != Some(residency.lock_hex.as_str())
                    || state.model.desired == Desired::Unloaded)
            {
                return Next::Wait;
            }
        }
        let pins = pin_problems(&self.worker.expect, &inventory);
        if !pins.is_empty() {
            problems.extend(pins);
            return Next::Wait;
        }
        // Q9 for the generation of the last accepted heartbeat.
        let Some(generation) = state.generation else {
            return Next::Wait;
        };
        if state.reconciled_generation != Some(generation) {
            if let Err(detail) = self.reconcile(at).await {
                problems.push(problem("reconcile", detail));
                return Next::Wait;
            }
            // The agent opens on the next heartbeat reply carrying the reconciled generation; its `open` report
            // wakes the driver (O182), so an `open` read now may predate the reconcile. A draining worker waits for
            // no report (its agent stays fenced): the next pass marks it.
            return match self.registry.mark_reconciled(self.id(), round, generation) {
                Ok(()) if state.draining => Next::Now,
                Ok(()) => Next::Wait,
                Err(e) => self.refused(round, e),
            };
        }
        // Q13: a draining agent fences and refuses SubmitWork, so a draining worker becomes Ready (read as
        // draining) without the smoke; undrain starts a new round that smokes it.
        if !state.draining && state.admission_state != Some(AdmissionState::Open) {
            return Next::Wait;
        }
        if !state.draining && !matches!(state.smoke, SmokeStatus::Passed { .. }) {
            // The pins passed, so the identity fields are set (a validated inventory sets all or none).
            let Some(instance_id) = inventory.instance_id.clone() else {
                return Next::Wait;
            };
            let smoke = match self.smoke(at).await {
                Ok(()) => SmokeStatus::Passed { instance_id },
                Err(detail) => {
                    eprintln!(
                        "cered remote worker={}@{} event=smoke state=failed round={round}",
                        self.id(),
                        at.worker_epoch
                    );
                    problems.push(problem("smoke", detail.clone()));
                    SmokeStatus::Failed {
                        instance_id,
                        detail,
                    }
                }
            };
            let passed = matches!(smoke, SmokeStatus::Passed { .. });
            if let Err(e) = self.registry.set_smoke(self.id(), round, smoke) {
                return self.refused(round, e);
            }
            if !passed {
                return Next::Wait;
            }
        }
        match self.registry.mark_ready(self.id(), round, unix_ms()) {
            Ok(()) => {
                crate::fed::remote::push_view(&self.registry, &self.admission, &self.worker);
                eprintln!(
                    "cered remote worker={}@{} event=ready round={round} generation={generation}",
                    self.id(),
                    at.worker_epoch
                );
                Next::Wait
            }
            Err(e) => self.refused(round, e),
        }
    }

    /// Q10a: DescribeResources; the exact bytes must hash to the `inventory_sha256` the agent reported.
    async fn describe(
        &self,
        state: &RemoteState,
        at: &AuthorityBase,
    ) -> Result<(Inventory, [u8; 32]), Problem> {
        let reply = self.client.describe(at).await.map_err(|e| {
            problem(
                "inventory",
                format!("DescribeResources: {}", status_text(&e)),
            )
        })?;
        let bytes = reply.bounded_resource_inventory_json;
        let sha = Inventory::sha256(&bytes);
        if state.reported_inventory_sha256 != Some(sha) {
            let reported = state
                .reported_inventory_sha256
                .map_or_else(|| "none".to_owned(), |r| hex(&r));
            return Err(problem(
                "inventory_sha256",
                format!(
                    "the described inventory hashes to {}, the agent reports {reported}",
                    hex(&sha)
                ),
            ));
        }
        let inventory = Inventory::from_json_bytes(&bytes)
            .map_err(|e| problem("inventory", format!("DescribeResources: {e}")))?;
        Ok((inventory, sha))
    }

    /// The Q8 residency step (design-m36-coord design note §3.1) on `state`'s inventory, acted on only when it is
    /// managed, current (its bytes are the reported ones) and described in the worker's current epoch: a model
    /// under another lock is problem `lock_mismatch` and gets no call; otherwise the table's call for the
    /// operator's intent goes out at most once per `(intent_seq, epoch, term)`, is retried unchanged until
    /// receipted, then polled to its terminal. Returns `true` once it took the worker into a new Joining round.
    async fn residency(
        &self,
        state: &RemoteState,
        at: &AuthorityBase,
        residency: &mut Residency,
        problems: &mut Vec<Problem>,
    ) -> bool {
        let model = &state.model;
        let (Some(epoch), Some(term), Some(described)) = (state.epoch, state.term, model.state)
        else {
            return false;
        };
        if model.managed != Some(true)
            || state.inventory_epoch != Some(epoch)
            || state.inventory.is_none()
            || state.inventory_sha256 != state.reported_inventory_sha256
        {
            return false;
        }
        if model.lock_sha256.as_deref() != Some(residency.lock_hex.as_str()) {
            problems.push(problem(
                "lock_mismatch",
                format!(
                    "the worker's model is under lock {}, the coordinator's is {}",
                    model.lock_sha256.as_deref().unwrap_or("none"),
                    residency.lock_hex
                ),
            ));
            return false;
        }
        if described != ModelState::Failed {
            residency.failed_under = None;
        } else if residency
            .failed_under
            .is_none_or(|(seen_epoch, _)| seen_epoch != epoch)
        {
            residency.failed_under = Some((epoch, model.intent_seq));
        }
        if residency
            .call
            .as_ref()
            .is_some_and(|c| (c.intent_seq, c.epoch, c.term) != (model.intent_seq, epoch, term))
        {
            residency.call = None;
        }
        let call = match &mut residency.call {
            Some(call) => call,
            None => {
                let reset = residency
                    .failed_under
                    .is_some_and(|(_, seen)| model.intent_seq > seen);
                let Some(method) = residency_call(model.desired, described, reset) else {
                    return false;
                };
                let verb = if method == wire::METHOD_LOAD {
                    "load"
                } else {
                    "unload"
                };
                residency.call.insert(Call {
                    intent_seq: model.intent_seq,
                    epoch,
                    term,
                    method,
                    op_id: format!("residency-{term}-{}-{epoch}-{verb}", model.intent_seq),
                    authority: None,
                    receipted: false,
                    done: false,
                    reason: String::new(),
                })
            }
        };
        if call.done {
            if !call.reason.is_empty() {
                problems.push(problem("residency", call.reason.clone()));
            }
            return false;
        }
        if call.receipted {
            self.poll(call, at, state.round).await
        } else {
            self.send(call, state, at).await
        }
    }

    /// One try of an unreceipted residency call under its Authority (built on the first try with the generation of
    /// the last accepted heartbeat, none before the first one: the agent would refuse `stale_generation`; else
    /// reused unchanged). A transport failure, `model_busy` or `quota_exhausted` is retried on a later round; a
    /// `stale_*` refusal drops the Authority (rebuilt next round, same op id) and re-joins; any other refusal ends
    /// the call for this `(intent_seq, epoch, term)`. Returns `true` once it re-joined the worker.
    async fn send(&self, call: &mut Call, state: &RemoteState, at: &AuthorityBase) -> bool {
        let authority = match (&call.authority, state.generation) {
            (Some(authority), _) => authority.clone(),
            (None, None) => return false,
            (None, Some(generation)) => call
                .authority
                .insert(
                    AuthorityBase {
                        model_generation: Some(generation),
                        ..at.clone()
                    }
                    .op(
                        CONTROL_SCOPE,
                        &call.op_id,
                        wire::lifecycle_sha256(
                            call.method,
                            CONTROL_SCOPE,
                            &self.deployment_lock,
                            "",
                            &[],
                        )
                        .to_vec(),
                        wire::WORK_CAP_MS,
                    ),
                )
                .clone(),
        };
        let log = |what: &str| {
            eprintln!(
                "cered remote worker={}@{} event=residency method={} operation={} {what}",
                self.id(),
                call.epoch,
                call.method,
                call.op_id
            );
        };
        let refusal = match self
            .client
            .lifecycle(call.method, authority, &self.deployment_lock)
            .await
        {
            Ok(receipt) if receipt.accepted => {
                log("receipted");
                call.receipted = true;
                return false;
            }
            Ok(receipt) => format!("not accepted: {}", receipt.reason_code),
            Err(status) => match wire::reason(&status) {
                wire::MODEL_BUSY | wire::QUOTA_EXHAUSTED => {
                    log(&format!("retry={}", wire::reason(&status)));
                    return false;
                }
                wire::STALE_TERM | wire::STALE_EPOCH | wire::STALE_GENERATION => {
                    log(&format!("rebuild={}", wire::reason(&status)));
                    call.authority = None;
                    return self.rejoin(state.round);
                }
                _ if is_transport(&status) => {
                    log("retry=transport");
                    return false;
                }
                _ => status_text(&status),
            },
        };
        log(&format!("refused={refusal}"));
        call.done = true;
        call.reason = format!("{} {} refused: {refusal}", call.method, call.op_id);
        false
    }

    /// `GetOperation` of a receipted residency call; on its terminal the call is done (a non-empty reason kept
    /// for `/readyz`) and the worker re-joins to re-describe. Returns `true` once it re-joined the worker.
    async fn poll(&self, call: &mut Call, at: &AuthorityBase, round: u64) -> bool {
        let status = match self
            .client
            .get_operation(at, CONTROL_SCOPE, &call.op_id)
            .await
        {
            Ok(status) => status,
            Err(e) => {
                eprintln!(
                    "cered remote worker={}@{} event=residency operation={} poll={}",
                    self.id(),
                    call.epoch,
                    call.op_id,
                    status_text(&e)
                );
                return false;
            }
        };
        if !status.terminal {
            return false;
        }
        eprintln!(
            "cered remote worker={}@{} event=residency method={} operation={} terminal reason={}",
            self.id(),
            call.epoch,
            call.method,
            call.op_id,
            status.reason_code
        );
        call.done = true;
        if !status.reason_code.is_empty() {
            call.reason = format!(
                "{} {} ended {}",
                call.method, call.op_id, status.reason_code
            );
        }
        self.rejoin(round)
    }

    /// Takes the worker from round `round` into a new Joining round (it re-describes) and shows admission the new
    /// view; `true` unless the worker is gone (Lost or no longer registered).
    fn rejoin(&self, round: u64) -> bool {
        let joined = match self.registry.enter_joining(self.id(), round) {
            Ok(_) | Err(RemoteError::StaleRound { .. }) => true,
            Err(e) => {
                eprintln!(
                    "cered remote worker={} not rejoined after a residency call: {e:?}",
                    self.id()
                );
                false
            }
        };
        crate::fed::remote::push_view(&self.registry, &self.admission, &self.worker);
        joined
    }

    /// Q9: lists every operation the agent holds and cancels each live SubmitWork no live ledger request
    /// dispatched to this worker incarnation owns, under the cancel id `<request_id>/cancel/<term>`.
    async fn reconcile(&self, at: &AuthorityBase) -> Result<(), String> {
        let operations = self
            .client
            .list_all(at)
            .await
            .map_err(|e| format!("ListOperations: {}", status_text(&e)))?;
        let owned: BTreeSet<(String, String)> = {
            let ledger = self.ledger.upgrade().ok_or("the coordinator stopped")?;
            let ledger = ledger.lock().unwrap_or_else(PoisonError::into_inner);
            ledger
                .in_flight()
                .into_iter()
                .filter(|r| {
                    r.worker
                        .as_ref()
                        .is_some_and(|(id, epoch)| id == self.id() && *epoch == at.worker_epoch)
                })
                .map(|r| (r.scope, r.rid))
                .collect()
        };
        for op in operations.iter().filter(|op| {
            op.method == wire::METHOD_SUBMIT
                && !op.terminal
                && !owned.contains(&(op.scope_id.clone(), op.request_id.clone()))
        }) {
            let cancel_id = format!("{}/cancel/{}", op.request_id, at.coordinator_term);
            let target = OpRef {
                scope_id: op.scope_id.clone(),
                request_id: op.request_id.clone(),
                attempt: op.attempt,
                completion_ticket: op.completion_ticket.clone(),
            };
            self.client
                .cancel(at, &cancel_id, &target)
                .await
                .map_err(|e| {
                    format!(
                        "CancelWork of {}/{}: {}",
                        op.scope_id,
                        op.operation_id,
                        status_text(&e)
                    )
                })?;
            eprintln!(
                "cered remote worker={}@{} event=reconcile cancelled={}/{} operation={cancel_id}",
                self.id(),
                at.worker_epoch,
                op.scope_id,
                op.operation_id
            );
        }
        Ok(())
    }

    /// The remote smoke: SubmitWork of one unary 1-token chat (the local smoke's body and rid minting) in scope
    /// [`REMOTE_SMOKE_SCOPE`] with operation id = request id = the rid, then StreamWork from 0; a single response
    /// event with a `stop` or full-budget `length` finish, then a natural terminal, passes. Like the local smoke
    /// it first waits for an admission slot of its worker (internal scope [`SMOKE_SCOPE`], P7). On failure the
    /// smoke is cancelled (`<rid>/cancel/1`) by a detached task that keeps the slot until the cancel ended.
    async fn smoke(&self, at: &AuthorityBase) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + self.smoke_deadline;
        let permit = self
            .admission
            .acquire(
                SMOKE_SCOPE,
                Priority::Normal,
                deadline,
                admission::Target::Only(Arc::from(self.id())),
                1,
            )
            .await
            .map_err(|e| format!("no admission slot for the smoke: {}", e.code()))?;
        let rid = self.rids.next_rid();
        let body = smoke_body(rid.as_str(), &self.worker.expect.served_model_name);
        let work = Work {
            scope_id: REMOTE_SMOKE_SCOPE.to_owned(),
            operation_id: rid.as_str().to_owned(),
            request_id: rid.as_str().to_owned(),
            attempt: 1,
            work_unit_json: UNARY_CHAT_UNIT.to_vec(),
            input_json: serde_json::to_vec(&body).map_err(|e| e.to_string())?,
        };
        let mut ticket = None;
        let outcome =
            tokio::time::timeout_at(deadline, self.run_smoke(at, &work, deadline, &mut ticket))
                .await
                .unwrap_or_else(|_| Err("the smoke did not finish before its deadline".to_owned()));
        if outcome.is_err() {
            self.cancel_smoke(at.clone(), work, ticket, permit);
        }
        outcome
    }

    async fn run_smoke(
        &self,
        at: &AuthorityBase,
        work: &Work,
        deadline: tokio::time::Instant,
        ticket: &mut Option<String>,
    ) -> Result<(), String> {
        let budget_ms = || {
            deadline
                .saturating_duration_since(tokio::time::Instant::now())
                .as_millis() as u64
        };
        let receipt = self
            .client
            .submit(at, work, budget_ms())
            .await
            .map_err(|e| format!("SubmitWork: {}", status_text(&e)))?;
        *ticket = Some(receipt.completion_ticket.clone());
        let target = OpRef {
            scope_id: work.scope_id.clone(),
            request_id: work.request_id.clone(),
            attempt: work.attempt,
            completion_ticket: receipt.completion_ticket,
        };
        let mut events = self
            .client
            .stream(at, &work.operation_id, &target, 0, budget_ms())
            .await
            .map_err(|e| format!("StreamWork: {}", status_text(&e)))?;
        let mut response = None;
        loop {
            let event = events
                .message()
                .await
                .map_err(|e| format!("StreamWork: {}", status_text(&e)))?
                .ok_or("the work stream ended without a terminal event")?;
            if event.terminal {
                if !event.reason_code.is_empty() {
                    return Err(format!("the smoke ended with {}", event.reason_code));
                }
                break;
            }
            if response.is_some() {
                return Err("more than one response event for a unary chat".to_owned());
            }
            let value: Value = serde_json::from_slice(&event.bounded_output_event_json)
                .map_err(|e| format!("unparsable response event: {e}"))?;
            response = Some(value);
        }
        let response = response.ok_or("no response event before the terminal event")?;
        match classify_response(&response, 1, 1).map_err(|e| e.to_string())? {
            Finish::Stop | Finish::Length => Ok(()),
            other => Err(format!("smoke chat finished with {}", other.as_str())),
        }
    }

    /// Detached CancelWork of a failed smoke; a lost Submit receipt is fetched again by retrying the Submit
    /// (dedupe answers the original receipt, Q12).
    fn cancel_smoke(&self, at: AuthorityBase, work: Work, ticket: Option<String>, permit: Permit) {
        let client = self.client.clone();
        let budget_ms = self.smoke_deadline.as_millis() as u64;
        tokio::spawn(async move {
            let ticket = match ticket {
                Some(ticket) => ticket,
                None => match client.submit(&at, &work, budget_ms).await {
                    Ok(receipt) => receipt.completion_ticket,
                    // Refused again: the agent holds no such operation.
                    Err(_) => return,
                },
            };
            let target = OpRef {
                scope_id: work.scope_id,
                request_id: work.request_id,
                attempt: work.attempt,
                completion_ticket: ticket,
            };
            let cancel_id = format!("{}/cancel/1", target.request_id);
            if let Err(e) = client.cancel(&at, &cancel_id, &target).await {
                eprintln!(
                    "cered readiness: cancel of remote smoke rid={} on worker={} failed: {}",
                    target.request_id,
                    at.worker_id,
                    status_text(&e)
                );
            }
            drop(permit);
        });
    }
}

/// One problem per `[federation.workers.expect]` pin the described inventory breaks. A null identity field (no
/// confirmed backend, m36 design note Q7) breaks its pin.
fn pin_problems(expect: &RemoteExpect, inventory: &Inventory) -> Vec<Problem> {
    let mut problems = Vec::new();
    let mut pin = |field: &str, expected: String, found: Option<String>| {
        if Some(&expected) != found.as_ref() {
            let found = found.as_deref().unwrap_or("null");
            problems.push(problem(
                field,
                format!("expected {expected}, the worker reports {found}"),
            ));
        }
    };
    pin(
        "served_model_name",
        expect.served_model_name.clone(),
        Some(inventory.served_model_name.clone()),
    );
    pin(
        "tp_size",
        expect.tp_size.to_string(),
        inventory.tp_size.map(|n| n.to_string()),
    );
    pin(
        "kv_cache_dtype",
        expect.kv_cache_dtype.clone(),
        inventory.kv_cache_dtype.clone(),
    );
    problems
}

/// A refusal as `<code> <reason>`.
fn status_text(status: &tonic::Status) -> String {
    format!("{:?} {}", status.code(), wire::reason(status))
}

#[cfg(test)]
mod tests {
    #[test]
    fn binding_problem_checks_the_cuda_and_hip_bindings() {
        let some = |v: &str| Some(v.to_owned());
        let uuid = "GPU-aaaaaaaa-0000-4000-8000-000000000000";
        assert_eq!(super::binding_problem(uuid, some(uuid), None), None);
        assert_eq!(super::binding_problem(uuid, some(uuid), some("0")), None);
        assert!(super::binding_problem(uuid, None, None)
            .unwrap()
            .contains("expected CUDA_VISIBLE_DEVICES"));
        assert!(super::binding_problem(uuid, some("GPU-other"), None).is_some());
        assert_eq!(super::binding_problem("hip:0", None, some("0")), None);
        assert_eq!(super::binding_problem("hip:12", None, some("12")), None);
        assert!(super::binding_problem("hip:0", None, None)
            .unwrap()
            .contains("expected HIP_VISIBLE_DEVICES 0"));
        assert!(super::binding_problem("hip:0", None, some("1")).is_some());
        assert!(super::binding_problem("hip:0", some("0"), some("0"))
            .unwrap()
            .contains("must be unset"));
        assert!(super::binding_problem("hip:0", some(""), some("0")).is_some());
    }

    /// The Vulkan binding `vk:<ordinal>` reads neither `CUDA_VISIBLE_DEVICES` nor `HIP_VISIBLE_DEVICES`: any
    /// combination of them is accepted (the device is chosen by the executor device id alone).
    #[test]
    fn binding_problem_ignores_the_visible_devices_variables_for_a_vulkan_binding() {
        let some = |v: &str| Some(v.to_owned());
        for cuda in [None, some("0"), some(""), some("GPU-other")] {
            for hip in [None, some("0"), some("1"), some("")] {
                for visible in ["vk:0", "vk:1", "vk:12"] {
                    assert_eq!(
                        super::binding_problem(visible, cuda.clone(), hip.clone()),
                        None,
                        "{visible} {cuda:?} {hip:?}"
                    );
                }
            }
        }
        // A string that is not a `vk:` ordinal falls to the CUDA rule (the config validator refuses it earlier).
        for bad in ["vk:00", "vk:", "vk:x", "VK:0"] {
            assert!(super::binding_problem(bad, None, None).is_some(), "{bad}");
        }
    }

    use super::*;
    use crate::sglang::{Rid, Timeouts};
    use cere_mock_sglang::{Fault, MockConfig, MockHandle, MockSglang, ServerInfo};
    use std::path::Path;

    /// GrammarSpec GS-04: "off" needs nothing; "on" refuses a library without the two calls, a profile without a
    /// grammar output and a grammar process without the feature.
    #[test]
    fn grammar_drafts_on_needs_the_library_calls_and_the_grammar_feature() {
        let info = |drafts| GrammarInfo {
            xgrammar: "0.2.1".to_owned(),
            torch: "2".to_owned(),
            stop_token_ids: vec![1],
            vocab_size: 64,
            drafts,
        };
        let (served, unserved) = (info(true), info(false));
        assert_eq!(
            grammar_drafts_problem(GrammarDrafts::Off, false, None),
            None
        );
        assert_eq!(
            grammar_drafts_problem(GrammarDrafts::On, true, Some(&served)),
            None
        );
        let problem =
            |library, grammar| grammar_drafts_problem(GrammarDrafts::On, library, grammar).unwrap();
        assert!(problem(false, Some(&served)).contains("cere_executor_round_drafts"));
        assert!(problem(true, None).contains("grammar output"));
        assert!(problem(true, Some(&unserved)).contains("serving grammar_drafts"));
    }

    const TIMEOUTS: Timeouts = Timeouts {
        connect: Duration::from_secs(1),
        control: Duration::from_secs(2),
        stream_idle: Duration::from_secs(2),
    };

    fn live_config(base_url: &str) -> RuntimeConfig {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/sglang-t4-qwen38.toml");
        let mut config = RuntimeConfig::load(path).unwrap();
        config.deployments[0].base_url = Some(base_url.to_owned());
        config
    }

    async fn start(port: u16) -> MockHandle {
        MockSglang::start(MockConfig {
            port,
            ..MockConfig::default()
        })
        .await
        .unwrap()
    }

    fn readiness(base_url: &str) -> Readiness {
        let client = SglangHttp::new(base_url, TIMEOUTS).unwrap();
        let admission = Admission::new(crate::admission::AdmissionLimits {
            max_inflight: 1,
            per_scope_inflight: 1,
            queue_depth: 1,
        });
        admission.set_workers(vec![crate::admission::WorkerCap {
            id: "gpu0".to_owned(),
            cap: 1,
            aging: crate::worker::Aging {
                suspect_after_ms: 15_000,
                lost_after_ms: u64::MAX,
            },
            deployment: 0,
        }]);
        Readiness::new(
            client,
            "gpu0",
            &live_config(base_url),
            0,
            Arc::new(RidGenerator::with_nonce(7, 0xabc)),
            admission,
            Duration::from_secs(5),
        )
        .unwrap()
    }

    fn fields(snapshot: &ReadinessSnapshot) -> Vec<&str> {
        snapshot.problems.iter().map(|p| p.field.as_str()).collect()
    }

    #[test]
    fn production_interval_is_q8_minimum() {
        assert_eq!(readiness("http://127.0.0.1:9").interval, PROBE_INTERVAL);
        assert!(PROBE_INTERVAL >= Duration::from_secs(5));
        assert_eq!(crate::app::Tuning::default().probe_interval, PROBE_INTERVAL);
        assert_eq!(
            crate::app::Tuning::default().remote_interval,
            PROBE_INTERVAL
        );
        // Q7/Q10: a local worker turns suspect after three probe periods without a healthy probe.
        assert_eq!(
            crate::app::Tuning::default().suspect_after,
            3 * PROBE_INTERVAL
        );
        // Orphan retention reads the system clock in production (RtFlake2 D1).
        assert!(matches!(
            crate::app::Tuning::default().retention_clock,
            crate::app::RetentionClock::Wall
        ));
    }

    /// Spec m28 D7 (Z2a P2): the native probe reports the grammar process respawning or down as a `grammar`
    /// problem that leaves the worker ready (the tokenizer sidecar and unconstrained serving go on); any other
    /// problem still makes it not ready.
    #[test]
    fn grammar_problem_is_reported_and_keeps_the_worker_ready() {
        assert_eq!(grammar_problem(GrammarHealth::Ready), None);
        let respawning = grammar_problem(GrammarHealth::Respawning).unwrap();
        let down = grammar_problem(GrammarHealth::Down).unwrap();
        assert_eq!(
            (respawning.field.as_str(), down.field.as_str()),
            ("grammar", "grammar")
        );
        assert_ne!(respawning.detail, down.detail);

        let readiness = readiness("http://127.0.0.1:9");
        let smoke = SmokeStatus::Pending;
        let id = Some("native-0123456789abcdef".to_owned());
        let ready = readiness
            .inner
            .publish(Vec::new(), vec![down.clone()], id.clone(), &smoke);
        assert!(ready.ready, "{ready:?}");
        assert_eq!(fields(&ready), ["grammar"]);
        assert_eq!(readiness.snapshot(), ready);

        let sidecar = problem(
            "sidecar",
            "the tokenizer sidecar exited and did not respawn".to_owned(),
        );
        let not_ready = readiness
            .inner
            .publish(vec![sidecar], vec![respawning], id, &smoke);
        assert!(!not_ready.ready);
        assert_eq!(fields(&not_ready), ["sidecar", "grammar"]);
    }

    #[tokio::test]
    async fn not_ready_before_first_probe_and_posture_reported() {
        let snapshot = readiness("http://127.0.0.1:9").snapshot();
        assert!(!snapshot.ready);
        assert_eq!(fields(&snapshot), ["probe"]);
        assert_eq!(
            snapshot.tenancy,
            TenancyPosture {
                shared_backend_tenancy: SharedBackendTenancy::SharedAcknowledged,
                allow_cross_scope_prefix_cache: false,
                prefix_cache_isolation: PrefixCacheIsolation::None,
                scope_count: 1,
            }
        );
        let json = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(
            json["tenancy"]["shared_backend_tenancy"],
            "shared_acknowledged"
        );
        assert_eq!(json["tenancy"]["prefix_cache_isolation"], "none");
        assert_eq!(json["smoke"]["state"], "pending");
    }

    #[tokio::test]
    async fn drift_names_the_field_then_recovers_with_one_smoke_per_identity() {
        let mock = start(0).await;
        let readiness = readiness(&mock.base_url());

        let ready = readiness.probe().await;
        assert!(ready.ready, "{:?}", ready.problems);
        let instance_id = ready.instance_id.clone().unwrap();
        assert!(instance_id.starts_with("sglang-") && instance_id.len() == 23);
        assert_eq!(
            ready.smoke,
            SmokeStatus::Passed {
                instance_id: instance_id.clone()
            }
        );
        let smoke_rids = mock.dispatched_rids();
        assert_eq!(smoke_rids.len(), 1);
        assert_eq!(Rid::parse(&smoke_rids[0]).unwrap().epoch(), 7);

        // Same identity: no second smoke.
        assert!(readiness.probe().await.ready);
        assert_eq!(mock.dispatched_rids().len(), 1);

        mock.set_server_info_override(Some(ServerInfo {
            speculative_num_steps: Some(4),
            ..ServerInfo::default()
        }));
        let drifted = readiness.probe().await;
        assert!(!drifted.ready);
        assert_eq!(fields(&drifted), ["speculative_num_steps"]);
        assert_eq!(drifted.problems[0].detail, "expected 3, backend reports 4");
        assert_eq!(drifted.instance_id, None);
        assert_eq!(readiness.snapshot(), drifted);

        mock.set_server_info_override(None);
        let recovered = readiness.probe().await;
        assert!(recovered.ready, "{:?}", recovered.problems);
        assert_eq!(recovered.instance_id.as_deref(), Some(instance_id.as_str()));
        assert_eq!(
            mock.dispatched_rids().len(),
            2,
            "identity re-confirmed → smoked again"
        );
    }

    #[tokio::test]
    async fn every_pinned_field_is_compared() {
        let mock = start(0).await;
        let readiness = readiness(&mock.base_url());
        mock.set_server_info_override(Some(ServerInfo {
            version: "0.5.20".to_owned(),
            chat_template: Some("custom.jinja".to_owned()),
            allow_auto_truncate: true,
            context_length: 100_000,
            served_model_name: "other-model".to_owned(),
            ..ServerInfo::default()
        }));
        let snapshot = readiness.probe().await;
        assert!(!snapshot.ready);
        assert_eq!(
            fields(&snapshot),
            [
                "served_model_name",
                "allow_auto_truncate",
                "chat_template",
                "context_length",
                "version"
            ]
        );
        assert!(
            mock.dispatched_rids().is_empty(),
            "no smoke while not matched"
        );
    }

    #[tokio::test]
    async fn absent_nullable_pins_require_null() {
        let mock = start(0).await;
        let mut config = live_config(&mock.base_url());
        let expect = config.deployments[0].expect.as_mut().unwrap();
        expect.quantization = None;
        expect.speculative_algorithm = None;
        expect.speculative_num_steps = None;
        expect.speculative_eagle_topk = None;
        expect.speculative_num_draft_tokens = None;
        let admission = Admission::new(crate::admission::AdmissionLimits {
            max_inflight: 1,
            per_scope_inflight: 1,
            queue_depth: 1,
        });
        admission.set_workers(vec![crate::admission::WorkerCap {
            id: "gpu0".to_owned(),
            cap: 1,
            aging: crate::worker::Aging {
                suspect_after_ms: 15_000,
                lost_after_ms: u64::MAX,
            },
            deployment: 0,
        }]);
        let readiness = Readiness::new(
            SglangHttp::new(&mock.base_url(), TIMEOUTS).unwrap(),
            "gpu0",
            &config,
            0,
            Arc::new(RidGenerator::with_nonce(7, 0xabc)),
            admission,
            Duration::from_secs(5),
        )
        .unwrap();
        let nulls = ServerInfo {
            quantization: None,
            speculative_algorithm: None,
            speculative_num_steps: None,
            speculative_eagle_topk: None,
            speculative_num_draft_tokens: None,
            ..ServerInfo::default()
        };
        mock.set_server_info_override(Some(ServerInfo {
            quantization: Some("fp8".to_owned()),
            ..nulls.clone()
        }));
        let drifted = readiness.probe().await;
        assert!(!drifted.ready);
        assert_eq!(fields(&drifted), ["quantization"]);
        assert_eq!(
            drifted.problems[0].detail,
            "expected null, backend reports \"fp8\""
        );
        mock.set_server_info_override(Some(nulls));
        let ready = readiness.probe().await;
        assert!(ready.ready, "{:?}", ready.problems);
    }

    /// A lost backend reads as unreachable: both endpoints refuse with the connect class of [`BackendError`]
    /// (its Display prefix, not the HTTP client's wording) and no smoke runs.
    fn assert_backend_lost(lost: &ReadinessSnapshot) {
        let connect = crate::sglang::BackendError::Connect(String::new()).to_string();
        assert!(!lost.ready);
        assert_eq!(fields(lost), ["/get_server_info", "/v1/models"]);
        for problem in &lost.problems {
            assert!(problem.detail.starts_with(&connect), "{lost:?}");
        }
        assert_eq!(lost.smoke, SmokeStatus::Pending);
    }

    #[tokio::test]
    async fn connection_loss_is_not_ready_then_recovers() {
        let mock = start(0).await;
        let port = mock.addr().port();
        let readiness = readiness(&mock.base_url());
        assert!(readiness.probe().await.ready);

        mock.shutdown().await.unwrap();
        assert_backend_lost(&readiness.probe().await);

        let mock = start(port).await;
        let back = readiness.probe().await;
        assert!(back.ready, "{:?}", back.problems);
        assert_eq!(mock.dispatched_rids().len(), 1, "returning backend smoked");
    }

    /// A loopback TCP relay to a backend. [`Relay::lose`] forces the timing a loaded host can show the client
    /// of a backend shutdown: the listener is gone, but every connection kept from before reads open until the
    /// client writes its next request on it, which is then dropped unanswered.
    struct Relay {
        addr: std::net::SocketAddr,
        lost: tokio::sync::watch::Sender<bool>,
        accept: tokio::task::JoinHandle<()>,
    }

    impl Relay {
        async fn start(port: u16, backend: std::net::SocketAddr) -> Relay {
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
                .await
                .unwrap();
            let addr = listener.local_addr().unwrap();
            let (lost, mut lost_rx) = tokio::sync::watch::channel(false);
            let conn_lost = lost.subscribe();
            let accept = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        accepted = listener.accept() => {
                            let Ok((client, _)) = accepted else { return };
                            tokio::spawn(Self::pipe(client, backend, conn_lost.clone()));
                        }
                        _ = lost_rx.wait_for(|lost| *lost) => return,
                    }
                }
            });
            Relay { addr, lost, accept }
        }

        fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        /// Closes the listener (returns once it is closed) and loses every relayed connection.
        async fn lose(self) {
            self.lost.send_replace(true);
            self.accept.await.unwrap();
        }

        async fn pipe(
            mut client: tokio::net::TcpStream,
            backend: std::net::SocketAddr,
            mut lost: tokio::sync::watch::Receiver<bool>,
        ) {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let Ok(mut upstream) = tokio::net::TcpStream::connect(backend).await else {
                return;
            };
            let (mut client_rx, mut client_tx) = client.split();
            let (mut upstream_rx, mut upstream_tx) = upstream.split();
            let (mut request, mut response) = ([0u8; 8192], [0u8; 8192]);
            loop {
                let gone = *lost.borrow();
                tokio::select! {
                    read = client_rx.read(&mut request) => match read {
                        // Lost: the request written on the kept connection closes it without an answer.
                        Ok(n) if n > 0 && !*lost.borrow() => {
                            if upstream_tx.write_all(&request[..n]).await.is_err() {
                                return;
                            }
                        }
                        _ => return,
                    },
                    read = upstream_rx.read(&mut response), if !gone => match read {
                        Ok(n) if n > 0 => {
                            if client_tx.write_all(&response[..n]).await.is_err() {
                                return;
                            }
                        }
                        _ => return,
                    },
                    changed = lost.changed(), if !gone => {
                        if changed.is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }

    /// The backend goes away while the probe client still holds a kept-alive connection it has not yet seen
    /// closed (forced by [`Relay::lose`]); the probe still reports the backend unreachable, then recovers.
    #[tokio::test]
    async fn connection_loss_behind_a_kept_connection_is_not_ready_then_recovers() {
        let mock = start(0).await;
        let relay = Relay::start(0, mock.addr()).await;
        let port = relay.addr.port();
        let readiness = readiness(&relay.base_url());
        assert!(readiness.probe().await.ready);

        relay.lose().await;
        mock.shutdown().await.unwrap();
        assert_backend_lost(&readiness.probe().await);

        let mock = start(0).await;
        let _relay = Relay::start(port, mock.addr()).await;
        let back = readiness.probe().await;
        assert!(back.ready, "{:?}", back.problems);
        assert_eq!(mock.dispatched_rids().len(), 1, "returning backend smoked");
    }

    #[tokio::test]
    async fn failed_smoke_is_not_ready_and_retried() {
        let mock = start(0).await;
        mock.set_fault(Fault::ServerError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            remaining: 1,
        });
        let readiness = readiness(&mock.base_url());
        let failed = readiness.probe().await;
        assert!(!failed.ready);
        assert_eq!(fields(&failed), ["smoke"]);
        assert!(matches!(failed.smoke, SmokeStatus::Failed { .. }));
        assert!(readiness.probe().await.ready);
        assert_eq!(mock.dispatched_rids().len(), 2);
    }

    #[tokio::test]
    async fn background_probe_tracks_drift() {
        let mock = start(0).await;
        let readiness = readiness(&mock.base_url()).with_probe_interval(Duration::from_millis(20));
        let task = readiness.spawn();
        let wait_for = |ready: bool| {
            let readiness = readiness.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(5), async {
                    while readiness.snapshot().ready != ready {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                })
                .await
                .expect("readiness transition");
            }
        };
        wait_for(true).await;
        mock.set_server_info_override(Some(ServerInfo {
            kv_cache_dtype: "auto".to_owned(),
            ..ServerInfo::default()
        }));
        wait_for(false).await;
        assert_eq!(fields(&readiness.snapshot()), ["kv_cache_dtype"]);
        mock.set_server_info_override(None);
        wait_for(true).await;
        task.abort();
    }

    /// M2.11 C6 (ledger K, C 56): a library with a snapshot arena reports `CERE_OPS_SNAPSHOT` and
    /// `CERE_OPS_RESTORE`; readiness names them `snapshot`/`restore`, so a dev/live profile pinning them matches.
    #[test]
    fn snapshot_and_restore_operations_are_named() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/deployments/qwen38-flash-next-native.capabilities.json");
        let mut profile: ExecutionCapabilities =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let base = ffi::OPS_PREFILL | ffi::OPS_DECODE | ffi::OPS_CANCEL;
        let operations = |bits: u64| ffi::ExecutionCapabilities {
            operations: bits,
            ..Default::default()
        };
        let flagged = |reported: &ffi::ExecutionCapabilities, profile: &ExecutionCapabilities| {
            capability_problems(reported, profile, 0, 0, 0)
                .into_iter()
                .any(|p| p.field == "capabilities.operations")
        };
        assert!(!flagged(&operations(base), &profile));
        let snapshot = operations(base | ffi::OPS_SNAPSHOT | ffi::OPS_RESTORE);
        assert!(flagged(&snapshot, &profile));
        profile
            .operations
            .extend(["snapshot".to_owned(), "restore".to_owned()]);
        assert!(!flagged(&snapshot, &profile));
    }

    /// M2.6 C6 (D9/D10): an executor reporting one row against the native profile's 16 (4 before m23 design note
    /// P8) is refused, naming the rows fields; the same report with the profile's rows matches.
    #[test]
    fn reported_rows_must_equal_the_profile_rows() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-g0.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let profile = &config.deployments[0].lock.as_ref().unwrap().profile;
        assert_eq!(profile.decode_batch_max, 16);
        let one_row = ffi::ExecutionCapabilities {
            decode_batch_min: 1,
            decode_batch_max: 1,
            max_context_tokens: profile.max_context_tokens,
            execution_mode: ffi::EXECUTION_MODE_NATIVE,
            operations: ffi::OPS_PREFILL | ffi::OPS_DECODE | ffi::OPS_CANCEL,
            kv_pool_tokens: 200_000,
            kv_page_tokens: 64,
            // The shipped profile has sampled rows (M2.7 D11), which need the device sampler, and grammar
            // outputs (M2.8 D10), which need the vocab mask.
            native_sampler: 1,
            native_token_mask: 1,
            ..ffi::ExecutionCapabilities::default()
        };
        let fields: Vec<String> = capability_problems(&one_row, profile, 200_000, 0, 0)
            .into_iter()
            .map(|p| p.field)
            .collect();
        assert_eq!(
            fields,
            [
                "capabilities.decode_batch_max",
                "capabilities.native_decode_batch"
            ]
        );
        let profile_rows = ffi::ExecutionCapabilities {
            decode_batch_max: 16,
            native_decode_batch: 1,
            ..one_row
        };
        assert!(capability_problems(&profile_rows, profile, 200_000, 0, 0).is_empty());
    }

    /// batch-concurrency design note BC2 (fail-first): the 16-row `-v58` (production's `-v53` composition at
    /// `decode_batch_max` 16) is refused at readiness by an executor whose rows differ from the profile's, below
    /// (the 4 rows `-v53` serves) or above (rows the profile never validated); the executor's 16 match.
    #[test]
    fn v58_rows_other_than_the_profiles_sixteen_are_refused() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0-rows16-v58.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let entry = &config.deployments[0];
        let profile = &entry.lock.as_ref().unwrap().profile;
        let n = entry.native.as_ref().unwrap();
        assert_eq!(profile.decode_batch_max, 16);
        let pool = expected_kv_pool_tokens(n, profile.decode_batch_max);
        assert_eq!(Some(pool), n.kv_pool_tokens);
        let rows = |decode_batch_max| ffi::ExecutionCapabilities {
            execution_mode: ffi::EXECUTION_MODE_NATIVE,
            operations: ffi::OPS_PREFILL
                | ffi::OPS_DECODE
                | ffi::OPS_VERIFY
                | ffi::OPS_SNAPSHOT
                | ffi::OPS_RESTORE
                | ffi::OPS_CANCEL,
            decode_batch_min: 1,
            decode_batch_max,
            native_decode_batch: 1,
            max_context_tokens: profile.max_context_tokens,
            kv_pool_tokens: pool,
            kv_page_tokens: 64,
            native_speculative_batch: 1,
            native_state_snapshot: 1,
            native_prefix_cache: 1,
            speculative_draft_tokens: u8::try_from(n.speculative_draft_tokens).unwrap(),
            native_sampler: 1,
            native_token_mask: 1,
            ..ffi::ExecutionCapabilities::default()
        };
        let fields = |reported: u64| -> Vec<String> {
            capability_problems(
                &rows(reported),
                profile,
                pool,
                n.prefix_state_slots,
                n.speculative_draft_tokens,
            )
            .into_iter()
            .map(|p| p.field)
            .collect()
        };
        assert!(fields(16).is_empty(), "{:?}", fields(16));
        for reported in [1, 4, 8, 17, 64] {
            assert_eq!(
                fields(reported),
                ["capabilities.decode_batch_max"],
                "{reported}"
            );
        }
    }

    /// m25 design note D4, C3 and ledger CR1: readiness compares the executor's `kv_pool_tokens` with the
    /// configured non-zero pool (the shipped 200,000 the profile's 16 rows share), else the derived one, the rows
    /// (`decode_batch_max`, the plan's `max_sessions`) x the context rounded up to 64-token pages.
    #[test]
    fn the_kv_pool_is_compared_with_the_configured_or_derived_pool() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-g0.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let entry = &config.deployments[0];
        let profile = &entry.lock.as_ref().unwrap().profile;
        let rows = profile.decode_batch_max;
        assert_eq!(rows, 16);
        let mut n = entry.native.clone().unwrap();
        assert_eq!(expected_kv_pool_tokens(&n, rows), 200_000);
        n.kv_pool_tokens = None;
        assert_eq!(expected_kv_pool_tokens(&n, rows), 3_200_000);
        assert_eq!(expected_kv_pool_tokens(&n, 1), 200_000);
        n.kv_pool_tokens = Some(0);
        n.max_context_tokens = 2_000;
        assert_eq!(expected_kv_pool_tokens(&n, rows), 32_768);
        n.kv_pool_tokens = Some(4_096);
        assert_eq!(expected_kv_pool_tokens(&n, rows), 4_096);

        let reported = |kv_pool_tokens| ffi::ExecutionCapabilities {
            execution_mode: ffi::EXECUTION_MODE_NATIVE,
            operations: ffi::OPS_PREFILL | ffi::OPS_DECODE | ffi::OPS_CANCEL,
            decode_batch_min: 1,
            decode_batch_max: 16,
            native_decode_batch: 1,
            max_context_tokens: 200_000,
            kv_pool_tokens,
            kv_page_tokens: 64,
            native_sampler: 1,
            native_token_mask: 1,
            ..Default::default()
        };
        assert!(capability_problems(&reported(200_000), profile, 200_000, 0, 0).is_empty());
        let problems = capability_problems(&reported(0), profile, 200_000, 0, 0);
        let fields: Vec<&str> = problems.iter().map(|p| p.field.as_str()).collect();
        assert_eq!(fields, ["capabilities.kv_pool_tokens"]);
        assert_eq!(
            problems[0].detail,
            "the deployment pins 200000, the executor reports 0"
        );
    }

    /// m29 design note C6 (D11/D12, I3): the speculation profile pins `native_speculative_batch` true and the
    /// `verify` operation; an executor reporting no speculation (a library or model without the draft) is refused
    /// on both fields and the draft tokens, and the full report matches. The plain profile refuses a library that
    /// speculates, and (16 rows since m23 design note P8, the speculation profile's 4) its rows.
    #[test]
    fn the_speculation_profile_is_refused_by_an_executor_without_the_draft() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-g0-spec.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let entry = &config.deployments[0];
        let profile = &entry.lock.as_ref().unwrap().profile;
        let n = entry.native.as_ref().unwrap();
        assert!(profile.native_speculative_batch);
        assert!(profile.operations.iter().any(|o| o == "verify"));
        assert_eq!(n.speculative_draft_tokens, 4);
        let pool = expected_kv_pool_tokens(n, profile.decode_batch_max);
        let plain = ffi::ExecutionCapabilities {
            execution_mode: ffi::EXECUTION_MODE_NATIVE,
            operations: ffi::OPS_PREFILL | ffi::OPS_DECODE | ffi::OPS_CANCEL,
            decode_batch_min: 1,
            decode_batch_max: 4,
            native_decode_batch: 1,
            max_context_tokens: profile.max_context_tokens,
            kv_pool_tokens: pool,
            kv_page_tokens: 64,
            native_sampler: 1,
            native_token_mask: 1,
            ..Default::default()
        };
        let fields = |reported: &ffi::ExecutionCapabilities, profile, drafts| -> Vec<String> {
            capability_problems(reported, profile, pool, 0, drafts)
                .into_iter()
                .map(|p| p.field)
                .collect()
        };
        assert_eq!(
            fields(&plain, profile, 4),
            [
                "capabilities.operations",
                "capabilities.native_speculative_batch",
                "capabilities.speculative_draft_tokens"
            ]
        );
        let speculating = ffi::ExecutionCapabilities {
            operations: plain.operations | ffi::OPS_VERIFY,
            native_speculative_batch: 1,
            speculative_draft_tokens: 4,
            ..plain
        };
        assert!(fields(&speculating, profile, 4).is_empty());
        let shipped = RuntimeConfig::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-g0.toml"),
        )
        .unwrap();
        let shipped = &shipped.deployments[0].lock.as_ref().unwrap().profile;
        assert_eq!(
            fields(&speculating, shipped, 0),
            [
                "capabilities.operations",
                "capabilities.decode_batch_max",
                "capabilities.native_speculative_batch",
                "capabilities.speculative_draft_tokens"
            ]
        );
    }

    /// m210 design note D6 (ledger C 60): the executor reports `native_prefix_cache` 1 exactly for a model created
    /// with `prefix_state_slots > 0`, so readiness compares the flag with the configured slots.
    #[test]
    fn the_prefix_cache_flag_is_compared_with_the_configured_slots() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/deployments/qwen38-flash-next-native.capabilities.json");
        let profile: ExecutionCapabilities =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let reported = |native_prefix_cache| ffi::ExecutionCapabilities {
            execution_mode: ffi::EXECUTION_MODE_NATIVE,
            operations: ffi::OPS_PREFILL | ffi::OPS_DECODE | ffi::OPS_CANCEL,
            decode_batch_min: 1,
            decode_batch_max: 16,
            native_decode_batch: 1,
            max_context_tokens: 200_000,
            kv_pool_tokens: 200_000,
            kv_page_tokens: 64,
            native_sampler: 1,
            native_token_mask: 1,
            native_prefix_cache,
            ..Default::default()
        };
        let fields = |reported, slots| -> Vec<String> {
            capability_problems(&reported, &profile, 200_000, slots, 0)
                .into_iter()
                .map(|p| p.field)
                .collect()
        };
        assert!(fields(reported(1), 16).is_empty());
        assert!(fields(reported(0), 0).is_empty());
        assert_eq!(
            fields(reported(0), 16),
            ["capabilities.native_prefix_cache"]
        );
        assert_eq!(fields(reported(1), 0), ["capabilities.native_prefix_cache"]);
    }

    /// BC1 leftover: a capabilities file that declares `prefill_signature` refuses an executor reporting another
    /// digest or none; an undeclared one accepts any.
    #[test]
    fn a_declared_prefill_signature_must_equal_the_reported_one() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/deployments/qwen38-flash-next-native.capabilities.json");
        let mut profile: ExecutionCapabilities =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let pinned = "ab".repeat(32);
        let other = "cd".repeat(32);
        profile.prefill_signature = None;
        for reported in [None, Some(pinned.as_str())] {
            assert!(prefill_signature_problem(reported, &profile).is_none());
        }
        profile.prefill_signature = Some(pinned.clone());
        assert!(prefill_signature_problem(Some(&pinned), &profile).is_none());
        for (reported, shown) in [(Some(other.as_str()), other.as_str()), (None, "none")] {
            let p = prefill_signature_problem(reported, &profile).unwrap();
            assert_eq!(p.field, "capabilities.prefill_signature");
            assert_eq!(
                p.detail,
                format!("the deployment pins {pinned}, the executor reports {shown}")
            );
        }
    }

    /// Opportunity O47 (build-loop-ple design note): with `host_share_dir` the executor's device id carries the share
    /// options (the directory and the expected manifest sha256) after the device, whichever device the environment
    /// names; without it the device id is unchanged.
    #[test]
    fn the_device_id_carries_the_host_share_options() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-g0.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let mut n = config.deployments[0].native.clone().unwrap();
        assert_eq!(native_device_id(None, &n, 0), "0");
        assert_eq!(
            native_device_id(Some("toy,sampler=1".into()), &n, 0),
            "toy,sampler=1"
        );
        n.host_share_dir = Some("/dev/shm/cere-host".into());
        let options = format!(
            ",host_share=/dev/shm/cere-host,artifact={}",
            n.expected_manifest_sha256
        );
        assert_eq!(native_device_id(None, &n, 0), format!("0{options}"));
        assert_eq!(
            native_device_id(Some("toy,sampler=1".into()), &n, 0),
            format!("toy,sampler=1{options}")
        );
        n.keep_host_share = true;
        assert_eq!(
            native_device_id(None, &n, 0),
            format!("0{options},host_share_keep=1")
        );
    }

    /// m32n design note D3: a replica's default executor device is its ordinal among the visible cards (the
    /// second replica of `CUDA_VISIBLE_DEVICES="GPU-a,GPU-b"` is device 1); both replicas share one host copy.
    #[test]
    fn a_vulkan_binding_passes_its_ordinal_as_the_device_id() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-g0.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let mut n = config.deployments[0].native.clone().unwrap();
        // A GPU uuid keeps the replica index; `vk:<ordinal>` passes the ordinal, whatever the replica index.
        assert_eq!(native_device_id(None, &n, 0), "0");
        n.device_uuid = Some("vk:0".into());
        assert_eq!(native_device_id(None, &n, 0), "0");
        n.device_uuid = Some("vk:1".into());
        assert_eq!(native_device_id(None, &n, 0), "1");
        n.device_uuid = Some("vk:12".into());
        assert_eq!(native_device_id(None, &n, 0), "12");
        // The device-id environment override still wins; `hip:` keeps the replica index.
        assert_eq!(native_device_id(Some("toy".into()), &n, 0), "toy");
        n.device_uuid = Some("hip:1".into());
        assert_eq!(native_device_id(None, &n, 0), "0");
        // Not a `vk:` ordinal: the replica index.
        n.device_uuid = Some("vk:01".into());
        assert_eq!(native_device_id(None, &n, 0), "0");
        // The share options follow the ordinal.
        n.device_uuid = Some("vk:1".into());
        n.host_share_dir = Some("/dev/shm/cere-host".into());
        assert_eq!(
            native_device_id(None, &n, 0),
            format!(
                "1,host_share=/dev/shm/cere-host,artifact={}",
                n.expected_manifest_sha256
            )
        );
    }

    #[test]
    fn a_replica_runs_on_the_device_of_its_ordinal() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-g0.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let mut n = config.deployments[0].native.clone().unwrap();
        assert_eq!(native_device_id(None, &n, 1), "1");
        assert_eq!(
            native_device_id(Some("toy,sampler=1".into()), &n, 1),
            "toy,sampler=1"
        );
        // The toy's per-replica options: entry `ordinal`, the last for every later replica.
        let both = || Some("toy;toy,fail_forward=3".to_owned());
        assert_eq!(native_device_id(both(), &n, 0), "toy");
        assert_eq!(native_device_id(both(), &n, 1), "toy,fail_forward=3");
        assert_eq!(native_device_id(both(), &n, 2), "toy,fail_forward=3");
        n.host_share_dir = Some("/dev/shm/cere-host".into());
        assert_eq!(
            native_device_id(None, &n, 1),
            format!(
                "1,host_share=/dev/shm/cere-host,artifact={}",
                n.expected_manifest_sha256
            )
        );
    }

    /// m32n design note D10: every replica of the multi-replica form asks its executor to release the card's
    /// primary context when it is destroyed (an operator `unload`), after the share options; the single form never.
    #[test]
    fn a_replica_releases_its_card_on_unload() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/runtime/native-2gpu.toml");
        let config = RuntimeConfig::load(path).unwrap();
        let n = config.deployments[0].native.clone().unwrap();
        let share = format!(
            ",host_share={},artifact={}",
            n.host_share_dir.as_deref().unwrap(),
            n.expected_manifest_sha256
        );
        for ordinal in 0..2 {
            assert_eq!(
                native_device_id(None, &n, ordinal),
                format!("{ordinal}{share},release_device=1")
            );
        }
        assert_eq!(
            native_device_id(Some("toy;toy,fail_forward=3".into()), &n, 1),
            format!("toy,fail_forward=3{share},release_device=1")
        );
        let mut single = n.clone();
        single.replicas = None;
        assert_eq!(native_device_id(None, &single, 0), format!("0{share}"));
    }

    /// m32n design note D6: admission tracks each replica's KV pool only in the multi-replica form (the
    /// configured `kv_pool_tokens` of native-2gpu.toml); the single form leaves the pool untracked.
    #[test]
    fn only_the_replica_form_tracks_a_kv_pool() {
        let load = |name: &str| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/runtime")
                .join(name);
            RuntimeConfig::load(path).unwrap()
        };
        assert_eq!(replica_pool_tokens(&load("native-g0.toml")), None);
        assert_eq!(
            replica_pool_tokens(&load("native-2gpu.toml")),
            Some(200_000)
        );
    }
}
