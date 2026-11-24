//! M3 restart recovery over N replicas (m3 design note Q9, row C7) and the per-worker observability of Q11
//! (row C8): an orphan is aborted on, charged to and load-polled against its own worker; a never-dispatched M3
//! orphan holds nothing; an unconfigured worker's orphan is held globally and aborted by URL; a pre-M3 journal
//! keeps the legacy rule for one replica and broadcasts for two; status, `cere`, `/v1/cere/stats` and the stderr
//! lines carry the worker. In-process `cered` against `cere_mock_sglang`, plus one real `cered` process for the
//! log positions; CI evidence only.

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang};
use cere_runtime::app::{App, Tuning};
use cere_runtime::config::RuntimeConfig;
use cere_runtime::server;
use serde_json::{json, Value};
use support::*;
use tokio::sync::oneshot;

/// Room for every seeded orphan plus the readiness smokes: the global cap may reach N x the per-worker cap (4).
const ROOM: [(&str, &str); 2] = [
    ("max_backend_inflight = 2", "max_backend_inflight = 4"),
    ("per_scope_inflight = 2", "per_scope_inflight = 4"),
];

/// The pre-M3 fixture's two non-terminal requests (epoch 2), scopes `a` and `b`.
const FIXTURE_ORPHANS: [&str; 2] = [
    "cere-00000000000000000002-00000000000000000001-0123456789abcdef",
    "cere-00000000000000000002-00000000000000000002-0123456789abcdef",
];

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Long generations (20 ms per token, 2 000 tokens).
fn slow_mock() -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(20),
        natural_output_tokens: 2_000,
        ..MockConfig::default()
    }
}

async fn slow_mocks(n: usize) -> Vec<MockHandle> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(MockSglang::start(slow_mock()).await.unwrap());
    }
    out
}

/// How a seeded request of the previous incarnation ended up.
enum Seed<'a> {
    /// Its last `dispatched` record names `worker` at `epoch` via `base_url`.
    Dispatched {
        worker: &'a str,
        epoch: u64,
        base_url: &'a str,
    },
    /// Admitted and running under M3, but crashed before its `dispatched` record.
    NotDispatched,
}

/// Journals one non-terminal request per `(scope, seed)` in `config`'s state dir, all admitted at `admitted_ms`,
/// as a crashed M3 `cered` (with `config`'s workers) left them; returns their rids in order.
fn seed(config: &RuntimeConfig, admitted_ms: u64, requests: &[(&str, Seed)]) -> Vec<String> {
    use cere_runtime::digest::{DigestKeys, RequestDigestInput};
    use cere_runtime::ledger::{AcceptOutcome, AcceptRequest, Ledger, LedgerLimits, WorkerTarget};
    let dir = &config.ledger.state_dir;
    let keys = Arc::new(DigestKeys::open(dir).unwrap());
    let local = |base_url: &str| WorkerTarget::Local {
        base_url: base_url.to_owned(),
    };
    let workers: BTreeMap<String, WorkerTarget> = config.deployments[0]
        .replica_list(&config.scheduler.devices)
        .into_iter()
        .map(|r| (r.device, local(&r.base_url)))
        .collect();
    let (mut ledger, recovery) = Ledger::open(
        LedgerLimits::from_config(config),
        keys,
        dir,
        &workers,
        admitted_ms,
    )
    .unwrap();
    let mut rids = Vec::with_capacity(requests.len());
    for (index, (scope, seed)) in requests.iter().enumerate() {
        let rid = format!(
            "cere-{:020}-{:020}-0123456789abcdef",
            recovery.epoch,
            index + 1
        );
        let digest = ledger
            .digest_keys()
            .scope_key(scope)
            .request_digest(&RequestDigestInput {
                request: &chat_body(true),
                session_id: None,
                expected_revision: None,
                served_model: MODEL,
                deadline_ms: 1_000,
                config_hash: "seed",
                surface: None,
            });
        let request = AcceptRequest {
            scope: (*scope).into(),
            rid: rid.clone(),
            key: None,
            digest,
            stream: true,
            surface: None,
            session: None,
        };
        assert_eq!(
            ledger.accept(request, admitted_ms),
            Ok(AcceptOutcome::Accepted)
        );
        ledger.mark_queued(&rid, admitted_ms).unwrap();
        ledger.mark_running(&rid, admitted_ms).unwrap();
        if let Seed::Dispatched {
            worker,
            epoch,
            base_url,
        } = seed
        {
            ledger
                .dispatched(&rid, worker, *epoch, &local(base_url), admitted_ms)
                .unwrap();
        }
        rids.push(rid);
    }
    rids
}

/// An in-process `cered` over `mocks` (device `gpu<i>` = `mocks[i]`, or the single form's `gpu1` for
/// [`Rig::single`]) whose state dir `prepare` fills before the open. It does not wait for readiness: the
/// recovery runs inside `App::open`.
struct Rig {
    /// The live mocks, in configuration order (a dead replica of [`Rig::urls`] has none).
    mocks: Vec<MockHandle>,
    app: Arc<App>,
    base: String,
    http: reqwest::Client,
    state_dir: StateDir,
    /// Dropping it would start the graceful shutdown.
    _stop: oneshot::Sender<()>,
}

impl Rig {
    async fn replicas(
        mocks: Vec<MockHandle>,
        tweaks: &[(String, String)],
        prepare: impl FnOnce(&RuntimeConfig),
    ) -> Self {
        let urls: Vec<String> = mocks.iter().map(MockHandle::base_url).collect();
        Self::urls(&urls, mocks, tweaks, prepare).await
    }

    /// Device `gpu<i>` at `urls[i]`, whether or not a mock of `mocks` serves it.
    async fn urls(
        urls: &[String],
        mocks: Vec<MockHandle>,
        tweaks: &[(String, String)],
        prepare: impl FnOnce(&RuntimeConfig),
    ) -> Self {
        let state_dir = StateDir::new();
        let tweaks: Vec<(&str, &str)> = ROOM
            .into_iter()
            .chain(tweaks.iter().map(|(a, b)| (a.as_str(), b.as_str())))
            .collect();
        let text = replicas_config_text(urls, state_dir.path(), 18390, &tweaks);
        Self::open(mocks, state_dir, &text, prepare).await
    }

    async fn single(mock: MockHandle, prepare: impl FnOnce(&RuntimeConfig)) -> Self {
        let state_dir = StateDir::new();
        let text = config_text(&mock.base_url(), state_dir.path(), 18390, &ROOM);
        Self::open(vec![mock], state_dir, &text, prepare).await
    }

    async fn open(
        mocks: Vec<MockHandle>,
        state_dir: StateDir,
        text: &str,
        prepare: impl FnOnce(&RuntimeConfig),
    ) -> Self {
        let config = RuntimeConfig::from_toml_str(text, &repo_root()).unwrap();
        prepare(&config);
        let credentials = config
            .resolve_auth_with(|name| match name {
                "TOK_A" => Some(TOKEN_A.into()),
                "TOK_B" => Some(TOKEN_B.into()),
                _ => None,
            })
            .unwrap();
        let tuning: Tuning = test_tuning();
        let app = App::open(config, &credentials, None, tuning).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = oneshot::channel::<()>();
        tokio::spawn(server::serve(app.clone(), listener, async move {
            let _ = stopped.await;
        }));
        Self {
            mocks,
            app,
            base,
            http: http(),
            state_dir,
            _stop: stop,
        }
    }

    /// `/v1/cere/stats` of the scope of `token`.
    async fn stats(&self, token: &str) -> Value {
        let r = self
            .http
            .get(format!("{}/v1/cere/stats", self.base))
            .bearer_auth(token)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        r.json().await.unwrap()
    }

    /// Slots charged to each worker, summed over scopes `a` and `b`, from `/v1/cere/stats`.
    async fn worker_charges(&self) -> BTreeMap<String, u64> {
        let mut out = BTreeMap::new();
        for token in [TOKEN_A, TOKEN_B] {
            let stats = self.stats(token).await;
            let workers = stats["workers"]
                .as_object()
                .unwrap_or_else(|| panic!("stats without workers: {stats}"));
            for (id, w) in workers {
                *out.entry(id.clone()).or_insert(0) += w["inflight"].as_u64().unwrap();
            }
        }
        out
    }

    /// Waits (bounded) until admission charges exactly `inflight` slots: the readiness smokes (charged too)
    /// have ended and only the held orphans remain.
    async fn settle_inflight(&self, inflight: u64) {
        let until = tokio::time::Instant::now() + Duration::from_secs(30);
        while self.app.admission_snapshot().inflight != inflight {
            assert!(
                tokio::time::Instant::now() < until,
                "admission never settled at {inflight}: {:?}",
                self.app.admission_snapshot()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn status(&self, rid: &str, token: &str) -> Value {
        let r = self
            .http
            .get(format!("{}/v1/cere/requests/{rid}", self.base))
            .bearer_auth(token)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        r.json().await.unwrap()
    }

    fn aborted(&self, index: usize) -> Vec<String> {
        let mut rids: Vec<String> = self.mocks[index]
            .abort_requests()
            .into_iter()
            .map(|a| a.rid)
            .collect();
        rids.sort();
        rids
    }

    /// Waits (bounded) until mock `index` saw `/abort_request` for each of `rids`.
    async fn wait_aborted(&self, index: usize, rids: &[&str]) {
        for _ in 0..400 {
            let seen = self.aborted(index);
            if rids.iter().all(|rid| seen.iter().any(|s| s == rid)) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!(
            "mock {index} never saw aborts of {rids:?}: {:?}",
            self.aborted(index)
        );
    }

    fn released(&self) -> Vec<(String, String)> {
        journal(self.state_dir.path())
            .iter()
            .filter(|r| r["type"] == "orphan_released")
            .map(|r| {
                (
                    r["rid"].as_str().unwrap().to_owned(),
                    r["cause"].as_str().unwrap().to_owned(),
                )
            })
            .collect()
    }
}

fn charges(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
    pairs.iter().map(|(id, n)| ((*id).to_owned(), *n)).collect()
}

/// Q9: an orphan dispatched to `gpu1` is aborted on `gpu1`'s mock only, held against `gpu1`'s cap (and the global
/// one), and its status after the replay still names the worker and epoch it was dispatched to.
#[tokio::test(flavor = "multi_thread")]
async fn c7_orphan_is_aborted_on_its_own_worker_only_and_held_against_its_cap() {
    let mocks = slow_mocks(2).await;
    let b_url = mocks[1].base_url();
    let mut rids = Vec::new();
    let rig = Rig::replicas(mocks, &[], |config| {
        rids = seed(
            config,
            unix_ms(),
            &[(
                "a",
                Seed::Dispatched {
                    worker: "gpu1",
                    epoch: 1,
                    base_url: &b_url,
                },
            )],
        );
    })
    .await;
    let rid = rids[0].as_str();
    rig.wait_aborted(1, &[rid]).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(rig.aborted(0), Vec::<String>::new(), "aborted on gpu0");
    assert_eq!(rig.aborted(1), [rid]);
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu0", 0), ("gpu1", 1)])
    );
    rig.settle_inflight(1).await;
    assert!(rig.released().is_empty(), "released inside the hour");
    let status = rig.status(rid, TOKEN_A).await;
    assert_eq!(
        (
            &status["state"],
            &status["reason"],
            &status["worker_id"],
            &status["worker_epoch"]
        ),
        (
            &json!("failed"),
            &json!("worker_restart"),
            &json!("gpu1"),
            &json!(1)
        )
    );
}

/// Q9 (m1close Q6 per worker): each worker's orphans are released by that worker's own load. `gpu1`'s orphan
/// still generates on its mock (`L_gpu1 = 1 > C_gpu1 = 0`) and stays held while `gpu0`'s idle mock (`L = 0 = C`)
/// releases `gpu0`'s orphan by load; once `gpu1`'s generation ends, its orphan is released by load too.
#[tokio::test(flavor = "multi_thread")]
async fn c7_load_release_is_per_worker() {
    let mocks = slow_mocks(2).await;
    let urls: Vec<String> = mocks.iter().map(MockHandle::base_url).collect();
    let mut tweaks: Vec<(String, String)> = urls
        .iter()
        .map(|url| {
            (
                format!("base_url = \"{url}\"\n"),
                format!("base_url = \"{url}\"\nload_endpoint = \"/get_load\"\n"),
            )
        })
        .collect();
    tweaks.push((
        "terminal_retention_ms = 3600000".into(),
        "terminal_retention_ms = 3600000\norphan_load_poll_ms = 1000".into(),
    ));
    let mut rids = Vec::new();
    let rig = Rig::replicas(mocks, &tweaks, |config| {
        rids = seed(
            config,
            unix_ms(),
            &[
                (
                    "a",
                    Seed::Dispatched {
                        worker: "gpu0",
                        epoch: 1,
                        base_url: &urls[0],
                    },
                ),
                (
                    "a",
                    Seed::Dispatched {
                        worker: "gpu1",
                        epoch: 1,
                        base_url: &urls[1],
                    },
                ),
            ],
        );
    })
    .await;
    let (on_a, on_b) = (rids[0].as_str(), rids[1].as_str());
    rig.wait_aborted(1, &[on_b]).await;
    // `gpu1`'s orphan keeps generating after its abort (the abort reached the backend before the rid did).
    let orphan = rig
        .http
        .post(format!("{}/v1/chat/completions", urls[1]))
        .json(&json!({
            "rid": on_b,
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there"}],
            "max_completion_tokens": 2_000,
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(orphan.status(), 200);
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    while rig.released().is_empty() {
        assert!(tokio::time::Instant::now() < until, "nothing released");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Two more polls of `gpu1` while its orphan still runs.
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    assert_eq!(rig.mocks[1].load(), 1, "gpu1's orphan stopped generating");
    assert_eq!(rig.released(), [(on_a.to_owned(), "load".to_owned())]);
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu0", 0), ("gpu1", 1)])
    );
    drop(orphan);
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    while rig.released().len() < 2 {
        assert!(
            tokio::time::Instant::now() < until,
            "gpu1's orphan never released"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        rig.released(),
        [
            (on_a.to_owned(), "load".to_owned()),
            (on_b.to_owned(), "load".to_owned())
        ]
    );
    // The journal record precedes the permit drop (`OrphanHold::run`): wait for the slot, then read the charges.
    rig.settle_inflight(0).await;
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu0", 0), ("gpu1", 0)])
    );
}

/// Q9: an M3 admission without a `dispatched` record never reached a backend: terminal `worker_restart` with no
/// worker, `orphan_released` cause `not_dispatched` at once, no abort anywhere and no slot held.
#[tokio::test(flavor = "multi_thread")]
async fn c7_m3_orphan_without_dispatched_holds_nothing() {
    let mut rids = Vec::new();
    let rig = Rig::replicas(slow_mocks(2).await, &[], |config| {
        rids = seed(config, unix_ms(), &[("a", Seed::NotDispatched)]);
    })
    .await;
    let rid = rids[0].as_str();
    assert_eq!(
        rig.released(),
        [(rid.to_owned(), "not_dispatched".to_owned())]
    );
    rig.settle_inflight(0).await;
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu0", 0), ("gpu1", 0)])
    );
    let status = rig.status(rid, TOKEN_A).await;
    assert_eq!(
        (
            &status["state"],
            &status["reason"],
            &status["worker_id"],
            &status["worker_epoch"]
        ),
        (
            &json!("failed"),
            &json!("worker_restart"),
            &Value::Null,
            &Value::Null
        )
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(rig.aborted(0).is_empty() && rig.aborted(1).is_empty());
}

/// Q9 [REVISED: build M3 C7]: orphans of a worker that is not configured with its journaled `base_url` are
/// charged to no worker (global cap only) and aborted only on replicas whose normalised `base_url` equals the
/// journaled one: an unknown id at `gpu1`'s URL and `gpu0` whose URL moved to `gpu0`'s current one from `gpu1`'s
/// go to `gpu1`'s mock; `gpu1` journaled at a URL nobody serves is aborted nowhere.
#[tokio::test(flavor = "multi_thread")]
async fn c7_unconfigured_worker_orphans_are_held_globally_and_aborted_by_url() {
    let mocks = slow_mocks(2).await;
    let b_url = mocks[1].base_url();
    let gone = format!("http://127.0.0.1:{}", free_port());
    let mut rids = Vec::new();
    let rig = Rig::replicas(mocks, &[], |config| {
        rids = seed(
            config,
            unix_ms(),
            &[
                (
                    "a",
                    Seed::Dispatched {
                        worker: "gpu7",
                        epoch: 3,
                        base_url: &b_url,
                    },
                ),
                (
                    "a",
                    Seed::Dispatched {
                        worker: "gpu0",
                        epoch: 1,
                        base_url: &b_url,
                    },
                ),
                (
                    "b",
                    Seed::Dispatched {
                        worker: "gpu1",
                        epoch: 1,
                        base_url: &gone,
                    },
                ),
            ],
        );
    })
    .await;
    let (unknown, moved, nowhere) = (rids[0].as_str(), rids[1].as_str(), rids[2].as_str());
    rig.wait_aborted(1, &[unknown, moved]).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(rig.aborted(0), Vec::<String>::new(), "aborted on gpu0");
    assert_eq!(rig.aborted(1), [unknown, moved]);
    assert!(!rig.aborted(1).iter().any(|r| r == nowhere));
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu0", 0), ("gpu1", 0)]),
        "charged to a worker"
    );
    rig.settle_inflight(3).await;
    assert!(rig.released().is_empty());
    let status = rig.status(unknown, TOKEN_A).await;
    assert_eq!(
        (&status["worker_id"], &status["worker_epoch"]),
        (&json!("gpu7"), &json!(3))
    );
}

/// The committed pre-M3 journal (bytes of `main` 97222b4), its two orphans re-dated to now so the hour of
/// retention holds them (the records keep their pre-M3 shape: no `dispatch_journaled`, no `dispatched`).
fn pre_m3_journal(config: &RuntimeConfig) {
    let now = unix_ms();
    let text = std::fs::read_to_string(
        repo_root().join("crates/cere-runtime/tests/fixtures/journal_pre_m3.jsonl"),
    )
    .unwrap();
    for at in ["\"at_ms\":2001}", "\"at_ms\":2003}"] {
        assert_eq!(text.matches(at).count(), 1, "fixture changed: {at}");
    }
    let text = text
        .replace("\"at_ms\":2001}", &format!("\"at_ms\":{now}}}"))
        .replace("\"at_ms\":2003}", &format!("\"at_ms\":{}}}", now + 2));
    let dir = &config.ledger.state_dir;
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("journal.jsonl"), text).unwrap();
}

/// Q9 legacy rule, one replica: both pre-M3 orphans are aborted on it and charged to it, as before M3.
#[tokio::test(flavor = "multi_thread")]
async fn c7_pre_m3_journal_with_one_replica_keeps_the_legacy_rule() {
    let mock = MockSglang::start(slow_mock()).await.unwrap();
    let rig = Rig::single(mock, pre_m3_journal).await;
    rig.wait_aborted(0, &FIXTURE_ORPHANS).await;
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu1", 2)]),
        "not charged to the only worker"
    );
    rig.settle_inflight(2).await;
    for (rid, token) in FIXTURE_ORPHANS.into_iter().zip([TOKEN_A, TOKEN_B]) {
        let status = rig.status(rid, token).await;
        assert_eq!(
            (&status["state"], &status["worker_id"]),
            (&json!("failed"), &Value::Null),
            "{rid}"
        );
    }
    assert!(rig.released().is_empty());
}

/// Q9 legacy rule, two replicas: the pre-M3 orphans' worker is unknown, so each is aborted on every replica and
/// held against the global cap only.
#[tokio::test(flavor = "multi_thread")]
async fn c7_pre_m3_journal_with_two_replicas_broadcasts_and_holds_globally() {
    let rig = Rig::replicas(slow_mocks(2).await, &[], pre_m3_journal).await;
    rig.wait_aborted(0, &FIXTURE_ORPHANS).await;
    rig.wait_aborted(1, &FIXTURE_ORPHANS).await;
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu0", 0), ("gpu1", 0)]),
        "charged to a worker"
    );
    rig.settle_inflight(2).await;
    assert!(rig.released().is_empty());
}

/// Review J1: a legacy orphan's aborts go to every replica at once, so a dead `gpu0` (nothing listens on its
/// URL) never keeps the abort from reaching the live `gpu1`.
#[tokio::test(flavor = "multi_thread")]
async fn j1_a_dead_replica_never_blocks_the_abort_on_another() {
    let live = MockSglang::start(slow_mock()).await.unwrap();
    let urls = [format!("http://127.0.0.1:{}", free_port()), live.base_url()];
    let rig = Rig::urls(&urls, vec![live], &[], pre_m3_journal).await;
    rig.wait_aborted(0, &FIXTURE_ORPHANS).await;
    rig.settle_inflight(2).await;
    assert!(rig.released().is_empty());
}

/// Review J1: an orphan's slot is released at its retention bound (`admitted + terminal_retention_ms`) even
/// though its worker never answers the abort; the worker's charge and the global one go back to 0.
///
/// O368: `admitted` is seeded before the rig starts, so the bound covers the rig's start: 8 s (was 2 s, which a rig
/// start at load 19 overran, CI `ec426ff1`), and the in-bound charge read is checked to have ended inside it.
#[tokio::test(flavor = "multi_thread")]
async fn j1_the_slot_is_released_at_the_bound_while_the_target_stays_dead() {
    const RETENTION_MS: u64 = 8_000;
    let live = MockSglang::start(slow_mock()).await.unwrap();
    let dead = format!("http://127.0.0.1:{}", free_port());
    let urls = [dead.clone(), live.base_url()];
    let retention = [(
        "terminal_retention_ms = 3600000".to_owned(),
        format!("terminal_retention_ms = {RETENTION_MS}"),
    )];
    let admitted = unix_ms();
    let mut rids = Vec::new();
    let rig = Rig::urls(&urls, vec![live], &retention, |config| {
        rids = seed(
            config,
            admitted,
            &[(
                "a",
                Seed::Dispatched {
                    worker: "gpu0",
                    epoch: 1,
                    base_url: &dead,
                },
            )],
        );
    })
    .await;
    let rid = rids[0].as_str();
    let inside = rig.worker_charges().await;
    let read_ms = unix_ms() - admitted;
    assert!(
        read_ms < RETENTION_MS,
        "the rig start and charge read took {read_ms} ms, past the {RETENTION_MS} ms bound"
    );
    assert_eq!(
        inside,
        charges(&[("gpu0", 1), ("gpu1", 0)]),
        "not held against gpu0 inside the bound"
    );
    rig.settle_inflight(0).await;
    let released = journal(rig.state_dir.path())
        .into_iter()
        .find(|r| r["type"] == "orphan_released")
        .expect("no orphan_released record");
    assert_eq!(
        (&released["rid"], &released["cause"]),
        (&json!(rid), &json!("retention"))
    );
    assert!(
        released["at_ms"].as_u64().unwrap() >= admitted + RETENTION_MS,
        "released before the bound: {released}"
    );
    assert_eq!(
        rig.worker_charges().await,
        charges(&[("gpu0", 0), ("gpu1", 0)])
    );
}

/// The `/v1/cere/stats` `workers` object for the caller's scope (Q11): every configured worker, exactly
/// `{state, epoch, cap, inflight}`.
fn stats_workers(stats: &Value) -> &serde_json::Map<String, Value> {
    let workers = stats["workers"]
        .as_object()
        .unwrap_or_else(|| panic!("stats without workers: {stats}"));
    for (id, w) in workers {
        let mut keys: Vec<&str> = w.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["cap", "epoch", "inflight", "state"], "{id}: {w}");
    }
    workers
}

/// Q11 / C8 on 2 workers: `/v1/cere/stats` reports each worker's state, epoch, cap and the caller scope's
/// in-flight on it; status and the `cere` object carry `worker_id` + `worker_epoch`, null for `queue_full`, set
/// for a completed request and still set after a restart replays the journal.
#[tokio::test(flavor = "multi_thread")]
async fn c8_stats_status_and_cere_carry_the_worker() {
    let mocks = slow_mocks(2).await;
    let h = Harness::start_replicas(
        mocks,
        &[
            ("max_backend_inflight = 2", "max_backend_inflight = 1"),
            ("per_scope_inflight = 2", "per_scope_inflight = 1"),
            ("queue_depth = 64", "queue_depth = 1"),
        ],
    )
    .await;
    let (_, ready) = readyz(&h).await;
    let unary = |tokens: u32| {
        let mut body = chat_body(false);
        body["max_completion_tokens"] = tokens.into();
        let request = h.chat(&body);
        tokio::spawn(async move {
            let r = request.send().await.unwrap();
            let status = r.status().as_u16();
            let rid = r.headers()["x-cere-request-id"]
                .to_str()
                .unwrap()
                .to_owned();
            (status, rid, r.json::<Value>().await.unwrap())
        })
    };
    let first = unary(40);
    let until = tokio::time::Instant::now() + Duration::from_secs(5);
    while h.app.admission_snapshot().inflight != 1 {
        assert!(tokio::time::Instant::now() < until, "never admitted");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let stats_a: Value = h
        .get("/v1/cere/stats", TOKEN_A)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let stats_b: Value = h
        .get("/v1/cere/stats", TOKEN_B)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let (a, b) = (stats_workers(&stats_a), stats_workers(&stats_b));
    assert_eq!(a.keys().collect::<Vec<_>>(), ["gpu0", "gpu1"]);
    for (id, w) in a {
        assert_eq!(w["state"], "ready", "{id}");
        assert_eq!(w["epoch"], ready["workers"][id]["epoch"], "{id}");
        assert_eq!(w["cap"], 4, "{id}");
    }
    let running: Vec<&String> = a
        .iter()
        .filter(|(_, w)| w["inflight"] == 1)
        .map(|(id, _)| id)
        .collect();
    assert_eq!(running.len(), 1, "{stats_a}");
    assert_eq!(
        a.values()
            .map(|w| w["inflight"].as_u64().unwrap())
            .sum::<u64>(),
        1
    );
    assert!(
        b.values().all(|w| w["inflight"] == 0),
        "scope b sees scope a's in-flight: {stats_b}"
    );
    let second = unary(1);
    while h.app.admission_snapshot().queued != 1 {
        assert!(tokio::time::Instant::now() < until, "never queued");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let (code, full_rid, full) = unary(1).await.unwrap();
    assert_eq!(code, 429, "{full}");
    assert_eq!(
        (&full["cere"]["worker_id"], &full["cere"]["worker_epoch"]),
        (&Value::Null, &Value::Null),
        "{full}"
    );
    let status = h.terminal(&full_rid).await;
    assert_eq!(status["reason"], "queue_full");
    assert_eq!(
        (&status["worker_id"], &status["worker_epoch"]),
        (&Value::Null, &Value::Null)
    );
    let (code, rid, done) = first.await.unwrap();
    assert_eq!(code, 200, "{done}");
    let worker = running[0].as_str();
    let epoch = ready["workers"][worker]["epoch"].clone();
    assert_eq!(
        (&done["cere"]["worker_id"], &done["cere"]["worker_epoch"]),
        (&json!(worker), &epoch),
        "{done}"
    );
    let status = h.terminal(&rid).await;
    assert_eq!(
        (&status["worker_id"], &status["worker_epoch"]),
        (&json!(worker), &epoch)
    );
    assert_eq!(second.await.unwrap().0, 200);
    let h = h.restart().await;
    let (_, replayed) = h.status(&rid, TOKEN_A).await;
    assert_eq!(
        (&replayed["worker_id"], &replayed["worker_epoch"]),
        (&json!(worker), &epoch),
        "after replay"
    );
    let (_, ready) = readyz(&h).await;
    assert!(
        ready["workers"][worker]["epoch"].as_u64() > epoch.as_u64(),
        "the restart registered a higher epoch: {ready}"
    );
}

/// The stderr line of `stderr` starting with `prefix`.
fn line<'a>(stderr: &'a str, prefix: &str) -> &'a str {
    stderr
        .lines()
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("no line starting {prefix:?}:\n{stderr}"))
}

/// Q11 / C8 on the real binary: `worker=<id>@<epoch>` (or `-` before dispatch) sits right after `scope=` in
/// `cered request` lines and right after `rid=` in `cered recovery` lines, the positions `smoke_live.py`'s
/// `request_line` / `recovery_abort_line` parse (`WORKER_FIELD`). Replicas: `gpu0`, `gpu1` (mocks) and `gpu2`
/// (nothing listens, so its orphan's abort keeps failing).
#[tokio::test(flavor = "multi_thread")]
async fn c8_log_lines_carry_the_worker_at_the_q11_positions() {
    let mocks = slow_mocks(2).await;
    let mut urls: Vec<String> = mocks.iter().map(MockHandle::base_url).collect();
    urls.push(format!("http://127.0.0.1:{}", free_port()));
    let state = StateDir::new();
    let text = replicas_config_text(&urls, state.path(), free_port(), &ROOM);
    let parsed = RuntimeConfig::from_toml_str(&text, &repo_root()).unwrap();
    let rids = seed(
        &parsed,
        unix_ms(),
        &[
            (
                "a",
                Seed::Dispatched {
                    worker: "gpu1",
                    epoch: 1,
                    base_url: &urls[1],
                },
            ),
            (
                "a",
                Seed::Dispatched {
                    worker: "gpu7",
                    epoch: 4,
                    base_url: &urls[1],
                },
            ),
            ("b", Seed::NotDispatched),
            (
                "a",
                Seed::Dispatched {
                    worker: "gpu2",
                    epoch: 1,
                    base_url: &urls[2],
                },
            ),
        ],
    );
    let (known, unconfigured, undispatched, failing) = (&rids[0], &rids[1], &rids[2], &rids[3]);
    let config = ConfigFile::new(&text);
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    cered.wait_ready().await;
    let r = http()
        .post(cered.url("/v1/chat/completions"))
        .bearer_auth(TOKEN_B)
        .json(&chat_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let body: Value = r.json().await.unwrap();
    let worker = format!(
        "{}@{}",
        body["cere"]["worker_id"].as_str().unwrap(),
        body["cere"]["worker_epoch"]
    );
    for _ in 0..250 {
        if mocks[1].abort_requests().len() >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    cered.signal("TERM");
    let (_, stderr) = cered.exit(Duration::from_secs(20)).await;
    let terminal = line(&stderr, &format!("cered request rid={rid} "));
    assert!(
        terminal.starts_with(&format!(
            "cered request rid={rid} scope=b worker={worker} state=completed reason=- stream=false "
        )),
        "{terminal}"
    );
    for (orphan, scope, field) in [
        (known, "a", "gpu1@1"),
        (unconfigured, "a", "gpu7@4"),
        (undispatched, "b", "-"),
        (failing, "a", "gpu2@1"),
    ] {
        line(
            &stderr,
            &format!(
                "cered request rid={orphan} scope={scope} worker={field} epoch=1 state=failed reason=worker_restart"
            ),
        );
    }
    line(
        &stderr,
        &format!("cered recovery: aborted rid={known} worker=gpu1@1; slot held until unix_ms="),
    );
    // M5 D5: a worker outside every deployment and a never-dispatched request have no model (`model=-`).
    let unconfigured_line = line(
        &stderr,
        &format!("cered recovery: rid={unconfigured} worker=gpu7@4 unconfigured"),
    );
    assert!(
        unconfigured_line.ends_with(" model=-"),
        "{unconfigured_line}"
    );
    line(
        &stderr,
        &format!(
            "cered recovery: aborted rid={unconfigured} worker=gpu7@4; slot held until unix_ms="
        ),
    );
    let undispatched_line = line(
        &stderr,
        &format!("cered recovery: released rid={undispatched} worker=- cause=not_dispatched"),
    );
    assert!(
        undispatched_line.ends_with(" model=-"),
        "{undispatched_line}"
    );
    line(
        &stderr,
        &format!("cered recovery: abort failed rid={failing} worker=gpu2@1 ("),
    );
    assert!(
        !stderr.contains(&format!("aborted rid={failing}")),
        "{stderr}"
    );
}
