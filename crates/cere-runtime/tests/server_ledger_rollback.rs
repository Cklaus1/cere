//! OH-03 rollback (ops-hardening design note, `ledger.evict_terminal_when_full`) end to end on the real `cered`
//! and the CPU toy executor (the ledger does not depend on the device): a full ledger of eight records, keyed and
//! unkeyed terminal ones plus one held in flight, evicts; the candidate restarts from a crash; then, when
//! `CERE_ROLLBACK_CERED` names an earlier release's `cered` (one before OH-03, e.g. the installed
//! `/opt/cere/<sha>/bin/cered`), that binary opens copies of the journal the candidate left, once from the crash
//! and once from the restarted candidate's graceful stop, with the rollback config (the same profile without
//! `evict_terminal_when_full`). Without the variable the rollback half is skipped.

mod native_server;
mod native_support;
mod support;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use native_server::{Case, Native, OPERATOR_TOKEN};
use serde_json::{json, Value};
use support::{http, Event, SseReader, StateDir, MODEL};

/// The earlier release's `cered` the rollback half runs.
const ROLLBACK_ENV: &str = "CERE_ROLLBACK_CERED";
const MAX_RECORDS: u64 = 8;
/// Long enough for the whole scenario before the first retained record expires, which the rollback half waits for.
const RETENTION_MS: u64 = 60_000;
/// The in-flight request: the toy holds its session after its first token (once) until the process dies.
const HELD: &str = "toy,hold_after=1,hold_ms=600000,hold_count=1";

fn candidate() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_cered"))
}

fn body(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "cere-native-sentinel-prompt-rollback-1"}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": stream,
    })
}

fn rid(response: &reqwest::Response) -> String {
    response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned()
}

/// A unary request (with `key` as its `Idempotency-Key`): status, request id header (absent on a refusal), body.
async fn unary(native: &Native, key: Option<&str>) -> (u16, Option<String>, Value) {
    let mut request = native.chat(&body(false));
    if let Some(key) = key {
        request = request.header("Idempotency-Key", key);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    let id = response
        .headers()
        .get("x-cere-request-id")
        .map(|v| v.to_str().unwrap().to_owned());
    (status, id, response.json().await.unwrap())
}

/// A unary request that is admitted and completes; its request id.
async fn completes(native: &Native, key: Option<&str>) -> String {
    let (status, id, body) = unary(native, key).await;
    assert_eq!(status, 200, "{body}");
    let id = id.unwrap();
    assert_eq!(native.terminal(&id).await["state"], "completed");
    id
}

/// The `error.code` of a refused unary request.
async fn refused(native: &Native, key: Option<&str>) -> (u16, String) {
    let (status, _, body) = unary(native, key).await;
    assert_ne!(status, 200, "{body}");
    (status, body["error"]["code"].as_str().unwrap().to_owned())
}

/// `GET /v1/cere/requests/{rid}`: HTTP status and state (`None` when unknown).
async fn state(native: &Native, rid: &str) -> (u16, Option<String>) {
    let (status, body) = native.get(&format!("/v1/cere/requests/{rid}")).await;
    (status, body["state"].as_str().map(str::to_owned))
}

/// OH-05 ledger occupancy of the candidate: (live, terminal, evictions).
async fn ledger(native: &Native) -> (u64, u64, u64) {
    let response = http()
        .get(native.cered.url("/metrics"))
        .bearer_auth(OPERATOR_TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let text = response.text().await.unwrap();
    let sample = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
            .unwrap_or_else(|| panic!("no {name} in {text}"))
    };
    assert_eq!(sample("cere_ledger_capacity"), MAX_RECORDS);
    (
        sample("cere_ledger_records{state=\"live\"}"),
        sample("cere_ledger_records{state=\"terminal\"}"),
        sample("cere_ledger_evictions_total"),
    )
}

/// A copy of the ledger state (journal and digest secret, not the lock), as a rollback operator would take it.
fn copy_state(from: &Path) -> StateDir {
    use std::os::unix::fs::DirBuilderExt;
    let to = StateDir::new();
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(to.path())
        .unwrap();
    for file in ["journal.jsonl", "server_secret"] {
        std::fs::copy(from.join(file), to.path().join(file)).unwrap();
    }
    to
}

/// The rollback config: the candidate's without `ledger.evict_terminal_when_full` (unknown to an earlier
/// release), on `state`, on a fresh port.
fn rollback_config(native: &Native, state: &StateDir, name: &str) -> PathBuf {
    let text = std::fs::read_to_string(&native.config_path).unwrap();
    let mut config: toml::Table = toml::from_str(&text).unwrap();
    let ledger = config["ledger"].as_table_mut().unwrap();
    assert!(ledger.remove("evict_terminal_when_full").is_some());
    ledger.insert(
        "state_dir".into(),
        state.path().display().to_string().into(),
    );
    config["server"]["bind"] = format!("127.0.0.1:{}", support::free_port()).into();
    let path = native.config_path.with_file_name(name);
    std::fs::write(&path, toml::to_string(&config).unwrap()).unwrap();
    path
}

/// The keyed requests' ids and the rest of the scenario's.
struct Ids {
    held: String,
    k1: String,
    k3: String,
    unkeyed: Vec<String>,
}

#[tokio::test]
async fn oh03_rollback_of_an_evicting_ledger() {
    let rollback = std::env::var_os(ROLLBACK_ENV).map(PathBuf::from);
    let mut native = Native::ready(Case {
        device_id: HELD.to_owned(),
        ledger: vec![
            ("max_records", (MAX_RECORDS as i64).into()),
            ("terminal_retention_ms", (RETENTION_MS as i64).into()),
            ("evict_terminal_when_full", true.into()),
        ],
        ..Case::matching()
    })
    .await;

    // Fill: one stream held in flight, then seven terminal records (k1, k2 keyed; u1..u5 unkeyed).
    let response = native.chat(&body(true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let held = rid(&response);
    let mut reader = SseReader::new(response);
    loop {
        match reader.next().await {
            Some(Event::Data(v)) if v.pointer("/choices/0/delta/content").is_some() => break,
            Some(Event::Data(v)) => assert!(v.get("error").is_none(), "{v}"),
            other => panic!("the held stream ended: {other:?}"),
        }
    }
    let k1 = completes(&native, Some("k1")).await;
    completes(&native, Some("k2")).await;
    let mut unkeyed = Vec::new();
    for _ in 0..5 {
        unkeyed.push(completes(&native, None).await);
    }
    assert_eq!(ledger(&native).await, (1, 7, 0));
    // Full: four more admissions evict the four oldest terminal records (k1, k2, u1, u2), never the held one.
    unkeyed.push(completes(&native, None).await);
    unkeyed.push(completes(&native, None).await);
    let (status, k3, k3_body) = unary(&native, Some("k3")).await;
    assert_eq!(status, 200, "{k3_body}");
    let k3 = k3.unwrap();
    assert_eq!(native.terminal(&k3).await["state"], "completed");
    unkeyed.push(completes(&native, None).await);
    assert_eq!(ledger(&native).await, (1, 7, 4));
    assert_eq!(state(&native, &held).await, (200, Some("running".into())));
    assert_eq!(state(&native, &k1).await, (404, None));
    assert_eq!(state(&native, &unkeyed[0]).await, (404, None));
    assert_eq!(state(&native, &unkeyed[1]).await, (404, None));
    assert_eq!(state(&native, &unkeyed[2]).await.0, 200);
    // The evicted key is expired, the retained one replays its stored body (no new admission: still 1 + 7).
    assert_eq!(
        refused(&native, Some("k1")).await,
        (409, "idempotency_key_expired".into())
    );
    let (status, replay_id, replay) = unary(&native, Some("k3")).await;
    assert_eq!((status, replay_id.as_deref()), (200, Some(k3.as_str())));
    assert_eq!(replay, k3_body);
    assert_eq!(ledger(&native).await, (1, 7, 4));
    let ids = Ids {
        held,
        k1,
        k3,
        unkeyed,
    };

    // Crash (no shutdown path) with the stream in flight; keep the journal as it is for the first rollback.
    drop(reader);
    native.cered.kill();
    let crashed = copy_state(native.state.path());
    let journal_lines = |dir: &Path| {
        std::fs::read_to_string(dir.join("journal.jsonl"))
            .unwrap()
            .lines()
            .count()
    };
    eprintln!("crashed journal: {} lines", journal_lines(crashed.path()));

    // Candidate restart: replay drops k1/k2 (key_expired) and evicts u1/u2 again (10 replayed > 8); the held
    // request is failed `worker_restart` and keeps its slot; a full ledger still admits by evicting.
    let config = native.config_path.clone();
    native.respawn(&candidate(), &config, "toy").await;
    assert_eq!(native.settled().await.0, 200);
    let (live, terminal, evictions) = ledger(&native).await;
    assert_eq!((live, terminal), (0, MAX_RECORDS));
    eprintln!("restart evictions: {evictions}");
    assert_eq!(
        state(&native, &ids.held).await,
        (200, Some("failed".into()))
    );
    assert_eq!(native.terminal(&ids.held).await["reason"], "worker_restart");
    for gone in [&ids.k1, &ids.unkeyed[0], &ids.unkeyed[1]] {
        assert_eq!(state(&native, gone).await, (404, None), "{gone}");
    }
    assert_eq!(state(&native, &ids.unkeyed[2]).await.0, 200);
    assert_eq!(
        refused(&native, Some("k1")).await,
        (409, "idempotency_key_expired".into())
    );
    assert_eq!(
        refused(&native, Some("k3")).await,
        (409, "response_not_retained".into())
    );
    let f1 = completes(&native, None).await;
    let f2 = completes(&native, None).await;
    let (live, terminal, after) = ledger(&native).await;
    assert_eq!((live, terminal, after), (0, MAX_RECORDS, evictions + 2));
    assert_eq!(state(&native, &ids.unkeyed[2]).await, (404, None));
    assert_eq!(state(&native, &ids.unkeyed[3]).await, (404, None));
    native.cered.signal("TERM");
    let (code, stderr) = native.cered.exit(Duration::from_secs(30)).await;
    assert_eq!(code, Some(0), "{stderr}");
    let stopped = copy_state(native.state.path());
    eprintln!("stopped journal: {} lines", journal_lines(stopped.path()));

    let Some(rollback) = rollback else {
        eprintln!("{ROLLBACK_ENV} unset: the rollback half is skipped");
        return;
    };
    let oldest_retained = ids.unkeyed[4].clone();

    // Rollback from the crash: the earlier release replays every admission the journal still holds (evicted
    // ones too; it only drops records past retention), so it opens over its bound. The rollback config has
    // another config hash, which every request digest binds (docs/API.md): a key retried across the rollback is
    // `idempotency_conflict`, an evicted one too (its record is back), and nothing runs.
    let path = rollback_config(&native, &crashed, "rollback-crashed.toml");
    native.respawn(&rollback, &path, "toy").await;
    let (ready, readyz) = native.settled().await;
    assert_eq!(ready, 200, "{readyz}");
    assert_eq!(refused(&native, None).await, (503, "overloaded".into()));
    assert_eq!(
        state(&native, &ids.unkeyed[0]).await.0,
        200,
        "u1 comes back"
    );
    assert_eq!(state(&native, &ids.k1).await.0, 200, "k1 comes back");
    let submissions = native.submissions().await;
    for key in ["k1", "k3"] {
        assert_eq!(
            refused(&native, Some(key)).await,
            (409, "idempotency_conflict".into()),
            "{key}"
        );
    }
    assert_eq!(native.submissions().await, submissions);
    native.cered.signal("TERM");
    let (code, stderr) = native.cered.exit(Duration::from_secs(30)).await;
    assert_eq!(code, Some(0), "{stderr}");

    // Rollback from the graceful stop: the journal holds exactly the candidate's eight records (and k1/k2's
    // tombstones), so the earlier release opens full and refuses new requests until a record expires.
    let path = rollback_config(&native, &stopped, "rollback-stopped.toml");
    native.respawn(&rollback, &path, "toy").await;
    let (ready, readyz) = native.settled().await;
    assert_eq!(ready, 200, "{readyz}");
    assert_eq!(refused(&native, None).await, (503, "overloaded".into()));
    assert_eq!(state(&native, &ids.k1).await, (404, None));
    let submissions = native.submissions().await;
    assert_eq!(
        refused(&native, Some("k1")).await,
        (409, "idempotency_key_expired".into())
    );
    assert_eq!(
        refused(&native, Some("k3")).await,
        (409, "idempotency_conflict".into())
    );
    assert_eq!(native.submissions().await, submissions);
    for kept in [&ids.held, &ids.k3, &oldest_retained, &f1, &f2] {
        assert_eq!(state(&native, kept).await.0, 200, "{kept}");
    }
    let oldest = native.terminal(&oldest_retained).await;
    eprintln!("oldest retained: {oldest}");
    let started = Instant::now();
    let first = loop {
        let (status, id, body) = unary(&native, None).await;
        if status == 200 {
            break id.unwrap();
        }
        assert_eq!(body["error"]["code"], "overloaded", "{body}");
        assert!(
            started.elapsed() < Duration::from_millis(RETENTION_MS + 10_000),
            "the rollback never admitted a new request"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    eprintln!(
        "rollback(stopped) first admission after {:?} of refusals",
        started.elapsed()
    );
    assert_eq!(native.terminal(&first).await["state"], "completed");
    assert_eq!(state(&native, &oldest_retained).await, (404, None));
}
