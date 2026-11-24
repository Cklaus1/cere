//! The real `cered` binary: startup auth failure (S5), crash replay after SIGKILL (L3), the graceful
//! SIGTERM shutdown with its exit-code contract (L8) and the backend-launch-change log line (m1close R5).
mod support;

use std::time::Duration;

use cere_mock_sglang::{Fault, MockConfig, MockSglang};
use serde_json::Value;
use support::*;

fn slow_mock() -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(50),
        natural_output_tokens: 2_000,
        ..MockConfig::default()
    }
}

fn long_stream() -> Value {
    let mut body = chat_body(true);
    body["max_completion_tokens"] = 2_000.into();
    body
}

/// Opens a stream on `cered` and reads the accepted event plus one token; returns (rid, reader).
async fn open_stream(base: &str, key: Option<&str>) -> (String, SseReader) {
    let mut request = http()
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN_A)
        .json(&long_stream());
    if let Some(key) = key {
        request = request.header("Idempotency-Key", key);
    }
    let response = request.send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut reader = SseReader::new(response);
    assert_eq!(
        reader.next().await.unwrap().data()["cere"]["event"],
        "accepted"
    );
    assert!(
        reader.next().await.unwrap().data()["cere"]["seq"]
            .as_u64()
            .unwrap()
            >= 1
    );
    (rid, reader)
}

async fn wait_aborted(mock: &cere_mock_sglang::MockHandle, rid: &str) {
    for _ in 0..250 {
        if mock.abort_requests().iter().any(|a| a.rid == rid) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the mock never saw /abort_request for {rid}");
}

#[tokio::test(flavor = "multi_thread")]
async fn s5_missing_or_empty_token_env_fails_startup() {
    let mock = MockSglang::start(MockConfig::default()).await.unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[],
    ));
    for value in [None, Some("")] {
        let Err((code, stderr)) = Cered::spawn(&config.path, &[("TOK_B", value)]).await else {
            panic!("cered started without TOK_B ({value:?})");
        };
        assert_eq!(code, Some(2), "{stderr}");
        assert!(stderr.contains("TOK_B"), "{stderr}");
    }
    assert!(
        mock.dispatched_rids().is_empty(),
        "nothing reached the backend"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn l3_crash_replay_fails_aborts_and_never_regenerates() {
    let mock = MockSglang::start(slow_mock()).await.unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[],
    ));
    let mut first = Cered::spawn(&config.path, &[]).await.unwrap();
    first.wait_ready().await;
    let (rid, _reader) = open_stream(&first.base, Some("crash-1")).await;
    first.kill();

    let second = Cered::spawn(&config.path, &[]).await.unwrap();
    assert_eq!(second.epoch, first.epoch + 1);
    // The orphan is failed at startup and aborted on the backend.
    wait_aborted(&mock, &rid).await;
    second.wait_ready().await;
    let r = http()
        .get(second.url(&format!("/v1/cere/requests/{rid}")))
        .bearer_auth(TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let record: Value = r.json().await.unwrap();
    assert_eq!(record["state"], "failed");
    assert_eq!(record["reason"], "worker_restart");
    assert_eq!(record["epoch"], first.epoch);

    // The same key is answered from the ledger, never generated again.
    let dispatched = mock.dispatched_rids();
    assert_eq!(dispatched.iter().filter(|d| **d == rid).count(), 1);
    let retry = http()
        .post(second.url("/v1/chat/completions"))
        .bearer_auth(TOKEN_A)
        .header("Idempotency-Key", "crash-1")
        .json(&long_stream())
        .send()
        .await
        .unwrap();
    assert_eq!(retry.status(), 409);
    let body: Value = retry.json().await.unwrap();
    assert_eq!(body["error"]["code"], "stream_not_replayable");
    assert_eq!(
        body["error"]["status_url"],
        format!("/v1/cere/requests/{rid}").as_str()
    );
    assert_eq!(
        mock.dispatched_rids().len(),
        dispatched.len(),
        "no new dispatch"
    );
}

/// m1close Q6/Q7 across real crashes: the orphan's backend request keeps generating after `cered` is killed
/// (SGLang ignores the late abort). The next incarnation holds it while `L > C`; killed during that hold, the one
/// after charges it again (`re-charged`) and releases it by load once the generation really ended. The orphan
/// generates until the test ends it (`MockHandle::end_kept`) after the third incarnation aborted it, so no
/// incarnation's start races the generation's own end.
#[tokio::test(flavor = "multi_thread")]
async fn q7_sigkill_during_the_hold_recharges_then_releases_by_load() {
    // 100 ms per token up to the stream's 2 000-token budget: left alone, the orphan would outlast the test.
    let mock = MockSglang::start(MockConfig {
        token_latency: Duration::from_millis(100),
        natural_output_tokens: 100_000,
        ..MockConfig::default()
    })
    .await
    .unwrap();
    mock.set_fault(Fault::KeepGeneratingAfterDisconnect);
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[
            (
                "shared_backend_tenancy = \"shared_acknowledged\"",
                "shared_backend_tenancy = \"shared_acknowledged\"\nload_endpoint = \"/get_load\"",
            ),
            (
                "terminal_retention_ms = 3600000",
                "terminal_retention_ms = 3600000\norphan_load_poll_ms = 1000",
            ),
        ],
    ));
    let aborts = |rid: &str| {
        mock.abort_requests()
            .iter()
            .filter(|a| a.rid == rid)
            .count()
    };
    let mut first = Cered::spawn(&config.path, &[]).await.unwrap();
    first.wait_ready().await;
    let (rid, _reader) = open_stream(&first.base, None).await;
    first.kill();

    let mut second = Cered::spawn(&config.path, &[]).await.unwrap();
    wait_aborted(&mock, &rid).await;
    // A load poll (every 1 s) or more sees the orphan generating (L = 1 > C = 0): the hold goes on.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert_eq!(mock.load(), 1, "the orphan stopped generating");
    second.kill();
    let (_, stderr) = second.exit(Duration::from_secs(5)).await;
    assert!(!stderr.contains("cered recovery: released"), "{stderr}");
    assert!(non_terminal(&journal(state.path())).is_empty());

    let mut third = Cered::spawn(&config.path, &[]).await.unwrap();
    // The third incarnation charged the orphan again before its abort: only now does the generation end.
    let until = std::time::Instant::now() + Duration::from_secs(15);
    while aborts(&rid) < 2 {
        assert!(
            std::time::Instant::now() < until,
            "the third incarnation never aborted the orphan"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    mock.end_kept();
    let until = std::time::Instant::now() + Duration::from_secs(15);
    let released = loop {
        let records = journal(state.path());
        if let Some(r) = records.iter().find(|r| r["type"] == "orphan_released") {
            break r.clone();
        }
        assert!(std::time::Instant::now() < until, "orphan never released");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(
        (released["rid"].as_str(), released["cause"].as_str()),
        (Some(rid.as_str()), Some("load"))
    );
    assert_eq!(mock.load(), 0);
    assert_eq!(aborts(&rid), 2, "each restart aborts the orphan again");
    // The journal record precedes the `released` stderr line, and the slot is freed after that line
    // (`OrphanHold::run`): kill only once scope `a` charges nothing.
    let until = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let stats: Value = http()
            .get(format!("{}/v1/cere/stats", third.base))
            .bearer_auth(TOKEN_A)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let inflight: u64 = stats["workers"]
            .as_object()
            .unwrap_or_else(|| panic!("stats without workers: {stats}"))
            .values()
            .map(|w| w["inflight"].as_u64().unwrap())
            .sum();
        if inflight == 0 {
            break;
        }
        assert!(
            std::time::Instant::now() < until,
            "slot never freed: {stats}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    third.kill();
    let (_, stderr) = third.exit(Duration::from_secs(5)).await;
    // Q11: the worker (`gpu1`, the single form's device, at the first incarnation's epoch 1) right after `rid=`;
    // M5 D5: the model of that worker's deployment right after `scope=`.
    assert!(
        stderr.contains(&format!(
            "cered recovery: re-charged rid={rid} worker=gpu1@1 scope=a model=Qwen3.8-Flash-Next (held"
        )),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!(
            "cered recovery: released rid={rid} worker=gpu1@1 cause=load"
        )),
        "{stderr}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn l8_sigterm_mid_stream_settles_and_exits_clean() {
    let mock = MockSglang::start(slow_mock()).await.unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[("shutdown_drain_ms = 30000", "shutdown_drain_ms = 300")],
    ));
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    cered.wait_ready().await;
    let (rid, mut reader) = open_stream(&cered.base, None).await;
    cered.signal("TERM");

    // While draining: not ready, new work refused before any ledger write.
    let http = http();
    let mut refused = false;
    for _ in 0..50 {
        let ready = http.get(cered.url("/readyz")).send().await.unwrap();
        if ready.status() == 503 {
            assert_eq!(ready.json::<Value>().await.unwrap()["stopping"], true);
            let r = http
                .post(cered.url("/v1/chat/completions"))
                .bearer_auth(TOKEN_A)
                .json(&chat_body(false))
                .send()
                .await
                .unwrap();
            assert_eq!(r.status(), 503);
            assert!(r.headers().get("x-cere-request-id").is_none());
            assert_eq!(
                r.json::<Value>().await.unwrap()["error"]["code"],
                "shutting_down"
            );
            refused = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(refused, "readiness never reported stopping");

    // `open_stream` consumed the accepted event and one more.
    let events = reader.rest().await;
    let terminal = check_stream_tail(&events, 1);
    assert!(terminal.get("error").is_some(), "{terminal}");
    assert_eq!(terminal["cere"]["state"], "cancelled");
    assert_eq!(terminal["cere"]["reason"], "shutdown");

    let (code, stderr) = cered.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    assert!(stderr.contains("cered shutdown clean=true"), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "rid={rid} scope=a worker=gpu1@1 state=cancelled reason=shutdown"
        )),
        "{stderr}"
    );
    wait_aborted(&mock, &rid).await;
    assert_eq!(non_terminal(&journal(state.path())), Vec::<String>::new());
}

#[tokio::test(flavor = "multi_thread")]
async fn l8_forced_stop_never_reports_success() {
    let mock = MockSglang::start(slow_mock()).await.unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[("shutdown_drain_ms = 30000", "shutdown_drain_ms = 0")],
    ));
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    cered.wait_ready().await;
    // The backend keeps generating after the abort (the mock fixes the fault at dispatch): the stop cannot
    // be clean.
    mock.set_fault(Fault::IgnoreAbort);
    let (rid, mut reader) = open_stream(&cered.base, None).await;
    cered.signal("TERM");
    let events = reader.rest().await;
    assert_eq!(events.last(), Some(&Event::Done));
    let (code, stderr) = cered.exit(Duration::from_secs(20)).await;
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("open_backend_streams=1"), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "rid={rid} scope=a worker=gpu1@1 state=failed reason=cancel_unacknowledged"
        )),
        "{stderr}"
    );
    wait_aborted(&mock, &rid).await;
    assert_eq!(non_terminal(&journal(state.path())), Vec::<String>::new());
}

/// H2/Q5: the stderr terminal line carries the reported counts (a pinned cache report: miss -> 0/reported).
#[tokio::test(flavor = "multi_thread")]
async fn h2_stderr_terminal_line_reports_usage() {
    let mock = MockSglang::start(MockConfig {
        server_info: cere_mock_sglang::ServerInfo {
            enable_cache_report: true,
            ..Default::default()
        },
        ..MockConfig::default()
    })
    .await
    .unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[("enable_cache_report = false", "enable_cache_report = true")],
    ));
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    cered.wait_ready().await;
    let r = http()
        .post(cered.url("/v1/chat/completions"))
        .bearer_auth(TOKEN_A)
        .json(&chat_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.starts_with(&format!("cered request rid={rid} ")))
        .unwrap_or_else(|| panic!("no terminal line: {stderr}"));
    // Q11: the worker right after the scope.
    assert!(
        line.starts_with(&format!(
            "cered request rid={rid} scope=a worker=gpu1@1 state=completed "
        )),
        "{line}"
    );
    assert!(
        line.ends_with(" prompt_tokens=2 cached_tokens=0 cache_report=reported"),
        "{line}"
    );
}

/// m1close R5/Q10: the real binary's background probe (production interval) logs a backend relaunch exactly
/// once, `cered readiness: backend launch changed instance_id=<old> -> <new>; smoking again`, and later probes
/// of the same launch stay silent.
#[tokio::test(flavor = "multi_thread")]
async fn r5_launch_change_is_logged_once() {
    let launch = |seed: u64| cere_mock_sglang::ServerInfo {
        random_seed: Some(seed),
        startup_time: Some(serde_json::json!({"total": 40.0})),
        ..Default::default()
    };
    let mock = MockSglang::start(MockConfig {
        server_info: launch(1),
        ..MockConfig::default()
    })
    .await
    .unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[],
    ));
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    cered.wait_ready().await;
    // The single-form worker's Q11 entry (`scheduler.devices[0]` of the t4 profile).
    let readyz = || async {
        let r = http().get(cered.url("/readyz")).send().await.unwrap();
        r.json::<Value>().await.unwrap()["workers"]["gpu1"].clone()
    };
    let old = readyz().await["instance_id"].as_str().unwrap().to_owned();
    mock.set_server_info_override(Some(launch(2)));
    // Ready under the new launch, then one more probe of it.
    let mut relaunched_at = None;
    for _ in 0..1_000 {
        let body = readyz().await;
        let probed = body["probed_at_unix_ms"].as_u64().unwrap();
        match relaunched_at {
            None if body["ready"] == true && body["instance_id"] != old.as_str() => {
                relaunched_at = Some(probed);
            }
            Some(at) if probed > at => break,
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let at = relaunched_at.expect("never ready under the new launch");
    assert!(
        readyz().await["probed_at_unix_ms"].as_u64().unwrap() > at,
        "no probe after the relaunch"
    );
    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    let logged: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("backend launch changed"))
        .collect();
    assert_eq!(logged.len(), 1, "{stderr}");
    assert!(
        logged[0].starts_with(&format!(
            "cered readiness: backend launch changed instance_id={old} -> sglang-"
        )),
        "{}",
        logged[0]
    );
}

/// m1close R5/Q10 + X3 S9: a real relaunch takes the backend away for minutes, so readiness passes through a
/// not-ready probe (pins unreadable or different) before the new launch appears. The relaunch is still logged,
/// once, against the last identity the gate opened for.
#[tokio::test(flavor = "multi_thread")]
async fn r5_launch_change_across_an_outage_is_logged() {
    let launch = |seed: u64| cere_mock_sglang::ServerInfo {
        random_seed: Some(seed),
        startup_time: Some(serde_json::json!({"total": 40.0})),
        ..Default::default()
    };
    let mock = MockSglang::start(MockConfig {
        server_info: launch(1),
        ..MockConfig::default()
    })
    .await
    .unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[],
    ));
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    cered.wait_ready().await;
    let readyz = || async {
        let r = http().get(cered.url("/readyz")).send().await.unwrap();
        r.json::<Value>().await.unwrap()["workers"]["gpu1"].clone()
    };
    let old = readyz().await["instance_id"].as_str().unwrap().to_owned();
    // The outage: a probe sees a backend that does not match the pins.
    mock.set_server_info_override(Some(cere_mock_sglang::ServerInfo {
        version: "restarting".to_owned(),
        ..launch(1)
    }));
    let mut down = false;
    for _ in 0..1_000 {
        if readyz().await["ready"] == false {
            down = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(down, "the pin drift never made cered not ready");
    mock.set_server_info_override(Some(launch(2)));
    let mut relaunched = false;
    for _ in 0..1_000 {
        let body = readyz().await;
        if body["ready"] == true && body["instance_id"] != old.as_str() {
            relaunched = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(relaunched, "never ready under the new launch");
    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    let logged: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("backend launch changed"))
        .collect();
    assert_eq!(logged.len(), 1, "{stderr}");
    assert!(
        logged[0].starts_with(&format!(
            "cered readiness: backend launch changed instance_id={old} -> sglang-"
        )),
        "{}",
        logged[0]
    );
}
