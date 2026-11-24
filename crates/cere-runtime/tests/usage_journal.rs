//! net-teleport design note 4.4 / bar S7: the usage journal (`<state_dir>/usage/YYYY-MM-DD.jsonl`) against the
//! real `cered` binary on a mock backend. Its totals equal the sum of the `cered request` lines (exact token
//! counts) across a graceful restart and a SIGKILL (its in-flight request failed `worker_restart` by the next
//! incarnation), it never holds a request id or content, and `[usage] enabled = false` writes nothing. Also the
//! writer's retention prune and its repair of a crash-cut last line.
//!
//! With `CERE_USAGE_EVIDENCE_DIR` set, the restart test copies each incarnation's stderr and the usage directory
//! there (`scripts/live/usage_report.py --request-log` then recomputes the comparison; an internal benchmark report).
mod support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockSglang, ServerInfo};
use cere_runtime::usage::{day_file_name, OwnerOnlySpan, UsageJournal, UsageRecord, USAGE_DIR};
use serde_json::{json, Value};
use support::*;

/// Appears in every prompt; must never reach the journal.
const MARKER: &str = "usage-marker-7f3a";

const DAY_MS: u64 = 86_400_000;

/// `(scope, worker id, state)` -> `[requests, prompt, cached]`.
type Totals = BTreeMap<(String, String, String), [u64; 3]>;

fn cache_mock() -> MockConfig {
    MockConfig {
        server_info: ServerInfo {
            enable_cache_report: true,
            ..Default::default()
        },
        cache_granularity: 4,
        ..MockConfig::default()
    }
}

fn long_mock() -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(50),
        natural_output_tokens: 2_000,
        ..cache_mock()
    }
}

fn body(words: usize, max_tokens: u64, stream: bool) -> Value {
    let text = std::iter::repeat_n(MARKER, words)
        .collect::<Vec<_>>()
        .join(" ");
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_completion_tokens": max_tokens,
        "stream": stream,
    })
}

/// Sends one chat request and reads its whole response; returns the status.
async fn send(cered: &Cered, token: &str, body: &Value) -> u16 {
    let r = http()
        .post(cered.url("/v1/chat/completions"))
        .bearer_auth(token)
        .json(body)
        .send()
        .await
        .unwrap();
    let status = r.status().as_u16();
    r.bytes().await.unwrap();
    status
}

/// A mix of scopes, prompt lengths, output lengths, stream and unary, and repeats (prefix-cache hits).
async fn traffic(cered: &Cered, round: usize) -> usize {
    let mut sent = 0;
    for (i, (token, words, max, stream)) in [
        (TOKEN_A, 1, 4, false),
        (TOKEN_A, 1, 4, false),
        (TOKEN_A, 7 + round, 9, true),
        (TOKEN_B, 3, 2, false),
        (TOKEN_B, 3, 2, true),
        (TOKEN_A, 40, 16, false),
        (TOKEN_A, 40, 16, true),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            send(cered, token, &body(words, max, stream)).await,
            200,
            "request {i}"
        );
        sent += 1;
    }
    sent
}

/// The field `key=` of a space-separated line.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split(' ')
        .find_map(|kv| kv.strip_prefix(key)?.strip_prefix('='))
}

fn count(v: Option<&str>) -> u64 {
    v.and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// The `cered request` lines of `stderr`, summed.
fn line_totals(stderr: &str, into: &mut Totals) -> usize {
    let mut n = 0;
    for line in stderr.lines().filter(|l| l.starts_with("cered request ")) {
        let worker = field(line, "worker").unwrap();
        let worker = worker.split('@').next().unwrap();
        let key = (
            field(line, "scope").unwrap().to_owned(),
            worker.to_owned(),
            field(line, "state").unwrap().to_owned(),
        );
        let t = into.entry(key).or_default();
        t[0] += 1;
        t[1] += count(field(line, "prompt_tokens"));
        t[2] += count(field(line, "cached_tokens"));
        n += 1;
    }
    n
}

/// Output tokens of the chat access lines (the request line has none).
fn access_output_tokens(stderr: &str) -> u64 {
    stderr
        .lines()
        .filter_map(|l| l.strip_prefix("cered access "))
        .map(|j| serde_json::from_str::<Value>(j).unwrap())
        .filter(|v| v["route"] == "/v1/chat/completions")
        .map(|v| v["output_tokens"].as_u64().unwrap_or(0))
        .sum()
}

fn usage_dir(state: &Path) -> PathBuf {
    state.join(USAGE_DIR)
}

/// Every journal line under `dir`, parsed, plus the raw text.
fn journal_rows(dir: &Path) -> (Vec<Value>, String) {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    names.sort();
    let mut text = String::new();
    for name in names {
        text.push_str(&std::fs::read_to_string(name).unwrap());
    }
    let rows = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
        .collect();
    (rows, text)
}

fn row_totals(rows: &[Value]) -> (Totals, u64) {
    let mut totals = Totals::new();
    let mut output = 0;
    for r in rows.iter().filter(|r| r["kind"] == "request") {
        let key = (
            r["scope"].as_str().unwrap().to_owned(),
            r["worker"].as_str().unwrap_or("-").to_owned(),
            r["state"].as_str().unwrap().to_owned(),
        );
        let t = totals.entry(key).or_default();
        t[0] += 1;
        t[1] += r["prompt_tokens"].as_u64().unwrap_or(0);
        t[2] += r["cached_tokens"].as_u64().unwrap_or(0);
        output += r["output_tokens"].as_u64().unwrap_or(0);
    }
    (totals, output)
}

/// Waits until the journal holds `n` lines (the writer runs behind the serving path).
async fn wait_rows(dir: &Path, n: usize) {
    for _ in 0..250 {
        if dir.exists() && journal_rows(dir).0.len() >= n {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the usage journal never reached {n} rows");
}

/// Bar S7: N requests, graceful restart, M requests, SIGKILL with one request in flight, restart (which fails it
/// `worker_restart`), K requests, stop. The journal equals the request lines of all three incarnations.
#[tokio::test(flavor = "multi_thread")]
async fn s7_journal_equals_request_lines_across_restarts() {
    let mock = MockSglang::start(long_mock()).await.unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[("enable_cache_report = false", "enable_cache_report = true")],
    ));
    let dir = usage_dir(state.path());
    let mut logs = Vec::new();

    // Incarnation 1: N requests, SIGTERM.
    let mut sent = 0;
    let mut first = Cered::spawn(&config.path, &[]).await.unwrap();
    first.wait_ready().await;
    sent += traffic(&first, 0).await;
    first.signal("TERM");
    let (code, stderr) = first.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    logs.push(stderr);

    // Incarnation 2: M requests, then one long stream in flight and SIGKILL.
    let mut second = Cered::spawn(&config.path, &[]).await.unwrap();
    second.wait_ready().await;
    sent += traffic(&second, 1).await;
    wait_rows(&dir, sent).await;
    let response = http()
        .post(second.url("/v1/chat/completions"))
        .bearer_auth(TOKEN_A)
        .json(&body(2, 2_000, true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let mut reader = SseReader::new(response);
    reader.next().await.unwrap();
    reader.next().await.unwrap();
    second.kill();
    logs.push(second.stderr_so_far());

    // Incarnation 3: fails the orphan, K requests, SIGTERM.
    let mut third = Cered::spawn(&config.path, &[]).await.unwrap();
    third.wait_ready().await;
    sent += traffic(&third, 2).await;
    third.signal("TERM");
    let (code, stderr) = third.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    logs.push(stderr);

    let mut lines = Totals::new();
    let n_lines: usize = logs.iter().map(|l| line_totals(l, &mut lines)).sum();
    assert_eq!(
        n_lines,
        sent + 1,
        "every request and the orphan have a line"
    );
    assert_eq!(
        lines.get(&("a".into(), "gpu1".into(), "failed".into())),
        Some(&[1, 0, 0]),
        "the SIGKILLed stream failed worker_restart: {lines:?}"
    );
    let (rows, text) = journal_rows(&dir);
    let (journal, output) = row_totals(&rows);
    assert_eq!(journal, lines, "usage journal != cered request lines");
    assert!(lines.values().map(|t| t[2]).sum::<u64>() > 0, "cache hits");
    let access_output: u64 = logs.iter().map(|l| access_output_tokens(l)).sum();
    assert_eq!(output, access_output, "output tokens != access lines");

    // No request id, no content; exactly the documented fields.
    let fields = [
        "kind",
        "ts_ms",
        "scope",
        "worker",
        "worker_owner",
        "tier",
        "class",
        "state",
        "reason",
        "prompt_tokens",
        "cached_tokens",
        "output_tokens",
        "queue_wait_ms",
        "wall_ms",
    ];
    for row in &rows {
        let keys: Vec<&str> = row.as_object().unwrap().keys().map(|k| &**k).collect();
        let mut want = fields.to_vec();
        want.sort_unstable();
        let mut keys = keys.clone();
        keys.sort_unstable();
        assert_eq!(keys, want, "{row}");
        assert_eq!(row["tier"], "own", "a local worker carries server.tier");
        assert_eq!(row["worker_owner"], Value::Null);
        // S6 (C17): the default class, as no request names one; unknown (null) only for a request the previous
        // incarnation left in flight (the class is memory-only).
        let class = if row["reason"] == "worker_restart" {
            Value::Null
        } else {
            json!("interactive")
        };
        assert_eq!(row["class"], class, "{row}");
    }
    assert!(!text.contains(MARKER), "prompt content in the journal");
    assert!(!text.contains("req_"), "a request id in the journal");
    for log in &logs {
        for rid in log
            .lines()
            .filter(|l| l.starts_with("cered request "))
            .filter_map(|l| field(l, "rid"))
        {
            assert!(!text.contains(rid), "{rid} in the journal");
        }
    }

    if let Ok(out) = std::env::var("CERE_USAGE_EVIDENCE_DIR") {
        let out = PathBuf::from(out);
        std::fs::create_dir_all(out.join("usage")).unwrap();
        for (i, log) in logs.iter().enumerate() {
            std::fs::write(out.join(format!("cered-{}.stderr.log", i + 1)), log).unwrap();
        }
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), out.join("usage").join(entry.file_name())).unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn usage_disabled_writes_nothing() {
    let mock = MockSglang::start(MockConfig::default()).await.unwrap();
    let state = StateDir::new();
    let mut text = config_text(&mock.base_url(), state.path(), free_port(), &[]);
    text.push_str("\n[usage]\nenabled = false\n");
    let config = ConfigFile::new(&text);
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    cered.wait_ready().await;
    assert_eq!(send(&cered, TOKEN_A, &body(1, 4, false)).await, 200);
    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    assert!(stderr.contains("cered request "), "{stderr}");
    assert!(!usage_dir(state.path()).exists());
}

/// Retention: a start prunes day files `retention_days` or more before today and keeps the rest (and other
/// names); a crash-cut last line is closed before the next row.
#[test]
fn retention_prune_and_crash_tail_repair() {
    let state = StateDir::new();
    let dir = usage_dir(state.path());
    std::fs::create_dir_all(&dir).unwrap();
    let today = 20_780; // 2026-11-23
    let now_ms = today * DAY_MS + 3_600_000;
    for day in [today - 90, today - 89, today - 1] {
        std::fs::write(dir.join(day_file_name(day)), "{}\n").unwrap();
    }
    std::fs::write(dir.join("notes.txt"), "kept").unwrap();
    // A row cut by a crash mid-write.
    std::fs::write(dir.join(day_file_name(today)), "{\"kind\":\"requ").unwrap();

    let journal = UsageJournal::start(&dir, 90, now_ms).unwrap();
    journal.record_owner_only(OwnerOnlySpan {
        worker: "gpu1".to_owned(),
        owner: Some("a".to_owned()),
        start_ms: now_ms - 7_200_000,
        end_ms: now_ms,
    });
    journal.close();
    assert_eq!(journal.dropped(), 0);

    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            day_file_name(today - 89),
            day_file_name(today - 1),
            day_file_name(today),
            "notes.txt".to_owned(),
        ]
    );
    let text = std::fs::read_to_string(dir.join(day_file_name(today))).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    assert_eq!(lines[0], "{\"kind\":\"requ");
    let record: UsageRecord = serde_json::from_str(lines[1]).unwrap();
    assert!(matches!(&record, UsageRecord::OwnerOnly(s) if s.worker == "gpu1"));
    let v: Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(v["kind"], "owner_only");
    assert!(text.ends_with('\n'));
}
