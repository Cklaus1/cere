//! ops-hardening design note OH-05/OH-06: `GET /metrics` (operator auth, a strictly parsed Prometheus text
//! exposition whose counters move with traffic, the native worker's prefix cache counters) and the `cered access`
//! line of each finished request (fields, no content, no secret).
mod native_server;
mod native_support;
mod support;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockSglang};
use cere_runtime::metrics::METRIC_NAMES;
use native_server::{Case, Native, WORKER};
use serde_json::{json, Value};
use support::{
    chat_body, config_text, free_port, http, Cered, ConfigFile, Event, Harness, SseReader,
    StateDir, MODEL, OPERATOR_TOKEN, OPERATOR_TWEAK, TOKEN_A, TOKEN_B,
};

/// The single-form worker of `configs/runtime/sglang-t4-qwen38.toml` (`scheduler.devices[0]`).
const SGLANG_WORKER: &str = "gpu1";

/// A text exposition, checked line by line ([`parse`]).
#[derive(Debug)]
struct Scrape {
    /// Family name -> its `# TYPE`.
    families: BTreeMap<String, String>,
    /// Family names in exposition order.
    order: Vec<String>,
    /// The series (`name{labels}` as rendered) -> its value.
    samples: BTreeMap<String, f64>,
}

impl Scrape {
    fn value(&self, series: &str) -> f64 {
        *self
            .samples
            .get(series)
            .unwrap_or_else(|| panic!("no series {series}: {:?}", self.samples.keys()))
    }

    /// The value of `series`, 0 when absent (a counter not yet incremented).
    fn counter(&self, series: &str) -> f64 {
        self.samples.get(series).copied().unwrap_or(0.0)
    }

    /// Every series of family `name` (samples whose name is exactly `name`).
    fn series_of(&self, name: &str) -> Vec<(&str, f64)> {
        self.samples
            .iter()
            .filter(|(s, _)| s.split('{').next() == Some(name))
            .map(|(s, v)| (s.as_str(), *v))
            .collect()
    }
}

fn is_metric_name(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == ':')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
}

fn is_label_name(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `{k="v",...}` (the braces stripped) into its pairs, values unescaped; panics on any syntax error.
fn parse_labels(text: &str, line: &str) -> Vec<(String, String)> {
    let mut labels = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let (key, after) = rest
            .split_once("=\"")
            .unwrap_or_else(|| panic!("label without =\": {line}"));
        assert!(is_label_name(key), "label name {key:?}: {line}");
        let mut value = String::new();
        let mut chars = after.char_indices();
        let end = loop {
            match chars.next() {
                Some((_, '\\')) => match chars.next() {
                    Some((_, '\\')) => value.push('\\'),
                    Some((_, '"')) => value.push('"'),
                    Some((_, 'n')) => value.push('\n'),
                    other => panic!("bad escape {other:?}: {line}"),
                },
                Some((i, '"')) => break i,
                Some((_, '\n')) | None => panic!("unterminated label value: {line}"),
                Some((_, c)) => value.push(c),
            }
        };
        assert!(
            labels.iter().all(|(k, _)| k != key),
            "duplicate label {key}: {line}"
        );
        labels.push((key.to_owned(), value));
        rest = &after[end + 1..];
        if let Some(next) = rest.strip_prefix(',') {
            assert!(!next.is_empty(), "trailing comma: {line}");
            rest = next;
        } else {
            assert!(rest.is_empty(), "junk after a label: {line}");
        }
    }
    labels
}

/// Parses the text exposition format 0.0.4 strictly: every family opens with `# HELP` then `# TYPE` (counter,
/// gauge or histogram) and appears once; every sample belongs to the open family (a histogram's `_bucket`, `_sum`,
/// `_count`); label syntax and escapes are valid; no series repeats; counters are named `_total` and never
/// negative; each histogram series has ascending `le` buckets, cumulative counts, a last `+Inf` bucket equal to its
/// `_count`, and a `_sum`.
fn parse(text: &str) -> Scrape {
    assert!(text.ends_with('\n'), "no final newline");
    let mut scrape = Scrape {
        families: BTreeMap::new(),
        order: Vec::new(),
        samples: BTreeMap::new(),
    };
    let mut open: Option<(String, String)> = None;
    let mut pending_help: Option<String> = None;
    // Histogram label set (without `le`) -> its (le, cumulative count) buckets, sum and count.
    type Buckets = Vec<(f64, f64)>;
    let mut histograms: BTreeMap<String, (Buckets, Option<f64>, Option<f64>)> = BTreeMap::new();
    for line in text.lines() {
        assert!(!line.is_empty(), "empty line");
        if let Some(rest) = line.strip_prefix("# HELP ") {
            let (name, help) = rest.split_once(' ').unwrap_or((rest, ""));
            assert!(is_metric_name(name), "{line}");
            assert!(!help.trim().is_empty(), "empty help: {line}");
            assert!(pending_help.is_none(), "HELP without TYPE before {line}");
            pending_help = Some(name.to_owned());
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, kind) = rest.split_once(' ').unwrap_or_else(|| panic!("{line}"));
            assert_eq!(
                pending_help.take().as_deref(),
                Some(name),
                "TYPE without its HELP: {line}"
            );
            assert!(
                ["counter", "gauge", "histogram"].contains(&kind),
                "type {kind}: {line}"
            );
            assert!(
                scrape
                    .families
                    .insert(name.to_owned(), kind.to_owned())
                    .is_none(),
                "family twice: {line}"
            );
            if kind == "counter" {
                assert!(name.ends_with("_total"), "counter without _total: {line}");
            }
            scrape.order.push(name.to_owned());
            open = Some((name.to_owned(), kind.to_owned()));
            continue;
        }
        assert!(!line.starts_with('#'), "unknown comment: {line}");
        assert!(pending_help.is_none(), "HELP without TYPE before {line}");
        let (series, value) = line
            .rsplit_once(' ')
            .unwrap_or_else(|| panic!("no value: {line}"));
        let value: f64 = value
            .parse()
            .unwrap_or_else(|_| panic!("bad value: {line}"));
        assert!(value.is_finite(), "{line}");
        let (name, labels) = match series.split_once('{') {
            Some((name, rest)) => {
                let inner = rest
                    .strip_suffix('}')
                    .unwrap_or_else(|| panic!("unclosed labels: {line}"));
                (name, parse_labels(inner, line))
            }
            None => (series, Vec::new()),
        };
        assert!(is_metric_name(name), "{line}");
        let (family, kind) = open
            .as_ref()
            .unwrap_or_else(|| panic!("sample before any TYPE: {line}"));
        match kind.as_str() {
            "histogram" => {
                let suffix = name
                    .strip_prefix(family.as_str())
                    .unwrap_or_else(|| panic!("sample outside {family}: {line}"));
                let mut key: Vec<_> = labels.iter().filter(|(k, _)| k != "le").cloned().collect();
                key.sort();
                let entry = histograms.entry(format!("{family}{key:?}")).or_insert((
                    Vec::new(),
                    None,
                    None,
                ));
                match suffix {
                    "_bucket" => {
                        let le = &labels
                            .iter()
                            .find(|(k, _)| k == "le")
                            .unwrap_or_else(|| panic!("bucket without le: {line}"))
                            .1;
                        let le = if le == "+Inf" {
                            f64::INFINITY
                        } else {
                            le.parse().unwrap_or_else(|_| panic!("le {le}: {line}"))
                        };
                        assert!(entry.1.is_none(), "bucket after _sum: {line}");
                        entry.0.push((le, value));
                    }
                    "_sum" => entry.1 = Some(value),
                    "_count" => entry.2 = Some(value),
                    _ => panic!("not a histogram sample of {family}: {line}"),
                }
            }
            _ => {
                assert_eq!(name, family, "sample outside its family: {line}");
                assert!(labels.iter().all(|(k, _)| k != "le"), "{line}");
                if kind == "counter" {
                    assert!(value >= 0.0, "{line}");
                }
            }
        }
        assert!(
            scrape.samples.insert(series.to_owned(), value).is_none(),
            "series twice: {line}"
        );
    }
    assert!(pending_help.is_none(), "HELP without TYPE at the end");
    for (series, (buckets, sum, count)) in &histograms {
        assert!(!buckets.is_empty(), "{series}: no buckets");
        for pair in buckets.windows(2) {
            assert!(pair[0].0 < pair[1].0, "{series}: le not ascending");
            assert!(pair[0].1 <= pair[1].1, "{series}: buckets not cumulative");
        }
        let last = buckets.last().unwrap();
        assert_eq!(last.0, f64::INFINITY, "{series}: no +Inf bucket");
        assert_eq!(Some(last.1), *count, "{series}: +Inf != _count");
        assert!(sum.is_some(), "{series}: no _sum");
    }
    scrape
}

async fn get_metrics(base: &str, auth: Option<&str>) -> reqwest::Response {
    let mut request = http().get(format!("{base}/metrics"));
    if let Some(value) = auth {
        request = request.header("authorization", value);
    }
    request.send().await.unwrap()
}

async fn scrape(base: &str) -> Scrape {
    let response = get_metrics(base, Some(&format!("Bearer {OPERATOR_TOKEN}"))).await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    parse(&response.text().await.unwrap())
}

/// The parser itself rejects what Prometheus rejects (negative cases of [`parse`]).
#[test]
fn the_strict_parser_rejects_malformed_expositions() {
    let good = "# HELP a_total A.\n# TYPE a_total counter\na_total{x=\"1\"} 2\n";
    assert_eq!(parse(good).value("a_total{x=\"1\"}"), 2.0);
    for bad in [
        "a_total 1\n",
        "# TYPE a_total counter\na_total 1\n",
        "# HELP a_total A.\n# TYPE a_total counter\na_total{x=\"1} 2\n",
        "# HELP a_total A.\n# TYPE a_total counter\na_total{x=\"1\",x=\"2\"} 2\n",
        "# HELP a_total A.\n# TYPE a_total counter\na_total 1\na_total 1\n",
        "# HELP a_total A.\n# TYPE a_total counter\nb_total 1\n",
        "# HELP a A.\n# TYPE a counter\na 1\n",
        "# HELP h A.\n# TYPE h histogram\nh_bucket{le=\"1\"} 2\nh_bucket{le=\"+Inf\"} 1\nh_sum 1\nh_count 1\n",
        "# HELP h A.\n# TYPE h histogram\nh_bucket{le=\"1\"} 1\nh_bucket{le=\"+Inf\"} 1\nh_sum 1\nh_count 2\n",
        "# HELP h A.\n# TYPE h histogram\nh_bucket{le=\"1\"} 1\nh_sum 1\nh_count 1\n",
        "# HELP a_total A.\n# TYPE a_total counter\na_total 1",
    ] {
        let result = std::panic::catch_unwind(|| parse(bad));
        assert!(result.is_err(), "accepted {bad:?}");
    }
}

/// OH-05: without `[auth] operator_token_env` the route does not exist (404), like every operator route.
#[tokio::test(flavor = "multi_thread")]
async fn metrics_without_an_operator_token_is_404() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    for auth in [
        None,
        Some("Bearer secret-a"),
        Some("Bearer secret-operator"),
    ] {
        let response = get_metrics(&h.base, auth).await;
        assert_eq!(response.status(), 404, "{auth:?}");
    }
}

/// OH-05: operator token only: none or a wrong one is 401, a client scope's token 403.
#[tokio::test(flavor = "multi_thread")]
async fn metrics_is_operator_only() {
    let h = Harness::start(MockConfig::default(), &[OPERATOR_TWEAK]).await;
    for auth in [
        None,
        Some("Bearer wrong"),
        Some("Bearer "),
        Some("bearer secret-operator"),
    ] {
        let response = get_metrics(&h.base, auth).await;
        assert_eq!(response.status(), 401, "{auth:?}");
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "unauthorized", "{body}");
    }
    let response = get_metrics(&h.base, Some(&format!("Bearer {TOKEN_A}"))).await;
    assert_eq!(response.status(), 403);
    scrape(&h.base).await;
}

async fn request_status(base: &str, rid: &str) -> Value {
    let response = http()
        .get(format!("{base}/v1/cere/requests/{rid}"))
        .bearer_auth(TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    response.json().await.unwrap()
}

/// Sends `body` as scope `a`, reads the whole response; returns the request id.
async fn finish(base: &str, body: &Value) -> String {
    let response = http()
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN_A)
        .json(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    response.bytes().await.unwrap();
    rid
}

/// OH-05: the exposition holds exactly the `METRIC_NAMES` families, and a unary plus a streaming request move the
/// request, duration, token and TTFT counters by their reported amounts; refusals and unmatched paths count too.
#[tokio::test(flavor = "multi_thread")]
async fn counters_move_with_requests() {
    let h = Harness::start(MockConfig::default(), &[OPERATOR_TWEAK]).await;
    let before = scrape(&h.base).await;
    // Without `[[telemetry.clients]]` the roadmap v0.7 A2 families are absent (the exposition is byte-identical).
    let without_clients: Vec<&str> = METRIC_NAMES
        .into_iter()
        .filter(|n| !n.starts_with("cere_client_"))
        .collect();
    assert_eq!(before.order, without_clients, "families and their order");
    assert_eq!(before.families["cere_ttft_seconds"], "histogram");
    assert_eq!(
        before.families["cere_request_duration_seconds"],
        "histogram"
    );
    assert_eq!(before.value("cere_ready"), 1.0);
    let up = format!("cere_worker_up{{worker=\"{SGLANG_WORKER}\"}}");
    assert_eq!(before.series_of("cere_worker_up"), vec![(up.as_str(), 1.0)]);
    // An SGLang worker has no native state: no prefix, respawn or poisoned series.
    for family in [
        "cere_prefix_cache_hits_total",
        "cere_sidecar_respawns_total",
        "cere_device_poisoned",
    ] {
        assert!(before.series_of(family).is_empty(), "{family}");
    }
    assert_eq!(before.value("cere_ledger_capacity"), 10_000.0);
    assert_eq!(before.value("cere_ledger_evictions_total"), 0.0);

    let unary = finish(&h.base, &chat_body(false)).await;
    let stream = finish(&h.base, &chat_body(true)).await;
    let refused = http()
        .post(h.url("/v1/chat/completions"))
        .json(&chat_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 401);
    let unmatched = http().get(h.url("/no/such/route")).send().await.unwrap();
    assert_eq!(unmatched.status(), 404);

    let after = scrape(&h.base).await;
    let chat = "route=\"/v1/chat/completions\"";
    let moved = |series: &str| after.counter(series) - before.counter(series);
    assert_eq!(
        moved(&format!("cere_requests_total{{{chat},status=\"200\"}}")),
        2.0
    );
    assert_eq!(
        moved(&format!("cere_requests_total{{{chat},status=\"401\"}}")),
        1.0
    );
    assert_eq!(
        moved("cere_requests_total{route=\"unmatched\",status=\"404\"}"),
        1.0
    );
    assert_eq!(
        moved(&format!("cere_request_duration_seconds_count{{{chat}}}")),
        3.0
    );
    // The first scrape is counted once its response ended (the second one is still being answered).
    assert_eq!(
        after.value("cere_requests_total{route=\"/metrics\",status=\"200\"}"),
        1.0
    );

    let mut prompt = 0;
    let mut completion = 0;
    for rid in [&unary, &stream] {
        let status = request_status(&h.base, rid).await;
        assert_eq!(status["state"], "completed", "{status}");
        prompt += status["usage"]["prompt_tokens"].as_u64().unwrap();
        completion += status["usage"]["completion_tokens"].as_u64().unwrap();
    }
    assert!(prompt > 0 && completion > 0);
    assert_eq!(moved("cere_tokens_total{kind=\"input\"}"), prompt as f64);
    assert_eq!(
        moved("cere_tokens_total{kind=\"output\"}"),
        completion as f64
    );
    assert_eq!(moved("cere_tokens_total{kind=\"cached\"}"), 0.0);
    // Only the stream has a first visible output.
    assert_eq!(moved("cere_ttft_seconds_count"), 1.0);
    assert_eq!(after.value("cere_ledger_records{state=\"live\"}"), 0.0);
    assert!(after.value("cere_ledger_records{state=\"terminal\"}") >= 2.0);
}

/// The backend body without its per-request `rid`.
fn without_rid(bytes: &[u8]) -> Value {
    let mut body: Value = serde_json::from_slice(bytes).unwrap();
    body.as_object_mut().unwrap().remove("rid");
    body
}

/// ApiLenient AL-3 (D4, D5): an inert chat field is dropped before the backend and reported by
/// `X-Cere-Ignored-Params`, `cere.ignored_params` (unary and the `accepted` chunk) and
/// `cere_request_params_total`; an `Idempotency-Key` replay that differs only in it replays.
#[tokio::test(flavor = "multi_thread")]
async fn inert_fields_are_reported_and_counted() {
    let mock = MockConfig {
        record_bodies: true,
        ..MockConfig::default()
    };
    let h = Harness::start(mock, &[OPERATOR_TWEAK]).await;
    let before = scrape(&h.base).await;

    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(!r.headers().contains_key("x-cere-ignored-params"));
    let plain: Value = r.json().await.unwrap();
    assert!(plain["cere"].get("ignored_params").is_none(), "{plain}");
    let plain_sent = without_rid(h.mock.received_bodies().last().unwrap());

    let mut body = chat_body(false);
    body["user"] = json!("u-1");
    body["metadata"] = json!({"k": "v"});
    let r = h
        .chat(&body)
        .header("Idempotency-Key", "lenient-1")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["x-cere-ignored-params"], "metadata,user");
    let first: Value = r.json().await.unwrap();
    assert_eq!(first["cere"]["ignored_params"], json!(["metadata", "user"]));
    let sent = h.mock.received_bodies();
    assert_eq!(
        without_rid(sent.last().unwrap()),
        plain_sent,
        "backend body"
    );

    // D5: a replay that differs only in an inert field replays the first response.
    body["user"] = json!("u-2");
    let r = h
        .chat(&body)
        .header("Idempotency-Key", "lenient-1")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap(), first);
    assert_eq!(
        h.mock.received_bodies().len(),
        sent.len(),
        "no second backend call"
    );

    let mut stream = chat_body(true);
    stream["user"] = json!("u-3");
    let r = h.chat(&stream).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["x-cere-ignored-params"], "user");
    let text = r.text().await.unwrap();
    let accepted: Value =
        serde_json::from_str(text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap()).unwrap();
    assert_eq!(accepted["cere"]["event"], "accepted", "{accepted}");
    assert_eq!(accepted["cere"]["ignored_params"], json!(["user"]));

    let after = scrape(&h.base).await;
    let moved = |param: &str| {
        let series = format!(
            "cere_request_params_total{{surface=\"chat\",param=\"{param}\",action=\"ignored\"}}"
        );
        after.counter(&series) - before.counter(&series)
    };
    assert_eq!(moved("user"), 3.0);
    assert_eq!(moved("metadata"), 2.0);
}

/// The `cered access` lines of `stderr`, parsed.
fn access_lines(stderr: &str) -> Vec<Value> {
    stderr
        .lines()
        .filter_map(|l| l.strip_prefix("cered access "))
        .map(|json| serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}: {json}")))
        .collect()
}

/// OH-06: one `cered access` line per finished request with the request id, matched route, status, ledger state,
/// token counts, TTFT and total time; never the prompt, the output or a token; none for the probes and the scrape.
#[tokio::test(flavor = "multi_thread")]
async fn access_line_per_request_without_content() {
    const MARKER: &str = "zebra-quartz-prompt-marker";
    let mock = MockSglang::start(MockConfig::default()).await.unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[OPERATOR_TWEAK],
    ));
    let mut cered = Cered::spawn(&config.path, &[("TOK_OP", Some(OPERATOR_TOKEN))])
        .await
        .unwrap();
    cered.wait_ready().await;
    let mut body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": MARKER}],
        "max_completion_tokens": 8,
    });
    let unary = finish(&cered.base, &body).await;
    body["stream"] = true.into();
    let stream = finish(&cered.base, &body).await;
    let refused = http()
        .post(cered.url("/v1/chat/completions"))
        .bearer_auth("not-a-token")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 401);
    scrape(&cered.base).await;
    let statuses = [
        request_status(&cered.base, &unary).await,
        request_status(&cered.base, &stream).await,
    ];

    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    let lines = access_lines(&stderr);
    let keys: BTreeSet<&str> = [
        "request_id",
        "method",
        "route",
        "status",
        "state",
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "ttft_ms",
        "total_ms",
    ]
    .into();
    for line in &lines {
        let found: BTreeSet<&str> = line
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(found, keys, "{line}");
        assert!(line["total_ms"].is_u64(), "{line}");
        for route in ["/healthz", "/readyz", "/metrics"] {
            assert_ne!(line["route"], route, "{line}");
        }
    }
    for (rid, status) in [&unary, &stream].into_iter().zip(&statuses) {
        let mine: Vec<&Value> = lines.iter().filter(|l| l["request_id"] == *rid).collect();
        assert_eq!(mine.len(), 1, "{lines:?}");
        let line = mine[0];
        assert_eq!(line["method"], "POST");
        assert_eq!(line["route"], "/v1/chat/completions");
        assert_eq!(line["status"], 200);
        assert_eq!(line["state"], "completed");
        assert_eq!(line["input_tokens"], status["usage"]["prompt_tokens"]);
        assert_eq!(line["cached_tokens"], status["usage"]["cached_tokens"]);
        assert_eq!(line["output_tokens"], status["usage"]["completion_tokens"]);
        assert_eq!(line["ttft_ms"], status["timings"]["first_visible_ms"]);
    }
    assert!(
        statuses[1]["timings"]["first_visible_ms"].is_u64(),
        "the stream had a first visible output: {}",
        statuses[1]
    );
    let refusals: Vec<&Value> = lines.iter().filter(|l| l["status"] == 401).collect();
    assert_eq!(refusals.len(), 1, "{lines:?}");
    assert_eq!(refusals[0]["request_id"], Value::Null);
    assert_eq!(refusals[0]["state"], Value::Null);
    // The two status reads are requests too (no request id: they answer about one, they are not one).
    let reads: Vec<&Value> = lines
        .iter()
        .filter(|l| l["route"] == "/v1/cere/requests/{id}")
        .collect();
    assert_eq!(reads.len(), 2, "{lines:?}");
    assert!(reads
        .iter()
        .all(|l| l["method"] == "GET" && l["status"] == 200));
    assert_eq!(lines.len(), 5, "{lines:?}");
    let access: String = stderr
        .lines()
        .filter(|l| l.starts_with("cered access "))
        .collect();
    for secret in [MARKER, TOKEN_A, OPERATOR_TOKEN, "not-a-token", "Bearer"] {
        assert!(!access.contains(secret), "{secret} in {access}");
    }
}

/// Roadmap v0.7 A2: scope `a`'s rules label a request by whether it sent `X-Cere-Priority`, a named client
/// (`X-Cere-Client`) wins over them, a name no rule of the scope holds is ignored, and scope `b` (no rule) is
/// `other`. The labels reach the `cered request` and `cered access` lines and the `cere_client_*` families.
#[tokio::test(flavor = "multi_thread")]
async fn client_labels_in_lines_and_metrics() {
    const CLIENTS: (&str, &str) = (
        "raw_content = false\n",
        "raw_content = false\n\n[[telemetry.clients]]\nclient = \"coding-agent\"\nworkload = \"interactive\"\n\
         scope = \"a\"\npriority_header = true\n\n[[telemetry.clients]]\nclient = \"eval\"\n\
         workload = \"batch-eval\"\nscope = \"a\"\npriority_header = false\n\n[[telemetry.clients]]\n\
         client = \"probe\"\nworkload = \"probe\"\nscope = \"a\"\npriority_header = true\n",
    );
    let mock = MockSglang::start(MockConfig::default()).await.unwrap();
    let state = StateDir::new();
    let config = ConfigFile::new(&config_text(
        &mock.base_url(),
        state.path(),
        free_port(),
        &[OPERATOR_TWEAK, CLIENTS],
    ));
    let mut cered = Cered::spawn(&config.path, &[("TOK_OP", Some(OPERATOR_TOKEN))])
        .await
        .unwrap();
    cered.wait_ready().await;
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 8,
    });
    let send = |token: &'static str, headers: &'static [(&'static str, &'static str)]| {
        let mut request = http()
            .post(cered.url("/v1/chat/completions"))
            .bearer_auth(token)
            .json(&body);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        async move {
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), 200);
            let rid = response.headers()["x-cere-request-id"]
                .to_str()
                .unwrap()
                .to_owned();
            response.bytes().await.unwrap();
            rid
        }
    };
    let cases = [
        (
            send(TOKEN_A, &[("x-cere-priority", "high")]).await,
            "a",
            "coding-agent",
            "interactive",
        ),
        (send(TOKEN_A, &[]).await, "a", "eval", "batch-eval"),
        (send(TOKEN_A, &[]).await, "a", "eval", "batch-eval"),
        (
            send(
                TOKEN_A,
                &[("x-cere-priority", "low"), ("x-cere-client", "eval")],
            )
            .await,
            "a",
            "eval",
            "batch-eval",
        ),
        (
            send(TOKEN_A, &[("x-cere-client", "nobody")]).await,
            "a",
            "eval",
            "batch-eval",
        ),
        (
            send(TOKEN_A, &[("x-cere-client", "probe")]).await,
            "a",
            "probe",
            "probe",
        ),
        (
            send(TOKEN_B, &[("x-cere-client", "eval")]).await,
            "b",
            "other",
            "other",
        ),
    ];
    let mut tokens: BTreeMap<(&str, &str, &str), [f64; 3]> = BTreeMap::new();
    for (rid, scope, client, workload) in &cases {
        let line = access_line_of(&cered, rid).await;
        assert_eq!(
            (&line["scope"], &line["client"], &line["workload"]),
            (&json!(scope), &json!(client), &json!(workload)),
            "{line}"
        );
        let t = tokens.entry((*scope, *client, *workload)).or_default();
        for (sum, kind) in t
            .iter_mut()
            .zip(["input_tokens", "cached_tokens", "output_tokens"])
        {
            *sum += line[kind].as_f64().unwrap_or(0.0);
        }
    }
    let metrics = scrape(&cered.base).await;
    assert_eq!(metrics.order, METRIC_NAMES, "families and their order");
    let mut requests: BTreeMap<String, f64> = BTreeMap::new();
    for (_, scope, client, workload) in &cases {
        *requests
            .entry(format!(
                "cere_client_requests_total{{scope=\"{scope}\",client=\"{client}\",workload=\"{workload}\",worker=\"gpu1\",state=\"completed\"}}"
            ))
            .or_default() += 1.0;
    }
    let found: BTreeMap<String, f64> = metrics
        .series_of("cere_client_requests_total")
        .into_iter()
        .map(|(s, v)| (s.to_owned(), v))
        .collect();
    assert_eq!(found, requests);
    for ((scope, client, workload), sums) in &tokens {
        for (sum, kind) in sums.iter().zip(["input", "cached", "output"]) {
            assert_eq!(
                metrics.counter(&format!(
                    "cere_client_tokens_total{{scope=\"{scope}\",client=\"{client}\",workload=\"{workload}\",worker=\"gpu1\",kind=\"{kind}\"}}"
                )),
                *sum,
                "{scope} {client} {kind}"
            );
        }
        assert!(
            metrics.value(&format!(
                "cere_client_request_seconds_total{{scope=\"{scope}\",client=\"{client}\",workload=\"{workload}\",worker=\"gpu1\"}}"
            )) >= 0.0
        );
    }
    assert_eq!(
        metrics.series_of("cere_client_request_seconds_total").len(),
        4
    );

    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(15)).await;
    assert_eq!(code, Some(0), "{stderr}");
    for (rid, _, client, workload) in &cases {
        let tail = format!(" client={client} workload={workload}");
        let request: Vec<&str> = stderr
            .lines()
            .filter(|l| l.starts_with(&format!("cered request rid={rid} ")))
            .collect();
        assert_eq!(request.len(), 1, "{stderr}");
        assert!(request[0].ends_with(&tail), "{}", request[0]);
    }
}

/// The access line of request `rid` once `cered` printed it (it waits for the request's terminal).
async fn access_line_of(cered: &Cered, rid: &str) -> Value {
    for _ in 0..400 {
        if let Some(line) = access_lines(&cered.stderr_so_far())
            .into_iter()
            .find(|l| l["request_id"] == rid)
        {
            return line;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("no access line for {rid}: {}", cered.stderr_so_far());
}

/// The rids of the `cered request` lines of `stderr` with `state=cancelled reason=client_disconnected stream=<s>`.
fn disconnected_rids(stderr: &str, stream: bool) -> Vec<String> {
    let tail = format!(" state=cancelled reason=client_disconnected stream={stream} ");
    stderr
        .lines()
        .filter(|l| l.starts_with("cered request rid=") && l.contains(&tail))
        .map(|l| {
            l["cered request rid=".len()..]
                .split(' ')
                .next()
                .unwrap()
                .to_owned()
        })
        .collect()
}

/// A client that leaves mid-stream (after the 200 headers and the first tokens) or before a unary response: the
/// access line carries the admitted request id, status 499 and the ledger's final state (`cancelled`, written once
/// the executor settled the disconnect), and `cere_requests_total` counts both under 499. The toy holds each
/// generation after its 2nd token until cancelled.
#[tokio::test]
async fn a_client_that_leaves_is_logged_with_the_final_state() {
    let native = Native::ready(Case {
        device_id: "toy,hold_after=2".to_owned(),
        ..Case::matching()
    })
    .await;
    // The toy's output for this prompt (716 tokens ending in '1') has no stop ID in its first tokens.
    let mut content = "cere-native-sentinel-prompt".to_owned();
    while content.len() < 714 {
        content.push('x');
    }
    content.push('1');
    let body = |stream: bool| {
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": stream,
        })
    };
    let before = scrape(&native.cered.base).await;

    let response = native.chat(&body(true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let streamed = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut reader = SseReader::new(response);
    loop {
        match reader.next().await {
            Some(Event::Data(v)) if v.pointer("/choices/0/delta/content") == Some(&json!("ab")) => {
                break
            }
            Some(Event::Data(v)) => assert!(v.get("error").is_none(), "{v}"),
            other => panic!("the stream ended before the hold: {other:?}"),
        }
    }
    drop(reader);
    let line = access_line_of(&native.cered, &streamed).await;
    assert_eq!(line["status"], 499, "{line}");
    assert_eq!(line["state"], "cancelled", "{line}");
    assert_eq!(line["route"], "/v1/chat/completions", "{line}");
    assert!(
        line["ttft_ms"].is_u64(),
        "the stream had visible output: {line}"
    );

    // The unary request is held before its response: the client gives up without a request id.
    let gave_up = native
        .chat(&body(false))
        .timeout(Duration::from_millis(1_500))
        .send()
        .await;
    assert!(gave_up.unwrap_err().is_timeout());
    let mut unary = Vec::new();
    for _ in 0..400 {
        unary = disconnected_rids(&native.cered.stderr_so_far(), false);
        if !unary.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(unary.len(), 1, "{}", native.cered.stderr_so_far());
    let line = access_line_of(&native.cered, &unary[0]).await;
    assert_eq!(line["status"], 499, "{line}");
    assert_eq!(line["state"], "cancelled", "{line}");
    assert_eq!(
        disconnected_rids(&native.cered.stderr_so_far(), true),
        vec![streamed.clone()]
    );

    let after = scrape(&native.cered.base).await;
    let chat = |status: &str| {
        format!("cere_requests_total{{route=\"/v1/chat/completions\",status=\"{status}\"}}")
    };
    assert_eq!(
        after.counter(&chat("499")) - before.counter(&chat("499")),
        2.0
    );
    assert_eq!(after.counter(&chat("200")), before.counter(&chat("200")));
    // Each request has exactly one access line.
    let lines = access_lines(&native.cered.stderr_so_far());
    for rid in [&streamed, &unary[0]] {
        let mine = lines.iter().filter(|l| l["request_id"] == **rid).count();
        assert_eq!(mine, 1, "{lines:?}");
    }
}

/// OH-05: a native worker's prefix cache counters (the values `GET /v1/cere/prefix_cache` reports), its sidecar
/// respawns, `cere_worker_up` and `cere_device_poisoned` 0 while healthy.
#[tokio::test]
async fn native_worker_series() {
    let native = Native::ready(Case {
        prefix_state_slots: Some(16),
        ..Case::matching()
    })
    .await;
    // `two_messages(9_000, 100, '1', 4)` of `server_native.rs`: long enough to capture, the second send hits.
    let mut history = "cere-native-sentinel-prompt".to_owned();
    while history.len() < 8_999 {
        history.push('x');
    }
    let body = json!({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": history},
            {"role": "user", "content": format!("1{}", "q".repeat(98))},
        ],
        "max_tokens": 4,
        "temperature": 0,
    });
    finish(&native.cered.base, &body).await;
    finish(&native.cered.base, &body).await;
    let s = scrape(&native.cered.base).await;
    let response = http()
        .get(native.cered.url("/v1/cere/prefix_cache"))
        .bearer_auth(OPERATOR_TOKEN)
        .send()
        .await
        .unwrap();
    let body: Value = response.json().await.unwrap();
    let counters = &body["workers"][WORKER]["prefix_cache"];
    let worker = format!("{{worker=\"{WORKER}\"}}");
    for key in [
        "hits",
        "captures",
        "capture_skipped",
        "capture_page_mismatch",
        "inserts",
        "state_evictions",
        "leaf_evictions",
        "host_spills",
        "host_loads",
        "host_drops",
        "alt_edges",
    ] {
        assert_eq!(
            s.value(&format!("cere_prefix_cache_{key}_total{worker}")),
            counters[key].as_u64().unwrap_or_else(|| panic!("{body}")) as f64,
            "{key}"
        );
    }
    // PT0a: capture_skipped by site (the sites sum to the total) and the drafting lookups' lost hit tokens.
    let sites = &counters["capture_skipped_by_site"];
    let mut by_site = 0.0;
    for site in ["prefill_slot", "decode_track", "end_pair"] {
        let v = s.value(&format!(
            "cere_prefix_cache_capture_skipped_by_site_total{{worker=\"{WORKER}\",site=\"{site}\"}}"
        ));
        assert_eq!(
            v,
            sites[site].as_u64().unwrap_or_else(|| panic!("{body}")) as f64,
            "{site}"
        );
        by_site += v;
    }
    assert_eq!(
        by_site,
        s.value(&format!("cere_prefix_cache_capture_skipped_total{worker}"))
    );
    for key in ["hit_shortfall_tokens", "hit_shortfall_lookups"] {
        assert_eq!(
            s.value(&format!("cere_prefix_cache_{key}_total{worker}")),
            counters[key].as_u64().unwrap_or_else(|| panic!("{body}")) as f64,
            "{key}"
        );
    }
    assert!(s.value(&format!("cere_prefix_cache_inserts_total{worker}")) >= 1.0);
    assert_eq!(
        s.value(&format!("cere_prefix_cache_hits_total{worker}")),
        1.0
    );
    assert_eq!(
        s.value(&format!("cere_sidecar_respawns_total{worker}")),
        0.0
    );
    assert_eq!(s.value(&format!("cere_worker_up{worker}")), 1.0);
    assert_eq!(s.value(&format!("cere_device_poisoned{worker}")), 0.0);
    // A second scrape of unchanged counters adds nothing (the read is delta-accumulated).
    let again = scrape(&native.cered.base).await;
    assert_eq!(
        again.value(&format!("cere_prefix_cache_inserts_total{worker}")),
        s.value(&format!("cere_prefix_cache_inserts_total{worker}"))
    );
}
