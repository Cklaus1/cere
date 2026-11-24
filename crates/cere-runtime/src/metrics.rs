//! ops-hardening design note OH-05/OH-06: the Prometheus exposition of `GET /metrics` and the access log.
//!
//! The text exposition (format 0.0.4) is written here by hand: no metrics crate is in the lockfile. Every HTTP
//! request passes [`track`] (a router layer): it counts `cere_requests_total{route,status}`, observes
//! `cere_request_duration_seconds{route}` when the response body has ended (an SSE stream at its last event) and
//! prints one `cered access {...}` line ([`access_line`]) per request except the probe and scrape routes. A
//! request whose client left before its response ended is counted and logged as [`CLIENT_CLOSED`]; when its
//! ledger record is not terminal yet (the executor settles the disconnect after), its line waits for the
//! request's terminal ([`Metrics::deferred_access`]). Each request terminal adds its token counts and its time
//! to first visible output ([`Metrics::observe_terminal`]).
//! The per-worker native counters (`cere_prefix_cache_*`, `cere_sidecar_respawns_total`) restart with each load
//! of a model; [`Accumulated`] folds them into counters that only ever grow for the life of the process.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::{MatchedPath, Request, State};
use axum::http::{header, HeaderValue, Method};
use axum::middleware::Next;
use axum::response::Response;
use futures_util::Stream;
use serde_json::json;

use crate::app::{App, Backend};
use crate::config::ClientRule;
use crate::labels::{ClientRules, RequestLabels};
use crate::ledger::RequestStatus;
use crate::native::ffi::{PrefixCaptureStats, PrefixStats};
use crate::server::REQUEST_ID_HEADER;

/// The `Content-Type` of the text exposition.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Every metric family `GET /metrics` exports, in exposition order. The alert rules
/// (`deploy/monitoring/cere-alerts.yml`) may reference only these (`tests/test_monitoring_alerts.py`).
pub const METRIC_NAMES: [&str; 29] = [
    "cere_requests_total",
    "cere_request_duration_seconds",
    "cere_ttft_seconds",
    "cere_tokens_total",
    "cere_client_requests_total",
    "cere_client_tokens_total",
    "cere_client_request_seconds_total",
    "cere_request_params_total",
    "cere_prefix_cache_hits_total",
    "cere_prefix_cache_captures_total",
    "cere_prefix_cache_capture_skipped_total",
    "cere_prefix_cache_capture_page_mismatch_total",
    "cere_prefix_cache_inserts_total",
    "cere_prefix_cache_state_evictions_total",
    "cere_prefix_cache_leaf_evictions_total",
    "cere_prefix_cache_host_spills_total",
    "cere_prefix_cache_host_loads_total",
    "cere_prefix_cache_host_drops_total",
    "cere_prefix_cache_alt_edges_total",
    "cere_prefix_cache_capture_skipped_by_site_total",
    "cere_prefix_cache_hit_shortfall_tokens_total",
    "cere_prefix_cache_hit_shortfall_lookups_total",
    "cere_ledger_records",
    "cere_ledger_capacity",
    "cere_ledger_evictions_total",
    "cere_sidecar_respawns_total",
    "cere_worker_up",
    "cere_ready",
    "cere_device_poisoned",
];

/// The route label of a request no route matched.
const UNMATCHED_ROUTE: &str = "unmatched";

/// Routes without an access line: the probes and the scrape, polled every few seconds.
const UNLOGGED_ROUTES: [&str; 3] = ["/healthz", "/readyz", "/metrics"];

/// The status of a request whose client went away before the response ended (nginx's 499): before any response,
/// or during a stream whose `200` headers were already sent.
const CLIENT_CLOSED: u16 = 499;

tokio::task_local! {
    /// The request id the ledger admitted for the request [`track`] is running ([`note_admitted`]).
    static ADMITTED: Arc<OnceLock<String>>;
}

/// Records the request id the ledger just admitted for the HTTP request in flight, so its access line names it
/// even when the client leaves before a response carries it. A no-op outside [`track`].
pub(crate) fn note_admitted(rid: &str) {
    let _ = ADMITTED.try_with(|admitted| admitted.set(rid.to_owned()));
}

/// `cere_request_duration_seconds` buckets: a probe in milliseconds to a long generation in minutes.
const DURATION_BUCKETS: [f64; 16] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

/// `cere_ttft_seconds` buckets: a cached prefix to a 262K-token prefill.
const TTFT_BUCKETS: [f64; 13] = [
    0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
];

/// The prefix cache counters of `cere_model_prefix_stats`, with their family.
type PrefixField = (&'static str, fn(&PrefixStats) -> u64, &'static str);
const PREFIX_FAMILIES: [PrefixField; 11] = [
    (
        "cere_prefix_cache_hits_total",
        |s| s.hits,
        "Prefix cache hits.",
    ),
    (
        "cere_prefix_cache_captures_total",
        |s| s.captures,
        "Prefix states captured.",
    ),
    (
        "cere_prefix_cache_capture_skipped_total",
        |s| s.capture_skipped,
        "Prefix captures skipped.",
    ),
    (
        "cere_prefix_cache_capture_page_mismatch_total",
        |s| s.capture_page_mismatch,
        "Prefix captures refused: the tree holds other pages for equal tokens.",
    ),
    (
        "cere_prefix_cache_inserts_total",
        |s| s.inserts,
        "Prefix tree inserts.",
    ),
    (
        "cere_prefix_cache_state_evictions_total",
        |s| s.state_evictions,
        "Prefix states evicted.",
    ),
    (
        "cere_prefix_cache_leaf_evictions_total",
        |s| s.leaf_evictions,
        "Prefix tree leaves evicted.",
    ),
    (
        "cere_prefix_cache_host_spills_total",
        |s| s.host_spills,
        "Prefix states spilled to host memory.",
    ),
    (
        "cere_prefix_cache_host_loads_total",
        |s| s.host_loads,
        "Prefix states loaded back from host memory.",
    ),
    (
        "cere_prefix_cache_host_drops_total",
        |s| s.host_drops,
        "Host-spilled prefix states dropped.",
    ),
    (
        "cere_prefix_cache_alt_edges_total",
        |s| s.alt_edges,
        "Alternative prefix edges added (prefix_alt_edges only).",
    ),
];

/// prefix-token design note PT0a: `cere_prefix_cache_capture_skipped_total` split by `site` (the sum of the sites),
/// from `cere_model_prefix_capture_stats`.
const SKIPPED_BY_SITE: &str = "cere_prefix_cache_capture_skipped_by_site_total";

/// The capture-skip sites: (accumulator key, `site` label, field).
type SiteField = (&'static str, &'static str, fn(&PrefixCaptureStats) -> u64);
const SKIP_SITES: [SiteField; 3] = [
    (
        "cere_prefix_cache_capture_skipped_by_site_total/prefill_slot",
        "prefill_slot",
        |s| s.skipped_prefill_slot,
    ),
    (
        "cere_prefix_cache_capture_skipped_by_site_total/decode_track",
        "decode_track",
        |s| s.skipped_decode_track,
    ),
    (
        "cere_prefix_cache_capture_skipped_by_site_total/end_pair",
        "end_pair",
        |s| s.skipped_end_pair,
    ),
];

/// The lost hit counters of `cere_model_prefix_capture_stats`, with their family.
type ShortfallField = (&'static str, fn(&PrefixCaptureStats) -> u64, &'static str);
const SHORTFALL_FAMILIES: [ShortfallField; 2] = [
    (
        "cere_prefix_cache_hit_shortfall_tokens_total",
        |s| s.hit_shortfall_tokens,
        "Prompt tokens drafting lookups recomputed because no draft state sat at the deepest prefill-chunk multiple \
         their page match passed.",
    ),
    (
        "cere_prefix_cache_hit_shortfall_lookups_total",
        |s| s.hit_shortfall_lookups,
        "Drafting lookups that lost any tokens to a missing draft state.",
    ),
];

const SIDECAR_RESPAWNS: &str = "cere_sidecar_respawns_total";

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A histogram with fixed upper bounds; `counts[i]` holds the observations in `(bounds[i-1], bounds[i]]`.
#[derive(Debug, Clone)]
struct Histogram {
    bounds: &'static [f64],
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: vec![0; bounds.len()],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        let bucket = self.bounds.partition_point(|bound| *bound < value);
        if let Some(count) = self.counts.get_mut(bucket) {
            *count += 1;
        }
        self.sum += value;
        self.count += 1;
    }
}

/// The loads of one counter [`Accumulated`] remembers; a read of an older load counts as a new one.
const KEPT_LOADS: usize = 4;

/// A native counter that restarts at 0 with each load of the model, folded into a process-lifetime total. A load is
/// told apart by its backend `instance_id` (readiness hashes the in-process load count into it, so every reload or
/// unload/load has a new one); a value below the last one of the same load is a stale read, never a reset.
/// Concurrent scrapes apply their reads in any order: each load keeps the highest value applied, and a read adds
/// only what it exceeds that by, so the total does not depend on the order. Increments of a load after its last
/// read are lost with it. Only the last [`KEPT_LOADS`] loads are remembered: a scrape still holding a read from
/// before that many reloads would count its load again (a scrape takes milliseconds, a load far longer).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Accumulated {
    /// `(instance_id, highest value applied)`, oldest first.
    loads: Vec<(String, u64)>,
    total: u64,
}

impl Accumulated {
    fn observe(&mut self, instance: &str, raw: u64) {
        if let Some((_, last)) = self.loads.iter_mut().find(|(id, _)| id == instance) {
            self.total += raw.saturating_sub(*last);
            *last = (*last).max(raw);
            return;
        }
        self.total += raw;
        if self.loads.len() == KEPT_LOADS {
            self.loads.remove(0);
        }
        self.loads.push((instance.to_owned(), raw));
    }
}

/// One matched route's requests: counts by status and the duration histogram.
#[derive(Debug)]
struct RouteStats {
    statuses: BTreeMap<u16, u64>,
    duration: Histogram,
}

#[derive(Debug)]
struct Terminals {
    ttft: Histogram,
    input_tokens: u64,
    cached_tokens: u64,
    output_tokens: u64,
}

/// A2: one `(scope, client, workload)`'s request terminals (only with `[[telemetry.clients]]`).
#[derive(Debug, Default)]
struct ClientStats {
    /// By terminal state.
    states: BTreeMap<&'static str, u64>,
    input_tokens: u64,
    cached_tokens: u64,
    output_tokens: u64,
    /// The terminals' `total_ms` (admission to terminal), summed.
    wall_ms: u64,
}

/// `(scope, client, workload, worker)`; the worker is the one the request was last dispatched to, [`NO_WORKER`]
/// when it never was (net-teleport design note 4.4, S7).
type ClientKey = (String, Arc<str>, Arc<str>, String);

/// The `worker` label of a request terminal that was never dispatched (the request line's `worker=-`).
const NO_WORKER: &str = "-";

/// The process's metrics state (one per [`App`]).
#[derive(Debug)]
pub struct Metrics {
    /// By route label.
    http: Mutex<BTreeMap<String, RouteStats>>,
    terminals: Mutex<Terminals>,
    /// `(worker, family)` -> its accumulated native counter.
    native: Mutex<BTreeMap<(String, &'static str), Accumulated>>,
    /// The access lines that wait for their request's terminal, by request id ([`Metrics::deferred_access`]).
    deferred: Mutex<HashMap<String, Deferred>>,
    /// ApiLenient D4: `(surface, param, action)` -> requests ([`Metrics::observe_params`]).
    params: Mutex<BTreeMap<(&'static str, &'static str, &'static str), u64>>,
    /// A2: the `[[telemetry.clients]]` rules; empty → no client labels anywhere (lines and exposition unchanged).
    pub(crate) clients: ClientRules,
    /// A2: request terminals by `(scope, client, workload)` ([`Metrics::observe_terminal`]).
    by_client: Mutex<BTreeMap<ClientKey, ClientStats>>,
}

/// An access line whose client left while its request was not terminal yet.
#[derive(Debug)]
struct Deferred {
    method: Method,
    route: String,
    total_ms: u64,
}

impl Metrics {
    pub(crate) fn new(clients: &[ClientRule]) -> Self {
        Self {
            http: Mutex::new(BTreeMap::new()),
            terminals: Mutex::new(Terminals {
                ttft: Histogram::new(&TTFT_BUCKETS),
                input_tokens: 0,
                cached_tokens: 0,
                output_tokens: 0,
            }),
            native: Mutex::new(BTreeMap::new()),
            deferred: Mutex::new(HashMap::new()),
            params: Mutex::new(BTreeMap::new()),
            clients: ClientRules::new(clients),
            by_client: Mutex::new(BTreeMap::new()),
        }
    }

    fn observe_request(&self, route: &str, status: u16, seconds: f64) {
        let mut http = lock(&self.http);
        // A route's first request allocates its label; later ones only look it up.
        if !http.contains_key(route) {
            http.insert(
                route.to_owned(),
                RouteStats {
                    statuses: BTreeMap::new(),
                    duration: Histogram::new(&DURATION_BUCKETS),
                },
            );
        }
        if let Some(stats) = http.get_mut(route) {
            *stats.statuses.entry(status).or_default() += 1;
            stats.duration.observe(seconds);
        }
    }

    /// A2: the labels a line or counter shows for `status`'s request: `None` without `[[telemetry.clients]]`, else
    /// its own ([`crate::ledger::Ledger::set_labels`]) or `other` (a request replayed from the journal).
    pub(crate) fn labels_of<'a>(&'a self, status: &'a RequestStatus) -> Option<&'a RequestLabels> {
        if self.clients.is_empty() {
            return None;
        }
        Some(status.labels.as_ref().unwrap_or(self.clients.other()))
    }

    /// A request terminal (once per request, [`App::log_terminal`]): its reported token counts and its time to
    /// first visible output, when it had one; with `[[telemetry.clients]]` also its client's counters.
    pub(crate) fn observe_terminal(&self, status: &RequestStatus) {
        let usage = &status.usage;
        {
            let mut t = lock(&self.terminals);
            t.input_tokens += usage.prompt_tokens.unwrap_or(0);
            t.cached_tokens += usage.cached_tokens.unwrap_or(0);
            t.output_tokens += usage.completion_tokens.unwrap_or(0);
            if let Some(ms) = status.timings.first_visible_ms {
                t.ttft.observe(ms as f64 / 1000.0);
            }
        }
        let Some(labels) = self.labels_of(status) else {
            return;
        };
        let mut by_client = lock(&self.by_client);
        let c = by_client
            .entry((
                status.scope.clone(),
                labels.client.clone(),
                labels.workload.clone(),
                status
                    .worker_id
                    .clone()
                    .unwrap_or_else(|| NO_WORKER.to_owned()),
            ))
            .or_default();
        *c.states.entry(status.state.as_str()).or_default() += 1;
        c.input_tokens += usage.prompt_tokens.unwrap_or(0);
        c.cached_tokens += usage.cached_tokens.unwrap_or(0);
        c.output_tokens += usage.completion_tokens.unwrap_or(0);
        c.wall_ms += status.timings.total_ms.unwrap_or(0);
    }

    /// The access line [`Tracked`] deferred for `status`'s request, now terminal ([`App::log_terminal`]).
    pub(crate) fn deferred_access(&self, status: &RequestStatus) -> Option<String> {
        let deferred = lock(&self.deferred).remove(&status.rid)?;
        Some(access_line(&Access {
            request_id: Some(&status.rid),
            method: deferred.method.as_str(),
            route: &deferred.route,
            status: CLIENT_CLOSED,
            record: Some(status),
            labels: self.labels_of(status),
            total_ms: deferred.total_ms,
        }))
    }

    fn accumulate(&self, worker: &str, family: &'static str, instance: &str, raw: u64) {
        lock(&self.native)
            .entry((worker.to_owned(), family))
            .or_default()
            .observe(instance, raw);
    }

    /// ApiLenient D4: one request's mapped and ignored parameters (labels bounded by the D1 lists).
    pub(crate) fn observe_params(
        &self,
        surface: &'static str,
        mapped: &[&'static str],
        ignored: &[String],
    ) {
        if mapped.is_empty() && ignored.is_empty() {
            return;
        }
        let mut params = lock(&self.params);
        for param in mapped {
            *params.entry((surface, *param, "mapped")).or_default() += 1;
        }
        for path in ignored {
            *params
                .entry((surface, crate::validate::param_label(path), "ignored"))
                .or_default() += 1;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Exposition

/// Escapes a label value: backslash, double quote and line feed.
fn escape_label(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

/// The exposition text under construction.
struct Exposition(String);

impl Exposition {
    fn family(&mut self, name: &str, kind: &str, help: &str) {
        let _ = writeln!(self.0, "# HELP {name} {help}\n# TYPE {name} {kind}");
    }

    fn sample(&mut self, name: &str, labels: &[(&str, &str)], value: impl std::fmt::Display) {
        self.0.push_str(name);
        if !labels.is_empty() {
            self.0.push('{');
            for (i, (key, val)) in labels.iter().enumerate() {
                if i > 0 {
                    self.0.push(',');
                }
                let _ = write!(self.0, "{key}=\"{}\"", escape_label(val));
            }
            self.0.push('}');
        }
        let _ = writeln!(self.0, " {value}");
    }

    fn histogram(&mut self, name: &str, labels: &[(&str, &str)], h: &Histogram) {
        let bucket = format!("{name}_bucket");
        let mut cumulative = 0;
        for (bound, count) in h.bounds.iter().zip(&h.counts) {
            cumulative += count;
            let le = bound.to_string();
            let mut with_le = labels.to_vec();
            with_le.push(("le", &le));
            self.sample(&bucket, &with_le, cumulative);
        }
        let mut with_le = labels.to_vec();
        with_le.push(("le", "+Inf"));
        self.sample(&bucket, &with_le, h.count);
        self.sample(&format!("{name}_sum"), labels, h.sum);
        self.sample(&format!("{name}_count"), labels, h.count);
    }
}

/// One local worker's native state at a scrape.
struct NativeRead {
    worker: String,
    instance: String,
    prefix: Option<PrefixStats>,
    /// `None` also on a library without `cere_model_prefix_capture_stats`.
    sites: Option<PrefixCaptureStats>,
    respawns: u64,
    poisoned: bool,
}

/// The exposition of `GET /metrics` (operator auth is the caller's). Reads every loaded native worker's prefix
/// counters (a driver round trip each, off the async threads).
pub(crate) async fn render(app: &App) -> String {
    let mut reads = Vec::new();
    for replica in app.replicas() {
        let Backend::Native(slot) = &replica.backend else {
            continue;
        };
        let worker = replica.worker_id().to_owned();
        let faulted = slot.is_faulted();
        match slot.get() {
            Some(native) => reads.push(NativeRead {
                worker,
                instance: native.instance_id.clone(),
                respawns: native.sidecar.stats().respawns,
                poisoned: faulted || native.fault.load(Ordering::Acquire),
                prefix: crate::operator::read_prefix_stats(native.clone())
                    .await
                    .ok(),
                sites: crate::operator::read_prefix_capture_stats(native).await,
            }),
            None => reads.push(NativeRead {
                worker,
                instance: String::new(),
                prefix: None,
                sites: None,
                respawns: 0,
                poisoned: faulted,
            }),
        }
    }
    let metrics = &app.metrics;
    for read in reads.iter().filter(|r| !r.instance.is_empty()) {
        if let Some(stats) = &read.prefix {
            for (family, field, _) in &PREFIX_FAMILIES {
                metrics.accumulate(&read.worker, family, &read.instance, field(stats));
            }
        }
        if let Some(sites) = &read.sites {
            for (key, _, field) in &SKIP_SITES {
                metrics.accumulate(&read.worker, key, &read.instance, field(sites));
            }
            for (family, field, _) in &SHORTFALL_FAMILIES {
                metrics.accumulate(&read.worker, family, &read.instance, field(sites));
            }
        }
        metrics.accumulate(
            &read.worker,
            SIDECAR_RESPAWNS,
            &read.instance,
            read.respawns,
        );
    }
    let (ready, readyz) = crate::server::readyz_report(app);
    let ledger = app.ledger().ledger_stats();

    let mut out = Exposition(String::with_capacity(8192));
    {
        let http = lock(&metrics.http);
        out.family(
            "cere_requests_total",
            "counter",
            "HTTP requests by matched route and response status (499: the client left before the response ended).",
        );
        for (route, stats) in http.iter() {
            for (status, count) in &stats.statuses {
                out.sample(
                    "cere_requests_total",
                    &[("route", route), ("status", &status.to_string())],
                    count,
                );
            }
        }
        out.family(
            "cere_request_duration_seconds",
            "histogram",
            "HTTP request duration until the response body ended, by matched route.",
        );
        for (route, stats) in http.iter() {
            out.histogram(
                "cere_request_duration_seconds",
                &[("route", route)],
                &stats.duration,
            );
        }
    }
    {
        let t = lock(&metrics.terminals);
        out.family(
            "cere_ttft_seconds",
            "histogram",
            "Time from admission to the first visible output, per request terminal that had one.",
        );
        out.histogram("cere_ttft_seconds", &[], &t.ttft);
        out.family(
            "cere_tokens_total",
            "counter",
            "Reported tokens of finished requests; cached is the part of input served from a prefix cache.",
        );
        out.sample("cere_tokens_total", &[("kind", "input")], t.input_tokens);
        out.sample("cere_tokens_total", &[("kind", "cached")], t.cached_tokens);
        out.sample("cere_tokens_total", &[("kind", "output")], t.output_tokens);
    }
    if !metrics.clients.is_empty() {
        clients_exposition(&mut out, &lock(&metrics.by_client));
    }
    {
        let params = lock(&metrics.params);
        out.family(
            "cere_request_params_total",
            "counter",
            "Request parameters the request layer accepted without effect (ignored) or rewrote to their canonical field (mapped), by surface (tasks/api-lenient-spec.md D4).",
        );
        for ((surface, param, action), count) in params.iter() {
            out.sample(
                "cere_request_params_total",
                &[("surface", surface), ("param", param), ("action", action)],
                count,
            );
        }
    }
    {
        let native = lock(&metrics.native);
        for (family, _, help) in &PREFIX_FAMILIES {
            out.family(family, "counter", help);
            for ((worker, f), acc) in native.iter() {
                if f == family {
                    out.sample(family, &[("worker", worker)], acc.total);
                }
            }
        }
        out.family(
            SKIPPED_BY_SITE,
            "counter",
            "Prefix captures skipped, by site: prefill_slot (a prefill capture found no slot), decode_track (a \
             decode track capture found no slot, once per ticket), end_pair (an end-state capture found no slot pair).",
        );
        for ((worker, f), acc) in native.iter() {
            if let Some((_, site, _)) = SKIP_SITES.iter().find(|(key, _, _)| key == f) {
                out.sample(
                    SKIPPED_BY_SITE,
                    &[("worker", worker), ("site", site)],
                    acc.total,
                );
            }
        }
        for (family, _, help) in &SHORTFALL_FAMILIES {
            out.family(family, "counter", help);
            for ((worker, f), acc) in native.iter() {
                if f == family {
                    out.sample(family, &[("worker", worker)], acc.total);
                }
            }
        }
        out.family(
            "cere_ledger_records",
            "gauge",
            "Request ledger records: live (not terminal) and terminal (retained).",
        );
        out.sample("cere_ledger_records", &[("state", "live")], ledger.live);
        out.sample(
            "cere_ledger_records",
            &[("state", "terminal")],
            ledger.terminal,
        );
        out.family("cere_ledger_capacity", "gauge", "ledger.max_records.");
        out.sample("cere_ledger_capacity", &[], ledger.capacity);
        out.family(
            "cere_ledger_evictions_total",
            "counter",
            "Terminal ledger records evicted to admit a request.",
        );
        out.sample("cere_ledger_evictions_total", &[], ledger.evictions);
        out.family(
            SIDECAR_RESPAWNS,
            "counter",
            "Tokenizer sidecar child respawns per native worker.",
        );
        for ((worker, f), acc) in native.iter() {
            if *f == SIDECAR_RESPAWNS {
                out.sample(SIDECAR_RESPAWNS, &[("worker", worker)], acc.total);
            }
        }
    }
    // Team pools (net-teleport design note 4.4, S6): only with a team-pool key in the config.
    if crate::pool::configured(&app.settings().config) {
        let grants = app.admission.pool_grants();
        out.family(
            "cere_admission_pool_grants_total",
            "counter",
            "Admission grants made while both classes had a grantable waiter, by the class granted (kind contested_interactive, contested_batch), and grants owner priority gave a worker's owner ahead of DRR (kind owner_priority).",
        );
        for (kind, n) in [
            ("contested_interactive", grants.contested[0]),
            ("contested_batch", grants.contested[1]),
            ("owner_priority", grants.owner_priority),
        ] {
            out.sample("cere_admission_pool_grants_total", &[("kind", kind)], n);
        }
        out.family(
            "cere_worker_owner_only",
            "gauge",
            "1 while an owned worker serves only its owner (an owner_only window or its local switch, as of its last heartbeat), else 0.",
        );
        for (worker, _, since) in app.admission.owner_only_states().1 {
            out.sample(
                "cere_worker_owner_only",
                &[("worker", &worker)],
                u8::from(since.is_some()),
            );
        }
    }
    out.family(
        "cere_worker_up",
        "gauge",
        "1 while the worker accepts new work (its /readyz entry is ready), else 0.",
    );
    if let Some(workers) = readyz["workers"].as_object() {
        for (worker, entry) in workers {
            let up = u8::from(entry["ready"].as_bool() == Some(true));
            out.sample("cere_worker_up", &[("worker", worker)], up);
        }
    }
    out.family(
        "cere_ready",
        "gauge",
        "1 while GET /readyz answers 200, else 0.",
    );
    out.sample("cere_ready", &[], u8::from(ready));
    out.family(
        "cere_device_poisoned",
        "gauge",
        "1 after an internal executor fault on the worker's card (until a re-smoke passes; sticky once its reload failed).",
    );
    for read in &reads {
        out.sample(
            "cere_device_poisoned",
            &[("worker", &read.worker)],
            u8::from(read.poisoned),
        );
    }
    out.0
}

// ---------------------------------------------------------------------------------------------------------
// Request tracking and the access log (OH-06)

/// One finished HTTP request as the access line reports it. Never content: no prompt, output or header value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Access<'a> {
    pub request_id: Option<&'a str>,
    pub method: &'a str,
    pub route: &'a str,
    pub status: u16,
    /// The request's ledger record at the response end, when the response named one the caller's scope holds.
    pub record: Option<&'a RequestStatus>,
    /// A2: the record's client and workload ([`Metrics::labels_of`]); `None` without `[[telemetry.clients]]`.
    pub labels: Option<&'a RequestLabels>,
    pub total_ms: u64,
}

/// `cered access {json}`: `request_id`, `method`, `route` (the matched route template, never the raw path),
/// `status`, the ledger `state`, `input_tokens`, `cached_tokens`, `output_tokens`, `ttft_ms` (first visible
/// output, from admission) and `total_ms` (until the response body ended); `null` when not known. With
/// `[[telemetry.clients]]` (A2) a line with a record also has its `scope`, `client` and `workload`.
pub(crate) fn access_line(a: &Access<'_>) -> String {
    let record = a.record;
    let usage = record.map(|r| r.usage);
    let mut line = json!({
        "request_id": a.request_id,
        "method": a.method,
        "route": a.route,
        "status": a.status,
        "state": record.map(|r| r.state.as_str()),
        "input_tokens": usage.and_then(|u| u.prompt_tokens),
        "cached_tokens": usage.and_then(|u| u.cached_tokens),
        "output_tokens": usage.and_then(|u| u.completion_tokens),
        "ttft_ms": record.and_then(|r| r.timings.first_visible_ms),
        "total_ms": a.total_ms,
    });
    if let (Some(r), Some(labels), Some(map)) = (record, a.labels, line.as_object_mut()) {
        map.insert("scope".to_owned(), json!(r.scope));
        map.insert("client".to_owned(), json!(&*labels.client));
        map.insert("workload".to_owned(), json!(&*labels.workload));
    }
    format!("cered access {line}")
}

/// The A2 families: request terminals, tokens and request wall by `(scope, client, workload, worker)`.
fn clients_exposition(out: &mut Exposition, by_client: &BTreeMap<ClientKey, ClientStats>) {
    fn key((scope, client, workload, worker): &ClientKey) -> [(&'static str, &str); 4] {
        [
            ("scope", scope.as_str()),
            ("client", &**client),
            ("workload", &**workload),
            ("worker", worker.as_str()),
        ]
    }
    out.family(
        "cere_client_requests_total",
        "counter",
        "Request terminals by scope, client and workload ([[telemetry.clients]]), worker and terminal state.",
    );
    for (k, c) in by_client {
        let [scope, client, workload, worker] = key(k);
        for (state, count) in &c.states {
            out.sample(
                "cere_client_requests_total",
                &[scope, client, workload, worker, ("state", *state)],
                count,
            );
        }
    }
    out.family(
        "cere_client_tokens_total",
        "counter",
        "Reported tokens of request terminals by scope, client, workload and worker; cached is the part of input served from a prefix cache.",
    );
    for (k, c) in by_client {
        let [scope, client, workload, worker] = key(k);
        for (kind, count) in [
            ("input", c.input_tokens),
            ("cached", c.cached_tokens),
            ("output", c.output_tokens),
        ] {
            out.sample(
                "cere_client_tokens_total",
                &[scope, client, workload, worker, ("kind", kind)],
                count,
            );
        }
    }
    out.family(
        "cere_client_request_seconds_total",
        "counter",
        "Admission-to-terminal seconds of request terminals (the request line's total_ms), summed by scope, client, workload and worker.",
    );
    for (k, c) in by_client {
        out.sample(
            "cere_client_request_seconds_total",
            &key(k),
            format_args!("{}.{:03}", c.wall_ms / 1000, c.wall_ms % 1000),
        );
    }
}

/// A request in flight through [`track`]; dropping it records the request (the response body ended, or the
/// client went away first: `status` is then `None`, [`CLIENT_CLOSED`]).
struct Tracked {
    app: Arc<App>,
    started: Instant,
    method: Method,
    route: Option<MatchedPath>,
    authorization: Option<HeaderValue>,
    status: Option<u16>,
    request_id: Option<HeaderValue>,
    /// Set when the ledger admitted the request ([`note_admitted`]); the response may never carry it.
    admitted: Arc<OnceLock<String>>,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        let route = self
            .route
            .as_ref()
            .map_or(UNMATCHED_ROUTE, MatchedPath::as_str);
        let status = self.status.unwrap_or(CLIENT_CLOSED);
        self.app
            .metrics
            .observe_request(route, status, elapsed.as_secs_f64());
        if UNLOGGED_ROUTES.contains(&route) {
            return;
        }
        let request_id = self
            .request_id
            .as_ref()
            .and_then(|v| v.to_str().ok())
            .or_else(|| self.admitted.get().map(String::as_str));
        let total_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
        let scope = request_id.and_then(|_| {
            self.app
                .auth
                .authenticate(self.authorization.as_ref().map(HeaderValue::as_bytes))
        });
        let line = {
            // Read and deferred under the ledger lock: the terminal that `App::log_terminal` then logs is
            // written under it too, so the line is printed exactly once, here or there.
            let ledger = self.app.ledger();
            let record = request_id
                .zip(scope)
                .and_then(|(rid, scope)| ledger.status(scope, rid));
            match (request_id, &record) {
                (Some(rid), Some(r)) if self.status.is_none() && !r.state.is_terminal() => {
                    lock(&self.app.metrics.deferred).insert(
                        rid.to_owned(),
                        Deferred {
                            method: self.method.clone(),
                            route: route.to_owned(),
                            total_ms,
                        },
                    );
                    return;
                }
                _ => access_line(&Access {
                    request_id,
                    method: self.method.as_str(),
                    route,
                    status,
                    record: record.as_ref(),
                    labels: record.as_ref().and_then(|r| self.app.metrics.labels_of(r)),
                    total_ms,
                }),
            }
        };
        eprintln!("{line}");
    }
}

/// An SSE body that holds its [`Tracked`] until the stream ends or is dropped; a stream dropped before its end
/// (the client left) records [`CLIENT_CLOSED`] instead of the status its headers sent.
struct TrackedStream {
    inner: axum::body::BodyDataStream,
    /// The response status, recorded once the stream has ended.
    sent: Option<u16>,
    tracked: Tracked,
}

impl Stream for TrackedStream {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let next = Pin::new(&mut this.inner).poll_next(cx);
        // The end, or a body error (the server's): not the client leaving.
        if matches!(next, Poll::Ready(None | Some(Err(_)))) {
            this.tracked.status = this.sent;
        }
        next
    }
}

/// The router layer of every route: request counts and durations (`GET /metrics`) and the access line.
pub(crate) async fn track(State(app): State<Arc<App>>, request: Request, next: Next) -> Response {
    let admitted = Arc::new(OnceLock::new());
    let mut tracked = Tracked {
        app,
        started: Instant::now(),
        method: request.method().clone(),
        route: request.extensions().get::<MatchedPath>().cloned(),
        authorization: request.headers().get(header::AUTHORIZATION).cloned(),
        status: None,
        request_id: None,
        admitted: admitted.clone(),
    };
    let response = ADMITTED.scope(admitted, next.run(request)).await;
    tracked.request_id = response.headers().get(REQUEST_ID_HEADER).cloned();
    let streaming = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    if !streaming {
        tracked.status = Some(response.status().as_u16());
        return response;
    }
    let (parts, body) = response.into_parts();
    let stream = TrackedStream {
        inner: body.into_data_stream(),
        sent: Some(parts.status.as_u16()),
        tracked,
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    use crate::api_types::{CacheReport, RequestUsage};
    use crate::ledger::{RequestState, Timings};

    #[test]
    fn histogram_buckets_are_upper_inclusive_and_cumulative() {
        let mut h = Histogram::new(&[1.0, 2.0]);
        for v in [0.5, 1.0, 1.5, 3.0] {
            h.observe(v);
        }
        assert_eq!(h.counts, vec![2, 1]);
        assert_eq!(h.count, 4);
        let mut out = Exposition(String::new());
        out.histogram("x", &[("route", "/a")], &h);
        assert_eq!(
            out.0,
            "x_bucket{route=\"/a\",le=\"1\"} 2\nx_bucket{route=\"/a\",le=\"2\"} 3\n\
             x_bucket{route=\"/a\",le=\"+Inf\"} 4\nx_sum{route=\"/a\"} 6\nx_count{route=\"/a\"} 4\n"
        );
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape_label("a\\b\"c\nd"), "a\\\\b\\\"c\\nd");
        assert_eq!(escape_label("gpu0"), "gpu0");
    }

    /// A reload (new instance) adds its whole count; the same instance adds only its increase; a repeated read
    /// adds nothing.
    #[test]
    fn accumulated_counters_survive_a_reload() {
        let mut a = Accumulated::default();
        a.observe("native-1", 5);
        a.observe("native-1", 5);
        assert_eq!(a.total, 5);
        a.observe("native-1", 8);
        assert_eq!(a.total, 8);
        a.observe("native-2", 3);
        assert_eq!(a.total, 11);
        a.observe("native-2", 12);
        assert_eq!(a.total, 20);
    }

    /// Concurrent scrapes apply their reads out of order: scrape A reads 10, scrape B reads 12 and applies first,
    /// then A applies its older 10. Within one instance the total follows the highest value read, so the stale read
    /// adds nothing and the next read of 12 adds nothing either.
    #[test]
    fn a_stale_read_of_the_same_instance_never_counts_twice() {
        let mut a = Accumulated::default();
        a.observe("native-1", 5);
        a.observe("native-1", 12);
        a.observe("native-1", 10);
        assert_eq!(a.total, 12);
        a.observe("native-1", 12);
        assert_eq!(a.total, 12);
        a.observe("native-1", 13);
        assert_eq!(a.total, 13);
    }

    /// A scrape that read the old load (8 -> 10) applies after another scrape applied the new load: the old load
    /// adds only its own increase (2), it is not mistaken for a third load, and the new load is not re-added.
    #[test]
    fn a_stale_read_of_the_previous_load_adds_only_its_increase() {
        let mut a = Accumulated::default();
        a.observe("native-1", 8);
        a.observe("native-2", 3);
        assert_eq!(a.total, 11);
        a.observe("native-1", 10);
        assert_eq!(a.total, 13);
        a.observe("native-2", 3);
        assert_eq!(a.total, 13);
        a.observe("native-1", 9);
        assert_eq!(a.total, 13);
    }

    /// Only the last `KEPT_LOADS` loads are remembered: the oldest is forgotten once that many newer loads were read.
    #[test]
    fn only_the_last_loads_are_remembered() {
        let mut a = Accumulated::default();
        for load in 0..KEPT_LOADS as u64 {
            a.observe(&format!("native-{load}"), 1);
        }
        a.observe("native-0", 1);
        assert_eq!(a.total, KEPT_LOADS as u64);
        a.observe("native-new", 1);
        assert_eq!(a.loads.len(), KEPT_LOADS);
        assert_eq!(a.loads[0].0, "native-1");
        assert_eq!(a.total, KEPT_LOADS as u64 + 1);
    }

    fn record() -> RequestStatus {
        RequestStatus {
            scope: "a".to_owned(),
            rid: "req_1".to_owned(),
            key: None,
            state: RequestState::Completed,
            reason: None,
            attempt: 1,
            epoch: 1,
            stream: true,
            surface: None,
            session_id: None,
            timings: Timings {
                first_visible_ms: Some(42),
                total_ms: Some(90),
                ..Timings::default()
            },
            usage: RequestUsage {
                prompt_tokens: Some(100),
                completion_tokens: Some(7),
                cached_tokens: Some(64),
                cache_report: CacheReport::Reported,
            },
            worker_id: Some("gpu0".to_owned()),
            worker_epoch: Some(1),
            labels: None,
            class: None,
        }
    }

    #[test]
    fn access_line_has_every_field() {
        let record = record();
        let line = access_line(&Access {
            request_id: Some("req_1"),
            method: "POST",
            route: "/v1/chat/completions",
            status: 200,
            record: Some(&record),
            labels: None,
            total_ms: 95,
        });
        let json = line.strip_prefix("cered access ").unwrap();
        let v: Value = serde_json::from_str(json).unwrap();
        assert_eq!(
            v,
            json!({
                "request_id": "req_1", "method": "POST", "route": "/v1/chat/completions", "status": 200,
                "state": "completed", "input_tokens": 100, "cached_tokens": 64, "output_tokens": 7,
                "ttft_ms": 42, "total_ms": 95,
            })
        );
    }

    #[test]
    fn access_line_without_a_record_is_null_filled() {
        let line = access_line(&Access {
            request_id: None,
            method: "POST",
            route: UNMATCHED_ROUTE,
            status: CLIENT_CLOSED,
            record: None,
            labels: None,
            total_ms: 3,
        });
        let v: Value = serde_json::from_str(line.strip_prefix("cered access ").unwrap()).unwrap();
        for key in [
            "request_id",
            "state",
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "ttft_ms",
        ] {
            assert_eq!(v[key], Value::Null, "{key}");
        }
        assert_eq!(v["status"], 499);
    }

    fn rule(client: &str, workload: &str, scope: &str, priority: Option<bool>) -> ClientRule {
        ClientRule {
            client: client.to_owned(),
            workload: workload.to_owned(),
            scope: Some(scope.to_owned()),
            priority_header: priority,
        }
    }

    fn production_rules() -> Vec<ClientRule> {
        vec![
            rule("coding-agent", "interactive", "a", Some(true)),
            rule("eval", "batch-eval", "a", None),
        ]
    }

    /// A2: with `[[telemetry.clients]]` the line names the record's scope, client and workload.
    #[test]
    fn a_labelled_access_line_names_scope_client_and_workload() {
        let metrics = Metrics::new(&production_rules());
        let mut record = record();
        record.labels = Some(metrics.clients.resolve("a", None, false));
        let line = access_line(&Access {
            request_id: Some("req_1"),
            method: "POST",
            route: "/v1/chat/completions",
            status: 200,
            record: Some(&record),
            labels: metrics.labels_of(&record),
            total_ms: 95,
        });
        let v: Value = serde_json::from_str(line.strip_prefix("cered access ").unwrap()).unwrap();
        assert_eq!(
            v,
            json!({
                "request_id": "req_1", "method": "POST", "route": "/v1/chat/completions", "status": 200,
                "state": "completed", "input_tokens": 100, "cached_tokens": 64, "output_tokens": 7,
                "ttft_ms": 42, "total_ms": 95, "scope": "a", "client": "eval", "workload": "batch-eval",
            })
        );
    }

    /// A2: without rules nothing is labelled and no client family is kept; with them every terminal counts under
    /// its labels, a replayed request (no labels) under `other`.
    #[test]
    fn client_counters_follow_the_rules() {
        let plain = Metrics::new(&[]);
        let mut r = record();
        r.labels = Some(RequestLabels::other());
        assert_eq!(plain.labels_of(&r), None);
        plain.observe_terminal(&r);
        assert!(lock(&plain.by_client).is_empty());

        let metrics = Metrics::new(&production_rules());
        let mut judge = record();
        judge.labels = Some(metrics.clients.resolve("a", None, false));
        metrics.observe_terminal(&judge);
        metrics.observe_terminal(&judge);
        let mut qwen = record();
        qwen.labels = Some(metrics.clients.resolve("a", None, true));
        qwen.state = RequestState::Failed;
        qwen.timings.total_ms = Some(1_234);
        metrics.observe_terminal(&qwen);
        // S7: never dispatched, so its `worker` label is `-`.
        let mut replayed = record();
        replayed.worker_id = None;
        replayed.worker_epoch = None;
        metrics.observe_terminal(&replayed);
        let mut out = Exposition(String::new());
        clients_exposition(&mut out, &lock(&metrics.by_client));
        let samples: Vec<&str> = out.0.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(
            samples,
            [
                r#"cere_client_requests_total{scope="a",client="coding-agent",workload="interactive",worker="gpu0",state="failed"} 1"#,
                r#"cere_client_requests_total{scope="a",client="eval",workload="batch-eval",worker="gpu0",state="completed"} 2"#,
                r#"cere_client_requests_total{scope="a",client="other",workload="other",worker="-",state="completed"} 1"#,
                r#"cere_client_tokens_total{scope="a",client="coding-agent",workload="interactive",worker="gpu0",kind="input"} 100"#,
                r#"cere_client_tokens_total{scope="a",client="coding-agent",workload="interactive",worker="gpu0",kind="cached"} 64"#,
                r#"cere_client_tokens_total{scope="a",client="coding-agent",workload="interactive",worker="gpu0",kind="output"} 7"#,
                r#"cere_client_tokens_total{scope="a",client="eval",workload="batch-eval",worker="gpu0",kind="input"} 200"#,
                r#"cere_client_tokens_total{scope="a",client="eval",workload="batch-eval",worker="gpu0",kind="cached"} 128"#,
                r#"cere_client_tokens_total{scope="a",client="eval",workload="batch-eval",worker="gpu0",kind="output"} 14"#,
                r#"cere_client_tokens_total{scope="a",client="other",workload="other",worker="-",kind="input"} 100"#,
                r#"cere_client_tokens_total{scope="a",client="other",workload="other",worker="-",kind="cached"} 64"#,
                r#"cere_client_tokens_total{scope="a",client="other",workload="other",worker="-",kind="output"} 7"#,
                r#"cere_client_request_seconds_total{scope="a",client="coding-agent",workload="interactive",worker="gpu0"} 1.234"#,
                r#"cere_client_request_seconds_total{scope="a",client="eval",workload="batch-eval",worker="gpu0"} 0.180"#,
                r#"cere_client_request_seconds_total{scope="a",client="other",workload="other",worker="-"} 0.090"#,
            ]
        );
        // The unlabelled totals still count every terminal.
        assert_eq!(lock(&metrics.terminals).output_tokens, 28);
    }

    #[test]
    fn metric_names_are_unique_and_prefixed() {
        let mut names = METRIC_NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), METRIC_NAMES.len());
        assert!(METRIC_NAMES.iter().all(|n| n.starts_with("cere_")));
        for (family, _, _) in &PREFIX_FAMILIES {
            assert!(METRIC_NAMES.contains(family), "{family}");
        }
    }
}
