# Observability

Record stable request/session/attempt/trial identities, worker epoch, config/artifact/profile fingerprints, time spent queued/restoring/prefilling/decoding, cached-prefix tokens, new tokens and actual execution mode/fallback reason. Include startup phase and memory high-water metrics. Root spans such as model load or worker restart need not have a session ID.

Use monotonic time for local intervals and wall time plus clock-domain/boot identity for correlation. GPU events measure execution, host monotonic clocks measure end-to-end latency; do not subtract uncalibrated host/GPU timestamps. Distributed wall clocks require uncertainty handling.

Default event payload is metadata-only. No raw prompts, token IDs, KV, secrets or scope salts in default logs. A full diagnostic trace is explicitly authorized, bounded and retention-controlled; drop-count telemetry makes sampling/overflow visible. Trace loss never blocks the GPU indefinitely.

Separate aggregate completed output / observation window from average per-request decode rate. Track generated first token and first visible token separately. Monitor queue depth, per-scope goodput, cache-hit ratios, restore/replay bytes, in-flight allocations, draft acceptance, rejected work, power and clocks. Measure telemetry overhead with tracing on/off and exclude neither costs nor failures silently.

## Implemented in `cered` (M1)

`cered` writes one stderr line per request terminal and never logs prompt or output text:

```text
cered request rid=<rid> scope=<scope_id> worker=<id@epoch|-> state=<completed|failed|cancelled> reason=<reason|-> stream=<bool> model=<served_model_name|-> queue_wait_ms=<n|-> first_generated_ms=<n|-> first_visible_ms=<n|-> total_ms=<n|-> client_backpressure_ms=<n> prompt_tokens=<n|-> cached_tokens=<n|-> cache_report=<reported|unavailable>
```

`-` means not measured or not reported. `worker=<id>@<epoch>` (M3) is the worker id (its device, e.g. `gpu0`) and worker epoch the request was dispatched to, and `worker=-` for a request that never reached a worker; the same field follows `rid=<rid>` in `cered recovery:` and `cered abort` lines. A worker epoch change prints `cered worker: <id> epoch <old> -> <new> (relaunch|rejoin, instance_id=<id>)`: `relaunch` when the backend reports a new `instance_id`, `rejoin` when a Lost worker passes its probe again. Status, the `cere` response object and `/v1/cere/stats` carry the same `worker_id`/`worker_epoch` ([API.md](API.md)). `prompt_tokens`, `cached_tokens` and `cache_report` are the request's status `usage` ([API.md](API.md), cached-count rule Q1): `cached_tokens` is the backend's count or, under the `enable_cache_report = true` pin, `0` for a miss; it is never estimated.

`model=` (M5 D5, `crates/cere-runtime/src/app.rs:1407-1411`) is the `served_model_name` of the `[[deployments]]` entry the request's worker belongs to, `-` before any dispatch; it follows `stream=` so the earlier field order stays. The startup recovery lines (`app.rs:580-647`) carry the same field, from the dispatched worker's deployment (`-` without one):

```text
cered request rid=<rid> scope=<scope_id> worker=<id@epoch|-> epoch=<n> state=failed reason=worker_restart model=<name|->
cered recovery: re-charged rid=<rid> worker=<worker> scope=<scope_id> model=<name|-> (held when the previous incarnation stopped)
cered recovery: not charged rid=<rid> worker=<worker> scope=<scope_id> model=<name|-> (the native executor ended with its process)
cered recovery: released rid=<rid> worker=<worker> cause=not_dispatched model=<name|->
cered recovery: rid=<rid> worker=<worker> unconfigured model=<name|->
```

The later `cered recovery:` hold and release lines below carry no `model=`.

The journal `Terminal` record (`<state_dir>/journal.jsonl`) carries the same `usage` object, `{"prompt_tokens","completion_tokens","cached_tokens","cache_report"}`. The field is additive: records written before it existed replay with every count `null` and `cache_report: "unavailable"`.

Per-scope aggregates (terminals by state, prompt and cached token totals, reported/unavailable counts since process start) are served by `GET /v1/cere/stats` to the scope's own token only; the cross-scope operator view is `GET /metrics` ([below](#prometheus-metrics)).

Session owner changes (M4) print one stderr line each, after the change is journaled: `cered event=session_owner session=<id> worker=<w> generation=<g> cause=pin|repin|migrate|force`. `pin` is a session's first turn, `repin` a turn granted away from an owner that was not eligible, `migrate`/`force` an operator migration ([API.md](API.md#operator-routes-session-owner-and-migration)).

## Prometheus metrics

`GET /metrics` (ops-hardening design note OH-05, `crates/cere-runtime/src/metrics.rs`; operator token only, [API.md](API.md#operator-route-metrics)) serves the Prometheus text exposition format 0.0.4, written by hand (no metrics crate). Counters are process-lifetime: they restart at 0 when `cered` restarts, which `rate()`/`increase()` handle. No family carries prompt or output text, a token or a request id; only the `cere_client_*` families carry a scope id (the configured `scope_id`, never a token). The families, in exposition order (the `METRIC_NAMES` const; `tests/test_monitoring_alerts.py` checks the alert rules against it):

| Family | Type | Labels | Meaning |
|---|---|---|---|
| `cere_requests_total` | counter | `route`, `status` | HTTP requests by matched route template (`/v1/cere/requests/{id}`, never the raw path; `unmatched` for a path no route matches) and response status; `499` when the client left before the response ended: before any response, or mid-stream after a stream's `200` headers (the wire status then was `200`; a request that got its whole response never counts as `499`). Counted when the response body ended (an SSE stream at its last event) or the client left. |
| `cere_request_duration_seconds` | histogram | `route` | Time from the request to the end of its response body (buckets 5 ms to 600 s). |
| `cere_ttft_seconds` | histogram | — | Admission to first visible output (the ledger's `first_visible_ms`), once per request terminal that had one: streamed requests only (buckets 25 ms to 120 s). |
| `cere_tokens_total` | counter | `kind` = `input`, `cached`, `output` | Reported `prompt_tokens`, `cached_tokens` and `completion_tokens` of each request terminal. `cached` is the part of `input` served from a prefix cache, so `cached / input` is the token hit rate; a backend that reports no cached count adds nothing. |
| `cere_client_requests_total` | counter | `scope`, `client`, `workload`, `worker`, `state` = `completed`, `failed`, `cancelled` | Request terminals by the request's [client and workload](#client-and-workload-labels-roadmap-v07-a2) and the worker it was last dispatched to (`worker` = the configured `worker_id`, `-` when never dispatched; net-teleport design note 4.4, S7). Only with `[[telemetry.clients]]`: without rules the family has no `# HELP` either, so the exposition is byte-identical to a config without them. |
| `cere_client_tokens_total` | counter | `scope`, `client`, `workload`, `worker`, `kind` = `input`, `cached`, `output` | `cere_tokens_total` split by client, workload and worker (every terminal; a failed request mostly reports none). Only with `[[telemetry.clients]]`. |
| `cere_client_request_seconds_total` | counter | `scope`, `client`, `workload`, `worker` | The request terminals' admission-to-terminal time (the `cered request` line's `total_ms`), summed, in seconds with millisecond resolution. Only with `[[telemetry.clients]]`. |
| `cere_request_params_total` | counter | `surface` = `chat`, `anthropic`, `responses`; `param` (the closed `MAPPED_PARAMS`/`INERT_PARAMS`/Anthropic lists of `validate.rs`, anything else `unlisted`); `action` = `ignored`, `mapped` | Requests that sent a parameter the request layer accepted without effect (`ignored`, also listed in the response header `X-Cere-Ignored-Params`) or rewrote to its canonical field (`mapped`), counted once per validated request (api-lenient design note D4). |
| `cere_prefix_cache_{hits,captures,capture_skipped,capture_page_mismatch,inserts,state_evictions,leaf_evictions,host_spills,host_loads,host_drops,alt_edges}_total` | counter | `worker` | Each local native worker's `cere_model_prefix_stats` (the counters of [`GET /v1/cere/prefix_cache`](API.md#operator-route-prefix_cache)), read at the scrape. The library restarts them at each model load; a load is told apart by its backend `instance_id` (new at every load), never by a value going down. The exported counter adds a new load's whole count and, for a load already read, only what a read exceeds the highest value read of that load by, so it never decreases and concurrent scrapes never count an increase twice (increments between the last scrape and an unload are lost). A cache-off worker has no series. |
| `cere_prefix_cache_capture_skipped_by_site_total` | counter | `worker`, `site` = `prefill_slot`, `decode_track`, `end_pair` | `capture_skipped` split by the capture site that found no free state slot (`capture_skipped_by_site` of [`GET /v1/cere/prefix_cache`](API.md#operator-route-prefix_cache), `cere_model_prefix_capture_stats`, prefix-token design note PT0a); the sites sum to `cere_prefix_cache_capture_skipped_total`. Accumulated across loads like the prefix counters; no series on a library without the export. |
| `cere_prefix_cache_hit_shortfall_tokens_total`, `cere_prefix_cache_hit_shortfall_lookups_total` | counter | `worker` | Prompt tokens drafting prefix lookups recomputed because no draft state sat at the deepest prefill-chunk multiple their page match passed, and the lookups that lost any (PT0a); accumulated like the prefix counters. |
| `cere_ledger_records` | gauge | `state` = `live`, `terminal` | Request ledger records: not yet terminal, and terminal within retention. |
| `cere_ledger_capacity` | gauge | — | `ledger.max_records`. |
| `cere_ledger_evictions_total` | counter | — | Terminal records evicted to admit a request (OH-03). |
| `cere_sidecar_respawns_total` | counter | `worker` | Tokenizer sidecar respawns of each loaded native worker (OH-01), accumulated across loads like the prefix counters. |
| `cere_worker_up` | gauge | `worker` | 1 while the worker's `GET /readyz` entry is `ready` (local and remote workers), else 0. |
| `cere_ready` | gauge | — | 1 while `GET /readyz` answers 200, else 0. |
| `cere_device_poisoned` | gauge | `worker` | 1 after an internal executor fault on a local native worker's card, until a re-smoke passes (sticky once its reload failed); a worker with no model loaded reports its slot's fault. |

A Prometheus scrape job (the operator token in a file readable only by Prometheus):

```yaml
scrape_configs:
  - job_name: cere
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/cere-operator-token
    static_configs:
      - targets: ["127.0.0.1:18080"]
```

The alert rules are `deploy/monitoring/cere-alerts.yml`; the response to each is in [RUNBOOK.md](RUNBOOK.md#monitoring-and-alerts). `deploy/monitoring/cere-alerts.test.yml` holds their behavioral tests (`promtool test rules`, run from `deploy/monitoring/`); `tests/test_monitoring_alerts.py` runs them with a local `promtool`, else in the local `prom/prometheus` docker image, and skips with the reason when neither is available.

On gpumaster, Alertmanager (`deploy/monitoring/alertmanager.yml`, 127.0.0.1:9093) delivers the alerts to the operator's desktop and to `/var/log/cere-alerts.log` through the loopback webhook `deploy/monitoring/cere_alert_notify.py`. See [RUNBOOK.md](RUNBOOK.md#alert-delivery).

## Access log

`cered` writes one stderr line per finished HTTP request (ops-hardening design note OH-06), except `GET /healthz`, `GET /readyz` and `GET /metrics`, when the response body ended (an SSE stream at its last event) or the client left first:

```text
cered access {"cached_tokens":<n|null>,"input_tokens":<n|null>,"method":"POST","output_tokens":<n|null>,"request_id":"<rid>"|null,"route":"/v1/chat/completions","state":"<state>"|null,"status":200,"total_ms":<n>,"ttft_ms":<n|null>}
```

`route` is the matched route template (`unmatched` otherwise), `status` the HTTP status (`499`: the client left before the response ended, as `cere_requests_total` counts it; a stream cut after its `200` headers is `499`, not `200`, so a cut stream never reads as a success). `request_id` is the request id the ledger admitted (the response's `X-Cere-Request-Id`, or the admitted id when the client left before any response): `null` only for a request refused before the ledger accepted it (a `401`, for one) and a route that answers about a request rather than being one (`GET /v1/cere/requests/{id}`). The line of a request whose client left while it was still in flight is written when the request ends, right after its `cered request` line, so `state` is the final one (`cancelled`, reason `client_disconnected` in the `cered request` line; with `X-Cere-On-Disconnect: continue` the state the detached request ended in); `total_ms` stays the time until the client left. With a request id, `state`, the token counts (`input_tokens`/`cached_tokens`/`output_tokens`: the status `usage` `prompt_tokens`/`cached_tokens`/`completion_tokens`) and `ttft_ms` (the status `timings.first_visible_ms`, streams only) are read from the caller's own ledger record; `null` when not known. `total_ms` is measured at the HTTP layer. An idempotent replay logs again under the same `request_id`. The line holds no prompt or output text and no header value; it names the scope only with `[[telemetry.clients]]` (below).

### Client and workload labels (roadmap v0.7 A2)

With `[[telemetry.clients]]` rules ([CONFIGURATION.md](CONFIGURATION.md#state-telemetry-safety), `crates/cere-runtime/src/labels.rs`) every chat request gets a client and a workload at acceptance, kept with its ledger record (memory only). A request that names a client in `X-Cere-Client` gets the first rule of that client whose `scope` matches; otherwise the first rule whose `scope` (any when absent) and `priority_header` (whether the request carried `X-Cere-Priority`; either when absent) match; otherwise `other`/`other`. A request replayed from the journal after a restart is `other`. The label set is bounded by the config (at most 16 rules), so a client cannot mint series. Then:

- the `cered request` line ends in ` client=<client> workload=<workload>` (before ` research_plan=`, which stays last);
- the `cered access` line of a request with a record also has `"scope"`, `"client"` and `"workload"`;
- `GET /metrics` exports the three `cere_client_*` families above.

Without rules none of this appears: the lines and the exposition are byte-identical to a `cered` without the feature. Production's rules (`configs/runtime/prod-t4.toml`) derive the client from what the clients already send: frontend's proxy adds `X-Cere-Priority` on its `coding-agent` route only (route key `admission_priority`), so a keyless `frontend` request with it is `coding-agent`/`interactive` and one without it `judge`/`batch-eval` (every other frontend route to CERE without `admission_priority` counts there too); scope `ae` is `axon`/`eval-repair`. The census headline (CensusW12, an internal benchmark report) is then one command, read-only:

```bash
python3 scripts/live/census.py --metrics http://127.0.0.1:18080/metrics --token-env CERE_OPERATOR_TOKEN \
  --journal <(journalctl -u cere-t4 --since -8d -o short-iso-precise --no-pager)
```

It reads requests, tokens and request wall per client from `/metrics` (since the process start) and fits the decode wall (`total_ms = a + b·uncached + c·output`, refit on the 90 % smallest residuals) on the journal's labelled `cered access` lines of the same process (after its last `cered listening` line). Against a `cered` without the families it prints one `all` row and says so.

### Usage journal (NetTeleport S7)

net-teleport design note 4.4 "Usage report" (C20, F22, Q-T9), `crates/cere-runtime/src/usage.rs`. With `[usage]` enabled (the default; [CONFIGURATION.md](CONFIGURATION.md#usage)) `cered` appends one JSON line per request terminal, the same terminals the `cered request` line reports (startup-recovery failures included), to `<ledger.state_dir>/usage/YYYY-MM-DD.jsonl` by the terminal's UTC day:

```json
{"kind":"request","ts_ms":1795444829668,"scope":"a","worker":"gpu1","worker_owner":null,"tier":"own","class":null,"state":"completed","reason":null,"prompt_tokens":1,"cached_tokens":0,"output_tokens":4,"queue_wait_ms":0,"wall_ms":210}
```

- `scope` is the person (auth scope id); `worker` the configured `worker_id` it was last dispatched to (`null` if never); `tier` is `server.tier` for a local worker and, for a remote one, its tier for the row's scope (`own` when the scope is its `[[federation.workers]] owner`, else its `tier`), `null` if never dispatched; `worker_owner` is a remote worker's `owner` (`null` for a local or unowned worker); `class` is the request's `interactive` or `batch` (`X-Cere-Class`, else the scope's `class`), `null` for a request the previous incarnation left in flight (all three are set in one place, `UsageRow::attribute`). Token counts are the backend's reported ones (`null` = not reported), `queue_wait_ms` and `wall_ms` (admission to terminal) the request line's `queue_wait_ms` and `total_ms`.
- No prompt or output text and no request id: nothing maps a row to a prompt (spec 4.1).
- `{"kind":"owner_only","worker":…,"owner":…,"start_ms":…,"end_ms":…}` rows record an ended owner-only (opt-out) span of a worker; S6 writes them (`UsageJournal::record_owner_only`), filed by the end's day.
- The serving path only queues a row (bounded queue of 4096, never blocks; a full queue drops the row and prints `cered usage: row dropped (...) dropped_total=<n>` at the first drop and each power of two). One writer thread appends each batch and `fdatasync`s it, so a written row survives a restart or crash; a graceful stop writes and syncs every queued row before the next incarnation can start. A day file a crash cut mid-row gets a newline before the next row (the report skips the cut line and counts it).
- Day files older than `usage.retention_days` (default 90, today included) are deleted at start and on each new UTC day.

The report, read-only, prints person x machine tables (requests, prompt/cached/output tokens, GPU-wall = `wall_ms - queue_wait_ms`, queue wait), each machine owner's self-use versus use by others (machines whose rows carry no `worker_owner` are listed as not recorded) and opt-out hours per machine (owner-only spans clipped to the window); `--json` prints the same as JSON. `--request-log` (repeatable: `cered` stderr or `journalctl -o short-iso-precise` output) checks bar S7: the window's rows against the `cered request` lines (count, prompt and cached tokens per person, machine and state) and the output tokens of the chat `cered access` lines; exit 1 on any difference.

```bash
python3 scripts/live/usage_report.py --from 2026-11-01 --to 2026-11-30 --state-dir /var/lib/cere/state
```

## Implemented federation log lines (M3.5)

No `cere-worker` or `cered` federation line carries prompt or output text, a completion ticket, a token or key material. Fields are metadata only: identifiers, epochs, terms, generations, rounds, sequence numbers, reason codes and times. `<scope>/<op>` is an operation key (`scope_id`/`operation_id`). Reason codes are the [WORKER_PROTOCOL.md](WORKER_PROTOCOL.md) and [API.md](API.md) codes. The harness tool `cere-rpc-probe` (below) prints completion tickets, never text.

### `cere-worker`

stdout, once the agent has taken its state-dir lock, persisted the new epoch, loaded the PKI and started the listener, the coordinator link and the backend probe (`generation` is the persisted model generation, `0` before the first confirmed backend):

```text
cere-worker listening on <addr> worker=<worker_id>@<epoch> generation=<model_generation>
```

`--check-config <toml>` prints `config ok worker=<worker_id>` on stdout and exits 0, or prints the error on stderr and exits 2; it opens no PEM. `--version` prints `cere-worker <PACKAGE_VERSION>`. A startup failure prints `cere-worker: <error>` on stderr and exits 2; a usage error prints `usage: cere-worker [--help | --version | --check-config <path> | --config <path>]` and exits 2. SIGTERM or SIGINT stops the agent with exit 0.

Events go to stderr as `cere-worker event=<kind> <key>=<value> ...`:

| Kind | Fields | When |
|---|---|---|
| `register` | `term=<t> accepted=true epoch=<e>` or `term=<t> accepted=false epoch=<e> reason=<code>` | each Register reply for the current epoch |
| `lease` | `state=granted sequence=<n> until_ms=<unix_ms>` | an accepted heartbeat granted a new lease (normally once per heartbeat interval) |
| `lease` | `state=expired sequence=<n>` | once per lapse, when the last lease ran out without a new grant |
| `fence` | `reason=coordinator_restart term=<new> previous_term=<old>` | the agent adopted a higher coordinator term (not printed for the first term) |
| `generation` | `generation=<g> instance_id=<id>` | a new backend launch was confirmed (a new model generation) |
| `accept` | `op=<scope>/<op> fresh=<true\|false>` | a SubmitWork or CancelWork was accepted; `fresh=false` is a matching retry answered from the journal |
| `terminal` | `op=<scope>/<op> reason=<code>` | an operation became terminal; an empty `reason=` is a normal end (and every CancelWork operation itself) |
| `cleanup` | `op=<scope>/<op>` | the operation's backend hold was released and the cleanup journaled |
| `poisoned` | `reason=journal_write` or `reason=state_write` | a journal or state-file write failed; the agent stays `poisoned` for the life of the process |

Terminal reasons seen here: `deadline_exceeded`, `client_backpressure`, `cancel_requested`, `worker_restart`, `coordinator_restart`, `model_generation_superseded` and the backend error codes of [API.md](API.md) (`backend_error`, `backend_protocol_error`, ...).

Other stderr lines: `cere-worker: backend identity not confirmed: <problem>` (once per distinct problem), `cere-worker: <Register|Heartbeat> failed: <gRPC code> <reason>` (once per distinct code, so an unreachable coordinator is one line), `cere-worker: inventory not built: <error>` and `cere-worker: listener error: <error>`.

### `cered`

With `[federation]`, `cered` prints on stderr, before the HTTP listening line:

```text
cered federation listening on <addr> cluster=<cluster_id> term=<term>
```

`term` is the process epoch printed by `cered listening on http://<addr> epoch=<n>` on stdout. Remote worker lines on stderr:

```text
cered remote worker=<id>@<epoch> event=register outcome=<incarnation|term|reconnect> term=<t> round=<r>
cered remote worker=<id>@<epoch> event=register refused=epoch_lost
cered remote worker=<id>@<epoch> event=joining round=<r> generation=<g> admission_state=<open|fenced|poisoned>
cered remote worker=<id>@<epoch> event=reconcile cancelled=<scope>/<op> operation=<cancel_id>
cered remote worker=<id>@<epoch> event=smoke state=failed round=<r>
cered remote worker=<id>@<epoch> event=ready round=<r> generation=<g>
cered remote worker=<id> event=stale_round round=<r> current=<c>
```

`register` is an accepted Register (`incarnation`: a new epoch, journaled; `term`: the first Register of this epoch in this term; `reconnect`: already registered). `joining` is a `ready` worker sent back to a new round. Other Register and heartbeat refusals are answered, not logged; Suspect and Lost are computed from receive times when read and are visible in `/readyz`, not as lines. Failures: `cered readiness: cancel of remote smoke rid=<rid> on worker=<id> failed: <status>` and `cered federation: registry listener failed: <error>`.

Remote dispatch and orphan lines on stderr (metadata only):

```text
cered remote worker=<id> not rejoined after a refusal: <error>
cered remote submit rid=<rid> worker=<id> try <n> failed: <status>
cered remote cancel rid=<rid> worker=<id> failed: <status>
cered remote stream parked rid=<rid> worker=<id>@<epoch>: <status>
cered remote orphan rid=<rid> worker=<worker>: not holdable in the ledger; hold not journaled
cered remote orphan: cannot journal the hold rid=<rid> worker=<worker>: <error>
cered remote orphan poll failed rid=<rid> worker=<worker>: <status>; held
cered remote orphan rid=<rid> worker=<worker>: not held in the ledger; release not journaled
cered remote orphan: cannot journal the release rid=<rid> worker=<worker>: <error>
cered remote orphan released rid=<rid> worker=<worker> cause=<cause>
cered recovery: held rid=<rid> worker=<worker> until unix_ms=<ms>; released earlier once its agent reports the cleanup in term <t>
cered recovery: poll failed rid=<rid> worker=<worker>: <code> <message>; held
cered recovery: rid=<rid> worker=<worker> not held in the ledger; release not journaled
cered recovery: cannot journal the release rid=<rid> worker=<worker>: <error>
cered recovery: released rid=<rid> worker=<worker> cause=<cause>
```

`cause` is `cleanup_acknowledged` (the agent reported the operation cleaned up) or `retention` (the request's retention bound passed). A remote orphan hold is journaled when it starts (`orphan_held`) and when it ends (`orphan_released`); a hold whose start could not be journaled is not charged again after a restart. `cered recovery:` lines are the coordinator-restart (Q15) holds of remote orphans recovered from the journal: prior-term requests, and holds journaled `orphan_held` without an `orphan_released` (stderr `cered recovery: re-charged rid=<rid> ...` first).

### `cere-rpc-probe`

The live harness probe prints one JSON line per call on stdout:

```json
{"call": "submit", "n": 0, "code": "OK", "reason": "", "accepted": true, "ticket": "…", "ms": 12}
```

`call` is `register`, `heartbeat`, `describe`, `list`, `submit`, `stream`, `cancel` or `oversize`; `n` is the repeat index of `submit --repeat` (else 0); `code` is the gRPC code name, or `TRANSPORT` with `reason` `handshake` or `connect` when the call failed in the transport; `reason` is the reason code; `ms` is the call's wall time. `describe` adds `inventory_sha256` (hex, `""` on failure); `submit --stream` adds a `stream` line. The submit body is synthetic, and prompt or output text is never printed. Exit 0 when every call printed its line, 2 on a usage error, 1 otherwise.

## Model-management events and diagnostics

Add lifecycle telemetry for `model_load_requested`, `model_loading`, `model_ready`, `model_drain_started`, `model_unload_requested`, `model_unloaded`, `model_load_failed`, residency-lease changes and eviction reasons. Record load wait separately from request queue, prefill and decode.

Diagnostics should expose identities, sizes, lifecycle state, device assignment and cache-miss reasons without leaking prompt content or cross-scope cache membership. See [DIAGNOSTICS.md](DIAGNOSTICS.md).


## Federated worker dimensions

Every execution event in a multi-node-capable trace carries `worker_id` + `worker_epoch`. Request attempts may not silently move between workers in the v1.x federation contract. Worker registration/heartbeat/suspect/lost and model unload events are separately observable. Metrics aggregate by logical worker and epoch so a reboot does not merge two daemon incarnations.

## v0.6 complete evidence

An accepted request may precede worker assignment; its worker fields can be null until dispatch. Token events require a worker identity. Non-null artifact/config/session/trial identities cannot conflict within an attempt. Independent admission rosters detect entire missing attempts; without one the analyzer explicitly reports that completeness is not independently established.

Use one observer clock domain for end-to-end benchmarking; do not combine peer monotonic timestamps into a latency subtraction. Replayed output events are not new model-generated tokens.
