# Worker protocol revision 2 - implemented subset (M3.5, M3.6)

Protocol revision 2 is **not** the product's v2 distributed-inference milestone. M3.5 implements the subset below
over mTLS gRPC (`proto/cere_worker.proto`, package `ai.cere.worker.v2`, bindings in `crates/cere-proto`); the proto
was amended in place because there were no deployed peers. A Register with any other `protocol_revision` is
refused `unsupported_revision`. Revision-1 sketches are rejected, never interpreted.

Built: the coordinator's `WorkerRegistryService` inside `cered` (`[federation]`, `crates/cere-runtime/src/fed/`)
and the node agent `cere-worker` (`crates/cere-runtime/src/agent/`), which serves one worker over one loopback SGLang
backend: in **attached** mode (no `[backend.launch]`) an operator-started backend, in **managed** mode (M3.6,
`[backend.launch]`) a backend the agent launches and stops itself on `LoadModel`/`UnloadModel`
([Model lifecycle](#model-lifecycle-m36)). The coordinator dispatches client requests to a Ready remote worker
(`SubmitWork`, `StreamWork`, `CancelWork`; `crates/cere-runtime/src/fed/remote.rs`).

## Roles and authority

A **Node** is a PC/host. A **Worker** is one exclusively managed execution/resource domain on that node. Worker IDs
are cluster-unique, not CUDA ordinals. Worker epochs fence restarted agents; model generations fence backend
relaunches inside an epoch; the coordinator term fences coordinator restarts.

| Service | Implemented by | Called by | RPCs |
|---|---|---|---|
| `WorkerRegistryService` | coordinator (`cered`) | admitted workers | `Register`, `Heartbeat` |
| `WorkerService` | worker (`cere-worker`) | the coordinator | `DescribeResources`, `CancelWork`, `GetOperation`, `ListOperations`, `StreamWork`, `LoadModel`, `DrainModel`, `UnloadModel` |
| `WorkerInferenceService` | worker | the coordinator | `SubmitWork` |

`StreamTelemetry` answers `UNIMPLEMENTED`. `SubmitWork` lives in its own service so it gets the 8 MiB decode limit
while the control services keep 1 MiB (tonic limits are per service).

## Error channel

`WorkerService`/`WorkerInferenceService` refusals are gRPC statuses whose message is exactly the reason code
(`OperationReceipt.accepted` is true on OK). `WorkerRegistryService` protocol refusals are `accepted: false` plus
`reason_code` with status OK; authentication, quota, limits and malformed messages use statuses. A message that
breaks its contract (missing authority or target, `payload_sha256` not 32 bytes, unparsable `work_unit_json`, a
body whose `rid` is not `request_id`, a list limit outside 1..=256) is `INVALID_ARGUMENT malformed_request`
(`src/fed/wire.rs`).

## Authentication and authorization

- One private cluster CA per deployment. The server requires a client certificate under that CA; the client trusts
  only that CA and checks the server against the expected peer principal (`domain_name`), never an IP
  (`src/fed/tls.rs`).
- A principal is the **only** SAN of the CA-verified leaf and must be a valid, non-wildcard dNSName. No SAN, several
  SANs, or an IP/URI/email SAN: `PERMISSION_DENIED unauthenticated_peer`. Caller-supplied ids are never identity.
- At load, the own leaf must name the configured `principal`, and the key file must hold exactly one unencrypted
  PKCS#8, PKCS#1 or SEC1 `PRIVATE KEY` block.
- Role table, checked before any handler state: on the coordinator, `Register`/`Heartbeat` callers must be the
  principal bound to the claimed `worker_id` in `[[federation.workers]]`; on the agent, every caller must be the
  configured `coordinator.principal`. Anything else is `PERMISSION_DENIED permission_denied`.
- After authorization each principal holds a bounded number of concurrent unary calls (coordinator
  `rpc_quota_per_principal`, agent `limits.control_quota`); over the bound is `RESOURCE_EXHAUSTED quota_exhausted`
  (`src/fed/quota.rs`).

## Clocks

No cross-machine timestamp is compared (m35 design note Q11). The coordinator ages a worker by its own receive
time of the last accepted heartbeat. The agent measures a granted lease from the instant it sent the heartbeat that
earned it (`until = sent + admission_lease_ms`), so the agent's view ends no later than the coordinator's. The
worker-local work deadline is the receipt instant + min(`remaining_budget_ms`, 600 000 ms); a retry never extends it.

## Epoch, term and generation

- **Worker epoch.** `cere-worker` raises `worker_epoch` by 1 at every start and persists it in
  `<state_dir>/agent_state.json` (temp file 0600, fsync, rename, directory fsync) before serving or registering.
  Operations of an older epoch become terminal `worker_restart` and are aborted on the backend. The file also holds
  `model_generation`, `instance_id` and the highest accepted `coordinator_term`; never prompt or output text.
- **Coordinator term.** The ledger's process epoch of `cered`. The agent persists the highest term accepted from the
  coordinator principal; a higher term ends every live operation `coordinator_restart` and needs a new Register.
- **Model generation.** The agent probes its backend (`GET /get_server_info`, `GET /v1/models`) every heartbeat
  interval (attached mode: always; managed mode: only while the model is `loading`, `ready` or `draining`, and a
  confirmation counts only while the recorded child is alive). A confirmed identity has `instance_id = "sglang-" + `
  the first 8 bytes (lowercase hex) of SHA-256 over the canonical JSON of the base URL, served model name, pins and
  launch (`random_seed`, `startup_time`). A different `instance_id` bumps the durable generation and ends
  older-generation `SubmitWork`/`CancelWork` operations `model_generation_superseded` (lifecycle operations end only
  through the model state machine); an outage followed by the same `instance_id` changes nothing.

### Register outcomes (coordinator, `src/fed/registry.rs`)

In order: `protocol_revision != 2` -> `unsupported_revision`; another `cluster_id` -> `wrong_cluster`; epoch 0 ->
`INVALID_ARGUMENT`; an unconfigured worker -> `unknown_worker`; epoch below the journaled one -> `stale_epoch` with
`journaled_worker_epoch` set (the agent jumps to it + 1). Otherwise accepted:

| Case | Outcome |
|---|---|
| epoch above the journaled one (or none) | `incarnation`: journaled before the reply, state reset to Joining |
| equal epoch, first Register this term | `term`: state reset to Joining |
| equal epoch, already registered this term, not Lost | `reconnect`: state kept |
| equal epoch, already registered this term, Lost | refused `epoch_lost` |

An accepted reply carries `coordinator_term`, `admission_lease_ms` and `requires_reconciliation = true`.

## Lease and fence

Heartbeat refusals, in order: `wrong_cluster`, `unknown_worker` (unconfigured, no Register this term, or a higher
epoch), `stale_term`, `stale_epoch`, `epoch_lost`, `replayed_heartbeat` (sequence not strictly increasing in the
epoch). An accepted heartbeat grants the next lease: `lease_sequence` restarts at 1 per term, `desired_state` is
`DRAINING` while drain intent is set, and `reconciled_generation` is set once reconciliation finished for this term
and epoch. A Ready worker returns to Joining on a new generation or inventory hash, an admission state other than
`open` (or `fenced` while draining), or an expired lease view. A draining worker's Joining round skips the smoke
(its agent refuses `SubmitWork`); undrain starts a new round with the smoke.

The agent reports `admission_state`: `poisoned` after a journal or state write failure (for the life of the
process); `open` only while the lease is live under the current term, the worker is not draining, the generation is
at least 1 and `reconciled_generation` equals it; `fenced` otherwise. A `SubmitWork` with a new `(scope_id,
operation_id)` is admitted only while `open` (`UNAVAILABLE poisoned` or `UNAVAILABLE lease_expired`); a retry of an
already journaled key is answered by the journal in any state except `poisoned` (same identity: the original receipt,
`fresh=false`; another identity: `operation_conflict`). `CancelWork` is refused only when poisoned. Every call
carries `Authority`; a `coordinator_term`, `worker_epoch` or `model_generation` other than the agent's current one is
`FAILED_PRECONDITION stale_term`/`stale_epoch`/`stale_generation`, another target worker or coordinator id is
`wrong_worker`/`wrong_coordinator`.

Team pools (net-teleport design note 4.4, C19): an agent with an `owner` reports `owner_only = true` (field 7)
while one of its `owner_only` windows is open or its local `cere-worker owner-only on` switch is set, and refuses a
new `SubmitWork` key of any scope but its owner and the readiness smoke's `cere-smoke` as `UNAVAILABLE owner_only`.
The coordinator then grants the worker only to its owner; an older agent never sets the field (false).

## Operation journal

`<state_dir>/operations.jsonl` (directory 0700, file 0600), one JSON record per line (`accepted`, `terminal`,
`cleanup`), fsynced before the call returns (`src/agent/journal.rs`, m35 design note Q12). Journaled methods are
`SubmitWork`, `CancelWork`, `LoadModel`, `DrainModel` and `UnloadModel`; reads are not journaled. The journal is
forward-only: an agent older than M3.6 refuses a journal holding a lifecycle method (`unknown method`).

- Key `(scope_id, operation_id)`. Identity = method, keyed digest, worker epoch, model generation, coordinator term.
  The journal stores `HMAC-SHA256(agent_secret, len(scope_id) || scope_id || payload_sha256)`, never the raw digest;
  `agent_secret` is 32 bytes hex in `<state_dir>/agent_secret` (0600), created on first start.
- Same key and identity returns the original receipt (no second backend call); a different identity is
  `ALREADY_EXISTS operation_conflict`. The completion ticket is 128 random bits (hex).
- `journal.max_operations` bounds live + retained operations; a new key beyond it is
  `RESOURCE_EXHAUSTED journal_full` and nothing is evicted. A terminal, cleanup-acknowledged operation is dropped at
  max(terminal + `retry_horizon_ms`, deadline + `rpc_timeout_ms`). `SubmitWork` operations not yet
  cleanup-acknowledged count toward the live-operation bound (`CancelWork` and lifecycle operations never do).
- A torn final line is cut at open. A failed write poisons the agent.

## SubmitWork, StreamWork and CancelWork

- `payload_sha256` is SHA-256 over the u64 big-endian length-prefixed concatenation of method, `scope_id`,
  `request_id`, `attempt` (decimal ASCII), `work_unit_json` and `bounded_inference_input_json` (SubmitWork) or
  `completion_ticket` (CancelWork). Budget, term, epoch and generation are not inputs. The agent recomputes it:
  `INVALID_ARGUMENT payload_digest_mismatch`.
- `work_unit_json` is `{"kind":"chat","stream":<bool>}`; the body's `rid` must equal `request_id`. Streamed chat
  maps to the backend's streaming call, unary to one chat call.
- Each backend chunk becomes one `WorkEvent` (the parsed value re-serialized); `sequence` runs from 1 and
  `committed_token_offset = sequence - 1`. An event over 64 KiB ends the operation `backend_protocol_error`.
  Unsent events wait in a buffer of `limits.output_buffer_bytes`; when it is full the pump stops reading, and after
  `limits.output_stall_ms` without a drain the operation ends `client_backpressure`.
- `StreamWork` attaches by `(authority.scope_id, authority.operation_id)`: unknown -> `NOT_FOUND unknown_operation`;
  a different `request_id`, `attempt` or ticket -> `FAILED_PRECONDITION ticket_mismatch`; a second concurrent attach
  -> `ALREADY_EXISTS operation_conflict`. It replays from `after_sequence` while the operation's buffer is held. A
  terminal operation whose cleanup is journaled and that a stream has attached to before leaves the table and frees
  its buffer; so does every operation after an agent restart. A later attach then gets only the terminal event, at
  `after_sequence + 1`, with no replay: unsent events are gone (opportunities design note O24; `cered` attaches once,
  from 0). Open streams per principal are bounded by the backend's `max_running_requests` + 1 (`RESOURCE_EXHAUSTED
  quota_exhausted`). Agent shutdown ends open streams `UNAVAILABLE worker_restart`.
- Terminal reasons: `""` (natural completion), a backend error code, `deadline_exceeded`, `client_backpressure`,
  `backend_protocol_error`, `cancel_requested`, `coordinator_restart`, `model_generation_superseded`,
  `worker_restart`. The journal's first terminal record wins, so an operation never has two.
- Cleanup is acknowledged only after the backend stream for the operation ended. An operation this process does not
  hold (accepted by an earlier epoch) is cleanup-acknowledged by the `/get_load` rule when `backend.load_endpoint`
  is set, else at its worker-local deadline.
- `GetOperation`/`ListOperations` report `OperationStatus` with `phase` `accepted` | `terminal` | `released`;
  `ListOperations` pages by `(scope_id, operation_id)`, at most 256 per page.

## Model lifecycle (M3.6)

`LoadModel`, `DrainModel` and `UnloadModel` (`ModelLifecycleRequest`, `src/agent/service.rs`, `src/agent/node.rs`)
drive the one model of a **managed** agent (`[backend.launch]` in its TOML, [CONFIGURATION.md](CONFIGURATION.md)).
A request names only the deployment lock: no path, URL, image or command. The agent launches only the argv of its
own config.

**Check order.** Principal authz -> control quota -> request shape and target -> admission (`UNAVAILABLE poisoned`)
-> attached mode `FAILED_PRECONDITION model_unmanaged` -> the recomputed digest -> journal lookup -> for a new key
only: the fences (current term, epoch and generation), `lock_not_approved`, `unsupported_placement`, the state
table. A journaled key with the same identity returns its stored receipt (and ticket) whatever the current term,
epoch or generation; another identity is `operation_conflict`. A retry resends the original `Authority` unchanged.

- Shape: `deployment_lock_sha256` exactly 32 bytes (else `INVALID_ARGUMENT malformed_request`); `device_ids` must be
  empty (`FAILED_PRECONDITION unsupported_placement`); `instance_id` empty for Load, and for Drain/Unload empty or
  the current one (else `stale_generation`). A lock other than the agent's `[backend] lock_sha256` is
  `FAILED_PRECONDITION lock_not_approved`, before any journal or process state.
- **Lifecycle digest**: `payload_sha256` = SHA-256 over the u64 big-endian length-prefixed `method`, `scope_id`,
  `deployment_lock_sha256` (32 raw bytes), `instance_id`, then each `device_id` (`wire::lifecycle_sha256`). Term,
  epoch and generation are not inputs.
- `Authority.model_generation` is required and equals the current generation for a new key (0 before the first
  confirmed backend). Journal fields: `request_id = operation_id`, `attempt = 1`, a completion ticket as for
  SubmitWork, deadline = accepted + 600 000 ms (bounds tombstone retention only). Receipts are two-phase: `accepted`
  once journaled, completion polled with `GetOperation`, which reports the `method`; `model_generation` there stays
  the accepted one (the new generation is read from the inventory).
- Lifecycle operations are cleanup-acknowledged at their terminal: they never count as live operations or toward
  `queue_full`. The term, epoch and generation fences end only `SubmitWork`/`CancelWork` operations and never stop
  or detach the child; an agent restart ends previous-epoch lifecycle operations `worker_restart` (the child was
  killed).

**States** (not durable; every start begins `unloaded`): `unloaded`, `loading`, `ready`, `draining`, `unloading`,
`failed`. There is no `preparing` on the agent. Transitions run under the journal lock.

| Call / event | In state | Result |
|---|---|---|
| `LoadModel` | `unloaded`, `failed` | probe `base_url` once: a listener there -> `failed`, op `backend_port_in_use`, no spawn; else spawn -> `loading` (after `backend_child.json` is written). Until `loading` is reported, a further Load joins and Drain/Unload are `model_busy` although the inventory still shows `unloaded`/`failed` |
| `LoadModel` | `loading`, `ready` | joins the current load (ends with it) |
| `LoadModel` | `draining` | cancels the drain -> `ready`; pending Drain and Unload ops end `cancel_requested`, the Load ends `""` at once |
| `LoadModel` | `unloading` | `FAILED_PRECONDITION model_busy` (before the journal) |
| confirmed identity with the configured `served_model_name` | `loading` | `ready`; the generation bump (`agent_state.json`) and the `""` terminal of every pending Load are written under one journal lock |
| child exit, `startup_timeout_ms`, served-name mismatch | `loading` | stop the group -> `failed`; pending Loads `model_load_failed`. The deadline is final: a confirmation at or after the deadline instant is ignored |
| `DrainModel` | `ready` / `draining` | -> `draining` (joins); the op ends `""` once no live operation is left; the state stays `draining` |
| `DrainModel` | other | `model_busy` |
| `UnloadModel` | `ready`, `draining` | -> `draining`; no live operation left -> `unloading` -> stop -> `unloaded`, op `""` |
| `UnloadModel` | `unloading` | joins |
| `UnloadModel` | `unloaded`, `failed` | -> `unloaded`, op `""` at once |
| `UnloadModel` | `loading` | `model_busy` |
| child exit | `ready`, `draining` | stop the group -> `failed` (a pending Unload: `unloaded`, op `""`); pending Drains `backend_exited`; live SubmitWork ops `backend_protocol_error` |

No drain timeout: work ends by its own deadline (at most 600 000 ms). A new `SubmitWork` key needs `open` admission
**and** model state `ready`: `UNAVAILABLE model_draining` while draining, `UNAVAILABLE backend_unconfirmed`
otherwise; journaled retries are answered as before.

**Stop.** `kill(-pgid, SIGTERM)`, wait up to `stop_timeout_ms` for the leader, `kill(-pgid, SIGKILL)`, reap the
leader, wait up to `stop_timeout_ms` more for `kill(-pgid, 0)` to fail with `ESRCH`; only then is the stop
confirmed and `<state_dir>/backend_child.json` deleted, and every terminal `SubmitWork` op of that child
cleanup-acknowledged. A stop that is not confirmed (the group survives, or the record cannot be removed) poisons
the agent: the model goes `failed`, pending Unload/Drain ops end `poisoned`, the group stays recorded and the next
agent start retries the kill.

**Agent start.** Before the listener and the identity probe, a recorded group of a previous process is killed
(every process whose pgrp is the recorded pgid and whose starttime is at or after the recorded leader's), then the
file is deleted; nothing is signalled when the recorded leader pid now has another starttime. Survivors after
`stop_timeout_ms` exit the agent with status 1; every other startup error exits 2, including an agent running as
root without `[backend.launch] uid`/`gid` and a non-root agent with `uid`/`gid` set.

**Inventory** (`schema_version` 2; 1 is refused). Top-level `tp_size`, `kv_cache_dtype`, `max_running_requests`
and `instance_id` are `null` unless a backend identity is confirmed and (managed) the state is `ready`, `draining` or
`unloading`. `models[]` holds one item `{served_model_name, lock_sha256, state, instance_id, model_generation}`;
attached mode reports `lock_sha256: null, state: "attached"` once confirmed. A managed agent always answers
`DescribeResources`; attached mode answers `backend_unconfirmed` before the first confirmation. The inventory hash
covers `state`, so every state change sends a Ready worker back to Joining.

A backend crash observed through a `SubmitWork` transport error ends a pending Drain `backend_exited`: a leader that
is exiting (`PF_EXITING` in `/proc/<pid>/stat`, set before the kernel closes its sockets) is not alive, so the Drain
waits for its exit report.

## Limits

| Limit | Value |
|---|---|
| Control decode limit (`WorkerRegistryService`, `WorkerService`) | 1 MiB |
| Inference decode limit (`WorkerInferenceService`) | 8 MiB |
| `WorkEvent.bounded_output_event_json` | 64 KiB |
| `bounded_resource_inventory_json` | 64 KiB, `schemas/worker_inventory.schema.json`, unknown fields refused |
| `ListOperations` page | 1..=256 |
| Work deadline cap | 600 000 ms |

## Reason codes

`stale_term`, `stale_epoch`, `stale_generation`, `wrong_cluster`, `wrong_worker`, `wrong_coordinator`,
`epoch_lost`, `unknown_worker`, `unsupported_revision`, `replayed_heartbeat`, `lease_expired`, `queue_full`,
`journal_full`, `poisoned`, `operation_conflict`, `payload_digest_mismatch`, `ticket_mismatch`,
`unknown_operation`, `quota_exhausted`, `permission_denied`, `unauthenticated_peer`, `worker_restart`,
`coordinator_restart`, `model_generation_superseded`, `client_backpressure`, `cancel_requested`,
`deadline_exceeded`, `malformed_request`, `backend_unconfirmed`, `lock_not_approved`, `model_unmanaged`,
`model_busy`, `unsupported_placement`, `model_draining`, `model_load_failed`, `backend_exited`,
`backend_port_in_use` (`src/fed/wire.rs`). `backend_unconfirmed` is the `UNAVAILABLE` answer to `DescribeResources`
before an attached agent has a confirmed backend identity, and to a new `SubmitWork` while a managed model is not
`ready` or `draining`.
