# Multi-node CERE

Status: architectural contract for v1.x. M3.5 adds a node agent, `cere-worker`, and the coordinator's worker registry in `cered`, over mTLS gRPC; see [WORKER_PROTOCOL.md](WORKER_PROTOCOL.md) for the implemented subset.

## Goal

Allow multiple PCs with GPUs to join one CERE-managed resource pool without changing the cognitive/session/model abstractions. The first implementation is still one machine / one GPU. Multi-PC **management** is a v1.x feature. Cross-PC **cooperative inference for one forward pass** is v2 research.

```text
                         CERE coordinator
                  global scheduling / registry
                              |
                   Tailscale private network
                    application RPC control
          +-------------------+-------------------+
          |                   |                   |
      worker-a            worker-b            worker-c
      2x PRO 6000         2x 5090             future GB300
          |                   |                   |
     local backend        local backend        local backend
```

A worker may be local to the coordinator or remote. The coordinator schedules against the same logical `Worker` contract either way.

## Control plane versus data plane

### v1.x control plane

Normal CERE operations use a typed, versioned RPC protocol over the private tailnet:

- worker registration and version negotiation;
- heartbeat / epoch / health reporting;
- resource and topology inventory;
- model/artifact inventory;
- load, drain and unload model lifecycle requests;
- work submission and cancellation;
- request/worker telemetry streams.

Tailscale provides encrypted network connectivity and deployment identity/policy. CERE does not treat possession of a reachable IP address as authorization; coordinator-side policy still decides which workers are admitted to the cluster.

`proto/cere_worker.proto` is compiled into Rust bindings by `crates/cere-proto` (tonic). The implemented subset (registry, heartbeat, inventory, submit/cancel/stream, operation status, and since M3.6 `LoadModel`/`DrainModel`/`UnloadModel`) is described in [WORKER_PROTOCOL.md](WORKER_PROTOCOL.md); only `StreamTelemetry` answers `UNIMPLEMENTED`.

### SSH

SSH is an operator escape hatch for installation, upgrades, log inspection and emergency recovery. It is intentionally not the CERE orchestration protocol. Avoid `ssh worker "load-model ..."` or shell-based heartbeat/scheduling logic.

### v2 data plane

If CERE later performs one model operation across machines, the high-rate data path may require NCCL/NIXL/UCX/RDMA/RoCE/InfiniBand or another measured transport. That accelerated data plane is separate from the RPC control plane. No current contract claims cross-node tensor, expert, context or pipeline parallelism.

## Worker identity and epochs

Each worker advertises a stable logical `worker_id` and a monotonically increasing `worker_epoch` for a new agent incarnation. In M3.5 the agent is `cere-worker`, which persists epoch + 1 in its state directory at every start before it serves or registers.

Rules:

1. A stale epoch cannot replace or complete work owned by a newer epoch.
2. A transient heartbeat lapse moves READY -> SUSPECT and removes the worker from new scheduling.
3. Recovery from SUSPECT with the same epoch is permitted before LOST.
4. LOST is terminal for that epoch. Rejoining requires a higher epoch.
5. DRAINING accepts no new work but may finish active work.
6. Coordinator restart must reconstruct worker/model/session authority from durable metadata before admitting new work; workers never self-promote authority from local cached state.

These rules prevent a rebooted worker from returning a late completion that corrupts a newer request attempt.

Implemented (M3.5, `cered` worker registry): rules 1, 3 (as a new Joining round, below), 4, 5 for operator drain intent, and the term half of rule 6: every `cered` start is a new coordinator term, every remote worker starts in Joining, and the agent fences older-term operations. A worker reads Joining until its first accepted Register of the term; an accepted Register at a higher epoch is journaled before the reply. A heartbeat that finds the worker Suspect, or reports a new generation, inventory or an admission state other than `open` (`fenced` while draining), sends it back to Joining for a new round (inventory, pins, reconciliation, smoke; a draining worker skips the smoke because its agent refuses work) instead of straight to Ready. The rebuild of request authority after a coordinator restart for remote work depends on remote dispatch, which is in progress.

## Resource advertisement (design; M3.5 subset)

A worker reports measured/probed local resources rather than marketing capacities:

- GPUs/device memory and current free/reserved bytes;
- CPU/NUMA and host memory;
- local NVMe/artifacts;
- backend capability profiles;
- resident model instances;
- temperatures/power/health where available;
- local topology and P2P capabilities;
- measured control-path latency for diagnostics.

The global ResourceGraph can additionally contain `machine` nodes and `network` links. Network values are evidence-tagged; an unmeasured Tailscale path must not be used as if it were an RDMA fabric.

M3.5 and M3.6 implement a subset: `DescribeResources` returns the bounded inventory JSON (`schemas/worker_inventory.schema.json`, `schema_version` 2 since M3.6: served model name, `tp_size`, `kv_cache_dtype`, `max_running_requests`, `instance_id` (all null while no backend identity is confirmed), model generation, admission state, the worker's one model with its approved `lock_sha256` and model state (`attached` for an agent without `[backend.launch]`), and empty GPU and artifact lists), hashed and checked against the agent-reported `inventory_sha256` and the `[federation.workers.expect]` pins. The other items above are design.

## Scheduling policy for federated workers (design; M3.5 dispatches a request to one Ready worker through admission)

For v1.x, place a WorkUnit wholly on one eligible worker. Selection considers:

1. exact compatible state/session locality;
2. already-resident requested model;
3. local artifact availability and expected load time;
4. device memory admission and backend native capability;
5. queue length / latency objective / priority;
6. worker health and epoch;
7. network control latency as a secondary tie-breaker.

Prefer sticky session routing: once a session has hot prefix/KV state on a worker, keep subsequent turns there unless waiting is demonstrably worse than reconstruction/migration.

Do not migrate an active session simply to equalize GPU utilization. Initial scale-out should route **new** sessions to other workers.

## Team pools (net-teleport design note 4.4, S6)

One coordinator `cered` on an always-on host; each member's GPU host runs `cere-worker` with `owner = "<scope>"`, and
the coordinator lists it in `[[federation.workers]]` with the same `owner`, a `tier` (`own`, `team`, `provider`) and
optional `owner_only` windows. One person is one auth scope; `token_envs` gives it one token per device. Keys and
rules: [CONFIGURATION.md](CONFIGURATION.md) (`[admission]`, `[auth]`, "Federation"); request side:
[API.md](API.md) "Team pools".

- **Placement (D-T7).** A worker is eligible for a scope only when its tier for that scope (`own` for its owner, else
  its `tier`; a local worker is this install's `server.tier`) is in the scope's `allowed_tiers` (default `["own"]`).
  The filter applies to `Any` and `Pinned` targets at the grant (a pin to a disallowed worker waits for an allowed one
  of its deployment). No allowed worker: the request queues, then fails `503 no_allowed_worker`; it never spills.
- **Owner priority (Q-T6).** When a slot frees on an owned worker and its owner has a grantable waiter, the owner is
  granted first, ahead of the round-robin. No rows are reserved. On an SGLang worker the owner waits at most one
  non-owner job, bounded by `admission.owner_wait_cap_ms` (default 120 s: a non-owner's request granted an owned
  worker gets its deadline lowered to at most that long minus 500 ms after the grant, the 500 ms left for its
  abort and the slot's release). Federation workers are SGLang agents only
  (`deployments.kind = "sglang_http"`), so the native D8 park of a non-owner generation does not arise in this build.
- **Classes (C17).** `interactive` (default) is served before `batch` across scopes; while both wait, `batch` keeps
  `admission.batch_floor` (default 10 %) of the grants. The weighted round-robin orders scopes within a class and
  `X-Cere-Priority` a scope's own queue.
- **Quotas (Q-T7, C18).** Per-scope `per_scope_inflight`, `max_sessions_per_scope` and `max_batch_inflight`; no token
  budgets.
- **Opt-out windows (C19, Q-T8).** Inside an `owner_only` window, or while the agent's local switch is on
  (`cere-worker owner-only on|off --config <toml>`, reported in the heartbeat and exposed as the `/metrics` gauge
  `cere_worker_owner_only`), only the owner is granted the worker; in-flight non-owner work finishes, bounded by
  `owner_wait_cap_ms`. The agent itself refuses other scopes' new `SubmitWork` (`owner_only`), since it sees each
  operation's `scope_id` (F2).
- **Evidence.** `CERE_LIVE=1 python3 scripts/live/federation_live.py --pool` runs the bar S6 checks against mock
  backends on this host only (no GPU, no node-b) and writes a JSON and a Markdown report.

## Model management across machines

The global model manager maintains logical desired residency while each worker retains physical load/unload authority for its device allocations.

**Built (M3.6), operator-triggered only.** Each worker belongs to one approved deployment (below). `POST /v1/cere/workers/{id}/load` or `/unload` ([API.md](API.md#operator-routes-load-and-unload)) journals the operator's intent (`desired` `resident` or `unloaded`) for one managed remote worker; the coordinator then sends that worker's agent `LoadModel` (naming only its own deployment lock) or `UnloadModel`, at most once per `(intent_seq, worker_epoch, term)`, and polls the operation. The agent loads only the lock in its own `[backend] lock_sha256` and launches only the argv of its own `[backend.launch]` ([WORKER_PROTOCOL.md](WORKER_PROTOCOL.md#model-lifecycle-m36)); it drains, then stops the backend's process group on unload. The worker turns Ready once its model is `ready` under its deployment's lock and passes the M3.5 Joining round; **new** sessions then use it (fewest in-flight), sessions pinned elsewhere stay. An unload takes the worker out of admission before the route answers; requests running there finish and its pinned sessions re-pin on their next turn. After an agent restart (new epoch) a `resident` intent loads again; a `failed` model is loaded again only by the operator's next `load`. One model per worker; no SSH, paths, URLs or commands in any request. An agent without `[backend.launch]` keeps the M3.5 attached mode (lifecycle calls answer `model_unmanaged`).

**Worker -> deployment binding (M5, m5 design note D2(e), D6, D8, D9).** One `cered` serves 1..16 `[[deployments]]` entries ([CONFIGURATION.md](CONFIGURATION.md)); every local replica and every `[[federation.workers]]` worker belongs to exactly one entry. A local replica belongs to the entry that lists it; a remote worker to the entry whose `served_model_name` equals its `[federation.workers.expect] served_model_name` (no such entry is a config error, also with one entry). A remote worker keeps its own `expect` pins, checked in its Joining round as before; the coordinator-side residency (`LoadModel`, the inventory lock check, `lock_mismatch`, `mark_ready`) uses its entry's lock, so with two deployments a managed `LoadModel` names the lock of the worker's own deployment. Admission routes a request only to Ready workers of the request's deployment and re-pins a session owner outside it inside it; a migration to another deployment's worker is `409 model_mismatch` ([API.md](API.md#operator-routes-session-owner-and-migration)). The worker RPC and the agent are unchanged.

**Real model (N3, 2026-10-08).** Federation, managed placement and migration were run live with `google/gemma-4-e4b-it` on two independent SGLang servers of one deployment: the GPU1 twin (local worker `gpu1`) and node-b's 5090 (`worker-a`, attached or managed) ([VALIDATION_REPORT.md](VALIDATION_REPORT.md)). The twins share one API and one served model name but nothing else: KV and prefix cache stay per server, sticky routing keeps a session on its worker, and a turn that moves (re-pin, migration, force) re-prefills from the client's history text. No cross-worker determinism is claimed: the same request on the two workers may answer differently.

Not built: automatic placement under queue pressure (the example below), several models per worker, model swaps, artifact locality, residency leases and idle eviction, telemetry streaming. Several models per `cered` exist since M5 only as separate deployments on separate workers.

Known limit: while a worker is not Ready its `/readyz` `model.state` can lag the agent by up to one probe round (5 s).

Example:

```text
30 coding agents need model A
  worker-a: A READY
  worker-b: A READY
  worker-c: A artifact-local, UNLOADED

worker-a queue rises -> coordinator may request A load on worker-c,
then route new sessions there after READY acknowledgement.
```

Model loading remains single-flight per `(deployment_lock, worker, device placement)`. A remote load request is not permission to download arbitrary URLs; only approved immutable locks/artifacts are eligible.

## Failure behavior

Heartbeats use configured thresholds:

- READY -> SUSPECT after `suspect_after_ms` with no heartbeat;
- SUSPECT -> LOST after `lost_after_ms`;
- SUSPECT recovers to READY on a valid same-epoch heartbeat;
- LOST requires a new epoch to register again.

As implemented in the M3.5 registry, timers run on coordinator receive time: Ready turns Suspect after `suspect_after_ms` of silence or when its admission lease (`admission_lease_ms` after the last accepted heartbeat) expires; Ready, Suspect and Joining turn Lost at `lost_after_ms` after the last accepted heartbeat (absolute bound); a Suspect worker that heartbeats again passes a new Joining round before Ready. The agent independently fences its own admission when its lease expires. In-flight remote requests (M3.5): a work stream broken on the transport parks the request (slot charged, client stream open) until the worker turns Lost (`worker_lost`), registers a higher epoch (`worker_epoch_superseded`), reconnects at the same epoch (`backend_protocol_error`; no stream resumes) or the request is cancelled. Nothing is rescheduled. A request whose work may still run on the agent keeps its slot until the agent reports the operation cleaned up or the request's retention bound passes; the release is journaled. The general rules:

When a worker is SUSPECT/LOST:

- no new work is scheduled there;
- in-flight requests are not immediately assumed failed while within the configured grace period;
- after LOST, only work with a defined safe retry/resume contract may be rescheduled;
- partially streamed outputs use attempt/event identities so clients can reject duplicate/stale data;
- tool side effects are never retried by the inference layer.

## Cross-install session teleport (NetTeleport S4)

L2 of net-teleport design note 4.2: a client moves a conversation's cached prefix state from one native `cered` to
another install it trusts, over S1's sealed-blob routes ([API.md](API.md#state-teleport-netteleport-s1)). Nothing
here touches the worker protocol: the blob never travels on the control path, only between the client and the two
loopback client APIs. The client is `scripts/live/teleport.py` (Python stdlib only, any device).

- **Key custody (D-T5).** `teleport.py keygen` writes 32 random bytes as 64 hex digits to `~/.config/cere/teleport.key`
  (or `$CERE_TELEPORT_KEY_FILE`): file 0600, directory 0700, never overwritten; a group- or world-readable key file is
  refused. The key rides each call in `X-Cere-State-Key`; neither install stores it and the tool never prints it. An
  OS keychain is future work.
- **Tier policy (D-T7).** `--dest-tier own|team|provider` says what the destination is to the user, and every move
  is refused before any byte moves unless that tier, and a vault host's `--vault-tier`, is in the allowed tiers
  (`--allowed-tiers`, else `$CERE_TELEPORT_ALLOWED_TIERS`, else `own`, Q-T5). That refusal exits 2 with `"mode":
  "refused"` and does not suggest a replay, which would send the conversation to the same tier. Both installs check
  again (S1: the blob's allowed tiers at the source, the install's `server.tier` at the destination).
- **Transfer.** Both state routes are on the loopback client API (F18), so endpoint URLs must be `http://127.0.0.1:<port>`.
  A remote install is reached through an ssh tunnel the tool opens (`--source-ssh HOST` / `--dest-ssh HOST`: `ssh -N
  -L 127.0.0.1:<free>:127.0.0.1:<port>`, `ExitOnForwardFailure`). `move --via direct` pipes the export response into
  the import request (no copy at rest). `--via vault --vault SPEC`, and `export --to-vault` / `import --from-vault`
  when the two installs are not online together, park the sealed blob in a vault.
- **Vault.** `scripts/live/teleport_vault.py`: a directory on the client (`/abs/dir`) or on another host over
  ssh/scp (`host:/abs/dir`, e.g. node-b over the tailnet, a GPU-less relay, Q-T2). It holds sealed `CERESTB1` files
  only (mode 0600, written as a part file then renamed), named by a random id, never by session or content; it never
  sees the key. `teleport.py vault scan --vault SPEC` checks at rest, on the vault host itself (the helper runs there
  as `python3 -` with its source on stdin, so only the verdict crosses the link): each file is a format-1 `CERESTB1`
  blob whose size matches its header, with no `CERESNP1` magic, no run of 16 little-endian u32 token ids, and at least
  7.99 bits per byte in every sampled 1 MiB window. An import from the vault removes the vault copy unless `--keep`.
- **Decision rule (4.2 L2).** `move` ships state iff export + blob bytes / link rate + import + the first turn after
  import < the destination's re-prefill (its cold TTFT), or `--contended` (the destination GPU serves other rows, so a
  re-prefill would stall them, F17). Otherwise it answers `"mode": "replay"` with the reason and the client sends its
  next turn to the destination, which cold-prefills (L0); a refusal by either install also completes as replay with
  the install's code (D-T1). The estimates are the measured table `scripts/live/teleport_estimates.json` (written by
  `scripts/live/teleport_bench.py --write-table`), interpolated linearly in tokens per board, with three links:
  `loopback` (one host: the transfer is inside export and import), `tailnet-hop` (one direct hop between two hosts)
  and `tailnet-relay` (push to and pull from a vault). `--link-mbps` prices another link; a board without rows
  answers replay. `teleport.py decide --tokens N --link L` prints the rule's answer alone.
- **Output.** One JSON object on stdout: mode, reason, timings, sizes, refusal codes and vault names; never the key,
  token ids or content (4.1).

```text
teleport.py move --source http://127.0.0.1:18309 --source-ssh laptop \
    --dest http://127.0.0.1:18309 --source-token-env CERE_TOKEN_ME --dest-token-env CERE_TOKEN_ME \
    --model "$MODEL" --messages chat.json --prompt-tokens 72000 --dest-tier own --link tailnet-hop
```

**Agent-harness hook (not built here).** An agent harness hooks the tool at a session handoff: before it sends the next turn
to a different install, it runs `teleport.py move` with the session's messages (the chat body it would send, minus
the new turn) and `--prompt-tokens` from the last turn's `usage.prompt_tokens`, then sends the turn to the
destination whatever the answer (`state`: the turn is a prefix hit; `replay` or a transfer failure: it cold-prefills).
A `refused` answer (exit 2) means the user's tier policy forbids the destination: the harness keeps the session where
it is. The hook passes bearer tokens through environment variables, never argv, and keeps one key file per user.

**Measured (bar S4, K3).** an internal benchmark report, GPU0, Qwen3.8-Flash-Next snapshot
profile, two installs with distinct `state_dir`s and server secrets, run one after the other (install A alone held
80,660 of 97,887 MiB; the executor may take 81,920 MiB, so two loads do not fit). Medians of 3 prompts:

| tokens | blob | export | import | first turn | re-prefill | loopback e2e | node-b relay e2e |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17,984 | 295 MB | 0.43 s | 0.48 s | 0.09 s | 1.75 s | 0.90 s | 6.96 s |
| 72,000 | 1.00 GB | 1.42 s | 1.62 s | 0.19 s | 7.02 s | 3.04 s | 22.04 s |
| 128,000 | 1.73 GB | 2.37 s | 2.80 s | 0.29 s | 12.81 s | 5.17 s | 37.56 s |

The tailnet path to node-b is direct (LAN, rtt 1 ms; 1 GbE (inferred from the rate)): about 96 MB/s each way, 52.8 MB/s effective through the
vault. Same-host teleport beats re-prefill at every size (about 2.5x at 128K); a direct hop (105.7 MB/s) and the relay
lose at every size, extrapolated to 200K too, so **K3 fires**: WAN L2 is skipped, S4 ships same-host (loopback) and
vault teleport, and the rule answers replay across hosts. All 9 blobs on node-b scanned sealed. Every imported first
turn matched the source's warm answer. One prompt of exactly 128,000 tokens missed after a successful import: a
lookup caps the prefix at the last 64-token page before the prompt's end (one row computes the last logits), so an
imported state as long as the whole prompt cannot serve a resend of that same prompt; the next turn, which appends
tokens, hits.

## Security

Recommended deployment posture (M3.5 implements mTLS under a private cluster CA with a fixed principal role table, canonical loopback/tailnet listen literals and the identity, revision and epoch checks; tailnet Grants/ACLs are deployment policy that CERE does not configure or check):

- workers/coordinator live on a Tailscale tailnet;
- Tailscale Grants/ACL policy limits the worker RPC port to authorized coordinator identities;
- CERE performs its own cluster allowlist/version/epoch checks in addition to network reachability;
- no public worker RPC listener;
- metadata-only telemetry by default;
- no SSH command execution through the scheduler;
- no cross-authorization-scope cache or session reuse.

The tailnet is transport/security infrastructure, not the CERE authorization database.

## Version boundary

### v1 architecture (now)

Contracts include Worker identity, epoch, ResourceGraph ownership and transport-independent scheduling interfaces even when `worker_count=1`.

### v1.x implementation

Federated independent workers over Tailscale RPC. M3.5 builds the worker registry in `cered`, the `cere-worker` agent, the mTLS gRPC protocol and coordinator dispatch of client requests to remote workers (live-checked with mock backends on 2026-10-07).

### v2 research

Cross-node state migration, distributed State Fabric, accelerated GPU data plane, cooperative inference and coordinator HA. Each is separately gated by correctness and measured benefit.

## v0.6 authority correction

Worker health and operator intent are independent: DRAINING still times out, and neither recovery nor re-registration at a higher epoch clears drain intent; only an explicit operator undrain does. Admission, completion and eligibility age the worker on the coordinator clock first. Registration is not readiness. All health timers use coordinator-local receive time and increasing heartbeat sequences. RPC commands use an application principal, scoped operation ID, coordinator term, worker epoch and model generation; matching retries return the stored operation. Workers stop accepting new work when their admission lease expires.

Ordinary prompts and generated-output streams may travel over the tailnet; the prohibited substitution is sending arbitrary bulk model/KV/collective traffic through the small control-message path. The full normative protocol is [WORKER_PROTOCOL.md](WORKER_PROTOCOL.md).
