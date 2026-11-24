# State Fabric and Session DAG

## Three identities

1. **Task identity:** common purpose, repository and ancestry. This is metadata, not KV compatibility.
2. **Input identity:** exact ordered token prefix plus model-conditioning inputs.
3. **Execution identity:** artifact/plugin/layout, adapters, numerical mode and precision/position semantics.

Authorized scope is mandatory. Prefix cache keys are scoped keyed hashes of the parent key, exact token block and compatibility fingerprint. The authentication layer derives an unguessable per-scope secret; clients cannot select another scope. Store neither the secret nor raw tokens in ordinary telemetry.

`input_fingerprint` must include tokenizer, chat template, tool-rendering rules, position/RoPE/mask semantics, media preprocessing and content, adapters, and injected embeddings where used. `model_fingerprint` covers actual weights, not a mutable model alias. `execution_fingerprint` covers runtime state layout and numerical compatibility.

## Sharing rules

Immutable full attention pages can be shared through references. A writer receives a private tail before mutation. Recurrent/SSM/hybrid state is checkpointed at supported boundaries and may require copying a fixed-size state and replaying a suffix. Fork metadata may be cheap even when physical state copy is not. Reject impossible rewind/restore points.

Do not merge divergent attention/recurrent state. A reviewer that receives new instructions before the common source text may share less than expected; only an exact prefix counts. Similar code/language does not authorize expert-route or KV equivalence.

## Checkpoint bundle

A continuation checkpoint records exact committed tokens, prefix boundary, recurrent state, KV layout, RNG counters/stream ID, sampler/penalty history, grammar state, stop matcher, tokenizer/template, draft state and worker/execution identity. Speculative state is transactional: commit verified tokens only; restore all coupled state after rejection.

## Memory lifecycle

Logical references are not the same as GPU-in-flight leases. Buffer states are resident, copying, parked, evictable or invalid. Evict only after all completion fences and references permit it. Track resident and logical bytes separately; shared prefix bytes are counted once physically.

A memory broker assigns budgets to typed managers for immutable weights, KV pages, recurrent state, draft state, graph buffers and workspace. Transfer a state object only to a compatible backend or reconstruct from text. Peer memory is not automatically faster than host memory and may not be accessible.

## Tool pauses

Estimate tool duration, reuse probability, queue pressure and transfer cost. Keep hot state for short waits; spill only when it creates more useful capacity than it costs. Always permit eviction under genuine pressure through explicit restore/replay semantics. `park` is a policy action, not an unconditional tool-call hook.

## Durability

CPU session metadata can be durable without preserving GPU pointers. Journal revisions and request outcomes; checkpoints are atomic, checksummed, immutable and permission-restricted. After worker loss, restore compatible saved state or replay committed tokens. Bound recovery time and replay token budgets. Replaying tokens must not rerun external tools.

The Python reference SessionStore demonstrates only immutable **token metadata** and scope/revision checks. It is not a KV allocator and does not claim GPU copy-on-write.


## Node locality

State identity is global metadata; physical state residency is worker-local in v1.x. CERE may know that a session's compatible state lives on worker A without treating it as remotely addressable memory from worker B. Initial federation routes later turns back to A. Cross-node transfer requires a future explicit snapshot/transport capability and transactional ownership change. M4 built the ownership change without the transfer: an operator may move a session's durable owner to another Ready worker (local or remote), and the destination replays from the messages of the next turn ([MULTI_GPU.md](MULTI_GPU.md#migration-transaction-m4)).

Native snapshot (M2.11, m211 design note): the native executor can copy an idle session's state (token ids, KV, GDN conv/ssm, QSA keys/ring) into a pinned host arena of the model (`[deployments.native] snapshot_host_bytes`, 0 in the shipped config) and restore it into a fresh session bit-exactly. `cered` uses it for U1 preemption (`preempt = true`: a request without a free slot parks the generation with the most committed tokens and resumes it FIFO). Snapshots live only in pinned host memory: never written to disk, a log or a report (digests only), and they are not durable: a worker restart loses them. Idle-session swap is M2.10's host tier on this mechanism; NVMe spill and cross-device resume stay off.

State teleport (NetTeleport S1, net-teleport design note 4.2 L2, 4.3): `POST /v1/cere/state/export` and `/import` ([API.md](API.md#state-teleport-netteleport-s1)) move one cached prefix state across installs as an explicit, client-carried blob. The `CERESNP1` image leaves the server only sealed (AES-256-GCM under the client's key, which the server never stores). The image is opened into one `mlock`ed buffer and inserted into the importing scope's own namespace, so ownership does not move and nothing is shared across scopes. Admission is by `state_compat_id` (library, layout, numerics-changing levers) and the trust-tier policy (the scope's `allowed_tiers` at the source, the install's `server.tier` at the destination). Any refusal imports nothing and leaves replay from the messages as the fallback (D-T1).

Warm restart (NetTeleport S3, L1, net-teleport design note 4.2 L1, C7; opt-in `[state] park_on_stop`, [CONFIGURATION.md](CONFIGURATION.md)): the native backend keeps the prompt token ids of its most recent text-only, namespaced requests in memory only (`state.park_max_sessions`; a session's next turn replaces its previous one). At a graceful stop, after the drain, `cered` exports the deepest cached prefix state of each, most recent first, until `state.park_deadline_ms` or `state.park_max_bytes`, and writes each sealed into `<ledger.state_dir>/parked/` (0700 directory, 0600 files, fsync'd). A parked file is `CEREPRK1`, the session's 16-byte namespace, then the S1 `CERESTB1` blob sealed under a key HMAC-derived from the install's `server_secret` and that namespace (D-T5), so a file moved to another namespace or another install fails its key check. At the next start, after the model loads and before the backend serves, every image whose library, capability profile, layout, `state_compat_id` and tier match strictly (L1 keeps the profile check, D-T3) is imported into its own namespace; every file is deleted, imported or refused, and only counts, bytes, refusal codes and timings are logged (`cered state park:` / `cered state unpark:`). A refused image costs a replay: the client re-issues its turn and the prompt is prefilled cold. SIGKILL or a crash parks nothing.
