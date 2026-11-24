# Architecture

The Work/State/Model/Resource/ExecutionPlan abstractions are retained. v0.6 preserves those ownership boundaries, adds explicit model-residency lifecycle management, and makes Worker/Node location an explicit scheduling dimension.

```text
MiCode: tasks, tools, permissions       Axon: strategies, bounded priorities
                   \                  /
                      CERE control
          request/session ledger | budgets | placement | telemetry
                             |
                   backend capability boundary
                 /                              \
       Backend-managed lane                 Native lane (later)
       SGLang / other server                CERE local scheduler
       owns KV + token batches              owns state allocator + batches
       pinned executable/image              C/CUDA executor ABI
                 |                              |
           one GPU initially             one GPU experimental
                             |
               second GPU = independent replica
```

Do not build a wrapper that serializes a native backend. CERE's logical Session DAG can route matching histories to the same backend but cannot claim physical page sharing unless the backend explicitly implements it. Exported snapshots are backend/version/layout-specific; a CERE ID is not a portable KV payload.

## Ownership

| Resource | Owner | Release authority |
|---|---|---|
| Agent goals and tool actions | MiCode/Axon | upstream policy |
| Request IDs/revisions/budgets | CERE control | terminal ledger transition |
| External-backend KV and batches | backend | backend completion/eviction |
| Native device allocations | worker allocator | completion fences + lease count |
| Immutable artifact bytes | artifact store | no active readers, approved GC |
| Long-lived metadata | durable control ledger | retention/authorization policy |

Typed managers coordinate through one memory budget broker. They do not treat weights, recurrent state, attention pages and graph scratch as interchangeable buffers.

The N-resource design avoids singleton assumptions but does not imply multi-rack consensus, distributed storage or all-to-all kernels are implemented. Replica mode is the first two-device feature, not runtime tensor-parallel switching.

## Federated workers

Above the backend boundary, the scheduler addresses a `Worker` abstraction rather than a GPU directly. In M1 the worker is local; M3 adds a second local replica; M3.5 allows remote independent workers over the same logical interface. `docs/MULTI_NODE.md` is normative for the intended v1.x boundary.

```text
CERE coordinator
   |-- local worker -> GPU0
   |-- local worker -> GPU1
   `-- remote worker -- Tailscale RPC --> remote GPU host
```

RPC carries control, lifecycle, work descriptors and telemetry. It is not the high-bandwidth collective data plane for future cross-node tensor/expert parallelism.
