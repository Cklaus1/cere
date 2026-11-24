# Bounded resumable output streams

CERE should support explicit stream resumption after the basic M1 streaming/cancellation contract is stable.

Each committed output event has a request identity and monotonic event sequence. A client that disconnects may reconnect with the same authorized request and last-seen cursor when the request was submitted with continue-on-disconnect policy.

## Guarantees

- Reconnection replays committed events from the **same** request; it does not regenerate them.
- Replay retention is bounded by bytes and time.
- Expired or unauthorized cursors fail visibly.
- Detached computation has an explicit deadline and budget.
- Client delivery is at-least-once; clients deduplicate stable event IDs.
- Tool side effects use their own idempotency contract. A replayed tool delta must not execute a tool again.

Stream replay does not recover computation after worker loss. Worker/session recovery is a separate checkpoint/replay concern.

## What shipped (M1, `cered`)

The M1 front end implements this design for streamed chat requests; the wire contract is in [API.md](API.md#stream-resume).

- Opt-in per request: `X-Cere-On-Disconnect: continue`. Event identity is the existing `cere.seq`, sent as the SSE `id:` line. Resume is `GET /v1/cere/requests/{rid}/events?after=<seq>` (or `Last-Event-ID`) in the request's own bearer scope.
- Replay is from the same request; it never regenerates. Bounds: bytes per request, bytes per process and time after the terminal (`api.resume_buffer_bytes`, `api.resume_buffer_total_bytes`, `api.resume_retention_ms`). Expired cursors fail `410 cursor_expired`; unknown or other-scope rids `404`; requests that were not sent in continue mode `409 not_resumable`.
- Detached computation keeps the request's own deadline and output budget.
- Delivery: a reader gets the retained events after its cursor, then live events; a client that resumes from its last seen `id` sees each event once, and deduplicating by `id` covers any other overlap. One live reader at a time: a new reader supersedes the old one, which gets `: superseded` and a close.
- Buffers are memory only: a `cered` restart loses every buffer (a prior-epoch rid answers `410`), and the interrupted request is recovered as an orphan (`worker_restart`). Tools are not implemented, so the tool-idempotency guarantee has no M1 counterpart.
