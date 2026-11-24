//! The agent's operation runtime (m35 design note Q11, Q12, Q14, H2, H3): per-op output buffers, backpressure,
//! stream attach, cancel, deadlines and cleanup. Track WORK (build-loop-m35 design note).
//!
//! Every SubmitWork op this process accepted is a [`LiveOp`] in [`Ops`]'s table, with one pump task that runs the
//! backend call on [`Shared::backend`]: streamed chat = `chat_stream` + `next_chunk`, unary chat = `chat`. Each
//! backend chunk (the whole response for unary) becomes one event: `serde_json::to_vec` of the parsed value (H3(a)),
//! at most [`wire::WORK_EVENT_MAX`] bytes (larger: the op fails `backend_protocol_error`), `sequence` consecutive
//! from 1, `committed_token_offset` = sequence - 1. Events wait in a buffer of at most `output_buffer_bytes`; an
//! event handed to a `StreamWork` stream is *sent* and may be evicted to make room. When unsent events fill the
//! buffer the pump stops reading the backend, and after `output_stall_ms` without a drain the op ends
//! `client_backpressure`. A terminal op a stream has attached to leaves the table, and frees its buffer, once its
//! `cleanup` is journaled and no stream is attached (its stream broke before the terminal, or delivered it); a
//! later `StreamWork` gets just its terminal event, as for an op of an earlier process. An op no stream has
//! attached to yet keeps its buffer for the first attach.
//!
//! Terminal: natural completion (reason ""), a backend error ([`BackendError::op_reason`]), `deadline_exceeded` (receipt +
//! min(budget, [`wire::WORK_CAP_MS`])), `client_backpressure`, `backend_protocol_error`, `cancel_requested` (CancelWork)
//! and the fences, which LINK journals itself (`coordinator_restart`, `model_generation_superseded`,
//! `worker_restart`): the journal's terminal record wins, so an op never gets two terminal records or events. A
//! forced terminal drops the buffered events and follows the last sent one. After a terminal the pump keeps reading
//! the backend until its stream ends (the rid is aborted) and only then records `cleanup` (Q12). An op this
//! process does not hold (journaled by an earlier epoch) is cleanup-acknowledged by the `/get_load` rule when
//! `[backend] load_endpoint` is set, else at its worker-local deadline (H3(b)).
//!
//! At shutdown pumps drop their backend call without touching the journal (the next start ends the ops
//! `worker_restart`), and every open stream ends `UNAVAILABLE worker_restart` (H2).

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::Duration;

use cere_proto::worker::v2 as pb;
use futures_util::stream;
use serde_json::value::RawValue;
use serde_json::Value;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tonic::{Code, Status};

use crate::agent::journal::{Accepted, OpError, OpIdentity, OpKey, OpView};
use crate::agent::node::{now_ms, stopped, work_op, Shared};
use crate::fed::wire::{self, AdmissionState};
use crate::sglang::{BackendError, ChatStream, Rid};
use crate::validate::TOOL_CALL_PARSER;

/// A gRPC server stream as the handlers return it.
pub(crate) type RpcStream<T> =
    Pin<Box<dyn tokio_stream::Stream<Item = Result<T, Status>> + Send + 'static>>;

/// The H3(b) `/get_load` rule: two polls at least this far apart.
const LOAD_POLL: Duration = Duration::from_secs(1);
/// A backend call outlives the op deadline by this much, so the deadline (not the HTTP timeout) ends the op.
const BACKEND_GRACE: Duration = Duration::from_secs(1);

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn backend_reason(error: &BackendError) -> &'static str {
    error.op_reason()
}

fn protocol_error() -> &'static str {
    backend_reason(&BackendError::Protocol(String::new()))
}

/// What every event of an op carries.
#[derive(Debug, Clone)]
struct Header {
    worker: pb::WorkerIdentity,
    term: u64,
    key: OpKey,
    request_id: String,
    attempt: u64,
    ticket: String,
}

impl Header {
    fn of(shared: &Shared, view: &OpView) -> Header {
        Header {
            worker: pb::WorkerIdentity {
                cluster_id: shared.config.cluster_id.clone(),
                worker_id: shared.config.worker_id.clone(),
                worker_epoch: view.worker_epoch,
            },
            term: view.coordinator_term,
            key: view.key.clone(),
            request_id: view.request_id.clone(),
            attempt: view.attempt,
            ticket: view.ticket.clone(),
        }
    }

    fn event(&self, sequence: u64, body: Vec<u8>, terminal: Option<&str>) -> pb::WorkEvent {
        pb::WorkEvent {
            worker: Some(self.worker.clone()),
            coordinator_term: self.term,
            scope_id: self.key.scope_id.clone(),
            request_id: self.request_id.clone(),
            attempt: self.attempt,
            sequence,
            completion_ticket: self.ticket.clone(),
            committed_token_offset: sequence - 1,
            bounded_output_event_json: body,
            terminal: terminal.is_some(),
            reason_code: terminal.unwrap_or_default().to_owned(),
        }
    }
}

#[derive(Debug, Default)]
struct Buffer {
    /// Retained non-terminal events `(sequence, body)`, consecutive sequences.
    events: VecDeque<(u64, Vec<u8>)>,
    /// Bytes of `events`.
    bytes: usize,
    /// Sequence of the next event (the terminal event takes it).
    next: u64,
    /// Highest sequence handed to a stream.
    sent: u64,
    /// +1 per event handed to a stream for the first time (the stall clock).
    drains: u64,
    terminal: Option<String>,
    attached: bool,
    /// A stream has attached at least once.
    streamed: bool,
    /// The terminal event was handed to a stream.
    delivered: bool,
    /// The pump has ended (backend stream over, or shutdown).
    pumped: bool,
    /// The journal holds the op's `cleanup`.
    cleaned: bool,
}

/// What an attaching stream finds in a held op's buffer.
#[derive(Debug, PartialEq, Eq)]
enum Attach {
    /// A stream is attached: `ALREADY_EXISTS operation_conflict`.
    Conflict,
    /// Released (Q14): answered like an op no longer held, with just its terminal event (this local reason; the
    /// journal view the caller holds may predate the terminal).
    Released(String),
    /// Attached: the stream replays the buffer after its cursor.
    Attached,
}

impl Buffer {
    /// Q14's release rule: terminal, cleaned up, a stream attached once and none attached now.
    fn releasable(&self) -> bool {
        self.cleaned && self.terminal.is_some() && self.streamed && !self.attached
    }

    /// A stream attaches; `journal_cleaned`: the journal view the ticket was checked against holds the op's
    /// cleanup. [`Ops::cleanup`] journals the cleanup before it marks the buffer, so that view can be ahead of
    /// `cleaned`: the journal is the record, and the buffer takes it here, so an op the release rule frees is
    /// released for every attach that can see its cleanup, never replayed.
    fn attach(&mut self, journal_cleaned: bool) -> Attach {
        if self.attached {
            return Attach::Conflict;
        }
        self.cleaned |= journal_cleaned;
        if self.releasable() {
            if let Some(reason) = &self.terminal {
                return Attach::Released(reason.clone());
            }
        }
        self.attached = true;
        self.streamed = true;
        Attach::Attached
    }
}

/// One SubmitWork op held by this process.
#[derive(Debug)]
pub(crate) struct LiveOp {
    header: Header,
    rid: Rid,
    buffer: Mutex<Buffer>,
    /// +1 per buffer change; pumps and streams wait on it.
    version: watch::Sender<u64>,
}

enum Push {
    Stored,
    Terminal,
    Stalled,
}

impl LiveOp {
    fn new(header: Header, rid: Rid) -> LiveOp {
        LiveOp {
            header,
            rid,
            buffer: Mutex::new(Buffer {
                next: 1,
                ..Buffer::default()
            }),
            version: watch::Sender::new(0),
        }
    }

    fn changed(&self) {
        self.version.send_modify(|v| *v = v.wrapping_add(1));
    }

    fn is_terminal(&self) -> bool {
        lock(&self.buffer).terminal.is_some()
    }

    /// Records the terminal locally (first reason wins). A forced terminal drops the events never handed to a
    /// stream and takes the sequence after the last sent one.
    fn finish(&self, reason: &str) {
        {
            let mut guard = lock(&self.buffer);
            let b = &mut *guard;
            if b.terminal.is_some() {
                return;
            }
            if !reason.is_empty() {
                let sent = b.sent;
                while b.events.back().is_some_and(|(seq, _)| *seq > sent) {
                    if let Some((_, body)) = b.events.pop_back() {
                        b.bytes -= body.len();
                    }
                }
                b.next = sent + 1;
            }
            b.terminal = Some(reason.to_owned());
        }
        self.changed();
    }

    /// Stores one event; blocks while unsent events fill `cap` (sent ones are evicted oldest first), and gives up
    /// after `stall` without a drain.
    async fn push(&self, body: Vec<u8>, cap: usize, stall: Duration) -> Push {
        let mut rx = self.version.subscribe();
        let mut drains = None;
        let mut since = Instant::now();
        loop {
            rx.borrow_and_update();
            {
                let mut guard = lock(&self.buffer);
                let b = &mut *guard;
                if b.terminal.is_some() {
                    return Push::Terminal;
                }
                if drains != Some(b.drains) {
                    drains = Some(b.drains);
                    since = Instant::now();
                }
                let sent = b.sent;
                while b.bytes + body.len() > cap
                    && b.events.front().is_some_and(|(seq, _)| *seq <= sent)
                {
                    if let Some((_, old)) = b.events.pop_front() {
                        b.bytes -= old.len();
                    }
                }
                // An event larger than the whole buffer still goes in once nothing else is retained.
                if b.bytes + body.len() <= cap || b.events.is_empty() {
                    let seq = b.next;
                    b.next += 1;
                    b.bytes += body.len();
                    b.events.push_back((seq, body));
                    drop(guard);
                    self.changed();
                    return Push::Stored;
                }
            }
            tokio::select! {
                _ = rx.changed() => {}
                () = tokio::time::sleep_until(since + stall) => {
                    if drains == Some(lock(&self.buffer).drains) {
                        return Push::Stalled;
                    }
                }
            }
        }
    }

    /// The next event after `cursor` for the attached stream, or `None` while there is none yet. Handing out the
    /// terminal event marks it delivered and frees the buffer.
    fn next_for(&self, cursor: u64) -> Option<pb::WorkEvent> {
        let mut guard = lock(&self.buffer);
        let b = &mut *guard;
        if let Some(front) = b.events.front().map(|(seq, _)| *seq) {
            let index = (cursor + 1).saturating_sub(front) as usize;
            if let Some((seq, body)) = b.events.get(index).cloned() {
                let drained = seq > b.sent;
                if drained {
                    b.sent = seq;
                    b.drains += 1;
                }
                drop(guard);
                if drained {
                    self.changed();
                }
                return Some(self.header.event(seq, body, None));
            }
        }
        let reason = b.terminal.clone()?;
        let seq = b.next.max(cursor + 1);
        b.delivered = true;
        b.events.clear();
        b.bytes = 0;
        Some(self.header.event(seq, Vec::new(), Some(&reason)))
    }

    /// Terminal delivered and backend call over: nothing left to hold.
    fn done(&self) -> bool {
        let b = lock(&self.buffer);
        b.delivered && b.pumped
    }
}

/// Clears `attached` when a stream goes away, then offers the op for release ([`Ops::release`]).
struct Attachment {
    op: Arc<LiveOp>,
    ops: Weak<Ops>,
}

impl Drop for Attachment {
    fn drop(&mut self) {
        lock(&self.op.buffer).attached = false;
        self.op.changed();
        if let Some(ops) = self.ops.upgrade() {
            ops.release(&self.op);
        }
    }
}

/// Per-principal pool of open `StreamWork` streams (Q14, H1): the live-op bound + 1.
#[derive(Debug, Clone, Default)]
pub(crate) struct StreamPool {
    held: Arc<Mutex<HashMap<String, usize>>>,
}

/// One held stream slot; dropping it frees the slot.
#[derive(Debug)]
pub(crate) struct StreamSlot {
    principal: String,
    held: Arc<Mutex<HashMap<String, usize>>>,
}

impl StreamPool {
    pub(crate) fn try_acquire(&self, principal: &str, limit: usize) -> Result<StreamSlot, Status> {
        let mut held = lock(&self.held);
        let count = held.entry(principal.to_owned()).or_insert(0);
        if *count >= limit {
            return Err(wire::status(Code::ResourceExhausted, wire::QUOTA_EXHAUSTED));
        }
        *count += 1;
        Ok(StreamSlot {
            principal: principal.to_owned(),
            held: Arc::clone(&self.held),
        })
    }
}

impl Drop for StreamSlot {
    fn drop(&mut self) {
        let mut held = lock(&self.held);
        if let Some(count) = held.get_mut(&self.principal) {
            *count -= 1;
            if *count == 0 {
                held.remove(&self.principal);
            }
        }
    }
}

/// A validated SubmitWork.
pub(crate) struct NewSubmit {
    pub key: OpKey,
    pub identity: OpIdentity,
    pub request_id: String,
    pub rid: Rid,
    pub attempt: u64,
    pub remaining_budget_ms: u64,
    /// The coordinator's body bytes, forwarded verbatim (M2.4 D5: the client's `tools` bytes reach SGLang).
    pub body: Box<RawValue>,
    pub stream: bool,
    /// The body has `tools` (not null).
    pub tools: bool,
}

/// A validated CancelWork.
pub(crate) struct NewCancel {
    pub key: OpKey,
    pub identity: OpIdentity,
    pub request_id: String,
    pub attempt: u64,
    pub ticket: String,
    pub remaining_budget_ms: u64,
}

/// The backend side of one op.
enum Source {
    Opening(Pin<Box<dyn Future<Output = Result<ChatStream, BackendError>> + Send>>),
    Streaming(ChatStream),
    Unary(Pin<Box<dyn Future<Output = Result<Value, BackendError>> + Send>>),
    Done,
}

impl Source {
    /// The next chunk (the whole response for unary), `None` at the end. Cancel-safe: the pending call stays in
    /// `self`.
    async fn next(&mut self) -> Result<Option<Value>, BackendError> {
        loop {
            match self {
                Source::Opening(call) => match call.as_mut().await {
                    Ok(chat) => *self = Source::Streaming(chat),
                    Err(e) => {
                        *self = Source::Done;
                        return Err(e);
                    }
                },
                Source::Streaming(chat) => {
                    let next = chat.next_chunk().await;
                    if !matches!(next, Ok(Some(_))) {
                        *self = Source::Done;
                    }
                    return next;
                }
                Source::Unary(call) => {
                    let response = call.as_mut().await;
                    *self = Source::Done;
                    return response.map(Some);
                }
                Source::Done => return Ok(None),
            }
        }
    }
}

/// The operation runtime: the live-op table and its tasks.
pub(crate) struct Ops {
    shared: Arc<Shared>,
    table: Mutex<HashMap<OpKey, Arc<LiveOp>>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl Ops {
    /// The runtime plus its fence watcher and the H3(b) cleaner of unheld ops.
    pub(crate) fn start(shared: Arc<Shared>) -> Arc<Ops> {
        let ops = Arc::new(Ops {
            shared,
            table: Mutex::new(HashMap::new()),
            tasks: Mutex::new(Vec::new()),
        });
        ops.spawn(Arc::clone(&ops).watch_fences());
        ops.spawn(Arc::clone(&ops).clean_unheld());
        ops
    }

    fn spawn(&self, task: impl Future<Output = ()> + Send + 'static) {
        let mut tasks = lock(&self.tasks);
        tasks.retain(|t| !t.is_finished());
        tasks.push(tokio::spawn(task));
    }

    /// Waits for every task (they end once `Shared::stop` is set).
    pub(crate) async fn join(&self) {
        loop {
            let tasks = std::mem::take(&mut *lock(&self.tasks));
            if tasks.is_empty() {
                return;
            }
            for task in tasks {
                let _ = task.await;
            }
        }
    }

    /// Refuses an operation whose authority no longer matches the node under the journal lock (a fence ran
    /// between `Shared::authorize` and the journal).
    fn recheck(&self, identity: &OpIdentity) -> Result<(), Status> {
        let refuse = |reason| Err(wire::status(Code::FailedPrecondition, reason));
        if identity.worker_epoch != self.shared.epoch() {
            return refuse(wire::STALE_EPOCH);
        }
        if identity.coordinator_term != self.shared.term() {
            return refuse(wire::STALE_TERM);
        }
        if identity.model_generation != self.shared.generation() {
            return refuse(wire::STALE_GENERATION);
        }
        Ok(())
    }

    fn op_error(&self, error: OpError) -> Status {
        self.shared.op_error(error)
    }

    fn emit_accept(&self, key: &OpKey, fresh: bool) {
        self.shared.events.emit(
            "accept",
            format_args!("op={}/{} fresh={fresh}", key.scope_id, key.operation_id),
        );
    }

    fn emit_terminal(&self, key: &OpKey, reason: &str) {
        self.shared.events.emit(
            "terminal",
            format_args!("op={}/{} reason={reason}", key.scope_id, key.operation_id),
        );
    }

    fn emit_cleanup(&self, key: &OpKey) {
        self.shared.events.emit(
            "cleanup",
            format_args!("op={}/{}", key.scope_id, key.operation_id),
        );
    }

    fn abort(&self, rid: &Rid) {
        let backend = self.shared.backend.clone();
        let rid = rid.clone();
        tokio::spawn(async move {
            let _ = backend.abort(&rid).await;
        });
    }

    /// Journals SubmitWork. A journaled key is answered by the journal in every admission state (the service
    /// already refused `poisoned`): the same identity gets the original receipt and starts nothing, another one
    /// is `operation_conflict` (Q11). A new key needs a serving model ([`Shared::work_refusal`]: `model_draining`,
    /// `backend_unconfirmed`; m36 design note Q5) and `open` admission, checked here under the journal lock so a
    /// fence racing the call cannot slip it in (`fenced`: `UNAVAILABLE lease_expired`), and is `queue_full` beyond
    /// the backend's live-op bound. A fresh op starts its pump.
    pub(crate) fn submit(self: &Arc<Self>, s: NewSubmit) -> Result<Accepted, Status> {
        let mut journal = lock(&self.shared.journal);
        self.recheck(&s.identity)?;
        if journal.get(&s.key).is_none() {
            self.shared.work_refusal()?;
            // Team pools (C19, F2): owner-only (switch or window) refuses other scopes' new work.
            if self.shared.owner_only.refuses(&s.key.scope_id, now_ms()) {
                return Err(wire::status(Code::Unavailable, wire::OWNER_ONLY));
            }
            match self.shared.admission_state() {
                AdmissionState::Open => {}
                AdmissionState::Poisoned => {
                    return Err(wire::status(Code::Unavailable, wire::POISONED))
                }
                AdmissionState::Fenced => {
                    return Err(wire::status(Code::Unavailable, wire::LEASE_EXPIRED))
                }
            }
            let cap = self
                .shared
                .identity()
                .map_or(0, |identity| identity.max_running_requests);
            if journal.live_ops() as u64 >= cap {
                return Err(wire::status(Code::ResourceExhausted, wire::QUEUE_FULL));
            }
        }
        let accepted = journal
            .accept(
                s.key.clone(),
                s.identity,
                &s.request_id,
                s.attempt,
                s.remaining_budget_ms,
                now_ms(),
            )
            .map_err(|e| self.op_error(e))?;
        if accepted.fresh {
            let view = journal
                .get(&s.key)
                .ok_or_else(|| self.op_error(OpError::Unknown))?;
            let op = Arc::new(LiveOp::new(Header::of(&self.shared, &view), s.rid));
            {
                let mut table = lock(&self.table);
                table.retain(|key, op| !op.done() && journal.get(key).is_some());
                table.insert(s.key.clone(), Arc::clone(&op));
            }
            let deadline = Instant::now()
                + Duration::from_millis(accepted.deadline_ms.saturating_sub(now_ms()));
            let source = self.source(s.body, s.tools, s.stream, deadline);
            self.spawn(Arc::clone(self).pump(op, source, deadline));
        }
        drop(journal);
        self.emit_accept(&s.key, accepted.fresh);
        Ok(accepted)
    }

    /// The backend call of a fresh op. A tools body on a backend whose confirmed `tool_call_parser` is not
    /// `qwen3_coder` fails `unsupported_parameter` without a backend call (M2.4 D1; M24 Z2 R1-06).
    fn source(&self, body: Box<RawValue>, tools: bool, stream: bool, deadline: Instant) -> Source {
        let parser = self.shared.identity().and_then(|i| i.tool_call_parser);
        if tools && parser.as_deref() != Some(TOOL_CALL_PARSER) {
            let refused = BackendError::ToolsUnsupported;
            return if stream {
                Source::Opening(Box::pin(async { Err(refused) }))
            } else {
                Source::Unary(Box::pin(async { Err(refused) }))
            };
        }
        let backend = self.shared.backend.clone();
        let budget = deadline.saturating_duration_since(Instant::now()) + BACKEND_GRACE;
        if stream {
            Source::Opening(Box::pin(
                async move { backend.chat_stream(&body, budget).await },
            ))
        } else {
            Source::Unary(Box::pin(async move { backend.chat(&body, budget).await }))
        }
    }

    /// Journals CancelWork: the target is the SubmitWork op of this scope with this request id and attempt, whose
    /// ticket and generation must match (else `ticket_mismatch`, nothing journaled). A fresh cancel is terminal and
    /// cleaned up at once and ends a nonterminal target `cancel_requested` (aborted); a repeat returns the
    /// original receipt. A journal write failure poisons the agent; the target still ends locally (its terminal
    /// event line only when journaled).
    pub(crate) fn cancel(&self, c: NewCancel) -> Result<Accepted, Status> {
        let mut journal = lock(&self.shared.journal);
        self.recheck(&c.identity)?;
        let target = if journal.get(&c.key).is_none() {
            let (all, _) = journal.list(None, usize::MAX);
            let target = all.into_iter().find(|op| {
                op.method == wire::METHOD_SUBMIT
                    && op.key.scope_id == c.key.scope_id
                    && op.request_id == c.request_id
                    && op.attempt == c.attempt
            });
            match target {
                Some(op)
                    if op.ticket == c.ticket
                        && op.model_generation == c.identity.model_generation =>
                {
                    Some(op)
                }
                _ => {
                    return Err(wire::status(
                        Code::FailedPrecondition,
                        wire::TICKET_MISMATCH,
                    ))
                }
            }
        } else {
            None
        };
        let accepted = journal
            .accept(
                c.key.clone(),
                c.identity,
                &c.request_id,
                c.attempt,
                c.remaining_budget_ms,
                now_ms(),
            )
            .map_err(|e| self.op_error(e))?;
        let mut writes_ok = true;
        // The nonterminal target this cancel ends, and whether its terminal was journaled.
        let mut ended = None;
        if accepted.fresh {
            let now = now_ms();
            writes_ok &= journal.mark_terminal(&c.key, "", now).is_ok();
            writes_ok &= journal.mark_cleanup(&c.key, now).is_ok();
            if let Some(target) = target.filter(|op| !op.terminal) {
                let journaled = journal
                    .mark_terminal(&target.key, wire::CANCEL_REQUESTED, now)
                    .is_ok();
                writes_ok &= journaled;
                ended = Some((target, journaled));
            }
        }
        let live = ended
            .as_ref()
            .and_then(|(target, _)| lock(&self.table).get(&target.key).cloned());
        drop(journal);
        self.emit_accept(&c.key, accepted.fresh);
        if accepted.fresh && writes_ok {
            self.emit_terminal(&c.key, "");
            self.emit_cleanup(&c.key);
        }
        if !writes_ok {
            self.shared.poison("journal_write");
        }
        if let Some((target, journaled)) = ended {
            if journaled {
                self.emit_terminal(&target.key, wire::CANCEL_REQUESTED);
            }
            match live {
                Some(op) => {
                    op.finish(wire::CANCEL_REQUESTED);
                    self.abort(&op.rid);
                }
                None => {
                    if let Ok(rid) = Rid::parse(&target.request_id) {
                        self.abort(&rid);
                    }
                }
            }
        }
        Ok(accepted)
    }

    /// Makes `op` terminal `reason` unless the journal already holds its terminal (that reason wins: no second
    /// record, event or abort). A journal write failure poisons the agent; the op still ends locally.
    fn end(&self, op: &LiveOp, reason: &str, abort: bool) {
        let earlier = {
            let mut journal = lock(&self.shared.journal);
            match journal.get(&op.header.key) {
                Some(view) if view.terminal => Ok(Some(view.reason)),
                _ => journal
                    .mark_terminal(&op.header.key, reason, now_ms())
                    .map(|()| None),
            }
        };
        match earlier {
            Ok(Some(reason)) => {
                op.finish(&reason);
                return;
            }
            Ok(None) => self.emit_terminal(&op.header.key, reason),
            Err(_) => self.shared.poison("journal_write"),
        }
        op.finish(reason);
        if abort {
            self.abort(&op.rid);
        }
    }

    /// Journals `key`'s cleanup; an op this process holds is then released ([`Ops::release`]) and a draining model
    /// rechecks quiescence (design-m36-agent.md §3.7).
    fn cleanup(&self, key: &OpKey) {
        let result = lock(&self.shared.journal).mark_cleanup(key, now_ms());
        match result {
            Ok(()) => {
                self.emit_cleanup(key);
                let held = lock(&self.table).get(key).cloned();
                if let Some(op) = held {
                    lock(&op.buffer).cleaned = true;
                    self.release(&op);
                }
                self.shared.quiesce();
            }
            Err(_) => self.shared.poison("journal_write"),
        }
    }

    /// Drops `op` from the table, and its buffer with it, once it is terminal, cleaned up and a stream has attached
    /// to it but none is attached now (an attached stream keeps it until it detaches; [`Attachment`] offers it
    /// again). An op no stream has attached to yet stays for the first attach.
    fn release(&self, op: &Arc<LiveOp>) {
        let mut table = lock(&self.table);
        let releasable = lock(&op.buffer).releasable();
        if releasable
            && table
                .get(&op.header.key)
                .is_some_and(|held| Arc::ptr_eq(held, op))
        {
            table.remove(&op.header.key);
        }
    }

    /// One op's backend call until the backend stream ends, or until shutdown (journal untouched).
    async fn pump(self: Arc<Self>, op: Arc<LiveOp>, source: Source, deadline: Instant) {
        let stop = self.shared.stop.subscribe();
        tokio::select! {
            biased;
            () = stopped(stop) => {}
            () = self.drive(&op, source, deadline) => {}
        }
        lock(&op.buffer).pumped = true;
        op.changed();
    }

    async fn drive(&self, op: &LiveOp, mut source: Source, deadline: Instant) {
        let cap = self.shared.config.limits.output_buffer_bytes;
        let stall = Duration::from_millis(self.shared.config.limits.output_stall_ms);
        loop {
            let live = !op.is_terminal();
            let item = tokio::select! {
                biased;
                () = tokio::time::sleep_until(deadline), if live => {
                    self.end(op, wire::DEADLINE_EXCEEDED, true);
                    continue;
                }
                item = source.next() => item,
            };
            match item {
                Ok(Some(value)) => {
                    if op.is_terminal() {
                        continue;
                    }
                    let body = match serde_json::to_vec(&value) {
                        Ok(body) if body.len() <= wire::WORK_EVENT_MAX => body,
                        _ => {
                            self.end(op, protocol_error(), true);
                            continue;
                        }
                    };
                    let pushed = tokio::select! {
                        biased;
                        () = tokio::time::sleep_until(deadline) => None,
                        pushed = op.push(body, cap, stall) => Some(pushed),
                    };
                    match pushed {
                        None => self.end(op, wire::DEADLINE_EXCEEDED, true),
                        Some(Push::Stalled) => self.end(op, wire::CLIENT_BACKPRESSURE, true),
                        Some(Push::Stored | Push::Terminal) => {}
                    }
                }
                Ok(None) => {
                    self.end(op, "", false);
                    break;
                }
                Err(e) => {
                    self.end(op, backend_reason(&e), true);
                    break;
                }
            }
        }
        self.cleanup(&op.header.key);
    }

    /// Delivers every fence LINK journaled to the live ops it ended.
    async fn watch_fences(self: Arc<Self>) {
        let mut fence = self.shared.fence.subscribe();
        let mut stop = self.shared.stop.subscribe();
        loop {
            tokio::select! {
                _ = stop.wait_for(|stop| *stop) => return,
                changed = fence.changed() => if changed.is_err() {
                    return;
                },
            }
            let live: Vec<Arc<LiveOp>> = lock(&self.table).values().cloned().collect();
            let ended: Vec<(Arc<LiveOp>, String)> = {
                let journal = lock(&self.shared.journal);
                live.into_iter()
                    .filter_map(|op| {
                        let view = journal.get(&op.header.key)?;
                        view.terminal.then_some((op, view.reason))
                    })
                    .collect()
            };
            for (op, reason) in ended {
                op.finish(&reason);
            }
        }
    }

    /// H3(b): ops of an earlier process (terminal, not cleanup-acknowledged, not in the table) are released by the
    /// `/get_load` rule when `load_endpoint` is set (total <= this process's held streams on two polls >= 1 s
    /// apart), else at their worker-local deadline.
    async fn clean_unheld(self: Arc<Self>) {
        let mut stop = self.shared.stop.subscribe();
        let mut below_since: Option<Instant> = None;
        loop {
            let (unheld, held) = {
                let journal = lock(&self.shared.journal);
                let table = lock(&self.table);
                let (all, _) = journal.list(None, usize::MAX);
                let unheld: Vec<OpView> = all
                    .into_iter()
                    .filter(|op| {
                        work_op(op.method)
                            && op.terminal
                            && !op.cleanup_acknowledged
                            && !table.contains_key(&op.key)
                    })
                    .collect();
                let held = table.values().filter(|op| !lock(&op.buffer).pumped).count() as u64;
                (unheld, held)
            };
            if unheld.is_empty() {
                return;
            }
            let mut wait = LOAD_POLL;
            match &self.shared.config.backend.load_endpoint {
                Some(endpoint) => match self.shared.backend.load(endpoint).await {
                    Ok(total) if total <= held => match below_since {
                        Some(at) if at.elapsed() >= LOAD_POLL => {
                            for op in &unheld {
                                self.cleanup(&op.key);
                            }
                            return;
                        }
                        Some(_) => {}
                        None => below_since = Some(Instant::now()),
                    },
                    _ => below_since = None,
                },
                None => {
                    let now = now_ms();
                    for op in &unheld {
                        if now >= op.deadline_ms {
                            self.cleanup(&op.key);
                        } else {
                            wait = wait.min(Duration::from_millis(op.deadline_ms - now));
                        }
                    }
                }
            }
            tokio::select! {
                _ = stop.wait_for(|stop| *stop) => return,
                () = tokio::time::sleep(wait) => {}
            }
        }
    }

    /// A `StreamWork` stream for `view` (ticket already checked): one attached stream per op (a second is
    /// `ALREADY_EXISTS operation_conflict`), replay from `after`; an op no longer held that is terminal, or a held
    /// one the release rule frees ([`Buffer::attach`]), yields just its terminal event. At shutdown the stream ends
    /// `UNAVAILABLE worker_restart` (H2).
    pub(crate) fn attach(
        self: &Arc<Self>,
        view: &OpView,
        after: u64,
        slot: StreamSlot,
    ) -> Result<RpcStream<pb::WorkEvent>, Status> {
        let terminal_only = |header: &Header, reason: &str| -> RpcStream<pb::WorkEvent> {
            let event = header.event(after + 1, Vec::new(), Some(reason));
            Box::pin(stream::iter([Ok(event)]))
        };
        let live = lock(&self.table).get(&view.key).cloned();
        let Some(op) = live else {
            if !view.terminal {
                return Err(wire::status(Code::NotFound, wire::UNKNOWN_OPERATION));
            }
            drop(slot);
            return Ok(terminal_only(&Header::of(&self.shared, view), &view.reason));
        };
        let attached = lock(&op.buffer).attach(view.cleanup_acknowledged);
        match attached {
            Attach::Conflict => {
                return Err(wire::status(Code::AlreadyExists, wire::OPERATION_CONFLICT));
            }
            Attach::Released(reason) => {
                self.release(&op);
                drop(slot);
                return Ok(terminal_only(&op.header, &reason));
            }
            Attach::Attached => {}
        }
        let state = Attached {
            rx: op.version.subscribe(),
            op: Attachment {
                op,
                ops: Arc::downgrade(self),
            },
            cursor: after,
            stop: self.shared.stop.subscribe(),
            done: false,
            _slot: slot,
        };
        Ok(Box::pin(stream::unfold(state, |mut st| async move {
            let item = st.next().await?;
            Some((item, st))
        })))
    }
}

/// The state of one attached stream; dropping it detaches the op (offering it for release) and frees the stream
/// slot.
struct Attached {
    op: Attachment,
    rx: watch::Receiver<u64>,
    cursor: u64,
    stop: watch::Receiver<bool>,
    done: bool,
    _slot: StreamSlot,
}

impl Attached {
    async fn next(&mut self) -> Option<Result<pb::WorkEvent, Status>> {
        if self.done {
            return None;
        }
        let restart = || Some(Err(wire::status(Code::Unavailable, wire::WORKER_RESTART)));
        loop {
            if *self.stop.borrow() {
                self.done = true;
                return restart();
            }
            self.rx.borrow_and_update();
            if let Some(event) = self.op.op.next_for(self.cursor) {
                self.cursor = event.sequence;
                self.done = event.terminal;
                return Some(Ok(event));
            }
            tokio::select! {
                _ = self.rx.changed() => {}
                changed = self.stop.changed() => if changed.is_err() {
                    self.done = true;
                    return restart();
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A terminal op whose broken stream has detached; `cleaned`: the buffer has seen the journal's cleanup.
    fn detached_terminal(cleaned: bool) -> Buffer {
        Buffer {
            next: 41,
            terminal: Some("stop".to_owned()),
            streamed: true,
            cleaned,
            ..Buffer::default()
        }
    }

    /// O-WORK (ci-docker `3f53be2`, `agent_work::a_cleaned_up_op_frees_its_buffer`): `Ops::cleanup` journals the
    /// cleanup before it marks the buffer, so a stream that attaches in between, with a journal view that holds the
    /// cleanup, must already find the op released, not replay its buffer.
    #[test]
    fn an_attach_between_the_journaled_cleanup_and_the_buffer_mark_is_released() {
        let mut b = detached_terminal(false);
        assert_eq!(b.attach(true), Attach::Released("stop".to_owned()));
        assert!(!b.attached);
        assert!(b.releasable());
    }

    #[test]
    fn the_release_rule_holds_for_every_other_attach() {
        // Cleaned on both sides: released (normally gone from the table already).
        assert_eq!(
            detached_terminal(true).attach(true),
            Attach::Released("stop".to_owned())
        );
        // The journal view predates the cleanup and the buffer has not seen it: attach and replay.
        let mut b = detached_terminal(false);
        assert_eq!(b.attach(false), Attach::Attached);
        assert!(b.attached && b.streamed);
        // A stream is attached: conflict, the buffer unchanged.
        assert_eq!(b.attach(true), Attach::Conflict);
        // Never streamed: the first attach replays even after cleanup.
        let mut first = Buffer {
            streamed: false,
            ..detached_terminal(true)
        };
        assert_eq!(first.attach(true), Attach::Attached);
        // Not terminal: attach.
        let mut live = Buffer {
            terminal: None,
            ..detached_terminal(false)
        };
        assert_eq!(live.attach(true), Attach::Attached);
    }
}
