//! `Backend::Remote`: the coordinator's dispatch of chat requests to a remote worker over the worker RPC
//! (m35 design note Q11, Q15, H2). Tracks DISPATCH and ORPHAN (build-loop-m35 design note "CREM split").
//!
//! A [`RemoteWorker`] is one `[[federation.workers]]` entry with its agent client; it takes requests through
//! admission like a local replica. Each granted request builds a [`RemoteCall`] with the authority of the
//! worker's registry state at the grant: `SubmitWork` (operation id = request id = the rid, attempt 1,
//! `work_unit_json` `{"kind":"chat","stream":<bool>}`, the exact backend body as the inference input), then
//! `StreamWork` from sequence 0 with the receipt's ticket. Each work event before the terminal carries one backend
//! chunk (a streamed chat) or the whole response (a unary chat) as JSON; the natural terminal (empty reason) ends
//! the call.
//!
//! Submit faults (H2, track DB1): [`RemoteCall::submit`] retries a transport failure with the same operation
//! inside the request budget and classes a refusal: the suspect class re-queues the request in admission (which
//! marks the worker suspect), the joining class takes the worker back to Joining first ([`RemoteWorker::rejoin`]),
//! anything else is `backend_protocol_error`.
//!
//! Cancel and the work's end (H2, track DB2): [`RemoteCall::cancel`] sends one `CancelWork` (`<rid>/cancel/1`,
//! naming the Submit by its receipt's ticket; a cancel before the receipt goes out once the receipt arrived),
//! never after the natural terminal. A terminal with a reason ends the call with that reason as the request's
//! terminal reason and error code (`cancel_requested`, the agent's abort, is `backend_abort`), or with the agent's
//! backend failure (`backend_unavailable`, `backend_timeout`, `context_overflow`, `invalid_value` = a tool
//! schema the backend refused, M2.4 D5; any other reason is `backend_protocol_error`). A protocol violation (an
//! event above [`wire::WORK_EVENT_MAX`], an event out of offset order, an unparsable event, the stream ending
//! without a terminal, a unary chat without exactly one response event, the stream failing with a status
//! outside the transport class) cancels the work and fails `backend_protocol_error` at once.
//!
//! The orphan hold (H2, track DB3): a call that may still hold the agent's work when its slot is released (its
//! Submit went out, even if the receipt came too late, and it did not reach its natural terminal,
//! [`RemoteCall::held`]) keeps the slot charged to the worker and counted retained there ([`hold_orphan`]) until
//! the agent reports the operation cleaned up (asked under the worker's current registry authority, not the
//! call's) or the request's retention bound passed; either release is journaled.
//!
//! Lost and D5 (H2, track ORPHAN-A): every RPC of a call (the Submit, the stream's open, each event) races
//! [`RemoteCall::gone`], and the first to finish wins: once the worker reads Lost the call ends `worker_lost`,
//! once it registered a higher epoch `worker_epoch_superseded`, that reason recorded as the request's like a
//! terminal's. An event already received goes first; after the natural terminal nothing races.
//!
//! Parking (H2): a work stream that breaks on the transport (an RPC failure without a `wire` reason, or the
//! agent's `UNAVAILABLE worker_restart` at its shutdown) parks the request: no output, its slot still charged,
//! the client stream open, until the worker is gone (as above), reconnects at the call's epoch (a new registry
//! round, not Lost: `backend_protocol_error` with the work cancelled, as no stream resumes in M3.5), or the
//! request's cancel ([`RemoteCall::cancel`]: its deadline or its client gone), whose reason then ends it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::Duration;

use cere_proto::worker::v2 as pb;
use serde_json::Value;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tonic::{Code, Status, Streaming};

use crate::admission::{Admission, Permit};
use crate::app::{
    unix_ms, REASON_BACKEND_ABORT, REASON_CLIENT_BACKPRESSURE, REASON_DEADLINE_EXCEEDED,
    REASON_WORKER_LOST,
};
use crate::config::RemoteWorkerConfig;
use crate::fed::client::{AuthorityBase, OpRef, Work, WorkerClient};
use crate::fed::orphan::registered_authority;
use crate::fed::registry::{RemoteRegistry, RemoteState};
use crate::fed::wire;
use crate::journal::OrphanReleaseCause;
use crate::ledger::{Ledger, REASON_WORKER_EPOCH_SUPERSEDED};
use crate::sglang::{BackendError, GRAMMAR_COMPILE_REASON};
use crate::validate::BackendBody;
use crate::worker::WorkerState;

/// The first cleanup poll of an orphan hold ([`hold_orphan`]); later gaps double up to the hold's `period`. A
/// cancelled work's cleanup is usually acknowledged within this, so its slot frees then rather than a whole
/// `period` later (an owner waiting on its owned worker, net-teleport design note 4.4 Q-T6).
const ORPHAN_FIRST_POLL: Duration = Duration::from_millis(100);

/// One configured remote worker: its `[[federation.workers]]` entry, its agent client and the registry that
/// tracks it (clones share the channel and the retained count).
#[derive(Debug, Clone)]
pub struct RemoteWorker {
    config: RemoteWorkerConfig,
    client: WorkerClient,
    cluster_id: String,
    coordinator_id: String,
    registry: Arc<RemoteRegistry>,
    /// Held slots of calls whose work was cancelled or ended and that only wait for their release
    /// ([`hold_orphan`]); excluded from the worker's open backend streams at shutdown.
    retained: Arc<AtomicU64>,
}

impl RemoteWorker {
    pub(crate) fn new(
        config: RemoteWorkerConfig,
        client: WorkerClient,
        cluster_id: &str,
        coordinator_id: &str,
        registry: Arc<RemoteRegistry>,
    ) -> RemoteWorker {
        RemoteWorker {
            config,
            client,
            cluster_id: cluster_id.to_owned(),
            coordinator_id: coordinator_id.to_owned(),
            registry,
            retained: Arc::default(),
        }
    }

    pub fn worker_id(&self) -> &str {
        &self.config.worker_id
    }

    /// The configured agent endpoint, journaled with every `dispatched` record of this worker (Q15).
    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    /// The worker's count of held slots that only wait for their release.
    pub(crate) fn retained(&self) -> &Arc<AtomicU64> {
        &self.retained
    }

    /// The authority of work granted on this worker at `epoch`, from its registry `state` at the grant: `None`
    /// unless the worker is Ready at that epoch in this term.
    pub(crate) fn authority(&self, state: &RemoteState, epoch: u64) -> Option<AuthorityBase> {
        if state.state != WorkerState::Ready || state.epoch != Some(epoch) {
            return None;
        }
        Some(AuthorityBase {
            cluster_id: self.cluster_id.clone(),
            worker_id: self.config.worker_id.clone(),
            worker_epoch: epoch,
            coordinator_id: self.coordinator_id.clone(),
            coordinator_term: state.term?,
            model_generation: state.generation,
        })
    }

    /// The call of request `rid` in `scope` with the backend body `body` (which carries `rid`), under `at`.
    pub(crate) fn call(
        &self,
        at: AuthorityBase,
        scope: &str,
        rid: &str,
        stream: bool,
        body: &BackendBody,
    ) -> RemoteCall {
        let unit = format!(r#"{{"kind":"chat","stream":{stream}}}"#);
        RemoteCall {
            client: self.client.clone(),
            worker_id: self.config.worker_id.clone(),
            registry: self.registry.clone(),
            at,
            work: Work {
                scope_id: scope.to_owned(),
                operation_id: rid.to_owned(),
                request_id: rid.to_owned(),
                attempt: 1,
                work_unit_json: unit.into_bytes(),
                input_json: serde_json::to_vec(body).expect("backend bodies serialize"),
            },
            state: Mutex::new(CallState::default()),
            cancelled: watch::Sender::new(false),
        }
    }

    /// H2, joining class: the worker refused work under an out-of-date authority. It goes back to Joining in its
    /// current round (a Lost or re-registered worker stays as it is), and admission sees the new view at once.
    pub(crate) fn rejoin(&self, registry: &RemoteRegistry, admission: &Admission) {
        let id = self.worker_id();
        if let Some(state) = registry.get(id, unix_ms()) {
            if let Err(e) = registry.enter_joining(id, state.round) {
                eprintln!("cered remote worker={id} not rejoined after a refusal: {e:?}");
            }
        }
        push_view(registry, admission, &self.config);
    }
}

/// Milliseconds from now until `at` (0 once it passed).
fn ms_until(at: Instant) -> u64 {
    at.saturating_duration_since(Instant::now()).as_millis() as u64
}

/// SubmitWork tries of one grant: the first and two retries of a transport failure (H2).
const SUBMIT_TRIES: u32 = 3;

/// How the coordinator re-queues a refused Submit (H2's refusal table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The worker cannot take the work now, or did not answer: admission marks it suspect and re-queues.
    Suspect,
    /// The work's authority is out of date: the worker re-enters Joining, then the request re-queues.
    Joining,
    /// Team pools (net-teleport design note 4.4, C19): the worker is owner-only and the request's scope is not
    /// its owner. Admission takes the worker out of the scope's grants (its switch reads on until the next
    /// heartbeat says otherwise) and the request re-queues.
    OwnerOnly,
}

/// A failed SubmitWork, classed by the `wire` reason of its status.
enum Failure {
    Refused(Refusal),
    /// The RPC itself failed (no `wire` reason): retried with the same operation.
    Transport,
    /// Any other refusal: the coordinator sent work the agent cannot take.
    Protocol,
}

/// m36 design note Q10 (review M36-05): a managed agent's model refusals (`model_draining`,
/// `backend_unconfirmed`, `backend_exited`) are suspect-class, so the request re-queues to another eligible worker.
fn classify(status: &Status) -> Failure {
    match wire::reason(status) {
        wire::LEASE_EXPIRED
        | wire::QUEUE_FULL
        | wire::JOURNAL_FULL
        | wire::POISONED
        | wire::QUOTA_EXHAUSTED
        | wire::MODEL_DRAINING
        | wire::BACKEND_UNCONFIRMED
        | wire::BACKEND_EXITED => Failure::Refused(Refusal::Suspect),
        wire::STALE_TERM
        | wire::STALE_EPOCH
        | wire::STALE_GENERATION
        | wire::WRONG_CLUSTER
        | wire::WRONG_WORKER
        | wire::WRONG_COORDINATOR => Failure::Refused(Refusal::Joining),
        wire::OWNER_ONLY => Failure::Refused(Refusal::OwnerOnly),
        reason if is_wire_reason(reason) => Failure::Protocol,
        _ if matches!(
            status.code(),
            Code::Unavailable
                | Code::Unknown
                | Code::Cancelled
                | Code::DeadlineExceeded
                | Code::Internal
        ) =>
        {
            Failure::Transport
        }
        _ => Failure::Protocol,
    }
}

/// A `wire` reason code: the agent's refusals carry the bare snake_case code as the status message
/// ([`wire::status`]); a transport failure's message is prose.
fn is_wire_reason(message: &str) -> bool {
    !message.is_empty() && message.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
}

/// An RPC failure without a `wire` reason (H2): the agent may not have seen the call, which is retried unchanged
/// (the residency driver's lifecycle calls, m36 design note Q8).
pub(crate) fn is_transport(status: &Status) -> bool {
    matches!(classify(status), Failure::Transport)
}

/// H2: a work stream that ended with `status` broke on the transport (an RPC failure without a `wire` reason, or
/// the agent's `UNAVAILABLE worker_restart` at its shutdown) and parks; any other status is a protocol violation.
fn broken(status: &Status) -> bool {
    matches!(classify(status), Failure::Transport)
        || (status.code() == Code::Unavailable && wire::reason(status) == wire::WORKER_RESTART)
}

/// One request's work on a remote worker (Q11), built at the grant: the authority from the worker's registry
/// state then, the Submit operation, and the call's progress.
#[derive(Debug)]
pub(crate) struct RemoteCall {
    client: WorkerClient,
    worker_id: String,
    /// The registry that tracks the worker: [`RemoteCall::gone`] reads it.
    registry: Arc<RemoteRegistry>,
    at: AuthorityBase,
    work: Work,
    state: Mutex<CallState>,
    /// Set once the work's cancel was requested ([`RemoteCall::cancel`]): it ends a parked stream.
    cancelled: watch::Sender<bool>,
}

#[derive(Debug, Default)]
struct CallState {
    /// The receipt's completion ticket, once SubmitWork was accepted.
    ticket: Option<String>,
    /// A SubmitWork try went out: the agent may have journaled the work even without a receipt (a late receipt).
    sent: bool,
    /// The work's cancel was requested; its `CancelWork` goes out once the ticket is known.
    cancel: bool,
    /// The natural terminal (empty reason) arrived: the work ended by itself and is never cancelled.
    natural: bool,
    /// The coordinator reason of a terminal that carried one (H2): the request's terminal reason.
    reason: Option<&'static str>,
}

impl RemoteCall {
    fn state(&self) -> MutexGuard<'_, CallState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// SubmitWork with the remaining budget until `budget_until` (the request deadline; each try's RPC deadline
    /// is min(rpc timeout, that budget), recomputed per try and never renewed), until the worker is
    /// [`gone`](Self::gone). `Ok(None)` once accepted (the ticket is stored), `Ok(Some(_))` for a refusal to
    /// re-queue (a transport failure counts as suspect once the tries ran out), `Err(Timeout)` once the budget
    /// ran out before a receipt, `Err(Protocol)` for any other refusal, a receipt not accepted or the worker gone
    /// (its reason recorded).
    pub(crate) async fn submit(
        self: &Arc<Self>,
        budget_until: Instant,
    ) -> Result<Option<Refusal>, BackendError> {
        tokio::select! {
            biased;
            submitted = self.submit_tries(budget_until) => submitted,
            reason = self.gone() => Err(self.abandon(reason)),
        }
    }

    async fn submit_tries(
        self: &Arc<Self>,
        budget_until: Instant,
    ) -> Result<Option<Refusal>, BackendError> {
        let mut tries = 0;
        loop {
            let budget_ms = ms_until(budget_until);
            if budget_ms == 0 {
                return Err(BackendError::Timeout(
                    "SubmitWork: the request budget ran out before a receipt".to_owned(),
                ));
            }
            tries += 1;
            self.state().sent = true;
            let status = match self.client.submit(&self.at, &self.work, budget_ms).await {
                Ok(receipt) if receipt.accepted => {
                    self.accept(receipt.completion_ticket);
                    return Ok(None);
                }
                Ok(receipt) => {
                    return Err(protocol(&format!(
                        "SubmitWork not accepted: {}",
                        receipt.reason_code
                    )))
                }
                Err(status) => status,
            };
            match classify(&status) {
                Failure::Refused(refusal) => return Ok(Some(refusal)),
                Failure::Transport if tries < SUBMIT_TRIES => eprintln!(
                    "cered remote submit rid={} worker={} try {tries} failed: {}",
                    self.work.request_id,
                    self.worker_id,
                    status_text(&status)
                ),
                Failure::Transport => return Ok(Some(Refusal::Suspect)),
                Failure::Protocol => {
                    return Err(protocol(&format!("SubmitWork: {}", status_text(&status))))
                }
            }
        }
    }

    /// Stores the receipt's `ticket`; a cancel requested before the receipt goes out now.
    fn accept(self: &Arc<Self>, ticket: String) {
        let cancel = {
            let mut state = self.state();
            state.ticket = Some(ticket.clone());
            state.cancel
        };
        if cancel {
            self.send_cancel(ticket);
        }
    }

    /// H2: cancels the work, once and never after its natural terminal: `CancelWork` with the receipt's ticket, or
    /// before the receipt only the request (the receipt sends it). A parked stream ends at once.
    pub(crate) fn cancel(self: &Arc<Self>) {
        let ticket = {
            let mut state = self.state();
            if state.cancel || state.natural {
                return;
            }
            state.cancel = true;
            state.ticket.clone()
        };
        self.cancelled.send_replace(true);
        if let Some(ticket) = ticket {
            self.send_cancel(ticket);
        }
    }

    /// `CancelWork` `<rid>/cancel/1` of the Submit with `ticket`, on its own task; a failure is logged (the
    /// work's own deadline still ends it).
    fn send_cancel(self: &Arc<Self>, ticket: String) {
        let call = self.clone();
        tokio::spawn(async move {
            let operation_id = format!("{}/cancel/1", call.work.request_id);
            let target = call.target(ticket);
            if let Err(e) = call.client.cancel(&call.at, &operation_id, &target).await {
                eprintln!(
                    "cered remote cancel rid={} worker={} failed: {}",
                    call.work.request_id,
                    call.worker_id,
                    status_text(&e)
                );
            }
        });
    }

    /// The Submit operation with the receipt's `ticket`, as a cancel or stream names it.
    fn target(&self, ticket: String) -> OpRef {
        OpRef {
            scope_id: self.work.scope_id.clone(),
            request_id: self.work.request_id.clone(),
            attempt: self.work.attempt,
            completion_ticket: ticket,
        }
    }

    /// The coordinator reason of the work's terminal, when it carried one (H2).
    pub(crate) fn reason(&self) -> Option<&'static str> {
        self.state().reason
    }

    /// The work may still hold the agent's resources: a Submit went out (receipt or not) and the work did not
    /// reach its natural terminal. Without a ticket no `CancelWork` can target it: the hold ends at the agent's
    /// cleanup report (or `unknown_operation`) or the retention bound.
    pub(crate) fn held(&self) -> bool {
        let state = self.state();
        state.sent && !state.natural
    }

    /// `GetOperation` of the Submit under the worker's current registry authority ([`registered_authority`]; a
    /// restart at a higher epoch refuses the call's own): `Ok(true)` once the agent reports it
    /// `cleanup_acknowledged` or no longer knows it (`NOT_FOUND unknown_operation`); `Ok(false)` without asking
    /// while the worker has no accepted Register in this term.
    pub(crate) async fn cleaned_up(&self) -> Result<bool, Status> {
        let Some(at) = registered_authority(
            &self.registry,
            &self.worker_id,
            &self.at.cluster_id,
            &self.at.coordinator_id,
        ) else {
            return Ok(false);
        };
        let queried = self
            .client
            .get_operation(&at, &self.work.scope_id, &self.work.operation_id)
            .await;
        match queried {
            Ok(status) => Ok(status.cleanup_acknowledged),
            Err(e) if e.code() == Code::NotFound && e.message() == wire::UNKNOWN_OPERATION => {
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }

    /// H2/D5: resolves once the call's worker is gone: `worker_epoch_superseded` once the registry holds a higher
    /// epoch of it, else `worker_lost` once it reads Lost. Reads the registry again on every round bump (a new
    /// incarnation registers in a new round) and at the worker's Lost bound, `last_seen_ms + lost_after_ms` as
    /// read (Lost by aging bumps no round; a later sighting moves the bound).
    async fn gone(&self) -> &'static str {
        let mut rounds = self.registry.subscribe();
        let lost_after_ms = self.registry.timing().lost_after_ms;
        loop {
            let now = unix_ms();
            let lost = self.registry.state_at(&self.worker_id, now) == Some(WorkerState::Lost);
            let Some(view) = self.registry.worker_view(&self.worker_id) else {
                // Not registered in this term: no work is ever granted on it.
                return std::future::pending().await;
            };
            if view.epoch > self.at.worker_epoch {
                return REASON_WORKER_EPOCH_SUPERSEDED;
            }
            if lost {
                return REASON_WORKER_LOST;
            }
            let lost_at = view.last_seen_ms.saturating_add(lost_after_ms);
            let bound = tokio::time::sleep(Duration::from_millis(lost_at.saturating_sub(now)));
            // The registry owns the round sender as long as this call holds it: `changed` never fails.
            tokio::select! {
                _ = rounds.changed() => {}
                () = bound => {}
            }
        }
    }

    /// H2: resolves once the call's worker reconnected at the call's epoch after the registry round read first
    /// (when a broken stream parks): its round went above that one while it is not Lost. (A same-epoch Register
    /// bumps no round; its first heartbeat after the lease view expired starts a new one, Q7a.)
    async fn reconnected(&self) {
        // Subscribed before the read: a bump after it is never missed.
        let mut rounds = self.registry.subscribe();
        let Some(parked) = self.registry.get(&self.worker_id, unix_ms()) else {
            // Not a configured remote worker: no work is ever granted on it.
            return std::future::pending().await;
        };
        loop {
            // The registry owns the round sender as long as this call holds it: `changed` never fails.
            let _ = rounds.changed().await;
            let Some(state) = self.registry.get(&self.worker_id, unix_ms()) else {
                continue;
            };
            if state.epoch == Some(self.at.worker_epoch)
                && state.round > parked.round
                && state.state != WorkerState::Lost
            {
                return;
            }
        }
    }

    /// Resolves once the work's cancel was requested ([`Self::cancel`]).
    async fn cancel_requested(&self) {
        let mut cancelled = self.cancelled.subscribe();
        // The call owns the sender: `wait_for` never fails.
        let _ = cancelled.wait_for(|cancelled| *cancelled).await;
    }

    /// D5: the call's worker is gone with `reason` ([`Self::gone`]): like a terminal's reason, it is the request's
    /// terminal reason and error code.
    fn abandon(&self, reason: &'static str) -> BackendError {
        self.state().reason = Some(reason);
        protocol(&format!("the remote worker is gone: {reason}"))
    }

    /// The terminal event with `reason`: `Ok(None)` for the natural terminal (empty), else the failure that ended
    /// the work, with its coordinator reason recorded.
    fn ended(&self, reason: &str) -> Result<Option<Value>, BackendError> {
        let coordinator = match reason {
            "" => {
                self.state().natural = true;
                return Ok(None);
            }
            wire::DEADLINE_EXCEEDED => Some(REASON_DEADLINE_EXCEEDED),
            wire::CLIENT_BACKPRESSURE => Some(REASON_CLIENT_BACKPRESSURE),
            wire::WORKER_RESTART => Some(wire::WORKER_RESTART),
            wire::MODEL_GENERATION_SUPERSEDED => Some(wire::MODEL_GENERATION_SUPERSEDED),
            wire::COORDINATOR_RESTART => Some(wire::COORDINATOR_RESTART),
            wire::CANCEL_REQUESTED => Some(REASON_BACKEND_ABORT),
            _ => None,
        };
        self.state().reason = coordinator;
        let detail = format!("the remote work ended {reason}");
        // The agent ends a failed backend call with the backend error's API code.
        Err(match reason {
            "backend_unavailable" => BackendError::Connect(detail),
            "backend_timeout" => BackendError::Timeout(detail),
            "context_overflow" => BackendError::ContextOverflow { message: detail },
            "invalid_value" => BackendError::ToolSchema { index: None },
            "unsupported_parameter" => BackendError::ToolsUnsupported,
            GRAMMAR_COMPILE_REASON => BackendError::GrammarCompile,
            _ => BackendError::Protocol(detail),
        })
    }

    /// StreamWork of the accepted Submit from sequence 0 with the receipt's ticket, open until `until`; fails
    /// once the worker is [`gone`](Self::gone) first.
    pub(crate) async fn open(
        self: &Arc<Self>,
        until: Instant,
    ) -> Result<RemoteStream, BackendError> {
        let ticket = self
            .state()
            .ticket
            .clone()
            .ok_or_else(|| protocol("StreamWork before the Submit receipt"))?;
        let target = self.target(ticket);
        let opened = tokio::select! {
            biased;
            opened = self.client.stream(
                &self.at,
                &self.work.operation_id,
                &target,
                0,
                ms_until(until),
            ) => opened,
            reason = self.gone() => return Err(self.abandon(reason)),
        };
        let events = opened
            .map_err(|e| BackendError::Protocol(format!("StreamWork: {}", status_text(&e))))?;
        Ok(RemoteStream {
            call: self.clone(),
            events,
            offset: 0,
            done: false,
        })
    }

    /// A unary chat: the single event before the natural terminal is the whole response; none or more than one
    /// is a violation.
    pub(crate) async fn chat(self: &Arc<Self>, until: Instant) -> Result<Value, BackendError> {
        let mut stream = self.open(until).await?;
        let Some(response) = stream.next_chunk().await? else {
            return Err(stream.violation("the work ended without a response event"));
        };
        match stream.next_chunk().await? {
            None => Ok(response),
            Some(_) => Err(stream.violation("more than one response event for a unary chat")),
        }
    }
}

/// The work events of one remote call, as backend chunks.
pub(crate) struct RemoteStream {
    call: Arc<RemoteCall>,
    events: Streaming<pb::WorkEvent>,
    /// The `committed_token_offset` of the next event before the terminal: its index from 0.
    offset: u64,
    done: bool,
}

impl RemoteStream {
    /// The next chunk, `Ok(None)` at the natural terminal (and after the end), or the failure that ended the call.
    /// The next event races [`RemoteCall::gone`]; one already received goes first. A stream broken on the
    /// transport parks first ([`Self::park`]).
    pub(crate) async fn next_chunk(&mut self) -> Result<Option<Value>, BackendError> {
        if self.done {
            return Ok(None);
        }
        let item = tokio::select! {
            biased;
            received = self.events.message() => match received {
                Err(e) if broken(&e) => Err(self.park(&e).await),
                received => self.event(received),
            },
            reason = self.call.gone() => Err(self.call.abandon(reason)),
        };
        if !matches!(item, Ok(Some(_))) {
            self.done = true;
        }
        item
    }

    /// H2: the work stream broke on the transport (`e`): the request parks, with no output and its slot still
    /// charged, until the first of: the worker is gone ([`RemoteCall::gone`]); it reconnected at the call's epoch
    /// ([`RemoteCall::reconnected`]): a violation, as no stream resumes in M3.5; the request's cancel (its
    /// deadline or its client gone): the request then ends with that cancel's reason, not this failure.
    async fn park(&mut self, e: &Status) -> BackendError {
        let call = &self.call;
        eprintln!(
            "cered remote stream parked rid={} worker={}@{}: {}",
            call.work.request_id,
            call.worker_id,
            call.at.worker_epoch,
            status_text(e)
        );
        tokio::select! {
            biased;
            reason = call.gone() => call.abandon(reason),
            () = call.reconnected() => {
                self.violation("the worker reconnected at the same epoch after its work stream broke")
            }
            () = call.cancel_requested() => protocol("the parked work stream was cancelled"),
        }
    }

    /// The checks of a `received` event run in order: size, terminal, offset, then the payload's JSON.
    fn event(
        &mut self,
        received: Result<Option<pb::WorkEvent>, Status>,
    ) -> Result<Option<Value>, BackendError> {
        let event = match received {
            Err(e) => return Err(self.violation(&format!("StreamWork: {}", status_text(&e)))),
            Ok(None) => {
                return Err(self.violation("the work stream ended without a terminal event"))
            }
            Ok(Some(event)) => event,
        };
        let size = event.bounded_output_event_json.len();
        if size > wire::WORK_EVENT_MAX {
            return Err(self.violation(&format!("a work event of {size} bytes")));
        }
        if event.terminal {
            return self.call.ended(&event.reason_code);
        }
        if event.committed_token_offset != self.offset {
            return Err(self.violation(&format!(
                "a work event at offset {} where {} was due",
                event.committed_token_offset, self.offset
            )));
        }
        self.offset += 1;
        match serde_json::from_slice(&event.bounded_output_event_json) {
            Ok(chunk) => Ok(Some(chunk)),
            Err(e) => Err(self.violation(&format!("unparsable work event: {e}"))),
        }
    }

    /// H2: the agent broke the work event protocol (`detail`): the work is cancelled and the call fails.
    fn violation(&self, detail: &str) -> BackendError {
        self.call.cancel();
        protocol(detail)
    }
}

fn protocol(detail: &str) -> BackendError {
    BackendError::Protocol(detail.to_owned())
}

/// H2: holds `permit`, the slot of `call` whose work may still hold the agent's resources ([`RemoteCall::held`]),
/// counted in the worker's `retained` meanwhile, until the agent reports the operation cleaned up (polled after
/// [`ORPHAN_FIRST_POLL`], then at doubling gaps up to `period`) or the request's retention bound
/// (`admitted_ms + retention_ms`) passed. Journals the hold first
/// (`orphan_held`, [`Ledger::hold_remote_orphan`]; the executor may journal the request's terminal before or
/// after it), so a restart before the release charges the slot again; journals the release
/// (`cleanup_acknowledged` or `retention`), then uncounts the slot and releases it. A failed hold or release
/// write is logged and the hold goes on.
pub(crate) async fn hold_orphan(
    call: Arc<RemoteCall>,
    permit: Permit,
    ledger: Weak<Mutex<Ledger>>,
    retained: Arc<AtomicU64>,
    period: Duration,
    retention_ms: u64,
) {
    retained.fetch_add(1, Ordering::SeqCst);
    let rid = call.work.request_id.as_str();
    let worker = call.worker_id.as_str();
    // The Q11 `worker=<id>@<epoch>` log field.
    let worker_log = format!("{worker}@{}", call.at.worker_epoch);
    // Journaled before the first poll: from here on a restart charges the slot again.
    let (admitted_ms, hold) = match ledger.upgrade() {
        Some(ledger) => {
            let mut ledger = lock(&ledger);
            let hold = ledger.hold_remote_orphan(rid, worker, unix_ms());
            (ledger.admitted_ms(rid).unwrap_or(0), Some(hold))
        }
        None => (0, None),
    };
    match hold {
        Some(Ok(true)) | None => {}
        Some(Ok(false)) => eprintln!(
            "cered remote orphan rid={rid} worker={worker_log}: not holdable in the ledger; hold not journaled"
        ),
        Some(Err(e)) => eprintln!(
            "cered remote orphan: cannot journal the hold rid={rid} worker={worker_log}: {e:?}"
        ),
    }
    let release_ms = admitted_ms.saturating_add(retention_ms);
    eprintln!("cered remote orphan held rid={rid} worker={worker_log} until unix_ms={release_ms}");
    let bound = Instant::now() + Duration::from_millis(release_ms.saturating_sub(unix_ms()));
    let polls = async {
        let mut failing = false;
        let mut gap = ORPHAN_FIRST_POLL.min(period);
        loop {
            tokio::time::sleep(gap).await;
            gap = (gap * 2).min(period);
            match call.cleaned_up().await {
                Ok(true) => return,
                Ok(false) => failing = false,
                Err(e) => {
                    if !failing {
                        eprintln!(
                            "cered remote orphan poll failed rid={rid} worker={worker_log}: {}; held",
                            status_text(&e)
                        );
                    }
                    failing = true;
                }
            }
        }
    };
    let cleaned = tokio::time::timeout_at(bound, polls).await.is_ok();
    let cause = if cleaned {
        OrphanReleaseCause::CleanupAcknowledged
    } else {
        OrphanReleaseCause::Retention
    };
    let journaled = ledger.upgrade().map(|ledger| {
        let mut ledger = lock(&ledger);
        if cleaned {
            ledger.release_remote_orphan(rid, worker, unix_ms())
        } else {
            ledger.release_orphan(rid, cause, unix_ms()).map(|()| true)
        }
    });
    match journaled {
        Some(Ok(true)) | None => {}
        Some(Ok(false)) => eprintln!(
            "cered remote orphan rid={rid} worker={worker_log}: not held in the ledger; release not journaled"
        ),
        Some(Err(e)) => eprintln!(
            "cered remote orphan: cannot journal the release rid={rid} worker={worker_log}: {e:?}"
        ),
    }
    eprintln!(
        "cered remote orphan released rid={rid} worker={worker_log} cause={}",
        cause.as_str()
    );
    // Uncount first: shutdown may briefly see the slot as open, never an open slot as retained.
    retained.fetch_sub(1, Ordering::SeqCst);
    drop(permit);
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A gRPC status as `code reason` (the agent puts its `wire` reason in the message).
fn status_text(status: &Status) -> String {
    format!("{:?} {}", status.code(), status.message())
}

/// P7: the admission cap of remote worker `worker`: `max_inflight`, lowered to the inventory's
/// `max_running_requests` once an inventory with a confirmed backend identity was read.
pub(crate) fn admission_cap(worker: &RemoteWorkerConfig, state: &RemoteState) -> u64 {
    state
        .inventory
        .as_ref()
        .and_then(|i| i.max_running_requests)
        .map_or(worker.max_inflight, |n| n.min(worker.max_inflight))
}

/// Pushes remote worker `worker`'s admission cap and then its registry view (health, epoch, drain intent, lease;
/// aged at each grant) into `admission`. The cap goes first, so a view turning Ready never grants above it.
pub(crate) fn push_view(
    registry: &RemoteRegistry,
    admission: &Admission,
    worker: &RemoteWorkerConfig,
) {
    let id = worker.worker_id.as_str();
    if let Some(state) = registry.get(id, unix_ms()) {
        admission.set_cap(id, admission_cap(worker, &state));
        // Team pools (C19): the agent's owner-only report (local switch or window) as of its last heartbeat.
        admission.set_owner_only(id, state.owner_only);
    }
    if let Some(view) = registry.worker_view(id) {
        admission.update_worker(id, view);
    }
}

/// The view pump (P7): pushes every remote worker's view and cap on every round bump and every `period` (half
/// the heartbeat interval, so a renewed lease reaches admission well before the old one runs out). Stops once
/// the ledger is gone.
pub(crate) fn spawn_view_pump(
    registry: Arc<RemoteRegistry>,
    admission: Admission,
    workers: Vec<RemoteWorkerConfig>,
    ledger: Weak<Mutex<Ledger>>,
    period: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut rounds = registry.subscribe();
        while ledger.strong_count() > 0 {
            for worker in &workers {
                push_view(&registry, &admission, worker);
            }
            tokio::select! {
                changed = rounds.changed() => if changed.is_err() { return },
                () = tokio::time::sleep(period) => {}
            }
        }
    })
}
