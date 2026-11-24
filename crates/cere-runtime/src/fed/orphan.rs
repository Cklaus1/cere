//! Coordinator-restart release of remote orphans: a prior-term request dispatched to a remote worker, or a
//! request whose H2 hold (`orphan_held`) was not released when the previous incarnation ended, stays charged to
//! that worker until its agent reports the operation `cleanup_acknowledged` (or no longer knows it), or until
//! its retention bound (m35 design note Q15). Track ORPHAN-B (build-loop-m35 design note).
//!
//! Startup recovery failed the request `worker_restart` (or re-charged a held one, whatever its terminal); when
//! its last `dispatched` record names a configured remote worker at its configured endpoint, `App::open` charges
//! the slot to that worker's cap and hands it to a [`RemoteOrphan`]. The hold counts the slot in the worker's
//! `retained` (it holds no coordinator stream) and sends nothing itself: reconciliation
//! (`readiness::RemoteDriver::reconcile`) cancels the operation, which no live ledger request owns, once the
//! agent joins. The hold asks the agent for the operation (`GetOperation`, operation id = the rid, in the
//! orphan's scope) every `period`, starting one period after the registry holds the agent's accepted Register of
//! this term: the agent has fenced the older term by then (Q6), so an answer is the new term's view. The
//! authority comes from the worker's registry state at each poll (its registered epoch, this term, its last
//! generation). A `cleanup_acknowledged` report or `NOT_FOUND unknown_operation` releases the
//! slot early (journaled through `Ledger::release_remote_orphan`); otherwise it is released at
//! `admitted_ms + terminal_retention_ms` (journaled `retention`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::Duration;

use tokio::time::Instant;
use tonic::{Code, Status};

use crate::admission::Permit;
use crate::app::{unix_ms, worker_field};
use crate::fed::client::{AuthorityBase, WorkerClient};
use crate::fed::registry::RemoteRegistry;
use crate::fed::wire;
use crate::journal::OrphanReleaseCause;
use crate::ledger::{Ledger, Orphan};

/// One recovered orphan's slot, charged to the remote worker of its last `dispatched` record.
pub(crate) struct RemoteOrphan {
    /// The worker charged with the slot.
    pub(crate) worker_id: String,
    /// The worker's agent client.
    pub(crate) client: WorkerClient,
    /// The registry of this term: the hold polls only after the worker's accepted Register in it.
    pub(crate) registry: Arc<RemoteRegistry>,
    pub(crate) cluster_id: String,
    pub(crate) coordinator_id: String,
    /// Weak: a hold never keeps the ledger (and its state-dir lock) alive past the `App`.
    pub(crate) ledger: Weak<Mutex<Ledger>>,
    /// The worker's count of held slots that only wait for their release (`RemoteWorker::retained`).
    pub(crate) retained: Arc<AtomicU64>,
    /// Gap between two `GetOperation` polls (`orphan_load_poll_ms`, at least 1 s).
    pub(crate) period: Duration,
    /// `admitted_ms + terminal_retention_ms`: the latest release, never extended by a restart.
    pub(crate) release_ms: u64,
}

impl RemoteOrphan {
    /// Holds `permit`, counted retained, until the agent reports `orphan`'s operation cleaned up or the retention
    /// bound passed; journals the release, then uncounts the slot and releases it.
    pub(crate) async fn run(self, orphan: Orphan, permit: Permit) {
        let rid = orphan.rid.as_str();
        let worker = self.worker_id.as_str();
        let worker_log = worker_field(orphan.dispatched.as_ref());
        self.retained.fetch_add(1, Ordering::SeqCst);
        eprintln!(
            "cered recovery: held rid={rid} worker={worker_log} until unix_ms={}; released earlier once its agent reports the cleanup in term {}",
            self.release_ms,
            self.registry.term()
        );
        let bound =
            Instant::now() + Duration::from_millis(self.release_ms.saturating_sub(unix_ms()));
        let cleaned = tokio::time::timeout_at(bound, self.cleaned_up(&orphan, &worker_log))
            .await
            .is_ok();
        let journaled = self.ledger.upgrade().map(|ledger| {
            let mut ledger = lock(&ledger);
            if cleaned {
                ledger.release_remote_orphan(rid, worker, unix_ms())
            } else {
                ledger
                    .release_orphan(rid, OrphanReleaseCause::Retention, unix_ms())
                    .map(|()| true)
            }
        });
        match journaled {
            Some(Ok(true)) | None => {}
            Some(Ok(false)) => eprintln!(
                "cered recovery: rid={rid} worker={worker_log} not held in the ledger; release not journaled"
            ),
            Some(Err(e)) => eprintln!(
                "cered recovery: cannot journal the release rid={rid} worker={worker_log}: {e:?}"
            ),
        }
        let cause = if cleaned {
            OrphanReleaseCause::CleanupAcknowledged
        } else {
            OrphanReleaseCause::Retention
        };
        eprintln!(
            "cered recovery: released rid={rid} worker={worker_log} cause={}",
            cause.as_str()
        );
        // Uncount first: shutdown may briefly see the slot as open, never an open slot as retained.
        self.retained.fetch_sub(1, Ordering::SeqCst);
        drop(permit);
    }

    /// Returns once the agent, registered in this term, reports `orphan`'s operation `cleanup_acknowledged` or
    /// no longer knows it; polls every `period`, the first one a period after the Register.
    async fn cleaned_up(&self, orphan: &Orphan, worker_log: &str) {
        self.registered().await;
        let mut failing = false;
        loop {
            tokio::time::sleep(self.period).await;
            let at = self.registered().await;
            match self.query(&at, orphan).await {
                Ok(true) => return,
                Ok(false) => failing = false,
                Err(e) => {
                    if !failing {
                        eprintln!(
                            "cered recovery: poll failed rid={} worker={worker_log}: {:?} {}; held",
                            orphan.rid,
                            e.code(),
                            e.message()
                        );
                    }
                    failing = true;
                }
            }
        }
    }

    /// The read authority of the worker once the registry holds its accepted Register of this term
    /// ([`registered_authority`]). Waits for the Register (each one bumps the registry's round).
    async fn registered(&self) -> AuthorityBase {
        let mut rounds = self.registry.subscribe();
        loop {
            let at = registered_authority(
                &self.registry,
                &self.worker_id,
                &self.cluster_id,
                &self.coordinator_id,
            );
            if let Some(at) = at {
                return at;
            }
            if rounds.changed().await.is_err() {
                // The registry is gone: only the retention bound releases the slot.
                std::future::pending::<()>().await;
            }
        }
    }

    /// `GetOperation` of the orphan's Submit (operation id = the rid): `Ok(true)` once the agent reports it
    /// `cleanup_acknowledged` or no longer knows it (`NOT_FOUND unknown_operation`).
    async fn query(&self, at: &AuthorityBase, orphan: &Orphan) -> Result<bool, Status> {
        match self
            .client
            .get_operation(at, &orphan.scope, &orphan.rid)
            .await
        {
            Ok(status) => Ok(status.cleanup_acknowledged),
            Err(e) if e.code() == Code::NotFound && e.message() == wire::UNKNOWN_OPERATION => {
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }
}

/// The read authority of remote worker `worker_id` from its registry state now: its accepted Register of this
/// term (the registered epoch, this term, its last generation); `None` before one. A hold asks the agent under
/// it, never under the authority its work was granted with (a restart at a higher epoch refuses that one).
pub(crate) fn registered_authority(
    registry: &RemoteRegistry,
    worker_id: &str,
    cluster_id: &str,
    coordinator_id: &str,
) -> Option<AuthorityBase> {
    let state = registry.get(worker_id, unix_ms())?;
    Some(AuthorityBase {
        cluster_id: cluster_id.to_owned(),
        worker_id: worker_id.to_owned(),
        worker_epoch: state.epoch?,
        coordinator_id: coordinator_id.to_owned(),
        coordinator_term: state.term?,
        model_generation: state.generation,
    })
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
