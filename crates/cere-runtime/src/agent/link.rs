//! The agent's link to its coordinator (m35 design note Q4, Q6, Q7, Q7a, Q13): one task over one mTLS channel
//! to `[coordinator] endpoint` (`Pki::client_tls(coordinator principal)`, every call bounded by `rpc_timeout_ms`).
//!
//! Each heartbeat interval (`MissedTickBehavior::Delay`, calls strictly sequential) the task first runs
//! [`Shared::tick`] (journal upkeep, lease expiry, the self-lost rule), then sends either a `Heartbeat` (registered
//! under the current term and epoch; sequence strictly increasing per epoch) or a `Register` (after the first
//! confirmed backend; retried every interval until accepted). What a reply does (lease, term fence, incarnation
//! jumps, drain, reconciliation) is [`Shared::on_register`] / [`Shared::on_heartbeat`].

use std::sync::Arc;
use std::time::Duration;

use cere_proto::worker::v2 as pb;
use pb::worker_registry_service_client::WorkerRegistryServiceClient;
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tonic::transport::Channel;

use crate::agent::node::{stopped, Beat, Shared};
use crate::fed::wire;

/// Logs a failed call once per distinct code, so a coordinator outage is one line, not one per interval.
struct CallLog(Option<tonic::Code>);

impl CallLog {
    fn failed(&mut self, call: &str, status: &tonic::Status) {
        if self.0 != Some(status.code()) {
            eprintln!(
                "cere-worker: {call} failed: {:?} {}",
                status.code(),
                wire::reason(status)
            );
            self.0 = Some(status.code());
        }
    }

    fn ok(&mut self) {
        self.0 = None;
    }
}

pub(crate) async fn run(shared: Arc<Shared>, channel: Channel, stop: watch::Receiver<bool>) {
    let mut client = WorkerRegistryServiceClient::new(channel)
        .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX);
    let mut tick = tokio::time::interval(Duration::from_millis(
        shared.config.limits.heartbeat_interval_ms,
    ));
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let stop = stopped(stop);
    tokio::pin!(stop);
    let mut log = CallLog(None);
    loop {
        tokio::select! {
            () = &mut stop => return,
            _ = tick.tick() => {}
        }
        shared.tick();
        if let Some(beat) = shared.next_beat() {
            let request = heartbeat_request(&shared, &beat);
            let reply = tokio::select! {
                () = &mut stop => return,
                reply = client.heartbeat(request) => reply,
            };
            match reply {
                Ok(reply) => {
                    log.ok();
                    shared.on_heartbeat(&beat, reply.get_ref());
                }
                Err(status) => log.failed("Heartbeat", &status),
            }
        } else if let Some((epoch, inventory_sha256)) = shared.register_target() {
            let request = pb::RegisterRequest {
                proposed: Some(identity(&shared, epoch)),
                boot_id: shared.boot_id.clone(),
                protocol_revision: cere_proto::PROTOCOL_REVISION,
                inventory_sha256: inventory_sha256.to_vec(),
            };
            let reply = tokio::select! {
                () = &mut stop => return,
                reply = client.register(request) => reply,
            };
            match reply {
                Ok(reply) => {
                    log.ok();
                    shared.on_register(epoch, reply.get_ref());
                }
                Err(status) => log.failed("Register", &status),
            }
        }
    }
}

fn identity(shared: &Shared, epoch: u64) -> pb::WorkerIdentity {
    pb::WorkerIdentity {
        cluster_id: shared.config.cluster_id.clone(),
        worker_id: shared.config.worker_id.clone(),
        worker_epoch: epoch,
    }
}

fn heartbeat_request(shared: &Shared, beat: &Beat) -> pb::HeartbeatRequest {
    pb::HeartbeatRequest {
        worker: Some(identity(shared, beat.epoch)),
        sequence: beat.sequence,
        observed_coordinator_term: beat.term,
        inventory_sha256: beat.inventory_sha256.to_vec(),
        model_generation: beat.generation,
        admission_state: beat.admission.as_str().to_owned(),
        owner_only: beat.owner_only,
    }
}
