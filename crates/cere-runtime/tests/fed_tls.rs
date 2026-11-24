//! P2 mTLS, principals and decode limits over loopback gRPC (m35 design note Q2, Q14). Stub services call the
//! `Authz` gate first and count every call that passed it: a denied call must leave the counter at 0.

mod support;

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cere_proto::worker::v2 as pb;
use cere_runtime::fed::tls::{self, Authz, FedError, Pki};
use cere_runtime::fed::wire;
use pb::worker_inference_service_client::WorkerInferenceServiceClient;
use pb::worker_inference_service_server::{WorkerInferenceService, WorkerInferenceServiceServer};
use pb::worker_registry_service_client::WorkerRegistryServiceClient;
use pb::worker_registry_service_server::{WorkerRegistryService, WorkerRegistryServiceServer};
use pb::worker_service_client::WorkerServiceClient;
use pb::worker_service_server::{WorkerService, WorkerServiceServer};
use prost::Message;
use support::pki::{Leaf, San, TestPki};
use support::StateDir;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_stream::Stream;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity, Server};
use tonic::{Code, Request, Response, Status};

const COORD: &str = "cere-coordinator";
const WORKER_A: &str = "cere-worker-a";
const WORKER_B: &str = "cere-worker-b";

/// Calls that passed the authorization gate, plus the length of the last peer chain seen there.
#[derive(Clone, Default)]
struct Counter {
    passed: Arc<AtomicUsize>,
    chain_len: Arc<AtomicUsize>,
}

impl Counter {
    fn passed(&self) -> usize {
        self.passed.load(Ordering::SeqCst)
    }

    fn bump<T>(&self, req: &Request<T>) {
        let chain = req.peer_certs().map_or(0, |certs| certs.len());
        self.chain_len.store(chain, Ordering::SeqCst);
        self.passed.fetch_add(1, Ordering::SeqCst);
    }
}

struct Registry {
    authz: Authz,
    counter: Counter,
}

#[tonic::async_trait]
impl WorkerRegistryService for Registry {
    async fn register(
        &self,
        req: Request<pb::RegisterRequest>,
    ) -> Result<Response<pb::RegisterReply>, Status> {
        let claimed = req
            .get_ref()
            .proposed
            .as_ref()
            .map(|p| p.worker_id.clone())
            .unwrap_or_default();
        self.authz.registry_caller(&req, &claimed)?;
        self.counter.bump(&req);
        Ok(Response::new(pb::RegisterReply {
            accepted: true,
            ..Default::default()
        }))
    }

    async fn heartbeat(
        &self,
        req: Request<pb::HeartbeatRequest>,
    ) -> Result<Response<pb::HeartbeatReply>, Status> {
        let claimed = req
            .get_ref()
            .worker
            .as_ref()
            .map(|w| w.worker_id.clone())
            .unwrap_or_default();
        self.authz.registry_caller(&req, &claimed)?;
        self.counter.bump(&req);
        Ok(Response::new(pb::HeartbeatReply::default()))
    }
}

struct Worker {
    authz: Authz,
    counter: Counter,
}

impl Worker {
    fn admit<T>(&self, req: &Request<T>) -> Result<(), Status> {
        self.authz.coordinator_caller(req)?;
        self.counter.bump(req);
        Ok(())
    }
}

type EventStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl WorkerService for Worker {
    async fn describe_resources(
        &self,
        req: Request<pb::InventoryRequest>,
    ) -> Result<Response<pb::InventoryReply>, Status> {
        self.admit(&req)?;
        Ok(Response::new(pb::InventoryReply::default()))
    }

    async fn load_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.admit(&req)?;
        Err(Status::unimplemented(""))
    }

    async fn drain_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.admit(&req)?;
        Err(Status::unimplemented(""))
    }

    async fn unload_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.admit(&req)?;
        Err(Status::unimplemented(""))
    }

    async fn cancel_work(
        &self,
        req: Request<pb::CancelWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.admit(&req)?;
        Ok(Response::new(pb::OperationReceipt {
            accepted: true,
            ..Default::default()
        }))
    }

    async fn get_operation(
        &self,
        req: Request<pb::OperationQuery>,
    ) -> Result<Response<pb::OperationStatus>, Status> {
        self.admit(&req)?;
        Ok(Response::new(pb::OperationStatus::default()))
    }

    async fn list_operations(
        &self,
        req: Request<pb::ListOperationsRequest>,
    ) -> Result<Response<pb::ListOperationsReply>, Status> {
        self.admit(&req)?;
        Ok(Response::new(pb::ListOperationsReply::default()))
    }

    type StreamWorkStream = EventStream<pb::WorkEvent>;

    async fn stream_work(
        &self,
        req: Request<pb::WorkStreamRequest>,
    ) -> Result<Response<Self::StreamWorkStream>, Status> {
        self.admit(&req)?;
        Ok(Response::new(Box::pin(tokio_stream::empty())))
    }

    type StreamTelemetryStream = EventStream<pb::TelemetryEnvelope>;

    async fn stream_telemetry(
        &self,
        req: Request<pb::TelemetryRequest>,
    ) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        self.admit(&req)?;
        Err(Status::unimplemented(""))
    }
}

#[tonic::async_trait]
impl WorkerInferenceService for Worker {
    async fn submit_work(
        &self,
        req: Request<pb::SubmitWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.admit(&req)?;
        Ok(Response::new(pb::OperationReceipt {
            accepted: true,
            ..Default::default()
        }))
    }
}

/// One cluster CA plus a temp dir holding its PEM files.
struct Cluster {
    pki: TestPki,
    dir: StateDir,
    ca: PathBuf,
}

impl Cluster {
    fn new() -> Cluster {
        let pki = TestPki::new();
        let dir = StateDir::new();
        let ca = pki.write_ca(dir.path());
        Cluster { pki, dir, ca }
    }

    fn node(&self, principal: &str) -> Pki {
        let (cert, key) = self.pki.leaf(principal).write(self.dir.path());
        Pki::load(&self.ca, &cert, &key, principal).unwrap()
    }

    /// A client config built by hand (the Q2-violating leaves cannot pass `Pki::load`).
    fn raw_client(&self, cert_pem: &str, key_pem: &str, peer: &str) -> ClientTlsConfig {
        ClientTlsConfig::new()
            .ca_certificate(Certificate::from_pem(self.pki.ca_pem()))
            .identity(Identity::from_pem(cert_pem, key_pem))
            .domain_name(peer)
    }

    fn leaf_client(&self, leaf: &Leaf, peer: &str) -> ClientTlsConfig {
        self.raw_client(leaf.cert_pem(), leaf.key_pem(), peer)
    }
}

async fn listener() -> (tokio::net::TcpListener, SocketAddr) {
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

/// Serves one generated service over mTLS on its own loopback listener (one `Server` per service keeps the
/// per-service decode limits independent of any routing).
macro_rules! spawn_tls {
    ($pki:expr, $svc:expr) => {{
        let (listener, addr) = listener().await;
        let server = Server::builder().tls_config($pki.server_tls()).unwrap();
        tokio::spawn(server.serve_with_incoming($svc, TcpListenerStream::new(listener)));
        addr
    }};
}

async fn serve_registry(pki: &Pki, authz: Authz) -> (SocketAddr, Counter) {
    let counter = Counter::default();
    let svc = WorkerRegistryServiceServer::new(Registry {
        authz,
        counter: counter.clone(),
    })
    .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX);
    (spawn_tls!(pki, svc), counter)
}

/// The agent-side services: control (`WorkerService`) and inference (`WorkerInferenceService`).
struct WorkerAddrs {
    control: SocketAddr,
    inference: SocketAddr,
}

async fn serve_worker(pki: &Pki, authz: Authz) -> (WorkerAddrs, Counter) {
    let counter = Counter::default();
    let control = WorkerServiceServer::new(Worker {
        authz: authz.clone(),
        counter: counter.clone(),
    })
    .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX);
    let inference = WorkerInferenceServiceServer::new(Worker {
        authz,
        counter: counter.clone(),
    })
    .max_decoding_message_size(wire::INFERENCE_MESSAGE_MAX);
    let addrs = WorkerAddrs {
        control: spawn_tls!(pki, control),
        inference: spawn_tls!(pki, inference),
    };
    (addrs, counter)
}

/// Connects lazily so a handshake refusal surfaces on the first call as a transport error.
fn channel(addr: SocketAddr, tls: ClientTlsConfig) -> Channel {
    Endpoint::from_shared(format!("https://{addr}"))
        .unwrap()
        .timeout(Duration::from_secs(10))
        .tls_config(tls)
        .unwrap()
        .connect_lazy()
}

fn coordinator_authz() -> Authz {
    Authz::coordinator_side(BTreeMap::from([
        (WORKER_A.to_owned(), "worker-a".to_owned()),
        (WORKER_B.to_owned(), "worker-b".to_owned()),
    ]))
}

fn register(worker_id: &str) -> pb::RegisterRequest {
    pb::RegisterRequest {
        proposed: Some(pb::WorkerIdentity {
            cluster_id: "c".into(),
            worker_id: worker_id.into(),
            worker_epoch: 1,
        }),
        protocol_revision: cere_proto::PROTOCOL_REVISION,
        ..Default::default()
    }
}

fn heartbeat(worker_id: &str) -> pb::HeartbeatRequest {
    pb::HeartbeatRequest {
        worker: Some(pb::WorkerIdentity {
            cluster_id: "c".into(),
            worker_id: worker_id.into(),
            worker_epoch: 1,
        }),
        sequence: 1,
        ..Default::default()
    }
}

fn assert_denied(result: Result<impl std::fmt::Debug, Status>) {
    let status = result.expect_err("call must be refused");
    assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");
    assert_eq!(wire::reason(&status), wire::PERMISSION_DENIED_REASON);
}

/// A transport-level refusal: the call never reached a service. The status code tonic picks depends on when the
/// TLS alert arrives (UNAVAILABLE on connect, UNKNOWN mid-request), so the check is that the status wraps a
/// transport error rather than being an authorization status.
fn assert_handshake_failed(result: Result<impl std::fmt::Debug, Status>) {
    let status = result.expect_err("handshake must fail");
    assert_ne!(status.code(), Code::PermissionDenied, "{status:?}");
    let transport = std::error::Error::source(&status)
        .is_some_and(|source| source.is::<tonic::transport::Error>());
    assert!(transport, "not a transport error: {status:?}");
}

#[tokio::test]
async fn bound_principal_succeeds_and_leaf_is_first_in_peer_chain() {
    let cluster = Cluster::new();
    let coord = cluster.node(COORD);
    let (addr, counter) = serve_registry(&coord, coordinator_authz()).await;
    let worker = cluster.node(WORKER_A);
    let mut client = WorkerRegistryServiceClient::new(channel(addr, worker.client_tls(COORD)));
    let reply = client.register(register("worker-a")).await.unwrap();
    assert!(reply.get_ref().accepted);
    client.heartbeat(heartbeat("worker-a")).await.unwrap();
    assert_eq!(counter.passed(), 2);
    assert_eq!(counter.chain_len.load(Ordering::SeqCst), 1);

    // [INFERENCE] check (spec P2): with a leaf + CA chain presented, tonic's peer_certs() lists the leaf first.
    // Were the CA first, principal extraction would fail (the CA has no SAN) and the call would be denied.
    let leaf = cluster.pki.leaf(WORKER_A);
    let chain = format!("{}{}", leaf.cert_pem(), cluster.pki.ca_pem());
    let tls = cluster.raw_client(&chain, leaf.key_pem(), COORD);
    let mut client = WorkerRegistryServiceClient::new(channel(addr, tls));
    client.register(register("worker-a")).await.unwrap();
    assert_eq!(counter.passed(), 3);
    assert_eq!(counter.chain_len.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn foreign_ca_client_cert_fails_the_handshake() {
    let cluster = Cluster::new();
    let coord = cluster.node(COORD);
    let (addr, counter) = serve_registry(&coord, coordinator_authz()).await;
    let foreign = TestPki::new().leaf(WORKER_A);
    let mut client =
        WorkerRegistryServiceClient::new(channel(addr, cluster.leaf_client(&foreign, COORD)));
    assert_handshake_failed(client.register(register("worker-a")).await);
    assert_eq!(counter.passed(), 0);
}

#[tokio::test]
async fn san_rule_violations_are_denied_before_the_handler() {
    let cluster = Cluster::new();
    let coord = cluster.node(COORD);
    let (addr, counter) = serve_registry(&coord, coordinator_authz()).await;
    let ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
    let cases: Vec<(&str, Vec<San>)> = vec![
        ("zero SANs", vec![]),
        (
            "two DNS SANs",
            vec![San::Dns(WORKER_A.into()), San::Dns(WORKER_B.into())],
        ),
        ("IP SAN", vec![San::Ip(ip)]),
        (
            "DNS + IP SANs",
            vec![San::Dns(WORKER_A.into()), San::Ip(ip)],
        ),
        ("URI SAN", vec![San::Uri("spiffe://cere/worker-a".into())]),
        ("email SAN", vec![San::Email("worker-a@cere.test".into())]),
    ];
    for (name, sans) in cases {
        let leaf = cluster.pki.leaf_with_sans(&sans);
        assert!(
            tls::principal_from_der(leaf.der()).is_err(),
            "{name}: principal_from_der must refuse"
        );
        let mut client =
            WorkerRegistryServiceClient::new(channel(addr, cluster.leaf_client(&leaf, COORD)));
        let result = client.register(register("worker-a")).await;
        let status = result.expect_err(name);
        assert_eq!(status.code(), Code::PermissionDenied, "{name}: {status:?}");
        assert_eq!(
            wire::reason(&status),
            wire::PERMISSION_DENIED_REASON,
            "{name}"
        );
        assert_eq!(counter.passed(), 0, "{name}");
    }
}

#[tokio::test]
async fn unbound_principal_and_foreign_worker_id_are_denied() {
    let cluster = Cluster::new();
    let coord = cluster.node(COORD);
    let (addr, counter) = serve_registry(&coord, coordinator_authz()).await;

    // A CA-valid principal with no binding.
    let stranger = cluster.node("cere-worker-z");
    let mut client = WorkerRegistryServiceClient::new(channel(addr, stranger.client_tls(COORD)));
    assert_denied(client.register(register("worker-z")).await);
    assert_denied(client.heartbeat(heartbeat("worker-a")).await);

    // The coordinator's own principal is not a worker binding either.
    let mut client = WorkerRegistryServiceClient::new(channel(addr, coord.client_tls(COORD)));
    assert_denied(client.register(register("worker-a")).await);

    // A bound worker principal asserting another worker id (caller-supplied ids are never identity).
    let worker_a = cluster.node(WORKER_A);
    let mut client = WorkerRegistryServiceClient::new(channel(addr, worker_a.client_tls(COORD)));
    assert_denied(client.register(register("worker-b")).await);
    assert_denied(client.heartbeat(heartbeat("worker-b")).await);
    assert_denied(client.register(register("")).await);
    assert_eq!(counter.passed(), 0);
}

#[tokio::test]
async fn only_the_coordinator_principal_may_call_the_worker_services() {
    let cluster = Cluster::new();
    let agent = cluster.node(WORKER_A);
    let (addrs, counter) = serve_worker(&agent, Authz::agent_side(COORD.to_owned())).await;

    for principal in [WORKER_B, WORKER_A, "cere-coordinator-2"] {
        let caller = cluster.node(principal);
        let mut control =
            WorkerServiceClient::new(channel(addrs.control, caller.client_tls(WORKER_A)));
        assert_denied(
            control
                .describe_resources(pb::InventoryRequest::default())
                .await,
        );
        assert_denied(control.cancel_work(pb::CancelWorkRequest::default()).await);
        assert_denied(control.stream_work(pb::WorkStreamRequest::default()).await);
        let mut inference = WorkerInferenceServiceClient::new(channel(
            addrs.inference,
            caller.client_tls(WORKER_A),
        ));
        assert_denied(
            inference
                .submit_work(pb::SubmitWorkRequest::default())
                .await,
        );
    }
    assert_eq!(counter.passed(), 0);

    let coord = cluster.node(COORD);
    WorkerServiceClient::new(channel(addrs.control, coord.client_tls(WORKER_A)))
        .describe_resources(pb::InventoryRequest::default())
        .await
        .unwrap();
    WorkerInferenceServiceClient::new(channel(addrs.inference, coord.client_tls(WORKER_A)))
        .submit_work(pb::SubmitWorkRequest::default())
        .await
        .unwrap();
    assert_eq!(counter.passed(), 2);
}

#[tokio::test]
async fn coordinator_side_authz_refuses_worker_service_calls_and_agent_side_refuses_registry() {
    // Role table: the coordinator never serves WorkerService; an agent never serves the registry.
    let cluster = Cluster::new();
    let coord = cluster.node(COORD);
    let (addrs, counter) = serve_worker(&coord, coordinator_authz()).await;
    let worker_a = cluster.node(WORKER_A);
    let mut control = WorkerServiceClient::new(channel(addrs.control, worker_a.client_tls(COORD)));
    assert_denied(
        control
            .describe_resources(pb::InventoryRequest::default())
            .await,
    );
    let mut inference =
        WorkerInferenceServiceClient::new(channel(addrs.inference, worker_a.client_tls(COORD)));
    assert_denied(
        inference
            .submit_work(pb::SubmitWorkRequest::default())
            .await,
    );
    assert_eq!(counter.passed(), 0);

    let agent = cluster.node(WORKER_B);
    let (addr, counter) = serve_registry(&agent, Authz::agent_side(COORD.to_owned())).await;
    let mut client = WorkerRegistryServiceClient::new(channel(addr, coord.client_tls(WORKER_B)));
    assert_denied(client.register(register("worker-b")).await);
    assert_eq!(counter.passed(), 0);
}

#[tokio::test]
async fn client_expecting_another_principal_fails_the_handshake() {
    let cluster = Cluster::new();
    // The agent serves with a certificate naming cere-worker-b; the coordinator expects cere-worker-a.
    let impostor = cluster.node(WORKER_B);
    let (addrs, counter) = serve_worker(&impostor, Authz::agent_side(COORD.to_owned())).await;
    let coord = cluster.node(COORD);
    let mut control = WorkerServiceClient::new(channel(addrs.control, coord.client_tls(WORKER_A)));
    assert_handshake_failed(
        control
            .describe_resources(pb::InventoryRequest::default())
            .await,
    );
    assert_eq!(counter.passed(), 0);
}

/// A message whose encoding is exactly `target` bytes, padded through one bytes/string field.
fn sized<M: Message>(target: usize, make: impl Fn(usize) -> M) -> M {
    // The pad's own length prefix varies by a few bytes; walk down from the target until it fits exactly.
    let mut pad = target;
    loop {
        let msg = make(pad);
        match msg.encoded_len() {
            len if len == target => return msg,
            len if len > target => pad -= (len - target).min(pad),
            _ => panic!("no padding reaches {target} bytes"),
        }
    }
}

#[tokio::test]
async fn control_and_inference_decode_limits_refuse_before_the_handler() {
    let cluster = Cluster::new();
    let agent = cluster.node(WORKER_A);
    let (addrs, counter) = serve_worker(&agent, Authz::agent_side(COORD.to_owned())).await;
    let coord = cluster.node(COORD);
    let mut control = WorkerServiceClient::new(channel(addrs.control, coord.client_tls(WORKER_A)));
    let mut inference =
        WorkerInferenceServiceClient::new(channel(addrs.inference, coord.client_tls(WORKER_A)));

    let cancel = |pad: usize| pb::CancelWorkRequest {
        completion_ticket: "t".repeat(pad),
        ..Default::default()
    };
    let submit = |pad: usize| pb::SubmitWorkRequest {
        bounded_inference_input_json: vec![b'x'; pad],
        ..Default::default()
    };

    let over = control
        .cancel_work(sized(wire::CONTROL_MESSAGE_MAX + 1, cancel))
        .await
        .expect_err("1 MiB + 1 control message");
    assert_eq!(over.code(), Code::OutOfRange, "{over:?}");
    let over = inference
        .submit_work(sized(wire::INFERENCE_MESSAGE_MAX + 1, submit))
        .await
        .expect_err("8 MiB + 1 SubmitWork");
    assert_eq!(over.code(), Code::OutOfRange, "{over:?}");
    assert_eq!(counter.passed(), 0);

    // Exactly at each limit passes.
    control
        .cancel_work(sized(wire::CONTROL_MESSAGE_MAX, cancel))
        .await
        .unwrap();
    inference
        .submit_work(sized(wire::INFERENCE_MESSAGE_MAX, submit))
        .await
        .unwrap();
    // Limits are per service: a SubmitWork above the 1 MiB control limit still reaches the handler.
    inference
        .submit_work(sized(wire::CONTROL_MESSAGE_MAX + 1, submit))
        .await
        .unwrap();
    assert_eq!(counter.passed(), 3);
}

#[test]
fn peer_principal_without_tls_is_unauthenticated() {
    let status = tls::peer_principal(&Request::new(())).unwrap_err();
    assert_eq!(status.code(), Code::PermissionDenied);
    assert_eq!(wire::reason(&status), wire::UNAUTHENTICATED_PEER);
    let denied = Authz::agent_side(COORD.to_owned())
        .coordinator_caller(&Request::new(()))
        .unwrap_err();
    assert_eq!(denied.code(), Code::PermissionDenied);
    assert_eq!(wire::reason(&denied), wire::PERMISSION_DENIED_REASON);
}

#[test]
fn principal_from_der_returns_the_single_dns_san() {
    let pki = TestPki::new();
    assert_eq!(
        tls::principal_from_der(pki.leaf(WORKER_A).der()).unwrap(),
        WORKER_A
    );
    assert!(tls::principal_from_der(b"not a certificate").is_err());
    assert!(tls::principal_from_der(&[]).is_err());
    // A wildcard is not a principal.
    let wildcard = pki.leaf_with_sans(&[San::Dns("*.cere".into())]);
    assert!(tls::principal_from_der(wildcard.der()).is_err());
}

fn load(dir: &Path, ca: &Path, leaf: &Leaf, principal: &str) -> Result<Pki, FedError> {
    let (cert, key) = leaf.write(dir);
    Pki::load(ca, &cert, &key, principal)
}

#[test]
fn pki_load_checks_the_own_leaf() {
    let cluster = Cluster::new();
    let dir = cluster.dir.path();
    let ca = &cluster.ca;
    assert!(load(dir, ca, &cluster.pki.leaf(WORKER_A), WORKER_A).is_ok());
    // The configured principal must be the leaf's single DNS SAN.
    assert!(matches!(
        load(dir, ca, &cluster.pki.leaf(WORKER_A), WORKER_B),
        Err(FedError::PrincipalMismatch { .. })
    ));
    let two = cluster
        .pki
        .leaf_with_sans(&[San::Dns(WORKER_A.into()), San::Dns(WORKER_B.into())]);
    assert!(load(dir, ca, &two, WORKER_A).is_err());
    let missing = dir.join("missing.pem");
    let (cert, key) = cluster.pki.leaf(WORKER_A).write(dir);
    assert!(Pki::load(&missing, &cert, &key, WORKER_A).is_err());
    assert!(Pki::load(ca, &missing, &key, WORKER_A).is_err());
    assert!(Pki::load(ca, &cert, &missing, WORKER_A).is_err());
    // A key file holding no private key, a CA file holding no certificate.
    assert!(Pki::load(ca, &cert, &cert, WORKER_A).is_err());
    assert!(Pki::load(&key, &cert, &key, WORKER_A).is_err());
}
