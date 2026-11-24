//! The generated bindings are the wire contract (m35 design note Q8): package name, revision, services and the
//! M3.5 amendments are pinned here.
use cere_proto::worker::v2 as pb;
use prost::Message;

/// Minimal hand-written descriptor views (field numbers from descriptor.proto) so the test needs no prost-types.
mod descriptor {
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Method {
        #[prost(string, tag = "1")]
        pub name: String,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Service {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(message, repeated, tag = "2")]
        pub method: Vec<Method>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct File {
        #[prost(string, tag = "2")]
        pub package: String,
        #[prost(message, repeated, tag = "6")]
        pub service: Vec<Service>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Set {
        #[prost(message, repeated, tag = "1")]
        pub file: Vec<File>,
    }
}

fn services() -> Vec<(String, Vec<String>)> {
    let set = descriptor::Set::decode(cere_proto::FILE_DESCRIPTOR_SET).expect("descriptor set");
    assert_eq!(set.file.len(), 1);
    assert_eq!(set.file[0].package, "ai.cere.worker.v2");
    set.file[0]
        .service
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.method.iter().map(|m| m.name.clone()).collect(),
            )
        })
        .collect()
}

#[test]
fn the_compiled_package_is_ai_cere_worker_v2_revision_2() {
    assert_eq!(cere_proto::PACKAGE, "ai.cere.worker.v2");
    assert_eq!(cere_proto::PROTOCOL_REVISION, 2);
    services();
}

#[test]
fn submit_work_lives_only_in_the_inference_service() {
    let s = services();
    let names: Vec<&str> = s.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "WorkerRegistryService",
            "WorkerService",
            "WorkerInferenceService"
        ]
    );
    let methods = |svc: &str| s.iter().find(|(n, _)| n == svc).unwrap().1.clone();
    assert_eq!(methods("WorkerRegistryService"), ["Register", "Heartbeat"]);
    assert_eq!(
        methods("WorkerService"),
        [
            "DescribeResources",
            "LoadModel",
            "DrainModel",
            "UnloadModel",
            "CancelWork",
            "GetOperation",
            "ListOperations",
            "StreamWork",
            "StreamTelemetry"
        ]
    );
    assert_eq!(methods("WorkerInferenceService"), ["SubmitWork"]);
}

#[test]
fn authority_round_trips_with_an_explicit_zero_generation() {
    let a = pb::Authority {
        target: Some(pb::WorkerIdentity {
            cluster_id: "c".into(),
            worker_id: "worker-a".into(),
            worker_epoch: 3,
        }),
        coordinator_id: "cere-coordinator".into(),
        coordinator_term: 7,
        scope_id: "s".into(),
        operation_id: "cere-7-1-ab".into(),
        payload_sha256: vec![0xab; 32],
        remaining_budget_ms: 5000,
        model_generation: Some(0),
    };
    let back = pb::Authority::decode(a.encode_to_vec().as_slice()).unwrap();
    assert_eq!(back, a);
    // proto3 `optional`: an explicit 0 is distinguishable from unset.
    let unset = pb::Authority {
        model_generation: None,
        ..a
    };
    assert_eq!(
        pb::Authority::decode(unset.encode_to_vec().as_slice())
            .unwrap()
            .model_generation,
        None
    );
}

#[test]
fn registry_messages_carry_the_m35_handshake_fields() {
    let reply = pb::RegisterReply {
        accepted: false,
        reason_code: "stale_epoch".into(),
        journaled_worker_epoch: 9,
        ..Default::default()
    };
    assert_eq!(
        pb::RegisterReply::decode(reply.encode_to_vec().as_slice()).unwrap(),
        reply
    );

    let hb = pb::HeartbeatRequest {
        worker: None,
        sequence: 5,
        observed_coordinator_term: 2,
        inventory_sha256: vec![1; 32],
        model_generation: 4,
        admission_state: "fenced".into(),
        owner_only: true,
    };
    assert_eq!(
        pb::HeartbeatRequest::decode(hb.encode_to_vec().as_slice()).unwrap(),
        hb
    );

    // `reconciled_generation` is optional: generation 0 reconciled differs from "not reconciled".
    for g in [None, Some(0), Some(3)] {
        let r = pb::HeartbeatReply {
            accepted: true,
            coordinator_term: 2,
            lease_sequence: 1,
            admission_lease_ms: 4000,
            desired_state: "DRAINING".into(),
            reconciled_generation: g,
            reason_code: String::new(),
        };
        assert_eq!(
            pb::HeartbeatReply::decode(r.encode_to_vec().as_slice()).unwrap(),
            r
        );
    }
}

#[test]
fn list_operations_pages_carry_ownership_fields() {
    let status = pb::OperationStatus {
        operation_id: "cere-7-1-ab".into(),
        phase: "running".into(),
        terminal: false,
        cleanup_acknowledged: false,
        model_generation: 2,
        completion_ticket: "00".repeat(16),
        reason_code: String::new(),
        coordinator_term: 7,
        worker_epoch: 3,
        scope_id: "s".into(),
        request_id: "cere-7-1-ab".into(),
        attempt: 1,
        method: "SubmitWork".into(),
    };
    let reply = pb::ListOperationsReply {
        operations: vec![status],
        more: true,
    };
    assert_eq!(
        pb::ListOperationsReply::decode(reply.encode_to_vec().as_slice()).unwrap(),
        reply
    );
    let req = pb::ListOperationsRequest {
        authority: None,
        after_scope_id: "s".into(),
        after_operation_id: "x".into(),
        limit: 256,
    };
    assert_eq!(
        pb::ListOperationsRequest::decode(req.encode_to_vec().as_slice()).unwrap(),
        req
    );
}

#[test]
fn work_event_carries_output_bytes_verbatim_and_a_terminal_reason() {
    let chunk = br#"{"choices":[{"delta":{"content":"x"}}]}"#.to_vec();
    let e = pb::WorkEvent {
        sequence: 4,
        committed_token_offset: 4,
        bounded_output_event_json: chunk.clone(),
        terminal: true,
        reason_code: "model_generation_superseded".into(),
        ..Default::default()
    };
    let back = pb::WorkEvent::decode(e.encode_to_vec().as_slice()).unwrap();
    assert_eq!(back.bounded_output_event_json, chunk);
    assert_eq!(back.reason_code, "model_generation_superseded");
}

#[test]
fn every_service_generates_client_and_server_types() {
    // Type-level: the client constructors and server wrappers exist for every service.
    fn _client(ch: tonic::transport::Channel) {
        let _ = pb::worker_registry_service_client::WorkerRegistryServiceClient::new(ch.clone());
        let _ = pb::worker_service_client::WorkerServiceClient::new(ch.clone());
        let _ = pb::worker_inference_service_client::WorkerInferenceServiceClient::new(ch);
    }
    fn _server<
        R: pb::worker_registry_service_server::WorkerRegistryService,
        W: pb::worker_service_server::WorkerService,
        I: pb::worker_inference_service_server::WorkerInferenceService,
    >(
        r: R,
        w: W,
        i: I,
    ) {
        let _ = pb::worker_registry_service_server::WorkerRegistryServiceServer::new(r);
        let _ = pb::worker_service_server::WorkerServiceServer::new(w);
        let _ = pb::worker_inference_service_server::WorkerInferenceServiceServer::new(i);
    }
    assert_eq!(
        pb::worker_inference_service_server::SERVICE_NAME,
        "ai.cere.worker.v2.WorkerInferenceService"
    );
}
