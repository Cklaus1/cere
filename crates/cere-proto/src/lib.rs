//! Generated bindings for `proto/cere_worker.proto` (package `ai.cere.worker.v2`, protocol revision 2).

/// Protobuf package of every generated item.
pub const PACKAGE: &str = "ai.cere.worker.v2";

/// The only protocol revision `Register` accepts (docs/WORKER_PROTOCOL.md; m35 design note Q8).
pub const PROTOCOL_REVISION: u32 = 2;

/// Encoded `FileDescriptorSet` of the compiled proto.
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cere_worker_descriptor.bin"));

pub mod worker {
    pub mod v2 {
        tonic::include_proto!("ai.cere.worker.v2");
    }
}
