//! Worker RPC plumbing shared by `cered` (coordinator) and `cere-worker` (agent) (m35 design note Q1, Q2,
//! Q12, Q14). Module list owned by the build owner (build-loop-m35 design note).

// FED track
pub mod quota;
pub mod tls;
pub mod wire;

// REG track
pub mod registry;
// RDY track
pub mod client;
// CREM tracks (DISPATCH, ORPHAN)
pub mod remote;
// ORPHAN-B track
pub mod orphan;
