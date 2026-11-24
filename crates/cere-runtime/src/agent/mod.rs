//! `cere-worker`, the node agent (m35 design note Q3): one worker over one loopback backend, served to the
//! coordinator over mTLS gRPC. Module list owned by the build owner (build-loop-m35 design note).

// CFG track
pub mod config;
// OPJ track
pub mod journal;
// LINK track
pub mod identity;
pub mod link;
pub mod node;
pub mod state;
// WORK track
pub mod ops;
pub mod service;
// SUP track
pub mod supervisor;

pub use node::{Agent, AgentError, AgentHandle};
