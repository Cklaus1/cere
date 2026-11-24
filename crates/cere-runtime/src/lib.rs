//! `cered` runtime: a CERE front end for one pinned, backend-managed SGLang server (milestone M1).
//! Module list is owned by the build owner; each module belongs to one build track (build-loop design note).

// CFG track
pub mod config;
// CORE track
pub mod admission;
pub mod digest;
pub mod journal;
pub mod ledger;
pub mod session;
pub mod worker;
// BACK track
pub mod api_types;
pub mod constraint;
// DocInput (doc-input design note): document inputs (PDF text extraction, text documents, limits).
pub mod document;
// M2.13 (build-loop-m213 design note): image inputs (data-URL policy, header dimensions, resize).
pub mod image;
pub mod readiness;
pub mod responses;
pub mod sglang;
pub mod validate;
// SRV track
pub mod anthropic;
pub mod app;
pub mod auth;
pub mod pyjson;
pub mod reload;
pub mod resume;
pub mod server;
pub mod stream;
pub mod surface;
// RS track (M2.2a)
pub mod native;
// M3.5 federation (build-loop-m35 design note): the node agent (CFG, OPJ, LINK, WORK tracks), the worker RPC plumbing
// shared with the coordinator (FED, REG, RDY, CREM tracks) and the operator drain API (OPS track).
pub mod agent;
pub mod fed;
pub mod operator;
// m23 design note PI-01: the perf lever registry (single source of truth of every runtime-selected implementation).
pub mod levers;
// ab-split design note AB-01: the live A/B split of the runtime-layer levers.
pub mod ab;
// ops-hardening design note OH-05/OH-06: the Prometheus exposition and the access log.
pub mod metrics;
// Roadmap v0.7 A2: the client and workload labels of every chat request.
pub mod labels;
// net-teleport design note 4.1: trust tiers of installs, workers and person scopes.
pub mod tier;
// net-teleport design note 4.4 (S6): team pools (classes, worker owners and tiers, owner-only windows).
pub mod pool;
// net-teleport design note 4.4 (S7): the durable usage journal.
pub mod usage;
// server-info design note: `GET /get_server_info`, cered's own configuration in SGLang's shape.
pub mod server_info;
