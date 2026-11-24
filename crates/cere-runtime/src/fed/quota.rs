//! Per-principal concurrency quota for RPC handlers (m35 design note Q14): at most `limit` calls of one
//! authenticated principal in flight; the next is `RESOURCE_EXHAUSTED quota_exhausted`. A slot is held by a
//! `QuotaGuard` (movable into a response stream) and released when the guard drops.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use crate::fed::wire;

#[derive(Debug, Clone)]
pub struct PrincipalQuota {
    limit: usize,
    held: Arc<Mutex<HashMap<String, usize>>>,
}

/// One held slot; dropping it frees the slot.
#[derive(Debug)]
pub struct QuotaGuard {
    principal: String,
    held: Arc<Mutex<HashMap<String, usize>>>,
}

impl PrincipalQuota {
    pub fn new(limit: usize) -> PrincipalQuota {
        PrincipalQuota {
            limit,
            held: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn try_acquire(&self, principal: &str) -> Result<QuotaGuard, tonic::Status> {
        // The map is only counters; a panic elsewhere cannot leave it inconsistent, so poisoning is ignored.
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        if held.get(principal).copied().unwrap_or(0) >= self.limit {
            return Err(wire::status(
                tonic::Code::ResourceExhausted,
                wire::QUOTA_EXHAUSTED,
            ));
        }
        *held.entry(principal.to_owned()).or_insert(0) += 1;
        Ok(QuotaGuard {
            principal: principal.to_owned(),
            held: Arc::clone(&self.held),
        })
    }
}

impl Drop for QuotaGuard {
    fn drop(&mut self) {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(count) = held.get_mut(&self.principal) {
            *count -= 1;
            if *count == 0 {
                held.remove(&self.principal);
            }
        }
    }
}
