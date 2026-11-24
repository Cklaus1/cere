//! Minimal in-memory transition oracle. Not a durable request ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Queued,
    Running,
    CancelPending,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionFence {
    pub worker_id: String,
    pub worker_epoch: u64,
    pub model_generation: u64,
    pub submission_id: u64,
}

#[derive(Debug)]
pub struct RequestLifecycle {
    pub state: RequestState,
    pub fence: Option<ExecutionFence>,
}

impl Default for RequestLifecycle {
    fn default() -> Self {
        Self {
            state: RequestState::Queued,
            fence: None,
        }
    }
}

impl RequestLifecycle {
    pub fn start(&mut self, fence: ExecutionFence) -> Result<(), &'static str> {
        if fence.worker_id.is_empty()
            || fence.worker_epoch == 0
            || fence.model_generation == 0
            || fence.submission_id == 0
            || self.state != RequestState::Queued
        {
            return Err("invalid start");
        }
        self.fence = Some(fence);
        self.state = RequestState::Running;
        Ok(())
    }

    pub fn cancel(&mut self) {
        match self.state {
            RequestState::Queued => self.state = RequestState::Cancelled,
            RequestState::Running => self.state = RequestState::CancelPending,
            _ => {}
        }
    }

    pub fn finish(
        &mut self,
        fence: &ExecutionFence,
        terminal: RequestState,
    ) -> Result<(), &'static str> {
        if self.fence.as_ref() != Some(fence) {
            return Err("stale epoch");
        }
        if !matches!(
            self.state,
            RequestState::Running | RequestState::CancelPending
        ) {
            return Err("not executing or already terminal");
        }
        if !matches!(
            terminal,
            RequestState::Completed | RequestState::Failed | RequestState::Cancelled
        ) {
            return Err("invalid terminal state");
        }
        if terminal == RequestState::Cancelled && self.state != RequestState::CancelPending {
            return Err("cancellation not requested");
        }
        self.state = terminal;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionFence, RequestLifecycle, RequestState};

    fn fence(epoch: u64) -> ExecutionFence {
        ExecutionFence {
            worker_id: "local-worker".into(),
            worker_epoch: epoch,
            model_generation: 1,
            submission_id: 1,
        }
    }

    #[test]
    fn cancel_waits_for_ack() {
        let mut r = RequestLifecycle::default();
        r.start(fence(2)).unwrap();
        r.cancel();
        assert_eq!(r.state, RequestState::CancelPending);
        assert!(r.finish(&fence(1), RequestState::Cancelled).is_err());
        r.finish(&fence(2), RequestState::Cancelled).unwrap();
        assert!(r.finish(&fence(2), RequestState::Completed).is_err());
    }

    #[test]
    fn queued_cancel_cannot_restart() {
        let mut r = RequestLifecycle::default();
        r.cancel();
        assert!(r.start(fence(1)).is_err());
    }
}
