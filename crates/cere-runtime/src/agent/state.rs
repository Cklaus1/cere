//! The agent's durable counters (m35 design note Q4, Q5, Q6): `agent_state.json` in the `state_dir` holds the
//! worker epoch (+1 per start and per in-process incarnation jump), the model generation with the backend
//! `instance_id` behind it, and the highest coordinator term accepted from the bound coordinator principal.
//!
//! Every change is written before the agent acts on it: a temp file (0600) is written and fsynced, renamed over
//! the file, and the directory is fsynced, so a crash leaves either the old or the new state, never a mix. The
//! file holds counters and an identity digest only; never prompt or output text.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::journal::sync_dir;

/// The durable state file inside the agent `state_dir`.
pub const STATE_FILE: &str = "agent_state.json";
const STATE_TMP_FILE: &str = "agent_state.json.tmp";

/// `agent_state.json`. A missing file is the state before the first start (all zero, no instance).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentState {
    pub worker_epoch: u64,
    /// 0 until the first confirmed backend identity; then +1 per confirmed `instance_id` change.
    pub model_generation: u64,
    /// The confirmed `instance_id` behind `model_generation`.
    pub instance_id: Option<String>,
    /// Highest coordinator term accepted from the bound coordinator principal (0 = none yet).
    pub coordinator_term: u64,
}

impl AgentState {
    /// Reads the state of `state_dir`; a missing file is [`AgentState::default`], an unreadable or unparsable one
    /// an error (never silently reset: the epoch must not go backwards).
    pub fn load(state_dir: &Path) -> io::Result<AgentState> {
        let path = state_dir.join(STATE_FILE);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(AgentState::default()),
            Err(e) => return Err(e),
        };
        serde_json::from_slice(&bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })
    }

    /// Atomically replaces the state file (temp file, fsync, rename, directory fsync).
    pub fn store(&self, state_dir: &Path) -> io::Result<()> {
        let tmp = state_dir.join(STATE_TMP_FILE);
        let mut bytes = serde_json::to_vec(self).map_err(io::Error::other)?;
        bytes.push(b'\n');
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, state_dir.join(STATE_FILE))?;
        sync_dir(state_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cere-agent-state-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_is_default_and_store_round_trips() {
        let dir = temp_dir("roundtrip");
        assert_eq!(AgentState::load(&dir).unwrap(), AgentState::default());
        let state = AgentState {
            worker_epoch: 3,
            model_generation: 2,
            instance_id: Some("sglang-0011223344556677".into()),
            coordinator_term: 9,
        };
        state.store(&dir).unwrap();
        assert_eq!(AgentState::load(&dir).unwrap(), state);
        assert!(!dir.join(STATE_TMP_FILE).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_corrupt_file_is_an_error_not_a_reset() {
        let dir = temp_dir("corrupt");
        std::fs::write(dir.join(STATE_FILE), b"{\"worker_epoch\":").unwrap();
        assert_eq!(
            AgentState::load(&dir).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        std::fs::write(dir.join(STATE_FILE), b"{\"worker_epoch\":1,\"x\":2}").unwrap();
        assert!(AgentState::load(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
