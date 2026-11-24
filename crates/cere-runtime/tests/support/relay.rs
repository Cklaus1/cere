//! Loopback TCP relay between the real `cered` and `cere-worker` binaries that a test can black-hole: the CI form
//! of a network partition (m35 design note P9). Track MATRIX (build-loop-m35 design note).
//!
//! A [`Relay`] listens on a loopback port and forwards every accepted connection to its target. [`Relay::block`]
//! drops every live connection and then refuses ([`Block::Refuse`]: accept and close at once) or holds
//! ([`Block::Hold`]: accept and never answer, a silent peer) every new one; [`Relay::heal`] drops the held ones and
//! forwards again. TLS runs end to end through the relay, so both peers still check each other's principal.

use std::net::SocketAddr;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// What a blocked relay does with a new connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    /// Accept and close at once: the dialler sees the connection end.
    Refuse,
    /// Accept and never forward or close: the dialler's own deadlines and keepalives must notice.
    Hold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Forward,
    Blocked(Block),
}

/// A running relay; dropping it stops the listener and drops every connection.
pub struct Relay {
    addr: SocketAddr,
    /// (change counter, mode): every change ends the connections accepted before it.
    state: watch::Sender<(u64, Mode)>,
    accept: JoinHandle<()>,
}

impl Relay {
    /// Listens on a loopback port from [`super::free_port`] (reserved for this process) and forwards to `target`.
    pub async fn start(target: SocketAddr) -> Relay {
        let listener = TcpListener::bind(("127.0.0.1", super::free_port()))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let (state, _) = watch::channel((0, Mode::Forward));
        let states = state.clone();
        let accept = tokio::spawn(async move {
            loop {
                let Ok((inbound, _)) = listener.accept().await else {
                    continue;
                };
                let mut changes = states.subscribe();
                let (at, mode) = *changes.borrow_and_update();
                let changed = async move {
                    let _ = changes.wait_for(|(now, _)| *now != at).await;
                };
                match mode {
                    Mode::Blocked(Block::Refuse) => drop(inbound),
                    Mode::Blocked(Block::Hold) => {
                        tokio::spawn(async move {
                            changed.await;
                            drop(inbound);
                        });
                    }
                    Mode::Forward => {
                        tokio::spawn(async move {
                            let mut inbound = inbound;
                            let Ok(mut outbound) = TcpStream::connect(target).await else {
                                return;
                            };
                            tokio::select! {
                                _ = tokio::io::copy_bidirectional(&mut inbound, &mut outbound) => {}
                                () = changed => {}
                            }
                        });
                    }
                }
            }
        });
        Relay {
            addr,
            state,
            accept,
        }
    }

    /// The address to dial instead of the target.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Drops every live connection; new ones are refused or held per `block`.
    pub fn block(&self, block: Block) {
        self.set(Mode::Blocked(block));
    }

    /// Drops held connections and forwards new ones again.
    pub fn heal(&self) {
        self.set(Mode::Forward);
    }

    fn set(&self, mode: Mode) {
        self.state.send_modify(|(at, current)| {
            *at += 1;
            *current = mode;
        });
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.accept.abort();
        // Ends every connection task: their change watch sees one more change.
        self.set(Mode::Blocked(Block::Refuse));
    }
}
