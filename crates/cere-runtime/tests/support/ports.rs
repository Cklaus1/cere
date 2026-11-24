//! Loopback ports for listeners a test hands to a process it starts later (`cered`, `cere-worker`, an in-process
//! agent, a launched mock), O21 and O33. The window lies below Linux's ephemeral range (`net.ipv4.ip_local_port_range`,
//! 32768-60999 by default), so the kernel never hands one of these ports to a `bind(:0)` or an outgoing connection;
//! the only other takers are processes choosing ports here explicitly, and every such test process reserves first.
//! `scripts/live/_common.py` (`free_port`, `reserve_port`) is the Python side, with the same window and names.
//!
//! Plain std, no crate imports: `cere-mock-sglang`'s tests include this file by path.

use std::io::ErrorKind;
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixDatagram};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, Once, PoisonError};

const LOW: u32 = 20_000;
const SPAN: u32 = 12_000;
/// The abstract Unix socket name that reserves a port: `cere-test-port-<port>`.
pub const RESERVATION_PREFIX: &str = "cere-test-port-";

/// The reservations this process holds; never released before it exits.
static HELD: Mutex<Vec<UnixDatagram>> = Mutex::new(Vec::new());

/// A loopback port reserved for this process ([`reserve_port`]) and free when chosen: the process the caller starts
/// on it binds it later without a race, also across a restart on the same port. Candidates walk a per-process
/// sequence so concurrent test binaries start apart; panics when the whole window is taken.
pub fn free_port() -> u16 {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    static RANGE: Once = Once::new();
    RANGE.call_once(|| {
        let range = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range")
            .expect("read net.ipv4.ip_local_port_range");
        let ephemeral_low: u32 = range.split_whitespace().next().unwrap().parse().unwrap();
        assert!(
            LOW + SPAN <= ephemeral_low,
            "net.ipv4.ip_local_port_range starts at {ephemeral_low}, inside the test port window {LOW}..{}",
            LOW + SPAN
        );
    });
    let start = std::process::id().wrapping_mul(7_919) % SPAN;
    for _ in 0..SPAN {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let port = (LOW + (start + n.wrapping_mul(127)) % SPAN) as u16;
        if reserve_port(port) {
            return port;
        }
    }
    panic!("no free loopback port in {LOW}..{}", LOW + SPAN);
}

/// Takes `port` for this process's lifetime when no process holds it and no TCP socket uses it ([`port_in_use`]).
/// The reservation is the abstract Unix socket [`RESERVATION_PREFIX`]`<port>`: one per network namespace (as the
/// port), never a file, released by the kernel when this process exits (even on SIGKILL), close-on-exec so children
/// never hold it. It is taken before the check, so no other reserving process (a parallel test binary or a Python
/// harness) can take the port until this one exits.
pub fn reserve_port(port: u16) -> bool {
    let name = format!("{RESERVATION_PREFIX}{port}");
    let Ok(reservation) =
        SocketAddr::from_abstract_name(name.as_bytes()).and_then(|a| UnixDatagram::bind_addr(&a))
    else {
        return false;
    };
    if port_in_use(port) {
        return false;
    }
    HELD.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(reservation);
    true
}

/// Whether a TCP socket (IPv4 or IPv6, any state) has local port `port`, read from the kernel's socket tables. Not a
/// probe bind: a child that another test thread spawns while the probe socket is open holds a copy of it until the
/// child's `exec` ends, so the probe outlives its close and the caller's bind of the port fails `AddrInUse` (O21).
pub fn port_in_use(port: u16) -> bool {
    let suffix = format!(":{port:04X}");
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let rows = match std::fs::read_to_string(table) {
            Ok(rows) => rows,
            // A kernel without IPv6 has no tcp6 table.
            Err(e) if e.kind() == ErrorKind::NotFound && table.ends_with('6') => continue,
            Err(e) => panic!("read {table}: {e}"),
        };
        // After the header, `sl local_address rem_address st ...` per socket, an address as `<hex ip>:<hex port>`.
        let mut locals = rows
            .lines()
            .skip(1)
            .filter_map(|row| row.split_whitespace().nth(1));
        if locals.any(|local| local.ends_with(&suffix)) {
            return true;
        }
    }
    false
}
