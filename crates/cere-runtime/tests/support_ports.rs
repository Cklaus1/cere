//! `support::free_port` (O21, O33): a port handed to a test stays that process's until it exits, so the `cered`,
//! agent or mock started on it later binds it without racing another test process asking for a port. The other
//! process is this test binary re-executed (`CHILD_PORT` set), so the check crosses a process boundary as the race
//! does. `tests/test_bench_shared_helpers.py` (`FreePort`) checks the Python side and the shared names.

#[allow(dead_code)]
#[path = "support/ports.rs"]
mod ports;

use std::process::Command;

const CHILD_PORT: &str = "CERE_PORTS_CHILD_PORT";

/// In the re-executed child: prints whether this process could reserve `CHILD_PORT`. A no-op otherwise.
#[test]
fn child_reserves() {
    if let Ok(port) = std::env::var(CHILD_PORT) {
        println!("reserved={}", ports::reserve_port(port.parse().unwrap()));
    }
}

/// Runs `child_reserves` in a new process for `port`; true when that process could reserve it.
fn another_process_reserves(port: u16) -> bool {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_reserves",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_PORT, port.to_string())
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // libtest prints the line after the test's name: `test child_reserves ... reserved=<bool>`.
    match stdout
        .split("reserved=")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
    {
        Some("true") => true,
        Some("false") => false,
        _ => panic!("no answer from the child: {stdout}"),
    }
}

#[test]
fn a_port_handed_out_is_refused_to_another_process() {
    let port = ports::free_port();
    // Nothing listens on it yet (the window in which a parallel test binary used to probe it free).
    assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
    assert!(!another_process_reserves(port));
    assert!(!ports::reserve_port(port));
    assert_ne!(ports::free_port(), port);
}

#[test]
fn a_reservation_ends_with_its_process() {
    let port = ports::free_port();
    let free = (port + 1..)
        .find(|&p| another_process_reserves(p))
        .expect("some port above the reserved one is free");
    // The child that reserved `free` has exited, so its reservation is gone.
    assert!(ports::reserve_port(free));
}

/// The check `reserve_port` makes instead of a probe bind (O21) sees a listening socket, IPv4 and IPv6.
#[test]
fn a_port_a_socket_listens_on_is_in_use() {
    let v4 = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    assert!(ports::port_in_use(v4.local_addr().unwrap().port()));
    // The CI container may have no IPv6 loopback.
    if let Ok(v6) = std::net::TcpListener::bind("[::1]:0") {
        assert!(ports::port_in_use(v6.local_addr().unwrap().port()));
    }
}
