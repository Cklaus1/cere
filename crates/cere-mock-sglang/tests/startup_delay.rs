//! `--startup-delay-ms` (m36 design note P6, Q14): the mock binds only after the delay, so a supervising agent
//! can observe `loading`; a bad value is a usage error (exit 2).

use std::net::{Ipv4Addr, TcpStream};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[allow(dead_code)]
#[path = "../../cere-runtime/tests/support/ports.rs"]
mod ports;

use ports::free_port;

const BIN: &str = env!("CARGO_BIN_EXE_cere-mock-sglang");

fn accepts(port: u16) -> bool {
    TcpStream::connect((Ipv4Addr::LOCALHOST, port)).is_ok()
}

#[test]
fn binds_only_after_the_startup_delay() {
    let port = free_port();
    let started = Instant::now();
    let mut child = Command::new(BIN)
        .args(["--port", &port.to_string(), "--startup-delay-ms", "1500"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mock");
    sleep(Duration::from_millis(300));
    let early = accepts(port);
    let mut bound_at = None;
    while started.elapsed() < Duration::from_secs(10) {
        if accepts(port) {
            bound_at = Some(started.elapsed());
            break;
        }
        sleep(Duration::from_millis(20));
    }
    child.kill().expect("kill mock");
    child.wait().expect("reap mock");
    assert!(!early, "port accepted connections before the delay");
    let bound_at = bound_at.expect("mock never bound");
    assert!(
        bound_at >= Duration::from_millis(1500),
        "bound after {bound_at:?}"
    );
}

#[test]
fn bad_startup_delay_is_a_usage_error() {
    for value in ["-1", "600001", "x"] {
        let output = Command::new(BIN)
            .args(["--startup-delay-ms", value])
            .output()
            .expect("run mock");
        assert_eq!(output.status.code(), Some(2), "value {value:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("usage: cere-mock-sglang"), "{stderr}");
    }
}
