//! NetTeleport S3 (net-teleport design note 4.2 L1, C7, D-T3, D-T5): warm restart of one native `cered` over the
//! CPU toy executor (prefix cache on, a snapshot arena).
//!
//! With `[state] park_on_stop`, a SIGTERM parks scope A's cached prompt sealed under `<state_dir>/parked/` (0700,
//! 0600, no plaintext token run on disk); the next start imports it before serving and deletes the file, so A's first
//! chat hits the warm prefix and answers as before the restart, while scope B's namespace stays cold. An image whose
//! library identity is not the loaded one is refused by name, deleted, and the turn replays cold with the same answer.
//! Without the key nothing is parked.
mod native_server;
mod native_support;
mod support;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use native_server::{Case, Native, SnapshotProfile, TOKEN};
use serde_json::{json, Value};
use support::{http, MODEL};

const SENTINEL: &str = "cere-native-sentinel-prompt";
const TOKEN_B: &str = "secret-b";
/// The prompt: 9,100 tokens (`[1]` + the message's bytes); a repeat hits floor64(n - 1) of it (M2.10 R3).
const PROMPT_TOKENS: usize = 9_100;
/// What a `cered` here must be able to `mlock` to park and import the toy image (a few hundred KiB, mostly the
/// toy's fixed state; a CI container allows 8 MiB).
const MEMLOCK_ROOM: usize = 1 << 20;
const CACHED: u64 = (PROMPT_TOKENS as u64 - 1) / 64 * 64;
const EXIT_BOUND: Duration = Duration::from_secs(30);

fn message() -> String {
    let mut s = SENTINEL.to_owned();
    while s.len() < PROMPT_TOKENS - 2 {
        s.push('x');
    }
    s.push('1');
    s
}

fn chat_body() -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": message()}],
        "max_tokens": 4,
        "temperature": 0,
    })
}

/// A unary chat as `token`'s scope: its choices and `cached_tokens`.
async fn chat(native: &Native, token: &str) -> (Value, u64) {
    let response = http()
        .post(native.cered.url("/v1/chat/completions"))
        .bearer_auth(token)
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let v: Value = response.json().await.unwrap();
    let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("{v}"));
    (v["choices"].clone(), cached)
}

/// The native config with `[state] park_on_stop` and a fresh port, written beside it as `name`.
fn park_config(native: &Native, name: &str) -> PathBuf {
    let text = std::fs::read_to_string(&native.config_path).unwrap();
    let mut config: toml::Table = toml::from_str(&text).unwrap();
    config["state"]
        .as_table_mut()
        .unwrap()
        .insert("park_on_stop".into(), true.into());
    config["server"]["bind"] = format!("127.0.0.1:{}", support::free_port()).into();
    let path = native.config_path.with_file_name(name);
    std::fs::write(&path, toml::to_string(&config).unwrap()).unwrap();
    path
}

/// SIGTERM and the graceful exit's stderr.
async fn stop(native: &mut Native) -> String {
    native.cered.signal("TERM");
    let (code, stderr) = native.cered.exit(EXIT_BOUND).await;
    assert_eq!(code, Some(0), "{stderr}");
    stderr
}

async fn start(native: &mut Native, config: &Path) {
    native
        .respawn(Path::new(env!("CARGO_BIN_EXE_cered")), config, "toy")
        .await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 200, "{body}");
}

fn parked_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    files
}

fn log_line<'a>(stderr: &'a str, prefix: &str) -> &'a str {
    stderr
        .lines()
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("no {prefix:?} line in {stderr}"))
}

/// `CEREPRK1`, the namespace, then the `CERESTB1` blob: its JSON header and the frames.
fn split(file: &[u8]) -> (Vec<u8>, Value, Vec<u8>) {
    assert_eq!(&file[..8], b"CEREPRK1");
    let blob = &file[24..];
    assert_eq!(&blob[..8], b"CERESTB1");
    let len = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize;
    (
        file[..24].to_vec(),
        serde_json::from_slice(&blob[12..12 + len]).unwrap(),
        blob[12 + len..].to_vec(),
    )
}

fn rebuild(prefix: &[u8], header: &Value, rest: &[u8]) -> Vec<u8> {
    let text = serde_json::to_vec(header).unwrap();
    let mut out = prefix.to_vec();
    out.extend_from_slice(b"CERESTB1");
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(&text);
    out.extend_from_slice(rest);
    out
}

fn park_case() -> Case {
    Case {
        prefix_state_slots: Some(16),
        snapshot: Some(SnapshotProfile {
            preempt: false,
            preempt_min_tokens: 512,
            inflight: 2,
            preempt_every: 0,
            rows: 1,
            parked: 1,
            reserve: 0,
        }),
        ..Case::matching()
    }
}

/// A launcher that starts `cered` unable to lock any memory: `RLIMIT_MEMLOCK` 0 and, under root, without
/// `CAP_IPC_LOCK` (which would bypass the limit; dropped from the bounding set with util-linux `setpriv`).
fn memlock_starved_cered(native: &Native) -> PathBuf {
    let path = native.config_path.with_file_name("cered-no-memlock.sh");
    let cered = env!("CARGO_BIN_EXE_cered");
    // SAFETY: geteuid has no preconditions.
    let exec = if unsafe { libc::geteuid() } == 0 {
        format!("exec setpriv --bounding-set=-ipc_lock --inh-caps=-ipc_lock '{cered}' \"$@\"")
    } else {
        format!("exec '{cered}' \"$@\"")
    };
    std::fs::write(
        &path,
        format!("#!/bin/sh\nulimit -l 0 || exit 97\n{exec}\n"),
    )
    .unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}

/// The `cered` children inherit this process's `RLIMIT_MEMLOCK` and capabilities: below [`MEMLOCK_ROOM`] every park
/// refuses `state_memlock` by design (a refusal, never a pageable copy), so say so instead of failing obscurely.
fn require_memlock_room() {
    let buf = vec![0u8; MEMLOCK_ROOM];
    // SAFETY: the range is the live allocation `buf`; it is unlocked before it is freed.
    let locked = unsafe { libc::mlock(buf.as_ptr().cast(), buf.len()) } == 0;
    assert!(
        locked,
        "warm_restart needs RLIMIT_MEMLOCK >= {MEMLOCK_ROOM} bytes or CAP_IPC_LOCK: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: as above.
    unsafe { libc::munlock(buf.as_ptr().cast(), buf.len()) };
}

#[tokio::test]
async fn warm_restart_parks_imports_and_refuses_by_identity() {
    require_memlock_room();
    let mut native = Native::ready(park_case()).await;
    let parked = native.state.path().join("parked");

    // Off by default: a stop parks nothing and creates no directory.
    let (cold, cached) = chat(&native, TOKEN).await;
    assert_eq!(cached, 0);
    let stderr = stop(&mut native).await;
    assert!(!stderr.contains("cered state park"), "{stderr}");
    assert!(!parked.exists());

    // On: the directory is created private at start; A's prompt is cached, then parked at the stop.
    let config = park_config(&native, "park.toml");
    start(&mut native, &config).await;
    let mode = std::fs::metadata(&parked).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o700);
    assert_eq!(chat(&native, TOKEN).await, (cold.clone(), 0));
    assert_eq!(chat(&native, TOKEN).await, (cold.clone(), CACHED));
    let stderr = stop(&mut native).await;
    let line = log_line(&stderr, "cered state park:");
    assert!(line.contains("recorded=1 parked=1 "), "{line}");
    assert!(line.contains("refused=0"), "{line}");
    let files = parked_files(&parked);
    assert_eq!(files.len(), 1, "{files:?}");
    let mode = std::fs::metadata(&files[0]).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
    // 4.1: the prompt's ids never reach the disk in the clear (no 64-id run of them, little-endian u32).
    let bytes = std::fs::read(&files[0]).unwrap();
    let ids: Vec<u8> = message()
        .bytes()
        .skip(1_000)
        .take(64)
        .flat_map(|b| u32::from(b).to_le_bytes())
        .collect();
    assert!(!bytes.windows(ids.len()).any(|w| w == ids.as_slice()));
    assert!(!bytes
        .windows(SENTINEL.len())
        .any(|w| w == SENTINEL.as_bytes()));

    // The next start imports the image before serving and deletes it: A's first chat is warm and answers the same;
    // B's namespace stays cold (the image went back into A's own).
    start(&mut native, &config).await;
    let line = log_line(native.cered.stderr_so_far().as_str(), "cered state unpark:").to_owned();
    assert!(
        line.starts_with("cered state unpark: files=1 imported=1 "),
        "{line}"
    );
    assert!(line.contains("refused=0"), "{line}");
    assert!(parked_files(&parked).is_empty());
    assert_eq!(chat(&native, TOKEN).await, (cold.clone(), CACHED));
    assert_eq!(chat(&native, TOKEN_B).await, (cold.clone(), 0));

    // A library change refuses the parked image by name, deletes it, and the turn replays cold with the same
    // answer. The edit is the header's identity (the header is checked before any key use).
    let stderr = stop(&mut native).await;
    assert!(
        log_line(&stderr, "cered state park:").contains("parked=2 "),
        "{stderr}"
    );
    for file in parked_files(&parked) {
        let (prefix, mut header, rest) = split(&std::fs::read(&file).unwrap());
        header["library_sha256"] = json!("00".repeat(32));
        std::fs::write(&file, rebuild(&prefix, &header, &rest)).unwrap();
    }
    std::fs::write(parked.join(".0000000000000001-000.stb.tmp"), b"torn").unwrap();
    start(&mut native, &config).await;
    let line = log_line(native.cered.stderr_so_far().as_str(), "cered state unpark:").to_owned();
    assert!(
        line.starts_with("cered state unpark: files=2 imported=0 "),
        "{line}"
    );
    assert!(
        line.contains("refused=state_library_mismatch:2 stale=1"),
        "{line}"
    );
    assert!(parked_files(&parked).is_empty());
    assert_eq!(chat(&native, TOKEN).await, (cold.clone(), 0));
    assert_eq!(chat(&native, TOKEN).await, (cold, CACHED));
    stop(&mut native).await;
}

/// A start that cannot lock the image buffer refuses the parked image as `state_memlock`, deletes it and serves:
/// the turn replays cold with the same answer. A stop that cannot lock an export buffer parks nothing and still
/// exits cleanly.
#[tokio::test]
async fn a_start_short_on_memlock_refuses_the_parked_image_and_replays() {
    require_memlock_room();
    let mut native = Native::ready(park_case()).await;
    let parked = native.state.path().join("parked");
    let config = park_config(&native, "park.toml");
    stop(&mut native).await;
    start(&mut native, &config).await;
    let (cold, cached) = chat(&native, TOKEN).await;
    assert_eq!(cached, 0);
    assert_eq!(chat(&native, TOKEN).await, (cold.clone(), CACHED));
    let stderr = stop(&mut native).await;
    assert!(
        log_line(&stderr, "cered state park:").contains("parked=1 "),
        "{stderr}"
    );
    assert_eq!(parked_files(&parked).len(), 1);

    let starved = memlock_starved_cered(&native);
    native.respawn(&starved, &config, "toy").await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 200, "{body}");
    let line = log_line(native.cered.stderr_so_far().as_str(), "cered state unpark:").to_owned();
    assert!(
        line.starts_with("cered state unpark: files=1 imported=0 "),
        "{line}"
    );
    assert!(line.contains("refused=state_memlock:1 "), "{line}");
    assert!(parked_files(&parked).is_empty());
    assert_eq!(chat(&native, TOKEN).await, (cold.clone(), 0));
    assert_eq!(chat(&native, TOKEN).await, (cold, CACHED));

    let stderr = stop(&mut native).await;
    let line = log_line(&stderr, "cered state park:");
    assert!(line.contains("parked=0 "), "{line}");
    assert!(line.contains("refused=state_memlock:1 "), "{line}");
    assert!(parked_files(&parked).is_empty());
}
