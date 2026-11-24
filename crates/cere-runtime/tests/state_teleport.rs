//! NetTeleport S1 (net-teleport design note 4.2 L2, 4.3, D-T1..D-T7): `POST /v1/cere/state/export` and
//! `POST /v1/cere/state/import` on one native `cered` over the CPU toy executor (prefix cache on, a snapshot arena).
//!
//! Scope A's prompt is cached by a chat; A exports its deepest cached prefix sealed under a client key; scope B imports
//! the blob into its own namespace (D-T2) and its first chat of the same prompt hits the imported prefix and answers
//! as A's did. Every refusal of the matrix (bar c) is a named code and imports nothing (the prefix-cache counters do
//! not move); an export while a ticket is in flight waits `state.busy_wait_ms` and refuses `state_busy` without
//! touching the user's turn (D-T1).
mod native_server;
mod native_support;
mod support;

use std::time::{Duration, Instant};

use native_server::{Case, Native, SnapshotProfile, OPERATOR_TOKEN, TOKEN, WORKER};
use native_support::Tmp;
use serde_json::{json, Value};
use support::{http, MODEL};

const SENTINEL: &str = "cere-native-sentinel-prompt";
const TOKEN_B: &str = "secret-b";
const KEY: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const OTHER_KEY: &str = "2222222222222222222222222222222222222222222222222222222222222222";
/// The prompt: 9,100 tokens (`[1]` + the message's bytes); a repeat hits floor64(n - 1) of it (M2.10 R3).
const PROMPT_TOKENS: usize = 9_100;
const CACHED: u64 = (PROMPT_TOKENS as u64 - 1) / 64 * 64;

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

/// The stub sidecar's ids of [`chat_body`]: `[1]` then the message's bytes.
fn prompt_ids() -> Vec<u32> {
    std::iter::once(1)
        .chain(message().bytes().map(u32::from))
        .collect()
}

/// A unary chat as `token`'s scope: the response and its `cached_tokens`.
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
    (v, cached)
}

struct Exported {
    status: u16,
    tokens: Option<u64>,
    image_bytes: Option<u64>,
    body: Vec<u8>,
}

impl Exported {
    fn code(&self) -> Value {
        let v: Value = serde_json::from_slice(&self.body).unwrap_or(Value::Null);
        v["error"]["code"].clone()
    }
}

async fn export(native: &Native, token: &str, key: Option<&str>, body: &Value) -> Exported {
    let mut request = http()
        .post(native.cered.url("/v1/cere/state/export"))
        .bearer_auth(token)
        .json(body);
    if let Some(key) = key {
        request = request.header("X-Cere-State-Key", key);
    }
    let response = request.send().await.unwrap();
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
    };
    let tokens = header("x-cere-state-tokens");
    let image_bytes = header("x-cere-state-image-bytes");
    let status = response.status().as_u16();
    let body = response.bytes().await.unwrap().to_vec();
    Exported {
        status,
        tokens,
        image_bytes,
        body,
    }
}

async fn import(native: &Native, token: &str, key: &str, blob: Vec<u8>) -> (u16, Value) {
    let response = http()
        .post(
            native
                .cered
                .url(&format!("/v1/cere/state/import?model={MODEL}")),
        )
        .bearer_auth(token)
        .header("X-Cere-State-Key", key)
        .header("Content-Type", "application/vnd.cere.state-blob")
        .body(blob)
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// The worker's prefix-cache counters (`GET /v1/cere/prefix_cache`, operator).
async fn counters(native: &Native) -> Value {
    let response = http()
        .get(native.cered.url("/v1/cere/prefix_cache"))
        .bearer_auth(OPERATOR_TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let v: Value = response.json().await.unwrap();
    v["workers"][WORKER]["prefix_cache"].clone()
}

/// Rewrites the config's `[state]` `key` (`None` removes it) and reloads (both keys are live).
async fn reload_state(native: &Native, key: &str, value: Option<i64>) {
    let text = std::fs::read_to_string(&native.config_path).unwrap();
    let mut config: toml::Table = toml::from_str(&text).unwrap();
    let state = config["state"].as_table_mut().unwrap();
    match value {
        Some(v) => state.insert(key.into(), v.into()),
        None => state.remove(key),
    };
    std::fs::write(&native.config_path, toml::to_string(&config).unwrap()).unwrap();
    let response = http()
        .post(native.cered.url("/v1/cere/config/reload"))
        .bearer_auth(OPERATOR_TOKEN)
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    let body: Value = response.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
}

/// The blob's JSON header and the bytes after it.
fn split(blob: &[u8]) -> (Value, Vec<u8>) {
    assert_eq!(&blob[..8], b"CERESTB1");
    let len = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize;
    (
        serde_json::from_slice(&blob[12..12 + len]).unwrap(),
        blob[12 + len..].to_vec(),
    )
}

/// A blob with `header` re-serialized (its length re-stated) before `rest`.
fn rebuild(header: &Value, rest: &[u8]) -> Vec<u8> {
    let text = serde_json::to_vec(header).unwrap();
    let mut out = b"CERESTB1".to_vec();
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(&text);
    out.extend_from_slice(rest);
    out
}

fn edited(blob: &[u8], field: &str, value: Value) -> Vec<u8> {
    let (mut header, rest) = split(blob);
    header[field] = value;
    rebuild(&header, &rest)
}

/// The `memlock` refusal needs `cered`'s `mlock` to honour RLIMIT_MEMLOCK. As uid 0 this drops CAP_IPC_LOCK from
/// the test process's bounding set, so the `cered` it spawns cannot hold it (without CAP_SETPCAP the call fails, and
/// such a process has no CAP_IPC_LOCK to pass on). The soft limit is lowered to at most 16 MiB: the toy image fits
/// (the CI container's limit is 8 MiB), a 1 GiB image does not. Below 1 MiB the toy's own image (about 317 KB) cannot
/// be locked either, so every export answers `state_memlock`: the test names that limit and fails.
fn bound_locked_memory() {
    const CAP_IPC_LOCK: libc::c_ulong = 14;
    // SAFETY: prctl, getrlimit and setrlimit with a valid rlimit pointer; they touch no Rust memory else.
    unsafe {
        libc::prctl(libc::PR_CAPBSET_DROP, CAP_IPC_LOCK, 0, 0, 0);
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit), 0);
        limit.rlim_cur = limit.rlim_cur.min(16 << 20);
        assert_eq!(libc::setrlimit(libc::RLIMIT_MEMLOCK, &limit), 0);
        assert!(
            limit.rlim_cur >= 1 << 20,
            "RLIMIT_MEMLOCK soft limit {} B: the toy's ~317 KB state image cannot be locked (needs >= 1 MiB)",
            limit.rlim_cur
        );
    }
}

#[tokio::test]
async fn state_teleport_round_trip_refusals_and_busy() {
    bound_locked_memory();
    let t = Tmp::new("state-teleport");
    let gate = t.join("hold-gate");
    std::fs::write(&gate, b"").unwrap();
    let native = Native::ready(Case {
        // The first generation holds after its first token until the gate file is gone (the busy case).
        device_id: format!(
            "toy,hold_after=1,hold_ms=120000,hold_count=1,hold_gate={}",
            gate.display()
        ),
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
    })
    .await;
    reload_state(&native, "busy_wait_ms", Some(200)).await;
    let by_tokens = json!({"model": MODEL, "destination_tier": "own", "tokens": prompt_ids()});

    // D-T1: while A's held generation is in flight, an export waits busy_wait_ms and refuses by name; the
    // generation then completes as if no export had been asked.
    let held = {
        let url = native.cered.url("/v1/chat/completions");
        tokio::spawn(async move {
            http()
                .post(url)
                .bearer_auth(TOKEN)
                .json(&chat_body())
                .send()
                .await
                .unwrap()
        })
    };
    let until = Instant::now() + Duration::from_secs(60);
    loop {
        let got = export(&native, TOKEN, Some(KEY), &by_tokens).await;
        if got.status == 409 {
            assert_eq!(got.code(), "state_busy");
            break;
        }
        // Before the generation's ticket is in flight nothing is cached yet.
        assert_eq!((got.status, got.code()), (404, json!("state_no_prefix")));
        assert!(
            Instant::now() < until,
            "the held generation never went in flight"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    std::fs::remove_file(&gate).unwrap();
    let source = held.await.unwrap();
    assert_eq!(source.status(), 200);
    let source: Value = source.json().await.unwrap();
    assert_eq!(source["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    reload_state(&native, "busy_wait_ms", None).await;

    // Export refusals: key, tier policy at the source (A's scope allows only own), no prefix, body shape.
    let by_messages = {
        let mut body = chat_body();
        body["destination_tier"] = json!("own");
        body
    };
    let got = export(&native, TOKEN, None, &by_messages).await;
    assert_eq!((got.status, got.code()), (400, json!("state_key_invalid")));
    let got = export(&native, TOKEN, Some("abc"), &by_messages).await;
    assert_eq!((got.status, got.code()), (400, json!("state_key_invalid")));
    for tier in ["team", "provider"] {
        let mut body = by_messages.clone();
        body["destination_tier"] = json!(tier);
        let got = export(&native, TOKEN, Some(KEY), &body).await;
        assert_eq!(
            (got.status, got.code()),
            (403, json!("state_tier_not_allowed"))
        );
    }
    let got = export(
        &native,
        TOKEN,
        Some(KEY),
        &json!({"model": MODEL, "destination_tier": "own", "tokens": [1, 2, 3]}),
    )
    .await;
    assert_eq!((got.status, got.code()), (404, json!("state_no_prefix")));
    // B's namespace holds nothing: B cannot export A's prefix (D-T2).
    let got = export(&native, TOKEN_B, Some(KEY), &by_messages).await;
    assert_eq!((got.status, got.code()), (404, json!("state_no_prefix")));
    let mut both = by_messages.clone();
    both["tokens"] = json!([1]);
    assert_eq!(export(&native, TOKEN, Some(KEY), &both).await.status, 400);
    let got = export(
        &native,
        TOKEN,
        Some(KEY),
        &json!({"model": MODEL, "tokens": [1]}),
    )
    .await;
    assert_eq!(got.status, 400);
    // Oversize at the source: the toy's image (about 317 KB) exceeds a 64 KiB cap (live key).
    reload_state(&native, "max_blob_bytes", Some(1 << 16)).await;
    let got = export(&native, TOKEN, Some(KEY), &by_messages).await;
    assert_eq!((got.status, got.code()), (413, json!("state_oversize")));
    reload_state(&native, "max_blob_bytes", None).await;

    // The export: the deepest cached prefix, by the rendered messages and by the token ids alike.
    let blob = export(&native, TOKEN, Some(KEY), &by_messages).await;
    assert_eq!(
        blob.status,
        200,
        "{:?}",
        String::from_utf8_lossy(&blob.body)
    );
    assert_eq!(blob.tokens, Some(CACHED));
    let image_bytes = blob.image_bytes.unwrap();
    let again = export(&native, TOKEN, Some(KEY), &by_tokens).await;
    assert_eq!((again.status, again.tokens), (200, Some(CACHED)));
    let (header, rest) = split(&blob.body);
    assert_eq!(header["tokens"], CACHED);
    assert_eq!(header["image_bytes"], image_bytes);
    assert_eq!(header["source_tier"], "own");
    assert_eq!(header["allowed_tiers"], json!(["own"]));
    // After the header: each 4 MiB frame's ciphertext and its 16-byte tag.
    assert_eq!(
        rest.len() as u64,
        image_bytes + 16 * image_bytes.div_ceil(4 << 20)
    );
    // Sealed: the image's `CERESNP1` magic appears nowhere in the blob.
    assert!(!blob.body.windows(8).any(|w| w == b"CERESNP1"));

    // Import refusals (bar c): each named, each imports nothing.
    let before = counters(&native).await;
    let mut flipped = blob.body.clone();
    let at = flipped.len() - 100;
    flipped[at] ^= 1;
    let mut truncated = blob.body.clone();
    truncated.truncate(truncated.len() - 1);
    let mut extended = blob.body.clone();
    extended.push(0);
    let cases: Vec<(&str, &str, Vec<u8>, u16, &str)> = vec![
        (
            "wrong key",
            OTHER_KEY,
            blob.body.clone(),
            400,
            "state_wrong_key",
        ),
        ("tampered ciphertext", KEY, flipped, 400, "state_tampered"),
        ("truncated", KEY, truncated, 400, "state_tampered"),
        ("extended", KEY, extended, 400, "state_tampered"),
        (
            "edited AAD",
            KEY,
            edited(&blob.body, "created_unix_ms", json!(1)),
            400,
            "state_tampered",
        ),
        (
            "library",
            KEY,
            edited(&blob.body, "library_sha256", json!("ab".repeat(32))),
            409,
            "state_library_mismatch",
        ),
        (
            "layout",
            KEY,
            edited(&blob.body, "layout_id", json!("ab".repeat(32))),
            409,
            "state_layout_mismatch",
        ),
        (
            "compat id",
            KEY,
            edited(&blob.body, "state_compat_id", json!("ab".repeat(32))),
            409,
            "state_compat_mismatch",
        ),
        (
            "disallowed tier",
            KEY,
            edited(&blob.body, "allowed_tiers", json!(["team", "provider"])),
            403,
            "state_tier_not_allowed",
        ),
        (
            "oversize",
            KEY,
            edited(&blob.body, "image_bytes", json!(5u64 << 30)),
            413,
            "state_oversize",
        ),
        (
            // Within the 4 GiB cap, past the memlock limit: the image buffer cannot be locked (R-T8).
            "memlock",
            KEY,
            edited(&blob.body, "image_bytes", json!(1u64 << 30)),
            503,
            "state_memlock",
        ),
        (
            "not a blob",
            KEY,
            b"CERESNP1 hello".to_vec(),
            400,
            "state_malformed",
        ),
    ];
    for (name, key, body, status, code) in cases {
        let (got, v) = import(&native, TOKEN_B, key, body).await;
        assert_eq!(
            (got, v["error"]["code"].as_str()),
            (status, Some(code)),
            "{name}: {v}"
        );
    }
    let (got, v) = import(&native, TOKEN_B, "zz", blob.body.clone()).await;
    assert_eq!(
        (got, v["error"]["code"].as_str()),
        (400, Some("state_key_invalid")),
        "{v}"
    );
    // The real blob past a 64 KiB cap at the destination.
    reload_state(&native, "max_blob_bytes", Some(1 << 16)).await;
    let (got, v) = import(&native, TOKEN_B, KEY, blob.body.clone()).await;
    assert_eq!(
        (got, v["error"]["code"].as_str()),
        (413, Some("state_oversize")),
        "{v}"
    );
    reload_state(&native, "max_blob_bytes", None).await;
    assert_eq!(
        counters(&native).await,
        before,
        "a refused import stored nothing"
    );

    // The import: into B's own namespace; B's first chat of the prompt hits it and answers as A's did.
    let (got, v) = import(&native, TOKEN_B, KEY, blob.body.clone()).await;
    assert_eq!(got, 200, "{v}");
    assert_eq!(v["imported_tokens"], CACHED);
    assert_eq!(v["image_bytes"], image_bytes);
    assert_eq!(v["tier"], "own");
    let (teleported, cached) = chat(&native, TOKEN_B).await;
    assert_eq!(cached, CACHED);
    assert_eq!(teleported["choices"], source["choices"]);
    assert_eq!(
        teleported["usage"]["completion_tokens"],
        source["usage"]["completion_tokens"]
    );
}
