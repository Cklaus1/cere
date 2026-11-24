//! R5 (spec m22): the native request path. The tokenizer boundary is checked against B4's synthetic fixtures
//! (`tests/fixtures/native/{detok_cases,reasoning_cases}.json`, recorded from the image's `tokenizers` and SGLang's
//! `Qwen3Detector`), and the sidecar client speaks the Q9 frame protocol (a stub child here: the tests that
//! spawn processes live outside the lib test binary, whose journal tests re-take `flock`s a forked child can hold).
//!
//! The HTTP tests run the real `cered` with a `native_cuda` config over the CPU toy executor, the fixture artifact
//! and the stub sidecar ([`native_server`]). The stub encodes a user message as `[1]` plus its UTF-8 bytes, and
//! the toy's next token depends only on the last token and the position, so a prompt is chosen by its length and
//! last character; the expected completion is computed here with the toy rule and the artifact's tokenizer. A toy
//! with `hold_after=N` stops at its N-th generated token until cancelled: the deterministic mid-stream point.
mod image_support;
mod native_server;
mod native_support;
mod support;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cere_mock_sglang::{MockConfig, ServerInfo};
use cere_runtime::native::detok::Vocab;
use cere_runtime::native::reasoning::{split, Reasoning, Split, Stream};
use cere_runtime::native::sidecar::{Sidecar, SidecarError, SidecarHealth, SidecarStats};
use cere_runtime::validate::{ChatMessage, Role};
use image_support::png_url;
use native_server::{
    shipped_lock, Case, Images, Native, SnapshotProfile, OPERATOR_TOKEN, TOKEN, WORKER,
};
use native_support::{root, toy_tokens, Tmp, FIXTURE_VOCAB};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use support::{chat_body, check_stream, http, Event, Harness, SseReader, MODEL};

fn fixture(name: &str) -> Value {
    let path = root().join("tests/fixtures/native").join(name);
    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap()
}

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not a string"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .concat()
}

#[test]
fn detok_cases_all_pass() {
    let cases = fixture("detok_cases.json");
    let tokenizer = std::fs::read(
        root()
            .join("tests/fixtures/native")
            .join(text(&cases, "tokenizer")),
    )
    .unwrap();
    assert_eq!(sha256_hex(&tokenizer), text(&cases, "tokenizer_sha256"));
    assert_eq!(cases["skip_special_tokens"], true);
    let vocab = Vocab::from_json(&tokenizer).unwrap();
    let all = cases["cases"].as_array().unwrap();
    assert!(all.len() >= 100, "{} cases", all.len());
    for case in all {
        let name = text(case, "name");
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
            .collect();
        let whole = vocab.decode(&ids);
        assert_eq!(whole, text(case, "text"), "case {name}");
        assert_eq!(
            sha256_hex(whole.as_bytes()),
            text(case, "text_sha256"),
            "case {name}"
        );
        // Streaming: the per-token pieces concatenate to the whole decode.
        let mut decoder = vocab.decoder();
        let mut streamed: String = ids.iter().map(|&id| decoder.push(id)).collect();
        streamed.push_str(&decoder.finish());
        assert_eq!(streamed, whole, "case {name} streamed");
    }
}

fn expected(v: &Value) -> Split {
    Split {
        reasoning: text(v, "reasoning_text").to_owned(),
        normal: text(v, "normal_text").to_owned(),
    }
}

#[test]
fn reasoning_cases_all_pass() {
    let cases = fixture("reasoning_cases.json");
    assert_eq!(cases["parser"], "qwen3");
    assert_eq!(cases["force_reasoning"], false);
    let all = cases["cases"].as_array().unwrap();
    assert!(!all.is_empty());
    for case in all {
        let name = text(case, "name");
        assert_eq!(
            split(text(case, "text"), Reasoning::Detect),
            expected(&case["non_stream"]),
            "case {name}"
        );
        for (n, s) in case["streams"].as_array().unwrap().iter().enumerate() {
            let mut stream = Stream::default();
            let chunks = s["chunks"].as_array().unwrap();
            let deltas = s["deltas"].as_array().unwrap();
            assert_eq!(chunks.len(), deltas.len(), "case {name} stream {n}");
            for (i, (chunk, delta)) in chunks.iter().zip(deltas).enumerate() {
                let got = stream.push(chunk.as_str().unwrap());
                assert_eq!(got, expected(delta), "case {name} stream {n} chunk {i}");
            }
            assert_eq!(
                stream.finish(),
                expected(&s["finish"]),
                "case {name} stream {n} finish"
            );
        }
    }
}
/// A protocol stub (v3): `info` answers fixed hashes and an empty `features` list, `encode_chat` one ID per message
/// (its content length), the UTF-8 length of each stop and the parsed tools, or a typed `bad_request` for the content
/// "refuse" and an `invalid_value` naming `tools[0].function.parameters` for a tool named "bad"; `tool_param`/
/// `tool_args` echo their input. The mode argument breaks it on purpose (`v1`/`v2` are older sidecars, `no_features`
/// omits `info.features`, `bad_features` sends an unsorted list, `no_lens` drops the stop counts). The `boundary*`
/// modes advertise feature `message_boundary` (M2.10 D4 (c)): `boundary` replies `last_message_start` = the IDs
/// before the last message, `boundary_missing` omits it, `boundary_past_end` names one past the IDs, and
/// `boundary_unadvertised` sends the field without advertising the feature. The `hang_*` modes stop answering without
/// exiting (audit N1): `hang_encode` never answers an `encode_chat`, `hang_tool` never answers a `tool_param`, and
/// `no_read` answers `info` and then neither reads its stdin nor answers. An optional second argument names a starts
/// log: each child appends one line at startup, so a test counts respawns (OH-01) and the first child knows it is
/// the first: `hang_once` never answers the first child's `encode_chat` (its respawns answer), `exit_first` exits the
/// first child after one reply (its respawns serve), `die_first` exits the first child when it reads an
/// `encode_chat` (it dies under the call, without a reply; its respawns serve) and `die_encode` does that in every
/// child. An optional third argument names a hold file: a respawned child waits while it exists before it serves
/// (a respawn a test holds, with no wall-clock sleep).
const SIDECAR_STUB: &str = r#"
import json, os, struct, sys, time
mode = sys.argv[1]
inp, out = sys.stdin.buffer, sys.stdout.buffer
starts = 1
if len(sys.argv) > 2:
    with open(sys.argv[2], "a") as log:
        log.write("start\n")
    with open(sys.argv[2]) as log:
        starts = len(log.read().splitlines())
first = starts == 1
while len(sys.argv) > 3 and not first and os.path.exists(sys.argv[3]):
    time.sleep(0.01)
replies = 0
while True:
    header = inp.read(4)
    if len(header) < 4:
        sys.exit(0)
    request = json.loads(inp.read(struct.unpack("<I", header)[0]))
    if mode == "garbage":
        out.write(struct.pack("<I", 3) + b"{x}")
        out.flush()
        continue
    op = request["op"]
    if op == "encode_chat" and (mode == "die_encode" or (mode == "die_first" and first)):
        sys.exit(0)
    if (mode, op) in (("hang_encode", "encode_chat"), ("hang_tool", "tool_param")) or (
            mode == "hang_once" and first and op == "encode_chat"):
        time.sleep(3600)
    tools = json.loads(request["tools_json"]) if "tools_json" in request else None
    if op == "info":
        protocol = {"wrong_protocol": "other", "v1": "cere-tokenizer-sidecar-v1",
                    "v2": "cere-tokenizer-sidecar-v2"}.get(mode, "cere-tokenizer-sidecar-v3")
        reply = {"ok": True, "op": "info", "protocol": protocol, "tokenizer_sha256": "a" * 64,
                 "template_sha256": "b" * 64, "transformers": "5.12.1", "tokenizers": "0.22.2", "features": []}
        if mode in ("v1", "v2", "no_features"):
            del reply["features"]
        if mode == "bad_features":
            reply["features"] = ["b", "a"]
        if mode in ("boundary", "boundary_missing", "boundary_past_end"):
            reply["features"] = ["message_boundary"]
    elif op == "tool_param":
        reply = {"ok": True, "op": op, "fragment": json.dumps([request["tool_json"], request["func"],
                                                               request["name"], request["raw"]])}
    elif op == "tool_args":
        reply = {"ok": True, "op": op, "arguments": json.dumps([request["tool_json"], request["func"],
                                                                request["params"]])}
    elif any(m["content"] == "refuse" for m in request["messages"]):
        reply = {"ok": False, "error": {"type": "bad_request", "message": "fixed"}}
    elif tools and tools[0]["function"]["name"] == "bad":
        reply = {"ok": False, "error": {"type": "invalid_value", "message": "fixed",
                                        "param": "tools[0].function.parameters"}}
    else:
        reply = {"ok": True, "op": "encode_chat", "ids": [len(m["content"]) for m in request["messages"]],
                 "stop_token_lens": [len(s.encode()) for s in request["stop"]], "tools_prepared": tools}
        if mode == "no_lens":
            del reply["stop_token_lens"]
        if mode in ("boundary", "boundary_unadvertised"):
            reply["last_message_start"] = len(reply["ids"]) - 1
        if mode == "boundary_past_end":
            reply["last_message_start"] = len(reply["ids"]) + 1
    data = json.dumps(reply).encode()
    out.write(struct.pack("<I", len(data)) + data)
    out.flush()
    replies += 1
    if (mode == "exit_after_one" or (mode == "exit_first" and first)) and replies == 1:
        sys.exit(0)
    if mode == "no_read":
        time.sleep(3600)
"#;

/// The I/O deadline of the stubs that answer.
const STUB_DEADLINE: Duration = Duration::from_secs(60);

fn stub(mode: &str) -> Sidecar {
    stub_with(mode, STUB_DEADLINE)
}

fn stub_with(mode: &str, deadline: Duration) -> Sidecar {
    let argv = ["python3", "-c", SIDECAR_STUB, mode].map(str::to_owned);
    Sidecar::spawn(&argv, deadline).expect("spawning the stub")
}

/// [`stub_with`] whose children append a line to `starts` when they start (one line per child).
fn stub_logged(mode: &str, deadline: Duration, starts: &Path) -> Sidecar {
    let argv = ["python3", "-c", SIDECAR_STUB, mode]
        .into_iter()
        .map(str::to_owned)
        .chain([starts.display().to_string()])
        .collect::<Vec<_>>();
    Sidecar::spawn(&argv, deadline).expect("spawning the stub")
}

/// [`stub_logged`] whose respawned children wait while `hold` exists before they serve.
fn stub_held(mode: &str, starts: &Path, hold: &Path) -> Sidecar {
    let argv = ["python3", "-c", SIDECAR_STUB, mode]
        .into_iter()
        .map(str::to_owned)
        .chain([starts.display().to_string(), hold.display().to_string()])
        .collect::<Vec<_>>();
    Sidecar::spawn(&argv, STUB_DEADLINE).expect("spawning the stub")
}

/// How long a test waits for a health change it expects.
const WATCH: Duration = Duration::from_secs(10);

/// Every later change of `s`'s health, in order (its watcher).
fn watched(s: &Sidecar) -> std::sync::mpsc::Receiver<SidecarHealth> {
    let (tx, rx) = std::sync::mpsc::channel();
    s.watch(move |health| {
        let _ = tx.send(health);
    });
    rx
}

/// The children a [`stub_logged`] sidecar started.
fn start_count(starts: &Path) -> usize {
    std::fs::read_to_string(starts)
        .unwrap_or_default()
        .lines()
        .count()
}

fn user(content: &str) -> ChatMessage {
    ChatMessage::text(Role::User, content)
}

fn encode(s: &Sidecar, messages: &[ChatMessage]) -> Result<Vec<u32>, SidecarError> {
    s.encode_chat(messages, None, false, &[], &Default::default())
        .map(|e| e.ids)
}

#[test]
fn sidecar_info_and_encode_chat_round_trip() {
    let s = stub("ok");
    let info = s.info().unwrap();
    assert_eq!(info.tokenizer_sha256, "a".repeat(64));
    assert_eq!(info.template_sha256, "b".repeat(64));
    assert_eq!(
        (info.transformers.as_str(), info.tokenizers.as_str()),
        ("5.12.1", "0.22.2")
    );
    assert!(info.features.is_empty());
    assert_eq!(
        encode(&s, &[user("abc"), user("hello")]).unwrap(),
        vec![3, 5]
    );
    // A typed refusal leaves the child serving.
    match encode(&s, &[user("refuse")]) {
        Err(SidecarError::Refused { kind, param, .. }) => {
            assert_eq!((kind.as_str(), param), ("bad_request", None))
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(s.health(), SidecarHealth::Ready);
    assert_eq!(encode(&s, &[user("x")]).unwrap(), vec![1]);
}

/// v2 (spec m24 D8): `encode_chat` carries the raw tools, `render_tools` and the stops and returns the stop token
/// counts and the prepared tools; a schema refusal names its path; `tool_param`/`tool_args` send the prepared
/// tool, the function and the raw values.
#[test]
fn sidecar_v2_tools_stops_and_value_ops() {
    let s = stub("ok");
    let tools = r#"[{"type":"function","function":{"name":"f"}}]"#;
    let stops = ["ab".to_owned(), "é".to_owned()];
    let encoded = s
        .encode_chat(
            &[user("abc")],
            Some(tools),
            true,
            &stops,
            &Default::default(),
        )
        .unwrap();
    assert_eq!(encoded.ids, vec![3]);
    assert_eq!(encoded.stop_token_lens, vec![2, 2]);
    let prepared = encoded.tools_prepared.unwrap();
    assert_eq!(
        prepared,
        vec![json!({"type": "function", "function": {"name": "f"}})]
    );
    // Tool-free: no prepared tools.
    let encoded = s
        .encode_chat(&[user("abc")], None, false, &[], &Default::default())
        .unwrap();
    assert_eq!(
        (encoded.stop_token_lens, encoded.tools_prepared),
        (vec![], None)
    );
    let bad = r#"[{"type":"function","function":{"name":"bad"}}]"#;
    match s.encode_chat(&[user("abc")], Some(bad), false, &[], &Default::default()) {
        Err(SidecarError::Refused { kind, param, .. }) => assert_eq!(
            (kind.as_str(), param.as_deref()),
            ("invalid_value", Some("tools[0].function.parameters"))
        ),
        other => panic!("{other:?}"),
    }
    let fragment = s
        .tool_param(Some(&prepared[0]), Some("f"), "p", "1")
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&fragment).unwrap(),
        json!([prepared[0], "f", "p", "1"])
    );
    assert_eq!(
        s.tool_param(None, None, "p", "1").unwrap(),
        r#"[null, null, "p", "1"]"#
    );
    let params = [("a".to_owned(), "x".to_owned())];
    let arguments = s.tool_args(None, "g", &params).unwrap();
    assert_eq!(arguments, r#"[null, "g", [["a", "x"]]]"#);
    assert_eq!(s.health(), SidecarHealth::Ready);
    // A reply without the stop counts is outside the protocol.
    let s = stub("no_lens");
    assert!(matches!(
        s.encode_chat(&[user("abc")], None, false, &stops, &Default::default()),
        Err(SidecarError::Protocol(_))
    ));
    // Before its first `info` was answered the sidecar is never respawned in the background.
    assert_eq!(s.health(), SidecarHealth::Down);
}

/// M2.10 C5s (ledger S, D4 (c)): after an `info` advertising feature `message_boundary`, `encode_chat` carries
/// `last_message_start` (multi-turn: the IDs before the last message; one message: 0); without the feature the field
/// is 0 and a reply that carries it anyway is refused, as is an advertised reply without it or one past the IDs.
#[test]
fn sidecar_message_boundary_follows_the_feature() {
    let boundary = |s: &Sidecar, messages: &[ChatMessage]| {
        s.encode_chat(messages, None, false, &[], &Default::default())
            .map(|e| (e.ids, e.last_message_start))
    };
    let s = stub("boundary");
    assert_eq!(s.info().unwrap().features, ["message_boundary"]);
    assert_eq!(
        boundary(&s, &[user("abc"), user("hello"), user("x")]).unwrap(),
        (vec![3, 5, 1], 2)
    );
    assert_eq!(boundary(&s, &[user("abc")]).unwrap(), (vec![3], 0));
    // Without the feature: no hint.
    let s = stub("ok");
    s.info().unwrap();
    assert_eq!(
        boundary(&s, &[user("ab"), user("c")]).unwrap(),
        (vec![2, 1], 0)
    );
    for mode in [
        "boundary_missing",
        "boundary_past_end",
        "boundary_unadvertised",
    ] {
        let s = stub(mode);
        s.info().unwrap();
        let changes = watched(&s);
        assert!(
            matches!(
                boundary(&s, &[user("ab"), user("c")]),
                Err(SidecarError::Protocol(_))
            ),
            "{mode}"
        );
        // The child was stopped; its successor starts at once (the child had served `info`).
        assert_eq!(
            changes.recv_timeout(WATCH),
            Ok(SidecarHealth::Respawning),
            "{mode}"
        );
    }
}

/// OH-01: a child that exits between calls is noticed at once, before any call or probe (its pipe reader sees EOF:
/// the watcher hears `Respawning`), and its successor starts at once on the respawn thread. A call meanwhile waits
/// for it and is served by the respawned child, which reports the same `info`; the watcher then hears `Ready`. The
/// respawn is held (the stub waits while `hold` exists), so the `Respawning` window is observed without a race.
#[test]
fn sidecar_child_exit_is_noticed_at_once_and_respawned() {
    let t = Tmp::new("sidecar-exit-first");
    let (starts, hold) = (t.join("starts"), t.join("hold"));
    std::fs::write(&hold, b"").unwrap();
    let s = stub_held("exit_first", &starts, &hold);
    let changes = watched(&s);
    // Reply 1: the first child exits after it.
    s.info().unwrap();
    assert_eq!(changes.recv_timeout(WATCH), Ok(SidecarHealth::Respawning));
    assert_eq!(s.health(), SidecarHealth::Respawning);
    std::thread::scope(|scope| {
        let call = scope.spawn(|| encode(&s, &[user("ab")]));
        std::thread::sleep(Duration::from_millis(200));
        assert!(!call.is_finished(), "the call waits for the held respawn");
        std::fs::remove_file(&hold).unwrap();
        assert_eq!(call.join().unwrap().unwrap(), vec![2]);
    });
    assert_eq!(changes.recv_timeout(WATCH), Ok(SidecarHealth::Ready));
    assert_eq!(s.health(), SidecarHealth::Ready);
    assert_eq!(start_count(&starts), 2);
    assert_eq!(s.stats(), SidecarStats { respawns: 1 });
    assert!(changes.try_recv().is_err());
}

/// OH-01: a child that misses the I/O deadline once is killed and that call fails; the next call respawns the child
/// and is served by the same `Sidecar` (the `hang_once` stub answers on its respawn).
#[test]
fn sidecar_hung_once_is_respawned_and_the_next_call_is_served() {
    let t = Tmp::new("sidecar-hang-once");
    let starts = t.join("starts");
    let s = stub_logged("hang_once", Duration::from_millis(400), &starts);
    s.info().unwrap();
    match encode(&s, &[user("hi")]) {
        Err(SidecarError::Gone(m)) => assert!(m.contains("no reply within 400 ms"), "{m}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(encode(&s, &[user("abc")]).unwrap(), vec![3]);
    assert_eq!(s.health(), SidecarHealth::Ready);
    assert_eq!(start_count(&starts), 2);
    assert_eq!(s.stats(), SidecarStats { respawns: 1 });
}

/// OH-01 (ops-gpu-smoke run3-A, the request sent right after the kill): a child that dies under a call (it read the
/// request and exited without a reply, as a child killed in the middle of an exchange does) does not fail that call:
/// the ops are pure, so it runs once more on a successor started at once and is served. Fails before the fix: the
/// call failed `Gone` (503 `not_ready`) and only the next call was served.
#[test]
fn sidecar_child_that_dies_under_a_call_is_respawned_and_the_call_served() {
    let t = Tmp::new("sidecar-die-first");
    let starts = t.join("starts");
    let s = stub_logged("die_first", STUB_DEADLINE, &starts);
    s.info().unwrap();
    let changes = watched(&s);
    assert_eq!(encode(&s, &[user("ab")]).unwrap(), vec![2]);
    assert_eq!(start_count(&starts), 2);
    assert_eq!(s.stats(), SidecarStats { respawns: 1 });
    assert_eq!(
        changes.try_iter().collect::<Vec<_>>(),
        [SidecarHealth::Respawning, SidecarHealth::Ready]
    );
    assert_eq!(encode(&s, &[user("abc")]).unwrap(), vec![3]);
}

/// OH-01: that retry runs once per call and only on a successor started at once: when every child dies under
/// `encode_chat`, the call fails `Gone` after one respawn (the respawned child never served, so the next respawn
/// waits for its backoff). Fails before the fix: no respawn was tried within the call.
#[test]
fn sidecar_whose_children_all_die_under_a_call_fails_it_after_one_retry() {
    let t = Tmp::new("sidecar-die-encode");
    let starts = t.join("starts");
    let s = stub_logged("die_encode", STUB_DEADLINE, &starts);
    s.info().unwrap();
    let changes = watched(&s);
    let result = encode(&s, &[user("ab")]);
    assert!(matches!(result, Err(SidecarError::Gone(_))), "{result:?}");
    assert_eq!(s.stats(), SidecarStats { respawns: 1 });
    assert_eq!(start_count(&starts), 2);
    assert_eq!(
        changes.try_iter().collect::<Vec<_>>(),
        [
            SidecarHealth::Respawning,
            SidecarHealth::Ready,
            SidecarHealth::Down
        ]
    );
}

/// OH-01: a sidecar whose every child exits after its startup `info` is respawned under a bounded backoff (no
/// respawn storm: the first respawn at once, then 250, 500, 1,000 ms after each loss) and every call fails `Gone`.
#[test]
fn sidecar_that_always_exits_is_respawned_with_backoff() {
    let t = Tmp::new("sidecar-always-exits");
    let starts = t.join("starts");
    let s = stub_logged("exit_after_one", STUB_DEADLINE, &starts);
    s.info().unwrap();
    let start = Instant::now();
    let mut calls = 0;
    while start.elapsed() < Duration::from_secs(3) {
        let result = encode(&s, &[user("a")]);
        assert!(matches!(result, Err(SidecarError::Gone(_))), "{result:?}");
        s.health();
        calls += 1;
        std::thread::sleep(Duration::from_millis(10));
    }
    // A respawn may be in flight (counted, its child not started yet).
    let counted = s.stats().respawns;
    let respawns = start_count(&starts) as u64 - 1;
    assert!(
        (2..=5).contains(&counted) && (counted - 1..=counted).contains(&respawns),
        "{counted} respawns counted, {respawns} started, for {calls} calls"
    );
}

#[test]
fn sidecar_reply_outside_the_protocol_stops_the_child() {
    let s = stub("garbage");
    assert!(matches!(s.info(), Err(SidecarError::Protocol(_))));
    assert_eq!(s.health(), SidecarHealth::Down);
    for mode in ["wrong_protocol", "v1", "v2", "no_features", "bad_features"] {
        let s = stub(mode);
        assert!(matches!(s.info(), Err(SidecarError::Protocol(_))), "{mode}");
        assert_eq!(s.health(), SidecarHealth::Down, "{mode}");
    }
}

#[test]
fn sidecar_missing_program_fails_to_spawn() {
    let argv = ["/nonexistent/cere-sidecar".to_owned()];
    assert!(matches!(
        Sidecar::spawn(&argv, STUB_DEADLINE),
        Err(SidecarError::Spawn(_))
    ));
    assert!(matches!(
        Sidecar::spawn(&[], STUB_DEADLINE),
        Err(SidecarError::Spawn(_))
    ));
}

/// Audit N1 with OH-01: a child that stops answering without exiting fails the call at the I/O deadline (not never)
/// and is killed. The next call runs on a respawned child (this stub hangs again, so it fails at the deadline too),
/// and the calls after it fail at once until the respawn backoff allows another child.
#[test]
fn sidecar_that_stops_answering_is_killed_at_the_deadline() {
    for mode in ["hang_encode", "hang_tool"] {
        let s = stub_with(mode, Duration::from_millis(400));
        let call = || {
            if mode == "hang_encode" {
                encode(&s, &[user("hi")]).map(|_| ())
            } else {
                s.tool_param(None, None, "p", "v").map(|_| ())
            }
        };
        s.info().unwrap();
        let changes = watched(&s);
        let started = Instant::now();
        let result = call();
        let waited = started.elapsed();
        match result {
            Err(SidecarError::Gone(m)) => {
                assert!(m.contains("no reply within 400 ms"), "{mode}: {m}")
            }
            other => panic!("{mode}: {other:?}"),
        }
        assert!(waited >= Duration::from_millis(400), "{mode}: {waited:?}");
        assert!(waited < Duration::from_secs(30), "{mode}: {waited:?}");
        assert_eq!(
            changes.recv_timeout(WATCH),
            Ok(SidecarHealth::Respawning),
            "{mode}"
        );
        match call() {
            Err(SidecarError::Gone(m)) => {
                assert!(m.contains("no reply within 400 ms"), "{mode}: {m}")
            }
            other => panic!("{mode}: {other:?}"),
        }
        let again = Instant::now();
        match call() {
            Err(SidecarError::Gone(m)) => assert!(m.contains("respawn"), "{mode}: {m}"),
            other => panic!("{mode}: {other:?}"),
        }
        assert!(again.elapsed() < Duration::from_millis(300), "{mode}");
        assert_eq!(s.stats(), SidecarStats { respawns: 1 }, "{mode}");
        // The respawned child never served, so after its deadline the sidecar is down until the backoff ends.
        assert_eq!(
            changes.try_iter().collect::<Vec<_>>(),
            [SidecarHealth::Ready, SidecarHealth::Down],
            "{mode}"
        );
    }
}

/// Audit N1: `health` (the readiness probe) never waits for the call in flight, and the deadline kill is reported
/// (its successor starts at once).
#[test]
fn sidecar_health_does_not_wait_for_a_call_in_flight() {
    let s = stub_with("hang_encode", Duration::from_secs(3));
    s.info().unwrap();
    let changes = watched(&s);
    std::thread::scope(|scope| {
        let call = scope.spawn(|| encode(&s, &[user("hi")]));
        std::thread::sleep(Duration::from_millis(300));
        let probe = Instant::now();
        assert_eq!(
            s.health(),
            SidecarHealth::Ready,
            "the child is still serving its call"
        );
        assert!(
            probe.elapsed() < Duration::from_millis(500),
            "{:?}",
            probe.elapsed()
        );
        assert!(matches!(call.join().unwrap(), Err(SidecarError::Gone(_))));
    });
    assert_eq!(changes.recv_timeout(WATCH), Ok(SidecarHealth::Respawning));
}

/// Audit N1: a child that never reads its stdin blocks the request write on a full pipe; the deadline covers the
/// write too.
#[test]
fn sidecar_that_does_not_read_its_stdin_fails_the_write_at_the_deadline() {
    let s = stub_with("no_read", Duration::from_millis(500));
    s.info().unwrap();
    let changes = watched(&s);
    let big = "x".repeat(4 << 20);
    let started = Instant::now();
    let result = encode(&s, &[user(&big)]);
    assert!(matches!(result, Err(SidecarError::Gone(_))), "{result:?}");
    assert!(started.elapsed() < Duration::from_secs(30));
    assert_eq!(changes.recv_timeout(WATCH), Ok(SidecarHealth::Respawning));
}

/// Stand-ins for the image's libraries, so the real `scripts/native/tokenizer_sidecar.py` runs on the host: a
/// tokenizer that encodes code points (no added tokens, so N5 never escapes), and a detector whose `object`
/// conversion is `json.loads`, as SGLang's is.
const REFERENCE_STUBS: [(&str, &str); 10] = [
    (
        "transformers/__init__.py",
        "__version__ = '0-stub'\nclass _Backend:\n    def to_str(self):\n        return '{\"added_tokens\": []}'\nclass _Tok:\n    eos_token_id = None\n    added_tokens_decoder = {}\n    backend_tokenizer = _Backend()\n    def get_added_vocab(self):\n        return {}\n    def encode(self, text, **kw):\n        return [ord(c) for c in text]\n    def apply_chat_template(self, messages, **kw):\n        return ''.join(m['content'] for m in messages)\nclass AutoTokenizer:\n    @staticmethod\n    def from_pretrained(path, **kw):\n        return _Tok()\n",
    ),
    (
        "tokenizers/__init__.py",
        "__version__ = '0-stub'\nclass Tokenizer:\n    @staticmethod\n    def from_str(text):\n        return Tokenizer()\n",
    ),
    (
        "jsonschema/__init__.py",
        "class SchemaError(Exception):\n    pass\nclass Draft202012Validator:\n    @staticmethod\n    def check_schema(schema):\n        pass\n",
    ),
    ("sglang/__init__.py", ""),
    ("sglang/srt/__init__.py", ""),
    ("sglang/srt/entrypoints/__init__.py", ""),
    ("sglang/srt/entrypoints/openai/__init__.py", ""),
    (
        "sglang/srt/entrypoints/openai/protocol.py",
        "class Tool:\n    @classmethod\n    def model_validate(cls, d):\n        tool = cls()\n        tool.parameters = d['function']['parameters']\n        return tool\n",
    ),
    (
        "sglang/srt/function_call/utils.py",
        "def normalize_json_schema_types(schema):\n    pass\n",
    ),
    (
        "sglang/srt/function_call/qwen3_coder_detector.py",
        "import json\nclass Qwen3CoderDetector:\n    def _get_arguments_config(self, func, tools):\n        return tools[0].parameters['properties']\n    def _convert_param_value(self, value, name, config, func):\n        return json.loads(value) if config.get(name, {}).get('type') == 'object' else value\n",
    ),
];

/// D8/D9 over the real sidecar script: a converted value holding a lone surrogate (`json.loads` of the escape
/// `"\ud800"`) reaches the client as the typed `convert_error`, and the same child serves the next request.
#[test]
fn sidecar_unencodable_conversion_is_refused_and_the_child_serves_on() {
    let dir = support::StateDir::new();
    for (path, text) in REFERENCE_STUBS {
        let path = dir.path().join("stubs").join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    let pythonpath = format!("PYTHONPATH={}", dir.path().join("stubs").display());
    let script = root().join("scripts/native/tokenizer_sidecar.py");
    let artifact = root().join("tests/fixtures/model_src");
    let argv = [
        "env",
        &pythonpath,
        "PYTHONDONTWRITEBYTECODE=1",
        "python3",
        script.to_str().unwrap(),
        "--artifact",
        artifact.to_str().unwrap(),
    ]
    .map(str::to_owned);
    let s = Sidecar::spawn(&argv, STUB_DEADLINE).expect("spawning the sidecar");
    assert!(s.info().is_ok());
    let tool = json!({"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"o": {"type": "object"}}}}});
    let refused = |result: Result<String, SidecarError>| match result {
        Err(SidecarError::Refused { kind, param, .. }) => {
            assert_eq!((kind.as_str(), param), ("convert_error", None))
        }
        other => panic!("{other:?}"),
    };
    refused(s.tool_param(Some(&tool), Some("f"), "o", r#""\ud800""#));
    assert_eq!(s.health(), SidecarHealth::Ready);
    let params = [("o".to_owned(), r#"{"k": "\udfff"}"#.to_owned())];
    refused(s.tool_args(Some(&tool), "f", &params));
    assert_eq!(s.health(), SidecarHealth::Ready);
    assert_eq!(
        s.tool_param(Some(&tool), Some("f"), "o", r#"{"k": "\u00e9"}"#)
            .unwrap(),
        "\"o\": {\"k\": \"\u{e9}\"}"
    );
    assert_eq!(
        s.encode_chat(&[user("ab")], None, false, &[], &Default::default())
            .unwrap()
            .ids,
        vec![97, 98]
    );
}

// ---------------------------------------------------------------------------------------------------------
// The cered request path over the toy executor and the stub sidecar.

/// Starts every test prompt: the log scan looks for it.
const SENTINEL: &str = "cere-native-sentinel-prompt";

/// The stub sidecar encodes one user message as `[1]` + its UTF-8 bytes, and the toy's next token is
/// `(31·last + position) mod 70000`: the prompt length and its last byte decide the output. The fixture
/// artifact's vocabulary is `a`=0, `b`=1, `ab`=2 (every other ID decodes to nothing) and its stop IDs are 7, 6.
/// `TEXT`: [2235, 2 ("ab"), 780, …], no stop in reach. `STOP`: [2240, 7] (EOS).
const TEXT: (usize, char) = (716, '1');
const STOP: (usize, char) = (566, '6');
/// The served context (m25 D6) and a prompt that leaves 4 tokens of it (the toy's output is `TEXT`'s rule).
const CONTEXT: u64 = 200_000;
const LONG: (usize, char) = (199_996, '1');

/// A user message whose prompt is `len` tokens ending in `last`.
fn prompt((len, last): (usize, char)) -> String {
    let mut s = SENTINEL.to_owned();
    while s.len() < len - 2 {
        s.push('x');
    }
    s.push(last);
    s
}

fn body(p: (usize, char), max_tokens: u64, stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(p)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    })
}

/// What the toy commits for `p` within `max_tokens` (up to and including a stop ID).
fn toy_output((len, last): (usize, char), max_tokens: usize) -> Vec<u32> {
    let mut ids = toy_tokens(last as u32, len as u64, max_tokens, FIXTURE_VOCAB);
    if let Some(i) = ids.iter().position(|id| [6, 7].contains(id)) {
        ids.truncate(i + 1);
    }
    ids
}

fn expected_text(native: &Native, ids: &[u32]) -> String {
    Vocab::load(&native.artifact).unwrap().decode(ids)
}

fn rid(response: &reqwest::Response) -> String {
    response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned()
}

/// Every object key path of `v` (`[]` for array elements): the JSON shape, values aside.
fn paths(v: &Value, prefix: &str, out: &mut BTreeSet<String>) {
    match v {
        Value::Object(map) => {
            for (k, x) in map {
                let p = format!("{prefix}.{k}");
                out.insert(p.clone());
                paths(x, &p, out);
            }
        }
        Value::Array(items) => items
            .iter()
            .for_each(|x| paths(x, &format!("{prefix}[]"), out)),
        _ => {}
    }
}

fn shape<'a>(values: impl IntoIterator<Item = &'a Value>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    values.into_iter().for_each(|v| paths(v, "", &mut out));
    out
}

fn assert_same_shape(native: BTreeSet<String>, sglang: BTreeSet<String>) {
    let only_native: Vec<_> = native.difference(&sglang).collect();
    let only_sglang: Vec<_> = sglang.difference(&native).collect();
    assert!(
        only_native.is_empty() && only_sglang.is_empty(),
        "native only: {only_native:?}, sglang_http only: {only_sglang:?}"
    );
}

/// The `sglang_http` unary body and SSE events for the same API (mock SGLang with its cache report on, a
/// repeated request, so `prompt_tokens_details` is an object as on the native path).
async fn sglang_reference() -> (Value, Vec<Event>) {
    let mock = MockConfig {
        cache_granularity: 1,
        server_info: ServerInfo {
            enable_cache_report: true,
            ..ServerInfo::default()
        },
        ..MockConfig::default()
    };
    let h = Harness::start(
        mock,
        &[("enable_cache_report = false", "enable_cache_report = true")],
    )
    .await;
    let mut unary = Value::Null;
    for _ in 0..2 {
        unary = h
            .chat(&chat_body(false))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    }
    let mut stream = chat_body(true);
    stream["stream_options"] = json!({"include_usage": true});
    let mut events = Vec::new();
    for _ in 0..2 {
        events = SseReader::new(h.chat(&stream).send().await.unwrap())
            .rest()
            .await;
    }
    (unary, events)
}

/// Stops `cered` and checks that no prompt text reached its log or its state directory.
async fn assert_no_text(native: &mut Native) {
    native.cered.signal("TERM");
    let (code, stderr) = native.cered.exit(Duration::from_secs(30)).await;
    assert_eq!(code, Some(0), "{stderr}");
    assert!(!stderr.contains(SENTINEL), "prompt text in the cered log");
    let mut dirs = vec![native.state.path().to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                let text = String::from_utf8_lossy(&bytes);
                assert!(
                    !text.contains(SENTINEL),
                    "prompt text in {}",
                    path.display()
                );
            }
        }
    }
}

/// Unary: the detokenized toy output, `length`/`stop` finishes with `matched_stop`, usage with
/// `cached_tokens` 0, and the `sglang_http` body shape field for field.
#[tokio::test]
async fn unary_body_usage_and_finish_reasons() {
    let mut native = Native::ready(Case::matching()).await;
    assert_eq!(native.submissions().await, 0);

    let response = native.chat(&body(TEXT, 4, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let unary: Value = response.json().await.unwrap();
    let text = expected_text(&native, &toy_output(TEXT, 4));
    assert_eq!(text, "ab");
    assert_eq!(unary["object"], "chat.completion");
    assert_eq!(unary["id"], rid.as_str());
    assert_eq!(unary["model"], MODEL);
    let choice = &unary["choices"][0];
    assert_eq!(choice["message"]["role"], "assistant");
    assert_eq!(choice["message"]["content"], text.as_str());
    assert_eq!(choice["message"]["reasoning_content"], Value::Null);
    assert_eq!(choice["finish_reason"], "length");
    assert_eq!(choice["matched_stop"], Value::Null);
    assert_eq!(
        unary["usage"],
        json!({
            "prompt_tokens": TEXT.0,
            "completion_tokens": 4,
            "total_tokens": TEXT.0 + 4,
            "prompt_tokens_details": {"cached_tokens": 0},
            "reasoning_tokens": 0,
        })
    );
    let cere = &unary["cere"];
    assert_eq!(cere["state"], "completed");
    assert_eq!(cere["execution_mode"], "native");
    assert_eq!(
        (&cere["cached_tokens"], &cere["cache_report"]),
        (&json!(0), &json!("reported"))
    );
    let status = native.terminal(&rid).await;
    assert_eq!(status["state"], "completed");
    assert_eq!(status["usage"]["completion_tokens"], 4, "{status}");

    let stop: Value = native
        .chat(&body(STOP, 8, false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids = toy_output(STOP, 8);
    assert_eq!(ids.last(), Some(&7));
    assert_eq!(stop["choices"][0]["finish_reason"], "stop", "{stop}");
    assert_eq!(stop["choices"][0]["matched_stop"], 7);
    assert_eq!(stop["usage"]["completion_tokens"], ids.len());
    assert_eq!(native.submissions().await, 2);

    let (sglang, _) = sglang_reference().await;
    assert_same_shape(shape([&unary]), shape([&sglang]));
    assert_no_text(&mut native).await;
}

/// SSE: role chunk, content chunks, the finish chunk, the usage chunk and `[DONE]`; the streamed text equals the
/// unary text, and the events have the `sglang_http` shapes.
#[tokio::test]
async fn sse_matches_unary_text_and_sglang_shapes() {
    let mut native = Native::ready(Case::matching()).await;
    let unary: Value = native
        .chat(&body(TEXT, 6, false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut streamed = body(TEXT, 6, true);
    streamed["stream_options"] = json!({"include_usage": true});
    let response = native.chat(&streamed).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let events = SseReader::new(response).rest().await;
    let finish = check_stream(&events);
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(finish["cere"]["state"], "completed");
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) => v
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str),
            Event::Done => None,
        })
        .collect();
    assert_eq!(
        text,
        unary["choices"][0]["message"]["content"].as_str().unwrap()
    );
    assert_eq!(text, expected_text(&native, &toy_output(TEXT, 6)));
    let usage = &events[events.len() - 2].data()["usage"];
    assert_eq!(usage["completion_tokens"], 6, "{usage}");
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(native.terminal(&rid).await["state"], "completed");

    let (_, sglang) = sglang_reference().await;
    let data = |events: &[Event]| -> Vec<Value> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Data(v) => Some(v.clone()),
                Event::Done => None,
            })
            .collect()
    };
    assert_same_shape(shape(&data(&events)), shape(&data(&sglang)));
    assert_no_text(&mut native).await;
}

/// Fail closed (spec m22 X12 list): each refusal is a 400 with its typed reason before any executor call, and
/// so is an over-length prompt (`context_length_exceeded`); `native_submissions` stays 0 until a valid request.
#[tokio::test]
async fn refusals_and_over_context_never_reach_the_executor() {
    let native = Native::ready(Case::matching()).await;
    let valid = body(TEXT, 4, false);
    let with = |key: &str, value: Value| {
        let mut b = valid.clone();
        b[key] = value;
        b
    };
    let image = with(
        "messages",
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]),
    );
    // Stop strings and tools are served natively since M2.4 (C7), sampling fields and logprobs since M2.7 (D11's
    // profile declares them), an absent temperature samples since M2.7 D10 (Q4 (a), C8), sampled n > 1 since
    // ledger CR4; n above the profile's 16 rows (m26 D11; 4 before m23 design note P8) stays refused, and m213 D5:
    // the native profile has no image decode combination, so an image part is refused before its URL.
    let cases = [
        (with("n", json!(17)), "n"),
        (image, "messages[0].content[0]"),
    ];
    for (request, param) in cases {
        let response = native.chat(&request).send().await.unwrap();
        assert_eq!(response.status(), 400, "{param}");
        let error: Value = response.json().await.unwrap();
        assert_eq!(error["error"]["code"], "unsupported_parameter", "{error}");
        assert_eq!(error["error"]["param"], param, "{error}");
        assert_eq!(native.submissions().await, 0, "{param}");
    }
    // M2.7 D8: native no longer refuses top_p; with temperature 0 it is greedy decoding: accepted.
    let response = native
        .chat(&with("top_p", json!(0.5)))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.submissions().await, 1);

    // C3 (m25 D6): prompt + max_tokens one over the 200,000-token context (200,001); max_tokens stays within
    // `max_output_tokens` (2,048), so the prompt is long.
    let over = body(LONG, CONTEXT - LONG.0 as u64 + 1, false);
    let response = native.chat(&over).send().await.unwrap();
    assert_eq!(response.status(), 400);
    let rid = rid(&response);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["error"]["code"], "context_length_exceeded", "{error}");
    assert_eq!(error["cere"]["state"], "failed");
    assert_eq!(
        native.terminal(&rid).await["reason"],
        "context_length_exceeded"
    );
    assert_eq!(native.submissions().await, 1);
    // Exactly at the limit fits.
    let fits = body(LONG, CONTEXT - LONG.0 as u64, false);
    let response = native.chat(&fits).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.submissions().await, 2);
}

/// m213 D2/F4 on the native path: a text-only parts array is encoded as its joined string, so the usage and the
/// committed tokens equal the plain-string request's.
#[tokio::test]
async fn m213_text_only_parts_encode_like_the_joined_string() {
    let native = Native::ready(Case::matching()).await;
    let text = prompt(TEXT);
    let (head, tail) = text.split_at(text.len() / 2);
    let mut parts = body(TEXT, 4, false);
    parts["messages"][0]["content"] =
        json!([{"type": "text", "text": head}, {"type": "text", "text": tail}]);
    let mut usage = Vec::new();
    for request in [body(TEXT, 4, false), parts] {
        let response = native.chat(&request).send().await.unwrap();
        assert_eq!(response.status(), 200);
        let unary: Value = response.json().await.unwrap();
        usage.push((
            unary["usage"].clone(),
            unary["choices"][0]["message"]["content"].clone(),
        ));
    }
    assert_eq!(usage[0], usage[1]);
    assert_eq!(native.submissions().await, 2);
}

/// C3 (m25 D6, F1): `/v1/models` reports the lock's context as `max_model_len` (200,000).
#[tokio::test]
async fn models_report_the_served_context() {
    let native = Native::ready(Case::matching()).await;
    let (status, models) = native.get("/v1/models").await;
    assert_eq!(status, 200, "{models}");
    let cards = models["data"].as_array().unwrap();
    assert_eq!(cards.len(), 1, "{models}");
    assert_eq!(cards[0]["id"], MODEL, "{models}");
    assert_eq!(cards[0]["max_model_len"], CONTEXT, "{models}");
    assert_eq!(shipped_lock()["effective_context_tokens"], CONTEXT);
}

/// server-info design note SI-01..SI-04, SI-06: a `native_cuda` cered reports its own configuration on
/// `GET /get_server_info`: the served model and context, the bound loopback address, no content logging (telemetry
/// is metadata-only), no crash dump and no prompt export, and nothing it cannot vouch for; a vetting client
/// (Chief's `local-sglang` rule) accepts it on a loopback route.
#[tokio::test]
async fn server_info_reports_the_native_configuration() {
    let native = Native::ready(Case::matching()).await;
    let (status, info) = native.get("/get_server_info").await;
    assert_eq!(status, 200, "{info}");
    let port: u16 = native
        .cered
        .base
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let config: toml::Table =
        toml::from_str(&std::fs::read_to_string(&native.config_path).unwrap()).unwrap();
    assert_eq!(info["served_model_name"], MODEL, "{info}");
    assert_eq!(info["context_length"], CONTEXT, "{info}");
    assert_eq!(info["host"], "127.0.0.1", "{info}");
    assert_eq!(info["port"], port, "{info}");
    assert_eq!(
        info["max_running_requests"].as_i64(),
        config["admission"]["max_backend_inflight"].as_integer(),
        "{info}"
    );
    assert_eq!(info["log_requests"], false, "{info}");
    assert_eq!(info["crash_dump_folder"], Value::Null, "{info}");
    for key in support::CLIENT_EXPORT_FLAGS {
        assert!(
            matches!(info.get(key), Some(Value::Null | Value::Bool(false))),
            "{key}: {info}"
        );
    }
    assert_eq!(
        cere_runtime::server_info::EXPORT_KEYS,
        support::CLIENT_EXPORT_FLAGS
    );
    let cere = &info["cere"];
    assert_eq!(cere["unverified_workers"], json!([]), "{info}");
    assert_eq!(cere["telemetry_level"], "metadata", "{info}");
    assert_eq!(cere["raw_content"], false, "{info}");
    assert_eq!(cere["research"], false, "{info}");
    assert_eq!(cere["federation_listen"], Value::Null, "{info}");
    assert_eq!(support::client_refusal(&info, MODEL, false), None, "{info}");
    assert_eq!(
        support::client_refusal(&info, "another-model", false).as_deref(),
        Some("the server serves a different model than the route")
    );
}

/// Reads a held stream up to the content of the toy's held token boundary (`hold_after=2`: "ab" is the 2nd).
async fn read_to_hold(reader: &mut SseReader) {
    loop {
        match reader.next().await {
            Some(Event::Data(v)) if v.pointer("/choices/0/delta/content") == Some(&json!("ab")) => {
                return
            }
            Some(Event::Data(v)) => assert!(v.get("error").is_none(), "{v}"),
            other => panic!("the stream ended before the hold: {other:?}"),
        }
    }
}

fn held() -> Case {
    Case {
        device_id: "toy,hold_after=2".to_owned(),
        ..Case::matching()
    }
}

/// After a cancelled request the session and the admission slot are free: the next request completes.
async fn assert_freed(native: &Native) {
    let response = native.chat(&body(STOP, 8, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    assert_eq!(native.terminal(&rid).await["state"], "completed");
}

/// Cancel through cered while the toy holds mid-stream: `cere_executor_cancel`, the terminal poll ends the item
/// CANCELLED and releases the session, the stream ends with the cancelled error event, the ledger is terminal.
#[tokio::test]
async fn cancel_mid_stream_releases_the_session() {
    let native = Native::ready(held()).await;
    let response = native.chat(&body(TEXT, 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let mut reader = SseReader::new(response);
    assert_eq!(
        reader.next().await.unwrap().data()["cere"]["event"],
        "accepted"
    );
    read_to_hold(&mut reader).await;
    let cancel = http()
        .post(native.cered.url(&format!("/v1/cere/requests/{rid}/cancel")))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    let rest = reader.rest().await;
    assert!(matches!(rest.last(), Some(Event::Done)), "{rest:?}");
    let error = rest[rest.len() - 2].data();
    assert_eq!(error["cere"]["state"], "cancelled", "{error}");
    assert_eq!(error["cere"]["reason"], "cancel_requested", "{error}");
    let status = native.terminal(&rid).await;
    assert_eq!(
        (&status["state"], &status["reason"]),
        (&json!("cancelled"), &json!("cancel_requested"))
    );
    assert_freed(&native).await;
    assert_eq!(native.submissions().await, 2);
}

/// A client that goes away (no `X-Cere-On-Disconnect: continue`) cancels the generation the same way.
#[tokio::test]
async fn disconnect_cancels_the_generation() {
    let native = Native::ready(held()).await;
    let response = native.chat(&body(TEXT, 16, true)).send().await.unwrap();
    let rid = rid(&response);
    let mut reader = SseReader::new(response);
    read_to_hold(&mut reader).await;
    drop(reader);
    let status = native.terminal(&rid).await;
    assert_eq!(
        (&status["state"], &status["reason"]),
        (&json!("cancelled"), &json!("client_disconnected"))
    );
    assert_freed(&native).await;
}

/// `X-Cere-On-Disconnect: continue`: the native stream detaches like an SGLang one and resumes from the buffer.
#[tokio::test]
async fn continue_mode_stream_resumes() {
    let native = Native::ready(Case::matching()).await;
    let response = native
        .chat(&body(TEXT, 6, true))
        .header("X-Cere-On-Disconnect", "continue")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let mut reader = SseReader::new(response);
    assert_eq!(reader.next().await.unwrap().data()["cere"]["seq"], 0);
    drop(reader);
    assert_eq!(native.terminal(&rid).await["state"], "completed");
    let resumed = http()
        .get(
            native
                .cered
                .url(&format!("/v1/cere/requests/{rid}/events?after=0")),
        )
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(resumed.status(), 200);
    let events = SseReader::new(resumed).rest().await;
    let finish = support::check_stream_tail(&events, 0);
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) => v
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str),
            Event::Done => None,
        })
        .collect();
    assert_eq!(text, "ab");
}

/// A ByteLevel `tokenizer.json` in which every toy ID decodes to a distinct piece, `[id]`, unless `pieces` gives
/// it another (added tokens, so two IDs may share a piece). Any dropped, repeated or reordered token changes the
/// text.
fn distinct_tokenizer(pieces: &[(u32, &str)]) -> Vec<u8> {
    let added: Vec<Value> = (0..FIXTURE_VOCAB)
        .map(|id| {
            let content = pieces
                .iter()
                .find(|(i, _)| *i == id)
                .map_or_else(|| format!("[{id}]"), |(_, p)| (*p).to_owned());
            json!({"id": id, "content": content, "special": false})
        })
        .collect();
    serde_json::to_vec(&json!({
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {}},
        "added_tokens": added,
        "decoder": {"type": "ByteLevel"},
    }))
    .unwrap()
}

fn with_tokenizer(tokenizer: Vec<u8>) -> Case {
    Case {
        tokenizer: Some(tokenizer),
        ..Case::matching()
    }
}

/// The concatenated `content` and `reasoning_content` deltas of an SSE body.
fn streamed_text(events: &[Event]) -> (String, String) {
    let mut content = String::new();
    let mut reasoning = String::new();
    for e in events {
        if let Event::Data(v) = e {
            let delta = v.pointer("/choices/0/delta");
            content.push_str(delta.and_then(|d| d["content"].as_str()).unwrap_or(""));
            reasoning.push_str(
                delta
                    .and_then(|d| d["reasoning_content"].as_str())
                    .unwrap_or(""),
            );
        }
    }
    (content, reasoning)
}

/// Every committed token reaches the client exactly once, in order (the PREFILL token and the last DECODE token
/// included): unary text = SSE text = the decode of the toy's committed IDs, for budgets 1, 2, 3, 6 and a stop.
/// A matched stop token (here the distinct tokenizer's non-special 7) is counted but its text dropped (m212 D8).
#[tokio::test]
async fn every_committed_token_is_emitted_once() {
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&[]))).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    for (p, max_tokens) in [(TEXT, 1), (TEXT, 2), (TEXT, 3), (TEXT, 6), (STOP, 8)] {
        let ids = toy_output(p, max_tokens);
        let shown = match ids.last() {
            Some(7 | 6) => &ids[..ids.len() - 1],
            _ => &ids[..],
        };
        let want = vocab.decode(shown);
        let unary: Value = native
            .chat(&body(p, max_tokens as u64, false))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            unary["choices"][0]["message"]["content"], want,
            "unary max_tokens {max_tokens}"
        );
        assert_eq!(unary["usage"]["completion_tokens"], ids.len());
        let response = native
            .chat(&body(p, max_tokens as u64, true))
            .send()
            .await
            .unwrap();
        let events = SseReader::new(response).rest().await;
        check_stream(&events);
        assert_eq!(
            streamed_text(&events),
            (want.clone(), String::new()),
            "SSE max_tokens {max_tokens}"
        );
    }
}

/// Unary reasoning is SGLang's non-stream `detect_and_parse` over the whole text, SSE its streaming parser:
/// the fixture cases where the two differ (`late_block`, `double_open`), generated token by token by the toy.
#[tokio::test]
async fn unary_reasoning_uses_the_non_stream_parser() {
    let cases = fixture("reasoning_cases.json");
    let prompts = [TEXT, (136, 'x')];
    let mut pieces: Vec<(u32, String)> = Vec::new();
    let mut plans = Vec::new();
    for (name, p) in ["late_block", "double_open"].into_iter().zip(prompts) {
        let case = cases["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap()
            .clone();
        let chunks: Vec<String> = text(&case, "text")
            .as_bytes()
            .chunks(5)
            .map(|c| String::from_utf8(c.to_vec()).unwrap())
            .collect();
        let ids = toy_output(p, chunks.len());
        assert_eq!(ids.len(), chunks.len(), "{name}: the toy stopped early");
        for (&id, chunk) in ids.iter().zip(&chunks) {
            assert!(
                pieces.iter().all(|(i, _)| *i != id),
                "{name}: toy ID {id} reused"
            );
            pieces.push((id, chunk.clone()));
        }
        plans.push((name, p, case, chunks));
    }
    let pieces: Vec<(u32, &str)> = pieces.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&pieces))).await;
    for (name, p, case, chunks) in plans {
        let max_tokens = chunks.len() as u64;
        let unary: Value = native
            .chat(&body(p, max_tokens, false))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let message = &unary["choices"][0]["message"];
        let want = expected(&case["non_stream"]);
        assert_eq!(
            message["content"],
            want.normal.as_str(),
            "{name} unary content"
        );
        assert_eq!(
            message["reasoning_content"],
            want.reasoning.as_str(),
            "{name} unary reasoning"
        );

        let mut stream = Stream::default();
        let mut sse = Split::default();
        for chunk in &chunks {
            sse.append(stream.push(chunk));
        }
        sse.append(stream.finish());
        let response = native
            .chat(&body(p, max_tokens, true))
            .send()
            .await
            .unwrap();
        let events = SseReader::new(response).rest().await;
        check_stream(&events);
        assert_eq!(
            streamed_text(&events),
            (sse.normal, sse.reasoning),
            "{name} SSE"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------
// C6 (spec m213): image parts on the native path over the toy and the stub sidecar with feature `image`.

/// The A4 vocabulary of the vision artifact (`text_config.vocab_size`): the toy's next-token modulus.
const A4_VOCAB: u32 = 248_320;
const VISION_START: u32 = 248_053;
const IMAGE_PAD: u32 = 248_056;
const VISION_END: u32 = 248_054;
/// A 256x256 PNG: HF `smart_resize` keeps it (65,536 px = `min_pixels`), grid [1, 16, 16], 64 merged pads.
const SIDE: u32 = 256;
const PADS: usize = 64;

/// The image cases: the vision artifact (A4 config + `vision_config`, `model.visual.*` on the device), the
/// profile's greedy image combination and the stub sidecar's feature `image`, as `images` overrides.
fn image_case(images: Images) -> Case {
    Case {
        images,
        ..Case::matching()
    }
}

fn all_images() -> Images {
    Images {
        artifact: true,
        combo: true,
        sidecar: true,
        bad_offset: false,
    }
}

/// A user message of the sentinel text then one PNG image.
fn image_body(max_tokens: u64) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": SENTINEL},
            {"type": "image_url", "image_url": {"url": png_url(SIDE, SIDE)}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": false,
    })
}

/// The stub's expanded ids for [`image_body`]: `[1]`, the text bytes, then `vision_start`, the image's pads and
/// `vision_end` (the `<|image_pad|>` expansion D8 moves into the sidecar).
fn image_prompt_ids() -> Vec<u32> {
    let mut ids = vec![1];
    ids.extend(SENTINEL.bytes().map(u32::from));
    ids.push(VISION_START);
    ids.extend(std::iter::repeat(IMAGE_PAD).take(PADS));
    ids.push(VISION_END);
    ids
}

/// `api.max_request_body_bytes` of the native config the harness starts from.
fn max_request_body_bytes() -> u64 {
    let text = std::fs::read_to_string(root().join("configs/runtime/native-g0.toml")).unwrap();
    let config: toml::Table = toml::from_str(&text).unwrap();
    config["api"]["max_request_body_bytes"]
        .as_integer()
        .unwrap()
        .try_into()
        .unwrap()
}

/// C6: an image request reaches `cere_work` with its spans and pixel bytes. The toy's `plan_images` checks every
/// span against the prompt (`vision_start` before `token_offset`, exactly t·h·w/4 pads, no stray pad), the
/// `pixel_bytes` against the grid, `pad_id` non-zero and `pixel_sha256` non-zero (VisionLong C1: `cered` fills the
/// SHA-256 of the pixel bytes); so a completed request proves all of them, and a stub reporting every offset
/// one row late fails in the executor. `usage.prompt_tokens` counts the expanded ids, and `cered` started the
/// sidecar with `--max-request-bytes` = `api.max_request_body_bytes` + 64 KiB (D18). The grammar process (same stub,
/// same log, no image frames so the default limit) starts beside it (load-speed design note LS-04), so either
/// startup line may come first.
#[tokio::test]
async fn images_reach_the_executor_with_spans_and_bytes() {
    let native = Native::ready(image_case(all_images())).await;
    let want = format!("max_request_bytes {}", max_request_body_bytes() + 65_536);
    assert!(
        native.sidecar_log().contains(&want),
        "{:?}",
        native.sidecar_log()
    );
    let ids = image_prompt_ids();
    let max_tokens = 4;
    let mut want = toy_tokens(VISION_END, ids.len() as u64, max_tokens, A4_VOCAB);
    if let Some(i) = want.iter().position(|id| [6, 7].contains(id)) {
        want.truncate(i + 1);
    }
    let response = native
        .chat(&image_body(max_tokens as u64))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"], ids.len(), "{body}");
    assert_eq!(body["usage"]["completion_tokens"], want.len(), "{body}");
    assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
    assert_eq!(native.submissions().await, 1);

    let bad = Native::ready(image_case(Images {
        bad_offset: true,
        ..all_images()
    }))
    .await;
    let response = bad.chat(&image_body(4)).send().await.unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    // The executor's ARGUMENT refusal of the submit is the native backend's 500, relayed as a 502 backend error.
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["error"]["code"], "backend_error", "{body}");
    assert_eq!(body["cere"]["state"], "failed", "{body}");
    assert_eq!(bad.submissions().await, 1);
}

/// C6: a profile with the image combination over a text-only model (the toy reports `native_image_input` 0)
/// refuses an image part with 400 before the sidecar call: no `encode_chat` reaches the stub and nothing is
/// submitted; text requests still serve. Without feature `image` the sidecar keeps its default frame limit.
#[tokio::test]
async fn images_without_native_image_input_are_refused_before_the_sidecar() {
    let native = Native::ready(image_case(Images {
        artifact: false,
        ..all_images()
    }))
    .await;
    let before = native.sidecar_log();
    let response = native.chat(&image_body(4)).send().await.unwrap();
    let status = response.status();
    let error: Value = response.json().await.unwrap();
    assert_eq!(status, 400, "{error}");
    assert_eq!(error["error"]["code"], "unsupported_parameter", "{error}");
    assert_eq!(error["error"]["param"], "messages[0].content[1]", "{error}");
    assert_eq!(native.sidecar_log(), before, "the sidecar saw the request");
    assert_eq!(native.submissions().await, 0);
    let response = native.chat(&body(TEXT, 4, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.submissions().await, 1);

    let text = Native::ready(Case::matching()).await;
    assert_eq!(
        text.sidecar_log().first().map(String::as_str),
        Some("max_request_bytes 4194304")
    );
}

// ---------------------------------------------------------------------------------------------------------
// C7 (spec m24): tool calls and stop strings end to end over the toy and the stub sidecar v2.

/// The distinct tokenizer with the toy's stop IDs 7 and 6 special, as the model's are: with tools the decoder
/// keeps specials (D10), and a special stop token still never reaches the text.
fn tool_tokenizer(pieces: &[(u32, &str)]) -> Vec<u8> {
    let mut all = vec![(7, "<|im_end|>"), (6, "<|endoftext|>")];
    all.extend_from_slice(pieces);
    let mut t: Value = serde_json::from_slice(&distinct_tokenizer(&all)).unwrap();
    for id in [6, 7] {
        t["added_tokens"][id]["special"] = json!(true);
    }
    serde_json::to_vec(&t).unwrap()
}

fn tools() -> Value {
    json!([{"type": "function", "function": {
        "name": "get_weather",
        "description": "The weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}},
    }}])
}

/// A request with [`tools`] whose rendered prompt (the stub's `[2]` first) is `p`.
fn tool_body((len, last): (usize, char), max_tokens: u64, stream: bool) -> Value {
    let mut b = body((len - 1, last), max_tokens, stream);
    b["tools"] = tools();
    b
}

/// The one token before the toy's stop under `STOP` (2240, then 7) carries a whole call.
const CALL_TEXT: &str = "Let me check.<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
const CALL_ARGUMENTS: &str = r#"{"city": "Paris", "days": 3}"#;

/// `call_` + 24 lowercase hex digits (`serving_chat.py:1970-1989`).
fn assert_call_id(id: &Value) {
    let id = id.as_str().unwrap_or_else(|| panic!("call id {id}"));
    let hex = id.strip_prefix("call_").unwrap_or_else(|| panic!("{id}"));
    assert!(
        hex.len() == 24 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
        "{id}"
    );
}

async fn create_session(native: &Native) -> String {
    let r = http()
        .post(native.cered.url("/v1/cere/sessions"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);
    let body: Value = r.json().await.unwrap();
    body["session_id"].as_str().unwrap().to_owned()
}

fn turn(native: &Native, session: &str, revision: u64, body: &Value) -> reqwest::RequestBuilder {
    native
        .chat(body)
        .header("X-Cere-Session-Id", session)
        .header("X-Cere-Expected-Revision", revision.to_string())
}

/// The SSE choices of a stream: the data events with a choice, in order.
fn stream_choices(events: &[Event]) -> Vec<Value> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) => v.pointer("/choices/0").cloned(),
            Event::Done => None,
        })
        .collect()
}

/// The streamed `tool_calls` items, flattened, as `(id, index, name, arguments)`.
fn stream_calls(choices: &[Value]) -> Vec<(Value, Value, Value, Value)> {
    choices
        .iter()
        .filter_map(|c| c["delta"]["tool_calls"].as_array())
        .flatten()
        .map(|c| {
            assert_eq!(c["type"], "function", "{c}");
            (
                c["id"].clone(),
                c["index"].clone(),
                c["function"]["name"].clone(),
                c["function"]["arguments"].clone(),
            )
        })
        .collect()
}

/// The assistant message a client assembles from a stream (the OpenAI SDK idiom): content concatenated, calls
/// by `index`, `id`/`name` from the first item, `arguments` concatenated.
fn assembled(choices: &[Value]) -> Value {
    let content: String = choices
        .iter()
        .filter_map(|c| c["delta"]["content"].as_str())
        .collect();
    let mut calls: Vec<Value> = Vec::new();
    for (id, index, name, arguments) in stream_calls(choices) {
        let i = index.as_u64().unwrap() as usize;
        if calls.len() <= i {
            calls.push(
                json!({"id": id, "type": "function", "function": {"name": name, "arguments": ""}}),
            );
        }
        let args = calls[i]["function"]["arguments"]
            .as_str()
            .unwrap()
            .to_owned();
        calls[i]["function"]["arguments"] = json!(args + arguments.as_str().unwrap());
    }
    json!({"role": "assistant", "content": content, "tool_calls": calls})
}

/// Unary and SSE tool calls (D10): ids `call_` + 24 hex, `index`, the stream's id and name on a call's first item
/// then null, finish `tool_calls` (unary `matched_stop` null, stream the stop token); the next turn echoes the
/// response message verbatim (unary) or as assembled (stream) with the tool's answer and continues the session.
#[tokio::test]
async fn tool_calls_unary_and_stream_continue_the_session() {
    let native = Native::ready(with_tokenizer(tool_tokenizer(&[(2240, CALL_TEXT)]))).await;
    for stream in [false, true] {
        let session = create_session(&native).await;
        let first = tool_body(STOP, 8, stream);
        let response = turn(&native, &session, 0, &first).send().await.unwrap();
        assert_eq!(response.status(), 200, "stream {stream}");
        let rid = rid(&response);
        let message = if stream {
            let events = SseReader::new(response).rest().await;
            let finish = check_stream(&events);
            assert_eq!(finish["cere"]["revision"], 1, "{finish}");
            let choices = stream_choices(&events);
            let calls = stream_calls(&choices);
            assert_eq!(calls.len(), 5, "{calls:?}");
            assert_call_id(&calls[0].0);
            assert_eq!(
                (&calls[0].1, &calls[0].2, &calls[0].3),
                (&json!(0), &json!("get_weather"), &json!(""))
            );
            let rest: Vec<_> = calls[1..]
                .iter()
                .map(|c| (c.0.clone(), c.1.clone(), c.2.clone()))
                .collect();
            assert_eq!(rest, vec![(Value::Null, json!(0), Value::Null); 4]);
            let fragments: Vec<_> = calls.iter().map(|c| c.3.as_str().unwrap()).collect();
            assert_eq!(
                fragments,
                ["", "{", r#""city": "Paris""#, r#", "days": 3"#, "}"]
            );
            let last = choices.last().unwrap();
            assert_eq!(
                (&last["finish_reason"], &last["matched_stop"]),
                (&json!("tool_calls"), &json!(7)),
                "{last}"
            );
            let message = assembled(&choices);
            assert_eq!(message["content"], "Let me check.");
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                CALL_ARGUMENTS
            );
            message
        } else {
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["cere"]["revision"], 1, "{body}");
            let choice = &body["choices"][0];
            assert_eq!(
                (&choice["finish_reason"], &choice["matched_stop"]),
                (&json!("tool_calls"), &Value::Null),
                "{body}"
            );
            assert_eq!(body["usage"]["completion_tokens"], 2);
            let message = choice["message"].clone();
            assert_eq!(message["content"], "Let me check.");
            assert_eq!(message["reasoning_content"], Value::Null);
            let call = &message["tool_calls"][0];
            assert_call_id(&call["id"]);
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
            assert_eq!(
                call,
                &json!({"id": call["id"], "index": 0, "type": "function",
                        "function": {"name": "get_weather", "arguments": CALL_ARGUMENTS}})
            );
            message
        };
        assert_eq!(native.terminal(&rid).await["state"], "completed");
        let id = message["tool_calls"][0]["id"].clone();
        let mut second = tool_body(STOP, 4, false);
        let messages = second["messages"].as_array_mut().unwrap();
        messages.push(message);
        messages.push(json!({"role": "tool", "tool_call_id": id, "content": "sunny"}));
        let response = turn(&native, &session, 1, &second).send().await.unwrap();
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["cere"]["revision"], 2, "stream {stream}: {body}");
    }
}

/// D9: the unary parse falls back to the whole text, no calls, only on the reference's conversion exception
/// (`convert_error`); any other sidecar failure (a typed refusal, the sidecar gone) fails the request as the stream
/// path does, and nothing is committed as a successful turn.
#[tokio::test]
async fn unary_tool_fallback_is_for_convert_error_only() {
    for value in ["boom", "unavailable", "die"] {
        let text = format!(
            "Let me check.<tool_call>\n<function=get_weather>\n<parameter=city>\n{value}\n</parameter>\n</function>\n</tool_call>"
        );
        let native = Native::ready(with_tokenizer(tool_tokenizer(&[(2240, &text)]))).await;
        let response = native
            .chat(&tool_body(STOP, 8, false))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let rid = rid(&response);
        let body: Value = response.json().await.unwrap();
        if value == "boom" {
            assert_eq!(status, 200, "{value}: {body}");
            let message = &body["choices"][0]["message"];
            assert_eq!(message["content"], text.as_str(), "{body}");
            assert_eq!(message["tool_calls"], Value::Null, "{body}");
            assert_eq!(native.terminal(&rid).await["state"], "completed");
        } else {
            assert_eq!(status, 502, "{value}: {body}");
            assert_eq!(body["error"]["code"], "backend_error", "{value}: {body}");
            assert_eq!(native.terminal(&rid).await["state"], "failed", "{value}");
        }
    }
}

/// A `length` finish mid-call: the stream committed the closed parameters only, so the session holds unbalanced
/// `arguments` and the echo of that message is refused (400); the unary parse closes the call at the end.
#[tokio::test]
async fn length_mid_call_commits_unbalanced_arguments() {
    let ids = toy_output(TEXT, 3);
    let pieces = [
        (ids[0], "<tool_call>\n<function=f>\n<parameter=p>\n"),
        (ids[1], "v\n</parameter>\n<parameter=q>"),
        (ids[2], "\nw"),
    ];
    let native = Native::ready(with_tokenizer(tool_tokenizer(&pieces))).await;
    let body: Value = native
        .chat(&tool_body(TEXT, 3, false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "length", "{body}");
    assert_eq!(choice["message"]["content"], "");
    assert_eq!(
        choice["message"]["tool_calls"][0]["function"],
        json!({"name": "f", "arguments": r#"{"p": "v", "q": "w"}"#})
    );

    let session = create_session(&native).await;
    let first = tool_body(TEXT, 3, true);
    let response = turn(&native, &session, 0, &first).send().await.unwrap();
    let events = SseReader::new(response).rest().await;
    let finish = check_stream(&events);
    assert_eq!(finish["choices"][0]["finish_reason"], "length", "{finish}");
    let message = assembled(&stream_choices(&events));
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        r#"{"p": "v""#
    );
    let mut second = tool_body(TEXT, 3, false);
    let messages = second["messages"].as_array_mut().unwrap();
    messages.push(message);
    messages.push(json!({"role": "user", "content": "go on"}));
    let response = turn(&native, &session, 1, &second).send().await.unwrap();
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert_eq!(
        error["error"]["param"], "messages[1].tool_calls[0].function.arguments",
        "{error}"
    );
}

/// Stop strings (D11): the text is trimmed at the match, `matched_stop` is the string, `completion_tokens` counts
/// through the matching token, the stream holds a step whose window ends with a stop prefix and emits the trimmed
/// rest at the end, the generation is cancelled at the match and the ledger completes with the session released.
/// `TEXT` decodes to `[2235][2][780]…` under the distinct tokenizer.
#[tokio::test]
async fn stop_strings_trim_and_finish() {
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&[]))).await;
    let mut submissions = 0;
    for (stop, text) in [("[2]", "[2235]"), ("5][2", "[223")] {
        let mut unary = body(TEXT, 16, false);
        unary["stop"] = json!(["zzz", stop]);
        let response = native.chat(&unary).send().await.unwrap();
        assert_eq!(response.status(), 200, "{stop}");
        let unary_rid = rid(&response);
        let body: Value = response.json().await.unwrap();
        let choice = &body["choices"][0];
        assert_eq!(choice["message"]["content"], text, "{body}");
        assert_eq!(
            (&choice["finish_reason"], &choice["matched_stop"]),
            (&json!("stop"), &json!(stop))
        );
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(native.terminal(&unary_rid).await["state"], "completed");

        let mut stream = unary.clone();
        stream["stream"] = json!(true);
        stream["stream_options"] = json!({"include_usage": true});
        let response = native.chat(&stream).send().await.unwrap();
        let rid = rid(&response);
        let events = SseReader::new(response).rest().await;
        let finish = check_stream(&events);
        assert_eq!(streamed_text(&events).0, text, "{stop}");
        assert_eq!(
            (
                &finish["choices"][0]["finish_reason"],
                &finish["choices"][0]["matched_stop"]
            ),
            (&json!("stop"), &json!(stop))
        );
        let usage = events.iter().rev().find_map(|e| match e {
            Event::Data(v) if !v["usage"].is_null() => Some(v["usage"].clone()),
            _ => None,
        });
        assert_eq!(usage.unwrap()["completion_tokens"], 2);
        assert_eq!(native.terminal(&rid).await["state"], "completed");
        submissions += 2;
        assert_eq!(native.submissions().await, submissions);
    }
    assert_freed(&native).await;
}

/// `tool_choice: "none"` validates the tools but neither renders nor parses them (the markup is content); a
/// schema SGLang rejects is a 400 `invalid_value` at its path for both choices, before any submission.
#[tokio::test]
async fn tool_choice_none_and_invalid_schemas() {
    let native = Native::ready(with_tokenizer(tool_tokenizer(&[(2240, CALL_TEXT)]))).await;
    let mut none = body(STOP, 8, false);
    none["tools"] = tools();
    none["tool_choice"] = json!("none");
    let body: Value = native
        .chat(&none)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let choice = &body["choices"][0];
    assert_eq!(choice["message"]["content"], CALL_TEXT, "{body}");
    assert_eq!(choice["message"]["tool_calls"], Value::Null);
    assert_eq!(
        (&choice["finish_reason"], &choice["matched_stop"]),
        (&json!("stop"), &json!(7))
    );
    assert_eq!(native.submissions().await, 1);
    for choice in ["auto", "none"] {
        let mut bad = none.clone();
        bad["tool_choice"] = json!(choice);
        bad["tools"][0]["function"]["parameters"] = json!({"type": "invalid"});
        let response = native.chat(&bad).send().await.unwrap();
        assert_eq!(response.status(), 400, "{choice}");
        let error: Value = response.json().await.unwrap();
        assert_eq!(
            (&error["error"]["code"], &error["error"]["param"]),
            (
                &json!("invalid_value"),
                &json!("tools[0].function.parameters")
            ),
            "{error}"
        );
    }
    assert_eq!(native.submissions().await, 1);
}

/// Whether `native` has the shape of the oracle's `want` (`api_shapes.json`, SGLang's own response models):
/// every fixture key present with the same JSON type, recursively; protocol values (`object`, `index`, `role`,
/// `type`, `logprobs`, `finish_reason`, a `null`) equal; text the model generates (`content`, `arguments`,
/// `name`, `matched_stop`) by type only. The fixture's volatile pointers are skipped whole (`usage` depends on
/// the reference server's cache-report setting). A native key the fixture lacks must be `null` (the pre-M2.4
/// native delta carries all four fields, frozen by C11 (ii)); top-level `cere` is cered's envelope on both paths.
fn conform(native: &Value, want: &Value, at: &str, volatile: &[String], out: &mut Vec<String>) {
    const VALUES: [&str; 6] = [
        "object",
        "index",
        "role",
        "type",
        "logprobs",
        "finish_reason",
    ];
    let key = at.rsplit('/').next().unwrap_or("");
    let type_of = |v: &Value| match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    if volatile.iter().any(|p| p == at) {
        return;
    }
    if type_of(native) != type_of(want) {
        out.push(format!("{at}: {native} is not like {want}"));
        return;
    }
    match (native, want) {
        (Value::Object(n), Value::Object(w)) => {
            for (k, v) in w {
                let path = format!("{at}/{k}");
                match n.get(k) {
                    Some(nv) => conform(nv, v, &path, volatile, out),
                    None => out.push(format!("{path}: missing")),
                }
            }
            for (k, v) in n {
                if !(w.contains_key(k) || v.is_null() || (at.is_empty() && k == "cere")) {
                    out.push(format!("{at}/{k}: extra non-null {v}"));
                }
            }
        }
        (Value::Array(n), Value::Array(w)) => {
            if n.is_empty() != w.is_empty() {
                out.push(format!("{at}: {} items, fixture {}", n.len(), w.len()));
            }
            if let Some(first) = w.first() {
                for (i, item) in n.iter().enumerate() {
                    let fixture_item = w.get(i).unwrap_or(first);
                    conform(item, fixture_item, &format!("{at}/{i}"), volatile, out);
                }
            }
        }
        _ if VALUES.contains(&key) || want.is_null() => {
            if native != want {
                out.push(format!("{at}: {native} != {want}"));
            }
        }
        _ => {}
    }
}

/// The kind of a stream chunk: the role chunk, content, a call's first item, an arguments item, the finish.
fn chunk_kind(chunk: &Value) -> &'static str {
    let choice = &chunk["choices"][0];
    let delta = &choice["delta"];
    if !choice["finish_reason"].is_null() {
        "finish"
    } else if delta["role"] == "assistant" {
        "role"
    } else if let Some(call) = delta["tool_calls"].as_array().and_then(|c| c.first()) {
        if call["function"]["name"].is_null() {
            "arguments"
        } else {
            "call"
        }
    } else {
        "content"
    }
}

/// The mismatches of one native body (unary) or chunk list (stream) against fixture case `name`. Stream chunks
/// are matched by kind to the case's first chunk of that kind, else (a kind the case's chunking never shows, e.g.
/// arguments items of a call the reference received whole) to the first of any stream case; every kind of the
/// case must occur. cered's `accepted` event (no choice) is skipped.
fn shape_mismatches(shapes: &Value, name: &str, native: &[Value]) -> Vec<String> {
    let case = shapes["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no api_shapes case {name}"));
    let volatile: Vec<String> = serde_json::from_value(case["volatile"].clone()).unwrap();
    let mut out = Vec::new();
    if case["kind"] == "unary" {
        let volatile: Vec<String> = volatile
            .iter()
            .filter_map(|p| p.strip_prefix("/body").map(str::to_owned))
            .collect();
        conform(&native[0], &case["body"], "", &volatile, &mut out);
        return out;
    }
    let streams: Vec<&Value> = std::iter::once(case)
        .chain(
            shapes["cases"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["kind"] == "stream"),
        )
        .collect();
    let mut kinds_seen = BTreeSet::new();
    for (i, chunk) in native.iter().enumerate() {
        if chunk["choices"].as_array().is_some_and(Vec::is_empty) {
            continue;
        }
        let kind = chunk_kind(chunk);
        kinds_seen.insert(kind);
        let Some((source, j, want)) = streams.iter().find_map(|c| {
            c["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .find(|(_, w)| chunk_kind(w) == kind)
                .map(|(j, w)| (c, j, w))
        }) else {
            out.push(format!("chunk {i}: kind {kind} in no fixture stream"));
            continue;
        };
        let prefix = format!("/chunks/{j}/");
        let volatile: Vec<String> =
            serde_json::from_value::<Vec<String>>(source["volatile"].clone())
                .unwrap()
                .iter()
                .filter_map(|p| p.strip_prefix(&prefix).map(|rest| format!("/{rest}")))
                .collect();
        let mut found = Vec::new();
        conform(chunk, want, "", &volatile, &mut found);
        out.extend(
            found
                .into_iter()
                .map(|m| format!("chunk {i} ({kind}): {m}")),
        );
    }
    let fixture_kinds: BTreeSet<_> = case["chunks"]
        .as_array()
        .unwrap()
        .iter()
        .map(chunk_kind)
        .collect();
    if !fixture_kinds.is_subset(&kinds_seen) {
        out.push(format!(
            "chunk kinds {kinds_seen:?}, fixture {fixture_kinds:?}"
        ));
    }
    out
}

/// C7 shapes (spec m24 D10): native unary bodies and SSE chunks of a plain stop, a stop string, a length finish,
/// tool calls, a length mid-call, tools without a call and `tool_choice: "none"` conform to `api_shapes.json`.
#[tokio::test]
async fn bodies_match_api_shapes() {
    let shapes = fixture("api_shapes.json");
    assert_eq!(shapes["schema"], "cere-api-shapes-v1");
    let calls = Native::ready(with_tokenizer(tool_tokenizer(&[(2240, CALL_TEXT)]))).await;
    let ids = toy_output(TEXT, 3);
    let mid_call = Native::ready(with_tokenizer(tool_tokenizer(&[
        (ids[0], "<tool_call>\n<function=f>\n<parameter=p>\n"),
        (ids[1], "v\n</parameter>\n<parameter=q>"),
        (ids[2], "\nw"),
    ])))
    .await;
    let mut stop_string = body(TEXT, 16, false);
    stop_string["stop"] = json!(["[2]"]);
    let mut none = body(STOP, 8, false);
    none["tools"] = tools();
    none["tool_choice"] = json!("none");
    let plans = [
        ("plain", &calls, body(STOP, 8, false)),
        ("stop_string", &calls, stop_string),
        ("length", &calls, body(TEXT, 3, false)),
        ("tool_calls", &calls, tool_body(STOP, 8, false)),
        (
            "tool_calls_length_mid_call",
            &mid_call,
            tool_body(TEXT, 3, false),
        ),
        ("tools_no_call", &mid_call, tool_body(STOP, 8, false)),
        ("tool_choice_none", &calls, none),
    ];
    let mut failures = Vec::new();
    for (name, native, request) in plans {
        let unary: Value = native
            .chat(&request)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        for m in shape_mismatches(&shapes, &format!("{name}_unary"), &[unary]) {
            failures.push(format!("{name}_unary {m}"));
        }
        let mut stream = request.clone();
        stream["stream"] = json!(true);
        let response = native.chat(&stream).send().await.unwrap();
        let chunks: Vec<Value> = SseReader::new(response)
            .rest()
            .await
            .into_iter()
            .filter_map(|e| match e {
                Event::Data(v) => Some(v),
                Event::Done => None,
            })
            .collect();
        for m in shape_mismatches(&shapes, &format!("{name}_stream"), &chunks) {
            failures.push(format!("{name}_stream {m}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The C11 (ii) requests: tool-free and stop-free, unary and SSE, a `length` and a stop-token finish, with and
/// without the usage chunk, over the fixture tokenizer and the distinct one (every toy token visible).
fn c11_requests() -> Vec<(&'static str, Value)> {
    let mut stream_usage = body(TEXT, 6, true);
    stream_usage["stream_options"] = json!({"include_usage": true});
    vec![
        ("unary_length", body(TEXT, 4, false)),
        ("unary_stop_token", body(STOP, 8, false)),
        ("stream_length_usage", stream_usage),
        ("stream_stop_token", body(STOP, 8, true)),
    ]
}

/// The JSON of one body with the per-request and per-run fields masked: `id`, `created`, the `cere` request id
/// and timing fields, and the scratch deployment's identities (`config_hash`, `instance_id`, `lock_sha256`: the
/// config carries a free port, the lock the locally built toy library). The body must be canonical (`serde_json`
/// compact, sorted keys), so equal masked text means byte-equal bodies apart from the masked values.
fn masked(raw: &str) -> String {
    let mut v: Value = serde_json::from_str(raw).unwrap_or_else(|e| panic!("{e}: {raw}"));
    assert_eq!(serde_json::to_string(&v).unwrap(), raw, "not canonical");
    if v.get("id").is_some() {
        v["id"] = json!("<id>");
    }
    if v.get("created").is_some() {
        v["created"] = json!(0);
    }
    if let Some(cere) = v.get_mut("cere").and_then(Value::as_object_mut) {
        for key in [
            "request_id",
            "config_hash",
            "instance_id",
            "lock_sha256",
            "since_unix_ms",
            "ttft_ms",
            "e2e_ms",
            "queue_ms",
        ] {
            if cere.contains_key(key) {
                cere.insert(key.to_owned(), json!("<masked>"));
            }
        }
    }
    serde_json::to_string(&v).unwrap()
}

/// The masked bodies of the C11 requests: one line per unary body, one per SSE event.
async fn c11_bodies(native: &Native) -> String {
    let mut out = String::new();
    for (name, request) in c11_requests() {
        let response = native.chat(&request).send().await.unwrap();
        assert_eq!(response.status(), 200, "{name}");
        let raw = response.text().await.unwrap();
        out.push_str(&format!("# {name}\n"));
        if request["stream"] == true {
            for event in raw.split("\n\n").filter(|e| !e.is_empty()) {
                if event == "data: [DONE]" {
                    out.push_str("[DONE]\n");
                    continue;
                }
                let (id, data) = event
                    .split_once("\ndata: ")
                    .unwrap_or_else(|| panic!("not an id + data event: {event:?}"));
                out.push_str(id);
                out.push(' ');
                if data == "[DONE]" {
                    out.push_str("[DONE]\n");
                } else {
                    out.push_str(&masked(data));
                    out.push('\n');
                }
            }
        } else {
            out.push_str(&masked(&raw));
            out.push('\n');
        }
    }
    out
}

/// C11 (ii) regression guard (spec m24): native answers to tool-free, stop-free requests are byte-equal to the
/// bodies recorded before M2.4 (`85a123e` code), apart from the fields [`masked`] names.
#[tokio::test]
async fn tool_free_bodies_equal_pre_m24_goldens() {
    let mut got = Vec::new();
    for tokenizer in ["fixture", "distinct"] {
        let case = match tokenizer {
            "fixture" => Case::matching(),
            _ => with_tokenizer(distinct_tokenizer(&[])),
        };
        let native = Native::ready(case).await;
        got.push(format!("## {tokenizer}\n{}", c11_bodies(&native).await));
    }
    let got = got.concat();
    assert!(got == C11_GOLDEN, "C11 (ii) bodies differ:\n{got}");
}

/// Recorded on `bl/m24-native` @ `01a2713` (pre-M2.4 code: only `tasks/*` changed since `85a123e`).
/// `profile_id` moved to `-v2` with the 73,728 context (m22c D6), to `-v3` with 200,000 (m25 D6), to `-v4` with the
/// rows (m26 D10), to `-v5` with M2.12's capability fields (ledger P1) and to `-v7` with the sampled rows (m27 D11);
/// nothing else changed.
/// M2.12 D8 (integration-owner request, fold-m212): a matched stop token's text is dropped whether or not it is
/// special (SGLang's `output[:-1]`), so the distinct tokenizer's `stop` bodies lose the non-special `[7]`; the
/// fixture tokenizer's stop token decodes to nothing and its bodies are unchanged.
const C11_GOLDEN: &str = r#"## fixture
# unary_length
{"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null,"message":{"content":"ab","reasoning_content":null,"role":"assistant","tool_calls":null}}],"created":0,"id":"<id>","metadata":{"weight_version":"default"},"model":"Qwen3.8-Flash-Next","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":720}}
# unary_stop_token
{"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"matched_stop":7,"message":{"content":"","reasoning_content":null,"role":"assistant","tool_calls":null}}],"created":0,"id":"<id>","metadata":{"weight_version":"default"},"model":"Qwen3.8-Flash-Next","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":566,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":568}}
# stream_length_usage
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu0"},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"ab","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":3,"state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 4 {"cere":{"committed_tokens":null,"seq":4},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":{"completion_tokens":6,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":722}}
[DONE]
# stream_stop_token
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu0"},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":2,"state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null,"matched_stop":7}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
[DONE]
## distinct
# unary_length
{"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null,"message":{"content":"[2235][2][780][24899]","reasoning_content":null,"role":"assistant","tool_calls":null}}],"created":0,"id":"<id>","metadata":{"weight_version":"default"},"model":"Qwen3.8-Flash-Next","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":720}}
# unary_stop_token
{"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"matched_stop":7,"message":{"content":"[2240]","reasoning_content":null,"role":"assistant","tool_calls":null}}],"created":0,"id":"<id>","metadata":{"weight_version":"default"},"model":"Qwen3.8-Flash-Next","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":566,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":568}}
# stream_length_usage
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu0"},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"[2235]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"committed_tokens":null,"seq":3},"choices":[{"delta":{"content":"[2]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 4 {"cere":{"committed_tokens":null,"seq":4},"choices":[{"delta":{"content":"[780]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 5 {"cere":{"committed_tokens":null,"seq":5},"choices":[{"delta":{"content":"[24899]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 6 {"cere":{"committed_tokens":null,"seq":6},"choices":[{"delta":{"content":"[2589]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 7 {"cere":{"committed_tokens":null,"seq":7},"choices":[{"delta":{"content":"[10980]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 8 {"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":8,"state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 9 {"cere":{"committed_tokens":null,"seq":9},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":{"completion_tokens":6,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":722}}
[DONE]
# stream_stop_token
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu0"},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"[2240]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":3,"state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null,"matched_stop":7}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
[DONE]
"#;

// ---------------------------------------------------------------------------------------------------------
// M2.7 C6/C7 (spec m27 D1, D4): what cered submits, read from the toy's `record=` lines (numbers only).

/// One executed step as the toy recorded it: `key` -> value (`session`, `count`, `sampling`=`null` or the
/// sampled fields, `logprobs`, `token`).
type Step = BTreeMap<String, String>;

/// The matching case with the toy recording every step to a file in `t`.
fn recording(t: &Tmp) -> (Case, PathBuf) {
    let path = t.join("steps.record");
    let case = Case {
        device_id: format!("toy,record={}", path.display()),
        ..Case::matching()
    };
    (case, path)
}

fn steps(path: &Path) -> Vec<Step> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|line| {
            line.split(' ')
                .map(|kv| {
                    let (k, v) = kv.split_once('=').unwrap();
                    (k.to_owned(), v.to_owned())
                })
                .collect()
        })
        .collect()
}

/// `body(TEXT, max_tokens, false)` with `fields` set (`temperature` replaced when given).
fn sampled(max_tokens: u64, fields: Value) -> Value {
    let mut b = body(TEXT, max_tokens, false);
    for (k, v) in fields.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

/// Sends a unary `body` (200 expected) and returns the steps it recorded, grouped by session in order.
async fn submitted(native: &Native, path: &Path, body: &Value) -> Vec<Vec<Step>> {
    let before = steps(path).len();
    let response = native.chat(body).send().await.unwrap();
    assert_eq!(response.status(), 200, "{body}");
    let _: Value = response.json().await.unwrap();
    by_session(&steps(path)[before..])
}

/// The steps of each session in order, sessions in order of their first step (concurrent sessions' steps
/// interleave: M2.6 rounds step every decode-ready session once).
fn by_session(steps: &[Step]) -> Vec<Vec<Step>> {
    let mut out: Vec<Vec<Step>> = Vec::new();
    for step in steps {
        match out.iter_mut().find(|s| s[0]["session"] == step["session"]) {
            Some(session) => session.push(step.clone()),
            None => out.push(vec![step.clone()]),
        }
    }
    out
}

/// The request's single session: exactly one PREFILL item (the first step feeds the whole prompt, every later
/// step one token, D5), `max_tokens` steps.
fn one_prefill(sessions: &[Vec<Step>], max_tokens: usize) -> &[Step] {
    assert_eq!(sessions.len(), 1, "{sessions:?}");
    let s = &sessions[0];
    assert_eq!(s.len(), max_tokens, "{s:?}");
    assert_eq!(s[0]["count"], TEXT.0.to_string());
    assert!(s[1..].iter().all(|step| step["count"] == "1"), "{s:?}");
    s
}

/// C6 (D1): `sampling` is null iff the request is greedy (temperature < 1e-6 or top_k 1) with neutral processors;
/// otherwise every item (PREFILL and DECODE) carries the params, temperature 0 when greedy, `top_k` -1 or above
/// the vocabulary as the vocabulary, absent fields at their defaults (D10: the fixture's `generation_config.json`
/// top_k 20 / top_p 0.95, else SGLang's); one PREFILL item per request.
#[tokio::test]
async fn c6_sampling_is_null_iff_greedy_and_neutral() {
    let t = Tmp::new("c6-record");
    let (case, path) = recording(&t);
    let native = Native::ready(case).await;
    let vocab = FIXTURE_VOCAB.to_string();
    let greedy = [
        json!({"temperature": 0}),
        json!({"temperature": 1e-9}),
        json!({"temperature": 0.8, "top_k": 1}),
        json!({"temperature": 0, "top_p": 0.5, "min_p": 0.25, "top_k": 3, "seed": 9,
               "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "logit_bias": {}}),
    ];
    for fields in greedy {
        let sessions = submitted(&native, &path, &sampled(3, fields.clone())).await;
        let s = one_prefill(&sessions, 3);
        assert!(
            s.iter().all(|step| step["sampling"] == "null"),
            "{fields}: {s:?}"
        );
        let tokens: Vec<String> = s.iter().map(|step| step["token"].clone()).collect();
        let expected: Vec<String> = toy_output(TEXT, 3).iter().map(u32::to_string).collect();
        assert_eq!(tokens, expected, "{fields}");
    }
    let defaults = [
        ("top_k", "20"),
        ("top_p", "0.949999988"),
        ("min_p", "0"),
        ("frequency", "0"),
        ("presence", "0"),
        ("repetition", "1"),
        ("logit_bias", "0"),
    ];
    let params = [
        (
            json!({"temperature": 0, "frequency_penalty": 0.5}),
            vec![("temperature", "0"), ("frequency", "0.5")],
        ),
        (
            json!({"temperature": 1e-9, "presence_penalty": -1.5}),
            vec![("temperature", "0"), ("presence", "-1.5")],
        ),
        (
            json!({"temperature": 0.8, "top_k": 1, "repetition_penalty": 1.25}),
            vec![("temperature", "0"), ("top_k", "1"), ("repetition", "1.25")],
        ),
        (
            json!({"temperature": 0, "logit_bias": {"5": 1, "69999": -100}}),
            vec![("temperature", "0"), ("logit_bias", "2")],
        ),
        (
            json!({"temperature": 1, "top_k": -1}),
            vec![("temperature", "1"), ("top_k", vocab.as_str())],
        ),
        (
            json!({"temperature": 0.5, "top_k": 1_000_000, "top_p": 0.75, "min_p": 0.25}),
            vec![
                ("temperature", "0.5"),
                ("top_k", vocab.as_str()),
                ("top_p", "0.75"),
                ("min_p", "0.25"),
            ],
        ),
        (
            json!({"temperature": 1, "top_k": 70_000}),
            vec![("temperature", "1"), ("top_k", vocab.as_str())],
        ),
        (
            json!({"temperature": 1, "top_k": 3}),
            vec![("temperature", "1"), ("top_k", "3")],
        ),
    ];
    for (fields, set) in params {
        let sessions = submitted(&native, &path, &sampled(3, fields.clone())).await;
        let s = one_prefill(&sessions, 3);
        let mut want: BTreeMap<&str, &str> = defaults.into_iter().collect();
        want.extend(set);
        for step in s {
            for (k, v) in &want {
                assert_eq!(
                    step.get(*k).map(String::as_str),
                    Some(*v),
                    "{fields} {k}: {step:?}"
                );
            }
            assert!(step.contains_key("seed"), "{step:?}");
        }
    }
    // A `logit_bias` id outside the vocabulary is the request's fault and never reaches the executor (D8).
    let submissions = native.submissions().await;
    let response = native
        .chat(&sampled(
            3,
            json!({"temperature": 0, "logit_bias": {"70000": 1}}),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["error"]["code"], "invalid_value", "{error}");
    assert_eq!(error["error"]["param"], "logit_bias", "{error}");
    assert_eq!(native.submissions().await, submissions);
    // A positive `top_p`/`repetition_penalty` that rounds to 0 in the executor's f32 is out of range there, so it is
    // refused here as 400 `invalid_value` instead of failing the item in the executor (Z2A P3).
    for (param, value) in [("top_p", 1e-50), ("repetition_penalty", 1e-50)] {
        let submissions = native.submissions().await;
        let response = native
            .chat(&sampled(3, json!({"temperature": 1, param: value})))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400, "{param}");
        let error: Value = response.json().await.unwrap();
        assert_eq!(error["error"]["code"], "invalid_value", "{error}");
        assert_eq!(error["error"]["param"], param, "{error}");
        assert_eq!(native.submissions().await, submissions);
    }
}

/// C8 (D10, Q4 (a)): a native request without `temperature`/`top_k`/`top_p` samples with the artifact's
/// `generation_config.json` values read at upload (the fixture's 1.0/20/0.95, SGLang's `sampling_defaults="model"`),
/// min_p 0 and repetition 1; a field the request sets wins. The toy records the params of every item.
#[tokio::test]
async fn c8_absent_fields_sample_with_the_generation_config_defaults() {
    let t = Tmp::new("c8-record");
    let (case, path) = recording(&t);
    let native = Native::ready(case).await;
    // A fixed seed: a random one may sample the toy EOS (7) before the third step (see C7).
    let without = |mut fields: Value| {
        fields["seed"] = json!(11);
        let mut b = sampled(3, fields);
        b.as_object_mut().unwrap().remove("temperature");
        b
    };
    let cases = [
        (
            without(json!({})),
            vec![
                ("temperature", "1"),
                ("top_k", "20"),
                ("top_p", "0.949999988"),
            ],
        ),
        (
            without(json!({"top_k": 5})),
            vec![
                ("temperature", "1"),
                ("top_k", "5"),
                ("top_p", "0.949999988"),
            ],
        ),
        (
            without(json!({"top_p": 0.5})),
            vec![("temperature", "1"), ("top_k", "20"), ("top_p", "0.5")],
        ),
    ];
    for (body, set) in cases {
        let sessions = submitted(&native, &path, &body).await;
        let s = one_prefill(&sessions, 3);
        for step in s {
            for (k, v) in set.iter().chain(&[("min_p", "0"), ("repetition", "1")]) {
                assert_eq!(
                    step.get(*k).map(String::as_str),
                    Some(*v),
                    "{body} {k}: {step:?}"
                );
            }
        }
    }
}

fn tokens_of(session: &[Step]) -> Vec<String> {
    session.iter().map(|step| step["token"].clone()).collect()
}

/// The one seed every step of `session` carried.
fn seed_of(session: &[Step]) -> String {
    let seed = session[0]["seed"].clone();
    assert!(
        session.iter().all(|step| step["seed"] == seed),
        "{session:?}"
    );
    seed
}

/// A 16-token sampled request at T 1 over the toy's 8 finite logits, with `seed` when given.
fn seeded(seed: Option<u64>) -> Value {
    let mut fields = json!({"temperature": 1});
    if let Some(seed) = seed {
        fields["seed"] = json!(seed);
    }
    sampled(16, fields)
}

/// C7 (D4): the same seed gives the same tokens and another seed other tokens (toy logits, T 1), alone and as 2-4
/// concurrent requests (each its own session, its sampled row sharing the 4-row rounds, ledger CR4); a request
/// without `seed` gets its own random seed.
#[tokio::test]
async fn c7_seeds_repeat_alone_and_concurrent() {
    let t = Tmp::new("c7-record");
    let (case, path) = recording(&t);
    let native = Native::ready(case).await;
    let a = submitted(&native, &path, &seeded(Some(11))).await;
    let a = one_prefill(&a, 16).to_vec();
    assert_eq!(seed_of(&a), "11");
    let b = submitted(&native, &path, &seeded(Some(11))).await;
    let b = one_prefill(&b, 16).to_vec();
    let c = submitted(&native, &path, &seeded(Some(12))).await;
    let c = one_prefill(&c, 16).to_vec();
    assert_eq!(seed_of(&c), "12");
    let greedy: Vec<String> = toy_output(TEXT, 16).iter().map(u32::to_string).collect();
    assert_eq!(tokens_of(&a), tokens_of(&b));
    assert_ne!(tokens_of(&a), tokens_of(&c));
    assert_ne!(tokens_of(&a), greedy, "T 1 sampled the greedy sequence");
    // The largest accepted seed reaches the executor unchanged.
    let top = submitted(&native, &path, &seeded(Some(i64::MAX as u64))).await;
    assert_eq!(seed_of(one_prefill(&top, 16)), i64::MAX.to_string());

    let expected = |seed: &str| {
        if seed == "11" {
            tokens_of(&a)
        } else {
            tokens_of(&c)
        }
    };
    for n in 2..=4 {
        let before = steps(&path).len();
        let seeds = [11, 12, 11, 12];
        let calls = seeds[..n].iter().map(|seed| {
            let body = seeded(Some(*seed));
            let request = native.chat(&body);
            async move {
                let response = request.send().await.unwrap();
                assert_eq!(response.status(), 200);
                let _: Value = response.json().await.unwrap();
            }
        });
        futures_util::future::join_all(calls).await;
        let sessions = by_session(&steps(&path)[before..]);
        assert_eq!(sessions.len(), n, "{sessions:?}");
        for session in &sessions {
            assert_eq!(session.len(), 16);
            assert_eq!(tokens_of(session), expected(&seed_of(session)), "rows {n}");
        }
    }

    // Unseeded requests draw random seeds, so a session may sample the toy's EOS (7) before 16 tokens: check
    // the seed only (one session each), not the length.
    let x = submitted(&native, &path, &seeded(None)).await;
    let y = submitted(&native, &path, &seeded(None)).await;
    assert_eq!((x.len(), y.len()), (1, 1), "{x:?} {y:?}");
    let (x, y) = (seed_of(&x[0]), seed_of(&y[0]));
    assert_ne!(x, y, "two requests without seed drew the same seed");
}

/// m26 D11 / ledger CR4 (M8 channels (2, 3)): sampled `n` 2-4 runs one session per choice, decoded as sampled rows of
/// one round; choice 0 keeps the request's seed and commits what the request alone commits, choice i >= 1 samples
/// with `choice_seed(seed, i)`, so the choices' tokens and texts are distinct.
#[tokio::test]
async fn sampled_n_choices_draw_distinct_per_choice_seeds() {
    let t = Tmp::new("cr4-record");
    let (case, path) = recording(&t);
    let native = Native::ready(Case {
        tokenizer: Some(distinct_tokenizer(&[])),
        ..case
    })
    .await;
    let alone = submitted(&native, &path, &seeded(Some(11))).await;
    let alone = tokens_of(one_prefill(&alone, 16));
    for n in 2..=4u32 {
        let mut body = seeded(Some(11));
        body["n"] = json!(n);
        let before = steps(&path).len();
        let response = native.chat(&body).send().await.unwrap();
        assert_eq!(response.status(), 200, "n {n}");
        let got: Value = response.json().await.unwrap();
        let sessions = by_session(&steps(&path)[before..]);
        assert_eq!(sessions.len(), n as usize, "{sessions:?}");
        let mut want: BTreeSet<String> = (1..n)
            .map(|i| cere_runtime::native::sampling::choice_seed(11, i).to_string())
            .collect();
        want.insert("11".to_owned());
        let seeds: BTreeSet<String> = sessions.iter().map(|s| seed_of(s)).collect();
        assert_eq!(seeds, want, "n {n}");
        let mut tokens = BTreeSet::new();
        for session in &sessions {
            assert_eq!(session.len(), 16, "n {n}: {session:?}");
            if seed_of(session) == "11" {
                assert_eq!(tokens_of(session), alone, "n {n}: choice 0");
            }
            tokens.insert(tokens_of(session));
        }
        assert_eq!(tokens.len(), n as usize, "n {n}: choices repeat tokens");
        // The distinct tokenizer renders token `id` as `[id]`: each choice's text is its own session's tokens.
        let mut texts = BTreeSet::new();
        for choice in got["choices"].as_array().unwrap() {
            texts.insert(choice["message"]["content"].as_str().unwrap().to_owned());
        }
        let want: BTreeSet<String> = tokens
            .iter()
            .map(|t| {
                t.iter()
                    .map(|id| ["[", id.as_str(), "]"].concat())
                    .collect()
            })
            .collect();
        assert_eq!(texts, want, "n {n}: {got}");
    }
}

// M2.11 U1 preemption (spec m211 D8, C4): one executor slot, `preempt = true`, two requests in flight.

/// The long generation commits `HOLD` tokens, then the toy holds it (`hold_after`) until a cancel: there it has
/// enough committed tokens (`MIN` ≤ `HOLD`) to be the victim. After the resume its fresh session has fewer than
/// `HOLD` tokens left, so it runs to its end; the short request never reaches the hold.
const HOLD: u64 = 24;
const MIN: u64 = 16;
const LONG_MAX: u64 = 40;
const SHORT_MAX: u64 = 8;
const LONG_P: (usize, char) = TEXT;
const SHORT_P: (usize, char) = (136, 'x');
const THIRD_P: (usize, char) = (300, '2');
/// The third request's step count, which no other session of the U1 cases reaches (its session in the toy's record).
const THIRD_MAX: u64 = 2000;

fn snapshot_case(preempt: bool, min_tokens: u64, device_id: &str) -> Case {
    Case {
        device_id: device_id.to_owned(),
        tokenizer: Some(distinct_tokenizer(&[])),
        snapshot: Some(SnapshotProfile {
            preempt,
            preempt_min_tokens: min_tokens,
            inflight: if preempt { 2 } else { 1 },
            preempt_every: 0,
            rows: 1,
            parked: 1,
            reserve: 0,
        }),
        ..Case::matching()
    }
}

fn held_device() -> String {
    format!("toy,hold_after={HOLD},hold_ms=60000,hold_count=1")
}

/// The unary body of `p` with `max_tokens`.
async fn unary(native: &Native, p: (usize, char), max_tokens: u64) -> Value {
    let response = native
        .chat(&body(p, max_tokens, false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    response.json().await.unwrap()
}

/// The unary body equals the toy's solo run of `p`: text, finish reason and token count.
fn assert_solo(native: &Native, got: &Value, p: (usize, char), max_tokens: u64) {
    let ids = toy_output(p, max_tokens as usize);
    assert_eq!(
        ids.len() as u64,
        max_tokens,
        "the prompt reaches no stop ID"
    );
    assert_eq!(
        got["choices"][0]["message"]["content"],
        json!(expected_text(native, &ids)),
        "{got}"
    );
    assert_eq!(got["choices"][0]["finish_reason"], "length", "{got}");
    assert_eq!(got["usage"]["completion_tokens"], max_tokens, "{got}");
}

/// Waits until `native_submissions` reaches `n` (the generation has been started).
async fn started(native: &Native, n: u64) {
    for _ in 0..2000 {
        if native.submissions().await >= n {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("native_submissions never reached {n}");
}

/// C4: a long generation is preempted by a short request (snapshot, release, the short one runs, create +
/// restore + DECODE of the rest), both outputs equal their solo runs, one preemption is counted, and the parked
/// session resumes before the next grant: in the toy's record, a third request, started once the short one is,
/// takes its first step after the resumed generation's last. (Response arrival order is not the executor's: the
/// third request's toy steps take about a millisecond; the scheduler's order is `native::preempt`'s unit tests.)
#[tokio::test]
async fn preempt_parks_a_long_generation_for_a_short_request() {
    let t = Tmp::new("u1-preempt");
    let record = t.join("steps.record");
    let device = format!("{},record={}", held_device(), record.display());
    let native = Native::ready(snapshot_case(true, MIN, &device)).await;
    let short = async {
        started(&native, 1).await;
        unary(&native, SHORT_P, SHORT_MAX).await
    };
    let third = async {
        started(&native, 2).await;
        unary(&native, THIRD_P, THIRD_MAX).await
    };
    let (long, short, third) = tokio::join!(unary(&native, LONG_P, LONG_MAX), short, third);
    assert_solo(&native, &long, LONG_P, LONG_MAX);
    assert_solo(&native, &short, SHORT_P, SHORT_MAX);
    assert_solo(&native, &third, THIRD_P, THIRD_MAX);
    assert_eq!(native.preemptions().await, 1);
    assert_resumed_before_third(&steps(&record));
}

/// C4: `preempt = false` (the shipped profile, one request in flight): concurrent requests are served one after
/// the other exactly as before M2.11, nothing is preempted.
#[tokio::test]
async fn preempt_off_serves_requests_unchanged() {
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&[]))).await;
    let (long, short) = tokio::join!(
        unary(&native, LONG_P, LONG_MAX),
        unary(&native, SHORT_P, SHORT_MAX)
    );
    assert_solo(&native, &long, LONG_P, LONG_MAX);
    assert_solo(&native, &short, SHORT_P, SHORT_MAX);
    assert_eq!(native.preemptions().await, 0);
}

/// D13 (M211R-01): `preempt_every` forces a preemption of the only running generation after every N committed
/// tokens, with no competing request and below `preempt_min_tokens`: in the toy's record each session commits
/// exactly N = 8 tokens (five sessions for 40 tokens, none after the last: the budget ended it), and the output
/// equals the solo run.
#[tokio::test]
async fn preempt_every_forces_preemptions_of_a_lone_generation() {
    let t = Tmp::new("preempt-every");
    let path = t.join("steps.record");
    let mut case = snapshot_case(true, 512, &format!("toy,record={}", path.display()));
    if let Some(s) = case.snapshot.as_mut() {
        s.preempt_every = 8;
    }
    let native = Native::ready(case).await;
    let before = steps(&path).len();
    let long = unary(&native, LONG_P, LONG_MAX).await;
    assert_solo(&native, &long, LONG_P, LONG_MAX);
    let sessions = by_session(&steps(&path)[before..]);
    let lens: Vec<usize> = sessions.iter().map(Vec::len).collect();
    assert_eq!(lens, [8; 5], "{sessions:?}");
    assert_eq!(native.preemptions().await, 4);
}

/// Ledger CR13 (M2.7 x M2.11): a sampled request with a frequency penalty, preempted after every 4 committed tokens
/// (each resume a fresh session restored from the snapshot, whose kind 8 carries the output counts; the counter key
/// is the restored position), commits the tokens of the same request served without preemption. `CR13_P` at T 0
/// with frequency 2: the toy's top candidate at the 11th token is the 6th token (36,515, committed before the
/// preemption at 8), so the penalty picks 36,516; a resume that dropped the counts would commit 36,515 again. The
/// preemptions are at 4, 8 and 12 exactly (four sessions of 4 tokens in the toy's record).
const CR13_P: (usize, char) = (302, '0');

#[tokio::test]
async fn preempted_sampled_request_keeps_its_penalties() {
    let mut body = body(CR13_P, 16, false);
    body["frequency_penalty"] = json!(2);
    let t = Tmp::new("cr13-solo");
    let (case, path) = recording(&t);
    let solo = Native::ready(case).await;
    let alone = submitted(&solo, &path, &body).await;
    assert_eq!(alone.len(), 1, "{alone:?}");
    let want = tokens_of(&alone[0]);
    assert_eq!(want.len(), 16, "{want:?}");
    assert_eq!((want[5].as_str(), want[10].as_str()), ("36515", "36516"));

    let t = Tmp::new("cr13-preempt");
    let path = t.join("steps.record");
    let mut case = snapshot_case(true, 512, &format!("toy,record={}", path.display()));
    if let Some(s) = case.snapshot.as_mut() {
        s.preempt_every = 4;
    }
    let native = Native::ready(case).await;
    let sessions = submitted(&native, &path, &body).await;
    let lens: Vec<usize> = sessions.iter().map(Vec::len).collect();
    assert_eq!(
        lens, [4; 4],
        "not preempted at every 4 committed tokens: {sessions:?}"
    );
    assert_eq!(native.preemptions().await, 3);
    let got: Vec<String> = sessions.iter().flat_map(|s| tokens_of(s)).collect();
    assert_eq!(got, want);
}

// sessions design note SE-09 optimistic reservation: `reserve_output_tokens` R, a session of prompt + R that a
// generation grows (self-park, restore into its whole budget) when it uses it up.

/// A U1 case with `rows` decode rows, `parked` park slots = `inflight` (the SE-09 bound) and reserve `reserve`.
fn reserve_case(rows: u64, parked: u64, reserve: u64, device_id: &str) -> Case {
    Case {
        device_id: device_id.to_owned(),
        tokenizer: Some(distinct_tokenizer(&[])),
        snapshot: Some(SnapshotProfile {
            preempt: true,
            preempt_min_tokens: 512,
            inflight: parked,
            preempt_every: 0,
            rows,
            parked,
            reserve,
        }),
        ..Case::matching()
    }
}

/// SE-09: `TEXT` (716 tokens) with R 64 opens a session of 832 (= 64 · ceil(780 / 64)); its 116th token uses it up, so
/// the generation parks itself and resumes in a session of its whole budget (960 for 200 tokens), restored (its first
/// step feeds one token), for the other 84. The output equals the solo run; one grow, no preemption.
#[tokio::test]
async fn reserve_grows_a_generation_that_uses_its_session_up() {
    let t = Tmp::new("se09-grow");
    let path = t.join("steps.record");
    let native = Native::ready(reserve_case(
        1,
        1,
        64,
        &format!("toy,record={}", path.display()),
    ))
    .await;
    let before = steps(&path).len();
    let got = unary(&native, TEXT, 200).await;
    assert_solo(&native, &got, TEXT, 200);
    let sessions = by_session(&steps(&path)[before..]);
    let lens: Vec<usize> = sessions.iter().map(Vec::len).collect();
    assert_eq!(lens, [116, 84], "{sessions:?}");
    assert_eq!(sessions[1][0]["count"], "1", "{sessions:?}");
    assert_eq!((native.grows().await, native.preemptions().await), (1, 0));
}

/// SE-09 with MTP speculation and the prefix cache on (the `-v45` shape): a drafting generation that uses its
/// session up grows (its image carries kind 10) and equals its solo run; one grow, no preemption.
#[tokio::test]
async fn reserve_grows_a_drafting_generation() {
    let native = Native::ready(Case {
        speculation: true,
        prefix_state_slots: Some(32),
        ..reserve_case(1, 1, 64, "toy")
    })
    .await;
    let got = unary(&native, TEXT, 200).await;
    assert_solo(&native, &got, TEXT, 200);
    assert_eq!((native.grows().await, native.preemptions().await), (1, 0));
}

/// SE-09 with two decode rows (two park slots: the toy's arena is real host memory, 2.7 GB a slot): two concurrent
/// generations, each past its R 64, grow out of their batch and equal their solo runs; two grows.
#[tokio::test]
async fn reserve_grows_batched_generations() {
    let native = Native::ready(reserve_case(2, 2, 64, "toy")).await;
    let (a, b) = tokio::join!(unary(&native, TEXT, 200), unary(&native, THIRD_P, 200));
    assert_solo(&native, &a, TEXT, 200);
    assert_solo(&native, &b, THIRD_P, 200);
    assert_eq!((native.grows().await, native.preemptions().await), (2, 0));
}

/// A prompt (ending in `last`) whose toy output reaches a stop ID at its `lo..hi`-th token.
fn stopping_prompt(last: char, lo: usize, hi: usize) -> (usize, char) {
    (200..40_000)
        .map(|len| (len, last))
        .find(|&p| {
            let ids = toy_output(p, hi);
            (lo..hi).contains(&ids.len()) && [6, 7].contains(ids.last().unwrap())
        })
        .expect("a stopping prompt")
}

/// SE-09 admission: a generation with a 120,000-token budget that the toy holds (after 8 tokens, until its gate file
/// is gone) and a second one with the same budget, on the 200,000-token pool. With their whole budgets reserved
/// (R 0) the second waits for the first's pages; with R 64 both hold prompt + 64 only, so the second runs to its stop
/// while the first is held. Both stop before their reserve is used up: no grow, no preemption, outputs as solo.
#[tokio::test]
async fn reserve_admits_a_second_generation_the_whole_budget_would_block() {
    const BUDGET: u64 = 120_000;
    let held = stopping_prompt('4', 20, 60);
    for reserve in [0, 64] {
        let t = Tmp::new("se09-admit");
        let gate = t.join("hold.gate");
        std::fs::write(&gate, b"").unwrap();
        let device = format!(
            "toy,hold_after=8,hold_ms=60000,hold_count=1,hold_gate={}",
            gate.display()
        );
        let native = Native::ready(Case {
            max_output_tokens: Some(BUDGET),
            ..reserve_case(2, 2, reserve, &device)
        })
        .await;
        let first = unary(&native, held, BUDGET);
        let second = async {
            started(&native, 1).await;
            let got = tokio::select! {
                got = unary(&native, STOP, BUDGET) => Some(got),
                () = tokio::time::sleep(Duration::from_secs(2)) => None,
            };
            let ran_while_held = got.is_some();
            std::fs::remove_file(&gate).unwrap();
            let got = match got {
                Some(got) => got,
                None => unary(&native, STOP, BUDGET).await,
            };
            (got, ran_while_held)
        };
        let (first, (second, ran_while_held)) = tokio::join!(first, second);
        assert_eq!(ran_while_held, reserve > 0, "reserve {reserve}");
        for (got, p) in [(&first, held), (&second, STOP)] {
            // the toy's solo run of `p` ends in its stop ID, which the content leaves out
            let ids = toy_output(p, BUDGET as usize);
            assert!(
                [6, 7].contains(ids.last().unwrap()),
                "{p:?} reaches no stop ID"
            );
            assert_eq!(
                got["choices"][0]["message"]["content"],
                json!(expected_text(&native, &ids[..ids.len() - 1])),
                "reserve {reserve}: {got}"
            );
            assert_eq!(got["choices"][0]["finish_reason"], "stop", "{got}");
            assert_eq!(got["usage"]["completion_tokens"], ids.len(), "{got}");
        }
        assert_eq!((native.grows().await, native.preemptions().await), (0, 0));
    }
}

// ---------------------------------------------------------------------------------------------------------
// M2.11 U1 outcomes (spec m211 D8, C11).

/// C11: no eligible victim (the running generation has fewer than `preempt_min_tokens` committed tokens): the
/// new request waits for the slot and nothing is preempted. The long generation is held by the toy, so the short
/// request is still waiting when it is checked; cancelling the long one frees the slot.
#[tokio::test]
async fn u1_no_eligible_victim_waits_without_preemption() {
    let native = Native::ready(snapshot_case(true, 1000, &held_device())).await;
    let response = native
        .chat(&body(LONG_P, LONG_MAX, true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let long = rid(&response);
    let reader = SseReader::new(response);
    let short_done = std::sync::atomic::AtomicBool::new(false);
    let short = async {
        let v = unary(&native, SHORT_P, SHORT_MAX).await;
        short_done.store(true, std::sync::atomic::Ordering::SeqCst);
        v
    };
    let check = async {
        started(&native, 2).await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!short_done.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(native.preemptions().await, 0);
        let cancel = http()
            .post(
                native
                    .cered
                    .url(&format!("/v1/cere/requests/{long}/cancel")),
            )
            .bearer_auth(TOKEN)
            .send()
            .await
            .unwrap();
        assert_eq!(cancel.status(), 202);
    };
    let (short, ()) = tokio::join!(short, check);
    drop(reader);
    assert_solo(&native, &short, SHORT_P, SHORT_MAX);
    assert_eq!(native.terminal(&long).await["state"], "cancelled");
    assert_eq!(native.preemptions().await, 0);
}

/// C11: the arena cannot hold the victim (the executor refuses the snapshot with CAPACITY after the cancel): no
/// preemption; the victim keeps its session and continues its DECODE (restore-free resume), the new request
/// waits for the slot, and both outputs equal their solo runs.
#[tokio::test]
async fn u1_arena_too_small_means_no_preemption() {
    let device = format!("{},snapshot_capacity=1", held_device());
    let native = Native::ready(snapshot_case(true, MIN, &device)).await;
    let short = async {
        started(&native, 1).await;
        unary(&native, SHORT_P, SHORT_MAX).await
    };
    let (long, short) = tokio::join!(unary(&native, LONG_P, LONG_MAX), short);
    assert_solo(&native, &long, LONG_P, LONG_MAX);
    assert_solo(&native, &short, SHORT_P, SHORT_MAX);
    assert_eq!(native.preemptions().await, 0);
}

/// C11: the parked head's restore returns CAPACITY: it stays parked and retries (here with no session held), and
/// no new request is granted a session meanwhile. The toy refuses every restore while its gate file exists; the
/// test removes the gate only once the third request is queued for a slot (`native_submissions` 3, counted after
/// the queueing) and a restore was refused after that, so the head retried while the third request waited. The
/// executor's order is the toy's record: the resumed session runs to its end before the third request's first step.
#[tokio::test]
async fn u1_resume_restore_capacity_retries_without_a_new_grant() {
    let t = Tmp::new("u1-restore-gate");
    let (gate, record) = (t.join("restore.gate"), t.join("steps.record"));
    std::fs::write(&gate, "").unwrap();
    let device = format!(
        "{},restore_gate={},record={}",
        held_device(),
        gate.display(),
        record.display()
    );
    let native = Native::ready(snapshot_case(true, MIN, &device)).await;
    let short = async {
        started(&native, 1).await;
        unary(&native, SHORT_P, SHORT_MAX).await
    };
    let third = async {
        started(&native, 2).await;
        unary(&native, THIRD_P, THIRD_MAX).await
    };
    let refusals = || std::fs::read_to_string(&gate).unwrap().lines().count();
    let open_gate = async {
        started(&native, 3).await;
        let before = refusals();
        for _ in 0..2000 {
            if refusals() > before {
                std::fs::remove_file(&gate).unwrap();
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("no restore was refused after the third request was queued");
    };
    let (long, short, third, ()) =
        tokio::join!(unary(&native, LONG_P, LONG_MAX), short, third, open_gate);
    assert_solo(&native, &long, LONG_P, LONG_MAX);
    assert_solo(&native, &short, SHORT_P, SHORT_MAX);
    assert_solo(&native, &third, THIRD_P, THIRD_MAX);
    assert_eq!(native.preemptions().await, 1);
    assert_resumed_before_third(&steps(&record));
}

/// The toy's record of a U1 run with one preemption of the long generation and a third request: the resumed
/// session (the only one whose first step feeds one token: restored, not prefilled) runs its `LONG_MAX - HOLD`
/// steps before the first step of the third request's session (the one with `THIRD_MAX` steps).
fn assert_resumed_before_third(record: &[Step]) {
    let mut sessions: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, step) in record.iter().enumerate() {
        sessions.entry(&step["session"]).or_default().push(i);
    }
    let only = |what: &str, pick: &dyn Fn(&Vec<usize>) -> bool| {
        let found: Vec<&Vec<usize>> = sessions.values().filter(|s| pick(s)).collect();
        assert_eq!(found.len(), 1, "one {what} session: {sessions:?}");
        found[0].clone()
    };
    let resumed = only("resumed", &|s| record[s[0]]["count"] == "1");
    let third = only("third", &|s| s.len() as u64 == THIRD_MAX);
    assert_eq!(resumed.len() as u64, LONG_MAX - HOLD, "{sessions:?}");
    assert!(
        resumed.last() < third.first(),
        "the third request stepped before the resumed generation ended: {sessions:?}"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// M2.11 D9 (C11): the U3 export buffer is mlock'ed while alive and zeroed before it is freed.

/// The test binary's allocator: `System`, plus one watched allocation whose bytes are checked at `dealloc`.
struct Watching;

/// Address of the watched allocation (0 = none) and whether it was all zero when freed (0 unknown, 1 yes, 2 no).
static WATCHED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static WATCHED_ZERO: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

// SAFETY: forwards every call to `System`; `dealloc` only reads the block it is about to free.
unsafe impl std::alloc::GlobalAlloc for Watching {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        std::alloc::System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        use std::sync::atomic::Ordering::SeqCst;
        if ptr as usize == WATCHED.load(SeqCst) {
            let bytes = std::slice::from_raw_parts(ptr, layout.size());
            let zero = bytes.iter().all(|&b| b == 0);
            WATCHED_ZERO.store(if zero { 1 } else { 2 }, SeqCst);
            WATCHED.store(0, SeqCst);
        }
        std::alloc::System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: Watching = Watching;

/// `VmLck` of this process in bytes (`/proc/self/status`).
fn locked_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmLck:")).unwrap();
    let kb: u64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kb * 1024
}

/// C11 / D9: a `LockedBuf` (the U3 export buffer) is locked in RAM while it lives (VmLck grows by at least its
/// size, and shrinks back when it is dropped) and its bytes are zero when the allocator frees it. The `mlock`
/// failure path (export fails, no pageable fallback) cannot be shown here: the tests run as uid 0, whose
/// CAP_IPC_LOCK makes `mlock` ignore RLIMIT_MEMLOCK.
#[test]
fn u3_export_buffer_is_mlocked_and_zeroed_on_drop() {
    use cere_runtime::native::export::LockedBuf;
    use std::sync::atomic::Ordering::SeqCst;
    const LEN: usize = 4 << 20;
    // cered runs as a child process, so this test is the only one in this process that locks memory.
    let before = locked_bytes();
    let mut buf = LockedBuf::new(LEN).unwrap();
    let during = locked_bytes();
    assert!(
        during >= before + LEN as u64,
        "VmLck {before} -> {during}, buffer {LEN}"
    );
    buf.as_mut_slice().fill(0xA5);
    WATCHED.store(buf.as_slice().as_ptr() as usize, SeqCst);
    drop(buf);
    assert_eq!(WATCHED_ZERO.load(SeqCst), 1, "freed bytes all zero");
    assert!(locked_bytes() < during, "unlocked on drop");
}

/// M2.12 C6 first clause (D8): a native profile declaring `stop_token_ids` takes the request's ids next to the
/// model's on every ticket. The toy's `TEXT` output is [2235, 2, 780, ...]: id 780 ends DECODE, id 2235 the PREFILL
/// token; finish `stop`, `matched_stop` = the id, the stop token committed and counted, and its text (non-special
/// in the distinct tokenizer) left out of the content, unary and streamed. Without the capability: 400.
#[tokio::test]
async fn request_stop_token_ids_end_the_output() {
    let case = Case {
        capabilities_patch: json!({"token_controls": ["stop_token_ids"], "vocab_size": FIXTURE_VOCAB}),
        ..with_tokenizer(distinct_tokenizer(&[]))
    };
    let native = Native::ready(case).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let ids = toy_output(TEXT, 8);
    assert_eq!(ids[..3], [2235, 2, 780]);
    for (stop_id, len) in [(780, 3), (2235, 1)] {
        let mut request = body(TEXT, 8, false);
        request["stop_token_ids"] = json!([stop_id, 69_999]);
        let response = native.chat(&request).send().await.unwrap();
        assert_eq!(response.status(), 200);
        let unary: Value = response.json().await.unwrap();
        let choice = &unary["choices"][0];
        assert_eq!(
            (&choice["finish_reason"], &choice["matched_stop"]),
            (&json!("stop"), &json!(stop_id)),
            "{unary}"
        );
        assert_eq!(unary["usage"]["completion_tokens"], len);
        let want = vocab.decode(&ids[..len - 1]);
        assert_eq!(choice["message"]["content"], want.as_str());

        request["stream"] = json!(true);
        let response = native.chat(&request).send().await.unwrap();
        let events = SseReader::new(response).rest().await;
        check_stream(&events);
        assert_eq!(streamed_text(&events), (want, String::new()));
        let last = stream_choices(&events).pop().unwrap();
        assert_eq!(
            (&last["finish_reason"], &last["matched_stop"]),
            (&json!("stop"), &json!(stop_id))
        );
    }
    let mut request = body(TEXT, 8, false);
    request["stop_token_ids"] = json!([FIXTURE_VOCAB]);
    assert_eq!(native.chat(&request).send().await.unwrap().status(), 400);

    let shipped = Native::ready(pre_a1()).await;
    let mut request = body(TEXT, 8, false);
    request["stop_token_ids"] = json!([780]);
    let response = shipped.chat(&request).send().await.unwrap();
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["error"]["param"], "stop_token_ids", "{error}");
}

/// M2.12 C6 `ignore_eos` clauses (D9): a native profile declaring `ignore_eos` runs through the model's stop token
/// (the toy's EOS 7 after `STOP`) and a request `stop_token_ids` id to `length`, `completion_tokens` = `max_tokens`
/// (every ticket, PREFILL and DECODE, carries `CERE_WORK_IGNORE_STOP_TOKENS`; the matcher's token list is empty);
/// a stop string still ends it. Without the declaration (a pre-A1 profile) the field stays refused (old-library guard, D2).
#[tokio::test]
async fn ignore_eos_runs_through_stop_tokens_to_length() {
    let case = Case {
        capabilities_patch: json!({
            "token_controls": ["stop_token_ids", "ignore_eos"],
            "vocab_size": FIXTURE_VOCAB
        }),
        ..Case::matching()
    };
    let native = Native::ready(case).await;
    let ids = toy_output(STOP, 8);
    assert_eq!(ids[1], 7, "the toy's EOS is the second token after STOP");
    for stream in [false, true] {
        let mut request = body(STOP, 8, stream);
        request["ignore_eos"] = json!(true);
        request["stop_token_ids"] = json!([ids[0]]);
        if stream {
            request["stream_options"] = json!({ "include_usage": true });
        }
        let response = native.chat(&request).send().await.unwrap();
        assert_eq!(response.status(), 200);
        let (finish, matched, usage) = if stream {
            let events = SseReader::new(response).rest().await;
            check_stream(&events);
            let last = stream_choices(&events).pop().unwrap();
            let usage = events
                .iter()
                .filter_map(|e| match e {
                    Event::Data(v) => v.get("usage").filter(|u| !u.is_null()).cloned(),
                    Event::Done => None,
                })
                .last()
                .unwrap();
            (
                last["finish_reason"].clone(),
                last["matched_stop"].clone(),
                usage,
            )
        } else {
            let unary: Value = response.json().await.unwrap();
            let choice = &unary["choices"][0];
            (
                choice["finish_reason"].clone(),
                choice["matched_stop"].clone(),
                unary["usage"].clone(),
            )
        };
        assert_eq!(
            (finish, matched),
            (json!("length"), Value::Null),
            "stream {stream}"
        );
        assert_eq!(usage["completion_tokens"], 8, "stream {stream}: {usage}");
    }
    // A stop string still ends an `ignore_eos` request (the toy's TEXT output decodes "ab" at its 2nd token).
    let mut request = body(TEXT, 8, false);
    request["ignore_eos"] = json!(true);
    request["stop"] = json!(["ab"]);
    let unary: Value = native
        .chat(&request)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let choice = &unary["choices"][0];
    assert_eq!(
        (&choice["finish_reason"], &choice["matched_stop"]),
        (&json!("stop"), &json!("ab")),
        "{unary}"
    );
    // `ignore_eos: false` is the plain request: the EOS finishes it.
    let mut request = body(STOP, 8, false);
    request["ignore_eos"] = json!(false);
    let unary: Value = native
        .chat(&request)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(unary["choices"][0]["finish_reason"], "stop", "{unary}");
    assert_eq!(unary["usage"]["completion_tokens"], 2);

    let shipped = Native::ready(pre_a1()).await;
    let mut request = body(STOP, 8, false);
    request["ignore_eos"] = json!(true);
    let response = shipped.chat(&request).send().await.unwrap();
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["error"]["param"], "ignore_eos", "{error}");
}

/// M2.12 C3/D4 native thinking: on a profile declaring `thinking_toggle` (native max context 73,728 >= 73,728, Q6
/// satisfied since M2.2c), `enable_thinking: true` renders through the sidecar's `thinking` feature (the stub ends
/// the prompt with one more token) and the reasoning parser runs forced: the whole output is `reasoning_content`.
/// Off (explicit false or absent) renders and answers exactly as before.
#[tokio::test]
async fn native_thinking_renders_and_forces_the_reasoning_split() {
    let case = Case {
        capabilities_patch: json!({"thinking_toggle": true}),
        ..Case::matching()
    };
    let native = Native::ready(case).await;
    let unary = |thinking: Option<bool>| {
        let mut request = body(TEXT, 4, false);
        if let Some(on) = thinking {
            request["chat_template_kwargs"] = json!({"enable_thinking": on});
        }
        native.chat(&request).send()
    };
    let off: Value = unary(None).await.unwrap().json().await.unwrap();
    let explicit_off: Value = unary(Some(false)).await.unwrap().json().await.unwrap();
    let on: Value = unary(Some(true)).await.unwrap().json().await.unwrap();
    assert_eq!(off["choices"], explicit_off["choices"], "{explicit_off}");
    assert_eq!(off["usage"], explicit_off["usage"]);
    let prompt = off["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(on["usage"]["prompt_tokens"], prompt + 1, "{on}");
    let message = &on["choices"][0]["message"];
    assert_eq!(message["content"].as_str().unwrap_or_default(), "", "{on}");
    assert_eq!(message["reasoning_content"], "ab", "{on}");
    assert!(off["choices"][0]["message"]["content"]
        .as_str()
        .is_some_and(|c| !c.is_empty()));
}

/// api-lenient design note AL-2 (D3): `preserve_thinking: false` (either spelling) reaches the sidecar's
/// `encode_chat` frame (the stub renders one more prompt token for it); `true` and absence render the same prompt.
#[tokio::test]
async fn native_preserve_thinking_false_reaches_the_sidecar_frame() {
    let case = Case {
        capabilities_patch: json!({"thinking_toggle": true}),
        ..Case::matching()
    };
    let native = Native::ready(case).await;
    let prompt = |patch: Value| {
        let mut request = body(TEXT, 4, false);
        for (key, value) in patch.as_object().unwrap() {
            request[key] = value.clone();
        }
        let native = &native;
        async move {
            let response = native.chat(&request).send().await.unwrap();
            assert_eq!(response.status(), 200, "{request}");
            let body: Value = response.json().await.unwrap();
            body["usage"]["prompt_tokens"].as_u64().unwrap()
        }
    };
    let on = json!({"enable_thinking": true});
    let absent = prompt(json!({"chat_template_kwargs": on})).await;
    let kept = prompt(
        json!({"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}),
    )
    .await;
    let dropped = prompt(
        json!({"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": false}}),
    )
    .await;
    let top_level = prompt(json!({"chat_template_kwargs": on, "preserve_thinking": false})).await;
    assert_eq!(kept, absent);
    assert_eq!(dropped, absent + 1);
    assert_eq!(top_level, dropped);
}

/// M2.12 C5 (D7): `usage.reasoning_tokens` follows SGLang's rule (`Req.update_reasoning_tokens`, the
/// `thinking_cases.json` `reasoning_tokens` spans): with thinking resolved on, the output tokens up to and
/// including `</think>`, all of them when it never appears; 0 with thinking off. `completion_tokens` counts every
/// token and no `completion_tokens_details` key appears. The thinking stub's prompt makes the toy answer
/// `[2, 780, 24899, 2589]`; this tokenizer spells 780 `</think>`.
#[tokio::test]
async fn reasoning_tokens_count_through_the_think_end() {
    let tokenizer = distinct_tokenizer(&[(2, "ab"), (780, "</think>"), (24899, "c"), (2589, "d")]);
    let case = Case {
        capabilities_patch: json!({"thinking_toggle": true}),
        ..with_tokenizer(tokenizer)
    };
    let native = Native::ready(case).await;
    let usage = |body: &Value| -> (u64, u64) {
        let usage = &body["usage"];
        assert!(usage.get("completion_tokens_details").is_none(), "{body}");
        (
            usage["completion_tokens"].as_u64().unwrap(),
            usage["reasoning_tokens"].as_u64().unwrap(),
        )
    };
    let ask = |thinking: bool, max_tokens: u64, extra: Value| {
        let mut request = body(TEXT, max_tokens, false);
        request["chat_template_kwargs"] = json!({"enable_thinking": thinking});
        for (key, value) in extra.as_object().unwrap() {
            request[key] = value.clone();
        }
        native.chat(&request).send()
    };
    let on: Value = ask(true, 4, json!({})).await.unwrap().json().await.unwrap();
    assert_eq!(usage(&on), (4, 2), "{on}");
    assert_eq!(
        on["choices"][0]["message"]["reasoning_content"], "ab",
        "{on}"
    );
    assert_eq!(on["choices"][0]["message"]["content"], "cd", "{on}");
    // `separate_reasoning: false` changes the text, not the count (`require_reasoning` = the thinking flag).
    let raw: Value = ask(true, 4, json!({"separate_reasoning": false}))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(usage(&raw), (4, 2), "{raw}");
    // `length` inside the reasoning: every token is reasoning.
    let cut: Value = ask(true, 1, json!({})).await.unwrap().json().await.unwrap();
    assert_eq!(usage(&cut), (1, 1), "{cut}");
    assert_eq!(cut["choices"][0]["finish_reason"], "length");
    let off: Value = ask(false, 4, json!({}))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(usage(&off).1, 0, "{off}");
    // SSE: the usage chunk carries the same count.
    let mut request = body(TEXT, 4, true);
    request["chat_template_kwargs"] = json!({"enable_thinking": true});
    request["stream_options"] = json!({"include_usage": true});
    let response = native.chat(&request).send().await.unwrap();
    let events = SseReader::new(response).rest().await;
    let last_usage = events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) if !v["usage"].is_null() => Some(v["usage"].clone()),
            _ => None,
        })
        .last()
        .unwrap();
    assert_eq!(last_usage["reasoning_tokens"], 2, "{last_usage}");
    assert_eq!(last_usage["completion_tokens"], 4);
}

/// ApiFit AF-L7: the masked data events of one SSE chat body.
async fn masked_events(native: &Native, request: &Value) -> Vec<String> {
    let response = native.chat(request).send().await.unwrap();
    assert_eq!(response.status(), 200);
    SseReader::new(response)
        .rest()
        .await
        .iter()
        .map(|e| match e {
            Event::Data(v) => masked(&serde_json::to_string(v).unwrap()),
            Event::Done => "[DONE]".to_owned(),
        })
        .collect()
}

/// ApiFit AF-14 (AF-L7): a `thinking_budget` the thinking ends within changes nothing: unary and SSE bodies equal
/// the unbudgeted ones byte for byte (masked ids). The thinking stub's prompt makes the toy answer `[2, 780, 24899,
/// 2589]`; this tokenizer spells 780 `</think>` (thinking 2 tokens) and has the closing newlines.
#[tokio::test]
async fn af14_a_thinking_budget_the_thinking_ends_within_changes_nothing() {
    let tokenizer = distinct_tokenizer(&[
        (2, "ab"),
        (780, "</think>"),
        (24899, "c"),
        (2589, "d"),
        (FIXTURE_VOCAB - 2, "Ċ"),
        (FIXTURE_VOCAB - 3, "ĊĊ"),
    ]);
    let native = Native::ready(Case {
        capabilities_patch: json!({"thinking_toggle": true}),
        ..with_tokenizer(tokenizer)
    })
    .await;
    let request = |stream: bool, budget: Option<u64>| {
        let mut request = body(TEXT, 8, stream);
        request["chat_template_kwargs"] = json!({"enable_thinking": true});
        if stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        if let Some(budget) = budget {
            request["thinking_budget"] = json!(budget);
        }
        request
    };
    let unary = |budget| {
        let request = request(false, budget);
        let native = &native;
        async move {
            let response = native.chat(&request).send().await.unwrap();
            assert_eq!(response.status(), 200);
            masked(&response.text().await.unwrap())
        }
    };
    let free = unary(None).await;
    let parsed: Value = serde_json::from_str(&free).unwrap();
    assert_eq!(parsed["usage"]["reasoning_tokens"], 2, "{free}");
    // Budgets 2 (`</think>` is the budget's last token), 3 and 7 (max_tokens - 1).
    for budget in [2, 3, 7] {
        assert_eq!(unary(Some(budget)).await, free, "budget {budget}");
    }
    let free = masked_events(&native, &request(true, None)).await;
    for budget in [2, 3, 7] {
        assert_eq!(
            masked_events(&native, &request(true, Some(budget))).await,
            free,
            "budget {budget}"
        );
    }
}

/// ApiFit AF-15 (AF-L7): a binding `thinking_budget` B. This tokenizer spells `</think>` with an ID the toy does not
/// reach, so the unbudgeted thinking never ends; with B the output's thinking is its first B-2 tokens, `"\n"` and
/// `</think>` (`reasoning_tokens` = B), then `"\n\n"` and the continuation's answer up to `max_tokens`; unary and
/// SSE carry the same text and usage, and the stream is well formed (reasoning deltas, then content, one finish,
/// the usage chunk, `[DONE]`). With speculation too: a speculate round may commit past B, the overshoot is dropped.
#[tokio::test]
async fn af15_a_binding_thinking_budget_cuts_the_thinking_and_answers() {
    for speculation in [false, true] {
        binding_thinking_budget(speculation).await;
    }
}

async fn binding_thinking_budget(speculation: bool) {
    let (think_end, newline, blank) = (FIXTURE_VOCAB - 1, FIXTURE_VOCAB - 2, FIXTURE_VOCAB - 3);
    let tokenizer = distinct_tokenizer(&[(think_end, "</think>"), (newline, "Ċ"), (blank, "ĊĊ")]);
    let native = Native::ready(Case {
        capabilities_patch: json!({"thinking_toggle": true}),
        speculation,
        ..with_tokenizer(tokenizer)
    })
    .await;
    const MAX: u64 = 48;
    const BUDGET: u64 = 16;
    let request = |stream: bool, budget: Option<u64>| {
        let mut request = body(TEXT, MAX, stream);
        request["chat_template_kwargs"] = json!({"enable_thinking": true});
        if stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        if let Some(budget) = budget {
            request["thinking_budget"] = json!(budget);
        }
        request
    };
    let free: Value = native
        .chat(&request(false, None))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let free_reasoning = free["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap()
        .to_owned();
    let free_tokens = free["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(free_tokens > BUDGET, "{free}");
    assert_eq!(
        free["usage"]["reasoning_tokens"], free_tokens,
        "never closes: {free}"
    );
    assert!(!free_reasoning.contains('\n'), "{free}");

    let cut: Value = native
        .chat(&request(false, Some(BUDGET)))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let usage = &cut["usage"];
    assert_eq!(usage["reasoning_tokens"], BUDGET, "{cut}");
    let completion = usage["completion_tokens"].as_u64().unwrap();
    assert!(completion > BUDGET + 1 && completion <= MAX, "{cut}");
    let message = &cut["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap();
    // The kept thinking is the unbudgeted thinking's first B-2 tokens (each spelled `[id]`).
    assert_eq!(
        reasoning.trim_end().matches('[').count() as u64,
        BUDGET - 2,
        "{cut}"
    );
    assert!(
        free_reasoning.starts_with(reasoning.trim_end()),
        "{cut}\n{free}"
    );
    let content = message["content"].as_str().unwrap();
    assert!(!content.is_empty(), "{cut}");
    let finish = cut["choices"][0]["finish_reason"].as_str().unwrap();
    assert_eq!(finish == "length", completion == MAX, "{cut}");

    // SSE: the same text and usage, well formed across the seam.
    let response = native
        .chat(&request(true, Some(BUDGET)))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let events = SseReader::new(response).rest().await;
    assert!(matches!(events.last(), Some(Event::Done)));
    let (streamed_content, streamed_reasoning) = streamed_text(&events);
    assert_eq!(streamed_reasoning, reasoning, "{events:?}");
    assert_eq!(streamed_content, content, "{events:?}");
    let mut answered = false;
    let mut finishes = 0;
    for e in &events[..events.len() - 1] {
        let delta = &e.data()["choices"][0]["delta"];
        if delta["content"].as_str().is_some_and(|c| !c.is_empty()) {
            answered = true;
        }
        if delta["reasoning_content"]
            .as_str()
            .is_some_and(|r| !r.is_empty())
        {
            assert!(!answered, "reasoning after content: {events:?}");
        }
        if e.data()["choices"][0]["finish_reason"].is_string() {
            finishes += 1;
            assert_eq!(e.data()["choices"][0]["finish_reason"], finish);
        }
    }
    assert_eq!(finishes, 1, "{events:?}");
    let streamed_usage = events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) if !v["usage"].is_null() => Some(v["usage"].clone()),
            _ => None,
        })
        .last()
        .unwrap();
    assert_eq!(streamed_usage["reasoning_tokens"], BUDGET);
    assert_eq!(streamed_usage["completion_tokens"], completion);
    assert_eq!(streamed_usage["prompt_tokens"], usage["prompt_tokens"]);
}

/// ApiFit AF-16 (AF-L7): `thinking_budget` is refused without thinking, with logprobs, and when not a positive
/// integer; a profile without `thinking_toggle` refuses the field.
#[tokio::test]
async fn af16_thinking_budget_refusals() {
    let native = Native::ready(Case {
        capabilities_patch: json!({"thinking_toggle": true}),
        ..Case::matching()
    })
    .await;
    let refused = |extra: Value| {
        let mut request = body(TEXT, 8, false);
        for (key, value) in extra.as_object().unwrap() {
            request[key] = value.clone();
        }
        let native = &native;
        async move {
            let response = native.chat(&request).send().await.unwrap();
            let status = response.status().as_u16();
            let body: Value = response.json().await.unwrap();
            (
                status,
                body["error"]["param"].clone(),
                body["error"]["code"].clone(),
            )
        }
    };
    let on = json!({"enable_thinking": true});
    for (extra, code) in [
        (
            json!({"thinking_budget": 4, "chat_template_kwargs": {"enable_thinking": false}}),
            "invalid_value",
        ),
        (
            json!({"thinking_budget": 0, "chat_template_kwargs": on}),
            "invalid_value",
        ),
        (
            json!({"thinking_budget": "4", "chat_template_kwargs": on}),
            "invalid_value",
        ),
        (
            json!({"thinking_budget": 4, "chat_template_kwargs": on, "logprobs": true}),
            "unsupported_parameter",
        ),
    ] {
        let (status, param, got) = refused(extra.clone()).await;
        assert_eq!(
            (status, param, got),
            (400, json!("thinking_budget"), json!(code)),
            "{extra}"
        );
    }
    let plain = Native::ready(Case::matching()).await;
    let mut request = body(TEXT, 8, false);
    request["thinking_budget"] = json!(4);
    let response = plain.chat(&request).send().await.unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["param"], "thinking_budget", "{body}");
}

/// A native profile without the M2.12 fields (the pre-A1 `-v2` profile, the library before D9): every M2.12 field
/// is refused (old-library guard, D2).
fn pre_a1() -> Case {
    Case {
        capabilities_patch: json!({"thinking_toggle": null, "token_controls": null, "vocab_size": null}),
        ..Case::matching()
    }
}

/// The M2.6 C4 regression requests: `n` absent or 1, unary and SSE, a `length`, a stop-string and a tool-call
/// finish; `(tokenizer, name, body)`.
fn n_one_requests() -> Vec<(&'static str, &'static str, Value)> {
    let with_n = |mut b: Value| {
        b["n"] = json!(1);
        b
    };
    let usage = |mut b: Value| {
        b["stream_options"] = json!({"include_usage": true});
        b
    };
    let stop = |mut b: Value| {
        b["stop"] = json!(["zzz", "[2]"]);
        b
    };
    vec![
        ("distinct", "unary_n1_length", with_n(body(TEXT, 4, false))),
        (
            "distinct",
            "stream_n1_length_usage",
            with_n(usage(body(TEXT, 6, true))),
        ),
        ("distinct", "unary_stop_string", stop(body(TEXT, 16, false))),
        (
            "distinct",
            "stream_n1_stop_string_usage",
            with_n(usage(stop(body(TEXT, 16, true)))),
        ),
        ("tools", "unary_tool_call", tool_body(STOP, 8, false)),
        (
            "tools",
            "stream_n1_tool_call_usage",
            with_n(usage(tool_body(STOP, 8, true))),
        ),
    ]
}

/// [`masked`] plus every tool call's random `id`.
fn masked_calls(raw: &str) -> String {
    fn walk(v: &mut Value) {
        match v {
            Value::Object(map) => {
                if let Some(Value::Array(calls)) = map.get_mut("tool_calls") {
                    for call in calls.iter_mut() {
                        if call.get("id").is_some_and(Value::is_string) {
                            call["id"] = json!("<call_id>");
                        }
                    }
                }
                map.values_mut().for_each(walk);
            }
            Value::Array(items) => items.iter_mut().for_each(walk),
            _ => {}
        }
    }
    let mut v: Value = serde_json::from_str(&masked(raw)).unwrap();
    walk(&mut v);
    serde_json::to_string(&v).unwrap()
}

/// M2.6 C4 regression guard: `n` absent or 1 answers (stop strings and tools included) are byte-equal to the
/// bodies recorded on the merge base, apart from the values [`masked_calls`] names.
#[tokio::test]
async fn n_one_bodies_equal_merge_base_goldens() {
    let mut got = String::new();
    for tokenizer in ["distinct", "tools"] {
        let case = match tokenizer {
            "distinct" => with_tokenizer(distinct_tokenizer(&[])),
            _ => with_tokenizer(tool_tokenizer(&[(2240, CALL_TEXT)])),
        };
        let native = Native::ready(case).await;
        got.push_str(&format!("## {tokenizer}\n"));
        for (_, name, request) in n_one_requests().into_iter().filter(|r| r.0 == tokenizer) {
            let response = native.chat(&request).send().await.unwrap();
            assert_eq!(response.status(), 200, "{name}");
            let raw = response.text().await.unwrap();
            got.push_str(&format!("# {name}\n"));
            if request["stream"] == true {
                for event in raw.split("\n\n").filter(|e| !e.is_empty()) {
                    match event.split_once("\ndata: ") {
                        Some((id, "[DONE]")) => got.push_str(&format!("{id} [DONE]\n")),
                        Some((id, data)) => got.push_str(&format!("{id} {}\n", masked_calls(data))),
                        None => got.push_str(&format!("{event}\n")),
                    }
                }
            } else {
                got.push_str(&format!("{}\n", masked_calls(&raw)));
            }
        }
    }
    assert!(got == N_ONE_GOLDEN, "n absent/1 bodies differ:\n{got}");
}

/// Recorded on `bl/m26-rust`'s merge base `cfe1a3a` (pre-M2.6 code); `profile_id` moved to `-v4` with the rows
/// (m26 D10), to `-v5` with M2.12 (ledger P1) and to `-v7` with the sampled rows (m27 D11); nothing else changed.
const N_ONE_GOLDEN: &str = r#"## distinct
# unary_n1_length
{"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null,"message":{"content":"[2235][2][780][24899]","reasoning_content":null,"role":"assistant","tool_calls":null}}],"created":0,"id":"<id>","metadata":{"weight_version":"default"},"model":"Qwen3.8-Flash-Next","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":720}}
# stream_n1_length_usage
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu0"},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"[2235]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"committed_tokens":null,"seq":3},"choices":[{"delta":{"content":"[2]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 4 {"cere":{"committed_tokens":null,"seq":4},"choices":[{"delta":{"content":"[780]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 5 {"cere":{"committed_tokens":null,"seq":5},"choices":[{"delta":{"content":"[24899]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 6 {"cere":{"committed_tokens":null,"seq":6},"choices":[{"delta":{"content":"[2589]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 7 {"cere":{"committed_tokens":null,"seq":7},"choices":[{"delta":{"content":"[10980]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 8 {"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":8,"state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 9 {"cere":{"committed_tokens":null,"seq":9},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":{"completion_tokens":6,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":722}}
data: [DONE]
# unary_stop_string
{"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"matched_stop":"[2]","message":{"content":"[2235]","reasoning_content":null,"role":"assistant","tool_calls":null}}],"created":0,"id":"<id>","metadata":{"weight_version":"default"},"model":"Qwen3.8-Flash-Next","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":718}}
# stream_n1_stop_string_usage
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu0"},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"[2235]","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":3,"state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"stop","index":0,"logprobs":null,"matched_stop":"[2]"}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 4 {"cere":{"committed_tokens":null,"seq":4},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":{"completion_tokens":2,"prompt_tokens":716,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":718}}
data: [DONE]
## tools
# unary_tool_call
{"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"finish_reason":"tool_calls","index":0,"logprobs":null,"matched_stop":null,"message":{"content":"Let me check.","reasoning_content":null,"role":"assistant","tool_calls":[{"function":{"arguments":"{\"city\": \"Paris\", \"days\": 3}","name":"get_weather"},"id":"<call_id>","index":0,"type":"function"}]}}],"created":0,"id":"<id>","metadata":{"weight_version":"default"},"model":"Qwen3.8-Flash-Next","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":566,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":568}}
# stream_n1_tool_call_usage
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu0"},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"Let me check.","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"committed_tokens":null,"seq":3},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":[{"function":{"arguments":"","name":"get_weather"},"id":"<call_id>","index":0,"type":"function"}]},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 4 {"cere":{"committed_tokens":null,"seq":4},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":[{"function":{"arguments":"{","name":null},"id":null,"index":0,"type":"function"}]},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 5 {"cere":{"committed_tokens":null,"seq":5},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":[{"function":{"arguments":"\"city\": \"Paris\"","name":null},"id":null,"index":0,"type":"function"}]},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 6 {"cere":{"committed_tokens":null,"seq":6},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":[{"function":{"arguments":", \"days\": 3","name":null},"id":null,"index":0,"type":"function"}]},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 7 {"cere":{"committed_tokens":null,"seq":7},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":[{"function":{"arguments":"}","name":null},"id":null,"index":0,"type":"function"}]},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 8 {"cere":{"attempt":1,"cache_report":"reported","cached_tokens":0,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"native","instance_id":"<masked>","lock_sha256":"<masked>","profile_id":"native-g0-qwen38-flash-next-v8","recipe_id":"qwen38-flash-next-native","request_id":"<masked>","seq":8,"state":"completed","worker_epoch":1,"worker_id":"gpu0"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"tool_calls","index":0,"logprobs":null,"matched_stop":7}],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 9 {"cere":{"committed_tokens":null,"seq":9},"choices":[],"created":0,"id":"<id>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":{"completion_tokens":2,"prompt_tokens":566,"prompt_tokens_details":{"cached_tokens":0},"reasoning_tokens":0,"total_tokens":568}}
data: [DONE]
"#;

/// The streamed choices of index `i`.
fn index_choices(events: &[Event], i: u64) -> Vec<Value> {
    stream_choices(events)
        .into_iter()
        .filter(|c| c["index"] == i)
        .collect()
}

/// M2.6 C4 (D9): four concurrent requests on the 4-row toy each equal their sequential answer.
#[tokio::test]
async fn four_concurrent_requests_equal_their_sequential_answers() {
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&[]))).await;
    let bodies: Vec<Value> = [TEXT, STOP, (600, '3'), (650, '9')]
        .into_iter()
        .map(|p| body(p, 12, false))
        .collect();
    let answer = |v: Value| (v["choices"].clone(), v["usage"].clone());
    let mut sequential = Vec::new();
    for b in &bodies {
        let response = native.chat(b).send().await.unwrap();
        assert_eq!(response.status(), 200);
        sequential.push(answer(response.json().await.unwrap()));
    }
    let concurrent = futures_util::future::join_all(bodies.iter().map(|b| async {
        let response = native.chat(b).send().await.unwrap();
        assert_eq!(response.status(), 200);
        answer(response.json().await.unwrap())
    }))
    .await;
    assert_eq!(concurrent, sequential);
    assert_eq!(native.submissions().await, 8);
}

/// M2.6 C4 (D11): `n` = 3 unary runs 3 sessions; 3 identical choices with `index` 0..2, each the n = 1 answer;
/// usage counts the prompt once and sums the completions; the request record's usage equals the response's.
#[tokio::test]
async fn n_three_unary_choices_and_usage() {
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&[]))).await;
    let one: Value = native
        .chat(&body(TEXT, 6, false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut three = body(TEXT, 6, false);
    three["n"] = json!(3);
    let response = native.chat(&three).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let got: Value = response.json().await.unwrap();
    let choices = got["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3, "{got}");
    for (i, choice) in choices.iter().enumerate() {
        let mut want = one["choices"][0].clone();
        want["index"] = json!(i);
        assert_eq!(choice, &want, "{got}");
    }
    let prompt = one["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion = 3 * one["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(
        got["usage"],
        json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
               "prompt_tokens_details": {"cached_tokens": 0}, "reasoning_tokens": 0})
    );
    let status = native.terminal(&rid).await;
    assert_eq!(status["state"], "completed", "{status}");
    assert_eq!(
        (
            &status["usage"]["prompt_tokens"],
            &status["usage"]["completion_tokens"]
        ),
        (&json!(prompt), &json!(completion)),
        "{status}"
    );
    assert_eq!(native.submissions().await, 4);
}

/// M2.6 C4 (D11): `n` = 3 SSE: per-index chunks (each index's text the n = 1 text), one finish per index, one
/// usage chunk with the summed completions, the request record's usage equal to it.
#[tokio::test]
async fn n_three_stream_chunks_per_index() {
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&[]))).await;
    let mut one = body(TEXT, 6, true);
    one["stream_options"] = json!({"include_usage": true});
    let events = SseReader::new(native.chat(&one).send().await.unwrap())
        .rest()
        .await;
    let want_text = streamed_text(&events).0;
    let mut three = one.clone();
    three["n"] = json!(3);
    let response = native.chat(&three).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let events = SseReader::new(response).rest().await;
    assert!(matches!(events.last(), Some(Event::Done)), "{events:?}");
    assert_eq!(events[0].data()["cere"]["event"], "accepted");
    for i in 0..3 {
        let choices = index_choices(&events, i);
        let text: String = choices
            .iter()
            .filter_map(|c| c["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, want_text, "index {i}");
        let roles = choices.iter().filter(|c| c["delta"]["role"] == "assistant");
        assert_eq!(roles.count(), 1, "index {i}");
        let finishes: Vec<_> = choices
            .iter()
            .filter(|c| !c["finish_reason"].is_null())
            .collect();
        assert_eq!(finishes.len(), 1, "index {i}: {events:?}");
        assert_eq!(finishes[0]["finish_reason"], "length");
    }
    assert!(index_choices(&events, 3).is_empty());
    let data: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| e.data().clone())
        .collect();
    let seqs: Vec<_> = data
        .iter()
        .map(|d| d["cere"]["seq"].as_u64().unwrap())
        .collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "{seqs:?}");
    let usage: Vec<_> = data
        .iter()
        .filter(|d| d["choices"] == json!([]) && !d["usage"].is_null())
        .collect();
    assert_eq!(usage.len(), 1, "{events:?}");
    assert_eq!(usage[0]["usage"]["completion_tokens"], 18);
    assert_eq!(usage[0]["usage"]["prompt_tokens"], 716);
    // The last finish carries the terminal `cere`; the usage chunk follows it.
    let terminal = &data[data.len() - 2];
    assert_eq!(terminal["cere"]["state"], "completed", "{terminal}");
    assert!(!terminal["choices"][0]["finish_reason"].is_null());
    let status = native.terminal(&rid).await;
    assert_eq!(status["state"], "completed", "{status}");
    assert_eq!(status["usage"]["completion_tokens"], 18, "{status}");
}

/// M2.6 C4 (D11): every choice runs its own stop matcher and tool chain: n = 2 with a stop string ends both at
/// the match; n = 2 with tools calls in both (each call its own id), unary and SSE.
#[tokio::test]
async fn n_two_choices_each_run_stop_and_tools() {
    let native = Native::ready(with_tokenizer(distinct_tokenizer(&[]))).await;
    let mut stop = body(TEXT, 16, false);
    stop["stop"] = json!(["zzz", "[2]"]);
    stop["n"] = json!(2);
    let got: Value = native
        .chat(&stop)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for i in 0..2 {
        let choice = &got["choices"][i];
        assert_eq!(choice["index"], i, "{got}");
        assert_eq!(choice["message"]["content"], "[2235]", "{got}");
        assert_eq!(
            (&choice["finish_reason"], &choice["matched_stop"]),
            (&json!("stop"), &json!("[2]"))
        );
    }
    assert_eq!(got["usage"]["completion_tokens"], 4, "{got}");

    let native = Native::ready(with_tokenizer(tool_tokenizer(&[(2240, CALL_TEXT)]))).await;
    let mut unary = tool_body(STOP, 8, false);
    unary["n"] = json!(2);
    let got: Value = native
        .chat(&unary)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut ids = BTreeSet::new();
    for i in 0..2 {
        let choice = &got["choices"][i];
        assert_eq!(choice["index"], i, "{got}");
        assert_eq!(choice["finish_reason"], "tool_calls", "{got}");
        let call = &choice["message"]["tool_calls"][0];
        assert_call_id(&call["id"]);
        ids.insert(call["id"].as_str().unwrap().to_owned());
        assert_eq!(call["function"]["arguments"], CALL_ARGUMENTS, "{got}");
    }
    assert_eq!(ids.len(), 2, "{got}");
    let mut stream = tool_body(STOP, 8, true);
    stream["n"] = json!(2);
    let events = SseReader::new(native.chat(&stream).send().await.unwrap())
        .rest()
        .await;
    for i in 0..2 {
        let choices = index_choices(&events, i);
        let calls = stream_calls(&choices);
        assert_eq!(calls.len(), 5, "index {i}: {calls:?}");
        assert_call_id(&calls[0].0);
        let last = choices.last().unwrap();
        assert_eq!(
            (&last["finish_reason"], &last["matched_stop"]),
            (&json!("tool_calls"), &json!(7))
        );
    }
}

/// M2.6 C4 (D12): a request with `n` choices holds `n` rows of the 16 (4 before m23 design note P8): while an
/// n = 15 stream holds, an n = 2 request waits (its deadline passes while queued) and an n = 1 request runs; once the
/// stream ends, n = 2 runs.
#[tokio::test]
async fn n_choices_hold_n_rows() {
    let native = Native::ready(held()).await;
    let mut fifteen = body(TEXT, 16, true);
    fifteen["n"] = json!(15);
    let response = native.chat(&fifteen).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let held_rid = rid(&response);
    let mut reader = SseReader::new(response);
    read_to_hold(&mut reader).await;
    let mut two = body(STOP, 8, false);
    two["n"] = json!(2);
    let response = native
        .chat(&two)
        .header("X-Cere-Deadline-Ms", "300")
        .send()
        .await
        .unwrap();
    let waited = rid(&response);
    assert_eq!(response.status(), 504);
    let status = native.terminal(&waited).await;
    assert_eq!(status["reason"], "deadline_exceeded", "{status}");
    let response = native.chat(&body(STOP, 8, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let cancel = http()
        .post(
            native
                .cered
                .url(&format!("/v1/cere/requests/{held_rid}/cancel")),
        )
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    reader.rest().await;
    assert_eq!(native.terminal(&held_rid).await["state"], "cancelled");
    let response = native.chat(&two).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let got: Value = response.json().await.unwrap();
    assert_eq!(got["choices"].as_array().unwrap().len(), 2, "{got}");
}

/// A prompt whose session needs over half of the shipped 200,000-token pool: `round64(101,341 + 16)` = 1,584 of its
/// 3,125 pages. Its length is `TEXT`'s modulo 4,375, so the toy's 2nd token is "ab" too (the held boundary).
const BIG: (usize, char) = (101_341, '1');

async fn cancel_request(native: &Native, rid: &str) {
    let cancel = http()
        .post(native.cered.url(&format!("/v1/cere/requests/{rid}/cancel")))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
}

/// Ledger CR1 (m25 N2 (a), m26 N1 (d)): a session reserves its pages at create, sized
/// `min(max_context_tokens, round64(prompt + max_tokens))`, from the shipped 200,000-token pool. While a `BIG`
/// stream holds, a small request runs beside it; a second `BIG` request's create is CAPACITY, so cered queues it (no
/// error to the client) until the first ends, and then it runs.
#[tokio::test]
async fn pool_capacity_queues_the_request() {
    let native = Native::ready(held()).await;
    let response = native.chat(&body(BIG, 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let first_rid = rid(&response);
    let mut first = SseReader::new(response);
    read_to_hold(&mut first).await;
    let response = native.chat(&body(TEXT, 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let small_rid = rid(&response);
    let mut small = SseReader::new(response);
    read_to_hold(&mut small).await;
    let request = native.chat(&body(BIG, 16, true));
    let second = tokio::spawn(async move {
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), 200);
        let rid = rid(&response);
        let mut reader = SseReader::new(response);
        read_to_hold(&mut reader).await;
        (rid, reader)
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !second.is_finished(),
        "the second BIG request did not wait for pages"
    );
    cancel_request(&native, &first_rid).await;
    first.rest().await;
    let (second_rid, mut second) = second.await.unwrap();
    for rid in [&second_rid, &small_rid] {
        cancel_request(&native, rid).await;
    }
    second.rest().await;
    small.rest().await;
    for rid in [&first_rid, &second_rid, &small_rid] {
        assert_eq!(native.terminal(rid).await["state"], "cancelled");
    }
    assert_freed(&native).await;
}

/// Ledger CR1: a request waiting for pages leaves the queue on its own cancel (the endpoint, a deadline or a
/// disconnect take the same `cere_executor_cancel` path) while the holder keeps its pages: it ends cancelled with no
/// session, and the holder is unaffected.
#[tokio::test]
async fn a_cancel_ends_a_request_waiting_for_pages() {
    let native = Native::ready(held()).await;
    let response = native.chat(&body(BIG, 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let first_rid = rid(&response);
    let mut first = SseReader::new(response);
    read_to_hold(&mut first).await;
    let response = native
        .chat(&body(BIG, 16, false))
        .header("X-Cere-Deadline-Ms", "500")
        .send()
        .await
        .unwrap();
    let waited = rid(&response);
    assert_eq!(response.status(), 504);
    assert_eq!(
        native.terminal(&waited).await["reason"],
        "deadline_exceeded"
    );
    cancel_request(&native, &first_rid).await;
    first.rest().await;
    assert_eq!(native.terminal(&first_rid).await["state"], "cancelled");
    assert_freed(&native).await;
}

/// build/page-wait-wake: a request's deadline that fires while its page-queue create is with the executor (the toy's
/// `create_capacity_delay_ms` holds the CAPACITY answer 800 ms, past the 500 ms deadline) ends the request at once:
/// 504 `deadline_exceeded` once the create answers, not 502 `cancel_unacknowledged` after the 5 s ack bound (the
/// cancel's wake found no waiter, and the create went back to wait for a release that only the holder's end brings).
#[tokio::test]
async fn a_deadline_during_a_refused_create_ends_the_waiting_request_promptly() {
    let native = Native::ready(Case {
        device_id: "toy,hold_after=2,create_capacity_delay_ms=800".to_owned(),
        ..Case::matching()
    })
    .await;
    let response = native.chat(&body(BIG, 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let first_rid = rid(&response);
    let mut first = SseReader::new(response);
    read_to_hold(&mut first).await;
    let start = std::time::Instant::now();
    let response = native
        .chat(&body(BIG, 16, false))
        .header("X-Cere-Deadline-Ms", "500")
        .send()
        .await
        .unwrap();
    let elapsed = start.elapsed();
    let waited = rid(&response);
    assert_eq!(response.status(), 504);
    assert_eq!(
        native.terminal(&waited).await["reason"],
        "deadline_exceeded"
    );
    assert!(
        elapsed < Duration::from_millis(2500),
        "the waiting request ended {elapsed:?} after it was sent (the cancel ack bound is 5 s)"
    );
    cancel_request(&native, &first_rid).await;
    first.rest().await;
    assert_eq!(native.terminal(&first_rid).await["state"], "cancelled");
    assert_freed(&native).await;
}

// ---------------------------------------------------------------------------------------------------------
// M2.8 C7 (spec m28 D8, D15): the constrained serve loop over the toy and the stub grammar role. A regex
// "seq:A,B,..[;knob@K]" makes the stub's mask of position p allow exactly the p-th id (the last accepted one
// terminates the grammar), so the committed ids show which mask every position ran under; the toy commits the
// smallest allowed id above its own next under a mask, and records `mask=<allowed ids>` per masked step.

/// The tokenizer of the grammar tests: every id decodes to `[id]` (7 and 6 special stop tokens), `pieces` aside.
fn grammar_tokenizer(pieces: &[(u32, &str)]) -> Vec<u8> {
    tool_tokenizer(pieces)
}

/// A greedy `TEXT` request constrained by `regex` (the stub's script).
fn constrained(regex: &str, max_tokens: u64, stream: bool) -> Value {
    let mut b = body(TEXT, max_tokens, stream);
    b["regex"] = json!(regex);
    b
}

/// A unary request's status and body.
async fn answer(native: &Native, body: &Value) -> (u16, Value) {
    let response = native.chat(body).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// A stream's events, framing checked, and its terminal (finish chunk or error event).
async fn streamed(native: &Native, body: &Value) -> (Vec<Event>, Value) {
    let response = native.chat(body).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let events = SseReader::new(response).rest().await;
    let terminal = check_stream(&events);
    (events, terminal)
}

/// The committed ids of one recorded session.
fn ids_of(session: &[Step]) -> Vec<u32> {
    session
        .iter()
        .map(|s| s["token"].parse().unwrap())
        .collect()
}

/// C7 (D8 steps 1-3) and C10b: every position of a constrained request runs under its own mask, the PREFILL
/// token included (`mask=1` on every recorded step, the committed ids are the script's, not the toy's own
/// `[2235, 2, 780]`), so the DECODE ticket's position 0 ran under the mask the grammar returned after the
/// PREFILL token; an unconstrained request records no mask (`vocab_mask_words` 0). `length` mid-structure stays
/// `length`; a script ending on the model's stop token 7 ends `stop` there (the executor's own stop, the stop sets
/// agree). Unary and SSE agree; `GET /v1/cere/stats` reports the `grammar` latencies once a step ran.
#[tokio::test]
async fn c7_masks_apply_per_position_and_the_prefill_token_is_masked() {
    let t = Tmp::new("c7-masks");
    let (case, path) = recording(&t);
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[])),
        ..case
    })
    .await;
    let (status, stats) = native.get("/v1/cere/stats").await;
    assert_eq!(status, 200, "{stats}");
    assert_eq!(
        stats["grammar"],
        json!({"step_ms_p50": null, "step_ms_p99": null, "constrain_gap_ms_p50": null,
               "constrain_gap_ms_p99": null}),
        "{stats}"
    );

    let before = steps(&path).len();
    let (status, cut) = answer(&native, &constrained("seq:100,200,300,400,7", 3, false)).await;
    assert_eq!(status, 200, "{cut}");
    let sessions = by_session(&steps(&path)[before..]);
    let s = one_prefill(&sessions, 3);
    assert!(s.iter().all(|step| step["mask"] == "1"), "{s:?}");
    assert_eq!(ids_of(s), [100, 200, 300]);
    let choice = &cut["choices"][0];
    assert_eq!(choice["finish_reason"], "length", "{cut}");
    assert_eq!(choice["matched_stop"], Value::Null, "{cut}");
    assert_eq!(choice["message"]["content"], "[100][200][300]", "{cut}");

    let before = steps(&path).len();
    let (status, whole) = answer(&native, &constrained("seq:100,200,300,400,7", 8, false)).await;
    assert_eq!(status, 200, "{whole}");
    let sessions = by_session(&steps(&path)[before..]);
    let s = one_prefill(&sessions, 5);
    assert!(s.iter().all(|step| step["mask"] == "1"), "{s:?}");
    assert_eq!(ids_of(s), [100, 200, 300, 400, 7]);
    let choice = &whole["choices"][0];
    assert_eq!(
        (&choice["finish_reason"], &choice["matched_stop"]),
        (&json!("stop"), &json!(7)),
        "{whole}"
    );
    assert_eq!(
        choice["message"]["content"], "[100][200][300][400]",
        "{whole}"
    );
    assert_eq!(whole["usage"]["completion_tokens"], 5, "{whole}");

    let (events, terminal) =
        streamed(&native, &constrained("seq:100,200,300,400,7", 8, true)).await;
    assert_eq!(streamed_text(&events).0, "[100][200][300][400]");
    let finish = &terminal["choices"][0];
    assert_eq!(
        (&finish["finish_reason"], &finish["matched_stop"]),
        (&json!("stop"), &json!(7)),
        "{terminal}"
    );

    // C10b: an unconstrained request carries no mask.
    let sessions = submitted(&native, &path, &body(TEXT, 3, false)).await;
    let s = one_prefill(&sessions, 3);
    assert!(s.iter().all(|step| !step.contains_key("mask")), "{s:?}");
    assert_eq!(ids_of(s), toy_output(TEXT, 3));

    let (_, stats) = native.get("/v1/cere/stats").await;
    let grammar = &stats["grammar"];
    for key in [
        "step_ms_p50",
        "step_ms_p99",
        "constrain_gap_ms_p50",
        "constrain_gap_ms_p99",
    ] {
        assert!(
            grammar[key].as_f64().is_some_and(|ms| ms >= 0.0),
            "{key}: {stats}"
        );
    }
}

/// C7 (D8, D15 / ledger CR20): a grammar that terminates on a token the executor did not stop on cancels the
/// ticket and finishes `stop` with `matched_stop` = that token, its text trimmed as SGLang trims a matched stop
/// token; nothing is stepped after it. Under `ignore_eos` the executor runs through the model's stop token 7, so
/// the grammar's termination on 7 is what ends the request, with the same body as the executor's own stop.
#[tokio::test]
async fn c7_grammar_termination_stops_on_that_token() {
    let t = Tmp::new("c7-terminated");
    let (case, path) = recording(&t);
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[])),
        ..case
    })
    .await;
    let before = steps(&path).len();
    let (status, got) = answer(&native, &constrained("seq:100,200", 8, false)).await;
    assert_eq!(status, 200, "{got}");
    let sessions = by_session(&steps(&path)[before..]);
    assert_eq!(ids_of(one_prefill(&sessions, 2)), [100, 200]);
    let choice = &got["choices"][0];
    assert_eq!(
        (&choice["finish_reason"], &choice["matched_stop"]),
        (&json!("stop"), &json!(200)),
        "{got}"
    );
    assert_eq!(choice["message"]["content"], "[100]", "{got}");
    assert_eq!(got["usage"]["completion_tokens"], 2, "{got}");
    let (events, terminal) = streamed(&native, &constrained("seq:100,200", 8, true)).await;
    assert_eq!(streamed_text(&events).0, "[100]");
    assert_eq!(
        (
            &terminal["choices"][0]["finish_reason"],
            &terminal["choices"][0]["matched_stop"]
        ),
        (&json!("stop"), &json!(200)),
        "{terminal}"
    );

    let mut bodies = Vec::new();
    for ignore_eos in [false, true] {
        let mut request = constrained("seq:100,7", 8, false);
        request["ignore_eos"] = json!(ignore_eos);
        let before = steps(&path).len();
        let (status, got) = answer(&native, &request).await;
        assert_eq!(status, 200, "{got}");
        let sessions = by_session(&steps(&path)[before..]);
        assert_eq!(ids_of(one_prefill(&sessions, 2)), [100, 7], "{ignore_eos}");
        let choice = &got["choices"][0];
        assert_eq!(
            (&choice["finish_reason"], &choice["matched_stop"]),
            (&json!("stop"), &json!(7)),
            "{ignore_eos}: {got}"
        );
        assert_eq!(choice["message"]["content"], "[100]", "{got}");
        bodies.push((choice.clone(), got["usage"].clone()));
    }
    assert_eq!(bodies[0], bodies[1]);
}

/// C7 (D8 exceptions): a committed token the grammar rejects (`accepted: false`, here on the PREFILL token and on
/// a DECODE token) and a failing `grammar_step` are 500 `internal_error` (stream: an error event); the ticket is
/// cancelled, so nothing is stepped after the failing token, and the grammar closes: the stub keeps at most four
/// open grammars, and a constrained request still runs after seven failed ones.
#[tokio::test]
async fn c7_rejected_tokens_and_step_failures_are_500_and_cancel() {
    let t = Tmp::new("c7-failures");
    let (case, path) = recording(&t);
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[])),
        ..case
    })
    .await;
    for (regex, stepped) in [
        ("seq:100,200,300,400,7;reject@0", 1),
        ("seq:100,200,300,400,7;reject@2", 3),
        ("seq:100,200,300,400,7;fail@1", 2),
    ] {
        let before = steps(&path).len();
        let (status, got) = answer(&native, &constrained(regex, 8, false)).await;
        assert_eq!(status, 500, "{regex}: {got}");
        assert_eq!(got["error"]["code"], "internal_error", "{regex}: {got}");
        let sessions = by_session(&steps(&path)[before..]);
        assert_eq!(sessions.len(), 1, "{regex}: {sessions:?}");
        assert_eq!(sessions[0].len(), stepped, "{regex}: {sessions:?}");

        let (_, terminal) = streamed(&native, &constrained(regex, 8, true)).await;
        assert_eq!(terminal["error"]["code"], "internal_error", "{terminal}");
    }
    let (status, got) = answer(&native, &constrained("seq:100,7", 8, false)).await;
    assert_eq!(status, 200, "{got}");
    assert_freed(&native).await;
}

/// C7 (D8, A7): a client cancel while the ticket awaits its next mask (the grammar step of the first DECODE token
/// takes 2 s) ends the request `cancelled` once the step returns: the cancel clears `awaiting_mask`, so the
/// terminal poll reaches `CERE_FINISH_CANCELLED` and nothing hangs; the session is free afterwards.
#[tokio::test]
async fn c7_cancel_while_awaiting_a_mask_ends_cancelled() {
    let native = Native::ready(with_tokenizer(grammar_tokenizer(&[]))).await;
    let response = native
        .chat(&constrained("seq:100,200,300,400,7;sleep@1=2", 8, true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let mut reader = SseReader::new(response);
    loop {
        match reader.next().await {
            Some(Event::Data(v))
                if v.pointer("/choices/0/delta/content") == Some(&json!("[100]")) =>
            {
                break
            }
            Some(Event::Data(v)) => assert!(v.get("error").is_none(), "{v}"),
            other => panic!("the stream ended before the first token: {other:?}"),
        }
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    cancel_request(&native, &rid).await;
    let rest = tokio::time::timeout(Duration::from_secs(20), reader.rest())
        .await
        .expect("the cancelled stream did not end");
    assert!(matches!(rest.last(), Some(Event::Done)), "{rest:?}");
    let error = rest[rest.len() - 2].data();
    assert_eq!(error["cere"]["state"], "cancelled", "{error}");
    let (content, _) = streamed_text(&rest);
    assert!(!content.contains("[300]"), "{content}");
    assert_eq!(native.terminal(&rid).await["state"], "cancelled");
    assert_freed(&native).await;
}

/// The detail of the `/readyz` problem naming `field`, if any (the advisory `grammar` one of spec m28 D7, the
/// `sidecar` one of OH-01).
fn problem(body: &Value, field: &str) -> Option<String> {
    native_server::worker(body)["problems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["field"] == field)
        .map(|p| p["detail"].as_str().unwrap().to_owned())
}

/// Polls `/readyz` (published every 5 s probe) until `done` holds for its grammar problem, for at most 30 s; the
/// worker stays ready throughout (D7: the problem is advisory).
async fn grammar_readyz_until(native: &Native, what: &str, done: impl Fn(Option<&str>) -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let response = http()
            .get(native.cered.url("/readyz"))
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        let last: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "{last}");
        if done(problem(&last, "grammar").as_deref()) {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "/readyz never reported {what}: {last}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// C7 / D7 end to end: the grammar process dying in a step fails that request with 500 and is respawned; while
/// it respawns `/readyz` reports `grammar` (advisory: the worker stays ready), constrained requests are 503
/// `not_ready` and unconstrained ones run; once it is back the problem clears and constrained requests run again.
#[tokio::test]
async fn c7_grammar_process_death_fails_the_request_and_respawns() {
    let t = Tmp::new("c7-respawn");
    let slow = t.join("slow-respawn");
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[])),
        // Once `slow` exists a started child takes 8 s to answer `info`: a respawning window over a probe interval.
        sidecar_args: vec![
            "--slow-start-if".to_owned(),
            slow.display().to_string(),
            "8".to_owned(),
        ],
        ..Case::matching()
    })
    .await;
    std::fs::write(&slow, b"").unwrap();
    let (status, got) = answer(&native, &constrained("seq:100,200,300,7;die@1", 8, false)).await;
    assert_eq!(status, 500, "{got}");
    assert_eq!(got["error"]["code"], "internal_error", "{got}");
    grammar_readyz_until(&native, "the respawning grammar process", |p| {
        p.is_some_and(|d| d.contains("respawning"))
    })
    .await;
    let (status, refused) = answer(&native, &constrained("seq:100,7", 8, false)).await;
    assert_eq!(status, 503, "{refused}");
    assert_eq!(refused["error"]["code"], "not_ready", "{refused}");
    let (status, plain) = answer(&native, &body(STOP, 8, false)).await;
    assert_eq!(status, 200, "{plain}");
    grammar_readyz_until(&native, "the respawned grammar process", |p| p.is_none()).await;
    let (status, got) = answer(&native, &constrained("seq:100,7", 8, false)).await;
    assert_eq!(status, 200, "{got}");
    assert_eq!(got["choices"][0]["message"]["content"], "[100]", "{got}");
}

/// D7 end to end: a grammar process whose respawn fails stays down; `/readyz` reports it (the worker stays
/// ready), constrained requests are 503 and unconstrained ones run.
#[tokio::test]
async fn c7_grammar_process_that_did_not_respawn_refuses_constrained_requests() {
    let t = Tmp::new("c7-down");
    let marker = t.join("no-respawn");
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[])),
        sidecar_args: vec!["--fail-start-if".to_owned(), marker.display().to_string()],
        ..Case::matching()
    })
    .await;
    std::fs::write(&marker, b"").unwrap();
    let (status, got) = answer(&native, &constrained("seq:100,200,300,7;die@0", 8, false)).await;
    assert_eq!(status, 500, "{got}");
    grammar_readyz_until(&native, "the grammar process down", |p| {
        p.is_some_and(|d| d.contains("did not respawn"))
    })
    .await;
    let (status, refused) = answer(&native, &constrained("seq:100,7", 8, false)).await;
    assert_eq!(status, 503, "{refused}");
    assert_eq!(refused["error"]["code"], "not_ready", "{refused}");
    let (status, plain) = answer(&native, &body(STOP, 8, false)).await;
    assert_eq!(status, 200, "{plain}");
}

/// Polls `/readyz` (published every 5 s probe and at every sidecar health change) until the native worker is not
/// ready naming `sidecar` (`down`) or ready without it (`!down`), for at most 30 s; returns that body.
async fn sidecar_readyz_until(native: &Native, down: bool) -> Value {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let response = http()
            .get(native.cered.url("/readyz"))
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        let last: Value = response.json().await.unwrap();
        let named = problem(&last, "sidecar").is_some();
        if (down && status == 503 && named) || (!down && status == 200 && !named) {
            return last;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "/readyz never reported the sidecar {}: {last}",
            if down { "down" } else { "back" }
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Polls `/readyz` until the native worker lists a `sidecar` problem, for at most 30 s; returns that status and
/// body.
async fn sidecar_problem_listed(native: &Native) -> (u16, Value) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let response = http()
            .get(native.cered.url("/readyz"))
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        let last: Value = response.json().await.unwrap();
        if problem(&last, "sidecar").is_some() {
            return (status, last);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "/readyz never listed a sidecar problem: {last}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The stub tokenizer sidecar starts so far (its `--log` startup lines; the grammar process logs its own).
fn sidecar_starts(native: &Native) -> usize {
    native
        .sidecar_log()
        .iter()
        .filter(|l| l.starts_with("max_request_bytes"))
        .count()
}

/// The pid of `cered`'s tokenizer sidecar child (the stub without `--role grammar`).
fn tokenizer_sidecar_pid(native: &Native) -> i32 {
    let parent = native.cered.pid().to_string();
    std::fs::read_dir("/proc")
        .unwrap()
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<i32>().ok())
        .find(|pid| {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
            let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            let cmdline = String::from_utf8_lossy(&cmdline);
            // `pid (comm) state ppid ...`: the fields after the last `)`.
            stat.rsplit_once(") ")
                .and_then(|(_, rest)| rest.split(' ').nth(1))
                == Some(parent.as_str())
                && cmdline.contains("stub_sidecar.py")
                && !cmdline.contains("--role\0grammar")
        })
        .expect("the tokenizer sidecar child")
}

/// ops-hardening design note OH-01 end to end: a tokenizer sidecar that stops answering fails that request at the
/// I/O deadline (503 `not_ready`) and is killed; the next request respawns it and is served by the same `cered`,
/// and the worker is ready again.
#[tokio::test]
async fn oh01_hung_tokenizer_sidecar_fails_one_request_and_respawns() {
    let t = Tmp::new("oh01-hang");
    let slow = t.join("slow-encode");
    let native = Native::ready(Case {
        sidecar_deadline_ms: Some(1_000),
        // While `slow` exists every `encode_chat` reply waits an hour: a child that stopped answering.
        sidecar_args: vec![
            "--slow-encode-if".to_owned(),
            slow.display().to_string(),
            "3600".to_owned(),
        ],
        ..Case::matching()
    })
    .await;
    let started = sidecar_starts(&native);
    std::fs::write(&slow, b"").unwrap();
    let (status, got) = answer(&native, &body(STOP, 8, false)).await;
    assert_eq!(status, 503, "{got}");
    assert_eq!(got["error"]["code"], "not_ready", "{got}");
    std::fs::remove_file(&slow).unwrap();
    let (status, got) = answer(&native, &body(STOP, 8, false)).await;
    assert_eq!(status, 200, "{got}");
    assert_eq!(sidecar_starts(&native), started + 1);
    sidecar_readyz_until(&native, false).await;
    let log = native.cered.stderr_so_far();
    assert!(
        log.contains("cered: the tokenizer sidecar respawned (respawn 1)"),
        "{log}"
    );
}

/// OH-01 end to end: a killed tokenizer sidecar whose respawns keep failing leaves the worker not ready naming
/// `sidecar` across probes, with a bounded number of respawn attempts (the backoff); once a respawn starts the
/// worker is ready again and serves.
#[tokio::test]
async fn oh01_tokenizer_sidecar_that_does_not_respawn_stays_not_ready_until_one_does() {
    let t = Tmp::new("oh01-down");
    let marker = t.join("no-start");
    let native = Native::ready(Case {
        sidecar_args: vec!["--fail-start-if".to_owned(), marker.display().to_string()],
        ..Case::matching()
    })
    .await;
    std::fs::write(&marker, b"").unwrap();
    let pid = tokenizer_sidecar_pid(&native);
    // SAFETY: `kill` takes a pid and a signal and touches no memory; the pid is this test's cered's child.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
    let down = Instant::now();
    sidecar_readyz_until(&native, true).await;
    // At least one more probe (every 5 s) finds it still down.
    tokio::time::sleep(Duration::from_secs(6)).await;
    let body_down = sidecar_readyz_until(&native, true).await;
    let log = native.cered.stderr_so_far();
    let failed = log.matches("cered: the tokenizer sidecar respawn ").count();
    // One attempt per probe at most: never more than the probes since the kill, plus the first.
    let probes = down.elapsed().as_secs() / 5 + 2;
    assert!(
        (1..=probes as usize).contains(&failed),
        "{failed} failed respawns in {:?}: {log}\n{body_down}",
        down.elapsed()
    );
    let (status, got) = answer(&native, &body(STOP, 8, false)).await;
    assert_eq!(status, 503, "{got}");
    std::fs::remove_file(&marker).unwrap();
    sidecar_readyz_until(&native, false).await;
    let (status, got) = answer(&native, &body(STOP, 8, false)).await;
    assert_eq!(status, 200, "{got}");
}

/// OH-01 (ops-gpu-smoke run3-A): a killed tokenizer sidecar is reported at once, without waiting for the next 5 s
/// probe, as the advisory `sidecar` problem "respawning": its successor starts at once, so the worker stays ready and
/// a request in that window waits for the respawned child and is served; once the respawn finishes `/readyz` drops
/// the problem at once (no stale snapshot from a probe that raced the respawn). The respawn is held while `hold`
/// exists (the stub's `--hold-start-while`), so the window is deterministic. Fails before the fix: the first probe
/// after the kill published `sidecar` not ready (503), the request was refused `not_ready`, and nothing republished
/// when the respawn finished.
#[tokio::test]
async fn oh01_killed_tokenizer_sidecar_respawns_at_once_and_serves_requests_meanwhile() {
    let t = Tmp::new("oh01-kill");
    let hold = t.join("hold");
    let native = Native::ready(Case {
        sidecar_args: vec!["--hold-start-while".to_owned(), hold.display().to_string()],
        ..Case::matching()
    })
    .await;
    let started = sidecar_starts(&native);
    std::fs::write(&hold, b"").unwrap();
    let pid = tokenizer_sidecar_pid(&native);
    // SAFETY: `kill` takes a pid and a signal and touches no memory; the pid is this test's cered's child.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
    let (status, readyz) = sidecar_problem_listed(&native).await;
    assert_eq!(status, 200, "{readyz}");
    let detail = problem(&readyz, "sidecar").unwrap();
    assert!(detail.contains("respawning"), "{readyz}");
    let release = async {
        // The request is in by now: it waits for the held respawn.
        tokio::time::sleep(Duration::from_millis(500)).await;
        std::fs::remove_file(&hold).unwrap();
    };
    let request = body(STOP, 8, false);
    let ((status, got), ()) = tokio::join!(answer(&native, &request), release);
    assert_eq!(status, 200, "{got}");
    let served = Instant::now();
    let readyz = sidecar_readyz_until(&native, false).await;
    // O368: under the 5 s probe period (a republish by the next probe fails it), widened from 2 s, which a host at
    // load ~16 missed.
    assert!(
        served.elapsed() < Duration::from_secs(4),
        "the respawn was not republished at once: {readyz}"
    );
    assert_eq!(sidecar_starts(&native), started + 1);
    // O368: `cered`'s stderr is read by a pump, so its respawn line may land after the request is served (gate
    // `e9341cc0`, load ~16): wait for it.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let log = native.cered.stderr_so_far();
        if log.contains("cered: the tokenizer sidecar respawned (respawn 1)") {
            break;
        }
        assert!(Instant::now() < deadline, "{log}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// C7: a `required` and a named `tool_choice` constrain the output to the qwen3_coder structural tag (F11/F12:
/// the detector supports structural tags), so the output is the model-native call markup and SGLang parses it with
/// the qwen3_coder detector (`serving_chat.py` `_process_tool_calls`, streaming `use_native_parser`), not as a JSON
/// array: `tool_calls`, finish `tool_calls`, unary `matched_stop` null, the stream's calls carry the same name and
/// arguments.
#[tokio::test]
async fn c7_required_and_named_tool_choice_answer_tool_calls() {
    let native = Native::ready(with_tokenizer(grammar_tokenizer(&[
        (
            100,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
        ),
        (
            101,
            "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
        ),
    ])))
    .await;
    // The stub's tool grammar runs the script in the first tool's description (named: cered's filtered list).
    let mut tools = tools();
    tools[0]["function"]["description"] = json!("seq:100,101,7");
    for choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "get_weather"}}),
    ] {
        let mut request = body(TEXT, 8, false);
        request["tools"] = tools.clone();
        request["tool_choice"] = choice.clone();
        let (status, got) = answer(&native, &request).await;
        assert_eq!(status, 200, "{choice}: {got}");
        let c = &got["choices"][0];
        assert_eq!(c["finish_reason"], "tool_calls", "{choice}: {got}");
        assert_eq!(c["matched_stop"], Value::Null, "{choice}: {got}");
        let calls = c["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1, "{got}");
        assert_call_id(&calls[0]["id"]);
        assert_eq!(calls[0]["index"], 0);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], CALL_ARGUMENTS);

        request["stream"] = json!(true);
        let (events, terminal) = streamed(&native, &request).await;
        assert_eq!(
            terminal["choices"][0]["finish_reason"], "tool_calls",
            "{terminal}"
        );
        let message = assembled(&stream_choices(&events));
        assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            CALL_ARGUMENTS
        );
        assert_call_id(&message["tool_calls"][0]["id"]);
        assert_eq!(
            message["content"], "",
            "{choice}: no markup in content: {message}"
        );
    }
}

/// D15 / ledger CR20: with `enable_thinking` the grammar is deferred past `</think>` (the stub's reasoner contract,
/// SGLang's `ReasonerGrammarBackend`): the reasoning runs under all-ones masks (`mask=70000`) and the toy's own
/// tokens, the token after `</think>` under the grammar's initial mask. The thinking prompt makes the toy answer
/// `[2, 780, ...]`; this tokenizer spells 2 `ab` and 780 `</think>`.
#[tokio::test]
async fn d15_thinking_defers_the_grammar_past_the_think_end() {
    let t = Tmp::new("d15-thinking");
    let (case, path) = recording(&t);
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[(2, "ab"), (780, "</think>")])),
        capabilities_patch: json!({"thinking_toggle": true}),
        sidecar_args: vec!["--think-end-id".to_owned(), "780".to_owned()],
        ..case
    })
    .await;
    let mut request = constrained("seq:100,7", 8, false);
    request["chat_template_kwargs"] = json!({"enable_thinking": true});
    let before = steps(&path).len();
    let (status, got) = answer(&native, &request).await;
    assert_eq!(status, 200, "{got}");
    let sessions = by_session(&steps(&path)[before..]);
    assert_eq!(sessions.len(), 1, "{sessions:?}");
    let s = &sessions[0];
    assert_eq!(ids_of(s), [2, 780, 100, 7], "{s:?}");
    let masks: Vec<&str> = s.iter().map(|step| step["mask"].as_str()).collect();
    let all = FIXTURE_VOCAB.to_string();
    assert_eq!(masks, [all.as_str(), all.as_str(), "1", "1"], "{s:?}");
    let c = &got["choices"][0];
    assert_eq!(c["message"]["reasoning_content"], "ab", "{got}");
    assert_eq!(c["message"]["content"], "[100]", "{got}");
    assert_eq!(
        (&c["finish_reason"], &c["matched_stop"]),
        (&json!("stop"), &json!(7)),
        "{got}"
    );
}

/// D15 (`min_tokens` + constraint): native has no `min_tokens` (M2.12 refuses it before the M2.7 logits hook),
/// so the combination is refused at validation, naming `min_tokens`, and never reaches the grammar process.
#[tokio::test]
async fn d15_min_tokens_with_a_constraint_is_refused_on_native() {
    let native = Native::ready(with_tokenizer(grammar_tokenizer(&[]))).await;
    let mut request = constrained("seq:100,7", 8, false);
    request["min_tokens"] = json!(2);
    let (status, got) = answer(&native, &request).await;
    assert_eq!(status, 400, "{got}");
    assert_eq!(got["error"]["param"], "min_tokens", "{got}");
    assert_eq!(native.submissions().await, 0);
}

/// C7b (ledger CR5, second of M2.6/M2.8): two constrained requests run concurrently in decode rounds (their
/// recorded steps interleave), each under its own grammar, each answer its own script. The toy holds each session
/// after its first token while its gate file exists (`hold_ms` past the test's own 60 s bound); the test removes the
/// gate once both sessions recorded that token, so neither runs on before the other started, however late the
/// second request arrives.
#[tokio::test]
async fn c7b_concurrent_constrained_requests_are_each_valid() {
    let t = Tmp::new("c7b-rows");
    let (case, path) = recording(&t);
    let gate = t.join("hold.gate");
    std::fs::write(&gate, "").unwrap();
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[])),
        device_id: format!(
            "{},hold_after=1,hold_ms=120000,hold_gate={}",
            case.device_id,
            gate.display()
        ),
        ..case
    })
    .await;
    // Long scripts: both requests decode at the same time for most of their tokens.
    let script = |base: u32| {
        let ids: Vec<String> = (0..24).map(|i| (base + 100 * i).to_string()).collect();
        format!("seq:{},7", ids.join(","))
    };
    let before = steps(&path).len();
    let open_gate = async {
        let bound = std::time::Instant::now() + Duration::from_secs(60);
        while by_session(&steps(&path)[before..]).len() < 2 {
            assert!(
                std::time::Instant::now() < bound,
                "the two sessions never both recorded their first token"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        std::fs::remove_file(&gate).unwrap();
    };
    let (a, b, ()) = tokio::join!(
        native.chat(&constrained(&script(100), 32, false)).send(),
        native.chat(&constrained(&script(101), 32, false)).send(),
        open_gate,
    );
    let mut contents = Vec::new();
    for response in [a, b] {
        let response = response.unwrap();
        assert_eq!(response.status(), 200);
        let got: Value = response.json().await.unwrap();
        assert_eq!(got["choices"][0]["matched_stop"], 7, "{got}");
        contents.push(
            got["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_owned(),
        );
    }
    contents.sort();
    let text = |base: u32| -> String {
        use std::fmt::Write as _;
        (0..24).fold(String::new(), |mut s, i| {
            write!(s, "[{}]", base + 100 * i).unwrap();
            s
        })
    };
    assert_eq!(contents, [text(100), text(101)]);
    let all = steps(&path).split_off(before);
    let sessions = by_session(&all);
    assert_eq!(sessions.len(), 2, "{sessions:?}");
    for s in &sessions {
        assert!(s.iter().all(|step| step["mask"] == "1"), "{s:?}");
    }
    // Interleaved: some step of the second session comes before the last step of the first.
    let first = &all[0]["session"];
    let last_of_first = all.iter().rposition(|s| &s["session"] == first).unwrap();
    assert!(
        all[..last_of_first].iter().any(|s| &s["session"] != first),
        "{all:?}"
    );
}

/// `case` on native-g0-prefix.toml's prefix cache (D9C: 16 state slots, `cache_salt` isolation).
fn cache_on(case: Case) -> Case {
    Case {
        prefix_state_slots: Some(16),
        ..case
    }
}

/// M2.10 C5: a two-message request: a user history of `history` ids (`[1]`, the sentinel, then `x`s) and a last
/// user message of `last` ids starting with `tail`. The stub's `last_message_start` is `history`.
fn two_messages(history: usize, last: usize, tail: char, max_tokens: u64) -> Value {
    let mut h = SENTINEL.to_owned();
    while h.len() < history - 1 {
        h.push('x');
    }
    let mut l = tail.to_string();
    l.push_str(&"q".repeat(last - 2));
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": h}, {"role": "user", "content": l}],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
}

/// A unary chat as `token`'s scope: the `cached_tokens` of the usage and of the `cere` block, which agree.
async fn cached(native: &Native, token: &str, body: &Value) -> u64 {
    let response = http()
        .post(native.cered.url("/v1/chat/completions"))
        .bearer_auth(token)
        .json(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let v: Value = response.json().await.unwrap();
    let usage = v["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("{v}"));
    assert_eq!(v["cere"]["cached_tokens"], json!(usage), "{v}");
    usage
}

/// `POST /v1/cere/workers/{worker}/flush_cache` with `token` (none: no `Authorization`) and `body`.
async fn flush(native: &Native, worker: &str, token: Option<&str>, body: Value) -> (u16, Value) {
    let mut request = http().post(
        native
            .cered
            .url(&format!("/v1/cere/workers/{worker}/flush_cache")),
    );
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = request.json(&body).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// M2.10 C5 (D2, D3', D10): a repeated prompt hits its own scope's entry (`cached_tokens` = h in the usage and the
/// `cere` block); the other scope's namespace misses.
#[tokio::test]
async fn c5_prefix_hits_are_reported_and_scoped() {
    let native = Native::ready(cache_on(Case::matching())).await;
    let request = two_messages(9_000, 100, '1', 4);
    assert_eq!(cached(&native, TOKEN, &request).await, 0);
    // R3: the back-to-back repeat hits floor64(n - 1) of its n = 9,100 prompt tokens.
    assert_eq!(
        cached(&native, TOKEN, &request).await,
        (9_100 - 1) / 64 * 64
    );
    assert_eq!(cached(&native, "secret-b", &request).await, 0);
}

/// M2.10 C5 (D4 (c), W 132): a checkpoint at the last message's start (rounded down to 64) is taken only within
/// the limits (at least 8,192 tokens in, at most 1,024 tokens before the prompt end), so a request that shares
/// the history but not the last message hits it exactly then.
#[tokio::test]
async fn c5_last_message_checkpoint_only_within_limits() {
    let native = Native::ready(cache_on(Case::matching())).await;
    // (history, last message, expected hit of a request with another last message): outside the limits only
    // the 4,096-row call ends (D4 (b') (a)) below the history's end are checkpoints.
    for (history, last, hit) in [
        (9_000, 100, 8_960),
        (9_000, 1_100, 8_192),
        (8_000, 100, 4_096),
    ] {
        assert_eq!(
            cached(&native, TOKEN, &two_messages(history, last, '1', 4)).await,
            0
        );
        let other = cached(&native, TOKEN, &two_messages(history, last, '2', 4)).await;
        assert_eq!(other, hit, "history {history}, last {last}");
        let (status, body) = flush(&native, WORKER, Some(OPERATOR_TOKEN), json!({})).await;
        assert_eq!(status, 200, "{body}");
    }
}

/// M2.10 C5 (D11, N2 (a)): the flush route takes the operator token only, refuses an unknown worker and a bad
/// `timeout_s`, is 409 `worker_busy` while a ticket is in flight past `timeout_s`, and otherwise drops every
/// entry (the repeated prompt misses again).
#[tokio::test]
async fn c5_flush_route_auth_busy_and_flushed() {
    let native = Native::ready(cache_on(held())).await;
    let request = two_messages(9_000, 100, '1', 1);
    assert_eq!(cached(&native, TOKEN, &request).await, 0);
    assert!(cached(&native, TOKEN, &request).await > 0);

    assert_eq!(flush(&native, WORKER, None, json!({})).await.0, 401);
    assert_eq!(
        flush(&native, WORKER, Some("wrong"), json!({})).await.0,
        401
    );
    assert_eq!(flush(&native, WORKER, Some(TOKEN), json!({})).await.0, 403);
    assert_eq!(
        flush(&native, "gpu9", Some(OPERATOR_TOKEN), json!({}))
            .await
            .0,
        404
    );
    for bad in [
        json!({"timeout_s": 31}),
        json!({"timeout_s": -1}),
        json!({"wait": 1}),
    ] {
        assert_eq!(
            flush(&native, WORKER, Some(OPERATOR_TOKEN), bad.clone())
                .await
                .0,
            400,
            "{bad}"
        );
    }

    let response = native.chat(&body(TEXT, 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let held_rid = rid(&response);
    let mut reader = SseReader::new(response);
    read_to_hold(&mut reader).await;
    let (status, busy) = flush(
        &native,
        WORKER,
        Some(OPERATOR_TOKEN),
        json!({"timeout_s": 0}),
    )
    .await;
    assert_eq!(status, 409, "{busy}");
    assert_eq!(busy["error"]["code"], "worker_busy", "{busy}");
    cancel_request(&native, &held_rid).await;
    drop(reader);

    let (status, body) = flush(
        &native,
        WORKER,
        Some(OPERATOR_TOKEN),
        json!({"timeout_s": 30}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body,
        json!({"worker_id": WORKER, "prefix_cache": "flushed"})
    );
    assert_eq!(cached(&native, TOKEN, &request).await, 0);
}

/// Every key of one worker's `prefix_cache` counters (`GET /v1/cere/prefix_cache`), sorted.
const PREFIX_COUNTERS: [&str; 15] = [
    "alt_edges",
    "alt_edges_enabled",
    "capture_page_mismatch",
    "capture_skipped",
    "capture_skipped_by_site",
    "captures",
    "hit_shortfall_lookups",
    "hit_shortfall_tokens",
    "hits",
    "host_drops",
    "host_loads",
    "host_spills",
    "inserts",
    "leaf_evictions",
    "state_evictions",
];

/// The capture-skip sites of `capture_skipped_by_site` (PT0a), in their JSON order.
const SKIP_SITES: [&str; 3] = ["prefill_slot", "decode_track", "end_pair"];

/// `GET /v1/cere/prefix_cache` with `token` (none: no bearer): the status and the JSON body.
async fn prefix_cache(native: &Native, token: Option<&str>) -> (u16, Value) {
    let mut request = http().get(native.cered.url("/v1/cere/prefix_cache"));
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// The counters of `WORKER` in an operator read of `GET /v1/cere/prefix_cache`.
async fn prefix_counters(native: &Native) -> Value {
    let (status, body) = prefix_cache(native, Some(OPERATOR_TOKEN)).await;
    assert_eq!(status, 200, "{body}");
    let worker = &body["workers"][WORKER];
    assert_eq!(worker["loaded"], true, "{body}");
    assert_eq!(worker["reason"], Value::Null, "{body}");
    let counters = worker["prefix_cache"].clone();
    let mut found: Vec<&str> = counters
        .as_object()
        .unwrap_or_else(|| panic!("{body}"))
        .keys()
        .map(String::as_str)
        .collect();
    found.sort_unstable();
    assert_eq!(found, PREFIX_COUNTERS, "{body}");
    // PT0a: the sites split capture_skipped exactly.
    let sites = &counters["capture_skipped_by_site"];
    let mut keys: Vec<&str> = sites
        .as_object()
        .unwrap_or_else(|| panic!("{body}"))
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["decode_track", "end_pair", "prefill_slot"], "{body}");
    let sum: u64 = SKIP_SITES.iter().map(|s| sites[*s].as_u64().unwrap()).sum();
    assert_eq!(Some(sum), counters["capture_skipped"].as_u64(), "{body}");
    counters
}

/// The prefix cache counters of the single-form worker (prod's form, where `GET /v1/cere/cards` is `{}`): the route
/// is operator-only (401/403), starts at zero, counts a hit and a capture, never decreases (a flush keeps the
/// counters, another scope's request only adds), names no prompt or token, and reports the opt-in alt edges flag.
#[tokio::test]
async fn prefix_cache_counters_are_operator_only_and_monotonic() {
    let native = Native::ready(cache_on(Case::matching())).await;
    assert_eq!(prefix_cache(&native, None).await.0, 401);
    assert_eq!(prefix_cache(&native, Some("wrong")).await.0, 401);
    assert_eq!(prefix_cache(&native, Some(TOKEN)).await.0, 403);

    // Every counter, the three capture-skip sites included (PT0a).
    let counters = |v: &Value| -> Vec<u64> {
        PREFIX_COUNTERS
            .iter()
            .filter(|k| **k != "alt_edges_enabled")
            .flat_map(|k| match *k {
                "capture_skipped_by_site" => SKIP_SITES
                    .iter()
                    .map(|site| v[*k][*site].as_u64().unwrap())
                    .collect(),
                _ => vec![v[*k].as_u64().unwrap()],
            })
            .collect()
    };
    let fresh = prefix_counters(&native).await;
    assert_eq!(counters(&fresh), vec![0; 16], "{fresh}");
    assert_eq!(fresh["alt_edges_enabled"], false, "{fresh}");

    let request = two_messages(9_000, 100, '1', 4);
    assert_eq!(cached(&native, TOKEN, &request).await, 0);
    let first = prefix_counters(&native).await;
    assert!(first["captures"].as_u64().unwrap() >= 1, "{first}");
    assert!(first["inserts"].as_u64().unwrap() >= 1, "{first}");
    assert_eq!(first["hits"], 0, "{first}");
    assert!(cached(&native, TOKEN, &request).await > 0);
    let second = prefix_counters(&native).await;
    assert_eq!(second["hits"], 1, "{second}");
    assert_eq!(
        prefix_counters(&native).await,
        second,
        "a read changes nothing"
    );
    for (b, a) in counters(&first).iter().zip(counters(&second)) {
        assert!(a >= *b, "{first} -> {second}");
    }

    let (status, body) = flush(&native, WORKER, Some(OPERATOR_TOKEN), json!({})).await;
    assert_eq!(status, 200, "{body}");
    let flushed = prefix_counters(&native).await;
    assert_eq!(flushed, second, "a flush keeps the counters");
    assert_eq!(cached(&native, "secret-b", &request).await, 0);
    assert_eq!(cached(&native, TOKEN, &request).await, 0);
    let third = prefix_counters(&native).await;
    for (b, a) in counters(&flushed).iter().zip(counters(&third)) {
        assert!(a >= *b, "{flushed} -> {third}");
    }
    assert!(
        third["captures"].as_u64() > second["captures"].as_u64(),
        "{third}"
    );
    let text = prefix_cache(&native, Some(OPERATOR_TOKEN))
        .await
        .1
        .to_string();
    assert!(!text.contains(TOKEN) && !text.contains(OPERATOR_TOKEN) && !text.contains("xxxx"));
}

/// `prefix_alt_edges = true` shows as `alt_edges_enabled` (and the same zero counters before traffic).
#[tokio::test]
async fn prefix_cache_counters_report_the_alt_edges_flag() {
    let native = Native::ready(Case {
        prefix_alt_edges: true,
        ..cache_on(Case::matching())
    })
    .await;
    let fresh = prefix_counters(&native).await;
    assert_eq!(fresh["alt_edges_enabled"], true, "{fresh}");
    assert_eq!(fresh["alt_edges"], 0, "{fresh}");
}

/// Negative cases: a cache-off worker has no counters (`prefix_cache` null, reason `cache_off`), and a library from
/// before `cere_model_prefix_stats` still serves: `prefix_cache` null, reason `export_missing`, status 200.
#[tokio::test]
async fn prefix_cache_counters_are_null_without_a_cache_or_the_export() {
    let off = Native::ready(Case::matching()).await;
    let (status, body) = prefix_cache(&off, Some(OPERATOR_TOKEN)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body,
        json!({"workers": {WORKER: {"loaded": true, "prefix_cache": null, "reason": "cache_off"}}})
    );

    let old = native_support::toy_library_prestaging();
    let native = Native::ready(cache_on(Case {
        library_pin: native_server::file_sha256(&old),
        library: old,
        ..Case::matching()
    }))
    .await;
    let (status, body) = prefix_cache(&native, Some(OPERATOR_TOKEN)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body,
        json!({"workers": {WORKER: {
            "loaded": true, "prefix_cache": null, "reason": "export_missing"
        }}})
    );
    assert_eq!(
        cached(&native, TOKEN, &two_messages(9_000, 100, '1', 4)).await,
        0
    );
}

/// M2.10 C5 (D9C): the shipped native-g0 (`prefix_state_slots` absent = 0) runs the cache off: nothing hits and
/// the flush answers `"off"`.
#[tokio::test]
async fn c5_cache_off_never_hits_and_flush_is_off() {
    let native = Native::ready(Case::matching()).await;
    let request = two_messages(9_000, 100, '1', 4);
    assert_eq!(cached(&native, TOKEN, &request).await, 0);
    assert_eq!(cached(&native, TOKEN, &request).await, 0);
    let (status, body) = flush(&native, WORKER, Some(OPERATOR_TOKEN), json!({})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!({"worker_id": WORKER, "prefix_cache": "off"}));
}

/// m29 design note C8 (D11): the request battery of the speculation comparison, unary and stream bodies (SSE data
/// events, `[DONE]` as a string) in order: plain text, a stop token, stop strings that end inside a committed burst,
/// tool calls, logprobs with `top_logprobs`, and a penalised request (`NO_SPECULATION`, M29S2-08).
async fn speculation_battery(native: &Native) -> Vec<Value> {
    arm_battery(native, None).await
}

/// The battery with every request on A/B arm `arm` (ab-split design note AB-03, the operator's header).
async fn arm_battery(native: &Native, arm: Option<&str>) -> Vec<Value> {
    let send = |body: &Value| {
        let request = native.chat(body);
        match arm {
            Some(arm) => request.header("X-Cere-Ab-Arm", arm).header(
                "X-Cere-Operator-Authorization",
                format!("Bearer {OPERATOR_TOKEN}"),
            ),
            None => request,
        }
    };
    let mut requests = vec![body(TEXT, 6, false), body(STOP, 8, false)];
    for stop in ["[2]", "5][2"] {
        let mut b = body(TEXT, 16, false);
        b["stop"] = json!(["zzz", stop]);
        requests.push(b);
    }
    requests.push(tool_body(STOP, 8, false));
    let mut logprobs = body(TEXT, 6, false);
    logprobs["logprobs"] = json!(true);
    logprobs["top_logprobs"] = json!(2);
    requests.push(logprobs);
    let mut penalised = body(TEXT, 6, false);
    penalised["presence_penalty"] = json!(0.5);
    requests.push(penalised);
    let mut out = Vec::new();
    for unary in requests {
        let response = send(&unary).send().await.unwrap();
        assert_eq!(response.status(), 200, "{unary}");
        out.push(response.json().await.unwrap());
        let mut stream = unary.clone();
        stream["stream"] = json!(true);
        stream["stream_options"] = json!({"include_usage": true});
        let response = send(&stream).send().await.unwrap();
        assert_eq!(response.status(), 200, "{stream}");
        let events = SseReader::new(response).rest().await;
        check_stream(&events);
        out.push(Value::Array(
            events
                .into_iter()
                .map(|e| match e {
                    Event::Data(v) => v,
                    Event::Done => json!("[DONE]"),
                })
                .collect(),
        ));
    }
    out
}

/// The per-process values of a body: response and tool-call `id`s, `created`, and the `cere` extension's request
/// id, instance id and config hash (the speculation config differs by its draft tokens).
fn without_volatile(v: &mut Value) {
    match v {
        Value::Object(map) => {
            for (key, value) in map.iter_mut() {
                if ["id", "created", "request_id", "instance_id", "config_hash"]
                    .contains(&key.as_str())
                {
                    *value = Value::Null;
                } else {
                    without_volatile(value);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(without_volatile),
        _ => {}
    }
}

/// m29 design note C8 (D11): `cered` over the toy WITH speculation (the `-v19` profile's speculation fields,
/// 4 draft tokens) answers the battery with the spec-off run's unary and stream bodies, except the per-process
/// values: speculative greedy is bit-identical to plain greedy (tokens, stop strings inside a burst, tool calls,
/// usage, logprobs). `native_spec_verify_steps` advances only with speculation. Needs the HOST track's toy
/// speculation (build-loop-m29 design note C-ABI): a toy that refuses the drafts desc at ALLOCATE (UNSUPPORTED) leaves
/// the speculation load not ready, and the test says so on stderr.
#[tokio::test]
async fn speculation_bodies_equal_the_spec_off_run() {
    let tokenizer = tool_tokenizer(&[(2240, CALL_TEXT)]);
    let on = Native::start(Case {
        speculation: true,
        ..with_tokenizer(tokenizer.clone())
    })
    .await;
    let (status, ready) = on.settled().await;
    assert_eq!(status, 200, "{ready}");
    let off = Native::ready(with_tokenizer(tokenizer)).await;
    let mut plain = speculation_battery(&off).await;
    assert_eq!(off.spec_counters().await, (0, 0));
    let mut speculated = speculation_battery(&on).await;
    plain.iter_mut().for_each(without_volatile);
    speculated.iter_mut().for_each(without_volatile);
    for (i, (s, p)) in speculated.iter().zip(&plain).enumerate() {
        assert_eq!(s, p, "battery body {i}");
    }
    assert_eq!(speculated.len(), plain.len());
    let (steps, accepted) = on.spec_counters().await;
    assert!(steps > 0, "native_spec_verify_steps {steps}");
    assert!(
        accepted <= 4 * steps,
        "{accepted} accepted over {steps} rounds"
    );
}

/// m29l design note C5L (D4, D6): `cered` over the periodic toy (`cycle=5`, so the suffix lookup matches; and
/// `draft_miss=1`, so the MTP's first proposal is always wrong) with the `drafter` lever (m23 design note PI-02)
/// "mtp+lookup" and "lookup" answers the battery with the spec-off run's bodies, and lookup drafts are accepted
/// (`native_spec_accepted_drafts` > 0).
#[tokio::test]
async fn lookup_drafts_bodies_equal_the_spec_off_run() {
    let tokenizer = tool_tokenizer(&[(2240, CALL_TEXT)]);
    let device = "toy,cycle=5,draft_miss=1";
    let off = Native::ready(Case {
        device_id: device.to_owned(),
        ..with_tokenizer(tokenizer.clone())
    })
    .await;
    let mut plain = speculation_battery(&off).await;
    plain.iter_mut().for_each(without_volatile);
    let drafters: [&'static [(&str, &str)]; 2] =
        [&[("drafter", "mtp+lookup")], &[("drafter", "lookup")]];
    for levers in drafters {
        let drafts = levers[0].1;
        let on = Native::start(Case {
            device_id: device.to_owned(),
            speculation: true,
            levers,
            ..with_tokenizer(tokenizer.clone())
        })
        .await;
        let (status, ready) = on.settled().await;
        assert_eq!(status, 200, "{drafts}: {ready}");
        let mut speculated = speculation_battery(&on).await;
        speculated.iter_mut().for_each(without_volatile);
        assert_eq!(speculated.len(), plain.len());
        for (i, (s, p)) in speculated.iter().zip(&plain).enumerate() {
            assert_eq!(s, p, "{drafts}: battery body {i}");
        }
        let (steps, accepted) = on.spec_counters().await;
        assert!(steps > 0, "{drafts}: native_spec_verify_steps {steps}");
        assert!(
            accepted > 0,
            "{drafts}: no lookup draft accepted over {steps} rounds"
        );
    }
}

/// m23 design note PI-05: every registry lever at its default, as `GET /v1/cere/stats` reports it.
fn default_levers() -> Value {
    json!({"k2_gemm": "cublaslt", "prefill_attention": "cuda_core", "decode_step": "eager",
        "moe_gemm": "p3b", "draft_tokens": "max", "verify_attention": "per_row", "k2_weights": "bf16",
        "drafter": "mtp", "draft_policy": "fixed", "grammar_drafts": "off", "sdar_gemm": "identical"})
}

/// A case serving the dev hot swap (PI-04): `lever_hot_swap` (the shipped config has the operator token).
fn hot_swap(case: Case) -> Case {
    Case {
        lever_hot_swap: true,
        ..case
    }
}

fn error_code(body: &Value) -> &str {
    body["error"]["code"].as_str().unwrap_or_default()
}

/// m23 design note PI-02/PI-03/PI-05: a profile's `[deployments.native.levers]` reach the executor through
/// `cere_model_desc.selection`, and the stats report the set `cere_model_selection` reads back; a profile without levers
/// reports every default.
#[tokio::test]
async fn profile_levers_reach_the_executor() {
    let plain = Native::ready(Case::matching()).await;
    assert_eq!(plain.stats_levers().await, default_levers());
    drop(plain);
    let native = Native::ready(Case {
        levers: &[
            ("k2_gemm", "small_m"),
            ("prefill_attention", "p4"),
            ("k2_weights", "fp8"),
        ],
        ..Case::matching()
    })
    .await;
    let mut want = default_levers();
    want["k2_gemm"] = json!("small_m");
    want["prefill_attention"] = json!("p4");
    want["k2_weights"] = json!("fp8");
    assert_eq!(native.stats_levers().await, want);
    let response = native.chat(&body(STOP, 8, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.terminal(&rid(&response)).await["state"], "completed");
}

/// PI-04: `GET /v1/cere/levers` is the registry fixture, the active set and the flag; without `lever_hot_swap` the
/// swap is 403 `lever_hot_swap_disabled`, a client token 403 `forbidden`, and nothing changes.
#[tokio::test]
async fn hot_swap_is_refused_without_the_flag() {
    let native = Native::ready(Case::matching()).await;
    let (status, body) = native.levers().await;
    assert_eq!(status, 200, "{body}");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/lever_registry.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(body["registry"], fixture["levers"]);
    assert_eq!(body["active"], default_levers());
    assert_eq!(body["hot_swap"], false);
    let swap = json!({"levers": {"k2_gemm": "small_m"}});
    let (status, body) = native.set_levers(OPERATOR_TOKEN, &swap).await;
    assert_eq!(
        (status, error_code(&body)),
        (403, "lever_hot_swap_disabled"),
        "{body}"
    );
    let (status, body) = native.set_levers(TOKEN, &swap).await;
    assert_eq!((status, error_code(&body)), (403, "forbidden"), "{body}");
    assert_eq!(native.stats_levers().await, default_levers());
}

/// PI-04: a swap while a request is in flight (the toy holds it mid-stream) is 409 `requests_in_flight` and
/// changes nothing; once the request is terminal the same swap succeeds.
#[tokio::test]
async fn hot_swap_is_refused_while_a_request_is_in_flight() {
    let native = Native::ready(hot_swap(held())).await;
    let response = native.chat(&body(TEXT, 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = rid(&response);
    let mut reader = SseReader::new(response);
    read_to_hold(&mut reader).await;
    let swap = json!({"levers": {"k2_gemm": "small_m"}});
    let (status, body) = native.set_levers(OPERATOR_TOKEN, &swap).await;
    assert_eq!(
        (status, error_code(&body)),
        (409, "requests_in_flight"),
        "{body}"
    );
    assert_eq!(native.stats_levers().await, default_levers());
    let cancel = http()
        .post(native.cered.url(&format!("/v1/cere/requests/{rid}/cancel")))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    reader.rest().await;
    assert_eq!(native.terminal(&rid).await["state"], "cancelled");
    // The admission slot is released after the terminal is visible, so the swap may still see it briefly.
    let mut swapped = None;
    for _ in 0..200 {
        let (status, body) = native.set_levers(OPERATOR_TOKEN, &swap).await;
        if status == 409 && error_code(&body) == "requests_in_flight" {
            tokio::time::sleep(Duration::from_millis(25)).await;
            continue;
        }
        swapped = Some((status, body));
        break;
    }
    let (status, body) = swapped.expect("the request never left flight");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["active"]["k2_gemm"], "small_m");
}

/// PI-04: the swap refusals change nothing: 409 `lever_restart_only` for a restart-class lever (`moe_gemm` needs
/// another library, `k2_weights` is converted once at create, `sdar_gemm` is fixed at create), 400 `invalid_lever`
/// for an unknown lever or impl, a value the model cannot run (drafts without speculation) or a malformed body.
/// Re-sending a restart lever's active value is allowed.
#[tokio::test]
async fn hot_swap_refusals_change_nothing() {
    let native = Native::ready(hot_swap(Case::matching())).await;
    for restart in [
        json!({"levers": {"moe_gemm": "k7"}}),
        json!({"levers": {"k2_weights": "fp8"}}),
        json!({"levers": {"sdar_gemm": "changing"}}),
        json!({"levers": {"k2_gemm": "small_m", "k2_weights": "fp8"}}),
    ] {
        let (status, body) = native.set_levers(OPERATOR_TOKEN, &restart).await;
        assert_eq!(
            (status, error_code(&body)),
            (409, "lever_restart_only"),
            "{restart}: {body}"
        );
    }
    for bad in [
        json!({"levers": {"k2": "small_m"}}),
        json!({"levers": {"k2_gemm": "split_k9"}}),
        json!({"levers": {"drafter": "lookup"}}),
        json!({"levers": {"draft_tokens": "3"}}),
        json!({"levers": {"k2_gemm": 1}}),
        json!({"k2_gemm": "small_m"}),
        // One bad entry refuses the whole set.
        json!({"levers": {"k2_gemm": "small_m", "prefill_attention": "p5"}}),
    ] {
        let (status, body) = native.set_levers(OPERATOR_TOKEN, &bad).await;
        assert_eq!(
            (status, error_code(&body)),
            (400, "invalid_lever"),
            "{bad}: {body}"
        );
    }
    assert_eq!(native.stats_levers().await, default_levers());
    let (status, body) = native
        .set_levers(
            OPERATOR_TOKEN,
            &json!({"levers": {"moe_gemm": "p3b", "k2_weights": "bf16"}}),
        )
        .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["active"], default_levers());
}

/// PI-04/PI-05: a successful swap answers the set the executor reads back, the stats and `GET /v1/cere/levers`
/// report it, a partial swap leaves the other levers, and requests keep completing after each swap.
#[tokio::test]
async fn hot_swap_changes_the_active_set() {
    let native = Native::ready(hot_swap(Case::matching())).await;
    let (status, answer) = native
        .set_levers(
            OPERATOR_TOKEN,
            &json!({"levers": {"k2_gemm": "small_m", "prefill_attention": "p4"}}),
        )
        .await;
    assert_eq!(status, 200, "{answer}");
    let mut want = default_levers();
    want["k2_gemm"] = json!("small_m");
    want["prefill_attention"] = json!("p4");
    assert_eq!(answer["active"], want);
    assert_eq!(native.stats_levers().await, want);
    let (status, read) = native.levers().await;
    assert_eq!(status, 200, "{read}");
    assert_eq!((&read["active"], &read["hot_swap"]), (&want, &json!(true)));
    let response = native.chat(&body(STOP, 8, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.terminal(&rid(&response)).await["state"], "completed");
    // O368: the terminal state is visible before admission releases the slot, and a swap with the slot still held
    // is 409 `requests_in_flight`: wait for the in-flight count to reach 0.
    native.idle().await;
    let (status, answer) = native
        .set_levers(OPERATOR_TOKEN, &json!({"levers": {"k2_gemm": "cublaslt"}}))
        .await;
    assert_eq!(status, 200, "{answer}");
    want["k2_gemm"] = json!("cublaslt");
    assert_eq!(answer["active"], want);
    assert_eq!(native.stats_levers().await, want);
}

/// SelfTune D6 (O292 D-9, ST2 (c)): `params` on `POST /v1/cere/levers`. A launch param in its registry range swaps
/// (with or without an impl change of its lever); a build param is 409 `lever_build_param`; an out-of-range, unknown
/// or malformed param, or a body key other than `levers` / `params`, is 400 `invalid_lever`; every refusal changes
/// nothing.
#[tokio::test]
async fn hot_swap_params_are_checked_against_the_registry() {
    let native = Native::ready(hot_swap(Case {
        speculation: true,
        ..Case::matching()
    }))
    .await;
    let before = native.stats_levers().await;
    let (status, refused) = native
        .set_levers(
            OPERATOR_TOKEN,
            &json!({"levers": {"k2_gemm": "small_m"}, "params": {"k2_gemm": {"k_step": 64}}}),
        )
        .await;
    assert_eq!(
        (status, error_code(&refused)),
        (409, "lever_build_param"),
        "{refused}"
    );
    for bad in [
        json!({"levers": {"draft_tokens": "3"}, "params": {"draft_tokens": {"window": 4}}}),
        json!({"levers": {"draft_tokens": "3"}, "params": {"draft_tokens": {"tile": 1}}}),
        json!({"levers": {}, "params": {"turbo": {"window": 3}}}),
        json!({"levers": {}, "params": {"draft_tokens": {"window": -1}}}),
        json!({"levers": {}, "params": {"draft_tokens": {"window": "4"}}}),
        json!({"levers": {}, "params": {"draft_tokens": 4}}),
        json!({"levers": {}, "params": [4]}),
        json!({"params": {"draft_tokens": {"window": 4}}}),
        json!({"levers": {}, "params": {}, "budget": 1}),
    ] {
        let (status, refused) = native.set_levers(OPERATOR_TOKEN, &bad).await;
        assert_eq!(
            (status, error_code(&refused)),
            (400, "invalid_lever"),
            "{bad}: {refused}"
        );
    }
    assert_eq!(native.stats_levers().await, before);
    let (status, answer) = native
        .set_levers(
            OPERATOR_TOKEN,
            &json!({"levers": {"draft_tokens": "3"}, "params": {"draft_tokens": {"window": 3}}}),
        )
        .await;
    assert_eq!(status, 200, "{answer}");
    let mut want = before.clone();
    want["draft_tokens"] = json!("3");
    assert_eq!(answer["active"], want);
    let (status, answer) = native
        .set_levers(
            OPERATOR_TOKEN,
            &json!({"levers": {}, "params": {"draft_tokens": {"window": 3}}}),
        )
        .await;
    assert_eq!((status, &answer["active"]), (200, &want), "{answer}");
    assert_eq!(native.stats_levers().await, want);
    let response = native.chat(&body(STOP, 8, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.terminal(&rid(&response)).await["state"], "completed");
}

/// m29a design note A4: `cered` over the toy with the draft-policy lever "adaptive" (alone and with "mtp+lookup")
/// answers the battery with the spec-off run's bodies, on proposals always right and on a first proposal always wrong
/// (`draft_miss=1`, where the policy stops verifying drafts that keep being rejected).
#[tokio::test]
async fn adaptive_draft_policy_bodies_equal_the_spec_off_run() {
    let tokenizer = tool_tokenizer(&[(2240, CALL_TEXT)]);
    for device in ["toy", "toy,draft_miss=1", "toy,cycle=5,draft_miss=2"] {
        let off = Native::ready(Case {
            device_id: device.to_owned(),
            ..with_tokenizer(tokenizer.clone())
        })
        .await;
        let mut plain = speculation_battery(&off).await;
        plain.iter_mut().for_each(without_volatile);
        for levers in [
            &[("draft_policy", "adaptive")][..],
            &[("draft_policy", "adaptive"), ("drafter", "mtp+lookup")][..],
        ] {
            let on = Native::start(Case {
                device_id: device.to_owned(),
                speculation: true,
                levers,
                ..with_tokenizer(tokenizer.clone())
            })
            .await;
            let (status, ready) = on.settled().await;
            assert_eq!(status, 200, "{device} {levers:?}: {ready}");
            let mut speculated = speculation_battery(&on).await;
            speculated.iter_mut().for_each(without_volatile);
            assert_eq!(speculated.len(), plain.len());
            for (i, (s, p)) in speculated.iter().zip(&plain).enumerate() {
                assert_eq!(s, p, "{device} {levers:?}: battery body {i}");
            }
            let (steps, accepted) = on.spec_counters().await;
            assert!(
                steps > 0,
                "{device} {levers:?}: native_spec_verify_steps {steps}"
            );
            assert!(
                accepted <= 3 * steps,
                "{accepted} accepted over {steps} rounds"
            );
        }
    }
}

/// batch-concurrency design note BC1 / BC-D4 (F11, fail-first): `cered` over toys that time their rounds
/// (`timed_rounds=1`, as the CUDA device does) with the -v52 levers ("mtp+lookup", "adaptive": a sampled request is a
/// `CERE_WORK_LOOKUP_DRAFTS` ticket, serve.rs `draft_flags`) answers each seeded sampled request with the same
/// choices and usage one request at a time (c1) and as 4 concurrent requests sharing every round (c4), twice; the 4
/// seeds give 4 different answers, and drafts were verified and accepted.
///
/// O368: the seal times the toy's rounds into the adaptive policy's cost table (O95), so on a loaded host a noisy
/// table can price every draft out (train CI `ada85954`, load ~19: 0 accepted over 564 rounds). Drafting is this
/// test's precondition, not its subject: a load whose c1 pass accepted no draft is restarted (the seal re-times the
/// rounds), at most 5 loads, each failed load's `spec_round_costs` line named if all fail.
#[tokio::test]
async fn sampled_drafting_requests_repeat_at_c1_and_c4() {
    let levers: &'static [(&str, &str)] =
        &[("drafter", "mtp+lookup"), ("draft_policy", "adaptive")];
    let seeds = [11u64, 12, 13, 14];
    for device in ["toy,timed_rounds=1", "toy,timed_rounds=1,cycle=5"] {
        let mut priced_out = Vec::new();
        let (native, alone) = loop {
            let native = Native::start(Case {
                device_id: device.to_owned(),
                speculation: true,
                levers,
                ..with_tokenizer(distinct_tokenizer(&[]))
            })
            .await;
            let (status, ready) = native.settled().await;
            assert_eq!(status, 200, "{device}: {ready}");
            let mut alone = Vec::new();
            for seed in seeds {
                alone.push(sampled_answer(&native, seed).await);
            }
            let (steps, accepted) = native.spec_counters().await;
            if accepted > 0 {
                break (native, alone);
            }
            let stderr = native.cered.stderr_so_far();
            let costs = stderr
                .lines()
                .find(|l| l.contains("spec_round_costs"))
                .unwrap_or("no spec_round_costs line");
            priced_out.push(format!("0 accepted over {steps} rounds ({costs})"));
            assert!(
                priced_out.len() < 5,
                "{device}: no draft accepted on 5 loads: {priced_out:?}"
            );
        };
        for pass in 0..2 {
            let together = futures_util::future::join_all(
                seeds.iter().map(|seed| sampled_answer(&native, *seed)),
            )
            .await;
            for (i, (c4, c1)) in together.iter().zip(&alone).enumerate() {
                assert_eq!(c4, c1, "{device}: seed {} at c4 (pass {pass})", seeds[i]);
            }
        }
        let distinct: BTreeSet<String> = alone
            .iter()
            .map(|(choices, _)| choices.to_string())
            .collect();
        assert_eq!(
            distinct.len(),
            seeds.len(),
            "{device}: seeds repeat answers"
        );
        let (steps, accepted) = native.spec_counters().await;
        assert!(
            steps > 0 && accepted > 0,
            "{device}: {accepted} accepted over {steps} rounds"
        );
    }
}

/// `choices` and `usage` of a 48-token sampled request at temperature 1 with `seed`.
async fn sampled_answer(native: &Native, seed: u64) -> (Value, Value) {
    let response = native
        .chat(&sampled(48, json!({"temperature": 1, "seed": seed})))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    (body["choices"].clone(), body["usage"].clone())
}

/// ab-split design note AB-01: two arms of the speculation profile, the fixed and the adaptive draft policy.
fn ab_arms(salt: &str, fixed: u32, adaptive: u32) -> Value {
    json!({"salt": salt, "arms": [
        {"id": "fixed", "weight": fixed, "levers": {"draft_policy": "fixed"}},
        {"id": "adaptive", "weight": adaptive, "levers": {"draft_policy": "adaptive"}},
    ]})
}

/// `usage.completion_tokens` of a battery body: a unary body's, or a stream's usage chunk.
fn battery_completion_tokens(body: &Value) -> u64 {
    match body {
        Value::Array(events) => events
            .iter()
            .find_map(|e| e["usage"]["completion_tokens"].as_u64())
            .unwrap_or_else(|| panic!("a stream without usage: {body}")),
        _ => body["usage"]["completion_tokens"].as_u64().unwrap(),
    }
}

/// AB-01/AB-03/AB-04 (fail-first): every request on either arm (named by the operator) answers exactly the spec-off
/// run's bodies (an arm changes speed, never output), and `GET /v1/cere/stats` `ab` counts each arm's requests,
/// completion tokens, speculate rounds, TTFT, decode rate and accept length; the arms' rounds add up to the
/// process counter. A deployment without arms reports no `ab`.
#[tokio::test]
async fn ab_arms_answer_like_the_spec_off_run_and_report_per_arm_stats() {
    let tokenizer = tool_tokenizer(&[(2240, CALL_TEXT)]);
    let device = "toy,draft_miss=1";
    let off = Native::ready(Case {
        device_id: device.to_owned(),
        ..with_tokenizer(tokenizer.clone())
    })
    .await;
    let mut plain = speculation_battery(&off).await;
    plain.iter_mut().for_each(without_volatile);
    assert_eq!(off.stats_ab().await, Value::Null, "no arms, no `ab`");
    drop(off);
    let on = Native::ready(Case {
        device_id: device.to_owned(),
        speculation: true,
        ab: ab_arms("exp-1", 1, 1),
        ..with_tokenizer(tokenizer)
    })
    .await;
    let empty = on.stats_ab().await;
    for arm in ["fixed", "adaptive"] {
        let s = &empty[MODEL][arm];
        assert_eq!(
            (&s["requests"], &s["weight"]),
            (&json!(0), &json!(1)),
            "{empty}"
        );
        assert_eq!(
            s["ttft_ms"],
            json!({"count": 0, "mean": null, "p50": null, "p99": null})
        );
    }
    let (mut steps, mut accepted) = (0, 0);
    for arm in ["fixed", "adaptive"] {
        let mut bodies = arm_battery(&on, Some(arm)).await;
        let tokens: u64 = bodies.iter().map(battery_completion_tokens).sum();
        bodies.iter_mut().for_each(without_volatile);
        assert_eq!(bodies.len(), plain.len());
        for (i, (s, p)) in bodies.iter().zip(&plain).enumerate() {
            assert_eq!(s, p, "arm {arm}: battery body {i}");
        }
        let stats = on.stats_ab().await;
        let s = &stats[MODEL][arm];
        assert_eq!(s["levers"], json!({"draft_policy": arm}), "{stats}");
        let n = plain.len() as u64;
        assert_eq!(
            [
                &s["requests"],
                &s["completed"],
                &s["cancelled"],
                &s["errors"]
            ],
            [&json!(n), &json!(n), &json!(0), &json!(0)],
            "{stats}"
        );
        assert_eq!(s["completion_tokens"], tokens, "{stats}");
        assert_eq!(s["ttft_ms"]["count"], n, "{stats}");
        assert!(s["ttft_ms"]["p50"].as_f64().unwrap() >= 0.0, "{stats}");
        assert!(s["decode_tok_s"]["count"].as_u64().unwrap() > 0, "{stats}");
        assert!(s["accept_length"]["count"].as_u64().unwrap() > 0, "{stats}");
        assert!(
            s["accept_length"]["mean"].as_f64().unwrap() >= 1.0,
            "{stats}"
        );
        steps += s["spec_verify_steps"].as_u64().unwrap();
        accepted += s["spec_accepted_drafts"].as_u64().unwrap();
    }
    assert!(steps > 0);
    assert_eq!(on.spec_counters().await, (steps, accepted));
    // The first arm's stats did not move while the second ran.
    assert_eq!(on.stats_ab().await[MODEL]["fixed"]["requests"], plain.len());
}

/// AB-03: naming an arm needs the operator token in `X-Cere-Operator-Authorization` (403 `forbidden` without it or
/// with a client token), an unconfigured arm or a deployment without arms is 400 `invalid_value`, and a refused
/// request counts on no arm.
#[tokio::test]
async fn naming_an_ab_arm_needs_the_operator_token() {
    let native = Native::ready(Case {
        speculation: true,
        ab: ab_arms("exp-1", 1, 1),
        ..Case::matching()
    })
    .await;
    let named = |native: &Native, arm: &str, operator: Option<&str>| {
        let request = native
            .chat(&body(STOP, 8, false))
            .header("X-Cere-Ab-Arm", arm);
        match operator {
            Some(token) => {
                request.header("X-Cere-Operator-Authorization", format!("Bearer {token}"))
            }
            None => request,
        }
    };
    for (arm, operator, status, code) in [
        ("fixed", None, 403, "forbidden"),
        ("fixed", Some(TOKEN), 403, "forbidden"),
        ("fixed", Some("not-a-token"), 403, "forbidden"),
        ("control", Some(OPERATOR_TOKEN), 400, "invalid_value"),
    ] {
        let response = named(&native, arm, operator).send().await.unwrap();
        assert_eq!(response.status(), status, "{arm} {operator:?}");
        let error: Value = response.json().await.unwrap();
        assert_eq!(error_code(&error), code, "{error}");
    }
    let stats = native.stats_ab().await;
    for arm in ["fixed", "adaptive"] {
        assert_eq!(stats[MODEL][arm]["requests"], 0, "{stats}");
    }
    let response = named(&native, "adaptive", Some(OPERATOR_TOKEN))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.terminal(&rid(&response)).await["state"], "completed");
    let stats = native.stats_ab().await;
    assert_eq!(
        (
            &stats[MODEL]["fixed"]["requests"],
            &stats[MODEL]["adaptive"]["requests"]
        ),
        (&json!(0), &json!(1)),
        "{stats}"
    );
    drop(native);
    let plain = Native::ready(Case::matching()).await;
    let response = named(&plain, "fixed", Some(OPERATOR_TOKEN))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert_eq!(
        (error_code(&error), &error["error"]["param"]),
        ("invalid_value", &json!("X-Cere-Ab-Arm")),
        "{error}"
    );
}

/// One session turn on `native` (the previous turns' messages, their replies and a new user message); returns the
/// reply message.
async fn ab_turn(native: &Native, session: &str, messages: &mut Vec<Value>) -> Value {
    let revision = messages.len() as u64 / 2;
    messages.push(json!({"role": "user", "content": prompt(TEXT)}));
    let mut request = body(TEXT, 4, false);
    request["messages"] = json!(messages);
    let response = turn(native, session, revision, &request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "revision {revision}");
    let reply: Value = response.json().await.unwrap();
    assert_eq!(reply["cere"]["revision"], revision + 1, "{reply}");
    let message = reply["choices"][0]["message"].clone();
    messages.push(message.clone());
    message
}

/// `requests` of each arm in `ab` stats, `None` for an arm that is not configured.
async fn arm_requests(native: &Native) -> BTreeMap<String, u64> {
    let stats = native.stats_ab().await;
    stats[MODEL]
        .as_object()
        .map(|arms| {
            arms.iter()
                .map(|(id, s)| (id.clone(), s["requests"].as_u64().unwrap()))
                .collect()
        })
        .unwrap_or_default()
}

/// AB-03/AB-05 (fail-first): a session keeps its arm across a live reload that moves every weight to the other arm,
/// while new sessions and requests without a session take the new weights; a reload that drops an arm drops its
/// stats and moves its sessions; an arm with an executor-layer lever refuses the whole reload (409
/// `config_invalid`, nothing applied); a reload that removes the section ends the split.
#[tokio::test]
async fn reloaded_arms_reach_new_sessions_only() {
    let native = Native::ready(Case {
        speculation: true,
        ab: ab_arms("exp-1", 1, 0),
        ..Case::matching()
    })
    .await;
    let counts = |pairs: &[(&str, u64)]| -> BTreeMap<String, u64> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
    };
    let first = create_session(&native).await;
    let mut first_messages = Vec::new();
    ab_turn(&native, &first, &mut first_messages).await;
    assert_eq!(
        arm_requests(&native).await,
        counts(&[("fixed", 1), ("adaptive", 0)])
    );

    let (status, reply) = native.reload_ab(&ab_arms("exp-1", 0, 1)).await;
    assert_eq!((status, &reply["applied"]), (200, &json!(true)), "{reply}");
    ab_turn(&native, &first, &mut first_messages).await;
    let second = create_session(&native).await;
    ab_turn(&native, &second, &mut Vec::new()).await;
    let response = native.chat(&body(STOP, 8, false)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(native.terminal(&rid(&response)).await["state"], "completed");
    assert_eq!(
        arm_requests(&native).await,
        counts(&[("fixed", 2), ("adaptive", 2)]),
        "the first session kept its arm; the new ones took the new weights"
    );

    let executor = json!({"salt": "exp-1", "arms": [
        {"id": "adaptive", "weight": 1, "levers": {"draft_policy": "adaptive", "k2_gemm": "small_m"}},
    ]});
    let (status, reply) = native.reload_ab(&executor).await;
    assert_eq!(
        (status, error_code(&reply)),
        (409, "config_invalid"),
        "{reply}"
    );
    assert_eq!(
        arm_requests(&native).await,
        counts(&[("fixed", 2), ("adaptive", 2)])
    );

    let only_adaptive = json!({"salt": "exp-1", "arms": [
        {"id": "adaptive", "weight": 1, "levers": {"draft_policy": "adaptive"}},
    ]});
    let (status, reply) = native.reload_ab(&only_adaptive).await;
    assert_eq!(status, 200, "{reply}");
    assert_eq!(arm_requests(&native).await, counts(&[("adaptive", 2)]));
    ab_turn(&native, &first, &mut first_messages).await;
    assert_eq!(arm_requests(&native).await, counts(&[("adaptive", 3)]));

    let (status, reply) = native.reload_ab(&Value::Null).await;
    assert_eq!(status, 200, "{reply}");
    assert_eq!(native.stats_ab().await, Value::Null);
    ab_turn(&native, &first, &mut first_messages).await;
}

/// O79 (m210 design note PF1): a deployment with `decode_rounds_per_prefill_chunk` loads and serves on a library
/// that exports `cere_executor_set_decode_rounds`; a library predating it fails the load only when the key is set.
#[tokio::test]
async fn decode_rounds_per_prefill_chunk_is_applied_at_load() {
    let native = Native::ready(Case {
        decode_rounds_per_prefill_chunk: Some(4),
        ..Case::matching()
    })
    .await;
    let response = native.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(response.status().as_u16(), 200);

    let old = native_support::toy_library_prestaging();
    let old_case = || Case {
        library_pin: native_server::file_sha256(&old),
        library: old.clone(),
        ..Case::matching()
    };
    Native::ready(old_case()).await;
    let refused = Native::start(Case {
        decode_rounds_per_prefill_chunk: Some(4),
        ..old_case()
    })
    .await;
    let (status, body) = refused.settled().await;
    assert_ne!(status, 200, "{body}");
    assert!(
        body.to_string()
            .contains("the library predates cere_executor_set_decode_rounds"),
        "{body}"
    );
}

/// MixedPD (mixed-pd design note): a deployment with `prefill_slice_ms` loads and serves on a library that exports
/// `cere_executor_set_prefill_slice`; a library predating it fails the load only when the key is set.
#[tokio::test]
async fn prefill_slice_ms_is_applied_at_load() {
    let native = Native::ready(Case {
        prefill_slice_ms: Some(200),
        ..Case::matching()
    })
    .await;
    let response = native.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(response.status().as_u16(), 200);

    let old = native_support::toy_library_prestaging();
    let old_case = || Case {
        library_pin: native_server::file_sha256(&old),
        library: old.clone(),
        ..Case::matching()
    };
    Native::ready(old_case()).await;
    let refused = Native::start(Case {
        prefill_slice_ms: Some(200),
        ..old_case()
    })
    .await;
    let (status, body) = refused.settled().await;
    assert_ne!(status, 200, "{body}");
    assert!(
        body.to_string()
            .contains("the library predates cere_executor_set_prefill_slice"),
        "{body}"
    );
}

/// S1 G4 (beat-sglang design note): a deployment with `decode_share_pct` loads and serves on a library that exports
/// `cere_executor_set_decode_share`; a library predating it fails the load only when the key is set.
#[tokio::test]
async fn decode_share_pct_is_applied_at_load() {
    let native = Native::ready(Case {
        decode_share_pct: Some(100),
        ..Case::matching()
    })
    .await;
    let response = native.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(response.status().as_u16(), 200);

    let old = native_support::toy_library_prestaging();
    let old_case = || Case {
        library_pin: native_server::file_sha256(&old),
        library: old.clone(),
        ..Case::matching()
    };
    Native::ready(old_case()).await;
    let refused = Native::start(Case {
        decode_share_pct: Some(100),
        ..old_case()
    })
    .await;
    let (status, body) = refused.settled().await;
    assert_ne!(status, 200, "{body}");
    assert!(
        body.to_string()
            .contains("the library predates cere_executor_set_decode_share"),
        "{body}"
    );
}

/// BC4 (b) (batch-concurrency design note): a deployment with `prefill_age_calls` loads and serves on a library that
/// exports `cere_executor_set_prefill_age`.
#[tokio::test]
async fn prefill_age_calls_is_applied_at_load() {
    let native = Native::ready(Case {
        prefill_age_calls: Some(8),
        ..Case::matching()
    })
    .await;
    let response = native.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
}

// ---------------------------------------------------------------------------------------------------------
// GrammarSpec GS-04 (grammar-spec design note): the `grammar_drafts` lever over the drafting toy and the stub
// grammar role. The toy drafts its own chain, so an `allow:` grammar over the toy's output keeps every draft and a
// `seq:` script refuses them; the stub's `grammar_drafts` op walks a draft prefix as `grammar_step` would.

/// The constrained requests of the `grammar_drafts` comparison, unary and stream bodies (SSE data events, `[DONE]`
/// as a string) in order: an `allow:` grammar over the toy's own output (drafts kept), a script the drafts miss
/// ending on the stop token, the grammar's termination, and an unconstrained request.
async fn grammar_battery(native: &Native) -> Vec<Value> {
    let own: Vec<String> = toy_output(TEXT, 8).iter().map(u32::to_string).collect();
    let requests = [
        constrained(&format!("allow:{}", own.join(",")), 8, false),
        constrained("seq:100,200,300,400,7", 8, false),
        constrained("seq:100,200", 8, false),
        body(TEXT, 6, false),
    ];
    let mut out = Vec::new();
    for unary in requests {
        let (status, got) = answer(native, &unary).await;
        assert_eq!(status, 200, "{unary}: {got}");
        out.push(got);
        let mut stream = unary.clone();
        stream["stream"] = json!(true);
        let (events, _) = streamed(native, &stream).await;
        out.push(Value::Array(
            events
                .into_iter()
                .map(|e| match e {
                    Event::Data(v) => v,
                    Event::Done => json!("[DONE]"),
                })
                .collect(),
        ));
    }
    out
}

/// Whether the stub grammar role has answered a `grammar_drafts` op.
fn saw_grammar_drafts(native: &Native) -> bool {
    native.sidecar_log().iter().any(|op| op == "grammar_drafts")
}

/// `got` equals `want` body for body, the per-process values aside.
fn assert_bodies_equal(mut got: Vec<Value>, mut want: Vec<Value>) {
    got.iter_mut().for_each(without_volatile);
    want.iter_mut().for_each(without_volatile);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "battery body {i}");
    }
    assert_eq!(got.len(), want.len());
}

/// GS-04, the lever's bit_identical contract: on the speculative profile with `grammar_drafts` "off" (the registry
/// default) a constrained request runs as before the lever, NO_SPECULATION: after the PREFILL every recorded step
/// is one token under its own mask and no verify round runs. The grammar process never sees `grammar_drafts`, and
/// the bodies equal those of the spec-off deployment.
#[tokio::test]
async fn grammar_drafts_off_serves_constrained_requests_plain() {
    let t = Tmp::new("gs-off");
    let (case, path) = recording(&t);
    let off = Native::ready(Case {
        speculation: true,
        levers: &[("grammar_drafts", "off")],
        tokenizer: Some(grammar_tokenizer(&[])),
        ..case
    })
    .await;
    assert_eq!(off.stats_levers().await["grammar_drafts"], "off");
    let request = constrained("seq:100,200,300,400,7", 8, false);
    let sessions = submitted(&off, &path, &request).await;
    let s = one_prefill(&sessions, 5);
    assert!(s.iter().all(|step| step["mask"] == "1"), "{s:?}");
    assert_eq!(ids_of(s), [100, 200, 300, 400, 7]);
    assert_eq!(off.spec_counters().await, (0, 0));
    let got = grammar_battery(&off).await;
    assert!(!saw_grammar_drafts(&off));
    let plain = Native::ready(with_tokenizer(grammar_tokenizer(&[]))).await;
    assert_bodies_equal(got, grammar_battery(&plain).await);
}

/// GS-04 (B1 on the toy): with `grammar_drafts` "on" a constrained DECODE speculates. The grammar process filters
/// each round's drafts (the toy's own chain) with `grammar_drafts`, and the executor verifies the accepted prefix
/// under its masks. The bodies equal the "off" run's. The `allow:` grammar keeps the drafts, so
/// `native_spec_accepted_drafts` advances; the `seq:` scripts refuse them, so the round runs under the position's
/// mask alone; and the grammar's termination still ends the request on that token.
#[tokio::test]
async fn grammar_drafts_on_verifies_the_grammars_drafts_with_the_off_bodies() {
    let case = |levers: &'static [(&'static str, &'static str)]| Case {
        speculation: true,
        levers,
        ..with_tokenizer(grammar_tokenizer(&[]))
    };
    let off = Native::ready(case(&[])).await;
    let on = Native::ready(case(&[("grammar_drafts", "on")])).await;
    assert_eq!(on.stats_levers().await["grammar_drafts"], "on");
    let own: Vec<String> = toy_output(TEXT, 8).iter().map(u32::to_string).collect();
    let (status, got) = answer(
        &on,
        &constrained(&format!("allow:{}", own.join(",")), 8, false),
    )
    .await;
    assert_eq!(status, 200, "{got}");
    let (steps, accepted) = on.spec_counters().await;
    assert!(
        steps > 0 && accepted > 0,
        "{steps} rounds, {accepted} accepted"
    );
    assert!(saw_grammar_drafts(&on));
    let want = grammar_battery(&off).await;
    assert!(!saw_grammar_drafts(&off));
    assert_bodies_equal(grammar_battery(&on).await, want);
}

/// GS-04 / D15: with `grammar_drafts` "on" the reasoning deferral holds across drafts (the grammar process walks
/// them as `grammar_step` would, under all-ones masks until `</think>`). The d15 request answers as it does with the
/// lever off.
#[tokio::test]
async fn grammar_drafts_on_keeps_the_thinking_deferral() {
    let native = Native::ready(Case {
        tokenizer: Some(grammar_tokenizer(&[(2, "ab"), (780, "</think>")])),
        capabilities_patch: json!({"thinking_toggle": true}),
        sidecar_args: vec!["--think-end-id".to_owned(), "780".to_owned()],
        speculation: true,
        levers: &[("grammar_drafts", "on")],
        ..Case::matching()
    })
    .await;
    let mut request = constrained("seq:100,7", 8, false);
    request["chat_template_kwargs"] = json!({"enable_thinking": true});
    let (status, got) = answer(&native, &request).await;
    assert_eq!(status, 200, "{got}");
    let c = &got["choices"][0];
    assert_eq!(c["message"]["reasoning_content"], "ab", "{got}");
    assert_eq!(c["message"]["content"], "[100]", "{got}");
    assert_eq!(
        (&c["finish_reason"], &c["matched_stop"]),
        (&json!("stop"), &json!(7)),
        "{got}"
    );
    assert_eq!(got["usage"]["completion_tokens"], 4, "{got}");
    assert!(saw_grammar_drafts(&native));
}

/// GS-04: `grammar_drafts` "on" fails the load on a library without the constrained-draft calls (the pre-staging
/// toy) and with a grammar process that does not serve `grammar_drafts`. Each of those deployments loads with the
/// lever "off".
#[tokio::test]
async fn grammar_drafts_on_is_refused_at_load_without_the_library_calls_or_the_grammar_feature() {
    let old = native_support::toy_library_prestaging();
    let old_case = |levers: &'static [(&'static str, &'static str)]| Case {
        library_pin: native_server::file_sha256(&old),
        library: old.clone(),
        speculation: true,
        levers,
        ..Case::matching()
    };
    Native::ready(old_case(&[])).await;
    let refused = Native::start(old_case(&[("grammar_drafts", "on")])).await;
    let (status, body) = refused.settled().await;
    assert_ne!(status, 200, "{body}");
    assert!(
        body.to_string()
            .contains("needs a library exporting cere_executor_round_drafts"),
        "{body}"
    );

    let featureless = |levers: &'static [(&'static str, &'static str)]| Case {
        speculation: true,
        levers,
        sidecar_args: vec!["--no-grammar-drafts-feature".to_owned()],
        ..Case::matching()
    };
    Native::ready(featureless(&[])).await;
    let refused = Native::start(featureless(&[("grammar_drafts", "on")])).await;
    let (status, body) = refused.settled().await;
    assert_ne!(status, 200, "{body}");
    assert!(
        body.to_string()
            .contains("needs a grammar process serving grammar_drafts"),
        "{body}"
    );
}
