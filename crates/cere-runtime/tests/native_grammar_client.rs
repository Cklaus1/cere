//! M2.8 C4 (Rust half; spec m28 D7, A8): the grammar sidecar client against the stub sidecar's `--role grammar`
//! (`native_support/stub_sidecar.py`). A second child with its own mutex: `info` must report the `grammar` feature
//! and the dialect's xgrammar version; `grammar_open` has a compile deadline, and a missed deadline or a dead child
//! fails only that request, kills the child and respawns it while the tokenizer sidecar keeps serving.
mod native_support;

use std::time::{Duration, Instant};

use cere_runtime::native::sidecar::{
    GrammarError, GrammarHandle, GrammarHealth, GrammarOpen, GrammarSidecar, Sidecar, SidecarError,
    SidecarHealth, TemplateKwargs, GRAMMAR_TIMEOUT_MESSAGE,
};
use cere_runtime::validate::{ChatMessage, Role};
use native_support::root;

const XGRAMMAR: &str = "0.2.1";
const ZERO_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// 248,320 tokens / 32.
const MASK_WORDS: usize = 7_760;

/// The stub's argv (both roles); the grammar client appends `--role grammar`.
fn argv(extra: &[&str]) -> Vec<String> {
    let stub = root().join("crates/cere-runtime/tests/native_support/stub_sidecar.py");
    let mut argv = vec![
        "python3".to_owned(),
        stub.display().to_string(),
        "--tokenizer-sha256".to_owned(),
        ZERO_SHA.to_owned(),
        "--template-sha256".to_owned(),
        ZERO_SHA.to_owned(),
    ];
    argv.extend(extra.iter().map(|a| (*a).to_owned()));
    argv
}

fn grammar(extra: &[&str], deadline: Duration) -> GrammarSidecar {
    GrammarSidecar::spawn(&argv(extra), XGRAMMAR, deadline).expect("spawning the grammar stub")
}

/// `argv` started 500 ms late, as a loaded host starts the interpreter late (a 48-busy-loop stress run of the
/// deadline test: 19 of 320 children missed their startup `info` deadline).
fn slow_start(argv: Vec<String>) -> Vec<String> {
    let mut slow: Vec<String> = ["sh", "-c", "sleep 0.5; exec \"$@\"", "sh"]
        .map(str::to_owned)
        .into();
    slow.extend(argv);
    slow
}

/// `grammar_open` of a regex constraint as the request carries it: the JSON string token.
fn open_regex(
    g: &GrammarSidecar,
    pattern: &str,
    mask: &mut Vec<u32>,
) -> Result<GrammarHandle, GrammarError> {
    let raw = serde_json::to_string(pattern).unwrap();
    let open = GrammarOpen {
        regex: Some(&raw),
        ..GrammarOpen::default()
    };
    assert_eq!(open.param(), "regex");
    g.grammar_open(&open, mask)
}

/// The one token the stub's mask allows.
fn allowed(mask: &[u32]) -> u32 {
    assert_eq!(mask.len(), MASK_WORDS);
    let set: Vec<u32> = mask
        .iter()
        .enumerate()
        .flat_map(|(w, &bits)| {
            (0..32)
                .filter(move |b| (bits >> b) & 1 == 1)
                .map(move |b| w as u32 * 32 + b)
        })
        .collect();
    assert_eq!(set.len(), 1, "the stub allows exactly one token");
    set[0]
}

fn wait_ready(g: &GrammarSidecar) {
    let start = Instant::now();
    while !g.ready() {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the grammar sidecar never respawned"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn pid_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[test]
fn info_is_recorded_and_ops_decode_masks_into_the_reused_buffer() {
    let g = grammar(&[], Duration::from_secs(20));
    assert!(g.ready());
    let info = g.info();
    assert_eq!(info.xgrammar, XGRAMMAR);
    assert_eq!(info.stop_token_ids, [248_044, 248_046]);
    assert_eq!(info.torch, "stub");
    assert_eq!(info.vocab_size, 248_320);
    let mut mask = Vec::new();
    let handle = open_regex(&g, "[a-z]+", &mut mask).unwrap();
    assert_eq!(handle.key_type, "regex");
    assert_eq!(handle.key_sha256.len(), 64);
    let buffer = mask.as_ptr();
    let mut token = allowed(&mask);
    for i in 0..3 {
        let step = g.grammar_step(&handle, token, &mut mask).unwrap();
        assert!(step.accepted);
        assert_eq!(step.terminated, i == 2);
        assert_eq!(step.has_mask, i < 2);
        if step.has_mask {
            assert_eq!(mask.as_ptr(), buffer, "the mask buffer is reused");
            token = allowed(&mask);
        }
    }
    g.grammar_close(&handle).unwrap();
    // A closed handle is unknown to the process: a typed refusal, the process keeps serving.
    assert!(matches!(
        g.grammar_step(&handle, token, &mut mask),
        Err(GrammarError::Refused { ref kind, .. }) if kind == "invalid_request"
    ));
    let rejected = open_regex(&g, "x", &mut mask).unwrap();
    let step = g
        .grammar_step(&rejected, allowed(&mask) + 1, &mut mask)
        .unwrap();
    assert!(!step.accepted && !step.terminated && !step.has_mask);
    assert!(g.ready());
}

#[test]
fn mismatched_xgrammar_version_is_refused() {
    match GrammarSidecar::spawn(
        &argv(&["--xgrammar", "0.2.0"]),
        XGRAMMAR,
        Duration::from_secs(20),
    ) {
        Err(SidecarError::Protocol(m)) => {
            assert!(m.contains("0.2.0") && m.contains("0.2.1"), "{m}")
        }
        other => panic!("expected a protocol refusal, got {other:?}"),
    }
}

#[test]
fn missing_grammar_feature_is_refused() {
    match GrammarSidecar::spawn(
        &argv(&["--no-grammar-feature"]),
        XGRAMMAR,
        Duration::from_secs(20),
    ) {
        Err(SidecarError::Protocol(m)) => assert!(m.contains("feature grammar"), "{m}"),
        other => panic!("expected a protocol refusal, got {other:?}"),
    }
}

/// D7: a compile past the deadline fails that request with the fixed timeout text on its field, the stalled child
/// is killed and a new one spawned (not ready meanwhile, a request then is `NotReady`), handles of the old child
/// are lost, a following open succeeds, and the tokenizer sidecar serves `encode_chat` throughout.
#[test]
fn compile_past_the_deadline_times_out_and_respawns_while_the_tokenizer_serves() {
    // The deadline also bounds every child's startup `info`; a child started late (500 ms here, as on a loaded
    // host) has all of it, since the respawn is held not ready by `hold`, not by a startup sleep.
    let deadline = Duration::from_secs(2);
    let hold = std::env::temp_dir().join(format!("cere-grammar-hold-{}", std::process::id()));
    let _ = std::fs::remove_file(&hold);
    let hold_arg = hold.display().to_string();
    let g = GrammarSidecar::spawn(
        &slow_start(argv(&["--hold-start-while", &hold_arg])),
        XGRAMMAR,
        deadline,
    )
    .expect("spawning the grammar stub");
    // A loaded host (gate-100): the tokenizer stub descheduled for 600 ms during the stall, as the encodes sent
    // while `slow` exists are.
    let slow =
        std::env::temp_dir().join(format!("cere-grammar-slow-encode-{}", std::process::id()));
    let _ = std::fs::remove_file(&slow);
    let slow_arg = slow.display().to_string();
    let tokenizer = Sidecar::spawn(
        &argv(&["--slow-encode-if", &slow_arg, "0.6"]),
        Duration::from_secs(60),
    )
    .unwrap();
    // As cered does at startup: `info` records the features (`message_boundary`, M2.10 D4 (c)) the replies follow.
    tokenizer.info().unwrap();
    let mut mask = Vec::new();
    let old = open_regex(&g, "a", &mut mask).unwrap();
    let stalled_pid = g.pid().unwrap();
    std::fs::write(&hold, b"").unwrap();
    let encoded = std::thread::scope(|s| {
        let compile = s.spawn(|| {
            let start = Instant::now();
            let mut mask = Vec::new();
            let result = open_regex(&g, "stall", &mut mask);
            (result, start.elapsed())
        });
        // An encode the stall blocked would return only once the compile timed out: count the ones that returned
        // while it was still pending, however long each took (a loaded host deschedules the stub, gate-100).
        let mut encoded = 0;
        loop {
            let ids = tokenizer
                .encode_chat(
                    &[ChatMessage::text(Role::User, "hi")],
                    None,
                    false,
                    &[],
                    &TemplateKwargs::default(),
                )
                .unwrap()
                .ids;
            assert_eq!(ids, [1, b'h'.into(), b'i'.into()]);
            if compile.is_finished() {
                break;
            }
            encoded += 1;
            if encoded == 1 {
                std::fs::write(&slow, b"").unwrap();
            } else {
                let _ = std::fs::remove_file(&slow);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let (result, elapsed) = compile.join().unwrap();
        match result {
            Err(e @ GrammarError::TimedOut { param: "regex" }) => {
                assert_eq!(e.to_string(), GRAMMAR_TIMEOUT_MESSAGE);
                let api = e.api_error();
                assert_eq!((api.status, api.code), (400, "invalid_value"));
                assert_eq!(api.param.as_deref(), Some("regex"));
                assert_eq!(api.message, "Grammar compilation timed out");
            }
            other => panic!("expected a timeout, got {other:?}"),
        }
        assert!(
            elapsed >= deadline && elapsed < deadline + Duration::from_secs(1),
            "{elapsed:?}"
        );
        encoded
    });
    let _ = std::fs::remove_file(&slow);
    assert!(
        encoded >= 10,
        "the tokenizer sidecar was blocked: {encoded} encodes returned during the stall"
    );
    assert!(!pid_alive(stalled_pid), "the stalled child was killed");
    assert_eq!(GRAMMAR_TIMEOUT_MESSAGE, "Grammar compilation timed out");
    // The respawned child serves once `hold` is gone.
    assert!(!g.ready());
    assert!(matches!(
        open_regex(&g, "a", &mut mask),
        Err(GrammarError::NotReady)
    ));
    std::fs::remove_file(&hold).unwrap();
    wait_ready(&g);
    assert_ne!(g.pid(), Some(stalled_pid));
    let new = open_regex(&g, "a", &mut mask).unwrap();
    assert!(matches!(
        g.grammar_step(&old, 0, &mut mask),
        Err(GrammarError::Lost(_))
    ));
    // Closing a handle of the old child is a no-op.
    g.grammar_close(&old).unwrap();
    let step = g.grammar_step(&new, allowed(&mask), &mut mask).unwrap();
    assert!(step.accepted);
    assert_eq!(tokenizer.health(), SidecarHealth::Ready);
}

/// D7: a child that dies during `grammar_open` fails that request like a missed deadline (fixed timeout text) and
/// is respawned; one that dies during `grammar_step` loses the handle (500 at the caller) and is respawned.
#[test]
fn death_mid_op_fails_that_request_and_respawns() {
    let g = grammar(&[], Duration::from_secs(20));
    let mut mask = Vec::new();
    let first = open_regex(&g, "a", &mut mask).unwrap();
    let pid = g.pid().unwrap();
    assert!(matches!(
        open_regex(&g, "die", &mut mask),
        Err(GrammarError::TimedOut { param: "regex" })
    ));
    wait_ready(&g);
    assert_ne!(g.pid(), Some(pid));
    assert!(matches!(
        g.grammar_step(&first, 0, &mut mask),
        Err(GrammarError::Lost(_))
    ));
    let handle = open_regex(&g, "a", &mut mask).unwrap();
    let pid = g.pid().unwrap();
    assert!(matches!(
        g.grammar_step(&handle, u32::MAX, &mut mask),
        Err(GrammarError::Lost(_))
    ));
    wait_ready(&g);
    assert_ne!(g.pid(), Some(pid));
    open_regex(&g, "a", &mut mask).unwrap();
}

/// O46 (ops-hardening design note OH-02): a `grammar_step` shares the compile deadline. A child that stops
/// answering a step fails that request at the deadline (not never) and loses the handle; the child is killed and
/// respawned and the next constrained request succeeds.
#[test]
fn step_past_the_deadline_fails_that_request_and_respawns() {
    let deadline = Duration::from_secs(3);
    let g = grammar(&[], deadline);
    let mut mask = Vec::new();
    // The stub's script: step 0 sleeps an hour before replying.
    let handle = open_regex(&g, "seq:5,6;sleep@0=3600", &mut mask).unwrap();
    let pid = g.pid().unwrap();
    let start = Instant::now();
    let step = g.grammar_step(&handle, 5, &mut mask);
    let waited = start.elapsed();
    assert!(
        matches!(&step, Err(GrammarError::Lost(m)) if m.contains("no reply within 3000 ms")),
        "{step:?}"
    );
    assert!(waited >= deadline, "{waited:?}");
    assert!(waited < deadline + Duration::from_secs(10), "{waited:?}");
    wait_ready(&g);
    assert_ne!(g.pid(), Some(pid));
    assert!(!pid_alive(pid), "the stalled child was killed");
    let handle = open_regex(&g, "a", &mut mask).unwrap();
    let token = allowed(&mask);
    assert!(g.grammar_step(&handle, token, &mut mask).unwrap().accepted);
    g.grammar_close(&handle).unwrap();
}

/// A child that exits between requests: readiness notices, respawns, and the next open succeeds.
#[test]
fn exited_child_turns_not_ready_and_respawns() {
    let g = grammar(&["--exit-after", "2"], Duration::from_secs(20));
    let mut mask = Vec::new();
    // `info` was reply 1; this open is reply 2, after which the child exits.
    open_regex(&g, "a", &mut mask).unwrap();
    let start = Instant::now();
    while g.ready() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the exit was never noticed"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    wait_ready(&g);
    open_regex(&g, "a", &mut mask).unwrap();
}

/// Audit N4: every kill of a freshly respawned child is respawned in turn (the respawn flag is cleared in the
/// critical section that installs the child, so `fail` never skips a respawn and leaves the process `Down`). The
/// race itself is a window of nanoseconds between two stores and has no test hook: this checks that the chain of
/// kill, notice, respawn holds across several generations, each killed the moment it is ready.
#[test]
fn a_child_killed_right_after_its_respawn_is_respawned_again() {
    let g = grammar(&[], Duration::from_secs(20));
    let mut mask = Vec::new();
    for _ in 0..6 {
        let pid = g.pid().expect("a running child");
        // SAFETY: `kill` takes a pid and a signal and touches no memory; the pid is our own child's.
        assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
        let start = Instant::now();
        while !(g.ready() && g.pid().is_some_and(|now| now != pid)) {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "the killed child was never respawned"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_ne!(g.health(), GrammarHealth::Down);
    }
    wait_ready(&g);
    open_regex(&g, "a", &mut mask).unwrap();
}

/// D7 (Z2a P2): the readiness the probe reports. A killed child is `Respawning` until its successor answered
/// `info`; a respawn whose child fails at startup leaves the process `Down` for good (no second respawn).
#[test]
fn killed_child_is_respawning_then_a_failed_respawn_is_down_for_good() {
    let gate = std::env::temp_dir().join(format!("cere-grammar-gate-{}", std::process::id()));
    let _ = std::fs::remove_file(&gate);
    let gate_arg = gate.display().to_string();
    let g = grammar(&["--fail-start-if", &gate_arg], Duration::from_secs(20));
    assert_eq!(g.health(), GrammarHealth::Ready);
    let mut mask = Vec::new();
    assert!(open_regex(&g, "die", &mut mask).is_err());
    assert_ne!(g.health(), GrammarHealth::Ready);
    wait_ready(&g);
    assert_eq!(g.health(), GrammarHealth::Ready);

    std::fs::write(&gate, b"").unwrap();
    assert!(open_regex(&g, "die", &mut mask).is_err());
    let start = Instant::now();
    while g.health() != GrammarHealth::Down {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the failed respawn was never reported"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    std::fs::remove_file(&gate).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(g.health(), GrammarHealth::Down, "down for good: no respawn");
    assert!(!g.ready());
    assert_eq!(
        open_regex(&g, "a", &mut mask).unwrap_err(),
        GrammarError::NotReady
    );
}

/// D7 (Z2a): the startup `info` (the warm-up) shares the compile deadline; a child that misses it is killed and
/// the spawn fails.
#[test]
fn startup_info_past_the_deadline_fails_the_spawn() {
    let start = Instant::now();
    let spawned = GrammarSidecar::spawn(
        &argv(&["--startup-sleep", "5"]),
        XGRAMMAR,
        Duration::from_millis(1000),
    );
    let elapsed = start.elapsed();
    assert!(
        matches!(spawned, Err(SidecarError::Gone(_))),
        "{:?}",
        spawned.map(|g| g.info().clone())
    );
    assert!(elapsed < Duration::from_secs(3), "{elapsed:?}");
}

/// D7: at most 4 open handles per process; a 5th is `overloaded` (503 at the caller) and the process keeps serving.
#[test]
fn fifth_open_handle_is_overloaded() {
    let g = grammar(&[], Duration::from_secs(20));
    let mut mask = Vec::new();
    let handles: Vec<GrammarHandle> = (0..4)
        .map(|i| open_regex(&g, &format!("h{i}"), &mut mask).unwrap())
        .collect();
    let overloaded = open_regex(&g, "h4", &mut mask).unwrap_err();
    assert_eq!(overloaded, GrammarError::Overloaded);
    let api = overloaded.api_error();
    assert_eq!((api.status, api.code, api.param), (503, "overloaded", None));
    g.grammar_close(&handles[0]).unwrap();
    open_regex(&g, "h4", &mut mask).unwrap();
    assert!(g.ready());
}

/// A compile failure is the sidecar's typed `invalid_value` on the request field; the process keeps serving.
#[test]
fn compile_failure_is_a_typed_refusal_on_the_field() {
    let g = grammar(&[], Duration::from_secs(20));
    let mut mask = Vec::new();
    match open_regex(&g, "invalid", &mut mask) {
        Err(e @ GrammarError::Refused { .. }) => {
            let api = e.api_error();
            assert_eq!((api.status, api.code), (400, "invalid_value"));
            assert_eq!(api.param.as_deref(), Some("regex"));
            assert_eq!(api.message, "Grammar compilation failed");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    let open = GrammarOpen {
        response_format: Some(r#"{"type":"json_object"}"#),
        ..GrammarOpen::default()
    };
    assert_eq!(open.param(), "response_format");
    assert_eq!(g.grammar_open(&open, &mut mask).unwrap().key_type, "json");
    let tools = GrammarOpen {
        tools_json: Some("[]"),
        tool_choice: Some(r#""required""#),
        parallel_tool_calls: Some(false),
        ..GrammarOpen::default()
    };
    assert_eq!(tools.param(), "tool_choice");
    g.grammar_open(&tools, &mut mask).unwrap();
    assert!(g.ready());
}
