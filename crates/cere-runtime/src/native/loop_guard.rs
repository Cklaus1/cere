//! LoopGuard (loop-guard design note): the native repetition guard of `[deployments.native.loop_guard]`.
//!
//! LG-D1 ([`Repetition`]): after each committed token the choice's output text ends with at least `min_repeats`
//! exact copies of a unit of `min_period_bytes..=max_period_bytes` bytes covering at least `min_span_bytes`; the
//! output then finishes with `finish_reason` `repetition`. LG-D2 ([`tool_repeat`]): the request's conversation ends
//! with `tool_repeat_run` identical assistant tool call -> tool result pairs; the request is refused before encode.
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

use crate::api_types::{ApiError, INVALID_REQUEST};
use crate::config::LoopGuardConfig;
use crate::validate::{ChatMessage, Content, Role};

/// The `code` of an LG-D2 refusal.
pub const LOOP_DETECTED: &str = "loop_detected";

/// The guard of one loaded native backend: its thresholds and its counters (`GET /v1/cere/stats`
/// `native_loop_guard`).
#[derive(Debug)]
pub struct LoopGuard {
    pub config: LoopGuardConfig,
    repetition_stops: AtomicU64,
    tool_repeat_refusals: AtomicU64,
}

impl LoopGuard {
    pub fn new(config: LoopGuardConfig) -> Self {
        Self {
            config,
            repetition_stops: AtomicU64::new(0),
            tool_repeat_refusals: AtomicU64::new(0),
        }
    }

    /// LG-D1's detector for one choice; `None` when `min_repeats` is 0.
    pub fn repetition(&self) -> Option<Repetition> {
        Repetition::new(&self.config)
    }

    /// Counts one LG-D1 stop.
    pub fn stopped(&self) {
        self.repetition_stops.fetch_add(1, Ordering::Relaxed);
    }

    /// LG-D2 on a request's messages: `Err` (400 `loop_detected`, counted) when the conversation ends with at least
    /// `tool_repeat_run` identical pairs.
    pub fn check_messages(&self, messages: &[ChatMessage]) -> Result<(), ApiError> {
        let need = self.config.tool_repeat_run as usize;
        if need == 0 {
            return Ok(());
        }
        match tool_repeat(messages, &self.config.ignore_arg_keys) {
            Some(run) if run.run >= need => {
                self.tool_repeat_refusals.fetch_add(1, Ordering::Relaxed);
                Err(ApiError::new(
                    400,
                    INVALID_REQUEST,
                    LOOP_DETECTED,
                    format!(
                        "the conversation ends with {} identical tool calls ({}) with identical results; the loop \
                         guard refuses another turn (deployments.native.loop_guard.tool_repeat_run = {need})",
                        run.run,
                        run.names.join(", ")
                    ),
                    Some("messages".to_owned()),
                ))
            }
            _ => Ok(()),
        }
    }

    /// `{"repetition_stops", "tool_repeat_refusals"}`.
    pub fn counters(&self) -> (u64, u64) {
        (
            self.repetition_stops.load(Ordering::Relaxed),
            self.tool_repeat_refusals.load(Ordering::Relaxed),
        )
    }

    /// The stats object of several guards (one per replica), summed.
    pub fn stats<'a>(guards: impl Iterator<Item = &'a LoopGuard>) -> Option<Value> {
        let mut any = false;
        let (mut stops, mut refusals) = (0, 0);
        for guard in guards {
            any = true;
            let (s, r) = guard.counters();
            stops += s;
            refusals += r;
        }
        any.then(|| json!({ "repetition_stops": stops, "tool_repeat_refusals": refusals }))
    }
}

/// An LG-D1 detection: the smallest period of the repeating suffix and the suffix's length, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub period: usize,
    pub span: usize,
}

/// LG-D1 for one output: one prefix-function pass over the reversed tail window per check. Prefix length `m` of
/// the reversed window is the text's suffix of length `m`, and `m - pi[m - 1]` its smallest period. Buffers are
/// reused, so a check allocates nothing after the first.
#[derive(Debug)]
pub struct Repetition {
    min_period: usize,
    max_period: usize,
    min_repeats: usize,
    min_span: usize,
    window: usize,
    pi: Vec<u32>,
    /// The text length of the last check: an unchanged text is not checked again.
    checked: usize,
}

impl Repetition {
    pub fn new(config: &LoopGuardConfig) -> Option<Self> {
        if config.min_repeats == 0 {
            return None;
        }
        Some(Self {
            min_period: config.min_period_bytes as usize,
            max_period: config.max_period_bytes as usize,
            min_repeats: config.min_repeats as usize,
            min_span: config.min_span_bytes as usize,
            window: config.window_bytes() as usize,
            pi: Vec::new(),
            checked: 0,
        })
    }

    /// The detection at the text's end, if any: the shortest suffix whose smallest period `p` is in
    /// `min_period..=max_period` and whose length is at least `max(min_span, min_repeats x p)`.
    pub fn check(&mut self, text: &str) -> Option<Hit> {
        let bytes = text.as_bytes();
        if bytes.len() == self.checked {
            return None;
        }
        self.checked = bytes.len();
        let floor = self.min_span.max(self.min_repeats * self.min_period);
        let w = bytes.len().min(self.window);
        if w < floor {
            return None;
        }
        let tail = &bytes[bytes.len() - w..];
        let at = |i: usize| tail[w - 1 - i];
        self.pi.clear();
        self.pi.resize(w, 0);
        for i in 1..w {
            let c = at(i);
            let mut k = self.pi[i - 1] as usize;
            while k > 0 && at(k) != c {
                k = self.pi[k - 1] as usize;
            }
            if at(k) == c {
                k += 1;
            }
            self.pi[i] = k as u32;
            let span = i + 1;
            let period = span - k;
            if span >= floor
                && (self.min_period..=self.max_period).contains(&period)
                && span >= self.min_span.max(self.min_repeats * period)
            {
                return Some(Hit { period, span });
            }
        }
        None
    }
}

/// An LG-D2 tail: the count of identical pairs ending the conversation and the last pair's call names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolRun {
    pub run: usize,
    pub names: Vec<String>,
}

/// LG-D2: when the conversation ends with tool results, the number of (assistant `tool_calls`, its `tool` results)
/// pairs at its end equal to the last one. Calls compare by name and arguments (JSON, `ignore` keys removed at the
/// top level, key order ignored; unparseable arguments as the raw string); results by content; call ids and the
/// assistant's prose and reasoning are not compared.
pub fn tool_repeat(messages: &[ChatMessage], ignore: &[String]) -> Option<ToolRun> {
    let mut end = messages.len();
    let last = pair(messages, &mut end, ignore)?;
    let mut run = 1;
    while pair(messages, &mut end, ignore).as_ref() == Some(&last) {
        run += 1;
    }
    let mut names: Vec<String> = Vec::new();
    for (name, _) in &last.calls {
        if !names.iter().any(|n| n == name) {
            names.push((*name).to_owned());
        }
    }
    Some(ToolRun { run, names })
}

/// A call's arguments as compared: the parsed JSON (ignored keys removed) or, unparseable, the client string.
#[derive(Debug, PartialEq)]
enum Args<'a> {
    Json(Value),
    Raw(&'a str),
}

/// One assistant tool-call turn and the tool results that follow it, as compared.
#[derive(Debug, PartialEq)]
struct Pair<'a> {
    calls: Vec<(&'a str, Args<'a>)>,
    results: Vec<Option<&'a Content>>,
}

/// The pair ending at `messages[..*end]`, moving `end` to its start; `None` when the messages there are not an
/// assistant turn with calls followed by at least one tool result.
fn pair<'a>(messages: &'a [ChatMessage], end: &mut usize, ignore: &[String]) -> Option<Pair<'a>> {
    let mut start = *end;
    while start > 0 && messages[start - 1].role == Role::Tool {
        start -= 1;
    }
    if start == *end || start == 0 {
        return None;
    }
    let turn = &messages[start - 1];
    let calls = match (&turn.role, &turn.tool_calls) {
        (Role::Assistant, Some(calls)) if !calls.is_empty() => calls,
        _ => return None,
    };
    let calls = calls
        .iter()
        .map(|call| {
            let args = match serde_json::from_str::<Value>(&call.function.arguments) {
                Ok(mut value) => {
                    if let Value::Object(map) = &mut value {
                        for key in ignore {
                            map.remove(key);
                        }
                    }
                    Args::Json(value)
                }
                Err(_) => Args::Raw(&call.function.arguments),
            };
            (call.function.name.as_str(), args)
        })
        .collect();
    let results = messages[start..*end]
        .iter()
        .map(|m| m.content.as_ref())
        .collect();
    *end = start - 1;
    Some(Pair { calls, results })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::{ToolCall, ToolCallFunction};

    fn config(
        min_period: u32,
        max_period: u32,
        min_repeats: u32,
        min_span: u32,
    ) -> LoopGuardConfig {
        LoopGuardConfig {
            min_period_bytes: min_period,
            max_period_bytes: max_period,
            min_repeats,
            min_span_bytes: min_span,
            ..LoopGuardConfig::default()
        }
    }

    /// Feeds `text` in `step`-byte pieces (token-like growth); the first hit and the text length it fired at.
    fn first_hit(r: &mut Repetition, text: &str, step: usize) -> Option<(usize, Hit)> {
        let mut end = 0;
        while end < text.len() {
            end = (end + step).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            if let Some(hit) = r.check(&text[..end]) {
                return Some((end, hit));
            }
        }
        None
    }

    const BLOCK: &str =
        "<tool_call>\n<function=Bash>\n<parameter=command>\npython3 -c 'import calc'\n\
                         </parameter>\n</function>\n</tool_call>\n";

    #[test]
    fn a_looping_output_is_detected_at_the_first_token_where_the_bar_holds() {
        let cfg = config(16, 1024, 4, 256);
        let mut r = Repetition::new(&cfg).unwrap();
        // The prefix's last byte differs from the block's, so no rotation of the copies starts inside it.
        let prefix = "Let me check the import:";
        let text = format!("{prefix}{}", BLOCK.repeat(10));
        let (at, hit) = first_hit(&mut r, &text, 1).expect("the loop is detected");
        assert_eq!(hit.period, BLOCK.len());
        // Four copies (and at least 256 bytes) end the text for the first time at the 4th copy's last byte.
        let bar = (4 * BLOCK.len()).max(256);
        assert_eq!(at, prefix.len() + bar);
        assert_eq!(hit.span, bar);
        // A degenerate prose loop too, whose unit is not aligned with the start of the output.
        let mut r = Repetition::new(&cfg).unwrap();
        let prose = format!(
            "xyz{}",
            "I will now run the tests again to be sure. ".repeat(12)
        );
        let (_, hit) = first_hit(&mut r, &prose, 4).expect("the prose loop is detected");
        assert_eq!(
            hit.period,
            "I will now run the tests again to be sure. ".len()
        );
        // The default thresholds catch the race's flat loop within 8 copies.
        let mut r = Repetition::new(&LoopGuardConfig::default()).unwrap();
        let (at, _) = first_hit(&mut r, &BLOCK.repeat(20), 5).expect("default thresholds");
        assert!(at <= 8 * BLOCK.len(), "{at}");
    }

    #[test]
    fn near_repeats_short_periods_and_too_few_copies_are_not_flagged() {
        let cfg = config(16, 1024, 4, 256);
        // Near-repeat code: numbered tests and case tables differ line to line (8 KB).
        let mut code = String::from("import calc\n\n");
        let mut i = 0;
        while code.len() < 8192 {
            code.push_str(&format!(
                "def test_add_{i}():\n    assert calc.evaluate(\"{i} + {i}\") == {}\n\n",
                2 * i
            ));
            code.push_str(&format!("    case {i}: return \"{i}\";\n"));
            i += 1;
        }
        let mut r = Repetition::new(&cfg).unwrap();
        assert_eq!(first_hit(&mut r, &code, 3), None);
        // Short-period runs below min_period_bytes (rules, zero rows, fills).
        for run in ["=".repeat(4000), "0, ".repeat(1500), "x = 0\n".repeat(600)] {
            let mut r = Repetition::new(&cfg).unwrap();
            assert_eq!(first_hit(&mut r, &run, 7), None, "{:?}", &run[..12]);
        }
        // Three exact copies with min_repeats 4.
        let mut r = Repetition::new(&cfg).unwrap();
        assert_eq!(first_hit(&mut r, &BLOCK.repeat(3), 2), None);
        // Copies followed by other text: only the end of the output counts.
        let mut r = Repetition::new(&cfg).unwrap();
        let text = format!(
            "{}{}",
            BLOCK.repeat(3),
            "Done: the import works, all 51 tests pass now.\n"
        );
        assert_eq!(first_hit(&mut r, &text, 2), None);
        // A unit longer than max_period_bytes.
        let mut long = String::new();
        for k in 0..300 {
            std::fmt::Write::write_fmt(&mut long, format_args!("{k:03}")).unwrap();
        }
        let mut r = Repetition::new(&config(16, 512, 4, 256)).unwrap();
        assert_eq!(first_hit(&mut r, &long.repeat(5), 64), None);
        // Detector off.
        assert!(Repetition::new(&config(16, 1024, 0, 256)).is_none());
    }

    #[test]
    fn min_span_bytes_holds_a_short_unit_back() {
        // A 20-byte unit needs 1,024 bytes of copies, not 4 copies.
        let unit = "ok, retrying now...\n";
        let mut r = Repetition::new(&config(16, 1024, 4, 1024)).unwrap();
        let (at, hit) = first_hit(&mut r, &unit.repeat(100), 1).unwrap();
        assert_eq!((at, hit.period, hit.span), (1024, unit.len(), 1024));
    }

    fn bash(command: &str, description: &str) -> String {
        json!({"command": command, "description": description}).to_string()
    }

    fn assistant(calls: &[(&str, String)], text: &str) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: Some(Content::Text(text.to_owned())),
            reasoning_content: None,
            tool_calls: Some(
                calls
                    .iter()
                    .enumerate()
                    .map(|(k, (name, arguments))| ToolCall {
                        id: format!("call_{k}_{}", text.len()),
                        kind: "function".to_owned(),
                        function: ToolCallFunction {
                            name: (*name).to_owned(),
                            arguments: arguments.clone(),
                        },
                    })
                    .collect(),
            ),
            tool_call_id: None,
        }
    }

    fn result(id: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: Role::Tool,
            content: Some(Content::Text(text.to_owned())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(id.to_owned()),
        }
    }

    const ORDINALS: [&str; 12] = [
        "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth",
        "Tenth", "Eleventh", "Twelfth",
    ];

    /// The race `qwen2` shape: one Bash probe, its `description` relabelled on every repeat, the same output.
    fn race(n: usize) -> Vec<ChatMessage> {
        let mut m = vec![
            ChatMessage::text(Role::System, "You are a coding agent."),
            ChatMessage::text(Role::User, "Make the hidden tests pass."),
        ];
        for k in 0..n {
            let label = format!("{} identical interleave check", ORDINALS[k % 12]);
            m.push(assistant(
                &[("Bash", bash("python3 -m pytest -q test_calc.py", &label))],
                &format!("Checking again ({k})."),
            ));
            m.push(result(&format!("call_0_{k}"), "51 passed in 0.12s"));
        }
        m
    }

    fn ignore() -> Vec<String> {
        vec!["description".to_owned()]
    }

    #[test]
    fn identical_tool_call_pairs_are_counted_whatever_the_description() {
        let run = tool_repeat(&race(10), &ignore()).unwrap();
        assert_eq!(
            run,
            ToolRun {
                run: 10,
                names: vec!["Bash".to_owned()]
            }
        );
        // Without the ignore list the relabelled calls differ (the proxy's pre-Gen-14 miss).
        assert_eq!(tool_repeat(&race(10), &[]).unwrap().run, 1);
        // Key order does not matter; parallel calls compare as an ordered list.
        let mut m = race(3);
        let swapped =
            json!({"description": "x", "command": "python3 -m pytest -q test_calc.py"}).to_string();
        m.push(assistant(&[("Bash", swapped)], ""));
        m.push(result("c", "51 passed in 0.12s"));
        assert_eq!(tool_repeat(&m, &ignore()).unwrap().run, 4);
        let guard = LoopGuard::new(LoopGuardConfig::default());
        let refused = guard.check_messages(&race(10)).unwrap_err();
        assert_eq!((refused.status, refused.code), (400, LOOP_DETECTED));
        assert!(
            refused.message.contains("10 identical tool calls (Bash)"),
            "{}",
            refused.message
        );
        assert!(guard.check_messages(&race(9)).is_ok());
        assert_eq!(guard.counters(), (0, 1));
    }

    #[test]
    fn different_results_commands_or_a_text_tail_are_not_a_run() {
        // The last result differs: the model got new information.
        let mut m = race(10);
        let last = m.len() - 1;
        m[last] = result("x", "50 passed, 1 failed in 0.13s");
        assert_eq!(tool_repeat(&m, &ignore()).unwrap().run, 1);
        // A different command in the middle breaks the run.
        let mut m = race(10);
        m[2 + 2 * 4] = assistant(&[("Bash", bash("python3 -m pytest -q -x", "Fifth"))], "");
        assert_eq!(tool_repeat(&m, &ignore()).unwrap().run, 5);
        // A different tool with the same arguments.
        let mut m = race(4);
        m.push(assistant(
            &[("Read", bash("python3 -m pytest -q test_calc.py", "x"))],
            "",
        ));
        m.push(result("r", "51 passed in 0.12s"));
        assert_eq!(tool_repeat(&m, &ignore()).unwrap().run, 1);
        // The conversation ends with a user turn, not a tool result.
        let mut m = race(12);
        m.push(ChatMessage::text(Role::User, "Stop and summarise."));
        assert_eq!(tool_repeat(&m, &ignore()), None);
        // Unparseable arguments compare as raw strings.
        let mut m = race(1);
        for _ in 0..3 {
            m.push(assistant(&[("Bash", "{not json".to_owned())], ""));
            m.push(result("r", "error"));
        }
        assert_eq!(tool_repeat(&m, &ignore()).unwrap().run, 3);
        // LG-D2 off.
        let guard = LoopGuard::new(LoopGuardConfig {
            tool_repeat_run: 0,
            ..LoopGuardConfig::default()
        });
        assert!(guard.check_messages(&race(12)).is_ok());
        assert_eq!(guard.counters(), (0, 0));
    }

    #[test]
    fn several_results_of_one_turn_form_one_pair() {
        let calls = || {
            vec![
                ("Read", json!({"file_path": "/w/calc.py"}).to_string()),
                ("Read", json!({"file_path": "/w/test_calc.py"}).to_string()),
            ]
        };
        let mut m = vec![ChatMessage::text(Role::User, "go")];
        for _ in 0..3 {
            m.push(assistant(&calls(), ""));
            m.push(result("a", "calc source"));
            m.push(result("b", "test source"));
        }
        let run = tool_repeat(&m, &ignore()).unwrap();
        assert_eq!(
            run,
            ToolRun {
                run: 3,
                names: vec!["Read".to_owned()]
            }
        );
        // An assistant turn without calls before the tool results is no pair.
        let m = vec![
            ChatMessage::text(Role::User, "go"),
            ChatMessage::text(Role::Assistant, "hm"),
            result("a", "x"),
        ];
        assert_eq!(tool_repeat(&m, &ignore()), None);
        // A tool result without its call turn ends no pair.
        assert_eq!(tool_repeat(&[result("a", "x")], &ignore()), None);
    }
}
