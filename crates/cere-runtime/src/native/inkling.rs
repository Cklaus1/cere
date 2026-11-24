//! Inkling 2 Mini's output format (inkling design note S6, family `inkling2_loop`): SGLang's
//! `--reasoning-parser inkling` (`InklingDetector` in `$S/parser/reasoning_parser.py`) and
//! `--tool-call-parser inkling` (`$S/function_call/inkling_detector.py`) of the pinned image, std only.
//!
//! A model turn is a sequence of typed messages, each closed by `<|end_message|>`, and the turn by the EOS
//! `<|content_model_end_sampling|>` (a stop token, so never in the text):
//!
//! ```text
//! <|message_model|><|content_thinking|>reasoning<|end_message|>
//! <|message_model|><|content_text|>visible answer<|end_message|>
//! <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
//! ```
//!
//! The text keeps its special tokens (SGLang's `_patch_reasoning_skip_special_tokens` turns
//! `skip_special_tokens` off for this parser). [`Blocks`] routes thinking to `reasoning_content` and text to
//! `content`, passing a tool message's framing through in the normal text, where [`parse_non_stream`] and
//! [`Tools`] read the calls: `arguments` is `json.dumps(args, ensure_ascii=False)` of the payload's `args`, the
//! name is the payload's (the header name is not read, and a name outside the request's tools is still a call, as
//! the reference surfaces it), and one malformed or unterminated call fails the canonical parse of the whole text
//! (then one call is recovered from the last marker, else the text stays text).
//!
//! All control tokens are ASCII, so byte offsets here cut the text where the reference's code-point offsets do.
use crate::native::reasoning::Split;
use crate::native::tool_parser::Call;
use crate::pyjson::{self, Py};

const MESSAGE_MODEL: &str = "<|message_model|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const END_MESSAGE: &str = "<|end_message|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const CONTENT_INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";

/// `<|end_message|>`'s id in the fixed framing overlay (`$S/parser/inkling_tokenizer.py`
/// `INKLING_SPECIAL_TOKEN_IDS`): SGLang's `think_end_ids` for this parser, so `usage.reasoning_tokens` counts the
/// output through its first `<|end_message|>` (`schedule_batch.py` `update_reasoning_tokens`; `require_reasoning` is
/// always on: the detector's `reasoning_default` is `"always"`).
pub const END_MESSAGE_ID: u32 = 200_010;

/// `INKLING_CONTROL_TOKENS`: every framing token plus the two control tokens without a framing id. The reasoning
/// split and the tool detector share it.
const CONTROL_TOKENS: [&str; 18] = [
    "<|endoftext|>",
    "<|message_user|>",
    MESSAGE_MODEL,
    "<|message_system|>",
    "<|message_tool|>",
    CONTENT_TEXT,
    "<|content_image|>",
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_THINKING,
    "<|content_audio_input|>",
    "<|content_tool_error|>",
    "<|content_xml|>",
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    END_MESSAGE,
    "<|audio_end|>",
    "<|content_invoke_tool|>",
    "<|model_trigger_generation|>",
];

/// The first control token at or after `from` (`_INKLING_CONTROL_RE.finditer`): every token is `<|name|>` with no
/// `|>` inside, so at most one matches at a position.
fn next_control(text: &str, from: usize) -> Option<(usize, &'static str)> {
    text[from..].match_indices("<|").find_map(|(i, _)| {
        let at = from + i;
        CONTROL_TOKENS
            .iter()
            .find(|t| text[at..].starts_with(**t))
            .map(|t| (at, *t))
    })
}

/// Whether the last `len` bytes of `text` are a strict prefix of `token`.
fn ends_with_prefix(text: &str, len: usize, token: &str) -> bool {
    len < token.len()
        && len <= text.len()
        && text.is_char_boundary(text.len() - len)
        && token.starts_with(&text[text.len() - len..])
}

/// `_partial_control_length`: the longest tail that may still become a control token.
fn partial_control_length(text: &str) -> usize {
    let longest = CONTROL_TOKENS.iter().map(|t| t.len()).max().unwrap_or(0);
    (1..longest.min(text.len() + 1))
        .rev()
        .find(|&len| {
            CONTROL_TOKENS
                .iter()
                .any(|t| ends_with_prefix(text, len, t))
        })
        .unwrap_or(0)
}

/// `_clean_normal_text`: every control token removed.
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pos = 0;
    while let Some((at, token)) = next_control(text, pos) {
        out.push_str(&text[pos..at]);
        pos = at + token.len();
    }
    out.push_str(&text[pos..]);
    out
}

/// The block a [`Blocks`] parser is in (`_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// After `<|message_model|>`: the header (a tool name) is held until the content kind arrives.
    Header,
    Reasoning,
    Content,
    /// A tool message: its framing and payload pass through as normal text.
    Tool,
}

/// The reasoning detector of one response (`InklingDetector`, `stream_reasoning = true`): `push` is
/// `parse_streaming_increment`, `finish` its `finish` (which drops a held partial token).
#[derive(Clone, Debug, Default)]
pub struct Blocks {
    kind: Option<Kind>,
    header: String,
    buffer: String,
}

impl Blocks {
    fn emit(&mut self, text: &str, out: &mut Split) {
        match self.kind {
            Some(Kind::Reasoning) => out.reasoning.push_str(text),
            Some(Kind::Header) => self.header.push_str(text),
            _ => out.normal.push_str(text),
        }
    }

    /// `_parse_blocks`.
    fn parse(&mut self, text: &str) -> Split {
        let mut out = Split::default();
        let mut pos = 0;
        while let Some((at, token)) = next_control(text, pos) {
            self.emit(&text[pos..at], &mut out);
            pos = at + token.len();
            if token == MESSAGE_MODEL {
                if matches!(self.kind, None | Some(Kind::Header)) {
                    self.header.clear();
                    self.kind = Some(Kind::Header);
                } else {
                    // Inside an open block the token is payload the model wrote.
                    self.emit(token, &mut out);
                }
            } else if token == CONTENT_INVOKE_TOOL_JSON || token == CONTENT_INVOKE_TOOL_TEXT {
                // The tool framing stays in the normal text for the tool detector.
                if self.kind == Some(Kind::Header) {
                    out.normal.push_str(MESSAGE_MODEL);
                    out.normal.push_str(&std::mem::take(&mut self.header));
                }
                out.normal.push_str(token);
                self.kind = Some(Kind::Tool);
            } else if self.kind == Some(Kind::Tool) {
                out.normal.push_str(token);
                if token == END_MESSAGE || token == CONTENT_MODEL_END_SAMPLING {
                    self.kind = None;
                }
            } else if token == CONTENT_THINKING {
                self.header.clear();
                self.kind = Some(Kind::Reasoning);
            } else if token == CONTENT_TEXT {
                self.header.clear();
                self.kind = Some(Kind::Content);
            } else if token == END_MESSAGE || token == CONTENT_MODEL_END_SAMPLING {
                self.header.clear();
                self.kind = None;
            }
        }
        self.emit(&text[pos..], &mut out);
        out
    }

    /// `parse_streaming_increment(text)`: a tail that may still become a control token waits.
    pub fn push(&mut self, text: &str) -> Split {
        let mut text = std::mem::take(&mut self.buffer) + text;
        let hold = partial_control_length(&text);
        self.buffer = text.split_off(text.len() - hold);
        self.parse(&text)
    }

    /// `finish()`: the state is reset; a held partial token is dropped.
    pub fn finish(&mut self) -> Split {
        *self = Self::default();
        Split::default()
    }
}

/// `detect_and_parse` of a whole output (the unary response; its `stream_reasoning = false` flushes the same text).
pub fn split(text: &str) -> Split {
    Blocks::default().parse(text)
}

/// `has_tool_call`: the text holds a json or a raw-text tool marker.
pub fn has_tool_call(text: &str) -> bool {
    text.contains(CONTENT_INVOKE_TOOL_JSON) || text.contains(CONTENT_INVOKE_TOOL_TEXT)
}

/// `_next_tool_marker`: the earliest json or text marker at or after `from`, and whether it is the json one.
fn next_tool_marker(text: &str, from: usize) -> Option<(usize, &'static str, bool)> {
    let json = text[from..]
        .find(CONTENT_INVOKE_TOOL_JSON)
        .map(|i| (from + i, CONTENT_INVOKE_TOOL_JSON, true));
    let raw = text[from..]
        .find(CONTENT_INVOKE_TOOL_TEXT)
        .map(|i| (from + i, CONTENT_INVOKE_TOOL_TEXT, false));
    match (json, raw) {
        (Some(j), Some(r)) => Some(if j.0 <= r.0 { j } else { r }),
        (j, r) => j.or(r),
    }
}

/// `_pending_tool_header_start`: a trailing `<|message_model|>` whose header holds no complete control token.
fn pending_header_start(text: &str) -> Option<usize> {
    let at = text.rfind(MESSAGE_MODEL)?;
    let header = &text[at + MESSAGE_MODEL.len()..];
    (!CONTROL_TOKENS.iter().any(|t| header.contains(t))).then_some(at)
}

/// `_split_trailing_tool_header(text)[0]`: the text before a pending tool header.
fn before_header(text: &str) -> &str {
    pending_header_start(text).map_or(text, |at| &text[..at])
}

/// Python's `str.strip()`: `char::is_whitespace` plus the separators U+001C..U+001F.
fn py_strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// `_tool_call_item`: `name` a string and `args` a dict, else no call.
fn call_item(payload: &Py, index: usize) -> Option<Call> {
    let name = payload.get("name")?.as_str()?;
    let args = payload.get("args").filter(|a| a.is_dict())?;
    Some(Call {
        index,
        name: name.to_owned(),
        arguments: args.dumps(pyjson::DUMPS_UTF8),
    })
}

/// `_canonical_json_call`: strict JSON (`NaN`/`Infinity` refused) holding a dict.
fn canonical_call(body: &str, index: usize) -> Option<Call> {
    let payload = pyjson::loads(py_strip(body), false)?;
    call_item(&payload, index)
}

/// `_text_tool_call`: a headerless raw-text invocation, nameless, `arguments` `json.dumps({"text": body})`.
fn text_call(body: &str, index: usize) -> Call {
    let text = Py::Dict(vec![("text".to_owned(), Py::Str(clean(body)))]);
    Call {
        index,
        name: String::new(),
        arguments: text.dumps(pyjson::DUMPS),
    }
}

/// `_recover_last_json_call`: one call (index 0) from the last json marker through the next `<|end_message|>` or
/// the end, constants allowed, with a non-empty string name.
fn recover_last_call(text: &str) -> Option<Call> {
    let last = text.rfind(CONTENT_INVOKE_TOOL_JSON)?;
    let body = last + CONTENT_INVOKE_TOOL_JSON.len();
    let candidate = match text[body..].find(END_MESSAGE) {
        Some(end) => &text[body..body + end],
        None => &text[body..],
    };
    let candidate = clean(candidate);
    let payload = pyjson::loads(py_strip(&candidate), true)?;
    payload.get("name")?.as_str().filter(|n| !n.is_empty())?;
    call_item(&payload, 0)
}

/// `_parse_canonical`: the visible text and every call when each marker frames a valid call, else `None`.
fn parse_canonical(text: &str) -> Option<(String, Vec<Call>)> {
    let mut calls = Vec::new();
    let mut normal = String::new();
    let mut pos = 0;
    while pos < text.len() {
        let Some((marker, token, json)) = next_tool_marker(text, pos) else {
            normal.push_str(&text[pos..]);
            break;
        };
        normal.push_str(before_header(&text[pos..marker]));
        let body = marker + token.len();
        let end = body + text[body..].find(END_MESSAGE)?;
        let call = if json {
            canonical_call(&text[body..end], calls.len())?
        } else {
            text_call(&text[body..end], calls.len())
        };
        calls.push(call);
        pos = end + END_MESSAGE.len();
    }
    Some((clean(&normal), calls))
}

/// The unary path (`FunctionCallParser.parse_non_stream` over `detect_and_parse`): no marker -> the text, no
/// calls; else the canonical parse, else one recovered call and no text, else the cleaned text and no calls.
pub fn parse_non_stream(text: &str) -> (String, Vec<Call>) {
    if !has_tool_call(text) {
        return (text.to_owned(), Vec::new());
    }
    if let Some(parsed) = parse_canonical(text) {
        return parsed;
    }
    match recover_last_call(text) {
        Some(call) => (String::new(), vec![call]),
        None => (clean(text), Vec::new()),
    }
}

/// The streaming tool detector of one response (`parse_streaming_increment`): a call goes out whole once its
/// `<|end_message|>` arrives; the first call that fails to frame recovers one call from the last marker (or sends
/// the buffer as text) and the rest of the response passes through as text. Its end (no `finish` override,
/// `base_format_detector.py:353-359`) drops what is held.
#[derive(Clone, Debug, Default)]
pub struct Tools {
    buffer: String,
    index: usize,
    passthrough: bool,
}

impl Tools {
    pub fn push(&mut self, text: &str) -> (String, Vec<Call>) {
        self.buffer.push_str(text);
        if self.passthrough {
            return (clean(&std::mem::take(&mut self.buffer)), Vec::new());
        }
        let mut normal = String::new();
        let mut calls = Vec::new();
        while !self.buffer.is_empty() {
            let Some((marker, token, json)) = next_tool_marker(&self.buffer, 0) else {
                let hold = self.safe_split();
                let held = self.buffer.split_off(hold);
                normal.push_str(&clean(&std::mem::replace(&mut self.buffer, held)));
                break;
            };
            let prefix = clean(before_header(&self.buffer[..marker]));
            let body = marker + token.len();
            let Some(end) = self.buffer[body..].find(END_MESSAGE).map(|i| body + i) else {
                // Incomplete: the text before the region goes out, the marker and partial body wait.
                normal.push_str(&prefix);
                self.buffer.drain(..marker);
                break;
            };
            let call = if json {
                match canonical_call(&self.buffer[body..end], self.index) {
                    Some(call) => call,
                    // The reference returns the recovery alone: this push's earlier text and calls are dropped.
                    None => return self.recover_or_fallback(),
                }
            } else {
                text_call(&self.buffer[body..end], self.index)
            };
            normal.push_str(&prefix);
            calls.push(call);
            self.index += 1;
            self.buffer.drain(..end + END_MESSAGE.len());
        }
        (normal, calls)
    }

    /// `parse_stream_end`: nothing (the held text is dropped).
    pub fn finish(&mut self) -> (String, Vec<Call>) {
        (String::new(), Vec::new())
    }

    /// `_split_safe_text`: where the text that may still be a forming tool header or a split control token starts.
    fn safe_split(&self) -> usize {
        let text = &self.buffer;
        if let Some(at) = pending_header_start(text) {
            return at;
        }
        // `_ends_with_partial_token` per token: the shortest tail that is a strict prefix of it.
        let partial = CONTROL_TOKENS
            .iter()
            .map(|t| {
                (1..text.len().min(t.len() - 1) + 1)
                    .find(|&len| ends_with_prefix(text, len, t))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        text.len() - partial
    }

    /// `_stream_recover_or_fallback`.
    fn recover_or_fallback(&mut self) -> (String, Vec<Call>) {
        self.passthrough = true;
        let buffered = std::mem::take(&mut self.buffer);
        match recover_last_call(&buffered) {
            Some(mut call) => {
                call.index = self.index;
                self.index += 1;
                (String::new(), vec![call])
            }
            None => (clean(&buffered), Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(index: usize, name: &str, arguments: &str) -> Call {
        Call {
            index,
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        }
    }

    /// Every split of `text` into two pieces streams the unary result.
    fn streams_as_unary(text: &str) {
        let want = split(text);
        for cut in (0..=text.len()).filter(|&c| text.is_char_boundary(c)) {
            let mut blocks = Blocks::default();
            let mut got = blocks.push(&text[..cut]);
            got.append(blocks.push(&text[cut..]));
            got.append(blocks.finish());
            assert_eq!(got, want, "cut {cut} of {text:?}");
        }
    }

    #[test]
    fn thinking_and_text_split() {
        let text = "<|message_model|><|content_thinking|>Need sky.<|end_message|>\
                    <|message_model|><|content_text|>Rayleigh <b>scattering</b>.<|end_message|>";
        assert_eq!(
            split(text),
            Split {
                reasoning: "Need sky.".to_owned(),
                normal: "Rayleigh <b>scattering</b>.".to_owned(),
            }
        );
        streams_as_unary(text);
    }

    #[test]
    fn a_turn_without_framing_is_content() {
        assert_eq!(
            split("plain"),
            Split {
                reasoning: String::new(),
                normal: "plain".to_owned()
            }
        );
        streams_as_unary("plain <| not a token");
    }

    #[test]
    fn unterminated_thinking_stays_reasoning() {
        let text = "<|message_model|><|content_thinking|>The user asks: Explain";
        assert_eq!(split(text).reasoning, "The user asks: Explain");
        assert_eq!(split(text).normal, "");
        streams_as_unary(text);
    }

    #[test]
    fn message_model_inside_a_block_is_payload() {
        let text = "<|message_model|><|content_text|>quote <|message_model|> here<|end_message|>";
        assert_eq!(split(text).normal, "quote <|message_model|> here");
        streams_as_unary(text);
    }

    #[test]
    fn tool_framing_passes_through_the_split() {
        let tool =
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\
                    \"args\":{\"city\":\"Paris\"}}<|end_message|>";
        let text = format!("<|message_model|><|content_thinking|>Need tool.<|end_message|>{tool}");
        let split = split(&text);
        assert_eq!(split.reasoning, "Need tool.");
        assert_eq!(split.normal, tool);
        streams_as_unary(&text);
    }

    #[test]
    fn a_held_partial_token_is_dropped_at_finish() {
        let mut blocks = Blocks::default();
        let got = blocks.push("<|message_model|><|content_text|>hi<|end_mes");
        assert_eq!(got.normal, "hi");
        assert_eq!(blocks.finish(), Split::default());
    }

    #[test]
    fn calls_parse_with_python_arguments() {
        let text = "<|message_model|>get_weather<|content_invoke_tool_json|> {\"name\":\"get_weather\",\
                    \"args\":{\"city\":\"Zürich\",\"days\":2,\"t\":1e2,\"city\":\"Paris\"}}<|end_message|>\
                    <|message_model|>now<|content_invoke_tool_json|>{\"name\":\"now\",\"args\":{}}<|end_message|>";
        assert_eq!(
            parse_non_stream(text),
            (
                String::new(),
                vec![
                    call(
                        0,
                        "get_weather",
                        r#"{"city": "Paris", "days": 2, "t": 100.0}"#
                    ),
                    call(1, "now", "{}"),
                ]
            )
        );
    }

    #[test]
    fn text_around_calls_is_cleaned() {
        let text = "Sure.<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":\"é\"}}\
                    <|end_message|>after<|end_message|>";
        assert_eq!(
            parse_non_stream(text),
            ("Sure.after".to_owned(), vec![call(0, "f", r#"{"a": "é"}"#)])
        );
    }

    #[test]
    fn the_payload_name_wins_over_the_header() {
        let text =
            "<|message_model|>header_name<|content_invoke_tool_json|>{\"name\":\"payload_name\",\
                    \"args\":{}}<|end_message|>";
        assert_eq!(
            parse_non_stream(text).1,
            vec![call(0, "payload_name", "{}")]
        );
    }

    #[test]
    fn raw_text_calls_are_nameless() {
        let text = "<|message_model|><|content_invoke_tool_text|>ls -l \"x\"<|end_message|>";
        assert_eq!(
            parse_non_stream(text),
            (
                String::new(),
                vec![call(0, "", r#"{"text": "ls -l \"x\""}"#)]
            )
        );
    }

    #[test]
    fn malformed_json_recovers_the_last_call_or_stays_text() {
        // The first call is malformed: the canonical parse fails and the last marker's call is recovered.
        let text = "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":}<|end_message|>\
                    <|message_model|>g<|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{\"x\":NaN}}<|end_message|>";
        assert_eq!(
            parse_non_stream(text),
            (String::new(), vec![call(0, "g", r#"{"x": NaN}"#)])
        );
        // Nothing recoverable: the cleaned text, no calls.
        let text = "a<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":[1]}<|end_message|>";
        assert_eq!(
            parse_non_stream(text),
            ("af{\"name\":\"f\",\"args\":[1]}".to_owned(), Vec::new())
        );
        // A canonical NaN is refused; an empty recovered name is refused.
        let text =
            "<|content_invoke_tool_json|>{\"name\":\"\",\"args\":{\"x\":NaN}}<|end_message|>";
        assert_eq!(parse_non_stream(text).1, Vec::new());
    }

    #[test]
    fn an_unterminated_call_is_recovered_through_the_end() {
        let text =
            "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}";
        assert_eq!(
            parse_non_stream(text),
            (String::new(), vec![call(0, "f", r#"{"a": 1}"#)])
        );
        let cut = "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":";
        assert_eq!(
            parse_non_stream(cut),
            ("f{\"name\":\"f\",\"args\":{\"a\":".to_owned(), Vec::new())
        );
    }

    #[test]
    fn no_marker_is_text() {
        assert_eq!(
            parse_non_stream("plain <|end_message|>"),
            ("plain <|end_message|>".to_owned(), Vec::new())
        );
    }

    /// The streamed `(normal, calls)` of `pieces`, concatenated.
    fn stream(pieces: &[&str]) -> (String, Vec<Call>) {
        let mut tools = Tools::default();
        let (mut normal, mut calls) = (String::new(), Vec::new());
        for piece in pieces {
            let (n, c) = tools.push(piece);
            normal.push_str(&n);
            calls.extend(c);
        }
        let (n, c) = tools.finish();
        normal.push_str(&n);
        calls.extend(c);
        (normal, calls)
    }

    #[test]
    fn streamed_calls_go_out_whole_at_their_end() {
        let text = "Hi.<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}\
                    <|end_message|><|message_model|>g<|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{}}\
                    <|end_message|>";
        let want = parse_non_stream(text);
        for cut in 0..=text.len() {
            if !text.is_char_boundary(cut) {
                continue;
            }
            assert_eq!(stream(&[&text[..cut], &text[cut..]]), want, "cut {cut}");
        }
        let mut tools = Tools::default();
        let (normal, calls) =
            tools.push("Hi.<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\"");
        assert_eq!((normal.as_str(), calls.len()), ("Hi.", 0));
    }

    #[test]
    fn a_stream_that_fails_to_frame_passes_through() {
        let bad = "<|message_model|>f<|content_invoke_tool_json|>{oops}<|end_message|>";
        let (normal, calls) = stream(&[bad, "<|message_model|><|content_text|>more"]);
        assert_eq!((normal.as_str(), calls), ("f{oops}more", Vec::new()));
        // An unterminated call is held and dropped at the end, as the reference's base `finish`.
        let (normal, calls) = stream(&[
            "x<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}",
        ]);
        assert_eq!((normal.as_str(), calls), ("x", Vec::new()));
    }
}
