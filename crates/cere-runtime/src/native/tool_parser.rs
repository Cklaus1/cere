//! The native qwen3_coder tool-call detector (spec m24 D9, D10): the markup state machine of SGLang's
//! `Qwen3CoderDetector` (`$S/function_call/qwen3_coder_detector.py`) and its `FunctionCallParser` wrappers
//! (`function_call_parser.py:117-189`), std only. Parameter values are never converted here: every conversion is a
//! request to the caller ([`RawCall`] for the unary `arguments`, [`Item::Param`] for one streamed parameter), which
//! answers it with the sidecar's `tool_args` / `tool_param` ops, i.e. SGLang's own `_convert_param_value` and
//! `json.dumps` on the request's prepared tools (D8).
//!
//! All markers are ASCII, so byte offsets here cut the text where the reference's code-point offsets do.
use serde_json::Value;

use crate::levers::Family;
use crate::native::inkling;
use crate::native::reasoning::{self, Split};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_PREFIX: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";
/// The tags a `<` may still become (`qwen3_coder_detector.py:427-434`).
const TAGS: [&str; 6] = [
    TOOL_CALL_START,
    TOOL_CALL_END,
    FUNCTION_PREFIX,
    FUNCTION_END,
    PARAMETER_PREFIX,
    PARAMETER_END,
];

/// `has_tool_call` (:60-61): the text holds `<tool_call>`.
pub fn has_tool_call(text: &str) -> bool {
    text.contains(TOOL_CALL_START)
}

/// The prepared tool `_get_arguments_config` reads for `func` (:63-90): the first `type: "function"` entry whose
/// `function.name` is `func`; `None` = not defined (the value stays the raw string, but `null` still becomes
/// `None`).
pub fn tool_for<'t>(tools_prepared: &'t [Value], func: Option<&str>) -> Option<&'t Value> {
    let func = func?;
    tools_prepared
        .iter()
        .find(|t| t["type"] == "function" && t["function"]["name"] == func)
}

/// One call of the unary text: its position, name and parameters `(name, raw value)` in order, duplicates kept
/// (the reference's dict keeps the first position and the last value; `tool_args` rebuilds it the same way).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawCall {
    pub index: usize,
    pub name: String,
    pub params: Vec<(String, String)>,
}

/// A converted unary call: `arguments` is `json.dumps(parsed_params, ensure_ascii=False)` (:227).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub index: usize,
    pub name: String,
    pub arguments: String,
}

/// The first `start`..`end` span after `from` (`<start>(.*?)<end>`, DOTALL): the body's byte range and the end
/// of the match.
fn lazy_span(text: &str, from: usize, start: &str, end: &str) -> Option<(usize, usize, usize)> {
    let open = from + text[from..].find(start)?;
    let body = open + start.len();
    let close = body + text[body..].find(end)?;
    Some((body, close, close + end.len()))
}

/// `tool_call_regex.findall` (:34): the bodies of every `<tool_call>…</tool_call>`, left to right.
fn tool_call_bodies(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some((body, close, end)) = lazy_span(text, at, TOOL_CALL_START, TOOL_CALL_END) {
        out.push(&text[body..close]);
        at = end;
    }
    out
}

/// `tool_call_function_regex.findall` (:35-37), `<function=(.*?)</function>|<function=(.*)$` DOTALL, reduced to
/// `match[0] or match[1]`: a closed body, else everything after the prefix to the end (which ends the scan).
fn function_bodies(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(open) = text[at..].find(FUNCTION_PREFIX).map(|i| at + i) {
        let body = open + FUNCTION_PREFIX.len();
        match text[body..].find(FUNCTION_END) {
            Some(close) => {
                out.push(&text[body..body + close]);
                at = body + close + FUNCTION_END.len();
            }
            None => {
                out.push(&text[body..]);
                break;
            }
        }
    }
    out
}

/// Python's `$` without MULTILINE at byte `q` of `text`: the end, or just before a final `\n`.
fn at_dollar(text: &str, q: usize) -> bool {
    q == text.len() || (q + 1 == text.len() && text.as_bytes()[q] == b'\n')
}

/// `tool_call_parameter_regex.findall` (:38-41),
/// `<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)` DOTALL: the lazy body ends at the first
/// position where one alternative matches; only `</parameter>` is consumed.
fn parameter_bodies(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(open) = text[at..].find(PARAMETER_PREFIX).map(|i| at + i) {
        let body = open + PARAMETER_PREFIX.len();
        let mut q = body;
        let end = loop {
            let rest = &text[q..];
            if rest.starts_with(PARAMETER_END) {
                break q + PARAMETER_END.len();
            }
            if rest.starts_with(PARAMETER_PREFIX)
                || rest.starts_with(FUNCTION_END)
                || at_dollar(text, q)
            {
                break q;
            }
            q += rest.chars().next().map_or(1, char::len_utf8);
        };
        out.push(&text[body..q]);
        at = end;
    }
    out
}

/// The reference's value cleanup: one leading `\n`, then one trailing `\n`, removed.
fn strip_newlines(raw: &str) -> &str {
    let raw = raw.strip_prefix('\n').unwrap_or(raw);
    raw.strip_suffix('\n').unwrap_or(raw)
}

/// The markup half of `detect_and_parse` (:176-238): the normal text before the first call and the raw calls.
pub fn detect_and_parse(text: &str) -> (String, Vec<RawCall>) {
    if !has_tool_call(text) {
        return (text.to_owned(), Vec::new());
    }
    let mut bodies = tool_call_bodies(text);
    if bodies.is_empty() && text.contains(FUNCTION_PREFIX) {
        bodies = vec![text];
    }
    let mut calls = Vec::new();
    for body in bodies {
        for func in function_bodies(body) {
            let Some(name_end) = func.find('>') else {
                continue;
            };
            let params = parameter_bodies(&func[name_end + 1..])
                .into_iter()
                .filter_map(|p| {
                    let i = p.find('>')?;
                    Some((p[..i].to_owned(), strip_newlines(&p[i + 1..]).to_owned()))
                })
                .collect();
            calls.push(RawCall {
                index: calls.len(),
                name: func[..name_end].to_owned(),
                params,
            });
        }
    }
    let start = text
        .find(TOOL_CALL_START)
        .or_else(|| text.find(FUNCTION_PREFIX));
    let normal = match start {
        Some(i) if i > 0 => text[..i].to_owned(),
        _ => String::new(),
    };
    (normal, calls)
}

/// The unary path (`serving_chat.py:2006-2019` over `parse_non_stream`, `function_call_parser.py:132-152`): no
/// `<tool_call>` -> the whole text, no calls; else the normal text before the first call and the converted calls
/// (none for a malformed tail: truncated text). `convert` answers one call's `arguments`, or `None` for the
/// reference's conversion exception (:240-242): the whole text, no calls. Its `Err` (no conversion was made: the
/// converter failed) is the result.
pub fn parse_non_stream<E>(
    text: &str,
    mut convert: impl FnMut(&RawCall) -> Result<Option<String>, E>,
) -> Result<(String, Vec<Call>), E> {
    if !has_tool_call(text) {
        return Ok((text.to_owned(), Vec::new()));
    }
    let (normal, raw) = detect_and_parse(text);
    let mut calls = Vec::with_capacity(raw.len());
    for call in &raw {
        match convert(call)? {
            Some(arguments) => calls.push(Call {
                index: call.index,
                name: call.name.clone(),
                arguments,
            }),
            None => return Ok((text.to_owned(), Vec::new())),
        }
    }
    Ok((normal, calls))
}

/// One `ToolCallItem` of a streaming increment, before conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    /// A new call: `(tool_index, name, "")` (:270-292).
    Name { index: i64, name: String },
    /// A literal arguments fragment, `{` or `}` (:330-334, :394-403).
    Text { index: i64, text: &'static str },
    /// A closed parameter: the fragment is `", "` when `comma`, then `tool_param(func, name, raw)` (:336-376).
    Param {
        index: i64,
        func: Option<String>,
        name: String,
        raw: String,
        comma: bool,
    },
    /// A whole call (the Inkling detector, [`crate::native::inkling::Tools`]): name and `arguments` in one item; an
    /// empty name (a raw-text call) is sent as none, as `serving_chat.py`'s `if call_item.name`.
    Call(Call),
}

/// A converted `ToolCallItem`: `name` only on the first item of a call; `index` is the reference's
/// `current_tool_id` (-1 before any `<function=`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delta {
    pub index: i64,
    pub name: Option<String>,
    pub arguments: String,
}

impl Item {
    /// Converts the item; `param` answers `(func, name, raw)` with the `json_key_val` fragment.
    pub fn resolve<E>(
        self,
        param: impl FnOnce(Option<&str>, &str, &str) -> Result<String, E>,
    ) -> Result<Delta, E> {
        Ok(match self {
            Item::Name { index, name } => Delta {
                index,
                name: Some(name),
                arguments: String::new(),
            },
            Item::Text { index, text } => Delta {
                index,
                name: None,
                arguments: text.to_owned(),
            },
            Item::Param {
                index,
                func,
                name,
                raw,
                comma,
            } => {
                let fragment = param(func.as_deref(), &name, &raw)?;
                Delta {
                    index,
                    name: None,
                    arguments: if comma {
                        format!(", {fragment}")
                    } else {
                        fragment
                    },
                }
            }
            Item::Call(call) => Delta {
                index: call.index as i64,
                name: Some(call.name).filter(|name| !name.is_empty()),
                arguments: call.arguments,
            },
        })
    }
}

/// What one push yields: normal text and the call items in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Increment {
    pub normal: String,
    pub items: Vec<Item>,
}

/// `parse_streaming_increment` (:244-472) of one response: a cursor over the unparsed buffer.
#[derive(Clone, Debug)]
pub struct Stream {
    buffer: String,
    current_tool_id: i64,
    param_count: usize,
    json_started: bool,
    inside_tool_call: bool,
    func: Option<String>,
}

impl Default for Stream {
    fn default() -> Self {
        Self {
            buffer: String::new(),
            current_tool_id: -1,
            param_count: 0,
            json_started: false,
            inside_tool_call: false,
            func: None,
        }
    }
}

impl Stream {
    /// `parse_stream_chunk(text)`: what the text completes; a partial tag, a `<function=` without `>` and an
    /// unterminated parameter value are held (:299-301, :384-385, :441-448).
    pub fn push(&mut self, text: &str) -> Increment {
        self.buffer.push_str(text);
        let mut out = Increment::default();
        let mut pos = 0;
        while pos < self.buffer.len() {
            let slice = &self.buffer[pos..];
            if slice.starts_with(TOOL_CALL_START) {
                pos += TOOL_CALL_START.len();
                self.inside_tool_call = true;
                continue;
            }
            if slice.starts_with(FUNCTION_PREFIX) {
                let Some(end) = slice.find('>') else {
                    break;
                };
                let name = slice[FUNCTION_PREFIX.len()..end].to_owned();
                self.current_tool_id += 1;
                self.param_count = 0;
                self.json_started = false;
                self.func = Some(name.clone());
                out.items.push(Item::Name {
                    index: self.current_tool_id,
                    name,
                });
                pos += end + 1;
                continue;
            }
            if slice.starts_with(PARAMETER_PREFIX) {
                let Some(name_end) = slice.find('>') else {
                    break;
                };
                let rest = &slice[name_end + 1..];
                // The nearest of `</parameter>` (consumed), `<parameter=` and `</function>` (left in place); the
                // three never start at the same offset.
                let end = [
                    rest.find(PARAMETER_END).map(|i| (i, PARAMETER_END.len())),
                    rest.find(PARAMETER_PREFIX).map(|i| (i, 0)),
                    rest.find(FUNCTION_END).map(|i| (i, 0)),
                ]
                .into_iter()
                .flatten()
                .min_by_key(|&(i, _)| i);
                let Some((end, consumed)) = end else {
                    break;
                };
                if !self.json_started {
                    out.items.push(Item::Text {
                        index: self.current_tool_id,
                        text: "{",
                    });
                    self.json_started = true;
                }
                out.items.push(Item::Param {
                    index: self.current_tool_id,
                    func: self.func.clone(),
                    name: slice[PARAMETER_PREFIX.len()..name_end].to_owned(),
                    raw: strip_newlines(&rest[..end]).to_owned(),
                    comma: self.param_count > 0,
                });
                self.param_count += 1;
                pos += name_end + 1 + end + consumed;
                continue;
            }
            if slice.starts_with(FUNCTION_END) {
                if !self.json_started {
                    out.items.push(Item::Text {
                        index: self.current_tool_id,
                        text: "{",
                    });
                    self.json_started = true;
                }
                out.items.push(Item::Text {
                    index: self.current_tool_id,
                    text: "}",
                });
                pos += FUNCTION_END.len();
                self.func = None;
                continue;
            }
            if slice.starts_with(TOOL_CALL_END) {
                pos += TOOL_CALL_END.len();
                self.inside_tool_call = false;
                continue;
            }
            // Text up to the next `<`; a `<` that may still become a tag waits, any other is text.
            let take = match slice.find('<') {
                None => slice.len(),
                Some(0) if TAGS.iter().any(|t| t.starts_with(slice)) => break,
                Some(0) => 1,
                Some(i) => i,
            };
            if !self.inside_tool_call {
                out.normal.push_str(&slice[..take]);
            }
            pos += take;
        }
        self.buffer.drain(..pos);
        out
    }

    /// `parse_stream_end`: the detector has no `finish` override (`base_format_detector.py:353-359`), so the held
    /// text is dropped and nothing is emitted.
    pub fn finish(&mut self) -> Increment {
        Increment::default()
    }
}

/// One step of the native stream chain (D10): the reasoning text, then the normal text and call items left by
/// the tool parser (or the whole normal text without one).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChainStep {
    pub reasoning: String,
    pub normal: String,
    pub items: Vec<Item>,
}

/// The tool-call format of a deployment, by its family (`--tool-call-parser`): `qwen3_coder`, or `inkling` for the
/// `inkling2_loop` family (inkling design note S6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Qwen3Coder,
    Inkling,
}

impl Format {
    pub fn of(family: Family) -> Self {
        match family {
            Family::Inkling2Loop => Format::Inkling,
            _ => Format::Qwen3Coder,
        }
    }
}

/// The streaming tool detector of a [`Format`].
#[derive(Clone, Debug)]
enum Tools {
    Qwen3Coder(Stream),
    Inkling(inkling::Tools),
}

impl Tools {
    fn new(format: Format) -> Self {
        match format {
            Format::Qwen3Coder => Tools::Qwen3Coder(Stream::default()),
            Format::Inkling => Tools::Inkling(inkling::Tools::default()),
        }
    }

    fn increment((normal, calls): (String, Vec<Call>)) -> Increment {
        Increment {
            normal,
            items: calls.into_iter().map(Item::Call).collect(),
        }
    }

    fn push(&mut self, text: &str) -> Increment {
        match self {
            Tools::Qwen3Coder(stream) => stream.push(text),
            Tools::Inkling(tools) => Self::increment(tools.push(text)),
        }
    }

    fn finish(&mut self) -> Increment {
        match self {
            Tools::Qwen3Coder(stream) => stream.finish(),
            Tools::Inkling(tools) => Self::increment(tools.finish()),
        }
    }
}

/// `reasoning.rs` -> tool parser per piece, as `serving_chat.py:700-748`: on the last piece of a non-abort
/// finish, the reasoning parser's end text joins the piece before the tool parser sees it, then the tool parser's
/// own end (nothing) follows. An abort just stops pushing.
#[derive(Clone, Debug, Default)]
pub struct Chain {
    reasoning: reasoning::Stream,
    tools: Option<Tools>,
}

impl Chain {
    /// `tools`: the deployment's tool format when the request has tools and `tool_choice` `auto`; `mode`: the
    /// reasoning parser (spec m212 D4).
    pub fn new(tools: Option<Format>, mode: reasoning::Reasoning) -> Self {
        Self {
            reasoning: reasoning::Stream::new(mode),
            tools: tools.map(Tools::new),
        }
    }

    /// One piece; `finish` = the last piece of a stop/length finish.
    pub fn push(&mut self, piece: &str, finish: bool) -> ChainStep {
        let mut split: Split = self.reasoning.push(piece);
        if finish {
            split.append(self.reasoning.finish());
        }
        let Some(tools) = &mut self.tools else {
            return ChainStep {
                reasoning: split.reasoning,
                normal: split.normal,
                items: Vec::new(),
            };
        };
        let mut increment = tools.push(&split.normal);
        if finish {
            let end = tools.finish();
            increment.normal.push_str(&end.normal);
            increment.items.extend(end.items);
        }
        ChainStep {
            reasoning: split.reasoning,
            normal: increment.normal,
            items: increment.items,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(index: usize, name: &str, params: &[(&str, &str)]) -> RawCall {
        RawCall {
            index,
            name: name.to_owned(),
            params: params
                .iter()
                .map(|(n, v)| ((*n).to_owned(), (*v).to_owned()))
                .collect(),
        }
    }

    #[test]
    fn unary_markup_mirrors_the_regexes() {
        let text = "hi <tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=b>x\n</function>\n</tool_call>";
        let (normal, calls) = detect_and_parse(text);
        assert_eq!(normal, "hi ");
        assert_eq!(calls, vec![raw(0, "f", &[("a", "1"), ("b", "x")])]);
        // Unclosed call: the `<function=` fallback over the whole text; `$` stops before a final newline.
        let (normal, calls) = detect_and_parse("<tool_call><function=g><parameter=p>v\n\n");
        assert_eq!(normal, "");
        assert_eq!(calls, vec![raw(0, "g", &[("p", "v")])]);
        // A function without `>` is skipped; no `<tool_call>` = plain text even with `<function=`.
        assert_eq!(
            detect_and_parse("<tool_call><function=x</tool_call>").1,
            vec![]
        );
        assert_eq!(
            detect_and_parse("a<function=f>"),
            ("a<function=f>".to_owned(), vec![])
        );
    }

    /// `None` (the reference's conversion exception) is the whole text, no calls; an `Err` is the result.
    #[test]
    fn parse_non_stream_conversion_failure_is_the_whole_text() {
        let text = "x<tool_call><function=f></function></tool_call>";
        let (normal, calls) =
            parse_non_stream(text, |_| Ok::<_, ()>(Some("{}".to_owned()))).unwrap();
        assert_eq!(normal, "x");
        assert_eq!(calls.len(), 1);
        let (normal, calls) = parse_non_stream(text, |_| Ok::<_, ()>(None)).unwrap();
        assert_eq!((normal.as_str(), calls.len()), (text, 0));
        assert_eq!(parse_non_stream(text, |_| Err(7)), Err(7));
    }

    #[test]
    fn stream_holds_partial_tags_and_values() {
        let mut s = Stream::default();
        assert_eq!(s.push("a<tool").normal, "a");
        let inc = s.push("_call><function=f");
        assert!(inc.normal.is_empty() && inc.items.is_empty());
        let inc = s.push("><parameter=p>v");
        assert_eq!(
            inc.items,
            vec![Item::Name {
                index: 0,
                name: "f".to_owned()
            }]
        );
        let inc = s.push("</parameter></function></tool_call>b");
        assert_eq!(inc.normal, "b");
        assert_eq!(
            inc.items,
            vec![
                Item::Text {
                    index: 0,
                    text: "{"
                },
                Item::Param {
                    index: 0,
                    func: Some("f".to_owned()),
                    name: "p".to_owned(),
                    raw: "v".to_owned(),
                    comma: false
                },
                Item::Text {
                    index: 0,
                    text: "}"
                },
            ]
        );
        assert_eq!(s.push("<").normal, "");
        assert_eq!(s.finish(), Increment::default());
    }
}
