//! The Qwen3 reasoning split (spec m22 Q8, Q9; m212 D4): SGLang's `Qwen3Detector` (`--reasoning-parser qwen3`)
//! with no continued final message and no forced non-empty content. The deployment default is
//! `enable_thinking: false`, that is `force_reasoning = false` ([`Reasoning::Detect`]); a request whose thinking
//! resolves on runs it with `force_reasoning = true` ([`Reasoning::Forced`], `serving_chat.py:1809-1823`), and
//! `separate_reasoning: false` runs no parser at all ([`Reasoning::Off`], `serving_chat.py:700, 1809`).
//!
//! [`split`] is `detect_and_parse` (the unary response); [`Stream`] is `parse_streaming_increment` with
//! `stream_reasoning = true` and `finish` (the SSE path). The response maps `normal` to `content` and a non-empty
//! `reasoning` to `reasoning_content`.
use crate::levers::Family;
use crate::native::inkling;

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
/// Qwen3.5 may open a tool call without closing the block; the detector treats it as an implicit close.
const TOOL_START: &str = "<tool_call>";

/// Text separated into reasoning and normal text ("" for SGLang's `None`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Split {
    pub reasoning: String,
    pub normal: String,
}

impl Split {
    fn normal(text: &str) -> Split {
        Split {
            reasoning: String::new(),
            normal: text.to_owned(),
        }
    }

    fn reasoning(text: &str) -> Split {
        Split {
            reasoning: text.to_owned(),
            normal: String::new(),
        }
    }

    /// Splits `text` at byte `at`: reasoning before, normal from `at + skip`.
    fn at(text: &str, at: usize, skip: usize) -> Split {
        Split {
            reasoning: text[..at].to_owned(),
            normal: text[at + skip..].to_owned(),
        }
    }

    /// Appends `other` to both parts.
    pub fn append(&mut self, other: Split) {
        self.reasoning.push_str(&other.reasoning);
        self.normal.push_str(&other.normal);
    }
}

/// How a response's text is parsed (spec m212 D4; Inkling: inkling design note S6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reasoning {
    /// `separate_reasoning: false`: no parser, all text is content.
    Off,
    /// `force_reasoning = false`: reasoning only inside a `<think>` the output opens.
    #[default]
    Detect,
    /// `force_reasoning = true` (thinking on): the output starts inside the block.
    Forced,
    /// The `inkling2_loop` family's typed messages ([`crate::native::inkling::Blocks`]): thinking is whatever a
    /// `<|content_thinking|>` message holds, whatever the request's thinking resolved to.
    Inkling,
}

impl Reasoning {
    /// The parser of a request whose thinking resolved to `thinking` (D4/D13), with `separate_reasoning`.
    pub fn resolve(thinking: bool, separate_reasoning: bool) -> Self {
        match (separate_reasoning, thinking) {
            (false, _) => Reasoning::Off,
            (true, false) => Reasoning::Detect,
            (true, true) => Reasoning::Forced,
        }
    }

    /// The parser of a request on a deployment of `family`: the family's own format, or [`Self::resolve`].
    pub fn for_family(family: Family, thinking: bool, separate_reasoning: bool) -> Self {
        match family {
            Family::Inkling2Loop if separate_reasoning => Reasoning::Inkling,
            _ => Self::resolve(thinking, separate_reasoning),
        }
    }
}

/// `detect_and_parse` of a whole output.
pub fn split(text: &str, mode: Reasoning) -> Split {
    let in_reasoning = match mode {
        Reasoning::Off => false,
        Reasoning::Detect => text.contains(THINK_START),
        Reasoning::Forced => true,
        Reasoning::Inkling => return inkling::split(text),
    };
    if !in_reasoning {
        return Split::normal(text);
    }
    let mut processed = text;
    while let Some(rest) = processed.strip_prefix(THINK_START) {
        processed = rest;
    }
    if let Some(end) = processed.find(THINK_END) {
        return Split::at(processed, end, THINK_END.len());
    }
    match processed.find(TOOL_START) {
        Some(tool) => Split::at(processed, tool, 0),
        // The block was cut off before its end token.
        None => Split::reasoning(processed),
    }
}

/// Length of the longest strict prefix of `token` that `text` ends with.
fn partial_suffix(text: &str, token: &str) -> usize {
    let (text, token) = (text.as_bytes(), token.as_bytes());
    (1..token.len().min(text.len() + 1))
        .rev()
        .find(|&i| token.starts_with(&text[text.len() - i..]))
        .unwrap_or(0)
}

/// The streaming detector of one response.
#[derive(Clone, Debug, Default)]
pub struct Stream {
    off: bool,
    buffer: String,
    stripped_think_start: bool,
    in_reasoning: bool,
    inkling: Option<inkling::Blocks>,
}

impl Stream {
    /// A detector in `mode`: [`Reasoning::Forced`] starts inside the block.
    pub fn new(mode: Reasoning) -> Self {
        Self {
            off: mode == Reasoning::Off,
            in_reasoning: mode == Reasoning::Forced,
            inkling: (mode == Reasoning::Inkling).then(inkling::Blocks::default),
            ..Self::default()
        }
    }

    /// `parse_streaming_increment(text)`.
    pub fn push(&mut self, text: &str) -> Split {
        if let Some(blocks) = &mut self.inkling {
            return blocks.push(text);
        }
        if self.off {
            return Split::normal(text);
        }
        self.buffer.push_str(text);
        // A buffer that may still become a marker waits for more text.
        if [THINK_START, THINK_END, TOOL_START]
            .iter()
            .any(|t| t.starts_with(self.buffer.as_str()) && *t != self.buffer)
        {
            return Split::default();
        }
        if !self.stripped_think_start {
            if let Some(start) = self.buffer.find(THINK_START) {
                self.buffer
                    .replace_range(start..start + THINK_START.len(), "");
                self.stripped_think_start = true;
                self.in_reasoning = true;
            }
        }
        let current = std::mem::take(&mut self.buffer);
        if !self.in_reasoning {
            return Split::normal(&current);
        }
        if let Some(end) = current.find(THINK_END) {
            self.in_reasoning = false;
            return Split::at(&current, end, THINK_END.len());
        }
        if let Some(tool) = current.find(TOOL_START) {
            self.in_reasoning = false;
            return Split::at(&current, tool, 0);
        }
        // In the block: hold back a tail that may begin the end or a tool-call token, or the start token while
        // it is not stripped yet (a forced block).
        let mut hold =
            partial_suffix(&current, THINK_END).max(partial_suffix(&current, TOOL_START));
        if !self.stripped_think_start {
            hold = hold.max(partial_suffix(&current, THINK_START));
        }
        let cut = current.len() - hold;
        self.buffer = current[cut..].to_owned();
        Split::reasoning(&current[..cut])
    }

    /// `finish()` at the end of the output: what is still buffered, as reasoning inside an unterminated block.
    pub fn finish(&mut self) -> Split {
        if let Some(blocks) = &mut self.inkling {
            return blocks.finish();
        }
        let rest = std::mem::take(&mut self.buffer);
        if !self.in_reasoning {
            return Split::normal(&rest);
        }
        Split::reasoning(rest.strip_prefix(THINK_START).unwrap_or(&rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(reasoning: &str, normal: &str) -> Split {
        Split {
            reasoning: reasoning.to_owned(),
            normal: normal.to_owned(),
        }
    }

    #[test]
    fn non_stream_mirrors_qwen3_detector() {
        assert_eq!(split("plain", Reasoning::Detect), s("", "plain"));
        assert_eq!(split("<think>a</think>b", Reasoning::Detect), s("a", "b"));
        assert_eq!(
            split("<think><think>a</think>b</think>c", Reasoning::Detect),
            s("a", "b</think>c")
        );
        assert_eq!(
            split("x<think>a</think>b", Reasoning::Detect),
            s("x<think>a", "b")
        );
        assert_eq!(
            split("<think>a<tool_call>b", Reasoning::Detect),
            s("a", "<tool_call>b")
        );
        assert_eq!(split("<think>open", Reasoning::Detect), s("open", ""));
        assert_eq!(split("a</think>b", Reasoning::Detect), s("", "a</think>b"));
    }

    #[test]
    fn stream_holds_partial_markers() {
        let mut st = Stream::default();
        assert_eq!(st.push("<thi"), s("", ""));
        assert_eq!(st.push("nk>ab</"), s("ab", ""));
        assert_eq!(st.push("think>c"), s("", "c"));
        assert_eq!(st.push("<"), s("", ""));
        assert_eq!(st.finish(), s("", "<"));
        let mut st = Stream::default();
        assert_eq!(st.push("<think>r</th"), s("r", ""));
        assert_eq!(st.finish(), s("</th", ""));
    }
}
