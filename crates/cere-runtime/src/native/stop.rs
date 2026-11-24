//! Native stop strings (spec m24 D11): SGLang's per-step finish check at one accepted token per step
//! (`managers/schedule_batch.py:1439-1662`), the streaming hold (:1461-1488, `output_streamer.py:383-384`) and
//! the detokenizer's trim (`detokenizer_manager.py:176-206`).
//!
//! Each committed token: the window is the last `min(max(stop_token_lens) + 1, n)` output IDs decoded with
//! specials kept (`tail_str`, :1447-1457); the stops are tried in order and the first contained one finishes
//! (`FINISH_MATCHED_STR`, :1549-1559) with `finished_len = n` (`_locate_str_stop_finished_len` scans an empty
//! range at one token per step, :1514-1540); else a stop token finishes (:1488-1512); else the budget (:1652-1658).
//! A step that does not finish is output unless a window suffix is a stop prefix (`check_match_stop_str_prefix`).
use crate::native::detok::{Decoder, Vocab};

/// What ended the output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Matched {
    /// A stop string (`matched_stop` = the string).
    Str(String),
    /// A stop token (`matched_stop` = its ID).
    Token(u32),
    /// The output budget.
    Length,
    /// loop-guard design note LG-D1: the loop guard found the output repeating (`finish_reason` `repetition`,
    /// `matched_stop` null).
    Repetition,
}

/// A finished output: why, and how many output tokens it keeps (`finished_len`, usage counts these).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finished {
    pub matched: Matched,
    pub len: usize,
}

/// The verdict on one committed token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Not finished; `should_output` is false while the window ends with a stop prefix (a stream holds the step).
    Running {
        should_output: bool,
    },
    Finished(Finished),
}

/// The finish check of one output.
#[derive(Debug)]
pub struct StopMatcher<'v> {
    vocab: &'v Vocab,
    stops: Vec<String>,
    /// `stop_str_max_len + 1`.
    tail: usize,
    stop_token_ids: Vec<u32>,
    max_new_tokens: u64,
    ids: Vec<u32>,
    finished: Option<Finished>,
}

impl<'v> StopMatcher<'v> {
    /// `stop_token_lens`: the token count of each stop (the sidecar's `encode_chat`); `stop_token_ids`: the
    /// model's stop tokens; `max_new_tokens`: the output budget.
    pub fn new(
        vocab: &'v Vocab,
        stops: Vec<String>,
        stop_token_lens: &[u32],
        stop_token_ids: Vec<u32>,
        max_new_tokens: u64,
    ) -> Self {
        let max_len = stop_token_lens.iter().copied().max().unwrap_or(0) as usize;
        Self {
            vocab,
            stops,
            tail: max_len + 1,
            stop_token_ids,
            max_new_tokens,
            ids: Vec::new(),
            finished: None,
        }
    }

    /// The finish, once there is one.
    pub fn finished(&self) -> Option<&Finished> {
        self.finished.as_ref()
    }

    /// One committed token (`update_finish_state(1)`). After a finish the verdict stays that finish and the token
    /// is not counted.
    pub fn push(&mut self, id: u32) -> Step {
        if let Some(finished) = &self.finished {
            return Step::Finished(finished.clone());
        }
        self.ids.push(id);
        let n = self.ids.len();
        let tail = if self.stops.is_empty() {
            String::new()
        } else {
            let start = n - self.tail.min(n);
            self.vocab.decode_with(&self.ids[start..], false)
        };
        let matched = if let Some(stop) = self.stops.iter().find(|s| tail.contains(s.as_str())) {
            Some(Matched::Str(stop.clone()))
        } else if self.stop_token_ids.contains(&id) {
            Some(Matched::Token(id))
        } else if n as u64 >= self.max_new_tokens {
            Some(Matched::Length)
        } else {
            None
        };
        match matched {
            Some(matched) => {
                let finished = Finished { matched, len: n };
                self.finished = Some(finished.clone());
                Step::Finished(finished)
            }
            None => Step::Running {
                should_output: !self.holds(&tail),
            },
        }
    }

    /// The tokens counted so far.
    pub fn counted(&self) -> u64 {
        self.ids.len() as u64
    }

    /// One committed token on which the request's grammar terminated (spec m28 D8): the matcher's own verdict
    /// when it finishes the output here, else a stop on this token (`matched_stop` = `id`), as SGLang checks
    /// grammar termination after the other finish checks (`schedule_batch.py:1490-1492`).
    pub fn push_stop(&mut self, id: u32) -> Step {
        if let step @ Step::Finished(_) = self.push(id) {
            return step;
        }
        let finished = Finished {
            matched: Matched::Token(id),
            len: self.ids.len(),
        };
        self.finished = Some(finished.clone());
        Step::Finished(finished)
    }

    /// LG-D1: finishes the output on the last counted token with [`Matched::Repetition`] (no further token is
    /// counted); an output already finished keeps its finish.
    pub fn finish_repetition(&mut self) -> &Finished {
        self.finished.get_or_insert(Finished {
            matched: Matched::Repetition,
            len: self.ids.len(),
        })
    }

    /// `check_match_stop_str_prefix` on the window: a stop in it, or a window suffix that is a stop prefix
    /// (`schedule_batch.py:1461-1484`; code-point slices there, char boundaries here).
    fn holds(&self, tail: &str) -> bool {
        if tail.is_empty() {
            return false;
        }
        self.stops.iter().any(|stop| {
            tail.contains(stop.as_str())
                || stop
                    .char_indices()
                    .map(|(i, c)| i + c.len_utf8())
                    .any(|end| tail.ends_with(&stop[..end]))
        })
    }
}

/// `trim_matched_stop` for a stop string (`no_stop_trim` false): the text before its first occurrence, the whole
/// text when it does not occur or is empty (a falsy `matched` is returned untrimmed, `detokenizer_manager.py:182-184`;
/// `validate.rs` refuses empty stops, so only the fixture reaches this).
pub fn trim<'t>(text: &'t str, matched: &str) -> &'t str {
    if matched.is_empty() {
        return text;
    }
    text.find(matched).map_or(text, |at| &text[..at])
}

/// The response text of one output, built token by token as the matcher rules on them so a stream can send it as
/// it grows: SGLang's detokenizer output through `finished_len` under the request's `skip_special_tokens`, then
/// `trim_matched_stop` (`detokenizer_manager.py:176-206`). A token the matcher lets run is decoded; so is a
/// finishing token, except a matched stop token, special or not (SGLang's `output[:-1]`, spec m212 D8: a request's
/// `stop_token_ids` may name a non-special token, and with specials kept for the tool parser a special one would
/// show); the end flushes the decoder's held bytes and cuts the text at a matched stop string.
#[derive(Debug)]
pub struct Text<'v> {
    /// `None` once finished.
    decoder: Option<Decoder<'v>>,
    text: String,
}

impl<'v> Text<'v> {
    /// `decoder`: a fresh decoder under the request's `skip_special_tokens`.
    pub fn new(decoder: Decoder<'v>) -> Self {
        Self {
            decoder: Some(decoder),
            text: String::new(),
        }
    }

    /// The text so far (the trimmed final text after [`Text::finish`]).
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// One committed token and the matcher's verdict on it.
    pub fn push(&mut self, id: u32, step: &Step) {
        let Some(decoder) = self.decoder.as_mut() else {
            return;
        };
        let dropped = matches!(
            step,
            Step::Finished(Finished {
                matched: Matched::Token(_),
                ..
            })
        );
        if !dropped {
            self.text.push_str(&decoder.push(id));
        }
    }

    /// The final text of `finished` (U+FFFD for an incomplete trailing UTF-8 sequence).
    pub fn finish(&mut self, finished: &Finished) -> &str {
        if let Some(decoder) = self.decoder.take() {
            self.text.push_str(&decoder.finish());
        }
        if let Matched::Str(stop) = &finished.matched {
            let kept = trim(&self.text, stop).len();
            self.text.truncate(kept);
        }
        &self.text
    }

    /// Moves the text out (after [`Text::finish`]: the final text).
    pub fn take(&mut self) -> String {
        std::mem::take(&mut self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `a`=0, `b`=1, `<|end|>`=2 (special).
    fn vocab() -> Vocab {
        Vocab::from_json(
            br#"{"model":{"type":"BPE","vocab":{"a":0,"b":1}},"decoder":{"type":"ByteLevel"},
                "added_tokens":[{"id":2,"content":"<|end|>","special":true}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn string_beats_token_beats_length_and_holds_prefixes() {
        let v = vocab();
        let mut m = StopMatcher::new(&v, vec!["ab".to_owned()], &[2], vec![2], 4);
        assert_eq!(
            m.push(0),
            Step::Running {
                should_output: false
            }
        );
        assert_eq!(
            m.push(1),
            Step::Finished(Finished {
                matched: Matched::Str("ab".to_owned()),
                len: 2
            })
        );
        // A stop visible only with specials kept beats the stop token itself.
        let mut m = StopMatcher::new(&v, vec!["end".to_owned()], &[1], vec![2], 4);
        assert_eq!(
            m.push(1),
            Step::Running {
                should_output: true
            }
        );
        let finished = Finished {
            matched: Matched::Str("end".to_owned()),
            len: 2,
        };
        assert_eq!(m.push(2), Step::Finished(finished.clone()));
        let end = |skip| {
            let m = StopMatcher::new(&v, vec!["end".to_owned()], &[1], vec![2], 4);
            text(&v, skip, &[1, 2], m)
        };
        assert_eq!(end(true), "b");
        assert_eq!(end(false), "b<|");
        let mut m = StopMatcher::new(&v, vec![], &[], vec![2], 2);
        assert_eq!(
            m.push(0),
            Step::Running {
                should_output: true
            }
        );
        assert_eq!(
            m.push(0),
            Step::Finished(Finished {
                matched: Matched::Length,
                len: 2
            })
        );
    }

    /// `ids` through `m` and [`Text`] under `skip_special`: the final text.
    fn text(v: &Vocab, skip_special: bool, ids: &[u32], mut m: StopMatcher) -> String {
        let mut text = Text::new(v.decoder_with(skip_special));
        for &id in ids {
            let step = m.push(id);
            text.push(id, &step);
        }
        text.finish(m.finished().expect("finished")).to_owned()
    }

    /// A matched stop token stays out of the text, special or not, even with specials kept (SGLang's
    /// `output[:-1]`, spec m212 D8); the finishing token of a stop string and of the budget keep theirs.
    #[test]
    fn text_drops_a_matched_stop_token() {
        let v = vocab();
        for skip in [true, false] {
            let special = StopMatcher::new(&v, vec![], &[], vec![2], 8);
            assert_eq!(text(&v, skip, &[0, 1, 2], special), "ab", "skip {skip}");
            let plain = StopMatcher::new(&v, vec![], &[], vec![1], 8);
            assert_eq!(text(&v, skip, &[0, 0, 1], plain), "aa", "skip {skip}");
            let budget = StopMatcher::new(&v, vec![], &[], vec![], 3);
            let kept = if skip { "ab" } else { "ab<|end|>" };
            assert_eq!(text(&v, skip, &[0, 1, 2], budget), kept, "skip {skip}");
        }
    }
}
