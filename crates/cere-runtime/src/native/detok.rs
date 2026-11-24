//! The Rust detokenizer (spec m22 Q9): decodes generated token IDs the way the Hugging Face `tokenizers` library
//! decodes them for SGLang (`decode(ids, skip_special_tokens=True)`), from the artifact's `tokenizer.json`.
//!
//! Every ID maps to its token string, added tokens first, then the BPE vocabulary; an ID neither knows is skipped,
//! and so is a special added token. The `ByteLevel` decoder maps each character of a token back to its byte through
//! the GPT-2 byte alphabet (a token with a character outside it contributes its own UTF-8 bytes), and the joined
//! bytes are read as UTF-8 with one U+FFFD per invalid sequence (`String::from_utf8_lossy`). A `tokenizer.json`
//! without a decoder joins the token strings with one space. Any other model or decoder type is refused.
//!
//! [`Decoder`] is the incremental form for streaming: it emits text as soon as its bytes form complete UTF-8 and
//! holds an incomplete trailing sequence back, so the concatenation of its pieces equals [`Vocab::decode`].
use std::collections::HashSet;
use std::fmt;
use std::path::Path;

use serde_json::Value;

/// Token IDs at or above this are refused (the table is indexed by ID; Qwen's largest is 248,076).
const MAX_TOKEN_ID: u32 = 1 << 22;

/// The decoder of a `tokenizer.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Join {
    /// `ByteLevel`: token bytes concatenated, read as UTF-8.
    Bytes,
    /// No decoder: token strings joined with one space.
    Spaces,
}

/// One known token ID.
#[derive(Clone, Debug)]
struct Token {
    /// What the token contributes: its bytes under `ByteLevel`, else its UTF-8 string.
    bytes: Box<[u8]>,
    /// Skipped by `skip_special_tokens`.
    special: bool,
}

/// Why a `tokenizer.json` could not be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetokError(pub String);

impl fmt::Display for DetokError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer.json: {}", self.0)
    }
}

impl std::error::Error for DetokError {}

/// The ID → token table of one `tokenizer.json`.
#[derive(Clone, Debug)]
pub struct Vocab {
    tokens: Vec<Option<Token>>,
    join: Join,
    /// The token spelled `</think>` (an added token over a vocabulary entry), if any: SGLang's `think_end_ids`
    /// (spec m212 D7, `usage.reasoning_tokens`).
    think_end: Option<u32>,
    /// ApiFit AF-L7: the lowest non-special tokens spelled `"\n"` and `"\n\n"`, the template's thinking closure
    /// `"\n</think>\n\n"` around [`Vocab::think_end`].
    newline: Option<u32>,
    blank_line: Option<u32>,
}

/// Characters of the GPT-2 byte alphabet: printable Latin-1 stands for itself, the other 68 bytes for
/// `U+0100..=U+0143` in byte order.
const ALPHABET_CHARS: usize = 0x144;

/// The GPT-2 byte alphabet inverted: entry `c` is the byte that character `U+c` stands for.
fn byte_alphabet() -> [Option<u8>; ALPHABET_CHARS] {
    let printable = |b: u8| matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut table = [None; ALPHABET_CHARS];
    let mut shifted = 0x100;
    for b in 0..=u8::MAX {
        if printable(b) {
            table[b as usize] = Some(b);
        } else {
            table[shifted] = Some(b);
            shifted += 1;
        }
    }
    table
}

fn token_bytes(token: &str, join: Join, alphabet: &[Option<u8>; ALPHABET_CHARS]) -> Box<[u8]> {
    match join {
        Join::Spaces => token.as_bytes().into(),
        Join::Bytes => token
            .chars()
            .map(|c| alphabet.get(c as usize).copied().flatten())
            .collect::<Option<Vec<u8>>>()
            .map_or_else(|| token.as_bytes().into(), Vec::into_boxed_slice),
    }
}

fn field<'a>(value: &'a Value, path: &str) -> Result<&'a Value, DetokError> {
    value
        .pointer(path)
        .ok_or_else(|| DetokError(format!("missing {path}")))
}

impl Vocab {
    /// Reads `<artifact_dir>/tokenizer.json`.
    pub fn load(artifact_dir: &Path) -> Result<Vocab, DetokError> {
        let path = artifact_dir.join("tokenizer.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| DetokError(format!("reading {}: {e}", path.display())))?;
        Vocab::from_json(&bytes)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Vocab, DetokError> {
        let json: Value =
            serde_json::from_slice(bytes).map_err(|e| DetokError(format!("not JSON: {e}")))?;
        let model_type = field(&json, "/model/type")?;
        if model_type != "BPE" {
            return Err(DetokError(format!("model type {model_type} is not BPE")));
        }
        let join = match json.get("decoder") {
            None | Some(Value::Null) => Join::Spaces,
            Some(d) if d.get("type").is_some_and(|t| t == "ByteLevel") => Join::Bytes,
            Some(d) => {
                return Err(DetokError(format!(
                    "decoder {} is not ByteLevel",
                    d.get("type").unwrap_or(&Value::Null)
                )))
            }
        };
        let vocab = field(&json, "/model/vocab")?
            .as_object()
            .ok_or_else(|| DetokError("/model/vocab is not an object".to_owned()))?;
        let id_of = |id: &Value| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| id < MAX_TOKEN_ID)
        };
        let mut entries: Vec<(u32, &str)> = Vec::with_capacity(vocab.len());
        for (token, id) in vocab {
            let id =
                id_of(id).ok_or_else(|| DetokError(format!("vocab id {id} is out of range")))?;
            entries.push((id, token.as_str()));
        }
        let added = match json.get("added_tokens") {
            None | Some(Value::Null) => &[][..],
            Some(Value::Array(a)) => a.as_slice(),
            Some(_) => return Err(DetokError("added_tokens is not a list".to_owned())),
        };
        let mut specials = HashSet::new();
        for a in added {
            let id = a.get("id").and_then(id_of);
            let content = a.get("content").and_then(Value::as_str);
            let special = a.get("special").and_then(Value::as_bool);
            let (Some(id), Some(content), Some(special)) = (id, content, special) else {
                return Err(DetokError(
                    "an added token lacks an id in range, a string content or a boolean special"
                        .to_owned(),
                ));
            };
            if special {
                specials.insert(content);
            }
            // Later entries win: added tokens over the vocabulary.
            entries.push((id, content));
        }
        let think_end = entries
            .iter()
            .rev()
            .find(|(_, token)| *token == "</think>")
            .map(|(id, _)| *id);
        let alphabet = byte_alphabet();
        let size = entries.iter().map(|e| e.0 as usize + 1).max().unwrap_or(0);
        let mut tokens: Vec<Option<Token>> = vec![None; size];
        for (id, token) in entries {
            // A token is special when its string is a special added token's.
            tokens[id as usize] = Some(Token {
                bytes: token_bytes(token, join, &alphabet),
                special: specials.contains(token),
            });
        }
        let spelled = |bytes: &[u8]| {
            tokens
                .iter()
                .position(|t| t.as_ref().is_some_and(|t| !t.special && &*t.bytes == bytes))
                .map(|id| id as u32)
        };
        let (newline, blank_line) = (spelled(b"\n"), spelled(b"\n\n"));
        Ok(Vocab {
            tokens,
            join,
            think_end,
            newline,
            blank_line,
        })
    }

    /// What `id` contributes; `None` when it is unknown, or special under `skip_special`.
    fn piece(&self, id: u32, skip_special: bool) -> Option<&[u8]> {
        match self.tokens.get(id as usize) {
            Some(Some(t)) if !(skip_special && t.special) => Some(&t.bytes),
            _ => None,
        }
    }

    /// The `</think>` token (spec m212 D7), if the vocabulary has one.
    pub fn think_end(&self) -> Option<u32> {
        self.think_end
    }

    /// ApiFit AF-L7: the tokens that close a thinking cut at its budget, as the template renders a thinking turn
    /// (`"\n</think>\n\n"`): the thinking part (`"\n"` when `room` allows it, then `</think>`; at least `</think>`)
    /// and the token after it (`"\n\n"`). Either newline is left out when the vocabulary lacks it; `None` without
    /// a `</think>` token.
    pub fn thinking_closure(&self, room: u64) -> Option<(Vec<u32>, Option<u32>)> {
        let end = self.think_end?;
        let thinking = match self.newline.filter(|_| room >= 2) {
            Some(newline) => vec![newline, end],
            None => vec![end],
        };
        Some((thinking, self.blank_line))
    }

    /// Whether `id` is a special added token (an unknown id is not).
    pub fn is_special(&self, id: u32) -> bool {
        matches!(self.tokens.get(id as usize), Some(Some(t)) if t.special)
    }

    /// `decode(ids, skip_special_tokens=True)`.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_with(ids, true)
    }

    /// `decode(ids, skip_special_tokens=skip_special)` (spec m24 D11: the stop-string window keeps specials).
    pub fn decode_with(&self, ids: &[u32], skip_special: bool) -> String {
        let mut d = self.decoder_with(skip_special);
        let mut text = String::new();
        for &id in ids {
            d.push_into(id, &mut text);
        }
        d.finish_into(&mut text);
        text
    }

    /// An incremental [`Vocab::decode`] starting at the beginning of a text.
    pub fn decoder(&self) -> Decoder<'_> {
        self.decoder_with(true)
    }

    /// An incremental [`Vocab::decode_with`] starting at the beginning of a text.
    pub fn decoder_with(&self, skip_special: bool) -> Decoder<'_> {
        Decoder {
            vocab: self,
            skip_special,
            pending: Vec::new(),
            started: false,
        }
    }
}

/// Incremental [`Vocab::decode_with`]: the pieces of [`Decoder::push`] then [`Decoder::finish`] concatenate to it.
#[derive(Debug)]
pub struct Decoder<'v> {
    vocab: &'v Vocab,
    skip_special: bool,
    /// An incomplete UTF-8 sequence at the end of the bytes so far.
    pending: Vec<u8>,
    /// A token was emitted (the space joiner goes before every later token).
    started: bool,
}

impl Decoder<'_> {
    /// The text that `id` completes (empty while a UTF-8 sequence is incomplete or the ID is skipped).
    pub fn push(&mut self, id: u32) -> String {
        let mut text = String::new();
        self.push_into(id, &mut text);
        text
    }

    /// The rest at the end of the IDs: U+FFFD for an incomplete trailing sequence.
    pub fn finish(mut self) -> String {
        let mut text = String::new();
        self.finish_into(&mut text);
        text
    }

    fn push_into(&mut self, id: u32, out: &mut String) {
        let Some(piece) = self.vocab.piece(id, self.skip_special) else {
            return;
        };
        if self.vocab.join == Join::Spaces && self.started {
            self.pending.push(b' ');
        }
        self.started = true;
        self.pending.extend_from_slice(piece);
        let mut rest = self.pending.as_slice();
        loop {
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    out.push_str(valid);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    let (valid, after) = rest.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(valid).expect("valid up to the error"));
                    match e.error_len() {
                        Some(n) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            rest = &after[n..];
                        }
                        None => {
                            rest = after;
                            break;
                        }
                    }
                }
            }
        }
        let keep = rest.len();
        let start = self.pending.len() - keep;
        self.pending.drain(..start);
    }

    fn finish_into(&mut self, out: &mut String) {
        if !self.pending.is_empty() {
            out.push(char::REPLACEMENT_CHARACTER);
            self.pending.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ByteLevel tokenizer.json: the GPT-2 byte alphabet for "Hé" split across two tokens, a space token, two
    /// added tokens (one special) and one id the model does not know.
    fn byte_level() -> Vocab {
        let json = serde_json::json!({
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {"H": 0, "Ã": 1, "©": 2, "Ġ": 3, "Hi": 4}},
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            "added_tokens": [
                {"id": 10, "content": "<|im_end|>", "special": true},
                {"id": 11, "content": "<think>", "special": false},
            ],
        });
        Vocab::from_json(json.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn byte_level_decode_matches_hf() {
        let v = byte_level();
        assert_eq!(v.decode(&[0, 1, 2]), "Hé");
        assert_eq!(v.decode(&[11, 4, 3, 0, 10]), "<think>Hi H");
        // `skip_special_tokens=False` keeps the special added token.
        assert_eq!(
            v.decode_with(&[11, 4, 3, 0, 10], false),
            "<think>Hi H<|im_end|>"
        );
        // An unknown id is skipped; a lone UTF-8 lead byte at the end is one U+FFFD.
        assert_eq!(v.decode(&[4, 99, 1]), "Hi\u{FFFD}");
        // A broken sequence mid-text: the lead byte is one U+FFFD, decoding resumes at the next byte.
        assert_eq!(v.decode(&[1, 0, 2]), "\u{FFFD}H\u{FFFD}");
    }

    #[test]
    fn incremental_decode_holds_partial_utf8() {
        let v = byte_level();
        let mut d = v.decoder();
        assert_eq!(d.push(0), "H");
        assert_eq!(d.push(1), "");
        assert_eq!(d.push(2), "é");
        assert_eq!(d.push(10), "");
        assert_eq!(d.push(1), "");
        assert_eq!(d.push(0), "\u{FFFD}H");
        assert_eq!(d.push(1), "");
        assert_eq!(d.finish(), "\u{FFFD}");
    }

    #[test]
    fn incremental_equals_whole_decode() {
        let v = byte_level();
        let ids = [11, 1, 2, 3, 1, 0, 4, 10, 2, 1];
        let mut d = v.decoder();
        let mut text: String = ids.iter().map(|&id| d.push(id)).collect();
        text.push_str(&d.finish());
        assert_eq!(text, v.decode(&ids));
    }

    #[test]
    fn decoder_absent_joins_with_spaces() {
        let json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2}}}"#;
        let v = Vocab::from_json(json.as_bytes()).unwrap();
        assert_eq!(v.decode(&[2, 0, 7, 1]), "ab a b");
        let mut d = v.decoder();
        let pieces: Vec<String> = [2, 7, 0, 1].iter().map(|&id| d.push(id)).collect();
        assert_eq!(pieces, ["ab", "", " a", " b"]);
        assert_eq!(d.finish(), "");
    }

    #[test]
    fn unsupported_tokenizers_are_refused() {
        let cases = [
            r#"{"model":{"type":"WordPiece","vocab":{"a":0}}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":0}},"decoder":{"type":"Metaspace"}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":0}},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":"x"}]}"#,
            r#"{"model":{"type":"BPE"}}"#,
            "not json",
        ];
        for json in cases {
            assert!(Vocab::from_json(json.as_bytes()).is_err(), "{json}");
        }
    }
}
