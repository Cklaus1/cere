//! Python's `json` output of raw JSON (`json.loads` then `json.dumps`), key order kept: the Anthropic surface's
//! tool-use `input` and the Inkling tool detector's `arguments` (`native::inkling`) are written as SGLang writes them.

use std::fmt::Write as _;

/// A value as Python's `json.loads` returns it: a dict keeps its keys' first positions and last values, numbers keep
/// their kind (an integer's digits verbatim, a float as `float`).
#[derive(Clone, Debug, PartialEq)]
pub enum Py<'a> {
    /// `true`, `false`, `null`, and with constants allowed `NaN`, `Infinity`, `-Infinity`.
    Literal(&'a str),
    Int(&'a str),
    Float(f64),
    Str(String),
    List(Vec<Py<'a>>),
    Dict(Vec<(String, Py<'a>)>),
}

/// How `json.dumps` writes: `ensure_ascii` escapes every character outside `' '..='~'`; `spaced` separates with
/// `", "`/`": "` (the default) instead of `","`/`":"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    pub ensure_ascii: bool,
    pub spaced: bool,
}

/// `json.dumps(v)`.
pub const DUMPS: Style = Style {
    ensure_ascii: true,
    spaced: true,
};
/// `json.dumps(v, ensure_ascii=False)`.
pub const DUMPS_UTF8: Style = Style {
    ensure_ascii: false,
    spaced: true,
};
/// Starlette's `JSONResponse` (`json.dumps(v, ensure_ascii=False, separators=(",", ":"))`).
pub const STARLETTE: Style = Style {
    ensure_ascii: false,
    spaced: false,
};

impl<'a> Py<'a> {
    /// `value.get(key)` of a dict; `None` for any other value.
    pub fn get(&self, key: &str) -> Option<&Py<'a>> {
        match self {
            Py::Dict(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Py::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_dict(&self) -> bool {
        matches!(self, Py::Dict(_))
    }

    /// `json.dumps(value)` in `style`.
    pub fn dumps(&self, style: Style) -> String {
        let mut out = String::new();
        py_write(self, style, &mut out);
        out
    }
}

struct PyParser<'a> {
    text: &'a str,
    pos: usize,
    /// `NaN`, `Infinity` and `-Infinity` parse (`json.loads`'s default `parse_constant`).
    constants: bool,
}

impl<'a> PyParser<'a> {
    fn skip_ws(&mut self) {
        let bytes = self.text.as_bytes();
        while self.pos < bytes.len() && matches!(bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn string(&mut self) -> Option<String> {
        let bytes = self.text.as_bytes();
        let start = self.pos;
        self.pos += 1;
        while self.pos < bytes.len() {
            match bytes[self.pos] {
                b'\\' => self.pos += 2,
                b'"' => {
                    self.pos += 1;
                    return serde_json::from_str(&self.text[start..self.pos]).ok();
                }
                _ => self.pos += 1,
            }
        }
        None
    }

    fn value(&mut self) -> Option<Py<'a>> {
        self.skip_ws();
        let bytes = self.text.as_bytes();
        match self.peek()? {
            b'{' => {
                self.pos += 1;
                let mut entries: Vec<(String, Py<'a>)> = Vec::new();
                self.skip_ws();
                if self.peek()? == b'}' {
                    self.pos += 1;
                    return Some(Py::Dict(entries));
                }
                loop {
                    self.skip_ws();
                    let key = self.string()?;
                    self.skip_ws();
                    if self.peek()? != b':' {
                        return None;
                    }
                    self.pos += 1;
                    let value = self.value()?;
                    // A repeated key keeps its first position and takes the last value (a Python dict).
                    match entries.iter_mut().find(|(k, _)| *k == key) {
                        Some(entry) => entry.1 = value,
                        None => entries.push((key, value)),
                    }
                    self.skip_ws();
                    match self.peek()? {
                        b',' => self.pos += 1,
                        b'}' => {
                            self.pos += 1;
                            return Some(Py::Dict(entries));
                        }
                        _ => return None,
                    }
                }
            }
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.peek()? == b']' {
                    self.pos += 1;
                    return Some(Py::List(items));
                }
                loop {
                    items.push(self.value()?);
                    self.skip_ws();
                    match self.peek()? {
                        b',' => self.pos += 1,
                        b']' => {
                            self.pos += 1;
                            return Some(Py::List(items));
                        }
                        _ => return None,
                    }
                }
            }
            b'"' => self.string().map(Py::Str),
            _ => {
                const LITERALS: [&str; 6] =
                    ["true", "false", "null", "NaN", "Infinity", "-Infinity"];
                let allowed = if self.constants {
                    &LITERALS[..]
                } else {
                    &LITERALS[..3]
                };
                let rest = &self.text[self.pos..];
                match allowed.iter().find(|l| rest.starts_with(**l)) {
                    Some(l) => {
                        let start = self.pos;
                        self.pos += l.len();
                        Some(Py::Literal(&self.text[start..self.pos]))
                    }
                    None => self.number(bytes),
                }
            }
        }
    }

    /// `json.loads`'s number: `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`, a float iff it has a fraction or an exponent.
    fn number(&mut self, bytes: &[u8]) -> Option<Py<'a>> {
        let digit = |p: usize| bytes.get(p).is_some_and(u8::is_ascii_digit);
        let start = self.pos;
        let mut p = start + usize::from(bytes.get(start) == Some(&b'-'));
        match bytes.get(p) {
            Some(b'0') => p += 1,
            Some(b'1'..=b'9') => {
                while digit(p) {
                    p += 1;
                }
            }
            _ => return None,
        }
        let mut float = false;
        if bytes.get(p) == Some(&b'.') && digit(p + 1) {
            p += 1;
            while digit(p) {
                p += 1;
            }
            float = true;
        }
        if matches!(bytes.get(p), Some(b'e' | b'E')) {
            let mut q = p + 1;
            if matches!(bytes.get(q), Some(b'+' | b'-')) {
                q += 1;
            }
            if digit(q) {
                while digit(q) {
                    q += 1;
                }
                p = q;
                float = true;
            }
        }
        self.pos = p;
        let token = &self.text[start..p];
        if float {
            token.parse().ok().map(Py::Float)
        } else {
            Some(Py::Int(token))
        }
    }
}

/// Python `repr(float)` (shortest round-trip digits; scientific outside `1e-4 <= |x| < 1e16`).
fn py_float(f: f64, out: &mut String) {
    if f.is_nan() {
        out.push_str("NaN");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
        return;
    }
    let sci = format!("{f:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let n = digits.len() as i32;
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let int_len = exponent + 1;
            if n <= int_len {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', (int_len - n) as usize));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len as usize]);
                out.push('.');
                out.push_str(&digits[int_len as usize..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
}

/// A Python `json` string: `ensure_ascii` escapes everything outside `' '..='~'`.
fn py_string(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ensure_ascii && c as u32 > 0x7e) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn py_write(value: &Py, style: Style, out: &mut String) {
    let (item_sep, key_sep) = if style.spaced {
        (", ", ": ")
    } else {
        (",", ":")
    };
    match value {
        Py::Literal(l) => out.push_str(l),
        Py::Int(i) => out.push_str(if *i == "-0" { "0" } else { i }),
        Py::Float(f) => py_float(*f, out),
        Py::Str(s) => py_string(s, style.ensure_ascii, out),
        Py::List(items) => {
            out.push('[');
            for (k, item) in items.iter().enumerate() {
                if k > 0 {
                    out.push_str(item_sep);
                }
                py_write(item, style, out);
            }
            out.push(']');
        }
        Py::Dict(entries) => {
            out.push('{');
            for (k, (key, item)) in entries.iter().enumerate() {
                if k > 0 {
                    out.push_str(item_sep);
                }
                py_string(key, style.ensure_ascii, out);
                out.push_str(key_sep);
                py_write(item, style, out);
            }
            out.push('}');
        }
    }
}

/// `json.loads(raw)` of exactly one JSON value (surrounding whitespace allowed); `constants` admits `NaN`,
/// `Infinity` and `-Infinity` as Python does by default. `None` for anything else.
pub fn loads(raw: &str, constants: bool) -> Option<Py<'_>> {
    let mut parser = PyParser {
        text: raw,
        pos: 0,
        constants,
    };
    let value = parser.value()?;
    parser.skip_ws();
    (parser.pos == raw.len()).then_some(value)
}

/// Raw JSON as Python writes it after `json.loads`: `ascii` = `json.dumps(v)` (`", "`/`": "`, `ensure_ascii`),
/// else Starlette's `JSONResponse` (`","`/`":"`, UTF-8). `None` for text that is not one JSON value.
pub fn python_json(raw: &str, ascii: bool) -> Option<String> {
    let value = loads(raw, false)?;
    Some(value.dumps(if ascii { DUMPS } else { STARLETTE }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_floats_match_repr() {
        for (raw, want) in [
            ("1.0", "1.0"),
            ("0.1", "0.1"),
            ("1e-5", "1e-05"),
            ("0.0001", "0.0001"),
            ("1e16", "1e+16"),
            ("1e15", "1000000000000000.0"),
            ("123.456", "123.456"),
            ("-0.0", "-0.0"),
            ("1.5e300", "1.5e+300"),
            ("1e999", "Infinity"),
            ("2.5E-7", "2.5e-07"),
        ] {
            assert_eq!(python_json(raw, true).unwrap(), want, "{raw}");
        }
    }

    #[test]
    fn python_dict_keeps_first_position_last_value() {
        assert_eq!(
            python_json(r#"{"a":1,"b":2,"a":3}"#, true).unwrap(),
            r#"{"a": 3, "b": 2}"#
        );
        assert_eq!(
            python_json(r#"{"é":"ü\u007f"}"#, false).unwrap(),
            "{\"é\":\"ü\u{7f}\"}"
        );
    }
}
