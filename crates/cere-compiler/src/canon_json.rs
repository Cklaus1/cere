//! Canonical JSON (m21 design note Q17) and a duplicate-key-rejecting parser.
//!
//! The encoding equals Python's `json.dumps(v, sort_keys=True, indent=2, ensure_ascii=True,
//! separators=(",", ": ")) + "\n"`: UTF-8, ASCII-only, keys sorted, integers only. Inventory and manifest bytes are
//! produced with it, and verifiers re-encode a parsed manifest to require that the file is canonical.

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt::{self, Write};

/// Why bytes are not acceptable JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonError {
    NotUtf8,
    DuplicateKey(String),
    Syntax(String),
    /// A float (or anything else outside the canonical value set) where only integers are allowed.
    NotCanonicalValue,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonError::NotUtf8 => f.write_str("JSON is not UTF-8"),
            JsonError::DuplicateKey(k) => write!(f, "duplicate JSON key {k:?}"),
            JsonError::Syntax(e) => write!(f, "invalid JSON: {e}"),
            JsonError::NotCanonicalValue => {
                f.write_str("JSON float is not allowed in canonical JSON")
            }
        }
    }
}

impl std::error::Error for JsonError {}

/// Canonical text of `v`, trailing newline included. Floats are rejected.
pub fn to_string(v: &Value) -> Result<String, JsonError> {
    let mut out = String::new();
    write_value(&mut out, v, 0)?;
    out.push('\n');
    Ok(out)
}

/// Canonical bytes of `v`.
pub fn to_vec(v: &Value) -> Result<Vec<u8>, JsonError> {
    to_string(v).map(String::into_bytes)
}

/// True if `bytes` parse strictly and re-encode to exactly `bytes`.
pub fn is_canonical(bytes: &[u8]) -> Result<bool, JsonError> {
    let v = parse_strict(bytes)?;
    Ok(to_vec(&v)? == bytes)
}

fn indent(out: &mut String, level: usize) {
    out.push('\n');
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_value(out: &mut String, v: &Value, level: usize) -> Result<(), JsonError> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, n)?,
        Value::String(s) => write_str(out, s),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                indent(out, level + 1);
                write_value(out, item, level + 1)?;
            }
            indent(out, level);
            out.push(']');
        }
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            out.push('{');
            // serde_json's map is a BTreeMap<String, _>: bytewise UTF-8 order equals Python's code-point order.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                indent(out, level + 1);
                write_str(out, key);
                out.push_str(": ");
                write_value(out, &map[key], level + 1)?;
            }
            indent(out, level);
            out.push('}');
        }
    }
    Ok(())
}

fn write_number(out: &mut String, n: &Number) -> Result<(), JsonError> {
    if let Some(u) = n.as_u64() {
        write!(out, "{u}").expect("write to String");
    } else if let Some(i) = n.as_i64() {
        write!(out, "{i}").expect("write to String");
    } else {
        return Err(JsonError::NotCanonicalValue);
    }
    Ok(())
}

fn write_str(out: &mut String, s: &str) {
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
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(out, "\\u{unit:04x}").expect("write to String");
                }
            }
        }
    }
    out.push('"');
}

const DUPLICATE_MARK: &str = "cerec-duplicate-key:";

/// Parses JSON, rejecting non-UTF-8 input and duplicate object keys at any depth.
pub fn parse_strict(bytes: &[u8]) -> Result<Value, JsonError> {
    let text = std::str::from_utf8(bytes).map_err(|_| JsonError::NotUtf8)?;
    let mut de = serde_json::Deserializer::from_str(text);
    let value = StrictValue::deserialize(&mut de)
        .and_then(|v| de.end().map(|()| v.0))
        .map_err(classify)?;
    Ok(value)
}

/// Like [`parse_strict`], but keeps the top-level object's key order. `Ok(None)` if the top level is not an
/// object.
pub fn parse_strict_ordered(bytes: &[u8]) -> Result<Option<Vec<(String, Value)>>, JsonError> {
    let text = std::str::from_utf8(bytes).map_err(|_| JsonError::NotUtf8)?;
    let mut de = serde_json::Deserializer::from_str(text);
    let value = OrderedTop::deserialize(&mut de)
        .and_then(|v| de.end().map(|()| v.0))
        .map_err(classify)?;
    Ok(value)
}

fn classify(e: serde_json::Error) -> JsonError {
    let msg = e.to_string();
    match msg.strip_prefix(DUPLICATE_MARK) {
        Some(rest) => JsonError::DuplicateKey(
            rest.rsplit_once(" at line ")
                .map_or(rest, |(k, _)| k)
                .to_string(),
        ),
        None => JsonError::Syntax(msg),
    }
}

struct OrderedTop(Option<Vec<(String, Value)>>);

impl<'de> Deserialize<'de> for OrderedTop {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(OrderedTopVisitor).map(OrderedTop)
    }
}

struct OrderedTopVisitor;

impl<'de> Visitor<'de> for OrderedTopVisitor {
    type Value = Option<Vec<(String, Value)>>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entries: Vec<(String, Value)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("{DUPLICATE_MARK}{key}")));
            }
            let StrictValue(v) = map.next_value()?;
            entries.push((key, v));
        }
        Ok(Some(entries))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        StrictVisitor.visit_seq(seq).map(|_| None)
    }

    fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_str<E>(self, _: &str) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(None)
    }
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(StrictVisitor).map(StrictValue)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite number"))
    }

    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(StrictValue(v)) = seq.next_element()? {
            items.push(v);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut obj = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if obj.contains_key(&key) {
                return Err(de::Error::custom(format!("{DUPLICATE_MARK}{key}")));
            }
            let StrictValue(v) = map.next_value()?;
            obj.insert(key, v);
        }
        Ok(Value::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn encodes_like_python_sorted_indent2_ascii() {
        // python3: json.dumps(o, sort_keys=True, indent=2, ensure_ascii=True, separators=(",", ": ")) + "\n"
        let v = json!({"b": 1, "a": [1, -2, {"c": "é\u{7}\u{7f}\"\\\n\t/𝄞"}], "e": {}, "f": [], "g": null,
                       "h": true, "Z": 18446744073709551615u64});
        let want = "{\n  \"Z\": 18446744073709551615,\n  \"a\": [\n    1,\n    -2,\n    {\n      \"c\": \
                    \"\\u00e9\\u0007\\u007f\\\"\\\\\\n\\t/\\ud834\\udd1e\"\n    }\n  ],\n  \"b\": 1,\n  \"e\": {},\n  \
                    \"f\": [],\n  \"g\": null,\n  \"h\": true\n}\n";
        assert_eq!(to_string(&v).unwrap(), want);
    }

    #[test]
    fn rejects_floats() {
        assert!(to_string(&json!({"a": 1.5})).is_err());
    }

    #[test]
    fn strict_parse_rejects_duplicate_keys_at_any_depth() {
        assert!(parse_strict(br#"{"a": 1, "b": {"c": 2}}"#).is_ok());
        let e = parse_strict(br#"{"a": 1, "a": 2}"#).unwrap_err();
        assert_eq!(e, JsonError::DuplicateKey("a".into()));
        let e = parse_strict(br#"{"x": [{"k": 1, "k": 1}]}"#).unwrap_err();
        assert_eq!(e, JsonError::DuplicateKey("k".into()));
    }

    #[test]
    fn strict_parse_separates_utf8_from_syntax_errors() {
        assert_eq!(
            parse_strict(b"{\"a\": \"\xff\"}").unwrap_err(),
            JsonError::NotUtf8
        );
        assert!(matches!(
            parse_strict(b"{\"a\": }"),
            Err(JsonError::Syntax(_))
        ));
        assert!(matches!(parse_strict(b"{} x"), Err(JsonError::Syntax(_))));
    }

    #[test]
    fn canonical_round_trip_check() {
        let v = json!({"b": [1], "a": "x"});
        let text = to_string(&v).unwrap();
        assert!(is_canonical(text.as_bytes()).unwrap());
        assert!(!is_canonical(b"{\"b\": [1], \"a\": \"x\"}\n").unwrap());
        assert!(!is_canonical(text.trim_end().as_bytes()).unwrap());
    }
}
