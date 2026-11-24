//! M2.8 output constraints (m28 design note D3, D5): the shapes of `response_format`, `regex` and `ebnf`, and
//! the D5 schema walker that admits only what the pinned xgrammar enforces.
//!
//! Shapes are checked here; the request keeps each field as the client's bytes (`validate.rs`, D6). The walker
//! runs after the profile gate (D10), on `response_format.json_schema.schema` and on the `parameters` of strict
//! tools of a tool-constrained request.

use serde_json::{Map, Value};

use crate::api_types::ApiError;

/// D3: `regex`/`ebnf` are non-empty strings of at most this many bytes.
pub const MAX_GRAMMAR_BYTES: usize = 16 * 1024;
/// D3: a `json_schema` schema serializes to at most this many bytes...
pub const MAX_SCHEMA_BYTES: usize = 64 * 1024;
/// ...and nests at most this deep (the schema object itself is depth 1; the M2.4 D2 precedent).
pub const MAX_SCHEMA_NESTING: usize = 32;
/// D3: the `name` of a `json_schema` response format matches `^[A-Za-z0-9_-]{1,64}$`.
const MAX_SCHEMA_NAME_CHARS: usize = 64;

/// D5: the `format` values xgrammar 0.2.1 (the pinned SGLang image's version) enforces, i.e. the C4 enforcement
/// matrix's passing set (`tests/fixtures/native/grammar_keywords.json`; `uri-reference` failed); any other value is
/// refused.
pub const ENFORCED_FORMATS: &[&str] = &[
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
];

/// The constraint fields in D3 precedence: when a request carries more than one, the later field is refused.
pub const CONSTRAINT_FIELDS: [&str; 3] = ["response_format", "regex", "ebnf"];

/// Checks a non-null `response_format` (D3). `Ok(false)` for `{type:"text"}` (the same as absent), `Ok(true)` for
/// a constraint. The schema is not walked here: [`walk_schema`] runs after the profile gate.
pub fn response_format(value: &Value) -> Result<bool, ApiError> {
    const AT: &str = "response_format";
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::invalid_value(AT, "response_format must be an object"))?;
    let kind = match object.get("type") {
        None | Some(Value::Null) => {
            return Err(ApiError::missing_parameter("response_format.type"))
        }
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => {
            return Err(ApiError::invalid_value(
                "response_format.type",
                "type must be a string",
            ))
        }
    };
    match kind {
        "text" | "json_object" => {
            known_keys(object, AT, &["type"])?;
            Ok(kind == "json_object")
        }
        "json_schema" => {
            // N3: SGLang's shortcut without the `json_schema` wrapper is refused.
            let wrapper = match object.get("json_schema") {
                None | Some(Value::Null) if object.contains_key("schema") => {
                    return Err(ApiError::invalid_value(
                        "response_format.json_schema",
                        "the schema belongs in response_format.json_schema.schema",
                    ))
                }
                None | Some(Value::Null) => {
                    return Err(ApiError::missing_parameter("response_format.json_schema"))
                }
                Some(Value::Object(wrapper)) => wrapper,
                Some(_) => {
                    return Err(ApiError::invalid_value(
                        "response_format.json_schema",
                        "json_schema must be an object",
                    ))
                }
            };
            known_keys(object, AT, &["json_schema", "type"])?;
            json_schema_wrapper(wrapper)?;
            Ok(true)
        }
        "structural_tag" => Err(ApiError::unsupported_parameter(
            "response_format.type",
            "structural_tag response formats are not supported",
        )),
        _ => Err(ApiError::invalid_value(
            "response_format.type",
            "type must be \"text\", \"json_object\" or \"json_schema\"",
        )),
    }
}

fn json_schema_wrapper(wrapper: &Map<String, Value>) -> Result<(), ApiError> {
    const AT: &str = "response_format.json_schema";
    known_keys(wrapper, AT, &["description", "name", "schema", "strict"])?;
    match wrapper.get("name") {
        None | Some(Value::Null) => {
            return Err(ApiError::missing_parameter(
                "response_format.json_schema.name",
            ))
        }
        Some(Value::String(name))
            if (1..=MAX_SCHEMA_NAME_CHARS).contains(&name.len())
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') => {}
        Some(_) => {
            return Err(ApiError::invalid_value(
                "response_format.json_schema.name",
                format!("name must match ^[A-Za-z0-9_-]{{1,{MAX_SCHEMA_NAME_CHARS}}}$"),
            ))
        }
    }
    match wrapper.get("description") {
        None | Some(Value::Null | Value::String(_)) => {}
        Some(_) => {
            return Err(ApiError::invalid_value(
                "response_format.json_schema.description",
                "description must be a string",
            ))
        }
    }
    match wrapper.get("strict") {
        None | Some(Value::Null | Value::Bool(_)) => {}
        Some(_) => {
            return Err(ApiError::invalid_value(
                "response_format.json_schema.strict",
                "strict must be a boolean",
            ))
        }
    }
    match wrapper.get("schema") {
        None | Some(Value::Null) => Err(ApiError::missing_parameter(SCHEMA_PARAM)),
        Some(schema) => bounded_schema(schema, SCHEMA_PARAM),
    }
}

/// The refusal path of a `response_format` schema.
pub const SCHEMA_PARAM: &str = "response_format.json_schema.schema";

/// D3 bounds of a schema: an object of at most [`MAX_SCHEMA_BYTES`] serialized, nested at most
/// [`MAX_SCHEMA_NESTING`] deep.
pub fn bounded_schema(schema: &Value, param: &str) -> Result<(), ApiError> {
    let fits = schema.is_object()
        && depth(schema) <= MAX_SCHEMA_NESTING
        && serde_json::to_vec(schema).is_ok_and(|bytes| bytes.len() <= MAX_SCHEMA_BYTES);
    if fits {
        Ok(())
    } else {
        Err(ApiError::invalid_value(
            param,
            format!(
                "the schema must be a JSON object of at most {MAX_SCHEMA_BYTES} bytes nested at most \
                 {MAX_SCHEMA_NESTING} deep"
            ),
        ))
    }
}

/// Checks a non-null `regex` or `ebnf` (D3): a non-empty string of at most [`MAX_GRAMMAR_BYTES`].
pub fn grammar_text(value: &Value, field: &str) -> Result<(), ApiError> {
    match value {
        Value::String(text) if (1..=MAX_GRAMMAR_BYTES).contains(&text.len()) => Ok(()),
        _ => Err(ApiError::invalid_value(
            field,
            format!("{field} must be a non-empty string of at most {MAX_GRAMMAR_BYTES} bytes"),
        )),
    }
}

/// D5: `schema` uses only `keywords` (the profile's `json_schema_keywords`), never `oneOf`, `format` only with an
/// [`ENFORCED_FORMATS`] value, `pattern` only with its `^`/`$` anchors at the ends and numeric bounds only in a
/// [`bounds_enforced`] form (`number_ranges`: whether number ranges are compiled exactly, i.e. the
/// `response_format` schema of a native deployment); the walk descends into `properties`, `items`, `prefixItems`,
/// `anyOf`, `allOf`, `$defs` and `additionalProperties`. Every visited schema object also keeps to the forms
/// xgrammar 0.2.1's converter does not silently drop (B5): see [`structure`]. A refusal names `param`.
pub fn walk_schema(
    schema: &Value,
    keywords: &[String],
    param: &str,
    number_ranges: bool,
) -> Result<(), ApiError> {
    let defs = schema.get("$defs").and_then(Value::as_object);
    walk(schema, keywords, defs, true, number_ranges)
        .map_err(|reason| ApiError::invalid_value(param, reason))
}

fn walk(
    schema: &Value,
    keywords: &[String],
    defs: Option<&Map<String, Value>>,
    root: bool,
    number_ranges: bool,
) -> Result<(), String> {
    let object = match schema {
        Value::Object(object) => object,
        Value::Bool(_) => return Ok(()),
        _ => return Err("a schema must be an object or a boolean".to_owned()),
    };
    if let Some(keyword) = object
        .keys()
        .find(|k| *k == "oneOf" || !keywords.iter().any(|w| w == *k))
    {
        return Err(format!(
            "schema keyword '{keyword}' is not enforced by this deployment"
        ));
    }
    bounds_enforced(object, number_ranges)?;
    structure(object, defs, root)?;
    for (keyword, value) in object {
        match keyword.as_str() {
            "properties" | "$defs" => match value {
                Value::Object(schemas) => {
                    for schema in schemas.values() {
                        walk(schema, keywords, defs, false, number_ranges)?;
                    }
                }
                _ => return Err(format!("'{keyword}' must be an object of schemas")),
            },
            "items" | "additionalProperties" => walk(value, keywords, defs, false, number_ranges)?,
            "prefixItems" | "anyOf" | "allOf" => match value {
                Value::Array(schemas) => {
                    for schema in schemas {
                        walk(schema, keywords, defs, false, number_ranges)?;
                    }
                }
                _ => return Err(format!("'{keyword}' must be an array of schemas")),
            },
            "format" => match value {
                Value::String(format) if ENFORCED_FORMATS.contains(&format.as_str()) => {}
                _ => return Err(format!("format {value} is not enforced")),
            },
            "pattern" => match value {
                Value::String(pattern) if anchors_at_ends(pattern) => {}
                _ => {
                    return Err(
                        "pattern must be a string with ^ and $ anchors only at its ends".to_owned(),
                    )
                }
            },
            _ => {}
        }
    }
    Ok(())
}

/// Annotation keywords: never compiled, so allowed beside any other keyword.
const ANNOTATIONS: [&str; 3] = ["title", "description", "default"];

/// Keywords xgrammar 0.2.1's `SchemaParser::Parse` dispatches on before `type`, dropping every sibling.
const DISPATCH: [&str; 4] = ["$ref", "const", "enum", "anyOf"];

/// Keywords xgrammar applies only under one exact `type` (the bounds are [`bounds_enforced`]'s).
const TYPED: [(&str, &str); 11] = [
    ("minLength", "string"),
    ("maxLength", "string"),
    ("pattern", "string"),
    ("format", "string"),
    ("items", "array"),
    ("prefixItems", "array"),
    ("minItems", "array"),
    ("maxItems", "array"),
    ("properties", "object"),
    ("required", "object"),
    ("additionalProperties", "object"),
];

/// B5: the schema-object forms xgrammar 0.2.1 compiles without dropping a keyword.
/// * R1: an object with `$ref`, `const`, `enum` or `anyOf` carries nothing else but annotations (and, at the
///   root, `$defs`). [REVISED: owner 2026-10-17 O113] `enum` may also carry a `type` (a name or an array of
///   names) every enum value is an instance of: xgrammar drops that `type`, which then constrains nothing the
///   enum does not, so the compiled grammar is exactly the enum's (`grammar_keywords.json` `B:type+enum:*`,
///   `same_grammar_without_type`). A `type` some value contradicts is refused.
/// * R2: string/array/object keywords need `type` exactly `"string"`/`"array"`/`"object"`; a `type` array
///   carries no type-specific keyword.
/// * R3: a string uses at most one of `format`, `pattern`, `minLength`/`maxLength`.
/// * R4: `$ref` is `"#"` or `"#/$defs/<name>"` with `<name>` (no `/` or `~`) a key of the root `$defs`.
/// * R5: every `required` name is a key of the same object's `properties`.
fn structure(
    object: &Map<String, Value>,
    defs: Option<&Map<String, Value>>,
    root: bool,
) -> Result<(), String> {
    if let Some(dispatch) = DISPATCH.iter().find(|k| object.contains_key(**k)) {
        let typed_enum = *dispatch == "enum" && object.contains_key("type");
        let other = object.keys().find(|k| {
            let allowed = *k == dispatch
                || (typed_enum && *k == "type")
                || ANNOTATIONS.contains(&k.as_str())
                || (root && *k == "$defs");
            !allowed
        });
        if let Some(other) = other {
            return Err(format!(
                "'{other}' beside '{dispatch}' is not enforced: only annotations may accompany it"
            ));
        }
        if typed_enum && !enum_of_type(&object["enum"], &object["type"]) {
            return Err(
                "'type' beside 'enum' is enforced only when it names JSON types and every enum value is one of them"
                    .to_owned(),
            );
        }
    }
    let kind = object.get("type");
    for (keyword, needs) in TYPED {
        if object.contains_key(keyword) && kind.and_then(Value::as_str) != Some(needs) {
            return Err(format!(
                "'{keyword}' is enforced only with type \"{needs}\""
            ));
        }
    }
    let string_forms = usize::from(object.contains_key("format"))
        + usize::from(object.contains_key("pattern"))
        + usize::from(object.contains_key("minLength") || object.contains_key("maxLength"));
    if string_forms > 1 {
        return Err(
            "a string schema is enforced with at most one of format, pattern or minLength/maxLength"
                .to_owned(),
        );
    }
    if let Some(target) = object.get("$ref") {
        let local = match target.as_str() {
            Some("#") => true,
            // xgrammar drops empty pointer segments: "#/$defs/" would name the whole $defs map.
            Some(target) => target.strip_prefix("#/$defs/").is_some_and(|name| {
                !name.is_empty()
                    && !name.contains(['/', '~'])
                    && defs.is_some_and(|defs| defs.contains_key(name))
            }),
            None => false,
        };
        if !local {
            return Err(
                "$ref must be \"#\" or \"#/$defs/<name>\" naming a key of the root $defs"
                    .to_owned(),
            );
        }
    }
    if let Some(required) = object.get("required") {
        let properties = object.get("properties").and_then(Value::as_object);
        let declared = required.as_array().is_some_and(|names| {
            names.iter().all(|name| {
                name.as_str()
                    .is_some_and(|name| properties.is_some_and(|p| p.contains_key(name)))
            })
        });
        if !declared {
            return Err(
                "required must list names declared in the same object's properties".to_owned(),
            );
        }
    }
    Ok(())
}

/// Whether `kind` is a JSON type name or a non-empty array of them and every value of `values` (an array) is an
/// instance of one of them (an integer: a number without a fractional part, JSON Schema's rule).
fn enum_of_type(values: &Value, kind: &Value) -> bool {
    const TYPES: [&str; 7] = [
        "string", "integer", "number", "boolean", "null", "object", "array",
    ];
    let names: Vec<&str> = match kind {
        Value::String(name) => vec![name.as_str()],
        Value::Array(names) if !names.is_empty() => {
            match names.iter().map(Value::as_str).collect::<Option<Vec<_>>>() {
                Some(names) => names,
                None => return false,
            }
        }
        _ => return false,
    };
    let instance = |value: &Value, name: &str| match (name, value) {
        ("string", Value::String(_))
        | ("number", Value::Number(_))
        | ("boolean", Value::Bool(_))
        | ("null", Value::Null)
        | ("object", Value::Object(_))
        | ("array", Value::Array(_)) => true,
        ("integer", Value::Number(n)) => {
            n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0)
        }
        _ => false,
    };
    names.iter().all(|name| TYPES.contains(name))
        && values
            .as_array()
            .is_some_and(|values| values.iter().all(|v| names.iter().any(|n| instance(v, n))))
}

/// The bound forms cered enforces: no bound at all; a schema whose `type` is exactly `"integer"` with exactly one
/// upper bound (`maximum` or `exclusiveMaximum`) and at most one lower bound (`minimum` or `exclusiveMinimum`), the
/// C4 forms xgrammar 0.2.1 enforces (`grammar_keywords.json` `bound_forms`); or, when `number_ranges` (the
/// `response_format` schema of a native deployment, whose grammar sidecar compiles it), a [`number_range`]
/// [REVISED: owner 2026-10-18 number bounds]. A lone integer lower bound leaks, and xgrammar drops every number
/// bound, so a number bound anywhere else (an SGLang deployment, a tool's parameters) is refused.
fn bounds_enforced(object: &Map<String, Value>, number_ranges: bool) -> Result<(), String> {
    let has = |key: &str| object.contains_key(key);
    let upper = usize::from(has("maximum")) + usize::from(has("exclusiveMaximum"));
    let lower = usize::from(has("minimum")) + usize::from(has("exclusiveMinimum"));
    if upper + lower == 0 {
        return Ok(());
    }
    match object.get("type").and_then(Value::as_str) {
        Some("integer") if upper == 1 && lower <= 1 => Ok(()),
        Some("number") if number_ranges => number_range(object),
        Some("number") => Err(
            "number bounds are enforced only in the response_format schema of a native deployment".to_owned(),
        ),
        _ => Err(
            "numeric bounds are enforced only on type \"integer\" with one upper bound and at most one lower \
             bound, or on type \"number\" as an inclusive minimum and maximum"
                .to_owned(),
        ),
    }
}

/// The keywords a [`number_range`] schema may carry.
const NUMBER_RANGE_KEYS: [&str; 6] = [
    "type",
    "minimum",
    "maximum",
    "title",
    "description",
    "default",
];
/// Digits a [`number_range`] bound may have before, and after, its decimal point.
const NUMBER_BOUND_DIGITS: usize = 20;

/// B7 [REVISED: owner 2026-10-18 number bounds]: a `type: "number"` schema the grammar sidecar compiles to an exact
/// range rule (`tokenizer_sidecar.py` `number_range`): an inclusive `minimum` and `maximum`, nothing else but
/// annotations, `minimum <= maximum`, each bound at most [`NUMBER_BOUND_DIGITS`] digits before and after its decimal
/// point. Exclusive or lone number bounds are refused.
fn number_range(object: &Map<String, Value>) -> Result<(), String> {
    let pair = object.contains_key("minimum")
        && object.contains_key("maximum")
        && object
            .keys()
            .all(|k| NUMBER_RANGE_KEYS.contains(&k.as_str()));
    if !pair {
        return Err(
            "number bounds are enforced only as an inclusive minimum and maximum, beside nothing but type, \
             title, description and default"
                .to_owned(),
        );
    }
    let fits =
        |d: &Decimal| d.int.len() <= NUMBER_BOUND_DIGITS && d.frac.len() <= NUMBER_BOUND_DIGITS;
    match (decimal(&object["minimum"]), decimal(&object["maximum"])) {
        (Some(lo), Some(hi)) if fits(&lo) && fits(&hi) && lo.order(&hi).is_le() => Ok(()),
        _ => Err(format!(
            "number bounds need numbers with minimum <= maximum, each with at most {NUMBER_BOUND_DIGITS} digits \
             before and after the decimal point"
        )),
    }
}

/// A JSON number as an exact decimal: the sign (zero is not negative), the integer digits without leading zeros
/// ("0" for none) and the fraction digits without trailing zeros. A float is its shortest round-trip decimal, as
/// the sidecar's `Decimal(repr(value))`.
struct Decimal {
    negative: bool,
    int: String,
    frac: String,
}

impl Decimal {
    fn order(&self, other: &Self) -> std::cmp::Ordering {
        // Fractions without trailing zeros order as strings (".25" < ".5", ".2" < ".25").
        let magnitude = |a: &Self, b: &Self| {
            (a.int.len(), &a.int, &a.frac).cmp(&(b.int.len(), &b.int, &b.frac))
        };
        match (self.negative, other.negative) {
            (false, false) => magnitude(self, other),
            (true, true) => magnitude(other, self),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
        }
    }
}

fn decimal(value: &Value) -> Option<Decimal> {
    let number = value.as_number()?;
    let text = match (number.as_u64(), number.as_i64(), number.as_f64()) {
        (Some(u), _, _) => u.to_string(),
        (_, Some(i), _) => i.to_string(),
        (_, _, Some(f)) if f.is_finite() => format!("{f}"),
        _ => return None,
    };
    let (negative, magnitude) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.as_str()),
    };
    let (int, frac) = magnitude.split_once('.').unwrap_or((magnitude, ""));
    let int = match int.trim_start_matches('0') {
        "" => "0",
        digits => digits,
    };
    let frac = frac.trim_end_matches('0');
    Some(Decimal {
        negative: negative && !(int == "0" && frac.is_empty()),
        int: int.to_owned(),
        frac: frac.to_owned(),
    })
}

/// Whether every unescaped `^`/`$` outside a character class sits at the pattern's start/end.
fn anchors_at_ends(pattern: &str) -> bool {
    let bytes = pattern.as_bytes();
    let last = bytes.len().saturating_sub(1);
    let mut class = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                i += 2;
                continue;
            }
            b'[' if !class => {
                class = true;
                if bytes.get(i + 1) == Some(&b'^') {
                    i += 2;
                    continue;
                }
            }
            b']' if class => class = false,
            b'^' if !class && i != 0 => return false,
            b'$' if !class && i != last => return false,
            _ => {}
        }
        i += 1;
    }
    true
}

/// Container nesting depth of a JSON value (a scalar is 0, `{}` is 1).
pub fn depth(value: &Value) -> usize {
    match value {
        Value::Object(map) => 1 + map.values().map(depth).max().unwrap_or(0),
        Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
        _ => 0,
    }
}

/// Refuses the first key of `object` outside `keys` as `unsupported_parameter` `{at}.{key}`.
pub fn known_keys(object: &Map<String, Value>, at: &str, keys: &[&str]) -> Result<(), ApiError> {
    match object.keys().find(|key| !keys.contains(&key.as_str())) {
        Some(key) => Err(ApiError::unsupported_parameter(
            format!("{at}.{key}"),
            format!("field '{key}' is not supported"),
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::anchors_at_ends;

    #[test]
    fn anchors_only_at_the_ends() {
        for ok in ["", "^a$", "a", "[\\^$]x\\$", "[^a]+", "^[a-z]+$", "a\\^b"] {
            assert!(anchors_at_ends(ok), "{ok}");
        }
        for bad in ["a^b", "a$b", "(^a)", "x^", "$a", "(a$)"] {
            assert!(!anchors_at_ends(bad), "{bad}");
        }
    }
}
