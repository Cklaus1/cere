//! C4/C5 (spec m24 D9, D10): the native qwen3_coder tool parser against the oracle fixtures
//! (`tests/fixtures/native/{tool_parser_cases,tool_stream_chain_cases,tool_convert_cases}.json`, recorded in the
//! pinned image from SGLang's own `Qwen3CoderDetector`, `FunctionCallParser` and `ReasoningParser(qwen3)` by
//! `scripts/native/gen_api_fixtures.py`).
//!
//! The parser never converts a value: every conversion it asks for is answered here from
//! `tool_convert_cases.json` (SGLang's `_convert_param_value` + `json.dumps` on the prepared tools, the sidecar's
//! `tool_param` / `tool_args`), so a request the oracle did not record is a failure, as is a different prepared
//! tool for the function (`tool_for`).
mod native_support;

use std::collections::{BTreeMap, HashMap};

use cere_runtime::native::reasoning::Reasoning;
use cere_runtime::native::tool_parser::{self, Chain, Delta, Format, Item, RawCall, Stream};
use native_support::root;
use serde_json::{json, Value};

fn fixture(name: &str) -> Value {
    let path = root().join("tests/fixtures/native").join(name);
    serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not a string: {v}"))
}

/// The header every M2.4 fixture shares, from the same image and tokenizer.
fn check_header(v: &Value, stem: &str) {
    assert_eq!(v["schema"], format!("cere-{}-v1", stem.replace('_', "-")));
    assert_eq!(v["generator"], "scripts/native/gen_api_fixtures.py");
    for key in ["image_digest", "sglang_version", "source_tokenizer_sha256"] {
        assert!(
            v[key].as_str().is_some_and(|s| !s.is_empty()),
            "{stem}: {key}"
        );
    }
}

/// A conversion outcome: the fragment / arguments, or the reference's exception type.
type Converted = Result<String, String>;

fn outcome(case: &Value, key: &str) -> Converted {
    match case.get("error") {
        Some(error) => Err(error.as_str().unwrap().to_owned()),
        None => Ok(text(case, key).to_owned()),
    }
}

/// Key of a recorded `tool_args` conversion: (tool set, function, [(name, raw)]).
type ArgsKey = (String, Option<String>, Vec<(String, String)>);

/// The oracle's conversions, replayed.
struct Converter {
    prepared: HashMap<String, Vec<Value>>,
    param: HashMap<(String, Option<String>, String, String), Converted>,
    args: HashMap<ArgsKey, Converted>,
}

impl Converter {
    fn load() -> Converter {
        let v = fixture("tool_convert_cases.json");
        check_header(&v, "tool_convert_cases");
        let prepared: HashMap<String, Vec<Value>> = v["tool_sets"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(set, s)| (set.clone(), s["tools_prepared"].as_array().unwrap().clone()))
            .collect();
        let mut param = HashMap::new();
        let mut args = HashMap::new();
        let cases = v["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let name = text(case, "name");
            let set = text(case, "tool_set").to_owned();
            let func = case["func"].as_str().map(str::to_owned);
            // The prepared tool the reference reads for the function (`_get_arguments_config`).
            let tools = &prepared[&set];
            assert_eq!(
                tool_parser::tool_for(tools, func.as_deref()),
                case["tool_json"].as_object().map(|_| &case["tool_json"]),
                "case {name}: tool_for"
            );
            match text(case, "op") {
                "tool_param" => {
                    let key = (
                        set,
                        func,
                        text(case, "param").to_owned(),
                        text(case, "raw").to_owned(),
                    );
                    param.insert(key, outcome(case, "fragment"));
                }
                "tool_args" => {
                    let params = case["params"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| {
                            (
                                p[0].as_str().unwrap().to_owned(),
                                p[1].as_str().unwrap().to_owned(),
                            )
                        })
                        .collect();
                    let key = (set, func, params);
                    args.insert(key, outcome(case, "arguments"));
                }
                other => panic!("case {name}: op {other}"),
            }
        }
        Converter {
            prepared,
            param,
            args,
        }
    }

    fn tool_param(&self, set: &str, func: Option<&str>, name: &str, raw: &str) -> Converted {
        let key = (
            set.to_owned(),
            func.map(str::to_owned),
            name.to_owned(),
            raw.to_owned(),
        );
        self.param
            .get(&key)
            .unwrap_or_else(|| panic!("no recorded tool_param for {key:?}"))
            .clone()
    }
    fn tool_args(&self, set: &str, call: &RawCall) -> Converted {
        let key = (set.to_owned(), Some(call.name.clone()), call.params.clone());
        self.args
            .get(&key)
            .unwrap_or_else(|| panic!("no recorded tool_args for {key:?}"))
            .clone()
    }
}

/// `[tool_index, name | null, parameters]`, the fixtures' form of a `ToolCallItem`.
fn delta_json(d: &Delta) -> Value {
    json!([d.index, d.name, d.arguments])
}

/// Converts one push's items; the first failure is the push's exception.
fn resolve(conv: &Converter, set: &str, items: Vec<Item>) -> Result<Vec<Value>, String> {
    items
        .into_iter()
        .map(|item| {
            item.resolve(|func, name, raw| conv.tool_param(set, func, name, raw))
                .map(|d| delta_json(&d))
        })
        .collect()
}

/// The chunks of `text` at the code-point offsets `cuts` (`[0, …, len]`).
fn chunks<'t>(text: &'t str, cuts: &Value) -> Vec<&'t str> {
    let mut bytes: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    bytes.push(text.len());
    let cuts: Vec<usize> = cuts
        .as_array()
        .unwrap()
        .iter()
        .map(|c| bytes[c.as_u64().unwrap() as usize])
        .collect();
    assert_eq!(
        (cuts.first(), cuts.last()),
        (Some(&0), Some(&text.len())),
        "cuts span the text"
    );
    cuts.windows(2).map(|w| &text[w[0]..w[1]]).collect()
}

/// C4: the unary path (`serving_chat.py:2006-2019`): `has_tool_call` gates `parse_non_stream`; normal text and
/// calls equal the reference's, a converter failure included (the whole text, no calls).
#[test]
fn tool_parser_cases_unary() {
    let conv = Converter::load();
    let v = fixture("tool_parser_cases.json");
    check_header(&v, "tool_parser_cases");
    assert_eq!(v["parser"], "qwen3_coder");
    let cases = v["cases"].as_array().unwrap();
    assert!(
        cases.len() as u64 >= v["random_cases"].as_u64().unwrap(),
        "{} cases",
        cases.len()
    );
    let mut with_calls = 0;
    for case in cases {
        let name = text(case, "name");
        let set = text(case, "tool_set");
        assert!(
            conv.prepared.contains_key(set),
            "case {name}: tool set {set}"
        );
        let t = text(case, "text");
        let unary = &case["unary"];
        let has = tool_parser::has_tool_call(t);
        assert_eq!(json!(has), unary["has_tool_call"], "case {name}");
        let (content, calls) = if has {
            // A recorded failure is the reference's conversion exception.
            tool_parser::parse_non_stream(t, |call| Ok::<_, ()>(conv.tool_args(set, call).ok()))
                .unwrap()
        } else {
            (t.to_owned(), Vec::new())
        };
        assert_eq!(content, text(unary, "content"), "case {name} content");
        let calls: Vec<Value> = calls
            .iter()
            .map(|c| json!([c.index, c.name, c.arguments]))
            .collect();
        assert_eq!(
            Value::Array(calls.clone()),
            unary["calls"],
            "case {name} calls"
        );
        with_calls += usize::from(!calls.is_empty());
    }
    println!(
        "C4: {} tool_parser_cases unary pass ({with_calls} with calls)",
        cases.len()
    );
}

/// C5 (parser half): every case at every recorded chunking: per push `(normal_text, calls)` equal
/// `FunctionCallParser.parse_stream_chunk`'s, holds included; `parse_stream_end` emits nothing; a conversion
/// exception ends the stream.
#[test]
fn tool_parser_cases_stream() {
    let conv = Converter::load();
    let v = fixture("tool_parser_cases.json");
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for case in v["cases"].as_array().unwrap() {
        let name = text(case, "name");
        let set = text(case, "tool_set");
        let t = text(case, "text");
        for s in case["streams"].as_array().unwrap() {
            let chunking = text(s, "chunking");
            let at = format!("case {name} chunking {chunking}");
            let pieces = chunks(t, &s["cuts"]);
            let pushes = s["pushes"].as_array().unwrap();
            let mut stream = Stream::default();
            let mut errored = false;
            for (i, (piece, want)) in pieces.iter().zip(pushes).enumerate() {
                let inc = stream.push(piece);
                match resolve(&conv, set, inc.items) {
                    Ok(calls) => {
                        assert_eq!(
                            json!([inc.normal, calls]),
                            *want,
                            "{at} push {i} ({piece:?})"
                        );
                    }
                    Err(error) => {
                        assert_eq!(json!({ "error": error }), *want, "{at} push {i}");
                        assert_eq!(i + 1, pushes.len(), "{at}: an error ends the stream");
                        errored = true;
                        break;
                    }
                }
            }
            if errored {
                assert!(s["finish"].is_null(), "{at}");
            } else {
                assert_eq!(pieces.len(), pushes.len(), "{at}");
                let end = stream.finish();
                let calls = resolve(&conv, set, end.items).unwrap();
                assert_eq!(json!([end.normal, calls]), s["finish"], "{at} finish");
            }
            let count = counts.entry(chunking.to_owned()).or_default();
            count.0 += 1;
            count.1 += pushes.len();
        }
    }
    println!("C5 parser: streams (pushes) per chunking: {counts:?}");
    for chunking in ["whole", "char", "token"] {
        assert!(counts.contains_key(chunking), "no {chunking} streams");
    }
}

/// C5 (chain half): `reasoning::Stream` -> `tool_parser::Stream` per piece as `serving_chat.py:700-748`, the
/// finish flush (ReasoningParser end text, then the tool parser's end) on the last piece of a `stop` finish only;
/// an abort just stops.
#[test]
fn tool_stream_chain_cases() {
    let conv = Converter::load();
    let v = fixture("tool_stream_chain_cases.json");
    check_header(&v, "tool_stream_chain_cases");
    assert_eq!(v["reasoning_parser"], "qwen3");
    assert_eq!(v["force_reasoning"], false);
    assert_eq!(v["stream_reasoning"], true);
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut finishes: BTreeMap<String, usize> = BTreeMap::new();
    for case in v["cases"].as_array().unwrap() {
        let name = text(case, "name");
        let set = text(case, "tool_set");
        let t = text(case, "text");
        for s in case["streams"].as_array().unwrap() {
            let chunking = text(s, "chunking");
            let finish = text(s, "finish");
            let at = format!("case {name} chunking {chunking} finish {finish}");
            assert!(["stop", "abort"].contains(&finish), "{at}");
            let pieces = chunks(t, &s["cuts"]);
            let pushes = s["pushes"].as_array().unwrap();
            let mut chain = Chain::new(Some(Format::Qwen3Coder), Reasoning::Detect);
            for (i, (piece, want)) in pieces.iter().zip(pushes).enumerate() {
                let last = i + 1 == pieces.len();
                let step = chain.push(piece, last && finish == "stop");
                match resolve(&conv, set, step.items) {
                    Ok(calls) => assert_eq!(
                        json!([step.reasoning, step.normal, calls]),
                        *want,
                        "{at} push {i} ({piece:?})"
                    ),
                    Err(error) => {
                        assert_eq!(json!({ "error": error }), *want, "{at} push {i}");
                        assert_eq!(i + 1, pushes.len(), "{at}: an error ends the stream");
                        break;
                    }
                }
            }
            if pushes.last().is_none_or(|p| p.get("error").is_none()) {
                assert_eq!(pieces.len(), pushes.len(), "{at}");
            }
            let count = counts.entry(chunking.to_owned()).or_default();
            count.0 += 1;
            count.1 += pushes.len();
            *finishes.entry(finish.to_owned()).or_default() += 1;
        }
    }
    println!("C5 chain: streams (pushes) per chunking: {counts:?}; per finish: {finishes:?}");
    assert!(finishes.contains_key("stop") && finishes.contains_key("abort"));
}
