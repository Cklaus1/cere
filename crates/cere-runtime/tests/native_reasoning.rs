//! C4 (spec m212 D4): the Qwen3 reasoning split forced into reasoning, as SGLang runs it when thinking resolves on
//! (`ReasoningParser(qwen3, force_reasoning=True)`, `serving_chat.py:1809-1823, 2117-2130`), against
//! `tests/fixtures/native/thinking_cases.json` (D11, recorded in the pinned image) at whole, per-code-point, token
//! and tag-splitting chunking; `separate_reasoning: false` bypasses the parser (all text is content).

mod native_support;

use cere_runtime::native::reasoning::{split, Reasoning, Split, Stream};
use native_support::root;
use serde_json::Value;

fn cases() -> Vec<Value> {
    let path = root().join("tests/fixtures/native/thinking_cases.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(doc["schema"], "cere-thinking-cases-v1");
    doc["cases"].as_array().unwrap().clone()
}

/// `[reasoning_text, normal_text]` as the fixture spells a parser result.
fn pair(v: &Value) -> Split {
    Split {
        reasoning: v[0].as_str().unwrap().to_owned(),
        normal: v[1].as_str().unwrap().to_owned(),
    }
}

fn pieces<'t>(text: &'t str, cuts: &[usize]) -> Vec<&'t str> {
    // Cuts are code-point offsets.
    let bytes: Vec<usize> = text
        .char_indices()
        .map(|(b, _)| b)
        .chain([text.len()])
        .collect();
    cuts.windows(2)
        .map(|w| &text[bytes[w[0]]..bytes[w[1]]])
        .collect()
}

#[test]
fn forced_reasoning_equals_the_oracle() {
    let mut streams = 0;
    let reasoning: Vec<Value> = cases()
        .into_iter()
        .filter(|c| c["kind"] == "reasoning")
        .collect();
    assert!(reasoning.len() >= 10);
    for case in &reasoning {
        let name = case["name"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        assert_eq!(case["force_reasoning"], true);
        assert_eq!(
            split(text, Reasoning::Forced),
            pair(&case["unary"]),
            "case {name} unary"
        );
        for s in case["streams"].as_array().unwrap() {
            let chunking = s["chunking"].as_str().unwrap();
            let cuts: Vec<usize> = s["cuts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_u64().unwrap() as usize)
                .collect();
            let deltas = s["deltas"].as_array().unwrap();
            let pieces = pieces(text, &cuts);
            assert_eq!(pieces.len(), deltas.len(), "case {name} {chunking}");
            let mut stream = Stream::new(Reasoning::Forced);
            for (i, (piece, delta)) in pieces.iter().zip(deltas).enumerate() {
                assert_eq!(
                    stream.push(piece),
                    pair(delta),
                    "case {name} {chunking} piece {i}"
                );
            }
            assert_eq!(
                stream.finish(),
                pair(&s["finish"]),
                "case {name} {chunking} finish"
            );
            streams += 1;
        }
    }
    println!(
        "C4: {} forced cases, {streams} streams pass",
        reasoning.len()
    );
}

/// `separate_reasoning: false`: no parser (`serving_chat.py:700, 1809`), so a think block stays in the content,
/// unary and streamed, whatever the chunking.
#[test]
fn separate_reasoning_false_keeps_all_text_as_content() {
    for text in [
        "",
        "<think>a</think>b",
        "r</think>c",
        "<thi",
        "x<tool_call>y",
    ] {
        let all = Split {
            reasoning: String::new(),
            normal: text.to_owned(),
        };
        assert_eq!(split(text, Reasoning::Off), all);
        let mut stream = Stream::new(Reasoning::Off);
        let mut got = Split::default();
        for c in text.chars() {
            got.append(stream.push(&c.to_string()));
        }
        got.append(stream.finish());
        assert_eq!(got, all, "{text:?}");
    }
}

/// The resolution cered applies before the parser (D4/D13) matches `force_reasoning` of every oracle request that
/// renders: thinking on <=> forced.
#[test]
fn resolved_thinking_equals_the_oracle_force_reasoning() {
    let mut checked = 0;
    for case in cases().iter().filter(|c| c["kind"] == "resolve") {
        let Some(force) = case["expect"]["force_reasoning"].as_bool() else {
            continue;
        };
        let request = &case["request"];
        let explicit = request["chat_template_kwargs"]["enable_thinking"].as_bool();
        let effort = &request["reasoning_effort"];
        let resolved = explicit.unwrap_or(!effort.is_null() && effort.as_str() != Some("none"));
        let separate = request["separate_reasoning"].as_bool().unwrap_or(true);
        assert_eq!(
            Reasoning::resolve(resolved, separate),
            match (separate, force) {
                (false, _) => Reasoning::Off,
                (true, true) => Reasoning::Forced,
                (true, false) => Reasoning::Detect,
            },
            "case {}",
            case["name"]
        );
        checked += 1;
    }
    assert!(checked >= 30);
}
