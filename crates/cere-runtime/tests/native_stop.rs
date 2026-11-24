//! C6 (spec m24 D11): the native stop matcher and the response text it ships (`stop::Text`, the one the native
//! request path builds) against `tests/fixtures/native/stop_cases.json`, recorded in the pinned image from SGLang's
//! `SamplingParams.normalize`, `Req` finish checks (one accepted token per step, `stream_interval` 1) and
//! `trim_matched_stop`, over `stop_tokenizer_mini.json`.
mod native_support;

use cere_compiler::sha256_hex;
use cere_runtime::native::detok::Vocab;
use cere_runtime::native::stop::{Finished, Matched, Step, StopMatcher, Text};
use native_support::root;
use serde_json::{json, Value};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(root().join("tests/fixtures/native").join(name)).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
        .collect()
}

/// `(finish_type, matched, finished_len)` as the fixture spells them.
fn verdict(finished: &Finished) -> Value {
    let (kind, matched) = match &finished.matched {
        Matched::Str(stop) => ("stop", json!(stop)),
        Matched::Token(id) => ("stop", json!(id)),
        Matched::Length => ("length", Value::Null),
        Matched::Repetition => ("repetition", Value::Null),
    };
    json!([kind, matched, finished.len])
}

#[test]
fn stop_cases_all_pass() {
    let v: Value = serde_json::from_slice(&fixture("stop_cases.json")).unwrap();
    assert_eq!(v["schema"], "cere-stop-cases-v1");
    assert_eq!(
        (
            v["stream_interval"].as_u64(),
            v["accepted_per_step"].as_u64()
        ),
        (Some(1), Some(1))
    );
    let tokenizer = fixture(v["tokenizer"].as_str().unwrap());
    assert_eq!(sha256_hex(&tokenizer), v["tokenizer_sha256"]);
    let vocab = Vocab::from_json(&tokenizer).unwrap();
    let eos = ids(&v["eos_token_ids"]);
    let cases = v["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let (mut steps, mut held) = (0, 0);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let stops: Vec<String> = serde_json::from_value(case["stop"].clone()).unwrap();
        let lens = ids(&case["stop_token_lens"]);
        let output = ids(&case["ids"]);
        let mut matcher = StopMatcher::new(
            &vocab,
            stops,
            &lens,
            eos.clone(),
            case["max_new_tokens"].as_u64().unwrap(),
        );
        let skip = case["skip_special_tokens"].as_bool().unwrap();
        let mut text = Text::new(vocab.decoder_with(skip));
        let recorded = case["steps"].as_array().unwrap();
        for step in recorded {
            let n = step["n"].as_u64().unwrap() as usize;
            let at = format!("case {name} step {n}");
            let verdict_of_step = matcher.push(output[n - 1]);
            text.push(output[n - 1], &verdict_of_step);
            match verdict_of_step {
                Step::Running { should_output } => {
                    assert_eq!(step["finished"], false, "{at}");
                    assert_eq!(step["should_output"], should_output, "{at}");
                    held += usize::from(!should_output);
                }
                Step::Finished(finished) => {
                    assert_eq!(step["finished"], true, "{at}");
                    assert_eq!(
                        verdict(&finished),
                        json!([step["finish_type"], step["matched"], step["finished_len"]]),
                        "{at}"
                    );
                }
            }
            steps += 1;
        }
        let finished = matcher
            .finished()
            .unwrap_or_else(|| panic!("case {name}: no finish"))
            .clone();
        let last = recorded.last().unwrap();
        assert_eq!(last["finished"], true, "case {name}");
        let fin = &case["final"];
        assert_eq!(
            verdict(&finished),
            json!([fin["finish_type"], fin["matched"], fin["finished_len"]]),
            "case {name} final"
        );
        assert_eq!(
            vocab.decode_with(&output[..finished.len], skip),
            fin["text_untrimmed"],
            "case {name} untrimmed"
        );
        assert_eq!(text.finish(&finished), fin["text"], "case {name} text");
    }
    println!(
        "C6: {} stop cases, {steps} steps ({held} held) pass",
        cases.len()
    );
}

/// loop-guard design note LG-D1: the loop guard's finish keeps every counted token and its text (no trim), later
/// tokens are not counted, and an output the matcher already finished keeps its own finish.
#[test]
fn repetition_finish_keeps_the_counted_tokens() {
    let tokenizer = fixture("stop_tokenizer_mini.json");
    let vocab = Vocab::from_json(&tokenizer).unwrap();
    let mut matcher = StopMatcher::new(&vocab, vec!["zz".to_owned()], &[1], Vec::new(), 100);
    let mut text = Text::new(vocab.decoder_with(true));
    for id in [0, 1, 0] {
        let step = matcher.push(id);
        assert!(matches!(step, Step::Running { .. }));
        text.push(id, &step);
    }
    let finished = matcher.finish_repetition().clone();
    assert_eq!(
        finished,
        Finished {
            matched: Matched::Repetition,
            len: 3
        }
    );
    assert_eq!(matcher.push(1), Step::Finished(finished.clone()));
    assert_eq!(matcher.counted(), 3);
    assert_eq!(text.finish(&finished), vocab.decode_with(&[0, 1, 0], true));
    // A budget finish stays the budget's.
    let mut matcher = StopMatcher::new(&vocab, Vec::new(), &[], Vec::new(), 1);
    matcher.push(0);
    assert_eq!(matcher.finish_repetition().matched, Matched::Length);
}

/// M2.12 D8/D9 (iii): `Req._check_token_based_finish` cases of `token_control_cases.json` (D11). The matcher's
/// token list is the model's stop tokens then the request's `stop_token_ids` (D8), and **empty** under
/// `ignore_eos` (D9 iii: no token finish at all, the budget still ends it).
#[test]
fn token_control_finish_cases_pass() {
    let v: Value = serde_json::from_slice(&fixture("token_control_cases.json")).unwrap();
    assert_eq!(v["schema"], "cere-token-control-cases-v1");
    let tokenizer = fixture("stop_tokenizer_mini.json");
    let vocab = Vocab::from_json(&tokenizer).unwrap();
    let eos = ids(&v["eos_token_ids"]);
    let mut n = 0;
    for case in v["cases"].as_array().unwrap() {
        if case["kind"] != "finish" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let output = ids(&case["ids"]);
        let mut list = eos.clone();
        if case["stop_token_ids"].is_array() {
            for id in ids(&case["stop_token_ids"]) {
                if !list.contains(&id) {
                    list.push(id);
                }
            }
        }
        if case["ignore_eos"] == true {
            list.clear();
        }
        let mut matcher = StopMatcher::new(&vocab, Vec::new(), &[], list, output.len() as u64);
        for &id in &output {
            if let Step::Finished(_) = matcher.push(id) {
                break;
            }
        }
        let finished = matcher
            .finished()
            .unwrap_or_else(|| panic!("case {name}: no finish"))
            .clone();
        let want = &case["expect"];
        assert_eq!(
            verdict(&finished),
            json!([want["finish_type"], want["matched"], want["finished_len"]]),
            "case {name}"
        );
        n += 1;
    }
    assert_eq!(n, 6);
}
