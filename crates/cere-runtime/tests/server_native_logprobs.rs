//! C5 (spec m27 D7/F7): the native `logprobs` rendering, unary and SSE, over the CPU toy executor and the stub
//! sidecar ([`native_server`]). The toy commits its greedy token through the M2.7 sampler path when logprobs are
//! asked for: its row holds `8 - j` at id `(next + j) mod vocab` for `j < 8` and -inf elsewhere, so the committed
//! token is the toy's `next`, its logprob `-ln S` and the `j`-th alternative's `-j - ln S` (`S = Σ_{k<8} e^-k`);
//! past the eight finite entries the top list continues with -inf entries, ids ascending (ties to the lower id).
//!
//! The expected bodies are built here from SGLang's code, not from C3 fixtures (no C3 fixture renders these
//! cases): `serving_chat.py:1918-1958` (`{token, bytes, logprob, top_logprobs}`, bytes = UTF-8 of the token text),
//! `utils.py:17-51` (`{text: logprob}` as a Python dict: a later duplicate text keeps the first position and takes
//! the later value), the streaming attachment of `serving_chat.py:675-805` (a step's logprobs ride on its first
//! reasoning or content chunk, never on a tool-parser chunk, else on a standalone chunk) and the recorded -inf
//! rendering (`sampler_cases.json` `neg_inf_logprob`: the stream renders `null`, the unary rendering is not JSON, so
//! the entry is omitted, D7).
mod native_server;
mod native_support;
mod support;

use cere_runtime::native::detok::Vocab;
use cere_runtime::native::reasoning::{Split, Stream};
use native_server::{Case, Native};
use native_support::{root, toy_tokens, FIXTURE_VOCAB};
use serde_json::{json, Value};
use support::{check_stream, Event, SseReader, MODEL};

/// `TEXT` of `server_native.rs`: the toy's output for this prompt is `[2235, 2, 780, …]`, no stop ID in reach.
const TEXT: (usize, char) = (716, '1');
/// `STOP` of `server_native.rs`: `[2240, 7]`.
const STOP: (usize, char) = (566, '6');
const SENTINEL: &str = "cere-native-sentinel-prompt";
/// The toy's finite logits per row.
const FINITE: usize = 8;

fn prompt((len, last): (usize, char)) -> String {
    let mut s = SENTINEL.to_owned();
    while s.len() < len - 2 {
        s.push('x');
    }
    s.push(last);
    s
}

/// A greedy request with `logprobs: true` and `top_logprobs` (absent when `None`).
fn body(p: (usize, char), max_tokens: u64, stream: bool, top: Option<u32>) -> Value {
    let mut b = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(p)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "logprobs": true,
    });
    if let Some(top) = top {
        b["top_logprobs"] = json!(top);
    }
    b
}

fn toy_output((len, last): (usize, char), max_tokens: usize) -> Vec<u32> {
    let mut ids = toy_tokens(last as u32, len as u64, max_tokens, FIXTURE_VOCAB);
    if let Some(i) = ids.iter().position(|id| [6, 7].contains(id)) {
        ids.truncate(i + 1);
    }
    ids
}

/// A ByteLevel `tokenizer.json`: every toy ID an added token `[id]` unless `pieces` gives it another text, and the
/// IDs of `bytes` model-vocabulary tokens of that ByteLevel alphabet character (one byte each).
fn tokenizer(pieces: &[(u32, &str)], bytes: &[(u32, &str)], special: &[u32]) -> Vec<u8> {
    let added: Vec<Value> = (0..FIXTURE_VOCAB)
        .filter(|id| bytes.iter().all(|(i, _)| i != id))
        .map(|id| {
            let content = pieces
                .iter()
                .find(|(i, _)| *i == id)
                .map_or_else(|| format!("[{id}]"), |(_, p)| (*p).to_owned());
            json!({"id": id, "content": content, "special": special.contains(&id)})
        })
        .collect();
    let vocab: serde_json::Map<String, Value> = bytes
        .iter()
        .map(|(id, c)| ((*c).to_owned(), json!(id)))
        .collect();
    serde_json::to_vec(&json!({
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab},
        "added_tokens": added,
        "decoder": {"type": "ByteLevel"},
    }))
    .unwrap()
}

async fn native(tokenizer: Vec<u8>) -> Native {
    Native::ready(Case {
        tokenizer: Some(tokenizer),
        ..Case::matching()
    })
    .await
}

/// `ln S`, as the toy's reference computes it (`sampler_ref::logprobs`: f64, rounded to f32).
fn ln_s() -> f64 {
    (0..FINITE).map(|k| (-(k as f64)).exp()).sum::<f64>().ln()
}

/// One rendered entry (text = `decode_with(&[id], false)`, bytes its UTF-8); `None` = -inf.
fn entry(vocab: &Vocab, id: u32, logprob: Option<f64>) -> Value {
    let token = vocab.decode_with(&[id], false);
    json!({"token": token, "bytes": token.as_bytes(), "logprob": f32_value(logprob)})
}

/// The device's f32, widened.
fn f32_value(logprob: Option<f64>) -> Option<f64> {
    logprob.map(|v| f64::from(v as f32))
}

/// The toy's `(id, logprob)` top list after committing `next`, `top` long (-inf = `None`).
fn toy_top(next: u32, top: usize) -> Vec<(u32, Option<f64>)> {
    let finite: Vec<u32> = (0..FINITE as u32)
        .map(|j| (next + j) % FIXTURE_VOCAB)
        .collect();
    let mut out: Vec<(u32, Option<f64>)> = finite
        .iter()
        .enumerate()
        .map(|(j, &id)| (id, Some(-(j as f64) - ln_s())))
        .collect();
    out.extend(
        (0..FIXTURE_VOCAB)
            .filter(|id| !finite.contains(id))
            .map(|id| (id, None)),
    );
    out.truncate(top);
    out
}

/// The expected content entry of committed `next`: the stream renders -inf as `null`, unary omits it (D7).
fn toy_entry(vocab: &Vocab, next: u32, top: usize, stream: bool) -> Value {
    let mut e = entry(vocab, next, Some(-ln_s()));
    let mut tops: Vec<(String, Option<f64>)> = Vec::new();
    for (id, lp) in toy_top(next, top) {
        let text = vocab.decode_with(&[id], false);
        match tops.iter_mut().find(|(t, _)| *t == text) {
            Some(slot) => slot.1 = lp,
            None => tops.push((text, lp)),
        }
    }
    e["top_logprobs"] = tops
        .into_iter()
        .filter(|(_, lp)| stream || lp.is_some())
        .map(|(t, lp)| json!({"token": t, "bytes": t.as_bytes(), "logprob": f32_value(lp)}))
        .collect();
    e
}

/// `actual` equals `expected` with logprob numbers within 1e-6 and exactly f32 values (SGLang's floats are the
/// device's f32 widened to f64).
fn assert_close(actual: &Value, expected: &Value, at: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!((a - e).abs() <= 1e-6, "{at}: {a} vs {e}");
            // Within an f64 ULP or two (serde_json's default float parsing is not correctly rounded): an f32
            // printed as f32 (`-2.4583397`) is ~1e-8 away.
            let widened = f64::from(a as f32);
            assert!(
                (widened - a).abs() <= 4.0 * f64::EPSILON * a.abs(),
                "{at}: {a} is not an f32 value widened"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{at}: {actual} vs {expected}");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                assert_close(a, e, &format!("{at}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                e.keys().collect::<Vec<_>>(),
                "{at}"
            );
            for (k, e) in e {
                assert_close(&a[k], e, &format!("{at}.{k}"));
            }
        }
        _ => assert_eq!(actual, expected, "{at}"),
    }
}

async fn unary(native: &Native, b: &Value) -> Value {
    let response = native.chat(b).send().await.unwrap();
    assert_eq!(response.status(), 200);
    response.json().await.unwrap()
}

/// The stream's data choices (role chunk first, finish chunk last; the usage chunk has none).
async fn stream(native: &Native, b: &Value) -> Vec<Value> {
    let response = native.chat(b).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let events = SseReader::new(response).rest().await;
    check_stream(&events);
    events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) => v.pointer("/choices/0").cloned(),
            Event::Done => None,
        })
        .collect()
}

/// Every streamed `logprobs.content` entry, in order: what a client concatenates.
fn streamed_entries(choices: &[Value]) -> Vec<Value> {
    choices
        .iter()
        .filter_map(|c| c["logprobs"]["content"].as_array())
        .flatten()
        .cloned()
        .collect()
}

/// One SSE chunk as `(content, reasoning_content, logprobs entries)`.
fn view(c: &Value) -> (Value, Value, Value) {
    (
        c["delta"]["content"].clone(),
        c["delta"]["reasoning_content"].clone(),
        c["logprobs"]["content"].clone(),
    )
}

/// The plain (no tool parser) stream of per-token texts `pieces` with their entries: per step, the reasoning
/// parser's split (and, on the last step, its end flush), each non-empty one a chunk; the step's entry rides on
/// its first chunk, or on a standalone chunk when the step emitted none (`serving_chat.py:694-805`).
fn plain_oracle(pieces: &[&str], entries: &[Value]) -> Vec<(Value, Value, Value)> {
    let text = |s: String| if s.is_empty() { Value::Null } else { json!(s) };
    let mut parser = Stream::default();
    let mut out = Vec::new();
    for (i, (piece, e)) in pieces.iter().zip(entries).enumerate() {
        let mut splits = vec![parser.push(piece)];
        if i + 1 == pieces.len() {
            splits.push(parser.finish());
        }
        let mut attach = Some(json!([e]));
        for Split { reasoning, normal } in splits {
            if reasoning.is_empty() && normal.is_empty() {
                continue;
            }
            out.push((
                text(normal),
                text(reasoning),
                attach.take().unwrap_or(Value::Null),
            ));
        }
        if let Some(entries) = attach {
            out.push((Value::Null, Value::Null, entries));
        }
    }
    out
}

/// The chunks between the role chunk and the finish chunk, which both carry `logprobs: null`.
fn body_chunks(choices: &[Value]) -> Vec<(Value, Value, Value)> {
    let (first, last) = (&choices[0], choices.last().unwrap());
    assert_eq!(first["delta"]["role"], "assistant");
    assert_eq!(first["logprobs"], Value::Null);
    assert!(!last["finish_reason"].is_null(), "{last}");
    assert_eq!(last["logprobs"], Value::Null);
    choices[1..choices.len() - 1].iter().map(view).collect()
}

/// Unary `choices[0].logprobs = {content: [...]}` with one entry per committed token (top_logprobs 2, 0 and
/// absent = 0: `[]`); SSE carries the same entries, one per token step on that step's content chunk; logprobs
/// off renders `null` in both.
#[tokio::test]
async fn unary_and_stream_shapes() {
    let native = native(tokenizer(&[], &[], &[])).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let ids = toy_output(TEXT, 3);
    for top in [Some(2), Some(0), None] {
        let n = top.unwrap_or(0) as usize;
        let want: Vec<Value> = ids
            .iter()
            .map(|&id| toy_entry(&vocab, id, n, false))
            .collect();
        let u = unary(&native, &body(TEXT, 3, false, top)).await;
        assert_close(
            &u["choices"][0]["logprobs"],
            &json!({ "content": want }),
            "unary",
        );
        assert_eq!(u["usage"]["completion_tokens"], ids.len());

        let choices = stream(&native, &body(TEXT, 3, true, top)).await;
        let pieces: Vec<String> = ids.iter().map(|id| format!("[{id}]")).collect();
        let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let oracle = plain_oracle(&pieces, &want);
        let got = body_chunks(&choices);
        assert_eq!(got.len(), oracle.len(), "{got:?}");
        for (i, (g, o)) in got.iter().zip(&oracle).enumerate() {
            assert_eq!((&g.0, &g.1), (&o.0, &o.1), "chunk {i}");
            assert_close(&g.2, &o.2, &format!("chunk {i}"));
        }
    }
    let mut off = body(TEXT, 3, false, None);
    off.as_object_mut().unwrap().remove("logprobs");
    assert_eq!(
        unary(&native, &off).await["choices"][0]["logprobs"],
        Value::Null
    );
    off["stream"] = json!(true);
    for c in stream(&native, &off).await {
        assert_eq!(c["logprobs"], Value::Null, "{c}");
    }
}

/// `top_logprobs` is `{text: logprob}` as a Python dict: two alternatives with one text collapse into the first's
/// position with the later's logprob, so fewer than N entries come back.
#[tokio::test]
async fn duplicate_texts_collapse_as_a_python_dict() {
    let next = toy_output(TEXT, 1)[0];
    let (a, b) = ((next + 1) % FIXTURE_VOCAB, (next + 3) % FIXTURE_VOCAB);
    let native = native(tokenizer(&[(a, "dup"), (b, "dup")], &[], &[])).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let lp = |j: f64| Some(-j - ln_s());
    let tops = vec![
        entry(&vocab, next, lp(0.0)),
        entry(&vocab, a, lp(3.0)),
        entry(&vocab, (next + 2) % FIXTURE_VOCAB, lp(2.0)),
        entry(&vocab, (next + 4) % FIXTURE_VOCAB, lp(4.0)),
    ];
    let mut want = entry(&vocab, next, lp(0.0));
    want["top_logprobs"] = json!(tops);
    assert_eq!(tops[1]["token"], "dup");
    assert_close(&toy_entry(&vocab, next, 5, false), &want, "oracle");
    let u = unary(&native, &body(TEXT, 1, false, Some(5))).await;
    assert_close(
        &u["choices"][0]["logprobs"],
        &json!({ "content": [want] }),
        "unary",
    );
    let choices = stream(&native, &body(TEXT, 1, true, Some(5))).await;
    assert_close(&json!(streamed_entries(&choices)), &json!([want]), "stream");
}

/// A token that is a lone UTF-8 lead byte renders as its own decode, U+FFFD (bytes EF BF BD); its step emits no
/// text (the decoder holds the byte), so its logprobs ride on a standalone chunk.
#[tokio::test]
async fn partial_utf8_token_renders_replacement_bytes() {
    let ids = toy_output(TEXT, 3);
    // `Ã` is byte C3 in the ByteLevel alphabet: a UTF-8 lead byte without its continuation.
    let native = native(tokenizer(&[], &[(ids[1], "Ã")], &[])).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    assert_eq!(vocab.decode_with(&[ids[1]], false), "\u{FFFD}");
    let want: Vec<Value> = ids
        .iter()
        .map(|&id| toy_entry(&vocab, id, 1, false))
        .collect();
    assert_eq!(want[1]["token"], "\u{FFFD}");
    assert_eq!(want[1]["bytes"], json!([0xEF, 0xBF, 0xBD]));
    let u = unary(&native, &body(TEXT, 3, false, Some(1))).await;
    assert_close(
        &u["choices"][0]["logprobs"],
        &json!({ "content": want }),
        "unary",
    );
    let choices = stream(&native, &body(TEXT, 3, true, Some(1))).await;
    let got = body_chunks(&choices);
    let first = json!(format!("[{}]", ids[0]));
    let last = json!(format!("\u{FFFD}[{}]", ids[2]));
    assert_eq!(got.len(), 3, "{got:?}");
    assert_eq!((&got[0].0, &got[0].1), (&first, &Value::Null));
    assert_eq!((&got[1].0, &got[1].1), (&Value::Null, &Value::Null));
    assert_eq!((&got[2].0, &got[2].1), (&last, &Value::Null));
    for (i, g) in got.iter().enumerate() {
        assert_close(&g.2, &json!([want[i]]), &format!("chunk {i}"));
    }
}

/// A stop string trims the text, not the logprobs: one entry per counted token, the matching token's whole text.
/// SSE: the step whose window ends with a stop prefix emits no text (a standalone logprobs chunk); the finishing
/// step's entry rides on the trimmed rest.
#[tokio::test]
async fn stop_string_trims_text_not_logprobs() {
    let native = native(tokenizer(&[], &[], &[])).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let ids = toy_output(TEXT, 2);
    let want: Vec<Value> = ids
        .iter()
        .map(|&id| toy_entry(&vocab, id, 2, false))
        .collect();
    let mut u = body(TEXT, 16, false, Some(2));
    u["stop"] = json!(["5][2"]);
    let r = unary(&native, &u).await;
    assert_eq!(r["choices"][0]["message"]["content"], "[223");
    assert_eq!(r["usage"]["completion_tokens"], 2);
    assert_close(
        &r["choices"][0]["logprobs"],
        &json!({ "content": want }),
        "unary",
    );
    assert_eq!(want[1]["token"], "[2]");

    u["stream"] = json!(true);
    let choices = stream(&native, &u).await;
    let got = body_chunks(&choices);
    assert_eq!(got.len(), 2, "{got:?}");
    assert_eq!((&got[0].0, &got[0].1), (&Value::Null, &Value::Null));
    assert_eq!((&got[1].0, &got[1].1), (&json!("[223"), &Value::Null));
    for (i, g) in got.iter().enumerate() {
        assert_close(&g.2, &json!([want[i]]), &format!("chunk {i}"));
    }
}

/// Reasoning: a step's entry rides on its reasoning or content chunk, or alone when the parser emitted nothing
/// (the `<think>` / `</think>` steps); unary carries every token, tags included.
#[tokio::test]
async fn reasoning_steps_carry_their_logprobs() {
    let ids = toy_output(TEXT, 4);
    let texts = ["<think>", "plan", "</think>", "answer"];
    let pieces: Vec<(u32, &str)> = ids.iter().copied().zip(texts).collect();
    let native = native(tokenizer(&pieces, &[], &[])).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let want: Vec<Value> = ids
        .iter()
        .map(|&id| toy_entry(&vocab, id, 3, false))
        .collect();
    let u = unary(&native, &body(TEXT, 4, false, Some(3))).await;
    assert_eq!(u["choices"][0]["message"]["reasoning_content"], "plan");
    assert_close(
        &u["choices"][0]["logprobs"],
        &json!({ "content": want }),
        "unary",
    );
    let choices = stream(&native, &body(TEXT, 4, true, Some(3))).await;
    let oracle = plain_oracle(&texts, &want);
    let got = body_chunks(&choices);
    assert_eq!(got.len(), oracle.len(), "{got:?} vs {oracle:?}");
    for (i, (g, o)) in got.iter().zip(&oracle).enumerate() {
        assert_eq!((&g.0, &g.1), (&o.0, &o.1), "chunk {i}");
        assert_close(&g.2, &o.2, &format!("chunk {i}"));
    }
}

/// Tool calls: the tool parser's content and call chunks never carry logprobs; each step's entry follows them on
/// a standalone chunk (`serving_chat.py:777-805`). The stop token's entry renders its special text.
#[tokio::test]
async fn tool_call_steps_carry_logprobs_alone() {
    const CALL_TEXT: &str = "Let me check.<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
    let native = native(tokenizer(
        &[(7, "<|im_end|>"), (6, "<|endoftext|>"), (2240, CALL_TEXT)],
        &[],
        &[6, 7],
    ))
    .await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let ids = toy_output(STOP, 8);
    assert_eq!(ids, [2240, 7]);
    let want: Vec<Value> = ids
        .iter()
        .map(|&id| toy_entry(&vocab, id, 1, false))
        .collect();
    assert_eq!(want[1]["token"], "<|im_end|>");
    // The stub renders tools as a leading `[2]`: one prompt token fewer gives the same toy output.
    let mut b = body((STOP.0 - 1, STOP.1), 8, false, Some(1));
    b["tools"] = json!([{"type": "function", "function": {
        "name": "get_weather",
        "description": "The weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}},
    }}]);
    let u = unary(&native, &b).await;
    assert_eq!(u["choices"][0]["finish_reason"], "tool_calls");
    assert_close(
        &u["choices"][0]["logprobs"],
        &json!({ "content": want }),
        "unary",
    );
    b["stream"] = json!(true);
    let choices = stream(&native, &b).await;
    let got = body_chunks(&choices);
    let mut carried = Vec::new();
    for (i, c) in choices[1..choices.len() - 1].iter().enumerate() {
        let d = &c["delta"];
        if c["logprobs"].is_null() {
            continue;
        }
        assert!(
            d["content"].is_null() && d["reasoning_content"].is_null() && d["tool_calls"].is_null(),
            "chunk {i}: {c}"
        );
        carried.push(got[i].2.clone());
    }
    assert!(choices.iter().any(|c| c["delta"]["tool_calls"].is_array()));
    assert_close(
        &json!(carried),
        &json!(want.iter().map(|e| json!([e])).collect::<Vec<_>>()),
        "stream",
    );
}

/// `top_logprobs: 20` over a toy row with 8 finite logits (D7 -inf rule): SGLang's stream renders the 12 -inf
/// entries with `"logprob": null` (recorded, valid JSON: mirrored); its unary rendering is not JSON, so unary
/// omits them and returns 8.
#[tokio::test]
async fn neg_inf_alternatives_null_in_stream_omitted_in_unary() {
    let recorded = {
        let path = root().join("tests/fixtures/native/sampler_cases.json");
        let cases: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        cases["neg_inf_logprob"].clone()
    };
    assert_eq!(recorded["unary"]["valid_json"], false);
    assert_eq!(recorded["stream"]["valid_json"], true);
    let stream_text: Value =
        serde_json::from_str(recorded["stream"]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        stream_text["content"][0]["top_logprobs"][1]["logprob"],
        Value::Null
    );

    let native = native(tokenizer(&[], &[], &[])).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let next = toy_output(TEXT, 1)[0];
    let u = unary(&native, &body(TEXT, 1, false, Some(20))).await;
    let unary_want = toy_entry(&vocab, next, 20, false);
    assert_eq!(unary_want["top_logprobs"].as_array().unwrap().len(), FINITE);
    assert_close(
        &u["choices"][0]["logprobs"],
        &json!({ "content": [unary_want] }),
        "unary",
    );
    let stream_want = toy_entry(&vocab, next, 20, true);
    let tops = stream_want["top_logprobs"].as_array().unwrap();
    assert_eq!(tops.len(), 20);
    assert_eq!(tops[FINITE]["logprob"], Value::Null);
    assert_eq!(tops[FINITE]["token"], "[0]");
    let choices = stream(&native, &body(TEXT, 1, true, Some(20))).await;
    assert_close(
        &json!(streamed_entries(&choices)),
        &json!([stream_want]),
        "stream",
    );
}

/// M2.8 C7 (D9, R16-2): `top_logprobs: 20` on a constrained step with one allowed id (the stub's regex script
/// `seq:100,7` allows only 100 at position 0). Logprobs are post-mask, so the committed 100 has logprob 0 and the
/// other 19 alternatives are -inf (ids ascending, 100 skipped), rendered by the D7 rule of
/// [`neg_inf_alternatives_null_in_stream_omitted_in_unary`]: `null` in the stream, omitted in unary.
#[tokio::test]
async fn constrained_step_with_fewer_allowed_ids_than_top_logprobs() {
    let native = native(tokenizer(&[], &[], &[])).await;
    let vocab = Vocab::load(&native.artifact).unwrap();
    let request = |stream: bool| {
        let mut b = body(TEXT, 1, stream, Some(20));
        b["regex"] = json!("seq:100,7");
        b
    };
    let want = |stream: bool| {
        let mut e = entry(&vocab, 100, Some(0.0));
        let mut tops = vec![entry(&vocab, 100, Some(0.0))];
        if stream {
            tops.extend(
                (0..)
                    .filter(|id| *id != 100)
                    .take(19)
                    .map(|id| entry(&vocab, id, None)),
            );
        }
        e["top_logprobs"] = json!(tops);
        e
    };
    let u = unary(&native, &request(false)).await;
    assert_close(
        &u["choices"][0]["logprobs"],
        &json!({ "content": [want(false)] }),
        "unary",
    );
    let choices = stream(&native, &request(true)).await;
    assert_close(
        &json!(streamed_entries(&choices)),
        &json!([want(true)]),
        "stream",
    );
}
