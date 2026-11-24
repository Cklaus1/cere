//! Inkling 2 Mini's output parsers (inkling design note S6) against SGLang's `--reasoning-parser inkling` and
//! `--tool-call-parser inkling` as recorded in the pinned image (`tests/fixtures/native/inkling_parser_cases.json`,
//! `scripts/native/inkling_reference/gen_parser_fixtures.py`): the unary split and tool parse, and the stream chain
//! at whole, per-code-point, token and tag-splitting chunking with `stop` and `abort` finishes.

mod native_support;

use cere_runtime::levers::Family;
use cere_runtime::native::inkling;
use cere_runtime::native::reasoning::{split, Reasoning, Stream};
use cere_runtime::native::tool_parser::{Chain, Format, Item};
use native_support::root;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = root().join("tests/fixtures/native/inkling_parser_cases.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(doc["schema"], "cere-inkling-parser-cases-v1");
    assert_eq!(doc["reasoning_parser"], "inkling");
    assert_eq!(doc["tool_call_parser"], "inkling");
    doc
}

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not a string: {v}"))
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
    cuts.windows(2).map(|w| &text[w[0]..w[1]]).collect()
}

fn calls_json(items: Vec<Item>) -> Value {
    Value::Array(
        items
            .into_iter()
            .map(|item| match item {
                Item::Call(c) => json!([c.index, c.name, c.arguments]),
                other => panic!("an Inkling stream yields whole calls only: {other:?}"),
            })
            .collect(),
    )
}

#[test]
fn the_family_selects_the_inkling_parsers() {
    assert_eq!(
        Reasoning::for_family(Family::Inkling2Loop, false, true),
        Reasoning::Inkling
    );
    // `separate_reasoning: false` runs no parser, as for every family.
    assert_eq!(
        Reasoning::for_family(Family::Inkling2Loop, true, false),
        Reasoning::Off
    );
    assert_eq!(
        Reasoning::for_family(Family::Qwen4Exp, true, true),
        Reasoning::Forced
    );
    assert_eq!(Format::of(Family::Inkling2Loop), Format::Inkling);
    assert_eq!(Format::of(Family::Qwen4Exp), Format::Qwen3Coder);
    assert_eq!(Format::of(Family::Sdar), Format::Qwen3Coder);
}

/// The unary path: `detect_and_parse`, then with tools `has_tool_call` gates `parse_non_stream`.
#[test]
fn unary_matches_sglang() {
    let doc = fixture();
    let cases = doc["cases"].as_array().unwrap();
    let (mut with_calls, mut with_reasoning) = (0, 0);
    for case in cases {
        let name = text(case, "name");
        let t = text(case, "text");
        let want = &case["unary"];
        let s = split(t, Reasoning::Inkling);
        let (content, calls) = if case["tools"] == true && inkling::has_tool_call(&s.normal) {
            inkling::parse_non_stream(&s.normal)
        } else {
            (s.normal.clone(), Vec::new())
        };
        let calls: Vec<Value> = calls
            .iter()
            .map(|c| json!([c.index, c.name, c.arguments]))
            .collect();
        assert_eq!(
            json!({ "reasoning": s.reasoning, "content": content, "calls": calls }),
            *want,
            "case {name}"
        );
        with_calls += usize::from(!calls.is_empty());
        with_reasoning += usize::from(!s.reasoning.is_empty());
    }
    assert!(
        with_calls >= 20 && with_reasoning >= 20,
        "{with_calls} {with_reasoning}"
    );
    println!(
        "inkling unary: {} cases pass ({with_calls} with calls, {with_reasoning} with reasoning)",
        cases.len()
    );
}

/// The SSE path: per piece the reasoning stream (no tools) or the chain into the Inkling tool detector, the
/// finish flush on the last piece of a `stop` finish only.
#[test]
fn stream_matches_sglang() {
    let doc = fixture();
    let mut streams = 0;
    for case in doc["cases"].as_array().unwrap() {
        let name = text(case, "name");
        let t = text(case, "text");
        let tools = case["tools"] == true;
        for s in case["streams"].as_array().unwrap() {
            let chunking = text(s, "chunking");
            let finish = text(s, "finish");
            let at = format!("case {name} chunking {chunking} finish {finish}");
            let pieces = chunks(t, &s["cuts"]);
            let pushes = s["pushes"].as_array().unwrap();
            assert_eq!(pieces.len(), pushes.len(), "{at}");
            let mut chain = Chain::new(tools.then_some(Format::Inkling), Reasoning::Inkling);
            let mut plain = Stream::new(Reasoning::Inkling);
            for (i, (piece, want)) in pieces.iter().zip(pushes).enumerate() {
                let last = i + 1 == pieces.len() && finish == "stop";
                let got = if tools {
                    let step = chain.push(piece, last);
                    json!([step.reasoning, step.normal, calls_json(step.items)])
                } else {
                    let mut split = plain.push(piece);
                    if last {
                        split.append(plain.finish());
                    }
                    json!([split.reasoning, split.normal, []])
                };
                assert_eq!(got, *want, "{at} push {i} ({piece:?})");
            }
            streams += 1;
        }
    }
    println!("inkling stream: {streams} streams pass");
}

// ---------------------------------------------------------------- API surfaces
//
// The Anthropic Messages and Responses surfaces render the chat pipeline's output (`crate::surface`), whatever the
// family; the chat body and chunks below take the shapes `native/serve.rs` emits (`collect_unary`, `Emitter`) from
// the Inkling parsers' results, so each surface is checked on an Inkling turn of thinking, text and tool calls.

const THINKING_TEXT: &str =
    "<|message_model|><|content_thinking|>The user asks why. One sentence.<|end_message|>\
<|message_model|><|content_text|>Rayleigh scattering.<|end_message|><|content_model_end_sampling|>";
const THINKING_CALLS: &str = "<|message_model|><|content_thinking|>Need the weather twice.<|end_message|>\
<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":2}}<|end_message|>\
<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\"}}<|end_message|>\
<|content_model_end_sampling|>";

fn tool_call(index: usize, name: &str, arguments: &str) -> Value {
    json!({
        "id": format!("call_{index:024}"), "index": index, "type": "function",
        "function": { "name": name, "arguments": arguments },
    })
}

/// The unary `chat.completion` of an Inkling turn (tools declared, `tool_choice: auto`).
fn chat_unary(output: &str) -> Value {
    let s = split(output, Reasoning::Inkling);
    let (content, calls) = if inkling::has_tool_call(&s.normal) {
        inkling::parse_non_stream(&s.normal)
    } else {
        (s.normal, Vec::new())
    };
    let tool_calls: Vec<Value> = calls
        .iter()
        .map(|c| tool_call(c.index, &c.name, &c.arguments))
        .collect();
    let finish = if tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    json!({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "inkling-2-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning_content": if s.reasoning.is_empty() { Value::Null } else { json!(s.reasoning) },
                "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
            },
            "logprobs": null, "finish_reason": finish, "matched_stop": null,
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17 },
    })
}

fn chunk(delta: Value, finish: Value) -> Value {
    json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "inkling-2-mini",
        "choices": [{ "index": 0, "delta": delta, "logprobs": null, "finish_reason": finish, "matched_stop": null }],
        "usage": null,
    })
}

/// The content chunks of an Inkling turn streamed per code point through the tool chain, then the finish chunk.
fn chat_chunks(output: &str) -> (Vec<Value>, Value) {
    let mut chain = Chain::new(Some(Format::Inkling), Reasoning::Inkling);
    let mut chunks = vec![chunk(
        json!({ "role": "assistant", "content": "" }),
        Value::Null,
    )];
    let text = |s: String| if s.is_empty() { Value::Null } else { json!(s) };
    let pieces: Vec<(usize, char)> = output.char_indices().collect();
    let mut called = false;
    for (i, (at, c)) in pieces.iter().enumerate() {
        let step = chain.push(&output[*at..*at + c.len_utf8()], i + 1 == pieces.len());
        if !step.reasoning.is_empty() || !step.normal.is_empty() {
            chunks.push(chunk(
                json!({ "role": null, "content": text(step.normal), "reasoning_content": text(step.reasoning),
                        "tool_calls": null }),
                Value::Null,
            ));
        }
        for item in step.items {
            let Item::Call(call) = item else {
                panic!("an Inkling stream yields whole calls only: {item:?}")
            };
            called = true;
            chunks.push(chunk(
                json!({ "role": null, "content": null, "reasoning_content": null,
                        "tool_calls": [tool_call(call.index, &call.name, &call.arguments)] }),
                Value::Null,
            ));
        }
    }
    let finish = chunk(json!({}), json!(if called { "tool_calls" } else { "stop" }));
    (chunks, finish["choices"][0].clone())
}

fn usage() -> Value {
    json!({ "prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17 })
}

#[test]
fn chat_completion_carries_thinking_text_and_calls() {
    let message = &chat_unary(THINKING_TEXT)["choices"][0]["message"];
    assert_eq!(
        message["reasoning_content"],
        "The user asks why. One sentence."
    );
    assert_eq!(message["content"], "Rayleigh scattering.");
    assert_eq!(message["tool_calls"], Value::Null);
    let body = chat_unary(THINKING_CALLS);
    let message = &body["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "Need the weather twice.");
    assert_eq!(message["content"], "");
    assert_eq!(
        message["tool_calls"],
        json!([
            tool_call(0, "get_weather", r#"{"city": "Paris", "days": 2}"#),
            tool_call(1, "get_weather", r#"{"city": "Oslo"}"#),
        ])
    );
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    // The stream's deltas concatenate to the unary message.
    for output in [THINKING_TEXT, THINKING_CALLS] {
        let (chunks, _) = chat_chunks(output);
        let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
        for c in &chunks[1..] {
            let delta = &c["choices"][0]["delta"];
            reasoning += delta["reasoning_content"].as_str().unwrap_or("");
            content += delta["content"].as_str().unwrap_or("");
            calls.extend(delta["tool_calls"].as_array().cloned().unwrap_or_default());
        }
        let message = &chat_unary(output)["choices"][0]["message"];
        assert_eq!(json!(reasoning), message["reasoning_content"]);
        assert_eq!(json!(content), message["content"]);
        assert_eq!(
            json!(calls),
            json!(message["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default())
        );
    }
}

fn anthropic_frames(bytes: &[u8]) -> Vec<(String, Value)> {
    let text = std::str::from_utf8(bytes).unwrap();
    text.split_terminator("\n\n")
        .map(|frame| {
            let (event, data) = frame.split_once('\n').unwrap();
            let data = serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap();
            (event.strip_prefix("event: ").unwrap().to_owned(), data)
        })
        .collect()
}

#[test]
fn anthropic_renders_thinking_text_and_tool_use() {
    use cere_runtime::anthropic::Echo;
    use cere_runtime::surface::SurfaceRender;
    let echo = Echo::for_model("inkling-2-mini");
    let unary = |output: &str| -> Value {
        serde_json::from_slice(&echo.render_unary(&chat_unary(output), "r")).unwrap()
    };
    let body = unary(THINKING_TEXT);
    let blocks = body["content"].as_array().unwrap();
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["thinking"], "The user asks why. One sentence.");
    assert_eq!(
        blocks[1],
        json!({ "type": "text", "text": "Rayleigh scattering." })
    );
    assert_eq!(blocks.len(), 2);
    assert_eq!(body["stop_reason"], "end_turn");
    let body = unary(THINKING_CALLS);
    let blocks = body["content"].as_array().unwrap();
    let kinds: Vec<&str> = blocks.iter().map(|b| b["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["thinking", "tool_use", "tool_use"]);
    assert_eq!(blocks[1]["name"], "get_weather");
    assert_eq!(blocks[1]["input"], json!({ "city": "Paris", "days": 2 }));
    assert_eq!(blocks[2]["input"], json!({ "city": "Oslo" }));
    assert_eq!(body["stop_reason"], "tool_use");

    for (output, want_blocks, want_stop) in [
        (THINKING_TEXT, vec!["thinking", "text"], "end_turn"),
        (
            THINKING_CALLS,
            vec!["thinking", "tool_use", "tool_use"],
            "tool_use",
        ),
    ] {
        let mut stream = echo.stream();
        let mut bytes = stream.open(Some(10));
        let (chunks, finish) = chat_chunks(output);
        for c in &chunks {
            bytes.extend(stream.chunk(c));
        }
        bytes.extend(stream.complete(&finish, Some(&usage())));
        let frames = anthropic_frames(&bytes);
        let started: Vec<&str> = frames
            .iter()
            .filter(|(e, _)| e == "content_block_start")
            .map(|(_, d)| d["content_block"]["type"].as_str().unwrap())
            .collect();
        assert_eq!(started, want_blocks, "{output}");
        let (mut thinking, mut text, mut json_parts) = (String::new(), String::new(), Vec::new());
        for (event, data) in &frames {
            if event != "content_block_delta" {
                continue;
            }
            let delta = &data["delta"];
            match delta["type"].as_str().unwrap() {
                "thinking_delta" => thinking += delta["thinking"].as_str().unwrap(),
                "text_delta" => text += delta["text"].as_str().unwrap(),
                "input_json_delta" => {
                    json_parts.push(delta["partial_json"].as_str().unwrap().to_owned())
                }
                _ => {}
            }
        }
        let message = &chat_unary(output)["choices"][0]["message"];
        assert_eq!(json!(thinking), message["reasoning_content"]);
        assert_eq!(text, message["content"].as_str().unwrap());
        let inputs: Vec<Value> = json_parts
            .iter()
            .map(|p| serde_json::from_str(p).unwrap())
            .collect();
        let want: Vec<Value> = message["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .map(|c| {
                        serde_json::from_str(c["function"]["arguments"].as_str().unwrap()).unwrap()
                    })
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(inputs, want, "{output}");
        let stop = frames.iter().find(|(e, _)| e == "message_delta").unwrap();
        assert_eq!(stop.1["delta"]["stop_reason"], want_stop);
        assert_eq!(frames.last().unwrap().0, "message_stop");
    }
}

fn responses_policy() -> cere_runtime::validate::ChatPolicy {
    cere_runtime::validate::ChatPolicy {
        served_model_name: "inkling-2-mini".to_owned(),
        served_models: vec!["inkling-2-mini".to_owned()],
        max_messages: 256,
        default_max_output_tokens: 4096,
        max_output_tokens: 32768,
        greedy: true,
        sampled: true,
        tools_greedy: true,
        tools_sampled: true,
        grammar_decodes: Vec::new(),
        json_schema_keywords: Vec::new(),
        max_top_logprobs: None,
        sampling_params: Vec::new(),
        max_choices: 1,
        native: true,
        images_greedy: false,
        images_sampled: false,
        remote_workers: false,
        image_context_tokens: None,
        image_speculation_greedy: false,
        image_speculation_sampled: false,
        image_capture: false,
        thinking_toggle: false,
        token_controls: Vec::new(),
        vocab_size: None,
    }
}

#[test]
fn responses_render_reasoning_message_and_function_calls() {
    use cere_runtime::responses::{self, Ids};
    use cere_runtime::surface::SurfaceRender;
    let request = json!({
        "model": "inkling-2-mini", "input": "Weather in Paris and Oslo?",
        "tools": [{ "type": "function", "name": "get_weather", "parameters": {
            "type": "object", "properties": { "city": { "type": "string" }, "days": { "type": "integer" } } } }],
    });
    let converted =
        responses::convert(&serde_json::to_vec(&request).unwrap(), &responses_policy()).unwrap();
    let summary = |body: &Value| -> Vec<Value> {
        body["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| match item["type"].as_str().unwrap() {
                "reasoning" => json!(["reasoning", item["content"][0]["text"]]),
                "message" => json!(["message", item["content"][0]["text"]]),
                "function_call" => json!(["function_call", item["name"], item["arguments"]]),
                other => panic!("output item {other}"),
            })
            .collect()
    };
    let unary = |output: &str| -> Value {
        let node = responses::render_unary(
            &converted.echo,
            &mut Ids::seeded([3; 16]),
            &chat_unary(output),
        );
        serde_json::from_slice(&node.to_bytes()).unwrap()
    };
    let text_items = json!([
        ["reasoning", "The user asks why. One sentence."],
        ["message", "Rayleigh scattering."],
    ]);
    let call_items = json!([
        ["reasoning", "Need the weather twice."],
        [
            "function_call",
            "get_weather",
            r#"{"city": "Paris", "days": 2}"#
        ],
        ["function_call", "get_weather", r#"{"city": "Oslo"}"#],
    ]);
    assert_eq!(json!(summary(&unary(THINKING_TEXT))), text_items);
    assert_eq!(json!(summary(&unary(THINKING_CALLS))), call_items);

    let renderer = responses::Responses::new(converted.echo.clone()).unwrap();
    for (output, want) in [(THINKING_TEXT, &text_items), (THINKING_CALLS, &call_items)] {
        let mut stream = renderer.stream();
        let mut sse = stream.open(Some(10));
        let (chunks, finish) = chat_chunks(output);
        for c in &chunks {
            sse.extend(stream.chunk(c));
        }
        sse.extend(stream.complete(&finish, Some(&usage())));
        let text = String::from_utf8(sse).unwrap();
        let completed = text
            .split("\n\n")
            .filter_map(|frame| frame.strip_prefix("event: response.completed\ndata: "))
            .map(|data| serde_json::from_str::<Value>(data).unwrap())
            .next()
            .unwrap_or_else(|| panic!("no response.completed: {text}"));
        assert_eq!(json!(summary(&completed["response"])), *want, "{output}");
    }
}
