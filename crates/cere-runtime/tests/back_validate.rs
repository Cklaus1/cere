//! A3: accepted/rejected request table (each rejection names its field) and the explicit backend body.

mod image_support;

use cere_runtime::config::RuntimeConfig;
use cere_runtime::digest::canonical_json;
use cere_runtime::sglang::Rid;
use cere_runtime::validate::{
    assistant_reply, ChatMessage, ChatPolicy, Content, Role, SamplerMode, ToolCallAssembly,
    ToolChoice, MID_SYSTEM_CLOSE, MID_SYSTEM_OPEN,
};
use image_support::{b64, jpeg_url, png, png_url};
use serde_json::{json, Value};

const MODEL: &str = "Qwen3.8-Flash-Next";

fn live_config() -> RuntimeConfig {
    RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/sglang-t4-qwen38.toml"),
    )
    .unwrap()
}

fn policy() -> ChatPolicy {
    ChatPolicy::from_config(&live_config(), 0).unwrap()
}

/// The minimal valid body with `patch` merged in (`null` in the patch removes a key).
fn request(patch: Value) -> Value {
    let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]});
    for (key, value) in patch.as_object().unwrap() {
        if value.is_null() {
            body.as_object_mut().unwrap().remove(key);
        } else {
            body[key] = value.clone();
        }
    }
    body
}

fn messages(n: usize) -> Value {
    Value::Array(
        (0..n)
            .map(|_| json!({"role": "user", "content": "x"}))
            .collect(),
    )
}

#[test]
fn policy_reads_config_and_profile() {
    let policy = policy();
    assert_eq!(policy.served_model_name, MODEL);
    assert_eq!(policy.max_messages, 512);
    assert_eq!(policy.default_max_output_tokens, 4096);
    assert_eq!(policy.max_output_tokens, 32768);
    assert!(policy.greedy && policy.sampled);
}

#[test]
fn accepted_requests() {
    let policy = policy();
    let accepted = [
        json!({}),
        json!({"n": 1}),
        json!({"max_tokens": 1}),
        json!({"max_completion_tokens": 32768}),
        json!({"max_tokens": 16, "max_completion_tokens": Value::Null}),
        json!({"temperature": 0}),
        json!({"temperature": 2, "top_p": 1}),
        json!({"top_p": 0.5}),
        json!({"stop": "END"}),
        json!({"stop": ["a", "b"]}),
        // Audit F4: the caps are inclusive; the length is in bytes, not characters.
        json!({"stop": vec!["s"; 16]}),
        json!({"stop": "x".repeat(256)}),
        json!({"stop": ["\u{e9}".repeat(128)]}),
        json!({"stream": true}),
        json!({"stream": true, "stream_options": {"include_usage": true}}),
        json!({"stream": false}),
        json!({"messages": messages(512)}),
        json!({"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "a"}
        ]}),
    ];
    for patch in accepted {
        let body = request(patch.clone());
        assert!(
            policy.validate(&body).is_ok(),
            "{patch} → {:?}",
            policy.validate(&body)
        );
    }
}

#[test]
fn rejected_requests_name_the_field() {
    let policy = policy();
    // (patch, status, code, param)
    let rejected = [
        (json!({"seed": 7}), 400, "unsupported_parameter", "seed"),
        (json!({"tools": []}), 400, "invalid_value", "tools"),
        (
            json!({"tool_choice": "required"}),
            400,
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"response_format": {"type": "structural_tag"}}),
            400,
            "unsupported_parameter",
            "response_format.type",
        ),
        // D9: the SGLang profile does not declare `min_p` (or `seed`, above).
        (json!({"min_p": 0.1}), 400, "unsupported_parameter", "min_p"),
        // Q2: the salt is server-derived; a client cannot pick a cache namespace.
        (
            json!({"cache_salt": "x"}),
            400,
            "unsupported_parameter",
            "cache_salt",
        ),
        (
            json!({"rid": "cere-x"}),
            400,
            "unsupported_parameter",
            "rid",
        ),
        (
            json!({"made_up": 1}),
            400,
            "unsupported_parameter",
            "made_up",
        ),
        (json!({"n": 2}), 400, "unsupported_parameter", "n"),
        (
            json!({"stream": true, "stream_options": {"include_usage": true, "continuous_usage_stats": true}}),
            400,
            "unsupported_parameter",
            "stream_options.continuous_usage_stats",
        ),
        (
            json!({"stream_options": {"include_usage": true}}),
            400,
            "invalid_value",
            "stream_options",
        ),
        (
            json!({"stream": true, "stream_options": {"include_usage": 1}}),
            400,
            "invalid_value",
            "stream_options.include_usage",
        ),
        (json!({"stream": "yes"}), 400, "invalid_value", "stream"),
        (
            json!({"messages": [{"role": "assistant", "content": Value::Null}]}),
            400,
            "unsupported_parameter",
            "messages[0].content",
        ),
        (
            json!({"messages": [{"role": "tool", "content": "a", "tool_call_id": "c", "name": 5}]}),
            400,
            "invalid_value",
            "messages[0].name",
        ),
        (
            json!({"messages": [{"role": "tool", "content": "a", "tool_call_id": "c", "name": {"n": 1}}]}),
            400,
            "invalid_value",
            "messages[0].name",
        ),
        (
            json!({"messages": [{"role": "user", "content": "a", "name": 5}]}),
            400,
            "invalid_value",
            "messages[0].name",
        ),
        (
            json!({"messages": [{"role": "tool", "content": "a"}]}),
            400,
            "missing_required_parameter",
            "messages[0].tool_call_id",
        ),
        (
            json!({"messages": [{"role": "critic", "content": "a"}]}),
            400,
            "unsupported_parameter",
            "messages[0].role",
        ),
        (
            json!({"messages": [{"content": "a"}]}),
            400,
            "missing_required_parameter",
            "messages[0].role",
        ),
        (
            json!({"messages": [{"role": "user"}]}),
            400,
            "missing_required_parameter",
            "messages[0].content",
        ),
        (
            json!({"messages": ["hi"]}),
            400,
            "invalid_value",
            "messages[0]",
        ),
        (json!({"messages": []}), 400, "invalid_value", "messages"),
        (
            json!({"messages": messages(513)}),
            400,
            "invalid_value",
            "messages",
        ),
        (
            json!({"messages": Value::Null}),
            400,
            "missing_required_parameter",
            "messages",
        ),
        (
            json!({"max_tokens": 8, "max_completion_tokens": 8}),
            400,
            "conflicting_parameters",
            "max_completion_tokens",
        ),
        (
            json!({"max_tokens": 32769}),
            400,
            "invalid_value",
            "max_tokens",
        ),
        (
            json!({"max_completion_tokens": 0}),
            400,
            "invalid_value",
            "max_completion_tokens",
        ),
        (
            json!({"max_tokens": -1}),
            400,
            "invalid_value",
            "max_tokens",
        ),
        (
            json!({"max_tokens": 1.5}),
            400,
            "invalid_value",
            "max_tokens",
        ),
        (json!({"model": "gpt-4o"}), 404, "model_not_found", "model"),
        (
            json!({"model": Value::Null}),
            400,
            "missing_required_parameter",
            "model",
        ),
        (json!({"model": 1}), 400, "invalid_value", "model"),
        (
            json!({"temperature": 2.5}),
            400,
            "invalid_value",
            "temperature",
        ),
        (
            json!({"temperature": -0.1}),
            400,
            "invalid_value",
            "temperature",
        ),
        (json!({"top_p": 0}), 400, "invalid_value", "top_p"),
        (json!({"top_p": 1.1}), 400, "invalid_value", "top_p"),
        (json!({"stop": ""}), 400, "invalid_value", "stop"),
        (json!({"stop": ["a", 1]}), 400, "invalid_value", "stop[1]"),
        (json!({"stop": 3}), 400, "invalid_value", "stop"),
        // Audit F4: 17 strings, one string of 257 bytes (the one 129 two-byte characters make), named by index.
        (json!({"stop": vec!["s"; 17]}), 400, "invalid_value", "stop"),
        (
            json!({"stop": "x".repeat(257)}),
            400,
            "invalid_value",
            "stop",
        ),
        (
            json!({"stop": ["a", "\u{e9}".repeat(129)]}),
            400,
            "invalid_value",
            "stop[1]",
        ),
    ];
    for (patch, status, code, param) in rejected {
        let error = policy.validate(&request(patch.clone())).unwrap_err();
        assert_eq!(
            (error.status, error.code, error.param.as_deref()),
            (status, code, Some(param)),
            "{patch}: {}",
            error.message
        );
        let body = error.body();
        assert_eq!(body["error"]["param"], param);
        assert_eq!(body["error"]["code"], code);
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
    let error = policy
        .validate(&json!(["not", "an", "object"]))
        .unwrap_err();
    assert_eq!(
        (error.status, error.code, error.param),
        (400, "invalid_json", None)
    );
}

#[test]
fn sampling_requires_a_validated_profile_combination() {
    let mut config = live_config();
    config.deployments[0]
        .lock
        .as_mut()
        .unwrap()
        .profile
        .validated_combinations
        .retain(|c| c.sampler_mode != "sampled");
    let greedy_only = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(greedy_only.greedy && !greedy_only.sampled);
    for patch in [
        json!({}),
        json!({"temperature": 0.7}),
        json!({"top_p": 0.9}),
    ] {
        let error = greedy_only.validate(&request(patch.clone())).unwrap_err();
        assert_eq!(
            (error.code, error.param.as_deref()),
            ("unsupported_parameter", Some("temperature")),
            "{patch}"
        );
    }
    let greedy = greedy_only
        .validate(&request(json!({"temperature": 0, "top_p": 0.9})))
        .unwrap();
    assert_eq!(greedy.sampler, SamplerMode::Greedy);
}

#[test]
fn defaults_are_explicit_in_the_backend_body() {
    let policy = policy();
    let rid = Rid::new(2, 5, 0xab);
    let unary = policy.validate(&request(json!({"stop": "END"}))).unwrap();
    assert_eq!(unary.max_output_tokens, 4096);
    assert_eq!(unary.sampler, SamplerMode::Sampled);
    assert_eq!(unary.messages, [ChatMessage::text(Role::User, "hi")]);
    assert_eq!(
        serde_json::to_value(unary.backend_body(&rid)).unwrap(),
        json!({
            "rid": rid.as_str(),
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 4096,
            "stop": ["END"],
            "stream": false,
        })
    );

    let streamed = policy
        .validate(&request(
            json!({"stream": true, "max_tokens": 9, "temperature": 0, "top_p": 0.5}),
        ))
        .unwrap();
    assert!(!streamed.include_usage);
    assert_eq!(
        serde_json::to_value(streamed.backend_body(&rid)).unwrap(),
        json!({
            "rid": rid.as_str(),
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 9,
            "temperature": 0.0,
            "top_p": 0.5,
            "stream": true,
            "stream_options": {"include_usage": true},
        })
    );
}

#[test]
fn normalization_equates_equivalent_spellings() {
    let policy = policy();
    let a = policy
        .validate(&request(json!({"max_tokens": 4096, "stop": "x", "n": 1})))
        .unwrap();
    let b = policy
        .validate(&request(json!({"stop": ["x"], "temperature": Value::Null})))
        .unwrap();
    assert_eq!(a.normalized(), b.normalized());
    let c = policy
        .validate(&request(json!({"max_tokens": 4095})))
        .unwrap();
    assert_ne!(a.normalized(), c.normalized());
}

/// The six pre-M2.4 bodies of the C2 regression guard.
fn pre_m24_bodies() -> [Value; 6] {
    [
        request(json!({})),
        request(json!({"temperature": 0, "stop": "END", "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "a b"},
            {"role": "assistant", "content": "c"},
            {"role": "user", "content": "d"}
        ]})),
        request(
            json!({"stream": true, "stream_options": {"include_usage": true}, "max_tokens": 9, "top_p": 0.5}),
        ),
        request(
            json!({"stop": ["\u{e9}", "x y"], "messages": [{"role": "user", "content": "\u{e9}\u{1f600}\n\"q\""}]}),
        ),
        request(
            json!({"max_completion_tokens": 16, "n": 1, "temperature": Value::Null, "stream": false}),
        ),
        request(json!({"temperature": 0.7, "stream": true, "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "b"}
        ]})),
    ]
}

/// C2 regression guard: `normalized()` (request digest input) and the message values (session digest input) of
/// six pre-M2.4 bodies equal their canonical JSON recorded on `85a123e`.
#[test]
fn c2_pre_m24_digest_inputs_are_unchanged() {
    let policy = policy();
    let mut got = Vec::new();
    for body in pre_m24_bodies() {
        let request = policy.validate(&body).unwrap();
        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        got.push((
            canonical_json(&request.normalized()),
            canonical_json(&Value::Array(messages)),
        ));
    }
    let golden = [
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":false,"temperature":null,"top_p":null}"#,
            r#"[{"content":"hi","role":"user"}]"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"s","role":"system"},{"content":"a b","role":"user"},{"content":"c","role":"assistant"},{"content":"d","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":["END"],"stream":false,"temperature":0.0,"top_p":null}"#,
            r#"[{"content":"s","role":"system"},{"content":"a b","role":"user"},{"content":"c","role":"assistant"},{"content":"d","role":"user"}]"#,
        ),
        (
            r#"{"include_usage":true,"max_completion_tokens":9,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":true,"temperature":null,"top_p":0.5}"#,
            r#"[{"content":"hi","role":"user"}]"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"é😀\n\"q\"","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":["é","x y"],"stream":false,"temperature":null,"top_p":null}"#,
            r#"[{"content":"é😀\n\"q\"","role":"user"}]"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":16,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":false,"temperature":null,"top_p":null}"#,
            r#"[{"content":"hi","role":"user"}]"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"","role":"user"},{"content":"a","role":"assistant"},{"content":"b","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":true,"temperature":0.7,"top_p":null}"#,
            r#"[{"content":"","role":"user"},{"content":"a","role":"assistant"},{"content":"b","role":"user"}]"#,
        ),
    ];
    for (i, ((request, messages), (want_request, want_messages))) in
        got.iter().zip(golden).enumerate()
    {
        assert_eq!(request, want_request, "body {i}: normalized()");
        assert_eq!(messages, want_messages, "body {i}: session messages");
    }
}

fn native_policy() -> ChatPolicy {
    let config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0.toml"),
    )
    .unwrap();
    ChatPolicy::from_config(&config, 0).unwrap()
}

/// The shipped native profile without M2.7 D11's additions (no sampled rows, `sampling_params` or
/// `max_top_logprobs`): what a greedy-only native profile refuses.
fn native_greedy_policy() -> ChatPolicy {
    let mut config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0.toml"),
    )
    .unwrap();
    let profile = &mut config.deployments[0].lock.as_mut().unwrap().profile;
    profile
        .validated_combinations
        .retain(|c| c.sampler_mode != "sampled");
    profile.sampling_params = None;
    profile.max_top_logprobs = None;
    profile.validate().unwrap();
    ChatPolicy::from_config(&config, 0).unwrap()
}

/// O51 (spec m22 X12): every validator probe of `scripts/live/native_parity.py`'s fail-closed phase
/// (`tests/fixtures/x12_probes.json`) is refused with its fixture code and param on the profiles X12 runs on - the
/// shipped native-g0 and the snapshot profile of `--native-config` runs - while the harness's parity request shape
/// (temperature 0, stream false) is served. A probe the validator serves is a stale probe that fails every live run.
#[test]
fn x12_probes_are_refused_by_the_shipped_native_profiles() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/x12_probes.json")).unwrap(),
    )
    .unwrap();
    let probes = fixture["probes"].as_array().unwrap();
    assert!(probes.len() >= 8, "{}", probes.len());
    let base = json!({"model": MODEL, "messages": [{"role": "user", "content": "x12"}], "max_tokens": 1,
                      "temperature": 0, "stream": false});
    for config in [
        "configs/runtime/native-g0.toml",
        "configs/deployments/qwen38-flash-next-native-snapshot.toml",
    ] {
        let policy =
            ChatPolicy::from_config(&RuntimeConfig::load(root.join(config)).unwrap(), 0).unwrap();
        assert!(policy.native, "{config}");
        policy.validate(&base).unwrap();
        for probe in probes {
            let mut body = base.clone();
            for (key, value) in probe["fields"].as_object().unwrap() {
                body[key] = value.clone();
            }
            let error = match policy.validate(&body) {
                Ok(_) => panic!("{config}: X12 probe {} is served", probe["name"]),
                Err(error) => error,
            };
            assert_eq!(
                (error.status, error.code, error.param.as_deref()),
                (
                    400,
                    probe["code"].as_str().unwrap(),
                    probe["param"].as_str()
                ),
                "{config}: X12 probe {}: {}",
                probe["name"],
                error.message
            );
        }
    }
}

/// SelfTune ST4 S3d: the DICE-8B GPU0 dev deployment (`configs/runtime/native-g0-dice8b.toml`, profile
/// `native-g0-dice-8b-v1`) serves greedy text only. A greedy text request is accepted; sampling, logprobs, a
/// grammar, `n` > 1, tools, images, thinking fields and token controls are each 400 naming the field.
#[test]
fn dice_8b_profile_serves_greedy_text_only() {
    assert_greedy_text_only("configs/runtime/native-g0-dice8b.toml", "DICE-8B", 1, false);
}

/// Inkling S5 and S6g (inkling design note D-I5): the Inkling 2 Mini GPU0 dev deployments serve greedy and sampled
/// text, logprobs, auto tool calls and images, `n` up to their 4 sessions (decode_batch_max 4) and the template's
/// `reasoning_effort` (`thinking_toggle`; golden p06 renders effort "none"). A grammar, `n` 5 and token controls are
/// each 400 naming the field. Both lanes: the default (`configs/runtime/native-g0-inkling.toml`, profile
/// `native-g0-inkling-2-mini-mtp-v68`, MTP drafter on since D-I-S9B) and the drafts-off one
/// (`configs/runtime/native-g0-inkling-plain.toml`, profile `native-g0-inkling-2-mini-v1`).
#[test]
fn inkling_2_mini_profile_serves_sampled_tools_and_images() {
    assert_inkling_serves_sampled_tools_and_images("configs/runtime/native-g0-inkling.toml");
}

#[test]
fn inkling_2_mini_plain_profile_serves_sampled_tools_and_images() {
    assert_inkling_serves_sampled_tools_and_images("configs/runtime/native-g0-inkling-plain.toml");
}

fn assert_inkling_serves_sampled_tools_and_images(path: &str) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config = RuntimeConfig::load(root.join(path)).unwrap();
    let policy = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(policy.native && policy.greedy && policy.sampled);
    let base = json!({"model": "Inkling-2-Mini", "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 8, "temperature": 0, "stream": false});
    let accepted: [(&str, Value); 7] = [
        ("greedy", json!({})),
        (
            "sampled",
            json!({"temperature": 0.8, "top_p": 0.95, "top_k": 50, "seed": 7}),
        ),
        ("logprobs", json!({"logprobs": true, "top_logprobs": 5})),
        ("n", json!({"n": 4})),
        ("tools", json!({"tools": [tool("f")]})),
        (
            "image",
            json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": png_url(2, 2)}}, {"type": "text", "text": "x"}]}]}),
        ),
        (
            "reasoning_effort",
            json!({"reasoning_effort": "none", "separate_reasoning": false}),
        ),
    ];
    let refused: [(&str, Value, &str); 3] = [
        (
            "json_schema",
            json!({"response_format": {"type": "json_schema", "json_schema":
            {"name": "x", "schema": {"type": "object"}}}}),
            "response_format",
        ),
        ("n", json!({"n": 5}), "n"),
        (
            "stop_token_ids",
            json!({"stop_token_ids": [1]}),
            "stop_token_ids",
        ),
    ];
    let patched = |patch: &Value| {
        let mut body = base.clone();
        for (key, value) in patch.as_object().unwrap() {
            body[key] = value.clone();
        }
        body
    };
    for (name, patch) in accepted {
        if let Err(error) = policy.validate(&patched(&patch)) {
            panic!("{name} refused: {:?} {}", error.param, error.message);
        }
    }
    for (name, patch, param) in refused {
        let error = match policy.validate(&patched(&patch)) {
            Ok(_) => panic!("Inkling serves {name}"),
            Err(error) => error,
        };
        assert_eq!(error.status, 400, "{name}: {}", error.message);
        assert!(
            error.param.as_deref().is_some_and(|p| p.starts_with(param)),
            "{name}: param {:?} ({}), want {param}",
            error.param,
            error.message
        );
    }
}

/// Inkling S13 AMD: the opt-in Vulkan lane (`configs/runtime/native-amd-inkling-vk.toml`, profile
/// `native-amd-inkling-2-mini-vk-v67`, device `vk:0`) loads (its shared levers at their defaults pass on Vulkan: the
/// inkling library runs none of them) and serves greedy text only with one session.
#[test]
fn inkling_2_mini_evo_vulkan_profile_serves_greedy_text_only() {
    assert_greedy_text_only(
        "configs/runtime/native-amd-inkling-vk.toml",
        "Inkling-2-Mini",
        1,
        true,
    );
}

/// `max_n`: the largest `n` the profile serves (accepted; one more is the `n` refusal). `thinking`: the profile
/// declares `thinking_toggle` (a thinking request accepted), else the thinking fields are refused.
fn assert_greedy_text_only(toml: &str, model: &str, max_n: u64, thinking: bool) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config = RuntimeConfig::load(root.join(toml)).unwrap();
    let policy = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(policy.native && policy.greedy && !policy.sampled);
    let base = json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                      "temperature": 0, "stream": false});
    policy.validate(&base).unwrap();
    let mut most = base.clone();
    most["n"] = json!(max_n);
    policy.validate(&most).unwrap();
    let mut effort = base.clone();
    effort["reasoning_effort"] = json!("none");
    effort["separate_reasoning"] = json!(false);
    assert_eq!(
        policy.validate(&effort).is_ok(),
        thinking,
        "{model}: reasoning_effort"
    );
    let mut cases: Vec<(&str, Value, &str)> = vec![
        ("sampled", json!({"temperature": 0.7}), "temperature"),
        ("logprobs", json!({"logprobs": true}), "logprobs"),
        (
            "top_logprobs",
            json!({"logprobs": true, "top_logprobs": 2}),
            "logprobs",
        ),
        (
            "json_schema",
            json!({"response_format": {"type": "json_schema", "json_schema":
            {"name": "x", "schema": {"type": "object"}}}}),
            "response_format",
        ),
        ("n", json!({"n": max_n + 1}), "n"),
        ("tools", json!({"tools": [tool("f")]}), "tools"),
        (
            "image",
            json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": png_url(2, 2)}}]}]}),
            "messages[0].content[1]",
        ),
        (
            "stop_token_ids",
            json!({"stop_token_ids": [1]}),
            "stop_token_ids",
        ),
    ];
    if !thinking {
        cases.push((
            "thinking",
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            "chat_template_kwargs",
        ));
    }
    for (name, patch, param) in cases {
        let mut body = base.clone();
        for (key, value) in patch.as_object().unwrap() {
            body[key] = value.clone();
        }
        let error = match policy.validate(&body) {
            Ok(_) => panic!("{model} serves {name}"),
            Err(error) => error,
        };
        assert_eq!(error.status, 400, "{name}: {}", error.message);
        assert!(
            error.param.as_deref().is_some_and(|p| p.starts_with(param)),
            "{name}: param {:?} ({}), want {param}",
            error.param,
            error.message
        );
    }
}

fn tool(name: &str) -> Value {
    json!({"type": "function", "function": {
        "name": name,
        "description": "d",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }})
}

/// The assistant message SGLang returns for a tool call (`choices[0].message`), echoed verbatim.
fn echoed_call() -> Value {
    json!({"role": "assistant", "content": Value::Null, "reasoning_content": Value::Null, "tool_calls": [
        {"id": "call_1", "index": 0, "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
    ]})
}

/// A tool session of `n` messages: a system and a user message, then assistant tool calls and their results.
fn tool_session(n: usize) -> Value {
    let mut session = vec![
        json!({"role": "system", "content": "s"}),
        json!({"role": "user", "content": "u"}),
    ];
    while session.len() < n {
        session.push(if session.len() % 2 == 0 {
            echoed_call()
        } else {
            json!({"role": "tool", "tool_call_id": "call_1", "content": "r"})
        });
    }
    Value::Array(session)
}

/// max-messages design note MM-03: the speculation profiles and production (`prod-t4.toml`) serve long tool
/// sessions (frontend's capture: p99.9 1,930 messages, max 2,274; the proxy's chat form max 2,291) up to 4,096
/// messages and refuse one more with the cap's 400.
#[test]
fn the_speculation_and_production_profiles_take_4096_messages() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for config in [
        "configs/runtime/native-g0-spec.toml",
        "configs/runtime/prod-t4.toml",
    ] {
        let policy =
            ChatPolicy::from_config(&RuntimeConfig::load(root.join(config)).unwrap(), 0).unwrap();
        assert_eq!(policy.max_messages, 4096, "{config}");
        let body = |n| json!({"model": MODEL, "messages": tool_session(n), "tools": [tool("get_weather")]});
        for n in [2300, 4096] {
            if let Err(error) = policy.validate(&body(n)) {
                panic!("{config}: {n} messages: {}", error.message);
            }
        }
        let error = match policy.validate(&body(4097)) {
            Ok(_) => panic!("{config}: 4,097 messages are served"),
            Err(error) => error,
        };
        assert_eq!(
            (
                error.status,
                error.code,
                error.param.as_deref(),
                error.message.as_str()
            ),
            (
                400,
                "invalid_value",
                Some("messages"),
                "at most 4096 messages are allowed"
            ),
            "{config}"
        );
    }
}

/// `parameters` with `depth` nested containers, the `parameters` object itself being depth 1.
fn nested(depth: usize) -> Value {
    let mut value = json!({});
    for _ in 1..depth {
        value = json!({"properties": value});
    }
    value
}

/// C1 (D2/D3/D12): every M2.4 shape the SGLang qwen38 profile accepts.
#[test]
fn c1_tool_and_logprob_shapes_are_accepted() {
    let policy = policy();
    assert!(policy.tools_greedy && policy.tools_sampled);
    assert_eq!(policy.max_top_logprobs, Some(20));
    let conversation = json!([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "weather?"},
        echoed_call(),
        {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "It is sunny.", "reasoning_content": Value::Null, "tool_calls": Value::Null},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "function": {"name": "f", "arguments": "{}"}}
        ]},
        {"role": "user", "content": "thanks"}
    ]);
    let accepted = [
        json!({"tools": [tool("get_weather")]}),
        json!({"tools": [tool("a"), tool("b-c_D9")], "tool_choice": "auto", "parallel_tool_calls": false}),
        json!({"tools": [tool("a")], "tool_choice": "none", "temperature": 0}),
        json!({"tools": [{"type": "function", "function": {"name": "bare"}}]}),
        json!({"tools": [{"type": "function", "function": {"name": "s", "strict": false, "description": Value::Null, "parameters": Value::Null}}]}),
        json!({"tools": [{"type": "function", "function": {"name": "deep", "parameters": nested(32)}}]}),
        json!({"tools": (0..128).map(|i| tool(&format!("t{i}"))).collect::<Vec<_>>()}),
        json!({"tool_choice": "auto"}),
        json!({"parallel_tool_calls": true}),
        json!({"logprobs": true}),
        json!({"logprobs": true, "top_logprobs": 0}),
        json!({"logprobs": true, "top_logprobs": 20}),
        json!({"logprobs": false}),
        json!({"messages": conversation, "tools": [tool("get_weather")]}),
        json!({"messages": conversation}),
    ];
    for patch in accepted {
        let body = request(patch.clone());
        let verdict = policy.validate(&body);
        assert!(verdict.is_ok(), "{patch} → {verdict:?}");
    }
    let parsed = policy
        .validate(&request(json!({"messages": [echoed_call()], "tools": [tool("x")], "logprobs": true, "top_logprobs": 3})))
        .unwrap();
    assert!(parsed.render_tools() && parsed.logprobs);
    assert_eq!(parsed.top_logprobs, Some(3));
    assert_eq!(parsed.messages[0].content, None);
    let call = &parsed.messages[0].tool_calls.as_ref().unwrap()[0];
    assert_eq!(
        (call.id.as_str(), call.kind.as_str()),
        ("call_1", "function")
    );
    assert_eq!(call.function.arguments, "{\"city\": \"Paris\"}");
    let none = policy
        .validate(&request(
            json!({"tools": [tool("x")], "tool_choice": "none"}),
        ))
        .unwrap();
    assert!(none.tools_json().is_some() && !none.render_tools());
}

/// C1: each M2.4 refusal names its field path.
#[test]
fn c1_tool_and_logprob_refusals_name_their_path() {
    let policy = policy();
    let fun = |function: Value| json!({"tools": [{"type": "function", "function": function}]});
    let msg = |message: Value| json!({"messages": [{"role": "user", "content": "q"}, message]});
    let call = |call: Value| {
        msg(json!({"role": "assistant", "content": Value::Null, "tool_calls": [call]}))
    };
    let rejected = [
        (
            json!({"tool_choice": "sometimes"}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"parallel_tool_calls": "yes"}),
            "invalid_value",
            "parallel_tool_calls",
        ),
        (
            fun(json!({"name": "a", "strict": 1})),
            "invalid_value",
            "tools[0].function.strict",
        ),
        // D2: `strict` only `false` (SGLang's `Function.strict: bool` refuses null).
        (
            fun(json!({"name": "a", "strict": null})),
            "invalid_value",
            "tools[0].function.strict",
        ),
        (
            fun(json!({"name": "bad name"})),
            "invalid_value",
            "tools[0].function.name",
        ),
        (
            fun(json!({"name": ""})),
            "invalid_value",
            "tools[0].function.name",
        ),
        (
            fun(json!({"name": "x".repeat(65)})),
            "invalid_value",
            "tools[0].function.name",
        ),
        (
            fun(json!({"name": 7})),
            "invalid_value",
            "tools[0].function.name",
        ),
        (
            fun(json!({"description": "d"})),
            "missing_required_parameter",
            "tools[0].function.name",
        ),
        (
            json!({"tools": [tool("a"), tool("b"), tool("a")]}),
            "invalid_value",
            "tools[2].function.name",
        ),
        (
            fun(json!({"name": "a", "parameters": []})),
            "invalid_value",
            "tools[0].function.parameters",
        ),
        (
            fun(json!({"name": "a", "parameters": nested(33)})),
            "invalid_value",
            "tools[0].function.parameters",
        ),
        (
            fun(json!({"name": "a", "description": 3})),
            "invalid_value",
            "tools[0].function.description",
        ),
        (
            fun(json!({"name": "a", "defer_loading": true})),
            "unsupported_parameter",
            "tools[0].function.defer_loading",
        ),
        (
            json!({"tools": [{"type": "function", "function": {"name": "a"}, "cache_control": {}}]}),
            "unsupported_parameter",
            "tools[0].cache_control",
        ),
        (
            json!({"tools": [{"type": "custom", "function": {"name": "a"}}]}),
            "invalid_value",
            "tools[0].type",
        ),
        (
            json!({"tools": [{"function": {"name": "a"}}]}),
            "missing_required_parameter",
            "tools[0].type",
        ),
        (
            json!({"tools": [{"type": "function"}]}),
            "missing_required_parameter",
            "tools[0].function",
        ),
        (json!({"tools": ["a"]}), "invalid_value", "tools[0]"),
        (
            json!({"tools": {"type": "function"}}),
            "invalid_value",
            "tools",
        ),
        (
            json!({"tools": (0..129).map(|i| tool(&format!("t{i}"))).collect::<Vec<_>>()}),
            "invalid_value",
            "tools",
        ),
        (json!({"top_logprobs": 2}), "invalid_value", "top_logprobs"),
        (
            json!({"logprobs": false, "top_logprobs": 2}),
            "invalid_value",
            "top_logprobs",
        ),
        (
            json!({"logprobs": true, "top_logprobs": 21}),
            "invalid_value",
            "top_logprobs",
        ),
        (
            json!({"logprobs": true, "top_logprobs": -1}),
            "invalid_value",
            "top_logprobs",
        ),
        (json!({"logprobs": "yes"}), "invalid_value", "logprobs"),
        (
            msg(json!({"role": "tool", "content": "r"})),
            "missing_required_parameter",
            "messages[1].tool_call_id",
        ),
        (
            msg(json!({"role": "tool", "content": "r", "tool_call_id": 5})),
            "invalid_value",
            "messages[1].tool_call_id",
        ),
        (
            msg(json!({"role": "tool", "content": Value::Null, "tool_call_id": "c"})),
            "unsupported_parameter",
            "messages[1].content",
        ),
        (
            msg(json!({"role": "assistant", "content": Value::Null})),
            "unsupported_parameter",
            "messages[1].content",
        ),
        (
            msg(json!({"role": "assistant", "content": Value::Null, "tool_calls": []})),
            "invalid_value",
            "messages[1].tool_calls",
        ),
        (
            msg(json!({"role": "assistant", "content": "a", "reasoning_content": 5})),
            "invalid_value",
            "messages[1].reasoning_content",
        ),
        (
            msg(json!({"role": "assistant", "content": "a", "reasoning_content": ["t"]})),
            "invalid_value",
            "messages[1].reasoning_content",
        ),
        (
            msg(json!({"role": "user", "content": "a", "reasoning_content": "t"})),
            "unsupported_parameter",
            "messages[1].reasoning_content",
        ),
        (
            msg(json!({"role": "user", "content": "a", "tool_calls": Value::Null})),
            "unsupported_parameter",
            "messages[1].tool_calls",
        ),
        (
            msg(json!({"role": "user", "content": "a", "tool_call_id": "c"})),
            "unsupported_parameter",
            "messages[1].tool_call_id",
        ),
        (
            call(json!({"id": "c", "function": {"name": "f", "arguments": "[1]"}})),
            "invalid_value",
            "messages[1].tool_calls[0].function.arguments",
        ),
        (
            call(json!({"id": "c", "function": {"name": "f", "arguments": "{"}})),
            "invalid_value",
            "messages[1].tool_calls[0].function.arguments",
        ),
        (
            call(json!({"id": "c", "function": {"name": "f", "arguments": {"a": 1}}})),
            "invalid_value",
            "messages[1].tool_calls[0].function.arguments",
        ),
        (
            call(json!({"id": "c", "function": {"name": "f"}})),
            "missing_required_parameter",
            "messages[1].tool_calls[0].function.arguments",
        ),
        (
            call(json!({"function": {"name": "f", "arguments": "{}"}})),
            "missing_required_parameter",
            "messages[1].tool_calls[0].id",
        ),
        (
            call(json!({"id": "c", "index": -1, "function": {"name": "f", "arguments": "{}"}})),
            "invalid_value",
            "messages[1].tool_calls[0].index",
        ),
        (
            call(
                json!({"id": "c", "type": "custom", "function": {"name": "f", "arguments": "{}"}}),
            ),
            "invalid_value",
            "messages[1].tool_calls[0].type",
        ),
        (
            call(json!({"id": "c", "extra": 1, "function": {"name": "f", "arguments": "{}"}})),
            "unsupported_parameter",
            "messages[1].tool_calls[0].extra",
        ),
        (
            call(json!({"id": "c", "function": {"name": "f", "arguments": "{}", "x": 1}})),
            "unsupported_parameter",
            "messages[1].tool_calls[0].function.x",
        ),
    ];
    for (patch, code, param) in rejected {
        let error = policy.validate(&request(patch.clone())).unwrap_err();
        assert_eq!(
            (error.status, error.code, error.param.as_deref()),
            (400, code, Some(param)),
            "{patch}: {}",
            error.message
        );
    }
}

/// C1 (D1/D12): tools need a `tool_calls` decode combination for the sampler mode (and the qwen3_coder parser on
/// SGLang); `logprobs` needs `max_top_logprobs` (native names M2.7).
#[test]
fn c1_tools_and_logprobs_are_gated_by_the_profile() {
    let mut config = live_config();
    config.deployments[0]
        .lock
        .as_mut()
        .unwrap()
        .profile
        .validated_combinations
        .retain(|c| !(c.output == "tool_calls" && c.sampler_mode == "sampled"));
    let greedy_tools = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(greedy_tools.tools_greedy && !greedy_tools.tools_sampled);
    greedy_tools
        .validate(&request(json!({"tools": [tool("a")], "temperature": 0})))
        .unwrap();
    let error = greedy_tools
        .validate(&request(json!({"tools": [tool("a")]})))
        .unwrap_err();
    assert_eq!(
        (error.code, error.param.as_deref()),
        ("unsupported_parameter", Some("tools"))
    );

    let mut config = live_config();
    config.deployments[0]
        .expect
        .as_mut()
        .unwrap()
        .tool_call_parser = "hermes".to_owned();
    let other_parser = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(!other_parser.tools_greedy && !other_parser.tools_sampled);
    let error = other_parser
        .validate(&request(json!({"tools": [tool("a")]})))
        .unwrap_err();
    assert_eq!(
        (error.code, error.param.as_deref()),
        ("unsupported_parameter", Some("tools"))
    );

    let native = native_greedy_policy();
    assert!(native.tools_greedy && !native.tools_sampled);
    assert_eq!(native.max_top_logprobs, None);
    let body = |patch: Value| {
        let mut body = request(patch);
        body["temperature"] = json!(0);
        body
    };
    let tools = native
        .validate(&body(json!({"tools": [tool("a")]})))
        .unwrap();
    assert!(tools.render_tools());
    let error = native
        .validate(&body(json!({"logprobs": true})))
        .unwrap_err();
    assert_eq!(
        (error.code, error.param.as_deref()),
        ("unsupported_parameter", Some("logprobs"))
    );
    assert!(error.message.contains("M2.7"), "{}", error.message);
    native.validate(&body(json!({"logprobs": false}))).unwrap();

    // An SGLang profile without `max_top_logprobs` (e.g. gemma) refuses without naming the native milestone.
    let mut config = live_config();
    config.deployments[0]
        .lock
        .as_mut()
        .unwrap()
        .profile
        .max_top_logprobs = None;
    let error = ChatPolicy::from_config(&config, 0)
        .unwrap()
        .validate(&request(json!({"logprobs": true})))
        .unwrap_err();
    assert_eq!(
        (error.code, error.param.as_deref()),
        ("unsupported_parameter", Some("logprobs"))
    );
    assert!(
        !error.message.contains("native") && !error.message.contains("M2.7"),
        "{}",
        error.message
    );
}

/// Validates raw body text the way the HTTP route does (D4: `tools` keeps its bytes).
fn validate_text(text: &str) -> cere_runtime::validate::ChatRequest {
    let value: Value = serde_json::from_str(text).unwrap();
    policy().validate_bytes(&value, text.as_bytes()).unwrap()
}

/// A raw body: the minimal request plus raw `"key":value` members.
fn raw_body(extra: &str) -> String {
    format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#)
}

/// C2 (D7): tools that differ only in key order are different prompts, so they digest differently; an absent
/// `tool_choice` with tools equals `"auto"`, and the other defaults (`parallel_tool_calls: true`,
/// `logprobs: false`) equal their absence.
#[test]
fn c2_tools_digest_their_bytes_and_defaults_are_canonical() {
    let a = validate_text(&raw_body(
        r#","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","x-a":1}}}]"#,
    ));
    let b = validate_text(&raw_body(
        r#","tools":[{"function":{"name":"f","parameters":{"type":"object","x-a":1}},"type":"function"}]"#,
    ));
    assert_ne!(a.normalized(), b.normalized(), "key order must count");
    let c = validate_text(&raw_body(
        r#","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","x-a":1.0}}}]"#,
    ));
    assert_ne!(a.normalized(), c.normalized(), "number spelling must count");

    let tools = r#","tools":[{"type":"function","function":{"name":"f"}}]"#;
    let absent = validate_text(&raw_body(tools));
    for extra in [
        r#","tool_choice":"auto""#,
        r#","parallel_tool_calls":true"#,
        r#","logprobs":false"#,
    ] {
        let explicit = validate_text(&raw_body(&format!("{tools}{extra}")));
        assert_eq!(absent.normalized(), explicit.normalized(), "{extra}");
    }
    let none = validate_text(&raw_body(&format!(r#"{tools},"tool_choice":"none""#)));
    assert_ne!(absent.normalized(), none.normalized());
}

fn assistant_turn(call: Value) -> Vec<Value> {
    let body = request(json!({"messages": [
        {"role": "user", "content": "weather?"},
        call,
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]}));
    policy()
        .validate(&body)
        .unwrap()
        .messages
        .iter()
        .map(|m| serde_json::to_value(m).unwrap())
        .collect()
}

/// C2 (D3/D7) session digest inputs: `arguments` re-spelled digests differently; `content` null vs `""`,
/// with or without `tool_calls[].index`, null `reasoning_content`/`tool_calls` digest equal; the committed reply
/// (unary message and stream fragments) equals its verbatim echo.
#[test]
fn c2_assistant_messages_digest_canonically() {
    let call = |content: Value, arguments: &str, extra: Value| {
        let mut call = json!({"id": "call_1", "type": "function",
                              "function": {"name": "get_weather", "arguments": arguments}});
        for (k, v) in extra.as_object().unwrap() {
            call[k] = v.clone();
        }
        json!({"role": "assistant", "content": content, "tool_calls": [call]})
    };
    let base = assistant_turn(call(Value::Null, r#"{"city": "Paris"}"#, json!({})));
    assert_ne!(
        base,
        assistant_turn(call(Value::Null, r#"{"city":"Paris"}"#, json!({}))),
        "arguments re-spelled"
    );
    assert_eq!(
        base,
        assistant_turn(call(json!(""), r#"{"city": "Paris"}"#, json!({})))
    );
    assert_eq!(
        base,
        assistant_turn(call(
            Value::Null,
            r#"{"city": "Paris"}"#,
            json!({"index": 0})
        ))
    );
    let mut with_null_reasoning = call(Value::Null, r#"{"city": "Paris"}"#, json!({}));
    with_null_reasoning["reasoning_content"] = Value::Null;
    assert_eq!(base, assistant_turn(with_null_reasoning));
    let mut untyped = call(Value::Null, r#"{"city": "Paris"}"#, json!({}));
    untyped["tool_calls"][0]
        .as_object_mut()
        .unwrap()
        .remove("type");
    assert_eq!(base, assistant_turn(untyped));

    // A text reply: `tool_calls: null` equals its absence.
    let text = |extra: Value| {
        let mut m = json!({"role": "assistant", "content": "ok"});
        for (k, v) in extra.as_object().unwrap() {
            m[k] = v.clone();
        }
        assistant_turn(m)
    };
    assert_eq!(
        text(json!({})),
        text(json!({"tool_calls": Value::Null, "reasoning_content": Value::Null}))
    );

    // What `cered` commits for a reply equals the echo of `choices[0].message`.
    let echoed = json!({"role": "assistant", "content": Value::Null, "reasoning_content": Value::Null,
        "tool_calls": [{"id": "call_1", "index": 0, "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]});
    let echo = assistant_turn(echoed.clone())[1].clone();
    let mut unary = ToolCallAssembly::default();
    unary.push(echoed.get("tool_calls"));
    let committed = assistant_reply(String::new(), unary.into_calls());
    assert_eq!(serde_json::to_value(&committed).unwrap(), echo, "unary");
    let mut stream = ToolCallAssembly::default();
    for fragment in [
        json!([{"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": ""}}]),
        json!([{"index": 0, "function": {"arguments": "{\"city\": "}}]),
        json!([{"index": 0, "function": {"arguments": "\"Paris\"}"}}]),
    ] {
        stream.push(Some(&fragment));
    }
    let committed = assistant_reply(String::new(), stream.into_calls());
    assert_eq!(serde_json::to_value(&committed).unwrap(), echo, "stream");
}

/// O113 (owner 2026-10-17): an assistant message's string `reasoning_content` is kept, forwarded to SGLang (whose
/// chat template renders it in the turn's `<think>` block) and bound by the request digest; `""` digests and
/// forwards like its absence. A message `name` (system, user, assistant) is accepted and dropped: the pinned
/// template never reads it, so it neither forwards nor digests. Fails on main 49397403 (400 on both fields).
#[test]
fn o113_reasoning_content_and_name_are_accepted() {
    let policy = policy();
    let turn = |reasoning: Value, name: Option<&str>| {
        let mut assistant =
            json!({"role": "assistant", "content": "a", "reasoning_content": reasoning});
        let mut user = json!({"role": "user", "content": "next"});
        let mut system = json!({"role": "system", "content": "s"});
        if let Some(name) = name {
            for m in [&mut assistant, &mut user, &mut system] {
                m["name"] = json!(name);
            }
        }
        let body = request(
            json!({"messages": [system, {"role": "user", "content": "q"}, assistant, user]}),
        );
        match policy.validate(&body) {
            Ok(request) => request,
            Err(error) => panic!("{body} -> {error:?}"),
        }
    };
    let rid = Rid::new(1, 2, 3);
    let thought = turn(json!("plan the call"), Some("agent"));
    assert_eq!(
        thought.messages[2].reasoning_content.as_deref(),
        Some("plan the call")
    );
    let body = serde_json::to_value(thought.backend_body(&rid)).unwrap();
    assert_eq!(
        body["messages"][2],
        json!({"role": "assistant", "content": "a", "reasoning_content": "plan the call"})
    );
    assert_eq!(
        body["messages"][3],
        json!({"role": "user", "content": "next"})
    );
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": "s"})
    );
    let plain = turn(Value::Null, None);
    assert_eq!(
        thought.normalized(),
        turn(json!("plan the call"), None).normalized(),
        "name never digests"
    );
    assert_ne!(
        thought.normalized(),
        plain.normalized(),
        "reasoning digests"
    );
    assert_ne!(
        thought.normalized(),
        turn(json!("plan another call"), None).normalized()
    );
    let empty = turn(json!(""), Some(""));
    assert_eq!(empty.normalized(), plain.normalized());
    assert_eq!(
        serde_json::to_vec(&empty.backend_body(&rid)).unwrap(),
        serde_json::to_vec(&plain.backend_body(&rid)).unwrap()
    );
}

/// O113 extended to tool messages (Main 2026-10-17): tool-eval-bench puts a string `name` on every role:tool
/// message. It is accepted and dropped like on the other roles: the pinned Qwen3.8 template's tool branch reads only
/// `message.content` (`chat_template.jinja` sha256 c3cf9e34...), so it neither forwards to SGLang, reaches the
/// sidecar nor digests. Fails on main 186deb08 (400 `unsupported_parameter` `messages[2].name`).
#[test]
fn o113_tool_message_name_is_accepted_and_dropped() {
    let policy = policy();
    let turn = |name: Option<Value>| {
        let mut result = json!({"role": "tool", "content": "sunny", "tool_call_id": "call_1"});
        if let Some(name) = name {
            result["name"] = name;
        }
        let body = request(json!({"messages": [
            {"role": "user", "content": "weather?"}, echoed_call(), result,
            {"role": "user", "content": "thanks"}
        ]}));
        match policy.validate(&body) {
            Ok(request) => request,
            Err(error) => panic!("{body} -> {error:?}"),
        }
    };
    let rid = Rid::new(1, 2, 3);
    let plain = turn(None);
    for name in [json!("get_weather"), json!(""), Value::Null] {
        let named = turn(Some(name.clone()));
        let body = serde_json::to_value(named.backend_body(&rid)).unwrap();
        assert_eq!(
            body["messages"][2],
            json!({"role": "tool", "content": "sunny", "tool_call_id": "call_1"}),
            "{name}"
        );
        assert_eq!(
            serde_json::to_vec(&named.backend_body(&rid)).unwrap(),
            serde_json::to_vec(&plain.backend_body(&rid)).unwrap(),
            "{name} forwards like its absence"
        );
        assert_eq!(
            named.normalized(),
            plain.normalized(),
            "{name} never digests"
        );
        assert_eq!(named.messages, plain.messages, "{name} never renders");
    }
}

/// C10 (M2.8): grammar-free requests keep their pre-M2.8 SGLang body bytes and digest input. The goldens are
/// SHA-256s of the serialized backend body and of the canonical `normalized()`, captured at the merge base.
#[test]
fn c10_grammar_free_bodies_and_digests_are_unchanged() {
    let tools = r#","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"c":{"type":"string","default":"Z\u00e9"}},"x-max":1E5}}}]"#;
    let call = r#","messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","index":0,"type":"function","function":{"name":"f","arguments":"{\"c\": 1}"}}]},{"role":"tool","tool_call_id":"c1","content":"r"}]"#;
    let cases = [
        raw_body(""),
        raw_body(
            r#","stream":true,"stream_options":{"include_usage":true},"temperature":0,"top_p":0.5,"stop":["x"],"max_tokens":9"#,
        ),
        raw_body(tools),
        raw_body(&format!(r#"{tools},"tool_choice":"auto""#)),
        raw_body(&format!(
            r#"{tools},"tool_choice":"none","parallel_tool_calls":false,"logprobs":true,"top_logprobs":2"#
        )),
        format!(r#"{{"model":"{MODEL}"{call}{tools}}}"#),
    ];
    let sha = |bytes: &[u8]| {
        use sha2::{Digest, Sha256};
        use std::fmt::Write;
        Sha256::digest(bytes)
            .iter()
            .fold(String::new(), |mut out, b| {
                let _ = write!(out, "{b:02x}");
                out
            })
    };
    let rid = Rid::new(3, 7, 0x5e);
    let actual: Vec<(String, String)> = cases
        .iter()
        .map(|text| {
            let request = validate_text(text);
            (
                sha(&serde_json::to_vec(&request.backend_body(&rid)).unwrap()),
                sha(canonical_json(&request.normalized()).as_bytes()),
            )
        })
        .collect();
    let golden = [
        (
            "45ea002030309213e740ac362c26d85bb94553bd8b4a94bd1a58f3065c54723b",
            "bf85ce8e090ad539f85d45b5295275d984dbaa4c21e6a1441199854abb5f45f8",
        ),
        (
            "54ec8bde07fc40cba313f7b145345b14b50e31c02fb78ea0d68e6295ab1f1932",
            "9be2bef565fb6a1ec295ee40233813443620b1c866ecd9e6296d2ed2e02d2c45",
        ),
        (
            "072a69aca2970bf988addbf99540970eb1a0096c479fe1a74af9156564b4f875",
            "7b64e6a1a21261f7dd663e4413430c47875e7fa3506a6c4466f800c79bbcbf9a",
        ),
        (
            "072a69aca2970bf988addbf99540970eb1a0096c479fe1a74af9156564b4f875",
            "7b64e6a1a21261f7dd663e4413430c47875e7fa3506a6c4466f800c79bbcbf9a",
        ),
        (
            "582a6786f9eef12cbe236240093c4d7e99f533ff20b63b97112582eaaf325ddf",
            "8f32f0c7832670d5bde78264fe522c48fb961278e5ecf668d1b3aff92ff3b1ab",
        ),
        (
            "cb16af2db5453be9ffec215eaec655699ee5e31f939c4658ade68ed36e3d9aa3",
            "93e1642dc797ecf3f33bc95048543b614072cc625b9859b2de5788cc35642241",
        ),
    ];
    let golden: Vec<(String, String)> = golden
        .iter()
        .map(|(b, n)| (b.to_string(), n.to_string()))
        .collect();
    assert_eq!(actual, golden);
}

/// A schema using every keyword (and annotation) the SGLang profile advertises: the D5 candidates that passed C4,
/// with numeric bounds in each of C4's passing integer forms, each in a B5 form xgrammar does not drop.
fn full_schema() -> Value {
    json!({
        "title": "t", "description": "d", "type": "object",
        "properties": {
            "a": {"type": "string", "format": "email"},
            "a2": {"type": "string", "pattern": "^[a-z]+$", "title": "p"},
            "a3": {"type": "string", "minLength": 1, "maxLength": 9},
            "b": {"enum": [0, 9], "default": 3},
            "g": {"type": "array", "prefixItems": [
                {"type": "integer", "maximum": 9},
                {"type": "integer", "exclusiveMaximum": 9},
                {"type": "integer", "maximum": 9, "minimum": 0},
                {"type": "integer", "maximum": 9, "exclusiveMinimum": 0},
                {"type": "integer", "exclusiveMaximum": 9, "minimum": 0},
                {"type": "integer", "exclusiveMaximum": 9, "exclusiveMinimum": 0},
            ]},
            "c": {"type": "array", "items": {"$ref": "#/$defs/d"}, "minItems": 1, "maxItems": 3},
            "e": {"type": "array", "prefixItems": [{"enum": ["x", "y"]}, {"const": 1, "description": "one"}]},
            "f": {"anyOf": [{"type": "null"}, {"type": "string", "format": "uri-template"}]},
            "h": {"type": ["string", "null"]},
            "r": {"$ref": "#"},
        },
        "required": ["a", "b"],
        "additionalProperties": false,
        "$defs": {"d": {"type": "string", "pattern": "[\\^$]x\\$"}},
    })
}

fn json_schema_format(schema: Value) -> Value {
    json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": schema}}})
}

fn strict_tool(name: &str, parameters: Value) -> Value {
    json!({"type": "function", "function": {"name": name, "strict": true, "parameters": parameters}})
}

fn refusal(body: &Value) -> (&'static str, Option<String>) {
    let error = policy().validate(body).unwrap_err();
    (error.code, error.param)
}

/// C1 (D3/D4/D5): every D3 shape, the D4 tool choices and `strict: true` tools are accepted on the SGLang profile,
/// and the contract accessors classify them.
#[test]
fn c1_constraint_shapes_are_accepted() {
    let policy = policy();
    let named = json!({"type": "function", "function": {"name": "b"}});
    // (patch, constraint_field, grammar_output, tool_constrained, render_tools)
    type Case = (
        Value,
        Option<&'static str>,
        Option<&'static str>,
        bool,
        bool,
    );
    let cases: Vec<Case> = vec![
        (json!({}), None, None, false, false),
        (
            json!({"response_format": {"type": "text"}}),
            None,
            None,
            false,
            false,
        ),
        (
            json!({"response_format": Value::Null}),
            None,
            None,
            false,
            false,
        ),
        (
            json!({"response_format": {"type": "json_object"}}),
            Some("response_format"),
            Some("json_schema"),
            false,
            false,
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "A-z_9", "description": "d", "strict": true, "schema": full_schema()}}}),
            Some("response_format"),
            Some("json_schema"),
            false,
            false,
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "x".repeat(64), "strict": false, "schema": {}}}}),
            Some("response_format"),
            Some("json_schema"),
            false,
            false,
        ),
        (
            json!({"regex": "[0-9]+"}),
            Some("regex"),
            Some("regex"),
            false,
            false,
        ),
        (
            json!({"ebnf": "root ::= \"a\""}),
            Some("ebnf"),
            Some("ebnf"),
            false,
            false,
        ),
        (
            json!({"regex": Value::Null, "ebnf": Value::Null}),
            None,
            None,
            false,
            false,
        ),
        (
            json!({"tool_choice": "required", "tools": [tool("a")]}),
            None,
            Some("tool_calls_constrained"),
            true,
            true,
        ),
        (
            json!({"tool_choice": named, "tools": [tool("a"), tool("b")]}),
            None,
            Some("tool_calls_constrained"),
            true,
            true,
        ),
        (
            json!({"tools": [tool("a"), strict_tool("b", full_schema())]}),
            None,
            Some("tool_calls_constrained"),
            true,
            true,
        ),
        (
            json!({"tool_choice": "none", "tools": [strict_tool("b", json!({"oneOf": []}))]}),
            None,
            None,
            false,
            false,
        ),
        // A non-strict tool constrains nothing: its parameters are not walked and it composes with a constraint.
        (
            json!({"tools": [{"type": "function", "function": {"name": "a", "parameters": {"oneOf": []}}}],
                   "regex": "a"}),
            Some("regex"),
            Some("regex"),
            false,
            true,
        ),
        (
            json!({"tools": [tool("a")], "tool_choice": "auto", "response_format": {"type": "json_object"},
                   "temperature": 0}),
            Some("response_format"),
            Some("json_schema"),
            false,
            true,
        ),
    ];
    for (patch, field, output, constrained, render) in cases {
        let accepted = policy
            .validate(&request(patch.clone()))
            .unwrap_or_else(|e| panic!("{patch} -> {e:?}"));
        assert_eq!(
            (
                accepted.constraint_field(),
                accepted.grammar_output(),
                accepted.tool_constrained(),
                accepted.render_tools()
            ),
            (field, output, constrained, render),
            "{patch}"
        );
    }
    let accepted = policy
        .validate(&request(
            json!({"tool_choice": named, "tools": [tool("a"), tool("b")]}),
        ))
        .unwrap();
    assert_eq!(accepted.tool_choice, ToolChoice::Named("b".to_owned()));
    let accepted = policy
        .validate(&request(
            json!({"tool_choice": "required", "tools": [tool("a")]}),
        ))
        .unwrap();
    assert_eq!(accepted.tool_choice, ToolChoice::Required);
}

/// C1: each refusal names its field path.
#[test]
fn c1_constraint_refusals_name_their_path() {
    let schema_param = "response_format.json_schema.schema";
    let params = "tools[0].function.parameters";
    let named = |name: &str| json!({"type": "function", "function": {"name": name}});
    let rejected: Vec<(Value, &str, &str)> = vec![
        // At most one constraint, named by the precedence response_format < regex < ebnf.
        (
            json!({"response_format": {"type": "json_object"}, "regex": "a"}),
            "invalid_value",
            "regex",
        ),
        (
            json!({"regex": "a", "ebnf": "root ::= \"a\""}),
            "invalid_value",
            "ebnf",
        ),
        (
            json!({"response_format": {"type": "json_object"}, "ebnf": "e"}),
            "invalid_value",
            "ebnf",
        ),
        (
            json!({"response_format": {"type": "json_object"}, "regex": "a", "ebnf": "e"}),
            "invalid_value",
            "ebnf",
        ),
        // D3 shapes.
        (
            json!({"response_format": "json"}),
            "invalid_value",
            "response_format",
        ),
        (
            json!({"response_format": {}}),
            "missing_required_parameter",
            "response_format.type",
        ),
        (
            json!({"response_format": {"type": 1}}),
            "invalid_value",
            "response_format.type",
        ),
        (
            json!({"response_format": {"type": "regex"}}),
            "invalid_value",
            "response_format.type",
        ),
        (
            json!({"response_format": {"type": "ebnf"}}),
            "invalid_value",
            "response_format.type",
        ),
        (
            json!({"response_format": {"type": "structural_tag", "structures": [], "triggers": []}}),
            "unsupported_parameter",
            "response_format.type",
        ),
        (
            json!({"response_format": {"type": "text", "x": 1}}),
            "unsupported_parameter",
            "response_format.x",
        ),
        (
            json!({"response_format": {"type": "json_object", "schema": {}}}),
            "unsupported_parameter",
            "response_format.schema",
        ),
        // N3: SGLang's unwrapped shortcut is refused.
        (
            json!({"response_format": {"type": "json_schema", "schema": {"type": "object"}}}),
            "invalid_value",
            "response_format.json_schema",
        ),
        (
            json!({"response_format": {"type": "json_schema"}}),
            "missing_required_parameter",
            "response_format.json_schema",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": []}}),
            "invalid_value",
            "response_format.json_schema",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n"}}}),
            "missing_required_parameter",
            schema_param,
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema": {}}}}),
            "missing_required_parameter",
            "response_format.json_schema.name",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "a b", "schema": {}}}}),
            "invalid_value",
            "response_format.json_schema.name",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "x".repeat(65), "schema": {}}}}),
            "invalid_value",
            "response_format.json_schema.name",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "n", "description": 1, "schema": {}}}}),
            "invalid_value",
            "response_format.json_schema.description",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "n", "strict": "yes", "schema": {}}}}),
            "invalid_value",
            "response_format.json_schema.strict",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "n", "schema": {}, "x": 1}}}),
            "unsupported_parameter",
            "response_format.json_schema.x",
        ),
        (json_schema_format(json!([])), "invalid_value", schema_param),
        (
            json_schema_format(json!(true)),
            "invalid_value",
            schema_param,
        ),
        // Sizes.
        (json!({"regex": ""}), "invalid_value", "regex"),
        (json!({"regex": 7}), "invalid_value", "regex"),
        (
            json!({"regex": "a".repeat(16 * 1024 + 1)}),
            "invalid_value",
            "regex",
        ),
        (json!({"ebnf": ""}), "invalid_value", "ebnf"),
        (json!({"ebnf": ["root"]}), "invalid_value", "ebnf"),
        (
            json!({"ebnf": "a".repeat(16 * 1024 + 1)}),
            "invalid_value",
            "ebnf",
        ),
        (
            json_schema_format(json!({"type": "string", "description": "x".repeat(64 * 1024)})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(nested(33)),
            "invalid_value",
            schema_param,
        ),
        // D5 walker.
        (
            json_schema_format(json!({"type": "object", "x-a": 1})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"$schema": "s"})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"oneOf": [{"type": "string"}]})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "object", "properties": {"a": {"not": {"type": "string"}}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "format": "regex"})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "format": 1})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "pattern": "a^b"})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "pattern": "a$b"})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "pattern": "(^a)"})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"$defs": {"d": {"type": "string", "pattern": "x^"}}})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"items": {"anyOf": [{"oneOf": []}]}})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "array", "prefixItems": [{"type": "string", "format": "phone"}]}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            // C4: xgrammar 0.2.1 does not enforce `uri-reference`.
            json_schema_format(
                json!({"type": "object", "properties": {"u": {"type": "string",
                "format": "uri-reference"}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            // C4: a lone lower bound leaks on xgrammar 0.2.1.
            json_schema_format(
                json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            // C4: xgrammar drops number bounds, so the SGLang profile refuses them (B7: only a native profile's
            // grammar sidecar compiles a number range).
            json_schema_format(
                json!({"type": "object", "properties": {"n": {"type": "number", "maximum": 3}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"$defs": {"n": {"maximum": 3}}})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "object", "properties": {"n": {"type": ["integer"], "maximum": 3}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            // Untested forms: two upper or two lower bounds.
            json_schema_format(
                json!({"type": "object", "properties": {"n": {"type": "integer", "maximum": 3,
                "exclusiveMaximum": 4}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "array", "items": {"type": "integer", "maximum": 3, "minimum": 0,
                "exclusiveMinimum": -1}}),
            ),
            "invalid_value",
            schema_param,
        ),
        // C4: `allOf` failed enforcement, so the SGLang profile does not advertise it.
        (
            json_schema_format(json!({"anyOf": [{"allOf": [{"type": "string"}]}]})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "object", "properties": []})),
            "invalid_value",
            schema_param,
        ),
        (
            json!({"tools": [strict_tool("a", json!({"oneOf": []}))]}),
            "invalid_value",
            params,
        ),
        (
            json!({"tools": [strict_tool("a", json!({"type": "object",
                "properties": {"c": {"type": "string", "pattern": "a^"}}}))],
                   "tool_choice": "required"}),
            "invalid_value",
            params,
        ),
        (
            json!({"tools": [tool("a"), strict_tool("b", json!({"$defs": {"x": {"type": "string",
                "format": "x"}}}))]}),
            "invalid_value",
            "tools[1].function.parameters",
        ),
        // B5 R1: `$ref`, `const`, `enum` and `anyOf` dispatch before `type` and drop their siblings.
        (
            json_schema_format(
                json!({"$defs": {"d": {"type": "object"}}, "$ref": "#/$defs/d",
                "required": []}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "object", "properties": {"a": {"type": "string"}},
                "anyOf": [{"type": "object"}]}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "integer", "maximum": 5, "anyOf": [{"type": "integer"}]}),
            ),
            "invalid_value",
            schema_param,
        ),
        // O113: `type` beside `enum` only when every value is of that type; xgrammar drops the `type`.
        (
            json_schema_format(
                json!({"type": "object", "properties": {"a": {"type": "string", "enum": ["a", 1]}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "integer", "enum": [1, 1.5]})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": ["string", "null"], "enum": ["a", 1]})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "strng", "enum": ["a"]})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": [], "enum": []})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "enum": ["a"], "minLength": 1})),
            "invalid_value",
            schema_param,
        ),
        (
            json!({"tools": [strict_tool("a", json!({"type": "object",
                "properties": {"u": {"type": "string", "enum": ["c", 1]}}}))]}),
            "invalid_value",
            params,
        ),
        (
            json_schema_format(
                json!({"type": "object", "properties": {"a": {"const": 1, "enum": [1]}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "object", "properties": {"a": {"$ref": "#", "$defs": {}}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        // B5 R2: type-specific keywords need that exact `type`.
        (
            json_schema_format(json!({"minLength": 1})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "integer", "pattern": "^a$"})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "object", "items": {"type": "string"}})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"properties": {"a": {"type": "string"}}})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "required": []})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "array", "additionalProperties": false})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": ["string", "null"], "maxLength": 3})),
            "invalid_value",
            schema_param,
        ),
        // B5 R3: one string form per schema.
        (
            json_schema_format(json!({"type": "string", "format": "email", "maxLength": 9})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "pattern": "^a$", "minLength": 1})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "string", "format": "email", "pattern": "^a$"})),
            "invalid_value",
            schema_param,
        ),
        // B5 R4: local `$ref` into the root `$defs` only.
        (
            json_schema_format(
                json!({"type": "object", "properties": {"a": {"$ref": "https://x/a.json"}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "object", "default": {"type": "string"},
                "properties": {"a": {"$ref": "#/default"}}})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"$defs": {"d": {"type": "string"}}, "type": "object",
                "properties": {"a": {"$ref": "#/$defs/e"}}})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"$defs": {"d": {"type": "object", "properties": {"x": {}}}},
                "type": "object", "properties": {"a": {"$ref": "#/$defs/d/properties/x"}}}),
            ),
            "invalid_value",
            schema_param,
        ),
        // Z2a round 2: xgrammar drops empty pointer segments, so "#/$defs/" is the whole $defs map.
        (
            json_schema_format(json!({"$defs": {"": {"type": "integer"}}, "type": "object",
                "properties": {"a": {"$ref": "#/$defs/"}}, "required": ["a"]})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "object", "properties": {"a": {"$ref": 1}}})),
            "invalid_value",
            schema_param,
        ),
        // B5 R5: `required` names declared properties.
        (
            json_schema_format(json!({"type": "object", "required": ["a"]})),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(
                json!({"type": "object", "properties": {"a": {}}, "required": ["a", "b"]}),
            ),
            "invalid_value",
            schema_param,
        ),
        (
            json_schema_format(json!({"type": "object", "properties": {"a": {}}, "required": [1]})),
            "invalid_value",
            schema_param,
        ),
        (
            json!({"tools": [strict_tool("a", json!({"type": "object",
                "properties": {"q": {"type": "string"}}, "required": ["r"]}))]}),
            "invalid_value",
            params,
        ),
        (
            json!({"tools": [strict_tool("a", json!({"type": "object",
                "properties": {"q": {"type": "string", "format": "email", "minLength": 3}}}))]}),
            "invalid_value",
            params,
        ),
        // D4.
        (
            json!({"tool_choice": "required"}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tool_choice": named("a")}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tool_choice": named("c"), "tools": [tool("a")]}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tool_choice": {"type": "function"}, "tools": [tool("a")]}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tool_choice": {"type": "tool", "function": {"name": "a"}}, "tools": [tool("a")]}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tool_choice": {"type": "function", "function": {"name": 1}}, "tools": [tool("a")]}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tool_choice": "any", "tools": [tool("a")]}),
            "invalid_value",
            "tool_choice",
        ),
        // N1: a tool constraint plus a response_format/regex/ebnf constraint.
        (
            json!({"tool_choice": "required", "tools": [tool("a")], "response_format": {"type": "json_object"}}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tools": [strict_tool("a", json!({"type": "object"}))], "regex": "a"}),
            "invalid_value",
            "tool_choice",
        ),
        (
            json!({"tool_choice": named("a"), "tools": [tool("a")], "ebnf": "e"}),
            "invalid_value",
            "tool_choice",
        ),
    ];
    for (patch, code, param) in rejected {
        assert_eq!(
            refusal(&request(patch.clone())),
            (code, Some(param.to_owned())),
            "{patch}"
        );
    }

    // The second constraint is named by precedence, not by its position in the body.
    for text in [
        raw_body(r#","ebnf":"e","regex":"a""#),
        raw_body(r#","regex":"a","ebnf":"e""#),
    ] {
        let value: Value = serde_json::from_str(&text).unwrap();
        let error = policy()
            .validate_bytes(&value, text.as_bytes())
            .unwrap_err();
        assert_eq!(
            (error.code, error.param.as_deref()),
            ("invalid_value", Some("ebnf"))
        );
    }
    for text in [
        raw_body(r#","regex":"a","response_format":{"type":"json_object"}"#),
        raw_body(r#","response_format":{"type":"json_object"},"regex":"a""#),
    ] {
        let value: Value = serde_json::from_str(&text).unwrap();
        let error = policy()
            .validate_bytes(&value, text.as_bytes())
            .unwrap_err();
        assert_eq!(
            (error.code, error.param.as_deref()),
            ("invalid_value", Some("regex"))
        );
    }
}

/// B7 [REVISED: owner 2026-10-18 number bounds]: the native profile serves a `type: "number"` schema with an
/// inclusive `minimum` and `maximum` (the grammar sidecar compiles the exact range; TC-64 of the FP4-Local
/// tool-eval-bench is 0..10) in every position the walker visits. Every other number bound form, a number bound in
/// a tool's parameters and any number bound on the SGLang profile (whose xgrammar drops them) stay 400
/// invalid_value; integer bounds are unchanged.
#[test]
fn b7_native_number_ranges_are_served_other_number_bounds_refused() {
    let native = native_policy();
    let tc64 = json!({"type": "number", "minimum": 0, "maximum": 10});
    let served = [
        tc64.clone(),
        json!({"type": "number", "minimum": 0.5, "maximum": 2.25}),
        json!({"type": "number", "minimum": -7.25, "maximum": -1.5}),
        json!({"type": "number", "minimum": -3, "maximum": 4}),
        json!({"type": "number", "minimum": 1.5, "maximum": 1.5}),
        json!({"type": "number", "minimum": 0.0, "maximum": 1e1, "title": "t", "description": "d",
               "default": 5}),
        json!({"type": "number", "minimum": -1e19, "maximum": 1e-20}),
        json!({"type": "number", "minimum": 0, "maximum": 12_345_678_901_234_567_890_u64}),
        json!({"type": "object", "properties": {"x": tc64.clone()}, "required": ["x"]}),
        json!({"type": "array", "items": tc64.clone()}),
        json!({"$defs": {"r": tc64.clone()}, "type": "object",
               "properties": {"a": {"$ref": "#/$defs/r"}}}),
        json!({"anyOf": [tc64.clone(), {"type": "string"}]}),
    ];
    for schema in served {
        for temperature in [json!(0), json!(0.7)] {
            let mut body = request(json_schema_format(schema.clone()));
            body["temperature"] = temperature;
            assert!(native.validate(&body).is_ok(), "{schema}");
        }
    }
    let schema_param = "response_format.json_schema.schema";
    let refused = [
        json!({"type": "number", "maximum": 10}),
        json!({"type": "number", "minimum": 0}),
        json!({"type": "number", "exclusiveMinimum": 0, "maximum": 10}),
        json!({"type": "number", "minimum": 0, "exclusiveMaximum": 10}),
        json!({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 10}),
        json!({"type": "number", "minimum": 0, "maximum": 10, "exclusiveMaximum": 11}),
        json!({"type": "number", "minimum": 10, "maximum": 0}),
        json!({"type": "number", "minimum": -1.5, "maximum": -7.25}),
        json!({"type": "number", "minimum": 0, "maximum": 1e21}),
        json!({"type": "number", "minimum": 1e-21, "maximum": 1}),
        json!({"type": "number", "minimum": "0", "maximum": 10}),
        json!({"type": ["number"], "minimum": 0, "maximum": 10}),
        json!({"minimum": 0, "maximum": 10}),
        json!({"type": "object", "properties": {"x": {"type": "number", "maximum": 10}}}),
        json!({"$defs": {"r": {"type": "number", "minimum": 0}}, "type": "object",
               "properties": {"a": {"$ref": "#/$defs/r"}}}),
        json!({"type": "integer", "minimum": 3}),
    ];
    for schema in refused {
        let error = native
            .validate(&request(json_schema_format(schema.clone())))
            .unwrap_err();
        assert_eq!(
            (error.status, error.code, error.param.as_deref()),
            (400, "invalid_value", Some(schema_param)),
            "{schema}"
        );
        assert!(
            error.message.contains("bounds"),
            "{schema}: {}",
            error.message
        );
    }
    let error = native
        .validate(&request(json!({
            "tools": [strict_tool("f", json!({"type": "object", "properties": {"x": tc64.clone()}}))],
            "tool_choice": "required"})))
        .unwrap_err();
    assert_eq!(
        (error.code, error.param.as_deref()),
        ("invalid_value", Some("tools[0].function.parameters"))
    );
    assert_eq!(
        refusal(&request(json_schema_format(tc64))),
        ("invalid_value", Some(schema_param.to_owned()))
    );
}

/// C1 (D10): a constraint needs the profile's `decode` combination for its grammar output and sampler mode;
/// the native profile (`-v8`, M2.8 D10) has greedy and sampled ones for each.
#[test]
fn c1_constraints_are_gated_by_the_profile() {
    let constraints = [
        (
            json!({"response_format": {"type": "json_object"}}),
            "json_schema",
            "response_format",
        ),
        (
            json_schema_format(json!({"type": "string"})),
            "json_schema",
            "response_format",
        ),
        (json!({"regex": "a"}), "regex", "regex"),
        (json!({"ebnf": "e"}), "ebnf", "ebnf"),
        (
            json!({"tool_choice": "required", "tools": [tool("a")]}),
            "tool_calls_constrained",
            "tool_choice",
        ),
        (
            json!({"tools": [strict_tool("a", json!({"type": "object"}))]}),
            "tool_calls_constrained",
            "tool_choice",
        ),
    ];
    for (patch, output, param) in &constraints {
        let mut config = live_config();
        config.deployments[0]
            .lock
            .as_mut()
            .unwrap()
            .profile
            .validated_combinations
            .retain(|c| !(c.output == *output && c.sampler_mode == "sampled"));
        let greedy_only = ChatPolicy::from_config(&config, 0).unwrap();
        let mut greedy = request(patch.clone());
        greedy["temperature"] = json!(0);
        greedy_only
            .validate(&greedy)
            .unwrap_or_else(|e| panic!("{patch} -> {e:?}"));
        let error = greedy_only.validate(&request(patch.clone())).unwrap_err();
        assert_eq!(
            (error.code, error.param.as_deref()),
            ("unsupported_parameter", Some(*param)),
            "{patch}"
        );

        let native = native_policy();
        let mut body = request(patch.clone());
        native
            .validate(&body)
            .unwrap_or_else(|e| panic!("native {patch} -> {e:?}"));
        body["temperature"] = json!(0);
        native
            .validate(&body)
            .unwrap_or_else(|e| panic!("native greedy {patch} -> {e:?}"));
    }

    // The walker allows exactly the profile's keywords.
    let mut config = live_config();
    let profile = &mut config.deployments[0].lock.as_mut().unwrap().profile;
    profile.json_schema_keywords.retain(|k| k != "pattern");
    let narrow = ChatPolicy::from_config(&config, 0).unwrap();
    narrow
        .validate(&request(json_schema_format(json!({"type": "string"}))))
        .unwrap();
    let error = narrow
        .validate(&request(json_schema_format(json!({"pattern": "a"}))))
        .unwrap_err();
    assert_eq!(
        (error.code, error.param.as_deref()),
        ("invalid_value", Some("response_format.json_schema.schema"))
    );
}

/// C2 (D6): constraint fields digest as their raw bytes, `{type:"text"}` equals absent, `tool_choice` is
/// canonical, and the backend body forwards the client's exact bytes.
#[test]
fn c2_constraints_digest_their_bytes_and_forward_verbatim() {
    let absent = validate_text(&raw_body(""));
    let text = validate_text(&raw_body(r#","response_format":{"type":"text"}"#));
    assert_eq!(absent.normalized(), text.normalized());
    let rid = Rid::new(1, 2, 3);
    assert_eq!(
        serde_json::to_vec(&absent.backend_body(&rid)).unwrap(),
        serde_json::to_vec(&text.backend_body(&rid)).unwrap()
    );

    let schema_a = r#","response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object","properties":{"a":{"type":"number","default":1E5,"description":"\u00e9"}}}}}"#;
    let schema_b = r#","response_format":{"json_schema":{"name":"n","schema":{"properties":{"a":{"type":"number","default":1E5,"description":"\u00e9"}},"type":"object"}},"type":"json_schema"}"#;
    let a = validate_text(&raw_body(schema_a));
    let b = validate_text(&raw_body(schema_b));
    assert_ne!(a.normalized(), b.normalized(), "key order must count");
    let regex_a = validate_text(&raw_body(r#","regex":"\u0061+""#));
    let regex_b = validate_text(&raw_body(r#","regex":"a+""#));
    assert_ne!(
        regex_a.normalized(),
        regex_b.normalized(),
        "regex bytes count"
    );
    assert_ne!(regex_b.normalized(), absent.normalized());
    let ebnf = validate_text(&raw_body(r#","ebnf":"a+""#));
    assert_ne!(ebnf.normalized(), regex_b.normalized(), "the field counts");

    let tools = r#","tools":[{"type":"function","function":{"name":"f"}}]"#;
    let auto = validate_text(&raw_body(tools));
    let required = validate_text(&raw_body(&format!(r#"{tools},"tool_choice":"required""#)));
    let named_a = validate_text(&raw_body(&format!(
        r#"{tools},"tool_choice":{{"type":"function","function":{{"name":"f"}}}}"#
    )));
    let named_b = validate_text(&raw_body(&format!(
        r#"{tools},"tool_choice":{{"function":{{"name":"f"}},"type":"function"}}"#
    )));
    assert_ne!(auto.normalized(), required.normalized());
    assert_ne!(auto.normalized(), named_a.normalized());
    assert_ne!(required.normalized(), named_a.normalized());
    assert_eq!(
        named_a.normalized(),
        named_b.normalized(),
        "tool_choice is canonical"
    );

    // Forwarded verbatim: each subtree's bytes appear unchanged in the body.
    for (extra, subtree) in [
        (schema_b, &schema_b[r#","response_format":"#.len()..]),
        (r#","regex":"\u0061+""#, r#""\u0061+""#),
        (
            r#","ebnf":"root ::= \"\u00e9\"""#,
            r#""root ::= \"\u00e9\"""#,
        ),
    ] {
        let request = validate_text(&raw_body(extra));
        let body =
            String::from_utf8(serde_json::to_vec(&request.backend_body(&rid)).unwrap()).unwrap();
        assert!(body.contains(subtree), "{body}");
    }
    let choice = r#"{"function":{"name":"f"},"type":"function"}"#;
    let request = validate_text(&raw_body(&format!(r#"{tools},"tool_choice":{choice}"#)));
    let body = String::from_utf8(serde_json::to_vec(&request.backend_body(&rid)).unwrap()).unwrap();
    assert!(
        body.contains(&format!(r#""tool_choice":{choice}"#)),
        "{body}"
    );
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        value["tool_choice"],
        serde_json::from_str::<Value>(choice).unwrap()
    );
}

/// B5: the schema forms xgrammar 0.2.1 compiles without dropping a keyword pass the walker, as a response format
/// and as a strict tool's parameters.
#[test]
fn b5_enforced_schema_forms_are_accepted() {
    let policy = policy();
    let schemas = [
        full_schema(),
        json!({"$defs": {"d": {"type": "string"}}, "$ref": "#/$defs/d", "title": "t"}),
        json!({"enum": [1, "a"], "description": "d", "default": 1}),
        json!({"const": {"a": 1}, "title": "c"}),
        json!({"anyOf": [{"type": "integer", "maximum": 3}, {"$ref": "#"}], "default": 0}),
        json!({"type": ["string", "integer", "null"], "title": "t"}),
        json!({"type": "string", "minLength": 1}),
        json!({"type": "string", "maxLength": 9, "minLength": 2}),
        json!({"type": "object", "properties": {"a": {}}, "required": ["a"]}),
        json!({"type": "object", "properties": {}, "required": []}),
        json!({"type": "array", "items": false, "prefixItems": [{"type": "boolean"}]}),
        json!({}),
        // O113: `type` beside `enum`, every value of the type.
        json!({"type": "string", "enum": ["c", "f"], "description": "unit"}),
        json!({"type": "integer", "enum": [1, 2, 3.0]}),
        json!({"type": ["string", "null"], "enum": ["a", null]}),
        json!({"type": "object", "properties": {"u": {"type": "string", "enum": ["c", "f"]}},
            "required": ["u"]}),
        json!({"type": "number", "enum": [1, 2.5]}),
        json!({"type": "boolean", "enum": [true]}),
    ];
    for schema in schemas {
        for patch in [
            json_schema_format(schema.clone()),
            json!({"tools": [strict_tool("a", schema.clone())]}),
        ] {
            if let Err(error) = policy.validate(&request(patch.clone())) {
                panic!("{patch} -> {error:?}");
            }
        }
    }
}

/// O113 (owner 2026-10-17), tied to the pinned-image oracle (`grammar_keywords.json` `combination_variants`): every
/// `B:type+enum:*` form compiles to exactly the enum's grammar (`same_grammar_without_type`), is enforced, and the
/// walker accepts it as a response format and as a strict tool's parameters; every `A:type+enum:*` form (a value
/// contradicting the `type`, which xgrammar drops) is refused. Fails on main 49397403 (no such variants; the
/// walker refused every `type` beside `enum`).
#[test]
fn o113_type_beside_enum_follows_the_oracle() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/native/grammar_keywords.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let policy = policy();
    let mut seen = [0usize; 2];
    for v in doc["combination_variants"].as_array().unwrap() {
        let name = v["name"].as_str().unwrap();
        if !name.contains(":type+enum:") {
            continue;
        }
        let enforced = v["group"] == "B";
        seen[usize::from(enforced)] += 1;
        if enforced {
            assert_eq!(v["pass"], true, "{name}");
            assert_eq!(v["same_grammar_without_type"], true, "{name}");
        }
        for (patch, param) in [
            (
                json_schema_format(v["schema"].clone()),
                "response_format.json_schema.schema",
            ),
            (
                json!({"tools": [strict_tool("a", v["schema"].clone())]}),
                "tools[0].function.parameters",
            ),
        ] {
            let result = policy.validate(&request(patch.clone()));
            if enforced {
                assert!(result.is_ok(), "{name}: {patch} -> {:?}", result.err());
            } else {
                let error = result
                    .err()
                    .unwrap_or_else(|| panic!("{name}: {patch} accepted"));
                assert_eq!(
                    (error.code, error.param.as_deref()),
                    ("invalid_value", Some(param)),
                    "{name}"
                );
            }
        }
    }
    assert!(seen[0] >= 2 && seen[1] >= 4, "{seen:?}");
}

/// The seven M2.7 request sampling fields (D8), in `SAMPLING_PARAMS` order.
const SAMPLING_FIELDS: [&str; 7] = [
    "top_k",
    "min_p",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "logit_bias",
];

/// The five the SGLang profile declares (D9): `seed` and `min_p` stay refused there.
const SGLANG_SAMPLING_FIELDS: [&str; 5] = [
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "logit_bias",
];

/// The shipped native profile (D11): sampled decode rows, all seven `sampling_params` and `max_top_logprobs: 20`.
fn native_sampling_policy() -> ChatPolicy {
    let config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0.toml"),
    )
    .unwrap();
    let profile = &config.deployments[0].lock.as_ref().unwrap().profile;
    assert_eq!(
        profile.sampling_params,
        Some(SAMPLING_FIELDS.map(str::to_owned).to_vec())
    );
    assert_eq!(profile.max_top_logprobs, Some(20));
    let policy = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(policy.sampled && policy.tools_sampled);
    policy
}

/// The live SGLang policy with `sampling_params` set to `names` (`None` = absent).
fn sglang_policy_sampling(names: Option<&[&str]>) -> ChatPolicy {
    let mut config = live_config();
    config.deployments[0]
        .lock
        .as_mut()
        .unwrap()
        .profile
        .sampling_params = names.map(|n| n.iter().map(|s| s.to_string()).collect());
    ChatPolicy::from_config(&config, 0).unwrap()
}

/// Valid values of each sampling field, boundaries included (F7 + D8; `logit_bias` values per Q6).
fn valid_sampling_values() -> Vec<(&'static str, Value)> {
    vec![
        ("top_k", json!(-1)),
        ("top_k", json!(1)),
        ("top_k", json!(1_000_000)),
        ("min_p", json!(0)),
        ("min_p", json!(0.05)),
        ("min_p", json!(1)),
        ("seed", json!(0)),
        ("seed", json!(i64::MAX)),
        ("frequency_penalty", json!(-2)),
        ("frequency_penalty", json!(2.0)),
        ("presence_penalty", json!(-2.0)),
        ("presence_penalty", json!(0.5)),
        ("repetition_penalty", json!(1e-6)),
        ("repetition_penalty", json!(2)),
        ("logit_bias", json!({})),
        ("logit_bias", json!({"0": -100, "151643": 100, "7": 0.5})),
    ]
}

/// C1 (D8): the seven fields pass on a native profile that declares them, at their range boundaries; `null` is
/// absent.
#[test]
fn c1_sampling_fields_are_accepted_when_declared() {
    let native = native_sampling_policy();
    for (field, value) in valid_sampling_values() {
        for temperature in [json!(0), json!(0.7)] {
            let body = request(json!({field: value.clone(), "temperature": temperature}));
            if let Err(e) = native.validate(&body) {
                panic!("{body}: {e:?}");
            }
        }
    }
    for field in SAMPLING_FIELDS {
        native
            .validate(&request(json!({field: Value::Null, "temperature": 0.7})))
            .unwrap();
    }
    let all = request(
        json!({"top_k": 20, "min_p": 0.1, "seed": 42, "frequency_penalty": 0.5, "temperature": 0.7,
        "presence_penalty": -0.5, "repetition_penalty": 1.1, "logit_bias": {"5": -100}}),
    );
    let validated = native.validate(&all).unwrap();
    assert_eq!(validated.sampler, SamplerMode::Sampled);
}

/// C1 (D8, F7, Q6, C7): out-of-range, non-integer and non-number values of each field are 400 `invalid_value`
/// naming the field.
#[test]
fn c1_sampling_field_ranges_are_invalid_value() {
    let native = native_sampling_policy();
    let invalid = [
        ("top_k", json!(0)),
        ("top_k", json!(-2)),
        ("top_k", json!(1.5)),
        ("top_k", json!(2.0)),
        ("top_k", json!("5")),
        ("min_p", json!(-0.01)),
        ("min_p", json!(1.01)),
        ("min_p", json!("0.1")),
        ("seed", json!(-1)),
        ("seed", json!(9_223_372_036_854_775_808u64)),
        ("seed", json!(u64::MAX)),
        ("seed", json!(1.5)),
        ("seed", json!(3.0)),
        ("seed", json!("1")),
        ("frequency_penalty", json!(2.01)),
        ("frequency_penalty", json!(-2.01)),
        ("frequency_penalty", json!(true)),
        ("presence_penalty", json!(2.5)),
        ("presence_penalty", json!(-3)),
        ("presence_penalty", json!([0])),
        ("repetition_penalty", json!(0)),
        ("repetition_penalty", json!(-1)),
        ("repetition_penalty", json!(2.01)),
        ("repetition_penalty", json!("1")),
        ("logit_bias", json!([1])),
        ("logit_bias", json!("x")),
        ("logit_bias", json!({"a": 1})),
        ("logit_bias", json!({"-1": 1})),
        ("logit_bias", json!({"01": 1})),
        ("logit_bias", json!({"+1": 1})),
        ("logit_bias", json!({" 1": 1})),
        ("logit_bias", json!({"": 1})),
        ("logit_bias", json!({"4294967296": 1})),
        ("logit_bias", json!({"1": 100.5})),
        ("logit_bias", json!({"1": -101})),
        ("logit_bias", json!({"1": "1"})),
        ("logit_bias", json!({"1": null})),
    ];
    for (field, value) in invalid {
        let body = request(json!({field: value, "temperature": 0.7}));
        let error = native.validate(&body).unwrap_err();
        assert_eq!(
            (error.status, error.code, error.param.as_deref()),
            (400, "invalid_value", Some(field)),
            "{body}"
        );
    }
}

/// C1: native no longer refuses `top_p` (D8); `max_top_logprobs: 20` bounds `top_logprobs`; greedy `n: 2` passes and
/// sampled `n: 2` stays 400.
#[test]
fn c1_native_top_p_logprobs_and_n() {
    let stock = native_policy();
    let greedy = stock
        .validate(&request(json!({"temperature": 0, "top_p": 0.9})))
        .unwrap();
    assert_eq!(greedy.top_p, Some(0.9));
    let native = native_sampling_policy();
    native
        .validate(&request(json!({"temperature": 0.7, "top_p": 0.9})))
        .unwrap();
    let top = native
        .validate(&request(
            json!({"temperature": 0.7, "logprobs": true, "top_logprobs": 20}),
        ))
        .unwrap();
    assert_eq!((top.logprobs, top.top_logprobs), (true, Some(20)));
    let error = native
        .validate(&request(
            json!({"temperature": 0.7, "logprobs": true, "top_logprobs": 21}),
        ))
        .unwrap_err();
    assert_eq!(
        (error.code, error.param.as_deref()),
        ("invalid_value", Some("top_logprobs"))
    );
    // The shipped profile's 16 rows (4 before m23 design note P8) bound `n` (m26 D11), sampled n included (ledger CR4: per-row sampling and
    // distinct per-choice seeds).
    for policy in [&stock, &native] {
        for temperature in [json!(0), json!(0.7)] {
            let validated = policy
                .validate(&request(json!({"temperature": temperature, "n": 2})))
                .unwrap();
            assert_eq!(validated.n, 2);
            let error = policy
                .validate(&request(json!({"temperature": temperature, "n": 17})))
                .unwrap_err();
            assert_eq!(
                (error.status, error.code, error.param.as_deref()),
                (400, "unsupported_parameter", Some("n"))
            );
        }
    }
    // D10 (Q4 (a)): without a temperature native passes, `null` included; the field stays absent and native samples
    // it with the model's `generation_config.json` defaults (C8).
    for patch in [json!({}), json!({"temperature": null})] {
        let validated = native
            .validate(&request(patch.clone()))
            .unwrap_or_else(|e| panic!("{patch}: {e:?}"));
        assert_eq!(validated.temperature, None, "{patch}");
    }
}

/// C1 (D9): on the SGLang profile the five declared fields pass; `seed` and `min_p` are 400
/// `unsupported_parameter` naming the field. A profile without `sampling_params` refuses each of the seven by
/// name, `null` included (the key is refused as before M2.7).
#[test]
fn c1_sampling_fields_are_gated_by_the_profile() {
    let sglang = sglang_policy_sampling(Some(&SGLANG_SAMPLING_FIELDS));
    for (field, value) in valid_sampling_values() {
        let body = request(json!({field: value}));
        let verdict = sglang.validate(&body);
        if SGLANG_SAMPLING_FIELDS.contains(&field) {
            if let Err(e) = verdict {
                panic!("{body}: {e:?}");
            }
        } else {
            let error = verdict.unwrap_err();
            assert_eq!(
                (error.status, error.code, error.param.as_deref()),
                (400, "unsupported_parameter", Some(field)),
                "{body}"
            );
            assert!(error.message.contains(field), "{}", error.message);
        }
    }
    for undeclared in [sglang_policy_sampling(None), native_greedy_policy()] {
        for field in SAMPLING_FIELDS {
            for value in [json!(1), Value::Null] {
                let mut body = request(json!({"temperature": 0}));
                body[field] = value;
                let error = undeclared.validate(&body).unwrap_err();
                assert_eq!(
                    (error.status, error.code, error.param.as_deref()),
                    (400, "unsupported_parameter", Some(field)),
                    "{body}"
                );
            }
        }
    }
}

/// B5: every G1-G9 request of the oracle fixture passes the walker with the SGLang profile's keywords, its tools
/// all made strict so their parameters are walked too.
#[test]
fn b5_grammar_cases_pass_the_walker() {
    let policy = policy();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/native/grammar_cases.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases: Vec<&Value> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| {
            case["name"]
                .as_str()
                .is_some_and(|name| name.starts_with('G'))
        })
        .collect();
    assert_eq!(cases.len(), 9);
    for case in cases {
        let mut patch = json!({});
        for (key, text) in case["request"].as_object().unwrap() {
            let mut value: Value = serde_json::from_str(text.as_str().unwrap()).unwrap();
            let key = if key == "tools_json" {
                for tool in value.as_array_mut().unwrap() {
                    tool["function"]["strict"] = json!(true);
                }
                "tools"
            } else {
                key.as_str()
            };
            patch[key] = value;
        }
        let accepted = policy
            .validate(&request(patch.clone()))
            .unwrap_or_else(|e| panic!("{}: {patch} -> {e:?}", case["name"]));
        assert!(
            accepted.constraint_field().is_some() || accepted.tool_constrained(),
            "{}",
            case["name"]
        );
    }
}

/// C1 (D9): the SGLang body carries each declared field the client sent, with its value; absent fields stay
/// absent.
#[test]
fn c1_sglang_body_forwards_the_sampling_fields() {
    let sglang = sglang_policy_sampling(Some(&SGLANG_SAMPLING_FIELDS));
    let rid = Rid::new(2, 5, 0xab);
    let sent = json!({"top_k": 20, "frequency_penalty": 0.5, "presence_penalty": -1.5,
        "repetition_penalty": 1.25, "logit_bias": {"151643": -100, "7": 2.5}});
    let body = serde_json::to_value(
        sglang
            .validate(&request(sent.clone()))
            .unwrap()
            .backend_body(&rid),
    )
    .unwrap();
    for field in SGLANG_SAMPLING_FIELDS {
        assert_eq!(body[field], sent[field], "{field}: {body}");
    }
    let plain = serde_json::to_value(
        sglang
            .validate(&request(json!({})))
            .unwrap()
            .backend_body(&rid),
    )
    .unwrap();
    for field in SAMPLING_FIELDS {
        assert!(plain.get(field).is_none(), "{field}: {plain}");
    }
}

/// C1 (D8): each sampling field enters the request digest only when it is not at its default (`seed` whenever
/// given), so a pre-M2.7 body digests as before (the C2 goldens above) and an explicit default equals absence.
#[test]
fn c1_sampling_fields_digest_only_when_not_default() {
    let native = native_sampling_policy();
    // Every body carries the same explicit temperature.
    let normalized = |mut patch: Value| {
        patch["temperature"] = json!(0.7);
        native.validate(&request(patch)).unwrap().normalized()
    };
    let plain = normalized(json!({}));
    let defaults = normalized(json!({"top_k": -1, "min_p": 0, "frequency_penalty": 0.0,
        "presence_penalty": 0, "repetition_penalty": 1, "logit_bias": {}}));
    assert_eq!(plain, defaults);
    for field in SAMPLING_FIELDS {
        assert!(plain.get(field).is_none(), "{field}: {plain}");
    }
    for (field, value) in [
        ("top_k", json!(20)),
        ("min_p", json!(0.1)),
        ("seed", json!(0)),
        ("frequency_penalty", json!(0.5)),
        ("presence_penalty", json!(-0.5)),
        ("repetition_penalty", json!(1.1)),
        ("logit_bias", json!({"5": -100})),
    ] {
        let got = normalized(json!({field: value}));
        assert_ne!(got, plain, "{field}");
        assert!(got.get(field).is_some(), "{field}: {got}");
    }
    // Equal values digest equally however they are spelled (`1` and `1.0`, key order of `logit_bias`).
    assert_eq!(
        normalized(json!({"repetition_penalty": 2, "logit_bias": {"5": 1, "10": -2}})),
        normalized(json!({"repetition_penalty": 2.0, "logit_bias": {"10": -2.0, "5": 1.0}}))
    );
    assert_eq!(
        canonical_json(&normalized(json!({"top_k": 20, "min_p": 0.1, "seed": 42,
            "frequency_penalty": 0.5, "presence_penalty": -0.5, "repetition_penalty": 1.1,
            "logit_bias": {"10": -2, "5": 1}}))),
        r#"{"frequency_penalty":0.5,"include_usage":false,"logit_bias":{"10":-2.0,"5":1.0},"max_completion_tokens":1024,"messages":[{"content":"hi","role":"user"}],"min_p":0.1,"model":"Qwen3.8-Flash-Next","presence_penalty":-0.5,"repetition_penalty":1.1,"seed":42,"stop":[],"stream":false,"temperature":0.7,"top_k":20,"top_p":null}"#
    );
    // The pre-M2.7 bodies keep their digest input whether or not the profile declares the seven fields.
    let declaring = sglang_policy_sampling(Some(&SAMPLING_FIELDS));
    let undeclaring = sglang_policy_sampling(None);
    for body in pre_m24_bodies() {
        assert_eq!(
            canonical_json(&declaring.validate(&body).unwrap().normalized()),
            canonical_json(&undeclaring.validate(&body).unwrap().normalized())
        );
    }
}

/// D9: the shipped SGLang profile declares the five forwarded fields, and the live policy reads them.
#[test]
fn d9_shipped_sglang_profile_declares_five_sampling_fields() {
    let live = policy();
    assert_eq!(live.sampling_params, SGLANG_SAMPLING_FIELDS);
    live.validate(&request(
        json!({"top_k": 20, "frequency_penalty": 0.5, "presence_penalty": 0.5,
        "repetition_penalty": 1.1, "logit_bias": {"5": -100}}),
    ))
    .unwrap();
}

/// The eight pre-M2.12 bodies of the M2.12 C2/C9 regression guards (M2.4 fields included).
fn pre_m212_bodies() -> Vec<String> {
    let mut bodies: Vec<String> = pre_m24_bodies()
        .iter()
        .map(|b| serde_json::to_string(b).unwrap())
        .collect();
    bodies.push(raw_body(
        r#","temperature":0,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","x-a":1.0}}}],"tool_choice":"none","parallel_tool_calls":false"#,
    ));
    bodies.push(raw_body(
        r#","stream":true,"logprobs":true,"top_logprobs":3,"stop":["A","B"],"max_tokens":32"#,
    ));
    bodies
}

/// C2 + C9 regression guards (I12): `normalized()` (request digest input) and the SGLang body of eight pre-M2.12
/// bodies equal their canonical JSON / serialized bytes recorded on the merge-base `f16560b`.
#[test]
fn c2_c9_pre_m212_digest_and_body_are_unchanged() {
    let golden = [
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":false,"temperature":null,"top_p":null}"#,
            r#"{"max_completion_tokens":4096,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stream":false}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"s","role":"system"},{"content":"a b","role":"user"},{"content":"c","role":"assistant"},{"content":"d","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":["END"],"stream":false,"temperature":0.0,"top_p":null}"#,
            r#"{"max_completion_tokens":4096,"messages":[{"content":"s","role":"system"},{"content":"a b","role":"user"},{"content":"c","role":"assistant"},{"content":"d","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stop":["END"],"stream":false,"temperature":0.0}"#,
        ),
        (
            r#"{"include_usage":true,"max_completion_tokens":9,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":true,"temperature":null,"top_p":0.5}"#,
            r#"{"max_completion_tokens":9,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stream":true,"stream_options":{"include_usage":true},"top_p":0.5}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"é😀\n\"q\"","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":["é","x y"],"stream":false,"temperature":null,"top_p":null}"#,
            r#"{"max_completion_tokens":4096,"messages":[{"content":"é😀\n\"q\"","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stop":["é","x y"],"stream":false}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":16,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":false,"temperature":null,"top_p":null}"#,
            r#"{"max_completion_tokens":16,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stream":false}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"","role":"user"},{"content":"a","role":"assistant"},{"content":"b","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":true,"temperature":0.7,"top_p":null}"#,
            r#"{"max_completion_tokens":4096,"messages":[{"content":"","role":"user"},{"content":"a","role":"assistant"},{"content":"b","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stream":true,"stream_options":{"include_usage":true},"temperature":0.7}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","parallel_tool_calls":false,"stop":[],"stream":false,"temperature":0.0,"tool_choice":"none","tools":"[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"x-a\":1.0}}}]","top_p":null}"#,
            r#"{"max_completion_tokens":4096,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","parallel_tool_calls":false,"rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stream":false,"temperature":0.0,"tool_choice":"none","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","x-a":1.0}}}]}"#,
        ),
        (
            r#"{"include_usage":false,"logprobs":true,"max_completion_tokens":32,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":["A","B"],"stream":true,"temperature":null,"top_logprobs":3,"top_p":null}"#,
            r#"{"logprobs":true,"max_completion_tokens":32,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000002-00000000000000000005-00000000000000ab","stop":["A","B"],"stream":true,"stream_options":{"include_usage":true},"top_logprobs":3}"#,
        ),
    ];
    let bodies = pre_m212_bodies();
    assert_eq!(bodies.len(), golden.len());
    for (i, (text, (want_normalized, want_body))) in bodies.iter().zip(golden).enumerate() {
        let request = validate_text(text);
        let body = request.backend_body(&Rid::new(2, 5, 0xab));
        assert_eq!(
            canonical_json(&request.normalized()),
            want_normalized,
            "body {i}: normalized()"
        );
        assert_eq!(
            serde_json::to_string(&body).unwrap(),
            want_body,
            "body {i}: backend body"
        );
    }
}

/// `(code, param)` of a refused body.
fn refused_by(policy: &ChatPolicy, body: &Value) -> (&'static str, String) {
    let error = policy.validate(body).unwrap_err();
    (error.code, error.param.unwrap_or_default())
}

/// The live SGLang config with its profile's M2.12 fields replaced (`None` = absent, as gemma's profile).
fn sglang_policy_with(
    thinking_toggle: Option<bool>,
    token_controls: Option<&[&str]>,
    vocab_size: Option<u32>,
) -> ChatPolicy {
    let mut config = live_config();
    let profile = &mut config.deployments[0].lock.as_mut().unwrap().profile;
    profile.thinking_toggle = thinking_toggle;
    profile.token_controls = token_controls.map(|c| c.iter().map(|s| s.to_string()).collect());
    profile.vocab_size = vocab_size;
    ChatPolicy::from_config(&config, 0).unwrap()
}

/// The shipped native config with its profile's M2.12 fields replaced.
fn native_policy_with(token_controls: &[&str], vocab_size: u32) -> ChatPolicy {
    let mut config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0.toml"),
    )
    .unwrap();
    let profile = &mut config.deployments[0].lock.as_mut().unwrap().profile;
    profile.token_controls = Some(token_controls.iter().map(|s| s.to_string()).collect());
    profile.vocab_size = Some(vocab_size);
    ChatPolicy::from_config(&config, 0).unwrap()
}

/// M2.12 C1 (D1/D13): the six fields are accepted on the SGLang Qwen profile; duplicates in `stop_token_ids`
/// collapse; `chat_template_kwargs` `{}`/`null` are absent; thinking resolves as SGLang (explicit flag wins, else
/// `reasoning_effort != "none"`, else off).
#[test]
fn c1_m212_fields_are_accepted_on_the_sglang_profile() {
    let policy = policy();
    let all = policy
        .validate(&request(json!({
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": true},
            "separate_reasoning": false,
            "stop_token_ids": [5, 3, 3],
            "min_tokens": 16,
            "ignore_eos": true,
            "reasoning_effort": "low",
        })))
        .unwrap();
    assert!(all.thinking);
    assert_eq!(all.enable_thinking, Some(true));
    assert_eq!(all.separate_reasoning, Some(false));
    assert_eq!(all.stop_token_ids.as_deref(), Some(&[3, 5][..]));
    assert_eq!(all.min_tokens, Some(16));
    assert_eq!(all.ignore_eos, Some(true));
    assert_eq!(all.reasoning_effort, Some(json!("low")));

    let absent = policy.validate(&request(json!({}))).unwrap();
    assert!(!absent.thinking);
    assert_eq!(
        (
            absent.enable_thinking,
            absent.separate_reasoning,
            absent.min_tokens,
            absent.ignore_eos
        ),
        (None, None, None, None)
    );
    assert_eq!(
        (absent.stop_token_ids, absent.reasoning_effort),
        (None, None)
    );
    for kwargs in [json!({}), Value::Null] {
        let mut body = request(json!({}));
        body["chat_template_kwargs"] = kwargs.clone();
        let r = policy.validate(&body).unwrap();
        assert!(!r.thinking && r.enable_thinking.is_none(), "{kwargs}");
    }
    // 32 distinct ids, the largest one below the vocab.
    let ids: Vec<u32> = (1..32).chain([248_319]).collect();
    assert_eq!(ids.len(), 32);
    policy
        .validate(&request(json!({"stop_token_ids": ids})))
        .unwrap();
    policy
        .validate(&request(json!({"stop_token_ids": []})))
        .unwrap();
    policy.validate(&request(json!({"min_tokens": 0}))).unwrap();

    // D13 resolution: accepted values and the resolved flag.
    for (patch, thinking) in [
        (json!({"reasoning_effort": "none"}), false),
        (json!({"reasoning_effort": "low"}), true),
        (json!({"reasoning_effort": "medium"}), true),
        (json!({"reasoning_effort": "xhigh"}), true),
        (
            json!({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        (
            json!({"reasoning_effort": "max", "chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        (
            json!({"reasoning_effort": 0.5, "chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        (
            json!({"reasoning_effort": 0, "chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        (
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": true}}),
            true,
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            false,
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            true,
        ),
    ] {
        let r = policy.validate(&request(patch.clone())).unwrap();
        assert_eq!(r.thinking, thinking, "{patch}");
    }
}

/// M2.12 C1 (D1/D5/D13) refusals with their paths.
#[test]
fn c1_m212_refusals_name_their_field() {
    let policy = policy();
    let invalid = "invalid_value";
    let unsupported = "unsupported_parameter";
    let too_many: Vec<u32> = (0..33).collect();
    for (patch, want) in [
        (
            json!({"chat_template_kwargs": {"foo": 1}}),
            (unsupported, "chat_template_kwargs.foo"),
        ),
        (
            // ApiLenient: `true` is the identity (D3), `false` is honoured (AL-2, api_lenient.rs); other types fail.
            json!({"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": "no"}}),
            (invalid, "chat_template_kwargs.preserve_thinking"),
        ),
        (
            json!({"chat_template_kwargs": "x"}),
            (invalid, "chat_template_kwargs"),
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": "yes"}}),
            (invalid, "chat_template_kwargs.enable_thinking"),
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": null}}),
            (invalid, "chat_template_kwargs.enable_thinking"),
        ),
        (
            json!({"separate_reasoning": "no"}),
            (invalid, "separate_reasoning"),
        ),
        (
            json!({"reasoning_effort": "ultra"}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"reasoning_effort": 1.0}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"reasoning_effort": -0.1}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"reasoning_effort": true}),
            (invalid, "reasoning_effort"),
        ),
        // Thinking resolved on: the pinned template accepts only low, medium, xhigh.
        (
            json!({"reasoning_effort": "minimal"}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"reasoning_effort": "high"}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"reasoning_effort": "max"}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"reasoning_effort": 0.5}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}}),
            (invalid, "reasoning_effort"),
        ),
        (
            json!({"stream_reasoning": true}),
            (unsupported, "stream_reasoning"),
        ),
        (
            json!({"thinking": {"type": "enabled"}}),
            (unsupported, "thinking"),
        ),
        (
            json!({"stop_token_ids": too_many}),
            (invalid, "stop_token_ids"),
        ),
        (
            json!({"stop_token_ids": [248_320]}),
            (invalid, "stop_token_ids"),
        ),
        (json!({"stop_token_ids": [-1]}), (invalid, "stop_token_ids")),
        (
            json!({"stop_token_ids": ["1"]}),
            (invalid, "stop_token_ids"),
        ),
        (json!({"stop_token_ids": 1}), (invalid, "stop_token_ids")),
        (
            json!({"max_tokens": 16, "min_tokens": 17}),
            (invalid, "min_tokens"),
        ),
        (json!({"min_tokens": 4097}), (invalid, "min_tokens")),
        (json!({"min_tokens": -1}), (invalid, "min_tokens")),
        (json!({"min_tokens": 1.5}), (invalid, "min_tokens")),
        (json!({"ignore_eos": 1}), (invalid, "ignore_eos")),
    ] {
        let (code, param) = refused_by(&policy, &request(patch.clone()));
        assert_eq!((code, param.as_str()), want, "{patch}");
    }
}

/// M2.12 C1 (D2/D5/D10/D14): each field needs its profile capability; a refusal names the milestone that brings
/// it (M2.7 for native `min_tokens`).
#[test]
fn c1_m212_fields_are_gated_by_the_profile() {
    let fields = [
        json!({"chat_template_kwargs": {"enable_thinking": true}}),
        json!({"separate_reasoning": false}),
        json!({"reasoning_effort": "low"}),
        json!({"stop_token_ids": [1]}),
        json!({"min_tokens": 1}),
        json!({"ignore_eos": true}),
    ];
    let param = |patch: &Value| patch.as_object().unwrap().keys().next().unwrap().clone();

    // The shipped gemma profile declares none of the three fields; a profile like it refuses all six.
    let gemma: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/deployments/gemma-4-e4b-it.capabilities.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for key in ["thinking_toggle", "token_controls", "vocab_size"] {
        assert!(gemma.get(key).is_none(), "{key}");
    }
    let bare = sglang_policy_with(None, None, None);
    for patch in &fields {
        let error = bare.validate(&request(patch.clone())).unwrap_err();
        assert_eq!(
            (error.code, error.param.as_deref()),
            ("unsupported_parameter", Some(param(patch).as_str()))
        );
        assert!(error.message.contains("M2.12"), "{}", error.message);
    }
    bare.validate(&request(json!({"chat_template_kwargs": {}})))
        .unwrap();

    // D5: an SGLang profile without `vocab_size` refuses `stop_token_ids` only.
    let no_vocab = sglang_policy_with(
        Some(true),
        Some(&["stop_token_ids", "ignore_eos", "min_tokens"]),
        None,
    );
    assert_eq!(
        refused_by(&no_vocab, &request(json!({"stop_token_ids": [1]}))),
        ("unsupported_parameter", "stop_token_ids".to_owned())
    );
    no_vocab
        .validate(&request(json!({"min_tokens": 1, "ignore_eos": true})))
        .unwrap();

    // Native without the M2.12 fields (a pre-A1 profile, the old library): every field refused; `min_tokens`
    // names M2.7, the rest M2.12.
    let mut config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0.toml"),
    )
    .unwrap();
    let profile = &mut config.deployments[0].lock.as_mut().unwrap().profile;
    (
        profile.thinking_toggle,
        profile.token_controls,
        profile.vocab_size,
    ) = (None, None, None);
    let native = ChatPolicy::from_config(&config, 0).unwrap();
    for patch in &fields {
        let mut body = request(patch.clone());
        body["temperature"] = json!(0);
        let error = native.validate(&body).unwrap_err();
        assert_eq!(
            (error.code, error.param.as_deref()),
            ("unsupported_parameter", Some(param(patch).as_str()))
        );
        let milestone = if param(patch) == "min_tokens" {
            "M2.7"
        } else {
            "M2.12"
        };
        assert!(error.message.contains(milestone), "{}", error.message);
    }
    // The shipped native profile (A1, `-v5`): thinking, `stop_token_ids` and `ignore_eos` are accepted;
    // `min_tokens` names M2.7.
    let shipped = native_policy();
    for patch in &fields {
        let mut body = request(patch.clone());
        body["temperature"] = json!(0);
        if param(patch) == "min_tokens" {
            let error = shipped.validate(&body).unwrap_err();
            assert_eq!(error.param.as_deref(), Some("min_tokens"));
            assert!(error.message.contains("M2.7"), "{}", error.message);
        } else {
            shipped.validate(&body).unwrap();
        }
    }
    // Native with `stop_token_ids` declared (D8 needs no ABI change): accepted below the vocab, refused at it.
    let native = native_policy_with(&["stop_token_ids"], 248_320);
    let mut body = request(json!({"stop_token_ids": [248_319, 7]}));
    body["temperature"] = json!(0);
    let r = native.validate(&body).unwrap();
    assert_eq!(r.stop_token_ids.as_deref(), Some(&[7, 248_319][..]));
    body["stop_token_ids"] = json!([248_320]);
    assert_eq!(
        refused_by(&native, &body),
        ("invalid_value", "stop_token_ids".to_owned())
    );
    // Native `min_tokens` names M2.7 even when a profile lists it (no logits hook before order 16).
    let native = native_policy_with(&["stop_token_ids", "min_tokens"], 248_320);
    let mut body = request(json!({"min_tokens": 1}));
    body["temperature"] = json!(0);
    let error = native.validate(&body).unwrap_err();
    assert!(error.message.contains("M2.7"), "{}", error.message);
}

/// M2.12 C2 (D6): digest inputs. Absent, `{}` and `{"enable_thinking": false}` digest equal; thinking on differs;
/// `reasoning_effort` enters when sent (`"none"` resolves thinking off like absence); `stop_token_ids` digest as
/// the sorted set; defaults (`min_tokens: 0`, `ignore_eos: false`, `separate_reasoning: true`, `[]`) equal absence.
#[test]
fn c2_m212_digest_inputs_are_canonical() {
    let policy = policy();
    let normalized =
        |patch: Value| canonical_json(&policy.validate(&request(patch)).unwrap().normalized());
    let absent = normalized(json!({}));
    for patch in [
        json!({"chat_template_kwargs": {}}),
        json!({"chat_template_kwargs": {"enable_thinking": false}}),
        json!({"stop_token_ids": []}),
        json!({"min_tokens": 0}),
        json!({"ignore_eos": false}),
        json!({"separate_reasoning": true}),
    ] {
        assert_eq!(normalized(patch.clone()), absent, "{patch}");
    }
    for patch in [
        json!({"chat_template_kwargs": {"enable_thinking": true}}),
        json!({"reasoning_effort": "low"}),
        json!({"reasoning_effort": "none"}),
        json!({"stop_token_ids": [1]}),
        json!({"min_tokens": 1}),
        json!({"ignore_eos": true}),
        json!({"separate_reasoning": false}),
    ] {
        assert_ne!(normalized(patch.clone()), absent, "{patch}");
    }
    assert!(
        !policy
            .validate(&request(json!({"reasoning_effort": "none"})))
            .unwrap()
            .thinking
    );
    assert_eq!(
        normalized(json!({"stop_token_ids": [2, 1, 1]})),
        normalized(json!({"stop_token_ids": [1, 2]}))
    );
    assert_eq!(
        normalized(json!({"chat_template_kwargs": {"enable_thinking": true}})),
        normalized(
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": true}})
        )
        .replace(r#","reasoning_effort":"low""#, "")
    );

    // Session revisions digest the messages only: turn 2 with thinking toggled on has the same revision digest.
    let dir = std::env::temp_dir().join(format!("cere-m212-c2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let key = cere_runtime::digest::DigestKeys::open(&dir)
        .unwrap()
        .scope_key("s");
    let salt = cere_runtime::digest::Salt::random().unwrap();
    let turn2 = |patch: Value| {
        let mut body = request(patch);
        body["messages"] = json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}
        ]);
        let messages: Vec<Value> = policy
            .validate(&body)
            .unwrap()
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        key.prefix_digest(&salt, &messages)
    };
    assert_eq!(
        turn2(json!({})),
        turn2(json!({"chat_template_kwargs": {"enable_thinking": true}}))
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// M2.12 D1/F4: the `SamplingParams` verdicts of `token_control_cases.json` (D11) on the SGLang profile:
/// `stop_token_ids` collapse to a sorted set (an empty one finishes nothing: SGLang's `None`), `min_tokens` in `0..=max_tokens`, `ignore_eos` as sent.
#[test]
fn c1_token_control_params_equal_the_oracle() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/native/token_control_cases.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut n = 0;
    for case in doc["cases"].as_array().unwrap() {
        if case["kind"] != "params" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let mut patch = json!({ "max_tokens": case["max_new_tokens"] });
        for (key, value) in case["fields"].as_object().unwrap() {
            let key = if key == "min_new_tokens" {
                "min_tokens"
            } else {
                key
            };
            patch[key] = value.clone();
        }
        let got = policy().validate(&request(patch));
        let want = &case["expect"];
        if want.get("error").is_some() {
            let error = got.unwrap_err();
            assert_eq!(
                (error.code, error.param.as_deref()),
                ("invalid_value", Some("min_tokens")),
                "case {name}"
            );
        } else {
            let r = got.unwrap_or_else(|e| panic!("case {name}: {e:?}"));
            assert_eq!(
                json!([
                    r.stop_token_ids.filter(|ids| !ids.is_empty()),
                    r.min_tokens.unwrap_or(0),
                    r.ignore_eos.unwrap_or(false)
                ]),
                json!([
                    want["stop_token_ids"],
                    want["min_new_tokens"],
                    want["ignore_eos"]
                ]),
                "case {name}"
            );
        }
        n += 1;
    }
    assert_eq!(n, 9);
}

/// M2.12 D4/D13: on the SGLang profile, the resolved thinking flag equals the oracle's `force_reasoning`
/// (`_get_reasoning_from_request`, `thinking_cases.json`) for every request SGLang renders, and a value SGLang's
/// validator or the pinned template refuses (`error.stage` `validate` / `render`) is 400 `invalid_value` at
/// `reasoning_effort` before any dispatch, on this path as on native (a forwarded render refusal would come back as
/// an unclassified backend 400).
#[test]
fn c2_resolved_thinking_equals_the_oracle() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/native/thinking_cases.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let (mut rendered, mut refused) = (0, 0);
    for case in doc["cases"].as_array().unwrap() {
        if case["kind"] != "resolve" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let mut body = request(json!({}));
        for (key, value) in case["request"].as_object().unwrap() {
            body[key] = value.clone();
        }
        let got = policy().validate(&body);
        let want = &case["expect"];
        match want["error"]["stage"].as_str() {
            Some(_) => {
                let error = got.unwrap_err();
                assert_eq!(
                    (error.code, error.param.as_deref()),
                    ("invalid_value", Some("reasoning_effort")),
                    "case {name}"
                );
                refused += 1;
            }
            None => {
                let r = got.unwrap_or_else(|e| panic!("case {name}: {e:?}"));
                assert_eq!(json!(r.thinking), want["force_reasoning"], "case {name}");
                rendered += 1;
            }
        }
    }
    assert!(rendered >= 30 && refused >= 1, "{rendered} {refused}");
}

/// C3 regression guard (m213 design note D6, C4): `normalized()`, the session messages and the exact SGLang body
/// bytes of four pre-M2.13 bodies (tools, tool turns, logprobs, literal vision markers in text) equal their values
/// recorded on `f16560b`, before content parts existed.
#[test]
fn c3_pre_m213_digest_inputs_and_bodies_are_unchanged() {
    let policy = policy();
    let golden = [
        (
            r#"{"include_usage":false,"logprobs":true,"max_completion_tokens":4096,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":false,"temperature":0.3,"tools":"[{\"function\":{\"description\":\"d\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}},\"type\":\"function\"}]","top_logprobs":2,"top_p":null}"#,
            r#"[{"content":"hi","role":"user"}]"#,
            r#"{"logprobs":true,"max_completion_tokens":4096,"messages":[{"content":"hi","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000001-00000000000000000002-0000000000000003","stream":false,"temperature":0.3,"tool_choice":"auto","top_logprobs":2,"tools":[{"function":{"description":"d","name":"get_weather","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}},"type":"function"}]}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"s","role":"system"},{"content":"w","role":"user"},{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"city\": \"Paris\"}","name":"get_weather"},"id":"call_1","type":"function"}]},{"content":"{\"t\": 21}","role":"tool","tool_call_id":"call_1"},{"content":"x","role":"user"}],"model":"Qwen3.8-Flash-Next","parallel_tool_calls":false,"stop":[],"stream":false,"temperature":null,"tool_choice":"none","tools":"[{\"function\":{\"description\":\"d\",\"name\":\"a\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}},\"type\":\"function\"},{\"function\":{\"description\":\"d\",\"name\":\"b\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}},\"type\":\"function\"}]","top_p":null}"#,
            r#"[{"content":"s","role":"system"},{"content":"w","role":"user"},{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"city\": \"Paris\"}","name":"get_weather"},"id":"call_1","type":"function"}]},{"content":"{\"t\": 21}","role":"tool","tool_call_id":"call_1"},{"content":"x","role":"user"}]"#,
            r#"{"max_completion_tokens":4096,"messages":[{"content":"s","role":"system"},{"content":"w","role":"user"},{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"city\": \"Paris\"}","name":"get_weather"},"id":"call_1","type":"function"}]},{"content":"{\"t\": 21}","role":"tool","tool_call_id":"call_1"},{"content":"x","role":"user"}],"model":"Qwen3.8-Flash-Next","parallel_tool_calls":false,"rid":"cere-00000000000000000001-00000000000000000002-0000000000000003","stream":false,"tool_choice":"none","tools":[{"function":{"description":"d","name":"a","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}},"type":"function"},{"function":{"description":"d","name":"b","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}},"type":"function"}]}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":4096,"messages":[{"content":"<|image_pad|> <|vision_start|> literal","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":["\n\n"],"stream":true,"temperature":0.0,"top_p":null}"#,
            r#"[{"content":"<|image_pad|> <|vision_start|> literal","role":"user"}]"#,
            r#"{"max_completion_tokens":4096,"messages":[{"content":"<|image_pad|> <|vision_start|> literal","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000001-00000000000000000002-0000000000000003","stop":["\n\n"],"stream":true,"stream_options":{"include_usage":true},"temperature":0.0}"#,
        ),
        (
            r#"{"include_usage":false,"max_completion_tokens":3,"messages":[{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"city\": \"Paris\"}","name":"get_weather"},"id":"call_1","type":"function"}]},{"content":"y","role":"user"}],"model":"Qwen3.8-Flash-Next","stop":[],"stream":false,"temperature":null,"top_p":null}"#,
            r#"[{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"city\": \"Paris\"}","name":"get_weather"},"id":"call_1","type":"function"}]},{"content":"y","role":"user"}]"#,
            r#"{"max_completion_tokens":3,"messages":[{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"city\": \"Paris\"}","name":"get_weather"},"id":"call_1","type":"function"}]},{"content":"y","role":"user"}],"model":"Qwen3.8-Flash-Next","rid":"cere-00000000000000000001-00000000000000000002-0000000000000003","stream":false}"#,
        ),
    ];
    for (i, (body, (want_request, want_messages, want_body))) in
        pre_m213_bodies().into_iter().zip(golden).enumerate()
    {
        let request = policy.validate(&body).unwrap();
        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        assert_eq!(
            canonical_json(&request.normalized()),
            want_request,
            "body {i}: normalized()"
        );
        assert_eq!(
            canonical_json(&Value::Array(messages)),
            want_messages,
            "body {i}: session messages"
        );
        let sent = serde_json::to_string(&request.backend_body(&Rid::new(1, 2, 3))).unwrap();
        assert_eq!(sent, want_body, "body {i}: SGLang body");
    }
}

fn pre_m213_bodies() -> [Value; 4] {
    let call = json!({"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}});
    [
        request(
            json!({"tools": [tool("get_weather")], "temperature": 0.3, "logprobs": true, "top_logprobs": 2}),
        ),
        request(
            json!({"tools": [tool("a"), tool("b")], "tool_choice": "none", "parallel_tool_calls": false, "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "w"},
                {"role": "assistant", "content": Value::Null, "tool_calls": [call]},
                {"role": "tool", "content": "{\"t\": 21}", "tool_call_id": "call_1"},
                {"role": "user", "content": "x"}
            ]}),
        ),
        request(
            json!({"stream": true, "stop": ["\n\n"], "temperature": 0, "messages": [
                {"role": "user", "content": "<|image_pad|> <|vision_start|> literal"}
            ]}),
        ),
        request(json!({"max_tokens": 3, "messages": [
            {"role": "assistant", "content": "", "tool_calls": [call], "reasoning_content": Value::Null},
            {"role": "user", "content": "y"}
        ]})),
    ]
}

// ---- M2.13 (m213 design note D2-D7): content parts and images ----

fn image(url: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}

fn user_parts(parts: Value) -> Value {
    request(json!({"messages": [{"role": "user", "content": parts}]}))
}

fn federation_policy() -> ChatPolicy {
    let config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/federation-dev.toml"),
    )
    .unwrap();
    ChatPolicy::from_config(&config, 0).unwrap()
}

/// D5: the shipped SGLang qwen38 profile declares `image` with greedy and sampled image decode combinations,
/// rows 1-4 and 200,000 tokens like its text twins; the notes say they are declared, unmeasured.
#[test]
fn d5_shipped_sglang_profile_declares_image_combinations() {
    let profile = &live_config().deployments[0].lock.clone().unwrap().profile;
    assert_eq!(profile.modalities, ["text", "image"]);
    let images: Vec<(&str, &[u64], u64)> = profile
        .validated_combinations
        .iter()
        .filter(|c| c.modality == "image")
        .map(|c| {
            assert_eq!(
                (c.operation.as_str(), c.output.as_str()),
                ("decode", "text")
            );
            (
                c.sampler_mode.as_str(),
                c.row_counts.as_slice(),
                c.max_context_tokens,
            )
        })
        .collect();
    assert_eq!(
        images,
        [
            ("greedy", &[1, 2, 3, 4][..], 200_000),
            ("sampled", &[1, 2, 3, 4][..], 200_000)
        ]
    );
    assert!(profile.notes.contains("Image input"), "{}", profile.notes);
    let policy = policy();
    assert!(policy.images_greedy && policy.images_sampled && !policy.remote_workers);
    let native = native_policy();
    assert!(!native.images_greedy && !native.images_sampled);
}

/// C1/D2/D4: accepted parts arrays, the limits at their edges, and each image's grid.
#[test]
fn c1_accepted_content_parts() {
    let policy = policy();
    let png = png_url(64, 48);
    let text_only = json!([{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]);
    let accepted = [
        user_parts(text_only.clone()),
        user_parts(json!([{"type": "text", "text": "what is this?"}, image(&png)])),
        user_parts(json!([image(&jpeg_url(640, 480))])),
        user_parts(json!([{"type": "image_url", "image_url": {"url": png, "detail": "auto"}}])),
        user_parts(
            json!([{"type": "image_url", "image_url": {"url": png, "detail": Value::Null}}]),
        ),
        // 32 images across the history (user and tool messages, C7), 4,096 tokens for one.
        request(json!({"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [image(&png_url(2048, 2048)), image(&png_url(1024, 512))]},
            {"role": "assistant", "content": "a", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "shot"}, image(&png_url(1280, 900))]},
            {"role": "user", "content": [image(&png_url(512, 512)), {"type": "text", "text": "t"}, image(&jpeg_url(512, 512))]},
            {"role": "user", "content": (0..27).map(|_| image(&png)).collect::<Vec<_>>()}
        ]})),
        user_parts(Value::Array(
            (0..64)
                .map(|_| json!({"type": "text", "text": "x"}))
                .collect(),
        )),
    ];
    for body in &accepted {
        if let Err(e) = policy.validate(body) {
            panic!("{body:.300}: {e:?}");
        }
    }
    let validated = policy.validate(&accepted[5]).unwrap();
    let grids: Vec<([u32; 3], u32)> = validated
        .images()
        .map(|i| (i.grid_thw(), i.tokens()))
        .collect();
    assert_eq!(grids.len(), 32);
    assert_eq!(
        grids[..5],
        [
            ([1, 128, 128], 4096),
            ([1, 32, 64], 512),
            ([1, 56, 80], 1120),
            ([1, 32, 32], 256),
            ([1, 32, 32], 256)
        ]
    );
    // Native: a text-only parts array is accepted in a user or tool message (it renders as the joined string, F4).
    // A tool message's array without an image is that string (DocInput DI-02); with an image it stays parts (C7).
    let call =
        json!([{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]);
    let tool = |policy: &ChatPolicy, content: Value| {
        let mut body = request(json!({"messages": [
            {"role": "user", "content": text_only.clone()},
            {"role": "assistant", "content": "a", "tool_calls": call.clone()},
            {"role": "tool", "tool_call_id": "c", "content": content}
        ]}));
        body["temperature"] = json!(0);
        policy.validate(&body).unwrap().messages[2].content.clone()
    };
    let joined: String = text_only
        .as_array()
        .unwrap()
        .iter()
        .map(|part| part["text"].as_str().unwrap())
        .collect();
    assert_eq!(
        tool(&native_policy(), text_only.clone()),
        Some(Content::Text(joined))
    );
    let shot = json!([{"type": "text", "text": "shot"}, image(&png_url(64, 64))]);
    assert!(matches!(tool(&policy, shot), Some(Content::Parts(parts)) if parts.len() == 2));
}

/// VisionLong C2: the images' tokens together fit the profile's image context (the -v6 lab profile: 2,048).
#[test]
fn c2_image_tokens_fit_the_image_context() {
    let mut policy = policy();
    policy.image_context_tokens = Some(2048);
    // 1,024 + 961 + 64 = 2,049 tokens in one request.
    let body = user_parts(json!([
        image(&png_url(1024, 1024)),
        image(&png_url(992, 992)),
        image(&png_url(64, 64))
    ]));
    let e = policy.validate(&body).unwrap_err();
    assert_eq!(
        e.param.as_deref(),
        Some("messages[0].content[2].image_url.url"),
        "{e:?}"
    );
    policy.image_context_tokens = Some(2049);
    policy.validate(&body).unwrap();
}

/// VisionLong C3: image speculation and prefix capture derive from the profile's image combinations. The -v6 lab
/// profile (one greedy image decode combination, rows [1], 2,048) keeps context 2,048, no speculation and no
/// capture; adding image `verify` (greedy) and `prefill` combinations turns them on for image requests only.
#[test]
fn c3_image_speculation_and_capture_follow_the_image_combinations() {
    let mut config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0-vision.toml"),
    )
    .unwrap();
    let lab = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(lab.images_greedy && !lab.images_sampled);
    assert_eq!(lab.image_context_tokens, Some(2048));
    assert!(!lab.image_speculation_greedy && !lab.image_speculation_sampled && !lab.image_capture);
    let mut body = user_parts(json!([image(&png_url(64, 64))]));
    body["temperature"] = json!(0);
    let request = lab.validate(&body).unwrap();
    assert!(!request.image_speculation && !request.image_capture);

    let profile = &mut config.deployments[0].lock.as_mut().unwrap().profile;
    let decode = profile
        .validated_combinations
        .iter()
        .find(|c| c.modality == "image")
        .unwrap()
        .clone();
    let mut verify = decode.clone();
    verify.operation = "verify".to_owned();
    let mut prefill = decode;
    prefill.operation = "prefill".to_owned();
    prefill.sampler_mode = "not_applicable".to_owned();
    profile.validated_combinations.extend([verify, prefill]);
    let long = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(long.image_speculation_greedy && !long.image_speculation_sampled && long.image_capture);
    let request = long.validate(&body).unwrap();
    assert!(request.image_speculation && request.image_capture);
    let text = long.validate(&request_text_greedy()).unwrap();
    assert!(!text.image_speculation && !text.image_capture && text.image_context_tokens.is_none());
}

fn request_text_greedy() -> Value {
    request(json!({"temperature": 0}))
}

/// C1: every refusal names its field; nothing reaches a backend.
#[test]
fn c1_image_refusals_name_their_path() {
    let policy = policy();
    let png = png_url(64, 64);
    let payload = b64(&image_support::png(64, 64));
    let url = "messages[0].content[0].image_url.url";
    let thirty_three: Vec<Value> = (0..33).map(|_| image(&png)).collect();
    let sixteen = Value::Array((0..16).map(|_| image(&png)).collect());
    let cases = [
        (
            user_parts(json!([image("http://example.invalid/a.png")])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image("https://example.invalid/a.png")])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image("file:///etc/hostname")])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image("/tmp/a.png")])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image(&format!("data:image/gif;base64,{payload}"))])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image(&format!("data:image/png;base64,{payload}!"))])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image(&format!(
                "data:image/png;base64,{} {}",
                &payload[..4],
                &payload[4..]
            ))])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image(&png_url(65535, 65535))])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!([image(&png_url(201, 1))])),
            "invalid_value",
            url,
        ),
        // 4,097+ tokens: a 64 x 65 grid (4,160).
        (
            user_parts(json!([image(&png_url(2080, 2048))])),
            "invalid_value",
            url,
        ),
        (
            user_parts(Value::Array(thirty_three)),
            "invalid_value",
            "messages[0].content[32].image_url.url",
        ),
        (
            request(json!({"messages": [
                {"role": "user", "content": sixteen.clone()},
                {"role": "assistant", "content": "a", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c", "content": sixteen.clone()},
                {"role": "user", "content": [{"type": "text", "text": "t"}, image(&png)]}
            ]})),
            "invalid_value",
            "messages[3].content[1].image_url.url",
        ),
        (
            request(
                json!({"messages": [{"role": "system", "content": [image(&png)]}, {"role": "user", "content": "u"}]}),
            ),
            "unsupported_parameter",
            "messages[0].content",
        ),
        (
            request(
                json!({"messages": [{"role": "system", "content": [{"type": "text", "text": "s"}]}, {"role": "user", "content": "u"}]}),
            ),
            "unsupported_parameter",
            "messages[0].content",
        ),
        (
            request(
                json!({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": [image(&png)]}]}),
            ),
            "unsupported_parameter",
            "messages[1].content",
        ),
        (
            request(
                json!({"messages": [{"role": "user", "content": "u"}, {"role": "tool", "tool_call_id": "c", "content": [{"type": "image", "image": png}]}]}),
            ),
            "unsupported_parameter",
            "messages[1].content[0].type",
        ),
        (
            user_parts(json!([{"type": "image_url", "image_url": {"url": png, "detail": "high"}}])),
            "unsupported_parameter",
            "messages[0].content[0].image_url.detail",
        ),
        (
            user_parts(json!([{"type": "image_url", "image_url": {"url": png, "detail": 1}}])),
            "invalid_value",
            "messages[0].content[0].image_url.detail",
        ),
        (
            user_parts(
                json!([{"type": "image_url", "image_url": {"url": png, "max_dynamic_patch": 4}}]),
            ),
            "unsupported_parameter",
            "messages[0].content[0].image_url.max_dynamic_patch",
        ),
        (
            user_parts(
                json!([{"type": "image_url", "image_url": {"url": png}, "min_dynamic_patch": 1}]),
            ),
            "unsupported_parameter",
            "messages[0].content[0].min_dynamic_patch",
        ),
        (
            user_parts(json!([{"type": "video_url", "video_url": {"url": png}}])),
            "unsupported_parameter",
            "messages[0].content[0].type",
        ),
        (
            user_parts(json!([{"type": "input_image", "image_url": png}])),
            "unsupported_parameter",
            "messages[0].content[0].type",
        ),
        (
            user_parts(json!([{"type": "image", "image": png}])),
            "unsupported_parameter",
            "messages[0].content[0].type",
        ),
        (
            user_parts(json!([{"text": "a"}])),
            "missing_required_parameter",
            "messages[0].content[0].type",
        ),
        (
            user_parts(json!([{"type": 1}])),
            "invalid_value",
            "messages[0].content[0].type",
        ),
        (
            user_parts(json!([{"type": "text", "text": 1}])),
            "invalid_value",
            "messages[0].content[0].text",
        ),
        (
            user_parts(json!([{"type": "text"}])),
            "missing_required_parameter",
            "messages[0].content[0].text",
        ),
        (
            user_parts(json!([{"type": "text", "text": "a", "cache_control": {}}])),
            "unsupported_parameter",
            "messages[0].content[0].cache_control",
        ),
        (
            user_parts(json!([{"type": "image_url"}])),
            "missing_required_parameter",
            "messages[0].content[0].image_url",
        ),
        (
            user_parts(json!([{"type": "image_url", "image_url": png}])),
            "invalid_value",
            "messages[0].content[0].image_url",
        ),
        (
            user_parts(json!([{"type": "image_url", "image_url": {}}])),
            "missing_required_parameter",
            url,
        ),
        (
            user_parts(json!([{"type": "image_url", "image_url": {"url": 7}}])),
            "invalid_value",
            url,
        ),
        (
            user_parts(json!(["a"])),
            "invalid_value",
            "messages[0].content[0]",
        ),
        (
            user_parts(json!([])),
            "invalid_value",
            "messages[0].content",
        ),
        (
            user_parts(Value::Array(
                (0..65)
                    .map(|_| json!({"type": "text", "text": "x"}))
                    .collect(),
            )),
            "invalid_value",
            "messages[0].content",
        ),
    ];
    for (body, code, param) in cases {
        let error = policy.validate(&body).unwrap_err();
        assert_eq!(
            (error.code, error.param.as_deref()),
            (code, Some(param)),
            "{}",
            error.message
        );
        assert!(
            !error.message.contains(&payload[..16]) && !error.message.contains("example.invalid")
        );
    }
}

/// C1/D5: an image needs the profile's image decode combination for the request's sampler mode; a deployment
/// with federation workers refuses images (Q3 a).
#[test]
fn c1_images_need_an_image_combination_and_a_local_route() {
    let png = png_url(64, 64);
    let body = |temperature: f64| {
        let mut body = user_parts(json!([image(&png)]));
        body["temperature"] = json!(temperature);
        body
    };
    let refused = |policy: &ChatPolicy, body: &Value| {
        let error = policy.validate(body).unwrap_err();
        (error.code, error.param)
    };
    let image_path = (
        "unsupported_parameter",
        Some("messages[0].content[0]".to_owned()),
    );
    assert_eq!(refused(&native_policy(), &body(0.0)), image_path);
    let mut config = live_config();
    config.deployments[0]
        .lock
        .as_mut()
        .unwrap()
        .profile
        .validated_combinations
        .retain(|c| !(c.modality == "image" && c.sampler_mode == "greedy"));
    let sampled_only = ChatPolicy::from_config(&config, 0).unwrap();
    assert!(!sampled_only.images_greedy && sampled_only.images_sampled);
    assert_eq!(refused(&sampled_only, &body(0.0)), image_path);
    sampled_only.validate(&body(0.7)).unwrap();
    // The refusal precedes the URL check: a native client learns images are unsupported, not why the URL is.
    let bad_url = user_parts(json!([image("data:,")]));
    assert_eq!(refused(&native_policy(), &bad_url), image_path);

    let federated = federation_policy();
    assert!(federated.remote_workers && federated.images_sampled);
    assert_eq!(refused(&federated, &body(0.7)), image_path);
    federated
        .validate(&user_parts(json!([{"type": "text", "text": "t"}])))
        .unwrap();
}

/// C3/D6: the parts array digests as sent: one changed image byte or reordered parts change the request and
/// session digests; the same parts digest equal.
#[test]
fn c3_image_digests_follow_the_bytes() {
    let policy = policy();
    let mut other = png(64, 64);
    *other.last_mut().unwrap() ^= 1;
    let a = image(&png_url(64, 64));
    let b = image(&format!("data:image/png;base64,{}", b64(&other)));
    let text = json!({"type": "text", "text": "t"});
    let digest = |parts: Value| {
        let request = policy.validate(&user_parts(parts)).unwrap();
        (
            canonical_json(&request.normalized()),
            canonical_json(&serde_json::to_value(&request.messages).unwrap()),
        )
    };
    let base = digest(json!([text, a]));
    assert_eq!(base, digest(json!([text, a])));
    let changed = digest(json!([text, b]));
    assert!(base.0 != changed.0 && base.1 != changed.1);
    let reordered = digest(json!([a, text]));
    assert!(base.0 != reordered.0 && base.1 != reordered.1);
    // A text-only parts array is digested as sent, not as its joined string.
    assert_ne!(digest(json!([text])), {
        let request = policy.validate(&user_parts(json!("t"))).unwrap();
        (
            canonical_json(&request.normalized()),
            canonical_json(&serde_json::to_value(&request.messages).unwrap()),
        )
    });
}

/// D7: the SGLang body forwards the validated parts value-exact: `type`, `text`, `image_url.url`, and `detail` only
/// when sent.
#[test]
fn d7_backend_body_forwards_parts_value_exact() {
    let policy = policy();
    let png = png_url(300, 200);
    let parts = json!([
        {"type": "text", "text": "compare "},
        {"type": "image_url", "image_url": {"url": png}},
        {"type": "image_url", "image_url": {"detail": "auto", "url": jpeg_url(32, 32)}}
    ]);
    let request = policy.validate(&user_parts(parts.clone())).unwrap();
    let body = serde_json::to_value(request.backend_body(&Rid::new(1, 2, 3))).unwrap();
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": parts}])
    );
    let bytes = serde_json::to_string(&request.backend_body(&Rid::new(1, 2, 3))).unwrap();
    assert!(bytes.contains(&format!("\"url\":\"{png}\"")));
    assert_eq!(bytes.matches("\"detail\"").count(), 1);
}

/// The native-g0 policy with its profile's `decode_batch_max` set to `rows` (D10's R).
fn native_policy_rows(rows: u64) -> ChatPolicy {
    let mut config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0.toml"),
    )
    .unwrap();
    let profile = &mut config.deployments[0].lock.as_mut().unwrap().profile;
    profile.decode_batch_max = rows;
    profile.native_decode_batch = rows > 1;
    ChatPolicy::from_config(&config, 0).unwrap()
}

fn refusal_status(policy: &ChatPolicy, patch: Value) -> (u16, &'static str, Option<String>) {
    let error = policy.validate(&request(patch)).unwrap_err();
    (error.status, error.code, error.param)
}

/// M2.6 C3 (D11): a native profile with R = 4 accepts n 1..4 and refuses n 5 (OQ5) and any other spelling; ledger
/// CR4: sampled n > 1 (temperature > 0 or absent, D10) is served iff the profile has a sampled decode combination
/// (else the sampled refusal); n stays out of the backend body and enters the digest only when it is not 1.
#[test]
fn c3_native_n_is_bounded_by_the_profile_rows() {
    let policy = native_policy_rows(4);
    assert_eq!(policy.max_choices, 4);
    for n in 1..=4u32 {
        let request = policy
            .validate(&request(json!({"n": n, "temperature": 0})))
            .unwrap();
        assert_eq!(request.n, n);
        assert!(request
            .backend_body(&Rid::new(1, 1, 1))
            .fields
            .get("n")
            .is_none());
    }
    assert_eq!(
        policy
            .validate(&request(json!({"temperature": 0})))
            .unwrap()
            .n,
        1
    );
    for n in [json!(5), json!(0), json!(2.5), json!("2"), json!(-1)] {
        let patch = json!({"n": n, "temperature": 0});
        assert_eq!(
            refusal_status(&policy, patch.clone()),
            (400, "unsupported_parameter", Some("n".to_owned())),
            "{patch}"
        );
    }
    // Ledger CR4: the shipped profile's sampled combination serves sampled n 1..4 (distinct per-choice seeds, M8
    // (2, 3)); n 5 stays refused on `n`.
    for n in 1..=4u32 {
        for patch in [json!({"n": n, "temperature": 0.7}), json!({"n": n})] {
            assert_eq!(policy.validate(&request(patch)).unwrap().n, n);
        }
    }
    assert_eq!(
        refusal_status(&policy, json!({"n": 5, "temperature": 0.7})),
        (400, "unsupported_parameter", Some("n".to_owned()))
    );
    // A greedy-only native profile refuses sampled n with the sampled refusal.
    let mut greedy_only = policy.clone();
    greedy_only.sampled = false;
    let sampled = (400, "unsupported_parameter", Some("temperature".to_owned()));
    assert_eq!(
        refusal_status(&greedy_only, json!({"n": 2, "temperature": 0.7})),
        sampled
    );
    assert_eq!(refusal_status(&greedy_only, json!({"n": 2})), sampled);
    // n enters the digest only when it is not 1.
    let one = policy
        .validate(&request(json!({"temperature": 0, "n": 1})))
        .unwrap();
    let absent = policy
        .validate(&request(json!({"temperature": 0})))
        .unwrap();
    let two = policy
        .validate(&request(json!({"temperature": 0, "n": 2})))
        .unwrap();
    assert_eq!(one.normalized(), absent.normalized());
    assert!(absent.normalized().get("n").is_none());
    assert_eq!(two.normalized()["n"], 2);
}

/// M2.6 C3: one row (R = 1, today's profile) and every SGLang profile still refuse n != 1.
#[test]
fn c3_one_row_and_sglang_refuse_n_above_one() {
    for policy in [native_policy_rows(1), policy()] {
        assert_eq!(policy.max_choices, 1);
        assert_eq!(
            refusal_status(&policy, json!({"n": 2, "temperature": 0})),
            (400, "unsupported_parameter", Some("n".to_owned()))
        );
        policy
            .validate(&request(json!({"n": 1, "temperature": 0})))
            .unwrap();
    }
    // The SGLang profile declares decode_batch_max 4, but its rows are concurrent requests (OQ4: n stays refused).
    let config = live_config();
    assert_eq!(
        config.deployments[0]
            .lock
            .as_ref()
            .unwrap()
            .profile
            .decode_batch_max,
        4
    );
}

/// M2.6 C3 (D11, OQ9): n > 1 on a session turn (`X-Cere-Session-Id`/`Expected-Revision`) or in continue mode
/// (`X-Cere-On-Disconnect: continue`) is refused `unsupported_parameter` `n`; n 1 passes either.
#[test]
fn c3_n_above_one_refuses_session_turns_and_continue_mode() {
    let policy = native_policy_rows(4);
    let two = policy
        .validate(&request(json!({"n": 2, "temperature": 0, "stream": true})))
        .unwrap();
    two.check_choices(false, false).unwrap();
    for (session, continue_mode) in [(true, false), (false, true), (true, true)] {
        let error = two.check_choices(session, continue_mode).unwrap_err();
        assert_eq!(
            (error.status, error.code, error.param.as_deref()),
            (400, "unsupported_parameter", Some("n")),
            "session {session}, continue {continue_mode}"
        );
    }
    let one = policy
        .validate(&request(json!({"temperature": 0, "stream": true})))
        .unwrap();
    one.check_choices(true, true).unwrap();
}

/// ApiFit AF-02 (api-fit design note): a system message after `messages[0]` (the pinned template raises on it,
/// F1) validates and renders in place as a user message holding its content between the marks; message 0 and every
/// other message are unchanged, on the backend body and the digest. Folding is idempotent: the folded conversation
/// validates to the same request.
#[test]
fn af02_mid_conversation_system_messages_render_in_place() {
    let policy = policy();
    let rid = Rid::new(1, 2, 3);
    let system = |text: &str| json!({"role": "system", "content": text});
    let user = |text: &str| json!({"role": "user", "content": text});
    let tool = json!({"role": "tool", "content": "sunny", "tool_call_id": "call_1"});
    let cases = [
        // After a user turn, behind a leading system message (a coding agent's `<system-reminder>`).
        vec![
            system("s"),
            user("q"),
            system("<system-reminder>r</system-reminder>"),
            user("n"),
        ],
        // After an assistant turn, no leading system message.
        vec![
            user("q"),
            json!({"role": "assistant", "content": "a"}),
            system("r"),
            user("n"),
        ],
        // Two leading system messages: only the first leads.
        vec![system("s"), system("r"), user("q")],
        // Between tool results.
        vec![
            user("q"),
            echoed_call(),
            tool.clone(),
            system("r"),
            tool.clone(),
        ],
        // The last message.
        vec![system("s"), user("q"), system("r")],
    ];
    for messages in cases {
        let folded: Vec<Value> = messages
            .iter()
            .enumerate()
            .map(|(i, m)| match (i, m["role"].as_str()) {
                (1.., Some("system")) => json!({"role": "user", "content": format!(
                    "{MID_SYSTEM_OPEN}{}{MID_SYSTEM_CLOSE}",
                    m["content"].as_str().unwrap()
                )}),
                _ => m.clone(),
            })
            .collect();
        let validate = |messages: &Vec<Value>| {
            let body = request(json!({"messages": messages}));
            policy
                .validate(&body)
                .unwrap_or_else(|e| panic!("{body} -> {e:?}"))
        };
        let got = validate(&messages);
        let want = validate(&folded);
        assert_eq!(got.normalized(), want.normalized(), "{messages:?}");
        let body = serde_json::to_value(got.backend_body(&rid)).unwrap();
        let sent = body["messages"].as_array().unwrap();
        assert_eq!(sent.len(), messages.len());
        for (i, (sent, sent_as)) in sent.iter().zip(&messages).enumerate() {
            let role = if i > 0 && sent_as["role"] == "system" {
                assert_eq!(
                    sent["content"],
                    format!(
                        "<system-message>\n{}\n</system-message>",
                        sent_as["content"].as_str().unwrap()
                    )
                );
                "user"
            } else {
                sent_as["role"].as_str().unwrap()
            };
            assert_eq!(sent["role"], role, "{body}");
        }
    }
}

/// ApiFit AF-03: a request whose only system message is first, or that has none, validates exactly as on main (the
/// fold never touches it): its messages reach the backend as sent.
#[test]
fn af03_leading_only_system_is_unchanged() {
    let policy = policy();
    let rid = Rid::new(1, 2, 3);
    for messages in [
        json!([{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
               {"role": "assistant", "content": "a"}, {"role": "user", "content": "n"}]),
        json!([{"role": "user", "content": "q"}]),
    ] {
        let request = policy
            .validate(&request(json!({"messages": messages})))
            .unwrap();
        let body = serde_json::to_value(request.backend_body(&rid)).unwrap();
        assert_eq!(body["messages"], messages);
        assert_eq!(request.normalized()["messages"], messages);
    }
}
