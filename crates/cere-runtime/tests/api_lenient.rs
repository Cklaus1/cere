//! ApiLenient (api-lenient design note): lenient OpenAI parameter handling at the request layer.
//!
//! AL-0: the omp 18.12.6 wire bodies (`tests/fixtures/api/omp_wire/*.json`, captured by
//! `scripts/fixtures/capture_omp_wire.py`) and the identity corpus `tests/fixtures/api/identity_corpus.json`: request
//! bodies valid before ApiLenient with their validated request, digest input (`normalized()`), SGLang backend body
//! and tokenizer sidecar `encode_chat` line, recorded on origin/main `2f6f836a` before any ApiLenient change.
//!
//! Record: `CERE_API_IDENTITY_RECORD=<path> cargo test -p cere-runtime --test api_lenient valid_today` writes the
//! corpus outcomes to `<path>` instead of comparing.
use cere_runtime::config::RuntimeConfig;
use cere_runtime::sglang::Rid;
use cere_runtime::validate::{ChatPolicy, ChatRequest};
use serde_json::{json, Map, Value};

const MODEL: &str = "Qwen3.8-Flash-Next";

fn repo(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

fn config(name: &str) -> RuntimeConfig {
    RuntimeConfig::load(repo(&format!("configs/runtime/{name}"))).unwrap()
}

fn policy_of(name: &str) -> ChatPolicy {
    ChatPolicy::from_config(&config(name), 0).unwrap()
}

/// The corpus policies: the SGLang deployment (cap 32768) and the shipped native speculation profile.
fn policies() -> [(&'static str, ChatPolicy); 2] {
    [
        ("sglang", policy_of("sglang-t4-qwen38.toml")),
        ("native", policy_of("native-g0-spec.toml")),
    ]
}

fn omp_fixture(name: &str) -> Value {
    let path = repo(&format!("tests/fixtures/api/omp_wire/{name}.json"));
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    fixture["body"].clone()
}

const OMP_LEVELS: [&str; 5] = ["off", "low", "medium", "high", "xhigh"];

/// The tokenizer sidecar's `encode_chat` request as `native/serve.rs::render` and `native/sidecar.rs::encode_chat`
/// build it (the same function as `document_identity.rs`).
fn sidecar_line(request: &ChatRequest) -> String {
    let messages = request
        .messages
        .iter()
        .map(|m| m.joined().unwrap_or_else(|| m.clone()))
        .collect::<Vec<_>>();
    let mut line = json!({
        "op": "encode_chat",
        "messages": messages,
        "render_tools": request.render_tools(),
        "stop": request.stop,
    });
    if let Some(tools) = request.tools_json() {
        line["tools_json"] = json!(tools);
    }
    if request.thinking {
        line["enable_thinking"] = json!(true);
    }
    if let Some(effort) = &request.reasoning_effort {
        line["reasoning_effort"] = effort.clone();
    }
    if let Some(preserve) = request.preserve_thinking {
        line["preserve_thinking"] = json!(preserve);
    }
    serde_json::to_string(&line).unwrap()
}

/// The four byte surfaces of a validated request.
fn bytes_of(request: &ChatRequest) -> Map<String, Value> {
    let rid = Rid::new(1, 2, 3);
    Map::from_iter([
        (
            "validated".to_owned(),
            json!(serde_json::to_string(request).unwrap()),
        ),
        (
            "normalized".to_owned(),
            json!(serde_json::to_string(&request.normalized()).unwrap()),
        ),
        (
            "backend".to_owned(),
            json!(serde_json::to_string(&request.backend_body(&rid)).unwrap()),
        ),
        ("sidecar".to_owned(), json!(sidecar_line(request))),
    ])
}

fn validate(
    policy: &ChatPolicy,
    body: &Value,
) -> Result<ChatRequest, cere_runtime::api_types::ApiError> {
    let bytes = serde_json::to_vec(body).unwrap();
    policy.validate_bytes(body, &bytes)
}

fn body(patch: Value) -> Value {
    let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]});
    for (key, value) in patch.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

const TOOLS: &str = r#"[{"type":"function","function":{"name":"Read","description":"Read a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}]"#;

/// Request bodies of `back_validate.rs`, `document_identity.rs` and the omp snippet fixtures, one per accepted
/// shape; each policy keeps those it validated when the corpus was recorded.
fn corpus_bodies() -> Vec<(String, Value)> {
    let tools: Value = serde_json::from_str(TOOLS).unwrap();
    let call = json!({"id": "call_1", "type": "function",
        "function": {"name": "Read", "arguments": "{\"file_path\":\"a.txt\"}"}});
    let patches = vec![
        ("minimal", json!({})),
        ("n_1", json!({"n": 1})),
        ("max_tokens_1", json!({"max_tokens": 1})),
        (
            "max_completion_2048",
            json!({"max_completion_tokens": 2048}),
        ),
        (
            "max_completion_32768",
            json!({"max_completion_tokens": 32768}),
        ),
        (
            "max_tokens_null_pair",
            json!({"max_tokens": 16, "max_completion_tokens": null}),
        ),
        ("greedy", json!({"temperature": 0})),
        ("sampled_full", json!({"temperature": 2, "top_p": 1})),
        ("top_p", json!({"top_p": 0.5})),
        ("stop_string", json!({"stop": "END"})),
        ("stop_list", json!({"stop": ["a", "b"]})),
        ("stream", json!({"stream": true})),
        (
            "stream_usage",
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        ),
        ("stream_false", json!({"stream": false})),
        (
            "system_user_assistant",
            json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": ""},
                {"role": "assistant", "content": "a"}]}),
        ),
        (
            "mid_system",
            json!({"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "s"},
                {"role": "user", "content": "b"}]}),
        ),
        (
            "text_parts",
            json!({"temperature": 0, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]}),
        ),
        (
            "named_messages",
            json!({"messages": [{"role": "system", "content": "s", "name": "n"},
                {"role": "user", "content": "u", "name": "m"}]}),
        ),
        (
            "reasoning_content",
            json!({"messages": [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b", "reasoning_content": "think"},
                {"role": "user", "content": "c"}]}),
        ),
        ("tools", json!({"temperature": 0, "tools": tools})),
        ("tools_auto", json!({"tools": tools, "tool_choice": "auto"})),
        ("tools_none", json!({"tools": tools, "tool_choice": "none"})),
        (
            "tools_parallel_false",
            json!({"tools": tools, "parallel_tool_calls": false}),
        ),
        (
            "tool_turns",
            json!({"temperature": 0, "tools": tools, "messages": [
                {"role": "user", "content": "read a.txt"},
                {"role": "assistant", "content": null, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": "text"},
                {"role": "user", "content": "thanks"}]}),
        ),
        (
            "json_object",
            json!({"temperature": 0, "response_format": {"type": "json_object"}}),
        ),
        (
            "response_format_text",
            json!({"response_format": {"type": "text"}}),
        ),
        (
            "thinking_on",
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
        ),
        (
            "thinking_off",
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
        ),
        ("thinking_kwargs_empty", json!({"chat_template_kwargs": {}})),
        ("effort_low", json!({"reasoning_effort": "low"})),
        ("effort_medium", json!({"reasoning_effort": "medium"})),
        ("effort_xhigh", json!({"reasoning_effort": "xhigh"})),
        ("effort_none", json!({"reasoning_effort": "none"})),
        (
            "thinking_on_effort_low",
            json!({"chat_template_kwargs": {"enable_thinking": true}, "reasoning_effort": "low"}),
        ),
        (
            "thinking_off_effort_high",
            json!({"chat_template_kwargs": {"enable_thinking": false}, "reasoning_effort": "high"}),
        ),
        (
            "separate_reasoning_false",
            json!({"separate_reasoning": false}),
        ),
        ("top_k", json!({"top_k": 20})),
        ("min_p", json!({"min_p": 0.05})),
        ("seed", json!({"seed": 7})),
        (
            "penalties",
            json!({"frequency_penalty": 0.5, "presence_penalty": -0.5, "repetition_penalty": 1.1}),
        ),
        ("logit_bias", json!({"logit_bias": {"11": 1, "12": -1.5}})),
        ("logprobs", json!({"logprobs": true, "top_logprobs": 2})),
        ("stop_token_ids", json!({"stop_token_ids": [3, 1, 3]})),
        ("ignore_eos", json!({"ignore_eos": true})),
        (
            "thinking_budget",
            json!({"chat_template_kwargs": {"enable_thinking": true}, "thinking_budget": 64}),
        ),
        ("n_2", json!({"n": 2})),
        (
            "nulls",
            json!({"temperature": null, "stop": null, "tools": null, "chat_template_kwargs": null}),
        ),
    ];
    let mut bodies: Vec<(String, Value)> = patches
        .into_iter()
        .map(|(name, patch)| (name.to_owned(), body(patch)))
        .collect();
    for name in ["snippet", "snippet_off"] {
        bodies.push((format!("omp_{name}"), omp_fixture(name)));
    }
    bodies
}

fn corpus_outcomes() -> Map<String, Value> {
    let mut all = Map::new();
    for (policy_name, policy) in policies() {
        for (name, body) in corpus_bodies() {
            if let Ok(request) = validate(&policy, &body) {
                all.insert(
                    format!("{policy_name}/{name}"),
                    Value::Object(bytes_of(&request)),
                );
            }
        }
    }
    all
}

/// AL-0: every corpus request valid before ApiLenient validates to the same validated, normalized, backend and
/// sidecar bytes, on every phase (D8: identity is untouched).
#[test]
fn valid_today_bytes_identical() {
    let got = corpus_outcomes();
    if let Ok(path) = std::env::var("CERE_API_IDENTITY_RECORD") {
        std::fs::write(path, serde_json::to_vec_pretty(&got).unwrap()).unwrap();
        return;
    }
    let golden: Map<String, Value> = serde_json::from_slice(
        &std::fs::read(repo("tests/fixtures/api/identity_corpus.json")).unwrap(),
    )
    .unwrap();
    assert!(golden.len() >= 80, "only {} corpus entries", golden.len());
    for (name, want) in &golden {
        let have = got
            .get(name)
            .unwrap_or_else(|| panic!("{name}: no longer validates"));
        for (field, value) in want.as_object().unwrap() {
            assert_eq!(&have[field], value, "{name}: {field}");
        }
    }
    // AL-5 (D7): exactly the bodies over the old native cap of 2048 became valid (the recorded refusals were
    // `invalid_value` on `max_completion_tokens`); every other refusal stands.
    let extra: Vec<&str> = got
        .keys()
        .filter(|k| !golden.contains_key(*k))
        .map(String::as_str)
        .collect();
    assert_eq!(
        extra,
        [
            "native/max_completion_32768",
            "native/omp_snippet",
            "native/omp_snippet_off"
        ],
        "corpus bodies valid now but not when recorded"
    );
}

/// AL-0 (red before AL-1): each omp 18.12.6 wire body validates on a `thinking_toggle` profile with cap 32768.
#[test]
fn omp_wire_bodies_validate() {
    let policy = policy_of("sglang-t4-qwen38.toml");
    assert!(policy.thinking_toggle && policy.max_output_tokens == 32768);
    for level in OMP_LEVELS.into_iter().chain(["snippet", "snippet_off"]) {
        let body = omp_fixture(level);
        if let Err(e) = validate(&policy, &body) {
            panic!(
                "omp {level}: {} {} {:?} {}",
                e.status, e.code, e.param, e.message
            );
        }
    }
}

/// The validated request's byte surfaces, or a panic naming the body.
fn accepted(policy: &ChatPolicy, body: &Value) -> ChatRequest {
    validate(policy, body).unwrap_or_else(|e| {
        panic!(
            "{body}: {} {} {:?} {}",
            e.status, e.code, e.param, e.message
        )
    })
}

fn refused(policy: &ChatPolicy, body: &Value) -> cere_runtime::api_types::ApiError {
    match validate(policy, body) {
        Ok(_) => panic!("{body}: accepted"),
        Err(e) => e,
    }
}

fn sglang() -> ChatPolicy {
    policy_of("sglang-t4-qwen38.toml")
}

/// AL-1 (a): every omp level validates to the bytes of the compat-snippet body of the same thinking state.
#[test]
fn omp_bodies_render_as_the_snippet() {
    for (policy_name, policy) in [
        ("sglang", sglang()),
        ("native", policy_of("native-g0-spec.toml")),
    ] {
        if policy.max_output_tokens < 32768 {
            continue; // AL-5 raises the native cap; `shipped_native_configs_accept_omp_budget` guards it.
        }
        for level in OMP_LEVELS {
            let snippet = if level == "off" {
                "snippet_off"
            } else {
                "snippet"
            };
            let request = accepted(&policy, &omp_fixture(level));
            assert_eq!(
                bytes_of(&request),
                bytes_of(&accepted(&policy, &omp_fixture(snippet))),
                "{policy_name} {level}"
            );
            assert_eq!(
                request.mapped_params,
                [
                    "chat_template_kwargs.preserve_thinking",
                    "enable_thinking",
                    "preserve_thinking"
                ],
                "{level}"
            );
            assert!(request.ignored_params.is_empty());
        }
    }
}

/// AL-1/AL-4 (Q5): each alias validates to the bytes of its canonical spelling and is recorded as mapped.
#[test]
fn aliases_equal_their_canonical_spelling() {
    let policy = sglang();
    let cases = [
        (
            json!({"enable_thinking": true}),
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            vec!["enable_thinking"],
        ),
        (
            json!({"enable_thinking": false, "chat_template_kwargs": {"enable_thinking": false}}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            vec!["enable_thinking"],
        ),
        (
            json!({"chat_template_kwargs": {"thinking": true}}),
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            vec!["chat_template_kwargs.thinking"],
        ),
        (
            json!({"preserve_thinking": true}),
            json!({}),
            vec!["preserve_thinking"],
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}),
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            vec!["chat_template_kwargs.preserve_thinking"],
        ),
        (
            json!({"preserve_thinking": false, "chat_template_kwargs": {"enable_thinking": true}}),
            json!({"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": false}}),
            vec!["preserve_thinking"],
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}}),
            json!({"chat_template_kwargs": {"enable_thinking": true}, "reasoning_effort": "low"}),
            vec!["chat_template_kwargs.reasoning_effort"],
        ),
        (
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "low"}}),
            json!({"reasoning_effort": "low"}),
            vec!["chat_template_kwargs.reasoning_effort"],
        ),
        (
            json!({"messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}]}),
            json!({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]}),
            vec!["messages.role.developer"],
        ),
        (
            json!({"messages": [{"role": "user", "content": "a"}, {"role": "developer", "content": "s"},
                {"role": "user", "content": "b"}]}),
            json!({"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "s"},
                {"role": "user", "content": "b"}]}),
            vec!["messages.role.developer"],
        ),
    ];
    for (alias, canonical, mapped) in cases {
        let request = accepted(&policy, &body(alias.clone()));
        assert_eq!(
            bytes_of(&request),
            bytes_of(&accepted(&policy, &body(canonical))),
            "{alias}"
        );
        assert_eq!(request.mapped_params, mapped, "{alias}");
    }
}

/// AL-1 (b)-(e), AL-4 (c): conflicting pairs, bad alias types, the thinking_toggle gate and unread kwargs.
#[test]
fn alias_refusals() {
    let policy = sglang();
    let cases = [
        (
            json!({"enable_thinking": true, "chat_template_kwargs": {"enable_thinking": false}}),
            "conflicting_parameters",
            "enable_thinking",
            "different values",
        ),
        (
            json!({"chat_template_kwargs": {"thinking": true, "enable_thinking": false}}),
            "conflicting_parameters",
            "chat_template_kwargs.thinking",
            "different values",
        ),
        (
            json!({"enable_thinking": false, "chat_template_kwargs": {"thinking": true}}),
            "conflicting_parameters",
            "enable_thinking",
            "chat_template_kwargs.thinking (true)",
        ),
        (
            json!({"preserve_thinking": true, "chat_template_kwargs": {"preserve_thinking": false}}),
            "conflicting_parameters",
            "preserve_thinking",
            "different values",
        ),
        (
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "medium"}}),
            "conflicting_parameters",
            "chat_template_kwargs.reasoning_effort",
            "different values",
        ),
        (
            json!({"preserve_thinking": "yes"}),
            "invalid_value",
            "preserve_thinking",
            "must be a boolean",
        ),
        (
            json!({"chat_template_kwargs": {"preserve_thinking": 1}}),
            "invalid_value",
            "chat_template_kwargs.preserve_thinking",
            "must be a boolean",
        ),
        (
            json!({"enable_thinking": "on"}),
            "invalid_value",
            "enable_thinking",
            "must be a boolean",
        ),
        (
            json!({"chat_template_kwargs": {"thinking": null, "reasoning_effort": "max", "enable_thinking": true}}),
            "invalid_value",
            "reasoning_effort",
            "low, medium, xhigh only",
        ),
        (
            json!({"chat_template_kwargs": {"foo": 1}}),
            "unsupported_parameter",
            "chat_template_kwargs.foo",
            "is not read by this deployment's chat template",
        ),
    ];
    for (patch, code, param, message) in cases {
        let e = refused(&policy, &body(patch.clone()));
        assert_eq!(
            (e.status, e.code, e.param.as_deref()),
            (400, code, Some(param)),
            "{patch}: {}",
            e.message
        );
        assert!(e.message.contains(message), "{patch}: {}", e.message);
    }
    // (d) a profile without thinking_toggle refuses every spelling as M2.12 does.
    let mut plain = sglang();
    plain.thinking_toggle = false;
    for (patch, param) in [
        (
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            "chat_template_kwargs",
        ),
        (json!({"enable_thinking": true}), "enable_thinking"),
        (json!({"preserve_thinking": true}), "preserve_thinking"),
        (
            json!({"chat_template_kwargs": {"thinking": false}}),
            "chat_template_kwargs.thinking",
        ),
    ] {
        let e = refused(&plain, &body(patch.clone()));
        assert_eq!(
            (e.code, e.param.as_deref()),
            ("unsupported_parameter", Some(param)),
            "{patch}"
        );
        assert!(
            e.message.contains("does not declare thinking_toggle"),
            "{patch}: {}",
            e.message
        );
    }
}

/// AL-2 (d)/(e), D3: `preserve_thinking: false` (either spelling) is honoured: it enters `normalized()`, the SGLang
/// body's `chat_template_kwargs` and the sidecar frame; `true` stays the identity (absent) on all three.
#[test]
fn preserve_thinking_false_is_forwarded() {
    for (policy_name, policy) in policies() {
        for thinking in [json!({"enable_thinking": true}), json!({})] {
            let with = |preserve: Option<bool>| {
                let mut kwargs = thinking.clone();
                if let Some(p) = preserve {
                    kwargs["preserve_thinking"] = json!(p);
                }
                accepted(&policy, &body(json!({ "chat_template_kwargs": kwargs })))
            };
            let (absent, kept, dropped) = (with(None), with(Some(true)), with(Some(false)));
            assert_eq!(absent.normalized(), kept.normalized(), "{policy_name}");
            assert_ne!(absent.normalized(), dropped.normalized(), "{policy_name}");
            assert_eq!(
                dropped.normalized()["chat_template_kwargs"],
                json!({"preserve_thinking": false}),
                "{policy_name}"
            );
            assert!(absent.normalized().get("chat_template_kwargs").is_none());
            let rid = Rid::new(1, 2, 3);
            let kwargs_of = |r: &ChatRequest| -> Value {
                let body = serde_json::to_value(r.backend_body(&rid)).unwrap();
                body["chat_template_kwargs"].clone()
            };
            assert_eq!(kwargs_of(&absent), kwargs_of(&kept), "{policy_name}");
            let mut want = match kwargs_of(&absent) {
                Value::Null => json!({}),
                kwargs => kwargs,
            };
            want["preserve_thinking"] = json!(false);
            assert_eq!(kwargs_of(&dropped), want, "{policy_name} {thinking}");
            assert_eq!(absent.preserve_thinking, None);
            assert_eq!(kept.preserve_thinking, None);
            assert_eq!(dropped.preserve_thinking, Some(false));
            let frame: Value = serde_json::from_str(&sidecar_line(&dropped)).unwrap();
            assert_eq!(frame["preserve_thinking"], json!(false), "{policy_name}");
            let frame: Value = serde_json::from_str(&sidecar_line(&kept)).unwrap();
            assert!(frame.get("preserve_thinking").is_none(), "{policy_name}");
        }
    }
}

/// AL-3 (a)/(b): each inert field validates to the bytes of the body without it and is reported; a bad type is
/// 400 `invalid_value` naming the path.
#[test]
fn inert_fields_are_dropped_and_reported() {
    let policy = sglang();
    let cases = [
        (json!({"user": "u-1"}), "user"),
        (json!({"safety_identifier": "hash"}), "safety_identifier"),
        (json!({"metadata": {"run": "7", "lane": "a"}}), "metadata"),
        (json!({"metadata": {}}), "metadata"),
        (json!({"prompt_cache_key": "k"}), "prompt_cache_key"),
        (
            json!({"prompt_cache_retention": "24h"}),
            "prompt_cache_retention",
        ),
        (json!({"store": false}), "store"),
        (json!({"service_tier": "auto"}), "service_tier"),
        (json!({"service_tier": "default"}), "service_tier"),
        (json!({"modalities": ["text"]}), "modalities"),
    ];
    for (patch, path) in cases {
        let request = accepted(&policy, &body(patch.clone()));
        assert_eq!(
            bytes_of(&request),
            bytes_of(&accepted(&policy, &body(json!({})))),
            "{patch}"
        );
        assert_eq!(request.ignored_params, [path], "{patch}");
        assert!(request.mapped_params.is_empty());
    }
    let obfuscated = json!({"stream": true, "stream_options": {"include_usage": true, "include_obfuscation": false}});
    let request = accepted(&policy, &body(obfuscated));
    let plain = accepted(
        &policy,
        &body(json!({"stream": true, "stream_options": {"include_usage": true}})),
    );
    assert_eq!(bytes_of(&request), bytes_of(&plain));
    assert_eq!(
        request.ignored_params,
        ["stream_options.include_obfuscation"]
    );
    // All at once: sorted by the list, nulls absent and unreported.
    let request = accepted(
        &policy,
        &body(
            json!({"user": "u", "store": false, "metadata": {"a": "b"}, "safety_identifier": null}),
        ),
    );
    assert_eq!(request.ignored_params, ["metadata", "store", "user"]);

    for (patch, path) in [
        (json!({"user": 5}), "user"),
        (json!({"metadata": "x"}), "metadata"),
        (json!({"metadata": {"a": 1}}), "metadata"),
        (json!({"store": "no"}), "store"),
        (json!({"service_tier": 1}), "service_tier"),
        (json!({"modalities": "text"}), "modalities"),
        (json!({"prompt_cache_key": ["k"]}), "prompt_cache_key"),
        (
            json!({"stream": true, "stream_options": {"include_obfuscation": "x"}}),
            "stream_options.include_obfuscation",
        ),
    ] {
        let e = refused(&policy, &body(patch.clone()));
        assert_eq!(
            (e.status, e.code, e.param.as_deref()),
            (400, "invalid_value", Some(path)),
            "{patch}"
        );
    }
}

/// AL-4 (a): every reject row of the inventory keeps its status, type and param and names the remedy (D9).
#[test]
fn reject_rows_name_the_remedy() {
    let policy = sglang();
    let thinking_on = json!({"enable_thinking": true});
    let rows: Vec<(Value, &str, &str, &str)> = vec![
        (
            json!({"store": true}),
            "unsupported_parameter",
            "store",
            "store: false is accepted and ignored",
        ),
        (
            json!({"service_tier": "priority"}),
            "unsupported_parameter",
            "service_tier",
            "'auto' and 'default' are accepted and ignored",
        ),
        (
            json!({"service_tier": "flex"}),
            "unsupported_parameter",
            "service_tier",
            "'auto' and 'default'",
        ),
        (
            json!({"service_tier": "scale"}),
            "unsupported_parameter",
            "service_tier",
            "'auto' and 'default'",
        ),
        (
            json!({"modalities": ["audio"]}),
            "unsupported_parameter",
            "modalities",
            "[\"text\"] is accepted",
        ),
        (
            json!({"modalities": ["text", "audio"]}),
            "unsupported_parameter",
            "modalities",
            "outputs text only",
        ),
        (
            json!({"audio": {"voice": "x", "format": "wav"}}),
            "unsupported_parameter",
            "audio",
            "omit audio",
        ),
        (
            json!({"functions": []}),
            "unsupported_parameter",
            "functions",
            "send tools",
        ),
        (
            json!({"function_call": "auto"}),
            "unsupported_parameter",
            "function_call",
            "send tool_choice",
        ),
        (
            json!({"prediction": {"type": "content", "content": "x"}}),
            "unsupported_parameter",
            "prediction",
            "omit prediction",
        ),
        (
            json!({"verbosity": "low"}),
            "unsupported_parameter",
            "verbosity",
            "max_completion_tokens",
        ),
        (
            json!({"web_search_options": {}}),
            "unsupported_parameter",
            "web_search_options",
            "define a search tool",
        ),
        (
            json!({"reasoning": {"effort": "low"}}),
            "unsupported_parameter",
            "reasoning",
            "send reasoning_effort",
        ),
        (
            json!({"thinking": {"type": "enabled"}}),
            "unsupported_parameter",
            "thinking",
            "send chat_template_kwargs.enable_thinking",
        ),
        (
            json!({"deterministic": true}),
            "unsupported_parameter",
            "deterministic",
            "reserved CERE extension (tasks/batch-concurrency-spec.md BC-D6)",
        ),
        (
            json!({"seed": 7}),
            "unsupported_parameter",
            "seed",
            "(sampling_params)",
        ),
        (
            json!({"chat_template_kwargs": {"foo": 1}}),
            "unsupported_parameter",
            "chat_template_kwargs.foo",
            "it reads enable_thinking",
        ),
        (
            json!({"messages": [{"role": "critic", "content": "a"}]}),
            "unsupported_parameter",
            "messages[0].role",
            "use system (or developer), user, assistant or tool",
        ),
        (
            json!({"max_completion_tokens": 32769}),
            "invalid_value",
            "max_completion_tokens",
            "exceeds the maximum of 32768",
        ),
    ];
    let efforts = ["high", "minimal", "max"].map(|effort| {
        let mut patch = thinking_on.clone();
        patch["reasoning_effort"] = json!(effort);
        (
            patch,
            "invalid_value",
            "reasoning_effort",
            "accepts reasoning_effort low, medium, xhigh only",
        )
    });
    for (patch, code, param, remedy) in rows.into_iter().chain(efforts) {
        let e = refused(&policy, &body(patch.clone()));
        assert_eq!(
            (e.status, e.error_type, e.code, e.param.as_deref()),
            (400, "invalid_request_error", code, Some(param)),
            "{patch}: {}",
            e.message
        );
        assert!(e.message.contains(remedy), "{patch}: {}", e.message);
    }
}

/// AL-4 (b): a reserved CERE extension name is refused on every surface, Anthropic included (dropped before).
#[test]
fn reserved_extensions_refused_on_every_surface() {
    let policy = policy_of("native-g0-spec.toml");
    let anthropic = json!({"model": MODEL, "max_tokens": 16, "deterministic": true,
        "messages": [{"role": "user", "content": "hi"}]});
    let e =
        cere_runtime::anthropic::convert_request(&serde_json::to_vec(&anthropic).unwrap(), &policy)
            .err()
            .expect("anthropic deterministic refused");
    assert_eq!(
        (e.status, e.code.as_str(), e.param.as_deref()),
        (400, "unsupported_parameter", Some("deterministic"))
    );
    assert!(
        e.message.contains("reserved CERE extension"),
        "{}",
        e.message
    );
    let responses = json!({"model": MODEL, "input": "hi", "deterministic": true});
    let Err(e) =
        cere_runtime::responses::convert(&serde_json::to_vec(&responses).unwrap(), &policy)
    else {
        panic!("responses deterministic refused");
    };
    assert_eq!(
        (e.status, e.code, e.param.as_deref()),
        (400, "unsupported_parameter", Some("deterministic"))
    );
    assert!(
        e.message.contains("reserved CERE extension"),
        "{}",
        e.message
    );
    let e = refused(&policy, &body(json!({"deterministic": true})));
    assert_eq!(
        (e.code, e.param.as_deref()),
        ("unsupported_parameter", Some("deterministic"))
    );
}

/// AL-5 (D7, Q3): every shipped native config with a context of at least 65,536 tokens accepts omp's 32,768-token
/// budget; one token over its cap is still 400 (no clamp). Every such config validates omp's `high` fixture.
#[test]
fn shipped_native_configs_accept_omp_budget() {
    let high = omp_fixture("high");
    let mut checked = Vec::new();
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(repo("configs/runtime")).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if !(name.starts_with("native-") && name.ends_with(".toml")) {
            continue;
        }
        let config = config(&name);
        let context = config.deployments[0]
            .native
            .as_ref()
            .map_or(0, |native| native.max_context_tokens);
        let policy = ChatPolicy::from_config(&config, 0).unwrap();
        let model = policy.served_model_name.clone();
        if context < 65_536 {
            assert_eq!(policy.max_output_tokens, 2048, "{name}");
            continue;
        }
        // omp names the deployment's served model (Inkling-2-Mini on the Inkling config).
        let mut omp = high.clone();
        omp["model"] = json!(model);
        if let Err(e) = validate(&policy, &omp) {
            failures.push(format!("{name}: {} {:?} {}", e.code, e.param, e.message));
            continue;
        }
        let cap = if name.ends_with("-eval.toml") {
            131_072
        } else {
            32_768.min(context)
        };
        assert_eq!(policy.max_output_tokens, cap, "{name}");
        let over =
            body(json!({"model": model, "max_completion_tokens": policy.max_output_tokens + 1}));
        let e = refused(&policy, &over);
        assert_eq!(
            (e.code, e.param.as_deref()),
            ("invalid_value", Some("max_completion_tokens")),
            "{name}"
        );
        checked.push(name);
    }
    assert!(
        failures.is_empty(),
        "{} configs refuse omp's budget:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(checked.len(), 39, "{checked:?}");
}
