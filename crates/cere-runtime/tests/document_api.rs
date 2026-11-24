//! DocInput wire contract (doc-input design note DI-01..DI-03, DI-05): Anthropic `document` blocks on
//! `/v1/messages` and OpenAI `file` content parts on `/v1/chat/completions` become the rendered document text;
//! everything CERE cannot read is a 400 naming the block. Requests without documents convert as before (the
//! goldens in `server_anthropic.rs` and `back_validate.rs` stay unchanged).
mod document_support;

use cere_runtime::anthropic;
use cere_runtime::config::RuntimeConfig;
use cere_runtime::surface::OwnedError;
use cere_runtime::validate::{ChatPolicy, Content};
use document_support::{base64, simple, text_pdf, HELVETICA};
use serde_json::{json, Value};

fn policy() -> ChatPolicy {
    let config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/sglang-t4-qwen38.toml"),
    )
    .unwrap();
    ChatPolicy::from_config(&config, 0).unwrap()
}

fn convert(body: &Value) -> Result<Value, OwnedError> {
    let (chat, _) = anthropic::convert_request(&serde_json::to_vec(body).unwrap(), &policy())?;
    Ok(serde_json::from_slice(&chat).unwrap())
}

fn anthropic_user(content: Value) -> Value {
    json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": content}]})
}

fn pdf_block(bytes: &[u8]) -> Value {
    json!({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64(bytes)}})
}

const TWO_PAGES: &str =
    "<document media_type=\"application/pdf\" pages=\"2\">\n<page number=\"1\">\nAlpha\n\
                         </page>\n<page number=\"2\">\nBeta\n</page>\n</document>\n";

#[test]
fn an_anthropic_pdf_block_becomes_page_text() {
    let pdf = text_pdf(&["Alpha", "Beta"]);
    let chat = convert(&anthropic_user(json!([
        {"type": "text", "text": "Summarize:"},
        pdf_block(&pdf),
        {"type": "text", "text": "Thanks."}
    ])))
    .unwrap();
    assert_eq!(
        chat["messages"],
        json!([{"role": "user", "content": format!("Summarize:\n{TWO_PAGES}Thanks.")}])
    );
}

/// a coding agent's Read tool: `[{text: "PDF file read: ..."}, {document}]` inside a tool_result.
#[test]
fn a_tool_result_document_joins_the_tool_message() {
    let pdf = text_pdf(&["Alpha", "Beta"]);
    let body = json!({"model": "m", "max_tokens": 8, "messages": [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a.pdf"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "PDF file read: /a.pdf (1 KB)"},
            pdf_block(&pdf)
        ]}]}
    ]});
    let chat = convert(&body).unwrap();
    assert_eq!(
        chat["messages"][2],
        json!({"role": "tool", "tool_call_id": "t1", "content": format!("PDF file read: /a.pdf (1 KB)\n{TWO_PAGES}")})
    );
}

#[test]
fn text_and_content_sources_render_as_text_documents() {
    let chat = convert(&anthropic_user(json!([
        {"type": "document", "title": "notes", "context": "from Slack", "source": {"type": "text", "media_type": "text/plain", "data": "one\ntwo"}},
        {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}
    ])))
    .unwrap();
    assert_eq!(
        chat["messages"][0]["content"],
        "<document title=\"notes\" media_type=\"text/plain\">\n<context>\nfrom Slack\n</context>\none\ntwo\n\
         </document>\n<document media_type=\"text/plain\">\na\nb\n</document>\n"
    );
    let base64_text = convert(&anthropic_user(json!([
        {"type": "document", "source": {"type": "base64", "media_type": "text/plain", "data": base64(b"plain")}}
    ])))
    .unwrap();
    assert_eq!(
        base64_text["messages"][0]["content"],
        "<document media_type=\"text/plain\">\nplain\n</document>\n"
    );
}

#[test]
fn anthropic_document_refusals_name_the_block() {
    let encrypted = simple(
        &[b"BT /F1 12 Tf (x) Tj ET"],
        HELVETICA,
        false,
        "/Encrypt << /Filter /Standard >>",
    );
    let loc = "messages[0].content[0]";
    let cases = [
        (
            json!({"type": "document", "citations": {"enabled": true}, "source": {"type": "text", "data": "x"}}),
            "unsupported_parameter",
            format!("{loc}.citations"),
        ),
        (
            json!({"type": "document", "source": {"type": "url", "url": "https://example.invalid/a.pdf"}}),
            "unsupported_parameter",
            format!("{loc}.source"),
        ),
        (
            json!({"type": "document", "source": {"type": "file", "file_id": "file_1"}}),
            "unsupported_parameter",
            format!("{loc}.source"),
        ),
        (
            json!({"type": "document", "source": {"type": "base64", "media_type": "application/msword", "data": "QQ=="}}),
            "unsupported_parameter",
            format!("{loc}.source"),
        ),
        (
            pdf_block(&encrypted),
            "unsupported_parameter",
            format!("{loc}.source"),
        ),
        (
            json!({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "not base64"}}),
            "invalid_value",
            format!("{loc}.source"),
        ),
        (
            json!({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64(b"plain text")}}),
            "invalid_value",
            format!("{loc}.source"),
        ),
        (
            json!({"type": "document", "source": {"type": "content", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QQ=="}}]}}),
            "unsupported_parameter",
            format!("{loc}.source.content[0]"),
        ),
    ];
    for (block, code, param) in cases {
        let err = convert(&anthropic_user(json!([block]))).unwrap_err();
        assert_eq!(
            (err.status, err.code.as_str(), err.param.as_deref()),
            (400, code, Some(param.as_str())),
            "{block}: {}",
            err.message
        );
    }
    // citations explicitly off is accepted.
    convert(&anthropic_user(json!([
        {"type": "document", "citations": {"enabled": false}, "source": {"type": "text", "data": "x"}}
    ])))
    .unwrap();
}

#[test]
fn anthropic_document_shapes_are_checked() {
    let cases = [
        (json!({"type": "document"}), "missing_parameter"),
        (json!({"type": "document", "source": "x"}), "invalid_value"),
        (
            json!({"type": "document", "source": {"type": "pdf"}}),
            "invalid_value",
        ),
        (
            json!({"type": "document", "source": {"type": "base64", "data": "QQ=="}}),
            "missing_parameter",
        ),
        (
            json!({"type": "document", "source": {"type": "text", "data": 1}}),
            "invalid_value",
        ),
        (
            json!({"type": "document", "title": 3, "source": {"type": "text", "data": "x"}}),
            "invalid_value",
        ),
        (
            json!({"type": "document", "citations": {"enabled": "yes"}, "source": {"type": "text", "data": "x"}}),
            "invalid_value",
        ),
    ];
    for (block, code) in cases {
        let err = convert(&anthropic_user(json!([block]))).unwrap_err();
        assert_eq!(
            (err.status, err.code.as_str()),
            (400, code),
            "{block}: {}",
            err.message
        );
    }
    let err = convert(&anthropic_user(
        json!([{"type": "document", "source": {"type": "pdf"}}]),
    ))
    .unwrap_err();
    assert!(
        err.message
            .contains("'base64', 'text', 'content', 'url' or 'file'"),
        "{}",
        err.message
    );
    // An unknown block type keeps SGLang's message: `document` is not added to its tag list.
    let err = convert(&anthropic_user(json!([{"type": "video"}]))).unwrap_err();
    assert!(!err.message.contains("document"), "{}", err.message);
}

#[test]
fn the_page_limit_spans_the_anthropic_history() {
    let sixty: Vec<String> = (0..60).map(|i| format!("p{i}")).collect();
    let refs: Vec<&str> = sixty.iter().map(String::as_str).collect();
    let pdf = text_pdf(&refs);
    let body = json!({"model": "m", "max_tokens": 8, "messages": [
        {"role": "user", "content": [pdf_block(&pdf)]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [pdf_block(&pdf)]}
    ]});
    let err = convert(&body).unwrap_err();
    assert_eq!(
        (err.code.as_str(), err.param.as_deref()),
        ("invalid_value", Some("messages[2].content[0].source"))
    );
    assert!(
        err.message.contains("at most 100 PDF pages"),
        "{}",
        err.message
    );
}

// ---- Chat completions: the OpenAI `file` content part ----

fn chat(messages: Value) -> Value {
    json!({"model": policy().served_model_name, "messages": messages})
}

fn file_part(media: &str, bytes: &[u8], filename: Option<&str>) -> Value {
    let mut file = json!({"file_data": format!("data:{media};base64,{}", base64(bytes))});
    if let Some(name) = filename {
        file["filename"] = json!(name);
    }
    json!({"type": "file", "file": file})
}

fn joined(request: &cere_runtime::validate::ChatRequest, index: usize) -> String {
    match &request.messages[index].joined().unwrap().content {
        Some(Content::Text(text)) => text.clone(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_chat_file_part_becomes_a_text_part() {
    let pdf = text_pdf(&["Alpha", "Beta"]);
    let request = policy()
        .validate(&chat(json!([{"role": "user", "content": [
            {"type": "text", "text": "Summarize:"},
            file_part("application/pdf", &pdf, Some("a.pdf")),
            file_part("text/plain", b"tail", None)
        ]}])))
        .unwrap();
    assert_eq!(
        joined(&request, 0),
        format!(
            "Summarize:\n<document title=\"a.pdf\" media_type=\"application/pdf\" pages=\"2\">\n<page number=\"1\">\n\
             Alpha\n</page>\n<page number=\"2\">\nBeta\n</page>\n</document>\n\
             <document media_type=\"text/plain\">\ntail\n</document>\n"
        )
    );
}

#[test]
fn a_tool_message_may_carry_text_and_file_parts() {
    let pdf = text_pdf(&["Alpha", "Beta"]);
    let call =
        json!({"id": "t1", "type": "function", "function": {"name": "Read", "arguments": "{}"}});
    let request = policy()
        .validate(&chat(json!([
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": Value::Null, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "t1", "content": [
                {"type": "text", "text": "PDF file read: /a.pdf (1 KB)"},
                file_part("application/pdf", &pdf, None)
            ]}
        ])))
        .unwrap();
    assert_eq!(
        request.messages[2].content,
        Some(Content::Text(format!(
            "PDF file read: /a.pdf (1 KB)\n{TWO_PAGES}"
        )))
    );
}

#[test]
fn chat_file_refusals_name_their_path() {
    let at = "messages[0].content[0]";
    let encrypted = simple(
        &[b"BT /F1 12 Tf (x) Tj ET"],
        HELVETICA,
        false,
        "/Encrypt << /Filter /Standard >>",
    );
    let cases = [
        (
            json!({"type": "file", "file": {"file_id": "file-1"}}),
            "unsupported_parameter",
            format!("{at}.file.file_id"),
        ),
        (
            json!({"type": "file"}),
            "missing_required_parameter",
            format!("{at}.file"),
        ),
        (
            json!({"type": "file", "file": {}}),
            "missing_required_parameter",
            format!("{at}.file.file_data"),
        ),
        (
            json!({"type": "file", "file": {"file_data": "QQ==", "extra": 1}}),
            "unsupported_parameter",
            format!("{at}.file.extra"),
        ),
        (
            json!({"type": "file", "file": {"file_data": "data:application/msword;base64,QQ=="}}),
            "unsupported_parameter",
            format!("{at}.file.file_data"),
        ),
        (
            json!({"type": "file", "file": {"file_data": "data:application/pdf;base64,QQ"}}),
            "invalid_value",
            format!("{at}.file.file_data"),
        ),
        (
            file_part("application/pdf", &encrypted, None),
            "unsupported_parameter",
            format!("{at}.file.file_data"),
        ),
        (
            file_part("application/pdf", b"%PDF-1.4\nno objects", None),
            "invalid_value",
            format!("{at}.file.file_data"),
        ),
        (
            file_part("text/plain", b"\xff\xfe", None),
            "invalid_value",
            format!("{at}.file.file_data"),
        ),
    ];
    for (part, code, param) in cases {
        let err = policy()
            .validate(&chat(json!([{"role": "user", "content": [part]}])))
            .unwrap_err();
        assert_eq!(
            (err.status, err.code, err.param.as_deref()),
            (400, code, Some(param.as_str())),
            "{part}: {}",
            err.message
        );
    }
    // Arrays stay refused in system and assistant messages.
    let err = policy()
        .validate(&chat(json!([
            {"role": "system", "content": [{"type": "text", "text": "s"}]},
            {"role": "user", "content": "u"}
        ])))
        .unwrap_err();
    assert_eq!(err.param.as_deref(), Some("messages[0].content"));
}

#[test]
fn the_page_limit_spans_the_chat_history() {
    let sixty: Vec<String> = (0..60).map(|i| format!("p{i}")).collect();
    let refs: Vec<&str> = sixty.iter().map(String::as_str).collect();
    let pdf = text_pdf(&refs);
    let err = policy()
        .validate(&chat(json!([
            {"role": "user", "content": [file_part("application/pdf", &pdf, None)]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [file_part("application/pdf", &pdf, None)]}
        ])))
        .unwrap_err();
    assert_eq!(
        (err.code, err.param.as_deref()),
        (
            "invalid_value",
            Some("messages[2].content[0].file.file_data")
        )
    );
}
