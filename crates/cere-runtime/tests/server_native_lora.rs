//! Track LoRA (Wave P): the vLLM adapter routes and per-request adapters over the CPU toy executor. The toy turns an
//! adapter into one offset (1 + FNV-1a 64 over rank, scale bits and per target its id, A and B bytes, mod vocab - 1)
//! added to every id a bound session scores, so an adapter request's tokens are computed here exactly; a base request
//! is the plain toy sequence (D9). The cases serve the vision artifact (the A4 text config: 48 layers, hidden 2560,
//! every fourth layer full attention, shared expert intermediate 640; the text fixture's has no full-attention layer
//! and no shared expert): the adapters adapt layer 0's shared expert `up_proj` and layer 3's `q_proj`.
mod native_server;
mod native_support;
mod support;

use std::path::Path;

use native_server::{Case, Images, Native, SnapshotProfile, TOKEN};
use native_support::{toy_library_prestaging, toy_next, Tmp};
use serde_json::{json, Value};
use support::{http, Event, SseReader, MODEL};

const SENTINEL: &str = "cere-native-sentinel-prompt";
/// Prompt length and last character (`server_native.rs` TEXT).
const TEXT: (usize, char) = (716, '1');
/// The A4 vocabulary of the vision artifact: the toy's next-token modulus.
const VOCAB: u32 = 248_320;
const RANK: u32 = 8;
const ALPHA: f32 = 16.0;
/// `(target id, PEFT module path, A shape [r, in], B shape [out, r])`: layer 0 * 8 + CERE_LORA_SHARED_UP, layer 3 * 8
/// + CERE_LORA_Q_PROJ (q's out is heads 24 x head_dim 256 x 2), ascending target ids as the registry passes them.
const TARGETS: [(u32, &str, u64, u64); 2] = [
    (5, "layers.0.mlp.shared_expert.up_proj", 2560, 640),
    (24, "layers.3.self_attn.q_proj", 2560, 12288),
];

fn prompt((len, last): (usize, char)) -> String {
    let mut s = SENTINEL.to_owned();
    while s.len() < len - 2 {
        s.push('x');
    }
    s.push(last);
    s
}

fn body(model: &str, max_tokens: u64, stream: bool) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt(TEXT)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    })
}

/// `n` BF16 values `1 + k/64` (finite, distinct per seed and salt).
fn bf16_bytes(n: u64, seed: u64, salt: u64) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let v = 1.0f32 + ((i * seed + salt) % 64) as f32 / 64.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// `(A, B)` bytes of target `i` of adapter `seed`.
fn pair(seed: u64, i: usize) -> (Vec<u8>, Vec<u8>) {
    let (_, _, input, output) = TARGETS[i];
    let r = u64::from(RANK);
    (
        bf16_bytes(r * input, seed, 2 * i as u64),
        bf16_bytes(output * r, seed, 2 * i as u64 + 1),
    )
}

/// A safetensors file of `(name, dtype, shape, bytes)`.
fn safetensors(tensors: &[(String, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend_from_slice(bytes);
    }
    let mut text = serde_json::to_vec(&header).unwrap();
    while text.len() % 8 != 0 {
        text.push(b' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(text);
    out.extend(data);
    out
}

/// Writes PEFT adapter `seed` as `<root>/<dir>`.
fn write_adapter(root: &Path, dir: &str, seed: u64) {
    let dir = root.join(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let config = json!({
        "peft_type": "LORA", "r": RANK, "lora_alpha": ALPHA, "bias": "none",
        "target_modules": ["up_proj", "q_proj"], "base_model_name_or_path": "Qwen3.8-Flash-Next",
    });
    std::fs::write(dir.join("adapter_config.json"), config.to_string()).unwrap();
    let r = u64::from(RANK);
    let mut tensors = Vec::new();
    for (i, (_, module, input, output)) in TARGETS.iter().enumerate() {
        let (a, b) = pair(seed, i);
        let prefix = format!("base_model.model.model.{module}");
        tensors.push((
            format!("{prefix}.lora_A.weight"),
            "BF16",
            vec![r, *input],
            a,
        ));
        tensors.push((
            format!("{prefix}.lora_B.weight"),
            "BF16",
            vec![*output, r],
            b,
        ));
    }
    std::fs::write(dir.join("adapter_model.safetensors"), safetensors(&tensors)).unwrap();
}

/// The toy offset of adapter `seed` (toy_device.cpp `lora_load`).
fn toy_offset(seed: u64) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for b in bytes {
            h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(&RANK.to_le_bytes());
    mix(&(ALPHA / RANK as f32).to_bits().to_le_bytes());
    for (i, (target, ..)) in TARGETS.iter().enumerate() {
        let (a, b) = pair(seed, i);
        mix(&target.to_le_bytes());
        mix(&a);
        mix(&b);
    }
    1 + (h % u64::from(VOCAB - 1)) as u32
}

/// What the toy commits for the TEXT prompt within `n` tokens with offset `off` (0 = the base), up to a stop id.
fn toy_output(off: u32, n: usize) -> Vec<u32> {
    let (mut last, mut len) = (TEXT.1 as u32, TEXT.0 as u64);
    let mut ids = Vec::new();
    for _ in 0..n {
        last = (toy_next(last, len, VOCAB) + off) % VOCAB;
        len += 1;
        ids.push(last);
        if [6, 7].contains(&last) {
            break;
        }
    }
    ids
}

fn lora_case(root: &Path, preload: Value) -> Case {
    Case {
        lora: json!({
            "max_adapters": 2, "max_rank": 16, "adapter_root": root.display().to_string(), "preload": preload,
        }),
        images: Images {
            artifact: true,
            ..Images::default()
        },
        capabilities_patch: json!({"token_controls": ["stop_token_ids"], "vocab_size": VOCAB}),
        ..Case::matching()
    }
}

async fn post(native: &Native, path: &str, body: &Value) -> (u16, String) {
    let response = http()
        .post(native.cered.url(path))
        .bearer_auth(TOKEN)
        .json(body)
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.text().await.unwrap())
}

async fn load(native: &Native, name: &str, path: &str) -> (u16, String) {
    post(
        native,
        "/v1/load_lora_adapter",
        &json!({"lora_name": name, "lora_path": path}),
    )
    .await
}

async fn unload(native: &Native, name: &str) -> (u16, String) {
    post(
        native,
        "/v1/unload_lora_adapter",
        &json!({"lora_name": name}),
    )
    .await
}

async fn model_ids(native: &Native) -> Vec<String> {
    let (status, models) = native.get("/v1/models").await;
    assert_eq!(status, 200, "{models}");
    models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_owned())
        .collect()
}

/// A unary chat for `model` stopping at `stop` (the toy's ids past the fixture tokenizer's 3 decode to nothing, so
/// the cases read ids through `stop_token_ids`: `matched_stop` and `completion_tokens` pin the sequence).
async fn complete(native: &Native, model: &str, stop: u32) -> (u16, Value) {
    let mut request = body(model, 4, false);
    request["stop_token_ids"] = json!([stop]);
    let response = native.chat(&request).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// The second toy id of the TEXT prompt under offset `off` (neither id a stop id).
fn second(off: u32) -> u32 {
    let ids = toy_output(off, 2);
    assert_eq!(ids.len(), 2, "{ids:?}");
    ids[1]
}

/// `model` generates the toy sequence of offset `off`: it stops at that sequence's second id, and runs to the
/// length limit when stopped at the sequence of `other` (a different offset).
async fn assert_serves(native: &Native, model: &str, off: u32, other: u32) {
    let (status, out) = complete(native, model, second(off)).await;
    assert_eq!(status, 200, "{model}: {out}");
    assert_eq!(out["model"], model, "{out}");
    let choice = &out["choices"][0];
    assert_eq!(
        (
            &choice["finish_reason"],
            &choice["matched_stop"],
            &out["usage"]["completion_tokens"]
        ),
        (&json!("stop"), &json!(second(off)), &json!(2)),
        "{model}: {out}"
    );
    let (status, out) = complete(native, model, second(other)).await;
    assert_eq!(status, 200, "{model}: {out}");
    assert_eq!(
        out["choices"][0]["finish_reason"], "length",
        "{model}: {out}"
    );
}

/// Load, list, serve, unload: an adapter request's tokens carry its offset, two adapters differ, the base request is
/// the plain toy sequence before, between and after; the vLLM status codes for the refusals.
#[tokio::test]
async fn adapters_load_serve_and_unload_and_the_base_is_unchanged() {
    let root = Tmp::new("lora-root");
    let native = Native::ready(lora_case(root.path(), json!([]))).await;
    for (dir, seed) in [("a1", 1), ("a2", 2), ("a3", 3)] {
        write_adapter(root.path(), dir, seed);
    }
    let [one, two, three] = [1, 2, 3].map(toy_offset);
    assert_serves(&native, MODEL, 0, one).await;
    assert_eq!(model_ids(&native).await, [MODEL]);

    let (status, text) = load(&native, "a1", "a1").await;
    assert_eq!(
        (status, text.as_str()),
        (200, "Success: LoRA adapter 'a1' added successfully.")
    );
    assert_eq!(model_ids(&native).await, [MODEL, "a1"]);
    let (status, card) = native.get("/v1/models/a1").await;
    assert_eq!(status, 200, "{card}");
    assert_eq!(card["parent"], MODEL);
    assert_eq!(
        card["root"],
        root.path()
            .canonicalize()
            .unwrap()
            .join("a1")
            .display()
            .to_string()
    );

    assert_serves(&native, "a1", one, 0).await;

    // A path outside the root and a missing directory are refused while a slot is free.
    for (path, want) in [
        ("../", "outside the adapter root"),
        ("missing", "lora_path missing"),
    ] {
        let (status, text) = load(&native, "a2", path).await;
        assert_eq!(status, 400, "{path}: {text}");
        assert!(text.contains(want), "{path}: {text}");
    }
    // An absolute path inside the root loads too.
    let (status, text) = load(&native, "a2", &root.join("a2").display().to_string()).await;
    assert_eq!(status, 200, "{text}");
    assert_serves(&native, "a2", two, one).await;

    // D9: the base request is bit-identical with adapters loaded.
    assert_serves(&native, MODEL, 0, two).await;

    // Refusals: a duplicate, a served model's name, a bad name, a full pool (max_adapters 2), no lora_path.
    for (name, path, want) in [
        ("a1", "a3", "already loaded"),
        (MODEL, "a3", "is a served model"),
        ("bad name", "a3", "lora_name must match"),
        ("a3", "a3", "no free adapter slot"),
    ] {
        let (status, text) = load(&native, name, path).await;
        assert_eq!(status, 400, "{name} {path}: {text}");
        assert!(text.contains(want), "{name} {path}: {text}");
    }
    let (status, text) = post(
        &native,
        "/v1/load_lora_adapter",
        &json!({"lora_name": "a3"}),
    )
    .await;
    assert_eq!(status, 400, "{text}");

    let (status, text) = unload(&native, "a1").await;
    assert_eq!(
        (status, text.as_str()),
        (200, "Success: LoRA adapter 'a1' removed successfully.")
    );
    let (status, text) = unload(&native, "a1").await;
    assert_eq!(status, 404, "{text}");
    let (status, gone) = complete(&native, "a1", second(one)).await;
    assert_eq!(status, 404, "{gone}");
    assert_eq!(gone["error"]["code"], "model_not_found", "{gone}");
    assert_eq!(model_ids(&native).await, [MODEL, "a2"]);

    // The freed slot takes the next adapter.
    let (status, text) = load(&native, "a3", "a3").await;
    assert_eq!(status, 200, "{text}");
    assert_serves(&native, "a3", three, one).await;
}

/// A preloaded adapter is listed at ready; unload is 409 while a request streams with it, 200 once it ended.
#[tokio::test]
async fn preload_and_unload_is_busy_while_a_request_holds_the_adapter() {
    let root = Tmp::new("lora-root");
    write_adapter(root.path(), "a1", 1);
    let case = Case {
        device_id: "toy,hold_after=2".to_owned(),
        ..lora_case(
            root.path(),
            json!([{"name": "pre", "path": root.join("a1").display().to_string()}]),
        )
    };
    let native = Native::ready(case).await;
    assert_eq!(model_ids(&native).await, [MODEL, "pre"]);

    let response = native.chat(&body("pre", 16, true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut reader = SseReader::new(response);
    // The toy holds after its second token.
    let mut tokens = 0;
    while tokens < 1 {
        match reader.next().await {
            Some(Event::Data(v)) if v.pointer("/choices/0/delta/content").is_some() => tokens += 1,
            Some(Event::Data(v)) => assert!(v.get("error").is_none(), "{v}"),
            other => panic!("the stream ended before the hold: {other:?}"),
        }
    }
    let (status, text) = unload(&native, "pre").await;
    assert_eq!(status, 409, "{text}");
    assert!(text.contains("in use"), "{text}");

    let cancel = http()
        .post(native.cered.url(&format!("/v1/cere/requests/{rid}/cancel")))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    reader.rest().await;
    assert_eq!(native.terminal(&rid).await["state"], "cancelled");
    let mut unloaded = (0, String::new());
    for _ in 0..200 {
        unloaded = unload(&native, "pre").await;
        if unloaded.0 != 409 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert_eq!(unloaded.0, 200, "{}", unloaded.1);
    assert_eq!(model_ids(&native).await, [MODEL]);
}

/// A library without the LoRA symbols (the pre-staging toy) refuses a lora config at load, naming them.
#[tokio::test]
async fn a_library_without_lora_refuses_the_lora_config() {
    let root = Tmp::new("lora-root");
    let library = toy_library_prestaging();
    let case = Case {
        library_pin: native_server::file_sha256(&library),
        library,
        ..lora_case(root.path(), json!([]))
    };
    let native = Native::start(case).await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 503, "{body}");
    assert!(
        body.to_string().contains("cere_model_lora_reserve"),
        "{body}"
    );
}

/// Without `[deployments.native.lora]` the adapter routes do not exist.
#[tokio::test]
async fn without_lora_the_routes_are_absent() {
    let native = Native::ready(Case::matching()).await;
    let (status, _) = load(&native, "a1", "/tmp").await;
    assert_eq!(status, 404);
    let (status, _) = unload(&native, "a1").await;
    assert_eq!(status, 404);
}

/// Track LoRA x Sessions SE-09: an adapter session is never snapshotted, so it cannot grow. With
/// `reserve_output_tokens` 64 a base request would hold 832 and grow at its 117th token; an adapter request holds its
/// whole budget (960 for 200 tokens) from the start and runs to its toy sequence with no grow and no preemption (a
/// grow would wait forever on the refused snapshot).
#[tokio::test]
async fn an_adapter_request_holds_its_whole_budget_under_reserve() {
    let root = Tmp::new("lora-root");
    write_adapter(root.path(), "a1", 1);
    let case = Case {
        snapshot: Some(SnapshotProfile {
            preempt: true,
            preempt_min_tokens: 512,
            inflight: 1,
            preempt_every: 0,
            rows: 1,
            parked: 1,
            reserve: 64,
        }),
        ..lora_case(
            root.path(),
            json!([{"name": "pre", "path": root.join("a1").display().to_string()}]),
        )
    };
    let native = Native::ready(case).await;
    let expected = toy_output(toy_offset(1), 200);
    assert!(
        expected.len() > 116,
        "the sequence stops before the reserve is used up: {expected:?}"
    );
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        native.chat(&body("pre", 200, false)).send(),
    )
    .await
    .expect("the adapter request ends (no grow wait)")
    .unwrap();
    assert_eq!(response.status(), 200);
    let out: Value = response.json().await.unwrap();
    assert_eq!(
        out["usage"]["completion_tokens"],
        json!(expected.len()),
        "{out}"
    );
    assert_eq!((native.grows().await, native.preemptions().await), (0, 0));
}
