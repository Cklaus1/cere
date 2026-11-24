//! Sessions over HTTP: create, continuation with expected revision (S2), fork (S3), scope isolation (S5).
mod support;

use std::time::Duration;

use cere_mock_sglang::MockConfig;
use serde_json::{json, Value};
use support::*;

async fn create(h: &Harness, token: &str) -> String {
    let r = h.post("/v1/cere/sessions", token).send().await.unwrap();
    assert_eq!(r.status(), 201);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["revision"], 0);
    body["session_id"].as_str().unwrap().to_owned()
}

fn turn_body(messages: &[(&str, &str)]) -> Value {
    let messages: Vec<Value> = messages
        .iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    json!({"model": MODEL, "messages": messages, "max_completion_tokens": 4})
}

fn turn(h: &Harness, session: &str, revision: u64, body: &Value) -> reqwest::RequestBuilder {
    h.chat(body)
        .header("X-Cere-Session-Id", session)
        .header("X-Cere-Expected-Revision", revision.to_string())
}

async fn session(h: &Harness, id: &str, token: &str) -> (u16, Value) {
    let r = h
        .get(&format!("/v1/cere/sessions/{id}"), token)
        .send()
        .await
        .unwrap();
    (r.status().as_u16(), r.json().await.unwrap())
}

const REPLY: &str = "tok0 tok1 tok2 tok3";

#[tokio::test]
async fn s2_turns_advance_and_conflicts_are_rejected() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let id = create(&h, TOKEN_A).await;

    let first = turn_body(&[("user", "hi")]);
    let r = turn(&h, &id, 0, &first).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], REPLY);
    assert_eq!(body["cere"]["session_id"], id.as_str());
    assert_eq!(body["cere"]["revision"], 1);
    let (code, view) = session(&h, &id, TOKEN_A).await;
    assert_eq!(code, 200);
    assert_eq!(
        view,
        json!({"session_id": id, "revision": 1, "message_count": 2, "model": MODEL, "parent": null})
    );

    // Turn 2 as a stream: the finish chunk carries the new revision.
    let mut second = turn_body(&[("user", "hi"), ("assistant", REPLY), ("user", "more")]);
    second["stream"] = json!(true);
    let r = turn(&h, &id, 1, &second).send().await.unwrap();
    let events = SseReader::new(r).rest().await;
    let finish = check_stream(&events);
    assert_eq!(finish["cere"]["state"], "completed");
    assert_eq!(finish["cere"]["session_id"], id.as_str());
    assert_eq!(finish["cere"]["revision"], 2);
    assert_eq!(session(&h, &id, TOKEN_A).await.1["message_count"], 4);

    let dispatched = h.mock.dispatched_rids().len();
    // Stale revision.
    let stale = turn(
        &h,
        &id,
        1,
        &turn_body(&[("user", "hi"), ("assistant", REPLY), ("user", "x")]),
    )
    .send()
    .await
    .unwrap();
    assert_eq!(stale.status(), 409);
    assert_eq!(
        stale.json::<Value>().await.unwrap()["error"]["code"],
        "revision_conflict"
    );
    // Current revision, edited history.
    let edited = turn_body(&[
        ("user", "HI"),
        ("assistant", REPLY),
        ("user", "more"),
        ("assistant", REPLY),
        ("user", "next"),
    ]);
    let r = turn(&h, &id, 2, &edited).send().await.unwrap();
    assert_eq!(r.status(), 409);
    assert_eq!(
        r.json::<Value>().await.unwrap()["error"]["code"],
        "revision_conflict"
    );
    assert_eq!(
        h.mock.dispatched_rids().len(),
        dispatched,
        "conflicts never reach the backend"
    );

    // Header pairing and unknown sessions.
    let r = h
        .chat(&first)
        .header("X-Cere-Session-Id", id.as_str())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let r = turn(&h, "sess_unknown", 0, &first).send().await.unwrap();
    assert_eq!(r.status(), 404);
    assert_eq!(
        r.json::<Value>().await.unwrap()["error"]["code"],
        "session_not_found"
    );
}

#[tokio::test]
async fn s2_simultaneous_writers_one_wins() {
    let h = Harness::start(
        MockConfig {
            token_latency: Duration::from_millis(40),
            ..MockConfig::default()
        },
        &[],
    )
    .await;
    let id = create(&h, TOKEN_A).await;
    let dispatched = h.mock.dispatched_rids().len();
    let body = turn_body(&[("user", "hi")]);
    let (a, b) = tokio::join!(
        turn(&h, &id, 0, &body).send(),
        turn(&h, &id, 0, &body).send()
    );
    let mut codes = vec![a.unwrap().status().as_u16(), b.unwrap().status().as_u16()];
    codes.sort_unstable();
    assert_eq!(codes, vec![200, 409]);
    assert_eq!(
        h.mock.dispatched_rids().len(),
        dispatched + 1,
        "one backend call"
    );
    assert_eq!(session(&h, &id, TOKEN_A).await.1["revision"], 1);
}

#[tokio::test]
async fn s3_fork_and_s5_scope_isolation() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let parent = create(&h, TOKEN_A).await;
    let r = turn(&h, &parent, 0, &turn_body(&[("user", "hi")]))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();

    let fork = h
        .post(&format!("/v1/cere/sessions/{parent}/fork"), TOKEN_A)
        .json(&json!({"revision": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(fork.status(), 201);
    let fork: Value = fork.json().await.unwrap();
    let child = fork["session_id"].as_str().unwrap().to_owned();
    assert_ne!(child, parent);
    assert_eq!(fork["parent_session_id"], parent.as_str());
    assert_eq!(fork["parent_revision"], 1);
    let (code, view) = session(&h, &child, TOKEN_A).await;
    assert_eq!(code, 200);
    assert_eq!(view["revision"], fork["revision"]);
    assert_eq!(view["message_count"], 2);
    assert_eq!(view["parent"], json!({"session_id": parent, "revision": 1}));
    // The fork continues from the parent's history at that revision.
    let child_rev = view["revision"].as_u64().unwrap();
    let body = turn_body(&[("user", "hi"), ("assistant", REPLY), ("user", "branch")]);
    let r = turn(&h, &child, child_rev, &body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.json::<Value>().await.unwrap()["cere"]["revision"],
        child_rev + 1
    );
    assert_eq!(
        session(&h, &parent, TOKEN_A).await.1["revision"],
        1,
        "the parent is unchanged"
    );

    let r = h
        .post(&format!("/v1/cere/sessions/{parent}/fork"), TOKEN_A)
        .json(&json!({"revision": 5}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 409);

    // Scope b sees nothing of scope a.
    let (code, missing) = session(&h, &parent, TOKEN_B).await;
    assert_eq!(code, 404);
    assert_eq!(missing, session(&h, "sess_unknown", TOKEN_B).await.1);
    let r = h
        .post(&format!("/v1/cere/sessions/{parent}/fork"), TOKEN_B)
        .json(&json!({"revision": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let r = h
        .post("/v1/chat/completions", TOKEN_B)
        .header("X-Cere-Session-Id", parent.as_str())
        .header("X-Cere-Expected-Revision", "1")
        .json(&turn_body(&[
            ("user", "hi"),
            ("assistant", REPLY),
            ("user", "x"),
        ]))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let (code, _) = h.status(&rid, TOKEN_B).await;
    assert_eq!(code, 404);
}
