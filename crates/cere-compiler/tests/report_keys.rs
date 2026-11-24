//! R4/R5: the real `cerec verify`, `probe-load` and `evict` reports, run on a copy of the v2 `examples/artifact`,
//! have exactly the key sets of `tests/fixtures/cerec_report_keys.json`, which `scripts/live/artifact_live.py` reads
//! and the fake cerec of `tests/test_artifact_live.py` mirrors. Every number in them is an integer.
mod support;

use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use support::*;

fn spec() -> Value {
    serde_json::from_slice(&std::fs::read(fixture("cerec_report_keys.json")).unwrap()).unwrap()
}

fn names(v: &Value) -> BTreeSet<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap().to_string())
        .collect()
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

/// No float anywhere in a report (Q17: integers only).
fn assert_no_floats(v: &Value, at: &str) {
    match v {
        Value::Number(n) => assert!(n.is_u64() || n.is_i64(), "{at} is not an integer: {n}"),
        Value::Array(a) => a.iter().for_each(|x| assert_no_floats(x, at)),
        Value::Object(o) => o
            .iter()
            .for_each(|(k, x)| assert_no_floats(x, &format!("{at}.{k}"))),
        _ => {}
    }
}

fn assert_residency(v: &Value, spec: &Value, at: &str) {
    assert_eq!(keys(v), names(&spec["residency"]), "{at}");
}

fn cerec_json(t: &Tmp, args: &[&str], dir: &Path, sha: Option<&str>) -> Value {
    let mut c = cerec();
    c.arg(args[0]).arg(dir);
    if let Some(sha) = sha {
        c.args(["--expected-manifest-sha256", sha]);
    }
    let out = ok(run(c
        .args(&args[1..])
        .env("CEREC_TEST_MEMINFO", meminfo(t, 64 << 30))));
    let v = json(&out);
    assert_no_floats(&v, args[0]);
    v
}

#[test]
fn verify_probe_load_and_evict_reports_have_the_live_script_key_sets() {
    let s = spec();
    let t = Tmp::new("keys");
    let dir = t.join("fixture-model");
    copy_tree(&root().join("examples/artifact"), &dir);
    let sha = std::fs::read_to_string(dir.join("manifest.sha256"))
        .unwrap()
        .trim()
        .to_string();
    let sha = Some(sha.as_str());

    let full = cerec_json(
        &t,
        &["verify", "--mode", "full", "--cache", "cold"],
        &dir,
        sha,
    );
    let mut want = names(&s["verify"]);
    want.insert("cache".into());
    assert_eq!(keys(&full), want, "verify --mode full --cache cold");
    assert_eq!(
        keys(&full["cache"]),
        names(&s["verify_cache"]),
        "verify cache"
    );
    assert_residency(
        &full["cache"]["residency_before"],
        &s,
        "verify cache.residency_before",
    );
    assert_eq!(
        keys(&full["stages_ms"]),
        names(&s["verify_stages_ms"]["full"]),
        "verify full stages_ms"
    );

    let stamp = cerec_json(&t, &["verify", "--mode", "stamp"], &dir, sha);
    assert_eq!(keys(&stamp), names(&s["verify"]), "verify --mode stamp");
    assert_eq!(
        keys(&stamp["stages_ms"]),
        names(&s["verify_stages_ms"]["stamp"]),
        "verify stamp stages_ms"
    );

    for (variant, extra) in [
        ("hash_off", &["--cache", "cold", "--hash", "off"][..]),
        ("hash_on", &["--cache", "warm", "--hash", "on"][..]),
        (
            "baseline",
            &["--cache", "cold", "--hash", "off", "--baseline"][..],
        ),
    ] {
        let mut args = vec!["probe-load", "--class", "all"];
        args.extend_from_slice(extra);
        let r = cerec_json(&t, &args, &dir, sha);
        assert_eq!(keys(&r), names(&s["probe_load"]), "probe-load {variant}");
        assert_eq!(
            keys(&r["stages_ms"]),
            names(&s["probe_load_stages_ms"][variant]),
            "probe-load {variant} stages_ms"
        );
        assert_residency(&r["residency_before"], &s, "probe-load residency_before");
        assert_residency(&r["residency_after"], &s, "probe-load residency_after");
    }

    let ev = cerec_json(&t, &["evict"], &dir, None);
    assert_eq!(keys(&ev), names(&s["evict"]), "evict");
    assert_residency(&ev["residency_after"], &s, "evict residency_after");
}
