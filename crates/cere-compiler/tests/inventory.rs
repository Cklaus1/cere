//! R2 (m21 design note Q12, Q17): `cerec inventory` is byte-identical to `scripts/model_inventory.py`.
mod support;

use support::*;

fn both(src: &std::path::Path, families: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
    let rust = ok(run(cerec()
        .arg("inventory")
        .arg(src)
        .arg("--families")
        .arg(families)))
    .stdout;
    let py = ok(run(python("model_inventory.py")
        .arg(src)
        .arg("--families")
        .arg(families)))
    .stdout;
    (rust, py)
}

#[test]
fn fixture_inventory_matches_python_and_the_committed_file() {
    let families = fixture("model_src.families.json");
    let (rust, py) = both(&model_src(), &families);
    assert_eq!(String::from_utf8_lossy(&rust), String::from_utf8_lossy(&py));
    let committed = std::fs::read(fixture("model_src.inventory.json")).unwrap();
    assert_eq!(rust, committed);
}

#[test]
fn header_digest_uses_bytewise_name_order() {
    // Rename the shards so that name order differs from header-hash order; both tools must still agree.
    let t = Tmp::new("inv-order");
    let src = t.join("src");
    copy_tree(&model_src(), &src);
    let inv: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture("model_src.inventory.json")).unwrap())
            .unwrap();
    let mut by_hash: Vec<(String, String)> = inv["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["header_sha256"].as_str().unwrap().to_string(),
                f["path"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    by_hash.sort();
    // Name the file with the largest hash "a…", the smallest "z…".
    for (i, (_, name)) in by_hash.iter().rev().enumerate() {
        let new = format!("{}-{name}", ["a", "m", "z"][i]);
        std::fs::rename(src.join(name), src.join(&new)).unwrap();
    }
    let families = fixture("model_src.families.json");
    let (rust, py) = both(&src, &families);
    assert_eq!(String::from_utf8_lossy(&rust), String::from_utf8_lossy(&py));
    let v: serde_json::Value = serde_json::from_slice(&rust).unwrap();
    let mut names: Vec<(String, String)> = v["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["path"].as_str().unwrap().into(),
                f["header_sha256"].as_str().unwrap().into(),
            )
        })
        .collect();
    names.sort();
    let lines = names.iter().fold(String::new(), |mut s, (n, h)| {
        s.push_str(&format!("{h}  {n}\n"));
        s
    });
    use sha2::Digest;
    assert_eq!(
        v["header_digest"],
        cere_compiler::hex(&sha2::Sha256::digest(lines.as_bytes()))
    );
    let mut wrong: Vec<String> = lines.lines().map(|l| format!("{l}\n")).collect();
    wrong.sort();
    assert_ne!(
        v["header_digest"],
        cere_compiler::hex(&sha2::Sha256::digest(wrong.concat().as_bytes()))
    );
}

#[test]
fn doubly_matched_rule_exits_2() {
    let t = Tmp::new("inv-dup");
    let mut rules: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture("model_src.families.json")).unwrap())
            .unwrap();
    let mut dup = rules["rules"][4].clone();
    dup["family"] = "target_dense_again".into();
    rules["rules"].as_array_mut().unwrap().push(dup);
    let path = t.join("families.json");
    std::fs::write(&path, serde_json::to_vec(&rules).unwrap()).unwrap();
    let out = run(cerec()
        .arg("inventory")
        .arg(model_src())
        .arg("--families")
        .arg(&path));
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(class_of(&out), "family_ambiguous");
    assert!(out.stdout.is_empty());
}

/// The `inventory` cases of the shared table: the fixture source with its HF tree file replaced.
#[test]
fn every_inventory_case_is_rejected_with_its_class() {
    let t: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture("artifact_invalid.json")).unwrap()).unwrap();
    let cases: Vec<&serde_json::Value> = t["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["tag"] == "inventory")
        .collect();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let tmp = Tmp::new(&format!("inv-{id}"));
        let src = tmp.join("src");
        copy_tree(&model_src(), &src);
        let trees = src.join(".cache/huggingface/trees");
        for old in std::fs::read_dir(&trees).unwrap() {
            std::fs::remove_file(old.unwrap().path()).unwrap();
        }
        std::fs::write(
            trees.join(case["tree_name"].as_str().unwrap()),
            case["tree_text"].as_str().unwrap(),
        )
        .unwrap();
        let out = run(cerec()
            .arg("inventory")
            .arg(&src)
            .arg("--families")
            .arg(fixture("model_src.families.json")));
        let want = case["error"].as_str().unwrap();
        if out.status.code() != Some(2) || class_of(&out) != want {
            failures.push(format!(
                "{id}: exit {:?} class {:?}, want {want}",
                out.status.code(),
                class_of(&out)
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
