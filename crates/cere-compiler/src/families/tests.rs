use super::*;
use crate::safetensors::parse_file_bytes;
use crate::Limits;

const RULES: &[u8] = include_bytes!("../../../../tests/fixtures/model_src.families.json");
const EXPERTS: &[u8] = include_bytes!(
    "../../../../tests/fixtures/model_src/layer-00000-experts-0000-0001.safetensors"
);
const BF16: &[u8] =
    include_bytes!("../../../../tests/fixtures/model_src/model-bf16-00001.safetensors");
const PLE: &[u8] =
    include_bytes!("../../../../tests/fixtures/model_src/model-plefp8-00000.safetensors");

fn rules() -> Rules {
    Rules::from_value(serde_json::from_slice(RULES).unwrap()).unwrap()
}

fn shards() -> Vec<Header> {
    [EXPERTS, BF16, PLE]
        .iter()
        .map(|b| parse_file_bytes(b, &Limits::default()).unwrap())
        .collect()
}

fn run(shards: &[Header]) -> Result<Classified, FamilyError> {
    classify(&rules(), &shards.iter().collect::<Vec<_>>())
}

fn mutate(f: impl FnOnce(&mut Vec<Header>)) -> &'static str {
    let mut s = shards();
    f(&mut s);
    run(&s).unwrap_err().class()
}

fn find(s: &mut [Header], name: &str) -> (usize, usize) {
    s.iter()
        .enumerate()
        .find_map(|(i, h)| {
            h.tensors
                .iter()
                .position(|t| t.name.ends_with(name))
                .map(|j| (i, j))
        })
        .unwrap()
}

#[test]
fn fixture_classifies_into_every_rule() {
    let c = run(&shards()).unwrap();
    let r = rules();
    let by: BTreeMap<&str, Tally> = r
        .rules
        .iter()
        .map(|x| x.family.as_str())
        .zip(c.tallies.iter().copied())
        .collect();
    assert_eq!(
        by["target_nvfp4_weight"],
        Tally {
            count: 6,
            bytes: 6 * 256,
            misaligned: 0
        }
    );
    assert_eq!(by["target_ple_table"].count, 4);
    assert_eq!(by["target_ple_module"].count, 6);
    assert_eq!(
        by["target_ple_index"],
        Tally {
            count: 3,
            bytes: 88,
            misaligned: 3
        }
    );
    assert!(c.tallies.iter().all(|t| t.count > 0));
    let total: usize = shards().iter().map(|h| h.tensors.len()).sum();
    assert_eq!(c.per_shard.iter().map(Vec::len).sum::<usize>(), total);
}

#[test]
fn rejects_unmatched_ambiguous_dtype_and_shape() {
    assert_eq!(
        mutate(|s| {
            let (i, j) = find(s, "lm_head.weight");
            s[i].tensors[j].name = "nowhere".into()
        }),
        "family_unmatched"
    );
    let mut r = rules();
    let dup = r.rules[0].clone();
    r.rules.push(Rule {
        family: "dup".into(),
        ..dup
    });
    assert_eq!(
        classify(&r, &shards().iter().collect::<Vec<_>>())
            .unwrap_err()
            .class(),
        "family_ambiguous"
    );
    assert_eq!(
        mutate(|s| {
            let (i, j) = find(s, "lm_head.weight");
            s[i].tensors[j].dtype = Dtype::F16
        }),
        "family_dtype"
    );
    assert_eq!(
        mutate(|s| {
            let (i, j) = find(s, "experts.0.up_proj.input_scale");
            s[i].tensors[j].shape = vec![1]
        }),
        "family_shape"
    );
}

/// L5 (m216 design note C4): classification runs per shard on parallel threads, and the error is still the serial
/// pass's first: step 10 in shard order over every shard before any step-11 (expert metadata) error.
#[test]
fn parallel_classification_reports_the_serial_first_error() {
    let many = || -> Vec<Header> { (0..16).flat_map(|_| shards()).collect() };
    let rename = |s: &mut [Header], si: usize, to: &str| {
        let j = s[si]
            .tensors
            .iter()
            .position(|t| t.name.ends_with("lm_head.weight"))
            .unwrap();
        s[si].tensors[j].name = to.into();
    };
    // BF16 shards are at 1, 4, 7, ...; the expert shards (with metadata) at 0, 3, 6, ...
    let mut s = many();
    s[3].metadata
        .as_mut()
        .unwrap()
        .insert("expert_end".into(), "3".into());
    rename(&mut s, 46, "late");
    rename(&mut s, 31, "early");
    let mut d = s.clone();
    let (i, j) = find(&mut d[40..], "lm_head.weight");
    d[40 + i].tensors[j].dtype = Dtype::F16;
    for shards in [&s, &d] {
        for _ in 0..8 {
            match run(shards).unwrap_err() {
                FamilyError::Unmatched(n) => assert_eq!(n, "early"),
                other => panic!("expected the unmatched name of shard 31, got {other:?}"),
            }
        }
    }
    // Without step-10 errors, the first step-11 error in shard order.
    let mut m = many();
    m[3].metadata
        .as_mut()
        .unwrap()
        .insert("expert_end".into(), "3".into());
    m[45]
        .metadata
        .as_mut()
        .unwrap()
        .insert("layer".into(), "1".into());
    let serial = {
        let mut one = shards();
        one[0]
            .metadata
            .as_mut()
            .unwrap()
            .insert("expert_end".into(), "3".into());
        run(&one).unwrap_err().to_string()
    };
    for _ in 0..8 {
        assert_eq!(run(&m).unwrap_err().to_string(), serial);
    }
}

#[test]
fn rejects_group_violations() {
    let drop = |name: &'static str| {
        move |s: &mut Vec<Header>| {
            let (i, j) = find(s, name);
            s[i].tensors.remove(j);
        }
    };
    assert_eq!(
        mutate(drop("experts.1.up_proj.weight_scale_2")),
        "group_nvfp4"
    );
    assert_eq!(
        mutate(|s| {
            let (i, j) = find(s, "experts.0.gate_proj.weight_scale");
            s[i].tensors[j].shape = vec![16, 3]
        }),
        "group_nvfp4"
    );
    assert_eq!(mutate(drop("shard_2.weight")), "group_ple_table");
    assert_eq!(
        mutate(|s| {
            let (i, j) = find(s, "shard_1.weight");
            s[i].tensors[j].shape = vec![4, 10]
        }),
        "group_ple_table"
    );
    assert_eq!(mutate(drop("ngram_heads_offsets")), "group_ple_table");
    assert_eq!(mutate(drop("ple.norm_key.weight")), "group_ple_module");
}

const PLE_PREFIX: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";

/// The fixture rules with the D8 NVFP4 PLE families (as configs/artifacts/qwen4_exp-nvfp4-plenvfp4.families.json):
/// U8 table shards, F8_E4M3 per-shard scales and one F32 `weight_scale_2`, all PLE roles.
fn nvfp4_ple_rules() -> Rules {
    let mut v: serde_json::Value = serde_json::from_slice(RULES).unwrap();
    let rules = v["rules"].as_array_mut().unwrap();
    let p = "model\\.language_model\\.layers\\.[0-9]+\\.ple\\.ple_embedding\\.ngram_embedding\\.";
    for r in rules.iter_mut() {
        match r["family"].as_str().unwrap() {
            "target_ple_table" => r["dtype"] = "U8".into(),
            "target_ple_scale" => {
                r["dtype"] = "F32".into();
                r["pattern"] = format!("{p}weight_scale_2").into();
            }
            _ => {}
        }
    }
    let mut shard_scale = rules
        .iter()
        .find(|r| r["family"] == "target_ple_scale")
        .unwrap()
        .clone();
    shard_scale["family"] = "target_ple_shard_scale".into();
    shard_scale["dtype"] = "F8_E4M3".into();
    shard_scale["pattern"] = format!("{p}shard_[0-9]+\\.weight_scale").into();
    shard_scale["shape_rule"] = "rank:2".into();
    rules.push(shard_scale);
    Rules::from_value(v).unwrap()
}

/// The fixture shards with the PLE table in the D8 NVFP4 layout: each `[5, 8]` shard becomes U8 (16 E2M1 values per
/// row) with an F8_E4M3 `[5, 1]` scale; the BF16 `weight_scale` becomes the F32 `[1]` `weight_scale_2`.
fn nvfp4_ple_shards() -> Vec<Header> {
    let mut s = shards();
    let (i, j) = find(&mut s, "ngram_embedding.weight_scale");
    let g = &mut s[i].tensors[j];
    g.name = format!("{PLE_PREFIX}weight_scale_2");
    g.dtype = Dtype::F32;
    g.shape = vec![1];
    for h in &mut s {
        let mut scales = Vec::new();
        for t in &mut h.tensors {
            if t.name.starts_with(PLE_PREFIX) && shard_index(&t.name).is_some() {
                t.dtype = Dtype::U8;
                let mut sc = t.clone();
                sc.name = format!("{}_scale", t.name);
                sc.dtype = Dtype::F8E4m3;
                sc.shape = vec![t.shape[0], 1];
                scales.push(sc);
            }
        }
        h.tensors.extend(scales);
    }
    s
}

fn nvfp4_run(f: impl FnOnce(&mut Vec<Header>)) -> Result<Classified, FamilyError> {
    let mut s = nvfp4_ple_shards();
    f(&mut s);
    classify(&nvfp4_ple_rules(), &s.iter().collect::<Vec<_>>())
}

#[test]
fn nvfp4_ple_layout_is_accepted() {
    let c = nvfp4_run(|_| {}).unwrap();
    let r = nvfp4_ple_rules();
    let count = |fam: &str| c.tallies[r.rules.iter().position(|x| x.family == fam).unwrap()].count;
    assert_eq!(
        (
            count("target_ple_table"),
            count("target_ple_shard_scale"),
            count("target_ple_scale")
        ),
        (4, 4, 1)
    );
    // The FP8 family keeps its exact check: the NVFP4 tensors fail its dtype rule.
    let s = nvfp4_ple_shards();
    assert_eq!(
        classify(&rules(), &s.iter().collect::<Vec<_>>())
            .unwrap_err()
            .class(),
        "family_unmatched",
        "the FP8 family has no rule for the shard scales"
    );
}

#[test]
fn nvfp4_ple_malformed_scales_are_rejected() {
    let drop = |name: &'static str| {
        move |s: &mut Vec<Header>| {
            let (i, j) = find(s, name);
            s[i].tensors.remove(j);
        }
    };
    let class = |r: Result<Classified, FamilyError>| r.unwrap_err().class();
    assert_eq!(
        class(nvfp4_run(drop("shard_2.weight_scale"))),
        "group_ple_table"
    );
    assert_eq!(
        class(nvfp4_run(drop("ngram_embedding.weight_scale_2"))),
        "group_ple_table"
    );
    assert_eq!(
        class(nvfp4_run(|s| {
            let (i, j) = find(s, "shard_1.weight_scale");
            s[i].tensors[j].shape = vec![5, 2];
        })),
        "group_ple_table",
        "scale count per row must be 2 * cols / 16"
    );
    assert_eq!(
        class(nvfp4_run(|s| {
            let (i, j) = find(s, "ngram_embedding.weight_scale_2");
            let mut extra = s[i].tensors[j].clone();
            // A second global: another layer's module path still matches the rule.
            extra.name =
                "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.weight_scale_2"
                    .into();
            s[i].tensors.push(extra);
        })),
        "group_ple_table",
    );
}

#[test]
fn rejects_expert_metadata_disagreement() {
    assert_eq!(
        mutate(|s| {
            s[0].metadata
                .as_mut()
                .unwrap()
                .insert("expert_end".into(), "3".into());
        }),
        "expert_metadata"
    );
    assert_eq!(
        mutate(|s| {
            s[0].metadata
                .as_mut()
                .unwrap()
                .insert("layer".into(), "1".into());
        }),
        "expert_metadata"
    );
    assert_eq!(
        mutate(|s| {
            s[0].metadata.as_mut().unwrap().remove("expert_start");
        }),
        "expert_metadata"
    );
    assert_eq!(
        mutate(|s| {
            s[0].metadata
                .as_mut()
                .unwrap()
                .insert("layer".into(), "+0".into());
        }),
        "expert_metadata"
    );
}

#[test]
fn layer_expert_follows_python_search() {
    assert_eq!(
        layer_expert("model.language_model.layers.12.mlp.experts.7.up_proj.weight"),
        Some((12, 7))
    );
    assert_eq!(layer_expert("layers.3.mlp.experts.4.x"), Some((3, 4)));
    assert_eq!(layer_expert("xlayers.3.mlp.experts.4.x"), None);
    assert_eq!(layer_expert("a.layers.3.mlp.experts.4"), None);
    assert_eq!(
        layer_expert("a.layers.x.mlp.experts.4.b.layers.5.mlp.experts.6.c"),
        Some((5, 6))
    );
}

#[test]
fn rules_file_structure_is_checked() {
    let mut v: Value = serde_json::from_slice(RULES).unwrap();
    v["rules"][0]["shape_rule"] = "rank:17".into();
    assert!(Rules::from_value(v.clone()).is_err());
    v["rules"][0]["shape_rule"] = "rank:2".into();
    v["rules"][0]["extra"] = 1.into();
    assert!(Rules::from_value(v).is_err());
    assert_eq!(ShapeRule::parse("rank:0"), Some(ShapeRule::Rank(0)));
    assert_eq!(ShapeRule::parse("rank:01"), None);
}
