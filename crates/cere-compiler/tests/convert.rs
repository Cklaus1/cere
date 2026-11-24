//! `cerec convert` (fp4local design note FL-10): per-rule numerics on synthetic tensors, and an end-to-end run on a
//! synthetic FP4-Local-shaped checkpoint.
mod support;

use cere_compiler::convert::{
    bf16_rne, csf_expand, e4m3_value, f32_to_bf16_rne, mxfp8_to_bf16, nvfp4_to_bf16,
    CONVERSION_FILE, MTP_FILE,
};
use serde_json::{json, Value};
use std::path::Path;
use support::{cerec, class_of, ok, run, Tmp};

fn bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn bf16_value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

// ---- MXFP8 -> BF16 (D4) ----

#[test]
fn mxfp8_upcast_is_exact_incl_subnormals_and_negative_zero() {
    // One 32-wide group per row; row 0 scale 2^0, row 1 scale 2^-68 (byte 59, the smallest in FP4L).
    let mut w = vec![0u8; 64];
    let probes = [0x00u8, 0x80, 0x01, 0x81, 0x07, 0x08, 0x7E, 0xFE, 0x38, 0x3F];
    w[..probes.len()].copy_from_slice(&probes);
    w[32..32 + probes.len()].copy_from_slice(&probes);
    let out = bf16_bits(&mxfp8_to_bf16(&w, &[127, 59], 2, 32).unwrap());
    for (row, scale) in [(0usize, 0i32), (1, -68)] {
        for (j, &b) in probes.iter().enumerate() {
            let want = f64::from(e4m3_value(b).unwrap()) * 2f64.powi(scale);
            let got = out[row * 32 + j];
            assert_eq!(bf16_value(got), want, "row {row} byte {b:#04x}");
            assert_eq!(got >> 15 == 1, b & 0x80 != 0, "sign of {b:#04x}");
        }
    }
    assert_eq!(out[0], 0x0000);
    assert_eq!(out[1], 0x8000, "-0 stays -0");
    assert_eq!(bf16_value(out[2]), 2f64.powi(-9), "smallest E4M3 subnormal");
    assert_eq!(bf16_value(out[32 + 2]), 2f64.powi(-77));
    assert_eq!(bf16_value(out[6]), 448.0);
}

#[test]
fn mxfp8_refuses_nan_and_inexact_values() {
    let mut w = vec![0u8; 32];
    w[3] = 0x7F; // E4M3 NaN
    let e = mxfp8_to_bf16(&w, &[127], 1, 32).unwrap_err();
    assert_eq!(e.class, "convert_mxfp8");
    assert!(
        mxfp8_to_bf16(&[0u8; 32], &[0xFF], 1, 32).is_err(),
        "UE8M0 NaN scale"
    );
    let mut w = vec![0u8; 32];
    w[0] = 0x01; // 2^-9 x 2^-127 = 2^-136: below the smallest BF16 subnormal (2^-133)
    assert!(mxfp8_to_bf16(&w, &[0], 1, 32).is_err());
    assert!(
        mxfp8_to_bf16(&[0u8; 32], &[127, 127], 1, 32).is_err(),
        "scale count"
    );
}

// ---- NVFP4-CSF -> E4M3 (D5) ----

/// Canonical streams for `plane` with the given row bases (the reference `csf_encode`).
fn csf_encode(plane: &[u8], rows: usize, cols: usize, bases: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let mut fixed = Vec::new();
    let mut words = Vec::new();
    for slab in 0..rows / 16 {
        fixed.extend_from_slice(&bases[slab * 16..slab * 16 + 16]);
        for r in slab * 16..slab * 16 + 16 {
            let mut off = vec![0u8; cols];
            for c in 0..cols {
                let v = plane[r * cols + c];
                match v.checked_sub(bases[r]) {
                    Some(d) if d <= 15 => off[c] = d,
                    _ => words.push((u32::from(v) << 24) | (r * cols + c) as u32),
                }
            }
            fixed.extend(off.chunks_exact(2).map(|p| p[0] | (p[1] << 4)));
        }
    }
    (fixed, words)
}

fn sample_plane(rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    let bases: Vec<u8> = (0..rows).map(|r| (r * 7 % 200) as u8 + 20).collect();
    let plane = (0..rows * cols)
        .map(|i| {
            let r = i / cols;
            if i % 13 == 5 {
                (i * 31 % 256) as u8 // often outside the window: an exception
            } else {
                bases[r] + (i % 16) as u8
            }
        })
        .collect();
    (plane, bases)
}

#[test]
fn csf_round_trip_and_nibble_order() {
    let (rows, cols) = (32, 6);
    let (plane, bases) = sample_plane(rows, cols);
    let (fixed, words) = csf_encode(&plane, rows, cols, &bases);
    assert!(!words.is_empty());
    assert_eq!(csf_expand(&fixed, &words, rows, cols).unwrap(), plane);
    // The even column sits in the low nibble: base 0, codes 0x21 -> [1, 2].
    let mut f = vec![0u8; 16 * (1 + 1)];
    f[16] = 0x21;
    assert_eq!(&csf_expand(&f, &[], 16, 2).unwrap()[..2], &[1, 2]);
}

#[test]
fn csf_refuses_every_non_canonical_form() {
    let (rows, cols) = (16, 4);
    let (plane, bases) = sample_plane(rows, cols);
    let (fixed, words) = csf_encode(&plane, rows, cols, &bases);
    assert!(words.len() >= 2);
    let class = |f: &[u8], w: &[u32]| csf_expand(f, w, rows, cols).unwrap_err().class;
    // Row base above 240.
    let mut f = fixed.clone();
    f[0] = 241;
    assert_eq!(class(&f, &words), "convert_csf");
    // Positions not strictly increasing.
    let mut w = words.clone();
    w.swap(0, 1);
    assert_eq!(class(&fixed, &w), "convert_csf");
    let mut w = words.clone();
    w.insert(1, w[0]);
    assert_eq!(class(&fixed, &w), "convert_csf");
    // Position outside the plane.
    let mut w = words.clone();
    w.push((0x10 << 24) | (rows * cols) as u32);
    assert_eq!(class(&fixed, &w), "convert_csf");
    // Exception over a non-zero nibble.
    let p = (words[0] & 0xFF_FFFF) as usize;
    let mut f = fixed.clone();
    let byte = 16 + (p / cols) * (cols / 2) + (p % cols) / 2;
    f[byte] |= if p % 2 == 1 { 0x10 } else { 0x01 };
    assert_eq!(class(&f, &words), "convert_csf");
    // Exception whose byte lies inside its row's window (base + 3).
    let r0 = 0usize;
    let mut w = words.clone();
    let inside = (u32::from(bases[r0] + 3) << 24) | 1;
    let at = w.iter().position(|&x| x & 0xFF_FFFF > 1).unwrap();
    if !w.iter().any(|&x| x & 0xFF_FFFF == 1) {
        w.insert(at, inside);
        let mut f = fixed.clone();
        f[16] &= 0x0F; // column 1 of row 0: zero its (high) nibble
        assert_eq!(class(&f, &w), "convert_csf");
    }
    // Fixed stream length that does not match the plane.
    assert_eq!(class(&fixed[..fixed.len() - 1], &words), "convert_csf");
}

// ---- F32 norms -> BF16 RNE (D6) ----

#[test]
fn rne_ties_go_to_even_and_inexact_values_are_counted() {
    assert_eq!(
        bf16_rne(f32::from_bits(0x3F80_8000)),
        0x3F80,
        "tie, even kept"
    );
    assert_eq!(
        bf16_rne(f32::from_bits(0x3F81_8000)),
        0x3F82,
        "tie, odd rounds up"
    );
    assert_eq!(
        bf16_rne(f32::from_bits(0x3F80_8001)),
        0x3F81,
        "above the tie"
    );
    assert_eq!(
        bf16_rne(f32::from_bits(0x3F80_7FFF)),
        0x3F80,
        "below the tie"
    );
    assert_eq!(
        bf16_rne(f32::from_bits(0xBF81_8000)),
        0xBF82,
        "negative tie"
    );
    assert_eq!(
        bf16_rne(f32::from_bits(0x7F7F_FFFF)),
        0x7F80,
        "overflow to +inf"
    );
    assert_eq!(bf16_rne(-0.0), 0x8000);
    assert!(f32::from_bits(u32::from(bf16_rne(f32::NAN)) << 16).is_nan());
    let src: Vec<u8> = [0x3F80_0000u32, 0x3F80_8000, 0x3F81_8000, 0x4000_0001]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let (out, inexact) = f32_to_bf16_rne(&src);
    assert_eq!(bf16_bits(&out), [0x3F80, 0x3F80, 0x3F82, 0x4000]);
    assert_eq!(inexact, 3);
}

// ---- MTP NVFP4 dequant (D7) ----

#[test]
fn nvfp4_dequant_nibble_order_scale_and_global() {
    // Row of 16: codes 0..15 (byte j = code 2j low, 2j+1 high); scale 0x38 = 1.0; global 2.0 (exact).
    let packed: Vec<u8> = (0..8u8).map(|j| (2 * j) | ((2 * j + 1) << 4)).collect();
    let mut out = Vec::new();
    let inexact = nvfp4_to_bf16(&packed, &[0x38], 2.0, 1, 16, &mut out).unwrap();
    assert_eq!(inexact, 0);
    let e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for (c, bits) in bf16_bits(&out).into_iter().enumerate() {
        let want = e2m1[c % 8] * if c >= 8 { -2.0 } else { 2.0 };
        assert_eq!(bf16_value(bits), want, "column {c}");
    }
    // A global that is not a power of two: (E2M1 x E4M3) x g in F32, then one RNE.
    let g = f32::from_bits(0x3F9D_F3B7);
    let mut out = Vec::new();
    let inexact = nvfp4_to_bf16(&packed, &[0x30], g, 1, 16, &mut out).unwrap();
    let scale = e4m3_value(0x30).unwrap();
    let mut changed = 0;
    for (c, bits) in bf16_bits(&out).into_iter().enumerate() {
        let x = cere_compiler::convert::E2M1[c] * scale * g;
        changed += u64::from(x.to_bits() & 0xFFFF != 0);
        assert_eq!(bits, bf16_rne(x), "column {c}");
    }
    assert_eq!(inexact, changed);
    assert!(inexact > 0);
    assert!(
        nvfp4_to_bf16(&packed, &[0x7F], 1.0, 1, 16, &mut Vec::new()).is_err(),
        "NaN block scale"
    );
}

// ---- End to end ----

const REV: &str = "b9ce18a6887e18c089fdea8509a33e63429a3300";

fn safetensors(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut obj = serde_json::Map::new();
    let mut off = 0usize;
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        obj.insert(
            name.to_string(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}),
        );
        off += bytes.len();
        data.extend_from_slice(bytes);
    }
    let mut head = serde_json::to_vec(&Value::Object(obj)).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    out.extend_from_slice(&data);
    out
}

fn read_st(path: &Path) -> (Value, Vec<u8>) {
    let b = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    assert_eq!((8 + n) % 8, 0, "data region is 8-aligned");
    (
        serde_json::from_slice(&b[8..8 + n]).unwrap(),
        b[8 + n..].to_vec(),
    )
}

fn tensor<'a>(h: &Value, data: &'a [u8], name: &str) -> (String, Vec<u64>, &'a [u8]) {
    let e = &h[name];
    let o: Vec<usize> = e["data_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let shape = e["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    (
        e["dtype"].as_str().unwrap().to_string(),
        shape,
        &data[o[0]..o[1]],
    )
}

const L: &str = "model.language_model.layers.0";
const PLE: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
const K_PROJ: &str = "model.language_model.layers.0.self_attn.k_proj.weight";

/// An MXFP8 pair: weight bytes + shape, scale bytes + shape.
type Mx = (Vec<u8>, Vec<u64>, Vec<u8>, Vec<u64>);

/// `k_proj`: -0, E4M3 subnormals and the extremes at the canonical scale bounds 10 (2^-117) and 246 (2^119).
fn k_proj_default() -> Mx {
    let probes = [0x00u8, 0x80, 0x01, 0x81, 0x07, 0x08, 0x7E, 0xFE, 0x38, 0x3F];
    let mut w = vec![0u8; 64];
    w[..probes.len()].copy_from_slice(&probes);
    w[32..32 + probes.len()].copy_from_slice(&probes);
    (w, vec![2, 32], vec![10, 246], vec![2, 1])
}

/// MTP expert `e`, projection `k` (0 gate, 1 up, 2 down): (E2M1 code, E4M3 block-scale byte, F32 global).
fn mtp_param(e: usize, k: usize) -> (usize, u8, f32) {
    let i = e * 3 + k;
    (1 + i, 0x38 + i as u8, 0.3 * (1 + i) as f32)
}

/// A small FP4-Local-shaped source: shards under `tensors/`, an HF tree, a DOWNLOAD_PIN.
fn make_source(t: &Tmp, csf_fixed_override: Option<Vec<u8>>) -> std::path::PathBuf {
    make_source_with(t, csf_fixed_override, k_proj_default())
}

fn make_source_with(
    t: &Tmp,
    csf_fixed_override: Option<Vec<u8>>,
    k_proj: Mx,
) -> std::path::PathBuf {
    let src = t.join("src");
    std::fs::create_dir_all(src.join("tensors")).unwrap();
    std::fs::create_dir_all(src.join(".cache/huggingface/trees")).unwrap();
    let mx_w: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) & 0x7E).collect();
    let (plane, bases) = sample_plane(16, 4);
    let (fixed, words) = csf_encode(&plane, 16, 4, &bases);
    let fixed = csf_fixed_override.unwrap_or(fixed);
    let words_b: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    // 1 + 2^-8 is a BF16 tie: RNE goes to the even 1.0.
    let norm: Vec<u8> = [1.0f32, f32::from_bits(0x3F80_8000), 0.3]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let shard1 = safetensors(&[
        (
            &format!("{L}.self_attn.q_proj.weight"),
            "F8_E4M3",
            vec![2, 32],
            mx_w.clone(),
        ),
        (
            &format!("{L}.self_attn.q_proj.weight_scale"),
            "U8",
            vec![2, 1],
            vec![127, 120],
        ),
        (
            &format!("{L}.mlp.experts.0.down_proj.weight_scale.nvfp4_csf_fixed"),
            "U8",
            vec![1, 48],
            fixed,
        ),
        (
            &format!("{L}.mlp.experts.0.down_proj.weight_scale.nvfp4_csf_exceptions"),
            "U32",
            vec![words.len() as u64],
            words_b,
        ),
        (
            &format!("{L}.mlp.experts.0.down_proj.weight"),
            "U8",
            vec![16, 32],
            vec![0x5A; 512],
        ),
        (
            &format!("{L}.attn_hyper_connection.hc_norm.weight"),
            "F32",
            vec![3],
            norm,
        ),
        (
            "model.visual.blocks.0.attn.qkv.weight",
            "F8_E4M3",
            vec![1, 32],
            vec![1; 32],
        ),
        (
            "model.visual.blocks.0.attn.qkv.weight_scale",
            "U8",
            vec![1, 1],
            vec![127],
        ),
        (K_PROJ, "F8_E4M3", k_proj.1, k_proj.0),
        (&format!("{K_PROJ}_scale"), "U8", k_proj.3, k_proj.2),
    ]);
    // MTP experts: I = 16, H = 32: gate/up U8 [16, 16] (cols 32), down U8 [32, 8] (cols 16).
    let mut shard2: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    for e in 0..2u8 {
        for (k, proj) in ["gate", "up", "down"].iter().enumerate() {
            let (rows, cols) = if *proj == "down" {
                (32u64, 16u64)
            } else {
                (16, 32)
            };
            let p = format!("mtp.layers.0.mlp.experts.{e}.{proj}_proj");
            let (code, scale, global) = mtp_param(e as usize, k);
            let code = code as u8; // distinct per expert/proj
            shard2.push((
                format!("{p}.weight"),
                "U8",
                vec![rows, cols / 2],
                vec![code | (code << 4); (rows * cols / 2) as usize],
            ));
            shard2.push((
                format!("{p}.weight_scale"),
                "F8_E4M3",
                vec![rows, cols / 16],
                vec![scale; (rows * cols / 16) as usize],
            ));
            shard2.push((
                format!("{p}.weight_scale_2"),
                "F32",
                vec![],
                global.to_le_bytes().to_vec(),
            ));
        }
    }
    shard2.push((
        format!("{PLE}.shard_0.weight"),
        "U8",
        vec![4, 80],
        vec![0x33; 320],
    ));
    shard2.push((
        format!("{PLE}.shard_0.weight_scale"),
        "F8_E4M3",
        vec![4, 10],
        vec![0x38; 40],
    ));
    shard2.push((
        format!("{PLE}.weight_scale_2"),
        "F32",
        vec![1],
        0.75f32.to_le_bytes().to_vec(),
    ));
    shard2.push((
        "lm_head.weight".into(),
        "BF16",
        vec![2, 2],
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    ));
    let shard2: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = shard2
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone()))
        .collect();
    let shard2 = safetensors(&shard2);
    let mut weight_map = serde_json::Map::new();
    for (file, bytes) in [
        ("model-00001-of-00002.safetensors", &shard1),
        ("model-00002-of-00002.safetensors", &shard2),
    ] {
        std::fs::write(src.join("tensors").join(file), bytes).unwrap();
        let (h, _) = read_st(&src.join("tensors").join(file));
        for name in h
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| *k != "__metadata__")
        {
            weight_map.insert(name.clone(), json!(format!("tensors/{file}")));
        }
    }
    let index =
        serde_json::to_vec(&json!({"metadata": {"total_size": 0}, "weight_map": weight_map}))
            .unwrap();
    std::fs::write(src.join("model.safetensors.index.json"), &index).unwrap();
    std::fs::write(
        src.join("config.json"),
        br#"{"text_config":{"vocab_size":8}}"#,
    )
    .unwrap();
    std::fs::write(src.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(src.join("preprocessor_config.json"), b"{}").unwrap();
    std::fs::write(
        src.join("DOWNLOAD_PIN.json"),
        format!(r#"{{"revision": "{REV}", "bad": []}}"#),
    )
    .unwrap();
    let blob = "0".repeat(40);
    let tree = json!({"format_version": 1, "files": {
        "config.json": {"size": 33, "blob_id": blob},
        "model.safetensors.index.json": {"size": index.len(), "blob_id": blob, "lfs_sha256": cere_compiler::sha256_hex(&index)},
    }});
    std::fs::write(
        src.join(format!(".cache/huggingface/trees/{REV}.json")),
        serde_json::to_vec(&tree).unwrap(),
    )
    .unwrap();
    src
}

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn convert_writes_the_cere_layout() {
    let t = Tmp::new("convert");
    let src = make_source(&t, None);
    let out = t.join("out");
    let o = ok(run(cerec()
        .arg("convert")
        .arg(&src)
        .arg("--out")
        .arg(&out)
        .args(["--converter-git-sha", SHA, "--threads", "2"])));
    let report: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(report["dropped_vision_tensors"], 2);
    assert_eq!(report["counts"]["tensors_by_rule"]["mxfp8_to_bf16"], 2);
    assert_eq!(report["counts"]["tensors_by_rule"]["csf_to_e4m3"], 1);
    assert_eq!(report["counts"]["tensors_by_rule"]["f32_norm_to_bf16"], 1);
    assert_eq!(
        report["counts"]["tensors_by_rule"]["mtp_nvfp4_to_bf16_fused"],
        2
    );
    assert_eq!(report["counts"]["norm_values"], 3);
    assert_eq!(report["counts"]["norm_rne_inexact"], 2);

    let (h1, d1) = read_st(&out.join("cere-model-00001-of-00002.safetensors"));
    assert!(
        h1.as_object()
            .unwrap()
            .keys()
            .all(|k| !k.starts_with("model.visual.")),
        "no vision tensor (D10)"
    );
    let (dt, shape, b) = tensor(&h1, &d1, &format!("{L}.self_attn.q_proj.weight"));
    assert_eq!((dt.as_str(), shape), ("BF16", vec![2, 32]));
    assert_eq!(
        b,
        mxfp8_to_bf16(
            &(0..64u8)
                .map(|i| i.wrapping_mul(37) & 0x7E)
                .collect::<Vec<_>>(),
            &[127, 120],
            2,
            32
        )
        .unwrap()
    );
    assert!(h1
        .get(format!("{L}.self_attn.q_proj.weight_scale"))
        .is_none());
    let (dt, shape, b) = tensor(
        &h1,
        &d1,
        &format!("{L}.mlp.experts.0.down_proj.weight_scale"),
    );
    assert_eq!((dt.as_str(), shape), ("F8_E4M3", vec![16, 4]));
    assert_eq!(
        b,
        sample_plane(16, 4).0.as_slice(),
        "CSF expands byte-exactly"
    );
    let (dt, _, b) = tensor(
        &h1,
        &d1,
        &format!("{L}.attn_hyper_connection.hc_norm.weight"),
    );
    assert_eq!(dt, "BF16");
    assert_eq!(bf16_bits(b), [0x3F80, 0x3F80, bf16_rne(0.3)]);
    let (dt, _, b) = tensor(&h1, &d1, &format!("{L}.mlp.experts.0.down_proj.weight"));
    assert_eq!(
        (dt.as_str(), b),
        ("U8", &[0x5A; 512][..]),
        "routed weight copied"
    );

    let (h2, d2) = read_st(&out.join("cere-model-00002-of-00002.safetensors"));
    for (n, dt) in [
        (format!("{PLE}.shard_0.weight"), "U8"),
        (format!("{PLE}.shard_0.weight_scale"), "F8_E4M3"),
        (format!("{PLE}.weight_scale_2"), "F32"),
    ] {
        assert_eq!(tensor(&h2, &d2, &n).0, dt, "PLE kept NVFP4 (D8): {n}");
    }
    assert!(h2
        .as_object()
        .unwrap()
        .keys()
        .all(|k| !k.contains(".experts.0.") || !k.starts_with("mtp.")));

    // MTP: gate_up [E, 2I, H] = [gate; up] per expert; down [E, H, I].
    let (hm, dm) = read_st(&out.join(MTP_FILE));
    let (dt, shape, gu) = tensor(&hm, &dm, "mtp.layers.0.mlp.experts.gate_up_proj");
    assert_eq!((dt.as_str(), shape), ("BF16", vec![2, 32, 32]));
    let (_, shape, down) = tensor(&hm, &dm, "mtp.layers.0.mlp.experts.down_proj");
    assert_eq!(shape, vec![2, 32, 16]);
    let e2m1 = cere_compiler::convert::E2M1;
    let mtp_bf16 = |e: usize, k: usize| {
        let (code, scale, global) = mtp_param(e, k);
        bf16_rne((e2m1[code] * e4m3_value(scale).unwrap()) * global)
    };
    let gu = bf16_bits(gu);
    for e in 0..2usize {
        for (half, k) in [(0usize, 0usize), (1, 1)] {
            let v = gu[e * 32 * 32 + half * 16 * 32];
            assert_eq!(
                v,
                mtp_bf16(e, k),
                "expert {e} {}",
                if half == 0 { "gate first" } else { "up second" }
            );
        }
        assert_eq!(bf16_bits(down)[e * 32 * 16], mtp_bf16(e, 2));
    }

    // Index, copied files, tree, CONVERSION.json.
    let index: Value =
        serde_json::from_slice(&std::fs::read(out.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    assert_eq!(
        index["weight_map"]["mtp.layers.0.mlp.experts.gate_up_proj"],
        MTP_FILE
    );
    assert_eq!(std::fs::read(out.join("tokenizer.json")).unwrap(), b"{}");
    assert!(
        !out.join("preprocessor_config.json").exists(),
        "vision preprocessor not copied"
    );
    let tree: Value = serde_json::from_slice(
        &std::fs::read(out.join(format!(".cache/huggingface/trees/{REV}.json"))).unwrap(),
    )
    .unwrap();
    assert!(
        tree["files"].get("config.json").is_some(),
        "byte-copied file keeps its upstream entry"
    );
    assert!(
        tree["files"].get("model.safetensors.index.json").is_none(),
        "rewritten index dropped from the tree"
    );
    let conv_bytes = std::fs::read(out.join(CONVERSION_FILE)).unwrap();
    assert_eq!(
        report["conversion_sha256"],
        cere_compiler::sha256_hex(&conv_bytes)
    );
    let conv: Value = serde_json::from_slice(&conv_bytes).unwrap();
    assert_eq!(conv["source"]["revision"], REV);
    assert_eq!(conv["converter"]["git_sha"], SHA);
    for (name, f) in conv["files"].as_object().unwrap() {
        let bytes = std::fs::read(out.join(name)).unwrap();
        assert_eq!(f["sha256"], cere_compiler::sha256_hex(&bytes), "{name}");
    }

    // Deterministic: a second run writes the same CONVERSION.json.
    let out2 = t.join("out2");
    ok(run(cerec()
        .arg("convert")
        .arg(&src)
        .arg("--out")
        .arg(&out2)
        .args(["--converter-git-sha", SHA])));
    assert_eq!(
        std::fs::read(out2.join(CONVERSION_FILE)).unwrap(),
        conv_bytes
    );
}

#[test]
fn convert_refuses_non_canonical_csf_and_a_missing_tree() {
    let t = Tmp::new("convert-bad");
    let (plane, bases) = sample_plane(16, 4);
    let (mut fixed, _) = csf_encode(&plane, 16, 4, &bases);
    fixed[0] = 250; // row base above 240
    let src = make_source(&t, Some(fixed));
    let o = run(cerec()
        .arg("convert")
        .arg(&src)
        .arg("--out")
        .arg(t.join("out"))
        .args(["--converter-git-sha", SHA]));
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(class_of(&o), "convert_csf");
    assert!(!t.join("out").join(CONVERSION_FILE).exists());

    let t = Tmp::new("convert-notree");
    let src = make_source(&t, None);
    std::fs::remove_dir_all(src.join(".cache")).unwrap();
    let o = run(cerec()
        .arg("convert")
        .arg(&src)
        .arg("--out")
        .arg(t.join("out"))
        .args(["--converter-git-sha", SHA]));
    assert_eq!(o.status.code(), Some(2));
    assert!(!t.join("out").exists());
}

// ---- S3 format (fp4local design note S3): MXFP8 dense and NVFP4 MTP experts kept ----

fn convert_with(src: &Path, out: &Path, format: Option<&str>) -> std::process::Output {
    let mut c = cerec();
    c.arg("convert").arg(src).arg("--out").arg(out).args([
        "--converter-git-sha",
        SHA,
        "--threads",
        "2",
    ]);
    if let Some(f) = format {
        c.args(["--format", f]);
    }
    run(&mut c)
}

/// name -> (dtype, shape, bytes) over every safetensors file of a converted directory.
fn all_tensors(dir: &Path) -> std::collections::BTreeMap<String, (String, Vec<u64>, Vec<u8>)> {
    let mut out = std::collections::BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) != Some("safetensors") {
            continue;
        }
        let (h, d) = read_st(&p);
        for name in h.as_object().unwrap().keys() {
            if name == "__metadata__" {
                continue;
            }
            let (dt, shape, b) = tensor(&h, &d, name);
            assert!(
                out.insert(name.clone(), (dt, shape, b.to_vec())).is_none(),
                "{name} twice"
            );
        }
    }
    out
}

fn src_tensors(src: &Path) -> std::collections::BTreeMap<String, (String, Vec<u64>, Vec<u8>)> {
    all_tensors(&src.join("tensors"))
}

#[test]
fn convert_s2_default_is_unchanged_by_the_format_option() {
    let t = Tmp::new("convert-s2-default");
    let src = make_source(&t, None);
    let (a, b) = (t.join("default"), t.join("s2"));
    ok(convert_with(&src, &a, None));
    ok(convert_with(&src, &b, Some("s2")));
    let names = |d: &Path| {
        let mut v: Vec<String> = std::fs::read_dir(d)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        v.sort();
        v
    };
    assert_eq!(names(&a), names(&b));
    for n in names(&a).iter().filter(|n| !n.starts_with('.')) {
        assert_eq!(
            std::fs::read(a.join(n)).unwrap(),
            std::fs::read(b.join(n)).unwrap(),
            "{n}"
        );
    }
    let conv: Value =
        serde_json::from_slice(&std::fs::read(a.join(CONVERSION_FILE)).unwrap()).unwrap();
    assert!(
        conv.get("format").is_none(),
        "S2 CONVERSION.json keeps its layout"
    );
    let o = convert_with(&src, &t.join("bad"), Some("s4"));
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(class_of(&o), "usage");
}

#[test]
fn convert_s3_keeps_mxfp8_and_fuses_nvfp4_mtp_experts() {
    let t = Tmp::new("convert-s3");
    let src = make_source(&t, None);
    let (s2_dir, s3_dir) = (t.join("s2"), t.join("s3"));
    ok(convert_with(&src, &s2_dir, None));
    let o = ok(convert_with(&src, &s3_dir, Some("s3")));
    let report: Value = serde_json::from_slice(&o.stdout).unwrap();
    let rules = &report["counts"]["tensors_by_rule"];
    assert_eq!(rules["mxfp8_e4m3_kept"], 2);
    assert_eq!(rules["mxfp8_scale_kept"], 2);
    assert_eq!(rules["mtp_nvfp4_fused"], 6);
    assert_eq!(rules["csf_to_e4m3"], 1);
    assert_eq!(rules["f32_norm_to_bf16"], 1);
    assert!(rules.get("mxfp8_to_bf16").is_none());
    assert!(rules.get("mtp_nvfp4_to_bf16_fused").is_none());
    assert_eq!(report["dropped_vision_tensors"], 2);

    let s2 = all_tensors(&s2_dir);
    let s3 = all_tensors(&s3_dir);
    let src_t = src_tensors(&src);

    // MXFP8: the source bytes kept; upcast with the S2 rule they are the S2 output (exact round trip).
    for w in [format!("{L}.self_attn.q_proj.weight"), K_PROJ.to_string()] {
        let sname = format!("{w}_scale");
        let (dt, shape, wb) = &s3[&w];
        assert_eq!((dt.as_str(), shape), ("F8_E4M3", &vec![2, 32]), "{w}");
        assert_eq!(wb, &src_t[&w].2, "{w}: E4M3 bytes copied");
        let (dt, shape, sb) = &s3[&sname];
        assert_eq!((dt.as_str(), shape), ("U8", &vec![2, 1]), "{sname}");
        assert_eq!(sb, &src_t[&sname].2, "{sname}: UE8M0 bytes copied");
        assert_eq!(
            mxfp8_to_bf16(wb, sb, 2, 32).unwrap(),
            s2[&w].2,
            "{w}: S3 upcast == S2"
        );
    }
    let kb = bf16_bits(&s2[K_PROJ].2);
    assert_eq!(kb[1], 0x8000, "-0");
    assert_eq!(
        kb[2], 0x0080,
        "subnormal 2^-9 at scale 10 = 2^-126, BF16 normal min"
    );
    assert_eq!(bf16_value(kb[32 + 6]), 448.0 * 2f64.powi(119), "scale 246");

    // MTP experts: packed bytes, block scales and globals, [gate; up] per expert.
    let p = "mtp.layers.0.mlp.experts";
    let want = [
        ("gate_up_proj", "U8", vec![2u64, 32, 16]),
        ("gate_up_proj_scale", "F8_E4M3", vec![2, 32, 2]),
        ("gate_up_proj_scale_2", "F32", vec![2, 2]),
        ("down_proj", "U8", vec![2, 32, 8]),
        ("down_proj_scale", "F8_E4M3", vec![2, 32, 1]),
        ("down_proj_scale_2", "F32", vec![2]),
    ];
    let (hm, _) = read_st(&s3_dir.join(MTP_FILE));
    assert_eq!(
        hm.as_object().unwrap().len(),
        want.len() + 1,
        "6 MTP tensors + metadata"
    );
    for (n, dt, shape) in &want {
        let (gdt, gshape, _) = &s3[&format!("{p}.{n}")];
        assert_eq!((gdt.as_str(), gshape), (*dt, shape), "{n}");
    }
    let piece = |e: usize, proj: &str, part: &str| -> Vec<u8> {
        src_t[&format!("{p}.{e}.{proj}_proj.{part}")].2.clone()
    };
    let mut gu = Vec::new();
    let mut gus = Vec::new();
    let mut gu2 = Vec::new();
    let mut d = Vec::new();
    let mut ds = Vec::new();
    let mut d2 = Vec::new();
    let mut s2_gu = Vec::new();
    let mut s2_d = Vec::new();
    for e in 0..2 {
        for proj in ["gate", "up"] {
            gu.extend(piece(e, proj, "weight"));
            gus.extend(piece(e, proj, "weight_scale"));
            gu2.extend(piece(e, proj, "weight_scale_2"));
        }
        d.extend(piece(e, "down", "weight"));
        ds.extend(piece(e, "down", "weight_scale"));
        d2.extend(piece(e, "down", "weight_scale_2"));
        assert_eq!(
            f32::from_le_bytes(piece(e, "up", "weight_scale_2").try_into().unwrap()),
            mtp_param(e, 1).2
        );
        // The S2 rule on the S3 pieces.
        for (proj, rows, cols) in [("gate", 16, 32), ("up", 16, 32), ("down", 32, 16)] {
            let g = f32::from_le_bytes(piece(e, proj, "weight_scale_2").try_into().unwrap());
            let out = if proj == "down" {
                &mut s2_d
            } else {
                &mut s2_gu
            };
            nvfp4_to_bf16(
                &piece(e, proj, "weight"),
                &piece(e, proj, "weight_scale"),
                g,
                rows,
                cols,
                out,
            )
            .unwrap();
        }
    }
    for (n, bytes) in [
        ("gate_up_proj", gu),
        ("gate_up_proj_scale", gus),
        ("gate_up_proj_scale_2", gu2),
        ("down_proj", d),
        ("down_proj_scale", ds),
        ("down_proj_scale_2", d2),
    ] {
        assert_eq!(s3[&format!("{p}.{n}")].2, bytes, "{n}: fuse order");
    }
    assert_eq!(
        s2[&format!("{p}.gate_up_proj")].2,
        s2_gu,
        "S2 rule on S3 gate_up"
    );
    assert_eq!(s2[&format!("{p}.down_proj")].2, s2_d, "S2 rule on S3 down");

    // Everything else exactly as S2.
    for (n, v) in &s2 {
        if n.starts_with(p) || n == K_PROJ || n.ends_with("q_proj.weight") {
            continue;
        }
        assert_eq!(&s3[n], v, "{n} as S2");
    }
    let mut new: Vec<&String> = s3.keys().filter(|n| !s2.contains_key(*n)).collect();
    new.sort();
    assert_eq!(
        new,
        [
            &format!("{K_PROJ}_scale"),
            &format!("{L}.self_attn.q_proj.weight_scale"),
            &format!("{p}.down_proj_scale"),
            &format!("{p}.down_proj_scale_2"),
            &format!("{p}.gate_up_proj_scale"),
            &format!("{p}.gate_up_proj_scale_2"),
        ],
        "new names: 2 MXFP8 scales + 4 MTP pieces"
    );

    let index: Value = serde_json::from_slice(
        &std::fs::read(s3_dir.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        index["weight_map"][format!("{K_PROJ}_scale")],
        index["weight_map"][K_PROJ]
    );
    assert_eq!(
        index["weight_map"][format!("{p}.down_proj_scale_2")],
        MTP_FILE
    );
    let conv: Value =
        serde_json::from_slice(&std::fs::read(s3_dir.join(CONVERSION_FILE)).unwrap()).unwrap();
    assert_eq!(conv["format"], "s3");
    assert_eq!(report["format"], "s3");
}

#[test]
fn convert_s3_refuses_non_canonical_mxfp8() {
    let (w, ws, _, ss) = k_proj_default();
    let mut nan7f = w.clone();
    nan7f[40] = 0x7F;
    let mut nanff = w.clone();
    nanff[5] = 0xFF;
    let cases: Vec<(&str, Mx)> = vec![
        ("scale 9", (w.clone(), ws.clone(), vec![9, 127], ss.clone())),
        (
            "scale 247",
            (w.clone(), ws.clone(), vec![127, 247], ss.clone()),
        ),
        (
            "scale 255",
            (w.clone(), ws.clone(), vec![255, 127], ss.clone()),
        ),
        ("NaN 0x7F", (nan7f, ws.clone(), vec![127, 127], ss.clone())),
        ("NaN 0xFF", (nanff, ws.clone(), vec![127, 127], ss.clone())),
        (
            "i % 32",
            (vec![0u8; 96], vec![2, 48], vec![127, 127], ss.clone()),
        ),
    ];
    for (what, mx) in cases {
        let t = Tmp::new("convert-s3-bad");
        let src = make_source_with(&t, None, mx);
        let o = convert_with(&src, &t.join("out"), Some("s3"));
        assert_eq!(o.status.code(), Some(2), "{what}");
        assert_eq!(class_of(&o), "convert_mxfp8", "{what}");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains(K_PROJ), "{what}: names the tensor: {err}");
        assert!(!t.join("out").join(CONVERSION_FILE).exists(), "{what}");
    }
    // Scale 9 is a canonical S2 input (every product is still exactly a BF16 value): only S3 refuses it.
    let t = Tmp::new("convert-s2-scale9");
    let src = make_source_with(&t, None, (w, ws, vec![9, 127], ss));
    ok(convert_with(&src, &t.join("out"), None));
}
