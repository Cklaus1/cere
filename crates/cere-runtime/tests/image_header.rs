//! M2.13 C2 (m213 design note D3, D4, D18): data-URL policy, PNG/JPEG header dimensions without decoding, and
//! the HF `smart_resize` port against the pinned image's oracle `tests/fixtures/native/image_resize_cases.json`.
mod image_support;

use cere_runtime::image::{
    anthropic_image_part, header_dimensions, parse_data_url, smart_resize, ImageError, MediaType,
    FACTOR, MAX_PIXELS, MAX_RATIO, MIN_PIXELS,
};
use image_support::{b64, jpeg, jpeg_url, png, png_url};
use serde_json::{json, Value};

#[test]
fn png_and_jpeg_headers_give_their_dimensions() {
    for (width, height) in [(1, 1), (640, 480), (41, 8192), (8192, 8192)] {
        let info = parse_data_url(&png_url(width, height)).unwrap();
        assert_eq!(
            (info.media, info.width, info.height),
            (MediaType::Png, width, height)
        );
        let info = parse_data_url(&jpeg_url(width as u16, height as u16)).unwrap();
        assert_eq!(
            (info.media, info.width, info.height),
            (MediaType::Jpeg, width, height)
        );
    }
    // A progressive SOF2 and a restart marker before the frame header.
    let mut progressive = jpeg(300, 200);
    let sof = progressive
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .unwrap();
    progressive[sof + 1] = 0xC2;
    progressive.splice(2..2, [0xFF, 0xD0]);
    assert_eq!(
        header_dimensions(MediaType::Jpeg, &progressive).unwrap(),
        (300, 200)
    );
}

#[test]
fn grid_and_tokens_follow_the_resize() {
    let info = parse_data_url(&png_url(1024, 1024)).unwrap();
    assert_eq!((info.resized_height, info.resized_width), (1024, 1024));
    assert_eq!(info.grid_thw(), [1, 64, 64]);
    assert_eq!(info.tokens(), 1024);
    let info = parse_data_url(&png_url(2000, 48)).unwrap();
    assert_eq!((info.resized_height, info.resized_width), (64, 1984));
    assert_eq!(info.tokens(), 2 * 62);
}

#[test]
fn malformed_and_truncated_headers_are_refused() {
    let full = png(64, 64);
    let mut bad_signature = full.clone();
    bad_signature[1] = b'Q';
    let mut not_ihdr = full.clone();
    not_ihdr[12] = b'X';
    let mut zero_width = full.clone();
    zero_width[16..20].copy_from_slice(&0u32.to_be_bytes());
    for (case, bytes) in [
        ("truncated", full[..20].to_vec()),
        ("empty", Vec::new()),
        ("signature", bad_signature),
        ("chunk", not_ihdr),
        ("zero width", zero_width),
        ("jpeg bytes", jpeg(64, 64)),
    ] {
        assert!(
            matches!(
                header_dimensions(MediaType::Png, &bytes),
                Err(ImageError::Malformed(_))
            ),
            "png {case}"
        );
    }
    let full = jpeg(64, 64);
    let sof = full.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
    let mut zero_height = full.clone();
    zero_height[sof + 5..sof + 7].copy_from_slice(&[0, 0]);
    let mut sos_first = full.clone();
    sos_first[sof + 1] = 0xDA;
    let mut short_segment = full.clone();
    short_segment[4..6].copy_from_slice(&[0, 1]);
    for (case, bytes) in [
        ("truncated in SOF", full[..sof + 6].to_vec()),
        ("truncated in APP0", full[..10].to_vec()),
        ("no SOI", full[2..].to_vec()),
        ("zero height", zero_height),
        ("SOS before SOF", sos_first),
        ("segment length 1", short_segment),
        ("png bytes", png(64, 64)),
    ] {
        assert!(
            matches!(
                header_dimensions(MediaType::Jpeg, &bytes),
                Err(ImageError::Malformed(_))
            ),
            "jpeg {case}"
        );
    }
}

#[test]
fn a_huge_declared_side_is_refused_without_decoding() {
    // Only the header exists: nothing beyond it could be decoded.
    for url in [
        png_url(65535, 65535),
        jpeg_url(65535, 65535),
        png_url(8193, 16),
    ] {
        assert!(matches!(
            parse_data_url(&url),
            Err(ImageError::SideTooLarge { .. })
        ));
    }
}

#[test]
fn only_png_and_jpeg_data_urls_with_strict_base64() {
    let payload = b64(&png(64, 64));
    for url in [
        "http://example.invalid/a.png".to_owned(),
        "https://example.invalid/a.png".to_owned(),
        "file:///etc/passwd".to_owned(),
        "/tmp/a.png".to_owned(),
        format!("data:image/gif;base64,{payload}"),
        format!("data:image/webp;base64,{payload}"),
        format!("data:image/PNG;base64,{payload}"),
        format!("data:image/png,{payload}"),
        format!("DATA:image/png;base64,{payload}"),
    ] {
        assert!(
            matches!(parse_data_url(&url), Err(ImageError::Scheme)),
            "{}",
            &url[..url.len().min(30)]
        );
    }
    let mut bad = Vec::new();
    bad.push(format!("{} {}", &payload[..8], &payload[8..]));
    bad.push(format!("{}\n{}", &payload[..8], &payload[8..]));
    bad.push(payload[..payload.len() - 1].to_owned());
    bad.push(payload.replace('+', "-").replace('/', "_") + "_");
    bad.push(format!("{payload}===="));
    bad.push(format!("={payload}"));
    bad.push(String::new());
    // Non-zero bits after the last byte: `QR==` is not the canonical `QQ==`.
    bad.push("QR==".to_owned());
    bad.push("QUJ=".to_owned());
    for data in bad {
        assert!(
            matches!(
                parse_data_url(&format!("data:image/png;base64,{data}")),
                Err(ImageError::Base64)
            ),
            "{data:.40}"
        );
    }
    // A JPEG body behind a PNG media type is malformed.
    assert!(matches!(
        parse_data_url(&format!("data:image/png;base64,{}", b64(&jpeg(64, 64)))),
        Err(ImageError::Malformed(_))
    ));
}

#[test]
fn errors_never_quote_the_url() {
    let secret = "data:image/png;base64,c2VjcmV0LWJ5dGVz";
    for url in [secret, "http://secret.example.invalid/secret.png"] {
        let error = parse_data_url(url).unwrap_err().to_string();
        assert!(
            !error.contains("secret") && !error.contains("c2Vj"),
            "{error}"
        );
    }
}

#[test]
fn smart_resize_equals_the_pinned_processor() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/native/image_resize_cases.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(fixture["schema"], "cere-image-resize-cases-v1");
    assert_eq!(fixture["factor"], FACTOR);
    assert_eq!(fixture["min_pixels"], MIN_PIXELS);
    assert_eq!(fixture["max_pixels"], MAX_PIXELS);
    assert_eq!(fixture["max_ratio"], MAX_RATIO);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 300);
    let (mut aspect, mut ties) = (0, 0);
    for case in cases {
        let (height, width) = (
            case["height"].as_u64().unwrap() as u32,
            case["width"].as_u64().unwrap() as u32,
        );
        let got = smart_resize(height, width);
        match case["error"].as_str() {
            Some("aspect") => {
                aspect += 1;
                assert!(matches!(got, Err(ImageError::Aspect)), "{height}x{width}");
            }
            None => {
                let want = &case["resized"];
                let (h, w) = got.unwrap_or_else(|e| panic!("{height}x{width}: {e}"));
                assert_eq!(json!([h, w]), *want, "{height}x{width}");
                ties += [48, 80, 112, 144].contains(&height) as u32;
            }
            other => panic!("unknown error {other:?}"),
        }
    }
    assert!(aspect >= 8 && ties >= 8, "{aspect} {ties}");
}

#[test]
fn aspect_201_is_refused_and_200_accepted() {
    assert!(smart_resize(32, 6400).is_ok());
    assert!(matches!(smart_resize(1, 201), Err(ImageError::Aspect)));
    assert!(matches!(
        parse_data_url(&png_url(201, 1)),
        Err(ImageError::Aspect)
    ));
}

/// D3: SGLang's `_convert_anthropic_image_source_to_openai_part` (`anthropic/serving.py:235-266`).
#[test]
fn anthropic_image_sources_convert_as_sglang() {
    let cases = [
        (
            json!({"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}),
            Some(json!({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}})),
        ),
        (
            json!({"type": "base64", "data": "QUJD"}),
            Some(json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}})),
        ),
        (json!({"type": "base64", "media_type": "image/png"}), None),
        (json!({"type": "base64", "data": ""}), None),
        (
            json!({"type": "url", "url": "https://example.invalid/a.png"}),
            Some(
                json!({"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}),
            ),
        ),
        (
            json!({"url": "data:image/png;base64,QUJD"}),
            Some(json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}})),
        ),
        (json!({"type": "url", "url": ""}), None),
        (json!({"type": "file", "file_id": "f"}), None),
        (json!("data:image/png;base64,QUJD"), None),
    ];
    for (source, want) in cases {
        assert_eq!(anthropic_image_part(&source), want, "{source}");
    }
    // The converted part then meets D3: a URL source is refused, a base64 PNG accepted.
    let url = anthropic_image_part(&json!({"type": "url", "url": "https://example.invalid/a.png"}))
        .unwrap();
    assert!(matches!(
        parse_data_url(url["image_url"]["url"].as_str().unwrap()),
        Err(ImageError::Scheme)
    ));
    let png_source =
        json!({"type": "base64", "media_type": "image/png", "data": b64(&png(64, 64))});
    let part = anthropic_image_part(&png_source).unwrap();
    assert!(parse_data_url(part["image_url"]["url"].as_str().unwrap()).is_ok());
}
