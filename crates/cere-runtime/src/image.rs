//! Image inputs of a chat request (m213 design note D3, D4, D18): the data-URL policy, the PNG/JPEG header
//! dimensions (std only, nothing decoded beyond the header), and the HF `smart_resize` that gives each image's
//! grid and token count. Errors never quote the URL or its bytes (D6).

use std::fmt;

use serde_json::{json, Value};

/// `patch_size` x `merge_size` (`preprocessor_config.json`): resized sides are multiples of it.
pub const FACTOR: u32 = 32;
/// The processor's `size.shortest_edge` / `size.longest_edge` (pixels, F3).
pub const MIN_PIXELS: u64 = 65_536;
pub const MAX_PIXELS: u64 = 16_777_216;
/// `smart_resize` raises above this aspect ratio.
pub const MAX_RATIO: u32 = 200;
/// D4: a header side above this is refused before any resize.
pub const MAX_SIDE: u32 = 8_192;
/// D4 / VisionLong C2: at most this many images per request (history included) and tokens per image (= 16,384
/// patches). The images' tokens per request are bounded only by the context (the prompt length check).
pub const MAX_IMAGES: usize = 32;
pub const MAX_TOKENS_PER_IMAGE: u32 = 4_096;
/// BF16 `pixel_values` bytes of one merged token: 4 patches x 1,536 values x 2 bytes.
pub const PIXEL_BYTES_PER_TOKEN: u64 = 4 * 1_536 * 2;
/// D2: a `content` array holds 1..=64 parts.
pub const MAX_PARTS: usize = 64;

const PATCH: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaType {
    Png,
    Jpeg,
}

/// One accepted image: its header dimensions and the processor's resize of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    pub media: MediaType,
    pub width: u32,
    pub height: u32,
    pub resized_height: u32,
    pub resized_width: u32,
}

impl ImageInfo {
    /// `image_grid_thw`: one temporal patch, then the 16-px patch rows and columns.
    pub fn grid_thw(&self) -> [u32; 3] {
        [1, self.resized_height / PATCH, self.resized_width / PATCH]
    }

    /// The image's placeholder tokens: the merged 2x2 patches.
    pub fn tokens(&self) -> u32 {
        (self.resized_height / FACTOR) * (self.resized_width / FACTOR)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// Not `data:image/png;base64,` or `data:image/jpeg;base64,` (D3).
    Scheme,
    /// The payload is not strict RFC 4648 base64 (standard alphabet, padded, canonical, no whitespace).
    Base64,
    /// The header is truncated or invalid for the declared media type.
    Malformed(&'static str),
    SideTooLarge {
        width: u32,
        height: u32,
    },
    /// Aspect ratio above [`MAX_RATIO`].
    Aspect,
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheme => f.write_str(
                "only data:image/png;base64, and data:image/jpeg;base64, URLs are accepted",
            ),
            Self::Base64 => f.write_str("the image data is not valid padded base64 (RFC 4648)"),
            Self::Malformed(why) => write!(f, "the image header is malformed: {why}"),
            Self::SideTooLarge { width, height } => write!(
                f,
                "the image is {width}x{height}; each side must be at most {MAX_SIDE} pixels"
            ),
            Self::Aspect => write!(f, "the image aspect ratio exceeds {MAX_RATIO}"),
        }
    }
}

/// Validates a data URL (D3) and reads its header (D4): sides <= [`MAX_SIDE`], then [`smart_resize`].
pub fn parse_data_url(url: &str) -> Result<ImageInfo, ImageError> {
    let (media, payload) = if let Some(p) = url.strip_prefix("data:image/png;base64,") {
        (MediaType::Png, p)
    } else if let Some(p) = url.strip_prefix("data:image/jpeg;base64,") {
        (MediaType::Jpeg, p)
    } else {
        return Err(ImageError::Scheme);
    };
    let bytes = Base64::new(payload.as_bytes())?;
    let (width, height) = header(media, &bytes)?;
    if width > MAX_SIDE || height > MAX_SIDE {
        return Err(ImageError::SideTooLarge { width, height });
    }
    let (resized_height, resized_width) = smart_resize(height, width)?;
    Ok(ImageInfo {
        media,
        width,
        height,
        resized_height,
        resized_width,
    })
}

/// `(width, height)` of a PNG (IHDR) or JPEG (first SOFn frame header).
pub fn header_dimensions(media: MediaType, bytes: &[u8]) -> Result<(u32, u32), ImageError> {
    header(media, &bytes)
}

/// HF `smart_resize` (`image_processing_qwen2_vl.py`) for [`FACTOR`], [`MIN_PIXELS`], [`MAX_PIXELS`]: Python's
/// `round` is ties-to-even (D18), the rest is the same f64 arithmetic in the same order. Returns `(h, w)`.
pub fn smart_resize(height: u32, width: u32) -> Result<(u32, u32), ImageError> {
    if height == 0 || width == 0 {
        return Err(ImageError::Malformed("a side is zero"));
    }
    let (h, w, f) = (f64::from(height), f64::from(width), f64::from(FACTOR));
    if h.max(w) / h.min(w) > f64::from(MAX_RATIO) {
        return Err(ImageError::Aspect);
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    let (min, max) = (MIN_PIXELS as f64, MAX_PIXELS as f64);
    if h_bar * w_bar > max {
        let beta = (h * w / max).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min {
        let beta = (min / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// D3: SGLang's `_convert_anthropic_image_source_to_openai_part` (`anthropic/serving.py:235-266`): a `base64`
/// source becomes a data URL (`media_type` default `image/png`), any other source with a non-empty `url` passes
/// that URL, otherwise `None`. The resulting D2 part then meets the D3 policy like any other.
pub fn anthropic_image_part(source: &Value) -> Option<Value> {
    let source = source.as_object()?;
    let text = |key: &str| source.get(key).and_then(Value::as_str);
    let url = if source.get("type").and_then(Value::as_str) == Some("base64") {
        let media_type = match source.get("media_type") {
            None => "image/png",
            Some(_) => text("media_type")?,
        };
        let data = text("data").filter(|d| !d.is_empty())?;
        format!("data:{media_type};base64,{data}")
    } else {
        text("url").filter(|u| !u.is_empty())?.to_owned()
    };
    Some(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// Random-access bytes of a header: a raw slice or a validated base64 payload decoded on demand.
trait Bytes {
    fn byte(&self, at: usize) -> Option<u8>;
}

impl Bytes for &[u8] {
    fn byte(&self, at: usize) -> Option<u8> {
        self.get(at).copied()
    }
}

/// A strict RFC 4648 payload, validated once; byte `i` is decoded from its quad when read, so a header is read
/// without decoding (or allocating) the image.
struct Base64<'a> {
    text: &'a [u8],
    len: usize,
}

impl<'a> Base64<'a> {
    fn new(text: &'a [u8]) -> Result<Self, ImageError> {
        if text.is_empty() || text.len() % 4 != 0 {
            return Err(ImageError::Base64);
        }
        let pad = text.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 {
            return Err(ImageError::Base64);
        }
        let body = &text[..text.len() - pad];
        if body.iter().any(|&c| sextet(c).is_none()) {
            return Err(ImageError::Base64);
        }
        // Canonical: the bits after the last byte are zero.
        let last = sextet(body[body.len() - 1]).unwrap_or(0);
        if (pad == 1 && last & 0x03 != 0) || (pad == 2 && last & 0x0F != 0) {
            return Err(ImageError::Base64);
        }
        Ok(Self {
            text,
            len: text.len() / 4 * 3 - pad,
        })
    }
}

impl Bytes for Base64<'_> {
    fn byte(&self, at: usize) -> Option<u8> {
        if at >= self.len {
            return None;
        }
        let quad = &self.text[at / 3 * 4..at / 3 * 4 + 4];
        let n = quad
            .iter()
            .fold(0u32, |n, &c| (n << 6) | u32::from(sextet(c).unwrap_or(0)));
        Some((n >> (16 - 8 * (at % 3))) as u8)
    }
}

fn sextet(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

const TRUNCATED: ImageError = ImageError::Malformed("truncated");

fn be16(bytes: &impl Bytes, at: usize) -> Result<u32, ImageError> {
    let hi = bytes.byte(at).ok_or(TRUNCATED)?;
    let lo = bytes.byte(at + 1).ok_or(TRUNCATED)?;
    Ok((u32::from(hi) << 8) | u32::from(lo))
}

fn be32(bytes: &impl Bytes, at: usize) -> Result<u32, ImageError> {
    Ok((be16(bytes, at)? << 16) | be16(bytes, at + 2)?)
}

fn header(media: MediaType, bytes: &impl Bytes) -> Result<(u32, u32), ImageError> {
    let (width, height) = match media {
        MediaType::Png => png(bytes)?,
        MediaType::Jpeg => jpeg(bytes)?,
    };
    if width == 0 || height == 0 {
        return Err(ImageError::Malformed("a side is zero"));
    }
    Ok((width, height))
}

/// The signature, then the first chunk: IHDR, length 13, width and height (big-endian).
fn png(bytes: &impl Bytes) -> Result<(u32, u32), ImageError> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    for (at, &want) in SIGNATURE.iter().enumerate() {
        if bytes.byte(at).ok_or(TRUNCATED)? != want {
            return Err(ImageError::Malformed("not a PNG signature"));
        }
    }
    if be32(bytes, 8)? != 13 || be32(bytes, 12)? != u32::from_be_bytes(*b"IHDR") {
        return Err(ImageError::Malformed("the first PNG chunk is not IHDR"));
    }
    let (width, height) = (be32(bytes, 16)?, be32(bytes, 20)?);
    if width > i32::MAX as u32 || height > i32::MAX as u32 {
        return Err(ImageError::Malformed("a PNG side exceeds 2^31 - 1"));
    }
    Ok((width, height))
}

/// SOI, then marker segments up to the first SOFn (C0-CF except DHT C4, JPG C8, DAC CC): precision, height,
/// width. Fill bytes (FF runs) and parameterless markers (TEM, RSTn) are skipped; SOS or EOI first is malformed.
fn jpeg(bytes: &impl Bytes) -> Result<(u32, u32), ImageError> {
    if bytes.byte(0).ok_or(TRUNCATED)? != 0xFF || bytes.byte(1).ok_or(TRUNCATED)? != 0xD8 {
        return Err(ImageError::Malformed("not a JPEG start of image"));
    }
    let mut at = 2;
    loop {
        if bytes.byte(at).ok_or(TRUNCATED)? != 0xFF {
            return Err(ImageError::Malformed("expected a JPEG marker"));
        }
        while bytes.byte(at).ok_or(TRUNCATED)? == 0xFF {
            at += 1;
        }
        let marker = bytes.byte(at).ok_or(TRUNCATED)?;
        at += 1;
        match marker {
            0x01 | 0xD0..=0xD7 => continue,
            0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => {
                if be16(bytes, at)? < 8 {
                    return Err(ImageError::Malformed("short JPEG frame header"));
                }
                return Ok((be16(bytes, at + 5)?, be16(bytes, at + 3)?));
            }
            0x00 | 0xD8 | 0xD9 | 0xDA => {
                return Err(ImageError::Malformed("no JPEG frame header"));
            }
            _ => {
                let length = be16(bytes, at)? as usize;
                if length < 2 {
                    return Err(ImageError::Malformed("short JPEG segment"));
                }
                at += length;
            }
        }
    }
}
