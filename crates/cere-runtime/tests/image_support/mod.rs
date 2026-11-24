//! Synthetic image headers for the M2.13 tests (m213 design note C1-C4): the bytes a header parser reads,
//! built in memory, never decodable pixel data and never written to disk.
#![allow(dead_code)]

/// RFC 4648 base64 (standard alphabet, padded).
pub fn b64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A PNG signature and IHDR chunk declaring `width` x `height` (8-bit RGB), then an IEND chunk.
pub fn png(width: u32, height: u32) -> Vec<u8> {
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    out.extend_from_slice(&13u32.to_be_bytes());
    out.extend_from_slice(b"IHDR");
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&[8, 2, 0, 0, 0]);
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(b"IEND");
    out.extend_from_slice(&[0; 4]);
    out
}

/// A JPEG SOI, an APP0 segment, fill bytes, a DQT segment, then a baseline SOF0 declaring `width` x `height`, SOS
/// and EOI.
pub fn jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8];
    out.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
    out.extend_from_slice(b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    out.extend_from_slice(&[0xFF, 0xFF, 0xDB, 0x00, 0x43, 0x00]);
    out.extend_from_slice(&[1; 64]);
    out.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    out.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 1, 1, 0, 0, 0x3F, 0]);
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

pub fn png_url(width: u32, height: u32) -> String {
    format!("data:image/png;base64,{}", b64(&png(width, height)))
}

pub fn jpeg_url(width: u16, height: u16) -> String {
    format!("data:image/jpeg;base64,{}", b64(&jpeg(width, height)))
}
