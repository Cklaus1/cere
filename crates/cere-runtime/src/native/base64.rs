//! The std-only base64 decoder of the grammar sidecar's token masks (spec m28 D7: "cered decodes base64 with a
//! std-only decoder into one reused per-job buffer; no `base64` crate"). The standard alphabet with `=` padding, as
//! Python's `base64.b64encode` writes it; the decoded bytes are little-endian `u32` words.

/// The text is not canonical padded base64 of a whole number of little-endian `u32` words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidBase64;

fn sextet(c: u8) -> Result<u32, InvalidBase64> {
    Ok(u32::from(match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return Err(InvalidBase64),
    }))
}

/// Decodes `text` into `out` (cleared first, its allocation reused) as little-endian `u32` words.
pub fn decode_le_u32(text: &[u8], out: &mut Vec<u32>) -> Result<(), InvalidBase64> {
    out.clear();
    append_le_u32(text, out)
}

/// Decodes `text` as little-endian `u32` words appended to `out` (GrammarSpec GS-03: the masks after each draft go
/// after the mask already in the job's buffer); on `Err`, `out` keeps only its words before the call.
pub fn append_le_u32(text: &[u8], out: &mut Vec<u32>) -> Result<(), InvalidBase64> {
    let start = out.len();
    let decoded = append(text, out);
    if decoded.is_err() {
        out.truncate(start);
    }
    decoded
}

fn append(text: &[u8], out: &mut Vec<u32>) -> Result<(), InvalidBase64> {
    if text.len() % 4 != 0 {
        return Err(InvalidBase64);
    }
    let pad = text.iter().rev().take_while(|&&c| c == b'=').count();
    if pad > 2 {
        return Err(InvalidBase64);
    }
    // The decoded length: 3 bytes per group minus the padding, a whole number of words.
    let bytes = text.len() / 4 * 3 - pad;
    if bytes % 4 != 0 {
        return Err(InvalidBase64);
    }
    out.reserve(bytes / 4);
    let (mut word, mut filled) = (0u32, 0u32);
    for (i, group) in text.chunks_exact(4).enumerate() {
        let last = (i + 1) * 4 == text.len();
        let data = if last { 4 - pad } else { 4 };
        let mut bits = 0u32;
        for &c in &group[..data] {
            bits = (bits << 6) | sextet(c)?;
        }
        bits <<= 6 * (4 - data);
        // Canonical: the bits a padded group does not carry are zero.
        if last && pad > 0 && bits & ((1 << (8 * pad)) - 1) != 0 {
            return Err(InvalidBase64);
        }
        for k in 0..data - 1 {
            word |= ((bits >> (16 - 8 * k)) & 0xff) << (8 * filled);
            filled += 1;
            if filled == 4 {
                out.push(word);
                (word, filled) = (0, 0);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{append_le_u32, decode_le_u32, InvalidBase64};

    /// Vectors written by Python's `base64.b64encode(struct.pack("<%dI" % len(words), *words))`.
    const VECTORS: &[(&str, &[u32])] = &[
        ("", &[]),
        ("AP/+AQ==", &[0x01fe_ff00]),
        ("/////wEAAIA=", &[0xffff_ffff, 0x8000_0001]),
        ("AQAAAAAAAIDvvq3e", &[0x1, 0x8000_0000, 0xdead_beef]),
        (
            "AAECAwQFBgcICQoLDA0OD/79/Ps=",
            &[
                0x0302_0100,
                0x0706_0504,
                0x0b0a_0908,
                0x0f0e_0d0c,
                0xfbfc_fdfe,
            ],
        ),
    ];

    #[test]
    fn python_vectors_decode_into_the_reused_buffer() {
        let mut out = Vec::with_capacity(8);
        let buffer = out.as_ptr();
        for (text, words) in VECTORS {
            decode_le_u32(text.as_bytes(), &mut out).unwrap();
            assert_eq!(out, *words, "{text}");
        }
        assert_eq!(
            out.as_ptr(),
            buffer,
            "the buffer is reused, not reallocated"
        );
    }

    #[test]
    fn non_canonical_or_partial_text_is_refused() {
        let mut out = vec![7];
        for text in [
            "Zm9v",      // 3 bytes: not a whole word
            "Zg==",      // 1 byte
            "AP/+AQ=",   // length not a multiple of 4
            "AP/+AQ=A",  // padding inside the last group
            "AP==AQ==",  // padding before the last group
            "AP/+AR==",  // bits beyond the data are set
            "AP/+AQ*=",  // outside the alphabet
            "A===",      // three pad characters
            "AP/+AQ\n=", // whitespace
        ] {
            assert_eq!(
                decode_le_u32(text.as_bytes(), &mut out),
                Err(InvalidBase64),
                "{text}"
            );
        }
    }

    /// GrammarSpec GS-03: appended words follow the ones already there; a text refused after some of its words
    /// were decoded leaves only those before.
    #[test]
    fn append_keeps_the_words_before_and_restores_them_on_refusal() {
        let mut out = vec![7];
        append_le_u32(b"/////wEAAIA=", &mut out).unwrap();
        assert_eq!(out, [7, 0xffff_ffff, 0x8000_0001]);
        append_le_u32(b"", &mut out).unwrap();
        assert_eq!(out, [7, 0xffff_ffff, 0x8000_0001]);
        assert_eq!(
            append_le_u32(b"AQAAAAAAAIDvvq3*", &mut out),
            Err(InvalidBase64)
        );
        assert_eq!(out, [7, 0xffff_ffff, 0x8000_0001]);
    }
}
