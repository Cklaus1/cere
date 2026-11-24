//! RFC 1950 (zlib) / RFC 1951 (DEFLATE) decoding for PDF `/FlateDecode` streams (doc-input design note DI-04):
//! std only, a canonical-Huffman decoder after zlib's `puff.c`, output bounded by the caller's limit.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InflateError {
    /// The input ended inside a block.
    Truncated,
    /// The bit stream is not valid DEFLATE.
    Invalid(&'static str),
    /// The output would exceed the caller's limit.
    TooLarge,
}

impl fmt::Display for InflateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("a compressed stream is truncated"),
            Self::Invalid(why) => write!(f, "a compressed stream is invalid: {why}"),
            Self::TooLarge => f.write_str("a compressed stream expands beyond the limit"),
        }
    }
}

/// Inflates a zlib stream (a raw DEFLATE stream when the two header bytes are not a zlib header, as some PDF
/// writers emit). The Adler-32 trailer is not checked. On error the output decoded so far is returned with it,
/// so a reader may keep the text of a stream whose tail is damaged.
pub fn zlib(data: &[u8], limit: usize) -> Result<Vec<u8>, (InflateError, Vec<u8>)> {
    let body = match data {
        [cmf, flg, rest @ ..]
            if cmf & 0x0f == 8 && ((u16::from(*cmf) << 8) | u16::from(*flg)) % 31 == 0 =>
        {
            if flg & 0x20 != 0 {
                return Err((InflateError::Invalid("a preset dictionary"), Vec::new()));
            }
            rest
        }
        _ => data,
    };
    let mut out = Vec::new();
    match inflate(body, limit, &mut out) {
        Ok(()) => Ok(out),
        Err(e) => Err((e, out)),
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    count: u32,
}

impl Bits<'_> {
    fn bits(&mut self, n: u32) -> Result<u32, InflateError> {
        while self.count < n {
            let byte = *self.data.get(self.pos).ok_or(InflateError::Truncated)?;
            self.pos += 1;
            self.buf |= u64::from(byte) << self.count;
            self.count += 8;
        }
        let value = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.count -= n;
        Ok(value)
    }

    fn align(&mut self) {
        let drop = self.count % 8;
        self.buf >>= drop;
        self.count -= drop;
    }
}

const MAX_BITS: usize = 15;

/// A canonical Huffman code: the number of codes of each length and the symbols in code order.
struct Huffman {
    count: [u16; MAX_BITS + 1],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self, InflateError> {
        let mut count = [0u16; MAX_BITS + 1];
        for &len in lengths {
            count[usize::from(len)] += 1;
        }
        let mut left: i32 = 1;
        for &c in &count[1..] {
            left = (left << 1) - i32::from(c);
            if left < 0 {
                return Err(InflateError::Invalid("an over-subscribed code"));
            }
        }
        let mut offs = [0u16; MAX_BITS + 1];
        for len in 1..MAX_BITS {
            offs[len + 1] = offs[len] + count[len];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (sym, &len) in lengths.iter().enumerate() {
            if len != 0 {
                symbol[usize::from(offs[usize::from(len)])] = sym as u16;
                offs[usize::from(len)] += 1;
            }
        }
        count[0] = 0;
        Ok(Self { count, symbol })
    }

    fn decode(&self, bits: &mut Bits) -> Result<u16, InflateError> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=MAX_BITS {
            code |= bits.bits(1)? as i32;
            let count = i32::from(self.count[len]);
            if code - count < first {
                return self
                    .symbol
                    .get((index + code - first) as usize)
                    .copied()
                    .ok_or(InflateError::Invalid("a code outside the table"));
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(InflateError::Invalid("an unused code"))
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The order of the code-length code lengths in a dynamic block header.
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn inflate(data: &[u8], limit: usize, out: &mut Vec<u8>) -> Result<(), InflateError> {
    let mut bits = Bits {
        data,
        pos: 0,
        buf: 0,
        count: 0,
    };
    loop {
        let last = bits.bits(1)? == 1;
        match bits.bits(2)? {
            0 => stored(&mut bits, limit, out)?,
            1 => {
                let (lit, dist) = fixed()?;
                codes(&mut bits, &lit, &dist, limit, out)?;
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits)?;
                codes(&mut bits, &lit, &dist, limit, out)?;
            }
            _ => return Err(InflateError::Invalid("block type 3")),
        }
        if last {
            return Ok(());
        }
    }
}

fn stored(bits: &mut Bits, limit: usize, out: &mut Vec<u8>) -> Result<(), InflateError> {
    bits.align();
    let len = bits.bits(16)?;
    let nlen = bits.bits(16)?;
    if len != !nlen & 0xffff {
        return Err(InflateError::Invalid("a stored block length check"));
    }
    if out.len() + len as usize > limit {
        return Err(InflateError::TooLarge);
    }
    for _ in 0..len {
        out.push(bits.bits(8)? as u8);
    }
    Ok(())
}

fn fixed() -> Result<(Huffman, Huffman), InflateError> {
    let mut lengths = [0u8; 288];
    lengths[..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    Ok((Huffman::new(&lengths)?, Huffman::new(&[5u8; 30])?))
}

fn dynamic(bits: &mut Bits) -> Result<(Huffman, Huffman), InflateError> {
    let nlen = bits.bits(5)? as usize + 257;
    let ndist = bits.bits(5)? as usize + 1;
    let ncode = bits.bits(4)? as usize + 4;
    if nlen > 286 || ndist > 30 {
        return Err(InflateError::Invalid("too many length or distance codes"));
    }
    let mut cl = [0u8; 19];
    for &index in &CL_ORDER[..ncode] {
        cl[index] = bits.bits(3)? as u8;
    }
    let cl = Huffman::new(&cl)?;
    let mut lengths = vec![0u8; nlen + ndist];
    let mut index = 0;
    while index < nlen + ndist {
        let symbol = cl.decode(bits)?;
        if symbol < 16 {
            lengths[index] = symbol as u8;
            index += 1;
            continue;
        }
        let (value, repeat) = match symbol {
            16 => {
                let previous = *index
                    .checked_sub(1)
                    .and_then(|i| lengths.get(i))
                    .ok_or(InflateError::Invalid("a repeat with no previous length"))?;
                (previous, 3 + bits.bits(2)? as usize)
            }
            17 => (0, 3 + bits.bits(3)? as usize),
            _ => (0, 11 + bits.bits(7)? as usize),
        };
        if index + repeat > nlen + ndist {
            return Err(InflateError::Invalid("too many code lengths"));
        }
        lengths[index..index + repeat].fill(value);
        index += repeat;
    }
    if lengths[256] == 0 {
        return Err(InflateError::Invalid("no end-of-block code"));
    }
    Ok((
        Huffman::new(&lengths[..nlen])?,
        Huffman::new(&lengths[nlen..])?,
    ))
}

fn codes(
    bits: &mut Bits,
    lit: &Huffman,
    dist: &Huffman,
    limit: usize,
    out: &mut Vec<u8>,
) -> Result<(), InflateError> {
    loop {
        let symbol = usize::from(lit.decode(bits)?);
        match symbol {
            0..=255 => {
                if out.len() == limit {
                    return Err(InflateError::TooLarge);
                }
                out.push(symbol as u8);
            }
            256 => return Ok(()),
            _ => {
                let at = symbol - 257;
                if at >= LEN_BASE.len() {
                    return Err(InflateError::Invalid("a length symbol above 285"));
                }
                let len = usize::from(LEN_BASE[at]) + bits.bits(u32::from(LEN_EXTRA[at]))? as usize;
                let at = usize::from(dist.decode(bits)?);
                if at >= DIST_BASE.len() {
                    return Err(InflateError::Invalid("a distance symbol above 29"));
                }
                let back =
                    usize::from(DIST_BASE[at]) + bits.bits(u32::from(DIST_EXTRA[at]))? as usize;
                if back > out.len() {
                    return Err(InflateError::Invalid("a distance before the output start"));
                }
                if out.len() + len > limit {
                    return Err(InflateError::TooLarge);
                }
                let start = out.len() - back;
                for k in 0..len {
                    out.push(out[start + k]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `zlib.compress(b"hello hello hello hello\n", 9)` (CPython 3.12): a fixed-Huffman block with a back-reference.
    const FIXED: [u8; 17] = [
        0x78, 0xda, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0xb9, 0x00, 0x70, 0xbe,
        0x08, 0xbb,
    ];

    #[test]
    fn a_stored_block_round_trips() {
        let data = [
            0x78, 0x01, 0x01, 0x05, 0x00, 0xfa, 0xff, b'a', b'b', b'c', b'd', b'e', 0, 0, 0, 0,
        ];
        assert_eq!(zlib(&data, 100).unwrap(), b"abcde");
    }

    #[test]
    fn a_fixed_block_with_a_back_reference_decodes() {
        assert_eq!(
            zlib(&FIXED, 100).unwrap_or_else(|(_, out)| out),
            b"hello hello hello hello\n"
        );
    }

    #[test]
    fn the_limit_is_enforced_and_keeps_the_partial_output() {
        let (e, partial) = zlib(&FIXED, 8).unwrap_err();
        assert_eq!(e, InflateError::TooLarge);
        assert!(partial.len() <= 8 && b"hello hello".starts_with(&partial));
        assert!(partial.starts_with(b"hello h"));
    }

    #[test]
    fn garbage_is_refused() {
        assert!(zlib(&[0x78, 0x9c, 0xff, 0xff, 0xff], 100).is_err());
        assert!(zlib(&[], 100).is_err());
    }

    /// `tests/fixtures/document/inflate/generate.py`'s `source()`: the same seeded xorshift32 and the same choices.
    fn source() -> Vec<u8> {
        const WORDS: [&[u8]; 16] = [
            b"page",
            b"text",
            b"layer",
            b"document",
            b"stream",
            b"object",
            b"the",
            b"of",
            b"and",
            b"PDF",
            b"CERE",
            b"inflate",
            b"Huffman",
            b"code",
            b"length",
            b"distance",
        ];
        let mut rng = Rng(0x2545_f491);
        let mut out = Vec::new();
        while out.len() < SOURCE_BYTES {
            match rng.next() % 10 {
                0..=6 => {
                    out.extend_from_slice(WORDS[rng.below(WORDS.len())]);
                    out.push(if rng.next() % 8 == 0 { b'\n' } else { b' ' });
                }
                7 => out.extend((0..64).map(|_| rng.next() as u8)),
                8 => {
                    let byte = rng.next() as u8;
                    let run = rng.below(300);
                    out.extend(std::iter::repeat_n(byte, run));
                }
                _ if !out.is_empty() => {
                    let distance = 1 + rng.below(out.len().min(32768));
                    for _ in 0..3 + rng.below(256) {
                        out.push(out[out.len() - distance]);
                    }
                }
                _ => {}
            }
        }
        out.truncate(SOURCE_BYTES);
        out
    }

    const SOURCE_BYTES: usize = 66_000;

    /// xorshift32, as `generate.py` has it.
    struct Rng(u32);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            self.next() as usize % n
        }
    }

    /// The checked-in streams: (name, the first block's BTYPE, bytes).
    fn fixtures() -> Vec<(&'static str, u8, Vec<u8>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/document/inflate");
        [
            ("level0", 0),
            ("level1", 2),
            ("level6", 2),
            ("level9", 2),
            ("fixed", 1),
        ]
        .into_iter()
        .map(|(name, btype)| {
            let bytes = std::fs::read(dir.join(format!("{name}.zz")))
                .unwrap_or_else(|e| panic!("{name}.zz: {e}"));
            (name, btype, bytes)
        })
        .collect()
    }

    /// DI-04c: Python zlib's streams at levels 0, 1, 6 and 9 and with fixed Huffman codes only (Z_FIXED) inflate
    /// to the source; one byte under its length is TooLarge with a prefix of it.
    #[test]
    fn python_zlib_streams_round_trip() {
        let source = source();
        for (name, btype, stream) in fixtures() {
            assert_eq!((stream[2] >> 1) & 3, btype, "{name}: first block type");
            match zlib(&stream, SOURCE_BYTES) {
                Ok(out) => assert!(
                    out == source,
                    "{name}: inflated bytes differ from the source"
                ),
                Err((e, _)) => panic!("{name}: {e}"),
            }
            let (e, partial) = zlib(&stream, SOURCE_BYTES - 1).unwrap_err();
            assert_eq!(e, InflateError::TooLarge, "{name}");
            assert!(
                partial.len() < SOURCE_BYTES && source.starts_with(&partial),
                "{name}"
            );
        }
    }

    /// DI-04b: 10,000 seeded hostile inputs (random bytes, random bytes behind a zlib header or a dynamic-block
    /// header, truncated streams, real streams with 1-8 flipped bits) never panic, and each ends Ok or with a typed
    /// error, its output (or the partial output kept with the error) within the limit.
    #[test]
    fn hostile_inputs_end_typed_within_the_limit() {
        const LIMITS: [usize; 6] = [0, 1, 64, 4096, SOURCE_BYTES, 1 << 20];
        let mut streams: Vec<Vec<u8>> = fixtures().into_iter().map(|(_, _, s)| s).collect();
        streams.push(FIXED.to_vec());
        let mut rng = Rng(0x00c0_ffee);
        let (mut ok, mut truncated, mut invalid, mut too_large) = (0, 0, 0, 0);
        for case in 0..10_000 {
            let data = match case % 5 {
                0 => (0..rng.below(600)).map(|_| rng.next() as u8).collect(),
                1 => {
                    let mut data = vec![0x78, 0x9c];
                    data.extend((0..rng.below(600)).map(|_| rng.next() as u8));
                    data
                }
                2 => {
                    // BFINAL 1, BTYPE 2: straight into the dynamic table reader.
                    let mut data: Vec<u8> =
                        (0..1 + rng.below(600)).map(|_| rng.next() as u8).collect();
                    data[0] = (data[0] & !7) | 0b101;
                    data
                }
                3 => {
                    let stream = &streams[rng.below(streams.len())];
                    stream[..rng.below(stream.len() + 1)].to_vec()
                }
                _ => {
                    let mut data = streams[rng.below(streams.len())].clone();
                    for _ in 0..1 + rng.below(8) {
                        let bit = rng.below(data.len() * 8);
                        data[bit / 8] ^= 1 << (bit % 8);
                    }
                    data
                }
            };
            let limit = LIMITS[rng.below(LIMITS.len())];
            match zlib(&data, limit) {
                Ok(out) => {
                    assert!(out.len() <= limit, "case {case}: {} > {limit}", out.len());
                    ok += 1;
                }
                Err((e, partial)) => {
                    assert!(
                        partial.len() <= limit,
                        "case {case}: {} > {limit}",
                        partial.len()
                    );
                    match e {
                        InflateError::Truncated => truncated += 1,
                        InflateError::Invalid(_) => invalid += 1,
                        InflateError::TooLarge => too_large += 1,
                    }
                }
            }
        }
        assert_eq!(ok + truncated + invalid + too_large, 10_000);
        // Every outcome is reached, so the cases cover each path out of the decoder.
        assert!(
            ok > 0 && truncated > 0 && invalid > 0 && too_large > 0,
            "{ok} {truncated} {invalid} {too_large}"
        );
    }
}
