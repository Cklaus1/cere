//! Single-byte font encodings and glyph names for PDF text extraction (doc-input design note DI-04): the
//! WinAnsi (cp1252) table, StandardEncoding's differences from it, MacRoman's text punctuation, and the Adobe
//! Glyph List names a PDF writer puts in an `/Encoding /Differences` array.

/// The predefined encoding a simple font names (`/Encoding` or `/BaseEncoding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    WinAnsi,
    Standard,
    MacRoman,
}

impl Base {
    pub fn named(name: &[u8]) -> Option<Self> {
        match name {
            b"WinAnsiEncoding" => Some(Self::WinAnsi),
            b"StandardEncoding" => Some(Self::Standard),
            b"MacRomanEncoding" => Some(Self::MacRoman),
            _ => None,
        }
    }

    /// The character of `code` in this encoding (`None`: undefined or not text).
    pub fn char(self, code: u8) -> Option<char> {
        match self {
            Self::WinAnsi => winansi(code),
            Self::Standard => match code {
                0x27 => Some('\u{2019}'),
                0x60 => Some('\u{2018}'),
                0x20..=0x7e => Some(char::from(code)),
                0xa1 => Some('¡'),
                0xa2 => Some('¢'),
                0xa3 => Some('£'),
                0xa5 => Some('¥'),
                0xa7 => Some('§'),
                0xa9 => Some('\''),
                0xaa => Some('\u{201c}'),
                0xab => Some('«'),
                0xae => Some('\u{fb01}'),
                0xaf => Some('\u{fb02}'),
                0xb1 => Some('\u{2013}'),
                0xb2 => Some('\u{2020}'),
                0xb3 => Some('\u{2021}'),
                0xb4 => Some('·'),
                0xb6 => Some('¶'),
                0xb7 => Some('\u{2022}'),
                0xb8 => Some('\u{201a}'),
                0xb9 => Some('\u{201e}'),
                0xba => Some('\u{201d}'),
                0xbb => Some('»'),
                0xbc => Some('\u{2026}'),
                0xbd => Some('\u{2030}'),
                0xbf => Some('¿'),
                0xd0 => Some('\u{2014}'),
                0xe1 => Some('Æ'),
                0xe8 => Some('Ł'),
                0xe9 => Some('Ø'),
                0xea => Some('Œ'),
                0xf1 => Some('æ'),
                0xf5 => Some('ı'),
                0xf8 => Some('ł'),
                0xf9 => Some('ø'),
                0xfa => Some('œ'),
                0xfb => Some('ß'),
                _ => None,
            },
            Self::MacRoman => match code {
                0x20..=0x7e => Some(char::from(code)),
                0xa5 => Some('\u{2022}'),
                0xa9 => Some('©'),
                0xaa => Some('\u{2122}'),
                0xc9 => Some('\u{2026}'),
                0xca => Some('\u{a0}'),
                0xd0 => Some('\u{2013}'),
                0xd1 => Some('\u{2014}'),
                0xd2 => Some('\u{201c}'),
                0xd3 => Some('\u{201d}'),
                0xd4 => Some('\u{2018}'),
                0xd5 => Some('\u{2019}'),
                0xde => Some('\u{fb01}'),
                0xdf => Some('\u{fb02}'),
                _ => None,
            },
        }
    }
}

/// cp1252 as PDF's WinAnsiEncoding (PDF 1.7 annex D.2); 0x7f and the five undefined C1 codes have no character.
pub fn winansi(code: u8) -> Option<char> {
    Some(match code {
        0x20..=0x7e | 0xa0..=0xff => char::from(code),
        0x80 => '\u{20ac}',
        0x82 => '\u{201a}',
        0x83 => '\u{0192}',
        0x84 => '\u{201e}',
        0x85 => '\u{2026}',
        0x86 => '\u{2020}',
        0x87 => '\u{2021}',
        0x88 => '\u{02c6}',
        0x89 => '\u{2030}',
        0x8a => '\u{0160}',
        0x8b => '\u{2039}',
        0x8c => '\u{0152}',
        0x8e => '\u{017d}',
        0x91 => '\u{2018}',
        0x92 => '\u{2019}',
        0x93 => '\u{201c}',
        0x94 => '\u{201d}',
        0x95 => '\u{2022}',
        0x96 => '\u{2013}',
        0x97 => '\u{2014}',
        0x98 => '\u{02dc}',
        0x99 => '\u{2122}',
        0x9a => '\u{0161}',
        0x9b => '\u{203a}',
        0x9c => '\u{0153}',
        0x9e => '\u{017e}',
        0x9f => '\u{0178}',
        _ => return None,
    })
}

/// The glyph names of WinAnsiEncoding codes 0x20..=0xff (`""` = undefined), so a `/Differences` name maps to the
/// character WinAnsi gives its code.
const WINANSI_NAMES: [&str; 224] = [
    "space",
    "exclam",
    "quotedbl",
    "numbersign",
    "dollar",
    "percent",
    "ampersand",
    "quotesingle",
    "parenleft",
    "parenright",
    "asterisk",
    "plus",
    "comma",
    "hyphen",
    "period",
    "slash",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "colon",
    "semicolon",
    "less",
    "equal",
    "greater",
    "question",
    "at",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "bracketleft",
    "backslash",
    "bracketright",
    "asciicircum",
    "underscore",
    "grave",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "braceleft",
    "bar",
    "braceright",
    "asciitilde",
    "",
    "Euro",
    "",
    "quotesinglbase",
    "florin",
    "quotedblbase",
    "ellipsis",
    "dagger",
    "daggerdbl",
    "circumflex",
    "perthousand",
    "Scaron",
    "guilsinglleft",
    "OE",
    "",
    "Zcaron",
    "",
    "",
    "quoteleft",
    "quoteright",
    "quotedblleft",
    "quotedblright",
    "bullet",
    "endash",
    "emdash",
    "tilde",
    "trademark",
    "scaron",
    "guilsinglright",
    "oe",
    "",
    "zcaron",
    "Ydieresis",
    "nbspace",
    "exclamdown",
    "cent",
    "sterling",
    "currency",
    "yen",
    "brokenbar",
    "section",
    "dieresis",
    "copyright",
    "ordfeminine",
    "guillemotleft",
    "logicalnot",
    "sfthyphen",
    "registered",
    "macron",
    "degree",
    "plusminus",
    "twosuperior",
    "threesuperior",
    "acute",
    "mu",
    "paragraph",
    "periodcentered",
    "cedilla",
    "onesuperior",
    "ordmasculine",
    "guillemotright",
    "onequarter",
    "onehalf",
    "threequarters",
    "questiondown",
    "Agrave",
    "Aacute",
    "Acircumflex",
    "Atilde",
    "Adieresis",
    "Aring",
    "AE",
    "Ccedilla",
    "Egrave",
    "Eacute",
    "Ecircumflex",
    "Edieresis",
    "Igrave",
    "Iacute",
    "Icircumflex",
    "Idieresis",
    "Eth",
    "Ntilde",
    "Ograve",
    "Oacute",
    "Ocircumflex",
    "Otilde",
    "Odieresis",
    "multiply",
    "Oslash",
    "Ugrave",
    "Uacute",
    "Ucircumflex",
    "Udieresis",
    "Yacute",
    "Thorn",
    "germandbls",
    "agrave",
    "aacute",
    "acircumflex",
    "atilde",
    "adieresis",
    "aring",
    "ae",
    "ccedilla",
    "egrave",
    "eacute",
    "ecircumflex",
    "edieresis",
    "igrave",
    "iacute",
    "icircumflex",
    "idieresis",
    "eth",
    "ntilde",
    "ograve",
    "oacute",
    "ocircumflex",
    "otilde",
    "odieresis",
    "divide",
    "oslash",
    "ugrave",
    "uacute",
    "ucircumflex",
    "udieresis",
    "yacute",
    "thorn",
    "ydieresis",
];

/// Names outside WinAnsi that text fonts commonly use.
const EXTRA_NAMES: [(&str, &str); 34] = [
    ("fi", "fi"),
    ("fl", "fl"),
    ("ff", "ff"),
    ("ffi", "ffi"),
    ("ffl", "ffl"),
    ("dotlessi", "ı"),
    ("Lslash", "Ł"),
    ("lslash", "ł"),
    ("minus", "\u{2212}"),
    ("fraction", "\u{2044}"),
    ("space", " "),
    ("nbspace", "\u{a0}"),
    ("hyphen", "-"),
    ("quotesingle", "'"),
    ("Omega", "Ω"),
    ("Delta", "Δ"),
    ("pi", "π"),
    ("summation", "∑"),
    ("product", "∏"),
    ("infinity", "∞"),
    ("notequal", "≠"),
    ("lessequal", "≤"),
    ("greaterequal", "≥"),
    ("partialdiff", "∂"),
    ("radical", "√"),
    ("integral", "∫"),
    ("approxequal", "≈"),
    ("lozenge", "◊"),
    ("arrowright", "→"),
    ("arrowleft", "←"),
    ("arrowup", "↑"),
    ("arrowdown", "↓"),
    ("checkmark", "✓"),
    ("degree", "°"),
];

/// The text of a glyph name (Adobe Glyph List conventions): `uniXXXX...`, `uXXXX[XX]`, a WinAnsi or common name,
/// `a_b` ligature components and `.suffix` variants. `None` when the name says nothing about its text.
pub fn glyph(name: &[u8]) -> Option<String> {
    let name = std::str::from_utf8(name).ok()?;
    let name = name.split('.').next().unwrap_or_default();
    if name.is_empty() {
        return None;
    }
    if name.contains('_') {
        let parts: Option<Vec<String>> = name.split('_').map(|p| glyph(p.as_bytes())).collect();
        return parts.map(|p| p.concat());
    }
    if let Some(hex) = name.strip_prefix("uni") {
        if hex.len() >= 4 && hex.len() % 4 == 0 {
            let units: Option<Vec<u16>> = (0..hex.len())
                .step_by(4)
                .map(|i| u16::from_str_radix(&hex[i..i + 4], 16).ok())
                .collect();
            if let Some(units) = units {
                return Some(String::from_utf16_lossy(&units));
            }
        }
    }
    if let Some(hex) = name.strip_prefix('u') {
        if (4..=6).contains(&hex.len()) && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return u32::from_str_radix(hex, 16)
                .ok()
                .and_then(char::from_u32)
                .map(String::from);
        }
    }
    if let Some((_, text)) = EXTRA_NAMES.iter().find(|(n, _)| *n == name) {
        return Some((*text).to_owned());
    }
    WINANSI_NAMES
        .iter()
        .position(|n| *n == name)
        .and_then(|i| winansi(0x20 + i as u8))
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_names_map_to_text() {
        assert_eq!(glyph(b"A").as_deref(), Some("A"));
        assert_eq!(glyph(b"eacute").as_deref(), Some("é"));
        assert_eq!(glyph(b"quoteright").as_deref(), Some("\u{2019}"));
        assert_eq!(glyph(b"uni0041").as_deref(), Some("A"));
        assert_eq!(glyph(b"uni00410042").as_deref(), Some("AB"));
        assert_eq!(glyph(b"u1F600").as_deref(), Some("\u{1f600}"));
        assert_eq!(glyph(b"f_f_i").as_deref(), Some("ffi"));
        assert_eq!(glyph(b"a.sc").as_deref(), Some("a"));
        assert_eq!(glyph(b"fi").as_deref(), Some("fi"));
        assert_eq!(glyph(b"g123"), None);
    }

    #[test]
    fn winansi_is_cp1252() {
        assert_eq!(winansi(b'A'), Some('A'));
        assert_eq!(winansi(0x93), Some('\u{201c}'));
        assert_eq!(winansi(0xe9), Some('é'));
        assert_eq!(winansi(0x81), None);
        assert_eq!(Base::Standard.char(0x27), Some('\u{2019}'));
    }
}
