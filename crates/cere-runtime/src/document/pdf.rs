//! PDF text extraction (doc-input design note DI-04): each page's text in content-stream order, std only.
//!
//! Objects are found by scanning the file for `N G obj` (no xref table is trusted, so a damaged or incrementally
//! updated file reads as its newest definitions) and by expanding object streams; the catalog comes from the
//! newest trailer or cross-reference stream. A page's text is its content streams (and form XObjects) run through
//! the text operators, decoded by each font's `/ToUnicode` CMap or its simple encoding, with line breaks where the
//! text line moves and spaces where a gap opens between glyphs. Encrypted files are refused; images are not read.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Range;
use std::rc::Rc;

use super::encoding::{self, Base};
use super::inflate;

/// Bytes one decoded stream may expand to, and all of a document's decoded streams together (decompression bombs).
const MAX_STREAM_BYTES: usize = 64 << 20;
const MAX_DECODED_BYTES: usize = 256 << 20;
/// Nesting bounds: page tree depth, reference chains, form XObjects within forms, and parsed object depth.
const MAX_TREE_DEPTH: usize = 64;
const MAX_REF_HOPS: usize = 32;
const MAX_FORM_DEPTH: usize = 8;
const MAX_OBJ_DEPTH: usize = 64;
/// A `bfrange` longer than this is ignored (a CMap maps at most 4-byte codes; real ranges stay within one byte).
const MAX_CMAP_RANGE: u32 = 1 << 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PdfError {
    /// No `%PDF-` header in the first 1,024 bytes.
    NotPdf,
    /// The file declares `/Encrypt`.
    Encrypted,
    /// No page tree or no pages.
    NoPages,
    /// More pages than the caller allows.
    TooManyPages(usize),
    /// Decoded streams exceed [`MAX_DECODED_BYTES`].
    TooLarge,
}

impl fmt::Display for PdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPdf => f.write_str("the document is not a PDF (no %PDF- header)"),
            Self::Encrypted => f.write_str("encrypted (password-protected) PDFs are not supported"),
            Self::NoPages => f.write_str("the PDF has no readable pages"),
            Self::TooManyPages(n) => write!(f, "the PDF has {n} pages"),
            Self::TooLarge => write!(
                f,
                "the PDF's compressed streams expand beyond {MAX_DECODED_BYTES} bytes"
            ),
        }
    }
}

/// The text of each page, in page order. A page with no text layer (a scan) yields an empty string.
/// `max_pages`: more pages than this refuses with [`PdfError::TooManyPages`] before any content is decoded.
pub fn extract(data: &[u8], max_pages: usize) -> Result<Vec<String>, PdfError> {
    let header = &data[..data.len().min(1024)];
    if !header.windows(5).any(|w| w == b"%PDF-") {
        return Err(PdfError::NotPdf);
    }
    let doc = Doc::scan(data)?;
    let pages = doc.pages();
    if pages.is_empty() {
        return Err(PdfError::NoPages);
    }
    if pages.len() > max_pages {
        return Err(PdfError::TooManyPages(pages.len()));
    }
    pages.iter().map(|page| doc.page_text(page)).collect()
}

// -------------------------------------------------------------------------------------------------------------
// Objects

#[derive(Clone, Debug)]
enum Obj {
    Null,
    Bool,
    Num(f64),
    Name(Vec<u8>),
    Str(Vec<u8>),
    Array(Vec<Obj>),
    Dict(Dict),
    Ref(u32),
    Stream(Rc<Stream>),
}

#[derive(Clone, Debug, Default)]
struct Dict(Vec<(Vec<u8>, Obj)>);

impl Dict {
    fn get(&self, key: &[u8]) -> &Obj {
        self.0
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map_or(&Obj::Null, |(_, v)| v)
    }
}

#[derive(Debug)]
struct Stream {
    dict: Dict,
    /// The encoded bytes, a range of the file.
    data: Range<usize>,
}

impl Obj {
    fn num(&self) -> Option<f64> {
        match self {
            Self::Num(n) => Some(*n),
            _ => None,
        }
    }

    fn name(&self) -> Option<&[u8]> {
        match self {
            Self::Name(n) => Some(n),
            _ => None,
        }
    }

    fn dict(&self) -> Option<&Dict> {
        match self {
            Self::Dict(d) => Some(d),
            Self::Stream(s) => Some(&s.dict),
            _ => None,
        }
    }
}

fn is_white(b: u8) -> bool {
    matches!(b, 0 | b'\t' | b'\n' | 0x0c | b'\r' | b' ')
}

fn is_delim(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

fn is_regular(b: u8) -> bool {
    !is_white(b) && !is_delim(b)
}

enum Tok<'a> {
    Obj(Obj),
    Keyword(&'a [u8]),
    ArrayStart,
    ArrayEnd,
    DictStart,
    DictEnd,
}

struct Lexer<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(s: &'a [u8], pos: usize) -> Self {
        Self { s, pos }
    }

    fn skip_white(&mut self) {
        while let Some(&b) = self.s.get(self.pos) {
            if is_white(b) {
                self.pos += 1;
            } else if b == b'%' {
                while let Some(&b) = self.s.get(self.pos) {
                    if b == b'\n' || b == b'\r' {
                        break;
                    }
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn token(&mut self) -> Option<Tok<'a>> {
        self.skip_white();
        let b = *self.s.get(self.pos)?;
        match b {
            b'/' => {
                self.pos += 1;
                Some(Tok::Obj(Obj::Name(self.name())))
            }
            b'(' => {
                self.pos += 1;
                Some(Tok::Obj(Obj::Str(self.literal())))
            }
            b'<' if self.s.get(self.pos + 1) == Some(&b'<') => {
                self.pos += 2;
                Some(Tok::DictStart)
            }
            b'<' => {
                self.pos += 1;
                Some(Tok::Obj(Obj::Str(self.hex())))
            }
            b'>' if self.s.get(self.pos + 1) == Some(&b'>') => {
                self.pos += 2;
                Some(Tok::DictEnd)
            }
            b'[' => {
                self.pos += 1;
                Some(Tok::ArrayStart)
            }
            b']' => {
                self.pos += 1;
                Some(Tok::ArrayEnd)
            }
            b'+' | b'-' | b'.' | b'0'..=b'9' => {
                let start = self.pos;
                while self.s.get(self.pos).is_some_and(|&b| is_regular(b)) {
                    self.pos += 1;
                }
                Some(Tok::Obj(Obj::Num(number(&self.s[start..self.pos]))))
            }
            _ if is_regular(b) => {
                let start = self.pos;
                while self.s.get(self.pos).is_some_and(|&b| is_regular(b)) {
                    self.pos += 1;
                }
                let word = &self.s[start..self.pos];
                Some(match word {
                    b"true" | b"false" => Tok::Obj(Obj::Bool),
                    b"null" => Tok::Obj(Obj::Null),
                    _ => Tok::Keyword(word),
                })
            }
            _ => {
                // A stray `)`, `>`, `{` or `}`: skip it.
                self.pos += 1;
                Some(Tok::Keyword(&self.s[self.pos - 1..self.pos]))
            }
        }
    }

    fn name(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        while let Some(&b) = self.s.get(self.pos) {
            if !is_regular(b) {
                break;
            }
            self.pos += 1;
            if b == b'#' {
                let hex = self.s.get(self.pos..self.pos + 2);
                if let Some(v) = hex.and_then(|h| std::str::from_utf8(h).ok()) {
                    if let Ok(v) = u8::from_str_radix(v, 16) {
                        out.push(v);
                        self.pos += 2;
                        continue;
                    }
                }
            }
            out.push(b);
        }
        out
    }

    fn literal(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut depth = 1;
        while let Some(&b) = self.s.get(self.pos) {
            self.pos += 1;
            match b {
                b'(' => {
                    depth += 1;
                    out.push(b);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(b);
                }
                b'\\' => {
                    let Some(&e) = self.s.get(self.pos) else {
                        break;
                    };
                    self.pos += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'0'..=b'7' => {
                            let mut v = u32::from(e - b'0');
                            for _ in 0..2 {
                                match self.s.get(self.pos) {
                                    Some(&d @ b'0'..=b'7') => {
                                        v = v * 8 + u32::from(d - b'0');
                                        self.pos += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        b'\r' => {
                            if self.s.get(self.pos) == Some(&b'\n') {
                                self.pos += 1;
                            }
                        }
                        b'\n' => {}
                        other => out.push(other),
                    }
                }
                b'\r' => {
                    if self.s.get(self.pos) == Some(&b'\n') {
                        self.pos += 1;
                    }
                    out.push(b'\n');
                }
                _ => out.push(b),
            }
        }
        out
    }

    fn hex(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut high: Option<u8> = None;
        while let Some(&b) = self.s.get(self.pos) {
            self.pos += 1;
            if b == b'>' {
                break;
            }
            let Some(v) = (b as char).to_digit(16) else {
                continue;
            };
            match high.take() {
                Some(h) => out.push((h << 4) | v as u8),
                None => high = Some(v as u8),
            }
        }
        if let Some(h) = high {
            out.push(h << 4);
        }
        out
    }

    /// One object; `int int R` is a reference when `refs` (file syntax). `None` at the end or on a keyword.
    fn object(&mut self, refs: bool, depth: usize) -> Option<Obj> {
        let tok = self.token()?;
        self.compose(tok, refs, depth)
    }

    /// `depth` bounds the nesting of arrays and dictionaries: a deeper one reads as nothing, so a hostile
    /// `[[[[…` cannot exhaust the stack.
    fn compose(&mut self, tok: Tok<'a>, refs: bool, depth: usize) -> Option<Obj> {
        if depth > MAX_OBJ_DEPTH {
            return None;
        }
        match tok {
            Tok::Obj(Obj::Num(n)) if refs && n >= 0.0 && n.fract() == 0.0 => {
                let save = self.pos;
                if let Some(Tok::Obj(Obj::Num(g))) = self.token() {
                    if g >= 0.0 && g.fract() == 0.0 {
                        if let Some(Tok::Keyword(b"R")) = self.token() {
                            return Some(Obj::Ref(n as u32));
                        }
                    }
                }
                self.pos = save;
                Some(Obj::Num(n))
            }
            Tok::Obj(obj) => Some(obj),
            Tok::ArrayStart => {
                let mut items = Vec::new();
                loop {
                    match self.token() {
                        None | Some(Tok::ArrayEnd) => break,
                        Some(Tok::Keyword(_)) | Some(Tok::DictEnd) => continue,
                        Some(tok) => {
                            if let Some(obj) = self.compose(tok, refs, depth + 1) {
                                items.push(obj);
                            }
                        }
                    }
                }
                Some(Obj::Array(items))
            }
            Tok::DictStart => {
                let mut entries = Vec::new();
                loop {
                    match self.token() {
                        None | Some(Tok::DictEnd) => break,
                        Some(Tok::Obj(Obj::Name(key))) => {
                            let value = match self.token() {
                                None | Some(Tok::DictEnd) => {
                                    entries.push((key, Obj::Null));
                                    break;
                                }
                                Some(tok) => {
                                    self.compose(tok, refs, depth + 1).unwrap_or(Obj::Null)
                                }
                            };
                            entries.push((key, value));
                        }
                        Some(_) => continue,
                    }
                }
                Some(Obj::Dict(Dict(entries)))
            }
            Tok::Keyword(_) | Tok::ArrayEnd | Tok::DictEnd => None,
        }
    }
}

/// A PDF number; malformed forms (`--5`, `1.2.3`) read as far as they parse, else 0.
fn number(text: &[u8]) -> f64 {
    let text = std::str::from_utf8(text).unwrap_or_default();
    let mut end = 0;
    let mut seen_dot = false;
    for (i, c) in text.char_indices() {
        let ok = match c {
            '+' | '-' => i == 0,
            '.' if !seen_dot => {
                seen_dot = true;
                true
            }
            '0'..='9' => true,
            _ => false,
        };
        if !ok {
            break;
        }
        end = i + 1;
    }
    let text = &text[..end];
    text.parse::<f64>().unwrap_or_else(|_| match text {
        "" | "+" | "-" | "." | "-." | "+." => 0.0,
        _ => text.trim_end_matches('.').parse().unwrap_or(0.0),
    })
}

// -------------------------------------------------------------------------------------------------------------
// The document

struct Doc<'a> {
    data: &'a [u8],
    /// Object number -> (file position of its definition or object stream, object).
    objects: HashMap<u32, (usize, Obj)>,
    root: Option<Obj>,
    decoded: std::cell::Cell<usize>,
    /// Loaded fonts by the address of their dictionary (owned by `objects`, so stable): pages share fonts.
    fonts: RefCell<HashMap<usize, Rc<Font>>>,
}

struct Page {
    contents: Obj,
    resources: Option<Dict>,
}

impl<'a> Doc<'a> {
    fn scan(data: &'a [u8]) -> Result<Self, PdfError> {
        let mut doc = Doc {
            data,
            objects: HashMap::new(),
            root: None,
            decoded: std::cell::Cell::new(0),
            fonts: RefCell::new(HashMap::new()),
        };
        let mut trailers: Vec<(usize, Dict)> = Vec::new();
        let mut pos = 0;
        while let Some(found) = find(data, b"obj", pos) {
            pos = found + 3;
            let Some(num) = object_header(data, found) else {
                continue;
            };
            if data.get(pos).is_some_and(|&b| is_regular(b)) {
                continue;
            }
            let mut lexer = Lexer::new(data, pos);
            let Some(obj) = lexer.object(true, 0) else {
                continue;
            };
            let obj = match obj {
                Obj::Dict(dict) => match stream_after(data, lexer.pos, &dict) {
                    Some(range) => {
                        pos = range.end;
                        Obj::Stream(Rc::new(Stream { dict, data: range }))
                    }
                    None => {
                        pos = lexer.pos;
                        Obj::Dict(dict)
                    }
                },
                other => {
                    pos = lexer.pos;
                    other
                }
            };
            if let Obj::Stream(s) = &obj {
                if s.dict.get(b"Type").name() == Some(b"XRef") {
                    trailers.push((found, s.dict.clone()));
                }
            }
            doc.objects.insert(num, (found, obj));
        }
        let mut pos = 0;
        while let Some(found) = find(data, b"trailer", pos) {
            pos = found + 7;
            if let Some(Obj::Dict(dict)) = Lexer::new(data, pos).object(true, 0) {
                trailers.push((found, dict));
            }
        }
        doc.expand_object_streams();
        if trailers
            .iter()
            .any(|(_, t)| !matches!(t.get(b"Encrypt"), Obj::Null))
        {
            return Err(PdfError::Encrypted);
        }
        trailers.sort_by_key(|(at, _)| *at);
        doc.root = trailers
            .iter()
            .rev()
            .map(|(_, t)| t.get(b"Root"))
            .find(|root| doc.resolve(root).dict().is_some())
            .cloned()
            .or_else(|| {
                doc.objects
                    .iter()
                    .filter(|(_, (_, o))| {
                        o.dict()
                            .is_some_and(|d| d.get(b"Type").name() == Some(b"Catalog"))
                    })
                    .max_by_key(|(_, (at, _))| *at)
                    .map(|(num, _)| Obj::Ref(*num))
            });
        Ok(doc)
    }

    /// Objects inside `/Type /ObjStm` streams; a definition replaces one at an earlier file position.
    fn expand_object_streams(&mut self) {
        let streams: Vec<(usize, Rc<Stream>)> = self
            .objects
            .values()
            .filter_map(|(at, o)| match o {
                Obj::Stream(s) if s.dict.get(b"Type").name() == Some(b"ObjStm") => {
                    Some((*at, s.clone()))
                }
                _ => None,
            })
            .collect();
        for (at, stream) in streams {
            let Some(body) = self.decode(&stream) else {
                continue;
            };
            let n = self.resolve(stream.dict.get(b"N")).num().unwrap_or(0.0) as usize;
            let first = self.resolve(stream.dict.get(b"First")).num().unwrap_or(0.0) as usize;
            let mut header = Lexer::new(&body, 0);
            for _ in 0..n.min(body.len()) {
                let (Some(Obj::Num(num)), Some(Obj::Num(off))) =
                    (header.object(false, 0), header.object(false, 0))
                else {
                    break;
                };
                let start = first.saturating_add(off as usize);
                if start >= body.len() {
                    continue;
                }
                let Some(obj) = Lexer::new(&body, start).object(true, 0) else {
                    continue;
                };
                let num = num as u32;
                if self.objects.get(&num).is_none_or(|(prev, _)| *prev < at) {
                    self.objects.insert(num, (at, obj));
                }
            }
        }
    }

    fn resolve<'o>(&'o self, mut obj: &'o Obj) -> &'o Obj {
        for _ in 0..MAX_REF_HOPS {
            match obj {
                Obj::Ref(num) => match self.objects.get(num) {
                    Some((_, target)) => obj = target,
                    None => return &Obj::Null,
                },
                _ => return obj,
            }
        }
        &Obj::Null
    }

    fn rdict<'o>(&'o self, obj: &'o Obj) -> Option<&'o Dict> {
        self.resolve(obj).dict()
    }

    /// The leaf pages in order: the catalog's page tree, else every `/Type /Page` object in file order.
    fn pages(&self) -> Vec<Page> {
        let mut out = Vec::new();
        if let Some(catalog) = self.root.as_ref().and_then(|r| self.rdict(r)) {
            let mut seen = HashSet::new();
            self.walk(catalog.get(b"Pages"), None, 0, &mut seen, &mut out);
        }
        if out.is_empty() {
            let mut leaves: Vec<(usize, &Dict)> = self
                .objects
                .values()
                .filter_map(|(at, o)| {
                    o.dict()
                        .filter(|d| d.get(b"Type").name() == Some(b"Page"))
                        .map(|d| (*at, d))
                })
                .collect();
            leaves.sort_by_key(|(at, _)| *at);
            out = leaves
                .into_iter()
                .map(|(_, d)| Page {
                    contents: d.get(b"Contents").clone(),
                    resources: self.rdict(d.get(b"Resources")).cloned(),
                })
                .collect();
        }
        out
    }

    fn walk(
        &self,
        node: &Obj,
        inherited: Option<&Dict>,
        depth: usize,
        seen: &mut HashSet<u32>,
        out: &mut Vec<Page>,
    ) {
        if depth > MAX_TREE_DEPTH {
            return;
        }
        if let Obj::Ref(num) = node {
            if !seen.insert(*num) {
                return;
            }
        }
        let Some(dict) = self.rdict(node) else {
            return;
        };
        let resources = self.rdict(dict.get(b"Resources")).or(inherited);
        match self.resolve(dict.get(b"Kids")) {
            Obj::Array(kids) if dict.get(b"Type").name() != Some(b"Page") => {
                for kid in kids {
                    self.walk(kid, resources, depth + 1, seen, out);
                }
            }
            _ => out.push(Page {
                contents: dict.get(b"Contents").clone(),
                resources: resources.cloned(),
            }),
        }
    }

    /// A stream's decoded bytes; `None` for a filter this reader does not implement or a failed decode.
    fn decode(&self, stream: &Stream) -> Option<Vec<u8>> {
        let mut data = self.data.get(stream.data.clone())?.to_vec();
        let filters = match self.resolve(stream.dict.get(b"Filter")) {
            Obj::Name(n) => vec![n.clone()],
            Obj::Array(items) => items
                .iter()
                .filter_map(|i| self.resolve(i).name().map(<[u8]>::to_vec))
                .collect(),
            _ => Vec::new(),
        };
        let parms: Vec<Option<&Dict>> = match self.resolve(stream.dict.get(b"DecodeParms")) {
            Obj::Array(items) => items.iter().map(|i| self.rdict(i)).collect(),
            other => vec![other.dict()],
        };
        for (i, filter) in filters.iter().enumerate() {
            let parm = parms.get(i).copied().flatten();
            data = match filter.as_slice() {
                b"FlateDecode" | b"Fl" => {
                    let out = match inflate::zlib(&data, MAX_STREAM_BYTES) {
                        Ok(out) => out,
                        Err((_, partial)) if !partial.is_empty() => partial,
                        Err(_) => return None,
                    };
                    self.predict(out, parm)?
                }
                b"ASCIIHexDecode" | b"AHx" => {
                    let mut lexer = Lexer::new(&data, 0);
                    lexer.hex()
                }
                b"ASCII85Decode" | b"A85" => ascii85(&data),
                b"LZWDecode" | b"LZW" => {
                    let early = parm
                        .map(|p| self.resolve(p.get(b"EarlyChange")).num().unwrap_or(1.0))
                        .unwrap_or(1.0);
                    self.predict(lzw(&data, early != 0.0), parm)?
                }
                _ => return None,
            };
            let total = self.decoded.get().saturating_add(data.len());
            self.decoded.set(total);
            if total > MAX_DECODED_BYTES {
                return None;
            }
        }
        Some(data)
    }

    /// The PNG predictors (`/Predictor` >= 10) of `/DecodeParms`; TIFF predictor 2 leaves the bytes as they are.
    fn predict(&self, data: Vec<u8>, parm: Option<&Dict>) -> Option<Vec<u8>> {
        let Some(parm) = parm else {
            return Some(data);
        };
        let get = |key: &[u8], default: f64| self.resolve(parm.get(key)).num().unwrap_or(default);
        if get(b"Predictor", 1.0) < 10.0 {
            return Some(data);
        }
        let colors = get(b"Colors", 1.0).max(1.0) as usize;
        let bpc = get(b"BitsPerComponent", 8.0).max(1.0) as usize;
        let columns = get(b"Columns", 1.0).max(1.0) as usize;
        let bpp = (colors * bpc).div_ceil(8).max(1);
        let row = (colors * bpc * columns).div_ceil(8);
        let mut out = Vec::with_capacity(data.len());
        let mut prev = vec![0u8; row];
        for chunk in data.chunks(row + 1) {
            let (kind, line) = chunk.split_first()?;
            let mut cur = line.to_vec();
            cur.resize(row, 0);
            for i in 0..row {
                let left = if i >= bpp { cur[i - bpp] } else { 0 };
                let up = prev[i];
                let corner = if i >= bpp { prev[i - bpp] } else { 0 };
                cur[i] = cur[i].wrapping_add(match kind {
                    1 => left,
                    2 => up,
                    3 => ((u16::from(left) + u16::from(up)) / 2) as u8,
                    4 => paeth(left, up, corner),
                    _ => 0,
                });
            }
            out.extend_from_slice(&cur[..line.len().min(row)]);
            prev = cur;
        }
        Some(out)
    }

    /// The concatenated decoded content of a `/Contents` stream or array.
    fn contents(&self, contents: &Obj) -> Vec<u8> {
        let mut out = Vec::new();
        let items: Vec<&Obj> = match self.resolve(contents) {
            Obj::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        for item in items {
            if let Obj::Stream(stream) = self.resolve(item) {
                if let Some(data) = self.decode(stream) {
                    out.extend_from_slice(&data);
                    out.push(b'\n');
                }
            }
        }
        out
    }

    fn page_text(&self, page: &Page) -> Result<String, PdfError> {
        let content = self.contents(&page.contents);
        let mut text = TextRun::default();
        let mut fonts = FontCache::default();
        self.run(
            &content,
            page.resources.as_ref(),
            &mut text,
            &mut fonts,
            IDENTITY,
            0,
        );
        if self.decoded.get() > MAX_DECODED_BYTES {
            return Err(PdfError::TooLarge);
        }
        Ok(tidy(&text.out))
    }

    /// Runs a content stream's text operators.
    fn run(
        &self,
        content: &[u8],
        resources: Option<&Dict>,
        text: &mut TextRun,
        fonts: &mut FontCache,
        ctm: [f64; 6],
        depth: usize,
    ) {
        let mut lexer = Lexer::new(content, 0);
        let mut operands: Vec<Obj> = Vec::new();
        let mut state = TextState {
            ctm,
            ..TextState::default()
        };
        let mut saved: Vec<TextState> = Vec::new();
        while let Some(tok) = lexer.token() {
            let op = match tok {
                Tok::Keyword(op) => op,
                tok => {
                    if let Some(obj) = lexer.compose(tok, false, 0) {
                        operands.push(obj);
                    }
                    if operands.len() > 4096 {
                        operands.clear();
                    }
                    continue;
                }
            };
            let num = |i: usize| -> f64 {
                operands
                    .len()
                    .checked_sub(i)
                    .and_then(|k| operands.get(k))
                    .and_then(Obj::num)
                    .unwrap_or(0.0)
            };
            match op {
                b"q" => saved.push(state.clone()),
                b"Q" => {
                    if let Some(s) = saved.pop() {
                        state = s;
                    }
                }
                b"BT" => {
                    state.tm = IDENTITY;
                    state.tlm = IDENTITY;
                }
                b"Tf" => {
                    state.size = num(1);
                    if let Some(Obj::Name(name)) =
                        operands.len().checked_sub(2).and_then(|k| operands.get(k))
                    {
                        state.font = fonts.get(self, resources, name);
                    }
                }
                b"Tc" => state.char_space = num(1),
                b"Tw" => state.word_space = num(1),
                b"Tz" => state.hscale = num(1) / 100.0,
                b"TL" => state.leading = num(1),
                b"Td" => state.move_line(num(2), num(1)),
                b"TD" => {
                    state.leading = -num(1);
                    state.move_line(num(2), num(1));
                }
                b"Tm" => {
                    state.tlm = [num(6), num(5), num(4), num(3), num(2), num(1)];
                    state.tm = state.tlm;
                }
                b"T*" => state.move_line(0.0, -state.leading),
                b"Tj" => {
                    if let Some(Obj::Str(s)) = operands.last() {
                        state.show(s, text);
                    }
                }
                b"'" => {
                    state.move_line(0.0, -state.leading);
                    if let Some(Obj::Str(s)) = operands.last() {
                        state.show(s, text);
                    }
                }
                b"\"" => {
                    state.word_space = num(3);
                    state.char_space = num(2);
                    state.move_line(0.0, -state.leading);
                    if let Some(Obj::Str(s)) = operands.last() {
                        state.show(s, text);
                    }
                }
                b"TJ" => {
                    if let Some(Obj::Array(items)) = operands.last() {
                        for item in items {
                            match item {
                                Obj::Str(s) => state.show(s, text),
                                Obj::Num(n) => state.advance(-n / 1000.0 * state.size),
                                _ => {}
                            }
                        }
                    }
                }
                b"cm" => {
                    let m = [num(6), num(5), num(4), num(3), num(2), num(1)];
                    state.ctm = mul(&m, &state.ctm);
                }
                b"Do" if depth < MAX_FORM_DEPTH => {
                    if let Some(Obj::Name(name)) = operands.last() {
                        self.form(name, resources, text, state.ctm, depth);
                    }
                }
                b"ID" => skip_inline_image(&mut lexer),
                _ => {}
            }
            operands.clear();
        }
    }

    fn form(
        &self,
        name: &[u8],
        resources: Option<&Dict>,
        text: &mut TextRun,
        ctm: [f64; 6],
        depth: usize,
    ) {
        let Some(xobjects) = resources.and_then(|r| self.rdict(r.get(b"XObject"))) else {
            return;
        };
        let Obj::Stream(stream) = self.resolve(xobjects.get(name)) else {
            return;
        };
        if stream.dict.get(b"Subtype").name() != Some(b"Form") {
            return;
        }
        let Some(content) = self.decode(stream) else {
            return;
        };
        let mut matrix = IDENTITY;
        if let Obj::Array(items) = self.resolve(stream.dict.get(b"Matrix")) {
            if items.len() == 6 {
                for (slot, item) in matrix.iter_mut().zip(items) {
                    *slot = self.resolve(item).num().unwrap_or(*slot);
                }
            }
        }
        let inner = self.rdict(stream.dict.get(b"Resources")).or(resources);
        self.run(
            &content,
            inner,
            text,
            &mut FontCache::default(),
            mul(&matrix, &ctm),
            depth + 1,
        );
    }
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = i16::from(a) + i16::from(b) - i16::from(c);
    let (pa, pb, pc) = (
        (p - i16::from(a)).abs(),
        (p - i16::from(b)).abs(),
        (p - i16::from(c)).abs(),
    );
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn ascii85(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut group = [0u32; 5];
    let mut n = 0;
    for &b in data {
        match b {
            b'~' => break,
            b'z' if n == 0 => out.extend_from_slice(&[0; 4]),
            b'!'..=b'u' => {
                group[n] = u32::from(b - b'!');
                n += 1;
                if n == 5 {
                    let v = group
                        .iter()
                        .fold(0u32, |acc, &d| acc.wrapping_mul(85).wrapping_add(d));
                    out.extend_from_slice(&v.to_be_bytes());
                    n = 0;
                }
            }
            _ => {}
        }
    }
    if n > 1 {
        for slot in group.iter_mut().skip(n) {
            *slot = 84;
        }
        let v = group
            .iter()
            .fold(0u32, |acc, &d| acc.wrapping_mul(85).wrapping_add(d));
        out.extend_from_slice(&v.to_be_bytes()[..n - 1]);
    }
    out
}

fn lzw(data: &[u8], early: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut table: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    table.push(Vec::new());
    table.push(Vec::new());
    let mut width = 9;
    let (mut buf, mut count) = (0u32, 0u32);
    let mut prev: Option<Vec<u8>> = None;
    for &b in data {
        buf = (buf << 8) | u32::from(b);
        count += 8;
        while count >= width {
            let code = ((buf >> (count - width)) & ((1 << width) - 1)) as usize;
            count -= width;
            match code {
                256 => {
                    table.truncate(258);
                    width = 9;
                    prev = None;
                    continue;
                }
                257 => return out,
                _ => {}
            }
            let entry = if code < table.len() {
                table[code].clone()
            } else if let Some(p) = &prev {
                let mut e = p.clone();
                e.push(p[0]);
                e
            } else {
                return out;
            };
            if out.len() + entry.len() > MAX_STREAM_BYTES {
                return out;
            }
            out.extend_from_slice(&entry);
            if let Some(mut p) = prev.take() {
                if table.len() < 4096 {
                    p.push(entry[0]);
                    table.push(p);
                }
            }
            prev = Some(entry);
            let next = table.len() + usize::from(early);
            width = match next {
                n if n >= 2048 => 12,
                n if n >= 1024 => 11,
                n if n >= 512 => 10,
                _ => 9,
            };
        }
    }
    out
}

/// After `ID`: one whitespace byte, then the image data up to `EI` between whitespace.
fn skip_inline_image(lexer: &mut Lexer) {
    let s = lexer.s;
    let mut i = lexer.pos + 1;
    while i + 2 <= s.len() {
        if &s[i..i + 2] == b"EI"
            && i > 0
            && is_white(s[i - 1])
            && s.get(i + 2).is_none_or(|&b| is_white(b))
        {
            lexer.pos = i + 2;
            return;
        }
        i += 1;
    }
    lexer.pos = s.len();
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= hay.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| i + from)
}

/// The object number of an `N G obj` header ending just before `at`.
fn object_header(data: &[u8], at: usize) -> Option<u32> {
    let mut i = at;
    let skip_white = |i: &mut usize| {
        while *i > 0 && is_white(data[*i - 1]) {
            *i -= 1;
        }
    };
    let digits = |i: &mut usize| -> Option<usize> {
        let end = *i;
        while *i > 0 && data[*i - 1].is_ascii_digit() {
            *i -= 1;
        }
        (*i < end && end - *i <= 10).then_some(end)
    };
    let before = i;
    skip_white(&mut i);
    if i == before {
        return None;
    }
    digits(&mut i)?;
    let gen_start = i;
    skip_white(&mut i);
    if i == gen_start {
        return None;
    }
    let num_end = digits(&mut i)?;
    if i > 0 && is_regular(data[i - 1]) {
        return None;
    }
    std::str::from_utf8(&data[i..num_end]).ok()?.parse().ok()
}

/// The encoded bytes of the stream whose dictionary ends at `pos`: a direct `/Length` when `endstream` follows
/// it, else up to the next `endstream` (end-of-line before it dropped).
fn stream_after(data: &[u8], pos: usize, dict: &Dict) -> Option<Range<usize>> {
    let mut lexer = Lexer::new(data, pos);
    lexer.skip_white();
    if !data[lexer.pos..].starts_with(b"stream") {
        return None;
    }
    let mut start = lexer.pos + 6;
    if data.get(start) == Some(&b'\r') {
        start += 1;
    }
    if data.get(start) == Some(&b'\n') {
        start += 1;
    }
    if let Some(len) = dict.get(b"Length").num() {
        let end = start.saturating_add(len.max(0.0) as usize);
        if end <= data.len() {
            let mut after = Lexer::new(data, end);
            after.skip_white();
            if data[after.pos..].starts_with(b"endstream") {
                return Some(start..end);
            }
        }
    }
    let found = find(data, b"endstream", start)?;
    let mut end = found;
    if end > start && data[end - 1] == b'\n' {
        end -= 1;
    }
    if end > start && data[end - 1] == b'\r' {
        end -= 1;
    }
    Some(start..end)
}

// -------------------------------------------------------------------------------------------------------------
// Fonts

/// How a font's codes become text.
struct Font {
    /// Bytes per code when no `/ToUnicode` codespace says otherwise.
    width: usize,
    cmap: Option<CMap>,
    /// A simple font's encoding (index = code).
    simple: Option<Vec<Option<String>>>,
    widths: Widths,
    /// Glyph space to text space (1/1000 except Type3's `/FontMatrix`).
    scale: f64,
}

enum Widths {
    Simple {
        first: u32,
        widths: Vec<f64>,
        missing: f64,
    },
    Cid {
        ranges: Vec<(u32, u32, f64)>,
        default: f64,
    },
}

impl Font {
    fn advance(&self, code: u32) -> f64 {
        let w = match &self.widths {
            Widths::Simple {
                first,
                widths,
                missing,
            } => code
                .checked_sub(*first)
                .and_then(|i| widths.get(i as usize))
                .copied()
                .filter(|w| *w > 0.0)
                .unwrap_or(*missing),
            Widths::Cid { ranges, default } => ranges
                .iter()
                .find(|(lo, hi, _)| (*lo..=*hi).contains(&code))
                .map_or(*default, |(_, _, w)| *w),
        };
        w * self.scale
    }

    /// The codes of a string with their text.
    fn decode(&self, bytes: &[u8]) -> Vec<(u32, String)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            // A simple font's codes are single bytes whatever its `/ToUnicode` codespace says.
            let len = if self.simple.is_some() {
                1
            } else {
                self.cmap
                    .as_ref()
                    .and_then(|c| c.code_len(&bytes[i..]))
                    .unwrap_or(self.width)
                    .min(bytes.len() - i)
                    .max(1)
            };
            let code = bytes[i..i + len]
                .iter()
                .fold(0u32, |acc, &b| (acc << 8) | u32::from(b));
            let text = self
                .cmap
                .as_ref()
                .and_then(|c| {
                    c.lookup(len, code)
                        .or_else(|| self.simple.is_some().then(|| c.lookup_any(code)).flatten())
                })
                .or_else(|| {
                    self.simple
                        .as_ref()
                        .and_then(|table| table.get(code as usize).cloned().flatten())
                })
                .unwrap_or_default();
            out.push((code, text));
            i += len;
        }
        out
    }
}

#[derive(Default)]
struct FontCache(HashMap<Vec<u8>, Option<Rc<Font>>>);

impl FontCache {
    fn get(&mut self, doc: &Doc, resources: Option<&Dict>, name: &[u8]) -> Option<Rc<Font>> {
        if let Some(font) = self.0.get(name) {
            return font.clone();
        }
        let font = resources
            .and_then(|r| doc.rdict(r.get(b"Font")))
            .and_then(|fonts| doc.rdict(fonts.get(name)))
            .map(|dict| {
                let key = std::ptr::from_ref(dict) as usize;
                if let Some(font) = doc.fonts.borrow().get(&key) {
                    return font.clone();
                }
                let font = Rc::new(load_font(doc, dict));
                doc.fonts.borrow_mut().insert(key, font.clone());
                font
            });
        self.0.insert(name.to_vec(), font.clone());
        font
    }
}

fn load_font(doc: &Doc, dict: &Dict) -> Font {
    let subtype = dict.get(b"Subtype").name().unwrap_or_default().to_vec();
    let cmap = match doc.resolve(dict.get(b"ToUnicode")) {
        Obj::Stream(s) => doc.decode(s).map(|data| CMap::parse(&data)),
        _ => None,
    };
    let num = |o: &Obj| doc.resolve(o).num();
    if subtype == b"Type0" {
        let descendant = match doc.resolve(dict.get(b"DescendantFonts")) {
            Obj::Array(items) => items.first().and_then(|d| doc.rdict(d)),
            _ => None,
        };
        let mut ranges = Vec::new();
        let default = descendant.and_then(|d| num(d.get(b"DW"))).unwrap_or(1000.0);
        if let Some(Obj::Array(w)) = descendant.map(|d| doc.resolve(d.get(b"W"))) {
            let mut i = 0;
            while i < w.len() {
                let Some(first) = num(&w[i]) else {
                    break;
                };
                match w.get(i + 1).map(|o| doc.resolve(o)) {
                    Some(Obj::Array(list)) => {
                        for (k, item) in list.iter().enumerate() {
                            if let Some(width) = num(item) {
                                let code = first as u32 + k as u32;
                                ranges.push((code, code, width));
                            }
                        }
                        i += 2;
                    }
                    Some(Obj::Num(last)) => {
                        let width = w.get(i + 2).and_then(num).unwrap_or(default);
                        ranges.push((first as u32, *last as u32, width));
                        i += 3;
                    }
                    _ => break,
                }
            }
        }
        return Font {
            width: 2,
            cmap,
            simple: None,
            widths: Widths::Cid { ranges, default },
            scale: 0.001,
        };
    }
    let base = match doc.resolve(dict.get(b"Encoding")) {
        Obj::Name(n) => Base::named(n),
        Obj::Dict(d) => d.get(b"BaseEncoding").name().and_then(Base::named),
        _ => None,
    };
    let symbolic = dict
        .get(b"BaseFont")
        .name()
        .is_some_and(|n| n.ends_with(b"Symbol") || n.ends_with(b"Dingbats"));
    let builtin = if base.is_none() && subtype == b"Type1" {
        builtin_encoding(doc, dict)
    } else {
        None
    };
    let mut table: Vec<Option<String>> = match builtin {
        Some(table) => table,
        None => (0..=255u8)
            .map(|code| match base {
                Some(base) => base.char(code).map(String::from),
                None if symbolic => None,
                None => encoding::winansi(code).map(String::from),
            })
            .collect(),
    };
    if let Obj::Dict(enc) = doc.resolve(dict.get(b"Encoding")) {
        if let Obj::Array(diffs) = doc.resolve(enc.get(b"Differences")) {
            let mut code = 0usize;
            for item in diffs {
                match doc.resolve(item) {
                    Obj::Num(n) => code = *n as usize,
                    Obj::Name(name) => {
                        if let Some(slot) = table.get_mut(code) {
                            *slot = encoding::glyph(name);
                        }
                        code += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    let first = num(dict.get(b"FirstChar")).unwrap_or(0.0).max(0.0) as u32;
    let widths = match doc.resolve(dict.get(b"Widths")) {
        Obj::Array(items) => items.iter().map(|w| num(w).unwrap_or(0.0)).collect(),
        _ => Vec::new(),
    };
    let missing = doc
        .rdict(dict.get(b"FontDescriptor"))
        .and_then(|d| num(d.get(b"MissingWidth")))
        .filter(|w| *w > 0.0)
        .unwrap_or(500.0);
    let scale = match (subtype.as_slice(), doc.resolve(dict.get(b"FontMatrix"))) {
        (b"Type3", Obj::Array(m)) => m.first().and_then(num).unwrap_or(0.001),
        _ => 0.001,
    };
    Font {
        width: 1,
        cmap,
        simple: Some(table),
        widths: Widths::Simple {
            first,
            widths,
            missing,
        },
        scale,
    }
}

/// The built-in encoding of an embedded Type 1 font program (`/FontFile`): the `dup <code> /<glyph> put` entries
/// of its clear-text part (TeX fonts carry their encoding only there). `None` for `StandardEncoding` or no program.
fn builtin_encoding(doc: &Doc, dict: &Dict) -> Option<Vec<Option<String>>> {
    let descriptor = doc.rdict(dict.get(b"FontDescriptor"))?;
    let Obj::Stream(stream) = doc.resolve(descriptor.get(b"FontFile")) else {
        return None;
    };
    let data = doc.decode(stream)?;
    let clear_len = doc
        .resolve(stream.dict.get(b"Length1"))
        .num()
        .map_or(data.len(), |n| (n.max(0.0) as usize).min(data.len()));
    let clear = &data[..clear_len];
    let at = find(clear, b"/Encoding", 0)?;
    let words = clear[at + 9..]
        .split(|&b| is_white(b))
        .filter(|w| !w.is_empty());
    let mut table: Vec<Option<String>> = vec![None; 256];
    let mut any = false;
    let mut window: [&[u8]; 3] = [b"", b"", b""];
    for word in words.take(4096) {
        if word == b"StandardEncoding" && !any {
            return None;
        }
        if word == b"def" || word == b"readonly" {
            if any {
                break;
            }
            continue;
        }
        if word == b"put" && window[0] == b"dup" {
            let code = std::str::from_utf8(window[1])
                .ok()
                .and_then(|s| s.parse::<usize>().ok());
            if let (Some(code), Some(name)) = (code, window[2].strip_prefix(b"/")) {
                if let Some(slot) = table.get_mut(code) {
                    *slot = encoding::glyph(name);
                    any = true;
                }
            }
        }
        window = [window[1], window[2], word];
    }
    any.then_some(table)
}

/// A `/ToUnicode` CMap: code space ranges, single codes and code ranges to UTF-16BE text.
#[derive(Default)]
struct CMap {
    space: Vec<(Vec<u8>, Vec<u8>)>,
    chars: HashMap<(usize, u32), String>,
    ranges: Vec<(usize, u32, u32, Vec<u16>)>,
}

impl CMap {
    fn parse(data: &[u8]) -> Self {
        let mut cmap = CMap::default();
        let mut lexer = Lexer::new(data, 0);
        let mut section: &[u8] = b"";
        let mut pending: Vec<Obj> = Vec::new();
        while let Some(tok) = lexer.token() {
            match tok {
                Tok::Keyword(word) => {
                    match word {
                        b"begincodespacerange" | b"beginbfchar" | b"beginbfrange" => {
                            section = word;
                        }
                        b"endcodespacerange" | b"endbfchar" | b"endbfrange" => section = b"",
                        _ => {}
                    }
                    pending.clear();
                }
                tok => {
                    let Some(obj) = lexer.compose(tok, false, 0) else {
                        continue;
                    };
                    pending.push(obj);
                    match (section, pending.as_slice()) {
                        (b"begincodespacerange", [Obj::Str(lo), Obj::Str(hi)]) => {
                            if lo.len() == hi.len() && (1..=4).contains(&lo.len()) {
                                cmap.space.push((lo.clone(), hi.clone()));
                            }
                            pending.clear();
                        }
                        (b"beginbfchar", [Obj::Str(src), dst]) => {
                            if let (Some(code), Some(text)) = (code_of(src), dst_text(dst)) {
                                cmap.chars.insert(code, text);
                            }
                            pending.clear();
                        }
                        (b"beginbfrange", [Obj::Str(lo), Obj::Str(hi), dst]) => {
                            cmap.add_range(lo, hi, dst);
                            pending.clear();
                        }
                        (b"", _) => pending.clear(),
                        (_, p) if p.len() >= 3 => pending.clear(),
                        _ => {}
                    }
                }
            }
        }
        cmap
    }

    fn add_range(&mut self, lo: &[u8], hi: &[u8], dst: &Obj) {
        let (Some((len, lo)), Some((hlen, hi))) = (code_of(lo), code_of(hi)) else {
            return;
        };
        if len != hlen || hi < lo || hi - lo >= MAX_CMAP_RANGE {
            return;
        }
        match dst {
            Obj::Str(base) if base.len() >= 2 => {
                let units: Vec<u16> = base
                    .chunks(2)
                    .map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]))
                    .collect();
                self.ranges.push((len, lo, hi, units));
            }
            Obj::Array(items) => {
                for (k, item) in items.iter().enumerate().take((hi - lo + 1) as usize) {
                    if let Some(text) = dst_text(item) {
                        self.chars.insert((len, lo + k as u32), text);
                    }
                }
            }
            _ => {}
        }
    }

    /// The byte length of the code at the start of `bytes` by the code space ranges.
    fn code_len(&self, bytes: &[u8]) -> Option<usize> {
        self.space
            .iter()
            .filter(|(lo, hi)| {
                bytes.len() >= lo.len()
                    && lo
                        .iter()
                        .zip(hi)
                        .zip(bytes)
                        .all(|((l, h), b)| (*l..=*h).contains(b))
            })
            .map(|(lo, _)| lo.len())
            .min()
    }

    /// A code by value at any code length (simple fonts whose CMap declares a wider codespace).
    fn lookup_any(&self, code: u32) -> Option<String> {
        (1..=4).find_map(|len| self.lookup(len, code))
    }

    fn lookup(&self, len: usize, code: u32) -> Option<String> {
        if let Some(text) = self.chars.get(&(len, code)) {
            return Some(text.clone());
        }
        self.ranges
            .iter()
            .rev()
            .find(|(l, lo, hi, _)| *l == len && (*lo..=*hi).contains(&code))
            .map(|(_, lo, _, base)| {
                let mut units = base.clone();
                if let Some(last) = units.last_mut() {
                    *last = last.wrapping_add((code - lo) as u16);
                }
                String::from_utf16_lossy(&units)
            })
    }
}

fn code_of(bytes: &[u8]) -> Option<(usize, u32)> {
    (1..=4).contains(&bytes.len()).then(|| {
        (
            bytes.len(),
            bytes.iter().fold(0u32, |acc, &b| (acc << 8) | u32::from(b)),
        )
    })
}

fn dst_text(dst: &Obj) -> Option<String> {
    match dst {
        Obj::Str(bytes) => {
            let units: Vec<u16> = bytes
                .chunks(2)
                .map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]))
                .collect();
            Some(String::from_utf16_lossy(&units))
        }
        Obj::Name(name) => encoding::glyph(name),
        _ => None,
    }
}

// -------------------------------------------------------------------------------------------------------------
// Text layout

const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

#[derive(Clone)]
struct TextState {
    tm: [f64; 6],
    tlm: [f64; 6],
    /// The current transformation matrix (`cm`; a Form XObject's `Matrix`).
    ctm: [f64; 6],
    font: Option<Rc<Font>>,
    size: f64,
    char_space: f64,
    word_space: f64,
    hscale: f64,
    leading: f64,
}

impl Default for TextState {
    fn default() -> Self {
        Self {
            tm: IDENTITY,
            tlm: IDENTITY,
            ctm: IDENTITY,
            font: None,
            size: 0.0,
            char_space: 0.0,
            word_space: 0.0,
            hscale: 1.0,
            leading: 0.0,
        }
    }
}

/// The page's text so far and where the last glyph ended (device space, x and baseline y).
#[derive(Default)]
struct TextRun {
    out: String,
    end: Option<(f64, f64)>,
}

impl TextState {
    fn move_line(&mut self, tx: f64, ty: f64) {
        let [a, b, c, d, e, f] = self.tlm;
        self.tlm = [a, b, c, d, e + tx * a + ty * c, f + tx * b + ty * d];
        self.tm = self.tlm;
    }

    /// Moves the text matrix `tx` text-space units along the line.
    fn advance(&mut self, tx: f64) {
        let tx = tx * self.hscale;
        self.tm[4] += tx * self.tm[0];
        self.tm[5] += tx * self.tm[1];
    }

    /// The text rendering matrix without the font size: the text matrix in device space.
    fn trm(&self) -> [f64; 6] {
        mul(&self.tm, &self.ctm)
    }

    /// The rendered font size: `Tf` size times the rendering matrix's vertical scale.
    fn em(&self) -> f64 {
        let m = self.trm();
        let scale = m[2].hypot(m[3]);
        (self.size * if scale > 0.0 { scale } else { 1.0 }).abs()
    }

    fn show(&mut self, bytes: &[u8], run: &mut TextRun) {
        let glyphs: Vec<(u32, String)> = match &self.font {
            Some(font) => font.decode(bytes),
            None => bytes
                .iter()
                .map(|&b| {
                    (
                        u32::from(b),
                        encoding::winansi(b).map(String::from).unwrap_or_default(),
                    )
                })
                .collect(),
        };
        let em = self.em().max(1e-3);
        for (code, text) in glyphs {
            let m = self.trm();
            let (x, y) = (m[4], m[5]);
            if let Some((ex, ey)) = run.end {
                // Distances along and across the baseline direction, in device space.
                let norm = m[0].hypot(m[1]).max(1e-9);
                let (dx, dy) = (x - ex, y - ey);
                let gap = (dx * m[0] + dy * m[1]) / norm;
                let line_moved = (dy * m[0] - dx * m[1]).abs() / norm > em * 0.5;
                let last_white = run.out.ends_with(char::is_whitespace) || run.out.is_empty();
                let starts_white = text.starts_with(char::is_whitespace);
                // A forward gap (character or word spacing included) is a word break; a small step back is
                // kerning, a long one a new column on the same baseline.
                if line_moved {
                    if !run.out.ends_with('\n') && !run.out.is_empty() {
                        run.out.push('\n');
                    }
                } else if (gap > em * 0.1 || gap < -em)
                    && !last_white
                    && !starts_white
                    && !text.is_empty()
                {
                    run.out.push(' ');
                }
            }
            run.out.push_str(&text);
            let width = self.font.as_ref().map_or(0.5, |f| f.advance(code));
            let single = self.font.as_ref().is_none_or(|f| f.width == 1);
            let word = if code == 32 && single {
                self.word_space
            } else {
                0.0
            };
            // The glyph's end is before the spacing that follows it.
            self.advance(width * self.size);
            if !text.is_empty() {
                let end = self.trm();
                run.end = Some((end[4], end[5]));
            }
            self.advance(self.char_space + word);
        }
    }
}

/// `a × b` for PDF's row-vector matrices `[a b c d e f]`.
fn mul(a: &[f64; 6], b: &[f64; 6]) -> [f64; 6] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

/// Control characters dropped, tabs as spaces, trailing spaces and runs of blank lines removed.
fn tidy(text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in text.split('\n') {
        let line: String = line
            .chars()
            .filter_map(|c| match c {
                '\t' => Some(' '),
                '\u{a0}' => Some(' '),
                c if c.is_control() => None,
                '\u{fffd}' | '\u{feff}' => None,
                c => Some(c),
            })
            .collect();
        let line = line.trim_end().to_owned();
        if line.is_empty() && lines.last().is_none_or(String::is_empty) {
            continue;
        }
        lines.push(line);
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_parse_leniently() {
        assert_eq!(number(b"12"), 12.0);
        assert_eq!(number(b"-.5"), -0.5);
        assert_eq!(number(b"4."), 4.0);
        assert_eq!(number(b"--5"), 0.0);
        assert_eq!(number(b"1.2.3"), 1.2);
    }

    #[test]
    fn literal_strings_unescape() {
        let mut lexer = Lexer::new(
            br"(a\(b\)c\\ \101\n(nested)\
x)",
            0,
        );
        let Some(Tok::Obj(Obj::Str(s))) = lexer.token() else {
            panic!("not a string");
        };
        assert_eq!(s, b"a(b)c\\ A\n(nested)x");
    }

    #[test]
    fn object_headers_need_two_integers() {
        let data = b"12 0 obj\nendobj 5 obj x0 0 obj";
        assert_eq!(object_header(data, 5), Some(12));
        assert_eq!(object_header(data, 18), None);
        assert_eq!(object_header(data, 27), None);
    }

    #[test]
    fn ascii85_and_lzw_decode() {
        assert_eq!(ascii85(b"87cURD]i,\"Ebo80~>"), b"Hello World!");
        assert_eq!(ascii85(b"87cURD]i,\"Ebo7~>"), b"Hello World");
        // `LZWDecode` example of the PDF reference (7.4.4.2), EarlyChange 1.
        let lzw_data = [0x80, 0x0b, 0x60, 0x50, 0x22, 0x0c, 0x0c, 0x85, 0x01];
        assert_eq!(lzw(&lzw_data, true), b"-----A---B");
    }

    #[test]
    fn tidy_drops_controls_and_blank_runs() {
        assert_eq!(tidy("a \u{1}b  \n\n\n\nc\t\n\n"), "a b\n\nc");
    }
}
