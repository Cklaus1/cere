//! DocInput test support: a minimal PDF writer (every fixture is built in the test, nothing read from disk).
#![allow(dead_code)]

pub struct Pdf {
    pub body: Vec<u8>,
    offsets: Vec<(u32, usize)>,
}

impl Pdf {
    pub fn new(version: &str) -> Self {
        let mut body = format!("%PDF-{version}\n").into_bytes();
        body.extend_from_slice(b"%\xe2\xe3\xcf\xd3\n");
        Self {
            body,
            offsets: Vec::new(),
        }
    }

    pub fn obj(&mut self, num: u32, content: &[u8]) {
        self.offsets.push((num, self.body.len()));
        self.body
            .extend_from_slice(format!("{num} 0 obj\n").as_bytes());
        self.body.extend_from_slice(content);
        self.body.extend_from_slice(b"\nendobj\n");
    }

    pub fn offset(&self, num: u32) -> Option<usize> {
        self.offsets
            .iter()
            .rev()
            .find(|(n, _)| *n == num)
            .map(|(_, at)| *at)
    }

    /// A classic cross-reference section for `nums` (all objects when empty) and its trailer; returns its offset.
    pub fn xref(&mut self, nums: &[u32], trailer: &str) -> usize {
        let at = self.body.len();
        let size = self.offsets.iter().map(|(n, _)| *n).max().unwrap_or(0) + 1;
        let mut out = String::from("xref\n");
        if nums.is_empty() {
            out.push_str(&format!("0 {size}\n0000000000 65535 f \n"));
            for num in 1..size {
                match self.offset(num) {
                    Some(off) => out.push_str(&format!("{off:010} 00000 n \n")),
                    None => out.push_str("0000000000 65535 f \n"),
                }
            }
        } else {
            for &num in nums {
                let off = self.offset(num).unwrap();
                out.push_str(&format!("{num} 1\n{off:010} 00000 n \n"));
            }
        }
        out.push_str(&format!(
            "trailer\n<< /Size {size} {trailer} >>\nstartxref\n{at}\n%%EOF\n"
        ));
        self.body.extend_from_slice(out.as_bytes());
        at
    }
}

pub fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut out = format!("<< /Length {} {dict} >>\nstream\n", data.len()).into_bytes();
    out.extend_from_slice(data);
    out.extend_from_slice(b"\nendstream");
    out
}

/// zlib with stored (uncompressed) deflate blocks.
pub fn stored_zlib(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let chunks: Vec<&[u8]> = if data.is_empty() {
        vec![&[]]
    } else {
        data.chunks(65_535).collect()
    };
    for (i, chunk) in chunks.iter().enumerate() {
        out.push(u8::from(i + 1 == chunks.len()));
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

/// PNG "Up" prediction (`/Predictor 12`) over rows of `columns` bytes; `data` is padded with spaces to whole rows.
pub fn png_up(data: &[u8], columns: usize) -> Vec<u8> {
    let mut data = data.to_vec();
    while data.len() % columns != 0 {
        data.push(b' ');
    }
    let mut out = Vec::new();
    let mut prev = vec![0u8; columns];
    for row in data.chunks(columns) {
        out.push(2);
        out.extend(row.iter().zip(&prev).map(|(b, p)| b.wrapping_sub(*p)));
        prev = row.to_vec();
    }
    out
}

pub fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

pub const HELVETICA: &str =
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";

/// Catalog 1, pages 2, font F1 3, then a page and its content per entry (page 4+2i, content 5+2i).
pub fn simple(contents: &[&[u8]], font: &str, flate: bool, trailer: &str) -> Vec<u8> {
    let mut pdf = Pdf::new("1.4");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    let kids: Vec<String> = (0..contents.len())
        .map(|i| format!("{} 0 R", 4 + 2 * i))
        .collect();
    pdf.obj(
        2,
        format!(
            "<< /Type /Pages /Kids [{}] /Count {} /Resources << /Font << /F1 3 0 R >> >> >>",
            kids.join(" "),
            contents.len()
        )
        .as_bytes(),
    );
    pdf.obj(3, font.as_bytes());
    for (i, content) in contents.iter().enumerate() {
        let page = 4 + 2 * i as u32;
        pdf.obj(
            page,
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R >>",
                page + 1
            )
            .as_bytes(),
        );
        let body = if flate {
            stream("/Filter /FlateDecode", &stored_zlib(content))
        } else {
            stream("", content)
        };
        pdf.obj(page + 1, &body);
    }
    pdf.xref(&[], &format!("/Root 1 0 R {trailer}"));
    pdf.body
}

/// A Helvetica PDF with one line of text per page.
pub fn text_pdf(pages: &[&str]) -> Vec<u8> {
    let contents: Vec<Vec<u8>> = pages
        .iter()
        .map(|text| format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET").into_bytes())
        .collect();
    let refs: Vec<&[u8]> = contents.iter().map(Vec::as_slice).collect();
    simple(&refs, HELVETICA, true, "")
}

pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let v = chunk
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &b)| acc | (u32::from(b) << (16 - 8 * i)));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((v >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}
