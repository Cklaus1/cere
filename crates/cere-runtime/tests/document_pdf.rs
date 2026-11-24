//! DocInput golden battery (doc-input design note DI-04/DI-05): PDFs built here, byte for byte, read back to
//! exact page text. Nothing is read from disk; `CERE_DOC_FIXTURE_DIR` (unset in CI) writes the fixtures out so a
//! developer can compare them against another reader (`pdftotext`, spec F7).

use cere_runtime::document::pdf::{self, PdfError};
use cere_runtime::document::{self, Budget, DocumentError, Meta, MAX_PAGES_PER_REQUEST, NO_TEXT};

mod document_support;

use document_support::{base64, hex, png_up, simple, stored_zlib, stream, Pdf, HELVETICA};

// ---------------------------------------------------------------------------------------------------------
// Fixtures: (name, PDF bytes, expected page texts).

type Fixture = (&'static str, Vec<u8>, Vec<String>);

fn pages(texts: &[&str]) -> Vec<String> {
    texts.iter().map(|t| (*t).to_owned()).collect()
}

fn winansi_three_pages() -> Fixture {
    let page1: &[u8] = b"BT /F1 12 Tf 14 TL 72 720 Td (Quarterly report) Tj T* \
        (Caf\\351 \\223quoted\\224 costs \\2005) Tj 0 -14 Td [(Ker) -50 (ning) -400 (gap)] TJ ET\n\
        BT /F1 12 Tf 1 0 0 1 72 600 Tm (Left) Tj 1 0 0 1 300 600 Tm (Right) Tj ET";
    let page2: &[u8] = b"BT /F1 10 Tf 72 700 Td (Second page) Tj ET";
    let page3: &[u8] = b"0 0 1 rg 72 72 100 100 re f";
    (
        "winansi_three_pages",
        simple(&[page1, page2, page3], HELVETICA, false, ""),
        pages(&[
            "Quarterly report\nCaf\u{e9} \u{201c}quoted\u{201d} costs \u{20ac}5\nKerning gap\nLeft Right",
            "Second page",
            "",
        ]),
    )
}

/// A content stream compressed by CPython `zlib.compress(..., 9)`: a dynamic-Huffman block.
const DYNAMIC: &str = "78da95cfc18e82301006e0fb3ec57fdc4dccaac4a05e8d9a3d6eb27d81a1b450848e140ac2d35bb8986cc2c1646e\
93f9fef94f02ebeb16db0842631f85d940a4f814b942ed8dbc2171dc5b687ea0f0d5bd0177caa10deb92c6012967df5f10052ee2e3f4dfdac\
4b3f54b81a90624c1e84d9b439b4e85cb515994a6f6ec029d358b4e7c8c66e7ef9e1bfb006b24e544d69e5c3baec2719aa929a0e37e19d91fd\
e2d06cac8d86531decde279b05419891faf7545363cc7f2d6409273c3e41987294872aad05252aa57d327cf447379";

fn flate_dynamic_huffman() -> Fixture {
    let mut pdf = Pdf::new("1.4");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>");
    pdf.obj(3, HELVETICA.as_bytes());
    pdf.obj(
        4,
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
    );
    pdf.obj(5, &stream("/Filter /FlateDecode", &hex(DYNAMIC)));
    pdf.xref(&[], "/Root 1 0 R");
    (
        "flate_dynamic_huffman",
        pdf.body,
        pages(&[
            "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.\n\
             Sphinx of black quartz, judge my vow.\nThe quick brown fox jumps over the lazy dog again.\n\
             Dynamic Huffman blocks carry their own code tables.",
        ]),
    )
}

const TO_UNICODE: &[u8] = b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
3 beginbfchar\n<0003> <0020>\n<0004> <D83DDE00>\n<0005> <00660069>\nendbfchar\n\
2 beginbfrange\n<0010> <0019> <0041>\n<0020> <0022> [<0078> <0079> <007A>]\nendbfrange\n\
endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend";

fn type0_to_unicode() -> Fixture {
    let mut pdf = Pdf::new("1.6");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>");
    pdf.obj(
        3,
        b"<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+Noto /Encoding /Identity-H \
          /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>",
    );
    pdf.obj(
        4,
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F2 3 0 R >> >> /Contents 5 0 R >>",
    );
    pdf.obj(
        5,
        &stream(
            "",
            b"BT /F2 12 Tf 72 700 Td <0010001100120003000400030005> Tj 0 -14 Td <002000210022> Tj ET",
        ),
    );
    pdf.obj(
        6,
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /ABCDEF+Noto /CIDSystemInfo \
          << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 /W [3 [600] 10 20 500] >>",
    );
    pdf.obj(7, &stream("/Filter /FlateDecode", &stored_zlib(TO_UNICODE)));
    pdf.xref(&[], "/Root 1 0 R");
    (
        "type0_to_unicode",
        pdf.body,
        pages(&["ABC \u{1f600} fi\nxyz"]),
    )
}

fn differences_encoding() -> Fixture {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding << /Type /Encoding \
                /BaseEncoding /WinAnsiEncoding /Differences [65 /uni03A9 /eacute 97 /f_i /a.sc] >> >>";
    (
        "differences_encoding",
        simple(
            &[b"BT /F1 12 Tf 72 700 Td (AB ab c) Tj ET"],
            font,
            false,
            "",
        ),
        pages(&["\u{3a9}\u{e9} fia c"]),
    )
}

/// PDF 1.5: the catalog, page tree and font inside a compressed object stream, a cross-reference stream (no
/// `trailer` keyword), and a content stream with PNG Up prediction.
fn object_stream_and_xref_stream() -> Fixture {
    let mut pdf = Pdf::new("1.5");
    let members: [(u32, &[u8]); 4] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        ),
        (4, HELVETICA.as_bytes()),
    ];
    let mut header = String::new();
    let mut objects = Vec::new();
    for (num, body) in members {
        header.push_str(&format!("{num} {} ", objects.len()));
        objects.extend_from_slice(body);
        objects.push(b'\n');
    }
    let mut data = header.clone().into_bytes();
    data.extend_from_slice(&objects);
    let content = b"BT /F1 12 Tf 72 700 Td (Packed in an object stream) Tj ET";
    pdf.obj(
        5,
        &stream(
            "/Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 4 >>",
            &stored_zlib(&png_up(content, 4)),
        ),
    );
    pdf.obj(
        6,
        &stream(
            &format!(
                "/Type /ObjStm /N 4 /First {} /Filter /FlateDecode",
                header.len()
            ),
            &stored_zlib(&data),
        ),
    );
    let xref_at = pdf.body.len();
    let mut entries = vec![0u8, 0, 0, 0, 0, 0xff, 0xff];
    for index in 0..4u16 {
        entries.push(2);
        entries.extend_from_slice(&6u32.to_be_bytes());
        entries.extend_from_slice(&index.to_be_bytes());
    }
    for off in [pdf.offset(5).unwrap(), pdf.offset(6).unwrap(), xref_at] {
        entries.push(1);
        entries.extend_from_slice(&(off as u32).to_be_bytes());
        entries.extend_from_slice(&[0, 0]);
    }
    pdf.obj(
        7,
        &stream("/Type /XRef /Size 8 /W [1 4 2] /Root 1 0 R", &entries),
    );
    pdf.body
        .extend_from_slice(format!("startxref\n{xref_at}\n%%EOF\n").as_bytes());
    (
        "object_stream_and_xref_stream",
        pdf.body,
        pages(&["Packed in an object stream"]),
    )
}

/// An incremental update replaces page 1's content: the newest definition wins.
fn incremental_update() -> Fixture {
    let mut pdf = Pdf::new("1.4");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>");
    pdf.obj(3, HELVETICA.as_bytes());
    pdf.obj(
        4,
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
    );
    pdf.obj(5, &stream("", b"BT /F1 12 Tf 72 700 Td (Old text) Tj ET"));
    let first = pdf.xref(&[], "/Root 1 0 R");
    pdf.obj(5, &stream("", b"BT /F1 12 Tf 72 700 Td (New text) Tj ET"));
    pdf.xref(&[5], &format!("/Root 1 0 R /Prev {first}"));
    ("incremental_update", pdf.body, pages(&["New text"]))
}

/// An inline image whose data contains `EI` not between whitespace, then a form XObject with its own font.
fn form_and_inline_image() -> Fixture {
    let mut pdf = Pdf::new("1.4");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>");
    pdf.obj(3, HELVETICA.as_bytes());
    pdf.obj(
        4,
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> /XObject << /Fm1 6 0 R >> >> \
          /Contents 5 0 R >>",
    );
    let mut content =
        b"BT /F1 12 Tf 72 700 Td (Before image) Tj ET\nq BI /W 4 /H 3 /BPC 8 /CS /G ID ".to_vec();
    content.extend_from_slice(b"\x00EI(x)Tj\xff\x01ab");
    content.extend_from_slice(
        b" EI Q\nq 1 0 0 1 0 -40 cm /Fm1 Do Q\nBT /F1 12 Tf 72 600 Td (After form) Tj ET",
    );
    pdf.obj(5, &stream("", &content));
    pdf.obj(
        6,
        &stream(
            "/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F9 7 0 R >> >>",
            b"BT /F9 12 Tf 72 650 Td (Inside form) Tj ET",
        ),
    );
    pdf.obj(7, HELVETICA.as_bytes());
    pdf.xref(&[], "/Root 1 0 R");
    (
        "form_and_inline_image",
        pdf.body,
        pages(&["Before image\nInside form\nAfter form"]),
    )
}

/// A scanned page: an image and no text.
fn scanned_page() -> Fixture {
    let mut pdf = Pdf::new("1.4");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    pdf.obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>",
    );
    pdf.obj(4, &stream("", b"q 612 0 0 792 0 0 cm /Im1 Do Q"));
    pdf.obj(
        5,
        &stream(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 8",
            b"\x00\xff\xff\x00",
        ),
    );
    pdf.xref(&[], "/Root 1 0 R");
    ("scanned_page", pdf.body, pages(&[""]))
}

/// Twenty pages; page 13 carries the fact the GPU row (DI-08) asks about.
fn twenty_pages() -> Fixture {
    let contents: Vec<Vec<u8>> = (1..=20)
        .map(|i| {
            let fact = if i == 13 {
                " T* (The vault code is 7-4-1-9.) Tj"
            } else {
                ""
            };
            format!("BT /F1 11 Tf 14 TL 72 720 Td (Page {i} of 20.) Tj{fact} ET").into_bytes()
        })
        .collect();
    let refs: Vec<&[u8]> = contents.iter().map(Vec::as_slice).collect();
    let expected: Vec<String> = (1..=20)
        .map(|i| {
            if i == 13 {
                "Page 13 of 20.\nThe vault code is 7-4-1-9.".to_owned()
            } else {
                format!("Page {i} of 20.")
            }
        })
        .collect();
    ("twenty_pages", simple(&refs, HELVETICA, true, ""), expected)
}

/// Every code 500 units wide, so other readers place glyphs exactly where this one does.
fn widths_500(first: u32, last: u32) -> String {
    let widths = vec!["500"; (last - first + 1) as usize].join(" ");
    format!("/FirstChar {first} /LastChar {last} /Widths [{widths}]")
}

/// groff's word breaks made by character spacing (`Tc`) inside a `TJ`, and text blocks placed by `cm` alone.
fn spacing_and_transforms() -> Fixture {
    let font = format!(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding {} >>",
        widths_500(32, 126)
    );
    let page: &[u8] =
        b"BT /F1 10 Tf 72 700 Td (corner) Tj 3 Tc [(,i) 300 (np) 300] TJ 0 Tc (ixels) Tj ET\n\
        q 1 0 0 1 72 650 cm BT /F1 10 Tf 0 0 Td (3) Tj ET Q\n\
        q 1 0 0 1 200 650 cm BT /F1 10 Tf 0 0 Td (Narrowing only) Tj ET Q";
    (
        "spacing_and_transforms",
        simple(&[page], &font, true, ""),
        pages(&["corner, in pixels\n3 Narrowing only"]),
    )
}

/// A Type 1 font with no `/Encoding`: the codes mean what the embedded program's clear-text part says (TeX).
fn type1_builtin_encoding() -> Fixture {
    let clear = b"%!PS-AdobeFont-1.0: XYZMR10 003.002\n/FontName /XYZMR10 def\n/Encoding 256 array\n\
        0 1 255 {1 index exch /.notdef put} for\ndup 22 /mu put\ndup 32 /space put\ndup 48 /zero put\n\
        dup 49 /one put\ndup 115 /s put\nreadonly def\ncurrentfile eexec\n";
    let mut program = clear.to_vec();
    program.extend_from_slice(&[0xd9, 0xd6, 0x6f, 0x63, 0x3b, 0x84, 0x6a, 0x98]);
    let mut pdf = Pdf::new("1.4");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>");
    pdf.obj(
        3,
        format!(
            "<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+XYZMR10 {} /FontDescriptor 6 0 R >>",
            widths_500(0, 255)
        )
        .as_bytes(),
    );
    pdf.obj(
        4,
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
    );
    pdf.obj(5, &stream("", b"BT /F1 12 Tf 72 700 Td (10 \\026s) Tj ET"));
    pdf.obj(
        6,
        b"<< /Type /FontDescriptor /FontName /ABCDEF+XYZMR10 /Flags 4 /FontBBox [0 -250 1000 750] \
          /ItalicAngle 0 /Ascent 750 /Descent -250 /CapHeight 700 /StemV 80 /FontFile 7 0 R >>",
    );
    pdf.obj(
        7,
        &stream(
            &format!("/Length1 {} /Length2 8 /Length3 0", clear.len()),
            &program,
        ),
    );
    pdf.xref(&[], "/Root 1 0 R");
    ("type1_builtin_encoding", pdf.body, pages(&["10 \u{b5}s"]))
}

/// A simple TrueType font whose `/ToUnicode` declares a two-byte codespace but maps one-byte codes (CUPS'
/// test page): codes stay single bytes, and the CMap wins over the encoding.
fn wide_codespace_to_unicode() -> Fixture {
    let cmap = b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
        /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\n\
        endcodespacerange\n7 beginbfchar\n<20> <0020>\n<21> <2713>\n<44> <0044>\n<65> <0065>\n<6E> <006E>\n\
        <6F> <006F>\n<50> <0050>\nendbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend";
    let mut pdf = Pdf::new("1.5");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>");
    pdf.obj(
        3,
        format!(
            "<< /Type /Font /Subtype /TrueType /BaseFont /ABCDEF+Ubuntu /Encoding /WinAnsiEncoding {} \
             /ToUnicode 6 0 R >>",
            widths_500(32, 126)
        )
        .as_bytes(),
    );
    pdf.obj(
        4,
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
    );
    pdf.obj(5, &stream("", b"BT /F1 17 Tf 72 700 Td (Done!) Tj ET"));
    pdf.obj(6, &stream("", cmap));
    pdf.xref(&[], "/Root 1 0 R");
    (
        "wide_codespace_to_unicode",
        pdf.body,
        pages(&["Done\u{2713}"]),
    )
}

fn fixtures() -> Vec<Fixture> {
    vec![
        winansi_three_pages(),
        flate_dynamic_huffman(),
        type0_to_unicode(),
        differences_encoding(),
        object_stream_and_xref_stream(),
        incremental_update(),
        form_and_inline_image(),
        scanned_page(),
        twenty_pages(),
        spacing_and_transforms(),
        type1_builtin_encoding(),
        wide_codespace_to_unicode(),
    ]
}

// ---------------------------------------------------------------------------------------------------------
// Golden battery.

#[test]
fn every_fixture_reads_to_its_golden_page_text() {
    for (name, bytes, expected) in fixtures() {
        let pages =
            pdf::extract(&bytes, MAX_PAGES_PER_REQUEST).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(pages, expected, "{name}");
    }
}

#[test]
fn a_rendered_pdf_has_page_markers_and_a_no_text_marker() {
    let (_, bytes, _) = winansi_three_pages();
    let rendered = Budget::default()
        .pdf(
            &bytes,
            Meta {
                title: Some("report.pdf"),
                context: None,
            },
        )
        .unwrap();
    assert_eq!(
        rendered,
        format!(
            "<document title=\"report.pdf\" media_type=\"application/pdf\" pages=\"3\">\n\
             <page number=\"1\">\nQuarterly report\nCaf\u{e9} \u{201c}quoted\u{201d} costs \u{20ac}5\nKerning gap\n\
             Left Right\n</page>\n<page number=\"2\">\nSecond page\n</page>\n<page number=\"3\">\n{NO_TEXT}\n\
             </page>\n</document>\n"
        )
    );
}

/// A scan (no page has a text layer) is not an empty success: it renders as one line naming the page count and
/// saying no text was found (maintainer, 2026-10-21; spec DI-D2).
#[test]
fn an_image_only_pdf_says_no_text_was_found() {
    let (_, scan, _) = scanned_page();
    let rendered = Budget::default()
        .pdf(
            &scan,
            Meta {
                title: Some("scan.pdf"),
                context: None,
            },
        )
        .unwrap();
    assert_eq!(
        rendered,
        "<document title=\"scan.pdf\" media_type=\"application/pdf\" pages=\"1\">\n\
         [no text found in this PDF's 1 page: no page has a text layer (a scan or an image-only file); CERE reads \
         a PDF's text layer only and renders no page images]\n</document>\n"
    );
    let three = simple(&[b"q Q", b"q 1 0 0 1 0 0 cm Q", b""], HELVETICA, true, "");
    let mut budget = Budget::default();
    let rendered = budget.pdf(&three, Meta::default()).unwrap();
    assert_eq!(
        rendered,
        format!(
            "<document media_type=\"application/pdf\" pages=\"3\">\n{}\n</document>\n",
            document::no_text_found(3)
        )
    );
    assert!(rendered.contains("no text found in this PDF's 3 pages"));
    // The pages still count against the request's page limit.
    let ninety_eight: Vec<&[u8]> = vec![b"q Q"; 98];
    let err = budget
        .pdf(
            &simple(&ninety_eight, HELVETICA, false, ""),
            Meta::default(),
        )
        .unwrap_err();
    assert_eq!(err, DocumentError::TooManyPages(101));
}

#[test]
fn the_twenty_page_document_keeps_page_order() {
    let (_, bytes, _) = twenty_pages();
    let encoded = base64(&bytes);
    let rendered = Budget::default()
        .base64(document::Media::Pdf, &encoded, Meta::default())
        .unwrap();
    let markers: Vec<usize> = (1..=20)
        .map(|i| rendered.find(&format!("<page number=\"{i}\">")).unwrap())
        .collect();
    assert!(markers.windows(2).all(|w| w[0] < w[1]));
    let page13 = &rendered[markers[12]..markers[13]];
    assert!(page13.contains("The vault code is 7-4-1-9."));
}

// ---------------------------------------------------------------------------------------------------------
// Refusals and limits.

#[test]
fn encrypted_pdfs_are_refused() {
    let bytes = simple(
        &[b"BT /F1 12 Tf (x) Tj ET"],
        HELVETICA,
        false,
        "/Encrypt << /Filter /Standard /V 1 /R 2 >>",
    );
    assert_eq!(pdf::extract(&bytes, 100), Err(PdfError::Encrypted));
}

#[test]
fn non_pdfs_and_empty_page_trees_are_refused() {
    assert_eq!(pdf::extract(b"hello world", 100), Err(PdfError::NotPdf));
    assert_eq!(pdf::extract(b"", 100), Err(PdfError::NotPdf));
    let mut pdf = Pdf::new("1.4");
    pdf.obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
    pdf.obj(2, b"<< /Type /Pages /Kids [] /Count 0 >>");
    pdf.xref(&[], "/Root 1 0 R");
    assert_eq!(pdf::extract(&pdf.body, 100), Err(PdfError::NoPages));
}

#[test]
fn page_limits_apply_per_document_and_per_request() {
    let contents: Vec<Vec<u8>> = (0..101)
        .map(|i| format!("BT /F1 12 Tf ({i}) Tj ET").into_bytes())
        .collect();
    let refs: Vec<&[u8]> = contents.iter().map(Vec::as_slice).collect();
    let big = simple(&refs, HELVETICA, false, "");
    assert_eq!(pdf::extract(&big, 100), Err(PdfError::TooManyPages(101)));
    assert_eq!(pdf::extract(&big, 101).unwrap().len(), 101);

    let sixty = simple(&refs[..60], HELVETICA, false, "");
    let mut budget = Budget::default();
    budget.pdf(&sixty, Meta::default()).unwrap();
    let err = budget.pdf(&sixty, Meta::default()).unwrap_err();
    assert_eq!(err, DocumentError::TooManyPages(120));
    assert!(err.to_string().contains("at most 100 PDF pages"), "{err}");
}

#[test]
fn damaged_files_never_panic() {
    for (name, bytes, _) in fixtures() {
        for cut in (0..bytes.len()).step_by(37) {
            let _ = pdf::extract(&bytes[..cut], 100);
        }
        let mut flipped = bytes.clone();
        for i in (9..flipped.len()).step_by(13) {
            flipped[i] ^= 0x5a;
            let _ = pdf::extract(&flipped, 100);
            flipped[i] ^= 0x5a;
        }
        let _ = name;
    }
}

/// Hostile nesting (`[[[[…` and `<<…<<` far past any real file) reads as nothing past the bound instead of
/// recursing until the stack overflows, in a content stream and in the file's own objects.
#[test]
fn deep_nesting_is_bounded() {
    let deep_array = "[".repeat(200_000);
    let deep_dict = "<< /K ".repeat(200_000);
    let content = format!("BT /F1 12 Tf 72 720 Td (Deep) Tj ET {deep_array} {deep_dict}");
    let bytes = simple(&[content.as_bytes()], HELVETICA, true, "");
    assert_eq!(pdf::extract(&bytes, 100).unwrap(), ["Deep"]);
    let bytes = simple(
        &[b"BT /F1 12 Tf 72 720 Td (Deep) Tj ET"],
        HELVETICA,
        false,
        &format!("/X {deep_array} /Y {deep_dict}"),
    );
    let _ = pdf::extract(&bytes, 100);
}

#[test]
fn a_truncated_file_still_reads_its_complete_pages() {
    let (_, bytes, _) = winansi_three_pages();
    // Drop the cross-reference table and trailer: the scan finds the objects and the catalog without them.
    let cut = bytes.windows(4).rposition(|w| w == b"xref").unwrap();
    let pages = pdf::extract(&bytes[..cut], 100).unwrap();
    assert_eq!(pages[1], "Second page");
}

/// Writes every fixture to `$CERE_DOC_FIXTURE_DIR` (developer cross-check against another PDF reader).
#[test]
fn fixtures_can_be_written_out() {
    let Some(dir) = std::env::var_os("CERE_DOC_FIXTURE_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    for (name, bytes, expected) in fixtures() {
        std::fs::write(dir.join(format!("{name}.pdf")), bytes).unwrap();
        std::fs::write(dir.join(format!("{name}.txt")), expected.join("\n\x0c")).unwrap();
    }
}

/// Extracts every `*.pdf` in `$CERE_DOC_EXTRACT_DIR` to `<file>.cere.txt` (pages split by form feeds, or the error)
/// for a developer comparison against another reader on real files (spec F8).
#[test]
fn pdfs_in_a_directory_can_be_extracted() {
    let Some(dir) = std::env::var_os("CERE_DOC_EXTRACT_DIR") else {
        return;
    };
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("pdf") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let text = match pdf::extract(&bytes, usize::MAX) {
            Ok(pages) => pages.join("\n\x0c"),
            Err(e) => format!("ERROR: {e}"),
        };
        std::fs::write(path.with_extension("pdf.cere.txt"), text).unwrap();
    }
}
