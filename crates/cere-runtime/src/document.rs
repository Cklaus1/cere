//! Document inputs (doc-input design note DI-01..DI-06): an Anthropic `document` block or an OpenAI `file`
//! content part becomes text that the chat template renders. A PDF becomes its extracted text page by page, each
//! page between page markers ([`pdf`]). A text document is rendered as sent. Limits apply per request, history
//! included. Error messages never quote document bytes.
//!
//! CERE renders no page images: the serving image has no PDF rasterizer (spec F5). A page with no text layer
//! renders as a marker saying so; a PDF none of whose pages has one (a scan) renders as one line naming its page
//! count and saying no text was found ([`no_text_found`]).
//!
//! The readers parse untrusted bytes in safe code only (`forbid(unsafe_code)` below, which binds every
//! submodule), and a panic inside the PDF reader fails that one document with [`DocumentError::ReaderFault`]
//! (a 400) instead of unwinding into the request handler (`contained`).

#![forbid(unsafe_code)]

mod encoding;
mod inflate;
pub mod pdf;

use std::fmt;
use std::fmt::Write as _;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Semaphore;

/// The decoded bytes allowed in one document (Anthropic's 32 MB request limit).
pub const MAX_DOCUMENT_BYTES: usize = 32 * 1024 * 1024;
/// The PDF pages allowed in one request (Anthropic's 100-page limit), history included.
pub const MAX_PAGES_PER_REQUEST: usize = 100;
/// The rendered document text allowed in one request, history included. 1 MiB is beyond any served context
/// window (262,144 tokens at ~4 bytes per token), so this bounds extraction work, not the prompt; the context
/// window's own 400 still applies.
pub const MAX_TEXT_BYTES_PER_REQUEST: usize = 1 << 20;

/// Audit F5: documents extracted at once, process-wide. Extraction is CPU work (inflate, PDF parse) of up to
/// [`MAX_DOCUMENT_BYTES`] per document and runs before admission, so no queue bounds it; a few hostile documents must
/// not occupy every core or every async worker.
pub const MAX_CONCURRENT_EXTRACTIONS: usize = 4;
/// Audit F5: the longest a request waits for an extraction slot and its extraction together; past it the request is
/// refused `503 overloaded` (an extraction already running is not interrupted, and keeps its slot until it ends).
pub const EXTRACTION_DEADLINE: Duration = Duration::from_secs(30);

static EXTRACTIONS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_EXTRACTIONS);

/// Why [`run`] did not return the work's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Offload {
    /// No extraction slot, or no result, within [`EXTRACTION_DEADLINE`].
    Busy,
    /// The blocking task panicked (the readers contain their own panics; this is anything else).
    Failed,
}

/// Whether the parsed request holds a `file` content part or a `document` block: an object whose `type` is one of
/// them, anywhere in the body (a looser test than the converters', which is the safe side: a false positive only
/// sends the work through [`run`]).
pub fn carries_documents(body: &Value) -> bool {
    match body {
        Value::Object(fields) => {
            matches!(fields.get("type"), Some(Value::String(t)) if t == "file" || t == "document")
                || fields.values().any(carries_documents)
        }
        Value::Array(items) => items.iter().any(carries_documents),
        _ => false,
    }
}

/// [`carries_documents`] of request `bytes`; a body that is not JSON carries none (its conversion refuses it).
pub fn bytes_carry_documents(bytes: &[u8]) -> bool {
    serde_json::from_slice::<Value>(bytes).is_ok_and(|body| carries_documents(&body))
}

/// Audit F5: `work`, which extracts and decodes documents, run where it cannot stall the async workers: inline
/// when `carries` is false (nothing to extract, no cost to move), else on the blocking pool behind
/// [`MAX_CONCURRENT_EXTRACTIONS`] slots and within [`EXTRACTION_DEADLINE`].
pub async fn run<T: Send + 'static>(
    carries: bool,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Offload> {
    if !carries {
        return Ok(work());
    }
    within(&EXTRACTIONS, EXTRACTION_DEADLINE, work).await
}

/// `work` on the blocking pool once `gate` has a slot, both within `deadline`. The slot moves into the task, so a
/// request that gave up at the deadline leaves its slot taken until its extraction really ends: the bound holds
/// for the work, not for the callers waiting on it.
async fn within<T: Send + 'static>(
    gate: &'static Semaphore,
    deadline: Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Offload> {
    let slot_and_run = async {
        let slot = gate
            .acquire()
            .await
            .expect("the extraction gate is never closed");
        tokio::task::spawn_blocking(move || {
            let _slot = slot;
            work()
        })
        .await
    };
    match tokio::time::timeout(deadline, slot_and_run).await {
        Err(_) => Err(Offload::Busy),
        Ok(Err(_)) => Err(Offload::Failed),
        Ok(Ok(value)) => Ok(value),
    }
}

/// The marker that stands in for a page with no text layer.
pub const NO_TEXT: &str = "[no text layer on this page]";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Media {
    Pdf,
    Text,
}

impl Media {
    /// `application/pdf` or `text/plain` (parameters such as `; charset=utf-8` are ignored).
    pub fn of(media_type: &str) -> Option<Self> {
        let essence = media_type.split(';').next().unwrap_or_default().trim();
        if essence.eq_ignore_ascii_case("application/pdf") {
            Some(Self::Pdf)
        } else if essence.eq_ignore_ascii_case("text/plain") {
            Some(Self::Text)
        } else {
            None
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pdf => "application/pdf",
            Self::Text => "text/plain",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocumentError {
    /// A media type other than `application/pdf` or `text/plain` (the type, at most 64 characters).
    MediaType(String),
    /// Not `data:application/pdf;base64,` or `data:text/plain;base64,`.
    Scheme,
    /// The payload is not strict RFC 4648 base64.
    Base64,
    /// A text document that is not UTF-8.
    NotUtf8,
    /// More than [`MAX_DOCUMENT_BYTES`] decoded bytes.
    TooLarge(usize),
    Pdf(pdf::PdfError),
    /// The request's PDFs have more than [`MAX_PAGES_PER_REQUEST`] pages (the count so far).
    TooManyPages(usize),
    /// The request's rendered documents exceed [`MAX_TEXT_BYTES_PER_REQUEST`] (the bytes so far).
    TextTooLarge(usize),
    /// The PDF reader panicked on this document (a CERE defect, contained by `contained`): it was not read.
    ReaderFault,
}

impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MediaType(media) => write!(
                f,
                "document media type '{media}' is not supported; use application/pdf or text/plain"
            ),
            Self::Scheme => f.write_str(
                "only data:application/pdf;base64, and data:text/plain;base64, file data is accepted",
            ),
            Self::Base64 => f.write_str("the document data is not valid padded base64 (RFC 4648)"),
            Self::NotUtf8 => f.write_str("a text/plain document must be UTF-8"),
            Self::TooLarge(bytes) => write!(
                f,
                "the document is {bytes} bytes; at most {MAX_DOCUMENT_BYTES} bytes are allowed per document"
            ),
            Self::TooManyPages(pages) => write!(
                f,
                "the request's PDF documents have at least {pages} pages; at most {MAX_PAGES_PER_REQUEST} PDF \
                 pages are allowed per request (history included)"
            ),
            Self::Pdf(e) => write!(f, "the PDF could not be read: {e}"),
            Self::TextTooLarge(bytes) => write!(
                f,
                "the request's documents render to {bytes} bytes of text; at most \
                 {MAX_TEXT_BYTES_PER_REQUEST} are allowed per request (history included)"
            ),
            Self::ReaderFault => f.write_str(
                "the PDF reader failed on this document, which was not read (a CERE defect; the bytes are not \
                 logged)",
            ),
        }
    }
}

impl DocumentError {
    /// The deployment cannot read this kind of document (400 `unsupported_parameter`); any other error is a broken
    /// document or one over a limit (400 `invalid_value`).
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::MediaType(_) | Self::Scheme | Self::Pdf(pdf::PdfError::Encrypted)
        )
    }
}

/// What a document block says about itself besides its source.
#[derive(Clone, Copy, Debug, Default)]
pub struct Meta<'a> {
    pub title: Option<&'a str>,
    pub context: Option<&'a str>,
}

/// The pages and rendered bytes a request's documents have used so far (history included).
#[derive(Debug, Default)]
pub struct Budget {
    pages: usize,
    text_bytes: usize,
}

impl Budget {
    /// A base64 payload of `media`, rendered.
    pub fn base64(
        &mut self,
        media: Media,
        payload: &str,
        meta: Meta,
    ) -> Result<String, DocumentError> {
        let bytes = decode_base64(payload)?;
        match media {
            Media::Pdf => self.pdf(&bytes, meta),
            Media::Text => {
                let text = std::str::from_utf8(&bytes).map_err(|_| DocumentError::NotUtf8)?;
                self.text(text, meta)
            }
        }
    }

    /// A text document, rendered.
    pub fn text(&mut self, text: &str, meta: Meta) -> Result<String, DocumentError> {
        if text.len() > MAX_DOCUMENT_BYTES {
            return Err(DocumentError::TooLarge(text.len()));
        }
        let mut out = open(Media::Text, meta, None);
        out.push_str(text);
        if !text.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("</document>\n");
        self.admit(out)
    }

    /// A PDF, rendered page by page.
    pub fn pdf(&mut self, bytes: &[u8], meta: Meta) -> Result<String, DocumentError> {
        if bytes.len() > MAX_DOCUMENT_BYTES {
            return Err(DocumentError::TooLarge(bytes.len()));
        }
        let left = MAX_PAGES_PER_REQUEST - self.pages;
        let pages = contained(|| pdf::extract(bytes, left))?.map_err(|e| match e {
            pdf::PdfError::TooManyPages(n) => DocumentError::TooManyPages(self.pages + n),
            e => DocumentError::Pdf(e),
        })?;
        self.pages += pages.len();
        let mut out = open(Media::Pdf, meta, Some(pages.len()));
        if pages.iter().all(|page| page.is_empty()) {
            out.push_str(&no_text_found(pages.len()));
            out.push_str("\n</document>\n");
            return self.admit(out);
        }
        for (i, page) in pages.iter().enumerate() {
            let _ = writeln!(out, "<page number=\"{}\">", i + 1);
            out.push_str(if page.is_empty() { NO_TEXT } else { page });
            out.push_str("\n</page>\n");
        }
        out.push_str("</document>\n");
        self.admit(out)
    }

    fn admit(&mut self, rendered: String) -> Result<String, DocumentError> {
        self.text_bytes += rendered.len();
        if self.text_bytes > MAX_TEXT_BYTES_PER_REQUEST {
            return Err(DocumentError::TextTooLarge(self.text_bytes));
        }
        Ok(rendered)
    }
}

/// The one line a PDF with no text on any of its `pages` renders as (a scan or an image-only file).
pub fn no_text_found(pages: usize) -> String {
    let noun = if pages == 1 { "page" } else { "pages" };
    format!(
        "[no text found in this PDF's {pages} {noun}: no page has a text layer (a scan or an image-only file); \
         CERE reads a PDF's text layer only and renders no page images]"
    )
}

/// Runs `read` (the PDF reader on untrusted bytes) with a panic in it contained: the panic fails this document
/// with [`DocumentError::ReaderFault`] and the request gets a 400, while the handler, its worker thread and every
/// other request carry on. The panic message (a source location, never document bytes) still reaches stderr through
/// the default hook. This relies on unwinding panics, the workspace default (`release_profile_unwinds` guards it).
fn contained<T>(read: impl FnOnce() -> T) -> Result<T, DocumentError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(read))
        .map_err(|_| DocumentError::ReaderFault)
}

/// `<document title=".." media_type=".." pages="..">` and the `<context>` element.
fn open(media: Media, meta: Meta, pages: Option<usize>) -> String {
    let mut out = String::from("<document");
    if let Some(title) = meta.title.filter(|t| !t.is_empty()) {
        let _ = write!(out, " title=\"{}\"", attribute(title));
    }
    let _ = write!(out, " media_type=\"{}\"", media.as_str());
    if let Some(pages) = pages {
        let _ = write!(out, " pages=\"{pages}\"");
    }
    out.push_str(">\n");
    if let Some(context) = meta.context.filter(|c| !c.is_empty()) {
        out.push_str("<context>\n");
        out.push_str(context);
        out.push_str("\n</context>\n");
    }
    out
}

fn attribute(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\n' | '\r' | '\t' => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// A rendered document placed after `previous` text: a line break first unless `previous` is empty or ends one.
pub fn after(previous: &str, rendered: String) -> String {
    if previous.is_empty() || previous.ends_with('\n') {
        rendered
    } else {
        format!("\n{rendered}")
    }
}

/// The media and base64 payload of an OpenAI `file_data` data URL.
pub fn data_url(url: &str) -> Result<(Media, &str), DocumentError> {
    if let Some(payload) = url.strip_prefix("data:application/pdf;base64,") {
        Ok((Media::Pdf, payload))
    } else if let Some(payload) = url.strip_prefix("data:text/plain;base64,") {
        Ok((Media::Text, payload))
    } else {
        Err(DocumentError::Scheme)
    }
}

/// Strict RFC 4648 base64 (standard alphabet, padded, no whitespace); the size is checked before decoding.
pub fn decode_base64(payload: &str) -> Result<Vec<u8>, DocumentError> {
    let text = payload.as_bytes();
    if text.len() % 4 != 0 {
        return Err(DocumentError::Base64);
    }
    let decoded = text.len() / 4 * 3;
    if decoded > MAX_DOCUMENT_BYTES + 2 {
        return Err(DocumentError::TooLarge(decoded));
    }
    let mut out = Vec::with_capacity(decoded);
    for (i, chunk) in text.chunks(4).enumerate() {
        let last = (i + 1) * 4 == text.len();
        let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
        if pad > 2 || (pad > 0 && !last) {
            return Err(DocumentError::Base64);
        }
        let mut v = 0u32;
        for &b in &chunk[..4 - pad] {
            v = (v << 6) | u32::from(sextet(b).ok_or(DocumentError::Base64)?);
        }
        v <<= 6 * pad as u32;
        let bytes = v.to_be_bytes();
        // Canonical: the padding bits are zero.
        if (pad == 1 && bytes[3] != 0) || (pad == 2 && (bytes[2] != 0 || bytes[3] != 0)) {
            return Err(DocumentError::Base64);
        }
        out.extend_from_slice(&bytes[1..4 - pad]);
    }
    if out.len() > MAX_DOCUMENT_BYTES {
        return Err(DocumentError::TooLarge(out.len()));
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::mpsc;

    /// Audit F5: the body test finds a `file` part or `document` block wherever it nests (escapes in the JSON text
    /// included, as it reads the parsed body) and nothing in a plain chat.
    #[test]
    fn documents_are_found_in_the_parsed_body() {
        assert!(carries_documents(
            &json!({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "file", "file": {"file_data": "x"}}
            ]}]})
        ));
        assert!(carries_documents(
            &json!({"messages": [{"role": "user", "content": [
                {"type": "tool_result", "content": [{"type": "document", "source": {}}]}
            ]}]})
        ));
        assert!(bytes_carry_documents(
            br#"{"m":[{"type":"\u0064ocument"}]}"#
        ));
        assert!(!carries_documents(
            &json!({"messages": [{"role": "user", "content": "a file, a document"}],
            "type": "text"})
        ));
        assert!(!bytes_carry_documents(b"not json"));
    }

    /// Audit F5: no documents, no cost: the work runs inline, on the caller's thread, past a full gate.
    #[tokio::test(flavor = "current_thread")]
    async fn work_without_documents_runs_inline() {
        let here = std::thread::current().id();
        let ran = run(false, || std::thread::current().id()).await.unwrap();
        assert_eq!(ran, here);
    }

    /// Audit F5: documents run on the blocking pool, not the async worker that awaits them.
    #[tokio::test(flavor = "current_thread")]
    async fn documents_run_off_the_async_worker() {
        static GATE: Semaphore = Semaphore::const_new(2);
        let here = std::thread::current().id();
        let ran = within(&GATE, Duration::from_secs(10), || {
            std::thread::current().id()
        })
        .await
        .unwrap();
        assert_ne!(ran, here);
        assert_eq!(GATE.available_permits(), 2, "the slot returns");
    }

    /// Audit F5: with every slot taken, a request is refused at the deadline and its work never runs; a slot
    /// freed later serves the next one.
    #[tokio::test(flavor = "current_thread")]
    async fn a_full_gate_refuses_at_the_deadline() {
        static GATE: Semaphore = Semaphore::const_new(1);
        let held = GATE.acquire().await.unwrap();
        let (tx, rx) = mpsc::channel();
        let started = std::time::Instant::now();
        let refused = within(&GATE, Duration::from_millis(100), move || {
            tx.send(()).unwrap()
        })
        .await;
        assert_eq!(refused, Err(Offload::Busy));
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(
            rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "the work did not run"
        );
        drop(held);
        assert_eq!(within(&GATE, Duration::from_secs(10), || 7).await, Ok(7));
    }

    /// Audit F5: a request that gives up at the deadline while its extraction runs leaves the slot taken until the
    /// extraction ends, so abandoned work still counts against the bound.
    #[tokio::test(flavor = "current_thread")]
    async fn an_abandoned_extraction_keeps_its_slot() {
        static GATE: Semaphore = Semaphore::const_new(1);
        let (release, wait) = mpsc::channel::<()>();
        let (running, is_running) = mpsc::channel::<()>();
        let slow = within(&GATE, Duration::from_millis(100), move || {
            running.send(()).unwrap();
            let _ = wait.recv_timeout(Duration::from_secs(30));
        })
        .await;
        assert_eq!(slow, Err(Offload::Busy));
        is_running.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(
            GATE.available_permits(),
            0,
            "the abandoned work still holds the slot"
        );
        release.send(()).unwrap();
        let start = std::time::Instant::now();
        while GATE.available_permits() == 0 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "the slot never came back"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Audit F5: a panic of the work is `Failed`, not a hang, and returns the slot.
    #[tokio::test(flavor = "current_thread")]
    async fn a_panicking_extraction_is_failed_and_returns_its_slot() {
        static GATE: Semaphore = Semaphore::const_new(1);
        let failed = within(&GATE, Duration::from_secs(10), || -> u8 {
            panic!("reader")
        })
        .await;
        assert_eq!(failed, Err(Offload::Failed));
        assert_eq!(GATE.available_permits(), 1);
    }

    #[test]
    fn base64_is_strict() {
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("aGk=").unwrap(), b"hi");
        assert_eq!(decode_base64("").unwrap(), b"");
        assert_eq!(decode_base64("aGVsbG8"), Err(DocumentError::Base64));
        assert_eq!(decode_base64("aGVs bG8="), Err(DocumentError::Base64));
        assert_eq!(decode_base64("aGk9"), Ok(b"hi=".to_vec()));
        assert_eq!(decode_base64("aGl="), Err(DocumentError::Base64));
        assert_eq!(decode_base64("aG==aGk="), Err(DocumentError::Base64));
    }

    #[test]
    fn text_documents_render_with_title_and_context() {
        let mut budget = Budget::default();
        let out = budget
            .text(
                "line one\nline two",
                Meta {
                    title: Some("a \"b\" <c>"),
                    context: Some("ctx"),
                },
            )
            .unwrap();
        assert_eq!(
            out,
            "<document title=\"a &quot;b&quot; &lt;c&gt;\" media_type=\"text/plain\">\n<context>\nctx\n</context>\n\
             line one\nline two\n</document>\n"
        );
        assert_eq!(after("x", out.clone()), format!("\n{out}"));
        assert_eq!(after("x\n", out.clone()), out);
    }

    #[test]
    fn media_types_and_data_urls() {
        assert_eq!(Media::of("application/pdf"), Some(Media::Pdf));
        assert_eq!(Media::of("text/plain; charset=utf-8"), Some(Media::Text));
        assert_eq!(Media::of("image/png"), None);
        assert_eq!(
            data_url("data:application/pdf;base64,QQ=="),
            Ok((Media::Pdf, "QQ=="))
        );
        assert_eq!(data_url("https://x/a.pdf"), Err(DocumentError::Scheme));
    }

    #[test]
    fn the_text_budget_is_per_request() {
        let mut budget = Budget::default();
        let big = "x".repeat(MAX_TEXT_BYTES_PER_REQUEST / 2);
        budget.text(&big, Meta::default()).unwrap();
        assert!(matches!(
            budget.text(&big, Meta::default()),
            Err(DocumentError::TextTooLarge(_))
        ));
    }

    #[test]
    fn a_reader_panic_fails_only_its_document() {
        let caught = contained(|| -> Vec<String> { panic!("index out of bounds") });
        assert_eq!(caught, Err(DocumentError::ReaderFault));
        assert!(!DocumentError::ReaderFault.is_unsupported());
        assert_eq!(contained(|| 7), Ok(7));
    }

    /// `contained` needs unwinding panics: a `panic = "abort"` profile would turn a reader panic into a process
    /// abort, every request on the worker with it.
    #[test]
    fn release_profile_unwinds() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for manifest in [root.join("../../Cargo.toml"), root.join("Cargo.toml")] {
            let text = std::fs::read_to_string(&manifest).unwrap();
            let abort = text.lines().any(|line| {
                let line = line.split('#').next().unwrap_or("").replace(' ', "");
                line.starts_with("panic=") && line.contains("abort")
            });
            assert!(
                !abort,
                "{}: panic = \"abort\" defeats document::contained",
                manifest.display()
            );
        }
    }
}
