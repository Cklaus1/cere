//! API-compat surfaces (M2.14 D3/D4): the chat pipeline is shared, and a
//! non-chat surface only renders the client-visible bytes (unary body, error
//! envelope, SSE stream). Surface modules implement [`SurfaceRender`] on their
//! request echo; this module never depends on them.

use std::sync::Arc;

use serde_json::Value;

/// A CERE error as a surface sees it before rendering its envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorView<'a> {
    pub status: u16,
    /// Api error type (`invalid_request_error`, `not_found_error`, ...).
    pub kind: &'a str,
    /// CERE ApiError code.
    pub code: &'a str,
    pub message: &'a str,
    pub param: Option<&'a str>,
}

/// Renders the chat pipeline's results in one non-chat surface's wire format.
pub trait SurfaceRender: Send + Sync {
    /// `"anthropic"` | `"responses"`; also the digest/ledger surface tag.
    fn name(&self) -> &'static str;
    /// Whole unary response body from the chat completion `chat` and request id.
    fn render_unary(&self, chat: &Value, rid: &str) -> Vec<u8>;
    /// Whole error response body.
    fn error_body(&self, e: &ErrorView) -> Vec<u8>;
    /// Fresh per-request stream renderer.
    fn stream(&self) -> Box<dyn SurfaceStream>;
    /// ApiLenient D4: the request fields the conversion dropped (sorted paths; the header and counter report them).
    fn ignored_params(&self) -> &[String] {
        &[]
    }
}

/// Per-request SSE renderer; each call returns whole SSE frames (possibly none).
pub trait SurfaceStream: Send {
    /// The first frames. `prompt_tokens`: the prompt's token count when the backend knows it before generating
    /// (a local native generation), else `None` (it arrives with the final usage).
    fn open(&mut self, prompt_tokens: Option<u64>) -> Vec<u8>;
    /// One chat.completion.chunk object.
    fn chunk(&mut self, chunk: &Value) -> Vec<u8>;
    /// `finish` is the final chunk's choice (finish_reason, matched_stop ...);
    /// `usage` the chat usage object when known.
    fn complete(&mut self, finish: &Value, usage: Option<&Value>) -> Vec<u8>;
    fn fail(&mut self, e: &ErrorView) -> Vec<u8>;
}

#[derive(Clone, Default)]
pub enum Surface {
    #[default]
    Chat,
    Compat(Arc<dyn SurfaceRender>),
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Surface {
    pub fn name(&self) -> &'static str {
        match self {
            Surface::Chat => "chat",
            Surface::Compat(r) => r.name(),
        }
    }

    pub fn is_chat(&self) -> bool {
        matches!(self, Surface::Chat)
    }

    /// Digest/ledger tag: `None` for chat (keeps chat digests and journals unchanged).
    pub fn tag(&self) -> Option<&'static str> {
        match self {
            Surface::Chat => None,
            Surface::Compat(r) => Some(r.name()),
        }
    }

    pub fn renderer(&self) -> Option<&Arc<dyn SurfaceRender>> {
        match self {
            Surface::Chat => None,
            Surface::Compat(r) => Some(r),
        }
    }

    pub fn stream(&self) -> Option<Box<dyn SurfaceStream>> {
        self.renderer().map(|r| r.stream())
    }
}

/// Which compat surface a deployment flag gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatKind {
    Anthropic,
    Responses,
}

impl CompatKind {
    pub fn name(self) -> &'static str {
        match self {
            CompatKind::Anthropic => "anthropic",
            CompatKind::Responses => "responses",
        }
    }

    /// The per-deployment flag for this surface.
    pub fn enabled(self, d: &crate::config::DeploymentConfig) -> bool {
        match self {
            CompatKind::Anthropic => d.api_compat_anthropic,
            CompatKind::Responses => d.api_compat_responses,
        }
    }
}

/// An owned [`ErrorView`]: a converter's refusal, rendered by [`CompatApi::error_body`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedError {
    pub status: u16,
    pub kind: String,
    pub code: String,
    pub message: String,
    pub param: Option<String>,
}

impl OwnedError {
    pub fn view(&self) -> ErrorView<'_> {
        ErrorView {
            status: self.status,
            kind: &self.kind,
            code: &self.code,
            message: &self.message,
            param: self.param.as_deref(),
        }
    }
}

/// What a converter sees of the deployment the request's `model` selected.
pub struct ConvertCtx<'a> {
    /// The deployment's served model name.
    pub served_model: &'a str,
    /// The deployment's chat validation policy (`native` etc.).
    pub policy: &'a crate::validate::ChatPolicy,
    pub deployment: &'a crate::config::DeploymentConfig,
}

/// One compat surface's entry points, used by `server::serve_compat`.
pub struct CompatApi {
    pub kind: CompatKind,
    /// The error envelope before a request echo exists (auth, body size, flag, conversion refusals, and every
    /// error of a request whose renderer is known goes through [`SurfaceRender::error_body`] instead).
    pub error_body: fn(&ErrorView) -> Vec<u8>,
    /// Request bytes -> chat body bytes (raw `tools[].parameters` preserved) and the request's renderer.
    #[allow(clippy::type_complexity)]
    pub convert: fn(&[u8], &ConvertCtx) -> Result<(Vec<u8>, Arc<dyn SurfaceRender>), OwnedError>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Stub;
    impl SurfaceRender for Stub {
        fn name(&self) -> &'static str {
            "anthropic"
        }
        fn render_unary(&self, _: &Value, rid: &str) -> Vec<u8> {
            rid.as_bytes().to_vec()
        }
        fn error_body(&self, e: &ErrorView) -> Vec<u8> {
            e.code.as_bytes().to_vec()
        }
        fn stream(&self) -> Box<dyn SurfaceStream> {
            unimplemented!()
        }
    }

    #[test]
    fn chat_has_no_tag_and_no_stream() {
        let s = Surface::Chat;
        assert_eq!(s.name(), "chat");
        assert_eq!(s.tag(), None);
        assert!(s.is_chat() && s.renderer().is_none() && s.stream().is_none());
    }

    #[test]
    fn compat_surface_tags_with_its_name() {
        let s = Surface::Compat(Arc::new(Stub));
        assert_eq!(s.name(), "anthropic");
        assert_eq!(s.tag(), Some("anthropic"));
        assert!(!s.is_chat());
        assert_eq!(
            s.renderer().unwrap().render_unary(&Value::Null, "r1"),
            b"r1"
        );
        assert_eq!(format!("{s:?}"), "anthropic");
    }
}
