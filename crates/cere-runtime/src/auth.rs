//! Bearer-token authentication (m1 design note S5, m35 design note Q13). The scope comes from the token
//! alone. The presented token is compared against every configured secret in constant time: both sides are
//! SHA-256 digests, every scope (and the operator token) is checked (no early exit), so timing does not depend on
//! which secret matched or on the token bytes. With `auth.keyless_scope` (g9-prod design note F2) a client
//! request without an `Authorization` header is that scope; a present header is judged as without the setting.

use sha2::{Digest, Sha256};

use crate::config::{ScopeCredential, SecretToken};

pub struct Authenticator {
    scopes: Vec<(String, [u8; 32])>,
    /// The digest of the operator token of the drain routes; `None` disables them.
    operator: Option<[u8; 32]>,
    /// The scope of a client request without an `Authorization` header; `None` → such a request is unauthenticated.
    keyless: Option<String>,
}

/// Who an `Authorization` header names on an operator route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorAccess {
    /// No operator token is configured: the operator routes do not exist (404).
    Disabled,
    /// The operator token.
    Operator,
    /// A client scope token: authenticated, not authorized (403).
    Client,
    /// Absent, malformed or unknown (401).
    Unauthenticated,
}

impl Authenticator {
    pub fn new(
        credentials: &[ScopeCredential],
        operator: Option<&SecretToken>,
        keyless: Option<&str>,
    ) -> Self {
        Self {
            scopes: credentials
                .iter()
                .map(|c| (c.scope_id.clone(), sha256(c.token.expose().as_bytes())))
                .collect(),
            operator: operator.map(|t| sha256(t.expose().as_bytes())),
            keyless: keyless.map(str::to_owned),
        }
    }

    /// The scope of an `Authorization: Bearer <token>` header value; `None` when not a known token, and when absent
    /// without a keyless scope.
    pub fn authenticate(&self, authorization: Option<&[u8]>) -> Option<&str> {
        let Some(authorization) = authorization else {
            return self.keyless.as_deref();
        };
        let token = authorization.strip_prefix(b"Bearer ")?;
        self.scope_of(&sha256(token))
    }

    /// The access an `Authorization` header value grants on an operator route.
    pub fn authorize_operator(&self, authorization: Option<&[u8]>) -> OperatorAccess {
        let Some(operator) = &self.operator else {
            return OperatorAccess::Disabled;
        };
        let Some(token) = authorization.and_then(|a| a.strip_prefix(b"Bearer ")) else {
            return OperatorAccess::Unauthenticated;
        };
        let presented = sha256(token);
        let is_operator = constant_time_eq(&presented, operator);
        let is_client = self.scope_of(&presented).is_some();
        if is_operator {
            OperatorAccess::Operator
        } else if is_client {
            OperatorAccess::Client
        } else {
            OperatorAccess::Unauthenticated
        }
    }

    fn scope_of(&self, presented: &[u8; 32]) -> Option<&str> {
        let mut found = None;
        for (scope, secret) in &self.scopes {
            if constant_time_eq(presented, secret) {
                found = Some(scope.as_str());
            }
        }
        found
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let diff = a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;
    use std::ffi::OsString;

    #[test]
    fn scope_comes_from_the_token_only() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = std::fs::read_to_string(root.join("configs/runtime/sglang-t4-qwen38.toml"))
            .unwrap()
            .replace(
                "[[auth.scopes]]\nscope_id = \"dev\"\ntoken_env = \"CERE_TOKEN_DEV\"",
                "[auth]\noperator_token_env = \"TOK_OP\"\n\n[[auth.scopes]]\nscope_id = \"a\"\ntoken_env = \"TOK_A\"\n\n[[auth.scopes]]\nscope_id = \"b\"\ntoken_env = \"TOK_B\"",
            )
            .replace(
                "allow_cross_scope_prefix_cache = false",
                "allow_cross_scope_prefix_cache = true",
            );
        let config = RuntimeConfig::from_toml_str(&text, &root).unwrap();
        let env = |name: &str| match name {
            "TOK_A" => Some(OsString::from("secret-a")),
            "TOK_B" => Some(OsString::from("secret-b")),
            "TOK_OP" => Some(OsString::from("secret-op")),
            _ => None,
        };
        let creds = config.resolve_auth_with(env).unwrap();
        let operator = config.resolve_operator_token_with(env).unwrap();
        let auth = Authenticator::new(&creds, None, None);
        assert_eq!(auth.authenticate(Some(b"Bearer secret-a")), Some("a"));
        assert_eq!(auth.authenticate(Some(b"Bearer secret-b")), Some("b"));
        for bad in [
            &b"Bearer secret-c"[..],
            b"Bearer secret-",
            b"Bearer ",
            b"secret-a",
            b"bearer secret-a",
            b"Basic secret-a",
        ] {
            assert_eq!(auth.authenticate(Some(bad)), None);
        }
        assert_eq!(auth.authenticate(None), None);
        // Without an operator token every header is Disabled.
        for header in [
            None,
            Some(&b"Bearer secret-op"[..]),
            Some(b"Bearer secret-a"),
        ] {
            assert_eq!(auth.authorize_operator(header), OperatorAccess::Disabled);
        }

        let auth = Authenticator::new(&creds, operator.as_ref(), None);
        let access = |h: &[u8]| auth.authorize_operator(Some(h));
        assert_eq!(access(b"Bearer secret-op"), OperatorAccess::Operator);
        assert_eq!(access(b"Bearer secret-a"), OperatorAccess::Client);
        assert_eq!(access(b"Bearer secret-b"), OperatorAccess::Client);
        for bad in [
            &b"Bearer secret-o"[..],
            b"Bearer ",
            b"secret-op",
            b"bearer secret-op",
        ] {
            assert_eq!(access(bad), OperatorAccess::Unauthenticated);
        }
        assert_eq!(
            auth.authorize_operator(None),
            OperatorAccess::Unauthenticated
        );
        // The operator token is not a client scope.
        assert_eq!(auth.authenticate(Some(b"Bearer secret-op")), None);

        // A keyless scope names a request without a header only: every present header is judged as before, and the
        // operator routes stay closed to an absent header.
        let auth = Authenticator::new(&creds, operator.as_ref(), Some("b"));
        assert_eq!(auth.authenticate(None), Some("b"));
        assert_eq!(auth.authenticate(Some(b"Bearer secret-a")), Some("a"));
        for bad in [
            &b"Bearer secret-c"[..],
            b"Bearer ",
            b"",
            b"secret-a",
            b"Bearer secret-op",
        ] {
            assert_eq!(auth.authenticate(Some(bad)), None);
        }
        assert_eq!(
            auth.authorize_operator(None),
            OperatorAccess::Unauthenticated
        );
        assert_eq!(
            auth.authorize_operator(Some(b"Bearer secret-op")),
            OperatorAccess::Operator
        );
    }
}
