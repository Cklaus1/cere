//! Roadmap v0.7 A2: the client and workload of every chat request, for its log lines and `GET /metrics`.
//!
//! A request names its client with the optional `X-Cere-Client` header, or the `[[telemetry.clients]]` rules derive
//! it from what clients already send: the auth scope and whether `X-Cere-Priority` is present (frontend's proxy
//! sends it on its `coding-agent` route only, route key `admission_priority`; a batch evaluation client's route sends none).
//! The label set is bounded by the config: a client and workload are always a configured rule's, else
//! [`OTHER`]. The labels are taken once, at acceptance, and kept with the request's ledger record (memory only: a
//! request replayed from the journal after a restart is [`OTHER`]).

use std::sync::Arc;

use crate::config::ClientRule;

/// The client and workload of a request that no rule matched (and of a replayed one).
pub const OTHER: &str = "other";

/// One request's client and workload (configured label values or [`OTHER`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestLabels {
    pub client: Arc<str>,
    pub workload: Arc<str>,
}

impl RequestLabels {
    /// [`OTHER`] for both.
    pub fn other() -> Self {
        let other: Arc<str> = Arc::from(OTHER);
        Self {
            client: other.clone(),
            workload: other,
        }
    }
}

#[derive(Debug)]
struct Rule {
    labels: RequestLabels,
    scope: Option<String>,
    priority_header: Option<bool>,
}

impl Rule {
    fn scope_matches(&self, scope: &str) -> bool {
        self.scope.as_deref().is_none_or(|s| s == scope)
    }
}

/// The `[[telemetry.clients]]` rules, built once at start (a `restart` key, [`crate::reload::KEYS`]).
#[derive(Debug)]
pub struct ClientRules {
    rules: Vec<Rule>,
    other: RequestLabels,
}

impl ClientRules {
    pub fn new(config: &[ClientRule]) -> Self {
        Self {
            rules: config
                .iter()
                .map(|r| Rule {
                    labels: RequestLabels {
                        client: Arc::from(r.client.as_str()),
                        workload: Arc::from(r.workload.as_str()),
                    },
                    scope: r.scope.clone(),
                    priority_header: r.priority_header,
                })
                .collect(),
            other: RequestLabels::other(),
        }
    }

    /// No rules: no request is labelled (the log lines and the exposition are those of a config without them).
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The labels of a request no rule matched.
    pub fn other(&self) -> &RequestLabels {
        &self.other
    }

    /// The labels of a request of `scope`. A request that names a client (`named`, the `X-Cere-Client` value) gets
    /// the first rule of that client whose scope matches. Otherwise (or when none does) it gets the first rule whose
    /// scope and `priority_header` match (`priority_sent`: the request carried `X-Cere-Priority`); else [`OTHER`].
    pub fn resolve(&self, scope: &str, named: Option<&str>, priority_sent: bool) -> RequestLabels {
        let by_name = named.and_then(|name| {
            self.rules
                .iter()
                .find(|r| &*r.labels.client == name && r.scope_matches(scope))
        });
        by_name
            .or_else(|| {
                self.rules.iter().find(|r| {
                    r.scope_matches(scope) && r.priority_header.is_none_or(|p| p == priority_sent)
                })
            })
            .map_or_else(|| self.other.clone(), |r| r.labels.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(
        client: &str,
        workload: &str,
        scope: Option<&str>,
        priority: Option<bool>,
    ) -> ClientRule {
        ClientRule {
            client: client.to_owned(),
            workload: workload.to_owned(),
            scope: scope.map(str::to_owned),
            priority_header: priority,
        }
    }

    /// The production rules (`configs/runtime/prod-t4.toml`).
    fn production() -> ClientRules {
        ClientRules::new(&[
            rule("coding-agent", "interactive", Some("frontend"), Some(true)),
            rule("eval", "batch-eval", Some("frontend"), None),
            rule("ae", "batch", Some("ae"), None),
        ])
    }

    fn pair(l: &RequestLabels) -> (&str, &str) {
        (&l.client, &l.workload)
    }

    #[test]
    fn derives_the_client_from_scope_and_priority_header() {
        let rules = production();
        assert_eq!(
            pair(&rules.resolve("frontend", None, false)),
            ("eval", "batch-eval")
        );
        assert_eq!(
            pair(&rules.resolve("frontend", None, true)),
            ("coding-agent", "interactive")
        );
        assert_eq!(pair(&rules.resolve("ae", None, true)), ("ae", "batch"));
        assert_eq!(pair(&rules.resolve("ae", None, false)), ("ae", "batch"));
    }

    #[test]
    fn a_named_client_wins_within_its_scope() {
        let rules = production();
        // The judge naming itself while sending a priority header: the name decides.
        assert_eq!(
            pair(&rules.resolve("frontend", Some("eval"), true)),
            ("eval", "batch-eval")
        );
        assert_eq!(
            pair(&rules.resolve("frontend", Some("coding-agent"), false)),
            ("coding-agent", "interactive")
        );
        // A name of another scope's rule cannot take that rule: the scope's own rules decide.
        assert_eq!(
            pair(&rules.resolve("ae", Some("eval"), false)),
            ("ae", "batch")
        );
        // An unknown name falls back to the derived client (labels stay bounded).
        assert_eq!(
            pair(&rules.resolve("frontend", Some("x".repeat(500).as_str()), false)),
            ("eval", "batch-eval")
        );
    }

    #[test]
    fn no_rule_is_other() {
        let rules = production();
        assert_eq!(pair(&rules.resolve("lab", None, false)), (OTHER, OTHER));
        assert_eq!(
            pair(&rules.resolve("lab", Some("eval"), false)),
            (OTHER, OTHER)
        );
        let none = ClientRules::new(&[]);
        assert_eq!(
            pair(&none.resolve("frontend", Some("eval"), true)),
            (OTHER, OTHER)
        );
    }

    #[test]
    fn a_rule_without_scope_matches_every_scope() {
        let rules = ClientRules::new(&[
            rule("eval", "batch-eval", Some("frontend"), Some(false)),
            rule("probe", "probe", None, None),
        ]);
        assert_eq!(
            pair(&rules.resolve("frontend", None, true)),
            ("probe", "probe")
        );
        assert_eq!(pair(&rules.resolve("ae", None, false)), ("probe", "probe"));
        assert_eq!(
            pair(&rules.resolve("frontend", None, false)),
            ("eval", "batch-eval")
        );
    }
}
