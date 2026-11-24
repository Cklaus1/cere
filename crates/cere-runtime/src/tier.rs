//! Trust tiers (net-teleport design note 4.1, D-T7): every install and every federated worker is `own`, `team` or
//! `provider` relative to the person who owns the machine, and each person scope lists the tiers its state and work
//! may go to (`allowed_tiers`, default `["own"]`, Q-T5).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Own,
    Team,
    Provider,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Own => "own",
            Tier::Team => "team",
            Tier::Provider => "provider",
        }
    }
}

/// The tier of an install or worker whose config names none.
pub const DEFAULT_TIER: Tier = Tier::Own;

/// The tiers a scope that names none allows (Q-T5).
pub const DEFAULT_ALLOWED_TIERS: &[Tier] = &[Tier::Own];

/// True iff `tier` is in `allowed`.
pub fn allows(allowed: &[Tier], tier: Tier) -> bool {
    allowed.contains(&tier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_names_are_lowercase() {
        for (t, s) in [
            (Tier::Own, "\"own\""),
            (Tier::Team, "\"team\""),
            (Tier::Provider, "\"provider\""),
        ] {
            assert_eq!(serde_json::to_string(&t).unwrap(), s);
            assert_eq!(serde_json::from_str::<Tier>(s).unwrap(), t);
            assert_eq!(format!("\"{}\"", t.as_str()), s);
        }
        assert!(serde_json::from_str::<Tier>("\"public\"").is_err());
    }

    #[test]
    fn default_allows_own_only() {
        assert!(allows(DEFAULT_ALLOWED_TIERS, Tier::Own));
        assert!(!allows(DEFAULT_ALLOWED_TIERS, Tier::Team));
        assert!(!allows(DEFAULT_ALLOWED_TIERS, Tier::Provider));
    }
}
