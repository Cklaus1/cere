//! Versioned contract types only. No model execution is implemented here.
pub mod capabilities;
pub mod lifecycle;
pub mod resource_graph;
pub mod state;
pub mod work_unit;

pub mod worker;

/// Deserializes a schema-required nullable key: explicit `null` is `None`, omission is an error.
/// Plain `Option<T>` fields would silently default an omitted key to `None`; a field with
/// `deserialize_with` gets serde's "missing field" error instead.
pub(crate) fn required_nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    serde::Deserialize::deserialize(deserializer)
}
