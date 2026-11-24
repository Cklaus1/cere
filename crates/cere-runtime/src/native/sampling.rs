//! The native request's sampling (spec m27 D1, D4, D8, D10): the validated request mapped to the
//! `cere_sampling_params` every work item of the request carries, or `None` for the unchanged greedy path.
use crate::api_types::ApiError;
use crate::native::driver::Sampling;
use crate::native::ffi::LogitBias;
use crate::validate::ChatRequest;

/// SGLang's greedy rule (F5, `sampling_params.py:144-147`): a temperature below this is greedy.
pub const GREEDY_TEMPERATURE: f64 = 1e-6;

/// The model's sampling defaults (D10, Q4 (a)): `temperature`/`top_k`/`top_p` of the artifact's
/// `generation_config.json`, read at upload, as SGLang's `sampling_defaults="model"` (F3) reads them. Each fills the
/// field a native request leaves absent; a field the file lacks keeps SGLang's default.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<i64>,
    pub top_p: Option<f64>,
}

/// D1: `None` iff the request is greedy (temperature < 1e-6 or `top_k` 1), every processor is neutral
/// (`frequency_penalty` 0, `presence_penalty` 0, `repetition_penalty` 1, no `logit_bias`) and it is not
/// `constrained` (M2.8: the executor applies a vocab mask in the sampler stage, so a masked item needs `sampling`;
/// M2.12's `min_tokens` does not exist natively). A greedy request with processors or a mask runs
/// greedy-with-processors (temperature 0).
/// D10: absent `temperature`/`top_k`/`top_p` take `defaults` (the model's), then SGLang's (temperature 1, `top_k`
/// all, `top_p` 1); `min_p` 0. D8 domain mappings: `top_k` -1 or above the vocabulary is the vocabulary. `seed` is
/// the request's seed or the per-request random one (D4). A `logit_bias` id at or above `vocab` is 400
/// `invalid_value` (F7: SGLang checks keys < vocab), and so is a validated positive `top_p` or
/// `repetition_penalty` that rounds to 0 in the executor's f32 (its range is > 0).
pub fn sampling(
    request: &ChatRequest,
    defaults: &GenerationDefaults,
    vocab: u32,
    seed: u64,
    constrained: bool,
) -> Result<Option<Sampling>, ApiError> {
    let temperature = request.temperature.or(defaults.temperature).unwrap_or(1.0);
    let top_k = request.top_k.or(defaults.top_k);
    let greedy = temperature < GREEDY_TEMPERATURE || top_k == Some(1);
    let frequency_penalty = request.frequency_penalty.unwrap_or(0.0);
    let presence_penalty = request.presence_penalty.unwrap_or(0.0);
    let repetition_penalty = request.repetition_penalty.unwrap_or(1.0);
    let logit_bias = logit_bias(request, vocab)?;
    let neutral = frequency_penalty == 0.0
        && presence_penalty == 0.0
        && repetition_penalty == 1.0
        && logit_bias.is_empty();
    if greedy && neutral && !constrained {
        return Ok(None);
    }
    let top_k = match top_k {
        Some(k) if k >= 1 && k <= i64::from(vocab) => k as u32,
        _ => vocab,
    };
    let top_p = positive_f32(request.top_p.or(defaults.top_p).unwrap_or(1.0), "top_p")?;
    let repetition_penalty = positive_f32(repetition_penalty, "repetition_penalty")?;
    Ok(Some(Sampling {
        temperature: if greedy { 0.0 } else { temperature as f32 },
        top_p,
        min_p: request.min_p.unwrap_or(0.0) as f32,
        top_k,
        frequency_penalty: frequency_penalty as f32,
        presence_penalty: presence_penalty as f32,
        repetition_penalty,
        seed,
        logit_bias,
    }))
}

/// A validated positive value as the executor's f32, which must stay positive.
fn positive_f32(value: f64, param: &'static str) -> Result<f32, ApiError> {
    let rounded = value as f32;
    if rounded > 0.0 {
        Ok(rounded)
    } else {
        Err(ApiError::invalid_value(
            param,
            format!("{param} {value} is below the smallest positive 32-bit float"),
        ))
    }
}

/// The validated `logit_bias` (canonical u32 keys, numbers in [-100, 100]) as entries, each id below `vocab`.
fn logit_bias(request: &ChatRequest, vocab: u32) -> Result<Vec<LogitBias>, ApiError> {
    let Some(bias) = &request.logit_bias else {
        return Ok(Vec::new());
    };
    bias.iter()
        .map(|(key, value)| {
            let id = key.parse::<u32>().ok().filter(|id| *id < vocab).ok_or_else(|| {
                ApiError::invalid_value(
                    "logit_bias",
                    format!("logit_bias key '{key}' is not a token id below the vocabulary size {vocab}"),
                )
            })?;
            let bias = value.as_f64().ok_or_else(|| {
                ApiError::invalid_value("logit_bias", format!("logit_bias['{key}'] is not a number"))
            })?;
            Ok(LogitBias {
                id,
                bias: bias as f32,
            })
        })
        .collect()
}

/// `murmur_hash32` (SGLang `kernels/ops/sampling/murmur_hash.py:20-48,92-101`, the executor's
/// `host/sampler_ref.cpp`): MurmurHash3_x86_32 with seed 0 over the 16 bytes seed_lo, seed_hi, `key`, `c`.
/// D4 / ledger M8: `h_c = murmur_hash32(seed, key, c)` for counter channel `c`.
pub fn murmur_hash32(seed: u64, key: u32, c: u32) -> u32 {
    fn mix(h: u32, k: u32) -> u32 {
        let k = k
            .wrapping_mul(0xCC9E_2D51)
            .rotate_left(15)
            .wrapping_mul(0x1B87_3593);
        (h ^ k)
            .rotate_left(13)
            .wrapping_mul(5)
            .wrapping_add(0xE654_6B64)
    }
    let mut h = mix(0, seed as u32);
    h = mix(h, (seed >> 32) as u32);
    h = mix(h, key);
    h = mix(h, c);
    h ^= 16;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^ (h >> 16)
}

/// The seed of sampled choice `i` >= 1 (D4, ledger M8 channels (2, 3), key = `i`): `h_2·2^32 + h_3`. Choice 0
/// keeps the request's `seed`; sampled `n` itself is M2.6 (m26 D11 cites this).
pub fn choice_seed(seed: u64, i: u32) -> u64 {
    (u64::from(murmur_hash32(seed, i, 2)) << 32) | u64::from(murmur_hash32(seed, i, 3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// C7 (D4, ledger M8): the port equals SGLang's `murmur_hash32` on the C3 fixture's known answers
    /// (`[seed, key, c, hash]`, seed as a decimal string), and a choice seed is `h_2·2^32 + h_3` with key = i.
    #[test]
    fn murmur_hash32_and_choice_seed_match_the_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/native/sampler_cases.json");
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let cases = fixture["murmur"].as_array().unwrap();
        assert!(cases.len() >= 1000, "{}", cases.len());
        let mut hashes = std::collections::HashMap::new();
        for case in cases {
            let seed: u64 = case[0].as_str().unwrap().parse().unwrap();
            let key = case[1].as_u64().unwrap() as u32;
            let c = case[2].as_u64().unwrap() as u32;
            let hash = case[3].as_u64().unwrap() as u32;
            assert_eq!(murmur_hash32(seed, key, c), hash, "{case}");
            hashes.insert((seed, key, c), hash);
        }
        let mut checked = 0;
        for (&(seed, key, c), &h2) in &hashes {
            if c != 2 {
                continue;
            }
            let h3 = hashes[&(seed, key, 3)];
            assert_eq!(
                choice_seed(seed, key),
                (u64::from(h2) << 32) | u64::from(h3)
            );
            checked += 1;
        }
        assert!(checked > 0);
    }
}
