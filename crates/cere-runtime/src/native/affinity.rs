//! m32n design note D11 (2), D13: prefix affinity of a multi-replica native deployment. The key of a request is
//! SHA-256 over its prefix-cache namespace (the scope's `cache_salt` bytes, or all-zero when scopes share the
//! cache) followed by the little-endian token ids of its first `min(floor64(n - 1), 4,096)` prompt tokens, so two
//! scopes with different salts never share a key. Each card holds the keys of the prefixes it captured in a
//! bounded in-memory LRU; nothing here is journaled or written (no prompt-derived data on disk).

use std::collections::HashMap;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

/// The longest prefix a key covers (D13).
pub const AFFINITY_PREFIX_TOKENS: usize = 4_096;
/// Keys one card remembers; the least recently used goes first.
pub const AFFINITY_CAPACITY: usize = 4_096;

/// A request's affinity key.
pub type Key = [u8; 32];

/// The affinity key of a prompt in `namespace`; `None` when the prompt is shorter than one 64-token page plus
/// the token the PREFILL scores (the cache captures whole pages before the last token only).
pub fn key(namespace: &[u8; 16], ids: &[u32]) -> Option<Key> {
    let pages = ids.len().saturating_sub(1) / 64 * 64;
    let len = pages.min(AFFINITY_PREFIX_TOKENS);
    if len == 0 {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(namespace);
    for id in &ids[..len] {
        hasher.update(id.to_le_bytes());
    }
    Some(hasher.finalize().into())
}

/// One card's keys, bounded LRU.
#[derive(Default)]
pub struct AffinityMap(Mutex<Lru>);

#[derive(Default)]
struct Lru {
    tick: u64,
    keys: HashMap<Key, u64>,
}

impl AffinityMap {
    /// Whether the card holds `key` (a hit refreshes it).
    pub fn holds(&self, key: &Key) -> bool {
        let mut lru = self.lock();
        lru.tick += 1;
        let tick = lru.tick;
        match lru.keys.get_mut(key) {
            Some(used) => {
                *used = tick;
                true
            }
            None => false,
        }
    }

    /// Records that the card captured `key`'s prefix, evicting the least recently used key when full.
    pub fn insert(&self, key: Key) {
        let mut lru = self.lock();
        lru.tick += 1;
        let tick = lru.tick;
        if lru.keys.len() >= AFFINITY_CAPACITY && !lru.keys.contains_key(&key) {
            if let Some(oldest) = lru.keys.iter().min_by_key(|(_, &t)| t).map(|(k, _)| *k) {
                lru.keys.remove(&oldest);
            }
        }
        lru.keys.insert(key, tick);
    }

    /// The number of keys held (operator `cards`).
    pub fn held(&self) -> usize {
        self.lock().keys.len()
    }

    /// Forgets every key (the card's prefix cache was flushed or its executor released).
    pub fn clear(&self) {
        self.lock().keys.clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Lru> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_covers_whole_pages_before_the_last_token() {
        let ns = [0u8; 16];
        assert_eq!(key(&ns, &[7; 64]), None);
        let a = key(&ns, &[7; 65]).unwrap();
        let mut longer = vec![7; 128];
        longer[100] = 9;
        // Tokens past floor64(n - 1) do not change the key.
        assert_eq!(key(&ns, &longer), Some(a));
        let mut first_page = vec![7; 65];
        first_page[3] = 1;
        assert_ne!(key(&ns, &first_page), Some(a));
        let mut far = vec![7u32; 9_000];
        let k = key(&ns, &far).unwrap();
        far[AFFINITY_PREFIX_TOKENS] = 1;
        assert_eq!(key(&ns, &far), Some(k));
    }

    #[test]
    fn two_namespaces_never_share_a_key() {
        let ids = vec![3u32; 200];
        let mut salt = [0u8; 16];
        let a = key(&salt, &ids);
        salt[0] = 1;
        assert_ne!(key(&salt, &ids), a);
    }

    #[test]
    fn the_map_is_bounded_and_evicts_the_least_recently_used() {
        let map = AffinityMap::default();
        let k = |i: u32| {
            let mut key = [0u8; 32];
            key[..4].copy_from_slice(&i.to_le_bytes());
            key
        };
        for i in 0..AFFINITY_CAPACITY as u32 {
            map.insert(k(i));
        }
        assert!(map.holds(&k(0)));
        map.insert(k(u32::MAX));
        assert_eq!(map.held(), AFFINITY_CAPACITY);
        assert!(map.holds(&k(0)));
        assert!(!map.holds(&k(1)));
        assert!(map.holds(&k(u32::MAX)));
    }
}
