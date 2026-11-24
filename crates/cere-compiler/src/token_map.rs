//! NEXTN token map conversion (m21 design note Q10): a stored torch zip holding one pickled int list becomes
//! `draft/token_map.u32le` (little-endian u32 ids). Own zip and pickle-grammar code; no CRC (the source sha256 is
//! recorded instead). Error variants are the `token_map` classes of `tests/fixtures/artifact_invalid.json`.

use crate::Limits;
use std::fmt;

/// Transform name recorded in the manifest.
pub const TRANSFORM: &str = "pickle_int_list_to_u32le:v1";
/// Artifact path of the converted map.
pub const OUTPUT_PATH: &str = "draft/token_map.u32le";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenMapError {
    /// No end-of-central-directory record, bad signatures, truncation, out-of-range offsets, or a zip64
    /// end record/locator inconsistent with the classic EOCD.
    ZipStructure,
    /// zip64 sentinels (classic EOCD or central entries), multi-disk, encryption, or a method other than stored.
    ZipUnsupported,
    /// Zero or several `*/data.pkl`, that entry's sizes, limit or local name disagree, or the file is over
    /// [`max_file_bytes`].
    ZipEntry,
    PickleGrammar,
    /// An id ≥ vocab size or negative.
    TokenRange,
    /// Not strictly increasing.
    TokenOrder,
    TokenEmpty,
}

impl TokenMapError {
    pub fn class(self) -> &'static str {
        match self {
            TokenMapError::ZipStructure => "zip_structure",
            TokenMapError::ZipUnsupported => "zip_unsupported",
            TokenMapError::ZipEntry => "zip_entry",
            TokenMapError::PickleGrammar => "pickle_grammar",
            TokenMapError::TokenRange => "token_range",
            TokenMapError::TokenOrder => "token_order",
            TokenMapError::TokenEmpty => "token_empty",
        }
    }
}

impl fmt::Display for TokenMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "token map rejected: {}", self.class())
    }
}

impl std::error::Error for TokenMapError {}

const EOCD_SIG: u32 = 0x0605_4b50;
const ZIP64_LOCATOR_SIG: u32 = 0x0706_4b50;
const ZIP64_RECORD_SIG: u32 = 0x0606_4b50;
const ZIP64_RECORD_LEN: usize = 56;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;
const EOCD_LEN: usize = 22;

fn u16_at(b: &[u8], at: usize) -> Result<u16, TokenMapError> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or(TokenMapError::ZipStructure)
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, TokenMapError> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or(TokenMapError::ZipStructure)
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, TokenMapError> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or(TokenMapError::ZipStructure)
}

/// Where the central directory must end: the EOCD, or PyTorch's zip64 end record when a locator sits right
/// before the EOCD. Record + locator are accepted only when exactly consistent with the classic EOCD
/// (`classic` = disk, cd_disk, entries on disk, entries, cd size, cd offset); any disagreement is `ZipStructure`.
fn zip64_trailer(zip: &[u8], eocd: usize, classic: [u64; 6]) -> Result<usize, TokenMapError> {
    let Some(loc) = eocd.checked_sub(20) else {
        return Ok(eocd);
    };
    if u32_at(zip, loc)? != ZIP64_LOCATOR_SIG {
        return Ok(eocd);
    }
    if u32_at(zip, loc + 4)? != 0 || u32_at(zip, loc + 16)? != 1 {
        return Err(TokenMapError::ZipStructure);
    }
    let rec = u64_at(zip, loc + 8)?;
    if rec.checked_add(ZIP64_RECORD_LEN as u64) != Some(loc as u64) {
        return Err(TokenMapError::ZipStructure);
    }
    let rec = rec as usize;
    if u32_at(zip, rec)? != ZIP64_RECORD_SIG || u64_at(zip, rec + 4)? != 44 {
        return Err(TokenMapError::ZipStructure);
    }
    let fields = [
        u64::from(u32_at(zip, rec + 16)?),
        u64::from(u32_at(zip, rec + 20)?),
        u64_at(zip, rec + 24)?,
        u64_at(zip, rec + 32)?,
        u64_at(zip, rec + 40)?,
        u64_at(zip, rec + 48)?,
    ];
    if fields != classic {
        return Err(TokenMapError::ZipStructure);
    }
    Ok(rec)
}

struct CentralEntry<'a> {
    name: &'a [u8],
    compressed: u32,
    uncompressed: u32,
    local_offset: u32,
}

/// Zip structure and the other entries a token-map file may carry beyond its `data.pkl` limit.
const ZIP_SLACK_BYTES: u64 = 1 << 20;

/// The largest token-map file read whole: `max_token_map_pkl_bytes` plus 1 MiB of zip structure and other entries.
pub fn max_file_bytes(limits: &Limits) -> u64 {
    limits
        .max_token_map_pkl_bytes
        .saturating_add(ZIP_SLACK_BYTES)
}

/// Returns the stored bytes of the single `*/data.pkl` entry.
fn data_pkl<'a>(zip: &'a [u8], limits: &Limits) -> Result<&'a [u8], TokenMapError> {
    if zip.len() as u64 > max_file_bytes(limits) {
        return Err(TokenMapError::ZipEntry);
    }
    if zip.len() < EOCD_LEN {
        return Err(TokenMapError::ZipStructure);
    }
    let search_from = zip.len().saturating_sub(EOCD_LEN + usize::from(u16::MAX));
    let eocd = (search_from..=zip.len() - EOCD_LEN)
        .rev()
        .find(|&at| {
            u32_at(zip, at) == Ok(EOCD_SIG)
                && u16_at(zip, at + 20).is_ok_and(|c| at + EOCD_LEN + usize::from(c) == zip.len())
        })
        .ok_or(TokenMapError::ZipStructure)?;
    let disk = u16_at(zip, eocd + 4)?;
    let cd_disk = u16_at(zip, eocd + 6)?;
    let entries_here = u16_at(zip, eocd + 8)?;
    let entries = u16_at(zip, eocd + 10)?;
    let cd_size = u32_at(zip, eocd + 12)?;
    let cd_offset = u32_at(zip, eocd + 16)?;
    if entries == u16::MAX || cd_size == u32::MAX || cd_offset == u32::MAX {
        return Err(TokenMapError::ZipUnsupported);
    }
    if disk != 0 || cd_disk != 0 || entries_here != entries {
        return Err(TokenMapError::ZipUnsupported);
    }
    let classic = [
        0,
        0,
        u64::from(entries_here),
        u64::from(entries),
        u64::from(cd_size),
        u64::from(cd_offset),
    ];
    let cd_limit = zip64_trailer(zip, eocd, classic)?;
    let cd_start = cd_offset as usize;
    let cd_end = cd_start
        .checked_add(cd_size as usize)
        .filter(|&end| end <= cd_limit)
        .ok_or(TokenMapError::ZipStructure)?;
    let locator = ZIP64_LOCATOR_SIG.to_le_bytes();
    if zip[cd_end..cd_limit].windows(4).any(|w| w == locator) {
        // A zip64 locator anywhere but immediately before the EOCD.
        return Err(TokenMapError::ZipStructure);
    }
    let mut at = cd_start;
    let mut central = Vec::with_capacity(usize::from(entries));
    for _ in 0..entries {
        if at + 46 > cd_end || u32_at(zip, at)? != CENTRAL_SIG {
            return Err(TokenMapError::ZipStructure);
        }
        let flags = u16_at(zip, at + 8)?;
        let method = u16_at(zip, at + 10)?;
        let compressed = u32_at(zip, at + 20)?;
        let uncompressed = u32_at(zip, at + 24)?;
        let name_len = usize::from(u16_at(zip, at + 28)?);
        let extra_len = usize::from(u16_at(zip, at + 30)?);
        let comment_len = usize::from(u16_at(zip, at + 32)?);
        let disk_start = u16_at(zip, at + 34)?;
        let local_offset = u32_at(zip, at + 42)?;
        // The entry's bounds first (`zip_structure`), then what it uses (`zip_unsupported`), as token_map.py.
        let name_end = at + 46 + name_len;
        let next = name_end + extra_len + comment_len;
        if next > cd_end {
            return Err(TokenMapError::ZipStructure);
        }
        if flags & 1 != 0 || method != 0 || disk_start != 0 {
            return Err(TokenMapError::ZipUnsupported);
        }
        if compressed == u32::MAX || uncompressed == u32::MAX || local_offset == u32::MAX {
            return Err(TokenMapError::ZipUnsupported);
        }
        central.push(CentralEntry {
            name: &zip[at + 46..name_end],
            compressed,
            uncompressed,
            local_offset,
        });
        at = next;
    }
    if at != cd_end {
        // Bytes after the last central entry that the central directory size still counts.
        return Err(TokenMapError::ZipStructure);
    }
    let mut matches = central.iter().filter(|e| is_data_pkl(e.name));
    let (Some(entry), None) = (matches.next(), matches.next()) else {
        return Err(TokenMapError::ZipEntry);
    };
    if entry.compressed != entry.uncompressed
        || u64::from(entry.compressed) > limits.max_token_map_pkl_bytes
    {
        return Err(TokenMapError::ZipEntry);
    }
    // Every defect of the selected entry's local record is `zip_entry`.
    let local = entry.local_offset as usize;
    if local + 30 > cd_start || u32_at(zip, local) != Ok(LOCAL_SIG) {
        return Err(TokenMapError::ZipEntry);
    }
    let name_len = usize::from(u16_at(zip, local + 26)?);
    let extra_len = usize::from(u16_at(zip, local + 28)?);
    let name = zip
        .get(local + 30..local + 30 + name_len)
        .ok_or(TokenMapError::ZipEntry)?;
    if name != entry.name {
        return Err(TokenMapError::ZipEntry);
    }
    let start = local + 30 + name_len + extra_len;
    let end = start + entry.compressed as usize;
    if end > cd_start {
        return Err(TokenMapError::ZipEntry);
    }
    Ok(&zip[start..end])
}

/// `<prefix>/data.pkl` with a non-empty, slash-free prefix.
fn is_data_pkl(name: &[u8]) -> bool {
    name.strip_suffix(b"/data.pkl")
        .is_some_and(|prefix| !prefix.is_empty() && !prefix.contains(&b'/'))
}

struct Cursor<'a> {
    p: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], TokenMapError> {
        let s = self
            .p
            .get(self.i..self.i + n)
            .ok_or(TokenMapError::PickleGrammar)?;
        self.i += n;
        Ok(s)
    }

    fn byte(&mut self) -> Result<u8, TokenMapError> {
        Ok(self.take(1)?[0])
    }

    /// The argument of an int opcode (`BININT`, `BININT1`, `BININT2`); `None` for any other opcode.
    fn int_arg(&mut self, op: u8) -> Result<Option<i64>, TokenMapError> {
        Ok(match op {
            b'J' => {
                let b = self.take(4)?;
                Some(i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            }
            b'K' => Some(i64::from(self.byte()?)),
            b'M' => {
                let b = self.take(2)?;
                Some(i64::from(u16::from_le_bytes([b[0], b[1]])))
            }
            _ => None,
        })
    }
}

/// The ids read so far and the first id defect, reported only once the whole grammar has parsed (as token_map.py:
/// empty, then range, then order).
struct Ids {
    ids: Vec<u32>,
    count: usize,
    last: Option<i64>,
    out_of_range: bool,
    unordered: bool,
    vocab_size: u64,
}

impl Ids {
    fn push(&mut self, id: i64) {
        self.count += 1;
        if self.last.is_some_and(|last| last >= id) {
            self.unordered = true;
        }
        self.last = Some(id);
        match u32::try_from(id)
            .ok()
            .filter(|&id| u64::from(id) < self.vocab_size)
        {
            Some(id) => self.ids.push(id),
            None => self.out_of_range = true,
        }
    }

    fn finish(self) -> Result<Vec<u32>, TokenMapError> {
        if self.count == 0 {
            Err(TokenMapError::TokenEmpty)
        } else if self.out_of_range {
            Err(TokenMapError::TokenRange)
        } else if self.unordered {
            Err(TokenMapError::TokenOrder)
        } else {
            Ok(self.ids)
        }
    }
}

/// Parses the pickle grammar of Q10 left to right (STOP must be the last byte), then checks the ids.
fn parse_pickle(p: &[u8], vocab_size: u64) -> Result<Vec<u32>, TokenMapError> {
    const G: TokenMapError = TokenMapError::PickleGrammar;
    let mut c = Cursor { p, i: 0 };
    if c.byte()? != 0x80 || !(2..=5).contains(&c.byte()?) || c.byte()? != b']' {
        return Err(G);
    }
    let mut ids = Ids {
        ids: Vec::new(),
        count: 0,
        last: None,
        out_of_range: false,
        unordered: false,
        vocab_size,
    };
    let mut op = c.byte()?;
    if op == b'q' {
        c.byte()?;
        op = c.byte()?;
    }
    while op != b'.' {
        if op == b'(' {
            loop {
                let next = c.byte()?;
                if next == b'e' {
                    break;
                }
                ids.push(c.int_arg(next)?.ok_or(G)?);
            }
        } else {
            let id = c.int_arg(op)?.ok_or(G)?;
            if c.byte()? != b'a' {
                return Err(G);
            }
            ids.push(id);
        }
        op = c.byte()?;
    }
    if c.i != p.len() {
        return Err(G);
    }
    ids.finish()
}

/// Converts the bytes of a torch token-map zip into ids in `[0, vocab_size)`.
pub fn convert(zip: &[u8], vocab_size: u64, limits: &Limits) -> Result<Vec<u32>, TokenMapError> {
    parse_pickle(data_pkl(zip, limits)?, vocab_size)
}

/// The artifact encoding: little-endian u32 per id.
pub fn to_u32le(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

/// Why `config.json` gives no vocabulary size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabError(pub String);

impl fmt::Display for VocabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config.json: {}", self.0)
    }
}

impl std::error::Error for VocabError {}

/// `text_config.vocab_size` of `config.json`.
pub fn vocab_size_from_config(config_json: &[u8]) -> Result<u64, VocabError> {
    let v = crate::canon_json::parse_strict(config_json).map_err(|e| VocabError(e.to_string()))?;
    v.pointer("/text_config/vocab_size")
        .and_then(serde_json::Value::as_u64)
        .filter(|&n| n >= 1)
        .ok_or_else(|| VocabError("no positive integer text_config.vocab_size".into()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A stored zip (flag bit 3 + 11 on every entry, sizes zero in the local header) with `entries`.
    pub(crate) fn zip(entries: &[(&str, &[u8])], method: u16, flags: u16) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in entries {
            let offset = out.len() as u32;
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&flags.to_le_bytes());
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&[0; 4]); // time, date
            out.extend_from_slice(&[0; 12]); // crc, sizes (in descriptor)
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            out.extend_from_slice(&0x0807_4b50u32.to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&flags.to_le_bytes());
            central.extend_from_slice(&method.to_le_bytes());
            central.extend_from_slice(&[0; 8]); // time, date, crc
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0; 12]); // extra, comment, disk, int attr, ext attr
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    /// `pickle.dumps(ids, protocol=2)` shape: PROTO 2, EMPTY_LIST, BINPUT 0, MARK ints APPENDS.
    pub(crate) fn pickle(ids: &[i64]) -> Vec<u8> {
        let mut p = vec![0x80, 2, b']', b'q', 0, b'('];
        for &id in ids {
            push_int(&mut p, id);
        }
        p.extend_from_slice(b"e.");
        p
    }

    fn push_int(p: &mut Vec<u8>, id: i64) {
        if (0..256).contains(&id) {
            p.extend_from_slice(&[b'K', id as u8]);
        } else if (0..65536).contains(&id) {
            p.push(b'M');
            p.extend_from_slice(&(id as u16).to_le_bytes());
        } else {
            p.push(b'J');
            p.extend_from_slice(&(id as i32).to_le_bytes());
        }
    }

    fn ok_zip(pkl: &[u8]) -> Vec<u8> {
        zip(&[("m/version", b"3\n"), ("m/data.pkl", pkl)], 0, 0x808)
    }

    fn c(zip: &[u8], vocab: u64) -> Result<Vec<u32>, TokenMapError> {
        convert(zip, vocab, &Limits::default())
    }

    #[test]
    fn converts_appends_groups_and_single_append() {
        let ids = c(&ok_zip(&pickle(&[1, 300, 70000])), 100_000).unwrap();
        assert_eq!(ids, vec![1, 300, 70000]);
        let mut single = vec![0x80, 2, b']'];
        single.extend_from_slice(b"K\x05aK\x07a(e.");
        assert_eq!(c(&ok_zip(&single), 10).unwrap(), vec![5, 7]);
        assert_eq!(to_u32le(&[1, 0x0102_0304]), vec![1, 0, 0, 0, 4, 3, 2, 1]);
    }

    #[test]
    fn rejects_pickle_outside_the_grammar() {
        let e = |pkl: &[u8], vocab| c(&ok_zip(pkl), vocab).unwrap_err();
        assert_eq!(e(&pickle(&[1, 100]), 100), TokenMapError::TokenRange);
        assert_eq!(e(&pickle(&[1, -1]), 100), TokenMapError::TokenRange);
        assert_eq!(e(&pickle(&[2, 2]), 100), TokenMapError::TokenOrder);
        assert_eq!(e(&pickle(&[]), 100), TokenMapError::TokenEmpty);
        let mut trailing = pickle(&[1]);
        trailing.push(0);
        assert_eq!(e(&trailing, 100), TokenMapError::PickleGrammar);
        let global = b"\x80\x02cbuiltins\nlist\nq\x00).";
        assert_eq!(e(global, 100), TokenMapError::PickleGrammar);
        let mut proto1 = pickle(&[1]);
        proto1[1] = 1;
        assert_eq!(e(&proto1, 100), TokenMapError::PickleGrammar);
        assert_eq!(e(b"\x80\x02](K\x01.", 100), TokenMapError::PickleGrammar);
        let limits = Limits {
            max_token_map_pkl_bytes: 4,
            ..Limits::default()
        };
        assert_eq!(
            convert(&ok_zip(&pickle(&[1])), 10, &limits),
            Err(TokenMapError::ZipEntry)
        );
    }

    #[test]
    fn rejects_zip_outside_the_contract() {
        let pkl = pickle(&[1]);
        let deflated = zip(&[("m/data.pkl", &pkl)], 8, 0x808);
        assert_eq!(c(&deflated, 10).unwrap_err(), TokenMapError::ZipUnsupported);
        let encrypted = zip(&[("m/data.pkl", &pkl)], 0, 0x809);
        assert_eq!(
            c(&encrypted, 10).unwrap_err(),
            TokenMapError::ZipUnsupported
        );
        let two = zip(&[("a/data.pkl", &pkl), ("b/data.pkl", &pkl)], 0, 0x808);
        assert_eq!(c(&two, 10).unwrap_err(), TokenMapError::ZipEntry);
        let none = zip(&[("a/other.pkl", &pkl)], 0, 0x808);
        assert_eq!(c(&none, 10).unwrap_err(), TokenMapError::ZipEntry);
        let mut z64 = ok_zip(&pkl);
        let eocd = z64.len() - 22;
        z64[eocd + 16..eocd + 20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(c(&z64, 10).unwrap_err(), TokenMapError::ZipUnsupported);
        let mut multi = ok_zip(&pkl);
        let eocd = multi.len() - 22;
        multi[eocd + 4] = 1;
        assert_eq!(c(&multi, 10).unwrap_err(), TokenMapError::ZipUnsupported);
        let mut renamed = ok_zip(&pkl);
        let at = renamed
            .windows(10)
            .position(|w| w == b"m/data.pkl")
            .unwrap();
        renamed[at] = b'x';
        assert_eq!(c(&renamed, 10).unwrap_err(), TokenMapError::ZipEntry);
        assert_eq!(
            c(b"not a zip", 10).unwrap_err(),
            TokenMapError::ZipStructure
        );
    }

    #[test]
    fn rejects_a_file_over_the_pkl_limit_plus_slack_before_parsing() {
        let limits = Limits {
            max_token_map_pkl_bytes: 8,
            ..Limits::default()
        };
        assert_eq!(max_file_bytes(&limits), 8 + (1 << 20));
        assert_eq!(
            convert(&vec![0; 8 + (1 << 20) + 1], 10, &limits),
            Err(TokenMapError::ZipEntry)
        );
        assert_eq!(
            convert(&vec![0; 8 + (1 << 20)], 10, &limits),
            Err(TokenMapError::ZipStructure)
        );
    }

    #[test]
    fn vocab_size_comes_from_text_config() {
        assert_eq!(
            vocab_size_from_config(br#"{"text_config": {"vocab_size": 248320}}"#),
            Ok(248320)
        );
        assert!(vocab_size_from_config(br#"{"vocab_size": 5}"#).is_err());
    }
}
