//! net-teleport design note 4.4 "Usage report" (S7, C20, F22, Q-T9): the durable usage journal.
//!
//! One JSON line per request terminal (the same terminals the `cered request` line reports, startup-recovery
//! failures included) in `<ledger.state_dir>/usage/YYYY-MM-DD.jsonl`, by the UTC day of the terminal. A row holds
//! the person scope, the worker, the worker's owner, tier and the request class, the reported token counts, the
//! queue wait and the wall time, and the terminal state. Never content, and no request id, so nothing maps a row to a
//! prompt. `scripts/live/usage_report.py` reads the files.
//!
//! The serving path only queues a row ([`UsageJournal::record`], a bounded channel, never blocking: a full queue
//! drops the row and says so on stderr). One writer thread appends what is queued as one batch and then
//! `fdatasync`s the day file, so a written row survives a `cered` crash or restart and a power loss loses at most
//! the batch in flight. A graceful stop ([`UsageJournal::close`]) writes and syncs every queued row. Files of days
//! older than `usage.retention_days` (default 90, Q-T9) are deleted at start and at each new day.
//!
//! Owner-only (opt-out) spans have their own row kind ([`OwnerOnlySpan`]); S6 records them, the report sums their
//! hours per worker.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Mutex, PoisonError};
use std::thread::JoinHandle;

use serde::{Deserialize, Serialize};

use crate::config::RuntimeConfig;
use crate::ledger::{Orphan, RequestStatus};
use crate::pool::Class;
use crate::tier::Tier;

/// The journal's directory under `ledger.state_dir`.
pub const USAGE_DIR: &str = "usage";

/// Rows the serving path may queue ahead of the writer before it drops one.
const QUEUE_ROWS: usize = 4096;

/// Rows one write and sync take at most.
const BATCH_ROWS: usize = 1024;

const DAY_MS: u64 = 86_400_000;

/// One request terminal. Built only by [`UsageRow::of_terminal`] and [`UsageRow::of_orphan`]; both leave
/// `worker_owner`, `tier` and `class` to [`UsageRow::attribute`], the one place that sets them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageRow {
    /// When the terminal was logged, unix ms; its UTC day names the file.
    pub ts_ms: u64,
    /// The person scope (`[[auth.scopes]] scope_id`).
    pub scope: String,
    /// The worker the request was last dispatched to; `None` when it never was.
    pub worker: Option<String>,
    /// The person scope that owns `worker` (`[[federation.workers]] owner`); `None` for a local or unowned worker.
    pub worker_owner: Option<String>,
    /// The tier of the machine that served, for this row's scope: this install's `server.tier` for a local
    /// worker, a remote worker's `tier` (`own` for its owner); `None` for a request that was never dispatched.
    pub tier: Option<Tier>,
    /// The request's scheduling class (`interactive` / `batch`); `None` for a request replayed from the journal.
    pub class: Option<String>,
    /// `completed`, `failed` or `cancelled`.
    pub state: String,
    pub reason: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub queue_wait_ms: Option<u64>,
    /// Admission to terminal (the request line's `total_ms`).
    pub wall_ms: Option<u64>,
}

impl UsageRow {
    /// The row of a terminal the `cered request` line reports ([`crate::app::App::log_terminal`]).
    pub fn of_terminal(config: &RuntimeConfig, s: &RequestStatus, ts_ms: u64) -> Self {
        Self {
            ts_ms,
            scope: s.scope.clone(),
            worker: s.worker_id.clone(),
            worker_owner: None,
            tier: None,
            class: None,
            state: s.state.as_str().to_owned(),
            reason: s.reason.clone(),
            prompt_tokens: s.usage.prompt_tokens,
            cached_tokens: s.usage.cached_tokens,
            output_tokens: s.usage.completion_tokens,
            queue_wait_ms: s.timings.queue_wait_ms,
            wall_ms: s.timings.total_ms,
        }
        .attribute(config, s.class)
    }

    /// The row of a request the previous incarnation left in flight, failed `worker_restart` at start: nothing
    /// reported, no timings.
    pub fn of_orphan(config: &RuntimeConfig, o: &Orphan, ts_ms: u64) -> Self {
        Self {
            ts_ms,
            scope: o.scope.clone(),
            worker: o.dispatched.as_ref().map(|(id, _)| id.clone()),
            worker_owner: None,
            tier: None,
            class: None,
            state: "failed".to_owned(),
            reason: Some("worker_restart".to_owned()),
            prompt_tokens: None,
            cached_tokens: None,
            output_tokens: None,
            queue_wait_ms: None,
            wall_ms: None,
        }
        .attribute(config, None)
    }

    /// The one site that sets who owns the serving machine, its tier and the request class. A local worker is
    /// unowned at this install's `server.tier`; a remote one has its `[[federation.workers]]` owner, and its tier
    /// for the row's scope (`own` for its owner, net-teleport design note 4.1); `class` is the request's
    /// (`None` when unknown: a replayed request).
    fn attribute(mut self, config: &RuntimeConfig, class: Option<Class>) -> Self {
        let remote = self.worker.as_deref().map(|id| {
            config
                .federation
                .iter()
                .flat_map(|f| &f.workers)
                .find(|w| w.worker_id == id)
        });
        (self.worker_owner, self.tier) = match remote {
            None => (None, None),
            Some(None) => (None, Some(config.server.tier())),
            Some(Some(w)) => {
                let tier = if w.owner.as_deref() == Some(self.scope.as_str()) {
                    Tier::Own
                } else {
                    w.tier()
                };
                (w.owner.clone(), Some(tier))
            }
        };
        self.class = class.map(|c| c.as_str().to_owned());
        self
    }
}

/// A span in which `worker` served only its owner (an opt-out window or the worker's local switch, S6). Recorded
/// once the span ends, in the file of its end day.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerOnlySpan {
    pub worker: String,
    pub owner: Option<String>,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// One journal line: `{"kind":"request",...}` or `{"kind":"owner_only",...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageRecord {
    Request(UsageRow),
    OwnerOnly(OwnerOnlySpan),
}

impl UsageRecord {
    fn ts_ms(&self) -> u64 {
        match self {
            Self::Request(r) => r.ts_ms,
            Self::OwnerOnly(s) => s.end_ms,
        }
    }
}

/// The journal: a bounded queue and its writer thread.
#[derive(Debug)]
pub struct UsageJournal {
    tx: Mutex<Option<SyncSender<UsageRecord>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
    dropped: AtomicU64,
}

impl UsageJournal {
    /// The journal `config` asks for (`usage.enabled`, default true), its directory created and old days
    /// pruned; `None` when disabled.
    pub fn open(config: &RuntimeConfig) -> Result<Option<Self>, String> {
        if !config.usage_enabled() {
            return Ok(None);
        }
        let dir = config.ledger.state_dir.join(USAGE_DIR);
        Self::start(&dir, config.usage_retention_days(), unix_ms())
            .map(Some)
            .map_err(|e| format!("cannot open usage journal in {}: {e}", dir.display()))
    }

    /// Creates `dir` (0700), prunes the days `retention_days` before `now_ms`'s and starts the writer.
    pub fn start(dir: &Path, retention_days: u64, now_ms: u64) -> std::io::Result<Self> {
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)?;
        let mut writer = Writer {
            dir: dir.to_owned(),
            retention_days,
            file: None,
            pruned_day: None,
        };
        writer.prune(now_ms / DAY_MS);
        let (tx, rx) = mpsc::sync_channel(QUEUE_ROWS);
        let handle = std::thread::Builder::new()
            .name("cered-usage".to_owned())
            .spawn(move || writer.run(rx))?;
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            writer: Mutex::new(Some(handle)),
            dropped: AtomicU64::new(0),
        })
    }

    /// Queues a request row; never blocks.
    pub fn record(&self, row: UsageRow) {
        self.send(UsageRecord::Request(row));
    }

    /// Queues an owner-only span; never blocks.
    pub fn record_owner_only(&self, span: OwnerOnlySpan) {
        self.send(UsageRecord::OwnerOnly(span));
    }

    fn send(&self, record: UsageRecord) {
        let tx = self.tx.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(tx) = tx.as_ref() else {
            return self.drop_row("journal closed");
        };
        match tx.try_send(record) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.drop_row("queue full"),
            Err(TrySendError::Disconnected(_)) => self.drop_row("writer stopped"),
        }
    }

    fn drop_row(&self, why: &str) {
        let n = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
        // The first drop and then each power of two: a stalled disk is visible without flooding stderr.
        if n.is_power_of_two() {
            eprintln!("cered usage: row dropped ({why}) dropped_total={n}");
        }
    }

    /// Rows dropped since start.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Writes and syncs every queued row and stops the writer; later rows are dropped. Idempotent.
    pub fn close(&self) {
        drop(
            self.tx
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
        let handle = self
            .writer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

impl Drop for UsageJournal {
    fn drop(&mut self) {
        self.close();
    }
}

struct Writer {
    dir: PathBuf,
    retention_days: u64,
    /// The open day file and its day number (days since the epoch).
    file: Option<(u64, File)>,
    /// The day the last prune ran for.
    pruned_day: Option<u64>,
}

impl Writer {
    fn run(mut self, rx: Receiver<UsageRecord>) {
        let mut batch = Vec::with_capacity(BATCH_ROWS);
        while let Ok(first) = rx.recv() {
            batch.push(first);
            while batch.len() < BATCH_ROWS {
                match rx.try_recv() {
                    Ok(record) => batch.push(record),
                    Err(_) => break,
                }
            }
            self.write_batch(&batch);
            batch.clear();
        }
    }

    /// Appends the batch, one write per run of rows of the same day, and syncs each file it wrote.
    fn write_batch(&mut self, batch: &[UsageRecord]) {
        let mut buf = Vec::new();
        let mut day = None;
        for record in batch {
            let d = record.ts_ms() / DAY_MS;
            if day != Some(d) {
                if let Some(day) = day {
                    self.append(day, &buf);
                }
                buf.clear();
                day = Some(d);
            }
            if serde_json::to_writer(&mut buf, record).is_ok() {
                buf.push(b'\n');
            }
        }
        if let Some(day) = day {
            self.append(day, &buf);
        }
    }

    fn append(&mut self, day: u64, buf: &[u8]) {
        if buf.is_empty() {
            return;
        }
        let rows = buf.iter().filter(|b| **b == b'\n').count();
        let path = self.dir.join(day_file_name(day));
        if let Err(e) = self.append_synced(day, &path, buf) {
            eprintln!(
                "cered usage: cannot write {}: {e} (rows lost: {rows})",
                path.display()
            );
            self.file = None;
        }
    }

    fn append_synced(&mut self, day: u64, path: &Path, buf: &[u8]) -> std::io::Result<()> {
        if self.file.as_ref().is_none_or(|(d, _)| *d != day) {
            self.file = None;
            self.prune(day);
            self.file = Some((day, open_day_file(path)?));
        }
        let Some((_, file)) = self.file.as_mut() else {
            unreachable!("opened above");
        };
        file.write_all(buf)?;
        file.sync_data()
    }

    /// Deletes the day files of days `retention_days` or more before `today`, once per new day.
    fn prune(&mut self, today: u64) {
        if self.pruned_day.is_some_and(|d| d >= today) {
            return;
        }
        self.pruned_day = Some(today);
        let Some(first_kept) = today.checked_sub(self.retention_days.saturating_sub(1)) else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(day) = name.to_str().and_then(parse_day_file_name) else {
                continue;
            };
            if day < first_kept {
                if let Err(e) = std::fs::remove_file(entry.path()) {
                    eprintln!("cered usage: cannot prune {}: {e}", entry.path().display());
                }
            }
        }
    }
}

/// Opens a day file for append (0600). A file a crash left without its last newline gets one first, so the next
/// row starts on its own line.
fn open_day_file(path: &Path) -> std::io::Result<File> {
    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .mode(0o600)
        .open(path)?;
    let len = file.metadata()?.len();
    if len > 0 {
        let mut last = [0u8; 1];
        file.seek(SeekFrom::Start(len - 1))?;
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            file.write_all(b"\n")?;
        }
    }
    Ok(file)
}

/// `YYYY-MM-DD.jsonl` of the UTC day `day` (days since 1970-01-01).
pub fn day_file_name(day: u64) -> String {
    let (y, m, d) = civil_from_days(day);
    format!("{y:04}-{m:02}-{d:02}.jsonl")
}

/// The day of a `YYYY-MM-DD.jsonl` name; `None` for any other name.
pub fn parse_day_file_name(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".jsonl")?;
    let b = stem.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let num = |s: &str| -> Option<u64> {
        s.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| s.parse().ok())?
    };
    let (y, m, d) = (num(&stem[..4])?, num(&stem[5..7])?, num(&stem[8..])?);
    let day = days_from_civil(y, m, d)?;
    (civil_from_days(day) == (y, m, d)).then_some(day)
}

/// Howard Hinnant's `civil_from_days` for days since 1970-01-01 (no dates before it occur here).
fn civil_from_days(day: u64) -> (u64, u64, u64) {
    let z = day + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + u64::from(m <= 2);
    (y, m, d)
}

/// The inverse of [`civil_from_days`]; `None` before 1970 or for a month or day out of range.
fn days_from_civil(y: u64, m: u64, d: u64) -> Option<u64> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = if m <= 2 { y.checked_sub(1)? } else { y };
    let era = y / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146_097 + doe).checked_sub(719_468)
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_names_round_trip() {
        assert_eq!(day_file_name(0), "1970-01-01.jsonl");
        // 2026-11-23 is day 20_780.
        assert_eq!(day_file_name(20_780), "2026-11-23.jsonl");
        assert_eq!(day_file_name(20_878), "2027-03-01.jsonl");
        for day in [0, 59, 60, 365, 11_016, 20_780, 20_878, 30_000] {
            assert_eq!(parse_day_file_name(&day_file_name(day)), Some(day));
        }
        for bad in [
            "2026-02-30.jsonl",
            "2026-13-01.jsonl",
            "2026-1-01.jsonl",
            "2026-01-01.json",
            "+026-01-01.jsonl",
            "notes.txt",
        ] {
            assert_eq!(parse_day_file_name(bad), None, "{bad}");
        }
    }

    /// S6 hand-off: a row carries the serving worker's owner, its tier for the row's scope and the request class.
    #[test]
    fn rows_carry_worker_owner_tier_and_class() {
        use crate::api_types::{CacheReport, RequestUsage};
        use crate::ledger::{RequestState, Timings};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config =
            RuntimeConfig::load(root.join("configs/runtime/federation-dev.toml")).unwrap();
        config.server.tier = Some(Tier::Team);
        let fed = config.federation.as_mut().unwrap();
        fed.workers[0].owner = Some("alice".to_owned());
        fed.workers[0].tier = Some(Tier::Provider);
        let remote = fed.workers[0].worker_id.clone();
        let status = |scope: &str, worker: Option<&str>, class: Option<Class>| RequestStatus {
            scope: scope.to_owned(),
            rid: "req_1".to_owned(),
            key: None,
            state: RequestState::Completed,
            reason: None,
            attempt: 1,
            epoch: 1,
            stream: false,
            surface: None,
            session_id: None,
            timings: Timings::default(),
            usage: RequestUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(2),
                cached_tokens: None,
                cache_report: CacheReport::Reported,
            },
            worker_id: worker.map(str::to_owned),
            worker_epoch: Some(1),
            labels: None,
            class,
        };
        let row = |s: RequestStatus| UsageRow::of_terminal(&config, &s, 0);
        // A non-owner on the owned remote worker: its owner and configured tier.
        let r = row(status("bob", Some(&remote), Some(Class::Batch)));
        assert_eq!(r.worker_owner.as_deref(), Some("alice"));
        assert_eq!(r.tier, Some(Tier::Provider));
        assert_eq!(r.class.as_deref(), Some("batch"));
        // The owner on its own worker: tier own.
        let r = row(status("alice", Some(&remote), Some(Class::Interactive)));
        assert_eq!(r.worker_owner.as_deref(), Some("alice"));
        assert_eq!(r.tier, Some(Tier::Own));
        assert_eq!(r.class.as_deref(), Some("interactive"));
        // A local worker: unowned, this install's tier.
        let r = row(status("bob", Some("gpu0"), Some(Class::Interactive)));
        assert_eq!(r.worker_owner, None);
        assert_eq!(r.tier, Some(Tier::Team));
        // Never dispatched, class unknown (replayed): all null.
        let r = row(status("bob", None, None));
        assert_eq!((r.worker_owner, r.tier, r.class), (None, None, None));
    }
}
