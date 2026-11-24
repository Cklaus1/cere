//! Team pools (net-teleport design note 4.4, C13-C19, D-T7, D-T8): scheduling classes, worker owners and tiers,
//! and owner-only windows. Admission ([`crate::admission`]) applies them at the grant; the node agent
//! ([`crate::agent`]) enforces the windows and the local owner-only switch on its side too.
//!
//! - **Effective tier.** A worker's configured `tier` is what it is for everyone but its `owner`; for the owner it
//!   is `own`. A request may be granted only on a worker whose effective tier its scope's `allowed_tiers` lists.
//! - **Windows.** `"<days> <HH:MM>-<HH:MM> <zone>"`, e.g. `"Mon-Fri 09:00-17:00 Europe/Berlin"`: days are `Mon`..`Sun`,
//!   single days, ranges (`Fri-Mon` wraps) or comma lists; the start is before the end (`24:00` ends at midnight;
//!   an overnight window is two entries). The zone is `UTC` or an IANA name read from the system time-zone
//!   database (`$TZDIR`, else `/usr/share/zoneinfo`; RFC 8536 TZif with its POSIX footer rule), no crate.
//! - **Classes.** `interactive` (default) or `batch` per request (`X-Cere-Class`, else the scope's `class`).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

use crate::admission::Admission;
use crate::app::App;
use crate::config::RuntimeConfig;
use crate::tier::Tier;
use crate::usage::OwnerOnlySpan;

/// The request header naming a request's class.
pub const CLASS_HEADER: &str = "x-cere-class";
/// A request whose scope allows no eligible worker (tier or owner-only window) queued until its deadline.
pub const REASON_NO_ALLOWED_WORKER: &str = "no_allowed_worker";
/// Most `owner_only` windows of one worker.
pub const MAX_WINDOWS: usize = 16;
/// `admission.batch_floor` when absent: batch gets at least 10 % of the grants while both classes wait.
pub const DEFAULT_BATCH_FLOOR: f64 = 0.1;
/// `admission.owner_wait_cap_ms` when absent (Q-T6, Q-T8).
pub const DEFAULT_OWNER_WAIT_CAP_MS: u64 = 120_000;
/// The part of `owner_wait_cap_ms` kept for a capped non-owner work's end: its deadline falls this long before the
/// cap, so its abort, the agent's cleanup and the slot's release (an orphan hold's first poll) fit inside the cap
/// and the owner's queue wait stays within it (Q-T6, bar S6). Below the cap's 1 s minimum.
pub const OWNER_CAP_RELEASE: Duration = Duration::from_millis(500);
/// The agent's local owner-only switch: this file in its `state_dir` exists while the switch is on.
pub const OWNER_ONLY_MARKER: &str = "owner_only";

/// Gives admission every worker's pool policy: a local worker is unowned at this install's `server.tier`, a
/// remote one has its `[[federation.workers]]` owner, tier and windows.
pub fn configure_workers<'a>(
    admission: &Admission,
    config: &RuntimeConfig,
    local: impl IntoIterator<Item = &'a str>,
) {
    for id in local {
        let team = WorkerPool {
            tier: config.server.tier(),
            ..WorkerPool::default()
        };
        admission.set_worker_pool(id, team);
    }
    for w in config.federation.iter().flat_map(|f| &f.workers) {
        admission.set_worker_pool(&w.worker_id, w.pool());
    }
}

/// `[[auth.scopes]] max_sessions_per_scope` overrides by scope (C18).
pub fn session_caps(config: &RuntimeConfig) -> HashMap<String, u64> {
    config
        .auth
        .scopes
        .iter()
        .filter_map(|s| Some((s.scope_id.clone(), s.max_sessions_per_scope?)))
        .collect()
}

/// Whether the config sets a team-pool scheduling key: a worker `owner`, `tier` or `owner_only`, a scope `class`,
/// or `admission.batch_floor` / `owner_wait_cap_ms`. Only then does `/metrics` carry the team-pool families, so
/// the exposition of a config without them stays as it was.
pub fn configured(config: &RuntimeConfig) -> bool {
    config
        .federation
        .iter()
        .flat_map(|f| &f.workers)
        .any(|w| w.owner.is_some() || w.tier.is_some() || w.owner_only.is_some())
        || config.auth.scopes.iter().any(|s| s.class.is_some())
        || config.admission.batch_floor.is_some()
        || config.admission.owner_wait_cap_ms.is_some()
}

/// How often the owner-only span recorder samples admission's worker states: an opt-out span's recorded ends are
/// at most this late.
pub const OWNER_ONLY_SAMPLE: Duration = Duration::from_secs(1);

/// Turns samples of every owned worker's owner-only state ([`Admission::owner_only_states`]) into ended spans
/// for the usage journal (net-teleport design note 4.4 "Usage report": opt-out hours). A span is open from its
/// window or switch start (never before the tracker started, so a restart records no hour twice) and ends at
/// the first sample outside it, or at [`Self::close`].
#[derive(Debug)]
pub struct OwnerOnlyTracker {
    started_ms: u64,
    open: HashMap<Arc<str>, (Arc<str>, u64)>,
}

impl OwnerOnlyTracker {
    pub fn new(started_ms: u64) -> Self {
        Self {
            started_ms,
            open: HashMap::new(),
        }
    }

    /// Takes one sample at `now_ms`; returns the spans it ended. A span whose start moved (one span ended and
    /// another began between samples) ends at `now_ms` and the new one opens.
    pub fn observe(
        &mut self,
        now_ms: u64,
        states: Vec<(Arc<str>, Arc<str>, Option<u64>)>,
    ) -> Vec<OwnerOnlySpan> {
        let mut ended = Vec::new();
        for (worker, owner, since) in states {
            let since = since.map(|s| s.max(self.started_ms));
            match (self.open.get(&worker), since) {
                (Some(&(_, start)), Some(s)) if start == s => {}
                (open, since) => {
                    if let Some((owner, start)) = open.cloned() {
                        ended.push(span(&worker, &owner, start, now_ms));
                    }
                    match since {
                        Some(s) => self.open.insert(worker, (owner, s)),
                        None => self.open.remove(&worker),
                    };
                }
            }
        }
        ended
    }

    /// Ends every open span at `now_ms` (a graceful stop).
    pub fn close(&mut self, now_ms: u64) -> Vec<OwnerOnlySpan> {
        let mut ended: Vec<_> = self
            .open
            .drain()
            .map(|(worker, (owner, start))| span(&worker, &owner, start, now_ms))
            .collect();
        ended.sort_by(|a, b| a.worker.cmp(&b.worker));
        ended
    }
}

fn span(worker: &str, owner: &str, start_ms: u64, end_ms: u64) -> OwnerOnlySpan {
    OwnerOnlySpan {
        worker: worker.to_owned(),
        owner: Some(owner.to_owned()),
        start_ms,
        end_ms: end_ms.max(start_ms),
    }
}

/// Samples `app`'s owner-only states into its usage journal (`closing`: and ends every open span). No-op without
/// a usage journal.
pub(crate) fn record_owner_only_spans(app: &App, closing: bool) {
    let Some(usage) = &app.usage else {
        return;
    };
    let (now_ms, states) = app.admission.owner_only_states();
    let mut tracker = app
        .owner_only_spans
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let mut ended = tracker.observe(now_ms, states);
    if closing {
        ended.extend(tracker.close(now_ms));
    }
    drop(tracker);
    for s in ended {
        usage.record_owner_only(s);
    }
}

/// The sampling task of [`record_owner_only_spans`]; `None` when no worker has an owner or usage is disabled.
pub(crate) fn spawn_span_recorder(app: &Arc<App>) -> Option<JoinHandle<()>> {
    let owned = app
        .config
        .federation
        .iter()
        .flat_map(|f| &f.workers)
        .any(|w| w.owner.is_some());
    if !owned || app.usage.is_none() {
        return None;
    }
    let weak = Arc::downgrade(app);
    Some(tokio::spawn(async move {
        let mut tick = tokio::time::interval(OWNER_ONLY_SAMPLE);
        loop {
            tick.tick().await;
            let Some(app) = weak.upgrade() else {
                return;
            };
            record_owner_only_spans(&app, false);
        }
    }))
}

/// A request's scheduling class (C17). The derived order is the service order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Interactive,
    Batch,
}

impl Class {
    /// `interactive` or `batch`, case-insensitive.
    pub fn parse(value: &str) -> Option<Self> {
        [Self::Interactive, Self::Batch]
            .into_iter()
            .find(|c| value.eq_ignore_ascii_case(c.as_str()))
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Batch => "batch",
        }
    }

    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// A scope's class when it names none.
pub const DEFAULT_CLASS: Class = Class::Interactive;

/// A request's class: `X-Cere-Class` (`header`), else its scope's `class`, else [`DEFAULT_CLASS`] (C17). Admission
/// resolves the same way at the ticket.
pub fn class_of(config: &RuntimeConfig, scope: &str, header: Option<Class>) -> Class {
    header.unwrap_or_else(|| {
        config
            .auth
            .scopes
            .iter()
            .find(|s| s.scope_id == scope)
            .map_or(DEFAULT_CLASS, |s| s.class())
    })
}

/// The `<state_dir>/owner_only` marker of the agent's local switch.
pub fn owner_only_marker(state_dir: &Path) -> PathBuf {
    state_dir.join(OWNER_ONLY_MARKER)
}

/// A person scope id (the `[[auth.scopes]] scope_id` pattern): 1..64 of `[a-z0-9_-]`, starting `[a-z0-9]`.
pub fn is_scope_id(value: &str) -> bool {
    let lower = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    match value.as_bytes().split_first() {
        Some((&head, tail)) => {
            value.len() <= 64 && lower(head) && tail.iter().all(|&c| lower(c) || b"_-".contains(&c))
        }
        None => false,
    }
}

/// The node agent's side of owner-only (C19, F2): with an `owner`, while the local switch
/// ([`owner_only_marker`]) is on or the clock is inside a window, a new SubmitWork of any other scope is refused
/// `owner_only` (the remote readiness smoke's scope excepted, so a worker can still join). The coordinator learns the
/// state from the heartbeat; the agent's refusal holds even if the coordinator's view is stale or wrong.
#[derive(Debug, Clone)]
pub struct AgentOwnerOnly {
    owner: Option<String>,
    windows: Vec<Window>,
    marker: PathBuf,
}

impl AgentOwnerOnly {
    pub fn new(owner: Option<String>, windows: Vec<Window>, state_dir: &Path) -> Self {
        Self {
            owner,
            windows,
            marker: owner_only_marker(state_dir),
        }
    }

    /// Owner-only at `now_ms`: an owner, and the switch on or a window open.
    pub fn active(&self, now_ms: u64) -> bool {
        self.owner.is_some()
            && (self.marker.exists() || self.windows.iter().any(|w| w.since(now_ms).is_some()))
    }

    /// A new SubmitWork of `scope` at `now_ms` must be refused.
    pub fn refuses(&self, scope: &str, now_ms: u64) -> bool {
        self.owner.as_deref() != Some(scope)
            && scope != crate::readiness::REMOTE_SMOKE_SCOPE
            && self.active(now_ms)
    }
}

/// `cere-worker owner-only on|off`: sets the local switch of the agent whose `state_dir` this is. The running
/// agent reads it at every new SubmitWork and reports it in its next heartbeat.
pub fn set_owner_only_switch(state_dir: &Path, on: bool) -> std::io::Result<()> {
    let marker = owner_only_marker(state_dir);
    if on {
        std::fs::write(&marker, b"")
    } else {
        match std::fs::remove_file(&marker) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }
}

/// What admission knows of a worker's pool policy (C13, C15, C19).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerPool {
    /// The person scope owning the worker; `None` for an unowned worker (every local worker).
    pub owner: Option<Arc<str>>,
    /// The tier for everyone but the owner.
    pub tier: Tier,
    /// Owner-only windows (only with an owner).
    pub windows: Vec<Window>,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self {
            owner: None,
            tier: crate::tier::DEFAULT_TIER,
            windows: Vec::new(),
        }
    }
}

impl WorkerPool {
    pub fn is_owner(&self, scope: &str) -> bool {
        self.owner.as_deref() == Some(scope)
    }

    /// The worker's tier for `scope`: `own` for its owner, else the configured tier.
    pub fn tier_for(&self, scope: &str) -> Tier {
        if self.is_owner(scope) {
            Tier::Own
        } else {
            self.tier
        }
    }

    /// The start (unix ms) of the owner-only window `now_ms` lies in, if any.
    pub fn window_since(&self, now_ms: u64) -> Option<u64> {
        self.windows.iter().filter_map(|w| w.since(now_ms)).min()
    }

    /// The first window start or end strictly after `now_ms`.
    pub fn next_window_change(&self, now_ms: u64) -> Option<u64> {
        self.windows
            .iter()
            .filter_map(|w| w.next_change(now_ms))
            .min()
    }
}

/// One owner-only window (C19), kept with its text (the config's serialized form).
#[derive(Clone, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Window {
    text: String,
    /// Bit `d` set: weekday `d` (Mon = 0) is in the window.
    days: u8,
    start_min: u32,
    end_min: u32,
    zone: Arc<Zone>,
}

impl fmt::Debug for Window {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window({:?})", self.text)
    }
}

impl PartialEq for Window {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text
    }
}

impl Eq for Window {}

impl From<Window> for String {
    fn from(w: Window) -> String {
        w.text
    }
}

impl TryFrom<String> for Window {
    type Error = String;

    fn try_from(text: String) -> Result<Self, String> {
        Window::parse(&text)
    }
}

const DAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DAY_S: i64 = 86_400;

impl Window {
    /// Parses `"<days> <HH:MM>-<HH:MM> <zone>"`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let bad = |why: &str| format!("owner_only window {text:?}: {why}");
        let parts: Vec<&str> = text.split(' ').collect();
        let [days, times, zone] = parts[..] else {
            return Err(bad("expected \"<days> <HH:MM>-<HH:MM> <zone>\""));
        };
        let day = |name: &str| DAYS.iter().position(|d| *d == name);
        let mut mask = 0u8;
        for item in days.split(',') {
            let (a, b) = item.split_once('-').unwrap_or((item, item));
            let (Some(a), Some(b)) = (day(a), day(b)) else {
                return Err(bad("days are Mon..Sun, ranges a-b or comma lists"));
            };
            let mut d = a;
            loop {
                mask |= 1 << d;
                if d == b {
                    break;
                }
                d = (d + 1) % 7;
            }
        }
        let (start, end) = times
            .split_once('-')
            .ok_or_else(|| bad("times are HH:MM-HH:MM"))?;
        let minute = |t: &str| -> Option<u32> {
            let (h, m) = t.split_once(':')?;
            if h.len() != 2
                || m.len() != 2
                || !h.bytes().chain(m.bytes()).all(|c| c.is_ascii_digit())
            {
                return None;
            }
            let (h, m): (u32, u32) = (h.parse().ok()?, m.parse().ok()?);
            (m < 60 && (h < 24 || (h == 24 && m == 0))).then_some(h * 60 + m)
        };
        let (Some(start_min), Some(end_min)) = (minute(start), minute(end)) else {
            return Err(bad("times are HH:MM (00:00..24:00)"));
        };
        if start_min >= end_min {
            return Err(bad(
                "the start must be before the end (an overnight window is two entries)",
            ));
        }
        let zone = Zone::load(zone).map_err(|e| bad(&e))?;
        Ok(Self {
            text: text.to_owned(),
            days: mask,
            start_min,
            end_min,
            zone: Arc::new(zone),
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The window's start (unix ms) if `now_ms` lies inside it.
    pub fn since(&self, now_ms: u64) -> Option<u64> {
        let utc = (now_ms / 1000) as i64;
        let local = utc + i64::from(self.zone.offset_at(utc));
        let day = local.div_euclid(DAY_S);
        let minute = (local.rem_euclid(DAY_S) / 60) as u32;
        let inside = self.days & (1 << weekday(day)) != 0
            && (self.start_min..self.end_min).contains(&minute);
        inside.then(|| {
            let start = self
                .zone
                .to_utc(day * DAY_S + i64::from(self.start_min) * 60);
            (start.max(0) as u64 * 1000).min(now_ms)
        })
    }

    /// The first start or end of this window strictly after `now_ms` (within the next eight days).
    pub fn next_change(&self, now_ms: u64) -> Option<u64> {
        let utc = (now_ms / 1000) as i64;
        let today = (utc + i64::from(self.zone.offset_at(utc))).div_euclid(DAY_S);
        (today - 1..today + 8)
            .filter(|&d| self.days & (1 << weekday(d)) != 0)
            .flat_map(|d| {
                [self.start_min, self.end_min]
                    .map(|m| self.zone.to_utc(d * DAY_S + i64::from(m) * 60))
            })
            .filter_map(|s| u64::try_from(s).ok().map(|s| s * 1000))
            .filter(|&ms| ms > now_ms)
            .min()
    }
}

/// Mon = 0 for days since 1970-01-01 (a Thursday).
fn weekday(day: i64) -> u32 {
    (day + 3).rem_euclid(7) as u32
}

/// A time zone: UTC offsets over time from a TZif file and its POSIX footer rule (RFC 8536).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Zone {
    /// `(transition time, UTC offset from then on)`, ascending.
    transitions: Vec<(i64, i32)>,
    /// The offset before the first transition (time type 0).
    initial: i32,
    /// The footer rule, for times after the last transition.
    rule: Option<PosixTz>,
}

impl Zone {
    pub fn utc() -> Self {
        Self {
            transitions: Vec::new(),
            initial: 0,
            rule: None,
        }
    }

    /// `UTC` or an IANA name from `$TZDIR` (else `/usr/share/zoneinfo`).
    pub fn load(name: &str) -> Result<Self, String> {
        if name == "UTC" {
            return Ok(Self::utc());
        }
        let valid = !name.is_empty()
            && name.len() <= 64
            && name.split('/').all(|part| {
                !part.is_empty()
                    && !part.starts_with('.')
                    && part
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'+'))
            });
        if !valid {
            return Err(format!("zone {name:?} is not an IANA zone name"));
        }
        let dir = std::env::var_os("TZDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/share/zoneinfo"));
        let path = dir.join(name);
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("zone {name:?}: cannot read {}: {e}", path.display()))?;
        Self::from_tzif(&bytes)
            .ok_or_else(|| format!("zone {name:?}: {} is not TZif", path.display()))
    }

    /// Parses TZif bytes (version 1 data, or the 64-bit block and footer of version 2+).
    pub fn from_tzif(b: &[u8]) -> Option<Self> {
        let header = |o: usize| -> Option<[usize; 6]> {
            if b.get(o..o + 4)? != b"TZif" {
                return None;
            }
            let mut counts = [0usize; 6];
            for (i, c) in counts.iter_mut().enumerate() {
                let at = o + 20 + 4 * i;
                *c = u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?) as usize;
            }
            Some(counts)
        };
        let [isut, isstd, leap, time, types, chars] = header(0)?;
        let v1 = time * 5 + types * 6 + chars + leap * 8 + isstd + isut;
        let (o, size, counts) = if *b.get(4)? >= b'2' {
            let o = 44 + v1;
            (o + 44, 8, header(o)?)
        } else {
            (44, 4, [isut, isstd, leap, time, types, chars])
        };
        let [isut, isstd, leap, time, types, chars] = counts;
        let times = b.get(o..o + time * size)?;
        let idx = b.get(o + time * size..o + time * (size + 1))?;
        let tt = o + time * (size + 1);
        let offset = |t: usize| -> Option<i32> {
            let at = tt + 6 * t;
            (t < types).then_some(())?;
            Some(i32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
        };
        let initial = offset(0)?;
        let mut transitions = Vec::with_capacity(time);
        for i in 0..time {
            let raw = &times[i * size..(i + 1) * size];
            let at = if size == 8 {
                i64::from_be_bytes(raw.try_into().ok()?)
            } else {
                i64::from(i32::from_be_bytes(raw.try_into().ok()?))
            };
            transitions.push((at, offset(usize::from(idx[i]))?));
        }
        let end = tt + types * 6 + chars + leap * (size + 4) + isstd + isut;
        let rule = if size == 8 {
            let footer = b.get(end..)?;
            let text = std::str::from_utf8(footer.strip_prefix(b"\n")?).ok()?;
            let text = text.split('\n').next()?;
            if text.is_empty() {
                None
            } else {
                Some(PosixTz::parse(text)?)
            }
        } else {
            None
        };
        Some(Self {
            transitions,
            initial,
            rule,
        })
    }

    /// The UTC offset (seconds east) at `utc` (unix seconds).
    pub fn offset_at(&self, utc: i64) -> i32 {
        let n = self.transitions.partition_point(|&(at, _)| at <= utc);
        match (n, &self.rule) {
            (0, Some(rule)) if self.transitions.is_empty() => rule.offset_at(utc),
            (0, _) => self.initial,
            (n, Some(rule)) if n == self.transitions.len() => rule.offset_at(utc),
            (n, _) => self.transitions[n - 1].1,
        }
    }

    /// The UTC instant of local time `local` (seconds); in a gap or overlap one of the two readings.
    fn to_utc(&self, local: i64) -> i64 {
        let guess = local - i64::from(self.offset_at(local));
        local - i64::from(self.offset_at(guess))
    }
}

/// A POSIX TZ rule (`CET-1CEST,M3.5.0,M10.5.0/3`).
#[derive(Debug, Clone, PartialEq, Eq)]
struct PosixTz {
    std: i32,
    dst: Option<Dst>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Dst {
    offset: i32,
    start: (RuleDay, i64),
    end: (RuleDay, i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleDay {
    /// `Mm.w.d`: day `d` (Sun = 0) of week `w` (5 = last) of month `m`.
    Month(u32, u32, u32),
    /// `Jn`: day 1..365, February 29 never counted.
    Julian(u32),
    /// `n`: day 0..365, February 29 counted.
    Day(u32),
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl Cursor<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        let hit = self.peek() == Some(c);
        self.i += usize::from(hit);
        hit
    }

    fn name(&mut self) -> Option<()> {
        let start = self.i;
        if self.eat(b'<') {
            while self.peek()? != b'>' {
                self.i += 1;
            }
            self.i += 1;
            return (self.i - start >= 5).then_some(());
        }
        while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            self.i += 1;
        }
        (self.i - start >= 3).then_some(())
    }

    fn number(&mut self) -> Option<i64> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()?
            .parse()
            .ok()
    }

    /// `[+-]h[:mm[:ss]]` in seconds.
    fn hms(&mut self) -> Option<i64> {
        let sign = if self.eat(b'-') {
            -1
        } else {
            self.eat(b'+');
            1
        };
        let mut secs = self.number()? * 3600;
        if self.eat(b':') {
            secs += self.number()? * 60;
            if self.eat(b':') {
                secs += self.number()?;
            }
        }
        Some(sign * secs)
    }

    fn rule(&mut self) -> Option<(RuleDay, i64)> {
        let day = if self.eat(b'M') {
            let m = self.number()? as u32;
            self.eat(b'.').then_some(())?;
            let w = self.number()? as u32;
            self.eat(b'.').then_some(())?;
            let d = self.number()? as u32;
            ((1..=12).contains(&m) && (1..=5).contains(&w) && d <= 6).then_some(())?;
            RuleDay::Month(m, w, d)
        } else if self.eat(b'J') {
            let n = self.number()? as u32;
            (1..=365).contains(&n).then_some(())?;
            RuleDay::Julian(n)
        } else {
            let n = self.number()? as u32;
            (n <= 365).then_some(())?;
            RuleDay::Day(n)
        };
        let time = if self.eat(b'/') { self.hms()? } else { 7200 };
        Some((day, time))
    }
}

impl PosixTz {
    fn parse(text: &str) -> Option<Self> {
        let mut c = Cursor {
            b: text.as_bytes(),
            i: 0,
        };
        c.name()?;
        let std = i32::try_from(-c.hms()?).ok()?;
        if c.peek().is_none() {
            return Some(Self { std, dst: None });
        }
        c.name()?;
        let offset = match c.peek() {
            Some(b',') => std + 3600,
            _ => i32::try_from(-c.hms()?).ok()?,
        };
        c.eat(b',').then_some(())?;
        let start = c.rule()?;
        c.eat(b',').then_some(())?;
        let end = c.rule()?;
        c.peek().is_none().then_some(())?;
        Some(Self {
            std,
            dst: Some(Dst { offset, start, end }),
        })
    }

    fn offset_at(&self, utc: i64) -> i32 {
        let Some(dst) = &self.dst else {
            return self.std;
        };
        let year = civil_from_days((utc + i64::from(self.std)).div_euclid(DAY_S)).0;
        let start = rule_day(year, dst.start.0) * DAY_S + dst.start.1 - i64::from(self.std);
        let end = rule_day(year, dst.end.0) * DAY_S + dst.end.1 - i64::from(dst.offset);
        let in_dst = if start < end {
            (start..end).contains(&utc)
        } else {
            !(end..start).contains(&utc)
        };
        if in_dst {
            dst.offset
        } else {
            self.std
        }
    }
}

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

fn month_days(y: i64, m: u32) -> i64 {
    match m {
        2 if is_leap(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// Days since the epoch of a rule's day in `year`.
fn rule_day(year: i64, day: RuleDay) -> i64 {
    let jan1 = days_from_civil(year, 1, 1);
    match day {
        RuleDay::Month(m, w, d) => {
            let first = days_from_civil(year, m, 1);
            // Sun = 0: 1970-01-01 was a Thursday (4).
            let wd = (first + 4).rem_euclid(7);
            let mut day = first + (i64::from(d) - wd).rem_euclid(7) + 7 * (i64::from(w) - 1);
            while day >= first + month_days(year, m) {
                day -= 7;
            }
            day
        }
        RuleDay::Julian(n) => jan1 + i64::from(n) - 1 + i64::from(is_leap(year) && n >= 60),
        RuleDay::Day(n) => jan1 + i64::from(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(y: i64, m: u32, d: u32, h: i64, min: i64) -> u64 {
        ((days_from_civil(y, m, d) * DAY_S + h * 3600 + min * 60) * 1000) as u64
    }

    #[test]
    fn civil_round_trip() {
        for z in [-800_000, -1, 0, 1, 19_000, 20_513, 2_000_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(days_from_civil(2026, 11, 23), 20_780);
        assert_eq!(weekday(20_780), 0, "2026-11-23 is a Monday");
    }

    #[test]
    fn posix_rule_berlin() {
        let tz = PosixTz::parse("CET-1CEST,M3.5.0,M10.5.0/3").unwrap();
        let at = |ms: u64| tz.offset_at((ms / 1000) as i64);
        // 2026: DST from Sun 29 March 01:00 UTC to Sun 25 October 01:00 UTC.
        assert_eq!(at(utc(2026, 3, 29, 0, 59)), 3600);
        assert_eq!(at(utc(2026, 3, 29, 1, 0)), 7200);
        assert_eq!(at(utc(2026, 10, 25, 0, 59)), 7200);
        assert_eq!(at(utc(2026, 10, 25, 1, 0)), 3600);
        assert_eq!(at(utc(2026, 11, 23, 12, 0)), 3600);
        // Southern hemisphere: DST across the new year.
        let syd = PosixTz::parse("AEST-10AEDT,M10.1.0,M4.1.0/3").unwrap();
        assert_eq!(
            syd.offset_at((utc(2026, 1, 10, 0, 0) / 1000) as i64),
            39_600
        );
        assert_eq!(
            syd.offset_at((utc(2026, 7, 10, 0, 0) / 1000) as i64),
            36_000
        );
        for fixed in ["UTC0", "<+03>-3", "EST5"] {
            assert!(PosixTz::parse(fixed).unwrap().dst.is_none(), "{fixed}");
        }
        for bad in [
            "",
            "C-1",
            "CET",
            "CET-1CEST",
            "CET-1CEST,M13.1.0,M10.5.0",
            "CET-1x",
        ] {
            assert!(PosixTz::parse(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn system_zone_matches_the_rule() {
        // The system database (present on every build host; the agent and cered read the same).
        let berlin = Zone::load("Europe/Berlin").unwrap();
        assert_eq!(
            berlin.offset_at((utc(2026, 7, 1, 12, 0) / 1000) as i64),
            7200
        );
        assert_eq!(
            berlin.offset_at((utc(2026, 12, 1, 12, 0) / 1000) as i64),
            3600
        );
        assert_eq!(
            berlin.offset_at((utc(1990, 7, 1, 12, 0) / 1000) as i64),
            7200
        );
        assert_eq!(Zone::load("UTC").unwrap(), Zone::utc());
        for bad in [
            "",
            "../etc/passwd",
            "Europe/../x",
            "Nowhere/City",
            ".hidden",
        ] {
            assert!(Zone::load(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn window_parse_and_membership() {
        let w = Window::parse("Mon-Fri 09:00-17:00 Europe/Berlin").unwrap();
        // Mon 23 Nov 2026 (CET, UTC+1): 09:00 local = 08:00 UTC.
        let start = utc(2026, 11, 23, 8, 0);
        assert_eq!(w.since(start - 1000), None);
        assert_eq!(w.since(start), Some(start));
        assert_eq!(w.since(start + 3_600_000), Some(start));
        assert_eq!(
            w.since(utc(2026, 11, 23, 16, 0)),
            None,
            "17:00 local ends it"
        );
        assert_eq!(w.next_change(start - 1000), Some(start));
        assert_eq!(w.next_change(start), Some(utc(2026, 11, 23, 16, 0)));
        // Saturday: none; the next change is Monday's start.
        assert_eq!(w.since(utc(2026, 11, 28, 10, 0)), None);
        assert_eq!(
            w.next_change(utc(2026, 11, 28, 10, 0)),
            Some(utc(2026, 11, 30, 8, 0))
        );
        // Summer time: 09:00 CEST = 07:00 UTC.
        assert!(w.since(utc(2026, 7, 1, 7, 0)).is_some());
        assert!(w.since(utc(2026, 7, 1, 6, 59)).is_none());
        let wrap = Window::parse("Fri-Mon,Wed 00:00-24:00 UTC").unwrap();
        // Bit 0 = Mon: Mon, Wed, Fri, Sat, Sun.
        assert_eq!(wrap.days, 0b111_0101);
        assert_eq!(wrap.since(utc(2026, 11, 24, 12, 0)), None, "Tuesday");
        assert_eq!(
            wrap.since(utc(2026, 11, 25, 23, 59)),
            Some(utc(2026, 11, 25, 0, 0))
        );
        let json = serde_json::to_string(&w).unwrap();
        assert_eq!(json, "\"Mon-Fri 09:00-17:00 Europe/Berlin\"");
        assert_eq!(serde_json::from_str::<Window>(&json).unwrap(), w);
        for bad in [
            "",
            "Mon-Fri 09:00-17:00",
            "mon 09:00-17:00 UTC",
            "Mon 17:00-09:00 UTC",
            "Mon 09:00-09:00 UTC",
            "Mon 9:00-17:00 UTC",
            "Mon 09:00-24:01 UTC",
            "Mon 09:60-10:00 UTC",
            "Mon  09:00-17:00 UTC",
            "Mon 09:00-17:00 Mars/Olympus",
        ] {
            assert!(Window::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn effective_tier_is_own_for_the_owner() {
        let pool = WorkerPool {
            owner: Some(Arc::from("bob")),
            tier: Tier::Team,
            windows: Vec::new(),
        };
        assert_eq!(pool.tier_for("bob"), Tier::Own);
        assert_eq!(pool.tier_for("alice"), Tier::Team);
        assert_eq!(WorkerPool::default().tier_for("x"), Tier::Own);
        assert_eq!(Class::parse("BATCH"), Some(Class::Batch));
        assert_eq!(Class::parse("bulk"), None);
    }
}
