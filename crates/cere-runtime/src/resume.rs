//! Resumable stream buffers (m1close design note Q3-Q5), memory only: a `cered` restart loses them.
//!
//! A stream sent with `X-Cere-On-Disconnect: continue` gets a [`Producer`] when its SSE response opens. The
//! relay stores every event (`id: <seq>` + `data:` bytes) as it creates it and marks the terminal; the HTTP
//! response, and later every `GET /v1/cere/requests/{rid}/events`, is a [`Reader`] that replays the retained
//! events after its cursor and then follows live until `[DONE]`. Nothing waits for a reader, so a stalled or
//! gone client never holds the request back (it is detached, not cancelled).
//!
//! Bounds: each buffer keeps at most `api.resume_buffer_bytes` (the oldest events go first); all buffers
//! together at most `api.resume_buffer_total_bytes` (the oldest terminal buffers go first; with none left, the
//! growing buffer drops its own oldest events); a buffer is released `api.resume_retention_ms` after its
//! terminal. A released buffer leaves a tombstone, so its rid answers `cursor_expired` rather than
//! `not_resumable`, until the ledger has forgotten the request too. Expiry is swept on every store operation.
//!
//! One live reader per request: attaching a reader supersedes the previous one, which ends with the comment
//! `: superseded` (never `[DONE]` or an error event). A reader whose next event was evicted ends without
//! `[DONE]`; reconnecting with its cursor then fails `cursor_expired`.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::body::{Body, Bytes};
use futures_util::stream;
use tokio::sync::watch;
use tokio::time::Instant;

use crate::config::ApiConfig;

pub(crate) const KEEPALIVE: &[u8] = b": keepalive\n\n";
pub(crate) const DONE: &[u8] = b"data: [DONE]\n\n";
const SUPERSEDED: &[u8] = b": superseded\n\n";

/// Why a reader cannot attach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttachError {
    /// No buffer was ever opened for this rid in this process (not a continue-mode stream).
    NotResumable,
    /// The buffer was released, or the events after the cursor were evicted.
    CursorExpired,
}

#[derive(Debug, Clone, Copy)]
struct Limits {
    per_request: u64,
    total: u64,
    retention: Duration,
    /// How long a released buffer's tombstone outlives the release: the ledger's terminal retention.
    tombstone: Duration,
}

pub(crate) struct ResumeStore {
    limits: Limits,
    state: Mutex<StoreState>,
}

#[derive(Default)]
struct StoreState {
    buffers: HashMap<String, Buffer>,
    /// Bytes held by all buffers.
    bytes: u64,
    /// Terminal buffers still holding events, oldest terminal first (expiry and total-cap eviction order).
    terminal: VecDeque<(Instant, String)>,
    /// Released buffers (tombstones), oldest release first.
    released: VecDeque<(Instant, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    Open,
    /// The terminal events are stored; `[DONE]` follows them.
    Done,
    /// The producer went away without a terminal: readers end without `[DONE]`.
    Closed,
}

struct Buffer {
    scope: String,
    events: VecDeque<(u64, Bytes)>,
    bytes: u64,
    /// Every event with a seq at or below this one was evicted.
    evicted_through: Option<u64>,
    end: End,
    released: bool,
    /// The attached reader's generation; older readers are superseded.
    reader: u64,
    changed: watch::Sender<()>,
}

impl Buffer {
    /// Whether the event after `cursor` (or the first event) is gone.
    fn expired_after(&self, cursor: Option<u64>) -> bool {
        let next = cursor.map_or(0, |c| c.saturating_add(1));
        self.released || self.evicted_through.is_some_and(|e| e >= next)
    }

    fn evict_front(&mut self) -> u64 {
        let (seq, bytes) = self
            .events
            .pop_front()
            .expect("evicting from a non-empty buffer");
        self.evicted_through = Some(seq);
        let len = bytes.len() as u64;
        self.bytes -= len;
        len
    }
}

impl ResumeStore {
    pub(crate) fn new(api: &ApiConfig, ledger_retention_ms: u64) -> Self {
        Self {
            limits: Limits {
                per_request: api.resume_buffer_bytes(),
                total: api.resume_buffer_total_bytes(),
                retention: Duration::from_millis(api.resume_retention_ms()),
                tombstone: Duration::from_millis(ledger_retention_ms),
            },
            state: Mutex::new(StoreState::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, StoreState> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.sweep(&self.limits, Instant::now());
        st
    }

    /// Opens the buffer of a continue-mode stream and attaches its first reader (the original response).
    pub(crate) fn open(
        self: &Arc<Self>,
        rid: &str,
        scope: &str,
        keepalive: Duration,
    ) -> (Producer, Reader) {
        let (changed, rx) = watch::channel(());
        let buffer = Buffer {
            scope: scope.to_owned(),
            events: VecDeque::new(),
            bytes: 0,
            evicted_through: None,
            end: End::Open,
            released: false,
            reader: 1,
            changed,
        };
        self.lock().buffers.insert(rid.to_owned(), buffer);
        let producer = Producer {
            store: self.clone(),
            rid: rid.to_owned(),
            ended: false,
        };
        let reader = Reader {
            store: self.clone(),
            rid: rid.to_owned(),
            generation: 1,
            cursor: None,
            rx,
            keepalive,
            finished: false,
        };
        (producer, reader)
    }

    /// Attaches a reader after `cursor` (`None`: from the first event), superseding the current reader.
    pub(crate) fn attach(
        self: &Arc<Self>,
        scope: &str,
        rid: &str,
        cursor: Option<u64>,
        keepalive: Duration,
    ) -> Result<Reader, AttachError> {
        let mut st = self.lock();
        let buffer = st
            .buffers
            .get_mut(rid)
            .filter(|b| b.scope == scope)
            .ok_or(AttachError::NotResumable)?;
        if buffer.expired_after(cursor) {
            return Err(AttachError::CursorExpired);
        }
        buffer.reader += 1;
        let generation = buffer.reader;
        let rx = buffer.changed.subscribe();
        buffer.changed.send_modify(|_| ());
        Ok(Reader {
            store: self.clone(),
            rid: rid.to_owned(),
            generation,
            cursor,
            rx,
            keepalive,
            finished: false,
        })
    }

    /// Whether `rid` is a continue-mode stream this process buffered (live, retained or released).
    pub(crate) fn knows(&self, rid: &str) -> bool {
        self.lock().buffers.contains_key(rid)
    }

    fn push(&self, rid: &str, seq: u64, event: Bytes) {
        let limits = self.limits;
        let mut st = self.lock();
        let st = &mut *st;
        let Some(buffer) = st.buffers.get_mut(rid) else {
            return;
        };
        let len = event.len() as u64;
        buffer.events.push_back((seq, event));
        buffer.bytes += len;
        st.bytes += len;
        while buffer.bytes > limits.per_request && buffer.events.len() > 1 {
            st.bytes -= buffer.evict_front();
        }
        buffer.changed.send_modify(|_| ());
        while st.bytes > limits.total {
            if let Some((_, oldest)) = st.terminal.pop_front() {
                st.release(&oldest, Instant::now());
                continue;
            }
            // Backstop only: config validation requires `resume_buffer_total_bytes >= resume_buffer_bytes x
            // admission.max_backend_inflight`, one live buffer per backend slot, each within `per_request`
            // (bar a single event larger than that), so live buffers alone do not exceed the total. If they
            // ever do, the pushing buffer drops its own oldest events (down to one), and a reader whose next
            // event goes ends without `[DONE]`.
            let buffer = st.buffers.get_mut(rid).expect("buffer of the pushing rid");
            if buffer.events.len() <= 1 {
                break;
            }
            st.bytes -= buffer.evict_front();
            buffer.changed.send_modify(|_| ());
        }
    }

    fn end(&self, rid: &str, end: End) {
        let mut st = self.lock();
        let Some(buffer) = st.buffers.get_mut(rid) else {
            return;
        };
        if buffer.end != End::Open {
            return;
        }
        buffer.end = end;
        buffer.changed.send_modify(|_| ());
        if !buffer.released {
            st.terminal.push_back((Instant::now(), rid.to_owned()));
        }
    }
}

impl StoreState {
    /// Releases buffers past their retention and forgets tombstones the ledger no longer knows either.
    fn sweep(&mut self, limits: &Limits, now: Instant) {
        while let Some((at, rid)) = self.terminal.front() {
            if now.saturating_duration_since(*at) < limits.retention {
                break;
            }
            let rid = rid.clone();
            self.terminal.pop_front();
            self.release(&rid, now);
        }
        while let Some((at, rid)) = self.released.front() {
            if now.saturating_duration_since(*at) < limits.tombstone {
                break;
            }
            let rid = rid.clone();
            self.released.pop_front();
            if self.buffers.get(&rid).is_some_and(|b| b.released) {
                self.buffers.remove(&rid);
            }
        }
    }

    /// Drops a buffer's events and keeps its tombstone.
    fn release(&mut self, rid: &str, now: Instant) {
        let Some(buffer) = self.buffers.get_mut(rid) else {
            return;
        };
        if buffer.released {
            return;
        }
        buffer.released = true;
        self.bytes -= buffer.bytes;
        buffer.bytes = 0;
        buffer.events = VecDeque::new();
        buffer.changed.send_modify(|_| ());
        self.released.push_back((now, rid.to_owned()));
    }
}

/// The relay's handle on a continue-mode buffer. Dropped without [`Producer::finish`], the buffer ends
/// without `[DONE]`.
pub(crate) struct Producer {
    store: Arc<ResumeStore>,
    rid: String,
    ended: bool,
}

impl Producer {
    /// Stores one event (`id:` + `data:` lines) under its seq.
    pub(crate) fn push(&self, seq: u64, event: Bytes) {
        self.store.push(&self.rid, seq, event);
    }

    /// Marks the terminal: readers deliver the stored events, then `[DONE]`.
    pub(crate) fn finish(&mut self) {
        self.ended = true;
        self.store.end(&self.rid, End::Done);
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        if !self.ended {
            self.store.end(&self.rid, End::Closed);
        }
    }
}

/// One client connection reading a buffer from a cursor.
pub(crate) struct Reader {
    store: Arc<ResumeStore>,
    rid: String,
    generation: u64,
    /// The last delivered seq.
    cursor: Option<u64>,
    rx: watch::Receiver<()>,
    keepalive: Duration,
    finished: bool,
}

/// What a reader does next, decided under the store lock.
enum Next {
    Send(Bytes),
    Last(Bytes),
    Close,
    Wait,
}

impl Reader {
    fn poll_buffer(&mut self) -> Next {
        let st = self.store.lock();
        self.rx.mark_unchanged();
        let Some(buffer) = st.buffers.get(&self.rid) else {
            return Next::Close;
        };
        if buffer.reader != self.generation {
            return Next::Last(Bytes::from_static(SUPERSEDED));
        }
        if buffer.expired_after(self.cursor) {
            return Next::Close;
        }
        let next = self.cursor.map_or(0, |c| c.saturating_add(1));
        let index = buffer.events.partition_point(|(seq, _)| *seq < next);
        if let Some((seq, event)) = buffer.events.get(index) {
            self.cursor = Some(*seq);
            return Next::Send(event.clone());
        }
        match buffer.end {
            End::Open => Next::Wait,
            End::Done => Next::Last(Bytes::from_static(DONE)),
            End::Closed => Next::Close,
        }
    }

    /// The next bytes for the client; `None` ends the response.
    async fn next(&mut self) -> Option<Bytes> {
        if self.finished {
            return None;
        }
        loop {
            match self.poll_buffer() {
                Next::Send(event) => return Some(event),
                Next::Last(event) => {
                    self.finished = true;
                    return Some(event);
                }
                Next::Close => return None,
                Next::Wait => {}
            }
            let quiet = Instant::now() + self.keepalive;
            tokio::select! {
                changed = self.rx.changed() => {
                    if changed.is_err() {
                        return None;
                    }
                }
                _ = tokio::time::sleep_until(quiet) => return Some(Bytes::from_static(KEEPALIVE)),
            }
        }
    }

    /// The response body: replay, then live events.
    pub(crate) fn into_body(self) -> Body {
        Body::from_stream(stream::unfold(self, |mut reader| async move {
            let bytes = reader.next().await?;
            Some((Ok::<_, Infallible>(bytes), reader))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(per_request: u64, total: u64, retention_ms: u64) -> Arc<ResumeStore> {
        Arc::new(ResumeStore {
            limits: Limits {
                per_request,
                total,
                retention: Duration::from_millis(retention_ms),
                tombstone: Duration::from_secs(60),
            },
            state: Mutex::new(StoreState::default()),
        })
    }

    /// A 10-byte event.
    fn event(seq: u64) -> Bytes {
        Bytes::from(format!("event{seq:05}"))
    }

    const KEEPALIVE_IDLE: Duration = Duration::from_secs(60);

    async fn drain(mut reader: Reader) -> Vec<Bytes> {
        let mut out = Vec::new();
        while let Some(bytes) = reader.next().await {
            out.push(bytes);
        }
        out
    }

    #[tokio::test]
    async fn per_request_bound_evicts_the_oldest_events() {
        let store = store(30, 1000, 60_000);
        let (mut producer, first) = store.open("r", "s", KEEPALIVE_IDLE);
        drop(first);
        for seq in 0..5 {
            producer.push(seq, event(seq));
        }
        producer.finish();
        // 30 bytes keep seqs 2..=4.
        for cursor in [None, Some(0)] {
            assert_eq!(
                store.attach("s", "r", cursor, KEEPALIVE_IDLE).err(),
                Some(AttachError::CursorExpired)
            );
        }
        let reader = store.attach("s", "r", Some(1), KEEPALIVE_IDLE).unwrap();
        assert_eq!(
            drain(reader).await,
            vec![event(2), event(3), event(4), Bytes::from_static(DONE)]
        );
        assert_eq!(
            store.attach("t", "r", Some(1), KEEPALIVE_IDLE).err(),
            Some(AttachError::NotResumable),
            "another scope never sees the buffer"
        );
    }

    #[tokio::test]
    async fn total_bound_releases_the_oldest_terminal_buffer_first() {
        let store = store(1000, 50, 60_000);
        for rid in ["old", "newer"] {
            let (mut producer, _) = store.open(rid, "s", KEEPALIVE_IDLE);
            producer.push(0, event(0));
            producer.push(1, event(1));
            producer.finish();
        }
        let (producer, _) = store.open("live", "s", KEEPALIVE_IDLE);
        producer.push(0, event(0));
        assert!(store.attach("s", "old", None, KEEPALIVE_IDLE).is_ok());
        // 40 + 20 bytes exceed 50: only the oldest terminal buffer goes.
        producer.push(1, event(1));
        producer.push(2, event(2));
        assert_eq!(
            store.attach("s", "old", None, KEEPALIVE_IDLE).err(),
            Some(AttachError::CursorExpired)
        );
        assert!(store.attach("s", "newer", None, KEEPALIVE_IDLE).is_ok());
        assert!(store.attach("s", "live", None, KEEPALIVE_IDLE).is_ok());
        assert!(store.knows("old"), "a released buffer leaves a tombstone");
    }

    #[tokio::test]
    async fn buffers_are_released_after_retention() {
        let store = store(1000, 1000, 50);
        let (mut producer, _) = store.open("r", "s", KEEPALIVE_IDLE);
        producer.push(0, event(0));
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(
            store.attach("s", "r", None, KEEPALIVE_IDLE).is_ok(),
            "retention runs from the terminal"
        );
        producer.finish();
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(
            store.attach("s", "r", None, KEEPALIVE_IDLE).err(),
            Some(AttachError::CursorExpired)
        );
    }

    #[tokio::test]
    async fn producer_gone_without_terminal_ends_readers_without_done() {
        let store = store(1000, 1000, 60_000);
        let (producer, reader) = store.open("r", "s", KEEPALIVE_IDLE);
        producer.push(0, event(0));
        drop(producer);
        assert_eq!(drain(reader).await, vec![event(0)]);
    }

    #[tokio::test]
    async fn a_new_reader_supersedes_the_live_one() {
        let store = store(1000, 1000, 60_000);
        let (producer, mut first) = store.open("r", "s", KEEPALIVE_IDLE);
        producer.push(0, event(0));
        assert_eq!(first.next().await, Some(event(0)));
        let second = store.attach("s", "r", Some(0), KEEPALIVE_IDLE).unwrap();
        assert_eq!(
            drain(first).await,
            vec![Bytes::from_static(SUPERSEDED)],
            "the old reader ends with the comment only"
        );
        producer.push(1, event(1));
        drop(producer);
        assert_eq!(drain(second).await, vec![event(1)]);
    }
}
