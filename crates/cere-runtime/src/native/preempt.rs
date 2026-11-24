//! U1 preemption bookkeeping (m211 design note D8): which generation holds an executor session, which one is
//! parked in the snapshot arena, and who may call `cere_session_create` next. [`Sched`] holds no handles and
//! makes no host call: the generation threads (`serve.rs`) ask it, act on the driver, and report back.
//!
//! The rules: a request that finds no free slot (`cere_session_create` CAPACITY) names a victim, the running
//! generation with the most committed tokens, at least `preempt_min_tokens` of them and as many since its last
//! preemption, provided the arena can hold its snapshot (size from its position, the kinds this build writes)
//! and fewer than `max_parked_sessions` are parked; the slot the victim frees is reserved for that request.
//! Without a victim the request waits for a slot, FIFO. A parked session resumes, FIFO, before any new grant:
//! a fresh request may not create a session while one is parked; a parked head whose restore found no room
//! retries only after the next release. With `preempt_every` (D13) every generation's tickets end at each
//! `every` tokens committed since its last preemption, and there it asks [`Sched::force`] whether it parks.
//! With `reserve_output_tokens` (sessions design note SE-09) a session is created for its prompt plus that much
//! output; a generation that reaches the end of its session parks itself ([`Sched::grow`]) and resumes in a session
//! of its whole budget, so it waits in the arena instead of holding pages while it waits.
use std::collections::VecDeque;

use super::snapshot::snapshot_bytes;

/// The configured limits (`[deployments.native]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// `preempt_min_tokens`.
    pub min_tokens: u64,
    /// `snapshot_host_bytes`.
    pub arena_bytes: u64,
    /// `max_parked_sessions`.
    pub max_parked: u64,
    /// `preempt_every` (D13 dev knob): force a preemption of the oldest running generation after every this many
    /// committed tokens, at its ticket boundary ([`Sched::force`]); 0 = off.
    pub every: u64,
    /// SE-09 `reserve_output_tokens`: the output tokens a fresh session is created for; 0 = its whole budget.
    pub reserve: u64,
    /// The model has drafts (M2.9): its snapshots carry kind 10.
    pub draft: bool,
}

#[derive(Debug)]
struct Running {
    id: u64,
    prompt_tokens: u64,
    committed: u64,
    /// `committed` when it was last preempted (0 before).
    mark: u64,
    /// A preemption was asked of it and not yet resolved.
    victim: bool,
    /// It is parking itself to grow its session (SE-09).
    growing: bool,
}

#[derive(Debug)]
struct Parked {
    id: u64,
    prompt_tokens: u64,
    committed: u64,
    /// Arena bytes the snapshot takes (256-B aligned).
    bytes: u64,
}

/// What a fresh request does after `cere_session_create` returned CAPACITY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoSlot {
    /// Preempt this generation; the slot it frees is reserved for the request.
    Preempt(u64),
    /// Wait for a slot (no eligible victim, the arena cannot hold one, or a preemption is already under way).
    Wait,
}

#[derive(Debug)]
pub struct Sched {
    policy: Policy,
    running: Vec<Running>,
    parked: VecDeque<Parked>,
    /// Fresh requests not yet holding a session, in arrival order.
    waiting: VecDeque<u64>,
    /// The request a preemption is freeing a slot for.
    reserved: Option<u64>,
    /// Slots freed so far (sessions released, generations parked).
    releases: u64,
    /// `releases` when the parked head's restore last found no room.
    head_blocked_at: Option<u64>,
    preemptions: u64,
    /// Self-parks of a generation that reached the end of its session (SE-09).
    grows: u64,
}

impl Sched {
    pub fn new(policy: Policy) -> Self {
        Sched {
            policy,
            running: Vec::new(),
            parked: VecDeque::new(),
            waiting: VecDeque::new(),
            reserved: None,
            releases: 0,
            head_blocked_at: None,
            preemptions: 0,
            grows: 0,
        }
    }

    /// A fresh request joins the queue for a slot (at `start`, so arrival order is kept).
    pub fn enqueue(&mut self, id: u64) {
        self.waiting.push_back(id);
    }

    /// May the fresh request `id` call `cere_session_create` now? The one a preemption reserved a slot for may;
    /// otherwise only the queue head, and only while nothing is parked, being preempted or growing.
    pub fn may_create(&self, id: u64) -> bool {
        match self.reserved {
            Some(r) => r == id,
            // M211R-03: a forced (`preempt_every`) victim reserves nothing; its slot is the parked session's (as a
            // growing generation's, SE-09).
            None => {
                self.parked.is_empty()
                    && !self.running.iter().any(|r| r.victim || r.growing)
                    && self.waiting.front() == Some(&id)
            }
        }
    }

    /// The fresh request `id` holds a session.
    pub fn created(&mut self, id: u64, prompt_tokens: u64) {
        self.waiting.retain(|w| *w != id);
        if self.reserved == Some(id) {
            self.reserved = None;
        }
        self.running.push(Running {
            id,
            prompt_tokens,
            committed: 0,
            mark: 0,
            victim: false,
            growing: false,
        });
    }

    /// `cere_session_create` returned CAPACITY for the fresh request `id`.
    pub fn no_slot(&mut self, id: u64) -> NoSlot {
        if self.reserved.is_some() {
            return NoSlot::Wait;
        }
        match self.victim() {
            Some(v) => {
                if let Some(r) = self.running.iter_mut().find(|r| r.id == v) {
                    r.victim = true;
                }
                self.reserved = Some(id);
                NoSlot::Preempt(v)
            }
            None => NoSlot::Wait,
        }
    }

    fn arena_free(&self) -> u64 {
        let used: u64 = self.parked.iter().map(|p| p.bytes).sum();
        self.policy.arena_bytes.saturating_sub(used)
    }

    fn victim(&self) -> Option<u64> {
        if self.parked.len() as u64 >= self.policy.max_parked {
            return None;
        }
        let free = self.arena_free();
        let min = self.policy.min_tokens;
        let mut best: Option<&Running> = None;
        for r in &self.running {
            let eligible = !r.victim
                && !r.growing
                && r.committed >= min
                && r.committed - r.mark >= min
                && self
                    .bytes_at(r.prompt_tokens + r.committed)
                    .is_some_and(|b| b <= free);
            if eligible && best.is_none_or(|b| r.committed > b.committed) {
                best = Some(r);
            }
        }
        best.map(|r| r.id)
    }

    /// `n` more tokens committed by the generation `id`, as its emitter receives them: the counts a competing
    /// request's victim is chosen by.
    pub fn committed(&mut self, id: u64, n: u64) {
        if let Some(r) = self.running.iter_mut().find(|r| r.id == id) {
            r.committed += n;
        } else if let Some(p) = self.parked.iter_mut().find(|p| p.id == id) {
            p.committed += n;
        }
    }

    /// `preempt_every` (D13 dev knob).
    pub fn every(&self) -> u64 {
        self.policy.every
    }

    /// D13: the generation `id` stopped at a `preempt_every` boundary with `committed` tokens and nothing in
    /// flight. It becomes a victim (`true`: its own thread preempts it; no slot is reserved, the parked session
    /// resumes first, D8) when it is the oldest running generation, no preemption is under way or reserved, fewer
    /// than `max_parked_sessions` are parked and the arena can hold its snapshot.
    pub fn force(&mut self, id: u64, committed: u64) -> bool {
        if self.policy.every == 0
            || self.reserved.is_some()
            || self.parked.len() as u64 >= self.policy.max_parked
            || self.running.iter().any(|r| r.victim || r.growing)
        {
            return false;
        }
        let Some(oldest) = self.running.iter().min_by_key(|r| r.id) else {
            return false;
        };
        let position = oldest.prompt_tokens + committed;
        if oldest.id != id
            || self
                .bytes_at(position)
                .is_none_or(|b| b > self.arena_free())
        {
            return false;
        }
        if let Some(r) = self.running.iter_mut().find(|r| r.id == id) {
            r.victim = true;
        }
        true
    }

    /// SE-09: the generation `id` reached the end of its session with budget left and nothing in flight. It parks
    /// itself (`true`; no slot is reserved, like a forced victim) when fewer than `max_parked_sessions` are parked
    /// and the arena can hold its snapshot at `position`; a preemption already asked of it is resolved by the same
    /// park. The config bound (`max_parked_sessions` ≥ the in-flight cap, each at the context limit) makes that
    /// always so.
    pub fn grow(&mut self, id: u64, position: u64) -> bool {
        let full = self.parked.len() as u64 >= self.policy.max_parked;
        let fits = self
            .bytes_at(position)
            .is_some_and(|b| b <= self.arena_free());
        let Some(r) = self.running.iter_mut().find(|r| r.id == id) else {
            return false;
        };
        if full || !fits {
            return false;
        }
        r.growing = true;
        true
    }

    /// The snapshot's arena bytes at `position` (kinds this build writes, 256-B aligned).
    fn bytes_at(&self, position: u64) -> Option<u64> {
        snapshot_bytes(position, position, self.policy.draft).map(aligned)
    }

    /// The victim `id` was snapshotted (`bytes` long) and its session released: it waits in the arena.
    pub fn parked(&mut self, id: u64, bytes: u64) {
        let Some(i) = self.running.iter().position(|r| r.id == id) else {
            return;
        };
        let r = self.running.remove(i);
        self.parked.push_back(Parked {
            id,
            prompt_tokens: r.prompt_tokens,
            committed: r.committed,
            bytes: aligned(bytes),
        });
        if r.growing {
            self.grows += 1;
        } else {
            self.preemptions += 1;
        }
        self.releases += 1;
    }

    /// The victim `id` keeps its session (its DECODE ended before the cancel, or the snapshot was refused): it
    /// is not preempted again before `preempt_min_tokens` more tokens, and the reserving request asks again. A
    /// growing generation's refused snapshot only ends its grow (it asks again after a wait, SE-09).
    pub fn kept(&mut self, id: u64) {
        if let Some(r) = self.running.iter_mut().find(|r| r.id == id) {
            r.growing = false;
            if r.victim {
                r.victim = false;
                r.mark = r.committed;
                self.reserved = None;
            }
        }
    }

    /// May the parked `id` try create + restore now? Only the parked head, when no preemption is under way,
    /// and, after its restore found no room, only once another slot was released, or when no session is held
    /// at all (no release would come: it retries on its own poll).
    pub fn may_resume(&self, id: u64) -> bool {
        self.reserved.is_none()
            && self.parked.front().is_some_and(|p| p.id == id)
            && (self.head_blocked_at != Some(self.releases) || self.running.is_empty())
    }

    /// The parked head's restore returned CAPACITY: it stays at the head and retries after the next release.
    pub fn restore_blocked(&mut self, id: u64) {
        if self.parked.front().is_some_and(|p| p.id == id) {
            self.head_blocked_at = Some(self.releases);
        }
    }

    /// The parked `id` holds a restored session again.
    pub fn resumed(&mut self, id: u64) {
        let Some(i) = self.parked.iter().position(|p| p.id == id) else {
            return;
        };
        let Some(p) = self.parked.remove(i) else {
            return;
        };
        self.head_blocked_at = None;
        self.running.push(Running {
            id,
            prompt_tokens: p.prompt_tokens,
            committed: p.committed,
            mark: p.committed,
            victim: false,
            growing: false,
        });
    }

    /// The generation `id` ended (its session, if any, released; its snapshot, if parked, released).
    pub fn ended(&mut self, id: u64) {
        self.waiting.retain(|w| *w != id);
        if let Some(i) = self.running.iter().position(|r| r.id == id) {
            self.running.remove(i);
            self.releases += 1;
        }
        if let Some(i) = self.parked.iter().position(|p| p.id == id) {
            self.parked.remove(i);
            if i == 0 {
                self.head_blocked_at = None;
            }
            self.releases += 1;
        }
        if self.reserved == Some(id) {
            self.reserved = None;
        }
    }

    /// Preemptions done (a victim parked).
    pub fn preemptions(&self) -> u64 {
        self.preemptions
    }

    /// SE-09 self-parks done (a generation grew its session).
    pub fn grows(&self) -> u64 {
        self.grows
    }

    /// SE-09 `reserve_output_tokens` (0 = off).
    pub fn reserve(&self) -> u64 {
        self.policy.reserve
    }

    /// Generations parked in the arena.
    pub fn parked_count(&self) -> usize {
        self.parked.len()
    }
}

/// Arena sub-allocations are 256-B aligned (D2).
fn aligned(bytes: u64) -> u64 {
    bytes.saturating_add(255) / 256 * 256
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The policy fields the cases do not vary: no `preempt_every`, no SE-09 reserve, a model without drafts.
    const BASE: Policy = Policy {
        min_tokens: 4,
        arena_bytes: 0,
        max_parked: 1,
        every: 0,
        reserve: 0,
        draft: false,
    };

    fn sched(arena_bytes: u64) -> Sched {
        Sched::new(Policy {
            arena_bytes,
            ..BASE
        })
    }

    /// D13: `preempt_every` forces a generation at its boundary (below `min_tokens`, no competing request) only when
    /// it is the oldest running one, never while a preemption is under way or the park list is full; a refused
    /// snapshot lets it be forced again at its next boundary.
    #[test]
    fn preempt_every_forces_the_oldest_generation() {
        let mut s = Sched::new(Policy {
            min_tokens: 1000,
            arena_bytes: BIG,
            every: 3,
            ..BASE
        });
        s.enqueue(0);
        s.created(0, 10);
        s.enqueue(1);
        s.created(1, 10);
        assert!(!s.force(1, 3));
        assert!(s.force(0, 3));
        // Under way: neither is forced.
        assert!(!s.force(0, 3) && !s.force(1, 6));
        s.parked(0, 1000);
        assert_eq!(s.preemptions(), 1);
        // The park list is full.
        assert!(!s.force(1, 6));
        s.resumed(0);
        assert!(!s.force(1, 9));
        assert!(s.force(0, 6));
        s.kept(0);
        assert!(s.force(0, 9));
    }

    #[test]
    fn preempt_every_off_never_forces() {
        let mut s = sched(BIG);
        s.enqueue(0);
        s.created(0, 10);
        assert!(!s.force(0, 3));
    }

    /// D13: no forced preemption while a competing request's preemption holds a reservation, or when the arena
    /// cannot hold the snapshot at the generation's position.
    #[test]
    fn a_forced_preemption_needs_no_reservation_and_room() {
        let bytes = aligned(snapshot_bytes(13, 13, false).unwrap());
        let policy = Policy {
            min_tokens: 1,
            arena_bytes: bytes - 1,
            every: 3,
            ..BASE
        };
        let mut s = Sched::new(policy);
        run(&mut s, 0, 3);
        assert!(!s.force(0, 3));
        let mut s = Sched::new(Policy {
            arena_bytes: bytes,
            ..policy
        });
        run(&mut s, 0, 3);
        s.enqueue(1);
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
        assert!(!s.force(0, 3));
        s.kept(0);
        assert!(s.force(0, 3));
    }

    /// M211R-03: while a forced victim is being preempted (no slot reserved), the fresh queue head may not create
    /// into the slot it is about to release; after the park it still waits for the resume (D8).
    #[test]
    fn no_grant_while_a_forced_victim_parks() {
        let mut s = Sched::new(Policy {
            min_tokens: 1000,
            arena_bytes: BIG,
            every: 3,
            ..BASE
        });
        s.enqueue(0);
        s.created(0, 10);
        s.enqueue(1);
        assert!(s.may_create(1));
        assert!(s.force(0, 3));
        assert!(!s.may_create(1));
        s.parked(0, 1000);
        assert!(!s.may_create(1) && s.may_resume(0));
        s.resumed(0);
        assert!(s.may_create(1));
        // A refused snapshot clears the victim: the head may ask again.
        assert!(s.force(0, 3));
        s.kept(0);
        assert!(s.may_create(1));
    }

    const BIG: u64 = 1 << 40;

    /// A running generation `id` with `committed` tokens after a 10-token prompt.
    fn run(s: &mut Sched, id: u64, committed: u64) {
        s.enqueue(id);
        assert!(s.may_create(id));
        s.created(id, 10);
        s.committed(id, committed);
    }

    #[test]
    fn the_parked_session_resumes_before_the_next_grant() {
        let mut s = sched(BIG);
        run(&mut s, 0, 8);
        s.enqueue(1);
        s.enqueue(2);
        assert!(s.may_create(1) && !s.may_create(2));
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
        // The preemption is under way: only the request it frees a slot for may create.
        assert_eq!(s.no_slot(1), NoSlot::Wait);
        assert!(!s.may_create(2));
        s.parked(0, 1000);
        assert_eq!(s.preemptions(), 1);
        assert!(s.may_create(1) && !s.may_create(2) && !s.may_resume(0));
        s.created(1, 10);
        // Parked: no new grant, the parked head resumes first.
        assert!(!s.may_create(2) && s.may_resume(0));
        s.ended(1);
        s.resumed(0);
        assert!(s.may_create(2));
    }

    #[test]
    fn no_eligible_victim_means_wait() {
        let mut s = sched(BIG);
        run(&mut s, 0, 3);
        s.enqueue(1);
        assert_eq!(s.no_slot(1), NoSlot::Wait);
        s.committed(0, 1);
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
        assert_eq!(s.preemptions(), 0);
    }

    #[test]
    fn an_arena_too_small_for_the_victim_means_wait() {
        let bytes = aligned(snapshot_bytes(18, 18, false).unwrap());
        let mut s = sched(bytes - 1);
        run(&mut s, 0, 8);
        s.enqueue(1);
        assert_eq!(s.no_slot(1), NoSlot::Wait);
        let mut s = sched(bytes);
        run(&mut s, 0, 8);
        s.enqueue(1);
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
    }

    #[test]
    fn a_full_park_list_means_wait() {
        let mut s = sched(BIG);
        run(&mut s, 0, 8);
        s.enqueue(1);
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
        s.parked(0, 1000);
        s.created(1, 10);
        s.committed(1, 8);
        s.enqueue(2);
        assert_eq!(s.no_slot(2), NoSlot::Wait);
    }

    #[test]
    fn a_blocked_restore_retries_after_the_next_release_and_grants_nothing_meanwhile() {
        let mut s = sched(BIG);
        run(&mut s, 0, 8);
        s.enqueue(1);
        s.no_slot(1);
        s.parked(0, 1000);
        s.created(1, 10);
        assert!(s.may_resume(0));
        s.restore_blocked(0);
        s.enqueue(2);
        assert!(!s.may_resume(0) && !s.may_create(2));
        // The next release lets the head retry; still no grant while it is parked.
        s.ended(1);
        assert!(s.may_resume(0) && !s.may_create(2));
        // Blocked again with no session held anywhere: no release would come, so it retries on its own.
        s.restore_blocked(0);
        assert!(s.may_resume(0) && !s.may_create(2));
        s.resumed(0);
        assert!(s.may_create(2));
    }

    #[test]
    fn a_kept_victim_waits_min_tokens_before_the_next_preemption() {
        let mut s = sched(BIG);
        run(&mut s, 0, 8);
        s.enqueue(1);
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
        s.kept(0);
        assert_eq!(s.no_slot(1), NoSlot::Wait);
        s.committed(0, 4);
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
    }

    /// SE-09: a growing generation parks without a reservation, counts as a grow (not a preemption), is no victim
    /// while it grows, blocks fresh grants until it resumed, and resumes before the queue head.
    #[test]
    fn a_growing_generation_parks_itself_and_resumes_first() {
        let mut s = sched(BIG);
        run(&mut s, 0, 8);
        s.enqueue(1);
        assert!(s.may_create(1));
        assert!(s.grow(0, 18));
        assert!(!s.may_create(1));
        assert_eq!(s.no_slot(1), NoSlot::Wait);
        s.parked(0, 1000);
        assert_eq!((s.grows(), s.preemptions()), (1, 0));
        assert!(!s.may_create(1) && s.may_resume(0));
        s.resumed(0);
        assert!(s.may_create(1));
    }

    /// SE-09: no grow while the park list is full or the arena cannot hold the snapshot at the position (the
    /// generation waits and asks again); a refused snapshot ends the grow only, so it may ask again at once.
    #[test]
    fn a_grow_needs_a_park_slot_and_room() {
        let bytes = aligned(snapshot_bytes(18, 18, false).unwrap());
        let mut s = sched(bytes - 1);
        run(&mut s, 0, 8);
        assert!(!s.grow(0, 18));
        let mut s = sched(bytes);
        run(&mut s, 0, 8);
        run(&mut s, 1, 8);
        assert!(s.grow(0, 18));
        s.kept(0);
        assert_eq!((s.grows(), s.preemptions()), (0, 0));
        assert!(s.grow(0, 18));
        s.parked(0, bytes);
        assert!(!s.grow(1, 18), "the park list is full");
    }

    /// SE-09: a generation a fresh request asked to preempt that grows instead parks once, for both: the slot stays
    /// reserved for the fresh request.
    #[test]
    fn a_victim_that_grows_parks_once_for_both() {
        let mut s = sched(BIG);
        run(&mut s, 0, 8);
        s.enqueue(1);
        assert_eq!(s.no_slot(1), NoSlot::Preempt(0));
        assert!(s.grow(0, 18));
        s.parked(0, 1000);
        assert_eq!((s.grows(), s.preemptions()), (1, 0));
        assert!(s.may_create(1) && !s.may_resume(0));
        s.created(1, 10);
        assert!(s.may_resume(0));
    }

    /// SE-09: the arena check of a model with drafts includes kind 10.
    #[test]
    fn a_drafting_snapshot_is_sized_with_kind_10() {
        let plain = aligned(snapshot_bytes(18, 18, false).unwrap());
        let drafted = aligned(snapshot_bytes(18, 18, true).unwrap());
        assert!(drafted > plain);
        let mut s = Sched::new(Policy {
            arena_bytes: drafted - 1,
            draft: true,
            ..BASE
        });
        run(&mut s, 0, 8);
        assert!(!s.grow(0, 18));
        let mut s = Sched::new(Policy {
            arena_bytes: drafted,
            draft: true,
            ..BASE
        });
        run(&mut s, 0, 8);
        assert!(s.grow(0, 18));
    }
}
