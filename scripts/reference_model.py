"""Small in-memory CPU oracles. Not durable storage, a GPU allocator or an API server."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from cere_contracts import ContractError

from contract_primitives import uint as _uint, identifier, Ticket, add

@dataclass
class Reservation:
    bytes: int
    epoch: int
    ticket: Ticket
    cancel_pending: bool = False

class MemoryLedger:
    """Single-threaded atomic accounting oracle; callers supply measured peak bytes."""
    def __init__(self, capacity: int, reserve: int = 0):
        _uint(capacity, "capacity", True); _uint(reserve, "reserve")
        if reserve > capacity: raise ContractError("reserve exceeds capacity")
        self.capacity, self.reserve = capacity, reserve
        self.entries: dict[str, Reservation] = {}
        self.sequence = 0
    @property
    def used(self) -> int:
        return sum(r.bytes for r in self.entries.values())
    def admit(self, key: str, peak_bytes: int, epoch: int) -> Ticket:
        _uint(peak_bytes, "peak_bytes"); _uint(epoch, "epoch", True)
        identifier(key, "reservation id")
        if key in self.entries: raise ContractError("duplicate active reservation")
        if self.used + peak_bytes > self.capacity - self.reserve:
            raise ContractError("insufficient reserved capacity")
        self.sequence = add(self.sequence, 1)
        ticket = Ticket("local-worker", epoch, 1, self.sequence, "memory")
        self.entries[key] = Reservation(peak_bytes, epoch, ticket)
        return ticket
    def _get(self, key: str, ticket: Ticket) -> Reservation:
        r = self.entries.get(key)
        if r is None or not isinstance(ticket, Ticket) or r.ticket != ticket:
            raise ContractError("unknown/stale reservation ticket")
        return r
    def request_cancel(self, key: str, ticket: Ticket) -> None:
        self._get(key, ticket).cancel_pending = True  # memory remains leased
    def acknowledge_completion(self, key: str, ticket: Ticket) -> None:
        self._get(key, ticket)
        del self.entries[key]  # represents an actual safe completion fence

@dataclass(frozen=True)
class Prefix:
    parent: Optional['Prefix']
    delta: tuple[int, ...]

@dataclass
class Session:
    scope: str
    revision: int
    prefix: Prefix
    fingerprint: str
    owner: Optional[str] = None
    owner_generation: int = 0  # 0 = no owner; every owner change bumps by 1

class SessionStore:
    """Copy-on-write token metadata only. No KV/recurrent/GPU-state implementation."""
    def __init__(self): self.sessions: dict[tuple[str,str], Session] = {}
    @staticmethod
    def _tokens(tokens) -> tuple[int,...]:
        result = tuple(tokens)
        for t in result:
            _uint(t, "token")
            if t >= 1 << 32: raise ContractError("token exceeds executor uint32 boundary")
        return result
    def create(self, scope: str, sid: str, tokens, fingerprint: str) -> None:
        identifier(scope, "scope"); identifier(sid, "session id"); identifier(fingerprint, "fingerprint label")
        key = scope, sid
        if key in self.sessions: raise ContractError("session exists")
        self.sessions[key] = Session(scope, 0, Prefix(None, self._tokens(tokens)), fingerprint)
    def get(self, scope: str, sid: str) -> Session:
        identifier(scope, "scope"); identifier(sid, "session id")
        try: return self.sessions[(scope,sid)]
        except KeyError as e: raise ContractError("unknown scoped session") from e
    def fork(self, scope: str, parent: str, child: str, expected_revision: int) -> None:
        _uint(expected_revision, "expected revision"); identifier(child, "child session id")
        p = self.get(scope,parent)
        if p.revision != expected_revision: raise ContractError("stale session revision")
        if (scope,child) in self.sessions: raise ContractError("session exists")
        self.sessions[(scope,child)] = Session(scope, 0, p.prefix, p.fingerprint, p.owner, int(p.owner is not None))
    def pin(self, scope: str, sid: str, worker: str) -> int:
        identifier(worker, "worker id"); s = self.get(scope,sid)
        if s.owner != worker: s.owner, s.owner_generation = worker, add(s.owner_generation, 1)
        return s.owner_generation
    def migrate(self, scope: str, sid: str, to_worker: str, expected_generation: int) -> int:
        _uint(expected_generation, "expected owner generation"); identifier(to_worker, "worker id")
        s = self.get(scope,sid)
        if s.owner_generation != expected_generation: raise ContractError("owner generation conflict")
        if s.owner == to_worker: raise ContractError("already owner")
        return self.pin(scope, sid, to_worker)
    def append(self, scope: str, sid: str, expected_revision: int, tokens, fingerprint: str, held_generation: Optional[int] = None) -> int:
        _uint(expected_revision, "expected revision")
        s = self.get(scope,sid)
        if held_generation is not None and held_generation != s.owner_generation: raise ContractError("session owner superseded")
        if s.revision != expected_revision: raise ContractError("conflicting writer")
        if s.fingerprint != fingerprint: raise ContractError("incompatible input/execution state")
        delta = self._tokens(tokens)
        revision = add(s.revision, 1)
        s.prefix = Prefix(s.prefix, delta)
        s.revision = revision
        return s.revision
    def tokens(self, scope: str, sid: str) -> tuple[int,...]:
        node = self.get(scope,sid).prefix
        parts = []
        while node is not None:
            parts.append(node.delta); node = node.parent
        return tuple(t for part in reversed(parts) for t in part)

@dataclass
class Request:
    digest: str
    state: str = "queued"
    epoch: Optional[int] = None
    ticket: Optional[Ticket] = None
    attempt: int = 0
    terminal_count: int = 0

class RequestLedger:
    """In-memory idempotency/lifecycle oracle. Durable concurrency is a later gate."""
    TERMINAL = {"completed","failed","cancelled"}
    def __init__(self):
        self.requests: dict[tuple[str,str], Request] = {}
        self.sequence = 0
    def accept(self, scope: str, rid: str, digest: str) -> Request:
        identifier(scope, "scope"); identifier(rid, "request id"); identifier(digest, "digest label")
        key = scope,rid
        if key in self.requests:
            r = self.requests[key]
            if r.digest != digest: raise ContractError("idempotency payload conflict")
            return r
        r = Request(digest); self.requests[key] = r; return r
    def get(self, scope: str, rid: str) -> Request:
        identifier(scope, "scope"); identifier(rid, "request id")
        try: return self.requests[(scope,rid)]
        except KeyError as e: raise ContractError("unknown request") from e
    def start(self, scope: str, rid: str, epoch: int, *, worker_id: str = "local-worker") -> Ticket:
        _uint(epoch,"worker epoch",True); identifier(worker_id)
        r = self.get(scope,rid)
        if r.state != "queued": raise ContractError("request not queued")
        self.sequence = add(self.sequence, 1)
        r.ticket = Ticket(worker_id, epoch, 1, self.sequence, "request")
        r.state, r.epoch, r.attempt = "running", epoch, add(r.attempt, 1)
        return r.ticket
    def cancel(self, scope: str, rid: str) -> str:
        r = self.get(scope,rid)
        if r.state == "queued":
            r.state = "cancelled"; r.terminal_count += 1
        elif r.state == "running": r.state = "cancel_pending"
        return r.state
    def finish(self, scope: str, rid: str, ticket: Ticket, terminal: str) -> None:
        r = self.get(scope,rid)
        if not isinstance(ticket, Ticket) or ticket != r.ticket: raise ContractError("stale worker completion")
        if r.state in self.TERMINAL: raise ContractError("duplicate terminal event")
        if terminal not in self.TERMINAL: raise ContractError("invalid terminal state")
        if r.state not in ("running","cancel_pending"): raise ContractError("not executing")
        if terminal == "cancelled" and r.state != "cancel_pending":
            raise ContractError("cancellation was not requested")
        r.state = terminal; r.terminal_count += 1

def should_park(*, wait_us: int, spill_us: int, restore_us: int, memory_pressure: bool) -> bool:
    """Illustrative break-even rule, not a workload predictor or optimal policy."""
    for n,v in (("wait",wait_us),("spill",spill_us),("restore",restore_us)): _uint(v,n)
    if type(memory_pressure) is not bool: raise ContractError("memory pressure must be boolean")
    return memory_pressure and wait_us > add(spill_us, restore_us)
