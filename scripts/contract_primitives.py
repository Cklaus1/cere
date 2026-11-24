"""Bounded scalar/identity primitives for single-threaded CPU reference oracles."""
from __future__ import annotations
from dataclasses import dataclass, field
from uuid import uuid4
import re
from cere_contracts import ContractError

MAX_U64 = (1 << 64) - 1

def uint(value: int, name: str, positive: bool = False) -> None:
    if type(value) is not int or not (1 if positive else 0) <= value <= MAX_U64:
        raise ContractError(f"invalid {name}: expected bounded uint64")

def identifier(value: str, name: str = "identity") -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ContractError(f"invalid {name}")

def add(a: int, b: int) -> int:
    uint(a, "left operand"); uint(b, "right operand")
    if a > MAX_U64 - b:
        raise ContractError("uint64 addition overflow")
    return a + b

@dataclass(frozen=True)
class Ticket:
    """Identity, not a credential. Authenticated adapters must bind its provenance."""
    worker_id: str
    worker_epoch: int
    generation: int
    sequence: int
    kind: str
    operation_id: str = field(default_factory=lambda: uuid4().hex)
    def __post_init__(self):
        identifier(self.worker_id, "worker id")
        for name in ("worker_epoch", "generation", "sequence"):
            uint(getattr(self, name), name, True)
        identifier(self.kind, "ticket kind")
        identifier(self.operation_id, "operation id")

class Clock:
    """One local monotonic clock domain. Never accept a peer's monotonic timestamp."""
    def __init__(self): self.last = 0
    def check(self, now: int) -> None:
        uint(now, "time")
        if now < self.last: raise ContractError("non-monotonic local time")
    def commit(self, now: int) -> None:
        self.check(now); self.last = now
