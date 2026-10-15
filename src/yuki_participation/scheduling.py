"""Bounded round-robin observation coalescing and independent provider health."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .models import CandidateKind, Scope, ScopedEvent, Snapshot


@dataclass
class Pending:
    event: ScopedEvent
    kind: CandidateKind
    first_dirty: float
    last_dirty: float


class ObservationQueue:
    """One instance per scope; one in-flight request, bounded next snapshot sources."""

    def __init__(self, scope: Scope, *, max_pending: int = 64) -> None:
        self.scope = scope
        if max_pending < 1:
            raise ValueError("max_pending_must_be_positive")
        self.pending: OrderedDict[str, Pending] = OrderedDict()
        self.max_pending = max_pending
        self.last_call = float("-inf")
        self.sequence = 0
        self.in_flight: Snapshot | None = None
        self.dropped = 0
        self.expired = 0
        self.invalidated = 0

    def offer(self, event: ScopedEvent, kind: CandidateKind = CandidateKind.CONVERSATION) -> None:
        if kind is CandidateKind.INTRINSIC:
            raise ValueError("intrinsic_opportunity_has_no_semantic_focus")
        if (
            event.scope != self.scope
            or event.kind != "human"
            or kind is not CandidateKind.CONVERSATION
        ):
            return
        # Coalesce revisions of the same fact, not different people/messages in one unit.
        # Context inclusion is not a semantic evaluation of an earlier invitation.
        key = event.ref.event_id
        old = self.pending.get(key)
        if old and old.event.ref == event.ref:
            return
        if old and (
            event.at < old.event.at
            or (
                event.ref.event_id == old.event.ref.event_id
                and event.ref.revision < old.event.ref.revision
            )
        ):
            return
        self.pending[key] = Pending(event, kind, old.first_dirty if old else event.at, event.at)
        while len(self.pending) > self.max_pending:
            evict = next(
                (key for key, item in self.pending.items() if not item.event.observation_priority),
                next(iter(self.pending)),
            )
            del self.pending[evict]
            self.dropped += 1

    @staticmethod
    def expires_at(pending: Pending) -> float:
        # A focus older than the conversation window is context, not a useful
        # request to score. Candidate validity starts only after observation.
        return pending.event.at + 75

    def discard_unavailable(self, now: float, events: dict[str, ScopedEvent]) -> None:
        """Drop revoked or expired facts before their text can leave the host."""
        for key, pending in list(self.pending.items()):
            if events.get(key) != pending.event:
                del self.pending[key]
                self.invalidated += 1
            elif self.expires_at(pending) <= now:
                del self.pending[key]
                self.expired += 1

    def take(
        self, now: float, *, active: bool, context: tuple[ScopedEvent, ...]
    ) -> Snapshot | None:
        if self.in_flight is not None or not self.pending:
            return None
        for key, pending in list(self.pending.items()):
            if self.expires_at(pending) <= now:
                del self.pending[key]
                self.expired += 1
        if not self.pending:
            return None
        debounce, interval, wait = (2, 8, 15) if active else (3, 12, 20)
        if now < self.last_call + interval:
            return None
        # A freshly edited head must not hold up a ready focus in another unit.
        ready = [
            (key, pending)
            for key, pending in self.pending.items()
            if now >= min(pending.last_dirty + debounce, pending.first_dirty + wait)
        ]
        if not ready:
            return None
        # Host priority only chooses which real focus Jev sees first; it never
        # classifies that focus or grants a reply. Preserve older invitations,
        # while ordinary sparse sampling follows the newest conversation turn.
        priority = next((item for item in ready if item[1].event.observation_priority), None)
        key, pending = priority or max(ready, key=lambda item: item[1].event.at)
        del self.pending[key]
        unit = (pending.event.thread, pending.event.target)
        for sibling, item in list(self.pending.items()):
            if (item.event.thread, item.event.target) == unit:
                self.pending.move_to_end(sibling)
        self.sequence += 1
        context = tuple(
            sorted(
                (
                    e
                    for e in context
                    if e.scope == self.scope
                    and e.ref != pending.event.ref
                    and e.at <= pending.event.at
                ),
                key=lambda e: (e.at, e.ref.event_id),
            )
        )
        self.in_flight = Snapshot(
            scope=self.scope,
            focus=pending.event,
            context=context,
            sequence=self.sequence,
            issued_at=now,
            kind=pending.kind,
        )
        self.last_call = now
        return self.in_flight

    def finish(self, sequence: int) -> None:
        if self.in_flight and self.in_flight.sequence == sequence:
            self.in_flight = None

    def checkpoint(self) -> dict[str, object]:
        return {
            "pending": [
                {
                    "ref": item.event.ref.model_dump(mode="json"),
                    "kind": item.kind.value,
                    "first_dirty": item.first_dirty,
                    "last_dirty": item.last_dirty,
                }
                for item in self.pending.values()
            ],
            "last_call": self.last_call if self.last_call != float("-inf") else None,
            "sequence": self.sequence,
            "in_flight": {
                "ref": self.in_flight.focus.ref.model_dump(mode="json"),
                "kind": self.in_flight.kind.value,
                "sequence": self.in_flight.sequence,
            }
            if self.in_flight
            else None,
            "dropped": self.dropped,
            "expired": self.expired,
            "invalidated": self.invalidated,
        }

    def restore(
        self, checkpoint: dict[str, Any], *, now: float, events: dict[str, ScopedEvent]
    ) -> None:
        self.sequence = int(checkpoint.get("sequence", 0))
        last_call = checkpoint.get("last_call")
        self.last_call = float(last_call) if last_call is not None else float("-inf")
        for name in ("dropped", "expired", "invalidated"):
            setattr(self, name, int(checkpoint.get(name, 0)))
        for item in checkpoint.get("pending", []):
            if item["kind"] != CandidateKind.CONVERSATION:
                continue
            event = events.get(item["ref"]["event_id"])
            if event is None or event.ref.model_dump() != item["ref"]:
                self.invalidated += 1
                continue
            self.offer(event, CandidateKind(item["kind"]))
            restored = self.pending.get(event.ref.event_id)
            if restored:
                restored.first_dirty = float(item["first_dirty"])
                restored.last_dirty = float(item["last_dirty"])
        if checkpoint.get("in_flight"):
            inflight = checkpoint["in_flight"]
            self.sequence = max(self.sequence, int(inflight["sequence"]))
            # Scoring has no execution side effect. Re-evaluate only still-permitted,
            # fresh material, with a new request sequence and the original interval.
            event = events.get(inflight["ref"]["event_id"])
            if (
                inflight["kind"] == CandidateKind.CONVERSATION
                and event is not None
                and event.ref.model_dump() == inflight["ref"]
            ):
                self.offer(event, CandidateKind(inflight["kind"]))
            else:
                self.invalidated += 1
        self.discard_unavailable(now, events)


@dataclass
class ProviderHealth:
    failures: int = 0
    failed_since: float | None = None
    last_success: float | None = None
    degraded: bool = False

    def success(self, now: float) -> None:
        # Legitimate unknown is a transport success, not a reason to fall back.
        self.failures = 0
        self.failed_since = None
        self.last_success = now
        self.degraded = False

    def failure(self, now: float) -> None:
        self.failures += 1
        if self.failed_since is None:
            self.failed_since = now
        self.degraded = self.degraded or self.failures >= 3
