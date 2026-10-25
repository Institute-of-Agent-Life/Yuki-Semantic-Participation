"""Sparse source-bound observation coalescing and independent provider health."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby
from typing import Any

from .models import CandidateKind, Scope, ScopedEvent, Snapshot
from .participation import ParticipationUnit


@dataclass
class Pending:
    event: ScopedEvent
    kind: CandidateKind
    first_dirty: float
    last_dirty: float
    unit: ParticipationUnit | None = None


class ObservationQueue:
    """One scope's original sources, with one replaceable review per matched unit."""

    def __init__(self, scope: Scope, *, max_pending: int = 64) -> None:
        self.scope = scope
        if max_pending < 1:
            raise ValueError("max_pending_must_be_positive")
        self.pending: OrderedDict[str, Pending] = OrderedDict()
        # An index over pending, never an identity or semantic-result cache.
        self._unit_pending: dict[ParticipationUnit, str] = {}
        self.max_pending = max_pending
        self.last_call = float("-inf")
        self.sequence = 0
        self.in_flight: Snapshot | None = None
        self._in_flight_pending: Pending | None = None
        self.dropped = 0
        self.expired = 0
        self.invalidated = 0
        self.changed = True

    def discard(self, key: str) -> Pending:
        item = self.pending.pop(key)
        self.changed = True
        if item.unit is not None:
            self._unit_pending.pop(item.unit, None)
        return item

    def offer(
        self,
        event: ScopedEvent,
        kind: CandidateKind = CandidateKind.CONVERSATION,
        *,
        unit: ParticipationUnit | None = None,
    ) -> None:
        if kind is CandidateKind.INTRINSIC:
            raise ValueError("intrinsic_opportunity_has_no_semantic_focus")
        if (
            event.scope != self.scope
            or event.kind != "human"
            or kind is not CandidateKind.CONVERSATION
        ):
            return
        if event.observation_priority:
            unit = None
        key = event.ref.event_id
        previous_key = self._unit_pending.get(unit, key) if unit is not None else key
        old = self.pending.get(previous_key)
        if old and old.event.ref == event.ref:
            return
        if old and (
            event.at < old.event.at
            or (
                event.ref.event_id == old.event.ref.event_id
                and event.ref.revision < old.event.ref.revision
            )
            or (
                event.at == old.event.at
                and event.source_order is not None
                and old.event.source_order is not None
                and event.source_order < old.event.source_order
            )
        ):
            return
        if old is not None:
            self.discard(previous_key)
        # A revision can change which existing unit matches; retain only its current source.
        same_source = self.pending.get(key)
        if same_source is not None:
            self.discard(key)
        self.pending[key] = Pending(
            event, kind, old.first_dirty if old else event.at, event.at, unit
        )
        self.changed = True
        if unit is not None:
            self._unit_pending[unit] = key
        while len(self.pending) > self.max_pending:
            evict = next(
                (key for key, item in self.pending.items() if not item.event.observation_priority),
                next(iter(self.pending)),
            )
            self.discard(evict)
            self.dropped += 1

    @staticmethod
    def expires_at(pending: Pending) -> float:
        return pending.event.at + 75

    def discard_unavailable(self, now: float, events: dict[str, ScopedEvent]) -> None:
        """Drop revoked or expired facts before their text can leave the host."""
        for key, pending in list(self.pending.items()):
            if events.get(key) != pending.event:
                self.discard(key)
                self.invalidated += 1
            elif self.expires_at(pending) <= now:
                self.discard(key)
                self.expired += 1

    def take(self, now: float, *, context: tuple[ScopedEvent, ...]) -> Snapshot | None:
        if self.in_flight is not None or not self.pending:
            return None
        for key, pending in list(self.pending.items()):
            if self.expires_at(pending) <= now:
                self.discard(key)
                self.expired += 1
        if not self.pending or now < self.last_call + 12:
            return None
        ready = [
            (key, pending)
            for key, pending in self.pending.items()
            if now >= min(pending.last_dirty + 3, pending.first_dirty + 20)
        ]
        if not ready:
            return None
        # Independent invitations remain distinct; only already matched ordinary
        # continuation is replaced. Context inclusion is not evaluating each source.
        priority = next((item for item in ready if item[1].event.observation_priority), None)
        key, pending = priority or max(ready, key=lambda item: item[1].event.at)
        self.discard(key)
        self._in_flight_pending = pending
        self.sequence += 1
        preceding = sorted(
            (
                e
                for e in context
                if e.scope == self.scope
                and e.ref != pending.event.ref
                and e.at <= pending.event.at
                and not (
                    e.at == pending.event.at
                    and e.source_order is not None
                    and pending.event.source_order is not None
                    and e.source_order > pending.event.source_order
                )
            ),
            key=lambda e: e.at,
        )
        ordered = []
        for _, same_time in groupby(preceding, key=lambda e: e.at):
            sources = list(same_time)
            if all(e.source_order is not None for e in sources):
                sources.sort(key=lambda e: e.source_order or 0)
            ordered.extend(sources)
        context = tuple(ordered)
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

    def ready(self, now: float, *, retry_after: float = 0.0) -> bool:
        """Project the original request boundary without consuming a pending source."""
        return self.in_flight is None and any(
            max(min(p.last_dirty + 3, p.first_dirty + 20), self.last_call + 12, retry_after)
            <= now
            < self.expires_at(p)
            for p in self.pending.values()
        )

    def next_due(self, *, retry_after: float = 0.0) -> float | None:
        boundaries = [self.expires_at(p) for p in self.pending.values()]
        if self.in_flight is None:
            boundaries.extend(
                due
                for p in self.pending.values()
                if (
                    due := max(
                        min(p.last_dirty + 3, p.first_dirty + 20), self.last_call + 12, retry_after
                    )
                )
                < self.expires_at(p)
            )
        return min(boundaries, default=None)

    def retry(self) -> None:
        """Retry the paid focus only if it has not been superseded by newer material."""
        pending = self._in_flight_pending
        if pending is None:
            return
        if pending.unit is not None and pending.unit in self._unit_pending:
            return
        self.offer(pending.event, pending.kind, unit=pending.unit)
        restored = self.pending.get(pending.event.ref.event_id)
        if restored is not None and restored.event.ref == pending.event.ref:
            first = min(restored.first_dirty, pending.first_dirty)
            self.changed |= restored.first_dirty != first
            restored.first_dirty = first

    def finish(self, sequence: int) -> None:
        if self.in_flight and self.in_flight.sequence == sequence:
            self.in_flight = None
            self._in_flight_pending = None
            self.changed = True

    @staticmethod
    def _pending_record(item: Pending) -> dict[str, object]:
        return {
            "ref": item.event.ref.model_dump(mode="json"),
            "kind": item.kind.value,
            "first_dirty": item.first_dirty,
            "last_dirty": item.last_dirty,
            "unit": item.unit.model_dump(mode="json") if item.unit else None,
        }

    def checkpoint(self) -> dict[str, object]:
        return {
            "pending": [self._pending_record(item) for item in self.pending.values()],
            "last_call": self.last_call if self.last_call != float("-inf") else None,
            "sequence": self.sequence,
            "in_flight": {
                **self._pending_record(self._in_flight_pending),
                "sequence": self.in_flight.sequence,
            }
            if self.in_flight and self._in_flight_pending
            else None,
            "dropped": self.dropped,
            "expired": self.expired,
            "invalidated": self.invalidated,
        }

    def restore(
        self,
        checkpoint: dict[str, Any],
        *,
        now: float,
        events: dict[str, ScopedEvent],
        unit_for: Callable[[ScopedEvent], ParticipationUnit | None] = lambda event: None,
    ) -> None:
        self.sequence = int(checkpoint.get("sequence", 0))
        last_call = checkpoint.get("last_call")
        self.last_call = float(last_call) if last_call is not None else float("-inf")
        for name in ("dropped", "expired", "invalidated"):
            setattr(self, name, int(checkpoint.get(name, 0)))
        pending = list(checkpoint.get("pending", []))
        if checkpoint.get("in_flight"):
            inflight = checkpoint["in_flight"]
            self.sequence = max(self.sequence, int(inflight["sequence"]))
            pending.append(inflight)
        for item in pending:
            if item["kind"] != CandidateKind.CONVERSATION:
                continue
            event = events.get(item["ref"]["event_id"])
            if event is None or event.ref.model_dump() != item["ref"]:
                self.invalidated += 1
                continue
            unit = (
                ParticipationUnit.model_validate(item["unit"])
                if item.get("unit")
                else unit_for(event)
            )
            previous_key = self._unit_pending.get(unit) if unit else None
            previous = self.pending.get(previous_key) if previous_key else None
            self.offer(event, CandidateKind(item["kind"]), unit=unit)
            restored = self.pending.get(event.ref.event_id)
            if restored:
                restored.first_dirty = min(
                    float(item.get("first_dirty", event.at)),
                    previous.first_dirty if previous else event.at,
                )
                restored.last_dirty = float(item.get("last_dirty", event.at))
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
