"""One scope's asynchronous observer lifecycle, separate from host execution.

The host calls tick on its own clock and persists checkpoint at returned boundaries.
There is no internal perpetual timer and no second Agent executor.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

import httpx
from pydantic import TypeAdapter

from .controller import ActorIdentity, Controller, original_actor
from .models import CandidateKind, ScopedEvent, Snapshot, SourceRef
from .observer import SemanticObserver
from .participation import ParticipationUnit
from .scheduling import ObservationQueue, ProviderHealth


class ObservationSession:
    def __init__(
        self,
        controller: Controller,
        observer: SemanticObserver,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> None:
        self.controller = controller
        self.observer = observer
        self.queue = ObservationQueue(controller.state.scope)
        self.health = ProviderHealth()
        self.retry_after = 0.0
        self.last_error: str | None = None
        self.last_failure: dict[str, object] | None = None
        checkpoint = controller.state.observer_checkpoint
        if checkpoint:
            if checkpoint.get("version") != 1:
                raise ValueError("unsupported_observer_checkpoint")
            queue = checkpoint.get("queue")
            retry_after = checkpoint.get("retry_after")
            last_error = checkpoint.get("last_error")
            failure = checkpoint.get("last_failure")
            if (
                not isinstance(queue, dict)
                or not isinstance(retry_after, (int, float))
                or (last_error is not None and not isinstance(last_error, str))
                or (failure is not None and not isinstance(failure, dict))
            ):
                raise ValueError("invalid_observer_checkpoint")
            self.queue.restore(
                queue,
                now=controller.state.now,
                events=controller.state.events,
                unit_for=lambda event: self._review_unit(event, actor_identity),
            )
            self.health = TypeAdapter(ProviderHealth).validate_python(checkpoint.get("health"))
            self.retry_after = float(retry_after)
            self.last_error = last_error
            self.last_failure = failure
        # Existing observations must never be mistaken for newer responses after a restart.
        self.queue.sequence = max(
            self.queue.sequence,
            max((o.snapshot.sequence for o in controller.state.observations.values()), default=0),
        )
        self.checkpoint()

    def checkpoint(self) -> None:
        """Host persists controller.state; queue material refers to its canonical source copy."""
        self.controller.state = self.controller.state.model_copy(
            update={
                "observer_checkpoint": {
                    "version": 1,
                    "queue": self.queue.checkpoint(),
                    "health": asdict(self.health),
                    "retry_after": self.retry_after,
                    "last_error": self.last_error,
                    "last_failure": self.last_failure,
                }
            }
        )

    def observe(self, event: ScopedEvent, kind: CandidateKind = CandidateKind.CONVERSATION) -> None:
        """Record context only. Hosts separately request observation when needed."""
        self.controller.observe_committed_event(event)

    def request_observation(
        self,
        ref: SourceRef,
        kind: CandidateKind = CandidateKind.CONVERSATION,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> bool:
        event = self.controller.state.events.get(ref.event_id)
        if event is None or event.ref != ref or not self.controller._valid(ref):
            return False
        if event.kind != "human" or kind is not CandidateKind.CONVERSATION:
            return False
        source = ref.model_dump(mode="json")
        if (
            self.last_failure is not None
            and self.last_failure.get("category") == "request_validation"
            and self.last_failure.get("source") == source
        ):
            return False
        self.queue.offer(event, kind, unit=self._review_unit(event, actor_identity))
        self.checkpoint()
        return True

    def _review_unit(
        self, event: ScopedEvent, actor_identity: ActorIdentity
    ) -> ParticipationUnit | None:
        if event.observation_priority:
            return None
        matched = self.controller.participation_view(
            event, max(self.controller.state.now, event.at), actor_identity=actor_identity
        ).matched_unit
        return (
            ParticipationUnit(
                thread=matched.unit.thread, target=actor_identity(matched.unit.target)
            )
            if matched is not None
            else None
        )

    def _discard_quiet_reviews(self, now: float, actor_identity: ActorIdentity) -> None:
        checkpoint = self.controller._participation_checkpoint()
        quiet = {
            (unit.unit.thread, actor_identity(unit.unit.target))
            for unit in self.controller._current_units(now, checkpoint, actor_identity)
            if unit.engage == "quiet"
        }
        if not quiet:
            return
        admitted: dict[SourceRef, set[tuple[str, str]]] = {}
        hints: dict[tuple[str, str], list[tuple[float, SourceRef | None]]] = {}
        observed: dict[tuple[str, str], set[SourceRef]] = {}
        for unit in checkpoint.units.values():
            identity = (unit.binding.unit.thread, actor_identity(unit.binding.unit.target))
            if identity not in quiet or not self.controller._binding_valid(
                unit.binding, checkpoint
            ):
                continue
            for ref in unit.continuation_refs:
                admitted.setdefault(ref, set()).add(identity)
            hint = unit.hint
            if (
                hint is not None
                and hint.delta.engage == "quiet"
                and all(self.controller._unit_ref_valid(ref, unit) for ref in unit.hint_basis)
            ):
                hints.setdefault(identity, []).append(
                    (hint.at, self.controller._hint_source(hint, unit.hint_basis))
                )
                observed.setdefault(identity, set()).update(unit.hint_basis)
        for key, pending in list(self.queue.pending.items()):
            if pending.event.observation_priority:
                continue
            # Only actual admitted continuation can be redundant. Merely included
            # context, in-flight requests and newer invitations are not cleared.
            identities = admitted.get(pending.event.ref, set())
            if len(identities) != 1:
                continue
            identity = next(iter(identities))
            if pending.event.ref in observed.get(identity, ()) or any(
                source == pending.event.ref
                or self.controller._source_before(pending.event.at, pending.event.ref, at, source)
                for at, source in hints.get(identity, ())
            ):
                self.queue.discard(key)

    async def evaluate_due(
        self, now: float, *, actor_identity: ActorIdentity = original_actor
    ) -> bool:
        """At most one real request. Inputs may arrive while HTTP is in progress."""
        self.queue.discard_unavailable(now, self.controller.state.events)
        self._discard_quiet_reviews(now, actor_identity)
        self.checkpoint()
        if now < self.retry_after:
            return False
        snapshot = self.queue.take(
            now,
            context=tuple(self.controller.state.events.values()),
        )
        if snapshot is None:
            return False
        snapshot = self.controller.freeze_observation(snapshot, actor_identity=actor_identity)
        self.queue.in_flight = snapshot
        self.checkpoint()
        try:
            observation = (
                await self.observer.evaluate(snapshot)
                if actor_identity is original_actor
                else await self.observer.evaluate(snapshot, actor_identity=actor_identity)
            )
            # Provider identity checks and payload parsing belong to the adapter. A custom
            # observer must still return the exact snapshot actually submitted.
            if observation.snapshot != snapshot:
                raise ValueError("observation_snapshot_mismatch")
            applied = self.controller.apply_semantic_observation(
                observation, actor_identity=actor_identity
            )
            required = {"interaction_mark", "information_state", "floor_state"}
            if not required <= observation.answers.keys():
                # Keep usable partial interpretation (e.g. an actual stop), but repeated
                # malformed required dimensions cannot advertise a healthy channel.
                self._failed(snapshot, now, "partial_required_dimensions_invalid")
                return False
            self.health.success(max(now, observation.received_at))
            self.last_error = None
            self.retry_after = 0.0
            return applied
        except (httpx.HTTPError, ValueError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            category = (
                "authentication"
                if status in {401, 403}
                else "request_validation"
                if status == 422
                else "rate_limit"
                if status == 429
                else "provider_http"
                if status is not None
                else "response_invalid"
                if isinstance(exc, ValueError)
                else "transport"
            )
            self._failed(snapshot, now, category, status=status)
            return False
        except asyncio.CancelledError:
            self.queue.retry()
            raise
        finally:
            self.queue.finish(snapshot.sequence)
            self.queue.discard_unavailable(now, self.controller.state.events)
            self.checkpoint()

    def _failed(
        self, snapshot: Snapshot, now: float, category: str, *, status: int | None = None
    ) -> None:
        """Keep bounded diagnostics and retry only still-fresh real inputs."""
        self.last_error = category
        self.last_failure = {
            "category": category,
            "status": status,
            "at": now,
            "source": snapshot.focus.ref.model_dump(mode="json"),
        }
        if status == 422:
            # This request was rejected. Do not retry the same source or turn it
            # into a permanent scope-wide configuration/ownership failure.
            return
        self.health.failure(now)
        if status in {401, 403}:
            self.health.degraded = True
            delay = min(900, 300 * 2 ** min(self.health.failures - 1, 2))
        else:
            delay = min(180, 30 * 2 ** min(self.health.failures - 1, 3))
        self.retry_after = now + delay
        self.queue.retry()
