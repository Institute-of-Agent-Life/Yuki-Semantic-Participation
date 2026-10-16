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
from .scheduling import ObservationQueue, ProviderHealth


class ObservationSession:
    def __init__(self, controller: Controller, observer: SemanticObserver) -> None:
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
            self.queue.restore(queue, now=controller.state.now, events=controller.state.events)
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
        self, ref: SourceRef, kind: CandidateKind = CandidateKind.CONVERSATION
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
        self.queue.offer(event, kind)
        self.checkpoint()
        return True

    async def evaluate_due(
        self, now: float, *, active: bool, actor_identity: ActorIdentity = original_actor
    ) -> bool:
        """At most one real request. Inputs may arrive while HTTP is in progress."""
        self.queue.discard_unavailable(now, self.controller.state.events)
        self.checkpoint()
        if now < self.retry_after:
            return False
        snapshot = self.queue.take(
            now,
            active=active,
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
            self.queue.offer(snapshot.focus, snapshot.kind)
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
        self.queue.offer(snapshot.focus, snapshot.kind)
