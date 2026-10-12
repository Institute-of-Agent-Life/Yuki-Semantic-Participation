import asyncio

import httpx
import pytest
from test_controller import controller, event, observation

from yuki_participation.controller import Controller
from yuki_participation.observer import JevObserver
from yuki_participation.session import ObservationSession
from yuki_participation.store import SnapshotConflict, SnapshotStore


@pytest.mark.asyncio
async def test_single_inflight_and_new_input_is_coalesced():
    started, release = asyncio.Event(), asyncio.Event()

    class Delayed:
        async def evaluate(self, snapshot):
            started.set()
            await release.wait()
            return observation(snapshot.focus).model_copy(
                update={"snapshot": snapshot, "received_at": snapshot.issued_at}
            )

    session = ObservationSession(controller(), Delayed())
    queue_observation(session, event())
    task = asyncio.create_task(session.evaluate_due(130, active=False))
    await started.wait()
    queue_observation(session, event("later", at=131))
    assert not await session.evaluate_due(200, active=False)
    release.set()
    assert await task
    assert len(session.queue.pending) == 1


@pytest.mark.asyncio
async def test_real_failures_backoff_and_keep_pending_material():
    class Unavailable:
        async def evaluate(self, snapshot):
            raise httpx.ConnectError("synthetic")

    session = ObservationSession(controller(), Unavailable())
    queue_observation(session, event())
    assert not await session.evaluate_due(130, active=False)
    assert session.queue.pending
    assert session.health.failures == 1
    assert not await session.evaluate_due(131, active=False)
    assert session.health.failures == 1


@pytest.mark.asyncio
async def test_unknown_result_is_success_without_a_candidate():
    class Unknown:
        async def evaluate(self, snapshot):
            return observation(snapshot.focus, act="unknown").model_copy(
                update={"snapshot": snapshot, "received_at": snapshot.issued_at}
            )

    session = ObservationSession(controller(), Unknown())
    queue_observation(session, event())
    assert await session.evaluate_due(130, active=False)
    assert not session.controller.state.candidates
    assert session.health.failures == 0


@pytest.mark.asyncio
async def test_revoked_focus_is_never_sent_to_provider():
    class NeverCalled:
        async def evaluate(self, snapshot):
            pytest.fail("revoked text must not leave the host")

    session = ObservationSession(controller(), NeverCalled())
    source = event()
    queue_observation(session, source)
    session.controller.observe_source_change(source.ref)
    assert not await session.evaluate_due(130, active=False)
    assert session.queue.invalidated == 1
    assert not session.queue.pending


@pytest.mark.asyncio
async def test_pending_and_health_survive_store_restore(tmp_path):
    class Unavailable:
        async def evaluate(self, snapshot):
            raise httpx.ConnectError("synthetic")

    session = ObservationSession(controller(), Unavailable())
    source = event()
    queue_observation(session, source)
    assert not await session.evaluate_due(108, active=True)
    store = SnapshotStore(tmp_path / "controller.sqlite3")
    store.save(session.controller.state, expected_revision=0)
    _, saved = store.load(source.scope)
    restored = ObservationSession(Controller.restore(saved, 109), Unavailable())
    assert list(restored.queue.pending) == [source.ref.event_id]
    assert restored.queue.sequence == 1
    assert restored.health.failures == 1
    assert restored.retry_after == 138
    assert not await restored.evaluate_due(137, active=True)
    assert restored.health.failures == 1
    assert not await restored.evaluate_due(138, active=True)
    assert restored.health.failures == 2
    store.close()


@pytest.mark.asyncio
async def test_interrupted_scoring_requeues_only_fresh_original_source(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    class Delayed:
        async def evaluate(self, snapshot):
            started.set()
            await release.wait()
            return observation(snapshot.focus).model_copy(
                update={
                    "snapshot": snapshot,
                    "received_at": snapshot.issued_at,
                }
            )

    session = ObservationSession(controller(), Delayed())
    source = event()
    queue_observation(session, source)
    task = asyncio.create_task(session.evaluate_due(108, active=True))
    await started.wait()
    store = SnapshotStore(tmp_path / "controller.sqlite3")
    store.save(session.controller.state, expected_revision=0)
    _, saved = store.load(source.scope)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    restored = ObservationSession(Controller.restore(saved, 109), Delayed())
    assert restored.queue.in_flight is None
    assert restored.queue.sequence == 1
    assert list(restored.queue.pending) == [source.ref.event_id]
    release.set()
    assert await restored.evaluate_due(138, active=True)
    assert restored.controller.state.observations[source.ref.event_id].snapshot.sequence == 2
    expired = ObservationSession(Controller.restore(saved, 190), Delayed())
    assert not expired.queue.pending
    assert not await expired.evaluate_due(190, active=True)
    store.close()


def test_retired_scope_deletion_checks_revision_and_releases_capacity(tmp_path):
    c = controller()
    store = SnapshotStore(tmp_path / "controller.sqlite3")
    store.save(c.state, expected_revision=0)
    with pytest.raises(SnapshotConflict):
        store.delete_scope(c.state.scope, expected_revision=0)
    assert store.load(c.state.scope) is not None
    store.delete_scope(c.state.scope, expected_revision=1)
    assert store.load(c.state.scope) is None
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing,failures", [("floor_state", 1), ("boundary_scope", 0)])
async def test_required_and_optional_dimension_failures_differ(missing, failures):
    class Partial:
        async def evaluate(self, snapshot):
            response = observation(snapshot.focus)
            return response.model_copy(
                update={
                    "snapshot": snapshot,
                    "received_at": snapshot.issued_at,
                    "answers": {k: v for k, v in response.answers.items() if k != missing},
                    "invalid_dimensions": (missing,),
                }
            )

    session = ObservationSession(controller(), Partial())
    queue_observation(session, event())
    await session.evaluate_due(108, active=True)
    assert session.health.failures == failures
    assert bool(session.queue.pending) == bool(failures)


@pytest.mark.asyncio
async def test_large_source_uses_real_provider_failure_and_health():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        observer = JevObserver("synthetic-key", client=client)
        session = ObservationSession(controller(), observer)
        oversized = event().model_copy(update={"text": "文" * 6000})
        queue_observation(session, oversized)
        assert not await session.evaluate_due(108, active=True)
        assert session.queue.pending
        assert session.health.failures == 1
        assert len(calls) == 1
        assert len(calls[0].content) > 16000
        restored = ObservationSession(Controller.restore(session.controller.state, 110), observer)
        assert restored.queue.pending
        assert restored.health.failures == 1
        assert not await restored.evaluate_due(110, active=True)
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_observer_cannot_replace_the_submitted_snapshot():
    class WrongPreparation:
        async def evaluate(self, snapshot):
            return observation(snapshot.focus).model_copy(
                update={"snapshot": snapshot.model_copy(update={"focus": event("forged")})}
            )

    session = ObservationSession(controller(), WrongPreparation())
    queue_observation(session, event())
    assert not await session.evaluate_due(108, active=True)
    assert session.health.failures == 1
    assert session.queue.pending


@pytest.mark.asyncio
async def test_provider_health_survives_expiry_and_restart_until_real_unknown():
    class Unavailable:
        async def evaluate(self, snapshot):
            raise httpx.ConnectError("synthetic")

    class Unknown:
        async def evaluate(self, snapshot):
            return observation(snapshot.focus, act="unknown").model_copy(
                update={"snapshot": snapshot, "received_at": snapshot.issued_at}
            )

    session = ObservationSession(controller(), Unavailable())
    queue_observation(session, event())
    for at in (108, 138, 198):
        if at == 198:
            queue_observation(session, event("fresh-before-third-failure", at=180))
        assert not await session.evaluate_due(at, active=True)
    assert session.health.degraded
    assert not await session.evaluate_due(800, active=False)
    assert not session.queue.pending
    assert session.health.degraded
    session.checkpoint()
    restored = ObservationSession(Controller.restore(session.controller.state, 801), Unknown())
    assert restored.health.degraded
    queue_observation(restored, event("fresh-after-outage", at=802))
    assert await restored.evaluate_due(840, active=False)
    assert not restored.health.degraded
    assert not restored.controller.state.candidates


@pytest.mark.parametrize(
    "field, value",
    [
        ("queue", []),
        ("retry_after", "invalid"),
        ("last_error", {}),
        ("health", {"failures": "invalid"}),
    ],
)
def test_observer_checkpoint_rejects_invalid_shapes_before_restore(field, value):
    class Unused:
        async def evaluate(self, snapshot):
            pytest.fail("checkpoint validation cannot invoke a provider")

    session = ObservationSession(controller(), Unused())
    session.controller.state.observer_checkpoint[field] = value
    with pytest.raises(ValueError):
        ObservationSession(session.controller, Unused())


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 503])
async def test_http_failure_retries_fresh_input_and_success_recovers(status):
    calls = []

    class Recovering:
        async def evaluate(self, snapshot):
            calls.append(snapshot.focus.ref)
            if len(calls) == 1:
                request = httpx.Request("POST", "https://fixture.invalid")
                response = httpx.Response(status, request=request)
                raise httpx.HTTPStatusError(
                    "secret response must not persist", request=request, response=response
                )
            return observation(snapshot.focus, act="unknown").model_copy(
                update={"snapshot": snapshot, "received_at": snapshot.issued_at}
            )

    session = ObservationSession(controller(), Recovering())
    queue_observation(session, event())
    assert not await session.evaluate_due(108, active=True)
    delay = 300 if status in {401, 403} else 30
    assert session.retry_after == 108 + delay
    assert session.last_failure["status"] == status
    assert "secret" not in str(session.controller.state.observer_checkpoint)
    restored = ObservationSession(
        Controller.restore(session.controller.state, 109), session.observer
    )
    fresh_at = 108 + delay - 5
    queue_observation(restored, event("fresh", at=fresh_at))
    assert not await restored.evaluate_due(108 + delay - 1, active=True)
    assert await restored.evaluate_due(108 + delay, active=True)
    assert len(calls) == 2
    assert calls[-1].event_id == "fresh"
    assert restored.health.failures == 0 and not restored.health.degraded
    assert restored.last_error is None and restored.retry_after == 0


@pytest.mark.asyncio
async def test_422_rejects_only_original_request_and_does_not_poison_next_source():
    calls = []

    class RejectFirst:
        async def evaluate(self, snapshot):
            calls.append(snapshot.focus.ref)
            if len(calls) == 1:
                request = httpx.Request("POST", "https://fixture.invalid")
                response = httpx.Response(422, request=request)
                raise httpx.HTTPStatusError("synthetic", request=request, response=response)
            return observation(snapshot.focus).model_copy(
                update={"snapshot": snapshot, "received_at": snapshot.issued_at}
            )

    session = ObservationSession(controller(), RejectFirst())
    queue_observation(session, event())
    assert not await session.evaluate_due(108, active=True)
    assert session.last_error == "request_validation"
    assert session.last_failure["source"] == event().ref.model_dump(mode="json")
    assert session.health.failures == 0 and not session.queue.pending
    queue_observation(session, event())
    assert not await session.evaluate_due(116, active=True)
    assert len(calls) == 1
    queue_observation(session, event("valid-next", at=117))
    assert await session.evaluate_due(120, active=True)
    assert calls[-1].event_id == "valid-next"


@pytest.mark.asyncio
async def test_old_configuration_invalid_checkpoint_can_retry_without_resetting_controller():
    class Unknown:
        async def evaluate(self, snapshot):
            return observation(snapshot.focus, act="unknown").model_copy(
                update={"snapshot": snapshot, "received_at": snapshot.issued_at}
            )

    session = ObservationSession(controller(), Unknown())
    queue_observation(session, event())
    checkpoint = session.controller.state.observer_checkpoint
    checkpoint["health"].update(configuration_valid=False, failures=1, degraded=True)
    checkpoint["retry_after"] = 138
    restored = ObservationSession(Controller.restore(session.controller.state, 109), Unknown())
    assert "configuration_valid" not in restored.controller.state.observer_checkpoint["health"]
    assert not await restored.evaluate_due(137, active=True)
    assert await restored.evaluate_due(138, active=True)
    assert not restored.health.degraded and restored.health.failures == 0


def queue_observation(session, source):
    session.controller.observe_committed_event(source)
    session.request_observation(source.ref)
