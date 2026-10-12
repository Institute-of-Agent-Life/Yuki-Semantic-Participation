import httpx
import pytest
from test_controller import SCOPE, event

from yuki_participation.models import Snapshot
from yuki_participation.observer import JevObserver
from yuki_participation.rubric import CRITERIA
from yuki_participation.scheduling import ObservationQueue, ProviderHealth


def test_round_robin_and_inflight_coalescing():
    queue = ObservationQueue(SCOPE)
    first = event()
    second = event("e2", at=101).model_copy(update={"thread": "other"})
    queue.offer(first)
    queue.offer(second)
    assert queue.take(102, active=False, context=()) is None
    snapshot = queue.take(103, active=False, context=())
    assert snapshot.focus == first
    queue.offer(event("e3", at=105))
    assert queue.take(115, active=False, context=()) is None
    queue.finish(snapshot.sequence)
    assert queue.take(114, active=False, context=()) is None
    assert queue.take(115, active=False, context=()).focus.ref.event_id == "e3"


def test_earlier_invitation_is_not_replaced_by_later_same_unit_message():
    queue = ObservationQueue(SCOPE)
    first = event("invite", at=100).model_copy(update={"observation_priority": True})
    second = event("ack", at=101)
    queue.offer(first)
    queue.offer(second)
    assert len(queue.pending) == 2
    snapshot = queue.take(110, active=True, context=(first, second))
    assert snapshot.focus == first
    queue.finish(snapshot.sequence)
    assert queue.take(118, active=True, context=(first, second)).focus == second


def test_priority_focus_survives_bounded_queue_overflow():
    queue = ObservationQueue(SCOPE, max_pending=2)
    named = event("named", at=100).model_copy(update={"observation_priority": True})
    queue.offer(named)
    queue.offer(event("ordinary-1", at=101))
    queue.offer(event("ordinary-2", at=102))
    assert set(queue.pending) == {"named", "ordinary-2"}
    assert queue.take(110, active=True, context=()).focus == named


def test_ready_other_unit_is_not_blocked_by_updated_head():
    queue = ObservationQueue(SCOPE)
    head = event("head", at=100)
    other = event("other", at=101).model_copy(update={"thread": "other"})
    queue.offer(head)
    queue.offer(other)
    queue.offer(
        head.model_copy(update={"at": 120, "ref": head.ref.model_copy(update={"revision": 2})})
    )
    assert queue.take(110, active=True, context=()).focus == other


def test_recent_ready_focus_precedes_older_ordinary_foci():
    queue = ObservationQueue(SCOPE)
    queue.offer(event("one", at=100))
    queue.offer(event("two", at=101))
    other = event("other", at=102).model_copy(update={"thread": "other"})
    queue.offer(other)
    snapshot = queue.take(110, active=True, context=())
    assert snapshot.focus == other
    queue.finish(snapshot.sequence)
    assert queue.take(118, active=True, context=()).focus.ref.event_id == "two"


def test_unknown_success_and_quiet_group_are_not_outages():
    health = ProviderHealth()
    assert not health.degraded
    for t in (1, 2, 3):
        health.failure(t)
    assert health.degraded
    health.success(5)
    assert not health.degraded


@pytest.mark.asyncio
async def test_native_payload_probabilities_and_partial_invalid_dimensions():
    seen = []

    def handle(request):
        import json

        payload = json.loads(request.content)
        seen.append(payload)
        answers = {}
        for dimension, question in payload["questions"].items():
            options = question["criteria"]
            chosen = next(iter(options))
            answers[dimension] = {
                "type": "choice",
                "choice": chosen,
                "confidence": 1,
                "probabilities": {k: float(k == chosen) for k in options},
            }
        answers["floor_state"]["probabilities"] = {"yuki": 1}
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 921, "output_tokens": 70},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        observer = JevObserver("synthetic-key", client=client)
        result = await observer.evaluate(
            Snapshot(
                scope=SCOPE,
                focus=event(),
                context=(),
                issued_at=100,
                sequence=1,
            )
        )
    assert "messages" not in seen[0]
    assert "floor_state" not in result.answers
    assert result.invalid_dimensions == ("floor_state",)
    assert result.input_tokens == 921
    assert set(result.answers["interaction_mark"].probabilities) == set(
        CRITERIA["interaction_mark"]
    )


@pytest.mark.asyncio
async def test_no_immediate_retry_on_rate_limit():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        observer = JevObserver("synthetic-key", client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await observer.evaluate(
                Snapshot(scope=SCOPE, focus=event(), context=(), issued_at=100, sequence=1)
            )
    assert len(calls) == 1


def test_missing_key_or_floating_model_is_not_ready():
    with pytest.raises(ValueError):
        JevObserver("")
    with pytest.raises(ValueError):
        JevObserver("synthetic-key", model="jev-latest")


@pytest.mark.asyncio
async def test_malformed_usage_is_protocol_error_not_unhandled_attribute_error():
    def handler(request):
        import json

        answers = {}
        for name, question in json.loads(request.content)["questions"].items():
            chosen = "unknown"
            answers[name] = {
                "type": "choice",
                "choice": chosen,
                "probabilities": {k: float(k == chosen) for k in question["criteria"]},
            }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": ["invalid"],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        observer = JevObserver("synthetic-key", client=client)
        with pytest.raises(ValueError, match="semantic_usage_invalid"):
            await observer.evaluate(
                Snapshot(scope=SCOPE, focus=event(), context=(), issued_at=100, sequence=1)
            )


@pytest.mark.asyncio
async def test_full_request_preserves_focus_context_and_reply_anchor():
    import json

    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(request.content)
        answers = {
            name: {
                "type": "choice",
                "choice": "unknown",
                "probabilities": {key: float(key == "unknown") for key in question["criteria"]},
            }
            for name, question in payload["questions"].items()
        }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 432, "output_tokens": 24},
            },
        )

    anchor = event("anchor", at=90).model_copy(update={"text": "锚" * 700})
    extras = tuple(
        event(f"extra-{i}", at=91 + i).model_copy(update={"text": "文" * 4000}) for i in range(2)
    )
    focus = event(reply_to=anchor.ref)
    snapshot = Snapshot(
        scope=SCOPE, focus=focus, context=(anchor, *extras), issued_at=100, sequence=1
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        observer = JevObserver("synthetic-key", client=client)
        result = await observer.evaluate(snapshot)
    assert result.snapshot.focus == focus
    assert anchor in result.snapshot.context
    assert result.snapshot == snapshot
    assert result.request_bytes == len(seen[0]) > 16000
    assert result.input_tokens == 432  # Provider measurement, never derived from byte count.
    state = json.loads(seen[0])["state"]
    assert state["focus"]["text"] == focus.text
    assert state["focus"]["reply_to"] == state["context"][0]["id"]
    assert state["omitted_context"] == result.snapshot.omitted_context
    assert not {"scope", "generation", "revision", "sequence", "issued_at"} & state.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("large_anchor", [False, True])
async def test_large_focus_or_anchor_reaches_the_real_provider_capacity(large_anchor):
    calls = []

    def reject(request):
        calls.append(request)
        return httpx.Response(413)

    source = event().model_copy(update={"text": "文" * 6000})
    focus = event("focus", at=101, reply_to=source.ref) if large_anchor else source
    context = (source,) if large_anchor else ()
    snapshot = Snapshot(scope=SCOPE, focus=focus, context=context, issued_at=101, sequence=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
        observer = JevObserver("synthetic-key", client=client)
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await observer.evaluate(snapshot)
    assert raised.value.response.status_code == 413
    assert len(calls) == 1
    assert len(calls[0].content) > 16000


def test_semantic_projection_keeps_relationships_without_internal_or_label_ids():
    import json

    anchor = event("label_reveals_expected_close_topic", at=90).model_copy(
        update={
            "kind": "self",
            "author": "Yuki",
            "thread": "sensitive-thread-key",
        }
    )
    focus = event("label_reveals_expected_stop", at=100, reply_to=anchor.ref).model_copy(
        update={
            "thread": "sensitive-thread-key",
        }
    )
    snapshot = Snapshot(scope=SCOPE, focus=focus, context=(anchor,), sequence=99, issued_at=101)
    state = JevObserver._semantic_state(snapshot)
    serialized = json.dumps(state)
    assert "label_reveals" not in serialized
    assert "sensitive-thread-key" not in serialized
    assert SCOPE.conversation_id not in serialized
    assert state["focus"]["reply_to"] == state["context"][0]["id"]
    assert state["context"][0]["author"] == "Yuki"
    assert state["context"][0]["kind"] == "self"
    assert state["context"][0]["seconds_from_focus"] == -10
    assert state["focus_age_seconds"] == 1
    assert state["focus"]["target"] == focus.target


def test_missing_reply_anchor_stays_explicitly_unknown_in_projection():
    from yuki_participation.models import SourceRef

    snapshot = Snapshot(
        scope=SCOPE,
        context=(),
        issued_at=100,
        sequence=1,
        focus=event(reply_to=SourceRef(event_id="private-missing-source", revision=7)),
    )
    state = JevObserver._semantic_state(snapshot)
    assert state["focus"]["reply_to"] == "unavailable1"
    assert state["focus"]["reply_to_unavailable"] is True
