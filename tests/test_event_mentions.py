"""Native mention snapshots reach Jev without changing participation targets."""

import json

import httpx
import pytest
from jev_factory import jev_observer

from tests.test_controller import SCOPE, event
from yuki_participation.controller import Controller
from yuki_participation.models import EventMention, Snapshot
from yuki_participation.observer import JevObserver
from yuki_participation.store import SnapshotStore


def snapshot(focus, context=()):
    return Snapshot(scope=SCOPE, focus=focus, context=context, sequence=1, issued_at=focus.at)


def mentions():
    return (
        EventMention(
            kind="member",
            display_name="油桃",
            account_id="account-oil",
            assignment_id="stage-oil",
            person_id="original-oil",
            text_start=0,
            text_end=3,
        ),
        EventMention(
            kind="member",
            display_name="远野",
            account_id="account-far",
            assignment_id="stage-far",
            person_id="original-far",
            text_start=6,
            text_end=9,
        ),
    )


@pytest.mark.asyncio
async def test_native_mentions_reach_real_http_payload_without_becoming_participation_target():
    focus = event().model_copy(update={"text": "@油桃 你是@远野的数字人么", "mentions": mentions()})
    original = snapshot(focus)
    frozen = original.model_dump_json()
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        answers = {}
        for dimension, question in payload["questions"].items():
            options = question["criteria"]
            selected = next(iter(options))
            answers[dimension] = {
                "type": "choice",
                "choice": selected,
                "probabilities": {name: float(name == selected) for name in options},
            }
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await jev_observer(
            "synthetic-key", identity=lambda: {"display_name": "Yuki", "aliases": ()}, client=client
        ).evaluate(original)
    sent = requests[0]["state"]["focus"]
    assert sent["text"] == focus.text and sent["target"] == focus.target
    assert [focus.text[item.text_start : item.text_end] for item in mentions()] == [
        "@油桃",
        "@远野",
    ]
    assert sent["mentions"] == [
        {
            "kind": "member",
            "display_name": "油桃",
            "account": "a1",
            "person": "original-oil",
            "text_start": 0,
            "text_end": 3,
        },
        {
            "kind": "member",
            "display_name": "远野",
            "account": "a2",
            "person": "original-far",
            "text_start": 6,
            "text_end": 9,
        },
    ]
    encoded = json.dumps(requests[0], ensure_ascii=False)
    assert "account-oil" not in encoded and "stage-oil" not in encoded
    assert result.snapshot.model_dump_json() == frozen == original.model_dump_json()


def test_duplicates_unknown_offsets_and_yuki_or_all_are_original_facts():
    first = EventMention(
        kind="yuki",
        display_name="Yuki",
        account_id="bot-account",
        assignment_id="bot-stage",
        presence_id="original-presence",
        text_start=0,
        text_end=5,
    )
    native = (
        first,
        first.model_copy(update={"text_start": 8, "text_end": 13}),
        EventMention(kind="member", display_name="陌生Bot", account_id="external-account"),
        EventMention(kind="all", display_name="全体成员"),
        EventMention(kind="yuki", display_name="Yuki", account_id="second-bot-account"),
    )
    focus = event().model_copy(update={"mentions": native})
    anchor = event("previous", at=99).model_copy(update={"mentions": (first,)})
    material = snapshot(focus, (anchor,))
    state = JevObserver._semantic_state(material)
    observed = state["focus"]["mentions"]
    assert [item["kind"] for item in observed] == ["yuki", "yuki", "member", "all", "yuki"]
    assert (
        observed[0]["account"]
        == observed[1]["account"]
        == state["context"][0]["mentions"][0]["account"]
    )
    assert observed[2] == {"kind": "member", "display_name": "陌生Bot", "account": "a2"}
    assert observed[4]["account"] != observed[0]["account"]
    assert "original-presence" not in json.dumps(state)
    assert material.focus.mentions == native


def test_person_semantic_projection_matches_author_without_rewriting_original_snapshot():
    focus = event().model_copy(update={"author": "original-oil", "mentions": mentions()})
    material = snapshot(focus)
    frozen = material.model_dump_json()
    state = JevObserver._semantic_state(
        material, actor_identity=lambda actor: "current-oil" if actor == "original-oil" else actor
    )
    assert state["focus"]["author"] == state["focus"]["mentions"][0]["person"] == "current-oil"
    assert material.model_dump_json() == frozen


def test_snapshot_store_roundtrip_keeps_original_targets_positions_and_duplicates(tmp_path):
    native = (*mentions(), mentions()[0])
    focus = event().model_copy(update={"mentions": native})
    controller = Controller(SCOPE, 100)
    assert controller.observe_committed_event(focus)
    store = SnapshotStore(tmp_path / "participation.sqlite3")
    try:
        revision = store.save(controller.state, expected_revision=0)
        current, state = store.load(SCOPE)
        assert revision == current == 1
        assert state.events[focus.ref.event_id] == focus
        material = snapshot(focus)
        assert Snapshot.model_validate_json(material.model_dump_json()).focus.mentions == native
    finally:
        store.connection.close()


def test_plain_at_text_and_old_snapshot_do_not_gain_native_mentions():
    material = snapshot(event().model_copy(update={"text": "@Yuki 是普通文字"}))
    raw = material.model_dump(mode="json")
    raw["focus"].pop("mentions")
    raw["projection_revision"] = "semantic-state-v2"
    restored = Snapshot.model_validate(raw)
    assert restored.focus.mentions == ()
    state = JevObserver._semantic_state(restored)
    assert state["focus"]["text"] == material.focus.text
    assert "mentions" not in state["focus"]
    assert state["projection_revision"] == "semantic-state-v2"
