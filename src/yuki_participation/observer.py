"""Jev native protocol adapter. No chat fallback, automatic retry or fabricated scores."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Protocol
from uuid import uuid4

import httpx
from pydantic import ValidationError

from .controller import ActorIdentity, original_actor
from .models import Choice, Observation, ScopedEvent, Snapshot
from .rubric import REVISION, questions

PROJECTION_REVISION = "semantic-state-v3"


class SemanticObserver(Protocol):
    async def evaluate(
        self, snapshot: Snapshot, *, actor_identity: ActorIdentity = original_actor
    ) -> Observation: ...


class JevObserver:
    def __init__(
        self,
        api_key: str,
        *,
        identity: Callable[[], Mapping[str, object]],
        model: str = "jev-1.13.0",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip() or model in {"jev-latest", "latest"}:
            raise ValueError("semantic_provider_requires_key_and_pinned_model")
        self._key = api_key
        self._identity = identity
        self.model = model
        self._owned = client is None
        self._client = client or httpx.AsyncClient(timeout=12)

    def _encode_request(
        self, snapshot: Snapshot, *, actor_identity: ActorIdentity = original_actor
    ) -> bytes:
        state = self._semantic_state(snapshot, actor_identity=actor_identity)
        state["agent_identity"] = dict(self._identity())
        return json.dumps(
            {
                "model": self.model,
                "state": state,
                "questions": questions(
                    unit_options=snapshot.focus.unit_options
                    if snapshot.focus.unit_ambiguous
                    else (),
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _semantic_state(
        snapshot: Snapshot, *, actor_identity: ActorIdentity = original_actor
    ) -> dict[str, object]:
        """Project only evidence needed to interpret speech, with neutral local references.

        Canonical IDs/revisions, scope/generation, request sequence and absolute timestamps
        stay in the original Snapshot for local validation. They are not semantic evidence.
        """
        events = (*snapshot.context, snapshot.focus)
        references = {
            (event.ref.event_id, event.ref.revision): f"e{index}"
            for index, event in enumerate(events, start=1)
        }
        threads = {
            name: f"t{index}"
            for index, name in enumerate(
                dict.fromkeys(
                    [event.thread for event in events]
                    + [option.thread for option in snapshot.focus.unit_options]
                ),
                start=1,
            )
        }
        accounts = {
            account: f"a{index}"
            for index, account in enumerate(
                dict.fromkeys(
                    mention.account_id
                    for event in events
                    for mention in event.mentions
                    if mention.account_id is not None
                ),
                start=1,
            )
        }
        missing_references: dict[tuple[str, int], str] = {}

        def project(event: ScopedEvent) -> dict[str, object]:
            result: dict[str, object] = {
                "id": references[(event.ref.event_id, event.ref.revision)],
                "author": actor_identity(event.author) if event.kind == "human" else event.author,
                "kind": event.kind,
                "thread": threads[event.thread],
                "target": actor_identity(event.target),
                "text": event.text,
                "seconds_from_focus": round(event.at - snapshot.focus.at, 3),
            }
            if event.mentions:
                result["mentions"] = [
                    {
                        "kind": mention.kind,
                        "display_name": mention.display_name,
                        **({"account": accounts[mention.account_id]} if mention.account_id else {}),
                        **(
                            {"person": actor_identity(mention.person_id)}
                            if mention.person_id
                            else {}
                        ),
                        **(
                            {"text_start": mention.text_start}
                            if mention.text_start is not None
                            else {}
                        ),
                        **({"text_end": mention.text_end} if mention.text_end is not None else {}),
                    }
                    for mention in event.mentions
                ]
            if event.reply_to:
                anchor = (event.reply_to.event_id, event.reply_to.revision)
                if anchor in references:
                    result["reply_to"] = references[anchor]
                else:
                    if anchor not in missing_references:
                        missing_references[anchor] = f"unavailable{len(missing_references) + 1}"
                    result["reply_to"] = missing_references[anchor]
                    result["reply_to_unavailable"] = True
            return result

        state: dict[str, object] = {
            "focus": project(snapshot.focus),
            "context": [project(event) for event in snapshot.context],
            "focus_age_seconds": round(snapshot.issued_at - snapshot.focus.at, 3),
            "omitted_context": snapshot.omitted_context,
        }
        if snapshot.focus.unit_ambiguous:
            state["unit_options"] = [
                {
                    "key": option.key,
                    "thread": threads[option.thread],
                    "target": actor_identity(option.target),
                    "label": option.label,
                    **(
                        {
                            "self_anchor": references.get(
                                (option.self_anchor.event_id, option.self_anchor.revision),
                                "unavailable",
                            )
                        }
                        if option.self_anchor is not None
                        else {}
                    ),
                }
                for option in snapshot.focus.unit_options
            ]
        if snapshot.projection_revision is not None:
            state["projection_revision"] = snapshot.projection_revision
        participation = snapshot.participation
        if participation is not None:
            own: dict[str, object] = {}
            if participation.scope_intent is not None:
                hint = participation.scope_intent
                own["scope_intent"] = {
                    "engage": hint.engage,
                    "source_seconds_from_focus": round(hint.at - snapshot.focus.at, 3),
                }
            if participation.units:
                own["units"] = [
                    {
                        "thread": threads[hint.thread],
                        "target": actor_identity(hint.target),
                        "engage": hint.engage,
                        "source_seconds_from_focus": round(hint.at - snapshot.focus.at, 3),
                    }
                    for hint in participation.units
                ]
            if own:
                state["yuki_participation"] = own
        return state

    async def evaluate(
        self, snapshot: Snapshot, *, actor_identity: ActorIdentity = original_actor
    ) -> Observation:
        rubric = questions(
            unit_options=snapshot.focus.unit_options if snapshot.focus.unit_ambiguous else (),
        )
        encoded = self._encode_request(snapshot, actor_identity=actor_identity)
        result = await self._client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            content=encoded,
            timeout=12,
        )
        result.raise_for_status()
        data = result.json()
        if not isinstance(data, dict) or data.get("model") != self.model:
            raise ValueError("semantic_model_revision_mismatch")
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, dict):
            raise ValueError("semantic_answers_missing")
        answers = {}
        invalid = []
        for name, question in rubric.items():
            raw = raw_answers.get(name)
            try:
                if not isinstance(raw, dict) or raw.get("type") != "choice":
                    raise ValueError("wrong_answer_type")
                value = Choice(probabilities=raw["probabilities"], choice=raw["choice"])
                if set(value.probabilities) != set(question["criteria"]):
                    raise ValueError("wrong_options")
                answers[name] = value
            except (ValidationError, ValueError, KeyError, TypeError):
                invalid.append(name)
        if not answers:
            raise ValueError("semantic_all_dimensions_invalid")
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("semantic_usage_invalid")
        selection = answers.get("unit_selection")
        resolved = None
        if (
            selection
            and selection.choice != "unknown"
            and sum(
                p == max(selection.probabilities.values()) for p in selection.probabilities.values()
            )
            == 1
        ):
            resolved = next(
                (
                    option
                    for option in snapshot.focus.unit_options
                    if option.key == selection.choice
                ),
                None,
            )
        return Observation(
            observation_id=str(uuid4()),
            snapshot=snapshot,
            provider="typesafe",
            model_revision=self.model,
            rubric_revision=REVISION,
            received_at=time.time(),
            answers=answers,
            invalid_dimensions=tuple(invalid),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            request_bytes=len(encoded),
            resolved_unit=resolved,
        )

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()
