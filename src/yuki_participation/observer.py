"""System One protocol adapter. No chat fallback, automatic retry or fabricated scores."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

import httpx
from pydantic import ValidationError

from .controller import ActorIdentity, original_actor
from .models import Choice, Observation, ScopedEvent, Snapshot
from .rubric import REVISION, questions

PROJECTION_REVISION = "semantic-state-v3"


@dataclass(frozen=True)
class SemanticConnection:
    """One saved observation endpoint. The revision identifies the connection, not its secret."""

    endpoint: str
    api_key: str
    request_model: str
    response_model: str
    timeout_seconds: float
    name: str
    revision: str


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
        endpoint: str,
        request_model: str,
        response_model: str | None = None,
        timeout_seconds: float = 12,
        name: str = "semantic",
        revision: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._identity = identity
        self._owned = client is None
        self._client = client or httpx.AsyncClient()
        self._connection = self._require(
            endpoint=endpoint,
            api_key=api_key,
            request_model=request_model,
            response_model=response_model or request_model,
            timeout_seconds=timeout_seconds,
            name=name,
            revision=revision,
        )

    @staticmethod
    def _require(
        *,
        endpoint: str,
        api_key: str,
        request_model: str,
        response_model: str,
        timeout_seconds: float,
        name: str,
        revision: str,
    ) -> SemanticConnection:
        if (
            not api_key.strip()
            or not endpoint.strip()
            or not request_model.strip()
            or not response_model.strip()
            or timeout_seconds <= 0
            or not name.strip()
        ):
            raise ValueError("semantic_provider_requires_key_and_pinned_model")
        return SemanticConnection(
            endpoint=endpoint,
            api_key=api_key,
            request_model=request_model,
            response_model=response_model,
            timeout_seconds=timeout_seconds,
            name=name,
            revision=revision,
        )

    def replace_connection(self, connection: SemanticConnection) -> None:
        """Publish the next request's connection. A request already encoding keeps its copy."""

        if not isinstance(connection, SemanticConnection):
            raise ValueError("semantic_provider_requires_key_and_pinned_model")
        self._connection = connection

    @property
    def connection_revision(self) -> str:
        return self._connection.revision

    @property
    def model(self) -> str:
        return self._connection.request_model

    def available(self) -> bool:
        connection = self._connection
        return bool(
            connection.api_key.strip()
            and connection.endpoint.strip()
            and connection.request_model.strip()
            and connection.response_model.strip()
        )

    def _encode_request(
        self,
        snapshot: Snapshot,
        *,
        actor_identity: ActorIdentity = original_actor,
        connection: SemanticConnection | None = None,
    ) -> bytes:
        selected = self._connection if connection is None else connection
        state = self._semantic_state(snapshot, actor_identity=actor_identity)
        identity = dict(self._identity())
        state["agent_identity"] = identity
        return json.dumps(
            {
                "model": selected.request_model,
                "state": state,
                "questions": questions(
                    name=str(identity["display_name"]),
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
        connection = self._connection
        rubric = questions(
            unit_options=snapshot.focus.unit_options if snapshot.focus.unit_ambiguous else (),
        )
        encoded = self._encode_request(
            snapshot, actor_identity=actor_identity, connection=connection
        )
        result = await self._client.post(
            connection.endpoint,
            headers={
                "Authorization": f"Bearer {connection.api_key}",
                "Content-Type": "application/json",
            },
            content=encoded,
            timeout=connection.timeout_seconds,
        )
        result.raise_for_status()
        data = result.json()
        if not isinstance(data, dict) or data.get("model") != connection.response_model:
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
            provider=connection.name,
            model_revision=connection.response_model,
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
