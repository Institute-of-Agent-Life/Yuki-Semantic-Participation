"""Optional own-state tail. It never supplies evidence about another participant."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from .models import Record, SourceRef, Timestamp


class SelfDelta(Record):
    engage: Literal["join", "stay", "quiet"] | None = None

    @model_validator(mode="before")
    @classmethod
    def retire_mood(cls, value: object) -> object:
        if isinstance(value, dict) and "mood" in value:
            return {key: item for key, item in value.items() if key != "mood"}
        return value


class SelfReport(Record):
    run_ref: str
    sequence: int = Field(ge=1)
    response_id: str
    at: Timestamp
    delta: SelfDelta
    source: SourceRef | None = None


def extract_tail(text: str) -> tuple[str, SelfDelta | None]:
    """Strip a reserved trailing control section even when parsing fails; never retry the model.

    Call on model output, not on user text. Host applies its normal send/voice sanitization
    to the returned body. No host integration is implied by this pure helper.
    """
    marker = "<yuki-state>"
    start = text.rfind(marker)
    if start < 0:
        return text, None
    body, tail = text[:start].rstrip(), text[start + len(marker) :].strip()
    if not tail.endswith("</yuki-state>"):
        return body, None
    raw = tail[: -len("</yuki-state>")].strip()
    if len(raw) > 256:
        return body, None
    try:
        delta = SelfDelta.model_validate(json.loads(raw))
    except (ValidationError, ValueError, TypeError):
        return body, None
    return body, delta if delta.engage is not None else None
