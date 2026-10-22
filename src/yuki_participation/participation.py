"""Typed views over Host-bound participation; no execution or observation policy."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .models import Choice, Record, Scope, SourceRef, Timestamp
from .self_report import SelfReport

CHECKPOINT_KEY = "participation_v1"


class ParticipationUnit(Record):
    thread: str = Field(min_length=1)
    target: str = Field(min_length=1)


class UnitBinding(Record):
    scope: Scope
    unit: ParticipationUnit
    actor: str = Field(min_length=1)
    basis: tuple[SourceRef, ...] = Field(min_length=1)


class UnitSemanticReview(Record):
    """Latest actual, source-bound interpretation of this relationship."""

    source: SourceRef
    at: Timestamp
    sequence: int = Field(ge=1)
    observation_id: str
    basis: tuple[SourceRef, ...]
    resolved_unit: ParticipationUnit | None
    interaction: Choice | None
    floor: Choice | None
    source_order: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def retire_review_progress(cls, value: object) -> object:
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if key != "covered_fast_sequence"}
        return value


class UnitState(Record):
    binding: UnitBinding
    last_at: Timestamp
    input_ref: SourceRef | None = None
    hint: SelfReport | None = None
    hint_basis: tuple[SourceRef, ...] = ()
    anchors: tuple[SourceRef, ...] = ()
    retired_refs: tuple[SourceRef, ...] = ()
    # Original admissions, needed for late feedback and equal-time source ordering.
    # They never grant, spend or replenish continuation permission.
    continuation_refs: tuple[SourceRef, ...] = ()
    review: UnitSemanticReview | None = None

    @model_validator(mode="before")
    @classmethod
    def retire_review_progress(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        result = {
            key: item for key, item in value.items() if key not in {"fast_sequence", "fast_inputs"}
        }
        if "continuation_refs" not in result and "fast_inputs" in value:
            result["continuation_refs"] = tuple(
                item["source"] for item in (value["fast_inputs"] or ())
            )
        return result


class ExpressionBinding(Record):
    binding: UnitBinding
    anchor: SourceRef | None = None


class ParticipationCheckpoint(Record):
    version: Literal[1] = 1
    units: dict[str, UnitState] = Field(default_factory=dict)
    expressions: dict[str, ExpressionBinding] = Field(default_factory=dict)


class UnitParticipation(Record):
    binding: UnitBinding
    last_at: Timestamp
    anchors: tuple[SourceRef, ...] = ()
    engage: Literal["join", "stay", "quiet"] | None = None
    expressed: bool = False
    closed: bool = False
    semantic_continuation: bool = True

    @property
    def unit(self) -> ParticipationUnit:
        return self.binding.unit


class ParticipationView(Record):
    scope: Scope
    event_ref: SourceRef
    source_valid: bool
    current_resolved: bool = False
    current_unit: UnitParticipation | None = None
    addressed: bool = False
    matched_unit: UnitParticipation | None = None
    candidates: tuple[UnitParticipation, ...] = ()
    ambiguous: bool = False
    needs_observation: bool = True
