"""Typed views over Host-bound participation; no execution or observation policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from .models import Choice, Record, Scope, SourceRef, Timestamp
from .self_report import SelfReport

CHECKPOINT_KEY = "participation_v1"


class _RetiredRefs(tuple[SourceRef, ...]):
    """Ordered original references with a derived, non-persisted membership index."""

    _members: frozenset[SourceRef]

    def __new__(cls, values: Iterable[SourceRef]) -> Self:
        result = super().__new__(cls, values)
        result._members = frozenset(result)
        return result

    def __contains__(self, value: object) -> bool:
        return value in self._members


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

    @field_validator("retired_refs", mode="after")
    @classmethod
    def index_retired_refs(cls, value: tuple[SourceRef, ...]) -> tuple[SourceRef, ...]:
        return _RetiredRefs(value)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        if update is not None and "retired_refs" in update:
            update = {**update, "retired_refs": _RetiredRefs(update["retired_refs"])}
        return super().model_copy(update=update, deep=deep)

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


@dataclass(frozen=True, slots=True)
class UnitHintFact:
    engage: Literal["join", "stay", "quiet"] | None
    source: SourceRef | None
    basis: tuple[SourceRef, ...]
    at: float
    sequence: int


@dataclass(frozen=True, slots=True)
class UnitReviewFact:
    interaction: str | None
    floor: str | None
    source: SourceRef
    at: float
    source_order: int | None
    sequence: int
    boundary_scope: str | None


@dataclass(frozen=True, slots=True)
class UnitContextFact:
    actor: str
    hint: UnitHintFact | None
    review: UnitReviewFact | None
