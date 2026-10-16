"""Typed views over Host-bound participation; no execution or observation policy."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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


class FastInput(Record):
    source: SourceRef
    at: Timestamp
    sequence: int = Field(ge=1)
    source_order: int | None = Field(default=None, ge=0)


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
    covered_fast_sequence: int = Field(default=0, ge=0)


class UnitState(Record):
    binding: UnitBinding
    last_at: Timestamp
    input_ref: SourceRef | None = None
    hint: SelfReport | None = None
    hint_basis: tuple[SourceRef, ...] = ()
    anchors: tuple[SourceRef, ...] = ()
    retired_refs: tuple[SourceRef, ...] = ()
    fast_sequence: int = Field(default=0, ge=0)
    # None means an old reader did not retain review progress, not a full allowance.
    fast_inputs: tuple[FastInput, ...] | None = None
    review: UnitSemanticReview | None = None


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
    unconfirmed_inputs: int | None = None
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
