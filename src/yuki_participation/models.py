"""Wire types: scoped facts, observations and opportunities; no execution authority."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Timestamp = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Scope(Record):
    conversation_id: str = Field(min_length=1)
    generation: int = Field(ge=0)


class SourceRef(Record):
    event_id: str = Field(min_length=1)
    revision: int = Field(ge=1)


class HostUnitOption(Record):
    key: str = Field(min_length=1)
    thread: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str = ""
    self_anchor: SourceRef | None = None


class EventMention(Record):
    """Original native target and rendered position supplied by the Host."""

    kind: str
    display_name: str
    account_id: str | None = None
    assignment_id: str | None = None
    person_id: str | None = None
    presence_id: str | None = None
    text_start: int | None = None
    text_end: int | None = None


class ScopedEvent(Record):
    scope: Scope
    ref: SourceRef
    thread: str = Field(min_length=1)
    author: str = Field(min_length=1)
    target: str = Field(min_length=1)
    text: str
    at: Timestamp
    kind: Literal["human", "self"] = "human"
    reply_to: SourceRef | None = None
    unit_ambiguous: bool = False
    unit_options: tuple[HostUnitOption, ...] = ()
    observation_priority: bool = False
    mentions: tuple[EventMention, ...] = ()
    source_order: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_options(self) -> ScopedEvent:
        keys = [option.key for option in self.unit_options]
        if len(keys) != len(set(keys)) or "unknown" in keys:
            raise ValueError("invalid_unit_options")
        return self


class CandidateKind(StrEnum):
    CONVERSATION = "conversation"
    # Historical proposal kinds only. They cannot enter observation or sampling.
    RECALL = "recall"
    CONTACT = "contact"
    INTRINSIC = "intrinsic"


class Choice(Record):
    probabilities: dict[str, Probability]
    choice: str

    @model_validator(mode="after")
    def distribution(self) -> Choice:
        if not self.probabilities or self.choice not in self.probabilities:
            raise ValueError("invalid_choice")
        if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-5):
            raise ValueError("probabilities_must_sum_to_one")
        if self.probabilities[self.choice] < max(self.probabilities.values()) - 1e-5:
            raise ValueError("choice_must_be_maximum")
        return self

    def p(self, key: str) -> float:
        return self.probabilities.get(key, 0.0)


class EngageIntent(Record):
    engage: Literal["join", "stay", "quiet"]
    at: Timestamp
    run_ref: str
    sequence: int = Field(ge=1)
    response_id: str
    basis: tuple[SourceRef, ...] = ()


class UnitIntent(EngageIntent):
    thread: str
    target: str


class ParticipationContext(Record):
    scope_intent: EngageIntent | None = None
    units: tuple[UnitIntent, ...] = ()


class UnitReviewSnapshot(Record):
    """Local original relationship; not a Jev question or authority."""

    thread: str
    target: str
    actor: str
    basis: tuple[SourceRef, ...]
    borrowed: bool = False

    @model_validator(mode="before")
    @classmethod
    def retire_review_progress(cls, value: object) -> object:
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if key != "fast_sequence"}
        return value


class Snapshot(Record):
    scope: Scope
    focus: ScopedEvent
    context: tuple[ScopedEvent, ...]
    sequence: int = Field(ge=1)
    issued_at: Timestamp
    kind: CandidateKind = CandidateKind.CONVERSATION
    omitted_context: int = Field(default=0, ge=0)
    projection_revision: str | None = None
    participation: ParticipationContext | None = None
    unit_reviews: tuple[UnitReviewSnapshot, ...] = ()

    @model_validator(mode="after")
    def same_scope(self) -> Snapshot:
        if any(e.scope != self.scope for e in (self.focus, *self.context)):
            raise ValueError("cross_scope_snapshot")
        if self.focus.at > self.issued_at:
            raise ValueError("future_source")
        return self


class Observation(Record):
    observation_id: str = Field(min_length=1)
    snapshot: Snapshot
    provider: str
    model_revision: str
    rubric_revision: str
    received_at: Timestamp
    answers: dict[str, Choice]
    invalid_dimensions: tuple[str, ...] = ()
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(default=1, ge=1)
    request_bytes: int | None = Field(default=None, ge=0)
    resolved_unit: HostUnitOption | None = None


class Estimate(Record):
    value: Probability
    observation_id: str
    valid_until: Timestamp


class Support(Record):
    kind: Literal["observed", "predicted"]
    scope: Scope
    thread: str
    target: str
    observation_id: str
    covered: tuple[SourceRef, ...]
    basis: SourceRef
    dependencies: tuple[SourceRef, ...] = ()
    issued_at: Timestamp
    valid_until: Timestamp

    def strength(self, now: float) -> float:
        if now >= self.valid_until or now < self.issued_at:
            return 0
        if self.kind == "observed":
            return 1
        return 0.5 * (self.valid_until - now) / max(1, self.valid_until - self.issued_at)


class Proposal(Record):
    proposal_id: str
    scope: Scope
    controller_epoch: int = Field(ge=0)
    kind: CandidateKind
    thread: str
    target_hint: str
    sources: tuple[SourceRef, ...]
    support: Support | None = None
    created_at: Timestamp
    expires_at: Timestamp
    supports: tuple[Support, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def retire_fingerprints(cls, value: object) -> object:
        if isinstance(value, dict) and "source_fingerprints" in value:
            return {key: item for key, item in value.items() if key != "source_fingerprints"}
        return value

    @model_validator(mode="after")
    def valid_cause(self) -> Proposal:
        if self.kind is CandidateKind.INTRINSIC:
            if self.sources or self.support is not None or self.supports:
                raise ValueError("intrinsic_proposal_has_no_semantic_source")
        elif not self.sources:
            raise ValueError("semantic_proposal_requires_source")
        return self


class Effect(Record):
    effect_id: str = Field(min_length=1)
    kind: Literal["message", "compute", "tool"]
    at: Timestamp
    actual_targets: tuple[str, ...] = ()


class Feedback(Record):
    run_ref: str
    proposal_id: str
    sequence: int = Field(ge=1)
    outcome: Literal["accepted", "busy", "rejected", "completed", "interrupted", "no_reply"]
    at: Timestamp
    considered_refs: tuple[SourceRef, ...] = ()
    actual_targets: tuple[str, ...] = ()
    effects: tuple[Effect, ...] = ()
