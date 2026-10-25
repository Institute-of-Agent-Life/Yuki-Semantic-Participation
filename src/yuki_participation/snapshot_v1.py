"""Frozen snapshot V1 State schema (79d266); format conversion never advances a controller."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .models import (
    CandidateKind,
    Choice,
    Effect,
    Estimate,
    Feedback,
    HostUnitOption,
    Proposal,
    Record,
    Scope,
    ScopedEvent,
    SourceRef,
    Support,
)
from .participation import CHECKPOINT_KEY
from .self_report import SelfReport


class Candidate(Record):
    event: ScopedEvent
    kind: CandidateKind
    support: Support
    value: Estimate
    floor: Estimate


class Boundary(Record):
    source: SourceRef
    thread: str
    target: str
    group_wide: bool = False
    dependencies: tuple[SourceRef, ...] = ()
    author: str = ""
    explicit_stop: bool = False
    released_by: SourceRef | None = None
    release_dependencies: tuple[SourceRef, ...] = ()


class SeenSource(Record):
    revision: int
    at: float
    source_order: int | None = Field(default=None, ge=0)


class RecordedEffect(Record):
    proposal_id: str
    run_ref: str
    effect: Effect


class BeliefBaseline(Record):
    thread: str
    target: str
    at: float = Field(ge=0, allow_inf_nan=False)
    value: tuple[float, float, float, float, float]
    last_action: float = Field(ge=0, allow_inf_nan=False)


class TraceBaseline(Record):
    at: float = Field(ge=0, allow_inf_nan=False)
    value: float = Field(ge=0, allow_inf_nan=False)


class WorkPulse(Record):
    started_at: float = Field(ge=0, allow_inf_nan=False)
    public_at: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    archived_positive: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)


class OutboundAnchor(Record):
    run_ref: str
    at: float = Field(ge=0, allow_inf_nan=False)


class Reception(Record):
    run_ref: str
    at: float = Field(ge=0, allow_inf_nan=False)
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)


class ReceptionBaseline(Record):
    at: float = Field(ge=0, allow_inf_nan=False)
    value: float = Field(allow_inf_nan=False)


class ObservationRefs(Record):
    focus: SourceRef
    context: tuple[SourceRef, ...]
    sequence: int


class StoredObservation(Record):
    observation_id: str
    snapshot: ObservationRefs
    answers: dict[str, Choice]
    received_at: float
    provider: str
    model_revision: str
    rubric_revision: str
    input_tokens: int | None
    output_tokens: int | None
    invalid_dimensions: tuple[str, ...]
    request_bytes: int | None = None
    matching_self_anchor: SourceRef | None = None
    resolved_unit: HostUnitOption | None = None
    unit_resolution: Literal["observer", "semantic_new"] | None = None


class LegacyState(Record):
    scope: Scope
    now: float = Field(ge=0, allow_inf_nan=False)
    epoch: int = 0
    events: dict[str, ScopedEvent] = Field(default_factory=dict)
    observations: dict[str, StoredObservation] = Field(default_factory=dict)
    feedback: dict[str, Feedback] = Field(default_factory=dict)
    effects: dict[str, RecordedEffect] = Field(default_factory=dict)
    proposal_runs: dict[str, str] = Field(default_factory=dict)
    candidates: dict[str, Candidate] = Field(default_factory=dict)
    consumed: dict[str, int] = Field(default_factory=dict)
    invalidated: dict[str, int] = Field(default_factory=dict)
    seen: dict[str, SeenSource] = Field(default_factory=dict)
    boundaries: dict[str, Boundary] = Field(default_factory=dict)
    proposals: dict[str, Proposal] = Field(default_factory=dict)
    pending: str | None = None
    attentions: dict[str, float] = Field(default_factory=dict)
    last_accepted: float = -1e9
    last_human_at: float | None = None
    human_activity_initialized: bool = False
    last_self_message_at: float | None = None
    skipped_seconds: float = 0
    capacity_blocked: bool = False
    observer_checkpoint: dict[str, object] = Field(default_factory=dict)
    host_checkpoint: dict[str, object] = Field(default_factory=dict)
    self_reports: dict[str, SelfReport] = Field(default_factory=dict)
    engagement_report: SelfReport | None = None
    replay_after: float = Field(default=-1, ge=-1, allow_inf_nan=False)
    belief_baselines: dict[str, BeliefBaseline] = Field(default_factory=dict)
    baseline_evictions: int = Field(default=0, ge=0)
    baseline_invalidations: int = Field(default=0, ge=0)
    trace_baselines: dict[str, TraceBaseline] = Field(default_factory=dict)
    familiarities: dict[str, TraceBaseline] = Field(default_factory=dict)
    work_pulses: dict[str, WorkPulse] = Field(default_factory=dict)
    outbound_anchors: dict[str, OutboundAnchor] = Field(default_factory=dict)
    receptions: dict[str, Reception] = Field(default_factory=dict)
    reception_baseline: ReceptionBaseline | None = None
    last_sample_at: float | None = None
    sample_sequence: int = 0

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_gates(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        # Old proposal-time stamps are not evidence of an accepted SELF run.
        obsolete = {
            "threshold",
            "hazard",
            "intrinsic_base_at",
            "last_intrinsic_at",
            "self_reference_at",
            "last_intrinsic_accepted_at",
            "content_keys",
            "seed_claims",
            "seed_attempts",
        }
        migrated = {key: item for key, item in value.items() if key not in obsolete}
        events = migrated.get("events", {})
        if isinstance(events, dict):
            seeds = {
                key: event
                for key, event in events.items()
                if isinstance(event, dict) and event.get("kind") == "seed"
            }
            migrated["events"] = {key: event for key, event in events.items() if key not in seeds}
            # Keep source identity required by old admitted units, never seed text.
            seen = dict(migrated.get("seen", {}))
            for key, event in seeds.items():
                seen.setdefault(key, {"revision": event["ref"]["revision"], "at": event["at"]})
            migrated["seen"] = seen
            observations = migrated.get("observations", {})
            if isinstance(observations, dict):
                migrated["observations"] = {
                    key: observation
                    for key, observation in observations.items()
                    if key not in seeds
                }
            host = dict(migrated.get("host_checkpoint", {}))
            host.pop("seed_cursor", None)
            host.pop("seed_versions", None)
            if isinstance(raw := host.get(CHECKPOINT_KEY), dict) and raw.get("version") == 1:
                units = {}
                for key, original in raw.get("units", {}).items():
                    unit = dict(original)
                    if isinstance(hint := unit.get("hint"), dict):
                        if isinstance(delta := hint.get("delta"), dict):
                            unit["hint"] = {
                                **hint,
                                "delta": {
                                    name: item for name, item in delta.items() if name != "mood"
                                },
                            }
                    refs = (*unit["binding"]["basis"], *unit.get("hint_basis", ()))
                    retired = list(unit.get("retired_refs", ()))
                    for ref in refs:
                        old = seeds.get(ref["event_id"])
                        if old is not None and old["ref"] == ref and ref not in retired:
                            retired.append(ref)
                    unit["retired_refs"] = retired
                    units[key] = unit
                host[CHECKPOINT_KEY] = {**raw, "units": units}
            migrated["host_checkpoint"] = host
        candidates = migrated.get("candidates", {})
        if isinstance(candidates, dict):
            migrated["candidates"] = {
                key: candidate
                for key, candidate in candidates.items()
                if not isinstance(candidate, dict)
                or candidate.get("kind") == CandidateKind.CONVERSATION
            }
        if isinstance(traces := migrated.get("trace_baselines"), dict):
            migrated["trace_baselines"] = {
                key: item
                for key, item in traces.items()
                if key in {"compute", "activity", "social_context", "no_reply", "human_activity"}
            }
        if "work_pulses" not in migrated:
            # Only currently retained, proposal-bound real effects can seed the
            # new density. Retired ratio/message traces cannot prove a Work.
            pulses: dict[str, dict[str, float | None]] = {}
            proposals = migrated.get("proposals", {})
            effects = migrated.get("effects", {})
            if isinstance(proposals, dict) and isinstance(effects, dict):
                for record in effects.values():
                    if not isinstance(record, dict):
                        continue
                    proposal_id = record.get("proposal_id")
                    if not isinstance(proposal_id, str) or proposal_id not in proposals:
                        continue
                    run_ref = record.get("run_ref")
                    effect = record.get("effect")
                    if not isinstance(run_ref, str) or not isinstance(effect, dict):
                        continue
                    kind, at = effect.get("kind"), effect.get("at")
                    if kind not in {"compute", "message"} or not isinstance(at, int | float):
                        continue
                    if not math.isfinite(at) or at < 0:
                        continue
                    pulse = pulses.setdefault(run_ref, {"started_at": at, "public_at": None})
                    started_at = pulse["started_at"]
                    pulse["started_at"] = (
                        min(float(started_at), at) if started_at is not None else at
                    )
                    if kind == "message":
                        public_at = pulse["public_at"]
                        pulse["public_at"] = (
                            min(float(public_at), at) if public_at is not None else at
                        )
            migrated["work_pulses"] = pulses
        return migrated
