"""Scope-local controller. Calls return opportunities, never language or executable work."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Any, Literal, TypeGuard
from uuid import uuid4

from pydantic import Field, model_validator

from . import dynamics
from .autonomy_parameters import DEFAULT_AUTONOMY_PARAMETERS, AutonomyParameters
from .models import (
    CandidateKind,
    Choice,
    Effect,
    EngageIntent,
    Estimate,
    Feedback,
    HostUnitOption,
    Observation,
    ParticipationContext,
    Proposal,
    Record,
    Scope,
    ScopedEvent,
    Snapshot,
    SourceRef,
    Support,
    UnitIntent,
    UnitReviewSnapshot,
)
from .participation import (
    CHECKPOINT_KEY,
    ExpressionBinding,
    ParticipationCheckpoint,
    ParticipationUnit,
    ParticipationView,
    UnitBinding,
    UnitParticipation,
    UnitSemanticReview,
    UnitState,
)
from .rubric import CRITERIA, REVISION
from .self_report import SelfReport

HOT_FIELDS = frozenset(
    {
        "now",
        "last_sample_at",
        "sample_sequence",
        "skipped_seconds",
        "replay_after",
        "baseline_evictions",
        "baseline_invalidations",
        "belief_baselines",
        "trace_baselines",
        "attentions",
        "reception_baseline",
    }
)


ActorIdentity = Callable[[str], str]


def original_actor(actor: str) -> str:
    return actor


def _dimension_known(answer: Choice | None) -> TypeGuard[Choice]:
    """Unknown-winning distributions stay diagnostic/soft evidence, not eligibility.

    Ties involving unknown are insufficient too. Usable probabilities remain unchanged;
    this is not a conversion to the provider's selected choice or a renormalization.
    """
    return answer is not None and answer.p("unknown") < max(
        (value for name, value in answer.probabilities.items() if name != "unknown"),
        default=0.0,
    )


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
    value: dynamics.Belief
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
    """Original text is stored once in State.events, not copied into every snapshot."""

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

    @classmethod
    def from_observation(cls, observation: Observation) -> StoredObservation:
        return cls(
            **observation.model_dump(exclude={"snapshot", "attempt_count"}),
            snapshot=ObservationRefs(
                focus=observation.snapshot.focus.ref,
                context=tuple(e.ref for e in observation.snapshot.context),
                sequence=observation.snapshot.sequence,
            ),
            unit_resolution="observer" if observation.resolved_unit is not None else None,
        )


type BeliefAction = (
    tuple[float, Literal["human"], str, StoredObservation | UnitState]
    | tuple[float, Literal["self"], str, RecordedEffect]
)


class State(Record):
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


def _same_fact(left: object, right: object) -> bool:
    """Compare ordered facts without encoding retained bodies or losing signed zero."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, Record) and isinstance(right, Record):
        return _same_fact(left.__dict__, right.__dict__)
    if isinstance(left, dict) and isinstance(right, dict):
        return len(left) == len(right) and all(
            key == other and _same_fact(value, right_value)
            for (key, value), (other, right_value) in zip(left.items(), right.items(), strict=True)
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _same_fact(value, other) for value, other in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float) and left == right == 0:
        return math.copysign(1, left) == math.copysign(1, right)
    return left == right


class Controller:
    """Semantic opportunities with a single Host-owned admission boundary."""

    def __init__(
        self,
        scope: Scope,
        now: float,
        parameters: AutonomyParameters = DEFAULT_AUTONOMY_PARAMETERS,
    ) -> None:
        self.state = State(scope=scope, now=now, last_sample_at=now)
        self.parameters = parameters
        self.body_changed = True
        self.progress_changed = True

    def set_parameters(self, parameters: AutonomyParameters) -> None:
        """Apply a validated profile to future sampling without changing persisted receipts."""
        self.parameters = parameters

    def _set(self, **updates: object) -> None:
        for name, value in updates.items():
            old = getattr(self.state, name)
            changed = not _same_fact(old, value)
            if changed:
                if name in HOT_FIELDS:
                    self.progress_changed = True
                else:
                    self.body_changed = True
        self.state = self.state.model_copy(update=updates)

    def _put(self, name: str, key: str, value: Any) -> None:
        mapping = getattr(self.state, name)
        if key not in mapping or not _same_fact(mapping[key], value):
            if name == "attentions":
                self.progress_changed = True
            if name != "attentions" or key not in mapping:
                self.body_changed = True
        mapping[key] = value

    def _pop(self, name: str, key: str, *default: Any) -> Any:
        mapping = getattr(self.state, name)
        if key in mapping:
            self.body_changed = True
        return mapping.pop(key, *default)

    def set_state_entry(self, name: str, key: str, value: Any) -> None:
        """Apply a Host-owned fact at the original state mutation boundary."""
        self._put(name, key, value)

    def remove_state_entry(self, name: str, key: str) -> None:
        self._pop(name, key, None)

    def take_changes(self) -> tuple[bool, bool]:
        changes = self.progress_changed, self.body_changed
        self.progress_changed = self.body_changed = False
        return changes

    def _resolved_event(self, event: ScopedEvent) -> ScopedEvent:
        observation = self.state.observations.get(event.ref.event_id)
        unit = observation.resolved_unit if observation else None
        return (
            event.model_copy(
                update={"thread": unit.thread, "target": unit.target, "unit_ambiguous": False}
            )
            if unit
            else event
        )

    def resolved_unit(self, event: ScopedEvent) -> ScopedEvent | None:
        """Return the stored, current Host unit once resolved; never invent an ID."""
        if event.scope != self.state.scope or not self._valid(event.ref):
            return None
        stored = self._resolved_event(self.state.events[event.ref.event_id])
        return None if stored.unit_ambiguous else stored

    def observe_committed_event(self, event: ScopedEvent) -> bool:
        if (
            event.scope != self.state.scope
            or event.at < self.state.now - 600
            or event.at <= self.state.replay_after
        ):
            return False
        seen = self.state.seen.get(event.ref.event_id)
        old = self.state.events.get(event.ref.event_id)
        if seen and seen.revision >= event.ref.revision:
            if (
                seen.revision == event.ref.revision
                and seen.source_order is None
                and event.source_order is not None
                and old is not None
                and old.model_copy(update={"source_order": event.source_order}) == event
            ):
                # A Host reread can supply the original local order of an old
                # source. Keep paid events and snapshots byte-for-byte stable.
                self._put(
                    "seen",
                    event.ref.event_id,
                    seen.model_copy(update={"source_order": event.source_order}),
                )
            return False
        if seen is None and len(self.state.seen) >= 1024:
            self._make_participation_source_room()
            if len(self.state.seen) >= 1024:
                return False
        if old and old.ref.revision >= event.ref.revision:
            return False
        if event.ref.revision <= self.state.invalidated.get(event.ref.event_id, 0):
            return False
        if old:
            self.observe_source_change(old.ref)
        self._put("events", event.ref.event_id, event)
        self._put(
            "seen",
            event.ref.event_id,
            SeenSource(revision=event.ref.revision, at=event.at, source_order=event.source_order),
        )
        if event.kind == "human":
            self._set(last_human_at=max(self.state.last_human_at or 0, event.at))
        elif event.kind == "self":
            self._set(last_self_message_at=max(self.state.last_self_message_at or 0, event.at))
        self._prune()
        return True

    def observe_source_change(
        self, invalid: SourceRef, *, actor_identity: ActorIdentity = original_actor
    ) -> None:
        key = invalid.event_id
        if key not in self.state.seen and len(self.state.seen) >= 1024:
            raise ValueError("source_tombstone_capacity")
        old_seen = self.state.seen.get(key)
        if self.state.belief_baselines and (
            old_seen is None
            or old_seen.at <= self.state.replay_after
            or old_seen.revision > invalid.revision
        ):
            # Detailed provenance before the finite replay boundary has been retired.
            # An explicit old-source withdrawal cannot be reconstructed safely: forget
            # derived baselines conservatively, retaining independent stops/receipts.
            self._set(
                belief_baselines={
                    unit: baseline.model_copy(update={"value": dynamics.IDLE})
                    for unit, baseline in self.state.belief_baselines.items()
                },
                baseline_invalidations=self.state.baseline_invalidations + 1,
            )
        if old_seen is None or old_seen.at <= self.state.replay_after:
            # Retired human activity cannot be subtracted exactly. Forget its
            # social-context contribution rather than retaining a ghost motive.
            baselines = dict(self.state.trace_baselines)
            baselines["social_context"] = TraceBaseline(
                at=max(0.0, self.state.replay_after), value=0.0
            )
            baselines["human_activity"] = TraceBaseline(
                at=max(0.0, self.state.replay_after), value=0.0
            )
            self._set(trace_baselines=baselines)
        self._put(
            "seen",
            key,
            SeenSource(
                revision=max(invalid.revision, old_seen.revision if old_seen else 0),
                at=old_seen.at if old_seen else self.state.now,
                source_order=old_seen.source_order if old_seen else None,
            ),
        )
        self._put("invalidated", key, max(invalid.revision, self.state.invalidated.get(key, 0)))
        current = self.state.events.get(key)
        if current is None or current.ref.revision <= invalid.revision:
            self._pop("receptions", key, None)
        if current and current.ref.revision <= invalid.revision:
            self._pop("events", key)
        for boundary_key, boundary in list(self.state.boundaries.items()):
            if any(
                ref.event_id == key and ref.revision <= invalid.revision
                for ref in (boundary.source, *boundary.dependencies)
            ):
                self._pop("boundaries", boundary_key)
            elif any(
                ref.event_id == key and ref.revision <= invalid.revision
                for ref in ((boundary.released_by,) if boundary.released_by else ())
                + boundary.release_dependencies
            ):
                self._put(
                    "boundaries",
                    boundary_key,
                    boundary.model_copy(update={"released_by": None, "release_dependencies": ()}),
                )
        for observation_key, observation in list(self.state.observations.items()):
            sources = (observation.snapshot.focus, *observation.snapshot.context)
            if observation.matching_self_anchor is not None:
                sources = (*sources, observation.matching_self_anchor)
            if any(ref.event_id == key and ref.revision <= invalid.revision for ref in sources):
                self._invalidate_observation(observation_key)
        for candidate_id, candidate in list(self.state.candidates.items()):
            if any(
                ref.event_id == key and ref.revision <= invalid.revision
                for ref in (
                    *candidate.support.covered,
                    candidate.support.basis,
                    *candidate.support.dependencies,
                )
            ):
                self._pop("candidates", candidate_id)
        self._refresh_releases(actor_identity=actor_identity)

        self._prune_participation()

    def _invalidate_observation(self, key: str) -> None:
        for boundary_key, boundary in list(self.state.boundaries.items()):
            if boundary.released_by and boundary.released_by.event_id == key:
                self._put(
                    "boundaries",
                    boundary_key,
                    boundary.model_copy(update={"released_by": None, "release_dependencies": ()}),
                )
        observation = self._pop("observations", key, None)
        self._pop("receptions", key, None)
        if observation is None:
            return
        self._pop("boundaries", key, None)
        for candidate_id, candidate in list(self.state.candidates.items()):
            if candidate.support.observation_id == observation.observation_id:
                self._pop("candidates", candidate_id)
                self._pop("attentions", candidate_id, None)

    def _valid(self, ref: SourceRef) -> bool:
        event = self.state.events.get(ref.event_id)
        return bool(
            event
            and event.ref == ref
            and ref.revision > self.state.invalidated.get(ref.event_id, 0)
        )

    def _source_order(self, ref: SourceRef) -> int | None:
        seen = self.state.seen.get(ref.event_id)
        return seen.source_order if seen is not None and seen.revision == ref.revision else None

    def apply_semantic_observation(
        self, observation: Observation, *, actor_identity: ActorIdentity = original_actor
    ) -> bool:
        if observation.snapshot.kind is not CandidateKind.CONVERSATION:
            return False
        snap = observation.snapshot
        event = snap.focus
        key = event.ref.event_id
        old = self.state.observations.get(key)
        if (
            snap.scope != self.state.scope
            or not self._valid(event.ref)
            or self.state.events.get(key) != event
            or any(not self._valid(e.ref) for e in snap.context)
            or event.at < self.state.now - 600
            or event.at <= self.state.replay_after
            or snap.issued_at > observation.received_at
            or observation.rubric_revision != REVISION
            or (old and old.snapshot.sequence >= snap.sequence)
        ):
            return False
        for dimension, answer in observation.answers.items():
            options = (
                {option.key for option in event.unit_options} | {"unknown"}
                if dimension == "unit_selection" and event.unit_ambiguous
                else set(CRITERIA.get(dimension, {}))
            )
            if not options or set(answer.probabilities) != options:
                return False
        if event.unit_ambiguous:
            selection = observation.answers.get("unit_selection")
            selected = observation.resolved_unit
            if selected is not None and (
                selected not in event.unit_options
                or not _dimension_known(selection)
                or selection.choice != selected.key
                or sum(
                    p == max(selection.probabilities.values())
                    for p in selection.probabilities.values()
                )
                != 1
            ):
                return False
        elif observation.resolved_unit is not None:
            return False
        # Replace interpretation by source, not append another social event.
        self._invalidate_observation(key)
        act = observation.answers.get("interaction_mark")
        floor = observation.answers.get("floor_state")
        semantic_new = None
        if (
            event.kind == "human"
            and event.unit_ambiguous
            and observation.resolved_unit is None
            and (selection := observation.answers.get("unit_selection")) is not None
            and selection.choice == "unknown"
            and _dimension_known(act)
            and act.choice in {"invite_yuki", "extend_yuki"}
            and _dimension_known(floor)
            and floor.choice == "yuki"
        ):
            # Addressed speech does not require choosing an earlier topic.
            # Only use the Host's fresh unit; never invent a target or identity.
            semantic_new = next(
                (
                    option
                    for option in event.unit_options
                    if option.key == "new"
                    and option.thread == event.thread
                    and option.target == event.author
                    and option.self_anchor is None
                ),
                None,
            )
        effective_unit = observation.resolved_unit or semantic_new
        if effective_unit is not None:
            unit = effective_unit
            event = event.model_copy(
                update={"thread": unit.thread, "target": unit.target, "unit_ambiguous": False}
            )
        anchor_ref = event.reply_to
        if anchor_ref is None and observation.resolved_unit is not None:
            interaction = observation.answers.get("interaction_mark")
            if _dimension_known(interaction) and interaction.p("extend_yuki") >= 0.5:
                anchor_ref = observation.resolved_unit.self_anchor
        anchor = self.state.events.get(anchor_ref.event_id) if anchor_ref else None
        matching_anchor = (
            anchor_ref
            if (
                anchor is not None
                and anchor.ref == anchor_ref
                and anchor.kind == "self"
                and anchor.thread == event.thread
                and anchor.target in {event.target, "group"}
                and anchor.at <= event.at
                and (
                    anchor_ref == event.reply_to
                    or any(context.ref == anchor_ref for context in snap.context)
                )
            )
            else None
        )
        self._put(
            "observations",
            key,
            StoredObservation.from_observation(observation).model_copy(
                update={
                    "matching_self_anchor": matching_anchor,
                    "resolved_unit": effective_unit,
                    "unit_resolution": "semantic_new"
                    if semantic_new is not None
                    else ("observer" if observation.resolved_unit is not None else None),
                },
            ),
        )
        self._record_unit_reviews(observation, event, actor_identity=actor_identity)
        self._record_reception(key, self.state.observations[key])
        self._pop("candidates", key, None)
        self._pop("boundaries", key, None)
        self._prune_participation()
        if event.unit_ambiguous:
            return True
        info = observation.answers.get("information_state")
        boundary = observation.answers.get("boundary_scope")
        if (
            _dimension_known(act)
            and _dimension_known(boundary)
            and act.p("close_topic") + act.p("ask_yuki_stop") >= 0.5
        ):
            self._put(
                "boundaries",
                key,
                Boundary(
                    source=event.ref,
                    thread=event.thread,
                    target=event.target,
                    group_wide=boundary.p("group_thread") >= 0.5,
                    dependencies=tuple(e.ref for e in snap.context),
                    author=event.author,
                    explicit_stop=act.p("ask_yuki_stop") >= 0.5,
                ),
            )
        self._refresh_releases(actor_identity=actor_identity)
        if self.state.consumed.get(key, 0) >= event.ref.revision:
            return True
        if not (_dimension_known(act) and _dimension_known(floor)):
            return True
        addressed = act.choice in {"invite_yuki", "extend_yuki"} and floor.choice == "yuki"
        open_floor = act.choice == "open_group" and floor.choice in {"open", "yuki"}
        if not (addressed or open_floor):
            return True
        if act.p("close_topic") + act.p("ask_yuki_stop") >= 0.5:
            return True
        if act.p("invite_yuki") + act.p("extend_yuki") >= 0.5 and floor.p("other") >= 0.5:
            return True  # conflicting dimensions, no multiplied confidence
        value = 0.5 * (act.p("invite_yuki") + act.p("extend_yuki") + 0.6 * act.p("open_group"))
        if _dimension_known(info):
            value += 0.5 * (info.p("new") + 0.8 * info.p("refine"))
        if snap.kind == CandidateKind.CONVERSATION and observation.received_at >= event.at + 90:
            return True  # A stale result remains evidence but cannot revive an old invitation.
        # Bound total source age while allowing the evaluated opportunity a real
        # response window. This is not a permission to reply to old conversation.
        until = (
            min(event.at + 150, observation.received_at + 75)
            if snap.kind == CandidateKind.CONVERSATION
            else event.at + 180
        )
        support = Support(
            kind="observed",
            scope=self.state.scope,
            thread=event.thread,
            target=event.target,
            observation_id=observation.observation_id,
            covered=(event.ref,),
            basis=event.ref,
            dependencies=tuple(e.ref for e in snap.context),
            issued_at=observation.received_at,
            valid_until=until,
        )
        self._put(
            "candidates",
            key,
            Candidate(
                event=event,
                kind=snap.kind,
                support=support,
                value=Estimate(
                    value=value, observation_id=observation.observation_id, valid_until=until
                ),
                floor=Estimate(
                    value=floor.p("yuki") + 0.7 * floor.p("open"),
                    observation_id=observation.observation_id,
                    valid_until=until,
                ),
            ),
        )
        self._prune()
        return True

    def _participation_checkpoint(self) -> ParticipationCheckpoint:
        return self._read_participation_checkpoint() or ParticipationCheckpoint()

    def _read_participation_checkpoint(self) -> ParticipationCheckpoint | None:
        raw = self.state.host_checkpoint.get(CHECKPOINT_KEY)
        if raw is None:
            return ParticipationCheckpoint()
        try:
            return ParticipationCheckpoint.model_validate(raw)
        except ValueError:
            # Unknown extensions are retained, but never used as relationship evidence.
            return None

    def _binding_valid(
        self, binding: UnitBinding, checkpoint: ParticipationCheckpoint | None
    ) -> bool:
        if binding.scope != self.state.scope:
            return False
        retired = {r for r in binding.basis if not self._valid(r)}
        if not retired:
            return True
        known = self._known_participation_refs(binding, checkpoint)
        return all(self._retired_participation_ref_valid(ref, known) for ref in retired)

    def _unit_expression_ids(self, checkpoint: ParticipationCheckpoint) -> set[str]:
        """One existing logical message receipt per established unit, without its body."""
        latest: dict[str, tuple[float, str]] = {}
        for effect_id, association in checkpoint.expressions.items():
            key = self._unit_key(association.binding.unit.thread, association.binding.unit.target)
            unit = checkpoint.units.get(key)
            record = self.state.effects.get(effect_id)
            if (
                unit is not None
                and self._bound_to_unit(association.binding, unit)
                and record is not None
                and record.effect.kind == "message"
            ):
                latest[key] = max(latest.get(key, (-1.0, "")), (record.effect.at, effect_id))
        return {effect_id for _, effect_id in latest.values()}

    def _participation_refs(self, checkpoint: ParticipationCheckpoint) -> set[SourceRef]:
        refs = {
            ref
            for unit in checkpoint.units.values()
            for ref in (
                *unit.binding.basis,
                *((unit.input_ref,) if unit.input_ref else ()),
                *unit.hint_basis,
                *unit.anchors,
                *(unit.review.basis if unit.review else ()),
                *unit.continuation_refs,
            )
        }
        for effect_id in self._unit_expression_ids(checkpoint):
            refs.update(checkpoint.expressions[effect_id].binding.basis)
        return refs

    def _unit_ref_valid(self, ref: SourceRef, unit: UnitState) -> bool:
        if self._valid(ref):
            return True
        return self._retired_participation_ref_valid(ref, set(unit.retired_refs))

    def _retired_participation_ref_valid(self, ref: SourceRef, known: set[SourceRef]) -> bool:
        seen = self.state.seen.get(ref.event_id)
        return bool(
            seen
            and seen.revision == ref.revision
            and ref.revision > self.state.invalidated.get(ref.event_id, 0)
            and ref in known
        )

    def _known_participation_refs(
        self, binding: UnitBinding, checkpoint: ParticipationCheckpoint | None
    ) -> set[SourceRef]:
        key = self._unit_key(binding.unit.thread, binding.unit.target)
        if checkpoint is not None:
            unit = checkpoint.units.get(key)
        else:
            # The existing single-unit continuation reader can recognize its
            # original proof even when it cannot write an extended namespace.
            raw = self.state.host_checkpoint.get(CHECKPOINT_KEY)
            if not isinstance(raw, dict) or raw.get("version") != 1:
                return set()
            units = raw.get("units")
            if not isinstance(units, dict) or key not in units:
                return set()
            try:
                unit = UnitState.model_validate(units[key])
            except ValueError:
                return set()
        return (
            set(unit.retired_refs)
            if unit is not None and self._same_unit_actor(unit.binding, binding)
            else set()
        )

    def _unit_closed(self, unit: ParticipationUnit, actor_identity: ActorIdentity) -> bool:
        return any(
            b.released_by is None
            and b.thread == unit.thread
            and (actor_identity(b.target) == actor_identity(unit.target) or b.group_wide)
            for b in self.state.boundaries.values()
        )

    def _save_participation(self, checkpoint: ParticipationCheckpoint) -> None:
        self._set(
            host_checkpoint={
                **self.state.host_checkpoint,
                CHECKPOINT_KEY: checkpoint.model_dump(mode="json"),
            }
        )

    @staticmethod
    def _same_unit_actor(left: UnitBinding, right: UnitBinding) -> bool:
        return left.scope == right.scope and left.unit == right.unit and left.actor == right.actor

    @classmethod
    def _bound_to_unit(cls, binding: UnitBinding, unit: UnitState) -> bool:
        return cls._same_unit_actor(binding, unit.binding) and set(unit.binding.basis).issubset(
            binding.basis
        )

    def observe_unit_input(
        self, binding: UnitBinding, event_ref: SourceRef, *, continuation: bool | None = False
    ) -> bool:
        """Replay one real admission without buying or spending semantic review."""
        event = self.state.events.get(event_ref.event_id)
        checkpoint = self._read_participation_checkpoint()
        if (
            checkpoint is None
            or not self._binding_valid(binding, checkpoint)
            or event is None
            or event.ref != event_ref
            or not self._valid(event_ref)
            or event.kind != "human"
            or binding.actor != event.author
        ):
            return False
        key = self._unit_key(binding.unit.thread, binding.unit.target)
        old = checkpoint.units.get(key)
        if old is not None and not self._same_unit_actor(old.binding, binding):
            old = None
        if old is not None and old.input_ref == event_ref:
            return self._same_unit_actor(old.binding, binding)
        consumed = self.state.consumed.get(event_ref.event_id, 0) >= event_ref.revision
        if old is None and consumed:
            # An old replay cannot re-establish a relationship retired by its source.
            return False
        refs = old.continuation_refs if old else ()
        if continuation and not consumed:
            refs = (*refs, event_ref)
        changes: dict[str, object] = {
            "last_at": max(event.at, old.last_at if old else 0),
            "continuation_refs": refs,
        }
        # A/B/C hydration is a replay, not permission to rewind the latest input.
        previous = self.state.seen.get(old.input_ref.event_id) if old and old.input_ref else None
        event_order = self._source_order(event_ref)
        if (
            previous is None
            or event.at > previous.at
            or (
                event.at == previous.at
                and (
                    event_order is not None
                    and previous.source_order is not None
                    and event_order >= previous.source_order
                )
            )
        ):
            changes["input_ref"] = event_ref
        unit = old or UnitState(binding=binding, last_at=event.at)
        if continuation is False and not consumed:
            # An independent admission owns its new source, but cannot erase quiet.
            changes.update(binding=binding, review=None, continuation_refs=())
        checkpoint.units[key] = unit.model_copy(update=changes)
        self._put(
            "consumed",
            event_ref.event_id,
            max(event_ref.revision, self.state.consumed.get(event_ref.event_id, 0)),
        )
        self._save_participation(checkpoint)
        self._prune_participation(checkpoint)
        return True

    def observe_unit_expression(
        self,
        binding: UnitBinding,
        run_ref: str,
        effect: Effect,
        *,
        anchor: SourceRef | None = None,
    ) -> bool:
        """Associate a confirmed logical expression without altering its transport target."""
        checkpoint = self._read_participation_checkpoint()
        if (
            checkpoint is None
            or not self._binding_valid(binding, checkpoint)
            or effect.kind != "message"
            or not run_ref
            or (
                anchor is not None
                and (not self._valid(anchor) or self.state.events[anchor.event_id].kind != "self")
            )
        ):
            return False
        association = ExpressionBinding(binding=binding, anchor=anchor)
        old = checkpoint.expressions.get(effect.effect_id)
        if old is not None:
            record = self.state.effects.get(effect.effect_id)
            if (
                old.binding != binding
                or record is None
                or record.effect != effect
                or record.run_ref != run_ref
            ):
                return False
            if anchor is not None:
                key = self._unit_key(binding.unit.thread, binding.unit.target)
                unit = checkpoint.units.get(key)
                if (
                    unit is not None
                    and self._bound_to_unit(binding, unit)
                    and anchor not in unit.anchors
                ):
                    checkpoint.units[key] = unit.model_copy(
                        update={"anchors": (*unit.anchors, anchor)}
                    )
                    self._save_participation(checkpoint)
            return True
        record = self.state.effects.get(effect.effect_id)
        if record is not None and (record.run_ref != run_ref or record.effect != effect):
            return False
        if record is None:
            self.observe_committed_effect(run_ref, effect)
        checkpoint.expressions[effect.effect_id] = association
        key = self._unit_key(binding.unit.thread, binding.unit.target)
        unit = checkpoint.units.get(key)
        if unit is None or not self._bound_to_unit(binding, unit):
            # A late transport receipt remains factual, but cannot re-establish a
            # unit retired by a real interpretation change or resource eviction.
            self._save_participation(checkpoint)
            return True
        anchors = tuple(
            dict.fromkeys((*(unit.anchors if unit else ()), *((anchor,) if anchor else ())))
        )
        checkpoint.units[key] = unit.model_copy(
            update={"last_at": max(effect.at, unit.last_at), "anchors": anchors}
        )
        self._save_participation(checkpoint)
        self._prune_participation(checkpoint)
        return True

    def observe_unit_hint(
        self,
        binding: UnitBinding,
        report: SelfReport,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> bool:
        """Optional own intent from a Host-bound Main response; it is not a user stop."""
        checkpoint = self._read_participation_checkpoint()
        if (
            checkpoint is None
            or not self._binding_valid(binding, checkpoint)
            or report.delta.engage is None
        ):
            return False
        key = self._unit_key(binding.unit.thread, binding.unit.target)
        old = checkpoint.units.get(key)
        if old is None or not self._bound_to_unit(binding, old):
            return False
        if report.source is None:
            source_ref = self._hint_source(report, binding.basis)
            if source_ref is not None:
                report = report.model_copy(update={"source": source_ref})
        if report.source is not None:
            source = self.state.seen.get(report.source.event_id)
            if (
                report.source not in binding.basis
                or source is None
                or source.revision != report.source.revision
                or source.at != report.at
            ):
                return False
        if old.hint:
            previous = old.hint
            previous_source = self._hint_source(previous, old.hint_basis)
            order = self._source_order(report.source) if report.source else None
            previous_order = self._source_order(previous_source) if previous_source else None
            if (
                report.response_id == previous.response_id
                or report.at < previous.at
                or (
                    report.at == previous.at
                    and order is not None
                    and previous_order is not None
                    and order < previous_order
                )
                or (report.run_ref == previous.run_ref and report.sequence <= previous.sequence)
            ):
                return False
        checkpoint.units[key] = old.model_copy(
            update={
                "last_at": max(report.at, old.last_at),
                "hint": report,
                "hint_basis": binding.basis,
            }
        )
        self._save_participation(checkpoint)
        if report.delta.engage == "quiet":
            self._retire_superseded_invitations(checkpoint, actor_identity)
        self._prune_participation(checkpoint)
        return True

    def _hint_source(self, report: SelfReport, basis: tuple[SourceRef, ...]) -> SourceRef | None:
        if report.source is not None:
            return report.source
        # Old reports did not carry a source. Only a unique original basis
        # source at their real source time can recover that association.
        sources = tuple(
            ref
            for ref in basis
            if (seen := self.state.seen.get(ref.event_id)) is not None
            and seen.revision == ref.revision
            and seen.at == report.at
        )
        return sources[0] if len(sources) == 1 else None

    def _prune_participation(
        self,
        checkpoint: ParticipationCheckpoint | None = None,
        *,
        previous: ParticipationCheckpoint | None = None,
        normalize: bool = False,
    ) -> None:
        if checkpoint is None:
            checkpoint = self._read_participation_checkpoint()
        if checkpoint is None:
            return
        expressions = {
            k: v
            for k, v in checkpoint.expressions.items()
            if k in self.state.effects and self._binding_valid(v.binding, checkpoint)
        }
        units = {
            k: v.model_copy(
                update={
                    "anchors": tuple(r for r in v.anchors if self._unit_ref_valid(r, v)),
                    "hint": v.hint
                    if all(self._unit_ref_valid(r, v) for r in v.hint_basis)
                    else None,
                }
            )
            for k, v in checkpoint.units.items()
            if self._binding_valid(v.binding, checkpoint) and self._unit_interpretation_valid(v)
        }
        # Use the existing bounded discussion resource policy, not a conversation expiry rule.
        if len(units) > 64:
            ordered = sorted(units, key=lambda k: (-units[k].last_at, k))
            units = {k: units[k] for k in ordered[:64]}
        updated = checkpoint.model_copy(update={"units": units, "expressions": expressions})
        if CHECKPOINT_KEY in self.state.host_checkpoint and (
            normalize or updated != (previous if previous is not None else checkpoint)
        ):
            self._save_participation(updated)

    def _retire_participation(
        self, boundary: float, checkpoint: ParticipationCheckpoint | None
    ) -> ParticipationCheckpoint:
        """Fold only already verified unit identities before dropping raw interpretation."""
        if checkpoint is None:
            return ParticipationCheckpoint()
        units = {}
        expressions = self._unit_expression_ids(checkpoint)
        expression_refs: dict[str, set[SourceRef]] = {}
        for effect_id in expressions:
            association = checkpoint.expressions[effect_id]
            key = self._unit_key(association.binding.unit.thread, association.binding.unit.target)
            original = checkpoint.units.get(key)
            if original is not None and self._same_unit_actor(
                association.binding, original.binding
            ):
                expression_refs.setdefault(key, set()).update(association.binding.basis)
        for key, unit in checkpoint.units.items():
            if not self._binding_valid(
                unit.binding, checkpoint
            ) or not self._unit_interpretation_valid(unit):
                continue
            retired_anchors = [
                r
                for r in unit.anchors
                if self._unit_ref_valid(r, unit)
                and (seen := self.state.seen.get(r.event_id)) is not None
                and seen.at <= boundary
            ]
            newest = max(
                retired_anchors,
                key=lambda r: (self.state.seen[r.event_id].at, r.event_id),
                default=None,
            )
            anchors = tuple(
                r
                for r in unit.anchors
                if self._valid(r) and self.state.events[r.event_id].at > boundary
            ) + ((newest,) if newest else ())
            refs = {
                *unit.binding.basis,
                *((unit.input_ref,) if unit.input_ref else ()),
                *unit.hint_basis,
                *anchors,
                *(unit.review.basis if unit.review else ()),
                *unit.continuation_refs,
            }
            refs.update(expression_refs.get(key, ()))
            retired = tuple(
                sorted(
                    (
                        r
                        for r in refs
                        if self._unit_ref_valid(r, unit)
                        and (seen := self.state.seen.get(r.event_id)) is not None
                        and (r in unit.retired_refs or seen.at <= boundary)
                    ),
                    key=lambda r: (r.event_id, r.revision),
                )
            )
            units[key] = unit.model_copy(update={"anchors": anchors, "retired_refs": retired})
        ordered = sorted(units, key=lambda k: (-units[k].last_at, k))
        return checkpoint.model_copy(update={"units": {k: units[k] for k in ordered[:64]}})

    def _make_participation_source_room(self) -> None:
        # Retire derived units under the existing fingerprint resource bound. Never
        # revive raw sources or block an execution merely to retain a relationship.
        self._prune()
        while len(self.state.seen) >= 1024:
            checkpoint = self._read_participation_checkpoint()
            if checkpoint is None or not checkpoint.units:
                return
            oldest = min(checkpoint.units, key=lambda k: (checkpoint.units[k].last_at, k))
            checkpoint.units.pop(oldest)
            self._save_participation(checkpoint)
            self._prune()

    def participation_actor_ids(self) -> frozenset[str]:
        """Original actors/targets for Host identity lookup, without deriving eligibility."""
        checkpoint = self._participation_checkpoint()
        return frozenset(
            actor
            for unit in checkpoint.units.values()
            if unit.binding.actor != "SELF"
            for actor in (unit.binding.actor, unit.binding.unit.target)
        )

    def participating_units(
        self, now: float, *, actor_identity: ActorIdentity = original_actor
    ) -> tuple[UnitParticipation, ...]:
        """Read existing ordinary and SELF units; never queue, consume, or advance."""
        checkpoint = self._participation_checkpoint()
        return self._current_units(now, checkpoint, actor_identity)

    def _source_before(
        self, left_at: float, left: SourceRef | None, right_at: float, right: SourceRef | None
    ) -> bool:
        if left_at != right_at:
            return left_at < right_at
        left_order = self._source_order(left) if left else None
        right_order = self._source_order(right) if right else None
        return left_order is not None and right_order is not None and left_order < right_order

    def _current_units(
        self, now: float, checkpoint: ParticipationCheckpoint, actor_identity: ActorIdentity
    ) -> tuple[UnitParticipation, ...]:
        """Interpret current identity over original facts, without rewriting their owners."""
        groups: dict[tuple[str, str, str], list[UnitParticipation]] = {}
        for unit in self._participating_units(now, checkpoint, actor_identity=actor_identity):
            identity = (
                unit.unit.target
                if unit.binding.actor == "SELF"
                else actor_identity(unit.unit.target)
            )
            groups.setdefault(
                (unit.unit.thread, identity, unit.binding.actor == "SELF" and "SELF" or "human"), []
            ).append(unit)
        result = []
        for values in groups.values():
            if len(values) == 1:
                result.append(values[0])
                continue
            originals = [
                checkpoint.units[self._unit_key(v.unit.thread, v.unit.target)] for v in values
            ]
            hints = [
                (unit.hint, self._hint_source(unit.hint, unit.hint_basis))
                for unit in originals
                if unit.hint is not None
                and all(self._unit_ref_valid(ref, unit) for ref in unit.hint_basis)
            ]
            latest_hints = [
                hint
                for hint, source in hints
                if not any(
                    self._source_before(hint.at, source, other.at, other_source)
                    or (hint.run_ref == other.run_ref and hint.sequence < other.sequence)
                    for other, other_source in hints
                )
            ]
            intents = {hint.delta.engage for hint in latest_hints}
            reviews = [unit.review for unit in originals if unit.review is not None]
            latest_reviews = [
                review
                for review in reviews
                if not any(
                    self._source_before(review.at, review.source, other.at, other.source)
                    or (review.source == other.source and review.sequence < other.sequence)
                    for other in reviews
                )
            ]
            verdicts = {
                self._review_verdict(review, values[0].unit, actor_identity=actor_identity)
                for review in latest_reviews
            }
            result.append(
                values[0].model_copy(
                    update={
                        "anchors": tuple(
                            dict.fromkeys(ref for value in values for ref in value.anchors)
                        ),
                        "expressed": any(value.expressed for value in values),
                        "closed": any(value.closed for value in values),
                        "engage": next(iter(intents)) if len(intents) == 1 else None,
                        "semantic_continuation": len(intents) <= 1
                        and (not verdicts or verdicts == {True}),
                    }
                )
            )
        return tuple(result)

    def _participating_units(
        self,
        now: float,
        checkpoint: ParticipationCheckpoint,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> tuple[UnitParticipation, ...]:
        units = {
            k: v
            for k, v in checkpoint.units.items()
            if self._binding_valid(v.binding, checkpoint) and self._unit_interpretation_valid(v)
        }
        expressed = {
            self._unit_key(v.binding.unit.thread, v.binding.unit.target)
            for effect_id, v in checkpoint.expressions.items()
            if self._binding_valid(v.binding, checkpoint)
            and (
                bound_unit := units.get(
                    self._unit_key(v.binding.unit.thread, v.binding.unit.target)
                )
            )
            is not None
            and self._bound_to_unit(v.binding, bound_unit)
            and (record := self.state.effects.get(effect_id)) is not None
            and record.effect.kind == "message"
            and record.effect.at <= now
        }
        anchors_by_run: dict[str, list[SourceRef]] = {}
        for key, value in self.state.outbound_anchors.items():
            event = self.state.events.get(key)
            if event is not None and self._valid(event.ref):
                anchors_by_run.setdefault(value.run_ref, []).append(event.ref)
        for record in self.state.effects.values():
            proposal = self.state.proposals.get(record.proposal_id)
            if proposal is None or record.effect.kind != "message" or record.effect.at > now:
                continue
            anchors = tuple(anchors_by_run.get(record.run_ref, ()))
            basis = tuple(r for r in proposal.sources if self._valid(r)) or anchors
            if not basis:
                continue
            unit = ParticipationUnit(thread=proposal.thread, target=proposal.target_hint)
            key = self._unit_key(unit.thread, unit.target)
            expressed.add(key)
            if key not in units:
                units[key] = UnitState(
                    binding=UnitBinding(
                        scope=self.state.scope, unit=unit, actor="SELF", basis=basis
                    ),
                    last_at=record.effect.at,
                    anchors=anchors,
                )
        closed_units = {
            (b.thread, actor_identity(b.target))
            for b in self.state.boundaries.values()
            if b.released_by is None
        }
        closed_threads = {
            b.thread
            for b in self.state.boundaries.values()
            if b.released_by is None and b.group_wide
        }
        return tuple(
            UnitParticipation(
                binding=v.binding,
                last_at=v.last_at,
                anchors=tuple(r for r in v.anchors if self._unit_ref_valid(r, v)),
                engage=v.hint.delta.engage
                if v.hint and all(self._unit_ref_valid(r, v) for r in v.hint_basis)
                else None,
                expressed=self._unit_key(v.binding.unit.thread, v.binding.unit.target) in expressed,
                closed=(v.binding.unit.thread, actor_identity(v.binding.unit.target))
                in closed_units
                or v.binding.unit.thread in closed_threads,
                semantic_continuation=self._semantic_continuation(v, actor_identity=actor_identity),
            )
            for v in sorted(units.values(), key=lambda v: (-v.last_at, v.binding.unit.thread))
            if v.last_at <= now
        )

    def _unit_interpretation_valid(self, unit: UnitState) -> bool:
        ref = unit.input_ref
        if ref is None:
            return True
        if not self._unit_ref_valid(ref, unit):
            return False
        # Interpretations qualify continuation, not the existence of real inputs,
        # own quiet reports or transport receipts. Keep those independent facts.
        return True

    def participation_view(
        self,
        event: ScopedEvent,
        now: float,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> ParticipationView:
        """Relationship candidates are distinct from the focus's observed resolution."""
        return self._participation_view(
            event,
            now,
            self._participation_checkpoint(),
            actor_identity=actor_identity,
        )

    def _participation_view(
        self,
        event: ScopedEvent,
        now: float,
        checkpoint: ParticipationCheckpoint,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> ParticipationView:
        valid = event.scope == self.state.scope and self._valid(event.ref)
        if not valid:
            return ParticipationView(
                scope=self.state.scope, event_ref=event.ref, source_valid=False
            )
        event = self.state.events[event.ref.event_id]
        observed = self.state.observations.get(event.ref.event_id)
        resolved = self._resolved_event(event)
        current_resolved = observed is not None and not resolved.unit_ambiguous
        units = self._current_units(now, checkpoint, actor_identity)
        candidates = tuple(
            u
            for u in units
            if not u.closed
            and (
                actor_identity(event.author) == actor_identity(u.unit.target)
                or event.reply_to in u.anchors
            )
        )
        personal = tuple(
            u
            for u in candidates
            if actor_identity(u.unit.target) == actor_identity(event.author)
            and (u.expressed or u.engage in {"join", "stay"})
        )
        explicit = tuple(u for u in personal if event.reply_to and event.reply_to in u.anchors)
        matches = explicit or personal
        matches = tuple(u for u in matches if u.engage != "quiet")
        matches = tuple(u for u in matches if u.semantic_continuation)
        if current_resolved:
            matches = tuple(u for u in matches if u.unit.thread == resolved.thread)
        matched = matches[0] if len(matches) == 1 else None
        current = None
        if current_resolved:
            unit = ParticipationUnit(thread=resolved.thread, target=resolved.target)
            current = UnitParticipation(
                binding=UnitBinding(
                    scope=self.state.scope, unit=unit, actor=event.author, basis=(event.ref,)
                ),
                last_at=event.at,
                closed=self._unit_closed(unit, actor_identity),
            )
        candidate = self.state.candidates.get(event.ref.event_id)
        addressed = bool(
            candidate
            and candidate.support.kind == "observed"
            and self._addressed(candidate)
            and any(
                candidate in members
                for members in self._eligible_groups(now, actor_identity=actor_identity).values()
            )
            and not self._invitation_superseded(candidate.event, checkpoint, actor_identity)
        )
        return ParticipationView(
            scope=self.state.scope,
            event_ref=event.ref,
            source_valid=True,
            current_resolved=current_resolved,
            current_unit=current,
            addressed=addressed,
            matched_unit=matched,
            candidates=candidates,
            ambiguous=len(matches) > 1,
            needs_observation=not current_resolved,
        )

    def _invitation_superseded(
        self,
        event: ScopedEvent,
        checkpoint: ParticipationCheckpoint,
        actor_identity: ActorIdentity,
    ) -> bool:
        """A newer real brake retires the old opportunity, not a future invitation."""
        for unit in checkpoint.units.values():
            if (
                unit.binding.unit.thread != event.thread
                or actor_identity(unit.binding.unit.target) != actor_identity(event.target)
                or not self._binding_valid(unit.binding, checkpoint)
            ):
                continue
            review = unit.review
            if (
                review is not None
                and self._review_verdict(review, unit.binding.unit, actor_identity=actor_identity)
                is False
                and all(self._unit_ref_valid(ref, unit) for ref in review.basis)
                and self._source_before(event.at, event.ref, review.at, review.source)
            ):
                return True
            hint = unit.hint
            if (
                hint is not None
                and hint.delta.engage == "quiet"
                and all(self._unit_ref_valid(ref, unit) for ref in unit.hint_basis)
                and (
                    event.ref in unit.hint_basis
                    or self._source_before(
                        event.at, event.ref, hint.at, self._hint_source(hint, unit.hint_basis)
                    )
                )
            ):
                return True
        return False

    def _retire_superseded_invitations(
        self, checkpoint: ParticipationCheckpoint, actor_identity: ActorIdentity
    ) -> None:
        for key, candidate in tuple(self.state.candidates.items()):
            if self._invitation_superseded(candidate.event, checkpoint, actor_identity):
                self._pop("candidates", key)

    def continuation_current(
        self,
        event_ref: SourceRef,
        binding: UnitBinding,
        now: float,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> bool:
        """Recheck an unadmitted opportunity at the original Host boundary; no I/O."""
        event = self.state.events.get(event_ref.event_id)
        checkpoint = self._read_participation_checkpoint()
        if (
            event is None
            or event.ref != event_ref
            or binding.actor != event.author
            or not self._binding_valid(binding, checkpoint)
        ):
            return False
        matched = self._participation_view(
            event,
            now,
            checkpoint or ParticipationCheckpoint(),
            actor_identity=actor_identity,
        ).matched_unit
        return (
            matched is not None
            and matched.binding.scope == binding.scope
            and matched.unit.thread == binding.unit.thread
            and actor_identity(matched.unit.target) == actor_identity(binding.unit.target)
        )

    @staticmethod
    def _review_verdict(
        review: UnitSemanticReview,
        unit: ParticipationUnit,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> bool | None:
        same_unit = (
            review.resolved_unit is not None
            and review.resolved_unit.thread == unit.thread
            and actor_identity(review.resolved_unit.target) == actor_identity(unit.target)
        )
        if review.resolved_unit is not None and not same_unit:
            return False
        act, floor = review.interaction, review.floor
        if _dimension_known(act) and act.choice in {
            "other_exchange",
            "off_focus",
            "close_topic",
            "ask_yuki_stop",
        }:
            return False
        if _dimension_known(act) and _dimension_known(floor):
            if floor.choice == "other":
                return False
            if (
                act.choice in {"invite_yuki", "extend_yuki"}
                and floor.choice == "yuki"
                and same_unit
            ):
                return True
        return None

    def _semantic_continuation(
        self, unit: UnitState, *, actor_identity: ActorIdentity = original_actor
    ) -> bool:
        review = unit.review
        return review is None or (
            all(self._unit_ref_valid(ref, unit) for ref in review.basis)
            and self._review_verdict(review, unit.binding.unit, actor_identity=actor_identity)
            is True
        )

    def _record_unit_reviews(
        self,
        observation: Observation,
        event: ScopedEvent,
        *,
        actor_identity: ActorIdentity = original_actor,
    ) -> None:
        """Attach a real result to its original borrowed relationship, in source order."""
        checkpoint = self._read_participation_checkpoint()
        if checkpoint is None:
            return
        frozen = {
            self._unit_key(item.thread, item.target): item
            for item in observation.snapshot.unit_reviews
        }
        negative = False
        for key, unit in tuple(checkpoint.units.items()):
            origin = frozen.get(key)
            original_input = unit.input_ref == event.ref or event.ref in unit.continuation_refs
            resolved = (
                ParticipationUnit(thread=event.thread, target=event.target)
                if not event.unit_ambiguous
                else None
            )
            if origin is not None and (
                origin.actor != unit.binding.actor or origin.basis != unit.binding.basis
            ):
                continue
            if (
                origin is None
                and not original_input
                and not (
                    resolved is not None
                    and resolved.thread == unit.binding.unit.thread
                    and actor_identity(resolved.target) == actor_identity(unit.binding.unit.target)
                )
            ):
                continue
            if actor_identity(unit.binding.actor) != actor_identity(event.author):
                continue
            review = UnitSemanticReview(
                source=event.ref,
                at=event.at,
                sequence=observation.snapshot.sequence,
                observation_id=observation.observation_id,
                basis=tuple(
                    dict.fromkeys((event.ref, *(e.ref for e in observation.snapshot.context)))
                ),
                resolved_unit=resolved,
                interaction=observation.answers.get("interaction_mark"),
                floor=observation.answers.get("floor_state"),
                source_order=self._source_order(event.ref),
            )
            verdict = self._review_verdict(review, unit.binding.unit, actor_identity=actor_identity)
            if verdict is None:
                continue
            if verdict is False and not original_input and (origin is None or not origin.borrowed):
                continue
            previous = unit.review
            if previous is not None:
                if review.at < previous.at:
                    continue
                if review.at == previous.at:
                    previous_order = self._source_order(previous.source)
                    if review.source == previous.source:
                        if review.sequence <= previous.sequence:
                            continue
                    elif review.source_order is not None and previous_order is not None:
                        if review.source_order < previous_order:
                            continue
                    else:
                        positions = {ref: index for index, ref in enumerate(unit.continuation_refs)}
                        if (
                            review.source not in positions
                            or previous.source not in positions
                            or positions[review.source] <= positions[previous.source]
                        ):
                            continue
            checkpoint.units[key] = unit.model_copy(update={"review": review})
            negative = negative or verdict is False
        self._save_participation(checkpoint)
        if negative:
            self._retire_superseded_invitations(checkpoint, actor_identity)

    def freeze_observation(
        self, snapshot: Snapshot, *, actor_identity: ActorIdentity = original_actor
    ) -> Snapshot:
        """Freeze existing intent and original review positions once, before HTTP."""
        checkpoint = self._participation_checkpoint()
        focus = snapshot.focus
        keys = {(focus.thread, focus.target)} | {
            (option.thread, option.target) for option in focus.unit_options
        }
        current_keys = {(thread, actor_identity(target)) for thread, target in keys}
        units = []
        reviews = []
        matched = self._participation_view(
            focus,
            snapshot.issued_at,
            checkpoint,
            actor_identity=actor_identity,
        ).matched_unit
        for unit in checkpoint.units.values():
            if not self._binding_valid(
                unit.binding, checkpoint
            ) or not self._unit_interpretation_valid(unit):
                continue
            hint = unit.hint
            if (
                (unit.binding.unit.thread, actor_identity(unit.binding.unit.target)) in current_keys
                and hint is not None
                and hint.delta.engage is not None
                and hint.at <= snapshot.issued_at
                and all(self._unit_ref_valid(ref, unit) for ref in unit.hint_basis)
            ):
                units.append(
                    UnitIntent(
                        thread=unit.binding.unit.thread,
                        target=unit.binding.unit.target,
                        engage=hint.delta.engage,
                        at=hint.at,
                        run_ref=hint.run_ref,
                        sequence=hint.sequence,
                        response_id=hint.response_id,
                        basis=unit.hint_basis,
                    )
                )
            if (
                unit.input_ref == focus.ref
                or focus.ref in unit.continuation_refs
                or (
                    matched is not None
                    and matched.unit.thread == unit.binding.unit.thread
                    and actor_identity(matched.unit.target)
                    == actor_identity(unit.binding.unit.target)
                    and unit.binding.actor != "SELF"
                )
                or (
                    (unit.binding.unit.thread, actor_identity(unit.binding.unit.target))
                    in current_keys
                    and actor_identity(unit.binding.actor) == actor_identity(focus.author)
                )
            ):
                reviews.append(
                    UnitReviewSnapshot(
                        thread=unit.binding.unit.thread,
                        target=unit.binding.unit.target,
                        actor=unit.binding.actor,
                        basis=unit.binding.basis,
                        borrowed=unit.input_ref == focus.ref
                        or focus.ref in unit.continuation_refs
                        or (
                            matched is not None
                            and matched.unit.thread == unit.binding.unit.thread
                            and actor_identity(matched.unit.target)
                            == actor_identity(unit.binding.unit.target)
                            and unit.binding.actor != "SELF"
                        ),
                    )
                )
        report = self.state.engagement_report
        scope_intent = (
            EngageIntent(
                engage=report.delta.engage,
                at=report.at,
                run_ref=report.run_ref,
                sequence=report.sequence,
                response_id=report.response_id,
            )
            if report is not None
            and report.delta.engage is not None
            and report.at <= snapshot.issued_at
            else None
        )
        return snapshot.model_copy(
            update={
                "projection_revision": "semantic-state-v3",
                "participation": ParticipationContext(
                    scope_intent=scope_intent, units=tuple(units)
                ),
                "unit_reviews": tuple(reviews),
            }
        )

    @staticmethod
    def _unit_key(thread: str, target: str) -> str:
        return json.dumps((thread, target), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _familiarity_key(kind: CandidateKind, thread: str, target: str) -> str:
        return json.dumps((kind.value, thread, target), ensure_ascii=False, separators=(",", ":"))

    def _familiarity(self, kind: CandidateKind, thread: str, target: str, now: float) -> float:
        baseline = self.state.familiarities.get(self._familiarity_key(kind, thread, target))
        if baseline is None:
            return 0.0
        tau = 600 if kind == CandidateKind.CONVERSATION else 21600
        return baseline.value * math.exp(-max(0.0, now - baseline.at) / tau)

    def _consider(self, proposal: Proposal, now: float) -> None:
        key = self._familiarity_key(proposal.kind, proposal.thread, proposal.target_hint)
        previous = self.state.familiarities.get(key)
        now = max(now, previous.at) if previous else now
        old = self._familiarity(proposal.kind, proposal.thread, proposal.target_hint, now)
        familiarities = dict(self.state.familiarities)
        familiarities[key] = TraceBaseline(at=now, value=old + 0.45 * (1 - old))
        if len(familiarities) > 64:
            oldest = min(familiarities, key=lambda item: (familiarities[item].at, item))
            familiarities.pop(oldest)
        self._set(familiarities=familiarities)

    def _belief_actions(
        self, checkpoint: ParticipationCheckpoint
    ) -> dict[tuple[str, str], list[BeliefAction]]:
        """Group current facts once for this synchronous calculation, never across mutations."""
        actions: dict[tuple[str, str], list[BeliefAction]] = {}
        for obs in self.state.observations.values():
            event = self.state.events.get(obs.snapshot.focus.event_id)
            if event is not None:
                event = self._resolved_event(event)
            if (
                event is not None
                and self._valid(event.ref)
                and event.kind == "human"
                and not event.unit_ambiguous
            ):
                actions.setdefault((event.thread, event.target), []).append(
                    (event.at, "human", event.ref.event_id, obs)
                )
        for unit in checkpoint.units.values():
            ref = unit.input_ref
            event = self.state.events.get(ref.event_id) if ref is not None else None
            if (
                event is not None
                and event.ref == ref
                and event.ref.event_id not in self.state.observations
                and self._binding_valid(unit.binding, checkpoint)
            ):
                key = (unit.binding.unit.thread, unit.binding.unit.target)
                actions.setdefault(key, []).append((event.at, "human", event.ref.event_id, unit))
        for record in self.state.effects.values():
            if record.effect.kind != "message":
                continue
            proposal = self.state.proposals.get(record.proposal_id)
            association = checkpoint.expressions.get(record.effect.effect_id)
            units = (
                {(proposal.thread, target) for target in record.effect.actual_targets}
                if proposal is not None
                else set()
            )
            if association is not None and self._binding_valid(association.binding, checkpoint):
                units.add((association.binding.unit.thread, association.binding.unit.target))
            for key in units:
                actions.setdefault(key, []).append(
                    (record.effect.at, "self", record.effect.effect_id, record)
                )
        return actions

    def belief(self, thread: str, target: str, now: float) -> dynamics.Belief:
        actions = self._belief_actions(self._participation_checkpoint())
        return self._belief_from_actions(thread, target, now, actions.get((thread, target), []))

    def _belief_from_actions(
        self, thread: str, target: str, now: float, actions: list[BeliefAction]
    ) -> dynamics.Belief:
        baseline = self.state.belief_baselines.get(self._unit_key(thread, target))
        if baseline is not None and now < baseline.at:
            # Retired history is not a queryable historical reconstruction.
            return dynamics.IDLE
        b = baseline.value if baseline else dynamics.IDLE
        last = baseline.at if baseline else min((a[0] for a in actions), default=now)
        for at, _kind, _, value in sorted(actions, key=lambda x: x[:3]):
            if at > now or at <= self.state.replay_after:
                continue
            b = dynamics.decay(b, at - last)
            if isinstance(value, RecordedEffect):
                b = dynamics.self_expression(b)
            elif isinstance(value, UnitState):
                b = dynamics.admitted_input(b)
            else:
                act = value.answers.get("interaction_mark")
                if act:
                    b = dynamics.observe(
                        b,
                        act,
                        matched_self=value.matching_self_anchor is not None,
                    )
            last = at
        return dynamics.decay(b, now - last)

    def _advance_replay_boundary(
        self, boundary: float, checkpoint: ParticipationCheckpoint | None = None
    ) -> None:
        if boundary <= self.state.replay_after or boundary < 0:
            return
        if checkpoint is None:
            checkpoint = self._participation_checkpoint()
        units = {
            (baseline.thread, baseline.target) for baseline in self.state.belief_baselines.values()
        }
        units.update(
            (v.binding.unit.thread, v.binding.unit.target)
            for v in checkpoint.units.values()
            if self._binding_valid(v.binding, checkpoint)
        )
        for observation in self.state.observations.values():
            event = self.state.events.get(observation.snapshot.focus.event_id)
            if event is not None:
                event = self._resolved_event(event)
                units.add((event.thread, event.target))
        for record in self.state.effects.values():
            proposal = self.state.proposals.get(record.proposal_id)
            if proposal and record.effect.kind == "message":
                units.update((proposal.thread, target) for target in record.effect.actual_targets)
        prepared = []
        actions_by_unit = self._belief_actions(checkpoint)
        for thread, target in units:
            key = self._unit_key(thread, target)
            previous = self.state.belief_baselines.get(key)
            last_action = previous.last_action if previous else -1.0
            actions = actions_by_unit.get((thread, target), [])
            last_action = max(
                last_action,
                max(
                    (
                        action[0]
                        for action in actions
                        if self.state.replay_after < action[0] <= boundary
                    ),
                    default=-1.0,
                ),
            )
            if last_action < 0:
                continue
            prepared.append(
                (
                    key,
                    BeliefBaseline(
                        thread=thread,
                        target=target,
                        at=boundary,
                        value=self._belief_from_actions(thread, target, boundary, actions),
                        last_action=last_action,
                    ),
                )
            )
        # A resource eviction is explicit and distinct from ordinary time decay.
        # Discarded units cannot replay retired events and manufacture a new baseline.
        prepared.sort(key=lambda item: (-item[1].last_action, item[0]))
        traces = {
            "compute": TraceBaseline(
                at=boundary,
                value=self._trace("compute", boundary, self.parameters.compute_decay_seconds),
            ),
            "activity": TraceBaseline(at=boundary, value=self._activity(boundary)),
            "social_context": TraceBaseline(at=boundary, value=self._social_context(boundary)),
            "no_reply": TraceBaseline(at=boundary, value=self._no_reply(boundary)),
        }
        old_human_activity = self.state.trace_baselines.get("human_activity")
        traces["human_activity"] = (
            old_human_activity
            if old_human_activity is not None and boundary < old_human_activity.at
            else TraceBaseline(at=boundary, value=self._human_activity(boundary))
        )
        self._set(
            belief_baselines=dict(prepared[:64]),
            baseline_evictions=self.state.baseline_evictions + max(0, len(prepared) - 64),
            replay_after=boundary,
            trace_baselines=traces,
        )

    def _closed(self, event: ScopedEvent, actor_identity: ActorIdentity = original_actor) -> bool:
        # Closing records persist independently of C decay; a newer source is not automatically
        # a reopening. Only a separately supported scoped invitation can release one.
        for boundary in self.state.boundaries.values():
            if (
                boundary.released_by is None
                and boundary.thread == event.thread
                and (
                    actor_identity(boundary.target) == actor_identity(event.target)
                    or boundary.group_wide
                )
            ):
                return True
        return False

    def source_allowed(self, event: ScopedEvent) -> bool:
        """Apply shared source/closure fences to a host's alternative admission path.

        The event must already be committed here. This does not assert a semantic
        opportunity or grant host permissions; it only prevents fallback from bypassing
        retractions, ambiguous units, consumed sources, and persistent stop boundaries.
        """
        if event.scope != self.state.scope or not self._valid(event.ref):
            return False
        stored = self._resolved_event(self.state.events[event.ref.event_id])
        return (
            not stored.unit_ambiguous
            and self.state.consumed.get(event.ref.event_id, 0) < event.ref.revision
            and not self._closed(stored)
        )

    def legacy_source_allowed(self, event: ScopedEvent) -> bool:
        """Share provenance and stop fences without requiring a disabled observer.

        Ambiguous legacy material does not acquire a semantic resolution. Every
        Host-offered unit must respect its existing boundary before legacy may
        independently consider the event under its own participation policy.
        """
        if (
            event.scope != self.state.scope
            or not self._valid(event.ref)
            or self.state.consumed.get(event.ref.event_id, 0) >= event.ref.revision
        ):
            return False
        stored = self._resolved_event(self.state.events[event.ref.event_id])
        if not stored.unit_ambiguous:
            return not self._closed(stored)
        choices = [stored]
        choices.extend(
            stored.model_copy(update={"thread": option.thread, "target": option.target})
            for option in stored.unit_options
        )
        return all(not self._closed(choice) for choice in choices)

    def _refresh_releases(self, *, actor_identity: ActorIdentity) -> None:
        """A late stop re-score must not undo a subsequent verified reopening."""
        for boundary_key, boundary in list(self.state.boundaries.items()):
            original = self.state.seen.get(boundary.source.event_id)
            if original is None:
                continue
            reopenings = []
            for observation in self.state.observations.values():
                raw = self.state.events.get(observation.snapshot.focus.event_id)
                if raw is None:
                    continue
                event = self._resolved_event(raw)
                act = observation.answers.get("interaction_mark")
                scope = observation.answers.get("boundary_scope")
                if (
                    event.kind != "human"
                    or event.unit_ambiguous
                    or not self._source_before(original.at, boundary.source, event.at, event.ref)
                    or event.thread != boundary.thread
                    or (
                        actor_identity(event.target) != actor_identity(boundary.target)
                        and not boundary.group_wide
                    )
                    or not _dimension_known(act)
                    or act.p("invite_yuki") < 0.8
                    or not _dimension_known(scope)
                ):
                    continue
                if scope.p("target_thread") >= 0.8 or scope.p("group_thread") >= 0.8:
                    reopenings.append(
                        (event.at, self._source_order(event.ref) or 0, event, observation)
                    )
            if reopenings:
                _, _, event, observation = max(reopenings, key=lambda item: item[:2])
                self._put(
                    "boundaries",
                    boundary_key,
                    boundary.model_copy(
                        update={
                            "released_by": event.ref,
                            "release_dependencies": observation.snapshot.context,
                        }
                    ),
                )

    def _eligible_groups(
        self,
        now: float,
        *,
        include_addressed: bool = True,
        allowed_kinds: frozenset[CandidateKind] | None = None,
        actor_identity: ActorIdentity = original_actor,
    ) -> dict[str, list[Candidate]]:
        groups: dict[tuple[CandidateKind, str, str], list[Candidate]] = {}
        for key, candidate in self.state.candidates.items():
            if candidate.kind is not CandidateKind.CONVERSATION or (
                allowed_kinds is not None and candidate.kind not in allowed_kinds
            ):
                continue
            support = candidate.support
            if (
                support.kind == "predicted"
                or (not include_addressed and self._addressed(candidate))
                or not self._valid(candidate.event.ref)
                or not self._valid(support.basis)
                or any(not self._valid(ref) for ref in support.dependencies)
                or (
                    self.state.observations.get(support.basis.event_id) is None
                    or self.state.observations[support.basis.event_id].observation_id
                    != support.observation_id
                )
                or support.strength(now) == 0
                or self._closed(candidate.event, actor_identity)
                or candidate.value.valid_until <= now
                or candidate.floor.valid_until <= now
                or self.state.consumed.get(key, 0) >= candidate.event.ref.revision
            ):
                continue
            unit = (candidate.kind, candidate.event.thread, candidate.event.target)
            groups.setdefault(unit, []).append(candidate)
        return {
            max(items, key=lambda c: (c.event.at, c.event.ref.event_id)).event.ref.event_id: items
            for items in groups.values()
        }

    def opportunity_scores(
        self,
        now: float,
        *,
        include_addressed: bool = True,
        allowed_kinds: frozenset[CandidateKind] | None = None,
    ) -> dict[str, float]:
        """Rates for legal non-request SELF opportunities; invitations bypass sampling."""
        groups = self._eligible_groups(
            now, include_addressed=include_addressed, allowed_kinds=allowed_kinds
        )
        return self._opportunity_scores(now, groups, self._autonomy_pressure(now))

    def _opportunity_scores(
        self, now: float, groups: dict[str, list[Candidate]], pressure: float
    ) -> dict[str, float]:
        raw: dict[str, float] = {}
        context = self._social_context(now)
        actions_by_unit: dict[tuple[str, str], list[BeliefAction]] | None = None
        last_human_by_unit: dict[tuple[str, str], float] = {}
        for key, members in groups.items():
            candidate = self.state.candidates[key]
            tau = self.parameters.conversation_source_decay_seconds
            for member in members:
                self._put(
                    "attentions",
                    member.event.ref.event_id,
                    dynamics.source_attention(
                        member.value.value,
                        now - member.event.at,
                        now - max(member.support.issued_at, member.event.at),
                        tau,
                    ),
                )
            if any(self._addressed(member) for member in members):
                raw[key] = max(member.value.value * member.floor.value for member in members)
                continue
            if actions_by_unit is None:
                actions_by_unit = self._belief_actions(self._participation_checkpoint())
                for original in self.state.events.values():
                    event = self._resolved_event(original)
                    if (
                        event.scope == self.state.scope
                        and event.kind == "human"
                        and not event.unit_ambiguous
                        and event.at <= now
                        and self._valid(event.ref)
                    ):
                        unit = (event.thread, event.target)
                        last_human_by_unit[unit] = max(last_human_by_unit.get(unit, 0), event.at)
            unit = (candidate.event.thread, candidate.event.target)
            b = self._belief_from_actions(*unit, now, actions_by_unit.get(unit, []))
            x = dynamics.aggregate_attention(
                [
                    (member.value.value, member.event.at, member.support.issued_at)
                    for member in members
                ],
                now,
                tau,
            )
            last_human_fragment = last_human_by_unit.get(unit, candidate.event.at)
            human_sources = [
                (member.value.value, member.event.at, member.support.issued_at)
                for member in members
                if member.event.kind == "human"
            ]
            raw[key] = dynamics.source_rate(
                b,
                attention=x,
                familiarity=self._familiarity(
                    candidate.kind, candidate.event.thread, candidate.event.target, now
                ),
                support=min(member.support.strength(now) for member in members),
                floor=candidate.floor.value,
                gap=now - last_human_fragment,
                context=dynamics.social_context(
                    context, dynamics.aggregate_attention(human_sources, now, tau)
                ),
                tendency=self._willingness(now),
                pressure=pressure,
                parameters=self.parameters,
            )
        return raw

    def _addressed(self, candidate: Candidate) -> bool:
        observation = self.state.observations.get(candidate.support.basis.event_id)
        if (
            candidate.kind != CandidateKind.CONVERSATION
            or candidate.support.kind != "observed"
            or candidate.event.ref != candidate.support.basis
            or observation is None
        ):
            return False
        act = observation.answers.get("interaction_mark")
        floor = observation.answers.get("floor_state")
        return bool(
            _dimension_known(act)
            and _dimension_known(floor)
            and act.choice in {"invite_yuki", "extend_yuki"}
            and floor.choice == "yuki"
        )

    def _trace(self, kind: str, now: float, tau: float) -> float:
        events = sorted(
            e.effect.at
            for e in self.state.effects.values()
            if e.effect.kind == kind and e.effect.at <= now
        )
        return self._leaky_trace(kind, events, now, tau, self.parameters.compute_increment)

    def _activity(self, now: float) -> float:
        events = sorted(
            e.at for e in self.state.events.values() if e.kind == "human" and e.at <= now
        )
        return self._leaky_trace(
            "activity",
            events,
            now,
            self.parameters.activity_decay_seconds,
            self.parameters.activity_increment,
        )

    def _social_context(self, now: float) -> float:
        events = sorted(
            e.at for e in self.state.events.values() if e.kind == "human" and e.at <= now
        )
        return self._leaky_trace(
            "social_context",
            events,
            now,
            self.parameters.social_context_decay_seconds,
            self.parameters.social_context_increment,
        )

    def _human_activity(self, now: float) -> float:
        """Linear 48-hour human-event trace, independent of short source context."""
        baseline = self.state.trace_baselines.get("human_activity")
        if baseline is not None and now < baseline.at:
            return 0.0
        tau = self.parameters.human_activity_decay_seconds
        value = baseline.value * math.exp(-(now - baseline.at) / tau) if baseline else 0.0
        return value + sum(
            math.exp(-(now - event.at) / tau)
            for event in self.state.events.values()
            if event.kind == "human"
            and event.at <= now
            and event.at > (baseline.at if baseline else self.state.replay_after)
        )

    def initialize_human_activity(
        self,
        at: float,
        historical_buckets: tuple[tuple[float, int], ...],
        last_historical_human_at: float | None,
    ) -> None:
        """Seed a legacy snapshot from bounded Host aggregates once."""
        if self.state.human_activity_initialized:
            return
        if not math.isfinite(at) or at < max(0.0, self.state.replay_after):
            raise ValueError("human_activity_boundary_invalid")
        if any(
            not math.isfinite(event_at) or event_at < 0 or event_at > at or count <= 0
            for event_at, count in historical_buckets
        ) or (
            last_historical_human_at is not None
            and (
                not math.isfinite(last_historical_human_at)
                or last_historical_human_at < 0
                or last_historical_human_at > at
            )
        ):
            raise ValueError("human_activity_history_invalid")
        tau = self.parameters.human_activity_decay_seconds
        value = sum(
            count * math.exp(-(at - event_at) / tau) for event_at, count in historical_buckets
        )
        baselines = dict(self.state.trace_baselines)
        baselines["human_activity"] = TraceBaseline(at=at, value=value)
        self._set(
            trace_baselines=baselines,
            human_activity_initialized=True,
            last_human_at=max(
                (
                    value
                    for value in (self.state.last_human_at, last_historical_human_at)
                    if value is not None
                ),
                default=None,
            ),
        )

    def _no_reply(self, now: float) -> float:
        events = sorted(
            report.at
            for report in self.state.feedback.values()
            if report.outcome == "no_reply"
            and report.at <= now
            and self.state.proposals.get(report.proposal_id) is not None
            and self.state.proposals[report.proposal_id].kind == CandidateKind.INTRINSIC
        )
        return self._leaky_trace(
            "no_reply",
            events,
            now,
            self.parameters.no_reply_decay_seconds,
            self.parameters.no_reply_increment,
        )

    def _work_density(self, now: float, tau: float) -> float:
        return sum(
            math.exp(-(now - pulse.started_at) / tau)
            for pulse in self.state.work_pulses.values()
            if pulse.started_at <= now
        )

    def _reception(self, now: float) -> float:
        baseline = self.state.reception_baseline
        total = (
            baseline.value
            * math.exp(-(now - baseline.at) / self.parameters.reception_decay_seconds)
            if baseline is not None and baseline.at <= now
            else 0.0
        ) + sum(
            record.score * math.exp(-(now - record.at) / self.parameters.reception_decay_seconds)
            for record in self.state.receptions.values()
            if record.at <= now
        )
        return math.tanh(total)

    def _exposure(self, now: float) -> float:
        scores = {
            run_ref: pulse.archived_positive for run_ref, pulse in self.state.work_pulses.items()
        }
        for reception in self.state.receptions.values():
            if reception.at <= now:
                scores[reception.run_ref] = max(scores.get(reception.run_ref, 0), reception.score)
        return sum(
            (1 - max(0.0, scores.get(run_ref, 0)))
            * (1 - math.exp(-(now - pulse.public_at) / self.parameters.exposure_rise_seconds))
            * math.exp(-(now - pulse.public_at) / self.parameters.exposure_decay_seconds)
            for run_ref, pulse in self.state.work_pulses.items()
            if pulse.public_at is not None and pulse.public_at <= now
        )

    def _autonomy_pressure(self, now: float) -> float:
        return dynamics.autonomy_pressure(
            tendency=self._willingness(now),
            activity=self._activity(now),
            compute=self._trace("compute", now, self.parameters.compute_decay_seconds),
            no_reply=self._no_reply(now),
            work_fast=self._work_density(now, self.parameters.work_fast_decay_seconds),
            work_slow=self._work_density(now, self.parameters.work_slow_decay_seconds),
            exposure=self._exposure(now),
            reception=self._reception(now),
            parameters=self.parameters,
        )

    def observe_public_anchor(self, run_ref: str, event_id: str, at: float) -> bool:
        """Bind a confirmed internal outgoing event to its autonomous Work."""
        if not run_ref or not event_id or not math.isfinite(at) or at < 0:
            raise ValueError("invalid_outbound_anchor")
        old = self.state.outbound_anchors.get(event_id)
        if old is not None:
            return old == OutboundAnchor(run_ref=run_ref, at=at)
        self._put("outbound_anchors", event_id, OutboundAnchor(run_ref=run_ref, at=at))
        for key, observation in self.state.observations.items():
            if observation.matching_self_anchor is not None:
                if observation.matching_self_anchor.event_id == event_id:
                    self._record_reception(key, observation)
        return True

    def _record_reception(self, event_id: str, observation: StoredObservation) -> None:
        anchor = observation.matching_self_anchor
        event = self.state.events.get(event_id)
        act = observation.answers.get("interaction_mark")
        if event is None or event.kind != "human" or not _dimension_known(act):
            self._pop("receptions", event_id, None)
            return
        if anchor is None:
            # Other group talk is weak evidence of being passed over, never proof
            # of rejection. Its influence decays with distance from the last Work.
            recent = max(
                (
                    (pulse.public_at, run_ref)
                    for run_ref, pulse in self.state.work_pulses.items()
                    if pulse.public_at is not None and pulse.public_at < event.at
                ),
                default=None,
            )
            if recent is None or act.p("other_exchange") == 0:
                self._pop("receptions", event_id, None)
                return
            sent_at, run_ref = recent
            assert sent_at is not None
            score = (
                -self.parameters.unanchored_reception_weight
                * act.p("other_exchange")
                * math.exp(
                    -(event.at - sent_at) / self.parameters.unanchored_reception_decay_seconds
                )
            )
            self._put("receptions", event_id, Reception(run_ref=run_ref, at=event.at, score=score))
            return
        run = self.state.outbound_anchors.get(anchor.event_id)
        if run is None:
            self._pop("receptions", event_id, None)
            return
        positive = act.p("extend_yuki") + act.p("acknowledge") + act.p("invite_yuki")
        negative = act.p("ask_yuki_stop")
        score = max(-1.0, min(1.0, positive - negative))
        self._put("receptions", event_id, Reception(run_ref=run.run_ref, at=event.at, score=score))

    def _leaky_trace(
        self,
        kind: str,
        events: list[float],
        now: float,
        tau: float,
        increment: float,
    ) -> float:
        baseline = self.state.trace_baselines.get(kind)
        if baseline is not None and now < baseline.at:
            return 0.0
        value = baseline.value if baseline else 0.0
        last = baseline.at if baseline else (events[0] if events else now)
        for at in events:
            if at <= self.state.replay_after:
                continue
            value *= math.exp(-(at - last) / tau)
            value += increment * (1 - value)
            last = at
        return value * math.exp(-(now - last) / tau)

    def observe_self_report(self, report: SelfReport) -> bool:
        if report.run_ref not in self.state.feedback and not any(
            record.run_ref == report.run_ref for record in self.state.effects.values()
        ):
            return False
        old = self.state.self_reports.get(report.run_ref)
        if old and (report.sequence <= old.sequence or report.response_id == old.response_id):
            return False
        self._put("self_reports", report.run_ref, report)
        last = self.state.engagement_report
        if report.delta.engage is not None and (last is None or report.at >= last.at):
            self._set(engagement_report=report)
        return True

    def observe_committed_effect(self, run_ref: str, effect: Effect) -> bool:
        """Record a real non-semantic host effect without fabricating a proposal.

        Self ledger events are anchors only. Hosts use the same logical effect ID
        here and in semantic feedback when those describe the same actual action.
        This path contributes to scope traces, not target-specific belief changes.
        """
        if not run_ref:
            raise ValueError("effect_run_required")
        old = self.state.effects.get(effect.effect_id)
        if old is not None:
            return old.run_ref == run_ref and old.effect == effect
        if effect.kind == "message":
            self._set(last_self_message_at=max(self.state.last_self_message_at or 0, effect.at))
        # The replay boundary already retired these transient trace inputs.
        # Real execution/delivery remains in the Host ledger; do not recreate it here.
        if effect.at <= self.state.replay_after:
            return False
        self._put(
            "effects",
            effect.effect_id,
            RecordedEffect(proposal_id=f"host:{run_ref}", run_ref=run_ref, effect=effect),
        )
        return True

    def _willingness(self, now: float) -> float:
        last = self.state.engagement_report
        if last is None or last.at > now or last.delta.engage is None:
            return 0
        value = {
            "join": self.parameters.willingness_magnitude,
            "stay": 0,
            "quiet": -self.parameters.willingness_magnitude,
        }[last.delta.engage]
        return value * math.exp(-(now - last.at) / self.parameters.willingness_decay_seconds)

    def intrinsic_opportunity(self, now: float) -> float:
        return self._intrinsic_opportunity(now, self._autonomy_pressure(now))

    def _intrinsic_opportunity(self, now: float, pressure: float) -> float:
        return dynamics.intrinsic_rate(
            human_activity=self._human_activity(now),
            seconds_since_human=(
                max(0.0, now - self.state.last_human_at)
                if self.state.last_human_at is not None
                else None
            ),
            pressure=pressure,
            parameters=self.parameters,
        )

    def _sample(self, sequence: int, stream: str) -> float:
        scope = self.state.scope
        source = (
            f"{scope.conversation_id}:{scope.generation}:{self.state.epoch}:{sequence}:{stream}"
        ).encode()
        number = int.from_bytes(hashlib.sha256(source).digest()[:8], "big")
        return (number + 0.5) / 2**64

    def advance(
        self,
        now: float,
        *,
        controller_epoch: int,
        host_available: bool,
        intrinsic_allowed: bool = False,
        include_addressed: bool = True,
        allowed_kinds: frozenset[CandidateKind] | None = None,
    ) -> Proposal | None:
        if not math.isfinite(now) or now < self.state.now:
            raise ValueError("clock_must_be_finite_and_monotonic")
        if controller_epoch != self.state.epoch:
            # Switching invalidates pending proposals, not feedback for accepted runs.
            self._set(epoch=controller_epoch, pending=None, now=now, last_sample_at=now)
            return None
        elapsed = now - self.state.now
        if elapsed > 5:
            self._set(skipped_seconds=self.state.skipped_seconds + elapsed - 5)
        can_propose = host_available and not self.state.capacity_blocked and not self.state.pending
        self._set(now=now)
        self._prune()
        if not can_propose:
            self._set(last_sample_at=now)
            return None
        groups = self._eligible_groups(
            now, include_addressed=include_addressed, allowed_kinds=allowed_kinds
        )
        pressure = self._autonomy_pressure(now)
        scores = self._opportunity_scores(now, groups, pressure)
        addressed = [
            key
            for key, members in groups.items()
            if any(self._addressed(member) for member in members)
        ]
        if addressed:
            key = max(addressed, key=lambda item: (scores[item], item))
            self._set(last_sample_at=now)
        else:
            if intrinsic_allowed and not any(
                boundary.explicit_stop and boundary.group_wide and boundary.released_by is None
                for boundary in self.state.boundaries.values()
            ):
                scores["__intrinsic__"] = self._intrinsic_opportunity(now, pressure)
            if not scores:
                self._set(last_sample_at=now)
                return None
            total = sum(max(0.0, rate) for rate in scores.values())
            last_sample = self.state.last_sample_at
            dt = max(0.0, now - (last_sample if last_sample is not None else now))
            sequence = self.state.sample_sequence + 1
            self._set(last_sample_at=now, sample_sequence=sequence)
            if total <= 0 or self._sample(sequence, "arrival") >= -math.expm1(-total * dt):
                return None
            cursor = self._sample(sequence, "selection") * total
            key = sorted(scores)[-1]
            for candidate_key, rate in sorted(scores.items()):
                cursor -= max(0.0, rate)
                if cursor < 0:
                    key = candidate_key
                    break
        if key == "__intrinsic__":
            proposal = Proposal(
                proposal_id=str(uuid4()),
                scope=self.state.scope,
                controller_epoch=controller_epoch,
                kind=CandidateKind.INTRINSIC,
                thread=f"intrinsic:{self.state.scope.generation}",
                target_hint="group",
                sources=(),
                support=None,
                created_at=now,
                expires_at=now + 60,
            )
        else:
            candidate = self.state.candidates[key]
            members = groups[key]
            proposal = Proposal(
                proposal_id=str(uuid4()),
                scope=self.state.scope,
                controller_epoch=controller_epoch,
                kind=candidate.kind,
                thread=candidate.event.thread,
                target_hint=candidate.event.target,
                sources=tuple(member.event.ref for member in members),
                support=candidate.support,
                supports=tuple(member.support for member in members),
                created_at=now,
                expires_at=min(member.support.valid_until for member in members),
            )
        self._put("proposals", proposal.proposal_id, proposal)
        self._set(pending=proposal.proposal_id)
        return proposal

    def discard_unaccepted_proposal(self, proposal_id: str) -> bool:
        """Retire a pending proposal after Host confirms it has no accepted run.

        Source interpretations and candidates remain available for ordinary admission.
        An external unknown admission must first be resolved by Host using its original ID.
        """
        if (
            self.state.pending != proposal_id
            or proposal_id not in self.state.proposals
            or proposal_id in self.state.proposal_runs
            or any(f.proposal_id == proposal_id for f in self.state.feedback.values())
            or any(e.proposal_id == proposal_id for e in self.state.effects.values())
        ):
            return False
        self._pop("proposals", proposal_id)
        self._set(pending=None)
        return True

    def observe_run_feedback(self, feedback: Feedback) -> bool:
        proposal = self.state.proposals.get(feedback.proposal_id)
        old = self.state.feedback.get(feedback.run_ref)
        if proposal is None or (
            old and (old.sequence >= feedback.sequence or old.proposal_id != feedback.proposal_id)
        ):
            return False
        registered = self.state.proposal_runs.get(proposal.proposal_id)
        if registered is not None and registered != feedback.run_ref:
            return False
        if old and old.outcome in {"completed", "interrupted", "no_reply", "rejected"}:
            if feedback.outcome != old.outcome and not feedback.effects:
                return False
            # Late actual receipts remain facts even after host cancellation. They can
            # supplement the terminal run but cannot turn it back into an accepted run.
            feedback = feedback.model_copy(update={"outcome": old.outcome})
        incoming_effects: dict[str, Effect] = {}
        for effect in feedback.effects:
            previous_effect = incoming_effects.get(effect.effect_id)
            if previous_effect is not None and previous_effect != effect:
                return False
            incoming_effects[effect.effect_id] = effect
        if any(
            effect.effect_id in self.state.effects
            and (
                self.state.effects[effect.effect_id].run_ref != feedback.run_ref
                or self.state.effects[effect.effect_id].effect != effect
                or self.state.effects[effect.effect_id].proposal_id
                not in {proposal.proposal_id, f"host:{feedback.run_ref}"}
            )
            for effect in incoming_effects.values()
        ):
            return False
        if any(ref not in proposal.sources for ref in feedback.considered_refs):
            return False
        self._put("feedback", feedback.run_ref, feedback)
        self._put("proposal_runs", proposal.proposal_id, feedback.run_ref)
        for effect in incoming_effects.values():
            self._put(
                "effects",
                effect.effect_id,
                RecordedEffect(
                    proposal_id=proposal.proposal_id,
                    run_ref=feedback.run_ref,
                    effect=effect,
                ),
            )
            if effect.kind == "message":
                self._set(last_self_message_at=max(self.state.last_self_message_at or 0, effect.at))
        if feedback.outcome not in {"busy", "rejected"}:
            pulse = self.state.work_pulses.get(feedback.run_ref)
            computes = [
                effect.at for effect in incoming_effects.values() if effect.kind == "compute"
            ]
            messages = [
                effect.at for effect in incoming_effects.values() if effect.kind == "message"
            ]
            if pulse is None and (computes or messages):
                pulse = WorkPulse(started_at=min(computes or messages))
            if pulse is not None:
                was_public = pulse.public_at is not None
                if computes and min(computes) < pulse.started_at:
                    pulse = pulse.model_copy(update={"started_at": min(computes)})
                if messages:
                    first_public = min(messages)
                    pulse = pulse.model_copy(
                        update={
                            "public_at": min(pulse.public_at, first_public)
                            if pulse.public_at is not None
                            else first_public
                        }
                    )
                self._put("work_pulses", feedback.run_ref, pulse)
                if messages and not was_public:
                    for event_id, observation in self.state.observations.items():
                        event = self.state.events.get(event_id)
                        if event is not None and event.at > min(messages):
                            self._record_reception(event_id, observation)
        if feedback.outcome not in {"busy", "rejected"}:
            # A recovered terminal result also proves this proposal was accepted. The
            # acceptance notification can be missing; its sources must not reopen merely
            # because the final receipt contains no additional considered_refs.
            for ref in proposal.sources:
                self._put(
                    "consumed",
                    ref.event_id,
                    max(
                        ref.revision,
                        self.state.consumed.get(ref.event_id, 0),
                    ),
                )
            if old is None or old.outcome in {"busy", "rejected"}:
                self._set(last_accepted=max(self.state.last_accepted, feedback.at))
                if proposal.kind != CandidateKind.INTRINSIC:
                    self._consider(proposal, feedback.at)
        # Pending owns only an unanswered proposal. The Host's durable Work and
        # execution lease own accepted work, including waits and later resumes.
        if self.state.pending == proposal.proposal_id:
            self._set(pending=None)
        return True

    def _prune(self, *, normalize_checkpoint: bool = False) -> None:
        cutoff = self.state.now - 600
        # Keep enough pulses for the largest active Work/exposure time constant.
        social_cutoff = self.state.now - max(
            3 * 86400,
            12
            * max(
                self.parameters.work_fast_decay_seconds,
                self.parameters.work_slow_decay_seconds,
                self.parameters.exposure_decay_seconds,
            ),
        )
        for run_ref, pulse in list(self.state.work_pulses.items()):
            if pulse.started_at < social_cutoff:
                self._pop("work_pulses", run_ref)
        for event_id, anchor in list(self.state.outbound_anchors.items()):
            if anchor.at < self.state.now - 3600:
                self._pop("outbound_anchors", event_id)
        reception_cutoff = self.state.now - 2 * 86400
        evicted: list[tuple[str, Reception]] = []
        if len(self.state.receptions) > 2048 or any(
            record.at < reception_cutoff for record in self.state.receptions.values()
        ):
            ordered_receptions = sorted(
                self.state.receptions.items(), key=lambda item: (item[1].at, item[0])
            )
            excess = max(0, len(ordered_receptions) - 2048)
            evicted = [
                (event_id, record)
                for index, (event_id, record) in enumerate(ordered_receptions)
                if record.at <= self.state.now and (record.at < reception_cutoff or index < excess)
            ]
        if evicted:
            old_base = self.state.reception_baseline
            boundary_at = max(
                max(record.at for _, record in evicted),
                old_base.at if old_base else 0,
            )
            value = (
                old_base.value
                * math.exp(-(boundary_at - old_base.at) / self.parameters.reception_decay_seconds)
                if old_base
                else 0.0
            ) + sum(
                record.score
                * math.exp(-(boundary_at - record.at) / self.parameters.reception_decay_seconds)
                for _, record in evicted
            )
            self._set(reception_baseline=ReceptionBaseline(at=boundary_at, value=value))
            for event_id, record in evicted:
                work_pulse = self.state.work_pulses.get(record.run_ref)
                if work_pulse is not None and record.score > work_pulse.archived_positive:
                    self._put(
                        "work_pulses",
                        record.run_ref,
                        work_pulse.model_copy(update={"archived_positive": record.score}),
                    )
                self._pop("receptions", event_id)
        boundary = max(self.state.replay_after, math.nextafter(cutoff, -math.inf))
        retained = sorted(
            (event for event in self.state.events.values() if event.at > boundary),
            key=lambda event: (event.at, event.ref.event_id),
        )
        retained_bytes = sum(len(event.text.encode("utf-8")) for event in retained)
        remaining = len(retained)
        offset = 0
        while remaining > 256 or retained_bytes > 256 * 1024:
            boundary = retained[offset].at
            # A timestamp is one replay boundary. Never summarize one event at t while
            # retaining another at t, which would make late rescoring ambiguous.
            while offset < len(retained) and retained[offset].at <= boundary:
                retained_bytes -= len(retained[offset].text.encode("utf-8"))
                remaining -= 1
                offset += 1
        original_checkpoint = self._read_participation_checkpoint()
        checkpoint = self._retire_participation(boundary, original_checkpoint)
        participation_refs = {r.event_id for r in self._participation_refs(checkpoint)}
        expression_ids = self._unit_expression_ids(checkpoint)
        self._advance_replay_boundary(boundary, checkpoint)
        # Drop invalidated/consumed fingerprints only with their source window. Older sources
        # cannot re-enter through observe_committed_event's time fence.
        for key, event in list(self.state.events.items()):
            if event.at <= self.state.replay_after:
                self._pop("events", key)
                self._pop("observations", key, None)
                self._pop("candidates", key, None)
        boundary_refs = {
            ref.event_id
            for boundary in self.state.boundaries.values()
            for ref in (
                boundary.source,
                *boundary.dependencies,
                *((boundary.released_by,) if boundary.released_by else ()),
                *boundary.release_dependencies,
            )
        }
        for key, seen in list(self.state.seen.items()):
            if (
                seen.at <= self.state.replay_after
                and key not in boundary_refs
                and key not in participation_refs
            ):
                self._pop("seen", key)
        for name in ("consumed", "invalidated"):
            for key in list(getattr(self.state, name)):
                if key not in self.state.seen and key not in self.state.boundaries:
                    self._pop(name, key)
        recent_runs = {
            record.run_ref for record in self.state.effects.values() if record.effect.at >= cutoff
        }
        for run, feedback in list(self.state.feedback.items()):
            recent_effect = run in recent_runs
            if feedback.at < cutoff and not recent_effect:
                # Accepted execution belongs to the Host, even if its final
                # short-lived scheduling feedback never reached this controller.
                if self.state.pending == feedback.proposal_id:
                    self._set(pending=None)
                self._pop("feedback", run)
                self._pop("proposals", feedback.proposal_id, None)
                self._pop("proposal_runs", feedback.proposal_id, None)
                self._pop("self_reports", run, None)
        for key, effect in list(self.state.effects.items()):
            if (
                effect.effect.at < cutoff
                and effect.run_ref not in self.state.feedback
                and key not in expression_ids
            ):
                self._pop("effects", key)
        referenced = {feedback.proposal_id for feedback in self.state.feedback.values()}
        for key, proposal in list(self.state.proposals.items()):
            if proposal.expires_at < cutoff and key not in referenced and key != self.state.pending:
                self._pop("proposals", key)
        # Never silently discard unresolved boundary or run identities to make room.
        self._set(
            capacity_blocked=len(self.state.boundaries) > 256 or len(self.state.proposals) > 256
        )
        for key in list(self.state.attentions):
            if key not in self.state.candidates:
                self._pop("attentions", key)
        if self.state.pending:
            proposal = self.state.proposals[self.state.pending]
            accepted = any(
                f.proposal_id == proposal.proposal_id and f.outcome == "accepted"
                for f in self.state.feedback.values()
            )
            # Old snapshots retained accepted proposals in this slot. Their real
            # acceptance feedback already transferred ownership to the Host.
            if accepted:
                self._set(pending=None)
        if original_checkpoint is not None:
            self._prune_participation(
                checkpoint, previous=original_checkpoint, normalize=normalize_checkpoint
            )

    @classmethod
    def restore(
        cls,
        state: State,
        now: float,
        parameters: AutonomyParameters = DEFAULT_AUTONOMY_PARAMETERS,
    ) -> Controller:
        controller = cls(state.scope, now, parameters)
        # No elapsed-time catch-up and no resubmission of an uncertain proposal.
        controller.state = state.model_copy(update={"now": now, "last_sample_at": now}, deep=True)
        if controller.state.last_human_at is None:
            last_human = max(
                (event.at for event in state.events.values() if event.kind == "human"),
                default=None,
            )
            if last_human is not None:
                controller._set(last_human_at=last_human)
        if controller.state.last_self_message_at is None:
            last_speech = max(
                (
                    *(event.at for event in state.events.values() if event.kind == "self"),
                    *(
                        record.effect.at
                        for record in state.effects.values()
                        if record.effect.kind == "message"
                    ),
                ),
                default=None,
            )
            if last_speech is not None:
                controller._set(last_self_message_at=last_speech)
        # Persist existing schema migrations once on load, even if retirement is
        # unchanged. Steady-state pruning needs no identical serialization.
        controller._prune(normalize_checkpoint=True)
        return controller
