"""One logical State, encoded as a small progress record and an immutable body."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .controller import HOT_FIELDS, State
from .models import Record, Scope


class Progress(Record):
    now: float
    last_sample_at: float | None
    sample_sequence: int
    skipped_seconds: float
    replay_after: float
    baseline_evictions: int
    baseline_invalidations: int
    belief: tuple[tuple[float, tuple[float, float, float, float, float], float], ...]
    trace: dict[str, tuple[float, float]]
    attention: tuple[float, ...]
    reception: tuple[float, float] | None


class Body(Record):
    state: dict[str, Any]
    belief: tuple[tuple[str, str, str], ...]
    attention: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    belief: tuple[tuple[str, str, str], ...]
    attention: tuple[str, ...]
    trace: tuple[str, ...]
    reception: bool


@dataclass(frozen=True)
class SnapshotWrite:
    scope: Scope
    progress: bytes
    body: bytes | None
    logical_size_delta: int | None
    layout: Layout


def progress_of(state: State) -> Progress:
    return Progress(
        now=state.now,
        last_sample_at=state.last_sample_at,
        sample_sequence=state.sample_sequence,
        skipped_seconds=state.skipped_seconds,
        replay_after=state.replay_after,
        baseline_evictions=state.baseline_evictions,
        baseline_invalidations=state.baseline_invalidations,
        belief=tuple((b.at, b.value, b.last_action) for b in state.belief_baselines.values()),
        trace={key: (b.at, b.value) for key, b in state.trace_baselines.items()},
        attention=tuple(state.attentions.values()),
        reception=(state.reception_baseline.at, state.reception_baseline.value)
        if state.reception_baseline is not None
        else None,
    )


def layout_of(state: State) -> Layout:
    return Layout(
        tuple((key, b.thread, b.target) for key, b in state.belief_baselines.items()),
        tuple(state.attentions),
        tuple(state.trace_baselines),
        state.reception_baseline is not None,
    )


def freeze(
    state: State, *, body_changed: bool = True, saved_layout: Layout | None = None
) -> SnapshotWrite:
    layout = layout_of(state)
    progress = progress_of(state).model_dump_json().encode("utf-8")
    body = None
    delta = None
    if body_changed or layout != saved_layout:
        body = (
            Body(
                state=state.model_dump(mode="json", exclude=set(HOT_FIELDS)),
                belief=layout.belief,
                attention=layout.attention,
            )
            .model_dump_json()
            .encode("utf-8")
        )
        delta = len(state.model_dump_json().encode("utf-8")) - len(progress)
    return SnapshotWrite(state.scope, progress, body, delta, layout)


def assemble(progress_bytes: bytes | str, body_bytes: bytes | str) -> dict[str, Any]:
    progress = Progress.model_validate_json(progress_bytes)
    body = Body.model_validate_json(body_bytes)
    state = body.state
    state.update(
        {
            name: getattr(progress, name)
            for name in (
                "now",
                "last_sample_at",
                "sample_sequence",
                "skipped_seconds",
                "replay_after",
                "baseline_evictions",
                "baseline_invalidations",
            )
        }
    )
    state["belief_baselines"] = {
        key: dict(
            thread=thread, target=target, at=values[0], value=values[1], last_action=values[2]
        )
        for (key, thread, target), values in zip(body.belief, progress.belief, strict=True)
    }
    state["attentions"] = dict(zip(body.attention, progress.attention, strict=True))
    state["trace_baselines"] = {
        key: dict(at=values[0], value=values[1]) for key, values in progress.trace.items()
    }
    state["reception_baseline"] = (
        dict(at=progress.reception[0], value=progress.reception[1])
        if progress.reception is not None
        else None
    )
    return state


def thaw(scope: Scope, progress: bytes | str, body: bytes | str) -> State:
    state = State.model_validate(assemble(progress, body))
    if state.scope != scope:
        raise ValueError("snapshot_scope_mismatch")
    return state
