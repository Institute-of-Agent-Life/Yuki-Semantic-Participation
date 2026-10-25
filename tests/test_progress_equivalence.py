"""Exact old-pin numerical/dirty oracle; synthetic inputs, no provider or Host I/O."""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from uuid import UUID

from tests.test_controller import SCOPE, event, observation
from yuki_participation import controller as current
from yuki_participation import models, self_report
from yuki_participation.snapshot import freeze, thaw

PIN = "79d266918dc6ce19b7d81568565deaed450301d5"


def old_package(tmp_path):
    root = tmp_path / "participation_oracle"
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", PIN, "src/yuki_participation"], text=True
    ).splitlines()
    for source in paths:
        if source.endswith(".py"):
            target = root / Path(source).relative_to("src/yuki_participation")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{PIN}:{source}"]))
    spec = importlib.util.spec_from_file_location(
        "participation_oracle", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return importlib.import_module("participation_oracle.controller")


def test_original_ticks_hash_stream_restore_and_every_frozen_field(tmp_path, monkeypatch):
    old = old_package(tmp_path)
    originals = old.Controller(old.Scope.model_validate(SCOPE.model_dump()), 100)
    actual = current.Controller(SCOPE, 100)
    for module in (old, current):
        values = iter(UUID(int=i) for i in range(1, 1000))
        monkeypatch.setattr(module, "uuid4", lambda values=values: next(values))
    saved_layout = body = None

    def verify():
        nonlocal saved_layout, body
        assert len(type(actual.state).model_fields) == 38
        assert actual.state.model_dump_json() == originals.state.model_dump_json()
        frozen = freeze(
            actual.state, body_changed=actual.take_changes()[1], saved_layout=saved_layout
        )
        if frozen.body is not None:
            body = frozen.body
        assert (
            thaw(SCOPE, frozen.progress, body).model_dump_json()
            == originals.state.model_dump_json()
        )
        saved_layout = frozen.layout
        return frozen

    def apply(method, *args, **kwargs):
        legacy = []
        for value in args:
            if hasattr(value, "model_dump_json"):
                module = importlib.import_module(
                    "participation_oracle." + type(value).__module__.rsplit(".", 1)[-1]
                )
                value = getattr(module, type(value).__name__).model_validate_json(
                    value.model_dump_json()
                )
            legacy.append(value)
        expected = getattr(originals, method)(*legacy, **kwargs)
        result = getattr(actual, method)(*args, **kwargs)
        if hasattr(result, "model_dump_json"):
            assert result.model_dump_json() == expected.model_dump_json()
        else:
            assert result == expected
        verify()
        return result

    verify()
    apply("initialize_human_activity", 100, ((50, 4), (80, 2)), 80)
    for index, at in enumerate((100, 102, 110, 130)):
        source = event(f"source-{index}", at=at)
        apply("observe_committed_event", source)
        apply("apply_semantic_observation", observation(source, act="other_exchange", floor="open"))
        for moment in (at, at + 0.25, at + 2, at + 2):
            proposal = apply(
                "advance",
                moment,
                controller_epoch=0,
                host_available=True,
                intrinsic_allowed=True,
                include_addressed=False,
            )
            if proposal is not None:
                run = f"run-{index}-{moment}"
                apply(
                    "observe_run_feedback",
                    models.Feedback(
                        run_ref=run,
                        proposal_id=proposal.proposal_id,
                        sequence=1,
                        outcome="accepted",
                        at=moment,
                    ),
                )
                apply(
                    "observe_committed_effect",
                    run,
                    models.Effect(effect_id=f"compute-{run}", kind="compute", at=moment),
                )
                apply(
                    "observe_committed_effect",
                    run,
                    models.Effect(
                        effect_id=f"message-{run}",
                        kind="message",
                        at=moment,
                        actual_targets=("group",),
                    ),
                )
                apply(
                    "observe_self_report",
                    self_report.SelfReport(
                        run_ref=run,
                        sequence=1,
                        response_id=f"response-{run}",
                        at=moment,
                        delta=self_report.SelfDelta(engage="stay"),
                    ),
                )
                apply(
                    "observe_run_feedback",
                    models.Feedback(
                        run_ref=run,
                        proposal_id=proposal.proposal_id,
                        sequence=2,
                        outcome="completed",
                        at=moment,
                    ),
                )
    apply("observe_source_change", models.SourceRef(event_id="source-1", revision=1))
    for moment in (140, 200, 700, 702, 1800, 1802, 1900):
        apply("advance", moment, controller_epoch=0, host_available=False)
    parameters = actual.parameters.model_copy(update={"compute_decay_seconds": 913.0})
    apply("set_parameters", parameters)
    frozen = verify()
    actual = current.Controller.restore(thaw(SCOPE, frozen.progress, body), 2000, parameters)
    originals = old.Controller.restore(originals.state, 2000, originals.parameters)
    verify()
    for moment in (2000, 2002, 2004, 2012, 2400):
        apply("advance", moment, controller_epoch=0, host_available=True, intrinsic_allowed=True)


def test_readiness_never_consumes_and_backoff_still_cleans_quiet_and_expiry():
    from tests.test_participation_continuation import binding as bind
    from tests.test_participation_continuation import report
    from yuki_participation.session import ObservationSession

    controller = current.Controller(SCOPE, 100)
    source = event()
    controller.observe_committed_event(source)
    binding = bind(source)
    controller.observe_unit_input(binding, source.ref, continuation=True)
    session = ObservationSession(controller, None)
    session.request_observation(source.ref)
    session.retry_after = 160
    session.checkpoint()
    before = (session.queue.sequence, session.queue.last_call, session.queue.in_flight)
    assert not session.ready(104) and session.next_due() == 160
    session.maintain(104)
    assert session.queue.pending
    assert (session.queue.sequence, session.queue.last_call, session.queue.in_flight) == before
    controller.observe_unit_hint(binding, report("quiet"))
    session.maintain(105)
    assert not session.queue.pending and session.next_due() is None
    other = event("other", at=106)
    session.observe(other)
    session.request_observation(other.ref)
    session.maintain(181)
    assert session.queue.expired == 1 and not session.queue.pending
    assert controller.state.observer_checkpoint["queue"]["expired"] == 1
