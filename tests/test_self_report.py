import pytest
from test_controller import controller, event, observation

from yuki_participation.models import Feedback
from yuki_participation.self_report import SelfDelta, SelfReport, extract_tail


@pytest.mark.parametrize(
    "tail", ["{broken}</yuki-state>", '{"engage":"admin"}</yuki-state>', '{"mood":"curious"}']
)
def test_malformed_optional_control_tail_does_not_leak_or_require_retry(tail):
    body, delta = extract_tail("正文。\n<yuki-state>" + tail)
    assert body == "正文。"
    assert delta is None


def test_normal_body_and_optional_partial_report():
    assert extract_tail("正文。") == ("正文。", None)
    body, delta = extract_tail('正文。\n<yuki-state>{"mood":"curious"}</yuki-state>')
    assert body == "正文。"
    assert delta is None


def test_self_report_does_not_change_other_state_or_renew_support():
    c = controller()
    e = event()
    c.observe_committed_event(e)
    c.apply_semantic_observation(observation(e))
    p = c.advance(105, controller_epoch=0, host_available=True)
    c.observe_run_feedback(
        Feedback(run_ref="run", proposal_id=p.proposal_id, sequence=1, outcome="accepted", at=105)
    )
    belief = c.belief("topic", "A", 105)
    support = c.state.candidates[e.ref.event_id].support
    report = SelfReport(
        run_ref="run", sequence=1, response_id="r1", at=105, delta=SelfDelta(engage="join")
    )
    assert c.observe_self_report(report)
    assert not c.observe_self_report(report)
    assert c.belief("topic", "A", 105) == belief
    assert c.state.candidates[e.ref.event_id].support == support
    willingness = c._willingness(110)
    assert c.observe_self_report(
        SelfReport(
            run_ref="run", sequence=2, response_id="r2", at=110, delta=SelfDelta(mood="calm")
        )
    )
    assert c._willingness(110) == willingness
    assert not c.observe_self_report(report.model_copy(update={"run_ref": "unknown"}))
