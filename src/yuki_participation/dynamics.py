"""V6 equations with the default kappa=0; no learned psychological probabilities."""

from __future__ import annotations

import math

from .autonomy_parameters import DEFAULT_AUTONOMY_PARAMETERS, AutonomyParameters
from .models import Choice

Belief = tuple[float, float, float, float, float]  # O, H, Y, E, C
IDLE: Belief = (1, 0, 0, 0, 0)


def decay(b: Belief, seconds: float) -> Belief:
    tails = tuple(
        x * math.exp(-max(0, seconds) / tau)
        for x, tau in zip(b[1:], (120, 300, 240, 300), strict=True)
    )
    return (1 - sum(tails), tails[0], tails[1], tails[2], tails[3])


def observe(b: Belief, act: Choice, *, matched_self: bool, omega: float = 0.8) -> Belief:
    result = [(1 - omega) * x for x in b]
    for mark, weight in act.probabilities.items():
        target = None
        if mark == "invite_yuki":
            target = 1
        elif mark == "extend_yuki" and matched_self:
            target = 3
        elif mark in {"close_topic", "ask_yuki_stop"}:
            target = 4
        if target is None:
            for i, x in enumerate(b):
                result[i] += omega * weight * x
        else:
            result[target] += omega * weight
    return (result[0], result[1], result[2], result[3], result[4])


def self_expression(b: Belief) -> Belief:
    # O->Y, H->E; repeated Y never creates E and C is not reopened.
    return (0, 0, b[2] + b[0], b[3] + b[1], b[4])


def admitted_input(b: Belief) -> Belief:
    """A real Host-admitted human turn addresses Yuki, without a Jev distribution.

    It does not itself establish mutual engagement or release a closing boundary.
    """
    return (0, b[0] + b[1] + b[2], 0, b[3], b[4])


def attention(value: float, stimulus: float, seconds: float) -> float:
    return stimulus + (value - stimulus) * math.exp(-max(0, seconds) / 4)


def source_attention(weight: float, source_age: float, elapsed: float, tau: float) -> float:
    """Exact kappa=0 response to one exponentially decaying source, from x(0)=0."""
    dt = max(0, elapsed)
    initial = weight * math.exp(-max(0, source_age - dt) / tau)
    return max(0, initial * tau / (tau - 4) * (math.exp(-dt / tau) - math.exp(-dt / 4)))


def stimulus(sources: list[tuple[float, float]], now: float, tau: float) -> float:
    return 1 - math.prod(1 - w * math.exp(-max(0, now - at) / tau) for w, at in sources)


def aggregate_attention(sources: list[tuple[float, float, float]], now: float, tau: float) -> float:
    """Bounded convolution of the actual product stimulus, with source activation times.

    Single-source response is exact. Multiple sources use midpoint integration at
    0.125 seconds; the discarded response tail is at most exp(-20), below 2.1e-9.
    """
    if not sources:
        return 0.0
    if len(sources) == 1:
        weight, at, start = sources[0]
        return source_attention(weight, now - at, now - max(at, start), tau)
    start = max(min(max(at, issued) for _, at, issued in sources), now - 80)
    if start >= now:
        return 0.0
    boundaries = sorted(
        {
            start,
            now,
            *(max(at, issued) for _, at, issued in sources if start < max(at, issued) < now),
        }
    )
    value = 0.0
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        count = max(1, math.ceil((right - left) / 0.125))
        dt = (right - left) / count
        for index in range(count):
            t = left + (index + 0.5) * dt
            u = stimulus([(w, at) for w, at, issued in sources if max(at, issued) <= t], t, tau)
            value = attention(value, u, dt)
    return value


def participation_factor(tendency: float, *, minimum: float = 0.15) -> float:
    return minimum + (1 - minimum) * (max(-1, min(1, tendency)) + 1) / 2


def social_context(context: float, human_attention: float) -> float:
    """Fresh human evidence and slow group context are bounded alternative inputs."""
    return 1 - (1 - context) * (1 - human_attention)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def autonomy_pressure(
    *,
    tendency: float,
    activity: float,
    compute: float,
    no_reply: float,
    work_fast: float,
    work_slow: float,
    exposure: float,
    reception: float,
    parameters: AutonomyParameters = DEFAULT_AUTONOMY_PARAMETERS,
) -> float:
    """Continuous cost of starting another autonomous Work, never a message quota."""
    return (
        parameters.pressure_bias
        + parameters.tendency_weight * tendency
        - parameters.activity_weight * activity
        - parameters.compute_weight * compute
        - parameters.no_reply_weight * no_reply
        - parameters.work_fast_weight * work_fast
        - parameters.work_slow_weight * work_slow
        - parameters.exposure_weight * exposure
        + parameters.reception_weight * reception
    )


def source_rate(
    b: Belief,
    *,
    attention: float,
    familiarity: float,
    support: float,
    floor: float,
    gap: float,
    context: float,
    tendency: float,
    pressure: float,
    parameters: AutonomyParameters = DEFAULT_AUTONOMY_PARAMETERS,
) -> float:
    social = (
        participation_factor(tendency, minimum=parameters.source_participation_minimum)
        * attention
        / (1 + parameters.source_familiarity_weight * familiarity)
        + parameters.source_invitation_weight * b[1]
        + parameters.source_extension_weight * b[3]
        - parameters.source_self_weight * b[2]
        - parameters.source_closed_weight * b[4]
    )
    opening = (1 - math.exp(-max(0.0, gap) / parameters.source_opening_seconds)) * (
        parameters.source_opening_floor + (1 - parameters.source_opening_floor) * floor
    )
    return (
        (1 / parameters.source_interval_seconds)
        * support
        * opening
        * context
        * sigmoid(pressure + parameters.source_social_weight * social)
    )


def intrinsic_rate(
    *,
    human_activity: float,
    seconds_since_human: float | None,
    pressure: float,
    parameters: AutonomyParameters = DEFAULT_AUTONOMY_PARAMETERS,
) -> float:
    gap = math.inf if seconds_since_human is None else max(0.0, seconds_since_human)
    activity = human_activity / (human_activity + parameters.human_activity_half_saturation)
    silence = (-math.expm1(-gap / parameters.silence_rise_seconds)) * math.exp(
        -gap / parameters.silence_decay_seconds
    )
    return (1 / parameters.intrinsic_interval_seconds) * activity * silence * sigmoid(pressure)
