from __future__ import annotations

import numpy as np
import pytest

from lekit.robots.jaka_robot.servo_p_debug import (
    DiagnosticSettings,
    _set_cartesian_nlf,
    build_targets,
    run_stream,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


class FakeRC:
    def __init__(self, clock: FakeClock, *, latency_s: float = 0.001, queue_depth: int = 2) -> None:
        self.clock = clock
        self.latency_s = latency_s
        self.queue_depth = queue_depth
        self.calls: list[tuple] = []

    def servo_p(self, pose, move_mode, step_num):
        self.calls.append((pose, move_mode, step_num))
        self.clock.now += self.latency_s
        return (0, self.queue_depth)


def test_cartesian_nlf_uses_jaka_sdk_units_and_can_be_restored():
    calls = []

    class FilterRC:
        def servo_move_use_carte_NLF(self, *args):  # noqa: N802
            calls.append(("cartesian_nlf", *args))
            return (0,)

        def servo_move_use_none_filter(self):
            calls.append(("none",))
            return (0,)

    rc = FilterRC()
    _set_cartesian_nlf(rc, True)
    _set_cartesian_nlf(rc, False)

    assert calls == [
        ("cartesian_nlf", 50.0, 200.0, 1000.0, 0.5, 1.0, 10.0),
        ("none",),
    ]


def test_hold_and_z_bump_targets_are_precomputed_and_return_to_start():
    initial = np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3])
    hold = build_targets(
        initial,
        DiagnosticSettings(mode="hold", duration_s=0.032, spin_threshold_s=0.0),
    )
    assert len(hold) == 5
    assert all(np.array_equal(target, initial) for target in hold)

    bump = build_targets(
        initial,
        DiagnosticSettings(
            mode="z-bump",
            duration_s=0.032,
            settle_s=0.008,
            amplitude_mm=1.0,
            spin_threshold_s=0.0,
        ),
    )
    assert np.array_equal(bump[0], initial)
    assert np.array_equal(bump[-1], initial)
    assert max(target[2] for target in bump) == pytest.approx(301.0)
    assert all(np.array_equal(target[[0, 1, 3, 4, 5]], initial[[0, 1, 3, 4, 5]]) for target in bump)


def test_stream_calls_only_raw_servo_p_at_cumulative_eight_ms_deadlines():
    clock = FakeClock()
    rc = FakeRC(clock)
    settings = DiagnosticSettings(mode="hold", duration_s=0.032, spin_threshold_s=0.0)
    targets = build_targets((100.0, 200.0, 300.0, 0.1, 0.2, 0.3), settings)

    samples = run_stream(rc, targets, settings, clock=clock, sleep=clock.sleep)

    assert len(rc.calls) == len(targets)
    assert all(call[1:] == (0, 1) for call in rc.calls)
    assert [sample.started_s for sample in samples] == pytest.approx(
        [index * 0.008 for index in range(len(samples))]
    )
    assert [sample.period_s for sample in samples[1:]] == pytest.approx([0.008] * 4)
    assert all(sample.sdk_latency_s == pytest.approx(0.001) for sample in samples)


def test_stream_aborts_on_queue_growth_or_repeated_timing_overruns():
    clock = FakeClock()
    queue_full_rc = FakeRC(clock, queue_depth=80)
    settings = DiagnosticSettings(mode="hold", duration_s=0.032, spin_threshold_s=0.0)
    targets = build_targets((100.0, 200.0, 300.0, 0.1, 0.2, 0.3), settings)
    with pytest.raises(RuntimeError, match="queue depth reached 80"):
        run_stream(queue_full_rc, targets, settings, clock=clock, sleep=clock.sleep)

    clock = FakeClock()
    slow_rc = FakeRC(clock, latency_s=0.020)
    overrun_settings = DiagnosticSettings(
        mode="hold",
        duration_s=0.050,
        max_consecutive_overruns=2,
        spin_threshold_s=0.0,
    )
    targets = build_targets((100.0, 200.0, 300.0, 0.1, 0.2, 0.3), overrun_settings)
    with pytest.raises(RuntimeError, match="2 consecutive overruns"):
        run_stream(slow_rc, targets, overrun_settings, clock=clock, sleep=clock.sleep)


def test_stream_allows_sdk_latency_between_period_and_overrun_threshold():
    clock = FakeClock()
    rc = FakeRC(clock, latency_s=0.009)
    settings = DiagnosticSettings(
        mode="hold",
        duration_s=0.050,
        max_consecutive_overruns=2,
        spin_threshold_s=0.0,
    )
    targets = build_targets((100.0, 200.0, 300.0, 0.1, 0.2, 0.3), settings)

    samples = run_stream(rc, targets, settings, clock=clock, sleep=clock.sleep)

    assert len(samples) == len(targets)
