from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from lekit.teleoperators.isaac_teleop.protocol import TeleopFrame, neutral_action
from lekit.teleoperators.isaac_teleop.subscriber import (
    IsaacTeleopNodeConfig,
    IsaacTeleopNodeSubscriber,
)


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeReceiver:
    def __init__(self, frames=()) -> None:
        self.frames = deque(frames)
        self.closed = False

    def receive_latest(self, *, timeout_s: float = 0.0) -> TeleopFrame | None:
        del timeout_s
        return self.frames.popleft() if self.frames else None

    def close(self) -> None:
        self.closed = True


def action(*, squeeze: float = 0.0, engaged: bool = False, trigger: float = 0.0):
    result = neutral_action()
    for side in ("left", "right"):
        result[f"{side}.squeeze"] = squeeze
        result[f"{side}.is_tracking"] = True
        result[f"{side}.is_aim_tracking"] = True
        result[f"{side}.is_engaged"] = engaged
    result["right.trigger"] = trigger
    return result


def frame(sequence: int, *, session: str = "session-1", **action_kwargs) -> TeleopFrame:
    return TeleopFrame(session, sequence, sequence, sequence, action(**action_kwargs))


def subscriber(tmp_path, receiver: FakeReceiver, clock: ManualClock, **config_kwargs):
    config = IsaacTeleopNodeConfig(
        endpoint="tcp://127.0.0.1:5557",
        calibration_dir=tmp_path,
        first_frame_timeout_s=0.05,
        stale_after_s=0.25,
        **config_kwargs,
    )
    return IsaacTeleopNodeSubscriber(config, receiver_factory=lambda _endpoint: receiver, clock=clock)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint": "http://localhost:5557"}, "endpoint"),
        ({"first_frame_timeout_s": 0.0}, "first_frame_timeout_s"),
        ({"stale_after_s": -1.0}, "stale_after_s"),
        ({"rearm_squeeze_threshold": 1.1}, "rearm_squeeze_threshold"),
    ],
)
def test_subscriber_config_rejects_unsafe_values(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IsaacTeleopNodeConfig(**kwargs)


def test_connect_waits_for_first_valid_frame_and_exposes_standard_features(tmp_path) -> None:
    receiver = FakeReceiver([frame(7, trigger=0.4)])
    clock = ManualClock()
    teleop = subscriber(tmp_path, receiver, clock)

    teleop.connect()

    assert teleop.is_connected
    assert set(teleop.get_action()) == set(teleop.action_features)
    assert teleop.get_action()["right.trigger"] == pytest.approx(0.4)
    teleop.disconnect()
    assert receiver.closed


def test_connect_timeout_closes_receiver(tmp_path) -> None:
    receiver = FakeReceiver()
    clock = ManualClock()

    def advancing_clock() -> float:
        clock.now += 0.02
        return clock.now

    config = IsaacTeleopNodeConfig(
        endpoint="tcp://127.0.0.1:5557",
        calibration_dir=tmp_path,
        first_frame_timeout_s=0.05,
    )
    teleop = IsaacTeleopNodeSubscriber(
        config,
        receiver_factory=lambda _endpoint: receiver,
        clock=advancing_clock,
    )

    with pytest.raises(TimeoutError, match="first teleop frame"):
        teleop.connect()

    assert not teleop.is_connected
    assert receiver.closed


def test_disconnect_invalidates_state_when_receiver_close_fails(tmp_path) -> None:
    class CloseFailureReceiver(FakeReceiver):
        def close(self) -> None:
            super().close()
            raise RuntimeError("receiver close failed")

    receiver = CloseFailureReceiver([frame(0, squeeze=0.0), frame(1, squeeze=0.8, engaged=True)])
    teleop = subscriber(tmp_path, receiver, ManualClock())
    teleop.connect()
    assert teleop.get_action()["right.is_engaged"] is True

    with pytest.raises(RuntimeError, match="receiver close failed"):
        teleop.disconnect()

    assert not teleop.is_connected
    assert teleop.latest_frame is None


def test_failed_close_still_requires_release_after_reconnect(tmp_path) -> None:
    class CloseFailureReceiver(FakeReceiver):
        def close(self) -> None:
            super().close()
            raise RuntimeError("receiver close failed")

    first = CloseFailureReceiver([frame(0, squeeze=0.0), frame(1, squeeze=0.8, engaged=True)])
    second = FakeReceiver([frame(0, squeeze=0.8, engaged=True, trigger=0.9)])
    receivers = iter((first, second))
    config = IsaacTeleopNodeConfig(
        endpoint="tcp://127.0.0.1:5557",
        calibration_dir=tmp_path,
        first_frame_timeout_s=0.05,
        stale_after_s=0.25,
    )
    teleop = IsaacTeleopNodeSubscriber(
        config,
        receiver_factory=lambda _endpoint: next(receivers),
        clock=ManualClock(),
    )
    teleop.connect()
    assert teleop.get_action()["right.is_engaged"] is True

    with pytest.raises(RuntimeError, match="receiver close failed"):
        teleop.disconnect()

    teleop.connect()
    inhibited = teleop.get_action()
    assert inhibited["right.is_tracking"] is False
    assert inhibited["right.is_engaged"] is False

    second.frames.extend(
        [
            frame(1, squeeze=0.0, engaged=False, trigger=0.2),
            frame(2, squeeze=0.8, engaged=True, trigger=0.6),
        ]
    )
    released = teleop.get_action()
    assert released["right.is_tracking"] is True
    assert released["right.is_engaged"] is False

    reengaged = teleop.get_action()
    assert reengaged["right.is_engaged"] is True
    assert reengaged["right.trigger"] == pytest.approx(0.6)


def test_connect_timeout_preserves_primary_when_receiver_close_fails(tmp_path) -> None:
    class CloseFailureReceiver(FakeReceiver):
        def close(self) -> None:
            super().close()
            raise RuntimeError("receiver close failed")

    receiver = CloseFailureReceiver()
    clock = ManualClock()

    def advancing_clock() -> float:
        clock.now += 0.02
        return clock.now

    config = IsaacTeleopNodeConfig(
        endpoint="tcp://127.0.0.1:5557",
        calibration_dir=tmp_path,
        first_frame_timeout_s=0.05,
    )
    teleop = IsaacTeleopNodeSubscriber(
        config,
        receiver_factory=lambda _endpoint: receiver,
        clock=advancing_clock,
    )

    with pytest.raises(TimeoutError, match="first teleop frame"):
        teleop.connect()

    assert not teleop.is_connected
    assert teleop.latest_frame is None


def test_stale_frame_returns_a_complete_neutral_action(tmp_path) -> None:
    receiver = FakeReceiver([frame(0, squeeze=0.0), frame(1, squeeze=0.8, engaged=True)])
    clock = ManualClock()
    teleop = subscriber(tmp_path, receiver, clock)
    teleop.connect()
    assert teleop.get_action()["right.is_engaged"] is True

    clock.now = 0.251
    stale = teleop.get_action()

    assert stale["left.is_tracking"] is False
    assert stale["right.is_tracking"] is False
    assert stale["left.is_engaged"] is False
    assert stale["right.is_engaged"] is False
    np.testing.assert_array_equal(stale["right.translation"], [0.0, 0.0, 0.0])


def test_sequence_rollback_requires_release_before_actions_can_reengage(tmp_path) -> None:
    receiver = FakeReceiver(
        [
            frame(10, squeeze=0.0),
            frame(11, squeeze=0.8, engaged=True, trigger=0.1),
            frame(0, squeeze=0.8, engaged=True, trigger=0.9),
            frame(1, squeeze=0.8, engaged=True, trigger=0.8),
            frame(2, squeeze=0.0, engaged=False, trigger=0.2),
            frame(3, squeeze=0.8, engaged=True, trigger=0.6),
        ]
    )
    clock = ManualClock()
    teleop = subscriber(tmp_path, receiver, clock)
    teleop.connect()

    assert teleop.get_action()["right.trigger"] == pytest.approx(0.1)

    reset = teleop.get_action()
    assert reset["right.is_tracking"] is False
    assert reset["right.is_engaged"] is False
    assert reset["right.trigger"] == 0.0

    held_after_reset = teleop.get_action()
    assert held_after_reset["right.is_tracking"] is False
    assert held_after_reset["right.is_engaged"] is False

    released = teleop.get_action()
    assert released["right.is_tracking"] is True
    assert released["right.is_engaged"] is False
    assert released["right.trigger"] == pytest.approx(0.2)

    reengaged = teleop.get_action()
    assert reengaged["right.is_engaged"] is True
    assert reengaged["right.trigger"] == pytest.approx(0.6)


def test_new_session_requires_release_before_actions_can_reengage(tmp_path) -> None:
    receiver = FakeReceiver(
        [
            frame(0, session="old", squeeze=0.0),
            frame(0, session="new", squeeze=0.8, engaged=True, trigger=0.8),
            frame(1, session="new", squeeze=0.0, engaged=False, trigger=0.1),
            frame(2, session="new", squeeze=0.8, engaged=True, trigger=0.6),
        ]
    )
    clock = ManualClock()
    teleop = subscriber(tmp_path, receiver, clock)
    teleop.connect()

    inhibited = teleop.get_action()
    assert inhibited["right.is_tracking"] is False
    assert inhibited["right.is_engaged"] is False
    assert inhibited["right.trigger"] == 0.0

    released = teleop.get_action()
    assert released["right.is_tracking"] is True
    assert released["right.is_engaged"] is False
    assert released["right.trigger"] == pytest.approx(0.1)

    reengaged = teleop.get_action()
    assert reengaged["right.is_engaged"] is True
    assert reengaged["right.trigger"] == pytest.approx(0.6)


def test_stream_must_be_released_before_reengaging_after_stale_data(tmp_path) -> None:
    receiver = FakeReceiver([frame(0, squeeze=0.0), frame(1, squeeze=0.8, engaged=True)])
    clock = ManualClock()
    teleop = subscriber(tmp_path, receiver, clock)
    teleop.connect()
    assert teleop.get_action()["right.is_engaged"] is True

    clock.now = 0.251
    assert teleop.get_action()["right.is_engaged"] is False

    receiver.frames.extend(
        [
            frame(2, squeeze=0.8, engaged=True, trigger=0.8),
            frame(3, squeeze=0.0, engaged=False, trigger=0.1),
            frame(4, squeeze=0.8, engaged=True, trigger=0.6),
        ]
    )
    held_after_recovery = teleop.get_action()
    assert held_after_recovery["right.is_tracking"] is False
    assert held_after_recovery["right.trigger"] == 0.0
    assert teleop.get_action()["right.is_engaged"] is False
    assert teleop.get_action()["right.is_engaged"] is True
