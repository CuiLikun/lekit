from __future__ import annotations

from collections import deque

import pytest

from lekit.teleoperators.isaac_teleop.protocol import ACTION_KEYS, decode_action_frame, neutral_action
from lekit.teleoperators.isaac_teleop.teleop_node import TeleopNode, TeleopNodeConfig


class ManualTime:
    def __init__(self) -> None:
        self.now = 10.0
        self.utc = 1_000_000_000

    def monotonic(self) -> float:
        return self.now

    def time_ns(self) -> int:
        return self.utc

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.utc += int(seconds * 1_000_000_000)


class FakePublisher:
    def __init__(self, publish_results=()) -> None:
        self.frames = []
        self.statuses = []
        self.closed = False
        self.publish_results = deque(publish_results)

    def publish_action(self, frame) -> bool:
        self.frames.append(frame)
        return self.publish_results.popleft() if self.publish_results else True

    def publish_status(self, status) -> bool:
        self.statuses.append(dict(status))
        return True

    def close(self) -> None:
        self.closed = True


class FakeController:
    def __init__(self, samples, *, notify_waiting: bool = False) -> None:
        self.samples = deque(samples)
        self.notify_waiting = notify_waiting
        self.wait_callback = None
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def set_connect_wait_callback(self, callback) -> None:
        self.wait_callback = callback

    def connect(self) -> None:
        self.connect_calls += 1
        if self.notify_waiting:
            self.wait_callback()
        self.is_connected = True

    def get_action(self):
        sample = self.samples.popleft()
        if isinstance(sample, BaseException):
            raise sample
        return sample

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class FakeMonitor:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FakeControlNode:
    def __init__(self, *, published: bool = True) -> None:
        self.events: list[str] = []
        self.published = []
        self._published = published

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def publish(self, payload: bytes, *, captured_monotonic_ns: int, captured_utc_ns: int) -> bool:
        self.published.append(
            (
                payload,
                {
                    "captured_monotonic_ns": captured_monotonic_ns,
                    "captured_utc_ns": captured_utc_ns,
                },
            )
        )
        return self._published


def tracked_action(trigger: float = 0.0):
    action = neutral_action()
    action["right.trigger"] = trigger
    action["right.is_tracking"] = True
    return action


def make_node(config, controllers, publisher, clock) -> TeleopNode:
    queue = deque(controllers)
    return TeleopNode(
        config,
        controller_factory=lambda _config: queue.popleft(),
        publisher_factory=lambda _endpoint: publisher,
        monitor_factory=lambda _state, _host, _port: FakeMonitor(),
        monotonic=clock.monotonic,
        utc_ns=clock.time_ns,
        sleep=clock.sleep,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"publish_endpoint": "http://127.0.0.1:5557"}, "publish_endpoint"),
        ({"rate_hz": 0.0}, "rate_hz"),
        ({"retry_delay_s": -1.0}, "retry_delay_s"),
        ({"monitor_port": 0}, "monitor_port"),
    ],
)
def test_node_config_rejects_invalid_runtime_values(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TeleopNodeConfig(**kwargs)


def test_node_publishes_atomic_sequenced_frames_and_updates_metrics() -> None:
    publisher = FakePublisher()
    controller = FakeController([tracked_action(0.1), tracked_action(0.2)], notify_waiting=True)
    clock = ManualTime()
    config = TeleopNodeConfig(rate_hz=10.0, monitor_enabled=False)
    node = make_node(config, [controller], publisher, clock)

    node.run(max_frames=2)

    assert controller.connect_calls == 1
    assert controller.disconnect_calls == 1
    assert publisher.closed
    assert [item.sequence for item in publisher.frames] == [0, 1]
    assert len({item.session_id for item in publisher.frames}) == 1
    assert all(set(item.action) == set(ACTION_KEYS) for item in publisher.frames)
    assert publisher.frames[1].action["right.trigger"] == pytest.approx(0.2)
    assert any(status["state"] == "waiting_for_headset" for status in publisher.statuses)
    assert any(status["state"] == "streaming" for status in publisher.statuses)
    assert node.state.snapshot().published_frames == 2
    assert node.state.snapshot().publish_rate_hz == pytest.approx(10.0)
    assert node.state.snapshot().state == "stopped"


def test_successful_headset_connection_prints_monitor_page(capsys) -> None:
    publisher = FakePublisher()
    controller = FakeController([tracked_action()])
    clock = ManualTime()
    config = TeleopNodeConfig(monitor_host="192.168.5.24", monitor_port=8000)
    node = make_node(config, [controller], publisher, clock)

    node.run(max_frames=1)

    output = capsys.readouterr().out
    assert output.count("Quest 3 connected; controller input is streaming.") == 1
    assert "Open monitor: http://192.168.5.24:8000" in output


def test_successful_reconnection_prints_a_new_connection_message(capsys) -> None:
    publisher = FakePublisher()
    first = FakeController([RuntimeError("OpenXR IPC disconnected")])
    second = FakeController([tracked_action()])
    clock = ManualTime()
    config = TeleopNodeConfig(retry_delay_s=0.0, monitor_enabled=False)
    node = make_node(config, [first, second], publisher, clock)

    node.run(max_frames=1)

    output = capsys.readouterr().out
    assert output.count("Quest 3 connected; controller input is streaming.") == 2


def test_connection_message_omits_monitor_page_when_monitor_is_disabled(capsys) -> None:
    publisher = FakePublisher()
    controller = FakeController([tracked_action()])
    clock = ManualTime()
    config = TeleopNodeConfig(monitor_enabled=False)
    node = make_node(config, [controller], publisher, clock)

    node.run(max_frames=1)

    output = capsys.readouterr().out
    assert "Quest 3 connected; controller input is streaming." in output
    assert "Open monitor:" not in output


def test_xr_failure_reconnects_with_a_new_session_and_sequence() -> None:
    publisher = FakePublisher()
    first = FakeController([tracked_action(0.1), RuntimeError("OpenXR IPC disconnected")])
    second = FakeController([tracked_action(0.2)])
    clock = ManualTime()
    config = TeleopNodeConfig(rate_hz=20.0, retry_delay_s=0.1, monitor_enabled=False)
    node = make_node(config, [first, second], publisher, clock)

    node.run(max_frames=2)

    assert first.disconnect_calls == 1
    assert second.disconnect_calls == 1
    assert [item.sequence for item in publisher.frames] == [0, 0]
    assert publisher.frames[0].session_id != publisher.frames[1].session_id
    reconnecting = [status for status in publisher.statuses if status["state"] == "reconnecting"]
    assert reconnecting
    assert reconnecting[-1]["last_error"] == "OpenXR IPC disconnected"


def test_zero_max_frames_starts_and_stops_without_opening_xr() -> None:
    publisher = FakePublisher()
    clock = ManualTime()
    config = TeleopNodeConfig(monitor_enabled=False)
    node = make_node(config, [], publisher, clock)

    node.run(max_frames=0)

    assert publisher.frames == []
    assert publisher.closed
    assert node.state.snapshot().state == "stopped"


def test_stop_request_during_headset_wait_exits_without_reconnecting() -> None:
    publisher = FakePublisher()
    stop_event = __import__("threading").Event()

    class WaitingController(FakeController):
        def __init__(self) -> None:
            super().__init__([])
            self.get_action_calls = 0

        def connect(self) -> None:
            self.connect_calls += 1
            stop_event.set()
            self.wait_callback()

        def get_action(self):
            self.get_action_calls += 1
            raise AssertionError("sampling must not begin after stop was requested")

    controller = WaitingController()
    clock = ManualTime()
    node = make_node(TeleopNodeConfig(monitor_enabled=False), [controller], publisher, clock)

    node.run(stop_event=stop_event)

    assert controller.get_action_calls == 0
    assert not any(status["state"] == "reconnecting" for status in publisher.statuses)
    assert node.state.snapshot().state == "stopped"


def test_dropped_transport_frames_are_visible_and_not_counted_as_published() -> None:
    publisher = FakePublisher([False, True])
    controller = FakeController([tracked_action(0.1), tracked_action(0.2)])
    clock = ManualTime()
    node = make_node(TeleopNodeConfig(rate_hz=10.0, monitor_enabled=False), [controller], publisher, clock)

    node.run(max_frames=2)

    snapshot = node.state.snapshot()
    assert snapshot.sampled_frames == 2
    assert snapshot.published_frames == 1
    assert snapshot.dropped_frames == 1
    assert snapshot.sequence == 1


def test_managed_teleop_publishes_encoded_frame_with_exact_capture_metadata() -> None:
    control_node = FakeControlNode()
    controller = FakeController([tracked_action(0.3)])
    clock = ManualTime()
    node = TeleopNode(
        TeleopNodeConfig(monitor_enabled=False),
        control_node=control_node,
        controller_factory=lambda _config: controller,
        monotonic=clock.monotonic,
        utc_ns=clock.time_ns,
        sleep=clock.sleep,
    )

    node.run(max_frames=1)

    payload, metadata = control_node.published[0]
    decoded = decode_action_frame(payload)
    assert set(decoded.action) == set(neutral_action())
    assert metadata["captured_monotonic_ns"] == decoded.captured_monotonic_ns
    assert metadata["captured_utc_ns"] == decoded.captured_utc_ns
    assert control_node.events == ["start", "stop"]


def test_managed_teleop_stops_control_node_after_xr_connect_failure() -> None:
    control_node = FakeControlNode()
    stop_event = __import__("threading").Event()
    clock = ManualTime()

    class FailingController(FakeController):
        def __init__(self) -> None:
            super().__init__([])

        def connect(self) -> None:
            self.connect_calls += 1
            stop_event.set()
            raise RuntimeError("XR unavailable")

    node = TeleopNode(
        TeleopNodeConfig(monitor_enabled=False),
        control_node=control_node,
        controller_factory=lambda _config: FailingController(),
        monotonic=clock.monotonic,
        utc_ns=clock.time_ns,
        sleep=clock.sleep,
    )

    node.run(max_frames=1, stop_event=stop_event)

    assert control_node.events == ["start", "stop"]
