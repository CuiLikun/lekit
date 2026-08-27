from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from lekit.control.model import (
    PROTOCOL_VERSION,
    ActionEnvelope,
    ControlHandle,
    ManagementMessage,
    NodeRole,
    RobotControlState,
    TimingConfig,
)
from lekit.control.robot import HoldResult, PassiveHold, RobotNode, RobotNodeConfig
from lekit.control.runtime import ReceivedAction, ReceivedManagement
from lerobot.robots.robot import Robot


class Clock:
    def __init__(self) -> None:
        self.monotonic = 0
        self.utc = 1_000

    def monotonic_ns(self) -> int:
        return self.monotonic

    def utc_ns(self) -> int:
        return self.utc

    def advance(self, seconds: float) -> None:
        delta = round(seconds * 1_000_000_000)
        self.monotonic += delta
        self.utc += delta


class FakeRobot(Robot):
    name = "fake"
    config_class = object

    def __init__(self) -> None:
        self.connected = False
        self.x = 0.25
        self.sent: list[dict[str, float]] = []
        self.calls: list[str] = []
        self.observation_error: Exception | None = None
        self.send_error: Exception | None = None
        self.disconnect_error: Exception | None = None

    @property
    def observation_features(self) -> dict[str, type]:
        return {"x": float}

    @property
    def action_features(self) -> dict[str, type]:
        return {"x": float}

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self.calls.append("connect")
        self.connected = True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, float]:
        self.calls.append("get_observation")
        if self.observation_error is not None:
            raise self.observation_error
        return {"x": self.x}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.calls.append("send_action")
        if self.send_error is not None:
            raise self.send_error
        sent = dict(action)
        self.sent.append(sent)
        return sent

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.connected = False
        if self.disconnect_error is not None:
            raise self.disconnect_error


class FakeProcessor:
    def __init__(self) -> None:
        self.reset_count = 0
        self.calls: list[tuple[bytes, dict[str, float]]] = []
        self.outputs: deque[dict[str, float] | Exception] = deque()
        self.state = "unarmed"
        self.tracking = True
        self.engaged = False
        self.fault: str | None = None
        self.reset_error: Exception | None = None
        self.call_hook = None

    def __call__(self, payload: bytes, observation: dict[str, float]) -> dict[str, float]:
        self.calls.append((payload, dict(observation)))
        if self.call_hook is not None:
            self.call_hook()
        output = self.outputs.popleft() if self.outputs else {"x": float(payload.decode())}
        if isinstance(output, Exception):
            raise output
        return dict(output)

    def reset(self) -> None:
        self.reset_count += 1
        if self.reset_error is not None:
            raise self.reset_error
        self.state = "unarmed"

    def status(self) -> dict[str, object]:
        return {
            "processor_state": self.state,
            "tracking": self.tracking,
            "engaged": self.engaged,
            "error": self.fault,
        }


class RecordingReceiver:
    def __init__(self) -> None:
        self.frames: deque[ReceivedAction] = deque()
        self.timeouts: list[float] = []
        self.closed = False
        self.close_error: Exception | None = None

    def receive_latest(self, *, timeout_s: float = 0.0) -> ReceivedAction | None:
        self.timeouts.append(timeout_s)
        if self.closed or not self.frames:
            return None
        latest = self.frames[-1]
        self.frames.clear()
        return latest

    def close(self) -> None:
        if self.close_error is not None:
            raise self.close_error
        self.closed = True
        self.frames.clear()


class RecordingNodeChannel:
    def __init__(self) -> None:
        self.inbox: deque[ReceivedManagement] = deque()
        self.sent: list[ManagementMessage] = []
        self.send_result = True
        self.send_error: Exception | None = None
        self.closed = False
        self.close_error: Exception | None = None
        self.condition = threading.Condition()

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        del timeout_s
        with self.condition:
            if self.closed or not self.inbox:
                return None
            return self.inbox.popleft()

    def send(self, message: ManagementMessage) -> bool:
        self.sent.append(message)
        if self.send_error is not None:
            raise self.send_error
        return self.send_result and not self.closed

    def close(self) -> None:
        with self.condition:
            if self.close_error is not None:
                raise self.close_error
            self.closed = True
            self.condition.notify_all()


class RecordingRuntime:
    def __init__(self) -> None:
        self.channels: list[RecordingNodeChannel] = []
        self.receivers: list[RecordingReceiver] = []
        self.receiver_endpoints: list[str] = []
        self.closed = False
        self.open_node_error: Exception | None = None
        self.node_send_result = True
        self.node_send_error: Exception | None = None
        self.node_close_error: Exception | None = None

    def open_node(self, node_id: str, session_id: str, *, hub_seed: str | None) -> RecordingNodeChannel:
        del node_id, session_id, hub_seed
        if self.open_node_error is not None:
            raise self.open_node_error
        channel = RecordingNodeChannel()
        channel.send_result = self.node_send_result
        channel.send_error = self.node_send_error
        channel.close_error = self.node_close_error
        self.channels.append(channel)
        return channel

    def open_action_receiver(self, endpoint: str) -> RecordingReceiver:
        self.receiver_endpoints.append(endpoint)
        receiver = RecordingReceiver()
        self.receivers.append(receiver)
        return receiver

    def open_action_publisher(self, endpoint: str):
        del endpoint
        raise AssertionError("RobotNode must not open an action publisher")

    def open_hub(self, endpoint: str, *, hub_epoch: str, advertise_endpoint: str | None = None):
        del endpoint, hub_epoch, advertise_endpoint
        raise AssertionError("RobotNode must not open a Hub channel")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def runtime() -> RecordingRuntime:
    return RecordingRuntime()


@pytest.fixture
def robot() -> FakeRobot:
    return FakeRobot()


@pytest.fixture
def processor() -> FakeProcessor:
    return FakeProcessor()


@pytest.fixture
def robot_node(
    tmp_path: Path,
    clock: Clock,
    runtime: RecordingRuntime,
    robot: FakeRobot,
    processor: FakeProcessor,
) -> RobotNode:
    node_id_path = tmp_path / "robot-id"
    node_id_path.write_text("robot-1\n", encoding="utf-8")
    node = RobotNode(
        robot,
        processor,
        RobotNodeConfig(
            node_id_path=node_id_path,
            display_name="Robot",
            accepted_payload_schemas=("lekit.action.v1",),
            timing=TimingConfig(handle_ttl_s=3.0),
        ),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    node.start()
    node.receive_management(registered(node, runtime))
    yield node
    node.stop()


@pytest.fixture
def handle(robot_node: RobotNode, clock: Clock) -> ControlHandle:
    return ControlHandle(
        handle_id="handle-1",
        hub_epoch="hub-epoch-1",
        robot_id=robot_node.node_id,
        robot_session_id=robot_node.session_id,
        controller_id="controller-1",
        controller_session_id="controller-session-1",
        controller_action_endpoint="memory://controller-actions",
        action_schema="lekit.action.v1",
        control_mode="teleop",
        fencing_token=7,
        issued_at_ns=clock.utc_ns(),
        expires_at_ns=clock.utc_ns() + 10_000_000_000,
    )


def messages(runtime: RecordingRuntime, kind: str) -> list[ManagementMessage]:
    return [message for channel in runtime.channels for message in channel.sent if message.kind == kind]


def registered(
    node: RobotNode,
    runtime: RecordingRuntime,
    *,
    epoch: str = "hub-epoch-1",
    correlation_id: str | None = None,
    sequence: int = 0,
) -> ManagementMessage:
    registration = messages(runtime, "register")[-1]
    return ManagementMessage(
        protocol_version=PROTOCOL_VERSION,
        kind="registered",
        correlation_id=correlation_id or registration.correlation_id,
        sender_id="hub",
        sender_session_id=epoch,
        sequence=sequence,
        sent_at_ns=1,
        body={"hub_epoch": epoch},
    )


def command(
    kind: str,
    handle: ControlHandle | None,
    *,
    correlation_id: str = "command-1",
    sequence: int = 1,
    reason: str | None = None,
) -> ManagementMessage:
    epoch = "hub-epoch-1" if handle is None else handle.hub_epoch
    body = {"hub_epoch": epoch}
    if handle is not None:
        body["handle"] = asdict(handle)
    if reason is not None:
        body["reason"] = reason
    return ManagementMessage(
        protocol_version=PROTOCOL_VERSION,
        kind=kind,
        correlation_id=correlation_id,
        sender_id="hub",
        sender_session_id=epoch,
        sequence=sequence,
        sent_at_ns=1,
        body=body,
    )


def envelope(handle: ControlHandle, **changes) -> ActionEnvelope:
    values = {
        "handle_id": handle.handle_id,
        "hub_epoch": handle.hub_epoch,
        "fencing_token": handle.fencing_token,
        "controller_id": handle.controller_id,
        "controller_session_id": handle.controller_session_id,
        "stream_session_id": "stream-1",
        "sequence": 0,
        "captured_monotonic_ns": 0,
        "captured_utc_ns": 0,
        "payload_schema": handle.action_schema,
        "payload": b"0.5",
    }
    values.update(changes)
    return ActionEnvelope(**values)


def activate(node: RobotNode, runtime: RecordingRuntime, handle: ControlHandle) -> RecordingReceiver:
    assert node.receive_management(command("grant", handle, sequence=1))
    assert node.receive_management(command("take_over", handle, sequence=2))
    return runtime.receivers[-1]


def inject(receiver: RecordingReceiver, frame: ActionEnvelope, clock: Clock, *, age_s: float = 0.0) -> None:
    receiver.frames.append(ReceivedAction(frame, clock.monotonic_ns() - round(age_s * 1_000_000_000)))


def unregistered_node(
    tmp_path: Path,
    clock: Clock,
    runtime: RecordingRuntime,
    robot: FakeRobot,
    processor: FakeProcessor,
    *,
    hold=None,
) -> RobotNode:
    node = RobotNode(
        robot,
        processor,
        RobotNodeConfig(
            node_id_path=tmp_path / "unregistered-robot-id",
            display_name="Unregistered Robot",
            accepted_payload_schemas=("lekit.action.v1",),
        ),
        runtime=runtime,
        hold=hold,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    node.start()
    return node


def test_robot_public_types_are_exported() -> None:
    import lekit.control as control

    assert control.RobotNode is RobotNode
    assert control.RobotNodeConfig is RobotNodeConfig
    assert control.PassiveHold is PassiveHold
    assert control.HoldResult is HoldResult


def test_hold_result_carries_adapter_diagnostic_detail() -> None:
    result = HoldResult(active=False, detail="feedback incomplete")

    assert result.detail == "feedback incomplete"


def test_start_connects_once_registers_real_features_and_starts_in_hold(
    robot_node: RobotNode, robot: FakeRobot, runtime: RecordingRuntime
) -> None:
    robot_node.start()
    descriptor = messages(runtime, "register")[0].body["descriptor"]

    assert robot.calls == ["connect"]
    assert descriptor["role"] == NodeRole.ROBOT.value
    assert descriptor["observation_features"] == {"x": "float"}
    assert descriptor["action_features"] == {"x": "float"}
    assert descriptor["administratively_enabled"] is True
    assert robot_node.control_state is RobotControlState.HOLD


def test_grant_is_only_cached_and_take_over_opens_receiver_while_held(
    robot_node: RobotNode,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    handle: ControlHandle,
) -> None:
    assert robot_node.receive_management(command("grant", handle, sequence=1))
    assert runtime.receivers == []
    assert robot_node.control_state is RobotControlState.HOLD

    assert robot_node.receive_management(command("take_over", handle, sequence=2))

    assert runtime.receiver_endpoints == [handle.controller_action_endpoint]
    assert processor.reset_count == 1
    assert robot_node.control_state is RobotControlState.HOLD
    assert messages(runtime, "robot_ready")[-1].body["handle"]["handle_id"] == handle.handle_id


def test_grant_and_take_over_are_rejected_before_exact_registration_ack(
    tmp_path: Path,
    clock: Clock,
    runtime: RecordingRuntime,
    robot: FakeRobot,
    processor: FakeProcessor,
) -> None:
    node = unregistered_node(tmp_path, clock, runtime, robot, processor)
    pending_handle = ControlHandle(
        handle_id="pre-registration",
        hub_epoch="hub-epoch-1",
        robot_id=node.node_id,
        robot_session_id=node.session_id,
        controller_id="controller-1",
        controller_session_id="controller-session-1",
        controller_action_endpoint="memory://pre-registration",
        action_schema="lekit.action.v1",
        control_mode="teleop",
        fencing_token=1,
        issued_at_ns=clock.utc_ns(),
        expires_at_ns=clock.utc_ns() + 3_000_000_000,
    )
    try:
        assert not node.receive_grant(pending_handle)
        assert not node.receive_management(command("grant", pending_handle, sequence=1))
        assert not node.receive_management(command("take_over", pending_handle, sequence=2))
        assert runtime.receivers == []
    finally:
        node.stop()


def test_disabled_node_rejects_grant_and_take_over(
    tmp_path: Path, clock: Clock, runtime: RecordingRuntime, robot: FakeRobot, processor: FakeProcessor
) -> None:
    node_id_path = tmp_path / "disabled-id"
    node_id_path.write_text("disabled-robot\n", encoding="utf-8")
    node = RobotNode(
        robot,
        processor,
        RobotNodeConfig(
            node_id_path=node_id_path,
            display_name="Disabled",
            accepted_payload_schemas=("lekit.action.v1",),
            control_enabled=False,
        ),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    node.start()
    node.receive_management(registered(node, runtime))
    disabled_handle = ControlHandle(
        handle_id="disabled-handle",
        hub_epoch="hub-epoch-1",
        robot_id=node.node_id,
        robot_session_id=node.session_id,
        controller_id="controller-1",
        controller_session_id="controller-session-1",
        controller_action_endpoint="memory://disabled",
        action_schema="lekit.action.v1",
        control_mode="teleop",
        fencing_token=1,
        issued_at_ns=clock.utc_ns(),
        expires_at_ns=clock.utc_ns() + 1_000_000_000,
    )
    try:
        assert not node.receive_management(command("grant", disabled_handle, sequence=1))
        assert not node.receive_management(command("take_over", disabled_handle, sequence=2))
        assert runtime.receivers == []
    finally:
        node.stop()


def test_valid_action_uses_nonblocking_receiver_and_standard_robot_path(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    inject(receiver, envelope(handle), clock)

    observation = robot_node.run_cycle()

    assert observation == {"x": 0.25}
    assert receiver.timeouts == [0.0]
    assert processor.calls == [(b"0.5", {"x": 0.25})]
    assert robot.sent == [{"x": 0.5}]
    assert robot_node.control_state is RobotControlState.CONTROLLING
    assert robot_node.stream_session_id == "stream-1"


def test_status_includes_processor_observability_and_rejection_counters(
    robot_node: RobotNode,
    processor: FakeProcessor,
) -> None:
    processor.state = "idle"
    processor.tracking = False
    processor.engaged = True
    processor.fault = "tracking lost"
    robot_node.rejections["wrong_fencing"] = 2

    status = robot_node.status

    assert status["report"]["processor_state"] == "idle"
    assert status["report"]["tracking"] is False
    assert status["report"]["engaged"] is True
    assert status["report"]["error"] == "tracking lost"
    assert status["rejections"] == {"wrong_fencing": 2}


def test_management_status_carries_coalesced_robot_diagnostics(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
) -> None:
    robot_node.rejections["stale_action"] = 3
    clock.advance(0.1)

    robot_node.run_management_once()

    status = messages(runtime, "status")[-1].body
    assert status["diagnostics"] == {
        "rejections": {"stale_action": 3},
        "robot_connected": robot.is_connected,
    }


def test_wrong_fencing_and_duplicate_sequence_never_reach_robot(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    inject(receiver, envelope(handle, fencing_token=handle.fencing_token - 1), clock)
    robot_node.run_cycle()
    assert robot_node.receive_management(command("take_over", handle, sequence=3))
    receiver = runtime.receivers[-1]
    inject(receiver, envelope(handle, sequence=4), clock)
    robot_node.run_cycle()
    inject(receiver, envelope(handle, sequence=4), clock)
    robot_node.run_cycle()

    assert robot.sent == [{"x": 0.5}]
    assert robot_node.rejections["wrong_fencing"] == 1
    assert robot_node.rejections["sequence_regressed"] == 1


@pytest.mark.parametrize(
    "transition",
    ["hand_over", "revoke", "expiry", "force_hold", "safety", "management_loss"],
)
@pytest.mark.parametrize("cleanup_failure", ["reset", "hold", "receiver_close"])
def test_authority_loss_is_fail_closed_before_fallible_cleanup(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
    transition: str,
    cleanup_failure: str,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    original_hold = robot_node._hold
    if cleanup_failure == "reset":
        processor.reset_error = RuntimeError("reset cleanup failed")
    elif cleanup_failure == "hold":
        robot_node._hold = lambda *_args: (_ for _ in ()).throw(RuntimeError("hold cleanup failed"))
    else:
        receiver.close_error = RuntimeError("receiver close failed")

    try:
        if transition in {"hand_over", "revoke", "force_hold"}:
            robot_node.receive_management(command(transition, handle, sequence=3, reason="review probe"))
        elif transition == "expiry":
            clock.advance(3.001)
            robot_node.run_cycle()
        elif transition == "safety":
            robot_node.enter_safety("review probe")
        else:
            runtime.channels[-1].send_result = False
            clock.advance(0.5)
            robot_node.run_management_once()

        receiver.frames.append(ReceivedAction(envelope(handle, sequence=9), clock.monotonic_ns()))
        robot_node.run_cycle()

        assert robot.sent == []
        assert "send_action" not in robot.calls
        assert robot_node.control_state in {RobotControlState.HOLD, RobotControlState.SAFETY}
        assert "cleanup failed" in robot_node.status["report"]["error"]
    finally:
        processor.reset_error = None
        receiver.close_error = None
        robot_node._hold = original_hold


def test_validation_order_rejects_local_staleness_before_wrong_epoch(
    robot_node: RobotNode,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    inject(receiver, envelope(handle, hub_epoch="wrong-epoch"), clock, age_s=0.101)

    robot_node.run_cycle()

    assert robot_node.rejections["stale_action"] == 1
    assert robot_node.rejections["wrong_hub_epoch"] == 0


def test_watchdog_uses_received_monotonic_time_not_controller_capture_clock(
    robot_node: RobotNode,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    inject(receiver, envelope(handle, captured_monotonic_ns=999_999_999_999), clock)
    robot_node.run_cycle()
    resets_after_control = processor.reset_count

    clock.advance(0.101)
    robot_node.run_cycle()

    assert robot_node.control_state is RobotControlState.HOLD
    assert processor.reset_count == resets_after_control + 1
    assert robot_node.rejections["stale_action"] == 1


def test_empty_processor_output_does_not_reset_on_each_idle_frame_but_resets_on_transition(
    robot_node: RobotNode,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    processor.outputs.extend([{}, {}, {"x": 0.4}, {}])
    initial_resets = processor.reset_count

    for sequence in range(4):
        inject(receiver, envelope(handle, sequence=sequence), clock)
        robot_node.run_cycle()

    assert processor.reset_count == initial_resets + 1
    assert robot_node.control_state is RobotControlState.HOLD


@pytest.mark.parametrize("cleanup_failure", ["reset", "hold"])
def test_nonterminal_hold_cleanup_failure_terminalizes_authority_before_next_frame(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
    cleanup_failure: str,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    inject(receiver, envelope(handle, sequence=0), clock)
    robot_node.run_cycle()
    assert robot_node.control_state is RobotControlState.CONTROLLING

    original_hold = robot_node._hold
    processor.outputs.append({})
    if cleanup_failure == "reset":
        processor.reset_error = RuntimeError("nonterminal reset failed")
    else:
        robot_node._hold = lambda *_args: (_ for _ in ()).throw(RuntimeError("nonterminal hold failed"))

    try:
        inject(receiver, envelope(handle, sequence=1), clock)
        robot_node.run_cycle()
    finally:
        processor.reset_error = None
        robot_node._hold = original_hold

    receiver.frames.append(ReceivedAction(envelope(handle, sequence=2), clock.monotonic_ns()))
    robot_node.run_cycle()

    assert robot.sent == [{"x": 0.5}]
    assert receiver.closed
    assert robot_node.control_state is RobotControlState.HOLD
    assert not robot_node.receive_management(command("take_over", handle, sequence=3))
    assert "cleanup failed" in robot_node.status["report"]["error"]


def test_empty_processor_frame_while_held_does_not_clear_active_hold(
    robot_node: RobotNode,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    robot_node._hold = lambda *_args: HoldResult(active=True)
    receiver = activate(robot_node, runtime, handle)
    robot_node.receive_management(command("force_hold", handle, sequence=3, reason="pause"))
    assert robot_node.receive_management(command("take_over", handle, sequence=4))
    receiver = runtime.receivers[-1]
    processor.outputs.append({})
    inject(receiver, envelope(handle), clock)

    robot_node.run_cycle()

    assert robot_node.control_state is RobotControlState.HOLD
    assert robot_node.status["report"]["active_hold"] is True


def test_new_stream_session_forces_hold_and_requires_fresh_take_over(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    inject(receiver, envelope(handle, stream_session_id="stream-a"), clock)
    robot_node.run_cycle()
    inject(receiver, envelope(handle, stream_session_id="stream-b", sequence=1), clock)
    robot_node.run_cycle()

    assert robot_node.control_state is RobotControlState.HOLD
    assert receiver.closed
    assert robot_node.rejections["wrong_stream_session"] == 1
    assert robot.sent == [{"x": 0.5}]

    assert robot_node.receive_management(command("take_over", handle, sequence=3))
    new_receiver = runtime.receivers[-1]
    inject(new_receiver, envelope(handle, stream_session_id="stream-b", sequence=0), clock)
    robot_node.run_cycle()
    assert robot.sent == [{"x": 0.5}, {"x": 0.5}]


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ({}, "processor_not_armed"),
        ({"y": 1.0}, "invalid_robot_action"),
        ({"x": math.nan}, "invalid_robot_action"),
        ({"x": True}, "invalid_robot_action"),
    ],
)
def test_processor_and_action_validation_never_forward_invalid_output(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
    output: dict[str, float],
    reason: str,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    processor.outputs.append(output)
    inject(receiver, envelope(handle), clock)

    robot_node.run_cycle()

    assert robot.sent == []
    assert robot_node.control_state is RobotControlState.HOLD
    assert robot_node.rejections[reason] == 1


def test_action_validation_accepts_a_supported_feature_subset(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        FakeRobot,
        "action_features",
        property(lambda self: {"x": float, "gripper": float}),
    )
    receiver = activate(robot_node, runtime, handle)
    processor.outputs.append({"x": 0.5})
    inject(receiver, envelope(handle), clock)

    robot_node.run_cycle()

    assert robot.sent == [{"x": 0.5}]
    assert robot_node.control_state is RobotControlState.CONTROLLING


def test_processor_exception_is_rejected_without_escaping_control_cycle(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    processor.outputs.append(ValueError("bad payload"))
    inject(receiver, envelope(handle), clock)

    robot_node.run_cycle()

    assert robot.sent == []
    assert robot_node.rejections["processor_invalid"] == 1


@pytest.mark.parametrize(
    ("advance_s", "reason"),
    [(0.101, "stale_action"), (3.001, "handle_expired")],
)
def test_processing_is_speculative_across_freshness_and_expiry_deadlines(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
    advance_s: float,
    reason: str,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    processor.call_hook = lambda: clock.advance(advance_s)
    inject(receiver, envelope(handle), clock)

    robot_node.run_cycle()

    assert robot.sent == []
    assert robot_node.rejections[reason] == 1
    assert robot_node.control_state is RobotControlState.HOLD


@pytest.mark.parametrize("transition", ["revoke", "safety"])
def test_authority_transition_during_processing_completes_and_prevents_send(
    robot_node: RobotNode,
    robot: FakeRobot,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
    transition: str,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    entered = threading.Event()
    release = threading.Event()
    transitioned = threading.Event()

    def block_processor() -> None:
        entered.set()
        assert release.wait(timeout=1.0)

    processor.call_hook = block_processor
    inject(receiver, envelope(handle), clock)
    cycle = threading.Thread(target=robot_node.run_cycle)
    cycle.start()
    assert entered.wait(timeout=1.0)

    def remove_authority() -> None:
        if transition == "revoke":
            robot_node.receive_management(command("revoke", handle, sequence=3, reason="race"))
        else:
            robot_node.enter_safety("race")
        transitioned.set()

    authority_change = threading.Thread(target=remove_authority)
    authority_change.start()
    try:
        assert transitioned.wait(timeout=0.2)
    finally:
        release.set()
        cycle.join(timeout=1.0)
        authority_change.join(timeout=1.0)

    assert not cycle.is_alive()
    assert not authority_change.is_alive()
    assert robot.sent == []


def test_robot_send_failure_enters_hold_without_escaping_control_cycle(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    robot.send_error = OSError("robot link lost")
    inject(receiver, envelope(handle), clock)

    robot_node.run_cycle()

    assert robot_node.control_state is RobotControlState.HOLD
    assert robot_node.rejections["robot_send_failed"] == 1
    assert "robot link lost" in robot_node.status["report"]["error"]


def test_handle_deadline_is_clamped_to_local_ttl(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, replace(handle, expires_at_ns=clock.utc_ns() + 60_000_000_000))
    clock.advance(3.001)
    inject(receiver, envelope(handle), clock)

    robot_node.run_cycle()

    assert robot.sent == []
    assert robot_node.rejections["handle_expired"] == 1
    assert robot_node.control_state is RobotControlState.HOLD

    renewed = replace(
        handle,
        expires_at_ns=clock.utc_ns() + 120_000_000_000,
    )
    assert not robot_node.receive_management(command("renewal", renewed, sequence=3))
    assert not robot_node.receive_management(command("take_over", renewed, sequence=4))


@pytest.mark.parametrize("terminal_kind", ["hand_over", "revoke", "force_hold"])
def test_renewal_does_not_make_old_expiry_terminal_command_miss_current_authority(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
    terminal_kind: str,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    renewed = replace(handle, expires_at_ns=handle.expires_at_ns + 10_000_000_000)
    assert robot_node.receive_management(command("renewal", renewed, sequence=3))

    assert robot_node.receive_management(
        command(terminal_kind, handle, sequence=4, reason="old-expiry command")
    )
    assert robot_node.receive_management(
        command(terminal_kind, handle, sequence=5, reason="duplicate command")
    )
    assert not robot_node.receive_management(
        command(terminal_kind, handle, sequence=4, reason="out of order")
    )
    receiver.frames.append(ReceivedAction(envelope(handle, sequence=8), clock.monotonic_ns()))
    robot_node.run_cycle()

    assert robot.sent == []


def test_take_over_uses_immutable_authority_without_replacing_renewed_deadline(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    short = replace(handle, expires_at_ns=clock.utc_ns() + 200_000_000)
    activate(robot_node, runtime, short)
    renewed = replace(short, expires_at_ns=clock.utc_ns() + 10_000_000_000)
    assert robot_node.receive_management(command("renewal", renewed, sequence=3))

    assert robot_node.receive_management(command("take_over", short, sequence=4))
    receiver = runtime.receivers[-1]
    clock.advance(0.25)
    inject(receiver, envelope(short, sequence=0), clock)
    robot_node.run_cycle()

    assert robot.sent == [{"x": 0.5}]


def test_take_over_before_renewal_keeps_same_authority_renewable(
    robot_node: RobotNode,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    assert robot_node.receive_management(command("grant", handle, sequence=1))
    assert robot_node.receive_management(command("take_over", handle, sequence=2))
    renewed = replace(handle, expires_at_ns=clock.utc_ns() + 20_000_000_000)

    assert robot_node.receive_management(command("renewal", renewed, sequence=3))


def test_renewal_after_local_deadline_terminalizes_without_an_expiry_cycle(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    clock.advance(3.001)
    renewed = replace(handle, expires_at_ns=clock.utc_ns() + 20_000_000_000)

    assert not robot_node.receive_management(command("renewal", renewed, sequence=3))
    receiver.frames.append(ReceivedAction(envelope(handle, sequence=5), clock.monotonic_ns()))
    robot_node.run_cycle()

    assert robot.sent == []
    assert robot_node.control_state is RobotControlState.HOLD
    assert not robot_node.receive_management(command("take_over", renewed, sequence=4))


def test_registered_and_heartbeat_ack_require_exact_correlation_and_increasing_sequence(
    robot_node: RobotNode, runtime: RecordingRuntime, clock: Clock
) -> None:
    assert not robot_node.receive_management(
        registered(robot_node, runtime, correlation_id="wrong", sequence=1)
    )
    clock.advance(0.5)
    robot_node.run_management_once()
    heartbeat = messages(runtime, "heartbeat")[-1]
    wrong = ManagementMessage(
        protocol_version=PROTOCOL_VERSION,
        kind="heartbeat_ack",
        correlation_id="wrong",
        sender_id="hub",
        sender_session_id="hub-epoch-1",
        sequence=1,
        sent_at_ns=clock.utc_ns(),
        body={"hub_epoch": "hub-epoch-1"},
    )
    assert not robot_node.receive_management(wrong)
    assert robot_node.receive_management(replace(wrong, correlation_id=heartbeat.correlation_id, sequence=2))
    clock.advance(0.5)
    robot_node.run_management_once()
    next_heartbeat = messages(runtime, "heartbeat")[-1]
    assert not robot_node.receive_management(
        replace(wrong, correlation_id=next_heartbeat.correlation_id, sequence=2)
    )


def test_three_missed_heartbeat_acks_create_fresh_safe_session(
    robot_node: RobotNode,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    old_session = robot_node.session_id

    for _ in range(4):
        clock.advance(0.5)
        robot_node.run_management_once()

    assert robot_node.session_id != old_session
    assert receiver.closed
    assert robot_node.control_state is RobotControlState.HOLD
    assert len(messages(runtime, "register")) == 2
    assert not robot_node.receive_management(command("take_over", handle, sequence=10))


def test_reconnect_rejects_new_session_grant_until_new_registration_ack(
    robot_node: RobotNode,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    activate(robot_node, runtime, handle)
    runtime.channels[-1].send_result = False
    clock.advance(0.5)
    robot_node.run_management_once()
    fresh = replace(
        handle,
        handle_id="fresh-handle",
        robot_session_id=robot_node.session_id,
        fencing_token=handle.fencing_token + 1,
        issued_at_ns=handle.issued_at_ns + 1,
        expires_at_ns=handle.expires_at_ns + 1,
    )

    assert not robot_node.receive_grant(fresh)
    assert not robot_node.receive_management(command("grant", fresh, sequence=1))
    assert robot_node.receive_management(registered(robot_node, runtime, sequence=0))
    assert robot_node.receive_management(command("grant", fresh, sequence=1))


def test_failed_terminal_report_does_not_ack_or_poison_fresh_management_session(
    robot_node: RobotNode,
    runtime: RecordingRuntime,
    handle: ControlHandle,
) -> None:
    activate(robot_node, runtime, handle)
    old_session = robot_node.session_id
    runtime.channels[-1].send_result = False

    assert not robot_node.receive_management(command("hand_over", handle, sequence=7))

    assert robot_node.session_id != old_session
    assert messages(runtime, "hand_over_ack") == []
    assert [message.kind for message in runtime.channels[-1].sent] == ["register"]
    assert robot_node.receive_management(registered(robot_node, runtime, sequence=0))


def test_terminal_report_exception_cannot_escape_or_preserve_authority(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    runtime.channels[-1].send_error = RuntimeError("report send raised")

    assert not robot_node.receive_management(command("revoke", handle, sequence=3, reason="report"))

    receiver.frames.append(ReceivedAction(envelope(handle, sequence=7), clock.monotonic_ns()))
    robot_node.run_cycle()
    assert robot.sent == []
    assert robot_node.control_state is RobotControlState.HOLD
    assert "report send raised" in robot_node.status["report"]["error"]


def test_cleanup_callback_cannot_reenter_take_over_before_hold_finishes(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    nested_results: list[bool] = []

    def reentrant_hold(*_args) -> HoldResult:
        nested_results.append(robot_node.receive_management(command("take_over", handle, sequence=4)))
        return HoldResult(active=False)

    robot_node._hold = reentrant_hold
    assert robot_node.receive_management(command("force_hold", handle, sequence=3, reason="reentrant hold"))
    receiver.frames.append(ReceivedAction(envelope(handle, sequence=8), clock.monotonic_ns()))
    robot_node.run_cycle()

    assert nested_results == [False]
    assert len(runtime.receivers) == 1
    assert robot.sent == []


def test_reconnect_open_failure_stays_fail_closed_and_reports_degraded(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    runtime.open_node_error = RuntimeError("reconnect open failed")
    runtime.channels[-1].send_result = False
    clock.advance(0.5)

    robot_node.run_management_once()

    receiver.frames.append(ReceivedAction(envelope(handle, sequence=8), clock.monotonic_ns()))
    robot_node.run_cycle()
    assert robot.sent == []
    assert robot_node.control_state is RobotControlState.HOLD
    assert robot_node.status["report"]["runtime_state"] == "degraded"
    assert "reconnect open failed" in robot_node.status["report"]["error"]


def test_hand_over_revoke_force_hold_and_safety_are_locally_authoritative(
    robot_node: RobotNode,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    inject(receiver, envelope(handle), clock)
    robot_node.run_cycle()
    assert robot_node.receive_management(command("force_hold", handle, sequence=3, reason="operator"))
    assert robot_node.control_state is RobotControlState.HOLD
    assert messages(runtime, "force_hold_ack")

    assert robot_node.receive_management(command("take_over", handle, sequence=4))
    receiver = runtime.receivers[-1]
    robot_node.enter_safety("local_estop")
    inject(receiver, envelope(handle, sequence=1), clock)
    robot_node.run_cycle()
    assert robot_node.control_state is RobotControlState.SAFETY
    assert robot_node.rejections["safety"] == 1
    assert not robot_node.receive_management(command("grant", handle, sequence=5))

    robot_node.clear_safety()
    assert robot_node.control_state is RobotControlState.HOLD
    assert not robot_node.receive_management(command("take_over", handle, sequence=6))
    new_handle = replace(
        handle,
        handle_id="handle-2",
        fencing_token=handle.fencing_token + 1,
        issued_at_ns=handle.issued_at_ns + 1,
        expires_at_ns=handle.expires_at_ns + 1,
    )
    assert robot_node.receive_management(command("grant", new_handle, sequence=7))
    assert robot_node.receive_management(command("take_over", new_handle, sequence=8))
    assert robot_node.receive_management(command("hand_over", new_handle, sequence=9))
    assert runtime.receivers[-1].closed
    assert messages(runtime, "hand_over_ack")


def test_enter_safety_resets_processor_once_and_never_restores_cached_authority(
    robot_node: RobotNode,
    processor: FakeProcessor,
    runtime: RecordingRuntime,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    resets_before = processor.reset_count

    robot_node.enter_safety("local_estop")

    assert processor.reset_count == resets_before + 1
    assert receiver.closed
    robot_node.clear_safety()
    assert not robot_node.receive_management(command("take_over", handle, sequence=3))


def test_reentrant_clear_safety_cannot_unlatch_outer_safety_transition(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    clock: Clock,
    handle: ControlHandle,
) -> None:
    old_receiver = activate(robot_node, runtime, handle)
    callback_states: list[RobotControlState] = []
    original_hold = robot_node._hold

    def reentrant_clear(*_args) -> HoldResult:
        robot_node.clear_safety()
        callback_states.append(robot_node.control_state)
        return HoldResult(active=False)

    robot_node._hold = reentrant_clear
    fresh = replace(
        handle,
        handle_id="handle-after-safety",
        fencing_token=handle.fencing_token + 1,
        issued_at_ns=handle.issued_at_ns + 1,
        expires_at_ns=handle.expires_at_ns + 1,
    )
    try:
        robot_node.enter_safety("reentrant local estop")

        assert callback_states == [RobotControlState.SAFETY]
        assert robot_node.control_state is RobotControlState.SAFETY
        assert not robot_node.receive_management(command("grant", fresh, sequence=3))
        assert not robot_node.receive_management(command("take_over", fresh, sequence=4))
        old_receiver.frames.append(ReceivedAction(envelope(fresh, sequence=0), clock.monotonic_ns()))
        robot_node.run_cycle()
        assert robot.sent == []

        robot_node.clear_safety()
        assert robot_node.receive_management(command("grant", fresh, sequence=5))
        assert robot_node.receive_management(command("take_over", fresh, sequence=6))
        receiver = runtime.receivers[-1]
        inject(receiver, envelope(fresh, sequence=0), clock)
        robot_node.run_cycle()
        assert robot.sent == [{"x": 0.5}]
    finally:
        robot_node._hold = original_hold


def test_active_hold_result_is_reported_and_stop_disconnects_without_closing_shared_runtime(
    tmp_path: Path, clock: Clock, runtime: RecordingRuntime, robot: FakeRobot, processor: FakeProcessor
) -> None:
    reasons: list[str] = []

    def hold(_robot: Robot, _observation: dict[str, float] | None, reason: str) -> HoldResult:
        reasons.append(reason)
        return HoldResult(active=True)

    node = RobotNode(
        robot,
        processor,
        RobotNodeConfig(
            node_id_path=tmp_path / "active-hold-id",
            display_name="Active hold",
            accepted_payload_schemas=("lekit.action.v1",),
        ),
        runtime=runtime,
        hold=hold,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    node.start()
    node.stop()

    assert reasons[-1] == "stopped"
    assert robot.calls == ["connect", "disconnect"]
    assert runtime.closed is False


def test_active_hold_is_cleared_only_after_final_validated_send(
    tmp_path: Path,
    clock: Clock,
    runtime: RecordingRuntime,
    robot: FakeRobot,
    processor: FakeProcessor,
) -> None:
    node = unregistered_node(
        tmp_path,
        clock,
        runtime,
        robot,
        processor,
        hold=lambda *_args: HoldResult(active=True),
    )
    node.receive_management(registered(node, runtime))
    live = ControlHandle(
        handle_id="active-hold",
        hub_epoch="hub-epoch-1",
        robot_id=node.node_id,
        robot_session_id=node.session_id,
        controller_id="controller-1",
        controller_session_id="controller-session-1",
        controller_action_endpoint="memory://active-hold",
        action_schema="lekit.action.v1",
        control_mode="teleop",
        fencing_token=1,
        issued_at_ns=clock.utc_ns(),
        expires_at_ns=clock.utc_ns() + 3_000_000_000,
    )
    try:
        receiver = activate(node, runtime, live)
        node.receive_management(command("force_hold", live, sequence=3, reason="pause"))
        assert node.status["report"]["active_hold"] is True
        assert node.receive_management(command("take_over", live, sequence=4))
        assert node.status["report"]["active_hold"] is True
        receiver = runtime.receivers[-1]
        inject(receiver, envelope(live), clock)
        node.run_cycle()
        assert node.control_state is RobotControlState.CONTROLLING
        assert node.status["report"]["active_hold"] is False
    finally:
        node.stop()


def test_run_preserves_control_failure_when_disconnect_also_fails(
    robot_node: RobotNode, robot: FakeRobot
) -> None:
    robot.observation_error = RuntimeError("control failure")
    robot.disconnect_error = OSError("disconnect failure")

    with pytest.raises(RuntimeError, match="control failure"):
        robot_node.run()


@pytest.mark.parametrize("resource", ["receiver", "channel", "thread"])
def test_stop_attempts_disconnect_after_each_owned_resource_cleanup_failure(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    handle: ControlHandle,
    resource: str,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    original_thread = robot_node._management_thread

    class JoinFailure:
        def join(self, timeout: float) -> None:
            del timeout
            raise RuntimeError("thread join failed")

    if resource == "receiver":
        receiver.close_error = RuntimeError("receiver close failed")
    elif resource == "channel":
        runtime.channels[-1].close_error = RuntimeError("channel close failed")
    else:
        robot_node._management_thread = JoinFailure()

    with pytest.raises(RuntimeError, match=f"{resource} .*failed"):
        robot_node.stop()

    assert robot.calls[-1] == "disconnect"
    assert robot.is_connected is False
    if original_thread is not None and resource == "thread":
        original_thread.join(timeout=1.0)


def test_run_preserves_primary_failure_across_all_cleanup_failures(
    robot_node: RobotNode,
    robot: FakeRobot,
    runtime: RecordingRuntime,
    handle: ControlHandle,
) -> None:
    receiver = activate(robot_node, runtime, handle)
    robot.observation_error = RuntimeError("primary control failure")
    receiver.close_error = OSError("receiver cleanup failure")
    runtime.channels[-1].close_error = OSError("channel cleanup failure")
    robot.disconnect_error = OSError("disconnect cleanup failure")

    with pytest.raises(RuntimeError, match="primary control failure"):
        robot_node.run()

    assert "disconnect" in robot.calls


@pytest.mark.parametrize("failure", ["open_node", "registration_send", "thread_start"])
def test_startup_failure_unwinds_connection_and_restores_stopped_state(
    tmp_path: Path,
    clock: Clock,
    runtime: RecordingRuntime,
    robot: FakeRobot,
    processor: FakeProcessor,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import lekit.control.robot as robot_module

    if failure == "open_node":
        runtime.open_node_error = RuntimeError("open node failed")
    elif failure == "registration_send":
        runtime.node_send_result = False
    else:

        class StartFailureThread:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs

            def start(self) -> None:
                raise RuntimeError("thread start failed")

        monkeypatch.setattr(robot_module.threading, "Thread", StartFailureThread)

    node = RobotNode(
        robot,
        processor,
        RobotNodeConfig(
            node_id_path=tmp_path / f"startup-{failure}",
            display_name="Startup failure",
            accepted_payload_schemas=("lekit.action.v1",),
        ),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )

    with pytest.raises(RuntimeError, match="failed"):
        node.start()

    assert robot.calls == ["connect", "disconnect"]
    assert robot.is_connected is False
    assert node.status["report"]["runtime_state"] == "stopped"
    node.stop()


def test_startup_preserves_raised_registration_error_across_cleanup_failures(
    tmp_path: Path,
    clock: Clock,
    runtime: RecordingRuntime,
    robot: FakeRobot,
    processor: FakeProcessor,
) -> None:
    runtime.node_send_error = ValueError("registration transport raised")
    runtime.node_close_error = OSError("startup channel close failed")
    robot.disconnect_error = RuntimeError("startup disconnect failed")
    node = RobotNode(
        robot,
        processor,
        RobotNodeConfig(
            node_id_path=tmp_path / "startup-registration-exception",
            display_name="Registration exception",
            accepted_payload_schemas=("lekit.action.v1",),
        ),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )

    with pytest.raises(ValueError, match="registration transport raised") as raised:
        node.start()

    notes = getattr(raised.value, "__notes__", [])
    assert len(notes) == 1
    assert "startup channel close failed" in notes[0]
    assert "startup disconnect failed" in notes[0]
    assert robot.calls == ["connect", "disconnect"]
    assert node.status["report"]["runtime_state"] == "stopped"
    assert "registration transport raised" in node.status["report"]["error"]
