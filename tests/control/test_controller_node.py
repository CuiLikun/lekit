from __future__ import annotations

import threading
from collections import deque
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from lekit.control.controller import (
    ControllerNode,
    ControllerNodeConfig,
    HandleExpired,
    HandleNotGranted,
)
from lekit.control.memory_runtime import MemoryRuntime
from lekit.control.model import (
    PROTOCOL_VERSION,
    ControlHandle,
    ControllerControlState,
    ManagementMessage,
    NodeRole,
    TimingConfig,
)
from lekit.control.runtime import ReceivedManagement


class Clock:
    def __init__(self) -> None:
        self.monotonic = 0
        self.utc = 1_000

    def monotonic_ns(self) -> int:
        return self.monotonic

    def utc_ns(self) -> int:
        return self.utc

    def advance(self, nanoseconds: int) -> None:
        self.monotonic += nanoseconds
        self.utc += nanoseconds


class RecordingNodeChannel:
    def __init__(self) -> None:
        self.inbox: deque[ReceivedManagement] = deque()
        self.sent: list[ManagementMessage] = []
        self.send_result = True
        self.closed = False
        self.condition = threading.Condition()

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        del timeout_s
        with self.condition:
            if self.closed or not self.inbox:
                return None
            return self.inbox.popleft()

    def send(self, message: ManagementMessage) -> bool:
        self.sent.append(message)
        return self.send_result and not self.closed

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class RecordingRuntime:
    def __init__(self) -> None:
        self.actions = MemoryRuntime()
        self.channels: list[RecordingNodeChannel] = []

    def open_node(self, node_id: str, session_id: str, *, hub_seed: str | None) -> RecordingNodeChannel:
        del node_id, session_id, hub_seed
        channel = RecordingNodeChannel()
        self.channels.append(channel)
        return channel

    def open_action_publisher(self, endpoint: str):
        return self.actions.open_action_publisher(endpoint)

    def open_action_receiver(self, endpoint: str):
        return self.actions.open_action_receiver(endpoint)

    def open_hub(self, endpoint: str, *, hub_epoch: str, advertise_endpoint: str | None = None):
        del endpoint, hub_epoch, advertise_endpoint
        raise AssertionError("ControllerNode must not open a Hub channel")

    def close(self) -> None:
        self.actions.close()


class BlockingPublisher:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.envelopes = []

    def send(self, envelope) -> bool:
        self.entered.set()
        assert self.release.wait(timeout=1.0)
        self.envelopes.append(envelope)
        return True

    def close(self) -> None:
        self.release.set()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def runtime() -> RecordingRuntime:
    return RecordingRuntime()


@pytest.fixture
def controller(tmp_path: Path, clock: Clock, runtime: RecordingRuntime) -> ControllerNode:
    node_id_path = tmp_path / "controller-id"
    node_id_path.write_text("controller-1\n", encoding="utf-8")
    node = ControllerNode(
        ControllerNodeConfig(
            node_id_path=node_id_path,
            display_name="Controller",
            action_schemas=("lekit.action.v1",),
            action_endpoint="memory://controller-actions",
            timing=TimingConfig(handle_ttl_s=3.0),
        ),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    node.start()
    yield node
    node.stop()


@pytest.fixture
def handle(controller: ControllerNode, clock: Clock) -> ControlHandle:
    return ControlHandle(
        handle_id="handle-1",
        hub_epoch="hub-epoch-1",
        robot_id="robot-1",
        robot_session_id="robot-session-1",
        controller_id=controller.node_id,
        controller_session_id=controller.session_id,
        controller_action_endpoint="memory://controller-actions",
        action_schema="lekit.action.v1",
        control_mode="teleop",
        fencing_token=1,
        issued_at_ns=clock.utc_ns(),
        expires_at_ns=clock.utc_ns() + 10_000_000_000,
    )


@pytest.fixture
def action_receiver(runtime: RecordingRuntime):
    receiver = runtime.open_action_receiver("memory://controller-actions")
    yield receiver
    receiver.close()


def messages(runtime: RecordingRuntime, kind: str) -> list[ManagementMessage]:
    return [message for channel in runtime.channels for message in channel.sent if message.kind == kind]


def activate(controller: ControllerNode, handle: ControlHandle) -> None:
    controller.receive_grant(handle)
    controller.take_over(handle)
    controller.receive_robot_ready(handle)


def command(
    kind: str,
    handle: ControlHandle,
    *,
    correlation_id: str = "command-1",
    sequence: int = 1,
) -> ManagementMessage:
    return ManagementMessage(
        protocol_version=PROTOCOL_VERSION,
        kind=kind,
        correlation_id=correlation_id,
        sender_id="hub",
        sender_session_id=handle.hub_epoch,
        sequence=sequence,
        sent_at_ns=1,
        body={"hub_epoch": handle.hub_epoch, "handle": asdict(handle)},
    )


def registered(
    controller: ControllerNode,
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


def heartbeat_ack(*, epoch: str, correlation_id: str, sequence: int) -> ManagementMessage:
    return ManagementMessage(
        protocol_version=PROTOCOL_VERSION,
        kind="heartbeat_ack",
        correlation_id=correlation_id,
        sender_id="hub",
        sender_session_id=epoch,
        sequence=sequence,
        sent_at_ns=1,
        body={"hub_epoch": epoch},
    )


def test_start_registers_controller_identity_and_never_enables_publication(
    controller: ControllerNode, runtime: RecordingRuntime
) -> None:
    registration = messages(runtime, "register")

    assert len(registration) == 1
    descriptor = registration[0].body["descriptor"]
    assert descriptor["node_id"] == "controller-1"
    assert descriptor["session_id"] == controller.session_id
    assert descriptor["role"] == NodeRole.CONTROLLER.value
    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20) is False


def test_controller_advertises_presentation_and_exposes_frozen_current_handle(
    controller: ControllerNode,
    handle: ControlHandle,
    runtime: RecordingRuntime,
) -> None:
    assert controller.current_handle is None

    controller.receive_grant(handle)

    assert controller.current_handle == handle
    with pytest.raises(FrozenInstanceError):
        controller.current_handle.handle_id = "other"  # type: ignore[misc]
    assert messages(runtime, "register")[0].body["descriptor"]["presentation"] == {
        "monitor_url": None,
        "video_status_url": None,
        "cameras": (),
    }


def test_controller_public_types_are_exported() -> None:
    import lekit.control as control

    assert control.ControllerNode is ControllerNode
    assert control.ControllerNodeConfig is ControllerNodeConfig


def test_take_over_requires_a_granted_current_handle(
    controller: ControllerNode, handle: ControlHandle
) -> None:
    with pytest.raises(HandleNotGranted):
        controller.take_over(handle)

    controller.receive_grant(handle)
    controller.take_over(handle)

    assert controller.control_state is ControllerControlState.TAKING_OVER


def test_robot_ready_starts_new_stream_and_wraps_payload(
    controller: ControllerNode, handle: ControlHandle, action_receiver
) -> None:
    controller.receive_grant(handle)
    controller.take_over(handle)
    controller.receive_robot_ready(handle)

    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20)
    received = action_receiver.receive_latest()
    assert received is not None
    assert received.envelope.handle_id == handle.handle_id
    assert received.envelope.payload == b"frame"
    assert received.envelope.sequence == 0
    assert received.envelope.controller_session_id == controller.session_id
    assert received.envelope.stream_session_id == controller.stream_session_id


def test_deferred_take_over_waits_for_source_action_before_requesting_robot(
    tmp_path: Path,
    clock: Clock,
    runtime: RecordingRuntime,
) -> None:
    node_id_path = tmp_path / "deferred-controller-id"
    node_id_path.write_text("deferred-controller\n", encoding="utf-8")
    controller = ControllerNode(
        ControllerNodeConfig(
            node_id_path=node_id_path,
            display_name="Deferred Controller",
            action_schemas=("lekit.action.v1",),
            action_endpoint="memory://deferred-actions",
            defer_take_over_until_first_action=True,
            timing=TimingConfig(handle_ttl_s=3.0),
        ),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    controller.start()
    handle = ControlHandle(
        handle_id="deferred-handle",
        hub_epoch="hub-epoch-1",
        robot_id="robot-1",
        robot_session_id="robot-session-1",
        controller_id=controller.node_id,
        controller_session_id=controller.session_id,
        controller_action_endpoint="memory://deferred-actions",
        action_schema="lekit.action.v1",
        control_mode="teleop",
        fencing_token=1,
        issued_at_ns=clock.utc_ns(),
        expires_at_ns=clock.utc_ns() + 10_000_000_000,
    )

    try:
        assert controller.receive_management(registered(controller, runtime, sequence=0))
        assert controller.receive_management(command("grant", handle, sequence=1))
        assert controller.receive_management(command("take_over", handle, sequence=2))
        assert controller.control_state is ControllerControlState.TAKING_OVER
        assert messages(runtime, "take_over_requested") == []

        assert not controller.publish(b"ready", captured_monotonic_ns=10, captured_utc_ns=20)
        request = messages(runtime, "take_over_requested")[-1]
        assert controller.receive_management(
            command("robot_ready", handle, correlation_id=request.correlation_id, sequence=3)
        )
        assert controller.control_state is ControllerControlState.STREAMING

        assert controller.publish(b"first", captured_monotonic_ns=30, captured_utc_ns=40)
        assert len(messages(runtime, "controller_streaming")) == 1
    finally:
        controller.stop()


def test_hand_over_stops_publication_before_management_request(
    controller: ControllerNode, handle: ControlHandle, runtime: RecordingRuntime
) -> None:
    activate(controller, handle)
    controller.hand_over(handle)

    assert controller.publish(b"late", captured_monotonic_ns=30, captured_utc_ns=40) is False
    assert messages(runtime, "hand_over_requested")[-1].body["handle"] == asdict(handle)
    assert messages(runtime, "controller_released")[-1].body["handle"] == asdict(handle)
    assert controller.control_state is ControllerControlState.IDLE
    replacement = replace(handle, handle_id="handle-2", fencing_token=2)
    assert controller.receive_grant(replacement)


def test_valid_management_command_acks_exact_handle_and_correlation(
    controller: ControllerNode, handle: ControlHandle, runtime: RecordingRuntime
) -> None:
    controller.receive_management(command("grant", handle, correlation_id="grant-1"))

    acknowledgement = messages(runtime, "grant_ack")[-1]
    assert acknowledgement.correlation_id == "grant-1"
    assert acknowledgement.body == {"handle": asdict(handle)}
    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20) is False


def test_routed_hand_over_reports_release_request_back_to_hub(
    controller: ControllerNode,
    handle: ControlHandle,
    runtime: RecordingRuntime,
    clock: Clock,
) -> None:
    assert controller.receive_management(registered(controller, runtime, sequence=0))
    assert controller.receive_management(command("grant", handle, sequence=1))

    assert controller.receive_management(
        command("hand_over", handle, correlation_id="hand-over-1", sequence=2)
    )

    request = messages(runtime, "hand_over_requested")[-1]
    assert request.body == {"handle": asdict(handle)}
    assert messages(runtime, "controller_released")[-1].body == {"handle": asdict(handle)}
    assert messages(runtime, "hand_over_ack")[-1].correlation_id == "hand-over-1"
    assert controller.control_state is ControllerControlState.IDLE
    clock.advance(round(1_000_000_000 / controller.config.timing.status_rate_hz))
    controller.run_management_once()
    assert messages(runtime, "status")[-1].body["report"]["error"] is None


@pytest.mark.parametrize(
    "wrong_handle",
    [
        lambda handle: replace(handle, hub_epoch="stale-epoch"),
        lambda handle: replace(handle, controller_session_id="stale-session"),
        lambda handle: replace(handle, handle_id="stale-handle"),
        lambda handle: replace(handle, fencing_token=handle.fencing_token + 1),
    ],
)
def test_stale_management_command_cannot_grant_or_refresh_authority(
    controller: ControllerNode, handle: ControlHandle, wrong_handle
) -> None:
    controller.receive_grant(handle)
    stale = wrong_handle(handle)

    controller.receive_management(command("renewal", stale))

    controller.take_over(handle)
    controller.receive_robot_ready(handle)
    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20)


def test_expiry_uses_local_ttl_cap_while_hub_is_absent(
    controller: ControllerNode, handle: ControlHandle, clock: Clock
) -> None:
    activate(controller, handle)
    clock.advance(3_000_000_000)

    assert controller.publish(b"late", captured_monotonic_ns=10, captured_utc_ns=20) is False
    assert controller.control_state is ControllerControlState.IDLE
    with pytest.raises(HandleExpired):
        controller.take_over(handle)


def test_robot_ready_rejects_expired_local_authority_before_streaming(
    controller: ControllerNode, handle: ControlHandle, clock: Clock, runtime: RecordingRuntime
) -> None:
    controller.receive_grant(handle)
    controller.take_over(handle)
    clock.advance(controller.config.timing.handle_ttl_ns)

    assert controller.receive_robot_ready(handle) is False
    assert controller.control_state is ControllerControlState.IDLE
    assert messages(runtime, "controller_streaming") == []


def test_duplicate_grant_does_not_replace_the_cached_handle_or_extend_its_lease(
    controller: ControllerNode, handle: ControlHandle, clock: Clock
) -> None:
    assert controller.receive_grant(handle)
    clock.advance(2_000_000_000)
    duplicate = replace(handle, expires_at_ns=clock.utc_ns() + 100_000_000_000)

    assert controller.receive_grant(duplicate)
    with pytest.raises(HandleNotGranted):
        controller.take_over(duplicate)
    controller.take_over(handle)
    controller.receive_robot_ready(handle)
    clock.advance(1_000_000_001)
    assert controller.publish(b"late", captured_monotonic_ns=10, captured_utc_ns=20) is False


def test_regressed_renewal_cannot_extend_the_cached_local_deadline(
    controller: ControllerNode, handle: ControlHandle, clock: Clock
) -> None:
    assert controller.receive_management(command("grant", handle, correlation_id="grant-1", sequence=1))
    renewed = replace(handle, expires_at_ns=clock.utc_ns() + 100_000_000_000)
    assert controller.receive_management(command("renewal", renewed, correlation_id="renewal-1", sequence=2))
    controller.take_over(renewed)
    controller.receive_robot_ready(renewed)
    clock.advance(1_000_000_000)

    assert (
        controller.receive_management(command("renewal", handle, correlation_id="renewal-2", sequence=3))
        is False
    )
    clock.advance(2_000_000_001)
    assert controller.publish(b"late", captured_monotonic_ns=10, captured_utc_ns=20) is False


def test_registered_requires_the_current_registration_correlation(
    controller: ControllerNode, runtime: RecordingRuntime
) -> None:
    assert controller.receive_management(registered(controller, runtime, correlation_id="unrelated")) is False
    assert controller.receive_management(registered(controller, runtime))


def test_heartbeat_ack_requires_a_strictly_increasing_hub_sequence(
    controller: ControllerNode, clock: Clock, runtime: RecordingRuntime, handle: ControlHandle
) -> None:
    assert controller.receive_management(registered(controller, runtime, sequence=0))
    clock.advance(controller.config.timing.heartbeat_interval_ns)
    controller.run_management_once()
    heartbeat = messages(runtime, "heartbeat")[-1]

    assert (
        controller.receive_management(
            heartbeat_ack(epoch="hub-epoch-1", correlation_id=heartbeat.correlation_id, sequence=0)
        )
        is False
    )
    assert controller.receive_management(
        heartbeat_ack(epoch="hub-epoch-1", correlation_id=heartbeat.correlation_id, sequence=1)
    )
    assert (
        controller.receive_management(command("grant", handle, correlation_id="grant-1", sequence=1)) is False
    )


def test_failed_routed_take_over_does_not_ack_or_mutate_the_new_session(
    controller: ControllerNode, handle: ControlHandle, runtime: RecordingRuntime
) -> None:
    assert controller.receive_grant(handle)
    old_session = controller.session_id
    runtime.channels[-1].send_result = False

    assert (
        controller.receive_management(command("take_over", handle, correlation_id="take-over-1", sequence=1))
        is False
    )
    assert controller.session_id != old_session
    assert [message.kind for message in runtime.channels[-1].sent] == ["register"]
    assert controller.receive_management(registered(controller, runtime, sequence=0))


@pytest.mark.parametrize("invalidation", ["hand_over", "revoke", "expiry", "stop", "management_loss"])
def test_invalidation_waits_for_an_in_flight_action_send(
    controller: ControllerNode,
    handle: ControlHandle,
    clock: Clock,
    runtime: RecordingRuntime,
    invalidation: str,
) -> None:
    activate(controller, handle)
    publisher = BlockingPublisher()
    controller._action_publisher = publisher
    published: list[bool] = []
    sender = threading.Thread(
        target=lambda: published.append(
            controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20)
        )
    )
    sender.start()
    assert publisher.entered.wait(timeout=1.0)
    invalidated = threading.Event()

    def invalidate() -> None:
        if invalidation == "hand_over":
            controller.hand_over(handle)
        elif invalidation == "revoke":
            controller.receive_management(command("revoke", handle, sequence=1))
        elif invalidation == "expiry":
            clock.advance(controller.config.timing.handle_ttl_ns)
            assert controller.publish(b"late", captured_monotonic_ns=30, captured_utc_ns=40) is False
        elif invalidation == "stop":
            controller.stop()
        else:
            runtime.channels[-1].send_result = False
            controller.hand_over(handle)
        invalidated.set()

    invalidator = threading.Thread(target=invalidate)
    invalidator.start()
    try:
        assert invalidated.wait(timeout=0.05) is False
    finally:
        publisher.release.set()
        sender.join(timeout=1.0)
        invalidator.join(timeout=1.0)
    assert sender.is_alive() is False
    assert invalidator.is_alive() is False
    assert published == [True]
    assert len(publisher.envelopes) == 1


def test_routed_robot_ready_requires_the_current_take_over_correlation(
    controller: ControllerNode, handle: ControlHandle, runtime: RecordingRuntime
) -> None:
    controller.receive_management(command("grant", handle, correlation_id="grant-1"))
    controller.receive_management(command("take_over", handle, correlation_id="hub-take-over", sequence=2))
    forwarded = messages(runtime, "take_over_requested")[-1]

    assert (
        controller.receive_management(command("robot_ready", handle, correlation_id="unknown", sequence=3))
        is False
    )
    assert controller.control_state is ControllerControlState.TAKING_OVER
    assert controller.receive_management(
        command("robot_ready", handle, correlation_id=forwarded.correlation_id, sequence=4)
    )
    assert controller.control_state is ControllerControlState.STREAMING


def test_renewal_uses_the_canonical_kind_and_restarts_the_local_ttl(
    controller: ControllerNode, handle: ControlHandle, clock: Clock, runtime: RecordingRuntime
) -> None:
    controller.receive_management(command("grant", handle, correlation_id="grant-1"))
    clock.advance(2_000_000_000)
    renewed = replace(handle, expires_at_ns=clock.utc_ns() + 10_000_000_000)

    assert controller.receive_management(command("renewal", renewed, correlation_id="renewal-1", sequence=2))
    acknowledgement = messages(runtime, "renewal_ack")[-1]
    assert acknowledgement.correlation_id == "renewal-1"
    assert acknowledgement.body == {"handle": asdict(renewed)}
    controller.take_over(renewed)
    controller.receive_robot_ready(renewed)
    clock.advance(2_999_999_999)
    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20)


def test_replayed_renewal_cannot_extend_local_authority(
    controller: ControllerNode, handle: ControlHandle, clock: Clock
) -> None:
    controller.receive_management(command("grant", handle, correlation_id="grant-1", sequence=1))
    clock.advance(1)
    renewed = replace(handle, expires_at_ns=clock.utc_ns() + 10_000_000_000)
    renewal = command("renewal", renewed, correlation_id="renewal-1", sequence=2)
    assert controller.receive_management(renewal)
    controller.take_over(renewed)
    controller.receive_robot_ready(renewed)
    clock.advance(2_900_000_000)

    assert controller.receive_management(renewal) is False
    clock.advance(100_000_001)
    assert controller.publish(b"late", captured_monotonic_ns=10, captured_utc_ns=20) is False


def test_three_missed_heartbeat_ack_intervals_reconnect_without_resuming(
    controller: ControllerNode, handle: ControlHandle, clock: Clock, runtime: RecordingRuntime
) -> None:
    activate(controller, handle)
    old_session = controller.session_id
    for _ in range(4):
        clock.advance(controller.config.timing.heartbeat_interval_ns)
        controller.run_management_once()

    assert controller.session_id != old_session
    assert controller.control_state is ControllerControlState.IDLE
    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20) is False
    assert len(messages(runtime, "register")) == 2


def test_management_send_loss_creates_fresh_session_without_resuming_stream(
    controller: ControllerNode, handle: ControlHandle, runtime: RecordingRuntime
) -> None:
    activate(controller, handle)
    old_session = controller.session_id
    runtime.channels[-1].send_result = False

    controller.hand_over(handle)

    assert controller.session_id != old_session
    assert len(runtime.channels) == 2
    assert controller.control_state is ControllerControlState.IDLE
    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20) is False
    assert messages(runtime, "register")[-1].sender_session_id == controller.session_id
