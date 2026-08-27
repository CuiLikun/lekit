import sqlite3
import threading
from dataclasses import fields, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from lekit.control import (
    PROTOCOL_VERSION,
    ActionEnvelope,
    ControlConflict,
    ControlHandle,
    ControllerControlState,
    HandleState,
    Hub,
    HubConfig,
    HubSnapshot,
    HubStore,
    IncompatibleNode,
    ManagementMessage,
    NodeDescriptor,
    NodeReport,
    NodeRole,
    NodeUnavailable,
    ReceivedManagement,
    RobotControlState,
    RuntimeState,
)


class Clock:
    def __init__(self) -> None:
        self.monotonic = 0
        self.utc = 1_000_000_000

    def monotonic_ns(self) -> int:
        return self.monotonic

    def utc_ns(self) -> int:
        return self.utc

    def advance(self, seconds: float) -> None:
        amount = round(seconds * 1_000_000_000)
        self.monotonic += amount
        self.utc += amount


class SpyHubChannel:
    def __init__(self) -> None:
        self.inbox: list[object] = []
        self.sent: list[tuple[str, ManagementMessage]] = []
        self.send_results: dict[str, bool] = {}
        self.closed = False

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        del timeout_s
        if not self.inbox:
            return None
        value = self.inbox.pop(0)
        return value  # type: ignore[return-value]

    def send(self, peer_id: str, message: ManagementMessage) -> bool:
        self.sent.append((peer_id, message))
        return self.send_results.get(peer_id, True)

    def close(self) -> None:
        self.closed = True


class SpyRuntime:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.channel = SpyHubChannel()
        self.opened: list[tuple[str, str, str | None]] = []
        self.action_opens = 0
        self.closed = False

    def open_hub(
        self,
        endpoint: str,
        *,
        hub_epoch: str,
        advertise_endpoint: str | None = None,
    ) -> SpyHubChannel:
        self.events.append("runtime.open_hub")
        self.opened.append((endpoint, hub_epoch, advertise_endpoint))
        return self.channel

    def open_node(self, node_id: str, session_id: str, *, hub_seed: str | None):
        del node_id, session_id, hub_seed
        raise AssertionError("Hub must not open Node channels")

    def open_action_publisher(self, endpoint: str):
        del endpoint
        self.action_opens += 1
        raise AssertionError("Hub must not open the action path")

    def open_action_receiver(self, endpoint: str):
        del endpoint
        self.action_opens += 1
        raise AssertionError("Hub must not open the action path")

    def close(self) -> None:
        self.closed = True


def descriptor(
    role: NodeRole,
    *,
    session: str | None = None,
    endpoint: str | None = None,
    enabled: bool = True,
    protocol_version: int = PROTOCOL_VERSION,
    schemas: tuple[str, ...] = ("lekit.isaac_teleop.action.v1",),
    modes: tuple[str, ...] = ("teleop",),
) -> NodeDescriptor:
    controller = role is NodeRole.CONTROLLER
    node_id = "quest3-main" if controller else "piper-01"
    return NodeDescriptor(
        protocol_version=protocol_version,
        schema_version=1,
        node_id=node_id,
        session_id=session or f"{node_id}-session-1",
        role=role,
        display_name="Quest 3" if controller else "Piper",
        administratively_enabled=enabled,
        capabilities=("teleop",),
        action_schemas=schemas,
        control_modes=modes,
        action_endpoint=(endpoint or "tcp://10.0.0.8:5557") if controller else endpoint,
        observation_features={} if controller else {"joint": {"dtype": "float", "shape": [6]}},
        action_features={} if controller else {"joint": {"dtype": "float", "shape": [6]}},
        software_version="1.0.0",
        diagnostics={},
    )


def report(
    node: NodeDescriptor,
    *,
    handle: ControlHandle | None = None,
    robot_state: RobotControlState | None = None,
    controller_state: ControllerControlState | None = None,
    runtime_state: RuntimeState = RuntimeState.ONLINE,
    error: str | None = None,
    reported_at_ns: int = 999_999_999_999,
) -> NodeReport:
    return NodeReport(
        node_id=node.node_id,
        session_id=node.session_id,
        runtime_state=runtime_state,
        robot_control_state=robot_state,
        controller_control_state=controller_state,
        handle_id=handle.handle_id if handle is not None else None,
        fencing_token=handle.fencing_token if handle is not None else None,
        action_rate_hz=60.0 if handle is not None else 0.0,
        frame_age_ms=5.0 if handle is not None and node.role is NodeRole.ROBOT else None,
        last_sequence=8 if handle is not None else None,
        tracking=True if handle is not None else None,
        engaged=True if handle is not None else None,
        processor_state="active" if handle is not None and node.role is NodeRole.ROBOT else None,
        active_hold=robot_state is RobotControlState.HOLD if robot_state is not None else None,
        error=error,
        reported_at_ns=reported_at_ns,
    )


def management(
    kind: str,
    node: NodeDescriptor,
    body: dict[str, object],
    *,
    sequence: int = 0,
    correlation_id: str | None = None,
) -> ManagementMessage:
    return ManagementMessage(
        protocol_version=PROTOCOL_VERSION,
        kind=kind,
        correlation_id=correlation_id or f"{kind}-{sequence}",
        sender_id=node.node_id,
        sender_session_id=node.session_id,
        sequence=sequence,
        sent_at_ns=123_000_000_000,
        body=body,
    )


def as_wire(value: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(value):  # type: ignore[arg-type]
        item = getattr(value, field.name)
        if hasattr(item, "value"):
            item = item.value
        elif isinstance(item, MappingProxyType):
            item = dict(item)
        result[field.name] = item
    return result


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def runtime() -> SpyRuntime:
    return SpyRuntime()


@pytest.fixture
def hub(tmp_path: Path, clock: Clock, runtime: SpyRuntime) -> Hub:
    instance = Hub(
        HubConfig(management_endpoint="memory://hub", database_path=tmp_path / "hub.sqlite3"),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    instance.start()
    return instance


@pytest.fixture
def registered_pair(hub: Hub, clock: Clock) -> tuple[NodeDescriptor, NodeDescriptor]:
    robot = descriptor(NodeRole.ROBOT)
    controller = descriptor(NodeRole.CONTROLLER)
    hub.register(robot, received_monotonic_ns=clock.monotonic_ns())
    hub.register(controller, received_monotonic_ns=clock.monotonic_ns())
    return robot, controller


def activate(hub: Hub, pair: tuple[NodeDescriptor, NodeDescriptor]) -> ControlHandle:
    robot, controller = pair
    handle = hub.assign(robot.node_id, controller.node_id)
    hub.receive_report(
        report(robot, handle=handle, robot_state=RobotControlState.HOLD),
    )
    hub.receive_report(
        report(controller, handle=handle, controller_state=ControllerControlState.STREAMING),
    )
    return handle


def node_snapshot(hub: Hub, node_id: str):
    return next(node for node in hub.list_nodes() if node.descriptor.node_id == node_id)


def sent_messages(
    runtime: SpyRuntime,
    *,
    kind: str | None = None,
    peer_id: str | None = None,
) -> list[tuple[str, ManagementMessage]]:
    return [
        (peer, message)
        for peer, message in runtime.channel.sent
        if (kind is None or message.kind == kind) and (peer_id is None or peer == peer_id)
    ]


def fail_audit_event(hub: Hub, event: str) -> None:
    store = hub._store
    assert store is not None
    original = store.append_audit

    def append_audit(**kwargs):
        if kwargs["event"] == event:
            raise sqlite3.OperationalError("forced audit failure")
        return original(**kwargs)

    store.append_audit = append_audit  # type: ignore[method-assign]


def test_start_creates_epoch_before_opening_runtime_and_never_opens_action_path(tmp_path: Path, clock: Clock):
    events: list[str] = []
    runtime = SpyRuntime(events)
    store = HubStore(tmp_path / "hub.sqlite3")
    original_begin_epoch = store.begin_epoch

    def begin_epoch(*, started_at_ns: int) -> str:
        events.append("store.begin_epoch")
        return original_begin_epoch(started_at_ns=started_at_ns)

    store.begin_epoch = begin_epoch  # type: ignore[method-assign]
    hub = Hub(
        HubConfig(management_endpoint="memory://hub", database_path=tmp_path / "ignored.sqlite3"),
        runtime=runtime,
        store=store,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )

    hub.start()

    assert events == ["store.begin_epoch", "runtime.open_hub"]
    assert runtime.opened[0][1] == hub.hub_epoch
    assert runtime.action_opens == 0
    assert hub.list_nodes() == ()


def test_wildcard_management_bind_without_advertisement_disables_discovery(tmp_path: Path, clock: Clock):
    runtime = SpyRuntime()
    hub = Hub(
        HubConfig(management_endpoint="tcp://0.0.0.0:5560", database_path=tmp_path / "hub.sqlite3"),
        runtime=runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )

    hub.start()

    assert runtime.opened[0][2] is None


def test_assign_checks_schema_mode_sessions_and_exclusivity(hub: Hub, registered_pair):
    handle = hub.assign("piper-01", "quest3-main")

    assert handle.action_schema == "lekit.isaac_teleop.action.v1"
    assert handle.robot_session_id == "piper-01-session-1"
    assert handle.controller_session_id == "quest3-main-session-1"
    with pytest.raises(ControlConflict):
        hub.assign("piper-01", "quest3-main")


def test_assignment_does_not_treat_pre_grant_reports_as_handle_mismatches(
    hub: Hub,
    runtime: SpyRuntime,
    registered_pair: tuple[NodeDescriptor, NodeDescriptor],
) -> None:
    robot, controller = registered_pair
    hub.receive_report(report(robot, robot_state=RobotControlState.HOLD))
    hub.receive_report(report(controller, controller_state=ControllerControlState.IDLE))

    handle = hub.assign(robot.node_id, controller.node_id)
    hub.tick()

    control = hub.get_snapshot(handle.handle_id)
    assert control.handle_state is HandleState.ASSIGNED
    assert control.mismatch_codes == ()
    assert sent_messages(runtime, kind="revoke") == []


def test_grant_ack_discards_unbound_status_that_was_in_flight_before_grant(
    hub: Hub,
    runtime: SpyRuntime,
    registered_pair: tuple[NodeDescriptor, NodeDescriptor],
) -> None:
    robot, controller = registered_pair
    handle = hub.assign(robot.node_id, controller.node_id)
    hub.receive_report(report(robot, robot_state=RobotControlState.HOLD))
    hub.receive_report(report(controller, controller_state=ControllerControlState.IDLE))

    hub.tick()
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ASSIGNED

    for node in (robot, controller):
        grant = sent_messages(runtime, kind="grant", peer_id=node.node_id)[-1][1]
        ack = management(
            "grant_ack",
            node,
            {"handle": as_wire(handle)},
            sequence=1,
            correlation_id=grant.correlation_id,
        )
        runtime.channel.inbox.append(
            ReceivedManagement(node.node_id, ack, "10.0.0.8", 0)
        )
        assert hub.run_once() is True

    hub.tick()

    control = hub.get_snapshot(handle.handle_id)
    assert control.handle_state is HandleState.ASSIGNED
    assert control.mismatch_codes == ()


def test_assignment_rejects_incompatible_mode_and_schema(hub: Hub, clock: Clock):
    robot = descriptor(NodeRole.ROBOT, modes=("policy",))
    controller = descriptor(NodeRole.CONTROLLER, schemas=("other.action.v1",))
    hub.register(robot, received_monotonic_ns=clock.monotonic_ns())
    hub.register(controller, received_monotonic_ns=clock.monotonic_ns())

    with pytest.raises(IncompatibleNode, match="control mode"):
        hub.assign(robot.node_id, controller.node_id)


@pytest.mark.parametrize(
    ("runtime_state", "robot_state", "error"),
    [
        (RuntimeState.ONLINE, RobotControlState.SAFETY, None),
        (RuntimeState.FAULT, RobotControlState.HOLD, "fault"),
        (RuntimeState.DEGRADED, RobotControlState.HOLD, "control unavailable"),
    ],
)
def test_assignment_rejects_unsafe_robot_status(
    hub: Hub,
    clock: Clock,
    runtime_state: RuntimeState,
    robot_state: RobotControlState,
    error: str | None,
):
    robot = descriptor(NodeRole.ROBOT)
    controller = descriptor(NodeRole.CONTROLLER)
    hub.register(robot, received_monotonic_ns=clock.monotonic_ns())
    hub.register(controller, received_monotonic_ns=clock.monotonic_ns())
    hub.receive_report(
        report(robot, runtime_state=runtime_state, robot_state=robot_state, error=error),
        received_monotonic_ns=clock.monotonic_ns(),
    )

    with pytest.raises(NodeUnavailable):
        hub.assign(robot.node_id, controller.node_id)


def test_assignment_rejects_administratively_disabled_robot(hub: Hub, clock: Clock):
    robot = descriptor(NodeRole.ROBOT, enabled=False)
    controller = descriptor(NodeRole.CONTROLLER)
    hub.register(robot, received_monotonic_ns=clock.monotonic_ns())
    hub.register(controller, received_monotonic_ns=clock.monotonic_ns())

    with pytest.raises(NodeUnavailable, match="disabled"):
        hub.assign(robot.node_id, controller.node_id)


def test_registration_resolves_controller_wildcard_endpoint_from_management_peer(hub: Hub, clock: Clock):
    controller = descriptor(NodeRole.CONTROLLER, endpoint="tcp://0.0.0.0:5557")

    registered = hub.register(
        controller,
        peer_host="10.42.0.9",
        received_monotonic_ns=clock.monotonic_ns(),
    )

    assert registered.action_endpoint == "tcp://10.42.0.9:5557"


def test_registration_rejects_unresolved_wildcard_and_protocol_mismatch(hub: Hub, clock: Clock):
    wildcard = descriptor(NodeRole.CONTROLLER, endpoint="tcp://0.0.0.0:5557")
    incompatible = descriptor(NodeRole.ROBOT, protocol_version=PROTOCOL_VERSION + 1)

    with pytest.raises(IncompatibleNode, match="wildcard"):
        hub.register(wildcard, received_monotonic_ns=clock.monotonic_ns())
    with pytest.raises(IncompatibleNode, match="protocol"):
        hub.register(incompatible, received_monotonic_ns=clock.monotonic_ns())


def test_registration_rejects_controller_with_robot_feature_metadata(hub: Hub, clock: Clock):
    controller = replace(
        descriptor(NodeRole.CONTROLLER),
        observation_features={"joint": {"dtype": "float", "shape": [6]}},
    )

    with pytest.raises(IncompatibleNode, match="feature metadata"):
        hub.register(controller, received_monotonic_ns=clock.monotonic_ns())


def test_superseded_session_cannot_register_itself_current_again(hub: Hub, clock: Clock):
    first = descriptor(NodeRole.ROBOT, session="robot-session-1")
    second = replace(first, session_id="robot-session-2")
    hub.register(first, received_monotonic_ns=clock.monotonic_ns())
    hub.register(second, received_monotonic_ns=clock.monotonic_ns())

    with pytest.raises(ControlConflict, match="stale session"):
        hub.register(first, received_monotonic_ns=clock.monotonic_ns())

    assert node_snapshot(hub, first.node_id).descriptor.session_id == "robot-session-2"


def test_delayed_same_session_registration_cannot_erase_newer_status(
    hub: Hub, runtime: SpyRuntime, clock: Clock
):
    robot = descriptor(NodeRole.ROBOT)
    registration = management("register", robot, {"descriptor": as_wire(robot)}, sequence=0)
    status_report = report(robot, robot_state=RobotControlState.HOLD)
    status = management("status", robot, {"report": as_wire(status_report)}, sequence=5)
    delayed_registration = management(
        "register",
        robot,
        {"descriptor": as_wire(robot)},
        sequence=1,
        correlation_id="delayed-register",
    )
    runtime.channel.inbox.extend(
        [
            ReceivedManagement(robot.node_id, registration, "10.0.0.4", clock.monotonic_ns()),
            ReceivedManagement(robot.node_id, status, "10.0.0.4", clock.monotonic_ns()),
            ReceivedManagement(
                robot.node_id,
                delayed_registration,
                "10.0.0.4",
                clock.monotonic_ns(),
            ),
        ]
    )

    hub.run_once()
    hub.run_once()
    hub.run_once()

    assert node_snapshot(hub, robot.node_id).report == status_report


def test_active_requires_both_ready_and_streaming_reports(hub: Hub, registered_pair):
    robot, controller = registered_pair
    handle = hub.assign(robot.node_id, controller.node_id)

    hub.receive_report(report(robot, handle=handle, robot_state=RobotControlState.HOLD))
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.TAKING_OVER

    hub.receive_report(report(controller, handle=handle, controller_state=ControllerControlState.STREAMING))
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ACTIVE


def test_streaming_without_robot_ready_does_not_activate(hub: Hub, registered_pair):
    robot, controller = registered_pair
    handle = hub.assign(robot.node_id, controller.node_id)

    hub.receive_report(report(controller, handle=handle, controller_state=ControllerControlState.STREAMING))

    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ASSIGNED


def test_wire_status_cannot_replace_explicit_ready_and_streaming_acknowledgements(
    hub: Hub, runtime: SpyRuntime, clock: Clock, registered_pair
):
    robot, controller = registered_pair
    handle = hub.assign(robot.node_id, controller.node_id)
    robot_status = management(
        "status",
        robot,
        {"report": as_wire(report(robot, handle=handle, robot_state=RobotControlState.HOLD))},
        sequence=1,
    )
    controller_status = management(
        "status",
        controller,
        {
            "report": as_wire(
                report(
                    controller,
                    handle=handle,
                    controller_state=ControllerControlState.STREAMING,
                )
            )
        },
        sequence=1,
    )
    runtime.channel.inbox.extend(
        [
            ReceivedManagement(robot.node_id, robot_status, "10.0.0.4", clock.monotonic_ns()),
            ReceivedManagement(
                controller.node_id,
                controller_status,
                "10.0.0.8",
                clock.monotonic_ns(),
            ),
        ]
    )

    assert hub.run_once() is True
    assert hub.run_once() is True

    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ASSIGNED

    robot_ready = management(
        "robot_ready",
        robot,
        {"handle": as_wire(handle)},
        sequence=2,
    )
    controller_streaming = management(
        "controller_streaming",
        controller,
        {"handle": as_wire(handle)},
        sequence=2,
    )
    runtime.channel.inbox.extend(
        [
            ReceivedManagement(robot.node_id, robot_ready, "10.0.0.4", clock.monotonic_ns()),
            ReceivedManagement(
                controller.node_id,
                controller_streaming,
                "10.0.0.8",
                clock.monotonic_ns(),
            ),
        ]
    )
    hub.run_once()
    hub.run_once()
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ACTIVE


def test_three_missed_heartbeats_uses_received_time_marks_offline_and_revokes(
    hub: Hub, runtime: SpyRuntime, clock: Clock, registered_pair
):
    handle = activate(hub, registered_pair)
    clock.advance(1.51)

    hub.tick()

    assert node_snapshot(hub, "quest3-main").online is False
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.REVOKING
    assert any(
        peer == handle.controller_id and message.kind == "revoke" for peer, message in runtime.channel.sent
    )
    assert any(
        peer == handle.robot_id and message.kind == "force_hold" for peer, message in runtime.channel.sent
    )


def test_sender_reported_clock_cannot_keep_node_online(hub: Hub, clock: Clock, registered_pair):
    robot, _controller = registered_pair
    hub.receive_report(
        report(robot, robot_state=RobotControlState.HOLD, reported_at_ns=10**18),
        received_monotonic_ns=0,
    )
    clock.advance(1.51)

    hub.tick()

    assert node_snapshot(hub, robot.node_id).online is False


def test_invalid_handle_report_cannot_refresh_liveness(hub: Hub, clock: Clock, registered_pair):
    robot, _controller = registered_pair
    handle = hub.assign("piper-01", "quest3-main")
    clock.advance(1.4)
    stale = replace(handle, fencing_token=handle.fencing_token + 1)
    with pytest.raises(ControlConflict, match="fencing"):
        hub.receive_report(
            report(robot, handle=stale, robot_state=RobotControlState.HOLD),
            received_monotonic_ns=clock.monotonic_ns(),
        )
    clock.advance(0.11)

    hub.tick()

    assert node_snapshot(hub, robot.node_id).online is False


def test_robot_hold_is_sufficient_to_finish_release_when_controller_ack_is_lost(hub: Hub, registered_pair):
    robot, _controller = registered_pair
    handle = activate(hub, registered_pair)
    hub.request_hand_over(handle)

    hub.receive_report(report(robot, handle=handle, robot_state=RobotControlState.HOLD))

    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.RELEASED


def test_revocation_transitions_through_revoking_and_robot_hold_finishes_it(hub: Hub, registered_pair):
    robot, _controller = registered_pair
    handle = activate(hub, registered_pair)

    hub.revoke(handle, reason="operator request", actor="operator-1")
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.REVOKING

    hub.receive_report(report(robot, handle=handle, robot_state=RobotControlState.HOLD))
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.REVOKED


def test_old_session_and_handle_messages_cannot_reactivate_revoked_handle(
    hub: Hub, clock: Clock, registered_pair
):
    robot, controller = registered_pair
    handle = activate(hub, registered_pair)
    hub.revoke(handle, reason="test")
    hub.receive_report(report(robot, handle=handle, robot_state=RobotControlState.HOLD))
    replacement = replace(controller, session_id="quest3-main-session-2")
    hub.register(replacement, received_monotonic_ns=clock.monotonic_ns())

    with pytest.raises(ControlConflict, match="session"):
        hub.receive_report(
            report(controller, handle=handle, controller_state=ControllerControlState.STREAMING)
        )
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.REVOKED


def test_tick_renews_once_per_interval_without_changing_identity(hub: Hub, clock: Clock, registered_pair):
    handle = hub.assign("piper-01", "quest3-main")
    original_expiry = handle.expires_at_ns
    clock.advance(1.01)

    hub.tick()
    renewed = hub.renew(handle)

    assert renewed.handle_id == handle.handle_id
    assert renewed.fencing_token == handle.fencing_token
    assert renewed.expires_at_ns > original_expiry


def test_invalid_actor_is_rejected_before_assignment_mutates_authority(hub: Hub, registered_pair):
    with pytest.raises(ValueError, match="actor"):
        hub.assign("piper-01", "quest3-main", actor="")

    handle = hub.assign("piper-01", "quest3-main", actor="operator-1")
    assert handle.robot_id == "piper-01"


def test_heartbeat_ack_uses_request_correlation_and_current_epoch(
    hub: Hub, runtime: SpyRuntime, registered_pair, clock: Clock
):
    _robot, controller = registered_pair
    request = management("heartbeat", controller, {}, sequence=1, correlation_id="heartbeat-correlation")
    runtime.channel.inbox.append(
        ReceivedManagement(controller.node_id, request, "10.0.0.8", clock.monotonic_ns())
    )

    assert hub.run_once() is True

    peer, reply = runtime.channel.sent[-1]
    assert peer == controller.node_id
    assert reply.kind == "heartbeat_ack"
    assert reply.correlation_id == "heartbeat-correlation"
    assert reply.body["hub_epoch"] == hub.hub_epoch


def test_run_once_registers_and_dispatches_status_from_management_messages(
    hub: Hub, runtime: SpyRuntime, clock: Clock
):
    robot = descriptor(NodeRole.ROBOT)
    registration = management("register", robot, {"descriptor": as_wire(robot)})
    runtime.channel.inbox.append(
        ReceivedManagement(robot.node_id, registration, "10.0.0.4", clock.monotonic_ns())
    )
    assert hub.run_once() is True

    status_report = report(robot, robot_state=RobotControlState.HOLD)
    status = management("status", robot, {"report": as_wire(status_report)}, sequence=1)
    runtime.channel.inbox.append(ReceivedManagement(robot.node_id, status, "10.0.0.4", clock.monotonic_ns()))
    assert hub.run_once() is True

    snapshot = node_snapshot(hub, robot.node_id)
    assert snapshot.report is not None
    assert snapshot.report.robot_control_state is RobotControlState.HOLD


def test_service_loop_survives_stale_unregistered_node_packet(
    hub: Hub, runtime: SpyRuntime, clock: Clock
) -> None:
    stale = descriptor(NodeRole.ROBOT, session="stale-session")
    message = management("status", stale, {}, sequence=1)
    runtime.channel.inbox.append(
        ReceivedManagement(stale.node_id, message, "127.0.0.1", clock.monotonic_ns())
    )

    class StopAfterOneIteration:
        calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls > 1

    hub.run(stop_event=StopAfterOneIteration())

    assert runtime.channel.closed is True


@pytest.mark.parametrize("kind", ["status", "fault"])
def test_report_form_message_identity_is_bound_to_authenticated_sender(
    kind: str,
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
    registered_pair,
):
    robot, controller = registered_pair
    handle = activate(hub, registered_pair)
    malicious_report = report(
        robot,
        handle=handle,
        robot_state=RobotControlState.SAFETY,
        runtime_state=RuntimeState.ONLINE,
    )
    message = management(kind, controller, {"report": as_wire(malicious_report)}, sequence=1)
    before_robot = node_snapshot(hub, robot.node_id)
    sent_before = len(runtime.channel.sent)
    clock.advance(1.0)
    runtime.channel.inbox.append(
        ReceivedManagement(controller.node_id, message, "10.0.0.8", clock.monotonic_ns())
    )

    with pytest.raises(ControlConflict, match="sender"):
        hub.run_once()

    after_robot = node_snapshot(hub, robot.node_id)
    assert after_robot.last_seen_ns == before_robot.last_seen_ns
    assert after_robot.report == before_robot.report
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ACTIVE
    assert len(runtime.channel.sent) == sent_before


@pytest.mark.parametrize(
    ("kind", "sender_role"),
    [
        ("robot_ready", NodeRole.ROBOT),
        ("controller_streaming", NodeRole.CONTROLLER),
        ("robot_holding", NodeRole.ROBOT),
        ("controller_released", NodeRole.CONTROLLER),
    ],
)
def test_terminal_handle_lifecycle_message_is_rejected_before_liveness_or_routing(
    kind: str,
    sender_role: NodeRole,
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
    registered_pair,
):
    robot, controller = registered_pair
    handle = activate(hub, registered_pair)
    hub.revoke(handle, reason="terminal test")
    hub.receive_report(report(robot, handle=handle, robot_state=RobotControlState.HOLD))
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.REVOKED
    sender = robot if sender_role is NodeRole.ROBOT else controller
    before = node_snapshot(hub, sender.node_id)
    sent_before = len(runtime.channel.sent)
    clock.advance(1.0)
    message = management(kind, sender, {"handle": as_wire(handle)}, sequence=1)
    runtime.channel.inbox.append(
        ReceivedManagement(sender.node_id, message, "10.0.0.8", clock.monotonic_ns())
    )

    with pytest.raises(ControlConflict, match="terminal"):
        hub.run_once()

    after = node_snapshot(hub, sender.node_id)
    assert after.last_seen_ns == before.last_seen_ns
    assert len(runtime.channel.sent) == sent_before


def test_run_once_rejects_action_envelope_even_if_runtime_breaks_its_protocol(hub: Hub, runtime: SpyRuntime):
    runtime.channel.inbox.append(
        ReceivedManagement(
            "quest3-main",
            ActionEnvelope(
                handle_id="handle-1",
                hub_epoch=hub.hub_epoch,
                fencing_token=1,
                controller_id="quest3-main",
                controller_session_id="quest3-main-session-1",
                stream_session_id="stream-1",
                sequence=0,
                captured_monotonic_ns=0,
                captured_utc_ns=0,
                payload_schema="lekit.action.v1",
                payload=b"opaque",
            ),
            None,
            0,
        )
    )

    with pytest.raises(TypeError, match="management"):
        hub.run_once()


def test_force_hold_without_handle_is_routed_and_always_audited(
    hub: Hub, runtime: SpyRuntime, registered_pair
):
    hub.force_hold("piper-01", reason="safety check", actor="operator-1")

    peer, command = runtime.channel.sent[-1]
    assert peer == "piper-01"
    assert command.kind == "force_hold"
    assert command.body["reason"] == "safety check"
    history = hub.list_history()
    assert history[0]["event"] == "force_hold_requested"
    assert history[0]["actor"] == "operator-1"


def test_public_force_hold_does_not_send_when_durable_audit_fails(
    hub: Hub,
    runtime: SpyRuntime,
    registered_pair,
):
    sent_before = len(runtime.channel.sent)
    fail_audit_event(hub, "force_hold_requested")

    with pytest.raises(sqlite3.OperationalError, match="forced audit failure"):
        hub.force_hold("piper-01", reason="audit gate", actor="operator-1")

    assert len(runtime.channel.sent) == sent_before


def test_force_hold_is_distinct_from_revocation(hub: Hub, runtime: SpyRuntime, registered_pair):
    handle = activate(hub, registered_pair)

    hub.force_hold(handle.robot_id, reason="pause", actor="operator-1")

    assert runtime.channel.sent[-1][1].kind == "force_hold"
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ACTIVE


def test_handing_over_transient_is_not_auto_revoked(
    hub: Hub,
    runtime: SpyRuntime,
    registered_pair: tuple[NodeDescriptor, NodeDescriptor],
) -> None:
    handle = activate(hub, registered_pair)

    hub.request_hand_over(handle)
    hub.tick()

    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.HANDING_OVER
    assert sent_messages(runtime, kind="revoke") == []


def test_failed_grant_delivery_is_retried_by_tick(hub: Hub, runtime: SpyRuntime, registered_pair):
    runtime.channel.send_results["quest3-main"] = False
    handle = hub.assign("piper-01", "quest3-main")
    initial_attempts = [
        message for peer, message in runtime.channel.sent if peer == "quest3-main" and message.kind == "grant"
    ]
    runtime.channel.send_results["quest3-main"] = True

    hub.tick()

    attempts = [
        message for peer, message in runtime.channel.sent if peer == "quest3-main" and message.kind == "grant"
    ]
    assert len(initial_attempts) == 1
    assert len(attempts) == 2
    assert attempts[-1].body["handle"]["handle_id"] == handle.handle_id


def test_failed_grant_delivery_is_audited_alerted_and_cleared_after_retry(
    hub: Hub,
    runtime: SpyRuntime,
    registered_pair,
):
    runtime.channel.send_results["quest3-main"] = False
    handle = hub.assign("piper-01", "quest3-main")

    pending = [alert for alert in hub.get_snapshot().alerts if alert["code"] == "management_delivery_pending"]
    assert pending == [
        {
            "code": "management_delivery_pending",
            "correlation_id": f"assignment:{handle.handle_id}",
            "handle_id": handle.handle_id,
            "kind": "grant",
            "node_id": "quest3-main",
            "severity": "error",
        }
    ]
    failure = next(event for event in hub.list_history() if event["event"] == "management_delivery_failed")
    assert failure["details"]["kind"] == "grant"
    assert failure["details"]["node_id"] == "quest3-main"

    runtime.channel.send_results["quest3-main"] = True
    hub.tick()

    assert not any(
        alert["code"] == "management_delivery_pending"
        and alert["kind"] == "grant"
        and alert["node_id"] == "quest3-main"
        for alert in hub.get_snapshot().alerts
    )


@pytest.mark.parametrize("command", ["take_over", "force_hold"])
def test_failed_lifecycle_delivery_is_audited_and_remains_visible(
    command: str,
    hub: Hub,
    runtime: SpyRuntime,
    registered_pair,
):
    handle = hub.assign("piper-01", "quest3-main")
    target = handle.controller_id if command == "take_over" else handle.robot_id
    runtime.channel.send_results[target] = False

    if command == "take_over":
        hub.request_take_over(handle)
    else:
        hub.force_hold(handle.robot_id, reason="delivery test")

    assert any(
        alert["code"] == "management_delivery_pending"
        and alert["kind"] == command
        and alert["node_id"] == target
        for alert in hub.get_snapshot().alerts
    )
    assert any(
        event["event"] == "management_delivery_failed"
        and event["details"]["kind"] == command
        and event["details"]["node_id"] == target
        for event in hub.list_history()
    )


def test_failed_registration_reply_is_immediately_audited_and_visible(
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
):
    robot = descriptor(NodeRole.ROBOT)
    runtime.channel.send_results[robot.node_id] = False
    registration = management("register", robot, {"descriptor": as_wire(robot)})
    runtime.channel.inbox.append(
        ReceivedManagement(robot.node_id, registration, "10.0.0.4", clock.monotonic_ns())
    )

    assert hub.run_once() is True

    assert any(
        alert["code"] == "management_delivery_pending"
        and alert["kind"] == "registered"
        and alert["node_id"] == robot.node_id
        for alert in hub.get_snapshot().alerts
    )
    assert any(
        event["event"] == "management_delivery_failed" and event["details"]["kind"] == "registered"
        for event in hub.list_history()
    )


def trigger_automatic_revoke(
    source: str,
    hub: Hub,
    clock: Clock,
    registered_pair: tuple[NodeDescriptor, NodeDescriptor],
) -> None:
    robot, controller = registered_pair
    handle = next(control for control in hub.get_snapshot().controls).handle_id
    live_handle = hub._handles[handle].record.handle
    if source == "session_replacement":
        hub.register(
            replace(controller, session_id="quest3-main-session-2"),
            received_monotonic_ns=clock.monotonic_ns(),
        )
    elif source == "mismatch":
        hub.receive_report(report(robot, handle=live_handle, robot_state=RobotControlState.HOLD))
        hub.tick()
    elif source == "safety":
        hub.receive_report(report(robot, handle=live_handle, robot_state=RobotControlState.SAFETY))
    elif source == "offline":
        clock.advance(1.51)
        hub.tick()
    else:
        raise AssertionError(source)


@pytest.mark.parametrize("source", ["session_replacement", "mismatch", "safety", "offline"])
def test_automatic_revocation_is_centralized_routed_audited_and_idempotent(
    source: str,
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
    registered_pair,
):
    handle = activate(hub, registered_pair)
    sent_before = len(runtime.channel.sent)

    trigger_automatic_revoke(source, hub, clock, registered_pair)

    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.REVOKING
    automatic_messages = runtime.channel.sent[sent_before:]
    assert [(peer, message.kind) for peer, message in automatic_messages] == [
        (handle.controller_id, "revoke"),
        (handle.robot_id, "revoke"),
        (handle.robot_id, "force_hold"),
    ]
    history = hub.list_history()
    assert sum(event["event"] == "automatic_revocation_requested" for event in history) == 1
    assert sum(event["event"] == "force_hold_requested" for event in history) == 1

    sent_after_first = len(runtime.channel.sent)
    trigger_automatic_revoke(source, hub, clock, registered_pair)
    assert len(runtime.channel.sent) == sent_after_first
    history = hub.list_history()
    assert sum(event["event"] == "automatic_revocation_requested" for event in history) == 1
    assert sum(event["event"] == "force_hold_requested" for event in history) == 1


@pytest.mark.parametrize("source", ["session_replacement", "mismatch", "safety", "offline"])
def test_automatic_force_hold_does_not_send_when_durable_audit_fails(
    source: str,
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
    registered_pair,
):
    handle = activate(hub, registered_pair)
    fail_audit_event(hub, "force_hold_requested")

    with pytest.raises(sqlite3.OperationalError, match="forced audit failure"):
        trigger_automatic_revoke(source, hub, clock, registered_pair)

    assert not sent_messages(runtime, kind="force_hold", peer_id=handle.robot_id)


def test_automatic_robot_commands_have_distinct_correlations_and_independent_acks(
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
    registered_pair,
):
    robot, _controller = registered_pair
    handle = activate(hub, registered_pair)
    clock.advance(1.51)
    hub.tick()
    robot_revoke = sent_messages(runtime, kind="revoke", peer_id=handle.robot_id)[-1][1]
    robot_force_hold = sent_messages(runtime, kind="force_hold", peer_id=handle.robot_id)[-1][1]

    assert robot_revoke.correlation_id != robot_force_hold.correlation_id

    for sequence, (kind, correlation_id) in enumerate(
        [
            ("revoke_ack", robot_revoke.correlation_id),
            ("force_hold_ack", robot_force_hold.correlation_id),
        ],
        start=1,
    ):
        ack = management(
            kind,
            robot,
            {"handle": as_wire(handle)},
            sequence=sequence,
            correlation_id=correlation_id,
        )
        runtime.channel.inbox.append(ReceivedManagement(robot.node_id, ack, "10.0.0.4", clock.monotonic_ns()))
        assert hub.run_once() is True

    evidence = [
        event
        for event in hub.list_history()
        if event["event"] == "command_acknowledged" and event["details"]["node_id"] == robot.node_id
    ]
    assert {event["details"]["kind"] for event in evidence} == {"revoke_ack", "force_hold_ack"}


def outstanding_take_over(
    hub: Hub,
    runtime: SpyRuntime,
    registered_pair: tuple[NodeDescriptor, NodeDescriptor],
) -> tuple[ControlHandle, ManagementMessage]:
    handle = hub.assign("piper-01", "quest3-main")
    hub.request_take_over(handle)
    command = sent_messages(runtime, kind="take_over", peer_id=handle.controller_id)[-1][1]
    return handle, command


@pytest.mark.parametrize(
    "mismatch",
    [
        "unknown_correlation",
        "wrong_kind",
        "wrong_peer_role",
        "wrong_session",
        "wrong_handle_id",
        "wrong_epoch",
        "wrong_fencing",
    ],
)
def test_ack_mismatch_cannot_refresh_liveness_or_create_success_evidence(
    mismatch: str,
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
    registered_pair,
):
    robot, controller = registered_pair
    handle, command = outstanding_take_over(hub, runtime, registered_pair)
    sender = controller
    kind = "take_over_ack"
    correlation_id = command.correlation_id
    body_handle = as_wire(handle)
    if mismatch == "unknown_correlation":
        correlation_id = "unknown-correlation"
    elif mismatch == "wrong_kind":
        kind = "hand_over_ack"
    elif mismatch == "wrong_peer_role":
        sender = robot
    elif mismatch == "wrong_session":
        sender = replace(controller, session_id="quest3-main-stale-session")
    elif mismatch == "wrong_handle_id":
        body_handle["handle_id"] = "wrong-handle"
    elif mismatch == "wrong_epoch":
        body_handle["hub_epoch"] = "wrong-epoch"
    elif mismatch == "wrong_fencing":
        body_handle["fencing_token"] = handle.fencing_token + 1
    before = node_snapshot(hub, sender.node_id if mismatch != "wrong_session" else controller.node_id)
    clock.advance(1.0)
    ack = management(
        kind,
        sender,
        {"handle": body_handle},
        sequence=1,
        correlation_id=correlation_id,
    )
    runtime.channel.inbox.append(ReceivedManagement(sender.node_id, ack, "10.0.0.8", clock.monotonic_ns()))

    with pytest.raises(ControlConflict, match="ack|session"):
        hub.run_once()

    after = node_snapshot(hub, before.descriptor.node_id)
    assert after.last_seen_ns == before.last_seen_ns
    assert not any(event["event"] == "command_acknowledged" for event in hub.list_history())


def test_exact_outstanding_ack_is_evidence_once_and_duplicate_is_rejected(
    hub: Hub,
    runtime: SpyRuntime,
    clock: Clock,
    registered_pair,
):
    _robot, controller = registered_pair
    handle, command = outstanding_take_over(hub, runtime, registered_pair)
    clock.advance(1.0)
    ack = management(
        "take_over_ack",
        controller,
        {"handle": as_wire(handle)},
        sequence=1,
        correlation_id=command.correlation_id,
    )
    runtime.channel.inbox.append(
        ReceivedManagement(controller.node_id, ack, "10.0.0.8", clock.monotonic_ns())
    )

    assert hub.run_once() is True
    assert node_snapshot(hub, controller.node_id).last_seen_ns == clock.monotonic_ns()
    evidence = [event for event in hub.list_history() if event["event"] == "command_acknowledged"]
    assert len(evidence) == 1
    assert evidence[0]["details"]["kind"] == "take_over_ack"

    duplicate = replace(ack, sequence=2)
    runtime.channel.inbox.append(
        ReceivedManagement(controller.node_id, duplicate, "10.0.0.8", clock.monotonic_ns())
    )
    with pytest.raises(ControlConflict, match="ack"):
        hub.run_once()
    evidence = [event for event in hub.list_history() if event["event"] == "command_acknowledged"]
    assert len(evidence) == 1


def test_revoking_handle_is_not_renewed_and_expires_without_robot_hold(
    hub: Hub, clock: Clock, registered_pair
):
    handle = activate(hub, registered_pair)
    hub.revoke(handle, reason="lost acknowledgement")
    clock.advance(1.01)
    hub.tick()
    expiry_after_tick = hub.get_snapshot(handle.handle_id).expires_at_ns
    assert expiry_after_tick == handle.expires_at_ns

    clock.advance(2.0)
    hub.tick()
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.EXPIRED


def test_restart_invalidates_old_handles_without_restoring_persisted_liveness(tmp_path: Path, clock: Clock):
    database_path = tmp_path / "hub.sqlite3"
    first_runtime = SpyRuntime()
    first = Hub(
        HubConfig(management_endpoint="memory://hub", database_path=database_path),
        runtime=first_runtime,
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    first.start()
    robot = descriptor(NodeRole.ROBOT)
    controller = descriptor(NodeRole.CONTROLLER)
    first.register(robot, received_monotonic_ns=0)
    first.register(controller, received_monotonic_ns=0)
    handle = first.assign(robot.node_id, controller.node_id)
    first.stop()

    second = Hub(
        HubConfig(management_endpoint="memory://hub", database_path=database_path),
        runtime=SpyRuntime(),
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )
    second.start()

    assert second.list_nodes() == ()
    with pytest.raises(KeyError):
        second.get_snapshot(handle.handle_id)
    assert HubStore(database_path).get_handle(handle.handle_id).state is HandleState.EXPIRED


def test_start_rejects_wildcard_advertised_endpoint(tmp_path: Path, clock: Clock):
    hub = Hub(
        HubConfig(
            management_endpoint="tcp://0.0.0.0:5560",
            advertise_endpoint="tcp://0.0.0.0:5560",
            database_path=tmp_path / "hub.sqlite3",
        ),
        runtime=SpyRuntime(),
        monotonic_ns=clock.monotonic_ns,
        utc_ns=clock.utc_ns,
    )

    with pytest.raises(ValueError, match="non-wildcard"):
        hub.start()


def test_low_rate_snapshot_persistence_is_limited_to_one_hz(hub: Hub, clock: Clock, registered_pair):
    robot, _controller = registered_pair
    hub.receive_report(report(robot, robot_state=RobotControlState.HOLD))
    hub.tick()
    hub.receive_report(report(robot, robot_state=RobotControlState.HOLD))
    hub.tick()

    with sqlite3.connect(hub.config.database_path) as connection:
        first = connection.execute("SELECT snapshot_json FROM snapshots").fetchone()
    assert first is not None

    clock.advance(0.99)
    hub.tick()
    with sqlite3.connect(hub.config.database_path) as connection:
        before_one_second = connection.execute("SELECT snapshot_json FROM snapshots").fetchone()[0]
    assert before_one_second == first[0]

    clock.advance(0.02)
    hub.tick()
    with sqlite3.connect(hub.config.database_path) as connection:
        after_one_second = connection.execute("SELECT snapshot_json FROM snapshots").fetchone()[0]
    assert after_one_second != before_one_second


def test_watch_returns_after_snapshot_version_advances(hub: Hub, clock: Clock):
    initial = hub.get_snapshot()
    assert isinstance(initial, HubSnapshot)
    result: list[HubSnapshot] = []
    waiting = threading.Event()

    def watch() -> None:
        waiting.set()
        result.append(hub.watch(after_version=initial.version, timeout_s=1.0))

    thread = threading.Thread(target=watch)
    thread.start()
    assert waiting.wait(timeout=0.1)
    hub.register(descriptor(NodeRole.ROBOT), received_monotonic_ns=clock.monotonic_ns())
    thread.join(timeout=0.2)

    assert thread.is_alive() is False
    assert result[0].version > initial.version


def test_stop_closes_only_management_channel_without_restoring_nodes(
    hub: Hub, runtime: SpyRuntime, registered_pair
):
    hub.stop()

    assert runtime.channel.closed is True
    assert runtime.closed is False
    assert hub.list_nodes() == ()
