import sqlite3
from dataclasses import replace

import pytest

from lekit.control import (
    ControlHandle,
    ControllerControlState,
    HandleState,
    HubSnapshot,
    NodeDescriptor,
    NodeReport,
    NodeRole,
    RobotControlState,
    RuntimeState,
)
from lekit.control.handles import (
    HandleRecord,
    InvalidHandleTransition,
    StaleTransition,
    correlate_control,
    transition_handle,
)
from lekit.control.store import HubStore


def _descriptor(role: NodeRole) -> NodeDescriptor:
    is_controller = role is NodeRole.CONTROLLER
    return NodeDescriptor(
        protocol_version=1,
        schema_version=1,
        node_id="controller-1" if is_controller else "robot-1",
        session_id="controller-session-1" if is_controller else "robot-session-1",
        role=role,
        display_name="Controller" if is_controller else "Robot",
        administratively_enabled=True,
        capabilities=("teleop",),
        action_schemas=("lekit.action.v1",),
        control_modes=("teleop",),
        action_endpoint="tcp://controller:5557" if is_controller else None,
        observation_features={},
        action_features={},
        software_version="1.0.0",
        diagnostics={},
    )


@pytest.fixture
def descriptors() -> tuple[NodeDescriptor, NodeDescriptor]:
    return _descriptor(NodeRole.ROBOT), _descriptor(NodeRole.CONTROLLER)


@pytest.fixture
def handle_record(descriptors: tuple[NodeDescriptor, NodeDescriptor]) -> HandleRecord:
    robot, controller = descriptors
    return HandleRecord(
        handle=ControlHandle(
            handle_id="handle-1",
            hub_epoch="epoch-1",
            robot_id=robot.node_id,
            robot_session_id=robot.session_id,
            controller_id=controller.node_id,
            controller_session_id=controller.session_id,
            controller_action_endpoint=controller.action_endpoint or "",
            action_schema="lekit.action.v1",
            control_mode="teleop",
            fencing_token=1,
            issued_at_ns=100,
            expires_at_ns=1_000,
        ),
        state=HandleState.ASSIGNED,
        transition_sequence=0,
        correlation_id="assignment-1",
        updated_at_ns=100,
        reason=None,
    )


def _robot_report(
    record: HandleRecord,
    *,
    state: RobotControlState,
    runtime: RuntimeState = RuntimeState.ONLINE,
    engaged: bool = True,
) -> NodeReport:
    return NodeReport(
        node_id=record.handle.robot_id,
        session_id=record.handle.robot_session_id,
        runtime_state=runtime,
        robot_control_state=state,
        controller_control_state=None,
        handle_id=record.handle.handle_id,
        fencing_token=record.handle.fencing_token,
        action_rate_hz=60.0,
        frame_age_ms=5.0,
        last_sequence=8,
        tracking=True,
        engaged=engaged,
        processor_state="active",
        active_hold=state is RobotControlState.HOLD,
        error=None,
        reported_at_ns=200,
    )


def _controller_report(
    record: HandleRecord,
    *,
    state: ControllerControlState,
    runtime: RuntimeState = RuntimeState.ONLINE,
) -> NodeReport:
    return NodeReport(
        node_id=record.handle.controller_id,
        session_id=record.handle.controller_session_id,
        runtime_state=runtime,
        robot_control_state=None,
        controller_control_state=state,
        handle_id=record.handle.handle_id,
        fencing_token=record.handle.fencing_token,
        action_rate_hz=60.0,
        frame_age_ms=None,
        last_sequence=8,
        tracking=True,
        engaged=True,
        processor_state=None,
        active_hold=None,
        error=None,
        reported_at_ns=200,
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (HandleState.ASSIGNED, HandleState.TAKING_OVER),
        (HandleState.TAKING_OVER, HandleState.ACTIVE),
        (HandleState.ACTIVE, HandleState.HANDING_OVER),
        (HandleState.HANDING_OVER, HandleState.RELEASED),
        (HandleState.ACTIVE, HandleState.REVOKING),
        (HandleState.REVOKING, HandleState.REVOKED),
    ],
)
def test_valid_transitions(source, target, handle_record):
    current = replace(handle_record, state=source, transition_sequence=4)
    updated = transition_handle(
        current,
        target,
        transition_sequence=5,
        correlation_id="correlation-5",
        at_ns=500,
    )
    assert updated.state is target


def test_terminal_handle_cannot_reactivate(handle_record):
    current = replace(handle_record, state=HandleState.REVOKED, transition_sequence=9)
    with pytest.raises(InvalidHandleTransition):
        transition_handle(
            current, HandleState.ACTIVE, transition_sequence=10, correlation_id="late", at_ns=600
        )


def test_transition_is_idempotent_only_for_the_same_sequence_state_and_correlation(handle_record):
    repeated = transition_handle(
        handle_record,
        HandleState.ASSIGNED,
        transition_sequence=0,
        correlation_id="assignment-1",
        at_ns=999,
    )
    assert repeated is handle_record

    with pytest.raises(StaleTransition, match="regressed"):
        transition_handle(
            handle_record,
            HandleState.TAKING_OVER,
            transition_sequence=-1,
            correlation_id="old",
            at_ns=101,
        )
    with pytest.raises(StaleTransition, match="already used"):
        transition_handle(
            handle_record,
            HandleState.TAKING_OVER,
            transition_sequence=0,
            correlation_id="different",
            at_ns=101,
        )


@pytest.mark.parametrize(
    ("state", "robot_state", "controller_state", "controller_runtime", "expected_code"),
    [
        (
            HandleState.ACTIVE,
            RobotControlState.HOLD,
            ControllerControlState.IDLE,
            RuntimeState.ONLINE,
            "desired_active_robot_hold",
        ),
        (
            HandleState.ACTIVE,
            RobotControlState.HOLD,
            ControllerControlState.STREAMING,
            RuntimeState.ONLINE,
            "controller_streaming_robot_not_accepting",
        ),
        (
            HandleState.ACTIVE,
            RobotControlState.CONTROLLING,
            ControllerControlState.STREAMING,
            RuntimeState.STOPPED,
            "controller_offline_robot_controlling",
        ),
        (
            HandleState.ASSIGNED,
            RobotControlState.CONTROLLING,
            ControllerControlState.STREAMING,
            RuntimeState.ONLINE,
            "orphan_observed_control",
        ),
        (
            HandleState.REVOKED,
            RobotControlState.CONTROLLING,
            ControllerControlState.IDLE,
            RuntimeState.ONLINE,
            "terminal_handle_accepted",
        ),
        (
            HandleState.ACTIVE,
            RobotControlState.SAFETY,
            ControllerControlState.STREAMING,
            RuntimeState.ONLINE,
            "robot_safety_with_active_desire",
        ),
    ],
)
def test_correlation_classifies_severe_control_mismatches(
    handle_record,
    state,
    robot_state,
    controller_state,
    controller_runtime,
    expected_code,
):
    record = replace(handle_record, state=state)
    snapshot = correlate_control(
        record,
        _robot_report(record, state=robot_state),
        _controller_report(record, state=controller_state, runtime=controller_runtime),
        now_ns=250,
    )

    assert expected_code in snapshot.mismatch_codes
    assert snapshot.healthy is False
    assert snapshot.error in snapshot.mismatch_codes


def test_active_clutch_released_hold_is_not_a_control_mismatch(handle_record) -> None:
    record = replace(handle_record, state=HandleState.ACTIVE)

    snapshot = correlate_control(
        record,
        _robot_report(record, state=RobotControlState.HOLD, engaged=False),
        _controller_report(record, state=ControllerControlState.STREAMING),
        now_ns=250,
    )

    assert snapshot.mismatch_codes == ()
    assert snapshot.healthy is True


def test_store_assigns_monotonic_fencing_and_exclusive_handle(tmp_path, descriptors):
    store = HubStore(tmp_path / "hub.sqlite3")
    first = store.create_assignment(*descriptors, now_ns=10, ttl_ns=3_000_000_000)
    store.transition(first.handle_id, HandleState.REVOKING, 1, "revoke-requested", 20, "test")
    store.transition(first.handle_id, HandleState.REVOKED, 2, "revoke-completed", 21, "test")
    second = store.create_assignment(*descriptors, now_ns=30, ttl_ns=3_000_000_000)

    assert second.fencing_token == first.fencing_token + 1
    assert second.handle_id != first.handle_id
    with pytest.raises(ValueError, match="exclusive"):
        store.create_assignment(*descriptors, now_ns=40, ttl_ns=3_000_000_000)


def test_store_rejects_direct_revocation_that_bypasses_revoke_lifecycle(tmp_path, descriptors):
    store = HubStore(tmp_path / "hub.sqlite3")
    handle = store.create_assignment(*descriptors, now_ns=10, ttl_ns=100)

    with pytest.raises(InvalidHandleTransition, match="forbidden"):
        store.transition(handle.handle_id, HandleState.REVOKED, 1, "revoke", 20, "test")


def test_assignment_rolls_back_handle_and_fencing_when_handle_insert_fails(tmp_path, descriptors):
    path = tmp_path / "hub.sqlite3"
    store = HubStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_handle_insert BEFORE INSERT ON handles "
            "BEGIN SELECT RAISE(ABORT, 'forced handle insert failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced handle insert failure"):
        store.create_assignment(*descriptors, now_ns=10, ttl_ns=3_000_000_000)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM handles").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM robot_fencing").fetchone()[0] == 0


def test_store_renewal_keeps_identity_and_fencing(tmp_path, descriptors):
    store = HubStore(tmp_path / "hub.sqlite3")
    handle = store.create_assignment(*descriptors, now_ns=10, ttl_ns=100)

    renewed = store.renew_handle(handle.handle_id, expires_at_ns=500, at_ns=100)

    assert renewed.handle_id == handle.handle_id
    assert renewed.fencing_token == handle.fencing_token
    assert renewed.expires_at_ns == 500


def test_store_rejects_renewal_after_the_current_lease_expires(tmp_path, descriptors):
    store = HubStore(tmp_path / "hub.sqlite3")
    handle = store.create_assignment(*descriptors, now_ns=10, ttl_ns=100)

    with pytest.raises(ValueError, match="expired"):
        store.renew_handle(handle.handle_id, expires_at_ns=500, at_ns=110)


def test_store_rejects_renewal_expiry_at_or_before_the_renewal_time(tmp_path, descriptors):
    store = HubStore(tmp_path / "hub.sqlite3")
    handle = store.create_assignment(*descriptors, now_ns=10, ttl_ns=100)

    with pytest.raises(ValueError, match="after renewal"):
        store.renew_handle(handle.handle_id, expires_at_ns=50, at_ns=50)


def test_restart_invalidation_expires_other_epoch_handles(tmp_path, descriptors):
    store = HubStore(tmp_path / "hub.sqlite3")
    previous_epoch = store.begin_epoch(started_at_ns=1)
    handle = store.create_assignment(*descriptors, now_ns=10, ttl_ns=100)
    current_epoch = store.begin_epoch(started_at_ns=20)

    invalidated = store.invalidate_previous_epochs(current_epoch, at_ns=30)

    assert previous_epoch != current_epoch
    assert invalidated == (handle.handle_id,)
    assert store.get_handle(handle.handle_id).state is HandleState.EXPIRED


def test_store_persists_nodes_audit_and_last_snapshot_as_compact_sorted_json(tmp_path, descriptors):
    robot, controller = descriptors
    store = HubStore(tmp_path / "hub.sqlite3")
    store.upsert_node(robot, seen_at_ns=10)
    store.upsert_node(controller, seen_at_ns=11)
    store.append_audit(
        event="operator_action",
        at_ns=12,
        actor="operator-1",
        correlation_id="audit-1",
        details={"z": 1, "a": {"b": True}},
    )
    epoch = store.begin_epoch(started_at_ns=13)
    store.save_snapshot(
        HubSnapshot(
            version=1,
            hub_epoch=epoch,
            generated_at_ns=14,
            nodes=(),
            controls=(),
            alerts=({"z": 1, "a": True},),
        )
    )

    history = store.list_history()
    assert history[0]["event"] == "operator_action"
    assert history[0]["details"] == {"a": {"b": True}, "z": 1}

    with sqlite3.connect(tmp_path / "hub.sqlite3") as connection:
        audit_json = connection.execute("SELECT details_json FROM audit_events").fetchone()[0]
        snapshot_json = connection.execute("SELECT snapshot_json FROM snapshots").fetchone()[0]
    assert audit_json == '{"a":{"b":true},"z":1}'
    assert snapshot_json == (
        '{"alerts":[{"a":true,"z":1}],"controls":[],"generated_at_ns":14,'
        f'"hub_epoch":"{epoch}","nodes":[],"version":1}}'
    )
