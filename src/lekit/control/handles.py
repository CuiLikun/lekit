"""Pure Control Handle lifecycle reduction and observed-state correlation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .model import (
    TERMINAL_HANDLE_STATES,
    ControlHandle,
    ControllerControlState,
    ControlSnapshot,
    HandleState,
    NodeReport,
    RobotControlState,
    RuntimeState,
)


class InvalidHandleTransition(ValueError):  # noqa: N818 - public interface fixed by the protocol.
    """Raised when a Handle lifecycle transition is not allowed."""


class StaleTransition(ValueError):  # noqa: N818 - public interface fixed by the protocol.
    """Raised when a transition sequence is stale or conflicts with history."""


@dataclass(frozen=True, slots=True)
class HandleRecord:
    """A persisted Handle together with its current desired lifecycle state."""

    handle: ControlHandle
    state: HandleState
    transition_sequence: int
    correlation_id: str
    updated_at_ns: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ControlHandle):
            raise ValueError("handle must be a ControlHandle")
        if not isinstance(self.state, HandleState):
            raise ValueError("state must be a HandleState")
        if isinstance(self.transition_sequence, bool) or self.transition_sequence < 0:
            raise ValueError("transition_sequence must be a non-negative integer")
        if not isinstance(self.correlation_id, str) or not self.correlation_id:
            raise ValueError("correlation_id must not be empty")
        if isinstance(self.updated_at_ns, bool) or self.updated_at_ns < 0:
            raise ValueError("updated_at_ns must be a non-negative integer")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("reason must be non-empty when present")


@dataclass(frozen=True, slots=True)
class HandleTransition:
    """An immutable, auditable Handle state transition."""

    handle_id: str
    fencing_token: int
    state: HandleState
    transition_sequence: int
    correlation_id: str
    at_ns: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.handle_id, str) or not self.handle_id:
            raise ValueError("handle_id must not be empty")
        if isinstance(self.fencing_token, bool) or self.fencing_token < 1:
            raise ValueError("fencing_token must be a positive integer")
        if not isinstance(self.state, HandleState):
            raise ValueError("state must be a HandleState")
        if isinstance(self.transition_sequence, bool) or self.transition_sequence < 0:
            raise ValueError("transition_sequence must be a non-negative integer")
        if not isinstance(self.correlation_id, str) or not self.correlation_id:
            raise ValueError("correlation_id must not be empty")
        if isinstance(self.at_ns, bool) or self.at_ns < 0:
            raise ValueError("at_ns must be a non-negative integer")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("reason must be non-empty when present")


_ALLOWED = {
    HandleState.ASSIGNED: {
        HandleState.TAKING_OVER,
        HandleState.REVOKING,
        HandleState.EXPIRED,
        HandleState.FAULT,
    },
    HandleState.TAKING_OVER: {
        HandleState.ACTIVE,
        HandleState.REVOKING,
        HandleState.EXPIRED,
        HandleState.FAULT,
    },
    HandleState.ACTIVE: {
        HandleState.HANDING_OVER,
        HandleState.REVOKING,
        HandleState.EXPIRED,
        HandleState.FAULT,
    },
    HandleState.HANDING_OVER: {
        HandleState.RELEASED,
        HandleState.REVOKING,
        HandleState.EXPIRED,
        HandleState.FAULT,
    },
    HandleState.REVOKING: {HandleState.REVOKED, HandleState.EXPIRED, HandleState.FAULT},
}


def transition_handle(
    record: HandleRecord,
    target: HandleState,
    *,
    transition_sequence: int,
    correlation_id: str,
    at_ns: int,
    reason: str | None = None,
) -> HandleRecord:
    """Apply one ordered, idempotent Handle lifecycle transition."""
    if transition_sequence < record.transition_sequence:
        raise StaleTransition("transition sequence regressed")
    if transition_sequence == record.transition_sequence:
        if target is record.state and correlation_id == record.correlation_id:
            return record
        raise StaleTransition("sequence already used by another transition")
    if target not in _ALLOWED.get(record.state, set()):
        raise InvalidHandleTransition(f"{record.state} -> {target} is forbidden")
    return replace(
        record,
        state=target,
        transition_sequence=transition_sequence,
        correlation_id=correlation_id,
        updated_at_ns=at_ns,
        reason=reason,
    )


def correlate_control(
    record: HandleRecord,
    robot_report: NodeReport | None,
    controller_report: NodeReport | None,
    *,
    now_ns: int,
) -> ControlSnapshot:
    """Produce a stable read model from desired Handle state and endpoint reports."""
    codes: list[str] = []
    robot_state = robot_report.robot_control_state if robot_report is not None else None
    controller_state = controller_report.controller_control_state if controller_report is not None else None
    desired_active = record.state is HandleState.ACTIVE
    robot_controlling = robot_state is RobotControlState.CONTROLLING
    controller_streaming = controller_state is ControllerControlState.STREAMING

    if desired_active and robot_state is RobotControlState.HOLD:
        codes.append("desired_active_robot_hold")
    if controller_streaming and not robot_controlling:
        codes.append("controller_streaming_robot_not_accepting")
    if (
        robot_controlling
        and controller_report is not None
        and controller_report.runtime_state is not RuntimeState.ONLINE
    ):
        codes.append("controller_offline_robot_controlling")
    if not desired_active and (robot_controlling or controller_streaming):
        codes.append("orphan_observed_control")
    if record.state in TERMINAL_HANDLE_STATES and robot_controlling:
        codes.append("terminal_handle_accepted")
    if desired_active and robot_state is RobotControlState.SAFETY:
        codes.append("robot_safety_with_active_desire")
    if robot_report is not None and (
        robot_report.handle_id != record.handle.handle_id
        or robot_report.fencing_token != record.handle.fencing_token
    ):
        codes.append("robot_handle_mismatch")
    if controller_report is not None and (
        controller_report.handle_id != record.handle.handle_id
        or controller_report.fencing_token != record.handle.fencing_token
    ):
        codes.append("controller_handle_mismatch")

    unique_codes = tuple(dict.fromkeys(codes))
    return ControlSnapshot(
        handle_id=record.handle.handle_id,
        robot_id=record.handle.robot_id,
        controller_id=record.handle.controller_id,
        desired_state=record.state,
        handle_state=record.state,
        robot_control_state=robot_state,
        controller_control_state=controller_state,
        action_rate_hz=robot_report.action_rate_hz if robot_report is not None else 0.0,
        frame_age_ms=robot_report.frame_age_ms if robot_report is not None else None,
        last_sequence=robot_report.last_sequence if robot_report is not None else None,
        issued_at_ns=record.handle.issued_at_ns,
        expires_at_ns=record.handle.expires_at_ns,
        last_updated_ns=now_ns,
        healthy=not unique_codes,
        error=unique_codes[0] if unique_codes else None,
        mismatch_codes=unique_codes,
    )
