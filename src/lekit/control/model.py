"""Versioned immutable domain models for the Control Hub."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

PROTOCOL_VERSION = 1


class HandleState(StrEnum):
    ASSIGNED = "assigned"
    TAKING_OVER = "taking_over"
    ACTIVE = "active"
    HANDING_OVER = "handing_over"
    RELEASED = "released"
    REVOKING = "revoking"
    REVOKED = "revoked"
    EXPIRED = "expired"
    FAULT = "fault"


TERMINAL_HANDLE_STATES = frozenset(
    {HandleState.RELEASED, HandleState.REVOKED, HandleState.EXPIRED, HandleState.FAULT}
)


class NodeRole(StrEnum):
    CONTROLLER = "controller"
    ROBOT = "robot"


class RuntimeState(StrEnum):
    STARTING = "starting"
    ONLINE = "online"
    DEGRADED = "degraded"
    FAULT = "fault"
    STOPPED = "stopped"


class RobotControlState(StrEnum):
    HOLD = "hold"
    CONTROLLING = "controlling"
    SAFETY = "safety"


class ControllerControlState(StrEnum):
    IDLE = "idle"
    TAKING_OVER = "taking_over"
    STREAMING = "streaming"
    HANDING_OVER = "handing_over"
    FAULT = "fault"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _require_finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite positive")
    return result


def _require_text_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of non-empty strings")
    for item in value:
        _require_text(name, item)
    return value


def _normalize_json_value(value: object, *, feature_metadata: bool = False) -> Any:
    if feature_metadata and value is float:
        return "float"
    if feature_metadata and value is int:
        return "int"
    if feature_metadata and value is bool:
        return "bool"
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("feature metadata must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError("feature metadata must use string keys")
            if feature_metadata and key == "shape":
                if not isinstance(nested_value, (tuple, list)) or any(
                    isinstance(item, bool) or not isinstance(item, int) for item in nested_value
                ):
                    raise ValueError("feature metadata shape must be an integer array")
                result[key] = list(nested_value)
            else:
                result[key] = _normalize_json_value(nested_value, feature_metadata=feature_metadata)
        return result
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(item, feature_metadata=feature_metadata) for item in value]
    raise ValueError("feature metadata must be JSON-compatible")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(nested_value) for key, nested_value in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any], *, feature_metadata: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("feature metadata must be a mapping")
    return _deep_freeze(_normalize_json_value(value, feature_metadata=feature_metadata))


def normalize_feature_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert LeRobot feature descriptions into stable JSON-compatible metadata."""
    if not isinstance(metadata, Mapping):
        raise ValueError("feature metadata must be a mapping")
    return _normalize_json_value(metadata, feature_metadata=True)


@dataclass(frozen=True, slots=True)
class NodeDescriptor:
    protocol_version: int
    schema_version: int
    node_id: str
    session_id: str
    role: NodeRole
    display_name: str
    administratively_enabled: bool
    capabilities: tuple[str, ...]
    action_schemas: tuple[str, ...]
    control_modes: tuple[str, ...]
    action_endpoint: str | None
    observation_features: Mapping[str, Any]
    action_features: Mapping[str, Any]
    software_version: str
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_positive_int("protocol_version", self.protocol_version)
        _require_positive_int("schema_version", self.schema_version)
        for name in ("node_id", "session_id", "display_name", "software_version"):
            _require_text(name, getattr(self, name))
        if not isinstance(self.role, NodeRole):
            raise ValueError("role must be a NodeRole")
        if not isinstance(self.administratively_enabled, bool):
            raise ValueError("administratively_enabled must be a bool")
        for name in ("capabilities", "action_schemas", "control_modes"):
            _require_text_tuple(name, getattr(self, name))
        if self.action_endpoint is not None:
            _require_text("action_endpoint", self.action_endpoint)
        if self.role is NodeRole.CONTROLLER:
            if self.action_endpoint is None:
                raise ValueError("controller action_endpoint must not be empty")
            if not self.action_schemas:
                raise ValueError("controller action_schemas must not be empty")
        object.__setattr__(
            self, "observation_features", _freeze_mapping(self.observation_features, feature_metadata=True)
        )
        object.__setattr__(
            self, "action_features", _freeze_mapping(self.action_features, feature_metadata=True)
        )
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ControlHandle:
    handle_id: str
    hub_epoch: str
    robot_id: str
    robot_session_id: str
    controller_id: str
    controller_session_id: str
    controller_action_endpoint: str
    action_schema: str
    control_mode: str
    fencing_token: int
    issued_at_ns: int
    expires_at_ns: int

    def __post_init__(self) -> None:
        for name in (
            "handle_id",
            "hub_epoch",
            "robot_id",
            "robot_session_id",
            "controller_id",
            "controller_session_id",
            "controller_action_endpoint",
            "action_schema",
            "control_mode",
        ):
            _require_text(name, getattr(self, name))
        if (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token < 1
        ):
            raise ValueError("Handle timing and fencing values are invalid")
        if (
            isinstance(self.issued_at_ns, bool)
            or not isinstance(self.issued_at_ns, int)
            or self.issued_at_ns < 0
            or isinstance(self.expires_at_ns, bool)
            or not isinstance(self.expires_at_ns, int)
            or self.expires_at_ns <= self.issued_at_ns
        ):
            raise ValueError("Handle timing and fencing values are invalid")


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    handle_id: str
    hub_epoch: str
    fencing_token: int
    controller_id: str
    controller_session_id: str
    stream_session_id: str
    sequence: int
    captured_monotonic_ns: int
    captured_utc_ns: int
    payload_schema: str
    payload: bytes

    def __post_init__(self) -> None:
        for name in (
            "handle_id",
            "hub_epoch",
            "controller_id",
            "controller_session_id",
            "stream_session_id",
            "payload_schema",
        ):
            _require_text(name, getattr(self, name))
        _require_positive_int("fencing_token", self.fencing_token)
        for name in ("sequence", "captured_monotonic_ns", "captured_utc_ns"):
            _require_nonnegative_int(name, getattr(self, name))
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("payload must be non-empty bytes")


@dataclass(frozen=True, slots=True)
class NodeReport:
    node_id: str
    session_id: str
    runtime_state: RuntimeState
    robot_control_state: RobotControlState | None
    controller_control_state: ControllerControlState | None
    handle_id: str | None
    fencing_token: int | None
    action_rate_hz: float
    frame_age_ms: float | None
    last_sequence: int | None
    tracking: bool | None
    engaged: bool | None
    processor_state: str | None
    active_hold: bool | None
    error: str | None
    reported_at_ns: int

    def __post_init__(self) -> None:
        _require_text("node_id", self.node_id)
        _require_text("session_id", self.session_id)
        if not isinstance(self.runtime_state, RuntimeState):
            raise ValueError("runtime_state must be a RuntimeState")
        if self.robot_control_state is not None and not isinstance(
            self.robot_control_state, RobotControlState
        ):
            raise ValueError("robot_control_state must be a RobotControlState")
        if self.controller_control_state is not None and not isinstance(
            self.controller_control_state, ControllerControlState
        ):
            raise ValueError("controller_control_state must be a ControllerControlState")
        if self.robot_control_state is not None and self.controller_control_state is not None:
            raise ValueError("a node report may describe only one control role")
        if self.handle_id is not None:
            _require_text("handle_id", self.handle_id)
        if self.fencing_token is not None:
            _require_positive_int("fencing_token", self.fencing_token)
        _require_finite_nonnegative("action_rate_hz", self.action_rate_hz)
        if self.frame_age_ms is not None:
            _require_finite_nonnegative("frame_age_ms", self.frame_age_ms)
        if self.last_sequence is not None:
            _require_nonnegative_int("last_sequence", self.last_sequence)
        for name in ("tracking", "engaged", "active_hold"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool or None")
        for name in ("processor_state", "error"):
            value = getattr(self, name)
            if value is not None:
                _require_text(name, value)
        _require_nonnegative_int("reported_at_ns", self.reported_at_ns)


@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    handle_id: str
    robot_id: str
    controller_id: str
    desired_state: HandleState
    handle_state: HandleState
    robot_control_state: RobotControlState | None
    controller_control_state: ControllerControlState | None
    action_rate_hz: float
    frame_age_ms: float | None
    last_sequence: int | None
    issued_at_ns: int
    expires_at_ns: int
    last_updated_ns: int
    healthy: bool
    error: str | None
    mismatch_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("handle_id", "robot_id", "controller_id"):
            _require_text(name, getattr(self, name))
        for name in ("desired_state", "handle_state"):
            if not isinstance(getattr(self, name), HandleState):
                raise ValueError(f"{name} must be a HandleState")
        if self.robot_control_state is not None and not isinstance(
            self.robot_control_state, RobotControlState
        ):
            raise ValueError("robot_control_state must be a RobotControlState")
        if self.controller_control_state is not None and not isinstance(
            self.controller_control_state, ControllerControlState
        ):
            raise ValueError("controller_control_state must be a ControllerControlState")
        _require_finite_nonnegative("action_rate_hz", self.action_rate_hz)
        if self.frame_age_ms is not None:
            _require_finite_nonnegative("frame_age_ms", self.frame_age_ms)
        if self.last_sequence is not None:
            _require_nonnegative_int("last_sequence", self.last_sequence)
        _require_nonnegative_int("issued_at_ns", self.issued_at_ns)
        _require_nonnegative_int("expires_at_ns", self.expires_at_ns)
        if self.expires_at_ns <= self.issued_at_ns:
            raise ValueError("expires_at_ns must be after issued_at_ns")
        _require_nonnegative_int("last_updated_ns", self.last_updated_ns)
        if not isinstance(self.healthy, bool):
            raise ValueError("healthy must be a bool")
        if self.error is not None:
            _require_text("error", self.error)
        _require_text_tuple("mismatch_codes", self.mismatch_codes)


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    descriptor: NodeDescriptor
    online: bool
    last_seen_ns: int
    report: NodeReport | None

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, NodeDescriptor):
            raise ValueError("descriptor must be a NodeDescriptor")
        if not isinstance(self.online, bool):
            raise ValueError("online must be a bool")
        _require_nonnegative_int("last_seen_ns", self.last_seen_ns)
        if self.report is not None:
            if not isinstance(self.report, NodeReport):
                raise ValueError("report must be a NodeReport")
            if (self.report.node_id, self.report.session_id) != (
                self.descriptor.node_id,
                self.descriptor.session_id,
            ):
                raise ValueError("report identity must match descriptor")


@dataclass(frozen=True, slots=True)
class HubSnapshot:
    version: int
    hub_epoch: str
    generated_at_ns: int
    nodes: tuple[NodeSnapshot, ...]
    controls: tuple[ControlSnapshot, ...]
    alerts: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        _require_nonnegative_int("version", self.version)
        _require_text("hub_epoch", self.hub_epoch)
        _require_nonnegative_int("generated_at_ns", self.generated_at_ns)
        if not isinstance(self.nodes, tuple) or not all(
            isinstance(node, NodeSnapshot) for node in self.nodes
        ):
            raise ValueError("nodes must be a tuple of NodeSnapshot")
        if not isinstance(self.controls, tuple) or not all(
            isinstance(control, ControlSnapshot) for control in self.controls
        ):
            raise ValueError("controls must be a tuple of ControlSnapshot")
        if not isinstance(self.alerts, tuple):
            raise ValueError("alerts must be a tuple of mappings")
        object.__setattr__(
            self,
            "alerts",
            tuple(_freeze_mapping(alert) for alert in self.alerts),
        )


@dataclass(frozen=True, slots=True)
class ManagementMessage:
    protocol_version: int
    kind: str
    correlation_id: str
    sender_id: str
    sender_session_id: str
    sequence: int
    sent_at_ns: int
    body: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_positive_int("protocol_version", self.protocol_version)
        for name in ("kind", "correlation_id", "sender_id", "sender_session_id"):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("sequence", self.sequence)
        _require_nonnegative_int("sent_at_ns", self.sent_at_ns)
        object.__setattr__(self, "body", _freeze_mapping(self.body))


@dataclass(frozen=True, slots=True)
class TimingConfig:
    action_rate_hz: float = 60.0
    action_stale_s: float = 0.1
    status_rate_hz: float = 10.0
    heartbeat_rate_hz: float = 2.0
    discovery_rate_hz: float = 1.0
    renewal_interval_s: float = 1.0
    handle_ttl_s: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "action_rate_hz",
            "action_stale_s",
            "status_rate_hz",
            "heartbeat_rate_hz",
            "discovery_rate_hz",
            "renewal_interval_s",
            "handle_ttl_s",
        ):
            _require_finite_positive(name, getattr(self, name))

    @property
    def action_stale_ns(self) -> int:
        return round(self.action_stale_s * 1_000_000_000)

    @property
    def heartbeat_interval_ns(self) -> int:
        return round(1_000_000_000 / self.heartbeat_rate_hz)

    @property
    def renewal_interval_ns(self) -> int:
        return round(self.renewal_interval_s * 1_000_000_000)

    @property
    def handle_ttl_ns(self) -> int:
        return round(self.handle_ttl_s * 1_000_000_000)


def load_or_create_node_id(path: Path | str) -> str:
    """Load a stable node ID, creating one when the path does not exist."""
    node_id_path = Path(path)
    if node_id_path.exists():
        return _require_text("node_id", node_id_path.read_text(encoding="utf-8").strip())
    node_id_path.parent.mkdir(parents=True, exist_ok=True)
    node_id = str(uuid.uuid4())
    node_id_path.write_text(node_id + "\n", encoding="utf-8")
    return node_id
