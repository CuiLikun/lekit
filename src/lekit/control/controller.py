"""Generic Controller node lifecycle and Handle-wrapped action publication."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

from .hub import ControlError
from .model import (
    PROTOCOL_VERSION,
    ActionEnvelope,
    ControlHandle,
    ControllerControlState,
    ManagementMessage,
    NodeDescriptor,
    NodeReport,
    NodeRole,
    RuntimeState,
    TimingConfig,
    load_or_create_node_id,
)
from .runtime import ActionPublisher, NodeChannel, Runtime


class HandleNotGranted(ControlError):  # noqa: N818 - public protocol name.
    """Raised when a Controller tries to use authority it has not received."""


class HandleExpired(ControlError):  # noqa: N818 - public protocol name.
    """Raised when a Controller tries to reuse terminal local authority."""


@dataclass(kw_only=True)
class ControllerNodeConfig:
    """Configuration for one independently running Controller session."""

    node_id_path: Path
    display_name: str
    action_schemas: tuple[str, ...]
    action_endpoint: str = "tcp://0.0.0.0:5557"
    control_modes: tuple[str, ...] = ("teleop",)
    hub_seed: str | None = None
    timing: TimingConfig = field(default_factory=TimingConfig)

    def __post_init__(self) -> None:
        self.node_id_path = Path(self.node_id_path)
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must not be empty")
        for name in ("action_schemas", "control_modes"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
                raise ValueError(f"{name} must be a tuple of non-empty strings")
        if not self.action_schemas:
            raise ValueError("action_schemas must not be empty")
        if not isinstance(self.action_endpoint, str) or not self.action_endpoint:
            raise ValueError("action_endpoint must not be empty")
        if self.hub_seed is not None and (not isinstance(self.hub_seed, str) or not self.hub_seed):
            raise ValueError("hub_seed must be non-empty when present")
        if not isinstance(self.timing, TimingConfig):
            raise ValueError("timing must be a TimingConfig")


class ControllerNode:
    """Own one Controller process session and publish only current Handle frames."""

    def __init__(
        self,
        config: ControllerNodeConfig,
        *,
        runtime: Runtime,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not isinstance(config, ControllerNodeConfig):
            raise TypeError("config must be a ControllerNodeConfig")
        self.config = config
        self._runtime = runtime
        self._monotonic_ns = monotonic_ns
        self._utc_ns = utc_ns
        self._lock = threading.RLock()
        self._node_id = load_or_create_node_id(config.node_id_path)
        self._session_id = "not-started"
        self._hub_epoch: str | None = None
        self._channel: NodeChannel | None = None
        self._action_publisher: ActionPublisher | None = None
        self._started = False
        self._management_stop = threading.Event()
        self._management_thread: threading.Thread | None = None
        self._outgoing_sequence = 0
        self._last_hub_command_sequence = -1
        self._control_state = ControllerControlState.IDLE
        self._handle: ControlHandle | None = None
        self._terminal_handle_ids: set[str] = set()
        self._local_expiry_monotonic_ns = 0
        self._streaming_enabled = False
        self._stream_session_id: str | None = None
        self._sequence = 0
        self._last_error: str | None = None
        self._next_heartbeat_ns = 0
        self._next_status_ns = 0
        self._pending_heartbeat_correlation: str | None = None
        self._missed_heartbeat_acks = 0
        self._take_over_correlation: str | None = None
        self._registration_correlation: str | None = None

    @property
    def node_id(self) -> str:
        """Return the Controller's stable identity."""
        return self._node_id

    @property
    def session_id(self) -> str:
        """Return the current process session identity."""
        with self._lock:
            return self._session_id

    @property
    def control_state(self) -> ControllerControlState:
        """Return the locally observed Controller control state."""
        with self._lock:
            return self._control_state

    @property
    def stream_session_id(self) -> str | None:
        """Return the current direct-action stream identity, if streaming."""
        with self._lock:
            return self._stream_session_id

    def start(self) -> None:
        """Create a fresh process session, register, and launch management work."""
        with self._lock:
            if self._started:
                return
            self._started = True
            self._management_stop.clear()
            self._open_session_locked()
            if not self._send_registration_locked():
                self._management_lost_locked("management_send_failed")
            self._management_thread = threading.Thread(
                target=self._run_management,
                name=f"controller-management-{self._node_id}",
                daemon=True,
            )
            self._management_thread.start()

    def stop(self) -> None:
        """Synchronously disable authority and close this process session."""
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._management_stop.set()
            self._disable_streaming_locked("stopped", terminal=True)
            self._control_state = ControllerControlState.IDLE
            channel = self._channel
            publisher = self._action_publisher
            thread = self._management_thread
            self._channel = None
            self._action_publisher = None
            self._management_thread = None
        if channel is not None:
            channel.close()
        if publisher is not None:
            publisher.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def take_over(self, handle: ControlHandle) -> None:
        """Request activation of a previously granted current Handle."""
        with self._lock:
            self._validate_grant_target_locked(handle)
            if handle.handle_id in self._terminal_handle_ids:
                raise HandleExpired(handle.handle_id)
            if self._handle is None or self._handle != handle:
                raise HandleNotGranted(handle.handle_id)
            if self._control_state is ControllerControlState.STREAMING:
                return
            self._streaming_enabled = False
            self._control_state = ControllerControlState.TAKING_OVER
            self._send_take_over_requested_locked(self._handle)

    def hand_over(self, handle: ControlHandle) -> None:
        """Stop direct action publication before asking the Hub to release authority."""
        with self._lock:
            if handle.handle_id in self._terminal_handle_ids:
                return
            self._validate_grant_target_locked(handle)
            if self._handle is None or not self._same_authority(self._handle, handle):
                raise HandleNotGranted(handle.handle_id)
            self._disable_streaming_locked("hand_over")
            self._control_state = ControllerControlState.HANDING_OVER
            if self._send_handle_event_locked("hand_over_requested", handle):
                self._complete_hand_over_locked(handle)

    def receive_grant(self, handle: ControlHandle) -> bool:
        """Cache a current Hub grant without enabling the direct action stream."""
        with self._lock:
            return self._started and self._accept_grant_locked(handle)

    def receive_robot_ready(self, handle: ControlHandle) -> bool:
        """Enable one new direct-action stream after the Robot has armed it."""
        with self._lock:
            if not self._started or not self._matches_current_handle_locked(handle):
                return False
            if self._control_state is not ControllerControlState.TAKING_OVER:
                return False
            if self._monotonic_ns() >= self._local_expiry_monotonic_ns:
                self._disable_streaming_locked("handle_expired", terminal=True)
                self._control_state = ControllerControlState.IDLE
                return False
            self._stream_session_id = str(uuid.uuid4())
            self._sequence = 0
            self._streaming_enabled = True
            self._control_state = ControllerControlState.STREAMING
            self._send_handle_event_locked("controller_streaming", handle)
            return self._streaming_enabled

    def receive_management(self, message: ManagementMessage) -> bool:
        """Validate and apply one Hub management message without trusting stale authority."""
        with self._lock:
            if not self._started or not self._is_hub_message(message):
                return False
            if message.kind == "registered":
                if message.sequence <= self._last_hub_command_sequence:
                    return False
                epoch = message.body.get("hub_epoch")
                if not isinstance(epoch, str) or not epoch or message.sender_session_id != epoch:
                    return False
                if message.correlation_id != self._registration_correlation:
                    return False
                if self._hub_epoch is not None and self._hub_epoch != epoch:
                    self._disable_streaming_locked("hub_epoch_changed", terminal=True)
                    self._handle = None
                self._hub_epoch = epoch
                self._last_hub_command_sequence = message.sequence
                self._registration_correlation = None
                return True
            if message.kind == "heartbeat_ack":
                if message.sequence <= self._last_hub_command_sequence:
                    return False
                accepted = self._receive_heartbeat_ack_locked(message)
                if accepted:
                    self._last_hub_command_sequence = message.sequence
                return accepted
            if message.kind not in {"grant", "renewal", "take_over", "robot_ready", "hand_over", "revoke"}:
                return False
            if message.sequence <= self._last_hub_command_sequence:
                return False
            handle = self._handle_from_message_locked(message)
            if handle is None:
                return False
            if message.kind == "grant":
                accepted = self._accept_grant_locked(handle)
            elif message.kind == "renewal":
                accepted = self._renew_locked(handle)
            elif message.kind == "take_over":
                accepted = self._accept_take_over_locked(handle)
            elif message.kind == "robot_ready":
                accepted = message.correlation_id == self._take_over_correlation and self.receive_robot_ready(
                    handle
                )
                if accepted:
                    self._take_over_correlation = None
            elif message.kind == "hand_over":
                accepted = self._accept_hand_over_locked(handle)
            else:
                accepted = self._accept_revoke_locked(handle)
            if accepted and self._send_handle_ack_locked(message.kind, handle, message.correlation_id):
                self._last_hub_command_sequence = message.sequence
                return True
            return False

    def run_management_once(self, *, timeout_s: float = 0.0) -> bool:
        """Run one bounded management receive, heartbeat, status, and expiry iteration."""
        with self._lock:
            if not self._started or self._channel is None:
                return False
            channel = self._channel
        received = channel.receive(timeout_s=max(0.0, timeout_s))
        processed = False
        if received is not None and received.peer_id == "hub":
            processed = self.receive_management(received.message)
        with self._lock:
            if not self._started:
                return processed
            now = self._monotonic_ns()
            if self._streaming_enabled and now >= self._local_expiry_monotonic_ns:
                self._disable_streaming_locked("handle_expired", terminal=True)
                self._control_state = ControllerControlState.IDLE
            if now >= self._next_heartbeat_ns:
                if self._pending_heartbeat_correlation is not None:
                    self._missed_heartbeat_acks += 1
                    if self._missed_heartbeat_acks >= 3:
                        self._management_lost_locked("heartbeat_ack_timeout")
                        return processed
                correlation_id = self._new_correlation("heartbeat")
                if self._send_locked("heartbeat", correlation_id, {}):
                    self._pending_heartbeat_correlation = correlation_id
                    self._next_heartbeat_ns = now + self.config.timing.heartbeat_interval_ns
                return processed
            if now >= self._next_status_ns:
                report = self._report_locked(now)
                self._send_locked("status", self._new_correlation("status"), {"report": self._wire(report)})
                self._next_status_ns = now + round(1_000_000_000 / self.config.timing.status_rate_hz)
        return processed

    def publish(self, payload: bytes, *, captured_monotonic_ns: int, captured_utc_ns: int) -> bool:
        """Publish one non-blocking Handle-wrapped frame, if local authority is current."""
        with self._lock:
            if not self._streaming_enabled or self._handle is None or self._stream_session_id is None:
                return False
            if self._monotonic_ns() >= self._local_expiry_monotonic_ns:
                self._disable_streaming_locked("handle_expired", terminal=True)
                self._control_state = ControllerControlState.IDLE
                return False
            publisher = self._action_publisher
            if publisher is None:
                return False
            envelope = ActionEnvelope(
                handle_id=self._handle.handle_id,
                hub_epoch=self._handle.hub_epoch,
                fencing_token=self._handle.fencing_token,
                controller_id=self._node_id,
                controller_session_id=self._session_id,
                stream_session_id=self._stream_session_id,
                sequence=self._sequence,
                captured_monotonic_ns=captured_monotonic_ns,
                captured_utc_ns=captured_utc_ns,
                payload_schema=self._handle.action_schema,
                payload=payload,
            )
            self._sequence += 1
            return publisher.send(envelope)

    def _run_management(self) -> None:
        while not self._management_stop.is_set():
            self.run_management_once(timeout_s=0.05)
            self._management_stop.wait(0.01)

    def _open_session_locked(self) -> None:
        self._session_id = str(uuid.uuid4())
        self._hub_epoch = None
        self._outgoing_sequence = 0
        self._last_hub_command_sequence = -1
        self._take_over_correlation = None
        self._registration_correlation = None
        self._channel = self._runtime.open_node(
            self._node_id, self._session_id, hub_seed=self.config.hub_seed
        )
        if self._action_publisher is None:
            self._action_publisher = self._runtime.open_action_publisher(self.config.action_endpoint)
        now = self._monotonic_ns()
        self._next_heartbeat_ns = now + self.config.timing.heartbeat_interval_ns
        self._next_status_ns = now + round(1_000_000_000 / self.config.timing.status_rate_hz)
        self._pending_heartbeat_correlation = None
        self._missed_heartbeat_acks = 0

    def _send_registration_locked(self) -> bool:
        correlation_id = self._new_correlation("register")
        self._registration_correlation = correlation_id
        return self._send_locked(
            "register",
            correlation_id,
            {"descriptor": self._wire(self._descriptor_locked())},
            reconnect_on_failure=False,
        )

    def _descriptor_locked(self) -> NodeDescriptor:
        return NodeDescriptor(
            protocol_version=PROTOCOL_VERSION,
            schema_version=1,
            node_id=self._node_id,
            session_id=self._session_id,
            role=NodeRole.CONTROLLER,
            display_name=self.config.display_name,
            administratively_enabled=True,
            capabilities=("controller",),
            action_schemas=self.config.action_schemas,
            control_modes=self.config.control_modes,
            action_endpoint=self.config.action_endpoint,
            observation_features={},
            action_features={},
            software_version="lekit",
            diagnostics={},
        )

    def _accept_grant_locked(self, handle: ControlHandle) -> bool:
        if not self._accept_handle_target_locked(handle):
            return False
        if handle.handle_id in self._terminal_handle_ids:
            return False
        if self._handle is not None and not self._same_authority(self._handle, handle):
            return False
        if self._hub_epoch is not None and self._hub_epoch != handle.hub_epoch:
            return False
        if self._handle is not None:
            return True
        self._hub_epoch = handle.hub_epoch
        self._handle = handle
        self._set_local_expiry_locked(handle)
        self._streaming_enabled = False
        if self._control_state is not ControllerControlState.HANDING_OVER:
            self._control_state = ControllerControlState.IDLE
        return True

    def _renew_locked(self, handle: ControlHandle) -> bool:
        if (
            self._handle is None
            or not self._same_authority(self._handle, handle)
            or self._hub_epoch != handle.hub_epoch
            or self._session_id != handle.controller_session_id
            or handle.expires_at_ns <= self._handle.expires_at_ns
        ):
            return False
        self._handle = handle
        self._set_local_expiry_locked(handle)
        return True

    def _accept_take_over_locked(self, handle: ControlHandle) -> bool:
        if not self._matches_current_handle_locked(handle):
            return False
        self._streaming_enabled = False
        self._control_state = ControllerControlState.TAKING_OVER
        return self._send_take_over_requested_locked(handle)

    def _accept_hand_over_locked(self, handle: ControlHandle) -> bool:
        if not self._matches_current_handle_locked(handle):
            return False
        self._disable_streaming_locked("hand_over")
        self._control_state = ControllerControlState.HANDING_OVER
        return self._send_handle_event_locked(
            "hand_over_requested", handle
        ) and self._complete_hand_over_locked(handle)

    def _complete_hand_over_locked(self, handle: ControlHandle) -> bool:
        self._disable_streaming_locked("hand_over", terminal=True)
        self._control_state = ControllerControlState.IDLE
        return self._send_handle_event_locked("controller_released", handle)

    def _accept_revoke_locked(self, handle: ControlHandle) -> bool:
        if not self._matches_current_handle_locked(handle):
            return False
        self._disable_streaming_locked("revoked", terminal=True)
        self._control_state = ControllerControlState.IDLE
        return True

    def _matches_current_handle_locked(self, handle: ControlHandle) -> bool:
        return (
            handle.handle_id not in self._terminal_handle_ids
            and self._handle is not None
            and self._handle == handle
            and self._hub_epoch == handle.hub_epoch
            and self._session_id == handle.controller_session_id
        )

    @staticmethod
    def _same_authority(left: ControlHandle, right: ControlHandle) -> bool:
        names = (
            "handle_id",
            "hub_epoch",
            "robot_id",
            "robot_session_id",
            "controller_id",
            "controller_session_id",
            "controller_action_endpoint",
            "action_schema",
            "control_mode",
            "fencing_token",
            "issued_at_ns",
        )
        return all(getattr(left, name) == getattr(right, name) for name in names)

    def _validate_grant_target_locked(self, handle: ControlHandle) -> None:
        if not self._accept_handle_target_locked(handle):
            raise HandleNotGranted(handle.handle_id)

    def _accept_handle_target_locked(self, handle: ControlHandle) -> bool:
        return (
            isinstance(handle, ControlHandle)
            and handle.controller_id == self._node_id
            and handle.controller_session_id == self._session_id
            and handle.action_schema in self.config.action_schemas
            and handle.control_mode in self.config.control_modes
        )

    def _set_local_expiry_locked(self, handle: ControlHandle) -> None:
        remaining_ns = max(0, handle.expires_at_ns - self._utc_ns())
        self._local_expiry_monotonic_ns = self._monotonic_ns() + min(
            self.config.timing.handle_ttl_ns,
            remaining_ns,
        )

    def _disable_streaming_locked(self, reason: str, *, terminal: bool = False) -> None:
        self._streaming_enabled = False
        self._stream_session_id = None
        self._sequence = 0
        self._last_error = reason
        if terminal and self._handle is not None:
            self._terminal_handle_ids.add(self._handle.handle_id)
            self._handle = None
            self._local_expiry_monotonic_ns = 0

    def _is_hub_message(self, message: ManagementMessage) -> bool:
        return (
            isinstance(message, ManagementMessage)
            and message.protocol_version == PROTOCOL_VERSION
            and message.sender_id == "hub"
        )

    def _handle_from_message_locked(self, message: ManagementMessage) -> ControlHandle | None:
        body = message.body
        raw_handle = body.get("handle")
        if not isinstance(raw_handle, Mapping):
            return None
        try:
            handle = ControlHandle(**dict(raw_handle))
        except (TypeError, ValueError):
            return None
        body_epoch = body.get("hub_epoch")
        if (
            body_epoch != handle.hub_epoch
            or message.sender_session_id != handle.hub_epoch
            or (self._hub_epoch is not None and self._hub_epoch != handle.hub_epoch)
        ):
            return None
        return handle

    def _receive_heartbeat_ack_locked(self, message: ManagementMessage) -> bool:
        epoch = message.body.get("hub_epoch")
        if (
            self._hub_epoch is None
            or epoch != self._hub_epoch
            or message.sender_session_id != self._hub_epoch
            or message.correlation_id != self._pending_heartbeat_correlation
        ):
            return False
        self._pending_heartbeat_correlation = None
        self._missed_heartbeat_acks = 0
        return True

    def _send_handle_event_locked(self, kind: str, handle: ControlHandle) -> bool:
        return self._send_locked(kind, self._new_correlation(kind), {"handle": asdict(handle)})

    def _send_take_over_requested_locked(self, handle: ControlHandle) -> bool:
        correlation_id = self._new_correlation("take_over_requested")
        self._take_over_correlation = correlation_id
        return self._send_locked("take_over_requested", correlation_id, {"handle": asdict(handle)})

    def _send_handle_ack_locked(self, kind: str, handle: ControlHandle, correlation_id: str) -> bool:
        return self._send_locked(f"{kind}_ack", correlation_id, {"handle": asdict(handle)})

    def _send_locked(
        self,
        kind: str,
        correlation_id: str,
        body: Mapping[str, Any],
        *,
        reconnect_on_failure: bool = True,
    ) -> bool:
        channel = self._channel
        if channel is None:
            return False
        message = ManagementMessage(
            protocol_version=PROTOCOL_VERSION,
            kind=kind,
            correlation_id=correlation_id,
            sender_id=self._node_id,
            sender_session_id=self._session_id,
            sequence=self._outgoing_sequence,
            sent_at_ns=self._utc_ns(),
            body=body,
        )
        self._outgoing_sequence += 1
        if channel.send(message):
            return True
        if reconnect_on_failure:
            self._management_lost_locked("management_send_failed")
        return False

    def _management_lost_locked(self, reason: str) -> None:
        self._disable_streaming_locked(reason, terminal=True)
        self._control_state = ControllerControlState.IDLE
        old_channel = self._channel
        self._channel = None
        if old_channel is not None:
            old_channel.close()
        if not self._started:
            return
        self._open_session_locked()
        self._send_registration_locked()

    def _report_locked(self, now: int) -> NodeReport:
        handle = self._handle
        return NodeReport(
            node_id=self._node_id,
            session_id=self._session_id,
            runtime_state=RuntimeState.ONLINE,
            robot_control_state=None,
            controller_control_state=self._control_state,
            handle_id=handle.handle_id if handle is not None else None,
            fencing_token=handle.fencing_token if handle is not None else None,
            action_rate_hz=self.config.timing.action_rate_hz if self._streaming_enabled else 0.0,
            frame_age_ms=None,
            last_sequence=self._sequence - 1 if self._sequence else None,
            tracking=None,
            engaged=None,
            processor_state=None,
            active_hold=None,
            error=self._last_error,
            reported_at_ns=now,
        )

    @staticmethod
    def _wire(value: object) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for data_field in fields(value):  # type: ignore[arg-type]
            item = getattr(value, data_field.name)
            result[data_field.name] = item.value if isinstance(item, Enum) else item
        return result

    @staticmethod
    def _new_correlation(prefix: str) -> str:
        return f"{prefix}:{uuid.uuid4()}"
