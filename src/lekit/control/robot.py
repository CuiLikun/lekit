"""Generic LeRobot Robot ownership, authorization, and local safety loop."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol

from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation

from .model import (
    PROTOCOL_VERSION,
    ControlHandle,
    ManagementMessage,
    NodeDescriptor,
    NodePresentation,
    NodeReport,
    NodeRole,
    RobotControlState,
    RuntimeState,
    TimingConfig,
    load_or_create_node_id,
)
from .runtime import LatestActionReceiver, NodeChannel, ReceivedAction, Runtime


class PayloadProcessor(Protocol):
    """Map one opaque Controller payload and observation to a Robot action."""

    def __call__(self, payload: bytes, observation: RobotObservation) -> RobotAction:
        """Validate and map one complete payload."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all state that could retain motion authority."""
        raise NotImplementedError


class ObservationSink(Protocol):
    """Accept one successfully-read Robot observation without affecting control."""

    def publish(self, observation: RobotObservation, *, captured_monotonic_ns: int) -> None:
        """Offer an observation to an optional non-blocking side channel."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HoldResult:
    """Observed outcome of one local HOLD request."""

    active: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise ValueError("active must be a bool")
        if self.detail is not None and (not isinstance(self.detail, str) or not self.detail):
            raise ValueError("detail must be non-empty when present")


class PassiveHold:
    """Generic HOLD: stop forwarding commands without claiming an active stop."""

    def __call__(
        self,
        robot: Robot,
        observation: RobotObservation | None,
        reason: str,
    ) -> HoldResult:
        del robot, observation, reason
        return HoldResult(active=False)


@dataclass(kw_only=True)
class RobotNodeConfig:
    """Configuration for one independently running Robot process."""

    node_id_path: Path
    display_name: str
    accepted_payload_schemas: tuple[str, ...]
    control_modes: tuple[str, ...] = ("teleop",)
    control_enabled: bool = True
    control_rate_hz: float = 60.0
    action_stale_s: float = 0.1
    hub_seed: str | None = None
    presentation: NodePresentation = field(default_factory=NodePresentation)
    timing: TimingConfig = field(default_factory=TimingConfig)

    def __post_init__(self) -> None:
        self.node_id_path = Path(self.node_id_path)
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must not be empty")
        for name in ("accepted_payload_schemas", "control_modes"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or not value
                or not all(isinstance(item, str) and item for item in value)
            ):
                raise ValueError(f"{name} must be a non-empty tuple of non-empty strings")
        if not isinstance(self.control_enabled, bool):
            raise ValueError("control_enabled must be a bool")
        for name in ("control_rate_hz", "action_stale_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite positive")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite positive")
            setattr(self, name, value)
        if self.hub_seed is not None and (not isinstance(self.hub_seed, str) or not self.hub_seed):
            raise ValueError("hub_seed must be non-empty when present")
        if not isinstance(self.presentation, NodePresentation):
            raise ValueError("presentation must be a NodePresentation")
        if not isinstance(self.timing, TimingConfig):
            raise ValueError("timing must be a TimingConfig")

    @property
    def action_stale_ns(self) -> int:
        return round(self.action_stale_s * 1_000_000_000)


class RobotNode:
    """Own one standard Robot and remain the final local motion authority."""

    _TRANSPORT_REJECTIONS = frozenset(
        {
            "stale_action",
            "wrong_hub_epoch",
            "wrong_handle",
            "wrong_fencing",
            "wrong_controller",
            "wrong_controller_session",
            "wrong_stream_session",
            "sequence_regressed",
            "wrong_payload_schema",
            "processor_invalid",
            "invalid_robot_action",
            "robot_send_failed",
            "handle_expired",
        }
    )

    def __init__(
        self,
        robot: Robot,
        processor: PayloadProcessor,
        config: RobotNodeConfig,
        *,
        runtime: Runtime,
        hold: Callable[[Robot, RobotObservation | None, str], HoldResult] | None = None,
        observation_sinks: tuple[ObservationSink, ...] = (),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not isinstance(config, RobotNodeConfig):
            raise TypeError("config must be a RobotNodeConfig")
        if not callable(processor) or not callable(getattr(processor, "reset", None)):
            raise TypeError("processor must be callable and provide reset()")
        if not isinstance(observation_sinks, tuple) or not all(
            callable(getattr(sink, "publish", None)) for sink in observation_sinks
        ):
            raise TypeError("observation_sinks must be a tuple of ObservationSink")
        self.robot = robot
        self.processor = processor
        self.config = config
        self.runtime = runtime
        self._hold = hold or PassiveHold()
        self._monotonic_ns = monotonic_ns
        self._utc_ns = utc_ns
        self._observation_sinks = observation_sinks
        self._lock = threading.RLock()
        self._node_id = load_or_create_node_id(config.node_id_path)
        self._session_id = "not-started"
        self._hub_epoch: str | None = None
        self._registered = False
        self._channel: NodeChannel | None = None
        self._receiver: LatestActionReceiver | None = None
        self._started = False
        self._management_stop = threading.Event()
        self._control_stop = threading.Event()
        self._management_thread: threading.Thread | None = None
        self._outgoing_sequence = 0
        self._last_hub_command_sequence = -1
        self._registration_correlation: str | None = None
        self._pending_heartbeat_correlation: str | None = None
        self._missed_heartbeat_acks = 0
        self._next_heartbeat_ns = 0
        self._next_status_ns = 0
        self._runtime_state = RuntimeState.STOPPED
        self._control_state = RobotControlState.HOLD
        self._handle: ControlHandle | None = None
        self._terminal_handle_ids: set[str] = set()
        self._terminal_authorities: dict[str, tuple[Any, ...]] = {}
        self._local_expiry_monotonic_ns = 0
        self._authority_generation = 0
        self._authority_transition_depth = 0
        self._stream_session_id: str | None = None
        self._last_sequence: int | None = None
        self._last_received_monotonic_ns: int | None = None
        self._accepted_timestamps: deque[int] = deque()
        self._last_error: str | None = None
        self._observation_sink_errors: Counter[str] = Counter()
        self._active_hold = False
        self._safety_reason: str | None = None
        self.rejections: Counter[str] = Counter()

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    @property
    def control_state(self) -> RobotControlState:
        with self._lock:
            return self._control_state

    @property
    def stream_session_id(self) -> str | None:
        with self._lock:
            return self._stream_session_id

    @property
    def status(self) -> Mapping[str, Any]:
        """Return a deterministic local diagnostic snapshot."""
        with self._lock:
            now = self._monotonic_ns()
            processor_status = self._processor_status_locked()
            report = self._report_locked(now, processor_status=processor_status)
            return {
                "report": self._wire(report),
                **self._diagnostics_locked(processor_status=processor_status),
            }

    def start(self) -> None:
        """Connect exactly once, register, and start bounded management work."""
        primary: BaseException | None = None
        channel: NodeChannel | None = None
        thread: threading.Thread | None = None
        thread_started = False
        connect_attempted = False
        with self._lock:
            if self._started:
                return
            self._runtime_state = RuntimeState.STARTING
            self._management_stop.clear()
            self._control_stop.clear()
            try:
                connect_attempted = True
                self.robot.connect()
                self._open_session_locked()
                if not self._send_registration_locked():
                    raise RuntimeError("registration send failed")
                thread = threading.Thread(
                    target=self._run_management,
                    name=f"robot-management-{self._node_id}",
                    daemon=True,
                )
                self._management_thread = thread
                thread.start()
                thread_started = True
                self._started = True
                self._runtime_state = RuntimeState.STARTING
                self._control_state = RobotControlState.HOLD
                return
            except BaseException as error:
                primary = error
                self._started = False
                self._registered = False
                self._runtime_state = RuntimeState.STOPPED
                self._control_state = RobotControlState.HOLD
                self._last_error = f"startup failed: {error}"
                self._management_stop.set()
                self._control_stop.set()
                channel = self._channel
                self._channel = None
                self._management_thread = None
                self._receiver = None
                self._handle = None
                self._local_expiry_monotonic_ns = 0
        assert primary is not None
        cleanup_errors: list[BaseException] = []
        if channel is not None:
            try:
                channel.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if thread_started and thread is not None:
            try:
                thread.join(timeout=1.0)
            except BaseException as error:
                cleanup_errors.append(error)
        if connect_attempted:
            try:
                self.robot.disconnect()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            primary.add_note(f"startup cleanup failures: {details}")
            with self._lock:
                self._last_error = f"startup failed: {primary}; cleanup failed: {details}"
        raise primary

    def stop(self) -> None:
        """Remove authority, close owned channels, and disconnect the Robot."""
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._management_stop.set()
            self._control_stop.set()
            cleanup_errors = self._enter_hold_locked(
                "stopped", None, close_receiver=True, distinct=True, terminal=True
            )
            channel = self._channel
            thread = self._management_thread
            self._channel = None
            self._management_thread = None
            self._runtime_state = RuntimeState.STOPPED
        all_errors = list(cleanup_errors)
        if channel is not None:
            try:
                channel.close()
            except BaseException as error:
                all_errors.append(error)
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=1.0)
            except BaseException as error:
                all_errors.append(error)
        try:
            self.robot.disconnect()
        except BaseException as error:
            all_errors.append(error)
        if all_errors:
            primary = all_errors[0]
            if len(all_errors) > 1:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in all_errors[1:])
                primary.add_note(f"additional cleanup failures: {details}")
                with self._lock:
                    self._last_error = f"{self._last_error}; additional cleanup failures: {details}"
            raise primary

    def run(self) -> None:
        """Run the Robot loop at the configured monotonic control rate until stopped."""
        period_ns = round(1_000_000_000 / self.config.control_rate_hz)
        deadline = self._monotonic_ns()
        primary: BaseException | None = None
        try:
            while not self._control_stop.is_set():
                self.run_cycle()
                deadline += period_ns
                remaining_ns = deadline - self._monotonic_ns()
                if remaining_ns > 0:
                    self._control_stop.wait(remaining_ns / 1_000_000_000)
                else:
                    deadline = self._monotonic_ns()
        except BaseException as error:
            primary = error
        finally:
            try:
                self.stop()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(f"shutdown failure: {type(cleanup_error).__name__}: {cleanup_error}")
        if primary is not None:
            raise primary

    def run_cycle(self) -> RobotObservation:
        """Read observation and consider at most the latest non-blocking action."""
        observation = self.robot.get_observation()
        captured_monotonic_ns = self._monotonic_ns()
        for sink in self._observation_sinks:
            try:
                sink.publish(observation, captured_monotonic_ns=captured_monotonic_ns)
            except Exception as error:
                with self._lock:
                    self._observation_sink_errors[f"{type(error).__name__}: {error}"] += 1
        with self._lock:
            if self._authority_transition_depth:
                return observation
            if self._control_state is RobotControlState.SAFETY:
                self.rejections["safety"] += 1
                return observation
            handle = self._handle
            if handle is None:
                self._watchdog_locked(observation)
                return observation
            now = self._monotonic_ns()
            if now >= self._local_expiry_monotonic_ns:
                self._reject_and_hold_locked("handle_expired", observation)
                return observation
            receiver = self._receiver
            if receiver is None:
                self._watchdog_locked(observation)
                return observation
            received = receiver.receive_latest(timeout_s=0.0)
            if received is None:
                self._watchdog_locked(observation)
                return observation
            reason = self._validate_frame_locked(received, now)
            if reason is not None:
                self._reject_and_hold_locked(reason, observation)
                return observation
            envelope = received.envelope
            if self._stream_session_id is None:
                self._stream_session_id = envelope.stream_session_id
            self._last_sequence = envelope.sequence
            self._last_received_monotonic_ns = received.received_monotonic_ns
            generation = self._authority_generation
            authority = self._authority_identity(handle)
            stream_session_id = envelope.stream_session_id
            sequence = envelope.sequence

        try:
            action = self.processor(envelope.payload, observation)
        except Exception as error:
            with self._lock:
                if self._speculation_is_current_locked(
                    generation,
                    authority,
                    receiver,
                    stream_session_id,
                    sequence,
                ):
                    self._reject_and_hold_locked("processor_invalid", observation)
                    self._last_error = f"processor_invalid: {error}"
            return observation
        if not action:
            with self._lock:
                if self._speculation_is_current_locked(
                    generation,
                    authority,
                    receiver,
                    stream_session_id,
                    sequence,
                ):
                    self.rejections["processor_not_armed"] += 1
                    self._enter_hold_locked("processor_not_armed", observation)
                    if self._last_error == "processor_not_armed":
                        self._last_error = None
            return observation
        try:
            self._validate_robot_action(action)
        except (TypeError, ValueError) as error:
            with self._lock:
                if self._speculation_is_current_locked(
                    generation,
                    authority,
                    receiver,
                    stream_session_id,
                    sequence,
                ):
                    self._reject_and_hold_locked("invalid_robot_action", observation)
                    self._last_error = f"invalid_robot_action: {error}"
            return observation

        with self._lock:
            final_reason = self._final_send_rejection_locked(
                generation,
                authority,
                receiver,
                received,
                stream_session_id,
                sequence,
            )
            if final_reason == "authority_changed":
                return observation
            if final_reason is not None:
                self._reject_and_hold_locked(final_reason, observation)
                return observation
            try:
                self.robot.send_action(action)
            except Exception as error:
                self._reject_and_hold_locked("robot_send_failed", observation)
                self._last_error = f"robot_send_failed: {error}"
                return observation
            self._control_state = RobotControlState.CONTROLLING
            self._active_hold = False
            self._last_error = None
            accepted_at = self._monotonic_ns()
            self._accepted_timestamps.append(accepted_at)
            self._trim_action_rate_locked(accepted_at)
            return observation

    def _speculation_is_current_locked(
        self,
        generation: int,
        authority: tuple[Any, ...],
        receiver: LatestActionReceiver,
        stream_session_id: str,
        sequence: int,
    ) -> bool:
        handle = self._handle
        return (
            self._control_state is not RobotControlState.SAFETY
            and self._authority_transition_depth == 0
            and handle is not None
            and self._authority_generation == generation
            and self._authority_identity(handle) == authority
            and self._receiver is receiver
            and self._stream_session_id == stream_session_id
            and self._last_sequence == sequence
        )

    def _final_send_rejection_locked(
        self,
        generation: int,
        authority: tuple[Any, ...],
        receiver: LatestActionReceiver,
        received: ReceivedAction,
        stream_session_id: str,
        sequence: int,
    ) -> str | None:
        handle = self._handle
        if (
            self._control_state is RobotControlState.SAFETY
            or self._authority_transition_depth != 0
            or handle is None
            or self._authority_generation != generation
            or self._authority_identity(handle) != authority
        ):
            return "authority_changed"
        now = self._monotonic_ns()
        if now >= self._local_expiry_monotonic_ns:
            return "handle_expired"
        if (
            self._receiver is not receiver
            or self._stream_session_id != stream_session_id
            or self._last_sequence != sequence
        ):
            return "authority_changed"
        if now - received.received_monotonic_ns > self.config.action_stale_ns:
            return "stale_action"
        return None

    def receive_grant(self, handle: ControlHandle) -> bool:
        """Cache a Hub grant without opening or accepting action input."""
        with self._lock:
            return self._started and self._accept_grant_locked(handle)

    def receive_management(self, message: ManagementMessage) -> bool:
        """Validate and apply one current Hub command idempotently."""
        with self._lock:
            if not self._started or not self._is_hub_message(message):
                return False
            if message.kind == "registered":
                if message.sequence <= self._last_hub_command_sequence:
                    return False
                epoch = message.body.get("hub_epoch")
                if (
                    not isinstance(epoch, str)
                    or not epoch
                    or message.sender_session_id != epoch
                    or message.correlation_id != self._registration_correlation
                ):
                    return False
                if self._hub_epoch is not None and self._hub_epoch != epoch:
                    self._clear_authority_locked("hub_epoch_changed")
                self._hub_epoch = epoch
                self._registered = True
                self._runtime_state = RuntimeState.ONLINE
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
            if message.kind not in {"grant", "renewal", "take_over", "hand_over", "revoke", "force_hold"}:
                return False
            if not self._registered or self._hub_epoch is None:
                return False
            if self._authority_transition_depth:
                return False
            if message.sequence <= self._last_hub_command_sequence:
                return False
            command_session = self._session_id
            if message.kind == "force_hold" and "handle" not in message.body:
                if message.sender_session_id != self._hub_epoch:
                    return False
                reason = message.body.get("reason", "force_hold")
                self._enter_hold_locked(str(reason), None, close_receiver=True, distinct=True)
                accepted = self._send_ack_locked("force_hold", None, message.correlation_id)
            else:
                handle = self._handle_from_message_locked(message)
                if handle is None:
                    return False
                if message.kind == "grant":
                    accepted = self._accept_grant_locked(handle)
                elif message.kind == "renewal":
                    accepted = self._renew_locked(handle)
                elif message.kind == "take_over":
                    accepted = self._accept_take_over_locked(handle, message.correlation_id)
                elif message.kind == "hand_over":
                    accepted = self._accept_terminal_command_locked(
                        handle, "hand_over", message.correlation_id
                    )
                elif message.kind == "revoke":
                    accepted = self._accept_terminal_command_locked(handle, "revoke", message.correlation_id)
                else:
                    accepted = self._accept_force_hold_locked(handle, message)
                if accepted and self._session_id == command_session:
                    accepted = self._send_ack_locked(message.kind, handle, message.correlation_id)
                else:
                    accepted = False
            if accepted and self._session_id == command_session:
                self._last_hub_command_sequence = message.sequence
                return True
            return False

    def run_management_once(self, *, timeout_s: float = 0.0) -> bool:
        """Run one bounded receive, heartbeat, status, and expiry iteration."""
        with self._lock:
            if not self._started:
                return False
            if self._channel is None:
                self._attempt_reconnect_locked("management_reconnect")
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
            if self._handle is not None and now >= self._local_expiry_monotonic_ns:
                self._reject_and_hold_locked("handle_expired", None)
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
                processor_status = self._processor_status_locked()
                report = self._report_locked(now, processor_status=processor_status)
                self._send_locked(
                    "status",
                    self._new_correlation("status"),
                    {
                        "report": self._wire(report),
                        "diagnostics": self._diagnostics_locked(processor_status=processor_status),
                    },
                )
                self._next_status_ns = now + round(1_000_000_000 / self.config.timing.status_rate_hz)
        return processed

    def enter_safety(self, reason: str) -> None:
        """Preempt all authority until an explicit local clear."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must not be empty")
        with self._lock:
            if self._control_state is RobotControlState.SAFETY:
                return
            self._safety_reason = reason
            self._enter_hold_locked(
                reason,
                None,
                close_receiver=True,
                distinct=True,
                terminal=True,
                safety=True,
            )

    def clear_safety(self) -> None:
        """Locally clear SAFETY without restoring any prior authority."""
        with self._lock:
            if self._authority_transition_depth or self._control_state is not RobotControlState.SAFETY:
                return
            self._safety_reason = None
            self._control_state = RobotControlState.HOLD
            self._last_error = None

    def _run_management(self) -> None:
        while not self._management_stop.is_set():
            try:
                self.run_management_once(timeout_s=0.05)
            except Exception as error:
                with self._lock:
                    if self._started:
                        self._last_error = f"management loop failed: {error}"
                        self._management_lost_locked("management_loop_failed")
            self._management_stop.wait(0.01)

    def _open_session_locked(self) -> None:
        self._session_id = str(uuid.uuid4())
        self._hub_epoch = None
        self._registered = False
        self._outgoing_sequence = 0
        self._last_hub_command_sequence = -1
        self._registration_correlation = None
        self._pending_heartbeat_correlation = None
        self._missed_heartbeat_acks = 0
        self._channel = self.runtime.open_node(self._node_id, self._session_id, hub_seed=self.config.hub_seed)
        now = self._monotonic_ns()
        self._next_heartbeat_ns = now + self.config.timing.heartbeat_interval_ns
        self._next_status_ns = now + round(1_000_000_000 / self.config.timing.status_rate_hz)

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
            role=NodeRole.ROBOT,
            display_name=self.config.display_name,
            administratively_enabled=self.config.control_enabled,
            capabilities=("robot",),
            action_schemas=self.config.accepted_payload_schemas,
            control_modes=self.config.control_modes,
            action_endpoint=None,
            observation_features=self.robot.observation_features,
            action_features=self.robot.action_features,
            software_version="lekit",
            diagnostics={"passive_hold": isinstance(self._hold, PassiveHold)},
            presentation=self.config.presentation,
        )

    def _accept_grant_locked(self, handle: ControlHandle) -> bool:
        if (
            not self._registered
            or self._hub_epoch is None
            or not self.config.control_enabled
            or self._control_state is RobotControlState.SAFETY
        ):
            return False
        if not self._accept_handle_target_locked(handle) or handle.handle_id in self._terminal_handle_ids:
            return False
        if self._hub_epoch != handle.hub_epoch:
            return False
        if self._handle is not None:
            return self._same_authority(self._handle, handle)
        self._handle = handle
        self._set_local_expiry_locked(handle)
        return True

    def _renew_locked(self, handle: ControlHandle) -> bool:
        if self._handle is not None and self._monotonic_ns() >= self._local_expiry_monotonic_ns:
            self._reject_and_hold_locked("handle_expired", None)
            return False
        if (
            self._control_state is RobotControlState.SAFETY
            or self._handle is None
            or not self._same_authority(self._handle, handle)
            or handle.expires_at_ns <= self._handle.expires_at_ns
        ):
            return False
        self._handle = handle
        self._set_local_expiry_locked(handle)
        return True

    def _accept_take_over_locked(self, handle: ControlHandle, correlation_id: str) -> bool:
        if (
            not self.config.control_enabled
            or self._control_state is RobotControlState.SAFETY
            or not self._matches_current_handle_locked(handle)
            or self._monotonic_ns() >= self._local_expiry_monotonic_ns
        ):
            return False
        cleanup_errors = self._prepare_take_over_locked()
        if cleanup_errors:
            return False
        try:
            receiver = self.runtime.open_action_receiver(handle.controller_action_endpoint)
        except Exception as error:
            self._record_cleanup_errors_locked("take_over", [error])
            return False
        self._receiver = receiver
        self._authority_generation += 1
        self._stream_session_id = None
        self._last_sequence = None
        self._last_received_monotonic_ns = None
        self._last_error = None
        return self._send_handle_event_locked("robot_ready", handle, correlation_id)

    def _accept_terminal_command_locked(self, handle: ControlHandle, kind: str, correlation_id: str) -> bool:
        if not self._matches_current_handle_locked(handle):
            return self._is_terminal_authority_locked(handle)
        self._enter_hold_locked(kind, None, close_receiver=True, distinct=True, terminal=True)
        return self._send_handle_event_locked("robot_holding", handle, correlation_id)

    def _accept_force_hold_locked(self, handle: ControlHandle, message: ManagementMessage) -> bool:
        if not self._matches_current_handle_locked(handle):
            return False
        reason = message.body.get("reason", "force_hold")
        self._enter_hold_locked(str(reason), None, close_receiver=True, distinct=True)
        return True

    def _validate_frame_locked(self, received: ReceivedAction, now: int) -> str | None:
        envelope = received.envelope
        handle = self._handle
        if now - received.received_monotonic_ns > self.config.action_stale_ns:
            return "stale_action"
        assert handle is not None
        if envelope.hub_epoch != handle.hub_epoch or envelope.hub_epoch != self._hub_epoch:
            return "wrong_hub_epoch"
        if envelope.handle_id != handle.handle_id:
            return "wrong_handle"
        if envelope.fencing_token != handle.fencing_token:
            return "wrong_fencing"
        if envelope.controller_id != handle.controller_id:
            return "wrong_controller"
        if envelope.controller_session_id != handle.controller_session_id:
            return "wrong_controller_session"
        if self._stream_session_id is not None and envelope.stream_session_id != self._stream_session_id:
            return "wrong_stream_session"
        if self._last_sequence is not None and envelope.sequence <= self._last_sequence:
            return "sequence_regressed"
        if (
            envelope.payload_schema != handle.action_schema
            or envelope.payload_schema not in self.config.accepted_payload_schemas
        ):
            return "wrong_payload_schema"
        return None

    def _validate_robot_action(self, action: RobotAction) -> None:
        if not isinstance(action, Mapping):
            raise TypeError("action must be a mapping")
        features = self.robot.action_features
        unknown = set(action) - set(features)
        if unknown:
            raise ValueError(f"action contains unsupported features: {sorted(unknown)}")
        for key, value in action.items():
            self._validate_feature_value(value, features[key], key)

    @classmethod
    def _validate_feature_value(cls, value: Any, feature: Any, name: str) -> None:
        if feature is float:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real scalar")
            return
        if feature is int:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer scalar")
            return
        if isinstance(feature, tuple) and all(isinstance(size, int) and size >= 0 for size in feature):
            shape = tuple(getattr(value, "shape", ()))
            if not shape and isinstance(value, (list, tuple)):
                shape = cls._nested_shape(value)
            if shape != feature:
                raise ValueError(f"{name} must have shape {feature}")
            for scalar in cls._flatten(value):
                if (
                    isinstance(scalar, bool)
                    or not isinstance(scalar, Real)
                    or not math.isfinite(float(scalar))
                ):
                    raise ValueError(f"{name} must contain finite real values")
            return
        if isinstance(feature, type) and isinstance(value, feature):
            return
        raise ValueError(f"unsupported or incompatible action feature {name}")

    @classmethod
    def _nested_shape(cls, value: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
        if not value:
            return (0,)
        child_shapes = [cls._nested_shape(item) if isinstance(item, (list, tuple)) else () for item in value]
        if any(shape != child_shapes[0] for shape in child_shapes):
            return (-1,)
        return (len(value), *child_shapes[0])

    @classmethod
    def _flatten(cls, value: Any):
        if hasattr(value, "flat"):
            yield from value.flat
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._flatten(item)
        else:
            yield value

    def _watchdog_locked(self, observation: RobotObservation | None) -> None:
        if self._receiver is None or self._last_received_monotonic_ns is None:
            return
        if self._monotonic_ns() - self._last_received_monotonic_ns > self.config.action_stale_ns:
            self._reject_and_hold_locked("stale_action", observation)

    def _reject_and_hold_locked(self, reason: str, observation: RobotObservation | None) -> None:
        self.rejections[reason] += 1
        self._last_error = reason
        close_receiver = reason in self._TRANSPORT_REJECTIONS
        self._enter_hold_locked(
            reason,
            observation,
            close_receiver=close_receiver,
            distinct=close_receiver,
            terminal=reason == "handle_expired",
        )

    def _enter_hold_locked(
        self,
        reason: str,
        observation: RobotObservation | None,
        *,
        close_receiver: bool = False,
        distinct: bool = False,
        terminal: bool = False,
        safety: bool = False,
    ) -> list[BaseException]:
        target_state = RobotControlState.SAFETY if safety else RobotControlState.HOLD
        if self._control_state is RobotControlState.SAFETY and not safety:
            target_state = RobotControlState.SAFETY
        transitioned = self._control_state is not target_state
        self._control_state = target_state
        self._last_error = reason
        detached_receiver = self._receiver if close_receiver else None
        if close_receiver:
            self._receiver = None
            self._stream_session_id = None
            self._last_sequence = None
            self._last_received_monotonic_ns = None
        if terminal and self._handle is not None:
            self._mark_terminal_locked(self._handle)
            self._handle = None
            self._local_expiry_monotonic_ns = 0
        needs_cleanup = transitioned or distinct
        if close_receiver or terminal or needs_cleanup:
            self._authority_generation += 1

        errors: list[BaseException] = []
        self._authority_transition_depth += 1
        try:
            if detached_receiver is not None:
                try:
                    detached_receiver.close()
                except BaseException as error:
                    errors.append(error)
            if needs_cleanup:
                self._active_hold = False
                try:
                    self.processor.reset()
                except BaseException as error:
                    errors.append(error)
                try:
                    self._invoke_hold_locked(reason, observation)
                except BaseException as error:
                    errors.append(error)
            if errors:
                failed_receiver = self._receiver
                self._receiver = None
                self._stream_session_id = None
                self._last_sequence = None
                self._last_received_monotonic_ns = None
                if self._handle is not None:
                    self._mark_terminal_locked(self._handle)
                    self._handle = None
                    self._local_expiry_monotonic_ns = 0
                self._authority_generation += 1
                if failed_receiver is not None:
                    try:
                        failed_receiver.close()
                    except BaseException as error:
                        errors.append(error)
        finally:
            self._authority_transition_depth -= 1
        if errors:
            self._record_cleanup_errors_locked(reason, errors)
        return errors

    def _prepare_take_over_locked(self) -> list[BaseException]:
        self._control_state = RobotControlState.HOLD
        detached_receiver = self._receiver
        self._receiver = None
        self._stream_session_id = None
        self._last_sequence = None
        self._last_received_monotonic_ns = None
        self._authority_generation += 1
        errors: list[BaseException] = []
        self._authority_transition_depth += 1
        try:
            if detached_receiver is not None:
                try:
                    detached_receiver.close()
                except BaseException as error:
                    errors.append(error)
            try:
                self.processor.reset()
            except BaseException as error:
                errors.append(error)
        finally:
            self._authority_transition_depth -= 1
        if errors:
            self._record_cleanup_errors_locked("take_over", errors)
        return errors

    def _invoke_hold_locked(self, reason: str, observation: RobotObservation | None) -> None:
        result = self._hold(self.robot, observation, reason)
        if not isinstance(result, HoldResult):
            raise TypeError("hold callback must return HoldResult")
        self._active_hold = result.active
        if result.detail is not None:
            self._last_error = f"{reason}: {result.detail}"

    def _clear_authority_locked(self, reason: str, *, terminal: bool = False) -> None:
        handle = self._handle
        if terminal and handle is not None:
            self._mark_terminal_locked(handle)
        self._handle = None
        self._local_expiry_monotonic_ns = 0
        self._enter_hold_locked(reason, None, close_receiver=True, distinct=True)

    def _close_receiver_locked(self) -> None:
        receiver = self._receiver
        self._receiver = None
        if receiver is not None:
            receiver.close()

    def _matches_current_handle_locked(self, handle: ControlHandle) -> bool:
        return (
            handle.handle_id not in self._terminal_handle_ids
            and self._handle is not None
            and self._same_authority(self._handle, handle)
            and self._hub_epoch == handle.hub_epoch
            and self._session_id == handle.robot_session_id
        )

    def _mark_terminal_locked(self, handle: ControlHandle) -> None:
        self._terminal_handle_ids.add(handle.handle_id)
        self._terminal_authorities[handle.handle_id] = self._authority_identity(handle)

    def _is_terminal_authority_locked(self, handle: ControlHandle) -> bool:
        return self._terminal_authorities.get(handle.handle_id) == self._authority_identity(handle)

    def _accept_handle_target_locked(self, handle: ControlHandle) -> bool:
        return (
            isinstance(handle, ControlHandle)
            and handle.robot_id == self._node_id
            and handle.robot_session_id == self._session_id
            and handle.action_schema in self.config.accepted_payload_schemas
            and handle.control_mode in self.config.control_modes
        )

    @staticmethod
    def _same_authority(left: ControlHandle, right: ControlHandle) -> bool:
        return RobotNode._authority_identity(left) == RobotNode._authority_identity(right)

    @staticmethod
    def _authority_identity(handle: ControlHandle) -> tuple[Any, ...]:
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
        return tuple(getattr(handle, name) for name in names)

    def _set_local_expiry_locked(self, handle: ControlHandle) -> None:
        remaining_ns = max(0, handle.expires_at_ns - self._utc_ns())
        self._local_expiry_monotonic_ns = self._monotonic_ns() + min(
            self.config.timing.handle_ttl_ns, remaining_ns
        )

    def _handle_from_message_locked(self, message: ManagementMessage) -> ControlHandle | None:
        if not self._registered or self._hub_epoch is None:
            return None
        raw_handle = message.body.get("handle")
        if not isinstance(raw_handle, Mapping):
            return None
        try:
            handle = ControlHandle(**dict(raw_handle))
        except (TypeError, ValueError):
            return None
        if (
            message.body.get("hub_epoch") != handle.hub_epoch
            or message.sender_session_id != handle.hub_epoch
            or self._hub_epoch != handle.hub_epoch
        ):
            return None
        return handle

    def _is_hub_message(self, message: ManagementMessage) -> bool:
        return (
            isinstance(message, ManagementMessage)
            and message.protocol_version == PROTOCOL_VERSION
            and message.sender_id == "hub"
        )

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

    def _send_ack_locked(self, kind: str, handle: ControlHandle | None, correlation_id: str) -> bool:
        body: dict[str, Any] = {}
        if handle is not None:
            body["handle"] = asdict(handle)
        return self._send_locked(f"{kind}_ack", correlation_id, body)

    def _send_handle_event_locked(self, kind: str, handle: ControlHandle, correlation_id: str) -> bool:
        return self._send_locked(kind, correlation_id, {"handle": asdict(handle)})

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
        try:
            sent = channel.send(message)
        except Exception as error:
            if reconnect_on_failure:
                self._management_lost_locked(f"management_send_failed: {error}")
            else:
                self._last_error = f"management_send_failed: {error}"
                raise
            return False
        if sent:
            return True
        if reconnect_on_failure:
            self._management_lost_locked("management_send_failed")
        return False

    def _management_lost_locked(self, reason: str) -> None:
        self._clear_authority_locked(reason, terminal=True)
        self._registered = False
        self._runtime_state = RuntimeState.DEGRADED
        old_channel = self._channel
        self._channel = None
        if old_channel is not None:
            try:
                old_channel.close()
            except BaseException as error:
                self._record_cleanup_errors_locked(reason, [error])
        if not self._started:
            return
        self._attempt_reconnect_locked(reason)

    def _attempt_reconnect_locked(self, reason: str) -> bool:
        try:
            self._open_session_locked()
            if not self._send_registration_locked():
                raise RuntimeError("registration send failed")
        except Exception as error:
            failed_channel = self._channel
            self._channel = None
            self._registered = False
            self._runtime_state = RuntimeState.DEGRADED
            errors: list[BaseException] = [error]
            if failed_channel is not None:
                try:
                    failed_channel.close()
                except BaseException as close_error:
                    errors.append(close_error)
            details = "; ".join(f"{type(item).__name__}: {item}" for item in errors)
            self._last_error = f"{reason}; reconnect failed: {details}"
            return False
        self._runtime_state = RuntimeState.DEGRADED
        return True

    def _record_cleanup_errors_locked(self, reason: str, errors: list[BaseException]) -> None:
        details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        diagnostic = f"{reason}; cleanup failed: {details}"
        if self._last_error and self._last_error != reason:
            self._last_error = f"{self._last_error}; {diagnostic}"
        else:
            self._last_error = diagnostic

    def _processor_status_locked(self) -> Mapping[str, Any]:
        status_method = getattr(self.processor, "status", None)
        if not callable(status_method):
            return {}
        try:
            candidate = status_method()
        except Exception as error:
            return {"error": f"processor status failed: {error}"}
        return dict(candidate) if isinstance(candidate, Mapping) else {}

    def _report_locked(
        self,
        now: int,
        *,
        processor_status: Mapping[str, Any] | None = None,
    ) -> NodeReport:
        self._trim_action_rate_locked(now)
        frame_age_ms = None
        if self._last_received_monotonic_ns is not None:
            frame_age_ms = max(0.0, (now - self._last_received_monotonic_ns) / 1_000_000)
        if processor_status is None:
            processor_status = self._processor_status_locked()
        processor_state = processor_status.get("processor_state", getattr(self.processor, "state", None))
        if isinstance(processor_state, Enum):
            processor_state = processor_state.value
        elif processor_state is not None:
            processor_state = str(processor_state)
        tracking = processor_status.get("tracking")
        if not isinstance(tracking, bool):
            tracking = None
        engaged = processor_status.get("engaged")
        if not isinstance(engaged, bool):
            engaged = None
        processor_error = processor_status.get("error")
        if processor_error is not None:
            processor_error = str(processor_error) or None
        report_error = self._last_error
        if processor_error is not None:
            report_error = (
                processor_error
                if report_error is None or report_error == processor_error
                else f"{report_error}; processor error: {processor_error}"
            )
        handle = self._handle
        return NodeReport(
            node_id=self._node_id,
            session_id=self._session_id,
            runtime_state=self._runtime_state,
            robot_control_state=self._control_state,
            controller_control_state=None,
            handle_id=handle.handle_id if handle is not None else None,
            fencing_token=handle.fencing_token if handle is not None else None,
            action_rate_hz=float(len(self._accepted_timestamps)),
            frame_age_ms=frame_age_ms,
            last_sequence=self._last_sequence,
            tracking=tracking,
            engaged=engaged,
            processor_state=processor_state,
            active_hold=self._active_hold,
            error=report_error,
            reported_at_ns=now,
        )

    def _diagnostics_locked(
        self,
        *,
        processor_status: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if processor_status is None:
            processor_status = self._processor_status_locked()
        return {
            "rejections": dict(sorted(self.rejections.items())),
            "observation_sink_errors": dict(sorted(self._observation_sink_errors.items())),
            "robot_connected": bool(self.robot.is_connected),
            "processor_status": dict(processor_status),
        }

    def _trim_action_rate_locked(self, now: int) -> None:
        cutoff = now - 1_000_000_000
        while self._accepted_timestamps and self._accepted_timestamps[0] < cutoff:
            self._accepted_timestamps.popleft()

    @staticmethod
    def _wire(value: object) -> dict[str, Any]:
        def wire(item: object) -> object:
            if isinstance(item, Enum):
                return item.value
            if is_dataclass(item) and not isinstance(item, type):
                return {data_field.name: wire(getattr(item, data_field.name)) for data_field in fields(item)}
            if isinstance(item, Mapping):
                return {key: wire(nested) for key, nested in item.items()}
            if isinstance(item, tuple):
                return [wire(nested) for nested in item]
            return item

        encoded = wire(value)
        if not isinstance(encoded, dict):
            raise TypeError("wire values must be dataclasses")
        return encoded

    @staticmethod
    def _new_correlation(prefix: str) -> str:
        return f"{prefix}:{uuid.uuid4()}"


__all__ = [
    "HoldResult",
    "PassiveHold",
    "PayloadProcessor",
    "RobotNode",
    "RobotNodeConfig",
]
