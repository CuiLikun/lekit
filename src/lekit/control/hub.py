"""Authoritative registry, scheduling, and live correlation for Control Hub."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .handles import HandleRecord, InvalidHandleTransition, correlate_control
from .model import (
    PROTOCOL_VERSION,
    TERMINAL_HANDLE_STATES,
    ControlHandle,
    ControllerControlState,
    ControlSnapshot,
    HandleState,
    HubSnapshot,
    ManagementMessage,
    NodeDescriptor,
    NodeReport,
    NodeRole,
    NodeSnapshot,
    RobotControlState,
    RuntimeState,
    TimingConfig,
)
from .runtime import HubChannel, ReceivedManagement, Runtime
from .store import HubStore

_HUB_ID = "hub"
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "*"})
_HANDLE_IDENTITY_FIELDS = (
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


class ControlError(RuntimeError):
    """Base class for Hub authority errors."""


class ControlConflict(ControlError):  # noqa: N818 - public protocol name.
    """Raised when current authority conflicts with a requested action."""


class NodeUnavailable(ControlError):  # noqa: N818 - public protocol name.
    """Raised when a Node cannot currently be scheduled or commanded."""


class IncompatibleNode(ControlError):  # noqa: N818 - public protocol name.
    """Raised when a Node or Node pair violates the Hub protocol contract."""


@dataclass(kw_only=True)
class HubConfig:
    """Configuration for one authoritative Hub process."""

    management_endpoint: str = "tcp://0.0.0.0:5560"
    advertise_endpoint: str | None = None
    database_path: Path = Path(".lekit/control-hub.sqlite3")
    timing: TimingConfig = field(default_factory=TimingConfig)
    auto_revoke_mismatches: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.management_endpoint, str) or not self.management_endpoint:
            raise ValueError("management_endpoint must not be empty")
        if self.advertise_endpoint is not None and (
            not isinstance(self.advertise_endpoint, str) or not self.advertise_endpoint
        ):
            raise ValueError("advertise_endpoint must be non-empty when present")
        self.database_path = Path(self.database_path)
        if not isinstance(self.timing, TimingConfig):
            raise ValueError("timing must be a TimingConfig")
        if not isinstance(self.auto_revoke_mismatches, bool):
            raise ValueError("auto_revoke_mismatches must be a bool")


@dataclass(slots=True)
class _LiveNode:
    descriptor: NodeDescriptor
    last_seen_monotonic_ns: int
    online: bool = True
    report: NodeReport | None = None


@dataclass(slots=True)
class _LiveHandle:
    record: HandleRecord
    robot_ready: bool = False
    controller_streaming: bool = False
    controller_released: bool = False
    pending_grants: set[str] = field(default_factory=set)
    last_renewed_monotonic_ns: int = 0


@dataclass(frozen=True, slots=True)
class _PendingDelivery:
    node_id: str
    kind: str
    correlation_id: str
    handle_id: str | None

    def alert(self) -> Mapping[str, Any]:
        return {
            "code": "management_delivery_pending",
            "correlation_id": self.correlation_id,
            "handle_id": self.handle_id,
            "kind": self.kind,
            "node_id": self.node_id,
            "severity": "error",
        }


@dataclass(frozen=True, slots=True)
class _OutstandingCommand:
    node_id: str
    node_session_id: str
    node_role: NodeRole
    expected_ack_kind: str
    correlation_id: str
    handle_id: str
    hub_epoch: str
    fencing_token: int


class Hub:
    """Own registry, Handle authority, liveness, routing, and snapshots."""

    def __init__(
        self,
        config: HubConfig | None = None,
        *,
        runtime: Runtime,
        store: HubStore | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.config = config or HubConfig()
        self._runtime = runtime
        self._store = store
        self._monotonic_ns = monotonic_ns
        self._utc_ns = utc_ns
        self._timing = self.config.timing
        self._lock = threading.RLock()
        self._snapshot_changed = threading.Condition(self._lock)
        self._channel: HubChannel | None = None
        self._started = False
        self._hub_epoch = "not-started"
        self._nodes: dict[str, _LiveNode] = {}
        self._handles: dict[str, _LiveHandle] = {}
        self._seen_sessions: dict[str, set[str]] = {}
        self._last_sequences: dict[tuple[str, str], int] = {}
        self._pending_deliveries: dict[tuple[str, str, str], _PendingDelivery] = {}
        self._outstanding_commands: dict[tuple[str, str], _OutstandingCommand] = {}
        self._outgoing_sequence = 0
        self._last_persisted_monotonic_ns: int | None = None
        self._snapshot = HubSnapshot(
            version=0,
            hub_epoch=self._hub_epoch,
            generated_at_ns=max(0, self._utc_ns()),
            nodes=(),
            controls=(),
            alerts=(),
        )

    @property
    def hub_epoch(self) -> str:
        """Return the current process epoch, or the pre-start marker."""
        with self._lock:
            return self._hub_epoch

    def start(self) -> None:
        """Start a fresh authority epoch and open only the management channel."""
        with self._lock:
            if self._started:
                return
            self._validate_advertisement()
            if self._store is None:
                self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
                self._store = HubStore(self.config.database_path)
            now_utc_ns = self._checked_now(self._utc_ns(), "utc_ns")
            epoch = self._store.begin_epoch(started_at_ns=now_utc_ns)
            self._store.invalidate_previous_epochs(epoch, at_ns=now_utc_ns)
            self._hub_epoch = epoch
            self._nodes.clear()
            self._handles.clear()
            self._seen_sessions.clear()
            self._last_sequences.clear()
            self._pending_deliveries.clear()
            self._outstanding_commands.clear()
            self._last_persisted_monotonic_ns = None
            self._channel = self._runtime.open_hub(
                self.config.management_endpoint,
                hub_epoch=epoch,
                advertise_endpoint=self.config.advertise_endpoint,
            )
            self._started = True
            self._publish_snapshot_locked()

    def stop(self) -> None:
        """Close the Hub channel and discard all process-local liveness."""
        with self._lock:
            channel = self._channel
            self._channel = None
            self._started = False
            self._nodes.clear()
            self._handles.clear()
            self._seen_sessions.clear()
            self._last_sequences.clear()
            self._pending_deliveries.clear()
            self._outstanding_commands.clear()
            self._publish_snapshot_locked()
        if channel is not None:
            channel.close()

    def register(
        self,
        descriptor: NodeDescriptor,
        *,
        peer_id: str | None = None,
        peer_host: str | None = None,
        received_monotonic_ns: int | None = None,
    ) -> NodeDescriptor:
        """Register a current Node session using a Hub-received liveness timestamp."""
        with self._lock:
            self._require_started()
            if not isinstance(descriptor, NodeDescriptor):
                raise TypeError("descriptor must be a NodeDescriptor")
            if descriptor.protocol_version != PROTOCOL_VERSION:
                raise IncompatibleNode("Node protocol version does not match Hub protocol")
            if peer_id is not None and peer_id != descriptor.node_id:
                raise IncompatibleNode("management peer identity does not match descriptor")
            descriptor = self._validated_descriptor(descriptor, peer_host=peer_host)
            received_ns = self._received_now(received_monotonic_ns)
            previous = self._nodes.get(descriptor.node_id)
            if previous is not None and previous.descriptor.role is not descriptor.role:
                raise IncompatibleNode("stable Node identity cannot change role")
            seen_sessions = self._seen_sessions.setdefault(descriptor.node_id, set())
            if (
                descriptor.session_id in seen_sessions
                and previous is not None
                and previous.descriptor.session_id != descriptor.session_id
            ):
                raise ControlConflict("registration uses a stale session")
            if previous is not None and previous.descriptor.session_id != descriptor.session_id:
                self._revoke_session_handles_locked(
                    previous.descriptor.node_id,
                    previous.descriptor.session_id,
                    reason="node session changed",
                )
            seen_sessions.add(descriptor.session_id)
            self._nodes[descriptor.node_id] = _LiveNode(
                descriptor=descriptor,
                last_seen_monotonic_ns=received_ns,
            )
            self._store_required().upsert_node(descriptor, seen_at_ns=self._utc_ns())
            self._publish_snapshot_locked()
            return descriptor

    def receive_report(
        self,
        report: NodeReport,
        *,
        received_monotonic_ns: int | None = None,
        correlation_id: str | None = None,
        _lifecycle_ack: bool = True,
    ) -> None:
        """Accept one current-session status report and correlate observed control."""
        with self._lock:
            self._require_started()
            if not isinstance(report, NodeReport):
                raise TypeError("report must be a NodeReport")
            node = self._require_current_node(report.node_id, report.session_id)
            self._validate_report_role(node.descriptor.role, report)
            if (report.handle_id is None) != (report.fencing_token is None):
                raise ControlConflict("report Handle identity is incomplete")
            live = None
            if report.handle_id is not None:
                live = self._require_report_handle(node, report)
            node.online = True
            node.last_seen_monotonic_ns = self._received_now(received_monotonic_ns)
            node.report = report
            if live is not None and _lifecycle_ack:
                self._apply_observed_lifecycle_locked(
                    live,
                    node.descriptor.role,
                    report,
                    correlation_id=correlation_id,
                )
            self._apply_fault_or_safety_locked(node, report, correlation_id=correlation_id)
            self._publish_snapshot_locked()

    def assign(
        self,
        robot: str,
        controller: str,
        *,
        control_mode: str = "teleop",
        actor: str | None = None,
    ) -> ControlHandle:
        """Mint one exclusive, compatible, expiring Control Handle."""
        self._validate_actor(actor)
        with self._lock:
            self._require_started()
            robot_node = self._require_schedulable_robot(robot)
            controller_node = self._require_online_controller(controller)
            schema = self._select_schema(robot_node, controller_node, control_mode)
            now_utc_ns = self._utc_ns()
            try:
                handle = self._store_required().create_assignment(
                    robot_node.descriptor,
                    controller_node.descriptor,
                    now_ns=now_utc_ns,
                    ttl_ns=self._timing.handle_ttl_ns,
                    action_schema=schema,
                    control_mode=control_mode,
                )
            except ValueError as error:
                if "exclusive" in str(error):
                    raise ControlConflict(str(error)) from error
                raise IncompatibleNode(str(error)) from error
            record = self._store_required().get_handle(handle.handle_id)
            live = _LiveHandle(record=record, last_renewed_monotonic_ns=self._monotonic_ns())
            self._handles[handle.handle_id] = live
            # Reports received before this assignment cannot describe its
            # Handle identity. Wait for post-grant reports before correlating.
            robot_node.report = None
            controller_node.report = None
            self._audit_locked(
                "assignment_requested",
                actor=actor,
                correlation_id=record.correlation_id,
                details={
                    "controller_id": controller,
                    "handle_id": handle.handle_id,
                    "robot_id": robot,
                },
            )
            self._send_grant_locked(live)
            self._publish_snapshot_locked()
            return handle

    def renew(self, handle: ControlHandle | str, *, actor: str | None = None) -> ControlHandle:
        """Renew current authority without changing Handle identity or fencing."""
        self._validate_actor(actor)
        with self._lock:
            live = self._resolve_handle(handle)
            if not self._can_renew_locked(live):
                raise NodeUnavailable("both current Node sessions must be online to renew")
            renewed = self._renew_locked(live, actor=actor)
            self._publish_snapshot_locked()
            return renewed

    def request_take_over(
        self,
        handle: ControlHandle | str,
        *,
        actor: str | None = None,
    ) -> None:
        """Request take-over through the assigned Controller."""
        self._validate_actor(actor)
        with self._lock:
            live = self._resolve_handle(handle)
            if live.record.state is HandleState.ASSIGNED:
                self._transition_locked(
                    live,
                    HandleState.TAKING_OVER,
                    correlation_id=self._new_correlation("take-over"),
                )
            elif live.record.state is not HandleState.TAKING_OVER:
                raise ControlConflict(f"cannot request take-over from {live.record.state}")
            correlation_id = self._new_correlation("request-take-over")
            self._audit_locked(
                "take_over_requested",
                actor=actor,
                correlation_id=correlation_id,
                details={"handle_id": live.record.handle.handle_id},
            )
            self._send_handle_command_locked(
                live.record.handle.controller_id,
                "take_over",
                live.record.handle,
                correlation_id=correlation_id,
            )
            self._publish_snapshot_locked()

    def request_hand_over(
        self,
        handle: ControlHandle | str,
        *,
        actor: str | None = None,
    ) -> None:
        """Request hand-over through the Controller before Robot release."""
        self._validate_actor(actor)
        with self._lock:
            live = self._resolve_handle(handle)
            correlation_id = self._new_correlation("request-hand-over")
            if live.record.state is HandleState.ACTIVE:
                self._transition_locked(
                    live,
                    HandleState.HANDING_OVER,
                    correlation_id=correlation_id,
                )
            elif live.record.state in {HandleState.ASSIGNED, HandleState.TAKING_OVER}:
                self._transition_locked(
                    live,
                    HandleState.REVOKING,
                    correlation_id=correlation_id,
                    reason="hand-over before activation",
                )
            elif live.record.state not in {HandleState.HANDING_OVER, HandleState.REVOKING}:
                raise ControlConflict(f"cannot request hand-over from {live.record.state}")
            self._audit_locked(
                "hand_over_requested",
                actor=actor,
                correlation_id=correlation_id,
                details={"handle_id": live.record.handle.handle_id},
            )
            self._send_handle_command_locked(
                live.record.handle.controller_id,
                "hand_over",
                live.record.handle,
                correlation_id=correlation_id,
            )
            self._publish_snapshot_locked()

    def revoke(
        self,
        handle: ControlHandle | str,
        *,
        reason: str,
        actor: str | None = None,
    ) -> None:
        """Begin the mandatory REVOKING phase for a current Handle."""
        self._validate_actor(actor)
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must not be empty")
        with self._lock:
            live = self._resolve_handle(handle)
            if live.record.state is HandleState.REVOKED:
                return
            if live.record.state in TERMINAL_HANDLE_STATES:
                raise ControlConflict(f"cannot revoke terminal Handle {live.record.state}")
            correlation_id = self._new_correlation("revoke")
            if live.record.state is not HandleState.REVOKING:
                self._transition_locked(
                    live,
                    HandleState.REVOKING,
                    correlation_id=correlation_id,
                    reason=reason,
                )
            self._audit_locked(
                "revocation_requested",
                actor=actor,
                correlation_id=correlation_id,
                details={"handle_id": live.record.handle.handle_id, "reason": reason},
            )
            self._route_revocation_locked(live, correlation_id=correlation_id, reason=reason)
            self._publish_snapshot_locked()

    def force_hold(self, robot: str, *, reason: str, actor: str | None = None) -> None:
        """Route an audited force-HOLD command, with or without an assignment."""
        self._validate_actor(actor)
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must not be empty")
        with self._lock:
            node = self._nodes.get(robot)
            if node is None or node.descriptor.role is not NodeRole.ROBOT:
                raise NodeUnavailable(f"unknown Robot {robot}")
            correlation_id = self._new_correlation("force-hold")
            live = self._live_handle_for_node(robot)
            self._force_hold_locked(
                robot,
                reason=reason,
                actor=actor,
                correlation_id=correlation_id,
                live=live,
            )
            self._publish_snapshot_locked()

    def list_nodes(self) -> tuple[NodeSnapshot, ...]:
        """Return immutable live registry entries sorted by stable Node ID."""
        with self._lock:
            return tuple(self._node_snapshot(node) for _, node in sorted(self._nodes.items()))

    def get_snapshot(self, handle_id: str | None = None) -> HubSnapshot | ControlSnapshot:
        """Return the full read model or one Handle's correlated control snapshot."""
        with self._lock:
            if handle_id is None:
                return self._snapshot
            for control in self._snapshot.controls:
                if control.handle_id == handle_id:
                    return control
            raise KeyError(handle_id)

    def watch(self, after_version: int = -1, timeout_s: float | None = None) -> HubSnapshot:
        """Wait until a newer immutable Hub snapshot is available."""
        with self._snapshot_changed:
            self._snapshot_changed.wait_for(
                lambda: self._snapshot.version > after_version,
                timeout=timeout_s,
            )
            return self._snapshot

    def list_history(self, *, limit: int = 200) -> tuple[Mapping[str, Any], ...]:
        """Return newest persisted audit events first."""
        with self._lock:
            return self._store_required().list_history(limit=limit)

    def run_once(self, timeout_s: float = 0.0) -> bool:
        """Receive and dispatch at most one management message."""
        with self._lock:
            self._require_started()
            channel = self._channel_required()
        received = channel.receive(timeout_s=timeout_s)
        if received is None:
            return False
        if not isinstance(received, ReceivedManagement):
            raise TypeError("Runtime must return ReceivedManagement")
        if not isinstance(received.message, ManagementMessage):
            raise TypeError("Hub accepts management messages only")
        self._dispatch(received)
        return True

    def tick(self) -> None:
        """Drive liveness, expiry, renewal, retry, correlation, and diagnostics once."""
        with self._lock:
            self._require_started()
            now_monotonic_ns = self._monotonic_ns()
            now_utc_ns = self._utc_ns()
            offline_after_ns = 3 * self._timing.heartbeat_interval_ns
            newly_offline: list[str] = []
            for node_id, node in self._nodes.items():
                if node.online and now_monotonic_ns - node.last_seen_monotonic_ns >= offline_after_ns:
                    node.online = False
                    newly_offline.append(node_id)
            for node_id in newly_offline:
                self._revoke_node_handles_locked(node_id, reason="heartbeat timeout")

            for live in tuple(self._handles.values()):
                if live.record.state in TERMINAL_HANDLE_STATES:
                    continue
                if now_utc_ns >= live.record.handle.expires_at_ns:
                    self._transition_locked(
                        live,
                        HandleState.EXPIRED,
                        correlation_id=self._new_correlation("expired"),
                        reason="Handle lease expired",
                    )
                    continue
                if (
                    now_monotonic_ns - live.last_renewed_monotonic_ns >= self._timing.renewal_interval_ns
                    and self._can_renew_locked(live)
                ):
                    self._renew_locked(live, actor=None)
                if live.pending_grants:
                    self._send_grant_locked(live, only=tuple(live.pending_grants))

            self._respond_to_mismatches_locked()
            self._publish_snapshot_locked()
            if (
                self._last_persisted_monotonic_ns is None
                or now_monotonic_ns - self._last_persisted_monotonic_ns >= 1_000_000_000
            ):
                self._store_required().save_snapshot(self._snapshot)
                self._last_persisted_monotonic_ns = now_monotonic_ns

    def run(self, *, stop_event: Any) -> None:
        """Run bounded management waits and deterministic service ticks."""
        if not hasattr(stop_event, "is_set"):
            raise TypeError("stop_event must provide is_set()")
        if not self._started:
            self.start()
        wait_s = min(0.1, self._timing.heartbeat_interval_ns / 2_000_000_000)
        try:
            while not stop_event.is_set():
                # A rejected or stale peer message is a protocol event, not
                # a reason to terminate the authoritative service loop.
                with suppress(ControlError):
                    self.run_once(timeout_s=wait_s)
                self.tick()
        finally:
            self.stop()

    def _dispatch(self, received: ReceivedManagement) -> None:
        message = received.message
        if message.protocol_version != PROTOCOL_VERSION:
            raise IncompatibleNode("management protocol version does not match Hub")
        if received.peer_id != message.sender_id:
            raise ControlConflict("management peer identity does not match sender")
        if message.kind == "register":
            with self._lock:
                key = (message.sender_id, message.sender_session_id)
                previous_sequence = self._last_sequences.get(key)
                current = self._nodes.get(message.sender_id)
                if (
                    previous_sequence is not None
                    and message.sequence <= previous_sequence
                    and current is not None
                    and current.descriptor.session_id == message.sender_session_id
                ):
                    self._send_message_locked(
                        message.sender_id,
                        "registered",
                        message.correlation_id,
                        {"hub_epoch": self._hub_epoch},
                    )
                    self._publish_snapshot_locked()
                    return
            descriptor = self._descriptor_from_body(message.body)
            if (descriptor.node_id, descriptor.session_id) != (
                message.sender_id,
                message.sender_session_id,
            ):
                raise ControlConflict("registration sender does not match descriptor")
            self.register(
                descriptor,
                peer_id=received.peer_id,
                peer_host=received.peer_host,
                received_monotonic_ns=received.received_monotonic_ns,
            )
            with self._lock:
                self._last_sequences[(message.sender_id, message.sender_session_id)] = message.sequence
                self._send_message_locked(
                    message.sender_id,
                    "registered",
                    message.correlation_id,
                    {"hub_epoch": self._hub_epoch},
                )
                self._publish_snapshot_locked()
            return

        with self._lock:
            self._require_current_node(message.sender_id, message.sender_session_id)
            if not self._accept_sequence_locked(message):
                return
            if message.kind == "heartbeat":
                node = self._nodes[message.sender_id]
                node.online = True
                node.last_seen_monotonic_ns = self._received_now(received.received_monotonic_ns)
                self._send_message_locked(
                    message.sender_id,
                    "heartbeat_ack",
                    message.correlation_id,
                    {"hub_epoch": self._hub_epoch},
                )
                self._publish_snapshot_locked()
                return

        if message.kind == "status":
            report = self._report_from_body(message.body)
            self._validate_report_sender(message, report)
            self.receive_report(
                report,
                received_monotonic_ns=received.received_monotonic_ns,
                correlation_id=message.correlation_id,
                _lifecycle_ack=False,
            )
            return
        with self._lock:
            if message.kind == "take_over_requested":
                self._on_take_over_requested_locked(message)
            elif message.kind == "robot_ready":
                self._on_robot_ready_locked(message, received.received_monotonic_ns)
            elif message.kind == "controller_streaming":
                self._on_controller_streaming_locked(message, received.received_monotonic_ns)
            elif message.kind == "hand_over_requested":
                self._on_hand_over_requested_locked(message)
            elif message.kind == "robot_holding":
                self._on_robot_holding_locked(message, received.received_monotonic_ns)
            elif message.kind == "controller_released":
                self._on_controller_released_locked(message, received.received_monotonic_ns)
            elif message.kind == "fault":
                self._on_fault_locked(message, received.received_monotonic_ns)
            elif message.kind.endswith("_ack"):
                self._on_command_ack_locked(message, received.received_monotonic_ns)
            else:
                raise ControlConflict(f"unsupported management kind {message.kind}")
            self._publish_snapshot_locked()

    def _on_take_over_requested_locked(self, message: ManagementMessage) -> None:
        live = self._live_from_message(message)
        if message.sender_id != live.record.handle.controller_id:
            raise ControlConflict("take-over request must come from assigned Controller")
        if live.record.state is HandleState.ASSIGNED:
            self._transition_locked(
                live,
                HandleState.TAKING_OVER,
                correlation_id=message.correlation_id,
            )
        elif live.record.state is not HandleState.TAKING_OVER:
            raise ControlConflict(f"take-over request is stale for {live.record.state}")
        self._send_handle_command_locked(
            live.record.handle.robot_id,
            "take_over",
            live.record.handle,
            correlation_id=message.correlation_id,
        )

    def _on_robot_ready_locked(self, message: ManagementMessage, received_ns: int) -> None:
        live = self._live_from_message(message)
        if message.sender_id != live.record.handle.robot_id:
            raise ControlConflict("robot_ready must come from assigned Robot")
        self._touch_node_locked(message.sender_id, received_ns)
        live.robot_ready = True
        if live.record.state is HandleState.ASSIGNED:
            self._transition_locked(
                live,
                HandleState.TAKING_OVER,
                correlation_id=message.correlation_id,
            )
        self._activate_if_acknowledged_locked(live, message.correlation_id)
        self._send_handle_command_locked(
            live.record.handle.controller_id,
            "robot_ready",
            live.record.handle,
            correlation_id=message.correlation_id,
        )

    def _on_controller_streaming_locked(self, message: ManagementMessage, received_ns: int) -> None:
        live = self._live_from_message(message)
        if message.sender_id != live.record.handle.controller_id:
            raise ControlConflict("controller_streaming must come from assigned Controller")
        self._touch_node_locked(message.sender_id, received_ns)
        live.controller_streaming = True
        self._activate_if_acknowledged_locked(live, message.correlation_id)

    def _on_hand_over_requested_locked(self, message: ManagementMessage) -> None:
        live = self._live_from_message(message)
        if message.sender_id != live.record.handle.controller_id:
            raise ControlConflict("hand-over request must come from assigned Controller")
        if live.record.state is HandleState.ACTIVE:
            self._transition_locked(
                live,
                HandleState.HANDING_OVER,
                correlation_id=message.correlation_id,
            )
        elif live.record.state not in {HandleState.HANDING_OVER, HandleState.REVOKING}:
            raise ControlConflict(f"hand-over request is stale for {live.record.state}")
        self._send_handle_command_locked(
            live.record.handle.robot_id,
            "hand_over",
            live.record.handle,
            correlation_id=message.correlation_id,
        )

    def _on_robot_holding_locked(self, message: ManagementMessage, received_ns: int) -> None:
        live = self._live_from_message(message)
        if message.sender_id != live.record.handle.robot_id:
            raise ControlConflict("robot_holding must come from assigned Robot")
        self._touch_node_locked(message.sender_id, received_ns)
        self._finish_from_robot_hold_locked(live, message.correlation_id)

    def _on_controller_released_locked(self, message: ManagementMessage, received_ns: int) -> None:
        live = self._live_from_message(message)
        if message.sender_id != live.record.handle.controller_id:
            raise ControlConflict("controller_released must come from assigned Controller")
        self._touch_node_locked(message.sender_id, received_ns)
        live.controller_released = True

    def _on_fault_locked(self, message: ManagementMessage, received_ns: int) -> None:
        if "report" in message.body or "runtime_state" in message.body:
            report = self._report_from_body(message.body)
            self._validate_report_sender(message, report)
            self.receive_report(
                report,
                received_monotonic_ns=received_ns,
                correlation_id=message.correlation_id,
                _lifecycle_ack=False,
            )
            return
        live = self._live_from_message(message)
        if live.record.state not in TERMINAL_HANDLE_STATES:
            self._transition_locked(
                live,
                HandleState.FAULT,
                correlation_id=message.correlation_id,
                reason=str(message.body.get("reason", "Node fault")),
            )

    def _on_command_ack_locked(self, message: ManagementMessage, received_ns: int) -> None:
        key = (message.correlation_id, message.sender_id)
        outstanding = self._outstanding_commands.get(key)
        if outstanding is None:
            raise ControlConflict("ack does not match an outstanding command")
        node = self._require_current_node(message.sender_id, message.sender_session_id)
        if message.kind != outstanding.expected_ack_kind:
            raise ControlConflict("ack kind does not match outstanding command")
        if message.sender_session_id != outstanding.node_session_id:
            raise ControlConflict("ack session does not match outstanding command")
        if node.descriptor.role is not outstanding.node_role:
            raise ControlConflict("ack role does not match outstanding command")
        body = message.body.get("handle", message.body)
        if not isinstance(body, Mapping):
            raise ControlConflict("ack Handle body is invalid")
        if body.get("handle_id") != outstanding.handle_id:
            raise ControlConflict("ack Handle ID does not match outstanding command")
        if body.get("hub_epoch") != outstanding.hub_epoch:
            raise ControlConflict("ack epoch does not match outstanding command")
        if body.get("fencing_token") != outstanding.fencing_token:
            raise ControlConflict("ack fencing does not match outstanding command")
        self._audit_locked(
            "command_acknowledged",
            actor=None,
            correlation_id=message.correlation_id,
            details={
                "handle_id": outstanding.handle_id,
                "kind": message.kind,
                "node_id": message.sender_id,
            },
        )
        if message.kind == "grant_ack":
            # Any unbound status received before this ACK predates the
            # node's acceptance of the new Handle.
            node.report = None
        self._outstanding_commands.pop(key)
        self._touch_node_locked(message.sender_id, received_ns)

    def _apply_observed_lifecycle_locked(
        self,
        live: _LiveHandle,
        role: NodeRole,
        report: NodeReport,
        *,
        correlation_id: str | None,
    ) -> None:
        correlation = correlation_id or self._new_correlation("status")
        if role is NodeRole.ROBOT:
            if report.robot_control_state is RobotControlState.HOLD:
                if live.record.state in {HandleState.HANDING_OVER, HandleState.REVOKING}:
                    self._finish_from_robot_hold_locked(live, correlation)
                    return
                if live.record.state is HandleState.ASSIGNED:
                    live.robot_ready = True
                    self._transition_locked(
                        live,
                        HandleState.TAKING_OVER,
                        correlation_id=correlation,
                    )
            elif report.robot_control_state is RobotControlState.CONTROLLING:
                live.robot_ready = True
        elif report.controller_control_state is ControllerControlState.STREAMING:
            live.controller_streaming = True
        self._activate_if_acknowledged_locked(live, correlation)

    def _apply_fault_or_safety_locked(
        self,
        node: _LiveNode,
        report: NodeReport,
        *,
        correlation_id: str | None,
    ) -> None:
        live = self._live_handle_for_node(node.descriptor.node_id)
        if live is None or live.record.state in TERMINAL_HANDLE_STATES:
            return
        correlation = correlation_id or self._new_correlation("fault-status")
        if report.runtime_state is RuntimeState.FAULT:
            self._transition_locked(
                live,
                HandleState.FAULT,
                correlation_id=correlation,
                reason=report.error or "Node fault",
            )
        elif report.robot_control_state is RobotControlState.SAFETY:
            self._automatic_revoke_locked(
                live,
                source="safety",
                reason="Robot entered SAFETY",
                correlation_id=correlation,
            )

    def _activate_if_acknowledged_locked(self, live: _LiveHandle, correlation_id: str) -> None:
        if live.record.state is HandleState.TAKING_OVER and live.robot_ready and live.controller_streaming:
            self._transition_locked(
                live,
                HandleState.ACTIVE,
                correlation_id=correlation_id,
            )

    def _finish_from_robot_hold_locked(self, live: _LiveHandle, correlation_id: str) -> None:
        if live.record.state is HandleState.HANDING_OVER:
            self._transition_locked(
                live,
                HandleState.RELEASED,
                correlation_id=correlation_id,
            )
        elif live.record.state is HandleState.REVOKING:
            self._transition_locked(
                live,
                HandleState.REVOKED,
                correlation_id=correlation_id,
            )

    def _respond_to_mismatches_locked(self) -> None:
        if not self.config.auto_revoke_mismatches:
            return
        for live in tuple(self._handles.values()):
            if live.record.state in TERMINAL_HANDLE_STATES or live.record.state in {
                HandleState.HANDING_OVER,
                HandleState.REVOKING,
            }:
                continue
            if live.record.state in {HandleState.ASSIGNED, HandleState.TAKING_OVER} and any(
                command.handle_id == live.record.handle.handle_id
                and command.expected_ack_kind == "grant_ack"
                for command in self._outstanding_commands.values()
            ):
                continue
            control = self._correlate_live(live)
            if not control.mismatch_codes:
                continue
            reason = f"live mismatch: {control.mismatch_codes[0]}"
            self._automatic_revoke_locked(
                live,
                source="mismatch",
                reason=reason,
                correlation_id=self._new_correlation("mismatch"),
            )

    def _renew_locked(self, live: _LiveHandle, *, actor: str | None) -> ControlHandle:
        now_utc_ns = self._utc_ns()
        renewed = self._store_required().renew_handle(
            live.record.handle.handle_id,
            expires_at_ns=now_utc_ns + self._timing.handle_ttl_ns,
            at_ns=now_utc_ns,
        )
        live.record = self._store_required().get_handle(renewed.handle_id)
        live.last_renewed_monotonic_ns = self._monotonic_ns()
        correlation_id = self._new_correlation("renew")
        self._audit_locked(
            "renewal_requested",
            actor=actor,
            correlation_id=correlation_id,
            details={"expires_at_ns": renewed.expires_at_ns, "handle_id": renewed.handle_id},
        )
        for node_id in (renewed.robot_id, renewed.controller_id):
            self._send_handle_command_locked(
                node_id,
                "renewal",
                renewed,
                correlation_id=correlation_id,
            )
        return renewed

    def _can_renew_locked(self, live: _LiveHandle) -> bool:
        handle = live.record.handle
        if live.record.state in TERMINAL_HANDLE_STATES or live.record.state is HandleState.REVOKING:
            return False
        robot = self._nodes.get(handle.robot_id)
        controller = self._nodes.get(handle.controller_id)
        return bool(
            robot is not None
            and controller is not None
            and robot.online
            and controller.online
            and robot.descriptor.session_id == handle.robot_session_id
            and controller.descriptor.session_id == handle.controller_session_id
        )

    def _send_grant_locked(self, live: _LiveHandle, *, only: tuple[str, ...] | None = None) -> None:
        handle = live.record.handle
        targets = only or (handle.robot_id, handle.controller_id)
        for node_id in targets:
            sent = self._send_handle_command_locked(
                node_id,
                "grant",
                handle,
                correlation_id=live.record.correlation_id,
            )
            if sent:
                live.pending_grants.discard(node_id)
            else:
                live.pending_grants.add(node_id)

    def _force_hold_intent_locked(
        self,
        robot_id: str,
        *,
        reason: str,
        actor: str | None,
        correlation_id: str,
        live: _LiveHandle | None,
        source: str = "public",
    ) -> tuple[dict[str, Any], ControlHandle | None]:
        handle = live.record.handle if live is not None else None
        self._audit_locked(
            "force_hold_requested",
            actor=actor,
            correlation_id=correlation_id,
            details={
                "handle_id": handle.handle_id if handle is not None else None,
                "reason": reason,
                "robot_id": robot_id,
                "source": source,
            },
        )
        body: dict[str, Any] = {"reason": reason}
        if handle is not None:
            body["handle"] = self._wire_dataclass(handle)
        return body, handle

    def _send_force_hold_intent_locked(
        self,
        robot_id: str,
        *,
        correlation_id: str,
        body: Mapping[str, Any],
        handle: ControlHandle | None,
    ) -> bool:
        return self._send_message_locked(
            robot_id,
            "force_hold",
            correlation_id,
            body,
            handle=handle,
            expect_ack=handle is not None,
        )

    def _force_hold_locked(
        self,
        robot_id: str,
        *,
        reason: str,
        actor: str | None,
        correlation_id: str,
        live: _LiveHandle | None,
        source: str = "public",
    ) -> bool:
        body, handle = self._force_hold_intent_locked(
            robot_id,
            reason=reason,
            actor=actor,
            correlation_id=correlation_id,
            live=live,
            source=source,
        )
        return self._send_force_hold_intent_locked(
            robot_id,
            correlation_id=correlation_id,
            body=body,
            handle=handle,
        )

    def _route_revocation_locked(
        self,
        live: _LiveHandle,
        *,
        correlation_id: str,
        reason: str,
    ) -> tuple[bool, bool]:
        handle = live.record.handle
        controller_sent = self._send_handle_command_locked(
            handle.controller_id,
            "revoke",
            handle,
            correlation_id=correlation_id,
            extra={"reason": reason},
        )
        robot_sent = self._send_handle_command_locked(
            handle.robot_id,
            "revoke",
            handle,
            correlation_id=correlation_id,
            extra={"reason": reason},
        )
        return controller_sent, robot_sent

    def _automatic_revoke_locked(
        self,
        live: _LiveHandle,
        *,
        source: str,
        reason: str,
        correlation_id: str,
    ) -> bool:
        if live.record.state in TERMINAL_HANDLE_STATES or live.record.state is HandleState.REVOKING:
            return False
        handle = live.record.handle
        force_hold_correlation_id = self._new_correlation("automatic-force-hold")
        force_body, force_handle = self._force_hold_intent_locked(
            handle.robot_id,
            reason=reason,
            actor=None,
            correlation_id=force_hold_correlation_id,
            live=live,
            source=source,
        )
        self._audit_locked(
            "automatic_revocation_requested",
            actor=None,
            correlation_id=correlation_id,
            details={
                "handle_id": handle.handle_id,
                "reason": reason,
                "source": source,
            },
        )
        self._transition_locked(
            live,
            HandleState.REVOKING,
            correlation_id=correlation_id,
            reason=reason,
        )
        self._route_revocation_locked(live, correlation_id=correlation_id, reason=reason)
        self._send_force_hold_intent_locked(
            handle.robot_id,
            correlation_id=force_hold_correlation_id,
            body=force_body,
            handle=force_handle,
        )
        return True

    def _send_handle_command_locked(
        self,
        node_id: str,
        kind: str,
        handle: ControlHandle,
        *,
        correlation_id: str,
        extra: Mapping[str, Any] | None = None,
    ) -> bool:
        body: dict[str, Any] = {"handle": self._wire_dataclass(handle), "hub_epoch": self._hub_epoch}
        if extra is not None:
            body.update(extra)
        return self._send_message_locked(
            node_id,
            kind,
            correlation_id,
            body,
            handle=handle,
            expect_ack=True,
        )

    def _send_message_locked(
        self,
        node_id: str,
        kind: str,
        correlation_id: str,
        body: Mapping[str, Any],
        *,
        handle: ControlHandle | None = None,
        expect_ack: bool = False,
    ) -> bool:
        message = ManagementMessage(
            protocol_version=PROTOCOL_VERSION,
            kind=kind,
            correlation_id=correlation_id,
            sender_id=_HUB_ID,
            sender_session_id=self._hub_epoch,
            sequence=self._outgoing_sequence,
            sent_at_ns=self._utc_ns(),
            body=body,
        )
        self._outgoing_sequence += 1
        sent = self._channel_required().send(node_id, message)
        delivery_key = (correlation_id, node_id, kind)
        if not sent:
            pending = _PendingDelivery(
                node_id=node_id,
                kind=kind,
                correlation_id=correlation_id,
                handle_id=handle.handle_id if handle is not None else None,
            )
            self._pending_deliveries[delivery_key] = pending
            self._outstanding_commands.pop((correlation_id, node_id), None)
            self._audit_locked(
                "management_delivery_failed",
                actor=None,
                correlation_id=correlation_id,
                details={
                    "handle_id": pending.handle_id,
                    "kind": kind,
                    "node_id": node_id,
                },
            )
            return False
        self._pending_deliveries.pop(delivery_key, None)
        if expect_ack:
            if handle is None:
                raise RuntimeError("acknowledged command requires a Handle")
            node = self._nodes.get(node_id)
            if node is None:
                raise NodeUnavailable(f"Node {node_id} is not registered")
            self._outstanding_commands[(correlation_id, node_id)] = _OutstandingCommand(
                node_id=node_id,
                node_session_id=node.descriptor.session_id,
                node_role=node.descriptor.role,
                expected_ack_kind=f"{kind}_ack",
                correlation_id=correlation_id,
                handle_id=handle.handle_id,
                hub_epoch=handle.hub_epoch,
                fencing_token=handle.fencing_token,
            )
        return True

    def _transition_locked(
        self,
        live: _LiveHandle,
        target: HandleState,
        *,
        correlation_id: str,
        reason: str | None = None,
    ) -> None:
        if live.record.state is target:
            return
        try:
            live.record = self._store_required().transition(
                live.record.handle.handle_id,
                target,
                live.record.transition_sequence + 1,
                correlation_id,
                self._utc_ns(),
                reason,
            )
        except InvalidHandleTransition as error:
            raise ControlConflict(str(error)) from error

    def _resolve_handle(self, reference: ControlHandle | str) -> _LiveHandle:
        self._require_started()
        if isinstance(reference, ControlHandle):
            handle_id = reference.handle_id
        elif isinstance(reference, str) and reference:
            handle_id = reference
        else:
            raise TypeError("handle must be a ControlHandle or non-empty Handle ID")
        live = self._handles.get(handle_id)
        if live is None:
            raise ControlConflict(f"unknown or stale Handle {handle_id}")
        if isinstance(reference, ControlHandle):
            current = live.record.handle
            if any(getattr(reference, name) != getattr(current, name) for name in _HANDLE_IDENTITY_FIELDS):
                raise ControlConflict("Handle identity, epoch, session, or fencing is stale")
        return live

    def _live_from_message(self, message: ManagementMessage) -> _LiveHandle:
        body = message.body.get("handle", message.body)
        if not isinstance(body, Mapping):
            raise ControlConflict("management Handle body is invalid")
        handle_id = body.get("handle_id")
        fencing_token = body.get("fencing_token")
        hub_epoch = body.get("hub_epoch", self._hub_epoch)
        if not isinstance(handle_id, str) or not handle_id:
            raise ControlConflict("management message requires handle_id")
        live = self._handles.get(handle_id)
        if live is None:
            raise ControlConflict("management message references stale Handle")
        if live.record.state in TERMINAL_HANDLE_STATES:
            raise ControlConflict("management message references terminal Handle")
        if hub_epoch != self._hub_epoch or fencing_token != live.record.handle.fencing_token:
            raise ControlConflict("management message has stale epoch or fencing")
        self._validate_message_handle_sessions(body, live.record.handle)
        return live

    @staticmethod
    def _validate_message_handle_sessions(body: Mapping[str, Any], handle: ControlHandle) -> None:
        for name in ("robot_id", "robot_session_id", "controller_id", "controller_session_id"):
            if name in body and body[name] != getattr(handle, name):
                raise ControlConflict("management message has stale Handle session")

    def _require_report_handle(self, node: _LiveNode, report: NodeReport) -> _LiveHandle:
        live = self._handles.get(report.handle_id or "")
        if live is None:
            raise ControlConflict("report references unknown or stale Handle")
        handle = live.record.handle
        if handle.hub_epoch != self._hub_epoch or report.fencing_token != handle.fencing_token:
            raise ControlConflict("report has stale epoch or fencing")
        expected = handle.robot_id if node.descriptor.role is NodeRole.ROBOT else handle.controller_id
        expected_session = (
            handle.robot_session_id
            if node.descriptor.role is NodeRole.ROBOT
            else handle.controller_session_id
        )
        if node.descriptor.node_id != expected or node.descriptor.session_id != expected_session:
            raise ControlConflict("report session does not match Handle authority")
        return live

    def _require_schedulable_robot(self, node_id: str) -> _LiveNode:
        node = self._nodes.get(node_id)
        if node is None or node.descriptor.role is not NodeRole.ROBOT:
            raise NodeUnavailable(f"unknown Robot {node_id}")
        if not node.online:
            raise NodeUnavailable(f"Robot {node_id} is offline")
        if not node.descriptor.administratively_enabled:
            raise NodeUnavailable(f"Robot {node_id} is administratively disabled")
        if node.report is not None:
            if node.report.robot_control_state is RobotControlState.SAFETY:
                raise NodeUnavailable(f"Robot {node_id} is in SAFETY")
            if node.report.runtime_state is RuntimeState.FAULT:
                raise NodeUnavailable(f"Robot {node_id} is in FAULT")
            if node.report.runtime_state is RuntimeState.DEGRADED and node.report.error is not None:
                raise NodeUnavailable(f"Robot {node_id} has a degraded control error")
        if self._live_handle_for_node(node_id) is not None:
            raise ControlConflict(f"Robot {node_id} already has a non-terminal Handle")
        return node

    def _require_online_controller(self, node_id: str) -> _LiveNode:
        node = self._nodes.get(node_id)
        if node is None or node.descriptor.role is not NodeRole.CONTROLLER:
            raise NodeUnavailable(f"unknown Controller {node_id}")
        if not node.online:
            raise NodeUnavailable(f"Controller {node_id} is offline")
        if not node.descriptor.administratively_enabled:
            raise NodeUnavailable(f"Controller {node_id} is administratively disabled")
        if node.report is not None and node.report.runtime_state in {
            RuntimeState.FAULT,
            RuntimeState.STOPPED,
        }:
            raise NodeUnavailable(f"Controller {node_id} is unavailable")
        if self._live_handle_for_node(node_id) is not None:
            raise ControlConflict(f"Controller {node_id} already has a non-terminal Handle")
        return node

    @staticmethod
    def _select_schema(robot: _LiveNode, controller: _LiveNode, control_mode: str) -> str:
        if (
            control_mode not in robot.descriptor.control_modes
            or control_mode not in controller.descriptor.control_modes
        ):
            raise IncompatibleNode("control mode is not supported by both Nodes")
        for schema in controller.descriptor.action_schemas:
            if schema in robot.descriptor.action_schemas:
                return schema
        raise IncompatibleNode("Nodes have no shared action schema")

    def _live_handle_for_node(self, node_id: str) -> _LiveHandle | None:
        for live in self._handles.values():
            handle = live.record.handle
            if live.record.state not in TERMINAL_HANDLE_STATES and node_id in {
                handle.robot_id,
                handle.controller_id,
            }:
                return live
        return None

    def _revoke_session_handles_locked(self, node_id: str, session_id: str, *, reason: str) -> None:
        for live in tuple(self._handles.values()):
            handle = live.record.handle
            matches = (handle.robot_id, handle.robot_session_id) == (node_id, session_id) or (
                handle.controller_id,
                handle.controller_session_id,
            ) == (node_id, session_id)
            if (
                matches
                and live.record.state not in TERMINAL_HANDLE_STATES
                and live.record.state is not HandleState.REVOKING
            ):
                self._automatic_revoke_locked(
                    live,
                    source="session_replacement",
                    reason=reason,
                    correlation_id=self._new_correlation("session-revoke"),
                )

    def _revoke_node_handles_locked(self, node_id: str, *, reason: str) -> None:
        live = self._live_handle_for_node(node_id)
        if live is None:
            return
        self._automatic_revoke_locked(
            live,
            source="offline",
            reason=reason,
            correlation_id=self._new_correlation("offline-revoke"),
        )

    def _publish_snapshot_locked(self) -> None:
        controls = tuple(self._correlate_live(live) for _, live in sorted(self._handles.items()))
        mismatch_alerts = tuple(
            {
                "code": code,
                "handle_id": control.handle_id,
                "severity": "error",
            }
            for control in controls
            for code in control.mismatch_codes
        )
        delivery_alerts = tuple(pending.alert() for _, pending in sorted(self._pending_deliveries.items()))
        self._snapshot = HubSnapshot(
            version=self._snapshot.version + 1,
            hub_epoch=self._hub_epoch,
            generated_at_ns=max(0, self._utc_ns()),
            nodes=tuple(self._node_snapshot(node) for _, node in sorted(self._nodes.items())),
            controls=controls,
            alerts=mismatch_alerts + delivery_alerts,
        )
        self._snapshot_changed.notify_all()

    def _correlate_live(self, live: _LiveHandle):
        handle = live.record.handle
        robot = self._nodes.get(handle.robot_id)
        controller = self._nodes.get(handle.controller_id)
        robot_report = robot.report if robot is not None and robot.online else None
        controller_report = controller.report if controller is not None and controller.online else None
        return correlate_control(
            live.record,
            robot_report,
            controller_report,
            now_ns=max(0, self._utc_ns()),
        )

    @staticmethod
    def _node_snapshot(node: _LiveNode) -> NodeSnapshot:
        return NodeSnapshot(
            descriptor=node.descriptor,
            online=node.online,
            last_seen_ns=node.last_seen_monotonic_ns,
            report=node.report,
        )

    def _validated_descriptor(self, descriptor: NodeDescriptor, *, peer_host: str | None) -> NodeDescriptor:
        if not descriptor.action_schemas:
            raise IncompatibleNode("Node must declare at least one action schema")
        if not descriptor.control_modes:
            raise IncompatibleNode("Node must declare at least one control mode")
        if descriptor.role is NodeRole.ROBOT:
            if descriptor.action_endpoint is not None:
                raise IncompatibleNode("Robot descriptor cannot advertise a Controller endpoint")
            return descriptor
        if descriptor.observation_features or descriptor.action_features:
            raise IncompatibleNode("Controller descriptor cannot carry Robot feature metadata")
        endpoint = descriptor.action_endpoint
        if endpoint is None:
            raise IncompatibleNode("Controller descriptor requires an action endpoint")
        endpoint = self._resolve_consumer_endpoint(endpoint, peer_host=peer_host)
        return replace(descriptor, action_endpoint=endpoint)

    @staticmethod
    def _resolve_consumer_endpoint(endpoint: str, *, peer_host: str | None) -> str:
        parsed = urlsplit(endpoint)
        if parsed.hostname not in _WILDCARD_HOSTS:
            return endpoint
        if not peer_host:
            raise IncompatibleNode("wildcard Controller endpoint requires a management peer host")
        if parsed.port is None:
            raise IncompatibleNode("wildcard Controller endpoint requires a port")
        replacement = f"{peer_host}:{parsed.port}"
        return urlunsplit((parsed.scheme, replacement, parsed.path, parsed.query, parsed.fragment))

    def _validate_advertisement(self) -> None:
        advertised = self.config.advertise_endpoint
        if advertised is not None and urlsplit(advertised).hostname in _WILDCARD_HOSTS:
            raise ValueError("advertise_endpoint must be reachable and non-wildcard")

    @staticmethod
    def _validate_report_role(role: NodeRole, report: NodeReport) -> None:
        if role is NodeRole.ROBOT:
            if report.robot_control_state is None or report.controller_control_state is not None:
                raise IncompatibleNode("Robot report is inconsistent with registered role")
        elif report.controller_control_state is None or report.robot_control_state is not None:
            raise IncompatibleNode("Controller report is inconsistent with registered role")

    @staticmethod
    def _validate_report_sender(message: ManagementMessage, report: NodeReport) -> None:
        if (report.node_id, report.session_id) != (
            message.sender_id,
            message.sender_session_id,
        ):
            raise ControlConflict("report identity does not match authenticated sender")

    def _descriptor_from_body(self, body: Mapping[str, Any]) -> NodeDescriptor:
        value = body.get("descriptor", body)
        if not isinstance(value, Mapping):
            raise IncompatibleNode("registration descriptor body is invalid")
        values = dict(value)
        try:
            values["role"] = NodeRole(values["role"])
            for name in ("capabilities", "action_schemas", "control_modes"):
                values[name] = tuple(values[name])
            return NodeDescriptor(**values)
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleNode("registration descriptor body is invalid") from error

    def _report_from_body(self, body: Mapping[str, Any]) -> NodeReport:
        value = body.get("report", body)
        if not isinstance(value, Mapping):
            raise IncompatibleNode("status report body is invalid")
        values = dict(value)
        try:
            values["runtime_state"] = RuntimeState(values["runtime_state"])
            if values.get("robot_control_state") is not None:
                values["robot_control_state"] = RobotControlState(values["robot_control_state"])
            if values.get("controller_control_state") is not None:
                values["controller_control_state"] = ControllerControlState(
                    values["controller_control_state"]
                )
            return NodeReport(**values)
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleNode("status report body is invalid") from error

    @staticmethod
    def _wire_dataclass(value: object) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for data_field in fields(value):  # type: ignore[arg-type]
            item = getattr(value, data_field.name)
            result[data_field.name] = item.value if isinstance(item, Enum) else item
        return result

    def _accept_sequence_locked(self, message: ManagementMessage) -> bool:
        key = (message.sender_id, message.sender_session_id)
        previous = self._last_sequences.get(key)
        if previous is not None and message.sequence <= previous:
            return False
        self._last_sequences[key] = message.sequence
        return True

    def _touch_node_locked(self, node_id: str, received_ns: int) -> None:
        node = self._nodes[node_id]
        node.online = True
        node.last_seen_monotonic_ns = self._received_now(received_ns)

    def _require_current_node(self, node_id: str, session_id: str) -> _LiveNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise NodeUnavailable(f"Node {node_id} is not registered")
        if node.descriptor.session_id != session_id:
            raise ControlConflict("message session is stale")
        return node

    def _audit_locked(
        self,
        event: str,
        *,
        actor: str | None,
        correlation_id: str,
        details: Mapping[str, Any],
    ) -> None:
        self._store_required().append_audit(
            event=event,
            at_ns=self._utc_ns(),
            actor=actor,
            correlation_id=correlation_id,
            details=details,
        )

    @staticmethod
    def _validate_actor(actor: str | None) -> None:
        if actor is not None and (not isinstance(actor, str) or not actor):
            raise ValueError("actor must be non-empty when present")

    @staticmethod
    def _new_correlation(prefix: str) -> str:
        return f"{prefix}:{uuid.uuid4()}"

    def _received_now(self, received_monotonic_ns: int | None) -> int:
        value = self._monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        return self._checked_now(value, "received_monotonic_ns")

    @staticmethod
    def _checked_now(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Hub is not started")

    def _store_required(self) -> HubStore:
        if self._store is None:
            raise RuntimeError("Hub store is not open")
        return self._store

    def _channel_required(self) -> HubChannel:
        if self._channel is None:
            raise RuntimeError("Hub management channel is not open")
        return self._channel
