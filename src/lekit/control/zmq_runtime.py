"""Bounded ZeroMQ action transport for Control Hub nodes."""

from __future__ import annotations

import math
import queue
import socket
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from urllib.parse import urlparse

import zmq
from zmq.utils.monitor import parse_monitor_message

from .codec import decode_action_envelope, decode_management, encode_action_envelope, encode_management
from .discovery import (
    CooperativeDiscoveryListener,
    DiscoveryBeacon,
    HubBeaconListener,
    HubBeaconPublisher,
    validate_management_endpoint,
)
from .model import PROTOCOL_VERSION, ActionEnvelope, ManagementMessage
from .runtime import ReceivedAction, ReceivedManagement

_CONTEXT_OWNERS_LOCK = threading.RLock()
_CONTEXT_OWNERS: dict[
    int,
    tuple[weakref.ReferenceType[zmq.Context], weakref.ReferenceType[ZmqContextOwner]],
] = {}


def _associate_owned_context(context: zmq.Context, owner: ZmqContextOwner) -> None:
    context_id = id(context)

    def remove(released_context: weakref.ReferenceType[zmq.Context]) -> None:
        with _CONTEXT_OWNERS_LOCK:
            entry = _CONTEXT_OWNERS.get(context_id)
            if entry is not None and entry[0] is released_context:
                _CONTEXT_OWNERS.pop(context_id, None)

    with _CONTEXT_OWNERS_LOCK:
        _CONTEXT_OWNERS[context_id] = (weakref.ref(context, remove), weakref.ref(owner))


def _owner_for_context(context: zmq.Context) -> ZmqContextOwner | None:
    context_id = id(context)
    with _CONTEXT_OWNERS_LOCK:
        entry = _CONTEXT_OWNERS.get(context_id)
        if entry is None:
            return None
        registered_context = entry[0]()
        owner = entry[1]()
        if registered_context is context and owner is not None:
            return owner
        if _CONTEXT_OWNERS.get(context_id) is entry:
            _CONTEXT_OWNERS.pop(context_id, None)
        return None


def _resolve_channel_owner(
    context: zmq.Context,
    explicit_owner: ZmqContextOwner | None,
) -> ZmqContextOwner | None:
    canonical_owner = _owner_for_context(context)
    if canonical_owner is not None:
        if explicit_owner is not None and explicit_owner is not canonical_owner:
            raise ValueError("explicit owner conflicts with context canonical owner")
        return canonical_owner
    return explicit_owner


class MalformedAction(ValueError):  # noqa: N818
    """Raised when a received action packet cannot satisfy the strict codec."""


class ZmqContextOwner:
    """Own a context and the action channels opened through its factories."""

    def __init__(self, *, context: zmq.Context | None = None) -> None:
        if context is not None and _owner_for_context(context) is not None:
            raise ValueError("context already has a canonical owner")
        self._context = zmq.Context() if context is None else context
        self._owns_context = context is None
        self._lock = threading.RLock()
        self._closed = False
        self._close_complete = threading.Event()
        self._channels: set[object] = set()
        if self._owns_context:
            _associate_owned_context(self._context, self)

    @property
    def context(self) -> zmq.Context:
        """Return the context shared by transport channels."""

        return self._context

    @property
    def live_channel_count(self) -> int:
        """Return the number of currently open channels owned by this context."""

        with self._lock:
            return len(self._channels)

    def open_action_publisher(self, endpoint: str) -> ZmqActionPublisher:
        """Open and track a publisher whose lifetime belongs to this owner."""

        with self._lock:
            self._require_open()
            return ZmqActionPublisher(endpoint, context=self._context, _owner=self)

    def open_action_receiver(
        self,
        endpoint: str,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> ZmqLatestActionReceiver:
        """Open and track a receiver whose lifetime belongs to this owner."""

        with self._lock:
            self._require_open()
            return ZmqLatestActionReceiver(
                endpoint,
                context=self._context,
                monotonic_ns=monotonic_ns,
                _owner=self,
            )

    def close(self) -> None:
        """Quiesce tracked channels before terminating an owned context."""

        with self._lock:
            if self._closed:
                close_complete = self._close_complete
                leader = False
            else:
                self._closed = True
                channels = tuple(self._channels)
                close_complete = self._close_complete
                leader = True
        if not leader:
            close_complete.wait()
            return
        try:
            for channel in channels:
                channel.close()
            if self._owns_context and not self._context.closed:
                self._context.term()
        finally:
            close_complete.set()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("ZeroMQ context owner is closed")

    def _admit_channel(self, channel: object, open_socket: Callable[[], None]) -> None:
        with self._lock:
            self._require_open()
            open_socket()
            self._channels.add(channel)

    def _deregister_channel(self, channel: object) -> None:
        with self._lock:
            self._channels.discard(channel)


class ZmqActionPublisher:
    """A non-blocking, one-part PUB publisher for action envelopes."""

    def __init__(
        self,
        endpoint: str,
        *,
        context: zmq.Context,
        _owner: ZmqContextOwner | None = None,
        _on_close: Callable[[], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._socket: zmq.Socket | None = None
        self._on_close = _on_close
        owner = _resolve_channel_owner(context, _owner)
        if owner is not None and owner.context is not context:
            raise ValueError("owner context does not match publisher context")
        self._owner_ref = weakref.ref(owner) if owner is not None else None
        try:
            if owner is None:
                self._open_socket(context, endpoint)
            else:
                owner._admit_channel(self, lambda: self._open_socket(context, endpoint))
        except BaseException:
            self.close()
            raise

    def _open_socket(self, context: zmq.Context, endpoint: str) -> None:
        socket = context.socket(zmq.PUB)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, 10)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.bind(endpoint)
        except BaseException:
            socket.close(linger=0)
            raise
        self._socket = socket

    def send(self, envelope: ActionEnvelope) -> bool:
        """Publish one encoded envelope, dropping it rather than blocking."""

        encoded = encode_action_envelope(envelope)
        with self._lock:
            socket = self._socket
            if socket is None:
                return False
            try:
                socket.send(encoded, flags=zmq.NOBLOCK)
            except (zmq.Again, zmq.ZMQError):
                return False
            return True

    def close(self) -> None:
        """Close the PUB socket without waiting for queued messages."""

        with self._lock:
            socket, self._socket = self._socket, None
            if socket is not None:
                socket.close(linger=0)
        self._deregister()
        if self._on_close is not None:
            callback, self._on_close = self._on_close, None
            callback()

    def _set_close_callback(self, callback: Callable[[], None]) -> None:
        self._on_close = callback

    def _deregister(self) -> None:
        if self._owner_ref is not None and (owner := self._owner_ref()) is not None:
            owner._deregister_channel(self)


class ZmqLatestActionReceiver:
    """A latest-only, one-part SUB receiver for action envelopes."""

    def __init__(
        self,
        endpoint: str,
        *,
        context: zmq.Context,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        _owner: ZmqContextOwner | None = None,
        _on_close: Callable[[], None] | None = None,
    ) -> None:
        self._lock = threading.Condition(threading.RLock())
        self._monotonic_ns = monotonic_ns
        self._close_requested = threading.Event()
        self._close_complete = threading.Event()
        self._close_lock = threading.Lock()
        self._inflight_decodes = 0
        self._socket: zmq.Socket | None = None
        self._on_close = _on_close
        owner = _resolve_channel_owner(context, _owner)
        if owner is not None and owner.context is not context:
            raise ValueError("owner context does not match receiver context")
        self._owner_ref = weakref.ref(owner) if owner is not None else None
        try:
            if owner is None:
                self._open_socket(context, endpoint)
            else:
                owner._admit_channel(self, lambda: self._open_socket(context, endpoint))
        except BaseException:
            self.close()
            raise

    def _open_socket(self, context: zmq.Context, endpoint: str) -> None:
        socket = context.socket(zmq.SUB)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.connect(endpoint)
        except BaseException:
            socket.close(linger=0)
            raise
        self._socket = socket

    def receive_latest(self, *, timeout_s: float = 0.0) -> ReceivedAction | None:
        """Return the newest unread action without using multipart receive."""

        deadline = time.monotonic() + _timeout_s(timeout_s)
        while True:
            if self._close_requested.is_set():
                return None
            remaining_s = max(0.0, deadline - time.monotonic())
            with self._lock:
                socket = self._socket
                if socket is None or self._close_requested.is_set():
                    return None
                try:
                    if socket.poll(timeout=min(math.ceil(remaining_s * 1_000), 10), flags=zmq.POLLIN):
                        packet = socket.recv(flags=zmq.NOBLOCK)
                        received_monotonic_ns = self._monotonic_ns()
                        if self._close_requested.is_set():
                            return None
                        self._inflight_decodes += 1
                        break
                except (zmq.Again, zmq.ZMQError):
                    return None
            if remaining_s == 0.0:
                return None
        envelope: ActionEnvelope | None = None
        malformed: MalformedAction | None = None
        malformed_cause: BaseException | None = None
        try:
            try:
                envelope = decode_action_envelope(packet)
            except (RecursionError, ValueError) as error:
                malformed = MalformedAction("invalid action packet")
                malformed_cause = error
        finally:
            with self._lock:
                self._inflight_decodes -= 1
                self._lock.notify_all()
                closed = self._close_requested.is_set()
        if closed:
            return None
        if malformed is not None:
            assert malformed_cause is not None
            raise malformed from malformed_cause
        assert envelope is not None
        return ReceivedAction(envelope=envelope, received_monotonic_ns=received_monotonic_ns)

    def close(self) -> None:
        """Close the SUB socket without waiting for queued messages."""

        with self._close_lock:
            already_closing = self._close_requested.is_set()
            self._close_requested.set()
        if already_closing:
            self._close_complete.wait()
            self._deregister()
            return
        try:
            with self._lock:
                socket, self._socket = self._socket, None
                if socket is not None:
                    socket.close(linger=0)
                while self._inflight_decodes:
                    self._lock.wait()
        finally:
            self._close_complete.set()
        self._deregister()
        if self._on_close is not None:
            callback, self._on_close = self._on_close, None
            callback()

    def _set_close_callback(self, callback: Callable[[], None]) -> None:
        self._on_close = callback

    def _deregister(self) -> None:
        if self._owner_ref is not None and (owner := self._owner_ref()) is not None:
            owner._deregister_channel(self)


def _timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("timeout_s must be a finite non-negative number")
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout_s must be a finite non-negative number")
    return timeout


_MANAGEMENT_HWM = 256


def _management_identity(node_id: str, session_id: str) -> bytes:
    if not isinstance(node_id, str) or not node_id or "/" in node_id:
        raise ValueError("node_id must be non-empty UTF-8 text without slash")
    if not isinstance(session_id, str) or not session_id or "/" in session_id:
        raise ValueError("session_id must be non-empty UTF-8 text without slash")
    try:
        return f"{node_id}/{session_id}".encode()
    except UnicodeEncodeError as error:
        raise ValueError("node identity must be UTF-8") from error


def _parse_management_identity(identity: bytes) -> tuple[str, str] | None:
    try:
        node_id, session_id = identity.decode("utf-8").split("/", 1)
    except (UnicodeDecodeError, ValueError):
        return None
    try:
        _management_identity(node_id, session_id)
    except ValueError:
        return None
    return node_id, session_id


def _peer_host(frame: zmq.Frame) -> str | None:
    """Return the numeric/hostname part of libzmq's peer metadata when present."""

    value = frame.get("Peer-Address")
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    parsed = urlparse(value if "://" in value else f"tcp://{value}")
    return parsed.hostname


def _validate_connect_endpoint(endpoint: str) -> str:
    """Permit local inproc transport in direct channel tests, TCP in production."""

    if isinstance(endpoint, str) and endpoint.startswith("inproc://") and len(endpoint) > len("inproc://"):
        return endpoint
    return validate_management_endpoint(endpoint)


def _seed_is_ready(endpoint: str) -> bool:
    """A seed wins discovery only after the TCP peer accepts a connection."""

    parsed = urlparse(endpoint)
    if parsed.hostname is None or parsed.port is None:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.05):
            return True
    except OSError:
        return False


class _ManagedZmqChannel:
    """Shared close/ownership mechanics for ROUTER and DEALER channels."""

    def __init__(self, *, context: zmq.Context, owner: ZmqContextOwner | None) -> None:
        resolved_owner = _resolve_channel_owner(context, owner)
        if resolved_owner is not None and resolved_owner.context is not context:
            raise ValueError("owner context does not match management channel context")
        self._owner_ref = weakref.ref(resolved_owner) if resolved_owner is not None else None
        self._lock = threading.RLock()
        self._socket: zmq.Socket | None = None
        self._closed = threading.Event()
        self._on_close: Callable[[], None] | None = None

    def _set_close_callback(self, callback: Callable[[], None]) -> None:
        self._on_close = callback

    def _open_owned(self, open_socket: Callable[[], None]) -> None:
        owner = self._owner_ref() if self._owner_ref is not None else None
        try:
            if owner is None:
                open_socket()
            else:
                owner._admit_channel(self, open_socket)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            socket, self._socket = self._socket, None
            if socket is not None:
                socket.close(linger=0)
        if self._owner_ref is not None and (owner := self._owner_ref()) is not None:
            owner._deregister_channel(self)
        if self._on_close is not None:
            self._on_close()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


class ZmqHubChannel(_ManagedZmqChannel):
    """Strict ROUTER management adapter with explicit peer identities."""

    def __init__(
        self,
        endpoint: str,
        *,
        context: zmq.Context,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        _owner: ZmqContextOwner | None = None,
    ) -> None:
        super().__init__(context=context, owner=_owner)
        self._monotonic_ns = monotonic_ns
        self._identities: dict[tuple[str, str], bytes] = {}
        self._active_sessions: dict[str, str] = {}
        self._pending_registrations: dict[str, str] = {}
        self.malformed_message_count = 0
        self.endpoint = endpoint
        self._beacon: HubBeaconPublisher | None = None
        self.discovery_enabled = False

        def open_socket() -> None:
            socket = context.socket(zmq.ROUTER)
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDHWM, _MANAGEMENT_HWM)
                socket.setsockopt(zmq.RCVHWM, _MANAGEMENT_HWM)
                socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
                socket.bind(endpoint)
                self.endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8")
            except BaseException:
                socket.close(linger=0)
                raise
            self._socket = socket

        self._open_owned(open_socket)

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        deadline = time.monotonic() + _timeout_s(timeout_s)
        while not self.closed:
            with self._lock:
                socket = self._socket
                if socket is None:
                    return None
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    if not socket.poll(timeout=min(math.ceil(remaining * 1_000), 10), flags=zmq.POLLIN):
                        if remaining == 0.0:
                            return None
                        continue
                    frames = socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                    received_ns = self._monotonic_ns()
                except (zmq.Again, zmq.ZMQError):
                    return None
            if len(frames) != 2:
                self.malformed_message_count += 1
                continue
            identity_frame, payload_frame = frames
            identity = bytes(identity_frame)
            payload = bytes(payload_frame)
            parsed = _parse_management_identity(identity)
            if parsed is None:
                self.malformed_message_count += 1
                continue
            try:
                message = decode_management(payload)
            except (RecursionError, ValueError):
                self.malformed_message_count += 1
                continue
            node_id, session_id = parsed
            if message.sender_id != node_id or message.sender_session_id != session_id:
                self.malformed_message_count += 1
                continue
            self._identities[(node_id, session_id)] = identity
            if message.kind == "register":
                self._pending_registrations[node_id] = session_id
            return ReceivedManagement(
                peer_id=node_id,
                message=message,
                peer_host=_peer_host(identity_frame),
                received_monotonic_ns=received_ns,
            )
        return None

    def send(self, peer_id: str, message: ManagementMessage) -> bool:
        if not isinstance(peer_id, str) or not peer_id:
            return False
        encoded = encode_management(message)
        with self._lock:
            socket = self._socket
            session_id = self._active_sessions.get(peer_id)
            if message.kind == "registered":
                session_id = self._pending_registrations.get(peer_id, session_id)
            identity = self._identities.get((peer_id, session_id)) if session_id is not None else None
            if socket is None or identity is None:
                return False
            try:
                socket.send_multipart([identity, encoded], flags=zmq.NOBLOCK)
            except (zmq.Again, zmq.ZMQError):
                return False
            if message.kind == "registered" and session_id is not None:
                self._active_sessions[peer_id] = session_id
                self._pending_registrations.pop(peer_id, None)
        return True

    def start_beacon(
        self,
        hub_epoch: str,
        advertise_endpoint: str,
        publisher: HubBeaconPublisher,
    ) -> None:
        validate_management_endpoint(advertise_endpoint)
        self._beacon = publisher
        try:
            publisher.start(DiscoveryBeacon(PROTOCOL_VERSION, hub_epoch, advertise_endpoint))
        except BaseException:
            self._beacon = None
            publisher.close()
            raise
        self.discovery_enabled = True

    def close(self) -> None:
        publisher = self._beacon
        if publisher is not None:
            publisher.close()
            self._beacon = None
            self.discovery_enabled = False
        super().close()


class ZmqNodeChannel(_ManagedZmqChannel):
    """Strict DEALER management adapter for one node process session."""

    def __init__(
        self,
        endpoint: str,
        node_id: str,
        session_id: str,
        *,
        context: zmq.Context,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        hub_epoch: str | None = None,
        _owner: ZmqContextOwner | None = None,
    ) -> None:
        super().__init__(context=context, owner=_owner)
        self.endpoint = _validate_connect_endpoint(endpoint)
        self._context = context
        self.node_id = node_id
        self.session_id = session_id
        self._identity = _management_identity(node_id, session_id)
        self._monotonic_ns = monotonic_ns
        self.malformed_message_count = 0
        self._hub_epoch = hub_epoch
        self._hub_session_id: str | None = hub_epoch
        self._registered = False

        self._resolver: Callable[[], tuple[str, str | None]] | None = None
        self._monitor: zmq.Socket | None = None
        self._open_owned(self._open_socket)

    def _open_socket(self) -> None:
        socket = self._context.socket(zmq.DEALER)
        monitor: zmq.Socket | None = None
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, _MANAGEMENT_HWM)
            socket.setsockopt(zmq.RCVHWM, _MANAGEMENT_HWM)
            socket.setsockopt(zmq.IDENTITY, self._identity)
            socket.setsockopt(zmq.HEARTBEAT_IVL, 1_000)
            socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 3_000)
            monitor_endpoint = f"inproc://lekit-node-monitor-{uuid.uuid4()}"
            socket.monitor(monitor_endpoint, zmq.EVENT_DISCONNECTED)
            monitor = self._context.socket(zmq.PAIR)
            monitor.setsockopt(zmq.LINGER, 0)
            monitor.connect(monitor_endpoint)
            socket.connect(self.endpoint)
        except BaseException:
            if monitor is not None:
                monitor.close(linger=0)
            socket.close(linger=0)
            raise
        self._socket = socket
        self._monitor = monitor

    def _set_resolver(self, resolver: Callable[[], tuple[str, str | None]]) -> None:
        self._resolver = resolver

    def rediscover(self) -> bool:
        """Replace the DEALER atomically after management loss or a new beacon."""

        if self.closed or self._resolver is None:
            return False
        endpoint, hub_epoch = self._resolver()
        endpoint = _validate_connect_endpoint(endpoint)
        with self._lock:
            epoch_changed = hub_epoch != self._hub_epoch
            if self.closed or (endpoint == self.endpoint and not epoch_changed):
                return False
            socket, self._socket = self._socket, None
            monitor, self._monitor = self._monitor, None
            if monitor is not None:
                monitor.close(linger=0)
            if socket is not None:
                socket.close(linger=0)
            self.endpoint = endpoint
            self._hub_epoch = hub_epoch
            self._hub_session_id = hub_epoch
            self._registered = False
            self._open_socket()
        return True

    def _monitor_disconnected_locked(self) -> bool:
        monitor = self._monitor
        if monitor is None:
            return False
        try:
            if not monitor.poll(timeout=0, flags=zmq.POLLIN):
                return False
            event = parse_monitor_message(monitor.recv_multipart(flags=zmq.NOBLOCK))
        except (zmq.Again, zmq.ZMQError, ValueError):
            return False
        return event.get("event") == zmq.EVENT_DISCONNECTED

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        deadline = time.monotonic() + _timeout_s(timeout_s)
        while not self.closed:
            with self._lock:
                if self._monitor_disconnected_locked():
                    # `rediscover` owns the same reentrant lock and replaces both sockets.
                    self.rediscover()
                socket = self._socket
                if socket is None:
                    return None
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    if not socket.poll(timeout=min(math.ceil(remaining * 1_000), 10), flags=zmq.POLLIN):
                        if remaining == 0.0:
                            return None
                        continue
                    frames = socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                    received_ns = self._monotonic_ns()
                except (zmq.Again, zmq.ZMQError):
                    return None
            if len(frames) != 1:
                self.malformed_message_count += 1
                continue
            try:
                message = decode_management(bytes(frames[0]))
            except (RecursionError, ValueError):
                self.malformed_message_count += 1
                continue
            if message.sender_id != "hub":
                self.malformed_message_count += 1
                continue
            body_epoch = message.body.get("hub_epoch")
            if body_epoch is not None and (not isinstance(body_epoch, str) or not body_epoch.strip()):
                self.malformed_message_count += 1
                continue
            if not self._registered:
                if (
                    message.kind != "registered"
                    or not isinstance(body_epoch, str)
                    or not body_epoch.strip()
                    or message.sender_session_id != body_epoch
                    or (self._hub_epoch is not None and body_epoch != self._hub_epoch)
                ):
                    self.malformed_message_count += 1
                    continue
                self._hub_epoch = body_epoch
                self._hub_session_id = message.sender_session_id
                self._registered = True
            elif self._hub_epoch is not None:
                if message.sender_session_id != self._hub_session_id or (
                    body_epoch is not None and body_epoch != self._hub_epoch
                ):
                    self.malformed_message_count += 1
                    continue
            return ReceivedManagement("hub", message, None, received_ns)
        return None

    def send(self, message: ManagementMessage) -> bool:
        if message.sender_id != self.node_id or message.sender_session_id != self.session_id:
            return False
        encoded = encode_management(message)
        with self._lock:
            socket = self._socket
            if socket is None:
                return False
            try:
                socket.send(encoded, flags=zmq.NOBLOCK)
            except (zmq.Again, zmq.ZMQError):
                return False
        return True

    def close(self) -> None:
        with self._lock:
            monitor, self._monitor = self._monitor, None
            if monitor is not None:
                monitor.close(linger=0)
        super().close()


class ZmqRuntime:
    """Production Runtime adapter; only this class constructs transport sockets."""

    def __init__(
        self,
        *,
        context: zmq.Context | None = None,
        discovery_enabled: bool = True,
        discovery_group: str = "239.255.42.99",
        discovery_port: int = 45990,
        discovery_loopback: bool = True,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        discovery_listener_factory: Callable[[], CooperativeDiscoveryListener] | None = None,
        seed_ready: Callable[[str], bool] | None = None,
        cooperative_seed_probe: Callable[[str, threading.Event], bool] | None = None,
        beacon_publisher_factory: Callable[..., HubBeaconPublisher] | None = None,
    ) -> None:
        if not isinstance(discovery_enabled, bool):
            raise ValueError("discovery_enabled must be a bool")
        if not isinstance(discovery_loopback, bool):
            raise ValueError("discovery_loopback must be a bool")
        self._owner = ZmqContextOwner(context=context)
        self._context = self._owner.context
        self._discovery_enabled = discovery_enabled
        self._discovery_group = discovery_group
        self._discovery_port = discovery_port
        self._discovery_loopback = discovery_loopback
        self._monotonic_ns = monotonic_ns
        self._discovery_listener_factory = discovery_listener_factory
        self._seed_ready = seed_ready or _seed_is_ready
        self._cooperative_seed_probe = cooperative_seed_probe
        self._beacon_publisher_factory = beacon_publisher_factory or HubBeaconPublisher
        self._lock = threading.RLock()
        self._closed = False
        self._channels: set[object] = set()

    @property
    def live_channel_count(self) -> int:
        """Return the number of runtime channels that have not been closed."""

        with self._lock:
            return len(self._channels)

    def _track(self, channel: _ManagedZmqChannel | ZmqActionPublisher | ZmqLatestActionReceiver):
        self._channels.add(channel)
        channel._set_close_callback(lambda: self._discard(channel))
        return channel

    def _discard(self, channel: object) -> None:
        with self._lock:
            self._channels.discard(channel)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("ZeroMQ runtime is closed")

    def open_hub(
        self,
        endpoint: str,
        *,
        hub_epoch: str,
        advertise_endpoint: str | None = None,
    ) -> ZmqHubChannel:
        with self._lock:
            self._require_open()
            if not isinstance(hub_epoch, str) or not hub_epoch.strip():
                raise ValueError("hub_epoch must not be blank")
            if advertise_endpoint is not None:
                validate_management_endpoint(advertise_endpoint)
            channel: ZmqHubChannel | None = None
            try:
                channel = ZmqHubChannel(
                    endpoint,
                    context=self._context,
                    monotonic_ns=self._monotonic_ns,
                    _owner=self._owner,
                )
                advertised = advertise_endpoint or channel.endpoint
                try:
                    validate_management_endpoint(advertised)
                except ValueError:
                    if advertise_endpoint is not None:
                        raise
                else:
                    if self._discovery_enabled:
                        publisher = self._beacon_publisher_factory(
                            self._discovery_group,
                            self._discovery_port,
                            loopback=self._discovery_loopback,
                        )
                        channel.start_beacon(hub_epoch, advertised, publisher)
                return self._track(channel)
            except BaseException:
                if channel is not None:
                    channel.close()
                raise

    def open_node(self, node_id: str, session_id: str, *, hub_seed: str | None) -> ZmqNodeChannel:
        with self._lock:
            self._require_open()
            endpoint, hub_epoch = self._resolve_endpoint(hub_seed)
            try:
                channel = ZmqNodeChannel(
                    endpoint,
                    node_id,
                    session_id,
                    context=self._context,
                    monotonic_ns=self._monotonic_ns,
                    hub_epoch=hub_epoch,
                    _owner=self._owner,
                )
                channel._set_resolver(lambda: self._resolve_endpoint(hub_seed, prefer_seed=False))
                return self._track(channel)
            except BaseException:
                raise

    def _listener(self) -> CooperativeDiscoveryListener:
        if self._discovery_listener_factory is not None:
            listener = self._discovery_listener_factory()
        else:
            listener = HubBeaconListener(
                self._discovery_group,
                self._discovery_port,
                loopback=self._discovery_loopback,
            )
        if (
            getattr(listener, "cooperative_discovery_listener", None) is not True
            or not callable(getattr(listener, "receive", None))
            or not callable(getattr(listener, "close", None))
        ):
            raise ValueError("discovery listener must declare cooperative discovery listener contract")
        return listener

    def _resolve_endpoint(self, hub_seed: str | None, *, prefer_seed: bool = True) -> tuple[str, str | None]:
        seed = validate_management_endpoint(hub_seed) if hub_seed else None
        if not self._discovery_enabled:
            if seed is None:
                raise ValueError("no valid hub seed available while discovery is disabled")
            return seed, None
        listener = self._listener()
        if (
            seed is not None
            and prefer_seed
            and self._cooperative_seed_probe is None
            and self._seed_ready(seed)
        ):
            listener.close()
            return seed, None
        results: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=2)
        cancelled = threading.Event()

        def report(kind: str, value: object) -> None:
            if not cancelled.is_set():
                results.put((kind, value))

        def wait_beacon() -> None:
            beacon = listener.receive(timeout_s=1.0)
            if beacon is not None:
                report("beacon", beacon)

        def wait_seed() -> None:
            if seed is None or not prefer_seed or self._cooperative_seed_probe is None:
                return
            ready = self._cooperative_seed_probe(seed, cancelled)
            if ready:
                report("seed", seed)

        beacon_worker = threading.Thread(target=wait_beacon, name="lekit-beacon-resolver", daemon=True)
        seed_worker = threading.Thread(target=wait_seed, name="lekit-seed-resolver", daemon=True)
        try:
            beacon_worker.start()
            if seed is not None and prefer_seed:
                seed_worker.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    kind, value = results.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if kind == "seed":
                    return value, None  # type: ignore[return-value]
                beacon = value
                assert isinstance(beacon, DiscoveryBeacon)
                return beacon.management_endpoint, beacon.hub_epoch
            if seed is not None:
                return seed, None
            raise ValueError("no valid hub seed or discovery beacon available")
        finally:
            cancelled.set()
            listener.close()
            beacon_worker.join(timeout=0.05)
            if seed_worker.is_alive():
                seed_worker.join(timeout=0.05)

    def open_action_publisher(self, endpoint: str) -> ZmqActionPublisher:
        with self._lock:
            self._require_open()
            return self._track(ZmqActionPublisher(endpoint, context=self._context, _owner=self._owner))

    def open_action_receiver(self, endpoint: str) -> ZmqLatestActionReceiver:
        with self._lock:
            self._require_open()
            return self._track(
                ZmqLatestActionReceiver(
                    endpoint,
                    context=self._context,
                    monotonic_ns=self._monotonic_ns,
                    _owner=self._owner,
                )
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            channels = tuple(self._channels)
            self._channels.clear()
        for channel in channels:
            channel.close()  # type: ignore[union-attr]
        self._owner.close()


__all__ = [
    "MalformedAction",
    "ZmqActionPublisher",
    "ZmqContextOwner",
    "ZmqHubChannel",
    "ZmqLatestActionReceiver",
    "ZmqNodeChannel",
    "ZmqRuntime",
]
