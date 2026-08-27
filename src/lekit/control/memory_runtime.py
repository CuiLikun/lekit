"""Deterministic in-process implementation of the Control Hub runtime."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .model import ActionEnvelope, ManagementMessage
from .runtime import ReceivedAction, ReceivedManagement

_COALESCED_MANAGEMENT_KINDS = frozenset({"heartbeat", "status"})


@dataclass(slots=True)
class _ManagementInbox:
    maximum: int
    condition: threading.Condition = field(default_factory=threading.Condition)
    entries: deque[ReceivedManagement] = field(default_factory=deque)
    closed: bool = False

    def put(
        self,
        *,
        peer_id: str,
        message: ManagementMessage,
        peer_host: str | None,
        monotonic_ns: Callable[[], int],
    ) -> bool:
        with self.condition:
            if self.closed:
                return False
            if message.kind in _COALESCED_MANAGEMENT_KINDS:
                key = (peer_id, message.kind)
                for index, queued in enumerate(self.entries):
                    if (queued.peer_id, queued.message.kind) == key:
                        self.entries[index] = ReceivedManagement(peer_id, message, peer_host, monotonic_ns())
                        self.condition.notify()
                        return True
            if len(self.entries) >= self.maximum:
                return False
            self.entries.append(ReceivedManagement(peer_id, message, peer_host, monotonic_ns()))
            self.condition.notify()
            return True

    def receive(self, *, timeout_s: float) -> ReceivedManagement | None:
        timeout_s = max(0.0, timeout_s)
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while not self.entries and not self.closed:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    return None
                self.condition.wait(remaining_s)
            if self.closed:
                return None
            return self.entries.popleft()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.entries.clear()
            self.condition.notify_all()

    def is_closed(self) -> bool:
        with self.condition:
            return self.closed


@dataclass(slots=True)
class _ActionSlot:
    condition: threading.Condition = field(default_factory=threading.Condition)
    value: ReceivedAction | None = None
    closed: bool = False

    def put(self, envelope: ActionEnvelope, monotonic_ns: Callable[[], int]) -> bool:
        with self.condition:
            if self.closed:
                return False
            self.value = ReceivedAction(envelope, monotonic_ns())
            self.condition.notify_all()
            return True

    def receive_latest(self, *, timeout_s: float) -> ReceivedAction | None:
        timeout_s = max(0.0, timeout_s)
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.value is None and not self.closed:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    return None
                self.condition.wait(remaining_s)
            if self.closed:
                return None
            received = self.value
            self.value = None
            return received

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.value = None
            self.condition.notify_all()

    def is_closed(self) -> bool:
        with self.condition:
            return self.closed


class _MemoryHubChannel:
    def __init__(
        self,
        runtime: MemoryRuntime,
        endpoint: str,
        hub_epoch: str,
        advertise_endpoint: str | None,
        inbox: _ManagementInbox,
    ) -> None:
        self._runtime = runtime
        self.endpoint = endpoint
        self.hub_epoch = hub_epoch
        self.advertise_endpoint = advertise_endpoint
        self._inbox = inbox

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        return self._inbox.receive(timeout_s=timeout_s)

    def send(self, peer_id: str, message: ManagementMessage) -> bool:
        return self._runtime._send_from_hub(self, peer_id, message)

    def close(self) -> None:
        self._inbox.close()

    def _receive(self, peer_id: str, message: ManagementMessage) -> bool:
        return self._inbox.put(
            peer_id=peer_id,
            message=message,
            peer_host=None,
            monotonic_ns=self._runtime.monotonic_ns,
        )

    def _is_closed(self) -> bool:
        return self._inbox.is_closed()


class _MemoryNodeChannel:
    def __init__(
        self,
        runtime: MemoryRuntime,
        node_id: str,
        session_id: str,
        hub_seed: str | None,
        inbox: _ManagementInbox,
    ) -> None:
        self._runtime = runtime
        self.node_id = node_id
        self.session_id = session_id
        self.hub_seed = hub_seed
        self._inbox = inbox

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        return self._inbox.receive(timeout_s=timeout_s)

    def send(self, message: ManagementMessage) -> bool:
        return self._runtime._send_from_node(self, message)

    def close(self) -> None:
        self._inbox.close()

    def _receive(self, message: ManagementMessage) -> bool:
        return self._inbox.put(
            peer_id="hub",
            message=message,
            peer_host=None,
            monotonic_ns=self._runtime.monotonic_ns,
        )

    def _is_closed(self) -> bool:
        return self._inbox.is_closed()


class _MemoryActionPublisher:
    def __init__(self, runtime: MemoryRuntime, slot: _ActionSlot) -> None:
        self._runtime = runtime
        self._slot = slot

    def send(self, envelope: ActionEnvelope) -> bool:
        return self._slot.put(envelope, self._runtime.monotonic_ns)

    def close(self) -> None:
        self._slot.close()


class _MemoryLatestActionReceiver:
    def __init__(self, slot: _ActionSlot) -> None:
        self._slot = slot

    def receive_latest(self, *, timeout_s: float = 0.0) -> ReceivedAction | None:
        return self._slot.receive_latest(timeout_s=timeout_s)

    def close(self) -> None:
        self._slot.close()


class MemoryRuntime:
    """A thread-safe in-memory Runtime with bounded management queues."""

    def __init__(self, *, management_inbox_max: int = 256) -> None:
        if (
            not isinstance(management_inbox_max, int)
            or isinstance(management_inbox_max, bool)
            or management_inbox_max < 1
        ):
            raise ValueError("management_inbox_max must be a positive integer")
        self._management_inbox_max = management_inbox_max
        self._lock = threading.RLock()
        self._closed = False
        self._hub: _MemoryHubChannel | None = None
        self._nodes: dict[str, _MemoryNodeChannel] = {}
        self._action_slots: dict[str, _ActionSlot] = {}

    @staticmethod
    def monotonic_ns() -> int:
        """Return the runtime's monotonic timestamp source."""
        return time.monotonic_ns()

    def open_hub(
        self,
        endpoint: str,
        *,
        hub_epoch: str,
        advertise_endpoint: str | None = None,
    ) -> _MemoryHubChannel:
        with self._lock:
            self._require_open()
            if self._hub is not None:
                self._hub.close()
            hub = _MemoryHubChannel(
                self,
                endpoint,
                hub_epoch,
                advertise_endpoint,
                _ManagementInbox(self._management_inbox_max),
            )
            self._hub = hub
            return hub

    def open_node(
        self,
        node_id: str,
        session_id: str,
        *,
        hub_seed: str | None,
    ) -> _MemoryNodeChannel:
        with self._lock:
            self._require_open()
            existing = self._nodes.get(node_id)
            if existing is not None:
                existing.close()
            node = _MemoryNodeChannel(
                self,
                node_id,
                session_id,
                hub_seed,
                _ManagementInbox(self._management_inbox_max),
            )
            self._nodes[node_id] = node
            return node

    def open_action_publisher(self, endpoint: str) -> _MemoryActionPublisher:
        return _MemoryActionPublisher(self, self._open_action_slot(endpoint))

    def open_action_receiver(self, endpoint: str) -> _MemoryLatestActionReceiver:
        return _MemoryLatestActionReceiver(self._open_action_slot(endpoint))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._hub is not None:
                self._hub.close()
            for node in self._nodes.values():
                node.close()
            for slot in self._action_slots.values():
                slot.close()

    def _open_action_slot(self, endpoint: str) -> _ActionSlot:
        with self._lock:
            self._require_open()
            slot = self._action_slots.get(endpoint)
            if slot is None or slot.is_closed():
                slot = _ActionSlot()
                self._action_slots[endpoint] = slot
            return slot

    def _send_from_hub(self, hub: _MemoryHubChannel, peer_id: str, message: ManagementMessage) -> bool:
        with self._lock:
            if self._closed or self._hub is not hub or hub._is_closed():
                return False
            node = self._nodes.get(peer_id)
            if node is None or node._is_closed():
                return False
            return node._receive(message)

    def _send_from_node(self, node: _MemoryNodeChannel, message: ManagementMessage) -> bool:
        with self._lock:
            hub = self._hub
            if self._closed or self._nodes.get(node.node_id) is not node or node._is_closed() or hub is None:
                return False
            if hub._is_closed() or not self._matches_hub_seed(node.hub_seed, hub):
                return False
            return hub._receive(node.node_id, message)

    @staticmethod
    def _matches_hub_seed(seed: str | None, hub: _MemoryHubChannel) -> bool:
        return seed is None or seed in {hub.endpoint, hub.advertise_endpoint}

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryRuntime is closed")
