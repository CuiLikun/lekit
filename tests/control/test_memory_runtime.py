import threading

import pytest

from lekit.control import ActionEnvelope, ManagementMessage, MemoryRuntime


class _GatedCondition:
    """Pause one selected critical-section entry without relying on sleeps."""

    def __init__(self, condition: threading.Condition, *, gate_on_entry: int) -> None:
        self._condition = condition
        self._gate_on_entry = gate_on_entry
        self._entry_count = 0
        self.accepting = threading.Event()
        self.release = threading.Event()

    def __enter__(self) -> "_GatedCondition":
        self._condition.acquire()
        self._entry_count += 1
        if self._entry_count == self._gate_on_entry:
            self.accepting.set()
            self._condition.release()
            assert self.release.wait(timeout=1.0)
            self._condition.acquire()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._condition.release()

    def notify(self) -> None:
        self._condition.notify()

    def notify_all(self) -> None:
        self._condition.notify_all()

    def wait(self, timeout: float | None = None) -> bool:
        return self._condition.wait(timeout)


def message(kind: str, *, sender: str, sequence: int = 0) -> ManagementMessage:
    return ManagementMessage(
        protocol_version=1,
        kind=kind,
        correlation_id=f"{kind}-{sequence}",
        sender_id=sender,
        sender_session_id=f"{sender}-session",
        sequence=sequence,
        sent_at_ns=sequence,
        body={"sequence": sequence},
    )


def envelope(*, sequence: int) -> ActionEnvelope:
    return ActionEnvelope(
        handle_id="handle-1",
        hub_epoch="epoch-1",
        fencing_token=1,
        controller_id="quest",
        controller_session_id="quest-session",
        stream_session_id="stream-1",
        sequence=sequence,
        captured_monotonic_ns=sequence,
        captured_utc_ns=sequence,
        payload_schema="lekit.action.v1",
        payload=b"opaque",
    )


def test_management_routes_between_hub_and_named_node():
    runtime = MemoryRuntime()
    hub = runtime.open_hub("memory://hub", hub_epoch="epoch-1", advertise_endpoint="memory://advertised")
    node = runtime.open_node("robot-1", "session-1", hub_seed="memory://hub")

    assert node.send(message("register", sender="robot-1")) is True
    received = hub.receive(timeout_s=0.0)
    assert received is not None
    assert received.peer_id == "robot-1"
    assert received.peer_host is None
    assert received.message.kind == "register"
    assert received.received_monotonic_ns >= 0

    assert hub.send("robot-1", message("registered", sender="hub")) is True
    reply = node.receive(timeout_s=0.0)
    assert reply is not None
    assert reply.message.kind == "registered"


def test_action_receiver_keeps_only_latest_atomic_envelope():
    runtime = MemoryRuntime()
    publisher = runtime.open_action_publisher("memory://quest/actions")
    receiver = runtime.open_action_receiver("memory://quest/actions")

    assert publisher.send(envelope(sequence=1)) is True
    assert publisher.send(envelope(sequence=2)) is True

    received = receiver.receive_latest(timeout_s=0.0)
    assert received is not None
    assert received.envelope.sequence == 2
    assert received.received_monotonic_ns >= 0
    assert receiver.receive_latest(timeout_s=0.0) is None


def test_management_inbox_bounds_non_coalesced_messages_without_blocking_sender():
    runtime = MemoryRuntime(management_inbox_max=2)
    hub = runtime.open_hub("memory://hub", hub_epoch="epoch-1")
    node = runtime.open_node("robot-1", "session-1", hub_seed="memory://hub")

    assert node.send(message("register", sender="robot-1", sequence=1)) is True
    assert node.send(message("command", sender="robot-1", sequence=2)) is True
    assert node.send(message("command", sender="robot-1", sequence=3)) is False

    first = hub.receive(timeout_s=0.0)
    second = hub.receive(timeout_s=0.0)
    assert first is not None
    assert second is not None
    assert [first.message.sequence, second.message.sequence] == [1, 2]


def test_management_coalesces_status_and_heartbeat_by_peer_and_kind():
    runtime = MemoryRuntime(management_inbox_max=3)
    hub = runtime.open_hub("memory://hub", hub_epoch="epoch-1")
    robot = runtime.open_node("robot-1", "session-1", hub_seed="memory://hub")
    controller = runtime.open_node("controller-1", "session-1", hub_seed="memory://hub")

    assert robot.send(message("status", sender="robot-1", sequence=1)) is True
    assert robot.send(message("status", sender="robot-1", sequence=2)) is True
    assert robot.send(message("heartbeat", sender="robot-1", sequence=3)) is True
    assert controller.send(message("status", sender="controller-1", sequence=4)) is True
    assert robot.send(message("heartbeat", sender="robot-1", sequence=5)) is True

    received = [hub.receive(timeout_s=0.0) for _ in range(3)]
    assert [
        (item.peer_id, item.message.kind, item.message.sequence) for item in received if item is not None
    ] == [
        ("robot-1", "status", 2),
        ("robot-1", "heartbeat", 5),
        ("controller-1", "status", 4),
    ]


def test_close_wakes_blocked_receivers_and_rejects_later_sends():
    runtime = MemoryRuntime()
    hub = runtime.open_hub("memory://hub", hub_epoch="epoch-1")
    node = runtime.open_node("robot-1", "session-1", hub_seed="memory://hub")
    result: list[object | None] = []
    receiving = threading.Event()

    def receive() -> None:
        receiving.set()
        result.append(hub.receive(timeout_s=1.0))

    thread = threading.Thread(target=receive)
    thread.start()
    assert receiving.wait(timeout=0.02)
    hub.close()
    thread.join(timeout=0.02)

    assert thread.is_alive() is False
    assert result == [None]
    assert hub.receive(timeout_s=0.0) is None
    assert node.send(message("register", sender="robot-1")) is False
    assert hub.send("robot-1", message("registered", sender="hub")) is False


def test_action_close_wakes_receiver_and_rejects_later_sends():
    runtime = MemoryRuntime()
    publisher = runtime.open_action_publisher("memory://quest/actions")
    receiver = runtime.open_action_receiver("memory://quest/actions")
    result: list[object | None] = []
    receiving = threading.Event()

    def receive() -> None:
        receiving.set()
        result.append(receiver.receive_latest(timeout_s=1.0))

    thread = threading.Thread(target=receive)
    thread.start()
    assert receiving.wait(timeout=0.02)
    receiver.close()
    thread.join(timeout=0.02)

    assert thread.is_alive() is False
    assert result == [None]
    assert receiver.receive_latest(timeout_s=0.0) is None
    assert publisher.send(envelope(sequence=1)) is False


@pytest.mark.parametrize("maximum", [False, True, -1, 0, 1.0, float("nan"), float("inf")])
def test_management_inbox_max_rejects_non_positive_or_non_integer_values(maximum: object):
    with pytest.raises(ValueError, match="positive integer"):
        MemoryRuntime(management_inbox_max=maximum)  # type: ignore[arg-type]


def test_management_timestamp_is_captured_inside_destination_acceptance_lock():
    runtime = MemoryRuntime()
    hub = runtime.open_hub("memory://hub", hub_epoch="epoch-1")
    node = runtime.open_node("robot-1", "session-1", hub_seed="memory://hub")
    gate = _GatedCondition(hub._inbox.condition, gate_on_entry=2)
    hub._inbox.condition = gate
    timestamp_called = threading.Event()

    def monotonic_ns() -> int:
        timestamp_called.set()
        return 22 if gate.release.is_set() else 11

    runtime.monotonic_ns = monotonic_ns
    sent: list[bool] = []
    sender = threading.Thread(target=lambda: sent.append(node.send(message("register", sender="robot-1"))))
    sender.start()
    assert gate.accepting.wait(timeout=1.0)
    try:
        assert timestamp_called.is_set() is False
    finally:
        gate.release.set()
        sender.join(timeout=1.0)
    received = hub.receive(timeout_s=0.0)
    assert sender.is_alive() is False
    assert sent == [True]
    assert received is not None
    assert received.received_monotonic_ns == 22


def test_action_timestamp_is_captured_inside_destination_acceptance_lock():
    runtime = MemoryRuntime()
    publisher = runtime.open_action_publisher("memory://quest/actions")
    receiver = runtime.open_action_receiver("memory://quest/actions")
    gate = _GatedCondition(publisher._slot.condition, gate_on_entry=1)
    publisher._slot.condition = gate
    timestamp_called = threading.Event()

    def monotonic_ns() -> int:
        timestamp_called.set()
        return 22 if gate.release.is_set() else 11

    runtime.monotonic_ns = monotonic_ns
    sent: list[bool] = []
    sender = threading.Thread(target=lambda: sent.append(publisher.send(envelope(sequence=1))))
    sender.start()
    assert gate.accepting.wait(timeout=1.0)
    try:
        assert timestamp_called.is_set() is False
    finally:
        gate.release.set()
        sender.join(timeout=1.0)
    received = receiver.receive_latest(timeout_s=0.0)
    assert sender.is_alive() is False
    assert sent == [True]
    assert received is not None
    assert received.received_monotonic_ns == 22
