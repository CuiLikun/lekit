"""Real loopback tests for the ZeroMQ action transport."""

from __future__ import annotations

import struct
import threading
import time
import uuid
from collections.abc import Callable, Iterator

import pytest
import zmq

import lekit.control.zmq_runtime as zmq_runtime
from lekit.control import (
    ActionEnvelope,
    MalformedAction,
    ReceivedAction,
    ZmqActionPublisher,
    ZmqContextOwner,
    ZmqLatestActionReceiver,
    encode_action_envelope,
)


def management_message(
    kind: str,
    *,
    sender_id: str = "robot-1",
    sender_session_id: str = "session-1",
    sequence: int = 1,
    body: dict[str, object] | None = None,
) -> object:
    from lekit.control import ManagementMessage

    return ManagementMessage(
        protocol_version=1,
        kind=kind,
        correlation_id=f"correlation-{sequence}",
        sender_id=sender_id,
        sender_session_id=sender_session_id,
        sequence=sequence,
        sent_at_ns=sequence,
        body={} if body is None else body,
    )


@pytest.fixture
def zmq_context() -> Iterator[zmq.Context]:
    context = zmq.Context()
    try:
        yield context
    finally:
        context.term()


def envelope(sequence: int, payload: bytes | None = None) -> ActionEnvelope:
    return ActionEnvelope(
        handle_id="handle-1",
        hub_epoch="epoch-1",
        fencing_token=1,
        controller_id="controller-1",
        controller_session_id="controller-session-1",
        stream_session_id="stream-1",
        sequence=sequence,
        captured_monotonic_ns=sequence,
        captured_utc_ns=sequence,
        payload_schema="lekit.action.v1",
        payload=payload or f"frame-{sequence}".encode(),
    )


def open_tcp_publisher(context: zmq.Context) -> tuple[ZmqActionPublisher, str]:
    publisher = ZmqActionPublisher("tcp://127.0.0.1:*", context=context)
    assert publisher._socket is not None
    endpoint = publisher._socket.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
    return publisher, endpoint


def eventually_receive(
    receiver: ZmqLatestActionReceiver,
    *,
    timeout_s: float = 1.0,
) -> ReceivedAction:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        received = receiver.receive_latest(timeout_s=0.01)
        if received is not None:
            return received
    pytest.fail("timed out waiting for loopback action")


def wait_for_subscription(
    publisher: ZmqActionPublisher,
    receiver: ZmqLatestActionReceiver,
) -> None:
    deadline = time.monotonic() + 1.0
    probe = envelope(10_000, b"subscription-probe")
    while time.monotonic() < deadline:
        assert publisher.send(probe) is True
        received = receiver.receive_latest(timeout_s=0.01)
        if received is not None and received.envelope == probe:
            return
    pytest.fail("ZeroMQ SUB subscription did not settle")


def test_action_publisher_and_receiver_deliver_one_complete_encoded_frame(
    zmq_context: zmq.Context,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)
    try:
        wait_for_subscription(publisher, receiver)
        expected = envelope(7, b"complete-frame")

        assert publisher.send(expected) is True
        received = eventually_receive(receiver)

        assert received.envelope == expected
        assert received.received_monotonic_ns >= 0
    finally:
        receiver.close()
        publisher.close()


def test_slow_receiver_observes_latest_complete_frame(
    zmq_context: zmq.Context,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)
    try:
        wait_for_subscription(publisher, receiver)
        for sequence in range(100):
            assert publisher.send(envelope(sequence, f"frame-{sequence}".encode())) is True

        deadline = time.monotonic() + 1.0
        received = None
        while time.monotonic() < deadline:
            candidate = receiver.receive_latest(timeout_s=0.01)
            if candidate is not None and candidate.envelope.sequence == 99:
                received = candidate
                break

        assert received is not None
        assert received.envelope.sequence == 99
        assert received.envelope.payload == b"frame-99"
    finally:
        receiver.close()
        publisher.close()


def test_settled_inproc_receiver_gets_latest_frame_on_its_first_burst_receive() -> None:
    context = zmq.Context()
    endpoint = f"inproc://latest-first-{uuid.uuid4()}"
    publisher = ZmqActionPublisher(endpoint, context=context)
    receiver = ZmqLatestActionReceiver(endpoint, context=context)
    observer = ZmqLatestActionReceiver(endpoint, context=context)
    try:
        wait_for_subscription(publisher, receiver)
        wait_for_subscription(publisher, observer)
        for sequence in range(100):
            assert publisher.send(envelope(sequence, f"frame-{sequence}".encode())) is True

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            observed = observer.receive_latest(timeout_s=0.01)
            if observed is not None and observed.envelope.sequence == 99:
                break
        else:
            pytest.fail("final frame did not enter the settled inproc transport")

        received = receiver.receive_latest(timeout_s=0.0)

        assert received is not None
        assert received.envelope.sequence == 99
        assert received.envelope.payload == b"frame-99"
    finally:
        observer.close()
        receiver.close()
        publisher.close()
        context.term()


def test_publisher_sends_exactly_one_message_part(
    zmq_context: zmq.Context,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    inspector = zmq_context.socket(zmq.SUB)
    inspector.setsockopt(zmq.LINGER, 0)
    inspector.setsockopt(zmq.SUBSCRIBE, b"")
    inspector.connect(endpoint)
    try:
        probe = envelope(10_001, b"inspector-probe")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            assert publisher.send(probe) is True
            if inspector.poll(timeout=10, flags=zmq.POLLIN):
                assert inspector.recv_multipart(flags=zmq.NOBLOCK) == [encode_action_envelope(probe)]
                break
        else:
            pytest.fail("ZeroMQ SUB subscription did not settle")
    finally:
        inspector.close(linger=0)
        publisher.close()


def test_receiver_records_timestamp_before_decoding_malformed_bytes(
    zmq_context: zmq.Context,
) -> None:
    sender = zmq_context.socket(zmq.PUB)
    sender.setsockopt(zmq.LINGER, 0)
    sender.bind("tcp://127.0.0.1:*")
    endpoint = sender.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
    timestamps: list[int] = []

    def monotonic_ns() -> int:
        timestamps.append(123)
        return 123

    receiver = ZmqLatestActionReceiver(
        endpoint,
        context=zmq_context,
        monotonic_ns=monotonic_ns,
    )
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            sender.send(b"not-an-action", flags=zmq.NOBLOCK)
            try:
                receiver.receive_latest(timeout_s=0.01)
            except MalformedAction:
                break
        else:
            pytest.fail("ZeroMQ SUB subscription did not settle")
        assert timestamps == [123]
    finally:
        receiver.close()
        sender.close(linger=0)


def test_receiver_translates_deep_json_recursion_to_malformed_action() -> None:
    context = zmq.Context()
    endpoint = f"inproc://deep-malformed-{uuid.uuid4()}"
    sender = context.socket(zmq.PUB)
    sender.setsockopt(zmq.LINGER, 0)
    sender.bind(endpoint)
    receiver = ZmqLatestActionReceiver(endpoint, context=context)
    header = b'{"header":' + (b"[" * 20_000) + b"0" + (b"]" * 20_000) + b"}"
    packet = struct.pack("!I", len(header)) + header
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            sender.send(packet, flags=zmq.NOBLOCK)
            try:
                receiver.receive_latest(timeout_s=0.01)
            except MalformedAction as error:
                assert isinstance(error.__cause__, RecursionError)
                break
        else:
            pytest.fail("deep malformed packet did not reach receiver")
    finally:
        receiver.close()
        sender.close(linger=0)
        context.term()


def test_receiver_returns_none_when_no_action_is_available(
    zmq_context: zmq.Context,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)
    try:
        assert receiver.receive_latest(timeout_s=0.0) is None
        with pytest.raises(ValueError, match="non-negative"):
            receiver.receive_latest(timeout_s=-0.1)
    finally:
        receiver.close()
        publisher.close()


def test_action_socket_options_are_bounded_and_zero_linger(
    zmq_context: zmq.Context,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)
    try:
        assert publisher._socket.getsockopt(zmq.LINGER) == 0
        assert publisher._socket.getsockopt(zmq.SNDHWM) == 10
        assert receiver._socket.getsockopt(zmq.LINGER) == 0
        assert receiver._socket.getsockopt(zmq.RCVHWM) == 1
        assert receiver._socket.getsockopt(zmq.CONFLATE) == 1
    finally:
        receiver.close()
        publisher.close()


def test_close_is_idempotent_zero_linger_and_rejects_future_operations(
    zmq_context: zmq.Context,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)

    started = time.monotonic()
    publisher.close()
    publisher.close()
    receiver.close()
    receiver.close()

    assert time.monotonic() - started < 0.1
    assert publisher.send(envelope(1)) is False
    assert receiver.receive_latest(timeout_s=0.0) is None


def test_close_serializes_with_send_without_leaking_zmq_errors(
    zmq_context: zmq.Context,
) -> None:
    publisher, _ = open_tcp_publisher(zmq_context)
    sender_started = threading.Event()
    release_sender = threading.Event()
    result: list[bool] = []

    def send_after_close() -> None:
        sender_started.set()
        assert release_sender.wait(timeout=1.0)
        result.append(publisher.send(envelope(1)))

    thread = threading.Thread(target=send_after_close)
    thread.start()
    assert sender_started.wait(timeout=1.0)
    publisher.close()
    release_sender.set()
    thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert result == [False]


def test_receiver_close_wakes_a_waiting_receive(
    zmq_context: zmq.Context,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)
    real_socket = receiver._socket
    polling = threading.Event()
    released = threading.Event()
    result: list[object | None] = []

    class BlockingSocket:
        def poll(self, *, timeout: int, flags: int) -> int:
            del flags
            polling.set()
            released.wait(timeout=timeout / 1_000)
            return 0

        def recv(self, *, flags: int) -> bytes:
            del flags
            raise AssertionError("recv should not be called")

        def close(self, *, linger: int) -> None:
            assert linger == 0
            released.set()

    assert real_socket is not None
    real_socket.close(linger=0)
    receiver._socket = BlockingSocket()  # type: ignore[assignment]

    thread = threading.Thread(target=lambda: result.append(receiver.receive_latest(timeout_s=1.0)))
    thread.start()
    assert polling.wait(timeout=1.0)
    started = time.monotonic()
    receiver.close()
    thread.join(timeout=0.1)

    assert time.monotonic() - started < 0.1
    assert thread.is_alive() is False
    assert result == [None]
    publisher.close()


def test_receiver_close_waits_for_decode_then_discards_inflight_action(
    zmq_context: zmq.Context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, endpoint = open_tcp_publisher(zmq_context)
    receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)
    decode_started = threading.Event()
    release_decode = threading.Event()
    close_returned = threading.Event()
    received: list[ReceivedAction | None] = []
    original_decode = zmq_runtime.decode_action_envelope

    def blocked_decode(packet: bytes) -> ActionEnvelope:
        decode_started.set()
        assert release_decode.wait(timeout=1.0)
        return original_decode(packet)

    try:
        wait_for_subscription(publisher, receiver)
        monkeypatch.setattr(zmq_runtime, "decode_action_envelope", blocked_decode)
        assert publisher.send(envelope(1)) is True
        receiver_thread = threading.Thread(
            target=lambda: received.append(receiver.receive_latest(timeout_s=1.0))
        )
        receiver_thread.start()
        assert decode_started.wait(timeout=1.0)

        closer = threading.Thread(target=lambda: (receiver.close(), close_returned.set()))
        closer.start()
        assert receiver._close_requested.wait(timeout=1.0)
        assert close_returned.is_set() is False

        release_decode.set()
        receiver_thread.join(timeout=1.0)
        closer.join(timeout=1.0)

        assert receiver_thread.is_alive() is False
        assert closer.is_alive() is False
        assert close_returned.is_set() is True
        assert received == [None]
    finally:
        release_decode.set()
        receiver.close()
        publisher.close()


def test_context_owner_does_not_terminate_external_context(zmq_context: zmq.Context) -> None:
    owner = ZmqContextOwner(context=zmq_context)

    owner.close()

    assert zmq_context.closed is False
    probe = zmq_context.socket(zmq.PAIR)
    probe.close(linger=0)


def test_context_owner_terminates_only_its_owned_context() -> None:
    owner = ZmqContextOwner()
    context = owner.context

    owner.close()
    owner.close()

    assert context.closed is True


def test_context_owner_closes_live_owned_action_channels_before_terminating() -> None:
    owner = ZmqContextOwner()
    context = owner.context
    endpoint = f"inproc://owner-live-channels-{uuid.uuid4()}"
    publisher = owner.open_action_publisher(endpoint)
    receiver = owner.open_action_receiver(endpoint)

    owner.close()

    assert context.closed is True
    assert publisher.send(envelope(1)) is False
    assert receiver.receive_latest(timeout_s=0.0) is None
    with pytest.raises(RuntimeError, match="closed"):
        ZmqActionPublisher(f"inproc://owner-direct-rejected-{uuid.uuid4()}", context=context)
    with pytest.raises(RuntimeError, match="closed"):
        owner.open_action_publisher(f"inproc://owner-rejected-{uuid.uuid4()}")


def test_primary_owned_context_rejects_secondary_wrapper_without_hidden_channels() -> None:
    primary = ZmqContextOwner()
    context = primary.context

    with pytest.raises(ValueError, match="canonical owner"):
        ZmqContextOwner(context=context)

    started = time.monotonic()
    primary.close()

    assert time.monotonic() - started < 0.2
    assert context.closed is True
    assert primary.live_channel_count == 0


def test_explicit_adapter_owner_cannot_override_context_canonical_owner() -> None:
    primary = ZmqContextOwner()
    secondary = ZmqContextOwner()
    try:
        with pytest.raises(ValueError, match="canonical owner"):
            ZmqActionPublisher(
                f"inproc://canonical-publisher-{uuid.uuid4()}",
                context=primary.context,
                _owner=secondary,
            )
        with pytest.raises(ValueError, match="canonical owner"):
            ZmqLatestActionReceiver(
                f"inproc://canonical-receiver-{uuid.uuid4()}",
                context=primary.context,
                _owner=secondary,
            )
        assert primary.live_channel_count == 0
    finally:
        primary.close()
        secondary.close()


def test_concurrent_secondary_wrapper_attempts_leave_primary_context_unclaimed() -> None:
    primary = ZmqContextOwner()
    context = primary.context
    start = threading.Barrier(9)
    errors: list[BaseException] = []

    def wrap() -> None:
        try:
            start.wait(timeout=1.0)
            ZmqContextOwner(context=context)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=wrap) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.wait(timeout=1.0)
    for thread in threads:
        thread.join(timeout=0.2)

    assert all(thread.is_alive() is False for thread in threads)
    assert len(errors) == 8
    assert all(isinstance(error, ValueError) for error in errors)
    primary.close()
    assert context.closed is True


def test_owner_tracks_direct_adapters_created_from_its_context() -> None:
    owner = ZmqContextOwner()
    context = owner.context
    endpoint = f"inproc://owner-direct-channels-{uuid.uuid4()}"
    publisher = ZmqActionPublisher(endpoint, context=context)
    receiver = ZmqLatestActionReceiver(endpoint, context=context)
    closed = threading.Event()
    thread = threading.Thread(target=lambda: (owner.close(), closed.set()))
    thread.start()
    thread.join(timeout=0.2)
    completed_boundedly = not thread.is_alive()
    if thread.is_alive():
        receiver.close()
        publisher.close()
        thread.join(timeout=1.0)

    assert completed_boundedly is True
    assert closed.is_set() is True
    assert context.closed is True
    assert publisher.send(envelope(1)) is False
    assert receiver.receive_latest(timeout_s=0.0) is None


def test_independently_closed_direct_adapters_deregister_from_owner() -> None:
    owner = ZmqContextOwner()
    context = owner.context
    for _ in range(20):
        endpoint = f"inproc://owner-deregister-{uuid.uuid4()}"
        publisher = ZmqActionPublisher(endpoint, context=context)
        receiver = ZmqLatestActionReceiver(endpoint, context=context)

        assert owner.live_channel_count == 2
        publisher.close()
        receiver.close()
        assert owner.live_channel_count == 0
    owner.close()


def test_direct_owner_registration_and_close_races_stay_bounded() -> None:
    for _ in range(50):
        owner = ZmqContextOwner()
        start = threading.Barrier(3)
        created: list[ZmqActionPublisher] = []
        errors: list[BaseException] = []

        def construct(
            current_owner: ZmqContextOwner = owner,
            current_start: threading.Barrier = start,
            current_created: list[ZmqActionPublisher] = created,
            current_errors: list[BaseException] = errors,
        ) -> None:
            try:
                current_start.wait(timeout=1.0)
                current_created.append(
                    ZmqActionPublisher(
                        f"inproc://owner-direct-race-{uuid.uuid4()}",
                        context=current_owner.context,
                    )
                )
            except BaseException as caught:
                current_errors.append(caught)

        constructor = threading.Thread(target=construct)
        closer = threading.Thread(
            target=lambda current_start=start, current_owner=owner: (
                current_start.wait(timeout=1.0),
                current_owner.close(),
            )
        )
        constructor.start()
        closer.start()
        start.wait(timeout=1.0)
        constructor.join(timeout=0.2)
        closer.join(timeout=0.2)

        assert constructor.is_alive() is False
        assert closer.is_alive() is False
        assert all(isinstance(error, RuntimeError) for error in errors)
        assert owner.context.closed is True
        assert all(publisher.send(envelope(1)) is False for publisher in created)


def test_external_context_direct_adapters_are_not_owned_by_wrapper() -> None:
    context = zmq.Context()
    owner = ZmqContextOwner(context=context)
    publisher = ZmqActionPublisher(f"inproc://external-direct-{uuid.uuid4()}", context=context)
    try:
        owner.close()

        assert context.closed is False
        assert publisher.send(envelope(1)) is True
        assert owner.live_channel_count == 0
    finally:
        publisher.close()
        context.term()


def test_multiple_nonowning_wrappers_can_share_an_external_context() -> None:
    context = zmq.Context()
    first = ZmqContextOwner(context=context)
    second = ZmqContextOwner(context=context)
    endpoint = f"inproc://multiple-external-wrappers-{uuid.uuid4()}"
    publisher = first.open_action_publisher(endpoint)
    receiver = second.open_action_receiver(endpoint)
    try:
        first.close()
        assert context.closed is False
        assert publisher.send(envelope(1)) is False
        assert receiver.receive_latest(timeout_s=0.0) is None

        second.close()
        assert context.closed is False
    finally:
        receiver.close()
        publisher.close()
        context.term()


def test_context_owner_and_channel_close_concurrently_without_hanging() -> None:
    owner = ZmqContextOwner()
    endpoint = f"inproc://owner-concurrent-close-{uuid.uuid4()}"
    publisher = owner.open_action_publisher(endpoint)
    receiver = owner.open_action_receiver(endpoint)
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def close(callable_: Callable[[], None]) -> None:
        try:
            start.wait(timeout=1.0)
            callable_()
        except BaseException as error:
            errors.append(error)

    owner_thread = threading.Thread(target=lambda: close(owner.close))
    channel_thread = threading.Thread(target=lambda: close(receiver.close))
    owner_thread.start()
    channel_thread.start()
    start.wait(timeout=1.0)
    owner_thread.join(timeout=0.5)
    channel_thread.join(timeout=0.5)

    assert owner_thread.is_alive() is False
    assert channel_thread.is_alive() is False
    assert errors == []
    assert owner.context.closed is True
    assert publisher.send(envelope(1)) is False


def test_action_adapters_repeatedly_open_and_close_under_shared_context(
    zmq_context: zmq.Context,
) -> None:
    for _ in range(3):
        endpoint = f"inproc://action-adapter-{uuid.uuid4()}"
        publisher = ZmqActionPublisher(endpoint, context=zmq_context)
        receiver = ZmqLatestActionReceiver(endpoint, context=zmq_context)
        receiver.close()
        publisher.close()
    assert zmq_context.closed is False


def test_discovery_beacon_is_strict_bounded_and_rejects_unreachable_endpoint() -> None:
    from lekit.control import DiscoveryBeacon

    beacon = DiscoveryBeacon(1, "epoch-1", "tcp://127.0.0.1:5560")
    encoded = beacon.encode()

    assert len(encoded) <= 1_024
    assert DiscoveryBeacon.decode(encoded) == beacon
    with pytest.raises(ValueError, match="endpoint"):
        DiscoveryBeacon(1, "epoch-1", "tcp://0.0.0.0:5560")
    with pytest.raises(ValueError, match="endpoint"):
        DiscoveryBeacon(1, "epoch-1", "tcp://127.0.0.1:0")
    with pytest.raises(ValueError):
        DiscoveryBeacon.decode(b'{"protocol_version":1,"hub_epoch":"epoch-1"}')


def test_discovery_listener_rejects_non_multicast_group() -> None:
    from lekit.control import HubBeaconListener

    with pytest.raises(ValueError, match="multicast"):
        HubBeaconListener("127.0.0.1", 45990)


def test_node_uses_seed_when_listener_has_no_valid_beacon() -> None:
    from lekit.control import ZmqRuntime

    runtime = ZmqRuntime(discovery_enabled=False)
    hub = runtime.open_hub("tcp://127.0.0.1:*", hub_epoch="epoch-1")
    try:
        endpoint = hub.endpoint
        node = runtime.open_node("robot-1", "session-1", hub_seed=endpoint)
        try:
            assert node.send(management_message("register")) is True
            received = hub.receive(timeout_s=1.0)
            assert received is not None
            assert received.peer_id == "robot-1"
            assert received.message.sender_session_id == "session-1"
        finally:
            node.close()
    finally:
        runtime.close()


def test_router_routes_only_current_dealer_session_after_restart() -> None:
    from lekit.control import ZmqRuntime

    runtime = ZmqRuntime(discovery_enabled=False)
    hub = runtime.open_hub("tcp://127.0.0.1:*", hub_epoch="epoch-1")
    try:
        first = runtime.open_node("robot-1", "session-1", hub_seed=hub.endpoint)
        assert first.send(management_message("register")) is True
        assert hub.receive(timeout_s=1.0) is not None
        first.close()

        second = runtime.open_node("robot-1", "session-2", hub_seed=hub.endpoint)
        try:
            assert (
                second.send(management_message("register", sender_session_id="session-2", sequence=2)) is True
            )
            received = hub.receive(timeout_s=1.0)
            assert received is not None
            assert received.peer_id == "robot-1"
            assert received.message.sender_session_id == "session-2"
            assert (
                hub.send(
                    "robot-1",
                    management_message(
                        "registered",
                        sender_id="hub",
                        sender_session_id="hub-session",
                        sequence=3,
                        body={"hub_epoch": "hub-session"},
                    ),
                )
                is True
            )
            reply = second.receive(timeout_s=1.0)
            assert reply is not None
            assert reply.message.kind == "registered"
            assert first.receive(timeout_s=0.0) is None
        finally:
            second.close()
    finally:
        runtime.close()


def test_router_drops_multipart_and_identity_mismatch_without_dispatching() -> None:
    from lekit.control import ZmqHubChannel

    context = zmq.Context()
    hub = ZmqHubChannel("inproc://router-strict", context=context)
    attacker = context.socket(zmq.DEALER)
    attacker.setsockopt(zmq.IDENTITY, b"robot-1/session-1")
    attacker.connect("inproc://router-strict")
    try:
        attacker.send_multipart([b"unexpected", b"extra"])
        assert hub.receive(timeout_s=0.05) is None
        assert hub.malformed_message_count == 1
    finally:
        attacker.close(linger=0)
        hub.close()
        context.term()


def test_router_exposes_loopback_peer_host_from_real_dealer() -> None:
    from lekit.control import ZmqHubChannel, ZmqNodeChannel

    context = zmq.Context()
    hub = ZmqHubChannel("tcp://127.0.0.1:*", context=context)
    node = ZmqNodeChannel(hub.endpoint, "controller-1", "session-1", context=context)
    try:
        assert node.send(management_message("register", sender_id="controller-1"))
        received = hub.receive(timeout_s=1.0)
        assert received is not None
        assert received.peer_host == "127.0.0.1"
    finally:
        node.close()
        hub.close()
        context.term()


def test_router_keeps_new_registered_session_route_when_old_session_sends_later() -> None:
    from lekit.control import ZmqHubChannel, ZmqNodeChannel

    context = zmq.Context()
    hub = ZmqHubChannel("tcp://127.0.0.1:*", context=context)
    old = ZmqNodeChannel(hub.endpoint, "robot-1", "old", context=context)
    new = ZmqNodeChannel(hub.endpoint, "robot-1", "new", context=context)
    try:
        assert old.send(management_message("register", sender_session_id="old"))
        assert hub.receive(timeout_s=1.0) is not None
        assert new.send(management_message("register", sender_session_id="new", sequence=2))
        assert hub.receive(timeout_s=1.0) is not None
        assert old.send(management_message("heartbeat", sender_session_id="old", sequence=3))
        assert hub.receive(timeout_s=1.0) is not None
        assert hub.send(
            "robot-1",
            management_message(
                "registered",
                sender_id="hub",
                sender_session_id="epoch",
                sequence=4,
                body={"hub_epoch": "epoch"},
            ),
        )
        assert new.receive(timeout_s=1.0) is not None
        assert old.receive(timeout_s=0.05) is None
    finally:
        old.close()
        new.close()
        hub.close()
        context.term()


def test_node_rejects_spoofed_or_wrong_epoch_hub_message() -> None:
    from lekit.control import ZmqNodeChannel, encode_management

    context = zmq.Context()
    endpoint = f"inproc://node-hub-auth-{uuid.uuid4()}"
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)
    node = ZmqNodeChannel(endpoint, "robot-1", "session-1", context=context)
    try:
        assert node.send(management_message("register"))
        identity, _payload = router.recv_multipart()
        router.send_multipart(
            [
                identity,
                encode_management(
                    management_message(
                        "registered",
                        sender_id="hub",
                        sender_session_id="epoch-1",
                        body={"hub_epoch": "epoch-1"},
                    )
                ),
            ]
        )
        assert node.receive(timeout_s=1.0) is not None
        router.send_multipart(
            [
                identity,
                encode_management(
                    management_message("force_hold", sender_id="intruder", sender_session_id="x")
                ),
            ]
        )
        assert node.receive(timeout_s=0.05) is None
        router.send_multipart(
            [
                identity,
                encode_management(
                    management_message("force_hold", sender_id="hub", sender_session_id="epoch-2", sequence=2)
                ),
            ]
        )
        assert node.receive(timeout_s=0.05) is None
        assert node.malformed_message_count == 2
    finally:
        node.close()
        router.close(linger=0)
        context.term()


def test_discovery_defaults_loopback_and_accepts_only_current_nonblank_beacon() -> None:
    from lekit.control import DiscoveryBeacon, HubBeaconListener, HubBeaconPublisher

    publisher = HubBeaconPublisher()
    try:
        assert publisher.loopback is True
    finally:
        publisher.close()
    assert (
        HubBeaconListener.decode_datagram(DiscoveryBeacon(1, "epoch-1", "tcp://127.0.0.1:5560").encode())
        is not None
    )
    assert (
        HubBeaconListener.decode_datagram(DiscoveryBeacon(2, "epoch-1", "tcp://127.0.0.1:5560").encode())
        is None
    )
    with pytest.raises(ValueError):
        DiscoveryBeacon(1, "  ", "tcp://127.0.0.1:5560")


def test_wildcard_implicit_advertisement_disables_discovery_and_failed_open_rolls_back() -> None:
    from lekit.control import ZmqRuntime

    runtime = ZmqRuntime()
    try:
        hub = runtime.open_hub("tcp://0.0.0.0:*", hub_epoch="epoch-1")
        assert hub.discovery_enabled is False
        hub.close()
        with pytest.raises(ValueError):
            runtime.open_hub(
                "tcp://127.0.0.1:*", hub_epoch="epoch-1", advertise_endpoint="tcp://0.0.0.0:5560"
            )
        assert runtime.live_channel_count == 0
    finally:
        runtime.close()


def test_dead_seed_loses_to_current_beacon_and_reconnect_uses_changed_endpoint() -> None:
    from lekit.control import DiscoveryBeacon, ZmqRuntime

    context = zmq.Context()
    first = context.socket(zmq.ROUTER)
    second = context.socket(zmq.ROUTER)
    first.setsockopt(zmq.LINGER, 0)
    second.setsockopt(zmq.LINGER, 0)
    first.bind("tcp://127.0.0.1:*")
    second.bind("tcp://127.0.0.1:*")
    endpoints = iter(
        [
            DiscoveryBeacon(1, "epoch-1", first.getsockopt(zmq.LAST_ENDPOINT).decode()),
            DiscoveryBeacon(1, "epoch-2", second.getsockopt(zmq.LAST_ENDPOINT).decode()),
        ]
    )

    class Listener:
        cooperative_discovery_listener = True

        def receive(self, *, timeout_s: float) -> object:
            if timeout_s == 0.0:
                return None
            assert timeout_s >= 1.0
            return next(endpoints)

        def close(self) -> None:
            pass

    runtime = ZmqRuntime(
        context=context,
        discovery_listener_factory=lambda: Listener(),
        seed_ready=lambda _endpoint: False,
    )
    try:
        node = runtime.open_node("robot-1", "session-1", hub_seed="tcp://127.0.0.1:1")
        assert node.endpoint == first.getsockopt(zmq.LAST_ENDPOINT).decode()
        node.rediscover()
        assert node.endpoint == second.getsockopt(zmq.LAST_ENDPOINT).decode()
        node.close()
        assert runtime.live_channel_count == 0
    finally:
        runtime.close()
        first.close(linger=0)
        second.close(linger=0)
        context.term()


def test_default_seed_probe_does_not_select_unreachable_seed() -> None:
    from lekit.control import DiscoveryBeacon, ZmqRuntime

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind("tcp://127.0.0.1:*")
    endpoint = router.getsockopt(zmq.LAST_ENDPOINT).decode()

    class Listener:
        cooperative_discovery_listener = True

        def receive(self, *, timeout_s: float) -> object:
            if timeout_s == 0.0:
                return None
            assert timeout_s >= 1.0
            return DiscoveryBeacon(1, "epoch-1", endpoint)

        def close(self) -> None:
            pass

    runtime = ZmqRuntime(context=context, discovery_listener_factory=lambda: Listener())
    try:
        node = runtime.open_node("robot-1", "session-1", hub_seed="tcp://127.0.0.1:1")
        assert node.endpoint == endpoint
        node.close()
    finally:
        runtime.close()
        router.close(linger=0)
        context.term()


def test_current_beacon_wins_when_it_arrived_before_ready_seed() -> None:
    from lekit.control import DiscoveryBeacon, ZmqRuntime

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind("tcp://127.0.0.1:*")
    endpoint = router.getsockopt(zmq.LAST_ENDPOINT).decode()

    class Listener:
        cooperative_discovery_listener = True

        def receive(self, *, timeout_s: float) -> object:
            assert timeout_s >= 1.0
            return DiscoveryBeacon(1, "epoch-1", endpoint)

        def close(self) -> None:
            pass

    runtime = ZmqRuntime(
        context=context,
        discovery_listener_factory=lambda: Listener(),
        cooperative_seed_probe=lambda _endpoint, _cancelled: True,
    )
    try:
        node = runtime.open_node("robot-1", "session-1", hub_seed="tcp://127.0.0.1:1")
        assert node.endpoint == endpoint
        node.close()
    finally:
        runtime.close()
        router.close(linger=0)
        context.term()


def test_node_monitor_rediscovery_replaces_disconnected_management_endpoint() -> None:
    from lekit.control import ZmqRuntime

    context = zmq.Context()
    first = context.socket(zmq.ROUTER)
    second = context.socket(zmq.ROUTER)
    first.setsockopt(zmq.LINGER, 0)
    second.setsockopt(zmq.LINGER, 0)
    first.bind("tcp://127.0.0.1:*")
    second.bind("tcp://127.0.0.1:*")
    first_endpoint = first.getsockopt(zmq.LAST_ENDPOINT).decode()
    second_endpoint = second.getsockopt(zmq.LAST_ENDPOINT).decode()
    runtime = ZmqRuntime(context=context, discovery_enabled=False)
    try:
        node = runtime.open_node("robot-1", "session-1", hub_seed=first_endpoint)
        node._set_resolver(lambda: (second_endpoint, "epoch-2"))
        assert node.send(management_message("register"))
        assert first.recv_multipart()
        first.close(linger=0)
        node._monitor_disconnected_locked = lambda: True  # type: ignore[method-assign]
        node.receive(timeout_s=0.0)
        assert node.endpoint == second_endpoint
        node.close()
    finally:
        runtime.close()
        second.close(linger=0)
        context.term()


def test_runtime_deregisters_closed_management_and_action_channels() -> None:
    from lekit.control import ZmqRuntime

    runtime = ZmqRuntime(discovery_enabled=False)
    hub = runtime.open_hub("tcp://127.0.0.1:*", hub_epoch="epoch-1")
    node = runtime.open_node("robot-1", "session-1", hub_seed=hub.endpoint)
    publisher = runtime.open_action_publisher(f"inproc://runtime-deregister-{uuid.uuid4()}")
    try:
        assert runtime.live_channel_count == 3
        node.close()
        publisher.close()
        hub.close()
        assert runtime.live_channel_count == 0
    finally:
        runtime.close()


def test_runtime_rejects_non_bool_discovery_flags_and_blank_hub_epoch() -> None:
    from lekit.control import ZmqRuntime

    with pytest.raises(ValueError):
        ZmqRuntime(discovery_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ZmqRuntime(discovery_loopback=0)  # type: ignore[arg-type]
    runtime = ZmqRuntime(discovery_enabled=False)
    try:
        with pytest.raises(ValueError):
            runtime.open_hub("tcp://127.0.0.1:*", hub_epoch=" ")
    finally:
        runtime.close()


def test_resolver_beacon_wins_while_seed_probe_is_still_blocked() -> None:
    from lekit.control import DiscoveryBeacon, ZmqRuntime

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind("tcp://127.0.0.1:*")
    endpoint = router.getsockopt(zmq.LAST_ENDPOINT).decode()
    seed_started = threading.Event()
    release_seed = threading.Event()

    class Listener:
        cooperative_discovery_listener = True

        def receive(self, *, timeout_s: float) -> object:
            assert seed_started.wait(timeout=1.0)
            return DiscoveryBeacon(1, "epoch-1", endpoint)

        def close(self) -> None:
            pass

    def slow_seed(_endpoint: str, cancelled: threading.Event) -> bool:
        seed_started.set()
        while not release_seed.is_set() and not cancelled.wait(timeout=0.01):
            pass
        return True

    runtime = ZmqRuntime(
        context=context,
        discovery_listener_factory=lambda: Listener(),
        cooperative_seed_probe=slow_seed,
    )
    try:
        node = runtime.open_node("robot-1", "session-1", hub_seed="tcp://127.0.0.1:1")
        assert node.endpoint == endpoint
        release_seed.set()
        node.close()
    finally:
        release_seed.set()
        runtime.close()
        router.close(linger=0)
        context.term()


def test_runtime_rejects_noncooperative_listener_before_starting_resolver_worker() -> None:
    from lekit.control import ZmqRuntime

    class NonCooperativeListener:
        def receive(self, *, timeout_s: float) -> None:
            return None

        def close(self) -> None:
            pass

    runtime = ZmqRuntime(
        discovery_listener_factory=lambda: NonCooperativeListener(),
        seed_ready=lambda _endpoint: False,
    )
    try:
        for _ in range(3):
            with pytest.raises(ValueError, match="cooperative discovery listener"):
                runtime.open_node("robot-1", "session-1", hub_seed="tcp://127.0.0.1:1")
        assert not [thread for thread in threading.enumerate() if thread.name == "lekit-beacon-resolver"]
    finally:
        runtime.close()


def test_unregistered_node_rejects_all_hub_commands_then_registered_pins_identity() -> None:
    from lekit.control import ZmqNodeChannel, encode_management

    context = zmq.Context()
    endpoint = f"inproc://prepin-{uuid.uuid4()}"
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)
    node = ZmqNodeChannel(endpoint, "robot-1", "session-1", context=context)
    try:
        assert node.send(management_message("register"))
        identity, _payload = router.recv_multipart()
        for sequence in range(100):
            router.send_multipart(
                [
                    identity,
                    encode_management(
                        management_message(
                            "force_hold", sender_id="hub", sender_session_id=f"fake-{sequence}"
                        )
                    ),
                ]
            )
            assert node.receive(timeout_s=0.05) is None
        assert node.malformed_message_count == 100
        router.send_multipart(
            [
                identity,
                encode_management(
                    management_message(
                        "registered",
                        sender_id="hub",
                        sender_session_id="epoch-1",
                        body={"hub_epoch": "epoch-1"},
                    )
                ),
            ]
        )
        assert node.receive(timeout_s=1.0) is not None
        router.send_multipart(
            [
                identity,
                encode_management(
                    management_message(
                        "force_hold", sender_id="hub", sender_session_id="epoch-1", sequence=101
                    )
                ),
            ]
        )
        assert node.receive(timeout_s=1.0) is not None
    finally:
        node.close()
        router.close(linger=0)
        context.term()


def test_rediscovery_same_endpoint_new_epoch_resets_registration_pin() -> None:
    from lekit.control import ZmqNodeChannel

    context = zmq.Context()
    endpoint = f"inproc://same-endpoint-{uuid.uuid4()}"
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)
    node = ZmqNodeChannel(endpoint, "robot-1", "session-1", context=context, hub_epoch="epoch-1")
    try:
        node._registered = True
        node._set_resolver(lambda: (endpoint, "epoch-2"))
        assert node.rediscover() is True
        assert node._hub_epoch == "epoch-2"
        assert node._registered is False
    finally:
        node.close()
        router.close(linger=0)
        context.term()


def test_beacon_start_failure_closes_publisher_and_releases_hub_channel() -> None:
    from lekit.control import ZmqRuntime

    class FailingPublisher:
        closed = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self, _beacon: object) -> None:
            raise RuntimeError("start failed")

        def close(self) -> None:
            type(self).closed = True

    runtime = ZmqRuntime(beacon_publisher_factory=FailingPublisher)
    try:
        with pytest.raises(RuntimeError, match="start failed"):
            runtime.open_hub(
                "tcp://127.0.0.1:*", hub_epoch="epoch-1", advertise_endpoint="tcp://127.0.0.1:5560"
            )
        assert FailingPublisher.closed is True
        assert runtime.live_channel_count == 0
    finally:
        runtime.close()


def test_listener_drops_oversize_beacon_without_parsing() -> None:
    from lekit.control import HubBeaconListener

    assert HubBeaconListener.decode_datagram(b"x" * 1_025) is None


def test_management_router_uses_bounded_options_and_captures_receive_timestamp() -> None:
    from lekit.control import ZmqHubChannel, ZmqNodeChannel

    context = zmq.Context()
    hub = ZmqHubChannel("tcp://127.0.0.1:*", context=context, monotonic_ns=lambda: 4242)
    node = ZmqNodeChannel(hub.endpoint, "robot-1", "session-1", context=context)
    try:
        assert hub._socket is not None
        assert hub._socket.getsockopt(zmq.LINGER) == 0
        assert hub._socket.getsockopt(zmq.SNDHWM) == 256
        assert hub._socket.getsockopt(zmq.RCVHWM) == 256
        assert node.send(management_message("register"))
        received = hub.receive(timeout_s=1.0)
        assert received is not None
        assert received.received_monotonic_ns == 4242
    finally:
        node.close()
        hub.close()
        context.term()


def test_router_mandatory_rejects_unknown_or_disconnected_route() -> None:
    from lekit.control import ZmqHubChannel

    context = zmq.Context()
    hub = ZmqHubChannel("tcp://127.0.0.1:*", context=context)
    try:
        assert (
            hub.send("missing", management_message("registered", sender_id="hub", sender_session_id="epoch"))
            is False
        )
    finally:
        hub.close()
        context.term()


@pytest.mark.parametrize(
    "endpoint",
    ["tcp://127.0.0.1:5560", "tcp://[::1]:5560", "tcp://hub.example.com:5560"],
)
def test_management_endpoint_accepts_strict_host_matrix(endpoint: str) -> None:
    from lekit.control.discovery import validate_management_endpoint

    assert validate_management_endpoint(endpoint) == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://0.0.0:5560",
        "tcp://0x00000000:5560",
        "tcp://0x7f.0x0.0x0.0x1:5560",
        "tcp://0.0.0.0.:5560",
        "tcp://127.0.0.1.:5560",
        "tcp://[127.0.0.1]:5560",
        "tcp://[::]:5560",
        "tcp://[::ffff:0.0.0.0]:5560",
        "tcp://user@host:5560",
        "tcp://hub.example.com.:5560",
    ],
)
def test_management_endpoint_rejects_alias_and_noncanonical_hosts(endpoint: str) -> None:
    from lekit.control.discovery import validate_management_endpoint

    with pytest.raises(ValueError):
        validate_management_endpoint(endpoint)


def test_beacon_known_epoch_rejects_forged_bootstrap_then_accepts_match() -> None:
    from lekit.control import ZmqNodeChannel, encode_management

    context = zmq.Context()
    endpoint = f"inproc://known-epoch-{uuid.uuid4()}"
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)
    node = ZmqNodeChannel(endpoint, "robot-1", "session-1", context=context, hub_epoch="beacon-epoch")
    try:
        assert node.send(management_message("register"))
        identity, _ = router.recv_multipart()
        for sequence in range(100):
            router.send_multipart(
                [
                    identity,
                    encode_management(
                        management_message(
                            "registered",
                            sender_id="hub",
                            sender_session_id="forged",
                            sequence=sequence,
                            body={"hub_epoch": "forged"},
                        )
                    ),
                ]
            )
            assert node.receive(timeout_s=0.01) is None
        router.send_multipart(
            [
                identity,
                encode_management(
                    management_message(
                        "registered",
                        sender_id="hub",
                        sender_session_id="beacon-epoch",
                        sequence=101,
                        body={"hub_epoch": "beacon-epoch"},
                    )
                ),
            ]
        )
        assert node.receive(timeout_s=1.0) is not None
    finally:
        node.close()
        router.close(linger=0)
        context.term()
