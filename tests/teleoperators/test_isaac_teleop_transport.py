from __future__ import annotations

import socket
import time

import pytest
import zmq

from lekit.teleoperators.isaac_teleop.protocol import TeleopFrame, neutral_action
from lekit.teleoperators.isaac_teleop.transport import ZmqTeleopPublisher, ZmqTeleopReceiver


def available_endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


def frame(sequence: int, session_id: str = "session-1") -> TeleopFrame:
    action = neutral_action()
    action["right.trigger"] = sequence / 100.0
    return TeleopFrame(session_id, sequence, sequence + 1, sequence + 2, action)


def establish_subscription(publisher: ZmqTeleopPublisher, receiver: ZmqTeleopReceiver) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        publisher.publish_action(frame(1))
        received = receiver.receive_latest(timeout_s=0.05)
        if received is not None:
            return
    pytest.fail("ZeroMQ subscription was not established")


def test_late_subscriber_receives_only_frames_published_after_it_connects() -> None:
    endpoint = available_endpoint()
    with ZmqTeleopPublisher(endpoint) as publisher:
        publisher.publish_action(frame(1))
        with ZmqTeleopReceiver(endpoint) as receiver:
            assert receiver.receive_latest(timeout_s=0.05) is None
            establish_subscription(publisher, receiver)
            assert receiver.receive_latest(timeout_s=0.0) is None


def test_receiver_conflates_a_burst_to_the_newest_action_frame() -> None:
    endpoint = available_endpoint()
    with ZmqTeleopPublisher(endpoint) as publisher, ZmqTeleopReceiver(endpoint) as receiver:
        establish_subscription(publisher, receiver)
        for sequence in range(2, 51):
            publisher.publish_action(frame(sequence))

        deadline = time.monotonic() + 2.0
        received = None
        while time.monotonic() < deadline:
            publisher.publish_action(frame(50))
            candidate = receiver.receive_latest(timeout_s=0.05)
            if candidate is not None:
                received = candidate
            if received is not None and received.sequence == 50:
                break

        assert received is not None
        assert received.sequence == 50
        assert received.action["right.trigger"] == pytest.approx(0.5)


def test_action_receiver_ignores_status_topic() -> None:
    endpoint = available_endpoint()
    with ZmqTeleopPublisher(endpoint) as publisher, ZmqTeleopReceiver(endpoint) as receiver:
        establish_subscription(publisher, receiver)

        publisher.publish_status({"state": "waiting_for_headset"})

        assert receiver.receive_latest(timeout_s=0.05) is None


def test_transport_closes_without_lingering_and_rejects_further_use() -> None:
    context = zmq.Context()
    endpoint = available_endpoint()
    publisher = ZmqTeleopPublisher(endpoint, context=context)
    receiver = ZmqTeleopReceiver(endpoint, context=context)

    receiver.close()
    publisher.close()
    context.term()

    with pytest.raises(RuntimeError, match="closed"):
        receiver.receive_latest()
    with pytest.raises(RuntimeError, match="closed"):
        publisher.publish_action(frame(1))
