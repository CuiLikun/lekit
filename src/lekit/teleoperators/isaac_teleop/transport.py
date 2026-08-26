"""Bounded ZeroMQ transport for independent Isaac teleop processes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import zmq

from .protocol import TeleopFrame, decode_action_frame, encode_action_frame

ACTION_TOPIC = b"isaac_teleop/action/v1"
STATUS_TOPIC = b"isaac_teleop/status/v1"
_TOPIC_SEPARATOR = b" "


class ZmqTeleopPublisher:
    """Non-blocking PUB socket used by the sole hardware-owning process."""

    def __init__(self, endpoint: str, *, context: zmq.Context | None = None) -> None:
        self.endpoint = endpoint
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket: zmq.Socket | None = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 10)
        try:
            self._socket.bind(endpoint)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> ZmqTeleopPublisher:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def publish_action(self, frame: TeleopFrame) -> bool:
        """Publish one atomic action frame, dropping it instead of blocking."""

        return self._send(ACTION_TOPIC, encode_action_frame(frame))

    def publish_status(self, status: Mapping[str, Any]) -> bool:
        """Publish one JSON diagnostic status snapshot."""

        payload = json.dumps(dict(status), separators=(",", ":"), allow_nan=False).encode("utf-8")
        return self._send(STATUS_TOPIC, payload)

    def _send(self, topic: bytes, payload: bytes) -> bool:
        socket = self._require_socket()
        try:
            socket.send(topic + _TOPIC_SEPARATOR + payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

    def close(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close(linger=0)
        if self._owns_context and not self._context.closed:
            self._context.term()

    def _require_socket(self) -> zmq.Socket:
        if self._socket is None:
            raise RuntimeError("teleop publisher is closed")
        return self._socket


class ZmqTeleopReceiver:
    """SUB socket that retains only the latest complete action message."""

    def __init__(self, endpoint: str, *, context: zmq.Context | None = None) -> None:
        self.endpoint = endpoint
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket: zmq.Socket | None = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, ACTION_TOPIC + _TOPIC_SEPARATOR)
        try:
            self._socket.connect(endpoint)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> ZmqTeleopReceiver:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def receive_latest(self, *, timeout_s: float = 0.0) -> TeleopFrame | None:
        """Return the newest available frame, or ``None`` before the timeout."""

        if timeout_s < 0.0:
            raise ValueError("timeout_s must be non-negative")
        socket = self._require_socket()
        timeout_ms = max(0, int(timeout_s * 1_000))
        if not socket.poll(timeout=timeout_ms, flags=zmq.POLLIN):
            return None
        message = socket.recv()
        prefix = ACTION_TOPIC + _TOPIC_SEPARATOR
        if not message.startswith(prefix):
            return None
        return decode_action_frame(message[len(prefix) :])

    def close(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close(linger=0)
        if self._owns_context and not self._context.closed:
            self._context.term()

    def _require_socket(self) -> zmq.Socket:
        if self._socket is None:
            raise RuntimeError("teleop receiver is closed")
        return self._socket


__all__ = [
    "ACTION_TOPIC",
    "STATUS_TOPIC",
    "ZmqTeleopPublisher",
    "ZmqTeleopReceiver",
]
