"""Transport-only runtime interfaces for Control Hub channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model import ActionEnvelope, ManagementMessage


@dataclass(frozen=True, slots=True)
class ReceivedManagement:
    """A management message received from a named peer."""

    peer_id: str
    message: ManagementMessage
    peer_host: str | None
    received_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class ReceivedAction:
    """An opaque action envelope received from an action endpoint."""

    envelope: ActionEnvelope
    received_monotonic_ns: int


class HubChannel(Protocol):
    """Management channel owned by the hub."""

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        """Receive the next management message, if available."""
        raise NotImplementedError

    def send(self, peer_id: str, message: ManagementMessage) -> bool:
        """Send a management message to one peer without waiting."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the channel and wake waiting receivers."""
        raise NotImplementedError


class NodeChannel(Protocol):
    """Management channel owned by one node."""

    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        """Receive the next management message, if available."""
        raise NotImplementedError

    def send(self, message: ManagementMessage) -> bool:
        """Send a management message to the hub without waiting."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the channel and wake waiting receivers."""
        raise NotImplementedError


class ActionPublisher(Protocol):
    """Non-blocking publisher for opaque action envelopes."""

    def send(self, envelope: ActionEnvelope) -> bool:
        """Publish an action envelope without waiting."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the publisher."""
        raise NotImplementedError


class LatestActionReceiver(Protocol):
    """Receiver that consumes only the latest unread action envelope."""

    def receive_latest(self, *, timeout_s: float = 0.0) -> ReceivedAction | None:
        """Receive the latest unread action envelope, if available."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the receiver and wake waiting callers."""
        raise NotImplementedError


class Runtime(Protocol):
    """Factory for management and action transport channels."""

    def open_hub(
        self,
        endpoint: str,
        *,
        hub_epoch: str,
        advertise_endpoint: str | None = None,
    ) -> HubChannel:
        """Open the hub management channel."""
        raise NotImplementedError

    def open_node(self, node_id: str, session_id: str, *, hub_seed: str | None) -> NodeChannel:
        """Open a node management channel."""
        raise NotImplementedError

    def open_action_publisher(self, endpoint: str) -> ActionPublisher:
        """Open an action publisher for an endpoint."""
        raise NotImplementedError

    def open_action_receiver(self, endpoint: str) -> LatestActionReceiver:
        """Open a latest-only action receiver for an endpoint."""
        raise NotImplementedError

    def close(self) -> None:
        """Close all runtime channels."""
        raise NotImplementedError
