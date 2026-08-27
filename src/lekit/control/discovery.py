"""Strict UDP multicast discovery for Control Hub management endpoints."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from .model import PROTOCOL_VERSION

DEFAULT_MULTICAST_GROUP = "239.255.42.99"
DEFAULT_DISCOVERY_PORT = 45990
_MAX_DATAGRAM_BYTES = 1_024
_BEACON_FIELDS = frozenset({"protocol_version", "hub_epoch", "management_endpoint"})
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_NUMERIC_ALIAS = re.compile(r"(?:0[xX][0-9A-Fa-f]+|0[0-7]+|[0-9]+)$")


def _validate_group(group: str) -> str:
    if not isinstance(group, str):
        raise ValueError("multicast group must be an IPv4 multicast address")
    try:
        address = ipaddress.IPv4Address(group)
    except ipaddress.AddressValueError as error:
        raise ValueError("multicast group must be an IPv4 multicast address") from error
    if not address.is_multicast:
        raise ValueError("multicast group must be an IPv4 multicast address")
    return group


def _validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise ValueError("discovery port must be an integer in [1, 65535]")
    return port


def validate_management_endpoint(endpoint: str) -> str:
    """Accept only reachable, non-wildcard TCP management endpoints."""

    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("management endpoint must be a non-wildcard TCP endpoint")
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as error:
        raise ValueError("management endpoint must be a non-wildcard TCP endpoint") from error
    if (
        parsed.scheme != "tcp"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or port is None
        or not 1 <= port <= 65_535
    ):
        raise ValueError("management endpoint must be a non-wildcard TCP endpoint")
    host = parsed.hostname
    bracketed = parsed.netloc.startswith("[")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if bracketed or all(_NUMERIC_ALIAS.fullmatch(label) for label in host.split(".")):
            raise ValueError("management endpoint must be a non-wildcard TCP endpoint") from None
        if (
            host in {"*", "0"}
            or host.isdigit()
            or host.endswith(".")
            or len(host) > 253
            or any(not _DNS_LABEL.fullmatch(label) for label in host.rstrip(".").split("."))
        ):
            raise ValueError("management endpoint must be a non-wildcard TCP endpoint") from None
    else:
        if (
            address.is_unspecified
            or (isinstance(address, ipaddress.IPv4Address) and str(address) != host)
            or (
                isinstance(address, ipaddress.IPv6Address)
                and address.ipv4_mapped is not None
                and address.ipv4_mapped.is_unspecified
            )
        ):
            raise ValueError("management endpoint must be a non-wildcard TCP endpoint")
    if host in {"*", "0"}:
        raise ValueError("management endpoint must be a non-wildcard TCP endpoint")
    return endpoint


@dataclass(frozen=True, slots=True)
class DiscoveryBeacon:
    """The complete, versioned UDP discovery payload."""

    protocol_version: int
    hub_epoch: str
    management_endpoint: str

    def __post_init__(self) -> None:
        if isinstance(self.protocol_version, bool) or not isinstance(self.protocol_version, int):
            raise ValueError("protocol version must be an integer")
        if not isinstance(self.hub_epoch, str) or not self.hub_epoch.strip():
            raise ValueError("hub epoch must not be empty")
        validate_management_endpoint(self.management_endpoint)

    def encode(self) -> bytes:
        payload = json.dumps(
            {
                "protocol_version": self.protocol_version,
                "hub_epoch": self.hub_epoch,
                "management_endpoint": self.management_endpoint,
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(payload) > _MAX_DATAGRAM_BYTES:
            raise ValueError("discovery beacon exceeds 1 KiB")
        return payload

    @classmethod
    def decode(cls, payload: bytes) -> DiscoveryBeacon:
        if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_DATAGRAM_BYTES:
            raise ValueError("discovery beacon payload is invalid")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("discovery beacon payload is invalid") from error
        if not isinstance(value, dict) or set(value) != _BEACON_FIELDS:
            raise ValueError("discovery beacon fields are invalid")
        return cls(**value)


class HubBeaconPublisher:
    """Low-rate multicast beacon publisher with explicit loopback control."""

    def __init__(
        self,
        group: str = DEFAULT_MULTICAST_GROUP,
        port: int = DEFAULT_DISCOVERY_PORT,
        *,
        rate_hz: float = 1.0,
        loopback: bool = True,
    ) -> None:
        self._group = _validate_group(group)
        self._port = _validate_port(port)
        if (
            isinstance(rate_hz, bool)
            or not isinstance(rate_hz, (int, float))
            or not math.isfinite(float(rate_hz))
            or rate_hz <= 0
        ):
            raise ValueError("discovery rate must be positive")
        self._interval_s = 1.0 / float(rate_hz)
        if not isinstance(loopback, bool):
            raise ValueError("discovery loopback must be a bool")
        self._loopback = loopback
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, int(self._loopback))
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def loopback(self) -> bool:
        """Whether host-local multicast loopback is enabled."""

        return self._loopback

    def publish(self, beacon: DiscoveryBeacon) -> bool:
        if self._closed.is_set() or beacon.protocol_version != PROTOCOL_VERSION:
            return False
        try:
            self._socket.sendto(beacon.encode(), (self._group, self._port))
        except OSError:
            return False
        return True

    def start(self, beacon: DiscoveryBeacon) -> None:
        if self._thread is not None:
            raise RuntimeError("discovery publisher is already running")

        def run() -> None:
            while not self._closed.is_set():
                self.publish(beacon)
                self._closed.wait(self._interval_s)

        self._thread = threading.Thread(target=run, name="lekit-hub-discovery", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self._interval_s + 0.1))


class CooperativeDiscoveryListener(Protocol):
    """Contract for injected listeners used by the resolver worker.

    Implementations declare this marker only when ``receive(timeout_s=...)``
    honours its timeout and ``close()`` unblocks an in-flight receive.
    """

    cooperative_discovery_listener: Literal[True]

    def receive(self, *, timeout_s: float = 0.0) -> DiscoveryBeacon | None: ...

    def close(self) -> None: ...


class HubBeaconListener(CooperativeDiscoveryListener):
    """Multicast listener; malformed and incompatible datagrams are ignored."""

    cooperative_discovery_listener = True

    def __init__(
        self,
        group: str = DEFAULT_MULTICAST_GROUP,
        port: int = DEFAULT_DISCOVERY_PORT,
        *,
        loopback: bool = True,
        interface: str = "0.0.0.0",
    ) -> None:
        self._group = _validate_group(group)
        self._port = _validate_port(port)
        if not isinstance(loopback, bool):
            raise ValueError("discovery loopback must be a bool")
        self._closed = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("" if loopback else interface, self._port))
        membership = socket.inet_aton(self._group) + socket.inet_aton(interface)
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)

    @staticmethod
    def decode_datagram(payload: bytes) -> DiscoveryBeacon | None:
        try:
            beacon = DiscoveryBeacon.decode(payload)
        except ValueError:
            return None
        return beacon if beacon.protocol_version == PROTOCOL_VERSION else None

    def receive(self, *, timeout_s: float = 0.0) -> DiscoveryBeacon | None:
        if self._closed:
            return None
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be a finite non-negative number")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_s must be a finite non-negative number")
        deadline = time.monotonic() + timeout
        while not self._closed:
            remaining = max(0.0, deadline - time.monotonic())
            self._socket.settimeout(remaining)
            try:
                payload, _address = self._socket.recvfrom(_MAX_DATAGRAM_BYTES + 1)
            except (BlockingIOError, TimeoutError, OSError):
                return None
            beacon = self.decode_datagram(payload)
            if beacon is not None:
                return beacon
            if remaining == 0.0:
                return None
        return None

    def close(self) -> None:
        self._closed = True
        self._socket.close()


__all__ = [
    "DEFAULT_DISCOVERY_PORT",
    "DEFAULT_MULTICAST_GROUP",
    "CooperativeDiscoveryListener",
    "DiscoveryBeacon",
    "HubBeaconListener",
    "HubBeaconPublisher",
    "validate_management_endpoint",
]
