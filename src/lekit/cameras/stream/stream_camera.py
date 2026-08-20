"""Expose ``camera-stream`` resources through the LeRobot camera API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from camera_stream import CameraStream, StreamClient
from numpy.typing import NDArray

from lerobot.cameras import Camera, CameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError


@CameraConfig.register_subclass("stream")
@dataclass
class StreamCameraConfig(CameraConfig):
    """Configuration for one ``camera-stream`` ``<camera>/color`` topic.

    Args:
        endpoint: Public camera-stream PUB endpoint, for example
            ``tcp://192.168.5.24:5555``.
        camera_name: Server-side camera name. The subscribed topic is always
            ``<camera_name>/color``.
        color_mode: Output channel order. camera-stream transports BGR pixels;
            RGB is the LeRobot-compatible default.
        timeout_ms: Default timeout for :meth:`read`.
        warmup_s: Maximum time ``connect(warmup=True)`` waits for the first
            valid image.
    """

    endpoint: str
    camera_name: str
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 5_000
    warmup_s: float = 5.0

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        _validate_endpoint(self.endpoint)
        if not self.camera_name or "/" in self.camera_name:
            raise ValueError("`camera_name` must be a non-empty camera name without '/'.")
        if self.timeout_ms <= 0:
            raise ValueError("`timeout_ms` must be positive.")
        if self.warmup_s < 0:
            raise ValueError("`warmup_s` must be non-negative.")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("`fps` must be positive when set.")
        if self.width is not None and self.width <= 0:
            raise ValueError("`width` must be positive when set.")
        if self.height is not None and self.height <= 0:
            raise ValueError("`height` must be positive when set.")
        if (self.width is None) != (self.height is None):
            raise ValueError("`width` and `height` must either both be set or both be omitted.")


class StreamCamera(Camera):
    """Thin LeRobot adapter around :class:`camera_stream.StreamClient`."""

    def __init__(self, config: StreamCameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self.endpoint = config.endpoint
        self.camera_name = config.camera_name
        self.topic = f"{self.camera_name}/color"
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms
        self._client: StreamClient | None = None
        self._stream: CameraStream | None = None

    def __str__(self) -> str:
        return f"StreamCamera({self.topic}@{self.endpoint})"

    @property
    def is_connected(self) -> bool:
        """Whether the camera-stream subscription is open."""
        return self._client is not None and self._stream is not None and not self._stream.is_closed

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Remote topics require an endpoint and cannot be found globally."""
        raise NotImplementedError("Configure `endpoint` and `camera_name` for camera-stream cameras.")

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Create a ``StreamClient`` subscription and optionally await its first frame."""
        client = StreamClient(self.endpoint)
        try:
            stream = client.subscribe(
                self.topic,
                warm_up=warmup,
                warm_up_timeout=self.config.warmup_s if warmup else None,
            )
            frame = stream.last_frame
            if frame is not None:
                self._validate_dimensions(frame.image)
        except Exception:
            client.close()
            raise
        self._client = client
        self._stream = stream

    @check_if_not_connected
    def read(self) -> NDArray[np.uint8]:
        """Wait for and consume the newest unread frame using the configured timeout."""
        return self.async_read(timeout_ms=self.timeout_ms)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[np.uint8]:
        """Wait for and consume the newest unread frame from ``camera-stream``."""
        if timeout_ms < 0:
            raise ValueError("`timeout_ms` must be non-negative.")
        stream = self._require_stream()
        frame = stream.read(timeout=timeout_ms / 1_000)
        if frame is None:
            raise RuntimeError(f"{self} did not return a frame.")
        return self._convert_frame(frame.image)

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.uint8]:
        """Return the newest buffered frame without consuming it."""
        if max_age_ms < 0:
            raise ValueError("`max_age_ms` must be non-negative.")
        stream = self._require_stream()
        frame = stream.read(block=False)
        if frame is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.monotonic_ns() - frame.received_monotonic_ns) / 1_000_000
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return self._convert_frame(frame.image)

    def disconnect(self) -> None:
        """Close the ``camera-stream`` client and all of its subscriptions."""
        if self._client is None and self._stream is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        client, self._client = self._client, None
        self._stream = None
        if client is not None:
            client.close()

    def _require_stream(self) -> CameraStream:
        if self._stream is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._stream

    def _convert_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        self._validate_dimensions(frame)
        return frame[..., ::-1].copy() if self.color_mode == ColorMode.RGB else frame

    def _validate_dimensions(self, frame: NDArray[np.uint8]) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"{self} received invalid image shape {frame.shape}; expected (height, width, 3)."
            )
        height, width = frame.shape[:2]
        if self.width is None:
            self.width, self.height = width, height
        elif (self.width, self.height) != (width, height):
            raise RuntimeError(
                f"{self} frame dimensions {width}x{height} do not match configured {self.width}x{self.height}."
            )


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"`endpoint` has an invalid port: {endpoint!r}") from exc
    if parsed.scheme != "tcp" or not parsed.hostname or port is None:
        raise ValueError("`endpoint` must be a TCP URL such as `tcp://192.168.5.24:5555`.")
