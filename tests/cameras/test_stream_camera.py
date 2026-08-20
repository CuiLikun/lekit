from __future__ import annotations

import json
import socket
import time

import numpy as np
import pytest
import zmq

from lekit.cameras.stream import StreamCamera, StreamCameraConfig
from lerobot.cameras import make_cameras_from_configs


def _available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Publisher:
    def __init__(self, camera_name: str = "front") -> None:
        self.camera_name = camera_name
        self.port = _available_port()
        self.endpoint = f"tcp://127.0.0.1:{self.port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.endpoint)

    def send(self, image: np.ndarray, *, sequence: int = 1, codec: str = "raw_bgr8") -> None:
        height, width, _ = image.shape
        if codec == "raw_bgr8":
            payload = image.tobytes()
        elif codec == "jpeg":
            cv2 = pytest.importorskip("cv2")
            success, encoded = cv2.imencode(".jpg", image)
            assert success
            payload = encoded.tobytes()
        else:
            raise ValueError(f"unsupported test codec: {codec}")
        header = {
            "schema_version": 1,
            "camera": self.camera_name,
            "stream": "color",
            "sequence": sequence,
            "captured_monotonic_ns": time.monotonic_ns(),
            "captured_utc_ns": time.time_ns(),
            "timestamp_source": "host",
            "width": width,
            "height": height,
            "pixel_format": "bgr8",
            "codec": codec,
            "payload_size": len(payload),
        }
        self.socket.send_multipart(
            [f"{self.camera_name}/color".encode(), json.dumps(header).encode(), payload]
        )

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()


@pytest.fixture
def publisher():
    resource = Publisher()
    try:
        yield resource
    finally:
        resource.close()


def _connect_after_subscription(
    camera: StreamCamera,
    publisher: Publisher,
    image: np.ndarray | None = None,
    *,
    codec: str = "raw_bgr8",
) -> None:
    camera.connect(warmup=False)
    # PUB/SUB subscriptions propagate asynchronously; sending briefly avoids a timing-dependent test.
    image = np.array([[[3, 2, 1]]], dtype=np.uint8) if image is None else image
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        publisher.send(image, codec=codec)
        try:
            camera.read_latest(max_age_ms=100)
        except RuntimeError:
            time.sleep(0.01)
        else:
            return
    pytest.fail("camera did not receive a subscribed frame")


def _read_published_frame(camera: StreamCamera, publisher: Publisher, image: np.ndarray) -> np.ndarray:
    """Publish until the receiver consumes the requested frame, ignoring setup leftovers."""
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        publisher.send(image, sequence=2)
        try:
            frame = camera.async_read(timeout_ms=100)
        except TimeoutError:
            continue
        if np.array_equal(frame, image[..., ::-1]):
            return frame
    pytest.fail("camera did not consume the expected published frame")


def _drain_unread_frames(camera: StreamCamera) -> None:
    while True:
        try:
            camera.async_read(timeout_ms=50)
        except TimeoutError:
            return


def test_raw_stream_frames_follow_lerobot_read_contract(publisher: Publisher) -> None:
    camera = StreamCamera(StreamCameraConfig(endpoint=publisher.endpoint, camera_name="front"))
    try:
        _connect_after_subscription(camera, publisher)
        assert camera.width == 1
        assert camera.height == 1
        assert camera.read_latest().tolist() == [[[1, 2, 3]]]

        target = np.array([[[8, 7, 6]]], dtype=np.uint8)
        assert _read_published_frame(camera, publisher, target).tolist() == [[[6, 7, 8]]]
        _drain_unread_frames(camera)
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=20)
        assert camera.read_latest().tolist() == [[[6, 7, 8]]]
    finally:
        camera.disconnect()


def test_bgr_output_stale_frames_and_factory(publisher: Publisher) -> None:
    config = StreamCameraConfig(
        endpoint=publisher.endpoint,
        camera_name="front",
        width=1,
        height=1,
        color_mode="bgr",
    )
    camera = make_cameras_from_configs({"front": config})["front"]
    assert isinstance(camera, StreamCamera)
    try:
        _connect_after_subscription(camera, publisher)
        assert camera.read_latest().tolist() == [[[3, 2, 1]]]
        time.sleep(0.02)
        with pytest.raises(TimeoutError, match="too old"):
            camera.read_latest(max_age_ms=1)
    finally:
        camera.disconnect()


def test_jpeg_stream_frames_are_decoded_when_opencv_is_installed(publisher: Publisher) -> None:
    pytest.importorskip("cv2")
    image = np.full((8, 8, 3), [30, 20, 10], dtype=np.uint8)
    camera = StreamCamera(StreamCameraConfig(endpoint=publisher.endpoint, camera_name="front"))
    try:
        _connect_after_subscription(camera, publisher, image, codec="jpeg")
        np.testing.assert_allclose(camera.read_latest(), image[..., ::-1], atol=3)
    finally:
        camera.disconnect()


def test_config_rejects_invalid_stream_settings() -> None:
    with pytest.raises(ValueError, match="TCP URL"):
        StreamCameraConfig(endpoint="http://localhost:5555", camera_name="front")
    with pytest.raises(ValueError, match="camera_name"):
        StreamCameraConfig(endpoint="tcp://localhost:5555", camera_name="front/color")
    with pytest.raises(ValueError, match="width.*height"):
        StreamCameraConfig(endpoint="tcp://localhost:5555", camera_name="front", width=640)
