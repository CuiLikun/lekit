from __future__ import annotations

import threading
import time
from queue import Empty, Full
from socket import AF_INET, SOCK_STREAM, socket

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import lekit.control.video as video_module
from lekit.control.model import CameraStreamDescriptor
from lekit.control.video import LatestJpegStore, RobotVideoServer, RobotVideoServerConfig


@pytest.fixture
def video_server() -> RobotVideoServer:
    server = RobotVideoServer(
        RobotVideoServerConfig(
            cameras=(
                CameraStreamDescriptor(
                    name="HAND_VIEW",
                    stream_url="http://robot.example:8081/api/cameras/HAND_VIEW/stream.mjpg",
                    width=640,
                    height=480,
                    fps=30.0,
                ),
            )
        )
    )
    yield server
    server.stop()


def _available_loopback_port() -> int:
    with socket(AF_INET, SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _server(*, port: int | None = None, encoder=None) -> RobotVideoServer:
    port = _available_loopback_port() if port is None else port
    return RobotVideoServer(
        RobotVideoServerConfig(
            cameras=(
                CameraStreamDescriptor(
                    name="HAND_VIEW",
                    stream_url=f"http://127.0.0.1:{port}/api/cameras/HAND_VIEW/stream.mjpg",
                    width=1,
                    height=1,
                    fps=30.0,
                ),
            ),
            host="127.0.0.1",
            port=port,
        ),
        encoder=encoder,
    )


def test_publish_replaces_pending_frame_without_waiting() -> None:
    entered = threading.Event()
    release = threading.Event()
    latest_entered = threading.Event()
    release_latest = threading.Event()

    def encoder(image: np.ndarray) -> bytes:
        value = int(image[0, 0, 0])
        if value == 1:
            entered.set()
            assert release.wait(1)
        if value == 3:
            latest_entered.set()
            assert release_latest.wait(1)
        return bytes([value])

    store = LatestJpegStore(("HAND_VIEW",), encoder=encoder)
    try:
        store.publish({"HAND_VIEW": np.full((1, 1, 3), 1, np.uint8)}, captured_monotonic_ns=1)
        assert entered.wait(1)

        store.publish({"HAND_VIEW": np.full((1, 1, 3), 2, np.uint8)}, captured_monotonic_ns=2)
        store.publish({"HAND_VIEW": np.full((1, 1, 3), 3, np.uint8)}, captured_monotonic_ns=3)
        release.set()

        first = store.wait_encoded("HAND_VIEW", after_sequence=-1, timeout_s=1)
        assert first is not None
        assert first.jpeg == b"\x01"
        assert latest_entered.wait(1)
        release_latest.set()
        latest = store.wait_encoded("HAND_VIEW", after_sequence=first.sequence, timeout_s=1)
        assert latest is not None
        assert latest.jpeg == b"\x03"
        assert latest.captured_monotonic_ns == 3
    finally:
        release.set()
        release_latest.set()
        store.close()


def test_publish_keeps_newest_frame_when_worker_dequeues_during_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_queue = video_module.queue.Queue

    class DequeueBetweenFullAndReplacementQueue(original_queue):
        def __init__(self, maxsize: int = 0) -> None:
            super().__init__(maxsize=maxsize)
            self.armed = threading.Event()
            self.worker_took = threading.Event()

        def put_nowait(self, item) -> None:
            try:
                original_queue.put(self, item, block=False)
            except Full:
                self.armed.set()
                raise

        def get(self, block: bool = True, timeout: float | None = None):
            if threading.current_thread().name.startswith("robot-video-encoder"):
                if not self.armed.wait(timeout):
                    raise Empty
                item = original_queue.get(self, block=False)
                self.worker_took.set()
                return item
            return original_queue.get(self, block=block, timeout=timeout)

        def get_nowait(self):
            if threading.current_thread().name.startswith("robot-video-encoder"):
                if not self.armed.is_set():
                    raise Empty
                item = original_queue.get(self, block=False)
                self.worker_took.set()
                return item
            assert self.armed.is_set()
            self.worker_took.wait(0.1)
            return original_queue.get(self, block=False)

    monkeypatch.setattr(video_module.queue, "Queue", DequeueBetweenFullAndReplacementQueue)
    store = LatestJpegStore(("HAND_VIEW",), encoder=lambda image: bytes([int(image[0, 0, 0])]))
    try:
        store.publish({"HAND_VIEW": np.full((1, 1, 3), 2, np.uint8)}, captured_monotonic_ns=2)
        store.publish({"HAND_VIEW": np.full((1, 1, 3), 3, np.uint8)}, captured_monotonic_ns=3)

        frame = store.wait_encoded("HAND_VIEW", after_sequence=-1, timeout_s=1)

        assert frame is not None
        assert frame.jpeg == b"\x03"
        assert frame.captured_monotonic_ns == 3
    finally:
        store.close()


def test_stop_cancels_active_mjpeg_stream_and_joins_http_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    server = _server()

    def slow_wait_encoded(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return None

    monkeypatch.setattr(server._store, "wait_encoded", slow_wait_encoded)
    server.start()
    stop_finished = threading.Event()
    stop_errors: list[BaseException] = []

    def stop_server() -> None:
        try:
            server.stop()
        except BaseException as error:
            stop_errors.append(error)
        finally:
            stop_finished.set()

    try:
        with httpx.Client(timeout=2.0) as client, client.stream(
            "GET",
            f"http://127.0.0.1:{server.config.port}/api/cameras/HAND_VIEW/stream.mjpg",
        ) as response:
            assert response.status_code == 200
            assert entered.wait(1)
            stopper = threading.Thread(target=stop_server)
            stopper.start()

            assert stop_finished.wait(1.5)
            stopper.join(timeout=0.1)
            assert not stop_errors
            assert server._thread is None
            assert not any(
                thread.name == "robot-video-http" and thread.is_alive()
                for thread in threading.enumerate()
            )
    finally:
        release.set()
        if "stopper" in locals():
            stopper.join(timeout=4)
        server.stop()


def test_stop_reports_blocked_encoder_then_joins_after_release() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_encoder(_image: np.ndarray) -> bytes:
        entered.set()
        assert release.wait(5)
        return b"jpeg"

    server = _server(encoder=blocking_encoder)
    worker = server._store._stores["HAND_VIEW"]._worker
    try:
        server.publish({"HAND_VIEW": np.zeros((1, 1, 3), np.uint8)}, captured_monotonic_ns=1)
        assert entered.wait(1)

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="encoder worker"):
            server.stop()

        assert time.monotonic() - started < 3.5
        assert worker.is_alive()
        release.set()
        server.stop()
        assert not worker.is_alive()
    finally:
        release.set()
        server.stop()


def test_video_app_exposes_only_read_routes_and_configured_metadata(video_server: RobotVideoServer) -> None:
    client = TestClient(video_server.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/cameras").json() == {
        "cameras": [
            {
                "name": "HAND_VIEW",
                "stream_url": "http://robot.example:8081/api/cameras/HAND_VIEW/stream.mjpg",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "health": {"state": "waiting"},
            }
        ]
    }
    assert client.post("/api/cameras").status_code == 405
    assert client.put("/health").status_code == 405
    assert video_server.describe() == video_server.config.cameras


def test_video_status_allows_read_only_hub_cross_origin_access(
    video_server: RobotVideoServer,
) -> None:
    response = TestClient(video_server.app).get(
        "/api/cameras",
        headers={"Origin": "http://robot.example:8080"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
