"""Read-only, latest-frame MJPEG video owned by a Robot process."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from numbers import Integral, Real
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from lerobot.types import RobotObservation

from .model import CameraStreamDescriptor

JpegEncoder = Callable[[Any], bytes]


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    """One immutable, encoded image retained for browser clients."""

    sequence: int
    captured_monotonic_ns: int
    encoded_monotonic_ns: int
    width: int
    height: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class RobotVideoServerConfig:
    """Public stream descriptors and local listener settings for Robot video."""

    cameras: tuple[CameraStreamDescriptor, ...]
    host: str = "127.0.0.1"
    port: int = 8081

    def __post_init__(self) -> None:
        if not isinstance(self.cameras, tuple) or not all(
            isinstance(camera, CameraStreamDescriptor) for camera in self.cameras
        ):
            raise ValueError("cameras must be a tuple of CameraStreamDescriptor")
        if len({camera.name for camera in self.cameras}) != len(self.cameras):
            raise ValueError("camera names must be unique")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must not be empty")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer from 1 through 65535")


class _CameraStore:
    """One capacity-one work queue and latest encoded frame for a camera."""

    def __init__(self, name: str, encoder: JpegEncoder, monotonic_ns: Callable[[], int]) -> None:
        self.name = name
        self._encoder = encoder
        self._monotonic_ns = monotonic_ns
        self._pending: queue.Queue[tuple[Any, int]] = queue.Queue(maxsize=1)
        self._mailbox_lock = threading.Lock()
        self._closed = threading.Event()
        self._condition = threading.Condition()
        self._latest: EncodedFrame | None = None
        self._next_sequence = 0
        self._last_error: str | None = None
        self._worker = threading.Thread(
            target=self._run,
            name=f"robot-video-encoder-{name}",
            daemon=True,
        )
        self._worker.start()

    def publish(self, image: Any, captured_monotonic_ns: int) -> None:
        if self._closed.is_set():
            return
        self._validate_image(image)
        work = (image, captured_monotonic_ns)
        with self._mailbox_lock:
            if self._closed.is_set():
                return
            try:
                self._pending.put_nowait(work)
                return
            except queue.Full:
                self._pending.get_nowait()
                self._pending.put_nowait(work)

    def wait_encoded(self, after_sequence: int, timeout_s: float) -> EncodedFrame | None:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("after_sequence must be an integer")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real) or timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while not self._closed.is_set():
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def health(self) -> dict[str, Any]:
        with self._condition:
            if self._last_error is not None:
                return {"state": "error", "error": self._last_error}
            if self._latest is None:
                return {"state": "closed" if self._closed.is_set() else "waiting"}
            frame = self._latest
            return {
                "state": "live",
                "sequence": frame.sequence,
                "captured_monotonic_ns": frame.captured_monotonic_ns,
                "encoded_monotonic_ns": frame.encoded_monotonic_ns,
                "width": frame.width,
                "height": frame.height,
            }

    def close(self) -> None:
        self._closed.set()
        with self._condition:
            self._condition.notify_all()

    def join(self, timeout_s: float) -> bool:
        self._worker.join(timeout=timeout_s)
        return not self._worker.is_alive()

    def _run(self) -> None:
        while not self._closed.is_set():
            work = self._take_pending()
            if work is None:
                self._closed.wait(0.01)
                continue
            image, captured_monotonic_ns = work
            try:
                jpeg = self._encoder(image)
            except Exception as error:
                with self._condition:
                    self._last_error = f"{type(error).__name__}: {error}"
                    self._condition.notify_all()
                continue
            height, width = image.shape[:2]
            with self._condition:
                self._latest = EncodedFrame(
                    sequence=self._next_sequence,
                    captured_monotonic_ns=captured_monotonic_ns,
                    encoded_monotonic_ns=self._monotonic_ns(),
                    width=width,
                    height=height,
                    jpeg=jpeg,
                )
                self._next_sequence += 1
                self._last_error = None
                self._condition.notify_all()

    def _take_pending(self) -> tuple[Any, int] | None:
        with self._mailbox_lock:
            try:
                return self._pending.get_nowait()
            except queue.Empty:
                return None

    @staticmethod
    def _validate_image(image: Any) -> None:
        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", None)
        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, Integral) for value in shape)
            or shape[0] < 1
            or shape[1] < 1
            or shape[2] != 3
            or str(dtype) != "uint8"
        ):
            raise ValueError("camera images must be uint8 HWC arrays with three RGB channels")


class LatestJpegStore:
    """Offer named RGB observations to independent capacity-one encoders."""

    def __init__(
        self,
        camera_names: tuple[str, ...],
        *,
        encoder: JpegEncoder | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(camera_names, tuple) or not camera_names or not all(
            isinstance(name, str) and name for name in camera_names
        ):
            raise ValueError("camera_names must be a non-empty tuple of non-empty strings")
        if len(set(camera_names)) != len(camera_names):
            raise ValueError("camera_names must be unique")
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable")
        self._closed = threading.Event()
        self._stores = {
            name: _CameraStore(name, encoder or _encode_rgb_jpeg, monotonic_ns) for name in camera_names
        }

    def publish(self, observation: RobotObservation, *, captured_monotonic_ns: int) -> None:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        if (
            isinstance(captured_monotonic_ns, bool)
            or not isinstance(captured_monotonic_ns, int)
            or captured_monotonic_ns < 0
        ):
            raise ValueError("captured_monotonic_ns must be a non-negative integer")
        if self._closed.is_set():
            return
        for name, store in self._stores.items():
            image = observation.get(name)
            if image is not None:
                store.publish(image, captured_monotonic_ns)

    def wait_encoded(self, name: str, *, after_sequence: int, timeout_s: float) -> EncodedFrame | None:
        return self._store(name).wait_encoded(after_sequence, timeout_s)

    def health(self, name: str) -> dict[str, Any]:
        return self._store(name).health()

    def close(self) -> None:
        self._closed.set()
        for store in self._stores.values():
            store.close()

    def join(self, timeout_s: float) -> bool:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real) or timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + float(timeout_s)
        joined = True
        for store in self._stores.values():
            joined = store.join(max(0.0, deadline - time.monotonic())) and joined
        return joined

    def is_closed(self) -> bool:
        return self._closed.is_set()

    def _store(self, name: str) -> _CameraStore:
        try:
            return self._stores[name]
        except KeyError as error:
            raise KeyError(f"unknown camera {name!r}") from error


class RobotVideoServer:
    """Bounded owner for read-only Robot video HTTP and MJPEG delivery."""

    def __init__(self, config: RobotVideoServerConfig, *, encoder: JpegEncoder | None = None) -> None:
        if not isinstance(config, RobotVideoServerConfig):
            raise TypeError("config must be a RobotVideoServerConfig")
        self.config = config
        self._store = LatestJpegStore(tuple(camera.name for camera in config.cameras), encoder=encoder)
        self._server: Any | None = None
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()
        self._app = self._create_app()

    @property
    def app(self) -> FastAPI:
        """Expose the application for local ASGI composition and tests."""
        return self._app

    def start(self) -> None:
        """Serve the read-only application without entering the Robot control loop."""
        with self._lifecycle_lock:
            if self._thread is not None:
                return
            if self._store.is_closed():
                raise RuntimeError("Robot video server cannot restart after stop")
            import uvicorn

            self._server = uvicorn.Server(
                uvicorn.Config(
                    self._app,
                    host=self.config.host,
                    port=self.config.port,
                    log_level="warning",
                    access_log=False,
                    timeout_graceful_shutdown=1,
                )
            )
            self._thread = threading.Thread(target=self._server.run, name="robot-video-http", daemon=True)
            self._thread.start()
            deadline = time.monotonic() + 3.0
            while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not self._server.started:
                self.stop()
                raise RuntimeError("Robot video server did not start")

    def publish(self, observation: RobotObservation, *, captured_monotonic_ns: int) -> None:
        """Non-blockingly replace each configured camera's pending frame."""
        self._store.publish(observation, captured_monotonic_ns=captured_monotonic_ns)

    def describe(self) -> tuple[CameraStreamDescriptor, ...]:
        """Return immutable public camera metadata for Robot registration."""
        return self.config.cameras

    def stop(self) -> None:
        """Close frame work, signal Uvicorn, and wait at most three seconds."""
        deadline = time.monotonic() + 3.0
        with self._lifecycle_lock:
            self._store.close()
            server, thread = self._server, self._thread
            errors: list[str] = []
            if server is not None and thread is not None:
                server.should_exit = True
                if thread is threading.current_thread():
                    errors.append("Robot video server cannot join its own HTTP thread")
                else:
                    thread.join(timeout=min(1.5, max(0.0, deadline - time.monotonic())))
                    if thread.is_alive():
                        server.force_exit = True
                        thread.join(timeout=max(0.0, deadline - time.monotonic()))
                    if thread.is_alive():
                        errors.append("Robot video server did not stop within 3 seconds")
                    else:
                        self._server = None
                        self._thread = None
            if not self._store.join(max(0.0, deadline - time.monotonic())):
                errors.append("Robot video encoder worker did not stop within 3 seconds")
            if errors:
                raise RuntimeError("; ".join(errors))

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Lekit Robot Video", docs_url=None, redoc_url=None)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/api/cameras")
        def cameras() -> JSONResponse:
            return JSONResponse(
                content={
                    "cameras": [
                        {**asdict(camera), "health": self._store.health(camera.name)}
                        for camera in self.config.cameras
                    ]
                }
            )

        @app.get("/api/cameras/{name}/stream.mjpg")
        async def stream(name: str) -> StreamingResponse:
            try:
                self._store.health(name)
            except KeyError as error:
                raise HTTPException(status_code=404, detail="unknown camera") from error

            async def frames():
                sequence = -1
                while not self._store.is_closed():
                    frame = await anyio.to_thread.run_sync(
                        partial(self._store.wait_encoded, name, after_sequence=sequence, timeout_s=0.25),
                        abandon_on_cancel=True,
                    )
                    if frame is None:
                        continue
                    sequence = frame.sequence
                    yield _multipart_frame(frame.jpeg)

            return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

        return app


def _encode_rgb_jpeg(image: Any) -> bytes:
    """Encode one RGB uint8 frame as browser-compatible JPEG outside Robot locks."""
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return bytes(encoded)


def _multipart_frame(jpeg: bytes) -> bytes:
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(jpeg)).encode()
        + b"\r\n\r\n"
        + jpeg
        + b"\r\n"
    )
