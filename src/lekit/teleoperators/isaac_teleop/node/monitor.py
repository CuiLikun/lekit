"""Thread-safe service state and read-only Web monitor for teleop-node."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from ..protocol import TeleopFrame, neutral_action, normalize_action

_MAX_MONITOR_RATE_HZ = 60.0
_MONITOR_HEARTBEAT_S = 1.0


@dataclass(frozen=True)
class TeleopNodeSnapshot:
    """One immutable monitoring snapshot."""

    state: str
    session_id: str
    sequence: int | None
    sampled_frames: int
    published_frames: int
    dropped_frames: int
    publish_rate_hz: float
    uptime_s: float
    uptime: str
    started_utc_ns: int
    last_frame_utc_ns: int | None
    last_frame_age_ms: float | None
    last_error: str | None
    publish_endpoint: str
    monitor_url: str
    action: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "sampled_frames": self.sampled_frames,
            "published_frames": self.published_frames,
            "dropped_frames": self.dropped_frames,
            "publish_rate_hz": self.publish_rate_hz,
            "uptime_s": self.uptime_s,
            "uptime": self.uptime,
            "started_utc_ns": self.started_utc_ns,
            "last_frame_utc_ns": self.last_frame_utc_ns,
            "last_frame_age_ms": self.last_frame_age_ms,
            "last_error": self.last_error,
            "publish_endpoint": self.publish_endpoint,
            "monitor_url": self.monitor_url,
            "action": self.action,
        }


class TeleopNodeState:
    """Mutable service state shared by the sampling and HTTP threads."""

    def __init__(
        self,
        *,
        session_id: str,
        publish_endpoint: str,
        monitor_url: str,
        monotonic: Callable[[], float] = time.monotonic,
        utc_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._session_id = session_id
        self.publish_endpoint = publish_endpoint
        self.monitor_url = monitor_url
        self._monotonic = monotonic
        self._utc_ns = utc_ns
        self._lock = threading.Lock()
        self._started_at = monotonic()
        self._started_utc_ns = utc_ns()
        self._state = "starting"
        self._sequence: int | None = None
        self._sampled_frames = 0
        self._published_frames = 0
        self._dropped_frames = 0
        self._publish_rate_hz = 0.0
        self._last_frame_utc_ns: int | None = None
        self._last_frame_received_at: float | None = None
        self._last_error: str | None = None
        self._action = neutral_action()
        self._revision = 0

    def set_state(self, state: str, *, error: str | None = None) -> None:
        """Update lifecycle state and its current fault detail."""

        if not state:
            raise ValueError("state must not be empty")
        with self._lock:
            self._state = state
            self._last_error = error
            self._revision += 1

    def begin_session(self, session_id: str) -> None:
        """Invalidate the previous XR session while preserving process totals."""

        if not session_id:
            raise ValueError("session_id must not be empty")
        with self._lock:
            self._session_id = session_id
            self._sequence = None
            self._publish_rate_hz = 0.0
            self._last_frame_utc_ns = None
            self._last_frame_received_at = None
            self._action = neutral_action()
            self._revision += 1

    def record_frame(
        self,
        frame: TeleopFrame,
        *,
        publish_rate_hz: float,
        published: bool = True,
    ) -> None:
        """Record one sampled frame and whether the transport accepted it."""

        action = normalize_action(frame.action)
        with self._lock:
            self._sequence = frame.sequence
            self._sampled_frames += 1
            if published:
                self._published_frames += 1
            else:
                self._dropped_frames += 1
            self._publish_rate_hz = round(float(publish_rate_hz), 3)
            self._last_frame_utc_ns = frame.captured_utc_ns
            self._last_frame_received_at = self._monotonic()
            self._action = action
            self._revision += 1

    def snapshot(self) -> TeleopNodeSnapshot:
        return self.versioned_snapshot()[1]

    def versioned_snapshot(self) -> tuple[int, TeleopNodeSnapshot]:
        """Return one atomic latest-value snapshot and its change revision."""

        now = self._monotonic()
        with self._lock:
            age_ms = (
                None
                if self._last_frame_received_at is None
                else round((now - self._last_frame_received_at) * 1_000.0, 3)
            )
            action = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in normalize_action(self._action).items()
            }
            snapshot = TeleopNodeSnapshot(
                state=self._state,
                session_id=self._session_id,
                sequence=self._sequence,
                sampled_frames=self._sampled_frames,
                published_frames=self._published_frames,
                dropped_frames=self._dropped_frames,
                publish_rate_hz=self._publish_rate_hz,
                uptime_s=round(now - self._started_at, 3),
                uptime=_format_duration(now - self._started_at),
                started_utc_ns=self._started_utc_ns,
                last_frame_utc_ns=self._last_frame_utc_ns,
                last_frame_age_ms=age_ms,
                last_error=self._last_error,
                publish_endpoint=self.publish_endpoint,
                monitor_url=self.monitor_url,
                action=action,
            )
            return self._revision, snapshot

    def snapshot_dict(self) -> dict[str, Any]:
        return self.snapshot().to_dict()


def create_monitor_app(state: TeleopNodeState) -> FastAPI:
    """Create a read-only FastAPI application over one state store."""

    app = FastAPI(title="Isaac Teleop Node Monitor", docs_url=None, redoc_url=None)
    asset_directory = Path(__file__).parent / "assets"
    vendor_directory = asset_directory / "vendor"
    html = asset_directory.joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(html)

    @app.get("/assets/teleop_node_monitor.js", response_class=FileResponse)
    def monitor_script() -> FileResponse:
        return FileResponse(
            asset_directory / "app.js",
            media_type="text/javascript",
        )

    @app.get("/assets/teleop_node_monitor_trajectory.js", response_class=FileResponse)
    def monitor_trajectory_script() -> FileResponse:
        return FileResponse(
            asset_directory / "trajectory.js",
            media_type="text/javascript",
        )

    @app.get("/assets/three.module.min.js", response_class=FileResponse)
    def three_module() -> FileResponse:
        return FileResponse(
            vendor_directory / "three.module.min.js",
            media_type="text/javascript",
        )

    @app.get("/assets/three.core.min.js", response_class=FileResponse)
    def three_core_module() -> FileResponse:
        return FileResponse(
            vendor_directory / "three.core.min.js",
            media_type="text/javascript",
        )

    @app.get("/assets/OrbitControls.js", response_class=FileResponse)
    def orbit_controls_module() -> FileResponse:
        return FileResponse(
            vendor_directory / "OrbitControls.js",
            media_type="text/javascript",
        )

    @app.get("/assets/three.LICENSE.txt", response_class=FileResponse)
    def three_license() -> FileResponse:
        return FileResponse(
            vendor_directory / "three.LICENSE.txt",
            media_type="text/plain",
        )

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return state.snapshot_dict()

    @app.websocket("/api/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        period_s = 1.0 / _MAX_MONITOR_RATE_HZ
        last_revision = -1
        last_sent_at = float("-inf")
        try:
            while True:
                revision, snapshot = state.versioned_snapshot()
                now = asyncio.get_running_loop().time()
                if revision != last_revision or now - last_sent_at >= _MONITOR_HEARTBEAT_S:
                    await websocket.send_json(snapshot.to_dict())
                    last_revision = revision
                    last_sent_at = now
                await asyncio.sleep(period_s)
        except WebSocketDisconnect:
            pass

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        snapshot = state.snapshot()
        return {"ok": snapshot.state != "stopped", "state": snapshot.state}

    return app


def _format_duration(duration_s: float) -> str:
    total_seconds = max(0, int(duration_s))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


__all__ = ["TeleopNodeSnapshot", "TeleopNodeState", "create_monitor_app"]
