"""WebSocket-backed browser debugger for a :class:`JakaRobot`.

Run from the repository root with::

    uv run python -m robots.jaka_robot.web_debug --ip 192.168.1.31

The server binds to all local interfaces by default. It does not connect,
power, enable, or start Servo Move until an operator presses the corresponding
control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from collections import deque
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .jaka_robot import JakaRobot, JakaRobotConfig

_JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
_EEF_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
_CONTROL_KEYS = (*_JOINT_KEYS, *_EEF_KEYS, "gripper.pos")
_Command = Literal["connection", "power", "enable", "servo", "abort"]
_DASHBOARD_PATH = Path(__file__).with_name("web_debug.html")


class JakaWebDebugger:
    """Own one robot and serialize all browser-triggered hardware operations."""

    def __init__(self, robot: JakaRobot) -> None:
        self.robot = robot
        self._lock = threading.RLock()
        self._motion_lock = threading.Lock()
        self._console_events: deque[dict[str, Any]] = deque(maxlen=120)
        self._next_event_id = 1
        self._last_servo_telemetry_at = 0.0
        self._last_servo_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return a compact, JSON-safe hardware snapshot for one browser frame."""

        with self._lock:
            connected = self.robot.is_connected
            controller = {
                "connected": connected,
                "powered_on": False,
                "enabled": False,
                "servo_active": False,
            }
            observation: dict[str, float] = {}
            servo_status = self.robot.get_servo_status()
            if connected:
                state = self.robot.get_controller_state()
                controller.update(
                    powered_on=state["powered_on"],
                    enabled=state["enabled"],
                    servo_active=servo_status["active"],
                )
                values = self.robot.get_observation()
                observation = {
                    key: float(values[key])
                    for key in _CONTROL_KEYS
                    if key in values and isinstance(values[key], (int, float, np.number))
                }
            self._record_servo_status(servo_status)
            return {
                "timestamp_ns": time.monotonic_ns(),
                "controller": controller,
                "observation": observation,
                "servo": servo_status,
                "limits": self._limits(),
                "console": list(self._console_events),
            }

    def command(self, command: _Command) -> dict[str, Any]:
        """Execute one lifecycle command and return the resulting state frame."""

        with self._lock:
            self._record("info", f"Command requested: {command}")
            try:
                if command == "connection":
                    if self.robot.is_connected:
                        self.robot.disconnect()
                    else:
                        self.robot.connect()
                elif command == "power":
                    self._require_connection()
                    state = self.robot.get_controller_state()
                    if state["powered_on"]:
                        if self.robot.servo_active:
                            self.robot.servo_enable(False)
                        if state["enabled"]:
                            self.robot.disable_robot()
                        self.robot.power_off()
                    else:
                        self.robot.power_on()
                elif command == "enable":
                    self._require_connection()
                    if self.robot.get_controller_state()["enabled"]:
                        self.robot.disable_robot()
                    else:
                        self.robot.enable_robot()
                elif command == "servo":
                    self._require_connection()
                    self.robot.servo_enable(not self.robot.servo_active)
                elif command == "abort":
                    self._require_connection()
                    self.robot.motion_abort()
                else:
                    raise ValueError(f"unsupported command: {command}")
            except Exception as exc:
                self._record("error", f"Command {command} failed: {exc}")
                raise
            self._record("info", f"Command completed: {command}")
        return self.snapshot()

    def relative_action(self, key: str, delta: Any) -> dict[str, Any]:
        """Send one bounded relative action and return the resulting state frame."""

        if key not in _CONTROL_KEYS:
            raise ValueError(f"unsupported control field: {key}")
        with self._motion_lock:
            try:
                with self._lock:
                    use_servo = self.robot.servo_active
                    mode = "Servo Move" if use_servo else "controller-planned move"
                    self._record("info", f"Relative action requested: {key} {float(delta):+.5f}")
                    self._record("info", f"Action mode selected: {mode}")
                    applied = self.robot.send_relative_action({key: delta}, use_servo=use_servo)
                    target = float(applied[key])
                    self._record("info", f"Applied target: {key}={target:.5f}")

                if use_servo and key != "gripper.pos":
                    with self._lock:
                        self._record("info", f"Servo target updated: {key}={target:.5f}")
                elif key != "gripper.pos":
                    with self._lock:
                        self._record("info", f"Controller-planned move completed: {key}={target:.5f}")
            except Exception as exc:
                with self._lock:
                    self._record("error", f"Relative action failed for {key}: {exc}")
                raise
        return self.snapshot()

    def disconnect(self) -> None:
        """Release the connected robot during application shutdown."""

        with self._lock:
            self.robot.disconnect()

    def record_error(self, message: str) -> None:
        """Add a transport-level failure to the browser console."""

        with self._lock:
            self._record("error", message)

    def _record_servo_status(self, status: Mapping[str, Any]) -> None:
        error = status.get("last_error") or status.get("watchdog")
        if error and error != self._last_servo_error:
            self._record("error", f"Servo sender stopped: {error}")
            self._last_servo_error = str(error)
        if not status.get("active"):
            return
        now = time.monotonic()
        if now - self._last_servo_telemetry_at < 1.0:
            return
        self._last_servo_telemetry_at = now
        queue_depth = status.get("queue_depth")
        queue_text = "n/a" if queue_depth is None else str(queue_depth)
        target_age = status.get("target_age_s")
        age_text = "n/a" if target_age is None else f"{float(target_age):.2f}s"
        self._record(
            "info",
            f"Servo {status.get('representation') or 'idle'} | "
            f"{float(status.get('send_rate_hz', 0.0)):.1f} Hz | "
            f"p95 {float(status.get('period_p95_ms', 0.0)):.2f} ms | "
            f"max {float(status.get('period_max_ms', 0.0)):.2f} ms | "
            f"overruns {int(status.get('overruns', 0))} | queue {queue_text} | target age {age_text}",
        )

    def _record(self, level: Literal["info", "error"], message: str) -> None:
        self._console_events.append(
            {
                "id": self._next_event_id,
                "time": time.strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
        )
        self._next_event_id += 1

    def _limits(self) -> dict[str, list[float] | None]:
        configured = {
            **self.robot.config.joint_position_limits,
            **self.robot.config.eef_pose_limits,
            "gripper.pos": (0.0, 1.0),
        }
        return {key: list(configured[key]) if key in configured else None for key in _CONTROL_KEYS}

    def _require_connection(self) -> None:
        if not self.robot.is_connected:
            raise RuntimeError("Connect the robot before issuing this command.")


def create_app(
    config: JakaRobotConfig,
    *,
    refresh_hz: float = 30.0,
) -> FastAPI:
    """Create a local browser debugger app for one JAKA controller."""

    if refresh_hz <= 0:
        raise ValueError("refresh_hz must be positive")

    debugger = JakaWebDebugger(JakaRobot(config))
    broadcaster = _StateBroadcaster(debugger, period_s=1.0 / refresh_hz)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        observation_task = asyncio.create_task(broadcaster.run(), name="jaka-observation-broadcast")
        try:
            yield
        finally:
            broadcaster.stop()
            await observation_task
            await asyncio.to_thread(debugger.disconnect)

    app = FastAPI(title="JAKA Robot Debugger", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.debugger = debugger
    app.state.broadcaster = broadcaster

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(_DASHBOARD_PATH, media_type="text/html")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()
        sender = asyncio.create_task(_send_state_frames(websocket, broadcaster, send_lock))
        receiver = asyncio.create_task(_receive_commands(websocket, debugger, send_lock))
        done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError):
                await task
        for task in done:
            with suppress(WebSocketDisconnect):
                task.result()

    return app


class _StateBroadcaster:
    """Poll the hardware once and share each frame with every WebSocket."""

    def __init__(self, debugger: JakaWebDebugger, *, period_s: float) -> None:
        self.debugger = debugger
        self.period_s = period_s
        self._condition = asyncio.Condition()
        self._stop = asyncio.Event()
        self._frame: dict[str, Any] | None = None
        self._version = 0

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while not self._stop.is_set():
            try:
                frame = await asyncio.to_thread(self.debugger.snapshot)
            except Exception as exc:
                self.debugger.record_error(f"WebSocket observation failed: {exc}")
            else:
                async with self._condition:
                    self._frame = frame
                    self._version += 1
                    self._condition.notify_all()
            deadline += self.period_s
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, deadline - loop.time()))
            if loop.time() - deadline > self.period_s:
                deadline = loop.time()

    def stop(self) -> None:
        self._stop.set()

    async def next_frame(self, version: int) -> tuple[int, dict[str, Any]]:
        async with self._condition:
            await self._condition.wait_for(lambda: self._frame is not None and self._version != version)
            assert self._frame is not None
            return self._version, self._frame


async def _send_state_frames(
    websocket: WebSocket, broadcaster: _StateBroadcaster, send_lock: asyncio.Lock
) -> None:
    version = -1
    while True:
        version, frame = await broadcaster.next_frame(version)
        async with send_lock:
            await websocket.send_json({"type": "state", **frame})


async def _receive_commands(websocket: WebSocket, debugger: JakaWebDebugger, send_lock: asyncio.Lock) -> None:
    while True:
        message = await websocket.receive_json()
        try:
            await _handle_message(debugger, message)
        except Exception as exc:
            debugger.record_error(f"WebSocket command failed: {exc}")
            async with send_lock:
                await websocket.send_json({"type": "error", "message": str(exc)})


async def _handle_message(debugger: JakaWebDebugger, message: Mapping[str, Any]) -> None:
    message_type = message.get("type")
    if message_type == "command":
        command = message.get("command")
        if command not in {"connection", "power", "enable", "servo", "abort"}:
            raise ValueError(f"unsupported command: {command}")
        await asyncio.to_thread(debugger.command, command)
        return
    if message_type == "relative":
        await asyncio.to_thread(debugger.relative_action, message.get("key"), message.get("delta"))
        return
    raise ValueError(f"unsupported message type: {message_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-based JAKA robot debugger")
    parser.add_argument("--ip", required=True, help="JAKA controller IP address")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server bind address (default: all local interfaces)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--refresh-hz", type=float, default=30.0, help="WebSocket state push rate")
    parser.add_argument(
        "--joint-limits-json",
        type=_parse_limits_json,
        default={},
        metavar="JSON",
        help='Optional joint bounds, e.g. {"joint_1.pos":[-3.14,3.14]}',
    )
    parser.add_argument(
        "--eef-limits-json",
        type=_parse_limits_json,
        default={},
        metavar="JSON",
        help='Optional TCP bounds, e.g. {"ee.z":[0.1,0.8]}',
    )
    parser.add_argument("--power-on", action="store_true", help="Power on after browser Connect")
    parser.add_argument("--enable", action="store_true", help="Enable and enter Servo Move after Connect")
    return parser.parse_args()


def _parse_limits_json(raw: str) -> dict[str, tuple[float, float]]:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return {key: tuple(bounds) for key, bounds in value.items()}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid limits JSON: {exc}") from exc


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    config = JakaRobotConfig(
        ip=args.ip,
        auto_power_on=args.power_on,
        auto_enable=args.enable,
        auto_enable_servo=args.enable,
        joint_position_limits=args.joint_limits_json,
        eef_pose_limits=args.eef_limits_json,
    )
    uvicorn.run(create_app(config, refresh_hz=args.refresh_hz), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
