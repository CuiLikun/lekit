from __future__ import annotations

import asyncio
import sys

import pytest
from fastapi.testclient import TestClient

from robots.jaka_robot.jaka_robot import JakaRobotConfig
from robots.jaka_robot.web_debug import (
    JakaWebDebugger,
    _handle_message,
    _StateBroadcaster,
    create_app,
    parse_args,
)


class FakeRobot:
    def __init__(self) -> None:
        self.is_connected = False
        self._servo_active = False
        self.config = type(
            "Config",
            (),
            {"joint_position_limits": {"joint_1.pos": (-1.0, 1.0)}, "eef_pose_limits": {}},
        )()
        self.calls: list[tuple[str, object]] = []

    @property
    def servo_active(self) -> bool:
        return self._servo_active

    def connect(self) -> None:
        self.is_connected = True
        self.calls.append(("connect", None))

    def disconnect(self) -> None:
        self.is_connected = False
        self._servo_active = False
        self.calls.append(("disconnect", None))

    def get_controller_state(self) -> dict[str, bool]:
        return {"powered_on": True, "enabled": True}

    def get_observation(self) -> dict[str, float]:
        return {"joint_1.pos": 0.1, "ee.x": 0.2, "gripper.pos": 0.3}

    def get_servo_status(self) -> dict[str, object]:
        return {
            "active": self._servo_active,
            "worker_alive": self._servo_active,
            "representation": "joints" if self._servo_active else None,
            "target_age_s": 0.1 if self._servo_active else None,
            "send_rate_hz": 125.0 if self._servo_active else 0.0,
            "period_p95_ms": 8.1 if self._servo_active else 0.0,
            "period_max_ms": 8.2 if self._servo_active else 0.0,
            "frames_sent": 10 if self._servo_active else 0,
            "overruns": 0,
            "consecutive_overruns": 0,
            "queue_depth": 2 if self._servo_active else None,
            "watchdog": None,
            "last_error": None,
        }

    def power_on(self) -> None:
        self.calls.append(("power_on", None))

    def power_off(self) -> None:
        self.calls.append(("power_off", None))

    def enable_robot(self) -> None:
        self.calls.append(("enable_robot", None))

    def disable_robot(self) -> None:
        self.calls.append(("disable_robot", None))

    def servo_enable(self, enabled: bool) -> None:
        self._servo_active = enabled
        self.calls.append(("servo_enable", enabled))

    def motion_abort(self) -> None:
        self.calls.append(("motion_abort", None))

    def send_relative_action(self, action: dict[str, float], *, use_servo: bool = True) -> dict[str, float]:
        self.calls.append(("relative", action, use_servo))
        return {**action, "gripper.pos": 0.3}

    def send_action(self, action: dict[str, float], *, use_servo: bool = True) -> dict[str, float]:
        self.calls.append(("servo", action, use_servo))
        return action

    def servo_frame_period_s(self) -> float:
        return 0.01


def test_web_debugger_returns_json_safe_live_state_and_enforces_known_controls():
    robot = FakeRobot()
    debugger = JakaWebDebugger(robot)  # type: ignore[arg-type]

    disconnected = debugger.snapshot()
    assert disconnected["controller"]["connected"] is False
    assert disconnected["limits"]["joint_1.pos"] == [-1.0, 1.0]

    debugger.command("connection")
    snapshot = debugger.snapshot()
    assert snapshot["controller"] == {
        "connected": True,
        "powered_on": True,
        "enabled": True,
        "servo_active": False,
    }
    assert snapshot["observation"]["ee.x"] == 0.2

    debugger.relative_action("joint_1.pos", 0.02)
    assert robot.calls[-1] == ("relative", {"joint_1.pos": 0.02}, False)
    with pytest.raises(ValueError, match="unsupported control field"):
        debugger.relative_action("bad.key", 0.1)


def test_web_message_dispatch_and_app_creation():
    robot = FakeRobot()
    debugger = JakaWebDebugger(robot)  # type: ignore[arg-type]

    asyncio.run(_handle_message(debugger, {"type": "command", "command": "connection"}))
    asyncio.run(_handle_message(debugger, {"type": "relative", "key": "ee.z", "delta": 0.005}))
    assert robot.calls[-1] == ("relative", {"ee.z": 0.005}, False)

    app = create_app(
        JakaRobotConfig(
            ip="10.0.0.2",
            auto_power_on=False,
            auto_enable=False,
            auto_enable_servo=False,
        )
    )
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "JAKA Robot Debugger" in page.text
        assert 'class="state-columns"' in page.text
        assert '<span id="message" class="message">' in page.text
        assert 'id="debug-console"' in page.text
        assert page.text.index("Joint State") < page.text.index("Gripper") < page.text.index("TCP State")
        with client.websocket_connect("/ws") as websocket:
            frame = websocket.receive_json()
            assert frame["type"] == "state"
            assert frame["controller"]["connected"] is False


def test_web_debugger_binds_all_local_interfaces_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["web_debug", "--ip", "10.0.0.2"])

    assert parse_args().host == "0.0.0.0"


def test_relative_control_only_updates_the_robot_owned_servo_target():
    robot = FakeRobot()
    debugger = JakaWebDebugger(robot)  # type: ignore[arg-type]
    robot.is_connected = True
    robot._servo_active = True

    debugger.relative_action("joint_1.pos", 0.02)

    assert robot.calls == [("relative", {"joint_1.pos": 0.02}, True)]
    assert any("Servo target updated" in event["message"] for event in debugger.snapshot()["console"])
    assert any("125.0 Hz" in event["message"] for event in debugger.snapshot()["console"])


def test_web_debugger_selects_non_servo_actions_when_servo_is_off():
    robot = FakeRobot()
    robot.is_connected = True
    debugger = JakaWebDebugger(robot)  # type: ignore[arg-type]

    debugger.relative_action("ee.z", 0.005)

    assert robot.calls == [("relative", {"ee.z": 0.005}, False)]
    assert any("controller-planned move" in event["message"] for event in debugger.snapshot()["console"])


def test_state_broadcaster_shares_one_polled_frame_with_multiple_subscribers():
    class CountingDebugger:
        def __init__(self) -> None:
            self.calls = 0

        def snapshot(self) -> dict[str, int]:
            self.calls += 1
            return {"sequence": self.calls}

        def record_error(self, _message: str) -> None:
            raise AssertionError("snapshot should not fail")

    async def scenario() -> None:
        debugger = CountingDebugger()
        broadcaster = _StateBroadcaster(debugger, period_s=0.005)  # type: ignore[arg-type]
        task = asyncio.create_task(broadcaster.run())
        first_version, _ = await broadcaster.next_frame(-1)
        first_subscriber = asyncio.create_task(broadcaster.next_frame(first_version))
        second_subscriber = asyncio.create_task(broadcaster.next_frame(first_version))
        first, second = await asyncio.gather(first_subscriber, second_subscriber)
        broadcaster.stop()
        await task

        assert first == second
        assert first[1]["sequence"] == first[0]
        assert debugger.calls == first[0]

    asyncio.run(scenario())
