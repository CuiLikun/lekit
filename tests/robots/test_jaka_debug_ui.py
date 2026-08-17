import asyncio
import sys

import pytest
from textual.widgets import Button, Select

from lekit.robots.jaka_robot.debug_ui import JakaDebugApp, parse_args
from lekit.robots.jaka_robot.jaka_robot import JakaRobotConfig


class ConnectedRC:
    def servo_move_enable(self, _enabled, _blocking=True):
        return (0,)

    def logout(self):
        return (0,)


def test_debugger_defaults_to_non_actuating_connection():
    app = JakaDebugApp(
        JakaRobotConfig(
            ip="10.0.0.2",
            auto_power_on=False,
            auto_enable=False,
            auto_enable_servo=False,
        )
    )

    assert app.robot.config.auto_power_on is False
    assert app.robot.config.auto_enable is False
    assert app.refresh_period_s == pytest.approx(1 / 30)


def test_debugger_composes_all_live_controls_without_a_controller():
    async def check() -> None:
        app = JakaDebugApp(
            JakaRobotConfig(
                ip="10.0.0.2",
                auto_power_on=False,
                auto_enable=False,
                auto_enable_servo=False,
                joint_position_limits={"joint_1.pos": (-3.14, 3.14)},
                eef_pose_limits={"ee.z": (0.1, 0.8)},
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert len(app.query("Input")) == 0
            assert len(app.query("TabbedContent")) == 0
            assert len(app.query("#control-pane")) == 0
            assert len(app.query(Button)) == 31
            assert len(app.query(Select)) == 4
            assert len(app.query("#observation")) == 1
            assert str(app.query_one("#connect-toggle", Button).label) == "Connect"
            assert app.query_one("#connect-toggle", Button).variant == "primary"
            assert app.query_one("#power-toggle", Button).disabled is True

            app.robot.rc = ConnectedRC()
            app.robot._servo_active = True
            app._powered_on = True
            app._enabled = True
            app._refresh_controls()

            assert str(app.query_one("#connect-toggle", Button).label) == "Disconnect"
            assert app.query_one("#connect-toggle", Button).variant == "success"
            assert str(app.query_one("#power-toggle", Button).label) == "Power Off"
            assert app.query_one("#power-toggle", Button).variant == "success"
            assert str(app.query_one("#enable-toggle", Button).label) == "Disable"
            assert app.query_one("#enable-toggle", Button).variant == "success"
            assert str(app.query_one("#servo-toggle", Button).label) == "Servo Off"
            assert app.query_one("#servo-toggle", Button).variant == "success"
            assert app.query_one("#plus-joint-1-pos", Button).disabled is False
            assert app.query_one("#plus-ee-x", Button).disabled is False
            assert str(app.query_one("#minimum-joint-1-pos").render()) == "-3.14000"
            assert str(app.query_one("#maximum-joint-1-pos").render()) == "3.14000"
            assert str(app.query_one("#minimum-joint-2-pos").render()) == "n/a"
            assert str(app.query_one("#minimum-ee-z").render()) == "0.10000"
            assert str(app.query_one("#maximum-gripper-pos").render()) == "1.00000"

    asyncio.run(check())


def test_debugger_cli_has_no_control_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["debug_ui", "--ip", "10.0.0.2"])
    args = parse_args()

    assert not hasattr(args, "mode")
