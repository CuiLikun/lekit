"""Interactive Textual debugger for a connected :class:`JakaRobot`.

Run from the repository root with:

``uv run python -m robots.jaka_robot.debug_ui --ip 192.168.1.31``

The default connection does not power or enable the arm. Every motion button
submits a bounded delta relative to a fresh hardware observation.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    RichLog,
    Select,
    Static,
)

from .jaka_robot import JakaRobot, JakaRobotConfig

_JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
_EEF_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
_TRANSLATION_KEYS = frozenset(_EEF_KEYS[:3])
_CONTROL_KEYS = (*_JOINT_KEYS, *_EEF_KEYS, "gripper.pos")


def _safe_id(key: str) -> str:
    return key.replace(".", "-").replace("_", "-")


class JakaDebugApp(App[None]):
    """Live observation and relative-motion controls for one JAKA arm."""

    TITLE = "JAKA Robot Debugger"
    SUB_TITLE = "Disconnected"
    BINDINGS = [("q", "quit", "Quit"), ("escape", "abort_motion", "Abort")]

    CSS = """
    Screen { layout: vertical; }
    #connection { height: 3; padding: 0 1; background: $panel; }
    #connection Button { width: 14; margin-right: 1; }
    #state { width: 1fr; content-align: right middle; color: $text-muted; }
    #observation-pane { height: 1fr; width: 1fr; border: round $primary; padding: 0 1; }
    #observation { height: 8; min-height: 4; }
    #events { height: 7; border: round $warning; }
    .section-title { height: 2; color: $text; text-style: bold; content-align: left middle; }
    .step-row { height: 3; }
    .step-row Label { width: 18; content-align: left middle; }
    .step-row Select { width: 24; }
    .axis-row { height: 3; }
    .axis-name { width: 13; content-align: left middle; }
    .limit-value { width: 11; content-align: right middle; color: $text-muted; padding-right: 1; }
    .axis-value { width: 11; content-align: right middle; color: $accent; padding-right: 1; }
    .axis-row Button { width: 7; margin-left: 1; }
    .limit-header { height: 2; color: $text-muted; }
    .limit-header .axis-name { text-style: bold; }
    """

    def __init__(self, config: JakaRobotConfig, refresh_hz: float = 30.0) -> None:
        super().__init__()
        self.robot = JakaRobot(config)
        self.refresh_period_s = 1.0 / refresh_hz
        self._robot_lock = threading.RLock()
        self._polling = False
        self._state_polling = False
        self._powered_on = False
        self._enabled = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="connection"):
            yield Button("Connect", id="connect-toggle", variant="primary")
            yield Button("Power On", id="power-toggle", variant="warning")
            yield Button("Enable", id="enable-toggle")
            yield Button("Servo On", id="servo-toggle")
            yield Button("Abort", id="abort", variant="error")
            yield Static("Disconnected", id="state")
        with VerticalScroll(id="observation-pane"):
            yield Label("Joint State", classes="section-title")
            yield self._step_select(
                "Step (rad)",
                "joint-step",
                (("0.005", 0.005), ("0.01", 0.01), ("0.02", 0.02), ("0.05", 0.05), ("0.1", 0.1)),
                0.02,
            )
            yield self._limit_header()
            yield from self._axis_rows(_JOINT_KEYS, "joint")

            yield Label("TCP State", classes="section-title")
            yield self._step_select(
                "XYZ step (m)",
                "linear-step",
                (("0.001", 0.001), ("0.002", 0.002), ("0.005", 0.005), ("0.01", 0.01)),
                0.005,
            )
            yield self._step_select(
                "RPY step (rad)",
                "angular-step",
                (("0.01", 0.01), ("0.02", 0.02), ("0.05", 0.05), ("0.1", 0.1)),
                0.05,
            )
            yield self._limit_header()
            yield from self._axis_rows(_EEF_KEYS, "eef")

            yield Label("Gripper State", classes="section-title")
            yield self._step_select(
                "Step",
                "gripper-step",
                (("0.01", 0.01), ("0.02", 0.02), ("0.05", 0.05), ("0.1", 0.1)),
                0.05,
            )
            yield self._limit_header()
            yield from self._axis_rows(("gripper.pos",), "gripper")

            yield Label("Additional Observations", classes="section-title")
            yield DataTable(id="observation", cursor_type="row", zebra_stripes=True)
            yield Label("Events", classes="section-title")
            yield RichLog(id="events", markup=True, wrap=True, highlight=True)
        yield Footer()

    @staticmethod
    def _step_select(
        label: str,
        select_id: str,
        options: tuple[tuple[str, float], ...],
        default: float,
    ) -> Horizontal:
        return Horizontal(
            Label(label),
            Select(options, value=default, allow_blank=False, id=select_id),
            classes="step-row",
        )

    @staticmethod
    def _axis_rows(keys: Iterable[str], group: str) -> Iterable[Horizontal]:
        for key in keys:
            control_id = _safe_id(key)
            yield Horizontal(
                Label(key, classes="axis-name"),
                Static("n/a", id=f"minimum-{control_id}", classes="limit-value"),
                Static("--", id=f"current-{control_id}", classes="axis-value"),
                Static("n/a", id=f"maximum-{control_id}", classes="limit-value"),
                Button("-", id=f"minus-{control_id}", classes=f"relative {group}"),
                Button("+", id=f"plus-{control_id}", classes=f"relative {group}"),
                classes="axis-row",
            )

    @staticmethod
    def _limit_header() -> Horizontal:
        return Horizontal(
            Label("Axis", classes="axis-name"),
            Static("Min", classes="limit-value"),
            Static("Current", classes="axis-value"),
            Static("Max", classes="limit-value"),
            classes="limit-header",
        )

    def on_mount(self) -> None:
        self.query_one("#observation", DataTable).add_columns("Field", "Value")
        self.set_interval(self.refresh_period_s, self._start_poll)
        self.set_interval(0.5, self._start_state_poll)
        self._render_limits()
        self._refresh_controls()

    def _render_limits(self) -> None:
        limits = {
            **self.robot.config.joint_position_limits,
            **self.robot.config.eef_pose_limits,
            "gripper.pos": (0.0, 1.0),
        }
        for key in _CONTROL_KEYS:
            bounds = limits.get(key)
            if bounds is None:
                continue
            minimum, maximum = bounds
            control_id = _safe_id(key)
            self.query_one(f"#minimum-{control_id}", Static).update(self._format_value(minimum))
            self.query_one(f"#maximum-{control_id}", Static).update(self._format_value(maximum))

    def action_abort_motion(self) -> None:
        if self.robot.is_connected:
            self._command_worker("abort")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        command_ids: dict[str, Literal["connection", "power", "enable", "servo", "abort"]] = {
            "connect-toggle": "connection",
            "power-toggle": "power",
            "enable-toggle": "enable",
            "servo-toggle": "servo",
            "abort": "abort",
        }
        if button_id in command_ids:
            self._command_worker(command_ids[button_id])
            return
        for prefix, direction in (("minus-", -1.0), ("plus-", 1.0)):
            if button_id.startswith(prefix):
                key = self._key_from_control_id(button_id.removeprefix(prefix))
                self._relative_worker(key, direction * self._step_for_key(key))
                return

    @staticmethod
    def _key_from_control_id(control_id: str) -> str:
        for key in _CONTROL_KEYS:
            if _safe_id(key) == control_id:
                return key
        raise ValueError(f"unknown motion control: {control_id}")

    def _step_for_key(self, key: str) -> float:
        if key in _JOINT_KEYS:
            select_id = "#joint-step"
        elif key in _TRANSLATION_KEYS:
            select_id = "#linear-step"
        elif key in _EEF_KEYS:
            select_id = "#angular-step"
        else:
            select_id = "#gripper-step"
        return float(self.query_one(select_id, Select).value)

    def _start_poll(self) -> None:
        if self.robot.is_connected and not self._polling:
            self._poll_observation()

    def _start_state_poll(self) -> None:
        if self.robot.is_connected and not self._state_polling:
            self._poll_controller_state()

    @work(thread=True, group="observation", exclusive=True)
    def _poll_observation(self) -> None:
        self._polling = True
        try:
            with self._robot_lock:
                observation = self.robot.get_observation()
            self.call_from_thread(self._render_observation, observation)
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Observation failed:[/] {exc}")
        finally:
            self._polling = False

    @work(thread=True, group="controller-state", exclusive=True)
    def _poll_controller_state(self) -> None:
        self._state_polling = True
        try:
            with self._robot_lock:
                state = self.robot.get_controller_state()
            self.call_from_thread(self._apply_controller_state, state)
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]State refresh failed:[/] {exc}")
        finally:
            self._state_polling = False

    def _render_observation(self, observation: dict[str, Any]) -> None:
        table = self.query_one("#observation", DataTable)
        table.clear(columns=False)
        for key, value in observation.items():
            if key not in _CONTROL_KEYS:
                table.add_row(key, self._format_value(value))
        for key in _CONTROL_KEYS:
            if key in observation:
                self.query_one(f"#current-{_safe_id(key)}", Static).update(
                    self._format_value(observation[key])
                )

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (int, float, np.number)):
            return f"{float(value):.5f}"
        if isinstance(value, np.ndarray):
            return f"array{tuple(value.shape)} {value.dtype}"
        return str(value)

    @work(thread=True, group="command", exclusive=True)
    def _command_worker(self, command: Literal["connection", "power", "enable", "servo", "abort"]) -> None:
        try:
            with self._robot_lock:
                if command == "connection":
                    if self.robot.is_connected:
                        self.robot.disconnect()
                    else:
                        self.robot.connect()
                elif command == "power":
                    if self._powered_on:
                        if self.robot._servo_active:
                            self.robot.servo_enable(False)
                        if self._enabled:
                            self.robot.disable_robot()
                        self.robot.power_off()
                    else:
                        self.robot.power_on()
                elif command == "enable":
                    if self._enabled:
                        self.robot.disable_robot()
                    else:
                        self.robot.enable_robot()
                elif command == "servo":
                    self.robot.servo_enable(not self.robot._servo_active)
                else:
                    self.robot.motion_abort()
                state = self.robot.get_controller_state() if self.robot.is_connected else None
            self.call_from_thread(self._command_complete, command, state)
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]{command.title()} failed:[/] {exc}")

    def _command_complete(self, command: str, state: dict[str, bool] | None) -> None:
        if state is None:
            self._powered_on = False
            self._enabled = False
        else:
            self._apply_controller_state(state)
        self._refresh_controls()
        self._log(f"[green]{command.title()} state changed.[/]")

    def _apply_controller_state(self, state: dict[str, bool]) -> None:
        self._powered_on = state["powered_on"]
        self._enabled = state["enabled"]
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        connected = self.robot.is_connected
        servo_active = connected and self.robot._servo_active

        self._set_button(
            "#connect-toggle",
            "Disconnect" if connected else "Connect",
            "success" if connected else "primary",
        )
        self._set_button(
            "#power-toggle",
            "Power Off" if self._powered_on else "Power On",
            "success" if self._powered_on else "warning",
            disabled=not connected,
        )
        self._set_button(
            "#enable-toggle",
            "Disable" if self._enabled else "Enable",
            "success" if self._enabled else "default",
            disabled=not connected or not self._powered_on,
        )
        self._set_button(
            "#servo-toggle",
            "Servo Off" if servo_active else "Servo On",
            "success" if servo_active else "default",
            disabled=not connected or not self._enabled,
        )
        self.query_one("#abort", Button).disabled = not connected

        for button in self.query("Button.relative"):
            button.disabled = not servo_active

        if not connected:
            state_text = "Disconnected"
        else:
            state_text = (
                f"Connected | Power {'on' if self._powered_on else 'off'} | "
                f"Enabled {'yes' if self._enabled else 'no'} | Servo {'on' if servo_active else 'off'}"
            )
        self.query_one("#state", Static).update(state_text)
        self.sub_title = state_text

    def _set_button(
        self,
        selector: str,
        label: str,
        variant: Literal["default", "primary", "success", "warning", "error"],
        *,
        disabled: bool = False,
    ) -> None:
        button = self.query_one(selector, Button)
        button.label = label
        button.variant = variant
        button.disabled = disabled

    @work(thread=True, group="command", exclusive=True)
    def _relative_worker(self, key: str, delta: float) -> None:
        try:
            with self._robot_lock:
                applied = self.robot.send_relative_action({key: delta})
            self.call_from_thread(self._log, f"[green]{key} {delta:+.5f}:[/] {applied[key]:.5f}")
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Relative action failed:[/] {exc}")

    def _log(self, message: str) -> None:
        self.query_one("#events", RichLog).write(message)

    def on_unmount(self) -> None:
        with self._robot_lock:
            self.robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live JAKA robot debugger")
    parser.add_argument("--ip", required=True, help="JAKA controller IP address")
    parser.add_argument("--refresh-hz", type=float, default=30.0)
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
    parser.add_argument("--power-on", action="store_true", help="Power the controller on after connection")
    parser.add_argument("--enable", action="store_true", help="Enable robot servos after connection")
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
    if args.refresh_hz <= 0:
        raise SystemExit("--refresh-hz must be positive")
    config = JakaRobotConfig(
        ip=args.ip,
        auto_power_on=args.power_on,
        auto_enable=args.enable,
        auto_enable_servo=args.enable,
        joint_position_limits=args.joint_limits_json,
        eef_pose_limits=args.eef_limits_json,
    )
    JakaDebugApp(config, refresh_hz=args.refresh_hz).run()


if __name__ == "__main__":
    main()
