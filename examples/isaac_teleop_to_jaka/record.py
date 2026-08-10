#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record a LeRobot dataset by driving a JAKA arm with an Isaac Teleop XR controller.

The XR controller is clutch-rebased onto the measured JAKA TCP pose. Each engaged frame
becomes an absolute ``ee.x/y/z/roll/pitch/yaw`` target (metres and radians) sent through
the JAKA SDK ``servo_p`` Cartesian Servo Move. No host-side inverse kinematics.

Usage:

uv run python -m examples.isaac_teleop_to_jaka.record \
    --robot.type=jaka_robot \
    --robot.ip=192.168.1.31 \
    --robot.id=jaka_arm \
    --robot.cameras="{ hand: {type: intelrealsense, serial_number_or_name: '342522070741', width: 640, height: 480, fps: 30}}" \
    --teleop.type=xr_controller \
    --teleop.lock_pose="[0.0, 0.0, 0.0]" \
    --teleop.use_head_yaw=true \
    --teleop.operator_yaw_deg=0 \
    --dataset.repo_id="sorel/pick-cube" \
    --dataset.single_task="Pick up the object" \
    --dataset.fps=30 \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=9999 \
    --dataset.reset_time_s=5 \
    --dataset.streaming_encoding=True \
    --dataset.push_to_hub=False

The XR trigger toggles the gripper between closed (0) and open (1) on each press.
Keyboard shortcuts: Right/n ends and saves the current episode, Left/r discards and
re-records it, and Esc/q stops immediately. All frames, including
clutch-disengaged hold frames, are recorded.
"""

import logging
import math
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from threading import Lock
from typing import Any

import numpy as np
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers "opencv")
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401  (registers "intelrealsense")
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    safe_stop_image_writer,
)
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from robots.jaka_robot import JakaRobot, JakaRobotConfig
from robots.jaka_robot.dataset_features import build_dataset_features

from .control_trace import ControlTraceWriter
from .xr import CLOUDXR_ENV_FILE, IsaacTeleopConfig, make_xr_device

DELTA_POSITION_BAR_SPAN_M = 0.05
DELTA_POSITION_BAR_WIDTH = 31
DELTA_POSITION_DISPLAY_DEADBAND_M = 0.0002
CONTROL_RATE_WINDOW_S = 1.0

# ── Hold latch ──────────────────────────────────────────────────────────────


class HoldLatch:
    """Hold the measured pose captured on the first idle frame.

    Capturing feedback at the deadman release edge prevents the controller from
    continuing toward a hand target that the arm has not reached yet.
    """

    def __init__(self, action_keys: list[str]):
        self._action_keys = action_keys
        self._held: dict[str, float] | None = None

    def resolve(self, action: dict | None, obs: dict) -> dict:
        if action is not None:
            self._held = None
            return dict(action)
        if self._held is None:
            self._held = {k: float(obs[k]) for k in self._action_keys if k in obs}
        return self._held


class LoopRateMonitor:
    """Measure control-loop frequency over a rolling wall-clock window."""

    def __init__(self, window_s: float = CONTROL_RATE_WINDOW_S):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("loop-rate window must be positive and finite")
        self.window_s = float(window_s)
        self._starts: deque[float] = deque()

    def update(self, loop_started_at: float) -> float | None:
        if not math.isfinite(loop_started_at):
            raise ValueError("loop start time must be finite")
        if self._starts and loop_started_at <= self._starts[-1]:
            raise ValueError("loop start times must increase monotonically")

        self._starts.append(float(loop_started_at))
        cutoff = loop_started_at - self.window_s
        while len(self._starts) > 2 and self._starts[0] < cutoff:
            self._starts.popleft()
        if len(self._starts) < 2:
            return None
        elapsed = self._starts[-1] - self._starts[0]
        return (len(self._starts) - 1) / elapsed


class TriggerGripperToggle:
    """Convert an analog XR trigger into a one-press gripper toggle."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._position: float | None = None
        self._pressed = False

    def apply(self, action: dict, obs: dict, trigger: float) -> dict:
        if self._position is None:
            self._position = float(np.clip(float(obs.get("gripper.pos", 0.0)), 0.0, 1.0))

        pressed = math.isfinite(trigger) and trigger >= self.threshold
        should_send = pressed and not self._pressed
        if should_send:
            self._position = 1.0 - self._position
        self._pressed = pressed

        result = dict(action)
        result.pop("gripper.pos", None)
        if should_send:
            result["gripper.pos"] = self._position
        return result


def _send_action_for_clutch_state(
    robot: JakaRobot,
    action: dict,
    *,
    engaged: bool,
    observation: dict | None = None,
) -> dict:
    """Run Cartesian Servo only while the XR deadman is engaged."""

    if engaged:
        if not robot.is_in_servo():
            robot.servo_enable(True, representation="eef")
        return robot.send_action(action)

    if robot.is_in_servo():
        robot.servo_enable(False)
    gripper_action = {"gripper.pos": action["gripper.pos"]} if "gripper.pos" in action else None
    if gripper_action is not None:
        robot.send_action(gripper_action)

    held = (
        {key: float(observation[key]) for key in robot.action_features if key in observation}
        if observation is not None
        else {}
    )
    held.update(action)
    return held


# ── Keyboard control ────────────────────────────────────────────────────────


def init_keyboard_listener(stop_callback: Callable[[], None] | None = None):
    """Wire Right/Left/Esc shortcuts to recording control events.

    When stdin is a TTY, prefer the stdlib :class:`TerminalKeyListener` (works over SSH
    and emits canonical key names); the dispatcher maps ``n``/``r``/``q`` to the same
    events as the arrow keys / ``Esc`` so a laggy terminal splitting escape sequences
    still gets through. For non-interactive stdin, use the upstream listener when no
    callback is needed, or the shared listener factory when immediate teardown is needed.
    """
    from lerobot.utils.keyboard_input import apply_recording_control

    if not (sys.stdin is not None and sys.stdin.isatty()) and stop_callback is None:
        from lerobot.utils.keyboard_input import init_keyboard_listener as _upstream

        return _upstream()

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)
            if stop_callback is not None:
                stop_callback()

    if sys.stdin is not None and sys.stdin.isatty():
        from lerobot.utils.keyboard_input import TerminalKeyListener

        listener = TerminalKeyListener(on_key)
        listener.start()
        logging.info(
            "Keyboard control via terminal — keep this terminal focused: "
            "Right/n = end episode early, Left/r = re-record, Esc/q = stop."
        )
        return listener, events

    from lerobot.utils.keyboard_input import create_key_listener

    listener = create_key_listener(
        on_key,
        controls_help="Right/Left/Esc, or n=next, r=re-record, q=quit",
    )
    return listener, events


def _make_hardware_stop_callback(robot: JakaRobot, device: "Device") -> Callable[[], None]:
    """Return an idempotent best-effort teardown callback for keyboard handlers."""

    stopped = False
    lock = Lock()

    def stop_hardware_now() -> None:
        nonlocal stopped
        with lock:
            if stopped:
                return
            stopped = True

            # Stop Servo/XR before disconnecting the robot. Each operation is guarded so
            # a failure in one teardown step cannot leave the other connection open.
            with suppress(Exception):
                device.cleanup()
            with suppress(Exception):
                if robot.is_connected:
                    robot.disconnect()

    return stop_hardware_now


# ── Device bundle ───────────────────────────────────────────────────────────


@dataclass
class Device:
    """Per-frame XR -> JAKA glue. ``compute(obs)`` returns ``None`` while the clutch
    is disengaged so the loop can hold the measured pose."""

    compute: Callable[[dict | None], dict | None]
    startup: Callable[[], None]
    cleanup: Callable[[], None]
    telemetry: dict[str, Any] = field(default_factory=dict)


def build_device(cfg: "RecordConfig") -> tuple[JakaRobot, Device]:
    """Connect the JAKA arm and build the XR -> JAKA device bundle.

    Connects the follower FIRST so the clutch-home seed (in ``device.startup``) can
    read the live EE pose. On any failure after ``connect()`` the follower is
    disconnected so the connection never leaks.
    """
    if cfg.teleop.cloudxr_env_file is None:
        cfg.teleop.cloudxr_env_file = CLOUDXR_ENV_FILE

    # The recorder owns Servo Move startup: wait until XR is ready and the
    # clutch home has been measured. Feedback uses a separate SDK handle so
    # 30 Hz observations cannot interrupt the controller's 8 ms command stream.
    cfg.robot.auto_enable_servo = False
    cfg.robot.separate_feedback_connection = True
    # The recorder uses the XR teleoperator's responsive Cartesian profile. JAKA's
    # controller-side Cartesian NLF remains enabled below for jitter suppression.
    for robot_name, teleop_name in (
        ("servo_eef_max_velocity_m_s", "servo_linear_velocity_m_s"),
        ("servo_eef_max_acceleration_m_s2", "servo_linear_acceleration_m_s2"),
        ("servo_filter_eef_max_jerk_m_s3", "servo_linear_jerk_m_s3"),
        ("servo_eef_max_angular_velocity_rad_s", "servo_angular_velocity_rad_s"),
        ("servo_eef_max_angular_acceleration_rad_s2", "servo_angular_acceleration_rad_s2"),
        ("servo_filter_eef_max_angular_jerk_rad_s3", "servo_angular_jerk_rad_s3"),
    ):
        if hasattr(cfg.teleop, teleop_name):
            setattr(cfg.robot, robot_name, getattr(cfg.teleop, teleop_name))
    if cfg.robot.servo_filter_mode == "none":
        cfg.robot.servo_filter_mode = "cartesian_nlf"
    elif cfg.robot.servo_filter_mode != "cartesian_nlf":
        raise ValueError("isaac_teleop_to_jaka.record requires cartesian_nlf for Cartesian Servo P control")

    driver_module = sys.modules[JakaRobot.__module__]
    logging.info(
        "[JAKA-SERVO] driver=%s filter=%s linear(v=%.3f m/s, a=%.3f m/s^2, j=%.3f m/s^3) ",
        Path(driver_module.__file__).resolve(),
        cfg.robot.servo_filter_mode,
        getattr(cfg.robot, "servo_eef_max_velocity_m_s", math.nan),
        getattr(cfg.robot, "servo_eef_max_acceleration_m_s2", math.nan),
        getattr(cfg.robot, "servo_filter_eef_max_jerk_m_s3", math.nan),
    )

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    device: Device | None = None
    try:
        if not isinstance(robot, JakaRobot):
            raise ValueError(
                f"isaac_teleop_to_jaka.record requires --robot.type=jaka_robot, got {type(robot).__name__}"
            )
        if robot.config.user_frame_id != 0:
            raise ValueError(
                "isaac_teleop_to_jaka.record requires --robot.user_frame_id=0 because the "
                "default XR transform targets the robot base frame."
            )
        bundle = make_xr_device(robot, cfg.teleop)
        device = Device(
            compute=bundle["compute"],
            startup=bundle["startup"],
            cleanup=bundle["cleanup"],
            telemetry=bundle["telemetry"],
        )
        device.startup()
    except BaseException:
        if device is not None:
            with suppress(Exception):
                device.cleanup()
        robot.disconnect()
        raise

    return robot, device


# ── Control panel ───────────────────────────────────────────────────────────


def _signed_horizontal_bar(value: float, span: float, *, width: int = DELTA_POSITION_BAR_WIDTH) -> Text:
    """Render a fixed-width bar with negative values left of center."""

    if width < 3 or width % 2 == 0:
        raise ValueError("horizontal bar width must be an odd integer of at least 3")
    if not math.isfinite(span) or span <= 0:
        raise ValueError("horizontal bar span must be positive and finite")

    half = (width - 1) // 2
    filled = int(round(float(np.clip(abs(value) / span, 0.0, 1.0)) * half))
    negative_fill = filled if value < 0 else 0
    positive_fill = filled if value > 0 else 0

    bar = Text()
    bar.append("─" * (half - negative_fill), style="dim")
    bar.append("█" * negative_fill, style="bold bright_cyan")
    bar.append("│", style="bright_white")
    bar.append("█" * positive_fill, style="bold bright_green")
    bar.append("─" * (half - positive_fill), style="dim")
    return bar


def _jaka_status(robot: JakaRobot) -> dict[str, Any]:
    """Return a best-effort controller and managed Servo status snapshot."""

    status: dict[str, Any] = {}
    try:
        status.update(robot.get_controller_state())
        status["servo_on"] = robot.is_in_servo()
    except Exception as exc:
        status["controller_error"] = str(exc)

    servo = robot.get_servo_status()
    status.update(
        servo_sender_active=servo["active"],
        servo_sender_alive=servo["worker_alive"],
        servo_representation=servo["representation"],
        servo_rate_hz=round(float(servo["send_rate_hz"]), 1),
        servo_frames_sent=servo["frames_sent"],
        servo_queue_depth=servo["queue_depth"],
        servo_last_error=servo["last_error"],
    )
    return status


def _jaka_status_line(robot_status: dict[str, Any]) -> Text:
    line = Text()
    for index, (label, key) in enumerate(
        (("POWER", "powered_on"), ("ENABLED", "enabled"), ("SERVO", "servo_on"))
    ):
        if index:
            line.append("   ")
        active = bool(robot_status.get(key, False))
        line.append(f"{label} ", style="dim")
        line.append("ON" if active else "OFF", style="green" if active else "yellow")
    servo_rate_hz = robot_status.get("servo_rate_hz")
    if isinstance(servo_rate_hz, (int, float)) and servo_rate_hz > 0:
        line.append(f"   {servo_rate_hz:.1f} Hz", style="dim")
    return line


def _control_rate_line(measured_hz: object, target_hz: object) -> Text:
    try:
        measured = float(measured_hz)
        target = float(target_hz)
    except (TypeError, ValueError):
        return Text("warming up", style="dim")
    if not math.isfinite(measured) or not math.isfinite(target) or measured < 0 or target <= 0:
        return Text("unavailable", style="yellow")

    ratio = measured / target
    style = "green" if ratio >= 0.95 else "yellow" if ratio >= 0.8 else "bold red"
    return Text(f"{measured:.1f} / {target:.1f} Hz", style=style)


def _format_six_vector(values: list[object]) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def _control_panel(
    telemetry: dict[str, Any],
    action: dict[str, float],
    obs: dict,
    robot_status: dict[str, Any],
) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column()

    engaged = bool(telemetry.get("clutch_engaged", False))
    mode = Text("ENGAGED" if engaged else "HOLD", style="bold green" if engaged else "bold yellow")
    if telemetry.get("head_is_tracking") is False:
        mode.append("   XR TRACKING LOST", style="bold red")
    mode.append("   LOOP ", style="dim")
    mode.append_text(
        _control_rate_line(
            robot_status.get("control_rate_hz"),
            robot_status.get("control_target_hz"),
        )
    )
    episode_number = robot_status.get("episode_number")
    episode_total = robot_status.get("episode_total")
    episode_elapsed_s = robot_status.get("episode_elapsed_s")
    if (
        isinstance(episode_number, int)
        and isinstance(episode_total, int)
        and isinstance(episode_elapsed_s, (int, float))
    ):
        if robot_status.get("recording", False):
            table.add_row(
                "Episode",
                f"{episode_number} / {episode_total}   {float(episode_elapsed_s):.1f} s",
            )
        else:
            table.add_row(
                "Reset",
                f"after {episode_number} / {episode_total}   {float(episode_elapsed_s):.1f} s",
            )
    table.add_row("Robot", _jaka_status_line(robot_status))

    position = {axis: action[f"ee.{axis}"] for axis in ("x", "y", "z") if f"ee.{axis}" in action}
    actual_pos = {axis: obs[f"ee.{axis}"] for axis in ("x", "y", "z") if f"ee.{axis}" in obs}
    pos_delta = {k: actual_pos[k] - position[k] for k in actual_pos if k in position}

    tcp_keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
    if all(key in obs for key in tcp_keys):
        table.add_row("TCP(m/rad)", _format_six_vector([obs[key] for key in tcp_keys]))
    joint_values = [obs.get(f"joint_{index}.pos") for index in range(1, 7)]
    if all(value is not None for value in joint_values):
        table.add_row("Joint(rad)", _format_six_vector(joint_values))
    if "gripper.pos" in obs:
        table.add_row("Grip", f"{float(obs['gripper.pos']):.2f}")
    if pos_delta:
        error_norm_mm = math.sqrt(sum(delta**2 for delta in pos_delta.values())) * 1000.0
        table.add_row("Error", f"{error_norm_mm:.1f} mm")
        for axis in ("x", "y", "z"):
            if axis not in pos_delta:
                continue
            delta = pos_delta[axis]
            if abs(delta) < DELTA_POSITION_DISPLAY_DEADBAND_M:
                delta = 0.0
            table.add_row(
                f"{axis.upper()} {delta * 1000:+.1f} mm",
                _signed_horizontal_bar(delta, DELTA_POSITION_BAR_SPAN_M),
            )

    return Panel(table, title="JAKA Teleop", subtitle=mode)


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class RecordConfig:
    """CLI config for Isaac Teleop -> JAKA Cartesian dataset recording."""

    robot: JakaRobotConfig
    teleop: IsaacTeleopConfig
    dataset: DatasetRecordConfig

    # Resume recording on an existing (previously interrupted) dataset.
    resume: bool = False
    control_trace_csv: Path | None = None
    control_trace_flush_frames: int = 30


# ── Record loop ─────────────────────────────────────────────────────────────


@safe_stop_image_writer
def _record_loop(
    robot: JakaRobot,
    device: Device,
    action_keys: list[str],
    events: dict,
    fps: int,
    live: Live,
    dataset: LeRobotDataset | None = None,
    control_time_s: float = 0.0,
    single_task: str | None = None,
    gripper_toggle: TriggerGripperToggle | None = None,
    control_trace: ControlTraceWriter | None = None,
    episode_number: int | None = None,
    episode_total: int | None = None,
) -> None:
    """Run one episode or reset phase of the Cartesian control loop.

    When ``dataset`` is None the loop still controls the robot so the operator can
    reposition during reset, but frames are not recorded.
    """
    control_interval = 1.0 / fps
    rate_monitor = LoopRateMonitor()
    hold = HoldLatch(action_keys)
    gripper_toggle = gripper_toggle or TriggerGripperToggle()
    robot_status: dict[str, Any] = {}
    next_status_refresh_at = 0.0
    start_t = time.perf_counter()
    timestamp = 0.0
    record_frames = dataset is not None

    while timestamp < control_time_s:
        loop_start = time.perf_counter()
        control_rate_hz = rate_monitor.update(loop_start)

        if events["exit_early"]:
            events["exit_early"] = False
            break

        try:
            obs = robot.get_observation()
        except Exception:
            if events["stop_recording"]:
                break
            raise

        if events["stop_recording"]:
            break

        if record_frames:
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

        # XR clutch disengaged: hold the TCP pose latched on the idle edge.
        try:
            raw = device.compute(obs)
        except Exception:
            if events["stop_recording"]:
                break
            raise
        action = gripper_toggle.apply(
            hold.resolve(raw, obs),
            obs,
            float(device.telemetry.get("trigger", 0.0)),
        )

        if events["stop_recording"]:
            break

        try:
            sent_action = _send_action_for_clutch_state(
                robot,
                action,
                engaged=bool(device.telemetry.get("clutch_engaged", False)),
                observation=obs,
            )
        except Exception:
            if events["stop_recording"]:
                break
            raise

        if loop_start >= next_status_refresh_at:
            robot_status = _jaka_status(robot)
            next_status_refresh_at = loop_start + 1.0
        robot_status["control_rate_hz"] = control_rate_hz
        robot_status["control_target_hz"] = float(fps)
        robot_status["episode_number"] = episode_number
        robot_status["episode_total"] = episode_total
        robot_status["episode_elapsed_s"] = loop_start - start_t
        robot_status["recording"] = record_frames

        if record_frames:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **action_frame, "task": single_task})

        # Work time of this iteration: obs read + compute + target update + record.
        # The robot-owned 8 ms sender continues independently if this loop is late.
        frame_ms = (time.perf_counter() - loop_start) * 1000
        if control_trace is not None:
            control_trace.write_frame(
                phase="record" if record_frames else "reset",
                raw_action=raw,
                action=action,
                sent_action=sent_action,
                observation=obs,
                telemetry=device.telemetry,
                servo_status=robot.get_servo_status(),
                frame_ms=frame_ms,
                control_rate_hz=control_rate_hz,
                control_target_hz=float(fps),
            )
        live.update(
            _control_panel(
                device.telemetry,
                action,
                obs,
                robot_status,
            ),
            refresh=True,
        )

        precise_sleep(max(control_interval - (time.perf_counter() - loop_start), 0.0))
        timestamp = time.perf_counter() - start_t


# ── Entry point ─────────────────────────────────────────────────────────────


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot, device = build_device(cfg)
    stop_hardware_now = _make_hardware_stop_callback(robot, device)

    dataset_features = build_dataset_features(robot, use_videos=cfg.dataset.video)

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    image_writer_threads = cfg.dataset.num_image_writer_threads_per_camera * num_cameras

    dataset: LeRobotDataset | None = None
    listener = None
    try:
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=image_writer_threads if num_cameras > 0 else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=image_writer_threads,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        listener, events = init_keyboard_listener(stop_callback=stop_hardware_now)

        # The recorder commands Cartesian Servo Move. JAKA still returns its
        # full fixed action schema so joint feedback is recorded in the action
        # frame, but idle hold commands must contain only the EEF representation.
        action_keys = [key for key in robot.action_features if key.startswith("ee.")]
        if "gripper.pos" in robot.action_features:
            action_keys.append("gripper.pos")

        loop_kwargs = {
            "robot": robot,
            "device": device,
            "action_keys": action_keys,
            "events": events,
            "fps": cfg.dataset.fps,
            "live": None,  # bound below
            "single_task": cfg.dataset.single_task,
            "gripper_toggle": TriggerGripperToggle(),
        }

        initial_panel = Panel(
            "Waiting for the first control frame...",
            title="[bold cyan]Control frame[/bold cyan]",
            border_style="cyan",
        )
        trace_context = (
            ControlTraceWriter(
                cfg.control_trace_csv,
                flush_every=cfg.control_trace_flush_frames,
            )
            if cfg.control_trace_csv is not None
            else nullcontext(None)
        )
        with (
            Live(initial_panel, refresh_per_second=max(cfg.dataset.fps, 1), transient=False) as live,
            VideoEncodingManager(dataset),
            trace_context as control_trace,
        ):
            if control_trace is not None:
                logging.info("[JAKA-TRACE] Writing control trace to %s", control_trace.path)
            loop_kwargs["live"] = live
            loop_kwargs["control_trace"] = control_trace
            recorded_episodes = 0
            episode_total = dataset.num_episodes + cfg.dataset.num_episodes
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                logging.info(f"Recording episode {dataset.num_episodes}")
                episode_number = dataset.num_episodes + 1
                _record_loop(
                    **loop_kwargs,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    episode_number=episode_number,
                    episode_total=episode_total,
                )

                # ESC/q tears down the arm from the keyboard thread. Do not reset the
                # scene or save a partial episode after that immediate stop.
                if events["stop_recording"]:
                    break

                # Reset window: give the operator time to reposition the scene.
                # Skipped for the last episode (or if stop_recording was set).
                if not events["stop_recording"] and (
                    recorded_episodes < cfg.dataset.num_episodes - 1 or events["rerecord_episode"]
                ):
                    logging.info("Reset the environment")
                    _record_loop(
                        **loop_kwargs,
                        dataset=None,
                        control_time_s=cfg.dataset.reset_time_s,
                        episode_number=episode_number,
                        episode_total=episode_total,
                    )

                if events["rerecord_episode"]:
                    logging.info("Re-record episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        logging.info("Stop recording")

        # The keyboard callback normally performs this before the loop unwinds. The
        # same callback here covers normal completion, Ctrl-C, and startup races.
        stop_hardware_now()

        # Restore the terminal before the (potentially long) finalize/encode.
        if listener is not None:
            with suppress(Exception):
                listener.stop()

        if dataset is not None:
            dataset.finalize()

        if cfg.dataset.push_to_hub:
            if dataset is not None and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        logging.info("Exiting")

    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
