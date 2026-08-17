#!/usr/bin/env python
"""Run a remote policy server against a JAKA robot in real time.

Press Space to start, pause, or resume inference. Press R to execute the
configured joint reset pose. Press G to toggle the gripper. Press Esc or Q to
leave the program safely.

Example usage:

uv run python -m examples.isaac_teleop_to_jaka.infer \
  --robot.type=jaka_robot \
  --robot.id=jaka_arm \
  --robot.ip=192.168.1.31 \
  --robot.cameras="{hand: {type: intelrealsense, serial_number_or_name: '342522070741', width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: '347522072196', width: 640, height: 480, fps: 30}}" \
  --robot.reset_joints="[-0.956, 1.903, 1.427, 1.368, -1.590, -0.290]" \
  --proxy.addr=tcp://127.0.0.1:9000 \
  --task="pick up the tube from the pad" \
  --control_mode=eef \
  --fps=30

"""

import logging
import math
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pprint import pformat
from threading import Lock
from typing import Any, Literal

import numpy as np
import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lekit.robots.jaka_robot import JakaCameraTimeoutError, JakaRobot, JakaRobotConfig
from lekit.robots.jaka_robot.dataset_features import build_dataset_features
from lekit.tools.proxy import Proxy, ProxyConfig as BaseProxyConfig
from lekit.utils.rerun_utils import RerunLogger


@dataclass
class ProxyConfig(BaseProxyConfig):
    """Inference proxy configuration with its explicit ZMQ endpoint."""

    addr: str = field(default="tcp://127.0.0.1:9000", metadata={"help": "Policy server ZMQ address"})

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.addr.startswith("tcp://"):
            raise ValueError("proxy addr must use the tcp://host:port form")


@dataclass
class InferConfig:
    robot: JakaRobotConfig
    proxy: ProxyConfig
    task: str
    control_mode: Literal["eef", "joint"] = field(
        default="eef", metadata={"help": "Arm control representation: eef or joint"}
    )
    fps: int = 30
    rerun_url: str | None = "rerun+http://127.0.0.1:9876/proxy"

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if self.fps < 1:
            raise ValueError("fps must be at least 1")


class LoopRateMonitor:
    """Compute a rolling loop rate without adding another dependency."""

    def __init__(self, window_s: float = 1.0):
        self._window_s = window_s
        self._timestamps: deque[float] = deque()

    def update(self, timestamp: float) -> float | None:
        self._timestamps.append(timestamp)
        cutoff = timestamp - self._window_s
        while len(self._timestamps) > 2 and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) < 2:
            return None
        return (len(self._timestamps) - 1) / (self._timestamps[-1] - self._timestamps[0])


class PolicyGripperGate:
    """Turn noisy continuous policy output into sparse open/close commands."""

    def __init__(
        self,
        *,
        open_threshold: float = 0.65,
        close_threshold: float = 0.35,
        retry_interval_s: float = 1.0,
    ):
        if not 0.0 <= close_threshold < open_threshold <= 1.0:
            raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 1")
        if retry_interval_s <= 0.0:
            raise ValueError("gripper retry interval must be positive")
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.retry_interval_s = retry_interval_s
        self._position: float | None = None
        self._pending: float | None = None
        self._last_attempt_at = -math.inf

    def next_command(self, value: float, *, observed_position: float, now: float) -> float | None:
        if value >= self.open_threshold:
            target = 1.0
        elif value <= self.close_threshold:
            target = 0.0
        else:
            return None

        if self._position is None:
            self._position = float(np.clip(observed_position, 0.0, 1.0))
        if target == self._position:
            self._pending = None
            return None
        if self._pending == target and now - self._last_attempt_at < self.retry_interval_s:
            return None
        self._pending = target
        self._last_attempt_at = now
        return target

    def mark_applied(self, position: float) -> None:
        self._position = position
        self._pending = None

    def mark_failed(self, *, now: float) -> None:
        self._last_attempt_at = now

    def toggle_manual(self, *, observed_position: float) -> float:
        """Toggle the gripper once; subsequent policy outputs remain authoritative."""

        if self._position is None:
            self._position = float(np.clip(observed_position, 0.0, 1.0))
        self._position = 1.0 - self._position
        self._pending = self._position
        self._last_attempt_at = -math.inf
        return self._position


def init_keyboard_listener(stop_callback: Callable[[], None] | None = None):
    """Return keyboard events for start/pause, reset, and safe exit."""

    events = {"running": False, "reset": False, "toggle_gripper": False, "stop": False}

    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("space", " "):
            events["running"] = not events["running"]
            logging.info("Policy inference %s", "resumed" if events["running"] else "paused")
        elif key == "r":
            events["reset"] = True
        elif key == "g":
            events["toggle_gripper"] = True
        elif key in ("esc", "q"):
            events["stop"] = True
            if stop_callback is not None:
                stop_callback()

    if sys.stdin is not None and sys.stdin.isatty():
        from lerobot.utils.keyboard_input import TerminalKeyListener

        listener = TerminalKeyListener(on_key)
        listener.start()
    else:
        from lerobot.utils.keyboard_input import create_key_listener

        listener = create_key_listener(on_key, controls_help="Space=start/pause, g=gripper, r=reset, Esc/q=quit")
    logging.info("Keyboard controls: Space = start/pause, G = gripper, R = reset, Esc/Q = quit")
    return listener, events


def _make_stop_callback(robot: JakaRobot) -> Callable[[], None]:
    stopped = False
    lock = Lock()

    def stop() -> None:
        nonlocal stopped
        with lock:
            if stopped:
                return
            stopped = True
            with suppress(Exception):
                if robot.is_connected:
                    robot.disconnect()

    return stop


def _noop() -> None:
    pass


def _configure_robot(config: JakaRobotConfig, *, control_mode: Literal["eef", "joint"]) -> JakaRobot:
    """Configure JAKA Servo for the selected policy control representation."""

    config.auto_enable_servo = False
    config.separate_feedback_connection = False
    if control_mode == "eef":
        config.servo_process = True
        if config.servo_filter_mode == "none":
            config.servo_filter_mode = "cartesian_nlf"
        if config.servo_filter_mode != "cartesian_nlf":
            raise ValueError("JAKA EEF online inference requires cartesian_nlf servo filtering")
    else:
        # The managed Servo process currently supports only Cartesian EEF targets.
        config.servo_process = False
        # Joint Servo already applies velocity/acceleration limiting in its scheduler.
        # The generic absolute-goal clamp would otherwise rewrite policy targets.
        config.max_relative_target = None
        if config.servo_filter_mode == "none":
            config.servo_filter_mode = "joint_nlf"
        if config.servo_filter_mode not in {"joint_lpf", "joint_nlf"}:
            raise ValueError("JAKA joint online inference requires joint_lpf or joint_nlf servo filtering")

    robot = make_robot_from_config(config)
    if not isinstance(robot, JakaRobot):
        raise ValueError(f"Expected --robot.type=jaka_robot, got {type(robot).__name__}")
    if robot.config.user_frame_id != 0:
        raise ValueError("JAKA online inference requires --robot.user_frame_id=0 (base frame).")
    robot.connect()
    return robot


_POLICY_ACTION_NAMES = (*JakaRobot._JOINT_KEYS, "gripper.pos", *JakaRobot._EEF_KEYS)


def _policy_action_names() -> list[str]:
    """Return the fixed 13-field schema produced by the policy server."""

    return list(_POLICY_ACTION_NAMES)


def _to_robot_action(
    values: torch.Tensor, names: list[str], *, control_mode: Literal["eef", "joint"]
) -> tuple[dict[str, float], np.ndarray]:
    """Map the policy's 13-dimensional output to the selected arm representation.

    The policy vector contains joints, gripper, and EEF targets. The selected
    arm representation and gripper are the only fields sent to the robot.
    """

    vector = np.asarray(values.detach().cpu(), dtype=float).reshape(-1)
    if vector.size != len(names):
        raise ValueError(
            f"Policy returned {vector.size} values, but its action schema has {len(names)} fields"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Policy returned a non-finite action")
    if tuple(names) != _POLICY_ACTION_NAMES:
        raise ValueError(f"Policy action schema must be {_POLICY_ACTION_NAMES!r}")
    raw = dict(zip(_POLICY_ACTION_NAMES, map(float, vector), strict=True))
    arm_keys = JakaRobot._EEF_KEYS if control_mode == "eef" else JakaRobot._JOINT_KEYS
    action = {key: raw[key] for key in arm_keys}
    action["gripper.pos"] = raw["gripper.pos"]
    return action, vector


def _reset_robot(robot: JakaRobot) -> None:
    reset_joints = robot.config.reset_joints
    if reset_joints is None:
        raise ValueError("Reset requires --robot.reset_joints with six angles in radians.")
    if robot.is_in_servo():
        robot.servo_enable(False)
    action = {f"joint_{index}.pos": value for index, value in enumerate(reset_joints, start=1)}
    previous_limit = robot.config.max_relative_target
    try:
        robot.config.max_relative_target = None
        robot.send_action(action, use_servo=False)
    finally:
        robot.config.max_relative_target = previous_limit


def _reset_policy(proxy: Proxy, address: str) -> None:
    """Discard queued action chunks after the robot has moved to its reset pose."""

    proxy.switch_policy(address)


def _fill_missing_camera_frames(
    robot: JakaRobot, observation: dict[str, Any], latest_frames: dict[str, Any]
) -> bool:
    expected = [name for name, camera in robot.cameras.items() if getattr(camera, "use_rgb", True)]
    for name in expected:
        if name in observation:
            latest_frames[name] = observation[name]
        elif name in latest_frames:
            observation[name] = latest_frames[name]
    return all(name in observation for name in expected)


def _log_rerun(
    logger: RerunLogger,
    robot: JakaRobot,
    observation: dict[str, Any],
    action: dict[str, float] | None,
    policy_vector: np.ndarray | None,
    *,
    task: str,
    state: str,
    metrics: dict[str, float],
) -> None:
    frame = {
        f"observation.{key}": value
        for key, value in observation.items()
        if np.isscalar(value) and not isinstance(value, (str, bytes))
    }
    frame["observation.state"] = np.asarray(
        [observation[name] for name in robot.action_features], dtype=np.float32
    )
    if action is not None:
        frame.update({f"action.{key}": value for key, value in action.items()})
    frame.update({f"metrics.{key}": value for key, value in metrics.items() if math.isfinite(value)})
    frame.update(task=task, record_state=state)
    if policy_vector is not None:
        frame["policy"] = policy_vector
    for name, camera in robot.cameras.items():
        if getattr(camera, "use_rgb", True) and name in observation:
            frame[f"observation.images.{name}"] = observation[name]
    logger.log(frame)


def infer_loop(
    robot: JakaRobot,
    proxy: Proxy,
    features: dict[str, dict],
    events: dict[str, bool],
    *,
    task: str,
    control_mode: Literal["eef", "joint"],
    fps: int,
    rerun_logger: RerunLogger | None,
    policy_address: str,
) -> None:
    """Continuously observe, infer, and execute while inference is running."""

    interval_s = 1.0 / fps
    rate_monitor = LoopRateMonitor()
    action_names = _policy_action_names()
    if tuple(action_names) != _POLICY_ACTION_NAMES:
        raise ValueError(f"Policy action schema must be {_POLICY_ACTION_NAMES!r}")
    latest_frames: dict[str, Any] = {}
    latest_action: dict[str, float] | None = None
    latest_vector: np.ndarray | None = None
    gripper_gate = PolicyGripperGate()

    while not events["stop"]:
        started_at = time.perf_counter()
        loop_rate_hz = rate_monitor.update(started_at)
        observation_started_at = time.perf_counter()
        try:
            observation = robot.get_observation()
        except JakaCameraTimeoutError as exc:
            observation = exc.observation
        observation_ms = (time.perf_counter() - observation_started_at) * 1000.0
        cameras_ready = _fill_missing_camera_frames(robot, observation, latest_frames)

        if events["reset"]:
            events["reset"] = False
            events["running"] = False
            _reset_robot(robot)
            _reset_policy(proxy, policy_address)
            latest_action = None
            latest_vector = None

        if events.pop("toggle_gripper", False):
            gripper_command = gripper_gate.toggle_manual(
                observed_position=float(observation.get("gripper.pos", 0.0))
            )
            try:
                robot.send_action({"gripper.pos": gripper_command})
            except Exception as exc:
                gripper_gate.mark_failed(now=time.monotonic())
                logging.warning("JAKA manual gripper command rejected: %s", exc)
            else:
                gripper_gate.mark_applied(gripper_command)

        policy_ms = 0.0
        action_ms = 0.0
        state = "recording" if events["running"] else "pause"
        if events["running"] and cameras_ready:
            if not robot.is_in_servo():
                robot.servo_enable(True, representation=control_mode)
            policy_observation = build_dataset_frame(features, observation, prefix=OBS_STR)
            policy_observation["task"] = task
            policy_started_at = time.perf_counter()
            values = proxy.require_action(policy_observation, timeout_s=interval_s)
            policy_ms = (time.perf_counter() - policy_started_at) * 1000.0
            if values is not None:
                latest_action, latest_vector = _to_robot_action(
                    values, action_names, control_mode=control_mode
                )
                action_started_at = time.perf_counter()
                arm_action = dict(latest_action)
                gripper_value = arm_action.pop("gripper.pos", None)
                robot.send_action(arm_action)
                if gripper_value is not None:
                    gripper_command = gripper_gate.next_command(
                        gripper_value,
                        observed_position=float(observation.get("gripper.pos", 0.0)),
                        now=time.monotonic(),
                    )
                    if gripper_command is not None:
                        try:
                            robot.send_action({"gripper.pos": gripper_command})
                        except Exception as exc:
                            gripper_gate.mark_failed(now=time.monotonic())
                            logging.warning("JAKA gripper command rejected: %s", exc)
                        else:
                            gripper_gate.mark_applied(gripper_command)
                action_ms = (time.perf_counter() - action_started_at) * 1000.0
        elif robot.is_in_servo():
            robot.servo_enable(False)

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        metrics = {
            "loop_rate_hz": loop_rate_hz or 0.0,
            "observation_ms": observation_ms,
            "policy_ms": policy_ms,
            "action_ms": action_ms,
            "loop_ms": elapsed_ms,
        }
        if rerun_logger is not None:
            _log_rerun(
                rerun_logger,
                robot,
                observation,
                latest_action,
                latest_vector,
                task=task,
                state=state,
                metrics=metrics,
            )
        precise_sleep(max(interval_s - (time.perf_counter() - started_at), 0.0))


@parser.wrap()
def infer(cfg: InferConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    robot: JakaRobot | None = None
    proxy: Proxy | None = None
    listener = None
    rerun_logger: RerunLogger | None = None
    stop_hardware: Callable[[], None] = _noop
    try:
        proxy = Proxy(cfg.proxy)
        proxy.switch_policy(cfg.proxy.addr)
        robot = _configure_robot(cfg.robot, control_mode=cfg.control_mode)
        stop_hardware = _make_stop_callback(robot)
        listener, events = init_keyboard_listener(stop_hardware)
        features = build_dataset_features(robot, use_videos=True)
        rerun_logger = RerunLogger(url=cfg.rerun_url) if cfg.rerun_url else None
        infer_loop(
            robot,
            proxy,
            features,
            events,
            task=cfg.task,
            control_mode=cfg.control_mode,
            fps=cfg.fps,
            rerun_logger=rerun_logger,
            policy_address=cfg.proxy.addr,
        )
    finally:
        if listener is not None:
            with suppress(Exception):
                listener.stop()
        if rerun_logger is not None:
            with suppress(Exception):
                rerun_logger.flush(timeout=1.0)
            with suppress(Exception):
                rerun_logger.stop()
        if proxy is not None:
            with suppress(Exception):
                proxy.stop()
        stop_hardware()


def main() -> None:
    infer()


if __name__ == "__main__":
    main()
